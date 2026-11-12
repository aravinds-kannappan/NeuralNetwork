"""Test semantic continuity after removing repeated text and marked tokens."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, WALK, cosine, read_js, write_csv, write_json
from .similarity import bootstrap_mean, encode, similarity_test


def layer_bootstrap(values, layers, config):
    grouped = defaultdict(list)
    for value, layer in zip(values, layers):
        grouped[layer].append(value)
    array = np.array([[sum(v), len(v)] for _, v in sorted(grouped.items())])
    if not len(array):
        return {"mean": None, "ci95": None, "n_layers": 0}
    rng = np.random.default_rng(config["seed"])
    sampled = array[rng.integers(len(array), size=(config["bootstrap_replicates"], len(array)))].sum(1)
    draws = sampled[:, 0] / sampled[:, 1]
    return {"mean": float(np.mean(values)), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "n_layers": len(array), "n_poles": len(values),
            "estimand": "Mean query-neuron contrast; resample complete layer blocks including all query contrasts."}


def retained_indices(neuron, mode):
    source = [i for i, _ in neuron["steps"][13]["top"]]
    target = [i for i, _ in neuron["steps"][15]["top"]]
    keys = [c[1].strip().lower() if mode == "token" else "".join(c) for c in neuron["ctx"]]
    common = {keys[i] for i in source} & {keys[i] for i in target}
    return [i for i in source if keys[i] not in common], [i for i in target if keys[i] not in common], keys


def compare(embeddings, index, neurons, mode, config):
    rng = np.random.default_rng(config["seed"])
    indices = index["per_pole_context_indices"]
    assert index["poles"] == [n["pole"] for n in neurons]
    rows, exclusions = [], []
    for i, neuron in enumerate(neurons):
        if neuron["group"] != "random":
            continue
        q, target, keys = retained_indices(neuron, mode)
        if min(len(q), len(target)) < 3:
            exclusions.append({"pole": neuron["pole"], "reason": "fewer than 3 retained contexts", "n_query": len(q), "n_target": len(target)})
            continue
        a = embeddings[[indices[i][j] for j in q]]
        b = embeddings[[indices[i][j] for j in target]]
        qa = a.mean(0)
        actual_centroid = cosine(qa, b.mean(0))
        actual_pairwise = float(qa @ b.mean(0))
        donors, control_centroids, control_pairwise = [], [], []
        query_keys = {keys[j] for j in q}
        for j, other in enumerate(neurons):
            if other["pole"][:3] != neuron["pole"][:3] or other["pole"][:-1] == neuron["pole"][:-1]:
                continue
            selected = [k for k, _ in other["steps"][15]["top"]]
            def key(k):
                return other["ctx"][k][1].strip().lower() if mode == "token" else "".join(other["ctx"][k])
            selected = [k for k in selected if key(k) not in query_keys]
            if len(selected) < len(target):
                continue
            candidates = embeddings[[indices[j][k] for k in selected]]
            # Each negative-control target contains the same number of contexts
            # as this query's real target; uniform subsamples, no replacement.
            chosen = np.array([rng.choice(len(candidates), size=len(target), replace=False)
                               for _ in range(config["donor_subsamples"])])
            centers = candidates[chosen].mean(1)
            norms = np.linalg.norm(centers, axis=1) * np.linalg.norm(qa)
            assert (norms > 0).all()
            control_centroids.append(float(np.mean((centers @ qa) / norms)))
            control_pairwise.append(float(np.mean(centers @ qa)))
            donors.append(other["pole"])
        if not donors:
            exclusions.append({"pole": neuron["pole"], "reason": "no same-layer donor with enough retained contexts",
                               "n_query": len(q), "n_target": len(target)})
            continue
        control_centroid = float(np.mean(control_centroids))
        control_pair = float(np.mean(control_pairwise))
        rows.append({"pole": neuron["pole"], "neuron": neuron["pole"][:-1], "layer": neuron["pole"][:3],
                     "n_query": len(q), "n_target": len(target), "n_donors": len(donors),
                     "donors": ";".join(donors), "centroid_same": actual_centroid,
                     "centroid_control": control_centroid, "centroid_delta": actual_centroid - control_centroid,
                     "pairwise_same": actual_pairwise, "pairwise_control": control_pair,
                     "pairwise_delta": actual_pairwise - control_pair})
    summaries = {}
    for metric in ("centroid", "pairwise"):
        values = [r[metric + "_delta"] for r in rows]
        summaries[metric + "_neuron"] = bootstrap_mean(values, [r["neuron"] for r in rows], config)
        summaries[metric + "_layer"] = layer_bootstrap(values, [r["layer"] for r in rows], config)
        summaries[metric + "_same_mean"] = float(np.mean([r[metric + "_same"] for r in rows])) if rows else None
        summaries[metric + "_control_mean"] = float(np.mean([r[metric + "_control"] for r in rows])) if rows else None
    summaries.update({"removal": mode, "exclusions": exclusions,
                      "retained_query_range": [min(r["n_query"] for r in rows), max(r["n_query"] for r in rows)] if rows else None,
                      "retained_target_range": [min(r["n_target"] for r in rows), max(r["n_target"] for r in rows)] if rows else None})
    return summaries, rows


def run(out, config):
    source = Path(config["artifact_root"]) / "runs" / config["similarity_source_run"]
    manifest = json.loads((ROOT / "embedding_manifest.json").read_text())
    assert manifest["run_id"] == config["similarity_source_run"]
    for record in manifest["files"]:
        data = (source / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
        assert len(data) == record["bytes"]
    print("CACHED_INPUTS_VERIFIED", len(manifest["files"]), config["similarity_source_run"], flush=True)
    data = read_js(WALK + "neuron_across_training/data.js")
    neurons = data["neurons"]
    results = {"source_run": config["similarity_source_run"], "donor_subsamples": config["donor_subsamples"],
               "verified_cache_files": len(manifest["files"]), "comparisons": {}}
    for key in ("primary_full", "primary_prefix", "primary_masked", "secondary_full"):
        vectors = np.load(source / f"embeddings-{key}.npz", allow_pickle=False)["embeddings"]
        index = json.loads((source / f"embedding-index-{key}.json").read_text())
        modes = ("text", "token") if key == "primary_full" else ("text",)
        for mode in modes:
            label = key + "_disjoint_" + mode
            summary, rows = compare(vectors, index, neurons, mode, config)
            write_csv(out / f"{label}.csv", rows)
            results["comparisons"][label] = summary
            print("FOLLOWUP", label, json.dumps(summary), flush=True)
    original = list(csv.DictReader((source / "matched-controls.csv").open()))
    results["original_layer_bootstrap"] = layer_bootstrap([float(r["delta"]) for r in original], [r["pole"][:3] for r in original], config)
    deleted, _, deleted_meta = encode(data, config["embedding_model"], config["embedding_revision"], "deleted", out, config)
    results["deleted_original_contrast"] = similarity_test(deleted, neurons, config)[0]
    vectors = np.load(out / "embeddings-primary_deleted.npz", allow_pickle=False)["embeddings"]
    index = json.loads((out / "embedding-index-primary_deleted.json").read_text())
    summary, rows = compare(vectors, index, neurons, "text", config)
    results["comparisons"]["primary_deleted_disjoint_text"] = summary
    results["deleted_embedding"] = deleted_meta
    write_csv(out / "primary_deleted_disjoint_text.csv", rows)
    write_json(out / "embedding-metadata.json", [deleted_meta])
    results["limits"] = ["Data-dependent exploratory follow-up, same sample and same training run.",
                          "Same-layer donor pool includes outcome-selected poles.",
                          "Layer bootstrap handles shared donors within layer but cannot correct viewer selection.",
                          "Token-disjoint comparison changes the eligible sample; do not interpret differences between conditions as causal token effects."]
    print("FOLLOWUP_SUMMARY", json.dumps(results), flush=True)
    return results
