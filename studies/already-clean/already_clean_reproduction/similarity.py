"""Exploratory semantic continuity, lexical controls and clustered uncertainty."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .common import WALK, cosine, read_js, read_json, write_csv, write_json
from .reproduce import parent_records, parent_bootstrap


def normalize_rows(array):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def pole_weights(pole, top):
    weights = np.array([a for _, a in top], dtype=float) * (1 if pole.endswith("+") else -1)
    assert np.isfinite(weights).all() and (weights >= 0).all(), pole
    return weights if weights.sum() > 0 else np.ones(len(top))


def bootstrap_mean(values, units, config):
    grouped = defaultdict(list)
    for value, unit in zip(values, units):
        grouped[unit].append(value)
    # The estimand is an equally weighted mean over underlying neurons.
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    if not len(means):
        return {"n_neurons": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(config["seed"])
    draws = means[rng.integers(len(means), size=(config["bootstrap_replicates"], len(means)))].mean(1)
    return {"n_neurons": len(means), "n_poles": len(values), "mean": float(means.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist()}


def similarity_test(centroids, neurons, config, group="random", donor_pool="all", sign=False, checkpoint=13):
    # Donors are from the same layer and cannot be the opposite pole of the same neuron.
    n = len(neurons)
    matrix = centroids[:, checkpoint] @ centroids[:, -1].T
    selected = [i for i, row in enumerate(neurons) if group == "all" or row["group"] == group]
    rows = []
    for i in selected:
        pole = neurons[i]["pole"]
        donors = [j for j in range(n) if neurons[j]["pole"][:3] == pole[:3]
                  and neurons[j]["pole"][:-1] != pole[:-1]
                  and (not sign or neurons[j]["pole"][-1] == pole[-1])
                  and (donor_pool == "all" or neurons[j]["group"] == donor_pool)]
        if not donors:
            continue
        matched, mismatch = float(matrix[i, i]), float(matrix[i, donors].mean())
        rows.append({"pole": pole, "neuron": pole[:-1], "n_donors": len(donors),
                     "matched": matched, "mismatched": mismatch, "delta": matched - mismatch})
    summary = bootstrap_mean([r["delta"] for r in rows], [r["neuron"] for r in rows], config)
    summary.update({"group": group, "donor_pool": donor_pool, "match_sign": sign,
                    "checkpoint": checkpoint, "excluded_poles": [neurons[i]["pole"] for i in selected
                                                                 if neurons[i]["pole"] not in {r["pole"] for r in rows}],
                    "mean_matched": float(np.mean([r["matched"] for r in rows])) if rows else None,
                    "mean_mismatched": float(np.mean([r["mismatched"] for r in rows])) if rows else None})
    return summary, rows


def identity_permutation(centroids, neurons, config, sign=True, checkpoint=13):
    # Restrict this null to the random subset so selection of late poles cannot
    # create an apparent identity effect. Discard singleton strata explicitly.
    strata = defaultdict(list)
    for i, neuron in enumerate(neurons):
        if neuron["group"] == "random":
            pole = neuron["pole"]
            strata[(pole[:3], pole[-1] if sign else "both")].append(i)
    strata = [ids for ids in strata.values() if len(ids) > 1]
    ids = [i for group in strata for i in group]
    if not ids:
        return {"n_poles": 0, "p_one_sided": None}
    matrix = centroids[:, checkpoint] @ centroids[:, -1].T
    observed = float(np.mean([matrix[i, i] for i in ids]))
    rng = np.random.default_rng(config["seed"])
    null = np.zeros(config["permutations"])
    # All possible assignments are enumerated when the finite null is small.
    import itertools, math
    n_assignments = math.prod(math.factorial(len(group)) for group in strata)
    if n_assignments <= 100000:
        null = np.array([np.mean([matrix[i, j] for group, perm in zip(strata, assignment)
                                 for i, j in zip(group, perm)])
                         for assignment in itertools.product(*(itertools.permutations(g) for g in strata))])
        p = float(np.mean(null >= observed - 1e-12))
        method = "exact conditional enumeration"
    else:
        for iteration in range(len(null)):
            null[iteration] = np.mean([matrix[i, j] for group in strata for i, j in zip(group, rng.permutation(group))])
        p = float((1 + (null >= observed).sum()) / (len(null) + 1))
        method = "Monte Carlo with plus-one correction"
    return {"n_poles": len(ids), "strata_sizes": [len(g) for g in strata], "matched_cosine": observed,
            "null_mean": float(null.mean()), "null95": np.quantile(null, [.025, .975]).tolist(),
            "p_one_sided": p, "null_draws": len(null), "method": method, "match_sign": sign,
            "scope": "Conditional identity test within the published random subset; no new model seeds."}


def lexical_arrays(data):
    neurons = data["neurons"]
    text_docs, token_docs, keys = [], [], []
    context_sets = []
    for neuron in neurons:
        sets = []
        for t, step in enumerate(neuron["steps"]):
            ctx = [neuron["ctx"][i] for i, _ in step["top"]]
            text_docs.append("\n".join("".join(c) for c in ctx))
            token_docs.append(Counter(c[1].strip().lower() for c in ctx))
            keys.append((neuron["pole"], t))
            sets.append({i for i, _ in step["top"]})
        context_sets.append(sets)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(text_docs)
    # Compute within-pole matrices sparsely, avoiding a large dense vocabulary tensor.
    tfidf_matrices = np.stack([(tfidf[i * 16:(i + 1) * 16] @ tfidf[i * 16:(i + 1) * 16].T).toarray()
                               for i in range(len(neurons))])
    vocabulary = sorted(set(k for doc in token_docs for k in doc))
    lookup = {token: i for i, token in enumerate(vocabulary)}
    token_array = np.zeros((len(token_docs), len(vocabulary)))
    for i, counts in enumerate(token_docs):
        for token, count in counts.items():
            token_array[i, lookup[token]] = count
    token_array = normalize_rows(token_array).reshape(len(neurons), 16, -1)
    token_matrices = token_array @ token_array.transpose(0, 2, 1)
    overlap = np.array([[[len(a & b) / len(a | b) for b in sets] for a in sets] for sets in context_sets])
    return tfidf_matrices, token_matrices, overlap, {"tfidf_features": len(vectorizer.vocabulary_),
                                                   "token_types": len(vocabulary)}


def encode(data, model_name, revision, variant, out, config):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    # Do not execute remote model code; pin both model weights and tokenizer revision.
    model = SentenceTransformer(model_name, revision=revision, device=config["device"], trust_remote_code=False)
    texts, mapping = [], {}
    indices = []
    for neuron in data["neurons"]:
        ids = []
        for before, token, after in neuron["ctx"]:
            if variant == "prefix":
                text = before + token
            elif variant == "masked":
                text = before + " [MASK] " + after
            elif variant == "deleted":
                text = before + " " + after
            else:
                text = before + token + after
            if text not in mapping:
                mapping[text] = len(texts)
                texts.append(text)
            ids.append(mapping[text])
        indices.append(ids)
    print("EMBED_START", json.dumps({"model": model_name, "revision": revision, "variant": variant,
                                      "unique_texts": len(texts), "max_seq_length": model.max_seq_length}), flush=True)
    embeddings = []
    for start in range(0, len(texts), 512):
        embeddings.append(model.encode(texts[start:start + 512], batch_size=64, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False))
        print("EMBED_PROGRESS", variant, start + len(embeddings[-1]), "/", len(texts), flush=True)
    embeddings = np.vstack(embeddings)
    assert np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-5)
    centroids = []
    weighted = []
    for neuron, ids in zip(data["neurons"], indices):
        per_step, weight_step = [], []
        for snapshot in neuron["steps"]:
            selected = np.array([ids[i] for i, _ in snapshot["top"]])
            per_step.append(embeddings[selected].mean(0))
            weights = pole_weights(neuron["pole"], snapshot["top"])
            weight_step.append(np.average(embeddings[selected], axis=0, weights=weights))
        centroids.append(per_step)
        weighted.append(weight_step)
    centroids = normalize_rows(np.asarray(centroids))
    weighted = normalize_rows(np.asarray(weighted))
    key = ("primary" if model_name == config["embedding_model"] else "secondary") + "_" + variant
    # Save vectors and their exact text index, enabling reanalysis without downloading a model.
    np.savez_compressed(out / f"embeddings-{key}.npz", embeddings=embeddings, centroids=centroids, weighted=weighted)
    write_json(out / f"embedding-index-{key}.json", {"texts": texts, "per_pole_context_indices": indices,
                                                   "poles": [n["pole"] for n in data["neurons"]]})
    metadata = {"model": model_name, "revision": revision, "variant": variant,
                "unique_texts": len(texts), "dimensions": embeddings.shape[1],
                "max_seq_length": model.max_seq_length,
                "weights": "uniform top-20 contexts, normalize each embedding then normalize the mean",
                "npz_sha256": hashlib.sha256((out / f"embeddings-{key}.npz").read_bytes()).hexdigest()}
    del model
    return centroids, weighted, metadata


def fixed_activation_geometry():
    examples = read_json("studies/next/10_recount2/out/06_examples.json")
    results = {}
    for name, row in examples.items():
        array = np.array(row["a_plus_by_t"])
        matrix = [[cosine(a, b) for b in array] for a in array]
        results[name] = {"pole": row["pole"], "n_fixed_contexts": array.shape[1],
                         "cosine": matrix, "norms": np.linalg.norm(array, axis=1).tolist(),
                         "undefined_zero_checkpoints": np.where(np.linalg.norm(array, axis=1) == 0)[0].tolist(),
                         "scope": "Own-direction clipped activations on end-selected fixed contexts; descriptive, not a representative input sample."}
    return results


def run(out, config):
    data = read_js(WALK + "neuron_across_training/data.js")
    neurons = data["neurons"]
    tfidf, token, jaccard, lexical_meta = lexical_arrays(data)
    main, weighted, metadata = encode(data, config["embedding_model"], config["embedding_revision"], "full", out, config)
    matrices = main @ main.transpose(0, 2, 1)
    np.savez_compressed(out / "similarity-matrices.npz", semantic=matrices, tfidf=tfidf, token=token, exact_jaccard=jaccard)
    tests = {}
    main_test, rows = similarity_test(main, neurons, config)
    tests["primary_random_same_layer"] = main_test
    write_csv(out / "matched-controls.csv", rows)
    for name, kwargs in [("random_donors_only", {"donor_pool": "random"}),
                         ("same_sign_donors", {"sign": True}),
                         ("all_poles", {"group": "all"}),
                         ("activation_weighted", {})]:
        tests[name] = similarity_test(weighted if name == "activation_weighted" else main, neurons, config, **kwargs)[0]
    tests["permutation_random_layer"] = identity_permutation(main, neurons, config, sign=False)
    tests["permutation_random_layer_sign"] = identity_permutation(main, neurons, config, sign=True)
    print("SIMILARITY_PRIMARY", json.dumps(tests), flush=True)

    records, curves = [], []
    for i, neuron in enumerate(neurons):
        for t, snapshot in enumerate(data["steps"]):
            records.append({"pole": neuron["pole"], "neuron": neuron["pole"][:-1], "group": neuron["group"],
                            "checkpoint": t, "step": snapshot["step"], "semantic_to_final": float(matrices[i, t, -1]),
                            "tfidf_to_final": float(tfidf[i, t, -1]), "token_cosine_to_final": float(token[i, t, -1]),
                            "context_jaccard_to_final": float(jaccard[i, t, -1]),
                            "semantic_previous": float(matrices[i, t - 1, t]) if t else None,
                            "max_activation": neuron["steps"][t]["max"]})
    for group in ("random", "late", "all"):
        for t, snapshot in enumerate(data["steps"]):
            items = [r for r in records if r["checkpoint"] == t and (group == "all" or r["group"] == group)]
            for metric in ("semantic_to_final", "tfidf_to_final", "token_cosine_to_final", "context_jaccard_to_final"):
                summary = bootstrap_mean([r[metric] for r in items], [r["neuron"] for r in items], config)
                curves.append({"group": group, "step": snapshot["step"], "checkpoint": t, "metric": metric,
                               "mean": summary["mean"], "lower": summary["ci95"][0], "upper": summary["ci95"][1],
                               "n_neurons": summary["n_neurons"]})
    write_csv(out / "checkpoint-similarities.csv", records)
    write_json(out / "curves.json", curves)
    random = [i for i, n in enumerate(neurons) if n["group"] == "random"]
    deltas = matrices[random, 13, 15] - matrices[random, 0, 15]
    temporal = bootstrap_mean(deltas, [neurons[i]["pole"][:-1] for i in random], config)
    late_summary = {metric: next(c for c in curves if c["group"] == "random" and c["checkpoint"] == 13 and c["metric"] == metric)
                    for metric in ("semantic_to_final", "tfidf_to_final", "token_cosine_to_final", "context_jaccard_to_final")}
    masks = []
    for variant in ("prefix", "masked"):
        c, _, meta = encode(data, config["embedding_model"], config["embedding_revision"], variant, out, config)
        test = similarity_test(c, neurons, config)[0]
        masks.append(meta)
        tests[variant] = test
        print("SIMILARITY_SENSITIVITY", variant, json.dumps(test), flush=True)
    second, _, second_meta = encode(data, config["second_embedding_model"], config["second_embedding_revision"], "full", out, config)
    tests["second_encoder"] = similarity_test(second, neurons, config)[0]
    _, parent_rows = parent_records()
    parent_uncertainty = parent_bootstrap(parent_rows, cluster="neuron", seed=config["seed"], replicates=config["bootstrap_replicates"])
    fixed = fixed_activation_geometry()
    write_json(out / "fixed-activation-geometry.json", fixed)
    write_json(out / "embedding-metadata.json", [metadata, *masks, second_meta])
    result = {"n_poles": len(neurons), "n_unique_neurons": len({n["pole"][:-1] for n in neurons}),
              "n_random_poles": len(random), "n_checkpoints": 16,
              "embedding": metadata, "lexical": lexical_meta, "tests": tests,
              "random_late_summary": late_summary, "random_late_minus_initial_cosine": temporal,
              "parent_neuron_bootstrap": parent_uncertainty,
              "fixed_example_zero_checkpoints": {k: v["undefined_zero_checkpoints"] for k, v in fixed.items()},
              "limits": ["41 selected poles, including 17 selected using final outcomes; random group excludes these poles.",
                         "One OLMo training run. Intervals resample observed neurons, not model training seeds.",
                         "Top-20, truncated contexts change across checkpoints; centroids describe selected text, not neural activation vectors.",
                         "Exact overlap uses within-pole exported context IDs; missing top-k activations are not zero-filled.",
                         "Primary same-layer donor pool includes selected late poles; random-donor and sign-matched sensitivities are reported.",
                         "All new hypotheses are exploratory. Encoder sensitivity tests are not independent replications."]}
    print("SIMILARITY_SUMMARY", json.dumps(result), flush=True)
    return result
