"""Recompute published summaries from public observations, without model reruns."""
from __future__ import annotations

import csv
import json
import runpy
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

from .common import INPUTS, WALK, birth, read_js, read_json, wilson, write_csv


def classify(row, theta):
    if row["f"] == 1:
        return "RARE"
    if "s" not in row:
        return "UNK"
    strong = [sum(x >= theta for x in row["s"][i:i + 5]) for i in (0, 5, 10)]
    passes = strong[0] >= 3 and strong[1] <= 2 and strong[2] <= 1
    if row["l"] == "ONE" and len(row["j"]) == 1 and passes:
        return "ONE"
    if row["l"] == "TWO-REL" and 2 <= len(row["j"]) <= 3 and set(row["j"]) == {"1"} and passes:
        return "TWO-REL"
    if row["l"] == "UNREL":
        supported = sum(related == "0" and sum(x >= theta for x in row.get("u", {}).get(str(i), [])) >= 3
                        for i, related in enumerate(row["j"]))
        if supported >= 2:
            return "UNREL"
    return "UNK"


def q1():
    data = read_js(WALK + "one_job/data.js")
    reference = runpy.run_path(str(INPUTS / WALK / "one_job/extract.py"))
    results = {}
    for model in data["models"]:
        rows = model["neurons"]
        assert len(rows) == 600 and len({r["p"] for r in rows}) == 600
        counts, classes = [], {}
        for theta in data["stops"]:
            labels = [classify(row, theta) for row in rows]
            for row, label in zip(rows, labels):
                original = dict(s=row.get("s"), rare=row["f"] == 1, l=row["l"],
                                rel=[x == "1" for x in row["j"]],
                                u={int(k): v for k, v in row.get("u", {}).items()})
                assert reference["class_at"](original, theta) == label
            count = Counter(labels)
            counts.append(count["ONE"])
            classes[str(theta)] = {label: {"n": count[label], "fraction": count[label] / 600,
                                                  "wilson95": wilson(count[label], 600)}
                                  for label in ("ONE", "TWO-REL", "UNREL", "UNK", "RARE")}
        results[model["key"]] = {"n_poles": 600, "classes": classes, "thresholds": data["stops"],
                                  "one_counts": counts, "published_one_counts": model["check"]["stops"],
                                  "match": counts == model["check"]["stops"]}
    return results


def parent_records():
    data = read_js(WALK + "judge/data.js")
    records = []
    for row in data["groups"]:
        pole, name, checkpoint, control, size, bits, decoys = row
        assert len(bits) == 4 and set(bits) <= {"0", "1"}
        records.append(dict(pole=pole, neuron=pole[:-1], name=name, checkpoint=checkpoint,
                            control=control, size=size, accuracy=bits.count("1") / 4))
    return data, records


def parent_bootstrap(records, cluster="pole", seed=0, replicates=10000):
    units = defaultdict(lambda: np.zeros(4))
    for row in records:
        vector = units[row[cluster]]
        pos = 2 if row["control"] else 0
        vector[pos] += row["accuracy"]
        vector[pos + 1] += 1
    array = np.array([units[key] for key in sorted(units)])
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 500):
        idx = rng.integers(len(array), size=(min(500, replicates - start), len(array)))
        totals = array[idx].sum(axis=1)
        assert (totals[:, 1] > 0).all() and (totals[:, 3] > 0).all()
        A, control = totals[:, 0] / totals[:, 1], totals[:, 2] / totals[:, 3]
        assert (control > .5).all()
        estimates.extend(np.clip((control - A) / (control - .5), 0, 1))
    return {"unit": cluster, "n_units": len(array), "replicates": replicates, "seed": seed,
            "ci95": np.quantile(estimates, [.025, .975]).tolist()}


def q2():
    data, records = parent_records()
    test = [r["accuracy"] for r in records if not r["control"]]
    controls = [r["accuracy"] for r in records if r["control"]]
    A, control = float(np.mean(test)), float(np.mean(controls))
    point = (control - A) / (control - .5)
    boot = parent_bootstrap(records)
    return {"n_late": len(test), "n_control": len(controls), "A": A, "a_ctrl": control,
            "p_u": point, "bootstrap": boot, "published_p_u": data["share"],
            "published_ci95": data["ci95"], "flagged_late": sum(x <= .25 for x in test),
            "match": bool(np.allclose([A, control, point, *boot["ci95"]],
                                      [data["A"], data["a_ctrl"], data["share"], *data["ci95"]], atol=1e-12))}


def oldest():
    path = INPUTS / "studies/next/10_recount2/out/births_all.csv"
    poles = defaultdict(list)
    for row in csv.DictReader(path.open()):
        if row["sample"] == "0":
            poles[row["pole"]].append(row)
    poles = {p: rows for p, rows in sorted(poles.items()) if len(rows) >= 2}
    all_births, hit, strict, sensitivity = [], [], [], {}
    def dates(rows, key):
        return np.array([int(r[key]) if r[key] not in ("", "None", "null") else 99 for r in rows])
    for rows in poles.values():
        b = dates(rows, "birth_pct")
        main = next(i for i, row in enumerate(rows) if row["is_main"] == "1")
        all_births.append(b)
        hit.append(b[main] == b.min())
        strict.append(b[main] == b.min() and (b == b.min()).sum() == 1)
    hit, strict = np.array(hit, float), np.array(strict, float)
    rng, nboot = np.random.default_rng(1105), 10000
    boot = hit[rng.integers(len(hit), size=(nboot, len(hit)))].mean(1)
    strict_boot = strict[rng.integers(len(hit), size=(nboot, len(hit)))].mean(1)
    perm = np.zeros(nboot)
    for b in all_births:
        perm += b[rng.integers(len(b), size=nboot)] == b.min()
    perm /= len(hit)
    for field in ("birth_pct_lower", "birth_pct_upper"):
        vals = []
        for rows in poles.values():
            b = dates(rows, field)
            main = next(i for i, row in enumerate(rows) if row["is_main"] == "1")
            vals.append(b[main] == b.min())
        sensitivity[field] = float(np.mean(vals))
    published = read_json("studies/next/10_recount2/out/04_b.json")
    result = {"n_poles": len(hit), "k": int(hit.sum()), "b": float(hit.mean()),
              "ci95": np.quantile(boot, [.025, .975]).tolist(), "strict_no_ties": float(strict.mean()),
              "strict_ci95": np.quantile(strict_boot, [.025, .975]).tolist(),
              "permutation_p95upper": float(np.quantile(perm, .975)),
              "permutation_p_one_sided": float((1 + (perm >= hit.mean()).sum()) / (nboot + 1)),
              "median_convention_sensitivity": sensitivity}
    result["published_b"] = published["b"]
    result["match"] = bool(np.allclose([result["b"], *result["ci95"], result["permutation_p_one_sided"]],
                                        [published["b"], *published["b_ci95_bootstrap_poles"],
                                         published["permutation_band"]["p_one_sided"]], atol=1e-12))
    return result


def viewer(out):
    data = read_js(WALK + "neuron_across_training/data.js")
    expected_lithium = [0, 0, 0, 0, 0, 0, 1, 0, 11, 20, 19, 19, 19, 1, 3, 0]
    rows = []
    for neuron in data["neurons"]:
        for i, snapshot in enumerate(neuron["steps"]):
            contexts = [neuron["ctx"][idx] for idx, activation in snapshot["top"]]
            counts = Counter(c[1] for c in contexts)
            assert counts == dict(snapshot["counts"])
            assert len(contexts) == 20
            rows.append({"pole": neuron["pole"], "group": neuron["group"], "checkpoint": i,
                         "step": data["steps"][i]["step"], "max_activation": snapshot["max"],
                         "lithium_count": counts[" lithium"]})
    lithium = [r for r in rows if r["pole"] == "L03N6372+"]
    random_ratios = [max(s["max"] for s in n["steps"]) / n["steps"][-1]["max"]
                     for n in data["neurons"] if n["group"] == "random"]
    write_csv(out / "checkpoint-observations.csv", rows)
    return {"n_poles": len(data["neurons"]), "n_checkpoints": len(data["steps"]),
            "group_counts": dict(Counter(n["group"] for n in data["neurons"])),
            "lithium_counts": [r["lithium_count"] for r in lithium],
            "lithium_counts_match": [r["lithium_count"] for r in lithium] == expected_lithium,
            "lithium_peak_to_final_max_ratio": max(r["max_activation"] for r in lithium) / lithium[-1]["max_activation"],
            "random_peak_to_final_median": float(np.median(random_ratios)),
            "random_ratio_above10": sum(x > 10 for x in random_ratios)}


def fixed_examples():
    examples = read_json("studies/next/10_recount2/out/06_examples.json")
    result = {}
    for name, row in examples.items():
        values = np.array(row["a_plus_by_t"])
        median = np.median(values, axis=1)
        present = (median > 0) & (median >= np.array(row["q999_by_t"]))
        date = birth(present.tolist())
        result[name] = {"pole": row["pole"], "n_contexts": values.shape[1], "birth": date,
                        "published_birth": row["birth_banked"], "birth_match": date == row["birth_banked"],
                        "max_median_rounding_error": float(np.max(np.abs(median - np.array(row["median_by_t"]))))}
    return result


def interventions():
    primary = read_json("studies/reuse/results/primary/verdict.json")
    errors = []
    for job in primary["jobs"]:
        for time in ("birth_load", "T15"):
            record = job[time]
            if record and record["I"] is not None and record["falls"]["own"]["mean"] != 0:
                ratio = record["falls"]["cross"]["mean"] / record["falls"]["own"]["mean"]
                errors.append(abs(ratio - record["I"]))
    live = [job["T15"]["I"] for job in primary["jobs"] if job["T15"] and not job["T15"]["cancelled"]]
    marked = read_json("studies/instruments/results/e6/verdict_e6c.json")
    cases = [r["I_marked_mean_e"] for r in marked["reported"]["per_case"] if r["live_marked"]]
    return {"ratio_checks": len(errors), "max_ratio_error": max(errors),
            "primary_mean_write": {"n": len(live), "median": float(np.median(live)),
                                   "published": primary["reported"]["I_T15"]["value"]},
            "marked_mean_write": {"n": len(cases), "median": float(np.median(cases)),
                                  "published": marked["readings"]["I_marked_mean_e"]["I"]["value"]},
            "limits": "Reaggregated stored job-level falls and ratios; no interventions rerun. Zero-write 0.293 lacks per-case public input in this subset."}


def run_original_extractor(out):
    # The original script writes its data.js: run on a private scratch copy.
    scratch = out / "original-extractor"
    shutil.copytree(INPUTS, scratch, dirs_exist_ok=True)
    # Source archives have no .git directory. Substitute only the metadata lookup,
    # using the manifest's pinned public revision; numerical code is unchanged.
    shim = '''import runpy, sys, subprocess
original_run = subprocess.run
def metadata_run(args, **kwargs):
    if args[0] == "git" and args[-2:] == ["rev-parse", "HEAD"]:
        return subprocess.CompletedProcess(args, 0, stdout=sys.argv[2] + "\\n", stderr="")
    return original_run(args, **kwargs)
subprocess.run = metadata_run
runpy.run_path(sys.argv[1], run_name="__main__")
'''
    from .common import ROOT
    pin = json.loads((ROOT / "source_manifest.json").read_text())["commit"]
    proc = subprocess.run([sys.executable, "-c", shim, str(scratch / WALK / "m05/extract.py"), pin],
                          text=True, capture_output=True, check=False)
    (out / "original-m05-stdout.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"Original m05 extractor failed: {proc.stderr[-2000:]}")
    original = read_js(WALK + "m05/data.js")
    text = (scratch / WALK / "m05/data.js").read_text()
    recomputed = json.loads(text[text.index("{"):].strip().removesuffix(";"))
    comparisons = {key: recomputed[key] == original[key] for key in ("snapshot_steps", "example", "gradual")}
    shutil.rmtree(scratch)
    return {"exit_code": proc.returncode, "sections_match": comparisons,
            "scope": "Original m05 numerical payload; timestamp excluded. Metadata-only git lookup supplied from pinned source manifest because run archives lack .git."}


def run(out):
    results = {}
    for name, function in [("q1", q1), ("parent_test", q2), ("main_oldest", oldest),
                           ("checkpoint_viewer", lambda: viewer(out)), ("fixed_examples", fixed_examples),
                           ("interventions", interventions), ("original_m05", lambda: run_original_extractor(out))]:
        results[name] = function()
        print("REPRODUCTION", name, json.dumps(results[name], allow_nan=False), flush=True)
    return results
