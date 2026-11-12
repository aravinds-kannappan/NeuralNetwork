"""Read-only arithmetic checks of the published neuron-viewer export; no model run.

Repository commit: de979b861f26a4235e38d0a605952d8d94d6a8e9
Source: https://media.githubusercontent.com/media/abaybektursun/already-clean-audit/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/walkthrough/neuron_across_training/data.js
Values in the export were rounded to four significant figures by its producer.
The random group contains 24 poles sampled after excluding the 17 H_MIG poles;
it is not an unbiased sample of all neurons or all development trajectories.
"""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "neuron-data.js").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "3df48c264043926b5da3813e2d76aa385004652535f51b20d474185fcf7a4b1a"
prefix = "window.NEURON_DATA = "
text = raw.decode()
assert text.startswith(prefix)
d = json.loads(text[len(prefix):].strip().rstrip(";"))
li = next(n for n in d["neurons"] if n["pole"] == "L03N6372+")
rows = []
for step, s in zip(d["steps"], li["steps"]):
    rows.append({
        "step": step["step"], "maximum": s["max"],
        "lithium_count_top20": sum(li["ctx"][i][1] == " lithium" for i, _ in s["top"]),
        "top_token_counts": s["counts"],
    })
groups = {}
for group in ("lithium", "late", "random"):
    ns = [n for n in d["neurons"] if n["group"] == group]
    ratios = [max(s["max"] for s in n["steps"]) / n["steps"][-1]["max"] for n in ns]
    groups[group] = {
        "n": len(ns), "median_peak_over_final": statistics.median(ratios),
        "count_peak_over_final_gt10": sum(r > 10 for r in ratios),
        "per_pole": dict(zip((n["pole"] for n in ns), ratios)),
    }
checks = {
    "kind": "arithmetic reanalysis of a published export; no experiment launched",
    "commit": "de979b861f26a4235e38d0a605952d8d94d6a8e9",
    "export_sha256": hashlib.sha256(raw).hexdigest(),
    "n_tokens_scored_reported_by_export": d["n_tokens_scored"],
    "lithium": rows,
    "groups": groups,
    "limitations": [
        "Only the top-20 contexts and top-50 maximum are exported, not all corpus activations.",
        "No full checkpoint-by-example matrix, raw bank manifests or original harvest execution verified.",
        "The random comparison group excludes H_MIG poles; no population prevalence claim.",
        "Raw amplitudes do not establish causal influence; OLMo3 normalizes the MLP output.",
    ],
}
(HERE / "export-checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"n_tokens": d["n_tokens_scored"], "groups": groups}, indent=2))
