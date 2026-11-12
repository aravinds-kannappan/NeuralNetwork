"""Build portable website data and downloads from recorded measurements (stdlib only)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from build_review import build as build_standalone_report

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/already-clean"
SOURCE = STUDY / "already_clean_reproduction"
SITE = ROOT / "already-clean"
R = "7a50ed02-338d-4457-a929-e43b192a1fdd"
S = "eb114b2c-a2c1-4fa1-9300-80d65f87669c"
F = "50287c02-bb89-433d-89f3-334474033fc2"


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def archive(target, paths, base):
    """Fixed timestamps make rebuilds byte-for-byte deterministic."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            entry = zipfile.ZipInfo(str(path.relative_to(base)), (2026, 11, 12, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            z.writestr(entry, path.read_bytes())


def main():
    manifest = read(SOURCE / "source_manifest.json")
    for row in manifest["files"]:
        data = (SOURCE / "inputs" / row["path"]).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
    r, s, f = [read(STUDY / "runs" / run / "results.json") for run in (R, S, F)]
    assert all(v["match"] for v in r["q1"].values())
    raw = (SOURCE / "inputs/studies/already-clean/walkthrough/neuron_across_training/data.js").read_text()
    viewer = json.loads(raw[raw.index("{"):].strip().removesuffix(";"))
    viewer["steps"] = [checkpoint["step"] for checkpoint in viewer["steps"]]
    rows = list(csv.DictReader((STUDY / "runs" / S / "checkpoint-similarities.csv").open()))
    lookup = {}
    for row in rows:
        lookup.setdefault(row["pole"], []).append({k: (float(v) if v else None) for k, v in row.items() if k not in {"pole", "neuron", "group"}})
    for n in viewer["neurons"]:
        n["similarities"] = lookup[n["pole"]]
    # Strip bulky upstream machine metadata from the web export only; pinned inputs stay intact.
    viewer.pop("meta")
    viewer["source_commit"] = manifest["commit"]
    write(SITE / "assets/explorer.json", viewer)
    chance = read(SOURCE / "inputs/studies/next/10_recount2/out/05_chance.json")["summary"]["R1t"]
    questions = read(STUDY / "questions.json")
    write(SITE / "assets/results.json", {
        "reproduction": r, "similarity": s, "followup": f,
        "curves": read(STUDY / "runs" / S / "curves.json"),
        "upstream_chance_reference": {"status": "Upstream export; not independently rerun", "data": chance},
        "questions": questions, "runs": {"reproduction": R, "similarity": S, "followup": F},
        "source_commit": manifest["commit"]
    })
    qmd = "# Open questions and next experiments\n\nAll experiments below are proposed, not completed results. The first three address the largest limits of the current study.\n\n"
    for i, q in enumerate(questions, 1):
        qmd += f"## {i}. {q['title']}\n\n**{q['priority']} · {q['status']}**\n\n{q['why']}\n\n**Test:** {q['test']}\n\n**Data needed:** {q['needs']}\n\n**What would weaken the interpretation:** {q['failure']}\n\n"
    (STUDY / "OPEN_QUESTIONS.md").write_text(qmd.rstrip() + "\n")
    (SITE / "figures").mkdir(exist_ok=True)
    chart = (STUDY / "figures/checkpoint-analysis.html").read_text()
    # Adapt only the presentation copy: narrow screens need row labels above the
    # intervals instead of clipped to the left. Recorded data and source figure stay intact.
    chart = chart.replace(
        " left=Math.max(left,Math.min(width*.35,Math.max(...rows.map(({run,p})=>(p.label||run.name).length))*7+28));",
        " const compact=width<560; if(compact){height=rows.length*64+70;chart.style.height=`${height}px`;left=32;}else{left=Math.max(left,Math.min(width*.35,Math.max(...rows.map(({run,p})=>(p.label||run.name).length))*7+28));}")
    chart = chart.replace(
        "{x:left-10,y:cy+4,'text-anchor':'end',fill:'var(--fg)','font-size':12},p.label||run.name)",
        "{x:compact?12:left-10,y:compact?cy-14:cy+4,'text-anchor':compact?'start':'end',fill:'var(--fg)','font-size':12},p.label||run.name)")
    (SITE / "figures/checkpoint-analysis.html").write_text(chart)
    srcfiles = [STUDY / p for p in ("pyproject.toml", "uv.lock", "reproduce_all.py", "RUNNING.md", "PUBLICATION_NOTES.md")]
    srcfiles += list((STUDY / "borrowed_recovery").glob("*.py"))
    srcfiles += [p for p in SOURCE.rglob("*") if p.is_file() and not any(x in p.parts for x in ("__pycache__", "scratch", "upstream"))]
    archive(STUDY / "reproduction-source.zip", srcfiles, STUDY)
    (SITE / "downloads").mkdir(exist_ok=True)
    shutil.copyfile(STUDY / "reproduction-source.zip", SITE / "downloads/reproduction-source.zip")
    resultfiles = [p for p in (STUDY / "runs").rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".txt"} and not p.name.startswith("embedding-index-")]
    resultfiles += [STUDY / p for p in ("report.md", "methods-and-reading.md", "missing-inputs.md", "OPEN_QUESTIONS.md", "PUBLICATION_NOTES.md")]
    archive(SITE / "downloads/results-and-tables.zip", resultfiles, STUDY)
    build_standalone_report(SITE / "downloads/beginner-report.html", published=True)
    files = [{"path": str(p.relative_to(STUDY)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(STUDY.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "deliverables.json"]
    write(STUDY / "deliverables.json", {"runs": [R, S, F], "files": files})
    (SITE / ".nojekyll").touch()
    print(f"Verified {len(manifest['files'])} upstream inputs; exported {len(viewer['neurons'])} poles × {len(viewer['steps'])} checkpoints; indexed {len(files)} study files.")


if __name__ == "__main__":
    main()
