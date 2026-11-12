"""Shared input, provenance and statistics utilities."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
WALK = "studies/already-clean/walkthrough/"


def read_json(rel):
    return json.loads((INPUTS / rel).read_text())


def read_js(rel):
    text = (INPUTS / rel).read_text()
    # These files are data assignments. Never execute upstream JavaScript.
    return json.loads(text[text.index("{"):].strip().removesuffix(";"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for record in manifest["files"]:
        data = (INPUTS / record["path"]).read_bytes()
        assert len(data) == record["bytes"], record["path"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
    return {"commit": manifest["commit"], "files_verified": len(manifest["files"])}


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / norm, -1, 1)) if norm > 0 else None


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0, center - width)), float(min(1, center + width))]


def birth(present):
    return next((i for i, value in enumerate(present)
                 if value and (i == len(present) - 1 or present[i + 1])), None)
