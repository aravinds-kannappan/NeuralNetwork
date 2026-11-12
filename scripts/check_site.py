"""Validate publication paths, provenance, and rendered data without external packages."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "already-clean"
STUDY = ROOT / "studies/already-clean"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.labels, self.inputs = [], [], set(), []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for key in ("src", "href"):
            if key in a:
                self.links.append(a[key])
        if tag == "label" and "for" in a:
            self.labels.add(a["for"])
        if tag in {"input", "select"} and not a.get("aria-label"):
            self.inputs.append(a.get("id"))


def main():
    pages = {p: Page(p.read_text()) for p in SITE.rglob("*.html")}
    count = 0
    for path, page in pages.items():
        assert len(page.ids) == len(set(page.ids)), f"Duplicate IDs in {path}"
        for control in page.inputs:
            assert control in page.labels, f"Unlabelled control: {control} in {path}"
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                if href.startswith("https://github.com/aravinds-kannappan/NeuralNetwork/"):
                    parts = unquote(url.path).split("/")
                    if len(parts) > 5 and parts[3] in {"blob", "tree"} and parts[4] == "main":
                        assert (ROOT / "/".join(parts[5:])).exists(), href
                continue
            assert not url.path.startswith("/"), f"Root-absolute path breaks the project Pages subpath: {href}"
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), f"Missing target: {href} from {path}"
            if url.fragment and target in pages:
                assert url.fragment in pages[target].ids, f"Missing fragment: {href}"
            count += 1
    data = json.loads((SITE / "assets/results.json").read_text())
    viewer = json.loads((SITE / "assets/explorer.json").read_text())
    assert len(viewer["neurons"]) == 41 and len(viewer["steps"]) == 16
    for n in viewer["neurons"]:
        assert len(n["steps"]) == len(n["similarities"]) == 16
        for i, checkpoint in enumerate(n["steps"]):
            assert len(checkpoint["top"]) == 20
            assert n["similarities"][i]["step"] == viewer["steps"][i]
            assert all(0 <= ctx < len(n["ctx"]) for ctx, _ in checkpoint["top"])
    assert len(data["questions"]) == 7
    assert data["similarity"]["tests"]["permutation_random_layer"]["p_one_sided"] == 0.0625
    result = data["followup"]["comparisons"]["primary_full_disjoint_text"]["centroid_layer"]
    assert round(result["mean"], 3) == .176 and result["n_poles"] == 13 and result["n_layers"] == 9
    for z in (SITE / "downloads").glob("*.zip"):
        with zipfile.ZipFile(z) as archive:
            assert archive.testzip() is None, z
    manifest = json.loads((STUDY / "deliverables.json").read_text())
    for row in manifest["files"]:
        p = STUDY / row["path"]
        assert p.stat().st_size == row["bytes"], p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"], p
    # Host paths may exist in unchanged upstream inputs, but never in our web assets/config.
    for path in [SITE / "assets/results.json", SITE / "assets/explorer.json", STUDY / "already_clean_reproduction/config.json"]:
        assert "/Users/aravindkannappan/" not in path.read_text(), path
    print(f"PASS: {len(pages)} HTML pages, {count} internal links, 41 interactive poles, 7 proposed experiments, download CRCs, and {len(manifest['files'])} artifact hashes.")


if __name__ == "__main__":
    main()
