"""Create one offline HTML report. Does not publish or access the network."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "already-clean"


def build(target: Path, *, published: bool = False):
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    page = (SITE / "index.html").read_text()
    css = (SITE / "assets/site.css").read_text()
    js = (SITE / "assets/site.js").read_text()
    data = {name: json.loads((SITE / name).read_text()) for name in ("assets/results.json", "assets/explorer.json")}
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    page = page.replace('<link rel="stylesheet" href="assets/site.css">', f"<style>\n{css}\n</style>")
    page = re.sub(r'^[ \t]*<script src="assets/site.js" defer></script>[ \t]*\n', "", page, flags=re.M)
    page = page.replace("</body>", f"<script>window.REPORT_DATA={serialized};</script>\n<script>\n{js}\n</script>\n</body>")
    figure = html.escape((SITE / "figures/checkpoint-analysis.html").read_text(), quote=True)
    page = page.replace('src="figures/checkpoint-analysis.html"', f'srcdoc="{figure}"')
    icon = base64.b64encode((SITE / "assets/favicon.svg").read_bytes()).decode()
    page = page.replace('href="assets/favicon.svg"', f'href="data:image/svg+xml;base64,{icon}"')
    if published:
        site_url = "https://aravinds-kannappan.github.io/NeuralNetwork/already-clean/"
        for asset in ("downloads/reproduction-source.zip", "downloads/results-and-tables.zip", "downloads/beginner-report.html", "assets/results.json"):
            page = page.replace(f'href="{asset}"', f'href="{site_url}{asset}"')
        target.write_text(page)
        print(f"Standalone HTML: {target.name} ({target.stat().st_size:,} bytes). No external script, style, or data dependency.")
        return
    page = page.replace("<title>What survives as a neuron learns? · NeuralNetwork</title>", "<title>Review draft · What survives as a neuron learns?</title>")
    page = re.sub(r'\s*<link rel="canonical"[^>]*>', '', page)
    page = re.sub(r'\s*<meta property="og:url"[^>]*>', '', page)
    page = page.replace('<p class="eyebrow"><span class="study-number">Study 01</span>', '<p class="review-banner">Review draft · Not yet published. The report, example data, and interactive figures are embedded in this HTML file.</p>\n        <p class="eyebrow"><span class="study-number">Study 01</span>')
    repo_url = "https://github.com/aravinds-kannappan/NeuralNetwork"
    def local(path):
        return Path(os.path.relpath(ROOT / path, target.parent)).as_posix()
    # Publication is pending approval. Link to the prepared local files instead of nonexistent remote paths.
    page = re.sub(r'href="' + re.escape(repo_url) + r'/(?:blob|tree)/main/([^"]+)"', lambda m: 'href="' + html.escape(local(m.group(1)), quote=True) + '"', page)
    page = page.replace(f'href="{repo_url}"', 'href="#reproduce"')
    page = page.replace('Code & data <span', 'Files & methods <span')
    page = page.replace('Explore the repository ↗', 'Inspect the local files ↑')
    page = page.replace('Read them on GitHub ↗', 'Read the question plans ↗')
    page = page.replace('<span class="file-type">GitHub</span>', '<span class="file-type">Files</span>')
    for asset in ("downloads/reproduction-source.zip", "downloads/results-and-tables.zip", "downloads/beginner-report.html", "assets/results.json"):
        page = page.replace(f'href="{asset}"', f'href="{html.escape(local("already-clean/" + asset), quote=True)}"')
    target.write_text(page)
    print(f"Review HTML: {target} ({target.stat().st_size:,} bytes). No external script, style, or data dependency.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published", action="store_true", help="Use public repository links and omit the review banner")
    args = parser.parse_args()
    build(args.output, published=args.published)


if __name__ == "__main__":
    main()
