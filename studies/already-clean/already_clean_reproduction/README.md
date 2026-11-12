# Already-clean audit reproduction

Original work: [Abay Bektursun](https://github.com/abaybektursun/already-clean-audit), snapshot `de979b861f26a4235e38d0a605952d8d94d6a8e9`.

This module is kept within the separate `studies/already-clean/` study. `inputs/` contains 56 unchanged, hash-verified public files; `source_manifest.json` records their provenance. `ANALYSIS_PLAN.md` and `FOLLOWUP_PLAN.md` document the analysis sequence.

From the parent study directory, run all phases with:

```sh
uv run --locked python reproduce_all.py --output ./outputs
```

Use a fresh output path. This is a public-data computational reproduction, not fresh model inference. See the parent `report.md`, `PUBLICATION_NOTES.md`, and `missing-inputs.md` for measured scope and interpretation. The `borrowed_recovery` module in the parent directory is a retained compatibility entrypoint from the original project; it is not needed when using the standalone runner.
