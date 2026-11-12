# Recorded robot-geometry experiments

Status: stopped at the user's request. No further compute is running.

Five trained policies have complete evaluations: pretrained/random/scrambled/MLP seed 42, plus pretrained seed 1337. This is ONE complete comparison seed, not a three-seed study. See report.html, data/, success-by-seed.csv, paired-outcomes.csv and provenance.json.

models/ contains selected trained adapters and the initial seed-42 pretrained adapter. Frozen model weights are not included in the archive. The source downloader retrieves the pinned public Gemma layer (about 958 MB). These adapters operate on 21 privileged simulator features; they are not language agents.

source/config.json preserves the run configuration and original local paths for provenance. To run on a different machine, choose a local cache_root and remove resume_from, result_cache and max_models_per_run (or deliberately configure new batches). Reduce training_seeds/modes if reproducing only a subset. The fixed execution recipe is:

    uv run --locked python -m borrowed_recovery.experiment

Do not launch this merely to read the results. Training uses substantial laptop memory and compute. The source's result-cache resume settings preserve prior evaluations; a fresh replication must remove them.

source/interface_exchange_NOT_RUN.py is a proposed follow-up, not an executed experiment and not wired into the entry point. It was only syntax-checked. It has no findings.

Neuron audit: analyze_results.py performs arithmetic on recorded fingerprints. The full fingerprints remain in the original run outputs; this compact archive includes the audit and per-neuron values instead. It does not execute the original lithium-neuron experiment.


Updated presentation
--------------------
report.html is the current chapter-based, self-contained walkthrough. It embeds
its saved measurements and replay and can be opened without a server. The earlier
report is preserved as report-first-edition.html.

To rebuild this presentation from the saved files only:

    python3 build_walkthrough.py

This command uses the Python standard library and launches no experiments.
walkthrough.css and walkthrough.js are the presentation source. The old
build_report.py produced the earlier edition; use build_walkthrough.py for the
current edition. presentation-checks.json records desktop/mobile and interaction
checks. check_walkthrough.py is the browser QA script; it requires Playwright.
