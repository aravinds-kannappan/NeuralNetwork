# Saved experiment evidence

`index.json` identifies each run and the files preserved here. `records.jsonl` contains the exact JSON measurement/configuration records extracted from that run's persisted output; dependency-installation chatter is omitted.

The interrupted run `ed2527c3-cc8f-4df8-a7ee-3a68cbdb6e5d` completed the pretrained, randomized, and scrambled seed-42 models. Its saved files were copied byte-for-byte into the resumable run `fb3dd5e8-79fb-4457-ba32-7fbf23077a3c`. The latter contains the consolidated files, so the large arrays are stored once.

The resumed run completed the seed-42 MLP and seed-1337 pretrained policy, then was cancelled at the user's request during random seed-1337 training. Do not interpret an initial adapter for that unfinished model as a completed policy or evaluation.

Checkpoint fingerprints are recorded arrays, not newly generated results. They contain fixed panel indices and signed hidden/MLP-neuron/action arrays for the saved steps. Policy files are selected trained adapters and normalizers; they exclude the frozen Gemma weights.

Original local paths remain in exact configurations and checkpoint metadata as provenance. They describe where those runs took place. Use the portable root configuration when starting a new run on another machine.
