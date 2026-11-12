# NeuralNetwork

Research connecting **Borrowed Geometry**, neuron-checkpoint analysis, and learned robot control.

## Research walkthroughs

| Study | Read | Code and evidence |
| --- | --- | --- |
| **What survives as a neuron learns?** — beginner-friendly reproduction of Abay’s checkpoint audit, embeddings, similarity controls, and open questions | [Live GitHub Pages report](https://aravinds-kannappan.github.io/NeuralNetwork/already-clean/) · [Standalone HTML](https://aravinds-kannappan.github.io/NeuralNetwork/already-clean/downloads/beginner-report.html) | [Separate checkpoint study](studies/already-clean/) |
| **Borrowed Geometry and learned robot control** — completed policy comparisons, limitations, and proposed tests | [Interactive robot-control walkthrough](artifacts/abay-review/geometry-transfer/report.html) | [Robot-control evidence bundle](artifacts/abay-review/geometry-transfer/) |

The checkpoint study contains its own locked Python environment, verified inputs, recorded results, and saved text embeddings. Its website is served from `already-clean/`; its analysis code is in `studies/already-clean/`. The root environment and experiment histories below describe the robot-control work.

For robot-control work, start with the **[interactive research walkthrough](artifacts/abay-review/geometry-transfer/report.html)**. It explains the terms for beginners, includes expandable technical methods, and lets you inspect saved robot outcomes and neuron-response comparisons. Download the HTML file or clone/download this repository and open `index.html` in a browser; GitHub's file view displays HTML source. Reading the walkthrough runs no model or simulation.

The [compact evidence bundle](artifacts/abay-review/geometry-transfer/reproduction.zip) includes the walkthrough, measurements, code, and selected trained adapters. The larger raw activation files are also included directly in this repository under `runs/`.

## Robot-control work: what actually completed

Five learned policies completed **600 robot evaluation episodes**: pretrained Gemma, randomized matrices, scrambled matrices, and a small MLP at training seed 42, plus a second pretrained policy at seed 1337. There is **one complete matched training seed**, not a completed three-seed study. The remaining computation was stopped at the user's request.

The original and randomized matrices tied in the complete seed, but succeeded on different episodes. This motivates testing complementary specialists and routing; it does not demonstrate a working robot team. See the walkthrough for measured results, limitations, and the proposed small tests.

The earlier scripted recovery pilot is preserved separately. It is not evidence that a pretrained neural representation was learned or transferred. Interface exchange, paired-state alignment, multi-arm coordination, expert routing, and recursive self-improvement were not run.

## Where to look

| Path | Contents |
| --- | --- |
| [`artifacts/abay-review/geometry-transfer/`](artifacts/abay-review/geometry-transfer/) | Current walkthrough, prior report, tables, neuron audit, recorded replay, trained policy adapters, and reproducibility bundle |
| [`borrowed_recovery/`](borrowed_recovery/) | Executed policy, simulator, training, evaluation, and pinned Gemma-layer extraction code |
| [`GEOMETRY_PROTOCOL.md`](GEOMETRY_PROTOCOL.md) | Learned-policy experiment design and control definitions |
| [`runs/index.json`](runs/index.json) | Index of the eight runs from this research thread, including interrupted and setup-failure runs |
| [`runs/fb3dd5e8-79fb-4457-ba32-7fbf23077a3c/`](runs/fb3dd5e8-79fb-4457-ba32-7fbf23077a3c/) | Consolidated completed policies, initial/final adapters, full saved activation fingerprints, traces, and evaluation records |
| [`data/demonstrations/`](data/demonstrations/) | Saved 16- and 256-demonstration datasets, validation examples, and checksum metadata |
| [`artifacts/abay-review/robot-recovery/`](artifacts/abay-review/robot-recovery/) | Earlier scripted recovery proposal, results, and code bundle |
| [`artifacts/abay-review/robotics-hackathon-brief.md`](artifacts/abay-review/robotics-hackathon-brief.md) | Research connections and source review |
| [`references/`](references/) | Original-source links, download checksums, and figure attribution |
| [`PUBLICATION.json`](PUBLICATION.json) | Publication scope and the small portability change to the root configuration |

## Reading and rebuilding the page

The current HTML embeds its measured data and replay, so it works offline. Rebuild only the presentation using the Python standard library:

```sh
python3 artifacts/abay-review/geometry-transfer/build_walkthrough.py
```

This does not train or evaluate a policy. The original report remains at `report-first-edition.html`. Browser checks are recorded in `presentation-checks.json`.

## Reproducing experiments — optional compute

The root contains Python 3.12 project metadata and the committed `uv.lock`. The execution command is:

```sh
uv sync --locked
uv run --locked python -m borrowed_recovery.experiment
```

Only run this when you intend to launch training and simulation. The frozen layer download is about 958 MB, and the full planned sweep takes substantial memory and compute. The repository does not automatically run experiments. GitHub Pages publishes the static website from `main`. A separate workflow validates the checkpoint-study inputs, generated files, and links; it does not train or evaluate models.

The root `config.json` keeps the scientific settings but changes `cache_root` to `.cache/geometry-control` and removes machine-specific resume paths. Exact executed configurations remain under `runs/<run-id>/config.json` and in the immutable experiment history. Configure a smaller seed/method subset if needed. The pinned downloader recreates the original public Gemma layer; full pretrained weights, dependency environments, and caches are not vendored.

The saved demonstration cache can be reused instead of recollecting it:

```sh
mkdir -p .cache/geometry-control/datasets
cp -R data/demonstrations/. .cache/geometry-control/datasets/
```

The raw `fingerprints-*.npz` files retain the same-state neuron responses at saved checkpoints. The selected `policy-*.pt` files contain trained interface weights, normalizers, checkpoint metadata, and the original configuration. They do not contain the frozen Gemma block. The `adapter-random-s1337-step0.pt` file is an initialization from the interrupted model, not a successfully evaluated policy.

## Preserved experiment history

The original commits are retained on five `experiments/` branches:

- `experiments/robot-recovery-baseline`
- `experiments/paired-controller-and-memory-faults`
- `experiments/false-freshness-stress-test`
- `experiments/learned-geometry-control-baseline`
- `experiments/frozen-gemma-low-data-structure-test`

These branches preserve the code and configuration that produced the recorded results. The publication on `main` adds reports, datasets, outputs, and navigation without rewriting those experiments.

## Research sources

- [Borrowed Geometry](https://www.alphaxiv.org/pdf/2605.00333v2)
- [Abay's checkpoint walkthrough](https://abaybektursun.github.io/already-clean-audit/#words) and [source repository](https://github.com/abaybektursun/already-clean-audit)
- [Goodfire's block sparse featurizers for vision](https://www.goodfire.com/research/bsf-vision)
- [Pinned public Gemma revision](https://huggingface.co/google/gemma-4-31B/tree/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89)

Full downloaded third-party papers and the upstream neuron-viewer data are linked in [`references/sources.json`](references/sources.json), rather than republished. Figure excerpts and attribution are listed in [`references/README.md`](references/README.md).
