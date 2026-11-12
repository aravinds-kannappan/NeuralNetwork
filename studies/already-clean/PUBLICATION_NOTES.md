# Publication notes and attribution

## Original work

Original research, upstream text, and released measurements are by **Abay Bektursun**, [already-clean-audit](https://github.com/abaybektursun/already-clean-audit), pinned to commit `de979b861f26a4235e38d0a605952d8d94d6a8e9`. The publication retains the 56 original input files byte for byte. Their hashes are in `already_clean_reproduction/source_manifest.json`.

The upstream notice licenses **text and figures under CC BY 4.0**. It is preserved in `already_clean_reproduction/inputs/LICENSE`; [license terms](https://creativecommons.org/licenses/by/4.0/). Do not infer that this notice grants a blanket license to all upstream code, third-party corpus material, model weights, or this repository. Those retain their applicable rights and terms. No new blanket license is asserted here.

This is Aravind Kannappan’s AI-assisted computational reproduction and exploratory extension, with an original website design inspired by the author’s guided walkthrough. The site is not Abay’s site and does not imply his endorsement. Original data collection is credited to him throughout.

## What changed for publication

- Code, reports, inputs, and recorded runs are grouped under this separate study directory.
- The default configuration now writes to `./outputs` and starts with the reproduction phase. The standalone runner still executes all three phases in order. The numerical analysis code is unchanged.
- Original run identifiers and code commit identifiers are retained as provenance. The original experiment commits are not ancestors of this standalone publication repository.
- The local machine’s artifact-root path in our JSON provenance was replaced with `<original-artifact-root>`. No numerical measurement or embedding vector was changed. Upstream input files, including their original metadata, were not modified.
- The source download was rebuilt from the portable publication files. A separate results archive contains compact tables and JSON summaries; all five larger embedding matrices and their text indices remain in the repository under `runs/`.
- The report now explicitly carries the upstream birth-order retraction and parent-test calibration caveat. This is an interpretive clarification, not a change to the recorded computational results. The original report understated these caveats; the publication corrects that omission.
- The failed preliminary similarity attempt is retained with provenance. It failed on negative-pole weighting before producing a scientific result; the repaired successful run and a numerical test are included.

## Corrections carried forward

The original “main is oldest” statistic is 273/574 = 0.475610, and its original within-pole rank-permutation arithmetic reproduces. The more appropriate size/top-rank matched random-group reference released by Abay is 0.530045. Consequently, the original biological-style growth interpretation was retracted on 5 November 2026. The random-group reference is **reported from the upstream export**, not freshly regenerated here: that simulation needs unavailable raw activation arrays.

The parent-test estimate 0.265404 reproduces, but its controls are easier than the test items. Abay’s overlap-reweighted sensitivity is 0.206 [0.132, 0.271], spanning the original decision threshold of 0.20. This sensitivity is quoted as an upstream result, not one of our new computations. It is not a replacement “corrected estimate.”

Sources: [recount §7](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/next/10_RECOUNT2.md), [journal, 24 October and 5 November](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/JOURNAL.md), and the unchanged `inputs/studies/next/10_recount2/out/05_chance.json`.

## Result files

| Phase | Original run ID | Source commit | Status |
|---|---|---|---|
| Public-data reproduction | `7a50ed02-338d-4457-a929-e43b192a1fdd` | `4a3d3ac713c386cb5671a9bf458c14286a651b8e` | Completed |
| Embeddings and controls | `eb114b2c-a2c1-4fa1-9300-80d65f87669c` | `91f52d0e22db17e0426924a7dac1eb27b592269f` | Completed |
| Disjoint-context follow-up | `50287c02-bb89-433d-89f3-334474033fc2` | `0e658294375cc7824f46884e03a8949a10b131e8` | Completed |

Each run’s `results.json` is the authoritative numerical record. Its `provenance.json` records configuration, dependencies, platform, source commit, and input verification. Bootstrap intervals condition on the observed sample; resampling is not independent model training. No new model inference is implied by publishing these artifacts.
