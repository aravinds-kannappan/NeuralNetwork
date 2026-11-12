# Reference sources and attribution

This directory records original-source URLs and SHA-256 checksums for the downloaded reference material used during the review. The full papers and upstream neuron-viewer dataset are linked, not copied into this repository. `sources.json` preserves their original local filenames and hashes so they can be retrieved and checked independently.

| Figure excerpts in `artifacts/abay-review/` | Original source |
| --- | --- |
| `borrowed-geometry-figure2.png`, `borrowed-geometry-figure4.png` | Abay Bektursun, [Borrowed Geometry, version 2](https://www.alphaxiv.org/pdf/2605.00333v2). The original review records the paper's CC BY 4.0 attribution. |
| `bsf-curve-manifold-figure11.png` | Goodfire, [block sparse featurizers for vision](https://www.goodfire.com/research/bsf-vision), [paper](https://www.alphaxiv.org/pdf/2606.25234). |
| `instagent-architecture-figure1.png` | The authors of [InstAgent](https://www.alphaxiv.org/pdf/2610.23098). |

The figure excerpts are included for attributed research discussion. They are not original experimental results from this project. Their extraction scripts and the accompanying review preserve the context and original links.

The interactive report's layout is inspired by [Abay's checkpoint walkthrough](https://abaybektursun.github.io/already-clean-audit/#words). Its prose, front-end implementation, and robot experiments were produced separately.

OGBench, MuJoCo, PyTorch, Transformers, Gemma, and the other dependencies retain their respective licenses. This repository pins dependencies; it does not relicense those projects or include the full pretrained model.
