# Reproducing Abay's checkpoint audit and extending it with similarity analysis

Completed 2026-11-11 Pacific. Source snapshot:
[`de979b861f26a4235e38d0a605952d8d94d6a8e9`](https://github.com/abaybektursun/already-clean-audit/tree/de979b861f26a4235e38d0a605952d8d94d6a8e9).
Original work and public measurements: **Abay Bektursun**.

The accessible calculations reproduced successfully. The strongest additional
finding is **semantic continuity beyond exact example reuse**: after removing
repeated contexts, a neuron's later examples remain more similar to its own final
examples than to other neurons in the same layer. The effect is
**0.176 [0.091, 0.293] cosine**, using 13 eligible neurons and a 95% bootstrap
interval over nine layer blocks. This is exploratory evidence about selected text,
not proof that a neuron's causal function stays the same.
[Follow-up measurements](runs/50287c02-bb89-433d-89f3-334474033fc2/results.json).

## What was actually reproduced

This is a **computational reproduction from published measurements**, plus new
analyses. We did not regenerate the original OLMo activations, judge responses,
or interventions. The current repository contains code and useful exports, but
the full raw banks still point outside it; the documented S3 object tested here
returned HTTP 403. [Exact missing inputs and hashes](missing-inputs.md).

Fifty-six input files (8,558,654 bytes) were SHA-256 verified. The completed run
independently recomputed classification rules, bootstrap estimates, birth rules,
and stored intervention ratios. It also ran the author's original `m05/extract.py`
and matched all three numerical payload sections. Only its Git metadata lookup
was supplied from the pinned source manifest because the run archive has no Git
history; the numerical script was unchanged.
[Reproduction output](runs/7a50ed02-338d-4457-a929-e43b192a1fdd/results.json),
[original extractor output](runs/7a50ed02-338d-4457-a929-e43b192a1fdd/original-m05-stdout.txt).

Under the original classification threshold, the one-job readout is:

| Model | Poles classified ONE | Fraction | Original Wilson 95% interval, recomputed | Reproduction |
|---|---:|---:|---:|---|
| Qwen3-4B-Base | 358/600 | 59.67% | 55.69–63.52% | Exact count match |
| Gemma-3-4B-pt | 338/600 | 56.33% | 52.34–60.25% | Exact count match |
| OLMo-3 7B, end of stage 1 | 311/600 | 51.83% | 47.84–55.81% | Exact count match |

These intervals reproduce the author's pole-based method; they are not a new
independent-model confidence statement. Counts also matched at both threshold
sensitivity settings. These are different models and definitions from the
original essay's roughly 90% estimate, so this is not a direct refutation of it.
[Original classification definition](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/SPEC_D1.md).

Other checks completed:

- **Parent test:** 1,238 late-born groups and 600 controls; accuracy
  $A=0.811591$ and control accuracy
  $a_{ctrl}=0.924167$ give
  $p_u=(a_{ctrl}-A)/(a_{ctrl}-0.5)=0.265404$.
  The original pole-bootstrap 95% interval is reproduced exactly:
  [0.210894, 0.320826].
  This is the judge-calibrated estimate of unrelated **late-born groups**, not the
  fraction of all neurons that are unrelated. The controls are easier than the test
  items: Abay's overlap-reweighted sensitivity is 0.206 [0.132, 0.271], spanning
  the original decision threshold of 0.20. This is an upstream sensitivity, not a
  replacement estimate or a computation independently repeated here.
  [Original journal, 24 October](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/JOURNAL.md).
- **Main group is oldest, including ties:** 273/574
  = 0.475610. The later public recount, including its seeded
  permutation result, reproduced exactly. Its Monte Carlo $p=0.000300$
  differs from the first report's rounded 0.0005; the recount uses its own seed
  and simulation stream. These are not contradictory point estimates. **The interpretation was retracted
  by Abay on 5 November:** random groups matched for size and top rank reach
  0.530045, above the observed 0.475610. The old rank-permutation null does not
  preserve the relevant group construction. We reproduced that old arithmetic,
  not evidence that the main job is oldest. The 0.530045 reference is reported
  from the upstream export, not independently regenerated without the raw arrays.
  [Upstream recount, section 7](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/next/10_RECOUNT2.md).
- **Checkpoint viewer:** all 16 published lithium-token counts matched; the
  peak/final maximum-activation ratio is 592.61.
  All 24 poles in the viewer's random group exceed a ratio of ten. These are
  selected-pole and rounded-export results, not a whole-network estimate.
- **Birth dating:** all three released fixed-context examples reproduced their
  recorded birth checkpoint. The original m05 extractor also reproduced its
  153-birth gradual-rise analysis. A threshold crossing is therefore not, by
  itself, evidence of an abrupt new mechanism.
- **Interventions:** all 300 available cross/own
  ratio identities checked exactly. The primary mean-write median was
  0.375346; the marked-token
  mean-write median was 0.230794.
  These are distinct conditions. The headline zero-write 0.293 and the 277-case
  driver-overlap correlation were **not independently recomputed** from this input
  subset; neither was fresh causal intervention performed.

The six-study repository contains more findings than these checks. Judge kappa,
every original sensitivity, all other studies' results, and private freeze
provenance were not regenerated. The detailed run output, rather than a blanket
claim of full replication, defines the completed scope.

## New analysis: semantic continuity versus changing examples

We encoded **10,363 unique full contexts** from 41 neuron poles across 16 training
checkpoints. Of these poles, 17 were chosen using final outcomes; the primary new
analyses therefore use the viewer's separate 24-pole random subset. It is still
a conditional sample: those 17 poles were excluded from its sampling pool.

The analysis uses pinned MiniLM sentence encoders, normalized centroid cosine,
word/bigram TF-IDF cosine, marked-token frequency cosine, and exact-context
Jaccard overlap. At step 560,000, the 24-pole random subset has mean semantic
cosine 0.652 to the final checkpoint,
but mean exact-context Jaccard 0.133.
These measures have different scales, so their numerical gap is not a test by
itself. The same-layer negative controls and disjoint-context follow-up provide
the relevant comparison. [Per-pole observations](runs/eb114b2c-a2c1-4fa1-9300-80d65f87669c/checkpoint-similarities.csv).

The interactive figure defaults to those controlled differences. Its second view
shows the trajectory and the lexical measures; exact values are available on hover.

![Checkpoint similarity and controlled continuity](figures/checkpoint-analysis.html)

[Open interactive figure](figures/checkpoint-analysis.html) ·
[Chart data](figures/chart-data.json) ·
[Statistical source](runs/50287c02-bb89-433d-89f3-334474033fc2/results.json).

**What the follow-up adds.** Removing repeated full contexts reduced the original
contrast but left a positive effect. The remaining same-neuron sets contained
12–20 contexts each; donor sets were subsampled to exactly the same target count
200 times. Removing shared marked-token identities also left a contrast of
0.160 [0.079, 0.277], on the same 13 eligible neurons, with 6–20 target contexts.
The prefix-only and second-encoder checks retained positive contrasts.
Mean cosine across individual cross-context pairs also stayed positive after
control subtraction, 0.026 [0.012, 0.046];
the finding is therefore not solely an artifact of normalizing diffuse centroids.
[Per-neuron disjoint-context contrasts](runs/50287c02-bb89-433d-89f3-334474033fc2/primary_full_disjoint_text.csv).

**Raw cosine needs a control.** The `[MASK]` intervention raised same-neuron raw
cosine from 0.598 to
0.830, but also raised mismatched-neuron cosine
from 0.322 to
0.702. A high absolute cosine can therefore
reflect shared formatting or embedding components. Deleting the marked token
instead of inserting a mask, and excluding repeated contexts, gave
0.178 [0.095, 0.294] after control subtraction.
[Embedding sensitivity output](runs/eb114b2c-a2c1-4fa1-9300-80d65f87669c/results.json),
[deletion follow-up](runs/50287c02-bb89-433d-89f3-334474033fc2/primary_deleted_disjoint_text.csv).

**The statistical limit matters.** The stricter random-only, same-layer identity
permutation test has just four pairs of eligible neurons: 16 possible assignments.
Its exact $p=0.0625$ is the smallest attainable p-value in that design, so it cannot
reject at 5%. Requiring the same pole sign leaves only one pair and $p=0.5$.
The positive bootstrap intervals and this underpowered exact test answer
different conditional questions; the former do not turn the latter into a
significant result. The follow-up's layer bootstrap handles shared control
donors within layers, but cannot repair selection of the exported sample.

## Additional checks and interpretation

The parent-test uncertainty was recomputed over **300 underlying neurons**, keeping
positive and negative poles together. Its interval, 21.28–31.90%,
is close to the original 600-pole interval. This dependence check did not overturn
the original estimate. It also does not remove judge calibration bias or validate
the control ceiling; those are different questions.

Actual activation-vector cosines were computed for the three fixed-example
exports, with only four or five contexts each. Zero activation vectors are marked
undefined, not assigned similarity zero. The access/CRUD example has zero vectors
at checkpoints 3 and 4. These matrices are descriptive, selected examples.
[Fixed-example activation matrices](runs/eb114b2c-a2c1-4fa1-9300-80d65f87669c/fixed-activation-geometry.json).

**Our interpretation:** changing strongest examples, shrinking activation
magnitudes, and gradual threshold crossings can coexist with some continuity in
the semantics of the text that activates a neuron. Similarity techniques reveal
this continuity without reducing the analysis to exact token identity. They do
not show that a neuron has one coherent job, that later clusters descend causally
from earlier clusters, or that the original recursive-generalization thesis is true.

## Strongest next experiments

1. **Fixed-input geometry over the full sample.** Obtain the already computed
   4.54 MB activation array and 4.92 MB birth record linked in
   [missing inputs](missing-inputs.md). Compare each pole on the same contexts
   at every checkpoint using raw and normalized cosine, rank correlation and
   matched-input CKA/RSA. Include same-layer, same-sign controls across the full
   sample. This directly addresses the top-k censoring and small-null limits.
2. **Calibrated semantic parent test.** With the full judge prompts, compare
   embedding-based parent ranking against the original judgments on held-out
   neurons. Match lexical overlap, token identity and training interval between
   tests and controls. This asks how much of the parent score is semantic rather
   than an easier control set; do not calibrate and evaluate on the same neurons.
3. **Semantic continuity versus causal reuse.** On a new, independently sampled
   set, test whether semantic continuity predicts measured cross-cluster
   intervention effects after controlling for shared drivers, layer and baseline
   activation. Similar text with different causal effects would falsify the
   interpretation that semantic persistence tracks the same mechanism.

The disjoint-context, token-disjoint and encoder/deletion analyses above were
actually run. These three larger experiments remain proposed; no results are
invented for unavailable raw data or new model runs.

## Files, provenance, and rerunning

- [Complete source package](reproduction-source.zip): locked dependencies,
  pinned input files, analysis plan, statistical checks, and a standalone runner.
- [Methods and original research](methods-and-reading.md): equations, encoder
  revisions, inference units, and the reason not to apply CKA to unmatched top-k lists.
- [Follow-up plan](followup-plan.md): explicitly written after the initial result.
- [Reproduction provenance](runs/7a50ed02-338d-4457-a929-e43b192a1fdd/provenance.json),
  [similarity provenance](runs/eb114b2c-a2c1-4fa1-9300-80d65f87669c/provenance.json),
  [follow-up provenance](runs/50287c02-bb89-433d-89f3-334474033fc2/provenance.json).
- [Source and output index](deliverables.json): every saved file and its role.

Three completed experiment nodes: public-data reproduction (commit 4a3d3ac),
similarity with controls (91f52d0), and disjoint-context follow-up (0e65829).
The similarity node had one implementation failure before any statistical result:
activation weighting needed the sign of negative poles. It was repaired, covered
by a numerical test, and rerun successfully. Eight focused numerical tests passed.
All measurements use one original OLMo training run; 10,000 bootstrap resamples
do not constitute 10,000 independent training runs. New analyses are exploratory
and use the same released dataset. Nothing was published to the upstream repository.
