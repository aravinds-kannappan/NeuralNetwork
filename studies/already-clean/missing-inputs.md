# What prevents a full regeneration of the original experiments

The completed reproduction uses published observations and code from commit
`de979b861f26a4235e38d0a605952d8d94d6a8e9`. It does not claim fresh OLMo
forward passes, fresh language-model judging, or fresh causal interventions.

The current public repository contains much more code than its top-level README
describes, but several necessary measurements remain pointers to the author's
external banks. The most useful missing inputs for the next similarity study are:

| Input | Published size / SHA-256 | Why it matters |
|---|---|---|
| `neurons/next/recount2/arrays.npz` | 4,543,036 bytes; `80e5ffac44ea44b3d7794f3e57bd51aa67ba5b6b8508985d60d1b12d6c6ddba4` | Compare activation vectors on the same fixed contexts across checkpoints for the full sample. |
| `neurons/next/recount2/births.jsonl` | 4,917,094 bytes; `f21230ec07a6912effb73807641e852b945ea1ef0b16f73b4f138cfe5955bb10` | Recompute complete per-cluster trajectories and birth rules rather than only exported dates. |
| Sixteen `w3/grid/{1,1b}/out/tNN_stepN/poles/exemplar_a.npz` and corresponding `samples/contexts.jsonl` | Individual hashes recorded in the public `arrays.pointer.json` | Regenerate the derived arrays and verify every selected context. |
| `w5/verdict/metrics.json`, its per-pole score and judgment records, manifests | Named in the public extraction scripts | Regenerate all original figures, judge agreement, and calibration sensitivities from full precision records. |
| Intervention per-context fall banks and execution manifests | Named in REUSE and INSTRUMENTS verdicts | Independently reaggregate every intervention condition, including the zero-write headline. |

The direct unauthenticated HTTPS request to the documented S3 object
`https://pg-lineage-769063704696.s3.amazonaws.com/results/neurons/w5/verdict/metrics.json`
returned **HTTP 403** on 2026-11-12 UTC (2026-11-11 Pacific). No credentials were
requested or used. This establishes that this direct public route was unavailable,
not that the data cannot be obtained from the author or another authorized route.

The local reproduction preserves the pointer files and their hashes. Supplying
the two small recount files above would support a substantially stronger follow-up
without re-running all 7B-model forward passes. A true model-level replication
also needs the exact corpus/tokenization, checkpoint revisions, code/environment,
and independently rerun judging/interventions. Original private freeze history
cannot be verified from the public snapshot alone.

Sources: [array pointer](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/next/10_recount2/out/arrays.pointer.json),
[birth pointer](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/next/10_recount2/out/births.pointer.json),
[checkpoint export provenance](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/walkthrough/neuron_across_training/README.md).
