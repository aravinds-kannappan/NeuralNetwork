# Verification record for Abay research review

Reviewed 2026-11-11. This review did not run a model, train an adapter, or launch an experiment. It inspected published methods/code and performed arithmetic on a published visualization export.

## Sources and scope

- Borrowed Geometry: [arXiv:2605.00333v2, original paper](https://www.alphaxiv.org/pdf/2605.00333v2), dated 18 May 2026. The paper reports the experimental results; I did not independently reproduce them. Its original figures are reproduced under CC BY 4.0 with attribution to Abay Bektursun.
- Neuron study: [repository snapshot de979b861f26a4235e38d0a605952d8d94d6a8e9](https://github.com/abaybektursun/already-clean-audit/tree/de979b861f26a4235e38d0a605952d8d94d6a8e9).
- [Viewer extraction method](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/walkthrough/neuron_across_training/extract.py).
- [Harvest code](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/infra/pg/neurons_harvest.py).
- [Birth and parent-test methods, section 2.4](https://github.com/abaybektursun/already-clean-audit/blob/de979b861f26a4235e38d0a605952d8d94d6a8e9/SPEC_D1.md#24-clusters-births-parents-q2-readout).
- [OLMo3 forward implementation in Transformers 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/olmo3/modeling_olmo3.py#L220-L276). This verifies the architecture definition, not the exact installed version used in the private original run.

## Export checks actually performed

Downloaded the Git LFS object for the viewer from the pinned repository snapshot. SHA-256 is `3df48c264043926b5da3813e2d76aa385004652535f51b20d474185fcf7a4b1a`, matching its LFS pointer; 2,666,255 bytes. Its metadata points to public commit `4247cc20113032ec749b6c32a457ce475a3276f9`. Replacing only that metadata commit with the extraction script's original `2e43474ae1bbfa353e222993cb706283a47bee7b` reproduces the README's documented `c7d820461e0e26220ca0f89266fd6b73b6bffa4f5e079cd0263b6720a825f740` hash. The build contract documents that metadata rewrite; it is not evidence of a measurement discrepancy.

The export contains 41 poles, 16 checkpoints, and reports 20,560,068 scored token positions. Counts of the exact token string ` lithium` in the top 20 were recomputed. The peak-to-final maximum ratio is approximately 592.6 for L03N6372+. All 24 poles in the export's random group exceed a ratio of 10; the group median is approximately 525.7. These are selected-pole, four-significant-digit exported values, not full-population measurements. The random group excludes the 17 positive/negative poles selected by H_MIG, and the highlight token is selected using all checkpoints.

Early top contexts contain ` licensed` and ` Licensed`; later top contexts include `/licenses`, ` OpenSSL`, and `/MIT`. This lexical recurrence motivates testing a persistent or re-emerging response component. It does not establish that these contexts have the same causal function. The final top 20 also include ` parchment` and `ampp`, so they do not uniquely establish a licensing role.

## Files

- [Arithmetic results](export-checks.json)
- [Arithmetic-check script](check_export.py)
- [Exact downloaded viewer export](https://raw.githubusercontent.com/abaybektursun/already-clean-audit/de979b861f26a4235e38d0a605952d8d94d6a8e9/studies/already-clean/walkthrough/neuron_across_training/data.js)
- [Original paper PDF](https://arxiv.org/pdf/2605.00333v2)
- [Original figure 2 crop](borrowed-geometry-figure2.png)
- [Original figure 4 crop](borrowed-geometry-figure4.png)
- [Figure extraction script](extract_original_figures.py)

## Unverified

No access to the original checkpoint-by-token activation banks, their execution manifests, or the private freeze history. The public export does not contain the complete fixed-example activation matrix or the final top-50 list. I could verify the extraction and harvest code's intended controls, but not independently verify their execution on every original run. I did not locate accessible Borrowed Geometry training/ablation code, full logs, or saved policies for the May paper. Claims about those runs remain author-reported. I did not treat the current public code snapshot as proof of the exact code originally executed.
