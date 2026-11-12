# Analysis plan — fixed before new similarity measurements

This study analyzes Abay Bektursun's public snapshot
`de979b861f26a4235e38d0a605952d8d94d6a8e9`, downloaded 2026-11-11.
All new comparisons are exploratory: the original data and headlines were already seen.

1. Recompute the three-model classification counts from exported per-pole test scores;
   recompute Wilson intervals and the registered threshold sensitivity. Recompute the
   parent-test accuracies and unrelated-share statistic from the 1,838 exported group
   records and use the original pole bootstrap. Recompute main-is-oldest from the
   public birth-date CSV, with its original permutation null. Check the neuron viewer's
   lithium counts, activation maxima, and fixed-example birth dates. Reaggregate the
   public intervention records and rerun the original m05 extractor.
2. Encode the 41-pole, 16-checkpoint top-context export with a pinned sentence encoder.
   Use the same encoder for every checkpoint. Compare normalized mean embeddings,
   word TF-IDF cosine, marked-token distributions, and exact exported context overlap.
   Do not fill absent top-k activations with zero: absence means censored, not inactive.
3. Test whether matched-pole similarity from step 560,000 to the final checkpoint is
   higher than a same-layer negative control. Primary sample: the 24 poles in the
   export's random group. Only poles with another distinct underlying neuron in their
   layer can enter this comparison; report exclusions. Compute one contrast per pole,
   average sign pairs at the underlying-neuron level, bootstrap neurons, and permute
   pole identities within layer and sign for an additional conditional null where possible.
   Report effect sizes and intervals, not a pooled test on thousands of context pairs.
4. Inspect all-pole and random-only time curves; treat the 17 outcome-selected poles
   separately. Recompute similarities with the marked token removed, without future
   text, and with a second encoder if feasible. Use no outcome-tuned thresholds.
5. Extend the parent-test uncertainty analysis to bootstrap underlying neurons so
   positive and negative poles stay together. For the fixed-example activation vectors,
   cosine is undefined when either vector is zero; report null, not zero similarity.
6. Report end-point selection, rounded/truncated text, one training run, sparse checkpoint
   spacing, and lack of access to the complete activation banks. Semantic similarity of
   selected contexts is evidence about their text, not proof of shared causal function.

Potential follow-up after these results: distinguish threshold crossing from gradual
drift; examine whether broad semantic continuity survives exact-example turnover;
compare semantic drift with intervention sensitivity. Any data-dependent follow-up
must be marked as such and tested on a new child experiment.

This reproduces computations on published measurements. It does not rerun OLMo's
original forward passes, private judge calls, or interventions. Those stages require
the missing raw banks, original execution environments, and substantial compute.
