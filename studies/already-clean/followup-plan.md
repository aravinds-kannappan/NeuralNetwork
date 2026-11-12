# Follow-up: does continuity survive reused examples?

This exploratory follow-up was formulated after reading the initial embedding
run's same-layer contrast (approximately +0.276 cosine). It is not preregistered
confirmation on unseen data. It reuses the pinned embeddings from that run.

Use the same preselected comparison: checkpoint 13 (step 560,000) versus final,
and the same random subset of the exported poles.

1. Remove every exactly repeated full context from **both** checkpoint sets.
   Compare the resulting centroids. Exclude pairs with fewer than three contexts
   on either side. Report all exclusions and retained-context counts.
2. For each eligible same-layer negative-control neuron, remove any context that
   repeats the retained query context. Match the number of target contexts to the
   same-neuron target set by 200 fixed-seed subsamples without replacement; exclude
   donors with too few candidates. Keep the retained query set fixed.
3. Measure both normalized-centroid cosine and mean cosine over all cross-context
   pairs. The latter checks whether normalization of a diffuse centroid drives
   the comparison. Report the negative-control subtraction for each metric.
4. Repeat on prefix-only, marked-position-masked, and second-encoder embeddings.
   Reuse the same full-text exclusion rule in every variant.
5. A stricter sensitivity removes every context whose normalized marked token
   appears in the other checkpoint's set. Apply the same no-shared-marked-token
   rule to negative controls and match counts. This can leave few eligible poles;
   report that attrition rather than silently replacing the sample.
6. Keep the initial neuron bootstrap and add a layer-block bootstrap, because
   neurons can share negative-control donors and layer-specific properties. Use
   10,000 resamples. Report the number of layers; do not treat this as new model
   seeds or a cure for selection bias.
7. The shared `[MASK]` token increased raw similarity even for negative controls
   in the parent. Encode a version that deletes the marked token (a space joins
   the prefix and suffix) to check whether the mask itself creates a common
   embedding component. Run the original control contrast and the disjoint-text
   contrast on this version. This sensitivity is chosen after seeing masking's
   effect and must be labeled exploratory.

Any positive result supports continuity in the **semantics of selected text**
beyond exact example reuse. It still does not establish persistent causal roles,
full-population neuron identity, or the original recursive-generalization thesis.
