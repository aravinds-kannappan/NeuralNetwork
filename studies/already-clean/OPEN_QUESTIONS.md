# Open questions and next experiments

All experiments below are proposed, not completed results. The first three address the largest limits of the current study.

## 1. Does continuity survive when the inputs stay fixed?

**Start here · Proposed · needs missing inputs**

Each checkpoint exposes a different top-20 list. The viewer can miss changes outside those selected examples.

**Test:** Give every checkpoint exactly the same text, then record each pole’s responses. Compare whether the response vectors point in similar directions (cosine), whether the same examples stay near the top (rank correlation), and whether larger response patterns agree (CKA/RSA). These last methods compare representations on matched inputs. Include other neurons in the same layer and with the same sign. Set aside whole neurons before choosing thresholds.

**Data needed:** The 4.54 MB activation array, 4.92 MB birth record, and their documented alignment metadata; the larger banks for a broader sample.

**What would weaken the interpretation:** If continuity vanishes on fixed inputs or does not exceed matched controls, the exported top-k selection explains the apparent persistence.

## 2. Can embeddings predict a cluster’s parent under fair controls?

**Priority 2 · Proposed · needs judge inputs**

The original parent test’s control items are easier. Its calibration changes when lexical overlap is matched.

**Test:** Ask an embedding model to rank possible earlier records for each later group. Compare its choices with the original judge on neurons held aside for evaluation. Make true and control comparisons equally easy with respect to layer, sign, shared highlighted tokens, passage length, and distance in training time. Also try a baseline that uses only word overlap. Choose any scoring thresholds on separate development neurons.

**Data needed:** Full judge-visible examples, labels, and source identifiers. A larger random neuron sample would give a more useful exact null.

**What would weaken the interpretation:** If a lexical-only baseline matches the embedding model, or the advantage disappears under matched controls, semantic ancestry is not established.

## 3. Does semantic continuity predict causal reuse?

**Priority 3 · Proposed · needs interventions**

Similar text can activate the same neuron even when its downstream role has changed.

**Test:** Choose new neurons before looking at their examples. Measure text continuity, then deliberately change each neuron’s contribution and record how the model’s predictions change. Ask whether continuity predicts similar intervention effects, after accounting for layer, response size, repeated text, and shared upstream drivers (earlier units feeding the neuron). Evaluate on held-out neurons; use independent training runs if available.

**Data needed:** Aligned per-context intervention measurements and activation banks, or fresh model runs for an independent sample.

**What would weaken the interpretation:** Stable semantic similarity with unrelated intervention effects would reject the stronger claim that text continuity tracks the same mechanism.

## 4. Is a cluster’s birth a real transition or a threshold crossing?

**Extension · Proposed · needs denser checkpoints**

The released examples rise gradually, and a two-checkpoint rule converts a continuous trajectory into one discrete date.

**Test:** Measure responses at more closely spaced training checkpoints around the apparent onset. Compare a smoothly changing response curve with a model that permits a sudden change in its behavior. Test which predicts unseen checkpoints better. Repeat the dating under different response thresholds and requirements for persistence. Compare real clusters with random groups matched for size and top rank.

**Data needed:** Denser aligned activation measurements and a preregistered transition criterion.

**What would weaken the interpretation:** If smooth trajectories predict held-out checkpoints equally well and estimated dates move with the threshold, an abrupt mechanistic birth is unsupported.

## 5. How much continuity comes from the embedding model?

**Feasible next · Proposed · public contexts suffice for a first pass**

Inserting a shared mask raises both matched and mismatched cosine. The two encoders used here also belong to the same MiniLM family.

**Test:** Use a sentence encoder from a different model family on the same passages. Separately estimate the typical embedding direction on background text, then test whether removing common directions changes the result. Those transformations must be fitted without using the evaluation neurons. Repeat the repeated-context removal and compare with simple word-based measures.

**Data needed:** The released context export, another locally runnable encoder, and an independent background text sample.

**What would weaken the interpretation:** If the controlled contrast disappears across encoder families or after independently fitted centering, the present semantic conclusion is encoder-dependent.

## 6. Does a stable average hide several changing jobs?

**Feasible next · Proposed · saved embeddings available**

A centroid compresses twenty examples into one direction. Different clusters can trade places while that direction stays similar.

**Test:** Look inside each set instead of relying on its average. Measure how spread out its passage vectors are and match individual passages or topic clusters across checkpoints. A cluster-level transport analysis asks how much matching is possible when the sets have different mixtures. Inspect cases where the centroid stays similar but the clusters change. Decide the outcomes in advance and account for running multiple tests.

**Data needed:** The saved embeddings and contexts; a larger unbiased sample for a confirmatory claim.

**What would weaken the interpretation:** If a stable centroid coexists with unstable clusters, centroid similarity cannot be read as persistence of one coherent job.

## 7. Does the effect generalize across signs, seeds, and models?

**Longer horizon · Proposed · requires new compute**

The primary comparison has thirteen eligible neurons; sign matching makes the exact test too small. All checkpoint observations come from one OLMo training run.

**Test:** Choose neurons before inspecting their responses, with enough positive and negative poles in each layer to form controls. Repeat the same analysis on independently trained models, including different training seeds (different random choices during training) and model families. Treat each training run or model as a replication; keep neurons nested within it instead of treating every text as independent evidence.

**Data needed:** Independent model checkpoints, a shared evaluation corpus, and enough within-layer controls for a meaningful randomization test.

**What would weaken the interpretation:** If the direction or size changes substantially across seeds or models, any conclusion must remain local to the original run and sample.
