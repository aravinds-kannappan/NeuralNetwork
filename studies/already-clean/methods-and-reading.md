# Methods and original research

The purpose is to measure whether the text associated with a neuron retains
semantic continuity as training progresses. It is not to equate a sentence
encoder with OLMo's internal representation or infer causality from text similarity.

## Text embeddings

Sentence-BERT motivates embedding texts independently and comparing them with
cosine similarity. Its authors write: “Instead, we always use cosine-similarity
to compare the similarity between two sentence embeddings.”
[Reimers and Gurevych, 2019](https://www.alphaxiv.org/pdf/1908.10084).
Our concrete encoders are pinned releases of
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
and [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc).
They are related MiniLM-family models, so agreement between them is an encoder
sensitivity check, not fully independent replication across embedding families.

For context embeddings $e_i$ normalized to unit length, the checkpoint summary is
$c_t=\operatorname{normalize}(\frac1{20}\sum_i e_i)$; its cosine to another
checkpoint is $c_t^\top c_s$. Uniform weights are primary. A sensitivity weights
each context by its activation in its pole's direction. We also encode the prefix
through the marked token (exclude later text) and a version replacing the marked
token with `[MASK]`. Masking the marked position does not remove every occurrence
of that word from the surrounding text.

## Controls and inference

Global semantic vectors can hide lexical and structural differences. This motivates
word/bigram TF-IDF cosine, marked-token frequency cosine, exact exported-context
Jaccard overlap, and comparisons against other neurons in the same layer.
This choice is consistent with the multidimensional caution in
[Qiu et al., 2026](https://www.alphaxiv.org/pdf/2611.01549), whose full text was read
for methodological context; their reported experiments were not reproduced here.

The primary new contrast is same-pole cosine from training step 560,000 to the
final step minus its average cosine to other neurons' final summaries in the same
layer. Exclude the opposite sign of the same underlying neuron from negative
controls. Average sign-pair observations within neuron before computing the mean
effect. Resample underlying neurons 10,000 times for percentile intervals.
This gives uncertainty conditional on the exported sample and the observed model
training run. It is not uncertainty across training seeds or a correction for
selection of the viewer examples.

Conditional permutation tests are separate: shuffle final identities within
same-layer strata of the random subset. Enumerate the finite null exactly when
small. Also restrict by pole sign as a sensitivity. Small strata can make these
tests severely underpowered, even when observed cosine contrasts are positive.
Do not substitute hundreds of dependent checkpoint/context pairs for independent
neurons to manufacture small p-values.

## Activation vectors and why CKA is not applied to changing top-k lists

The three fixed-example exports permit actual activation-vector cosine on the
same examples at every checkpoint. Zero vectors have undefined cosine and are
reported as null. These examples were selected at the final checkpoint, have only
four or five contexts, and cannot estimate a whole-model representation.

CKA compares two activation matrices evaluated on the **same examples**; its
invariances are useful when comparing representations across networks.
[Kornblith et al., 2019](https://www.alphaxiv.org/pdf/1905.00414).
Applying it directly to top-20 lists whose examples change across checkpoints
would violate that matching requirement. The missing activation bank is the right
input for a later CKA/RSA study; no unavailable activation values are imputed here.

## Reading selection

Discovery used the question about checkpoint behavior, embeddings and similarity,
plus focused method-title queries. Most relevant retrieved candidates, in order:
Sentence-BERT (1908.10084), Similarity of Neural Network Representations Revisited
(1905.00414), Revisiting LLM Homogeneity Through a Multidimensional Lens
(2611.01549), Relative Representations Enable Portable White-Box Monitoring
(2610.28366), Similarity of Neural Network Representations in Superposition
(2604.00208), and The Development of Sycophancy in Language Models (2610.26655).
Only the first three were read in full and used for methodological claims. No
date filter or popularity ranking was imposed.
