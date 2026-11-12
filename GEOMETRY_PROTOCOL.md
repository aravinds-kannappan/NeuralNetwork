# Frozen Gemma geometry in learned robot control

This research branch replaces the earlier scripted fault demonstration. It asks
whether pretrained weight structure helps a newly learned policy, and whether
the advantage survives matrix-scale and singular-spectrum controls.

## Scope fixed before measurement
- OGBench cube-double task1 physics, with a documented goal-jitter extension.
  Training and ID evaluation: target x/y offsets uniform ±3 cm. Shift evaluation:
  target x offset has magnitude 6–10 cm with random sign; y stays within ±3 cm.
- Expert state/action demonstrations come from the packaged Markov oracle.
  Training execution has small continuous-action noise; labels are clean expert
  commands at the resulting states. Validation uses separate unperturbed episodes.
- Train/validation/test episode seeds are disjoint. Every method sees identical
  demonstrations and paired evaluation resets. No test outcome selects a checkpoint.
- Input: 21 explicit current-state numbers, including joint state, effector pose,
  object pose, trigonometric yaw, gripper opening/contact and target position.
  Train-set feature means/stds only, with std floor 0.05.
- Behavior cloning: continuous-action MSE plus 0.25 times binary cross entropy for
  gripper direction. Identical AdamW steps, batch size, learning rate, clipping and
  validation-selection rule across variants. Compute cost is measured, not matched.

## Models and interpretations
- MLP: 155,834 trainable parameters, two GELU hidden layers of width 381.
- Frozen public Gemma 4 31B L26: train only a linear encoder, affine LayerNorm and
  linear action head (155,909 trainable parameters). Freeze the pretrained block.
- Matched random: replace each frozen linear matrix with Gaussian entries matching
  its mean/std; retain the same pretrained normalization vectors and layer scalar.
- Scrambled: independently permute rows and columns of each frozen linear matrix.
  This preserves each matrix's singular values exactly while disrupting coordinated
  feature directions between matrices. Keep normalization vectors and layer scalar.
- Input is ONE state token. With no past cache, Q/K and attention patterns cannot
  affect the attention output. This tests a frozen nonlinear state transformation,
  not token-copying or sequential attention. The paper's exact robot tokenization
  could not be verified because its linked code repository is unavailable.
- Public weights: google/gemma-4-31B, pinned revision
  5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89. The public configuration differs from
  dimensions stated in the paper; this is not an exact replication of its checkpoint.
  This model is multimodal; no text-only-pretraining claim is made.
- Abay's original trained interface, English-statistics matcher and IQL training are
  not reproduced. Positive results would support a narrower frozen-weight transfer
  claim; null results cannot refute his IQL result or other model/scale regimes.

## Baseline and comparison
First establish learned MLP control and expert success on both goal distributions.
Then freeze the comparison budget and run at least three paired training seeds.
Report per-training-seed outcomes, not just pooled episode confidence intervals.
Select the checkpoint with minimum validation imitation loss, then evaluate it once
on held-out resets. Log all checkpoint validation values, failures and timing.
Save actual learned policy adapters and trajectories, not just a schematic demo.

## Additional measurement
For frozen models record signed MLP activation vectors, hidden states and action
predictions on the SAME 128 validation states at checkpoints. The substrate remains
fixed; changes therefore arise from the learned interface and its inputs. Analyze
geometry jointly with held-out control behavior. No scalar-neuron cosine is used.

## Low-data comparison (fixed after baseline, before reading Gemma outcomes)
The MLP baseline with 256 demos was near ceiling (39/40 ID and shifted successes).
To create room for a data-efficiency effect, compare all four modes on 16 expert
episodes, three training seeds (42,1337,2024), and 1000 optimizer steps. Evaluate
60 untouched resets per condition, beginning at 30000/40000. Use the same 32
validation demonstration episodes and predeclared checkpoint-loss selection rule.
The 128 fixed-panel states are uniformly sampled without replacement from all
validation transitions with seed76129. Save their indices. Verify the extracted
block against the official Transformers implementation in BF16, and hash every
frozen tensor before and after training. This is a prospectively fixed exploratory
comparison, not a claim that the original offline-play IQL result is reproduced.
