# Robot recovery pilot: protocol fixed before measurements

## Scope
Use OGBench 1.2.2 and MuJoCo 3.3.7 on local CPU. The environment is
cube-double-play-singletask-task1-v0: one UR5e arm, two cubes, a single
pick-and-place objective. Use the package's CubeMarkovOracle with privileged
simulator object poses. This is a software coordination and fault-injection
pilot. It does not reproduce Borrowed Geometry's trained policy, use neural
embeddings, train agents, or demonstrate recursive self-improvement.

## Baseline
Establish success of the unmodified scripted oracle on four reset seeds and
check deterministic replay. The adapter exposes the benchmark's target to the
data-generation oracle; it does not alter the benchmark goal or success rule.

## Planned study
Reuse the established controller. For each held-out seed compare identical
initial states, fault time, maximum environment steps and refresh allowance.

Methods:
- single: one controller reading the shared object-state cache.
- takeover: a second copy of the controller takes over after missing primary
  heartbeats or 40 steps without observed goal-distance improvement.
- freshness: same takeover rule, plus refresh a stale cache when its age reaches
  the fixed TTL. Timestamp and refresh availability are observable to all methods.
- periodic: same takeover rule, plus refresh at a fixed interval matching the TTL.
  This separates an adaptive freshness rule from a simple polling alternative.

Faults:
- none: no injected failure.
- primary_dropout: primary stops returning commands after step 10; backup stays live.
- shared_stale: shared object-pose publisher stops after step 10; proprioception and
  an independent explicit refresh interface remain live.
- shared_blind: same publisher failure, but explicit refreshes return no observation.
  This is a common-sensor-failure negative control, not a recoverable software fault.

Backup controllers intentionally share the same oracle algorithm and memory.
This isolates redundancy and memory freshness; it does not test diversity between
learned experts. Both use the same pre-seeded oracle initialization.

## Outcomes
Primary: environment success by the fixed horizon. Report counts and Wilson 95%
intervals over independent reset seeds. Compare policies with paired bootstrap
intervals over reset seeds within each fault; do not count time steps as independent.
Secondary: steps, switches, refresh attempts/successes and final goal error.
No collision metric is claimed: ordinary gripper-object contact is part of the task.

## Cosine diagnostic
Offline only, compare the current true object pose and the cached object pose
inside an explicitly defined engineered state vector: joint positions, end-effector
position, gripper opening/contact, object pose and target position. No whitening
or learned encoder. Count episodes with pose error above 4 cm while whole-vector
cosine remains above 0.99. This checks this representation/metric choice only.
The controller and supervisor must never receive the diagnostic's true stale-pose
error. A separate pose-only cosine is recorded to reveal metric dependence.

## Integrity
Log effective config, package versions, code commit, seed, initial-state hash,
every episode outcome, and summary. Verify repeated baseline runs agree and
every paired condition starts from the same initial state. Use finite-value and
action-bound assertions. Preserve traces for selected seeds. Do not tune on the
held-out test results. Keep failed runs and unexpected results visible.
