# Shared research protocol

## Readiness and current state

**Status: DRAFT — validation transition pending; no campaign runs authorized by
the creation of these files.**

The user will change evaluation to validation scores. Leave that conversion to
the user unless they subsequently delegate it. Until it is verified, do not
launch benchmark evaluations, including benchmark preflights, that expose test
scores to the research loop. Planning, implementation, and synthetic-data checks
may proceed within the assigned scope.

The current evaluator reads test targets and RF test scores, and the trainer
labels its reports as test-based. Changing a report label alone is insufficient.
Before marking a protocol ready, a subagent must verify:

- Prediction scoring, RF-50+NFA thresholds, best-checkpoint selection, threshold
  crossings, preflight outputs, CLI summaries, and report metadata use validation.
- Validation/test targets never enter model inputs or synthetic-prior fitting.
- The baseline artifact contains valid validation scores for the exact suite
  and split masks; historical test thresholds are not reused.
- The fixed manifests, resource limits, seeds, evaluation grid, and timing
  definitions below are recorded, and targeted evaluator checks pass.

Keep historical test-based results labeled `historical / exploratory test`.
They are not validation records and do not provide an untouched final test.

## Freeze before comparing

Create `protocols/<version>/` and link it here before launching a ranked stage.
Store a machine-readable manifest, human-readable decisions, and artifact hashes.
Do not mutate a frozen version. A correctness fix or intentional protocol change
requires a new version, an explanation, and new comparable controls. Do not change
the primary metric or dataset population in response to an unfavorable result.

| Item | Required decision / initial value |
| --- | --- |
| Frozen protocol version | UNSET |
| Evaluation | GraphLand RL masks; validation scores only during development |
| Full suite | UNSET: explicit dataset names, versions, file and mask hashes |
| Tiny suite | Default families: artnet-views, city-reviews, tolokers-2, twitch-views; each has variants t-0 through t-4; verify and hash all 20 |
| Per-dataset metrics | Preserve the project-defined metric and direction for each dataset |
| Suite aggregation | Equal-weight arithmetic mean of raw dataset scores; higher is better; freeze before running |
| RF-50+NFA | 50 trees, NFA enabled, seed 42, RL masks, exact-suite validation aggregate; artifact path/hash and thresholds UNSET |
| Screening seed | 11; fixed across candidates and controls |
| Confirmation seeds | 11, 22, 33; report all three, including the original seed-11 run if exactly comparable |
| Other randomness | Freeze baseline, dataset/subsampling, preprocessing/PCA, and evaluation RNG settings |
| Model/budget tiers | UNSET: parameter counts, update/task/token budgets, active-time caps, and evaluation grid for each stage |
| Selection rule | Best suite aggregate at one scheduled checkpoint within the fixed budget; earliest checkpoint breaks exact ties |
| Practical improvement | UNSET: minimum score/time/update differences worth confirming, chosen before candidate results |
| Timing | Definitions below; verify available instrumentation before making a time claim |
| Campaign limits | UNSET: total GPU hours, deadline, maximum trials, confirmation reserve, CPU/RAM/disk limits |
| GPU eligibility | UNSET: reservation/eligibility source, sharing policy, and canonical team-lock path |
| Source and environment | Exact source/dependency hashes, commands, software versions, hardware and precision |
| Final-test policy | Frozen shortlist and checkpoints; isolated scorer if available; exposure history recorded |

The lead sets pending research choices within the user's authorized budget;
`UNSET` is not permission for unlimited compute. Do not run a stage until its
applicable fields are fixed. Resource calibration may use its own bounded pilot
protocol and must not masquerade as a ranked experiment.

Dataset discovery is not a benchmark definition: this checkout currently has
10 full-size dataset directories while the catalog recognizes 14 names. Resolve
and explicitly document the intended full-suite population before freezing it.
Identify subsets as subsets; do not silently redefine “full GraphLand” according
to which data happen to be present. Tiny variants share dataset families with
each other and with full graphs; they are not independent replication samples.

## Labels and evaluation integrity

- During benchmark inference, only the declared training/context labels may be
  supplied to the model. Validation labels are used only by the scorer and for
  declared model/configuration selection. Test labels and test-derived metrics
  must not influence research decisions, checkpointing, stopping, or tuning.
- Do not inspect withheld per-example labels, obtain them from another copy of
  the dataset, or use them to design features, synthetic tasks, priors, thresholds,
  normalization, label encodings, or target statistics. Public availability does
  not make test-label use acceptable.
- Synthetic query labels may supervise synthetic pretraining. Benchmark train
  labels may serve as context under the protocol. Learning from validation
  scores is permitted; fitting on validation targets is not.
- All-node features and graph topology may be used under the declared
  transductive setup. Record exactly which unlabeled data preprocessing uses.
  Target-dependent transformations use training/context labels only. Preserve
  declared task metadata without inferring extra information from held-out labels.
- Never encode benchmark identity, paths, node IDs, or label lookups to return
  benchmark-specific answers. Do not tune dataset membership, split masks,
  metric weights, or exclusion rules to improve the aggregate.
- Every required dataset must produce a valid metric. Missing scores, NaNs, OOMs,
  and crashes invalidate a complete-suite claim; never drop them from the mean.
  A partial evaluation may be retained with an explicit partial status.
- Selection uses a suite score at one checkpoint, not a mean of each dataset's
  individually best checkpoint. Full and tiny may select different checkpoints;
  identify both, and report both suites at the chosen scaling checkpoint too.
- Final test evaluation uses a preregistered shortlist and frozen checkpoints.
  Do not select a winner from test scores and resume tuning. Record every test
  exposure. If test feedback affects development, mark that holdout contaminated
  and require a fresh holdout for an unbiased final claim.

## Seeds, controls, and search fairness

Use the fixed seed lists above; never search seeds, keep only favorable seeds,
or change randomness to rescue a poor result. A genuinely failed run may be
retried with the same seed and configuration, retaining both attempts. A code
fix creates a new revision and requires affected controls to be reconsidered.

Separate RNG streams for initialization, task generation, masks, and evaluation
where supported. Preserve task indices and RNG state on resume. Pair seeds and,
when meaningful, synthetic task streams across controls. A changed prior can
change the task stream; document that rather than claiming identical training
data just because the integer seed matches.

Record batch size, accumulation, optimizer updates, synthetic tasks, and consumed
nodes/tokens. “Iteration” means one optimizer update. Report data exposure when
it changes: fewer updates with a bigger batch is not automatically better sample
efficiency. Freeze evaluation frequency in updates, not a manipulable epoch label.

Compare under the same model/budget tier and common evaluated budget points.
Report terminal scores as well as selected best scores. Account for evaluation,
preprocessing, compilation, and all hyperparameter-search attempts in total cost.
Early stopping/futility rules must be stated before a stage; a lead may terminate
for resource or correctness reasons, but must disclose the stop and not relabel
the shortened run as a completed equal-budget comparison.

## Thresholds and time

For each suite, the threshold is its frozen RF-50+NFA validation aggregate.
Record the first scheduled evaluation with score greater than or equal to it,
the cumulative optimizer updates at that point, and elapsed times through the
completion of that suite's evaluation. The evaluation grid and suite order must
be the same for compared runs. Record the preceding below-threshold evaluation;
the exact crossing between evaluations is unknown. Do not interpolate a faster
crossing or add candidate-specific evaluations to improve the claim.

Preflight is a correctness/memory check, not a trained-model threshold crossing.
A threshold not reached is `not reached by <budget>` with null crossing fields,
never zero. Retain censored runs when comparing seeds; report attainment counts
and each seed's outcome. Do not hide failures by averaging only successful hits.

Record these clocks separately:

1. **Training seconds:** optimizer/training work plus waiting for prior tasks,
   excluding evaluation. State the exact instrumentation boundaries.
2. **Active wall seconds:** process start through the measured result, including
   setup, data preparation, compilation, preflight, evaluation, and checkpoint
   work before that result. Sum active segments for resumed runs.
3. **Campaign elapsed time and cost:** include queue waits, interruption gaps,
   failed/retried runs, preparation work, and hyperparameter searches separately.

Current trainer wall counters start at training and exclude preflight and resume
downtime. Retain their original names/definitions. They cannot be relabeled as
the active wall clock above. Delegate additional instrumentation if needed;
until verified, mark the corresponding time claim unavailable or provisional.
Cold/warm compilation and cache conditions must be recorded and matched.

Confirm timing improvements on comparable hardware with comparable contention,
using a nearby control run. Log GPU UUID/model, memory use, utilization, and
co-tenant load without interfering with other users. Shared-GPU measurements
remain provisional until the timing comparison is sufficiently controlled.

## One GPU for the whole team

- Appoint exactly one runner. All GPU work, including smoke checks, inference,
  preflights, profiling, and resumed runs, goes through this runner.
- Use one canonical OS advisory lock at the repository's
  `workdir/research/gpu.lock`, shared across all agents and worktrees. Acquire it
  atomically (for example with `flock`) and hold it for the entire GPU process
  tree's lifetime. File existence alone is not a lock. This mutex coordinates
  this team; it does not reserve a GPU against other users.
- Prefer a free, unreserved GPU. If none is free, choose the least busy eligible
  GPU only when sharing is allowed and enough memory is available, or wait.
  A low utilization reading does not establish that a GPU is unreserved. Check
  the host's reservation/eligibility source; if eligibility is unknown, wait
  and report the missing information rather than guessing.
- Never book/reserve GPUs, cancel bookings, kill other users' processes, or
  manipulate other workloads. Observe reservations that begin during a run.
- Pin the run to one physical GPU using `CUDA_VISIBLE_DEVICES`; record its UUID.
  Do not use multi-GPU/distributed execution or a second GPU for evaluation.
  Before switching devices or replacing a runner, verify the previous process
  tree has stopped and the previous GPU is released.
- Checkpoint for preemption when feasible. A killed or preempted run is
  `interrupted`, not a bad score to discard. Resume the exact state or create a
  linked retry; retain consumed compute and the reason for interruption.
- Poll with bounded waits and backoff, keeping each wait at most 60 seconds
  while an interactive agent is active. Avoid busy loops and duplicate launches.
  If the campaign deadline or budget is exhausted, stop and leave a handoff.

## Provenance, registry, and review

Before launch, write `runs/<run-id>/manifest.json` with experiment/run/attempt IDs,
protocol version/hash, stage, hypothesis/control IDs, exact config and command,
seed, budgets, evaluation schedule, intended source hash, environment, and
resume parent if any. Add an event to [runs.jsonl](runs.jsonl) before execution.
The runner verifies the executed config/source match the declaration.

The registry is append-only JSON Lines, with one event per state change. Each
event includes `timestamp_utc`, `experiment_id`, `run_id`, `attempt_id`, `event`,
`protocol_id`, `stage`, `seed`, `config_path`, `source_hash`, `artifact_dir`,
`parent_run_id`, and `reason` (null where genuinely inapplicable). Events include
`planned`, `started`, `completed`, `failed`, `interrupted`, `invalidated`, and
`reviewed`. Corrections are new events; never erase unfavorable history.
Use unique IDs before launch; the runner may preallocate the trainer directory
or register a planned ID and record its mapping to the trainer's unique ID.

Preserve exact source, including dirty and untracked code that affects the run,
dependency lockfiles, configs, data/split hashes, logs, per-dataset metrics,
learning curves, and checkpoint hashes. A Git commit alone is insufficient for
a dirty checkout. The trainer currently snapshots only four Python files;
extend the saved provenance to other changed dependencies when necessary.
Run from an immutable copy; do not patch a running experiment's code.

An independent review must check aggregation, checkpoint selection, threshold
crossings, seed completeness, budget comparability, and consistency with raw
artifacts. Delegate targeted leakage checks: replacing withheld targets must not
change model inputs or predictions; the evaluator may use targets only to score
fixed predictions. For execution optimizations, check appropriate numerical
equivalence; for scientific changes, test the intended behavior, not equivalence
to the old model. Do not fabricate metrics or mark unrun checks as passed.

Instructions, hashes, and reviews provide auditability, not hard isolation.
For a protected final holdout, use a trusted scorer outside the agents' writable
workspace with test labels unavailable to the research processes, a frozen
scoring harness, and logged/limited submissions. Do not claim this protection
exists while all agents can read the same labels or edit the same scorer.

## Ranking and evidence

Maintain the report's main table in the four priority bands from the lead
instructions: full score, full threshold speed, tiny score, tiny threshold speed.
Assign each candidate its strongest supported band. Within a band, prefer
confirmed practically meaningful gains over uncertain one-seed records, then
effect size and cost. Explain exceptions and close ties; do not manufacture a
single weighted score after seeing the results. Keep score/time tradeoffs visible.

Separate model/budget tiers and protocol versions; never pool incomparable runs
into one ranking. Include the matched control and the best comparable result.
For confirmation, report mean and standard deviation across all fixed seeds,
paired differences to the control, per-seed threshold outcomes, attainment counts,
and timing dispersion. With three seeds, uncertainty estimates are limited; do
not claim certainty from a tiny numerical difference. Inspect per-dataset and
family effects rather than treating related tiny variants as independent evidence.

Use evidence labels `screening`, `confirmed`, `inconclusive`, or `invalid`.
Execution status is separate from evidence quality. An interrupted run can be
informative without being a completed comparison. List negative and invalid
results in the report/archive with README links and reasons.
