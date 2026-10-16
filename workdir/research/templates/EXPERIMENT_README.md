# exp-NNN-short-name: descriptive title

> Copy to `workdir/experiments/exp-NNN-short-name/README.md` before implementation
> or launch. Replace placeholders; mark unknowns explicitly. Links in the copied
> README must resolve relative to its experiment directory.

## Status and hypothesis

- Status: planned / implementing / queued / running / complete / stopped.
- Evidence: screening / confirmed / inconclusive / invalid / not evaluated.
- Owner, reviewer, and dates (UTC): UNSET.
- Idea reference and relevant literature: UNSET; new ideas are permitted.
- Hypothesis and mechanism: UNSET.
- Why this could help a longer run with a bigger model: UNSET.
- Predicted effects on full/tiny score, threshold speed, and memory: UNSET.
- What outcome would weaken or falsify the hypothesis: UNSET.

## Method and controls

Explain the method in enough detail to reproduce it. Include equations or
pseudocode when useful; identify changes to architecture, prior, training, or
execution. State which benchmark information the method can access.

- Frozen protocol link/version/hash: UNSET.
- Model/budget tier and stage: UNSET.
- Original control and immediate parent experiment/run IDs: UNSET.
- Exact changed files, flags, parameter count, and expected extra cost: UNSET.
- Controlled variables, including data/task streams and evaluation grid: UNSET.
- Ablations: UNSET. For an A+B claim, link base, A, B, and A+B comparisons.
- Bounded hyperparameter space and total search budget: UNSET.

## Plan fixed before launch

- Config paths/hashes and source snapshot/hash: UNSET.
- Screening and confirmation seeds: 11; then 11, 22, 33.
- Maximum optimizer updates, tasks/nodes/tokens, active time, and trial count: UNSET.
- Evaluation grid, suite order, and selected-checkpoint rule: UNSET.
- RF validation baseline artifact/hash and full/tiny thresholds: UNSET.
- Success, futility, failure, promotion, and stopping rules: UNSET.
- Required correctness/leakage checks and independent review: UNSET.
- GPU/CPU/RAM/disk requirements and expected duration: UNSET.

If the plan changes after observing results, retain the original plan and record
the reason, time, affected run IDs, and new stage/configuration.

## Reproduction and artifacts

Provide exact setup, implementation checks, training, resume, and evaluation
commands. State the working directory, environment/lockfile versions, immutable
source location, and which commands actually ran. GPU commands are executed
only by the designated runner under the shared lock.

Use the trainer's experiment-directory option to put outputs under this folder's
`runs/`. Link each run's manifest, source, config, log, raw metrics, report, and
checkpoint hashes. Explain any extra files required beyond the trainer's default
four-file source snapshot. Never put secret credentials in artifacts.

| Run / attempt / artifact links | Stage / config / seed | Parent / resume | Execution status | Protocol / source | GPU / contention | Budget used |
| --- | --- | --- | --- | --- | --- | --- |

## Results

No results yet. Never fill a score from an expectation or an unexecuted command.

| Run / seed | Full val: selected / terminal | Full threshold: updates / training s / active wall s | Tiny val: selected / terminal | Tiny threshold: updates / training s / active wall s | Checkpoint IDs | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |

Include the validation curves, per-dataset scores, paired control differences,
and all seeds. State threshold attainment counts and censored budgets. Separate
training time, active wall time, queue delay, and total search cost. If available
timers use different boundaries, label them and do not substitute them silently.

Report mean and standard deviation for confirmation scores, per-seed crossings,
and timing dispersion. Full and tiny selected checkpoints may differ: name both
and report both suites at the proposed scaling checkpoint. Explain shared-GPU
timing limitations and dataset-family correlations.

## Failures, deviations, and review

| Run / check | Failure or deviation | Evidence | Impact / remedy |
| --- | --- | --- | --- |

Record every OOM, crash, interruption, retry, excluded/invalid comparison, and
unexpected test exposure. Explain affected conclusions; do not erase attempts.
List actual reviewer checks with artifact links and results. Keep unrun checks
explicitly pending.

## Interpretation and next decision

- Supported result and uncertainty: UNSET.
- Priority band and reason: full score / full speed / tiny score / tiny speed / none.
- Regressions, tradeoffs, and unresolved confounders: UNSET.
- Attribution/stacking evidence and missing ablations: UNSET.
- Decision: stop / revise / confirm / ablate / stack / scale candidate.
- Exact proposed next experiment, expected information, and bounded cost: UNSET.

Link the campaign report and related experiments using valid relative paths.
