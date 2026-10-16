# Instructions for research subagents

Read your assignment, the experiment README, and [PROTOCOL.md](PROTOCOL.md).
Follow the assigned role; route research priorities and scope changes to the
lead. Do not launch extra trials or expand the search budget on your own.

## Before starting

- Confirm the experiment ID, hypothesis, matched control, protocol version,
  seeds, resource budget, owned files/workspace, and expected artifacts.
- Check that the task is not already running or completed. Preserve unrelated
  user changes and do not edit another agent's workspace or a running snapshot.
- If implementing, create or update the experiment's `README.md` using
  [the template](templates/EXPERIMENT_README.md). Record the planned method and
  acceptance checks before examining its results.
- If an assignment is incomplete, ask the lead for the missing part while
  continuing independent work that is already authorized.

## Implementation and checks

Implement only the assigned change and necessary checks. Keep controls available
through explicit configs or flags. Document equations or pseudocode when they
help explain the hypothesis, and identify exactly which code/config differs.
Make combinations independently ablatable before claiming component effects.

Run targeted correctness checks with synthetic data first. Tests should catch
meaningful failures: label leakage, invalid shapes, broken graph operations,
incorrect task generation, resume errors, or changed numerical behavior for an
execution optimization. Do not expose benchmark test results during debugging.

Provide the exact source snapshot or recoverable commit plus patch, including
relevant untracked files, and environment/configuration details. Notify the lead
of any evaluator, metric, preprocessing, split, or budget change before running
a comparison; such changes may invalidate the existing protocol.

## Execution

Only the designated runner may use a GPU, even for a short test. Other subagents
submit commands and expected costs to the runner through the lead. The runner
must apply the shared lock, eligibility checks, and one-GPU rule in the protocol.
Do not reserve hardware or kill unrelated processes.

Before launching, verify the protocol readiness gate, fixed run manifest,
registered attempt ID, source/config match, available budget, and output path.
Do not overwrite an existing run. Capture stdout/stderr, metrics, timings,
resource use, checkpoint paths, and final execution status. Report launches and
state changes promptly so the lead can update the registry and queue.

On OOM, error, or preemption, preserve logs and partial artifacts, state the
failure, and propose a fix or exact-state resume. Do not silently reduce the
evaluation suite, switch seeds, change hyperparameters, or restart from scratch
and present it as uninterrupted training. A retry needs a separate attempt record.

## README and final handoff

Every experiment folder must have a detailed README covering its hypothesis,
method, controls, reproducible commands, configs, seeds, budgets, all run
outcomes, artifact links, limitations, and proposed next step. Distinguish
planned work from executed work. Include failed and negative attempts.

Send the lead a concise handoff with:

1. Experiment/task IDs, status, source revision, changed files, and README path.
2. Commands and checks actually run, their outcomes, and any unrun checks.
3. Run/attempt IDs and paths to raw metrics, reports, logs, and checkpoints.
4. Full/tiny validation scores, threshold attainment, update counts, timing
   definitions, resource cost, and uncertainty, when applicable.
5. Deviations, failures, leakage/comparability concerns, active processes, and
   a recommended next action supported by the evidence.

Do not directly edit the lead's report, queue, or registry unless ownership is
explicitly reassigned. Never invent results, hide a failed seed, or present an
unverified timing comparison as a confirmed speedup.
