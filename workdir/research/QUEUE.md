# Research queue and handoff

**State:** setup only; no active campaign or GPU process assigned.
**Lead:** unassigned. **GPU runner:** unassigned. **Reviewer:** unassigned.
**Canonical repository / GPU lock:** resolve before execution; all worktrees
must use the same canonical `workdir/research/gpu.lock`.

## Budget

| Resource | Authorized cap | Used | Remaining |
| --- | --- | --- | --- |
| GPU hours, including failed/retried attempts | Not set | 0 campaign hours | Not set |
| Campaign deadline | Not set | — | Not set |
| Trial count / confirmation reserve | Not set | 0 | Not set |
| CPU workers / RAM / artifact storage | Not set | Not measured | Not set |

No budget is inferred from the trainer's defaults. The lead must bound each
stage within the authorized campaign before delegating launches.

## Tasks

| Priority | Task | State | Owner | Dependency / next action |
| --- | --- | --- | --- | --- |
| 0 | User's transition from test to validation evaluation | Waiting on user change | User | Verify scoring, RF thresholds, checkpoint selection, preflight, and metadata afterward |
| 1 | Freeze dataset manifest, budgets, seeds, timing, and GPU eligibility | Planned | Unassigned | See protocol readiness gate; investigate 10 available full datasets vs 14 catalog names |
| 2 | Review validation evaluator and artifact/lock instrumentation | Planned | Unassigned | User's evaluator change; targeted correctness and leakage checks |
| 3 | Calibrate resources and run the matched control | Planned | Unassigned | Authorized bounded pilot/stage and ready protocol |
| 4 | Prioritize first research experiments | Planned | Lead | Read ideas, estimate costs, state hypotheses; planning can start before evaluation is ready |

Replace planning tasks with bounded assignments using
[the task template](templates/TASK.md). Each active task needs an owner, immutable
source/workspace, experiment ID, budget, dependency, and expected completion
artifact. Keep CPU work ready while the GPU is occupied; do not queue duplicate
experiments or launch a second GPU worker.

## Current handoff

- Completed: instruction, protocol, report, queue, and template scaffolding.
- Active processes / GPU UUID / runner / lock holder: none assigned by this setup.
- Pending: validation readiness, explicit manifests and resource limits.
- Next action: when a campaign is requested, assign planning and protocol review;
  keep benchmark evaluation pending until the validation transition is verified.
