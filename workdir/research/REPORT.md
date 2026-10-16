# Research report

**Campaign status:** setup only; validation transition pending.
**Active frozen protocol:** none. See [PROTOCOL.md](PROTOCOL.md).
**Campaign run count / GPU cost:** no runs registered / no campaign compute used.
Existing repository artifacts have not been assessed as campaign results.

## Evaluation context

- Full/tiny dataset manifests and hashes: pending.
- RF-50+NFA validation baseline artifact, hash, and thresholds: pending.
- Model/budget tiers, evaluation schedule, practical improvement thresholds: pending.
- Screening seed: 11. Confirmation seeds: 11, 22, 33.
- Current comparable control and best result: none.
- Final-test exposure history: not audited; historical code evaluates test scores.
- Remaining campaign budget and deadline: not set; see [QUEUE.md](QUEUE.md).

Do not insert historical test scores into the validation ranking. Each score
must link to an artifact with the same declared protocol, suite, and budget.

## Ranked candidates for scaling

No candidates have been evaluated under a frozen validation protocol.

The lead must sort populated tables by supported scaling value: full score,
full RF-threshold speed, tiny score, then tiny RF-threshold speed. Use a separate
table for each protocol/model/budget tier. Include the matched control in each
table and link every experiment name to its README, which links all run artifacts.

| Rank / priority band | Experiment / short change | Evidence / seeds | Full val score / delta vs control | Full RF crossing: updates / active wall | Tiny val score / delta vs control | Tiny RF crossing: updates / active wall | Cost / parameters | Decision / why scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Report mean ± standard deviation for confirmed scores. Threshold cells include
attainment counts and link per-seed outcomes; use `not reached by <budget>` for
censored runs. Use `unavailable` when the required timing was not measured.
Name selected checkpoints in the README. Report terminal scores and both suites
at the proposed scaling checkpoint there, including regressions and tradeoffs.

## Negative, inconclusive, and invalid results

| Experiment / README | Status / evidence | Observed result | Reason / lesson | Next action |
| --- | --- | --- | --- | --- |

## Scaling decisions

None yet. For each recommendation record the configuration/checkpoint, matched
controls, confirmation and independent-review evidence, ablations completed,
unresolved risks, proposed bigger-model budget, and expected information gained.

## Decision and exposure log

| Date (UTC) | Decision or exposure | Evidence / artifact links | Consequence |
| --- | --- | --- | --- |

Record protocol changes, stopping decisions, unexpected test exposures, and
changes of direction here. All attempts remain in [runs.jsonl](runs.jsonl).
