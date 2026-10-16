# Repository agent instructions

## Agentic research

When asked to lead or contribute to an agentic research campaign, read
[RESEARCH_INSTRUCTIONS.md](RESEARCH_INSTRUCTIONS.md) and
[the shared protocol](workdir/research/PROTOCOL.md). Assigned subagents must also
read [their instructions](workdir/research/SUBAGENT_INSTRUCTIONS.md).

These research roles apply to research campaigns, not to unrelated repository
maintenance. Merely reading these files does not authorize starting a campaign.

- The lead prioritizes, delegates, reviews, and maintains the report and queue.
  Subagents implement, test, profile, and launch experiments.
- The entire team may use at most one physical GPU at a time, including evaluation
  and profiling. Only the designated runner may launch GPU work.
- Never select seeds, checkpoints, hyperparameters, or ideas using test results.
  Preserve all attempts, including negative, failed, and interrupted runs.
- Do not edit `research_ideas.md` unless the user asks. It is a source of ideas,
  not an exhaustive list or a constraint on the research space.
- Preserve unrelated user changes. Run from an immutable source snapshot;
  do not modify code underneath a running experiment.

The validation transition is pending the user's evaluator changes. Read the
readiness gate in the protocol before any benchmark evaluation or ranked run.
