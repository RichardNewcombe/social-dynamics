# Experiment 5: Summary

## Research Question

How does organizational structure affect a team's ability to solve hard problems — and at what cost?

Specifically: when a group of agents with different capabilities must collectively navigate a complex solution space with deceptive local optima, what organizational configurations (team coupling, role distribution, social learning dynamics) produce the best outcomes?

## What We Added to the Simulator

Two extensions to the core simulation, both opt-in and backward-compatible:

**Per-particle roles.** Each agent now has four capability scalars drawn from configurable distributions: gradient sensing accuracy (how well they read the problem), step magnitude (how aggressively they act on information), social influence weight (how much the team listens to them), and solution visibility (a rare ability to sense the global optimum). These interact through team aggregation — agents share noisy observations with their social neighbors, weighted by influence, and the team moves based on the collective signal.

**Separate identity and strategy vectors.** The existing preference vector (which drives social grouping and team formation) is now decoupled from a new strategy vector (which represents the team's position in the solution space). A coupling parameter controls the overlap: at 1.0 they're identical (what you believe = what you do), at 0.0 they're independent (team identity is separate from problem-solving approach). The social neighbor graph from preference space determines information flow in strategy space — your team is who you share observations with.

## What We've Found So Far

**Decoupling identity from strategy helps.** When team membership is independent of strategic position (low coupling), agents can form stable social groups while still exploring diverse solutions. High coupling locks teams onto whatever local optimum they land on first.

**A few well-placed rare capabilities outperform uniformly capable teams.** A small fraction of agents (5–10%) with strong global sensing, embedded in a flexible organization, produces better cost-adjusted outcomes than expensive teams where everyone has moderate capabilities.

**Social conformity is harmful on complex problems.** It accelerates convergence — onto the wrong answer. The nearest local optimum is usually not the best one.

**Social differentiation avoids bad answers but can't find good ones.** Pushing agents apart prevents premature convergence but also prevents the coordination needed to reach the global optimum.

## Gaps

**The solution landscape needs more structure.** Agents with global sensing still solve too easily by ignoring local information entirely. The landscape should force routing decisions where local knowledge matters.

**Cost is tracked but not binding.** There's no budget constraint, so expensive configurations always win if given enough time. A finite budget would force genuine tradeoffs in team composition.

**Team formation is weak.** The social dynamics aren't producing visible clusters in most conditions. Without coherent teams, the "team structure drives problem-solving" hypothesis is only partially testable.

**Role distribution is underexplored.** We've tested a few configurations but haven't mapped the space. Key questions: what's the minimum fraction of globally-sensing agents needed? Is there a phase transition? How does the answer change with problem complexity?

**Coupling is static.** Real organizations adapt their structure over time. Dynamic coupling — starting flexible and specializing as the team learns — may outperform any fixed configuration.
