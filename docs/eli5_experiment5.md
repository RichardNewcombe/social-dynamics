# Collective Problem-Solving with Heterogeneous Roles

## Research Question

How does organizational structure affect a team's ability to solve hard problems — and at what cost?

Specifically: when a group of agents with different capabilities must collectively navigate a complex solution space with deceptive local optima, what organizational configurations (team coupling, role distribution, social learning dynamics) produce the best outcomes?

## What Changed in the Simulator

Two extensions to the core simulation, both opt-in and backward-compatible with existing experiments 1–3.

**Per-particle roles.** Each agent has four capability scalars drawn from configurable distributions:

- **Researcher** — accuracy of local gradient sensing. Good researchers read the problem well; poor ones get noisy signals. Nobody knows who's accurate — it only emerges when the team aggregates observations and the noise cancels out.
- **Engineer** — magnitude of movement along the chosen direction. Engineers don't pick the direction; they amplify whatever the team decides.
- **Leader** — social influence weight in team aggregation. When the team averages its members' observations, leaders' votes count more — whether they're right or not.
- **Visionary** — a rare ability to faintly sense the global optimum. Subject to a rarity constraint: only a configurable fraction of agents (5–10%) receive any visionary ability at all.

These interact through team aggregation: agents share noisy observations with their social neighbors, weighted by leader influence, and the team moves based on the collective signal.

**Separate identity and strategy vectors.** The existing preference vector (which drives social grouping and team formation) is now decoupled from a new strategy vector (which represents the team's position in the solution space). A coupling parameter controls the overlap:

- **1.0** — specialist organization: identity and strategy are the same thing
- **0.0** — generalist organization: team identity is independent of problem-solving approach
- **0.5** — typical organization: partial overlap

The social neighbor graph from preference space determines information flow in strategy space — your team is who you share observations with.

## What We've Found So Far

**Decoupling identity from strategy helps.** When team membership is independent of strategic position (low coupling), agents can form stable social groups while still exploring diverse solutions. High coupling locks teams onto whatever local optimum they land on first.

**A few rare visionaries outperform uniformly capable teams.** A small fraction of agents (5–10%) with strong global sensing, embedded in a flexible organization, produces better cost-adjusted outcomes than expensive teams where everyone has moderate capabilities.

**Social conformity is harmful on complex problems.** It accelerates convergence — onto the wrong answer. The nearest local optimum is usually not the best one.

**Social differentiation avoids bad answers but can't find good ones.** Pushing agents apart prevents premature convergence but also prevents the coordination needed to reach the global optimum.

## Gaps

**The solution landscape needs more structure.** Visionaries still solve too easily by ignoring local information entirely. The landscape should force routing decisions where local knowledge matters.

**Cost is tracked but not binding.** There's no budget constraint, so expensive configurations always win if given enough time. A finite budget would force genuine tradeoffs in team composition.

**Team formation is weak.** The social dynamics aren't producing visible clusters in most conditions. Without coherent teams, the "team structure drives problem-solving" hypothesis is only partially testable.

**Role distribution is underexplored.** We've tested a few configurations but haven't mapped the space. Key questions: what's the minimum fraction of visionaries needed? Is there a phase transition? How does the answer change with problem complexity?

**Coupling is static.** Real organizations adapt their structure over time. Dynamic coupling — starting flexible and specializing as the team learns — may outperform any fixed configuration.

---

## How to Run

### Headless Experiments

All experiments run from the repo root. No GPU required.

```bash
# Experiment 1 — Social learning rate sweep
python -m experiments.exp1_hidden_target --trials 3 --max-steps 20000

# Experiment 2 — Diversity maintenance
python -m experiments.exp2_niche_coverage --trials 3 --max-steps 10000

# Experiment 3 — Institutional inertia
python -m experiments.exp3_ghost_colony --trials 3 --max-steps 5000

# Experiment 4 — Single-space problem-solving with roles
python -m experiments.exp4_mountain --trials 3 --max-steps 3000

# Experiment 5 — Dual-space problem-solving (the main experiment)
python -m experiments.exp5_dual_space --trials 3 --max-steps 3000
```

Results are saved as JSON to `results/exp{N}_results.json`. Each experiment prints a summary table and cost-efficiency Pareto frontier to stdout.

### Visualizer (requires GPU + display)

```bash
# 2D visualizer with experiment observatory panel
python -m sim_2d_exp

# 3D visualizer with mountain surface rendering
python -m 3D_sim.main
```

In the 2D visualizer, use the **Experiments** collapsing header in the right panel to select and start any of the five experiments. Parameters can be adjusted via sliders while the experiment runs.

In the 3D visualizer, expand the **Mountain / Cost Landscape** panel to toggle the fitness surface, cost overlay, and mountain mode (particles projected onto the surface). Particle colors reflect team identity (preferences); particle positions on the surface reflect strategy.

### Dependencies

```bash
pip install numpy scipy
pip install numba          # optional, for Numba physics engine
pip install imgui-bundle   # required for visualizers
pip install moderngl        # required for 3D visualizer
```
