# Problem-Solving Experiments

This package layers measurable problem-solving objectives on top of the
base social-dynamics simulation.  By defining what constitutes a "solution,"
we can measure how long the particle swarm takes to find it under different
organisational configurations — turning the simulation into a laboratory
for studying collective problem solving.

## Theoretical Background

The experiments draw on established models from organisational science and
complex systems:

- **NK Fitness Landscapes** (Kauffman 1993; Levinthal 1997) — complex
  problems with interdependent dimensions.
- **Exploration vs. Exploitation** (March 1991; Lazer & Friedman 2007) —
  the trade-off between searching for new solutions and refining known ones.
- **Cognitive Diversity** (Hong & Page 2004) — diverse teams outperform
  homogeneous ones on complex problems.
- **Organisational Memory** (Walsh & Ungson 1991) — institutional knowledge
  that persists beyond individual tenure.

## Experiments

### Experiment 1: Hidden Target Search

**File:** `exp1_hidden_target.py`

A hidden optimal preference vector is defined.  A weak fitness gradient
nudges particles toward the target, but social dynamics dominate movement.
The experiment sweeps the **social learning rate** to test whether conformity
or differentiation leads to faster discovery.

```bash
python -m experiments.exp1_hidden_target --trials 5 --max-steps 20000
```

**Key finding:** Moderate conformity converges fastest.  Strong differentiation
prevents clustering near the target.  Very strong conformity slows convergence
due to premature lock-in.

### Experiment 2: Multi-Niche Coverage

**File:** `exp2_niche_coverage.py`

Four distinct "expertise niches" are defined in preference space.  The
problem is solved when all four niches are simultaneously occupied.  The
experiment compares **Uniform Social Learning** vs. **Quiet-Dimension
Differentiation** to test which organisational culture better maintains
cognitive diversity.

```bash
python -m experiments.exp2_niche_coverage --trials 5 --max-steps 20000
```

**Key finding:** Quiet-Dimension Differentiation (social_mode=1) with
positive social learning reliably fills all niches, while uniform negative
social learning prevents convergence entirely.

### Experiment 3: Ghost Colony Escape

**File:** `exp3_ghost_colony.py`

Particles settle into Zone A preferences while the spatial memory field
accumulates an institutional imprint.  Then a "shock" flips all preferences
to Zone B.  The memory field fights back, dampening the new preferences
through its multiplicative modulation.  The experiment sweeps
**memory_strength** to measure institutional inertia.

```bash
python -m experiments.exp3_ghost_colony --trials 5 --max-steps 15000
```

**Key finding:** Adaptation time increases dramatically with memory strength.
At memory_strength=0, adaptation is instantaneous; at memory_strength=10,
it takes ~150× longer as the "ghost colony" resists the new direction.

## Architecture

```
experiments/
├── __init__.py              # Package docstring
├── README.md                # This file
├── fitness.py               # Fitness functions and gradient utilities
├── runner.py                # Headless experiment runner and sweep harness
├── exp1_hidden_target.py    # Experiment 1
├── exp2_niche_coverage.py   # Experiment 2
└── exp3_ghost_colony.py     # Experiment 3
```

### Key Components

- **`fitness.py`** — Reusable fitness functions: `hidden_target_fitness`,
  `hidden_target_gradient`, `niche_occupancy`, `generate_niches`,
  `zone_fitness`, `zone_gradient`.

- **`runner.py`** — `run_experiment()` configures params, creates a headless
  `Simulation`, steps it with optional callbacks, and returns timing/metric
  logs.  `run_sweep()` runs a Monte Carlo parameter sweep.

- **Each experiment** — Defines constants, callbacks (post-step hooks,
  termination checks, loggers), and a `main()` that runs the sweep and
  prints a summary table.

## Output

Results are saved as JSON files in `results/`.  Each file contains an array
of trial records with:

- `solved` — whether the objective was met
- `solve_step` — step at which it was met (or null)
- `wall_time` — wall-clock seconds
- `log` — periodic metric snapshots

## Requirements

The experiments use the NumPy physics engine (engine 1) by default, so no
GPU or display is needed.  Dependencies:

- `numpy`
- `scipy` (for memory field blur)
- `numba` (for spatial hashing)
