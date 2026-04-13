# Experimental System: Theory and Technical Architecture

## 1. Introduction

This document describes the experimental framework built on top of the HBFM (Heterogeneous Behavioral Field Model) particle simulator. The framework transforms a general-purpose social dynamics simulator into a platform for studying how organizational structure affects collective problem-solving — specifically, how teams of agents with heterogeneous capabilities navigate a complex fitness landscape under cost constraints.

The central metaphor is **mountain climbing**: an organization must find the summit of a rugged mountain (the optimal strategy) without a known path. The mountain has local peaks that trap greedy searchers, cost ridges that penalize certain routes, and the global summit is hidden behind deceptive terrain. Success depends not just on individual capability but on how the team is organized — who talks to whom, who leads, who explores, and how tightly team identity is coupled to strategic direction.

The system comprises five experiments of increasing sophistication, each building on the last:

| Experiment | Focus | Key Question |
|---|---|---|
| Exp 1: Hidden Target Search | Social learning rate | Does conformity or differentiation help a group converge on a target? |
| Exp 2: Multi-Niche Coverage | Diversity maintenance | Can social dynamics maintain coverage across multiple niches? |
| Exp 3: Ghost Colony Escape | Institutional memory | How does accumulated memory (inertia) slow adaptation to shocks? |
| Exp 4: Mountain Climbing | Role heterogeneity + fitness landscape | How do researcher/engineer/leader/visionary roles affect navigation of a rugged landscape? |
| Exp 5: Dual-Space Navigation | Preference-strategy separation | When team identity and strategic direction are decoupled, how does coupling strength affect performance? |

---

## 2. Theoretical Framework

### 2.1 The Dual-Space Model

The core theoretical contribution is the separation of organizational dynamics into two coupled but distinct spaces:

**Preference space** represents team identity and organizational culture. Each particle (agent) carries a K-dimensional preference vector that determines who they interact with, who influences them, and which cultural norms they adopt. Social learning, memory fields, and neighbor-finding all operate in this space. Preferences evolve slowly through social dynamics — conformity pulls preferences together (team formation), differentiation pushes them apart (diversity maintenance).

**Strategy space** represents the organization's position on the problem landscape. Each particle carries a separate K-dimensional strategy vector that determines their fitness (how good their current approach is) and their cost (how expensive their current path is). Strategy evolves through team-aggregated knowledge sharing — particles sense the local gradient with varying accuracy, share observations through their preference-space social network, and move collectively based on the team's aggregated signal.

The bridge between these two spaces is the **preference-space neighbor graph**. Social dynamics in preference space determine who talks to whom (the team structure). This team structure then determines information flow in strategy space — your team members are the people whose gradient observations you aggregate. The quality of your team's strategic movement depends on both the individual capabilities of its members (roles) and the social structure that connects them.

The **preference-strategy coupling parameter** (0.0 to 1.0) controls how much these two spaces overlap:

| Coupling | Organizational Archetype | Behavior |
|---|---|---|
| 1.0 | Specialist organization | Strategy is initialized as a copy of preferences. What you believe = what you do. Team identity locks strategic direction. |
| 0.5 | Typical organization | Strategy is a noisy blend of preferences and random initialization. Partial overlap between culture and strategy. |
| 0.0 | Generalist organization | Strategy is initialized independently. Team identity is separate from strategic approach. |

This coupling is itself an experimental variable. The hypothesis is that moderate coupling outperforms both extremes: pure specialists are too rigid (team identity locks strategy onto local peaks), while pure generalists waste the team structure (no knowledge transfer benefit from team proximity).

### 2.2 Per-Particle Role Heterogeneity

Real organizations are composed of individuals with different capabilities. The simulation models four distinct roles, each implemented as a per-particle scalar drawn from a configurable distribution:

**Researcher** (`role_gradient_noise`): Controls the accuracy of local gradient sensing. A particle with low noise reads the fitness landscape accurately — it knows which direction is uphill. A particle with high noise gets a corrupted signal — its gradient observation is unreliable. Critically, no one in the simulation *knows* who the good researchers are. The accuracy is revealed only through outcomes: when the team aggregates gradient observations, the good researchers' signals are coherent and reinforce each other, while the bad researchers' noise is random and cancels out. This is an emergent filtering mechanism, not an explicit one.

**Engineer** (`role_step_scale`): Controls the magnitude of movement along the chosen direction. A high-engineer particle takes large steps — it moves fast but may overshoot. A low-engineer particle takes small steps — it's precise but slow. Engineers don't determine direction; they amplify whatever direction the team chooses. The distribution is log-normal, so most particles are near 1.0 with a few outliers capable of large jumps.

**Leader** (`role_influence`): Controls the weight of a particle's gradient observation in the team aggregation. When the team averages its members' noisy gradient signals, leaders' observations count more. A high-influence leader pulls the team's collective direction toward their own observation — whether that observation is accurate or not. The interaction between leader influence and researcher accuracy is the key tension: a loud leader who is also a good researcher is invaluable; a loud leader who is a bad researcher is catastrophic. The distribution is log-normal, creating a natural hierarchy where a few particles have disproportionate influence.

**Visionary** (`role_visionary`): Controls the blend between local gradient sensing and a direct signal toward the global summit. A visionary particle partially ignores the local terrain and pulls toward the peak. The visionary signal is:

> `effective_gradient = (1 - v) * noisy_local_gradient + v * direction_to_summit`

where `v` is the per-particle visionary weight. Visionaries are subject to a **rarity constraint** (`role_visionary_fraction`): only a configurable fraction of particles (default 10%) receive any visionary ability at all. The rest have their visionary weight zeroed out regardless of the distribution parameters. This models the real-world observation that true strategic visionaries are rare.

The negative cost of too many visionaries emerges naturally from the simulation dynamics rather than from an artificial penalty:

1. **Cost ridges**: Visionaries march in a straight line toward the summit, ignoring terrain cost. The direct path may cross the most expensive regions.
2. **Search homogenization**: Every visionary points the same direction. With social learning, a team of visionaries becomes a herd marching in lockstep, losing the exploration diversity needed to find alternative routes.
3. **Signal conflict**: Visionary pull and researcher gradient pull point in different directions when the local gradient leads away from the summit (e.g., around a ridge). The blended signal becomes muddled.
4. **Employee cost**: Visionaries are the most expensive role (weighted at 2x in the cost function), creating direct budget pressure against stacking them.

### 2.3 The Employee Cost Model

Each particle incurs a per-step operating cost that is a function of its role magnitudes:

> `cost_i = base + w_e * step_scale_i + w_l * influence_i + w_r * (1 / noise_i) + w_v * visionary_i`

The key design choices:

- **Researcher cost is inverse noise**: A low-noise (accurate) researcher costs more than a high-noise one. Hiring accurate researchers is expensive.
- **Visionary cost is high-weighted** (2x): Visionaries are the most expensive individual capability, creating the budget tension that prevents "just make everyone a visionary."
- **Cost is per-step, accumulated over time**: A team that solves quickly pays less total cost than a team that wanders for thousands of steps. Speed matters.
- **Employee cost is independent of terrain cost**: You pay for your people regardless of where they are on the mountain. Terrain cost is an additional burden based on position.

### 2.4 The Fitness Landscape

The fitness landscape is the "mountain" the organization is trying to climb. It maps K-dimensional strategy vectors to scalar fitness values and provides gradient information for navigation.

The current implementation uses a `RuggedLandscape` class that composites three layers:

**Major peaks** (weight 0.55): Four large Gaussian bumps defining the macro structure. The global summit (height 1.0, sigma 0.22) is tucked in a corner at approximately [0.72, 0.78, 0.65]. Three prominent local optima create deceptive alternatives: a near-center peak (height 0.75, sigma 0.35) with a wide basin that is easy to find and hard to leave; a ridge peak (height 0.65) between the origin and summit that creates a false path; and a distant corner peak (height 0.55) that traps explorers who wander too far.

**Minor peaks** (weight 0.25): Twelve smaller Gaussian bumps creating foothills, false summits, and ridgeline bumps. These make local gradient following unreliable — a particle climbing what appears to be a promising slope may reach a minor peak and find itself stuck at fitness 0.40 with no obvious upward direction.

**Spectral noise** (weight 0.20): Twenty sum-of-sinusoid frequencies adding continuous roughness across the entire surface. This is a deterministic, differentiable alternative to Perlin noise. The amplitude decays with frequency magnitude (higher frequency = smaller bumps), creating a natural multi-scale texture of ridges, saddle points, and subtle valleys. The noise ensures that no region of the landscape is truly smooth — even the slopes of major peaks have local undulations that can mislead noisy gradient sensors.

The composite fitness is `0.55 * max_major + 0.25 * max_minor + 0.20 * noise`, clipped to [0, 1]. The gradient is computed analytically through all three layers, making it smooth and differentiable everywhere.

### 2.5 The Cost Terrain

The cost terrain is an independent landscape overlaid on the fitness mountain. It determines how expensive each region of strategy space is to traverse. The cost terrain is implemented as a `CostLandscape` with six Gaussian ridges plus a base cost:

- A **main cost ridge** (height 2.5) blocks the direct diagonal path from the origin to the summit
- A **secondary ridge** (height 1.8) guards the southern approach
- An **implementation cost zone** (height 1.2) near the summit itself — arriving at the peak is expensive
- Smaller ridges scattered across the landscape creating localized cost pockets

The deceptive peak near the center sits in a **low-cost zone** (no ridge nearby), making it doubly tempting: it's easy to find (wide basin), moderately fit (height 0.75), and cheap to reach. The optimal path to the summit requires navigating around the main cost ridge, which means accepting temporary fitness loss for long-term gain — a strategic decision that requires the team to resist the local gradient.

---

## 3. Technical Architecture

### 3.1 Core Simulation Extensions

The experimental system extends the base `Simulation` class in `sim_2d_exp/simulation.py` with three categories of additions, all backward-compatible (disabled by default):

**Per-particle role arrays** (initialized in `_init_roles()`):

| Array | Shape | Distribution | Default |
|---|---|---|---|
| `role_step_scale` | (N,) | Log-normal(0, std) | All 1.0 |
| `role_influence` | (N,) | Log-normal(0, std) | All 1.0 |
| `role_gradient_noise` | (N,) | |Normal(mean, std)| | All 0.5 |
| `role_visionary` | (N,) | Clipped Normal(mean, std) × mask | All 0.0 |

Roles are activated by setting `use_particle_roles=True`. The visionary rarity mask zeros out `(1 - fraction)` of particles' visionary weights, selected randomly.

**Strategy array and initialization** (in `reset()`):

When `strategy_enabled=True`, the simulation allocates a second state vector `self.strategy` of shape (N, strategy_k). Initialization depends on `pref_strategy_coupling`:

```
strategy = coupling * prefs[:, :strategy_k] + (1 - coupling) * random_uniform(-1, 1)
```

At coupling=1.0, strategy starts as a copy of preferences. At coupling=0.0, strategy starts fully random. At intermediate values, it's a weighted blend with added noise.

**Strategy memory field**: A second spatial grid (`self.strategy_memory`) of shape (G, G, strategy_k) that operates in strategy space, parallel to the existing cultural memory field in preference space. It supports the same write/decay/blur dynamics.

**`strategy_step()` method**: The Phase 2 mountain navigation loop, called by the experiment's post-step hook after `sim.step()`. The algorithm:

1. **Gradient sensing**: Each particle evaluates the true gradient of the fitness landscape at its strategy position, then corrupts it with Gaussian noise scaled by `role_gradient_noise`.
2. **Visionary blend**: Particles with non-zero visionary weight blend their noisy gradient with a unit vector pointing directly at the global summit.
3. **Team aggregation**: Using the preference-space neighbor graph from Phase 1, each particle's gradient observation is shared with its neighbors. The team's collective gradient is a weighted average of neighbor observations, where weights are `role_influence` values. Each particle also blends in its own observation at 50% weight (self vs. team).
4. **Strategy movement**: Each particle moves in strategy space along the team's collective gradient direction, scaled by `strategy_step_size * role_step_scale`.
5. **Knowledge memory** (optional): If enabled, the strategy memory field reads (nudges strategy), writes (deposits current strategy), decays, and blurs — accumulating institutional knowledge about the landscape.

### 3.2 Parameter System

Parameters are managed through a three-layer resolution system in `experiments/runner.py`:

1. **`SAFE_DEFAULTS`**: Baseline values that ensure reproducible headless runs (physics_engine=1/NumPy, seed=42, 500 particles, etc.)
2. **Ambient `params`**: The global mutable parameter dict from `params.py`, which the GUI modifies in real-time
3. **Experiment overrides**: Per-condition parameter dicts passed to `run_experiment()`

Resolution order: `SAFE_DEFAULTS` → ambient `params` → experiment overrides. This means headless experiments are fully reproducible (SAFE_DEFAULTS provide a stable base), while the visualizer can still influence experiments through the GUI.

New parameters added for the experimental system:

| Parameter | Default | Description |
|---|---|---|
| `use_particle_roles` | False | Enable per-particle role heterogeneity |
| `role_step_scale_std` | 0.0 | Std of log-normal engineer distribution |
| `role_influence_std` | 0.0 | Std of log-normal leader distribution |
| `role_gradient_noise_mean` | 0.5 | Mean researcher noise (lower = better) |
| `role_gradient_noise_std` | 0.0 | Std of researcher noise distribution |
| `role_visionary_mean` | 0.0 | Mean visionary blend weight |
| `role_visionary_std` | 0.0 | Std of visionary distribution |
| `role_visionary_fraction` | 1.0 | Fraction of particles eligible for visionary ability |
| `strategy_enabled` | False | Enable separate strategy vector |
| `strategy_k` | 3 | Dimensionality of strategy space |
| `pref_strategy_coupling` | 0.5 | Coupling between preferences and strategy |
| `strategy_step_size` | 0.003 | Base step size for strategy movement |
| `strategy_memory_enabled` | False | Enable knowledge memory field |
| `strategy_memory_strength` | 0.5 | Knowledge field modulation strength |
| `strategy_memory_write_rate` | 0.01 | Knowledge field write rate |
| `strategy_memory_decay` | 0.999 | Knowledge field decay per step |
| `strategy_memory_blur` | False | Enable Gaussian blur on knowledge field |
| `strategy_memory_blur_sigma` | 1.0 | Blur sigma for knowledge field |

### 3.3 Landscape Module

The `experiments/landscape.py` module provides three landscape classes and factory functions:

- **`GaussianPeakLandscape`**: Simple multi-peak landscape using max-over-Gaussians. Used by `make_rugged_landscape()` for quick testing.
- **`RuggedLandscape`**: Multi-scale landscape combining major peaks, minor peaks, and spectral noise. Used by `make_default_landscape()` for the main experiments.
- **`CostLandscape`**: Independent cost terrain using sum-of-Gaussians ridges. Used by `make_default_cost_landscape()`.

All landscapes provide `fitness(prefs)` → (fitness, peak_ids) and `gradient(prefs)` → (grad_unit, fitness, peak_ids) interfaces. The gradient is computed analytically, not numerically, ensuring smooth and accurate gradient signals for the researcher sensing mechanism.

The `compute_employee_cost(sim)` function evaluates the per-particle operating cost based on role magnitudes.

### 3.4 3D Visualization

The 3D visualizer (`3D_sim/main.py`) renders the mountain as a triangle mesh using ModernGL:

- **Mountain mesh**: Generated by `mountain_mesh.py`, which samples the fitness landscape on a regular grid over pref[0] × pref[1] (or strategy[0] × strategy[1]), computes normals via finite differences, and assigns a terrain colormap (blue → green → yellow → red → white).
- **Cost overlay**: Same mesh geometry, colored by the cost terrain (green → yellow → red). Rendered with polygon offset to avoid z-fighting.
- **Mountain mode**: When enabled, particles are projected onto the mountain surface. Their X,Y come from strategy[0:2] (when strategy is enabled) or prefs[0:2], and their Z = fitness(position). Particles "walk on the mountain."
- **Particle colors**: Always derived from preferences (RGB mapping), showing team identity regardless of mountain position. This creates the key visual: colored clusters (teams) moving across the mountain surface.

ImGui controls provide toggles for mountain visibility, cost overlay, mountain mode, alpha transparency, Z-scale, and mesh resolution.

---

## 4. Experiment Descriptions

### 4.1 Experiment 1: Hidden Target Search

**Question**: How does social learning rate affect convergence to a hidden target?

**Design**: A hidden target is placed in preference space. Each step, a gradient nudge pushes particles toward the target. The social learning rate is swept from strong differentiation (−0.03) through zero to strong conformity (+0.03) across 7 values, with 3 trials each.

**Metric**: Steps to solve (all particles within threshold of target).

**Key finding**: The relationship is a J-curve, not a U-shape. Strong negative social learning (differentiation) dramatically slows convergence because particles actively repel each other. Mild positive conformity is fastest. Strong conformity shows a slight slowdown from premature clustering.

### 4.2 Experiment 2: Multi-Niche Coverage

**Question**: Can social dynamics maintain coverage across multiple niches simultaneously?

**Design**: Four niche centers are placed in preference space. Gradient nudges attract particles to the nearest niche. The experiment tests uniform social learning (positive and negative), quiet-dimension differentiation, and a no-social baseline.

**Metric**: Steps until all 4 niches have at least one particle within threshold.

**Key finding**: Uniform social learning (both conformity and differentiation) fails to cover all niches — it homogenizes the swarm. The no-social baseline wins because the gradient alone is sufficient and social forces only interfere.

### 4.3 Experiment 3: Ghost Colony Escape

**Question**: How does accumulated spatial memory (institutional inertia) slow adaptation after a sudden environmental change?

**Design**: The simulation runs for 2500 steps to build up a memory field, then a "shock" resets all preferences to a new target. Memory strength is swept from 0.0 to 10.0 across 5 values.

**Metric**: Steps to re-adapt after the shock.

**Key finding**: At memory_strength=10, adaptation takes approximately 243 steps after the shock versus 1–2 steps with no memory. The memory field retains the old preference landscape and fights the new direction through multiplicative gating — a direct model of institutional inertia.

### 4.4 Experiment 4: Mountain Climbing (Single-Space)

**Question**: How do per-particle roles (researcher, engineer, leader, visionary) affect navigation of a rugged fitness landscape?

**Design**: Preferences serve double duty as both team identity and strategy position (coupling=1.0). The experiment sweeps 9 conditions: baseline, social conformity, social differentiation, heterogeneous leaders, heterogeneous engineers, full roles, world knowledge (memory), and visionary configurations at different rarity levels.

**Metrics**: Summit fraction (% of particles reaching the global peak), local trap fraction, total cost (terrain + employee), cost-efficiency (summit% / cost).

**Key finding**: The rugged landscape makes the problem genuinely hard — most conditions achieve less than 10% summit fraction. Social conformity herds everyone onto local peaks. Social differentiation avoids traps but can't converge. Visionaries help but are expensive, and their benefit diminishes with rarity constraints.

### 4.5 Experiment 5: Dual-Space Mountain Climbing

**Question**: When team identity (preferences) and strategic direction (strategy) are separate, how does the coupling between them affect organizational performance?

**Design**: The full dual-space architecture with Phase 1 (social dynamics in preference space) and Phase 2 (mountain navigation in strategy space). The experiment sweeps 12 conditions across three dimensions:

1. **Coupling sweep** (1.0, 0.75, 0.50, 0.25, 0.0): From specialist to generalist
2. **Social learning interaction**: Generalist + strong social, specialist + differentiation, moderate + no social
3. **Visionary configurations**: Rare strong visionaries at specialist and generalist coupling
4. **Knowledge memory**: Moderate coupling with strategy memory field enabled
5. **Baseline**: No roles, moderate coupling

**Metrics**: Summit fraction, local trap fraction, preference-strategy divergence (how far apart the two spaces have drifted), number of preference-space teams (DBSCAN clusters), total cost broken down by terrain and employee components, cost-efficiency Pareto frontier.

**Key findings from initial runs**:

- Generalist organizations (coupling=0.0) outperform specialists at reaching the summit, because strategy can evolve independently of team dynamics
- Rare visionaries combined with generalist structure show the highest cost-efficiency
- Specialist organizations with social differentiation avoid traps but can't converge — the differentiation in preference space also scatters strategy (because coupling=1.0)
- The preference-strategy divergence metric reveals how far the two spaces drift apart over time, providing a direct measure of organizational coherence

---

## 5. Simulation Loop Summary

The complete per-step simulation loop for a dual-space experiment:

```
for each step:
    # Phase 1: Social dynamics (preference space)
    sim.step()
        → Find neighbors by preference similarity (KD-tree)
        → Social learning: pull preferences toward/away from neighbors
        → Cultural memory field: read, write, decay, blur
        → Position update (spatial movement)
        → Role arrays scale movement (engineer) and influence (leader)

    # Phase 2: Mountain navigation (strategy space)
    post_step_fn(sim, step)
        → sim.strategy_step(gradient_fn, summit_center)
            → Each particle senses gradient with researcher noise
            → Visionaries blend toward summit
            → Team aggregation: weight by leader influence using
              the preference-space neighbor graph
            → Move in strategy space (engineer step scale)
            → Knowledge memory field: read, write, decay, blur
        → Accumulate terrain cost (from strategy position)
        → Accumulate employee cost (from role magnitudes)

    # Logging and termination check
    log_fn(sim, step)  → record metrics
    check_fn(sim, step) → test if solved
```

---

## 6. File Reference

| File | Purpose |
|---|---|
| `sim_2d_exp/simulation.py` | Core simulation: `_init_roles()`, `strategy` array, `strategy_step()`, strategy memory field |
| `sim_2d_exp/params.py` | All parameter definitions with defaults |
| `experiments/landscape.py` | `GaussianPeakLandscape`, `RuggedLandscape`, `CostLandscape`, `compute_employee_cost()` |
| `experiments/runner.py` | Headless experiment harness: `SAFE_DEFAULTS`, `run_experiment()`, `run_sweep()` |
| `experiments/exp1_hidden_target.py` | Experiment 1: social learning rate sweep |
| `experiments/exp2_niche_coverage.py` | Experiment 2: diversity maintenance |
| `experiments/exp3_ghost_colony.py` | Experiment 3: institutional inertia |
| `experiments/exp4_mountain.py` | Experiment 4: single-space mountain climbing with roles |
| `experiments/exp5_dual_space.py` | Experiment 5: dual-space mountain climbing |
| `experiments/controller.py` | Visualizer experiment integration (all 5 experiments) |
| `3D_sim/main.py` | 3D visualizer with mountain mesh rendering |
| `3D_sim/mountain_mesh.py` | Mountain mesh generation for ModernGL |
| `3D_sim/shaders3d.py` | GLSL shaders including `MESH_VERT`/`MESH_FRAG` |
| `scripts/render_mountain_screenshots.py` | Offline matplotlib rendering for static screenshots |

---

## 7. Open Questions and Future Directions

**Landscape complexity**: The current rugged landscape is significantly harder than the original three-Gaussian version, but the "all visionaries" condition still achieves 30% summit fraction by ignoring terrain entirely. More complex landscapes with narrow corridors, saddle points, and deceptive ridges would further stress the organizational dynamics.

**Visionary rarity tuning**: The hypothesis that "too many visionaries" is harmful should emerge from the cost terrain + social dynamics, but current results show visionaries consistently winning. Stronger cost ridges on the direct path, or a landscape where the straight-line path to the summit is genuinely impassable, would make the visionary tradeoff more visible.

**Knowledge memory field**: Initial results show the strategy memory field doesn't help. This may be because it encodes early random exploration (noise) rather than useful gradient knowledge. A more sophisticated write rule — depositing gradient direction rather than raw position — might make institutional knowledge genuinely useful.

**Budget constraints**: Currently cost is accumulated and reported but doesn't constrain behavior. Adding a hard budget cap (simulation terminates when cost exceeds threshold) would force genuine tradeoffs between exploration speed and cost efficiency.

**Adaptive coupling**: The preference-strategy coupling is currently fixed at initialization. In real organizations, coupling changes over time — startups begin as generalists and specialize as they find product-market fit. A dynamic coupling parameter that evolves based on fitness feedback would model organizational maturation.

**Higher-dimensional landscapes**: The current 3D visualization projects the landscape onto two dimensions. For K > 3, the landscape has structure in dimensions that aren't visible. Developing visualization techniques for higher-dimensional fitness landscapes (e.g., t-SNE projections, parallel coordinates) would enable richer experiments.
