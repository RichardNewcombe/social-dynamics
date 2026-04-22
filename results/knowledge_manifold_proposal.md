# Knowledge Manifold Implementation Proposal

## 1. The New Model

The fitness landscape is the **hidden truth** — a surface over (pref0, pref1) that defines the maximum achievable knowledge at each skill point. Particles never see this surface directly.

The **knowledge manifold** is a separate 2D grid over the same (pref0, pref1) space. It starts at zero everywhere and grows upward as teams explore and learn. Its height at any point represents the organization's accumulated understanding of that skill domain. It can never exceed the hidden fitness surface — experiments that would push knowledge above reality simply fail (capped).

Particles move on the knowledge manifold surface. Their (X, Z) position is determined by their skill preferences (pref0, pref1). Their Y position is the knowledge height at that point. The goal is to reach the peak of the hidden fitness function, but the organization can only see and walk on the knowledge manifold.

The **3rd preference (pref2)** is purely social — it affects who clusters with whom but has no bearing on mountain position. Two people with identical skills but different social preferences won't naturally form teams, creating realistic friction where optimal team composition is blocked by social incompatibility.

## 2. Code Audit: What Exists

### Richard's Original Code (main branch)

| Component | Location | Description |
|-----------|----------|-------------|
| **3D Simulation** | `3D_sim/simulation3d.py` | 882 lines. Particle positions (N,3), preferences (N,K), toroidal neighbor search (cKDTree or hash grid), physics step with social learning, direction memory, repulsion. Params: `k=3`, `n_neighbors=21`, `step_size=0.005`, etc. |
| **3D Visualizer** | `3D_sim/main.py` | 826 lines. ModernGL + imgui. Orbit camera, particle rendering, trail FBO, velocity field, neighbor lines, recording. No mountain/landscape code. |
| **3D Shaders** | `3D_sim/shaders3d.py` | Particle, trail, splat, box, line, overlay shaders. No mesh shader. |
| **2D Simulation** | `sim_2d_exp/simulation.py` | 1178 lines. Same particle physics but 2D. Includes **spatial memory field** (G x G x K grid), signal/response split, force landscape probing. |
| **2D Params** | `sim_2d_exp/params.py` | Full parameter registry including `memory_field`, `memory_write_rate`, `memory_strength`, `memory_decay`, `memory_blur`, `memory_blur_sigma`. |
| **2D Renderer** | `sim_2d_exp/renderer.py` | Full 2D visualizer with memory field visualization support. |

### Our Additions (feature branch)

| Component | Location | Keep/Discard | Rationale |
|-----------|----------|--------------|-----------|
| **RuggedLandscape** | `experiments/landscape.py` | **Keep** | The hidden fitness surface. Multi-scale peaks + spectral noise creates the "cave ceiling." |
| **CostLandscape** | `experiments/landscape.py` | **Rethink** | Cost could become the "difficulty of learning" at each point rather than movement cost. Or discard entirely — the knowledge manifold already creates natural cost via the gap between knowledge and fitness. |
| **mountain_mesh.py** | `3D_sim/mountain_mesh.py` | **Keep, modify** | The mesh generation and projection code is reusable. We need two meshes: knowledge surface (solid, growing) and fitness surface (wireframe ghost). |
| **Mesh shaders** | `3D_sim/shaders3d.py` | **Keep** | MESH_VERT/MESH_FRAG already work for rendering triangle meshes. |
| **simulation3d additions** | `3D_sim/simulation3d.py` | **Discard most** | The `strategy`, `strategy_step()`, role arrays, and dual-space architecture were designed for "move on the fitness surface." The knowledge manifold model is fundamentally different — particles don't have a separate strategy vector; they have skills (pref0, pref1) and the knowledge grid grows beneath them. |
| **main.py mountain mode** | `3D_sim/main.py` | **Discard, rewrite** | The mountain mode loop (strategy_step, gradient nudge, pos sync) is all wrong for the new model. Need a clean rewrite of the per-frame mountain logic. |
| **Experiments 1-5** | `experiments/exp*.py` | **Discard** | These were designed for the old model. New experiments needed. |
| **Role system** | `simulation3d.py` | **Rethink** | Per-particle roles (step_scale, influence, noise, visionary) could still make sense but need to be reframed. A "visionary" in the new model might be someone who can sense the fitness ceiling (the gap between knowledge and truth) rather than knowing where the summit is. |

## 3. The Knowledge Manifold Architecture

### 3.1 State

```
knowledge_grid: (G, G) float64    # accumulated knowledge at each (pref0, pref1) cell
                                   # starts at 0, capped at fitness_grid values
fitness_grid: (G, G) float64      # hidden truth — precomputed from RuggedLandscape
                                   # this is the ceiling; knowledge can never exceed it
```

### 3.2 Preferences (K=3)

| Dimension | Name | Role |
|-----------|------|------|
| pref[0] | Skill X | Determines X position on the mountain. Represents one axis of technical expertise. |
| pref[1] | Skill Y | Determines Z position on the mountain. Represents the other axis of technical expertise. |
| pref[2] | Social | Purely social preference. Affects neighbor graph and team formation. Has NO effect on mountain position. |

The existing social physics (social learning, repulsion, direction memory) operates on ALL 3 preferences as before. This means:
- Pref[0] and pref[1] evolve via social learning → particles with similar skills cluster → teams form in skill space
- Pref[2] evolves independently → social compatibility affects who you cluster with
- The tension: two people at the same skill point but with different pref[2] won't be spatial neighbors (because the 3D toroidal distance includes pref[2]), so they won't form a team even though they could productively collaborate

### 3.3 Per-Frame Loop

**Phase 1: Social Dynamics** (existing `sim.step()`, unchanged)

Richard's original physics runs on all 3 preferences in the toroidal cube. This builds the neighbor graph and applies social learning. Particles cluster by preference similarity (all 3 dims), forming organic teams.

**Phase 2: Knowledge Growth** (new, replaces `strategy_step()`)

For each particle at skill position (pref[0], pref[1]):
1. **Sense**: Take a noisy measurement of the hidden fitness at this skill point. The measurement is `true_fitness + noise`, where noise is high (individuals are nearly blind).
2. **Team aggregate**: Average the noisy measurements from all spatial neighbors (the team). With N neighbors, noise reduces by sqrt(N). This is THE mechanism for group advantage.
3. **Update knowledge grid**: Write the team-aggregated measurement to the knowledge grid at this cell (capped at the true fitness ceiling). Use `max(current_knowledge, measurement)` — knowledge only grows, never shrinks (within a decay cycle).
4. **Diffuse knowledge**: Apply Gaussian blur to the knowledge grid. This is the spreading effect — one team's discoveries raise the manifold in nearby skill regions, benefiting other teams.
5. **Decay** (optional): Slight decay so old knowledge fades if not reinforced. This creates pressure to maintain exploration.

**Phase 3: Position Sync** (modified from current)

Project particles onto the knowledge manifold surface (not the fitness surface):
- X = (pref[0] + 1) / 2
- Y = knowledge_grid[cell_x, cell_y] * z_scale  (normalized)
- Z = (pref[1] + 1) / 2

### 3.4 What Drives Particle Movement?

Particles don't follow the fitness gradient (they can't see it). Instead, their movement is driven by:

1. **Social learning** (Phase 1): Prefs drift toward neighbor means. This clusters particles in skill space, forming teams. The pref[2] social dimension creates friction — socially incompatible people resist clustering even if their skills overlap.

2. **Knowledge gradient** (optional, Phase 2): Particles could be attracted to the steepest uphill direction on the knowledge manifold — "go where we know the most." This is exploitation.

3. **Frontier attraction** (optional, Phase 2): Particles could be attracted to the boundary between known and unknown — the edge of the knowledge manifold where knowledge drops off. This is exploration.

4. **Visionaries**: Some particles have a weak signal pointing toward the true summit (or at least toward high-fitness regions). They pull their teams in productive directions even when the knowledge manifold is flat.

The balance between exploitation (climb the knowledge surface) and exploration (push into the unknown) is a key parameter.

### 3.5 Visualization

| Surface | Rendering | Description |
|---------|-----------|-------------|
| **Knowledge manifold** | Solid mesh, terrain colormap | What the organization knows. Starts flat, grows over time. Particles walk on this. |
| **Hidden fitness** | Transparent wireframe or ghost overlay | The truth. The "cave ceiling." Visible to the viewer but not to the particles. |
| **Gap** | The space between the two surfaces | Represents ignorance. The goal is to close this gap at the summit. |

The visual story: you watch the knowledge surface rise from flat toward the fitness ceiling as teams explore. Teams that cluster in productive skill regions raise the manifold faster. The pref[2] social dimension creates visible social clusters (color-coded) that may or may not align with optimal skill groupings.

## 4. Implementation Plan

### Step 1: Clean Branch from Richard's Main

Start a new branch from `origin/main` (Richard's clean code). Cherry-pick only what we need:
- `experiments/landscape.py` — the RuggedLandscape and factory functions (the hidden fitness surface)
- `3D_sim/mountain_mesh.py` — mesh generation utilities (modified for two surfaces)
- `3D_sim/shaders3d.py` — the MESH_VERT/MESH_FRAG shaders we added

### Step 2: Add Knowledge Grid to simulation3d.py

Minimal additions to Richard's original 3D Simulation class:
- `knowledge_grid: (G, G) float64` — the organizational knowledge field
- `fitness_grid: (G, G) float64` — precomputed hidden truth (from RuggedLandscape)
- `knowledge_step()` — Phase 2 method: sense, aggregate, update, diffuse
- Parameters: `knowledge_write_rate`, `knowledge_decay`, `knowledge_blur_sigma`, `sensing_noise`, `frontier_attraction`

This is structurally very similar to Richard's existing `memory_field` in the 2D sim. The key difference: `memory_field` stores K-dimensional preference vectors and modulates social learning; `knowledge_grid` stores scalar fitness knowledge and determines particle height.

### Step 3: Modify mountain_mesh.py

- `generate_knowledge_mesh()` — samples the knowledge grid (not the landscape) to produce the solid surface mesh. Called every frame (or every N frames) since the knowledge surface changes.
- `generate_fitness_ghost_mesh()` — samples the hidden fitness landscape once to produce the wireframe ceiling mesh.
- `project_particles_to_knowledge()` — projects particles onto the knowledge surface (Y = knowledge at their skill point).

### Step 4: Modify main.py Mountain Mode

Replace the current mountain mode loop with:
1. `sim.step()` — social dynamics (unchanged)
2. `sim.knowledge_step(landscape)` — knowledge growth
3. Project particles onto knowledge surface
4. Rebuild knowledge mesh (or update GPU buffer)
5. Render both knowledge mesh (solid) and fitness mesh (wireframe)

### Step 5: New Experiment

A clean experiment that measures:
- How fast the knowledge manifold grows toward the fitness ceiling
- Whether teams outperform isolated individuals
- How pref[2] social friction affects team formation and knowledge growth
- Whether visionaries accelerate summit discovery

## 5. Key Design Decisions to Confirm

1. **Does the knowledge grid use `max(current, measurement)` or `running_average`?** Max means knowledge only grows (within decay). Running average means noisy measurements can temporarily lower knowledge. I suggest max — it's more intuitive ("we learned this, we can't unlearn it") and creates a cleaner visual.

2. **How does knowledge diffusion work?** Gaussian blur is simple and matches Richard's memory_blur. But it means knowledge spreads equally in all directions. An alternative: knowledge only spreads along the neighbor graph (social connections), so socially isolated groups can't benefit from others' discoveries. This would make pref[2] even more impactful.

3. **Should particles be attracted to the knowledge gradient, the frontier, or neither?** If neither, particles only move via social learning (Phase 1), and knowledge grows passively wherever teams happen to be. This is the simplest model and might be enough — social learning already clusters particles, and clusters raise knowledge faster.

4. **Cost landscape**: Should we keep it? In the new model, the "cost" of exploration could be implicit — it takes time and team resources to raise the knowledge manifold. An explicit cost layer might be redundant. Or it could represent "difficulty of learning" — some skill regions are harder to learn about, requiring more measurements to raise the manifold.

5. **Pref[2] range in neighbor search**: Currently the 3D toroidal distance weights all 3 dims equally. Should pref[2] have a different weight? If it's too strong, no teams form across social lines. If it's too weak, it has no effect. This is a tuning question.

## 6. Lessons Learned from Previous Iterations

These are mistakes and insights from the current branch that should inform the new implementation:

1. **Toroidal coordinate mismatch**: When projecting particles onto a surface, `sim.pos` must be synced to the projected coordinates so that the spatial neighbor search operates on the visible positions. We solved this but it was a subtle bug.

2. **K > 2 dimension mismatch**: The mountain mesh is a 2D slice. Any projection must evaluate fitness/knowledge on the same 2D slice (zeroing dims 2+). This was another subtle bug.

3. **Normalization before averaging destroys noise reduction**: When aggregating noisy gradient observations, do NOT normalize individual observations to unit length before averaging. The signal-to-noise improvement from averaging requires the noise to be additive, not directional.

4. **Cost gradient magnitude calibration**: If cost and fitness have different magnitudes, one will dominate. Any composite gradient must be carefully calibrated.

5. **Random neighbors vs spatial neighbors**: Headless tests with random fixed neighbors underestimate team advantage because spatially clustered teams have coherent signals. Always test with the real spatial neighbor graph.

6. **Mesh update performance**: Rebuilding a triangle mesh every frame is expensive. For the knowledge manifold (which changes every frame), we should update only the Y coordinates of existing vertices, not rebuild the full mesh.
