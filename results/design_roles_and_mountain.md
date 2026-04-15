# Design: Per-Particle Roles & Mountain-Climbing Experiment

## Architecture Decision

After reading all physics backends (Numba, NumPy, PyTorch, Grid), the key observation is:

1. **`step_size`** is a scalar used in `new_pos = pos + step_size * movement` — appears in ALL backends
2. **`social`** is a scalar used in `new_prefs = (1-social)*prefs + social*nbr_mean` — appears in ALL backends
3. Movement direction is computed per-particle already (based on neighbor selection)

### Where to inject per-particle factors

**Option A: Modify all backend kernels** — Change Numba/NumPy/PyTorch/Grid signatures to accept per-particle arrays for step_size and social. This is invasive (4+ files, Numba recompilation) and breaks the clean scalar interface.

**Option B: Post-step layer in Simulation.step()** — Apply per-particle scaling AFTER the physics kernel returns but BEFORE position/pref update is finalized. This is clean but can't affect the INTERNAL movement computation (which neighbor is "best" depends on the kernel's internal logic).

**Option C: Pre/post modulation in Simulation.step()** — Similar to how memory_field already works: modulate inputs before the kernel, then adjust outputs after. The memory field already does `prefs = prefs * (1 + strength * field)` before physics, then restores and applies delta after. We can use the same pattern for roles.

### Chosen approach: Hybrid (new arrays + post-step layer)

The cleanest backward-compatible approach:

1. **Add per-particle arrays to Simulation**: `self.role_influence` (N,), `self.role_step_scale` (N,), `self.role_gradient_noise` (N,) — all default to 1.0/1.0/0.0 so existing behavior is unchanged.

2. **Apply role_step_scale in Simulation.step()** after the kernel: Instead of the kernel using scalar `step_size`, the kernel still uses scalar step_size, but we post-multiply movement by `role_step_scale` before the position update. Wait — the kernel already does `new_pos = pos + step_size * movement` internally. So we need to either:
   - (a) Change the kernel to NOT do the position update, just return movement — too invasive
   - (b) Apply a correction: `pos = pos + (role_step_scale - 1) * step_size * movement` — hacky
   - (c) **Best: add a new post-step hook point in Simulation.step()** that runs after the kernel but can adjust pos/prefs

Actually, looking more carefully at the code:

- For NumPy engine: `self.pos = (pos + step_size * movement) % SPACE` happens at line 753 — this is in `_step_numpy`, not in the kernel
- For Numba engine: `new_pos[i] = (pos[i] + step_size * movement[i]) % L` happens INSIDE the Numba kernel
- For PyTorch engine: same, inside `step_torch`

So the NumPy path is easy to modify (it's Python), but Numba/PyTorch would need kernel changes.

### Revised approach: Per-particle arrays + NumPy-layer application

Since the experiments already force `physics_engine=1` (NumPy) for headless, and the NumPy path has the position update in Python-accessible code, the cleanest approach is:

1. Add per-particle role arrays to `Simulation` (initialized to neutral values)
2. In `_step_numpy()`, use `self.role_step_scale[:, None]` to scale movement before position update
3. In the social learning section of `_step_numpy()`, use `self.role_influence` to weight neighbor contributions
4. For Numba/PyTorch, the role arrays are ignored (they still use scalar values) — this is documented as a known limitation
5. The experiment's post_step_fn handles gradient sensing with per-particle noise

This keeps changes minimal and backward-compatible. The Numba/PyTorch backends can be extended later if needed.

## Per-Particle Role Arrays

```python
# In Simulation.reset():
self.role_step_scale = np.ones(n, dtype=np.float64)    # engineer factor
self.role_influence = np.ones(n, dtype=np.float64)      # leader factor  
self.role_gradient_noise = np.zeros(n, dtype=np.float64) # researcher factor (noise std)
```

### Engineer factor (role_step_scale)
- Scales the movement magnitude per particle
- In _step_numpy: `self.pos = (pos + step_size * self.role_step_scale[:, None] * movement) % SPACE`
- Default 1.0 = no change. >1 = bigger steps. <1 = smaller steps.

### Leader factor (role_influence)  
- Weights how much this particle's prefs count when computing neighbor means
- In social learning: instead of uniform mean, use weighted mean where each neighbor j contributes `role_influence[j] * prefs[j]`
- Default 1.0 = equal influence (current behavior)

### Researcher factor (role_gradient_noise)
- NOT applied in the core simulation — this is experiment-layer only
- The experiment's post_step_fn computes the true fitness gradient, adds per-particle Gaussian noise scaled by `role_gradient_noise[i]`, and applies the noisy gradient as a pref nudge
- Low noise = good researcher (sees the mountain clearly)
- High noise = poor researcher (noisy gradient estimate)

## Fitness Landscape (Mountain)

Defined in the experiment layer, not in the core simulation. Options:

### NK Landscape
- N = K (preference dimensions), each dimension interacts with K_epistasis others
- Rugged with tunable ruggedness via K_epistasis
- Has local peaks that trap conformist groups
- Computationally cheap to evaluate

### Gaussian Mixture Peaks
- Multiple peaks at known locations in [-1,1]^K with different heights
- Global peak is tallest but may be far from starting positions
- Local peaks are shorter but closer
- Easy to visualize and understand

### Chosen: Gaussian Mixture Peaks (more intuitive for org metaphor)

```python
# 1 global peak (tall, far) + 2-3 local peaks (shorter, closer to start)
peaks = [
    {'center': [0.8, 0.8, 0.8], 'height': 1.0, 'sigma': 0.3},   # global
    {'center': [-0.3, 0.2, -0.1], 'height': 0.6, 'sigma': 0.25}, # local 1
    {'center': [0.1, -0.4, 0.3], 'height': 0.5, 'sigma': 0.2},   # local 2
]
fitness(pref) = max over peaks of: height * exp(-||pref - center||^2 / (2*sigma^2))
gradient(pref) = d(fitness)/d(pref)  # points uphill
```

## Mountain Experiment Design

### Setup
- Particles start distributed near the origin (world knowledge says "start here")
- Memory field initialized with historical gradient beliefs (pointing toward local peak)
- Each particle assigned role weights from a configurable distribution

### Per-step
1. `sim.step()` runs full physics (movement, social learning, memory field)
2. Post-step: compute true mountain gradient at each particle's pref position
3. Add per-particle noise (scaled by role_gradient_noise)
4. Scale gradient magnitude by a global `gradient_strength` parameter
5. Apply noisy gradient to prefs: `prefs += gradient_strength * noisy_gradient`
6. Call `apply_post_processing(sim)`

### Metrics (observatory-style)
- **Peak fitness**: mean/max fitness across all particles
- **Summit fraction**: fraction of particles near global peak
- **Local trap fraction**: fraction stuck on local peaks  
- **Exploration rate**: mean pref change per step
- **Team coherence**: DBSCAN cluster tightness
- **Path diversity**: how many distinct routes are being explored
- **Memory-reality gap**: difference between memory field direction and true gradient

### Sweep variables
- Role distribution (% researchers vs leaders vs engineers)
- Social learning rate
- Memory strength (world knowledge influence)
- Gradient strength (mountain signal clarity)

## Params to add

```python
# In params.py:
'use_particle_roles': False,      # enable per-particle role heterogeneity
'role_step_scale_std': 0.0,       # std of log-normal step scale distribution
'role_influence_std': 0.0,        # std of log-normal influence distribution
```

When `use_particle_roles` is False, all role arrays stay at 1.0 — zero behavioral change.
