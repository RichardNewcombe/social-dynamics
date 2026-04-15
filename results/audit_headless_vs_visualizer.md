# Audit: Headless Experiments vs Visualizer Experiments

## How the Visualizer Runs the Simulation

In `renderer.py`, the main loop (line 548-559):

```python
if running_sim:
    spf = params['steps_per_frame']
    reuse = params['reuse_neighbors']
    for sub in range(spf):
        sim.step(reuse_neighbors=(reuse and sub > 0))
        if exp_ctrl is not None:
            exp_ctrl.on_step(sim)
```

Key facts:
- Creates `sim = Simulation()` once (line ~480 in do_reset)
- Calls `sim.step()` — the FULL step method including memory field, physics dispatch, post-processing
- `exp_ctrl.on_step(sim)` runs AFTER `sim.step()` — purely observational (reads metrics)
- The controller's `post_step` callback is `None` for all three observatory experiments
- The controller's `init` callback only sets initial prefs (no physics bypass)
- Supports `reuse_neighbors` for sub-steps within a frame
- Uses whatever `physics_engine` the user has selected in the GUI

## How the Headless Runner Runs the Simulation

In `runner.py`, the `run_experiment()` function (line 222-237):

```python
sim = Simulation()
for step in range(1, max_steps + 1):
    sim.step()
    if post_step_fn is not None:
        post_step_fn(sim, step)
    if check_fn is not None and check_fn(sim, step):
        ...
```

Key facts:
- Creates `sim = Simulation()` — same class, same constructor
- Calls `sim.step()` — the FULL step method, identical call
- `post_step_fn` runs AFTER `sim.step()` — same position as visualizer

## DIVERGENCES FOUND

### 1. `reuse_neighbors` — MINOR
- **Visualizer**: passes `reuse_neighbors=(reuse and sub > 0)` — only reuses on sub-steps 1+ within a frame
- **Headless**: calls `sim.step()` with no argument → `reuse_neighbors=False` (default)
- **Impact**: The headless runner does a fresh neighbor search every step. The visualizer also does a fresh search on the first sub-step of each frame. Only when `steps_per_frame > 1` does the visualizer reuse neighbors on sub-steps 2+. Since headless runs use `steps_per_frame=1` (from SAFE_DEFAULTS), this is **functionally identical**.

### 2. `physics_engine` — INTENTIONAL DIFFERENCE
- **Visualizer**: uses whatever engine the user has selected (could be Numba, NumPy, PyTorch, Grid, GPU)
- **Headless**: forces `physics_engine=1` (NumPy) via SAFE_DEFAULTS/experiment overrides
- **Impact**: This is **intentional** — headless runs need to work without GPU/display. The NumPy engine implements the same physics as the other engines (verified by shadow sim divergence testing in the visualizer). However, the Numba engine (engine 0) is the default in the visualizer, not NumPy.

### 3. Experiments 1 & 2: GRADIENT INJECTION (post_step_fn) — MAJOR DIVERGENCE
- **Visualizer (controller.py)**: Experiments 1 and 2 have `post_step=None`. They are pure observatories — NO gradient nudges, NO external forces. They only read metrics.
- **Headless (exp1, exp2)**: Both inject gradient nudges via `post_step_fn`:
  - exp1: `hidden_target_gradient()` nudges particles toward a hidden target vector
  - exp2: `make_post_step()` attracts particles toward nearest niche centre
- **Impact**: **The headless experiments are fundamentally different from the visualizer experiments.** The headless versions inject artificial external forces that bypass the simulation's own physics, while the visualizer versions are pure observatories that only measure what the simulation does naturally.

### 4. Experiment 3: DIFFERENT DESIGN
- **Visualizer (controller.py)**: Pure observatory with manual "Trigger Shock" button. Measures field energy, culture-strategy alignment, cultural inertia. No automatic shock.
- **Headless (exp3)**: Automated two-phase design — 2500 settling steps, then automatic pref flip, then measures adaptation time. Manages its own Simulation lifecycle (doesn't use runner.py).
- **Impact**: **Different experimental design.** The headless version is a scripted experiment with a win condition (adaptation threshold). The visualizer version is an interactive observatory. Both use the full simulator, but they measure different things.

### 5. Experiment 3: Custom sim lifecycle — MINOR
- **Headless exp3**: Creates Simulation directly, manages params snapshot/restore itself (doesn't use `run_experiment()`)
- **Impact**: Still calls `sim.step()` on the real Simulation class. No physics shortcut. The custom lifecycle is needed for the two-phase design.

## SUMMARY

| Aspect | Visualizer | Headless | Match? |
|--------|-----------|----------|--------|
| Simulation class | `Simulation()` | `Simulation()` | YES |
| Step method | `sim.step()` | `sim.step()` | YES |
| Full physics pipeline | Yes (memory, physics, post-proc) | Yes (same) | YES |
| Physics engine | User-selected (default: Numba) | Forced NumPy | INTENTIONAL |
| Exp 1 design | Pure observatory (no forces) | Gradient injection toward target | **NO** |
| Exp 2 design | Pure observatory (no forces) | Gradient injection toward niches | **NO** |
| Exp 3 design | Interactive shock button | Automated two-phase script | **DIFFERENT** |
| Metrics computed | DBSCAN clustering, coverage, entropy | Target distance, niche occupancy | **DIFFERENT** |

## CONCLUSION

The headless experiments use the FULL simulator (same Simulation class, same sim.step() call, same physics pipeline). There are no physics shortcuts.

However, the headless experiment DESIGNS are fundamentally different from the visualizer observatory designs:
- The headless versions (exp1, exp2) inject artificial gradients that the visualizer versions don't
- The headless versions have win conditions; the visualizer versions are open-ended dashboards
- The metrics computed are different

This is because the headless scripts were written with the original "time-to-solve" framing, while the controller.py was rewritten to the "organizational observatory" design. **The headless scripts were never updated to match the observatory design.**
