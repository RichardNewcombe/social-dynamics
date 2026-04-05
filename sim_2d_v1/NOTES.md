# sim_2d v1

Snapshot date: 2026-03-11

## What's in this version

Initial refactor of `sim_gpu_compute.py` into a modular package:

- `params.py` — Global parameter dict + constants
- `shaders.py` — All GLSL shader strings
- `spatial.py` — Spatial hash grid + neighbor finding (numba JIT)
- `physics_numba.py` — Numba JIT physics kernels (`_step_inner_prod_avg`, `_step_per_dim`)
- `physics_torch.py` — PyTorch vectorized physics
- `simulation.py` — Simulation class (state + stepping + rendering data)
- `renderer.py` — GLFW / moderngl / imgui window + main loop
- `__main__.py` — Entry point (`python -m sim_2d`)

## Features added on top of original sim_gpu_compute.py

1. **"Binary d0 + noise" preference init** (pref_dist=4): 50% particles get dim 0 = +1, rest = -1. Remaining dims filled with uniform noise in [-eps, eps]. Eps controlled by slider.

2. **Best Neighbor mode combo** (replaces `best_by_magnitude` checkbox):
   - Mode 0 — Default: highest raw preference value
   - Mode 1 — Max Magnitude: highest absolute preference value
   - Mode 2 — Same-Sign Max Mag: highest magnitude among same-sign neighbors only. When no same-sign neighbor exists, the dimension contributes zero to movement and direction memory decays.

3. Bug fixes for race conditions: all three engines (Numba/NumPy/PyTorch) handle the no-valid-neighbor edge case consistently — zero contribution + direction memory decay.
