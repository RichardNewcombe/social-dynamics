# Social Dynamics — 2D Particle Simulation

Preference-directed particle simulation exploring emergent complex dynamics from simple interaction rules. Particles have preference vectors that determine attraction/repulsion patterns — the interplay between movement, neighbor topology, and preference evolution gives rise to rich self-organizing behavior.

Must run from **Terminal.app** on macOS for hardware GPU acceleration (VS Code terminal may fall back to software renderer).

## Quick Start (experimental branch)

```bash
cd gpu
python3 -m sim_2d_exp
```

### Dependencies

```bash
pip install numpy numba scipy glfw moderngl imgui-bundle PyOpenGL torch
```

### Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| R | Reset simulation |
| Q / Esc | Quit |
| Scroll | Zoom (left panel) |
| Drag | Pan (left panel) |
| Up/Down | Adjust step size |
| +/- | Adjust social learning rate |
| Cmd+Drag | Select particles for causal tracking |

### Right Panel Views

- **Trails** — temporal trail accumulation of particle positions
- **Velocity** — HSV velocity field (hue=direction, brightness=speed)
- **Causal** — causal tracking visualization (Cmd+drag to select seeds)
- **Pref2D** — preference space scatter plot (dim0 vs dim1)
- **Pref3D** — isometric projection of 3D preference space (120° axes)
- **MaxGrid** — per-dimension max preference grid visualization
- **Force** — force landscape with 3 sub-modes:
  - Max Pref (RGB) — strongest signal per dimension
  - Optimal Pref (RGB) — which preference vector maximizes force
  - Direction (HSV) — direction of maximum movement

### Physics Engines

| Engine | Method | Scaling | Best for |
|--------|--------|---------|----------|
| Numba | Pairwise KNN (CPU JIT) | O(N·K) | Small N, exact physics |
| NumPy | Vectorized CPU | O(N·K) | Reference/debugging |
| PyTorch | Pairwise (MPS/CUDA) | O(N·K) | GPU acceleration |
| Grid Field | Smooth field (Gaussian) | O(N + G²) | Large N, smooth dynamics |
| Grid Max CPU | Max-pool + position tracking | O(N + G²·P) | Large N, discrete dynamics |
| Grid Max GPU | Fused max_pool2d (MPS/CUDA) | O(N + G²) | Large N + GPU |

### Key Features

- **Signal/Response split** — separate broadcast identity from reaction weights
- **Social learning** — positive (conformity) or negative (differentiation)
- **Quiet-dim differentiation** — differentiate along dynamically inactive dimensions
- **Force landscape** — visualize the force field across space
- **Shadow simulation** — run a perturbed copy to measure chaotic sensitivity
- **Precision controls** — truncate position/preference mantissa bits, quantize to discrete levels

## Other Modules

```bash
python3 -m sim_2d        # base refactored simulation
python3 -m sim_2d_cuda   # CUDA-optimized variant (Windows/NVIDIA)
python3 particle_net_viz.py  # ParticleNet interactive visualizer
```

## Legacy (original monolithic version)

```bash
python3 sim_gpu_compute.py
```
