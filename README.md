# Social Dynamics — Particle Simulation

Preference-directed particle simulation exploring emergent complex dynamics from simple interaction rules. Particles have preference vectors that determine attraction/repulsion patterns — the interplay between movement, neighbor topology, and preference evolution gives rise to rich self-organizing behavior.

![Example Society](ExampleSociety.png)

## Quick Start

Requires a terminal with GPU access. On macOS, if you see "Software Renderer" warnings, try Terminal.app instead of VS Code's integrated terminal.

```bash
cd gpu
python3 -m sim_2d_exp
```

### Dependencies

```bash
pip install numpy numba scipy glfw moderngl imgui-bundle PyOpenGL torch triangle
```

## Controls

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

## Right Panel Views

- **Trails** — temporal trail accumulation of particle positions
- **Velocity** — HSV velocity field (hue=direction, brightness=speed)
- **Causal** — causal tracking visualization (Cmd+drag to select seeds)
- **Pref2D** — preference space scatter plot (dim0 vs dim1)
- **Pref3D** — isometric projection of 3D preference space (120° axes)
- **MaxGrid** — per-dimension max preference grid visualization
- **Force** — force landscape with temporal variance toggle:
  - Max Pref (RGB) — strongest signal per dimension
  - Optimal Pref (RGB) — which preference vector maximizes force
  - Direction (HSV) — direction of maximum movement
- **Memory** — spatial memory field (RGB for preference deposit, HSV for flow deposit)

## Neighbor Modes

| Mode | Method | Description |
|------|--------|-------------|
| KNN | cKDTree / Hash Grid | Fixed number of nearest neighbors |
| KNN + Radius | cKDTree | KNN filtered by distance |
| Radius Only | Hash Grid | All neighbors within radius |
| Delaunay | Triangle library | Planar triangulation, h-hop expansion via sparse adjacency matrix power |

## Best Neighbor Selection

| Mode | Behavior |
|------|----------|
| Default | Highest raw preference value per dimension |
| Max Magnitude | Highest absolute preference value |
| Same-Sign Max Mag | Highest magnitude among same-sign neighbors |
| Boltzmann Softmax | Soft selection with temperature β (β=0: mean, β→∞: argmax) |

## Physics Engines

| Engine | Method | Scaling | Best for |
|--------|--------|---------|----------|
| Numba | Pairwise KNN (CPU JIT) | O(N·K) | Small N, exact physics |
| NumPy | Vectorized CPU | O(N·K) | Reference/debugging |
| PyTorch | Pairwise (MPS/CUDA) | O(N·K) | GPU acceleration |
| Grid Field | Smooth field (Gaussian) | O(N + G²) | Large N, smooth dynamics |
| Grid Max CPU | Max-pool + position tracking | O(N + G²·P) | Large N, discrete dynamics |
| Grid Max GPU | Fused max_pool2d (MPS/CUDA) | O(N + G²) | Large N + GPU |

## Key Features

### Interaction Models
- **Signal/Response split** — separate broadcast identity from reaction weights
- **Boltzmann softmax** — continuous temperature control between cooperative (averaged) and competitive (winner-take-all) dynamics
- **Ignore Self Pref** — remove self-preference weighting from compatibility
- **Inner product weighting** — full preference vector alignment modulates force

### Preference Evolution
- **Social learning** — positive (conformity) or negative (differentiation)
- **Quiet-dim differentiation** — differentiate along dynamically inactive dimensions
- **Graph diffusion** — multi-hop preference blending over the neighbor graph (hops + alpha controls)

### Spatial Memory Field
- **Deposit modes**: raw preferences, movement vectors, or normalized compat direction
- **Read**: multiplicative preference modulation (field strength)
- **Gradient forces**: Divergence (convergent) + Curl (circulatory) from field gradients
- **Flow following**: direct flow-following force for movement/compat deposits
- **Diffusion**: Gaussian blur for spatial spreading
- **Decay**: exponential forgetting
- See [spatial memory writeup](docs/spatial_memory.pdf) for analysis

### Position EMA
- Per-particle exponential moving average of position
- Three modes: Normal (track only), Forward Predict, Backward Lag
- Velocity visualization (cyan=history, red=prediction lines)

### Delaunay Triangulation
- Periodic boundary handling via boundary-only particle replication
- Fast triangulation using Shewchuk's Triangle library (~4x faster than scipy)
- h-hop neighbor expansion via sparse adjacency matrix power (A + A² + ... + Aʰ)
- Visualization shows true planar triangulation edges

### Analysis Tools
- **Force landscape** — probe-based visualization of force magnitude and optimal preference at each point in space
- **Shadow simulation** — run a perturbed copy to track chaotic divergence with LSB perturbation
- **Precision controls** — position/preference dtype (f16/f32/f64), mantissa bit truncation, discrete level quantization

## Documentation

- **[Simulator Overview (PDF)](docs/simulator_overview.pdf)** — comprehensive mathematical description of all models
- **[Spatial Memory Field (PDF)](docs/spatial_memory.pdf)** — detailed analysis of the memory field mechanism

## Modules

| Directory | Description |
|-----------|-------------|
| `sim_2d_exp/` | Main experimental simulation (2D) |
| `sim_2d_cuda/` | CUDA-optimized variant for Windows/NVIDIA |
| `3D_sim/` | 3D particle simulation |

```bash
python3 -m sim_2d_exp    # main experiment
python3 -m sim_2d_cuda   # CUDA variant
python3 -m 3D_sim        # 3D simulation
```
