# 3D Visualizer Architecture Notes

## Key Facts

### Rendering Pipeline
- ModernGL + GLFW + imgui_bundle
- Left half: live 3D particles with orbit camera
- Right half: trail/velocity FBO display
- Shaders: PARTICLE_VERT/FRAG (point sprites), BOX (wireframe cube), LINE (neighbor edges), OVERLAY (2D divider)
- NO existing triangle-mesh shaders, NO surface normals, NO per-vertex scalar coloring

### Simulation (simulation3d.py)
- Completely separate from sim_2d_exp/simulation.py — different class, different params dict
- Positions are (n, 3) in [0, SPACE=1.0]^3 with periodic wrapping
- Preferences are (n, k) in [-1, 1]
- Physics engines: Numba (0), NumPy (1), PyTorch (2)
- NO role arrays (step_scale, influence, gradient_noise, visionary)
- NO memory field
- NO experiment controller integration

### Camera (camera3d.py)
- Orbit camera targeting [0.5, 0.5, 0.5] (center of unit cube)
- Distance 2.5, perspective projection
- get_right_up() for billboard geometry

### Landscape (experiments/landscape.py)
- GaussianPeakLandscape: fitness(prefs) and gradient(prefs) — works on (N, K) arrays
- CostLandscape: cost(prefs) — works on (N, K) arrays
- Both defined in [-1, 1]^K preference space
- Default mountain: global peak at [0.75]^K, local peaks at [-0.25, 0.15, 0.15] and [0.1, -0.4, -0.4]
- Default cost: ridges at [0.35]^K and [0.65]^K

## Key Design Decisions Needed

1. The 3D sim uses SPACE=[0,1]^3 for positions, but landscape is in [-1,1]^K for preferences.
   The mountain surface maps PREFERENCE space, not POSITION space.
   So the mountain mesh is a 2D grid over pref dims 0,1 with height = fitness(pref0, pref1, pref2_fixed).

2. For K=3, we need to pick 2 pref dims for the XY of the surface and use fitness as Z.
   Or: use pref[0], pref[1] as X,Y and fitness as Z, with pref[2] shown as color.

3. Particles need to be projected ONTO the mountain surface:
   Their X,Y come from prefs[0], prefs[1], Z = fitness(prefs).
   This is the "gravity holds them to the manifold" concept.

4. Cost overlay: same X,Y grid, but color = cost(prefs) instead of height.
   Could be a semi-transparent colored overlay on the mountain surface.
   Or: mountain color = blend of fitness (height) and cost (hue).

5. The 3D sim is a SEPARATE codebase from sim_2d_exp. Two options:
   a) Add mountain viz to the 3D_sim codebase (requires porting roles, memory, etc.)
   b) Add 3D rendering to the sim_2d_exp codebase (requires adding ModernGL rendering)
   c) Create a new hybrid that uses sim_2d_exp Simulation but 3D_sim rendering

   Option (c) seems cleanest — the 2D sim already has all the experiment machinery.
