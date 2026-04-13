# Coordinate System Debug

## 3D Sim (simulation3d.py)
- Positions: [0, SPACE]^3 where SPACE = 1.0 (from grid3d.py)
- Prefs: [-1, 1]^k
- Colors: (prefs[:,:3] + 1.0) * 0.5 → [0, 1] for RGB

## Mountain Mesh (mountain_mesh.py)
- Mesh vertices: pref [-1,1] → (pref + 1) * 0.5 → [0, 1] for X,Y
- Fitness → normalized to [0, z_scale] for Z

## Particle Projection (project_particles_to_surface)
- Takes prefs in [-1, 1]
- Maps: x = (prefs[:,0] + 1) * 0.5, y = (prefs[:,1] + 1) * 0.5
- Z = normalized fitness

## 3D main.py mountain mode (line ~484-493)
- Uses sim.strategy (if available) or sim.prefs
- Passes to project_particles_to_surface
- Writes result to vbo_pos

## BUG 1: Axis mirroring
- The mesh maps pref[0]→X, pref[1]→Y
- But np.meshgrid(lin, lin) returns gx varying along columns (axis=1) and gy along rows (axis=0)
- The mesh Y axis corresponds to gy which varies along rows
- When the 3D sim renders, its camera/projection may have Y pointing differently
- Need to check if the 3D sim's Y axis matches the mesh's Y axis

## BUG 2: Particles stop moving
- Mountain mode ONLY overrides the render positions (vbo_pos)
- The sim.step() still runs on the original sim.pos
- But wait — does the sim.step() use sim.pos for physics? YES
- So the physics should still work... unless mountain mode is interfering
- Check: is mountain mode writing back to sim.pos? NO, it writes to vbo_pos only
- The particles should still move in pref space, and the projection should show them moving on the mountain
- UNLESS: the prefs aren't changing because there's no gradient force in the 3D sim
- The 3D sim has no post_step_fn, no landscape gradient — it's just the raw social dynamics
- With default social=0.0, prefs don't change at all!
- That's the bug: particles don't move because prefs are static with no social learning
