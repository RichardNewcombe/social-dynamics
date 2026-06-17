# Off-Surface Particle Bug: Root Cause Found

## Two root causes identified:

### 1. K > 2 dimension mismatch (MAJOR)
- The mountain mesh evaluates fitness with dims 2+ set to 0.0 (`other_pref_val=0.0`)
- But particles have random values in pref[2] (and higher dims when K > 2)
- The landscape peaks have non-zero centers in dim 2 (e.g., summit at [0.72, 0.78, 0.65])
- So `fitness(0.5, 0.5, 0.0) = 0.180` but `fitness(0.5, 0.5, 0.3) = 0.384` — a huge difference!
- **31% of particles** have |fitness difference| > 0.05
- **18% of particles** have |fitness difference| > 0.10
- Y offset in render space ranges from -0.41 to +0.41 (mesh is only 0.5 tall!)
- This means particles can be nearly a full mesh height above or below the surface

### 2. f_min/f_max normalization mismatch (MINOR)
- Mesh uses 64x64 grid → f_min=0.0555, f_max=0.5223
- Projection uses 50x50 grid → f_min=0.0559, f_max=0.5226
- Small difference (~0.0004) but contributes to slight offset

## The side view slice (bottom right) clearly shows it:
- Black line = mesh surface (fitness evaluated with dim2=0)
- Red dots = particles (fitness evaluated with their actual dim2 values)
- Blue X's = same particles but with dim2 forced to 0
- Red dots scatter widely above and below the black line
- Blue X's sit exactly on the black line

## Fix options:
1. **When projecting particles, set dims 2+ to 0** before evaluating fitness
   - Simple, but means the mountain doesn't reflect the actual K-dimensional landscape
2. **Set K=2 for mountain mode** — eliminate the extra dimensions entirely
3. **Project using only dims 0,1 for fitness evaluation** — evaluate fitness
   at (pref[0], pref[1], 0, ..., 0) for the Y coordinate, regardless of actual pref values
   - This is the most correct fix: the mesh shows a 2D slice, particles should
     be projected onto that same 2D slice for visualization
