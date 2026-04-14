# Knowledge Manifold v2 Test Observations

## Major Improvements Over v1
1. **Peak knowledge = 0.993** (was 0.645) — nearly reaching the global summit!
2. **Knowledge peaks are sharp and distinct** — reduced diffusion creates clear spikes where teams have explored
3. **Coverage grows steadily** — 13% → 37% over 5000 steps, still climbing (was saturating at 55% in v1 due to over-diffusion)
4. **Skill diversity maintained** — pref_std stays around 0.53-0.58, no conformity collapse
5. **3D views look dramatic** — sharp knowledge spires rising toward the fitness ceiling, with clear gaps between explored and unexplored regions

## Remaining Issues
1. **Coverage still low (37%)** — most of the landscape remains unexplored
2. **Knowledge is concentrated in ~10 spikes** — teams dig deep but don't spread out enough
3. **The global summit area (0.6, 0.7) has some teams** — visible in the upper-right cluster at step 5000
4. **Mean knowledge is only 0.14** — the manifold is mostly flat with a few tall spires
5. **Gap map shows the summit region has moderate gap** — teams are there but haven't fully filled it

## The 3D Visualization Looks Great
- Step 0: flat green surface, particles scattered on the floor
- Step 1000: sharp green spires rising toward the blue wireframe ceiling
- Step 5000: taller, more numerous spires, some approaching the ceiling height
- The visual metaphor of "knowledge growing toward the hidden truth" is very clear

## Parameter Tuning Notes
- diffusion_sigma=0.3 gives much sharper features than 0.8
- decay=0.9995 provides very slow knowledge loss (preserves discoveries)
- gradient_noise=3.0 with nudge_rate=0.003 gives usable team signal
- knowledge_climb_rate=0.001 provides gentle attraction to known peaks
- explore_prob=0.003 with radius=0.2 provides occasional jumps
