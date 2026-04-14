# Knowledge Manifold Test Observations

## What's Working
1. **Knowledge surface grows from zero** — Step 0 is flat, by step 1000 there are clear peaks
2. **Particles cluster into teams** — social learning creates ~8-10 distinct clusters
3. **Knowledge deposits where teams are** — green peaks in the knowledge heatmap correspond to particle cluster locations
4. **Diffusion spreads knowledge** — the green patches are broader than just the cluster footprints
5. **Fitness ceiling is respected** — peak knowledge (0.645) is well below 1.0
6. **3D views look correct** — green knowledge surface rises from flat toward the blue wireframe fitness ceiling
7. **Particles sit ON the knowledge surface** — no off-surface particles

## Issues Found
1. **Premature saturation** — Coverage plateaus at 55% by step 1000 and never improves. The system reaches equilibrium too fast.
2. **Peak knowledge only 0.645** — The global summit (fitness=1.0) is never reached. Particles cluster at foothills.
3. **Social conformity collapse** — Pref diversity drops from 0.57 to 0.49 quickly, meaning particles are clustering but not exploring further.
4. **No team merging** — The ~8-10 clusters are stable and never merge or migrate toward better peaks.
5. **Gap map shows the summit (0.6, 0.7) is the largest unexplored region** — the darkest red in the gap plot is exactly where the global optimum is.
6. **Knowledge is too diffuse** — diffusion_sigma=0.8 spreads knowledge broadly but thinly. The knowledge manifold looks like a gentle rolling landscape rather than sharp peaks where teams have explored.

## Root Causes
- **No incentive to move toward higher knowledge** — particles follow noisy hidden gradient but social learning dominates, keeping them in local clusters
- **Gradient noise too high relative to nudge rate** — with noise=4.0 and nudge=0.002, the signal is too weak to pull teams toward the summit
- **Diffusion too aggressive** — sigma=0.8 on a 64-grid means knowledge spreads ~5 cells per step, washing out the spatial structure
- **No visionary/frontier attraction** — nothing pulls teams out of local optima toward the undiscovered summit

## Suggested Fixes
1. Reduce diffusion_sigma to 0.3-0.4 for sharper knowledge peaks
2. Increase nudge_rate or decrease gradient_noise for stronger directional signal
3. Add weak attraction toward the knowledge gradient (climb what you know)
4. Add exploration perturbation to escape local clusters
5. Consider the "revenue from knowledge height" mechanism to incentivize climbing
