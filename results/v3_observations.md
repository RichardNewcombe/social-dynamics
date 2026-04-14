# V3 Knowledge Manifold Test Observations

## Key findings from diagnostic images

### Heatmaps (knowledge_growth_v3.png)
- Knowledge grows as a broad, diffuse field — no thin spires (structural constraint working)
- Particles cluster into ~5-8 groups by step 500
- Coverage saturates at ~36% by step 500 and stays flat
- Peak knowledge only reaches 0.24 — very low, the manifold barely rises
- Reward-colored particles (pink = high reward) are concentrated in a few clusters

### 3D Views (knowledge_3d_v3.png)
- The knowledge surface (green) is nearly flat — barely distinguishable from the floor
- The fitness wireframe (blue) is visible but the knowledge surface is nowhere close to it
- Particles sit on the almost-flat knowledge surface
- By step 3000, the surface has gentle bumps but nothing dramatic

### Metrics (knowledge_metrics_v3.png)
- Coverage: rapid rise to 0.36 by step 500, then flat — no further exploration
- Peak knowledge: 0.24 — very low, structural constraint + diffusion is spreading knowledge too thin
- Pref std: drops from 0.58 to 0.50 — moderate clustering, not collapsing
- Mean noise: starts at 3.04, drops to 3.00 — adaptive noise is NOT activating! Everyone is "productive" because the threshold is too low
- Mean fitness at particles: 0.65 — particles are in decent fitness regions but can't raise knowledge there

### Adaptive Dynamics (knowledge_adaptive_v3.png)
- Reward distribution is bimodal: ~40 particles near 0 (stuck), ~70 particles near 0.12 (productive)
- Noise distribution: ALL particles at exactly 3.0 — adaptive noise never triggers
- The reward threshold (WRITE_RATE * 0.1 = 0.0005) is too low — everyone exceeds it

## Root causes
1. **Structural constraint too aggressive**: support_radius=3, max_slope=0.15 means knowledge can't rise much above the local mean. Combined with diffusion_sigma=1.5, the knowledge gets spread so thin it can never build height.
2. **Adaptive noise not working**: threshold too low, so noise never increases for stuck particles
3. **Write rate vs diffusion balance**: depositing 0.005 per particle per step, but diffusion spreads it across ~9 cells (sigma=1.5), so effective deposit per cell is tiny
4. **No knowledge climb**: removed the knowledge gradient climbing, so particles don't move toward their own peaks
