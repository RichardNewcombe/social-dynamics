# V3 Tuned Knowledge Manifold Observations

## Results Summary

| Metric | Before Tuning | After Tuning |
|--------|--------------|--------------|
| Peak knowledge | 0.24 | 0.94 |
| Coverage | 36% | 68% |
| Mean fitness at particles | 0.65 | 0.70 |
| Adaptive noise | Never activated | Spikes at steps 100-500 |

## 3D Views
- Dramatic improvement: knowledge surface now has tall peaks reaching toward the fitness ceiling
- Multiple distinct knowledge spires visible, corresponding to team clusters
- The structural constraint creates natural "mountain" shapes with broad bases supporting tall peaks
- Particles cluster at the peaks of the knowledge surface
- By step 3000, several peaks are nearly touching the fitness wireframe ceiling

## Heatmaps
- Clear team clusters with deep red knowledge hotspots (0.8-1.0)
- Knowledge diffuses outward from clusters, creating smooth gradients
- Multiple independent peaks, not just one dominant cluster
- Coverage at 68% means most of the domain has some knowledge

## Metrics
- Coverage rises rapidly to 68% and stabilizes
- Peak knowledge reaches 0.94 — very close to the fitness ceiling
- Adaptive noise shows brief spikes early on (steps 100-500) then settles — teams find productive positions quickly
- Pref std drops from 0.59 to ~0.54 — moderate clustering, good diversity maintained
- Mean fitness at particles: 0.70 — particles are in high-fitness regions

## Remaining Issues
- Adaptive noise still mostly at baseline (3.0) — the threshold may still be too easy
- Coverage plateaus — no mechanism to push teams into unexplored territory after initial clustering
- Knowledge gradient climbing may be pulling particles too aggressively toward existing peaks
