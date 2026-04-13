# Rugged Landscape Observations

## Landscape Structure
- The fitness landscape now has visible complexity: a prominent deceptive peak near center-left (height ~0.75), the global summit in the upper-right corner (gold star), a ridge peak between them, and a distant peak in the lower-left.
- The spectral noise creates visible roughness — the surface has ripples, ridges, and subtle undulations across the entire terrain.
- Multiple minor peaks create foothills visible as small bumps on the surface.
- The summit fitness at center is 0.627 (due to the w_major=0.55 weighting), meaning the composite peak is lower than 1.0.

## Cost Terrain
- The main cost ridge is clearly visible as a tall red spike between the origin and the summit.
- Multiple smaller cost bumps create a complex cost landscape.
- The deceptive peak sits in a low-cost zone — cheap to stay there, expensive to leave toward the summit.

## Experiment Results (much harder now!)
- Baseline: 6.4% summit (was 29.4% with simple landscape)
- All Visionaries: 100% summit (still solves — visionary signal is too strong)
- Specialist: 0.2% summit (stuck on local peaks)
- Generalist: 9.0% summit (was 30.2%)
- Rare Vis + Generalist: 13.2% summit (was 33.8%)
- Specialist + Differentiation: 12.4% summit (was 13.2%)

## Key Takeaway
The landscape is now genuinely challenging. Baseline performance dropped from 29% to 6%. But all-visionaries still trivially solves it because the visionary signal points directly at the summit regardless of terrain complexity. This reinforces the need for visionary rarity.
