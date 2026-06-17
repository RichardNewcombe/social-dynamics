# Dual-Space Mountain Mode Test Results

## Key Findings

1. **Strategy enabled**: True. `sim.strategy` is a separate (500, 3) array from `sim.prefs` (500, 3).
2. **Surface adherence**: Max surface offset = 0.00000000 at all timesteps. Particles sit exactly on the mountain.
3. **Dual-space divergence**: Prefs and strategy start at divergence ~0.7 (coupling=0.5 blends prefs with random), then converge to ~0.12 as the coupling drift pulls them together.
4. **Fitness climbing**: Mean fitness rises from 0.21 to 0.48 over 2000 steps — strategy is navigating the landscape.
5. **Diversity**: Pref std drops from 0.57 to 0.20 (social conformity), strategy std drops from 0.40 to 0.21 (gradient convergence). Strategy diversity stays slightly higher than pref diversity — the two spaces are genuinely separate.
6. **Summit distance**: Drops from ~1.4 to ~1.3 — slow progress, but the gradient is working. The summit is at [0.72, 0.78, 0.65] which is far from the initial uniform spread.

## Top-down view observations
- Step 0: Red (strategy) and blue (prefs) are scattered uniformly, overlapping
- Step 100: Both starting to cluster, strategy slightly more concentrated
- Step 500: Clear separation — strategy clusters near landscape peaks, prefs cluster differently due to social dynamics
- Step 1000-2000: Strategy forms distinct clusters near peaks; prefs form their own social clusters. The two spaces are visibly different.

## Architecture verification
- Phase 1 (sim.step): social dynamics in preference space → builds neighbor graph
- Phase 2 (strategy_step): team-aggregated mountain navigation in strategy space
- Projection: sim.strategy → mountain surface positions → sim.pos
- The neighbor graph from Phase 1 determines who talks to whom in Phase 2
