# 3D Visualizer Audit — Integration Checklist

## Checked Integration Points

1. **Landscape imports**: `make_default_landscape` and `make_default_cost_landscape` imported at line 56-58. Guarded by `_HAS_LANDSCAPE`. OK.

2. **Mountain mesh build**: `build_mountain_mesh()` creates both `mountain_landscape` and `cost_landscape` at lines 307-308. Uses the new rugged factories. OK.

3. **CLI --mountain defaults** (lines 1099-1123): Sets `strategy_enabled=True`, `strategy_k`, `pref_strategy_coupling=0.5`, `use_particle_roles=True`, high gradient noise (2.0), visionary params, cost_weight=0.3, momentum=0.3, explore params. OK.
   - NOTE: CLI sets cost_weight=0.3 but the param default is 0.1. The CLI override is intentional for mountain mode.

4. **Simulation reset**: `reset()` calls `_init_roles()` and `_init_strategy()` at lines 342-343. OK.

5. **do_reset()**: Syncs initial positions to surface using strategy if available (lines 422-429). OK.

6. **Main loop Phase 1**: `sim.step()` at line 490 handles social dynamics and builds neighbor graph (`self.nbr_ids`). OK.

7. **Main loop Phase 2**: When `mountain_mode` and `mountain_landscape` are set (line 495):
   - If `sim.strategy is not None`: calls `sim.strategy_step(gradient_fn, summit_center, cost_landscape)`. OK.
   - Fallback: manual gradient nudge on prefs (original Exp 4 behavior). OK.

8. **Position sync**: After strategy_step, projects mountain_coords to surface and writes to `sim.pos[:, :3]` (lines 529-532). OK.

9. **GPU upload**: Mountain mode reads `sim.pos[:, :3]` directly (line 540). OK.

10. **strategy_step()**: Uses `self.nbr_ids` from Phase 1 for team aggregation (line 617). Includes cost-aware gradient, noise without pre-normalization, visionary blend, team aggregation, momentum, exploration, memory, and coupling drift. OK.

11. **imgui controls**: All new params have sliders:
    - Strategy Enabled, Coupling, Strat Step (lines 847-859)
    - Cost Weight, Momentum, Explore Prob, Explore Radius (lines 860-875)
    - Use Roles, Influence Std, Step Scale Std, Grad Noise Mean/Std, Visionary Mean/Std (lines 880-908)
    OK.

## Issues Found

1. **Grad Noise Mean slider max is 2.0** (line 894) — but the default for mountain mode is 2.0, so the user can't increase it beyond the default. Should be higher (e.g., 5.0).

2. **No "Strat K" slider** — strategy_k is set from CLI but not adjustable in imgui. Minor — usually matches k.

3. **cost_weight default mismatch**: param default is 0.1, CLI --mountain sets 0.3. This is fine (CLI is the intended entry point for mountain mode), but could confuse if someone enables mountain mode via imgui checkbox without --mountain flag.

4. **Coupling drift uses strategy_k == k check** (line 729) — if strategy_k != k, coupling drift is silently skipped. This is correct behavior but not documented in imgui.
