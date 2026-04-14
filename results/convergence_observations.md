# Convergence Observations

## Figure 1: Headless Convergence (gradient nudge on prefs)

### Row 1: Baseline (no roles, social=0)
- Step 0: Particles uniformly scattered across [-1,1]^2, summit 2.2%
- Step 50: Some clustering on peaks, summit 3.8%
- Step 200: Particles clustering on multiple peaks, summit 6.2%
- Step 500: Clear clustering on local peaks, summit 6.4%
- Step 1000: Same as 500, summit 6.4% — STUCK on local peaks
- Step 2000: Same, summit 6.4% — no further progress

**Problem**: Most particles get trapped on local peaks. Only ~6% reach summit.
This is expected for baseline — no social learning to escape traps.

### Row 2: Roles + social=0.01
- Step 0: Same start, summit 2.2%
- Step 50: Summit 1.2% — DROPPED. Particles collapsing to a single cluster
- Step 200: Summit 0.0% — ALL particles collapsed to one tight cluster near center
- Step 500-2000: Summit 0.0% — stuck in a single blob, NOT at the summit

**MAJOR PROBLEM**: Social learning is causing ALL particles to collapse into a
single cluster at the deceptive peak (near center, height 0.75). The social
force overwhelms the gradient nudge. Once they cluster, they can't escape.

### Row 3: Roles + social=0.05
- Same pattern but even faster collapse
- By step 50 already a single tight cluster
- Summit 0.0% throughout

**DIAGNOSIS**: The social learning force (which pulls prefs toward neighbor means)
is much stronger than the gradient nudge (0.003 per step). Social learning
causes conformity collapse — all particles converge to one location, which
happens to be the deceptive peak (biggest basin of attraction near center).

## Figure 2: Visualizer-Mimic (sim.step + nudge + pos sync)

### Row 1: Baseline (social=0)
- Identical to headless — expected since social=0 means sim.step() only does
  repulsion in pos space, and pos sync maps back to surface. Summit 6.4%.

### Row 2: Roles + social=0.01
- Slightly different from headless: summit goes 2.2% → 2.6% → 2.0% → 1.8% → 1.8%
- Particles still collapse but less severely than headless.
- Still stuck at 1.8% — not reaching summit.

### Row 3: Roles + social=0.05
- Same conformity collapse as headless, summit 0.0% by step 50.

## Figure 3: sim.pos Debug
- sim.pos is correctly synced to the mountain surface coordinates after the fix.
- Particles in pos-space form a thin sheet — correct for mountain surface.

## Figure 4: Top-Down 2D View (headless)
- Row 1 (baseline): Particles climb to nearest local peaks. Multiple clusters.
  ~6% reach summit. Most trapped on deceptive peak near center.
- Row 2 (social=0.01): ALL particles collapse into ONE tight cluster at
  deceptive peak by step 200. Social conformity kills diversity.
- Row 3 (social=0.05): Even faster collapse by step 50.

## KEY ISSUES IDENTIFIED

1. **Conformity collapse**: Social learning (even at 0.01) causes all particles
   to converge to a single cluster at the deceptive peak. Gradient nudge (0.003)
   is too weak relative to social to maintain diversity or escape.

2. **Baseline only reaches 6.4%**: Without social learning, particles just
   climb to nearest peak. Most start closer to local peaks than summit.

3. **No role differentiation visible**: Visionary fraction (10%, mean 0.15)
   too weak to pull group toward summit.

4. **The real problem**: The simulation dynamics don't produce interesting
   convergence. Particles either scatter to local peaks (no social) or
   collapse to one deceptive peak (any social > 0).
