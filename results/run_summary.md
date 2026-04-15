# Headless Experiment Run Summary

## Experiment 1: Hidden Target Search
- **Status**: All 21 runs completed successfully (7 social values x 3 trials)
- **All solved**: Yes
- **Key findings**:
  - social=-0.030: mean 8561 steps (32.6s) — strong differentiation slows convergence dramatically
  - social=-0.010: mean 3556 steps (13.2s) — moderate differentiation still slow
  - social=-0.003: mean 57 steps (0.2s) — near-zero differentiation works well
  - social=0.000: mean 44 steps (0.1s) — no social learning baseline
  - social=+0.003: mean 29 steps (0.1s) — slight conformity is fastest
  - social=+0.010: mean 49 steps (0.2s) — moderate conformity fast
  - social=+0.030: mean 192 steps (0.7s) — strong conformity slightly slower

## Experiment 2: Multi-Niche Coverage
- **Status**: All 15 runs completed (5 conditions x 3 trials)
- **Key findings**:
  - Uniform s=+0.01: 0/3 solved — conformity kills diversity
  - Uniform s=-0.01: 0/3 solved — differentiation alone not enough
  - QuietDim s=+0.01: 2/3 solved (mean 794 steps) — best with social learning
  - QuietDim s=-0.01: 0/3 solved — differentiation in quiet dims fails
  - No social s=0.0: 3/3 solved (mean 191 steps) — baseline surprisingly good

## Experiment 3: Ghost Colony Escape
- **Status**: All 15 runs completed (5 memory_strength values x 3 trials)
- **All solved**: Yes
- **Key findings**:
  - memory_strength=0.0: mean 1 adapt step — instant adaptation (no memory)
  - memory_strength=1.0: mean 2 adapt steps — minimal inertia
  - memory_strength=3.0: mean 1 adapt step — surprisingly fast
  - memory_strength=6.0: mean 18 adapt steps — inertia emerging
  - memory_strength=10.0: mean 243 adapt steps — strong ghost colony effect

## Issues Found
- No crashes or errors in any experiment
- Exp3 shows the expected monotonic trend at high memory strengths (ghost colony effect)
- Exp3 at memory_strength=3.0 shows anomalous fast adaptation (1 step) — may need investigation
