# Redesign V2 Observations

## Cost Calibration
- Cost now peaks at ~1.7 (down from 4.5), much more reasonable
- Effective landscape shows the summit area (star) is in a positive zone (~0.3)
- The high-frequency cost noise creates visible diagonal striping = narrow corridors
- Cost still has a big hot zone in center-right, creating a meaningful obstacle

## Fitness Curves
- TEAM-NO-COST (purple, 0.40) clearly beats everything - team aggregation works!
- SOLO-CLEAN (green, 0.36) and NOISY-SOLO (red, 0.36) plateau at similar levels
- TEAM with cost (blue, 0.34) is still below solo - cost still slightly too aggressive
- TEAM-BIG (orange, 0.34) shows conformity collapse with random neighbors

## Trajectory Plots
- SOLO-CLEAN: particles spread across landscape, find local peaks
- NOISY-SOLO: random walk exploration, particles scattered everywhere
- TEAM-NO-COST: clear clustering near peaks, some particles near summit
- TEAM with cost: some clustering but pushed away from cost zones
- TEAM-BIG: strong conformity collapse, big cluster in one spot

## Key Insight
The test uses RANDOM fixed neighbors, not spatial neighbors. In the real sim:
- Spatial neighbors share similar local gradient info (coherent signal)
- Random neighbors across the landscape have conflicting signals
- This makes the test underestimate team advantage

## What to commit
- The landscape changes are good (rugged terrain, calibrated cost)
- The strategy_step normalization fix is correct (don't normalize before averaging)
- The cost_weight default should be lower (0.1 instead of 0.3)
- Need to test in real visualizer with spatial neighbors
