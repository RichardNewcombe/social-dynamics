# Redesign Diagnostic Observations

## Landscape Images
- Fitness landscape IS rugged - lots of texture, multiple peaks visible
- Cost landscape has a massive hot zone in the center-right area (cost up to 4.5!)
- The cost ridges are WAY too strong relative to fitness (max fitness ~0.55, max cost ~4.5)
- Effective landscape (fitness - 0.3*cost) is deeply negative in the center, only positive in corners
- The summit (star) sits at the edge of a high-cost zone
- High-frequency cost noise IS visible (diagonal striping pattern) - good

## Key Problems Identified
1. **Cost scale mismatch**: Cost peaks at 4.5, fitness peaks at ~0.55. Even with cost_weight=0.3, 
   the cost gradient dominates: 0.3 * 4.5 = 1.35 >> 0.55 fitness
2. **Random fixed neighbors**: The test uses random fixed neighbor assignments, not spatial neighbors.
   This means "teams" are random groups, not coherent spatial clusters. In the real sim, spatial
   neighbors form organically and share similar positions = more coherent gradient signals.
3. **NOISY-SOLO is surprisingly good**: With noise=2.0, individual movement is essentially random walk.
   On a rugged landscape, random walk explores well and finds local peaks by chance. The noise
   acts as exploration, not as a handicap.
4. **TEAM-NO-COST is the best**: This confirms that team aggregation DOES help (0.40 vs 0.36 for solo).
   The problem is that cost is too aggressive.
5. **TEAM-BIG is worst**: With 50 random neighbors, everyone converges to the same direction = 
   conformity collapse. Need spatial neighbors that change over time.

## Fixes Needed
1. Scale down cost ridge heights dramatically (max ~0.5, not 3.0-4.5)
2. Or reduce cost_weight default (0.05 instead of 0.3)
3. The landscape ruggedness is good - keep it
4. The team advantage IS there (TEAM-NO-COST > SOLO) - just need to not kill it with cost
