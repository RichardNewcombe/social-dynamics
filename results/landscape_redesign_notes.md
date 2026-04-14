# Landscape Redesign Notes

## Current Problems
1. Smooth Gaussians → gradient is reliable → individuals can solo climb
2. Cost is visual only → no simulation-time impact
3. Noise is low-frequency → no high-frequency ruggedness
4. No cost penalty on movement → all paths equally cheap
5. Gradient noise too low → averaging across team barely helps

## Design Goals
1. **Groups > Individuals**: High gradient noise + team aggregation = reliable signal only via groups
2. **Few viable routes**: Cost-penalized gradient creates narrow "cheap corridors" through expensive terrain
3. **Perturbation**: Exploration noise, occasional random jumps, visionary scouting
4. **Cost in simulation**: Effective gradient = fitness_gradient - cost_gradient (avoid expensive regions)

## Implementation Plan

### A. Landscape Changes (landscape.py)
- Increase spectral noise: more frequencies, higher amplitudes, especially high-freq
- Add "cliff" features: sharp sigmoid-based drop-offs that create walls
- Make minor peaks taller relative to major → more deceptive local optima

### B. Cost Integration (strategy_step)
- Compute cost gradient at each particle's position
- Effective movement direction = fitness_gradient - cost_weight * cost_gradient
- This naturally steers particles away from expensive regions
- High-frequency cost noise creates narrow cheap corridors

### C. Cost Landscape Changes
- Add high-frequency sinusoidal cost noise (like spectral noise in RuggedLandscape)
- This creates a "Swiss cheese" cost field with narrow cheap paths
- Cost ridges should block obvious routes to summit

### D. Gradient Noise Increase
- Default role_gradient_noise_mean should be much higher (e.g., 1.5-2.0)
- Individual gradient sensing is nearly random
- Team of 20 averaging → noise reduces by sqrt(20) ≈ 4.5x → usable signal
- This is THE key mechanism for group advantage

### E. Exploration Mechanisms
- Random perturbation: small probability of random jump per step
- Momentum/inertia: particles remember recent good directions
- Scouting: some particles explore randomly, report back via team aggregation
