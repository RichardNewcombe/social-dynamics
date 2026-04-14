"""
Headless test of the redesigned mountain simulation.

Runs two scenarios side-by-side:
  A) TEAM mode: dual-space with roles, team aggregation, cost-aware movement
  B) SOLO mode: same landscape but gradient_noise=0.1 (low), no team aggregation

Renders:
  1. The new rugged landscape surface (top-down heatmap)
  2. The cost landscape (top-down heatmap)
  3. Effective landscape (fitness - cost_weight * cost) showing viable corridors
  4. Particle trajectories for both modes overlaid on the landscape
  5. Fitness over time: team vs solo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from experiments.landscape import (
    make_default_landscape, make_default_cost_landscape
)

K = 3
LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)

# ── 1. Render landscape heatmaps ──────────────────────────────────────────────

res = 200
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
xx, yy = np.meshgrid(x, y)
grid_prefs = np.zeros((res * res, K))
grid_prefs[:, 0] = xx.ravel()
grid_prefs[:, 1] = yy.ravel()

fitness_vals, _ = LANDSCAPE.fitness(grid_prefs)
fitness_map = fitness_vals.reshape(res, res)

cost_vals = COST_LANDSCAPE.cost(grid_prefs)
cost_map = cost_vals.reshape(res, res)

cost_weight = 0.3
effective_map = fitness_map - cost_weight * cost_map

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(fitness_map, extent=[-1, 1, -1, 1], origin='lower',
                       cmap='terrain', aspect='equal')
axes[0].set_title('Fitness Landscape (rugged)')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(cost_map, extent=[-1, 1, -1, 1], origin='lower',
                       cmap='hot_r', aspect='equal')
axes[1].set_title('Cost Landscape (with HF noise)')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(effective_map, extent=[-1, 1, -1, 1], origin='lower',
                       cmap='RdYlGn', aspect='equal')
axes[2].set_title(f'Effective = Fitness - {cost_weight}×Cost')
plt.colorbar(im2, ax=axes[2], shrink=0.8)

# Mark summit
summit = LANDSCAPE.centers[0]
for ax in axes:
    ax.plot(summit[0], summit[1], 'w*', markersize=15, markeredgecolor='k')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/redesign_landscapes.png', dpi=150)
plt.close()
print("Saved results/redesign_landscapes.png")


# ── 2. Run simulation: TEAM vs SOLO ──────────────────────────────────────────

# We'll use the 3D sim's Simulation class directly
from importlib import import_module
sim3d_mod = import_module('3D_sim.simulation3d')
Simulation = sim3d_mod.Simulation
params = sim3d_mod.params

N = 200
STEPS = 1500
SAMPLE_EVERY = 30  # record positions every N steps

def run_scenario(label, gradient_noise_mean, n_neighbors, cost_wt,
                 explore_prob, momentum, use_roles=True):
    """Run a scenario and return trajectory data."""
    # Set params
    params['n'] = N
    params['k'] = K
    params['strategy_enabled'] = True
    params['strategy_k'] = K
    params['pref_strategy_coupling'] = 0.5
    params['use_particle_roles'] = use_roles
    params['role_gradient_noise_mean'] = gradient_noise_mean
    params['role_gradient_noise_std'] = 0.5 if use_roles else 0.01
    params['role_influence_std'] = 0.8 if use_roles else 0.0
    params['role_step_scale_std'] = 0.5 if use_roles else 0.0
    params['role_visionary_mean'] = 0.05
    params['role_visionary_std'] = 0.03
    params['n_neighbors'] = n_neighbors
    params['strategy_step_size'] = 0.003
    params['cost_weight'] = cost_wt
    params['explore_probability'] = explore_prob
    params['explore_radius'] = 0.15
    params['strategy_momentum'] = momentum
    params['use_seed'] = True
    params['seed'] = 42

    sim = Simulation()
    sim.reset()

    summit_center = LANDSCAPE.centers[0][:K]

    # Record data
    trajectory_samples = []  # list of (step, strategy_copy)
    fitness_history = []

    for step in range(STEPS):
        sim.step()
        sim.strategy_step(
            gradient_fn=LANDSCAPE.gradient,
            summit_center=summit_center,
            cost_landscape=COST_LANDSCAPE)

        if step % SAMPLE_EVERY == 0:
            trajectory_samples.append((step, sim.strategy.copy()))

        # Record mean fitness
        f, _ = LANDSCAPE.fitness(sim.strategy)
        fitness_history.append(f.mean())

    print(f"  {label}: final mean fitness = {fitness_history[-1]:.4f}")
    return trajectory_samples, fitness_history


print("\nRunning TEAM scenario (noise=2.0, 20 neighbors, cost-aware)...")
team_traj, team_fitness = run_scenario(
    "TEAM", gradient_noise_mean=2.0, n_neighbors=20, cost_wt=0.3,
    explore_prob=0.005, momentum=0.3, use_roles=True)

print("Running SOLO scenario (noise=0.3, 1 neighbor, no cost)...")
solo_traj, solo_fitness = run_scenario(
    "SOLO", gradient_noise_mean=0.3, n_neighbors=1, cost_wt=0.0,
    explore_prob=0.0, momentum=0.0, use_roles=False)

print("Running NOISY-SOLO scenario (noise=2.0, 1 neighbor, no cost)...")
noisy_solo_traj, noisy_solo_fitness = run_scenario(
    "NOISY-SOLO", gradient_noise_mean=2.0, n_neighbors=1, cost_wt=0.0,
    explore_prob=0.0, momentum=0.0, use_roles=False)


# ── 3. Plot trajectories ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax_idx, (traj, title) in enumerate([
    (team_traj, "TEAM (noise=2.0, 20 nbrs, cost-aware)"),
    (solo_traj, "SOLO (noise=0.3, 1 nbr, no cost)"),
    (noisy_solo_traj, "NOISY-SOLO (noise=2.0, 1 nbr)")
]):
    ax = axes[ax_idx]
    ax.imshow(effective_map, extent=[-1, 1, -1, 1], origin='lower',
              cmap='RdYlGn', alpha=0.6, aspect='equal')

    # Plot trajectories for a subset of particles
    n_show = 30
    for p_idx in range(0, N, N // n_show):
        xs = [s[1][p_idx, 0] for s in traj]
        ys = [s[1][p_idx, 1] for s in traj]
        ax.plot(xs, ys, '-', alpha=0.3, linewidth=0.5, color='blue')

    # Plot final positions
    final_strat = traj[-1][1]
    ax.scatter(final_strat[:, 0], final_strat[:, 1], s=2, c='red', alpha=0.5)

    # Mark summit
    ax.plot(summit[0], summit[1], 'w*', markersize=15, markeredgecolor='k')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('results/redesign_trajectories.png', dpi=150)
plt.close()
print("Saved results/redesign_trajectories.png")


# ── 4. Fitness over time ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
steps = range(STEPS)
ax.plot(steps, team_fitness, label='TEAM (noise=2.0, 20 nbrs, cost-aware)',
        color='blue', linewidth=1.5)
ax.plot(steps, solo_fitness, label='SOLO (noise=0.3, 1 nbr, no cost)',
        color='green', linewidth=1.5)
ax.plot(steps, noisy_solo_fitness, label='NOISY-SOLO (noise=2.0, 1 nbr)',
        color='red', linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Mean Fitness')
ax.set_title('Fitness Over Time: Team vs Solo')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/redesign_fitness_curves.png', dpi=150)
plt.close()
print("Saved results/redesign_fitness_curves.png")


# ── 5. Summary stats ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"TEAM final fitness:       {team_fitness[-1]:.4f}")
print(f"SOLO final fitness:       {solo_fitness[-1]:.4f}")
print(f"NOISY-SOLO final fitness: {noisy_solo_fitness[-1]:.4f}")
print(f"\nTeam advantage over noisy-solo: {team_fitness[-1] - noisy_solo_fitness[-1]:.4f}")
print(f"Team advantage over solo:       {team_fitness[-1] - solo_fitness[-1]:.4f}")

# Check if team beats solo (the key test)
if team_fitness[-1] > noisy_solo_fitness[-1]:
    print("\n✓ TEAM outperforms NOISY-SOLO — group advantage confirmed!")
else:
    print("\n✗ TEAM does NOT outperform NOISY-SOLO — group advantage NOT confirmed")

# Distance to summit
team_final = team_traj[-1][1]
solo_final = solo_traj[-1][1]
noisy_final = noisy_solo_traj[-1][1]

team_dist = np.linalg.norm(team_final[:, :2] - summit[:2], axis=1).mean()
solo_dist = np.linalg.norm(solo_final[:, :2] - summit[:2], axis=1).mean()
noisy_dist = np.linalg.norm(noisy_final[:, :2] - summit[:2], axis=1).mean()

print(f"\nMean distance to summit:")
print(f"  TEAM:       {team_dist:.4f}")
print(f"  SOLO:       {solo_dist:.4f}")
print(f"  NOISY-SOLO: {noisy_dist:.4f}")
