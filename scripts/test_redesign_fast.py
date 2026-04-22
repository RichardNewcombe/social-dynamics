"""
Fast headless test of the redesigned mountain simulation.

Uses the 2D simulation (sim_2d_exp) which is much faster than the 3D sim
for headless testing.  Compares TEAM vs SOLO on the new rugged landscape.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.landscape import (
    make_default_landscape, make_default_cost_landscape
)

K = 2  # Use K=2 for faster testing and clearer visualization
LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)

# ── 1. Render landscape heatmaps ──────────────────────────────────────────────

res = 300
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

summit = LANDSCAPE.centers[0][:K]

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

for ax in axes:
    ax.plot(summit[0], summit[1], 'w*', markersize=15, markeredgecolor='k')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/redesign_landscapes.png', dpi=150)
plt.close()
print("Saved results/redesign_landscapes.png")


# ── 2. Pure strategy navigation test (no social dynamics overhead) ────────────

N = 200
STEPS = 2000
rng = np.random.default_rng(42)

def run_pure_strategy(label, gradient_noise, team_size, cost_wt,
                      explore_prob, explore_radius, momentum, step_size=0.003):
    """
    Run pure strategy navigation without the full simulation overhead.
    
    team_size=1 means solo (no aggregation).
    team_size>1 means each particle averages observations from team_size neighbors.
    """
    # Initialize random positions in [-1, 1]^K
    strategy = rng.uniform(-1, 1, (N, K))
    
    # Build fixed neighbor graph (random teams for simplicity)
    if team_size > 1:
        nbr_ids = np.zeros((N, team_size), dtype=int)
        for i in range(N):
            candidates = np.delete(np.arange(N), i)
            nbr_ids[i] = rng.choice(candidates, team_size, replace=False)
    
    prev_grad = None
    fitness_history = []
    trajectory_samples = []
    
    for step in range(STEPS):
        # 1. Sense fitness gradient
        grad_unit, fitness, peak_ids = LANDSCAPE.gradient(strategy)
        
        # 1b. Cost-aware gradient
        if cost_wt > 0:
            cost_grad, cost_vals = COST_LANDSCAPE.cost_gradient(strategy)
            effective_grad = grad_unit - cost_wt * cost_grad
            eg_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
            effective_grad = effective_grad / np.maximum(eg_norms, 1e-12)
        else:
            effective_grad = grad_unit
        
        # 1c. Add noise (do NOT normalize before aggregation!)
        noise = rng.normal(0, gradient_noise, effective_grad.shape)
        noisy_obs = effective_grad + noise  # NOT normalized
        
        # 2. Team aggregation (average raw noisy observations)
        if team_size > 1:
            nbr_obs = noisy_obs[nbr_ids]  # (N, team_size, K)
            team_obs = nbr_obs.mean(axis=1)  # (N, K)
            # Blend own observation with team
            team_grad = 0.5 * noisy_obs + 0.5 * team_obs
        else:
            team_grad = noisy_obs
        # NOW normalize to get direction
        t_norms = np.linalg.norm(team_grad, axis=1, keepdims=True)
        team_grad = team_grad / np.maximum(t_norms, 1e-12)
        
        # 3. Momentum
        if momentum > 0 and prev_grad is not None:
            team_grad = (1.0 - momentum) * team_grad + momentum * prev_grad
            t_norms = np.linalg.norm(team_grad, axis=1, keepdims=True)
            team_grad = team_grad / np.maximum(t_norms, 1e-12)
        prev_grad = team_grad.copy()
        
        # 4. Move
        strategy = np.clip(strategy + step_size * team_grad, -1, 1)
        
        # 5. Exploration
        if explore_prob > 0:
            jumpers = rng.random(N) < explore_prob
            n_jump = jumpers.sum()
            if n_jump > 0:
                jump_dir = rng.normal(0, 1, (n_jump, K))
                jn = np.linalg.norm(jump_dir, axis=1, keepdims=True)
                jump_dir = jump_dir / np.maximum(jn, 1e-12)
                jump_dist = rng.uniform(0, explore_radius, (n_jump, 1))
                strategy[jumpers] = np.clip(
                    strategy[jumpers] + jump_dist * jump_dir, -1, 1)
        
        # Record
        f, _ = LANDSCAPE.fitness(strategy)
        fitness_history.append(f.mean())
        if step % 40 == 0:
            trajectory_samples.append((step, strategy.copy()))
    
    print(f"  {label}: final mean fitness = {fitness_history[-1]:.4f}")
    return trajectory_samples, fitness_history


print("\n--- Running scenarios ---")

# TEAM: high noise, team aggregation, cost-aware, exploration
team_traj, team_fit = run_pure_strategy(
    "TEAM (noise=2.0, team=20, cost, explore)",
    gradient_noise=2.0, team_size=20, cost_wt=0.3,
    explore_prob=0.005, explore_radius=0.15, momentum=0.3)

# SOLO-CLEAN: low noise, no team, no cost (the "cheater" baseline)
solo_clean_traj, solo_clean_fit = run_pure_strategy(
    "SOLO-CLEAN (noise=0.3, solo, no cost)",
    gradient_noise=0.3, team_size=1, cost_wt=0.0,
    explore_prob=0.0, explore_radius=0.0, momentum=0.0)

# NOISY-SOLO: high noise, no team (shows why teams matter)
noisy_solo_traj, noisy_solo_fit = run_pure_strategy(
    "NOISY-SOLO (noise=2.0, solo)",
    gradient_noise=2.0, team_size=1, cost_wt=0.0,
    explore_prob=0.0, explore_radius=0.0, momentum=0.0)

# TEAM-NO-COST: high noise, team, no cost (isolate cost effect)
team_nocost_traj, team_nocost_fit = run_pure_strategy(
    "TEAM-NO-COST (noise=2.0, team=20, no cost)",
    gradient_noise=2.0, team_size=20, cost_wt=0.0,
    explore_prob=0.005, explore_radius=0.15, momentum=0.3)

# TEAM-BIG: high noise, bigger team
team_big_traj, team_big_fit = run_pure_strategy(
    "TEAM-BIG (noise=2.0, team=50)",
    gradient_noise=2.0, team_size=50, cost_wt=0.3,
    explore_prob=0.005, explore_radius=0.15, momentum=0.3)


# ── 3. Plot trajectories ────────────────────────────────────────────────────

scenarios = [
    (team_traj, "TEAM (noise=2, team=20, cost)"),
    (solo_clean_traj, "SOLO-CLEAN (noise=0.3)"),
    (noisy_solo_traj, "NOISY-SOLO (noise=2, solo)"),
    (team_nocost_traj, "TEAM-NO-COST (noise=2, team=20)"),
    (team_big_traj, "TEAM-BIG (noise=2, team=50)"),
]

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for ax_idx, (traj, title) in enumerate(scenarios):
    ax = axes[ax_idx]
    ax.imshow(effective_map, extent=[-1, 1, -1, 1], origin='lower',
              cmap='RdYlGn', alpha=0.5, aspect='equal')

    n_show = 20
    for p_idx in range(0, N, N // n_show):
        xs = [s[1][p_idx, 0] for s in traj]
        ys = [s[1][p_idx, 1] for s in traj]
        ax.plot(xs, ys, '-', alpha=0.3, linewidth=0.5, color='blue')

    final_strat = traj[-1][1]
    ax.scatter(final_strat[:, 0], final_strat[:, 1], s=3, c='red', alpha=0.5)
    ax.plot(summit[0], summit[1], 'w*', markersize=12, markeredgecolor='k')
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('results/redesign_trajectories.png', dpi=150)
plt.close()
print("Saved results/redesign_trajectories.png")


# ── 4. Fitness over time ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
steps = range(STEPS)
ax.plot(steps, team_fit, label='TEAM (noise=2, team=20, cost)', color='blue', linewidth=1.5)
ax.plot(steps, solo_clean_fit, label='SOLO-CLEAN (noise=0.3)', color='green', linewidth=1.5)
ax.plot(steps, noisy_solo_fit, label='NOISY-SOLO (noise=2, solo)', color='red', linewidth=1.5)
ax.plot(steps, team_nocost_fit, label='TEAM-NO-COST (noise=2, team=20)', color='purple', linewidth=1.5)
ax.plot(steps, team_big_fit, label='TEAM-BIG (noise=2, team=50)', color='orange', linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Mean Fitness')
ax.set_title('Fitness Over Time: Team vs Solo on Rugged Landscape')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/redesign_fitness_curves.png', dpi=150)
plt.close()
print("Saved results/redesign_fitness_curves.png")


# ── 5. Summary ──────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"TEAM (noise=2, team=20, cost):    {team_fit[-1]:.4f}")
print(f"SOLO-CLEAN (noise=0.3):           {solo_clean_fit[-1]:.4f}")
print(f"NOISY-SOLO (noise=2, solo):       {noisy_solo_fit[-1]:.4f}")
print(f"TEAM-NO-COST (noise=2, team=20):  {team_nocost_fit[-1]:.4f}")
print(f"TEAM-BIG (noise=2, team=50):      {team_big_fit[-1]:.4f}")

print(f"\nKey comparisons:")
print(f"  Team vs Noisy-Solo (group advantage):  {team_fit[-1] - noisy_solo_fit[-1]:+.4f}")
print(f"  Team vs Solo-Clean (cost of noise):    {team_fit[-1] - solo_clean_fit[-1]:+.4f}")
print(f"  Team-Big vs Team (bigger teams):       {team_big_fit[-1] - team_fit[-1]:+.4f}")
print(f"  Team-NoCost vs Team (cost penalty):    {team_nocost_fit[-1] - team_fit[-1]:+.4f}")

# Distance to summit
for traj, name in scenarios:
    final = traj[-1][1]
    dist = np.linalg.norm(final[:, :2] - summit[:2], axis=1).mean()
    print(f"  {name}: mean dist to summit = {dist:.3f}")
