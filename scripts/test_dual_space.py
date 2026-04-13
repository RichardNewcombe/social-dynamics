#!/usr/bin/env python3
"""
Test dual-space mountain mode headlessly.

Runs the 3D simulation with strategy_enabled=True, calls strategy_step()
after each sim.step(), and renders snapshots showing:
  1. Particles on the mountain surface (strategy-based projection)
  2. Preference-space vs strategy-space divergence over time
  3. Surface adherence verification
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import 3D sim components
from importlib import import_module
sim3d = import_module('3D_sim.simulation3d')
params = sim3d.params
Simulation = sim3d.Simulation
SPACE = import_module('3D_sim.grid3d').SPACE

from experiments.landscape import make_default_landscape
mm = import_module('3D_sim.mountain_mesh')
generate_mountain_mesh = mm.generate_mountain_mesh
project_particles_to_surface = mm.project_particles_to_surface

# ── Configure for dual-space mountain mode ──
K = 3
params['k'] = K
params['num_particles'] = 500
params['n_neighbors'] = 21
params['step_size'] = 0.005
params['social'] = 0.01
params['inner_prod_avg'] = True
params['knn_method'] = 1
params['physics_engine'] = 1  # NumPy (no GPU needed)
params['use_f64'] = True
params['mountain_mode'] = True

# Enable dual-space
params['strategy_enabled'] = True
params['strategy_k'] = K
params['pref_strategy_coupling'] = 0.5
params['strategy_step_size'] = 0.003

# Enable roles
params['use_particle_roles'] = True
params['role_influence_std'] = 0.8
params['role_step_scale_std'] = 0.5
params['role_gradient_noise_mean'] = 0.5
params['role_gradient_noise_std'] = 0.2
params['role_visionary_mean'] = 0.05
params['role_visionary_std'] = 0.03

# ── Create landscape and mesh ──
LANDSCAPE = make_default_landscape(k=K)
Z_SCALE = 0.5
verts, normals, colors_f, indices, fitness_grid = generate_mountain_mesh(
    LANDSCAPE, resolution=64, z_scale=Z_SCALE, pref_dims=K)

# ── Create simulation ──
sim = Simulation()
print(f"Strategy enabled: {sim.strategy is not None}")
print(f"Strategy shape: {sim.strategy.shape if sim.strategy is not None else 'N/A'}")
print(f"Prefs shape: {sim.prefs.shape}")
print(f"Role arrays: step_scale={sim.role_step_scale.mean():.3f}, "
      f"influence={sim.role_influence.mean():.3f}, "
      f"grad_noise={sim.role_gradient_noise.mean():.3f}, "
      f"visionary={sim.role_visionary.mean():.3f}")

# Initial surface sync
proj = project_particles_to_surface(sim.strategy, LANDSCAPE, z_scale=Z_SCALE)
sim.pos[:, :3] = proj.astype(np.float64)

# ── Run simulation ──
N_STEPS = 2000
SNAPSHOT_STEPS = [0, 100, 500, 1000, 2000]
snapshots = {}
metrics = {'step': [], 'mean_fitness': [], 'pref_std': [], 'strat_std': [],
           'divergence': [], 'max_surface_offset': []}

summit = LANDSCAPE.centers[0][:K]

for step in range(N_STEPS + 1):
    if step in SNAPSHOT_STEPS:
        # Snapshot strategy positions on mountain
        snapshots[step] = {
            'strategy': sim.strategy.copy(),
            'prefs': sim.prefs.copy(),
            'pos': sim.pos[:, :3].copy(),
        }
        print(f"Step {step}: snapshot saved")

    # Log metrics
    if step % 50 == 0:
        fitness, _ = LANDSCAPE.fitness(sim.strategy)
        pref_std = sim.prefs.std(axis=0).mean()
        strat_std = sim.strategy.std(axis=0).mean()
        divergence = np.linalg.norm(
            sim.strategy - sim.prefs.astype(np.float64), axis=1).mean()

        # Check surface adherence
        proj_check = project_particles_to_surface(sim.strategy, LANDSCAPE, z_scale=Z_SCALE)
        surface_offset = np.abs(sim.pos[:, 1] - proj_check[:, 1].astype(np.float64))
        max_offset = surface_offset.max()

        metrics['step'].append(step)
        metrics['mean_fitness'].append(fitness.mean())
        metrics['pref_std'].append(pref_std)
        metrics['strat_std'].append(strat_std)
        metrics['divergence'].append(divergence)
        metrics['max_surface_offset'].append(max_offset)

    if step >= N_STEPS:
        break

    # Phase 1: social dynamics
    sim.step()

    # Phase 2: strategy navigation (like Exp 5)
    sim.strategy_step(
        gradient_fn=LANDSCAPE.gradient,
        summit_center=summit)

    # Sync pos to surface
    projected = project_particles_to_surface(sim.strategy, LANDSCAPE, z_scale=Z_SCALE)
    sim.pos[:, :3] = projected.astype(np.float64)

print("\n=== Final metrics ===")
print(f"Mean fitness: {metrics['mean_fitness'][-1]:.4f}")
print(f"Pref std: {metrics['pref_std'][-1]:.4f}")
print(f"Strategy std: {metrics['strat_std'][-1]:.4f}")
print(f"Pref-strategy divergence: {metrics['divergence'][-1]:.4f}")
print(f"Max surface offset: {metrics['max_surface_offset'][-1]:.8f}")

# ── Render snapshots ──
# Build mesh wireframe for plotting
res = 64
lin = np.linspace(0, 1, res)
mesh_x, mesh_z = np.meshgrid(lin, lin, indexing='ij')
mesh_y = fitness_grid  # already (res, res)
# Normalize mesh_y to [0, Z_SCALE]
f_min, f_max = fitness_grid.min(), fitness_grid.max()
f_range = max(f_max - f_min, 1e-8)
mesh_y_norm = (mesh_y - f_min) / f_range * Z_SCALE

fig = plt.figure(figsize=(20, 12))

for i, step_key in enumerate(SNAPSHOT_STEPS):
    snap = snapshots[step_key]
    pos = snap['pos']

    # 3D view
    ax = fig.add_subplot(2, len(SNAPSHOT_STEPS), i + 1, projection='3d')
    ax.plot_surface(mesh_x, mesh_y_norm, mesh_z, alpha=0.2, cmap='terrain',
                    linewidth=0, antialiased=False)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=2, c='red', alpha=0.5)
    ax.set_title(f'Step {step_key}', fontsize=10)
    ax.set_xlabel('X (pref[0])')
    ax.set_ylabel('Y (fitness)')
    ax.set_zlabel('Z (pref[1])')
    ax.view_init(elev=25, azim=45)

    # Top-down view (strategy space)
    ax2 = fig.add_subplot(2, len(SNAPSHOT_STEPS), len(SNAPSHOT_STEPS) + i + 1)
    strat = snap['strategy']
    prefs = snap['prefs']
    ax2.scatter(strat[:, 0], strat[:, 1], s=2, c='red', alpha=0.3, label='strategy')
    ax2.scatter(prefs[:, 0], prefs[:, 1], s=2, c='blue', alpha=0.3, label='prefs')
    # Mark summit
    ax2.scatter([summit[0]], [summit[1]], s=100, c='gold', marker='*',
                edgecolors='black', zorder=10, label='summit')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title(f'Step {step_key} (pref vs strat)', fontsize=10)
    ax2.set_xlabel('dim 0')
    ax2.set_ylabel('dim 1')
    if i == 0:
        ax2.legend(fontsize=7, loc='upper left')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/dual_space_snapshots.png', dpi=150)
print("Saved results/dual_space_snapshots.png")

# ── Metrics plot ──
fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(metrics['step'], metrics['mean_fitness'])
axes[0, 0].set_title('Mean Fitness (strategy space)')
axes[0, 0].set_xlabel('Step')

axes[0, 1].plot(metrics['step'], metrics['pref_std'], label='pref std')
axes[0, 1].plot(metrics['step'], metrics['strat_std'], label='strat std')
axes[0, 1].set_title('Diversity')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Step')

axes[0, 2].plot(metrics['step'], metrics['divergence'])
axes[0, 2].set_title('Pref-Strategy Divergence')
axes[0, 2].set_xlabel('Step')

axes[1, 0].plot(metrics['step'], metrics['max_surface_offset'])
axes[1, 0].set_title('Max Surface Offset')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Y offset')

# Summit distance
summit_dists = []
for s in metrics['step']:
    idx = metrics['step'].index(s)
    # Recompute from nearest snapshot or just use strat_std as proxy
summit_dists = []
for step_val in metrics['step']:
    # Find closest snapshot
    closest = min(SNAPSHOT_STEPS, key=lambda x: abs(x - step_val))
    if closest in snapshots:
        strat = snapshots[closest]['strategy']
        d = np.linalg.norm(strat - summit, axis=1).mean()
        summit_dists.append(d)
    else:
        summit_dists.append(np.nan)

axes[1, 1].plot(metrics['step'][:len(summit_dists)], summit_dists)
axes[1, 1].set_title('Mean Distance to Summit')
axes[1, 1].set_xlabel('Step')

axes[1, 2].axis('off')
axes[1, 2].text(0.1, 0.5, 
    f"Final metrics:\n"
    f"  Mean fitness: {metrics['mean_fitness'][-1]:.4f}\n"
    f"  Pref std: {metrics['pref_std'][-1]:.4f}\n"
    f"  Strategy std: {metrics['strat_std'][-1]:.4f}\n"
    f"  Divergence: {metrics['divergence'][-1]:.4f}\n"
    f"  Max surface offset: {metrics['max_surface_offset'][-1]:.2e}\n"
    f"  Strategy enabled: {sim.strategy is not None}\n"
    f"  Coupling: {params['pref_strategy_coupling']}",
    transform=axes[1, 2].transAxes, fontsize=12, verticalalignment='center',
    fontfamily='monospace')

plt.tight_layout()
plt.savefig('results/dual_space_metrics.png', dpi=150)
print("Saved results/dual_space_metrics.png")
