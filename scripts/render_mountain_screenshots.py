#!/usr/bin/env python3
"""
Render matplotlib 3D screenshots of the mountain experiment.

Generates a multi-panel figure showing different experimental conditions
with particles on the fitness landscape, colored by their preferences
(team identity).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from sim_2d_exp.params import params as _global_params
from sim_2d_exp.simulation import Simulation
from experiments.landscape import (
    make_default_landscape, make_default_cost_landscape,
    compute_employee_cost,
)

K = 3
LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)


def run_condition(label, overrides, n_steps=2000):
    """Run a simulation with given overrides and return final state."""
    # Snapshot
    snapshot = dict(_global_params)
    try:
        # Start from full defaults, then apply overrides
        _global_params['num_particles'] = 500
        _global_params['k'] = K
        _global_params['physics_engine'] = 1
        _global_params['social_mode'] = 0
        _global_params['use_seed'] = True
        _global_params['seed'] = 42
        _global_params['use_f64'] = True
        _global_params.update(overrides)

        sim = Simulation()

        # Gradient function for strategy_step
        summit_center = LANDSCAPE.centers[0]
        def gradient_fn(strategy):
            return LANDSCAPE.gradient(strategy)

        for step in range(1, n_steps + 1):
            sim.step()

            # If strategy enabled, do strategy_step
            if sim.strategy is not None:
                sim.strategy_step(gradient_fn=gradient_fn,
                                  summit_center=summit_center)
            else:
                # Exp4-style: gradient nudge on prefs directly
                prefs = sim.prefs.astype(np.float64)
                grad_unit, fitness, peak_ids = LANDSCAPE.gradient(prefs)
                noise = sim.rng.normal(0, 1, grad_unit.shape)
                noise *= sim.role_gradient_noise[:, None]
                noisy_grad = grad_unit + noise
                norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                noisy_grad = noisy_grad / norms

                # Visionary blend
                to_summit = summit_center - prefs
                summit_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
                summit_norms = np.maximum(summit_norms, 1e-12)
                summit_dir = to_summit / summit_norms
                v = sim.role_visionary[:, None]
                effective_grad = (1.0 - v) * noisy_grad + v * summit_dir
                eff_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
                eff_norms = np.maximum(eff_norms, 1e-12)
                effective_grad = effective_grad / eff_norms

                nudge = 0.003 * effective_grad
                sim.prefs = np.clip(prefs + nudge, -1, 1).astype(sim.prefs.dtype)

        # Collect results
        if sim.strategy is not None:
            coords = sim.strategy.astype(np.float64)
        else:
            coords = sim.prefs.astype(np.float64)

        fitness, peak_ids = LANDSCAPE.fitness(coords)
        pref_colors = sim.prefs.astype(np.float64)

        # Summit fraction
        diff = coords - summit_center
        dists = np.linalg.norm(diff, axis=1)
        summit_frac = (dists <= 0.35).sum() / len(coords)

        return {
            'label': label,
            'coords': coords,
            'fitness': fitness,
            'pref_colors': pref_colors,
            'summit_frac': summit_frac,
        }
    finally:
        _global_params.clear()
        _global_params.update(snapshot)


def render_mountain_surface(ax, landscape, alpha=0.3, resolution=60):
    """Render the fitness landscape as a wireframe surface."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate fitness at each grid point (fix other dims at 0)
    grid_points = np.zeros((resolution * resolution, K), dtype=np.float64)
    grid_points[:, 0] = X.ravel()
    grid_points[:, 1] = Y.ravel()
    Z_fitness, _ = landscape.fitness(grid_points)
    Z = Z_fitness.reshape(resolution, resolution)

    # Surface with terrain colormap
    norm = Normalize(vmin=0, vmax=1.0)
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=alpha, norm=norm,
                    linewidth=0, antialiased=True, zorder=1)
    return X, Y, Z


def render_cost_overlay(ax, cost_landscape, z_offset=0.02,
                        alpha=0.15, resolution=60):
    """Render cost terrain as a semi-transparent red overlay."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)

    grid_points = np.zeros((resolution * resolution, K), dtype=np.float64)
    grid_points[:, 0] = X.ravel()
    grid_points[:, 1] = Y.ravel()
    costs = cost_landscape.cost(grid_points)
    Z = costs.reshape(resolution, resolution)

    # Normalize cost to [0, 1] for coloring
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)

    # Plot as a faint red surface slightly above the mountain
    from matplotlib.colors import LinearSegmentedColormap
    cost_cmap = LinearSegmentedColormap.from_list(
        'cost', [(0.2, 0.8, 0.2, 0.0), (1.0, 0.8, 0.0, 0.3), (1.0, 0.0, 0.0, 0.6)])
    ax.plot_surface(X, Y, Z_norm * 0.3 + z_offset, cmap=cost_cmap,
                    alpha=alpha, linewidth=0, antialiased=True, zorder=0)


def render_particles(ax, result, z_scale=1.0):
    """Render particles on the mountain, colored by preferences."""
    coords = result['coords']
    fitness = result['fitness']
    pref_colors = result['pref_colors']

    # Map prefs to RGB: pref[0]->R, pref[1]->G, pref[2]->B, all in [0,1]
    rgb = (pref_colors[:, :3] + 1) / 2  # map [-1,1] to [0,1]
    rgb = np.clip(rgb, 0, 1)

    # Particle positions: x=coord[0], y=coord[1], z=fitness
    x = coords[:, 0]
    y = coords[:, 1]
    z = fitness * z_scale

    ax.scatter(x, y, z, c=rgb, s=4, alpha=0.7, depthshade=True, zorder=5)

    # Mark the summit (always centers[0] for both landscape types)
    summit = LANDSCAPE.centers[0]
    summit_fitness, _ = LANDSCAPE.fitness(summit.reshape(1, -1))
    ax.scatter([summit[0]], [summit[1]], [summit_fitness[0] * z_scale],
               c='gold', s=120, marker='*', edgecolors='black',
               linewidths=0.5, zorder=10, label='Summit')

    # Mark major local peaks (for RuggedLandscape, use major_centers;
    # for GaussianPeakLandscape, use centers)
    if hasattr(LANDSCAPE, 'major_centers'):
        peak_centers = LANDSCAPE.major_centers
    else:
        peak_centers = LANDSCAPE.centers
    for i in range(1, len(peak_centers)):
        lp = peak_centers[i]
        lp_fitness, _ = LANDSCAPE.fitness(lp.reshape(1, -1))
        ax.scatter([lp[0]], [lp[1]], [lp_fitness[0] * z_scale],
                   c='red', s=80, marker='^', edgecolors='black',
                   linewidths=0.5, zorder=10)


def main():
    # Define conditions to render
    conditions = [
        {
            'label': 'Exp4: Baseline\n(no roles, no social)',
            'overrides': {
                'social': 0.0,
                'use_particle_roles': False,
                'strategy_enabled': False,
            },
        },
        {
            'label': 'Exp4: All Visionaries\n(vis=0.80, coupling=1.0)',
            'overrides': {
                'social': 0.01,
                'use_particle_roles': True,
                'role_influence_std': 0.8,
                'role_step_scale_std': 0.5,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.80,
                'role_visionary_std': 0.10,
                'strategy_enabled': False,
            },
        },
        {
            'label': 'Exp5: Specialist\n(coupling=1.0)',
            'overrides': {
                'social': 0.01,
                'use_particle_roles': True,
                'role_influence_std': 0.8,
                'role_step_scale_std': 0.5,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.15,
                'role_visionary_std': 0.08,
                'role_visionary_fraction': 0.10,
                'strategy_enabled': True,
                'strategy_k': K,
                'pref_strategy_coupling': 1.0,
                'strategy_step_size': 0.003,
            },
        },
        {
            'label': 'Exp5: Generalist\n(coupling=0.0)',
            'overrides': {
                'social': 0.01,
                'use_particle_roles': True,
                'role_influence_std': 0.8,
                'role_step_scale_std': 0.5,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.15,
                'role_visionary_std': 0.08,
                'role_visionary_fraction': 0.10,
                'strategy_enabled': True,
                'strategy_k': K,
                'pref_strategy_coupling': 0.0,
                'strategy_step_size': 0.003,
            },
        },
        {
            'label': 'Exp5: Rare Vis + Generalist\n(5% vis, coupling=0.0)',
            'overrides': {
                'social': 0.01,
                'use_particle_roles': True,
                'role_influence_std': 0.8,
                'role_step_scale_std': 0.5,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.60,
                'role_visionary_std': 0.10,
                'role_visionary_fraction': 0.05,
                'strategy_enabled': True,
                'strategy_k': K,
                'pref_strategy_coupling': 0.0,
                'strategy_step_size': 0.003,
            },
        },
        {
            'label': 'Exp5: Specialist + Differentiation\n(coupling=1.0, s=-0.01)',
            'overrides': {
                'social': -0.01,
                'use_particle_roles': True,
                'role_influence_std': 0.8,
                'role_step_scale_std': 0.5,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.15,
                'role_visionary_std': 0.08,
                'role_visionary_fraction': 0.10,
                'strategy_enabled': True,
                'strategy_k': K,
                'pref_strategy_coupling': 1.0,
                'strategy_step_size': 0.003,
            },
        },
    ]

    print("Running simulations...")
    results = []
    for i, cond in enumerate(conditions):
        print(f"  [{i+1}/{len(conditions)}] {cond['label'].replace(chr(10), ' ')}...")
        result = run_condition(cond['label'], cond['overrides'], n_steps=2000)
        print(f"    Summit: {result['summit_frac']:.1%}")
        results.append(result)

    # Render multi-panel figure
    print("\nRendering figure...")
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('Mountain Experiment: Particle Positions on Fitness Landscape\n'
                 '(2000 steps, 500 particles, K=3)',
                 fontsize=16, fontweight='bold', y=0.98)

    for i, result in enumerate(results):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')

        # Render mountain surface
        render_mountain_surface(ax, LANDSCAPE, alpha=0.25, resolution=50)

        # Render particles
        render_particles(ax, result, z_scale=1.0)

        # Title with summit fraction
        ax.set_title(f"{result['label']}\nSummit: {result['summit_frac']:.1%}",
                     fontsize=11, pad=10)

        # Camera angle
        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel('Pref/Strat [0]', fontsize=8)
        ax.set_ylabel('Pref/Strat [1]', fontsize=8)
        ax.set_zlabel('Fitness', fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1.1)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(_pkg_root, 'results', 'mountain_screenshots.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved to {out_path}")
    plt.close()

    # Also render a cost overlay version
    print("\nRendering cost overlay version...")
    fig2 = plt.figure(figsize=(16, 6))
    fig2.suptitle('Fitness Mountain vs Cost Terrain\n'
                  '(gold star = global summit, red triangles = local peaks)',
                  fontsize=14, fontweight='bold', y=1.02)

    # Panel 1: Fitness landscape (top view)
    ax1 = fig2.add_subplot(1, 3, 1, projection='3d')
    render_mountain_surface(ax1, LANDSCAPE, alpha=0.6, resolution=60)
    summit = LANDSCAPE.centers[0]
    sf, _ = LANDSCAPE.fitness(summit.reshape(1, -1))
    ax1.scatter([summit[0]], [summit[1]], [sf[0]], c='gold', s=200,
                marker='*', edgecolors='black', linewidths=1, zorder=10)
    if hasattr(LANDSCAPE, 'major_centers'):
        ov_peaks = LANDSCAPE.major_centers
    else:
        ov_peaks = LANDSCAPE.centers
    for i in range(1, len(ov_peaks)):
        lp = ov_peaks[i]
        lf, _ = LANDSCAPE.fitness(lp.reshape(1, -1))
        ax1.scatter([lp[0]], [lp[1]], [lf[0]], c='red', s=100,
                    marker='^', edgecolors='black', linewidths=1, zorder=10)
    ax1.set_title('Fitness Landscape', fontsize=12)
    ax1.view_init(elev=35, azim=-50)
    ax1.set_xlabel('Dim 0')
    ax1.set_ylabel('Dim 1')
    ax1.set_zlabel('Fitness')

    # Panel 2: Cost terrain
    ax2 = fig2.add_subplot(1, 3, 2, projection='3d')
    x = np.linspace(-1, 1, 60)
    y = np.linspace(-1, 1, 60)
    X, Y = np.meshgrid(x, y)
    grid = np.zeros((3600, K), dtype=np.float64)
    grid[:, 0] = X.ravel()
    grid[:, 1] = Y.ravel()
    costs = COST_LANDSCAPE.cost(grid).reshape(60, 60)
    ax2.plot_surface(X, Y, costs, cmap='RdYlGn_r', alpha=0.7,
                     linewidth=0, antialiased=True)
    ax2.set_title('Cost Terrain\n(red = expensive, green = cheap)', fontsize=12)
    ax2.view_init(elev=35, azim=-50)
    ax2.set_xlabel('Dim 0')
    ax2.set_ylabel('Dim 1')
    ax2.set_zlabel('Cost')

    # Panel 3: Both overlaid with best condition particles
    ax3 = fig2.add_subplot(1, 3, 3, projection='3d')
    render_mountain_surface(ax3, LANDSCAPE, alpha=0.2, resolution=50)
    # Use the "Rare Vis + Generalist" result (index 4)
    best = results[4]
    render_particles(ax3, best, z_scale=1.0)
    ax3.set_title(f"Best Condition: Rare Vis + Generalist\nSummit: {best['summit_frac']:.1%}",
                  fontsize=12)
    ax3.view_init(elev=35, azim=-50)
    ax3.set_xlabel('Dim 0')
    ax3.set_ylabel('Dim 1')
    ax3.set_zlabel('Fitness')

    plt.tight_layout()
    out_path2 = os.path.join(_pkg_root, 'results', 'mountain_landscape_overview.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"Saved to {out_path2}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
