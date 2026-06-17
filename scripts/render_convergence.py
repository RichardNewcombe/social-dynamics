#!/usr/bin/env python3
"""
Render convergence snapshots of the mountain simulation.

Produces a grid of images showing particle positions on the fitness landscape
at multiple timesteps, for several conditions. This helps diagnose whether
particles stay on the surface, converge to the summit, get trapped, etc.

Two rendering modes:
  1. "headless" — pure gradient nudge on prefs (like exp4), no 3D physics
  2. "visualizer" — mimics the 3D visualizer's mountain mode: sim.step()
     runs the full toroidal physics, then gradient nudge on prefs, then
     sim.pos is synced to the mountain projection (the recent fix).
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
)

# Also import the 3D sim for the "visualizer" mode
try:
    from _3D_sim_pkg.simulation3d import Simulation as Simulation3D
    from _3D_sim_pkg.simulation3d import params as _3d_params
    HAS_3D = True
except ImportError:
    HAS_3D = False

K = 3
LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)
SUMMIT = LANDSCAPE.centers[0][:K]

SNAPSHOT_STEPS = [0, 50, 200, 500, 1000, 2000]
N_PARTICLES = 500


def project_to_surface(prefs, landscape):
    """Project prefs to (x, y, fitness) for plotting."""
    fitness, _ = landscape.fitness(prefs.astype(np.float64))
    return prefs[:, 0], prefs[:, 1], fitness


def run_headless(label, overrides, snapshot_steps=None):
    """Run headless sim with gradient nudge on prefs (exp4-style)."""
    if snapshot_steps is None:
        snapshot_steps = SNAPSHOT_STEPS

    snapshot = dict(_global_params)
    try:
        _global_params['num_particles'] = N_PARTICLES
        _global_params['k'] = K
        _global_params['physics_engine'] = 1
        _global_params['social_mode'] = 0
        _global_params['use_seed'] = True
        _global_params['seed'] = 42
        _global_params['use_f64'] = True
        _global_params.update(overrides)

        sim = Simulation()

        summit_center = SUMMIT.copy()

        snapshots = []
        max_step = max(snapshot_steps)

        for step in range(max_step + 1):
            if step in snapshot_steps:
                coords = sim.prefs.astype(np.float64).copy()
                fitness, peak_ids = LANDSCAPE.fitness(coords)
                diff = coords - summit_center
                dists = np.linalg.norm(diff, axis=1)
                summit_frac = (dists <= 0.35).sum() / len(coords)
                snapshots.append({
                    'step': step,
                    'coords': coords,
                    'fitness': fitness,
                    'summit_frac': summit_frac,
                    'pos': sim.pos.copy(),
                    'prefs': sim.prefs.copy(),
                })

            if step < max_step:
                sim.step()

                # Gradient nudge on prefs (exp4-style)
                prefs = sim.prefs.astype(np.float64)
                grad_unit, fitness, peak_ids = LANDSCAPE.gradient(prefs)
                noise = sim.rng.normal(0, 1, grad_unit.shape)
                noise *= sim.role_gradient_noise[:, None]
                noisy_grad = grad_unit + noise
                norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
                noisy_grad = noisy_grad / np.maximum(norms, 1e-12)

                # Visionary blend
                to_summit = summit_center - prefs
                summit_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
                summit_dir = to_summit / np.maximum(summit_norms, 1e-12)
                v = sim.role_visionary[:, None]
                effective_grad = (1.0 - v) * noisy_grad + v * summit_dir
                eff_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
                effective_grad = effective_grad / np.maximum(eff_norms, 1e-12)

                # Apply step-scale (engineer role)
                step_scale = sim.role_step_scale[:, None]
                nudge = 0.003 * step_scale * effective_grad
                sim.prefs = np.clip(prefs + nudge, -1, 1).astype(sim.prefs.dtype)

        return {'label': label, 'snapshots': snapshots}
    finally:
        _global_params.clear()
        _global_params.update(snapshot)


def run_visualizer_mimic(label, overrides, snapshot_steps=None):
    """Mimic the 3D visualizer's mountain mode loop.

    sim.step() runs full toroidal physics, then gradient nudge on prefs,
    then sim.pos is synced to the mountain projection.
    """
    if snapshot_steps is None:
        snapshot_steps = SNAPSHOT_STEPS

    snapshot = dict(_global_params)
    try:
        _global_params['num_particles'] = N_PARTICLES
        _global_params['k'] = K
        _global_params['physics_engine'] = 1
        _global_params['social_mode'] = 0
        _global_params['use_seed'] = True
        _global_params['seed'] = 42
        _global_params['use_f64'] = True
        _global_params.update(overrides)

        sim = Simulation()
        summit_center = SUMMIT.copy()

        # Initial pos sync to mountain surface
        coords = sim.prefs.astype(np.float64)
        fitness, _ = LANDSCAPE.fitness(coords)
        sim.pos[:, 0] = (coords[:, 0] + 1.0) * 0.5
        sim.pos[:, 1] = fitness * 0.5  # z_scale = 0.5
        if sim.pos.shape[1] > 2:
            sim.pos[:, 2] = (coords[:, 1] + 1.0) * 0.5 if K > 1 else 0.5

        snapshots = []
        max_step = max(snapshot_steps)

        for step in range(max_step + 1):
            if step in snapshot_steps:
                coords = sim.prefs.astype(np.float64).copy()
                fitness, peak_ids = LANDSCAPE.fitness(coords)
                diff = coords - summit_center
                dists = np.linalg.norm(diff, axis=1)
                summit_frac = (dists <= 0.35).sum() / len(coords)

                # Also record where sim.pos actually is
                pos_copy = sim.pos.copy()

                snapshots.append({
                    'step': step,
                    'coords': coords,
                    'fitness': fitness,
                    'summit_frac': summit_frac,
                    'pos': pos_copy,
                    'prefs': sim.prefs.copy(),
                })

            if step < max_step:
                # 1. Full toroidal physics step
                sim.step()

                # 2. Gradient nudge on prefs
                prefs = sim.prefs.astype(np.float64)
                grad_unit, fitness, peak_ids = LANDSCAPE.gradient(prefs)
                noise = sim.rng.normal(0, 1, grad_unit.shape)
                noise *= sim.role_gradient_noise[:, None]
                noisy_grad = grad_unit + noise
                norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
                noisy_grad = noisy_grad / np.maximum(norms, 1e-12)

                # Visionary blend
                to_summit = summit_center - prefs
                summit_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
                summit_dir = to_summit / np.maximum(summit_norms, 1e-12)
                v = sim.role_visionary[:, None]
                effective_grad = (1.0 - v) * noisy_grad + v * summit_dir
                eff_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
                effective_grad = effective_grad / np.maximum(eff_norms, 1e-12)

                step_scale = sim.role_step_scale[:, None]
                nudge = 0.003 * step_scale * effective_grad
                sim.prefs = np.clip(prefs + nudge, -1, 1).astype(sim.prefs.dtype)

                # 3. Sync sim.pos to mountain projection (THE FIX)
                coords = sim.prefs.astype(np.float64)
                fitness, _ = LANDSCAPE.fitness(coords)
                sim.pos[:, 0] = (coords[:, 0] + 1.0) * 0.5
                sim.pos[:, 1] = fitness * 0.5  # z_scale = 0.5
                if sim.pos.shape[1] > 2:
                    sim.pos[:, 2] = (coords[:, 1] + 1.0) * 0.5 if K > 1 else 0.5

        return {'label': label, 'snapshots': snapshots}
    finally:
        _global_params.clear()
        _global_params.update(snapshot)


def render_surface(ax, landscape, alpha=0.25, resolution=50):
    """Render the fitness landscape as a wireframe surface."""
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.zeros((resolution * resolution, K), dtype=np.float64)
    grid_points[:, 0] = X.ravel()
    grid_points[:, 1] = Y.ravel()
    Z_fitness, _ = landscape.fitness(grid_points)
    Z = Z_fitness.reshape(resolution, resolution)
    norm = Normalize(vmin=0, vmax=1.0)
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=alpha, norm=norm,
                    linewidth=0, antialiased=True, zorder=1)


def render_snapshot(ax, snap, landscape, title_extra=''):
    """Render a single snapshot on a 3D axis."""
    render_surface(ax, landscape, alpha=0.2, resolution=40)

    coords = snap['coords']
    fitness = snap['fitness']
    prefs = snap['prefs']

    # Color by prefs
    rgb = (prefs[:, :3].astype(float) + 1) / 2
    rgb = np.clip(rgb, 0, 1)

    ax.scatter(coords[:, 0], coords[:, 1], fitness,
               c=rgb, s=6, alpha=0.7, depthshade=True, zorder=5)

    # Mark summit
    sf, _ = landscape.fitness(SUMMIT.reshape(1, -1))
    ax.scatter([SUMMIT[0]], [SUMMIT[1]], [sf[0]],
               c='gold', s=120, marker='*', edgecolors='black',
               linewidths=0.5, zorder=10)

    # Mark local peaks
    if hasattr(landscape, 'major_centers'):
        for i in range(1, len(landscape.major_centers)):
            lp = landscape.major_centers[i]
            lf, _ = landscape.fitness(lp.reshape(1, -1))
            ax.scatter([lp[0]], [lp[1]], [lf[0]],
                       c='red', s=60, marker='^', edgecolors='black',
                       linewidths=0.5, zorder=10)

    step = snap['step']
    sfrac = snap['summit_frac']
    ax.set_title(f"Step {step}\nSummit: {sfrac:.1%}{title_extra}",
                 fontsize=9, pad=5)
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel('Dim 0', fontsize=7)
    ax.set_ylabel('Dim 1', fontsize=7)
    ax.set_zlabel('Fitness', fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.1)
    ax.tick_params(labelsize=6)


def render_pos_debug(ax, snap, title=''):
    """Render sim.pos in 3D to see where the physics engine thinks particles are."""
    pos = snap['pos']
    prefs = snap['prefs']
    rgb = (prefs[:, :3].astype(float) + 1) / 2
    rgb = np.clip(rgb, 0, 1)

    ax.scatter(pos[:, 0], pos[:, 1],
               pos[:, 2] if pos.shape[1] > 2 else np.zeros(len(pos)),
               c=rgb, s=6, alpha=0.7, depthshade=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('pos[0]', fontsize=7)
    ax.set_ylabel('pos[1]', fontsize=7)
    ax.set_zlabel('pos[2]', fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    os.makedirs(os.path.join(_pkg_root, 'results'), exist_ok=True)

    # ── Condition definitions ──
    conditions = [
        {
            'label': 'Baseline (no roles, social=0)',
            'overrides': {
                'social': 0.0,
                'use_particle_roles': False,
            },
        },
        {
            'label': 'Roles + social=0.01',
            'overrides': {
                'social': 0.01,
                'use_particle_roles': True,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.15,
                'role_visionary_std': 0.08,
                'role_visionary_fraction': 0.10,
            },
        },
        {
            'label': 'Roles + social=0.05',
            'overrides': {
                'social': 0.05,
                'use_particle_roles': True,
                'role_gradient_noise_mean': 0.5,
                'role_gradient_noise_std': 0.2,
                'role_visionary_mean': 0.15,
                'role_visionary_std': 0.08,
                'role_visionary_fraction': 0.10,
            },
        },
    ]

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 1: Headless convergence (gradient nudge only)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("FIGURE 1: Headless convergence (gradient nudge on prefs)")
    print("=" * 60)

    n_conds = len(conditions)
    n_snaps = len(SNAPSHOT_STEPS)

    fig1, axes1 = plt.subplots(n_conds, n_snaps, figsize=(4 * n_snaps, 4 * n_conds),
                               subplot_kw={'projection': '3d'})
    fig1.suptitle('Headless Convergence: Gradient Nudge on Prefs\n'
                  f'({N_PARTICLES} particles, K={K})',
                  fontsize=14, fontweight='bold', y=1.01)

    for ci, cond in enumerate(conditions):
        print(f"  [{ci+1}/{n_conds}] {cond['label']}...")
        result = run_headless(cond['label'], cond['overrides'])
        for si, snap in enumerate(result['snapshots']):
            ax = axes1[ci, si] if n_conds > 1 else axes1[si]
            extra = f"\n{cond['label']}" if si == 0 else ''
            render_snapshot(ax, snap, LANDSCAPE, title_extra=extra)

    plt.tight_layout()
    out1 = os.path.join(_pkg_root, 'results', 'convergence_headless.png')
    fig1.savefig(out1, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out1}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 2: Visualizer-mimic convergence (sim.step + nudge + pos sync)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FIGURE 2: Visualizer-mimic (sim.step + nudge + pos sync)")
    print("=" * 60)

    fig2, axes2 = plt.subplots(n_conds, n_snaps, figsize=(4 * n_snaps, 4 * n_conds),
                               subplot_kw={'projection': '3d'})
    fig2.suptitle('Visualizer-Mimic Convergence: sim.step() + Gradient Nudge + Pos Sync\n'
                  f'({N_PARTICLES} particles, K={K})',
                  fontsize=14, fontweight='bold', y=1.01)

    for ci, cond in enumerate(conditions):
        print(f"  [{ci+1}/{n_conds}] {cond['label']}...")
        result = run_visualizer_mimic(cond['label'], cond['overrides'])
        for si, snap in enumerate(result['snapshots']):
            ax = axes2[ci, si] if n_conds > 1 else axes2[si]
            extra = f"\n{cond['label']}" if si == 0 else ''
            render_snapshot(ax, snap, LANDSCAPE, title_extra=extra)

    plt.tight_layout()
    out2 = os.path.join(_pkg_root, 'results', 'convergence_visualizer.png')
    fig2.savefig(out2, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out2}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 3: sim.pos debug — where does the physics engine think
    # particles are in the visualizer-mimic mode?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FIGURE 3: sim.pos debug (visualizer-mimic)")
    print("=" * 60)

    # Run one condition and show sim.pos at each snapshot
    cond = conditions[1]  # Roles + social=0.01
    result = run_visualizer_mimic(cond['label'], cond['overrides'])

    fig3, axes3 = plt.subplots(2, n_snaps, figsize=(4 * n_snaps, 8),
                               subplot_kw={'projection': '3d'})
    fig3.suptitle(f'Debug: sim.pos vs Mountain Projection\n'
                  f'Condition: {cond["label"]}',
                  fontsize=14, fontweight='bold', y=1.01)

    for si, snap in enumerate(result['snapshots']):
        # Top row: mountain projection (prefs)
        render_snapshot(axes3[0, si], snap, LANDSCAPE,
                        title_extra='\n(mountain projection)')
        # Bottom row: raw sim.pos
        render_pos_debug(axes3[1, si], snap,
                         title=f'Step {snap["step"]}\nsim.pos (raw)')
        axes3[1, si].view_init(elev=30, azim=-60)

    plt.tight_layout()
    out3 = os.path.join(_pkg_root, 'results', 'convergence_pos_debug.png')
    fig3.savefig(out3, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out3}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # FIGURE 4: Top-down 2D view for clearer spatial analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FIGURE 4: Top-down 2D view (headless)")
    print("=" * 60)

    fig4, axes4 = plt.subplots(n_conds, n_snaps, figsize=(3.5 * n_snaps, 3.5 * n_conds))
    fig4.suptitle('Top-Down View: Particle Positions in Pref Space\n'
                  f'(headless, {N_PARTICLES} particles, K={K})',
                  fontsize=14, fontweight='bold', y=1.01)

    for ci, cond in enumerate(conditions):
        print(f"  [{ci+1}/{n_conds}] {cond['label']}...")
        result = run_headless(cond['label'], cond['overrides'])
        for si, snap in enumerate(result['snapshots']):
            ax = axes4[ci, si] if n_conds > 1 else axes4[si]

            coords = snap['coords']
            fitness = snap['fitness']

            # Contour plot of landscape
            x = np.linspace(-1, 1, 80)
            y = np.linspace(-1, 1, 80)
            X, Y = np.meshgrid(x, y)
            grid = np.zeros((6400, K), dtype=np.float64)
            grid[:, 0] = X.ravel()
            grid[:, 1] = Y.ravel()
            Z, _ = LANDSCAPE.fitness(grid)
            Z = Z.reshape(80, 80)
            ax.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.4)
            ax.contour(X, Y, Z, levels=10, colors='gray', linewidths=0.3, alpha=0.5)

            # Particles colored by fitness
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=fitness,
                            cmap='hot', s=4, alpha=0.7, vmin=0, vmax=1)

            # Summit marker
            ax.plot(SUMMIT[0], SUMMIT[1], '*', color='gold', markersize=12,
                    markeredgecolor='black', markeredgewidth=0.5)

            # Local peaks
            if hasattr(LANDSCAPE, 'major_centers'):
                for i in range(1, len(LANDSCAPE.major_centers)):
                    lp = LANDSCAPE.major_centers[i]
                    ax.plot(lp[0], lp[1], '^', color='red', markersize=8,
                            markeredgecolor='black', markeredgewidth=0.5)

            step = snap['step']
            sfrac = snap['summit_frac']
            title = f"Step {step} | Summit: {sfrac:.1%}"
            if si == 0:
                title = f"{cond['label']}\n{title}"
            ax.set_title(title, fontsize=8)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out4 = os.path.join(_pkg_root, 'results', 'convergence_topdown.png')
    fig4.savefig(out4, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out4}")
    plt.close()

    print("\nAll figures saved!")


if __name__ == '__main__':
    main()
