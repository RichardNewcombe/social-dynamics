#!/usr/bin/env python3
"""
Render hierarchical sim at l1_weight=0.1 with both particle and trail views.
Steps: 1, 100, 1000, 10000
"""

import numpy as np
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

from sim_hierarchical.hierarchical import HierarchicalSimulation


def render_particles(sim, step, output_path, figsize=(10, 10)):
    """Render particle positions with bonds and cluster hulls."""
    pos, colors, labels = sim.get_render_data()
    L = sim.L

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.axis('off')

    # Bonds
    ii, jj, strengths = sim.get_bond_arrays(min_strength=0.1)
    if len(ii) > 0:
        segments = []
        bond_colors = []
        for idx in range(len(ii)):
            delta = sim._periodic_delta(sim.pos[ii[idx]], sim.pos[jj[idx]])
            dist = np.sqrt((delta ** 2).sum())
            if dist < sim.p['bond_radius'] * 2:
                p1 = sim.pos[ii[idx]]
                p2 = p1 + delta
                segments.append([(p1[0], p1[1]), (p2[0], p2[1])])
                bond_colors.append((0.3, 0.7, 1.0, float(strengths[idx]) * 0.4))
        if segments:
            lc = LineCollection(segments, colors=bond_colors, linewidths=0.3)
            ax.add_collection(lc)

    # Cluster hulls
    if sim.n_clusters > 0:
        hull_cmap = plt.cm.Set2(np.linspace(0, 1, max(sim.n_clusters, 1)))
        for c in range(sim.n_clusters):
            members = np.where(labels == c)[0]
            if len(members) < 3:
                continue
            ref = sim.l1_centroids[c]
            deltas = sim._periodic_delta(ref, pos[members])
            try:
                hull = ConvexHull(deltas)
                verts = deltas[hull.vertices]
                verts = np.vstack([verts, verts[0]])
                abs_v = (ref + verts) % L
                hc = hull_cmap[c % len(hull_cmap)]
                ax.fill(abs_v[:, 0], abs_v[:, 1], alpha=0.08, color=hc)
                ax.plot(abs_v[:, 0], abs_v[:, 1], alpha=0.3, color=hc, linewidth=1)
            except:
                pass

    # Free particles
    free = labels == -1
    if free.any():
        ax.scatter(pos[free, 0], pos[free, 1], c=colors[free], s=6, alpha=0.6, edgecolors='none', zorder=2)

    # Bonded particles
    bonded = labels >= 0
    if bonded.any():
        ax.scatter(pos[bonded, 0], pos[bonded, 1], c=colors[bonded], s=14, alpha=0.9,
                   edgecolors='white', linewidths=0.3, zorder=3)

    # L1 centroids
    if sim.n_clusters > 0:
        if sim.k >= 3:
            cc = np.clip((sim.l1_prefs[:, :3] + 1.0) * 0.5, 0, 1)
        else:
            cc = np.full((sim.n_clusters, 3), 0.5)
        ax.scatter(sim.l1_centroids[:, 0], sim.l1_centroids[:, 1],
                   c=cc, s=80, alpha=1.0, edgecolors='gold', linewidths=2.0, zorder=5, marker='D')

    n_free = int(free.sum())
    n_bonded = int(bonded.sum())
    h = sim.history[-1] if sim.history else {}
    title = f"Step {step}  |  free:{n_free}  bonded:{n_bonded}  L1:{sim.n_clusters}  bonds:{h.get('n_bonds', 0)}"
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10, fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def render_trails(trail_buffer, sim, step, output_path, figsize=(10, 10)):
    """Render accumulated trail buffer as an image."""
    L = sim.L
    IMG = trail_buffer.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    # Normalize trail buffer for display
    display = trail_buffer.copy()
    max_val = display.max()
    if max_val > 0:
        # Gamma correction for better visibility
        display = np.power(display / max_val, 0.5)

    ax.imshow(display, origin='lower', extent=[0, L, 0, L], interpolation='bilinear')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.axis('off')

    h = sim.history[-1] if sim.history else {}
    title = f"Trails — Step {step}  |  L1:{sim.n_clusters}  bonds:{h.get('n_bonds', 0)}"
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10, fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def deposit_trails(trail_buffer, pos, prefs, decay, L):
    """Deposit particle positions into trail buffer with preference-based coloring."""
    IMG = trail_buffer.shape[0]
    k = prefs.shape[1]

    # Decay existing trails
    trail_buffer *= decay

    # Convert positions to pixel coordinates
    px = ((pos[:, 0] / L) * IMG).astype(np.int32) % IMG
    py = ((pos[:, 1] / L) * IMG).astype(np.int32) % IMG

    # RGB from preferences
    if k >= 3:
        rgb = np.clip((prefs[:, :3] + 1.0) * 0.5, 0, 1)
    elif k == 2:
        rgb = np.zeros((len(prefs), 3))
        rgb[:, :2] = np.clip((prefs[:, :2] + 1.0) * 0.5, 0, 1)
        rgb[:, 2] = 0.5
    else:
        rgb = np.full((len(prefs), 3), 0.5)
        rgb[:, 0] = np.clip((prefs[:, 0] + 1.0) * 0.5, 0, 1)

    # Deposit with additive blending
    np.add.at(trail_buffer, (py, px, 0), rgb[:, 0] * 0.15)
    np.add.at(trail_buffer, (py, px, 1), rgb[:, 1] * 0.15)
    np.add.at(trail_buffer, (py, px, 2), rgb[:, 2] * 0.15)


def main():
    N = 500
    K = 3
    SEED = 42
    STEPS = 10000
    TRAIL_SIZE = 512
    TRAIL_DECAY = 0.995

    snapshot_steps = {1, 100, 1000, 10000}

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hierarchical_l1_01')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Hierarchical L0+L1 — l1_weight=0.1")
    print(f"  N={N}  K={K}  seed={SEED}  steps={STEPS}")
    print(f"  Snapshots at: {sorted(snapshot_steps)}")
    print(f"  Output: {out_dir}/")
    print("=" * 65)

    sim = HierarchicalSimulation(
        num_particles=N, k=K, seed=SEED,
        step_size=0.005, n_neighbors=21,
        bond_radius=0.08, bond_alpha=0.01, bond_beta=0.005,
        bond_pref_weight=0.5, bond_threshold=0.85, cluster_min_size=3,
        l1_step_size=0.005, l1_n_neighbors=5,
        l1_weight=0.1,
    )

    trail_buffer = np.zeros((TRAIL_SIZE, TRAIL_SIZE, 3), dtype=np.float64)

    t0 = time.perf_counter()

    for step in range(1, STEPS + 1):
        sim.step()
        deposit_trails(trail_buffer, sim.pos, sim.prefs, TRAIL_DECAY, sim.L)

        if step in snapshot_steps:
            # Particle view
            ppath = os.path.join(out_dir, f"particles_{step:05d}.png")
            render_particles(sim, step, ppath)

            # Trail view
            tpath = os.path.join(out_dir, f"trails_{step:05d}.png")
            render_trails(trail_buffer, sim, step, tpath)

            h = sim.history[-1]
            elapsed = time.perf_counter() - t0
            print(f"  [{step:>5d}] {elapsed:.1f}s  free={h['n_free']}  "
                  f"bonded={h['n_bonded']}  L1={h['n_clusters']}  "
                  f"bonds={h['n_bonds']}  sizes={sorted(h['cluster_sizes'], reverse=True)[:8]}")

        if step % 1000 == 0 and step not in snapshot_steps:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            h = sim.history[-1]
            print(f"  progress: {step}/{STEPS}  ({rate:.0f} steps/s)  "
                  f"L1={h['n_clusters']}  bonds={h['n_bonds']}")

    total = time.perf_counter() - t0
    print(f"\nDone: {STEPS} steps in {total:.1f}s ({STEPS/total:.0f} steps/s)")
    print(f"Frames in: {out_dir}/")


if __name__ == '__main__':
    main()
