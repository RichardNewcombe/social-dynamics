#!/usr/bin/env python3
"""
Headless runner for the hierarchical (L0 + L1) simulation.

Renders particles colored by preference, bonds as lines,
cluster membership as convex hulls, and L1 centroids with
arrows showing L1 inter-group attraction.

Usage:
    cd ~/workspace/social-dynamics
    python -m sim_2d_cuda.headless_hierarchical
    python -m sim_2d_cuda.headless_hierarchical --steps 3000 --particles 800
"""

import argparse
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull


def render_frame(sim, step, output_path, figsize=(12, 10)):
    """Render particles, bonds, cluster hulls, and L1 centroids."""
    pos, colors, labels = sim.get_render_data()
    bond_data = sim.get_bond_data(min_strength=0.05)
    n_bonds = int((sim.bonds > 0.05).sum()) // 2

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.L)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Draw bonds ──
    if bond_data:
        segments = []
        bond_colors = []
        for p_i, p_j, strength in bond_data:
            delta = sim._periodic_delta(p_i, p_j)
            dist = np.sqrt((delta ** 2).sum())
            if dist < sim.p['bond_radius'] * 1.5:
                end = p_i + delta
                segments.append([(p_i[0], p_i[1]), (end[0], end[1])])
                bond_colors.append((0.3, 0.7, 1.0, float(strength) * 0.6))
        if segments:
            lc = LineCollection(segments, colors=bond_colors, linewidths=0.5)
            ax.add_collection(lc)

    # ── Draw cluster hulls ──
    hull_colors_palette = plt.cm.Set2(np.linspace(0, 1, max(sim.n_clusters, 1)))
    for c in range(sim.n_clusters):
        members = np.where(labels == c)[0]
        if len(members) < 3:
            continue

        member_pos = pos[members]
        # Use centroid-relative positions to handle wrapping
        ref = sim.l1_centroids[c]
        deltas = sim._periodic_delta(ref, member_pos)
        local_pts = deltas  # relative to centroid

        try:
            hull = ConvexHull(local_pts)
            hull_verts = local_pts[hull.vertices]
            # Close the polygon
            hull_verts = np.vstack([hull_verts, hull_verts[0]])
            # Shift back to absolute coords
            abs_verts = sim._periodic_wrap(ref + hull_verts)

            hc = hull_colors_palette[c % len(hull_colors_palette)]
            ax.fill(abs_verts[:, 0], abs_verts[:, 1],
                    alpha=0.12, color=hc, zorder=1)
            ax.plot(abs_verts[:, 0], abs_verts[:, 1],
                    alpha=0.4, color=hc, linewidth=1.5, zorder=1)
        except Exception:
            pass  # degenerate hull (collinear points)

    # ── Draw free particles (small) ──
    free = labels == -1
    if free.any():
        ax.scatter(pos[free, 0], pos[free, 1],
                   c=colors[free], s=6, alpha=0.6,
                   edgecolors='none', zorder=2)

    # ── Draw bonded particles (slightly larger) ──
    bonded = labels >= 0
    if bonded.any():
        ax.scatter(pos[bonded, 0], pos[bonded, 1],
                   c=colors[bonded], s=14, alpha=0.9,
                   edgecolors='white', linewidths=0.3, zorder=3)

    # ── Draw L1 centroids ──
    if sim.n_clusters > 0:
        cx = sim.l1_centroids[:, 0]
        cy = sim.l1_centroids[:, 1]

        # Color centroids by their L1 preference
        if sim.k >= 3:
            cent_colors = np.clip((sim.l1_prefs[:, :3] + 1.0) * 0.5, 0, 1)
        else:
            cent_colors = np.full((sim.n_clusters, 3), 0.5)

        ax.scatter(cx, cy, c=cent_colors, s=80, alpha=1.0,
                   edgecolors='gold', linewidths=2.0, zorder=5,
                   marker='D')

    # ── Title ──
    n_free = int((labels == -1).sum())
    n_bonded = int((labels >= 0).sum())
    cluster_sizes = ""
    if sim.n_clusters > 0:
        sizes = [int((labels == c).sum()) for c in range(sim.n_clusters)]
        cluster_sizes = f"  sizes={sorted(sizes, reverse=True)}"

    title = (f"Step {step}   |   free:{n_free}  bonded:{n_bonded}  "
             f"L1 groups:{sim.n_clusters}   |   bonds:{n_bonds}"
             f"{cluster_sizes}")
    ax.set_title(title, color='white', fontsize=11, fontweight='bold',
                 pad=10, fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  [{step:>5d}] saved → {output_path}   "
          f"(free={n_free} bonded={n_bonded} L1={sim.n_clusters})")


def make_gif(frame_paths, output_path, duration_ms=400):
    try:
        from PIL import Image
        frames = [Image.open(p).copy() for p in frame_paths]
        if frames:
            frames[0].save(output_path, save_all=True,
                           append_images=frames[1:],
                           duration=duration_ms, loop=0, optimize=True)
            print(f"\n  GIF saved → {output_path}")
    except Exception as e:
        print(f"  (GIF failed: {e})")


def main():
    parser = argparse.ArgumentParser(
        description='Headless hierarchical L0+L1 simulation')
    parser.add_argument('--particles', type=int, default=500)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step-size', type=float, default=0.005)
    parser.add_argument('--n-neighbors', type=int, default=21)
    parser.add_argument('--bond-radius', type=float, default=0.08)
    parser.add_argument('--bond-alpha', type=float, default=0.01)
    parser.add_argument('--bond-beta', type=float, default=0.005)
    parser.add_argument('--bond-pref-weight', type=float, default=0.5)
    parser.add_argument('--bond-threshold', type=float, default=0.85)
    parser.add_argument('--cluster-min-size', type=int, default=3)
    parser.add_argument('--l1-step-size', type=float, default=0.005)
    parser.add_argument('--l1-n-neighbors', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='hierarchical_frames')
    parser.add_argument('--snapshot-steps', type=str,
                        default='1,25,50,100,200,300,500,750,1000,1500,2000')
    args = parser.parse_args()

    snapshot_steps = set()
    for s in args.snapshot_steps.split(','):
        s = s.strip()
        if s:
            snapshot_steps.add(int(s))
    snapshot_steps.add(args.steps)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Hierarchical L0 + L1 Simulation")
    print(f"  particles={args.particles}  k={args.k}  steps={args.steps}")
    print(f"  L0: step_size={args.step_size}  n_neighbors={args.n_neighbors}")
    print(f"  Bonds: alpha={args.bond_alpha}  beta={args.bond_beta}")
    print(f"         pref_weight={args.bond_pref_weight}  radius={args.bond_radius}")
    print(f"         threshold={args.bond_threshold}  min_size={args.cluster_min_size}")
    print(f"  L1: step_size={args.l1_step_size}  n_neighbors={args.l1_n_neighbors}")
    print(f"  snapshots at: {sorted(snapshot_steps)}")
    print(f"  output → {out_dir}/")
    print("=" * 65)

    from .hierarchical import HierarchicalSimulation

    sim = HierarchicalSimulation(
        num_particles=args.particles,
        k=args.k,
        seed=args.seed,
        step_size=args.step_size,
        n_neighbors=args.n_neighbors,
        bond_radius=args.bond_radius,
        bond_alpha=args.bond_alpha,
        bond_beta=args.bond_beta,
        bond_pref_weight=args.bond_pref_weight,
        bond_threshold=args.bond_threshold,
        cluster_min_size=args.cluster_min_size,
        l1_step_size=args.l1_step_size,
        l1_n_neighbors=args.l1_n_neighbors,
    )

    frame_paths = []
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        sim.step()

        if step in snapshot_steps:
            path = os.path.join(out_dir, f"hier_step_{step:05d}.png")
            render_frame(sim, step, path)
            frame_paths.append(path)

        if step % max(1, args.steps // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            h = sim.history[-1] if sim.history else {}
            print(f"  progress: {step}/{args.steps}  "
                  f"({rate:.0f} steps/s)  "
                  f"free={h.get('n_free', '?')}  "
                  f"bonded={h.get('n_bonded', '?')}  "
                  f"L1={h.get('n_clusters', '?')}  "
                  f"bonds={h.get('n_bonds', '?')}")

    total = time.perf_counter() - t0
    print(f"\nDone: {args.steps} steps in {total:.1f}s ({args.steps/total:.0f} steps/s)")

    if sim.history:
        h = sim.history[-1]
        print(f"\nFinal state:")
        print(f"  Free agents: {h['n_free']}")
        print(f"  Bonded particles: {h['n_bonded']}")
        print(f"  L1 groups: {h['n_clusters']}")
        if h['cluster_sizes']:
            print(f"  Cluster sizes: {sorted(h['cluster_sizes'], reverse=True)}")
        print(f"  Active bonds: {h['n_bonds']}")

    if frame_paths:
        gif_path = os.path.join(out_dir, "hierarchical.gif")
        make_gif(frame_paths, gif_path, duration_ms=500)

    print(f"\nFrames in: {out_dir}/")


if __name__ == '__main__':
    main()
