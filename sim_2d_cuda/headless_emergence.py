#!/usr/bin/env python3
"""
Headless runner for the hierarchical emergence simulation.

Produces PNG snapshots and an animated GIF showing particles forming
bonds, clustering, and merging into higher-level entities.

Usage:
    cd ~/workspace/social-dynamics
    python -m sim_2d_cuda.headless_emergence
    python -m sim_2d_cuda.headless_emergence --steps 3000 --particles 800
"""

import argparse
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


def render_frame(sim, step, output_path, figsize=(12, 10)):
    """Render a single frame showing particles, bonds, and hierarchy."""
    pos, colors, masses, levels = sim.get_render_data()
    bond_data = sim.get_bond_data(min_strength=0.05)
    counts = sim.count_by_level()
    n_bonds = sim.count_bonds(min_strength=0.05)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    ax.set_xlim(0, sim.L)
    ax.set_ylim(0, sim.L)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Draw bonds as lines with opacity = strength ──
    if bond_data:
        segments = []
        bond_colors = []
        for p_i, p_j, strength in bond_data:
            # Handle periodic wrapping: only draw if distance < radius
            delta = sim._periodic_delta(p_i, p_j)
            dist = np.sqrt((delta ** 2).sum())
            if dist < sim.neighbor_radius * 1.5:
                end = p_i + delta
                segments.append([(p_i[0], p_i[1]), (end[0], end[1])])
                bond_colors.append((0.3, 0.7, 1.0, float(strength) * 0.6))

        if segments:
            lc = LineCollection(segments, colors=bond_colors, linewidths=0.5)
            ax.add_collection(lc)

    # ── Draw particles by level ──
    for lv in range(sim.ep['merge_max_level'] + 1):
        mask = levels == lv
        if not mask.any():
            continue

        lv_pos = pos[mask]
        lv_colors = colors[mask]
        lv_masses = masses[mask]

        if lv == 0:
            # Small dots
            sizes = 8 * np.ones(len(lv_pos))
            ax.scatter(lv_pos[:, 0], lv_pos[:, 1],
                       c=lv_colors, s=sizes, alpha=0.8,
                       edgecolors='none', zorder=2)
        elif lv == 1:
            # Larger circles with white border
            sizes = 30 * lv_masses ** 0.8
            ax.scatter(lv_pos[:, 0], lv_pos[:, 1],
                       c=lv_colors, s=sizes, alpha=0.95,
                       edgecolors='white', linewidths=1.5, zorder=3)
        else:
            # Level 2+: even larger with double border
            sizes = 60 * lv_masses ** 0.8
            ax.scatter(lv_pos[:, 0], lv_pos[:, 1],
                       c=lv_colors, s=sizes, alpha=1.0,
                       edgecolors='gold', linewidths=2.5, zorder=4)
            # Inner white ring
            ax.scatter(lv_pos[:, 0], lv_pos[:, 1],
                       c=lv_colors, s=sizes * 0.7, alpha=1.0,
                       edgecolors='white', linewidths=1.0, zorder=5)

    # ── Title with stats ──
    level_str = "  ".join([f"L{lv}:{counts.get(lv, 0)}" for lv in sorted(counts)])
    n_merges = len(sim.merge_events)
    title = f"Step {step}   |   {level_str}   |   bonds: {n_bonds}   |   merges: {n_merges}"
    ax.set_title(title, color='white', fontsize=13, fontweight='bold',
                 pad=10, fontfamily='monospace')

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  [{step:>5d}] saved → {output_path}   ({level_str})")


def make_gif(frame_paths, output_path, duration_ms=400):
    """Stitch PNG frames into an animated GIF."""
    try:
        from PIL import Image
        frames = []
        for p in frame_paths:
            img = Image.open(p)
            frames.append(img.copy())
            img.close()
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
            print(f"\n  GIF saved → {output_path}")
    except ImportError:
        print("  (Pillow not available — skipping GIF)")
    except Exception as e:
        print(f"  (GIF failed: {e})")


def main():
    parser = argparse.ArgumentParser(
        description='Headless hierarchical emergence simulation')
    parser.add_argument('--particles', type=int, default=500)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step-size', type=float, default=0.004)
    parser.add_argument('--radius', type=float, default=0.07)
    parser.add_argument('--repulsion', type=float, default=0.0005)
    parser.add_argument('--social', type=float, default=0.002)
    parser.add_argument('--bond-alpha', type=float, default=0.01)
    parser.add_argument('--bond-beta', type=float, default=0.005)
    parser.add_argument('--bond-pref-weight', type=float, default=0.5)
    parser.add_argument('--merge-threshold', type=float, default=0.85)
    parser.add_argument('--merge-min-size', type=int, default=3)
    parser.add_argument('--merge-max-level', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='emergence_frames')
    parser.add_argument('--snapshot-steps', type=str,
                        default='1,50,100,200,300,500,750,1000,1500,2000')
    args = parser.parse_args()

    # Parse snapshot steps
    snapshot_steps = set()
    for s in args.snapshot_steps.split(','):
        s = s.strip()
        if s:
            snapshot_steps.add(int(s))
    snapshot_steps.add(args.steps)  # always snapshot final

    # Output directory
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Hierarchical Emergence Simulation")
    print(f"  particles={args.particles}  k={args.k}  steps={args.steps}")
    print(f"  step_size={args.step_size}  radius={args.radius}")
    print(f"  repulsion={args.repulsion}  social={args.social}")
    print(f"  bond_alpha={args.bond_alpha}  bond_beta={args.bond_beta}")
    print(f"  bond_pref_weight={args.bond_pref_weight}")
    print(f"  merge_threshold={args.merge_threshold}")
    print(f"  merge_min_size={args.merge_min_size}")
    print(f"  merge_max_level={args.merge_max_level}")
    print(f"  snapshots at: {sorted(snapshot_steps)}")
    print(f"  output → {out_dir}/")
    print("=" * 65)

    from .emergence import EmergenceSimulation

    sim = EmergenceSimulation(
        num_particles=args.particles,
        k=args.k,
        seed=args.seed,
        step_size=args.step_size,
        neighbor_radius=args.radius,
        repulsion=args.repulsion,
        social=args.social,
        bond_alpha=args.bond_alpha,
        bond_beta=args.bond_beta,
        bond_pref_weight=args.bond_pref_weight,
        merge_threshold=args.merge_threshold,
        merge_min_size=args.merge_min_size,
        merge_max_level=args.merge_max_level,
    )

    frame_paths = []
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        sim.step()

        if step in snapshot_steps:
            path = os.path.join(out_dir, f"emergence_step_{step:05d}.png")
            render_frame(sim, step, path)
            frame_paths.append(path)

        # Progress every 10%
        if step % max(1, args.steps // 10) == 0:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            counts = sim.count_by_level()
            n_bonds = sim.count_bonds()
            lstr = " ".join([f"L{lv}:{c}" for lv, c in sorted(counts.items())])
            print(f"  progress: {step}/{args.steps}  "
                  f"({rate:.0f} steps/s)  {lstr}  bonds={n_bonds}")

    total = time.perf_counter() - t0
    print(f"\nDone: {args.steps} steps in {total:.1f}s ({args.steps/total:.0f} steps/s)")

    # Summary
    counts = sim.count_by_level()
    print(f"\nFinal state:")
    for lv, c in sorted(counts.items()):
        print(f"  Level {lv}: {c} particles")
    print(f"  Active bonds: {sim.count_bonds()}")
    print(f"  Total merge events: {len(sim.merge_events)}")
    if sim.merge_events:
        print(f"  First merge at step {sim.merge_events[0][0]}")

    # Make GIF
    if frame_paths:
        gif_path = os.path.join(out_dir, "emergence.gif")
        make_gif(frame_paths, gif_path, duration_ms=500)

    print(f"\nFrames in: {out_dir}/")


if __name__ == '__main__':
    main()
