#!/usr/bin/env python3
"""
Headless runner for sim_2d_exp — no display required.

Runs the simulation for N steps and saves particle positions + preferences
as PNG images (left panel = particle positions, right panel = trail accumulation).

Usage:
    python -m sim_2d_exp.headless                    # defaults
    python -m sim_2d_exp.headless --steps 500 --particles 2000
    python -m sim_2d_exp.headless --steps 1000 --output result.png
    python -m sim_2d_exp.headless --config '{"step_size": 0.01, "social": -0.003}'

Requires: numpy, scipy, numba, torch (optional), matplotlib (for PNG output)
Does NOT require: glfw, moderngl, imgui, OpenGL
"""

import argparse
import json
import time
import numpy as np


def run_headless(args):
    from .params import params, SPACE
    from .simulation import Simulation

    # Apply any config overrides
    if args.config:
        overrides = json.loads(args.config)
        params.update(overrides)

    # Apply CLI overrides
    if args.particles is not None:
        params['num_particles'] = args.particles
    if args.k is not None:
        params['k'] = args.k

    # Ensure sensible defaults for headless
    params.setdefault('physics_engine', 2)  # PyTorch
    params.setdefault('torch_precision', 3)  # f64
    params.setdefault('use_f64', True)
    params.setdefault('knn_method', 1)  # cKDTree f64

    n_steps = args.steps
    N = params['num_particles']
    K = params['k']

    print(f"Headless run: N={N}, K={K}, steps={n_steps}")
    print(f"Engine: {params['physics_engine']}, step_size={params['step_size']}")

    sim = Simulation()

    # Run simulation
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step()
        if step % max(1, n_steps // 10) == 0 or step == n_steps:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            print(f"  step {step}/{n_steps}  ({rate:.0f} steps/s)")

    total = time.perf_counter() - t0
    print(f"Done: {n_steps} steps in {total:.1f}s ({n_steps/total:.0f} steps/s)")

    # Generate output images
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping image output")
        # Save raw data instead
        np.savez(args.output.replace('.png', '.npz'),
                 pos=sim.pos, prefs=sim.prefs, response=sim.response,
                 movement=sim._movement)
        print(f"Raw data saved to {args.output.replace('.png', '.npz')}")
        return

    L = SPACE
    pos = sim.pos
    prefs = sim.prefs

    # Colors from preferences (same as renderer)
    colors = np.clip((prefs[:, :3] + 1.0) * 0.5, 0, 1)
    if K < 3:
        c = np.full((N, 3), 0.5)
        c[:, :min(K, 3)] = colors[:, :min(K, 3)]
        colors = c

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#141419')

    # Left panel: particle positions
    ax = axes[0]
    ax.set_facecolor('#141419')
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=args.point_size**2,
               edgecolors='none', alpha=0.9)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title('Particle Positions', color='white', fontsize=14)
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_color('gray')

    # Right panel: preference space (2D scatter of dim0 vs dim1)
    ax2 = axes[1]
    ax2.set_facecolor('#141419')
    if K >= 2:
        ax2.scatter(prefs[:, 0], prefs[:, 1], c=colors, s=args.point_size**2,
                    edgecolors='none', alpha=0.9)
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel('pref dim 0', color='gray')
        ax2.set_ylabel('pref dim 1', color='gray')
    else:
        ax2.scatter(prefs[:, 0], np.zeros(N), c=colors, s=args.point_size**2,
                    edgecolors='none', alpha=0.9)
        ax2.set_xlim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.set_title('Preference Space', color='white', fontsize=14)
    ax2.tick_params(colors='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Image saved to {args.output}")

    # Also save raw data if requested
    if args.save_data:
        data_path = args.output.replace('.png', '.npz')
        np.savez(data_path,
                 pos=sim.pos, prefs=sim.prefs, response=sim.response,
                 movement=sim._movement, memory_field=sim.memory_field,
                 memory_flow=sim.memory_flow)
        print(f"Data saved to {data_path}")


def main():
    parser = argparse.ArgumentParser(description='Headless sim_2d_exp runner')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of simulation steps')
    parser.add_argument('--particles', type=int, default=None,
                        help='Override number of particles')
    parser.add_argument('--k', type=int, default=None,
                        help='Override preference dimensions')
    parser.add_argument('--output', type=str, default='sim_output.png',
                        help='Output image path')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output image DPI')
    parser.add_argument('--point-size', type=float, default=3.0,
                        help='Particle point size in plot')
    parser.add_argument('--save-data', action='store_true',
                        help='Also save raw numpy data (.npz)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON string of param overrides')
    args = parser.parse_args()

    run_headless(args)


if __name__ == '__main__':
    main()
