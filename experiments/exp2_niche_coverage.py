#!/usr/bin/env python3
"""
Experiment 2 — Multi-Niche Coverage Problem
============================================
Tests whether the simulation can maintain cognitive diversity to cover
multiple distinct "expertise niches" simultaneously.

Hypothesis
----------
- Uniform social learning (social_mode=0) homogenises the swarm into one
  or two large clusters, leaving most niches empty.
- Quiet-Dimension Differentiation (social_mode=1) allows physical
  coordination while preserving diversity across secondary preference
  dimensions, enabling faster and more reliable niche coverage.

Design
------
Four well-separated niches are defined in K-dimensional preference space.
After each step, a mild attraction gradient pulls each particle toward the
nearest niche centre (simulating a landscape with multiple viable solutions).
The problem is "solved" when every niche contains at least ``min_per_niche``
particles.

The sweep variable is ``social_mode`` (0 = Uniform, 1 = Quiet-Dim Diff),
with a secondary sweep over the social learning rate.

All other simulation parameters inherit from the user's current
configuration (GUI or params.py), falling back to SAFE_DEFAULTS for
headless CLI runs.

Usage
-----
    python -m experiments.exp2_niche_coverage [--trials 10] [--max-steps 20000]
"""

import argparse
import json
import os
import sys
import numpy as np

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from experiments.runner import run_experiment, apply_post_processing
from experiments.fitness import niche_occupancy, generate_niches

# ── Experiment constants ────────────────────────────────────────────────

K = 3
NUM_PARTICLES = 500
NICHE_RADIUS = 0.35          # Euclidean distance in pref-space
MIN_PER_NICHE = 15           # particles required per niche to count
GRADIENT_STRENGTH = 0.0005   # attraction toward nearest niche
NICHES = generate_niches(K, n_niches=4, seed=42)

# Conditions to compare
CONDITIONS = [
    {'label': 'Uniform s=+0.01',    'social_mode': 0, 'social': 0.01},
    {'label': 'Uniform s=-0.01',    'social_mode': 0, 'social': -0.01},
    {'label': 'QuietDim s=+0.01',   'social_mode': 1, 'social': 0.01},
    {'label': 'QuietDim s=-0.01',   'social_mode': 1, 'social': -0.01},
    {'label': 'No social s=0.0',    'social_mode': 0, 'social': 0.0},
]


# ── Callbacks ───────────────────────────────────────────────────────────

def make_post_step(niches, strength):
    """Attract each particle toward its nearest niche centre.

    After modifying prefs, calls apply_post_processing() to honour any
    active normalisation / quantisation params the user may have enabled.
    """
    def post_step(sim, step):
        prefs_f64 = sim.prefs.astype(np.float64)
        niches_f64 = niches.astype(np.float64)
        # Find nearest niche for each particle
        dists = np.linalg.norm(
            prefs_f64[:, None, :] - niches_f64[None, :, :], axis=2
        )  # (N, M)
        nearest = np.argmin(dists, axis=1)  # (N,)
        target = niches_f64[nearest]  # (N, K)
        diff = target - prefs_f64
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        nudge = strength * (diff / norms)
        sim.prefs = np.clip(
            prefs_f64 + nudge, -1, 1
        ).astype(sim.prefs.dtype)
        apply_post_processing(sim)
    return post_step


def make_check(niches, radius, min_per_niche):
    """Solved when every niche has at least min_per_niche particles."""
    def check(sim, step):
        counts, occupied = niche_occupancy(sim.prefs, niches, radius)
        return bool(np.all(counts >= min_per_niche))
    return check


def make_log(niches, radius):
    """Log niche occupancy and diversity metrics."""
    def log_fn(sim, step):
        counts, occupied = niche_occupancy(sim.prefs, niches, radius)
        return {
            'niches_occupied': int(occupied.sum()),
            'niche_counts': counts.tolist(),
            'total_in_niches': int(counts.sum()),
            'pref_std': float(sim.prefs.std(axis=0).mean()),
        }
    return log_fn


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Multi-Niche Coverage")
    parser.add_argument('--trials', type=int, default=5,
                        help='Monte Carlo trials per condition')
    parser.add_argument('--max-steps', type=int, default=20_000,
                        help='Max simulation steps per trial')
    parser.add_argument('--output', type=str,
                        default='results/exp2_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # Only override the params this experiment needs to control.
    base_overrides = {
        'num_particles': NUM_PARTICLES,
        'k': K,
        'physics_engine': 1,  # NumPy — headless-safe
    }

    print(f"=== Experiment 2: Multi-Niche Coverage ===")
    print(f"Niches ({len(NICHES)}):")
    for i, n in enumerate(NICHES):
        print(f"  Niche {i}: {n}")
    print(f"Niche radius: {NICHE_RADIUS}")
    print(f"Min per niche: {MIN_PER_NICHE}")
    print(f"Trials per condition: {args.trials}")
    print()

    all_results = []

    for cond in CONDITIONS:
        label = cond['label']
        print(f"\n--- Condition: {label} ---")
        overrides = {
            **base_overrides,
            'social_mode': cond['social_mode'],
            'social': cond['social'],
        }

        for trial in range(args.trials):
            result = run_experiment(
                param_overrides=overrides,
                max_steps=args.max_steps,
                check_fn=make_check(NICHES, NICHE_RADIUS, MIN_PER_NICHE),
                post_step_fn=make_post_step(NICHES, GRADIENT_STRENGTH),
                log_fn=make_log(NICHES, NICHE_RADIUS),
                log_interval=200,
                seed=trial * 1000 + 7777,
            )
            result['condition'] = label
            result['social_mode'] = cond['social_mode']
            result['social'] = cond['social']
            result['trial'] = trial
            all_results.append(result)

            status = "SOLVED" if result['solved'] else "FAILED"
            print(f"  trial={trial}: {status} at step {result['solve_step']} "
                  f"({result['wall_time']:.1f}s)")

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"{'Condition':<25} | {'Solved':>8} | {'Mean Steps':>12} | {'Mean Time':>10}")
    print("-" * 65)
    for cond in CONDITIONS:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        n_solved = sum(1 for r in trials if r['solved'])
        steps = [r['solve_step'] for r in trials if r['solved']]
        mean_steps = np.mean(steps) if steps else float('inf')
        mean_time = np.mean([r['wall_time'] for r in trials])
        print(f"{label:<25} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_steps:>12.0f} | {mean_time:>9.1f}s")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if k == 'log':
                sr[k] = [{kk: convert(vv) for kk, vv in entry.items()}
                         for entry in v]
            else:
                sr[k] = convert(v)
        serializable.append(sr)

    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2, default=convert)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
