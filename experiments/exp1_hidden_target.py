#!/usr/bin/env python3
"""
Experiment 1 — Hidden Target Search
====================================
Tests how the exploration/exploitation balance (controlled by the social
learning rate) affects the time to find a hidden optimal preference vector.

Hypothesis
----------
- High positive social learning (conformity) causes premature convergence
  to local clusters, yielding long or infinite solve times.
- Moderate negative social learning (differentiation) maintains diversity,
  allowing the swarm to flow toward the hidden target faster.

Design
------
A hidden target vector P_target is defined in K-dimensional preference space.
After each simulation step, a weak fitness gradient nudges every particle's
preferences slightly toward the target.  The problem is "solved" when a
cluster of at least M particles has preferences within distance epsilon of
the target.

The sweep variable is ``social`` (the social learning rate).

All other simulation parameters inherit from the user's current
configuration (GUI or params.py), falling back to SAFE_DEFAULTS for
headless CLI runs.

Usage
-----
    python -m experiments.exp1_hidden_target [--trials 10] [--max-steps 20000]
"""

import argparse
import json
import os
import sys
import numpy as np

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from experiments.runner import run_experiment, run_sweep, apply_post_processing
from experiments.fitness import hidden_target_fitness, hidden_target_gradient

# ── Experiment constants ────────────────────────────────────────────────

K = 3                                         # preference dimensions
TARGET = np.array([0.7, -0.5, 0.3])          # hidden optimal vector
GRADIENT_STRENGTH = 0.002                     # environmental signal strength
EPSILON = 0.30                                # distance threshold
M_CLUSTER = 15                                # particles needed near target
NUM_PARTICLES = 500                           # smaller for fast headless runs

# Social learning rates to sweep — from strong differentiation to strong conformity
SWEEP_VALUES = [-0.03, -0.01, -0.003, 0.0, 0.003, 0.01, 0.03]


# ── Callbacks ───────────────────────────────────────────────────────────

def make_post_step(target, strength):
    """Return a post-step hook that applies the fitness gradient.

    After modifying prefs, calls apply_post_processing() to honour any
    active normalisation / quantisation params the user may have enabled.
    """
    def post_step(sim, step):
        nudge = hidden_target_gradient(sim.prefs, target, strength)
        sim.prefs = np.clip(
            sim.prefs.astype(np.float64) + nudge, -1, 1
        ).astype(sim.prefs.dtype)
        apply_post_processing(sim)
    return post_step


def make_check(target, epsilon, m_cluster):
    """Return a check function: solved when M particles within epsilon."""
    def check(sim, step):
        diff = sim.prefs.astype(np.float64) - target.astype(np.float64)
        dists = np.linalg.norm(diff, axis=1)
        near = (dists <= epsilon).sum()
        return near >= m_cluster
    return check


def make_log(target):
    """Return a logging function that records key metrics."""
    def log_fn(sim, step):
        fitness = hidden_target_fitness(sim.prefs, target)
        diff = sim.prefs.astype(np.float64) - target.astype(np.float64)
        dists = np.linalg.norm(diff, axis=1)
        return {
            'mean_fitness': float(fitness.mean()),
            'max_fitness': float(fitness.max()),
            'min_dist': float(dists.min()),
            'near_target': int((dists <= EPSILON).sum()),
            'pref_std': float(sim.prefs.std(axis=0).mean()),
        }
    return log_fn


# ── Experiment factory ──────────────────────────────────────────────────

def experiment_factory(param_overrides):
    """Build runner kwargs for a single trial."""
    return {
        'check_fn': make_check(TARGET, EPSILON, M_CLUSTER),
        'post_step_fn': make_post_step(TARGET, GRADIENT_STRENGTH),
        'log_fn': make_log(TARGET),
        'log_interval': 200,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Hidden Target Search")
    parser.add_argument('--trials', type=int, default=5,
                        help='Monte Carlo trials per sweep value')
    parser.add_argument('--max-steps', type=int, default=20_000,
                        help='Max simulation steps per trial')
    parser.add_argument('--output', type=str, default='results/exp1_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # Only override the params this experiment needs to control.
    # Everything else inherits from the user's current state,
    # falling back to SAFE_DEFAULTS for headless runs.
    base_overrides = {
        'num_particles': NUM_PARTICLES,
        'k': K,
        'social_mode': 0,     # Uniform social learning (experiment design)
        'physics_engine': 1,  # NumPy — headless-safe
    }

    print(f"=== Experiment 1: Hidden Target Search ===")
    print(f"Target: {TARGET}")
    print(f"Sweep: social = {SWEEP_VALUES}")
    print(f"Trials per value: {args.trials}")
    print(f"Max steps: {args.max_steps}")
    print()

    results = run_sweep(
        experiment_fn=experiment_factory,
        sweep_param='social',
        sweep_values=SWEEP_VALUES,
        n_trials=args.trials,
        base_overrides=base_overrides,
        max_steps=args.max_steps,
    )

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"{'social':>10} | {'solved':>8} | {'mean_steps':>12} | {'mean_time':>10}")
    print("-" * 50)
    for val in SWEEP_VALUES:
        trials = [r for r in results if r['sweep_value'] == val]
        n_solved = sum(1 for r in trials if r['solved'])
        steps = [r['solve_step'] for r in trials if r['solved']]
        mean_steps = np.mean(steps) if steps else float('inf')
        mean_time = np.mean([r['wall_time'] for r in trials])
        print(f"{val:>10.3f} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_steps:>12.0f} | {mean_time:>9.1f}s")

    # ── Save results ──
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
    for r in results:
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
