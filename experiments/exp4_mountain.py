#!/usr/bin/env python3
"""
Experiment 4 — Mountain Climbing with Heterogeneous Roles
=========================================================
Models an organisation trying to discover the best strategy (climb a
fitness mountain in preference space) when:

  - No one can see the summit directly.
  - Each person has a **researcher** factor that controls how accurately
    they sense the true gradient (low noise = good researcher).
  - Each person has a **leader** factor (``role_influence``) that weights
    how much their preferences pull neighbours during social learning.
  - Each person has an **engineer** factor (``role_step_scale``) that
    controls how far they move per step.
  - The **memory field** encodes historical "world knowledge" — prior
    beliefs about which direction is uphill.  These may point toward a
    local peak, not the global one.
  - Group dynamics (social learning rate, neighbour radius, conformity
    vs differentiation) determine whether the team converges on one
    path, explores multiple routes, or gets stuck on a local peak.

The mountain is a multi-peak Gaussian fitness landscape defined in
K-dimensional preference space.  There is one tall global peak and
several shorter local peaks.  The gradient nudge applied each step
simulates each person's noisy private reading of the terrain.

Design
------
Phase 1 (optional): Memory field pre-seeded with gradient pointing
    toward a local peak (simulating historical world knowledge).
Phase 2: Particles start near the origin and climb.  Each step:
    1. ``sim.step()`` runs full physics (movement, social learning,
       memory field read/write).
    2. Post-step: compute true gradient at each particle's pref position.
    3. Add per-particle Gaussian noise (researcher factor).
    4. Apply noisy gradient nudge (scaled by global ``gradient_strength``).

Metrics
-------
  - **Peak Fitness**: mean and max fitness across all particles.
  - **Summit Fraction**: fraction of particles near the global peak.
  - **Local Trap %**: fraction stuck on local peaks.
  - **Exploration Rate**: mean pref change per step.
  - **Team Coherence**: DBSCAN cluster tightness.
  - **Path Diversity**: number of distinct clusters in pref space.

Sweep Variables
---------------
  - ``social`` — social learning rate (conformity vs independence)
  - ``role_influence_std`` — leader heterogeneity
  - ``role_step_scale_std`` — engineer heterogeneity
  - ``gradient_noise_std`` — researcher noise (lower = better researchers)
  - ``memory_strength`` — world knowledge influence

Usage
-----
    python -m experiments.exp4_mountain [--trials 5] [--max-steps 10000]
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
from experiments.landscape import make_default_landscape

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── Experiment constants ────────────────────────────────────────────────

K = 3
NUM_PARTICLES = 500
GRADIENT_STRENGTH = 0.003       # base gradient nudge magnitude
GRADIENT_NOISE_STD = 0.5        # default researcher noise (0 = perfect sensing)
SUMMIT_RADIUS = 0.35            # distance threshold for "at the summit"

LANDSCAPE = make_default_landscape(k=K)


# ── Callbacks ───────────────────────────────────────────────────────────

def make_post_step(landscape, gradient_strength, noise_std_per_particle):
    """Return a post-step hook that applies noisy gradient nudges.

    Each particle senses the true gradient with per-particle noise,
    simulating heterogeneous research ability.

    Parameters
    ----------
    landscape : GaussianPeakLandscape
    gradient_strength : float
    noise_std_per_particle : (N,) array — per-particle noise std
        Lower values = better researchers (clearer gradient signal).
    """
    def post_step(sim, step):
        grad_unit, fitness, peak_ids = landscape.gradient(sim.prefs)

        # Per-particle noise (researcher factor)
        noise = sim.rng.normal(0, 1, grad_unit.shape)
        noise *= noise_std_per_particle[:, None]

        noisy_grad = grad_unit + noise
        # Re-normalize
        norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        noisy_grad = noisy_grad / norms

        # Apply nudge
        nudge = gradient_strength * noisy_grad
        sim.prefs = np.clip(
            sim.prefs.astype(np.float64) + nudge, -1, 1
        ).astype(sim.prefs.dtype)
        apply_post_processing(sim)

    return post_step


def make_log(landscape, summit_radius=SUMMIT_RADIUS):
    """Return a logging function that records mountain-climbing metrics."""
    def log_fn(sim, step):
        fitness, peak_ids = landscape.fitness(sim.prefs)

        # Summit fraction (particles near global peak = peak 0)
        global_center = landscape.centers[0]
        diff = sim.prefs.astype(np.float64) - global_center
        dists_to_summit = np.linalg.norm(diff, axis=1)
        summit_frac = float((dists_to_summit <= summit_radius).sum() / len(sim.prefs))

        # Local trap fraction (near any non-global peak)
        local_trap = 0
        for pid in range(1, landscape.n_peaks):
            diff_l = sim.prefs.astype(np.float64) - landscape.centers[pid]
            dists_l = np.linalg.norm(diff_l, axis=1)
            local_trap += (dists_l <= summit_radius).sum()
        local_trap_frac = float(local_trap / len(sim.prefs))

        # Cluster count
        cluster_count = 0
        if _HAS_SKLEARN:
            db = DBSCAN(eps=0.35, min_samples=5).fit(
                sim.prefs.astype(np.float64))
            labels = db.labels_
            cluster_count = len(set(labels) - {-1})

        return {
            'mean_fitness': float(fitness.mean()),
            'max_fitness': float(fitness.max()),
            'summit_frac': summit_frac,
            'local_trap_frac': local_trap_frac,
            'cluster_count': cluster_count,
            'pref_std': float(sim.prefs.std(axis=0).mean()),
        }
    return log_fn


def make_check(landscape, summit_radius, min_summit_frac=0.30):
    """Solved when enough particles reach the global summit."""
    def check(sim, step):
        global_center = landscape.centers[0]
        diff = sim.prefs.astype(np.float64) - global_center
        dists = np.linalg.norm(diff, axis=1)
        frac = (dists <= summit_radius).sum() / len(sim.prefs)
        return frac >= min_summit_frac
    return check


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Mountain Climbing with Heterogeneous Roles")
    parser.add_argument('--trials', type=int, default=3,
                        help='Monte Carlo trials per condition')
    parser.add_argument('--max-steps', type=int, default=10_000,
                        help='Max simulation steps per trial')
    parser.add_argument('--output', type=str,
                        default='results/exp4_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # ── Conditions to compare ──
    # Each condition varies one aspect of the org design
    conditions = [
        {
            'label': 'Baseline (no roles, no social)',
            'social': 0.0,
            'use_particle_roles': False,
            'role_influence_std': 0.0,
            'role_step_scale_std': 0.0,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'Social conformity (s=+0.01)',
            'social': 0.01,
            'use_particle_roles': False,
            'role_influence_std': 0.0,
            'role_step_scale_std': 0.0,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'Social differentiation (s=-0.01)',
            'social': -0.01,
            'use_particle_roles': False,
            'role_influence_std': 0.0,
            'role_step_scale_std': 0.0,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'Heterogeneous leaders (inf_std=0.8)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.0,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'Heterogeneous engineers (step_std=0.5)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.0,
            'role_step_scale_std': 0.5,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'Full roles (leaders + engineers)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': False,
        },
        {
            'label': 'World knowledge (memory, strength=3)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'gradient_noise_std': GRADIENT_NOISE_STD,
            'memory_field': True,
            'memory_strength': 3.0,
            'memory_decay': 0.999,
            'memory_write_rate': 0.05,
        },
    ]

    print(f"=== Experiment 4: Mountain Climbing ===")
    print(f"Landscape: {LANDSCAPE.n_peaks} peaks")
    for i, p in enumerate(LANDSCAPE.centers):
        h = LANDSCAPE.heights[i]
        s = LANDSCAPE.sigmas[i]
        label = "GLOBAL" if i == 0 else f"local {i}"
        print(f"  Peak {i} ({label}): center={p}, height={h:.2f}, sigma={s:.2f}")
    print(f"Gradient strength: {GRADIENT_STRENGTH}")
    print(f"Summit radius: {SUMMIT_RADIUS}")
    print(f"Trials per condition: {args.trials}")
    print()

    all_results = []

    for cond in conditions:
        label = cond['label']
        print(f"\n--- Condition: {label} ---")

        # Build param overrides (only what the experiment must control)
        overrides = {
            'num_particles': NUM_PARTICLES,
            'k': K,
            'physics_engine': 1,
            'social_mode': 0,
            'social': cond['social'],
            'use_particle_roles': cond.get('use_particle_roles', False),
            'role_influence_std': cond.get('role_influence_std', 0.0),
            'role_step_scale_std': cond.get('role_step_scale_std', 0.0),
        }
        if cond.get('memory_field', False):
            overrides.update({
                'memory_field': True,
                'memory_strength': cond.get('memory_strength', 3.0),
                'memory_decay': cond.get('memory_decay', 0.999),
                'memory_write_rate': cond.get('memory_write_rate', 0.05),
                'memory_blur': True,
                'memory_blur_sigma': 1.0,
            })
        else:
            overrides['memory_field'] = False

        noise_std = cond.get('gradient_noise_std', GRADIENT_NOISE_STD)

        for trial in range(args.trials):
            # Create per-particle noise array (researcher factor)
            # All particles get the same noise std in the base conditions;
            # future conditions could vary this per-particle.
            noise_per_particle = np.full(NUM_PARTICLES, noise_std)

            result = run_experiment(
                param_overrides=overrides,
                max_steps=args.max_steps,
                check_fn=make_check(LANDSCAPE, SUMMIT_RADIUS, 0.30),
                post_step_fn=make_post_step(
                    LANDSCAPE, GRADIENT_STRENGTH, noise_per_particle),
                log_fn=make_log(LANDSCAPE),
                log_interval=200,
                seed=trial * 1000 + 4444,
            )
            result['condition'] = label
            result['trial'] = trial
            all_results.append(result)

            status = "SOLVED" if result['solved'] else "FAILED"
            # Get final metrics
            if result['log']:
                last = result['log'][-1]
                summit = last.get('summit_frac', 0)
                trap = last.get('local_trap_frac', 0)
                fit = last.get('mean_fitness', 0)
                print(f"  trial={trial}: {status} at step {result['solve_step']} "
                      f"(summit={summit:.1%}, trap={trap:.1%}, "
                      f"fit={fit:.3f}, {result['wall_time']:.1f}s)")
            else:
                print(f"  trial={trial}: {status} ({result['wall_time']:.1f}s)")

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"{'Condition':<40} | {'Solved':>8} | {'Mean Steps':>12} | "
          f"{'Summit%':>8} | {'Trap%':>8}")
    print("-" * 90)
    for cond in conditions:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        n_solved = sum(1 for r in trials if r['solved'])
        steps = [r['solve_step'] for r in trials if r['solved']]
        mean_steps = np.mean(steps) if steps else float('inf')
        # Get final summit/trap fracs
        summit_fracs = []
        trap_fracs = []
        for r in trials:
            if r['log']:
                summit_fracs.append(r['log'][-1].get('summit_frac', 0))
                trap_fracs.append(r['log'][-1].get('local_trap_frac', 0))
        mean_summit = np.mean(summit_fracs) if summit_fracs else 0
        mean_trap = np.mean(trap_fracs) if trap_fracs else 0
        print(f"{label:<40} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_steps:>12.0f} | {mean_summit:>7.1%} | {mean_trap:>7.1%}")

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
        for k_name, v in r.items():
            if k_name == 'log':
                sr[k_name] = [{kk: convert(vv) for kk, vv in entry.items()}
                              for entry in v]
            else:
                sr[k_name] = convert(v)
        serializable.append(sr)

    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2, default=convert)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
