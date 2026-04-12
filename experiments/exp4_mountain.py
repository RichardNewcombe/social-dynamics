#!/usr/bin/env python3
"""
Experiment 4 — Mountain Climbing with Heterogeneous Roles & Cost
================================================================
Models an organisation trying to discover the best strategy (climb a
fitness mountain in preference space) while managing costs.

Role dimensions (per-particle, set in core sim):
  - **Researcher** (``role_gradient_noise``): noise std on gradient
    sensing.  Lower = better researcher = more expensive.
  - **Leader** (``role_influence``): social-learning weight.
  - **Engineer** (``role_step_scale``): movement magnitude multiplier.
  - **Visionary** (``role_visionary``): blend weight toward the true
    global summit.  Provides a faint directional signal but ignores
    local terrain — too many visionaries homogenise the search and
    march through expensive terrain.

Cost model:
  - **Terrain cost**: a second landscape layer (independent of fitness)
    that makes some paths through strategy space expensive.  The direct
    route to the summit crosses a high-cost ridge.
  - **Employee cost**: per-step cost proportional to each particle's
    capability: base + w_e*step_scale + w_l*influence + w_r*(1/noise)
    + w_v*visionary.  Specialists are expensive.
  - **Total org cost**: accumulated sum of (terrain + employee) cost
    across all particles and all steps.

The experiment metric is the **cost-efficiency frontier**: summit
fraction vs total cost.  The best team composition maximises fitness
gained per unit cost.

Usage
-----
    python -m experiments.exp4_mountain [--trials 3] [--max-steps 5000]
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
from experiments.landscape import (
    make_default_landscape,
    make_default_cost_landscape,
    compute_employee_cost,
)

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ── Experiment constants ────────────────────────────────────────────────

K = 3
NUM_PARTICLES = 500
GRADIENT_STRENGTH = 0.003       # base gradient nudge magnitude
SUMMIT_RADIUS = 0.35            # distance threshold for "at the summit"

LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)


# ── Callbacks ───────────────────────────────────────────────────────────

def make_post_step(landscape, cost_landscape, gradient_strength,
                   cost_accumulator):
    """Return a post-step hook with visionary blend and cost tracking.

    The effective gradient for each particle is:
        g_eff = (1 - v_i) * noisy_local_grad + v_i * direction_to_summit

    where v_i is the particle's visionary weight.  The noisy local
    gradient uses per-particle noise from ``sim.role_gradient_noise``.

    Cost is accumulated every step:
        cost_accumulator['total'] += sum(terrain_cost + employee_cost)
    """
    global_center = landscape.centers[0]

    def post_step(sim, step):
        prefs = sim.prefs.astype(np.float64)

        # ── True local gradient ──
        grad_unit, fitness, peak_ids = landscape.gradient(prefs)

        # ── Per-particle noise (researcher factor from core sim) ──
        noise = sim.rng.normal(0, 1, grad_unit.shape)
        noise *= sim.role_gradient_noise[:, None]
        noisy_grad = grad_unit + noise
        norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        noisy_grad = noisy_grad / norms

        # ── Visionary signal (direction to global summit) ──
        to_summit = global_center - prefs  # (N, K)
        summit_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
        summit_norms = np.maximum(summit_norms, 1e-12)
        summit_dir = to_summit / summit_norms  # unit vector toward summit

        # ── Blend: (1 - v) * local + v * summit ──
        v = sim.role_visionary[:, None]  # (N, 1)
        effective_grad = (1.0 - v) * noisy_grad + v * summit_dir
        # Re-normalize
        eff_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
        eff_norms = np.maximum(eff_norms, 1e-12)
        effective_grad = effective_grad / eff_norms

        # ── Apply nudge ──
        nudge = gradient_strength * effective_grad
        sim.prefs = np.clip(prefs + nudge, -1, 1).astype(sim.prefs.dtype)
        apply_post_processing(sim)

        # ── Accumulate cost ──
        terrain_cost = cost_landscape.cost(sim.prefs)  # (N,)
        employee_cost = compute_employee_cost(sim)       # (N,)
        step_cost = float((terrain_cost + employee_cost).sum())
        cost_accumulator['total'] += step_cost
        cost_accumulator['terrain'] += float(terrain_cost.sum())
        cost_accumulator['employee'] += float(employee_cost.sum())

    return post_step


def make_log(landscape, cost_accumulator, summit_radius=SUMMIT_RADIUS):
    """Return a logging function with fitness and cost metrics."""
    def log_fn(sim, step):
        fitness, peak_ids = landscape.fitness(sim.prefs)

        # Summit fraction
        global_center = landscape.centers[0]
        diff = sim.prefs.astype(np.float64) - global_center
        dists_to_summit = np.linalg.norm(diff, axis=1)
        summit_frac = float(
            (dists_to_summit <= summit_radius).sum() / len(sim.prefs))

        # Local trap fraction
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
            'total_cost': cost_accumulator['total'],
            'terrain_cost': cost_accumulator['terrain'],
            'employee_cost': cost_accumulator['employee'],
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
        description="Experiment 4: Mountain Climbing with Roles & Cost")
    parser.add_argument('--trials', type=int, default=3,
                        help='Monte Carlo trials per condition')
    parser.add_argument('--max-steps', type=int, default=5_000,
                        help='Max simulation steps per trial')
    parser.add_argument('--output', type=str,
                        default='results/exp4_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # ── Conditions to compare ──
    # Organised into groups: baseline, social, roles, visionary, cost-aware
    conditions = [
        # --- Baselines ---
        {
            'label': 'Baseline (no roles, no social)',
            'social': 0.0,
            'use_particle_roles': False,
            'memory_field': False,
        },
        {
            'label': 'Social conformity (s=+0.01)',
            'social': 0.01,
            'use_particle_roles': False,
            'memory_field': False,
        },
        {
            'label': 'Social differentiation (s=-0.01)',
            'social': -0.01,
            'use_particle_roles': False,
            'memory_field': False,
        },
        # --- Role heterogeneity ---
        {
            'label': 'Full roles (leaders + engineers)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.0,
            'role_visionary_std': 0.0,
            'memory_field': False,
        },
        # --- Visionary sweep ---
        {
            'label': 'Few visionaries (vis_mean=0.05)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.05,
            'role_visionary_std': 0.03,
            'memory_field': False,
        },
        {
            'label': 'Moderate visionaries (vis_mean=0.15)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.15,
            'role_visionary_std': 0.08,
            'memory_field': False,
        },
        {
            'label': 'Many visionaries (vis_mean=0.40)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.40,
            'role_visionary_std': 0.15,
            'memory_field': False,
        },
        {
            'label': 'All visionaries (vis_mean=0.80)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.80,
            'role_visionary_std': 0.10,
            'memory_field': False,
        },
        # --- World knowledge + visionaries ---
        {
            'label': 'Moderate vis + memory (strength=3)',
            'social': 0.01,
            'use_particle_roles': True,
            'role_influence_std': 0.8,
            'role_step_scale_std': 0.5,
            'role_gradient_noise_mean': 0.5,
            'role_gradient_noise_std': 0.2,
            'role_visionary_mean': 0.15,
            'role_visionary_std': 0.08,
            'memory_field': True,
            'memory_strength': 3.0,
            'memory_decay': 0.999,
            'memory_write_rate': 0.05,
        },
    ]

    print(f"=== Experiment 4: Mountain Climbing with Cost ===")
    print(f"Fitness landscape: {LANDSCAPE.n_peaks} peaks")
    for i, p in enumerate(LANDSCAPE.centers):
        h = LANDSCAPE.heights[i]
        s = LANDSCAPE.sigmas[i]
        label = "GLOBAL" if i == 0 else f"local {i}"
        print(f"  Peak {i} ({label}): center={p}, height={h:.2f}, "
              f"sigma={s:.2f}")
    print(f"Cost landscape: {COST_LANDSCAPE.n_ridges} ridges, "
          f"base_cost={COST_LANDSCAPE.base_cost}")
    print(f"Gradient strength: {GRADIENT_STRENGTH}")
    print(f"Summit radius: {SUMMIT_RADIUS}")
    print(f"Trials per condition: {args.trials}")
    print()

    all_results = []

    for cond in conditions:
        label = cond['label']
        print(f"\n--- Condition: {label} ---")

        # Build param overrides
        overrides = {
            'num_particles': NUM_PARTICLES,
            'k': K,
            'physics_engine': 1,
            'social_mode': 0,
            'social': cond['social'],
            'use_particle_roles': cond.get('use_particle_roles', False),
            'role_influence_std': cond.get('role_influence_std', 0.0),
            'role_step_scale_std': cond.get('role_step_scale_std', 0.0),
            'role_gradient_noise_mean': cond.get(
                'role_gradient_noise_mean', 0.5),
            'role_gradient_noise_std': cond.get(
                'role_gradient_noise_std', 0.0),
            'role_visionary_mean': cond.get('role_visionary_mean', 0.0),
            'role_visionary_std': cond.get('role_visionary_std', 0.0),
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

        for trial in range(args.trials):
            # Fresh cost accumulator per trial
            cost_acc = {'total': 0.0, 'terrain': 0.0, 'employee': 0.0}

            result = run_experiment(
                param_overrides=overrides,
                max_steps=args.max_steps,
                check_fn=make_check(LANDSCAPE, SUMMIT_RADIUS, 0.30),
                post_step_fn=make_post_step(
                    LANDSCAPE, COST_LANDSCAPE, GRADIENT_STRENGTH, cost_acc),
                log_fn=make_log(LANDSCAPE, cost_acc),
                log_interval=200,
                seed=trial * 1000 + 4444,
            )
            result['condition'] = label
            result['trial'] = trial
            result['final_cost'] = dict(cost_acc)
            all_results.append(result)

            status = "SOLVED" if result['solved'] else "FAILED"
            if result['log']:
                last = result['log'][-1]
                summit = last.get('summit_frac', 0)
                trap = last.get('local_trap_frac', 0)
                fit = last.get('mean_fitness', 0)
                total_cost = cost_acc['total']
                print(f"  trial={trial}: {status} at step "
                      f"{result['solve_step']} "
                      f"(summit={summit:.1%}, trap={trap:.1%}, "
                      f"fit={fit:.3f}, cost={total_cost:.0f}, "
                      f"{result['wall_time']:.1f}s)")
            else:
                print(f"  trial={trial}: {status} "
                      f"({result['wall_time']:.1f}s)")

    # ── Summary ──
    print("\n=== Summary ===")
    header = (f"{'Condition':<42} | {'Solved':>8} | {'Summit%':>8} | "
              f"{'Trap%':>8} | {'Total Cost':>12} | {'Emp Cost':>12}")
    print(header)
    print("-" * len(header))
    for cond in conditions:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        n_solved = sum(1 for r in trials if r['solved'])
        summit_fracs = []
        trap_fracs = []
        total_costs = []
        emp_costs = []
        for r in trials:
            if r['log']:
                summit_fracs.append(r['log'][-1].get('summit_frac', 0))
                trap_fracs.append(r['log'][-1].get('local_trap_frac', 0))
            total_costs.append(r['final_cost']['total'])
            emp_costs.append(r['final_cost']['employee'])
        mean_summit = np.mean(summit_fracs) if summit_fracs else 0
        mean_trap = np.mean(trap_fracs) if trap_fracs else 0
        mean_cost = np.mean(total_costs) if total_costs else 0
        mean_emp = np.mean(emp_costs) if emp_costs else 0
        print(f"{label:<42} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_summit:>7.1%} | {mean_trap:>7.1%} | "
              f"{mean_cost:>12.0f} | {mean_emp:>12.0f}")

    # ── Pareto analysis ──
    print("\n=== Cost-Efficiency Frontier ===")
    print("(Higher summit% at lower cost is better)")
    cond_stats = []
    for cond in conditions:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        summit_fracs = [r['log'][-1].get('summit_frac', 0)
                        for r in trials if r['log']]
        costs = [r['final_cost']['total'] for r in trials]
        mean_s = np.mean(summit_fracs) if summit_fracs else 0
        mean_c = np.mean(costs) if costs else 0
        efficiency = mean_s / (mean_c + 1) * 1e6  # summit% per M cost
        cond_stats.append((label, mean_s, mean_c, efficiency))

    # Sort by efficiency descending
    cond_stats.sort(key=lambda x: x[3], reverse=True)
    print(f"{'Condition':<42} | {'Summit%':>8} | {'Cost':>12} | "
          f"{'Efficiency':>12}")
    print("-" * 82)
    for label, s, c, eff in cond_stats:
        print(f"{label:<42} | {s:>7.1%} | {c:>12.0f} | {eff:>12.2f}")

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
