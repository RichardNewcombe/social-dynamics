#!/usr/bin/env python3
"""
Experiment 5 — Dual-Space Mountain Climbing
============================================
Models an organisation where **preferences** (team identity / culture)
and **strategy** (mountain position) are separate state vectors with
a tuneable coupling parameter.

Architecture
------------
- **Phase 1** (sim.step): social dynamics in preference space — team
  formation, cultural memory, social learning.  This produces the
  neighbor graph (who talks to whom).
- **Phase 2** (sim.strategy_step): mountain navigation in strategy
  space — each particle senses the gradient with researcher noise,
  visionaries blend toward the summit, the team aggregates via leader
  influence, engineers scale the step.  The preference-space neighbor
  graph determines information flow.

Key parameter: ``pref_strategy_coupling``
  - 1.0 = specialist org (strategy ≈ preferences, same as Exp 4)
  - 0.0 = generalist org (strategy independent of preferences)
  - 0.5 = typical org (strategy initialised as noisy blend of prefs)

The experiment sweeps coupling alongside role composition and social
settings to find the Pareto frontier of summit% vs total cost.

Usage
-----
    python -m experiments.exp5_dual_space [--trials 3] [--max-steps 5000]
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
SUMMIT_RADIUS = 0.35

LANDSCAPE = make_default_landscape(k=K)
COST_LANDSCAPE = make_default_cost_landscape(k=K)


# ── Callbacks ───────────────────────────────────────────────────────────

def make_post_step(landscape, cost_landscape, cost_accumulator):
    """Return a post-step hook that calls strategy_step for mountain navigation.

    This is the Phase 2 hook: after sim.step() handles social dynamics
    in preference space, this function drives mountain navigation in
    strategy space using the team structure (neighbor graph) from Phase 1.
    """
    summit_center = landscape.centers[0]  # global peak

    def _gradient_fn(strategy):
        """Wrapper matching the strategy_step interface."""
        return landscape.gradient(strategy)

    def post_step(sim, step):
        # Phase 2: team-aggregated mountain navigation
        sim.strategy_step(gradient_fn=_gradient_fn,
                          summit_center=summit_center)

        # Accumulate cost (evaluated in strategy space, not pref space)
        if sim.strategy is not None:
            terrain_cost = cost_landscape.cost(sim.strategy)
        else:
            terrain_cost = cost_landscape.cost(sim.prefs)
        employee_cost = compute_employee_cost(sim)
        step_cost = float((terrain_cost + employee_cost).sum())
        cost_accumulator['total'] += step_cost
        cost_accumulator['terrain'] += float(terrain_cost.sum())
        cost_accumulator['employee'] += float(employee_cost.sum())

    return post_step


def make_log(landscape, cost_accumulator, summit_radius=SUMMIT_RADIUS):
    """Return a logging function with dual-space metrics."""
    def log_fn(sim, step):
        # Use strategy for mountain metrics if available, else prefs
        coords = sim.strategy if sim.strategy is not None else sim.prefs
        coords_f64 = coords.astype(np.float64)

        fitness, peak_ids = landscape.fitness(coords_f64)

        # Summit fraction (in strategy space)
        global_center = landscape.centers[0]
        diff = coords_f64 - global_center
        dists_to_summit = np.linalg.norm(diff, axis=1)
        summit_frac = float(
            (dists_to_summit <= summit_radius).sum() / len(coords))

        # Local trap fraction
        local_trap = 0
        for pid in range(1, landscape.n_peaks):
            diff_l = coords_f64 - landscape.centers[pid]
            dists_l = np.linalg.norm(diff_l, axis=1)
            local_trap += (dists_l <= summit_radius).sum()
        local_trap_frac = float(local_trap / len(coords))

        # Preference-space team metrics
        pref_std = float(sim.prefs.std(axis=0).mean())
        n_clusters = 0
        if _HAS_SKLEARN:
            db = DBSCAN(eps=0.35, min_samples=5).fit(
                sim.prefs.astype(np.float64))
            n_clusters = len(set(db.labels_) - {-1})

        # Strategy-space diversity
        strat_std = float(coords_f64.std(axis=0).mean())

        # Strategy-preference divergence (how far apart the two spaces are)
        if sim.strategy is not None and sim.strategy_k == sim.k:
            divergence = float(np.linalg.norm(
                coords_f64 - sim.prefs.astype(np.float64),
                axis=1).mean())
        else:
            divergence = 0.0

        return {
            'mean_fitness': float(fitness.mean()),
            'max_fitness': float(fitness.max()),
            'summit_frac': summit_frac,
            'local_trap_frac': local_trap_frac,
            'pref_std': pref_std,
            'strat_std': strat_std,
            'n_clusters': n_clusters,
            'pref_strat_divergence': divergence,
            'total_cost': cost_accumulator['total'],
            'terrain_cost': cost_accumulator['terrain'],
            'employee_cost': cost_accumulator['employee'],
        }
    return log_fn


def make_check(landscape, summit_radius, min_summit_frac=0.30):
    """Solved when enough particles reach the global summit in strategy space."""
    def check(sim, step):
        coords = sim.strategy if sim.strategy is not None else sim.prefs
        global_center = landscape.centers[0]
        diff = coords.astype(np.float64) - global_center
        dists = np.linalg.norm(diff, axis=1)
        frac = (dists <= summit_radius).sum() / len(coords)
        return frac >= min_summit_frac
    return check


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Dual-Space Mountain Climbing")
    parser.add_argument('--trials', type=int, default=3,
                        help='Monte Carlo trials per condition')
    parser.add_argument('--max-steps', type=int, default=5_000,
                        help='Max simulation steps per trial')
    parser.add_argument('--output', type=str,
                        default='results/exp5_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    # ── Base role settings (shared across conditions) ──
    ROLE_BASE = {
        'use_particle_roles': True,
        'role_influence_std': 0.8,
        'role_step_scale_std': 0.5,
        'role_gradient_noise_mean': 0.5,
        'role_gradient_noise_std': 0.2,
        'role_visionary_mean': 0.15,
        'role_visionary_std': 0.08,
        'role_visionary_fraction': 0.10,
    }

    # ── Conditions ──
    conditions = [
        # --- Coupling sweep (core question) ---
        {
            'label': 'Specialist (coupling=1.0)',
            'pref_strategy_coupling': 1.0,
            'social': 0.01,
            **ROLE_BASE,
        },
        {
            'label': 'High coupling (0.75)',
            'pref_strategy_coupling': 0.75,
            'social': 0.01,
            **ROLE_BASE,
        },
        {
            'label': 'Moderate coupling (0.50)',
            'pref_strategy_coupling': 0.50,
            'social': 0.01,
            **ROLE_BASE,
        },
        {
            'label': 'Low coupling (0.25)',
            'pref_strategy_coupling': 0.25,
            'social': 0.01,
            **ROLE_BASE,
        },
        {
            'label': 'Generalist (coupling=0.0)',
            'pref_strategy_coupling': 0.0,
            'social': 0.01,
            **ROLE_BASE,
        },
        # --- Social learning interaction ---
        {
            'label': 'Generalist + strong social (s=+0.03)',
            'pref_strategy_coupling': 0.0,
            'social': 0.03,
            **ROLE_BASE,
        },
        {
            'label': 'Specialist + differentiation (s=-0.01)',
            'pref_strategy_coupling': 1.0,
            'social': -0.01,
            **ROLE_BASE,
        },
        {
            'label': 'Moderate + no social (s=0)',
            'pref_strategy_coupling': 0.50,
            'social': 0.0,
            **ROLE_BASE,
        },
        # --- Rare visionaries at different couplings ---
        {
            'label': 'Rare vis + specialist (coupling=1.0)',
            'pref_strategy_coupling': 1.0,
            'social': 0.01,
            **{**ROLE_BASE,
               'role_visionary_mean': 0.60,
               'role_visionary_std': 0.10,
               'role_visionary_fraction': 0.05},
        },
        {
            'label': 'Rare vis + generalist (coupling=0.0)',
            'pref_strategy_coupling': 0.0,
            'social': 0.01,
            **{**ROLE_BASE,
               'role_visionary_mean': 0.60,
               'role_visionary_std': 0.10,
               'role_visionary_fraction': 0.05},
        },
        # --- Knowledge memory ---
        {
            'label': 'Moderate + knowledge memory',
            'pref_strategy_coupling': 0.50,
            'social': 0.01,
            'strategy_memory_enabled': True,
            'strategy_memory_strength': 0.5,
            'strategy_memory_write_rate': 0.01,
            'strategy_memory_decay': 0.999,
            **ROLE_BASE,
        },
        # --- No roles baseline ---
        {
            'label': 'Baseline (no roles, coupling=0.5)',
            'pref_strategy_coupling': 0.50,
            'social': 0.0,
            'use_particle_roles': False,
        },
    ]

    print(f"=== Experiment 5: Dual-Space Mountain Climbing ===")
    print(f"Fitness landscape: {LANDSCAPE.n_peaks} peaks")
    if hasattr(LANDSCAPE, 'major_centers'):
        for i in range(LANDSCAPE.n_major):
            p = LANDSCAPE.major_centers[i]
            h = LANDSCAPE.major_heights[i]
            s = LANDSCAPE.major_sigmas[i]
            label_str = "GLOBAL" if i == 0 else f"local {i}"
            print(f"  Major {i} ({label_str}): center={np.round(p,2)}, "
                  f"height={h:.2f}, sigma={s:.2f}")
        print(f"  + {LANDSCAPE.n_minor} minor peaks, "
              f"{len(LANDSCAPE.noise_freqs)} noise frequencies")
    else:
        for i, p in enumerate(LANDSCAPE.centers):
            h = LANDSCAPE.heights[i]
            s = LANDSCAPE.sigmas[i]
            label_str = "GLOBAL" if i == 0 else f"local {i}"
            print(f"  Peak {i} ({label_str}): center={p}, height={h:.2f}, "
                  f"sigma={s:.2f}")
    print(f"Cost landscape: {COST_LANDSCAPE.n_ridges} ridges, "
          f"base_cost={COST_LANDSCAPE.base_cost}")
    print(f"Summit radius: {SUMMIT_RADIUS}")
    print(f"Trials per condition: {args.trials}")
    print()

    all_results = []

    for cond in conditions:
        label = cond['label']
        print(f"\n--- Condition: {label} ---")

        # Build param overrides — always enable strategy mode
        overrides = {
            'num_particles': NUM_PARTICLES,
            'k': K,
            'physics_engine': 1,
            'social_mode': 0,
            # Strategy space
            'strategy_enabled': True,
            'strategy_k': K,
            'strategy_step_size': 0.003,
        }
        # Copy condition params into overrides
        for key, val in cond.items():
            if key != 'label':
                overrides[key] = val

        # Cultural memory (in preference space) — off by default
        if 'memory_field' not in overrides:
            overrides['memory_field'] = False

        for trial in range(args.trials):
            cost_acc = {'total': 0.0, 'terrain': 0.0, 'employee': 0.0}

            result = run_experiment(
                param_overrides=overrides,
                max_steps=args.max_steps,
                check_fn=make_check(LANDSCAPE, SUMMIT_RADIUS, 0.30),
                post_step_fn=make_post_step(
                    LANDSCAPE, COST_LANDSCAPE, cost_acc),
                log_fn=make_log(LANDSCAPE, cost_acc),
                log_interval=200,
                seed=trial * 1000 + 5555,
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
                div = last.get('pref_strat_divergence', 0)
                total_cost = cost_acc['total']
                print(f"  trial={trial}: {status} at step "
                      f"{result['solve_step']} "
                      f"(summit={summit:.1%}, trap={trap:.1%}, "
                      f"fit={fit:.3f}, div={div:.3f}, "
                      f"cost={total_cost:.0f}, "
                      f"{result['wall_time']:.1f}s)")
            else:
                print(f"  trial={trial}: {status} "
                      f"({result['wall_time']:.1f}s)")

    # ── Summary ──
    print("\n=== Summary ===")
    header = (f"{'Condition':<42} | {'Solved':>8} | {'Summit%':>8} | "
              f"{'Trap%':>8} | {'Diverge':>8} | {'Teams':>6} | "
              f"{'Total Cost':>12}")
    print(header)
    print("-" * len(header))
    for cond in conditions:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        n_solved = sum(1 for r in trials if r['solved'])
        summit_fracs = []
        trap_fracs = []
        divergences = []
        team_counts = []
        total_costs = []
        for r in trials:
            if r['log']:
                summit_fracs.append(r['log'][-1].get('summit_frac', 0))
                trap_fracs.append(r['log'][-1].get('local_trap_frac', 0))
                divergences.append(
                    r['log'][-1].get('pref_strat_divergence', 0))
                team_counts.append(r['log'][-1].get('n_clusters', 0))
            total_costs.append(r['final_cost']['total'])
        mean_summit = np.mean(summit_fracs) if summit_fracs else 0
        mean_trap = np.mean(trap_fracs) if trap_fracs else 0
        mean_div = np.mean(divergences) if divergences else 0
        mean_teams = np.mean(team_counts) if team_counts else 0
        mean_cost = np.mean(total_costs) if total_costs else 0
        print(f"{label:<42} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_summit:>7.1%} | {mean_trap:>7.1%} | "
              f"{mean_div:>8.3f} | {mean_teams:>6.1f} | "
              f"{mean_cost:>12.0f}")

    # ── Pareto analysis ──
    print("\n=== Cost-Efficiency Frontier ===")
    cond_stats = []
    for cond in conditions:
        label = cond['label']
        trials = [r for r in all_results if r['condition'] == label]
        summit_fracs = [r['log'][-1].get('summit_frac', 0)
                        for r in trials if r['log']]
        costs = [r['final_cost']['total'] for r in trials]
        mean_s = np.mean(summit_fracs) if summit_fracs else 0
        mean_c = np.mean(costs) if costs else 0
        efficiency = mean_s / (mean_c + 1) * 1e6
        cond_stats.append((label, mean_s, mean_c, efficiency))

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
