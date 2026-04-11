#!/usr/bin/env python3
"""
Experiment 3 — Ghost Colony Escape (Institutional Inertia)
==========================================================
Tests how spatial memory (institutional knowledge) affects an
organisation's ability to adapt after an environmental shock.

Hypothesis
----------
- Strong spatial memory with slow decay creates "ghost colonies" that
  resist change, dramatically increasing adaptation time.
- Weak or no memory allows the swarm to shift quickly.

Design
------
The simulation's memory field works by modulating preferences
multiplicatively *during* the physics step, affecting which neighbors
attract/repel.  After the step, original prefs are restored plus the
social delta.  The field also deposits current prefs each step.

The key mechanism: during settling, particles with negative prefs deposit
negative values into the field.  When we flip prefs to positive in Phase 2,
the field still contains negative values.  The multiplicative modulation
``prefs * (1 + strength * field)`` will *dampen* the new positive prefs
(since field is negative), reducing the effective signal strength and
slowing convergence.

Phase 1 (Settling):  Particles start near Zone A (negative prefs).
    Memory field accumulates negative values.

Phase 2 (Shock):  All particle prefs are instantly flipped to Zone B
    (positive).  We measure how many steps until the *effective* prefs
    (after memory modulation) stabilize near Zone B, meaning the field
    has been overwritten.

The sweep variable is ``memory_strength``.

All other simulation parameters inherit from the user's current
configuration (GUI or params.py), falling back to SAFE_DEFAULTS for
headless CLI runs.

Usage
-----
    python -m experiments.exp3_ghost_colony [--trials 5] [--max-steps 15000]
"""

import argparse
import json
import os
import sys
import time as _time
import numpy as np

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from sim_2d_exp.params import params as _global_params, SPACE
from sim_2d_exp.simulation import Simulation
from experiments.runner import SAFE_DEFAULTS, apply_post_processing

# ── Experiment constants ────────────────────────────────────────────────

K = 3
NUM_PARTICLES = 400
ZONE_A_PREF = np.array([-0.7, -0.7, -0.7])
ZONE_B_PREF = np.array([0.7, 0.7, 0.7])
SETTLE_STEPS = 2500
ADAPTATION_THRESHOLD = 0.60  # fraction of particles whose modulated prefs are near Zone B

# Memory strength values to sweep
SWEEP_VALUES = [0.0, 1.0, 3.0, 6.0, 10.0]


def init_prefs_near(sim, target, noise_std=0.08):
    """Initialise particle preferences near a target vector.

    Uses the simulation's own RNG so that the experiment seed
    (params['seed']) fully controls reproducibility.
    """
    noise = sim.rng.normal(0, noise_std, (sim.n, sim.k))
    sim.prefs = np.clip(target + noise, -1, 1).astype(sim.prefs.dtype)
    sim.response = sim.prefs.copy()
    apply_post_processing(sim)


def get_modulated_prefs(sim):
    """Compute what the memory field would do to current prefs (read-only).

    Replicates the multiplicative gate from Simulation.step() so we can
    measure the *effective* preferences the physics engine would see.

    NOTE: This duplicates internal logic.  If the formula in
    simulation.py changes, this must be updated to match.
    """
    if not _global_params['memory_field']:
        return sim.prefs.copy()
    strength = _global_params['memory_strength']
    G = sim.memory_field.shape[0]
    inv_cell = G / SPACE
    cx = (sim.pos[:, 0] * inv_cell).astype(int) % G
    cy = (sim.pos[:, 1] * inv_cell).astype(int) % G
    field_at_particle = sim.memory_field[cy, cx]
    modulated = (sim.prefs.astype(np.float64)
                 * (1.0 + strength * field_at_particle))
    return np.clip(modulated, -1, 1)


def frac_near_zone(prefs, zone, radius=0.6):
    """Fraction of particles within Euclidean radius of zone in pref-space."""
    diff = prefs.astype(np.float64) - zone.astype(np.float64)
    dists = np.linalg.norm(diff, axis=1)
    return float((dists <= radius).sum() / len(prefs))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Ghost Colony Escape")
    parser.add_argument('--trials', type=int, default=5,
                        help='Monte Carlo trials per sweep value')
    parser.add_argument('--max-steps', type=int, default=15_000,
                        help='Max simulation steps per trial (incl. settling)')
    parser.add_argument('--output', type=str,
                        default='results/exp3_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print(f"=== Experiment 3: Ghost Colony Escape ===")
    print(f"Zone A prefs: {ZONE_A_PREF}")
    print(f"Zone B prefs: {ZONE_B_PREF}")
    print(f"Settling steps: {SETTLE_STEPS}")
    print(f"Sweep: memory_strength = {SWEEP_VALUES}")
    print(f"Trials per value: {args.trials}")
    print()

    all_results = []

    for mem_strength in SWEEP_VALUES:
        print(f"\n--- memory_strength = {mem_strength} ---")

        for trial in range(args.trials):
            use_memory = mem_strength > 0

            # ── Snapshot the ENTIRE global params dict ──
            full_snapshot = dict(_global_params)

            try:
                # ── Three-layer resolution ──
                # Layer 1: SAFE_DEFAULTS
                # Layer 2: ambient params (user's current state)
                # Layer 3: experiment overrides (only what we must control)
                resolved = {}
                resolved.update(SAFE_DEFAULTS)
                resolved.update(full_snapshot)
                resolved.update({
                    'num_particles': NUM_PARTICLES,
                    'k': K,
                    'physics_engine': 1,
                    'use_seed': True,
                    'seed': trial * 1000 + 9999,
                    'social': 0.008,
                    'social_mode': 0,
                    # Memory field — the swept variable
                    'memory_field': use_memory,
                    'memory_write_rate': 0.1,
                    'memory_strength': mem_strength,
                    'memory_decay': 0.9995,
                    'memory_blur': True,
                    'memory_blur_sigma': 1.0,
                })

                _global_params.clear()
                _global_params.update(resolved)

                sim = Simulation()
                init_prefs_near(sim, ZONE_A_PREF)

                log = []
                solved = False
                solve_step = None
                t0 = _time.perf_counter()

                for step in range(1, args.max_steps + 1):
                    sim.step()

                    if step == SETTLE_STEPS:
                        # ── SHOCK: instantly flip all prefs to Zone B ──
                        init_prefs_near(sim, ZONE_B_PREF, noise_std=0.08)

                    if step > SETTLE_STEPS:
                        # Check: are the *modulated* prefs near Zone B?
                        mod_prefs = get_modulated_prefs(sim)
                        frac_b = frac_near_zone(mod_prefs, ZONE_B_PREF, radius=0.6)
                        frac_a = frac_near_zone(mod_prefs, ZONE_A_PREF, radius=0.6)

                        if frac_b >= ADAPTATION_THRESHOLD and not solved:
                            solved = True
                            solve_step = step
                            mem_energy = float(np.abs(sim.memory_field).sum()) if use_memory else 0.0
                            log.append({
                                'step': step,
                                'phase': 'adapting',
                                'frac_mod_zone_a': frac_a,
                                'frac_mod_zone_b': frac_b,
                                'frac_raw_zone_b': frac_near_zone(sim.prefs, ZONE_B_PREF, 0.6),
                                'memory_energy': mem_energy,
                            })
                            break

                    # Periodic logging
                    if step % 200 == 0:
                        mod_prefs = get_modulated_prefs(sim)
                        frac_a = frac_near_zone(mod_prefs, ZONE_A_PREF, 0.6)
                        frac_b = frac_near_zone(mod_prefs, ZONE_B_PREF, 0.6)
                        phase = 'settling' if step <= SETTLE_STEPS else 'adapting'
                        mem_energy = float(np.abs(sim.memory_field).sum()) if use_memory else 0.0
                        log.append({
                            'step': step,
                            'phase': phase,
                            'frac_mod_zone_a': frac_a,
                            'frac_mod_zone_b': frac_b,
                            'frac_raw_zone_b': frac_near_zone(sim.prefs, ZONE_B_PREF, 0.6),
                            'memory_energy': mem_energy,
                        })

                wall_time = _time.perf_counter() - t0

            finally:
                # ── Restore the FULL original params ──
                _global_params.clear()
                _global_params.update(full_snapshot)

            adapt_steps = (solve_step - SETTLE_STEPS) if solved and solve_step else None

            result = {
                'solved': solved,
                'solve_step': solve_step,
                'adapt_steps': adapt_steps,
                'total_steps': step,
                'wall_time': wall_time,
                'memory_strength': mem_strength,
                'trial': trial,
                'log': log,
            }
            all_results.append(result)

            status = "SOLVED" if solved else "FAILED"
            adapt_str = str(adapt_steps) if adapt_steps else 'N/A'
            print(f"  trial={trial}: {status}, "
                  f"adapt_steps={adapt_str} ({wall_time:.1f}s)")

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"{'mem_strength':>14} | {'Solved':>8} | {'Mean Adapt Steps':>18} | "
          f"{'Mean Time':>10}")
    print("-" * 65)
    for val in SWEEP_VALUES:
        trials = [r for r in all_results if r['memory_strength'] == val]
        n_solved = sum(1 for r in trials if r['solved'])
        adapt = [r['adapt_steps'] for r in trials
                 if r['adapt_steps'] is not None]
        mean_adapt = np.mean(adapt) if adapt else float('inf')
        mean_time = np.mean([r['wall_time'] for r in trials])
        print(f"{val:>14.1f} | {n_solved:>5}/{len(trials):<2} | "
              f"{mean_adapt:>18.0f} | {mean_time:>9.1f}s")

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
