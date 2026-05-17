#!/usr/bin/env python3
"""
Verify that HierarchicalSimulation with l1_weight=0 produces
identical output to the standard sim_2d_exp physics.

Runs both sims with the same seed and params for N steps,
compares final positions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_hierarchical.hierarchical import HierarchicalSimulation
from sim_2d_exp.params import params, SPACE
from scipy.spatial import cKDTree


def run_reference_physics(num_particles, k, n_neighbors, step_size, seed, steps):
    """Replicate the standard per-dimension best-neighbor physics exactly."""
    rng = np.random.default_rng(seed)
    L = SPACE
    N = num_particles

    pos = rng.uniform(0, L, (N, 2)).astype(np.float64)
    prefs = rng.uniform(-1, 1, (N, k)).astype(np.float64)

    for step in range(steps):
        wrapped = pos % L
        tree = cKDTree(wrapped, boxsize=L)
        _, nbr_ids = tree.query(wrapped, k=n_neighbors + 1)
        nbr_ids = np.clip(nbr_ids[:, 1:], 0, N - 1)

        movement = np.zeros((N, 2), dtype=np.float64)

        for ki in range(k):
            nbr_prefs_k = prefs[nbr_ids, ki]
            best_local = np.argmax(nbr_prefs_k, axis=1)
            best_global = nbr_ids[np.arange(N), best_local]

            delta = pos[best_global] - pos
            delta -= L * np.round(delta / L)
            dist = np.sqrt((delta ** 2).sum(axis=1, keepdims=True))
            direction = delta / np.maximum(dist, 1e-12)

            compat = prefs[:, ki] * prefs[best_global, ki]
            movement += compat[:, None] * direction

        pos = (pos + step_size * movement) % L

    return pos, prefs


def main():
    N = 200
    K = 3
    SEED = 42
    STEPS = 100
    N_NEIGHBORS = 21
    STEP_SIZE = 0.005

    print(f"Comparing: N={N}, K={K}, seed={SEED}, steps={STEPS}")
    print(f"  n_neighbors={N_NEIGHBORS}, step_size={STEP_SIZE}")
    print()

    # Run reference
    print("Running reference physics...")
    ref_pos, ref_prefs = run_reference_physics(N, K, N_NEIGHBORS, STEP_SIZE, SEED, STEPS)

    # Run hierarchical with l1_weight=0
    print("Running hierarchical (l1_weight=0)...")
    sim = HierarchicalSimulation(
        num_particles=N, k=K, seed=SEED,
        step_size=STEP_SIZE, n_neighbors=N_NEIGHBORS,
        l1_weight=0.0,
    )
    for step in range(STEPS):
        sim.step()

    hier_pos = sim.pos

    # Compare
    pos_diff = np.abs(ref_pos - hier_pos)
    # Handle periodic wrapping differences
    pos_diff = np.minimum(pos_diff, SPACE - pos_diff)
    max_diff = pos_diff.max()
    mean_diff = pos_diff.mean()
    rms_diff = np.sqrt((pos_diff ** 2).mean())

    print(f"\nResults after {STEPS} steps:")
    print(f"  Max position difference:  {max_diff:.2e}")
    print(f"  Mean position difference: {mean_diff:.2e}")
    print(f"  RMS position difference:  {rms_diff:.2e}")

    # Check prefs are unchanged (no social learning)
    pref_diff = np.abs(ref_prefs - sim.prefs).max()
    print(f"  Max pref difference:      {pref_diff:.2e}")

    if max_diff < 1e-10:
        print("\n  ✅ IDENTICAL — hierarchical with l1_weight=0 matches reference exactly")
    elif max_diff < 1e-6:
        print("\n  ⚠️  CLOSE — tiny floating point differences only")
    else:
        print("\n  ❌ DIVERGED — physics differ!")

        # Find worst particle
        worst = np.argmax(pos_diff.sum(axis=1))
        print(f"\n  Worst particle #{worst}:")
        print(f"    ref pos:  {ref_pos[worst]}")
        print(f"    hier pos: {hier_pos[worst]}")
        print(f"    diff:     {pos_diff[worst]}")


if __name__ == '__main__':
    main()
