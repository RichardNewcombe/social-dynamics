"""
Fitness overlay functions for problem-solving experiments.

Each fitness function maps a K-dimensional preference vector to a scalar
fitness value.  These are layered on top of the base simulation to define
measurable problem-solving objectives.
"""

import numpy as np


# ── Experiment 1: Hidden Target ─────────────────────────────────────────

def hidden_target_fitness(prefs, target):
    """Fitness = 1 - normalised Euclidean distance to target.

    Args:
        prefs:  (N, K) array of particle preference vectors in [-1, 1].
        target: (K,) array — the hidden optimal preference vector.

    Returns:
        fitness: (N,) array in [0, 1].  1.0 = at the target.
    """
    diff = prefs.astype(np.float64) - target.astype(np.float64)
    dist = np.linalg.norm(diff, axis=1)
    max_dist = np.sqrt(prefs.shape[1]) * 2.0  # max possible in [-1,1]^K
    return 1.0 - dist / max_dist


def hidden_target_gradient(prefs, target, strength=0.001):
    """Compute a small preference nudge toward the hidden target.

    This simulates a weak environmental signal: particles don't know the
    target, but the landscape provides a slight bias toward higher fitness.

    Args:
        prefs:    (N, K) preference array.
        target:   (K,) target vector.
        strength: magnitude of the nudge per step.

    Returns:
        nudge: (N, K) array to be added to prefs after each step.
    """
    diff = target.astype(np.float64) - prefs.astype(np.float64)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return strength * (diff / norms)


# ── Experiment 2: Multi-Niche Coverage ──────────────────────────────────

def niche_occupancy(prefs, niches, radius):
    """Count how many particles fall within each niche.

    Args:
        prefs:  (N, K) particle preference vectors.
        niches: (M, K) array of niche centre vectors.
        radius: scalar — max Euclidean distance to count as "in niche".

    Returns:
        counts:   (M,) int array — number of particles in each niche.
        occupied: (M,) bool array — True if niche has >= 1 particle.
    """
    prefs_f64 = prefs.astype(np.float64)
    niches_f64 = niches.astype(np.float64)
    # (N, M) distance matrix
    dists = np.linalg.norm(
        prefs_f64[:, None, :] - niches_f64[None, :, :], axis=2
    )
    in_niche = dists <= radius  # (N, M)
    counts = in_niche.sum(axis=0)
    occupied = counts >= 1
    return counts, occupied


def generate_niches(k, n_niches=4, seed=123):
    """Generate well-separated niche centres in [-1, 1]^K.

    Uses a simple strategy: place niches at the vertices of a simplex-like
    arrangement in the first few dimensions, with remaining dims at 0.

    Args:
        k:        number of preference dimensions.
        n_niches: number of niches to generate.
        seed:     random seed for reproducibility.

    Returns:
        niches: (n_niches, K) array of niche centres.
    """
    rng = np.random.default_rng(seed)
    # Generate random candidate niches and keep the most spread-out set
    best_niches = None
    best_min_dist = -1.0
    for _ in range(1000):
        candidates = rng.uniform(-0.8, 0.8, (n_niches, k))
        # Compute minimum pairwise distance
        dists = []
        for i in range(n_niches):
            for j in range(i + 1, n_niches):
                dists.append(np.linalg.norm(candidates[i] - candidates[j]))
        min_dist = min(dists) if dists else 0.0
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_niches = candidates.copy()
    return best_niches


# ── Experiment 3: Ghost Colony Escape ───────────────────────────────────

def zone_fitness(prefs, zone_centre, zone_radius):
    """Fraction of particles within a preference-space zone.

    Args:
        prefs:       (N, K) preference vectors.
        zone_centre: (K,) centre of the zone.
        zone_radius: scalar — Euclidean radius of the zone.

    Returns:
        fraction: float in [0, 1] — fraction of particles inside the zone.
        count:    int — number of particles inside.
    """
    diff = prefs.astype(np.float64) - zone_centre.astype(np.float64)
    dists = np.linalg.norm(diff, axis=1)
    inside = dists <= zone_radius
    count = int(inside.sum())
    fraction = count / len(prefs)
    return fraction, count


def zone_gradient(prefs, zone_centre, strength=0.001):
    """Nudge toward a zone centre (attraction) or away (repulsion if negative)."""
    diff = zone_centre.astype(np.float64) - prefs.astype(np.float64)
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return strength * (diff / norms)
