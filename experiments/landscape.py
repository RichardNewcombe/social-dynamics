"""
Fitness landscapes for the mountain-climbing experiment.

A landscape maps K-dimensional preference vectors to scalar fitness values
and provides gradient information.  Landscapes are defined as collections
of Gaussian peaks with different heights, widths, and centres — creating
a rugged terrain with local optima that can trap conformist groups.
"""

import numpy as np


class GaussianPeakLandscape:
    """Multi-peak fitness landscape in [-1, 1]^K preference space.

    Each peak is a Gaussian bump:
        f_i(x) = height_i * exp(-||x - c_i||^2 / (2 * sigma_i^2))

    The landscape fitness at x is the maximum over all peaks (not sum),
    which creates distinct basins of attraction with saddle ridges.

    Parameters
    ----------
    peaks : list of dict
        Each dict has keys 'center' (array-like, K), 'height' (float),
        'sigma' (float).
    k : int
        Number of preference dimensions.
    """

    def __init__(self, peaks, k):
        self.k = k
        self.n_peaks = len(peaks)
        self.centers = np.array([p['center'][:k] for p in peaks],
                                dtype=np.float64)
        self.heights = np.array([p['height'] for p in peaks],
                                dtype=np.float64)
        self.sigmas = np.array([p['sigma'] for p in peaks],
                               dtype=np.float64)

    def fitness(self, prefs):
        """Compute fitness for each particle.

        Parameters
        ----------
        prefs : (N, K) array

        Returns
        -------
        fitness : (N,) array in [0, max_height]
        peak_ids : (N,) int array — which peak each particle is closest to
        """
        p = prefs.astype(np.float64)
        # (N, n_peaks) distance matrix
        diff = p[:, None, :] - self.centers[None, :, :]  # (N, P, K)
        dist_sq = (diff ** 2).sum(axis=2)  # (N, P)
        # Per-peak fitness
        peak_fitness = self.heights[None, :] * np.exp(
            -dist_sq / (2.0 * self.sigmas[None, :] ** 2))  # (N, P)
        peak_ids = np.argmax(peak_fitness, axis=1)  # (N,)
        fitness = peak_fitness[np.arange(len(p)), peak_ids]  # (N,)
        return fitness, peak_ids

    def gradient(self, prefs):
        """Compute the fitness gradient for each particle.

        The gradient points uphill on the nearest (highest-fitness) peak.
        This is the "true mountain signal" that researchers sense with noise.

        Parameters
        ----------
        prefs : (N, K) array

        Returns
        -------
        grad : (N, K) array — unit-normalized gradient direction
        fitness : (N,) array — current fitness values
        peak_ids : (N,) int array
        """
        p = prefs.astype(np.float64)
        fitness, peak_ids = self.fitness(p)

        # Gradient of the dominant peak for each particle
        centers_at = self.centers[peak_ids]  # (N, K)
        sigmas_at = self.sigmas[peak_ids]  # (N,)
        # d/dx [h * exp(-||x-c||^2/(2s^2))] = h * exp(...) * (c - x) / s^2
        # = fitness * (c - x) / s^2
        diff = centers_at - p  # (N, K)
        grad = fitness[:, None] * diff / (sigmas_at[:, None] ** 2)

        # Normalize to unit vectors (direction only; magnitude handled externally)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        grad_unit = grad / norms

        return grad_unit, fitness, peak_ids


class CostLandscape:
    """Terrain cost layer in [-1, 1]^K preference space.

    The cost terrain is independent of the fitness landscape.  It models
    the idea that some paths through strategy space are more expensive
    than others (regulatory hurdles, infrastructure investment, etc.).

    Implemented as a sum of Gaussian bumps (cost ridges) plus a base cost.
    High-cost regions are expensive to traverse; low-cost regions are cheap.

    Parameters
    ----------
    ridges : list of dict
        Each dict has keys 'center' (array-like, K), 'height' (float),
        'sigma' (float).  Height is the peak cost at the ridge centre.
    k : int
        Number of preference dimensions.
    base_cost : float
        Minimum per-step terrain cost everywhere.  Default 0.1.
    """

    def __init__(self, ridges, k, base_cost=0.1):
        self.k = k
        self.n_ridges = len(ridges)
        self.base_cost = base_cost
        self.centers = np.array([r['center'][:k] for r in ridges],
                                dtype=np.float64)
        self.heights = np.array([r['height'] for r in ridges],
                                dtype=np.float64)
        self.sigmas = np.array([r['sigma'] for r in ridges],
                               dtype=np.float64)

    def cost(self, prefs):
        """Compute terrain cost at each particle's position.

        Parameters
        ----------
        prefs : (N, K) array

        Returns
        -------
        cost : (N,) array — terrain cost per particle (>= base_cost)
        """
        p = prefs.astype(np.float64)
        # (N, n_ridges) distance matrix
        diff = p[:, None, :] - self.centers[None, :, :]  # (N, R, K)
        dist_sq = (diff ** 2).sum(axis=2)  # (N, R)
        # Sum of all ridge contributions (not max — ridges stack)
        ridge_costs = self.heights[None, :] * np.exp(
            -dist_sq / (2.0 * self.sigmas[None, :] ** 2))  # (N, R)
        total = self.base_cost + ridge_costs.sum(axis=1)  # (N,)
        return total


def compute_employee_cost(sim, base=1.0, w_engineer=0.5, w_leader=0.5,
                          w_researcher=1.0, w_visionary=2.0):
    """Compute per-step employee cost for each particle.

    Cost increases with capability:
      cost_i = base + w_e * step_scale_i + w_l * influence_i
                    + w_r * (1 / noise_i) + w_v * visionary_i

    Researcher cost is *inverse* noise: a low-noise (accurate) researcher
    is more expensive.  Visionary cost is direct and high-weighted to
    create the budget pressure against stacking visionaries.

    Parameters
    ----------
    sim : Simulation
    base, w_engineer, w_leader, w_researcher, w_visionary : float
        Cost weights.

    Returns
    -------
    cost : (N,) array — per-particle employee cost per step
    """
    cost = np.full(sim.n, base, dtype=np.float64)
    cost += w_engineer * sim.role_step_scale
    cost += w_leader * sim.role_influence
    # Researcher: lower noise = better = more expensive
    noise = np.maximum(sim.role_gradient_noise, 0.01)  # avoid div-by-zero
    cost += w_researcher * (1.0 / noise)
    cost += w_visionary * sim.role_visionary
    return cost


# ── Pre-built landscapes ────────────────────────────────────────────────────────

def make_default_landscape(k=3):
    """Create a landscape with 1 global peak + 2 local peaks.

    The global peak is far from the origin (requires exploration to find).
    Local peaks are closer to the origin (easy to find, tempting to stay).
    """
    peaks = [
        # Global summit — far corner, tall, moderately wide
        {'center': [0.75] * k, 'height': 1.0, 'sigma': 0.30},
        # Local peak 1 — near origin, shorter, narrow
        {'center': [-0.25] + [0.15] * (k - 1), 'height': 0.55, 'sigma': 0.22},
        # Local peak 2 — opposite quadrant, medium height
        {'center': [0.1] + [-0.4] * (k - 1), 'height': 0.45, 'sigma': 0.20},
    ]
    return GaussianPeakLandscape(peaks, k)


def make_rugged_landscape(k=3, n_peaks=6, seed=42):
    """Create a more rugged landscape with many peaks.

    One peak is designated the global optimum (height=1.0).
    Others have random heights in [0.3, 0.7].
    """
    rng = np.random.default_rng(seed)
    peaks = []
    # Global peak
    peaks.append({
        'center': rng.uniform(0.4, 0.8, k).tolist(),
        'height': 1.0,
        'sigma': 0.25 + rng.uniform(0, 0.1),
    })
    # Local peaks
    for _ in range(n_peaks - 1):
        peaks.append({
            'center': rng.uniform(-0.8, 0.8, k).tolist(),
            'height': 0.3 + rng.uniform(0, 0.4),
            'sigma': 0.15 + rng.uniform(0, 0.15),
        })
    return GaussianPeakLandscape(peaks, k)


def make_default_cost_landscape(k=3):
    """Create a cost terrain that makes the direct path to the summit expensive.

    The cost ridge sits between the origin (where particles start) and the
    global summit at [0.75]^K.  Taking the straight-line path means crossing
    a high-cost zone.  Routing around it is cheaper but slower.

    A second, smaller cost bump sits near the global summit itself — even
    arriving at the right answer has a cost (implementation, transition).
    """
    ridges = [
        # Main cost ridge — blocks the direct path to [0.75]^K
        {'center': [0.35] * k, 'height': 2.0, 'sigma': 0.25},
        # Implementation cost near the summit
        {'center': [0.65] * k, 'height': 0.8, 'sigma': 0.15},
        # Cheap corridor exists around [-0.3, 0.8, ...] — rewards exploration
    ]
    return CostLandscape(ridges, k, base_cost=0.1)
