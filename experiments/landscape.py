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


# ── Pre-built landscapes ──────────────────────────────────────────────

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

