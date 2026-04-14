"""
Fitness Landscape
=================

Defines the *hidden* fitness surface over the preference space.
This is the ceiling of the knowledge manifold — the maximum possible
knowledge at each point in skill space.

The landscape is a sum of Gaussian peaks plus spectral noise,
creating a rugged terrain with multiple local optima, ridges,
and narrow viable paths.
"""

import numpy as np


class RuggedLandscape:
    """Multi-peak fitness landscape with spectral noise.

    Parameters
    ----------
    peaks : list of dict
        Each dict has 'center' (K,), 'height' float, 'sigma' float.
    noise_frequencies : list of (freq, amplitude) tuples
        Spectral noise components added to the base peaks.
    seed : int
        Random seed for reproducible noise phases.
    """

    def __init__(self, peaks, noise_frequencies=None, seed=42):
        self.peaks = peaks
        self.noise_frequencies = noise_frequencies or []
        self._rng = np.random.default_rng(seed)
        # Pre-generate random phases for spectral noise
        n_freq = len(self.noise_frequencies)
        self._phases_x = self._rng.uniform(0, 2 * np.pi, n_freq)
        self._phases_y = self._rng.uniform(0, 2 * np.pi, n_freq)
        self._freq_dirs = self._rng.uniform(0, 2 * np.pi, n_freq)

    def fitness(self, prefs):
        """Evaluate fitness at preference positions.

        Parameters
        ----------
        prefs : ndarray (N, K)
            Preference vectors.  Only dims 0 and 1 are used for
            the 2D landscape.

        Returns
        -------
        f : ndarray (N,)
        """
        x = prefs[:, 0]
        y = prefs[:, 1]
        f = np.zeros(len(prefs), dtype=np.float64)

        # Sum of Gaussian peaks
        for peak in self.peaks:
            cx, cy = peak['center'][0], peak['center'][1]
            h = peak['height']
            s = peak['sigma']
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            f += h * np.exp(-dist_sq / (2.0 * s ** 2))

        # Spectral noise
        for i, (freq, amp) in enumerate(self.noise_frequencies):
            dx = np.cos(self._freq_dirs[i])
            dy = np.sin(self._freq_dirs[i])
            proj = x * dx + y * dy
            f += amp * np.sin(2 * np.pi * freq * proj + self._phases_x[i])

        return f

    def gradient(self, prefs, eps=1e-4):
        """Numerical gradient via central differences.

        Returns
        -------
        grad : ndarray (N, 2)
        """
        N = len(prefs)
        grad = np.zeros((N, 2), dtype=np.float64)

        for dim in range(2):
            p_plus = prefs.copy()
            p_minus = prefs.copy()
            p_plus[:, dim] += eps
            p_minus[:, dim] -= eps
            grad[:, dim] = (self.fitness(p_plus) - self.fitness(p_minus)) / (2 * eps)

        return grad


def make_default_landscape(seed=42):
    """Create a rugged landscape with multiple peaks and spectral noise.

    The landscape has:
    - A global summit (hard to reach)
    - Several foothills (easy local optima / traps)
    - High-frequency noise creating ridges and narrow paths
    """
    peaks = [
        # Global summit — high but narrow
        {'center': [0.6, 0.7], 'height': 1.0, 'sigma': 0.25},
        # Attractive foothill — wide, easy to find
        {'center': [-0.4, -0.3], 'height': 0.6, 'sigma': 0.4},
        # Secondary foothill
        {'center': [0.3, -0.5], 'height': 0.5, 'sigma': 0.3},
        # Small ridge
        {'center': [-0.7, 0.5], 'height': 0.35, 'sigma': 0.2},
    ]

    # Spectral noise: mix of frequencies for ruggedness
    noise_frequencies = [
        (1.5, 0.08),   # Low frequency — broad undulations
        (2.5, 0.06),
        (4.0, 0.05),   # Medium frequency — ridges
        (6.0, 0.04),
        (8.0, 0.03),   # High frequency — fine texture
        (10.0, 0.02),
        (12.0, 0.015),
    ]

    return RuggedLandscape(peaks, noise_frequencies, seed=seed)
