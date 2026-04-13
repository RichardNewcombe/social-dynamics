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


class RuggedLandscape:
    """Multi-scale rugged fitness landscape in [-1, 1]^K.

    Combines three layers to produce a realistic, challenging terrain:

    1. **Major peaks** — a few large Gaussian bumps defining the global
       summit and prominent local optima.  These set the macro structure.
    2. **Minor peaks** — many smaller Gaussian bumps creating foothills,
       false summits, and ridgelines.  These make local gradient following
       unreliable.
    3. **Spectral noise** — sum-of-sinusoids at multiple frequencies
       adding continuous roughness (ridges, saddle points, subtle valleys).
       This is a deterministic, differentiable alternative to Perlin noise.

    The final fitness is:
        f(x) = w_major * max_peak(major) + w_minor * max_peak(minor)
             + w_noise * noise(x)
    clipped to [0, 1].

    Parameters
    ----------
    major_peaks : list of dict
        Large-scale peaks (center, height, sigma).
    minor_peaks : list of dict
        Small-scale peaks (center, height, sigma).
    noise_freqs : (F, K) array
        Frequency vectors for spectral noise.
    noise_amps : (F,) array
        Amplitude of each frequency component.
    noise_phases : (F,) array
        Phase offset of each frequency component.
    k : int
        Dimensionality.
    w_major, w_minor, w_noise : float
        Blending weights for the three layers.
    """

    def __init__(self, major_peaks, minor_peaks,
                 noise_freqs, noise_amps, noise_phases,
                 k, w_major=0.6, w_minor=0.25, w_noise=0.15):
        self.k = k
        self.w_major = w_major
        self.w_minor = w_minor
        self.w_noise = w_noise

        # Major peaks
        self.n_major = len(major_peaks)
        self.major_centers = np.array([p['center'][:k] for p in major_peaks],
                                      dtype=np.float64)
        self.major_heights = np.array([p['height'] for p in major_peaks],
                                      dtype=np.float64)
        self.major_sigmas = np.array([p['sigma'] for p in major_peaks],
                                     dtype=np.float64)

        # Minor peaks
        self.n_minor = len(minor_peaks)
        self.minor_centers = np.array([p['center'][:k] for p in minor_peaks],
                                      dtype=np.float64)
        self.minor_heights = np.array([p['height'] for p in minor_peaks],
                                      dtype=np.float64)
        self.minor_sigmas = np.array([p['sigma'] for p in minor_peaks],
                                     dtype=np.float64)

        # Combined for interface compatibility
        # centers[0] is always the global summit
        self.centers = np.vstack([self.major_centers, self.minor_centers])
        self.n_peaks = self.n_major + self.n_minor

        # Spectral noise
        self.noise_freqs = np.array(noise_freqs, dtype=np.float64)  # (F, K)
        self.noise_amps = np.array(noise_amps, dtype=np.float64)    # (F,)
        self.noise_phases = np.array(noise_phases, dtype=np.float64)  # (F,)

    def _eval_peaks(self, prefs, centers, heights, sigmas):
        """Evaluate Gaussian peaks, return (N,) max fitness and (N,) peak ids."""
        p = prefs.astype(np.float64)
        diff = p[:, None, :] - centers[None, :, :]  # (N, P, K)
        dist_sq = (diff ** 2).sum(axis=2)  # (N, P)
        peak_fitness = heights[None, :] * np.exp(
            -dist_sq / (2.0 * sigmas[None, :] ** 2))  # (N, P)
        peak_ids = np.argmax(peak_fitness, axis=1)
        fitness = peak_fitness[np.arange(len(p)), peak_ids]
        return fitness, peak_ids

    def _eval_noise(self, prefs):
        """Evaluate spectral noise layer.

        noise(x) = sum_f amp_f * sin(2*pi * freq_f . x + phase_f)
        Normalized to [0, 1] range.
        """
        p = prefs.astype(np.float64)
        # (N, K) @ (K, F) -> (N, F)
        dot = p @ self.noise_freqs.T  # (N, F)
        waves = self.noise_amps[None, :] * np.sin(
            2.0 * np.pi * dot + self.noise_phases[None, :])  # (N, F)
        raw = waves.sum(axis=1)  # (N,)
        # Normalize: theoretical range is [-sum(amps), +sum(amps)]
        amp_sum = self.noise_amps.sum()
        if amp_sum > 0:
            normalized = (raw + amp_sum) / (2.0 * amp_sum)  # [0, 1]
        else:
            normalized = np.zeros(len(p))
        return normalized

    def _eval_noise_gradient(self, prefs):
        """Gradient of the spectral noise layer.

        d/dx noise = sum_f amp_f * cos(2*pi * freq_f . x + phase_f) * 2*pi * freq_f
        """
        p = prefs.astype(np.float64)
        dot = p @ self.noise_freqs.T  # (N, F)
        cos_waves = self.noise_amps[None, :] * np.cos(
            2.0 * np.pi * dot + self.noise_phases[None, :])  # (N, F)
        # (N, F) @ (F, K) -> (N, K), scaled by 2*pi
        grad = (2.0 * np.pi) * (cos_waves @ self.noise_freqs)  # (N, K)
        # Normalize same as noise
        amp_sum = self.noise_amps.sum()
        if amp_sum > 0:
            grad = grad / (2.0 * amp_sum)
        return grad

    def fitness(self, prefs):
        """Compute composite fitness.

        Returns
        -------
        fitness : (N,) array in [0, ~1]
        peak_ids : (N,) int array — which major peak is dominant
        """
        major_f, major_ids = self._eval_peaks(
            prefs, self.major_centers, self.major_heights, self.major_sigmas)
        minor_f, _ = self._eval_peaks(
            prefs, self.minor_centers, self.minor_heights, self.minor_sigmas)
        noise_f = self._eval_noise(prefs)

        composite = (self.w_major * major_f
                     + self.w_minor * minor_f
                     + self.w_noise * noise_f)
        composite = np.clip(composite, 0, 1)
        return composite, major_ids

    def gradient(self, prefs):
        """Compute composite fitness gradient.

        Returns
        -------
        grad : (N, K) unit-normalized gradient direction
        fitness : (N,) current fitness
        peak_ids : (N,) dominant major peak
        """
        p = prefs.astype(np.float64)
        fitness, peak_ids = self.fitness(p)

        # Major peak gradient (gradient of dominant peak)
        major_f, major_ids = self._eval_peaks(
            p, self.major_centers, self.major_heights, self.major_sigmas)
        mc = self.major_centers[major_ids]
        ms = self.major_sigmas[major_ids]
        major_grad = major_f[:, None] * (mc - p) / (ms[:, None] ** 2)

        # Minor peak gradient (gradient of dominant minor peak)
        minor_f, minor_ids = self._eval_peaks(
            p, self.minor_centers, self.minor_heights, self.minor_sigmas)
        nc = self.minor_centers[minor_ids]
        ns = self.minor_sigmas[minor_ids]
        minor_grad = minor_f[:, None] * (nc - p) / (ns[:, None] ** 2)

        # Noise gradient
        noise_grad = self._eval_noise_gradient(p)

        # Composite gradient
        grad = (self.w_major * major_grad
                + self.w_minor * minor_grad
                + self.w_noise * noise_grad)

        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        grad_unit = grad / norms

        return grad_unit, fitness, peak_ids


# ── Pre-built landscapes ────────────────────────────────────────────────────────

def make_default_landscape(k=3):
    """Create a rugged multi-scale landscape.

    The landscape has:
    - 4 major peaks (1 global summit + 3 prominent local optima)
    - 12 minor peaks (foothills, false summits, ridgeline bumps)
    - 20 spectral noise frequencies (continuous roughness)

    The global summit is in a corner that requires navigating past
    several deceptive local peaks.  The path is non-obvious.
    """
    rng = np.random.default_rng(2026)  # fixed seed for reproducibility

    # Major peaks — define the macro structure
    major_peaks = [
        # Global summit — tucked in a corner, tall but narrow approach
        {'center': [0.72, 0.78] + [0.65] * (k - 2), 'height': 1.0, 'sigma': 0.22},
        # Deceptive peak — near center, almost as tall, wide basin
        {'center': [-0.10, 0.05] + [0.0] * (k - 2), 'height': 0.75, 'sigma': 0.35},
        # Ridge peak — between origin and summit, creates a false path
        {'center': [0.35, 0.50] + [0.30] * (k - 2), 'height': 0.65, 'sigma': 0.20},
        # Distant local peak — opposite corner, moderate
        {'center': [-0.60, -0.55] + [-0.50] * (k - 2), 'height': 0.55, 'sigma': 0.28},
    ]

    # Minor peaks — foothills and false summits
    minor_peaks = [
        {'center': [0.50, 0.20] + [0.15] * (k - 2), 'height': 0.40, 'sigma': 0.12},
        {'center': [-0.40, 0.35] + [0.10] * (k - 2), 'height': 0.35, 'sigma': 0.14},
        {'center': [0.15, -0.30] + [-0.20] * (k - 2), 'height': 0.30, 'sigma': 0.13},
        {'center': [-0.20, -0.65] + [-0.30] * (k - 2), 'height': 0.38, 'sigma': 0.15},
        {'center': [0.60, -0.10] + [0.20] * (k - 2), 'height': 0.32, 'sigma': 0.11},
        {'center': [-0.70, 0.60] + [0.40] * (k - 2), 'height': 0.28, 'sigma': 0.16},
        {'center': [0.20, 0.70] + [0.50] * (k - 2), 'height': 0.42, 'sigma': 0.13},
        {'center': [-0.50, 0.00] + [-0.15] * (k - 2), 'height': 0.25, 'sigma': 0.10},
        {'center': [0.80, 0.40] + [0.35] * (k - 2), 'height': 0.36, 'sigma': 0.12},
        {'center': [0.00, -0.80] + [-0.60] * (k - 2), 'height': 0.30, 'sigma': 0.18},
        {'center': [-0.30, 0.80] + [0.70] * (k - 2), 'height': 0.33, 'sigma': 0.11},
        {'center': [0.45, 0.65] + [0.55] * (k - 2), 'height': 0.45, 'sigma': 0.10},
    ]

    # Spectral noise — multi-frequency roughness
    n_freqs = 20
    noise_freqs = rng.uniform(-3.0, 3.0, (n_freqs, k))
    # Amplitude decays with frequency magnitude (higher freq = smaller bumps)
    freq_mags = np.linalg.norm(noise_freqs, axis=1)
    noise_amps = 0.08 / (1.0 + freq_mags)  # gentle decay
    noise_phases = rng.uniform(0, 2 * np.pi, n_freqs)

    return RuggedLandscape(
        major_peaks, minor_peaks,
        noise_freqs, noise_amps, noise_phases,
        k, w_major=0.55, w_minor=0.25, w_noise=0.20,
    )


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
    """Create a complex cost terrain with multiple ridges and valleys.

    The cost terrain is designed to interact with the rugged fitness landscape:
    - A major cost ridge blocks the direct path to the summit
    - The deceptive peak (near center) sits in a low-cost zone, making it
      tempting to stay there
    - A secondary ridge guards the approach from the south
    - The cheapest corridor to the summit goes through a low-fitness valley,
      requiring the team to accept temporary fitness loss for long-term gain
    """
    ridges = [
        # Main cost ridge — blocks the direct diagonal to summit
        {'center': [0.30, 0.40] + [0.30] * (k - 2), 'height': 2.5, 'sigma': 0.22},
        # Secondary ridge — guards southern approach
        {'center': [0.55, 0.10] + [0.15] * (k - 2), 'height': 1.8, 'sigma': 0.18},
        # Implementation cost near the summit
        {'center': [0.65, 0.70] + [0.60] * (k - 2), 'height': 1.2, 'sigma': 0.15},
        # Cost pocket near the deceptive peak — low cost, tempting to stay
        # (modeled as negative height to create a cost valley — but since
        #  CostLandscape uses base_cost + sum, we just don't put a ridge here)
        # Small ridge near the distant local peak
        {'center': [-0.50, -0.45] + [-0.40] * (k - 2), 'height': 1.0, 'sigma': 0.20},
        # Scattered minor cost bumps
        {'center': [0.10, 0.65] + [0.45] * (k - 2), 'height': 0.6, 'sigma': 0.12},
        {'center': [-0.35, -0.20] + [-0.10] * (k - 2), 'height': 0.5, 'sigma': 0.14},
    ]
    return CostLandscape(ridges, k, base_cost=0.1)
