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

    Implemented as Gaussian cost ridges + optional high-frequency spectral
    noise that creates narrow cheap corridors through expensive terrain.

    Parameters
    ----------
    ridges : list of dict
        Each dict has keys 'center' (array-like, K), 'height' (float),
        'sigma' (float).  Height is the peak cost at the ridge centre.
    k : int
        Number of preference dimensions.
    base_cost : float
        Minimum per-step terrain cost everywhere.  Default 0.1.
    noise_freqs, noise_amps, noise_phases : arrays, optional
        High-frequency spectral noise for cost terrain.
    """

    def __init__(self, ridges, k, base_cost=0.1,
                 noise_freqs=None, noise_amps=None, noise_phases=None):
        self.k = k
        self.n_ridges = len(ridges)
        self.base_cost = base_cost
        self.centers = np.array([r['center'][:k] for r in ridges],
                                dtype=np.float64)
        self.heights = np.array([r['height'] for r in ridges],
                                dtype=np.float64)
        self.sigmas = np.array([r['sigma'] for r in ridges],
                               dtype=np.float64)
        # High-frequency spectral noise
        if noise_freqs is not None:
            self.noise_freqs = np.array(noise_freqs, dtype=np.float64)
            self.noise_amps = np.array(noise_amps, dtype=np.float64)
            self.noise_phases = np.array(noise_phases, dtype=np.float64)
        else:
            self.noise_freqs = None
            self.noise_amps = None
            self.noise_phases = None

    def _eval_noise(self, prefs):
        """Evaluate high-frequency cost noise.  Returns (N,) in [0, 1]."""
        if self.noise_freqs is None:
            return np.zeros(len(prefs))
        p = prefs.astype(np.float64)
        dot = p @ self.noise_freqs.T  # (N, F)
        waves = self.noise_amps[None, :] * np.sin(
            2.0 * np.pi * dot + self.noise_phases[None, :])  # (N, F)
        raw = waves.sum(axis=1)  # (N,)
        amp_sum = self.noise_amps.sum()
        if amp_sum > 0:
            return (raw + amp_sum) / (2.0 * amp_sum)  # [0, 1]
        return np.zeros(len(p))

    def _eval_noise_gradient(self, prefs):
        """Gradient of cost noise.  Returns (N, K)."""
        if self.noise_freqs is None:
            return np.zeros((len(prefs), self.k))
        p = prefs.astype(np.float64)
        dot = p @ self.noise_freqs.T  # (N, F)
        cos_waves = self.noise_amps[None, :] * np.cos(
            2.0 * np.pi * dot + self.noise_phases[None, :])  # (N, F)
        grad = (2.0 * np.pi) * (cos_waves @ self.noise_freqs)  # (N, K)
        amp_sum = self.noise_amps.sum()
        if amp_sum > 0:
            grad = grad / (2.0 * amp_sum)
        return grad

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
        # Add high-frequency noise
        total += self._eval_noise(p)
        return total

    def cost_gradient(self, prefs):
        """Compute gradient of terrain cost (points uphill in cost).

        Parameters
        ----------
        prefs : (N, K) array

        Returns
        -------
        grad : (N, K) array — cost gradient (NOT normalized)
        cost : (N,) array — cost values
        """
        p = prefs.astype(np.float64)
        diff = p[:, None, :] - self.centers[None, :, :]  # (N, R, K)
        dist_sq = (diff ** 2).sum(axis=2)  # (N, R)
        ridge_vals = self.heights[None, :] * np.exp(
            -dist_sq / (2.0 * self.sigmas[None, :] ** 2))  # (N, R)
        # Gradient of each ridge: d/dx [h*exp(-||x-c||^2/(2s^2))] = val * (c-x)/s^2
        # But we want d/dx cost, and cost = sum of ridges, so:
        # d/dx = sum_r  ridge_val_r * (-(x-c_r)) / s_r^2
        #      = sum_r  ridge_val_r * (c_r - x) / s_r^2
        # Actually: d/dx [exp(-||x-c||^2/(2s^2))] = exp(...) * (c-x)/s^2
        # So d/dx [h * exp(...)] = h * exp(...) * (c-x)/s^2 = ridge_val * (c-x)/s^2
        # But wait, (c-x) points TOWARD the center (downhill in cost).
        # We want the gradient pointing UPHILL in cost, which is -(c-x) = (x-c).
        # Actually no: the gradient of the Gaussian is (c-x)/s^2 * val, which
        # points toward the center = uphill in cost.  Let me be careful:
        # f(x) = h * exp(-||x-c||^2/(2s^2))
        # df/dx = h * exp(...) * (-2(x-c)/(2s^2)) = -val * (x-c)/s^2 = val*(c-x)/s^2
        # This points toward c (the peak of cost), i.e., UPHILL in cost. Correct.
        ridge_grad = (ridge_vals[:, :, None] *
                      (self.centers[None, :, :] - p[:, None, :]) /
                      (self.sigmas[None, :, None] ** 2))  # (N, R, K)
        grad = ridge_grad.sum(axis=1)  # (N, K)
        # Add noise gradient
        grad += self._eval_noise_gradient(p)
        cost = self.base_cost + ridge_vals.sum(axis=1) + self._eval_noise(p)
        return grad, cost


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
    """Create a highly rugged multi-scale landscape.

    The landscape has:
    - 4 major peaks (1 global summit + 3 prominent local optima)
    - 16 minor peaks (foothills, false summits, ridgeline bumps)
    - 50 spectral noise frequencies (aggressive roughness at multiple
      scales, creating ridges, saddle points, and false gradients)

    The global summit is tucked in a corner behind deceptive local peaks
    and cost barriers.  The terrain is rugged enough that individual
    gradient sensing is unreliable — team aggregation is required.
    """
    rng = np.random.default_rng(2026)  # fixed seed for reproducibility

    # Major peaks — define the macro structure
    major_peaks = [
        # Global summit — narrow, tucked in a corner
        {'center': [0.72, 0.78] + [0.65] * (k - 2), 'height': 1.0, 'sigma': 0.18},
        # Deceptive peak — near center, almost as tall, wide basin
        {'center': [-0.10, 0.05] + [0.0] * (k - 2), 'height': 0.82, 'sigma': 0.30},
        # Ridge peak — between origin and summit, creates a false path
        {'center': [0.35, 0.50] + [0.30] * (k - 2), 'height': 0.70, 'sigma': 0.18},
        # Distant local peak — opposite corner, moderate
        {'center': [-0.60, -0.55] + [-0.50] * (k - 2), 'height': 0.60, 'sigma': 0.25},
    ]

    # Minor peaks — foothills, false summits, and ridgeline bumps
    # More peaks, taller relative to major, tighter sigmas = more deceptive
    minor_peaks = [
        {'center': [0.50, 0.20] + [0.15] * (k - 2), 'height': 0.52, 'sigma': 0.10},
        {'center': [-0.40, 0.35] + [0.10] * (k - 2), 'height': 0.48, 'sigma': 0.11},
        {'center': [0.15, -0.30] + [-0.20] * (k - 2), 'height': 0.42, 'sigma': 0.10},
        {'center': [-0.20, -0.65] + [-0.30] * (k - 2), 'height': 0.50, 'sigma': 0.12},
        {'center': [0.60, -0.10] + [0.20] * (k - 2), 'height': 0.44, 'sigma': 0.09},
        {'center': [-0.70, 0.60] + [0.40] * (k - 2), 'height': 0.38, 'sigma': 0.13},
        {'center': [0.20, 0.70] + [0.50] * (k - 2), 'height': 0.55, 'sigma': 0.10},
        {'center': [-0.50, 0.00] + [-0.15] * (k - 2), 'height': 0.35, 'sigma': 0.08},
        {'center': [0.80, 0.40] + [0.35] * (k - 2), 'height': 0.46, 'sigma': 0.09},
        {'center': [0.00, -0.80] + [-0.60] * (k - 2), 'height': 0.40, 'sigma': 0.14},
        {'center': [-0.30, 0.80] + [0.70] * (k - 2), 'height': 0.43, 'sigma': 0.09},
        {'center': [0.45, 0.65] + [0.55] * (k - 2), 'height': 0.58, 'sigma': 0.08},
        # Additional deceptive bumps near the approach to the summit
        {'center': [0.55, 0.60] + [0.50] * (k - 2), 'height': 0.50, 'sigma': 0.07},
        {'center': [0.65, 0.55] + [0.45] * (k - 2), 'height': 0.45, 'sigma': 0.08},
        {'center': [0.40, 0.70] + [0.60] * (k - 2), 'height': 0.47, 'sigma': 0.09},
        {'center': [0.10, 0.40] + [0.25] * (k - 2), 'height': 0.40, 'sigma': 0.10},
    ]

    # Spectral noise — aggressive multi-frequency roughness
    # Low frequencies (2-4): broad undulations
    # Medium frequencies (5-8): ridges and valleys
    # High frequencies (9-12): fine-grained ruggedness that makes
    #   individual gradient sensing unreliable
    n_low = 10
    n_mid = 20
    n_high = 20
    n_freqs = n_low + n_mid + n_high

    low_freqs = rng.uniform(-3.0, 3.0, (n_low, k))
    mid_freqs = rng.uniform(-7.0, 7.0, (n_mid, k))
    high_freqs = rng.uniform(-12.0, 12.0, (n_high, k))
    noise_freqs = np.vstack([low_freqs, mid_freqs, high_freqs])

    # Amplitudes: low-freq are strongest, but high-freq are still
    # significant enough to create false gradients
    low_amps = np.full(n_low, 0.10)
    mid_amps = np.full(n_mid, 0.06)
    high_amps = np.full(n_high, 0.035)
    noise_amps = np.concatenate([low_amps, mid_amps, high_amps])
    noise_phases = rng.uniform(0, 2 * np.pi, n_freqs)

    return RuggedLandscape(
        major_peaks, minor_peaks,
        noise_freqs, noise_amps, noise_phases,
        k, w_major=0.45, w_minor=0.30, w_noise=0.25,
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
    """Create a complex cost terrain with high-frequency noise.

    The cost terrain is designed to interact with the rugged fitness landscape:
    - Major cost ridges block the direct path to the summit
    - The deceptive peak (near center) sits in a low-cost zone, making it
      tempting to stay there
    - High-frequency spectral noise creates narrow cheap corridors through
      expensive terrain — only teams that share information can find them
    - The cheapest corridor to the summit goes through a low-fitness valley,
      requiring the team to accept temporary fitness loss for long-term gain
    """
    rng = np.random.default_rng(2027)  # different seed from fitness landscape

    # Cost ridge heights are calibrated to be comparable to fitness values
    # (fitness peaks at ~0.55).  With cost_weight=0.3 (default), the cost
    # gradient contribution is ~0.3 * ridge_height / sigma^2, which should
    # be meaningful but not overwhelming relative to the fitness gradient.
    ridges = [
        # Main cost ridge — blocks the direct diagonal to summit
        {'center': [0.30, 0.40] + [0.30] * (k - 2), 'height': 0.80, 'sigma': 0.20},
        # Secondary ridge — guards southern approach
        {'center': [0.55, 0.10] + [0.15] * (k - 2), 'height': 0.60, 'sigma': 0.16},
        # Implementation cost near the summit
        {'center': [0.65, 0.70] + [0.60] * (k - 2), 'height': 0.45, 'sigma': 0.14},
        # Cost wall along the northern approach
        {'center': [0.50, 0.75] + [0.55] * (k - 2), 'height': 0.55, 'sigma': 0.15},
        # Small ridge near the distant local peak
        {'center': [-0.50, -0.45] + [-0.40] * (k - 2), 'height': 0.35, 'sigma': 0.18},
        # Scattered cost bumps creating a maze-like cost field
        {'center': [0.10, 0.65] + [0.45] * (k - 2), 'height': 0.25, 'sigma': 0.10},
        {'center': [-0.35, -0.20] + [-0.10] * (k - 2), 'height': 0.20, 'sigma': 0.12},
        {'center': [0.40, 0.30] + [0.25] * (k - 2), 'height': 0.30, 'sigma': 0.11},
        {'center': [0.70, 0.50] + [0.45] * (k - 2), 'height': 0.28, 'sigma': 0.10},
    ]

    # High-frequency spectral noise for cost — creates narrow cheap corridors
    # These are deliberately high-frequency so the cost field has fine-grained
    # structure that individual particles can't easily sense.
    n_cost_freqs = 30
    cost_noise_freqs = rng.uniform(-10.0, 10.0, (n_cost_freqs, k))
    # Moderate amplitude: corridors are meaningful but don't dominate
    cost_noise_amps = np.full(n_cost_freqs, 0.04)
    cost_noise_phases = rng.uniform(0, 2 * np.pi, n_cost_freqs)

    return CostLandscape(ridges, k, base_cost=0.05,
                         noise_freqs=cost_noise_freqs,
                         noise_amps=cost_noise_amps,
                         noise_phases=cost_noise_phases)
