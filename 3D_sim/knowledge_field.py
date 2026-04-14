"""
Knowledge Field
================

A 2D scalar grid over (pref0, pref1) skill space that represents the
organization's accumulated knowledge.  The hidden fitness landscape is
the ceiling; knowledge can never exceed it.

Particles live on the knowledge surface.  They raise it by performing
noisy research (sampling the hidden fitness gradient), and knowledge
diffuses spatially so one team's discoveries benefit neighbours.

Coordinate convention
---------------------
- pref0, pref1 ∈ [-1, 1]  (particle skill preferences)
- Grid cells map linearly:  cell_x = (pref0 + 1) / 2 * G
                             cell_y = (pref1 + 1) / 2 * G
- Knowledge height ∈ [0, 1] (normalised against fitness range)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class KnowledgeField:
    """Persistent 2D knowledge manifold over skill space."""

    def __init__(self, grid_res=64, diffusion_sigma=1.0, decay=1.0):
        """
        Parameters
        ----------
        grid_res : int
            Resolution of the square knowledge grid.
        diffusion_sigma : float
            Gaussian blur sigma for spatial diffusion each step.
        decay : float
            Multiplicative decay per step (1.0 = no decay).
        """
        self.G = grid_res
        self.diffusion_sigma = diffusion_sigma
        self.decay = decay

        # The knowledge surface: scalar height at each grid cell
        # Normalised to [0, 1] where 1 = max fitness
        self.grid = np.zeros((grid_res, grid_res), dtype=np.float64)

        # Cache for the hidden fitness surface (set once)
        self._fitness_grid = None   # (G, G) normalised fitness values
        self._f_min = 0.0
        self._f_max = 1.0

    def set_fitness_surface(self, landscape, f_min=None, f_max=None):
        """Pre-compute the hidden fitness at every grid cell.

        Parameters
        ----------
        landscape : object with .fitness(prefs) method
            The hidden fitness landscape.  prefs is (N, K) in [-1, 1].
        f_min, f_max : float or None
            If None, computed from the grid sampling.
        """
        G = self.G
        xs = np.linspace(-1, 1, G)
        ys = np.linspace(-1, 1, G)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        # Build (G*G, K) probe array with dims 2+ = 0
        probes = np.zeros((G * G, 3), dtype=np.float64)
        probes[:, 0] = gx.ravel()
        probes[:, 1] = gy.ravel()

        raw_fitness = landscape.fitness(probes).reshape(G, G)

        if f_min is None:
            f_min = raw_fitness.min()
        if f_max is None:
            f_max = raw_fitness.max()

        self._f_min = f_min
        self._f_max = f_max
        span = max(f_max - f_min, 1e-12)
        self._fitness_grid = (raw_fitness - f_min) / span  # [0, 1]

    def reset(self):
        """Zero out the knowledge grid."""
        self.grid[:] = 0.0

    # ── Coordinate helpers ─────────────────────────────────────────

    def _pref_to_cell(self, pref0, pref1):
        """Convert preference coordinates to grid cell indices.

        Parameters
        ----------
        pref0, pref1 : ndarray (N,)
            Skill preferences in [-1, 1].

        Returns
        -------
        cx, cy : ndarray (N,) int
            Grid cell indices, clamped to [0, G-1].
        """
        G = self.G
        cx = ((pref0 + 1.0) * 0.5 * (G - 1)).astype(int).clip(0, G - 1)
        cy = ((pref1 + 1.0) * 0.5 * (G - 1)).astype(int).clip(0, G - 1)
        return cx, cy

    # ── Read ───────────────────────────────────────────────────────

    def sample_knowledge(self, pref0, pref1):
        """Read the knowledge height at particle positions.

        Returns
        -------
        heights : ndarray (N,) float64  in [0, 1]
        """
        cx, cy = self._pref_to_cell(pref0, pref1)
        return self.grid[cx, cy]

    def sample_fitness(self, pref0, pref1):
        """Read the hidden fitness at particle positions (normalised).

        Returns
        -------
        fitness : ndarray (N,) float64  in [0, 1]
        """
        if self._fitness_grid is None:
            return np.zeros_like(pref0)
        cx, cy = self._pref_to_cell(pref0, pref1)
        return self._fitness_grid[cx, cy]

    def knowledge_gradient(self, pref0, pref1):
        """Compute the gradient of the knowledge surface at particle positions.

        Uses central differences on the grid, then samples at particle locations.

        Returns
        -------
        grad : ndarray (N, 2) float64
            Gradient in (pref0, pref1) directions.
        """
        G = self.G
        # Central differences with periodic wrapping
        grad_x = np.roll(self.grid, -1, axis=0) - np.roll(self.grid, 1, axis=0)
        grad_y = np.roll(self.grid, -1, axis=1) - np.roll(self.grid, 1, axis=1)
        # Scale by cell size (pref range is 2.0, G cells)
        cell_size = 2.0 / G
        grad_x /= (2.0 * cell_size)
        grad_y /= (2.0 * cell_size)

        cx, cy = self._pref_to_cell(pref0, pref1)
        grad = np.column_stack([grad_x[cx, cy], grad_y[cx, cy]])
        return grad

    # ── Write ──────────────────────────────────────────────────────

    def deposit_knowledge(self, pref0, pref1, amounts):
        """Raise the knowledge manifold at particle positions.

        Parameters
        ----------
        pref0, pref1 : ndarray (N,)
        amounts : ndarray (N,) float64
            How much to raise at each position.  Will be capped at
            the hidden fitness ceiling.
        """
        cx, cy = self._pref_to_cell(pref0, pref1)
        np.add.at(self.grid, (cx, cy), amounts)

        # Cap at the fitness ceiling
        if self._fitness_grid is not None:
            np.minimum(self.grid, self._fitness_grid, out=self.grid)

        # Also clamp to [0, 1] for safety
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    # ── Diffuse and decay ──────────────────────────────────────────

    def step_field(self):
        """Apply one step of decay + spatial diffusion."""
        # Decay
        if self.decay < 1.0:
            self.grid *= self.decay

        # Diffuse (Gaussian blur with periodic boundary)
        if self.diffusion_sigma > 0:
            self.grid = gaussian_filter(
                self.grid, sigma=self.diffusion_sigma, mode='wrap')

        # Re-cap after diffusion (blur can push above ceiling)
        if self._fitness_grid is not None:
            np.minimum(self.grid, self._fitness_grid, out=self.grid)
        np.clip(self.grid, 0.0, 1.0, out=self.grid)

    # ── Metrics ────────────────────────────────────────────────────

    def coverage(self):
        """Fraction of the fitness landscape that has been discovered.

        Returns the ratio of total knowledge volume to total fitness volume.
        """
        if self._fitness_grid is None:
            return 0.0
        fitness_vol = self._fitness_grid.sum()
        if fitness_vol < 1e-12:
            return 0.0
        return self.grid.sum() / fitness_vol

    def peak_knowledge(self):
        """Maximum knowledge height anywhere on the grid."""
        return self.grid.max()

    def peak_knowledge_location(self):
        """Grid cell of the highest knowledge point (as pref coords)."""
        idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
        G = self.G
        pref0 = idx[0] / (G - 1) * 2.0 - 1.0
        pref1 = idx[1] / (G - 1) * 2.0 - 1.0
        return pref0, pref1
