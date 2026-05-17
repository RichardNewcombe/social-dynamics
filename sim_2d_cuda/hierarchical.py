"""
Hierarchical physics: L0 individual + L1 group-level movement.

All particles remain as L0 individuals — no absorption.
Bonds form between nearby, preference-aligned particles.
Connected components above threshold become L1 groups.
L1 groups get a preference vector (per-dimension max of members)
and interact with OTHER L1 groups via per-dimension best-neighbor physics.
L1 group movement is applied uniformly to all members on top of
individual L0 movement.

Usage:
    cd ~/workspace/social-dynamics
    python -m sim_2d_cuda.headless_hierarchical
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .params import SPACE

HIER_DEFAULTS = dict(
    step_size=0.005,
    n_neighbors=21,
    bond_alpha=0.01,
    bond_beta=0.005,
    bond_pref_weight=0.5,
    bond_radius=0.08,
    bond_threshold=0.85,
    cluster_min_size=3,
    l1_step_size=0.005,
    l1_n_neighbors=5,
    l1_weight=1.0,          # multiplier on L1 group movement (0 = pure L0)
)


class HierarchicalSimulation:
    """Two-level physics: individual particles + emergent group dynamics."""

    def __init__(self, num_particles=500, k=3, seed=42, **kwargs):
        self.p = {**HIER_DEFAULTS, **kwargs}
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.L = SPACE
        self.N = num_particles
        self.step_count = 0

        # Particle state
        self.pos = self.rng.uniform(0, self.L, (self.N, 2)).astype(np.float64)
        self.prefs = self.rng.uniform(-1, 1, (self.N, k)).astype(np.float64)

        # Dense bond matrix (symmetric, only upper triangle used logically)
        self.bonds = np.zeros((self.N, self.N), dtype=np.float64)

        # Cluster state (recomputed each step)
        self.cluster_labels = -np.ones(self.N, dtype=np.int32)
        self.n_clusters = 0
        self.l1_prefs = np.zeros((0, k), dtype=np.float64)
        self.l1_centroids = np.zeros((0, 2), dtype=np.float64)

        # Stats
        self.history = []

    # ── Periodic helpers ────────────────────────────────────────────

    def _periodic_delta(self, a, b):
        d = b - a
        d -= self.L * np.round(d / self.L)
        return d

    def _periodic_wrap(self, pos):
        return pos % self.L

    # ── Main step ───────────────────────────────────────────────────

    def step(self):
        # 1. L0 individual physics
        l0_movement = self._l0_physics()

        # 2. Evolve bonds
        self._step_bonds()

        # 3. Identify clusters
        self._find_clusters()

        # 4. L1 group physics
        l1_movement = self._l1_physics()

        # 5. Compose and apply
        l1_weight = self.p['l1_weight']
        total_movement = l0_movement.copy()
        if l1_weight > 0:
            for c in range(self.n_clusters):
                members = np.where(self.cluster_labels == c)[0]
                total_movement[members] += l1_weight * l1_movement[c]

        self.pos = self._periodic_wrap(
            self.pos + self.p['step_size'] * total_movement
        )

        self.step_count += 1
        n_bonds = int((self.bonds > 0.01).sum()) // 2
        self.history.append({
            'step': self.step_count,
            'n_clusters': self.n_clusters,
            'n_free': int((self.cluster_labels == -1).sum()),
            'n_bonded': int((self.cluster_labels >= 0).sum()),
            'n_bonds': n_bonds,
            'cluster_sizes': [int((self.cluster_labels == c).sum())
                              for c in range(self.n_clusters)],
        })

    # ── L0 Physics (vectorized) ─────────────────────────────────────

    def _l0_physics(self):
        N, K = self.N, self.k
        n_nbrs = self.p['n_neighbors']

        pos = self._periodic_wrap(self.pos)
        tree = cKDTree(pos, boxsize=self.L)
        _, nbr_ids = tree.query(pos, k=n_nbrs + 1)
        nbr_ids = np.clip(nbr_ids[:, 1:], 0, N - 1)  # (N, n_nbrs)

        movement = np.zeros((N, 2), dtype=np.float64)

        for ki in range(K):
            nbr_prefs_k = self.prefs[nbr_ids, ki]  # (N, n_nbrs)
            best_local = np.argmax(nbr_prefs_k, axis=1)  # (N,)
            best_global = nbr_ids[np.arange(N), best_local]  # (N,)

            delta = self._periodic_delta(self.pos, self.pos[best_global])  # (N, 2)
            dist = np.sqrt((delta ** 2).sum(axis=1, keepdims=True))
            direction = delta / np.maximum(dist, 1e-12)

            compat = self.prefs[:, ki] * self.prefs[best_global, ki]  # (N,)
            movement += compat[:, None] * direction

        return movement

    # ── Bond evolution (vectorized) ─────────────────────────────────

    def _step_bonds(self):
        N = self.N
        alpha = self.p['bond_alpha']
        beta = self.p['bond_beta']
        pref_w = self.p['bond_pref_weight']
        radius = self.p['bond_radius']

        pos = self._periodic_wrap(self.pos)
        tree = cKDTree(pos, boxsize=self.L)
        pair_set = tree.query_pairs(radius)

        if not pair_set:
            # Decay all bonds
            self.bonds *= (1.0 - beta)
            self.bonds[self.bonds < 1e-6] = 0.0
            return

        pairs_arr = np.array(list(pair_set), dtype=np.int64)
        ii, jj = pairs_arr[:, 0], pairs_arr[:, 1]

        # Proximity
        delta = self._periodic_delta(self.pos[ii], self.pos[jj])
        dist = np.sqrt((delta ** 2).sum(axis=1))
        proximity = np.maximum(0.0, 1.0 - dist / radius)

        # Preference alignment
        pi, pj = self.prefs[ii], self.prefs[jj]
        dot = (pi * pj).sum(axis=1)
        ni = np.sqrt((pi ** 2).sum(axis=1))
        nj = np.sqrt((pj ** 2).sum(axis=1))
        denom = ni * nj
        cos_sim = np.where(denom > 1e-8, dot / denom, 0.0)
        pref_factor = np.maximum(0.0, cos_sim)

        # Growth signal
        growth = proximity * ((1.0 - pref_w) + pref_w * pref_factor)

        # Decay ALL bonds first
        self.bonds *= (1.0 - beta)

        # Then add growth for active pairs
        # Use np.add.at for duplicate-safe accumulation
        np.add.at(self.bonds, (ii, jj), alpha * growth)
        np.add.at(self.bonds, (jj, ii), alpha * growth)

        # Clip to [0, 1] and zero out tiny values
        np.clip(self.bonds, 0.0, 1.0, out=self.bonds)
        self.bonds[self.bonds < 1e-6] = 0.0

    # ── Cluster detection ───────────────────────────────────────────

    def _find_clusters(self):
        threshold = self.p['bond_threshold']
        min_size = self.p['cluster_min_size']

        self.cluster_labels[:] = -1
        self.n_clusters = 0

        # Build adjacency from thresholded bonds
        mask = self.bonds >= threshold
        if not mask.any():
            self.l1_prefs = np.zeros((0, self.k), dtype=np.float64)
            self.l1_centroids = np.zeros((0, 2), dtype=np.float64)
            return

        adj = csr_matrix(mask.astype(np.float64))
        n_comp, labels = connected_components(adj, directed=False)

        cluster_id = 0
        l1_prefs_list = []
        l1_centroids_list = []

        for comp in range(n_comp):
            members = np.where(labels == comp)[0]
            if len(members) < min_size:
                continue

            self.cluster_labels[members] = cluster_id

            # L1 preference: per-dimension MAX
            l1_pref = self.prefs[members].max(axis=0)
            l1_prefs_list.append(l1_pref)

            # Periodic centroid
            ref = self.pos[members[0]]
            deltas = self._periodic_delta(ref, self.pos[members])
            centroid = self._periodic_wrap(ref + deltas.mean(axis=0))
            l1_centroids_list.append(centroid)

            cluster_id += 1

        self.n_clusters = cluster_id
        if cluster_id > 0:
            self.l1_prefs = np.array(l1_prefs_list)
            self.l1_centroids = np.array(l1_centroids_list)
        else:
            self.l1_prefs = np.zeros((0, self.k), dtype=np.float64)
            self.l1_centroids = np.zeros((0, 2), dtype=np.float64)

    # ── L1 Physics (per-dimension best-neighbor between groups) ─────

    def _l1_physics(self):
        nc = self.n_clusters
        if nc < 2:
            return np.zeros((max(nc, 0), 2), dtype=np.float64)

        K = self.k
        n_nbrs = min(self.p['l1_n_neighbors'], nc - 1)

        tree = cKDTree(self.l1_centroids, boxsize=self.L)
        _, nbr_ids = tree.query(self.l1_centroids, k=n_nbrs + 1)
        nbr_ids = np.clip(nbr_ids[:, 1:], 0, nc - 1)  # (nc, n_nbrs)

        movement = np.zeros((nc, 2), dtype=np.float64)

        for ki in range(K):
            nbr_prefs_k = self.l1_prefs[nbr_ids, ki]  # (nc, n_nbrs)
            best_local = np.argmax(nbr_prefs_k, axis=1)
            best_nbr = nbr_ids[np.arange(nc), best_local]

            delta = self._periodic_delta(
                self.l1_centroids, self.l1_centroids[best_nbr]
            )
            dist = np.sqrt((delta ** 2).sum(axis=1, keepdims=True))
            direction = delta / np.maximum(dist, 1e-12)

            compat = self.l1_prefs[:, ki] * self.l1_prefs[best_nbr, ki]
            movement += compat[:, None] * direction

        # Scale by L1 step size ratio
        l1_ratio = self.p['l1_step_size'] / max(self.p['step_size'], 1e-12)
        return movement * l1_ratio

    # ── Rendering data ──────────────────────────────────────────────

    def get_render_data(self):
        K = self.k
        if K >= 3:
            colors = np.clip((self.prefs[:, :3] + 1.0) * 0.5, 0, 1)
        elif K == 2:
            colors = np.zeros((self.N, 3))
            colors[:, :2] = np.clip((self.prefs[:, :2] + 1.0) * 0.5, 0, 1)
            colors[:, 2] = 0.5
        else:
            colors = np.full((self.N, 3), 0.5)
            colors[:, 0] = np.clip((self.prefs[:, 0] + 1.0) * 0.5, 0, 1)
        return self.pos.copy(), colors, self.cluster_labels.copy()

    def get_bond_data(self, min_strength=0.01):
        ii, jj = np.where((self.bonds > min_strength) & (np.arange(self.N)[:, None] < np.arange(self.N)[None, :]))
        bonds = []
        for idx in range(len(ii)):
            bonds.append((self.pos[ii[idx]].copy(), self.pos[jj[idx]].copy(),
                          float(self.bonds[ii[idx], jj[idx]])))
        return bonds

    def get_bond_arrays(self, min_strength=0.01):
        """Vectorized bond data for fast rendering."""
        triu = np.triu(self.bonds, k=1)
        ii, jj = np.where(triu > min_strength)
        if len(ii) == 0:
            return np.array([]), np.array([]), np.array([])
        strengths = triu[ii, jj]
        return ii, jj, strengths
