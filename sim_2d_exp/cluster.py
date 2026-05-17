"""
Bond-based cluster tracking layered on top of the existing simulation.

Runs every step (when enabled) and produces:
  - cluster_labels: per-particle int label (-1 = unbonded)
  - n_clusters: number of detected clusters
  - l1_centroids: (n_clusters, 2) periodic-mean of member positions
  - l1_prefs:     (n_clusters, k) "L1 preference vector" (per-dim max
                  or per-dim argmax-by-absolute-value)

Does NOT mutate Simulation state — pure side observer for now. The
L1 prefs/centroids are intended to be drawn as larger translucent
markers on top of the L0 particles.

Bond update formula (matches sim_hierarchical default):
  bond[i, j] *= (1 - beta)
  bond[i, j] += alpha * proximity(d_ij) * ((1 - pref_w) + pref_w * max(0, cos_sim))
  cluster = connected_components(bond > threshold)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .params import SPACE


class ClusterTracker:
    """Observe-only cluster detection over an existing Simulation.

    Lazily allocates the (N, N) bond matrix on first enable. If N changes
    (e.g. after a reset with different num_particles), the matrix is
    re-allocated and bonds reset.
    """

    def __init__(self):
        self.bonds = None           # (N, N) float32
        self.cluster_labels = None  # (N,) int32, -1 = unbonded
        self.n_clusters = 0
        self.l1_centroids = np.zeros((0, 2), dtype=np.float64)
        self.l1_prefs = np.zeros((0, 0), dtype=np.float32)
        self._N = 0
        self._k = 0

    def _ensure(self, n, k):
        if self._N != n or self.bonds is None:
            self.bonds = np.zeros((n, n), dtype=np.float32)
            self.cluster_labels = -np.ones(n, dtype=np.int32)
            self._N = n
        self._k = k

    def reset(self):
        if self.bonds is not None:
            self.bonds.fill(0.0)
            self.cluster_labels.fill(-1)
        self.n_clusters = 0
        self.l1_centroids = np.zeros((0, 2), dtype=np.float64)
        self.l1_prefs = np.zeros((0, max(self._k, 1)), dtype=np.float32)

    @staticmethod
    def _periodic_delta(a, b, L):
        d = b - a
        d -= L * np.round(d / L)
        return d

    def step(self, pos, prefs, params):
        """Update bonds, find clusters, compute L1 centroids/prefs."""
        n = len(pos)
        k = prefs.shape[1]
        self._ensure(n, k)

        alpha = params['cluster_bond_alpha']
        beta = params['cluster_bond_beta']
        pref_w = params['cluster_bond_pref_weight']
        radius = params['cluster_bond_radius']
        threshold = params['cluster_threshold']
        min_size = max(1, params['cluster_min_size'])
        L = SPACE

        # 1. Decay all bonds
        self.bonds *= (1.0 - beta)

        # 2. Update bonds for nearby pairs
        wrapped = pos % L
        tree = cKDTree(wrapped.astype(np.float64), boxsize=L)
        pair_set = tree.query_pairs(radius)

        if pair_set:
            pairs_arr = np.fromiter(
                (idx for pair in pair_set for idx in pair),
                dtype=np.int64, count=len(pair_set) * 2
            ).reshape(-1, 2)
            ii, jj = pairs_arr[:, 0], pairs_arr[:, 1]

            delta = self._periodic_delta(pos[ii], pos[jj], L)
            dist = np.sqrt((delta ** 2).sum(axis=1))
            proximity = np.maximum(0.0, 1.0 - dist / radius)

            # Cosine sim
            pi, pj = prefs[ii].astype(np.float32), prefs[jj].astype(np.float32)
            dot = (pi * pj).sum(axis=1)
            ni = np.sqrt((pi ** 2).sum(axis=1))
            nj = np.sqrt((pj ** 2).sum(axis=1))
            denom = ni * nj
            cos_sim = np.where(denom > 1e-8, dot / denom, 0.0)
            pref_factor = np.maximum(0.0, cos_sim)

            growth = (alpha * proximity *
                      ((1.0 - pref_w) + pref_w * pref_factor)).astype(np.float32)

            np.add.at(self.bonds, (ii, jj), growth)
            np.add.at(self.bonds, (jj, ii), growth)

        # Clamp + zero tiny
        np.clip(self.bonds, 0.0, 1.0, out=self.bonds)
        self.bonds[self.bonds < 1e-6] = 0.0

        # 3. Connected components on thresholded bonds
        self.cluster_labels[:] = -1
        mask = self.bonds >= threshold
        if not mask.any():
            self.n_clusters = 0
            self.l1_centroids = np.zeros((0, 2), dtype=np.float64)
            self.l1_prefs = np.zeros((0, k), dtype=np.float32)
            return

        adj = csr_matrix(mask)
        n_comp, raw_labels = connected_components(adj, directed=False)

        # 4. Filter by min_size, compute centroids and L1 prefs
        l1_mode = params.get('cluster_l1_mode', 0)  # 0=max, 1=argmax-by-abs
        cluster_id = 0
        centroids = []
        l1prefs = []

        for comp in range(n_comp):
            members = np.where(raw_labels == comp)[0]
            if len(members) < min_size:
                continue
            self.cluster_labels[members] = cluster_id

            # Periodic-aware centroid: anchor on first member
            ref = pos[members[0]]
            deltas = self._periodic_delta(ref, pos[members], L)
            centroid = (ref + deltas.mean(axis=0)) % L
            centroids.append(centroid)

            # L1 preference
            member_prefs = prefs[members].astype(np.float32)
            if l1_mode == 1:
                # argmax-by-abs-value preserves sign
                abs_vals = np.abs(member_prefs)
                best_member = np.argmax(abs_vals, axis=0)  # per dim
                l1_pref = member_prefs[best_member, np.arange(k)]
            else:
                # per-dim max
                l1_pref = member_prefs.max(axis=0)
            l1prefs.append(l1_pref)

            cluster_id += 1

        self.n_clusters = cluster_id
        if cluster_id > 0:
            self.l1_centroids = np.asarray(centroids, dtype=np.float64)
            self.l1_prefs = np.asarray(l1prefs, dtype=np.float32)
        else:
            self.l1_centroids = np.zeros((0, 2), dtype=np.float64)
            self.l1_prefs = np.zeros((0, k), dtype=np.float32)

    def get_marker_data(self):
        """Returns (positions_xy, colors_rgb) for L1 markers, ready for GPU upload.

        Positions in sim coords [0, L]; colors mapped from L1 prefs the same
        way Simulation.get_render_data does (per-dim → RGB).
        """
        if self.n_clusters == 0:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32))
        pos = self.l1_centroids.astype(np.float32)
        K = self.l1_prefs.shape[1]
        if K >= 3:
            rgb = np.clip((self.l1_prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
        elif K == 2:
            rgb = np.full((self.n_clusters, 3), 0.5, np.float32)
            rgb[:, :2] = np.clip((self.l1_prefs[:, :2] + 1.0) * 0.5, 0, 1)
        else:
            rgb = np.full((self.n_clusters, 3), 0.5, np.float32)
            rgb[:, 0] = np.clip((self.l1_prefs[:, 0] + 1.0) * 0.5, 0, 1)
        return pos, rgb
