"""
Hierarchical Emergence — continuous bond formation and multi-level merge.

Particles accumulate pairwise bond strengths with nearby neighbors.
When a connected component in the bond graph is fully bonded above a
merge threshold, the group collapses into a higher-level particle
with a richer representation:

  Level 0: k-dim preference vector           → interact via inner product
  Level 1: k×k covariance matrix + mean pref → interact via Frobenius similarity
  Level 2: (k×k)×(k×k) structure tensor      → interact via tensor contraction

Each level genuinely adds representational complexity.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import connected_components

from .params import SPACE

# ── Default emergence parameters ────────────────────────────────────

EMERGENCE_DEFAULTS = dict(
    bond_alpha=0.01,        # bond growth rate per step
    bond_beta=0.005,        # bond decay rate per step
    bond_pref_weight=0.5,   # contribution of pref alignment to bond growth
    merge_threshold=0.85,   # bond strength required for merge
    merge_min_size=3,       # minimum group size for merge
    merge_max_level=2,      # maximum hierarchy depth (0-indexed: 0,1,2)
)


class EmergenceSimulation:
    """2D toroidal particle simulation with hierarchical emergence.

    Self-contained: does not depend on the Simulation class. Uses
    NumPy physics (simple preference-directed movement) and scipy
    cKDTree for neighbor search.
    """

    def __init__(self, num_particles=500, k=3, seed=42, **kwargs):
        self.ep = {**EMERGENCE_DEFAULTS, **kwargs}
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.L = SPACE
        self.step_count = 0

        # ── Core sim params ──
        self.step_size = kwargs.get('step_size', 0.005)
        self.neighbor_radius = kwargs.get('neighbor_radius', 0.08)
        self.repulsion = kwargs.get('repulsion', 0.0)
        self.social = kwargs.get('social', 0.0)

        # ── Particle state ──
        n = num_particles
        self.pos = self.rng.uniform(0, self.L, (n, 2)).astype(np.float64)
        self.prefs = self.rng.uniform(-1, 1, (n, k)).astype(np.float32)
        self.level = np.zeros(n, dtype=np.int32)           # hierarchy level
        self.mass = np.ones(n, dtype=np.float32)            # visual mass (sqrt of member count)
        self.alive = np.ones(n, dtype=bool)                 # False = merged away

        # Level 1+ representation: covariance matrices
        # For level 0 particles this is just the outer product of their own pref
        self.cov = np.zeros((n, k, k), dtype=np.float32)
        for i in range(n):
            self.cov[i] = np.outer(self.prefs[i], self.prefs[i])

        # Level 2 representation: (k*k)-dim flattened covariance-of-covariances
        self.l2_tensor = np.zeros((n, k*k, k*k), dtype=np.float32)

        # Member tracking (for stats)
        self.member_count = np.ones(n, dtype=np.int32)

        # ── Bond matrix (sparse, only alive×alive pairs) ──
        self.bonds = dok_matrix((n, n), dtype=np.float32)

        # ── Stats ──
        self.merge_events = []  # list of (step, level, member_count)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def n_alive(self):
        return int(self.alive.sum())

    @property
    def alive_idx(self):
        return np.where(self.alive)[0]

    def count_by_level(self):
        idx = self.alive_idx
        levels = self.level[idx]
        counts = {}
        for lv in range(self.ep['merge_max_level'] + 1):
            counts[lv] = int((levels == lv).sum())
        return counts

    def count_bonds(self, min_strength=0.01):
        """Count active bonds above a minimum strength."""
        count = 0
        for (i, j), v in self.bonds.items():
            if v > min_strength and i < j:
                count += 1
        return count

    # ── Periodic distance helpers ───────────────────────────────────

    def _periodic_delta(self, a, b):
        """Compute b - a with periodic wrapping."""
        d = b - a
        d -= self.L * np.round(d / self.L)
        return d

    def _periodic_dist(self, a, b):
        delta = self._periodic_delta(a, b)
        return np.sqrt((delta ** 2).sum(axis=-1))

    # ── Main step ───────────────────────────────────────────────────

    def step(self):
        """Run one full step: physics → bonds → merge check."""
        self._step_physics()
        self._step_bonds()
        self._check_merges()
        self.step_count += 1

    # ── Physics step (simplified preference-directed movement) ──────

    def _step_physics(self):
        """Move particles based on preference-weighted attraction to neighbors.

        Level 0-0 interactions: standard inner-product preference physics.
        Level 1-1 interactions: Frobenius similarity of covariance matrices.
        Cross-level (0-1, 1-2, etc.): use mean preference as fallback.
        """
        idx = self.alive_idx
        if len(idx) < 2:
            return

        pos = self.pos[idx]
        prefs = self.prefs[idx]
        levels = self.level[idx]
        covs = self.cov[idx]
        n = len(idx)

        # Build KD-tree on alive positions
        tree = cKDTree(pos % self.L, boxsize=self.L)
        pairs = tree.query_ball_tree(tree, self.neighbor_radius)

        movement = np.zeros((n, 2), dtype=np.float64)

        for li in range(n):
            nbrs = [j for j in pairs[li] if j != li]
            if not nbrs:
                continue

            nbr_arr = np.array(nbrs)
            nbr_pos = pos[nbr_arr]
            delta = self._periodic_delta(pos[li], nbr_pos)
            dists = np.sqrt((delta ** 2).sum(axis=1))
            dists_safe = np.maximum(dists, 1e-12)
            toward_unit = delta / dists_safe[:, None]

            my_level = levels[li]

            # Compute attraction weights per neighbor
            weights = np.zeros(len(nbrs), dtype=np.float64)

            for ni_local, ni in enumerate(nbrs):
                nbr_level = levels[ni]

                if my_level == 0 and nbr_level == 0:
                    # Standard inner product
                    weights[ni_local] = float(np.dot(prefs[li], prefs[ni]))
                elif my_level >= 1 and nbr_level >= 1:
                    # Frobenius inner product of covariance matrices
                    frob = float(np.sum(covs[li] * covs[ni]))
                    # Normalize by norms
                    n1 = np.sqrt(np.sum(covs[li] ** 2)) + 1e-12
                    n2 = np.sqrt(np.sum(covs[ni] ** 2)) + 1e-12
                    weights[ni_local] = frob / (n1 * n2)
                else:
                    # Cross-level: fall back to mean preference inner product
                    weights[ni_local] = float(np.dot(prefs[li], prefs[ni]))

            # Weighted movement toward/away from neighbors
            weighted_dir = (weights[:, None] * toward_unit).mean(axis=0)
            movement[li] = weighted_dir

            # Repulsion (distance-based)
            if self.repulsion != 0:
                push = (-toward_unit / dists_safe[:, None]).mean(axis=0)
                movement[li] += self.repulsion * push

        # Apply movement
        self.pos[idx] = (pos + self.step_size * movement) % self.L

        # Social learning (preference averaging with neighbors)
        if self.social != 0:
            new_prefs = prefs.copy()
            for li in range(n):
                nbrs = [j for j in pairs[li] if j != li]
                if not nbrs:
                    continue
                nbr_arr = np.array(nbrs)
                nbr_mean = prefs[nbr_arr].mean(axis=0)
                new_prefs[li] = (1.0 - self.social) * prefs[li] + self.social * nbr_mean
            self.prefs[idx] = np.clip(new_prefs, -1, 1).astype(np.float32)

    # ── Bond evolution ──────────────────────────────────────────────

    def _step_bonds(self):
        """Grow bonds between nearby particles, decay all bonds."""
        idx = self.alive_idx
        if len(idx) < 2:
            return

        pos = self.pos[idx]
        prefs = self.prefs[idx]
        n = len(idx)

        alpha = self.ep['bond_alpha']
        beta = self.ep['bond_beta']
        pref_w = self.ep['bond_pref_weight']

        # Build KD-tree
        tree = cKDTree(pos % self.L, boxsize=self.L)
        pairs = tree.query_ball_tree(tree, self.neighbor_radius)

        # Set of currently-interacting pairs (global indices)
        active_pairs = set()

        for li in range(n):
            gi = idx[li]
            nbrs = [j for j in pairs[li] if j != li]
            for nj in nbrs:
                gj = idx[nj]
                if gi >= gj:
                    continue  # only process each pair once
                active_pairs.add((gi, gj))

                # Proximity factor: 1 at distance 0, 0 at neighbor_radius
                delta = self._periodic_delta(pos[li], pos[nj])
                dist = np.sqrt((delta ** 2).sum())
                proximity = max(0.0, 1.0 - dist / self.neighbor_radius)

                # Preference alignment factor
                p_i = prefs[li]
                p_j = prefs[nj]
                norm_i = np.linalg.norm(p_i)
                norm_j = np.linalg.norm(p_j)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos_sim = float(np.dot(p_i, p_j) / (norm_i * norm_j))
                    pref_factor = max(0.0, cos_sim)  # only positive alignment bonds
                else:
                    pref_factor = 0.0

                # Combined growth signal
                growth = proximity * ((1.0 - pref_w) + pref_w * pref_factor)

                # Update bond
                current = self.bonds.get((gi, gj), 0.0)
                new_val = current + alpha * growth - beta * current
                new_val = np.clip(new_val, 0.0, 1.0)
                if new_val > 1e-6:
                    self.bonds[gi, gj] = new_val
                    self.bonds[gj, gi] = new_val
                elif (gi, gj) in self.bonds:
                    del self.bonds[gi, gj]
                    if (gj, gi) in self.bonds:
                        del self.bonds[gj, gi]

        # Decay bonds for pairs NOT currently interacting
        keys_to_update = []
        for (i, j), v in list(self.bonds.items()):
            if i >= j:
                continue
            if (i, j) not in active_pairs:
                new_val = v - beta * v
                keys_to_update.append((i, j, new_val))

        for i, j, new_val in keys_to_update:
            if new_val < 1e-6:
                if (i, j) in self.bonds:
                    del self.bonds[i, j]
                if (j, i) in self.bonds:
                    del self.bonds[j, i]
            else:
                self.bonds[i, j] = new_val
                self.bonds[j, i] = new_val

    # ── Merge check ─────────────────────────────────────────────────

    def _check_merges(self):
        """Find fully-bonded connected components and merge them."""
        idx = self.alive_idx
        if len(idx) < self.ep['merge_min_size']:
            return

        threshold = self.ep['merge_threshold']
        max_level = self.ep['merge_max_level']
        min_size = self.ep['merge_min_size']

        # Build adjacency matrix for alive particles with bonds > threshold
        n = len(idx)
        idx_to_local = {int(g): l for l, g in enumerate(idx)}

        # Sparse adjacency for thresholded bonds
        adj = dok_matrix((n, n), dtype=np.float32)
        for (i, j), v in self.bonds.items():
            if i < j and v >= threshold:
                if i in idx_to_local and j in idx_to_local:
                    li, lj = idx_to_local[i], idx_to_local[j]
                    adj[li, lj] = 1
                    adj[lj, li] = 1

        if adj.nnz == 0:
            return

        # Find connected components
        n_comp, labels = connected_components(adj.tocsr(), directed=False)

        for comp_id in range(n_comp):
            members_local = np.where(labels == comp_id)[0]
            if len(members_local) < min_size:
                continue

            members_global = idx[members_local]

            # Verify ALL internal pairs are bonded above threshold
            fully_bonded = True
            for a_idx in range(len(members_global)):
                for b_idx in range(a_idx + 1, len(members_global)):
                    ga, gb = int(members_global[a_idx]), int(members_global[b_idx])
                    bond_val = self.bonds.get((ga, gb), 0.0)
                    if bond_val < threshold:
                        fully_bonded = False
                        break
                if not fully_bonded:
                    break

            if not fully_bonded:
                continue

            # Determine merge level: max level of members + 1
            member_levels = self.level[members_global]
            new_level = int(member_levels.max()) + 1
            if new_level > max_level:
                continue

            # ── Perform merge ──
            self._merge_group(members_global, new_level)

    def _merge_group(self, members, new_level):
        """Merge a group of particles into a single higher-level particle.

        Reuses the slot of the first member.
        """
        k = self.k
        survivor = int(members[0])
        absorbed = members[1:]

        member_prefs = self.prefs[members]
        member_pos = self.pos[members]
        member_masses = self.member_count[members].astype(np.float32)
        total_mass = member_masses.sum()

        # Position: mass-weighted centroid (with periodic wrapping)
        # Use first member as reference to handle wrapping
        ref = member_pos[0]
        deltas = self._periodic_delta(ref, member_pos)
        weighted_delta = (deltas * member_masses[:, None]).sum(axis=0) / total_mass
        new_pos = (ref + weighted_delta) % self.L

        # Mean preference (mass-weighted)
        new_pref = (member_prefs * member_masses[:, None]).sum(axis=0) / total_mass

        # Covariance matrix of member preferences (captures internal diversity)
        centered = member_prefs - new_pref[None, :]
        # Weighted covariance
        new_cov = np.zeros((k, k), dtype=np.float32)
        for mi in range(len(members)):
            new_cov += member_masses[mi] * np.outer(centered[mi], centered[mi])
        new_cov /= total_mass
        # Add the mean outer product to make it a second moment matrix
        # This preserves information about both the center and spread
        new_cov += np.outer(new_pref, new_pref)

        # Level 2 tensor: covariance of covariance matrices
        if new_level >= 2:
            member_covs = self.cov[members]
            flat_covs = member_covs.reshape(len(members), k * k)
            mean_flat = flat_covs.mean(axis=0)
            centered_flat = flat_covs - mean_flat[None, :]
            l2 = np.zeros((k * k, k * k), dtype=np.float32)
            for mi in range(len(members)):
                l2 += np.outer(centered_flat[mi], centered_flat[mi])
            l2 /= len(members)
            self.l2_tensor[survivor] = l2

        # Update survivor
        self.pos[survivor] = new_pos
        self.prefs[survivor] = np.clip(new_pref, -1, 1).astype(np.float32)
        self.cov[survivor] = new_cov
        self.level[survivor] = new_level
        self.member_count[survivor] = int(total_mass)
        self.mass[survivor] = np.sqrt(float(total_mass))
        self.alive[survivor] = True

        # Kill absorbed particles
        self.alive[absorbed] = False

        # Clean up bonds involving absorbed particles
        for g in absorbed:
            keys_to_remove = [(i, j) for (i, j) in self.bonds.keys() if i == g or j == g]
            for key in keys_to_remove:
                if key in self.bonds:
                    del self.bonds[key]

        # Transfer bonds from absorbed to survivor
        # (bonds between absorbed members are already removed)

        # Record event
        self.merge_events.append((self.step_count, new_level, int(total_mass)))

    # ── Rendering data ──────────────────────────────────────────────

    def get_render_data(self):
        """Return (positions, colors, sizes, levels) for alive particles."""
        idx = self.alive_idx
        pos = self.pos[idx]
        prefs = self.prefs[idx]
        levels = self.level[idx]
        masses = self.mass[idx]

        # Colors from preferences (RGB from first 3 dims)
        k = self.k
        colors = np.clip((prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
        if k < 3:
            c = np.full((len(prefs), 3), 0.5, np.float32)
            c[:, :min(k, 3)] = colors[:, :min(k, 3)]
            colors = c

        return pos, colors, masses, levels

    def get_bond_data(self, min_strength=0.01):
        """Return list of (pos_i, pos_j, strength) for alive bonds."""
        bonds = []
        alive_set = set(self.alive_idx.tolist())
        seen = set()
        for (i, j), v in self.bonds.items():
            if i < j and v > min_strength and i in alive_set and j in alive_set:
                if (i, j) not in seen:
                    seen.add((i, j))
                    bonds.append((self.pos[i], self.pos[j], float(v)))
        return bonds
