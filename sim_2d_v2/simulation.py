"""
Simulation class — state container and stepping logic.

Manages particle positions, preferences, direction memory, neighbor
finding, physics dispatch (Numba / NumPy / PyTorch), causal tracking,
and crossover.
"""

import math
import time
import numpy as np
from scipy.spatial import cKDTree

from .params import params, SPACE
from .spatial import (
    grid_build, periodic_dist,
    _count_radius, _query_radius, _sort_radius_nbrs, _query_knn,
)
from .physics_numba import _step_inner_prod_avg, _step_per_dim
from .physics_torch import step_torch, _HAS_TORCH


class Simulation:
    """2D toroidal particle simulation with preference-directed physics."""

    def __init__(self):
        self.rng = np.random.default_rng(
            params['seed'] if params['use_seed'] else None)
        self.reset()

    def reset(self):
        """Reinitialise all state from current params."""
        self.rng = np.random.default_rng(
            params['seed'] if params['use_seed'] else None)
        self.n = params['num_particles']
        self.k = params['k']
        n, k = self.n, self.k
        self._arange = np.arange(n)

        pos_dtype = np.float64 if params['use_f64'] else np.float32
        self.pos = np.zeros((n, 2), dtype=pos_dtype)
        self.prefs = np.zeros((n, k), dtype=np.float32)
        self.dir_matrix = np.zeros((n, k, 2),
                                   np.float64 if params['use_f64'] else np.float32)
        self._movement = np.zeros((n, 2),
                                   np.float64 if params['use_f64'] else np.float32)
        self.step_count = 0
        self.nbr_ids = None
        self._valid_mask = None
        self._t_search = 0.0
        self._t_build = 0.0
        self._t_query = 0.0
        self._t_physics = 0.0
        self._n_nbrs = 0
        self._init_positions(params['pos_dist'])
        self._init_preferences(params['pref_dist'])

        # Response vector: same distribution, independent draw
        if params['use_signal_response']:
            self._init_response(params['pref_dist'])
        else:
            self.response = self.prefs.copy()

        if params['unit_prefs']:
            self._normalize_prefs()

        self.tracked = np.zeros(n, dtype=bool)
        self.tracked_seed = np.zeros(n, dtype=bool)

    # ── Initialisation helpers ──────────────────────────────────────

    def _init_positions(self, dist):
        n, rng = self.n, self.rng
        L = SPACE
        pdtype = self.pos.dtype
        if dist == 1:  # Gaussian
            sigma = params['gauss_sigma']
            self.pos[:] = (rng.normal(L / 2, L * sigma, (n, 2)) % L).astype(pdtype)
        else:  # 0 = Uniform
            self.pos[:] = rng.uniform(0, L, (n, 2)).astype(pdtype)

    def _fill_pref_array(self, arr, dist):
        """Fill a (N, K) preference array using the given distribution."""
        n, k, rng = self.n, self.k, self.rng
        arr[:] = 0.0
        if dist == 1:  # Gaussian
            arr[:] = np.clip(rng.normal(0, 0.5, (n, k)), -1, 1).astype(np.float32)
        elif dist == 2:  # Sparse ±1
            if k < 2:
                arr[:, 0] = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
            else:
                for i in range(n):
                    dims = rng.choice(k, size=2, replace=False)
                    arr[i, dims[0]] = 1.0
                    arr[i, dims[1]] = -1.0
        elif dist == 3:  # Unit Normalized
            raw = rng.normal(0, 1, (n, k))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr[:] = (raw / norms).astype(np.float32)
        elif dist == 4:  # Binary d0 + noise
            eps = params['binary_noise_eps']
            arr[:, 0] = 1.0
            arr[n // 2:, 0] = -1.0
            if k > 1:
                arr[:, 1:] = rng.uniform(-eps, eps, (n, k - 1)).astype(np.float32)
        else:  # 0 = Uniform [-1,1]
            arr[:] = rng.uniform(-1, 1, (n, k)).astype(np.float32)

    def _init_preferences(self, dist):
        self._fill_pref_array(self.prefs, dist)

    def _init_response(self, dist):
        self.response = np.zeros((self.n, self.k), dtype=np.float32)
        self._fill_pref_array(self.response, dist)

    def _normalize_prefs(self):
        """Normalize each particle's preference vector to unit length."""
        norms = np.linalg.norm(self.prefs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.prefs[:] = (self.prefs / norms).astype(np.float32)

    # ── Neighbor finding ────────────────────────────────────────────

    def _find_neighbors(self):
        """Find neighbors. Method selected by params['knn_method']:
           0 = Hash Grid, 1 = cKDTree f64, 2 = cKDTree f32."""
        pos = self.pos
        n = len(pos)
        nbr_mode = params['neighbor_mode']
        n_nbr = min(params['n_neighbors'], n - 1)
        radius = params['neighbor_radius']
        knn_method = params['knn_method']

        _tb0 = time.perf_counter()

        if knn_method == 0:
            self._find_neighbors_hash(pos, n, nbr_mode, n_nbr, radius, _tb0)
        else:
            if knn_method == 1:
                query_pos = pos.astype(np.float64)
            else:
                query_pos = pos.astype(np.float32).astype(np.float64)
            query_pos = query_pos % SPACE

            self._t_build = 0.0
            _tq0 = time.perf_counter()
            tree = cKDTree(query_pos, boxsize=SPACE)

            if nbr_mode == 2:
                counts = tree.query_ball_point(query_pos, radius, workers=-1,
                                               return_length=True)
                max_k = int(counts.max())
                max_k = min(max(max_k, 1), n - 1)
                if max_k < 2:
                    self.nbr_ids = np.zeros((n, 1), dtype=np.int64)
                    self._valid_mask = np.zeros((n, 1), dtype=bool)
                else:
                    dists_raw, nbr_ids = tree.query(query_pos, k=max_k, workers=-1)
                    nbr_ids = nbr_ids[:, 1:]
                    dists_raw = dists_raw[:, 1:]
                    valid = dists_raw <= radius
                    self.nbr_ids = nbr_ids.astype(np.int64)
                    self._valid_mask = valid
            else:
                _, nbr_ids = tree.query(query_pos, k=n_nbr + 1, workers=-1)
                self.nbr_ids = nbr_ids[:, 1:].astype(np.int64)
                self._valid_mask = None

            self._t_query = time.perf_counter() - _tq0

        self._t_search = self._t_build + self._t_query

        # Debug KNN validation
        if params['debug_knn'] and knn_method == 0 and nbr_mode in (0, 1):
            self._debug_knn(pos, n_nbr)

    def _find_neighbors_hash(self, pos, n, nbr_mode, n_nbr, radius, _tb0):
        """Hash grid neighbor search."""
        if nbr_mode == 2:
            cell_size = radius
            sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
                grid_build(pos, cell_size)
            self._t_build = time.perf_counter() - _tb0

            _tq0 = time.perf_counter()
            counts = _count_radius(pos, sort_order, cell_start, cell_end,
                                   grid_res, cell_size_actual, radius, SPACE)
            max_nbr = int(counts.max())
            max_nbr = max(max_nbr, 1)
            max_nbr = min(max_nbr, n - 1)
            nbr_ids, valid, _ = _query_radius(
                pos, sort_order, np.empty(0, dtype=np.int32),
                cell_start, cell_end, grid_res, cell_size_actual,
                radius, SPACE, max_nbr)
            _sort_radius_nbrs(pos, nbr_ids, valid, SPACE)
            self._t_query = time.perf_counter() - _tq0
            self.nbr_ids = nbr_ids
            self._valid_mask = valid

        elif nbr_mode == 0:
            knn_radius_est = math.sqrt(n_nbr / (math.pi * max(n, 1)))
            cell_size = knn_radius_est * 1.5
            sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
                grid_build(pos, cell_size)
            self._t_build = time.perf_counter() - _tb0

            _tq0 = time.perf_counter()
            nbr_ids = _query_knn(pos, sort_order, cell_start, cell_end,
                                 grid_res, cell_size_actual, SPACE, n_nbr)
            self._t_query = time.perf_counter() - _tq0
            self.nbr_ids = nbr_ids
            self._valid_mask = None

        else:  # mode 1: KNN + Radius
            knn_radius_est = math.sqrt(n_nbr / (math.pi * max(n, 1)))
            cell_size = knn_radius_est * 1.5
            sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
                grid_build(pos, cell_size)
            self._t_build = time.perf_counter() - _tb0

            _tq0 = time.perf_counter()
            nbr_ids = _query_knn(pos, sort_order, cell_start, cell_end,
                                 grid_res, cell_size_actual, SPACE, n_nbr)
            self._t_query = time.perf_counter() - _tq0
            self.nbr_ids = nbr_ids
            self._valid_mask = None

    def _debug_knn(self, pos, n_nbr):
        """Compare hash grid KNN results against cKDTree reference."""
        pos_f64 = pos.astype(np.float64) % SPACE
        tree = cKDTree(pos_f64, boxsize=SPACE)
        _, ref_ids = tree.query(pos_f64, k=n_nbr + 1, workers=-1)
        ref_ids = ref_ids[:, 1:]

        hash_ids = self.nbr_ids
        n_particles = len(pos)
        n_mismatched = 0
        n_wrong_set = 0
        for pi in range(n_particles):
            ref_set = set(ref_ids[pi])
            hash_set = set(hash_ids[pi])
            if ref_set != hash_set:
                n_wrong_set += 1
                if n_mismatched < 5:
                    missing = ref_set - hash_set
                    extra = hash_set - ref_set
                    if missing:
                        miss_dists = []
                        for m in missing:
                            d = pos_f64[m] - pos_f64[pi]
                            d -= SPACE * np.round(d / SPACE)
                            miss_dists.append(np.sqrt(d @ d))
                        extra_dists = []
                        for e in extra:
                            d = pos_f64[e] - pos_f64[pi]
                            d -= SPACE * np.round(d / SPACE)
                            extra_dists.append(np.sqrt(d @ d))
                        print(f"  KNN mismatch particle {pi}: "
                              f"missing {len(missing)} (dists {sorted(miss_dists)[:3]}), "
                              f"extra {len(extra)} (dists {sorted(extra_dists)[:3]})")
                n_mismatched += 1
        if n_wrong_set > 0:
            print(f"[debug_knn] step {self.step_count}: "
                  f"{n_wrong_set}/{n_particles} particles have wrong neighbor sets")
        elif self.step_count % 500 == 0:
            print(f"[debug_knn] step {self.step_count}: all KNN match cKDTree")

    # ── Stepping ────────────────────────────────────────────────────

    def step(self, reuse_neighbors=False):
        """Run one simulation step (neighbor find + physics + post-processing)."""
        # Swap arrays so physics sees response-as-signal and vice versa
        swapped = params['use_signal_response'] and params['swap_signal_response']
        if swapped:
            self.prefs, self.response = self.response, self.prefs
        self._step_impl(reuse_neighbors)
        if swapped:
            self.prefs, self.response = self.response, self.prefs
        if params['social_mode'] == 1 and params['social'] != 0:
            self._quiet_dim_social()
        # Response social learning (same rate as signal, applied post-step)
        if params['use_signal_response'] and params['social'] != 0 and params['social_mode'] == 0:
            self._response_social()
        if params['unit_prefs']:
            self._normalize_prefs()
        if params['crossover'] and self.step_count % max(1, params['crossover_interval']) == 0:
            self.crossover_step()
        mode = params['track_mode']
        if mode == 2:
            self.expand_tracked()
        elif mode == 1:
            self.update_tracked_with_neighbors()

    def _step_impl(self, reuse_neighbors=False):
        """Core physics step — dispatches to Numba, NumPy, or PyTorch engine."""
        pos, prefs, dm = self.pos, self.prefs, self.dir_matrix
        # Use response vector if signal/response mode is on, else prefs for both
        resp = self.response if params['use_signal_response'] else prefs
        n = len(pos)
        k = self.k
        n_nbr = min(params['n_neighbors'], n - 1)
        step_size = params['step_size']
        repulsion = params['repulsion']
        dir_memory = params['dir_memory']
        # When quiet-dim mode is active, suppress the kernel's uniform social
        # update — it will be applied as a post-step with per-dim weighting.
        social = 0.0 if params['social_mode'] == 1 else params['social']
        pref_weighted = params['pref_weighted_dir']
        pref_inner = params['pref_inner_prod']
        inner_avg = params['inner_prod_avg']
        pref_dist_w = params['pref_dist_weight']
        best_mode = params['best_mode']
        pref_dist_sigma = params['neighbor_radius'] / 4.0
        arange_n = self._arange

        nbr_mode = params['neighbor_mode']

        if not reuse_neighbors or self.nbr_ids is None:
            self._find_neighbors()

        nbr_ids = self.nbr_ids
        valid = self._valid_mask

        _tp0 = time.perf_counter()

        # For KNN+Radius mode, compute valid mask from distances
        has_mask = valid is not None
        if nbr_mode == 1 and not has_mask:
            nbr_pos = pos[nbr_ids]
            toward = periodic_dist(pos[:, None, :], nbr_pos)
            dists = np.linalg.norm(toward, axis=2)
            valid = dists <= params['neighbor_radius']
            self._valid_mask = valid
            has_mask = True
        if nbr_mode == 2:
            has_mask = True
        n_valid = valid.sum(axis=1).clip(1) if has_mask else n_nbr

        if params['physics_engine'] == 1:
            # ── NumPy vectorized physics ──
            self._step_numpy(pos, prefs, resp, dm, nbr_ids, valid,
                             has_mask, n_valid, arange_n,
                             n, k, step_size, repulsion, dir_memory,
                             social, pref_weighted, pref_inner, inner_avg,
                             pref_dist_w, pref_dist_sigma, best_mode, _tp0)
            return

        if params['physics_engine'] == 2 and _HAS_TORCH:
            # ── PyTorch vectorized physics ──
            new_pos, new_prefs, new_dm, mov = step_torch(
                pos, prefs, resp, dm, nbr_ids, valid,
                SPACE, k, step_size, repulsion, dir_memory,
                social, params['social_dist_weight'],
                pref_weighted, pref_inner, inner_avg,
                pref_dist_w, pref_dist_sigma, best_mode,
                torch_precision=params['torch_precision'],
                torch_device_idx=params['torch_device'])
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs
            self.dir_matrix = new_dm.astype(dm.dtype)
            self._movement = mov.astype(self._movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            self._n_nbrs = nbr_ids.shape[1]
            self.step_count += 1
            return

        # ── Numba physics path (default) ──
        if inner_avg:
            if valid is None:
                valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
            else:
                valid_arr = valid
            prefs_f64 = prefs.astype(np.float64)
            resp_f64 = resp.astype(np.float64)
            new_pos, new_prefs, mov = _step_inner_prod_avg(
                pos, prefs_f64, resp_f64, nbr_ids.astype(np.int64), valid_arr,
                SPACE, k, step_size, repulsion, social,
                params['social_dist_weight'], pref_dist_w, pref_dist_sigma)
            self.pos = new_pos
            self.prefs = new_prefs.astype(np.float32)
            self._movement = mov
        else:
            if valid is None:
                valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
            else:
                valid_arr = valid
            prefs_f64 = prefs.astype(np.float64)
            resp_f64 = resp.astype(np.float64)
            new_pos, new_prefs, new_dm, mov = _step_per_dim(
                pos, prefs_f64, resp_f64, dm, nbr_ids.astype(np.int64), valid_arr,
                SPACE, k, step_size, repulsion, social,
                params['social_dist_weight'], dir_memory,
                pref_weighted, pref_inner,
                pref_dist_w, pref_dist_sigma, best_mode)
            self.pos = new_pos
            self.prefs = new_prefs.astype(np.float32)
            self.dir_matrix = new_dm
            self._movement = mov

        self._t_physics = time.perf_counter() - _tp0
        self._n_nbrs = nbr_ids.shape[1]
        self.step_count += 1

    def _step_numpy(self, pos, prefs, resp, dm, nbr_ids, valid,
                    has_mask, n_valid, arange_n,
                    n, k, step_size, repulsion, dir_memory,
                    social, pref_weighted, pref_inner, inner_avg,
                    pref_dist_w, pref_dist_sigma, best_mode, _tp0):
        """NumPy vectorized physics (matches original sim_gpu_update_gui.py)."""
        movement = self._movement
        movement[:] = 0.0

        nbr_pos = pos[nbr_ids]
        toward = periodic_dist(pos[:, None, :], nbr_pos)
        dists = np.linalg.norm(toward, axis=2, keepdims=True)
        toward_unit = toward / np.maximum(dists, 1e-12)

        if inner_avg:
            if valid is None:
                valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
            else:
                valid_arr = valid
            prefs_f64 = prefs.astype(np.float64)
            resp_f64 = resp.astype(np.float64)
            new_pos, new_prefs, mov = _step_inner_prod_avg(
                pos.astype(np.float64), prefs_f64, resp_f64,
                nbr_ids.astype(np.int64), valid_arr,
                SPACE, k, step_size, repulsion, social,
                params['social_dist_weight'], pref_dist_w, pref_dist_sigma)
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs.astype(np.float32)
            self._movement = mov.astype(movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            self._n_nbrs = nbr_ids.shape[1]
            self.step_count += 1
            return

        for ki in range(k):
            nbr_pref_k = prefs[nbr_ids, ki]

            if pref_weighted:
                weights = nbr_pref_k[:, :, None]
                if pref_dist_w:
                    gw = np.exp(-dists**2 / (2.0 * pref_dist_sigma**2))
                    weights = weights * gw
                weighted = weights * toward_unit
                if has_mask:
                    weighted = weighted * valid[:, :, None]
                    weighted_dir = weighted.sum(axis=1) / n_valid[:, None]
                else:
                    weighted_dir = weighted.mean(axis=1)
                dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * weighted_dir
                movement += resp[:, ki:ki+1] * dm[:, ki, :]

            else:
                if best_mode == 2:
                    # Same-sign max magnitude
                    my_sign = (prefs[:, ki] >= 0)[:, None]         # (N, 1)
                    nbr_sign = (nbr_pref_k >= 0)                   # (N, n_nbr)
                    same_sign = (my_sign == nbr_sign)               # (N, n_nbr)
                    score = np.abs(nbr_pref_k)
                    score = np.where(same_sign, score, -np.inf)
                    # Track particles with no valid same-sign neighbor
                    if has_mask:
                        any_valid = (same_sign & valid).any(axis=1)  # (N,)
                    else:
                        any_valid = same_sign.any(axis=1)            # (N,)
                elif best_mode == 1:
                    score = np.abs(nbr_pref_k)
                    any_valid = None
                else:
                    score = nbr_pref_k
                    any_valid = None
                if has_mask:
                    masked_score = np.where(valid, score, -np.inf)
                    best_local = np.argmax(masked_score, axis=1)
                else:
                    best_local = np.argmax(score, axis=1)
                best_nbr = nbr_ids[arange_n, best_local]

                disp = periodic_dist(pos, pos[best_nbr])
                dist = np.linalg.norm(disp, axis=1, keepdims=True)
                unit_dir = disp / np.maximum(dist, 1e-12)

                if any_valid is not None:
                    # Zero out for particles with no same-sign neighbor
                    mask_f = any_valid[:, None].astype(unit_dir.dtype)
                    unit_dir = unit_dir * mask_f

                dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * unit_dir

                compat = resp[:, ki] * prefs[best_nbr, ki]
                if pref_inner:
                    full_compat = (resp * prefs[best_nbr]).sum(axis=1) / k
                    compat = compat * full_compat
                if pref_dist_w:
                    gw = np.exp(-dist[:, 0]**2 / (2.0 * pref_dist_sigma**2))
                    compat = compat * gw
                if any_valid is not None:
                    compat = compat * any_valid.astype(compat.dtype)
                movement += compat[:, None] * dm[:, ki, :]

        unit_away = -toward_unit
        push_raw = unit_away / np.maximum(dists, 1e-6)
        if has_mask:
            push_raw = push_raw * valid[:, :, None]
            push = push_raw.sum(axis=1) / n_valid[:, None]
        else:
            push = push_raw.mean(axis=1)
        movement += repulsion * push

        self.pos = (pos + step_size * movement) % SPACE

        if social != 0:
            nbr_prefs = prefs[nbr_ids]
            if params['social_dist_weight']:
                d = dists[:, :, 0]
                w = 1.0 / (d + 1e-6)
                if has_mask:
                    w = w * valid
                w_sum = w.sum(axis=1, keepdims=True)
                w /= np.maximum(w_sum, 1e-10)
                nbr_mean = (nbr_prefs * w[:, :, None]).sum(axis=1)
            else:
                if has_mask:
                    nbr_prefs_m = nbr_prefs * valid[:, :, None]
                    nbr_mean = nbr_prefs_m.sum(axis=1) / n_valid[:, None]
                else:
                    nbr_mean = nbr_prefs.mean(axis=1)
            prefs[:] = (1.0 - social) * prefs + social * nbr_mean
            np.clip(prefs, -1, 1, out=prefs)

        self._t_physics = time.perf_counter() - _tp0
        self._n_nbrs = nbr_ids.shape[1]
        self.step_count += 1

    # ── Quiet-dim social learning ──────────────────────────────────

    def _quiet_dim_social(self):
        """Apply social learning with per-dimension rates inversely
        proportional to each dimension's contribution to movement.

        Dimensions that contribute most to the current movement are left
        alone (conformity / preservation).  Dimensions that contribute
        least are differentiated most (pushed away from local mean).

        Uses the social slider magnitude as the max per-dim learning rate.
        """
        if self.nbr_ids is None:
            return
        pos = self.pos
        prefs = self.prefs
        nbr_ids = self.nbr_ids
        valid = self._valid_mask
        k = self.k
        n = self.n
        social = params['social']
        best_mode = params['best_mode']

        # ── Compute per-dimension contribution magnitude ──
        # For each particle i and dimension d, measure |compat_d * u_d|
        # which is just |compat_d| since u_d is a unit vector.
        arange_n = self._arange
        dim_contrib = np.zeros((n, k), dtype=np.float32)

        for d in range(k):
            nbr_pref_d = prefs[nbr_ids, d]  # (N, n_nbr)

            if best_mode == 2:
                my_sign = (prefs[:, d] >= 0)[:, None]
                nbr_sign = (nbr_pref_d >= 0)
                same_sign = (my_sign == nbr_sign)
                score = np.abs(nbr_pref_d)
                score = np.where(same_sign, score, -np.inf)
                if valid is not None:
                    any_valid = (same_sign & valid).any(axis=1)
                else:
                    any_valid = same_sign.any(axis=1)
            elif best_mode == 1:
                score = np.abs(nbr_pref_d)
                any_valid = None
            else:
                score = nbr_pref_d
                any_valid = None

            if valid is not None:
                masked_score = np.where(valid, score, -np.inf)
                best_local = np.argmax(masked_score, axis=1)
            else:
                best_local = np.argmax(score, axis=1)
            best_nbr = nbr_ids[arange_n, best_local]

            compat = np.abs(prefs[:, d] * prefs[best_nbr, d])
            if any_valid is not None:
                compat = compat * any_valid.astype(compat.dtype)
            dim_contrib[:, d] = compat

        # ── Normalize to [0, 1] per particle ──
        # High contribution dims → weight near 1 (preserve)
        # Low contribution dims → weight near 0 (differentiate)
        max_contrib = dim_contrib.max(axis=1, keepdims=True)
        max_contrib = np.maximum(max_contrib, 1e-12)
        importance = dim_contrib / max_contrib  # (N, K) in [0, 1]

        # ── Per-dim social rate ──
        # social < 0 (differentiation): scale magnitude by (1 - importance)
        #   quiet dims get full |social|, loud dims get ~0
        # social > 0 (conformity): scale magnitude by (1 - importance) too
        #   quiet dims conform more, loud dims left alone
        per_dim_rate = social * (1.0 - importance)  # (N, K)

        # ── Compute neighbor mean prefs ──
        nbr_prefs = prefs[nbr_ids]  # (N, n_nbr, K)
        if valid is not None:
            has_mask = True
            n_valid = valid.sum(axis=1).clip(1)
            nbr_prefs_m = nbr_prefs * valid[:, :, None]
            nbr_mean = nbr_prefs_m.sum(axis=1) / n_valid[:, None]
        else:
            has_mask = False
            nbr_mean = nbr_prefs.mean(axis=1)  # (N, K)

        # ── Apply per-dimension update ──
        # prefs[i, d] = (1 - rate_d) * prefs[i, d] + rate_d * nbr_mean[i, d]
        self.prefs = np.clip(
            (1.0 - per_dim_rate) * prefs + per_dim_rate * nbr_mean,
            -1, 1
        ).astype(np.float32)

    # ── Response social learning ────────────────────────────────────

    def _response_social(self):
        """Apply uniform social learning to the response vector."""
        if self.nbr_ids is None:
            return
        social = params['social']
        resp = self.response
        nbr_ids = self.nbr_ids
        valid = self._valid_mask

        nbr_resp = resp[nbr_ids]  # (N, n_nbr, K)
        if valid is not None:
            n_valid = valid.sum(axis=1).clip(1)
            nbr_resp_m = nbr_resp * valid[:, :, None]
            nbr_mean = nbr_resp_m.sum(axis=1) / n_valid[:, None]
        else:
            nbr_mean = nbr_resp.mean(axis=1)

        self.response = np.clip(
            (1.0 - social) * resp + social * nbr_mean,
            -1, 1
        ).astype(np.float32)

    # ── Tracking ────────────────────────────────────────────────────

    def expand_tracked(self):
        """Causal spread: mark neighbors of tracked particles as tracked."""
        if not self.tracked.any() or self.nbr_ids is None:
            return
        nbr_ids = self.nbr_ids
        tracked_rows = nbr_ids[self.tracked]
        if self._valid_mask is not None:
            valid_rows = self._valid_mask[self.tracked]
            new_ids = tracked_rows[valid_rows]
        else:
            new_ids = tracked_rows.ravel()
        self.tracked[new_ids] = True

    def update_tracked_with_neighbors(self):
        """Set tracked = seed + current neighbors of seed (no growth)."""
        if not self.tracked_seed.any() or self.nbr_ids is None:
            return
        self.tracked[:] = self.tracked_seed
        nbr_ids = self.nbr_ids
        seed_rows = nbr_ids[self.tracked_seed]
        if self._valid_mask is not None:
            valid_rows = self._valid_mask[self.tracked_seed]
            new_ids = seed_rows[valid_rows]
        else:
            new_ids = seed_rows.ravel()
        self.tracked[new_ids] = True

    def crossover_step(self):
        """Swap tail preference dims between mutual nearest-neighbor pairs."""
        if self.nbr_ids is None:
            return
        n = self.n
        k = self.k
        pct = params['crossover_pct']
        n_keep = round(k * pct / 100.0)
        n_keep = max(0, min(n_keep, k))
        if n_keep == k:
            return

        nn = self.nbr_ids[:, 0]

        if self._valid_mask is not None:
            has_valid = self._valid_mask[:, 0]
        else:
            has_valid = np.ones(n, dtype=bool)

        arange_n = np.arange(n)
        is_mutual = (nn[nn] == arange_n) & has_valid & has_valid[nn]
        candidates = np.where(is_mutual & (nn > arange_n))[0]

        if len(candidates) == 0:
            return

        i_idx = candidates
        j_idx = nn[candidates]

        prefs = self.prefs
        i_tail = prefs[i_idx, n_keep:].copy()
        j_tail = prefs[j_idx, n_keep:].copy()
        prefs[i_idx, n_keep:] = j_tail
        prefs[j_idx, n_keep:] = i_tail

    # ── Rendering data ──────────────────────────────────────────────

    def get_render_data(self):
        """Return (positions_f32, colors_f32) for GPU upload.

        Colors come from signal (prefs) or response depending on
        vis_pref_source param.
        """
        k = self.k
        if params['use_signal_response'] and params['vis_pref_source'] == 1:
            src = self.response
        else:
            src = self.prefs
        colors = np.clip((src[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
        if k < 3:
            c = np.full((len(src), 3), 0.5, np.float32)
            c[:, :min(k, 3)] = colors[:, :min(k, 3)]
            colors = c
        return self.pos.astype(np.float32), colors

    def get_vis_prefs(self):
        """Return the preference array currently selected for visualization."""
        if params['use_signal_response'] and params['vis_pref_source'] == 1:
            return self.response
        return self.prefs

    def get_velocity_colors(self):
        """HSV-based velocity coloring (hue=angle, value=magnitude)."""
        vx, vy = self._movement[:, 0], self._movement[:, 1]
        angle = np.arctan2(vy, vx)
        hue = (angle / (2.0 * np.pi)) % 1.0
        mag = np.hypot(vx, vy)
        p95 = np.percentile(mag, 95) + 1e-8
        val = np.clip(mag / p95, 0.0, 1.0).astype(np.float32)

        h6 = hue * 6.0
        sector = h6.astype(np.int32) % 6
        f = (h6 - np.floor(h6)).astype(np.float32)
        q = val * (1.0 - f)
        t = val * f

        rgb = np.zeros((len(vx), 3), dtype=np.float32)
        m0 = sector == 0; m1 = sector == 1; m2 = sector == 2
        m3 = sector == 3; m4 = sector == 4; m5 = sector == 5
        rgb[m0, 0] = val[m0]; rgb[m0, 1] = t[m0]
        rgb[m1, 0] = q[m1];   rgb[m1, 1] = val[m1]
        rgb[m2, 1] = val[m2]; rgb[m2, 2] = t[m2]
        rgb[m3, 1] = q[m3];   rgb[m3, 2] = val[m3]
        rgb[m4, 0] = t[m4];   rgb[m4, 2] = val[m4]
        rgb[m5, 0] = val[m5]; rgb[m5, 2] = q[m5]

        return rgb

    def get_neighbor_lines(self):
        """Generate line vertex pairs for neighbor edge visualization."""
        if self.nbr_ids is None:
            return np.zeros((0, 2), dtype=np.float32)
        pos = self.pos
        nbr_ids = self.nbr_ids
        n, n_nbr = nbr_ids.shape

        starts = np.repeat(pos, n_nbr, axis=0)
        nbr_pos = pos[nbr_ids.ravel()]
        delta = periodic_dist(starts, nbr_pos)
        ends = starts + delta

        if self._valid_mask is not None:
            mask = self._valid_mask.ravel()
            starts = starts[mask]
            ends = ends[mask]

        n_edges = len(starts)
        lines = np.empty((n_edges * 2, 2), dtype=np.float32)
        lines[0::2] = starts.astype(np.float32)
        lines[1::2] = ends.astype(np.float32)
        return lines
