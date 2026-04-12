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
from .physics_grid import step_grid_field, step_grid_max_field, _force_landscape_from_nbrs
from .physics_grid_gpu import step_grid_max_field_gpu


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
        _pref_dtypes = {0: np.float16, 1: np.float32, 2: np.float64}
        self._pref_dtype = _pref_dtypes.get(params['pref_precision'], np.float32)
        self.prefs = np.zeros((n, k), dtype=self._pref_dtype)
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
        self._t_deposit = 0.0
        self._t_propagate = 0.0
        self._t_movement = 0.0
        self._t_social_grid = 0.0
        self._n_nbrs = 0
        self._grid_max_pref = None
        self._force_mag = None
        self._force_pref = None
        self._force_dir = None
        self._init_positions(params['pos_dist'])
        if params['perturb_pos_bits']:
            self._perturb_pos_lsb()
        self._init_preferences(params['pref_dist'])

        # Response vector: same distribution, independent draw
        if params['use_signal_response']:
            self._init_response(params['pref_dist'])
        else:
            self.response = self.prefs.copy()

        if params['unit_prefs']:
            self._normalize_prefs()
        if params['truncate_pref_bits']:
            self._truncate_pref_bits()
        if params['quantize_pref']:
            self._quantize_prefs()

        self.tracked = np.zeros(n, dtype=bool)
        self.tracked_seed = np.zeros(n, dtype=bool)

        # Spatial memory field: (G, G, K)
        G = params['grid_res']
        self.memory_field = np.zeros((G, G, k), dtype=np.float64)

        # Per-particle role arrays (opt-in via use_particle_roles)
        self._init_roles()

    # ── Role initialisation ────────────────────────────────────────

    def _init_roles(self):
        """Initialise per-particle role arrays.

        When ``params['use_particle_roles']`` is False (default), all
        arrays are set to neutral values (1.0 / 1.0) so the simulation
        behaves identically to the original code.

        When enabled, step-scale and influence are drawn from log-normal
        distributions controlled by ``role_step_scale_std`` and
        ``role_influence_std``.  A std of 0 gives uniform values.
        """
        n = self.n
        if params.get('use_particle_roles', False):
            ss_std = params.get('role_step_scale_std', 0.0)
            inf_std = params.get('role_influence_std', 0.0)
            if ss_std > 0:
                self.role_step_scale = self.rng.lognormal(
                    0.0, ss_std, n).astype(np.float64)
            else:
                self.role_step_scale = np.ones(n, dtype=np.float64)
            if inf_std > 0:
                self.role_influence = self.rng.lognormal(
                    0.0, inf_std, n).astype(np.float64)
            else:
                self.role_influence = np.ones(n, dtype=np.float64)
        else:
            self.role_step_scale = np.ones(n, dtype=np.float64)
            self.role_influence = np.ones(n, dtype=np.float64)

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

    def _perturb_pos_lsb(self):
        """Randomize the N least-significant mantissa bits of positions.

        Uses a SEPARATE rng (not self.rng) so the perturbation doesn't
        change the random sequence for preferences/response init.
        """
        n_bits = max(1, min(20, params['perturb_pos_n_bits']))
        mask = np.int64((1 << n_bits) - 1)
        # Use a separate rng so perturbation doesn't affect pref init
        perturb_rng = np.random.default_rng()
        if self.pos.dtype == np.float64:
            int_view = self.pos.view(np.int64)
            random_bits = perturb_rng.integers(
                0, int(mask + 1), size=self.pos.shape, dtype=np.int64)
            int_view[:] = (int_view & ~mask) | random_bits
        else:
            mask32 = np.int32((1 << min(n_bits, 23)) - 1)
            int_view = self.pos.view(np.int32)
            random_bits = perturb_rng.integers(
                0, int(mask32 + 1), size=self.pos.shape, dtype=np.int32)
            int_view[:] = (int_view & ~mask32) | random_bits

    def _fill_pref_array(self, arr, dist):
        """Fill a (N, K) preference array using the given distribution."""
        n, k, rng = self.n, self.k, self.rng
        dt = self._pref_dtype
        arr[:] = 0.0
        if dist == 1:  # Gaussian
            arr[:] = np.clip(rng.normal(0, 0.5, (n, k)), -1, 1).astype(dt)
        elif dist == 2:  # Sparse ±1
            if k < 2:
                arr[:, 0] = rng.choice([-1.0, 1.0], size=n).astype(dt)
            else:
                for i in range(n):
                    dims = rng.choice(k, size=2, replace=False)
                    arr[i, dims[0]] = 1.0
                    arr[i, dims[1]] = -1.0
        elif dist == 3:  # Unit Normalized
            raw = rng.normal(0, 1, (n, k))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr[:] = (raw / norms).astype(dt)
        elif dist == 4:  # Binary d0 + noise
            eps = params['binary_noise_eps']
            arr[:, 0] = 1.0
            arr[n // 2:, 0] = -1.0
            if k > 1:
                arr[:, 1:] = rng.uniform(-eps, eps, (n, k - 1)).astype(dt)
        else:  # 0 = Uniform [-1,1]
            arr[:] = rng.uniform(-1, 1, (n, k)).astype(dt)

    def _init_preferences(self, dist):
        self._fill_pref_array(self.prefs, dist)

    def _init_response(self, dist):
        self.response = np.zeros((self.n, self.k), dtype=self._pref_dtype)
        self._fill_pref_array(self.response, dist)

    def _normalize_prefs(self):
        """Normalize each particle's preference vector to unit length."""
        norms = np.linalg.norm(self.prefs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.prefs[:] = (self.prefs / norms).astype(self._pref_dtype)

    def _truncate_pref_bits(self):
        """Truncate preference mantissa to N bits."""
        dt = self._pref_dtype
        if dt == np.float16:
            max_bits = 10
        elif dt == np.float64:
            max_bits = 52
        else:
            max_bits = 23
        bits = max(1, min(max_bits, params['pref_mantissa_bits']))
        drop = max_bits - bits
        if drop > 0:
            # Work in the native float/int size matching the dtype
            if dt == np.float16:
                self.prefs = (self.prefs.astype(np.float32).view(np.int32) >> (23 - bits) << (23 - bits)).view(np.float32).astype(np.float16)
                if params['use_signal_response']:
                    self.response = (self.response.astype(np.float32).view(np.int32) >> (23 - bits) << (23 - bits)).view(np.float32).astype(np.float16)
            elif dt == np.float64:
                self.prefs = (self.prefs.view(np.int64) >> drop << drop).view(np.float64)
                if params['use_signal_response']:
                    self.response = (self.response.view(np.int64) >> drop << drop).view(np.float64)
            else:
                self.prefs = (self.prefs.view(np.int32) >> drop << drop).view(np.float32)
                if params['use_signal_response']:
                    self.response = (self.response.view(np.int32) >> drop << drop).view(np.float32)

    def _quantize_prefs(self):
        """Quantize preferences to discrete levels in [-1, 1].

        levels=2: binary {-1, +1} via sign()
        levels=N (N≥3): N+1 evenly spaced values from -1 to 1
        """
        dt = self._pref_dtype
        n_levels = max(2, params['pref_quant_levels'])
        if n_levels == 2:
            self.prefs = np.where(self.prefs >= 0, 1.0, -1.0).astype(dt)
            if params['use_signal_response']:
                self.response = np.where(self.response >= 0, 1.0, -1.0).astype(dt)
        else:
            steps = n_levels - 1
            self.prefs = np.clip(
                np.round((self.prefs + 1.0) * 0.5 * steps) / steps * 2.0 - 1.0,
                -1, 1
            ).astype(dt)
            if params['use_signal_response']:
                self.response = np.clip(
                    np.round((self.response + 1.0) * 0.5 * steps) / steps * 2.0 - 1.0,
                    -1, 1
                ).astype(dt)

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
        # ── Memory field: read (modulate preferences before physics) ──
        prefs_backup = None
        resp_backup = None
        modulated_prefs_pre = None
        modulated_resp_pre = None
        if params['memory_field']:
            prefs_backup = self.prefs.copy()
            strength = params['memory_strength']
            G = self.memory_field.shape[0]
            inv_cell = G / SPACE
            cx = (self.pos[:, 0] * inv_cell).astype(int) % G
            cy = (self.pos[:, 1] * inv_cell).astype(int) % G
            field_at_particle = self.memory_field[cy, cx]  # (N, K)
            # Multiplicative gate: amplify/dampen preferences
            self.prefs = (self.prefs * (1.0 + strength * field_at_particle)
                          ).clip(-1, 1).astype(self._pref_dtype)
            modulated_prefs_pre = self.prefs.copy()  # save for delta computation
            if params['use_signal_response']:
                resp_backup = self.response.copy()
                self.response = (self.response * (1.0 + strength * field_at_particle)
                                 ).clip(-1, 1).astype(self._pref_dtype)
                modulated_resp_pre = self.response.copy()

        # Swap arrays so physics sees response-as-signal and vice versa
        swapped = params['use_signal_response'] and params['swap_signal_response']
        if swapped:
            self.prefs, self.response = self.response, self.prefs
        self._step_impl(reuse_neighbors)
        if swapped:
            self.prefs, self.response = self.response, self.prefs

        # ── Memory field: capture social delta, restore, write, decay ──
        if params['memory_field'] and prefs_backup is not None:
            # Delta = kernel output - pre-physics modulated input
            # This isolates what the kernel's social learning changed
            social_delta = self.prefs.astype(np.float64) - modulated_prefs_pre.astype(np.float64)

            # Restore original prefs + apply only the social delta
            self.prefs = np.clip(
                prefs_backup.astype(np.float64) + social_delta, -1, 1
            ).astype(self._pref_dtype)

            if resp_backup is not None:
                resp_delta = self.response.astype(np.float64) - modulated_resp_pre.astype(np.float64)
                self.response = np.clip(
                    resp_backup.astype(np.float64) + resp_delta, -1, 1
                ).astype(self._pref_dtype)

            # Write: deposit particle preferences into the field
            write_rate = params['memory_write_rate']
            G = self.memory_field.shape[0]
            inv_cell = G / SPACE
            cx = (self.pos[:, 0] * inv_cell).astype(int) % G
            cy = (self.pos[:, 1] * inv_cell).astype(int) % G
            # Accumulate preferences into grid cells
            k = self.k
            for d in range(k):
                np.add.at(self.memory_field[:, :, d], (cy, cx),
                          write_rate * self.prefs[:, d].astype(np.float64))

            # Decay
            self.memory_field *= params['memory_decay']

            # Blur (diffuse the field spatially)
            if params['memory_blur'] and params['memory_blur_sigma'] > 0:
                from scipy.ndimage import gaussian_filter
                for d in range(k):
                    self.memory_field[:, :, d] = gaussian_filter(
                        self.memory_field[:, :, d],
                        sigma=params['memory_blur_sigma'], mode='wrap')
        if params['social_mode'] == 1 and params['social'] != 0:
            self._quiet_dim_social()
        # Response social learning (same rate as signal, applied post-step)
        if params['use_signal_response'] and params['social'] != 0 and params['social_mode'] == 0:
            self._response_social()
        # Quantize positions to grid_res × grid_res (non-grid engines only)
        if params['quantize_pos'] and params['physics_engine'] not in (3, 4, 5):
            G = params['grid_res']
            L = SPACE
            self.pos = (np.floor(self.pos * G / L) + 0.5) * L / G
            self.pos = self.pos % L
        # Truncate position mantissa to N bits
        if params['truncate_pos_bits']:
            bits = max(1, min(52, params['pos_mantissa_bits']))
            # Zero out the lowest (52 - bits) mantissa bits
            drop = 52 - bits
            if drop > 0:
                self.pos = (self.pos.view(np.int64) >> drop << drop).view(np.float64)
        # Truncate preference mantissa to N bits (float32 has 23-bit mantissa)
        if params['truncate_pref_bits']:
            self._truncate_pref_bits()
        # Quantize preferences to discrete levels in [-1, 1]
        if params['quantize_pref']:
            self._quantize_prefs()
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

        # ── Grid field engine (no neighbor finding needed) ──
        if params['physics_engine'] == 3:
            _tp0 = time.perf_counter()
            new_pos, new_prefs, mov = step_grid_field(
                pos, prefs, resp, SPACE, k, step_size, repulsion,
                social, params['grid_res'], params['grid_sigma'])
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs.astype(self._pref_dtype)
            self._movement = mov.astype(self._movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            self._t_build = 0.0
            self._t_query = 0.0
            self._t_search = 0.0
            self._n_nbrs = 0
            self.step_count += 1
            return

        # ── Grid Max Field engine (no neighbor finding needed) ──
        if params['physics_engine'] == 4:
            _tp0 = time.perf_counter()
            new_pos, new_prefs, mov = step_grid_max_field(
                pos, prefs, resp, SPACE, k, step_size, repulsion,
                social, params['grid_res'], params['grid_max_spread'])
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs.astype(self._pref_dtype)
            self._movement = mov.astype(self._movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            # Per-step timing breakdown and grid data for visualization
            if hasattr(step_grid_max_field, '_timing'):
                self._t_deposit, self._t_propagate, self._t_movement, self._t_social_grid = \
                    step_grid_max_field._timing
            if hasattr(step_grid_max_field, '_max_pref'):
                self._grid_max_pref = step_grid_max_field._max_pref
            self._t_build = 0.0
            self._t_query = 0.0
            self._t_search = 0.0
            self._n_nbrs = 0
            self.step_count += 1
            return

        # ── Grid Max Field GPU engine (PyTorch MPS/CUDA) ──
        if params['physics_engine'] == 5:
            _tp0 = time.perf_counter()
            new_pos, new_prefs, mov = step_grid_max_field_gpu(
                pos, prefs, resp, SPACE, k, step_size,
                social, params['grid_res'], params['grid_max_spread'],
                circular=params['grid_circular'])
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs.astype(self._pref_dtype)
            self._movement = mov.astype(self._movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            if hasattr(step_grid_max_field_gpu, '_timing'):
                self._t_deposit, self._t_propagate, self._t_movement, self._t_social_grid = \
                    step_grid_max_field_gpu._timing
            if hasattr(step_grid_max_field_gpu, '_max_pref'):
                self._grid_max_pref = step_grid_max_field_gpu._max_pref
            self._t_build = 0.0
            self._t_query = 0.0
            self._t_search = 0.0
            self._n_nbrs = 0
            self.step_count += 1
            return

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
            self.prefs = new_prefs.astype(self._pref_dtype)
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
            self.prefs = new_prefs.astype(self._pref_dtype)
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
            self.prefs = new_prefs.astype(self._pref_dtype)
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

        # Per-particle step scaling (engineer role)
        if params.get('use_particle_roles', False):
            scaled_movement = movement * self.role_step_scale[:, None]
        else:
            scaled_movement = movement
        self.pos = (pos + step_size * scaled_movement) % SPACE

        if social != 0:
            nbr_prefs = prefs[nbr_ids]
            # Per-particle influence weighting (leader role)
            use_roles = params.get('use_particle_roles', False)
            if use_roles and self.role_influence is not None:
                # Weight each neighbor's contribution by their influence
                inf_weights = self.role_influence[nbr_ids]  # (N, n_nbr)
                if params['social_dist_weight']:
                    d = dists[:, :, 0]
                    w = inf_weights / (d + 1e-6)
                else:
                    w = inf_weights.copy()
                if has_mask:
                    w = w * valid
                w_sum = w.sum(axis=1, keepdims=True)
                w /= np.maximum(w_sum, 1e-10)
                nbr_mean = (nbr_prefs * w[:, :, None]).sum(axis=1)
            elif params['social_dist_weight']:
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

    # ── Grid visualization ────────────────────────────────────────

    def compute_max_pref_grid(self):
        """Build the max-pref grid from current particle state.

        Used for visualization when not running a grid-based engine.
        Uses the same deposit + propagate as Grid Max CPU (engine 4).
        """
        from .physics_grid import _deposit_max, _propagate_max_8

        G = params['grid_res']
        spread = params['grid_max_spread']
        k = self.k
        pos_f64 = self.pos.astype(np.float64)

        src = self.get_vis_prefs()
        prefs_f64 = src.astype(np.float64)

        max_pref = np.full((G, G, k), -np.inf, dtype=np.float64)
        max_pos = np.zeros((G, G, k, 2), dtype=np.float64)
        _deposit_max(pos_f64, prefs_f64, max_pref, max_pos, G, SPACE, k)

        for _ in range(spread):
            max_pref, max_pos = _propagate_max_8(max_pref, max_pos, G, k)

        self._grid_max_pref = max_pref

    def _find_neighbors_for_probes(self, probe_pos):
        """Find neighbors for arbitrary query points using the same method as the sim.

        Uses the current knn_method, neighbor_mode, n_neighbors, neighbor_radius.

        Args:
            probe_pos: (M, 2) query positions

        Returns:
            nbr_ids: (M, n_nbr) neighbor indices into self.pos
            valid:   (M, n_nbr) bool mask or None
        """
        from scipy.spatial import cKDTree
        from .spatial import grid_build, _query_knn, _count_radius, _query_radius, _sort_radius_nbrs

        pos = self.pos
        n = len(pos)
        n_nbr = min(params['n_neighbors'], n - 1)
        radius = params['neighbor_radius']
        nbr_mode = params['neighbor_mode']
        knn_method = params['knn_method']
        m = len(probe_pos)

        # Build tree/grid from real particles, query from probe positions
        if knn_method == 0:
            # Hash grid — build from particles, but query_knn only supports
            # self-query.  Fall back to cKDTree for probe queries.
            query_pos = pos.astype(np.float64) % SPACE
        elif knn_method == 2:
            query_pos = pos.astype(np.float32).astype(np.float64) % SPACE
        else:
            query_pos = pos.astype(np.float64) % SPACE

        tree = cKDTree(query_pos, boxsize=SPACE)
        probe_f64 = probe_pos.astype(np.float64) % SPACE

        if nbr_mode == 2:
            # Radius only — probes are NOT in the tree, so no self to skip
            counts = tree.query_ball_point(probe_f64, radius, workers=-1,
                                           return_length=True)
            max_k = int(counts.max()) if len(counts) > 0 else 1
            max_k = min(max(max_k, 1), n - 1)
            if max_k < 1:
                return np.zeros((m, 1), dtype=np.int64), np.zeros((m, 1), dtype=bool)
            dists_raw, nbr_ids = tree.query(probe_f64, k=max_k, workers=-1)
            if dists_raw.ndim == 1:
                dists_raw = dists_raw[:, None]
                nbr_ids = nbr_ids[:, None]
            valid = dists_raw <= radius
            return nbr_ids.astype(np.int64), valid
        else:
            # KNN or KNN+Radius — probes not in tree, query exactly n_nbr
            _, nbr_ids = tree.query(probe_f64, k=n_nbr, workers=-1)
            if nbr_ids.ndim == 1:
                nbr_ids = nbr_ids[:, None]
            if nbr_mode == 1:
                # KNN + Radius: compute valid mask
                nbr_pos = pos[nbr_ids]
                toward = nbr_pos - probe_f64[:, None, :]
                toward -= SPACE * np.round(toward / SPACE)
                dists = np.linalg.norm(toward, axis=2)
                valid = dists <= radius
                return nbr_ids, valid
            return nbr_ids, None

    def compute_force_landscape_grid(self):
        """Compute force landscape by probing grid centers against real particles.

        Places a virtual probe at each grid cell center, finds its real
        neighbors using the same method as the simulation, and computes
        the per-dim max signal, optimal preference, and max movement direction.
        """
        G = params['grid_res']
        k = self.k
        best_mode = params['best_mode']
        src = self.get_vis_prefs()

        # Build probe positions at grid cell centers
        cell_size = SPACE / G
        cx = (np.arange(G) + 0.5) * cell_size
        cy = (np.arange(G) + 0.5) * cell_size
        gx, gy = np.meshgrid(cx, cy)
        probe_pos = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)

        # Find neighbors using the sim's configured method
        nbr_ids, valid = self._find_neighbors_for_probes(probe_pos)

        # If valid mask exists, mask invalid neighbors by setting to particle 0
        # (the numba kernel skips based on pref values, not a mask)
        # For simplicity, when valid mask exists, set invalid entries' prefs
        # to -inf so they lose the per-dim max competition
        if valid is not None:
            # Create a modified prefs array with an extra "null" particle at the end
            null_prefs = np.full((1, k), -np.inf, dtype=np.float64)
            prefs_ext = np.vstack([src.astype(np.float64), null_prefs])
            null_idx = len(self.pos)
            nbr_ids_ext = nbr_ids.copy()
            nbr_ids_ext[~valid] = null_idx
        else:
            prefs_ext = src.astype(np.float64)
            nbr_ids_ext = nbr_ids

        pos_f64 = self.pos.astype(np.float64)
        # Extend particle positions with a dummy for null index
        if valid is not None:
            pos_ext = np.vstack([pos_f64, [[0.0, 0.0]]])
        else:
            pos_ext = pos_f64

        max_pref_at_probe, mag, pref_opt, direction = _force_landscape_from_nbrs(
            probe_pos, pos_ext, prefs_ext, nbr_ids_ext, G, SPACE, k, best_mode)

        self._grid_max_pref = max_pref_at_probe  # (G, G, K) for MaxGrid view too
        self._force_mag = mag          # (G, G)
        self._force_pref = pref_opt    # (G, G, K)
        self._force_dir = direction    # (G, G, 2)

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
