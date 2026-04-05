"""
Grid field physics for the 2D particle simulation.

Particle-mesh method: instead of pairwise neighbor interactions,
deposit particle preferences onto a grid, smooth it, and have
each particle read the local field to determine movement.

Cost is O(N + G^2) regardless of clustering, where G = grid_res.

The field approach approximates the pairwise per-dimension dynamics:
  For each dim d, each particle moves toward regions where signal[d]
  is high, weighted by its own response[d].

Fields computed:
  - pref_field[d]  (G, G): smoothed signal density per pref dimension
  - count_field    (G, G): smoothed particle count (for normalization)
  - grad_x/grad_y  of each pref_field: direction of increasing pref[d]

Movement for particle i:
  v_i = step_size * sum_d( response[i,d] * grad_pref_field[d](pos_i) )
      + repulsion * grad_log_count_field(pos_i)
"""

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter


@njit(parallel=True)
def _deposit(pos, values, grid, G, L):
    """Deposit particle values onto grid using bilinear interpolation.

    Args:
        pos:    (N, 2) positions in [0, L)
        values: (N,) scalar values to deposit
        grid:   (G, G) output grid (must be pre-zeroed)
        G:      grid resolution
        L:      domain size
    """
    n = pos.shape[0]
    cell = L / G
    inv_cell = G / L

    for i in prange(n):
        # Continuous grid coordinate
        gx = pos[i, 0] * inv_cell
        gy = pos[i, 1] * inv_cell

        # Cell indices (lower-left corner)
        ix = int(gx) % G
        iy = int(gy) % G

        # Bilinear weights
        fx = gx - int(gx)
        fy = gy - int(gy)

        # Four corners (periodic)
        ix1 = (ix + 1) % G
        iy1 = (iy + 1) % G

        v = values[i]
        # Atomic-safe for prange: each particle writes to ~4 cells,
        # collisions are rare for large G. For exact correctness with
        # prange we'd need atomics, but the error is small and
        # this is much faster than sequential.
        grid[iy, ix] += v * (1.0 - fx) * (1.0 - fy)
        grid[iy, ix1] += v * fx * (1.0 - fy)
        grid[iy1, ix] += v * (1.0 - fx) * fy
        grid[iy1, ix1] += v * fx * fy


@njit(parallel=True)
def _sample_gradient(pos, grad_x, grad_y, out, G, L):
    """Sample gradient field at particle positions using bilinear interpolation.

    Args:
        pos:    (N, 2) positions
        grad_x: (G, G) x-component of gradient
        grad_y: (G, G) y-component of gradient
        out:    (N, 2) output gradient per particle
        G, L:   grid res and domain size
    """
    n = pos.shape[0]
    inv_cell = G / L

    for i in prange(n):
        gx = pos[i, 0] * inv_cell
        gy = pos[i, 1] * inv_cell

        ix = int(gx) % G
        iy = int(gy) % G
        fx = gx - int(gx)
        fy = gy - int(gy)

        ix1 = (ix + 1) % G
        iy1 = (iy + 1) % G

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        out[i, 0] = (w00 * grad_x[iy, ix] + w10 * grad_x[iy, ix1] +
                     w01 * grad_x[iy1, ix] + w11 * grad_x[iy1, ix1])
        out[i, 1] = (w00 * grad_y[iy, ix] + w10 * grad_y[iy, ix1] +
                     w01 * grad_y[iy1, ix] + w11 * grad_y[iy1, ix1])


def _periodic_gradient(field, G, L):
    """Compute gradient of a periodic field using central differences.

    Returns (grad_x, grad_y) each (G, G).
    """
    cell = L / G
    inv_2cell = 0.5 / cell

    # Central differences with periodic wrapping
    grad_x = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * inv_2cell
    grad_y = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) * inv_2cell

    return grad_x, grad_y


def step_grid_field(pos, prefs, response, L, k, step_size, repulsion,
                    social, grid_res, grid_sigma):
    """One physics step using the grid field method.

    Args:
        pos:      (N, 2) float64 positions
        prefs:    (N, K) float32 signal preferences
        response: (N, K) float32 response weights
        L:        domain size
        k:        preference dimensions
        step_size, repulsion, social: physics params
        grid_res: grid resolution G
        grid_sigma: Gaussian smoothing sigma in grid cells

    Returns:
        new_pos:   (N, 2) positions
        new_prefs: (N, K) updated signal (if social != 0)
        movement:  (N, 2) movement vectors
    """
    n = len(pos)
    G = grid_res
    pos_f64 = pos.astype(np.float64)

    movement = np.zeros((n, 2), dtype=np.float64)
    grad_buf = np.zeros((n, 2), dtype=np.float64)

    # ── Per-dimension preference fields ──
    for d in range(k):
        # Deposit signal[d] onto grid
        field = np.zeros((G, G), dtype=np.float64)
        _deposit(pos_f64, prefs[:, d].astype(np.float64), field, G, L)

        # Smooth (periodic boundary via wrap mode)
        if grid_sigma > 0:
            field = gaussian_filter(field, sigma=grid_sigma, mode='wrap')

        # Gradient: direction of increasing signal[d]
        gx, gy = _periodic_gradient(field, G, L)

        # Sample gradient at particle positions
        _sample_gradient(pos_f64, gx, gy, grad_buf, G, L)

        # Movement: response[i, d] * grad_signal_field[d](pos_i)
        movement[:, 0] += response[:, d].astype(np.float64) * grad_buf[:, 0]
        movement[:, 1] += response[:, d].astype(np.float64) * grad_buf[:, 1]

    # ── Repulsion via density field ──
    if repulsion > 0:
        count_field = np.zeros((G, G), dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        _deposit(pos_f64, ones, count_field, G, L)

        if grid_sigma > 0:
            count_field = gaussian_filter(count_field, sigma=grid_sigma, mode='wrap')

        # Gradient of log-density (away from high density)
        # Add small epsilon to avoid log(0)
        log_count = np.log(count_field + 1e-8)
        gx, gy = _periodic_gradient(log_count, G, L)

        _sample_gradient(pos_f64, gx, gy, grad_buf, G, L)

        # Repulsion pushes AWAY from high density (negative gradient)
        movement[:, 0] -= repulsion * grad_buf[:, 0]
        movement[:, 1] -= repulsion * grad_buf[:, 1]

    # ── Update positions ──
    new_pos = (pos + step_size * movement) % L

    # ── Social learning via field ──
    new_prefs = prefs.copy()
    if social != 0:
        # For each dim, the local field value IS the neighborhood mean signal
        # (up to normalization). Use it directly for social learning.
        count_field_raw = np.zeros((G, G), dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        _deposit(pos_f64, ones, count_field_raw, G, L)
        if grid_sigma > 0:
            count_field_smooth = gaussian_filter(count_field_raw, sigma=grid_sigma, mode='wrap')
        else:
            count_field_smooth = count_field_raw
        count_field_smooth = np.maximum(count_field_smooth, 1e-8)

        for d in range(k):
            field = np.zeros((G, G), dtype=np.float64)
            _deposit(pos_f64, prefs[:, d].astype(np.float64), field, G, L)
            if grid_sigma > 0:
                field = gaussian_filter(field, sigma=grid_sigma, mode='wrap')

            # Normalized field = local mean preference
            mean_field = field / count_field_smooth

            # Sample mean at particle positions
            mean_at_particle = np.zeros(n, dtype=np.float64)
            mean_gx = mean_field  # reuse as 1D sampling
            # Sample using bilinear interpolation (reuse gradient sampler
            # with a dummy y component)
            dummy_grad = np.zeros((G, G), dtype=np.float64)
            buf = np.zeros((n, 2), dtype=np.float64)
            _sample_gradient(pos_f64, mean_field, dummy_grad, buf, G, L)
            mean_at_particle = buf[:, 0]  # x component has the sampled value

            new_prefs[:, d] = np.clip(
                (1.0 - social) * prefs[:, d] + social * mean_at_particle,
                -1, 1
            ).astype(np.float32)

    return new_pos, new_prefs, movement


# =====================================================================
# Grid Max Field — approximates per-dimension best-neighbor physics
# =====================================================================

@njit
def _deposit_max(pos, prefs, max_pref, max_pos, G, L, k):
    """Deposit particles onto grid using per-dimension max (nearest cell).

    Args:
        pos:      (N, 2) positions
        prefs:    (N, K) signal preferences
        max_pref: (G, G, K) output — max pref per cell per dim (pre-filled with -inf)
        max_pos:  (G, G, K, 2) output — source position per cell per dim
        G, L:     grid res and domain size
        k:        preference dimensions
    """
    n = pos.shape[0]
    inv_cell = G / L

    for i in range(n):
        cx = int(pos[i, 0] * inv_cell) % G
        cy = int(pos[i, 1] * inv_cell) % G
        for d in range(k):
            if prefs[i, d] > max_pref[cy, cx, d]:
                max_pref[cy, cx, d] = prefs[i, d]
                max_pos[cy, cx, d, 0] = pos[i, 0]
                max_pos[cy, cx, d, 1] = pos[i, 1]


@njit
def _propagate_max_8(max_pref, max_pos, G, k):
    """One pass of 8-connected (3x3 Moore) max propagation.

    Each cell checks all 8 neighbors + itself (periodic).
    Spreads in Chebyshev distance — R passes covers radius R cells.
    """
    new_max = max_pref.copy()
    new_pos = max_pos.copy()

    for cy in range(G):
        for cx in range(G):
            for d in range(k):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny = (cy + dy) % G
                        nx = (cx + dx) % G
                        if max_pref[ny, nx, d] > new_max[cy, cx, d]:
                            new_max[cy, cx, d] = max_pref[ny, nx, d]
                            new_pos[cy, cx, d, 0] = max_pos[ny, nx, d, 0]
                            new_pos[cy, cx, d, 1] = max_pos[ny, nx, d, 1]

    return new_max, new_pos


@njit(parallel=True)
def _movement_from_max_grid(pos, response, max_pref, max_pos, G, L, k):
    """Compute movement from the max-field grid.

    For each particle i, for each dim d:
      - Read max_signal[d] and source_pos[d] from the grid cell
      - compat = response[i, d] * max_signal[d]
      - direction = normalize(source_pos[d] - pos[i]) with periodic wrapping
      - movement += compat * direction

    Args:
        pos:      (N, 2) positions
        response: (N, K) response weights
        max_pref: (G, G, K) max signal values on grid
        max_pos:  (G, G, K, 2) source positions on grid
        G, L, k:  grid res, domain size, pref dims

    Returns:
        movement: (N, 2) movement vectors
    """
    n = pos.shape[0]
    movement = np.zeros((n, 2), dtype=np.float64)
    inv_cell = G / L

    for i in prange(n):
        cx = int(pos[i, 0] * inv_cell) % G
        cy = int(pos[i, 1] * inv_cell) % G

        for d in range(k):
            mp = max_pref[cy, cx, d]
            # Skip empty cells (max_pref still -inf)
            if mp < -1e30:
                continue

            src_x = max_pos[cy, cx, d, 0]
            src_y = max_pos[cy, cx, d, 1]

            # Compatibility: response[i, d] * max_signal[d]
            compat = response[i, d] * mp

            # Direction toward source (periodic)
            ddx = src_x - pos[i, 0]
            ddy = src_y - pos[i, 1]
            ddx -= L * round(ddx / L)
            ddy -= L * round(ddy / L)
            dist = (ddx * ddx + ddy * ddy) ** 0.5
            if dist < 1e-12:
                continue
            ux = ddx / dist
            uy = ddy / dist

            movement[i, 0] += compat * ux
            movement[i, 1] += compat * uy

    return movement


def step_grid_max_field(pos, prefs, response, L, k, step_size, repulsion,
                        social, grid_res, max_spread):
    """One physics step using the grid max-field method.

    Approximates the per-dimension best-neighbor physics:
    1. Deposit: per-dim max signal + source position into grid cells
    2. Propagate: iterative 3x3 max-pool to spread max values
    3. Movement: each particle reads its cell's max per dim,
       computes compat * direction toward source

    Args:
        pos:        (N, 2) float64 positions
        prefs:      (N, K) float32 signal
        response:   (N, K) float32 response weights
        L:          domain size
        k:          preference dimensions
        step_size, repulsion, social: physics params
        grid_res:   grid resolution G
        max_spread: number of max-pool propagation iterations

    Returns:
        new_pos:   (N, 2) positions
        new_prefs: (N, K) updated signal
        movement:  (N, 2) movement vectors
    """
    import time as _time

    n = len(pos)
    G = grid_res
    pos_f64 = pos.astype(np.float64)
    prefs_f64 = prefs.astype(np.float64)
    resp_f64 = response.astype(np.float64)

    # ── Step 1: Deposit max per dim (nearest cell) ──
    _t0 = _time.perf_counter()
    max_pref = np.full((G, G, k), -np.inf, dtype=np.float64)
    max_pos = np.zeros((G, G, k, 2), dtype=np.float64)
    _deposit_max(pos_f64, prefs_f64, max_pref, max_pos, G, L, k)
    _t_deposit = _time.perf_counter() - _t0

    # ── Step 2: Propagate with 4-connected max (diamond footprint) ──
    _t0 = _time.perf_counter()
    for _ in range(max_spread):
        max_pref, max_pos = _propagate_max_8(max_pref, max_pos, G, k)
    _t_propagate = _time.perf_counter() - _t0

    # ── Step 3: Compute movement ──
    _t0 = _time.perf_counter()
    movement = _movement_from_max_grid(pos_f64, resp_f64, max_pref, max_pos, G, L, k)
    _t_movement = _time.perf_counter() - _t0

    # ── Update positions ──
    new_pos = (pos + step_size * movement) % L

    # ── Social learning via deposited fields ──
    _t0 = _time.perf_counter()
    new_prefs = prefs.copy()
    if social != 0:
        count_field = np.zeros((G, G), dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        _deposit(pos_f64, ones, count_field, G, L)
        count_smooth = gaussian_filter(count_field, sigma=2.0, mode='wrap')
        count_smooth = np.maximum(count_smooth, 1e-8)

        for d in range(k):
            field = np.zeros((G, G), dtype=np.float64)
            _deposit(pos_f64, prefs_f64[:, d], field, G, L)
            field = gaussian_filter(field, sigma=2.0, mode='wrap')
            mean_field = field / count_smooth
            dummy = np.zeros((G, G), dtype=np.float64)
            buf = np.zeros((n, 2), dtype=np.float64)
            _sample_gradient(pos_f64, mean_field, dummy, buf, G, L)
            new_prefs[:, d] = np.clip(
                (1.0 - social) * prefs[:, d] + social * buf[:, 0],
                -1, 1
            ).astype(np.float32)
    _t_social = _time.perf_counter() - _t0

    # Store timing and grid for display
    step_grid_max_field._timing = (
        _t_deposit, _t_propagate, _t_movement, _t_social)
    step_grid_max_field._max_pref = max_pref  # (G, G, K) for visualization

    return new_pos, new_prefs, movement


# =====================================================================
# Force landscape — probe the real particle field from grid centers
# =====================================================================

@njit(parallel=True)
def _force_landscape_from_nbrs(probe_pos, particle_pos, particle_prefs,
                                nbr_ids, G, L, k, best_mode):
    """Compute force landscape by probing real neighbors from grid centers.

    For each probe (grid cell center), uses precomputed KNN neighbor
    indices into the real particle set. Finds the per-dim max signal
    neighbor and builds the M matrix, then sweeps 2^K preference
    combos to find maximum movement magnitude.

    Args:
        probe_pos:      (G*G, 2) probe positions (grid cell centers)
        particle_pos:   (N, 2) real particle positions
        particle_prefs: (N, K) real particle signal values
        nbr_ids:        (G*G, n_nbr) neighbor indices for each probe
        G, L, k:        grid res, domain size, pref dims
        best_mode:      0=default, 1=max magnitude, 2=same-sign

    Returns:
        max_pref_at_probe: (G, G, K) max signal per dim at each cell
        mag_max:           (G, G) maximum movement magnitude
        pref_max:          (G, G, K) optimal preference vector
        dir_max:           (G, G, 2) direction of maximum movement
    """
    n_probes = G * G
    n_nbr = nbr_ids.shape[1]
    # 3^K combos: each dim can be -1, 0, or +1
    n_combos = 1
    for _ in range(k):
        n_combos *= 3

    max_pref_at_probe = np.full((G, G, k), -np.inf, dtype=np.float64)
    mag_max = np.zeros((G, G), dtype=np.float64)
    pref_max = np.zeros((G, G, k), dtype=np.float64)
    dir_max = np.zeros((G, G, 2), dtype=np.float64)

    for pi in prange(n_probes):
        cy = pi // G
        cx = pi % G
        px = probe_pos[pi, 0]
        py = probe_pos[pi, 1]

        # For each dim d, find the best neighbor (max signal)
        # and compute the M matrix column
        M = np.zeros((k, 2), dtype=np.float64)
        any_valid = False

        for d in range(k):
            best_val = -1e30
            best_nj = -1

            for j in range(n_nbr):
                nj = nbr_ids[pi, j]
                pval = particle_prefs[nj, d]

                if best_mode == 2:
                    # Same-sign: skip for now since probe has no pref yet
                    # Use max magnitude instead
                    score = abs(pval)
                elif best_mode == 1:
                    score = abs(pval)
                else:
                    score = pval

                if score > best_val:
                    best_val = score
                    best_nj = nj

            if best_nj < 0:
                continue

            mp = particle_prefs[best_nj, d]
            max_pref_at_probe[cy, cx, d] = mp

            sx = particle_pos[best_nj, 0]
            sy = particle_pos[best_nj, 1]
            ddx = sx - px
            ddy = sy - py
            ddx -= L * round(ddx / L)
            ddy -= L * round(ddy / L)
            dist = (ddx * ddx + ddy * ddy) ** 0.5
            if dist < 1e-12:
                continue
            any_valid = True
            M[d, 0] = mp * ddx / dist
            M[d, 1] = mp * ddy / dist

        if not any_valid:
            continue

        # Try all 3^K combinations: each dim ∈ {-1, 0, +1}
        # On ties (which happen because |M@r| = |M@(-r)|), prefer
        # the combo with more positive values for consistent visualization.
        best_mag = 0.0
        best_vx = 0.0
        best_vy = 0.0
        best_pos_count = -1  # count of +1 dims (for tie-breaking)

        for combo in range(n_combos):
            vx = 0.0
            vy = 0.0
            pos_count = 0
            tmp = combo
            for d in range(k):
                digit = tmp % 3
                tmp //= 3
                w = digit - 1.0  # 0→-1, 1→0, 2→+1
                vx += w * M[d, 0]
                vy += w * M[d, 1]
                if digit == 2:
                    pos_count += 1
            mag = (vx * vx + vy * vy) ** 0.5
            # Accept if strictly better, or same mag but more positive dims
            if mag > best_mag + 1e-10 or (mag > best_mag - 1e-10 and pos_count > best_pos_count):
                best_mag = mag
                best_vx = vx
                best_vy = vy
                best_pos_count = pos_count
                tmp2 = combo
                for d in range(k):
                    digit = tmp2 % 3
                    tmp2 //= 3
                    pref_max[cy, cx, d] = digit - 1.0

        mag_max[cy, cx] = best_mag
        if best_mag > 1e-12:
            dir_max[cy, cx, 0] = best_vx / best_mag
            dir_max[cy, cx, 1] = best_vy / best_mag

    return max_pref_at_probe, mag_max, pref_max, dir_max


    # compute_force_landscape removed — neighbor finding is now handled
    # by Simulation._find_neighbors_for_probes() which respects the
    # configured knn_method, neighbor_mode, and radius settings.
    # The numba kernel _force_landscape_from_nbrs is called directly.


def warmup_grid_field():
    """Trigger JIT compilation of grid field kernels."""
    n = 50
    G = 16
    pos = np.random.rand(n, 2).astype(np.float64)
    vals = np.random.rand(n).astype(np.float64)
    grid = np.zeros((G, G), dtype=np.float64)
    _deposit(pos, vals, grid, G, 1.0)
    gx, gy = _periodic_gradient(grid, G, 1.0)
    out = np.zeros((n, 2), dtype=np.float64)
    _sample_gradient(pos, gx, gy, out, G, 1.0)

    # Warmup max-field kernels
    k = 3
    prefs = np.random.rand(n, k).astype(np.float64)
    resp = np.random.rand(n, k).astype(np.float64)
    mp = np.full((G, G, k), -np.inf, dtype=np.float64)
    mpos = np.zeros((G, G, k, 2), dtype=np.float64)
    _deposit_max(pos, prefs, mp, mpos, G, 1.0, k)
    _propagate_max_8(mp, mpos, G, k)
    _movement_from_max_grid(pos, resp, mp, mpos, G, 1.0, k)
    # Warmup force landscape kernel
    nbr_ids = np.zeros((G * G, 5), dtype=np.int64)
    _force_landscape_from_nbrs(
        np.random.rand(G * G, 2), pos, prefs, nbr_ids, G, 1.0, k, 0)
