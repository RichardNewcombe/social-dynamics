"""
Numba JIT physics kernels for the 2D particle simulation.

Two parallel kernels:
  _step_inner_prod_avg — inner-product-average mode
  _step_per_dim        — per-dimension best-neighbor mode

Both accept separate signal (prefs) and response arrays.
When signal/response mode is off, pass prefs for both.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _step_inner_prod_avg(pos, prefs, response, nbr_ids, valid, L, k,
                         step_size, repulsion, social, social_dist_weight,
                         pref_dist_weight, pref_dist_sigma):
    """Inner-product average physics step.

    Movement = mean over neighbors of (inner_product * unit_direction).
    Inner product uses response[i] dot signal[j] (prefs = signal).

    Returns:
        new_pos:   (N, 2) float64
        new_prefs: (N, K) float64  (updated signal)
        movement:  (N, 2) float64
    """
    n = pos.shape[0]
    n_nbr = nbr_ids.shape[1]
    new_pos = np.empty((n, 2), dtype=np.float64)
    new_prefs = np.empty((n, k), dtype=np.float64)
    movement = np.empty((n, 2), dtype=np.float64)

    for i in prange(n):
        mx, my = 0.0, 0.0
        rx, ry = 0.0, 0.0
        count = 0

        for j in range(n_nbr):
            if not valid[i, j]:
                continue
            nj = nbr_ids[i, j]
            dx = pos[nj, 0] - pos[i, 0]
            dy = pos[nj, 1] - pos[i, 1]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-12:
                continue
            inv_dist = 1.0 / dist
            ux = dx * inv_dist
            uy = dy * inv_dist
            ip = 0.0
            for d in range(k):
                ip += response[i, d] * prefs[nj, d]
            ip /= k
            if pref_dist_weight:
                gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                mx += ip * gw * ux
                my += ip * gw * uy
            else:
                mx += ip * ux
                my += ip * uy
            rx -= ux * inv_dist
            ry -= uy * inv_dist
            count += 1

        if count > 0:
            inv_c = 1.0 / count
            movement[i, 0] = mx * inv_c + repulsion * rx * inv_c
            movement[i, 1] = my * inv_c + repulsion * ry * inv_c
        else:
            movement[i, 0] = 0.0
            movement[i, 1] = 0.0

        new_pos[i, 0] = (pos[i, 0] + step_size * movement[i, 0]) % L
        new_pos[i, 1] = (pos[i, 1] + step_size * movement[i, 1]) % L

        # Social learning updates signal (prefs) only
        if social != 0.0:
            if social_dist_weight:
                w_total = 0.0
                for d in range(k):
                    new_prefs[i, d] = 0.0
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    ddx = pos[nj, 0] - pos[i, 0]
                    ddy = pos[nj, 1] - pos[i, 1]
                    ddx -= L * round(ddx / L)
                    ddy -= L * round(ddy / L)
                    dd = (ddx * ddx + ddy * ddy) ** 0.5
                    w = 1.0 / (dd + 1e-6)
                    w_total += w
                    for d in range(k):
                        new_prefs[i, d] += w * prefs[nj, d]
                if w_total > 1e-10:
                    for d in range(k):
                        new_prefs[i, d] /= w_total
                        new_prefs[i, d] = (1.0 - social) * prefs[i, d] + social * new_prefs[i, d]
                        new_prefs[i, d] = min(1.0, max(-1.0, new_prefs[i, d]))
                else:
                    for d in range(k):
                        new_prefs[i, d] = prefs[i, d]
            else:
                for d in range(k):
                    s = 0.0
                    cnt = 0
                    for j in range(n_nbr):
                        if valid[i, j]:
                            s += prefs[nbr_ids[i, j], d]
                            cnt += 1
                    if cnt > 0:
                        mean_p = s / cnt
                        v = (1.0 - social) * prefs[i, d] + social * mean_p
                        new_prefs[i, d] = min(1.0, max(-1.0, v))
                    else:
                        new_prefs[i, d] = prefs[i, d]
        else:
            for d in range(k):
                new_prefs[i, d] = prefs[i, d]

    return new_pos, new_prefs, movement


@njit(parallel=True)
def _step_per_dim(pos, prefs, response, dir_matrix, nbr_ids, valid, L, k,
                  step_size, repulsion, social, social_dist_weight,
                  dir_memory, pref_weighted, pref_inner,
                  pref_dist_weight, pref_dist_sigma, best_mode,
                  boltzmann_beta=5.0, ignore_self_pref=False,
                  normalize_direction=True):
    """Per-dimension physics step.

    Neighbor selection uses prefs (signal).
    Compatibility weighting uses response[i] * signal[j*].

    Returns:
        new_pos:   (N, 2) float64
        new_prefs: (N, K) float64  (updated signal)
        new_dm:    (N, K, 2) float64 direction memory
        movement:  (N, 2) float64
    """
    n = pos.shape[0]
    n_nbr = nbr_ids.shape[1]
    new_pos = np.empty((n, 2), dtype=np.float64)
    new_prefs = np.empty((n, k), dtype=np.float64)
    new_dm = np.empty((n, k, 2), dtype=np.float64)
    movement = np.empty((n, 2), dtype=np.float64)

    for i in prange(n):
        mx, my = 0.0, 0.0

        for ki in range(k):
            if pref_weighted:
                wx, wy = 0.0, 0.0
                wc = 0
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    dx = pos[nj, 0] - pos[i, 0]
                    dy = pos[nj, 1] - pos[i, 1]
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    dist = (dx * dx + dy * dy) ** 0.5
                    wc += 1
                    if dist < 1e-12:
                        continue
                    if normalize_direction:
                        ux = dx / dist
                        uy = dy / dist
                    else:
                        ux = dx
                        uy = dy
                    w = prefs[nj, ki]  # neighbor signal
                    if pref_dist_weight:
                        gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                        w *= gw
                    wx += w * ux
                    wy += w * uy
                if wc > 0:
                    wx /= wc
                    wy /= wc
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * wx
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * wy
                self_w = 1.0 if ignore_self_pref else response[i, ki]
                mx += self_w * new_dm[i, ki, 0]
                my += self_w * new_dm[i, ki, 1]
            elif best_mode == 3:
                # Boltzmann (corrected): separates direction from signal weighting
                # Converges exactly to hard argmax as β→∞:
                #   movement[d] = response[i,d] * signal[j*,d] * dir_to(j*)
                beta = boltzmann_beta
                # Two-pass for numerical stability
                max_score = -1e30
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    sc = beta * prefs[nj, ki]
                    if sc > max_score:
                        max_score = sc
                if max_score < -1e29:
                    new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0]
                    new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1]
                    continue
                # Second pass: accumulate
                w_sum = 0.0
                dx_sum, dy_sum = 0.0, 0.0  # pure direction average
                sig_sum = 0.0               # weighted signal average
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    w = np.exp(beta * prefs[nj, ki] - max_score)
                    if pref_dist_weight:
                        ddx = pos[nj, 0] - pos[i, 0]
                        ddy = pos[nj, 1] - pos[i, 1]
                        ddx -= L * round(ddx / L)
                        ddy -= L * round(ddy / L)
                        dd = (ddx * ddx + ddy * ddy) ** 0.5
                        gw = np.exp(-dd * dd / (2.0 * pref_dist_sigma * pref_dist_sigma))
                        w *= gw
                    w_sum += w
                    sig_sum += w * prefs[nj, ki]
                    # Direction (pure, no signal multiplication)
                    dx = pos[nj, 0] - pos[i, 0]
                    dy = pos[nj, 1] - pos[i, 1]
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > 1e-12:
                        if normalize_direction:
                            dx_sum += w * dx / dist
                            dy_sum += w * dy / dist
                        else:
                            dx_sum += w * dx
                            dy_sum += w * dy
                if w_sum > 1e-30:
                    dx_sum /= w_sum
                    dy_sum /= w_sum
                    sig_sum /= w_sum
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * dx_sum
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * dy_sum
                self_w = 1.0 if ignore_self_pref else response[i, ki]
                compat = self_w * sig_sum
                if pref_inner:
                    ip = 0.0
                    for di in range(k):
                        ip += response[i, di] * prefs[i, di]
                    ip /= k
                    compat *= ip
                mx += compat * new_dm[i, ki, 0]
                my += compat * new_dm[i, ki, 1]
            else:
                best_val = -1e30
                best_nj = -1
                # Selection uses signal (prefs) — both own sign check and neighbor score
                my_sign = 1.0 if prefs[i, ki] >= 0.0 else -1.0
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    if best_mode == 2:
                        nj_sign = 1.0 if prefs[nj, ki] >= 0.0 else -1.0
                        if nj_sign != my_sign:
                            continue
                        score = abs(prefs[nj, ki])
                    elif best_mode == 1:
                        score = abs(prefs[nj, ki])
                    else:
                        score = prefs[nj, ki]
                    if score > best_val:
                        best_val = score
                        best_nj = nj
                if best_nj < 0:
                    new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0]
                    new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1]
                    continue
                dx = pos[best_nj, 0] - pos[i, 0]
                dy = pos[best_nj, 1] - pos[i, 1]
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-12:
                    ux, uy = 0.0, 0.0
                elif normalize_direction:
                    ux = dx / dist
                    uy = dy / dist
                else:
                    ux = dx
                    uy = dy
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * ux
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * uy
                # Compatibility: response[i] * signal[j*]
                self_w = 1.0 if ignore_self_pref else response[i, ki]
                compat = self_w * prefs[best_nj, ki]
                if pref_inner:
                    ip = 0.0
                    for di in range(k):
                        ip += response[i, di] * prefs[best_nj, di]
                    ip /= k
                    compat *= ip
                if pref_dist_weight:
                    gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                    compat *= gw
                mx += compat * new_dm[i, ki, 0]
                my += compat * new_dm[i, ki, 1]

        rx, ry = 0.0, 0.0
        rc = 0
        for j in range(n_nbr):
            if not valid[i, j]:
                continue
            nj = nbr_ids[i, j]
            dx = pos[nj, 0] - pos[i, 0]
            dy = pos[nj, 1] - pos[i, 1]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dist = (dx * dx + dy * dy) ** 0.5
            rc += 1
            if dist < 1e-6:
                continue
            inv_d = 1.0 / dist
            rx -= (dx * inv_d) * inv_d
            ry -= (dy * inv_d) * inv_d

        if rc > 0:
            movement[i, 0] = mx + repulsion * rx / rc
            movement[i, 1] = my + repulsion * ry / rc
        else:
            movement[i, 0] = mx
            movement[i, 1] = my

        new_pos[i, 0] = (pos[i, 0] + step_size * movement[i, 0]) % L
        new_pos[i, 1] = (pos[i, 1] + step_size * movement[i, 1]) % L

        # Social learning updates signal (prefs) only
        if social != 0.0:
            if social_dist_weight:
                w_total = 0.0
                for d in range(k):
                    new_prefs[i, d] = 0.0
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    ddx = pos[nj, 0] - pos[i, 0]
                    ddy = pos[nj, 1] - pos[i, 1]
                    ddx -= L * round(ddx / L)
                    ddy -= L * round(ddy / L)
                    dd = (ddx * ddx + ddy * ddy) ** 0.5
                    w = 1.0 / (dd + 1e-6)
                    w_total += w
                    for d in range(k):
                        new_prefs[i, d] += w * prefs[nj, d]
                if w_total > 1e-10:
                    for d in range(k):
                        new_prefs[i, d] /= w_total
                        new_prefs[i, d] = (1.0 - social) * prefs[i, d] + social * new_prefs[i, d]
                        new_prefs[i, d] = min(1.0, max(-1.0, new_prefs[i, d]))
                else:
                    for d in range(k):
                        new_prefs[i, d] = prefs[i, d]
            else:
                for d in range(k):
                    s = 0.0
                    cnt = 0
                    for j in range(n_nbr):
                        if valid[i, j]:
                            s += prefs[nbr_ids[i, j], d]
                            cnt += 1
                    if cnt > 0:
                        mean_p = s / cnt
                        v = (1.0 - social) * prefs[i, d] + social * mean_p
                        new_prefs[i, d] = min(1.0, max(-1.0, v))
                    else:
                        new_prefs[i, d] = prefs[i, d]
        else:
            for d in range(k):
                new_prefs[i, d] = prefs[i, d]

    return new_pos, new_prefs, new_dm, movement


def warmup_numba_physics():
    """Trigger JIT compilation of physics kernels on dummy data."""
    n = 100
    pos = np.random.rand(n, 2).astype(np.float64)
    prefs = np.random.rand(n, 3).astype(np.float64)
    resp = np.random.rand(n, 3).astype(np.float64)
    dm = np.zeros((n, 3, 2), dtype=np.float64)
    nbr = np.zeros((n, 5), dtype=np.int64)
    valid = np.ones((n, 5), dtype=np.bool_)
    _step_inner_prod_avg(pos, prefs, resp, nbr, valid, 1.0, 3,
                         0.005, 0.0, 0.0, False, False, 0.01)
    _step_per_dim(pos, prefs, resp, dm, nbr, valid, 1.0, 3,
                  0.005, 0.0, 0.0, False, 0.0, False, False,
                  False, 0.01, 0, 5.0, False)
    # Also warmup Boltzmann mode
    _step_per_dim(pos, prefs, resp, dm, nbr, valid, 1.0, 3,
                  0.005, 0.0, 0.0, False, 0.0, False, False,
                  False, 0.01, 3, 5.0, False)
