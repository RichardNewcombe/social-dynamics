"""
3D Numba Physics Kernels
========================

All position/direction vectors are (n, 3).
All distances use dx² + dy² + dz² with toroidal wrapping.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _step_inner_prod_avg(pos, prefs, nbr_ids, valid, L, k,
                         step_size, repulsion, social, social_dist_weight,
                         pref_dist_weight, pref_dist_sigma):
    n = pos.shape[0]
    n_nbr = nbr_ids.shape[1]
    new_pos = np.empty((n, 3), dtype=np.float64)
    new_prefs = np.empty((n, k), dtype=np.float64)
    movement = np.empty((n, 3), dtype=np.float64)

    for i in prange(n):
        mx, my, mz = 0.0, 0.0, 0.0
        rx, ry, rz = 0.0, 0.0, 0.0
        count = 0

        for j in range(n_nbr):
            if not valid[i, j]:
                continue
            nj = nbr_ids[i, j]
            dx = pos[nj, 0] - pos[i, 0]
            dy = pos[nj, 1] - pos[i, 1]
            dz = pos[nj, 2] - pos[i, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < 1e-12:
                continue
            inv_dist = 1.0 / dist
            ux = dx * inv_dist
            uy = dy * inv_dist
            uz = dz * inv_dist
            ip = 0.0
            for d in range(k):
                ip += prefs[i, d] * prefs[nj, d]
            ip /= k
            if pref_dist_weight:
                gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                mx += ip * gw * ux
                my += ip * gw * uy
                mz += ip * gw * uz
            else:
                mx += ip * ux
                my += ip * uy
                mz += ip * uz
            rx -= ux * inv_dist
            ry -= uy * inv_dist
            rz -= uz * inv_dist
            count += 1

        if count > 0:
            inv_c = 1.0 / count
            movement[i, 0] = mx * inv_c + repulsion * rx * inv_c
            movement[i, 1] = my * inv_c + repulsion * ry * inv_c
            movement[i, 2] = mz * inv_c + repulsion * rz * inv_c
        else:
            movement[i, 0] = 0.0
            movement[i, 1] = 0.0
            movement[i, 2] = 0.0

        new_pos[i, 0] = (pos[i, 0] + step_size * movement[i, 0]) % L
        new_pos[i, 1] = (pos[i, 1] + step_size * movement[i, 1]) % L
        new_pos[i, 2] = (pos[i, 2] + step_size * movement[i, 2]) % L

        if social > 0.0:
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
                    ddz = pos[nj, 2] - pos[i, 2]
                    ddx -= L * round(ddx / L)
                    ddy -= L * round(ddy / L)
                    ddz -= L * round(ddz / L)
                    dd = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
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
def _step_per_dim(pos, prefs, dir_matrix, nbr_ids, valid, L, k,
                  step_size, repulsion, social, social_dist_weight,
                  dir_memory, pref_weighted, pref_inner,
                  pref_dist_weight, pref_dist_sigma, best_by_magnitude):
    n = pos.shape[0]
    n_nbr = nbr_ids.shape[1]
    new_pos = np.empty((n, 3), dtype=np.float64)
    new_prefs = np.empty((n, k), dtype=np.float64)
    new_dm = np.empty((n, k, 3), dtype=np.float64)
    movement = np.empty((n, 3), dtype=np.float64)

    for i in prange(n):
        mx, my, mz = 0.0, 0.0, 0.0

        for ki in range(k):
            if pref_weighted:
                wx, wy, wz = 0.0, 0.0, 0.0
                wc = 0
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    dx = pos[nj, 0] - pos[i, 0]
                    dy = pos[nj, 1] - pos[i, 1]
                    dz = pos[nj, 2] - pos[i, 2]
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    dz -= L * round(dz / L)
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    wc += 1
                    if dist < 1e-12:
                        continue
                    ux = dx / dist
                    uy = dy / dist
                    uz = dz / dist
                    w = prefs[nj, ki]
                    if pref_dist_weight:
                        gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                        w *= gw
                    wx += w * ux
                    wy += w * uy
                    wz += w * uz
                if wc > 0:
                    wx /= wc
                    wy /= wc
                    wz /= wc
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * wx
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * wy
                new_dm[i, ki, 2] = dir_memory * dir_matrix[i, ki, 2] + (1.0 - dir_memory) * wz
                mx += prefs[i, ki] * new_dm[i, ki, 0]
                my += prefs[i, ki] * new_dm[i, ki, 1]
                mz += prefs[i, ki] * new_dm[i, ki, 2]
            else:
                best_val = -1e30
                best_nj = -1
                for j in range(n_nbr):
                    if not valid[i, j]:
                        continue
                    nj = nbr_ids[i, j]
                    if best_by_magnitude:
                        score = abs(prefs[nj, ki])
                    else:
                        score = prefs[nj, ki]
                    if score > best_val:
                        best_val = score
                        best_nj = nj
                if best_nj < 0:
                    new_dm[i, ki, 0] = dir_matrix[i, ki, 0]
                    new_dm[i, ki, 1] = dir_matrix[i, ki, 1]
                    new_dm[i, ki, 2] = dir_matrix[i, ki, 2]
                    continue
                dx = pos[best_nj, 0] - pos[i, 0]
                dy = pos[best_nj, 1] - pos[i, 1]
                dz = pos[best_nj, 2] - pos[i, 2]
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                dz -= L * round(dz / L)
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dist < 1e-12:
                    ux, uy, uz = 0.0, 0.0, 0.0
                else:
                    ux = dx / dist
                    uy = dy / dist
                    uz = dz / dist
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * ux
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * uy
                new_dm[i, ki, 2] = dir_memory * dir_matrix[i, ki, 2] + (1.0 - dir_memory) * uz
                compat = prefs[i, ki] * prefs[best_nj, ki]
                if pref_inner:
                    ip = 0.0
                    for di in range(k):
                        ip += prefs[i, di] * prefs[best_nj, di]
                    ip /= k
                    compat *= ip
                if pref_dist_weight:
                    gw = np.exp(-dist * dist / (2.0 * pref_dist_sigma * pref_dist_sigma))
                    compat *= gw
                mx += compat * new_dm[i, ki, 0]
                my += compat * new_dm[i, ki, 1]
                mz += compat * new_dm[i, ki, 2]

        rx, ry, rz = 0.0, 0.0, 0.0
        rc = 0
        for j in range(n_nbr):
            if not valid[i, j]:
                continue
            nj = nbr_ids[i, j]
            dx = pos[nj, 0] - pos[i, 0]
            dy = pos[nj, 1] - pos[i, 1]
            dz = pos[nj, 2] - pos[i, 2]
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            rc += 1
            if dist < 1e-6:
                continue
            inv_d = 1.0 / dist
            rx -= (dx * inv_d) * inv_d
            ry -= (dy * inv_d) * inv_d
            rz -= (dz * inv_d) * inv_d

        if rc > 0:
            movement[i, 0] = mx + repulsion * rx / rc
            movement[i, 1] = my + repulsion * ry / rc
            movement[i, 2] = mz + repulsion * rz / rc
        else:
            movement[i, 0] = mx
            movement[i, 1] = my
            movement[i, 2] = mz

        new_pos[i, 0] = (pos[i, 0] + step_size * movement[i, 0]) % L
        new_pos[i, 1] = (pos[i, 1] + step_size * movement[i, 1]) % L
        new_pos[i, 2] = (pos[i, 2] + step_size * movement[i, 2]) % L

        if social > 0.0:
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
                    ddz = pos[nj, 2] - pos[i, 2]
                    ddx -= L * round(ddx / L)
                    ddy -= L * round(ddy / L)
                    ddz -= L * round(ddz / L)
                    dd = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
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
