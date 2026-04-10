"""
3D Spatial Hash Grid — numba-accelerated
=========================================

Cell index: (cz * res + cy) * res + cx
Neighbor loops: 3×3×3 = 27 cells
All distances include dz with toroidal wrapping.
"""

import math
import numpy as np
from numba import njit, prange


SPACE = 1.0


@njit
def _hash_build(pos, cell_size, grid_res, n):
    """Assign each particle to a 3D grid cell."""
    cell_indices = np.empty(n, dtype=np.int32)
    for i in range(n):
        cx = np.int32(pos[i, 0] / cell_size) % grid_res
        cy = np.int32(pos[i, 1] / cell_size) % grid_res
        cz = np.int32(pos[i, 2] / cell_size) % grid_res
        cell_indices[i] = (cz * grid_res + cy) * grid_res + cx
    return cell_indices


@njit
def _compute_cell_ranges(sorted_cells, num_cells, n):
    """Given sorted cell indices, compute start/end for each cell."""
    cell_start = np.full(num_cells, -1, dtype=np.int32)
    cell_end = np.full(num_cells, -1, dtype=np.int32)
    for i in range(n):
        c = sorted_cells[i]
        if i == 0 or sorted_cells[i - 1] != c:
            cell_start[c] = np.int32(i)
        if i == n - 1 or sorted_cells[i + 1] != c:
            cell_end[c] = np.int32(i + 1)
    return cell_start, cell_end


@njit(parallel=True)
def _query_radius(pos, sort_order, cell_indices_sorted, cell_start, cell_end,
                  grid_res, cell_size, radius, L, max_nbr):
    """Radius query: for each particle, find all neighbors within radius."""
    n = pos.shape[0]
    r2 = radius * radius
    nbr_ids = np.zeros((n, max_nbr), dtype=np.int64)
    valid = np.zeros((n, max_nbr), dtype=np.bool_)
    nbr_counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        pz = pos[i, 2]
        cx = np.int32(px / cell_size) % grid_res
        cy = np.int32(py / cell_size) % grid_res
        cz = np.int32(pz / cell_size) % grid_res
        count = 0

        for dcz in range(-1, 2):
            nz = (cz + dcz) % grid_res
            for dcy in range(-1, 2):
                ny = (cy + dcy) % grid_res
                for dcx in range(-1, 2):
                    nx = (cx + dcx) % grid_res
                    cell = (nz * grid_res + ny) * grid_res + nx
                    cs = cell_start[cell]
                    if cs < 0:
                        continue
                    ce = cell_end[cell]
                    for si in range(cs, ce):
                        j = sort_order[si]
                        if j == i:
                            continue
                        dx = pos[j, 0] - px
                        dy = pos[j, 1] - py
                        dz = pos[j, 2] - pz
                        dx -= L * round(dx / L)
                        dy -= L * round(dy / L)
                        dz -= L * round(dz / L)
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= r2:
                            if count < max_nbr:
                                nbr_ids[i, count] = j
                                valid[i, count] = True
                                count += 1
        nbr_counts[i] = np.int32(count)

    return nbr_ids, valid, nbr_counts


@njit(parallel=True)
def _count_radius(pos, sort_order, cell_start, cell_end,
                  grid_res, cell_size, radius, L):
    """First pass: count neighbors per particle to size output arrays."""
    n = pos.shape[0]
    r2 = radius * radius
    counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        pz = pos[i, 2]
        cx = np.int32(px / cell_size) % grid_res
        cy = np.int32(py / cell_size) % grid_res
        cz = np.int32(pz / cell_size) % grid_res
        count = 0

        for dcz in range(-1, 2):
            nz = (cz + dcz) % grid_res
            for dcy in range(-1, 2):
                ny = (cy + dcy) % grid_res
                for dcx in range(-1, 2):
                    nx = (cx + dcx) % grid_res
                    cell = (nz * grid_res + ny) * grid_res + nx
                    cs = cell_start[cell]
                    if cs < 0:
                        continue
                    ce = cell_end[cell]
                    for si in range(cs, ce):
                        j = sort_order[si]
                        if j == i:
                            continue
                        dx = pos[j, 0] - px
                        dy = pos[j, 1] - py
                        dz = pos[j, 2] - pz
                        dx -= L * round(dx / L)
                        dy -= L * round(dy / L)
                        dz -= L * round(dz / L)
                        d2 = dx * dx + dy * dy + dz * dz
                        if d2 <= r2:
                            count += 1
        counts[i] = np.int32(count)

    return counts


@njit
def _query_knn_single(i, pos, sort_order, cell_start, cell_end,
                      grid_res, cell_size, L, k_neighbors, max_ring,
                      nbr_ids):
    """Find k nearest neighbors for a single particle i (3D)."""
    px = pos[i, 0]
    py = pos[i, 1]
    pz = pos[i, 2]
    cx = np.int32(px / cell_size) % grid_res
    cy = np.int32(py / cell_size) % grid_res
    cz = np.int32(pz / cell_size) % grid_res

    max_cand = 1024
    cand_idx = np.empty(max_cand, dtype=np.int64)
    cand_d2 = np.empty(max_cand, dtype=np.float32)

    for ring in range(1, max_ring + 1):
        n_cand = 0
        for dcz in range(-ring, ring + 1):
            nz_val = (cz + dcz) % grid_res
            for dcy in range(-ring, ring + 1):
                ny_val = (cy + dcy) % grid_res
                for dcx in range(-ring, ring + 1):
                    nx_val = (cx + dcx) % grid_res
                    cell = (nz_val * grid_res + ny_val) * grid_res + nx_val
                    cs = cell_start[cell]
                    if cs < 0:
                        continue
                    ce = cell_end[cell]
                    for si in range(cs, ce):
                        j = sort_order[si]
                        if j == i:
                            continue
                        dx = pos[j, 0] - px
                        dy = pos[j, 1] - py
                        dz = pos[j, 2] - pz
                        dx -= L * round(dx / L)
                        dy -= L * round(dy / L)
                        dz -= L * round(dz / L)
                        d2 = dx * dx + dy * dy + dz * dz
                        if n_cand < max_cand:
                            cand_idx[n_cand] = j
                            cand_d2[n_cand] = np.float32(d2)
                            n_cand += 1

        if n_cand < k_neighbors:
            continue

        # Partial selection sort to find k nearest
        actual_k = min(k_neighbors, n_cand)
        for ki in range(actual_k):
            min_idx = ki
            for ci in range(ki + 1, n_cand):
                if cand_d2[ci] < cand_d2[min_idx]:
                    min_idx = ci
            if min_idx != ki:
                cand_d2[ki], cand_d2[min_idx] = cand_d2[min_idx], cand_d2[ki]
                cand_idx[ki], cand_idx[min_idx] = cand_idx[min_idx], cand_idx[ki]

        kth_d2 = cand_d2[actual_k - 1]
        coverage = ring * cell_size
        if kth_d2 < coverage * coverage:
            for ki in range(actual_k):
                nbr_ids[i, ki] = cand_idx[ki]
            return

    # Exhausted rings — use best candidates we have
    actual_k = min(k_neighbors, n_cand)
    for ki in range(actual_k):
        nbr_ids[i, ki] = cand_idx[ki]


@njit(parallel=True)
def _query_knn(pos, sort_order, cell_start, cell_end,
               grid_res, cell_size, L, k_neighbors):
    """KNN query: for each particle find k nearest neighbors using 3D hash grid."""
    n = pos.shape[0]
    nbr_ids = np.zeros((n, k_neighbors), dtype=np.int64)
    max_ring = max(grid_res // 2, 3)

    for i in prange(n):
        _query_knn_single(i, pos, sort_order, cell_start, cell_end,
                          grid_res, cell_size, L, k_neighbors, max_ring,
                          nbr_ids)

    return nbr_ids


def grid_build(pos, cell_size, L=SPACE):
    """Build 3D spatial hash grid. Returns (sort_order, cell_start, cell_end, grid_res, cell_size_actual)."""
    grid_res = max(1, int(L / cell_size))
    cell_size_actual = L / grid_res
    num_cells = grid_res ** 3
    n = len(pos)

    cell_indices = _hash_build(pos, cell_size_actual, grid_res, n)
    sort_order = np.argsort(cell_indices, kind='mergesort').astype(np.int32)
    sorted_cells = cell_indices[sort_order]
    cell_start, cell_end = _compute_cell_ranges(sorted_cells, num_cells, n)

    return sort_order, cell_start, cell_end, grid_res, cell_size_actual


def find_neighbors_radius_hash(pos, radius, L=SPACE):
    """Radius neighbor search using 3D spatial hash grid."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 1), dtype=np.int64), np.zeros((n, 1), dtype=bool)

    cell_size = radius
    sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
        grid_build(pos, cell_size, L)

    counts = _count_radius(pos, sort_order, cell_start, cell_end,
                           grid_res, cell_size_actual, radius, L)
    max_nbr = int(counts.max())
    max_nbr = max(max_nbr, 1)
    max_nbr = min(max_nbr, n - 1)

    nbr_ids, valid, _ = _query_radius(
        pos, sort_order, np.empty(0, dtype=np.int32),
        cell_start, cell_end, grid_res, cell_size_actual, radius, L, max_nbr)

    return nbr_ids, valid


def find_neighbors_knn_hash(pos, k_neighbors, L=SPACE):
    """KNN neighbor search using 3D spatial hash grid."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 1), dtype=np.int64)

    # 3D density heuristic: (K / (4/3 * pi * N))^(1/3)
    cell_size = max(0.01, 2.0 * (k_neighbors / (4.0 / 3.0 * math.pi * max(n, 1))) ** (1.0 / 3.0))
    sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
        grid_build(pos, cell_size, L)

    nbr_ids = _query_knn(pos, sort_order, cell_start, cell_end,
                         grid_res, cell_size_actual, L, k_neighbors)
    return nbr_ids


def periodic_dist(a, b, L=SPACE):
    """Periodic distance vector: b - a, wrapped to [-L/2, L/2)."""
    d = b - a
    d -= L * np.round(d / L)
    return d
