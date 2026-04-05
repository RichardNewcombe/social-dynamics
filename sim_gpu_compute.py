#!/usr/bin/env python3
"""
Preference-Directed Particle Simulation — Optimized Compute Version
===================================================================

Fork of sim_gpu_update_gui.py with performance optimizations for 50k-100k
particles on Apple Silicon:

1. Spatial hash grid replaces scipy.cKDTree (O(N) build, cache-friendly query)
2. Float32 positions (halves bandwidth, better SIMD)
3. Full numba physics pipeline via _step_per_dim with all features
4. Sub-step neighbor reuse when steps_per_frame > 1
5. Extended radius slider range (0.001 lower bound)
6. Detailed timing display (grid build / query / physics)

Rendering uses GLFW + moderngl (OpenGL 4.1 Core Profile on macOS).
Physics stays on CPU (numpy + numba JIT, no scipy in hot path).

Layout: left half = live particle positions, right half = trail
accumulation (temporal moving average of particle positions).

Controls:
  Space       — pause / resume
  r           — reset simulation and trails
  q / Esc     — quit
  Up / Down   — adjust step_size
  + / -       — adjust social learning rate
  imgui panel — full slider control for all parameters

Dependencies (Python 3.12):
  numpy, scipy (only for get_neighbor_lines viz), glfw, moderngl,
  imgui-bundle, PyOpenGL, numba
"""

# ── Imports ──────────────────────────────────────────────────────────
import math
import numpy as np

# Prevent duplicate GLFW library loading: both pyglfw and imgui_bundle
# ship their own libglfw.3.dylib.  On macOS, loading two copies causes
# duplicate ObjC class warnings and window creation failures.
# Fix: tell pyglfw to use imgui_bundle's GLFW (must happen before imports).
import os, site
for _p in [site.getusersitepackages()] + \
          (site.getsitepackages() if hasattr(site, 'getsitepackages') else []):
    _candidate = os.path.join(_p, 'imgui_bundle', 'libglfw.3.dylib')
    if os.path.isfile(_candidate):
        os.environ['PYGLFW_LIBRARY'] = _candidate
        break

import glfw
import moderngl                      # OpenGL 4.1 wrapper (Core Profile)
from OpenGL import GL as _GL         # raw GL calls for framebuffer capture
import time
import subprocess
import ctypes
from numba import njit, prange       # JIT compilation for physics kernels
from imgui_bundle import imgui       # Dear ImGui Python bindings (imgui_bundle)
from scipy.spatial import cKDTree    # used for debug KNN validation

try:
    import torch
    _HAS_TORCH = True
    _TORCH_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"PyTorch available: device={_TORCH_DEVICE}")
except ImportError:
    _HAS_TORCH = False
    _TORCH_DEVICE = 'cpu'
    print("PyTorch not available — PyTorch physics disabled")


# =====================================================================
# PARAMETERS
# =====================================================================
params = dict(
    num_particles=2000,
    k=3,
    n_neighbors=21,
    step_size=0.005,
    steps_per_frame=1,
    repulsion=0.0,
    dir_memory=0.0,
    social=0.0,
    social_dist_weight=False,
    pref_weighted_dir=False,
    pref_inner_prod=False,
    inner_prod_avg=False,
    pref_dist_weight=False,
    best_by_magnitude=False,
    neighbor_mode=0,
    neighbor_radius=0.06,
    trail_decay=0.98,
    point_size=3.0,
    right_view=0,
    show_box=False,
    trail_zoom=True,
    pos_dist=0,
    pref_dist=0,
    gauss_sigma=0.15,
    show_neighbors=False,
    show_radius=False,
    use_seed=True,
    seed=42,
    auto_scale=False,
    reuse_neighbors=True,
    debug_knn=False,
    knn_method=0,  # 0=Hash Grid, 1=cKDTree (f64 pos), 2=cKDTree (f32 pos)
    use_f64=True,  # True = float64 positions (match original)
    physics_engine=0,  # 0=Numba, 1=NumPy (original), 2=PyTorch
    torch_precision=2,  # 0=f16, 1=bf16, 2=f32, 3=f64
    torch_device=0,  # 0=auto (mps if available), 1=cpu
    unit_prefs=False,  # True = normalize preference vectors to unit length
    track_mode=2,  # 0=Frozen (seed only), 1=+Neighbors, 2=Causal Spread
    crossover=False,        # enable preference crossover
    crossover_pct=50,       # P% of dims each particle keeps from self
    crossover_interval=1,   # run crossover every N steps
)

auto_scale_ref = dict(
    n=2000,
    step_size=0.005,
    radius=0.06,
)

POS_DISTS = ["Uniform", "Gaussian"]
PREF_DISTS = ["Uniform [-1,1]", "Gaussian", "Sparse ±1", "Unit Normalized"]

SPACE = 1.0
WINDOW_W, WINDOW_H = 0, 0


# =====================================================================
# SHADERS (GLSL 4.10 Core)
# =====================================================================

PARTICLE_VERT = '''
#version 410 core
in vec2 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform vec2 viewport_offset;
uniform vec2 viewport_scale;
uniform vec2 view_center;
uniform float view_zoom;
uniform float point_size;

void main() {
    vec2 p = in_pos - view_center;
    p -= round(p);
    vec2 ndc = p * view_zoom * 2.0;
    ndc = ndc * viewport_scale + (viewport_offset * 2.0 - 1.0 + viewport_scale);
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
'''

PARTICLE_FRAG = '''
#version 410 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    if (dot(pc, pc) > 1.0) discard;
    fragColor = vec4(v_color, 1.0);
}
'''

QUAD_VERT = '''
#version 410 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
'''

TRAIL_FRAG = '''
#version 410 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D trail_tex;
uniform float decay;
void main() {
    fragColor = vec4(texture(trail_tex, v_uv).rgb * decay, 1.0);
}
'''

SPLAT_FRAG = '''
#version 410 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(pc, pc);
    if (r2 > 1.0) discard;
    float alpha = 1.0 - r2;
    fragColor = vec4(v_color * alpha, alpha);
}
'''

DISPLAY_FRAG = '''
#version 410 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D tex;
uniform vec2 view_center;
uniform float view_zoom;
void main() {
    vec2 uv = (v_uv - 0.5) / view_zoom + view_center;
    vec3 c = texture(tex, uv).rgb;
    float m = max(c.r, max(c.g, c.b));
    if (m > 1.0) c /= m;
    fragColor = vec4(c, 1.0);
}
'''

BOX_VERT = '''
#version 410 core
in vec2 in_pos;
uniform vec2 view_center;
uniform float view_zoom;
void main() {
    vec2 ndc = (in_pos - view_center) * view_zoom * 2.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
'''

BOX_FRAG = '''
#version 410 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
'''

LINE_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''


# =====================================================================
# SPATIAL HASH GRID (replaces scipy.cKDTree)
# =====================================================================

@njit
def _hash_build(pos, cell_size, grid_res, n):
    """Assign each particle to a grid cell and compute cell ranges."""
    cell_indices = np.empty(n, dtype=np.int32)
    for i in range(n):
        cx = np.int32(pos[i, 0] / cell_size) % grid_res
        cy = np.int32(pos[i, 1] / cell_size) % grid_res
        cell_indices[i] = cy * grid_res + cx
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
        cx = np.int32(px / cell_size) % grid_res
        cy = np.int32(py / cell_size) % grid_res
        count = 0

        for dcy in range(-1, 2):
            ny = (cy + dcy) % grid_res
            for dcx in range(-1, 2):
                nx = (cx + dcx) % grid_res
                cell = ny * grid_res + nx
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
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        if count < max_nbr:
                            nbr_ids[i, count] = j
                            valid[i, count] = True
                            count += 1
        nbr_counts[i] = np.int32(count)

    return nbr_ids, valid, nbr_counts


@njit(parallel=True)
def _sort_radius_nbrs(pos, nbr_ids, valid, L):
    """Sort each particle's radius neighbors by distance.

    Hash grid returns neighbors in cell-iteration order, which causes
    non-deterministic tie-breaking in the physics when multiple neighbors
    share the same preference score (common with Sparse ±1 prefs).
    Sorting by distance matches cKDTree behavior.
    """
    n = pos.shape[0]
    max_nbr = nbr_ids.shape[1]
    for i in prange(n):
        px = pos[i, 0]
        py = pos[i, 1]
        # Count valid neighbors
        count = 0
        for j in range(max_nbr):
            if valid[i, j]:
                count += 1
        if count < 2:
            continue
        # Gather valid neighbor ids and distances
        ids = np.empty(count, dtype=np.int64)
        d2s = np.empty(count, dtype=np.float64)
        idx = 0
        for j in range(max_nbr):
            if valid[i, j]:
                nj = nbr_ids[i, j]
                ids[idx] = nj
                dx = pos[nj, 0] - px
                dy = pos[nj, 1] - py
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                d2s[idx] = dx * dx + dy * dy
                idx += 1
        # Selection sort by distance (count is small, typically 20-40)
        for a in range(count - 1):
            mi = a
            for b in range(a + 1, count):
                if d2s[b] < d2s[mi]:
                    mi = b
            if mi != a:
                d2s[a], d2s[mi] = d2s[mi], d2s[a]
                ids[a], ids[mi] = ids[mi], ids[a]
        # Write back sorted
        for j in range(count):
            nbr_ids[i, j] = ids[j]


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
        cx = np.int32(px / cell_size) % grid_res
        cy = np.int32(py / cell_size) % grid_res
        count = 0

        for dcy in range(-1, 2):
            ny = (cy + dcy) % grid_res
            for dcx in range(-1, 2):
                nx = (cx + dcx) % grid_res
                cell = ny * grid_res + nx
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
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        count += 1
        counts[i] = np.int32(count)

    return counts


@njit
def _query_knn_single(i, pos, sort_order, cell_start, cell_end,
                      grid_res, cell_size, L, k_neighbors, max_ring,
                      nbr_ids):
    """Find k nearest neighbors for a single particle i."""
    px = pos[i, 0]
    py = pos[i, 1]
    cx = np.int32(px / cell_size) % grid_res
    cy = np.int32(py / cell_size) % grid_res

    max_cand = 1024
    cand_idx = np.empty(max_cand, dtype=np.int64)
    cand_d2 = np.empty(max_cand, dtype=np.float32)

    for ring in range(1, max_ring + 1):
        n_cand = 0
        for dcy in range(-ring, ring + 1):
            ny = (cy + dcy) % grid_res
            for dcx in range(-ring, ring + 1):
                nx = (cx + dcx) % grid_res
                cell = ny * grid_res + nx
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
                    dx -= L * round(dx / L)
                    dy -= L * round(dy / L)
                    d2 = dx * dx + dy * dy
                    if n_cand < max_cand:
                        cand_idx[n_cand] = j
                        cand_d2[n_cand] = np.float32(d2)
                        n_cand += 1

        if n_cand < k_neighbors:
            continue  # not enough candidates, expand

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

        # Validate: K-th nearest distance must be within guaranteed coverage.
        # Minimum coverage from any particle to edge of search region = ring * cell_size
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
    """KNN query: for each particle find k nearest neighbors using hash grid.

    Uses distance-validated ring expansion: after finding K candidates in a
    ring, checks whether the K-th nearest distance is within the guaranteed
    coverage (ring * cell_size).  If not, expands to the next ring since
    true nearest neighbors may lie outside the current search area.
    """
    n = pos.shape[0]
    nbr_ids = np.zeros((n, k_neighbors), dtype=np.int64)
    max_ring = max(grid_res // 2, 3)

    for i in prange(n):
        _query_knn_single(i, pos, sort_order, cell_start, cell_end,
                          grid_res, cell_size, L, k_neighbors, max_ring,
                          nbr_ids)

    return nbr_ids


def grid_build(pos, cell_size, L=SPACE):
    """Build spatial hash grid. Returns (sort_order, cell_start, cell_end, grid_res)."""
    grid_res = max(1, int(L / cell_size))
    cell_size_actual = L / grid_res
    num_cells = grid_res * grid_res
    n = len(pos)

    cell_indices = _hash_build(pos, cell_size_actual, grid_res, n)
    sort_order = np.argsort(cell_indices, kind='mergesort').astype(np.int32)
    sorted_cells = cell_indices[sort_order]
    cell_start, cell_end = _compute_cell_ranges(sorted_cells, num_cells, n)

    return sort_order, cell_start, cell_end, grid_res, cell_size_actual


def find_neighbors_radius_hash(pos, radius, L=SPACE):
    """Radius neighbor search using spatial hash grid."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 1), dtype=np.int64), np.zeros((n, 1), dtype=bool)

    cell_size = radius
    sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
        grid_build(pos, cell_size, L)

    # First pass: count neighbors to determine max_nbr
    counts = _count_radius(pos, sort_order, cell_start, cell_end,
                           grid_res, cell_size_actual, radius, L)
    max_nbr = int(counts.max())
    max_nbr = max(max_nbr, 1)
    max_nbr = min(max_nbr, n - 1)

    # Second pass: fill neighbor arrays
    nbr_ids, valid, _ = _query_radius(
        pos, sort_order, np.empty(0, dtype=np.int32),
        cell_start, cell_end, grid_res, cell_size_actual, radius, L, max_nbr)

    return nbr_ids, valid


def find_neighbors_knn_hash(pos, k_neighbors, L=SPACE):
    """KNN neighbor search using spatial hash grid."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 1), dtype=np.int64)

    # cell_size heuristic: density-based
    cell_size = max(0.01, 2.0 * (1.0 / n) ** 0.5)
    sort_order, cell_start, cell_end, grid_res, cell_size_actual = \
        grid_build(pos, cell_size, L)

    nbr_ids = _query_knn(pos, sort_order, cell_start, cell_end,
                         grid_res, cell_size_actual, L, k_neighbors)
    return nbr_ids


# Keep cKDTree-based functions as fallback for visualization
def periodic_dist(a, b, L=SPACE):
    d = b - a
    d -= L * np.round(d / L)
    return d

_N_CIRCLE_SEGS = 32
_circle_angles = np.linspace(0, 2 * np.pi, _N_CIRCLE_SEGS + 1)
_CIRCLE_STARTS = np.column_stack([np.cos(_circle_angles[:-1]),
                                   np.sin(_circle_angles[:-1])])
_CIRCLE_ENDS   = np.column_stack([np.cos(_circle_angles[1:]),
                                   np.sin(_circle_angles[1:])])


def make_radius_circles(positions, radius):
    starts_r = _CIRCLE_STARTS * radius
    ends_r   = _CIRCLE_ENDS   * radius
    all_starts = positions[:, None, :] + starts_r[None, :, :]
    all_ends   = positions[:, None, :] + ends_r[None, :, :]
    n = len(positions)
    lines = np.empty((n * _N_CIRCLE_SEGS * 2, 2), dtype=np.float32)
    lines[0::2] = all_starts.reshape(-1, 2)
    lines[1::2] = all_ends.reshape(-1, 2)
    return lines


# =====================================================================
# NUMBA-JIT PHYSICS KERNELS
# =====================================================================

@njit(parallel=True)
def _step_inner_prod_avg(pos, prefs, nbr_ids, valid, L, k,
                         step_size, repulsion, social, social_dist_weight,
                         pref_dist_weight, pref_dist_sigma):
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
                ip += prefs[i, d] * prefs[nj, d]
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
def _step_per_dim(pos, prefs, dir_matrix, nbr_ids, valid, L, k,
                  step_size, repulsion, social, social_dist_weight,
                  dir_memory, pref_weighted, pref_inner,
                  pref_dist_weight, pref_dist_sigma, best_by_magnitude):
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
                    ux = dx / dist
                    uy = dy / dist
                    w = prefs[nj, ki]
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
                mx += prefs[i, ki] * new_dm[i, ki, 0]
                my += prefs[i, ki] * new_dm[i, ki, 1]
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
                    continue
                dx = pos[best_nj, 0] - pos[i, 0]
                dy = pos[best_nj, 1] - pos[i, 1]
                dx -= L * round(dx / L)
                dy -= L * round(dy / L)
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1e-12:
                    ux, uy = 0.0, 0.0
                else:
                    ux = dx / dist
                    uy = dy / dist
                new_dm[i, ki, 0] = dir_memory * dir_matrix[i, ki, 0] + (1.0 - dir_memory) * ux
                new_dm[i, ki, 1] = dir_memory * dir_matrix[i, ki, 1] + (1.0 - dir_memory) * uy
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


# =====================================================================
# PYTORCH VECTORIZED PHYSICS
# =====================================================================

_TORCH_DTYPES = {
    0: 'float16',
    1: 'bfloat16',
    2: 'float32',
    3: 'float64',
}


def _torch_periodic_dist(a, b, L=1.0):
    """Periodic distance: b - a, wrapped to [-L/2, L/2)."""
    d = b - a
    d = d - L * torch.round(d / L)
    return d


def _step_torch(pos_np, prefs_np, dm_np, nbr_ids_np, valid_np,
                L, k, step_size, repulsion, dir_memory,
                social, social_dist_weight,
                pref_weighted, pref_inner, inner_avg,
                pref_dist_w, pref_dist_sigma, best_mag):
    """Full physics step using PyTorch vectorized ops."""
    prec = params['torch_precision']
    dtype_name = _TORCH_DTYPES.get(prec, 'float32')
    dtype = getattr(torch, dtype_name)

    dev_idx = params['torch_device']
    if dev_idx == 0:
        device = _TORCH_DEVICE
    else:
        device = 'cpu'

    # f64 not supported on MPS
    if device == 'mps' and dtype == torch.float64:
        device = 'cpu'
    # bf16 might not be supported on MPS
    if device == 'mps' and dtype == torch.bfloat16:
        try:
            torch.zeros(1, dtype=torch.bfloat16, device='mps')
        except Exception:
            device = 'cpu'

    n = len(pos_np)
    has_mask = valid_np is not None

    # Transfer to torch
    # Keep prefs as float32 to match NumPy path behavior (prefs stay f32 there)
    pos = torch.tensor(pos_np, dtype=dtype, device=device)
    prefs = torch.tensor(prefs_np, dtype=torch.float32, device=device)
    dm = torch.tensor(dm_np, dtype=dtype, device=device)
    nbr_ids = torch.tensor(nbr_ids_np.astype(np.int64), device=device)
    if has_mask:
        valid = torch.tensor(valid_np, device=device)
        n_valid = valid.sum(dim=1).clamp(min=1).to(dtype)
    else:
        valid = None
        n_nbr = nbr_ids.shape[1]

    # Neighbor positions and periodic displacement
    nbr_pos = pos[nbr_ids]  # (N, n_nbr, 2)
    toward = _torch_periodic_dist(pos.unsqueeze(1), nbr_pos, L)  # (N, n_nbr, 2)
    dists = toward.norm(dim=2, keepdim=True).clamp(min=1e-12)  # (N, n_nbr, 1)
    toward_unit = toward / dists

    movement = torch.zeros(n, 2, dtype=dtype, device=device)

    if inner_avg:
        # Inner product average mode
        # NumPy path uses numba kernel with prefs upcast to f64, so match that
        prefs_ip = prefs.to(dtype)
        ip = (prefs_ip.unsqueeze(1) * prefs_ip[nbr_ids]).sum(dim=2) / k  # (N, n_nbr)
        if pref_dist_w:
            gw = torch.exp(-dists.squeeze(2) ** 2 / (2.0 * pref_dist_sigma ** 2))
            ip = ip * gw
        weighted = ip.unsqueeze(2) * toward_unit  # (N, n_nbr, 2)
        if has_mask:
            weighted = weighted * valid.unsqueeze(2)
            movement = weighted.sum(dim=1) / n_valid.unsqueeze(1)
        else:
            movement = weighted.mean(dim=1)

        # Repulsion
        unit_away = -toward_unit
        push_raw = unit_away / dists.clamp(min=1e-6)
        if has_mask:
            push_raw = push_raw * valid.unsqueeze(2)
            push = push_raw.sum(dim=1) / n_valid.unsqueeze(1)
        else:
            push = push_raw.mean(dim=1)
        movement = movement + repulsion * push

        new_pos = (pos + step_size * movement) % L

        # Social learning
        new_prefs = prefs.clone()
        if social > 0:
            nbr_prefs = prefs[nbr_ids]  # (N, n_nbr, k)
            if social_dist_weight:
                d = dists.squeeze(2)  # (N, n_nbr)
                w = 1.0 / (d + 1e-6)
                if has_mask:
                    w = w * valid.to(dtype)
                w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-10)
                w = w / w_sum
                nbr_mean = (nbr_prefs * w.unsqueeze(2)).sum(dim=1)
            else:
                if has_mask:
                    nbr_prefs_m = nbr_prefs * valid.unsqueeze(2).float()
                    nbr_mean = nbr_prefs_m.sum(dim=1) / n_valid.unsqueeze(1)
                else:
                    nbr_mean = nbr_prefs.mean(dim=1)
            new_prefs = (1.0 - social) * prefs + social * nbr_mean
            new_prefs = new_prefs.clamp(-1, 1)

        # Transfer back
        return (new_pos.cpu().to(torch.float64).numpy(),
                new_prefs.cpu().to(torch.float32).numpy(),
                dm_np,  # dm unchanged in inner_avg mode
                movement.cpu().to(torch.float64).numpy())

    # Per-dimension mode
    arange_n = torch.arange(n, device=device)
    for ki in range(k):
        nbr_pref_k = prefs[nbr_ids, ki]  # (N, n_nbr)

        if pref_weighted:
            weights = nbr_pref_k.unsqueeze(2)  # (N, n_nbr, 1)
            if pref_dist_w:
                gw = torch.exp(-dists ** 2 / (2.0 * pref_dist_sigma ** 2))
                weights = weights * gw
            weighted = weights * toward_unit
            if has_mask:
                weighted = weighted * valid.unsqueeze(2).to(dtype)
                weighted_dir = weighted.sum(dim=1) / n_valid.unsqueeze(1)
            else:
                weighted_dir = weighted.mean(dim=1)
            dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * weighted_dir
            movement = movement + prefs[:, ki:ki+1] * dm[:, ki, :]

        else:
            score = nbr_pref_k.abs() if best_mag else nbr_pref_k
            if has_mask:
                masked_score = torch.where(valid, score, torch.tensor(float('-inf'), dtype=score.dtype, device=device))
                best_local = masked_score.argmax(dim=1)
            else:
                best_local = score.argmax(dim=1)
            best_nbr = nbr_ids[arange_n, best_local]

            disp = _torch_periodic_dist(pos, pos[best_nbr], L)
            dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-12)
            unit_dir = disp / dist

            dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * unit_dir

            compat = prefs[:, ki] * prefs[best_nbr, ki]
            if pref_inner:
                full_compat = (prefs * prefs[best_nbr]).sum(dim=1) / k
                compat = compat * full_compat
            if pref_dist_w:
                gw = torch.exp(-dist.squeeze(1) ** 2 / (2.0 * pref_dist_sigma ** 2))
                compat = compat * gw
            movement = movement + compat.unsqueeze(1) * dm[:, ki, :]

    # Repulsion
    unit_away = -toward_unit
    push_raw = unit_away / dists.clamp(min=1e-6)
    if has_mask:
        push_raw = push_raw * valid.unsqueeze(2).to(dtype)
        push = push_raw.sum(dim=1) / n_valid.unsqueeze(1)
    else:
        push = push_raw.mean(dim=1)
    movement = movement + repulsion * push

    new_pos = (pos + step_size * movement) % L

    # Social learning
    new_prefs = prefs.clone()
    if social > 0:
        nbr_prefs = prefs[nbr_ids]
        if social_dist_weight:
            d = dists.squeeze(2)
            w = 1.0 / (d + 1e-6)
            if has_mask:
                w = w * valid.to(dtype)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-10)
            w = w / w_sum
            nbr_mean = (nbr_prefs * w.unsqueeze(2)).sum(dim=1)
        else:
            if has_mask:
                nbr_prefs_m = nbr_prefs * valid.unsqueeze(2).float()
                nbr_mean = nbr_prefs_m.sum(dim=1) / n_valid.unsqueeze(1)
            else:
                nbr_mean = nbr_prefs.mean(dim=1)
        new_prefs = (1.0 - social) * prefs + social * nbr_mean
        new_prefs = new_prefs.clamp(-1, 1)

    # Transfer back to numpy
    return (new_pos.cpu().to(torch.float64).numpy(),
            new_prefs.cpu().to(torch.float32).numpy(),
            dm.cpu().to(torch.float64).numpy(),
            movement.cpu().to(torch.float64).numpy())


# =====================================================================
# SIMULATION (CPU — numpy + numba, no scipy in hot path)
# =====================================================================

class Simulation:
    def __init__(self):
        self.rng = np.random.default_rng(params['seed'] if params['use_seed'] else None)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(params['seed'] if params['use_seed'] else None)
        self.n = params['num_particles']
        self.k = params['k']
        n, k = self.n, self.k
        self._arange = np.arange(n)
        # Float32 positions for reduced bandwidth and better SIMD
        pos_dtype = np.float64 if params['use_f64'] else np.float32
        self.pos = np.zeros((n, 2), dtype=pos_dtype)
        self.prefs = np.zeros((n, k), dtype=np.float32)
        self.dir_matrix = np.zeros((n, k, 2), np.float64 if params['use_f64'] else np.float32)
        self._movement = np.zeros((n, 2), np.float64 if params['use_f64'] else np.float32)
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

        if params['unit_prefs']:
            self._normalize_prefs()

        self.tracked = np.zeros(n, dtype=bool)
        self.tracked_seed = np.zeros(n, dtype=bool)

    def _init_positions(self, dist):
        n, rng = self.n, self.rng
        L = SPACE
        pdtype = self.pos.dtype
        if dist == 1:  # Gaussian
            sigma = params['gauss_sigma']
            self.pos[:] = (rng.normal(L / 2, L * sigma, (n, 2)) % L).astype(pdtype)
        else:  # 0 = Uniform
            self.pos[:] = rng.uniform(0, L, (n, 2)).astype(pdtype)

    def _init_preferences(self, dist):
        n, k, rng = self.n, self.k, self.rng
        if dist == 1:  # Gaussian
            self.prefs[:] = np.clip(rng.normal(0, 0.5, (n, k)), -1, 1).astype(np.float32)
        elif dist == 2:  # Sparse ±1
            self.prefs[:] = 0.0
            if k < 2:
                self.prefs[:, 0] = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
            else:
                for i in range(n):
                    dims = rng.choice(k, size=2, replace=False)
                    self.prefs[i, dims[0]] = 1.0
                    self.prefs[i, dims[1]] = -1.0
        elif dist == 3:  # Unit Normalized
            raw = rng.normal(0, 1, (n, k))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            self.prefs[:] = (raw / norms).astype(np.float32)
        else:  # 0 = Uniform [-1,1]
            self.prefs[:] = rng.uniform(-1, 1, (n, k)).astype(np.float32)

    def _normalize_prefs(self):
        """Normalize each particle's preference vector to unit length."""
        norms = np.linalg.norm(self.prefs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.prefs[:] = (self.prefs / norms).astype(np.float32)

    def _find_neighbors(self):
        """Find neighbors. Method selected by params['knn_method']:
           0 = Hash Grid (compute), 1 = cKDTree f64, 2 = cKDTree f32."""
        pos = self.pos
        n = len(pos)
        nbr_mode = params['neighbor_mode']
        n_nbr = min(params['n_neighbors'], n - 1)
        radius = params['neighbor_radius']
        knn_method = params['knn_method']

        _tb0 = time.perf_counter()

        if knn_method == 0:
            # ── Hash Grid path ──
            self._find_neighbors_hash(pos, n, nbr_mode, n_nbr, radius, _tb0)
        else:
            # ── cKDTree path ──
            if knn_method == 1:
                query_pos = pos.astype(np.float64)
            else:
                query_pos = pos.astype(np.float64)
                # f32 positions: use float32-precision positions cast to f64
                # so cKDTree sees the same positions as the hash grid
                query_pos = pos.astype(np.float32).astype(np.float64)
            query_pos = query_pos % SPACE  # clamp for cKDTree boxsize

            self._t_build = 0.0
            _tq0 = time.perf_counter()
            tree = cKDTree(query_pos, boxsize=SPACE)

            if nbr_mode == 2:
                # Radius only
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
                # KNN (mode 0) or KNN+Radius (mode 1)
                _, nbr_ids = tree.query(query_pos, k=n_nbr + 1, workers=-1)
                self.nbr_ids = nbr_ids[:, 1:].astype(np.int64)
                self._valid_mask = None

            self._t_query = time.perf_counter() - _tq0

        self._t_search = self._t_build + self._t_query

        # ── Debug: compare hash grid KNN against cKDTree ──
        if params['debug_knn'] and knn_method == 0 and nbr_mode in (0, 1):
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

    def _find_neighbors_hash(self, pos, n, nbr_mode, n_nbr, radius, _tb0):
        """Hash grid neighbor search (original compute path)."""
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

        else:
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

    def step(self, reuse_neighbors=False):
        self._step_impl(reuse_neighbors)
        if params['unit_prefs']:
            self._normalize_prefs()
        if params['crossover'] and self.step_count % max(1, params['crossover_interval']) == 0:
            self.crossover_step()
        mode = params['track_mode']
        if mode == 2:
            self.expand_tracked()
        elif mode == 1:
            self.update_tracked_with_neighbors()

    def expand_tracked(self):
        if not self.tracked.any() or self.nbr_ids is None:
            return
        nbr_ids = self.nbr_ids
        tracked_rows = nbr_ids[self.tracked]  # (n_tracked, n_nbr)
        if self._valid_mask is not None:
            valid_rows = self._valid_mask[self.tracked]
            new_ids = tracked_rows[valid_rows]
        else:
            new_ids = tracked_rows.ravel()
        self.tracked[new_ids] = True

    def update_tracked_with_neighbors(self):
        """Set tracked = seed + current neighbors of seed (no permanent growth)."""
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
        n_keep = max(0, min(n_keep, k))  # clamp
        if n_keep == k:
            return  # nothing to swap

        # Find each particle's nearest neighbor (column 0 of nbr_ids)
        nn = self.nbr_ids[:, 0]  # shape (n,)

        # Guard against particles with no valid neighbors
        if self._valid_mask is not None:
            has_valid = self._valid_mask[:, 0]
        else:
            has_valid = np.ones(n, dtype=bool)

        arange_n = np.arange(n)
        # Mutual pairs: i's NN is j AND j's NN is i, both have valid neighbors
        is_mutual = (nn[nn] == arange_n) & has_valid & has_valid[nn]
        # Only process pairs where i < j to avoid duplicates
        candidates = np.where(is_mutual & (nn > arange_n))[0]

        if len(candidates) == 0:
            return

        i_idx = candidates
        j_idx = nn[candidates]

        # Swap tail dimensions
        prefs = self.prefs
        i_tail = prefs[i_idx, n_keep:].copy()
        j_tail = prefs[j_idx, n_keep:].copy()
        prefs[i_idx, n_keep:] = j_tail
        prefs[j_idx, n_keep:] = i_tail

    def _step_impl(self, reuse_neighbors=False):
        pos, prefs, dm = self.pos, self.prefs, self.dir_matrix
        n = len(pos)
        k = self.k
        n_nbr = min(params['n_neighbors'], n - 1)
        step_size = params['step_size']
        repulsion = params['repulsion']
        dir_memory = params['dir_memory']
        social = params['social']
        pref_weighted = params['pref_weighted_dir']
        pref_inner = params['pref_inner_prod']
        inner_avg = params['inner_prod_avg']
        pref_dist_w = params['pref_dist_weight']
        best_mag = params['best_by_magnitude']
        pref_dist_sigma = params['neighbor_radius'] / 4.0
        arange_n = self._arange

        nbr_mode = params['neighbor_mode']

        # Neighbor search (skip if reusing)
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
            # ── NumPy vectorized physics (matches original sim_gpu_update_gui.py) ──
            movement = self._movement
            movement[:] = 0.0

            nbr_pos = pos[nbr_ids]
            toward = periodic_dist(pos[:, None, :], nbr_pos)
            dists = np.linalg.norm(toward, axis=2, keepdims=True)
            toward_unit = toward / np.maximum(dists, 1e-12)

            if inner_avg:
                # inner_prod_avg uses the numba kernel even in numpy mode
                # (it was always numba in the original too)
                if valid is None:
                    valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
                else:
                    valid_arr = valid
                prefs_f64 = prefs.astype(np.float64)
                new_pos, new_prefs, mov = _step_inner_prod_avg(
                    pos.astype(np.float64), prefs_f64,
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
                    movement += prefs[:, ki:ki+1] * dm[:, ki, :]

                else:
                    score = np.abs(nbr_pref_k) if best_mag else nbr_pref_k
                    if has_mask:
                        masked_score = np.where(valid, score, -np.inf)
                        best_local = np.argmax(masked_score, axis=1)
                    else:
                        best_local = np.argmax(score, axis=1)
                    best_nbr = nbr_ids[arange_n, best_local]

                    disp = periodic_dist(pos, pos[best_nbr])
                    dist = np.linalg.norm(disp, axis=1, keepdims=True)
                    unit_dir = disp / np.maximum(dist, 1e-12)

                    dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * unit_dir

                    compat = prefs[:, ki] * prefs[best_nbr, ki]
                    if pref_inner:
                        full_compat = (prefs * prefs[best_nbr]).sum(axis=1) / k
                        compat = compat * full_compat
                    if pref_dist_w:
                        gw = np.exp(-dist[:, 0]**2 / (2.0 * pref_dist_sigma**2))
                        compat = compat * gw
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

            if social > 0:
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
            return

        if params['physics_engine'] == 2 and _HAS_TORCH:
            # ── PyTorch vectorized physics ──
            new_pos, new_prefs, new_dm, mov = _step_torch(
                pos, prefs, dm, nbr_ids, valid,
                SPACE, k, step_size, repulsion, dir_memory,
                social, params['social_dist_weight'],
                pref_weighted, pref_inner, inner_avg,
                pref_dist_w, pref_dist_sigma, best_mag)
            self.pos = new_pos.astype(pos.dtype)
            self.prefs = new_prefs
            self.dir_matrix = new_dm.astype(dm.dtype)
            self._movement = mov.astype(self._movement.dtype)
            self._t_physics = time.perf_counter() - _tp0
            self._n_nbrs = nbr_ids.shape[1]
            self.step_count += 1
            return

        # ── Numba physics path ──
        if inner_avg:
            if valid is None:
                valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
            else:
                valid_arr = valid
            prefs_f64 = prefs.astype(np.float64)
            new_pos, new_prefs, mov = _step_inner_prod_avg(
                pos, prefs_f64, nbr_ids.astype(np.int64), valid_arr,
                SPACE, k, step_size, repulsion, social,
                params['social_dist_weight'], pref_dist_w, pref_dist_sigma)
            self.pos = new_pos
            self.prefs = new_prefs.astype(np.float32)
            self._movement = mov
            self._t_physics = time.perf_counter() - _tp0
            self._n_nbrs = nbr_ids.shape[1]
            self.step_count += 1
            return
        else:
            if valid is None:
                valid_arr = np.ones((n, nbr_ids.shape[1]), dtype=np.bool_)
            else:
                valid_arr = valid
            prefs_f64 = prefs.astype(np.float64)
            new_pos, new_prefs, new_dm, mov = _step_per_dim(
                pos, prefs_f64, dm, nbr_ids.astype(np.int64), valid_arr,
                SPACE, k, step_size, repulsion, social,
                params['social_dist_weight'], dir_memory,
                pref_weighted, pref_inner,
                pref_dist_w, pref_dist_sigma, best_mag)
            self.pos = new_pos
            self.prefs = new_prefs.astype(np.float32)
            self.dir_matrix = new_dm
            self._movement = mov
            self._t_physics = time.perf_counter() - _tp0
            self._n_nbrs = nbr_ids.shape[1]
            self.step_count += 1
            return

    def get_render_data(self):
        k = self.k
        colors = np.clip((self.prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
        if k < 3:
            c = np.full((len(self.prefs), 3), 0.5, np.float32)
            c[:, :min(k, 3)] = colors[:, :min(k, 3)]
            colors = c
        return self.pos.astype(np.float32), colors

    def get_velocity_colors(self):
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


# =====================================================================
# MAIN LOOP (GLFW + imgui_bundle)
# =====================================================================

def main():
    # ── Initialize GLFW ──
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # ── Compute window size to fill the screen ──
    global WINDOW_W, WINDOW_H
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height
    WINDOW_H = screen_h - 130
    WINDOW_W = 2 * WINDOW_H
    if WINDOW_W > screen_w - 20:
        WINDOW_W = screen_w - 20
        WINDOW_H = WINDOW_W // 2

    # ── Request OpenGL 4.1 Core Profile ──
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_W, WINDOW_H,
                                "Particle Simulation — Compute", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # uncapped FPS

    # ── Query framebuffer size (Retina displays have 2x pixels) ──
    fb_w, fb_h = glfw.get_framebuffer_size(window)

    # ── Create moderngl context ──
    ctx = moderngl.create_context(require=410)
    renderer_name = ctx.info["GL_RENDERER"]
    gl_ver = ctx.info["GL_VERSION"]
    print(f"GL: {gl_ver}  |  Renderer: {renderer_name}")
    print(f"Window: {WINDOW_W}x{WINDOW_H}  Framebuffer: {fb_w}x{fb_h}")
    if "Software" in renderer_name:
        print("WARNING: Using software renderer.")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    # ── imgui setup (imgui_bundle) ──
    imgui_ctx = imgui.create_context()
    io = imgui.get_io()
    io.config_mac_osx_behaviors = True
    io.config_drag_click_to_input_text = True

    # ── View state ──
    view_center = [0.5, 0.5]
    view_zoom = 1.0
    pan_active = False
    prev_view_center = [0.5, 0.5]
    prev_view_zoom = 1.0
    prev_mouse_pos = [0.0, 0.0]

    # ── Selection state (right-click drag) ──
    selecting = False
    sel_start = [0.0, 0.0]  # screen coords
    sel_end = [0.0, 0.0]

    def screen_to_sim(sx, sy):
        """Convert screen pixel coords to simulation [0,1] space.
        Returns None if click is on the right half of the window."""
        hw = WINDOW_W // 2
        if sx >= hw:
            return None
        # Pixel coords -> NDC in the left viewport
        ndc_x = sx / hw * 2.0 - 1.0
        ndc_y = 1.0 - sy / WINDOW_H * 2.0
        # Inverse of the vertex shader transform:
        # ndc = (pos - view_center - round(pos - view_center)) * view_zoom * 2.0
        # We solve for the view-relative position, then add view_center
        sim_x = ndc_x / (view_zoom * 2.0) + view_center[0]
        sim_y = ndc_y / (view_zoom * 2.0) + view_center[1]
        return (sim_x, sim_y)

    def scroll_callback(win, xoffset, yoffset):
        nonlocal view_zoom
        if io.want_capture_mouse:
            return
        mx, my = glfw.get_cursor_pos(win)
        hw = WINDOW_W // 2
        ndc_x = (mx % hw) / hw * 2.0 - 1.0
        ndc_y = 1.0 - my / WINDOW_H * 2.0
        factor = 1.15 ** yoffset
        new_zoom = max(1.0, min(view_zoom * factor, 20.0))
        view_center[0] += ndc_x / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_center[1] += ndc_y / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_zoom = new_zoom

    def mouse_button_callback(win, button, action, mods):
        nonlocal pan_active, selecting
        if io.want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            cmd_held = bool(mods & glfw.MOD_SUPER)
            if action == glfw.PRESS:
                mx, my = glfw.get_cursor_pos(win)
                if cmd_held:
                    # Cmd+click: start selection rectangle
                    hw = WINDOW_W // 2
                    if mx < hw:
                        selecting = True
                        sel_start[0] = mx
                        sel_start[1] = my
                        sel_end[0] = mx
                        sel_end[1] = my
                else:
                    # Plain click: pan
                    pan_active = True
                    prev_mouse_pos[0] = mx
                    prev_mouse_pos[1] = my
            elif action == glfw.RELEASE:
                if selecting:
                    selecting = False
                    # Convert both corners to sim space
                    c0 = screen_to_sim(sel_start[0], sel_start[1])
                    c1 = screen_to_sim(sel_end[0], sel_end[1])
                    if c0 is not None and c1 is not None:
                        sx0 = min(c0[0], c1[0])
                        sx1 = max(c0[0], c1[0])
                        sy0 = min(c0[1], c1[1])
                        sy1 = max(c0[1], c1[1])
                        # Toroidal-aware selection: compute positions
                        # relative to view_center with wrapping
                        vc = np.array(view_center, dtype=np.float64)
                        pos = sim.pos.astype(np.float64)
                        rel = pos - vc
                        rel -= np.round(rel)
                        abs_pos = rel + vc
                        mask_x = (abs_pos[:, 0] >= sx0) & (abs_pos[:, 0] <= sx1)
                        mask_y = (abs_pos[:, 1] >= sy0) & (abs_pos[:, 1] <= sy1)
                        matches = mask_x & mask_y
                        sim.tracked_seed[matches] = True
                        sim.tracked[matches] = True
                pan_active = False

    def cursor_pos_callback(win, mx, my):
        if pan_active and not selecting and not io.want_capture_mouse:
            dx = mx - prev_mouse_pos[0]
            dy = my - prev_mouse_pos[1]
            hw = WINDOW_W // 2
            view_center[0] -= dx / hw / view_zoom
            view_center[1] += dy / WINDOW_H / view_zoom
        if selecting:
            sel_end[0] = mx
            sel_end[1] = my
        prev_mouse_pos[0] = mx
        prev_mouse_pos[1] = my

    def key_callback(win, key, scancode, action, mods):
        nonlocal running_sim
        if io.want_capture_keyboard:
            return
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                running_sim = not running_sim
            elif key == glfw.KEY_R:
                do_reset()
            elif key == glfw.KEY_UP:
                params['step_size'] = min(params['step_size'] + 0.001, 0.05)
            elif key == glfw.KEY_DOWN:
                params['step_size'] = max(params['step_size'] - 0.001, 0.001)
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                params['social'] = min(params['social'] + 0.005, 0.1)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                params['social'] = max(params['social'] - 0.005, 0.0)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)

    # Initialize imgui backends (AFTER our callbacks so imgui chains onto them)
    imgui.backends.opengl3_init("#version 410")
    window_ptr = ctypes.cast(window, ctypes.c_void_p).value
    imgui.backends.glfw_init_for_opengl(window_ptr, True)

    # ── Compile shader programs ──
    prog_particle = ctx.program(vertex_shader=PARTICLE_VERT,
                                fragment_shader=PARTICLE_FRAG)

    num_particles = params['num_particles']

    vbo_pos = ctx.buffer(reserve=num_particles * 2 * 4)
    vbo_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_particle = ctx.vertex_array(prog_particle, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # ── Trail FBO setup (ping-pong pair) ──
    trail_w, trail_h = fb_w // 2, fb_h
    trail_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo = ctx.framebuffer(color_attachments=[trail_tex])

    trail_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo2 = ctx.framebuffer(color_attachments=[trail_tex2])

    quad_data = np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    ], dtype='f4')
    vbo_quad = ctx.buffer(quad_data.tobytes())

    prog_trail_decay = ctx.program(vertex_shader=QUAD_VERT,
                                   fragment_shader=TRAIL_FRAG)
    vao_trail_decay = ctx.vertex_array(prog_trail_decay, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])

    prog_display = ctx.program(vertex_shader=QUAD_VERT,
                               fragment_shader=DISPLAY_FRAG)
    vao_display = ctx.vertex_array(prog_display, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])

    prog_splat = ctx.program(vertex_shader=PARTICLE_VERT,
                             fragment_shader=SPLAT_FRAG)
    vao_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    prog_box = ctx.program(vertex_shader=BOX_VERT, fragment_shader=BOX_FRAG)
    box_verts = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype='f4')
    vbo_box = ctx.buffer(box_verts.tobytes())
    vao_box = ctx.vertex_array(prog_box, [(vbo_box, '2f', 'in_pos')])

    prog_line = ctx.program(vertex_shader=BOX_VERT, fragment_shader=LINE_FRAG)
    n_max_edges = params['num_particles'] * params['n_neighbors']
    vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 2 * 4)
    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])

    # ── Velocity field FBO setup ──
    vel_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo = ctx.framebuffer(color_attachments=[vel_tex])
    vel_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo2 = ctx.framebuffer(color_attachments=[vel_tex2])

    vbo_vel_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_vel_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_vel_col, '3f', 'in_color'),
    ])

    # ── Causal tracking VBOs ──
    vbo_causal_pos = ctx.buffer(reserve=num_particles * 2 * 4)
    vbo_causal_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_causal_splat = ctx.vertex_array(prog_splat, [
        (vbo_causal_pos, '2f', 'in_pos'),
        (vbo_causal_col, '3f', 'in_color'),
    ])
    vao_causal_particle = ctx.vertex_array(prog_particle, [
        (vbo_causal_pos, '2f', 'in_pos'),
        (vbo_causal_col, '3f', 'in_color'),
    ])

    for tex in (trail_tex, trail_tex2, vel_tex, vel_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # ── Causal trail FBO setup (ping-pong pair) ──
    causal_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    causal_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    causal_fbo = ctx.framebuffer(color_attachments=[causal_tex])
    causal_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    causal_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    causal_fbo2 = ctx.framebuffer(color_attachments=[causal_tex2])
    for tex in (causal_tex, causal_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # ── Create simulation and state ──
    sim = Simulation()
    running_sim = True

    # ── Recording state ──
    rec_process = None
    rec_frame_count = 0
    rec_interval = 1.0
    rec_last_time = 0.0
    rec_filename = ""

    def start_recording():
        nonlocal rec_process, rec_frame_count, rec_last_time, rec_filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rec_filename = os.path.join(os.path.dirname(__file__) or ".",
                                    f"timelapse_{timestamp}.mp4")
        rec_frame_count = 0
        rec_last_time = time.monotonic()
        rec_process = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{fb_w}x{fb_h}",
            "-framerate", "30",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            rec_filename,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Recording started: {rec_filename}")

    def stop_recording():
        nonlocal rec_process
        if rec_process is not None:
            rec_process.stdin.close()
            rec_process.wait()
            print(f"Recording saved: {rec_filename} ({rec_frame_count} frames)")
            rec_process = None

    def capture_frame():
        nonlocal rec_frame_count
        data = _GL.glReadPixels(0, 0, fb_w, fb_h,
                                _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE)
        row_size = fb_w * 3
        flipped = bytearray(fb_h * row_size)
        for dst_row in range(fb_h):
            src_row = fb_h - 1 - dst_row
            flipped[dst_row * row_size:(dst_row + 1) * row_size] = \
                data[src_row * row_size:(src_row + 1) * row_size]
        try:
            rec_process.stdin.write(flipped)
            rec_frame_count += 1
        except BrokenPipeError:
            stop_recording()


    def rebuild_buffers():
        nonlocal vbo_pos, vbo_col, vbo_vel_col, vao_particle, vao_splat, vao_vel_splat
        nonlocal vbo_line, vao_line
        nonlocal vbo_causal_pos, vbo_causal_col, vao_causal_splat, vao_causal_particle
        n = params['num_particles']
        vbo_pos = ctx.buffer(reserve=n * 2 * 4)
        vbo_col = ctx.buffer(reserve=n * 3 * 4)
        vbo_vel_col = ctx.buffer(reserve=n * 3 * 4)
        vao_particle = ctx.vertex_array(prog_particle, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_vel_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_vel_col, '3f', 'in_color'),
        ])
        n_max_edges = n * params['n_neighbors']
        vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 2 * 4)
        vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])
        # Causal tracking buffers
        vbo_causal_pos = ctx.buffer(reserve=n * 2 * 4)
        vbo_causal_col = ctx.buffer(reserve=n * 3 * 4)
        vao_causal_splat = ctx.vertex_array(prog_splat, [
            (vbo_causal_pos, '2f', 'in_pos'),
            (vbo_causal_col, '3f', 'in_color'),
        ])
        vao_causal_particle = ctx.vertex_array(prog_particle, [
            (vbo_causal_pos, '2f', 'in_pos'),
            (vbo_causal_col, '3f', 'in_color'),
        ])

    def do_reset():
        nonlocal running_sim
        if params['auto_scale']:
            ref = auto_scale_ref
            scale = (ref['n'] / params['num_particles']) ** 0.5
            params['step_size'] = ref['step_size'] * scale
            params['neighbor_radius'] = ref['radius'] * scale
        sim.reset()
        rebuild_buffers()
        trail_fbo.use()
        ctx.clear(0, 0, 0)
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        vel_fbo.use()
        ctx.clear(0, 0, 0)
        vel_fbo2.use()
        ctx.clear(0, 0, 0)
        causal_fbo.use()
        ctx.clear(0, 0, 0)
        causal_fbo2.use()
        ctx.clear(0, 0, 0)
        running_sim = True

    def reset_view():
        nonlocal view_zoom
        view_center[0] = 0.5
        view_center[1] = 0.5
        view_zoom = 1.0


    # ── FPS tracking ──
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    t_sim = 0.0

    # ── JIT warmup ──
    print("Warming up numba JIT kernels...")
    _warmup_n = 100
    _warmup_pos = np.random.rand(_warmup_n, 2).astype(np.float64)
    _warmup_prefs = np.random.rand(_warmup_n, 3).astype(np.float64)
    _warmup_dm = np.zeros((_warmup_n, 3, 2), dtype=np.float64)
    _warmup_cell_size = 0.1
    _warmup_so, _warmup_cs, _warmup_ce, _warmup_gr, _warmup_csa = \
        grid_build(_warmup_pos, _warmup_cell_size)
    _warmup_counts = _count_radius(_warmup_pos, _warmup_so, _warmup_cs,
                                    _warmup_ce, _warmup_gr, _warmup_csa, 0.1, 1.0)
    _warmup_nbr, _warmup_val, _ = _query_radius(
        _warmup_pos, _warmup_so, np.empty(0, dtype=np.int32),
        _warmup_cs, _warmup_ce, _warmup_gr, _warmup_csa, 0.1, 1.0, 10)
    _sort_radius_nbrs(_warmup_pos, _warmup_nbr, _warmup_val, 1.0)
    _warmup_knn = _query_knn(_warmup_pos, _warmup_so, _warmup_cs, _warmup_ce,
                              _warmup_gr, _warmup_csa, 1.0, 5)
    _warmup_valid = np.ones((_warmup_n, _warmup_nbr.shape[1]), dtype=np.bool_)
    _step_inner_prod_avg(_warmup_pos, _warmup_prefs,
                         _warmup_nbr.astype(np.int64), _warmup_valid,
                         1.0, 3, 0.005, 0.0, 0.0, False, False, 0.01)
    _step_per_dim(_warmup_pos, _warmup_prefs, _warmup_dm,
                  _warmup_nbr.astype(np.int64), _warmup_valid,
                  1.0, 3, 0.005, 0.0, 0.0, False, 0.0, False, False,
                  False, 0.01, False)
    print("JIT warmup complete.")

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── Physics step ──
        t0 = time.perf_counter()
        if running_sim:
            spf = params['steps_per_frame']
            reuse = params['reuse_neighbors']
            for sub in range(spf):
                sim.step(reuse_neighbors=(reuse and sub > 0))
        t_sim = time.perf_counter() - t0

        # ── Upload particle data to GPU ──
        positions, colors = sim.get_render_data()
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        vel_colors = sim.get_velocity_colors()
        vbo_vel_col.write(vel_colors.tobytes())

        # ── Trail rendering pass ──
        trail_zoom_on = params['trail_zoom']
        view_changed = (view_center[0] != prev_view_center[0] or
                        view_center[1] != prev_view_center[1] or
                        view_zoom != prev_view_zoom)
        if trail_zoom_on and view_changed:
            for fbo in (trail_fbo, trail_fbo2, vel_fbo, vel_fbo2,
                        causal_fbo, causal_fbo2):
                fbo.use()
                ctx.clear(0, 0, 0)
        prev_view_center[0] = view_center[0]
        prev_view_center[1] = view_center[1]
        prev_view_zoom = view_zoom

        if trail_zoom_on:
            splat_center = tuple(view_center)
            splat_zoom = view_zoom
        else:
            splat_center = (0.5, 0.5)
            splat_zoom = 1.0

        # Pass 1: Decay
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: Splat
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = splat_center
        prog_splat['view_zoom'] = splat_zoom
        prog_splat['point_size'] = params['point_size']
        vao_splat.render(moderngl.POINTS)

        # Pass 3: Swap
        trail_tex, trail_tex2 = trail_tex2, trail_tex
        trail_fbo, trail_fbo2 = trail_fbo2, trail_fbo

        # ── Velocity field rendering pass ──
        vel_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        vel_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = splat_center
        prog_splat['view_zoom'] = splat_zoom
        prog_splat['point_size'] = params['point_size']
        vao_vel_splat.render(moderngl.POINTS)

        vel_tex, vel_tex2 = vel_tex2, vel_tex
        vel_fbo, vel_fbo2 = vel_fbo2, vel_fbo

        # ── Causal trail rendering pass ──
        if sim.tracked.any():
            tracked_mask = sim.tracked
            n_tracked = tracked_mask.sum()

            # Upload only tracked particles
            tracked_pos = positions[tracked_mask]
            tracked_col = colors[tracked_mask]
            vbo_causal_pos.orphan(n_tracked * 2 * 4)
            vbo_causal_pos.write(tracked_pos.tobytes())
            vbo_causal_col.orphan(n_tracked * 3 * 4)
            vbo_causal_col.write(tracked_col.tobytes())

            # Decay
            causal_fbo2.use()
            ctx.clear(0, 0, 0)
            ctx.blend_func = moderngl.ONE, moderngl.ZERO
            causal_tex.use(0)
            prog_trail_decay['trail_tex'] = 0
            prog_trail_decay['decay'] = params['trail_decay']
            vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

            # Splat (only tracked particles)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            prog_splat['viewport_offset'] = (0.0, 0.0)
            prog_splat['viewport_scale'] = (1.0, 1.0)
            prog_splat['view_center'] = splat_center
            prog_splat['view_zoom'] = splat_zoom
            prog_splat['point_size'] = params['point_size']
            vao_causal_splat.render(moderngl.POINTS, vertices=n_tracked)

            # Swap
            causal_tex, causal_tex2 = causal_tex2, causal_tex
            causal_fbo, causal_fbo2 = causal_fbo2, causal_fbo

        # ── Screen rendering ──
        ctx.screen.use()
        ctx.clear(0, 0, 0)

        # Left half: live particles
        ctx.viewport = (0, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if params['show_neighbors'] and sim.nbr_ids is not None:
            lines = sim.get_neighbor_lines()
            n_line_verts = len(lines)
            if n_line_verts > 0:
                needed = n_line_verts * 2 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])
                vbo_line.write(lines.tobytes())
                prog_line['view_center'] = tuple(view_center)
                prog_line['view_zoom'] = view_zoom
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.08)
                vao_line.render(moderngl.LINES, vertices=n_line_verts)

        if params['show_radius']:
            circles = make_radius_circles(
                sim.pos.astype(np.float32), params['neighbor_radius'])
            n_circle_verts = len(circles)
            if n_circle_verts > 0:
                needed = n_circle_verts * 2 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line,
                                                [(vbo_line, '2f', 'in_pos')])
                vbo_line.write(circles.tobytes())
                prog_line['view_center'] = tuple(view_center)
                prog_line['view_zoom'] = view_zoom
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.04)
                vao_line.render(moderngl.LINES, vertices=n_circle_verts)

        prog_particle['viewport_offset'] = (0.0, 0.0)
        prog_particle['viewport_scale'] = (1.0, 1.0)
        prog_particle['view_center'] = tuple(view_center)
        prog_particle['view_zoom'] = view_zoom
        prog_particle['point_size'] = params['point_size']
        vao_particle.render(moderngl.POINTS)

        # Highlight tracked particles with white rings
        if sim.tracked.any():
            n_tracked = sim.tracked.sum()
            highlight_pos = positions[sim.tracked]
            highlight_col = np.full((n_tracked, 3), 1.0, dtype=np.float32)
            vbo_causal_pos.orphan(n_tracked * 2 * 4)
            vbo_causal_pos.write(highlight_pos.tobytes())
            vbo_causal_col.orphan(n_tracked * 3 * 4)
            vbo_causal_col.write(highlight_col.tobytes())
            prog_particle['viewport_offset'] = (0.0, 0.0)
            prog_particle['viewport_scale'] = (1.0, 1.0)
            prog_particle['point_size'] = params['point_size'] + 3.0
            vao_causal_particle.render(moderngl.POINTS, vertices=n_tracked)
            prog_particle['point_size'] = params['point_size']

        # Right half: display selected view
        ctx.viewport = (fb_w // 2, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        right_tex = vel_tex if params['right_view'] == 1 else \
                    causal_tex if params['right_view'] == 2 else trail_tex
        right_tex.use(0)
        prog_display['tex'] = 0
        if trail_zoom_on:
            prog_display['view_center'] = (0.5, 0.5)
            prog_display['view_zoom'] = 1.0
        else:
            prog_display['view_center'] = tuple(view_center)
            prog_display['view_zoom'] = view_zoom
        vao_display.render(moderngl.TRIANGLE_STRIP)

        if params['show_box']:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            if trail_zoom_on:
                prog_box['view_center'] = (0.5, 0.5)
                prog_box['view_zoom'] = 1.0
            else:
                prog_box['view_center'] = tuple(view_center)
                prog_box['view_zoom'] = view_zoom
            vao_box.render(moderngl.LINE_LOOP)

        # ── Divider line ──
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        divider = np.array([0.5, 0.0, 0.5, 1.0], dtype='f4')
        vbo_line.write(divider.tobytes())
        prog_line['view_center'] = (0.5, 0.5)
        prog_line['view_zoom'] = 1.0
        prog_line['line_color'] = (1.0, 1.0, 1.0, 0.5)
        vao_line.render(moderngl.LINES, vertices=2)

        # ── imgui overlay ──
        ctx.viewport = (0, 0, fb_w, fb_h)
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        status = "Running" if running_sim else "Paused"

        imgui.set_next_window_pos(imgui.ImVec2(10, 10),
                                  imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(280, 400),
                                   imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_bg_alpha(0.8)
        imgui.begin("Controls")

        # ── Status display with detailed timing ──
        imgui.text(f"Status: {status}")
        imgui.text(f"Step: {sim.step_count}  FPS: {fps:.0f}")
        imgui.text(f"Sim: {t_sim*1000:.1f}ms  "
                   f"grid: {sim._t_build*1000:.1f}ms  "
                   f"query: {sim._t_query*1000:.1f}ms  "
                   f"physics: {sim._t_physics*1000:.1f}ms")
        imgui.text(f"Neighbors/particle: {sim._n_nbrs}")
        imgui.separator()

        # ── Pause / Reset / Step / Record buttons ──
        label = "Resume" if not running_sim else "Pause"
        if imgui.button(label, imgui.ImVec2(80, 0)):
            running_sim = not running_sim
        imgui.same_line()
        if imgui.button("Reset", imgui.ImVec2(80, 0)):
            do_reset()
        imgui.same_line()
        if imgui.button("Step", imgui.ImVec2(50, 0)):
            running_sim = False
            sim.step()
        if rec_process is not None:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1.0, 0.2, 0.2, 1.0), "REC")
            imgui.text(f"Frames: {rec_frame_count}  "
                       f"Interval: {rec_interval:.1f}s")
            if imgui.button("Stop Rec", imgui.ImVec2(80, 0)):
                stop_recording()
        else:
            changed, rec_interval = imgui.drag_float(
                "Rec Interval", rec_interval, 0.05, 0.1, 10.0, "%.1fs")
            if imgui.button("Record", imgui.ImVec2(80, 0)):
                start_recording()
        imgui.separator()

        # ── Right panel view selector ──
        if imgui.radio_button("Trails", params['right_view'] == 0):
            params['right_view'] = 0
        imgui.same_line()
        if imgui.radio_button("Velocity", params['right_view'] == 1):
            params['right_view'] = 1
        imgui.same_line()
        if imgui.radio_button("Causal", params['right_view'] == 2):
            params['right_view'] = 2

        changed, v = imgui.checkbox("Show Neighbors", params['show_neighbors'])
        if changed:
            params['show_neighbors'] = v
        imgui.same_line()
        changed, v = imgui.checkbox("Show Radius", params['show_radius'])
        if changed:
            params['show_radius'] = v

        # ── Zoom / pan controls ──
        changed, v = imgui.checkbox("Box", params['show_box'])
        if changed:
            params['show_box'] = v
        imgui.same_line()
        if imgui.button("Reset View"):
            reset_view()
        if view_zoom != 1.0:
            imgui.text(f"Zoom: {view_zoom:.1f}x")
        imgui.separator()

        # ── Tracking controls ──
        imgui.text(f"Tracked: {sim.tracked.sum()} / {sim.n}")
        _track_modes = ["Frozen", "+ Neighbors", "Causal Spread"]
        changed, v = imgui.combo("Track Mode", params['track_mode'], _track_modes)
        if changed:
            params['track_mode'] = v
        if imgui.button("Clear Tracking"):
            sim.tracked[:] = False
            sim.tracked_seed[:] = False
            causal_fbo.use()
            ctx.clear(0, 0, 0)
            causal_fbo2.use()
            ctx.clear(0, 0, 0)
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0),
                           "Cmd+drag to select")
        imgui.separator()

        # ── Live parameters ──
        if imgui.collapsing_header(
                "Live Parameters",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            changed, v = imgui.drag_float("Step Size", params['step_size'], 0.0001, 0.001, 0.05, "%.4f")
            if changed:
                params['step_size'] = v
            changed, v = imgui.drag_int("Steps/Frame", params['steps_per_frame'], 0.5, 1, 100)
            if changed:
                params['steps_per_frame'] = v
            # Reuse neighbors checkbox (only shown when steps_per_frame > 1)
            if params['steps_per_frame'] > 1:
                changed, v = imgui.checkbox("Reuse Neighbors", params['reuse_neighbors'])
                if changed:
                    params['reuse_neighbors'] = v
            changed, v = imgui.drag_float("Repulsion", params['repulsion'], 0.0001, 0.0, 0.02, "%.4f")
            if changed:
                params['repulsion'] = v
            changed, v = imgui.drag_float("Dir Memory", params['dir_memory'], 0.005, 0.0, 0.99, "%.3f")
            if changed:
                params['dir_memory'] = v
            changed, v = imgui.drag_float("Social", params['social'], 0.0005, 0.0, 0.1, "%.4f")
            if changed:
                params['social'] = v
            changed, v = imgui.checkbox("Dist-Weighted", params['social_dist_weight'])
            if changed:
                params['social_dist_weight'] = v
            changed, v = imgui.checkbox("Pref-Weighted Dir", params['pref_weighted_dir'])
            if changed:
                params['pref_weighted_dir'] = v
            changed, v = imgui.checkbox("Inner Prod Weight", params['pref_inner_prod'])
            if changed:
                params['pref_inner_prod'] = v
            changed, v = imgui.checkbox("Inner Prod Avg", params['inner_prod_avg'])
            if changed:
                params['inner_prod_avg'] = v
            changed, v = imgui.checkbox("Dist-Weighted Pref", params['pref_dist_weight'])
            if changed:
                params['pref_dist_weight'] = v
            changed, v = imgui.checkbox("Best by Magnitude", params['best_by_magnitude'])
            if changed:
                params['best_by_magnitude'] = v
            changed, v = imgui.checkbox("Unit Prefs", params['unit_prefs'])
            if changed:
                params['unit_prefs'] = v
            imgui.separator()
            imgui.text("Crossover")
            changed, v = imgui.checkbox("Enable Crossover", params['crossover'])
            if changed:
                params['crossover'] = v
            if params['crossover']:
                changed, v = imgui.drag_int("Keep %", params['crossover_pct'], 1.0, 0, 100)
                if changed:
                    params['crossover_pct'] = v
                changed, v = imgui.drag_int("Interval", params['crossover_interval'], 0.5, 1, 1000)
                if changed:
                    params['crossover_interval'] = v
            changed, v = imgui.drag_float("Trail Decay", params['trail_decay'], 0.005, 0.8, 1.0, "%.3f")
            if changed:
                params['trail_decay'] = v
            changed, v = imgui.drag_float("Point Size", params['point_size'], 0.1, 1.0, 20.0, "%.1f")
            if changed:
                params['point_size'] = v
            _nbr_modes = ["KNN", "KNN + Radius", "Radius Only"]
            changed, v = imgui.combo("Neighbor Mode", params['neighbor_mode'], _nbr_modes)
            if changed:
                params['neighbor_mode'] = v
            _knn_methods = ["Hash Grid", "cKDTree (f64)", "cKDTree (f32)"]
            changed, v = imgui.combo("KNN Method", params['knn_method'], _knn_methods)
            if changed:
                params['knn_method'] = v
            _physics_engines = ["Numba", "NumPy (original)", "PyTorch"]
            changed, v = imgui.combo("Physics", params['physics_engine'], _physics_engines)
            if changed:
                params['physics_engine'] = v
            if params['physics_engine'] == 2:
                _precisions = ["f16", "bf16", "f32", "f64"]
                changed, v = imgui.combo("Precision", params['torch_precision'], _precisions)
                if changed:
                    params['torch_precision'] = v
                _devices = ["Auto (%s)" % _TORCH_DEVICE, "CPU"]
                changed, v = imgui.combo("Device", params['torch_device'], _devices)
                if changed:
                    params['torch_device'] = v
                if not _HAS_TORCH:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "torch not installed!")
            changed, v = imgui.checkbox("Use f64 pos", params['use_f64'])
            if changed:
                params['use_f64'] = v
                imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0), "(reset to apply)")
            if params['knn_method'] == 0:
                changed, v = imgui.checkbox("Debug KNN", params['debug_knn'])
                if changed:
                    params['debug_knn'] = v
            if params['neighbor_mode'] < 2:
                changed, v = imgui.drag_int("Neighbors", params['n_neighbors'], 0.5, 1, 30)
                if changed:
                    params['n_neighbors'] = v
            # Extended radius range: 0.001 lower bound for large N
            changed, v = imgui.drag_float("Radius", params['neighbor_radius'], 0.001, 0.001, 0.3, "%.4f")
            if changed:
                params['neighbor_radius'] = v

        # ── Reset-required parameters ──
        if imgui.collapsing_header("Reset-Required Params"):
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                               "(changes apply on Reset)")
            changed, v = imgui.combo(
                "Pos Init", params['pos_dist'], POS_DISTS)
            if changed:
                params['pos_dist'] = v
            if params['pos_dist'] == 1:  # Gaussian
                changed, v = imgui.drag_float("Gauss Sigma", params['gauss_sigma'], 0.005, 0.01, 1.0, "%.3f")
                if changed:
                    params['gauss_sigma'] = v
            changed, v = imgui.combo(
                "Pref Init", params['pref_dist'], PREF_DISTS)
            if changed:
                params['pref_dist'] = v
            changed, v = imgui.drag_int("Particles", params['num_particles'], 5.0, 2, 200000)
            if changed:
                params['num_particles'] = v
            changed, v = imgui.checkbox("Auto-scale", params['auto_scale'])
            if changed:
                params['auto_scale'] = v
            if params['auto_scale']:
                ref = auto_scale_ref
                scale = (ref['n'] / params['num_particles']) ** 0.5
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.9, 0.6, 1.0),
                    f"  step={ref['step_size']*scale:.4f}  "
                    f"radius={ref['radius']*scale:.4f}")
                imgui.text_colored(
                    imgui.ImVec4(0.5, 0.5, 0.5, 1.0),
                    f"  ref: N={ref['n']} step={ref['step_size']:.4f} "
                    f"r={ref['radius']:.3f}")
                if imgui.button("Set Reference", imgui.ImVec2(120, 0)):
                    auto_scale_ref['n'] = params['num_particles']
                    auto_scale_ref['step_size'] = params['step_size']
                    auto_scale_ref['radius'] = params['neighbor_radius']
            changed, v = imgui.drag_int("K (dims)", params['k'], 0.5, 1, 100)
            if changed:
                params['k'] = v

        if imgui.collapsing_header("Seed"):
            changed, v = imgui.checkbox("Fixed Seed", params['use_seed'])
            if changed:
                params['use_seed'] = v
            if params['use_seed']:
                changed, v = imgui.input_int("Seed##value", params['seed'])
                if changed:
                    params['seed'] = v

        imgui.end()

        # ── Selection rectangle overlay ──
        if selecting:
            draw_list = imgui.get_background_draw_list()
            draw_list.add_rect(
                imgui.ImVec2(sel_start[0], sel_start[1]),
                imgui.ImVec2(sel_end[0], sel_end[1]),
                imgui.get_color_u32(imgui.ImVec4(1, 1, 0, 0.8)),
                thickness=2.0)

        # Finalize and render imgui
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        # ── Timelapse capture ──
        if rec_process is not None:
            now = time.monotonic()
            if now - rec_last_time >= rec_interval:
                rec_last_time = now
                capture_frame()

        # ── Swap and FPS ──
        glfw.swap_buffers(window)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        glfw.set_window_title(window,
            f"Particles [{status}] Step:{sim.step_count} FPS:{fps:.0f}")

    # ── Cleanup ──
    stop_recording()
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context(imgui_ctx)
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == '__main__':
    main()
