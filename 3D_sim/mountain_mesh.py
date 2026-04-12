"""
Mountain Mesh Generation
========================

Samples a GaussianPeakLandscape (and optionally a CostLandscape) into
a triangle mesh suitable for ModernGL rendering.

The mesh lives in a coordinate system where:
  X = pref[0]  mapped from [-1, 1] → [0, 1]
  Y = pref[1]  mapped from [-1, 1] → [0, 1]
  Z = fitness   scaled to fit within [0, z_scale]

This aligns with the 3D sim's [0, 1]^3 unit cube convention.
"""

import numpy as np


def generate_mountain_mesh(landscape, resolution=64, z_scale=0.5,
                           pref_dims=2, other_pref_val=0.0):
    """Sample a GaussianPeakLandscape into a triangle mesh.

    Args:
        landscape: GaussianPeakLandscape instance with .fitness() and .k
        resolution: grid resolution (vertices per side)
        z_scale: maximum Z height for the mesh
        pref_dims: number of preference dimensions (uses first 2 for XY)
        other_pref_val: value for pref dims beyond 0 and 1

    Returns:
        vertices: (N_verts, 3) float32 positions
        normals: (N_verts, 3) float32 normals
        colors_fitness: (N_verts, 3) float32 RGB colors based on fitness
        colors_cost: (N_verts, 3) float32 RGB colors based on cost (if cost_landscape given)
        indices: (N_tris * 3,) uint32 triangle indices
        fitness_grid: (resolution, resolution) float32 raw fitness values
    """
    k = landscape.k if hasattr(landscape, 'k') else pref_dims

    # Sample grid in preference space [-1, 1]^2
    lin = np.linspace(-1.0, 1.0, resolution)
    gx, gy = np.meshgrid(lin, lin)  # (res, res) each

    # Build preference array: (res*res, k)
    n_pts = resolution * resolution
    prefs = np.full((n_pts, k), other_pref_val, dtype=np.float64)
    prefs[:, 0] = gx.ravel()
    prefs[:, 1] = gy.ravel()

    # Evaluate fitness
    fitness_vals, _ = landscape.fitness(prefs)  # (n_pts,)
    fitness_grid = fitness_vals.reshape(resolution, resolution).astype(np.float32)

    # Normalize fitness to [0, z_scale]
    f_min = fitness_vals.min()
    f_max = fitness_vals.max()
    f_range = max(f_max - f_min, 1e-8)
    z_vals = ((fitness_vals - f_min) / f_range * z_scale).astype(np.float32)

    # Map pref coords to [0, 1] to match unit cube
    x_vals = ((gx.ravel() + 1.0) * 0.5).astype(np.float32)
    y_vals = ((gy.ravel() + 1.0) * 0.5).astype(np.float32)

    # Build vertices (N, 3)
    vertices = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)

    # Compute normals via finite differences on the grid
    normals = _compute_grid_normals(vertices.reshape(resolution, resolution, 3))
    normals = normals.reshape(-1, 3).astype(np.float32)

    # Fitness colormap: blue (low) → green (mid) → red (high)
    t = ((fitness_vals - f_min) / f_range).astype(np.float32)
    colors_fitness = _colormap_terrain(t)

    # Build triangle indices
    indices = _build_grid_indices(resolution)

    return vertices, normals, colors_fitness, indices, fitness_grid


def generate_cost_colors(cost_landscape, landscape_k, resolution=64,
                         other_pref_val=0.0):
    """Sample a CostLandscape and return per-vertex cost colors.

    Args:
        cost_landscape: CostLandscape instance with .cost()
        landscape_k: number of preference dimensions
        resolution: must match the mountain mesh resolution
        other_pref_val: value for pref dims beyond 0 and 1

    Returns:
        colors_cost: (N_verts, 3) float32 RGB colors
        cost_grid: (resolution, resolution) float32 raw cost values
    """
    lin = np.linspace(-1.0, 1.0, resolution)
    gx, gy = np.meshgrid(lin, lin)
    n_pts = resolution * resolution
    prefs = np.full((n_pts, landscape_k), other_pref_val, dtype=np.float64)
    prefs[:, 0] = gx.ravel()
    prefs[:, 1] = gy.ravel()

    cost_vals = cost_landscape.cost(prefs)  # (n_pts,)
    cost_grid = cost_vals.reshape(resolution, resolution).astype(np.float32)

    # Normalize
    c_min = cost_vals.min()
    c_max = cost_vals.max()
    c_range = max(c_max - c_min, 1e-8)
    t = ((cost_vals - c_min) / c_range).astype(np.float32)

    colors_cost = _colormap_cost(t)
    return colors_cost, cost_grid


def project_particles_to_surface(prefs, landscape, z_scale=0.5):
    """Project particle preferences onto the mountain surface.

    Args:
        prefs: (N, k) particle preferences in [-1, 1]
        landscape: GaussianPeakLandscape
        z_scale: must match the mesh z_scale

    Returns:
        positions_3d: (N, 3) float32 positions on the mountain surface
    """
    fitness_vals, _ = landscape.fitness(prefs)

    f_min_approx = 0.0  # landscapes have non-negative Gaussian peaks
    f_max_approx = float(landscape.heights.max())  # max peak height
    f_range = max(f_max_approx - f_min_approx, 1e-8)

    z = ((fitness_vals - f_min_approx) / f_range * z_scale).astype(np.float32)
    z = np.clip(z, 0, z_scale)

    x = ((prefs[:, 0] + 1.0) * 0.5).astype(np.float32)
    y = ((prefs[:, 1] + 1.0) * 0.5).astype(np.float32)

    return np.column_stack([x, y, z])


def _compute_grid_normals(grid_pos):
    """Compute per-vertex normals from a (res, res, 3) position grid.

    Uses central differences for interior, forward/backward at edges.
    """
    res = grid_pos.shape[0]
    normals = np.zeros_like(grid_pos)

    # Partial derivatives
    # dx: along axis 1 (columns = X direction)
    dx = np.zeros_like(grid_pos)
    dx[:, 1:-1] = grid_pos[:, 2:] - grid_pos[:, :-2]
    dx[:, 0] = grid_pos[:, 1] - grid_pos[:, 0]
    dx[:, -1] = grid_pos[:, -1] - grid_pos[:, -2]

    # dy: along axis 0 (rows = Y direction)
    dy = np.zeros_like(grid_pos)
    dy[1:-1, :] = grid_pos[2:, :] - grid_pos[:-2, :]
    dy[0, :] = grid_pos[1, :] - grid_pos[0, :]
    dy[-1, :] = grid_pos[-1, :] - grid_pos[-2, :]

    # Normal = cross(dx, dy), then normalize
    normals = np.cross(dx.reshape(-1, 3), dy.reshape(-1, 3))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    # Ensure normals point "up" (positive Z)
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    return normals.reshape(res, res, 3)


def _build_grid_indices(resolution):
    """Build triangle indices for a resolution x resolution grid.

    Two triangles per quad, counter-clockwise winding.
    """
    indices = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            tl = row * resolution + col
            tr = tl + 1
            bl = (row + 1) * resolution + col
            br = bl + 1
            # Two triangles per quad
            indices.extend([tl, bl, tr, tr, bl, br])
    return np.array(indices, dtype=np.uint32)


def _colormap_terrain(t):
    """Terrain colormap: deep blue → green → yellow → red → white.

    t: (N,) float32 in [0, 1]
    Returns: (N, 3) float32 RGB
    """
    colors = np.zeros((len(t), 3), dtype=np.float32)

    # 0.0-0.25: deep blue to green
    mask = t < 0.25
    s = t[mask] / 0.25
    colors[mask, 0] = 0.0
    colors[mask, 1] = s * 0.6
    colors[mask, 2] = 0.3 * (1.0 - s)

    # 0.25-0.5: green to yellow
    mask = (t >= 0.25) & (t < 0.5)
    s = (t[mask] - 0.25) / 0.25
    colors[mask, 0] = s * 0.8
    colors[mask, 1] = 0.6 + s * 0.2
    colors[mask, 2] = 0.0

    # 0.5-0.75: yellow to red
    mask = (t >= 0.5) & (t < 0.75)
    s = (t[mask] - 0.5) / 0.25
    colors[mask, 0] = 0.8 + s * 0.2
    colors[mask, 1] = 0.8 * (1.0 - s)
    colors[mask, 2] = 0.0

    # 0.75-1.0: red to white
    mask = t >= 0.75
    s = (t[mask] - 0.75) / 0.25
    colors[mask, 0] = 1.0
    colors[mask, 1] = s * 0.8
    colors[mask, 2] = s * 0.8

    return colors


def _colormap_cost(t):
    """Cost colormap: transparent green (low cost) → yellow → bright red (high cost).

    t: (N,) float32 in [0, 1]
    Returns: (N, 3) float32 RGB
    """
    colors = np.zeros((len(t), 3), dtype=np.float32)

    # 0.0-0.5: green to yellow
    mask = t < 0.5
    s = t[mask] / 0.5
    colors[mask, 0] = s
    colors[mask, 1] = 0.8
    colors[mask, 2] = 0.0

    # 0.5-1.0: yellow to red
    mask = t >= 0.5
    s = (t[mask] - 0.5) / 0.5
    colors[mask, 0] = 1.0
    colors[mask, 1] = 0.8 * (1.0 - s)
    colors[mask, 2] = 0.0

    return colors
