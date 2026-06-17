"""
Mountain Mesh Generation
========================

Generates triangle meshes for the 3D visualiser:
1. The knowledge manifold surface (solid, what particles walk on)
2. The hidden fitness surface (wireframe ghost overlay)

Both are 2D grids over (pref0, pref1) mapped to 3D (X, Y, Z):
  X = (pref0 + 1) / 2          in [0, 1]
  Z = (pref1 + 1) / 2          in [0, 1]
  Y = height * z_scale          in [0, z_scale]

Also provides particle projection onto the knowledge surface.
"""

import numpy as np


def generate_surface_mesh(height_grid, z_scale=0.5, color=(0.3, 0.6, 0.3)):
    """Generate a triangle mesh from a 2D height grid.

    Parameters
    ----------
    height_grid : ndarray (G, G)
        Height values in [0, 1].
    z_scale : float
        Vertical scale factor.
    color : tuple (r, g, b)
        Base colour for the mesh.

    Returns
    -------
    vertices : ndarray (V, 3) float32
    normals : ndarray (V, 3) float32
    colors : ndarray (V, 3) float32
    indices : ndarray (T*3,) uint32
    """
    G = height_grid.shape[0]
    xs = np.linspace(0, 1, G)
    zs = np.linspace(0, 1, G)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    gy = height_grid * z_scale

    vertices = np.column_stack([
        gx.ravel(), gy.ravel(), gz.ravel(),
    ]).astype(np.float32)

    # Normals via finite differences
    cell = 1.0 / (G - 1)
    dx = np.gradient(gy, cell, axis=0)
    dz = np.gradient(gy, cell, axis=1)
    nx = -dx
    ny = np.ones_like(dx)
    nz = -dz
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    mag = np.maximum(mag, 1e-8)
    normals = np.column_stack([
        (nx / mag).ravel(), (ny / mag).ravel(), (nz / mag).ravel(),
    ]).astype(np.float32)

    brightness = 0.4 + 0.6 * height_grid.ravel()
    colors = np.column_stack([
        np.full(G * G, color[0]) * brightness,
        np.full(G * G, color[1]) * brightness,
        np.full(G * G, color[2]) * brightness,
    ]).astype(np.float32)

    # Two triangles per quad
    indices = []
    for i in range(G - 1):
        for j in range(G - 1):
            v00 = i * G + j
            v10 = (i + 1) * G + j
            v01 = i * G + (j + 1)
            v11 = (i + 1) * G + (j + 1)
            indices.extend([v00, v10, v01, v10, v11, v01])
    indices = np.array(indices, dtype=np.uint32)

    return vertices, normals, colors, indices


def generate_wireframe_indices(G):
    """Generate line indices for a wireframe grid overlay.

    Parameters
    ----------
    G : int
        Grid resolution.

    Returns
    -------
    indices : ndarray (L*2,) uint32
    """
    lines = []
    step = max(1, G // 16)
    for i in range(0, G, step):
        for j in range(G - 1):
            lines.extend([i * G + j, i * G + (j + 1)])
    for j in range(0, G, step):
        for i in range(G - 1):
            lines.extend([i * G + j, (i + 1) * G + j])
    return np.array(lines, dtype=np.uint32)


def project_particles_to_knowledge_surface(knowledge_field, prefs, z_scale=0.5):
    """Project particles onto the knowledge manifold surface.

    Parameters
    ----------
    knowledge_field : KnowledgeField
    prefs : ndarray (N, K)
        Dims 0, 1 are skill coords.
    z_scale : float

    Returns
    -------
    positions : ndarray (N, 3) float32
    """
    x = (prefs[:, 0] + 1.0) * 0.5
    z = (prefs[:, 1] + 1.0) * 0.5
    heights = knowledge_field.sample_knowledge(prefs[:, 0], prefs[:, 1])
    y = heights * z_scale
    # Clamp to [0, 1-eps] to avoid cKDTree boxsize boundary issues
    result = np.column_stack([x, y, z]).astype(np.float32)
    np.clip(result, 0.0, 1.0 - 1e-7, out=result)
    return result
