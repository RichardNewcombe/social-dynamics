#!/usr/bin/env python3
"""Verify that the mountain_mesh fix puts particles on the surface."""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

_3d_sim_dir = os.path.join(_pkg_root, '3D_sim')
if _3d_sim_dir not in sys.path:
    sys.path.insert(0, _3d_sim_dir)

# Force reimport
if 'mountain_mesh' in sys.modules:
    del sys.modules['mountain_mesh']

import mountain_mesh as mm
from experiments.landscape import make_default_landscape

K = 3
LANDSCAPE = make_default_landscape(k=K)

z_scale = 0.5
resolution = 64

# Clear cache
mm._landscape_range_cache.clear()

# Generate mesh
verts, normals, colors_f, indices, fitness_grid = mm.generate_mountain_mesh(
    LANDSCAPE, resolution=resolution, z_scale=z_scale, pref_dims=max(K, 2))

print(f"Mesh f_min/f_max from cache: {mm._landscape_range_cache[id(LANDSCAPE)]}")

# Generate random particles with K=3
rng = np.random.default_rng(42)
N = 1000
random_prefs = rng.uniform(-1, 1, (N, K)).astype(np.float64)

# Project using the fixed function
projected = mm.project_particles_to_surface(random_prefs, LANDSCAPE, z_scale=z_scale)

# For each particle, find the nearest mesh vertex and compare Y
lin = np.linspace(-1.0, 1.0, resolution)
mesh_x_mapped = (lin + 1.0) * 0.5  # [0, 1]

offsets = []
for i in range(N):
    px, py, pz = projected[i]
    ix = np.argmin(np.abs(mesh_x_mapped - px))
    iz = np.argmin(np.abs(mesh_x_mapped - pz))
    mesh_y = verts[ix * resolution + iz, 1]
    offsets.append(py - mesh_y)

offsets = np.array(offsets)

print(f"\n=== PARTICLE-TO-MESH Y OFFSET (N={N}) ===")
print(f"Mean:  {offsets.mean():.6f}")
print(f"Std:   {offsets.std():.6f}")
print(f"Min:   {offsets.min():.6f}")
print(f"Max:   {offsets.max():.6f}")
print(f"|off| > 0.01: {(np.abs(offsets) > 0.01).sum()}/{N}")
print(f"|off| > 0.005: {(np.abs(offsets) > 0.005).sum()}/{N}")
print(f"|off| > 0.001: {(np.abs(offsets) > 0.001).sum()}/{N}")

# The remaining small offsets are from grid interpolation (nearest vertex, not bilinear)
# Let's also check by evaluating fitness at the exact 2D slice
prefs_2d = np.full((N, K), 0.0, dtype=np.float64)
prefs_2d[:, 0] = random_prefs[:, 0]
prefs_2d[:, 1] = random_prefs[:, 1]
fitness_2d, _ = LANDSCAPE.fitness(prefs_2d)

f_min, f_max = mm._landscape_range_cache[id(LANDSCAPE)]
f_range = max(f_max - f_min, 1e-8)
expected_y = ((fitness_2d - f_min) / f_range * z_scale).astype(np.float32)
expected_y = np.clip(expected_y, 0, z_scale)

exact_offsets = projected[:, 1] - expected_y
print(f"\n=== EXACT OFFSET (projection vs expected) ===")
print(f"Max |offset|: {np.abs(exact_offsets).max():.10f}")
print(f"All zero: {np.allclose(exact_offsets, 0, atol=1e-6)}")

# Render comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fix Verification: Particles on Mountain Surface', fontsize=14, fontweight='bold')

# 1. Histogram of offsets (nearest mesh vertex)
ax = axes[0]
ax.hist(offsets, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Y offset (particle - nearest mesh vertex)')
ax.set_ylabel('Count')
ax.set_title(f'Grid Interpolation Error\n(max |off|={np.abs(offsets).max():.4f})')

# 2. Side view slice at pref[1]≈0
ax = axes[1]
x_line = np.linspace(-1, 1, 200)
mesh_prefs = np.zeros((200, K), dtype=np.float64)
mesh_prefs[:, 0] = x_line
mesh_fitness, _ = LANDSCAPE.fitness(mesh_prefs)
f_min, f_max = mm._landscape_range_cache[id(LANDSCAPE)]
f_range = max(f_max - f_min, 1e-8)
mesh_y_line = (mesh_fitness - f_min) / f_range * z_scale
ax.plot((x_line + 1) * 0.5, mesh_y_line, 'k-', linewidth=2, label='Mesh surface')

mask = np.abs(random_prefs[:, 1]) < 0.1
if mask.sum() > 0:
    ax.scatter(projected[mask, 0], projected[mask, 1], c='red', s=15,
               alpha=0.7, label=f'Particles (n={mask.sum()})', zorder=5)
ax.set_xlabel('X (mapped pref[0])')
ax.set_ylabel('Y (fitness)')
ax.set_title('Side View: pref[1] ≈ 0')
ax.legend()

# 3. 3D scatter showing particles on surface
ax = fig.add_subplot(1, 3, 3, projection='3d')
# Subsample mesh for wireframe
step = max(1, resolution // 20)
mesh_reshaped = verts.reshape(resolution, resolution, 3)
ax.plot_wireframe(
    mesh_reshaped[::step, ::step, 0],
    mesh_reshaped[::step, ::step, 2],
    mesh_reshaped[::step, ::step, 1],
    color='gray', alpha=0.3, linewidth=0.5)
# Particles
subset = slice(0, min(N, 300))
ax.scatter(projected[subset, 0], projected[subset, 2], projected[subset, 1],
           c='red', s=4, alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y (fitness)')
ax.set_title('3D View: Particles on Surface')
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
out = os.path.join(_pkg_root, 'results', 'fix_verification.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out}")
plt.close()
