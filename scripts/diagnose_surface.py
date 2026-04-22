#!/usr/bin/env python3
"""
Diagnose off-surface particle bug.

This script mimics exactly what the 3D visualizer does:
1. Generate the mountain mesh using generate_mountain_mesh()
2. Run sim.step() + gradient nudge + project_particles_to_surface()
3. Compare particle Y positions to the mesh surface height at the same XZ

The key question: does project_particles_to_surface() produce Y values
that match the mesh surface? If not, why?
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from sim_2d_exp.params import params as _global_params
from sim_2d_exp.simulation import Simulation
from experiments.landscape import make_default_landscape
# 3D_sim is not a valid Python package name, so add its parent to path
# and import the module directly
import importlib
_3d_sim_dir = os.path.join(_pkg_root, '3D_sim')
if _3d_sim_dir not in sys.path:
    sys.path.insert(0, _3d_sim_dir)
import mountain_mesh as _mm
generate_mountain_mesh = _mm.generate_mountain_mesh
project_particles_to_surface = _mm.project_particles_to_surface
_landscape_range_cache = _mm._landscape_range_cache

# Rename the 3D_sim package import
sys.path.insert(0, _pkg_root)

K = 3
LANDSCAPE = make_default_landscape(k=K)


def diagnose():
    """Run the exact same code paths as the 3D visualizer and check for mismatches."""

    z_scale = 0.5
    resolution = 64

    # Step 1: Generate the mesh (same as build_mountain_mesh in main.py)
    verts, normals, colors_f, indices, fitness_grid = generate_mountain_mesh(
        LANDSCAPE, resolution=resolution, z_scale=z_scale, pref_dims=max(K, 2))

    print("=== MESH GENERATION ===")
    print(f"Mesh vertices shape: {verts.shape}")
    print(f"Mesh X range: [{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}]")
    print(f"Mesh Y range: [{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}]")
    print(f"Mesh Z range: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]")
    print(f"Fitness grid range: [{fitness_grid.min():.4f}, {fitness_grid.max():.4f}]")

    # What f_min, f_max did the mesh use?
    # The mesh computes its own f_min, f_max from the full grid
    mesh_f_min = fitness_grid.min()
    mesh_f_max = fitness_grid.max()
    mesh_f_range = mesh_f_max - mesh_f_min
    print(f"Mesh f_min={mesh_f_min:.6f}, f_max={mesh_f_max:.6f}, f_range={mesh_f_range:.6f}")

    # Step 2: Check what project_particles_to_surface uses for f_min, f_max
    # Clear the cache to see what it computes
    _landscape_range_cache.clear()

    # Create some test prefs at known positions
    test_prefs = np.array([
        [0.72, 0.78, 0.65],  # Near summit
        [-0.10, 0.05, 0.0],  # Near deceptive peak
        [0.0, 0.0, 0.0],     # Center
        [-1.0, -1.0, 0.0],   # Corner
        [1.0, 1.0, 0.0],     # Other corner
    ], dtype=np.float64)

    projected = project_particles_to_surface(test_prefs, LANDSCAPE, z_scale=z_scale)

    print("\n=== PROJECT_PARTICLES_TO_SURFACE ===")
    # Check what range it cached
    lid = id(LANDSCAPE)
    if lid in _landscape_range_cache:
        proj_f_min, proj_f_max = _landscape_range_cache[lid]
        print(f"Projection cached f_min={proj_f_min:.6f}, f_max={proj_f_max:.6f}")
        print(f"Projection f_range={proj_f_max - proj_f_min:.6f}")
    else:
        print("No cache entry found!")

    # Evaluate fitness at test points
    fitness_at_test, _ = LANDSCAPE.fitness(test_prefs)

    print("\nTest point comparison:")
    print(f"{'Pref[0]':>8} {'Pref[1]':>8} {'Fitness':>8} {'Proj_X':>8} {'Proj_Y':>8} {'Proj_Z':>8}")
    for i in range(len(test_prefs)):
        print(f"{test_prefs[i, 0]:8.3f} {test_prefs[i, 1]:8.3f} "
              f"{fitness_at_test[i]:8.4f} "
              f"{projected[i, 0]:8.4f} {projected[i, 1]:8.4f} {projected[i, 2]:8.4f}")

    # Step 3: For each test point, find where it would be on the mesh
    # The mesh samples at resolution points in [-1,1]^2
    # Find the nearest mesh vertex for each test point
    lin = np.linspace(-1.0, 1.0, resolution)
    mesh_x = ((lin + 1.0) * 0.5).astype(np.float32)  # mapped to [0,1]

    print("\n=== MESH vs PROJECTION COMPARISON ===")
    print("For each test point, compare projected Y to mesh Y at same XZ:")
    for i in range(len(test_prefs)):
        px, py, pz = projected[i]
        # Find nearest mesh grid indices
        ix = np.argmin(np.abs(mesh_x - px))
        iz = np.argmin(np.abs(mesh_x - pz))
        mesh_y = verts[ix * resolution + iz, 1]
        diff = py - mesh_y
        print(f"  Point {i}: proj_Y={py:.5f}, mesh_Y={mesh_y:.5f}, diff={diff:.5f}")

    # Step 4: The KEY diagnostic — check if the f_min/f_max differ
    print("\n=== ROOT CAUSE CHECK ===")
    print(f"Mesh uses f_min={mesh_f_min:.6f}, f_max={mesh_f_max:.6f}")
    if lid in _landscape_range_cache:
        proj_f_min, proj_f_max = _landscape_range_cache[lid]
        print(f"Projection uses f_min={proj_f_min:.6f}, f_max={proj_f_max:.6f}")
        if abs(mesh_f_min - proj_f_min) > 1e-4 or abs(mesh_f_max - proj_f_max) > 1e-4:
            print("*** MISMATCH! Mesh and projection use different normalization ranges! ***")
            print("This means particles will be at different heights than the mesh surface.")
        else:
            print("Ranges match — normalization is consistent.")

    # Step 5: Check with K > 2 — the mesh sets other dims to 0,
    # but particles have non-zero values in dims 2+
    print("\n=== K > 2 DIMENSION CHECK ===")
    print(f"K = {K}")
    print("The mesh evaluates fitness with dims 2+ set to other_pref_val=0.0")
    print("But particles have random values in dims 2+!")

    # Compare fitness at (0.5, 0.5, 0.0) vs (0.5, 0.5, 0.3)
    p1 = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    p2 = np.array([[0.5, 0.5, 0.3]], dtype=np.float64)
    p3 = np.array([[0.5, 0.5, 0.7]], dtype=np.float64)
    p4 = np.array([[0.5, 0.5, -0.5]], dtype=np.float64)

    f1, _ = LANDSCAPE.fitness(p1)
    f2, _ = LANDSCAPE.fitness(p2)
    f3, _ = LANDSCAPE.fitness(p3)
    f4, _ = LANDSCAPE.fitness(p4)

    print(f"  fitness(0.5, 0.5, 0.0) = {f1[0]:.6f}")
    print(f"  fitness(0.5, 0.5, 0.3) = {f2[0]:.6f}")
    print(f"  fitness(0.5, 0.5, 0.7) = {f3[0]:.6f}")
    print(f"  fitness(0.5, 0.5,-0.5) = {f4[0]:.6f}")

    if abs(f1[0] - f2[0]) > 0.01 or abs(f1[0] - f3[0]) > 0.01:
        print("*** SIGNIFICANT FITNESS DIFFERENCE from dim 2! ***")
        print("The mesh shows fitness at dim2=0, but particles have dim2!=0.")
        print("So particles evaluate to different fitness than the mesh surface,")
        print("making them appear above or below the mountain!")
    else:
        print("Dim 2 has minimal effect on fitness.")

    # Step 6: Quantify the effect across many random particles
    print("\n=== POPULATION-LEVEL ANALYSIS ===")
    rng = np.random.default_rng(42)
    N = 500
    random_prefs = rng.uniform(-1, 1, (N, K)).astype(np.float64)

    # Fitness with actual dim2 values
    fitness_actual, _ = LANDSCAPE.fitness(random_prefs)

    # Fitness with dim2 set to 0 (what the mesh shows)
    prefs_dim2_zero = random_prefs.copy()
    prefs_dim2_zero[:, 2:] = 0.0
    fitness_mesh_equiv, _ = LANDSCAPE.fitness(prefs_dim2_zero)

    diff = fitness_actual - fitness_mesh_equiv
    print(f"Fitness difference (actual - mesh_equiv) when dim2 varies:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Std:  {diff.std():.6f}")
    print(f"  Min:  {diff.min():.6f}")
    print(f"  Max:  {diff.max():.6f}")
    print(f"  |diff| > 0.05: {(np.abs(diff) > 0.05).sum()}/{N} particles")
    print(f"  |diff| > 0.10: {(np.abs(diff) > 0.10).sum()}/{N} particles")

    # Convert to Y offset in render space
    if lid in _landscape_range_cache:
        proj_f_min, proj_f_max = _landscape_range_cache[lid]
        proj_f_range = max(proj_f_max - proj_f_min, 1e-8)
    else:
        proj_f_range = mesh_f_range
    y_diff = diff / proj_f_range * z_scale
    print(f"\nY offset in render space (z_scale={z_scale}):")
    print(f"  Mean: {y_diff.mean():.6f}")
    print(f"  Std:  {y_diff.std():.6f}")
    print(f"  Min:  {y_diff.min():.6f}")
    print(f"  Max:  {y_diff.max():.6f}")

    # Step 7: Also check the f_min/f_max sampling resolution
    print("\n=== SAMPLING RESOLUTION CHECK ===")
    print("Mesh uses resolution=64 (4096 points) to compute f_min/f_max")
    print("Projection uses resolution=50 (2500 points) to compute f_min/f_max")
    # Re-sample at 50 resolution
    _lin50 = np.linspace(-1.0, 1.0, 50)
    _gx50, _gy50 = np.meshgrid(_lin50, _lin50, indexing='ij')
    _sample50 = np.full((50 * 50, K), 0.0, dtype=np.float64)
    _sample50[:, 0] = _gx50.ravel()
    _sample50[:, 1] = _gy50.ravel()
    _fvals50, _ = LANDSCAPE.fitness(_sample50)
    print(f"  50x50 grid: f_min={_fvals50.min():.6f}, f_max={_fvals50.max():.6f}")

    _lin64 = np.linspace(-1.0, 1.0, 64)
    _gx64, _gy64 = np.meshgrid(_lin64, _lin64, indexing='ij')
    _sample64 = np.full((64 * 64, K), 0.0, dtype=np.float64)
    _sample64[:, 0] = _gx64.ravel()
    _sample64[:, 1] = _gy64.ravel()
    _fvals64, _ = LANDSCAPE.fitness(_sample64)
    print(f"  64x64 grid: f_min={_fvals64.min():.6f}, f_max={_fvals64.max():.6f}")

    # Step 8: Render diagnostic figure
    print("\n=== RENDERING DIAGNOSTIC FIGURE ===")
    render_diagnostic(random_prefs, fitness_actual, fitness_mesh_equiv, diff, y_diff)


def render_diagnostic(prefs, fitness_actual, fitness_mesh_equiv, diff, y_diff):
    """Render diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Off-Surface Particle Diagnostic', fontsize=16, fontweight='bold')

    # 1. Scatter: fitness_actual vs fitness_mesh_equiv
    ax = axes[0, 0]
    ax.scatter(fitness_mesh_equiv, fitness_actual, s=4, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='perfect match')
    ax.set_xlabel('Fitness (mesh equiv, dim2=0)')
    ax.set_ylabel('Fitness (actual, dim2 varies)')
    ax.set_title('Fitness: Actual vs Mesh-Equivalent')
    ax.legend()
    ax.set_aspect('equal')

    # 2. Histogram of fitness difference
    ax = axes[0, 1]
    ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitness difference (actual - mesh)')
    ax.set_ylabel('Count')
    ax.set_title('Fitness Difference Distribution')

    # 3. Histogram of Y offset
    ax = axes[0, 2]
    ax.hist(y_diff, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Y offset in render space')
    ax.set_ylabel('Count')
    ax.set_title('Render Y Offset Distribution')

    # 4. Scatter: pref[2] vs fitness difference
    ax = axes[1, 0]
    ax.scatter(prefs[:, 2], diff, s=4, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('pref[2] value')
    ax.set_ylabel('Fitness difference')
    ax.set_title('Dim 2 Value vs Fitness Difference')

    # 5. Top-down view colored by Y offset
    ax = axes[1, 1]
    sc = ax.scatter(prefs[:, 0], prefs[:, 1], c=y_diff, cmap='RdBu_r',
                    s=6, alpha=0.7, vmin=-0.1, vmax=0.1)
    plt.colorbar(sc, ax=ax, label='Y offset')
    ax.set_xlabel('pref[0]')
    ax.set_ylabel('pref[1]')
    ax.set_title('Top-Down: Y Offset (red=above, blue=below)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    # 6. Side view showing particles vs surface
    ax = axes[1, 2]
    # Draw a 1D slice of the mesh at pref[1]=0
    x_line = np.linspace(-1, 1, 200)
    mesh_prefs = np.zeros((200, K), dtype=np.float64)
    mesh_prefs[:, 0] = x_line
    mesh_fitness, _ = LANDSCAPE.fitness(mesh_prefs)
    ax.plot(x_line, mesh_fitness, 'k-', linewidth=2, label='Mesh surface (dim2=0)')

    # Show particles near pref[1]=0 (within ±0.1)
    mask = np.abs(prefs[:, 1]) < 0.1
    if mask.sum() > 0:
        ax.scatter(prefs[mask, 0], fitness_actual[mask], c='red', s=15,
                   alpha=0.7, label=f'Particles (actual fitness, n={mask.sum()})')
        ax.scatter(prefs[mask, 0], fitness_mesh_equiv[mask], c='blue', s=15,
                   alpha=0.7, marker='x', label='Same particles (dim2=0 fitness)')
    ax.set_xlabel('pref[0]')
    ax.set_ylabel('Fitness')
    ax.set_title('Side View Slice (pref[1] ≈ 0)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(_pkg_root, 'results', 'surface_diagnostic.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    diagnose()
