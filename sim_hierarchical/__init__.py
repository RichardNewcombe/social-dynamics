"""
Hierarchical Particle Simulation — L0 individual + L1 group physics
====================================================================

Two-level dynamics: standard per-dimension best-neighbor physics on
all particles (L0), plus emergent group-level movement (L1) for
bonded clusters. Pure NumPy/SciPy — runs on any platform.

Usage:
    python -m sim_hierarchical.headless
    python -m sim_hierarchical.headless --steps 5000 --particles 1000 --l1-weight 0.1
"""
