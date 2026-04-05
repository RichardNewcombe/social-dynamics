"""
2D Preference-Directed Particle Simulation
===========================================

Refactored from sim_gpu_compute.py into separate modules:

  params.py        — Global parameter dict + constants
  shaders.py       — All GLSL shader strings
  spatial.py       — Spatial hash grid + neighbor finding (numba JIT)
  physics_numba.py — Numba JIT physics kernels
  physics_torch.py — PyTorch vectorized physics
  simulation.py    — Simulation class (state + stepping)
  renderer.py      — GLFW / moderngl / imgui window + render loop
  __main__.py      — Entry point
"""
