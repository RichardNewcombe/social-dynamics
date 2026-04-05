"""
Global parameters and constants for the 2D particle simulation.
"""

import numpy as np

SPACE = 1.0

# ── Distribution labels for GUI combos ──
POS_DISTS = ["Uniform", "Gaussian"]
PREF_DISTS = ["Uniform [-1,1]", "Gaussian", "Sparse ±1", "Unit Normalized",
              "Binary d0 + noise"]
BEST_MODES = ["Default", "Max Magnitude", "Same-Sign Max Mag"]
SOCIAL_MODES = ["Uniform", "Quiet-Dim Diff"]
VIS_PREF_SOURCES = ["Signal", "Response"]
PREF_COLOR_MODES = ["RGB", "Dim2 Heat", "HSV"]
# right_view: 0=Trails, 1=Velocity, 2=Causal, 3=Pref 2D, 4=Pref 3D

# ── Central mutable parameter dict ──
# Read by simulation + physics, mutated by imgui GUI + keyboard callbacks.
params = dict(
    num_particles=2000,
    k=3,
    n_neighbors=21,
    step_size=0.005,
    steps_per_frame=1,
    repulsion=0.0,
    dir_memory=0.0,
    social=0.0,
    social_mode=0,          # 0=Uniform, 1=Quiet-Dim Differentiation
    social_dist_weight=False,
    pref_weighted_dir=False,
    pref_inner_prod=False,
    inner_prod_avg=False,
    pref_dist_weight=False,
    best_mode=0,            # 0=Default, 1=Max Magnitude, 2=Same-Sign Max Mag
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
    knn_method=0,       # 0=Hash Grid, 1=cKDTree (f64 pos), 2=cKDTree (f32 pos)
    use_f64=True,       # True = float64 positions
    physics_engine=0,   # 0=Numba, 1=NumPy (original), 2=PyTorch
    torch_precision=2,  # 0=f16, 1=bf16, 2=f32, 3=f64
    torch_device=0,     # 0=auto (mps if available), 1=cpu
    unit_prefs=False,
    track_mode=2,       # 0=Frozen (seed only), 1=+Neighbors, 2=Causal Spread
    crossover=False,
    crossover_pct=50,
    crossover_interval=1,
    binary_noise_eps=0.1,   # max magnitude of noise dims for "Binary d0 + noise"
    pref_color_mode=0,      # 0=RGB, 1=Dim2 Heat, 2=HSV
    use_signal_response=False,  # split prefs into signal + response vectors
    swap_signal_response=False, # swap roles: signal↔response in physics
    vis_pref_source=0,      # 0=Signal, 1=Response (which to visualize)
)

auto_scale_ref = dict(
    n=2000,
    step_size=0.005,
    radius=0.06,
)

# ── Precomputed circle segments for radius visualisation ──
_N_CIRCLE_SEGS = 32
_circle_angles = np.linspace(0, 2 * np.pi, _N_CIRCLE_SEGS + 1)
CIRCLE_STARTS = np.column_stack([np.cos(_circle_angles[:-1]),
                                  np.sin(_circle_angles[:-1])])
CIRCLE_ENDS = np.column_stack([np.cos(_circle_angles[1:]),
                                np.sin(_circle_angles[1:])])
