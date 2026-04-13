"""
3D Simulation Class
===================

All positions are (n, 3), dir_matrix is (n, k, 3).
Presets generate 3D positions. Velocity colors use direction-to-RGB.
"""

import math
import time
import numpy as np
from scipy.spatial import cKDTree

from .grid3d import (
    SPACE, grid_build, periodic_dist,
    _count_radius, _query_radius, _query_knn,
)
from .physics3d import _step_inner_prod_avg, _step_per_dim

try:
    import torch
    _HAS_TORCH = True
    _TORCH_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
except ImportError:
    _HAS_TORCH = False
    _TORCH_DEVICE = 'cpu'


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
    neighbor_radius=0.14,
    trail_decay=0.98,
    point_size=4.0,
    right_view=0,
    show_box=False,
    trail_zoom=True,
    init_preset=0,
    show_neighbors=False,
    show_radius=False,
    use_seed=True,
    seed=42,
    auto_scale=False,
    reuse_neighbors=True,
    debug_knn=False,
    knn_method=1,       # 0=Hash Grid, 1=cKDTree (f64)
    use_f64=True,
    physics_engine=2,   # 0=Numba, 1=NumPy, 2=PyTorch
    torch_precision=3,  # 0=f16, 1=bf16, 2=f32, 3=f64
    torch_device=0,
    flat_z=False,
    show_axes=True,
    # Mountain visualization
    show_mountain=False,
    show_cost_overlay=False,
    mountain_alpha=0.4,
    cost_alpha=0.35,
    mountain_z_scale=0.5,
    mountain_resolution=64,
    mountain_mode=False,  # when True, particles are projected onto the surface
    # ── Dual-space (strategy separate from preferences) ──
    strategy_enabled=False,          # enable separate strategy vector for mountain navigation
    strategy_k=3,                    # dimensionality of strategy space
    pref_strategy_coupling=0.5,      # how much preferences overlap with strategy (0=independent, 1=identical)
    strategy_step_size=0.003,        # base step size for strategy movement
    strategy_memory_enabled=False,   # enable knowledge memory field in strategy space
    strategy_memory_strength=0.5,    # how strongly knowledge field nudges strategy
    strategy_memory_write_rate=0.01, # how fast particles deposit into knowledge field
    strategy_memory_decay=0.999,     # knowledge field decay per step
    strategy_memory_blur=False,      # apply Gaussian blur to knowledge field
    strategy_memory_blur_sigma=1.0,  # blur sigma for knowledge field
    # ── Per-particle roles ──
    use_particle_roles=False,        # enable per-particle step/influence scaling
    role_step_scale_std=0.0,         # std of log-normal step-scale distribution
    role_influence_std=0.0,          # std of log-normal influence distribution
    role_gradient_noise_mean=0.5,    # mean gradient noise (lower = better researcher)
    role_gradient_noise_std=0.0,     # std of gradient noise distribution
    role_visionary_mean=0.0,         # mean visionary blend weight
    role_visionary_std=0.0,          # std of visionary distribution
    role_visionary_fraction=1.0,     # fraction of particles that can be visionaries
)

auto_scale_ref = dict(
    n=2000,
    step_size=0.005,
    radius=0.14,
)

INIT_PRESETS = [
    "Random Uniform",
    "Grid Lattice",
    "Two Nations",
    "Polarized Society",
    "Cultural Gradient",
    "Elite Core",
    "Checkerboard",
    "Class Strata",
    "Minority Enclave",
    "Ring of Communities",
]


# =====================================================================
# PYTORCH VECTORIZED PHYSICS (3D)
# =====================================================================

_TORCH_DTYPES = {
    0: 'float16',
    1: 'bfloat16',
    2: 'float32',
    3: 'float64',
}


def _torch_periodic_dist(a, b, L=1.0):
    d = b - a
    d = d - L * torch.round(d / L)
    return d


def _step_torch(pos_np, prefs_np, dm_np, nbr_ids_np, valid_np,
                L, k, step_size, repulsion, dir_memory,
                social, social_dist_weight,
                pref_weighted, pref_inner, inner_avg,
                pref_dist_w, pref_dist_sigma, best_mag):
    """Full physics step using PyTorch vectorized ops (3D)."""
    prec = params['torch_precision']
    dtype_name = _TORCH_DTYPES.get(prec, 'float32')
    dtype = getattr(torch, dtype_name)

    dev_idx = params['torch_device']
    if dev_idx == 0:
        device = _TORCH_DEVICE
    else:
        device = 'cpu'

    if device == 'mps' and dtype == torch.float64:
        device = 'cpu'
    if device == 'mps' and dtype == torch.bfloat16:
        try:
            torch.zeros(1, dtype=torch.bfloat16, device='mps')
        except Exception:
            device = 'cpu'

    n = len(pos_np)
    has_mask = valid_np is not None

    pos = torch.tensor(pos_np, dtype=dtype, device=device)
    prefs = torch.tensor(prefs_np, dtype=torch.float32, device=device)
    dm = torch.tensor(dm_np, dtype=dtype, device=device)
    nbr_ids = torch.tensor(nbr_ids_np.astype(np.int64), device=device)
    if has_mask:
        valid = torch.tensor(valid_np, device=device)
        n_valid = valid.sum(dim=1).clamp(min=1).to(dtype)
    else:
        valid = None

    nbr_pos = pos[nbr_ids]  # (N, n_nbr, 3)
    toward = _torch_periodic_dist(pos.unsqueeze(1), nbr_pos, L)
    dists = toward.norm(dim=2, keepdim=True).clamp(min=1e-12)
    toward_unit = toward / dists

    movement = torch.zeros(n, 3, dtype=dtype, device=device)

    if inner_avg:
        prefs_ip = prefs.to(dtype)
        ip = (prefs_ip.unsqueeze(1) * prefs_ip[nbr_ids]).sum(dim=2) / k
        if pref_dist_w:
            gw = torch.exp(-dists.squeeze(2) ** 2 / (2.0 * pref_dist_sigma ** 2))
            ip = ip * gw
        weighted = ip.unsqueeze(2) * toward_unit
        if has_mask:
            weighted = weighted * valid.unsqueeze(2)
            movement = weighted.sum(dim=1) / n_valid.unsqueeze(1)
        else:
            movement = weighted.mean(dim=1)

        unit_away = -toward_unit
        push_raw = unit_away / dists.clamp(min=1e-6)
        if has_mask:
            push_raw = push_raw * valid.unsqueeze(2)
            push = push_raw.sum(dim=1) / n_valid.unsqueeze(1)
        else:
            push = push_raw.mean(dim=1)
        movement = movement + repulsion * push
        new_pos = (pos + step_size * movement) % L

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

        return (new_pos.cpu().to(torch.float64).numpy(),
                new_prefs.cpu().to(torch.float32).numpy(),
                dm_np,
                movement.cpu().to(torch.float64).numpy())

    # Per-dimension mode
    arange_n = torch.arange(n, device=device)
    for ki in range(k):
        nbr_pref_k = prefs[nbr_ids, ki]

        if pref_weighted:
            weights = nbr_pref_k.unsqueeze(2)
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
                masked_score = torch.where(valid, score,
                    torch.tensor(float('-inf'), dtype=score.dtype, device=device))
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

    return (new_pos.cpu().to(torch.float64).numpy(),
            new_prefs.cpu().to(torch.float32).numpy(),
            dm.cpu().to(torch.float64).numpy(),
            movement.cpu().to(torch.float64).numpy())


# =====================================================================
# SIMULATION CLASS
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
        pos_dtype = np.float64 if params['use_f64'] else np.float32
        self.pos = np.zeros((n, 3), dtype=pos_dtype)
        self.prefs = np.zeros((n, k), dtype=np.float32)
        self.dir_matrix = np.zeros((n, k, 3), np.float64 if params['use_f64'] else np.float32)
        self._movement = np.zeros((n, 3), np.float64 if params['use_f64'] else np.float32)
        self.step_count = 0
        self.nbr_ids = None
        self._valid_mask = None
        self._t_search = 0.0
        self._t_build = 0.0
        self._t_query = 0.0
        self._t_physics = 0.0
        self._n_nbrs = 0
        self._apply_preset(params['init_preset'])
        self._init_roles()
        self._init_strategy()

    def _apply_preset(self, preset):
        n, k, rng = self.n, self.k, self.rng
        L = SPACE
        pdtype = self.pos.dtype
        flat = params['flat_z']

        # Helper: generate positions with RNG-compatible shape.
        # When flat_z is on, generate (n, 2) so the RNG stream matches
        # the 2D version exactly, then set z=0.
        def _pos(shape_or_args, **kwargs):
            """Generate positions. shape_or_args is the shape tuple for uniform/normal."""
            return  # placeholder, each preset handles it inline

        if preset == 0:
            if flat:
                xy = rng.uniform(0, L, (n, 2)).astype(pdtype)
                self.pos[:, :2] = xy
                self.pos[:, 2] = 0.0
            else:
                self.pos[:] = rng.uniform(0, L, (n, 3)).astype(pdtype)
            self.prefs[:] = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        elif preset == 1:
            if flat:
                side = int(np.ceil(np.sqrt(n)))
                spacing = L / side
                xs = np.linspace(0, L, side, endpoint=False) + spacing / 2
                ys = np.linspace(0, L, side, endpoint=False) + spacing / 2
                gx, gy = np.meshgrid(xs, ys)
                self.pos[:, 0] = gx.ravel()[:n]
                self.pos[:, 1] = gy.ravel()[:n]
                self.pos[:, 2] = 0.0
                self.pos[:] = self.pos.astype(pdtype)
                noise = rng.normal(0, spacing * 0.05, (n, 2)).astype(pdtype)
                self.pos[:, :2] += noise
                self.pos[:, :2] %= L
            else:
                side = int(np.ceil(n ** (1.0 / 3.0)))
                spacing = L / side
                xs = np.linspace(0, L, side, endpoint=False) + spacing / 2
                gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
                pts = np.column_stack([gx.ravel()[:n], gy.ravel()[:n], gz.ravel()[:n]])
                self.pos[:] = pts.astype(pdtype)
                self.pos[:] += rng.normal(0, spacing * 0.05, (n, 3)).astype(pdtype)
                self.pos[:] %= L
            self.prefs[:] = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        elif preset == 2:
            half = n // 2
            if flat:
                self.pos[:half, :2] = rng.uniform([0, 0], [L / 2, L], (half, 2)).astype(pdtype)
                self.pos[:half, 2] = 0.0
                self.pos[half:, :2] = rng.uniform([L / 2, 0], [L, L], (n - half, 2)).astype(pdtype)
                self.pos[half:, 2] = 0.0
            else:
                self.pos[:half] = rng.uniform([0, 0, 0], [L / 2, L, L], (half, 3)).astype(pdtype)
                self.pos[half:] = rng.uniform([L / 2, 0, 0], [L, L, L], (n - half, 3)).astype(pdtype)
            self.prefs[:half] = np.clip(
                rng.normal(0.8, 0.2, (half, k)), -1, 1).astype(np.float32)
            self.prefs[half:] = np.clip(
                rng.normal(-0.8, 0.2, (n - half, k)), -1, 1).astype(np.float32)
        elif preset == 3:
            if flat:
                self.pos[:, :2] = rng.uniform(0, L, (n, 2)).astype(pdtype)
                self.pos[:, 2] = 0.0
            else:
                self.pos[:] = rng.uniform(0, L, (n, 3)).astype(pdtype)
            signs = rng.choice([-1.0, 1.0], size=(n, k))
            self.prefs[:] = np.clip(
                signs + rng.normal(0, 0.1, (n, k)), -1, 1).astype(np.float32)
        elif preset == 4:
            if flat:
                self.pos[:, :2] = rng.uniform(0, L, (n, 2)).astype(pdtype)
                self.pos[:, 2] = 0.0
            else:
                self.pos[:] = rng.uniform(0, L, (n, 3)).astype(pdtype)
            self.prefs[:, 0] = np.clip(
                2.0 * self.pos[:, 0] / L - 1.0 + rng.normal(0, 0.15, n),
                -1, 1).astype(np.float32)
            if k > 1:
                self.prefs[:, 1] = np.clip(
                    2.0 * self.pos[:, 1] / L - 1.0 + rng.normal(0, 0.15, n),
                    -1, 1).astype(np.float32)
            if k > 2:
                self.prefs[:, 2:] = rng.uniform(-1, 1, (n, k - 2)).astype(np.float32)
        elif preset == 5:
            n_elite = n // 5
            n_periph = n - n_elite
            if flat:
                self.pos[:n_elite, :2] = (rng.normal(L / 2, L * 0.08, (n_elite, 2)) % L).astype(pdtype)
                self.pos[:n_elite, 2] = 0.0
                self.pos[n_elite:, :2] = rng.uniform(0, L, (n_periph, 2)).astype(pdtype)
                self.pos[n_elite:, 2] = 0.0
            else:
                self.pos[:n_elite] = (rng.normal(L / 2, L * 0.08, (n_elite, 3)) % L).astype(pdtype)
                self.pos[n_elite:] = rng.uniform(0, L, (n_periph, 3)).astype(pdtype)
            self.prefs[:n_elite] = np.clip(
                rng.normal(0.9, 0.1, (n_elite, k)), -1, 1).astype(np.float32)
            self.prefs[n_elite:] = np.clip(
                rng.normal(0.0, 0.3, (n_periph, k)), -1, 1).astype(np.float32)
        elif preset == 6:
            if flat:
                side = int(np.ceil(np.sqrt(n)))
                spacing = L / side
                xs = np.linspace(0, L, side, endpoint=False) + spacing / 2
                ys = np.linspace(0, L, side, endpoint=False) + spacing / 2
                gx, gy = np.meshgrid(xs, ys)
                ix, iy = np.meshgrid(np.arange(side), np.arange(side))
                self.pos[:, 0] = gx.ravel()[:n]
                self.pos[:, 1] = gy.ravel()[:n]
                self.pos[:, 2] = 0.0
                sign = ((-1.0) ** (ix + iy)).ravel()[:n]
            else:
                side = int(np.ceil(n ** (1.0 / 3.0)))
                spacing = L / side
                xs = np.linspace(0, L, side, endpoint=False) + spacing / 2
                gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
                ix, iy, iz = np.meshgrid(np.arange(side), np.arange(side),
                                          np.arange(side), indexing='ij')
                self.pos[:] = np.column_stack([
                    gx.ravel()[:n], gy.ravel()[:n], gz.ravel()[:n]
                ]).astype(pdtype)
                sign = ((-1.0) ** (ix + iy + iz)).ravel()[:n]
            for ki in range(k):
                self.prefs[:, ki] = np.clip(
                    sign * rng.uniform(0.7, 1.0, n), -1, 1).astype(np.float32)
        elif preset == 7:
            if flat:
                self.pos[:, :2] = rng.uniform(0, L, (n, 2)).astype(pdtype)
                self.pos[:, 2] = 0.0
            else:
                self.pos[:] = rng.uniform(0, L, (n, 3)).astype(pdtype)
            n_bands = 4
            band_means = np.array([-0.8, -0.25, 0.25, 0.8])
            band_idx = np.clip(
                (self.pos[:, 1] / L * n_bands).astype(int), 0, n_bands - 1)
            for ki in range(k):
                self.prefs[:, ki] = np.clip(
                    band_means[band_idx] + rng.normal(0, 0.15, n),
                    -1, 1).astype(np.float32)
        elif preset == 8:
            n_min = n // 5
            n_maj = n - n_min
            if flat:
                self.pos[:n_maj, :2] = rng.uniform(0, L, (n_maj, 2)).astype(pdtype)
                self.pos[:n_maj, 2] = 0.0
                self.pos[n_maj:, :2] = (rng.normal(
                    [L * 0.2, L * 0.2], L * 0.06, (n_min, 2)) % L).astype(pdtype)
                self.pos[n_maj:, 2] = 0.0
            else:
                self.pos[:n_maj] = rng.uniform(0, L, (n_maj, 3)).astype(pdtype)
                self.pos[n_maj:] = (rng.normal(
                    [L * 0.2, L * 0.2, L * 0.2], L * 0.06, (n_min, 3)) % L).astype(pdtype)
            self.prefs[:n_maj] = np.clip(
                rng.normal(0.5, 0.2, (n_maj, k)), -1, 1).astype(np.float32)
            self.prefs[n_maj:] = np.clip(
                rng.normal(-0.7, 0.15, (n_min, k)), -1, 1).astype(np.float32)
        elif preset == 9:
            n_comm = 6
            per_comm = n // n_comm
            angles = np.linspace(0, 2 * np.pi, n_comm, endpoint=False)
            radius = L * 0.3
            for i in range(n_comm):
                start = i * per_comm
                end = start + per_comm if i < n_comm - 1 else n
                count = end - start
                cx = L / 2 + radius * np.cos(angles[i])
                cy = L / 2 + radius * np.sin(angles[i])
                if flat:
                    self.pos[start:end, :2] = (rng.normal(
                        [cx, cy], L * 0.05, (count, 2)) % L).astype(pdtype)
                    self.pos[start:end, 2] = 0.0
                else:
                    cz = L / 2
                    self.pos[start:end] = (rng.normal(
                        [cx, cy, cz], L * 0.05, (count, 3)) % L).astype(pdtype)
                comm_pref = rng.uniform(-1, 1, k)
                self.prefs[start:end] = np.clip(
                    comm_pref[None, :] + rng.normal(0, 0.15, (count, k)),
                    -1, 1).astype(np.float32)
        else:
            if flat:
                self.pos[:, :2] = rng.uniform(0, L, (n, 2)).astype(pdtype)
                self.pos[:, 2] = 0.0
            else:
                self.pos[:] = rng.uniform(0, L, (n, 3)).astype(pdtype)
            self.prefs[:] = rng.uniform(-1, 1, (n, k)).astype(np.float32)

    # ── Role initialisation ────────────────────────────────────────

    def _init_roles(self):
        """Initialise per-particle role arrays (ported from 2D sim)."""
        n = self.n
        if params.get('use_particle_roles', False):
            ss_std = params.get('role_step_scale_std', 0.0)
            self.role_step_scale = (
                self.rng.lognormal(0.0, ss_std, n).astype(np.float64)
                if ss_std > 0 else np.ones(n, dtype=np.float64))
            inf_std = params.get('role_influence_std', 0.0)
            self.role_influence = (
                self.rng.lognormal(0.0, inf_std, n).astype(np.float64)
                if inf_std > 0 else np.ones(n, dtype=np.float64))
            gn_mean = params.get('role_gradient_noise_mean', 0.5)
            gn_std = params.get('role_gradient_noise_std', 0.0)
            self.role_gradient_noise = (
                np.abs(self.rng.normal(gn_mean, gn_std, n)).astype(np.float64)
                if gn_std > 0
                else np.full(n, gn_mean, dtype=np.float64))
            vis_mean = params.get('role_visionary_mean', 0.0)
            vis_std = params.get('role_visionary_std', 0.0)
            vis_frac = params.get('role_visionary_fraction', 1.0)
            raw_vis = (
                np.clip(self.rng.normal(vis_mean, vis_std, n), 0.0, 1.0).astype(np.float64)
                if vis_std > 0
                else np.full(n, np.clip(vis_mean, 0, 1), dtype=np.float64))
            if vis_frac < 1.0:
                n_vis = max(1, int(round(n * vis_frac)))
                mask = np.zeros(n, dtype=bool)
                mask[self.rng.choice(n, size=n_vis, replace=False)] = True
                raw_vis[~mask] = 0.0
            self.role_visionary = raw_vis
        else:
            self.role_step_scale = np.ones(n, dtype=np.float64)
            self.role_influence = np.ones(n, dtype=np.float64)
            self.role_gradient_noise = np.full(n, 0.5, dtype=np.float64)
            self.role_visionary = np.zeros(n, dtype=np.float64)

    # ── Strategy initialisation ────────────────────────────────────

    def _init_strategy(self):
        """Initialise separate strategy vector for dual-space mode."""
        n, k = self.n, self.k
        if params.get('strategy_enabled', False):
            k_s = params.get('strategy_k', k)
            self.strategy_k = k_s
            coupling = params.get('pref_strategy_coupling', 0.5)
            rand_strat = self.rng.uniform(-1, 1, (n, k_s)).astype(np.float64)
            if k_s == k:
                pref_f64 = self.prefs.astype(np.float64)
                self.strategy = np.clip(
                    coupling * pref_f64 + (1.0 - coupling) * rand_strat,
                    -1, 1)
            else:
                self.strategy = rand_strat
            G = 32  # strategy memory grid resolution
            self.strategy_memory = np.zeros((G, G, k_s), dtype=np.float64)
        else:
            self.strategy = None
            self.strategy_k = k
            self.strategy_memory = None

    # ── Strategy step (Phase 2: mountain navigation) ───────────────

    def strategy_step(self, gradient_fn=None, summit_center=None):
        """Phase 2: team-aggregated mountain navigation in strategy space.

        Called after sim.step() handles social dynamics.  Uses the
        preference-space neighbor graph (self.nbr_ids) to aggregate
        gradient observations across team members.
        """
        if self.strategy is None or gradient_fn is None:
            return
        if self.nbr_ids is None:
            return

        n = self.n
        k_s = self.strategy_k
        nbr_ids = self.nbr_ids
        valid = self._valid_mask
        has_mask = valid is not None

        # 1. Each particle senses the gradient with researcher noise
        grad_unit, fitness, peak_ids = gradient_fn(self.strategy)
        noise = self.rng.normal(0, 1, grad_unit.shape)
        noise *= self.role_gradient_noise[:, None]
        noisy_grad = grad_unit + noise
        norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
        noisy_grad = noisy_grad / np.maximum(norms, 1e-12)

        # 2. Visionary blend toward global summit
        if summit_center is not None:
            to_summit = summit_center - self.strategy
            s_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
            summit_dir = to_summit / np.maximum(s_norms, 1e-12)
            v = self.role_visionary[:, None]
            noisy_grad = (1.0 - v) * noisy_grad + v * summit_dir
            norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
            noisy_grad = noisy_grad / np.maximum(norms, 1e-12)

        # 3. Team aggregation: weight by leader influence
        nbr_grads = noisy_grad[nbr_ids]
        inf_weights = self.role_influence[nbr_ids]
        if has_mask:
            inf_weights = inf_weights * valid
        w_sum = inf_weights.sum(axis=1, keepdims=True)
        w_norm = inf_weights / np.maximum(w_sum, 1e-10)
        team_grad = (nbr_grads * w_norm[:, :, None]).sum(axis=1)

        # Include own observation (self-weight = 1.0)
        team_grad = 0.5 * noisy_grad + 0.5 * team_grad
        t_norms = np.linalg.norm(team_grad, axis=1, keepdims=True)
        team_grad = team_grad / np.maximum(t_norms, 1e-12)

        # 4. Move in strategy space (engineer step scale)
        strat_step = params.get('strategy_step_size', 0.003)
        step_mag = strat_step * self.role_step_scale
        self.strategy = np.clip(
            self.strategy + step_mag[:, None] * team_grad, -1, 1)

        # 5. Strategy memory field
        if (self.strategy_memory is not None and
                params.get('strategy_memory_enabled', False)):
            G = self.strategy_memory.shape[0]
            inv_cell = G / SPACE
            strat_01 = (self.strategy[:, :2] + 1.0) * 0.5
            sx = (strat_01[:, 0] * inv_cell).astype(int) % G
            sy = (strat_01[:, 1] * inv_cell).astype(int) % G
            strength = params.get('strategy_memory_strength', 0.5)
            field_at = self.strategy_memory[sy, sx]
            self.strategy = np.clip(
                self.strategy + strength * 0.01 * field_at, -1, 1)
            write_rate = params.get('strategy_memory_write_rate', 0.01)
            for d in range(k_s):
                np.add.at(self.strategy_memory[:, :, d], (sy, sx),
                          write_rate * self.strategy[:, d])
            self.strategy_memory *= params.get('strategy_memory_decay', 0.999)
            if params.get('strategy_memory_blur', False):
                from scipy.ndimage import gaussian_filter
                sigma = params.get('strategy_memory_blur_sigma', 1.0)
                for d in range(k_s):
                    self.strategy_memory[:, :, d] = gaussian_filter(
                        self.strategy_memory[:, :, d],
                        sigma=sigma, mode='wrap')

        # 6. Preference-strategy coupling drift
        coupling = params.get('pref_strategy_coupling', 0.5)
        if coupling > 0 and self.strategy_k == self.k:
            pref_f64 = self.prefs.astype(np.float64)
            drift = coupling * 0.01 * (pref_f64 - self.strategy)
            self.strategy = np.clip(self.strategy + drift, -1, 1)

    def _find_neighbors(self):
        pos = self.pos
        n = len(pos)
        nbr_mode = params['neighbor_mode']
        n_nbr = min(params['n_neighbors'], n - 1)
        radius = params['neighbor_radius']
        knn_method = params['knn_method']

        _tb0 = time.perf_counter()

        if knn_method == 0:
            self._find_neighbors_hash(pos, n, nbr_mode, n_nbr, radius, _tb0)
        else:
            if knn_method == 1:
                query_pos = pos.astype(np.float64)
            else:
                query_pos = pos.astype(np.float32).astype(np.float64)
            query_pos = query_pos % SPACE

            self._t_build = 0.0
            _tq0 = time.perf_counter()
            tree = cKDTree(query_pos, boxsize=SPACE)

            if nbr_mode == 2:
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
                _, nbr_ids = tree.query(query_pos, k=n_nbr + 1, workers=-1)
                self.nbr_ids = nbr_ids[:, 1:].astype(np.int64)
                self._valid_mask = None

            self._t_query = time.perf_counter() - _tq0

        self._t_search = self._t_build + self._t_query

        # Debug KNN comparison
        if params['debug_knn'] and knn_method == 0 and nbr_mode in (0, 1):
            pos_f64 = pos.astype(np.float64) % SPACE
            tree = cKDTree(pos_f64, boxsize=SPACE)
            _, ref_ids = tree.query(pos_f64, k=n_nbr + 1, workers=-1)
            ref_ids = ref_ids[:, 1:]
            hash_ids = self.nbr_ids
            n_wrong_set = 0
            for pi in range(n):
                if set(ref_ids[pi]) != set(hash_ids[pi]):
                    n_wrong_set += 1
            if n_wrong_set > 0:
                print(f"[debug_knn] step {self.step_count}: "
                      f"{n_wrong_set}/{n} particles have wrong neighbor sets")
            elif self.step_count % 500 == 0:
                print(f"[debug_knn] step {self.step_count}: all KNN match cKDTree")

    def _find_neighbors_hash(self, pos, n, nbr_mode, n_nbr, radius, _tb0):
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
            self._t_query = time.perf_counter() - _tq0
            self.nbr_ids = nbr_ids
            self._valid_mask = valid

        elif nbr_mode == 0:
            # 3D density heuristic
            knn_radius_est = (n_nbr / (4.0 / 3.0 * math.pi * max(n, 1))) ** (1.0 / 3.0)
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
            knn_radius_est = (n_nbr / (4.0 / 3.0 * math.pi * max(n, 1))) ** (1.0 / 3.0)
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

        if not reuse_neighbors or self.nbr_ids is None:
            self._find_neighbors()

        nbr_ids = self.nbr_ids
        valid = self._valid_mask

        _tp0 = time.perf_counter()

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
            # NumPy vectorized physics
            movement = self._movement
            movement[:] = 0.0
            nbr_pos = pos[nbr_ids]
            toward = periodic_dist(pos[:, None, :], nbr_pos)
            dists = np.linalg.norm(toward, axis=2, keepdims=True)
            toward_unit = toward / np.maximum(dists, 1e-12)

            if inner_avg:
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

        # Numba physics path
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

    def get_render_data(self):
        k = self.k
        colors = np.clip((self.prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
        if k < 3:
            c = np.full((len(self.prefs), 3), 0.5, np.float32)
            c[:, :min(k, 3)] = colors[:, :min(k, 3)]
            colors = c
        return self.pos.astype(np.float32), colors

    def get_velocity_colors(self):
        """3D direction → RGB: abs(vx/mag), abs(vy/mag), abs(vz/mag) * brightness."""
        vx = self._movement[:, 0]
        vy = self._movement[:, 1]
        vz = self._movement[:, 2]
        mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        p95 = np.percentile(mag, 95) + 1e-8
        brightness = np.clip(mag / p95, 0.0, 1.0).astype(np.float32)
        safe_mag = np.maximum(mag, 1e-12)

        rgb = np.zeros((len(vx), 3), dtype=np.float32)
        rgb[:, 0] = np.abs(vx / safe_mag) * brightness
        rgb[:, 1] = np.abs(vy / safe_mag) * brightness
        rgb[:, 2] = np.abs(vz / safe_mag) * brightness
        return rgb

    def get_neighbor_lines(self):
        """Output (m, 3) line vertices for neighbor visualization."""
        if self.nbr_ids is None:
            return np.zeros((0, 3), dtype=np.float32)
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
        lines = np.empty((n_edges * 2, 3), dtype=np.float32)
        lines[0::2] = starts.astype(np.float32)
        lines[1::2] = ends.astype(np.float32)
        return lines


_N_CIRCLE_SEGS = 32
_circle_angles = np.linspace(0, 2 * np.pi, _N_CIRCLE_SEGS + 1)
_circle_cos = np.cos(_circle_angles).astype(np.float32)
_circle_sin = np.sin(_circle_angles).astype(np.float32)


def make_radius_circles_3d(positions, radius, cam_right, cam_up):
    """Generate billboard circle line segments facing the camera.

    Args:
        positions: (n, 3) float32 particle positions
        radius: float neighbor radius
        cam_right: (3,) camera right vector
        cam_up: (3,) camera up vector

    Returns:
        (n * N_SEGS * 2, 3) float32 line vertices
    """
    # Circle points in camera plane: radius * (cos*right + sin*up)
    # Shape: (_N_CIRCLE_SEGS+1, 3)
    circle = radius * (_circle_cos[:, None] * cam_right[None, :] +
                       _circle_sin[:, None] * cam_up[None, :])

    starts = circle[:-1]  # (_N_CIRCLE_SEGS, 3)
    ends = circle[1:]     # (_N_CIRCLE_SEGS, 3)

    n = len(positions)
    all_starts = positions[:, None, :] + starts[None, :, :]
    all_ends = positions[:, None, :] + ends[None, :, :]

    lines = np.empty((n * _N_CIRCLE_SEGS * 2, 3), dtype=np.float32)
    lines[0::2] = all_starts.reshape(-1, 3)
    lines[1::2] = all_ends.reshape(-1, 3)
    return lines
