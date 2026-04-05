"""
GPU-accelerated Grid Max Field physics using PyTorch.

Runs on MPS (Apple Silicon) or CUDA. Implements the same algorithm
as the CPU numba version in physics_grid.py but using tensor ops:

1. Deposit: scatter particles onto grid with per-dim max
2. Propagate: iterative max_pool2d (3x3 with periodic padding)
3. Movement: gather grid values at particle positions, compute weighted dirs

The trick for tracking source positions through max_pool2d:
  - Encode particle INDEX into the grid alongside max pref value
  - After pooling, decode the winning index to look up position
  - This avoids maintaining a separate position grid through pooling
"""

import time as _time
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_device():
    """Get best available torch device."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _sync(device):
    """Force GPU sync for accurate timing."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def step_grid_max_field_gpu(pos_np, prefs_np, response_np, L, k,
                            step_size, social, grid_res, max_spread,
                            **kwargs):
    """Grid Max Field physics step on GPU.

    Args:
        pos_np:      (N, 2) float64 positions
        prefs_np:    (N, K) float32 signal preferences
        response_np: (N, K) float32 response weights
        L:           domain size
        k:           preference dimensions
        step_size:   movement scale
        social:      social learning rate (0 = off)
        grid_res:    grid resolution G
        max_spread:  number of max-pool propagation passes

    Returns:
        new_pos:   (N, 2) numpy positions
        new_prefs: (N, K) numpy signal (updated if social != 0)
        movement:  (N, 2) numpy movement vectors
        max_pref_grid: (G, G, K) numpy grid for visualization
    """
    device = _get_device()
    dtype = torch.float32
    n = len(pos_np)
    G = grid_res

    t_deposit = t_propagate = t_movement = t_social = 0.0

    pos = torch.tensor(pos_np, dtype=dtype, device=device)
    prefs = torch.tensor(prefs_np, dtype=dtype, device=device)
    resp = torch.tensor(response_np, dtype=dtype, device=device)

    inv_cell = G / L

    # ── Step 1: Deposit — scatter max pref + particle index onto grid ──
    _sync(device)
    _t0 = _time.perf_counter()

    # Compute grid cell indices for each particle
    gx = (pos[:, 0] * inv_cell).long() % G
    gy = (pos[:, 1] * inv_cell).long() % G
    cell_idx = gy * G + gx  # flat cell index (N,)

    # For each dim d, scatter_reduce with "amax" to get per-cell max pref
    # Grid shape: (K, G*G) then reshape to (K, G, G)
    flat_size = G * G

    # Init grid to -inf
    max_pref_flat = torch.full((k, flat_size), -float('inf'), dtype=dtype, device=device)

    # Scatter max: for each dim, deposit particle prefs into cells
    cell_idx_exp = cell_idx.unsqueeze(0).expand(k, -1)  # (K, N)
    prefs_t = prefs.t()  # (K, N)
    max_pref_flat.scatter_reduce_(1, cell_idx_exp, prefs_t, reduce='amax',
                                  include_self=True)

    # Track which particle won in each cell — vectorized.
    # Approach: scatter ALL particle IDs to their cells (last writer wins),
    # then fix up: for each cell, check if the stored particle's pref matches
    # the cell max. If multiple particles share a cell, the last-written one
    # whose pref matches the max wins.

    particle_ids = torch.arange(n, dtype=torch.long, device=device)
    particle_ids_k = particle_ids.unsqueeze(0).expand(k, -1)  # (K, N)

    # First: scatter all particle IDs (last writer wins within each cell)
    winner_idx_flat = torch.full((k, flat_size), -1, dtype=torch.long, device=device)
    winner_idx_flat.scatter_(1, cell_idx_exp, particle_ids_k)

    # The stored ID might not be the max-pref particle in that cell.
    # Fix: for cells where the stored particle doesn't match the max,
    # we need the actual winner. Use scatter_reduce with amax on
    # a value that encodes both "is winner" and "particle ID":
    # Encode: winners get (particle_id), non-winners get -1
    cell_max_at_particle = max_pref_flat.gather(1, cell_idx_exp)  # (K, N)
    is_winner = (prefs_t >= cell_max_at_particle - 1e-7)  # (K, N) bool

    # Re-scatter only winners using float scatter_reduce amax
    # (MPS doesn't support int64 scatter_reduce)
    winner_ids_float = torch.where(is_winner, particle_ids_k.float(),
                                   torch.tensor(-1.0, dtype=dtype, device=device))
    winner_float_flat = torch.full((k, flat_size), -1.0, dtype=dtype, device=device)
    winner_float_flat.scatter_reduce_(1, cell_idx_exp, winner_ids_float,
                                      reduce='amax', include_self=True)
    winner_idx_flat = winner_float_flat.long()

    max_pref_grid = max_pref_flat.reshape(k, G, G)  # (K, G, G)
    winner_idx_grid = winner_idx_flat.reshape(k, G, G)  # (K, G, G)

    _sync(device)
    t_deposit = _time.perf_counter() - _t0

    # ── Step 2: Propagate — fused max over (2R+1)x(2R+1) neighborhood ──
    _sync(device)
    _t0 = _time.perf_counter()

    R = max_spread
    kernel_size = 2 * R + 1
    circular = kwargs.get('circular', False)

    # Periodic padding by R on each side
    padded_pref = _periodic_pad_2d(max_pref_grid, R)       # (K, G+2R, G+2R)
    padded_idx = _periodic_pad_2d_long(winner_idx_grid, R)  # (K, G+2R, G+2R)
    padded_w = G + 2 * R

    if circular:
        # ── Circular: unfold into windows, mask, argmax ──
        # Unfold pref grid into (2R+1)^2 patches per cell
        padded_4d = padded_pref.unsqueeze(0)  # (1, K, G+2R, G+2R)
        pref_windows = F.unfold(padded_4d, kernel_size=kernel_size)  # (1, K*ks^2, G*G)
        pref_windows = pref_windows.reshape(k, kernel_size * kernel_size, G * G)

        # Precompute circular mask
        ys = torch.arange(kernel_size, device=device) - R
        xs = torch.arange(kernel_size, device=device) - R
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        circle = (xx * xx + yy * yy <= R * R).reshape(-1)  # (ks^2,)

        # Mask out-of-circle positions to -inf
        pref_windows[:, ~circle, :] = -float('inf')

        # Argmax per cell per dim
        best_in_window = pref_windows.argmax(dim=1)  # (K, G*G)
        max_pref_grid = pref_windows.gather(1, best_in_window.unsqueeze(1)).squeeze(1).reshape(k, G, G)

        # Look up winner indices: unfold the index grid the same way
        padded_idx_float = padded_idx.unsqueeze(0).float()
        idx_windows = F.unfold(padded_idx_float, kernel_size=kernel_size)
        idx_windows = idx_windows.reshape(k, kernel_size * kernel_size, G * G).long()

        winner_idx_grid = idx_windows.gather(1, best_in_window.unsqueeze(1)).squeeze(1).reshape(k, G, G)

    else:
        # ── Square: native max_pool2d (single fused kernel) ──
        padded_4d = padded_pref.unsqueeze(0)  # (1, K, G+2R, G+2R)
        max_vals, flat_indices = F.max_pool2d(
            padded_4d, kernel_size=kernel_size, stride=1,
            padding=0, return_indices=True)
        # max_vals: (1, K, G, G), flat_indices: (1, K, G, G)
        max_pref_grid = max_vals.squeeze(0)
        flat_indices = flat_indices.squeeze(0)  # (K, G, G)

        # Map flat indices into padded grid → look up winner
        padded_idx_flat = padded_idx.reshape(k, -1)  # (K, padded_w^2)
        winner_idx_grid = padded_idx_flat.gather(
            1, flat_indices.reshape(k, -1)).reshape(k, G, G)

    _sync(device)
    t_propagate = _time.perf_counter() - _t0

    # ── Step 3: Movement — gather grid values at particle positions ──
    _sync(device)
    _t0 = _time.perf_counter()

    # Look up each particle's cell in the propagated grid
    # cell_idx already computed: (N,)
    # For each particle, for each dim, get the max_pref and winner_idx
    cell_idx_k = cell_idx.unsqueeze(0).expand(k, -1)  # (K, N)
    max_pref_at_particle = max_pref_grid.reshape(k, flat_size).gather(1, cell_idx_k)  # (K, N)
    winner_at_particle = winner_idx_grid.reshape(k, flat_size).gather(1, cell_idx_k)  # (K, N)

    max_pref_at_particle = max_pref_at_particle.t()  # (N, K)
    winner_at_particle = winner_at_particle.t()       # (N, K)

    # Look up source positions from winner indices
    # winner_at_particle is (N, K) with particle indices
    # Clamp invalid indices (-1) to 0, we'll mask them out
    valid_winner = winner_at_particle >= 0
    winner_clamped = winner_at_particle.clamp(min=0)  # (N, K)

    # Source positions: pos[winner] for each dim
    src_pos = pos[winner_clamped]  # (N, K, 2)

    # Direction toward source (periodic)
    disp = src_pos - pos.unsqueeze(1)  # (N, K, 2)
    disp = disp - L * torch.round(disp / L)

    dist = disp.norm(dim=2).clamp(min=1e-12)  # (N, K)
    unit_dir = disp / dist.unsqueeze(2)  # (N, K, 2)

    # Compatibility: response[i, d] * max_signal[d]
    compat = resp * max_pref_at_particle  # (N, K)

    # Mask out invalid (no winner) and self-hits (dist too small)
    compat = compat * valid_winner.float()
    too_close = dist < 1e-10
    compat = compat * (~too_close).float()

    # Weighted direction sum over dims
    movement = (compat.unsqueeze(2) * unit_dir).sum(dim=1)  # (N, 2)

    _sync(device)
    t_movement = _time.perf_counter() - _t0

    # ── Update positions ──
    new_pos = (pos + step_size * movement) % L

    # ── Social learning ──
    _sync(device)
    _t0 = _time.perf_counter()
    new_prefs = prefs.clone()
    if social != 0:
        # Simple field-based social: deposit sum, deposit count, normalize
        count_flat = torch.zeros(flat_size, dtype=dtype, device=device)
        count_flat.scatter_add_(0, cell_idx, torch.ones(n, dtype=dtype, device=device))
        count_grid = count_flat.reshape(G, G).clamp(min=1e-8)

        for d in range(k):
            sum_flat = torch.zeros(flat_size, dtype=dtype, device=device)
            sum_flat.scatter_add_(0, cell_idx, prefs[:, d])
            mean_grid = sum_flat.reshape(G, G) / count_grid

            # Sample mean at particle positions
            mean_at_particle = mean_grid.reshape(-1)[cell_idx]  # (N,)
            new_prefs[:, d] = ((1.0 - social) * prefs[:, d] +
                               social * mean_at_particle).clamp(-1, 1)

    _sync(device)
    t_social = _time.perf_counter() - _t0

    # Store timing
    step_grid_max_field_gpu._timing = (t_deposit, t_propagate, t_movement, t_social)

    # Return numpy arrays + grid for visualization
    max_pref_for_viz = max_pref_grid.permute(1, 2, 0).cpu().numpy()  # (G, G, K)

    step_grid_max_field_gpu._max_pref = max_pref_for_viz

    return (new_pos.cpu().to(torch.float64).numpy(),
            new_prefs.cpu().numpy(),
            movement.cpu().to(torch.float64).numpy())


def _periodic_pad_2d(x, pad):
    """Periodic (wrap) padding for a (C, H, W) tensor."""
    # Pad width (left, right)
    x = torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)
    # Pad height (top, bottom)
    x = torch.cat([x[:, -pad:, :], x, x[:, :pad, :]], dim=1)
    return x


def _periodic_pad_2d_long(x, pad):
    """Periodic padding for long tensor (C, H, W)."""
    x = torch.cat([x[:, :, -pad:], x, x[:, :, :pad]], dim=2)
    x = torch.cat([x[:, -pad:, :], x, x[:, :pad, :]], dim=1)
    return x
