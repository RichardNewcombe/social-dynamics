"""
PyTorch vectorized physics for the 2D particle simulation.

Reimplements the physics step using PyTorch tensor ops for GPU acceleration
(supports MPS on Apple Silicon, CUDA, or CPU fallback).
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
    _TORCH_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
except ImportError:
    _HAS_TORCH = False
    _TORCH_DEVICE = 'cpu'

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


def step_torch(pos_np, prefs_np, dm_np, nbr_ids_np, valid_np,
               L, k, step_size, repulsion, dir_memory,
               social, social_dist_weight,
               pref_weighted, pref_inner, inner_avg,
               pref_dist_w, pref_dist_sigma, best_mode,
               torch_precision=2, torch_device_idx=0):
    """Full physics step using PyTorch vectorized ops.

    Args:
        pos_np:     (N, 2) positions
        prefs_np:   (N, K) preferences
        dm_np:      (N, K, 2) direction memory
        nbr_ids_np: (N, n_nbr) neighbor indices
        valid_np:   (N, n_nbr) bool mask or None
        L, k, step_size, repulsion, dir_memory, social: physics params
        social_dist_weight, pref_weighted, pref_inner, inner_avg: mode flags
        pref_dist_w, pref_dist_sigma, best_mode: additional params
        torch_precision: 0=f16, 1=bf16, 2=f32, 3=f64
        torch_device_idx: 0=auto, 1=cpu

    Returns:
        (new_pos, new_prefs, new_dm, movement) as numpy arrays
    """
    dtype_name = _TORCH_DTYPES.get(torch_precision, 'float32')
    dtype = getattr(torch, dtype_name)

    if torch_device_idx == 0:
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
        n_nbr = nbr_ids.shape[1]

    nbr_pos = pos[nbr_ids]
    toward = _torch_periodic_dist(pos.unsqueeze(1), nbr_pos, L)
    dists = toward.norm(dim=2, keepdim=True).clamp(min=1e-12)
    toward_unit = toward / dists

    movement = torch.zeros(n, 2, dtype=dtype, device=device)

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
            if best_mode == 2:
                # Same-sign max magnitude: mask out opposite-sign neighbors
                my_sign = (prefs[:, ki] >= 0).unsqueeze(1)       # (N, 1)
                nbr_sign = (nbr_pref_k >= 0)                     # (N, n_nbr)
                same_sign = (my_sign == nbr_sign)                 # (N, n_nbr)
                score = nbr_pref_k.abs()
                neg_inf = torch.tensor(float('-inf'), dtype=score.dtype, device=device)
                score = torch.where(same_sign, score, neg_inf)
                # Track which particles have no valid same-sign neighbor
                if has_mask:
                    any_valid = (same_sign & valid).any(dim=1)    # (N,)
                else:
                    any_valid = same_sign.any(dim=1)              # (N,)
            elif best_mode == 1:
                score = nbr_pref_k.abs()
                any_valid = None
            else:
                score = nbr_pref_k
                any_valid = None
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

            if any_valid is not None:
                # Zero out unit_dir for particles with no valid same-sign neighbor
                # so direction memory decays rather than pointing at a wrong neighbor
                unit_dir = unit_dir * any_valid.unsqueeze(1).to(dtype)

            dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * unit_dir

            compat = prefs[:, ki] * prefs[best_nbr, ki]
            if pref_inner:
                full_compat = (prefs * prefs[best_nbr]).sum(dim=1) / k
                compat = compat * full_compat
            if pref_dist_w:
                gw = torch.exp(-dist.squeeze(1) ** 2 / (2.0 * pref_dist_sigma ** 2))
                compat = compat * gw
            if any_valid is not None:
                compat = compat * any_valid.to(compat.dtype)
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

    return (new_pos.cpu().to(torch.float64).numpy(),
            new_prefs.cpu().to(torch.float32).numpy(),
            dm.cpu().to(torch.float64).numpy(),
            movement.cpu().to(torch.float64).numpy())
