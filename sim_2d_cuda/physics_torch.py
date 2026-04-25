"""
PyTorch vectorized physics for the 2D particle simulation — CUDA variant.

Optimized for NVIDIA GPUs:
  - CUDA device detection prioritized over MPS
  - Persistent GPU tensors via step_torch_gpu() to avoid per-frame transfers
  - Fallback step_torch() for compatibility with Numba/NumPy engine switching
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
    if torch.cuda.is_available():
        _TORCH_DEVICE = 'cuda'
        _gpu_name = torch.cuda.get_device_name(0)
        print(f"PyTorch CUDA available: {_gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _TORCH_DEVICE = 'mps'
        print("PyTorch MPS available (Apple Silicon)")
    else:
        _TORCH_DEVICE = 'cpu'
        print("PyTorch: no GPU found, using CPU")
except ImportError:
    _HAS_TORCH = False
    _TORCH_DEVICE = 'cpu'
    print("PyTorch not available")

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


def _resolve_device_dtype(torch_precision, torch_device_idx):
    """Resolve device and dtype, handling GPU limitations."""
    dtype_name = _TORCH_DTYPES.get(torch_precision, 'float32')
    dtype = getattr(torch, dtype_name)

    device = _TORCH_DEVICE if torch_device_idx == 0 else 'cpu'

    # MPS doesn't support float64
    if device == 'mps' and dtype == torch.float64:
        device = 'cpu'
    # MPS bfloat16 may not be supported
    if device == 'mps' and dtype == torch.bfloat16:
        try:
            torch.zeros(1, dtype=torch.bfloat16, device='mps')
        except Exception:
            device = 'cpu'

    return device, dtype


# =====================================================================
# Persistent GPU state for CUDA — avoids per-frame CPU↔GPU transfers
# =====================================================================

class TorchGPUState:
    """Keeps particle data on GPU across frames.

    Call upload() to send initial data or when particle count changes.
    Call step() to run physics entirely on GPU.
    Call download() to get results back to CPU (for rendering / neighbor finding).
    """

    def __init__(self):
        self.device = None
        self.dtype = None
        self.n = 0
        self.k = 0
        # Persistent GPU tensors
        self.pos = None
        self.prefs = None
        self.resp = None
        self.dm = None

    def upload(self, pos_np, prefs_np, response_np, dm_np, device, dtype):
        """Transfer state from CPU to GPU."""
        self.device = device
        self.dtype = dtype
        self.n = len(pos_np)
        self.k = prefs_np.shape[1]
        self.pos = torch.tensor(pos_np, dtype=dtype, device=device)
        self.prefs = torch.tensor(prefs_np, dtype=torch.float32, device=device)
        self.resp = torch.tensor(response_np, dtype=torch.float32, device=device)
        self.dm = torch.tensor(dm_np, dtype=dtype, device=device)

    def download(self):
        """Transfer state from GPU back to CPU numpy arrays."""
        return (
            self.pos.cpu().to(torch.float64).numpy(),
            self.prefs.cpu().to(torch.float32).numpy(),
            self.resp.cpu().to(torch.float32).numpy(),
            self.dm.cpu().to(torch.float64).numpy(),
        )

    def step(self, nbr_ids_np, valid_np, L, k,
             step_size, repulsion, dir_memory,
             social, social_dist_weight,
             pref_weighted, pref_inner, inner_avg,
             pref_dist_w, pref_dist_sigma, best_mode):
        """Run one physics step entirely on GPU.

        Neighbor IDs still come from CPU (spatial hash grid).
        Returns movement as numpy for velocity visualization.
        """
        device = self.device
        dtype = self.dtype
        pos = self.pos
        prefs = self.prefs
        resp = self.resp
        dm = self.dm
        n = self.n
        has_mask = valid_np is not None

        nbr_ids = torch.tensor(nbr_ids_np.astype(np.int64), device=device)
        if has_mask:
            valid = torch.tensor(valid_np, device=device)
            n_valid = valid.sum(dim=1).clamp(min=1).to(dtype)
        else:
            valid = None

        nbr_pos = pos[nbr_ids]
        toward = _torch_periodic_dist(pos.unsqueeze(1), nbr_pos, L)
        dists = toward.norm(dim=2, keepdim=True).clamp(min=1e-12)
        toward_unit = toward / dists

        movement = torch.zeros(n, 2, dtype=dtype, device=device)

        if inner_avg:
            resp_ip = resp.to(dtype)
            prefs_ip = prefs.to(dtype)
            ip = (resp_ip.unsqueeze(1) * prefs_ip[nbr_ids]).sum(dim=2) / k
            if pref_dist_w:
                gw = torch.exp(-dists.squeeze(2) ** 2 / (2.0 * pref_dist_sigma ** 2))
                ip = ip * gw
            weighted = ip.unsqueeze(2) * toward_unit
            if has_mask:
                weighted = weighted * valid.unsqueeze(2)
                movement = weighted.sum(dim=1) / n_valid.unsqueeze(1)
            else:
                movement = weighted.mean(dim=1)
        else:
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
                    movement = movement + resp[:, ki:ki+1] * dm[:, ki, :]

                else:
                    neg_inf = torch.tensor(float('-inf'), dtype=torch.float32, device=device)
                    if best_mode == 2:
                        my_sign = (prefs[:, ki] >= 0).unsqueeze(1)
                        nbr_sign = (nbr_pref_k >= 0)
                        same_sign = (my_sign == nbr_sign)
                        score = nbr_pref_k.abs()
                        score = torch.where(same_sign, score, neg_inf)
                        if has_mask:
                            any_valid = (same_sign & valid).any(dim=1)
                        else:
                            any_valid = same_sign.any(dim=1)
                    elif best_mode == 1:
                        score = nbr_pref_k.abs()
                        any_valid = None
                    else:
                        score = nbr_pref_k
                        any_valid = None

                    if has_mask:
                        masked_score = torch.where(valid, score, neg_inf)
                        best_local = masked_score.argmax(dim=1)
                    else:
                        best_local = score.argmax(dim=1)
                    best_nbr = nbr_ids[arange_n, best_local]

                    disp = _torch_periodic_dist(pos, pos[best_nbr], L)
                    dist = disp.norm(dim=1, keepdim=True).clamp(min=1e-12)
                    unit_dir = disp / dist

                    if any_valid is not None:
                        unit_dir = unit_dir * any_valid.unsqueeze(1).to(dtype)

                    dm[:, ki, :] = dir_memory * dm[:, ki, :] + (1.0 - dir_memory) * unit_dir

                    compat = resp[:, ki] * prefs[best_nbr, ki]
                    if pref_inner:
                        full_compat = (resp * prefs[best_nbr]).sum(dim=1) / k
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

        # Update position (in-place on GPU)
        self.pos = (pos + step_size * movement) % L

        # Social learning (signal only, in-place on GPU)
        if social != 0:
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
            self.prefs = ((1.0 - social) * prefs + social * nbr_mean).clamp(-1, 1)

        # Return movement for velocity visualization (only transfer needed)
        return movement.cpu().to(torch.float64).numpy()


# Singleton for persistent GPU state
_gpu_state = TorchGPUState()


def step_torch(pos_np, prefs_np, response_np, dm_np, nbr_ids_np, valid_np,
               L, k, step_size, repulsion, dir_memory,
               social, social_dist_weight,
               pref_weighted, pref_inner, inner_avg,
               pref_dist_w, pref_dist_sigma, best_mode,
               torch_precision=2, torch_device_idx=0):
    """Full physics step — compatible interface used by simulation.py.

    On CUDA, uses persistent GPU state to minimize transfers.
    """
    device, dtype = _resolve_device_dtype(torch_precision, torch_device_idx)

    # Upload to GPU
    _gpu_state.upload(pos_np, prefs_np, response_np, dm_np, device, dtype)

    # Step on GPU
    mov_np = _gpu_state.step(
        nbr_ids_np, valid_np, L, k,
        step_size, repulsion, dir_memory,
        social, social_dist_weight,
        pref_weighted, pref_inner, inner_avg,
        pref_dist_w, pref_dist_sigma, best_mode)

    # Download results
    new_pos, new_prefs, new_resp, new_dm = _gpu_state.download()

    return new_pos, new_prefs, new_dm, mov_np
