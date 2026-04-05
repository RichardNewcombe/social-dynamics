# sim_2d v2

Snapshot date: 2026-03-13

## Changes since v1

### Signal / Response split (Factorization A)
- Each particle has two K-dim vectors: `signal` (what others read for neighbor selection) and `response` (how the particle weights interactions for movement)
- `use_signal_response` toggle enables the split
- `swap_signal_response` swaps the roles (signal used for weighting, response for selection) to test whether asymmetric clustering patterns come from dynamics vs initialization
- Swap implemented as array swap before/after `_step_impl` to avoid writeback corruption
- Both initialized via shared `_fill_pref_array()` — same distribution, independent draws
- Visualization toggle: "Visualize" combo switches left panel + pref space views between Signal and Response coloring
- Trail FBOs cleared on vis source change for immediate feedback

### Negative social learning (differentiation)
- Social slider range expanded to [-0.01, +0.01] with "0" reset button
- `social > 0` guard changed to `social != 0` across all three engines (Numba/NumPy/PyTorch)
- Keyboard shortcuts updated to match new range

### Quiet-dim differentiation (social_mode=1)
- Per-dimension social learning rate scaled by (1 - importance)
- Importance = normalized per-dim contribution to movement |compat_d|
- High-contribution dims preserved, low-contribution dims differentiated
- Self-regulating: as quiet dims grow, their importance rises, rate drops

### Best Neighbor modes
- `best_by_magnitude` (bool) replaced with `best_mode` (int: 0/1/2)
- Mode 2: Same-Sign Max Magnitude — skip opposite-sign neighbors, find max |pref| among same-sign
- All three engines handle no-valid-neighbor edge case: zero contribution + direction memory decay
- Torch/NumPy: `any_valid` mask prevents argmax from selecting wrong-sign neighbor on all-inf scores

### Binary d0 + noise init (pref_dist=4)
- Dim 0: half particles +1, half -1
- Dims 1+: uniform noise in [-eps, eps], controlled by slider

### Pref space visualization
- Right panel options: Pref2D (scatter dim0 vs dim1) and Pref3D (isometric projection of dims 0,1,2)
- Trail accumulation in pref space shows temporal preference evolution
- Three color modes: RGB, Dim2 Heat (blue-white-red), HSV
- Axis lines drawn on both views (R=d0, G=d1, B=d2, 120 degrees apart in isometric)
- FBO cleared on view mode switch

### Observations
- Signal and response show different clustering patterns under dynamics even from identical distributions
- This asymmetry comes from the physics: signal determines WHO you select, response determines HOW STRONGLY you react — the vector being "read by others" (signal) experiences different evolutionary pressure than the vector used for "self-weighting" (response)
