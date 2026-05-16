#!/usr/bin/env python3
"""
Render trail view of the Taichi social dynamics simulation.

Runs the simulation for 4000 steps, captures position history for the 
last 200 steps, then composites a trail image with exponential decay.
"""

import sys
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ───────────────────────────────────────────────────
TOTAL_STEPS = 4000
TRAIL_HISTORY = 200       # save last N steps of positions
DECAY = 0.95              # per-step brightness decay (faster = shorter trails but less saturation)
IMG_SIZE = 960
DOT_RADIUS = 2            # px radius for current-frame particles
TRAIL_RADIUS = 1          # px radius for trail dots
SEED = 42

# ── Set up Taichi ───────────────────────────────────────────────────
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from sim_2d_exp.params import params, SPACE

N = params['num_particles']
K = params['k']
N_NEIGHBORS = params['n_neighbors']
STEP_SIZE = params['step_size']
L = SPACE

print(f"[trails] N={N}, K={K}, neighbors={N_NEIGHBORS}, steps={TOTAL_STEPS}, trail_history={TRAIL_HISTORY}")

# ── Taichi fields (same as sim_taichi.py) ───────────────────────────
pos = ti.Vector.field(2, dtype=ti.f64, shape=N)
new_pos = ti.Vector.field(2, dtype=ti.f64, shape=N)
prefs = ti.field(dtype=ti.f64, shape=(N, K))
response = ti.field(dtype=ti.f64, shape=(N, K))
movement = ti.Vector.field(2, dtype=ti.f64, shape=N)
nbr_ids_f = ti.field(dtype=ti.i32, shape=(N, N_NEIGHBORS))

# ── Periodic helpers ────────────────────────────────────────────────
@ti.func
def periodic_dist_vec(a: ti.template(), b: ti.template(), Lv: float) -> ti.math.vec2:
    d = b - a
    d[0] = d[0] - Lv * ti.round(d[0] / Lv)
    d[1] = d[1] - Lv * ti.round(d[1] / Lv)
    return d

@ti.func
def periodic_wrap(v: float, Lv: float) -> float:
    r = v % Lv
    if r < 0.0:
        r += Lv
    return r

# ── pykdtree KNN ───────────────────────────────────────────────────
def find_neighbors_pykdtree():
    from pykdtree.kdtree import KDTree as PyKDTree
    pos_np = pos.to_numpy()
    query_pos = pos_np % L
    margin = 0.15 * L
    border = ((query_pos[:, 0] < margin) | (query_pos[:, 0] > L - margin) |
              (query_pos[:, 1] < margin) | (query_pos[:, 1] > L - margin))
    bpos = query_pos[border]
    bidx = np.where(border)[0]
    offsets = np.array([[-L, -L], [-L, 0], [-L, L],
                        [0, -L],           [0, L],
                        [L, -L],  [L, 0],  [L, L]])
    replicas, rep_idx = [], []
    for off in offsets:
        replicas.append(bpos + off)
        rep_idx.append(bidx)
    if replicas:
        all_pos = np.vstack([query_pos] + replicas)
        all_idx = np.concatenate([np.arange(N, dtype=np.int64)] + rep_idx)
    else:
        all_pos = query_pos
        all_idx = np.arange(N, dtype=np.int64)
    tree = PyKDTree(all_pos)
    _, raw_ids = tree.query(query_pos, k=N_NEIGHBORS + 1)
    mapped = all_idx[raw_ids]
    result = mapped[:, 1:].astype(np.int32)
    nbr_ids_f.from_numpy(result)

# ── Physics kernel (same as sim_taichi.py) ──────────────────────────
@ti.kernel
def physics_step():
    for p in range(N):
        mv = ti.math.vec2(0.0, 0.0)
        for ki in range(K):
            best_nbr = 0
            best_score = -1e20
            found = 0
            for s in range(N_NEIGHBORS):
                nid = nbr_ids_f[p, s]
                if nid >= 0:
                    score = prefs[nid, ki]
                    if score > best_score:
                        best_score = score
                        best_nbr = nid
                        found = 1
            if found == 1:
                disp = periodic_dist_vec(pos[p], pos[best_nbr], L)
                dist = ti.sqrt(disp[0] * disp[0] + disp[1] * disp[1])
                unit_dir = ti.math.vec2(0.0, 0.0)
                if dist > 1e-12:
                    unit_dir = disp / dist
                compat = response[p, ki] * prefs[best_nbr, ki]
                mv += compat * unit_dir
        new_x = periodic_wrap(pos[p][0] + STEP_SIZE * mv[0], L)
        new_y = periodic_wrap(pos[p][1] + STEP_SIZE * mv[1], L)
        pos[p] = ti.math.vec2(new_x, new_y)
        movement[p] = mv

# ── Initialize ──────────────────────────────────────────────────────
def initialize():
    rng = np.random.default_rng(SEED)
    pos.from_numpy(rng.uniform(0, L, (N, 2)))
    prefs.from_numpy(rng.uniform(-1, 1, (N, K)))
    response.from_numpy(rng.uniform(-1, 1, (N, K)))

# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Trail Renderer — Taichi Social Dynamics")
    print(f"{'='*60}\n")

    initialize()

    # Warm up
    find_neighbors_pykdtree()
    physics_step()
    ti.sync()
    print("[init] Kernels compiled")
    initialize()

    # Get particle colors from preferences (fixed at init)
    prefs_np = prefs.to_numpy()
    if K >= 3:
        rgb = np.clip((prefs_np[:, :3] + 1.0) * 0.5, 0, 1)
    elif K == 2:
        rgb = np.zeros((N, 3))
        rgb[:, :2] = np.clip((prefs_np[:, :2] + 1.0) * 0.5, 0, 1)
        rgb[:, 2] = 0.5
    else:
        rgb = np.zeros((N, 3))
        rgb[:, 0] = np.clip((prefs_np[:, 0] + 1.0) * 0.5, 0, 1)
        rgb[:, 1] = 0.5
        rgb[:, 2] = 0.5
    particle_colors = (rgb * 255).astype(np.uint8)  # (N, 3)

    # Ring buffer for position history
    history_start = TOTAL_STEPS - TRAIL_HISTORY  # start saving from this step
    pos_history = []  # list of (N, 2) arrays

    t_start = time.perf_counter()
    for step in range(1, TOTAL_STEPS + 1):
        find_neighbors_pykdtree()
        physics_step()
        ti.sync()

        # Save position history for the last TRAIL_HISTORY steps
        if step > history_start:
            pos_history.append(pos.to_numpy().copy())

        if step % 500 == 0:
            elapsed = time.perf_counter() - t_start
            sps = step / elapsed
            print(f"  step {step:>7d}/{TOTAL_STEPS}  {sps:.1f} steps/s", end="")
            if step > history_start:
                print(f"  (capturing trail: {len(pos_history)}/{TRAIL_HISTORY})", end="")
            print()

    t_sim = time.perf_counter() - t_start
    print(f"\n[sim] Done — {TOTAL_STEPS} steps in {t_sim:.1f}s ({TOTAL_STEPS/t_sim:.1f} steps/s)")
    print(f"[trail] Captured {len(pos_history)} frames of history")

    # ── Render trail composite ──────────────────────────────────────
    print("[render] Compositing trail image...")
    t_render_start = time.perf_counter()

    # Use float canvas for smooth decay
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)

    for frame_idx, frame_pos in enumerate(pos_history):
        # Apply decay to entire canvas
        canvas *= DECAY

        # Map positions to pixel coords
        px = ((frame_pos[:, 0] / L) * IMG_SIZE).astype(np.int32) % IMG_SIZE
        py = ((1.0 - frame_pos[:, 1] / L) * IMG_SIZE).astype(np.int32) % IMG_SIZE

        # Determine dot radius: larger for the last frame (current positions)
        is_last = (frame_idx == len(pos_history) - 1)
        r = DOT_RADIUS if is_last else TRAIL_RADIUS

        # Draw particles — direct paint (replaces pixel with particle color)
        for i in range(N):
            x, y = int(px[i]), int(py[i])
            cr = float(particle_colors[i, 0])
            cg = float(particle_colors[i, 1])
            cb = float(particle_colors[i, 2])
            # Full brightness for current, dimmer for trails
            if is_last:
                brightness = 1.0
            else:
                brightness = 0.65
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        xi = (x + dx) % IMG_SIZE
                        yi = (y + dy) % IMG_SIZE
                        # Direct paint — overwrites pixel with particle color
                        canvas[yi, xi, 0] = cr * brightness
                        canvas[yi, xi, 1] = cg * brightness
                        canvas[yi, xi, 2] = cb * brightness

        if (frame_idx + 1) % 50 == 0:
            print(f"  rendered frame {frame_idx+1}/{len(pos_history)}")

    # Convert to image
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    img = Image.fromarray(canvas_u8, 'RGB')

    # Add step counter overlay
    draw = ImageDraw.Draw(img)
    text = f"Step {TOTAL_STEPS}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    # Black shadow
    for ox, oy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]:
        draw.text((10+ox, 10+oy), text, fill=(0, 0, 0), font=font)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    output_path = "sim_taichi_trails.png"
    img.save(output_path)

    t_render = time.perf_counter() - t_render_start
    print(f"[render] Trail image saved to {output_path} ({t_render:.1f}s)")
    print(f"[total] {t_sim + t_render:.1f}s")


if __name__ == "__main__":
    main()
