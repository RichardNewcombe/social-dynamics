#!/usr/bin/env python3
"""
Taichi-accelerated social dynamics simulation.

Hybrid approach: pykdtree for neighbor finding + Taichi kernels for physics.
For GPU targets (Metal/CUDA), the neighbor search can also be Taichi-native.

Designed to scale to 100K+ particles and 1M+ steps.
"""

import argparse
import math
import time
import numpy as np

# ── Parse CLI args before ti.init ───────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Taichi social dynamics sim")
    p.add_argument("--N", type=int, default=None, help="Number of particles")
    p.add_argument("--K", type=int, default=None, help="Preference dimensions")
    p.add_argument("--steps", type=int, default=4000, help="Total simulation steps")
    p.add_argument("--n_neighbors", type=int, default=None, help="KNN count")
    p.add_argument("--step_size", type=float, default=None, help="Physics step size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--render_every", type=int, default=1000, help="Render image every N steps")
    p.add_argument("--img_size", type=int, default=960, help="Output image resolution")
    p.add_argument("--arch", type=str, default="cpu", choices=["auto", "cpu", "gpu", "metal", "cuda", "vulkan"])
    p.add_argument("--compare", action="store_true", help="Also run original sim for timing comparison")
    p.add_argument("--knn_mode", type=str, default="auto", choices=["auto", "pykdtree", "grid"],
                   help="KNN method: auto picks best for arch")
    return p.parse_args()

args = parse_args()

import taichi as ti

# ── Taichi init ─────────────────────────────────────────────────────
_arch_map = {
    "auto": ti.cpu,
    "cpu": ti.cpu,
    "gpu": ti.cuda,
    "cuda": ti.cuda,
    "metal": ti.metal,
    "vulkan": ti.vulkan,
}
ti.init(arch=_arch_map.get(args.arch, ti.cpu), default_fp=ti.f64)

# ── Load params from existing sim ───────────────────────────────────
from sim_2d_exp.params import params, SPACE

N = args.N or params['num_particles']
K = args.K or params['k']
N_NEIGHBORS = args.n_neighbors or params['n_neighbors']
STEP_SIZE = args.step_size or params['step_size']
SEED = args.seed
TOTAL_STEPS = args.steps
RENDER_EVERY = args.render_every
IMG_SIZE = args.img_size
L = SPACE

# KNN mode selection
USE_GRID_KNN = args.knn_mode == "grid" or (args.knn_mode == "auto" and args.arch not in ("auto", "cpu"))

print(f"[taichi] N={N}, K={K}, n_neighbors={N_NEIGHBORS}, step_size={STEP_SIZE}")
print(f"[taichi] KNN={('grid' if USE_GRID_KNN else 'pykdtree')}, steps={TOTAL_STEPS}")

# ── Taichi fields ───────────────────────────────────────────────────
pos = ti.Vector.field(2, dtype=ti.f64, shape=N)
new_pos = ti.Vector.field(2, dtype=ti.f64, shape=N)
prefs = ti.field(dtype=ti.f64, shape=(N, K))
response = ti.field(dtype=ti.f64, shape=(N, K))
movement = ti.Vector.field(2, dtype=ti.f64, shape=N)
nbr_ids_f = ti.field(dtype=ti.i32, shape=(N, N_NEIGHBORS))

# Grid KNN fields (only allocated if needed)
if USE_GRID_KNN:
    _expected_knn_dist = math.sqrt(N_NEIGHBORS / (math.pi * N)) * L
    CELL_SIZE = max(_expected_knn_dist * 1.5, L / 512)
    GRID_RES = max(4, min(512, int(math.ceil(L / CELL_SIZE))))
    CELL_SIZE = L / GRID_RES
    MAX_PPC = min(max(64, int(N / (GRID_RES * GRID_RES) * 8)), 512)
    cell_count = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES))
    cell_particles = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, MAX_PPC))
    particle_cell = ti.Vector.field(2, dtype=ti.i32, shape=N)
    nbr_dists_f = ti.field(dtype=ti.f64, shape=(N, N_NEIGHBORS))
    print(f"[taichi] grid: {GRID_RES}x{GRID_RES}, cell_size={CELL_SIZE:.6f}")


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


# ── Grid-based KNN kernels ─────────────────────────────────────────
if USE_GRID_KNN:
    @ti.kernel
    def build_cell_list():
        for i, j in cell_count:
            cell_count[i, j] = 0
        for p in range(N):
            cx = ti.cast(ti.floor(periodic_wrap(pos[p][0], L) / CELL_SIZE), ti.i32)
            cy = ti.cast(ti.floor(periodic_wrap(pos[p][1], L) / CELL_SIZE), ti.i32)
            cx = ti.min(ti.max(cx, 0), GRID_RES - 1)
            cy = ti.min(ti.max(cy, 0), GRID_RES - 1)
            particle_cell[p] = ti.Vector([cx, cy])
            idx = ti.atomic_add(cell_count[cx, cy], 1)
            if idx < MAX_PPC:
                cell_particles[cx, cy, idx] = p

    @ti.kernel
    def find_neighbors_grid():
        for p in range(N):
            px = pos[p]
            cx = particle_cell[p][0]
            cy = particle_cell[p][1]
            for slot in range(N_NEIGHBORS):
                nbr_ids_f[p, slot] = -1
                nbr_dists_f[p, slot] = 1e20
            worst_slot = 0
            worst_dist = 1e20
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    ncx = (cx + dx) % GRID_RES
                    ncy = (cy + dy) % GRID_RES
                    count = ti.min(cell_count[ncx, ncy], MAX_PPC)
                    for ci in range(count):
                        q = cell_particles[ncx, ncy, ci]
                        if q != p:
                            disp = periodic_dist_vec(px, pos[q], L)
                            dist_sq = disp[0] * disp[0] + disp[1] * disp[1]
                            if dist_sq < worst_dist:
                                nbr_ids_f[p, worst_slot] = q
                                nbr_dists_f[p, worst_slot] = dist_sq
                                worst_dist = nbr_dists_f[p, 0]
                                worst_slot = 0
                                for s in range(1, N_NEIGHBORS):
                                    if nbr_dists_f[p, s] > worst_dist:
                                        worst_dist = nbr_dists_f[p, s]
                                        worst_slot = s


# ── pykdtree-based KNN ─────────────────────────────────────────────
def find_neighbors_pykdtree():
    """Fast KNN using pykdtree with border replication for periodic BC."""
    from pykdtree.kdtree import KDTree as PyKDTree

    pos_np = pos.to_numpy()
    query_pos = pos_np % L

    margin = 0.15 * L
    bx_lo = query_pos[:, 0] < margin
    bx_hi = query_pos[:, 0] > L - margin
    by_lo = query_pos[:, 1] < margin
    by_hi = query_pos[:, 1] > L - margin
    border = bx_lo | bx_hi | by_lo | by_hi
    bpos = query_pos[border]
    bidx = np.where(border)[0]

    offsets = np.array([[-L, -L], [-L, 0], [-L, L],
                        [0, -L],           [0, L],
                        [L, -L],  [L, 0],  [L, L]])
    replicas = []
    rep_idx = []
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
    result = mapped[:, 1:].astype(np.int32)  # skip self
    nbr_ids_f.from_numpy(result)


# ── Physics kernel ──────────────────────────────────────────────────
@ti.kernel
def physics_step():
    """Per-dimension best-neighbor attraction (matches default PyTorch physics)."""
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


# ── Rendering ───────────────────────────────────────────────────────
def render_frame(step_num: int, filename: str):
    """Fast PIL-only renderer."""
    from PIL import Image, ImageDraw, ImageFont

    pos_np = pos.to_numpy()
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

    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    px = (pos_np[:, 0] / L * IMG_SIZE).astype(np.int32) % IMG_SIZE
    py = (IMG_SIZE - 1 - (pos_np[:, 1] / L * IMG_SIZE).astype(np.int32)) % IMG_SIZE
    rgb_u8 = (rgb * 255).astype(np.uint8)

    pixels = img.load()
    for i in range(N):
        x, y = int(px[i]), int(py[i])
        c = (int(rgb_u8[i, 0]), int(rgb_u8[i, 1]), int(rgb_u8[i, 2]))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pixels[(x + dx) % IMG_SIZE, (y + dy) % IMG_SIZE] = c

    draw = ImageDraw.Draw(img)
    text = f"Step {step_num}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    draw.text((11, 11), text, fill=(0, 0, 0), font=font)
    draw.text((9, 9), text, fill=(0, 0, 0), font=font)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    img.save(filename)


# ── Initialize ──────────────────────────────────────────────────────
def initialize():
    rng = np.random.default_rng(SEED)
    pos.from_numpy(rng.uniform(0, L, (N, 2)))
    prefs.from_numpy(rng.uniform(-1, 1, (N, K)))
    response.from_numpy(rng.uniform(-1, 1, (N, K)))


# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Taichi Social Dynamics Simulation")
    print(f"  N={N}  K={K}  neighbors={N_NEIGHBORS}  steps={TOTAL_STEPS}")
    print(f"{'='*60}\n")

    initialize()
    print("[init] Particles initialized")

    # Warm up kernels
    if USE_GRID_KNN:
        build_cell_list()
        find_neighbors_grid()
    else:
        find_neighbors_pykdtree()
    physics_step()
    ti.sync()
    print("[init] Kernels compiled")

    # Re-init after warmup
    initialize()

    t_search_total = 0.0
    t_physics_total = 0.0
    t_render_total = 0.0
    t_start = time.perf_counter()

    for step in range(1, TOTAL_STEPS + 1):
        # Neighbor finding
        t0 = time.perf_counter()
        if USE_GRID_KNN:
            build_cell_list()
            find_neighbors_grid()
            ti.sync()
        else:
            find_neighbors_pykdtree()
        t1 = time.perf_counter()
        t_search_total += t1 - t0

        # Physics
        physics_step()
        ti.sync()
        t2 = time.perf_counter()
        t_physics_total += t2 - t1

        # Render at intervals
        if step % RENDER_EVERY == 0 or step == TOTAL_STEPS:
            fname = f"sim_taichi_step_{step:06d}.png"
            t_r0 = time.perf_counter()
            render_frame(step, fname)
            t_r1 = time.perf_counter()
            t_render_total += t_r1 - t_r0

            elapsed = time.perf_counter() - t_start
            sps = step / (elapsed - t_render_total)
            print(f"  step {step:>7d}/{TOTAL_STEPS}  "
                  f"{sps:.1f} steps/s (sim only)  "
                  f"search={t_search_total/step*1000:.1f}ms  "
                  f"physics={t_physics_total/step*1000:.1f}ms  "
                  f"[render={t_r1-t_r0:.2f}s]")

    t_total = time.perf_counter() - t_start
    t_sim = t_total - t_render_total

    # Copy final frame
    import shutil
    final_name = f"sim_taichi_step_{TOTAL_STEPS:06d}.png"
    shutil.copy(final_name, "sim_taichi_final.png")

    print(f"\n{'='*60}")
    print(f"  DONE — {TOTAL_STEPS} steps in {t_sim:.2f}s (sim) + {t_render_total:.2f}s (render)")
    print(f"  Sim rate: {TOTAL_STEPS/t_sim:.1f} steps/s ({t_sim/TOTAL_STEPS*1000:.2f} ms/step)")
    print(f"  Search:  {t_search_total:.2f}s ({t_search_total/t_sim*100:.1f}%)")
    print(f"  Physics: {t_physics_total:.2f}s ({t_physics_total/t_sim*100:.1f}%)")
    print(f"{'='*60}")

    if args.compare:
        print("\n[compare] Running original simulation for comparison...")
        params['knn_method'] = 3
        params['num_particles'] = N
        params['k'] = K
        params['n_neighbors'] = N_NEIGHBORS
        from sim_2d_exp.simulation import Simulation
        sim = Simulation()
        n_cmp = min(500, TOTAL_STEPS)
        t0 = time.perf_counter()
        for _ in range(n_cmp):
            sim.step()
        t1 = time.perf_counter()
        print(f"[compare] Original: {n_cmp/(t1-t0):.1f} steps/s ({(t1-t0)/n_cmp*1000:.1f} ms/step)")


if __name__ == "__main__":
    main()
