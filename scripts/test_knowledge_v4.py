"""
Headless verification test for knowledge manifold v4:
  - Visionary-only hidden gradient (non-visionaries are blind)
  - Exponentially-scaled absolute height reward with growth bonus
  - Timing calibration: peak ~0.9 in ~15,000 steps

Tests:
  1. Non-visionary particles do NOT drift toward the hidden peak
  2. Visionary particles DO drift toward the hidden peak
  3. Reward is exponentially weighted but not overwhelming
  4. Knowledge manifold reaches ~0.9 peak in ~15,000 steps
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from experiments.landscape import make_default_landscape
import importlib
kf_mod = importlib.import_module('3D_sim.knowledge_field')
mm_mod = importlib.import_module('3D_sim.mountain_mesh')
KnowledgeField = kf_mod.KnowledgeField
project_particles_to_knowledge_surface = mm_mod.project_particles_to_knowledge_surface

# ── Parameters (matching main.py mountain_params defaults) ──────────
N = 300
K = 3
STEPS = 18000  # ~5 minutes at 60fps with steps_per_frame=1
GRID_RES = 64
Z_SCALE = 0.5

# Research dynamics
NOISE_BASE = 3.0
NOISE_MAX = 8.0
NOISE_ADAPT = 0.05
WRITE_RATE = 0.005
KNOW_CLIMB_RATE = 0.002

# Visionary
VIS_FRACTION = 0.02
VIS_NUDGE = 0.001

# Knowledge field
DIFFUSION_SIGMA = 0.5
DECAY = 0.9999
SUPPORT_RADIUS = 3
MAX_SLOPE = 0.4

# Reward
REWARD_EMA_ALPHA = 0.05
REWARD_EXP_SCALE = 3.0
REWARD_GROWTH_BETA = 1.0
REWARD_SOCIAL_SCALE = 5.0

# Social dynamics
SOCIAL = 0.01
N_NEIGHBORS = 20

# Exploration
EXPLORE_PROB = 0.003
EXPLORE_RADIUS = 0.2

# Global peak location (from landscape definition)
GLOBAL_PEAK = np.array([0.6, 0.7])

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)


# ── Minimal simulation ──────────────────────────────────────────────
class MinimalSim:
    def __init__(self, n, k, rng):
        self.n = n
        self.k = k
        self.prefs = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.nbr_ids = None
        self._valid_mask = None
        self._reward_ema = None

    def step(self):
        n = self.n
        eps = 1e-7
        mapped = np.zeros((n, 3), dtype=np.float64)
        mapped[:, 0] = np.clip((self.prefs[:, 0] + 1.0) * 0.5, 0, 1 - eps)
        mapped[:, 1] = np.clip((self.prefs[:, 1] + 1.0) * 0.5, 0, 1 - eps)
        mapped[:, 2] = np.clip((self.prefs[:, 2] + 1.0) * 0.5, 0, 1 - eps)
        tree = cKDTree(mapped, boxsize=1.0 + eps)
        k_nbr = min(N_NEIGHBORS + 1, n)
        _, nbr_ids = tree.query(mapped, k=k_nbr, workers=-1)
        self.nbr_ids = nbr_ids[:, 1:].astype(np.int64)
        self._valid_mask = None
        nbr_prefs = self.prefs[self.nbr_ids]
        nbr_mean = nbr_prefs.mean(axis=1)
        self.prefs[:] = (1.0 - SOCIAL) * self.prefs + SOCIAL * nbr_mean
        np.clip(self.prefs, -1, 1, out=self.prefs)


def reward_social_nudge(sim, reward_ema):
    if REWARD_SOCIAL_SCALE <= 0 or sim.nbr_ids is None:
        return
    prefs = sim.prefs
    nbr_ids = sim.nbr_ids
    nbr_reward = reward_ema[nbr_ids]
    w = 1.0 + REWARD_SOCIAL_SCALE * np.maximum(nbr_reward, 0.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-10)
    w_norm = w / w_sum
    nbr_prefs = prefs[nbr_ids]
    weighted_mean = (nbr_prefs * w_norm[:, :, None]).sum(axis=1)
    nudge_strength = SOCIAL * 0.5
    prefs[:] = (1.0 - nudge_strength) * prefs + nudge_strength * weighted_mean
    np.clip(prefs, -1, 1, out=prefs)


def knowledge_step(sim, kf, landscape, rng, reward_ema, noise_level):
    """One step of knowledge manifold dynamics (v4: visionary-only gradient)."""
    n = sim.n
    prefs = sim.prefs
    nbr_ids = sim.nbr_ids
    if nbr_ids is None:
        return

    # ── 1. Knowledge gradient climbing (all particles) ──
    know_grad = kf.knowledge_gradient(prefs[:, 0], prefs[:, 1])
    know_mag = np.linalg.norm(know_grad, axis=1, keepdims=True)
    know_mag = np.maximum(know_mag, 1e-10)
    know_dir = know_grad / know_mag
    lateral_nudge = KNOW_CLIMB_RATE * know_dir

    # ── 2. Visionary nudge (rare particles only) ──
    n_vis = max(1, int(n * VIS_FRACTION))
    vis_mask = np.zeros(n, dtype=bool)
    vis_mask[:n_vis] = True

    probes = np.zeros((n_vis, 3), dtype=np.float64)
    probes[:, 0] = prefs[vis_mask, 0]
    probes[:, 1] = prefs[vis_mask, 1]
    vis_grad = landscape.gradient(probes)
    vis_mag = np.linalg.norm(vis_grad, axis=1, keepdims=True)
    vis_mag = np.maximum(vis_mag, 1e-10)
    vis_dir = vis_grad / vis_mag
    lateral_nudge[vis_mask] += VIS_NUDGE * vis_dir

    # ── 3. Exploration ──
    if EXPLORE_PROB > 0:
        explore_mask = rng.random(n) < EXPLORE_PROB
        n_explore = explore_mask.sum()
        if n_explore > 0:
            jump = rng.normal(0, EXPLORE_RADIUS, (n_explore, 2))
            lateral_nudge[explore_mask] += jump

    prefs[:, 0] += lateral_nudge[:, 0].astype(prefs.dtype)
    prefs[:, 1] += lateral_nudge[:, 1].astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # ── 4. Deposit knowledge ──
    amounts = np.full(n, WRITE_RATE, dtype=np.float64)
    actual_growth = kf.deposit_knowledge(prefs[:, 0], prefs[:, 1], amounts)

    # ── 5. Reward tracking (exp-scaled height * growth bonus) ──
    heights = kf.sample_knowledge(prefs[:, 0], prefs[:, 1])
    if REWARD_EXP_SCALE > 0:
        exp_term = (np.exp(REWARD_EXP_SCALE * heights) - 1.0) / (np.exp(REWARD_EXP_SCALE) - 1.0)
    else:
        exp_term = heights
    growth_term = 1.0 + REWARD_GROWTH_BETA * np.clip(actual_growth / max(WRITE_RATE, 1e-10), 0.0, 1.0)
    raw_reward = exp_term * growth_term
    np.clip(raw_reward, 0.0, 2.0, out=raw_reward)
    reward_ema[:] = (1.0 - REWARD_EMA_ALPHA) * reward_ema + REWARD_EMA_ALPHA * raw_reward
    sim._reward_ema = reward_ema

    # ── 6. Adaptive noise ──
    reward_median = np.median(reward_ema)
    productive = reward_ema > reward_median
    stuck = ~productive
    noise_level[productive] += NOISE_ADAPT * (NOISE_BASE - noise_level[productive])
    noise_level[stuck] += NOISE_ADAPT * (NOISE_MAX - noise_level[stuck])
    np.clip(noise_level, NOISE_BASE, NOISE_MAX, out=noise_level)

    # ── 7. Diffuse ──
    kf.step_field()

    # ── 8. Sync positions ──
    projected = project_particles_to_knowledge_surface(kf, prefs, z_scale=Z_SCALE)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)


# ── Run simulation ──────────────────────────────────────────────────
print("Setting up...")
landscape = make_default_landscape(seed=42)
kf = KnowledgeField(
    grid_res=GRID_RES,
    diffusion_sigma=DIFFUSION_SIGMA,
    decay=DECAY,
    support_radius=SUPPORT_RADIUS,
    max_slope=MAX_SLOPE,
)
kf.set_fitness_surface(landscape)

sim = MinimalSim(N, K, rng)
reward_ema = np.zeros(N, dtype=np.float64)
noise_level = np.full(N, NOISE_BASE, dtype=np.float64)

# Initial position sync
projected = project_particles_to_knowledge_surface(kf, sim.prefs, z_scale=Z_SCALE)
sim.pos[:, :3] = projected.astype(sim.pos.dtype)

# Track metrics
metrics = {
    'step': [], 'coverage': [], 'peak_knowledge': [],
    'mean_reward': [], 'mean_noise': [],
    'vis_dist_to_peak': [], 'nonvis_dist_to_peak': [],
    'reward_at_04': [], 'reward_at_08': [],
}

# Visionary mask
n_vis = max(1, int(N * VIS_FRACTION))
vis_mask = np.zeros(N, dtype=bool)
vis_mask[:n_vis] = True
nonvis_mask = ~vis_mask

SNAPSHOT_STEPS = [0, 1000, 3000, 6000, 10000, 15000, 18000]
snapshots = {}

print(f"Running {STEPS} steps (N={N}, vis={n_vis})...")
import time
t_start = time.perf_counter()

for step in range(STEPS + 1):
    if step in SNAPSHOT_STEPS:
        print(f"  Step {step:6d}: coverage={kf.coverage():.3f}, "
              f"peak={kf.peak_knowledge():.3f}, "
              f"mean_reward={reward_ema.mean():.4f}")
        snapshots[step] = {
            'knowledge_grid': kf.grid.copy(),
            'prefs': sim.prefs.copy(),
            'reward_ema': reward_ema.copy(),
        }

    if step % 100 == 0:
        metrics['step'].append(step)
        metrics['coverage'].append(kf.coverage())
        metrics['peak_knowledge'].append(kf.peak_knowledge())
        metrics['mean_reward'].append(reward_ema.mean())
        metrics['mean_noise'].append(noise_level.mean())

        # Distance of visionaries vs non-visionaries to global peak
        vis_prefs = sim.prefs[vis_mask, :2]
        nonvis_prefs = sim.prefs[nonvis_mask, :2]
        vis_dist = np.linalg.norm(vis_prefs - GLOBAL_PEAK, axis=1).mean()
        nonvis_dist = np.linalg.norm(nonvis_prefs - GLOBAL_PEAK, axis=1).mean()
        metrics['vis_dist_to_peak'].append(vis_dist)
        metrics['nonvis_dist_to_peak'].append(nonvis_dist)

        # Reward at different heights (for exp-scale verification)
        if REWARD_EXP_SCALE > 0:
            r04 = (np.exp(REWARD_EXP_SCALE * 0.4) - 1.0) / (np.exp(REWARD_EXP_SCALE) - 1.0)
            r08 = (np.exp(REWARD_EXP_SCALE * 0.8) - 1.0) / (np.exp(REWARD_EXP_SCALE) - 1.0)
        else:
            r04, r08 = 0.4, 0.8
        metrics['reward_at_04'].append(r04)
        metrics['reward_at_08'].append(r08)

    if step < STEPS:
        sim.step()
        if step > 0:
            reward_social_nudge(sim, reward_ema)
        knowledge_step(sim, kf, landscape, rng, reward_ema, noise_level)

t_elapsed = time.perf_counter() - t_start
print(f"\nCompleted {STEPS} steps in {t_elapsed:.1f}s ({STEPS/t_elapsed:.0f} steps/sec)")

# ── Verification checks ─────────────────────────────────────────────
print("\n=== VERIFICATION ===")

# 1. Visionary drift check
final_vis_dist = metrics['vis_dist_to_peak'][-1]
final_nonvis_dist = metrics['nonvis_dist_to_peak'][-1]
initial_vis_dist = metrics['vis_dist_to_peak'][0]
initial_nonvis_dist = metrics['nonvis_dist_to_peak'][0]

vis_drift = initial_vis_dist - final_vis_dist
nonvis_drift = initial_nonvis_dist - final_nonvis_dist

print(f"\n1. VISIONARY DRIFT CHECK:")
print(f"   Visionaries:     initial dist={initial_vis_dist:.3f} → final={final_vis_dist:.3f} (drift={vis_drift:.3f})")
print(f"   Non-visionaries: initial dist={initial_nonvis_dist:.3f} → final={final_nonvis_dist:.3f} (drift={nonvis_drift:.3f})")
if vis_drift > nonvis_drift and vis_drift > 0:
    print(f"   ✓ Visionaries drift MORE toward peak than non-visionaries")
else:
    print(f"   ✗ WARNING: Visionaries should drift more toward peak!")

# 2. Reward scaling check
r04 = metrics['reward_at_04'][-1]
r08 = metrics['reward_at_08'][-1]
ratio = r08 / max(r04, 1e-10)
print(f"\n2. REWARD SCALING CHECK:")
print(f"   Reward at h=0.4: {r04:.4f}")
print(f"   Reward at h=0.8: {r08:.4f}")
print(f"   Ratio (0.8/0.4): {ratio:.1f}x")
if 2.0 < ratio < 20.0:
    print(f"   ✓ Ratio is in reasonable range (2-20x)")
else:
    print(f"   ✗ WARNING: Ratio {ratio:.1f}x is outside reasonable range!")

# 3. Timing check
peak = kf.peak_knowledge()
coverage = kf.coverage()
print(f"\n3. TIMING CHECK:")
print(f"   Peak knowledge: {peak:.3f}")
print(f"   Coverage: {coverage:.3f}")

# Find step where peak first exceeds 0.9
peak_09_step = None
for i, pk in enumerate(metrics['peak_knowledge']):
    if pk >= 0.9:
        peak_09_step = metrics['step'][i]
        break
if peak_09_step is not None:
    print(f"   Peak reached 0.9 at step {peak_09_step}")
    if 10000 <= peak_09_step <= 20000:
        print(f"   ✓ Within target range (10k-20k steps)")
    else:
        print(f"   ✗ Outside target range (10k-20k steps)")
else:
    print(f"   ✗ Peak never reached 0.9 in {STEPS} steps")

# 4. Reward distribution check
final_rewards = reward_ema.copy()
print(f"\n4. REWARD DISTRIBUTION:")
print(f"   Mean:   {final_rewards.mean():.4f}")
print(f"   Median: {np.median(final_rewards):.4f}")
print(f"   Max:    {final_rewards.max():.4f}")
print(f"   Min:    {final_rewards.min():.4f}")
print(f"   Std:    {final_rewards.std():.4f}")

# ── Render diagnostic plots ─────────────────────────────────────────
print("\nRendering plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Coverage and peak
ax = axes[0, 0]
ax.plot(metrics['step'], metrics['coverage'], label='Coverage')
ax.plot(metrics['step'], metrics['peak_knowledge'], label='Peak Knowledge')
ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='Target 0.9')
ax.set_xlabel('Step')
ax.set_ylabel('Value')
ax.set_title('Knowledge Growth')
ax.legend()
ax.grid(True, alpha=0.3)

# Distance to peak (visionary vs non-visionary)
ax = axes[0, 1]
ax.plot(metrics['step'], metrics['vis_dist_to_peak'], label=f'Visionaries (n={n_vis})')
ax.plot(metrics['step'], metrics['nonvis_dist_to_peak'], label=f'Non-visionaries (n={N-n_vis})')
ax.set_xlabel('Step')
ax.set_ylabel('Mean dist to global peak')
ax.set_title('Drift Toward Hidden Peak')
ax.legend()
ax.grid(True, alpha=0.3)

# Reward distribution over time
ax = axes[0, 2]
ax.plot(metrics['step'], metrics['mean_reward'], label='Mean Reward')
ax.set_xlabel('Step')
ax.set_ylabel('Reward EMA')
ax.set_title('Mean Reward Over Time')
ax.grid(True, alpha=0.3)

# Knowledge heatmap at final step
ax = axes[1, 0]
im = ax.imshow(kf.grid.T, origin='lower', extent=[-1, 1, -1, 1],
               vmin=0, vmax=1, cmap='YlOrRd')
ax.scatter(sim.prefs[vis_mask, 0], sim.prefs[vis_mask, 1],
           c='cyan', s=20, marker='*', label='Visionaries', zorder=5)
ax.scatter(sim.prefs[nonvis_mask, 0], sim.prefs[nonvis_mask, 1],
           c='blue', s=3, alpha=0.3, label='Regular')
ax.scatter(*GLOBAL_PEAK, c='lime', s=100, marker='x', linewidths=2,
           label='Global Peak', zorder=10)
ax.set_title(f'Final Knowledge (step {STEPS})')
ax.legend(fontsize=7)
plt.colorbar(im, ax=ax)

# Fitness ceiling for comparison
ax = axes[1, 1]
im2 = ax.imshow(kf._fitness_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                vmin=0, vmax=1, cmap='YlOrRd')
ax.scatter(*GLOBAL_PEAK, c='lime', s=100, marker='x', linewidths=2)
ax.set_title('Hidden Fitness Ceiling')
plt.colorbar(im2, ax=ax)

# Noise over time
ax = axes[1, 2]
ax.plot(metrics['step'], metrics['mean_noise'])
ax.set_xlabel('Step')
ax.set_ylabel('Mean Noise Level')
ax.set_title('Adaptive Noise')
ax.grid(True, alpha=0.3)

fig.suptitle('Knowledge Manifold v4: Visionary-Only Gradient + Exp Reward', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'knowledge_v4_verification.png'), dpi=150)
plt.close()
print(f"Saved: {os.path.join(RESULTS_DIR, 'knowledge_v4_verification.png')}")

# ── Reward scaling curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
h_range = np.linspace(0, 1, 100)
for alpha in [1.0, 2.0, 3.0, 4.0, 5.0]:
    r = (np.exp(alpha * h_range) - 1.0) / (np.exp(alpha) - 1.0)
    ax.plot(h_range, r, label=f'alpha={alpha}')
ax.axvline(0.4, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0.8, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Knowledge Height')
ax.set_ylabel('Reward (normalized)')
ax.set_title('Exponential Reward Scaling Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'reward_scaling_curves.png'), dpi=150)
plt.close()
print(f"Saved: {os.path.join(RESULTS_DIR, 'reward_scaling_curves.png')}")

print("\nDone!")
