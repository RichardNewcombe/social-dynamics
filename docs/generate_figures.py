"""Generate publication-quality figures for the knowledge manifold writeup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

from experiments.landscape import make_default_landscape
import importlib
kf_mod = importlib.import_module('3D_sim.knowledge_field')
KnowledgeField = kf_mod.KnowledgeField

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

landscape = make_default_landscape(seed=42)
G = 128
xs = np.linspace(-1, 1, G)
ys = np.linspace(-1, 1, G)
gx, gy = np.meshgrid(xs, ys, indexing='ij')
probes = np.zeros((G*G, 3), dtype=np.float64)
probes[:, 0] = gx.ravel()
probes[:, 1] = gy.ravel()
fitness_grid = landscape.fitness(probes).reshape(G, G)
f_min, f_max = fitness_grid.min(), fitness_grid.max()
fitness_norm = (fitness_grid - f_min) / (f_max - f_min)

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Hidden Fitness Landscape (2D heatmap + 3D surface)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 2D heatmap
ax = axes[0]
im = ax.imshow(fitness_norm.T, origin='lower', extent=[-1, 1, -1, 1],
               cmap='YlOrRd', vmin=0, vmax=1)
# Mark peaks
peaks = [
    ([0.6, 0.7], 1.0, 'Global summit'),
    ([-0.4, -0.3], 0.6, 'Foothill A'),
    ([0.3, -0.5], 0.5, 'Foothill B'),
    ([-0.7, 0.5], 0.35, 'Ridge'),
]
for (cx, cy), h, label in peaks:
    ax.plot(cx, cy, 'k+', markersize=8, markeredgewidth=1.5)
    ax.annotate(label, (cx, cy), textcoords='offset points',
                xytext=(5, 5), fontsize=7, color='black')
ax.set_xlabel('Skill dimension $p_0$')
ax.set_ylabel('Skill dimension $p_1$')
ax.set_title('(a) Hidden fitness $F(p_0, p_1)$')
plt.colorbar(im, ax=ax, shrink=0.85, label='Normalised fitness')

# 3D surface
ax2 = fig.add_subplot(122, projection='3d')
X, Y = gx[::2, ::2], gy[::2, ::2]
Z = fitness_norm[::2, ::2]
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(Z.T, cmap=cm.YlOrRd, vert_exag=0.5, blend_mode='soft')
ax2.plot_surface(X.T, Y.T, Z.T, facecolors=rgb, linewidth=0,
                 antialiased=True, shade=False)
ax2.set_xlabel('$p_0$')
ax2.set_ylabel('$p_1$')
ax2.set_zlabel('$F$')
ax2.set_title('(b) 3D view')
ax2.view_init(elev=30, azim=-60)
# Remove the flat 2D axes[1] since we replaced it
axes[1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_fitness_landscape.png'))
plt.close()
print("Saved fig1_fitness_landscape.png")

# ═══════════════════════════════════════════════════════════════════
# Figure 2: Structural Support Constraint illustration
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# Create a 1D cross-section illustration
x = np.linspace(0, 1, 200)

# Case 1: Narrow spike (rejected by support constraint)
spike = np.zeros_like(x)
spike[95:105] = 0.8
local_mean_spike = np.convolve(spike, np.ones(15)/15, mode='same')
max_allowed_spike = local_mean_spike + 0.4

ax = axes[0]
ax.fill_between(x, 0, spike, alpha=0.3, color='red', label='Attempted deposit')
ax.plot(x, max_allowed_spike, 'k--', linewidth=1.5, label='Support limit')
ax.plot(x, np.minimum(spike, max_allowed_spike), 'b-', linewidth=2, label='After constraint')
ax.set_xlabel('Skill position')
ax.set_ylabel('Knowledge height')
ax.set_title('(a) Narrow spike: rejected')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(0, 1.0)

# Case 2: Broad base (accepted)
broad = np.exp(-((x - 0.5)**2) / (2 * 0.08**2)) * 0.7
local_mean_broad = np.convolve(broad, np.ones(15)/15, mode='same')
max_allowed_broad = local_mean_broad + 0.4

ax = axes[1]
ax.fill_between(x, 0, broad, alpha=0.3, color='green', label='Broad deposit')
ax.plot(x, max_allowed_broad, 'k--', linewidth=1.5, label='Support limit')
ax.plot(x, np.minimum(broad, max_allowed_broad), 'b-', linewidth=2, label='After constraint')
ax.set_xlabel('Skill position')
ax.set_title('(b) Broad base: accepted')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(0, 1.0)

# Case 3: Two teams building shared base
team1 = np.exp(-((x - 0.45)**2) / (2 * 0.06**2)) * 0.5
team2 = np.exp(-((x - 0.55)**2) / (2 * 0.06**2)) * 0.5
combined = team1 + team2
local_mean_comb = np.convolve(combined, np.ones(15)/15, mode='same')
max_allowed_comb = local_mean_comb + 0.4

ax = axes[2]
ax.fill_between(x, 0, team1, alpha=0.3, color='blue', label='Team 1')
ax.fill_between(x, team1, combined, alpha=0.3, color='orange', label='Team 2')
ax.plot(x, max_allowed_comb, 'k--', linewidth=1.5, label='Support limit')
ax.set_xlabel('Skill position')
ax.set_title('(c) Shared base: higher ceiling')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_structural_support.png'))
plt.close()
print("Saved fig2_structural_support.png")

# ═══════════════════════════════════════════════════════════════════
# Figure 3: Exponential Reward Scaling
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

h_range = np.linspace(0, 1, 200)

# Left: different alpha values
ax = axes[0]
for alpha in [0, 1.0, 2.0, 3.0, 5.0]:
    if alpha == 0:
        r = h_range
        label = r'$\alpha=0$ (linear)'
    else:
        r = (np.exp(alpha * h_range) - 1.0) / (np.exp(alpha) - 1.0)
        label = rf'$\alpha={alpha:.0f}$'
    ax.plot(h_range, r, label=label, linewidth=1.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Knowledge height $h$')
ax.set_ylabel(r'Reward $R_{\mathrm{exp}}(h)$')
ax.set_title(r'(a) Exponential reward: $\frac{e^{\alpha h} - 1}{e^{\alpha} - 1}$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Right: combined reward with growth bonus
ax = axes[1]
alpha = 3.0
h_vals = [0.2, 0.5, 0.8]
growth_range = np.linspace(0, 1, 200)
for h in h_vals:
    exp_term = (np.exp(alpha * h) - 1.0) / (np.exp(alpha) - 1.0)
    combined = exp_term * (1.0 + 1.0 * growth_range)
    ax.plot(growth_range, combined, label=f'$h={h}$', linewidth=1.5)
ax.set_xlabel('Growth rate $g$')
ax.set_ylabel(r'Combined reward $R(h, g)$')
ax.set_title(r'(b) With growth bonus ($\beta=1$, $\alpha=3$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_reward_scaling.png'))
plt.close()
print("Saved fig3_reward_scaling.png")

# ═══════════════════════════════════════════════════════════════════
# Figure 4: Knowledge Growth Dynamics (from simulation)
# ═══════════════════════════════════════════════════════════════════
from scipy.spatial import cKDTree
mm_mod = importlib.import_module('3D_sim.mountain_mesh')
project_particles_to_knowledge_surface = mm_mod.project_particles_to_knowledge_surface

N = 300
K = 3
STEPS = 18000
rng = np.random.default_rng(42)

kf = KnowledgeField(grid_res=64, diffusion_sigma=0.5, decay=0.9999,
                    support_radius=3, max_slope=0.4)
kf.set_fitness_surface(landscape)

prefs = rng.uniform(-1, 1, (N, K)).astype(np.float32)
reward_ema = np.zeros(N, dtype=np.float64)
noise_level = np.full(N, 3.0, dtype=np.float64)

# Visionary setup
n_vis = max(1, int(N * 0.02))
vis_mask = np.zeros(N, dtype=bool)
vis_mask[:n_vis] = True

metrics = {'step': [], 'coverage': [], 'peak': [], 'mean_reward': [],
           'vis_dist': [], 'nonvis_dist': []}
GLOBAL_PEAK = np.array([0.6, 0.7])

snapshot_grids = {}
snapshot_steps = [0, 500, 2000, 6000, 12000, 18000]

for step in range(STEPS + 1):
    if step % 200 == 0:
        metrics['step'].append(step)
        metrics['coverage'].append(kf.coverage())
        metrics['peak'].append(kf.peak_knowledge())
        metrics['mean_reward'].append(reward_ema.mean())
        metrics['vis_dist'].append(np.linalg.norm(prefs[vis_mask, :2] - GLOBAL_PEAK, axis=1).mean())
        metrics['nonvis_dist'].append(np.linalg.norm(prefs[~vis_mask, :2] - GLOBAL_PEAK, axis=1).mean())

    if step in snapshot_steps:
        snapshot_grids[step] = kf.grid.copy()

    if step >= STEPS:
        break

    # Social step (simplified)
    eps = 1e-7
    mapped = np.zeros((N, 3), dtype=np.float64)
    mapped[:, 0] = np.clip((prefs[:, 0] + 1.0) * 0.5, 0, 1 - eps)
    mapped[:, 1] = np.clip((prefs[:, 1] + 1.0) * 0.5, 0, 1 - eps)
    mapped[:, 2] = np.clip((prefs[:, 2] + 1.0) * 0.5, 0, 1 - eps)
    tree = cKDTree(mapped, boxsize=1.0 + eps)
    _, nbr_ids = tree.query(mapped, k=21, workers=-1)
    nbr_ids = nbr_ids[:, 1:]

    # Social averaging
    nbr_prefs = prefs[nbr_ids]
    nbr_mean = nbr_prefs.mean(axis=1)
    prefs[:] = 0.99 * prefs + 0.01 * nbr_mean
    np.clip(prefs, -1, 1, out=prefs)

    # Reward-weighted social nudge
    if step > 0:
        nbr_reward = reward_ema[nbr_ids]
        w = 1.0 + 5.0 * np.maximum(nbr_reward, 0.0)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.maximum(w_sum, 1e-10)
        w_norm = w / w_sum
        weighted_mean = (nbr_prefs * w_norm[:, :, None]).sum(axis=1)
        prefs[:] = 0.995 * prefs + 0.005 * weighted_mean
        np.clip(prefs, -1, 1, out=prefs)

    # Knowledge gradient climbing
    know_grad = kf.knowledge_gradient(prefs[:, 0], prefs[:, 1])
    know_mag = np.linalg.norm(know_grad, axis=1, keepdims=True)
    know_mag = np.maximum(know_mag, 1e-10)
    know_dir = know_grad / know_mag
    lateral_nudge = 0.002 * know_dir

    # Visionary nudge
    probes_v = np.zeros((n_vis, 3), dtype=np.float64)
    probes_v[:, 0] = prefs[vis_mask, 0]
    probes_v[:, 1] = prefs[vis_mask, 1]
    vis_grad = landscape.gradient(probes_v)
    vis_mag = np.linalg.norm(vis_grad, axis=1, keepdims=True)
    vis_mag = np.maximum(vis_mag, 1e-10)
    vis_dir = vis_grad / vis_mag
    lateral_nudge[vis_mask] += 0.001 * vis_dir

    # Exploration
    explore_mask = rng.random(N) < 0.003
    n_explore = explore_mask.sum()
    if n_explore > 0:
        lateral_nudge[explore_mask] += rng.normal(0, 0.2, (n_explore, 2))

    prefs[:, 0] += lateral_nudge[:, 0].astype(prefs.dtype)
    prefs[:, 1] += lateral_nudge[:, 1].astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # Deposit
    amounts = np.full(N, 0.005, dtype=np.float64)
    actual_growth = kf.deposit_knowledge(prefs[:, 0], prefs[:, 1], amounts)

    # Reward
    heights = kf.sample_knowledge(prefs[:, 0], prefs[:, 1])
    exp_term = (np.exp(3.0 * heights) - 1.0) / (np.exp(3.0) - 1.0)
    growth_term = 1.0 + 1.0 * np.clip(actual_growth / 0.005, 0.0, 1.0)
    raw_reward = exp_term * growth_term
    np.clip(raw_reward, 0.0, 2.0, out=raw_reward)
    reward_ema[:] = 0.95 * reward_ema + 0.05 * raw_reward

    # Adaptive noise
    reward_median = np.median(reward_ema)
    productive = reward_ema > reward_median
    noise_level[productive] += 0.05 * (3.0 - noise_level[productive])
    noise_level[~productive] += 0.05 * (8.0 - noise_level[~productive])
    np.clip(noise_level, 3.0, 8.0, out=noise_level)

    # Diffuse
    kf.step_field()

print("Simulation complete. Generating figures...")

# Figure 4a: Knowledge growth snapshots (6 panels)
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for idx, (step, grid) in enumerate(snapshot_grids.items()):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(grid.T, origin='lower', extent=[-1, 1, -1, 1],
                   vmin=0, vmax=1, cmap='YlOrRd')
    ax.set_title(f'$t = {step}$', fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel('$p_1$')
    if idx >= 3:
        ax.set_xlabel('$p_0$')
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Knowledge manifold growth over time', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_knowledge_snapshots.png'))
plt.close()
print("Saved fig4_knowledge_snapshots.png")

# Figure 5: Coverage and peak over time + visionary drift
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes[0]
ax.plot(metrics['step'], metrics['coverage'], 'b-', linewidth=1.5, label='Coverage')
ax.plot(metrics['step'], metrics['peak'], 'r-', linewidth=1.5, label='Peak knowledge')
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4, label='$h = 0.9$')
ax.set_xlabel('Step $t$')
ax.set_ylabel('Value')
ax.set_title('(a) Knowledge growth')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.plot(metrics['step'], metrics['vis_dist'], 'b-', linewidth=1.5,
        label=f'Visionaries ($n={n_vis}$)')
ax.plot(metrics['step'], metrics['nonvis_dist'], color='orange', linewidth=1.5,
        label=f'Regular ($n={N-n_vis}$)')
ax.set_xlabel('Step $t$')
ax.set_ylabel('Mean distance to global peak')
ax.set_title('(b) Drift toward hidden peak')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.plot(metrics['step'], metrics['mean_reward'], 'g-', linewidth=1.5)
ax.set_xlabel('Step $t$')
ax.set_ylabel('Mean reward EMA')
ax.set_title('(c) Reward over time')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_dynamics.png'))
plt.close()
print("Saved fig5_dynamics.png")

# Figure 6: Final state comparison (knowledge vs fitness)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
im = ax.imshow(kf.grid.T, origin='lower', extent=[-1, 1, -1, 1],
               vmin=0, vmax=1, cmap='YlOrRd')
ax.scatter(prefs[vis_mask, 0], prefs[vis_mask, 1],
           c='cyan', s=30, marker='*', zorder=5, label='Visionaries')
ax.scatter(prefs[~vis_mask, 0], prefs[~vis_mask, 1],
           c='blue', s=3, alpha=0.3, label='Regular')
ax.scatter(0.6, 0.7, c='lime', s=80, marker='x', linewidths=2,
           zorder=10, label='Global peak')
ax.set_xlabel('$p_0$')
ax.set_ylabel('$p_1$')
ax.set_title(f'(a) Knowledge at $t={STEPS}$')
ax.legend(fontsize=7, loc='lower left')
plt.colorbar(im, ax=ax, shrink=0.85)

ax = axes[1]
im2 = ax.imshow(fitness_norm[:64, :64].T, origin='lower', extent=[-1, 1, -1, 1],
                vmin=0, vmax=1, cmap='YlOrRd')
ax.scatter(0.6, 0.7, c='lime', s=80, marker='x', linewidths=2, zorder=10)
ax.set_xlabel('$p_0$')
ax.set_ylabel('$p_1$')
ax.set_title('(b) Hidden fitness ceiling $F$')
plt.colorbar(im2, ax=ax, shrink=0.85)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_final_state.png'))
plt.close()
print("Saved fig6_final_state.png")

# Figure 7: Adaptive noise illustration
fig, ax = plt.subplots(figsize=(5, 3.5))
reward_range = np.linspace(0, 1, 200)
median_r = 0.5
noise_productive = 3.0 + 0.05 * (3.0 - 3.0) * np.ones_like(reward_range[:100])
noise_stuck = 3.0 + 0.05 * (8.0 - 3.0) * np.ones_like(reward_range[100:])

# Show the noise level as function of reward relative to median
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Median reward')
ax.annotate('Productive\n(noise $\\to \\sigma_{\\mathrm{base}}$)',
            xy=(0.7, 4.0), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.annotate('Stuck\n(noise $\\to \\sigma_{\\mathrm{max}}$)',
            xy=(0.3, 6.5), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Simulate convergence
steps_adapt = np.arange(100)
noise_prod_trace = 5.5 * np.exp(-0.05 * steps_adapt) + 3.0 * (1 - np.exp(-0.05 * steps_adapt))
noise_stuck_trace = 3.0 * np.exp(-0.05 * steps_adapt) + 8.0 * (1 - np.exp(-0.05 * steps_adapt))
ax.plot(steps_adapt / 100, noise_prod_trace, 'g-', linewidth=2, label='Productive particle')
ax.plot(steps_adapt / 100, noise_stuck_trace, 'r-', linewidth=2, label='Stuck particle')
ax.set_xlabel('Time (normalised)')
ax.set_ylabel('Noise level $\\sigma_i$')
ax.set_title('Adaptive noise dynamics')
ax.legend(fontsize=8)
ax.set_ylim(2, 9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_adaptive_noise.png'))
plt.close()
print("Saved fig7_adaptive_noise.png")

print("\nAll figures generated!")
