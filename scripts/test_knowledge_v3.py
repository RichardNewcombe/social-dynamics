"""
Headless test for the knowledge manifold v3:
  - Noisy-up research (not lateral gradient)
  - Structural support constraint
  - Adaptive noise (stuck → more noise)
  - Reward-modulated social dynamics (successful teams attract)
  - Knowledge diffusion and decay

Renders diagnostic images at multiple timesteps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

from experiments.landscape import make_default_landscape
# Import from 3D_sim using importlib since it has a dash-like name
import importlib
kf_mod = importlib.import_module('3D_sim.knowledge_field')
mm_mod = importlib.import_module('3D_sim.mountain_mesh')
KnowledgeField = kf_mod.KnowledgeField
project_particles_to_knowledge_surface = mm_mod.project_particles_to_knowledge_surface
generate_surface_mesh = mm_mod.generate_surface_mesh

# ── Parameters ──────────────────────────────────────────────────────
N = 300
K = 3
STEPS = 3000
SNAPSHOT_STEPS = [0, 100, 500, 1000, 2000, 3000]
GRID_RES = 64
Z_SCALE = 0.5

# Research dynamics
NOISE_BASE = 3.0
NOISE_MAX = 8.0
NOISE_ADAPT = 0.05
WRITE_RATE = 0.015
NUDGE_RATE = 0.003
KNOW_CLIMB_RATE = 0.002  # climb toward known-good regions

# Knowledge field
DIFFUSION_SIGMA = 0.5
DECAY = 0.9999
SUPPORT_RADIUS = 3
MAX_SLOPE = 0.4

# Social dynamics
SOCIAL = 0.01
STEP_SIZE = 0.003
N_NEIGHBORS = 20
REWARD_EMA_ALPHA = 0.05
REWARD_SOCIAL_SCALE = 5.0

# Exploration
EXPLORE_PROB = 0.003
EXPLORE_RADIUS = 0.2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)


# ── Minimal simulation (social dynamics only) ──────────────────────
class MinimalSim:
    """Minimal 3D social dynamics sim for headless testing."""

    def __init__(self, n, k, rng):
        self.n = n
        self.k = k
        self.prefs = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.nbr_ids = None
        self._valid_mask = None
        self._reward_ema = None

    def step(self):
        """Social dynamics: find neighbours, social learning on prefs."""
        n = self.n
        # Map skill prefs to [0, 1) for toroidal tree
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

        # Social learning: blend toward neighbour mean
        nbr_prefs = self.prefs[self.nbr_ids]
        nbr_mean = nbr_prefs.mean(axis=1)
        self.prefs[:] = (1.0 - SOCIAL) * self.prefs + SOCIAL * nbr_mean
        np.clip(self.prefs, -1, 1, out=self.prefs)


def reward_social_nudge(sim, reward_ema):
    """Phase 1.5: reward-weighted social nudge."""
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
    """One step of knowledge manifold dynamics (noisy-up model)."""
    n = sim.n
    prefs = sim.prefs
    nbr_ids = sim.nbr_ids
    if nbr_ids is None:
        return

    # 1. Sense hidden gradient (lateral hint)
    probes = np.zeros((n, 3), dtype=np.float64)
    probes[:, 0] = prefs[:, 0]
    probes[:, 1] = prefs[:, 1]
    true_grad = landscape.gradient(probes)
    grad_mag = np.linalg.norm(true_grad, axis=1, keepdims=True)
    grad_mag = np.maximum(grad_mag, 1e-10)
    true_grad_unit = true_grad / grad_mag

    # Per-particle adaptive noise
    noise = rng.normal(0, 1, (n, 2)) * noise_level[:, None]
    noisy_direction = true_grad_unit + noise

    # 2. Team averaging
    nbr_dir = noisy_direction[nbr_ids]
    team_dir = nbr_dir.mean(axis=1)
    team_mag = np.linalg.norm(team_dir, axis=1, keepdims=True)
    team_mag = np.maximum(team_mag, 1e-10)
    team_dir_unit = team_dir / team_mag

    # 3. Knowledge gradient climbing — move toward known-good regions
    know_grad = kf.knowledge_gradient(prefs[:, 0], prefs[:, 1])
    know_mag = np.linalg.norm(know_grad, axis=1, keepdims=True)
    know_mag = np.maximum(know_mag, 1e-10)
    know_dir = know_grad / know_mag

    # 3b. Lateral drift = research nudge + knowledge climb
    lateral_nudge = NUDGE_RATE * team_dir_unit + KNOW_CLIMB_RATE * know_dir
    if EXPLORE_PROB > 0:
        explore_mask = rng.random(n) < EXPLORE_PROB
        n_explore = explore_mask.sum()
        if n_explore > 0:
            jump = rng.normal(0, EXPLORE_RADIUS, (n_explore, 2))
            lateral_nudge[explore_mask] += jump
    prefs[:, 0] += lateral_nudge[:, 0].astype(prefs.dtype)
    prefs[:, 1] += lateral_nudge[:, 1].astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # 4. Deposit knowledge ("push up")
    amounts = np.full(n, WRITE_RATE, dtype=np.float64)
    actual_growth = kf.deposit_knowledge(prefs[:, 0], prefs[:, 1], amounts)

    # 5. Reward tracking
    reward_ema[:] = (1.0 - REWARD_EMA_ALPHA) * reward_ema + REWARD_EMA_ALPHA * actual_growth
    sim._reward_ema = reward_ema

    # 6. Adaptive noise
    reward_threshold = WRITE_RATE * 0.3
    productive = reward_ema > reward_threshold
    stuck = ~productive
    noise_level[productive] += NOISE_ADAPT * (NOISE_BASE - noise_level[productive])
    noise_level[stuck] += NOISE_ADAPT * (NOISE_MAX - noise_level[stuck])
    np.clip(noise_level, NOISE_BASE, NOISE_MAX, out=noise_level)

    # 7. Diffuse knowledge field
    kf.step_field()

    # 8. Sync positions to knowledge surface
    projected = project_particles_to_knowledge_surface(kf, prefs, z_scale=Z_SCALE)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)


# ── Run simulation ──────────────────────────────────────────────────
print("Setting up...")
landscape = make_default_landscape(K)
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

# Storage for metrics
metrics = {
    'step': [], 'coverage': [], 'peak_knowledge': [],
    'pref_std': [], 'mean_reward': [], 'mean_noise': [],
    'mean_fitness_at_particles': [],
}

# Storage for snapshots
snapshots = {}

print("Running simulation...")
for step in range(STEPS + 1):
    if step in SNAPSHOT_STEPS:
        print(f"  Step {step}: coverage={kf.coverage():.3f}, "
              f"peak={kf.peak_knowledge():.3f}, "
              f"mean_reward={reward_ema.mean():.5f}, "
              f"mean_noise={noise_level.mean():.2f}")
        snapshots[step] = {
            'knowledge_grid': kf.grid.copy(),
            'prefs': sim.prefs.copy(),
            'pos': sim.pos.copy(),
            'reward_ema': reward_ema.copy(),
            'noise_level': noise_level.copy(),
        }

    if step % 50 == 0:
        metrics['step'].append(step)
        metrics['coverage'].append(kf.coverage())
        metrics['peak_knowledge'].append(kf.peak_knowledge())
        metrics['pref_std'].append(sim.prefs[:, :2].std())
        metrics['mean_reward'].append(reward_ema.mean())
        metrics['mean_noise'].append(noise_level.mean())
        # Fitness at particle positions
        probes = np.zeros((N, 3), dtype=np.float64)
        probes[:, 0] = sim.prefs[:, 0]
        probes[:, 1] = sim.prefs[:, 1]
        f = landscape.fitness(probes)
        metrics['mean_fitness_at_particles'].append(f.mean())

    if step < STEPS:
        # Phase 1: Social dynamics
        sim.step()
        # Phase 1.5: Reward-weighted social nudge
        if reward_ema is not None and step > 0:
            reward_social_nudge(sim, reward_ema)
        # Phase 2: Knowledge step
        knowledge_step(sim, kf, landscape, rng, reward_ema, noise_level)

print("Rendering...")

# ── Figure 1: Knowledge growth heatmaps ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, step in enumerate(SNAPSHOT_STEPS):
    ax = axes[idx // 3, idx % 3]
    snap = snapshots[step]
    im = ax.imshow(snap['knowledge_grid'].T, origin='lower',
                   extent=[-1, 1, -1, 1], vmin=0, vmax=1, cmap='YlOrRd')
    # Overlay particle positions
    ax.scatter(snap['prefs'][:, 0], snap['prefs'][:, 1],
               c=snap['reward_ema'], cmap='cool', s=3, alpha=0.5,
               vmin=0, vmax=max(0.001, snap['reward_ema'].max()))
    ax.set_title(f"Step {step}\ncov={kf.coverage() if step == STEPS else snapshots[step]['knowledge_grid'].sum() / kf._fitness_grid.sum():.2f}")
    ax.set_xlabel('pref0 (skill X)')
    ax.set_ylabel('pref1 (skill Y)')
plt.colorbar(im, ax=axes.ravel().tolist(), label='Knowledge height')
fig.suptitle('Knowledge Manifold Growth (v3: noisy-up + reward social)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'knowledge_growth_v3.png'), dpi=150)
plt.close()

# ── Figure 2: 3D views ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
for idx, step in enumerate(SNAPSHOT_STEPS):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    snap = snapshots[step]
    G = GRID_RES
    xs = np.linspace(0, 1, G)
    zs = np.linspace(0, 1, G)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')

    # Knowledge surface
    gy_know = snap['knowledge_grid'] * Z_SCALE
    ax.plot_surface(gx, gy_know, gz, alpha=0.5, color='green',
                    rstride=4, cstride=4, linewidth=0)

    # Fitness ghost
    gy_fit = kf._fitness_grid * Z_SCALE
    ax.plot_wireframe(gx, gy_fit, gz, alpha=0.15, color='blue',
                      rstride=8, cstride=8, linewidth=0.5)

    # Particles
    p = snap['pos']
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='red', s=2, alpha=0.6)

    ax.set_title(f"Step {step}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, Z_SCALE)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X (skill 0)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (skill 1)')
    ax.view_init(elev=25, azim=45)

fig.suptitle('3D Knowledge Surface Growth', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'knowledge_3d_v3.png'), dpi=150)
plt.close()

# ── Figure 3: Metrics over time ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, key, label in zip(axes.ravel(), [
    'coverage', 'peak_knowledge', 'pref_std',
    'mean_reward', 'mean_noise', 'mean_fitness_at_particles'
], [
    'Coverage', 'Peak Knowledge', 'Pref Std (skills)',
    'Mean Reward (EMA)', 'Mean Noise Level', 'Mean Fitness at Particles'
]):
    ax.plot(metrics['step'], metrics[key])
    ax.set_xlabel('Step')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
fig.suptitle('Knowledge Manifold Metrics (v3)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'knowledge_metrics_v3.png'), dpi=150)
plt.close()

# ── Figure 4: Reward distribution and noise distribution ────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
final_snap = snapshots[STEPS]
axes[0].hist(final_snap['reward_ema'], bins=50, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Reward EMA')
axes[0].set_ylabel('Count')
axes[0].set_title('Final Reward Distribution')
axes[1].hist(final_snap['noise_level'], bins=50, color='coral', alpha=0.7)
axes[1].set_xlabel('Noise Level')
axes[1].set_ylabel('Count')
axes[1].set_title('Final Noise Distribution')
fig.suptitle('Adaptive Dynamics (v3)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'knowledge_adaptive_v3.png'), dpi=150)
plt.close()

print(f"Done! Results saved to {RESULTS_DIR}/knowledge_*_v3.png")
print(f"Final: coverage={kf.coverage():.3f}, peak={kf.peak_knowledge():.3f}")
