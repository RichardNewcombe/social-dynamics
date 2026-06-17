"""
Headless test of the Knowledge Manifold model.

Runs the simulation loop (social dynamics + knowledge step) without
the OpenGL visualiser, and renders matplotlib snapshots at several
timesteps showing:
  1. The hidden fitness surface (heatmap)
  2. The knowledge manifold growing over time (heatmap + contour)
  3. Particles on the knowledge surface (3D scatter)
  4. Metrics over time (coverage, peak knowledge, mean fitness)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from experiments.landscape import make_default_landscape

# We can't import from 3D_sim as a package easily, so import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3D_sim'))
from knowledge_field import KnowledgeField
from mountain_mesh import project_particles_to_knowledge_surface

# ── Minimal simulation stand-in ──────────────────────────────────────
# We replicate the essential loop without the full 3D sim's OpenGL deps.
# Phase 1: simple social learning on prefs (neighbour averaging)
# Phase 2: knowledge step (sense hidden gradient, aggregate, deposit)

from scipy.spatial import cKDTree


class MinimalSim:
    """Lightweight particle sim for headless testing."""

    def __init__(self, n=500, k=3, seed=None):
        rng = np.random.default_rng(seed)
        self.n = n
        self.k = k
        self.prefs = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.nbr_ids = None
        self._valid_mask = None
        self.step_count = 0

    def step(self, n_neighbors=20, social=0.01, step_size=0.003):
        """Phase 1: social dynamics via KNN + preference averaging."""
        n = self.n
        # Use prefs dims 0,1 mapped to [0,1] for spatial neighbour search
        # (matching the 3D visualiser's pos sync)
        query = np.column_stack([
            (self.prefs[:, 0] + 1) * 0.5,
            np.zeros(n),
            (self.prefs[:, 1] + 1) * 0.5,
        ])
        tree = cKDTree(query, boxsize=1.0)
        k_nbr = min(n_neighbors + 1, n)
        _, nbr_ids = tree.query(query, k=k_nbr, workers=-1)
        self.nbr_ids = nbr_ids[:, 1:]
        self._valid_mask = None

        # Social learning: blend prefs toward neighbour mean
        nbr_prefs = self.prefs[self.nbr_ids]
        nbr_mean = nbr_prefs.mean(axis=1)
        self.prefs[:] = (1.0 - social) * self.prefs + social * nbr_mean
        np.clip(self.prefs, -1, 1, out=self.prefs)

        self.step_count += 1


def knowledge_step(sim, knowledge_field, landscape, rng,
                   gradient_noise=4.0, write_rate=0.005,
                   nudge_rate=0.002, z_scale=0.5):
    """Phase 2: sense hidden gradient, aggregate, deposit, diffuse."""
    n = sim.n
    prefs = sim.prefs
    nbr_ids = sim.nbr_ids

    if nbr_ids is None:
        return

    # 1. Sense hidden gradient with noise
    probes = np.zeros((n, 3), dtype=np.float64)
    probes[:, 0] = prefs[:, 0]
    probes[:, 1] = prefs[:, 1]
    true_grad = landscape.gradient(probes)  # (N, 2)
    noise = rng.normal(0, gradient_noise, (n, 2))
    noisy_grad = true_grad + noise

    # 2. Aggregate across neighbours
    M = nbr_ids.shape[1]
    nbr_grad = noisy_grad[nbr_ids]
    team_grad = nbr_grad.mean(axis=1)

    # 3. Nudge skill prefs
    prefs[:, 0] += (nudge_rate * team_grad[:, 0]).astype(prefs.dtype)
    prefs[:, 1] += (nudge_rate * team_grad[:, 1]).astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # 4. Deposit knowledge
    amounts = np.full(n, write_rate, dtype=np.float64)
    knowledge_field.deposit_knowledge(prefs[:, 0], prefs[:, 1], amounts)

    # 5. Diffuse
    knowledge_field.step_field()

    # 6. Sync positions
    projected = project_particles_to_knowledge_surface(
        knowledge_field, prefs, z_scale=z_scale)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)


def main():
    os.makedirs('results', exist_ok=True)

    # ── Setup ──
    landscape = make_default_landscape(seed=42)
    kf = KnowledgeField(grid_res=64, diffusion_sigma=0.8, decay=0.999)
    kf.set_fitness_surface(landscape)

    sim = MinimalSim(n=500, k=3, seed=123)
    rng = np.random.default_rng(456)

    z_scale = 0.5
    n_steps = 3000
    snapshot_steps = [0, 100, 500, 1000, 2000, 3000]

    # Metrics tracking
    metrics = {
        'step': [], 'coverage': [], 'peak': [],
        'mean_knowledge': [], 'mean_fitness': [],
    }

    # ── Pre-compute hidden fitness heatmap ──
    G = 64
    xs = np.linspace(-1, 1, G)
    ys = np.linspace(-1, 1, G)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    probes = np.zeros((G * G, 3), dtype=np.float64)
    probes[:, 0] = gx.ravel()
    probes[:, 1] = gy.ravel()
    fitness_map = landscape.fitness(probes).reshape(G, G)

    # Sync initial positions
    projected = project_particles_to_knowledge_surface(
        kf, sim.prefs, z_scale=z_scale)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)

    # ── Snapshot storage ──
    snapshots = {}

    def take_snapshot(step):
        snapshots[step] = {
            'knowledge_grid': kf.grid.copy(),
            'prefs': sim.prefs.copy(),
            'pos': sim.pos.copy(),
            'coverage': kf.coverage(),
            'peak': kf.peak_knowledge(),
        }

    take_snapshot(0)

    # ── Run simulation ──
    print(f"Running {n_steps} steps with {sim.n} particles...")
    for step in range(1, n_steps + 1):
        sim.step(n_neighbors=20, social=0.01, step_size=0.003)
        knowledge_step(sim, kf, landscape, rng,
                       gradient_noise=4.0, write_rate=0.005,
                       nudge_rate=0.002, z_scale=z_scale)

        if step % 100 == 0 or step in snapshot_steps:
            cov = kf.coverage()
            peak = kf.peak_knowledge()
            mean_k = kf.grid.mean()
            mean_f = kf.sample_fitness(sim.prefs[:, 0], sim.prefs[:, 1]).mean()
            metrics['step'].append(step)
            metrics['coverage'].append(cov)
            metrics['peak'].append(peak)
            metrics['mean_knowledge'].append(mean_k)
            metrics['mean_fitness'].append(mean_f)

            if step % 500 == 0:
                print(f"  Step {step}: cov={cov:.1%} peak={peak:.3f} "
                      f"mean_k={mean_k:.4f}")

        if step in snapshot_steps:
            take_snapshot(step)

    # ── Render: Knowledge manifold growth ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Knowledge Manifold Growth Over Time', fontsize=16)

    for idx, step in enumerate(snapshot_steps):
        ax = axes[idx // 3, idx % 3]
        snap = snapshots[step]

        # Show knowledge grid as heatmap
        im = ax.imshow(snap['knowledge_grid'].T, origin='lower',
                       extent=[-1, 1, -1, 1], vmin=0, vmax=1,
                       cmap='YlGn', aspect='equal')
        # Overlay fitness contours
        ax.contour(gx, gy, fitness_map, levels=8, colors='blue',
                   alpha=0.3, linewidths=0.5)
        # Show particles
        ax.scatter(snap['prefs'][:, 0], snap['prefs'][:, 1],
                   s=2, c='red', alpha=0.5)
        ax.set_title(f'Step {step}\ncov={snap["coverage"]:.1%} '
                     f'peak={snap["peak"]:.3f}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig('results/knowledge_growth.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_growth.png")

    # ── Render: 3D views at key timesteps ──
    fig = plt.figure(figsize=(18, 6))
    for idx, step in enumerate([0, 1000, 3000]):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        snap = snapshots[step]

        # Knowledge surface
        xs_grid = np.linspace(0, 1, G)
        zs_grid = np.linspace(0, 1, G)
        X, Z = np.meshgrid(xs_grid, zs_grid, indexing='ij')
        Y_know = snap['knowledge_grid'] * z_scale
        ax.plot_surface(X, Y_know, Z, alpha=0.4, color='green',
                        edgecolor='none')

        # Ghost fitness surface
        Y_fit = kf._fitness_grid * z_scale
        ax.plot_wireframe(X, Y_fit, Z, alpha=0.15, color='blue',
                          rstride=4, cstride=4)

        # Particles
        pos = snap['pos']
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=3, c='red', alpha=0.6)

        ax.set_title(f'Step {step}')
        ax.set_xlabel('Skill 0')
        ax.set_ylabel('Height')
        ax.set_zlabel('Skill 1')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, z_scale)
        ax.set_zlim(0, 1)

    plt.tight_layout()
    plt.savefig('results/knowledge_3d_views.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_3d_views.png")

    # ── Render: Metrics over time ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Knowledge Manifold Metrics', fontsize=14)

    steps = metrics['step']
    axes[0, 0].plot(steps, metrics['coverage'], 'g-')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Coverage (knowledge / fitness volume)')

    axes[0, 1].plot(steps, metrics['peak'], 'b-')
    axes[0, 1].set_ylabel('Peak Knowledge')
    axes[0, 1].set_title('Peak Knowledge Height')

    axes[1, 0].plot(steps, metrics['mean_knowledge'], 'g--',
                    label='Mean Knowledge')
    axes[1, 0].plot(steps, metrics['mean_fitness'], 'b-',
                    label='Mean Fitness (at particle pos)')
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('Height')
    axes[1, 0].set_title('Mean Heights')

    # Pref spread over time
    pref_stds = []
    for step in snapshot_steps:
        snap = snapshots[step]
        pref_stds.append(snap['prefs'][:, :2].std())
    axes[1, 1].plot(snapshot_steps, pref_stds, 'r-o')
    axes[1, 1].set_ylabel('Pref Std (skills)')
    axes[1, 1].set_title('Skill Preference Diversity')

    for ax in axes.flat:
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/knowledge_metrics.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_metrics.png")

    # ── Render: Hidden fitness vs knowledge at final step ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hidden fitness
    im0 = axes[0].imshow(fitness_map.T, origin='lower',
                         extent=[-1, 1, -1, 1], cmap='hot')
    axes[0].set_title('Hidden Fitness Landscape')
    plt.colorbar(im0, ax=axes[0])

    # Final knowledge
    im1 = axes[1].imshow(kf.grid.T, origin='lower',
                         extent=[-1, 1, -1, 1], vmin=0, vmax=1,
                         cmap='YlGn')
    axes[1].set_title(f'Knowledge Manifold (step {n_steps})')
    plt.colorbar(im1, ax=axes[1])

    # Gap (fitness - knowledge)
    gap = kf._fitness_grid - kf.grid
    im2 = axes[2].imshow(gap.T, origin='lower',
                         extent=[-1, 1, -1, 1], cmap='Reds')
    axes[2].set_title('Gap (Fitness - Knowledge)')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.scatter(sim.prefs[:, 0], sim.prefs[:, 1],
                   s=1, c='cyan', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/knowledge_vs_fitness.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_vs_fitness.png")

    print("\nDone! All diagnostic images saved to results/")


if __name__ == '__main__':
    main()
