"""
Knowledge Manifold v2 — improved parameters and knowledge gradient attraction.

Changes from v1:
- Reduced diffusion sigma (0.8 → 0.3) for sharper knowledge peaks
- Added knowledge gradient attraction (particles climb what they know)
- Added exploration perturbation (random jumps to escape local clusters)
- Tuned noise/nudge ratio for better signal
- Longer run (5000 steps) to see if summit is reached
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3D_sim'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

from experiments.landscape import make_default_landscape
from knowledge_field import KnowledgeField
from mountain_mesh import project_particles_to_knowledge_surface


class MinimalSim:
    def __init__(self, n=500, k=3, seed=None):
        rng = np.random.default_rng(seed)
        self.n = n
        self.k = k
        self.prefs = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.nbr_ids = None
        self.step_count = 0

    def step(self, n_neighbors=20, social=0.008):
        """Phase 1: social dynamics."""
        n = self.n
        # Ensure prefs are in bounds before mapping to [0,1]
        np.clip(self.prefs, -1, 1, out=self.prefs)
        query = np.column_stack([
            (self.prefs[:, 0] + 1) * 0.5,
            np.zeros(n),
            (self.prefs[:, 1] + 1) * 0.5,
        ])
        # cKDTree requires data strictly < boxsize
        np.clip(query, 0.0, 1.0 - 1e-10, out=query)
        tree = cKDTree(query, boxsize=1.0)
        k_nbr = min(n_neighbors + 1, n)
        _, nbr_ids = tree.query(query, k=k_nbr, workers=-1)
        self.nbr_ids = nbr_ids[:, 1:]

        # Social learning on ALL 3 prefs (including social pref dim 2)
        nbr_prefs = self.prefs[self.nbr_ids]
        nbr_mean = nbr_prefs.mean(axis=1)
        self.prefs[:] = (1.0 - social) * self.prefs + social * nbr_mean
        np.clip(self.prefs, -1, 1, out=self.prefs)
        self.step_count += 1


def knowledge_step_v2(sim, kf, landscape, rng,
                      gradient_noise=3.0,
                      write_rate=0.003,
                      nudge_rate=0.003,
                      knowledge_climb_rate=0.001,
                      explore_prob=0.003,
                      explore_radius=0.2,
                      z_scale=0.5):
    """Phase 2: knowledge-aware movement and deposition."""
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

    # 2. Aggregate noisy observations across team neighbours
    nbr_grad = noisy_grad[nbr_ids]
    team_grad = nbr_grad.mean(axis=1)

    # 3. Knowledge gradient — climb what you know
    know_grad = kf.knowledge_gradient(prefs[:, 0], prefs[:, 1])

    # 4. Combined movement: hidden gradient signal + knowledge climb
    combined = nudge_rate * team_grad + knowledge_climb_rate * know_grad

    # 5. Exploration: random jumps for some particles
    explore_mask = rng.random(n) < explore_prob
    n_explore = explore_mask.sum()
    if n_explore > 0:
        jump = rng.normal(0, explore_radius, (n_explore, 2))
        combined[explore_mask] += jump

    # 6. Apply movement to skill prefs only (dims 0, 1)
    prefs[:, 0] += combined[:, 0].astype(prefs.dtype)
    prefs[:, 1] += combined[:, 1].astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # 7. Deposit knowledge — proportional to team size effect
    # Larger teams deposit more reliably (less noise in their estimate)
    amounts = np.full(n, write_rate, dtype=np.float64)
    kf.deposit_knowledge(prefs[:, 0], prefs[:, 1], amounts)

    # 8. Diffuse and decay
    kf.step_field()

    # 9. Sync positions to knowledge surface
    projected = project_particles_to_knowledge_surface(kf, prefs, z_scale=z_scale)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)


def main():
    os.makedirs('results', exist_ok=True)

    landscape = make_default_landscape(seed=42)
    kf = KnowledgeField(grid_res=64, diffusion_sigma=0.3, decay=0.9995)
    kf.set_fitness_surface(landscape)

    sim = MinimalSim(n=500, k=3, seed=123)
    rng = np.random.default_rng(456)

    z_scale = 0.5
    n_steps = 5000
    snapshot_steps = [0, 200, 500, 1000, 2000, 5000]

    metrics = {
        'step': [], 'coverage': [], 'peak': [],
        'mean_knowledge': [], 'mean_fitness': [],
        'pref_std': [],
    }

    # Pre-compute fitness heatmap
    G = 64
    xs = np.linspace(-1, 1, G)
    ys = np.linspace(-1, 1, G)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    probes = np.zeros((G * G, 3), dtype=np.float64)
    probes[:, 0] = gx.ravel()
    probes[:, 1] = gy.ravel()
    fitness_map = landscape.fitness(probes).reshape(G, G)

    # Initial position sync
    projected = project_particles_to_knowledge_surface(kf, sim.prefs, z_scale=z_scale)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)

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

    print(f"Running {n_steps} steps with {sim.n} particles...")
    for step in range(1, n_steps + 1):
        sim.step(n_neighbors=20, social=0.008)
        knowledge_step_v2(sim, kf, landscape, rng,
                          gradient_noise=3.0,
                          write_rate=0.003,
                          nudge_rate=0.003,
                          knowledge_climb_rate=0.001,
                          explore_prob=0.003,
                          explore_radius=0.2,
                          z_scale=z_scale)

        if step % 100 == 0:
            cov = kf.coverage()
            peak = kf.peak_knowledge()
            mean_k = kf.grid.mean()
            mean_f = kf.sample_fitness(sim.prefs[:, 0], sim.prefs[:, 1]).mean()
            pref_std = sim.prefs[:, :2].std()
            metrics['step'].append(step)
            metrics['coverage'].append(cov)
            metrics['peak'].append(peak)
            metrics['mean_knowledge'].append(mean_k)
            metrics['mean_fitness'].append(mean_f)
            metrics['pref_std'].append(pref_std)

            if step % 500 == 0:
                print(f"  Step {step}: cov={cov:.1%} peak={peak:.3f} "
                      f"mean_k={mean_k:.4f} pref_std={pref_std:.3f}")

        if step in snapshot_steps:
            take_snapshot(step)

    # ── Render: Knowledge growth ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Knowledge Manifold Growth (v2 — with knowledge climb + exploration)', fontsize=14)

    for idx, step in enumerate(snapshot_steps):
        ax = axes[idx // 3, idx % 3]
        snap = snapshots[step]
        im = ax.imshow(snap['knowledge_grid'].T, origin='lower',
                       extent=[-1, 1, -1, 1], vmin=0, vmax=1,
                       cmap='YlGn', aspect='equal')
        ax.contour(gx, gy, fitness_map, levels=8, colors='blue',
                   alpha=0.3, linewidths=0.5)
        ax.scatter(snap['prefs'][:, 0], snap['prefs'][:, 1],
                   s=2, c='red', alpha=0.5)
        ax.set_title(f'Step {step}\ncov={snap["coverage"]:.1%} '
                     f'peak={snap["peak"]:.3f}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig('results/knowledge_growth_v2.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_growth_v2.png")

    # ── Render: 3D views ──
    fig = plt.figure(figsize=(18, 6))
    view_steps = [0, 1000, 5000]
    for idx, step in enumerate(view_steps):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        snap = snapshots[step]
        xs_grid = np.linspace(0, 1, G)
        zs_grid = np.linspace(0, 1, G)
        X, Z = np.meshgrid(xs_grid, zs_grid, indexing='ij')
        Y_know = snap['knowledge_grid'] * z_scale
        ax.plot_surface(X, Y_know, Z, alpha=0.4, color='green', edgecolor='none')
        Y_fit = kf._fitness_grid * z_scale
        ax.plot_wireframe(X, Y_fit, Z, alpha=0.15, color='blue', rstride=4, cstride=4)
        pos = snap['pos']
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=3, c='red', alpha=0.6)
        ax.set_title(f'Step {step}')
        ax.set_xlabel('Skill 0')
        ax.set_ylabel('Height')
        ax.set_zlabel('Skill 1')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, z_scale)
        ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.savefig('results/knowledge_3d_v2.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_3d_v2.png")

    # ── Render: Metrics ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Knowledge Manifold Metrics (v2)', fontsize=14)
    steps = metrics['step']

    axes[0, 0].plot(steps, metrics['coverage'], 'g-')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Coverage')

    axes[0, 1].plot(steps, metrics['peak'], 'b-')
    axes[0, 1].set_ylabel('Peak Knowledge')
    axes[0, 1].set_title('Peak Knowledge')

    axes[1, 0].plot(steps, metrics['mean_knowledge'], 'g--', label='Mean Knowledge')
    axes[1, 0].plot(steps, metrics['mean_fitness'], 'b-', label='Mean Fitness')
    axes[1, 0].legend()
    axes[1, 0].set_title('Mean Heights')

    axes[1, 1].plot(steps, metrics['pref_std'], 'r-')
    axes[1, 1].set_ylabel('Pref Std')
    axes[1, 1].set_title('Skill Diversity')

    for ax in axes.flat:
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/knowledge_metrics_v2.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_metrics_v2.png")

    # ── Render: Gap comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(fitness_map.T, origin='lower',
                         extent=[-1, 1, -1, 1], cmap='hot')
    axes[0].set_title('Hidden Fitness')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(kf.grid.T, origin='lower',
                         extent=[-1, 1, -1, 1], vmin=0, vmax=1, cmap='YlGn')
    axes[1].set_title(f'Knowledge (step {n_steps})')
    plt.colorbar(im1, ax=axes[1])

    gap = kf._fitness_grid - kf.grid
    im2 = axes[2].imshow(gap.T, origin='lower',
                         extent=[-1, 1, -1, 1], cmap='Reds')
    axes[2].set_title('Gap')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.scatter(sim.prefs[:, 0], sim.prefs[:, 1], s=1, c='cyan', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/knowledge_gap_v2.png', dpi=150)
    plt.close()
    print("Saved results/knowledge_gap_v2.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
