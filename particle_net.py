#!/usr/bin/env python3
"""
Differentiable Particle Network (ParticleNet)
=============================================

Uses the inner-product-average particle physics from sim_gpu_compute.py as a
differentiable computational substrate for learning functions y = f(x).

Architecture:
    x (input)  →  Encoder MLP  →  initial pos (N×2) + prefs (N×K)
                                        │
                                        ▼
                              Differentiable Particle Sim (T steps)
                                  KNN on detached pos (no grad)
                                  Physics on attached pos/prefs (grad flows)
                                        │
                                        ▼
                              Decoder MLP  →  y_pred (output)

The straight-through estimator makes the non-differentiable KNN step
compatible with backpropagation: neighbor indices are found with no_grad,
but subsequent gather operations on the attached pos/prefs tensors let
gradients flow through the physics.

Usage:
    python particle_net.py             # train on sin(x)
    python particle_net.py --task 2d   # train on sin(πx₁)·cos(πx₂)
"""

import argparse
import math
import torch
import torch.nn as nn
import numpy as np

SPACE = 1.0


def _periodic_dist(a, b, L=SPACE):
    """Periodic displacement b - a, wrapped to [-L/2, L/2)."""
    d = b - a
    d = d - L * torch.round(d / L)
    return d


class ParticleNet(nn.Module):
    """
    Differentiable particle network.

    An encoder maps input x to initial particle positions and preferences,
    a differentiable particle simulation runs T steps of inner-product-average
    physics, and a decoder reads out the final state to predict y.

    Args:
        input_dim:      size of input x vectors
        output_dim:     size of output y vectors
        n_particles:    N, number of particles (default 64)
        k_dims:         K, preference dimensions per particle (default 8)
        n_neighbors:    KNN neighbors per particle (default 8)
        n_steps:        T, simulation steps (default 10)
        step_size:      physics step size (default 0.01)
        repulsion:      repulsion strength (default 0.001)
        social:         social learning rate during forward pass (default 0.0 = off)
        encoder_hidden: encoder MLP hidden dim (default 128)
        decoder_hidden: decoder MLP hidden dim (default 128)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_particles: int = 64,
        k_dims: int = 8,
        n_neighbors: int = 8,
        n_steps: int = 10,
        step_size: float = 0.01,
        repulsion: float = 0.001,
        social: float = 0.0,
        encoder_hidden: int = 128,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.k_dims = k_dims
        self.n_neighbors = n_neighbors
        self.n_steps = n_steps
        self.step_size = step_size
        self.repulsion = repulsion
        self.social = social

        state_dim = n_particles * (2 + k_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, output_dim),
        )

    def _find_neighbors(self, pos):
        """
        Straight-through KNN: find neighbor indices with no_grad.

        Computes all-pairs periodic distances and returns the indices of the
        closest n_neighbors particles for each particle. The returned indices
        are detached integers — gradients flow later when we use them to
        gather from the attached pos/prefs tensors.

        Args:
            pos: (B, N, 2) particle positions

        Returns:
            nbr_ids: (B, N, K_nbr) integer neighbor indices (detached)
        """
        n_nbr = min(self.n_neighbors, self.n_particles - 1)
        with torch.no_grad():
            # (B, N, 1, 2) - (B, 1, N, 2) → (B, N, N, 2)
            diff = _periodic_dist(pos.unsqueeze(2), pos.unsqueeze(1))
            dists = diff.norm(dim=-1)  # (B, N, N)
            # Exclude self by setting diagonal to inf
            dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
            _, nbr_ids = dists.topk(n_nbr, dim=-1, largest=False)  # (B, N, K_nbr)
        return nbr_ids

    def _physics_step(self, pos, prefs, nbr_ids):
        """
        One step of inner-product-average physics.

        Matches the inner_avg mode from sim_gpu_compute.py:
        - Compute periodic displacement to each neighbor
        - Inner product of preference vectors → attraction/repulsion weight
        - Average weighted unit vectors → movement
        - 1/dist² repulsion push
        - Update positions with modular arithmetic
        - Optional social learning of preferences

        Args:
            pos:     (B, N, 2) particle positions
            prefs:   (B, N, K) preference vectors
            nbr_ids: (B, N, K_nbr) neighbor indices

        Returns:
            new_pos:   (B, N, 2) updated positions
            new_prefs: (B, N, K) updated preferences
        """
        B, N, K = prefs.shape
        K_nbr = nbr_ids.shape[2]

        # Gather neighbor positions and preferences (gradients flow through here)
        batch_idx = torch.arange(B, device=pos.device)[:, None, None].expand_as(nbr_ids)
        nbr_pos = pos[batch_idx, nbr_ids]      # (B, N, K_nbr, 2)
        nbr_prefs = prefs[batch_idx, nbr_ids]  # (B, N, K_nbr, K)

        # Periodic displacement from particle i to its neighbors
        toward = _periodic_dist(
            pos.unsqueeze(2),   # (B, N, 1, 2)
            nbr_pos,            # (B, N, K_nbr, 2)
        )  # (B, N, K_nbr, 2)

        dists = toward.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, N, K_nbr, 1)
        unit = toward / dists  # (B, N, K_nbr, 2)

        # Inner product of preferences: ip_i,j = (prefs_i · prefs_j) / K
        # prefs_i: (B, N, 1, K),  nbr_prefs: (B, N, K_nbr, K)
        ip = (prefs.unsqueeze(2) * nbr_prefs).sum(dim=-1, keepdim=True) / K  # (B, N, K_nbr, 1)

        # Attraction/repulsion weighted by inner product
        weighted = ip * unit  # (B, N, K_nbr, 2)
        movement = weighted.mean(dim=2)  # (B, N, 2)

        # Repulsion: -unit / dist, averaged over neighbors
        push = (-unit / dists.clamp(min=1e-6)).mean(dim=2)  # (B, N, 2)

        # Update positions
        new_pos = (pos + self.step_size * movement + self.repulsion * push) % SPACE

        # Social learning (optional)
        if self.social > 0.0:
            nbr_mean = nbr_prefs.mean(dim=2)  # (B, N, K)
            new_prefs = (1.0 - self.social) * prefs + self.social * nbr_mean
            new_prefs = new_prefs.clamp(-1.0, 1.0)
        else:
            new_prefs = prefs

        return new_pos, new_prefs

    def _encode_state(self, x):
        """Encode input x into initial particle positions and preferences.

        Returns:
            pos:   (B, N, 2)  in [0, SPACE)
            prefs: (B, N, K)  in [-1, 1]
        """
        B = x.shape[0]
        N = self.n_particles
        K = self.k_dims

        state = self.encoder(x)  # (B, N*(2+K))
        pos_raw = state[:, :N * 2].reshape(B, N, 2)
        pref_raw = state[:, N * 2:].reshape(B, N, K)

        pos = torch.sigmoid(pos_raw) * SPACE
        prefs = torch.tanh(pref_raw)
        return pos, prefs

    def forward(self, x):
        """
        Forward pass: encode → simulate → decode.

        Args:
            x: (B, input_dim) input tensor

        Returns:
            y_pred: (B, output_dim) predicted output
        """
        B = x.shape[0]
        pos, prefs = self._encode_state(x)

        # Run T steps of differentiable particle simulation
        for _ in range(self.n_steps):
            nbr_ids = self._find_neighbors(pos)
            pos, prefs = self._physics_step(pos, prefs, nbr_ids)

        # Decode final state to output
        out = torch.cat([pos.reshape(B, -1), prefs.reshape(B, -1)], dim=1)  # (B, N*(2+K))
        return self.decoder(out)

    @torch.no_grad()
    def run_with_trajectory(self, x):
        """Run forward pass and record particle state at every step.

        Uses the same physics path as forward(). Intended for visualization.

        Args:
            x: (B, input_dim) input tensor (typically B=1 for visualization)

        Returns:
            pos_history:   list of T+1 arrays, each (B, N, 2) numpy
            prefs_history: list of T+1 arrays, each (B, N, K) numpy
            nbr_history:   list of T arrays, each (B, N, K_nbr) numpy
            y_pred:        (B, output_dim) numpy
        """
        pos, prefs = self._encode_state(x)

        pos_history = [pos.cpu().numpy().copy()]
        prefs_history = [prefs.cpu().numpy().copy()]
        nbr_history = []

        for _ in range(self.n_steps):
            nbr_ids = self._find_neighbors(pos)
            nbr_history.append(nbr_ids.cpu().numpy().copy())
            pos, prefs = self._physics_step(pos, prefs, nbr_ids)
            pos_history.append(pos.cpu().numpy().copy())
            prefs_history.append(prefs.cpu().numpy().copy())

        out = torch.cat([pos.reshape(x.shape[0], -1), prefs.reshape(x.shape[0], -1)], dim=1)
        y_pred = self.decoder(out).cpu().numpy()

        return pos_history, prefs_history, nbr_history, y_pred


# =====================================================================
# Training demo
# =====================================================================

def make_sin_data(n_samples, device, dtype):
    """Generate y = sin(x) data with x uniform in [-π, π]."""
    x = torch.rand(n_samples, 1, device=device, dtype=dtype) * 2 * math.pi - math.pi
    y = torch.sin(x)
    return x, y


def make_2d_data(n_samples, device, dtype):
    """Generate y = sin(πx₁)·cos(πx₂) with x in [-1, 1]²."""
    x = torch.rand(n_samples, 2, device=device, dtype=dtype) * 2 - 1
    y = (torch.sin(math.pi * x[:, 0:1]) * torch.cos(math.pi * x[:, 1:2]))
    return x, y


def _resolve_device_dtype(device_arg):
    """Pick device and the best dtype it supports (f64 if possible, else f32)."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_arg

    # f64 works on CPU and CUDA, but not on MPS
    if device == 'mps':
        dtype = torch.float32
    else:
        dtype = torch.float64

    return device, dtype


def _prefs_to_rgb(prefs):
    """Map preference vectors to RGB colors, matching sim_gpu_compute.py.

    Uses first 3 preference dims mapped from [-1,1] to [0,1].
    If K < 3, missing channels default to 0.5 (neutral gray).

    Args:
        prefs: (N, K) numpy array

    Returns:
        colors: (N, 3) numpy array in [0, 1]
    """
    N, K = prefs.shape
    rgb = np.full((N, 3), 0.5)
    k3 = min(K, 3)
    rgb[:, :k3] = np.clip((prefs[:, :k3] + 1.0) * 0.5, 0, 1)
    return rgb


def _periodic_wrap(trail, L=SPACE):
    """Detect periodic boundary jumps in a trail and insert NaNs for clean plotting."""
    # If any step jumps more than L/2, the particle wrapped — break the line
    diff = np.diff(trail, axis=0)
    jump = np.any(np.abs(diff) > L / 2, axis=1)
    if not np.any(jump):
        return trail
    # Insert NaN rows at wrap points to break the line
    result = []
    result.append(trail[0:1])
    for i in range(len(diff)):
        if jump[i]:
            result.append(np.full((1, 2), np.nan))
        result.append(trail[i + 1:i + 2])
    return np.concatenate(result, axis=0)


def visualize_particles(model, x_samples, device, dtype, out_path='particle_net_dynamics.png'):
    """Visualize particle dynamics for a few sample inputs.

    For each sample, shows:
    - Left panel: initial particle positions (from encoder)
    - Middle panel: final particle positions (after T steps) with trajectory trails
    - Right panel: final positions with neighbor connections

    Particles are colored by their preference vectors (first 3 dims → RGB),
    matching the coloring convention from sim_gpu_compute.py.

    Args:
        model:     trained ParticleNet
        x_samples: (S,) or (S, input_dim) tensor of sample inputs to visualize
        device:    torch device
        dtype:     torch dtype
        out_path:  output file path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    if x_samples.dim() == 1:
        x_samples = x_samples.unsqueeze(1)
    n_samples = min(len(x_samples), 4)  # show at most 4 rows
    x_samples = x_samples[:n_samples].to(device=device, dtype=dtype)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4.5 * n_samples),
                             squeeze=False)

    L = SPACE

    for row in range(n_samples):
        x_in = x_samples[row:row + 1]  # (1, input_dim)
        pos_hist, pref_hist, nbr_hist, y_pred = model.run_with_trajectory(x_in)

        # Extract batch item 0
        T = len(pos_hist) - 1
        pos_init = pos_hist[0][0]      # (N, 2)
        prefs_init = pref_hist[0][0]   # (N, K)
        pos_final = pos_hist[-1][0]    # (N, 2)
        prefs_final = pref_hist[-1][0] # (N, K)
        nbr_final = nbr_hist[-1][0]    # (N, K_nbr)

        colors_init = _prefs_to_rgb(prefs_init)
        colors_final = _prefs_to_rgb(prefs_final)

        x_val = x_in.cpu().numpy().ravel()
        x_str = ', '.join(f'{v:.2f}' for v in x_val)
        y_str = ', '.join(f'{v:.3f}' for v in y_pred[0])

        # --- Panel 1: initial positions ---
        ax = axes[row, 0]
        ax.scatter(pos_init[:, 0], pos_init[:, 1], c=colors_init, s=25,
                   edgecolors='k', linewidths=0.3, zorder=3)
        ax.set_xlim(-0.02, L + 0.02)
        ax.set_ylim(-0.02, L + 0.02)
        ax.set_aspect('equal')
        ax.set_title(f'Initial  (x=[{x_str}])', fontsize=10)
        ax.set_xlabel('pos_x')
        ax.set_ylabel('pos_y')

        # --- Panel 2: trails + final positions ---
        ax = axes[row, 1]
        # Draw trajectory trails for each particle
        N = pos_init.shape[0]
        for p in range(N):
            trail = np.array([pos_hist[t][0, p] for t in range(T + 1)])  # (T+1, 2)
            trail_wrapped = _periodic_wrap(trail, L)
            ax.plot(trail_wrapped[:, 0], trail_wrapped[:, 1],
                    color=colors_final[p], alpha=0.35, linewidth=0.8, zorder=1)
        # Final positions on top
        ax.scatter(pos_final[:, 0], pos_final[:, 1], c=colors_final, s=25,
                   edgecolors='k', linewidths=0.3, zorder=3)
        ax.set_xlim(-0.02, L + 0.02)
        ax.set_ylim(-0.02, L + 0.02)
        ax.set_aspect('equal')
        ax.set_title(f'Final + Trails  (T={T}, y=[{y_str}])', fontsize=10)
        ax.set_xlabel('pos_x')

        # --- Panel 3: final positions with neighbor connections ---
        ax = axes[row, 2]
        # Draw neighbor edges
        for i in range(N):
            for j_idx in range(nbr_final.shape[1]):
                j = nbr_final[i, j_idx]
                # Compute periodic-aware line segment
                d = pos_final[j] - pos_final[i]
                d = d - L * np.round(d / L)
                end = pos_final[i] + d
                ax.plot([pos_final[i, 0], end[0]], [pos_final[i, 1], end[1]],
                        color='gray', alpha=0.15, linewidth=0.5, zorder=1)
        ax.scatter(pos_final[:, 0], pos_final[:, 1], c=colors_final, s=25,
                   edgecolors='k', linewidths=0.3, zorder=3)
        ax.set_xlim(-0.02, L + 0.02)
        ax.set_ylim(-0.02, L + 0.02)
        ax.set_aspect('equal')
        ax.set_title(f'Final + Neighbors (K_nbr={nbr_final.shape[1]})', fontsize=10)
        ax.set_xlabel('pos_x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Particle dynamics plot saved to {out_path}")


def train(args):
    device, dtype = _resolve_device_dtype(args.device)

    dtype_name = 'float64' if dtype == torch.float64 else 'float32'
    print(f"Device: {device}  Dtype: {dtype_name}")

    # Select task
    if args.task == '2d':
        input_dim, output_dim = 2, 1
        make_data = make_2d_data
        task_name = 'sin(πx₁)·cos(πx₂)'
    else:
        input_dim, output_dim = 1, 1
        make_data = make_sin_data
        task_name = 'sin(x)'

    print(f"Task: {task_name}")
    print(f"Particles: N={args.n_particles}, K={args.k_dims}, "
          f"T={args.n_steps}, neighbors={args.n_neighbors}")

    # Data
    x_train, y_train = make_data(1000, device, dtype)
    x_test, y_test = make_data(200, device, dtype)

    # Model
    model = ParticleNet(
        input_dim=input_dim,
        output_dim=output_dim,
        n_particles=args.n_particles,
        k_dims=args.k_dims,
        n_neighbors=args.n_neighbors,
        n_steps=args.n_steps,
        step_size=args.step_size,
        repulsion=args.repulsion,
        social=args.social,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    train_losses = []
    test_losses = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_losses.append(train_loss)

        # Test
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = loss_fn(y_pred_test, y_test).item()
        test_losses.append(test_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}")

        # Verify gradients flow on first epoch
        if epoch == 1:
            has_grad = all(p.grad is not None for p in model.parameters())
            print(f"Gradients flowing: {has_grad}")
            if not has_grad:
                no_grad_params = [n for n, p in model.named_parameters() if p.grad is None]
                print(f"  Parameters with no grad: {no_grad_params}")

    # Plot results
    print("\nGenerating plot...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: true vs predicted
        model.eval()
        with torch.no_grad():
            if args.task == '2d':
                # 2D: plot a slice at x₂=0
                x_plot = torch.linspace(-1, 1, 200, device=device, dtype=dtype)
                x_plot_2d = torch.stack([x_plot, torch.zeros_like(x_plot)], dim=1)
                y_true = torch.sin(math.pi * x_plot).unsqueeze(1)  # cos(0)=1
                y_hat = model(x_plot_2d)
                axes[0].plot(x_plot.cpu().numpy(), y_true.cpu().numpy(), 'b-', label='true', linewidth=2)
                axes[0].plot(x_plot.cpu().numpy(), y_hat.cpu().numpy(), 'r--', label='predicted', linewidth=2)
                axes[0].set_xlabel('x₁ (x₂=0)')
                axes[0].set_title(f'Slice: sin(πx₁)·cos(0) = sin(πx₁)')
            else:
                x_plot = torch.linspace(-math.pi, math.pi, 200, device=device, dtype=dtype).unsqueeze(1)
                y_true = torch.sin(x_plot)
                y_hat = model(x_plot)
                axes[0].plot(x_plot.cpu().numpy(), y_true.cpu().numpy(), 'b-', label='true', linewidth=2)
                axes[0].plot(x_plot.cpu().numpy(), y_hat.cpu().numpy(), 'r--', label='predicted', linewidth=2)
                axes[0].set_xlabel('x')
                axes[0].set_title(f'True vs Predicted: {task_name}')

        axes[0].set_ylabel('y')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: loss curves
        axes[1].semilogy(train_losses, label='train', alpha=0.7)
        axes[1].semilogy(test_losses, label='test', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Training Progress')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = 'particle_net_result.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot saved to {out_path}")

        # Particle dynamics visualization
        # Pick a few representative inputs across the input range
        if args.task == '2d':
            viz_x = torch.tensor([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [0.8, -0.3]],
                                 device=device, dtype=dtype)
        else:
            viz_x = torch.tensor([[-2.5], [-1.0], [0.0], [1.5]],
                                 device=device, dtype=dtype)
        visualize_particles(model, viz_x, device, dtype)
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Differentiable Particle Network')
    parser.add_argument('--task', default='sin', choices=['sin', '2d'],
                        help='Task: sin (1D→1D) or 2d (2D→1D)')
    parser.add_argument('--n-particles', type=int, default=64)
    parser.add_argument('--k-dims', type=int, default=8)
    parser.add_argument('--n-neighbors', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=10)
    parser.add_argument('--step-size', type=float, default=0.01)
    parser.add_argument('--repulsion', type=float, default=0.001)
    parser.add_argument('--social', type=float, default=0.0)
    parser.add_argument('--encoder-hidden', type=int, default=128)
    parser.add_argument('--decoder-hidden', type=int, default=128)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device: auto picks cuda > mps > cpu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    train(args)
