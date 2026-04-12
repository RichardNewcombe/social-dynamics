"""
ExperimentController — live organizational observatory for the visualizer.

Each experiment is a dashboard of competing organizational health metrics.
There are no "win conditions" — the user adjusts simulation parameters and
watches the tradeoffs shift in real time.  The actionable learning is
understanding the Pareto frontier: what you gain and what you pay.

Design principles:
  - No gradient nudges, no external forces.  Pure simulation dynamics.
  - Optional init hook to set up a clean starting state (user opts in).
  - Metrics update every step.  Experiments run indefinitely.
  - Exp 3 has a "Trigger Shock" button for testing adaptability.
"""

import numpy as np

try:
    from imgui_bundle import imgui
    _HAS_IMGUI = True
except ImportError:
    _HAS_IMGUI = False

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from experiments.runner import apply_post_processing
from sim_2d_exp.params import params, SPACE


# ═══════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════

def _cluster_metrics(prefs, eps=0.35, min_samples=5):
    """Compute clustering metrics in preference space using DBSCAN.

    Returns a dict with:
      cluster_count: number of distinct clusters
      alignment:     mean intra-cluster std (lower = tighter teams)
      diversity:     mean pairwise inter-cluster centroid distance
      entropy:       Shannon entropy of cluster size distribution
      lone_wolves:   fraction of particles not in any cluster
    """
    p = prefs.astype(np.float64)
    n = len(p)

    if _HAS_SKLEARN:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(p)
        labels = db.labels_
    else:
        # Fallback: simple grid-based clustering
        # Quantize each dim to 4 bins and group by bin tuple
        bins = np.clip(((p + 1) / 2 * 4).astype(int), 0, 3)
        label_map = {}
        labels = np.full(n, -1, dtype=int)
        next_label = 0
        for i in range(n):
            key = tuple(bins[i])
            if key not in label_map:
                label_map[key] = next_label
                next_label += 1
            labels[i] = label_map[key]
        # Mark small clusters as noise
        for lbl in range(next_label):
            if (labels == lbl).sum() < min_samples:
                labels[labels == lbl] = -1

    unique_labels = set(labels)
    unique_labels.discard(-1)
    cluster_count = len(unique_labels)
    lone_wolves = float((labels == -1).sum()) / n

    if cluster_count == 0:
        return {
            'cluster_count': 0,
            'alignment': 0.0,
            'diversity': 0.0,
            'entropy': 0.0,
            'lone_wolves': lone_wolves,
            'largest_cluster_frac': 0.0,
        }

    # Intra-cluster alignment (mean std per cluster)
    centroids = []
    sizes = []
    intra_stds = []
    for lbl in sorted(unique_labels):
        mask = labels == lbl
        cluster_prefs = p[mask]
        centroids.append(cluster_prefs.mean(axis=0))
        sizes.append(mask.sum())
        intra_stds.append(cluster_prefs.std(axis=0).mean())
    centroids = np.array(centroids)
    sizes = np.array(sizes, dtype=float)
    alignment = float(np.mean(intra_stds))

    # Inter-cluster diversity (mean pairwise centroid distance)
    if cluster_count > 1:
        dists = []
        for i in range(cluster_count):
            for j in range(i + 1, cluster_count):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    # Shannon entropy of cluster sizes (normalized)
    probs = sizes / sizes.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = np.log(cluster_count + 1e-12)
    if max_entropy > 0:
        entropy /= max_entropy  # normalize to [0, 1]

    largest_cluster_frac = float(sizes.max() / n)

    return {
        'cluster_count': cluster_count,
        'alignment': alignment,
        'diversity': diversity,
        'entropy': entropy,
        'lone_wolves': lone_wolves,
        'largest_cluster_frac': largest_cluster_frac,
    }


def _pref_coverage(prefs):
    """Compute preference space coverage metrics.

    Returns:
      spread:     mean per-dimension std (overall diversity)
      dim_corr:   mean absolute correlation between pref dimensions
      range_frac: fraction of [-1,1]^K volume "covered" (approx)
    """
    p = prefs.astype(np.float64)
    spread = float(p.std(axis=0).mean())

    # Dimension correlation (groupthink indicator)
    if p.shape[1] >= 2:
        corr_matrix = np.corrcoef(p.T)
        # Mean absolute off-diagonal correlation
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        dim_corr = float(np.abs(corr_matrix[mask]).mean())
    else:
        dim_corr = 0.0

    # Coverage: fraction of grid cells occupied
    n_bins = 5
    bins = np.clip(((p + 1) / 2 * n_bins).astype(int), 0, n_bins - 1)
    occupied = len(set(map(tuple, bins)))
    total_cells = n_bins ** min(p.shape[1], 3)
    range_frac = float(occupied) / total_cells

    return {
        'spread': spread,
        'dim_corr': dim_corr,
        'range_frac': range_frac,
    }


def _exploration_rate(prefs, prev_prefs):
    """Mean per-particle preference change magnitude."""
    if prev_prefs is None:
        return 0.0
    diff = prefs.astype(np.float64) - prev_prefs.astype(np.float64)
    return float(np.linalg.norm(diff, axis=1).mean())


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1: Team Formation & Alignment
# ═══════════════════════════════════════════════════════════════════════

def _exp1_factory(ctrl):
    """Team Formation & Alignment.

    Org question: "How does our communication culture affect the speed
    and quality of team formation?"

    This observatory tracks how the simulation's social dynamics create
    (or destroy) team structure.  There is no right answer — the metrics
    reveal competing tradeoffs:

      - More alignment (tight teams) costs diversity (fewer distinct ideas)
      - Faster stability costs exploration (locked in early)
      - Balanced orgs (high entropy) are harder to achieve than dominant ones

    Key parameter: Social learning rate (Social slider).
    Also explore: n_neighbors, repulsion, social_mode.
    """
    _prev_prefs = [None]  # mutable container for closure
    _prev_cluster_count = [None]
    _stable_since = [0]

    def init(sim):
        """Scatter particles uniformly — maximum diversity starting point."""
        _prev_prefs[0] = None
        _prev_cluster_count[0] = None
        _stable_since[0] = 0
        sim.prefs = sim.rng.uniform(-1, 1, (sim.n, sim.k)).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)

    def metrics(sim, step):
        prefs = sim.prefs

        # Clustering
        cm = _cluster_metrics(prefs)

        # Stability tracking
        if _prev_cluster_count[0] == cm['cluster_count']:
            _stable_since[0] += 1
        else:
            _stable_since[0] = 0
        _prev_cluster_count[0] = cm['cluster_count']

        # Exploration rate
        exp_rate = _exploration_rate(prefs, _prev_prefs[0])
        _prev_prefs[0] = prefs.copy()

        # Coverage
        cov = _pref_coverage(prefs)

        return {
            'Teams (clusters)': str(cm['cluster_count']),
            'Team Coherence (lower=tighter)': f"{cm['alignment']:.3f}",
            'Cognitive Diversity': f"{cm['diversity']:.3f}",
            'Org Balance (entropy)': f"{cm['entropy']:.2f}",
            'Lone Wolves': f"{cm['lone_wolves']:.1%}",
            'Largest Team': f"{cm['largest_cluster_frac']:.1%}",
            'Stable For': f"{_stable_since[0]} steps",
            'Exploration Rate': f"{exp_rate:.4f}",
            'Idea Spread (std)': f"{cov['spread']:.3f}",
            'Groupthink (dim corr)': f"{cov['dim_corr']:.3f}",
            'Pref Space Coverage': f"{cov['range_frac']:.1%}",
        }

    return {
        'init': init,
        'post_step': None,
        'check': None,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 2: Innovation vs Efficiency
# ═══════════════════════════════════════════════════════════════════════

def _exp2_factory(ctrl):
    """Innovation vs Efficiency.

    Org question: "How do we maintain the ability to innovate while
    keeping operational efficiency?"

    This observatory measures the tension between tight, efficient teams
    and broad, innovative coverage of the idea space.  The key insight
    is that these two goals are fundamentally in conflict.

    Key parameter: Social mode (Uniform vs Quiet-Dim Differentiation).
    Also explore: Social rate, n_neighbors, repulsion.
    """
    _prev_prefs = [None]

    def init(sim):
        """Scatter particles uniformly — maximum innovation potential."""
        _prev_prefs[0] = None
        sim.prefs = sim.rng.uniform(-1, 1, (sim.n, sim.k)).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)

    def metrics(sim, step):
        prefs = sim.prefs

        # Clustering (efficiency = tight clusters)
        cm = _cluster_metrics(prefs)

        # Coverage (innovation = broad spread)
        cov = _pref_coverage(prefs)

        # Exploration rate
        exp_rate = _exploration_rate(prefs, _prev_prefs[0])
        _prev_prefs[0] = prefs.copy()

        # Per-dimension analysis (quiet-dim effect)
        p = prefs.astype(np.float64)
        dim_stds = p.std(axis=0)
        dim_stds_str = ', '.join(f'{s:.3f}' for s in dim_stds[:min(3, len(dim_stds))])

        # Specialization index: ratio of max to min dim std
        # High = some dims are tight while others are spread (specialization)
        # Low = all dims equally spread or equally tight (no specialization)
        if dim_stds.min() > 1e-6:
            specialization = float(dim_stds.max() / dim_stds.min())
        else:
            specialization = float('inf') if dim_stds.max() > 1e-6 else 1.0

        return {
            'Efficiency (team tightness)': f"{cm['alignment']:.3f}",
            'Innovation (idea spread)': f"{cov['spread']:.3f}",
            'Pref Space Coverage': f"{cov['range_frac']:.1%}",
            'Teams': str(cm['cluster_count']),
            'Groupthink (dim corr)': f"{cov['dim_corr']:.3f}",
            'Specialization Index': f"{specialization:.2f}",
            'Per-Dim Spread': dim_stds_str,
            'Exploration Rate': f"{exp_rate:.4f}",
            'Lone Wolves': f"{cm['lone_wolves']:.1%}",
        }

    return {
        'init': init,
        'post_step': None,
        'check': None,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 3: Organizational Memory & Adaptability
# ═══════════════════════════════════════════════════════════════════════

def _exp3_factory(ctrl):
    """Organizational Memory & Adaptability.

    Org question: "How much institutional memory helps vs hurts when
    the environment changes?"

    This observatory tracks the interplay between the spatial memory
    field and current particle preferences.  A "Trigger Shock" button
    flips all preferences, simulating a market disruption or strategic
    pivot.  The metrics then show how quickly (or slowly) the memory
    field adapts to the new reality.

    REQUIRES: Memory Field = ON, Field Strength > 0.
    Key parameters: memory_strength, memory_decay, memory_write_rate.
    """
    _shocked = [False]
    _shock_step = [0]
    _pre_shock_prefs = [None]

    def init(sim):
        """Seed particles near one corner to build coherent memory."""
        _shocked[0] = False
        _shock_step[0] = 0
        _pre_shock_prefs[0] = None
        # Start near Zone A (dark corner) so memory field builds up coherently
        noise = sim.rng.normal(0, 0.08, (sim.n, sim.k))
        sim.prefs = np.clip(
            np.array([-0.7, -0.7, -0.7]) + noise, -1, 1
        ).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        # Clear the memory field so it builds fresh
        if sim.memory_field is not None:
            sim.memory_field[:] = 0.0
        apply_post_processing(sim)

    def trigger_shock(sim, step):
        """Flip all preferences to the opposite corner."""
        _pre_shock_prefs[0] = sim.prefs.copy()
        noise = sim.rng.normal(0, 0.08, (sim.n, sim.k))
        sim.prefs = np.clip(
            np.array([0.7, 0.7, 0.7]) + noise, -1, 1
        ).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)
        _shocked[0] = True
        _shock_step[0] = step

    def metrics(sim, step):
        prefs = sim.prefs.astype(np.float64)
        mean_pref = prefs.mean(axis=0)

        # Memory field metrics
        has_field = (params['memory_field'] and sim.memory_field is not None
                     and np.abs(sim.memory_field).sum() > 0)

        if has_field:
            field = sim.memory_field
            field_energy = float(np.abs(field).sum())
            # Field-pref alignment: do the particles and the field agree?
            # Sample the field at particle positions
            G = field.shape[0]
            grid_idx = np.clip(
                (sim.pos.astype(np.float64) / SPACE * G).astype(int), 0, G - 1)
            field_at_particles = field[grid_idx[:, 0], grid_idx[:, 1], :]  # (N, K)
            # Correlation between field values and particle prefs
            field_mean = field_at_particles.mean(axis=0)
            alignment = float(np.sum(mean_pref * field_mean) /
                              (np.linalg.norm(mean_pref) *
                               np.linalg.norm(field_mean) + 1e-12))

            # Field sign: what fraction of cells are positive (Zone B aligned)?
            mean_per_cell = field.mean(axis=2)
            field_sign_pos = float((mean_per_cell > 0).sum()) / mean_per_cell.size

            # Memory diversity: spatial variance of field
            mem_diversity = float(field.var())

            # Effective pref shift: how much does the field change behavior?
            strength = params['memory_strength']
            modulated = prefs * (1.0 + strength * field_at_particles[:, :prefs.shape[1]])
            modulated = np.clip(modulated, -1, 1)
            eff_shift = float(np.linalg.norm(modulated - prefs, axis=1).mean())
        else:
            field_energy = 0.0
            alignment = 0.0
            field_sign_pos = 0.5
            mem_diversity = 0.0
            eff_shift = 0.0

        result = {
            'Mean Pref (RGB)': '{:.2f}, {:.2f}, {:.2f}'.format(
                *(mean_pref[:3] + 1) / 2),
            'Field Energy': f'{field_energy:.0f}',
            'Culture-Strategy Alignment': f'{alignment:.3f}',
            'Field Sign (% positive)': f'{field_sign_pos:.1%}',
            'Cultural Inertia (eff shift)': f'{eff_shift:.4f}',
            'Memory Diversity': f'{mem_diversity:.4f}',
        }

        if _shocked[0]:
            adapt_steps = step - _shock_step[0]
            result['Phase'] = f'Post-Shock (step {adapt_steps})'
            result['Adaptability'] = (
                'Adapted' if field_sign_pos > 0.90
                else f'Adapting ({field_sign_pos:.0%} flipped)')
        else:
            result['Phase'] = 'Pre-Shock (building memory)'

        if not params['memory_field']:
            result['WARNING'] = 'Enable Memory Field for this experiment!'

        return result

    # Store trigger_shock on the controller so the GUI can call it
    ctrl._exp3_trigger_shock = trigger_shock
    ctrl._exp3_shocked = _shocked

    return {
        'init': init,
        'post_step': None,
        'check': None,
        'metrics': metrics,
    }


# ══# ═══════════════════════════════════════════════════════════════════
# Experiment 4: Mountain Climbing with Roles
# ═══════════════════════════════════════════════════════════════════

def _exp4_factory(ctrl):
    """Mountain Climbing with Heterogeneous Roles & Cost.

    Org question: "How does team composition (researchers, leaders,
    engineers, visionaries) and social dynamics affect the ability to
    discover the best strategy in a rugged landscape — and at what cost?"

    A multi-peak fitness landscape + independent cost terrain.
    Each step, particles get a noisy gradient nudge blended with a
    visionary signal toward the global summit.  Cost is accumulated.

    Key parameters:
      - Social rate: conformity vs independence
      - use_particle_roles: enable heterogeneous roles
      - role_visionary_mean/std: visionary concentration
      - role_gradient_noise_mean/std: researcher quality
      - Memory field: historical world knowledge
    """
    from experiments.landscape import (
        make_default_landscape, make_default_cost_landscape,
        compute_employee_cost,
    )

    _prev_prefs = [None]
    _gradient_strength = [0.003]
    _cost_acc = [{'total': 0.0, 'terrain': 0.0, 'employee': 0.0}]

    def init(sim):
        """Start particles near origin (far from global peak)."""
        _prev_prefs[0] = None
        _cost_acc[0] = {'total': 0.0, 'terrain': 0.0, 'employee': 0.0}
        noise = sim.rng.normal(0, 0.15, (sim.n, sim.k))
        sim.prefs = np.clip(noise, -1, 1).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        sim._init_roles()
        if sim.memory_field is not None:
            sim.memory_field[:] = 0.0
        apply_post_processing(sim)

    def post_step(sim, step):
        """Noisy gradient + visionary blend + cost accumulation."""
        landscape = make_default_landscape(k=sim.k)
        cost_landscape = make_default_cost_landscape(k=sim.k)
        prefs = sim.prefs.astype(np.float64)

        # True local gradient
        grad_unit, fitness, peak_ids = landscape.gradient(prefs)

        # Per-particle noise (researcher factor from core sim)
        noise = sim.rng.normal(0, 1, grad_unit.shape)
        noise *= sim.role_gradient_noise[:, None]
        noisy_grad = grad_unit + noise
        norms = np.linalg.norm(noisy_grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        noisy_grad = noisy_grad / norms

        # Visionary signal (direction to global summit)
        global_center = landscape.centers[0]
        to_summit = global_center - prefs
        summit_norms = np.linalg.norm(to_summit, axis=1, keepdims=True)
        summit_norms = np.maximum(summit_norms, 1e-12)
        summit_dir = to_summit / summit_norms

        # Blend: (1 - v) * local + v * summit
        v = sim.role_visionary[:, None]
        effective_grad = (1.0 - v) * noisy_grad + v * summit_dir
        eff_norms = np.linalg.norm(effective_grad, axis=1, keepdims=True)
        eff_norms = np.maximum(eff_norms, 1e-12)
        effective_grad = effective_grad / eff_norms

        # Apply nudge
        nudge = _gradient_strength[0] * effective_grad
        sim.prefs = np.clip(prefs + nudge, -1, 1).astype(sim.prefs.dtype)
        apply_post_processing(sim)

        # Accumulate cost
        terrain_cost = cost_landscape.cost(sim.prefs)
        employee_cost = compute_employee_cost(sim)
        _cost_acc[0]['total'] += float((terrain_cost + employee_cost).sum())
        _cost_acc[0]['terrain'] += float(terrain_cost.sum())
        _cost_acc[0]['employee'] += float(employee_cost.sum())

    def metrics(sim, step):
        landscape = make_default_landscape(k=sim.k)
        fitness, peak_ids = landscape.fitness(sim.prefs)

        # Summit fraction
        global_center = landscape.centers[0]
        diff = sim.prefs.astype(np.float64) - global_center
        dists_to_summit = np.linalg.norm(diff, axis=1)
        summit_frac = float((dists_to_summit <= 0.35).sum() / len(sim.prefs))

        # Local trap
        local_trap = 0
        for pid in range(1, landscape.n_peaks):
            diff_l = sim.prefs.astype(np.float64) - landscape.centers[pid]
            dists_l = np.linalg.norm(diff_l, axis=1)
            local_trap += (dists_l <= 0.35).sum()
        local_trap_frac = float(local_trap / len(sim.prefs))

        # Clustering
        cm = _cluster_metrics(sim.prefs)

        # Exploration rate
        exp_rate = _exploration_rate(sim.prefs, _prev_prefs[0])
        _prev_prefs[0] = sim.prefs.copy()

        # Cost
        acc = _cost_acc[0]
        cost_str = f"{acc['total']:.0f} (terr={acc['terrain']:.0f}, emp={acc['employee']:.0f})"

        # Role info
        has_roles = params.get('use_particle_roles', False)
        role_info = ''
        if has_roles:
            ss = sim.role_step_scale
            inf = sim.role_influence
            gn = sim.role_gradient_noise
            vis = sim.role_visionary
            role_info = (f'eng={ss.mean():.2f}\u00b1{ss.std():.2f}, '
                         f'ldr={inf.mean():.2f}\u00b1{inf.std():.2f}, '
                         f'res_noise={gn.mean():.2f}\u00b1{gn.std():.2f}, '
                         f'vis={vis.mean():.2f}\u00b1{vis.std():.2f}')

        # Efficiency
        efficiency = summit_frac / (acc['total'] + 1) * 1e6

        result = {
            'Mean Fitness': f'{fitness.mean():.4f}',
            'Max Fitness': f'{fitness.max():.4f}',
            'Summit Fraction': f'{summit_frac:.1%}',
            'Local Trap %': f'{local_trap_frac:.1%}',
            'Teams (clusters)': str(cm['cluster_count']),
            'Exploration Rate': f'{exp_rate:.4f}',
            'Total Cost': cost_str,
            'Efficiency (summit/cost)': f'{efficiency:.2f}',
        }
        if role_info:
            result['Roles'] = role_info
        if not has_roles:
            result['Roles'] = 'Disabled (enable use_particle_roles)'
        return result

    return {
        'init': init,
        'post_step': post_step,
        'check': None,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5: Dual-Space Mountain Climbing
# ═══════════════════════════════════════════════════════════════════════

def _exp5_factory(ctrl):
    """Dual-Space Mountain Climbing.

    Org question: "Does separating team culture from strategy help or
    hurt mountain climbing — and at what coupling level?"

    Preferences drive team formation (social dynamics, neighbor graph).
    Strategy drives mountain position (gradient sensing, knowledge sharing).
    The coupling parameter controls how much these overlap.

    The post-step hook calls sim.strategy_step() which uses the
    preference-space neighbor graph for team-aggregated navigation.
    """
    from experiments.landscape import (
        make_default_landscape, make_default_cost_landscape,
        compute_employee_cost,
    )
    _prev_prefs = [None]
    _cost_acc = [{'total': 0.0, 'terrain': 0.0, 'employee': 0.0}]

    def init(sim):
        """Scatter particles uniformly and enable strategy mode."""
        _prev_prefs[0] = None
        _cost_acc[0] = {'total': 0.0, 'terrain': 0.0, 'employee': 0.0}
        sim.prefs = sim.rng.uniform(-1, 1, (sim.n, sim.k)).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)
        # Re-initialize strategy from prefs with coupling
        if sim.strategy is not None:
            coupling = params.get('pref_strategy_coupling', 0.5)
            noise = sim.rng.uniform(-1, 1, sim.strategy.shape).astype(
                sim.strategy.dtype)
            sim.strategy = (coupling * sim.prefs[:, :sim.strategy_k]
                            + (1 - coupling) * noise)
            np.clip(sim.strategy, -1, 1, out=sim.strategy)

    def post_step(sim, step):
        """Phase 2: team-aggregated mountain navigation via strategy_step."""
        landscape = make_default_landscape(k=sim.strategy_k if sim.strategy is not None else sim.k)
        cost_landscape = make_default_cost_landscape(k=sim.strategy_k if sim.strategy is not None else sim.k)
        summit_center = landscape.centers[0]

        def _gradient_fn(strategy):
            return landscape.gradient(strategy)

        # Call the core strategy_step method
        sim.strategy_step(gradient_fn=_gradient_fn,
                          summit_center=summit_center)

        # Accumulate cost in strategy space
        coords = sim.strategy if sim.strategy is not None else sim.prefs
        terrain_cost = cost_landscape.cost(coords)
        employee_cost = compute_employee_cost(sim)
        _cost_acc[0]['total'] += float((terrain_cost + employee_cost).sum())
        _cost_acc[0]['terrain'] += float(terrain_cost.sum())
        _cost_acc[0]['employee'] += float(employee_cost.sum())

    def metrics(sim, step):
        landscape = make_default_landscape(k=sim.strategy_k if sim.strategy is not None else sim.k)
        coords = sim.strategy if sim.strategy is not None else sim.prefs
        coords_f64 = coords.astype(np.float64)

        fitness, peak_ids = landscape.fitness(coords_f64)

        # Summit fraction (strategy space)
        global_center = landscape.centers[0]
        diff = coords_f64 - global_center
        dists_to_summit = np.linalg.norm(diff, axis=1)
        summit_frac = float((dists_to_summit <= 0.35).sum() / len(coords))

        # Local trap fraction
        local_trap = 0
        for pid in range(1, landscape.n_peaks):
            diff_l = coords_f64 - landscape.centers[pid]
            dists_l = np.linalg.norm(diff_l, axis=1)
            local_trap += (dists_l <= 0.35).sum()
        local_trap_frac = float(local_trap / len(coords))

        # Preference-space team metrics
        cm = _cluster_metrics(sim.prefs)

        # Strategy-preference divergence
        if sim.strategy is not None and sim.strategy_k == sim.k:
            divergence = float(np.linalg.norm(
                coords_f64 - sim.prefs.astype(np.float64),
                axis=1).mean())
        else:
            divergence = 0.0

        # Exploration rate (preference space)
        exp_rate = _exploration_rate(sim.prefs, _prev_prefs[0])
        _prev_prefs[0] = sim.prefs.copy()

        # Cost
        acc = _cost_acc[0]
        efficiency = summit_frac / (acc['total'] + 1) * 1e6

        # Coupling info
        coupling = params.get('pref_strategy_coupling', 0.5)

        result = {
            'Mean Fitness': f'{fitness.mean():.4f}',
            'Summit Fraction': f'{summit_frac:.1%}',
            'Local Trap %': f'{local_trap_frac:.1%}',
            'Teams (pref clusters)': str(cm['cluster_count']),
            'Pref-Strategy Divergence': f'{divergence:.3f}',
            'Coupling': f'{coupling:.2f}',
            'Exploration Rate': f'{exp_rate:.4f}',
            'Total Cost': f"{acc['total']:.0f}",
            'Efficiency (summit/cost)': f'{efficiency:.2f}',
        }

        # Strategy enabled status
        if sim.strategy is None:
            result['WARNING'] = 'strategy_enabled is OFF — using prefs as strategy'

        return result

    return {
        'init': init,
        'post_step': post_step,
        'check': None,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_REGISTRY = [
    {
        'name': '1: Team Formation & Alignment',
        'description': (
            'Org question: How does communication culture affect '
            'team formation?\n\n'
            'Tracks: cluster count, team coherence, cognitive diversity, '
            'org balance, lone wolves, stability, exploration rate.\n\n'
            'Try: Social > 0 (conformity) vs Social < 0 (differentiation).\n'
            'Also: n_neighbors, repulsion, social_mode.'
        ),
        'factory': _exp1_factory,
    },
    {
        'name': '2: Innovation vs Efficiency',
        'description': (
            'Org question: How do we maintain innovation while '
            'keeping operational efficiency?\n\n'
            'Tracks: team tightness (efficiency), idea spread (innovation), '
            'specialization, groupthink, pref space coverage.\n\n'
            'Try: Social Mode = Quiet-Dim Diff vs Uniform.\n'
            'Also: Social rate, repulsion.'
        ),
        'factory': _exp2_factory,
    },
    {
        'name': '3: Memory & Adaptability',
        'description': (
            'Org question: How much institutional memory helps vs '
            'hurts when the environment changes?\n\n'
            'Tracks: field energy, culture-strategy alignment, '
            'cultural inertia, memory diversity.\n\n'
            'Click "Trigger Shock" to simulate a market disruption.\n'
            'REQUIRES: Memory Field ON, Field Strength > 0.\n'
            'Try: different Field Strength and Decay values.'
        ),
        'factory': _exp3_factory,
    },
    {
        'name': '4: Mountain Climbing (Roles & Cost)',
        'description': (
            'Org question: How does team composition (researchers, '
            'leaders, engineers, visionaries) affect strategy discovery '
            '\u2014 and at what cost?\n\n'
            'Multi-peak fitness landscape + cost terrain. '
            'Visionaries sense the summit but march through expensive '
            'terrain. Tracks: fitness, summit %, traps, total cost, '
            'efficiency (summit/cost).\n\n'
            'ENABLE: use_particle_roles for heterogeneous teams.\n'
            'Try: role_visionary_mean, role_gradient_noise_mean, '
            'Social rate, Memory Field.'
        ),
        'factory': _exp4_factory,
    },
    {
        'name': '5: Dual-Space Mountain (Prefs \u2260 Strategy)',
        'description': (
            'Org question: Does separating team culture from strategy '
            'help or hurt mountain climbing?\n\n'
            'Preferences drive team formation (social dynamics). '
            'Strategy drives mountain position (gradient + knowledge '
            'sharing through teams). Coupling controls overlap.\n\n'
            'ENABLE: strategy_enabled, use_particle_roles.\n'
            'Try: pref_strategy_coupling (0=generalist, 1=specialist), '
            'Social rate, role params, strategy_memory_enabled.'
        ),
        'factory': _exp5_factory,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Controller
# ═══════════════════════════════════════════════════════════════════════

class ExperimentController:
    """Manages experiment lifecycle inside the visualizer."""

    def __init__(self):
        self.active = False
        self.selected_idx = 0
        self.exp_step = 0

        # User preference: whether to apply init conditions
        self.use_init = True

        # Callbacks (set when an experiment starts)
        self._post_step = None
        self._check = None
        self._metrics = None
        self._init = None

        # Last computed metrics for display
        self._last_metrics = {}

        # Experiment-specific state
        self._exp3_trigger_shock = None
        self._exp3_shocked = [False]

    def start(self, sim):
        """Start the selected experiment."""
        entry = EXPERIMENT_REGISTRY[self.selected_idx]
        callbacks = entry['factory'](self)
        self._post_step = callbacks.get('post_step')
        self._check = callbacks.get('check')
        self._metrics = callbacks.get('metrics')
        self._init = callbacks.get('init')
        self.active = True
        self.exp_step = 0
        self._last_metrics = {}

        # Apply init only if user opted in
        if self.use_init and self._init is not None:
            self._init(sim)

    def stop(self):
        """Stop the current experiment."""
        self.active = False
        self._post_step = None
        self._check = None
        self._metrics = None
        self._init = None

    def on_step(self, sim):
        """Called after each sim.step() in the main loop."""
        if not self.active:
            return
        self.exp_step += 1

        # Post-step hook (if any)
        if self._post_step is not None:
            self._post_step(sim, self.exp_step)

        # Update metrics (every step for responsiveness)
        if self._metrics is not None:
            self._last_metrics = self._metrics(sim, self.exp_step)

    def on_reset(self, sim):
        """Called when the user resets the simulation."""
        if self.active:
            self.exp_step = 0
            self._last_metrics = {}
            if self.use_init and self._init is not None:
                self._init(sim)

    def status_text(self):
        """Short status string for the window title bar."""
        if not self.active:
            return ""
        name = EXPERIMENT_REGISTRY[self.selected_idx]['name']
        return f"OBS [{name}] step {self.exp_step}"

    def draw_gui(self, sim):
        """Draw the experiment control panel using imgui."""
        if not _HAS_IMGUI:
            return
        imgui.separator()
        if not imgui.collapsing_header(
                "Experiments",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            return

        # Experiment selector (only when not running)
        names = [e['name'] for e in EXPERIMENT_REGISTRY]
        if not self.active:
            changed, v = imgui.combo("Experiment", self.selected_idx, names)
            if changed:
                self.selected_idx = v

        # Description
        desc = EXPERIMENT_REGISTRY[self.selected_idx]['description']
        imgui.text_wrapped(desc)

        # Init conditions checkbox (only when not running)
        if not self.active:
            changed, v = imgui.checkbox(
                "Set initial conditions (overwrites current prefs)",
                self.use_init)
            if changed:
                self.use_init = v

        # Start / Stop / Reset buttons
        if not self.active:
            if imgui.button("Start Observatory", imgui.ImVec2(160, 0)):
                self.start(sim)
        else:
            if imgui.button("Stop Observatory", imgui.ImVec2(160, 0)):
                self.stop()
            imgui.same_line()
            if imgui.button("Reset", imgui.ImVec2(80, 0)):
                self.on_reset(sim)

            # Exp 3: Shock button
            if (self.selected_idx == 2 and self.active
                    and self._exp3_trigger_shock is not None
                    and not self._exp3_shocked[0]):
                imgui.same_line()
                if imgui.button("Trigger Shock", imgui.ImVec2(120, 0)):
                    self._exp3_trigger_shock(sim, self.exp_step)

        # Status
        if self.active:
            imgui.text(f"Step: {self.exp_step}")

            # Metrics
            if self._last_metrics:
                imgui.separator()
                for key, val in self._last_metrics.items():
                    if key == 'WARNING':
                        imgui.text_colored(
                            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                            f"  {val}")
                    else:
                        imgui.text(f"  {key}: {val}")
