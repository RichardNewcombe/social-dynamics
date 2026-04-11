"""
ExperimentController — live experiment integration for the visualizer.

This controller attaches to the renderer's existing ``Simulation`` instance
(it does NOT create its own).  It provides:

  - ``on_step(sim)``   — called after each ``sim.step()`` in the main loop
  - ``on_reset(sim)``  — called when the user resets the simulation
  - ``draw_gui(sim)``  — draws an imgui panel for experiment controls
  - ``status_text()``  — returns a short status string for the title bar

The controller manages experiment lifecycle (start / stop / reset) and
applies post-step hooks (gradient nudges, fitness checks, metric logging)
without owning or replacing the simulation.

Each experiment has an ``init`` hook that sets up specific initial
conditions (e.g. all particles start far from the target, or clustered
in one corner of preference space).  This ensures the experiment is
non-trivial and produces visible changes in the display.
"""

import numpy as np

try:
    from imgui_bundle import imgui
    _HAS_IMGUI = True
except ImportError:
    _HAS_IMGUI = False

from experiments.runner import apply_post_processing
from experiments.fitness import (
    hidden_target_fitness, hidden_target_gradient,
    niche_occupancy, generate_niches,
    zone_fitness,
)
from sim_2d_exp.params import params, SPACE


# ═══════════════════════════════════════════════════════════════════════
# Experiment definitions
# ═══════════════════════════════════════════════════════════════════════

def _exp1_factory(ctrl):
    """Hidden Target Search.

    All particles start with preferences at the OPPOSITE corner from the
    target, so they must traverse the full preference space.  The gradient
    nudge is strong enough to produce a visible, steady colour shift from
    dark-blue toward the target colour (orange-ish).

    Solved when 80% of particles are within epsilon of the target.
    """
    target = np.array([0.7, -0.5, 0.3])
    start_pref = np.array([-0.7, 0.5, -0.3])   # opposite corner
    strength = 0.003
    epsilon = 0.15                               # tight — must really converge
    frac_required = 0.80                         # 80% of all particles
    noise_std = 0.05

    def init(sim):
        """Seed all particles at the opposite corner from the target."""
        noise = sim.rng.normal(0, noise_std, (sim.n, sim.k))
        sim.prefs = np.clip(start_pref + noise, -1, 1).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)

    def post_step(sim, step):
        nudge = hidden_target_gradient(sim.prefs, target, strength)
        sim.prefs = np.clip(
            sim.prefs.astype(np.float64) + nudge, -1, 1
        ).astype(sim.prefs.dtype)
        apply_post_processing(sim)

    def check(sim, step):
        diff = sim.prefs.astype(np.float64) - target.astype(np.float64)
        dists = np.linalg.norm(diff, axis=1)
        frac_near = float((dists <= epsilon).sum()) / sim.n
        return frac_near >= frac_required

    def metrics(sim, step):
        fitness = hidden_target_fitness(sim.prefs, target)
        diff = sim.prefs.astype(np.float64) - target.astype(np.float64)
        dists = np.linalg.norm(diff, axis=1)
        frac_near = float((dists <= epsilon).sum()) / sim.n
        return {
            'Mean Fitness': f'{fitness.mean():.3f}',
            'Mean Distance': f'{dists.mean():.3f}',
            f'Frac Near Target (>={frac_required:.0%})': f'{frac_near:.1%}',
            'Pref Std': f'{sim.prefs.std(axis=0).mean():.4f}',
        }

    return {
        'post_step': post_step,
        'check': check,
        'metrics': metrics,
        'init': init,
    }


def _exp2_factory(ctrl):
    """Multi-Niche Coverage.

    All particles start clustered at a single neutral point (gray, near
    the origin).  Four niches are placed at the corners of preference
    space.  The gradient nudges each particle toward its nearest niche,
    so you see the gray blob split into four distinct colour clusters.

    Solved when every niche has at least 10% of the total population.
    """
    niches = np.array([
        [ 0.8,  0.8,  0.8],   # white-ish
        [ 0.8, -0.8, -0.8],   # red
        [-0.8,  0.8, -0.8],   # green
        [-0.8, -0.8,  0.8],   # blue
    ])
    radius = 0.40
    frac_per_niche = 0.10                        # 10% of N in each niche
    strength = 0.002
    noise_std = 0.15                             # moderate spread so split is visible

    def init(sim):
        """Seed all particles near the origin (gray)."""
        noise = sim.rng.normal(0, noise_std, (sim.n, sim.k))
        sim.prefs = np.clip(noise, -1, 1).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)

    def post_step(sim, step):
        prefs_f64 = sim.prefs.astype(np.float64)
        niches_f64 = niches.astype(np.float64)
        dists = np.linalg.norm(
            prefs_f64[:, None, :] - niches_f64[None, :, :], axis=2
        )
        nearest = np.argmin(dists, axis=1)
        target_pref = niches_f64[nearest]
        diff = target_pref - prefs_f64
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        nudge = strength * (diff / norms)
        sim.prefs = np.clip(
            prefs_f64 + nudge, -1, 1
        ).astype(sim.prefs.dtype)
        apply_post_processing(sim)

    def check(sim, step):
        counts, _ = niche_occupancy(sim.prefs, niches, radius)
        min_needed = int(sim.n * frac_per_niche)
        return bool(np.all(counts >= min_needed))

    def metrics(sim, step):
        counts, occupied = niche_occupancy(sim.prefs, niches, radius)
        min_needed = int(sim.n * frac_per_niche)
        return {
            'Niches Filled': f'{int(occupied.sum())} / {len(niches)}',
            'Niche Counts': ', '.join(str(int(c)) for c in counts),
            f'Min Needed ({frac_per_niche:.0%} of N)': str(min_needed),
            'Pref Std': f'{sim.prefs.std(axis=0).mean():.4f}',
        }

    return {
        'post_step': post_step,
        'check': check,
        'metrics': metrics,
        'init': init,
    }


def _exp3_factory(ctrl):
    """Ghost Colony Escape.

    All particles start near Zone A (dark corner) and settle there,
    building spatial memory.  After a settling phase, a "shock" flips
    all preferences to Zone B (bright corner).  The memory field fights
    this change — you can see the particles' effective colours lag behind
    as the old memory decays.

    Solved when 60% of memory-modulated prefs are near Zone B.
    Enable Memory Field and increase Field Strength to see the effect.
    """
    zone_a = np.array([-0.7, -0.7, -0.7])
    zone_b = np.array([0.7, 0.7, 0.7])
    settle_steps = 2500
    threshold = 0.60
    noise_std = 0.08

    def init(sim):
        """Seed particles near Zone A."""
        noise = sim.rng.normal(0, noise_std, (sim.n, sim.k))
        sim.prefs = np.clip(zone_a + noise, -1, 1).astype(sim.prefs.dtype)
        sim.response = sim.prefs.copy()
        apply_post_processing(sim)
        ctrl._exp3_shocked = False
        ctrl._exp3_settle_step = ctrl.exp_step + settle_steps
        ctrl._exp3_shock_step = 0

    def _get_modulated_prefs(sim):
        """Read-only computation of memory-modulated prefs."""
        if not params['memory_field'] or sim.memory_field is None:
            return sim.prefs.copy()
        mem_strength = params['memory_strength']
        G = sim.memory_field.shape[0]
        inv_cell = G / SPACE
        cx = (sim.pos[:, 0] * inv_cell).astype(int) % G
        cy = (sim.pos[:, 1] * inv_cell).astype(int) % G
        field_at = sim.memory_field[cy, cx]
        modulated = sim.prefs.astype(np.float64) * (1.0 + mem_strength * field_at)
        return np.clip(modulated, -1, 1)

    def _frac_near(prefs, zone, r=0.6):
        diff = prefs.astype(np.float64) - zone.astype(np.float64)
        dists = np.linalg.norm(diff, axis=1)
        return float((dists <= r).sum() / len(prefs))

    def post_step(sim, step):
        if not ctrl._exp3_shocked and ctrl.exp_step >= ctrl._exp3_settle_step:
            # SHOCK: flip all prefs to Zone B
            noise = sim.rng.normal(0, noise_std, (sim.n, sim.k))
            sim.prefs = np.clip(zone_b + noise, -1, 1).astype(sim.prefs.dtype)
            sim.response = sim.prefs.copy()
            apply_post_processing(sim)
            ctrl._exp3_shocked = True
            ctrl._exp3_shock_step = ctrl.exp_step

    def check(sim, step):
        if not ctrl._exp3_shocked:
            return False
        mod_prefs = _get_modulated_prefs(sim)
        frac_b = _frac_near(mod_prefs, zone_b)
        return frac_b >= threshold

    def metrics(sim, step):
        mod_prefs = _get_modulated_prefs(sim)
        frac_a = _frac_near(mod_prefs, zone_a)
        frac_b = _frac_near(mod_prefs, zone_b)
        phase = 'Settling' if not ctrl._exp3_shocked else 'Adapting'
        adapt_steps = (ctrl.exp_step - ctrl._exp3_shock_step) if ctrl._exp3_shocked else 0
        mem_energy = float(np.abs(sim.memory_field).sum()) if sim.memory_field is not None else 0.0
        result = {
            'Phase': phase,
            'Frac Zone A (mod)': f'{frac_a:.3f}',
            'Frac Zone B (mod)': f'{frac_b:.3f}',
            'Memory Energy': f'{mem_energy:.1f}',
        }
        if ctrl._exp3_shocked:
            result['Adapt Steps'] = str(adapt_steps)
        else:
            remaining = ctrl._exp3_settle_step - ctrl.exp_step
            result['Settle Remaining'] = str(max(0, remaining))
        return result

    return {
        'post_step': post_step,
        'check': check,
        'metrics': metrics,
        'init': init,
    }


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_REGISTRY = [
    {
        'name': '1: Hidden Target Search',
        'description': (
            'All particles start at the OPPOSITE corner of preference '
            'space from the hidden target. Watch the swarm change colour '
            'as it converges. Tweak Social learning rate to see '
            'exploration vs exploitation.'
        ),
        'factory': _exp1_factory,
    },
    {
        'name': '2: Multi-Niche Coverage',
        'description': (
            'All particles start gray (near origin). Four coloured niches '
            'pull them apart. Watch the blob split into four distinct '
            'colour clusters. Compare Social Modes to see diversity.'
        ),
        'factory': _exp2_factory,
    },
    {
        'name': '3: Ghost Colony Escape',
        'description': (
            'Particles start dark (Zone A) and settle for 2500 steps, '
            'building spatial memory. Then a shock flips them bright '
            '(Zone B). Memory fights the change. Enable Memory Field '
            'and increase Field Strength to see institutional inertia.'
        ),
        'factory': _exp3_factory,
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
        self.solved = False
        self.solve_step = None

        # Callbacks (set when an experiment starts)
        self._post_step = None
        self._check = None
        self._metrics = None
        self._init = None

        # Last computed metrics for display
        self._last_metrics = {}

        # Experiment-specific state (used by exp3)
        self._exp3_shocked = False
        self._exp3_settle_step = 0
        self._exp3_shock_step = 0

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
        self.solved = False
        self.solve_step = None
        self._last_metrics = {}
        # Run init hook — sets up initial conditions so experiment
        # is non-trivial and produces visible changes
        if self._init is not None:
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
        if not self.active or self.solved:
            return
        self.exp_step += 1

        # Post-step hook (gradient injection, shock, etc.)
        if self._post_step is not None:
            self._post_step(sim, self.exp_step)

        # Check termination
        if self._check is not None and self._check(sim, self.exp_step):
            self.solved = True
            self.solve_step = self.exp_step

        # Update metrics (every step for responsiveness)
        if self._metrics is not None:
            self._last_metrics = self._metrics(sim, self.exp_step)

    def on_reset(self, sim):
        """Called when the user resets the simulation."""
        if self.active:
            # Re-initialize the experiment on the fresh sim
            self.exp_step = 0
            self.solved = False
            self.solve_step = None
            self._last_metrics = {}
            if self._init is not None:
                self._init(sim)

    def status_text(self):
        """Short status string for the window title bar."""
        if not self.active:
            return ""
        name = EXPERIMENT_REGISTRY[self.selected_idx]['name']
        if self.solved:
            return f"EXP [{name}] SOLVED at step {self.solve_step}"
        return f"EXP [{name}] step {self.exp_step}"

    def draw_gui(self, sim):
        """Draw the experiment control panel using imgui."""
        if not _HAS_IMGUI:
            return
        imgui.separator()
        if not imgui.collapsing_header(
                "Experiments",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            return

        # Experiment selector
        names = [e['name'] for e in EXPERIMENT_REGISTRY]
        if not self.active:
            changed, v = imgui.combo("Experiment", self.selected_idx, names)
            if changed:
                self.selected_idx = v

        # Description
        desc = EXPERIMENT_REGISTRY[self.selected_idx]['description']
        imgui.text_wrapped(desc)

        # Start / Stop / Reset buttons
        if not self.active:
            if imgui.button("Start Experiment", imgui.ImVec2(160, 0)):
                self.start(sim)
        else:
            if imgui.button("Stop Experiment", imgui.ImVec2(160, 0)):
                self.stop()
            imgui.same_line()
            if imgui.button("Reset Experiment", imgui.ImVec2(160, 0)):
                self.on_reset(sim)

        # Status
        if self.active:
            imgui.text(f"Experiment Step: {self.exp_step}")
            if self.solved:
                imgui.text_colored(
                    imgui.ImVec4(0.2, 1.0, 0.2, 1.0),
                    f"SOLVED at step {self.solve_step}!")
            else:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 1.0, 0.3, 1.0),
                    "Running...")

            # Metrics
            if self._last_metrics:
                imgui.separator()
                for key, val in self._last_metrics.items():
                    imgui.text(f"  {key}: {val}")
