"""
Headless experiment runner for the social-dynamics simulation.

Parameter resolution order
--------------------------
When ``run_experiment()`` is called, parameters are resolved in three layers:

  1. **SAFE_DEFAULTS** — a known-good baseline so that experiments are
     reproducible out of the box when run from the command line.
  2. **Ambient params** — the current contents of the global ``params`` dict
     (whatever the user has configured in the GUI or ``params.py``).  Any
     value the user has explicitly changed will override the safe default.
  3. **Experiment overrides** — the parameters the experiment *must* control
     (e.g. the swept variable).  These always win.

This means:
  - Running headlessly from the CLI gives clean, reproducible results.
  - Running from inside the visualiser after tweaking settings lets the
    user see how their changes interact with the experiment.
  - The experiment's own critical params always take priority.

After the experiment finishes, the FULL original params dict is restored
(even on error) so the user's GUI state is never permanently mutated.

Usage
-----
    from experiments.runner import run_experiment

    result = run_experiment(
        param_overrides={...},
        max_steps=50000,
        check_fn=my_check,       # (sim, step) -> bool
        post_step_fn=my_hook,    # (sim, step) -> None
        log_interval=100,
        log_fn=my_logger,        # (sim, step) -> dict
    )
"""

import sys
import os
import time
import numpy as np

# Ensure the package root is importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from sim_2d_exp.params import params as _global_params
from sim_2d_exp.simulation import Simulation


# ── Safe baseline defaults ──────────────────────────────────────────────
# A known-good starting point so experiments are reproducible when run
# headlessly with no prior GUI state.  The user's ambient params (layer 2)
# override these, and experiment overrides (layer 3) override both.

SAFE_DEFAULTS = {
    # Core
    'num_particles': 500,
    'k': 3,
    'n_neighbors': 15,
    'step_size': 0.005,
    'steps_per_frame': 1,
    'repulsion': 0.0,
    'dir_memory': 0.0,
    'social': 0.0,
    'social_mode': 0,
    'social_dist_weight': False,
    'pref_weighted_dir': False,
    'pref_inner_prod': False,
    'inner_prod_avg': False,
    'pref_dist_weight': False,
    'best_mode': 0,
    'neighbor_mode': 0,
    'neighbor_radius': 0.06,
    'pos_dist': 0,
    'pref_dist': 0,
    'gauss_sigma': 0.15,
    'binary_noise_eps': 0.1,
    # Seed
    'use_seed': True,
    'seed': 42,
    # Precision / dtype
    'use_f64': True,
    'pref_precision': 2,
    'perturb_pos_bits': False,
    'perturb_pos_n_bits': 1,
    'truncate_pos_bits': False,
    'pos_mantissa_bits': 52,
    'truncate_pref_bits': False,
    'pref_mantissa_bits': 23,
    'quantize_pos': False,
    'quantize_pref': False,
    'pref_quant_levels': 10,
    'unit_prefs': False,
    # Physics engine
    'physics_engine': 1,        # NumPy — headless-safe, no GPU needed
    'grid_res': 256,
    'grid_sigma': 2.0,
    'grid_max_spread': 16,
    'grid_circular': False,
    'torch_precision': 3,
    'torch_device': 0,
    # KNN
    'knn_method': 1,
    'reuse_neighbors': True,
    # Memory field
    'memory_field': False,
    'memory_write_rate': 0.01,
    'memory_strength': 0.5,
    'memory_decay': 0.999,
    'memory_blur': False,
    'memory_blur_sigma': 1.0,
    # Signal / response
    'use_signal_response': False,
    'swap_signal_response': False,
    # Crossover
    'crossover': False,
    'crossover_pct': 50,
    'crossover_interval': 1,
    # Per-particle roles
    'use_particle_roles': False,
    'role_step_scale_std': 0.0,
    'role_influence_std': 0.0,
    'role_gradient_noise_mean': 0.5,
    'role_gradient_noise_std': 0.0,
    'role_visionary_mean': 0.0,
    'role_visionary_std': 0.0,
    # Shadow
    'shadow_sim': False,
    'shadow_show_lines': True,
    # Display-only (safe values; not used headlessly but may be read
    # during Simulation.__init__)
    'auto_scale': False,
    'debug_knn': False,
    'track_mode': 0,
}


def apply_post_processing(sim):
    """Re-apply the simulation's post-step normalisation pipeline.

    Call this after any external modification to ``sim.prefs`` or
    ``sim.response`` to ensure the same processing that ``step()``
    performs is honoured (truncation, quantisation, unit-norm, etc.).
    """
    from sim_2d_exp.params import params
    if params.get('truncate_pref_bits'):
        sim._truncate_pref_bits()
    if params.get('quantize_pref'):
        sim._quantize_prefs()
    if params.get('unit_prefs'):
        sim._normalize_prefs()


def _resolve_params(param_overrides, seed=None):
    """Build the final param dict using three-layer resolution.

    Layer 1: SAFE_DEFAULTS  (reproducible baseline)
    Layer 2: ambient params (user's current GUI / params.py state)
    Layer 3: param_overrides (experiment-critical settings)

    Returns a dict ready to be written into _global_params.
    """
    resolved = {}
    resolved.update(SAFE_DEFAULTS)          # Layer 1
    resolved.update(_global_params)         # Layer 2 — user's state wins
    resolved.update(param_overrides)        # Layer 3 — experiment wins

    if seed is not None:
        resolved['use_seed'] = True
        resolved['seed'] = seed

    return resolved


def run_experiment(
    param_overrides: dict,
    max_steps: int = 50_000,
    check_fn=None,
    post_step_fn=None,
    log_interval: int = 100,
    log_fn=None,
    seed: int | None = None,
):
    """Run a single headless experiment trial.

    Parameters
    ----------
    param_overrides : dict
        Params the experiment *must* control (swept variable, etc.).
        Applied as layer 3 on top of SAFE_DEFAULTS and ambient state.
    max_steps : int
        Maximum simulation steps before declaring failure.
    check_fn : callable(sim, step) -> bool, optional
        Return True when the experiment objective is met.
    post_step_fn : callable(sim, step) -> None, optional
        Called after each ``sim.step()``.  If you modify ``sim.prefs``
        here, call ``apply_post_processing(sim)`` afterward.
    log_interval : int
        Record metrics every N steps.
    log_fn : callable(sim, step) -> dict, optional
        Returns a dict of named metrics to log.
    seed : int, optional
        If provided, forces ``use_seed=True`` and sets ``seed``.

    Returns
    -------
    dict with keys: solved, solve_step, total_steps, wall_time, log.
    """
    # ── Snapshot the ENTIRE global params dict ──
    full_snapshot = dict(_global_params)

    try:
        # ── Three-layer resolution ──
        resolved = _resolve_params(param_overrides, seed=seed)
        _global_params.clear()
        _global_params.update(resolved)

        # ── Create simulation ──
        sim = Simulation()

        log = []
        solved = False
        solve_step = None
        step = 0
        t0 = time.perf_counter()

        for step in range(1, max_steps + 1):
            sim.step()

            # Post-step hook (gradient injection, env shock, etc.)
            if post_step_fn is not None:
                post_step_fn(sim, step)

            # Check termination
            if check_fn is not None and check_fn(sim, step):
                solved = True
                solve_step = step
                if log_fn is not None:
                    entry = log_fn(sim, step)
                    entry['step'] = step
                    log.append(entry)
                break

            # Periodic logging
            if log_fn is not None and step % log_interval == 0:
                entry = log_fn(sim, step)
                entry['step'] = step
                log.append(entry)

    finally:
        # ── Restore the FULL original params ──
        _global_params.clear()
        _global_params.update(full_snapshot)

    wall_time = time.perf_counter() - t0

    return {
        'solved': solved,
        'solve_step': solve_step,
        'total_steps': step,
        'wall_time': wall_time,
        'log': log,
    }


def run_sweep(
    experiment_fn,
    sweep_param: str,
    sweep_values: list,
    n_trials: int = 10,
    base_overrides: dict | None = None,
    **runner_kwargs,
):
    """Run a parameter sweep: for each value of sweep_param, execute n_trials.

    Parameters
    ----------
    experiment_fn : callable(param_overrides) -> dict
        Returns runner kwargs (check_fn, post_step_fn, log_fn, etc.).
    sweep_param : str
        Name of the param to sweep.
    sweep_values : list
        Values to try.
    n_trials : int
        Monte Carlo trials per value.
    base_overrides : dict, optional
        Params shared across all runs (layer 3 overrides).
    **runner_kwargs
        Additional kwargs passed to ``run_experiment()``.

    Returns
    -------
    list of dicts, one per (value, trial) combination.
    """
    if base_overrides is None:
        base_overrides = {}

    results = []
    for val in sweep_values:
        for trial in range(n_trials):
            overrides = {**base_overrides, sweep_param: val}
            exp_kwargs = experiment_fn(overrides)
            merged_kwargs = {**runner_kwargs, **exp_kwargs}
            result = run_experiment(
                param_overrides=overrides,
                seed=trial * 1000 + hash(str(val)) % 10000,
                **merged_kwargs,
            )
            result['sweep_param'] = sweep_param
            result['sweep_value'] = val
            result['trial'] = trial
            results.append(result)
            status = "SOLVED" if result['solved'] else "FAILED"
            print(f"  {sweep_param}={val}, trial={trial}: "
                  f"{status} at step {result['solve_step']} "
                  f"({result['wall_time']:.1f}s)")
    return results
