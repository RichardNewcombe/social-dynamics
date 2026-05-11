#!/usr/bin/env python3
"""
Headless renderer for sim_2d_exp — no display required.

Uses moderngl standalone context for offscreen OpenGL rendering,
producing pixel-identical output to the windowed version including
trail accumulation on the right panel.

Supports PMem (per-particle memory histogram) view via --selected-particle
and --right-view 8 (or auto-detected when --selected-particle is set).

Usage:
    python -m sim_2d_exp.headless                    # defaults
    python -m sim_2d_exp.headless --steps 500 --particles 2000
    python -m sim_2d_exp.headless --steps 1000 --output result.png
    python -m sim_2d_exp.headless --config '{"social": -0.003, "memory_field": true}'
    python -m sim_2d_exp.headless --steps 1000 --selected-particle 0 --snapshot-steps 1,100,1000

Requires: numpy, scipy, numba, moderngl, Pillow (or matplotlib for fallback)
Does NOT require: glfw, imgui, PyOpenGL, a display server
"""

import argparse
import json
import os
import time
import platform
import numpy as np


_ENGINE_NAMES = {0: 'Numba', 1: 'NumPy', 2: 'PyTorch', 3: 'Grid Field', 4: 'Grid Max Field GPU'}
_KNN_NAMES = {0: 'Hash Grid', 1: 'cKDTree (f64)', 2: 'cKDTree (f32)'}
_NEIGHBOR_NAMES = {0: 'KNN', 1: 'KNN+Radius', 2: 'Radius', 3: 'Delaunay'}
_TORCH_PREC_NAMES = {0: 'f16', 1: 'bf16', 2: 'f32', 3: 'f64'}
_PREF_PREC_NAMES = {0: 'f16', 1: 'f32', 2: 'f64'}


def _print_startup_info(params, N, K, n_steps, W, H):
    eng = params['physics_engine']
    print("=" * 70)
    print(f"Headless run: N={N}, K={K}, steps={n_steps}, size={W}x{H}")
    print(f"Engine:        {eng} ({_ENGINE_NAMES.get(eng, '?')})")
    print(f"step_size:     {params['step_size']}")
    print(f"KNN method:    {params['knn_method']} ({_KNN_NAMES.get(params['knn_method'], '?')})")
    print(f"Neighbor mode: {params['neighbor_mode']} ({_NEIGHBOR_NAMES.get(params['neighbor_mode'], '?')})  n_neighbors={params['n_neighbors']}")
    print(f"Position dtype: {'f64' if params['use_f64'] else 'f32'}")
    print(f"Pref precision: {params['pref_precision']} ({_PREF_PREC_NAMES.get(params['pref_precision'], '?')})")

    # Engine-specific
    if eng == 2:  # PyTorch
        from .physics_torch import _HAS_TORCH, _TORCH_DEVICE
        dev_idx = params['torch_device']
        prec_idx = params['torch_precision']
        sel_dev = 'cpu' if dev_idx == 1 else _TORCH_DEVICE
        print(f"PyTorch:       available={_HAS_TORCH}  auto_device={_TORCH_DEVICE}  selected={sel_dev}")
        print(f"Torch dtype:   {prec_idx} ({_TORCH_PREC_NAMES.get(prec_idx, '?')})")
        try:
            import torch
            print(f"Torch version: {torch.__version__}")
            print(f"CUDA avail:    {torch.cuda.is_available()}", end='')
            if torch.cuda.is_available():
                print(f"  ({torch.cuda.device_count()}x {torch.cuda.get_device_name(0)})")
            else:
                print()
            mps_avail = getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
            print(f"MPS avail:     {bool(mps_avail)}")
        except Exception as e:
            print(f"Torch import failed: {e}")
    elif eng in (0,):  # Numba
        try:
            import numba
            print(f"Numba version: {numba.__version__}  threads={numba.get_num_threads()}")
        except Exception as e:
            print(f"Numba import failed: {e}")
    elif eng == 4:  # Grid GPU
        try:
            import torch
            print(f"Torch CUDA: {torch.cuda.is_available()}  MPS: {torch.backends.mps.is_available()}")
        except Exception:
            pass

    print(f"NumPy:         {np.__version__}")
    print(f"Python:        {platform.python_version()}  on {platform.machine()}")
    print("=" * 70)


def _save_frame(output_fbo, W, H, path, step_label):
    """Read pixels from output FBO and save as PNG with step overlay."""
    raw = output_fbo.read(components=3)
    img_array = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    img_array = img_array[::-1].copy()

    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        font_size = max(24, H // 30)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        x, y = 12, 8
        draw.text((x + 2, y + 2), step_label, fill=(0, 0, 0), font=font)
        draw.text((x, y), step_label, fill=(255, 255, 255), font=font)
        img.save(path)
    except ImportError:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(path, img_array)
        except ImportError:
            np.save(path.replace('.png', '.npy'), img_array)
            print(f"Neither Pillow nor matplotlib available — saved raw array to {path}")
            return

    print(f"  snapshot saved: {path}")


def _render_frame(ctx, output_fbo, half_w, H,
                  prog_particle, vao_particle, prog_display, vao_display,
                  point_size, trail_tex, pmem_trail_tex, right_view,
                  sel_idx, positions, colors):
    """Compose a full frame into output_fbo: left=particles, right=trail or pmem."""
    import moderngl
    output_fbo.use()
    ctx.clear(0.08, 0.08, 0.1)

    # Left half: particles
    ctx.viewport = (0, 0, half_w, H)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    prog_particle['viewport_offset'] = (0.0, 0.0)
    prog_particle['viewport_scale'] = (1.0, 1.0)
    prog_particle['view_center'] = (0.5, 0.5)
    prog_particle['view_zoom'] = 1.0
    prog_particle['point_size'] = point_size
    vao_particle.render(moderngl.POINTS)

    # Highlight selected particle with a larger yellow point
    if sel_idx >= 0 and positions is not None:
        sel_pos = positions[sel_idx:sel_idx+1].copy()
        sel_col = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # yellow
        from .shaders import PARTICLE_VERT, PARTICLE_FRAG
        # Reuse existing VBOs via a temporary VBO for the highlight
        vbo_sel_pos = ctx.buffer(sel_pos.tobytes())
        vbo_sel_col = ctx.buffer(sel_col.tobytes())
        vao_sel = ctx.vertex_array(prog_particle, [
            (vbo_sel_pos, '2f', 'in_pos'),
            (vbo_sel_col, '3f', 'in_color'),
        ])
        prog_particle['point_size'] = point_size + 8.0
        vao_sel.render(moderngl.POINTS)
        prog_particle['point_size'] = point_size
        vbo_sel_pos.release()
        vbo_sel_col.release()
        vao_sel.release()

    # Right half: trail or pmem texture
    ctx.viewport = (half_w, 0, half_w, H)
    ctx.blend_func = moderngl.ONE, moderngl.ZERO
    right_tex = pmem_trail_tex if right_view == 8 else trail_tex
    right_tex.use(0)
    prog_display['tex'] = 0
    prog_display['view_center'] = (0.5, 0.5)
    prog_display['view_zoom'] = 1.0
    vao_display.render(moderngl.TRIANGLE_STRIP)


def _pmem_isometric_project(prefs, k):
    """Project K-dim preferences to 2D isometric coords in [0,1]^2.

    Same projection as the interactive renderer's Pref3D / PMem views.
    """
    n = len(prefs)
    d0 = prefs[:, 0] if k > 0 else np.zeros(n, dtype=np.float32)
    d1 = prefs[:, 1] if k > 1 else np.zeros(n, dtype=np.float32)
    d2 = prefs[:, 2] if k > 2 else np.zeros(n, dtype=np.float32)
    sqrt2 = np.float32(np.sqrt(2))
    sqrt6 = np.float32(np.sqrt(6))
    px = (d0 - d2) / sqrt2
    py = (2.0 * d1 - d0 - d2) / sqrt6
    max_range = 4.0 / sqrt6
    scale = 0.45 / max_range
    pos = np.zeros((n, 2), dtype=np.float32)
    pos[:, 0] = px * scale + 0.5
    pos[:, 1] = py * scale + 0.5
    return pos, scale, sqrt2, sqrt6


def _prefs_to_rgb(prefs, k):
    """Map K-dim preferences to RGB colors (same as renderer)."""
    col = np.clip((prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
    if k < 3:
        c = np.full((len(prefs), 3), 0.5, np.float32)
        c[:, :min(k, 3)] = col[:, :min(k, 3)]
        col = c
    return col


def run_headless(args):
    import moderngl
    from .params import params, SPACE
    from .shaders import (
        PARTICLE_VERT, PARTICLE_FRAG, QUAD_VERT, TRAIL_FRAG,
        SPLAT_FRAG, DISPLAY_FRAG,
    )
    from .simulation import Simulation

    # Apply --config JSON first, then individual --<param> overrides
    if args.config:
        params.update(json.loads(args.config))
    for key in params.keys():
        v = getattr(args, key, None)
        if v is not None:
            params[key] = v

    n_steps = args.steps
    N = params['num_particles']
    K = params['k']
    W, H = args.width, args.height
    half_w = W // 2

    # ── PMem / right-view settings ──
    sel_idx = args.selected_particle if args.selected_particle is not None else -1
    right_view = args.right_view
    if right_view is None:
        right_view = 8 if sel_idx >= 0 else 0  # auto: pmem if particle selected, else trail

    # Parse snapshot steps
    snapshot_steps = set()
    if args.snapshot_steps:
        for s in args.snapshot_steps.split(','):
            s = s.strip()
            if s:
                snapshot_steps.add(int(s))
    # Always snapshot the final step
    snapshot_steps.add(n_steps)

    _print_startup_info(params, N, K, n_steps, W, H)
    if sel_idx >= 0:
        print(f"Selected particle: {sel_idx}  (right_view={right_view}{'=PMem' if right_view == 8 else ''})")
    if snapshot_steps - {n_steps}:
        print(f"Snapshot steps: {sorted(snapshot_steps)}")

    # ── Create standalone OpenGL context (no display) ──
    try:
        ctx = moderngl.create_standalone_context(require=330, backend='egl')
    except Exception:
        ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    # ── Output FBO ──
    output_tex = ctx.texture((W, H), 3, dtype='f1')
    output_fbo = ctx.framebuffer(color_attachments=[output_tex])

    # ── Compile shaders ──
    prog_particle = ctx.program(vertex_shader=PARTICLE_VERT,
                                fragment_shader=PARTICLE_FRAG)
    prog_splat = ctx.program(vertex_shader=PARTICLE_VERT,
                             fragment_shader=SPLAT_FRAG)
    prog_trail_decay = ctx.program(vertex_shader=QUAD_VERT,
                                   fragment_shader=TRAIL_FRAG)
    prog_display = ctx.program(vertex_shader=QUAD_VERT,
                               fragment_shader=DISPLAY_FRAG)

    # ── Particle VBOs ──
    vbo_pos = ctx.buffer(reserve=N * 2 * 4)
    vbo_col = ctx.buffer(reserve=N * 3 * 4)
    vao_particle = ctx.vertex_array(prog_particle, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])
    vao_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # ── Trail FBOs (ping-pong) for standard trail view ──
    trail_tex = ctx.texture((half_w, H), 3, dtype='f2')
    trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_tex.repeat_x = True
    trail_tex.repeat_y = True
    trail_fbo = ctx.framebuffer(color_attachments=[trail_tex])

    trail_tex2 = ctx.texture((half_w, H), 3, dtype='f2')
    trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_tex2.repeat_x = True
    trail_tex2.repeat_y = True
    trail_fbo2 = ctx.framebuffer(color_attachments=[trail_tex2])

    # ── PMem trail FBOs (ping-pong) ──
    pmem_trail_tex = ctx.texture((half_w, H), 3, dtype='f2')
    pmem_trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pmem_trail_fbo = ctx.framebuffer(color_attachments=[pmem_trail_tex])

    pmem_trail_tex2 = ctx.texture((half_w, H), 3, dtype='f2')
    pmem_trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pmem_trail_fbo2 = ctx.framebuffer(color_attachments=[pmem_trail_tex2])

    # ── PMem VBOs ──
    vbo_pmem_pos = ctx.buffer(reserve=N * 2 * 4)
    vbo_pmem_col = ctx.buffer(reserve=N * 3 * 4)
    vao_pmem_splat = ctx.vertex_array(prog_splat, [
        (vbo_pmem_pos, '2f', 'in_pos'),
        (vbo_pmem_col, '3f', 'in_color'),
    ])

    # ── Fullscreen quad ──
    quad_data = np.array([-1, -1, 0, 0, 1, -1, 1, 0,
                          -1, 1, 0, 1, 1, 1, 1, 1], dtype='f4')
    vbo_quad = ctx.buffer(quad_data.tobytes())
    vao_trail_decay = ctx.vertex_array(prog_trail_decay, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])
    vao_display = ctx.vertex_array(prog_display, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])

    # ── Create simulation ──
    sim = Simulation()

    point_size = params.get('point_size', 4.0)
    trail_decay = params.get('trail_decay', 0.98)

    # Auto-select particle near center if requested but index out of range
    if sel_idx >= sim.n:
        # Pick the particle closest to center
        center = np.array([0.5 * SPACE, 0.5 * SPACE])
        dists = np.sum((sim.pos - center) ** 2, axis=1)
        sel_idx = int(np.argmin(dists))
        print(f"  selected-particle adjusted to {sel_idx} (nearest center)")

    # ── Helper: generate snapshot output path ──
    def _snapshot_path(step):
        base, ext = os.path.splitext(args.output)
        return f"{base}_step{step:06d}{ext}"

    # ── Run simulation ──
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step()

        # Upload particle data
        positions, colors = sim.get_render_data()
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        # ── Standard trail pass: decay + splat ──
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = trail_decay
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = (0.5, 0.5)
        prog_splat['view_zoom'] = 1.0
        prog_splat['point_size'] = point_size
        vao_splat.render(moderngl.POINTS)

        trail_tex, trail_tex2 = trail_tex2, trail_tex
        trail_fbo, trail_fbo2 = trail_fbo2, trail_fbo

        # ── PMem trail pass (if right_view == 8 and a particle is selected) ──
        if right_view == 8 and sel_idx >= 0 and sel_idx < sim.n:
            n_pmem_pts = 0

            if sim.nbr_ids is not None:
                nbr_row = sim.nbr_ids[sel_idx]
                if sim._valid_mask is not None:
                    valid_row = sim._valid_mask[sel_idx]
                    nbr_indices = nbr_row[valid_row]
                else:
                    nbr_indices = nbr_row
                n_pmem_pts = len(nbr_indices)

                if n_pmem_pts > 0:
                    prefs = sim.get_vis_prefs()
                    k = sim.k
                    nbr_prefs = prefs[nbr_indices]

                    pmem_pos, scale, sqrt2, sqrt6 = _pmem_isometric_project(nbr_prefs, k)
                    pmem_col = _prefs_to_rgb(nbr_prefs, k)

                    vbo_pmem_pos.orphan(n_pmem_pts * 2 * 4)
                    vbo_pmem_pos.write(pmem_pos.tobytes())
                    vbo_pmem_col.orphan(n_pmem_pts * 3 * 4)
                    vbo_pmem_col.write(pmem_col.tobytes())

            # Decay
            pmem_trail_fbo2.use()
            ctx.clear(0, 0, 0)
            ctx.blend_func = moderngl.ONE, moderngl.ZERO
            pmem_trail_tex.use(0)
            prog_trail_decay['trail_tex'] = 0
            prog_trail_decay['decay'] = trail_decay
            vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

            # Splat neighbor points
            if n_pmem_pts > 0:
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
                prog_splat['viewport_offset'] = (0.0, 0.0)
                prog_splat['viewport_scale'] = (1.0, 1.0)
                prog_splat['view_center'] = (0.5, 0.5)
                prog_splat['view_zoom'] = 1.0
                prog_splat['point_size'] = point_size
                vao_pmem_splat.render(moderngl.POINTS, vertices=n_pmem_pts)

                # White marker for the selected particle itself
                prefs = sim.get_vis_prefs()
                k = sim.k
                sel_pref = prefs[sel_idx:sel_idx+1]
                sel_proj, _, _, _ = _pmem_isometric_project(sel_pref, k)
                sel_pt_col = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
                vbo_pmem_pos.orphan(1 * 2 * 4)
                vbo_pmem_pos.write(sel_proj.tobytes())
                vbo_pmem_col.orphan(1 * 3 * 4)
                vbo_pmem_col.write(sel_pt_col.tobytes())
                prog_splat['point_size'] = point_size + 4.0
                vao_pmem_splat.render(moderngl.POINTS, vertices=1)
                prog_splat['point_size'] = point_size

            # Swap
            pmem_trail_tex, pmem_trail_tex2 = pmem_trail_tex2, pmem_trail_tex
            pmem_trail_fbo, pmem_trail_fbo2 = pmem_trail_fbo2, pmem_trail_fbo

        # ── Snapshot at requested steps ──
        if step in snapshot_steps:
            _render_frame(ctx, output_fbo, half_w, H,
                          prog_particle, vao_particle,
                          prog_display, vao_display,
                          point_size, trail_tex, pmem_trail_tex, right_view,
                          sel_idx, positions, colors)
            snap_path = _snapshot_path(step)
            _save_frame(output_fbo, W, H, snap_path, f"Step {step}")

        if step % max(1, n_steps // 10) == 0 or step == n_steps:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            print(f"  step {step}/{n_steps}  ({rate:.0f} steps/s)")

    total = time.perf_counter() - t0
    print(f"Done: {n_steps} steps in {total:.1f}s ({n_steps/total:.0f} steps/s)")

    # ── Final frame (also saved as args.output) ──
    _render_frame(ctx, output_fbo, half_w, H,
                  prog_particle, vao_particle,
                  prog_display, vao_display,
                  point_size, trail_tex, pmem_trail_tex, right_view,
                  sel_idx, positions, colors)

    raw = output_fbo.read(components=3)
    img_array = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    img_array = img_array[::-1].copy()

    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        step_text = f"Step {n_steps}"
        font_size = max(24, H // 30)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        x, y = 12, 8
        draw.text((x + 2, y + 2), step_text, fill=(0, 0, 0), font=font)
        draw.text((x, y), step_text, fill=(255, 255, 255), font=font)
        img.save(args.output)
    except ImportError:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(args.output, img_array)
        except ImportError:
            np.save(args.output.replace('.png', '.npy'), img_array)
            print(f"Neither Pillow nor matplotlib available — saved raw array")
            return

    print(f"Image saved to {args.output}")

    if args.save_data:
        data_path = args.output.replace('.png', '.npz')
        np.savez(data_path,
                 pos=sim.pos, prefs=sim.prefs, response=sim.response,
                 movement=sim._movement, memory_field=sim.memory_field,
                 memory_flow=sim.memory_flow)
        print(f"Data saved to {data_path}")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    if s in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {v!r}")


def _add_param_args(parser):
    """Add one CLI flag per key in params dict, inferring type from default."""
    from .params import params
    group = parser.add_argument_group(
        'simulation params',
        'Any key in params.py is exposed as --<key>; default shown in [brackets]. '
        'These override --config JSON.')
    for key, default in params.items():
        if isinstance(default, bool):
            t = _str2bool
            metavar = 'BOOL'
        elif isinstance(default, int):
            t = int
            metavar = 'INT'
        elif isinstance(default, float):
            t = float
            metavar = 'FLOAT'
        else:
            t = str
            metavar = 'STR'
        group.add_argument(f'--{key}', type=t, default=None, metavar=metavar,
                           help=f'[{default}]')


def main():
    parser = argparse.ArgumentParser(
        description='Headless sim_2d_exp renderer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of simulation steps')
    parser.add_argument('--output', type=str, default='sim_output.png',
                        help='Output image path')
    parser.add_argument('--width', type=int, default=1920,
                        help='Output image width')
    parser.add_argument('--height', type=int, default=960,
                        help='Output image height')
    parser.add_argument('--save-data', action='store_true',
                        help='Also save raw numpy data (.npz)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON string of param overrides (applied before individual --<param> flags)')
    parser.add_argument('--selected-particle', type=int, default=None,
                        help='Particle index to track for PMem view (default: none)')
    parser.add_argument('--right-view', type=int, default=None,
                        help='Right panel view mode (0=trail, 8=PMem; auto-detected from --selected-particle)')
    parser.add_argument('--snapshot-steps', type=str, default=None,
                        help='Comma-separated step numbers to save snapshots (e.g. 1,100,1000)')
    _add_param_args(parser)
    args = parser.parse_args()

    run_headless(args)


if __name__ == '__main__':
    main()
