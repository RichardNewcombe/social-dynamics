#!/usr/bin/env python3
"""
Headless renderer for sim_2d_exp — no display required.

Uses moderngl standalone context for offscreen OpenGL rendering,
producing pixel-identical output to the windowed version including
trail accumulation on the right panel.

Usage:
    python -m sim_2d_exp.headless                    # defaults
    python -m sim_2d_exp.headless --steps 500 --particles 2000
    python -m sim_2d_exp.headless --steps 1000 --output result.png
    python -m sim_2d_exp.headless --config '{"social": -0.003, "memory_field": true}'

Requires: numpy, scipy, numba, moderngl, Pillow (or matplotlib for fallback)
Does NOT require: glfw, imgui, PyOpenGL, a display server
"""

import argparse
import json
import time
import numpy as np


def run_headless(args):
    import moderngl
    from .params import params, SPACE
    from .shaders import (
        PARTICLE_VERT, PARTICLE_FRAG, QUAD_VERT, TRAIL_FRAG,
        SPLAT_FRAG, DISPLAY_FRAG,
    )
    from .simulation import Simulation

    # Apply config overrides
    if args.config:
        params.update(json.loads(args.config))
    if args.particles is not None:
        params['num_particles'] = args.particles
    if args.k is not None:
        params['k'] = args.k

    n_steps = args.steps
    N = params['num_particles']
    K = params['k']
    W, H = args.width, args.height
    half_w = W // 2

    print(f"Headless run: N={N}, K={K}, steps={n_steps}, size={W}x{H}")
    print(f"Engine: {params['physics_engine']}, step_size={params['step_size']}")

    # ── Create standalone OpenGL context (no display) ──
    ctx = moderngl.create_standalone_context(require=330)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    # ── Output FBO (what we'll read pixels from) ──
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

    # ── Trail FBOs (ping-pong) ──
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

    # ── Run simulation with trail accumulation ──
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step()

        # Upload particle data
        positions, colors = sim.get_render_data()
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        # ── Trail pass: decay + splat ──
        # Pass 1: decay
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = trail_decay
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: splat particles
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = (0.5, 0.5)
        prog_splat['view_zoom'] = 1.0
        prog_splat['point_size'] = point_size
        vao_splat.render(moderngl.POINTS)

        # Swap
        trail_tex, trail_tex2 = trail_tex2, trail_tex
        trail_fbo, trail_fbo2 = trail_fbo2, trail_fbo

        if step % max(1, n_steps // 10) == 0 or step == n_steps:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            print(f"  step {step}/{n_steps}  ({rate:.0f} steps/s)")

    total = time.perf_counter() - t0
    print(f"Done: {n_steps} steps in {total:.1f}s ({n_steps/total:.0f} steps/s)")

    # ── Render final frame to output FBO ──
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

    # Right half: trail texture
    ctx.viewport = (half_w, 0, half_w, H)
    ctx.blend_func = moderngl.ONE, moderngl.ZERO
    trail_tex.use(0)
    prog_display['tex'] = 0
    prog_display['view_center'] = (0.5, 0.5)
    prog_display['view_zoom'] = 1.0
    vao_display.render(moderngl.TRIANGLE_STRIP)

    # ── Read pixels and save ──
    raw = output_fbo.read(components=3)
    img_array = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
    # Flip vertically (OpenGL origin is bottom-left)
    img_array = img_array[::-1].copy()

    # Save using Pillow or matplotlib
    try:
        from PIL import Image
        img = Image.fromarray(img_array)
        img.save(args.output)
    except ImportError:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(args.output, img_array)
        except ImportError:
            # Last resort: raw numpy
            np.save(args.output.replace('.png', '.npy'), img_array)
            print(f"Neither Pillow nor matplotlib available — saved raw array")
            return

    print(f"Image saved to {args.output}")

    # Save raw data if requested
    if args.save_data:
        data_path = args.output.replace('.png', '.npz')
        np.savez(data_path,
                 pos=sim.pos, prefs=sim.prefs, response=sim.response,
                 movement=sim._movement, memory_field=sim.memory_field,
                 memory_flow=sim.memory_flow)
        print(f"Data saved to {data_path}")


def main():
    parser = argparse.ArgumentParser(description='Headless sim_2d_exp renderer')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of simulation steps')
    parser.add_argument('--particles', type=int, default=None,
                        help='Override number of particles')
    parser.add_argument('--k', type=int, default=None,
                        help='Override preference dimensions')
    parser.add_argument('--output', type=str, default='sim_output.png',
                        help='Output image path')
    parser.add_argument('--width', type=int, default=1920,
                        help='Output image width')
    parser.add_argument('--height', type=int, default=960,
                        help='Output image height')
    parser.add_argument('--save-data', action='store_true',
                        help='Also save raw numpy data (.npz)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON string of param overrides')
    args = parser.parse_args()

    run_headless(args)


if __name__ == '__main__':
    main()
