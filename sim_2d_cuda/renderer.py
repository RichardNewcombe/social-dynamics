"""
GLFW / moderngl / imgui renderer and main loop for the 2D particle simulation.
"""

import os
import sys
import time
import subprocess
import ctypes
import numpy as np

# ── Prevent duplicate GLFW library loading ──
# On macOS, both pyglfw and imgui_bundle ship their own libglfw.
# On Windows/Linux this is typically not an issue but we handle it anyway.
import site
if sys.platform == 'darwin':
    for _p in [site.getusersitepackages()] + \
              (site.getsitepackages() if hasattr(site, 'getsitepackages') else []):
        _candidate = os.path.join(_p, 'imgui_bundle', 'libglfw.3.dylib')
        if os.path.isfile(_candidate):
            os.environ['PYGLFW_LIBRARY'] = _candidate
            break

import glfw
import moderngl
from OpenGL import GL as _GL
from imgui_bundle import imgui

from .params import (params, auto_scale_ref, SPACE, POS_DISTS, PREF_DISTS,
                     BEST_MODES, SOCIAL_MODES, PREF_COLOR_MODES,
                     VIS_PREF_SOURCES)
from .shaders import (
    PARTICLE_VERT, PARTICLE_FRAG, QUAD_VERT, TRAIL_FRAG,
    SPLAT_FRAG, DISPLAY_FRAG, BOX_VERT, BOX_FRAG, LINE_FRAG,
)
from .spatial import make_radius_circles, warmup_jit, periodic_dist
from .physics_numba import warmup_numba_physics
from .physics_torch import _HAS_TORCH, _TORCH_DEVICE
from .simulation import Simulation


WINDOW_W, WINDOW_H = 0, 0


def run():
    """Launch the simulation window and enter the main loop."""
    global WINDOW_W, WINDOW_H

    # ── Initialize GLFW ──
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height
    WINDOW_H = screen_h - 130
    WINDOW_W = 2 * WINDOW_H
    if WINDOW_W > screen_w - 20:
        WINDOW_W = screen_w - 20
        WINDOW_H = WINDOW_W // 2

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if sys.platform == 'darwin':
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_W, WINDOW_H,
                                "Particle Simulation — CUDA", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    fb_w, fb_h = glfw.get_framebuffer_size(window)

    ctx = moderngl.create_context(require=410)
    renderer_name = ctx.info["GL_RENDERER"]
    gl_ver = ctx.info["GL_VERSION"]
    print(f"GL: {gl_ver}  |  Renderer: {renderer_name}")
    print(f"Window: {WINDOW_W}x{WINDOW_H}  Framebuffer: {fb_w}x{fb_h}")
    if "Software" in renderer_name:
        print("WARNING: Using software renderer.")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    # ── imgui setup ──
    imgui_ctx = imgui.create_context()
    io = imgui.get_io()
    io.config_mac_osx_behaviors = (sys.platform == 'darwin')
    io.config_drag_click_to_input_text = True

    # ── View state ──
    view_center = [0.5, 0.5]
    view_zoom = 1.0
    pan_active = False
    prev_view_center = [0.5, 0.5]
    prev_view_zoom = 1.0
    prev_mouse_pos = [0.0, 0.0]

    # ── Selection state (Cmd+drag) ──
    selecting = False
    sel_start = [0.0, 0.0]
    sel_end = [0.0, 0.0]

    def screen_to_sim(sx, sy):
        hw = WINDOW_W // 2
        if sx >= hw:
            return None
        ndc_x = sx / hw * 2.0 - 1.0
        ndc_y = 1.0 - sy / WINDOW_H * 2.0
        sim_x = ndc_x / (view_zoom * 2.0) + view_center[0]
        sim_y = ndc_y / (view_zoom * 2.0) + view_center[1]
        return (sim_x, sim_y)

    def scroll_callback(win, xoffset, yoffset):
        nonlocal view_zoom
        if io.want_capture_mouse:
            return
        mx, my = glfw.get_cursor_pos(win)
        hw = WINDOW_W // 2
        ndc_x = (mx % hw) / hw * 2.0 - 1.0
        ndc_y = 1.0 - my / WINDOW_H * 2.0
        factor = 1.15 ** yoffset
        new_zoom = max(1.0, min(view_zoom * factor, 20.0))
        view_center[0] += ndc_x / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_center[1] += ndc_y / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_zoom = new_zoom

    def mouse_button_callback(win, button, action, mods):
        nonlocal pan_active, selecting
        if io.want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            cmd_held = bool(mods & glfw.MOD_SUPER)
            if action == glfw.PRESS:
                mx, my = glfw.get_cursor_pos(win)
                if cmd_held:
                    hw = WINDOW_W // 2
                    if mx < hw:
                        selecting = True
                        sel_start[0] = mx
                        sel_start[1] = my
                        sel_end[0] = mx
                        sel_end[1] = my
                else:
                    pan_active = True
                    prev_mouse_pos[0] = mx
                    prev_mouse_pos[1] = my
            elif action == glfw.RELEASE:
                if selecting:
                    selecting = False
                    c0 = screen_to_sim(sel_start[0], sel_start[1])
                    c1 = screen_to_sim(sel_end[0], sel_end[1])
                    if c0 is not None and c1 is not None:
                        sx0 = min(c0[0], c1[0])
                        sx1 = max(c0[0], c1[0])
                        sy0 = min(c0[1], c1[1])
                        sy1 = max(c0[1], c1[1])
                        vc = np.array(view_center, dtype=np.float64)
                        pos = sim.pos.astype(np.float64)
                        rel = pos - vc
                        rel -= np.round(rel)
                        abs_pos = rel + vc
                        mask_x = (abs_pos[:, 0] >= sx0) & (abs_pos[:, 0] <= sx1)
                        mask_y = (abs_pos[:, 1] >= sy0) & (abs_pos[:, 1] <= sy1)
                        matches = mask_x & mask_y
                        sim.tracked_seed[matches] = True
                        sim.tracked[matches] = True
                pan_active = False

    def cursor_pos_callback(win, mx, my):
        if pan_active and not selecting and not io.want_capture_mouse:
            dx = mx - prev_mouse_pos[0]
            dy = my - prev_mouse_pos[1]
            hw = WINDOW_W // 2
            view_center[0] -= dx / hw / view_zoom
            view_center[1] += dy / WINDOW_H / view_zoom
        if selecting:
            sel_end[0] = mx
            sel_end[1] = my
        prev_mouse_pos[0] = mx
        prev_mouse_pos[1] = my

    def key_callback(win, key, scancode, action, mods):
        nonlocal running_sim
        if io.want_capture_keyboard:
            return
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                running_sim = not running_sim
            elif key == glfw.KEY_R:
                do_reset()
            elif key == glfw.KEY_UP:
                params['step_size'] = min(params['step_size'] + 0.001, 0.05)
            elif key == glfw.KEY_DOWN:
                params['step_size'] = max(params['step_size'] - 0.001, 0.001)
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                params['social'] = min(params['social'] + 0.001, 0.01)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                params['social'] = max(params['social'] - 0.001, -0.01)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)

    imgui.backends.opengl3_init("#version 410")
    window_ptr = ctypes.cast(window, ctypes.c_void_p).value
    imgui.backends.glfw_init_for_opengl(window_ptr, True)

    # ── Compile shader programs ──
    num_particles = params['num_particles']

    prog_particle = ctx.program(vertex_shader=PARTICLE_VERT,
                                fragment_shader=PARTICLE_FRAG)
    vbo_pos = ctx.buffer(reserve=num_particles * 2 * 4)
    vbo_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_particle = ctx.vertex_array(prog_particle, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # Trail FBO setup (ping-pong pair)
    trail_w, trail_h = fb_w // 2, fb_h
    trail_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo = ctx.framebuffer(color_attachments=[trail_tex])
    trail_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo2 = ctx.framebuffer(color_attachments=[trail_tex2])

    quad_data = np.array([-1, -1, 0, 0, 1, -1, 1, 0,
                          -1, 1, 0, 1, 1, 1, 1, 1], dtype='f4')
    vbo_quad = ctx.buffer(quad_data.tobytes())
    prog_trail_decay = ctx.program(vertex_shader=QUAD_VERT,
                                   fragment_shader=TRAIL_FRAG)
    vao_trail_decay = ctx.vertex_array(prog_trail_decay, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])

    prog_display = ctx.program(vertex_shader=QUAD_VERT,
                               fragment_shader=DISPLAY_FRAG)
    vao_display = ctx.vertex_array(prog_display, [
        (vbo_quad, '2f 2f', 'in_pos', 'in_uv'),
    ])

    prog_splat = ctx.program(vertex_shader=PARTICLE_VERT,
                             fragment_shader=SPLAT_FRAG)
    vao_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    prog_box = ctx.program(vertex_shader=BOX_VERT, fragment_shader=BOX_FRAG)
    box_verts = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype='f4')
    vbo_box = ctx.buffer(box_verts.tobytes())
    vao_box = ctx.vertex_array(prog_box, [(vbo_box, '2f', 'in_pos')])

    prog_line = ctx.program(vertex_shader=BOX_VERT, fragment_shader=LINE_FRAG)
    n_max_edges = params['num_particles'] * params['n_neighbors']
    vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 2 * 4)
    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])

    # Velocity field FBO
    vel_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo = ctx.framebuffer(color_attachments=[vel_tex])
    vel_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo2 = ctx.framebuffer(color_attachments=[vel_tex2])

    vbo_vel_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_vel_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_vel_col, '3f', 'in_color'),
    ])

    # Causal tracking VBOs
    vbo_causal_pos = ctx.buffer(reserve=num_particles * 2 * 4)
    vbo_causal_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_causal_splat = ctx.vertex_array(prog_splat, [
        (vbo_causal_pos, '2f', 'in_pos'),
        (vbo_causal_col, '3f', 'in_color'),
    ])
    vao_causal_particle = ctx.vertex_array(prog_particle, [
        (vbo_causal_pos, '2f', 'in_pos'),
        (vbo_causal_col, '3f', 'in_color'),
    ])

    for tex in (trail_tex, trail_tex2, vel_tex, vel_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # Causal trail FBO
    causal_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    causal_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    causal_fbo = ctx.framebuffer(color_attachments=[causal_tex])
    causal_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    causal_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    causal_fbo2 = ctx.framebuffer(color_attachments=[causal_tex2])
    for tex in (causal_tex, causal_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # Pref-space trail FBO (ping-pong pair)
    pref_trail_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    pref_trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pref_trail_fbo = ctx.framebuffer(color_attachments=[pref_trail_tex])
    pref_trail_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    pref_trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pref_trail_fbo2 = ctx.framebuffer(color_attachments=[pref_trail_tex2])
    for tex in (pref_trail_tex, pref_trail_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # Pref-space VBOs: positions = (pref[0], pref[1]) mapped to [0,1]
    vbo_pref_pos = ctx.buffer(reserve=num_particles * 2 * 4)
    vbo_pref_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_pref_splat = ctx.vertex_array(prog_splat, [
        (vbo_pref_pos, '2f', 'in_pos'),
        (vbo_pref_col, '3f', 'in_color'),
    ])

    # ── Create simulation ──
    sim = Simulation()
    running_sim = True

    # ── Recording state ──
    rec_process = None
    rec_frame_count = 0
    rec_interval = 1.0
    rec_last_time = 0.0
    rec_filename = ""

    def start_recording():
        nonlocal rec_process, rec_frame_count, rec_last_time, rec_filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rec_filename = os.path.join(os.path.dirname(__file__) or ".",
                                    f"timelapse_{timestamp}.mp4")
        rec_frame_count = 0
        rec_last_time = time.monotonic()
        rec_process = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{fb_w}x{fb_h}",
            "-framerate", "30",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            rec_filename,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Recording started: {rec_filename}")

    def stop_recording():
        nonlocal rec_process
        if rec_process is not None:
            rec_process.stdin.close()
            rec_process.wait()
            print(f"Recording saved: {rec_filename} ({rec_frame_count} frames)")
            rec_process = None

    def capture_frame():
        nonlocal rec_frame_count
        data = _GL.glReadPixels(0, 0, fb_w, fb_h,
                                _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE)
        row_size = fb_w * 3
        flipped = bytearray(fb_h * row_size)
        for dst_row in range(fb_h):
            src_row = fb_h - 1 - dst_row
            flipped[dst_row * row_size:(dst_row + 1) * row_size] = \
                data[src_row * row_size:(src_row + 1) * row_size]
        try:
            rec_process.stdin.write(flipped)
            rec_frame_count += 1
        except BrokenPipeError:
            stop_recording()

    def rebuild_buffers():
        nonlocal vbo_pos, vbo_col, vbo_vel_col, vao_particle, vao_splat, vao_vel_splat
        nonlocal vbo_line, vao_line
        nonlocal vbo_causal_pos, vbo_causal_col, vao_causal_splat, vao_causal_particle
        nonlocal vbo_pref_pos, vbo_pref_col, vao_pref_splat
        n = params['num_particles']
        vbo_pos = ctx.buffer(reserve=n * 2 * 4)
        vbo_col = ctx.buffer(reserve=n * 3 * 4)
        vbo_vel_col = ctx.buffer(reserve=n * 3 * 4)
        vao_particle = ctx.vertex_array(prog_particle, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_vel_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '2f', 'in_pos'),
            (vbo_vel_col, '3f', 'in_color'),
        ])
        n_max_edges = n * params['n_neighbors']
        vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 2 * 4)
        vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])
        vbo_causal_pos = ctx.buffer(reserve=n * 2 * 4)
        vbo_causal_col = ctx.buffer(reserve=n * 3 * 4)
        vao_causal_splat = ctx.vertex_array(prog_splat, [
            (vbo_causal_pos, '2f', 'in_pos'),
            (vbo_causal_col, '3f', 'in_color'),
        ])
        vao_causal_particle = ctx.vertex_array(prog_particle, [
            (vbo_causal_pos, '2f', 'in_pos'),
            (vbo_causal_col, '3f', 'in_color'),
        ])
        vbo_pref_pos = ctx.buffer(reserve=n * 2 * 4)
        vbo_pref_col = ctx.buffer(reserve=n * 3 * 4)
        vao_pref_splat = ctx.vertex_array(prog_splat, [
            (vbo_pref_pos, '2f', 'in_pos'),
            (vbo_pref_col, '3f', 'in_color'),
        ])

    def do_reset():
        nonlocal running_sim
        if params['auto_scale']:
            ref = auto_scale_ref
            scale = (ref['n'] / params['num_particles']) ** 0.5
            params['step_size'] = ref['step_size'] * scale
            params['neighbor_radius'] = ref['radius'] * scale
        sim.reset()
        rebuild_buffers()
        for fbo in (trail_fbo, trail_fbo2, vel_fbo, vel_fbo2,
                    causal_fbo, causal_fbo2,
                    pref_trail_fbo, pref_trail_fbo2):
            fbo.use()
            ctx.clear(0, 0, 0)
        running_sim = True

    def reset_view():
        nonlocal view_zoom
        view_center[0] = 0.5
        view_center[1] = 0.5
        view_zoom = 1.0

    # ── JIT warmup ──
    print("Warming up numba JIT kernels...")
    warmup_nbr, warmup_val = warmup_jit()
    warmup_numba_physics()
    print("JIT warmup complete.")

    # ── FPS tracking ──
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    t_sim = 0.0
    prev_pref_view = -1    # track right_view to clear pref trail on mode switch
    prev_vis_source = -1   # track vis_pref_source to clear trails on switch

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── Clear trails if vis source changed ──
        cur_vis_source = params['vis_pref_source']
        if cur_vis_source != prev_vis_source and prev_vis_source >= 0:
            for fbo in (trail_fbo, trail_fbo2, vel_fbo, vel_fbo2,
                        causal_fbo, causal_fbo2,
                        pref_trail_fbo, pref_trail_fbo2):
                fbo.use()
                ctx.clear(0, 0, 0)
        prev_vis_source = cur_vis_source

        # ── Physics step ──
        t0 = time.perf_counter()
        if running_sim:
            spf = params['steps_per_frame']
            reuse = params['reuse_neighbors']
            for sub in range(spf):
                sim.step(reuse_neighbors=(reuse and sub > 0))
        t_sim = time.perf_counter() - t0

        # ── Upload particle data to GPU ──
        positions, colors = sim.get_render_data()
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        vel_colors = sim.get_velocity_colors()
        vbo_vel_col.write(vel_colors.tobytes())

        # ── Trail rendering pass ──
        trail_zoom_on = params['trail_zoom']
        view_changed = (view_center[0] != prev_view_center[0] or
                        view_center[1] != prev_view_center[1] or
                        view_zoom != prev_view_zoom)
        if trail_zoom_on and view_changed:
            for fbo in (trail_fbo, trail_fbo2, vel_fbo, vel_fbo2,
                        causal_fbo, causal_fbo2,
                        pref_trail_fbo, pref_trail_fbo2):
                fbo.use()
                ctx.clear(0, 0, 0)
        prev_view_center[0] = view_center[0]
        prev_view_center[1] = view_center[1]
        prev_view_zoom = view_zoom

        if trail_zoom_on:
            splat_center = tuple(view_center)
            splat_zoom = view_zoom
        else:
            splat_center = (0.5, 0.5)
            splat_zoom = 1.0

        # Pass 1: Decay
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: Splat
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = splat_center
        prog_splat['view_zoom'] = splat_zoom
        prog_splat['point_size'] = params['point_size']
        vao_splat.render(moderngl.POINTS)

        # Pass 3: Swap
        trail_tex, trail_tex2 = trail_tex2, trail_tex
        trail_fbo, trail_fbo2 = trail_fbo2, trail_fbo

        # ── Velocity field pass ──
        vel_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        vel_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = splat_center
        prog_splat['view_zoom'] = splat_zoom
        prog_splat['point_size'] = params['point_size']
        vao_vel_splat.render(moderngl.POINTS)

        vel_tex, vel_tex2 = vel_tex2, vel_tex
        vel_fbo, vel_fbo2 = vel_fbo2, vel_fbo

        # ── Causal trail pass ──
        if sim.tracked.any():
            tracked_mask = sim.tracked
            n_tracked = tracked_mask.sum()

            tracked_pos = positions[tracked_mask]
            tracked_col = colors[tracked_mask]
            vbo_causal_pos.orphan(n_tracked * 2 * 4)
            vbo_causal_pos.write(tracked_pos.tobytes())
            vbo_causal_col.orphan(n_tracked * 3 * 4)
            vbo_causal_col.write(tracked_col.tobytes())

            causal_fbo2.use()
            ctx.clear(0, 0, 0)
            ctx.blend_func = moderngl.ONE, moderngl.ZERO
            causal_tex.use(0)
            prog_trail_decay['trail_tex'] = 0
            prog_trail_decay['decay'] = params['trail_decay']
            vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            prog_splat['viewport_offset'] = (0.0, 0.0)
            prog_splat['viewport_scale'] = (1.0, 1.0)
            prog_splat['view_center'] = splat_center
            prog_splat['view_zoom'] = splat_zoom
            prog_splat['point_size'] = params['point_size']
            vao_causal_splat.render(moderngl.POINTS, vertices=n_tracked)

            causal_tex, causal_tex2 = causal_tex2, causal_tex
            causal_fbo, causal_fbo2 = causal_fbo2, causal_fbo

        # ── Pref-space trail pass ──
        cur_pref_view = params['right_view']
        if cur_pref_view in (3, 4) and cur_pref_view != prev_pref_view:
            # Mode switched — clear pref trail so old spatial data doesn't linger
            for fbo in (pref_trail_fbo, pref_trail_fbo2):
                fbo.use()
                ctx.clear(0, 0, 0)
        prev_pref_view = cur_pref_view

        prefs = sim.get_vis_prefs()
        k = sim.k
        pref_pos = np.zeros((sim.n, 2), dtype=np.float32)

        if cur_pref_view == 4:
            # Isometric projection of 3D pref cube viewed from (1,1,1) corner.
            # Axes project 120° apart. Input dims in [-1, 1].
            d0 = prefs[:, 0] if k > 0 else np.zeros(sim.n, dtype=np.float32)
            d1 = prefs[:, 1] if k > 1 else np.zeros(sim.n, dtype=np.float32)
            d2 = prefs[:, 2] if k > 2 else np.zeros(sim.n, dtype=np.float32)
            # proj_x = (d0 - d2) / sqrt(2),  range [-sqrt(2), sqrt(2)]
            # proj_y = (2*d1 - d0 - d2) / sqrt(6),  range [-4/sqrt(6), 4/sqrt(6)]
            sqrt2 = np.float32(np.sqrt(2))
            sqrt6 = np.float32(np.sqrt(6))
            px = (d0 - d2) / sqrt2           # range [-sqrt2, sqrt2]
            py = (2.0 * d1 - d0 - d2) / sqrt6  # range [-4/sqrt6, 4/sqrt6]
            # Uniform scale: fit both axes with margin, preserving aspect ratio
            max_range = 4.0 / sqrt6           # ~3.27, the larger range
            scale = 0.45 / max_range          # maps max_range to 0.45 (margin 0.05)
            pref_pos[:, 0] = px * scale + 0.5
            pref_pos[:, 1] = py * scale + 0.5
        else:
            # 2D scatter: (dim0, dim1) mapped to [0,1]
            pref_pos[:, 0] = (prefs[:, 0] + 1.0) * 0.5 if k > 0 else 0.5
            pref_pos[:, 1] = (prefs[:, 1] + 1.0) * 0.5 if k > 1 else 0.5

        vbo_pref_pos.write(pref_pos.tobytes())

        # Colors depend on pref_color_mode
        pcm = params['pref_color_mode']
        if pcm == 1:
            # Dim2 Heat: blue(-1) → white(0) → red(+1)
            d2 = prefs[:, 2] if k > 2 else np.zeros(sim.n, dtype=np.float32)
            pref_col = np.zeros((sim.n, 3), dtype=np.float32)
            pos_mask = d2 >= 0
            neg_mask = ~pos_mask
            # Red channel: ramp up for positive d2
            pref_col[pos_mask, 0] = 1.0
            pref_col[pos_mask, 1] = 1.0 - d2[pos_mask]
            pref_col[pos_mask, 2] = 1.0 - d2[pos_mask]
            # Blue channel: ramp up for negative d2
            pref_col[neg_mask, 0] = 1.0 + d2[neg_mask]
            pref_col[neg_mask, 1] = 1.0 + d2[neg_mask]
            pref_col[neg_mask, 2] = 1.0
        elif pcm == 2:
            # HSV: dim0 → H, dim1 → S (0.3–1.0), dim2 → V (0.3–1.0)
            h = (prefs[:, 0] + 1.0) * 0.5 if k > 0 else np.full(sim.n, 0.5)
            s = (prefs[:, 1] + 1.0) * 0.35 + 0.3 if k > 1 else np.full(sim.n, 0.9)
            v = (prefs[:, 2] + 1.0) * 0.35 + 0.3 if k > 2 else np.full(sim.n, 0.9)
            s = np.clip(s, 0.0, 1.0).astype(np.float32)
            v = np.clip(v, 0.0, 1.0).astype(np.float32)
            # HSV to RGB
            h6 = h * 6.0
            sector = h6.astype(np.int32) % 6
            f = (h6 - np.floor(h6)).astype(np.float32)
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            pref_col = np.zeros((sim.n, 3), dtype=np.float32)
            for si, (r, g, b) in enumerate([(v,t,p),(q,v,p),(p,v,t),
                                             (p,q,v),(t,p,v),(v,p,q)]):
                m = sector == si
                pref_col[m, 0] = r[m]; pref_col[m, 1] = g[m]; pref_col[m, 2] = b[m]
        else:
            # RGB: standard pref-to-rgb (same as left panel)
            pref_col = np.clip((prefs[:, :3] + 1.0) * 0.5, 0, 1).astype(np.float32)
            if k < 3:
                c = np.full((sim.n, 3), 0.5, np.float32)
                c[:, :min(k, 3)] = pref_col[:, :min(k, 3)]
                pref_col = c
        vbo_pref_col.write(pref_col.tobytes())

        # Decay
        pref_trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        pref_trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        # Splat (pref-space positions, pref colors, centered at 0.5,0.5, no zoom)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['view_center'] = (0.5, 0.5)
        prog_splat['view_zoom'] = 1.0
        prog_splat['point_size'] = params['point_size']
        vao_pref_splat.render(moderngl.POINTS)

        # Swap
        pref_trail_tex, pref_trail_tex2 = pref_trail_tex2, pref_trail_tex
        pref_trail_fbo, pref_trail_fbo2 = pref_trail_fbo2, pref_trail_fbo

        # ── Screen rendering ──
        ctx.screen.use()
        ctx.clear(0, 0, 0)

        # Left half: live particles
        ctx.viewport = (0, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if params['show_neighbors'] and sim.nbr_ids is not None:
            lines = sim.get_neighbor_lines()
            n_line_verts = len(lines)
            if n_line_verts > 0:
                needed = n_line_verts * 2 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '2f', 'in_pos')])
                vbo_line.write(lines.tobytes())
                prog_line['view_center'] = tuple(view_center)
                prog_line['view_zoom'] = view_zoom
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.08)
                vao_line.render(moderngl.LINES, vertices=n_line_verts)

        if params['show_radius']:
            circles = make_radius_circles(
                sim.pos.astype(np.float32), params['neighbor_radius'])
            n_circle_verts = len(circles)
            if n_circle_verts > 0:
                needed = n_circle_verts * 2 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line,
                                                [(vbo_line, '2f', 'in_pos')])
                vbo_line.write(circles.tobytes())
                prog_line['view_center'] = tuple(view_center)
                prog_line['view_zoom'] = view_zoom
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.04)
                vao_line.render(moderngl.LINES, vertices=n_circle_verts)

        prog_particle['viewport_offset'] = (0.0, 0.0)
        prog_particle['viewport_scale'] = (1.0, 1.0)
        prog_particle['view_center'] = tuple(view_center)
        prog_particle['view_zoom'] = view_zoom
        prog_particle['point_size'] = params['point_size']
        vao_particle.render(moderngl.POINTS)

        # Highlight tracked particles
        if sim.tracked.any():
            n_tracked = sim.tracked.sum()
            highlight_pos = positions[sim.tracked]
            highlight_col = np.full((n_tracked, 3), 1.0, dtype=np.float32)
            vbo_causal_pos.orphan(n_tracked * 2 * 4)
            vbo_causal_pos.write(highlight_pos.tobytes())
            vbo_causal_col.orphan(n_tracked * 3 * 4)
            vbo_causal_col.write(highlight_col.tobytes())
            prog_particle['viewport_offset'] = (0.0, 0.0)
            prog_particle['viewport_scale'] = (1.0, 1.0)
            prog_particle['point_size'] = params['point_size'] + 3.0
            vao_causal_particle.render(moderngl.POINTS, vertices=n_tracked)
            prog_particle['point_size'] = params['point_size']

        # Right half: display selected view
        ctx.viewport = (fb_w // 2, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        rv = params['right_view']
        right_tex = pref_trail_tex if rv in (3, 4) else \
                    vel_tex if rv == 1 else \
                    causal_tex if rv == 2 else trail_tex
        right_tex.use(0)
        prog_display['tex'] = 0
        if rv in (3, 4):
            # Pref views: no pan/zoom, always centered
            prog_display['view_center'] = (0.5, 0.5)
            prog_display['view_zoom'] = 1.0
        elif trail_zoom_on:
            prog_display['view_center'] = (0.5, 0.5)
            prog_display['view_zoom'] = 1.0
        else:
            prog_display['view_center'] = tuple(view_center)
            prog_display['view_zoom'] = view_zoom
        vao_display.render(moderngl.TRIANGLE_STRIP)

        # ── Pref-space axis lines ──
        if rv in (3, 4):
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            sqrt2 = np.float32(np.sqrt(2))
            sqrt6 = np.float32(np.sqrt(6))

            def _iso_project(d0, d1, d2):
                """Project 3D pref coords to [0,1]^2 (isometric)."""
                px = (d0 - d2) / sqrt2
                py = (2.0 * d1 - d0 - d2) / sqrt6
                max_range = 4.0 / sqrt6
                scale = 0.45 / max_range
                return px * scale + 0.5, py * scale + 0.5

            if rv == 4:
                # Isometric axes: origin → each axis tip, colored R/G/B
                ox, oy = _iso_project(0, 0, 0)
                # Axis tips at ±1 along each dim
                axes_data = []
                axis_colors = [
                    (1.0, 0.3, 0.3, 0.7),  # d0 = red
                    (0.3, 1.0, 0.3, 0.7),  # d1 = green
                    (0.3, 0.3, 1.0, 0.7),  # d2 = blue
                ]
                axis_tips = [
                    [( 1, 0, 0), (-1, 0, 0)],  # d0
                    [( 0, 1, 0), ( 0,-1, 0)],  # d1
                    [( 0, 0, 1), ( 0, 0,-1)],  # d2
                ]
                for dim_idx, (color, tips) in enumerate(zip(axis_colors, axis_tips)):
                    for tip in tips:
                        tx, ty = _iso_project(*tip)
                        line_data = np.array([ox, oy, tx, ty], dtype='f4')
                        vbo_line.write(line_data.tobytes())
                        prog_line['view_center'] = (0.5, 0.5)
                        prog_line['view_zoom'] = 1.0
                        prog_line['line_color'] = color
                        vao_line.render(moderngl.LINES, vertices=2)
            else:
                # 2D axes: crosshair at center
                prog_line['view_center'] = (0.5, 0.5)
                prog_line['view_zoom'] = 1.0
                # d0 axis (horizontal, red)
                line_data = np.array([0.0, 0.5, 1.0, 0.5], dtype='f4')
                vbo_line.write(line_data.tobytes())
                prog_line['line_color'] = (1.0, 0.3, 0.3, 0.5)
                vao_line.render(moderngl.LINES, vertices=2)
                # d1 axis (vertical, green)
                line_data = np.array([0.5, 0.0, 0.5, 1.0], dtype='f4')
                vbo_line.write(line_data.tobytes())
                prog_line['line_color'] = (0.3, 1.0, 0.3, 0.5)
                vao_line.render(moderngl.LINES, vertices=2)

        if params['show_box']:
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            if trail_zoom_on:
                prog_box['view_center'] = (0.5, 0.5)
                prog_box['view_zoom'] = 1.0
            else:
                prog_box['view_center'] = tuple(view_center)
                prog_box['view_zoom'] = view_zoom
            vao_box.render(moderngl.LINE_LOOP)

        # Divider line
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        divider = np.array([0.5, 0.0, 0.5, 1.0], dtype='f4')
        vbo_line.write(divider.tobytes())
        prog_line['view_center'] = (0.5, 0.5)
        prog_line['view_zoom'] = 1.0
        prog_line['line_color'] = (1.0, 1.0, 1.0, 0.5)
        vao_line.render(moderngl.LINES, vertices=2)

        # ── imgui overlay ──
        ctx.viewport = (0, 0, fb_w, fb_h)
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        status = "Running" if running_sim else "Paused"

        imgui.set_next_window_pos(imgui.ImVec2(10, 10),
                                  imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(280, 400),
                                   imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_bg_alpha(0.8)
        imgui.begin("Controls")

        # Status
        imgui.text(f"Status: {status}")
        imgui.text(f"Step: {sim.step_count}  FPS: {fps:.0f}")
        imgui.text(f"Sim: {t_sim*1000:.1f}ms  "
                   f"grid: {sim._t_build*1000:.1f}ms  "
                   f"query: {sim._t_query*1000:.1f}ms  "
                   f"physics: {sim._t_physics*1000:.1f}ms")
        imgui.text(f"Neighbors/particle: {sim._n_nbrs}")
        imgui.separator()

        # Pause / Reset / Step / Record
        label = "Resume" if not running_sim else "Pause"
        if imgui.button(label, imgui.ImVec2(80, 0)):
            running_sim = not running_sim
        imgui.same_line()
        if imgui.button("Reset", imgui.ImVec2(80, 0)):
            do_reset()
        imgui.same_line()
        if imgui.button("Step", imgui.ImVec2(50, 0)):
            running_sim = False
            sim.step()
        if rec_process is not None:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1.0, 0.2, 0.2, 1.0), "REC")
            imgui.text(f"Frames: {rec_frame_count}  "
                       f"Interval: {rec_interval:.1f}s")
            if imgui.button("Stop Rec", imgui.ImVec2(80, 0)):
                stop_recording()
        else:
            changed, rec_interval = imgui.drag_float(
                "Rec Interval", rec_interval, 0.05, 0.1, 10.0, "%.1fs")
            if imgui.button("Record", imgui.ImVec2(80, 0)):
                start_recording()
        imgui.separator()

        # Right panel view selector
        if imgui.radio_button("Trails", params['right_view'] == 0):
            params['right_view'] = 0
        imgui.same_line()
        if imgui.radio_button("Velocity", params['right_view'] == 1):
            params['right_view'] = 1
        imgui.same_line()
        if imgui.radio_button("Causal", params['right_view'] == 2):
            params['right_view'] = 2
        imgui.same_line()
        if imgui.radio_button("Pref2D", params['right_view'] == 3):
            params['right_view'] = 3
        imgui.same_line()
        if imgui.radio_button("Pref3D", params['right_view'] == 4):
            params['right_view'] = 4
        if params['right_view'] in (3, 4):
            changed, v = imgui.combo("Pref Colors", params['pref_color_mode'],
                                     PREF_COLOR_MODES)
            if changed:
                params['pref_color_mode'] = v

        changed, v = imgui.checkbox("Show Neighbors", params['show_neighbors'])
        if changed:
            params['show_neighbors'] = v
        imgui.same_line()
        changed, v = imgui.checkbox("Show Radius", params['show_radius'])
        if changed:
            params['show_radius'] = v

        changed, v = imgui.checkbox("Box", params['show_box'])
        if changed:
            params['show_box'] = v
        imgui.same_line()
        if imgui.button("Reset View"):
            reset_view()
        if view_zoom != 1.0:
            imgui.text(f"Zoom: {view_zoom:.1f}x")
        imgui.separator()

        # Tracking controls
        imgui.text(f"Tracked: {sim.tracked.sum()} / {sim.n}")
        _track_modes = ["Frozen", "+ Neighbors", "Causal Spread"]
        changed, v = imgui.combo("Track Mode", params['track_mode'], _track_modes)
        if changed:
            params['track_mode'] = v
        if imgui.button("Clear Tracking"):
            sim.tracked[:] = False
            sim.tracked_seed[:] = False
            causal_fbo.use()
            ctx.clear(0, 0, 0)
            causal_fbo2.use()
            ctx.clear(0, 0, 0)
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0),
                           "Cmd+drag to select")
        imgui.separator()

        # Live parameters
        if imgui.collapsing_header(
                "Live Parameters",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            changed, v = imgui.drag_float("Step Size", params['step_size'], 0.0001, 0.001, 0.05, "%.4f")
            if changed:
                params['step_size'] = v
            changed, v = imgui.drag_int("Steps/Frame", params['steps_per_frame'], 0.5, 1, 100)
            if changed:
                params['steps_per_frame'] = v
            if params['steps_per_frame'] > 1:
                changed, v = imgui.checkbox("Reuse Neighbors", params['reuse_neighbors'])
                if changed:
                    params['reuse_neighbors'] = v
            changed, v = imgui.drag_float("Repulsion", params['repulsion'], 0.0001, 0.0, 0.02, "%.4f")
            if changed:
                params['repulsion'] = v
            changed, v = imgui.drag_float("Dir Memory", params['dir_memory'], 0.005, 0.0, 0.99, "%.3f")
            if changed:
                params['dir_memory'] = v
            changed, v = imgui.drag_float("Social", params['social'], 0.0001, -0.01, 0.01, "%.4f")
            if changed:
                params['social'] = v
            imgui.same_line()
            if imgui.button("0##social", imgui.ImVec2(20, 0)):
                params['social'] = 0.0
            changed, v = imgui.combo("Social Mode", params['social_mode'], SOCIAL_MODES)
            if changed:
                params['social_mode'] = v
            changed, v = imgui.checkbox("Dist-Weighted", params['social_dist_weight'])
            if changed:
                params['social_dist_weight'] = v
            changed, v = imgui.checkbox("Pref-Weighted Dir", params['pref_weighted_dir'])
            if changed:
                params['pref_weighted_dir'] = v
            changed, v = imgui.checkbox("Inner Prod Weight", params['pref_inner_prod'])
            if changed:
                params['pref_inner_prod'] = v
            changed, v = imgui.checkbox("Inner Prod Avg", params['inner_prod_avg'])
            if changed:
                params['inner_prod_avg'] = v
            changed, v = imgui.checkbox("Dist-Weighted Pref", params['pref_dist_weight'])
            if changed:
                params['pref_dist_weight'] = v
            changed, v = imgui.combo("Best Neighbor", params['best_mode'], BEST_MODES)
            if changed:
                params['best_mode'] = v
            changed, v = imgui.checkbox("Unit Prefs", params['unit_prefs'])
            if changed:
                params['unit_prefs'] = v
            imgui.separator()
            imgui.text("Signal / Response")
            changed, v = imgui.checkbox("Split Signal+Response",
                                        params['use_signal_response'])
            if changed:
                params['use_signal_response'] = v
            if params['use_signal_response']:
                changed, v = imgui.combo("Visualize", params['vis_pref_source'],
                                         VIS_PREF_SOURCES)
                if changed:
                    params['vis_pref_source'] = v
            imgui.separator()
            imgui.text("Crossover")
            changed, v = imgui.checkbox("Enable Crossover", params['crossover'])
            if changed:
                params['crossover'] = v
            if params['crossover']:
                changed, v = imgui.drag_int("Keep %", params['crossover_pct'], 1.0, 0, 100)
                if changed:
                    params['crossover_pct'] = v
                changed, v = imgui.drag_int("Interval", params['crossover_interval'], 0.5, 1, 1000)
                if changed:
                    params['crossover_interval'] = v
            changed, v = imgui.drag_float("Trail Decay", params['trail_decay'], 0.005, 0.8, 1.0, "%.3f")
            if changed:
                params['trail_decay'] = v
            changed, v = imgui.drag_float("Point Size", params['point_size'], 0.1, 1.0, 20.0, "%.1f")
            if changed:
                params['point_size'] = v
            _nbr_modes = ["KNN", "KNN + Radius", "Radius Only"]
            changed, v = imgui.combo("Neighbor Mode", params['neighbor_mode'], _nbr_modes)
            if changed:
                params['neighbor_mode'] = v
            _knn_methods = ["Hash Grid", "cKDTree (f64)", "cKDTree (f32)"]
            changed, v = imgui.combo("KNN Method", params['knn_method'], _knn_methods)
            if changed:
                params['knn_method'] = v
            _physics_engines = ["Numba", "NumPy (original)", "PyTorch"]
            changed, v = imgui.combo("Physics", params['physics_engine'], _physics_engines)
            if changed:
                params['physics_engine'] = v
            if params['physics_engine'] == 2:
                _precisions = ["f16", "bf16", "f32", "f64"]
                changed, v = imgui.combo("Precision", params['torch_precision'], _precisions)
                if changed:
                    params['torch_precision'] = v
                _devices = ["Auto (%s)" % _TORCH_DEVICE, "CPU"]
                changed, v = imgui.combo("Device", params['torch_device'], _devices)
                if changed:
                    params['torch_device'] = v
                if not _HAS_TORCH:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), "torch not installed!")
            changed, v = imgui.checkbox("Use f64 pos", params['use_f64'])
            if changed:
                params['use_f64'] = v
                imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0), "(reset to apply)")
            if params['knn_method'] == 0:
                changed, v = imgui.checkbox("Debug KNN", params['debug_knn'])
                if changed:
                    params['debug_knn'] = v
            if params['neighbor_mode'] < 2:
                changed, v = imgui.drag_int("Neighbors", params['n_neighbors'], 0.5, 1, 30)
                if changed:
                    params['n_neighbors'] = v
            changed, v = imgui.drag_float("Radius", params['neighbor_radius'], 0.001, 0.001, 0.3, "%.4f")
            if changed:
                params['neighbor_radius'] = v

        # Reset-required parameters
        if imgui.collapsing_header("Reset-Required Params"):
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                               "(changes apply on Reset)")
            changed, v = imgui.combo("Pos Init", params['pos_dist'], POS_DISTS)
            if changed:
                params['pos_dist'] = v
            if params['pos_dist'] == 1:
                changed, v = imgui.drag_float("Gauss Sigma", params['gauss_sigma'], 0.005, 0.01, 1.0, "%.3f")
                if changed:
                    params['gauss_sigma'] = v
            changed, v = imgui.combo("Pref Init", params['pref_dist'], PREF_DISTS)
            if changed:
                params['pref_dist'] = v
            if params['pref_dist'] == 4:  # Binary d0 + noise
                changed, v = imgui.drag_float("Noise eps", params['binary_noise_eps'],
                                              0.005, 0.0, 1.0, "%.3f")
                if changed:
                    params['binary_noise_eps'] = v
            changed, v = imgui.drag_int("Particles", params['num_particles'], 5.0, 2, 200000)
            if changed:
                params['num_particles'] = v
            changed, v = imgui.checkbox("Auto-scale", params['auto_scale'])
            if changed:
                params['auto_scale'] = v
            if params['auto_scale']:
                ref = auto_scale_ref
                scale = (ref['n'] / params['num_particles']) ** 0.5
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.9, 0.6, 1.0),
                    f"  step={ref['step_size']*scale:.4f}  "
                    f"radius={ref['radius']*scale:.4f}")
                imgui.text_colored(
                    imgui.ImVec4(0.5, 0.5, 0.5, 1.0),
                    f"  ref: N={ref['n']} step={ref['step_size']:.4f} "
                    f"r={ref['radius']:.3f}")
                if imgui.button("Set Reference", imgui.ImVec2(120, 0)):
                    auto_scale_ref['n'] = params['num_particles']
                    auto_scale_ref['step_size'] = params['step_size']
                    auto_scale_ref['radius'] = params['neighbor_radius']
            changed, v = imgui.drag_int("K (dims)", params['k'], 0.5, 1, 100)
            if changed:
                params['k'] = v

        if imgui.collapsing_header("Seed"):
            changed, v = imgui.checkbox("Fixed Seed", params['use_seed'])
            if changed:
                params['use_seed'] = v
            if params['use_seed']:
                changed, v = imgui.input_int("Seed##value", params['seed'])
                if changed:
                    params['seed'] = v

        imgui.end()

        # Selection rectangle overlay
        if selecting:
            draw_list = imgui.get_background_draw_list()
            draw_list.add_rect(
                imgui.ImVec2(sel_start[0], sel_start[1]),
                imgui.ImVec2(sel_end[0], sel_end[1]),
                imgui.get_color_u32(imgui.ImVec4(1, 1, 0, 0.8)),
                thickness=2.0)

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        # Timelapse capture
        if rec_process is not None:
            now = time.monotonic()
            if now - rec_last_time >= rec_interval:
                rec_last_time = now
                capture_frame()

        # Swap and FPS
        glfw.swap_buffers(window)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        glfw.set_window_title(window,
            f"Particles [{status}] Step:{sim.step_count} FPS:{fps:.0f}")

    # ── Cleanup ──
    stop_recording()
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context(imgui_ctx)
    glfw.destroy_window(window)
    glfw.terminate()
