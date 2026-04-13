#!/usr/bin/env python3
"""
3D Preference-Directed Particle Simulation
===========================================

3D version of sim_gpu_compute.py with orbit camera, MVP projection,
wireframe cube, and all physics/neighbor features preserved.

Controls:
  Left drag   — orbit camera
  Scroll      — zoom in/out
  Space       — pause / resume
  r           — reset simulation and trails
  q / Esc     — quit
  imgui panel — full slider control for all parameters
"""

# ── Imports ──────────────────────────────────────────────────────────
import math
import numpy as np
import os, site
import time
import subprocess
import ctypes

# Prevent duplicate GLFW library loading
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

from .shaders3d import (
    PARTICLE_VERT, PARTICLE_FRAG,
    QUAD_VERT, TRAIL_FRAG, SPLAT_FRAG, DISPLAY_FRAG,
    BOX_VERT, BOX_FRAG,
    LINE_VERT, LINE_FRAG,
    OVERLAY_VERT, OVERLAY_FRAG,
    MESH_VERT, MESH_FRAG,
)
from .mountain_mesh import (
    generate_mountain_mesh, generate_cost_colors,
    project_particles_to_surface,
)

# ── Landscape imports (optional — only needed for mountain mode) ──
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from experiments.landscape import (
        make_default_landscape, make_default_cost_landscape,
    )
    _HAS_LANDSCAPE = True
except ImportError:
    _HAS_LANDSCAPE = False
from .camera3d import OrbitCamera
from .simulation3d import (
    Simulation, params, auto_scale_ref, INIT_PRESETS,
    _HAS_TORCH, _TORCH_DEVICE, make_radius_circles_3d,
)
from .grid3d import (
    SPACE, grid_build, _count_radius, _query_radius, _query_knn,
)
from .physics3d import _step_inner_prod_avg, _step_per_dim


WINDOW_W, WINDOW_H = 0, 0


def main():
    """Main entry point.  Use ``--mountain`` to start with the fitness
    landscape visible and particles constrained to the surface."""
    # ── Initialize GLFW ──
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    global WINDOW_W, WINDOW_H
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
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WINDOW_W, WINDOW_H,
                                "3D Particle Simulation", None, None)
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

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    # ── imgui setup ──
    imgui_ctx = imgui.create_context()
    io = imgui.get_io()
    io.config_mac_osx_behaviors = True
    io.config_drag_click_to_input_text = True

    # ── Camera ──
    camera = OrbitCamera()

    def scroll_callback(win, xoffset, yoffset):
        if io.want_capture_mouse:
            return
        camera.on_scroll(yoffset)

    def mouse_button_callback(win, button, action, mods):
        if io.want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            mx, my = glfw.get_cursor_pos(win)
            camera.on_mouse_button(0, 1 if action == glfw.PRESS else 0, mx, my)

    def cursor_pos_callback(win, mx, my):
        if not io.want_capture_mouse:
            camera.on_cursor_pos(mx, my, WINDOW_W, WINDOW_H)

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
                params['social'] = min(params['social'] + 0.005, 0.1)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                params['social'] = max(params['social'] - 0.005, 0.0)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)

    imgui.backends.opengl3_init("#version 410")
    window_ptr = ctypes.cast(window, ctypes.c_void_p).value
    imgui.backends.glfw_init_for_opengl(window_ptr, True)

    # ── Compile shader programs ──
    prog_particle = ctx.program(vertex_shader=PARTICLE_VERT,
                                fragment_shader=PARTICLE_FRAG)

    num_particles = params['num_particles']

    # 3D positions: n * 3 * 4 bytes
    vbo_pos = ctx.buffer(reserve=num_particles * 3 * 4)
    vbo_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_particle = ctx.vertex_array(prog_particle, [
        (vbo_pos, '3f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # ── Trail FBO setup ──
    trail_w, trail_h = fb_w // 2, fb_h
    trail_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo = ctx.framebuffer(color_attachments=[trail_tex])

    trail_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    trail_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    trail_fbo2 = ctx.framebuffer(color_attachments=[trail_tex2])

    quad_data = np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    ], dtype='f4')
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

    # Splat shader uses same PARTICLE_VERT (3D MVP) for trail accumulation
    prog_splat = ctx.program(vertex_shader=PARTICLE_VERT,
                             fragment_shader=SPLAT_FRAG)
    vao_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '3f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # ── Box wireframe (3D cube, 12 edges = 24 vertices) ──
    prog_box = ctx.program(vertex_shader=BOX_VERT, fragment_shader=BOX_FRAG)
    box_edges = np.array([
        # Bottom face
        0,0,0, 1,0,0,  1,0,0, 1,1,0,  1,1,0, 0,1,0,  0,1,0, 0,0,0,
        # Top face
        0,0,1, 1,0,1,  1,0,1, 1,1,1,  1,1,1, 0,1,1,  0,1,1, 0,0,1,
        # Vertical edges
        0,0,0, 0,0,1,  1,0,0, 1,0,1,  1,1,0, 1,1,1,  0,1,0, 0,1,1,
    ], dtype='f4')
    vbo_box = ctx.buffer(box_edges.tobytes())
    vao_box = ctx.vertex_array(prog_box, [(vbo_box, '3f', 'in_pos')])

    # ── Line shader (3D MVP for neighbor lines + axes) ──
    prog_line = ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)
    n_max_edges = params['num_particles'] * params['n_neighbors']
    vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 3 * 4)
    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '3f', 'in_pos')])

    # ── Coordinate axes (RGB = XYZ, from origin) ──
    axis_len = 0.3
    axis_verts = np.array([
        0, 0, 0,  axis_len, 0, 0,
        0, 0, 0,  0, axis_len, 0,
        0, 0, 0,  0, 0, axis_len,
    ], dtype='f4')
    vbo_axes = ctx.buffer(axis_verts.tobytes())
    vao_axes = ctx.vertex_array(prog_line, [(vbo_axes, '3f', 'in_pos')])

    # ── Overlay shader (2D divider line) ──
    prog_overlay = ctx.program(vertex_shader=OVERLAY_VERT,
                               fragment_shader=OVERLAY_FRAG)
    vbo_overlay = ctx.buffer(reserve=2 * 2 * 4)
    vao_overlay = ctx.vertex_array(prog_overlay, [(vbo_overlay, '2f', 'in_pos')])

    # ── Velocity field FBO ──
    vel_tex = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo = ctx.framebuffer(color_attachments=[vel_tex])
    vel_tex2 = ctx.texture((trail_w, trail_h), 3, dtype='f2')
    vel_tex2.filter = (moderngl.LINEAR, moderngl.LINEAR)
    vel_fbo2 = ctx.framebuffer(color_attachments=[vel_tex2])

    vbo_vel_col = ctx.buffer(reserve=num_particles * 3 * 4)
    vao_vel_splat = ctx.vertex_array(prog_splat, [
        (vbo_pos, '3f', 'in_pos'),
        (vbo_vel_col, '3f', 'in_color'),
    ])

    for tex in (trail_tex, trail_tex2, vel_tex, vel_tex2):
        tex.repeat_x = True
        tex.repeat_y = True

    # ── Create simulation ──
    sim = Simulation()
    running_sim = True

    # ── Mountain landscape setup ──
    mountain_landscape = None
    cost_landscape = None
    vbo_mtn_pos = None
    vbo_mtn_norm = None
    vbo_mtn_col = None
    vbo_cost_col = None
    ibo_mtn = None
    vao_mountain = None
    vao_cost = None
    mtn_n_indices = 0
    prog_mesh = ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)

    def build_mountain_mesh():
        """(Re)build the mountain and cost mesh GPU buffers."""
        nonlocal mountain_landscape, cost_landscape
        nonlocal vbo_mtn_pos, vbo_mtn_norm, vbo_mtn_col, vbo_cost_col
        nonlocal ibo_mtn, vao_mountain, vao_cost, mtn_n_indices

        if not _HAS_LANDSCAPE:
            return

        k = params['k']
        res = params['mountain_resolution']
        z_scale = params['mountain_z_scale']

        mountain_landscape = make_default_landscape(k=max(k, 2))
        cost_landscape = make_default_cost_landscape(k=max(k, 2))

        verts, normals, colors_f, indices, _ = generate_mountain_mesh(
            mountain_landscape, resolution=res, z_scale=z_scale,
            pref_dims=max(k, 2))

        colors_c, _ = generate_cost_colors(
            cost_landscape, landscape_k=max(k, 2), resolution=res)

        mtn_n_indices = len(indices)

        # Create GPU buffers
        vbo_mtn_pos = ctx.buffer(verts.tobytes())
        vbo_mtn_norm = ctx.buffer(normals.tobytes())
        vbo_mtn_col = ctx.buffer(colors_f.tobytes())
        vbo_cost_col = ctx.buffer(colors_c.tobytes())
        ibo_mtn = ctx.buffer(indices.tobytes())

        # VAO for fitness-colored mountain
        vao_mountain = ctx.vertex_array(prog_mesh, [
            (vbo_mtn_pos, '3f', 'in_pos'),
            (vbo_mtn_norm, '3f', 'in_normal'),
            (vbo_mtn_col, '3f', 'in_color'),
        ], index_buffer=ibo_mtn)

        # VAO for cost-colored overlay (same geometry, different colors)
        vao_cost = ctx.vertex_array(prog_mesh, [
            (vbo_mtn_pos, '3f', 'in_pos'),
            (vbo_mtn_norm, '3f', 'in_normal'),
            (vbo_cost_col, '3f', 'in_color'),
        ], index_buffer=ibo_mtn)

    # Build initial mountain mesh
    build_mountain_mesh()

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
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{fb_w}x{fb_h}",
            "-framerate", "30", "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
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
        data = _GL.glReadPixels(0, 0, fb_w, fb_h, _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE)
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
        n = params['num_particles']
        vbo_pos = ctx.buffer(reserve=n * 3 * 4)
        vbo_col = ctx.buffer(reserve=n * 3 * 4)
        vbo_vel_col = ctx.buffer(reserve=n * 3 * 4)
        vao_particle = ctx.vertex_array(prog_particle, [
            (vbo_pos, '3f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '3f', 'in_pos'),
            (vbo_col, '3f', 'in_color'),
        ])
        vao_vel_splat = ctx.vertex_array(prog_splat, [
            (vbo_pos, '3f', 'in_pos'),
            (vbo_vel_col, '3f', 'in_color'),
        ])
        n_max_edges = n * params['n_neighbors']
        vbo_line = ctx.buffer(reserve=n_max_edges * 2 * 3 * 4)
        vao_line = ctx.vertex_array(prog_line, [(vbo_line, '3f', 'in_pos')])

    def do_reset():
        nonlocal running_sim
        if params['auto_scale']:
            ref = auto_scale_ref
            scale = (ref['n'] / params['num_particles']) ** (1.0 / 3.0)
            params['step_size'] = ref['step_size'] * scale
            params['neighbor_radius'] = ref['radius'] * scale
        sim.reset()
        rebuild_buffers()
        trail_fbo.use()
        ctx.clear(0, 0, 0)
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        vel_fbo.use()
        ctx.clear(0, 0, 0)
        vel_fbo2.use()
        ctx.clear(0, 0, 0)
        running_sim = True

    # ── FPS tracking ──
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    t_sim = 0.0

    # ── JIT warmup ──
    print("Warming up numba JIT kernels (3D)...")
    _warmup_n = 100
    _warmup_pos = np.random.rand(_warmup_n, 3).astype(np.float64)
    _warmup_prefs = np.random.rand(_warmup_n, 3).astype(np.float64)
    _warmup_dm = np.zeros((_warmup_n, 3, 3), dtype=np.float64)
    _warmup_cell_size = 0.1
    _warmup_so, _warmup_cs, _warmup_ce, _warmup_gr, _warmup_csa = \
        grid_build(_warmup_pos, _warmup_cell_size)
    _warmup_counts = _count_radius(_warmup_pos, _warmup_so, _warmup_cs,
                                    _warmup_ce, _warmup_gr, _warmup_csa, 0.1, 1.0)
    _warmup_nbr, _warmup_val, _ = _query_radius(
        _warmup_pos, _warmup_so, np.empty(0, dtype=np.int32),
        _warmup_cs, _warmup_ce, _warmup_gr, _warmup_csa, 0.1, 1.0, 10)
    _warmup_knn = _query_knn(_warmup_pos, _warmup_so, _warmup_cs, _warmup_ce,
                              _warmup_gr, _warmup_csa, 1.0, 5)
    _warmup_valid = np.ones((_warmup_n, _warmup_nbr.shape[1]), dtype=np.bool_)
    _step_inner_prod_avg(_warmup_pos, _warmup_prefs,
                         _warmup_nbr.astype(np.int64), _warmup_valid,
                         1.0, 3, 0.005, 0.0, 0.0, False, False, 0.01)
    _step_per_dim(_warmup_pos, _warmup_prefs, _warmup_dm,
                  _warmup_nbr.astype(np.int64), _warmup_valid,
                  1.0, 3, 0.005, 0.0, 0.0, False, 0.0, False, False,
                  False, 0.01, False)
    print("JIT warmup complete.")

    # Track previous camera state for trail clearing
    prev_cam_az = camera.azimuth
    prev_cam_el = camera.elevation
    prev_cam_dist = camera.distance

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while not glfw.window_should_close(window):
        glfw.poll_events()

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
        # Mountain mode: project particles onto the fitness surface
        # Use strategy vector if available (dual-space), else prefs
        if params['mountain_mode'] and mountain_landscape is not None:
            mountain_coords = getattr(sim, 'strategy', None)
            if mountain_coords is None:
                mountain_coords = sim.prefs
            positions = project_particles_to_surface(
                mountain_coords, mountain_landscape,
                z_scale=params['mountain_z_scale'])
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        vel_colors = sim.get_velocity_colors()
        vbo_vel_col.write(vel_colors.tobytes())

        # ── Compute MVP ──
        aspect = (fb_w / 2.0) / fb_h
        mvp = camera.get_mvp(aspect)
        mvp_bytes = mvp.tobytes()

        # ── Check if camera moved (clear trails if so) ──
        cam_changed = (camera.azimuth != prev_cam_az or
                       camera.elevation != prev_cam_el or
                       camera.distance != prev_cam_dist)
        if cam_changed:
            for fbo in (trail_fbo, trail_fbo2, vel_fbo, vel_fbo2):
                fbo.use()
                ctx.clear(0, 0, 0)
            prev_cam_az = camera.azimuth
            prev_cam_el = camera.elevation
            prev_cam_dist = camera.distance

        # ── Trail rendering pass ──
        # Pass 1: Decay
        trail_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        trail_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: Splat particles into trail FBO
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['mvp'].write(mvp_bytes)
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['point_size'] = params['point_size']
        vao_splat.render(moderngl.POINTS)

        # Pass 3: Swap
        trail_tex, trail_tex2 = trail_tex2, trail_tex
        trail_fbo, trail_fbo2 = trail_fbo2, trail_fbo

        # ── Velocity field rendering pass ──
        vel_fbo2.use()
        ctx.clear(0, 0, 0)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        vel_tex.use(0)
        prog_trail_decay['trail_tex'] = 0
        prog_trail_decay['decay'] = params['trail_decay']
        vao_trail_decay.render(moderngl.TRIANGLE_STRIP)

        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        prog_splat['mvp'].write(mvp_bytes)
        prog_splat['viewport_offset'] = (0.0, 0.0)
        prog_splat['viewport_scale'] = (1.0, 1.0)
        prog_splat['point_size'] = params['point_size']
        vao_vel_splat.render(moderngl.POINTS)

        vel_tex, vel_tex2 = vel_tex2, vel_tex
        vel_fbo, vel_fbo2 = vel_fbo2, vel_fbo

        # ── Screen rendering ──
        ctx.screen.use()
        ctx.clear(0, 0, 0)

        # Left half: live 3D particles
        ctx.viewport = (0, 0, fb_w // 2, fb_h)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if params['show_neighbors'] and sim.nbr_ids is not None:
            lines = sim.get_neighbor_lines()
            n_line_verts = len(lines)
            if n_line_verts > 0:
                needed = n_line_verts * 3 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '3f', 'in_pos')])
                vbo_line.write(lines.tobytes())
                prog_line['mvp'].write(mvp_bytes)
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.08)
                vao_line.render(moderngl.LINES, vertices=n_line_verts)

        if params['show_radius']:
            cam_right, cam_up = camera.get_right_up()
            circles = make_radius_circles_3d(
                sim.pos.astype(np.float32), params['neighbor_radius'],
                cam_right, cam_up)
            n_circle_verts = len(circles)
            if n_circle_verts > 0:
                needed = n_circle_verts * 3 * 4
                if needed > vbo_line.size:
                    vbo_line = ctx.buffer(reserve=needed)
                    vao_line = ctx.vertex_array(prog_line, [(vbo_line, '3f', 'in_pos')])
                vbo_line.write(circles.tobytes())
                prog_line['mvp'].write(mvp_bytes)
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.04)
                vao_line.render(moderngl.LINES, vertices=n_circle_verts)

        # ── Render mountain mesh (before particles so particles appear on top) ──
        if params['show_mountain'] and vao_mountain is not None:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            # Light from upper-right
            light_dir = np.array([0.4, 0.3, 0.8], dtype='f4')
            light_dir /= np.linalg.norm(light_dir)
            prog_mesh['mvp'].write(mvp_bytes)
            prog_mesh['alpha'] = params['mountain_alpha']
            prog_mesh['light_dir'] = tuple(light_dir)
            vao_mountain.render(moderngl.TRIANGLES)

        # ── Render cost overlay (slightly offset Z to avoid z-fighting) ──
        if params['show_cost_overlay'] and vao_cost is not None:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            light_dir = np.array([0.4, 0.3, 0.8], dtype='f4')
            light_dir /= np.linalg.norm(light_dir)
            prog_mesh['mvp'].write(mvp_bytes)
            prog_mesh['alpha'] = params['cost_alpha']
            prog_mesh['light_dir'] = tuple(light_dir)
            # Use polygon offset to prevent z-fighting with mountain
            _GL.glEnable(_GL.GL_POLYGON_OFFSET_FILL)
            _GL.glPolygonOffset(-1.0, -1.0)
            vao_cost.render(moderngl.TRIANGLES)
            _GL.glDisable(_GL.GL_POLYGON_OFFSET_FILL)

        # Render particles
        prog_particle['mvp'].write(mvp_bytes)
        prog_particle['viewport_offset'] = (0.0, 0.0)
        prog_particle['viewport_scale'] = (1.0, 1.0)
        prog_particle['point_size'] = params['point_size']
        vao_particle.render(moderngl.POINTS)

        # Wireframe box
        if params['show_box']:
            prog_box['mvp'].write(mvp_bytes)
            vao_box.render(moderngl.LINES, vertices=24)

        # Coordinate axes
        if params['show_axes']:
            prog_line['mvp'].write(mvp_bytes)
            # X axis (red)
            prog_line['line_color'] = (1.0, 0.3, 0.3, 0.9)
            vao_axes.render(moderngl.LINES, vertices=2, first=0)
            # Y axis (green)
            prog_line['line_color'] = (0.3, 1.0, 0.3, 0.9)
            vao_axes.render(moderngl.LINES, vertices=2, first=2)
            # Z axis (blue)
            prog_line['line_color'] = (0.3, 0.3, 1.0, 0.9)
            vao_axes.render(moderngl.LINES, vertices=2, first=4)

        ctx.disable(moderngl.DEPTH_TEST)

        # Right half: display selected view
        ctx.viewport = (fb_w // 2, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.ONE, moderngl.ZERO
        right_tex = vel_tex if params['right_view'] == 1 else trail_tex
        right_tex.use(0)
        prog_display['tex'] = 0
        prog_display['view_center'] = (0.5, 0.5)
        prog_display['view_zoom'] = 1.0
        vao_display.render(moderngl.TRIANGLE_STRIP)

        if params['show_box'] or params['show_axes']:
            ctx.viewport = (fb_w // 2, 0, fb_w // 2, fb_h)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.enable(moderngl.DEPTH_TEST)
            if params['show_box']:
                prog_box['mvp'].write(mvp_bytes)
                vao_box.render(moderngl.LINES, vertices=24)
            if params['show_axes']:
                prog_line['mvp'].write(mvp_bytes)
                prog_line['line_color'] = (1.0, 0.3, 0.3, 0.9)
                vao_axes.render(moderngl.LINES, vertices=2, first=0)
                prog_line['line_color'] = (0.3, 1.0, 0.3, 0.9)
                vao_axes.render(moderngl.LINES, vertices=2, first=2)
                prog_line['line_color'] = (0.3, 0.3, 1.0, 0.9)
                vao_axes.render(moderngl.LINES, vertices=2, first=4)
            ctx.disable(moderngl.DEPTH_TEST)

        # ── Divider line (2D overlay) ──
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        divider = np.array([0.5, 0.0, 0.5, 1.0], dtype='f4')
        vbo_overlay.write(divider.tobytes())
        prog_overlay['line_color'] = (1.0, 1.0, 1.0, 0.5)
        vao_overlay.render(moderngl.LINES, vertices=2)

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

        imgui.text(f"Status: {status}")
        imgui.text(f"Step: {sim.step_count}  FPS: {fps:.0f}")
        imgui.text(f"Sim: {t_sim*1000:.1f}ms  "
                   f"grid: {sim._t_build*1000:.1f}ms  "
                   f"query: {sim._t_query*1000:.1f}ms  "
                   f"physics: {sim._t_physics*1000:.1f}ms")
        imgui.text(f"Neighbors/particle: {sim._n_nbrs}")
        imgui.separator()

        label = "Resume" if not running_sim else "Pause"
        if imgui.button(label, imgui.ImVec2(80, 0)):
            running_sim = not running_sim
        imgui.same_line()
        if imgui.button("Reset", imgui.ImVec2(80, 0)):
            do_reset()
        if rec_process is not None:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(1.0, 0.2, 0.2, 1.0), "REC")
            imgui.text(f"Frames: {rec_frame_count}  Interval: {rec_interval:.1f}s")
            if imgui.button("Stop Rec", imgui.ImVec2(80, 0)):
                stop_recording()
        else:
            changed, rec_interval = imgui.drag_float(
                "Rec Interval", rec_interval, 0.05, 0.1, 10.0, "%.1fs")
            if imgui.button("Record", imgui.ImVec2(80, 0)):
                start_recording()
        imgui.separator()

        # View selector
        if imgui.radio_button("Trails", params['right_view'] == 0):
            params['right_view'] = 0
        imgui.same_line()
        if imgui.radio_button("Velocity", params['right_view'] == 1):
            params['right_view'] = 1

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
        changed, v = imgui.checkbox("Axes", params['show_axes'])
        if changed:
            params['show_axes'] = v
        imgui.same_line()
        if imgui.button("Reset Camera"):
            camera.reset()
        imgui.text(f"Az: {math.degrees(camera.azimuth):.0f}  "
                   f"El: {math.degrees(camera.elevation):.0f}  "
                   f"Dist: {camera.distance:.1f}")
        imgui.separator()

        # ── Mountain visualization ──
        if _HAS_LANDSCAPE:
            if imgui.collapsing_header("Mountain / Cost Landscape"):
                changed, v = imgui.checkbox("Show Mountain", params['show_mountain'])
                if changed:
                    params['show_mountain'] = v
                    if v and vao_mountain is None:
                        build_mountain_mesh()
                imgui.same_line()
                changed, v = imgui.checkbox("Cost Overlay", params['show_cost_overlay'])
                if changed:
                    params['show_cost_overlay'] = v
                changed, v = imgui.checkbox("Mountain Mode", params['mountain_mode'])
                if changed:
                    params['mountain_mode'] = v
                if params['mountain_mode']:
                    imgui.same_line()
                    imgui.text_colored(imgui.ImVec4(0.3, 1.0, 0.3, 1.0),
                                       "particles on surface")
                changed, v = imgui.drag_float("Mtn Alpha", params['mountain_alpha'],
                                              0.01, 0.05, 1.0, "%.2f")
                if changed:
                    params['mountain_alpha'] = v
                changed, v = imgui.drag_float("Cost Alpha", params['cost_alpha'],
                                              0.01, 0.05, 1.0, "%.2f")
                if changed:
                    params['cost_alpha'] = v
                changed, v = imgui.drag_float("Z Scale", params['mountain_z_scale'],
                                              0.01, 0.1, 1.0, "%.2f")
                if changed:
                    params['mountain_z_scale'] = v
                    build_mountain_mesh()
                changed, v = imgui.drag_int("Mesh Res", params['mountain_resolution'],
                                            1.0, 16, 128)
                if changed:
                    params['mountain_resolution'] = v
                    build_mountain_mesh()
            imgui.separator()

        # ── Live parameters ──
        if imgui.collapsing_header(
                "Live Parameters",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            changed, v = imgui.drag_float("Step Size", params['step_size'],
                                          0.0001, 0.001, 0.05, "%.4f")
            if changed:
                params['step_size'] = v
            changed, v = imgui.drag_int("Steps/Frame", params['steps_per_frame'],
                                        0.5, 1, 100)
            if changed:
                params['steps_per_frame'] = v
            if params['steps_per_frame'] > 1:
                changed, v = imgui.checkbox("Reuse Neighbors", params['reuse_neighbors'])
                if changed:
                    params['reuse_neighbors'] = v
            changed, v = imgui.drag_float("Repulsion", params['repulsion'],
                                          0.0001, 0.0, 0.02, "%.4f")
            if changed:
                params['repulsion'] = v
            changed, v = imgui.drag_float("Dir Memory", params['dir_memory'],
                                          0.005, 0.0, 0.99, "%.3f")
            if changed:
                params['dir_memory'] = v
            changed, v = imgui.drag_float("Social", params['social'],
                                          0.0005, 0.0, 0.1, "%.4f")
            if changed:
                params['social'] = v
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
            changed, v = imgui.checkbox("Best by Magnitude", params['best_by_magnitude'])
            if changed:
                params['best_by_magnitude'] = v
            changed, v = imgui.drag_float("Trail Decay", params['trail_decay'],
                                          0.005, 0.8, 1.0, "%.3f")
            if changed:
                params['trail_decay'] = v
            changed, v = imgui.drag_float("Point Size", params['point_size'],
                                          0.1, 1.0, 20.0, "%.1f")
            if changed:
                params['point_size'] = v
            _nbr_modes = ["KNN", "KNN + Radius", "Radius Only"]
            changed, v = imgui.combo("Neighbor Mode", params['neighbor_mode'],
                                     _nbr_modes)
            if changed:
                params['neighbor_mode'] = v
            _knn_methods = ["Hash Grid", "cKDTree (f64)", "cKDTree (f32)"]
            changed, v = imgui.combo("KNN Method", params['knn_method'],
                                     _knn_methods)
            if changed:
                params['knn_method'] = v
            _physics_engines = ["Numba", "NumPy (original)", "PyTorch"]
            changed, v = imgui.combo("Physics", params['physics_engine'],
                                     _physics_engines)
            if changed:
                params['physics_engine'] = v
            if params['physics_engine'] == 2:
                _precisions = ["f16", "bf16", "f32", "f64"]
                changed, v = imgui.combo("Precision", params['torch_precision'],
                                         _precisions)
                if changed:
                    params['torch_precision'] = v
                _devices = ["Auto (%s)" % _TORCH_DEVICE, "CPU"]
                changed, v = imgui.combo("Device", params['torch_device'], _devices)
                if changed:
                    params['torch_device'] = v
                if not _HAS_TORCH:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                                       "torch not installed!")
            changed, v = imgui.checkbox("Use f64 pos", params['use_f64'])
            if changed:
                params['use_f64'] = v
                imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                                   "(reset to apply)")
            if params['knn_method'] == 0:
                changed, v = imgui.checkbox("Debug KNN", params['debug_knn'])
                if changed:
                    params['debug_knn'] = v
            if params['neighbor_mode'] < 2:
                changed, v = imgui.drag_int("Neighbors", params['n_neighbors'],
                                            0.5, 1, 30)
                if changed:
                    params['n_neighbors'] = v
            changed, v = imgui.drag_float("Radius", params['neighbor_radius'],
                                          0.001, 0.001, 0.3, "%.4f")
            if changed:
                params['neighbor_radius'] = v

        # ── Reset-required parameters ──
        if imgui.collapsing_header("Reset-Required Params"):
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                               "(changes apply on Reset)")
            changed, v = imgui.combo("Init Preset", params['init_preset'],
                                     INIT_PRESETS)
            if changed:
                params['init_preset'] = v
            changed, v = imgui.checkbox("Flat Z (2D debug)", params['flat_z'])
            if changed:
                params['flat_z'] = v
            changed, v = imgui.drag_int("Particles", params['num_particles'],
                                        5.0, 2, 200000)
            if changed:
                params['num_particles'] = v
            changed, v = imgui.checkbox("Auto-scale", params['auto_scale'])
            if changed:
                params['auto_scale'] = v
            if params['auto_scale']:
                ref = auto_scale_ref
                scale = (ref['n'] / params['num_particles']) ** (1.0 / 3.0)
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

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        # ── Timelapse capture ──
        if rec_process is not None:
            now = time.monotonic()
            if now - rec_last_time >= rec_interval:
                rec_last_time = now
                capture_frame()

        # ── Swap and FPS ──
        glfw.swap_buffers(window)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        glfw.set_window_title(window,
            f"3D Particles [{status}] Step:{sim.step_count} FPS:{fps:.0f}")

    # ── Cleanup ──
    stop_recording()
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context(imgui_ctx)
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D Particle Simulation')
    parser.add_argument('--mountain', action='store_true',
                        help='Start with mountain surface, cost overlay, and '
                             'mountain mode enabled')
    args = parser.parse_args()
    if args.mountain:
        from .simulation3d import params as _p3d
        _p3d['show_mountain'] = True
        _p3d['show_cost_overlay'] = True
        _p3d['mountain_mode'] = True
    main()
