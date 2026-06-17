"""
3D Preference-Directed Particle Simulation
===========================================

3D version of sim_gpu_compute.py with orbit camera, MVP projection,
wireframe cube, and all physics/neighbor features preserved.

Mountain Mode adds a knowledge manifold experiment:
  - Hidden fitness landscape (wireframe ghost)
  - Growing knowledge surface (solid mesh, particles walk on it)
  - Particles sense the hidden gradient with noise, aggregate across
    neighbours, and deposit knowledge where they stand.

Controls:
  Left drag   — orbit camera
  Scroll      — zoom in/out
  Space       — pause / resume
  r           — reset simulation and trails
  q / Esc     — quit
  imgui panel — full slider control for all parameters
"""

# ── Imports ──────────────────────────────────────────────────────────
import argparse
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
    GHOST_VERT, GHOST_FRAG,
)
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

# ── Mountain mode parameters (separate from sim params) ─────────────
mountain_params = dict(
    enabled=False,
    z_scale=0.5,
    grid_res=64,
    # --- Research dynamics ---
    gradient_noise_base=3.0,   # Base noise std on "noisy up" observations
    gradient_noise_max=8.0,    # Max noise when stuck (adaptive)
    noise_adapt_rate=0.05,     # How fast noise adapts to success/failure
    knowledge_write_rate=0.005, # Deposit per step (tuned for ~5 min experiment)
    knowledge_climb_rate=0.002, # Climb toward known-good regions on manifold
    # --- Visionary ---
    visionary_fraction=0.02,   # Fraction of particles that are visionaries
    visionary_nudge=0.001,     # Lateral nudge toward global peak (visionaries only)
    # --- Knowledge field ---
    diffusion_sigma=0.5,       # Knowledge spread radius
    decay=0.9999,              # Slower decay
    support_radius=3,          # Structural support neighbourhood (cells)
    max_slope=0.4,             # Max height above local mean
    # --- Reward-modulated social learning ---
    reward_ema_alpha=0.05,     # EMA smoothing for reward signal
    reward_exp_scale=3.0,      # Exponential scaling of absolute height for reward
    reward_growth_beta=1.0,    # Weight of growth rate in combined reward
    reward_social_scale=5.0,   # How much reward amplifies social pull
    # --- Exploration ---
    explore_prob=0.003,
    explore_radius=0.2,
    # --- Rendering ---
    show_ghost=True,
    ghost_alpha=0.15,
    knowledge_alpha=0.7,
)


def main():
    # ── Parse CLI ──
    parser = argparse.ArgumentParser()
    parser.add_argument('--mountain', action='store_true',
                        help='Enable knowledge manifold mountain mode')
    args, _ = parser.parse_known_args()

    if args.mountain:
        mountain_params['enabled'] = True
        # Good defaults for mountain mode
        params['k'] = 3
        params['social'] = 0.01
        params['step_size'] = 0.003
        params['inner_prod_avg'] = True
        params['num_particles'] = 500

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

    # ── Mesh shaders for mountain surfaces ──
    prog_mesh = ctx.program(vertex_shader=MESH_VERT,
                            fragment_shader=MESH_FRAG)
    prog_ghost = ctx.program(vertex_shader=GHOST_VERT,
                             fragment_shader=GHOST_FRAG)

    # Mountain mesh GPU buffers (allocated lazily on first build)
    mesh_vbo_pos = None
    mesh_vbo_norm = None
    mesh_vbo_col = None
    mesh_ibo = None
    mesh_vao = None
    mesh_n_indices = 0

    ghost_vbo_pos = None
    ghost_vbo_col = None
    ghost_ibo = None
    ghost_vao = None
    ghost_n_indices = 0

    # ── Mountain mode state ──
    knowledge_field = None
    landscape = None
    _mountain_rng = np.random.default_rng(123)
    _reward_ema = None       # (N,) per-particle smoothed reward
    _noise_level = None      # (N,) per-particle adaptive noise level

    def build_mountain():
        """Initialise the knowledge field and hidden fitness landscape."""
        nonlocal knowledge_field, landscape
        from .knowledge_field import KnowledgeField
        from experiments.landscape import make_default_landscape

        landscape = make_default_landscape(seed=42)
        G = mountain_params['grid_res']
        knowledge_field = KnowledgeField(
            grid_res=G,
            diffusion_sigma=mountain_params['diffusion_sigma'],
            decay=mountain_params['decay'],
            support_radius=mountain_params['support_radius'],
            max_slope=mountain_params['max_slope'],
        )
        knowledge_field.set_fitness_surface(landscape)
        print(f"Mountain mode: landscape built, grid {G}x{G}")

    def rebuild_mountain_mesh():
        """Regenerate GPU mesh buffers from current knowledge grid."""
        nonlocal mesh_vbo_pos, mesh_vbo_norm, mesh_vbo_col, mesh_ibo
        nonlocal mesh_vao, mesh_n_indices
        nonlocal ghost_vbo_pos, ghost_vbo_col, ghost_ibo
        nonlocal ghost_vao, ghost_n_indices

        if knowledge_field is None:
            return

        from .mountain_mesh import (
            generate_surface_mesh, generate_wireframe_indices,
        )

        z_scale = mountain_params['z_scale']

        # Knowledge surface (solid green)
        verts, norms, cols, indices = generate_surface_mesh(
            knowledge_field.grid, z_scale=z_scale, color=(0.2, 0.7, 0.3))
        mesh_n_indices = len(indices)

        if mesh_vbo_pos is None or mesh_vbo_pos.size < verts.nbytes:
            mesh_vbo_pos = ctx.buffer(verts.tobytes())
            mesh_vbo_norm = ctx.buffer(norms.tobytes())
            mesh_vbo_col = ctx.buffer(cols.tobytes())
            mesh_ibo = ctx.buffer(indices.tobytes())
            mesh_vao = ctx.vertex_array(prog_mesh, [
                (mesh_vbo_pos, '3f', 'in_pos'),
                (mesh_vbo_norm, '3f', 'in_normal'),
                (mesh_vbo_col, '3f', 'in_color'),
            ], mesh_ibo)
        else:
            mesh_vbo_pos.write(verts.tobytes())
            mesh_vbo_norm.write(norms.tobytes())
            mesh_vbo_col.write(cols.tobytes())
            mesh_ibo.write(indices.tobytes())

        # Ghost fitness surface (wireframe, pale blue)
        if knowledge_field._fitness_grid is not None:
            g_verts, g_norms, g_cols, _ = generate_surface_mesh(
                knowledge_field._fitness_grid, z_scale=z_scale,
                color=(0.4, 0.5, 0.9))
            G = knowledge_field.G
            wire_indices = generate_wireframe_indices(G)
            ghost_n_indices = len(wire_indices)

            if ghost_vbo_pos is None or ghost_vbo_pos.size < g_verts.nbytes:
                ghost_vbo_pos = ctx.buffer(g_verts.tobytes())
                ghost_vbo_col = ctx.buffer(g_cols.tobytes())
                ghost_ibo = ctx.buffer(wire_indices.tobytes())
                ghost_vao = ctx.vertex_array(prog_ghost, [
                    (ghost_vbo_pos, '3f', 'in_pos'),
                    (ghost_vbo_col, '3f', 'in_color'),
                ], ghost_ibo)
            else:
                ghost_vbo_pos.write(g_verts.tobytes())
                ghost_vbo_col.write(g_cols.tobytes())
                ghost_ibo.write(wire_indices.tobytes())

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

        # Reset knowledge field if mountain mode is on
        if mountain_params['enabled'] and knowledge_field is not None:
            knowledge_field.reset()
            knowledge_field.diffusion_sigma = mountain_params['diffusion_sigma']
            knowledge_field.decay = mountain_params['decay']
            # Sync initial positions to knowledge surface
            _sync_positions_to_knowledge()

    def _sync_positions_to_knowledge():
        """Project particle positions onto the knowledge manifold surface."""
        if knowledge_field is None:
            return
        from .mountain_mesh import project_particles_to_knowledge_surface
        projected = project_particles_to_knowledge_surface(
            knowledge_field, sim.prefs, z_scale=mountain_params['z_scale'])
        sim.pos[:, :3] = projected.astype(sim.pos.dtype)

    # ── Initialise mountain if enabled ──
    if mountain_params['enabled']:
        build_mountain()
        _sync_positions_to_knowledge()
        rebuild_mountain_mesh()

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
                # Phase 1: Social dynamics (Richard's original)
                sim.step(reuse_neighbors=(reuse and sub > 0))

                # Phase 1.5: Reward-weighted social nudge (mountain mode only)
                # Successful particles (high reward_ema) pull harder on
                # neighbours, making productive teams more attractive.
                # This runs AFTER Richard's social dynamics but BEFORE
                # the knowledge step, so it modulates team formation.
                if mountain_params['enabled'] and _reward_ema is not None:
                    _reward_social_nudge(sim, _reward_ema)

                # Phase 2: Knowledge manifold step (mountain mode only)
                if mountain_params['enabled'] and knowledge_field is not None:
                    # Lazy-init per-particle state
                    if _reward_ema is None or len(_reward_ema) != sim.n:
                        _reward_ema = np.zeros(sim.n, dtype=np.float64)
                        _noise_level = np.full(sim.n,
                                               mountain_params['gradient_noise_base'],
                                               dtype=np.float64)
                    _knowledge_step(sim, knowledge_field, landscape,
                                    _mountain_rng, _reward_ema, _noise_level)

        t_sim = time.perf_counter() - t0

        # ── Upload particle data to GPU ──
        positions, colors = sim.get_render_data()
        vbo_pos.write(positions.tobytes())
        vbo_col.write(colors.tobytes())

        vel_colors = sim.get_velocity_colors()
        vbo_vel_col.write(vel_colors.tobytes())

        # ── Update mountain mesh (every frame, not every sub-step) ──
        if mountain_params['enabled'] and knowledge_field is not None:
            rebuild_mountain_mesh()

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

        # ── Mountain mesh rendering ──
        if mountain_params['enabled'] and mesh_vao is not None:
            # Ghost fitness surface (wireframe, render first for depth)
            if mountain_params['show_ghost'] and ghost_vao is not None:
                prog_ghost['mvp'].write(mvp_bytes)
                prog_ghost['alpha'] = mountain_params['ghost_alpha']
                ghost_vao.render(moderngl.LINES)

            # Knowledge surface (solid triangles)
            prog_mesh['mvp'].write(mvp_bytes)
            prog_mesh['alpha'] = mountain_params['knowledge_alpha']
            prog_mesh['light_dir'] = (0.3, 1.0, 0.5)
            mesh_vao.render(moderngl.TRIANGLES)

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

        # Mountain metrics
        if mountain_params['enabled'] and knowledge_field is not None:
            cov = knowledge_field.coverage()
            peak = knowledge_field.peak_knowledge()
            imgui.text(f"Knowledge: cov={cov:.1%}  peak={peak:.3f}")

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

        # ── Mountain Mode ──
        if imgui.collapsing_header(
                "Mountain Mode",
                flags=int(imgui.TreeNodeFlags_.default_open.value)
                      if mountain_params['enabled'] else 0):
            changed, v = imgui.checkbox("Enable Mountain",
                                        mountain_params['enabled'])
            if changed:
                mountain_params['enabled'] = v
                if v and knowledge_field is None:
                    params['k'] = 3
                    params['social'] = 0.01
                    params['step_size'] = 0.003
                    params['inner_prod_avg'] = True
                    build_mountain()
                    do_reset()

            if mountain_params['enabled']:
                changed, v = imgui.drag_float(
                    "Z Scale", mountain_params['z_scale'],
                    0.01, 0.1, 2.0, "%.2f")
                if changed:
                    mountain_params['z_scale'] = v
                changed, v = imgui.drag_float(
                    "Noise Base", mountain_params['gradient_noise_base'],
                    0.1, 0.0, 10.0, "%.1f")
                if changed:
                    mountain_params['gradient_noise_base'] = v
                changed, v = imgui.drag_float(
                    "Noise Max", mountain_params['gradient_noise_max'],
                    0.1, 0.0, 20.0, "%.1f")
                if changed:
                    mountain_params['gradient_noise_max'] = v
                changed, v = imgui.drag_float(
                    "Noise Adapt", mountain_params['noise_adapt_rate'],
                    0.005, 0.0, 0.5, "%.3f")
                if changed:
                    mountain_params['noise_adapt_rate'] = v
                changed, v = imgui.drag_float(
                    "Write Rate", mountain_params['knowledge_write_rate'],
                    0.0005, 0.0, 0.1, "%.4f")
                if changed:
                    mountain_params['knowledge_write_rate'] = v
                changed, v = imgui.drag_float(
                    "Diffusion", mountain_params['diffusion_sigma'],
                    0.05, 0.0, 5.0, "%.2f")
                if changed:
                    mountain_params['diffusion_sigma'] = v
                    if knowledge_field is not None:
                        knowledge_field.diffusion_sigma = v
                changed, v = imgui.drag_float(
                    "Decay", mountain_params['decay'],
                    0.0005, 0.9, 1.0, "%.4f")
                if changed:
                    mountain_params['decay'] = v
                    if knowledge_field is not None:
                        knowledge_field.decay = v
                changed, v = imgui.drag_float(
                    "Know Climb", mountain_params['knowledge_climb_rate'],
                    0.0005, 0.0, 0.05, "%.4f")
                if changed:
                    mountain_params['knowledge_climb_rate'] = v
                imgui.separator()
                imgui.text("Visionary")
                changed, v = imgui.drag_float(
                    "Vis Fraction", mountain_params['visionary_fraction'],
                    0.005, 0.0, 0.2, "%.3f")
                if changed:
                    mountain_params['visionary_fraction'] = v
                changed, v = imgui.drag_float(
                    "Vis Nudge", mountain_params['visionary_nudge'],
                    0.0005, 0.0, 0.02, "%.4f")
                if changed:
                    mountain_params['visionary_nudge'] = v
                imgui.separator()
                imgui.text("Reward")
                changed, v = imgui.drag_float(
                    "Exp Scale", mountain_params['reward_exp_scale'],
                    0.1, 0.0, 10.0, "%.1f")
                if changed:
                    mountain_params['reward_exp_scale'] = v
                changed, v = imgui.drag_float(
                    "Growth Beta", mountain_params['reward_growth_beta'],
                    0.1, 0.0, 5.0, "%.1f")
                if changed:
                    mountain_params['reward_growth_beta'] = v
                changed, v = imgui.drag_float(
                    "Social Scale", mountain_params['reward_social_scale'],
                    0.1, 0.0, 20.0, "%.1f")
                if changed:
                    mountain_params['reward_social_scale'] = v
                changed, v = imgui.drag_float(
                    "Reward EMA", mountain_params['reward_ema_alpha'],
                    0.005, 0.0, 0.5, "%.3f")
                if changed:
                    mountain_params['reward_ema_alpha'] = v
                changed, v = imgui.drag_float(
                    "Explore Prob", mountain_params['explore_prob'],
                    0.001, 0.0, 0.1, "%.4f")
                if changed:
                    mountain_params['explore_prob'] = v
                changed, v = imgui.drag_float(
                    "Explore Radius", mountain_params['explore_radius'],
                    0.01, 0.0, 1.0, "%.3f")
                if changed:
                    mountain_params['explore_radius'] = v
                imgui.separator()
                changed, v = imgui.checkbox("Show Ghost",
                                            mountain_params['show_ghost'])
                if changed:
                    mountain_params['show_ghost'] = v
                changed, v = imgui.drag_float(
                    "Ghost Alpha", mountain_params['ghost_alpha'],
                    0.01, 0.0, 1.0, "%.2f")
                if changed:
                    mountain_params['ghost_alpha'] = v
                changed, v = imgui.drag_float(
                    "Surface Alpha", mountain_params['knowledge_alpha'],
                    0.01, 0.0, 1.0, "%.2f")
                if changed:
                    mountain_params['knowledge_alpha'] = v

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


# ====================================================================
# REWARD-WEIGHTED SOCIAL NUDGE (Phase 1.5)
# ====================================================================

def _reward_social_nudge(sim, reward_ema):
    """Pull particles toward their most successful neighbours.

    After Richard's social dynamics (Phase 1), this applies an additional
    small preference nudge that is weighted by each neighbour's reward
    (exponentially-scaled absolute knowledge height).  Particles on
    high knowledge peaks pull disproportionately harder, creating
    winner-take-all talent flow toward the most successful teams.

    This modulates *who clusters with whom* — not the knowledge dynamics
    directly.  The effect is that talent flows toward productive teams.
    """
    scale = mountain_params['reward_social_scale']
    if scale <= 0 or sim.nbr_ids is None:
        return

    prefs = sim.prefs  # (N, K)
    nbr_ids = sim.nbr_ids  # (N, M)
    valid = sim._valid_mask

    # Neighbour reward weights: 1 + scale * reward_ema[nbr]
    # Particles with zero reward have weight 1 (normal).
    # Particles with high reward have weight >> 1 (stronger pull).
    nbr_reward = reward_ema[nbr_ids]  # (N, M)
    w = 1.0 + scale * np.maximum(nbr_reward, 0.0)  # (N, M)

    if valid is not None:
        w = w * valid
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.maximum(w_sum, 1e-10)
    else:
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.maximum(w_sum, 1e-10)

    w_norm = w / w_sum  # (N, M) — normalized weights

    # Reward-weighted neighbour mean preference
    nbr_prefs = prefs[nbr_ids]  # (N, M, K)
    weighted_mean = (nbr_prefs * w_norm[:, :, None]).sum(axis=1)  # (N, K)

    # Small nudge toward the reward-weighted mean
    # Use a fraction of the normal social rate to avoid overwhelming
    # Richard's social dynamics.
    nudge_strength = params['social'] * 0.5  # half the normal social rate
    prefs[:] = (1.0 - nudge_strength) * prefs + nudge_strength * weighted_mean
    np.clip(prefs, -1, 1, out=prefs)


# ====================================================================
# KNOWLEDGE STEP — the core mountain mode logic
# ====================================================================

def _knowledge_step(sim, knowledge_field, landscape, rng,
                    reward_ema, noise_level):
    """One step of knowledge manifold dynamics.

    After sim.step() has run social dynamics (Phase 1), this function:

    1. Knowledge gradient climbing: particles drift toward known-good
       regions on the knowledge surface (information they actually have).

    2. Visionary nudge: rare visionary particles (fraction ~2%) get a
       small lateral nudge toward the global peak of the hidden fitness
       landscape.  Regular particles have NO access to the hidden gradient.

    3. Exploration: stuck particles (adaptive high noise) get random
       lateral jumps to escape local clusters.

    4. Deposit knowledge ("push up"): the main research action.  Each
       particle tries to raise the knowledge manifold at its current
       position.  Structural support constraint limits growth — you need
       a broad base to build height.

    5. Reward tracking: reward = exp(alpha*h) * (1 + beta*growth).
       The exponential term creates winner-take-all dynamics — teams
       on high knowledge peaks are disproportionately attractive.
       The growth term rewards active progress.  Feeds into social
       dynamics via _reward_social_nudge.

    6. Adaptive noise: productive particles (high reward) stay focused.
       Stuck particles (low reward) get increasing noise → explore.

    7. Diffuse knowledge field and sync positions to surface.

    Parameters
    ----------
    reward_ema : ndarray (N,) — modified in-place
        Per-particle exponential moving average of reward.
    noise_level : ndarray (N,) — modified in-place
        Per-particle current noise level (adaptive).
    """
    from .mountain_mesh import project_particles_to_knowledge_surface

    n = sim.n
    prefs = sim.prefs  # (N, K) — dims 0,1 are skills, dim 2 is social
    nbr_ids = sim.nbr_ids  # (N, M) from Phase 1
    valid = sim._valid_mask

    if nbr_ids is None:
        return

    write_rate = mountain_params['knowledge_write_rate']
    z_scale = mountain_params['z_scale']
    noise_base = mountain_params['gradient_noise_base']
    noise_max = mountain_params['gradient_noise_max']
    noise_adapt = mountain_params['noise_adapt_rate']
    ema_alpha = mountain_params['reward_ema_alpha']
    exp_scale = mountain_params['reward_exp_scale']
    growth_beta = mountain_params['reward_growth_beta']

    # ── 1. Knowledge gradient climbing ────────────────────────────────
    # Move toward known-good regions on the knowledge surface.
    # This is the ONLY lateral force for regular particles — they can
    # only see the knowledge manifold, not the hidden fitness landscape.
    climb_rate = mountain_params['knowledge_climb_rate']
    know_grad = knowledge_field.knowledge_gradient(prefs[:, 0], prefs[:, 1])
    know_mag = np.linalg.norm(know_grad, axis=1, keepdims=True)
    know_mag = np.maximum(know_mag, 1e-10)
    know_dir = know_grad / know_mag

    lateral_nudge = climb_rate * know_dir

    # ── 2. Visionary nudge (rare particles only) ──────────────────────
    # Only visionaries can sense the direction toward the global peak
    # of the hidden fitness landscape.  Everyone else is blind to it.
    vis_frac = mountain_params['visionary_fraction']
    vis_nudge = mountain_params['visionary_nudge']
    if vis_frac > 0 and vis_nudge > 0:
        # Deterministic visionary assignment based on particle index
        n_vis = max(1, int(n * vis_frac))
        vis_mask = np.zeros(n, dtype=bool)
        vis_mask[:n_vis] = True  # first n_vis particles are visionaries

        # Visionaries sense the hidden gradient at their position
        probes = np.zeros((n_vis, 3), dtype=np.float64)
        probes[:, 0] = prefs[vis_mask, 0]
        probes[:, 1] = prefs[vis_mask, 1]
        vis_grad = landscape.gradient(probes)  # (n_vis, 2)

        # Normalize to unit direction
        vis_mag = np.linalg.norm(vis_grad, axis=1, keepdims=True)
        vis_mag = np.maximum(vis_mag, 1e-10)
        vis_dir = vis_grad / vis_mag

        # Add visionary nudge to their lateral movement
        lateral_nudge[vis_mask] += vis_nudge * vis_dir

    # ── 3. Exploration: random jumps for stuck particles ──────────────
    explore_prob = mountain_params['explore_prob']
    explore_radius = mountain_params['explore_radius']
    if explore_prob > 0:
        explore_mask = rng.random(n) < explore_prob
        n_explore = explore_mask.sum()
        if n_explore > 0:
            jump = rng.normal(0, explore_radius, (n_explore, 2))
            lateral_nudge[explore_mask] += jump

    # Apply lateral drift to skill prefs (dims 0,1) — dim 2 untouched
    prefs[:, 0] += lateral_nudge[:, 0].astype(prefs.dtype)
    prefs[:, 1] += lateral_nudge[:, 1].astype(prefs.dtype)
    np.clip(prefs, -1, 1, out=prefs)

    # ── 4. Deposit knowledge ("push up") ────────────────────────────
    # The main research action: try to raise the knowledge manifold
    # at each particle's current position.
    # Amount is the write_rate — the structural support constraint
    # inside deposit_knowledge will limit how much actually sticks.
    amounts = np.full(n, write_rate, dtype=np.float64)
    actual_growth = knowledge_field.deposit_knowledge(
        prefs[:, 0], prefs[:, 1], amounts)

    # ── 5. Reward tracking ──────────────────────────────────────────
    # Combined reward = exp(alpha * h) * (1 + beta * growth_rate)
    # The exponential term creates winner-take-all dynamics: teams at
    # high knowledge peaks are disproportionately attractive.  The
    # growth_rate term gives a bonus to teams that are actively making
    # progress, not just sitting on old knowledge.
    # Normalize exp term to [0,1]: (exp(a*h) - 1) / (exp(a) - 1)
    heights = knowledge_field.sample_knowledge(prefs[:, 0], prefs[:, 1])
    if exp_scale > 0:
        exp_term = (np.exp(exp_scale * heights) - 1.0) / (np.exp(exp_scale) - 1.0)
    else:
        exp_term = heights  # linear fallback when scale = 0
    # Growth rate component: actual_growth is per-step, scale to [0,1]
    growth_term = 1.0 + growth_beta * np.clip(actual_growth / max(write_rate, 1e-10), 0.0, 1.0)
    raw_reward = exp_term * growth_term
    # Clamp to reasonable range before EMA
    np.clip(raw_reward, 0.0, 2.0, out=raw_reward)
    reward_ema[:] = (1.0 - ema_alpha) * reward_ema + ema_alpha * raw_reward

    # Store reward on the sim object so social dynamics can use it
    sim._reward_ema = reward_ema

    # ── 6. Adaptive noise ───────────────────────────────────────────
    # High reward (productive) → noise decreases toward base level
    # Low reward (stuck) → noise increases toward max level
    # This makes stuck teams explore laterally to find new territory.
    # Threshold: median reward — roughly half the particles are "stuck"
    reward_median = np.median(reward_ema)
    productive = reward_ema > reward_median
    stuck = ~productive

    # Productive: noise decays toward base
    noise_level[productive] += noise_adapt * (
        noise_base - noise_level[productive])
    # Stuck: noise grows toward max
    noise_level[stuck] += noise_adapt * (
        noise_max - noise_level[stuck])
    np.clip(noise_level, noise_base, noise_max, out=noise_level)

    # ── 7. Diffuse knowledge field ──────────────────────────────────
    knowledge_field.step_field()

    # ── 8. Sync 3D positions to the knowledge surface ───────────────
    projected = project_particles_to_knowledge_surface(
        knowledge_field, prefs, z_scale=z_scale)
    sim.pos[:, :3] = projected.astype(sim.pos.dtype)


if __name__ == '__main__':
    main()
