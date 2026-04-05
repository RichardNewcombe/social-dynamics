#!/usr/bin/env python3
"""
ParticleNet Real-Time Visualizer
================================

Interactive visualization of the differentiable particle network from
particle_net.py.  Reuses the GLFW + moderngl + imgui rendering
infrastructure from sim_gpu_compute.py.

Features:
  - Train the model interactively (start / stop / step)
  - Live function plot: true curve vs predicted curve
  - Select an input x and see encoded particle positions + preference coloring
  - Step through simulation steps, watching positions and decoded output evolve
  - KNN neighbor connections overlay

Controls:
  Space       — toggle auto-training
  Left/Right  — step sim step prev/next
  R           — reset model weights
  Q / Esc     — quit
  Scroll      — zoom particle view (left half)
  Drag        — pan particle view (left half)

Dependencies:
  torch, numpy, glfw, moderngl, imgui-bundle, PyOpenGL
"""

# ── Imports ──────────────────────────────────────────────────────────
import argparse
import math
import time
import ctypes
import numpy as np

# Prevent duplicate GLFW library loading (same fix as sim_gpu_compute.py)
import os, site
for _p in [site.getusersitepackages()] + \
          (site.getsitepackages() if hasattr(site, 'getsitepackages') else []):
    _candidate = os.path.join(_p, 'imgui_bundle', 'libglfw.3.dylib')
    if os.path.isfile(_candidate):
        os.environ['PYGLFW_LIBRARY'] = _candidate
        break

import glfw
import moderngl
from imgui_bundle import imgui
import torch
import torch.nn as nn

# Import from particle_net.py (same directory)
from particle_net import (
    ParticleNet, _prefs_to_rgb, _resolve_device_dtype,
    make_sin_data, make_2d_data, SPACE,
)


# =====================================================================
# SHADERS (GLSL 4.10 Core)
# =====================================================================

# Particle circles with viewport transform + zoom/pan (from sim_gpu_compute.py)
PARTICLE_VERT = '''
#version 410 core
in vec2 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform vec2 viewport_offset;
uniform vec2 viewport_scale;
uniform vec2 view_center;
uniform float view_zoom;
uniform float point_size;

void main() {
    vec2 p = in_pos - view_center;
    p -= round(p);
    vec2 ndc = p * view_zoom * 2.0;
    ndc = ndc * viewport_scale + (viewport_offset * 2.0 - 1.0 + viewport_scale);
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
'''

PARTICLE_FRAG = '''
#version 410 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    if (dot(pc, pc) > 1.0) discard;
    fragColor = vec4(v_color, 1.0);
}
'''

# Lines in sim-space with view_center/view_zoom (from sim_gpu_compute.py)
BOX_VERT = '''
#version 410 core
in vec2 in_pos;
uniform vec2 view_center;
uniform float view_zoom;
void main() {
    vec2 ndc = (in_pos - view_center) * view_zoom * 2.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
'''

LINE_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''

# Data-space vertex shader for function plots
PLOT_VERT = '''
#version 410 core
in vec2 in_pos;
uniform vec2 data_min;
uniform vec2 data_max;
uniform float point_size;
void main() {
    vec2 ndc = ((in_pos - data_min) / (data_max - data_min)) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = point_size;
}
'''

# Plot lines: flat color
PLOT_LINE_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''

# Plot point: circle discard (reuse for (x, pred_y) dot)
PLOT_POINT_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    if (dot(pc, pc) > 1.0) discard;
    fragColor = line_color;
}
'''

# Divider line in NDC (no transform)
NDC_VERT = '''
#version 410 core
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
'''

NDC_LINE_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='ParticleNet Visualizer')
    parser.add_argument('--task', default='sin', choices=['sin', '2d'])
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
                        choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    device, dtype = _resolve_device_dtype(args.device)
    dtype_name = 'float64' if dtype == torch.float64 else 'float32'
    print(f"Device: {device}  Dtype: {dtype_name}")

    # ── Task setup ──
    if args.task == '2d':
        input_dim, output_dim = 2, 1
        make_data = make_2d_data
        task_name = 'sin(pi*x1)*cos(pi*x2)'
    else:
        input_dim, output_dim = 1, 1
        make_data = make_sin_data
        task_name = 'sin(x)'

    is_2d = (args.task == '2d')

    # ── Data ──
    x_train, y_train = make_data(1000, device, dtype)
    x_test, y_test = make_data(200, device, dtype)

    # ── Plot x range ──
    N_PLOT = 200
    if is_2d:
        x_plot_np = np.linspace(-1, 1, N_PLOT).astype(np.float32)
        x_plot_tensor = torch.stack([
            torch.linspace(-1, 1, N_PLOT, device=device, dtype=dtype),
            torch.zeros(N_PLOT, device=device, dtype=dtype),
        ], dim=1)
        y_true_np = np.sin(math.pi * x_plot_np).astype(np.float32)
        data_xmin, data_xmax = -1.0, 1.0
    else:
        x_plot_np = np.linspace(-math.pi, math.pi, N_PLOT).astype(np.float32)
        x_plot_tensor = torch.linspace(
            -math.pi, math.pi, N_PLOT, device=device, dtype=dtype
        ).unsqueeze(1)
        y_true_np = np.sin(x_plot_np).astype(np.float32)
        data_xmin, data_xmax = -math.pi, math.pi

    data_ymin, data_ymax = -1.5, 1.5  # y range for plot

    # ── Build model ──
    def create_model():
        return ParticleNet(
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

    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Task: {task_name}  Parameters: {n_params:,}")

    # ── Training state ──
    training = False
    train_epoch = 0
    train_loss = float('inf')
    test_loss = float('inf')
    lr = 1e-3
    epochs_per_frame = 5
    target_epochs = 500

    # ── Exploration state ──
    explore_x = 0.0
    explore_x2 = 0.0
    explore_step = 0

    # ── Trajectory cache ──
    pos_history = None
    prefs_history = None
    nbr_history = None
    cache_key = None
    decoded_y_at_step = 0.0

    # ── Free-run state ──
    # When active, particles are simulated live each frame from encoder output
    # instead of replaying the cached trajectory.
    freerun_active = False
    freerun_pos = None       # (1, N, 2) tensor on device
    freerun_prefs = None     # (1, N, K) tensor on device
    freerun_nbr_ids = None   # (1, N, K_nbr) tensor (detached)
    freerun_step = 0
    freerun_decoded_y = 0.0

    # ── Initialize GLFW ──
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    screen_w, screen_h = mode.size.width, mode.size.height
    WIN_H = screen_h - 130
    WIN_W = 2 * WIN_H
    if WIN_W > screen_w - 20:
        WIN_W = screen_w - 20
        WIN_H = WIN_W // 2

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(WIN_W, WIN_H,
                                "ParticleNet Visualizer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    print(f"Window: {WIN_W}x{WIN_H}  Framebuffer: {fb_w}x{fb_h}")

    ctx = moderngl.create_context(require=410)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ── imgui setup ──
    imgui_ctx = imgui.create_context()
    io = imgui.get_io()
    io.config_mac_osx_behaviors = True
    io.config_drag_click_to_input_text = True

    # ── View state ──
    view_center = [0.5, 0.5]
    view_zoom = 1.0
    pan_active = False
    prev_mouse_pos = [0.0, 0.0]
    point_size = 6.0
    show_neighbors = False

    # ── Callbacks ──
    def scroll_callback(win, xoffset, yoffset):
        nonlocal view_zoom
        if io.want_capture_mouse:
            return
        mx, my = glfw.get_cursor_pos(win)
        hw = WIN_W // 2
        if mx >= hw:
            return  # only zoom particle view (left half)
        ndc_x = mx / hw * 2.0 - 1.0
        ndc_y = 1.0 - my / WIN_H * 2.0
        factor = 1.15 ** yoffset
        new_zoom = max(1.0, min(view_zoom * factor, 20.0))
        view_center[0] += ndc_x / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_center[1] += ndc_y / 2.0 * (1.0 / view_zoom - 1.0 / new_zoom)
        view_zoom = new_zoom

    def mouse_button_callback(win, button, action, mods):
        nonlocal pan_active
        if io.want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                mx, my = glfw.get_cursor_pos(win)
                hw = WIN_W // 2
                if mx < hw:
                    pan_active = True
                    prev_mouse_pos[0] = mx
                    prev_mouse_pos[1] = my
            elif action == glfw.RELEASE:
                pan_active = False

    def cursor_pos_callback(win, mx, my):
        if pan_active and not io.want_capture_mouse:
            dx = mx - prev_mouse_pos[0]
            dy = my - prev_mouse_pos[1]
            hw = WIN_W // 2
            view_center[0] -= dx / hw / view_zoom
            view_center[1] += dy / WIN_H / view_zoom
        prev_mouse_pos[0] = mx
        prev_mouse_pos[1] = my

    def key_callback(win, key, scancode, action, mods):
        nonlocal training, explore_step
        if io.want_capture_keyboard:
            return
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                training = not training
            elif key == glfw.KEY_R:
                reset_model()
            elif key == glfw.KEY_RIGHT:
                explore_step = min(explore_step + 1, args.n_steps)
            elif key == glfw.KEY_LEFT:
                explore_step = max(explore_step - 1, 0)

    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_key_callback(window, key_callback)

    # Initialize imgui backends (AFTER our callbacks)
    imgui.backends.opengl3_init("#version 410")
    window_ptr = ctypes.cast(window, ctypes.c_void_p).value
    imgui.backends.glfw_init_for_opengl(window_ptr, True)

    # ── Compile shader programs ──
    N = args.n_particles

    # 1. Particle program (left half: colored circles)
    prog_particle = ctx.program(vertex_shader=PARTICLE_VERT,
                                fragment_shader=PARTICLE_FRAG)
    vbo_pos = ctx.buffer(reserve=N * 2 * 4)
    vbo_col = ctx.buffer(reserve=N * 3 * 4)
    vao_particle = ctx.vertex_array(prog_particle, [
        (vbo_pos, '2f', 'in_pos'),
        (vbo_col, '3f', 'in_color'),
    ])

    # 2. Line program (neighbor connections + divider in sim space)
    prog_line = ctx.program(vertex_shader=BOX_VERT, fragment_shader=LINE_FRAG)
    max_edges = N * args.n_neighbors
    vbo_nbr_lines = ctx.buffer(reserve=max_edges * 2 * 2 * 4)
    vao_nbr_lines = ctx.vertex_array(prog_line,
                                     [(vbo_nbr_lines, '2f', 'in_pos')])

    # 3. Plot program (function curves, axes, marker)
    prog_plot = ctx.program(vertex_shader=PLOT_VERT,
                            fragment_shader=PLOT_LINE_FRAG)

    # True curve VBO
    true_curve = np.column_stack([x_plot_np, y_true_np]).astype(np.float32)
    vbo_true_curve = ctx.buffer(true_curve.tobytes())
    vao_true_curve = ctx.vertex_array(prog_plot,
                                      [(vbo_true_curve, '2f', 'in_pos')])

    # Predicted curve VBO
    vbo_pred_curve = ctx.buffer(reserve=N_PLOT * 2 * 4)
    vao_pred_curve = ctx.vertex_array(prog_plot,
                                      [(vbo_pred_curve, '2f', 'in_pos')])

    # Axes and grid lines
    axes_lines = []
    # x-axis (y=0)
    axes_lines.extend([data_xmin, 0.0, data_xmax, 0.0])
    # y-axis (x=0)
    axes_lines.extend([0.0, data_ymin, 0.0, data_ymax])
    # horizontal grid
    for yg in [-1.0, -0.5, 0.5, 1.0]:
        axes_lines.extend([data_xmin, yg, data_xmax, yg])
    # vertical grid
    if is_2d:
        for xg in [-0.5, 0.5]:
            axes_lines.extend([xg, data_ymin, xg, data_ymax])
    else:
        for xg in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
            if data_xmin <= xg <= data_xmax:
                axes_lines.extend([xg, data_ymin, xg, data_ymax])
    axes_np = np.array(axes_lines, dtype=np.float32)
    vbo_axes = ctx.buffer(axes_np.tobytes())
    vao_axes = ctx.vertex_array(prog_plot, [(vbo_axes, '2f', 'in_pos')])
    n_axes_verts = len(axes_lines) // 2

    # Vertical marker at selected x
    vbo_marker = ctx.buffer(reserve=2 * 2 * 4)
    vao_marker = ctx.vertex_array(prog_plot, [(vbo_marker, '2f', 'in_pos')])

    # 4. Plot point program (current (x, pred_y) dot)
    prog_plot_point = ctx.program(vertex_shader=PLOT_VERT,
                                  fragment_shader=PLOT_POINT_FRAG)
    vbo_plot_point = ctx.buffer(reserve=1 * 2 * 4)
    vao_plot_point = ctx.vertex_array(prog_plot_point,
                                      [(vbo_plot_point, '2f', 'in_pos')])

    # 5. NDC line program (for divider)
    prog_ndc_line = ctx.program(vertex_shader=NDC_VERT,
                                fragment_shader=NDC_LINE_FRAG)
    divider_data = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
    vbo_divider = ctx.buffer(divider_data.tobytes())
    vao_divider = ctx.vertex_array(prog_ndc_line,
                                   [(vbo_divider, '2f', 'in_pos')])

    # ── Helper: set plot uniforms ──
    def set_plot_uniforms(prog):
        prog['data_min'] = (data_xmin, data_ymin)
        prog['data_max'] = (data_xmax, data_ymax)

    # ── Model reset ──
    def reset_model():
        nonlocal model, optimizer, train_epoch, train_loss, test_loss
        nonlocal training, cache_key
        nonlocal freerun_active, freerun_pos, freerun_prefs, freerun_nbr_ids
        nonlocal freerun_step, freerun_decoded_y
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_epoch = 0
        train_loss = float('inf')
        test_loss = float('inf')
        training = False
        cache_key = None
        freerun_active = False
        freerun_pos = None
        freerun_prefs = None
        freerun_nbr_ids = None
        freerun_step = 0
        freerun_decoded_y = 0.0

    # ── Predicted curve update ──
    def update_pred_curve():
        model.eval()
        with torch.no_grad():
            y_hat = model(x_plot_tensor).cpu().numpy().ravel().astype(np.float32)
        pred_data = np.column_stack([x_plot_np, y_hat]).astype(np.float32)
        vbo_pred_curve.write(pred_data.tobytes())

    # ── Trajectory cache ──
    def update_trajectory():
        nonlocal pos_history, prefs_history, nbr_history, cache_key
        if is_2d:
            x_input = torch.tensor(
                [[explore_x, explore_x2]], device=device, dtype=dtype)
        else:
            x_input = torch.tensor(
                [[explore_x]], device=device, dtype=dtype)
        model.eval()
        pos_history, prefs_history, nbr_history, _ = \
            model.run_with_trajectory(x_input)
        cache_key = (explore_x, explore_x2 if is_2d else 0.0, train_epoch)

    # ── Decode at intermediate step ──
    def decode_at_step(step):
        if pos_history is None:
            return 0.0
        pos_t = torch.tensor(
            pos_history[step], device=device, dtype=dtype)  # (1, N, 2)
        prefs_t = torch.tensor(
            prefs_history[step], device=device, dtype=dtype)  # (1, N, K)
        state = torch.cat([
            pos_t.reshape(1, -1), prefs_t.reshape(1, -1)
        ], dim=1)
        model.eval()
        with torch.no_grad():
            y = model.decoder(state).cpu().numpy()[0]
        return float(y[0]) if y.size > 0 else 0.0

    # ── Compute true y for current input ──
    def true_y_at_input():
        if is_2d:
            return float(math.sin(math.pi * explore_x) *
                         math.cos(math.pi * explore_x2))
        else:
            return float(math.sin(explore_x))

    # ── Compute predicted y for current input ──
    def pred_y_at_input():
        if is_2d:
            x_in = torch.tensor(
                [[explore_x, explore_x2]], device=device, dtype=dtype)
        else:
            x_in = torch.tensor(
                [[explore_x]], device=device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            return float(model(x_in).cpu().numpy().ravel()[0])

    # ── Build neighbor line segments ──
    def build_neighbor_lines(step):
        """Build periodic-aware neighbor line segments for the given step."""
        if nbr_history is None or step < 1:
            return np.zeros((0, 2), dtype=np.float32)
        # Neighbors from step-1 produced the positions at step
        nbr_ids = nbr_history[step - 1][0]   # (N, K_nbr)
        pos_np = pos_history[step][0]         # (N, 2)
        L = SPACE
        segments = []
        N_p = pos_np.shape[0]
        for i in range(N_p):
            for j_idx in range(nbr_ids.shape[1]):
                j = nbr_ids[i, j_idx]
                d = pos_np[j] - pos_np[i]
                d = d - L * np.round(d / L)
                end = pos_np[i] + d
                segments.append(pos_np[i])
                segments.append(end)
        if not segments:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(segments, dtype=np.float32)

    # ── Free-run helpers ──
    def freerun_reset():
        """Encode current x input and initialize free-run pos/prefs."""
        nonlocal freerun_pos, freerun_prefs, freerun_nbr_ids, freerun_step
        nonlocal freerun_decoded_y
        if is_2d:
            x_input = torch.tensor(
                [[explore_x, explore_x2]], device=device, dtype=dtype)
        else:
            x_input = torch.tensor(
                [[explore_x]], device=device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            pos, prefs = model._encode_state(x_input)
        freerun_pos = pos        # (1, N, 2)
        freerun_prefs = prefs    # (1, N, K)
        freerun_nbr_ids = None
        freerun_step = 0
        freerun_decoded_y = _decode_from_state(pos, prefs)

    def freerun_physics_step():
        """Run one physics step on the live free-run state."""
        nonlocal freerun_pos, freerun_prefs, freerun_nbr_ids, freerun_step
        nonlocal freerun_decoded_y
        if freerun_pos is None:
            return
        model.eval()
        with torch.no_grad():
            nbr_ids = model._find_neighbors(freerun_pos)
            freerun_nbr_ids = nbr_ids
            freerun_pos, freerun_prefs = model._physics_step(
                freerun_pos, freerun_prefs, nbr_ids)
        freerun_step += 1
        freerun_decoded_y = _decode_from_state(freerun_pos, freerun_prefs)

    def _decode_from_state(pos_t, prefs_t):
        """Decode output from a (1, N, 2) pos and (1, N, K) prefs tensor."""
        state = torch.cat([
            pos_t.reshape(1, -1), prefs_t.reshape(1, -1)
        ], dim=1)
        model.eval()
        with torch.no_grad():
            y = model.decoder(state).cpu().numpy()[0]
        return float(y[0]) if y.size > 0 else 0.0

    def freerun_build_neighbor_lines():
        """Build neighbor lines from live free-run state."""
        if freerun_pos is None or freerun_nbr_ids is None:
            return np.zeros((0, 2), dtype=np.float32)
        pos_np = freerun_pos[0].cpu().numpy()       # (N, 2)
        nbr_ids = freerun_nbr_ids[0].cpu().numpy()  # (N, K_nbr)
        L = SPACE
        segments = []
        N_p = pos_np.shape[0]
        for i in range(N_p):
            for j_idx in range(nbr_ids.shape[1]):
                j = nbr_ids[i, j_idx]
                d = pos_np[j] - pos_np[i]
                d = d - L * np.round(d / L)
                end = pos_np[i] + d
                segments.append(pos_np[i].astype(np.float32))
                segments.append(end.astype(np.float32))
        if not segments:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(segments, dtype=np.float32)

    # ── FPS tracking ──
    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0

    # Initial state
    update_pred_curve()
    update_trajectory()

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── Training ──
        if training and train_epoch < target_epochs:
            model.train()
            for _ in range(epochs_per_frame):
                if train_epoch >= target_epochs:
                    training = False
                    break
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_epoch += 1

                if train_epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        test_loss = loss_fn(
                            model(x_test), y_test).item()

            # Invalidate cache and update predicted curve
            cache_key = None
            update_pred_curve()

        # ── Cache check (only when not free-running) ──
        current_key = (explore_x, explore_x2 if is_2d else 0.0, train_epoch)
        if not freerun_active and cache_key != current_key:
            update_trajectory()
            update_pred_curve()

        # ── Free-run physics ──
        if freerun_active:
            if freerun_pos is None:
                freerun_reset()
            freerun_physics_step()

        # ── Prepare current step data ──
        if freerun_active and freerun_pos is not None:
            # Use live free-run state
            pos_np = freerun_pos[0].detach().cpu().numpy().astype(np.float32)
            prefs_np = freerun_prefs[0].detach().cpu().numpy()
            colors_np = _prefs_to_rgb(prefs_np).astype(np.float32)

            vbo_pos.write(pos_np.tobytes())
            vbo_col.write(colors_np.tobytes())

            decoded_y_at_step = freerun_decoded_y
        elif pos_history is not None:
            step = min(explore_step, len(pos_history) - 1)
            pos_np = pos_history[step][0].astype(np.float32)     # (N, 2)
            prefs_np = prefs_history[step][0]                     # (N, K)
            colors_np = _prefs_to_rgb(prefs_np).astype(np.float32)  # (N, 3)

            vbo_pos.write(pos_np.tobytes())
            vbo_col.write(colors_np.tobytes())

            decoded_y_at_step = decode_at_step(step)
        else:
            decoded_y_at_step = 0.0

        # ── Update marker VBO ──
        marker_x = explore_x
        marker_data = np.array([
            marker_x, data_ymin,
            marker_x, data_ymax,
        ], dtype=np.float32)
        vbo_marker.write(marker_data.tobytes())

        # ── Update plot point VBO ──
        pred_y = pred_y_at_input()
        plot_pt = np.array([marker_x, pred_y], dtype=np.float32)
        vbo_plot_point.write(plot_pt.tobytes())

        # ── Get current framebuffer size (handles resize) ──
        fb_w, fb_h = glfw.get_framebuffer_size(window)

        # ================================================================
        # RENDER
        # ================================================================
        ctx.screen.use()
        ctx.clear(0.08, 0.08, 0.1)

        # ── LEFT HALF: Particle view ──
        ctx.viewport = (0, 0, fb_w // 2, fb_h)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Optional neighbor lines
        if show_neighbors:
            if freerun_active and freerun_pos is not None:
                lines = freerun_build_neighbor_lines()
            elif pos_history is not None and explore_step > 0:
                lines = build_neighbor_lines(explore_step)
            else:
                lines = np.zeros((0, 2), dtype=np.float32)
            n_line_verts = len(lines)
            if n_line_verts > 0:
                needed = n_line_verts * 2 * 4
                if needed > vbo_nbr_lines.size:
                    vbo_nbr_lines = ctx.buffer(reserve=needed)
                    vao_nbr_lines = ctx.vertex_array(
                        prog_line, [(vbo_nbr_lines, '2f', 'in_pos')])
                vbo_nbr_lines.write(lines.tobytes())
                prog_line['view_center'] = tuple(view_center)
                prog_line['view_zoom'] = view_zoom
                prog_line['line_color'] = (1.0, 1.0, 1.0, 0.15)
                vao_nbr_lines.render(moderngl.LINES, vertices=n_line_verts)

        # Particles
        prog_particle['viewport_offset'] = (0.0, 0.0)
        prog_particle['viewport_scale'] = (1.0, 1.0)
        prog_particle['view_center'] = tuple(view_center)
        prog_particle['view_zoom'] = view_zoom
        prog_particle['point_size'] = point_size
        vao_particle.render(moderngl.POINTS)

        # ── RIGHT HALF: Function plot ──
        ctx.viewport = (fb_w // 2, 0, fb_w // 2, fb_h)

        # Axes (dark gray)
        set_plot_uniforms(prog_plot)
        prog_plot['point_size'] = 1.0
        prog_plot['line_color'] = (0.3, 0.3, 0.3, 0.5)
        vao_axes.render(moderngl.LINES, vertices=n_axes_verts)

        # True curve (blue)
        prog_plot['line_color'] = (0.3, 0.5, 1.0, 1.0)
        vao_true_curve.render(moderngl.LINE_STRIP)

        # Predicted curve (orange)
        prog_plot['line_color'] = (1.0, 0.6, 0.2, 1.0)
        vao_pred_curve.render(moderngl.LINE_STRIP)

        # Vertical marker (yellow)
        prog_plot['line_color'] = (1.0, 1.0, 0.3, 0.6)
        vao_marker.render(moderngl.LINES, vertices=2)

        # Current (x, pred_y) dot (yellow)
        set_plot_uniforms(prog_plot_point)
        prog_plot_point['point_size'] = 10.0
        prog_plot_point['line_color'] = (1.0, 1.0, 0.3, 1.0)
        vao_plot_point.render(moderngl.POINTS, vertices=1)

        # ── DIVIDER LINE ──
        ctx.viewport = (0, 0, fb_w, fb_h)
        prog_ndc_line['line_color'] = (1.0, 1.0, 1.0, 0.4)
        vao_divider.render(moderngl.LINES, vertices=2)

        # ── imgui overlay ──
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        imgui.set_next_window_pos(imgui.ImVec2(10, 10),
                                  imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(310, 520),
                                   imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_bg_alpha(0.85)
        imgui.begin("Controls")

        # ── Status ──
        imgui.text(f"Epoch: {train_epoch}   FPS: {fps:.0f}")
        imgui.text(f"Train Loss: {train_loss:.6f}")
        imgui.text(f"Test Loss:  {test_loss:.6f}")
        imgui.separator()

        # ── Training controls ──
        if imgui.button("Train 10", imgui.ImVec2(75, 0)):
            # Run 10 epochs immediately
            model.train()
            for _ in range(10):
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_epoch += 1
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(x_test), y_test).item()
            cache_key = None
            update_pred_curve()

        imgui.same_line()
        if imgui.button("Train 100", imgui.ImVec2(75, 0)):
            model.train()
            for _ in range(100):
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_epoch += 1
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(x_test), y_test).item()
            cache_key = None
            update_pred_curve()

        imgui.same_line()
        label = "Stop" if training else "Auto Train"
        if imgui.button(label, imgui.ImVec2(85, 0)):
            training = not training

        changed, v = imgui.drag_int("Epochs/frame", epochs_per_frame,
                                    0.5, 1, 50)
        if changed:
            epochs_per_frame = v

        changed, v = imgui.drag_float("Learning Rate", lr,
                                      0.00001, 1e-5, 1e-1, "%.1e")
        if changed:
            lr = v
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        changed, v = imgui.drag_int("Target Epochs", target_epochs,
                                    5.0, 10, 10000)
        if changed:
            target_epochs = v

        if imgui.button("Reset Model", imgui.ImVec2(120, 0)):
            reset_model()
            update_pred_curve()
            update_trajectory()

        imgui.separator()

        # ── Input Exploration ──
        if imgui.collapsing_header(
                "Input Exploration",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            if is_2d:
                changed, v = imgui.slider_float("Input x1", explore_x,
                                                -1.0, 1.0, "%.3f")
                if changed:
                    explore_x = v
                    cache_key = None

                changed, v = imgui.slider_float("Input x2", explore_x2,
                                                -1.0, 1.0, "%.3f")
                if changed:
                    explore_x2 = v
                    cache_key = None
            else:
                changed, v = imgui.slider_float("Input x", explore_x,
                                                -math.pi, math.pi, "%.3f")
                if changed:
                    explore_x = v
                    cache_key = None

            true_y = true_y_at_input()
            imgui.text(f"True y:      {true_y:.4f}")
            imgui.text(f"Predicted y: {pred_y:.4f}")
            imgui.separator()

            changed, v = imgui.slider_int("Sim Step", explore_step,
                                          0, args.n_steps)
            if changed:
                explore_step = v

            if imgui.button("<< Prev", imgui.ImVec2(70, 0)):
                explore_step = max(explore_step - 1, 0)
            imgui.same_line()
            if imgui.button("Next >>", imgui.ImVec2(70, 0)):
                explore_step = min(explore_step + 1, args.n_steps)
            imgui.same_line()
            if imgui.button("Reset##step", imgui.ImVec2(50, 0)):
                explore_step = 0

            imgui.text(f"Decoded at step {explore_step}: "
                       f"{decoded_y_at_step:.4f}")

            imgui.separator()

            # ── Free Run controls ──
            imgui.text("Free Run")
            run_label = "Stop##freerun" if freerun_active else "Free Run"
            if imgui.button(run_label, imgui.ImVec2(85, 0)):
                if not freerun_active:
                    freerun_active = True
                    freerun_reset()
                else:
                    freerun_active = False
            imgui.same_line()
            if imgui.button("Reset (Encode)", imgui.ImVec2(110, 0)):
                freerun_reset()
                if not freerun_active:
                    # Also update the trajectory cache display
                    cache_key = None
            if freerun_active or freerun_pos is not None:
                imgui.text(f"Free-run step: {freerun_step}")
                imgui.text(f"Free-run decoded: {freerun_decoded_y:.4f}")

        # ── View controls ──
        if imgui.collapsing_header(
                "View",
                flags=int(imgui.TreeNodeFlags_.default_open.value)):
            changed, v = imgui.drag_float("Point Size", point_size,
                                          0.1, 1.0, 20.0, "%.1f")
            if changed:
                point_size = v

            changed, v = imgui.checkbox("Show Neighbors", show_neighbors)
            if changed:
                show_neighbors = v

            if imgui.button("Reset View", imgui.ImVec2(100, 0)):
                view_center[0] = 0.5
                view_center[1] = 0.5
                view_zoom = 1.0

            if view_zoom != 1.0:
                imgui.text(f"Zoom: {view_zoom:.1f}x")

        # ── Model info ──
        if imgui.collapsing_header("Model Info"):
            imgui.text(f"N={args.n_particles}  K={args.k_dims}  "
                       f"T={args.n_steps}")
            imgui.text(f"nbr={args.n_neighbors}  "
                       f"step={args.step_size}  "
                       f"rep={args.repulsion}")
            imgui.text(f"social={args.social}")
            imgui.text(f"Parameters: {n_params:,}")
            imgui.text(f"Device: {device}  Dtype: {dtype_name}")

        imgui.end()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        # ── Swap and FPS ──
        glfw.swap_buffers(window)

        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        status = "Training" if training else ("Free Run" if freerun_active else "Idle")
        glfw.set_window_title(
            window,
            f"ParticleNet [{status}] Epoch:{train_epoch} "
            f"Loss:{train_loss:.4f} FPS:{fps:.0f}")

    # ── Cleanup ──
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context(imgui_ctx)
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == '__main__':
    main()
