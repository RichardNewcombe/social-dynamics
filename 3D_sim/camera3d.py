"""
3D Orbit Camera + MVP Matrix
=============================

Spherical coordinate orbit camera with perspective projection.
Mouse left-drag orbits, scroll zooms.
"""

import math
import numpy as np


def _look_at(eye, target, up):
    """Build a 4x4 look-at view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]; m[0, 1] = s[1]; m[0, 2] = s[2]
    m[1, 0] = u[0]; m[1, 1] = u[1]; m[1, 2] = u[2]
    m[2, 0] = -f[0]; m[2, 1] = -f[1]; m[2, 2] = -f[2]
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_rad, aspect, near, far):
    """Build a 4x4 perspective projection matrix."""
    t = math.tan(fov_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * t)
    m[1, 1] = 1.0 / t
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -2.0 * far * near / (far - near)
    m[3, 2] = -1.0
    return m


def compute_mvp(azimuth, elevation, distance, target, aspect):
    """Compute model-view-projection matrix from orbit camera params.

    Args:
        azimuth: horizontal orbit angle (radians)
        elevation: vertical orbit angle (radians), clamped ±π/2
        distance: distance from target
        target: (3,) orbit center
        aspect: viewport width / height

    Returns:
        (4, 4) float32 MVP matrix
    """
    target = np.array(target, dtype=np.float32)

    # Spherical to Cartesian for eye position
    cos_el = math.cos(elevation)
    eye = target + distance * np.array([
        cos_el * math.sin(azimuth),
        math.sin(elevation),
        cos_el * math.cos(azimuth),
    ], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = _look_at(eye, target, up)
    proj = _perspective(math.radians(45.0), aspect, 0.01, 20.0)
    mvp = proj @ view
    # Transpose: NumPy is row-major, OpenGL expects column-major
    return np.ascontiguousarray(mvp.T)


class OrbitCamera:
    """3D orbit camera with mouse controls."""

    def __init__(self):
        self.azimuth = 0.0
        self.elevation = 0.3
        self.distance = 2.5
        self.target = [0.5, 0.5, 0.5]
        self._orbit_active = False
        self._prev_mouse = [0.0, 0.0]
        self._dirty = True

    def reset(self):
        self.azimuth = 0.0
        self.elevation = 0.3
        self.distance = 2.5
        self.target = [0.5, 0.5, 0.5]
        self._dirty = True

    @property
    def dirty(self):
        return self._dirty

    def clear_dirty(self):
        self._dirty = False

    def get_mvp(self, aspect):
        return compute_mvp(self.azimuth, self.elevation, self.distance,
                           self.target, aspect)

    def get_right_up(self):
        """Return camera right and up vectors for billboard rendering."""
        cos_el = math.cos(self.elevation)
        sin_el = math.sin(self.elevation)
        cos_az = math.cos(self.azimuth)
        sin_az = math.sin(self.azimuth)

        # Forward vector (from eye toward target)
        forward = np.array([-cos_el * sin_az, -sin_el, -cos_el * cos_az],
                           dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return right, up

    def on_scroll(self, yoffset):
        factor = 1.15 ** yoffset
        self.distance = max(0.5, min(self.distance / factor, 10.0))
        self._dirty = True

    def on_mouse_button(self, button, action, mx, my):
        """button: 0=left. action: 1=press, 0=release."""
        if button == 0:
            if action == 1:
                self._orbit_active = True
                self._prev_mouse = [mx, my]
            else:
                self._orbit_active = False

    def on_cursor_pos(self, mx, my, window_w, window_h):
        if self._orbit_active:
            dx = mx - self._prev_mouse[0]
            dy = my - self._prev_mouse[1]
            self.azimuth -= dx * 0.005
            self.elevation += dy * 0.005
            self.elevation = max(-math.pi / 2 + 0.01,
                                 min(math.pi / 2 - 0.01, self.elevation))
            self._dirty = True
        self._prev_mouse = [mx, my]
