"""
GLSL shaders for the 2D particle simulation (OpenGL 4.1 Core Profile).
"""

# ── Particle point rendering with toroidal wrap + pan/zoom ──
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

# ── Full-screen quad (trail decay + display) ──
QUAD_VERT = '''
#version 410 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
'''

TRAIL_FRAG = '''
#version 410 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D trail_tex;
uniform float decay;
void main() {
    fragColor = vec4(texture(trail_tex, v_uv).rgb * decay, 1.0);
}
'''

# ── Splat (additive soft circle) ──
SPLAT_FRAG = '''
#version 410 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(pc, pc);
    if (r2 > 1.0) discard;
    float alpha = 1.0 - r2;
    fragColor = vec4(v_color * alpha, alpha);
}
'''

# ── Display trail texture with zoom/pan and HDR clamp ──
DISPLAY_FRAG = '''
#version 410 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D tex;
uniform vec2 view_center;
uniform float view_zoom;
void main() {
    vec2 uv = (v_uv - 0.5) / view_zoom + view_center;
    vec3 c = texture(tex, uv).rgb;
    float m = max(c.r, max(c.g, c.b));
    if (m > 1.0) c /= m;
    fragColor = vec4(c, 1.0);
}
'''

# ── Box wireframe (sim-space) ──
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

BOX_FRAG = '''
#version 410 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
'''

# ── Uniform-color lines (neighbors, radius circles, divider) ──
LINE_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''
