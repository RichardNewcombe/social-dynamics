"""
GLSL Shaders for 3D Particle Simulation
========================================

PARTICLE_VERT: 3D particle positions → MVP projection with perspective point size
BOX_VERT: 3D wireframe cube via MVP
MESH_VERT/MESH_FRAG: Triangle mesh surface with per-vertex color and diffuse lighting
OVERLAY_VERT: 2D overlay (divider line) — maps vec2 directly to NDC
Screen-space shaders (QUAD_VERT, TRAIL_FRAG, SPLAT_FRAG, DISPLAY_FRAG) unchanged from 2D.
"""

PARTICLE_VERT = '''
#version 410 core
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform mat4 mvp;
uniform vec2 viewport_offset;
uniform vec2 viewport_scale;
uniform float point_size;

void main() {
    vec4 clip = mvp * vec4(in_pos, 1.0);
    vec2 ndc = clip.xy / clip.w;
    ndc = ndc * viewport_scale + (viewport_offset * 2.0 - 1.0 + viewport_scale);
    gl_Position = vec4(ndc, clip.z / clip.w, 1.0);
    gl_PointSize = point_size / clip.w;
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

BOX_VERT = '''
#version 410 core
in vec3 in_pos;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
'''

BOX_FRAG = '''
#version 410 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
'''

LINE_VERT = '''
#version 410 core
in vec3 in_pos;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
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

# ── Mesh surface shaders (mountain + cost overlay) ──

MESH_VERT = '''
#version 410 core
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_color = in_color;
    v_normal = in_normal;
    v_world_pos = in_pos;
}
'''

MESH_FRAG = '''
#version 410 core
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 fragColor;
uniform float alpha;
uniform vec3 light_dir;  // normalized direction TO the light

void main() {
    vec3 n = normalize(v_normal);
    // Simple diffuse + ambient lighting
    float diffuse = max(dot(n, light_dir), 0.0);
    float ambient = 0.3;
    float lighting = ambient + 0.7 * diffuse;
    vec3 lit_color = v_color * lighting;
    fragColor = vec4(lit_color, alpha);
}
'''

# 2D overlay shader for the divider line (no MVP needed)
OVERLAY_VERT = '''
#version 410 core
in vec2 in_pos;
void main() {
    vec2 ndc = in_pos * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
'''

OVERLAY_FRAG = '''
#version 410 core
out vec4 fragColor;
uniform vec4 line_color;
void main() {
    fragColor = line_color;
}
'''
