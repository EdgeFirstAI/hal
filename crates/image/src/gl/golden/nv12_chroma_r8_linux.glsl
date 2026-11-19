#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform vec3 y_coeffs;
uniform vec3 u_coeffs;
uniform vec3 v_coeffs;
uniform float y_offset;
uniform float c_offset;
out vec4 color;

uniform int chroma_row;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) - ivec2(0, chroma_row);
    ivec2 base = ivec2((p.x >> 1) << 1, p.y << 1);
    vec3 rgb = (texelFetch(tex, base, 0).rgb
        + texelFetch(tex, base + ivec2(1, 0), 0).rgb
        + texelFetch(tex, base + ivec2(0, 1), 0).rgb
        + texelFetch(tex, base + ivec2(1, 1), 0).rgb) * 0.25;
    vec3 coeffs = ((p.x & 1) == 0) ? u_coeffs : v_coeffs;
    float c = dot(coeffs, rgb) + c_offset;
    color = vec4(clamp(c, 0.0, 1.0), 0.0, 0.0, 1.0);
}
