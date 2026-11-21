#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform vec3 u_coeffs;
uniform vec3 v_coeffs;
uniform float c_offset;
out vec4 color;

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) << 1;
    vec3 rgb = (texelFetch(tex, base, 0).rgb
        + texelFetch(tex, base + ivec2(1, 0), 0).rgb
        + texelFetch(tex, base + ivec2(0, 1), 0).rgb
        + texelFetch(tex, base + ivec2(1, 1), 0).rgb) * 0.25;
    vec2 uv = vec2(dot(u_coeffs, rgb), dot(v_coeffs, rgb)) + c_offset;
    color = vec4(clamp(uv, 0.0, 1.0), 0.0, 1.0);
}
