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

in highp vec2 tc;

void main() {
    vec3 rgb = texture(tex, tc).rgb;
    float y = dot(y_coeffs, rgb) + y_offset;
    color = vec4(clamp(y, 0.0, 1.0), 0.0, 0.0, 1.0);
}
