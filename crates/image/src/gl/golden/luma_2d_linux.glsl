#version 300 es
precision highp float;
uniform highp sampler2D tex;
in vec3 fragPos;
in highp vec2 tc;
out vec4 color;

void main() {
    vec3 rgb = vec3(dot(texture(tex, tc).rgb, vec3(0.299, 0.587, 0.114)));
    color = vec4(rgb, 1.0);
}
