#version 300 es
precision highp float;
uniform vec2 uv;
out vec4 color;

void main() {
    float c = ((int(gl_FragCoord.x) & 1) == 0) ? uv.x : uv.y;
    color = vec4(c, 0.0, 0.0, 1.0);
}
