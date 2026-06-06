#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;

out vec3 fragPos;
out vec2 tc;

void main() {
    fragPos = pos;
    tc = texCoord;

    gl_Position = vec4(pos, 1.0);
}
