#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
// Reciprocal of the R8 texture's size: (row stride in bytes, rows).
uniform vec2 inv_tex;
// Inclusive texel bounds (x0, y0, x1, y1) of the source crop: every tap is
// clamped to them, so a crop never reads pixels from outside itself.
uniform ivec4 src_rect;
in vec3 fragPos;
in vec2 tc;
out vec4 color;

vec3 rgb_px(int x, int y) {
    float b = float(x * 3) + 0.5;
    float v = (float(y) + 0.5) * inv_tex.y;
    return vec3(texture(src, vec2(b * inv_tex.x, v)).r,
                texture(src, vec2((b + 1.0) * inv_tex.x, v)).r,
                texture(src, vec2((b + 2.0) * inv_tex.x, v)).r);
}

// Under LINEAR filtering, at texture row coordinate `v`. Each tap sits 1/1024
// texel right of its byte's centre so the horizontal weight quantizes to zero:
// a tap computed exactly at the centre can land a hair left of it and blend
// in a fraction of the previous byte, which belongs to another channel.
vec3 rgb_col(int x, float v) {
    float b = float(x * 3) + 0.5 + 1.0 / 1024.0;
    return vec3(texture(src, vec2(b * inv_tex.x, v)).r,
                texture(src, vec2((b + 1.0) * inv_tex.x, v)).r,
                texture(src, vec2((b + 2.0) * inv_tex.x, v)).r);
}

void main() {
    vec2 p = tc * vec2(img_size) - 0.5;
    vec2 p0 = floor(p);
    vec2 f = p - p0;
    ivec2 i0 = clamp(ivec2(p0), src_rect.xy, src_rect.zw);
    ivec2 i1 = clamp(ivec2(p0) + 1, src_rect.xy, src_rect.zw);
    vec3 rgb = mix(mix(rgb_px(i0.x, i0.y), rgb_px(i1.x, i0.y), f.x),
                   mix(rgb_px(i0.x, i1.y), rgb_px(i1.x, i1.y), f.x), f.y);
    color = vec4(rgb, 1.0);
}
