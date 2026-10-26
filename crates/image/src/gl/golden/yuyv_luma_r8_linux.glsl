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
// Luma range expansion from the source's colorimetry, as in the NV
// programs: limited range maps 16..235 onto 0..1, full range is unchanged.
uniform float y_offset;
uniform float y_scale;
in vec3 fragPos;
in vec2 tc;
out vec4 color;

vec3 luma(float y) {
    return vec3(clamp((y - y_offset) * y_scale, 0.0, 1.0));
}

vec3 src_px(int x, int y) {
    float b = float(x * 2) + 0.5;
    return luma(texture(src, vec2(b * inv_tex.x, (float(y) + 0.5) * inv_tex.y)).r);
}

// Under LINEAR filtering, at texture row coordinate `v`, nudged off the byte
// centre as in the packed-RGB header so no chroma byte blends in.
vec3 src_col(int x, float v) {
    float b = float(x * 2) + 0.5 + 1.0 / 1024.0;
    return luma(texture(src, vec2(b * inv_tex.x, v)).r);
}

void main() {
    ivec2 i = clamp(ivec2(tc * vec2(img_size)), src_rect.xy, src_rect.zw);
    vec3 rgb = src_px(i.x, i.y);
    color = vec4(rgb, 1.0);
}
