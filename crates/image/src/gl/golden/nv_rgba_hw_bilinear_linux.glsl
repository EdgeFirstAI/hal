#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int chroma_lines;
// Inclusive texel bounds (x0, y0, x1, y1) of the source crop: every tap is
// clamped to them, so a crop never reads pixels from outside itself.
uniform ivec4 src_rect;
// Hardware-filtered program only: the chroma plane as (U, V) texels on unit 1,
// the image's share of each texture (`*_scale`), and the crop's half-texel-
// inset rectangle on each (`*_clamp`), which keeps the LINEAR kernel inside
// the crop and the luma rows.
uniform highp sampler2D uv_tex;
uniform vec2 luma_scale;
uniform vec4 luma_clamp;
uniform vec2 chroma_scale;
uniform vec4 chroma_clamp;
// Per-tensor colorimetry (YUV→RGB matrix + range), set by draw_nv_texture_2d
// from the source tensor's resolved colorimetry. Path B applies the matrix in
// the shader, so it is correct regardless of driver EGL color-hint support.
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec3 fragPos;
in vec2 tc;
out vec4 color;

// Floor expanded luma at 0 to match the CPU `yuv` crate's saturating (Y-16)
// term (limited footroom Y<16 → 0). The top is left uncapped — the crate lets
// headroom exceed 1.0 and relies on the final RGB clamp, so the GL path must
// too. No-op for full range (y_offset=0, y_scale=1).
vec3 nv_yuv_to_rgb(float yv, float u, float v) {
    float yp = max((yv - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    return clamp(vec3(yp + c_vr * vp, yp - c_ug * up - c_vg * vp, yp + c_ub * up), 0.0, 1.0);
}

vec3 nv_rgb(int x, int y) {
    // Luma: direct 2D texel — no per-pixel integer divide/modulo (very slow on
    // some embedded GPUs, e.g. Vivante GC7000UL).
    float yv = texelFetch(src, ivec2(x, y), 0).r;

    // Interleaved CbCr plane begins at buffer row `h`. Each image-chroma-row
    // spans `chroma_lines` R8 rows: NV24's 2W-byte row wraps once at tex_width
    // (carry); NV12/NV16 fit one row. `cx` is even so `cx+1` stays in-row.
    int ccol = x >> chroma_shift.x;
    int crow = y >> chroma_shift.y;
    int ccol2 = ccol * 2;
    int carry = ccol2 >= tex_width ? 1 : 0;
    int cy = img_size.y + crow * chroma_lines + carry;
    int cx = ccol2 - carry * tex_width;
    float u = texelFetch(src, ivec2(cx, cy), 0).r;
    float v = texelFetch(src, ivec2(cx + 1, cy), 0).r;
    return nv_yuv_to_rgb(yv, u, v);
}

void main() {
    vec2 luv = clamp(tc * luma_scale, luma_clamp.xy, luma_clamp.zw);
    vec2 cuv = clamp(tc * chroma_scale, chroma_clamp.xy, chroma_clamp.zw);
    vec2 c = texture(uv_tex, cuv).rg;
    vec3 rgb = nv_yuv_to_rgb(texture(src, luv).r, c.r, c.g);
    color = vec4(rgb, 1.0);
}
