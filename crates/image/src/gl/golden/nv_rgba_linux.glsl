#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int chroma_lines;
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

void main() {
    int w = img_size.x;
    int h = img_size.y;
    int x = clamp(int(tc.x * float(w)), 0, w - 1);
    int y = clamp(int(tc.y * float(h)), 0, h - 1);

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
    int cy = h + crow * chroma_lines + carry;
    int cx = ccol2 - carry * tex_width;
    float u = texelFetch(src, ivec2(cx, cy), 0).r;
    float v = texelFetch(src, ivec2(cx + 1, cy), 0).r;

    // Floor expanded luma at 0 to match the CPU `yuv` crate's saturating
    // (Y-16) term (limited footroom Y<16 → 0). The top is left uncapped — the
    // crate lets headroom exceed 1.0 and relies on the final RGB clamp, so the
    // GL path must too. No-op for full range (y_offset=0, y_scale=1).
    float yp = max((yv - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    color = vec4(r, g, b, 1.0);
}
