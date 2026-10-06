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

    // Floor expanded luma at 0 to match the CPU `yuv` crate's saturating
    // (Y-16) term (limited footroom Y<16 → 0). The top is left uncapped — the
    // crate lets headroom exceed 1.0 and relies on the final RGB clamp, so the
    // GL path must too. No-op for full range (y_offset=0, y_scale=1).
    float yp = max((yv - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    return clamp(vec3(yp + c_vr * vp, yp - c_ug * up - c_vg * vp, yp + c_ub * up), 0.0, 1.0);
}

void main() {
    // Half-pixel-centred bilinear, the GL_LINEAR / OpenCV INTER_LINEAR
    // convention: this fragment's source position is tc * size - 0.5, with
    // edges clamped. `texelFetch` alone would be nearest-neighbour whenever
    // the convert rescales. Blending the four texels' converted RGB equals
    // converting at native resolution and then resizing, as the CPU backend
    // does; at 1:1 the fraction is zero and only one texel contributes.
    ivec2 last = img_size - 1;
    vec2 p = tc * vec2(img_size) - 0.5;
    vec2 p0 = floor(p);
    vec2 f = p - p0;
    ivec2 i0 = clamp(ivec2(p0), ivec2(0), last);
    ivec2 i1 = clamp(ivec2(p0) + 1, ivec2(0), last);
    vec3 rgb = mix(mix(nv_rgb(i0.x, i0.y), nv_rgb(i1.x, i0.y), f.x),
                   mix(nv_rgb(i0.x, i1.y), nv_rgb(i1.x, i1.y), f.x), f.y);
    color = vec4(rgb, 1.0);
}
