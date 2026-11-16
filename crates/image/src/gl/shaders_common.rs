// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL shader sources shared across platform backends (macOS ANGLE + Linux).

// ---------------------------------------------------------------------------
// Portable `#version 300 es` fragment shaders shared verbatim by the Linux
// (PBO/DMA EGLImage) and macOS (IOSurface/ANGLE) render paths.
// ---------------------------------------------------------------------------

/// RGBA8 -> packed planar-float fragment shader (`PlanarRgb` / `PlanarRgba`,
/// F16 or F32).
///
/// The render target is a single `RGBA16F` (F16) or `RGBA32F` (F32) texture
/// sized `(W/4, C*H)` for a `C`-plane destination. Each output texel packs 4
/// float channel samples into its four components, so a linear readout of the
/// rendered texture produces a tightly-packed `[C, H, W]` buffer (CHW order,
/// one plane per row-band). Width `W` must be a multiple of 4.
///
/// The plane index comes from the surface geometry (`floor(oy / H)`), and
/// `pad_color` is a `vec4` whose fourth component pads plane 3.
///
/// Shared by the Linux (PBO/DMA), macOS (IOSurface/ANGLE) and Windows (D3D11
/// texture) render paths.
pub(crate) const PLANAR_RGB_F16_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec2 dst_image_size;  // (W, H) — destination plane size
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) within one plane
uniform vec4 pad_color;       // per-channel pad value as written (RGBA)
uniform vec4 norm_scale;      // output = sample * norm_scale + norm_offset
uniform vec4 norm_offset;
out vec4 frag;

// Sample one planar element. Returns the per-channel value at
// (in_plane_x, in_plane_y) for the given plane (0=R, 1=G, 2=B, 3=A).
// Pad value is returned when (in_plane_x, in_plane_y) is outside
// `dst_rect_px`.
float sample_planar_element(float in_plane_x, float in_plane_y, float plane) {
    bool inside_dst = (in_plane_x >= dst_rect_px.x) &&
                      (in_plane_x <  dst_rect_px.x + dst_rect_px.z) &&
                      (in_plane_y >= dst_rect_px.y) &&
                      (in_plane_y <  dst_rect_px.y + dst_rect_px.w);
    if (inside_dst) {
        float u = (in_plane_x - dst_rect_px.x) / dst_rect_px.z;
        float v = (in_plane_y - dst_rect_px.y) / dst_rect_px.w;
        vec2 src_uv = clamp(src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw,
                            src_extent.xy, src_extent.zw);
        vec4 rgba = texture(src, src_uv) * norm_scale + norm_offset;
        if (plane < 0.5) {
            return rgba.r;
        } else if (plane < 1.5) {
            return rgba.g;
        } else if (plane < 2.5) {
            return rgba.b;
        } else {
            return rgba.a;
        }
    } else {
        if (plane < 0.5) {
            return pad_color.r;
        } else if (plane < 1.5) {
            return pad_color.g;
        } else if (plane < 2.5) {
            return pad_color.b;
        } else {
            return pad_color.a;
        }
    }
}

void main() {
    // gl_FragCoord is at pixel center (n+0.5). Floor for the integer
    // index of the output pixel on the (W/4 × 3H) surface.
    int ox = int(floor(gl_FragCoord.x));
    int oy = int(floor(gl_FragCoord.y));

    float plane = floor(float(oy) / dst_image_size.y);
    float in_plane_y = float(oy) - plane * dst_image_size.y + 0.5;

    // Sample 4 consecutive in-plane x positions. Pixel center is
    // (x+0.5) so the first element of pixel `ox` starts at logical
    // tensor column `ox*4` — add 0.5 for the texel center sampled
    // by the bilinear filter.
    float base_x = float(ox * 4) + 0.5;
    float e0 = sample_planar_element(base_x + 0.0, in_plane_y, plane);
    float e1 = sample_planar_element(base_x + 1.0, in_plane_y, plane);
    float e2 = sample_planar_element(base_x + 2.0, in_plane_y, plane);
    float e3 = sample_planar_element(base_x + 3.0, in_plane_y, plane);

    // An RGBA16F attachment narrows each f32 to f16 on write; an RGBA32F
    // one keeps it. Either way the resulting byte layout in the surface is
    // exactly the [C, H, W] planar order the consumer expects.
    frag = vec4(e0, e1, e2, e3);
}
"#;

/// RGBA8 -> packed interleaved-float fragment shader (`Rgb`, F16 or F32).
///
/// The render target is a single `RGBA16F` (F16) or `RGBA32F` (F32) texture
/// sized `(W*3/4, H)`. Each output texel packs 4 consecutive elements of the
/// destination's `[H, W, 3]` stream into its four components, so a linear
/// readout produces a tightly-packed NHWC buffer. `W` must be a multiple of 4.
///
/// The destination element index gives the pixel and channel, `dst_rect_px`
/// decides whether that pixel is content or pad, and `src_rect_uv` maps
/// content onto the source. The source is sampled once per element.
pub(crate) const FLOAT_NHWC_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) in pixel space
uniform vec4 pad_color;       // per-channel pad value as written (RGBA)
uniform vec4 norm_scale;      // output = sample * norm_scale + norm_offset
uniform vec4 norm_offset;
out vec4 frag;

// Sample one element of the interleaved [H, W, 3] stream. `element` is the
// element's index within its row, `oy` the destination row; the pad value is
// returned when the element's pixel lies outside `dst_rect_px`.
float sample_interleaved_element(int element, int oy) {
    int px = element / 3;
    int channel = element - px * 3;
    float fx = float(px);
    float fy = float(oy);
    bool inside_dst = (fx >= dst_rect_px.x) &&
                      (fx <  dst_rect_px.x + dst_rect_px.z) &&
                      (fy >= dst_rect_px.y) &&
                      (fy <  dst_rect_px.y + dst_rect_px.w);
    if (!inside_dst) {
        return (channel == 0) ? pad_color.r
             : (channel == 1) ? pad_color.g : pad_color.b;
    }
    // Pixel center (+0.5) so LINEAR filtering resamples about the right
    // point, matching the F32 NHWC PBO shader.
    float u = (fx + 0.5 - dst_rect_px.x) / dst_rect_px.z;
    float v = (fy + 0.5 - dst_rect_px.y) / dst_rect_px.w;
    vec2 src_uv = clamp(src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw,
                        src_extent.xy, src_extent.zw);
    vec4 rgba = texture(src, src_uv) * norm_scale + norm_offset;
    return (channel == 0) ? rgba.r
         : (channel == 1) ? rgba.g : rgba.b;
}

void main() {
    // gl_FragCoord is at pixel center (n+0.5). Floor for the integer index
    // of the output texel on the (W*3/4 x H) surface.
    int ox = int(floor(gl_FragCoord.x));
    int oy = int(floor(gl_FragCoord.y));
    int base = ox * 4;
    frag = vec4(sample_interleaved_element(base + 0, oy),
                sample_interleaved_element(base + 1, oy),
                sample_interleaved_element(base + 2, oy),
                sample_interleaved_element(base + 3, oy));
}
"#;

/// RGBA8 -> float `Rgba` fragment shader: the float paths' crop and pad
/// contract at one texel per pixel.
///
/// The render target is an `RGBA16F` (F16) or `RGBA32F` (F32) texture sized
/// `(W, H)`. The whole fetched texel is the output, alpha included.
/// Addressing comes from `gl_FragCoord`.
pub(crate) const FLOAT_RGBA_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) in pixel space
uniform vec4 pad_color;       // per-channel pad value as written (RGBA)
uniform vec4 norm_scale;      // output = sample * norm_scale + norm_offset
uniform vec4 norm_offset;
out vec4 color;

void main() {
    float px = float(int(floor(gl_FragCoord.x)));
    float py = float(int(floor(gl_FragCoord.y)));
    bool inside_dst = (px >= dst_rect_px.x) &&
                      (px <  dst_rect_px.x + dst_rect_px.z) &&
                      (py >= dst_rect_px.y) &&
                      (py <  dst_rect_px.y + dst_rect_px.w);
    if (!inside_dst) {
        color = pad_color;
        return;
    }
    // Pixel center (+0.5), as in the packed float shaders.
    float u = (px + 0.5 - dst_rect_px.x) / dst_rect_px.z;
    float v = (py + 0.5 - dst_rect_px.y) / dst_rect_px.w;
    vec2 src_uv = clamp(src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw,
                        src_extent.xy, src_extent.zw);
    color = texture(src, src_uv) * norm_scale + norm_offset;
}
"#;

/// Fullscreen-quad vertex shader. Position is a `vec3` (NDC); passes
/// `fragPos`/`tc` to the fragment stage. The VBO feeds `pos` (location 0) and
/// `texCoord` (location 1).
pub(crate) const VERTEX_SHADER: &str = "\
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
";

// ---------------------------------------------------------------------------
// NV12/NV16/NV24 (semi-planar YUV, single R8 plane) -> RGBA. Decodes the
// combined-plane byte layout (`PixelFormat::chroma_layout` +
// `combined_plane_height`) with direct 2D `texelFetch` and a branchless
// `carry` for NV24's wrapping 2W-byte chroma row.

/// The colorimetry uniforms every NV program reads (see `nv_yuv_to_rgb_fn`).
macro_rules! nv_colorimetry {
    () => {
        "// Per-tensor colorimetry (YUV→RGB matrix + range), set by draw_nv_texture_2d
// from the source tensor's resolved colorimetry.
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
"
    };
}

/// `nv_yuv_to_rgb`: the exact YUV->RGB matrix every NV program applies.
macro_rules! nv_yuv_to_rgb_fn {
    () => {
        "// Floors expanded luma at 0; the top is not capped.
vec3 nv_yuv_to_rgb(float yv, float u, float v) {
    float yp = max((yv - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    return clamp(vec3(yp + c_vr * vp, yp - c_ug * up - c_vg * vp, yp + c_ub * up), 0.0, 1.0);
}

"
    };
}

/// Declarations shared by the NV->RGBA programs, ending with `nv_rgb(x, y)`:
/// the YUV->RGB conversion of the luma texel at (x, y) and its co-sited
/// chroma.
macro_rules! nv_rgba_header {
    () => {
        concat!(
            "#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int chroma_lines;
// Inclusive texel bounds (x0, y0, x1, y1) of the source crop: every tap is
// clamped to them.
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
",
            nv_colorimetry!(),
            "in vec3 fragPos;
in vec2 tc;
out vec4 color;

",
            nv_yuv_to_rgb_fn!(),
            "vec3 nv_rgb(int x, int y) {
    // Luma: direct 2D texel.
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

"
        )
    };
}

/// `main()` statements of the single-tap NV programs, for draws that map the
/// crop 1:1: reads the texel under the fragment and leaves it in `rgb`.
macro_rules! nv_rgb_nearest_body {
    () => {
        "    ivec2 i = clamp(ivec2(tc * vec2(img_size)), src_rect.xy, src_rect.zw);
    vec3 rgb = nv_rgb(i.x, i.y);
"
    };
}

/// `main()` statements of the bilinear NV programs, for draws that rescale
/// the crop: half-pixel-centred bilinear over the four texels' converted RGB,
/// with every tap clamped to the crop.
macro_rules! nv_rgb_bilinear_body {
    () => {
        "    vec2 p = tc * vec2(img_size) - 0.5;
    vec2 p0 = floor(p);
    vec2 f = p - p0;
    ivec2 i0 = clamp(ivec2(p0), src_rect.xy, src_rect.zw);
    ivec2 i1 = clamp(ivec2(p0) + 1, src_rect.xy, src_rect.zw);
    vec3 rgb = mix(mix(nv_rgb(i0.x, i0.y), nv_rgb(i1.x, i0.y), f.x),
                   mix(nv_rgb(i0.x, i1.y), nv_rgb(i1.x, i1.y), f.x), f.y);
"
    };
}

/// `main()` statements of the hardware-filtered bilinear NV programs: luma
/// from the combined plane and chroma from its (U, V) view, each through the
/// texture unit's LINEAR filter, then the YUV->RGB matrix.
macro_rules! nv_rgb_hw_bilinear_body {
    () => {
        "    vec2 luv = clamp(tc * luma_scale, luma_clamp.xy, luma_clamp.zw);
    vec2 cuv = clamp(tc * chroma_scale, chroma_clamp.xy, chroma_clamp.zw);
    vec2 c = texture(uv_tex, cuv).rg;
    vec3 rgb = nv_yuv_to_rgb(texture(src, luv).r, c.r, c.g);
"
    };
}

/// The integer output mapping every int8 program applies to its RGB output.
/// Built with `QUANT_MAPPED`, normalization and quantization are uniforms and
/// the result is stored as the low byte; without it, the plain `x ^ 0x80` bias.
macro_rules! int_quantize {
    () => {
        "#ifdef QUANT_MAPPED
uniform highp vec3 q_scale;   // q = clamp(floor(v * q_scale + q_offset), q_lo, q_hi)
uniform highp vec3 q_offset;
uniform highp vec3 q_lo;
uniform highp vec3 q_hi;
uniform highp vec3 q_zero;    // zero point / 255; q_wrap marks a negative stored value
uniform highp vec3 q_wrap;
highp vec3 quantize(highp vec3 v) {
    highp vec3 q = clamp(floor(v * q_scale + q_offset), q_lo, q_hi);
    return q * (1.0 / 255.0) + q_zero + step(q, q_wrap) * (256.0 / 255.0);
}
#else
// The plain XOR 0x80 bias, `(q + 128) mod 256`, as the CPU `byte ^ 0x80`.
highp vec3 quantize(highp vec3 v) {
    highp vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}
#endif

"
    };
}

/// Declarations shared by the YUYV->RGBA program and its int8 twin.
macro_rules! yuyv_rgba_2d_header {
    () => {
        r#"#version 300 es
precision highp float;
uniform highp sampler2D tex;
uniform vec2 src_size;
uniform vec4 src_extent;
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec3 fragPos;
in vec2 tc;
out vec4 color;

"#
    };
}

/// `main()` statements shared by the YUYV->RGBA program and its int8 twin:
/// reads `tc` and the uniforms above, leaves the pixel in `r`, `g`, `b`.
macro_rules! yuyv_rgb_body {
    () => {
        r#"    vec2 texel = vec2(1.0) / src_size;
    // `src_extent` keeps the texel index inside the logical image when the
    // texture is larger than it; `src_size` is the texture's own grid.
    vec2 col = floor(clamp(tc, src_extent.xy, src_extent.zw) * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    // The clamp bounds `col`, not the partner. On an odd logical width the
    // partner of the last column is one texel past the logical image.
    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;

    vec4 self_rg = texture(tex, self_uv);
    vec4 pair_rg = texture(tex, pair_uv);
    float y = self_rg.r;
    float u, v;
    if (even) { u = self_rg.g; v = pair_rg.g; }
    else      { v = self_rg.g; u = pair_rg.g; }

    // Identical matrix/range math to the NV program: floor the expanded
    // luma at 0 (limited-range footroom folds to 0; full range is a no-op
    // with y_offset=0/y_scale=1), top end left to the per-channel clamp.
    float yp = max((y - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
"#
    };
}

/// YUYV (sampled as a GL_RG texture: R=Y, G=alternating U/V) → RGBA.
/// Each output pixel samples its own texel and the horizontal partner to
/// recover both chroma components; the YUV→RGB matrix + range come from
/// the per-tensor colorimetry uniforms (same names as the NV program).
/// `src_size` is the texture's texel grid and `src_extent` the logical
/// image's rectangle on it (`render::sample_clamp_rect`).
pub(crate) const YUYV_RGBA_2D_FRAGMENT: &str = concat!(
    yuyv_rgba_2d_header!(),
    "void main() {\n",
    yuyv_rgb_body!(),
    "    color = vec4(r, g, b, 1.0);\n}\n"
);

/// Int8 variant of [`YUYV_RGBA_2D_FRAGMENT`]: the same pixel, with the XOR
/// 0x80 bias every int8 program applies (`(q + 128) mod 256`).
pub(crate) const YUYV_RGBA_2D_INT8_FRAGMENT: &str = concat!(
    yuyv_rgba_2d_header!(),
    int_quantize!(),
    "void main() {\n",
    yuyv_rgb_body!(),
    "    color = vec4(quantize(vec3(r, g, b)), 1.0);\n}\n"
);

/// NV->RGBA fragment shader for draws that do not rescale. Vertex stage
/// ([`VERTEX_SHADER`]) provides `fragPos`/`tc`; output is `color`.
pub(crate) const NV_RGBA_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    "void main() {\n",
    nv_rgb_nearest_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`NV_RGBA_FRAGMENT`].
pub(crate) const NV_RGBA_INT8_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    int_quantize!(),
    "void main() {\n",
    nv_rgb_nearest_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// NV->RGBA fragment shader for draws that rescale: bilinear.
pub(crate) const NV_RGBA_BILINEAR_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    "void main() {\n",
    nv_rgb_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// NV->RGBA fragment shader for draws that rescale, where the chroma plane
/// is bound as its own texture: bilinear through the texture unit, two
/// filtered fetches per pixel.
pub(crate) const NV_RGBA_HW_BILINEAR_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    "void main() {\n",
    nv_rgb_hw_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`NV_RGBA_HW_BILINEAR_FRAGMENT`].
pub(crate) const NV_RGBA_HW_BILINEAR_INT8_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    int_quantize!(),
    "void main() {\n",
    nv_rgb_hw_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// Header of the Y2Y NV programs: `GL_EXT_YUV_target` samples an NV12
/// import as raw (Y, U, V), filtered by the texture unit. `src_extent` is the
/// rectangle a sample may reach.
macro_rules! nv_y2y_header {
    () => {
        concat!(
            "#version 300 es
#extension GL_EXT_YUV_target : require
precision highp float;
uniform __samplerExternal2DY2YEXT src;
uniform highp vec4 src_extent;
",
            nv_colorimetry!(),
            "in vec3 fragPos;
in highp vec2 tc;
out vec4 color;

",
            nv_yuv_to_rgb_fn!()
        )
    };
}

/// `main()` statements of the Y2Y NV programs: one filtered fetch.
macro_rules! nv_rgb_y2y_body {
    () => {
        "    vec3 yuv = texture(src, clamp(tc, src_extent.xy, src_extent.zw)).rgb;
    vec3 rgb = nv_yuv_to_rgb(yuv.r, yuv.g, yuv.b);
"
    };
}

/// NV->RGBA fragment shader for draws that rescale, on GPUs with
/// `GL_EXT_YUV_target`: bilinear through the texture unit from a single NV12
/// import, one fetch per pixel.
pub(crate) const NV_RGBA_Y2Y_FRAGMENT: &str = concat!(
    nv_y2y_header!(),
    "void main() {\n",
    nv_rgb_y2y_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`NV_RGBA_Y2Y_FRAGMENT`].
pub(crate) const NV_RGBA_Y2Y_INT8_FRAGMENT: &str = concat!(
    nv_y2y_header!(),
    int_quantize!(),
    "void main() {\n",
    nv_rgb_y2y_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// Int8 variant of [`NV_RGBA_BILINEAR_FRAGMENT`].
pub(crate) const NV_RGBA_BILINEAR_INT8_FRAGMENT: &str = concat!(
    nv_rgba_header!(),
    int_quantize!(),
    "void main() {\n",
    nv_rgb_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

// ---------------------------------------------------------------------------
// Packed sources from an R8 view of the buffer: one R8 texture `row_stride`
// texels wide and `height` rows tall, each pixel assembled from its bytes
// through `texture()` taps at texel centres. A header supplies `src_px(x, y)`,
// the exact pixel, and `src_col(x, v)`, the pixel's bytes blended between rows
// by a LINEAR sampler; the sampling bodies below are shared by every header.

/// Declarations every packed R8-view program shares: the layout uniforms.
macro_rules! packed_r8_uniforms {
    () => {
        "#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
// Reciprocal of the R8 texture's size: (row stride in bytes, rows).
uniform vec2 inv_tex;
// Inclusive texel bounds (x0, y0, x1, y1) of the source crop: every tap is
// clamped to them.
uniform ivec4 src_rect;
"
    };
}

/// Packed RGB, three bytes per pixel.
macro_rules! rgb_r8_header {
    () => {
        concat!(
            packed_r8_uniforms!(),
            "in vec3 fragPos;
in vec2 tc;
out vec4 color;

vec3 src_px(int x, int y) {
    float b = float(x * 3) + 0.5;
    float v = (float(y) + 0.5) * inv_tex.y;
    return vec3(texture(src, vec2(b * inv_tex.x, v)).r,
                texture(src, vec2((b + 1.0) * inv_tex.x, v)).r,
                texture(src, vec2((b + 2.0) * inv_tex.x, v)).r);
}

// Under LINEAR filtering, at texture row coordinate `v`. Each tap sits 1/1024
// texel right of its byte's centre.
vec3 src_col(int x, float v) {
    float b = float(x * 3) + 0.5 + 1.0 / 1024.0;
    return vec3(texture(src, vec2(b * inv_tex.x, v)).r,
                texture(src, vec2((b + 1.0) * inv_tex.x, v)).r,
                texture(src, vec2((b + 2.0) * inv_tex.x, v)).r);
}

"
        )
    };
}

/// YUYV luma only, for a Grey destination: each pixel's Y byte (byte `2x`),
/// range-expanded, in every channel.
macro_rules! yuyv_luma_r8_header {
    () => {
        concat!(
            packed_r8_uniforms!(),
            "// Luma range expansion from the source's colorimetry, as in the NV
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

// Under LINEAR filtering, at texture row coordinate `v`, with each tap 1/1024
// texel right of its byte's centre.
vec3 src_col(int x, float v) {
    float b = float(x * 2) + 0.5 + 1.0 / 1024.0;
    return luma(texture(src, vec2(b * inv_tex.x, v)).r);
}

"
        )
    };
}

/// `main()` statements of the single-tap packed programs (a draw that maps
/// the crop 1:1, or nearest interpolation): the pixel under `tc`.
macro_rules! packed_r8_nearest_body {
    () => {
        "    ivec2 i = clamp(ivec2(tc * vec2(img_size)), src_rect.xy, src_rect.zw);
    vec3 rgb = src_px(i.x, i.y);
"
    };
}

/// `main()` statements of the shader-filtered bilinear packed programs:
/// half-pixel-centred bilinear from four exact pixels, every tap clamped to
/// the crop.
macro_rules! packed_r8_bilinear_body {
    () => {
        "    vec2 p = tc * vec2(img_size) - 0.5;
    vec2 p0 = floor(p);
    vec2 f = p - p0;
    ivec2 i0 = clamp(ivec2(p0), src_rect.xy, src_rect.zw);
    ivec2 i1 = clamp(ivec2(p0) + 1, src_rect.xy, src_rect.zw);
    vec3 rgb = mix(mix(src_px(i0.x, i0.y), src_px(i1.x, i0.y), f.x),
                   mix(src_px(i0.x, i1.y), src_px(i1.x, i1.y), f.x), f.y);
"
    };
}

/// `main()` statements of the hardware-assisted bilinear packed programs:
/// the texture unit's LINEAR filter blends rows within each byte column and
/// the shader blends the two columns. The row coordinate is clamped to the
/// crop's row centres.
macro_rules! packed_r8_hw_bilinear_body {
    () => {
        "    vec2 p = tc * vec2(img_size) - 0.5;
    float x0 = floor(p.x);
    float fx = p.x - x0;
    int xa = clamp(int(x0), src_rect.x, src_rect.z);
    int xb = clamp(int(x0) + 1, src_rect.x, src_rect.z);
    float v = (clamp(p.y, float(src_rect.y), float(src_rect.w)) + 0.5) * inv_tex.y;
    vec3 rgb = mix(src_col(xa, v), src_col(xb, v), fx);
"
    };
}

/// Packed-RGB->RGBA fragment shader for draws that do not rescale.
pub(crate) const RGB_R8_RGBA_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    "void main() {\n",
    packed_r8_nearest_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`RGB_R8_RGBA_FRAGMENT`].
pub(crate) const RGB_R8_RGBA_INT8_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_nearest_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// Packed-RGB->RGBA fragment shader for draws that rescale, filtered in the
/// shader.
pub(crate) const RGB_R8_RGBA_BILINEAR_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    "void main() {\n",
    packed_r8_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`RGB_R8_RGBA_BILINEAR_FRAGMENT`].
pub(crate) const RGB_R8_RGBA_BILINEAR_INT8_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// Packed-RGB->RGBA fragment shader for draws that rescale, with the rows
/// blended by the texture unit.
pub(crate) const RGB_R8_RGBA_HW_BILINEAR_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    "void main() {\n",
    packed_r8_hw_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`RGB_R8_RGBA_HW_BILINEAR_FRAGMENT`].
pub(crate) const RGB_R8_RGBA_HW_BILINEAR_INT8_FRAGMENT: &str = concat!(
    rgb_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_hw_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// YUYV->Grey fragment shader for draws that do not rescale.
pub(crate) const YUYV_LUMA_R8_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    "void main() {\n",
    packed_r8_nearest_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`YUYV_LUMA_R8_FRAGMENT`].
pub(crate) const YUYV_LUMA_R8_INT8_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_nearest_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// YUYV->Grey fragment shader for draws that rescale, filtered in the shader.
pub(crate) const YUYV_LUMA_R8_BILINEAR_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    "void main() {\n",
    packed_r8_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`YUYV_LUMA_R8_BILINEAR_FRAGMENT`].
pub(crate) const YUYV_LUMA_R8_BILINEAR_INT8_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

/// YUYV->Grey fragment shader for draws that rescale, with the rows blended
/// by the texture unit.
pub(crate) const YUYV_LUMA_R8_HW_BILINEAR_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    "void main() {\n",
    packed_r8_hw_bilinear_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`YUYV_LUMA_R8_HW_BILINEAR_FRAGMENT`].
pub(crate) const YUYV_LUMA_R8_HW_BILINEAR_INT8_FRAGMENT: &str = concat!(
    yuyv_luma_r8_header!(),
    int_quantize!(),
    "void main() {\n",
    packed_r8_hw_bilinear_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

// ---------------------------------------------------------------------------
// RGBA -> Grey: the second pass of a colour source into a Grey destination.
// Samples the first pass's RGBA texture 1:1 and writes BT.601 full-range luma.

macro_rules! luma_2d_header {
    () => {
        "#version 300 es
precision highp float;
uniform highp sampler2D tex;
in vec3 fragPos;
in highp vec2 tc;
out vec4 color;

"
    };
}

macro_rules! luma_2d_body {
    () => {
        "    vec3 rgb = vec3(dot(texture(tex, tc).rgb, vec3(0.299, 0.587, 0.114)));
"
    };
}

/// RGBA->Grey fragment shader (BT.601 luma in every channel).
pub(crate) const LUMA_2D_FRAGMENT: &str = concat!(
    luma_2d_header!(),
    "void main() {\n",
    luma_2d_body!(),
    "    color = vec4(rgb, 1.0);\n}\n"
);

/// Int8 variant of [`LUMA_2D_FRAGMENT`].
pub(crate) const LUMA_2D_INT8_FRAGMENT: &str = concat!(
    luma_2d_header!(),
    int_quantize!(),
    "void main() {\n",
    luma_2d_body!(),
    "    color = vec4(quantize(rgb), 1.0);\n}\n"
);

#[cfg(test)]
mod nv_shader_golden {
    /// The golden with CRLF normalized to LF.
    fn golden(src: &str) -> String {
        src.replace("\r\n", "\n")
    }

    /// The NV->RGBA shader sources match their `golden/*.glsl` references byte for byte.
    #[test]
    fn nv_fragment_byte_identical() {
        assert_eq!(
            super::NV_RGBA_FRAGMENT,
            golden(include_str!("golden/nv_rgba_linux.glsl")).as_str(),
            "NV->RGBA shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::NV_RGBA_BILINEAR_FRAGMENT,
            golden(include_str!("golden/nv_rgba_bilinear_linux.glsl")).as_str(),
            "bilinear NV->RGBA shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::NV_RGBA_Y2Y_FRAGMENT,
            golden(include_str!("golden/nv_rgba_y2y_linux.glsl")).as_str(),
            "Y2Y NV->RGBA shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::NV_RGBA_HW_BILINEAR_FRAGMENT,
            golden(include_str!("golden/nv_rgba_hw_bilinear_linux.glsl")).as_str(),
            "hardware-bilinear NV->RGBA shader bytes drifted from the on-target-validated golden"
        );
    }

    /// The packed-RGB R8 shader sources match their goldens byte for byte.
    #[test]
    fn rgb_r8_fragment_byte_identical() {
        assert_eq!(
            super::RGB_R8_RGBA_FRAGMENT,
            golden(include_str!("golden/rgb_r8_rgba_linux.glsl")).as_str(),
            "packed-RGB shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::RGB_R8_RGBA_BILINEAR_FRAGMENT,
            golden(include_str!("golden/rgb_r8_rgba_bilinear_linux.glsl")).as_str(),
            "bilinear packed-RGB shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::RGB_R8_RGBA_HW_BILINEAR_FRAGMENT,
            golden(include_str!("golden/rgb_r8_rgba_hw_bilinear_linux.glsl")).as_str(),
            "hardware-bilinear packed-RGB shader bytes drifted from the on-target-validated golden"
        );
    }

    /// The YUYV luma and RGBA->Grey shader sources match their goldens byte for byte.
    #[test]
    fn grey_fragment_byte_identical() {
        assert_eq!(
            super::YUYV_LUMA_R8_FRAGMENT,
            golden(include_str!("golden/yuyv_luma_r8_linux.glsl")).as_str(),
            "YUYV luma shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::YUYV_LUMA_R8_BILINEAR_FRAGMENT,
            golden(include_str!("golden/yuyv_luma_r8_bilinear_linux.glsl")).as_str(),
            "bilinear YUYV luma shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::YUYV_LUMA_R8_HW_BILINEAR_FRAGMENT,
            golden(include_str!("golden/yuyv_luma_r8_hw_bilinear_linux.glsl")).as_str(),
            "hardware-bilinear YUYV luma shader bytes drifted from the on-target-validated golden"
        );
        assert_eq!(
            super::LUMA_2D_FRAGMENT,
            golden(include_str!("golden/luma_2d_linux.glsl")).as_str(),
            "RGBA->Grey shader bytes drifted from the on-target-validated golden"
        );
    }

    #[test]
    fn vertex_byte_identical() {
        assert_eq!(
            super::VERTEX_SHADER,
            golden(include_str!("golden/vertex.glsl")).as_str(),
            "vertex shader bytes drifted from the on-target-validated golden"
        );
    }
}
