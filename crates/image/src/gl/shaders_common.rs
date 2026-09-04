// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL shader sources shared across platform backends (macOS ANGLE + Linux).

// ---------------------------------------------------------------------------
// Portable `#version 300 es` fragment shaders shared verbatim by both the
// Linux (PBO/DMA EGLImage) and macOS (IOSurface/ANGLE) render paths, so the
// GLSL bytes are validated identically on every backend.
//
// (YUV→RGB matrix conversion lives in the per-backend NV shaders — Linux
// `shaders::generate_nv_to_rgba_shader_2d` and the macOS `NV_TO_RGBA_FRAGMENT`,
// each carrying the six per-tensor colorimetry uniforms from
// `crate::colorimetry::yuv_to_rgb_coeffs`. Packed YUYV uses the driver Path A
// EGL color-space/range hints. Only the colorimetry-agnostic float packers
// are shared here.)
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
/// The plane index comes from the surface geometry (`floor(oy / H)`), not from
/// a uniform, so a `PlanarRgba` surface of height `4*H` selects plane 3 on its
/// own; the sampler carries a branch for each of the four planes and
/// `pad_color` is a `vec4` so plane 3 pads with the same alpha the CPU
/// reference writes.
///
/// This is the single source of truth consumed by the Linux (PBO/DMA), macOS
/// (IOSurface/ANGLE) and Windows (D3D11 texture) render paths. The GLSL bytes
/// are validated against the CPU converter by the on-target F16 round-trip
/// tests, the macOS IOSurface integration tests and
/// `windows_float_destinations_match_the_cpu_converter_for_every_layout`.
pub(crate) const PLANAR_RGB_F16_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec2 dst_image_size;  // (W, H) — destination plane size
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) within one plane
uniform vec4 pad_color;       // per-channel normalized pad value (RGBA)
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
        vec4 rgba = texture(src, src_uv);
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
/// readout produces a tightly-packed NHWC buffer. `W*3` must divide into whole
/// texels, which for a 3-channel row means `W % 4 == 0`.
///
/// Same crop and pad contract as the planar packer: the destination element
/// index gives the pixel and channel, `dst_rect_px` decides whether that pixel
/// is content or pad, and `src_rect_uv` maps content onto the source. Sampling
/// the source once per element rather than packing an already-rendered u8
/// intermediate keeps the result identical to the F32 NHWC PBO path, which
/// samples the same way.
pub(crate) const FLOAT_NHWC_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) in pixel space
uniform vec4 pad_color;       // per-channel normalized pad value (RGBA)
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
    vec4 rgba = texture(src, src_uv);
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
/// `(W, H)` — an `Rgba` float destination needs no packing, so the whole
/// fetched texel is the output, alpha included.
///
/// Addressing comes from `gl_FragCoord`, not the interpolated texcoord: the
/// other float shaders read the destination row straight off `gl_FragCoord.y`
/// (framebuffer row 0 is the surface's first row of memory) and the quad's
/// texcoords run the other way down the viewport.
pub(crate) const FLOAT_RGBA_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 src_extent;      // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) in pixel space
uniform vec4 pad_color;       // per-channel normalized pad value (RGBA)
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
    color = texture(src, src_uv);
}
"#;

/// Fullscreen-quad vertex shader shared by both backends. Position is a `vec3`
/// (NDC, `z` unused for the 2D quad but carried so the same shader serves the
/// segmentation passes); passes `fragPos`/`tc` to the fragment stage. The VBO
/// feeds `pos` (location 0) and `texCoord` (location 1).
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
// NV12/NV16/NV24 (semi-planar YUV, single R8 plane) -> RGBA, shared across
// backends. Both the Linux (DMA-BUF EGLImage) and macOS (IOSurface/ANGLE) paths
// decode the SAME combined-plane byte layout (`PixelFormat::chroma_layout` +
// `combined_plane_height`) using the **divide-free** addressing form: direct 2D
// `texelFetch` + a branchless `carry` for NV24's wrapping 2W-byte chroma row.
// No per-pixel integer divide/modulo — that is pathologically slow on Vivante
// GC7000UL / V3D and is also the software-emulated slow path on Apple GPUs
// (variable divisor), so the divide-free form is the right default everywhere.
//
// Both backends use the same `VERTEX_SHADER` (above), so the fragment shader is
// byte-identical on both — ONE source string. The bytes are validated on-target
// (Linux) and frozen by `nv_fragment_byte_identical` below.

/// Shared `main()` body (statements + closing brace) for the NV->RGBA shader.
/// References `tc` (vertex UV), the `chroma_shift`/`chroma_lines`/`tex_width`
/// layout uniforms, and the six colorimetry uniforms; writes `color`.
macro_rules! nv_rgba_body_divfree {
    () => {
        "    int w = img_size.x;
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
"
    };
}

/// NV->RGBA fragment shader (Path B), shared verbatim by both backends. Vertex
/// stage ([`VERTEX_SHADER`]) provides `fragPos`/`tc`; output is `color`. The
/// bytes are validated on-target (Linux) and frozen by the golden test below.
/// YUYV (sampled as a GL_RG texture: R=Y, G=alternating U/V) → RGBA.
/// Each output pixel samples its own texel and the horizontal partner to
/// recover both chroma components; the YUV→RGB matrix + range come from
/// the per-tensor colorimetry uniforms (same names as the NV program).
/// `src_size` is the texture's texel grid and `src_extent` the logical
/// image's rectangle on it (`render::sample_clamp_rect`), which differ when
/// the import covers more of the texture than the logical image.
/// Portable `sampler2D` — shared by the IOSurface zero-copy source path
/// and any future heap-YUYV upload path.
pub(crate) const YUYV_RGBA_2D_FRAGMENT: &str = r#"#version 300 es
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

void main() {
    vec2 texel = vec2(1.0) / src_size;
    // `src_extent` keeps the texel index inside the logical image when the
    // texture is larger than it; `src_size` is the texture's own grid.
    vec2 col = floor(clamp(tc, src_extent.xy, src_extent.zw) * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    // The clamp bounds `col`, not the partner. On an odd logical width the
    // last column is even, so its partner is one texel past the logical
    // image: stale pool content on a narrowed import, and the column itself
    // under CLAMP_TO_EDGE otherwise. Out of scope, and unchanged by the
    // clamp: a YUYV chroma pair spans two columns, so an odd width has no
    // partner to read in any layout (the allocation is `width` texels wide,
    // not rounded up), and the engine's YUYV coverage is even widths only.
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
    color = vec4(r, g, b, 1.0);
}
"#;

pub(crate) const NV_RGBA_FRAGMENT: &str = concat!(
    "\
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
",
    nv_rgba_body_divfree!()
);

#[cfg(test)]
mod nv_shader_golden {
    /// The golden with CRLF normalized to LF. A checkout made with
    /// `core.autocrlf=true` (the Git for Windows default) gives
    /// `include_str!` CRLF line endings while rustc normalizes the string
    /// literal to LF. `.gitattributes` pins `*.glsl` to LF; this covers
    /// checkouts made before that attribute existed.
    fn golden(src: &str) -> String {
        src.replace("\r\n", "\n")
    }

    /// The NV->RGBA shader source is validated on-target (V3D/Mali/Vivante/Tegra);
    /// its bytes must not drift. `golden/nv_rgba_linux.glsl` is the frozen
    /// reference. This test runs on every platform (the module is uncfg'd), so
    /// byte-identity is enforced on the macOS and Windows hosts too, not just
    /// the Linux lane.
    #[test]
    fn nv_fragment_byte_identical() {
        assert_eq!(
            super::NV_RGBA_FRAGMENT,
            golden(include_str!("golden/nv_rgba_linux.glsl")).as_str(),
            "NV->RGBA shader bytes drifted from the on-target-validated golden"
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
