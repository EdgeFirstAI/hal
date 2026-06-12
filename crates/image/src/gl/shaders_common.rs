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
// EGL color-space/range hints. Only the colorimetry-agnostic PlanarRgb F16
// packer is shared here.)
// ---------------------------------------------------------------------------

/// RGBA8 -> packed RGBA16F PlanarRgb F16 fragment shader.
///
/// The render target is a single `RGBA16F` texture sized `(W/4, 3*H)`.
/// Each output texel packs 4 half-float channel samples into its four
/// components, so a linear readout of the rendered texture produces a
/// tightly-packed `[3, H, W]` F16 buffer (CHW / PlanarRgb order, one plane
/// per row-band).  Width `W` must be a multiple of 4.
///
/// This is the single source of truth consumed by both the Linux (PBO/DMA)
/// and macOS (IOSurface/ANGLE) render paths. The GLSL bytes are preserved
/// exactly as validated by the on-target Linux F16 round-trip tests and the
/// macOS IOSurface integration tests.
pub(crate) const PLANAR_RGB_F16_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec2 dst_image_size;  // (W, H) — destination plane size
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) within one plane
uniform vec3 pad_color;       // per-channel normalized pad value
out vec4 frag;

// Sample one planar element. Returns the per-channel value at
// (in_plane_x, in_plane_y) for the given plane (0=R, 1=G, 2=B).
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
        vec2 src_uv = src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw;
        vec4 rgba = texture(src, src_uv);
        if (plane < 0.5) {
            return rgba.r;
        } else if (plane < 1.5) {
            return rgba.g;
        } else {
            return rgba.b;
        }
    } else {
        if (plane < 0.5) {
            return pad_color.r;
        } else if (plane < 1.5) {
            return pad_color.g;
        } else {
            return pad_color.b;
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

    // RGBA16F framebuffer narrows each f32 to f16 on write. The
    // resulting byte layout in the locked IOSurface is exactly the
    // [3, H, W] f16 planar order the consumer expects.
    frag = vec4(e0, e1, e2, e3);
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
/// Portable `sampler2D` — shared by the IOSurface zero-copy source path
/// and any future heap-YUYV upload path.
pub(crate) const YUYV_RGBA_2D_FRAGMENT: &str = r#"#version 300 es
precision highp float;
uniform highp sampler2D tex;
uniform vec2 src_size;
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
    vec2 col = floor(tc * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
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
    /// The NV->RGBA shader source is validated on-target (V3D/Mali/Vivante/Tegra);
    /// its bytes must not drift. `golden/nv_rgba_linux.glsl` is the frozen
    /// reference. This test runs on every platform (the module is uncfg'd), so
    /// byte-identity is enforced on the macOS host too, not just the Linux lane.
    #[test]
    fn nv_fragment_byte_identical() {
        assert_eq!(
            super::NV_RGBA_FRAGMENT,
            include_str!("golden/nv_rgba_linux.glsl"),
            "NV->RGBA shader bytes drifted from the on-target-validated golden"
        );
    }

    #[test]
    fn vertex_byte_identical() {
        assert_eq!(
            super::VERTEX_SHADER,
            include_str!("golden/vertex.glsl"),
            "vertex shader bytes drifted from the on-target-validated golden"
        );
    }
}
