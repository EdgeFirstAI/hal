// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL shader sources shared across platform backends (macOS ANGLE + Linux).

// ---------------------------------------------------------------------------
// In-shader YUV→RGB conversion fragment shaders.
//
// Both backends import the raw YUV bytes into plain `sampler2D` textures and
// run the BT.601/BT.709/BT.2020 matrix in the fragment shader, steered by the
// six per-tensor colorimetry uniforms (`y_offset`, `y_scale`, `c_vr`, `c_ug`,
// `c_vg`, `c_ub`) computed by `crate::colorimetry::yuv_to_rgb_coeffs`. The
// GLSL is identical on every GL backend — only the texture *import* differs
// (macOS: IOSurface pbuffer; Linux: per-plane DMA-BUF EGLImage). No
// `samplerExternalOES`, no `GL_OES_EGL_image_external` extension, so these are
// fully portable `#version 300 es` shaders.
//
// The vertex stage emits a `vec2 v_uv` varying (the source texture
// coordinate). Both backends provide a compatible vertex shader: attribute
// location 0 = position, location 1 = the source UV. The Linux vertex shader
// writes `v_uv = texCoord`; the macOS one writes `v_uv = uv_in`.
// ---------------------------------------------------------------------------

/// Packed YUYV (a.k.a. YUY2) → RGBA fragment shader.
///
/// The source is bound as a single half-resolution-in-x `GR88`/`RG88`
/// `sampler2D`: each texel carries `Y` in `.r` and the alternating chroma
/// sample (`U` on even columns, `V` on odd columns) in `.g`. For each output
/// pixel we sample the current and the paired texel to recover both chroma
/// values, then apply the colorimetry matrix.
///
/// Single source of truth shared by the macOS ANGLE backend
/// (`macos_processor.rs`) and the Linux GLES backend
/// (`processor::draw_camera_texture_eglimage`).
pub(crate) const YUYV_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D src;
uniform vec2 src_size;
uniform float y_offset;   // luma black level, normalised (e.g. 16/255)
uniform float y_scale;    // luma gain (e.g. 1.164 limited, 1.0 full)
uniform float c_vr;       // V→R coefficient
uniform float c_ug;       // U→G coefficient
uniform float c_vg;       // V→G coefficient
uniform float c_ub;       // U→B coefficient
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 texel = vec2(1.0) / src_size;
    vec2 col = floor(v_uv * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;

    vec4 self_rg = texture(src, self_uv);
    vec4 pair_rg = texture(src, pair_uv);
    float y = self_rg.r;
    float u, v;
    if (even) { u = self_rg.g; v = pair_rg.g; }
    else      { v = self_rg.g; u = pair_rg.g; }

    float yp = (y - y_offset) * y_scale;
    float up = u - 128.0/255.0;
    float vp = v - 128.0/255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    frag = vec4(r, g, b, 1.0);
}
"#;

/// Int8 variant of [`YUYV_TO_RGBA_FRAGMENT`]: applies the bit-exact XOR 0x80
/// bias (`floor(v*255+0.5)+128 mod 256 / 255`) to each RGB channel after the
/// YUV→RGB matrix, matching the CPU `byte ^ 0x80` quantize. Linux-only: the
/// single-pass YUV→RGBA int8 DMA destination path. The shared base shader is
/// `#version 300 es`; this clones the body with the bias appended.
#[cfg(target_os = "linux")]
pub(crate) const YUYV_TO_RGBA_INT8_FRAGMENT: &str = r#"#version 300 es
precision highp float;
uniform sampler2D src;
uniform vec2 src_size;
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec2 v_uv;
out vec4 frag;

float int8_bias(float v) {
    float q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main() {
    vec2 texel = vec2(1.0) / src_size;
    vec2 col = floor(v_uv * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;

    vec4 self_rg = texture(src, self_uv);
    vec4 pair_rg = texture(src, pair_uv);
    float y = self_rg.r;
    float u, v;
    if (even) { u = self_rg.g; v = pair_rg.g; }
    else      { v = self_rg.g; u = pair_rg.g; }

    float yp = (y - y_offset) * y_scale;
    float up = u - 128.0/255.0;
    float vp = v - 128.0/255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    frag = vec4(int8_bias(r), int8_bias(g), int8_bias(b), 1.0);
}
"#;

/// Semi-planar NV12 / NV16 → RGBA fragment shader.
///
/// Two `sampler2D` inputs:
///   * `y_tex` — the luma plane, `R8`, full resolution. `Y` is in `.r`.
///   * `uv_tex` — the interleaved chroma plane, `GR88`/`RG88`. For NV12 this
///     is half resolution in both axes; for NV16 half resolution in x only.
///     The DRM `GR88` import exposes the first interleaved byte (`U`/`Cb`) in
///     `.r` and the second (`V`/`Cr`) in `.g`, matching NV12 byte order.
///
/// Both samplers are addressed with the same normalised `v_uv`; the GPU's
/// bilinear filter upsamples the chroma plane for free.
///
/// Single source of truth shared by the macOS ANGLE backend and the Linux
/// GLES backend.
// Consumed by the Linux per-plane DMA path today; the macOS backend does not
// yet have a semi-planar NV12 path, so suppress the unused warning there.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) const NV12_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
uniform float y_offset;   // luma black level, normalised (e.g. 16/255)
uniform float y_scale;    // luma gain (e.g. 1.164 limited, 1.0 full)
uniform float c_vr;       // V→R coefficient
uniform float c_ug;       // U→G coefficient
uniform float c_vg;       // V→G coefficient
uniform float c_ub;       // U→B coefficient
in vec2 v_uv;
out vec4 frag;

void main() {
    float y = texture(y_tex, v_uv).r;
    // NV12 byte order: U (Cb) then V (Cr). The GR88 import places the first
    // interleaved byte in .r and the second in .g, so .r = U, .g = V.
    vec2 uv = texture(uv_tex, v_uv).rg;
    float yp = (y - y_offset) * y_scale;
    float up = uv.x - 128.0/255.0;
    float vp = uv.y - 128.0/255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    frag = vec4(r, g, b, 1.0);
}
"#;

/// Int8 variant of [`NV12_TO_RGBA_FRAGMENT`]: applies the bit-exact XOR 0x80
/// bias to each RGB channel after the YUV→RGB matrix. Linux-only.
#[cfg(target_os = "linux")]
pub(crate) const NV12_TO_RGBA_INT8_FRAGMENT: &str = r#"#version 300 es
precision highp float;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec2 v_uv;
out vec4 frag;

float int8_bias(float v) {
    float q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main() {
    float y = texture(y_tex, v_uv).r;
    vec2 uv = texture(uv_tex, v_uv).rg;
    float yp = (y - y_offset) * y_scale;
    float up = uv.x - 128.0/255.0;
    float vp = uv.y - 128.0/255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    frag = vec4(int8_bias(r), int8_bias(g), int8_bias(b), 1.0);
}
"#;

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
