// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use edgefirst_tensor::PixelFormat;
use gbm::drm::buffer::DrmFourcc;
use log::error;
use std::ffi::{c_char, CString};
use std::ptr::null;
use std::str::FromStr;

pub(super) fn compile_shader_from_str(
    shader: u32,
    shader_source: &str,
    shader_name: &str,
) -> Result<(), ()> {
    let src = match CString::from_str(shader_source) {
        Ok(v) => v,
        Err(_) => return Err(()),
    };
    let src_ptr = src.as_ptr();
    unsafe {
        gls::gl::ShaderSource(shader, 1, &raw const src_ptr, null());
        gls::gl::CompileShader(shader);
        let mut is_compiled = 0;
        gls::gl::GetShaderiv(shader, gls::gl::COMPILE_STATUS, &raw mut is_compiled);
        if is_compiled == 0 {
            let mut max_length = 0;
            gls::gl::GetShaderiv(shader, gls::gl::INFO_LOG_LENGTH, &raw mut max_length);
            let mut error_log: Vec<u8> = vec![0; max_length as usize];
            gls::gl::GetShaderInfoLog(
                shader,
                max_length,
                &raw mut max_length,
                error_log.as_mut_ptr() as *mut c_char,
            );
            let msg = CString::from_vec_with_nul(error_log)
                .ok()
                .and_then(|c| c.into_string().ok())
                .unwrap_or_else(|| "<non-UTF8 shader log>".to_string());
            error!("Shader '{}' failed: {:?}\n", shader_name, msg);
            gls::gl::DeleteShader(shader);
            return Err(());
        }
        Ok(())
    }
}

pub(super) fn check_gl_error(name: &str, line: u32) -> Result<(), Error> {
    unsafe {
        let err = gls::gl::GetError();
        if err != gls::gl::NO_ERROR {
            error!("GL Error: {name}:{line}: {err:#X}");
            // panic!("GL Error: {err}");
            return Err(Error::OpenGl(format!("{err:#X}")));
        }
    }
    Ok(())
}

pub(super) fn pixel_format_to_drm(fmt: PixelFormat) -> Result<DrmFourcc, Error> {
    match fmt {
        PixelFormat::Rgba => Ok(DrmFourcc::Abgr8888),
        PixelFormat::Bgra => Ok(DrmFourcc::Argb8888),
        PixelFormat::Yuyv => Ok(DrmFourcc::Yuyv),
        PixelFormat::Vyuy => Ok(DrmFourcc::Vyuy),
        PixelFormat::Rgb => Ok(DrmFourcc::Bgr888),
        PixelFormat::Grey => Ok(DrmFourcc::R8),
        PixelFormat::Nv12 => Ok(DrmFourcc::Nv12),
        PixelFormat::PlanarRgb => Ok(DrmFourcc::R8),
        _ => Err(Error::NotSupported(format!(
            "PixelFormat {fmt:?} has no DRM format mapping"
        ))),
    }
}

pub(super) fn generate_vertex_shader() -> &'static str {
    "\
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
"
}

pub(super) fn generate_texture_fragment_shader() -> &'static str {
    "\
#version 300 es

precision mediump float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

pub(super) fn generate_texture_fragment_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

/// Planar RGB shader using `samplerExternalOES` for EGLImage sources.
///
/// Currently byte-identical to [`generate_texture_fragment_shader_yuv`] but
/// kept as a separate function so the planar draw path can diverge
/// independently (e.g., for custom per-channel operations). The `_2d`
/// variant ([`generate_planar_rgb_shader_2d`]) uses `sampler2D` instead.
pub(super) fn generate_planar_rgb_shader() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

/// Int8 variant of [`generate_texture_fragment_shader`]. Quantizes each RGB
/// channel to uint8, applies XOR 0x80 bias via `(q + 128) mod 256`, then
/// normalizes back. Intended for non-external 2D texture sources
/// (e.g., RGBA/BGRA/Grey textures bound as `sampler2D`). DMA/EGLImage and
/// other external-OES paths use [`generate_texture_int8_shader_yuv`].
pub(super) fn generate_texture_int8_shader() -> &'static str {
    "\
#version 300 es
precision highp float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

// XOR 0x80 bias: quantize to uint8, add 128 mod 256, normalize back.
// This matches the CPU `byte ^ 0x80` operation exactly.
vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_texture_fragment_shader_yuv`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion).
/// Used for single-pass int8 output with external OES sources (YUV EGLImage).
pub(super) fn generate_texture_int8_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_planar_rgb_shader`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion) using the bit-exact
/// quantize+mod approach: `floor(v * 255 + 0.5) + 128 mod 256 / 255`.
pub(super) fn generate_planar_rgb_int8_shader() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// 2D-sampler variant of [`generate_planar_rgb_shader`]. Uses `sampler2D`
/// instead of `samplerExternalOES` for sourcing from intermediate RGBA
/// textures (e.g., two-pass NV12→RGBA→PlanarRgb on Vivante).
pub(super) fn generate_planar_rgb_shader_2d() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

/// Int8 variant of [`generate_planar_rgb_shader_2d`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion) using the bit-exact
/// quantize+mod approach: `floor(v * 255 + 0.5) + 128 mod 256 / 255`.
pub(super) fn generate_planar_rgb_int8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// this shader requires a reshape of the segmentation output tensor to (H, W,
/// C/4, 4)
pub(super) fn generate_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
precision mediump sampler2DArray;

uniform sampler2DArray tex;
uniform vec4 colors[20];
uniform int background_index;
uniform float opacity;

in vec3 fragPos;
in vec2 tc;
in vec4 fragColor;

out vec4 color;

float max_arg(const in vec4 args, out int argmax) {
    if (args[0] >= args[1] && args[0] >= args[2] && args[0] >= args[3]) {
        argmax = 0;
        return args[0];
    }
    if (args[1] >= args[0] && args[1] >= args[2] && args[1] >= args[3]) {
        argmax = 1;
        return args[1];
    }
    if (args[2] >= args[0] && args[2] >= args[1] && args[2] >= args[3]) {
        argmax = 2;
        return args[2];
    }
    argmax = 3;
    return args[3];
}

void main() {
    mediump int layers = textureSize(tex, 0).z;
    float max_all = -4.0;
    int max_ind = 0;
    for (int i = 0; i < layers; i++) {
        vec4 d = texture(tex, vec3(tc, i));
        int max_ind_ = 0;
        float max_ = max_arg(d, max_ind_);
        if (max_ <= max_all) { continue; }
        max_all = max_;
        max_ind = i*4 + max_ind_;
    }
    if (max_ind == background_index) {
        discard;
    }
    max_ind = max_ind % 20;
    vec4 c = colors[max_ind];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

pub(super) fn generate_instanced_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform sampler2D mask0;
uniform vec4 colors[20];
uniform int class_index;
uniform float opacity;
in vec3 fragPos;
in vec2 tc;
in vec4 fragColor;

out vec4 color;
void main() {
    float r0 = texture(mask0, tc).r;
    float edge = smoothstep(0.5, 0.65, r0);
    if (edge <= 0.0) {
        discard;
    }
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * edge * opacity);
}
"
}

pub(super) fn generate_proto_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;  // ceil(num_protos/4) layers, RGBA = 4 channels per layer
uniform vec4 mask_coeff[8];        // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_layers;
uniform float opacity;

in vec2 tc;
out vec4 color;

void main() {
    float acc = 0.0;
    for (int i = 0; i < num_layers; i++) {
        // texture() returns bilinearly interpolated proto values (GL_LINEAR)
        acc += dot(mask_coeff[i], texture(proto_tex, vec3(tc, float(i))));
    }
    float mask = 1.0 / (1.0 + exp(-acc));  // sigmoid
    if (mask < 0.5) discard;
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

/// Int8 proto shader — nearest-neighbor only.
///
/// Uses `texelFetch()` at the nearest texel. No interpolation. Simplest and
/// fastest GPU execution but may show staircase artifacts at mask edges.
///
/// Layout: `GL_R8I` texture with 1 proto per layer (32 layers).
/// Mask coefficients packed as `vec4[8]`, indexed `mask_coeff[k/4][k%4]`.
pub(super) fn generate_proto_segmentation_shader_int8_nearest() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];         // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale
uniform float opacity;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        float raw = float(texelFetch(proto_tex, ivec3(ix, iy, k), 0).r);
        float val = raw * proto_scale + proto_scaled_zp;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

/// Int8 proto shader — shader-based bilinear interpolation (recommended).
///
/// Uses `texelFetch()` to fetch 4 neighboring texels per fragment, dequantizes
/// each, and computes bilinear weights from `fract(tc * textureSize)`.
///
/// Layout: `GL_R8I` texture with 1 proto per layer (32 layers).
pub(super) fn generate_proto_segmentation_shader_int8_bilinear() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];         // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale
uniform float opacity;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    // Compute continuous position (matching GL_LINEAR convention: center at +0.5)
    vec2 pos = tc * vec2(tex_size.xy) - 0.5;
    vec2 f = fract(pos);
    ivec2 p0 = ivec2(floor(pos));
    ivec2 p1 = p0 + 1;
    // Clamp to texture bounds
    p0 = clamp(p0, ivec2(0), tex_size.xy - 1);
    p1 = clamp(p1, ivec2(0), tex_size.xy - 1);

    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;

    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        float r00 = float(texelFetch(proto_tex, ivec3(p0.x, p0.y, k), 0).r);
        float r10 = float(texelFetch(proto_tex, ivec3(p1.x, p0.y, k), 0).r);
        float r01 = float(texelFetch(proto_tex, ivec3(p0.x, p1.y, k), 0).r);
        float r11 = float(texelFetch(proto_tex, ivec3(p1.x, p1.y, k), 0).r);
        float interp = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
        float val = interp * proto_scale + proto_scaled_zp;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

/// Int8 dequantization pass shader (two-pass Option C, pass 1).
///
/// Reads `GL_R8I` texel, dequantizes, and writes float to `GL_RGBA16F` render
/// target. This shader processes 4 protos at a time (packing into RGBA).
/// After this pass, the existing f16 shader reads the dequantized texture with
/// `GL_LINEAR`.
pub(super) fn generate_proto_dequant_shader_int8() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers of R8I (1 proto per layer)
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale
uniform int base_layer;             // first proto index for this output layer (0, 4, 8, ...)

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    vec4 result;
    for (int c = 0; c < 4; c++) {
        int layer = base_layer + c;
        float raw = float(texelFetch(proto_tex, ivec3(ix, iy, layer), 0).r);
        result[c] = raw * proto_scale + proto_scaled_zp;
    }
    color = result;
}
"
}

/// F32 proto shader — direct R32F texture with hardware bilinear filtering.
///
/// Same structure as int8 bilinear shader but uses `texture()` for hardware
/// interpolation (requires `GL_OES_texture_float_linear`). No dequantization.
///
/// Layout: `GL_R32F` texture with 1 proto per layer (32 layers).
pub(super) fn generate_proto_segmentation_shader_f32() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];        // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;
uniform float opacity;

in vec2 tc;
out vec4 color;

void main() {
    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        // texture() returns bilinearly interpolated proto value (GL_LINEAR on R32F)
        float val = texture(proto_tex, vec3(tc, float(k))).r;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

/// Tightly-packed NHWC F32 fragment shader for HailoRT consumption.
///
/// Render target is a single-channel `R32F` texture sized `(W*3, H)`:
/// each output texel holds exactly one float for one `(pixel, channel)` pair.
/// The mapping is `channel = x % 3`, `pixel_x = x / 3`, `pixel_y = y`,
/// so the linear read-out of the rendered texture produces a tightly-packed
/// `[H, W, 3]` F32 buffer (NHWC order, one image in the batch).
///
/// The RGBA8 source texture is fetched via a `sampler2D`, which normalizes
/// values to `[0, 1]` automatically — no explicit `/255` division needed.
/// Letterboxing is applied using `dst_rect_px` (the active image region in
/// pixel coords); pixels outside that rectangle are filled with `pad_color`.
///
/// Source sampling uses an output-pixel-center offset (`+0.5`) so the mapped
/// source UV lands on the correct location for bilinear resize. With LINEAR
/// source filtering this yields a proper bilinear resample; for an identity
/// crop (`src_w == dst_w`) the UV lands exactly on the texel center, giving an
/// exact passthrough.
// Consumed by `GLProcessorST` for the F32 NHWC PBO render path.
pub(super) fn generate_packed_f32_nhwc_shader() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_tex;        // RGBA8 source, normalized fetch -> [0,1]
uniform vec4 src_rect_uv;       // (origin_u, origin_v, size_u, size_v)
uniform vec4 dst_rect_px;       // (origin_x, origin_y, w, h) in pixel space
uniform vec3 pad_color;         // per-channel normalized pad value
out float frag_value;
void main() {
    int ox = int(gl_FragCoord.x);
    int oy = int(gl_FragCoord.y);
    int channel = ox % 3;
    int px = ox / 3;
    int py = oy;
    bool inside = (float(px) >= dst_rect_px.x) &&
                  (float(px) <  dst_rect_px.x + dst_rect_px.z) &&
                  (float(py) >= dst_rect_px.y) &&
                  (float(py) <  dst_rect_px.y + dst_rect_px.w);
    if (!inside) {
        frag_value = (channel == 0) ? pad_color.r
                   : (channel == 1) ? pad_color.g : pad_color.b;
        return;
    }
    float u = (float(px) + 0.5 - dst_rect_px.x) / dst_rect_px.z;
    float v = (float(py) + 0.5 - dst_rect_px.y) / dst_rect_px.w;
    vec2 src_uv = src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw;
    vec4 rgba = texture(u_tex, src_uv);
    frag_value = (channel == 0) ? rgba.r
               : (channel == 1) ? rgba.g : rgba.b;
}
"
}

/// RGBA8 → packed RGBA16F PlanarRgb F16 fragment shader.
///
/// The render target is a single `RGBA16F` texture sized `(W/4, 3*H)`.
/// Each output texel packs 4 half-float channel samples into its four
/// components, so a linear readout of the rendered texture produces a
/// tightly-packed `[3, H, W]` F16 buffer (CHW / PlanarRgb order, one plane
/// per row-band).  Width `W` must be a multiple of 4.
///
/// Normalization from RGBA8 → `[0, 1]` is performed for free by the hardware
/// texture fetch (`sampler2D` normalized fetch).  Letterboxing is applied via
/// `dst_rect_px`; pixels outside that rectangle are filled with `pad_color`.
///
/// Single source of truth lives in [`super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT`],
/// shared with the macOS IOSurface path (`macos_processor.rs`).
// Consumed by `GLProcessorST` for the F16 NCHW PBO and DMA-BUF render paths.
pub(super) fn generate_planar_rgb_f16_packed_shader() -> &'static str {
    super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT
}

pub(super) fn generate_color_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform vec4 colors[20];
uniform int class_index;
uniform float opacity;

out vec4 color;
void main() {
    int index = class_index % 20;
    vec4 c = colors[index];
    color = vec4(c.rgb, c.a * opacity);
}
"
}

/// Packed RGB -> RGBA8 packing shader (2D texture source, pass 2).
///
/// Reads from an intermediate RGBA texture and packs 3 RGB channels into
/// RGBA8 output pixels. Each output pixel stores 4 consecutive bytes of the
/// destination RGB buffer. Uses only 2 texture fetches per fragment (down
/// from 4) by exploiting the fact that 4 consecutive bytes span at most 2
/// source pixels.
pub(super) fn generate_packed_rgba8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D tex;
out vec4 color;
void main() {
    // gl_FragCoord is at pixel center (n+0.5). Use floor() for robust
    // integer pixel index on all GPUs (Vivante, Mali, Adreno).
    int out_x = int(floor(gl_FragCoord.x));
    int out_y = int(floor(gl_FragCoord.y));
    int base = out_x * 4;
    // 4 consecutive byte indices map to at most 2 source pixels
    int px0 = base / 3;
    int px1 = (base + 3) / 3;
    vec4 s0 = texelFetch(tex, ivec2(px0, out_y), 0);
    vec4 s1 = (px1 != px0) ? texelFetch(tex, ivec2(px1, out_y), 0) : s0;
    // Extract channels based on phase (base % 3)
    int phase = base - px0 * 3;
    if (phase == 0) {
        color = vec4(s0.r, s0.g, s0.b, s1.r);
    } else if (phase == 1) {
        color = vec4(s0.g, s0.b, s1.r, s1.g);
    } else {
        color = vec4(s0.b, s1.r, s1.g, s1.b);
    }
}
"
}

/// Packed RGB -> RGBA8 packing shader with int8 XOR 0x80 bias (2D source, pass 2).
///
/// Same packing logic as [`generate_packed_rgba8_shader_2d`] but applies
/// bit-exact XOR 0x80 bias via quantize+mod: `floor(v * 255 + 0.5) + 128
/// mod 256 / 255`. This matches the CPU `byte ^ 0x80` operation exactly.
pub(super) fn generate_packed_rgba8_int8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D tex;
out vec4 color;

vec4 int8_bias(vec4 v) {
    vec4 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main() {
    // gl_FragCoord is at pixel center (n+0.5). Use floor() for robust
    // integer pixel index on all GPUs (Vivante, Mali, Adreno).
    int out_x = int(floor(gl_FragCoord.x));
    int out_y = int(floor(gl_FragCoord.y));
    int base = out_x * 4;
    // 4 consecutive byte indices map to at most 2 source pixels
    int px0 = base / 3;
    int px1 = (base + 3) / 3;
    vec4 s0 = texelFetch(tex, ivec2(px0, out_y), 0);
    vec4 s1 = (px1 != px0) ? texelFetch(tex, ivec2(px1, out_y), 0) : s0;
    // Extract channels based on phase (base % 3), then apply int8 bias
    int phase = base - px0 * 3;
    if (phase == 0) {
        color = int8_bias(vec4(s0.r, s0.g, s0.b, s1.r));
    } else if (phase == 1) {
        color = int8_bias(vec4(s0.g, s0.b, s1.r, s1.g));
    } else {
        color = int8_bias(vec4(s0.b, s1.r, s1.g, s1.b));
    }
}
"
}

/// Semi-planar YUV (NV12/NV16/NV24) → RGBA, Path B (R8 sampler2D, ES 3.0 core).
///
/// The combined semi-planar buffer is imported as a single-plane R8 EGLImage
/// (width = `effective_row_stride`, height = luma_h + chroma_h) and bound as
/// `TEXTURE_2D`.  Y and UV texels are addressed directly with `texelFetch`,
/// parameterised by uniforms so one program serves all three subsamplings.
///
/// Uniforms:
///   * `img_size`     — logical (W, H); Y plane occupies rows [0, H).
///   * `tex_width`    — R8 texture width (= even buffer width / effective stride).
///   * `chroma_shift` — (cx, cy) right-shifts: NV12=(1,1), NV16=(1,0), NV24=(0,0).
///   * `chroma_lines`  — R8 buffer rows per image-chroma-row: NV12/NV16=1,
///     NV24=2 (NV24's 2W-byte CbCr row wraps at `tex_width`, spanning 2 rows;
///     the shader's `carry` term handles the wrap). Direct 2D addressing — no
///     per-pixel integer divide/modulo (pathologically slow on Vivante).
///
/// Vertex varying is `tc` (vec2, matching `generate_vertex_shader`).
/// BT.601 full-range matches the CPU kernels and the EGL YUV color hints.
/// No extension required: `texelFetch` + R8 is core ES 3.0.
///
/// CHROMA-LAYOUT CONTRACT: this shader and the macOS `NV_TO_RGBA_FRAGMENT`
/// (`macos_processor.rs`) decode the SAME `model-2` combined-plane byte layout
/// (`PixelFormat::chroma_layout` + `combined_plane_height`), but parameterise it
/// differently: this one uses `chroma_lines` + a branchless `carry` for direct
/// 2D `texelFetch`, while macOS uses `uv_row_bytes` + a linear `fetch_r(b)` with
/// `b % tex_width`. They are kept SEPARATE on purpose — the divide-free form
/// here is required for Vivante/V3D, while Apple-silicon ANGLE tolerates the
/// linear form. They are provably equivalent at every NV24 texel (the codec's
/// `decode_padded_grid_matches_tight` fixture and the `*_opengl_macos` GPU-vs-CPU
/// tests are the cross-checks); keep both in sync if the layout ever changes.
pub(super) fn generate_nv_to_rgba_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int chroma_lines;
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

    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    float r = clamp(yv + 1.402 * vp, 0.0, 1.0);
    float g = clamp(yv - 0.344 * up - 0.714 * vp, 0.0, 1.0);
    float b = clamp(yv + 1.772 * up, 0.0, 1.0);
    color = vec4(r, g, b, 1.0);
}
"
}

/// Int8 variant of [`generate_nv_to_rgba_shader_2d`].
///
/// Applies the same XOR 0x80 bias (`(q + 128) mod 256`) to each output RGB
/// channel as [`generate_texture_int8_shader`] and the other int8 shaders.
/// Used when the destination dtype is i8 so no CPU post-processing is needed.
pub(super) fn generate_nv_to_rgba_int8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int chroma_lines;
in vec3 fragPos;
in vec2 tc;
out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main() {
    int w = img_size.x;
    int h = img_size.y;
    int x = clamp(int(tc.x * float(w)), 0, w - 1);
    int y = clamp(int(tc.y * float(h)), 0, h - 1);

    // Luma: direct 2D texel — no per-pixel integer divide/modulo.
    float yv = texelFetch(src, ivec2(x, y), 0).r;

    int ccol = x >> chroma_shift.x;
    int crow = y >> chroma_shift.y;
    int ccol2 = ccol * 2;
    int carry = ccol2 >= tex_width ? 1 : 0;
    int cy = h + crow * chroma_lines + carry;
    int cx = ccol2 - carry * tex_width;
    float u = texelFetch(src, ivec2(cx, cy), 0).r;
    float v = texelFetch(src, ivec2(cx + 1, cy), 0).r;

    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    float r = clamp(yv + 1.402 * vp, 0.0, 1.0);
    float g = clamp(yv - 0.344 * up - 0.714 * vp, 0.0, 1.0);
    float b = clamp(yv + 1.772 * up, 0.0, 1.0);
    color = vec4(int8_bias(vec3(r, g, b)), 1.0);
}
"
}

/// HWC → layer-first (CHW) repack compute shader for int8 protos.
///
/// Reads proto data from an SSBO in row-major HWC layout `(H, W, num_protos)`.
/// Writes to a `GL_R32I` `GL_TEXTURE_2D_ARRAY` with one proto per layer via
/// `imageStore`. Each workgroup thread handles one `(x, y)` position and
/// writes all `num_protos` layers.
///
/// The SSBO stores raw `i8` bytes packed as `int[]` (4 bytes per element).
/// The shader extracts individual bytes and sign-extends them.
///
/// Requires GLES 3.1+.
pub(super) fn generate_proto_repack_compute_shader() -> &'static str {
    "\
#version 310 es
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer ProtoSSBO {
    int packed_data[];
};

layout(r32i, binding = 0) writeonly uniform highp iimage2DArray dst_tex;

uniform int width;
uniform int height;
uniform int num_protos;

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);

    if (x >= width || y >= height) return;

    int base = (y * width + x) * num_protos;

    for (int k = 0; k < num_protos; k++) {
        int byte_offset = base + k;
        int word_idx = byte_offset >> 2;
        int byte_idx = byte_offset & 3;
        int word = packed_data[word_idx];
        int val = (word >> (byte_idx * 8)) & 0xFF;
        if (val >= 128) val -= 256;
        imageStore(dst_tex, ivec3(x, y, k), ivec4(val, 0, 0, 0));
    }
}
"
}
