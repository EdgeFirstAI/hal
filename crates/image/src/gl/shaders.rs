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
    color = colors[max_ind];
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
in vec3 fragPos;
in vec2 tc;
in vec4 fragColor;

out vec4 color;
void main() {
    float r0 = texture(mask0, tc).r;
    int arg = int(r0>=0.5);
    if (arg == 0) {
        discard;
    }
    color = colors[class_index % 20];
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
    color = colors[class_index % 20];
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
    color = colors[class_index % 20];
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
    color = colors[class_index % 20];
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
    color = colors[class_index % 20];
}
"
}

/// Binary mask shader — int8, nearest-neighbor, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Avoids
/// the `exp()` per fragment; used by `decode_masks_atlas` where only mask
/// presence matters.
pub(super) fn generate_proto_mask_logit_shader_int8_nearest() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;
uniform float proto_scale;
uniform float coeff_sum_x_szp;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        vec4 raw = vec4(
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 1, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 2, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 3, num_protos - 1)), 0).r)
        );
        acc += dot(mask_coeff[i], raw);
    }
    float logit = acc * proto_scale + coeff_sum_x_szp;
    float mask = logit > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

/// Binary mask shader — int8, shader-based bilinear interpolation, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Used by
/// `decode_masks_atlas` for int8 models with bilinear interpolation.
pub(super) fn generate_proto_mask_logit_shader_int8_bilinear() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;
uniform float proto_scale;
uniform float coeff_sum_x_szp;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    vec2 pos = tc * vec2(tex_size.xy) - 0.5;
    vec2 f = fract(pos);
    ivec2 p0 = ivec2(floor(pos));
    ivec2 p1 = p0 + 1;
    p0 = clamp(p0, ivec2(0), tex_size.xy - 1);
    p1 = clamp(p1, ivec2(0), tex_size.xy - 1);

    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;

    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        int l0 = min(base, num_protos - 1);
        int l1 = min(base + 1, num_protos - 1);
        int l2 = min(base + 2, num_protos - 1);
        int l3 = min(base + 3, num_protos - 1);
        vec4 r00 = vec4(
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l3), 0).r)
        );
        vec4 r10 = vec4(
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l3), 0).r)
        );
        vec4 r01 = vec4(
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l3), 0).r)
        );
        vec4 r11 = vec4(
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l3), 0).r)
        );
        vec4 interp = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
        acc += dot(mask_coeff[i], interp);
    }
    float logit = acc * proto_scale + coeff_sum_x_szp;
    float mask = logit > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

/// Binary mask shader — f32 protos with hardware bilinear filtering, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Used by
/// `decode_masks_atlas` for f32 models.
pub(super) fn generate_proto_mask_logit_shader_f32() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;

in vec2 tc;
out vec4 color;

void main() {
    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        vec4 val = vec4(
            texture(proto_tex, vec3(tc, float(min(base, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 1, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 2, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 3, num_protos - 1)))).r
        );
        acc += dot(mask_coeff[i], val);
    }
    float mask = acc > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

pub(super) fn generate_color_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform vec4 colors[20];
uniform int class_index;

out vec4 color;
void main() {
    int index = class_index % 20;
    color = colors[index];
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
