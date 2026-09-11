// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::Error;
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
        edgefirst_gl::gl::ShaderSource(shader, 1, &raw const src_ptr, null());
        edgefirst_gl::gl::CompileShader(shader);
        let mut is_compiled = 0;
        edgefirst_gl::gl::GetShaderiv(
            shader,
            edgefirst_gl::gl::COMPILE_STATUS,
            &raw mut is_compiled,
        );
        if is_compiled == 0 {
            let mut max_length = 0;
            edgefirst_gl::gl::GetShaderiv(
                shader,
                edgefirst_gl::gl::INFO_LOG_LENGTH,
                &raw mut max_length,
            );
            let mut error_log: Vec<u8> = vec![0; max_length as usize];
            edgefirst_gl::gl::GetShaderInfoLog(
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
            edgefirst_gl::gl::DeleteShader(shader);
            return Err(());
        }
        Ok(())
    }
}

pub(super) fn check_gl_error(name: &str, line: u32) -> Result<(), Error> {
    unsafe {
        let err = edgefirst_gl::gl::GetError();
        if err != edgefirst_gl::gl::NO_ERROR {
            error!("GL Error: {name}:{line}: {err:#X}");
            // panic!("GL Error: {err}");
            return Err(Error::OpenGl(format!("{err:#X}")));
        }
    }
    Ok(())
}

pub(super) fn generate_vertex_shader() -> &'static str {
    // Shared with the macOS backend; see `shaders_common::VERTEX_SHADER`.
    super::shaders_common::VERTEX_SHADER
}

/// `sampler2D` source blit. `src_extent` is the rectangle a sample may reach
/// (`render::sample_clamp_rect`): the logical image's share of the texture,
/// inset by half a texel, so a `LINEAR` kernel at the edge of an import that
/// covers more than the logical image does not blend the texel beyond it.
///
/// `src_extent` is `highp` although this shader's default is `mediump`: a
/// `mediump` bound rounds to a multiple of 2^-11 near 1.0, which on a texture
/// wider than 1024 lands inside the true half-texel inset and would pull the
/// last column's sample in by up to half a texel. At `highp` the bound is
/// exact, so clamping to it reproduces `CLAMP_TO_EDGE` wherever the texture
/// is the logical image.
///
/// `tc` is `highp` for a second, sharper reason: the SAMPLED COORDINATE must
/// not go through a `mediump` ALU. Passing a varying straight to `texture()`
/// lets a driver hand the interpolator's own coordinate to the texture unit,
/// but `clamp()` is arithmetic, and at `mediump` it rounds the result to fp16
/// — a step of 2^-11 near 1.0, which is 0.625 texel on a 1280-wide source.
/// `LINEAR` then blends the neighbour in by that fraction. Measured on
/// Mali-G310 (i.MX 95) with a 1280x720 RGBA identity blit: 96% of the frame
/// differed from the CPU reference, worst 50/255 mid-frame at an implied
/// blend fraction of 0.40 texel, and the error was flat across the width
/// rather than confined to the edges — so it was the coordinate, not the
/// bound. Clamping to the full texture `[0,0,1,1]` reproduced the corruption
/// byte for byte; declaring `tc` `highp` removed it. Vivante GC7000UL and
/// V3D did not show it (issue #170).
///
/// Only the CONSUMER's qualifier matters here. The vertex stage still writes
/// `out vec2 tc;` under a `mediump` default (`shaders_common::VERTEX_SHADER`,
/// byte-pinned by `golden/vertex.glsl`) and is deliberately left alone:
/// GLSL ES 3.00 does not match precision across stages, so the fragment
/// shader's own qualifier governs the interpolation and the ALU it feeds.
/// Two independent facts confirm it. Removing only the `clamp()` while
/// leaving that same vertex shader in place made the Mali case pass, so the
/// vertex write was never the limiting step; and the `highp` shaders that
/// have always sampled correctly on Mali — `YUYV_RGBA_2D_FRAGMENT` and
/// `NV_RGBA_FRAGMENT` — are fed by this very vertex shader.
pub(super) fn generate_texture_fragment_shader() -> &'static str {
    "\
#version 300 es

precision mediump float;
uniform sampler2D tex;
uniform highp vec4 src_extent;
in vec3 fragPos;
in highp vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
}
"
}

/// `src_extent` is the rectangle a sample may reach
/// (`render::sample_clamp_rect`), the same uniform and the same reason as
/// [`generate_texture_fragment_shader`]: an EGLImage import can cover more
/// texture than the logical image — a narrowed pool buffer, or a source
/// rebased onto an aligned DMA-BUF offset (issue #170) — and a `LINEAR`
/// kernel at the edge must not blend the texels outside it. `highp` although
/// this shader's default is `mediump`, because a `mediump` bound rounds near
/// 1.0 and would pull the last column's sample in.
pub(super) fn generate_texture_fragment_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
uniform highp vec4 src_extent;
in vec3 fragPos;
in highp vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
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
uniform highp vec4 src_extent;
in vec3 fragPos;
in highp vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
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
uniform vec4 src_extent;
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
    vec4 c = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_texture_fragment_shader_yuv`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion).
/// Used for single-pass int8 output with external OES sources (YUV EGLImage).
/// Carries the same `src_extent` clamp, for the same reason.
pub(super) fn generate_texture_int8_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
uniform highp vec4 src_extent;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_planar_rgb_shader`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion) using the bit-exact
/// quantize+mod approach: `floor(v * 255 + 0.5) + 128 mod 256 / 255`.
/// Carries the same `src_extent` clamp, for the same reason.
pub(super) fn generate_planar_rgb_int8_shader() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
uniform highp vec4 src_extent;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, clamp(tc, src_extent.xy, src_extent.zw));
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
in highp vec2 tc;
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
        // Tail guard: when num_protos % 4 != 0 the last output layer's
        // trailing channels have no source proto. texelFetch beyond the
        // array depth is undefined in GLSL ES (a NaN there would survive
        // the zero coefficients downstream: NaN * 0 = NaN), so emit an
        // explicit zero proto instead.
        if (layer < tex_size.z) {
            float raw = float(texelFetch(proto_tex, ivec3(ix, iy, layer), 0).r);
            result[c] = raw * proto_scale + proto_scaled_zp;
        } else {
            result[c] = 0.0;
        }
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
uniform vec4 src_extent;        // (u_min, v_min, u_max, v_max) a sample may reach
uniform vec4 dst_rect_px;       // (origin_x, origin_y, w, h) in pixel space
uniform vec4 pad_color;         // per-channel normalized pad value (RGBA)
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
    vec2 src_uv = clamp(src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw,
                        src_extent.xy, src_extent.zw);
    vec4 rgba = texture(u_tex, src_uv);
    frag_value = (channel == 0) ? rgba.r
               : (channel == 1) ? rgba.g : rgba.b;
}
"
}

/// RGBA8 → packed planar-float fragment shader (`PlanarRgb` / `PlanarRgba`,
/// F16 or F32).
///
/// The render target is a single `RGBA16F` (F16) or `RGBA32F` (F32) texture
/// sized `(W/4, C*H)`. Each output texel packs 4 float channel samples into
/// its four components, so a linear readout of the rendered texture produces
/// a tightly-packed `[C, H, W]` buffer (CHW order, one plane per row-band).
/// Width `W` must be a multiple of 4.
///
/// Normalization from RGBA8 → `[0, 1]` is performed for free by the hardware
/// texture fetch (`sampler2D` normalized fetch).  Letterboxing is applied via
/// `dst_rect_px`; pixels outside that rectangle are filled with `pad_color`.
///
/// Single source of truth lives in [`super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT`],
/// shared with the macOS IOSurface and Windows D3D11 texture paths.
// Consumed by `GLProcessorST` for the F16 NCHW PBO path and the F16/F32 NCHW
// zero-copy render paths.
pub(super) fn generate_planar_rgb_f16_packed_shader() -> &'static str {
    super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT
}

/// RGBA8 → packed interleaved-float fragment shader (`Rgb`, F16 or F32) for
/// the `(W*3/4, H)` zero-copy render surface.
///
/// Single source of truth lives in
/// [`super::shaders_common::FLOAT_NHWC_PACKED_FRAGMENT`].
// Consumed by `GLProcessorST` for the interleaved float zero-copy render path.
pub(super) fn generate_float_nhwc_packed_shader() -> &'static str {
    super::shaders_common::FLOAT_NHWC_PACKED_FRAGMENT
}

/// RGBA8 → float `Rgba` fragment shader for the `(W, H)` zero-copy render
/// surface (one texel per pixel, no packing).
///
/// Single source of truth lives in
/// [`super::shaders_common::FLOAT_RGBA_FRAGMENT`].
// Consumed by `GLProcessorST` for the float RGBA zero-copy render path.
pub(super) fn generate_float_rgba_shader() -> &'static str {
    super::shaders_common::FLOAT_RGBA_FRAGMENT
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
    // Single source of truth shared with the macOS backend; see
    // `shaders_common::NV_RGBA_FRAGMENT` and its byte-identity golden test.
    super::shaders_common::NV_RGBA_FRAGMENT
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
// Per-tensor colorimetry (YUV→RGB matrix + range); see the non-int8 variant.
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tc_precision {
    /// Every shader this module can link, paired with its source: each
    /// generator here, plus the `shaders_common` fragment constants that
    /// carry a `tc` and are linked directly rather than through a generator
    /// (`YUYV_RGBA_2D_FRAGMENT` is, at `processor/mod.rs`'s
    /// `yuyv_program_2d`).
    ///
    /// The list is written out by hand rather than discovered, so a new
    /// shader has to be added here deliberately —
    /// [`the_list_names_every_shader`] fails until it is. Add new shaders to
    /// this list.
    fn all_shaders() -> Vec<(&'static str, &'static str)> {
        vec![
            ("generate_vertex_shader", super::generate_vertex_shader()),
            (
                "generate_texture_fragment_shader",
                super::generate_texture_fragment_shader(),
            ),
            (
                "generate_texture_fragment_shader_yuv",
                super::generate_texture_fragment_shader_yuv(),
            ),
            (
                "generate_planar_rgb_shader",
                super::generate_planar_rgb_shader(),
            ),
            (
                "generate_texture_int8_shader",
                super::generate_texture_int8_shader(),
            ),
            (
                "generate_texture_int8_shader_yuv",
                super::generate_texture_int8_shader_yuv(),
            ),
            (
                "generate_planar_rgb_int8_shader",
                super::generate_planar_rgb_int8_shader(),
            ),
            (
                "generate_planar_rgb_shader_2d",
                super::generate_planar_rgb_shader_2d(),
            ),
            (
                "generate_planar_rgb_int8_shader_2d",
                super::generate_planar_rgb_int8_shader_2d(),
            ),
            (
                "generate_segmentation_shader",
                super::generate_segmentation_shader(),
            ),
            (
                "generate_instanced_segmentation_shader",
                super::generate_instanced_segmentation_shader(),
            ),
            (
                "generate_proto_segmentation_shader",
                super::generate_proto_segmentation_shader(),
            ),
            (
                "generate_proto_segmentation_shader_int8_nearest",
                super::generate_proto_segmentation_shader_int8_nearest(),
            ),
            (
                "generate_proto_segmentation_shader_int8_bilinear",
                super::generate_proto_segmentation_shader_int8_bilinear(),
            ),
            (
                "generate_proto_dequant_shader_int8",
                super::generate_proto_dequant_shader_int8(),
            ),
            (
                "generate_proto_segmentation_shader_f32",
                super::generate_proto_segmentation_shader_f32(),
            ),
            (
                "generate_packed_f32_nhwc_shader",
                super::generate_packed_f32_nhwc_shader(),
            ),
            (
                "generate_planar_rgb_f16_packed_shader",
                super::generate_planar_rgb_f16_packed_shader(),
            ),
            (
                "generate_float_nhwc_packed_shader",
                super::generate_float_nhwc_packed_shader(),
            ),
            (
                "generate_float_rgba_shader",
                super::generate_float_rgba_shader(),
            ),
            ("generate_color_shader", super::generate_color_shader()),
            (
                "generate_packed_rgba8_shader_2d",
                super::generate_packed_rgba8_shader_2d(),
            ),
            (
                "generate_packed_rgba8_int8_shader_2d",
                super::generate_packed_rgba8_int8_shader_2d(),
            ),
            (
                "generate_nv_to_rgba_shader_2d",
                super::generate_nv_to_rgba_shader_2d(),
            ),
            (
                "generate_nv_to_rgba_int8_shader_2d",
                super::generate_nv_to_rgba_int8_shader_2d(),
            ),
            (
                "generate_proto_repack_compute_shader",
                super::generate_proto_repack_compute_shader(),
            ),
            // Linked straight from the constant, with no generator wrapper.
            (
                "YUYV_RGBA_2D_FRAGMENT",
                super::super::shaders_common::YUYV_RGBA_2D_FRAGMENT,
            ),
            (
                "NV_RGBA_FRAGMENT",
                super::super::shaders_common::NV_RGBA_FRAGMENT,
            ),
        ]
    }

    /// `//` and `/* */` stripped, so a commented-out sample cannot be counted
    /// as a real one. Without this a commented `texture(tex, tc)` would raise
    /// the verbatim-fetch tally and mask a genuine arithmetic use beside it.
    fn strip_comments(src: &str) -> String {
        let b = src.as_bytes();
        let mut out = String::with_capacity(src.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
            } else {
                out.push(b[i] as char);
                i += 1;
            }
        }
        out
    }

    /// Byte offsets of every occurrence of the identifier `tc` — `tc` with a
    /// non-identifier character (or nothing) on each side, so `tc.x` counts
    /// and `tcoord` does not.
    fn tc_identifier_offsets(src: &str) -> Vec<usize> {
        let b = src.as_bytes();
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let mut out = Vec::new();
        let mut i = 0;
        while let Some(k) = src[i..].find("tc") {
            let at = i + k;
            let before_ok = at == 0 || !ident(b[at - 1]);
            let after_ok = at + 2 >= b.len() || !ident(b[at + 2]);
            if before_ok && after_ok {
                out.push(at);
            }
            i = at + 2;
        }
        out
    }

    /// How many of those occurrences are `tc` handed VERBATIM to a texture
    /// fetch — the exact shape `texture(<sampler>, tc)`, the one use that
    /// needs no arithmetic and so needs no precision promise.
    fn verbatim_fetch_count(src: &str) -> usize {
        let mut n = 0;
        let mut i = 0;
        while let Some(k) = src[i..].find("texture(") {
            let at = i + k + "texture(".len();
            let rest = &src[at..];
            let name_len = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(0);
            if name_len > 0 && rest[name_len..].starts_with(", tc)") {
                n += 1;
            }
            i = at;
        }
        n
    }

    /// The effective precision of the `in ... vec2 tc;` declaration, and
    /// whether the shader does anything with `tc` beyond handing it straight
    /// to a texture fetch. `None` when the shader has no `tc` input.
    fn tc_precision_and_use(src: &str) -> Option<(&'static str, bool)> {
        let src = &strip_comments(src);
        let decl = src
            .lines()
            .find(|l| l.trim_start().starts_with("in ") && l.trim_end().ends_with("vec2 tc;"))?;
        let declared = if decl.contains("highp") {
            Some("highp")
        } else if decl.contains("mediump") {
            Some("mediump")
        } else if decl.contains("lowp") {
            Some("lowp")
        } else {
            None
        };
        let file_default = src
            .lines()
            .find_map(|l| l.trim().strip_prefix("precision ")?.strip_suffix(" float;"))
            .map(|p| match p {
                "highp" => "highp",
                "mediump" => "mediump",
                "lowp" => "lowp",
                _ => "unknown",
            })
            .unwrap_or("unknown");
        let effective = declared.unwrap_or(file_default);
        // Every `tc` outside the declaration line, minus the ones that are a
        // verbatim `texture(sampler, tc)` argument.
        let in_decl = tc_identifier_offsets(decl).len();
        let total = tc_identifier_offsets(src).len() - in_decl;
        Some((effective, total > verbatim_fetch_count(src)))
    }

    /// A shader that does ARITHMETIC on `tc` must declare it `highp`.
    ///
    /// Handing a varying straight to `texture()` lets a driver give the
    /// interpolator's own coordinate to the texture unit, but any operation
    /// on it — `clamp()`, a `vec3(tc, i)` constructor, a multiply — runs on
    /// the ALU, and a `mediump` ALU is fp16 on Mali: a 2^-11 step, which is
    /// 0.625 texel on a 1280-wide source, and `LINEAR` blends the neighbour
    /// in by that fraction (the measurement is in the comment above
    /// [`super::generate_texture_fragment_shader`]). No other lane can catch
    /// it — Vivante, V3D and every desktop GL evaluate `mediump` at fp32, and
    /// the desktop cannot import a DMA-BUF at all — so this test is the only
    /// guard, and it is textual on purpose.
    ///
    /// The rule: count the `tc` identifiers outside the declaration line; a
    /// shader is doing arithmetic if any of them is not the second argument
    /// of a verbatim `texture(<sampler>, tc)`.
    #[test]
    fn shaders_that_compute_on_tc_declare_it_highp() {
        let mut offenders = Vec::new();
        let mut checked = 0;
        for (name, src) in all_shaders() {
            let Some((precision, computes)) = tc_precision_and_use(src) else {
                continue;
            };
            checked += 1;
            if computes && precision != "highp" {
                offenders.push(format!("{name} (tc is {precision})"));
            }
        }
        // Cross-check the line parser against a dumb substring test: if the
        // parser silently stops matching a declaration, the two disagree.
        let expected = all_shaders()
            .iter()
            .filter(|(_, src)| {
                [
                    "in vec2 tc;",
                    "in highp vec2 tc;",
                    "in mediump vec2 tc;",
                    "in lowp vec2 tc;",
                ]
                .iter()
                .any(|d| src.contains(d))
            })
            .count();
        assert_eq!(
            checked, expected,
            "the declaration parser matched {checked} shaders but {expected} contain an \
             `in ... vec2 tc;` declaration — the parser stopped matching one"
        );
        assert!(
            offenders.is_empty(),
            "these shaders do arithmetic on a `tc` that is not `highp`; on Mali the \
             fp16 ALU quantizes the sampled coordinate to a 2^-11 step, which is \
             0.625 texel on a 1280-wide source, and the measured shift was 0.40 \
             texel -- a truncation, not a rounding, so the error does not halve: \
             {offenders:?}"
        );
    }

    /// Every shader name the sources declare: each function in this file
    /// whose name starts with `generate_`, at ANY visibility, plus each
    /// `shaders_common` constant ending `_FRAGMENT` whose body carries a
    /// `tc`. Textual on purpose — it must not share the parser it checks.
    fn shader_names_in_source() -> Vec<String> {
        let mut names = Vec::new();
        let this = include_str!("shaders.rs");
        let mut i = 0;
        // A definition, not a call: the marker is the `fn ` keyword, which
        // `super::generate_x()` in the list above does not have.
        while let Some(k) = this[i..].find("fn generate_") {
            let at = i + k + "fn ".len();
            let rest = &this[at..];
            let n = rest.find('(').unwrap_or(0);
            if n > 0
                && rest[..n]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                names.push(rest[..n].to_string());
            }
            i = at;
        }
        let common = include_str!("shaders_common.rs");
        let mut i = 0;
        while let Some(k) = common[i..].find("const ") {
            let at = i + k + "const ".len();
            let rest = &common[at..];
            let n = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(0);
            let name = &rest[..n];
            // `_FRAGMENT` also filters out GLSL's own `const` declarations.
            if n > 0 && name.ends_with("_FRAGMENT") {
                let body = &common[at + n..];
                let end = body.find("\npub(crate) const ").unwrap_or(body.len());
                if body[..end].contains("vec2 tc;") {
                    names.push(name.to_string());
                }
            }
            i = at;
        }
        names
    }

    /// [`all_shaders`] is hand-written, so it can fall behind the sources.
    /// This pins it by NAME rather than by count, so an omission, a
    /// duplicate and a shader declared at a different visibility all fail,
    /// each naming what is wrong. A count alone would let an omission and an
    /// accidental duplicate cancel out.
    #[test]
    fn the_list_names_every_shader() {
        let listed: Vec<String> = all_shaders().iter().map(|(n, _)| n.to_string()).collect();
        let mut unique = listed.clone();
        unique.sort();
        let dup_len = unique.len();
        unique.dedup();
        let duplicates: Vec<&String> = if unique.len() == dup_len {
            Vec::new()
        } else {
            listed
                .iter()
                .filter(|n| listed.iter().filter(|m| m == n).count() > 1)
                .collect()
        };
        assert!(
            duplicates.is_empty(),
            "`all_shaders` lists these names more than once: {duplicates:?}"
        );

        let mut declared = shader_names_in_source();
        declared.sort();
        declared.dedup();
        let missing: Vec<&String> = declared.iter().filter(|n| !unique.contains(n)).collect();
        let extra: Vec<&String> = unique.iter().filter(|n| !declared.contains(n)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "`all_shaders` is out of step with the sources. Declared but not listed \
             (add them, so they are precision-checked): {missing:?}. Listed but not \
             declared (renamed or removed): {extra:?}"
        );
    }

    /// The `tc` identifier scan must not match a longer identifier, or the
    /// rule would fire on unrelated names.
    #[test]
    fn tc_scan_matches_the_identifier_only() {
        assert_eq!(tc_identifier_offsets("tc").len(), 1);
        assert_eq!(tc_identifier_offsets("tc.x * 2.0").len(), 1);
        assert_eq!(tc_identifier_offsets("vec3(tc, i)").len(), 1);
        assert_eq!(tc_identifier_offsets("tcoord + stc + tc_2").len(), 0);
        assert_eq!(verbatim_fetch_count("texture(tex, tc)"), 1);
        assert_eq!(verbatim_fetch_count("texture(tex, clamp(tc, a, b))"), 0);
        assert_eq!(verbatim_fetch_count("texture(tex, vec3(tc, i))"), 0);
    }

    /// A commented-out verbatim sample must not be counted, or it would
    /// cancel a real arithmetic use sitting beside it and let an offender
    /// through.
    #[test]
    fn a_commented_out_sample_does_not_mask_an_arithmetic_use() {
        let src = "\
#version 300 es
precision mediump float;
uniform sampler2D tex;
in vec2 tc;
out vec4 color;
void main(){
    // color = texture(tex, tc);
    color = texture(tex, clamp(tc, vec2(0.0), vec2(1.0)));
}
";
        assert_eq!(verbatim_fetch_count(&strip_comments(src)), 0);
        assert_eq!(
            tc_precision_and_use(src),
            Some(("mediump", true)),
            "the commented sample must not hide the clamp"
        );

        // The block-comment form, and a comment that is the ONLY use.
        let only_comment = "\
#version 300 es
precision mediump float;
in vec2 tc;
void main(){ /* texture(tex, tc) */ color = vec4(0.0); }
";
        assert_eq!(tc_precision_and_use(only_comment), Some(("mediump", false)));
    }
}
