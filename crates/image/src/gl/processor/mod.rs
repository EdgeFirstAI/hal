// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use drm_fourcc::DrmFourcc;
use edgefirst_decoder::{DetectBox, ProtoData, ProtoLayout, Segmentation};
use edgefirst_tensor::{
    PixelFormat, PixelLayout, Tensor, TensorMapTrait, TensorMemory, TensorTrait,
};
use khronos_egl::{self as egl, Attrib};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr};
use std::os::fd::AsRawFd;
use std::ptr::null_mut;
use std::rc::Rc;
use std::time::Instant;

use super::cache::CachedEglImage;
use super::EglDisplayKind;

use super::cache::{CacheKind, EglCacheKey, EglImageCache, GlCacheStats};
use super::context::{egl_ext, GlContext};
use super::resources::{Buffer, EglImage, FrameBuffer, GlProgram, Texture};
use super::shaders::*;
use super::{Int8InterpolationMode, RegionOfInterest, TransferBackend};
use crate::{Crop, Error, Flip, ImageProcessorTrait, ResolvedCrop, Rotation, DEFAULT_COLORS};
use edgefirst_tensor::TensorDyn;

/// Linux GL float (F16/F32) preprocessing paths. Declared as a child of
/// `processor` so its `impl GLProcessorST` block can reach this module's
/// private fields (the float programs, EGLImage caches, capability flags, …)
/// without promoting them to `pub(super)`.
mod float;
// Re-export the float classifier/support items at the `processor` module path
// so the dispatch in `convert()` and the `gl::tests` unit tests keep using
// `processor::{...}` paths unchanged.
pub(super) use float::{classify_float_render, float_render_support, FloatRenderPath};
// `dma_f16_packed_layout` is only reached through this re-export by the
// `gl::tests` unit tests (via `processor::dma_f16_packed_layout`); the lib body
// itself does not reference it, so the unused-import lint can't see the use.
#[allow(unused_imports)]
pub(super) use float::dma_f16_packed_layout;

/// Which GPU/CPU path was taken for the most recent NV* convert call.
///
/// Recorded by `draw_nv_texture_2d` / `draw_camera_texture_eglimage` / CPU
/// fallback so that tests and the profiler can assert that DMA NV* inputs never
/// silently fall back to CPU. Only meaningful immediately after a convert whose
/// source is `Nv12`/`Nv16`/`Nv24`; it is not reset for non-NV* converts (a
/// reader observing it after, say, an RGBA convert sees the prior NV* value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvConvertPath {
    /// Driver-decoded YUV via `samplerExternalOES` EGLImage. The GPU driver
    /// performs YUV→RGB (matrix + chroma upsampling); colorimetry is only as
    /// correct as the driver's EGL hint support. Required for true-multiplane
    /// NV12 (separate Y/UV fds). Was "Path A".
    ExternalSampler,
    /// R8 `texelFetch` shader applying the exact per-tensor YUV→RGB matrix in
    /// the shader (ES 3.0, no extension). Portable and identical across GPUs;
    /// no width constraint (uses the 64-aligned stride). Was "Path B".
    ShaderR8,
    /// CPU fallback (EGLImage creation failed or format not DMA-backed).
    Cpu,
    /// Not yet set (initial state, or non-NV* convert).
    None,
}

/// Client preference for the NV* GPU conversion path, set via
/// `EDGEFIRST_NV_CONVERT_PATH` (`sampler` | `shader` | `auto`).
///
/// `Auto` prefers [`NvConvertPath::ShaderR8`] (portable, colorimetry-exact)
/// wherever possible, using [`NvConvertPath::ExternalSampler`] only when the
/// shader path is impossible (true-multiplane NV12). The forced variants are
/// for benchmarking and platform bring-up; an impossible force logs a warning
/// and falls back rather than failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvPathPref {
    Auto,
    ForceSampler,
    ForceShader,
}

/// Bound destination render target for one convert, produced by
/// [`GLProcessorST::bind_dst`]. The GL counterpart of
/// [`super::render::DstLowering`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DstTarget {
    /// The destination buffer itself is the FBO colour attachment (EGLImage
    /// renderbuffer or texture): the render writes it directly, no readback.
    ZeroCopyImage,
    /// Offscreen texture render target; `readback` copies the result out.
    Texture { readback: DstReadback },
}

/// How a [`DstTarget::Texture`] render reaches the destination tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DstReadback {
    /// `glReadnPixels` straight into the mapped Mem tensor.
    Mem,
    /// `glReadnPixels` into this destination PBO's PACK binding.
    Pbo(u32),
}

impl DstReadback {
    /// The PBO id in the `Option` shape `readback_rendered` consumes.
    fn pbo_id(self) -> Option<u32> {
        match self {
            DstReadback::Mem => None,
            DstReadback::Pbo(id) => Some(id),
        }
    }
}

/// Borrowed proto data for the layered-float render plans: the
/// dispatcher's dtype arm keeps the `TensorMap` alive and hands the typed
/// view down to `render_proto_layers`, which matches it against
/// `ProtoPlan::upload`.
enum ProtoLayersData<'a> {
    F32(ndarray::ArrayView3<'a, f32>),
    F16(ndarray::ArrayView3<'a, half::f16>),
}

/// Uniform locations for one `nv_r8` program variant, resolved once at link
/// time — `glGetUniformLocation` is a per-call string lookup in the driver
/// and the NV draw previously issued 12 per frame.
#[derive(Clone, Copy)]
struct NvUniformLocs {
    img_size: i32,
    tex_width: i32,
    chroma_shift: i32,
    chroma_lines: i32,
    y_offset: i32,
    y_scale: i32,
    c_vr: i32,
    c_ug: i32,
    c_vg: i32,
    c_ub: i32,
}

/// Per-program uniform state for one `nv_r8` variant. Uniform values are
/// per-program GL state and persist across draws, so the constant sampler
/// binding (`src` = unit 0) is uploaded once at resolve time and
/// `last_colorimetry` lets the draw skip re-uploading the six YUV-matrix
/// floats while the source's (encoding, range) is unchanged.
struct NvUniformState {
    locs: NvUniformLocs,
    last_colorimetry: Option<(
        edgefirst_tensor::ColorEncoding,
        edgefirst_tensor::ColorRange,
    )>,
}

impl NvUniformState {
    fn resolve(program: &GlProgram) -> Self {
        unsafe {
            gls::gl::UseProgram(program.id);
            let loc =
                |name: &std::ffi::CStr| gls::gl::GetUniformLocation(program.id, name.as_ptr());
            // Constant: the NV shaders always sample `src` from unit 0.
            gls::gl::Uniform1i(loc(c"src"), 0);
            NvUniformState {
                locs: NvUniformLocs {
                    img_size: loc(c"img_size"),
                    tex_width: loc(c"tex_width"),
                    chroma_shift: loc(c"chroma_shift"),
                    chroma_lines: loc(c"chroma_lines"),
                    y_offset: loc(c"y_offset"),
                    y_scale: loc(c"y_scale"),
                    c_vr: loc(c"c_vr"),
                    c_ug: loc(c"c_ug"),
                    c_vg: loc(c"c_vg"),
                    c_ub: loc(c"c_ub"),
                },
                last_colorimetry: None,
            }
        }
    }
}

/// OpenGL single-threaded image converter.
pub struct GLProcessorST {
    camera_eglimage_texture: Texture,
    camera_normal_texture: Texture,
    render_texture: Texture,
    segmentation_texture: Texture,
    segmentation_program: GlProgram,
    instanced_segmentation_program: GlProgram,
    proto_texture: Texture,
    proto_segmentation_program: GlProgram,
    proto_segmentation_int8_nearest_program: GlProgram,
    proto_segmentation_int8_bilinear_program: GlProgram,
    proto_dequant_int8_program: GlProgram,
    proto_segmentation_f32_program: GlProgram,
    color_program: GlProgram,
    /// Last opacity value set on shader uniforms (avoids redundant GL calls).
    cached_opacity: f32,
    /// Allocated proto texture dimensions for SubImage3D fast path: (w, h, layers, internal_fmt).
    proto_tex_dims: (usize, usize, usize, u32),
    /// Whether GL_OES_texture_float_linear is available (allows GL_LINEAR on R32F textures).
    has_float_linear: bool,
    /// Whether GL_EXT_texture_format_BGRA8888 is available (allows BGRA destinations).
    pub(super) has_bgra: bool,
    /// Whether `GL_EXT_color_buffer_float` is advertised in the
    /// extension string, i.e. the GPU can render to an F32 color
    /// attachment. The probe is extension-string only — GLES 3.2 core
    /// also mandates F32 color buffers, but this flag does not consult
    /// `GL_VERSION`, so a 3.2-core context without the extension string
    /// reports `false`. Surfaced through `RenderDtypeSupport`; on Linux
    /// this gates the F32 NHWC PBO render path
    /// (`FloatRenderPath::PboF32Nhwc`) via `classify_float_render`.
    pub(super) supports_f32_color: bool,
    /// Whether `GL_EXT_color_buffer_half_float` is advertised, i.e. the
    /// GPU can render to an F16 color attachment. Surfaced through
    /// `RenderDtypeSupport`; on Linux this gates the F16 NCHW PBO and
    /// zero-copy DMA-BUF render paths (`FloatRenderPath::PboF16Nchw`,
    /// `FloatRenderPath::DmaF16Nchw`) via `classify_float_render`.
    pub(super) supports_f16_color: bool,
    /// Interpolation mode for int8 proto textures.
    int8_interpolation_mode: Int8InterpolationMode,
    /// Intermediate FBO texture for two-pass int8 dequant path.
    proto_dequant_texture: Texture,
    /// Allocated dequant texture dims for the recreate-on-change gate:
    /// (w, h, layers, internal_fmt). Mirrors `proto_tex_dims`.
    proto_dequant_tex_dims: (usize, usize, usize, u32),
    /// Persistent FBO for the two-pass int8 dequant render — previously a
    /// fresh `FrameBuffer::new()` per call.
    proto_dequant_fbo: FrameBuffer,
    /// Per-detection quad uniform locations, `(program_id, mask_coeff,
    /// class_index)`, re-resolved only when the bound proto program
    /// changes. `Cell` because the resolving call sites hold `&self`
    /// borrows of the program; id 0 = unresolved.
    proto_quad_locs: std::cell::Cell<(u32, i32, i32)>,
    /// Compute shader program for HWC→CHW proto repack (GLES 3.1 only).
    proto_repack_compute_program: Option<u32>,
    /// `(width, height, num_protos)` uniform locations for the compute
    /// program, resolved once at compile — previously three
    /// GetUniformLocation string lookups per dispatch.
    proto_compute_locs: (i32, i32, i32),
    /// SSBO for proto data upload (compute shader path).
    proto_ssbo: u32,
    /// Current allocated size of proto SSBO in bytes (0 = not allocated).
    proto_ssbo_size: usize,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    /// Persistent FBO for the convert() render path.
    /// Created once, reused by re-attaching textures each frame.
    convert_fbo: FrameBuffer,
    /// Persistent FBO for draw_decoded_masks / draw_proto_masks render path.
    /// Separate from convert_fbo so EGLImage binding state is not shared
    /// between convert and draw, avoiding redundant EGLImageTargetTexture2DOES
    /// calls on every frame when both paths are used.
    draw_fbo: FrameBuffer,
    /// Texture used as FBO color attachment for draw operations (DMA path).
    /// Separate from render_texture so EGLImageTargetTexture2DOES calls made
    /// by convert do not invalidate the draw path's bound EGLImage, and
    /// vice versa.
    draw_render_texture: Texture,
    /// EGLImage cache for source DMA buffers.
    src_egl_cache: EglImageCache,
    /// EGLImage cache for destination DMA buffers.
    dst_egl_cache: EglImageCache,
    /// Whether the BGRA byte-swap workaround warning has been logged.
    bgra_warned: bool,
    /// Whether the GPU is a Verisilicon/Vivante core (detected via GL_RENDERER).
    /// Used to block operations known to cause unrecoverable GPU hangs.
    pub(super) is_vivante: bool,
    /// Whether to use renderbuffer-backed EGLImages for DMA destinations.
    ///
    /// Set `EDGEFIRST_OPENGL_RENDERSURFACE=1` to enable (required on i.MX 95 / Mali-G310
    /// with Neutron NPU DMA-BUF destinations). Defaults to `false` (texture path) for
    /// 0.13.x compatibility with Vivante (i.MX 8MP). Will become the automatic default
    /// on non-Vivante platforms in a future release after broader testing.
    use_renderbuffer: bool,
    /// Intermediate RGBA texture for two-pass packed RGB conversion.
    /// Pass 1 renders YUYV/NV12→RGBA here; Pass 2 packs RGBA→RGB to DMA dest.
    packed_rgb_intermediate_tex: Texture,
    /// FBO for pass 1 of packed RGB conversion (renders to intermediate texture).
    packed_rgb_fbo: FrameBuffer,
    /// Current allocated size of the intermediate texture (0,0 = unallocated).
    packed_rgb_intermediate_size: (usize, usize),
    texture_program: GlProgram,
    texture_program_yuv: GlProgram,
    /// Int8 variant of texture_program — applies XOR 0x80 bias in fragment shader.
    texture_int8_program: GlProgram,
    /// Int8 variant of texture_program_yuv — applies XOR 0x80 bias in fragment shader.
    texture_int8_program_yuv: GlProgram,
    texture_program_planar: GlProgram,
    /// Shader: existing planar RGB with int8 bias (XOR 0x80) applied to output.
    texture_program_planar_int8: GlProgram,
    /// Shader: packed RGB -> RGBA8 packing (2D texture source, pass 2).
    packed_rgba8_program_2d: GlProgram,
    /// Shader: packed RGB int8 -> RGBA8 packing with XOR 0x80 (2D texture source, pass 2).
    packed_rgba8_int8_program_2d: GlProgram,
    /// Shader: planar RGB from 2D texture (two-pass NV12→RGBA→PlanarRgb workaround).
    texture_program_planar_2d: GlProgram,
    /// Shader: planar RGB int8 from 2D texture (two-pass NV12→RGBA→PlanarRgb workaround).
    texture_program_planar_int8_2d: GlProgram,
    /// Shader: RGBA8 → R32F-wide F32 NHWC `[H,W,3]` packed render (PBO float path).
    float_f32_nhwc_program: GlProgram,
    /// Shader: RGBA8 → RGBA16F-packed F16 NCHW `[3,H,W]` render (PBO float path).
    float_f16_nchw_program: GlProgram,
    /// Float render target texture (R32F or RGBA16F), reattached to `convert_fbo`.
    float_render_texture: Texture,
    /// Cached storage spec of `float_render_texture`: `(packed_w, packed_h,
    /// internal_format)`. Mirrors the `proto_tex_dims` pattern: only call
    /// `TexImage2D` (storage spec) when this changes; otherwise reuse the
    /// existing storage (skip the per-frame reallocation on the fixed-size
    /// video path). `(0, 0, 0)` means unallocated.
    float_render_tex_dims: (usize, usize, u32),
    /// Path B shader: NV12/NV16/NV24 R8 → RGBA8 (u8 output).
    nv_r8_program: GlProgram,
    /// Path B shader: NV12/NV16/NV24 R8 → RGBA8 with int8 XOR 0x80 bias.
    nv_r8_int8_program: GlProgram,
    /// Link-time uniform locations + colorimetry-upload skip for `nv_r8_program`.
    nv_r8_uniforms: NvUniformState,
    /// Same for `nv_r8_int8_program`.
    nv_r8_int8_uniforms: NvUniformState,
    /// Texture for the Path-B R8 EGLImage source (TEXTURE_2D, not EXTERNAL_OES).
    nv_r8_texture: Texture,
    /// EGLImage cache for Path-B R8 source imports (keyed like src_egl_cache).
    nv_r8_egl_cache: EglImageCache,
    /// Which path ran for the most recent NV* convert (instrumentation).
    pub(super) last_nv_convert_path: NvConvertPath,
    /// Client preference for NV* path selection (`EDGEFIRST_NV_CONVERT_PATH`).
    nv_path_pref: NvPathPref,
    /// Colorimetry/performance trade-off (see [`crate::ColorimetryMode`]).
    colorimetry_mode: crate::ColorimetryMode,
    /// `true` when `EDGEFIRST_COLORIMETRY` pinned the mode for this
    /// processor's lifetime — `set_colorimetry_mode` then logs and keeps it.
    colorimetry_env_pinned: bool,
    /// When `true`, a convert's terminal `glFinish` is skipped so a batch of
    /// tiles rendered into one shared destination import syncs only once, via a
    /// single `finish_via_fence` at [`flush`](Self::flush). Set for the duration
    /// of a single [`convert_deferred`](Self::convert_deferred) and reset
    /// unconditionally (even on error) so a subsequent standalone `convert` never
    /// silently returns on an unfinished GPU. Always `false` between calls.
    pub(super) defer_finish: bool,
    /// `true` when one or more `convert_deferred` calls have rendered without a
    /// `glFinish` and a [`flush`](Self::flush) (single `finish_via_fence`) is
    /// still owed. Read by the CUDA map path to auto-flush before handing the
    /// device a buffer whose batched render may still be in flight; cleared by
    /// the flush. See [`flush_pending`](Self::flush_pending).
    pub(super) pending_flush: bool,
    pub(super) gl_context: GlContext,
}

impl Drop for GLProcessorST {
    fn drop(&mut self) {
        unsafe {
            {
                if self.proto_ssbo != 0 {
                    gls::gl::DeleteBuffers(1, &self.proto_ssbo);
                }
                if let Some(program) = self.proto_repack_compute_program {
                    gls::gl::DeleteProgram(program);
                }
            }
        }
    }
}

/// Emit a warning the first time a draw operation falls back from the
/// DMA-BUF fast path to the CPU readback fallback. The fast path is silent;
/// the slow path is loud (once) so a regression — for example a tensor with
/// a non-aligned pitch or a missing extension — does not silently degrade
/// performance by 10–20× without anyone noticing.
///
/// The message includes the failing call site, the failing setup function,
/// and the underlying error so the user can map it back to the root cause
/// (commonly Mali's 64-byte pitch alignment requirement). Subsequent
/// fallbacks are demoted to debug-level.
fn warn_slow_path_once(call_site: &str, failing_setup: &str, err: &crate::Error) {
    use std::sync::Once;
    static SLOW_PATH_WARNED: Once = Once::new();
    let mut emitted = false;
    SLOW_PATH_WARNED.call_once(|| {
        log::warn!(
            "{call_site}: GL DMA-BUF fast path unavailable, falling back to CPU \
             readback (10–20× slower). Cause: {failing_setup} returned {err:?}. \
             On Mali Valhall (i.MX 95) this is usually because the destination \
             tensor's row pitch is not 64-byte aligned — see \
             `ImageProcessor::create_image` for automatic alignment. \
             Subsequent fallbacks will be logged at debug level."
        );
        emitted = true;
    });
    if !emitted {
        log::debug!("{call_site}: GL DMA-BUF fast path unavailable ({failing_setup}: {err:?})");
    }
}

/// Reinterpret a `&mut Tensor<i8>` as `&mut Tensor<u8>`.
///
/// # Safety
/// `i8` and `u8` have identical size, alignment, and validity for all bit
/// patterns. `Tensor<T>` stores data behind indirection (DMA-BUF fd, SHM
/// mapping, mmap'd memory, or PBO) — the `T` parameter affects only the
/// typed view returned by `map()`, not the struct layout. The GL backend
/// operates on raw bytes and applies XOR 0x80 bias either in the fragment
/// shader or as a CPU post-process, so the reinterpretation is semantically
/// correct. This transmutation must not be used to access `chroma()` through
/// the returned reference — the chroma `Box<Tensor<T>>` would also be
/// reinterpreted and its drop glue could theoretically differ.
unsafe fn tensor_i8_as_u8_mut(t: &mut Tensor<i8>) -> &mut Tensor<u8> {
    &mut *(t as *mut Tensor<i8> as *mut Tensor<u8>)
}

/// Reinterpret a `&Tensor<i8>` as `&Tensor<u8>`.
///
/// # Safety
/// Same rationale as [`tensor_i8_as_u8_mut`]. The returned reference must not
/// be used to access `chroma()`.
unsafe fn tensor_i8_as_u8(t: &Tensor<i8>) -> &Tensor<u8> {
    &*(t as *const Tensor<i8> as *const Tensor<u8>)
}

/// Extract `&Tensor<u8>` and `PixelFormat` from a `&TensorDyn` source.
/// For I8 sources, reinterprets the bytes as u8.
fn dyn_to_u8_src(src: &TensorDyn) -> crate::Result<(&Tensor<u8>, PixelFormat)> {
    match src {
        TensorDyn::U8(t) => {
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            Ok((t, fmt))
        }
        TensorDyn::I8(t) => {
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            // SAFETY: i8/u8 are layout-identical
            Ok((unsafe { tensor_i8_as_u8(t) }, fmt))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "GL backend requires u8 or i8 source, got {:?}",
            src.dtype()
        ))),
    }
}

/// Extract `&mut Tensor<u8>`, `PixelFormat`, and `is_int8` from a `&mut TensorDyn` destination.
/// For I8 destinations, reinterprets the bytes as u8 and sets `is_int8 = true`.
fn dyn_to_u8_dst(dst: &mut TensorDyn) -> crate::Result<(&mut Tensor<u8>, PixelFormat, bool)> {
    match dst {
        TensorDyn::U8(t) => {
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            Ok((t, fmt, false))
        }
        TensorDyn::I8(t) => {
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            // SAFETY: i8/u8 are layout-identical
            Ok((unsafe { tensor_i8_as_u8_mut(t) }, fmt, true))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "GL backend requires u8 or i8 destination, got {:?}",
            dst.dtype()
        ))),
    }
}

impl ImageProcessorTrait for GLProcessorST {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // A view()/batch() destination is lowered to a glViewport/scissor band
        // only on the u8/i8 DMA packed path (see `setup_renderbuffer_dma` +
        // `convert_to`). For any other GL destination — a non-DMA CPU upload, a
        // PBO, or a float render target — that lowering does not exist, so
        // decline and let the dispatcher fall back to the CPU backend, which
        // writes the sub-region correctly via offset + parent stride.
        if dst.view_origin().is_some()
            && !(self.gl_context.transfer_backend.is_dma()
                && dst.memory() == TensorMemory::Dma
                && matches!(
                    dst.dtype(),
                    edgefirst_tensor::DType::U8 | edgefirst_tensor::DType::I8
                ))
        {
            return Err(Error::NotSupported(
                "GL view()/batch() destination is supported only for u8/i8 DMA buffers; \
                 CPU fallback handles other cases"
                    .into(),
            ));
        }

        let crop = crop.resolve(
            src.width().unwrap_or(0),
            src.height().unwrap_or(0),
            dst.width().unwrap_or(0),
            dst.height().unwrap_or(0),
        )?;

        // F16/F32 destination: check for a GL float render path BEFORE the u8
        // extraction functions reject the dtype. When a path is found, dispatch
        // to convert_float_to_pbo (stub → NotSupported → CPU fallback). When
        // None, fall through to the existing u8 path unchanged.
        let dst_dtype = dst.dtype();
        if matches!(
            dst_dtype,
            edgefirst_tensor::DType::F16 | edgefirst_tensor::DType::F32
        ) {
            let src_fmt = src.format().ok_or(Error::NotAnImage)?;
            let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
            let support = float_render_support(
                self.is_vivante,
                self.supports_f32_color,
                self.supports_f16_color,
            );
            let path = classify_float_render(src_fmt, dst_fmt, dst_dtype, dst.memory(), support);
            match path {
                FloatRenderPath::DmaF16Nchw => {
                    return self.convert_float_to_dma(src, dst, rotation, flip, crop);
                }
                FloatRenderPath::None => {}
                _ => {
                    return self.convert_float_to_pbo(src, dst, path, rotation, flip, crop);
                }
            }
            // path == None: fall through to the u8 path, which will reject
            // F16/F32 via dyn_to_u8_dst → CPU fallback (existing behavior).
        }

        // Capture odd-destination dims before the mutable borrow below, for the
        // defense-in-depth wrap at the end (rule #2).
        let dst_odd = match (dst.width(), dst.height()) {
            (Some(w), Some(h)) if w % 2 != 0 || h % 2 != 0 => Some((w, h)),
            _ => None,
        };

        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        let (dst_u8, dst_fmt, is_int8) = dyn_to_u8_dst(dst)?;

        let result = self.convert_impl(
            src_u8, src_fmt, dst_u8, dst_fmt, is_int8, rotation, flip, crop,
        );

        // Defense-in-depth (rule #2): `Tensor::image` now 64-aligns DMA strides,
        // so image()-allocated destinations import fine at any width. But an
        // externally-imported destination with an odd, non-64-aligned stride can
        // still be rejected by a GPU that requires an aligned EGLImage pitch
        // (Mali `BadAlloc`, Vivante `BadAccess`). Surface such EGL/GL failures as
        // a descriptive `NotSupported` that PRESERVES the underlying error and
        // flags it as a platform-consistency limitation, rather than leaking a
        // raw `EGL(BadAlloc)`. (The F16/F32 float paths return earlier; their
        // destinations are 64-aligned by the same allocator fix.)
        match result {
            Err(e) if dst_odd.is_some() && matches!(e, Error::EGL(_) | Error::OpenGl(_)) => {
                let (w, h) = dst_odd.unwrap();
                Err(Error::NotSupported(format!(
                    "Conversion failed with {e:?} and target tensor has odd \
                     dimensions {w}x{h}, which is not supported by all platforms"
                )))
            }
            other => other,
        }
    }

    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        // Render without the terminal `glFinish` (gated in `convert_to` by
        // `defer_finish`), leaving a single `finish_via_fence` owed at `flush`.
        // The flag resets unconditionally so a later standalone `convert` always
        // finishes; only a successful deferred render arms `pending_flush`.
        self.defer_finish = true;
        let result = self.convert(src, dst, rotation, flip, crop);
        self.defer_finish = false;
        if result.is_ok() {
            self.pending_flush = true;
        }
        result
    }

    fn flush(&mut self) -> Result<(), crate::Error> {
        let _span = tracing::trace_span!("image.flush.gl", pending = self.pending_flush).entered();
        self.flush_pending();
        Ok(())
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<(), crate::Error> {
        let bg = overlay.background.map(|bg| dyn_to_u8_src(bg)).transpose()?;
        let (dst_u8, dst_fmt, _is_int8) = dyn_to_u8_dst(dst)?;
        self.draw_decoded_masks_impl(
            dst_u8,
            dst_fmt,
            detect,
            segmentation,
            overlay.opacity,
            bg,
            overlay.color_mode,
        )
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: crate::MaskOverlay<'_>,
    ) -> crate::Result<()> {
        let bg = overlay.background.map(|bg| dyn_to_u8_src(bg)).transpose()?;
        let (dst_u8, dst_fmt, _is_int8) = dyn_to_u8_dst(dst)?;
        self.draw_proto_masks_impl(
            dst_u8,
            dst_fmt,
            detect,
            proto_data,
            overlay.opacity,
            bg,
            overlay.color_mode,
        )
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> crate::Result<()> {
        if colors.is_empty() {
            return Ok(());
        }
        let mut colors_f32 = colors
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            })
            .take(20)
            .collect::<Vec<[f32; 4]>>();

        self.segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.instanced_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_nearest_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_bilinear_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_f32_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        colors_f32.iter_mut().for_each(|c| {
            c[3] = 1.0; // set alpha to 1.0 for color rendering
        });
        self.color_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        Ok(())
    }
}

/// Whether `EDGEFIRST_ALLOW_SOFTWARE_GL=1` is set, opting in to running the GL
/// backend on a software renderer (Mesa llvmpipe/softpipe/swrast).
///
/// Off by default and in production; the CI coverage lane sets it so the GL
/// render path executes on llvmpipe where no GPU exists.
fn software_gl_override_enabled() -> bool {
    std::env::var_os("EDGEFIRST_ALLOW_SOFTWARE_GL").is_some_and(|v| v == "1")
}

/// Decide whether to reject an initialized GL context for being a software
/// renderer. Pure so it is unit-testable without touching the environment or a
/// real GL context: reject only when the renderer is software AND the override
/// is not enabled.
fn should_reject_software_gl(is_software_renderer: bool, override_enabled: bool) -> bool {
    is_software_renderer && !override_enabled
}

impl GLProcessorST {
    /// Issue the single batched GPU sync if a `convert_deferred` is still owed,
    /// then clear the pending flag. No-op when nothing is pending. Used by both
    /// [`flush`](ImageProcessorTrait::flush) and the CUDA map path (which
    /// auto-flushes before handing the device a possibly-in-flight buffer).
    ///
    /// # Safety contract
    /// Must run on the GL worker thread that owns the context (it is, being
    /// driven from the worker loop under `GL_MUTEX`).
    pub(super) fn flush_pending(&mut self) {
        if self.pending_flush {
            // SAFETY: called on the context-owning GL worker thread.
            unsafe { float::finish_via_fence() };
            self.pending_flush = false;
        }
    }

    pub fn new(kind: Option<EglDisplayKind>) -> Result<GLProcessorST, crate::Error> {
        // Display bring-up goes through the platform seam — the contract a
        // future platform (Windows/ANGLE) implements instead of forking this
        // engine. On Linux this delegates straight to `GlContext::new`.
        let gl_context =
            <super::platform::Platform as super::platform::GlPlatform>::init_display(kind)?;
        // Load the GL function pointers exactly once per process (the same
        // pattern as macOS's GL_LOADED). `gls` bindings are gl_generator
        // `static mut` function-pointer tables, so re-running `load_with` on
        // every processor construction is a data race the moment another
        // processor's worker thread is calling GL without holding `GL_MUTEX`
        // — a prerequisite for scoping the per-message lock to Vivante.
        // Once-loading is also semantically right: the pointers come from the
        // process-global shared EGL display, so every context resolves the
        // same addresses.
        static GL_LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        GL_LOADED.get_or_init(|| {
            gls::load_with(|s| {
                gl_context
                    .egl
                    .get_proc_address(s)
                    .map_or(std::ptr::null(), |p| p as *const _)
            });
        });

        let (
            has_float_linear,
            has_bgra,
            is_vivante,
            is_software_renderer,
            supports_f32_color,
            supports_f16_color,
        ) = Self::gl_check_support()?;

        // Software renderers (llvmpipe, softpipe, swrast) are CPU-based OpenGL
        // implementations that are slower and less capable than our native CPU
        // backend. Reject them early — the caller falls back to CPU automatically.
        //
        // `EDGEFIRST_ALLOW_SOFTWARE_GL=1` overrides the rejection. It exists
        // for the CI coverage lane, which runs the GL render path (e.g. the
        // float PBO roundtrips) on Mesa llvmpipe where no GPU is available —
        // the rendered output is numerically identical to a hardware GPU since
        // it runs the same GLSL. Production never sets it, so the default
        // (reject software GL, fall back to CPU) is unchanged.
        if should_reject_software_gl(is_software_renderer, software_gl_override_enabled()) {
            return Err(crate::Error::NotSupported(
                "software OpenGL renderer detected (llvmpipe/softpipe/swrast); \
                 GL backend disabled — check EGL ICD configuration if a \
                 hardware GPU is expected (set EDGEFIRST_ALLOW_SOFTWARE_GL=1 \
                 to override, e.g. for coverage on Mesa llvmpipe)"
                    .into(),
            ));
        }
        if is_software_renderer {
            log::warn!(
                "software OpenGL renderer in use (EDGEFIRST_ALLOW_SOFTWARE_GL=1); \
                 slower than the CPU backend — intended for CI coverage only"
            );
        }

        // Uploads and downloads are all packed with no alignment requirements
        unsafe {
            gls::gl::PixelStorei(gls::gl::PACK_ALIGNMENT, 1);
            gls::gl::PixelStorei(gls::gl::UNPACK_ALIGNMENT, 1);
        }

        let texture_program_planar =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader())?;

        let texture_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_fragment_shader())?;

        let texture_program_yuv = GlProgram::new(
            generate_vertex_shader(),
            generate_texture_fragment_shader_yuv(),
        )?;

        let texture_int8_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader())?;
        let texture_int8_program_yuv =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader_yuv())?;

        let segmentation_program =
            GlProgram::new(generate_vertex_shader(), generate_segmentation_shader())?;
        segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;
        let instanced_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_instanced_segmentation_shader(),
        )?;
        instanced_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Existing f16 proto shader (RGBA16F, 4 protos per layer)
        let proto_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader(),
        )?;
        proto_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Int8 proto shaders (R8I, 1 proto per layer, 32 layers)
        let proto_segmentation_int8_nearest_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_nearest(),
        )?;
        proto_segmentation_int8_nearest_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_segmentation_int8_bilinear_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_bilinear(),
        )?;
        proto_segmentation_int8_bilinear_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_dequant_int8_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_dequant_shader_int8(),
        )?;

        // F32 proto shader (R32F, 1 proto per layer, 32 layers)
        let proto_segmentation_f32_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_f32(),
        )?;
        proto_segmentation_f32_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let color_program = GlProgram::new(generate_vertex_shader(), generate_color_shader())?;
        color_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Int8 variant of the existing planar RGB shader (for planar RGB int8 destinations).
        let texture_program_planar_int8 =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_int8_shader())?;

        // Planar RGB shaders with sampler2D (for two-pass NV12→RGBA→PlanarRgb on Vivante)
        let texture_program_planar_2d =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader_2d())?;
        let texture_program_planar_int8_2d = GlProgram::new(
            generate_vertex_shader(),
            generate_planar_rgb_int8_shader_2d(),
        )?;

        // RGB packing shaders (2D only — used in pass 2 of two-pass pipeline)
        let packed_rgba8_program_2d =
            GlProgram::new(generate_vertex_shader(), generate_packed_rgba8_shader_2d())?;
        let packed_rgba8_int8_program_2d = GlProgram::new(
            generate_vertex_shader(),
            generate_packed_rgba8_int8_shader_2d(),
        )?;

        // Float render-to-PBO programs. Both are full-viewport fragment
        // shaders driven by `gl_FragCoord`; the standard vertex shader emits a
        // full-screen quad so every packed output texel is visited. Crop and
        // letterbox are entirely uniform-driven (src_rect_uv/dst_rect_px/
        // pad_color), matching the macOS IOSurface contract.
        let float_f32_nhwc_program =
            GlProgram::new(generate_vertex_shader(), generate_packed_f32_nhwc_shader())?;
        let float_f16_nchw_program = GlProgram::new(
            generate_vertex_shader(),
            generate_planar_rgb_f16_packed_shader(),
        )?;

        // Path B: NV12/NV16/NV24 → RGBA via R8 texelFetch shader (ES 3.0 core, no extension).
        let nv_r8_program =
            GlProgram::new(generate_vertex_shader(), generate_nv_to_rgba_shader_2d())?;
        let nv_r8_int8_program = GlProgram::new(
            generate_vertex_shader(),
            generate_nv_to_rgba_int8_shader_2d(),
        )?;
        // Resolve uniform locations once at link time (the NV draw is per-frame)
        // and upload the constant sampler bindings while the programs are fresh:
        // pass-2 packing samples the intermediate on unit 1, planar-2d on unit 0.
        let nv_r8_uniforms = NvUniformState::resolve(&nv_r8_program);
        let nv_r8_int8_uniforms = NvUniformState::resolve(&nv_r8_int8_program);
        packed_rgba8_program_2d.load_uniform_1i(c"tex", 1)?;
        packed_rgba8_int8_program_2d.load_uniform_1i(c"tex", 1)?;
        texture_program_planar_2d.load_uniform_1i(c"tex", 0)?;
        texture_program_planar_int8_2d.load_uniform_1i(c"tex", 0)?;

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let draw_render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let proto_texture = Texture::new();
        let proto_dequant_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

        // EGLImage cache capacity (per cache: src / dst / nv_r8). The key carries
        // geometry, so a pool buffer reused at N distinct sizes needs N live
        // EGLImages to avoid evict/re-import churn; a parallel decode pool wants
        // headroom for (pool slots × distinct sizes). Capacity is the eviction
        // bound only — EGLImages are lightweight views into the tensor's existing
        // DMA-BUF (no pixel copy) and are created on demand, so a large default is
        // free for fixed-dimension workloads (live camera) that only ever use a
        // size or two. Override with EDGEFIRST_EGL_CACHE_CAPACITY for high-
        // cardinality varied-size streams (e.g. dataset validation).
        const DEFAULT_EGL_CACHE_CAPACITY: usize = 64;
        let egl_cache_capacity = std::env::var("EDGEFIRST_EGL_CACHE_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&c| c > 0)
            .unwrap_or(DEFAULT_EGL_CACHE_CAPACITY);

        // NV* conversion-path preference. `auto` (default) prefers the portable,
        // colorimetry-exact in-shader ShaderR8 path; `sampler`/`shader` force a
        // path for benchmarking / platform bring-up (an impossible force warns
        // and falls back). Mirrors the EDGEFIRST_FORCE_TRANSFER idiom below.
        let nv_path_pref = match std::env::var("EDGEFIRST_NV_CONVERT_PATH") {
            Ok(v) => match v.to_ascii_lowercase().as_str() {
                "sampler" | "external" | "a" => NvPathPref::ForceSampler,
                "shader" | "r8" | "b" => NvPathPref::ForceShader,
                "auto" | "" => NvPathPref::Auto,
                other => {
                    log::warn!(
                        "EDGEFIRST_NV_CONVERT_PATH={other:?} not recognised \
                         (expected sampler|shader|auto), using auto"
                    );
                    NvPathPref::Auto
                }
            },
            Err(_) => NvPathPref::Auto,
        };
        if nv_path_pref != NvPathPref::Auto {
            log::info!("EDGEFIRST_NV_CONVERT_PATH override: {nv_path_pref:?}");
        }

        // Colorimetry/performance trade-off. The env var pins the mode for the
        // processor's lifetime (set_colorimetry_mode logs and keeps it);
        // otherwise the config default is Fast (issue #106 policy) and
        // `set_colorimetry_mode` may change it.
        let colorimetry_env = match std::env::var("EDGEFIRST_COLORIMETRY") {
            Ok(v) => match v.to_ascii_lowercase().as_str() {
                "exact" => Some(crate::ColorimetryMode::Exact),
                "fast" => Some(crate::ColorimetryMode::Fast),
                "" => None,
                other => {
                    log::warn!(
                        "EDGEFIRST_COLORIMETRY={other:?} not recognised \
                         (expected fast|exact), ignoring"
                    );
                    None
                }
            },
            Err(_) => None,
        };
        if let Some(mode) = colorimetry_env {
            log::info!("EDGEFIRST_COLORIMETRY override: {mode:?}");
        }

        let mut converter = GLProcessorST {
            gl_context,
            texture_program,
            texture_program_yuv,
            texture_int8_program,
            texture_int8_program_yuv,
            texture_program_planar,
            texture_program_planar_int8,
            packed_rgba8_program_2d,
            packed_rgba8_int8_program_2d,
            texture_program_planar_2d,
            texture_program_planar_int8_2d,
            float_f32_nhwc_program,
            float_f16_nchw_program,
            float_render_texture: Texture::new(),
            float_render_tex_dims: (0, 0, 0),
            camera_eglimage_texture,
            camera_normal_texture,
            segmentation_texture,
            proto_texture,
            proto_segmentation_int8_nearest_program,
            proto_segmentation_int8_bilinear_program,
            proto_dequant_int8_program,
            proto_segmentation_f32_program,
            has_float_linear,
            has_bgra,
            supports_f32_color,
            supports_f16_color,
            int8_interpolation_mode: Int8InterpolationMode::Bilinear,
            proto_dequant_texture,
            proto_dequant_tex_dims: (0, 0, 0, 0),
            proto_dequant_fbo: FrameBuffer::new(),
            proto_quad_locs: std::cell::Cell::new((0, -1, -1)),
            vertex_buffer,
            texture_buffer,
            convert_fbo: FrameBuffer::new(),
            draw_fbo: FrameBuffer::new(),
            draw_render_texture,
            src_egl_cache: EglImageCache::new(egl_cache_capacity),
            dst_egl_cache: EglImageCache::new(egl_cache_capacity),
            bgra_warned: false,
            is_vivante,
            use_renderbuffer: std::env::var("EDGEFIRST_OPENGL_RENDERSURFACE")
                .map(|v| v == "1")
                .unwrap_or(false),
            packed_rgb_intermediate_tex: Texture::new(),
            packed_rgb_fbo: FrameBuffer::new(),
            packed_rgb_intermediate_size: (0, 0),
            render_texture,
            segmentation_program,
            instanced_segmentation_program,
            proto_segmentation_program,
            color_program,
            cached_opacity: f32::NAN, // sentinel: forces first set_opacity_uniform to initialize all shaders
            proto_tex_dims: (0, 0, 0, 0),
            proto_repack_compute_program: None,
            proto_compute_locs: (-1, -1, -1),
            proto_ssbo: 0,
            proto_ssbo_size: 0,
            nv_r8_program,
            nv_r8_int8_program,
            nv_r8_uniforms,
            nv_r8_int8_uniforms,
            nv_r8_texture: Texture::new(),
            nv_r8_egl_cache: EglImageCache::new(egl_cache_capacity),
            last_nv_convert_path: NvConvertPath::None,
            nv_path_pref,
            colorimetry_mode: colorimetry_env.unwrap_or_default(),
            colorimetry_env_pinned: colorimetry_env.is_some(),
            defer_finish: false,
            pending_flush: false,
        };
        check_gl_error(function!(), line!())?;

        // Compile compute shader for proto repack if GLES 3.1 is available.
        // Enabled via EDGEFIRST_PROTO_COMPUTE=1 while validating on target GPUs.
        let compute_enabled = converter.gl_context.has_compute
            && std::env::var("EDGEFIRST_PROTO_COMPUTE")
                .map(|v| v == "1")
                .unwrap_or(false);
        if compute_enabled {
            match Self::compile_compute_program(generate_proto_repack_compute_shader()) {
                Ok(program) => {
                    log::info!("Proto repack compute shader compiled successfully");
                    converter.proto_repack_compute_program = Some(program);
                    unsafe {
                        converter.proto_compute_locs = (
                            gls::gl::GetUniformLocation(program, c"width".as_ptr()),
                            gls::gl::GetUniformLocation(program, c"height".as_ptr()),
                            gls::gl::GetUniformLocation(program, c"num_protos".as_ptr()),
                        );
                    }
                    let mut ssbo = 0u32;
                    unsafe { gls::gl::GenBuffers(1, &mut ssbo) };
                    converter.proto_ssbo = ssbo;
                }
                Err(e) => {
                    log::warn!("Proto repack compute shader failed: {e}; using CPU fallback");
                }
            }
        }

        log::debug!(
            "GLProcessorST: DMA destination attachment mode: {}",
            if converter.use_renderbuffer {
                "renderbuffer (EDGEFIRST_OPENGL_RENDERSURFACE=1)"
            } else {
                "texture (default)"
            }
        );

        // Verify DMA-buf actually works (catches NVIDIA discrete GPUs where
        // EGLImage creation succeeds but rendered data is all zeros)
        if converter.gl_context.transfer_backend.is_dma() && !converter.verify_dma_buf_roundtrip() {
            log::info!("DMA-buf verification failed — falling back to PBO transfers");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
        }

        // If DMA-buf failed/unavailable but GL is alive, use PBO transfers.
        // Software renderers never reach here — they are rejected above.
        if converter.gl_context.transfer_backend == TransferBackend::Sync {
            log::info!("Upgrading transfer backend from Sync to Pbo (GL context available)");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
        }

        // Allow env-var override for benchmarking specific transfer paths.
        // Values: "dmabuf", "pbo", "sync" (case-insensitive).
        if let Ok(val) = std::env::var("EDGEFIRST_FORCE_TRANSFER") {
            let forced = match val.to_ascii_lowercase().as_str() {
                "dmabuf" | "dma" => Some(TransferBackend::DmaBuf),
                "pbo" => Some(TransferBackend::Pbo),
                "sync" => Some(TransferBackend::Sync),
                other => {
                    log::warn!(
                        "EDGEFIRST_FORCE_TRANSFER={other:?} not recognised \
                         (expected dmabuf|pbo|sync), ignoring"
                    );
                    None
                }
            };
            if let Some(backend) = forced {
                log::info!(
                    "EDGEFIRST_FORCE_TRANSFER override: {:?} → {backend:?}",
                    converter.gl_context.transfer_backend
                );
                converter.gl_context.transfer_backend = backend;
            }
        }

        log::debug!(
            "GLConverter created (transfer={:?})",
            converter.gl_context.transfer_backend,
        );
        Ok(converter)
    }

    /// Verify that DMA-buf EGLImage round-trip actually works on this GPU.
    ///
    /// Renders a solid red quad to a 64x64 DMA-buf-backed RGBA texture via
    /// EGLImage, then reads it back and checks that the center pixel is red.
    /// Returns `true` if the data round-trips correctly.
    ///
    /// This catches GPUs like NVIDIA discrete where `eglCreateImage` from
    /// `dma_heap` fds succeeds but the rendered data is all zeros.
    fn verify_dma_buf_roundtrip(&mut self) -> bool {
        // Allocate a 64x64 RGBA DMA source tensor and fill it with solid red
        let src = match Tensor::<u8>::image(64, 64, PixelFormat::Rgba, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA source: {e}");
                return false;
            }
        };

        {
            let mut map = match src.map() {
                Ok(m) => m,
                Err(e) => {
                    log::info!("verify_dma_buf_roundtrip: failed to map DMA source: {e}");
                    return false;
                }
            };
            for pixel in map.chunks_exact_mut(4) {
                pixel[0] = 255; // R
                pixel[1] = 0; // G
                pixel[2] = 0; // B
                pixel[3] = 255; // A
            }
        }

        // Allocate a 64x64 RGBA DMA destination tensor
        let mut dst = match Tensor::<u8>::image(64, 64, PixelFormat::Rgba, Some(TensorMemory::Dma))
        {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA destination: {e}");
                return false;
            }
        };

        // Run the full DMA-buf EGLImage render pipeline (RGBA→RGBA DMA is a
        // single-pass zero-copy plan through the engine).
        if let Err(e) = self.convert_via_engine(
            &mut dst,
            PixelFormat::Rgba,
            &src,
            PixelFormat::Rgba,
            false,
            Rotation::None,
            Flip::None,
            ResolvedCrop::no_crop(),
        ) {
            log::info!("verify_dma_buf_roundtrip: convert failed: {e}");
            return false;
        }

        // Read back the center pixel at (32, 32) from the destination
        let map = match dst.map() {
            Ok(m) => m,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to map DMA destination: {e}");
                return false;
            }
        };

        let offset = (32 * 64 + 32) * 4;
        if map.len() < offset + 4 {
            log::info!("verify_dma_buf_roundtrip: destination buffer too small");
            return false;
        }

        let r = map[offset];
        let g = map[offset + 1];
        let b = map[offset + 2];
        let a = map[offset + 3];

        let pass = r > 250 && g < 5 && b < 5 && a > 250;

        if pass {
            log::info!("verify_dma_buf_roundtrip: PASSED (center pixel RGBA={r},{g},{b},{a})");
        } else {
            log::info!(
                "verify_dma_buf_roundtrip: FAILED (center pixel RGBA={r},{g},{b},{a}, \
                 expected ~255,0,0,255)"
            );
        }

        pass
    }

    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) {
        self.int8_interpolation_mode = mode;
        log::debug!("Int8 interpolation mode set to {:?}", mode);
    }

    /// Sets the colorimetry/performance trade-off (see
    /// [`crate::ColorimetryMode`]). When `EDGEFIRST_COLORIMETRY` pinned the
    /// mode at construction, the env value wins: this logs and keeps it.
    pub fn set_colorimetry_mode(&mut self, mode: crate::ColorimetryMode) {
        if self.colorimetry_env_pinned {
            if mode != self.colorimetry_mode {
                log::info!(
                    "ColorimetryMode::{mode:?} requested but EDGEFIRST_COLORIMETRY pins \
                     {:?} — keeping the env override",
                    self.colorimetry_mode
                );
            }
            return;
        }
        self.colorimetry_mode = mode;
        log::debug!("Colorimetry mode set to {:?}", mode);
    }

    /// Snapshot the EGLImage cache counters (src, dst, NV R8).
    ///
    /// Steady-state tests capture this after warmup and after an N-frame
    /// loop and assert `total_misses()` stays flat — the cache-behavior
    /// equality gate for GL refactors.
    pub(crate) fn egl_cache_stats(&self) -> GlCacheStats {
        GlCacheStats {
            src: self.src_egl_cache.stats(),
            dst: self.dst_egl_cache.stats(),
            nv_r8: self.nv_r8_egl_cache.stats(),
        }
    }

    // Internal methods operating on Tensor<u8> + PixelFormat directly.

    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_impl(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let _span = tracing::trace_span!(
            "image.convert.gl",
            ?src_fmt,
            ?dst_fmt,
            is_int8,
            src_memory = ?src.memory(),
            dst_memory = ?dst.memory(),
        )
        .entered();
        if !Self::check_src_format_supported(self.gl_context.transfer_backend, src, src_fmt) {
            if src_fmt == PixelFormat::Vyuy
                && self.gl_context.transfer_backend.is_dma()
                && src.memory() == TensorMemory::Dma
            {
                log::warn!(
                    "VYUY format not supported via EGL DMA-BUF import; \
                     falling back to CPU/G2D path"
                );
            }
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {src_fmt} source texture",
            )));
        }

        if !Self::check_dst_format_supported(
            self.gl_context.transfer_backend,
            dst,
            dst_fmt,
            is_int8,
            self.has_bgra,
        ) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {dst_fmt} destination texture",
            )));
        }

        log::debug!(
            "dst tensor: {:?} src tensor :{:?}",
            dst.memory(),
            src.memory()
        );
        check_gl_error(function!(), line!())?;
        log::trace!(
            "GL convert_impl: {src_fmt}→{dst_fmt} int8={is_int8} \
             src_mem={:?} dst_mem={:?} transfer={:?}",
            src.memory(),
            dst.memory(),
            self.gl_context.transfer_backend,
        );

        self.convert_via_engine(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop)
    }

    /// The single u8/i8 convert engine: classify the destination
    /// ([`super::render::lower_dst`]), pick the render plan
    /// ([`super::render::plan_convert`]), then `bind_dst` → render →
    /// readback. The two-pass plans delegate to their render strategies;
    /// every single-pass convert — DMA, Mem, or PBO on either side — runs
    /// the same code.
    #[allow(clippy::too_many_arguments)]
    fn convert_via_engine(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let lowering =
            super::render::lower_dst(self.gl_context.transfer_backend.is_dma(), dst.memory());
        let plan = super::render::plan_convert(src_fmt, dst_fmt, lowering);
        let _span = tracing::trace_span!(
            "image.convert.gl.engine",
            ?plan,
            ?lowering,
            src_pbo = src.memory() == TensorMemory::Pbo,
        )
        .entered();

        // v1 view()/batch() destination support covers only the single-pass
        // PACKED zero-copy path (the glViewport/scissor band set in
        // bind_dst). Packed RGB and every planar layout reinterpret the
        // destination geometry (`W*3/4 × H`, `H*3` bands), which the band
        // lowering does not yet handle, so a view destination declines them
        // → CPU fallback. (Band tiling for those is a follow-up.)
        if dst.view_origin().is_some()
            && lowering == super::render::DstLowering::ZeroCopy
            && (dst_fmt == PixelFormat::Rgb || dst_fmt.layout() == PixelLayout::Planar)
        {
            return Err(crate::Error::NotSupported(
                "GL view()/batch() destination not yet supported for two-pass packed-RGB / \
                 planar formats (CPU fallback handles it)"
                    .into(),
            ));
        }

        match plan {
            super::render::ConvertPlan::TwoPassPackedRgb => {
                return self.convert_to_packed_rgb(
                    src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop,
                )
            }
            super::render::ConvertPlan::TwoPassNvPlanar => {
                return self.convert_nv_to_planar_two_pass(
                    src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop,
                )
            }
            super::render::ConvertPlan::SinglePass => {}
        }

        let target = self.bind_dst(dst, dst_fmt, crop)?;

        // Bias the letterbox clear colour for int8 on every lowering: the
        // fragment shader XORs rendered pixels and no readback un-biases,
        // so the glClear'd letterbox region must be pre-biased.
        let crop = Self::int8_bias_clear(is_int8, crop);

        let start = Instant::now();
        match src.as_pbo() {
            Some(src_pbo) => {
                // A PBO source uploads via its UNPACK binding — mapping it on
                // the GL thread would deadlock on the Pbo message round-trip.
                if src_pbo.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert from a mapped PBO tensor".to_string(),
                    ));
                }
                let src_buffer_id = src_pbo.buffer_id();
                self.draw_src_texture_from_pbo(
                    src,
                    src_fmt,
                    src_buffer_id,
                    dst,
                    dst_fmt,
                    is_int8,
                    rotation,
                    flip,
                    crop,
                )?;
            }
            None => {
                self.render_packed_or_planar(
                    src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop,
                )?;
            }
        }
        log::debug!("engine render ({plan:?}) takes {:?}", start.elapsed());

        if let DstTarget::Texture { readback } = target {
            // Data is already int8-biased by the shader — readback copies bytes.
            let start = Instant::now();
            self.readback_rendered(dst, dst_fmt, readback.pbo_id())?;
            log::debug!("engine readback takes {:?}", start.elapsed());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_decoded_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        opacity: f32,
        background: Option<(&Tensor<u8>, PixelFormat)>,
        color_mode: crate::ColorMode,
    ) -> Result<(), crate::Error> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_decoded_masks");
        if !matches!(
            dst_fmt,
            PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb
        ) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let memory = dst.memory();
        // Trace logs are expensive to format (struct Debug, env reads). Gate
        // on the global level filter so they're a single integer compare
        // when trace logging is disabled.
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "draw_decoded_masks: dst.memory()={memory:?} {dst_w}x{dst_h} fmt={dst_fmt:?}"
            );
        }
        let pbo_buffer_id = if memory == TensorMemory::Pbo {
            match dst.as_pbo() {
                Some(p) if !p.is_mapped() => Some(p.buffer_id()),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            TensorMemory::Dma => match self.setup_draw_renderbuffer_dma(dst, dst_fmt) {
                Ok(()) => {
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "draw_decoded_masks: DMA fast path (setup_draw_renderbuffer_dma OK)"
                        );
                    }
                    true
                }
                Err(e) => {
                    warn_slow_path_once("draw_decoded_masks", "setup_draw_renderbuffer_dma", &e);
                    if let Some(buffer_id) = pbo_buffer_id {
                        self.setup_renderbuffer_from_pbo(dst, dst_fmt, buffer_id)?;
                    } else {
                        self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                    }
                    false
                }
            },
            _ if pbo_buffer_id.is_some() => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_decoded_masks: PBO path");
                }
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_decoded_masks: non-DMA fallback (memory={memory:?})");
                }
                self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                false
            }
        };

        // Write the base layer of the framebuffer *before* running mask
        // passes.  The framebuffer's backing storage (DMA renderbuffer or
        // render_texture) may contain stale pixels from a previous frame
        // in a triple-buffered pipeline, so we always actively write it.
        //
        // - background + DMA: GPU-blit bg → renderbuffer via EGLImage
        //   (zero-copy).
        // - background + non-DMA: upload bg pixels into the render_texture
        //   that backs the FBO (glTexImage2D). The CPU memcpy is into a
        //   GL texture, not into the user's DMA buffer, so it's a pure
        //   upload — not a cache-flushing mapped copy.
        // - no background: glClear(0x00000000) on the framebuffer.
        if let Some((bg, bg_fmt)) = background {
            if bg.width() != Some(dst_w) || bg.height() != Some(dst_h) {
                return Err(crate::Error::InvalidShape(
                    "background dimensions do not match dst".into(),
                ));
            }
            if is_dma && bg.memory() == TensorMemory::Dma {
                gls::disable(gls::gl::BLEND);
                let bg_egl = self.get_or_create_egl_image(CacheKind::Src, bg, bg_fmt)?;
                self.draw_camera_texture_eglimage(
                    bg,
                    bg_fmt,
                    bg_egl,
                    RegionOfInterest {
                        left: 0.0,
                        top: 1.0,
                        right: 1.0,
                        bottom: 0.0,
                    },
                    RegionOfInterest {
                        left: -1.0,
                        top: 1.0,
                        right: 1.0,
                        bottom: -1.0,
                    },
                    0,
                    crate::Flip::None,
                    false,
                )?;
            } else {
                // Non-DMA background: upload bg pixels into the
                // render_texture attached to the FBO. Writing to dst
                // directly would be wrong — the final readback later
                // overwrites dst with the framebuffer contents, so we
                // have to seed the framebuffer itself.
                self.upload_pixels_to_render_texture(bg, bg_fmt, dst_w, dst_h)?;
            }
        } else {
            // No background: actively clear the framebuffer so stale
            // pixels from the previous frame do not leak into the output.
            unsafe {
                gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            }
        }

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.set_opacity_uniform(opacity)?;
        self.render_box(dst_w, dst_h, detect, color_mode)?;
        self.render_segmentation(detect, segmentation, color_mode)?;

        gls::finish();
        if !is_dma {
            let format = match dst_fmt {
                PixelFormat::Rgb => gls::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                unsafe {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst_w as i32,
                        dst_h as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.len() as i32,
                        std::ptr::null_mut(),
                    );
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    gls::gl::Finish();
                }
                check_gl_error(function!(), line!())?;
                if dst_fmt == PixelFormat::Bgra {
                    let mut dst_map = dst.map()?;
                    for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }
                }
            } else {
                let mut dst_map = dst.map()?;
                unsafe {
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst_w as i32,
                        dst_h as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.len() as i32,
                        dst_map.as_mut_ptr() as *mut c_void,
                    );
                }
                check_gl_error(function!(), line!())?;
                if dst_fmt == PixelFormat::Bgra {
                    for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_proto_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        opacity: f32,
        background: Option<(&Tensor<u8>, PixelFormat)>,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_proto_masks");
        if !matches!(
            dst_fmt,
            PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb
        ) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let memory = dst.memory();
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "draw_proto_masks: dst.memory()={memory:?} {dst_w}x{dst_h} fmt={dst_fmt:?}"
            );
        }
        let pbo_buffer_id = if memory == TensorMemory::Pbo {
            match dst.as_pbo() {
                Some(p) if !p.is_mapped() => Some(p.buffer_id()),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            TensorMemory::Dma => match self.setup_draw_renderbuffer_dma(dst, dst_fmt) {
                Ok(()) => {
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "draw_proto_masks: DMA fast path (setup_draw_renderbuffer_dma OK)"
                        );
                    }
                    true
                }
                Err(e) => {
                    warn_slow_path_once("draw_proto_masks", "setup_draw_renderbuffer_dma", &e);
                    if let Some(buffer_id) = pbo_buffer_id {
                        self.setup_renderbuffer_from_pbo(dst, dst_fmt, buffer_id)?;
                    } else {
                        self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                    }
                    false
                }
            },
            _ if pbo_buffer_id.is_some() => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_proto_masks: PBO path");
                }
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_proto_masks: non-DMA fallback (memory={memory:?})");
                }
                self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                false
            }
        };

        // Write the base layer of the framebuffer before running mask
        // passes. See `draw_decoded_masks_impl` for the rationale — we
        // never assume dst is cleared on entry.
        if let Some((bg, bg_fmt)) = background {
            if bg.width() != Some(dst_w) || bg.height() != Some(dst_h) {
                return Err(crate::Error::InvalidShape(
                    "background dimensions do not match dst".into(),
                ));
            }
            if is_dma && bg.memory() == TensorMemory::Dma {
                gls::disable(gls::gl::BLEND);
                let bg_egl = self.get_or_create_egl_image(CacheKind::Src, bg, bg_fmt)?;
                self.draw_camera_texture_eglimage(
                    bg,
                    bg_fmt,
                    bg_egl,
                    RegionOfInterest {
                        left: 0.0,
                        top: 1.0,
                        right: 1.0,
                        bottom: 0.0,
                    },
                    RegionOfInterest {
                        left: -1.0,
                        top: 1.0,
                        right: 1.0,
                        bottom: -1.0,
                    },
                    0,
                    crate::Flip::None,
                    false,
                )?;
            } else {
                self.upload_pixels_to_render_texture(bg, bg_fmt, dst_w, dst_h)?;
            }
        } else {
            unsafe {
                gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            }
        }

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.set_opacity_uniform(opacity)?;
        self.render_box(dst_w, dst_h, detect, color_mode)?;
        self.render_proto_segmentation(detect, proto_data, color_mode)?;

        gls::finish();
        if !is_dma {
            let format = match dst_fmt {
                PixelFormat::Rgb => gls::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                unsafe {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst_w as i32,
                        dst_h as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.len() as i32,
                        std::ptr::null_mut(),
                    );
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    gls::gl::Finish();
                }
                check_gl_error(function!(), line!())?;
                if dst_fmt == PixelFormat::Bgra {
                    let mut dst_map = dst.map()?;
                    for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }
                }
            } else {
                let mut dst_map = dst.map()?;
                unsafe {
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst_w as i32,
                        dst_h as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.len() as i32,
                        dst_map.as_mut_ptr() as *mut c_void,
                    );
                }
                check_gl_error(function!(), line!())?;
                if dst_fmt == PixelFormat::Bgra {
                    for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn check_src_format_supported(
        backend: TransferBackend,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> bool {
        if backend.is_dma() && img.memory() == TensorMemory::Dma {
            // EGLImage DMA-BUF path supports:
            //   Path A (samplerExternalOES): RGBA, GREY, YUYV, NV12
            //   Path B (R8 texelFetch shader): NV16, NV24 (contiguous only)
            // VYUY excluded: Vivante GPU accepts the DRM fourcc but produces
            // incorrect output (similarity ~0.28 vs reference).
            // NV16/NV24 use Path B — the contiguous-only check is enforced at
            // EGLImage creation time in `from_tensor_nv_r8`; multiplane
            // sources fall back to CPU via the error path.
            matches!(
                fmt,
                PixelFormat::Rgba
                    | PixelFormat::Grey
                    | PixelFormat::Yuyv
                    | PixelFormat::Nv12
                    | PixelFormat::Nv16
                    | PixelFormat::Nv24
            )
        } else {
            // Non-DMA (PBO/Sync): packed RGB(A)/Grey upload via draw_src_texture,
            // plus single-plane NV12/NV16/NV24 via the R8-upload ShaderR8 path
            // (combined buffer uploaded as R8 + in-shader YUV — no DMA-BUF
            // EGLImage needed; the GPU NV path on e.g. orin). Multiplane NV12
            // can't be uploaded as one R8 texture and falls to CPU.
            matches!(
                fmt,
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey
            ) || (matches!(
                fmt,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) && !img.is_multiplane())
        }
    }

    pub(super) fn check_dst_format_supported(
        backend: TransferBackend,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        _is_int8: bool,
        has_bgra: bool,
    ) -> bool {
        if fmt == PixelFormat::Bgra && !has_bgra {
            return false;
        }
        if backend.is_dma() && img.memory() == TensorMemory::Dma {
            matches!(
                fmt,
                PixelFormat::Rgba
                    | PixelFormat::Bgra
                    | PixelFormat::Grey
                    | PixelFormat::PlanarRgb
                    | PixelFormat::Rgb
            )
        } else {
            matches!(
                fmt,
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Grey
            )
        }
    }

    /// Query GL capabilities and detect GPU vendor/renderer type.
    ///
    /// Returns `(has_float_linear, has_bgra, is_vivante, is_software_renderer,
    /// supports_f32_color, supports_f16_color)`. The two float-color flags
    /// report whether the GPU can render to F32 / F16 color attachments
    /// (independent extensions: a configuration may have one but not the
    /// other). On this Linux backend they are forward-compat capability
    /// probes surfaced via `RenderDtypeSupport`; the macOS IOSurface path
    /// is the only render destination that consumes float dtypes today.
    fn gl_check_support() -> Result<(bool, bool, bool, bool, bool, bool), crate::Error> {
        if let Ok(version) = gls::get_string(gls::gl::SHADING_LANGUAGE_VERSION) {
            log::debug!("GL Shading Language Version: {version:?}");
        } else {
            log::warn!("Could not get GL Shading Language Version");
        }

        // Detect GPU vendor and software renderers via GL_RENDERER string.
        let (is_vivante, is_software_renderer) = gls::get_string(gls::gl::RENDERER)
            .map(|r| {
                log::info!("GL_RENDERER: {r}");
                let lower = r.to_ascii_lowercase();
                let vivante = lower.contains("vivante")
                    || lower.contains("gc7000")
                    || lower.contains("galcore");
                let software = lower.contains("llvmpipe")
                    || lower.contains("softpipe")
                    || lower.contains("swrast")
                    || lower.contains("software rasterizer");
                (vivante, software)
            })
            .unwrap_or((false, false));
        if is_vivante {
            log::warn!(
                "Vivante GPU detected — NV12 → planar RGB conversions will use \
                 two-pass workaround to avoid GPU hang (EDGEAI-1180)"
            );
        }
        if is_software_renderer {
            log::warn!(
                "Software OpenGL renderer detected — GPU backend will be disabled. \
                 Image processing will use the CPU backend instead. \
                 Check EGL ICD configuration if a hardware GPU is expected."
            );
        }

        let extensions = unsafe {
            let str = gls::gl::GetString(gls::gl::EXTENSIONS);
            if str.is_null() {
                return Err(crate::Error::GLVersion(
                    "GL returned no supported extensions".to_string(),
                ));
            }
            CStr::from_ptr(str as *const c_char)
                .to_string_lossy()
                .to_string()
        };
        log::debug!("GL Extensions: {extensions}");
        let required_ext = ["GL_OES_EGL_image_external_essl3"];
        let extensions = extensions.split_ascii_whitespace().collect::<BTreeSet<_>>();
        for required in required_ext {
            if !extensions.contains(required) {
                return Err(crate::Error::GLVersion(format!(
                    "GL does not support {required} extension",
                )));
            }
        }

        // EDGEFIRST_GL_NO_FLOAT_LINEAR=1 forces the capability off so the
        // f32→f16-repack proto fallback (taken only on GPUs lacking the
        // extension) is reachable on float-linear-capable lanes — no CI lane
        // has such a GPU, so without the override the fallback arm has zero
        // coverage anywhere.
        let force_no_float_linear = std::env::var("EDGEFIRST_GL_NO_FLOAT_LINEAR")
            .map(|v| v == "1")
            .unwrap_or(false);
        let has_float_linear =
            extensions.contains("GL_OES_texture_float_linear") && !force_no_float_linear;
        if force_no_float_linear {
            log::info!("GL_OES_texture_float_linear forced OFF via EDGEFIRST_GL_NO_FLOAT_LINEAR");
        }
        log::debug!("GL_OES_texture_float_linear: {has_float_linear}");

        let has_bgra = extensions.contains("GL_EXT_texture_format_BGRA8888");
        log::debug!("GL_EXT_texture_format_BGRA8888: {has_bgra}");

        // Float-color-buffer extensions gate F32/F16 destinations on the
        // IOSurface render path. The two extensions are independent: a
        // configuration may expose one but not the other, so consumers
        // (e.g. profiler/ORT CoreML path) probe each separately and pick
        // the preferred dtype.
        let supports_f32_color = extensions.contains("GL_EXT_color_buffer_float");
        let supports_f16_color = extensions.contains("GL_EXT_color_buffer_half_float");
        log::debug!("GL_EXT_color_buffer_float: {supports_f32_color}");
        log::debug!("GL_EXT_color_buffer_half_float: {supports_f16_color}");

        Ok((
            has_float_linear,
            has_bgra,
            is_vivante,
            is_software_renderer,
            supports_f32_color,
            supports_f16_color,
        ))
    }

    /// Invalidate EGL binding state on all destination textures.
    /// Called when the dst EGLImage cache evicts or sweeps entries.
    fn invalidate_dst_textures(&mut self) {
        self.render_texture.invalidate_egl_binding();
        self.draw_render_texture.invalidate_egl_binding();
    }

    /// Invalidate EGL binding state on all source textures.
    /// Called when the src EGLImage cache evicts or sweeps entries.
    fn invalidate_src_textures(&mut self) {
        self.camera_eglimage_texture.invalidate_egl_binding();
        self.nv_r8_texture.invalidate_egl_binding();
    }

    /// Classify and bind the destination render target — the single entry
    /// point absorbing the per-memory `setup_renderbuffer_*` fan-out. The
    /// pure classification lives in [`super::render::lower_dst`]; this
    /// performs the GL work and tells the caller what completes the convert.
    fn bind_dst(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        crop: ResolvedCrop,
    ) -> crate::Result<DstTarget> {
        match super::render::lower_dst(self.gl_context.transfer_backend.is_dma(), dst.memory()) {
            super::render::DstLowering::ZeroCopy => {
                self.setup_renderbuffer_dma(dst, dst_fmt)?;
                Ok(DstTarget::ZeroCopyImage)
            }
            super::render::DstLowering::TexturePbo => {
                let dst_pbo = dst.as_pbo().ok_or_else(|| {
                    crate::Error::OpenGl(
                        "bind_dst: PBO-lowered destination is not a PBO tensor".to_string(),
                    )
                })?;
                if dst_pbo.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert to a mapped PBO tensor".to_string(),
                    ));
                }
                let id = dst_pbo.buffer_id();
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, id)?;
                Ok(DstTarget::Texture {
                    readback: DstReadback::Pbo(id),
                })
            }
            super::render::DstLowering::TextureMem => {
                self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;
                Ok(DstTarget::Texture {
                    readback: DstReadback::Mem,
                })
            }
        }
    }

    fn setup_renderbuffer_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
    ) -> crate::Result<()> {
        self.convert_fbo.bind();
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let (width, height) = if dst_fmt == PixelFormat::PlanarRgb {
            (dst_w as i32, (dst_h * 3) as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let dst_key = EglCacheKey::from_tensor(dst, dst_fmt, true);
        let luma_id = dst_key.luma_id;

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst, dst_fmt) {
            Some(rbo) => unsafe {
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
                gls::gl::FramebufferRenderbuffer(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    gls::gl::RENDERBUFFER,
                    rbo,
                );
                check_gl_error(function!(), line!())?;
            },
            None => unsafe {
                gls::gl::UseProgram(self.texture_program_yuv.id);
                gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
                super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::LINEAR);
                if self
                    .render_texture
                    .bind_egl_image(dst_key, dest_egl.as_ptr())
                {
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        self.render_texture.id,
                        0,
                    );
                    log::trace!("setup_renderbuffer_dma: bound new dst EGLImage id={luma_id:#x}");
                } else {
                    log::trace!(
                        "setup_renderbuffer_dma: reusing bound dst EGLImage id={luma_id:#x}"
                    );
                }
                check_gl_error(function!(), line!())?;
            },
        }

        // A view()/batch() destination imported the whole PARENT above (cache key
        // + `DmaImportAttrs` both pivot on `view_origin`); render this tile into
        // its band via `glViewport`. A whole tensor fills its full surface. The
        // matching `glScissor` (so the letterbox clear cannot wipe sibling tiles)
        // is set in `convert_to`, which owns the clear — both lower the band
        // through `region_to_viewport_top_down`, the single home for the
        // verified top-down orientation of the dst DMA EGLImage.
        let vp = match dst.view_origin() {
            Some(vo) => super::render::region_to_viewport_top_down(crate::Region::new(
                vo.x, vo.y, dst_w, dst_h,
            )),
            None => super::render::Viewport {
                x: 0,
                y: 0,
                w: width,
                h: height,
            },
        };
        unsafe {
            gls::gl::Viewport(vp.x, vp.y, vp.w, vp.h);
        }
        Ok(())
    }

    /// Variant of [`setup_renderbuffer_dma`] used exclusively by draw operations
    /// (`draw_decoded_masks`, `draw_proto_masks`).
    ///
    /// Uses a dedicated FBO (`draw_fbo`) and texture (`draw_render_texture`) so
    /// that `EGLImageTargetTexture2DOES` calls made during convert do not
    /// invalidate the draw path's cached binding, and vice versa.  On Vivante
    /// GC7000UL (texture path, `use_renderbuffer = false`) each
    /// `EGLImageTargetTexture2DOES` call costs ~4–11 ms, so eliminating the
    /// cross-path invalidation removes two redundant calls per frame when both
    /// convert and draw are active.
    fn setup_draw_renderbuffer_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
    ) -> crate::Result<()> {
        self.draw_fbo.bind();
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let (width, height) = if dst_fmt == PixelFormat::PlanarRgb {
            (dst_w as i32, (dst_h * 3) as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let dst_key = EglCacheKey::from_tensor(dst, dst_fmt, true);
        let luma_id = dst_key.luma_id;

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst, dst_fmt) {
            Some(rbo) => unsafe {
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
                gls::gl::FramebufferRenderbuffer(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    gls::gl::RENDERBUFFER,
                    rbo,
                );
                check_gl_error(function!(), line!())?;
            },
            None => unsafe {
                gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.draw_render_texture.id);
                super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::LINEAR);
                if self
                    .draw_render_texture
                    .bind_egl_image(dst_key, dest_egl.as_ptr())
                {
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        self.draw_render_texture.id,
                        0,
                    );
                    log::trace!(
                        "setup_draw_renderbuffer_dma: bound new dst EGLImage id={luma_id:#x}"
                    );
                } else {
                    log::trace!(
                        "setup_draw_renderbuffer_dma: reusing bound dst EGLImage id={luma_id:#x}"
                    );
                }
                check_gl_error(function!(), line!())?;
            },
        }

        unsafe {
            gls::gl::Viewport(0, 0, width, height);
        }
        Ok(())
    }

    /// Bias a letterbox clear colour by XOR 0x80 for int8 destinations, since
    /// `glClear` bypasses the shader that otherwise applies the bias. No-op when
    /// not int8 or when there is no fill colour. Shared by every destination path.
    fn int8_bias_clear(is_int8: bool, mut crop: ResolvedCrop) -> ResolvedCrop {
        if is_int8 {
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
        }
        crop
    }

    /// Render `src` into the already-bound destination FBO as `dst_fmt`,
    /// dispatching packed vs planar; the draws select their int8 program
    /// variants from `is_int8` at draw time. This is the converged per-tile
    /// draw shared by the texture and DMA destination paths — they differ
    /// only in FBO setup and readback, never in this draw.
    #[allow(clippy::too_many_arguments)]
    fn render_packed_or_planar(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        if dst_fmt.layout() == PixelLayout::Planar {
            self.convert_to_planar(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
        } else {
            self.convert_to(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
        }
    }

    /// Read the rendered FBO colour attachment (`COLOR_ATTACHMENT0`) into the
    /// destination, applying the BGRA byte-swap when `dst_fmt` is BGRA. `pbo_id`
    /// selects the target: `None` reads straight into the mapped Mem tensor (the
    /// preceding draw has already finished the GPU); `Some(id)` reads into the
    /// bound PBO PACK buffer and issues a finishing flush so the async transfer
    /// completes. The converged readback shared by the non-DMA and any-to-PBO
    /// paths — they differ only in this target selector.
    fn readback_rendered(
        &self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        pbo_id: Option<u32>,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "GL readback not supported for {dst_fmt}"
                )))
            }
        };
        let len = dst.len();
        match pbo_id {
            None => unsafe {
                let mut dst_map = dst.map()?;
                gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                gls::gl::ReadnPixels(
                    0,
                    0,
                    dst_w as i32,
                    dst_h as i32,
                    dest_format,
                    gls::gl::UNSIGNED_BYTE,
                    len as i32,
                    dst_map.as_mut_ptr() as *mut c_void,
                );
                if dst_fmt == PixelFormat::Bgra {
                    for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                        chunk.swap(0, 2);
                    }
                }
            },
            Some(buffer_id) => {
                unsafe {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst_w as i32,
                        dst_h as i32,
                        dest_format,
                        gls::gl::UNSIGNED_BYTE,
                        len as i32,
                        std::ptr::null_mut(),
                    );
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    gls::gl::Finish();
                }
                // BGRA R↔B swap must map the PBO on the GL thread. Int8 XOR 0x80
                // is handled in the fragment shader — no CPU map needed.
                if dst_fmt == PixelFormat::Bgra {
                    unsafe {
                        gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                        let ptr = gls::gl::MapBufferRange(
                            gls::gl::PIXEL_PACK_BUFFER,
                            0,
                            len as isize,
                            gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                        );
                        if ptr.is_null() {
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            return Err(crate::Error::OpenGl(
                                "glMapBufferRange returned null for BGRA byte-swap".to_string(),
                            ));
                        }
                        let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, len);
                        for chunk in slice.chunks_exact_mut(4) {
                            chunk.swap(0, 2);
                        }
                        gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                        gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    }
                }
            }
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn setup_renderbuffer_non_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let is_planar = dst_fmt.layout() == PixelLayout::Planar;

        let (width, height) = if is_planar {
            let width = dst_w / 4;
            let height = match dst_fmt.channels() {
                4 => dst_h * 4,
                3 => dst_h * 3,
                1 => dst_h,
                _ => unreachable!(),
            };
            (width as i32, height as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        // BGRA textures as framebuffer color attachments have GPU-dependent
        // swizzle behavior: some implementations don't swizzle fragment shader
        // output, causing R↔B channel swap. Work around this by using RGBA
        // format internally — BGRA pixel data is byte-swapped (R↔B) before
        // upload, and the readback path swaps back.
        let is_bgra = !is_planar && dst_fmt == PixelFormat::Bgra;
        if is_bgra && !self.bgra_warned {
            log::warn!(
                "BGRA destination: using RGBA internal format with CPU R↔B byte-swap workaround"
            );
            self.bgra_warned = true;
        }
        let format = if is_planar {
            gls::gl::RED
        } else {
            match dst_fmt {
                PixelFormat::Rgb => gls::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA, // BGRA uses RGBA internally
                PixelFormat::Grey => gls::gl::RED,
                _ => unreachable!(),
            }
        };

        let start = Instant::now();
        self.convert_fbo.bind();

        let map;
        let mut swapped_buf;

        let pixels = if crop.dst_rect.is_none_or(|crop| {
            crop.top == 0 && crop.left == 0 && crop.height == dst_h && crop.width == dst_w
        }) {
            std::ptr::null()
        } else {
            map = dst.map()?;
            if is_bgra {
                // Swap R↔B to convert BGRA→RGBA for the RGBA texture.
                swapped_buf = map.as_slice().to_vec();
                for chunk in swapped_buf.chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
                swapped_buf.as_ptr() as *const c_void
            } else {
                map.as_ptr() as *const c_void
            }
        };
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::LINEAR);

            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                format as i32,
                width,
                height,
                0,
                format,
                gls::gl::UNSIGNED_BYTE,
                pixels,
            );
            // TexImage2D overwrites any EGLImage binding on this texture.
            self.render_texture.invalidate_egl_binding();
            check_gl_error(function!(), line!())?;
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, width, height);
        }
        log::debug!("Set up framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    /// Set up a framebuffer for overlay rendering on a PBO-backed destination.
    ///
    /// Binds the PBO as `GL_PIXEL_UNPACK_BUFFER` and uploads its contents to
    /// the render texture via `TexImage2D` with a NULL pointer — GL reads
    /// directly from PBO memory without any CPU-side `map()` call. This avoids
    /// the deadlock that occurs when `setup_renderbuffer_non_dma` tries to
    /// `tensor.map()` a PBO on the GL thread.
    /// Upload CPU-side pixel data into `self.render_texture` which is
    /// attached as the FBO color attachment.  Used for the non-DMA
    /// background path in `draw_decoded_masks_impl` /
    /// `draw_proto_masks_impl`: the source buffer is a user-provided
    /// tensor in plain memory, so we do a single CPU→GPU upload into the
    /// texture rather than memcpying into the caller's dst (which the
    /// final framebuffer readback would overwrite anyway).
    ///
    /// If the source is BGRA we byte-swap into a scratch buffer so the
    /// RGBA-internal render target holds the correct pixels after the
    /// later readback-time R↔B swap.
    fn upload_pixels_to_render_texture(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst_w: usize,
        dst_h: usize,
    ) -> crate::Result<()> {
        use edgefirst_tensor::TensorMapTrait;
        let map = src.map()?;
        let src_slice = map.as_slice();
        let format = match src_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "non-DMA bg upload not supported for {src_fmt}",
                )))
            }
        };
        let swapped: Option<Vec<u8>> = if src_fmt == PixelFormat::Bgra {
            let mut v = src_slice.to_vec();
            for chunk in v.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
            Some(v)
        } else {
            None
        };
        let pixels = swapped.as_deref().unwrap_or(src_slice).as_ptr() as *const c_void;

        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                format as i32,
                dst_w as i32,
                dst_h as i32,
                0,
                format,
                gls::gl::UNSIGNED_BYTE,
                pixels,
            );
            // TexImage2D invalidates any prior EGLImage binding.
            self.render_texture.invalidate_egl_binding();
            check_gl_error(function!(), line!())?;
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Low-level PBO render-target setup with explicit GL format and type.
    ///
    /// Binds `buffer_id` as `GL_PIXEL_UNPACK_BUFFER`, calls `TexImage2D` with
    /// the provided `internal_format`, `client_format`, and `gl_type`, then
    /// attaches the texture to the FBO color attachment and sets the viewport.
    ///
    /// The `(width, height)` are the logical pixel dimensions of the render
    /// target (caller is responsible for planar-packing remapping if needed).
    ///
    /// # Safety (caller responsibility)
    /// `internal_format`, `client_format`, and `gl_type` must form a valid
    /// `TexImage2D` combination for the bound GLES context.
    fn setup_renderbuffer_from_pbo_inner(
        &mut self,
        width: i32,
        height: i32,
        buffer_id: u32,
        internal_format: u32,
        client_format: u32,
        gl_type: u32,
    ) -> crate::Result<()> {
        self.convert_fbo.bind();
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::LINEAR);

            // Upload existing PBO content to the render texture.
            // Binding PBO as UNPACK buffer makes TexImage2D read from it.
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, buffer_id);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                internal_format as i32,
                width,
                height,
                0,
                client_format,
                gl_type,
                std::ptr::null(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, 0);
            // TexImage2D overwrites any EGLImage binding on this texture.
            self.render_texture.invalidate_egl_binding();

            check_gl_error(function!(), line!())?;
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, width, height);
        }
        Ok(())
    }

    /// Set up a u8 (UNSIGNED_BYTE) PBO render target for the given format.
    ///
    /// Computes the correct GL `format` from `dst_fmt` (handling planar
    /// packing layout) and delegates to [`setup_renderbuffer_from_pbo_inner`]
    /// with `UNSIGNED_BYTE` as the GL type.  The next task adds an analogous
    /// float variant that calls the inner helper with `R32F`/`RGBA16F` and
    /// `FLOAT`/`HALF_FLOAT` respectively.
    fn setup_renderbuffer_from_pbo(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        buffer_id: u32,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let is_planar = dst_fmt.layout() == PixelLayout::Planar;

        let (width, height) = if is_planar {
            let width = dst_w / 4;
            let height = match dst_fmt.channels() {
                4 => dst_h * 4,
                3 => dst_h * 3,
                1 => dst_h,
                _ => unreachable!(),
            };
            (width as i32, height as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let format = if is_planar {
            gls::gl::RED
        } else {
            match dst_fmt {
                PixelFormat::Rgb => gls::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
                PixelFormat::Grey => gls::gl::RED,
                _ => {
                    return Err(crate::Error::NotSupported(format!(
                        "PBO renderbuffer not supported for {dst_fmt}",
                    )))
                }
            }
        };

        self.setup_renderbuffer_from_pbo_inner(
            width,
            height,
            buffer_id,
            format,
            format,
            gls::gl::UNSIGNED_BYTE,
        )
    }

    /// Upload source image from a PBO and render to the current framebuffer.
    /// This is the PBO equivalent of draw_src_texture — instead of mapping
    /// the tensor to CPU and calling glTexImage2D with a data pointer, we
    /// bind the source PBO as GL_PIXEL_UNPACK_BUFFER and pass NULL, causing
    /// GL to read directly from the PBO (zero CPU copy).
    #[allow(clippy::too_many_arguments)]
    fn draw_src_texture_from_pbo(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        src_buffer_id: u32,
        dst: &Tensor<u8>,
        _dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> Result<(), Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "PBO upload not supported for {src_fmt:?}",
                )));
            }
        };

        let has_crop = crop
            .dst_rect
            .is_some_and(|x| x.left != 0 || x.top != 0 || x.width != dst_w || x.height != dst_h);

        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest::from_crop_clamped(&crop, src_w, src_h)
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let mut dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst_w as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };

        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };

        unsafe {
            if has_crop {
                if let Some(dst_color) = crop.dst_color {
                    gls::gl::ClearColor(
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    );
                    gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                }
            }

            // Draw-time program selection (see draw_src_texture).
            gls::gl::UseProgram(if is_int8 {
                self.texture_int8_program.id
            } else {
                self.texture_program.id
            });
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            super::core::set_tex_filter_clamp(texture_target, gls::gl::LINEAR);
            if src_fmt == PixelFormat::Grey {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src_component) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src_component as i32);
                }
            }

            // Honour a padded source row stride: a PBO written by the CPU JPEG
            // decoder (or any producer) may have 64-byte-aligned rows, so the
            // bytes between logical rows must be skipped on upload. Mirror the
            // non-PBO `draw_src_texture` path (GL_UNPACK_ROW_LENGTH in pixels);
            // 0 means "tightly packed = src_w". Without this a padded PBO source
            // shears on every row after the first.
            let src_bpp = src_fmt.channels();
            let row_len_px = src
                .effective_row_stride()
                .map(|s| s / src_bpp)
                .filter(|&px| px != src_w)
                .unwrap_or(0);

            // Bind source PBO as UNPACK buffer — glTexImage2D reads from it
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, src_buffer_id);
            gls::gl::PixelStorei(gls::gl::UNPACK_ROW_LENGTH, row_len_px as i32);
            gls::gl::TexImage2D(
                texture_target,
                0,
                texture_format as i32,
                src_w as i32,
                src_h as i32,
                0,
                texture_format,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(), // NULL = read from bound UNPACK buffer
            );
            gls::gl::PixelStorei(gls::gl::UNPACK_ROW_LENGTH, 0);
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, 0);

            // Force texture cache state to be rebuilt next call
            self.camera_normal_texture.width = 0;

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                crate::Flip::None => {}
                crate::Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                crate::Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (camera_vertices.len() * std::mem::size_of::<f32>()) as isize,
                camera_vertices.as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::VertexAttribPointer(
                self.vertex_buffer.buffer_index,
                3,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                0,
                std::ptr::null(),
            );

            let texture_coords: [[f32; 8]; 4] = [
                [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ],
                [
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                ],
                [
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                ],
                [
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                ],
            ];
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (texture_coords[0].len() * std::mem::size_of::<f32>()) as isize,
                texture_coords[rotation_offset].as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::VertexAttribPointer(
                self.texture_buffer.buffer_index,
                2,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                0,
                std::ptr::null(),
            );
            gls::gl::DrawArrays(gls::gl::TRIANGLE_FAN, 0, 4);
            gls::gl::DisableVertexAttribArray(self.vertex_buffer.buffer_index);
            gls::gl::DisableVertexAttribArray(self.texture_buffer.buffer_index);

            gls::gl::Finish();
        }

        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Pick the NV* GPU conversion path for `src`/`src_fmt`, honoring the
    /// `EDGEFIRST_NV_CONVERT_PATH` preference. Returns the path to *attempt*;
    /// the caller maps an EGLImage-creation error to the [`NvConvertPath::Cpu`]
    /// fallback. Forcing an unavailable path logs a warning and falls back to
    /// the only viable one rather than failing.
    ///
    /// Capability:
    /// - [`NvConvertPath::ShaderR8`] needs a single combined-plane buffer, so it
    ///   covers single-plane NV12/NV16/NV24 but NOT true-multiplane NV12.
    /// - [`NvConvertPath::ExternalSampler`] (samplerExternalOES) is wired for
    ///   NV12 only (incl. multiplane); NV16/NV24 have no sampler path.
    ///
    /// `render_fmt` is the format of the render target this draw writes
    /// (Rgb/PlanarRgb arrive here only as the LOGICAL format of a two-pass
    /// convert whose pass 1 renders an RGBA intermediate; the only R8 render
    /// target reachable with an NV source is a single-pass Grey destination).
    fn select_nv_path(
        &self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        render_fmt: PixelFormat,
    ) -> NvConvertPath {
        // `ShaderR8` (R8 texelFetch) only applies to SINGLE-PLANE semi-planar
        // NV12/NV16/NV24. Everything else routed through this picker —
        // RGBA/Grey/YUYV/RGB and true-multiplane NV12 — uses the
        // `ExternalSampler` (samplerExternalOES / `draw_camera_texture_eglimage`)
        // path, so that is the default.
        let shader_capable = matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) && !src.is_multiplane();
        if !shader_capable {
            return NvConvertPath::ExternalSampler;
        }
        // From here: single-plane NV12/NV16/NV24 — ShaderR8 is viable. Only NV12
        // also has an ExternalSampler path (NV16/NV24 do not).
        match self.nv_path_pref {
            NvPathPref::ForceShader => NvConvertPath::ShaderR8,
            NvPathPref::ForceSampler => {
                if src_fmt == PixelFormat::Nv12
                    && self.is_vivante
                    && render_fmt == PixelFormat::Grey
                {
                    // See the Auto-arm comment: sampler → R8 target wedges the
                    // GC7000UL. Refuse the force rather than hang the GPU.
                    log::warn!(
                        "EDGEFIRST_NV_CONVERT_PATH=sampler refused for NV12→Grey on Vivante \
                         (samplerExternalOES → R8 render target hangs the GPU); using shader"
                    );
                    NvConvertPath::ShaderR8
                } else if src_fmt == PixelFormat::Nv12 {
                    NvConvertPath::ExternalSampler
                } else {
                    log::warn!(
                        "EDGEFIRST_NV_CONVERT_PATH=sampler unavailable for {src_fmt:?} \
                         (no external-sampler path); using shader"
                    );
                    NvConvertPath::ShaderR8
                }
            }
            // Auto: prefer the portable, colorimetry-exact in-shader ShaderR8
            // wherever it is also the fast path (Mali, V3D, Tegra — shader and
            // sampler are comparable there, so exactness is free).
            //
            // EXCEPTION — Vivante (i.MX 8MP): the texelFetch shader is ~12×
            // slower than the hardware samplerExternalOES (on-target benchmark:
            // NV12 720p convert 29ms shader vs 2.5ms sampler). For single-plane
            // 4-aligned NV12 (the sampler import constraint) the pick follows
            // the HIGH-PERFORMANCE-default policy (issue #106):
            //
            // * `ColorimetryMode::Fast` (default): take the sampler for EVERY
            //   colorimetry — the driver applies its fixed BT.601-limited
            //   matrix, which is exact for BT.601-limited sources (Phase 2
            //   probe: ≤1 vs CPU) and approximate for the rest.
            // * `ColorimetryMode::Exact` (config/env opt-in): take the sampler
            //   only when the driver matrix MATCHES the source's resolved
            //   (encoding, range); everything else renders through the exact
            //   in-shader matrix, at Vivante's 12× cost.
            NvPathPref::Auto => {
                // NEVER pair the external sampler with an R8 (Grey) render
                // target on Vivante: samplerExternalOES → R8 attachment wedges
                // the GC7000UL (the EDGEAI-1180 hang class — found when the
                // Fast policy first routed nv12@dma→grey@dma to the sampler;
                // the GPU hangs unrecoverably mid-draw). This gate applies in
                // BOTH colorimetry modes: the old BT.601-limited carve-out had
                // the same latent hang, just unreachable with HD sources.
                let vivante_sampler_usable = src_fmt == PixelFormat::Nv12
                    && self.is_vivante
                    && render_fmt != PixelFormat::Grey
                    && src.width().is_some_and(|w| w.is_multiple_of(4));
                let take_sampler = vivante_sampler_usable
                    && match self.colorimetry_mode {
                        crate::ColorimetryMode::Fast => true,
                        crate::ColorimetryMode::Exact => {
                            let cm = crate::colorimetry::resolve_colorimetry(
                                src.colorimetry(),
                                src.height(),
                            );
                            cm.encoding == Some(edgefirst_tensor::ColorEncoding::Bt601)
                                && cm.range == Some(edgefirst_tensor::ColorRange::Limited)
                        }
                    };
                if take_sampler {
                    NvConvertPath::ExternalSampler
                } else {
                    NvConvertPath::ShaderR8
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_to(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &Tensor<u8>,
        _dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> Result<(), crate::Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        check_gl_error(function!(), line!())?;

        // A view()/batch() destination shares ONE parent EGLImage with its
        // sibling tiles; the band viewport was set in `setup_renderbuffer_dma`.
        // Confine this tile's letterbox `glClear` (below) and draw to the same
        // band via `glScissor` so they cannot wipe a sibling. The RAII guard
        // disables `SCISSOR_TEST` on every exit (success, `?`, or unwind) so the
        // next convert/draw starts from a clean state.
        struct ScissorGuard(bool);
        impl Drop for ScissorGuard {
            fn drop(&mut self) {
                if self.0 {
                    unsafe { gls::gl::Disable(gls::gl::SCISSOR_TEST) };
                }
            }
        }
        // Band rect via `region_to_viewport_top_down` — the SAME lowering as
        // the band viewport in `bind_dst`'s setup, so scissor and viewport can
        // never disagree on the orientation convention.
        let _scissor = match dst.view_origin() {
            Some(vo) => {
                let vp = super::render::region_to_viewport_top_down(crate::Region::new(
                    vo.x, vo.y, dst_w, dst_h,
                ));
                unsafe {
                    gls::gl::Scissor(vp.x, vp.y, vp.w, vp.h);
                    gls::gl::Enable(gls::gl::SCISSOR_TEST);
                }
                ScissorGuard(true)
            }
            None => ScissorGuard(false),
        };

        let has_crop = crop
            .dst_rect
            .is_some_and(|x| x.left != 0 || x.top != 0 || x.width != dst_w || x.height != dst_h);
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                unsafe {
                    gls::gl::ClearColor(
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    );
                    gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                };
            }
        }

        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest::from_crop_clamped(&crop, src_w, src_h)
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst_w as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };
        if self.gl_context.transfer_backend.is_dma() && src.memory() == TensorMemory::Dma {
            // Choose the NV* path (ShaderR8 vs ExternalSampler) honoring the
            // EDGEFIRST_NV_CONVERT_PATH preference. See `select_nv_path`: Auto
            // prefers the portable, colorimetry-exact in-shader ShaderR8 for
            // single-plane NV12/NV16/NV24, using the driver ExternalSampler only
            // for true-multiplane NV12 (which ShaderR8 cannot import).
            let chosen = self.select_nv_path(src, src_fmt, _dst_fmt);

            if chosen == NvConvertPath::ShaderR8 {
                match self.get_or_create_nv_r8_egl_image(src, src_fmt) {
                    Ok(r8_egl) => {
                        tracing::trace!(
                            path = "ShaderR8",
                            src_fmt = ?src_fmt,
                            "image.convert.gl.nv_path"
                        );
                        self.last_nv_convert_path = NvConvertPath::ShaderR8;
                        self.draw_nv_texture_2d(
                            src,
                            src_fmt,
                            Some(r8_egl),
                            src_roi,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                        )?;
                    }
                    Err(e) => {
                        let src_w = src.width().unwrap_or(0);
                        let src_h = src.height().unwrap_or(0);
                        // Path B failed — this means no GPU NV* path is available.
                        // Record the CPU fallback so tests/profiler can detect it.
                        self.last_nv_convert_path = NvConvertPath::Cpu;
                        tracing::warn!(
                            src_fmt = ?src_fmt,
                            src_w,
                            src_h,
                            error = %e,
                            "Path B R8 EGLImage creation failed for {src_fmt} \
                             ({src_w}x{src_h}); falling back to CPU path (no GPU NV16/NV24)"
                        );
                        let start = Instant::now();
                        self.draw_src_texture(
                            src,
                            src_fmt,
                            src_roi,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                        )?;
                        log::debug!("draw_src_texture takes {:?}", start.elapsed());
                    }
                }
            } else {
                // ExternalSampler path (samplerExternalOES): NV12 (incl.
                // multiplane) and all non-shader DMA formats (RGBA/Grey/YUYV);
                // single-plane NV12/NV16/NV24 take ShaderR8 above.
                match self.get_or_create_egl_image(CacheKind::Src, src, src_fmt) {
                    Ok(src_egl) => {
                        if src_fmt == PixelFormat::Nv12 {
                            tracing::trace!(path = "ExternalSampler", src_fmt = ?src_fmt, "image.convert.gl.nv_path");
                            self.last_nv_convert_path = NvConvertPath::ExternalSampler;
                        }
                        self.draw_camera_texture_eglimage(
                            src,
                            src_fmt,
                            src_egl,
                            src_roi,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                        )?;
                    }
                    Err(e) => {
                        let src_w = src.width().unwrap_or(0);
                        let src_h = src.height().unwrap_or(0);
                        if src_fmt == PixelFormat::Nv12 {
                            self.last_nv_convert_path = NvConvertPath::Cpu;
                        }
                        log::warn!(
                            "EGL image creation failed for {src_fmt} ({src_w}x{src_h}), \
                             falling back to texture upload (slower): {e}"
                        );
                        let start = Instant::now();
                        self.draw_src_texture(
                            src,
                            src_fmt,
                            src_roi,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                        )?;
                        log::debug!("draw_src_texture takes {:?}", start.elapsed());
                    }
                }
            }
        } else if matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) && !src.is_multiplane()
        {
            // Non-DMA (PBO/Sync) single-plane NV*: GPU-convert via the SAME
            // ShaderR8 in-shader path used for DMA, but with the combined buffer
            // CPU-UPLOADED as R8 instead of EGLImage-imported. This enables the
            // NV shaders on backends without DMA-BUF EGLImage import (e.g. orin),
            // replacing the old CPU fallback. Multiplane NV12 (separate Y/UV
            // buffers) cannot be uploaded as one R8 texture → CPU below.
            tracing::trace!(path = "ShaderR8-upload", src_fmt = ?src_fmt, "image.convert.gl.nv_path");
            self.last_nv_convert_path = NvConvertPath::ShaderR8;
            self.draw_nv_texture_2d(
                src,
                src_fmt,
                None,
                src_roi,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
            )?;
        } else {
            // Non-DMA source, non-NV (or multiplane NV): CPU texture-upload path.
            if matches!(
                src_fmt,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) {
                self.last_nv_convert_path = NvConvertPath::Cpu;
            }
            let start = Instant::now();
            self.draw_src_texture(
                src,
                src_fmt,
                src_roi,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
            )?;
            log::debug!("draw_src_texture takes {:?}", start.elapsed());
        }

        // In a batch (`defer_finish`), skip the per-tile drain so the whole
        // batch syncs once via `finish_via_fence` in `convert_batch`. Outside a
        // batch this is the standalone convert's completion point and must run.
        if !self.defer_finish {
            let start = Instant::now();
            unsafe { gls::gl::Finish() };
            log::debug!("gl_Finish takes {:?}", start.elapsed());
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_to_planar(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> Result<(), crate::Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        // if let Some(crop) = crop.src_rect
        //     && (crop.left > 0
        //         || crop.top > 0
        //         || crop.height < src.height()
        //         || crop.width < src.width())
        // {
        //     return Err(crate::Error::NotSupported(
        //         "Cropping in planar RGB mode is not supported".to_string(),
        //     ));
        // }

        // if let Some(crop) = crop.dst_rect
        //     && (crop.left > 0
        //         || crop.top > 0
        //         || crop.height < src.height()
        //         || crop.width < src.width())
        // {
        //     return Err(crate::Error::NotSupported(
        //         "Cropping in planar RGB mode is not supported".to_string(),
        //     ));
        // }

        let alpha = match dst_fmt {
            PixelFormat::PlanarRgb => false,
            PixelFormat::PlanarRgba => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PlanarRgb or PlanarRgba".to_string(),
                ));
            }
        };
        log::trace!(
            "convert_to_planar: int8={is_int8}, interpolation={:?}",
            self.int8_interpolation_mode
        );

        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest::from_crop_clamped(&crop, src_w, src_h)
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst_w as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };

        let has_crop = crop
            .dst_rect
            .is_some_and(|x| x.left != 0 || x.top != 0 || x.width != dst_w || x.height != dst_h);
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                self.clear_rect_planar(
                    dst_w,
                    dst_h,
                    dst_roi,
                    [
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    ],
                    alpha,
                )?;
            }
        }

        let src_key = EglCacheKey::from_tensor(src, src_fmt, false);
        let src_egl = self.get_or_create_egl_image(CacheKind::Src, src, src_fmt)?;

        self.draw_camera_texture_to_rgb_planar(
            src_key,
            src_egl,
            src_roi,
            dst_roi,
            rotation_offset,
            flip,
            alpha,
            is_int8,
        )?;
        unsafe { gls::gl::Finish() };
        check_gl_error(function!(), line!())?;

        Ok(())
    }

    /// Render packed RGB (or int8 RGB) to a DMA destination buffer using a
    /// two-pass architecture:
    ///
    /// **Pass 1:** Render source → intermediate RGBA texture via `convert_to()`
    /// (reuses the battle-tested RGBA path with full crop/letterbox/rotation/flip).
    ///
    /// **Pass 2:** Pack intermediate RGBA → RGB DMA destination using a simple
    /// packing shader with 2D sampler. The destination DMA buffer is reinterpreted
    /// as RGBA8 at (W*3/4) x H dimensions.
    #[allow(clippy::too_many_arguments)]
    fn convert_to_packed_rgb(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        // Width must satisfy PackedRgba8 constraint: W*3 divisible by 4
        if !(dst_w * 3).is_multiple_of(4) {
            return Err(crate::Error::NotSupported(format!(
                "Packed RGB requires width*3 divisible by 4, got width={dst_w}"
            )));
        }

        let render_w = dst_w * 3 / 4;
        let render_h = dst_h;

        log::debug!(
            "convert_to_packed_rgb: {dst_w}x{dst_h} -> {render_w}x{render_h} two-pass int8={is_int8}",
        );

        // --- Pass 1: Render source → intermediate RGBA texture ---
        let _pass1 =
            tracing::trace_span!("image.convert.gl.pack_rgb.pass1_rgba", dst_w, dst_h).entered();
        self.ensure_packed_rgb_intermediate(dst_w, dst_h)?;
        self.packed_rgb_fbo.bind();
        unsafe {
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        }
        // convert_to() renders to the currently-bound FBO (packed_rgb_fbo → intermediate).
        // It uses dst only for width/height in ROI coordinate math.
        // Handles: source binding (DMA EGLImage or upload), crop, letterbox, rotation, flip.
        // Pass 1 renders UN-biased (is_int8 = false): the int8 XOR-0x80 bias is
        // applied once, by pass 2's packing shader.
        //
        // Pass 1 must NOT glFinish (convert_to's standalone-boundary sync):
        // same-context command ordering already guarantees pass 2 samples the
        // finished intermediate, and the convert syncs once at pass 2's end.
        // The flag restores before `?` so an error cannot leak defer state.
        let saved_defer = self.defer_finish;
        self.defer_finish = true;
        let pass1 = self.convert_to(src, src_fmt, dst, dst_fmt, false, rotation, flip, crop);
        self.defer_finish = saved_defer;
        pass1?;
        drop(_pass1);

        // --- Pass 2: Pack intermediate RGBA → RGB DMA destination ---
        let _pass2 =
            tracing::trace_span!("image.convert.gl.pack_rgb.pass2_pack", render_w, render_h)
                .entered();
        self.convert_fbo.bind();
        let dest_egl = self.get_or_create_egl_image_rgb(
            dst,
            dst_fmt,
            render_w,
            render_h,
            DrmFourcc::Abgr8888,
            4,
        )?;
        unsafe {
            match self.cached_dst_renderbuffer(dst, dst_fmt) {
                Some(rbo) => {
                    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
                    gls::gl::FramebufferRenderbuffer(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::RENDERBUFFER,
                        rbo,
                    );
                }
                None => {
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
                    super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::NEAREST);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_egl.as_ptr());
                    // Raw re-target bypasses bind_egl_image's key tracking:
                    // drop the cached key so a later bind_egl_image on this
                    // texture cannot "reuse" a binding this call replaced
                    // (silent write into the wrong destination buffer).
                    self.render_texture.invalidate_egl_binding();
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        self.render_texture.id,
                        0,
                    );
                }
            }
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, render_w as i32, render_h as i32);
        }

        // Bind intermediate RGBA texture as source for the packing shader
        let program = if is_int8 {
            &self.packed_rgba8_int8_program_2d
        } else {
            &self.packed_rgba8_program_2d
        };
        unsafe {
            gls::gl::UseProgram(program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE1);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.packed_rgb_intermediate_tex.id);
            super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::NEAREST);
        }

        // (`tex` = unit 1 is constant per program, uploaded at link time.)

        // Draw full-viewport quad to pack RGBA→RGB
        self.draw_fullscreen_quad()?;

        // Pass 2 bound the intermediate on TEXTURE1; restore unit 0 as the
        // active unit. Draw/setup sites select their unit before binding, but
        // the processor-wide invariant between operations is "unit 0 active" —
        // leaking unit 1 here is what broke the second heap-source convert
        // (GL_INVALID_VALUE upload into the wrong texture).
        unsafe { gls::gl::ActiveTexture(gls::gl::TEXTURE0) };

        unsafe { gls::gl::Finish() };
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Two-pass NV12→PlanarRgb workaround for Vivante GPU.
    ///
    /// Single-pass NV12→PlanarRgb causes an unrecoverable GPU hang on the
    /// Verisilicon/Vivante GC7000UL. This method splits the operation:
    ///
    /// **Pass 1:** NV12→RGBA into `packed_rgb_intermediate_tex` via `convert_to()`
    /// (full resize/crop/rotation/flip, no int8 bias).
    ///
    /// **Pass 2:** RGBA→PlanarRgb from intermediate to DMA destination via
    /// [`draw_intermediate_to_rgb_planar`] (channel deinterleave + optional int8 bias).
    #[allow(clippy::too_many_arguments)]
    fn convert_nv_to_planar_two_pass(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let alpha = match dst_fmt {
            PixelFormat::PlanarRgb => false,
            PixelFormat::PlanarRgba => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PlanarRgb or PlanarRgba".to_string(),
                ));
            }
        };

        log::debug!(
            "convert_nv_to_planar_two_pass: {src_fmt}→{dst_fmt} {dst_w}x{dst_h} \
             int8={is_int8} (Vivante two-pass workaround)",
        );

        // --- Pass 1: NV12→RGBA into intermediate texture ---
        // No int8 bias here — bias is applied in pass 2's planar shader.
        let _pass1 = tracing::trace_span!("image.convert.gl.nv_to_planar.pass1_rgba", dst_w, dst_h)
            .entered();
        self.ensure_packed_rgb_intermediate(dst_w, dst_h)?;
        self.packed_rgb_fbo.bind();
        unsafe {
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        }
        // convert_to() renders to the currently-bound FBO (packed_rgb_fbo → intermediate RGBA).
        // Note: dst_fmt is passed but ignored (_dst_fmt in convert_to's signature) — the actual
        // output format is RGBA because we bound packed_rgb_fbo above. dst is used only for
        // width/height in ROI coordinate math.
        // Pass 1 renders UN-biased (is_int8 = false): the int8 XOR-0x80 bias is
        // applied once, by pass 2's planar deinterleave shader.
        //
        // Pass 1 must NOT glFinish (convert_to's standalone-boundary sync):
        // same-context command ordering already guarantees pass 2 samples the
        // finished intermediate, and the convert syncs once at pass 2's end.
        let saved_defer = self.defer_finish;
        self.defer_finish = true;
        let pass1 = self.convert_to(src, src_fmt, dst, dst_fmt, false, rotation, flip, crop);
        self.defer_finish = saved_defer;
        pass1?;
        drop(_pass1);

        // --- Pass 2: RGBA→PlanarRgb to DMA destination ---
        // bind_dst rebinds convert_fbo with the DMA destination EGLImage,
        // replacing packed_rgb_fbo that was active during pass 1. It also sets the viewport
        // to (dst_w, dst_h * 3) for the tall R8 planar renderbuffer.
        let _pass2 = tracing::trace_span!(
            "image.convert.gl.nv_to_planar.pass2_deinterleave",
            dst_w,
            dst_h
        )
        .entered();
        self.bind_dst(dst, dst_fmt, crop)?;

        // Pass 2 is a fullscreen blit from the intermediate to the planar
        // destination. Pass 1 (convert_to above) already placed the image
        // content at the correct letterbox position within the intermediate
        // AND filled the padding region with the requested dst_color.
        // Re-applying the dst_rect crop here would map the full intermediate
        // (whose content is already correctly placed) into only the content
        // sub-region, shrinking the image by the letterbox content fraction
        // a second time. Observed downstream as bounding boxes compressed by
        // exactly (content/canvas) in the padded dimension on i.MX 8M Plus.
        //
        // No int8 bias-clear here either: pass 2 performs no clear, and the
        // planar deinterleave shader applies the int8 bias itself (is_int8
        // flag on the draw below).
        let dst_roi = RegionOfInterest {
            left: -1.,
            top: 1.,
            right: 1.,
            bottom: -1.,
        };

        self.draw_intermediate_to_rgb_planar(dst_roi, alpha, is_int8)?;

        unsafe { gls::gl::Finish() };
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Allocates or resizes the intermediate RGBA texture for two-pass packed RGB.
    fn ensure_packed_rgb_intermediate(&mut self, width: usize, height: usize) -> crate::Result<()> {
        if self.packed_rgb_intermediate_size == (width, height) {
            return Ok(());
        }
        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.packed_rgb_intermediate_tex.id);
            super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::NEAREST);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::RGBA as i32,
                width as i32,
                height as i32,
                0,
                gls::gl::RGBA,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            check_gl_error(function!(), line!())?;
        }
        self.packed_rgb_intermediate_size = (width, height);
        Ok(())
    }

    /// Draw a fullscreen quad for the currently-bound shader program.
    /// Used by the pass-2 packing shader in the two-pass packed RGB pipeline.
    fn draw_fullscreen_quad(&self) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let vertices: [f32; 12] = [
                -1.0, 1.0, 0.0, // top-left
                1.0, 1.0, 0.0, // top-right
                1.0, -1.0, 0.0, // bottom-right
                -1.0, -1.0, 0.0, // bottom-left
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * vertices.len()) as isize,
                vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            // Texture coordinates (the packed shader uses gl_FragCoord, not tc,
            // but we still need valid buffers for the vertex attribute layout)
            let tex_coords: [f32; 8] = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * tex_coords.len()) as isize,
                tex_coords.as_ptr() as *const c_void,
            );

            let indices: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                indices.len() as i32,
                gls::gl::UNSIGNED_INT,
                indices.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn clear_rect_planar(
        &self,
        width: usize,
        height: usize,
        dst_roi: RegionOfInterest,
        color: [f32; 4],
        alpha: bool,
    ) -> Result<(), Error> {
        if !alpha && color[0] == color[1] && color[1] == color[2] {
            unsafe {
                gls::gl::ClearColor(color[0], color[0], color[0], 1.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            };
        }

        let split = if alpha { 4 } else { 3 };

        unsafe {
            gls::gl::Enable(gls::gl::SCISSOR_TEST);
            let x = (((dst_roi.left + 1.0) / 2.0) * width as f32).round() as i32;
            let y = (((dst_roi.bottom + 1.0) / 2.0) * height as f32).round() as i32;
            let width = (((dst_roi.right - dst_roi.left) / 2.0) * width as f32).round() as i32;
            let height = (((dst_roi.top - dst_roi.bottom) / 2.0) * height as f32 / split as f32)
                .round() as i32;
            for (i, c) in color.iter().enumerate().take(split) {
                gls::gl::Scissor(x, y + i as i32 * height, width, height);
                gls::gl::ClearColor(*c, *c, *c, 1.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            }
            gls::gl::Disable(gls::gl::SCISSOR_TEST);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_camera_texture_to_rgb_planar(
        &mut self,
        src_key: EglCacheKey,
        egl_img: egl::Image,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        alpha: bool,
        int8: bool,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_EXTERNAL_OES;
        match flip {
            Flip::None => {}
            Flip::Vertical => {
                std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
            }
            Flip::Horizontal => {
                std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
            }
        }
        unsafe {
            let program = if int8 {
                &self.texture_program_planar_int8
            } else {
                &self.texture_program_planar
            };
            gls::gl::UseProgram(program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            super::core::set_tex_filter(texture_target, gls::gl::LINEAR);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_WRAP_S,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_WRAP_T,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            if self
                .camera_eglimage_texture
                .bind_egl_image_external(src_key, egl_img.as_ptr())
            {
                check_gl_error(function!(), line!())?;
                log::trace!(
                    "draw_camera_planar: bound new src EGLImage id={:#x}",
                    src_key.luma_id
                );
            } else {
                log::trace!(
                    "draw_camera_planar: reusing bound src EGLImage id={:#x}",
                    src_key.luma_id
                );
            }
            let y_centers = if alpha {
                vec![-3.0 / 4.0, -1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0]
            } else {
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            };
            let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE, gls::gl::ALPHA];
            // starts from bottom
            for (i, y_center) in y_centers.iter().enumerate() {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top / 3.0 + y_center,
                    0., // left top
                    dst_roi.right,
                    dst_roi.top / 3.0 + y_center,
                    0., // right top
                    dst_roi.right,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // right bottom
                    dst_roi.left,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // left bottom
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let texture_vertices: [f32; 16] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];

                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * 8) as isize,
                    (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );
                let vertices_index: [u32; 4] = [0, 1, 2, 3];
                // self.texture_program_planar
                //     .load_uniform_1i(c"color_index", 2 - i as i32);

                gls::gl::TexParameteri(
                    texture_target,
                    gls::gl::TEXTURE_SWIZZLE_R,
                    swizzles[i] as i32,
                );

                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
            // Reset the texture swizzle to identity before returning. The
            // loop above left `TEXTURE_SWIZZLE_R` pointing at the last
            // requested planar channel (`GL_BLUE` for RGB, `GL_ALPHA` when
            // `alpha` is true). The GLES spec says `GL_TEXTURE_SWIZZLE_*` is
            // undefined for `GL_TEXTURE_EXTERNAL_OES`, but NXP Vivante and
            // Mali drivers honor it on external textures and the swizzle
            // persists across subsequent samples — including the bg blit in
            // `draw_decoded_masks`, which then channel-permutes the entire
            // overlay's background. Restoring identity before any later
            // sampler sees this texture is the safe move.
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_SWIZZLE_R,
                gls::gl::RED as i32,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Draw the intermediate RGBA texture to planar RGB output.
    ///
    /// Pass 2 of the two-pass NV12→PlanarRgb workaround for Vivante GPUs.
    /// Mirrors [`draw_camera_texture_to_rgb_planar`] but sources from
    /// `packed_rgb_intermediate_tex` (a `TEXTURE_2D`) instead of an EGLImage
    /// (`TEXTURE_EXTERNAL_OES`). No rotation/flip — those were handled in pass 1.
    fn draw_intermediate_to_rgb_planar(
        &self,
        dst_roi: RegionOfInterest,
        alpha: bool,
        int8: bool,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        unsafe {
            let program = if int8 {
                &self.texture_program_planar_int8_2d
            } else {
                &self.texture_program_planar_2d
            };
            gls::gl::UseProgram(program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(texture_target, self.packed_rgb_intermediate_tex.id);
            super::core::set_tex_filter_clamp(texture_target, gls::gl::LINEAR);

            // (`tex` = unit 0 is constant per program, uploaded at link time.)
            check_gl_error(function!(), line!())?;

            let y_centers = if alpha {
                vec![-3.0 / 4.0, -1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0]
            } else {
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            };
            let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE, gls::gl::ALPHA];

            // Source ROI is always fullscreen (intermediate is already at destination size)
            let src_roi = RegionOfInterest {
                left: 0.0,
                top: 1.0,
                right: 1.0,
                bottom: 0.0,
            };

            for (i, y_center) in y_centers.iter().enumerate() {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top / 3.0 + y_center,
                    0., // left top
                    dst_roi.right,
                    dst_roi.top / 3.0 + y_center,
                    0., // right top
                    dst_roi.right,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // right bottom
                    dst_roi.left,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // left bottom
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                // No rotation — pass 1 already handled it. Use base texture coords directly.
                let texture_vertices: [f32; 8] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * texture_vertices.len()) as isize,
                    texture_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                let vertices_index: [u32; 4] = [0, 1, 2, 3];

                gls::gl::TexParameteri(
                    texture_target,
                    gls::gl::TEXTURE_SWIZZLE_R,
                    swizzles[i] as i32,
                );

                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
            // Reset the texture swizzle to identity (see the matching
            // comment in `draw_camera_texture_to_rgb_planar` above). This
            // path operates on a `TEXTURE_2D` intermediate, which honors the
            // swizzle per spec — leaving `TEXTURE_SWIZZLE_R` set to the last
            // planar channel selected (for example `GL_BLUE` for RGB or
            // `GL_ALPHA` for RGBA) poisons every later sampler bound to this
            // texture object.
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_SWIZZLE_R,
                gls::gl::RED as i32,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_src_texture(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
    ) -> Result<(), Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "draw_src_texture does not support {src_fmt:?} (use DMA-BUF path for YUV)",
                )));
            }
        };
        // Draw-time program selection: the int8 program is the same shader
        // plus the XOR-0x80 bias, selected here instead of swap-and-restore
        // around the render.
        let program_id = if is_int8 {
            self.texture_int8_program.id
        } else {
            self.texture_program.id
        };
        unsafe {
            gls::gl::UseProgram(program_id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            super::core::set_tex_filter_clamp(texture_target, gls::gl::LINEAR);
            if src_fmt == PixelFormat::Grey {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src_comp) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src_comp as i32);
                }
            }
            // The source map exposes the full row-padded allocation (DMA/IOSurface
            // tensors carry a 64-byte-aligned pitch). TexImage2D otherwise reads
            // `src_w` tight pixels per row from a padded buffer, shearing every row
            // after the first — the failure mode for odd-width Grey/RGB, whose
            // 1-/3-bpp pitch is not 4-aligned so they can't take the stride-aware
            // EGLImage path. Tell GL the real row length (in pixels) so it skips
            // the padding; reset afterwards so other uploads stay tight.
            let src_bpp = src_fmt.channels().max(1);
            let row_len_px = src
                .effective_row_stride()
                .map(|s| s / src_bpp)
                .filter(|&px| px != src_w)
                .unwrap_or(0);
            gls::gl::PixelStorei(gls::gl::UNPACK_ROW_LENGTH, row_len_px as i32);
            self.camera_normal_texture.update_texture(
                texture_target,
                src_w,
                src_h,
                texture_format,
                &src.map()?,
            );
            gls::gl::PixelStorei(gls::gl::UNPACK_ROW_LENGTH, 0);

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];

            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            check_gl_error(function!(), line!())?;

            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_camera_texture_eglimage(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        egl_img: egl::Image,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
    ) -> Result<(), Error> {
        let src_key = EglCacheKey::from_tensor(src, src_fmt, false);
        let luma_id = src_key.luma_id;

        // Draw-time program selection (see draw_src_texture).
        let program_id = if is_int8 {
            self.texture_int8_program_yuv.id
        } else {
            self.texture_program_yuv.id
        };
        let texture_target = gls::gl::TEXTURE_EXTERNAL_OES;
        unsafe {
            gls::gl::UseProgram(program_id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            super::core::set_tex_filter_clamp(texture_target, gls::gl::LINEAR);

            // Note: GL_TEXTURE_SWIZZLE_* is not supported for
            // GL_TEXTURE_EXTERNAL_OES in GLES. YUV→RGB conversion is
            // handled by the driver when sampling from an external texture,
            // and greyscale EGLImages replicate luma to all channels
            // automatically via the YUV shader.

            if self
                .camera_eglimage_texture
                .bind_egl_image_external(src_key, egl_img.as_ptr())
            {
                check_gl_error(function!(), line!())?;
                log::trace!("draw_camera: bound new src EGLImage id={luma_id:#x}");
            } else {
                log::trace!("draw_camera: reusing bound src EGLImage id={luma_id:#x}");
            }
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// `ShaderR8`: render NV12/NV16/NV24 → RGBA8 via the R8 texelFetch shader.
    ///
    /// The combined semi-planar buffer is presented as a single-plane R8
    /// `TEXTURE_2D` and the shader addresses Y and UV bytes directly,
    /// parameterised by four uniforms derived from the source format.
    ///
    /// `r8_src` selects how the R8 texture is established:
    /// - `Some(egl_img)` — zero-copy EGLImage import (DMA-BUF source, the
    ///   `create_image_from_dma_nv_r8` path).
    /// - `None` — CPU upload of the combined buffer via `glTexImage2D` (the
    ///   non-DMA / PBO path, e.g. orin where DMA-BUF EGLImage import is
    ///   unavailable). Same shader, same exact in-shader matrix.
    ///
    /// Renders into the currently-bound FBO (the RGBA8 intermediate or the
    /// DMA destination) — identical output slot to `draw_camera_texture_eglimage`.
    #[allow(clippy::too_many_arguments)]
    fn draw_nv_texture_2d(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        r8_src: Option<egl::Image>,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
    ) -> Result<(), Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let tex_width = src.effective_row_stride().unwrap_or(src_w) as i32;

        // Format-specific shader uniforms from the shared `chroma_layout`
        // (single source of truth with the codec writer + CPU readers + macOS
        // shader). `chroma_lines` (== `uv_rows_per_luma`) is the number of R8
        // buffer rows each image-chroma-row occupies (NV24's CbCr row is 2W
        // bytes = 2 rows; NV12/NV16 are W bytes = 1 row). The shader uses it for
        // direct 2D texel addressing (no per-pixel integer divide/modulo, which
        // is pathologically slow on some embedded GPUs e.g. Vivante GC7000UL).
        let layout = src_fmt.chroma_layout().ok_or_else(|| {
            Error::NotSupported(format!(
                "draw_nv_texture_2d: unsupported format {src_fmt:?}"
            ))
        })?;
        let chroma_shift_x = layout.shift_x as i32;
        let chroma_shift_y = layout.shift_y as i32;
        let chroma_lines = layout.uv_rows_per_luma as i32;

        let src_key = EglCacheKey::from_tensor(src, src_fmt, false);
        let luma_id = src_key.luma_id;

        // Draw-time program selection: the int8 NV program is the same shader
        // plus the XOR-0x80 bias. Selected here for EVERY destination lowering
        // (the old swap scheme only covered DMA destinations, leaving the
        // heap-source int8 NV output un-biased on texture destinations).
        let prog_id = if is_int8 {
            self.nv_r8_int8_program.id
        } else {
            self.nv_r8_program.id
        };

        // YUV→RGB matrix + range, resolved from the source tensor's
        // colorimetry (missing axes filled by the SD/HD height heuristic).
        // Path B applies the matrix in-shader, so it is colorimetry-correct
        // on every GPU regardless of EGL color-hint support.
        let cm = crate::colorimetry::resolve_colorimetry(src.colorimetry(), src.height());
        let colorimetry = (
            cm.encoding
                .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709),
            cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited),
        );
        let state = if is_int8 {
            &mut self.nv_r8_int8_uniforms
        } else {
            &mut self.nv_r8_uniforms
        };
        let locs = state.locs;
        // Uniform values persist per program: re-upload the six matrix floats
        // only when this program's (encoding, range) actually changed. The
        // marker is recorded AFTER the draw succeeds — an error between here
        // and the upload must not leave a "already uploaded" claim for values
        // that never reached the program.
        let upload_colorimetry = state.last_colorimetry != Some(colorimetry);

        unsafe {
            gls::gl::UseProgram(prog_id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.nv_r8_texture.id);
            // NEAREST — we address by integer texel; no interpolation wanted.
            super::core::set_tex_filter_clamp(gls::gl::TEXTURE_2D, gls::gl::NEAREST);

            match r8_src {
                Some(egl_img) => {
                    if self.nv_r8_texture.bind_egl_image(src_key, egl_img.as_ptr()) {
                        check_gl_error(function!(), line!())?;
                        log::trace!("draw_nv_texture_2d: bound new R8 EGLImage id={luma_id:#x}");
                    } else {
                        log::trace!(
                            "draw_nv_texture_2d: reusing bound R8 EGLImage id={luma_id:#x}"
                        );
                    }
                }
                None => {
                    // Non-DMA upload path: upload the combined semi-planar buffer
                    // as an R8 texture (tex_width × combined_plane_height). The
                    // shader addresses it identically to the EGLImage case, so
                    // the in-shader YUV matrix is byte-for-byte the same as the
                    // DMA ShaderR8 path. Used where DMA-BUF EGLImage import is
                    // unavailable (e.g. orin) or the source is heap-backed.
                    //
                    // A PBO source MUST NOT reach here: `map()` on a PBO tensor on
                    // the GL thread deadlocks (the buffer is GL-owned). PBO sources
                    // go through `draw_src_texture_from_pbo`; guard the invariant
                    // locally rather than relying solely on the dispatch call graph.
                    if src.as_pbo().is_some() {
                        return Err(Error::NotSupported(
                            "NV R8 upload cannot map a PBO source on the GL thread; \
                             route PBO sources through the PBO upload path"
                                .into(),
                        ));
                    }
                    let combined_h = src_fmt.combined_plane_height(src_h).ok_or_else(|| {
                        Error::NotSupported(format!(
                            "draw_nv_texture_2d upload: {src_fmt:?} is not semi-planar"
                        ))
                    })?;
                    let offset = src.plane_offset().unwrap_or(0);
                    let needed = tex_width as usize * combined_h;
                    let map = src.map()?;
                    let bytes = map.as_slice();
                    if offset + needed > bytes.len() {
                        return Err(Error::InvalidShape(format!(
                            "NV R8 upload: need {needed} bytes at offset {offset} but buffer is {}",
                            bytes.len()
                        )));
                    }
                    // `UNPACK_ALIGNMENT` is 1 (set once in `new`), so any row width
                    // is valid — no 4-aligned-pitch requirement on the upload.
                    // (TODO/EDGEAI: reuse storage via glTexSubImage2D when dims are
                    // unchanged instead of reallocating every frame — tracked
                    // follow-up; needs EGLImage-vs-upload storage tracking to be safe.)
                    gls::gl::TexImage2D(
                        gls::gl::TEXTURE_2D,
                        0,
                        gls::gl::R8 as i32,
                        tex_width,
                        combined_h as i32,
                        0,
                        gls::gl::RED,
                        gls::gl::UNSIGNED_BYTE,
                        bytes[offset..].as_ptr() as *const c_void,
                    );
                    check_gl_error(function!(), line!())?;
                    // The texture now holds uploaded pixels, not an EGLImage —
                    // clear the EGLImage binding key so a later DMA convert rebinds.
                    self.nv_r8_texture.invalidate_egl_binding();
                    log::trace!("draw_nv_texture_2d: uploaded R8 ({tex_width}x{combined_h})");
                }
            }

            // Per-source uniforms through link-time-cached locations (the
            // `src` sampler binding is constant and uploaded at resolve time).
            gls::gl::Uniform2i(locs.img_size, src_w as i32, src_h as i32);
            gls::gl::Uniform1i(locs.tex_width, tex_width);
            gls::gl::Uniform2i(locs.chroma_shift, chroma_shift_x, chroma_shift_y);
            gls::gl::Uniform1i(locs.chroma_lines, chroma_lines);

            if upload_colorimetry {
                let coeffs = crate::colorimetry::yuv_to_rgb_coeffs(colorimetry.0, colorimetry.1);
                gls::gl::Uniform1f(locs.y_offset, coeffs.y_offset);
                gls::gl::Uniform1f(locs.y_scale, coeffs.y_scale);
                gls::gl::Uniform1f(locs.c_vr, coeffs.c_vr);
                gls::gl::Uniform1f(locs.c_ug, coeffs.c_ug);
                gls::gl::Uniform1f(locs.c_vg, coeffs.c_vg);
                gls::gl::Uniform1f(locs.c_ub, coeffs.c_ub);
            }

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0.,
                dst_roi.right,
                dst_roi.top,
                0.,
                dst_roi.right,
                dst_roi.bottom,
                0.,
                dst_roi.left,
                dst_roi.bottom,
                0.,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        // The draw (and its colorimetry upload, when taken) succeeded — only
        // now record the program's uploaded (encoding, range).
        if upload_colorimetry {
            let state = if is_int8 {
                &mut self.nv_r8_int8_uniforms
            } else {
                &mut self.nv_r8_uniforms
            };
            state.last_colorimetry = Some(colorimetry);
        }
        Ok(())
    }

    /// Create an R8 EGLImage for Path B (NV* combined-plane import).
    fn create_image_from_dma_nv_r8(
        &self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
    ) -> Result<EglImage, Error> {
        let attrs = super::dma_import::DmaImportAttrs::from_tensor_nv_r8(src, src_fmt)?;
        let egl_img_attr = attrs.to_egl_attribs();
        self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    /// Look up or create the Path-B R8 EGLImage for an NV* source tensor.
    ///
    /// Uses a dedicated cache (`nv_r8_egl_cache`) that is keyed identically to
    /// `src_egl_cache` but stores R8 imports so that the two don't interfere.
    fn get_or_create_nv_r8_egl_image(
        &mut self,
        img: &Tensor<u8>,
        img_fmt: PixelFormat,
    ) -> Result<egl::Image, crate::Error> {
        // The NV R8 path imports a SOURCE (NV12/16/24 as one R8 texture), so it
        // never collapses onto a destination parent import.
        let id = EglCacheKey::from_tensor(img, img_fmt, false);
        let luma_id = id.luma_id;

        if self.nv_r8_egl_cache.sweep() {
            self.nv_r8_texture.invalidate_egl_binding();
        }

        {
            let ts = self.nv_r8_egl_cache.next_timestamp();
            if let Some(cached) = self.nv_r8_egl_cache.entries.get_mut(&id) {
                self.nv_r8_egl_cache.hits += 1;
                cached.last_used = ts;
                log::trace!("nv_r8_egl_cache hit: id={luma_id:#x}");
                return Ok(cached.egl_image.egl_image);
            }
            self.nv_r8_egl_cache.misses += 1;
            log::trace!("nv_r8_egl_cache miss: id={luma_id:#x}");
        }

        let egl_image_obj = self.create_image_from_dma_nv_r8(img, img_fmt)?;
        self.nv_r8_texture.invalidate_egl_binding();

        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        if self.nv_r8_egl_cache.entries.len() >= self.nv_r8_egl_cache.capacity {
            self.nv_r8_egl_cache.evict_lru();
        }
        let ts = self.nv_r8_egl_cache.next_timestamp();
        self.nv_r8_egl_cache.entries.insert(
            id,
            super::cache::CachedEglImage {
                egl_image: egl_image_obj,
                last_used: ts,
                guard,
                renderbuffer: None,
            },
        );
        Ok(handle)
    }

    fn create_image_from_dma2(
        &self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        for_dst: bool,
    ) -> Result<EglImage, crate::Error> {
        let attrs = super::dma_import::DmaImportAttrs::from_tensor(src, src_fmt, for_dst)?;
        let egl_img_attr = attrs.to_egl_attribs();
        self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    fn new_egl_image_owned(
        &'_ self,
        target: egl::Enum,
        attrib_list: &[Attrib],
    ) -> Result<EglImage, Error> {
        // Every EGLImage creation funnels through here (DMA-BUF, NV R8, RGB
        // renderbuffer paths), so one span = one actual `eglCreateImageKHR`.
        // Steady-state frame loops must show ZERO of these after warmup — the
        // span count is the observable for cache-behavior equality gates.
        let _span = tracing::trace_span!("image.convert.gl.egl_import", target).entered();
        // EGLImage creation is a display-level EGL op shared by every
        // processor — serialized by a dedicated short lock (zero cost in
        // steady state: creations are cache-miss-only).
        let _image_guard = super::context::image_lifecycle_guard();
        let image = GlContext::egl_create_image_with_fallback(
            &self.gl_context.egl,
            self.gl_context.display,
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            target,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            attrib_list,
        )?;
        Ok(EglImage {
            egl_image: image,
            display: self.gl_context.display,
            egl: Rc::clone(&self.gl_context.egl),
        })
    }

    /// Look up or create an EGLImage for a DMA tensor, returning the EGL image handle.
    ///
    /// Returns `egl::Image` (a `Copy` type wrapping `*const c_void`) to avoid borrow
    /// conflicts with the caller. The cache retains ownership of the `EglImage` value;
    /// the handle remains valid as long as the entry lives in the cache.
    fn get_or_create_egl_image(
        &mut self,
        cache: CacheKind,
        img: &Tensor<u8>,
        img_fmt: PixelFormat,
    ) -> Result<egl::Image, crate::Error> {
        // Identity + offset + geometry: sub-region views share one buffer
        // identity but need distinct EGLImages (offset), and a pooled buffer
        // reconfigured to a new size/format/stride needs a fresh import
        // (geometry) — see EglCacheKey. Only a destination view collapses onto
        // its parent key; a source view keys on its own region.
        let id = EglCacheKey::from_tensor(img, img_fmt, cache == CacheKind::Dst);
        let luma_id = id.luma_id;

        // Sweep dead entries opportunistically before looking up.
        // Invalidate texture binding state since sweep may remove a bound entry.
        match cache {
            CacheKind::Src => {
                if self.src_egl_cache.sweep() {
                    self.invalidate_src_textures();
                }
            }
            CacheKind::Dst => {
                if self.dst_egl_cache.sweep() {
                    self.invalidate_dst_textures();
                }
            }
        }

        {
            let egl_cache = match cache {
                CacheKind::Src => &mut self.src_egl_cache,
                CacheKind::Dst => &mut self.dst_egl_cache,
            };
            let ts = egl_cache.next_timestamp();
            if let Some(cached) = egl_cache.entries.get_mut(&id) {
                egl_cache.hits += 1;
                cached.last_used = ts;
                log::trace!("EglImageCache {:?} hit: id={luma_id:#x}", cache);
                return Ok(cached.egl_image.egl_image);
            }
            egl_cache.misses += 1;
            log::trace!("EglImageCache {:?} miss: id={luma_id:#x}", cache);
        }

        // Create the EGL image BEFORE evicting — if creation fails, we don't
        // want to have destroyed a valid cache entry for nothing. Only a
        // destination view imports its parent (glViewport tiling); a source view
        // imports its own region (it is sampled, not rendered into).
        let for_dst = cache == CacheKind::Dst;
        let egl_image_obj = self.create_image_from_dma2(img, img_fmt, for_dst)?;

        // Optionally create a GL renderbuffer backed by this EGLImage for use as an FBO
        // color attachment.  Renderbuffers are required on Mali/Neutron GPUs (i.MX 95)
        // but are not supported on all drivers (e.g. Vivante on i.MX 8MP).
        // Enabled via EDGEFIRST_OPENGL_RENDERSURFACE=1; defaults to the texture path.
        let rbo = if cache == CacheKind::Dst && self.use_renderbuffer {
            let mut rbo: u32 = 0;
            unsafe {
                gls::gl::GenRenderbuffers(1, &mut rbo);
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
                gls::gl::EGLImageTargetRenderbufferStorageOES(
                    gls::gl::RENDERBUFFER,
                    egl_image_obj.egl_image.as_ptr(),
                );
                if let Err(e) = check_gl_error(function!(), line!()) {
                    gls::gl::DeleteRenderbuffers(1, &rbo);
                    return Err(e);
                }
            }
            Some(rbo)
        } else {
            None
        };

        // Invalidate texture binding state: we're inserting a new entry and
        // may evict the currently-bound one.
        match cache {
            CacheKind::Src => self.invalidate_src_textures(),
            CacheKind::Dst => self.invalidate_dst_textures(),
        }

        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        let egl_cache = match cache {
            CacheKind::Src => &mut self.src_egl_cache,
            CacheKind::Dst => &mut self.dst_egl_cache,
        };
        // Evict least-recently-used entry if at capacity.
        if egl_cache.entries.len() >= egl_cache.capacity {
            egl_cache.evict_lru();
        }
        let ts = egl_cache.next_timestamp();
        egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: rbo,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    /// Look up the renderbuffer ID for a cached destination EGLImage.
    fn cached_dst_renderbuffer<T>(&self, img: &Tensor<T>, fmt: PixelFormat) -> Option<u32>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        let id = EglCacheKey::from_tensor(img, fmt, true);
        self.dst_egl_cache
            .entries
            .get(&id)
            .and_then(|entry| entry.renderbuffer)
    }

    /// Create an EGLImage from a DMA buffer with explicitly specified internal
    /// dimensions and format. Used when the GL render surface differs from the
    /// logical image dimensions (e.g., packed RGB reinterpretation, or the F16
    /// RGBA16F-packed planar destination).
    ///
    /// Generic over the tensor element type so it works for both `u8`
    /// (packed-RGB DMA) and `f16` (RGBA16F-packed DMA) destinations. Only the
    /// DMA fd, stride and offset are read from the tensor; `width`, `height`,
    /// `drm_format` and `bpp` describe the GL-visible packed surface.
    fn create_egl_image_with_dims<T>(
        &self,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<EglImage, crate::Error>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        let dma = img.as_dma().ok_or_else(|| {
            Error::NotImplemented("create_egl_image_with_dims requires DMA tensor".to_string())
        })?;
        let fd = dma.fd.as_raw_fd();

        // Use the tensor's stored stride when available (externally allocated
        // buffers with row padding), otherwise compute the tightly-packed pitch.
        let pitch = img.effective_row_stride().unwrap_or(width * bpp);
        let offset = img.plane_offset().unwrap_or(0);
        let egl_img_attr = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            drm_format as u32 as Attrib,
            khronos_egl::WIDTH as Attrib,
            width as Attrib,
            khronos_egl::HEIGHT as Attrib,
            height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            pitch as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            offset as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
            khronos_egl::NONE as Attrib,
        ];

        self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    /// Get or create an EGLImage for a packed DMA destination with
    /// reinterpreted dimensions. Uses the dst cache keyed by buffer identity.
    ///
    /// Generic over the tensor element type so it serves both the `u8`
    /// packed-RGB destination (`DrmFourcc::Abgr8888`, bpp=4) and the `f16`
    /// RGBA16F-packed planar destination (`DrmFourcc::Abgr16161616f`, bpp=8).
    fn get_or_create_egl_image_rgb<T>(
        &mut self,
        img: &Tensor<T>,
        img_fmt: PixelFormat,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<egl::Image, crate::Error>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        // Keyed identically to `cached_dst_renderbuffer` and the logical-dims
        // dst path: the packed render dims derive deterministically from the
        // tensor's logical geometry, so `from_tensor` is a consistent key.
        let id = EglCacheKey::from_tensor(img, img_fmt, true);
        if self.dst_egl_cache.sweep() {
            self.invalidate_dst_textures();
        }

        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            self.dst_egl_cache.hits += 1;
            cached.last_used = ts;
            log::trace!("EglImageCache dst (RGB) hit: id={:#x}", id.luma_id);
            return Ok(cached.egl_image.egl_image);
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (RGB) miss: id={:#x}", id.luma_id);
        // Invalidate dst texture binding state on cache miss (new EGLImage creation).
        self.invalidate_dst_textures();

        if self.dst_egl_cache.entries.len() >= self.dst_egl_cache.capacity {
            self.dst_egl_cache.evict_lru();
        }

        let egl_image_obj = self.create_egl_image_with_dims(img, width, height, drm_format, bpp)?;
        let handle = egl_image_obj.egl_image;

        let rbo = if self.use_renderbuffer {
            let mut rbo: u32 = 0;
            unsafe {
                gls::gl::GenRenderbuffers(1, &mut rbo);
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
                gls::gl::EGLImageTargetRenderbufferStorageOES(
                    gls::gl::RENDERBUFFER,
                    egl_image_obj.egl_image.as_ptr(),
                );
                if let Err(e) = check_gl_error(function!(), line!()) {
                    gls::gl::DeleteRenderbuffers(1, &rbo);
                    return Err(e);
                }
            }
            Some(rbo)
        } else {
            None
        };

        let guard = img.buffer_identity().weak();
        let ts = self.dst_egl_cache.next_timestamp();
        self.dst_egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: rbo,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    // Reshapes the segmentation to be compatible with RGBA texture array rendering.
    fn reshape_segmentation_to_rgba(&self, segmentation: &[u8], shape: [usize; 3]) -> Vec<u8> {
        let [height, width, classes] = shape;

        let n_layer_stride = height * width * 4;
        let n_row_stride = width * 4;
        let n_col_stride = 4;
        let row_stride = width * classes;
        let col_stride = classes;

        let mut new_segmentation = vec![0u8; n_layer_stride * classes.div_ceil(4)];

        for i in 0..height {
            for j in 0..width {
                for k in 0..classes.div_ceil(4) * 4 {
                    if k >= classes {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = 0;
                    } else {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = segmentation[i * row_stride + j * col_stride + k];
                    }
                }
            }
        }

        new_segmentation
    }

    fn render_modelpack_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 3],
    ) -> Result<(), crate::Error> {
        log::debug!("start render_segmentation_to_image");

        // TODO: Implement specialization for 2 classes and 4 classes which shouldn't
        // need rearranging the data
        let new_segmentation = self.reshape_segmentation_to_rgba(segmentation, shape);

        let [height, width, classes] = shape;

        let format = gls::gl::RGBA;
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;
        self.segmentation_program
            .load_uniform_1i(c"background_index", shape[2] as i32 - 1)?;

        gls::use_program(self.segmentation_program.id);

        gls::bind_texture(texture_target, self.segmentation_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_image3d(
            texture_target,
            0,
            format as i32,
            width as i32,
            height as i32,
            classes.div_ceil(4) as i32,
            0,
            format,
            gls::gl::UNSIGNED_BYTE,
            Some(&new_segmentation),
        );

        let src_roi = RegionOfInterest {
            left: 0.,
            top: 1.,
            right: 1.,
            bottom: 0.,
        };

        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[0..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }

        Ok(())
    }

    /// Bind the instanced segmentation program and configure the persistent
    /// mask texture for a batch of `render_yolo_segmentation` calls.
    ///
    /// Call once before the per-detection loop. Hoisting this out of
    /// `render_yolo_segmentation` avoids N× redundant `glTexParameteri`,
    /// `glUseProgram`, and `glBindTexture` calls per frame.
    fn setup_yolo_segmentation_pass(&self) {
        let texture_target = gls::gl::TEXTURE_2D;
        gls::use_program(self.instanced_segmentation_program.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.segmentation_texture.id);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
    }

    /// Render a single pre-decoded YOLO segmentation mask as a coloured
    /// overlay quad. The caller must invoke [`setup_yolo_segmentation_pass`]
    /// once before the first call in a batch so the program, bound texture,
    /// and texture parameters are already set.
    ///
    /// Each call uploads the mask via `glTexImage2D`, which implicitly
    /// orphans the previous storage on most drivers. This avoids the
    /// read-after-write hazard a persistent texture would introduce on
    /// Vivante, which serialises every `glTexSubImage2D` against the still
    /// in-flight previous draw and ends up slower than orphaning.
    /// Compared with the original implementation we still drop the
    /// per-instance CPU 1 px zero-pad copy — the shader's
    /// `smoothstep(0.5, 0.65)` provides the edge antialiasing that the
    /// padding used to proxy for, and texel-centre UVs keep bilinear
    /// sampling strictly inside the uploaded mask region.
    fn render_yolo_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 2],
        class: usize,
    ) -> Result<(), crate::Error> {
        let [height, width] = shape;
        let texture_target = gls::gl::TEXTURE_2D;

        // Per-instance allocation + upload, equivalent to the old code path
        // but without the CPU pad copy. `glTexImage2D` here implicitly
        // orphans the texture's previous backing store on Vivante / Mali —
        // the in-flight previous draw keeps the old storage alive while
        // the new mask uploads to fresh memory, preserving CPU/GPU
        // parallelism.
        gls::tex_image2d(
            texture_target,
            0,
            gls::gl::R8 as i32,
            width as i32,
            height as i32,
            0,
            gls::gl::RED,
            gls::gl::UNSIGNED_BYTE,
            Some(segmentation),
        );

        self.instanced_segmentation_program
            .load_uniform_1i(c"class_index", class as i32)?;

        // Texel-centre UVs: bilinear filtering returns the exact uploaded
        // texel value at each corner of the quad, so no sampling reaches
        // outside the mask data and no zero-padding border is needed.
        let u_lo = 0.5 / width as f32;
        let v_lo = 0.5 / height as f32;
        let u_hi = (width as f32 - 0.5) / width as f32;
        let v_hi = (height as f32 - 0.5) / height as f32;

        // tc.y = 0 corresponds to mask row 0 (image top); NDC top of the quad
        // (`dst_roi.top`, which holds `cvt_screen_coord(seg.ymax)`) renders
        // image bottom, so the first two tex vertices get `v_hi`.
        let src_roi = RegionOfInterest {
            left: u_lo,
            top: v_hi,
            right: u_hi,
            bottom: v_lo,
        };

        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );

            // Vivante (i.MX 8MP GC7000UL) regresses ~2× when many masks
            // queue without periodic drains: the driver appears to enter
            // a slow allocation path once its command buffer or texture
            // orphan queue fills. An explicit `glFinish()` per instance
            // matches the legacy behaviour and keeps the per-instance
            // cost bounded.
            //
            // Mali Valhall (i.MX 95) is the opposite — `glFinish()` here
            // forces a TBDR tile store-unload of the framebuffer per
            // draw, which dominates the per-instance cost. Letting the
            // pipeline batch into the single `gls::finish()` at the end
            // of `draw_decoded_masks_impl` recovers ~30% on a 40-mask
            // crowd scene.
            if self.is_vivante {
                gls::gl::Finish();
            }
        }

        Ok(())
    }

    /// Repack proto tensor `(H, W, num_protos)` as f32 into RGBA f16 layers
    /// suitable for upload to a GL_TEXTURE_2D_ARRAY with GL_RGBA16F.
    ///
    /// Returns `(repacked_bytes, num_layers)` where each layer is H*W*4 half-floats.
    /// Render YOLO proto segmentation masks using the fused GPU pipeline.
    ///
    /// Dispatches on the proto tensor's runtime dtype via
    /// [`edgefirst_tensor::TensorDyn::dtype`]:
    /// - `I8`: uploads raw int8 as `GL_R8I`, dequantizes in shader (requires
    ///   per-tensor [`edgefirst_tensor::Quantization`] on the proto tensor).
    /// - `F32`: uploads as `GL_R32F` with hardware bilinear (if available),
    ///   or falls back to the `RGBA16F` repack path for older drivers.
    /// - `F16`: repacks natively into `RGBA16F` without a widening step.
    fn render_proto_segmentation(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use edgefirst_tensor::{DType, QuantMode, TensorMapTrait, TensorTrait};

        if detect.is_empty() {
            return Ok(());
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(crate::Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        // Interpret shape based on physical layout.
        let (height, width, num_protos) = match proto_data.layout {
            ProtoLayout::Nhwc => (proto_shape[0], proto_shape[1], proto_shape[2]),
            ProtoLayout::Nchw => (proto_shape[1], proto_shape[2], proto_shape[0]),
        };
        let coeff_shape = proto_data.mask_coefficients.shape();
        if coeff_shape.len() != 2 || coeff_shape[1] != num_protos {
            return Err(crate::Error::InvalidShape(format!(
                "mask_coefficients shape {coeff_shape:?} incompatible with protos \
                 {proto_shape:?} (expected [N, {num_protos}])"
            )));
        }
        // Genuine "no detections this frame" → nothing to render.
        if coeff_shape[0] == 0 {
            return Ok(());
        }
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;

        // Pure path decision: upload strategy × program × count uniform.
        // Rejects NCHW layouts and unsupported proto dtypes up front (the
        // caller falls back to the CPU path on NotSupported).
        let plan = super::proto_dispatch::plan_proto(
            proto_data.protos.dtype(),
            proto_data.layout,
            self.proto_repack_compute_program.is_some(),
            self.has_float_linear,
            self.int8_interpolation_mode,
        )?;

        // The proto render's span — the path previously had none (only a
        // FunctionTimer wall-clock at the draw_proto_masks entry). Plan
        // fields make the chosen upload/program visible in traces.
        let _span = tracing::trace_span!(
            "image.draw.gl.proto",
            dtype = ?proto_data.protos.dtype(),
            upload = ?plan.upload,
            program = ?plan.program,
            num_protos,
            detections = detect.len(),
        )
        .entered();

        // Coefficient slice for the GL uniform upload path. The shaders
        // consume f32 regardless of source dtype, but we hold the F32 case
        // as a borrowed slice (no allocation) and only widen for F16. The
        // `Cow<[f32]>` lets both arms share a single downstream pass.
        //
        // The F16 widening is one flat `Vec<f32>`, replacing the prior
        // `Vec<Vec<f32>>` (one inner Vec per detection) — avoids N small
        // heap allocations per frame on the hot path.
        let mc_map_f32;
        let mc_map_f16;
        let mc_map_i8;
        let coeff_widen_f16: Vec<f32>;
        let coeff_dequant: Vec<f32>;
        let coeff_slice: &[f32] = match proto_data.mask_coefficients.dtype() {
            DType::F32 => {
                let t = proto_data.mask_coefficients.as_f32().expect("F32");
                mc_map_f32 = t.map()?;
                mc_map_f32.as_slice()
            }
            DType::F16 => {
                let t = proto_data.mask_coefficients.as_f16().expect("F16");
                mc_map_f16 = t.map()?;
                coeff_widen_f16 = mc_map_f16.as_slice().iter().map(|v| v.to_f32()).collect();
                &coeff_widen_f16[..]
            }
            DType::I8 => {
                let t = proto_data.mask_coefficients.as_i8().expect("I8");
                mc_map_i8 = t.map()?;
                let quant = t.quantization();
                coeff_dequant = super::proto_dispatch::dequant_coeffs(
                    mc_map_i8.as_slice(),
                    quant.as_ref().map(|q| q.mode()),
                    "I8",
                )?;
                &coeff_dequant[..]
            }
            DType::I16 => {
                let t = proto_data.mask_coefficients.as_i16().expect("I16");
                let mc_map_i16 = t.map()?;
                let quant = t.quantization();
                coeff_dequant = super::proto_dispatch::dequant_coeffs(
                    mc_map_i16.as_slice(),
                    quant.as_ref().map(|q| q.mode()),
                    "I16",
                )?;
                &coeff_dequant[..]
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported on GL seg path"
                )));
            }
        };

        // Each proto-dtype arm holds its `TensorMap` for the duration of the
        // GL upload and passes an `ArrayView3` borrowed from the mapped slice
        // to the helper — no per-frame `to_owned()` clone of the proto tensor.
        match proto_data.protos.dtype() {
            DType::I8 => {
                let t = proto_data.protos.as_i8().expect("I8");
                let m = t.map()?;
                let quant = t.quantization().ok_or_else(|| {
                    crate::Error::InvalidShape("I8 protos require quantization metadata".into())
                })?;
                // GL shader path: per-tensor quant only (shader uploads a
                // single scale/zp uniform). Per-channel would require a
                // shader rewrite — return NotSupported for now.
                let (scale, zp) = match quant.mode() {
                    QuantMode::PerTensor { scale, zero_point } => (scale, zero_point),
                    QuantMode::PerTensorSymmetric { scale } => (scale, 0_i32),
                    QuantMode::PerChannel { axis, .. }
                    | QuantMode::PerChannelSymmetric { axis, .. } => {
                        return Err(crate::Error::NotSupported(format!(
                            "GL seg path: per-channel quantization (axis={axis}) \
                             not yet supported — falls back to CPU in caller"
                        )));
                    }
                };
                let protos_view = ndarray::ArrayView3::<i8>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                let quantization = edgefirst_decoder::Quantization::new(scale, zp);
                self.render_proto_segmentation_int8(
                    plan,
                    detect,
                    coeff_slice,
                    protos_view,
                    &quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            DType::F32 => {
                let t = proto_data.protos.as_f32().expect("F32");
                let m = t.map()?;
                let protos_view = ndarray::ArrayView3::<f32>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                self.render_proto_layers(
                    plan,
                    detect,
                    coeff_slice,
                    ProtoLayersData::F32(protos_view),
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            DType::F16 => {
                let t = proto_data.protos.as_f16().expect("F16");
                let m = t.map()?;
                let protos_view = ndarray::ArrayView3::<half::f16>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                self.render_proto_layers(
                    plan,
                    detect,
                    coeff_slice,
                    ProtoLayersData::F16(protos_view),
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "GL seg path: proto dtype {other:?} not supported"
                )));
            }
        }

        unsafe { gls::gl::Finish() };
        Ok(())
    }

    /// Render detection quads using the active program. Shared by all proto
    /// shader paths.
    fn render_proto_detection_quads(
        &self,
        program: &GlProgram,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        num_protos: usize,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;

        // Resolve the per-detection uniform locations once per program
        // switch — `load_uniform_*` is a driver string lookup plus a
        // redundant UseProgram, and this loop previously paid both twice
        // per detection per frame. The caller has already bound `program`.
        let (cached_id, mut loc_coeff, mut loc_class) = self.proto_quad_locs.get();
        if cached_id != program.id {
            unsafe {
                loc_coeff = gls::gl::GetUniformLocation(program.id, c"mask_coeff".as_ptr());
                loc_class = gls::gl::GetUniformLocation(program.id, c"class_index".as_ptr());
            }
            self.proto_quad_locs.set((program.id, loc_coeff, loc_class));
        }

        // Stride the flat `mask_coefficients` buffer by `num_protos` so we
        // get one slice per detection. `chunks_exact` requires the total
        // length to be a multiple of `num_protos`; the dispatcher already
        // validates `coeff_shape == [N, num_protos]` upstream.
        for (idx, (det, coeff)) in detect
            .iter()
            .zip(mask_coefficients.chunks_exact(num_protos))
            .enumerate()
        {
            let color_index = color_mode.index(idx, det.label);
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (i, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[i / 4][i % 4] = *val;
            }

            unsafe {
                gls::gl::Uniform4fv(loc_coeff, 8, packed_coeff.as_ptr() as *const f32);
                gls::gl::Uniform1i(loc_class, color_index as i32);
            }

            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(det.bbox.xmin),
                top: cvt_screen_coord(det.bbox.ymax),
                right: cvt_screen_coord(det.bbox.xmax),
                bottom: cvt_screen_coord(det.bbox.ymin),
            };

            // Proto texture coords: tex row 0 = image top (data uploaded in
            // row-major order where y=0 is top of image, and GL treats the
            // first row of pixel data as the bottom of the texture — but
            // texelFetch(y=0) returns that bottom row, which is our image top).
            // So tc.y=0 → image top, tc.y=1 → image bottom.
            // At NDC top (higher Y = image bottom = ymax), we want tc.y = ymax.
            // At NDC bottom (lower Y = image top = ymin), we want tc.y = ymin.
            let src_roi = RegionOfInterest {
                left: det.bbox.xmin,
                top: det.bbox.ymax,
                right: det.bbox.xmax,
                bottom: det.bbox.ymin,
            };

            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.bottom,
                    0.,
                    dst_roi.left,
                    dst_roi.bottom,
                    0.,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

                let texture_vertices: [f32; 8] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    texture_vertices.as_ptr() as *const c_void,
                );

                let vertices_index: [u32; 4] = [0, 1, 2, 3];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        Ok(())
    }

    /// Int8 proto path: upload raw i8 protos per `plan.upload`, render per
    /// `plan.program`.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_int8(
        &mut self,
        plan: super::proto_dispatch::ProtoPlan,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        protos: ndarray::ArrayView3<'_, i8>,
        quantization: &edgefirst_decoder::Quantization,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        // Protos are (H, W, num_protos) in row-major HWC. The GL texture
        // needs layer-first CHW (one proto per layer).
        if plan.upload == super::proto_dispatch::ProtoUpload::I8Compute {
            let compute_program = self.proto_repack_compute_program.ok_or_else(|| {
                crate::Error::InvalidShape(
                    "plan selected I8Compute without a compiled compute program".into(),
                )
            })?;
            // === GLES 3.1 compute shader path ===
            // Upload HWC data as-is to SSBO, let GPU transpose via compute.
            // Fall through to CPU repack if proto array is non-contiguous.
            let mut data = match protos.as_slice() {
                Some(s) => std::borrow::Cow::Borrowed(s),
                None => std::borrow::Cow::Owned(protos.iter().copied().collect()),
            };
            // Pad to a 4-byte boundary for SSBO int[] alignment — by copying,
            // not by over-reading past the slice end.
            if data.len() % 4 != 0 {
                let mut owned = data.into_owned();
                owned.resize(owned.len().next_multiple_of(4), 0);
                data = std::borrow::Cow::Owned(owned);
            }
            let data_bytes = data.len();

            // Allocate as R32I (imageStore-compatible; the int8 fragment
            // shaders read integers identically from R8I or R32I via
            // texelFetch). Immutable storage + binding handled by the gate.
            if self.ensure_proto_texture(texture_target, gls::gl::R32I, width, height, num_protos) {
                Self::set_proto_tex_params(texture_target, gls::gl::NEAREST);
            }

            unsafe {
                // Upload HWC data to SSBO
                gls::gl::BindBuffer(gls::gl::SHADER_STORAGE_BUFFER, self.proto_ssbo);
                if data_bytes > self.proto_ssbo_size {
                    gls::gl::BufferData(
                        gls::gl::SHADER_STORAGE_BUFFER,
                        data_bytes as isize,
                        data.as_ptr() as *const std::ffi::c_void,
                        gls::gl::STREAM_DRAW,
                    );
                    self.proto_ssbo_size = data_bytes;
                } else {
                    gls::gl::BufferSubData(
                        gls::gl::SHADER_STORAGE_BUFFER,
                        0,
                        data_bytes as isize,
                        data.as_ptr() as *const std::ffi::c_void,
                    );
                }
                gls::gl::BindBufferBase(gls::gl::SHADER_STORAGE_BUFFER, 0, self.proto_ssbo);

                // Bind texture as image for compute write (R32I for compatibility)
                gls::gl::BindImageTexture(
                    0,
                    self.proto_texture.id,
                    0,
                    gls::gl::TRUE,
                    0,
                    gls::gl::WRITE_ONLY,
                    gls::gl::R32I,
                );

                // Dispatch compute (uniform locations resolved at compile)
                gls::gl::UseProgram(compute_program);
                let (loc_w, loc_h, loc_np) = self.proto_compute_locs;
                gls::gl::Uniform1i(loc_w, width as i32);
                gls::gl::Uniform1i(loc_h, height as i32);
                gls::gl::Uniform1i(loc_np, num_protos as i32);

                let groups_x = width.div_ceil(16) as u32;
                let groups_y = height.div_ceil(16) as u32;
                gls::gl::DispatchCompute(groups_x, groups_y, 1);
                gls::gl::MemoryBarrier(
                    gls::gl::TEXTURE_FETCH_BARRIER_BIT | gls::gl::SHADER_IMAGE_ACCESS_BARRIER_BIT,
                );

                // Unbind SSBO and log any GL errors from compute dispatch
                gls::gl::BindBuffer(gls::gl::SHADER_STORAGE_BUFFER, 0);
                loop {
                    let err = gls::gl::GetError();
                    if err == gls::gl::NO_ERROR {
                        break;
                    }
                    log::debug!("GL error after compute dispatch: 0x{err:x}");
                }
            }
        } else {
            // === GLES 3.0 fallback: CPU repack ===
            let tex_data = super::proto_dispatch::repack_layers(protos);

            self.upload_proto_texture(
                texture_target,
                gls::gl::R8I,
                width,
                height,
                num_protos,
                gls::gl::RED_INTEGER,
                gls::gl::BYTE,
                gls::gl::NEAREST,
                &tex_data,
            );
        }

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        match plan.program {
            super::proto_dispatch::ProtoProgram::Int8Nearest
            | super::proto_dispatch::ProtoProgram::Int8Bilinear => {
                let program = if plan.program == super::proto_dispatch::ProtoProgram::Int8Nearest {
                    &self.proto_segmentation_int8_nearest_program
                } else {
                    &self.proto_segmentation_int8_bilinear_program
                };
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(
                    program,
                    detect,
                    mask_coefficients,
                    num_protos,
                    color_mode,
                )?;
            }
            super::proto_dispatch::ProtoProgram::Int8TwoPass => {
                self.render_proto_int8_two_pass(
                    detect,
                    mask_coefficients,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto program {other:?} is not an int8 program"
                )));
            }
        }

        Ok(())
    }

    /// Two-pass int8 path: dequant int8→RGBA16F FBO, then render with
    /// existing f16 shader using GL_LINEAR.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_int8_two_pass(
        &mut self,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        quantization: &edgefirst_decoder::Quantization,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let num_layers = num_protos.div_ceil(4);

        // Save the caller's FBO and viewport so we can restore after dequant.
        let (saved_fbo, saved_viewport) = unsafe {
            let mut fbo: i32 = 0;
            gls::gl::GetIntegerv(gls::gl::FRAMEBUFFER_BINDING, &mut fbo);
            let mut vp = [0i32; 4];
            gls::gl::GetIntegerv(gls::gl::VIEWPORT, vp.as_mut_ptr());
            (fbo as u32, vp)
        };

        // Pass 1: Dequantize int8 → RGBA16F texture via the persistent
        // dequant FBO. The render target is gated like the proto texture
        // (recreate-on-change immutable storage) instead of a fresh
        // TexImage3D per call.
        if Self::ensure_immutable_tex_array(
            &mut self.proto_dequant_texture,
            &mut self.proto_dequant_tex_dims,
            texture_target,
            gls::gl::RGBA16F,
            width,
            height,
            num_layers,
        ) {
            Self::set_proto_tex_params(texture_target, gls::gl::LINEAR);
        }

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        let dequant_program = &self.proto_dequant_int8_program;
        gls::use_program(dequant_program.id);
        dequant_program.load_uniform_1f(c"proto_scale", proto_scale)?;
        dequant_program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;

        // Bind the int8 proto texture to TEXTURE0 for the dequant shader
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_texture.id);

        // Render each RGBA16F layer (4 protos per layer)
        for layer in 0..num_layers {
            self.proto_dequant_fbo.bind();
            unsafe {
                gls::gl::FramebufferTextureLayer(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    self.proto_dequant_texture.id,
                    0,
                    layer as i32,
                );
                gls::gl::Viewport(0, 0, width as i32, height as i32);
            }
            dequant_program.load_uniform_1i(c"base_layer", (layer * 4) as i32)?;
            self.draw_fullscreen_quad()?;
        }

        // Restore the caller's FBO and viewport.
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, saved_fbo);
            gls::gl::Viewport(
                saved_viewport[0],
                saved_viewport[1],
                saved_viewport[2],
                saved_viewport[3],
            );
        }

        // Pass 2: render with existing f16 shader reading from dequant texture
        let program = &self.proto_segmentation_program;
        gls::use_program(program.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_dequant_texture.id);
        program.load_uniform_1i(c"num_layers", num_layers as i32)?;
        self.render_proto_detection_quads(
            program,
            detect,
            mask_coefficients,
            num_protos,
            color_mode,
        )?;

        Ok(())
    }

    /// One render for every layered-float proto plan — F32→R32F native,
    /// F32→RGBA16F fallback, and native F16→RGBA16F — driven entirely by
    /// the [`ProtoPlan`](super::proto_dispatch::ProtoPlan): upload strategy
    /// picks the repack + texture format, `plan.program` the sampler
    /// program, `plan.count_uniform` the layer-count uniform. Replaces the
    /// three per-dtype `render_proto_segmentation_{f32,f16,f16_native}`
    /// bodies that differed only in those three choices.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_layers(
        &mut self,
        plan: super::proto_dispatch::ProtoPlan,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        data: ProtoLayersData<'_>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use super::proto_dispatch::{CountUniform, ProtoProgram, ProtoUpload};

        let num_layers = num_protos.div_ceil(4);
        match (plan.upload, &data) {
            (ProtoUpload::F32R32f, ProtoLayersData::F32(v)) => {
                // Repack protos to layer-first layout: (num_protos, H, W)
                let tex_data = super::proto_dispatch::repack_layers(*v);
                self.upload_proto_texture(
                    texture_target,
                    gls::gl::R32F,
                    width,
                    height,
                    num_protos,
                    gls::gl::RED,
                    gls::gl::FLOAT,
                    gls::gl::LINEAR,
                    &tex_data,
                );
            }
            (ProtoUpload::F32ToRgba16f, ProtoLayersData::F32(v)) => {
                let (tex_data, _) =
                    super::proto_dispatch::repack_rgba_f16_layers(*v, half::f16::from_f32);
                self.upload_proto_texture(
                    texture_target,
                    gls::gl::RGBA16F,
                    width,
                    height,
                    num_layers,
                    gls::gl::RGBA,
                    gls::gl::HALF_FLOAT,
                    gls::gl::LINEAR,
                    &tex_data,
                );
            }
            (ProtoUpload::F16Rgba16f, ProtoLayersData::F16(v)) => {
                let (tex_data, _) = super::proto_dispatch::repack_rgba_f16_layers(*v, |x| x);
                self.upload_proto_texture(
                    texture_target,
                    gls::gl::RGBA16F,
                    width,
                    height,
                    num_layers,
                    gls::gl::RGBA,
                    gls::gl::HALF_FLOAT,
                    gls::gl::LINEAR,
                    &tex_data,
                );
            }
            (upload, _) => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto upload {upload:?} incompatible with the provided proto data"
                )));
            }
        }

        let program = match plan.program {
            ProtoProgram::F32 => &self.proto_segmentation_f32_program,
            ProtoProgram::F16 => &self.proto_segmentation_program,
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto program {other:?} is not a layered-float program"
                )));
            }
        };
        gls::use_program(program.id);
        match plan.count_uniform {
            CountUniform::NumProtos => program.load_uniform_1i(c"num_protos", num_protos as i32)?,
            CountUniform::NumLayers => program.load_uniform_1i(c"num_layers", num_layers as i32)?,
        }
        self.render_proto_detection_quads(
            program,
            detect,
            mask_coefficients,
            num_protos,
            color_mode,
        )
    }

    fn render_segmentation(
        &mut self,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        if segmentation.is_empty() {
            return Ok(());
        }

        let is_modelpack = segmentation[0].segmentation.shape()[2] > 1;
        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        if is_modelpack {
            let seg = &segmentation[0];
            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(seg.xmin),
                top: cvt_screen_coord(seg.ymax),
                right: cvt_screen_coord(seg.xmax),
                bottom: cvt_screen_coord(seg.ymin),
            };
            let segment = seg.segmentation.as_standard_layout();
            let slice = segment.as_slice().ok_or(Error::Internal(
                "Cannot get slice of segmentation".to_owned(),
            ))?;

            self.render_modelpack_segmentation(
                dst_roi,
                slice,
                [
                    seg.segmentation.shape()[0],
                    seg.segmentation.shape()[1],
                    seg.segmentation.shape()[2],
                ],
            )?;
        } else {
            // Hoist program bind, texture bind, and texture parameter
            // configuration out of the per-detection loop. These never
            // change between masks, so a single setup call replaces
            // ~5×N redundant GL state calls.
            self.setup_yolo_segmentation_pass();

            for (idx, (seg, det)) in segmentation.iter().zip(detect).enumerate() {
                let color_index = color_mode.index(idx, det.label);
                let dst_roi = RegionOfInterest {
                    left: cvt_screen_coord(seg.xmin),
                    top: cvt_screen_coord(seg.ymax),
                    right: cvt_screen_coord(seg.xmax),
                    bottom: cvt_screen_coord(seg.ymin),
                };

                let segment = seg.segmentation.as_standard_layout();
                let slice = segment.as_slice().ok_or(Error::Internal(
                    "Cannot get slice of segmentation".to_owned(),
                ))?;

                self.render_yolo_segmentation(
                    dst_roi,
                    slice,
                    [seg.segmentation.shape()[0], seg.segmentation.shape()[1]],
                    color_index,
                )?;
            }
        }

        gls::disable(gls::gl::BLEND);
        Ok(())
    }

    /// Compile a GLES 3.1 compute shader program from source.
    fn compile_compute_program(source: &str) -> Result<u32, Error> {
        unsafe {
            let cs = gls::gl::CreateShader(gls::gl::COMPUTE_SHADER);
            if super::shaders::compile_shader_from_str(cs, source, "proto_repack_compute").is_err()
            {
                gls::gl::DeleteShader(cs);
                return Err(Error::OpenGl("compute shader compile failed".into()));
            }

            let program = gls::gl::CreateProgram();
            gls::gl::AttachShader(program, cs);
            gls::gl::LinkProgram(program);

            let mut linked: i32 = 0;
            gls::gl::GetProgramiv(program, gls::gl::LINK_STATUS, &mut linked);
            gls::gl::DeleteShader(cs);

            if linked == 0 {
                let mut log_len = 0;
                gls::gl::GetProgramiv(program, gls::gl::INFO_LOG_LENGTH, &mut log_len);
                let mut log_buf: Vec<u8> = vec![0; log_len as usize];
                gls::gl::GetProgramInfoLog(
                    program,
                    log_len,
                    std::ptr::null_mut(),
                    log_buf.as_mut_ptr() as *mut std::ffi::c_char,
                );
                let msg = String::from_utf8_lossy(&log_buf);
                gls::gl::DeleteProgram(program);
                return Err(Error::OpenGl(format!("compute program link failed: {msg}")));
            }
            Ok(program)
        }
    }

    /// Ensure the shared proto `GL_TEXTURE_2D_ARRAY` uses immutable storage and
    /// is recreated when dimensions or internal format change.
    #[allow(clippy::too_many_arguments)]
    /// Ensure `proto_texture` is an immutable-storage (`TexStorage3D`) array
    /// of exactly `(w, h, layers, internal_fmt)`, recreating the texture
    /// object on any change, and bind it on `TEXTURE0`. Returns `true` when
    /// recreated (sampler params reset with the object and must be
    /// re-applied by the caller).
    ///
    /// Immutable storage is what makes the compute path's
    /// `BindImageTexture` legal — ES 3.1 requires it, and `imageStore` into
    /// a `TexImage3D`-allocated texture silently writes nowhere on
    /// conformant drivers (Vivante and Mali both rendered garbage masks).
    /// Recreate-on-change is what keeps the `(dims, internal_fmt)` gate
    /// honest when renders alternate proto dtypes on one processor: the
    /// previous `TexImage3D` scheme left `proto_tex_dims` stale after a
    /// float upload re-allocated the shared texture, so the next int8
    /// upload `TexSubImage3D`'d into an R32F allocation → GL_INVALID_VALUE
    /// (0x501). Both found by the proto churn/compute regression tests.
    fn ensure_proto_texture(
        &mut self,
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
    ) -> bool {
        Self::ensure_immutable_tex_array(
            &mut self.proto_texture,
            &mut self.proto_tex_dims,
            target,
            internal_fmt,
            w,
            h,
            layers,
        )
    }

    /// The shared immutable-array allocation gate behind
    /// [`Self::ensure_proto_texture`] — also used by the two-pass dequant
    /// target, which keeps its own texture/dims pair.
    fn ensure_immutable_tex_array(
        texture: &mut Texture,
        tex_dims: &mut (usize, usize, usize, u32),
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
    ) -> bool {
        let dims = (w, h, layers, internal_fmt);
        gls::active_texture(gls::gl::TEXTURE0);
        if dims == *tex_dims {
            gls::bind_texture(target, texture.id);
            return false;
        }
        // Immutable storage cannot be re-specified: recreate the object
        // (the old one is deleted by Texture's Drop).
        *texture = Texture::new();
        gls::bind_texture(target, texture.id);
        unsafe {
            gls::gl::TexStorage3D(target, 1, internal_fmt, w as i32, h as i32, layers as i32);
        }
        *tex_dims = dims;
        true
    }

    /// Apply the proto texture sampler params (`filter` + CLAMP_TO_EDGE).
    /// Needed (only) after [`Self::ensure_proto_texture`] recreates the
    /// object; params are per-texture state.
    fn set_proto_tex_params(target: u32, filter: u32) {
        unsafe { super::core::set_tex_filter(target, filter) };
        gls::tex_parameteri(
            target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_proto_texture<T: Copy>(
        &mut self,
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
        format: u32,
        dtype: u32,
        filter: u32,
        data: &[T],
    ) {
        if self.ensure_proto_texture(target, internal_fmt, w, h, layers) {
            Self::set_proto_tex_params(target, filter);
        }
        unsafe {
            gls::gl::TexSubImage3D(
                target,
                0,
                0,
                0,
                0,
                w as i32,
                h as i32,
                layers as i32,
                format,
                dtype,
                data.as_ptr() as *const std::ffi::c_void,
            );
        }
    }

    /// Set the `opacity` uniform on all segmentation and color shader programs.
    /// Skips GL calls entirely when opacity hasn't changed since the last call.
    fn set_opacity_uniform(&mut self, opacity: f32) -> Result<(), Error> {
        if (opacity - self.cached_opacity).abs() < f32::EPSILON {
            return Ok(());
        }
        for program in [
            &self.color_program,
            &self.segmentation_program,
            &self.instanced_segmentation_program,
            &self.proto_segmentation_program,
            &self.proto_segmentation_int8_nearest_program,
            &self.proto_segmentation_int8_bilinear_program,
            &self.proto_segmentation_f32_program,
        ] {
            program.load_uniform_1f(c"opacity", opacity)?;
        }
        self.cached_opacity = opacity;
        Ok(())
    }

    fn render_box(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        detect: &[DetectBox],
        color_mode: crate::ColorMode,
    ) -> Result<(), Error> {
        unsafe {
            gls::gl::UseProgram(self.color_program.id);
            let rescale = |x: f32| x * 2.0 - 1.0;
            let thickness = 3.0;
            for (idx, d) in detect.iter().enumerate() {
                let color_index = color_mode.index(idx, d.label);
                self.color_program
                    .load_uniform_1i(c"class_index", color_index as i32)?;
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let bbox: [f32; 4] = d.bbox.into();
                let outer_box = [
                    bbox[0] - thickness / dst_w as f32,
                    bbox[1] - thickness / dst_h as f32,
                    bbox[2] + thickness / dst_w as f32,
                    bbox[3] + thickness / dst_h as f32,
                ];
                let camera_vertices: [f32; 24] = [
                    rescale(bbox[0]),
                    rescale(bbox[3]),
                    0., // bottom left
                    rescale(bbox[2]),
                    rescale(bbox[3]),
                    0., // bottom right
                    rescale(bbox[2]),
                    rescale(bbox[1]),
                    0., // top right
                    rescale(bbox[0]),
                    rescale(bbox[1]),
                    0., // top left
                    rescale(outer_box[0]),
                    rescale(outer_box[3]),
                    0., // bottom left
                    rescale(outer_box[2]),
                    rescale(outer_box[3]),
                    0., // bottom right
                    rescale(outer_box[2]),
                    rescale(outer_box[1]),
                    0., // top right
                    rescale(outer_box[0]),
                    rescale(outer_box[1]),
                    0., // top left
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                let vertices_index: [u32; 10] = [0, 1, 5, 2, 6, 3, 7, 0, 4, 5];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_STRIP,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Report the float render support that this processor instance should
    /// advertise.  Delegates to [`float_render_support`].
    /// Whether the GPU is a Verisilicon/Vivante core (GL_RENDERER). Used by
    /// tests to gate properties that only hold on Vivante's NV12 path (e.g. the
    /// contiguous-vs-multiplane equality, which splits Path A/B off Vivante).
    pub(super) fn is_vivante(&self) -> bool {
        self.is_vivante
    }

    pub(super) fn supported_render_dtypes(&self) -> crate::RenderDtypeSupport {
        float_render_support(
            self.is_vivante,
            self.supports_f32_color,
            self.supports_f16_color,
        )
    }

    /// Assemble the immutable capability surface for this processor.
    /// Captured ONCE by the dispatch wrapper at worker startup (before its
    /// message loop) — `serialize_gl` is the Vivante/galcore process-wide
    /// serialization requirement; see `PlatformCaps` in `platform/mod.rs`.
    pub(super) fn platform_caps(&self) -> super::platform::PlatformCaps {
        super::platform::PlatformCaps {
            transfer_backend: self.gl_context.transfer_backend,
            render_dtypes: self.supported_render_dtypes(),
            serialize_gl: self.is_vivante(),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::should_reject_software_gl;

    // The override env reader (`software_gl_override_enabled`) is a thin
    // `var_os == "1"` wrapper; it is exercised end-to-end by the GL-init path
    // on the Mesa-llvmpipe coverage lane. The decision logic below is the part
    // worth pinning in a pure unit test.

    #[test]
    fn software_gl_rejected_by_default() {
        // Software renderer, no override → reject (caller falls back to CPU).
        assert!(should_reject_software_gl(true, false));
    }

    #[test]
    fn software_gl_allowed_with_override() {
        // Software renderer + override → do not reject (CI coverage path).
        assert!(!should_reject_software_gl(true, true));
    }

    #[test]
    fn hardware_gl_never_rejected() {
        // A hardware renderer is never rejected, override or not.
        assert!(!should_reject_software_gl(false, false));
        assert!(!should_reject_software_gl(false, true));
    }
}
