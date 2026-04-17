// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_decoder::{DetectBox, ProtoData, ProtoTensor, Segmentation};
use edgefirst_tensor::{
    PixelFormat, PixelLayout, Tensor, TensorMapTrait, TensorMemory, TensorTrait,
};
use gbm::drm::buffer::DrmFourcc;
use khronos_egl::{self as egl, Attrib};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr};
use std::os::fd::AsRawFd;
use std::ptr::null_mut;
use std::rc::Rc;
use std::time::Instant;

use super::cache::CachedEglImage;
use super::EglDisplayKind;

use super::cache::{CacheKind, EglCacheKey, EglImageCache};
use super::context::{egl_ext, GlContext};
use super::resources::{Buffer, EglImage, FrameBuffer, GlProgram, Texture};
use super::shaders::*;
use super::{Int8InterpolationMode, RegionOfInterest, TransferBackend};
use crate::{Crop, Error, Flip, ImageProcessorTrait, Rect, Rotation, DEFAULT_COLORS};
use edgefirst_tensor::TensorDyn;

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
    /// Interpolation mode for int8 proto textures.
    int8_interpolation_mode: Int8InterpolationMode,
    /// Intermediate FBO texture for two-pass int8 dequant path.
    proto_dequant_texture: Texture,
    /// Compute shader program for HWC→CHW proto repack (GLES 3.1 only).
    proto_repack_compute_program: Option<u32>,
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
    is_vivante: bool,
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
        crop.check_crop_dyn(src, dst)?;
        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        let (dst_u8, dst_fmt, is_int8) = dyn_to_u8_dst(dst)?;

        self.convert_impl(
            src_u8, src_fmt, dst_u8, dst_fmt, is_int8, rotation, flip, crop,
        )
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

impl GLProcessorST {
    pub fn new(kind: Option<EglDisplayKind>) -> Result<GLProcessorST, crate::Error> {
        let gl_context = GlContext::new(kind)?;
        gls::load_with(|s| {
            gl_context
                .egl
                .get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });

        let (has_float_linear, has_bgra, is_vivante, is_software_renderer) =
            Self::gl_check_support()?;

        // Software renderers (llvmpipe, softpipe, swrast) are CPU-based OpenGL
        // implementations that are slower and less capable than our native CPU
        // backend. Reject them early — the caller falls back to CPU automatically.
        if is_software_renderer {
            return Err(crate::Error::NotSupported(
                "software OpenGL renderer detected (llvmpipe/softpipe/swrast); \
                 GL backend disabled — check EGL ICD configuration if a \
                 hardware GPU is expected"
                    .into(),
            ));
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

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let draw_render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let proto_texture = Texture::new();
        let proto_dequant_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

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
            int8_interpolation_mode: Int8InterpolationMode::Bilinear,
            proto_dequant_texture,
            vertex_buffer,
            texture_buffer,
            convert_fbo: FrameBuffer::new(),
            draw_fbo: FrameBuffer::new(),
            draw_render_texture,
            src_egl_cache: EglImageCache::new(8),
            dst_egl_cache: EglImageCache::new(8),
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
            proto_ssbo: 0,
            proto_ssbo_size: 0,
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

        // Run the full DMA-buf EGLImage render pipeline
        if let Err(e) = self.convert_dest_dma(
            &mut dst,
            PixelFormat::Rgba,
            &src,
            PixelFormat::Rgba,
            false,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        ) {
            log::info!("verify_dma_buf_roundtrip: convert_dest_dma failed: {e}");
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
        crop: Crop,
    ) -> crate::Result<()> {
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

        if self.gl_context.transfer_backend.is_dma() && dst.memory() == TensorMemory::Dma {
            log::trace!("GL path: DMA (zero-copy EGLImage)");
            let res =
                self.convert_dest_dma(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop);
            log::trace!("GL DMA result: {}", if res.is_ok() { "ok" } else { "err" });
            return res;
        }
        if src.memory() == TensorMemory::Pbo && dst.memory() == TensorMemory::Pbo {
            log::trace!("GL path: PBO-to-PBO");
            return self
                .convert_pbo_to_pbo(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop);
        }
        if dst.memory() == TensorMemory::Pbo {
            log::trace!("GL path: any-to-PBO (src={:?})", src.memory());
            return self
                .convert_any_to_pbo(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop);
        }
        if src.memory() == TensorMemory::Pbo {
            log::trace!("GL path: PBO-to-mem");
            return self
                .convert_pbo_to_mem(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop);
        }
        log::trace!("GL path: non-DMA (CPU upload/readback)");
        let start = Instant::now();
        let res =
            self.convert_dest_non_dma(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop);
        log::debug!("convert_dest_non_dma takes {:?}", start.elapsed());
        res
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
                        self.setup_renderbuffer_non_dma(
                            dst,
                            dst_fmt,
                            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                        )?;
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
                self.setup_renderbuffer_non_dma(
                    dst,
                    dst_fmt,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
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
                        self.setup_renderbuffer_non_dma(
                            dst,
                            dst_fmt,
                            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                        )?;
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
                self.setup_renderbuffer_non_dma(
                    dst,
                    dst_fmt,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
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
            // EGLImage supports RGBA, GREY, YUYV, and NV12 for DMA buffers.
            // VYUY excluded: Vivante GPU accepts the DRM fourcc but produces
            // incorrect output (similarity ~0.28 vs reference).
            matches!(
                fmt,
                PixelFormat::Rgba | PixelFormat::Grey | PixelFormat::Yuyv | PixelFormat::Nv12
            )
        } else {
            matches!(
                fmt,
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey
            )
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
    /// Returns `(has_float_linear, has_bgra, is_vivante, is_software_renderer)`.
    fn gl_check_support() -> Result<(bool, bool, bool, bool), crate::Error> {
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

        let has_float_linear = extensions.contains("GL_OES_texture_float_linear");
        log::debug!("GL_OES_texture_float_linear: {has_float_linear}");

        let has_bgra = extensions.contains("GL_EXT_texture_format_BGRA8888");
        log::debug!("GL_EXT_texture_format_BGRA8888: {has_bgra}");

        Ok((has_float_linear, has_bgra, is_vivante, is_software_renderer))
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

        let luma_id = dst.buffer_identity().id();
        let chroma_id = dst.chroma().map(|t| t.buffer_identity().id());
        let dst_key = (luma_id, chroma_id);

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst) {
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
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::LINEAR as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::LINEAR as i32,
                );
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

        unsafe {
            gls::gl::Viewport(0, 0, width, height);
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

        let luma_id = dst.buffer_identity().id();
        let chroma_id = dst.chroma().map(|t| t.buffer_identity().id());
        let dst_key = (luma_id, chroma_id);

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst) {
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
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::LINEAR as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::LINEAR as i32,
                );
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

    #[allow(clippy::too_many_arguments)]
    fn convert_dest_dma(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        assert!(self.gl_context.transfer_backend.is_dma());
        if dst_fmt == PixelFormat::Rgb {
            log::trace!(
                "GL DMA dispatch: {src_fmt}→{dst_fmt} int8={is_int8} → two-pass packed RGB \
                 (RGBA intermediate + packed_rgba8{}shader)",
                if is_int8 { "_int8_" } else { "_" }
            );
            self.convert_to_packed_rgb(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
        } else if dst_fmt.layout() == PixelLayout::Planar {
            if self.is_vivante && src_fmt == PixelFormat::Nv12 {
                // Two-pass workaround: NV12→RGBA intermediate → RGBA→PlanarRgb.
                // Single-pass NV12+planar causes an unrecoverable GPU hang on
                // Vivante GC7000UL (i.MX 8M Plus, galcore 6.4.11).
                log::trace!(
                    "GL DMA dispatch: {src_fmt}→{dst_fmt} int8={is_int8} → two-pass planar \
                     (Vivante NV12 workaround: RGBA intermediate + planar_2d{}shader)",
                    if is_int8 { "_int8_" } else { "_" }
                );
                self.convert_nv12_to_planar_two_pass(
                    src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop,
                )
            } else {
                log::trace!(
                    "GL DMA dispatch: {src_fmt}→{dst_fmt} int8={is_int8} → planar output \
                     (renderbuffer + planar{}shader)",
                    if is_int8 { "_int8_" } else { "_" }
                );
                self.setup_renderbuffer_dma(dst, dst_fmt)?;
                // Bias letterbox clear color for int8 — glClear bypasses the shader.
                let crop = if is_int8 {
                    let mut crop = crop;
                    if let Some(ref mut color) = crop.dst_color {
                        color[0] ^= 0x80;
                        color[1] ^= 0x80;
                        color[2] ^= 0x80;
                    }
                    crop
                } else {
                    crop
                };
                self.convert_to_planar(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
            }
        } else {
            log::trace!(
                "GL DMA dispatch: {src_fmt}→{dst_fmt} int8={is_int8} → single-pass EGLImage \
                 (renderbuffer, {}shader)",
                if is_int8 { "int8 " } else { "standard " }
            );
            self.setup_renderbuffer_dma(dst, dst_fmt)?;
            // For int8 output, swap to int8 shader programs that apply XOR 0x80
            // in the fragment shader — zero CPU readback.
            if is_int8 {
                std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
                std::mem::swap(
                    &mut self.texture_program_yuv,
                    &mut self.texture_int8_program_yuv,
                );
                // Bias the letterbox clear color since glClear bypasses the shader.
                let mut crop = crop;
                if let Some(ref mut color) = crop.dst_color {
                    color[0] ^= 0x80;
                    color[1] ^= 0x80;
                    color[2] ^= 0x80;
                }
                let result = self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop);
                std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
                std::mem::swap(
                    &mut self.texture_program_yuv,
                    &mut self.texture_int8_program_yuv,
                );
                result
            } else {
                self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_dest_non_dma(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;

        // Bias letterbox clear color for int8 — glClear bypasses the shader.
        let crop = if is_int8 {
            let mut crop = crop;
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
            crop
        } else {
            crop
        };

        // For int8 non-planar output, swap to int8 shader programs.
        if is_int8 && dst_fmt.layout() != PixelLayout::Planar {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        let start = Instant::now();
        let render_result = if dst_fmt.layout() == PixelLayout::Planar {
            self.convert_to_planar(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
        } else {
            self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop)
        };

        if is_int8 && dst_fmt.layout() != PixelLayout::Planar {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }
        render_result?;
        log::debug!("Draw to framebuffer takes {:?}", start.elapsed());

        // ReadnPixels into Mem dst — data is already int8-biased by the shader.
        let start = Instant::now();
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => unreachable!(),
        };

        unsafe {
            let mut dst_map = dst.map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst_w as i32,
                dst_h as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            if dst_fmt == PixelFormat::Bgra {
                for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
            }
        }
        check_gl_error(function!(), line!())?;
        log::debug!("Read from framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    fn setup_renderbuffer_non_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        crop: Crop,
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
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

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

    fn setup_renderbuffer_from_pbo(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        buffer_id: u32,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let (width, height) = (dst_w as i32, dst_h as i32);
        let format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO renderbuffer not supported for {dst_fmt}",
                )))
            }
        };
        self.convert_fbo.bind();
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

            // Upload existing PBO content to the render texture.
            // Binding PBO as UNPACK buffer makes TexImage2D read from it.
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, buffer_id);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                format as i32,
                width,
                height,
                0,
                format,
                gls::gl::UNSIGNED_BYTE,
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

    /// Convert between two PBO-backed images.
    ///
    /// Source PBO is bound as `GL_PIXEL_UNPACK_BUFFER` for zero-copy texture upload
    /// (avoids `tensor.map()` to prevent GL-thread deadlocks). Destination uses
    /// `GL_PIXEL_PACK_BUFFER` for zero-copy readback into the PBO.
    #[allow(clippy::too_many_arguments)]
    fn convert_pbo_to_pbo(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let (src_buffer_id, dst_buffer_id) = {
            let src_pbo = src.as_pbo().ok_or_else(|| {
                crate::Error::OpenGl("convert_pbo_to_pbo: src is not a PBO tensor".to_string())
            })?;
            let dst_pbo = dst.as_pbo().ok_or_else(|| {
                crate::Error::OpenGl("convert_pbo_to_pbo: dst is not a PBO tensor".to_string())
            })?;

            if src_pbo.is_mapped() || dst_pbo.is_mapped() {
                return Err(crate::Error::OpenGl(
                    "Cannot convert PBO tensors while they are mapped".to_string(),
                ));
            }

            (src_pbo.buffer_id(), dst_pbo.buffer_id())
        };

        self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;

        // For int8 output, swap to int8 shader programs so the GPU applies
        // XOR 0x80 in the fragment shader — no CPU readback needed for int8.
        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        // Bias letterbox clear color for int8 — glClear bypasses the shader.
        let crop = if is_int8 {
            let mut crop = crop;
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
            crop
        } else {
            crop
        };

        // Upload source from PBO and render.
        // We cannot call convert_to/draw_src_texture directly because they
        // call src.tensor().map() which sends a message back to THIS thread,
        // causing a deadlock. Instead, bind the source PBO as UNPACK buffer
        // and upload to the texture with a NULL pointer — GL reads directly
        // from the PBO, zero CPU copy.
        let start = Instant::now();
        let render_result = self.draw_src_texture_from_pbo(
            src,
            src_fmt,
            src_buffer_id,
            dst,
            dst_fmt,
            rotation,
            flip,
            crop,
        );
        // Swap shaders back before checking result
        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }
        render_result?;
        log::debug!("PBO render takes {:?}", start.elapsed());

        let start_read = Instant::now();
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {dst_fmt}",
                )))
            }
        };
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst_w as i32,
                dst_h as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.len() as i32,
                std::ptr::null_mut(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }

        check_gl_error(function!(), line!())?;

        // Handle BGRA R↔B swap if needed (must map PBO on the GL thread).
        // Int8 XOR 0x80 is handled in the fragment shader — no CPU map needed.
        if dst_fmt == PixelFormat::Bgra {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if ptr.is_null() {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    return Err(crate::Error::OpenGl(
                        "glMapBufferRange returned null for BGRA byte-swap".to_string(),
                    ));
                }
                let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.len());
                for chunk in slice.chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
                gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            }
            check_gl_error(function!(), line!())?;
        }

        log::debug!("PBO readback takes {:?}", start_read.elapsed());
        Ok(())
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
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
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
            RegionOfInterest {
                left: crop.left as f32 / src_w as f32,
                top: (crop.top + crop.height) as f32 / src_h as f32,
                right: (crop.left + crop.width) as f32 / src_w as f32,
                bottom: crop.top as f32 / src_h as f32,
            }
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

            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
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

            // Bind source PBO as UNPACK buffer — glTexImage2D reads from it
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, src_buffer_id);
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

    /// Convert any source (Mem/DMA) to a PBO destination.
    /// Source is uploaded via normal texture path (maps tensor for CPU upload).
    /// Destination readback uses PBO PACK binding (no map on GL thread).
    #[allow(clippy::too_many_arguments)]
    fn convert_any_to_pbo(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_pbo = dst.as_pbo().ok_or_else(|| {
            crate::Error::OpenGl("convert_any_to_pbo: dst is not a PBO tensor".to_string())
        })?;
        if dst_pbo.is_mapped() {
            return Err(crate::Error::OpenGl(
                "Cannot convert to a mapped PBO tensor".to_string(),
            ));
        }
        let dst_buffer_id = dst_pbo.buffer_id();

        self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;

        // For int8 non-planar output, swap to int8 shader programs.
        // Planar path handles int8 internally via its own int8 shader.
        if is_int8 && dst_fmt.layout() != PixelLayout::Planar {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        let start = Instant::now();
        let render_result = if dst_fmt.layout() == PixelLayout::Planar {
            self.convert_to_planar(src, src_fmt, dst, dst_fmt, is_int8, rotation, flip, crop)
        } else {
            self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop)
        };

        if is_int8 && dst_fmt.layout() != PixelLayout::Planar {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }
        render_result?;
        log::debug!("any-to-PBO render takes {:?}", start.elapsed());

        let start_read = Instant::now();
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {dst_fmt}",
                )))
            }
        };
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst_w as i32,
                dst_h as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.len() as i32,
                std::ptr::null_mut(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }
        check_gl_error(function!(), line!())?;

        // Handle BGRA R↔B swap if needed (must map PBO on the GL thread).
        // Int8 XOR 0x80 is handled in the fragment shader — no CPU map needed.
        if dst_fmt == PixelFormat::Bgra {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if ptr.is_null() {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    return Err(crate::Error::OpenGl(
                        "glMapBufferRange returned null for BGRA byte-swap".to_string(),
                    ));
                }
                let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.len());
                for chunk in slice.chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
                gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            }
            check_gl_error(function!(), line!())?;
        }

        log::debug!("any-to-PBO readback takes {:?}", start_read.elapsed());
        Ok(())
    }

    /// Convert a PBO source to a non-PBO (Mem) destination.
    /// Source is uploaded via PBO UNPACK binding (no map on GL thread).
    /// Destination readback uses normal ReadnPixels into mapped Mem tensor.
    #[allow(clippy::too_many_arguments)]
    fn convert_pbo_to_mem(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let src_pbo = src.as_pbo().ok_or_else(|| {
            crate::Error::OpenGl("convert_pbo_to_mem: src is not a PBO tensor".to_string())
        })?;
        if src_pbo.is_mapped() {
            return Err(crate::Error::OpenGl(
                "Cannot convert from a mapped PBO tensor".to_string(),
            ));
        }
        let src_buffer_id = src_pbo.buffer_id();

        self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;

        // For int8 output, swap to int8 shader programs so the GPU renders
        // XOR'd values — ReadnPixels then reads already-biased data.
        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        let start = Instant::now();
        let render_result = self.draw_src_texture_from_pbo(
            src,
            src_fmt,
            src_buffer_id,
            dst,
            dst_fmt,
            rotation,
            flip,
            crop,
        );

        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }
        render_result?;
        log::debug!("PBO-to-mem render takes {:?}", start.elapsed());

        // ReadnPixels into Mem dst — data is already int8-biased by the shader.
        let start = Instant::now();
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => gls::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => gls::gl::RGBA,
            PixelFormat::Grey => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {dst_fmt}",
                )))
            }
        };
        unsafe {
            let mut dst_map = dst.map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst_w as i32,
                dst_h as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            if dst_fmt == PixelFormat::Bgra {
                for chunk in dst_map.as_mut_slice().chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }
            }
        }
        check_gl_error(function!(), line!())?;
        log::debug!("PBO-to-mem readback takes {:?}", start.elapsed());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_to(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &Tensor<u8>,
        _dst_fmt: PixelFormat,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        check_gl_error(function!(), line!())?;

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
            RegionOfInterest {
                left: crop.left as f32 / src_w as f32,
                top: (crop.top + crop.height) as f32 / src_h as f32,
                right: (crop.left + crop.width) as f32 / src_w as f32,
                bottom: crop.top as f32 / src_h as f32,
            }
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
            match self.get_or_create_egl_image(CacheKind::Src, src, src_fmt) {
                Ok(src_egl) => self.draw_camera_texture_eglimage(
                    src,
                    src_fmt,
                    src_egl,
                    src_roi,
                    dst_roi,
                    rotation_offset,
                    flip,
                )?,
                Err(e) => {
                    let src_w = src.width().unwrap_or(0);
                    let src_h = src.height().unwrap_or(0);
                    log::warn!(
                        "EGL image creation failed for {src_fmt} ({src_w}x{src_h}), \
                         falling back to texture upload (slower): {e}"
                    );
                    let start = Instant::now();
                    self.draw_src_texture(src, src_fmt, src_roi, dst_roi, rotation_offset, flip)?;
                    log::debug!("draw_src_texture takes {:?}", start.elapsed());
                }
            }
        } else {
            let start = Instant::now();
            self.draw_src_texture(src, src_fmt, src_roi, dst_roi, rotation_offset, flip)?;
            log::debug!("draw_src_texture takes {:?}", start.elapsed());
        }

        let start = Instant::now();
        unsafe { gls::gl::Finish() };
        log::debug!("gl_Finish takes {:?}", start.elapsed());
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
        crop: Crop,
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
            RegionOfInterest {
                left: crop.left as f32 / src_w as f32,
                top: (crop.top + crop.height) as f32 / src_h as f32,
                right: (crop.left + crop.width) as f32 / src_w as f32,
                bottom: crop.top as f32 / src_h as f32,
            }
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

        let src_key = (
            src.buffer_identity().id(),
            src.chroma().map(|t| t.buffer_identity().id()),
        );
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
        crop: Crop,
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
        self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop)?;

        // --- Pass 2: Pack intermediate RGBA → RGB DMA destination ---
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
            match self.cached_dst_renderbuffer(dst) {
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
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::NEAREST as i32,
                    );
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::NEAREST as i32,
                    );
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_egl.as_ptr());
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
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
        }

        // Set uniform: tex = TEXTURE1 (intermediate RGBA texture)
        unsafe {
            let loc_tex = gls::gl::GetUniformLocation(program.id, c"tex".as_ptr());
            gls::gl::Uniform1i(loc_tex, 1);
        }

        // Draw full-viewport quad to pack RGBA→RGB
        self.draw_fullscreen_quad()?;

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
    fn convert_nv12_to_planar_two_pass(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
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
            "convert_nv12_to_planar_two_pass: {src_fmt}→{dst_fmt} {dst_w}x{dst_h} \
             int8={is_int8} (Vivante two-pass workaround)",
        );

        // --- Pass 1: NV12→RGBA into intermediate texture ---
        // No int8 bias here — bias is applied in pass 2's planar shader.
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
        self.convert_to(src, src_fmt, dst, dst_fmt, rotation, flip, crop)?;

        // --- Pass 2: RGBA→PlanarRgb to DMA destination ---
        // setup_renderbuffer_dma rebinds convert_fbo with the DMA destination EGLImage,
        // replacing packed_rgb_fbo that was active during pass 1. It also sets the viewport
        // to (dst_w, dst_h * 3) for the tall R8 planar renderbuffer.
        self.setup_renderbuffer_dma(dst, dst_fmt)?;

        // Bias letterbox clear color for int8 — glClear bypasses the shader.
        let crop = if is_int8 {
            let mut crop = crop;
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
            crop
        } else {
            crop
        };

        // Letterbox fill: clear planar regions outside the destination rect
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(rect) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(rect.left as f32 / dst_w as f32),
                top: cvt_screen_coord((rect.top + rect.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((rect.left + rect.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(rect.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
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
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
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
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
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
                    src_key.0
                );
            } else {
                log::trace!(
                    "draw_camera_planar: reusing bound src EGLImage id={:#x}",
                    src_key.0
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
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
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

            // Set tex uniform to unit 0
            let loc_tex = gls::gl::GetUniformLocation(program.id, c"tex".as_ptr());
            gls::gl::Uniform1i(loc_tex, 0);

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
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    fn draw_src_texture(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
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
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
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
            self.camera_normal_texture.update_texture(
                texture_target,
                src_w,
                src_h,
                texture_format,
                &src.map()?,
            );

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
    ) -> Result<(), Error> {
        let luma_id = src.buffer_identity().id();
        let chroma_id = src.chroma().map(|t| t.buffer_identity().id());
        let src_key = (luma_id, chroma_id);

        let texture_target = gls::gl::TEXTURE_EXTERNAL_OES;
        unsafe {
            gls::gl::UseProgram(self.texture_program_yuv.id);
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

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

    fn create_image_from_dma2(
        &self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
    ) -> Result<EglImage, crate::Error> {
        let attrs = super::dma_import::DmaImportAttrs::from_tensor(src, src_fmt)?;
        let egl_img_attr = attrs.to_egl_attribs();
        self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    fn new_egl_image_owned(
        &'_ self,
        target: egl::Enum,
        attrib_list: &[Attrib],
    ) -> Result<EglImage, Error> {
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
        let luma_id = img.buffer_identity().id();
        let chroma_id = img.chroma().map(|t| t.buffer_identity().id());
        let id = (luma_id, chroma_id);

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
        // want to have destroyed a valid cache entry for nothing.
        let egl_image_obj = self.create_image_from_dma2(img, img_fmt)?;

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
    fn cached_dst_renderbuffer(&self, img: &Tensor<u8>) -> Option<u32> {
        let luma_id = img.buffer_identity().id();
        let chroma_id = img.chroma().map(|t| t.buffer_identity().id());
        let id = (luma_id, chroma_id);
        self.dst_egl_cache
            .entries
            .get(&id)
            .and_then(|entry| entry.renderbuffer)
    }

    /// Create an EGLImage from a DMA buffer with explicitly specified internal
    /// dimensions and format. Used when the GL render surface differs from the
    /// logical image dimensions (e.g., packed RGB reinterpretation).
    fn create_egl_image_with_dims(
        &self,
        img: &Tensor<u8>,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<EglImage, crate::Error> {
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

    /// Get or create an EGLImage for a packed RGB DMA destination with
    /// reinterpreted dimensions. Uses the dst cache keyed by buffer identity.
    fn get_or_create_egl_image_rgb(
        &mut self,
        img: &Tensor<u8>,
        _img_fmt: PixelFormat,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<egl::Image, crate::Error> {
        let id = (img.buffer_identity().id(), None);
        if self.dst_egl_cache.sweep() {
            self.invalidate_dst_textures();
        }

        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            self.dst_egl_cache.hits += 1;
            cached.last_used = ts;
            log::trace!("EglImageCache dst (RGB) hit: id={:#x}", id.0);
            return Ok(cached.egl_image.egl_image);
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (RGB) miss: id={:#x}", id.0);
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
    fn repack_protos_to_rgba_f16(protos: &ndarray::Array3<f32>) -> (Vec<u8>, usize) {
        let (height, width, num_protos) = protos.dim();
        let num_layers = num_protos.div_ceil(4);
        // Each layer is H*W*4 half-floats, each half-float is 2 bytes
        let layer_stride = height * width * 4;
        let mut buf = vec![0u16; layer_stride * num_layers];

        for y in 0..height {
            for x in 0..width {
                for k in 0..num_layers * 4 {
                    let val = if k < num_protos {
                        half::f16::from_f32(protos[[y, x, k]])
                    } else {
                        half::f16::ZERO
                    };
                    let layer = k / 4;
                    let channel = k % 4;
                    buf[layer * layer_stride + y * width * 4 + x * 4 + channel] = val.to_bits();
                }
            }
        }

        // Reinterpret u16 buffer as bytes
        let byte_buf = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 2).to_vec()
        };
        (byte_buf, num_layers)
    }

    /// Render YOLO proto segmentation masks using the fused GPU pipeline.
    ///
    /// Dispatches to the appropriate shader based on `ProtoTensor` variant:
    /// - `Quantized`: uploads raw int8 as `GL_R8I`, dequantizes in shader
    /// - `Float`: uploads as `GL_R32F` with hardware bilinear (if available),
    ///   or falls back to f16 repack path
    fn render_proto_segmentation(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(());
        }

        let (height, width, num_protos) = proto_data.protos.dim();
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;

        match &proto_data.protos {
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                self.render_proto_segmentation_int8(
                    detect,
                    &proto_data.mask_coefficients,
                    protos,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            ProtoTensor::Float(protos_f32) => {
                if self.has_float_linear {
                    self.render_proto_segmentation_f32(
                        detect,
                        &proto_data.mask_coefficients,
                        protos_f32,
                        height,
                        width,
                        num_protos,
                        texture_target,
                        color_mode,
                    )?;
                } else {
                    // Fallback: repack to RGBA16F and use existing f16 shader
                    self.render_proto_segmentation_f16(
                        detect,
                        &proto_data.mask_coefficients,
                        protos_f32,
                        height,
                        width,
                        num_protos,
                        texture_target,
                        color_mode,
                    )?;
                }
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
        mask_coefficients: &[Vec<f32>],
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;

        for (idx, (det, coeff)) in detect.iter().zip(mask_coefficients.iter()).enumerate() {
            let color_index = color_mode.index(idx, det.label);
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (i, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[i / 4][i % 4] = *val;
            }

            program.load_uniform_4fv(c"mask_coeff", &packed_coeff)?;
            program.load_uniform_1i(c"class_index", color_index as i32)?;

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

    /// Int8 proto path: upload raw i8 protos as `GL_R8I`, dispatch by
    /// interpolation mode.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_int8(
        &mut self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos: &ndarray::Array3<i8>,
        quantization: &edgefirst_decoder::Quantization,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        // Upload raw int8 protos as R8I texture array (1 proto per layer)
        gls::bind_texture(texture_target, self.proto_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::NEAREST as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::NEAREST as i32,
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

        // Protos are (H, W, num_protos) in row-major HWC. The GL texture
        // needs layer-first CHW (one proto per layer).
        if let Some(compute_program) = self.proto_repack_compute_program {
            // === GLES 3.1 compute shader path ===
            // Upload HWC data as-is to SSBO, let GPU transpose via compute.
            // Fall through to CPU repack if proto array is non-contiguous.
            let data = match protos.as_slice() {
                Some(s) => std::borrow::Cow::Borrowed(s),
                None => std::borrow::Cow::Owned(protos.iter().copied().collect()),
            };
            // Pad to 4-byte boundary for SSBO int[] alignment
            let data_bytes = (data.len() + 3) & !3;

            // Allocate texture as R32I (required for imageStore in compute).
            // Fragment shaders using isampler2DArray read integers correctly
            // from either R8I or R32I via texelFetch.
            let dims = (width, height, num_protos, gls::gl::R32I);
            if dims != self.proto_tex_dims {
                gls::tex_image3d::<i32>(
                    texture_target,
                    0,
                    gls::gl::R32I as i32,
                    width as i32,
                    height as i32,
                    num_protos as i32,
                    0,
                    gls::gl::RED_INTEGER,
                    gls::gl::INT,
                    None,
                );
                self.proto_tex_dims = dims;
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

                // Dispatch compute
                gls::gl::UseProgram(compute_program);
                let loc_w = gls::gl::GetUniformLocation(compute_program, c"width".as_ptr());
                let loc_h = gls::gl::GetUniformLocation(compute_program, c"height".as_ptr());
                let loc_np = gls::gl::GetUniformLocation(compute_program, c"num_protos".as_ptr());
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
            let mut tex_data = vec![0i8; height * width * num_protos];
            for k in 0..num_protos {
                for y in 0..height {
                    for x in 0..width {
                        tex_data[k * height * width + y * width + x] = protos[[y, x, k]];
                    }
                }
            }

            self.upload_proto_texture(
                texture_target,
                gls::gl::R8I,
                width,
                height,
                num_protos,
                gls::gl::RED_INTEGER,
                gls::gl::BYTE,
                &tex_data,
            );
        }

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        match self.int8_interpolation_mode {
            Int8InterpolationMode::Nearest => {
                let program = &self.proto_segmentation_int8_nearest_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients, color_mode)?;
            }
            Int8InterpolationMode::Bilinear => {
                let program = &self.proto_segmentation_int8_bilinear_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients, color_mode)?;
            }
            Int8InterpolationMode::TwoPass => {
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
        }

        Ok(())
    }

    /// Two-pass int8 path: dequant int8→RGBA16F FBO, then render with
    /// existing f16 shader using GL_LINEAR.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_int8_two_pass(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
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

        // Pass 1: Dequantize int8 → RGBA16F texture via framebuffer
        let dequant_fbo = FrameBuffer::new();
        gls::bind_texture(texture_target, self.proto_dequant_texture.id);
        gls::tex_image3d::<u8>(
            texture_target,
            0,
            gls::gl::RGBA16F as i32,
            width as i32,
            height as i32,
            num_layers as i32,
            0,
            gls::gl::RGBA,
            gls::gl::HALF_FLOAT,
            None,
        );
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
            dequant_fbo.bind();
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

            // Full-screen quad
            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let verts: [f32; 12] = [
                    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 12) as isize,
                    verts.as_ptr() as *const c_void,
                );
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let tc: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    tc.as_ptr() as *const c_void,
                );
                let idx: [u32; 4] = [0, 1, 2, 3];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    4,
                    gls::gl::UNSIGNED_INT,
                    idx.as_ptr() as *const c_void,
                );
            }
        }

        // Drop the dequant FBO (its Drop unbinds to 0) and restore the caller's.
        drop(dequant_fbo);
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
        self.render_proto_detection_quads(program, detect, mask_coefficients, color_mode)?;

        Ok(())
    }

    /// F32 proto path: upload as `GL_R32F` with `GL_LINEAR` filtering.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_f32(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos_f32: &ndarray::Array3<f32>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let program = &self.proto_segmentation_f32_program;
        gls::use_program(program.id);
        gls::bind_texture(texture_target, self.proto_texture.id);
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

        // Repack protos to layer-first layout: (num_protos, H, W)
        let mut tex_data = vec![0.0f32; height * width * num_protos];
        for k in 0..num_protos {
            for y in 0..height {
                for x in 0..width {
                    tex_data[k * height * width + y * width + x] = protos_f32[[y, x, k]];
                }
            }
        }

        gls::tex_image3d(
            texture_target,
            0,
            gls::gl::R32F as i32,
            width as i32,
            height as i32,
            num_protos as i32,
            0,
            gls::gl::RED,
            gls::gl::FLOAT,
            Some(&tex_data),
        );

        program.load_uniform_1i(c"num_protos", num_protos as i32)?;
        self.render_proto_detection_quads(program, detect, mask_coefficients, color_mode)?;

        Ok(())
    }

    /// F16 fallback path: repack f32 protos to RGBA16F and use existing
    /// f16 shader with GL_LINEAR. Used when GL_OES_texture_float_linear
    /// is not available.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_f16(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos_f32: &ndarray::Array3<f32>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let num_layers = num_protos.div_ceil(4);
        let (tex_data, _) = Self::repack_protos_to_rgba_f16(protos_f32);

        let program = &self.proto_segmentation_program;
        gls::use_program(program.id);
        gls::bind_texture(texture_target, self.proto_texture.id);
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
            gls::gl::RGBA16F as i32,
            width as i32,
            height as i32,
            num_layers as i32,
            0,
            gls::gl::RGBA,
            gls::gl::HALF_FLOAT,
            Some(&tex_data),
        );

        program.load_uniform_1i(c"num_layers", num_layers as i32)?;
        self.render_proto_detection_quads(program, detect, mask_coefficients, color_mode)?;

        Ok(())
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

    /// Upload data to the proto `GL_TEXTURE_2D_ARRAY`, using `glTexSubImage3D`
    /// when the texture dimensions haven't changed (avoids driver reallocation).
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
        data: &[T],
    ) {
        let dims = (w, h, layers, internal_fmt);
        if dims == self.proto_tex_dims {
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
        } else {
            gls::tex_image3d(
                target,
                0,
                internal_fmt as i32,
                w as i32,
                h as i32,
                layers as i32,
                0,
                format,
                dtype,
                Some(data),
            );
            self.proto_tex_dims = dims;
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
}
