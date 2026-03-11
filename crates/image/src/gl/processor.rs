// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_decoder::{DetectBox, ProtoData, ProtoTensor, Segmentation};
use edgefirst_tensor::{TensorMemory, TensorTrait};
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

use super::cache::{CacheKind, EglImageCache};
use super::context::{egl_ext, GlContext};
use super::resources::{Buffer, EglImage, FrameBuffer, GlProgram, Texture};
use super::shaders::*;
use super::{Int8InterpolationMode, RegionOfInterest, TransferBackend};
use crate::{
    fourcc_is_int8, fourcc_is_packed_rgb, CPUProcessor, Crop, Error, Flip, ImageProcessorTrait,
    MaskRegion, Rect, Rotation, TensorImage, TensorImageRef, BGRA, DEFAULT_COLORS, GREY, NV12,
    PLANAR_RGB, PLANAR_RGBA, PLANAR_RGB_INT8, RGB, RGBA, RGB_INT8, VYUY, YUYV,
};

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
    /// Whether GL_OES_texture_float_linear is available (allows GL_LINEAR on R32F textures).
    has_float_linear: bool,
    /// Whether GL_EXT_texture_format_BGRA8888 is available (allows BGRA destinations).
    pub(super) has_bgra: bool,
    /// Interpolation mode for int8 proto textures.
    int8_interpolation_mode: Int8InterpolationMode,
    /// Intermediate FBO texture for two-pass int8 dequant path.
    proto_dequant_texture: Texture,
    proto_mask_logit_int8_bilinear_program: GlProgram,
    proto_mask_logit_int8_nearest_program: GlProgram,
    proto_mask_logit_f32_program: GlProgram,
    /// Dedicated FBO for mask rendering.
    mask_fbo: u32,
    /// R8 texture attached to mask_fbo.
    mask_fbo_texture: u32,
    /// Current allocated width of mask FBO texture.
    mask_fbo_width: usize,
    /// Current allocated height of mask FBO texture.
    mask_fbo_height: usize,
    /// PBO buffer ID for atlas readback (0 = not allocated).
    mask_atlas_pbo: u32,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    /// Persistent FBO for the convert() render path.
    /// Created once, reused by re-attaching textures each frame.
    convert_fbo: FrameBuffer,
    /// EGLImage cache for source DMA buffers.
    src_egl_cache: EglImageCache,
    /// EGLImage cache for destination DMA buffers.
    dst_egl_cache: EglImageCache,
    /// Intermediate RGBA texture for two-pass packed RGB conversion.
    /// Pass 1 renders YUYV/NV12→RGBA here; Pass 2 packs RGBA→RGB to DMA dest.
    packed_rgb_intermediate_tex: Texture,
    /// FBO for pass 1 of packed RGB conversion (renders to intermediate texture).
    packed_rgb_fbo: FrameBuffer,
    /// Current allocated size of the intermediate texture (0,0 = unallocated).
    packed_rgb_intermediate_size: (usize, usize),
    texture_program: GlProgram,
    texture_program_yuv: GlProgram,
    texture_program_planar: GlProgram,
    /// Shader: existing planar RGB with int8 bias (XOR 0x80) applied to output.
    texture_program_planar_int8: GlProgram,
    /// Shader: packed RGB -> RGBA8 packing (2D texture source, pass 2).
    packed_rgba8_program_2d: GlProgram,
    /// Shader: packed RGB int8 -> RGBA8 packing with XOR 0x80 (2D texture source, pass 2).
    packed_rgba8_int8_program_2d: GlProgram,
    /// Shader: direct RGB render with int8 XOR 0x80 bias (2D texture source).
    texture_int8_program: GlProgram,
    /// Shader: direct RGB render with int8 XOR 0x80 bias (external OES source).
    texture_int8_program_yuv: GlProgram,
    /// Whether the GPU supports direct RGB rendering via BGR888 renderbuffer.
    pub(crate) support_rgb_direct: bool,
    pub(super) gl_context: GlContext,
}

impl Drop for GLProcessorST {
    fn drop(&mut self) {
        unsafe {
            {
                if self.mask_fbo != 0 {
                    gls::gl::DeleteFramebuffers(1, &self.mask_fbo);
                }
                if self.mask_fbo_texture != 0 {
                    gls::gl::DeleteTextures(1, &self.mask_fbo_texture);
                }
                if self.mask_atlas_pbo != 0 {
                    gls::gl::DeleteBuffers(1, &self.mask_atlas_pbo);
                }
            }
        }
    }
}

impl ImageProcessorTrait for GLProcessorST {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        crop.check_crop(src, dst)?;
        if !Self::check_src_format_supported(self.gl_context.transfer_backend, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !Self::check_dst_format_supported(self.gl_context.transfer_backend, dst, self.has_bgra) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} destination texture",
                dst.fourcc().display()
            )));
        }
        log::debug!(
            "dst tensor: {:?} src tensor :{:?}",
            dst.tensor().memory(),
            src.tensor().memory()
        );
        check_gl_error(function!(), line!())?;
        if self.gl_context.transfer_backend.is_dma() && dst.tensor().memory() == TensorMemory::Dma {
            // Packed RGB is now supported via DMA with buffer reinterpretation
            let res = self.convert_dest_dma(dst, src, rotation, flip, crop);
            return res;
        }
        // PBO-to-PBO: both tensors are PBO-backed, use GL buffer bindings for
        // both upload and readback (zero CPU copy for both directions)
        if src.tensor().memory() == TensorMemory::Pbo && dst.tensor().memory() == TensorMemory::Pbo
        {
            return self.convert_pbo_to_pbo(dst, src, rotation, flip, crop);
        }
        // PBO dst with non-PBO src: use normal texture upload for src (which
        // maps the Mem/DMA tensor), but PBO PACK readback for dst.
        // This avoids the deadlock that would occur if convert_dest_non_dma
        // tried to map() the PBO dst on the GL thread.
        if dst.tensor().memory() == TensorMemory::Pbo {
            return self.convert_any_to_pbo(dst, src, rotation, flip, crop);
        }
        // PBO src with non-PBO dst: the src tensor's map() would deadlock on
        // the GL thread, so use PBO UNPACK upload. Readback goes to Mem dst
        // via normal ReadnPixels into mapped memory.
        if src.tensor().memory() == TensorMemory::Pbo {
            return self.convert_pbo_to_mem(dst, src, rotation, flip, crop);
        }
        let start = Instant::now();
        let res = self.convert_dest_non_dma(dst, src, rotation, flip, crop);
        log::debug!("convert_dest_non_dma takes {:?}", start.elapsed());
        res
    }

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // OpenGL doesn't support PLANAR_RGB output, delegate to CPU
        let mut cpu = CPUProcessor::new();
        cpu.convert_ref(src, dst, rotation, flip, crop)
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> Result<(), crate::Error> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_masks");
        if !matches!(dst.fourcc(), RGBA | BGRA | RGB) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        // Determine memory backend and set up the framebuffer.
        // PBO tensors need special handling: calling tensor.map() on the GL
        // thread would deadlock because PboOps sends a message back to this
        // thread. Instead, bind the PBO as GL_PIXEL_UNPACK_BUFFER so
        // TexImage2D reads directly from GPU memory (zero CPU copy).
        let memory = dst.tensor.memory();
        let pbo_buffer_id = if memory == edgefirst_tensor::TensorMemory::Pbo {
            match &dst.tensor {
                edgefirst_tensor::Tensor::Pbo(p) if !p.is_mapped() => Some(p.buffer_id()),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            edgefirst_tensor::TensorMemory::Dma if self.setup_renderbuffer_dma(dst).is_ok() => true,
            _ if pbo_buffer_id.is_some() => {
                self.setup_renderbuffer_from_pbo(dst, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                // Add dest rect to make sure dst is rendered fully
                self.setup_renderbuffer_non_dma(
                    dst,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
                false
            }
        };

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.render_box(dst, detect)?;
        self.render_segmentation(detect, segmentation)?;

        gls::finish();
        if !is_dma {
            let format = match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                // PBO readback: bind as PACK buffer, ReadnPixels writes
                // directly into PBO memory (zero CPU copy).
                unsafe {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst.width() as i32,
                        dst.height() as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.tensor.len() as i32,
                        std::ptr::null_mut(),
                    );
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    gls::gl::Finish();
                }
                check_gl_error(function!(), line!())?;
            } else {
                let mut dst_map = dst.tensor().map()?;
                unsafe {
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst.width() as i32,
                        dst.height() as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.tensor.len() as i32,
                        dst_map.as_mut_ptr() as *mut c_void,
                    );
                }
            }
        }

        Ok(())
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_masks_proto");
        if !matches!(dst.fourcc(), RGBA | BGRA | RGB) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        // PBO detection — same rationale as draw_masks.
        let memory = dst.tensor.memory();
        let pbo_buffer_id = if memory == edgefirst_tensor::TensorMemory::Pbo {
            match &dst.tensor {
                edgefirst_tensor::Tensor::Pbo(p) if !p.is_mapped() => Some(p.buffer_id()),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            edgefirst_tensor::TensorMemory::Dma if self.setup_renderbuffer_dma(dst).is_ok() => true,
            _ if pbo_buffer_id.is_some() => {
                self.setup_renderbuffer_from_pbo(dst, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                self.setup_renderbuffer_non_dma(
                    dst,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
                false
            }
        };

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.render_box(dst, detect)?;
        self.render_proto_segmentation(detect, proto_data)?;

        gls::finish();
        if !is_dma {
            let format = match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                unsafe {
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst.width() as i32,
                        dst.height() as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.tensor.len() as i32,
                        std::ptr::null_mut(),
                    );
                    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                    gls::gl::Finish();
                }
                check_gl_error(function!(), line!())?;
            } else {
                let mut dst_map = dst.tensor().map()?;
                unsafe {
                    gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                    gls::gl::ReadnPixels(
                        0,
                        0,
                        dst.width() as i32,
                        dst.height() as i32,
                        format,
                        gls::gl::UNSIGNED_BYTE,
                        dst.tensor.len() as i32,
                        dst_map.as_mut_ptr() as *mut c_void,
                    );
                }
            }
        }

        Ok(())
    }

    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        GLProcessorST::decode_masks_atlas(self, detect, &proto_data, output_width, output_height)
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

        let (has_float_linear, has_bgra) = Self::gl_check_support()?;

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

        // Binary logit-threshold mask shaders (atlas path — skip sigmoid)
        let proto_mask_logit_int8_nearest_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_int8_nearest(),
        )?;
        let proto_mask_logit_int8_bilinear_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_int8_bilinear(),
        )?;
        let proto_mask_logit_f32_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_f32(),
        )?;

        // Int8 variant of the existing planar RGB shader (for PLANAR_RGB_INT8 destinations).
        let texture_program_planar_int8 =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_int8_shader())?;

        // RGB packing shaders (2D only — used in pass 2 of two-pass pipeline)
        let packed_rgba8_program_2d =
            GlProgram::new(generate_vertex_shader(), generate_packed_rgba8_shader_2d())?;
        let packed_rgba8_int8_program_2d = GlProgram::new(
            generate_vertex_shader(),
            generate_packed_rgba8_int8_shader_2d(),
        )?;

        // Int8 direct-render shaders (for RGB_INT8 destinations via direct path)
        let texture_int8_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader())?;
        let texture_int8_program_yuv =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader_yuv())?;

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let proto_texture = Texture::new();
        let proto_dequant_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

        let mut converter = GLProcessorST {
            gl_context,
            texture_program,
            texture_program_yuv,
            texture_program_planar,
            texture_program_planar_int8,
            packed_rgba8_program_2d,
            packed_rgba8_int8_program_2d,
            texture_int8_program,
            texture_int8_program_yuv,
            support_rgb_direct: false, // will be probed in Task 3
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
            proto_mask_logit_int8_bilinear_program,
            proto_mask_logit_int8_nearest_program,
            proto_mask_logit_f32_program,
            mask_fbo: 0,
            mask_fbo_texture: 0,
            mask_fbo_width: 0,
            mask_fbo_height: 0,
            mask_atlas_pbo: 0,
            vertex_buffer,
            texture_buffer,
            convert_fbo: FrameBuffer::new(),
            src_egl_cache: EglImageCache::new(8),
            dst_egl_cache: EglImageCache::new(8),
            packed_rgb_intermediate_tex: Texture::new(),
            packed_rgb_fbo: FrameBuffer::new(),
            packed_rgb_intermediate_size: (0, 0),
            render_texture,
            segmentation_program,
            instanced_segmentation_program,
            proto_segmentation_program,
            color_program,
        };
        check_gl_error(function!(), line!())?;

        // Probe GPU capability for direct RGB rendering
        converter.support_rgb_direct = converter.probe_rgb_direct_support();

        // Verify DMA-buf actually works (catches NVIDIA discrete GPUs where
        // EGLImage creation succeeds but rendered data is all zeros)
        if converter.gl_context.transfer_backend.is_dma() && !converter.verify_dma_buf_roundtrip() {
            log::info!("DMA-buf verification failed — falling back to PBO transfers");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
            // RGB direct rendering also requires DMA, so disable it
            converter.support_rgb_direct = false;
        }

        // If DMA-buf failed/unavailable but GL is alive, use PBO transfers
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
                if !backend.is_dma() {
                    converter.support_rgb_direct = false;
                }
            }
        }

        log::debug!(
            "GLConverter created (transfer={:?}, rgb_direct={})",
            converter.gl_context.transfer_backend,
            converter.support_rgb_direct
        );
        Ok(converter)
    }

    /// Probe whether the GPU supports direct RGB rendering via BGR888 DMA-buf
    /// backed renderbuffer. Creates a small test FBO and checks completeness.
    /// Returns `false` on any failure (DMA unavailable, EGLImage rejected, FBO incomplete).
    fn probe_rgb_direct_support(&self) -> bool {
        if !self.gl_context.transfer_backend.is_dma() {
            log::debug!("probe_rgb_direct: no DMA support");
            return false;
        }

        // Check glEGLImageTargetRenderbufferStorageOES is available
        if self
            .gl_context
            .egl
            .get_proc_address("glEGLImageTargetRenderbufferStorageOES")
            .is_none()
        {
            log::debug!("probe_rgb_direct: glEGLImageTargetRenderbufferStorageOES not available");
            return false;
        }

        // Allocate a small test DMA buffer (64x64 RGB = 12288 bytes)
        let test_img = match TensorImage::new(64, 64, RGB, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::debug!("probe_rgb_direct: failed to allocate test DMA buffer: {e}");
                return false;
            }
        };

        // Create EGLImage from the test DMA buffer
        let egl_image =
            match self.create_egl_image_with_dims(&test_img, 64, 64, DrmFourcc::Bgr888, 3) {
                Ok(img) => img,
                Err(e) => {
                    log::debug!("probe_rgb_direct: EGLImage creation failed: {e}");
                    return false;
                }
            };

        // Create renderbuffer, bind EGLImage, create FBO, check completeness
        let result = unsafe {
            let mut rbo = 0u32;
            gls::gl::GenRenderbuffers(1, &mut rbo);
            gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
            gls::gl::EGLImageTargetRenderbufferStorageOES(
                gls::gl::RENDERBUFFER,
                egl_image.egl_image.as_ptr(),
            );

            let gl_err = gls::gl::GetError();
            if gl_err != gls::gl::NO_ERROR {
                log::debug!(
                    "probe_rgb_direct: EGLImageTargetRenderbufferStorageOES failed: {gl_err:#X}"
                );
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
                gls::gl::DeleteRenderbuffers(1, &rbo);
                return false;
            }

            let mut fbo = 0u32;
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::FramebufferRenderbuffer(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::RENDERBUFFER,
                rbo,
            );

            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            let complete = status == gls::gl::FRAMEBUFFER_COMPLETE;

            // Cleanup
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
            gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
            gls::gl::DeleteRenderbuffers(1, &rbo);

            complete
        };
        // egl_image and test_img drop automatically here

        log::info!("probe_rgb_direct: BGR888 renderbuffer FBO support = {result}");
        result
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
        let src = match TensorImage::new(64, 64, RGBA, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA source: {e}");
                return false;
            }
        };

        {
            let mut map = match src.tensor().map() {
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
        let mut dst = match TensorImage::new(64, 64, RGBA, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA destination: {e}");
                return false;
            }
        };

        // Run the full DMA-buf EGLImage render pipeline
        if let Err(e) =
            self.convert_dest_dma(&mut dst, &src, Rotation::None, Flip::None, Crop::no_crop())
        {
            log::info!("verify_dma_buf_roundtrip: convert_dest_dma failed: {e}");
            return false;
        }

        // Read back the center pixel at (32, 32) from the destination
        let map = match dst.tensor().map() {
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

    /// Compute padded bbox regions and atlas offsets for a set of detections.
    ///
    /// Returns the vector of `MaskRegion` with stacked atlas_y_offset values
    /// and the total compact atlas height.
    fn compute_atlas_regions(
        detect: &[DetectBox],
        output_width: usize,
        output_height: usize,
        padding: usize,
    ) -> (Vec<MaskRegion>, usize) {
        let ow = output_width as i32;
        let oh = output_height as i32;
        let owf = output_width as f32;
        let ohf = output_height as f32;
        let pad = padding as i32;

        let mut regions = Vec::with_capacity(detect.len());
        let mut atlas_y = 0usize;
        for det in detect.iter() {
            let bbox_x = (det.bbox.xmin * owf).round() as i32;
            let bbox_y = (det.bbox.ymin * ohf).round() as i32;
            let bbox_w = ((det.bbox.xmax - det.bbox.xmin) * owf).round() as i32;
            let bbox_h = ((det.bbox.ymax - det.bbox.ymin) * ohf).round() as i32;
            let bbox_x = bbox_x.max(0).min(ow);
            let bbox_y = bbox_y.max(0).min(oh);
            let bbox_w = bbox_w.max(1).min(ow - bbox_x);
            let bbox_h = bbox_h.max(1).min(oh - bbox_y);

            let padded_x = (bbox_x - pad).max(0);
            let padded_y = (bbox_y - pad).max(0);
            let padded_w = ((bbox_x + bbox_w + pad).min(ow) - padded_x).max(1);
            let padded_h = ((bbox_y + bbox_h + pad).min(oh) - padded_y).max(1);

            regions.push(MaskRegion {
                atlas_y_offset: atlas_y,
                padded_x: padded_x as usize,
                padded_y: padded_y as usize,
                padded_w: padded_w as usize,
                padded_h: padded_h as usize,
                bbox_x: bbox_x as usize,
                bbox_y: bbox_y as usize,
                bbox_w: bbox_w as usize,
                bbox_h: bbox_h as usize,
            });
            atlas_y += padded_h as usize;
        }
        (regions, atlas_y)
    }

    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) {
        self.int8_interpolation_mode = mode;
        log::debug!("Int8 interpolation mode set to {:?}", mode);
    }

    /// Ensures the mask FBO + R8 texture are allocated at the given dimensions.
    /// Creates or resizes the FBO and texture as needed.
    fn ensure_mask_fbo(&mut self, width: usize, height: usize) -> crate::Result<()> {
        if self.mask_fbo_width == width && self.mask_fbo_height == height && self.mask_fbo != 0 {
            return Ok(());
        }

        // Create FBO if needed
        if self.mask_fbo == 0 {
            unsafe {
                gls::gl::GenFramebuffers(1, &mut self.mask_fbo);
            }
        }
        // Create texture if needed
        if self.mask_fbo_texture == 0 {
            unsafe {
                gls::gl::GenTextures(1, &mut self.mask_fbo_texture);
            }
        }

        // Allocate R8 texture
        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.mask_fbo_texture);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::R8 as i32,
                width as i32,
                height as i32,
                0,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
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

        // Attach to FBO
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.mask_fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.mask_fbo_texture,
                0,
            );
            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if status != gls::gl::FRAMEBUFFER_COMPLETE {
                return Err(crate::Error::OpenGl(format!(
                    "Mask FBO incomplete: status=0x{status:X}"
                )));
            }
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        }

        self.mask_fbo_width = width;
        self.mask_fbo_height = height;
        log::debug!("Mask FBO allocated at {width}x{height}");
        Ok(())
    }

    /// Ensures the mask atlas FBO and PBO are allocated for the given total
    /// atlas dimensions.  Unlike `ensure_mask_atlas`, the caller provides
    /// the exact atlas height (e.g. sum of padded bbox heights).
    fn ensure_mask_atlas_size(&mut self, width: usize, atlas_height: usize) -> crate::Result<()> {
        if self.mask_fbo_width == width
            && self.mask_fbo_height >= atlas_height
            && self.mask_fbo != 0
            && self.mask_atlas_pbo != 0
        {
            return Ok(());
        }
        self.ensure_mask_fbo(width, atlas_height)?;
        let pbo_size = width * atlas_height;
        unsafe {
            if self.mask_atlas_pbo == 0 {
                gls::gl::GenBuffers(1, &mut self.mask_atlas_pbo);
            }
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, self.mask_atlas_pbo);
            gls::gl::BufferData(
                gls::gl::PIXEL_PACK_BUFFER,
                pbo_size as isize,
                std::ptr::null(),
                gls::gl::DYNAMIC_READ,
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
        }
        Ok(())
    }

    /// Decode all detection masks into a single atlas texture and read back
    /// as a contiguous buffer, with one PBO readback for all masks.
    ///
    /// Returns `(atlas_pixels, metadata)` where `atlas_pixels` is a contiguous
    /// `Vec<u8>` of size `output_width * compact_atlas_height` (where
    /// `compact_atlas_height` is the sum of padded bbox heights) and `metadata`
    /// contains per-detection bbox info (with empty pixel vecs).
    pub fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::decode_masks_atlas");

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let padding = 4usize;

        let (height, width, num_protos) = proto_data.protos.dim();
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;

        // Pre-compute atlas regions and total height to size the FBO/PBO
        let (regions, compact_atlas_height) =
            Self::compute_atlas_regions(detect, output_width, output_height, padding);

        // Save current FBO and viewport
        let (saved_fbo, saved_viewport) = unsafe {
            let mut fbo: i32 = 0;
            gls::gl::GetIntegerv(gls::gl::FRAMEBUFFER_BINDING, &mut fbo);
            let mut vp = [0i32; 4];
            gls::gl::GetIntegerv(gls::gl::VIEWPORT, vp.as_mut_ptr());
            (fbo as u32, vp)
        };

        // Ensure atlas FBO and PBO are allocated for the compact size
        self.ensure_mask_atlas_size(output_width, compact_atlas_height)?;

        // Upload proto texture array and select the logit-threshold shader
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_texture.id);
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

        let atlas_result = match &proto_data.protos {
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                let mut tex_data = vec![0i8; height * width * num_protos];
                for k in 0..num_protos {
                    for y in 0..height {
                        for x in 0..width {
                            tex_data[k * height * width + y * width + x] = protos[[y, x, k]];
                        }
                    }
                }
                gls::tex_image3d(
                    texture_target,
                    0,
                    gls::gl::R8I as i32,
                    width as i32,
                    height as i32,
                    num_protos as i32,
                    0,
                    gls::gl::RED_INTEGER,
                    gls::gl::BYTE,
                    Some(&tex_data),
                );

                let proto_scale = quantization.scale;
                let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

                let program = match self.int8_interpolation_mode {
                    Int8InterpolationMode::Nearest => &self.proto_mask_logit_int8_nearest_program,
                    _ => &self.proto_mask_logit_int8_bilinear_program,
                };
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;

                self.render_mask_atlas_compact(
                    program,
                    regions,
                    &proto_data.mask_coefficients,
                    output_width,
                    output_height,
                    Some(proto_scaled_zp),
                )
            }
            ProtoTensor::Float(protos_f32) => {
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
                if self.has_float_linear {
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
                }

                let program = &self.proto_mask_logit_f32_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;

                self.render_mask_atlas_compact(
                    program,
                    regions,
                    &proto_data.mask_coefficients,
                    output_width,
                    output_height,
                    None,
                )
            }
        };

        // Restore previous FBO + viewport
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, saved_fbo);
            gls::gl::Viewport(
                saved_viewport[0],
                saved_viewport[1],
                saved_viewport[2],
                saved_viewport[3],
            );
        }

        let (atlas_pixels, regions) = atlas_result?;
        Ok((atlas_pixels, regions))
    }

    /// Render all detection masks into a compact atlas where each strip is
    /// sized to the padded bounding box, not the full output resolution.
    ///
    /// The atlas width equals `output_width`; each detection occupies a
    /// horizontal strip whose height is the padded bbox height.  Strips are
    /// stacked vertically.  A single PBO readback retrieves the entire atlas.
    ///
    /// Returns `(atlas_pixels, regions)` where `regions` describes each
    /// detection's location within the atlas.
    #[allow(clippy::too_many_arguments)]
    fn render_mask_atlas_compact(
        &self,
        program: &GlProgram,
        regions: Vec<MaskRegion>,
        mask_coefficients: &[Vec<f32>],
        output_width: usize,
        output_height: usize,
        proto_scaled_zp: Option<f32>,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        if regions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let owf = output_width as f32;
        let ohf = output_height as f32;

        let atlas_height = regions.last().map_or(0, |r| r.atlas_y_offset + r.padded_h);
        let ahf = atlas_height as f32;

        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.mask_fbo);
            gls::gl::Viewport(0, 0, output_width as i32, atlas_height as i32);
            gls::gl::Disable(gls::gl::BLEND);
            gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        }

        if let Some(first_coeff) = mask_coefficients.first() {
            if first_coeff.len() > 32 {
                log::warn!(
                    "render_mask_atlas_compact: {} mask coefficients exceeds shader \
                     limit of 32 — coefficients will be truncated",
                    first_coeff.len()
                );
            }
        }

        for (region, coeff) in regions.iter().zip(mask_coefficients.iter()) {
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (j, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[j / 4][j % 4] = *val;
            }
            program.load_uniform_4fv(c"mask_coeff", &packed_coeff)?;

            // For int8 paths: upload precomputed coeff_sum * scaled_zp
            if let Some(szp) = proto_scaled_zp {
                let coeff_sum: f32 = coeff.iter().take(32).sum();
                program.load_uniform_1f(c"coeff_sum_x_szp", coeff_sum * szp)?;
            }

            // The bbox quad position in the atlas:
            // - X: the padded bbox horizontal position (same as in output coords)
            // - Y: the strip's vertical offset in the atlas
            let dst_left = region.padded_x as f32 / owf * 2.0 - 1.0;
            let dst_right = (region.padded_x + region.padded_w) as f32 / owf * 2.0 - 1.0;
            let dst_bottom = region.atlas_y_offset as f32 / ahf * 2.0 - 1.0;
            let dst_top = (region.atlas_y_offset + region.padded_h) as f32 / ahf * 2.0 - 1.0;

            // Proto texture coords map the padded bbox to proto space
            let src_left = region.padded_x as f32 / owf;
            let src_right = (region.padded_x + region.padded_w) as f32 / owf;
            let src_bottom = region.padded_y as f32 / ohf;
            let src_top = (region.padded_y + region.padded_h) as f32 / ohf;

            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let verts: [f32; 12] = [
                    dst_left, dst_top, 0.0, dst_right, dst_top, 0.0, dst_right, dst_bottom, 0.0,
                    dst_left, dst_bottom, 0.0,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 12) as isize,
                    verts.as_ptr() as *const c_void,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let tc: [f32; 8] = [
                    src_left, src_top, src_right, src_top, src_right, src_bottom, src_left,
                    src_bottom,
                ];
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

        // Single readback for the compact atlas
        let atlas_bytes = output_width * atlas_height;
        let mut pixels = vec![0u8; atlas_bytes];

        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, self.mask_atlas_pbo);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                output_width as i32,
                atlas_height as i32,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                atlas_bytes as i32,
                std::ptr::null_mut(),
            );
            gls::gl::Finish();

            let ptr = gls::gl::MapBufferRange(
                gls::gl::PIXEL_PACK_BUFFER,
                0,
                atlas_bytes as isize,
                gls::gl::MAP_READ_BIT,
            );
            if ptr.is_null() {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                return Err(crate::Error::OpenGl(
                    "Failed to map compact atlas PBO for readback".to_string(),
                ));
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, pixels.as_mut_ptr(), atlas_bytes);
            gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
        }

        Ok((pixels, regions))
    }

    pub(super) fn check_src_format_supported(backend: TransferBackend, img: &TensorImage) -> bool {
        if backend.is_dma() && img.tensor().memory() == TensorMemory::Dma {
            // EGLImage supports RGBA, GREY, YUYV, and NV12 for DMA buffers.
            // VYUY excluded: Vivante GPU accepts the DRM fourcc but produces
            // incorrect output (similarity ~0.28 vs reference).
            matches!(img.fourcc(), RGBA | GREY | YUYV | NV12)
        } else {
            matches!(img.fourcc(), RGB | RGBA | GREY)
        }
    }

    pub(super) fn check_dst_format_supported(
        backend: TransferBackend,
        img: &TensorImage,
        has_bgra: bool,
    ) -> bool {
        if img.fourcc() == BGRA && !has_bgra {
            return false;
        }
        if backend.is_dma() && img.tensor().memory() == TensorMemory::Dma {
            matches!(
                img.fourcc(),
                RGBA | BGRA | GREY | PLANAR_RGB | RGB | RGB_INT8 | PLANAR_RGB_INT8
            )
        } else {
            matches!(img.fourcc(), RGB | RGBA | BGRA | GREY | RGB_INT8)
        }
    }

    /// Checks required GL extensions and returns optional capability flags:
    /// `(has_float_linear, has_bgra)`.
    fn gl_check_support() -> Result<(bool, bool), crate::Error> {
        if let Ok(version) = gls::get_string(gls::gl::SHADING_LANGUAGE_VERSION) {
            log::debug!("GL Shading Language Version: {version:?}");
        } else {
            log::warn!("Could not get GL Shading Language Version");
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

        Ok((has_float_linear, has_bgra))
    }

    fn setup_renderbuffer_dma(&mut self, dst: &TensorImage) -> crate::Result<()> {
        self.convert_fbo.bind();

        let (width, height) = if matches!(dst.fourcc(), PLANAR_RGB | PLANAR_RGB_INT8) {
            let width = dst.width();
            let height = dst.height() * 3;
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };
        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst)?;
        unsafe {
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
            gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_egl.as_ptr());
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

    fn convert_dest_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        assert!(self.gl_context.transfer_backend.is_dma());
        if fourcc_is_packed_rgb(dst.fourcc()) {
            if self.support_rgb_direct {
                self.convert_to_rgb_direct(src, dst, rotation, flip, crop)
            } else {
                // Two-pass packed RGB is slower than G2D/CPU; decline so
                // ImageProcessor falls through to a faster backend.
                Err(crate::Error::NotSupported(
                    "OpenGL two-pass packed RGB disabled (no direct RGB support)".into(),
                ))
            }
        } else if dst.is_planar() {
            self.setup_renderbuffer_dma(dst)?;
            self.convert_to_planar(src, dst, rotation, flip, crop)
        } else {
            self.setup_renderbuffer_dma(dst)?;
            self.convert_to(src, dst, rotation, flip, crop)
        }
    }

    fn setup_renderbuffer_non_dma(&mut self, dst: &TensorImage, crop: Crop) -> crate::Result<()> {
        debug_assert!(matches!(
            dst.fourcc(),
            RGB | RGBA | BGRA | GREY | PLANAR_RGB | RGB_INT8
        ));
        let (width, height) = if dst.is_planar() {
            let width = dst.width() / 4;
            let height = match dst.fourcc() {
                RGBA => dst.height() * 4,
                RGB => dst.height() * 3,
                GREY => dst.height(),
                _ => unreachable!(),
            };
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };

        let format = if dst.is_planar() {
            gls::gl::RED
        } else {
            match dst.fourcc() {
                RGB | RGB_INT8 => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                GREY => gls::gl::RED,
                _ => unreachable!(),
            }
        };

        let start = Instant::now();
        self.convert_fbo.bind();

        let map;

        let pixels = if crop.dst_rect.is_none_or(|crop| {
            crop.top == 0
                && crop.left == 0
                && crop.height == dst.height()
                && crop.width == dst.width()
        }) {
            std::ptr::null()
        } else {
            map = dst.tensor().map()?;
            map.as_ptr() as *const c_void
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
    fn setup_renderbuffer_from_pbo(
        &mut self,
        dst: &TensorImage,
        buffer_id: u32,
    ) -> crate::Result<()> {
        let (width, height) = (dst.width() as i32, dst.height() as i32);
        let format = match dst.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO renderbuffer not supported for {}",
                    dst.fourcc().display()
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

    fn convert_dest_non_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }
        log::debug!("Draw to framebuffer takes {:?}", start.elapsed());
        let start = Instant::now();
        let dest_format = match dst.fourcc() {
            RGB | RGB_INT8 => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            GREY => gls::gl::RED,
            _ => unreachable!(),
        };

        unsafe {
            let mut dst_map = dst.tensor().map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            // Apply XOR 0x80 for int8 formats (convert uint8 → int8 representation)
            if fourcc_is_int8(dst.fourcc()) {
                for byte in dst_map.iter_mut() {
                    *byte ^= 0x80;
                }
            }
        }
        log::debug!("Read from framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    /// Convert between two PBO-backed images.
    ///
    /// Source PBO is bound as `GL_PIXEL_UNPACK_BUFFER` for zero-copy texture upload
    /// (avoids `tensor.map()` to prevent GL-thread deadlocks). Destination uses
    /// `GL_PIXEL_PACK_BUFFER` for zero-copy readback into the PBO.
    fn convert_pbo_to_pbo(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // Safety check: neither PBO must be mapped; extract buffer IDs before releasing borrows
        let (src_buffer_id, dst_buffer_id) = {
            let src_pbo = match &src.tensor {
                edgefirst_tensor::Tensor::Pbo(p) => p,
                _ => {
                    return Err(crate::Error::OpenGl(
                        "convert_pbo_to_pbo: src is not a PBO tensor".to_string(),
                    ))
                }
            };
            let dst_pbo = match &dst.tensor {
                edgefirst_tensor::Tensor::Pbo(p) => p,
                _ => {
                    return Err(crate::Error::OpenGl(
                        "convert_pbo_to_pbo: dst is not a PBO tensor".to_string(),
                    ))
                }
            };

            if src_pbo.is_mapped() || dst_pbo.is_mapped() {
                return Err(crate::Error::OpenGl(
                    "Cannot convert PBO tensors while they are mapped".to_string(),
                ));
            }

            (src_pbo.buffer_id(), dst_pbo.buffer_id())
        };

        // Setup renderbuffer (same as non-DMA path)
        self.setup_renderbuffer_non_dma(dst, crop)?;

        // Upload source from PBO and render.
        // We cannot call convert_to/draw_src_texture directly because they
        // call src.tensor().map() which sends a message back to THIS thread,
        // causing a deadlock. Instead, bind the source PBO as UNPACK buffer
        // and upload to the texture with a NULL pointer — GL reads directly
        // from the PBO, zero CPU copy.
        let start = Instant::now();
        self.draw_src_texture_from_pbo(src, src_buffer_id, dst, rotation, flip, crop)?;
        log::debug!("PBO render takes {:?}", start.elapsed());

        // Readback into destination PBO instead of CPU memory
        let start_read = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {}",
                    dst.fourcc().display()
                )))
            }
        };

        unsafe {
            // Bind destination PBO as PACK buffer — glReadnPixels will write into it
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                std::ptr::null_mut(), // NULL pointer = write to bound PACK buffer
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }

        check_gl_error(function!(), line!())?;

        // Handle int8 XOR if needed (must map PBO to do this on the GL thread
        // directly, since we're already on the GL thread)
        if fourcc_is_int8(dst.fourcc()) {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.tensor.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if !ptr.is_null() {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.tensor.len());
                    for byte in slice.iter_mut() {
                        *byte ^= 0x80;
                    }
                    gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                }
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
    fn draw_src_texture_from_pbo(
        &mut self,
        src: &TensorImage,
        src_buffer_id: u32,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "PBO upload not supported for {:?}",
                    src.fourcc()
                )));
            }
        };

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
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
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
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
            if src.fourcc() == crate::GREY {
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
                src.width() as i32,
                src.height() as i32,
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
    fn convert_any_to_pbo(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_buffer_id = match &dst.tensor {
            edgefirst_tensor::Tensor::Pbo(p) => {
                if p.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert to a mapped PBO tensor".to_string(),
                    ));
                }
                p.buffer_id()
            }
            _ => {
                return Err(crate::Error::OpenGl(
                    "convert_any_to_pbo: dst is not a PBO tensor".to_string(),
                ))
            }
        };

        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }
        log::debug!("any-to-PBO render takes {:?}", start.elapsed());

        // PBO readback
        let start_read = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {}",
                    dst.fourcc().display()
                )))
            }
        };
        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                std::ptr::null_mut(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }
        check_gl_error(function!(), line!())?;

        if fourcc_is_int8(dst.fourcc()) {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.tensor.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if !ptr.is_null() {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.tensor.len());
                    for byte in slice.iter_mut() {
                        *byte ^= 0x80;
                    }
                    gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                }
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
    fn convert_pbo_to_mem(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let src_buffer_id = match &src.tensor {
            edgefirst_tensor::Tensor::Pbo(p) => {
                if p.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert from a mapped PBO tensor".to_string(),
                    ));
                }
                p.buffer_id()
            }
            _ => {
                return Err(crate::Error::OpenGl(
                    "convert_pbo_to_mem: src is not a PBO tensor".to_string(),
                ))
            }
        };

        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        self.draw_src_texture_from_pbo(src, src_buffer_id, dst, rotation, flip, crop)?;
        log::debug!("PBO-to-mem render takes {:?}", start.elapsed());

        // Normal readback into Mem dst
        let start = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {}",
                    dst.fourcc().display()
                )))
            }
        };
        unsafe {
            let mut dst_map = dst.tensor().map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            if fourcc_is_int8(dst.fourcc()) {
                for byte in dst_map.iter_mut() {
                    *byte ^= 0x80;
                }
            }
        }
        log::debug!("PBO-to-mem readback takes {:?}", start.elapsed());
        Ok(())
    }

    fn convert_to(
        &mut self,
        src: &TensorImage,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        check_gl_error(function!(), line!())?;

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });
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

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
            }
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
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
        if self.gl_context.transfer_backend.is_dma() && src.tensor().memory() == TensorMemory::Dma {
            match self.get_or_create_egl_image(CacheKind::Src, src) {
                Ok(src_egl) => self.draw_camera_texture_eglimage(
                    src,
                    src_egl,
                    src_roi,
                    dst_roi,
                    rotation_offset,
                    flip,
                )?,
                Err(e) => {
                    log::warn!("EGL image creation failed for {:?}: {:?}", src.fourcc(), e);
                    let start = Instant::now();
                    self.draw_src_texture(src, src_roi, dst_roi, rotation_offset, flip)?;
                    log::debug!("draw_src_texture takes {:?}", start.elapsed());
                }
            }
        } else {
            let start = Instant::now();
            self.draw_src_texture(src, src_roi, dst_roi, rotation_offset, flip)?;
            log::debug!("draw_src_texture takes {:?}", start.elapsed());
        }

        let start = Instant::now();
        unsafe { gls::gl::Finish() };
        log::debug!("gl_Finish takes {:?}", start.elapsed());
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn convert_to_planar(
        &mut self,
        src: &TensorImage,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
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

        let alpha = match dst.fourcc() {
            PLANAR_RGB | PLANAR_RGB_INT8 => false,
            PLANAR_RGBA => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PLANAR_RGB, PLANAR_RGB_INT8, or PLANAR_RGBA"
                        .to_string(),
                ));
            }
        };
        let is_int8 = fourcc_is_int8(dst.fourcc());

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
            }
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
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

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                self.clear_rect_planar(
                    dst.width(),
                    dst.height(),
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

        let src_egl = self.get_or_create_egl_image(CacheKind::Src, src)?;

        self.draw_camera_texture_to_rgb_planar(
            src_egl,
            src_roi,
            dst_roi,
            rotation_offset,
            flip,
            alpha,
            is_int8,
        )?;
        unsafe { gls::gl::Finish() };

        Ok(())
    }

    /// Render packed RGB (or RGB_INT8) to a DMA destination buffer using a
    /// two-pass architecture:
    ///
    /// **Pass 1:** Render source → intermediate RGBA texture via `convert_to()`
    /// (reuses the battle-tested RGBA path with full crop/letterbox/rotation/flip).
    ///
    /// **Pass 2:** Pack intermediate RGBA → RGB DMA destination using a simple
    /// packing shader with 2D sampler. The destination DMA buffer is reinterpreted
    /// as RGBA8 at (W*3/4) x H dimensions.
    fn convert_to_packed_rgb(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_w = dst.width();
        let dst_h = dst.height();
        let is_int8 = fourcc_is_int8(dst.fourcc());

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
        self.convert_to(src, dst, rotation, flip, crop)?;

        // --- Pass 2: Pack intermediate RGBA → RGB DMA destination ---
        self.convert_fbo.bind();
        let dest_egl =
            self.get_or_create_egl_image_rgb(dst, render_w, render_h, DrmFourcc::Abgr8888, 4)?;
        unsafe {
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
        Ok(())
    }

    /// Render directly to an RGB8 renderbuffer backed by BGR888 DMA-buf.
    /// Single-pass: no intermediate texture, no packing shader.
    fn convert_to_rgb_direct(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let is_int8 = fourcc_is_int8(dst.fourcc());

        log::debug!(
            "convert_to_rgb_direct: {}x{} single-pass int8={is_int8}",
            dst.width(),
            dst.height(),
        );

        // Get or create cached renderbuffer
        let (rbo, width, height) = self.get_or_create_rgb_direct_rbo(dst)?;

        // Bind FBO with renderbuffer attachment
        self.convert_fbo.bind();
        unsafe {
            gls::gl::FramebufferRenderbuffer(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::RENDERBUFFER,
                rbo,
            );
            check_gl_error(function!(), line!())?;

            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if status != gls::gl::FRAMEBUFFER_COMPLETE {
                log::warn!("convert_to_rgb_direct: FBO incomplete (0x{status:x}), falling back");
                return self.convert_to_packed_rgb(src, dst, rotation, flip, crop);
            }

            gls::gl::Viewport(0, 0, width, height);
        }

        // For int8, temporarily swap to int8 shader programs and bias the clear color
        let crop = if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
            // Bias the letterbox clear color with XOR 0x80 since glClear bypasses
            // the fragment shader — the int8 bias must be applied to the color directly.
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

        let result = self.convert_to(src, dst, rotation, flip, crop);

        // Swap back
        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        result
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
        &self,
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

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.as_ptr());
            check_gl_error(function!(), line!())?;
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

    fn draw_src_texture(
        &mut self,
        src: &TensorImage,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            GREY => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "draw_src_texture does not support {:?} (use DMA-BUF path for YUV)",
                    src.fourcc()
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
            if src.fourcc() == GREY {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src as i32);
                }
            }
            self.camera_normal_texture.update_texture(
                texture_target,
                src.width(),
                src.height(),
                texture_format,
                &src.tensor().map()?,
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

    fn draw_camera_texture_eglimage(
        &self,
        src: &TensorImage,
        egl_img: egl::Image,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) -> Result<(), Error> {
        // let texture_target = gls::gl::TEXTURE_2D;
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

            if src.fourcc() == GREY {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src as i32);
                }
            }

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.as_ptr());
            check_gl_error(function!(), line!())?;
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

    fn create_image_from_dma2(&self, src: &TensorImage) -> Result<EglImage, crate::Error> {
        let width;
        let height;
        let format;
        let channels;

        // NV12 is semi-planar but handled specially via EGL multi-plane import
        if src.fourcc() == NV12 {
            if !src.width().is_multiple_of(4) {
                return Err(Error::NotSupported(
                    "OpenGL EGLImage doesn't support image widths which are not multiples of 4"
                        .to_string(),
                ));
            }
            width = src.width();
            height = src.height();
            format = fourcc_to_drm(NV12)?;
            channels = 1; // Y plane pitch is 1 byte per pixel
        } else if src.is_planar() {
            if !src.width().is_multiple_of(16) {
                return Err(Error::NotSupported(
                    "OpenGL Planar RGB EGLImage doesn't support image widths which are not multiples of 16"
                        .to_string(),
                ));
            }
            match src.fourcc() {
                PLANAR_RGB | PLANAR_RGB_INT8 => {
                    format = DrmFourcc::R8;
                    width = src.width();
                    height = src.height() * 3;
                    channels = 1;
                }
                fourcc => {
                    return Err(crate::Error::NotSupported(format!(
                        "Unsupported Planar FourCC {fourcc:?}"
                    )));
                }
            };
        } else {
            if !src.width().is_multiple_of(4) {
                return Err(Error::NotSupported(
                    "OpenGL EGLImage doesn't support image widths which are not multiples of 4"
                        .to_string(),
                ));
            }
            width = src.width();
            height = src.height();
            format = fourcc_to_drm(src.fourcc())?;
            channels = src.channels();
        }

        let fd = match &src.tensor {
            edgefirst_tensor::Tensor::Dma(dma_tensor) => dma_tensor.fd.as_raw_fd(),
            edgefirst_tensor::Tensor::Shm(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support SHM".to_string(),
                ));
            }
            edgefirst_tensor::Tensor::Mem(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support MEM".to_string(),
                ));
            }
            edgefirst_tensor::Tensor::Pbo(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support PBO".to_string(),
                ));
            }
        };

        // For NV12, plane0 pitch is width (Y is 1 byte/pixel)
        // For other formats, pitch is width * channels
        let plane0_pitch = if src.fourcc() == NV12 {
            width
        } else {
            width * channels
        };

        let mut egl_img_attr = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            format as Attrib,
            khronos_egl::WIDTH as Attrib,
            width as Attrib,
            khronos_egl::HEIGHT as Attrib,
            height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            plane0_pitch as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            0 as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
        ];

        // NV12 requires a second plane for UV data
        if src.fourcc() == NV12 {
            let uv_offset = width * height; // Y plane size
            egl_img_attr.append(&mut vec![
                egl_ext::DMA_BUF_PLANE1_FD as Attrib,
                fd as Attrib,
                egl_ext::DMA_BUF_PLANE1_OFFSET as Attrib,
                uv_offset as Attrib,
                egl_ext::DMA_BUF_PLANE1_PITCH as Attrib,
                width as Attrib, // UV plane has same width as Y plane
            ]);
        }

        if matches!(src.fourcc(), YUYV | VYUY | NV12) {
            egl_img_attr.append(&mut vec![
                egl_ext::YUV_COLOR_SPACE_HINT as Attrib,
                egl_ext::ITU_REC709 as Attrib,
                egl_ext::SAMPLE_RANGE_HINT as Attrib,
                egl_ext::YUV_NARROW_RANGE as Attrib,
            ]);
        }

        egl_img_attr.push(khronos_egl::NONE as Attrib);

        match self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr) {
            Ok(v) => Ok(v),
            Err(e) => Err(e),
        }
    }

    fn new_egl_image_owned(
        &'_ self,
        target: egl::Enum,
        attrib_list: &[Attrib],
    ) -> Result<EglImage, Error> {
        let image = GlContext::egl_create_image_with_fallback(
            &self.gl_context.egl,
            self.gl_context.display.as_display(),
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            target,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            attrib_list,
        )?;
        Ok(EglImage {
            egl_image: image,
            display: self.gl_context.display.as_display(),
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
        img: &TensorImage,
    ) -> Result<egl::Image, crate::Error> {
        let id = img.buffer_identity().id();

        // Sweep dead entries opportunistically before looking up.
        match cache {
            CacheKind::Src => self.src_egl_cache.sweep(),
            CacheKind::Dst => self.dst_egl_cache.sweep(),
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
                log::trace!("EglImageCache {:?} hit: id={id:#x}", cache);
                return Ok(cached.egl_image.egl_image);
            }
            egl_cache.misses += 1;
            log::trace!("EglImageCache {:?} miss: id={id:#x}", cache);
            // Evict least-recently-used entry if at capacity.
            if egl_cache.entries.len() >= egl_cache.capacity {
                egl_cache.evict_lru();
            }
        }

        let egl_image_obj = self.create_image_from_dma2(img)?;
        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        let egl_cache = match cache {
            CacheKind::Src => &mut self.src_egl_cache,
            CacheKind::Dst => &mut self.dst_egl_cache,
        };
        let ts = egl_cache.next_timestamp();
        egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: None,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    /// Create an EGLImage from a DMA buffer with explicitly specified internal
    /// dimensions and format. Used when the GL render surface differs from the
    /// logical TensorImage dimensions (e.g., packed RGB reinterpretation).
    fn create_egl_image_with_dims(
        &self,
        img: &TensorImage,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<EglImage, crate::Error> {
        let fd = match &img.tensor {
            edgefirst_tensor::Tensor::Dma(dma_tensor) => dma_tensor.fd.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "create_egl_image_with_dims requires DMA tensor".to_string(),
                ));
            }
        };

        let pitch = width * bpp;
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
            0 as Attrib,
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
        img: &TensorImage,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<egl::Image, crate::Error> {
        let id = img.buffer_identity().id();
        self.dst_egl_cache.sweep();

        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            self.dst_egl_cache.hits += 1;
            cached.last_used = ts;
            log::trace!("EglImageCache dst (RGB) hit: id={id:#x}");
            return Ok(cached.egl_image.egl_image);
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (RGB) miss: id={id:#x}");

        if self.dst_egl_cache.entries.len() >= self.dst_egl_cache.capacity {
            self.dst_egl_cache.evict_lru();
        }

        let egl_image_obj = self.create_egl_image_with_dims(img, width, height, drm_format, bpp)?;
        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        let ts = self.dst_egl_cache.next_timestamp();
        self.dst_egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: None,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    /// Get or create an EGLImage + renderbuffer for direct RGB rendering.
    /// Both are cached in dst_egl_cache keyed by buffer identity.
    /// Returns (renderbuffer_id, width, height).
    fn get_or_create_rgb_direct_rbo(
        &mut self,
        dst: &TensorImage,
    ) -> crate::Result<(u32, i32, i32)> {
        let id = dst.buffer_identity().id();
        let width = dst.width() as i32;
        let height = dst.height() as i32;

        self.dst_egl_cache.sweep();

        // Check cache for existing entry with renderbuffer
        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            if let Some(rbo) = cached.renderbuffer {
                self.dst_egl_cache.hits += 1;
                cached.last_used = ts;
                log::trace!("EglImageCache dst (rgb_direct) hit: id={id:#x}");
                return Ok((rbo, width, height));
            }
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (rgb_direct) miss: id={id:#x}");

        // Evict least-recently-used entry if at capacity
        if self.dst_egl_cache.entries.len() >= self.dst_egl_cache.capacity {
            self.dst_egl_cache.evict_lru();
        }

        // Create EGLImage from BGR888 DMA-buf
        let egl_image_obj =
            self.create_egl_image_with_dims(dst, dst.width(), dst.height(), DrmFourcc::Bgr888, 3)?;

        // Create renderbuffer and bind EGLImage to it
        let rbo = unsafe {
            let mut rbo = 0u32;
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
            rbo
        };

        // Cache both
        let guard = dst.buffer_identity().weak();
        let ts = self.dst_egl_cache.next_timestamp();
        self.dst_egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: Some(rbo),
                last_used: ts,
            },
        );

        Ok((rbo, width, height))
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

    fn render_yolo_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 2],
        class: usize,
    ) -> Result<(), crate::Error> {
        log::debug!("start render_yolo_segmentation");

        let [height, width] = shape;

        let format = gls::gl::RED;
        let texture_target = gls::gl::TEXTURE_2D;
        gls::use_program(self.instanced_segmentation_program.id);
        self.instanced_segmentation_program
            .load_uniform_1i(c"class_index", class as i32)?;
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

        gls::tex_image2d(
            texture_target,
            0,
            format as i32,
            width as i32,
            height as i32,
            0,
            format,
            gls::gl::UNSIGNED_BYTE,
            Some(segmentation),
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
                (texture_vertices).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            gls::gl::Finish();
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
    ) -> crate::Result<()> {
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;

        for (det, coeff) in detect.iter().zip(mask_coefficients.iter()) {
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (i, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[i / 4][i % 4] = *val;
            }

            program.load_uniform_4fv(c"mask_coeff", &packed_coeff)?;
            program.load_uniform_1i(c"class_index", det.label as i32)?;

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

        // Protos are (H, W, num_protos) in row-major. We need to repack to
        // layer-first layout: layer k = all (H, W) texels for proto k.
        let mut tex_data = vec![0i8; height * width * num_protos];
        for k in 0..num_protos {
            for y in 0..height {
                for x in 0..width {
                    tex_data[k * height * width + y * width + x] = protos[[y, x, k]];
                }
            }
        }

        gls::tex_image3d(
            texture_target,
            0,
            gls::gl::R8I as i32,
            width as i32,
            height as i32,
            num_protos as i32,
            0,
            gls::gl::RED_INTEGER,
            gls::gl::BYTE,
            Some(&tex_data),
        );

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        match self.int8_interpolation_mode {
            Int8InterpolationMode::Nearest => {
                let program = &self.proto_segmentation_int8_nearest_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients)?;
            }
            Int8InterpolationMode::Bilinear => {
                let program = &self.proto_segmentation_int8_bilinear_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients)?;
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
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

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
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

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
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

        Ok(())
    }

    fn render_segmentation(
        &mut self,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
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
            for (seg, det) in segmentation.iter().zip(detect) {
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
                    det.label,
                )?;
            }
        }

        gls::disable(gls::gl::BLEND);
        Ok(())
    }

    fn render_box(&mut self, dst: &TensorImage, detect: &[DetectBox]) -> Result<(), Error> {
        unsafe {
            gls::gl::UseProgram(self.color_program.id);
            let rescale = |x: f32| x * 2.0 - 1.0;
            let thickness = 3.0;
            for d in detect {
                self.color_program
                    .load_uniform_1i(c"class_index", d.label as i32)?;
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let bbox: [f32; 4] = d.bbox.into();
                let outer_box = [
                    bbox[0] - thickness / dst.width() as f32,
                    bbox[1] - thickness / dst.height() as f32,
                    bbox[2] + thickness / dst.width() as f32,
                    bbox[3] + thickness / dst.height() as f32,
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
