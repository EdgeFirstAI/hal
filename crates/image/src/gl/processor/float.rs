// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL float (F16/F32) preprocessing paths (Linux DMA-BUF + macOS IOSurface
//! zero-copy targets, Tegra PBO targets).
//!
//! Items grouped here:
//! * [`float_render_support`]: reportable float-render capability.
//! * [`packed_planar_layout`]: packed zero-copy surface geometry for a
//!   planar float destination.
//! * The [`GLProcessorST`] float render methods: `convert_float_to_pbo`,
//!   `convert_float_to_zero_copy`, `render_float_to_zero_copy_tail`,
//!   `feed_float_src`, `draw_float_quad`.

use std::ffi::{c_void, CStr};

use super::super::cache::{BufferImportKey, CacheKind};
use super::super::core::float_crop_uniforms;
use super::{
    check_gl_error, dyn_to_u8_src, plan_pbo_readback, spread_rows, with_mapped_pbo, GLProcessorST,
    PboReadbackLayout,
};
use crate::{Error, Flip, ResolvedCrop, Rotation};
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorDyn, TensorMemory, TensorTrait};

// Re-imports the float render-path decision from `gl::float_dispatch`.
pub(in super::super) use super::super::float_dispatch::{classify_float_render, FloatRenderPath};

/// How the float source reached the GPU this frame (see `ConvertStats`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum FloatSrcFeed {
    /// Zero-copy: the source's Dma buffer is the texture's storage.
    Import,
    /// GL-internal copy from the source PBO.
    Pbo,
    /// CPU map + TexImage upload.
    Upload,
}

/// Whether `EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT=1` disables the float-path
/// zero-copy source import.
fn float_src_import_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Reportable float render support: none on Vivante or when
/// `float_import_refused`, otherwise what the two extensions provide.
///
/// `float_import_refused` is a driver whose EGL imports no float DMA-BUF
/// format, running on a zero-copy transfer backend.
pub(in super::super) fn float_render_support(
    is_vivante: bool,
    float_import_refused: bool,
    f32_ext: bool,
    f16_ext: bool,
) -> crate::RenderDtypeSupport {
    if is_vivante || float_import_refused {
        return crate::RenderDtypeSupport {
            f32: false,
            f16: false,
        };
    }
    crate::RenderDtypeSupport {
        f32: f32_ext,
        f16: f16_ext,
    }
}

/// Packed float surface geometry for a `channels`-plane NCHW zero-copy
/// destination.
///
/// The logical destination is `[C, H, W]` f16 or f32 (`PlanarRgb` /
/// `PlanarRgba`). It is packed into an RGBA16F or RGBA32F surface where each
/// texel holds four contiguous planar elements, giving a GL-visible surface of
/// `(W/4, C*H)` texels.
///
/// Returns `None` when `W` is not divisible by 4.
pub(in super::super) fn packed_planar_layout(w: u32, h: u32, channels: u32) -> Option<(u32, u32)> {
    if !w.is_multiple_of(4) {
        return None;
    }
    Some((w / 4, channels.checked_mul(h)?))
}

/// Packed float surface geometry for an NHWC (`[H, W, 3]`) zero-copy
/// destination: four contiguous interleaved elements per texel give a surface
/// of `(W*3/4, H)`.
///
/// Returns `None` unless `W*3` is divisible by 4.
fn packed_interleaved_layout(w: u32, h: u32) -> Option<(u32, u32)> {
    let row = w.checked_mul(3)?;
    if !row.is_multiple_of(4) {
        return None;
    }
    Some((row / 4, h))
}

/// Fetch the PBO buffer id of a float PBO destination tensor. Returns an
/// `OpenGl` error when the tensor is not PBO-backed or the PBO is mapped.
fn float_pbo_buffer_id<T>(t: &edgefirst_tensor::Tensor<T>) -> crate::Result<u32>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    let buffer_id = t.pbo_id().ok_or_else(|| {
        crate::Error::OpenGl("convert_float_to_pbo: dst is not a PBO tensor".to_string())
    })?;
    if t.pbo_is_mapped() == Some(true) {
        return Err(crate::Error::OpenGl(
            "Cannot convert to a mapped PBO tensor".to_string(),
        ));
    }
    Ok(buffer_id)
}

/// Block until all previously-issued GL commands have completed, using a fence
/// sync (`glFenceSync` + `glClientWaitSync` with
/// `GL_SYNC_FLUSH_COMMANDS_BIT`). Falls back to a blocking `glFinish` when
/// the fence cannot be created, the wait times out or it fails.
///
/// # Safety
/// Must be called on the thread owning the current GL context.
pub(super) unsafe fn finish_via_fence() {
    unsafe {
        // 1 second, in nanoseconds.
        const TIMEOUT_NS: u64 = 1_000_000_000;
        let sync = edgefirst_gl::gl::FenceSync(edgefirst_gl::gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if sync.is_null() {
            // The fence could not be created.
            edgefirst_gl::gl::Finish();
            return;
        }
        let status = edgefirst_gl::gl::ClientWaitSync(
            sync,
            edgefirst_gl::gl::SYNC_FLUSH_COMMANDS_BIT,
            TIMEOUT_NS,
        );
        edgefirst_gl::gl::DeleteSync(sync);
        match status {
            s if s == edgefirst_gl::gl::ALREADY_SIGNALED
                || s == edgefirst_gl::gl::CONDITION_SATISFIED => {}
            // Timeout expired or the wait failed.
            _ => edgefirst_gl::gl::Finish(),
        }
    }
}

impl GLProcessorST {
    /// `glReadPixels` the packed float render target into the destination PBO,
    /// placing each destination row at its own pitch.
    ///
    /// A pitch that is a whole number of texels is handed to GL as
    /// `GL_PACK_ROW_LENGTH`; a pitch no texel size divides is read tight and
    /// spread inside the PBO's own mapping.
    ///
    /// # Safety
    /// Must run on the GL thread with the float render target bound as a complete
    /// read framebuffer. `layout` must describe this destination: every write
    /// stays inside its `needed` bytes.
    unsafe fn read_float_pixels_into_pbo(
        &mut self,
        layout: &PboReadbackLayout,
        buffer_id: u32,
        packed_w: usize,
        rows: usize,
        client_fmt: u32,
        gl_type: u32,
    ) -> crate::Result<()> {
        let &PboReadbackLayout {
            stride,
            tight_row,
            needed,
            row_length,
        } = layout;
        // SAFETY: the caller's contract; every span written lies within `needed`,
        // which `plan_pbo_readback` checked against the PBO's allocation.
        unsafe {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, buffer_id);
            edgefirst_gl::gl::ReadBuffer(edgefirst_gl::gl::COLOR_ATTACHMENT0);
            if let Some(pixels) = row_length {
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, pixels);
            }
            edgefirst_gl::gl::ReadPixels(
                0,
                0,
                packed_w as i32,
                rows as i32,
                client_fmt,
                gl_type,
                std::ptr::null_mut(),
            );
            if row_length.is_some() {
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, 0);
            }
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
        }
        let spread = stride > tight_row && row_length.is_none();
        // Waits for the readback into the destination PBO to complete, unless a
        // deferred convert leaves it to the batch's sync and nothing maps it below.
        self.wait_unless_deferred(spread, finish_via_fence);
        check_gl_error(function!(), line!())?;
        if spread {
            // SAFETY: GL thread, the read above has completed, and `needed` lies
            // within the allocation.
            unsafe {
                with_mapped_pbo(
                    buffer_id,
                    needed,
                    edgefirst_gl::gl::MAP_READ_BIT | edgefirst_gl::gl::MAP_WRITE_BIT,
                    "float PBO readback row placement",
                    |buf| spread_rows(buf, rows, tight_row, stride),
                )?;
            }
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }
}

impl GLProcessorST {
    /// Feed the RGBA8 source into `camera_normal_texture` for the float
    /// render paths, preferring zero-copy.
    ///
    /// Feed order:
    /// 1. **Import**: a `TensorMemory::DmaBuf` source on a zero-copy backend
    ///    is attached as the texture's storage through the
    ///    `get_or_create_egl_image(CacheKind::Src)` cache.
    /// 2. **PBO**: uploaded via `GL_PIXEL_UNPACK_BUFFER`.
    /// 3. **Upload**: CPU map + `TexImage2D`/`TexSubImage2D`.
    ///
    /// `EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT=1` disables arm 1.
    ///
    /// Resets the texture swizzle to identity first. After an import the
    /// texture's upload-tracking storage is forgotten, so a later upload
    /// allocates fresh storage.
    pub(super) fn feed_float_src(
        &mut self,
        src_u8: &Tensor<u8>,
        src_w: usize,
        src_h: usize,
        src_filter: i32,
        convert_span: &tracing::Span,
    ) -> crate::Result<FloatSrcFeed> {
        let src_tex_id = self.camera_normal_texture.id;
        unsafe {
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex_id);
            edgefirst_gl::gl::TexParameteri(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::TEXTURE_MIN_FILTER,
                src_filter,
            );
            edgefirst_gl::gl::TexParameteri(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::TEXTURE_MAG_FILTER,
                src_filter,
            );
            edgefirst_gl::gl::TexParameteri(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::TEXTURE_WRAP_S,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );
            edgefirst_gl::gl::TexParameteri(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::TEXTURE_WRAP_T,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );
            // Identity swizzle.
            for (swizzle, comp) in [
                (edgefirst_gl::gl::TEXTURE_SWIZZLE_R, edgefirst_gl::gl::RED),
                (edgefirst_gl::gl::TEXTURE_SWIZZLE_G, edgefirst_gl::gl::GREEN),
                (edgefirst_gl::gl::TEXTURE_SWIZZLE_B, edgefirst_gl::gl::BLUE),
                (edgefirst_gl::gl::TEXTURE_SWIZZLE_A, edgefirst_gl::gl::ALPHA),
            ] {
                edgefirst_gl::gl::TexParameteri(edgefirst_gl::gl::TEXTURE_2D, swizzle, comp as i32);
            }
        }
        let _ = src_tex_id; // bound above; arms below operate on the bound unit

        // ── Arm 1: zero-copy import of a Dma-backed source ──
        if !float_src_import_disabled()
            && self.gl_context.transfer_backend.is_zero_copy()
            && src_u8.memory() == TensorMemory::DmaBuf
        {
            let key = BufferImportKey::from_tensor(src_u8, PixelFormat::Rgba, false);
            match self.get_or_create_egl_image(CacheKind::Src, src_u8, PixelFormat::Rgba) {
                Ok(handle) => {
                    // SAFETY: camera_normal_texture is bound on the active unit; the
                    // handle's import is owned by the src cache.
                    let attach = unsafe {
                        self.camera_normal_texture
                            .bind_egl_image(&self.gl_context, key, handle)
                    };
                    // Checks glGetError after the attach; a failure falls back to the
                    // upload.
                    match attach.and_then(|_| check_gl_error(function!(), line!())) {
                        Ok(()) => {
                            // Forgets the upload-tracking storage.
                            self.camera_normal_texture.forget_storage();
                            self.convert_stats.src_imports += 1;
                            convert_span.record("src_feed", "import");
                            return Ok(FloatSrcFeed::Import);
                        }
                        Err(e) => {
                            self.camera_normal_texture.invalidate_egl_binding();
                            self.convert_stats.zero_copy_declines += 1;
                            log::debug!(
                                "float src zero-copy attach failed ({e:?}); uploading instead"
                            );
                        }
                    }
                }
                Err(e) => {
                    self.convert_stats.zero_copy_declines += 1;
                    log::debug!("float src zero-copy import failed ({e:?}); uploading instead");
                }
            }
        }

        // ── Arms 2 and 3: a PBO source (GL-internal copy), or a CPU map +
        // upload. This feed is RGBA-only: 4 bytes per pixel. The row length
        // comes from `unpack_row_length`.
        const SRC_BPP: usize = 4;
        let row_len_px =
            super::unpack_row_length(src_u8.effective_row_stride(), src_w, SRC_BPP, "RGBA")?;
        let pixels = super::SrcPixels::open(src_u8, &self.owned_pbos)?;
        pixels.record(&mut self.convert_stats, convert_span);
        let row_stride_b = if row_len_px == 0 {
            src_w * SRC_BPP
        } else {
            row_len_px as usize * SRC_BPP
        };
        let required = if src_h == 0 {
            0
        } else {
            (src_h - 1) * row_stride_b + src_w * SRC_BPP
        };
        let uploaded = unsafe {
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, row_len_px);
            let uploaded = self.camera_normal_texture.upload(
                edgefirst_gl::gl::TEXTURE_2D,
                src_w,
                src_h,
                edgefirst_gl::gl::RGBA,
                required,
                pixels.source(),
            );
            // Resets `UNPACK_ROW_LENGTH` on both outcomes.
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, 0);
            uploaded
        };
        uploaded?;
        Ok(match pixels {
            super::SrcPixels::Pbo { .. } => FloatSrcFeed::Pbo,
            super::SrcPixels::Mapped(_) => FloatSrcFeed::Upload,
        })
    }

    /// Run the float full-screen-quad draw used by both the PBO and zero-copy
    /// render paths.
    ///
    /// The caller must have already put the source pixels in `src_tex_id` and
    /// bound the render target to the active FBO, complete. This sets the
    /// viewport to `(packed_w, packed_h)`, binds the program, sets the crop
    /// uniforms, the sample clamp `src_extent` (`render::sample_clamp_rect`)
    /// and `dst_image_size` when it is `Some`, binds the source to TEXTURE0,
    /// and draws the quad. No readback is performed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_float_quad(
        &mut self,
        program_id: u32,
        sampler_name: &CStr,
        src_tex_id: u32,
        packed_w: u32,
        packed_h: u32,
        src_rect_uv: [f32; 4],
        src_extent: [f32; 4],
        dst_rect_px: [f32; 4],
        pad_color: [f32; 4],
        dst_image_size: Option<(f32, f32)>,
    ) -> crate::Result<()> {
        // Resolves the uniform locations once per program.
        let locs = *self
            .float_quad_locs
            .entry(program_id)
            .or_insert_with(|| unsafe {
                let loc =
                    |name: &CStr| edgefirst_gl::gl::GetUniformLocation(program_id, name.as_ptr());
                super::FloatQuadLocs {
                    sampler: loc(sampler_name),
                    src_rect_uv: loc(c"src_rect_uv"),
                    src_extent: loc(c"src_extent"),
                    dst_rect_px: loc(c"dst_rect_px"),
                    pad_color: loc(c"pad_color"),
                    dst_image_size: loc(c"dst_image_size"),
                    norm_scale: loc(c"norm_scale"),
                    norm_offset: loc(c"norm_offset"),
                }
            });
        let (pos_vbo, uv_vbo) = self.ensure_float_quad_vbos();
        unsafe {
            edgefirst_gl::gl::Viewport(0, 0, packed_w as i32, packed_h as i32);
            edgefirst_gl::gl::UseProgram(program_id);

            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex_id);
            edgefirst_gl::gl::Uniform1i(locs.sampler, 0);
            edgefirst_gl::gl::Uniform4f(
                locs.src_rect_uv,
                src_rect_uv[0],
                src_rect_uv[1],
                src_rect_uv[2],
                src_rect_uv[3],
            );
            edgefirst_gl::gl::Uniform4f(
                locs.src_extent,
                src_extent[0],
                src_extent[1],
                src_extent[2],
                src_extent[3],
            );
            edgefirst_gl::gl::Uniform4f(
                locs.dst_rect_px,
                dst_rect_px[0],
                dst_rect_px[1],
                dst_rect_px[2],
                dst_rect_px[3],
            );
            let norm = self.float_norm;
            let pad_color = norm.pad.unwrap_or(pad_color);
            edgefirst_gl::gl::Uniform4f(
                locs.pad_color,
                pad_color[0],
                pad_color[1],
                pad_color[2],
                pad_color[3],
            );
            edgefirst_gl::gl::Uniform4fv(locs.norm_scale, 1, norm.scale.as_ptr());
            edgefirst_gl::gl::Uniform4fv(locs.norm_offset, 1, norm.offset.as_ptr());
            if let Some((w, h)) = dst_image_size {
                edgefirst_gl::gl::Uniform2f(locs.dst_image_size, w, h);
            }
            check_gl_error(function!(), line!())?;

            // Full-screen quad: NDC -1..1 mapped to the whole viewport with 0..1
            // texcoords, from the two STATIC_DRAW VBOs. The attrib pointers are
            // pointed at them for the draw and restored to the engine's dynamic
            // buffers after.
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, pos_vbo);
            edgefirst_gl::gl::VertexAttribPointer(
                self.vertex_buffer.buffer_index,
                3,
                edgefirst_gl::gl::FLOAT,
                edgefirst_gl::gl::FALSE,
                0,
                std::ptr::null(),
            );
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, uv_vbo);
            edgefirst_gl::gl::VertexAttribPointer(
                self.texture_buffer.buffer_index,
                2,
                edgefirst_gl::gl::FLOAT,
                edgefirst_gl::gl::FALSE,
                0,
                std::ptr::null(),
            );
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            let quad_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                quad_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                quad_index.as_ptr() as *const c_void,
            );
            // Restores the attrib pointers to the engine's dynamic buffers.
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::VertexAttribPointer(
                self.vertex_buffer.buffer_index,
                3,
                edgefirst_gl::gl::FLOAT,
                edgefirst_gl::gl::FALSE,
                0,
                std::ptr::null(),
            );
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::VertexAttribPointer(
                self.texture_buffer.buffer_index,
                2,
                edgefirst_gl::gl::FLOAT,
                edgefirst_gl::gl::FALSE,
                0,
                std::ptr::null(),
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Create and upload the constant full-screen quad VBOs used by
    /// [`Self::draw_float_quad`] on first use.
    fn ensure_float_quad_vbos(&mut self) -> (u32, u32) {
        if self.float_quad_pos_vbo == 0 {
            const QUAD_POS: [f32; 12] = [
                -1.0, 1.0, 0.0, // left top
                1.0, 1.0, 0.0, // right top
                1.0, -1.0, 0.0, // right bottom
                -1.0, -1.0, 0.0, // left bottom
            ];
            const QUAD_UV: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            unsafe {
                let mut ids = [0u32; 2];
                edgefirst_gl::gl::GenBuffers(2, ids.as_mut_ptr());
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, ids[0]);
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    std::mem::size_of_val(&QUAD_POS) as isize,
                    QUAD_POS.as_ptr() as *const c_void,
                    edgefirst_gl::gl::STATIC_DRAW,
                );
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, ids[1]);
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    std::mem::size_of_val(&QUAD_UV) as isize,
                    QUAD_UV.as_ptr() as *const c_void,
                    edgefirst_gl::gl::STATIC_DRAW,
                );
                self.float_quad_pos_vbo = ids[0];
                self.float_quad_uv_vbo = ids[1];
            }
        }
        (self.float_quad_pos_vbo, self.float_quad_uv_vbo)
    }

    /// Render an RGBA8 source into a float PBO destination.
    ///
    /// `path` selects the packed layout:
    ///
    /// * [`FloatRenderPath::PboF32Nhwc`]: F32 `Rgb`, logical `[H,W,3]`. The
    ///   render target is a single-channel `R32F` texture sized `(W*3, H)`,
    ///   read back as `(RED, FLOAT)`.
    /// * [`FloatRenderPath::PboF16Nchw`] / [`FloatRenderPath::PboF32Nchw`]:
    ///   `PlanarRgb`, logical `[3,H,W]`. The render target is an `RGBA16F` /
    ///   `RGBA32F` texture sized `(W/4, 3*H)` packing four contiguous planar
    ///   elements per texel.
    /// * [`FloatRenderPath::PboF16Nhwc`]: F16 `Rgb`, packed into an `RGBA16F`
    ///   texture sized `(W*3/4, H)`.
    ///
    /// Returns `NotSupported` for a path that is not a PBO path, a
    /// non-`Rgba` source, or any rotation/flip.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_float_to_pbo(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        path: FloatRenderPath,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        // `pixel_bytes` is what one texel of the pack format occupies.
        let (internal, client_fmt, gl_type, pixel_bytes) = match path {
            FloatRenderPath::PboF32Nhwc => (
                edgefirst_gl::gl::R32F,
                edgefirst_gl::gl::RED,
                edgefirst_gl::gl::FLOAT,
                4,
            ),
            FloatRenderPath::PboF16Nchw | FloatRenderPath::PboF16Nhwc => (
                edgefirst_gl::gl::RGBA16F,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::HALF_FLOAT,
                8,
            ),
            FloatRenderPath::PboF32Nchw => (
                edgefirst_gl::gl::RGBA32F,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::FLOAT,
                16,
            ),
            FloatRenderPath::ZeroCopyF16Nchw
            | FloatRenderPath::ZeroCopyF32Nchw
            | FloatRenderPath::ZeroCopyFloatNhwc
            | FloatRenderPath::ZeroCopyFloatRgba
            | FloatRenderPath::None => {
                return Err(crate::Error::NotSupported(
                    "GL float render-to-PBO: not a PBO path; using CPU fallback".to_string(),
                ));
            }
        };

        if rotation != Rotation::None || flip != Flip::None {
            return Err(crate::Error::NotSupported(
                "GL float render-to-PBO: rotation/flip not supported on the float path; \
                 using CPU fallback"
                    .to_string(),
            ));
        }

        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        if src_fmt != PixelFormat::Rgba {
            return Err(crate::Error::NotSupported(format!(
                "GL float render-to-PBO: source format must be Rgba, got {src_fmt:?}; \
                 using CPU fallback"
            )));
        }
        let src_w = src_u8.width().ok_or(Error::NotAnImage)?;
        let src_h = src_u8.height().ok_or(Error::NotAnImage)?;

        // Destination dimensions and PBO buffer id, by dtype.
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dst_buffer_id = match dst.dtype() {
            DType::F32 => float_pbo_buffer_id(dst.as_typed::<f32>().expect("dtype checked"))?,
            DType::F16 => float_pbo_buffer_id(dst.as_typed::<half::f16>().expect("dtype checked"))?,
            other => {
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-PBO: dst dtype must be F32 or F16, got {other:?}"
                )));
            }
        };

        // Packed render-target dimensions.
        let (packed_w, packed_h) = match path {
            FloatRenderPath::PboF32Nhwc => (dst_w * 3, dst_h),
            FloatRenderPath::PboF16Nchw => {
                let layout = edgefirst_tensor::packed_rgba16f_layout(
                    PixelFormat::PlanarRgb,
                    edgefirst_tensor::DType::F16,
                    dst_w,
                    dst_h,
                )
                .ok_or_else(|| {
                    crate::Error::NotSupported(format!(
                        "GL float render-to-PBO: F16 NCHW requires width divisible by 4, \
                         got {dst_w}; using CPU fallback"
                    ))
                })?;
                (layout.surface_w, layout.surface_h)
            }
            FloatRenderPath::PboF32Nchw => packed_planar_layout(dst_w as u32, dst_h as u32, 3)
                .map(|(w, h)| (w as usize, h as usize))
                .ok_or_else(|| {
                    crate::Error::NotSupported(format!(
                        "GL float render-to-PBO: F32 NCHW requires width divisible by 4, \
                         got {dst_w}; using CPU fallback"
                    ))
                })?,
            FloatRenderPath::PboF16Nhwc => packed_interleaved_layout(dst_w as u32, dst_h as u32)
                .map(|(w, h)| (w as usize, h as usize))
                .ok_or_else(|| {
                    crate::Error::NotSupported(format!(
                        "GL float render-to-PBO: F16 NHWC requires width divisible by 4, \
                         got {dst_w}; using CPU fallback"
                    ))
                })?,
            _ => unreachable!(),
        };

        // ── Destination pitch ──
        // Surface row `y` is destination row `y`: the read is tight at
        // `packed_w * pixel_bytes` and the destination spaces its rows at
        // `effective_row_stride()`.
        let layout = plan_pbo_readback(
            dst.effective_row_stride().unwrap_or(packed_w * pixel_bytes),
            dst.view_origin().is_some(),
            dst.plane_offset().unwrap_or(0),
            dst.capacity_bytes(),
            packed_w,
            packed_h,
            pixel_bytes,
        )?;

        // Uniforms from crop: `src_rect_uv` is normalized to source dims,
        // `dst_rect_px` is in single-plane pixel coords, `pad_color` is
        // normalized [0,1].
        let (src_rect_uv, dst_rect_px, pad_color) =
            float_crop_uniforms(&crop, src_w, src_h, dst_w, dst_h)?;

        let program_id = match path {
            FloatRenderPath::PboF32Nhwc => self.float_f32_nhwc_program.id,
            FloatRenderPath::PboF16Nchw | FloatRenderPath::PboF32Nchw => {
                self.float_f16_nchw_program.id
            }
            FloatRenderPath::PboF16Nhwc => self.float_nhwc_pack_program.id,
            _ => unreachable!(),
        };
        // The R32F shader samples `u_tex`; the packed shaders sample `src`.
        let sampler_name: &CStr = match path {
            FloatRenderPath::PboF32Nhwc => c"u_tex",
            _ => c"src",
        };

        let render_tex_id = self.float_render_texture.id;
        let src_tex_id = self.camera_normal_texture.id;

        // Source sampling filter, from `Interpolation`.
        let src_filter = self.source_filter() as i32;

        // ── Source RGBA8 feed: zero-copy import, PBO upload, or CPU upload;
        // see `feed_float_src`.
        let feed = self.feed_float_src(src_u8, src_w, src_h, src_filter, convert_span)?;
        // Maps the source rectangle onto the texture the feed covers and clamps
        // samples to it.
        let (src_rect_uv, src_extent) =
            self.float_src_mapping_for_feed(feed, src_rect_uv, src_u8, src_w, src_h);

        unsafe {
            // ── Float render texture + FBO ──
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, render_tex_id);
            super::super::core::set_tex_filter(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::NEAREST,
            );
            // Specifies the render-target storage only when the packed dims or
            // internal format change.
            let render_dims = (packed_w, packed_h, internal);
            if self.float_render_tex_dims != render_dims {
                edgefirst_gl::gl::TexImage2D(
                    edgefirst_gl::gl::TEXTURE_2D,
                    0,
                    internal as i32,
                    packed_w as i32,
                    packed_h as i32,
                    0,
                    client_fmt,
                    gl_type,
                    std::ptr::null(),
                );
                self.float_render_tex_dims = render_dims;
                // Drops the cached EGLImage binding of the reallocated texture.
                self.float_render_texture.invalidate_egl_binding();
            }

            self.convert_fbo.bind();
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                render_tex_id,
                0,
            );
            if let Err(fbo_status) = super::super::core::check_framebuffer_complete() {
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-PBO: FBO incomplete (0x{fbo_status:x}) for {path:?}; \
                     using CPU fallback"
                )));
            }
        }

        // ── Render the full-screen quad with the float program ──
        let dst_image_size = match path {
            FloatRenderPath::PboF16Nchw | FloatRenderPath::PboF32Nchw => {
                Some((dst_w as f32, dst_h as f32))
            }
            _ => None,
        };
        self.draw_float_quad(
            program_id,
            sampler_name,
            src_tex_id,
            packed_w as u32,
            packed_h as u32,
            src_rect_uv,
            src_extent,
            dst_rect_px,
            pad_color,
            dst_image_size,
        )?;

        // ── Readback into the destination PBO, row by row at its pitch ──
        // SAFETY: on the GL thread, with the float render target complete and
        // bound as the read framebuffer by the block above.
        unsafe {
            self.read_float_pixels_into_pbo(
                &layout,
                dst_buffer_id,
                packed_w,
                packed_h,
                client_fmt,
                gl_type,
            )?
        }
        convert_span.record("dst_feed", "pbo");
        Ok(())
    }

    /// Render an RGBA8 source into a zero-copy float destination, writing the
    /// packed surface into the platform's GPU buffer through an imported
    /// renderbuffer or texture.
    ///
    /// `path` selects the destination layout, which
    /// [`Self::render_float_to_zero_copy_tail`] turns into a surface geometry,
    /// an import format and a program:
    ///
    /// * [`FloatRenderPath::ZeroCopyF16Nchw`] / [`FloatRenderPath::ZeroCopyF32Nchw`]:
    ///   `PlanarRgb` / `PlanarRgba`, three or four planes packed into a
    ///   `(W/4, C*H)` RGBA16F / RGBA32F surface.
    /// * [`FloatRenderPath::ZeroCopyFloatNhwc`]: `Rgb`, interleaved into a
    ///   `(W*3/4, H)` surface.
    /// * [`FloatRenderPath::ZeroCopyFloatRgba`]: `Rgba`, one texel per pixel
    ///   into a `(W, H)` surface.
    ///
    /// Returns `Err(NotSupported)` when the driver rejects the float import or
    /// the FBO is incomplete.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_float_to_zero_copy(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        path: FloatRenderPath,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if rotation != Rotation::None || flip != Flip::None {
            return Err(crate::Error::NotSupported(
                "GL float render-to-DMA: rotation/flip not supported on the float path; \
                 using CPU fallback"
                    .to_string(),
            ));
        }

        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        if src_fmt != PixelFormat::Rgba {
            return Err(crate::Error::NotSupported(format!(
                "GL float render-to-DMA: source format must be Rgba, got {src_fmt:?}; \
                 using CPU fallback"
            )));
        }
        let src_w = src_u8.width().ok_or(Error::NotAnImage)?;
        let src_h = src_u8.height().ok_or(Error::NotAnImage)?;

        // Destination geometry for the crop uniforms.
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let (src_rect_uv, dst_rect_px, pad_color) =
            float_crop_uniforms(&crop, src_w, src_h, dst_w, dst_h)?;

        let src_tex_id = self.camera_normal_texture.id;
        let src_filter = self.source_filter() as i32;

        // ── Source RGBA8 feed; see `feed_float_src`.
        let feed = self.feed_float_src(src_u8, src_w, src_h, src_filter, convert_span)?;
        // Maps the source rectangle onto the texture the feed covers.
        let (src_rect_uv, src_extent) =
            self.float_src_mapping_for_feed(feed, src_rect_uv, src_u8, src_w, src_h);

        self.render_float_to_zero_copy_tail(
            src_tex_id,
            src_rect_uv,
            src_extent,
            dst_rect_px,
            pad_color,
            dst,
            path,
            convert_span,
        )
    }

    /// The destination half of the zero-copy float render: import the float
    /// destination as its packed render surface, attach it to the FBO, and
    /// draw `src_tex_id` (an RGBA8 texture already holding the source pixels)
    /// through the program that packs it.
    ///
    /// `path` decides the destination this accepts, the surface it packs it
    /// into, the import format and the program:
    ///
    /// | path | dst | surface | import | program |
    /// |---|---|---|---|---|
    /// | `ZeroCopyF16Nchw` | `PlanarRgb`/`PlanarRgba` F16 | `(W/4, C*H)` | `Rgba16161616F` | `float_f16_nchw_program` |
    /// | `ZeroCopyF32Nchw` | `PlanarRgb`/`PlanarRgba` F32 | `(W/4, C*H)` | `Rgba32323232F` | `float_f16_nchw_program` |
    /// | `ZeroCopyFloatNhwc` | `Rgb` F16/F32 | `(W*3/4, H)` | per dtype | `float_nhwc_pack_program` |
    /// | `ZeroCopyFloatRgba` | `Rgba` F16/F32 | `(W, H)` | per dtype | `float_rgba_program` |
    ///
    /// Any other `(path, format, dtype)` combination is `NotSupported`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_float_to_zero_copy_tail(
        &mut self,
        src_tex_id: u32,
        src_rect_uv: [f32; 4],
        src_extent: [f32; 4],
        dst_rect_px: [f32; 4],
        pad_color: [f32; 4],
        dst: &mut TensorDyn,
        path: FloatRenderPath,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
        let dst_dtype = dst.dtype();

        if dst.memory() != TensorMemory::DmaBuf {
            return Err(crate::Error::NotSupported(
                "GL float render-to-DMA: dst is not a zero-copy GPU buffer; using CPU fallback"
                    .to_string(),
            ));
        }

        // Surface geometry, program and the plane size (`dst_image_size`, read
        // by the planar packer).
        let (surface, program_id, dst_image_size) = match (path, dst_fmt, dst_dtype) {
            (
                FloatRenderPath::ZeroCopyF16Nchw,
                PixelFormat::PlanarRgb | PixelFormat::PlanarRgba,
                DType::F16,
            )
            | (
                FloatRenderPath::ZeroCopyF32Nchw,
                PixelFormat::PlanarRgb | PixelFormat::PlanarRgba,
                DType::F32,
            ) => (
                packed_planar_layout(dst_w as u32, dst_h as u32, dst_fmt.channels() as u32),
                self.float_f16_nchw_program.id,
                Some((dst_w as f32, dst_h as f32)),
            ),
            (FloatRenderPath::ZeroCopyFloatNhwc, PixelFormat::Rgb, DType::F16 | DType::F32) => (
                packed_interleaved_layout(dst_w as u32, dst_h as u32),
                self.float_nhwc_pack_program.id,
                None,
            ),
            (FloatRenderPath::ZeroCopyFloatRgba, PixelFormat::Rgba, DType::F16 | DType::F32) => (
                Some((dst_w as u32, dst_h as u32)),
                self.float_rgba_program.id,
                None,
            ),
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: {path:?} does not render a {dst_fmt:?}/{dst_dtype:?} \
                     destination; using CPU fallback"
                )));
            }
        };
        let (surface_w, surface_h) = surface.ok_or_else(|| {
            crate::Error::NotSupported(format!(
                "GL float render-to-DMA: {dst_fmt:?} packing requires width divisible by 4, \
                 got {dst_w}; using CPU fallback"
            ))
        })?;

        // The float import format follows the destination's element width.
        let packed = match dst_dtype {
            DType::F16 => super::super::platform::PackedImportFormat::Rgba16161616F,
            DType::F32 => super::super::platform::PackedImportFormat::Rgba32323232F,
            other => {
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: dst dtype must be F16 or F32, got {other:?}; \
                     using CPU fallback"
                )));
            }
        };

        // The typed destination the import and renderbuffer cache key need.
        match dst_dtype {
            DType::F16 => {
                let dst_t = dst.as_typed_mut::<half::f16>().expect("dtype checked");
                self.render_float_dst(
                    dst_t,
                    dst_fmt,
                    surface_w,
                    surface_h,
                    packed,
                    program_id,
                    src_tex_id,
                    src_rect_uv,
                    src_extent,
                    dst_rect_px,
                    pad_color,
                    dst_image_size,
                    convert_span,
                )
            }
            _ => {
                let dst_t = dst.as_typed_mut::<f32>().expect("dtype checked");
                self.render_float_dst(
                    dst_t,
                    dst_fmt,
                    surface_w,
                    surface_h,
                    packed,
                    program_id,
                    src_tex_id,
                    src_rect_uv,
                    src_extent,
                    dst_rect_px,
                    pad_color,
                    dst_image_size,
                    convert_span,
                )
            }
        }
    }

    /// Import `dst` as a `(surface_w, surface_h)` packed float render surface,
    /// attach it to the convert FBO and draw `src_tex_id` through
    /// `program_id`. Generic over the destination element type.
    #[allow(clippy::too_many_arguments)]
    fn render_float_dst<T>(
        &mut self,
        dst: &Tensor<T>,
        dst_fmt: PixelFormat,
        surface_w: u32,
        surface_h: u32,
        packed: super::super::platform::PackedImportFormat,
        program_id: u32,
        src_tex_id: u32,
        src_rect_uv: [f32; 4],
        src_extent: [f32; 4],
        dst_rect_px: [f32; 4],
        pad_color: [f32; 4],
        dst_image_size: Option<(f32, f32)>,
        convert_span: &tracing::Span,
    ) -> crate::Result<()>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        // ── Import the destination as a renderable float surface ──
        self.convert_fbo.bind();
        let dest_egl = self.get_or_create_egl_image_rgb(
            dst,
            dst_fmt,
            surface_w as usize,
            surface_h as usize,
            packed,
        )?;

        // Attach the import (renderbuffer when supported, else texture) to the
        // FBO.
        //
        // SAFETY: convert_fbo is bound above on the thread's current GL
        // context; rbo comes from this call's own cache and
        // dest_egl/self.render_texture stay valid for the FBO calls below.
        unsafe {
            match self.cached_dst_renderbuffer(dst, dst_fmt) {
                Some(rbo) => {
                    edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                    edgefirst_gl::gl::FramebufferRenderbuffer(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::RENDERBUFFER,
                        rbo,
                    );
                    if let Err(fbo_status) = super::super::core::check_framebuffer_complete() {
                        edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                        return Err(crate::Error::NotSupported(format!(
                            "GL float render-to-DMA: FBO incomplete (0x{fbo_status:x}) for the \
                             {packed:?} import; using CPU fallback"
                        )));
                    }
                }
                None => {
                    Self::attach_import_target(
                        &self.gl_context,
                        &mut self.render_texture,
                        dest_egl,
                    )?;
                }
            }
            check_gl_error(function!(), line!())?;
        }

        // ── Float draw: viewport = the packed surface, program, uniforms,
        // quad ──
        let drawn = self.draw_float_quad(
            program_id,
            c"src",
            src_tex_id,
            surface_w,
            surface_h,
            src_rect_uv,
            src_extent,
            dst_rect_px,
            pad_color,
            dst_image_size,
        );

        // Waits for the render into the destination buffer to complete unless
        // a deferred convert leaves it to the batch's sync.
        self.wait_unless_deferred(drawn.is_err(), finish_via_fence);
        drawn?;
        check_gl_error(function!(), line!())?;
        convert_span.record("dst_feed", "zero_copy");
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{float_pbo_buffer_id, packed_interleaved_layout, packed_planar_layout};
    use crate::Error;

    /// Checks that `packed_interleaved_layout` accepts only widths divisible by 4.
    #[test]
    fn packed_interleaved_layout_takes_whole_texels_only() {
        assert_eq!(packed_interleaved_layout(640, 480), Some((480, 480)));
        assert_eq!(packed_interleaved_layout(4, 1), Some((3, 1)));
        for w in [1u32, 2, 3, 5, 6, 7] {
            assert_eq!(
                packed_interleaved_layout(w, 16),
                None,
                "{w}x3 is not a whole number of four-element texels"
            );
        }
        // The planar layout.
        assert_eq!(packed_planar_layout(640, 480, 3), Some((160, 1440)));
        assert_eq!(packed_planar_layout(6, 16, 3), None);
    }

    #[test]
    fn pbo_buffer_id_rejects_non_pbo_tensor() {
        // A Mem-backed tensor is not PBO-backed.
        let t = edgefirst_tensor::Tensor::<f32>::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = float_pbo_buffer_id(&t).unwrap_err();
        assert!(
            matches!(err, Error::OpenGl(_)),
            "expected OpenGl error, got {err:?}"
        );
    }
}
