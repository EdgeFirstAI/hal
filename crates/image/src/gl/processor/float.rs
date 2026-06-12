// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Linux GL float (F16/F32) preprocessing paths.
//!
//! This is a child module of [`super`] (`gl::processor`) so the
//! `impl GLProcessorST` block here can access the parent's PRIVATE fields
//! (`convert_fbo`, `float_render_texture`, the float programs, EGLImage caches,
//! capability flags, …) via the same-module-tree visibility rules. None of
//! those fields are promoted to `pub(super)` for this move.
//!
//! Items grouped here:
//! * [`float_render_support`] — reportable float-render capability.
//! * [`dma_f16_packed_layout`] — Linux DMA packed-surface geometry (thin
//!   wrapper over [`edgefirst_tensor::packed_rgba16f_layout`]).
//! * The [`GLProcessorST`] float render methods: `convert_float_to_pbo`,
//!   `convert_float_to_zero_copy`, `upload_float_src`, `draw_float_quad`.

use std::ffi::{c_void, CStr};

use super::super::core::float_crop_uniforms;
use super::super::platform::GlPlatform;
use super::{check_gl_error, dyn_to_u8_src, GLProcessorST};
use crate::{Error, Flip, ResolvedCrop, Rotation};
use edgefirst_tensor::{PixelFormat, TensorDyn, TensorTrait};

// The float render-path decision (`FloatRenderPath` + `classify_float_render`)
// is defined once in the cfg-agnostic `gl::float_dispatch` module so the Linux
// and macOS backends share a single source of truth. Re-imported here so this
// module's call sites and the `gl::tests` unit tests keep using the
// `processor::{FloatRenderPath, classify_float_render}` paths unchanged.
pub(in super::super) use super::super::float_dispatch::{classify_float_render, FloatRenderPath};

/// Decide reportable float render support. Vivante GC7000UL float readback is
/// 170-320 ms (probe-measured) so GL float is refused there; `ImageProcessor`
/// falls back to CPU float output (normalized to `[0, 1]`), not u8.
pub(in super::super) fn float_render_support(
    is_vivante: bool,
    f32_ext: bool,
    f16_ext: bool,
) -> crate::RenderDtypeSupport {
    if is_vivante {
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

/// Packed RGBA16F surface geometry for the F16 NCHW DMA/PBO destination.
///
/// The logical destination is `[3, H, W]` f16 (`PlanarRgb`). It is packed into
/// an `RGBA16F` surface where each texel holds four contiguous planar f16
/// elements, giving a GL-visible surface of `(W/4, 3*H)` texels at 8 bytes per
/// texel (`bpp = 8`) and a tight row pitch of `(W/4) * 8` bytes.
///
/// Returns `None` when `W` is not divisible by 4 (the packing requires whole
/// RGBA16F texels per row), signalling the caller to fall back to CPU.
///
/// Thin wrapper over [`edgefirst_tensor::packed_rgba16f_layout`] (the canonical
/// geometry source) preserving the historical `(surface_w, surface_h, pitch)`
/// tuple shape used by the Linux DMA path.
pub(in super::super) fn dma_f16_packed_layout(w: u32, h: u32) -> Option<(u32, u32, u32)> {
    let layout = edgefirst_tensor::packed_rgba16f_layout(
        PixelFormat::PlanarRgb,
        edgefirst_tensor::DType::F16,
        w as usize,
        h as usize,
    )?;
    Some((
        layout.surface_w as u32,
        layout.surface_h as u32,
        layout.pitch as u32,
    ))
}

/// Fetch the PBO buffer id of a float PBO destination tensor.
///
/// Shared by the `TensorDyn::F32`/`F16` arms of `convert_float_to_pbo`:
/// resolves the tensor's PBO (`NotSupported`-equivalent `OpenGl` error when
/// the tensor is not PBO-backed) and rejects a currently-mapped PBO.
fn float_pbo_buffer_id<T>(t: &edgefirst_tensor::Tensor<T>) -> crate::Result<u32>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
{
    let pbo = t.as_pbo().ok_or_else(|| {
        crate::Error::OpenGl("convert_float_to_pbo: dst is not a PBO tensor".to_string())
    })?;
    if pbo.is_mapped() {
        return Err(crate::Error::OpenGl(
            "Cannot convert to a mapped PBO tensor".to_string(),
        ));
    }
    Ok(pbo.buffer_id())
}

/// Block until all previously-issued GL commands have completed, using a fence
/// sync instead of a full-queue `glFinish` drain.
///
/// This preserves the same blocking-completion contract the float paths relied
/// on (`glFinish`): when this returns, the readback into the mapped PBO / the
/// render into the dma-buf is guaranteed complete, so the result is safe to
/// read. The difference is scope: `glFenceSync` + `glClientWaitSync` waits only
/// for the commands issued before the fence rather than draining the entire GPU
/// queue, so the producer thread can enqueue the next frame's work sooner.
///
/// `GL_SYNC_FLUSH_COMMANDS_BIT` is passed so the wait flushes the command
/// buffer and cannot deadlock waiting on un-submitted work. On any failure
/// (fence creation failed, timeout, or `GL_WAIT_FAILED`) it falls back to a
/// blocking `glFinish` so the completion guarantee is never silently dropped.
///
/// # Safety
/// Must be called on the thread owning the current GL context.
pub(super) unsafe fn finish_via_fence() {
    // 1 second, in nanoseconds — generous; a healthy convert completes in well
    // under a millisecond and never reaches the timeout.
    const TIMEOUT_NS: u64 = 1_000_000_000;
    let sync = gls::gl::FenceSync(gls::gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
    if sync.is_null() {
        // Fence could not be created; preserve the completion guarantee.
        gls::gl::Finish();
        return;
    }
    let status = gls::gl::ClientWaitSync(sync, gls::gl::SYNC_FLUSH_COMMANDS_BIT, TIMEOUT_NS);
    gls::gl::DeleteSync(sync);
    match status {
        s if s == gls::gl::ALREADY_SIGNALED || s == gls::gl::CONDITION_SATISFIED => {}
        // Timeout expired or the wait failed: fall back to a blocking drain so
        // the caller never proceeds on an incomplete readback/render.
        _ => gls::gl::Finish(),
    }
}

impl GLProcessorST {
    /// Upload an RGBA8 source image into `camera_normal_texture` for the float
    /// render paths (PBO and DMA F16).
    ///
    /// Resets the texture swizzle to identity (a prior Grey/planar conversion
    /// may have left it non-identity) and uploads either directly from the
    /// source PBO (`src_pbo_id`, zero CPU copy) or from a mapped CPU slice
    /// (`src_cpu_pixels`). Exactly one of `src_pbo_id` / `src_cpu_pixels` must
    /// be `Some`; the PBO branch is preferred when both are present.
    pub(super) fn upload_float_src(
        &mut self,
        src_tex_id: u32,
        src_w: usize,
        src_h: usize,
        src_filter: i32,
        src_pbo_id: Option<u32>,
        src_cpu_pixels: Option<&[u8]>,
    ) {
        unsafe {
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex_id);
            gls::gl::TexParameteri(gls::gl::TEXTURE_2D, gls::gl::TEXTURE_MIN_FILTER, src_filter);
            gls::gl::TexParameteri(gls::gl::TEXTURE_2D, gls::gl::TEXTURE_MAG_FILTER, src_filter);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_WRAP_S,
                gls::gl::CLAMP_TO_EDGE as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_WRAP_T,
                gls::gl::CLAMP_TO_EDGE as i32,
            );
            // Identity swizzle (a prior Grey/planar conversion may have left
            // TEXTURE_SWIZZLE_* in a non-identity state on this texture).
            for (swizzle, comp) in [
                (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                (gls::gl::TEXTURE_SWIZZLE_A, gls::gl::ALPHA),
            ] {
                gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, comp as i32);
            }
            if let Some(buffer_id) = src_pbo_id {
                // Upload directly from the source PBO (zero CPU copy). The PBO
                // path allocates the same RGBA8 storage that `update_texture`
                // would (internalformat RGBA, GL_RGBA, UNSIGNED_BYTE), so the
                // two paths share `camera_normal_texture`'s size/format cache.
                // On the steady-state video path (fixed input size) reuse the
                // existing storage with `TexSubImage2D` (PBO-bound, NULL data)
                // instead of reallocating with `TexImage2D` every frame.
                gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, buffer_id);
                let cache = &mut self.camera_normal_texture;
                let needs_alloc = cache.target != gls::gl::TEXTURE_2D
                    || cache.width != src_w
                    || cache.height != src_h
                    || cache.format != gls::gl::RGBA;
                if needs_alloc {
                    gls::gl::TexImage2D(
                        gls::gl::TEXTURE_2D,
                        0,
                        gls::gl::RGBA as i32,
                        src_w as i32,
                        src_h as i32,
                        0,
                        gls::gl::RGBA,
                        gls::gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                    // Record the true storage state so the cache reflects
                    // reality: a later interleaved u8 `update_texture` (same
                    // dims/format) will correctly take its TexSubImage2D fast
                    // path rather than be forced to reallocate. TexImage2D
                    // reallocated storage, so any EGLImage binding is stale.
                    cache.target = gls::gl::TEXTURE_2D;
                    cache.width = src_w;
                    cache.height = src_h;
                    cache.format = gls::gl::RGBA;
                    cache.invalidate_egl_binding();
                } else {
                    gls::gl::TexSubImage2D(
                        gls::gl::TEXTURE_2D,
                        0,
                        0,
                        0,
                        src_w as i32,
                        src_h as i32,
                        gls::gl::RGBA,
                        gls::gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                }
                gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, 0);
            } else {
                let pixels = src_cpu_pixels.expect("non-PBO source mapped by caller");
                self.camera_normal_texture.update_texture(
                    gls::gl::TEXTURE_2D,
                    src_w,
                    src_h,
                    gls::gl::RGBA,
                    pixels,
                );
            }
        }
    }

    /// Run the shared float full-screen-quad draw used by both the PBO and DMA
    /// F16 render paths.
    ///
    /// The caller must have already (a) uploaded the source via
    /// [`Self::upload_float_src`] and (b) bound the render target (float
    /// texture for PBO, EGLImage renderbuffer for DMA) to the active FBO and
    /// confirmed it complete. This sets the viewport to `(packed_w, packed_h)`,
    /// binds the program, sets the crop uniforms (and `dst_image_size` when
    /// `dst_image_size` is `Some`, required by the F16 NCHW shader), binds the
    /// source to TEXTURE0, and draws the quad. No readback is performed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_float_quad(
        &mut self,
        program_id: u32,
        sampler_name: &CStr,
        src_tex_id: u32,
        packed_w: u32,
        packed_h: u32,
        src_rect_uv: [f32; 4],
        dst_rect_px: [f32; 4],
        pad_color: [f32; 3],
        dst_image_size: Option<(f32, f32)>,
    ) -> crate::Result<()> {
        unsafe {
            gls::gl::Viewport(0, 0, packed_w as i32, packed_h as i32);
            gls::gl::UseProgram(program_id);

            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex_id);
            let loc_sampler = gls::gl::GetUniformLocation(program_id, sampler_name.as_ptr());
            gls::gl::Uniform1i(loc_sampler, 0);

            let loc_src_rect = gls::gl::GetUniformLocation(program_id, c"src_rect_uv".as_ptr());
            gls::gl::Uniform4f(
                loc_src_rect,
                src_rect_uv[0],
                src_rect_uv[1],
                src_rect_uv[2],
                src_rect_uv[3],
            );
            let loc_dst_rect = gls::gl::GetUniformLocation(program_id, c"dst_rect_px".as_ptr());
            gls::gl::Uniform4f(
                loc_dst_rect,
                dst_rect_px[0],
                dst_rect_px[1],
                dst_rect_px[2],
                dst_rect_px[3],
            );
            let loc_pad = gls::gl::GetUniformLocation(program_id, c"pad_color".as_ptr());
            gls::gl::Uniform3f(loc_pad, pad_color[0], pad_color[1], pad_color[2]);
            if let Some((w, h)) = dst_image_size {
                let loc_size = gls::gl::GetUniformLocation(program_id, c"dst_image_size".as_ptr());
                gls::gl::Uniform2f(loc_size, w, h);
            }
            check_gl_error(function!(), line!())?;

            // Full-screen quad: NDC -1..1 mapped to the whole viewport with
            // 0..1 texcoords. The float shaders ignore the interpolated
            // texcoords (they derive sampling from gl_FragCoord + uniforms),
            // but a valid quad is still needed to rasterize every fragment.
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
            let quad_pos: [f32; 12] = [
                -1.0, 1.0, 0.0, // left top
                1.0, 1.0, 0.0, // right top
                1.0, -1.0, 0.0, // right bottom
                -1.0, -1.0, 0.0, // left bottom
            ];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * quad_pos.len()) as isize,
                quad_pos.as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            let quad_uv: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * quad_uv.len()) as isize,
                quad_uv.as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            let quad_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                quad_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                quad_index.as_ptr() as *const c_void,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Render an RGBA8 source into a float PBO destination.
    ///
    /// Two packed layouts are produced, selected by `path`:
    ///
    /// * [`FloatRenderPath::PboF32Nhwc`] — F32 `Rgb`, logical `[H,W,3]`.
    ///   Render target is a single-channel `R32F` texture sized `(W*3, H)`;
    ///   the shader emits one channel per fragment and `glReadPixels` reads
    ///   `(RED, FLOAT)` straight into the PBO.
    /// * [`FloatRenderPath::PboF16Nchw`] — F16 `PlanarRgb`, logical `[3,H,W]`.
    ///   Render target is an `RGBA16F` texture sized `(W/4, 3*H)` packing four
    ///   contiguous planar f16 elements per texel; readback is
    ///   `(RGBA, HALF_FLOAT)`.
    ///
    /// [`FloatRenderPath::ZeroCopyF16Nchw`] and [`FloatRenderPath::None`] are not
    /// handled here — they return `NotSupported` so `convert()` falls back to
    /// CPU. Likewise,
    /// non-`Rgba` sources and any rotation/flip fall back: the
    /// float shaders are uniform-driven and normalize `[0,1]` via the texture
    /// fetch, but do not implement rotation/flip.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_float_to_pbo(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        path: FloatRenderPath,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        // Only the two PBO float paths are implemented here.
        let (internal, client_fmt, gl_type) = match path {
            FloatRenderPath::PboF32Nhwc => (gls::gl::R32F, gls::gl::RED, gls::gl::FLOAT),
            FloatRenderPath::PboF16Nchw => (gls::gl::RGBA16F, gls::gl::RGBA, gls::gl::HALF_FLOAT),
            FloatRenderPath::ZeroCopyF16Nchw | FloatRenderPath::None => {
                return Err(crate::Error::NotSupported(
                    "GL float render-to-PBO: only PBO F32 NHWC / F16 NCHW are implemented; \
                     using CPU fallback"
                        .to_string(),
                ));
            }
        };

        // Rotation/flip are not implemented by the float shaders.
        if rotation != Rotation::None || flip != Flip::None {
            return Err(crate::Error::NotSupported(
                "GL float render-to-PBO: rotation/flip not supported on the float path; \
                 using CPU fallback"
                    .to_string(),
            ));
        }

        // Source must be RGBA8.
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
        let dst_len = dst.size() / dst.dtype().size();
        let dst_buffer_id = match dst {
            TensorDyn::F32(t) => float_pbo_buffer_id(t)?,
            TensorDyn::F16(t) => float_pbo_buffer_id(t)?,
            other => {
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-PBO: dst dtype must be F32 or F16, got {:?}",
                    other.dtype()
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
            _ => unreachable!(),
        };

        // Uniforms from crop — identical contract to the macOS IOSurface path.
        // `src_rect_uv` is normalized to source dims; `dst_rect_px` is in
        // single-plane pixel coords; `pad_color` is normalized [0,1].
        let (src_rect_uv, dst_rect_px, pad_color) =
            float_crop_uniforms(&crop, src_w, src_h, dst_w, dst_h)?;

        let program_id = match path {
            FloatRenderPath::PboF32Nhwc => self.float_f32_nhwc_program.id,
            FloatRenderPath::PboF16Nchw => self.float_f16_nchw_program.id,
            _ => unreachable!(),
        };
        // F32 shader samples `u_tex`; F16 shader samples `src`.
        let sampler_name: &CStr = match path {
            FloatRenderPath::PboF32Nhwc => c"u_tex",
            FloatRenderPath::PboF16Nchw => c"src",
            _ => unreachable!(),
        };

        let render_tex_id = self.float_render_texture.id;
        let src_tex_id = self.camera_normal_texture.id;

        // PBO sources MUST be uploaded via a GL_PIXEL_UNPACK_BUFFER binding,
        // not `map()`: this method runs on the GL worker thread, and a PBO
        // `map()` sends a PboMap message back to this same thread, deadlocking.
        // Non-PBO (Mem/Dma) sources are mapped directly (no message channel).
        let src_pbo_id = src_u8.as_pbo().map(|p| p.buffer_id());
        let src_cpu_pixels = if src_pbo_id.is_none() {
            Some(src_u8.map()?)
        } else {
            None
        };

        // Source sampling filter. Both float shaders sample at output-pixel
        // centers (they add +0.5 to the integer output index before mapping to
        // the source UV), so LINEAR gives a correct bilinear resize on both
        // paths. LINEAR on the RGBA8 source is unconditionally supported —
        // GL_OES_texture_float_linear is irrelevant here because we filter the
        // u8 source texture, not the float render target.
        let src_filter = gls::gl::LINEAR as i32;

        // ── Source RGBA8 texture upload (shared with the DMA F16 path) ──
        self.upload_float_src(
            src_tex_id,
            src_w,
            src_h,
            src_filter,
            src_pbo_id,
            src_cpu_pixels.as_deref(),
        );

        unsafe {
            // ── Float render texture + FBO ──
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, render_tex_id);
            super::super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::NEAREST);
            // Only (re)spec the render-target storage when the packed dims or
            // internal format change (mirrors `proto_tex_dims` / Texture's
            // size cache). On the steady-state fixed-input video path this is
            // unchanged every frame, so we reuse the existing storage and skip
            // the per-frame `TexImage2D` reallocation entirely.
            let render_dims = (packed_w, packed_h, internal);
            if self.float_render_tex_dims != render_dims {
                gls::gl::TexImage2D(
                    gls::gl::TEXTURE_2D,
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
                // The float render texture storage was just (re)allocated; mark
                // the cached struct fields so a later EGLImage path won't assume
                // a stale binding/size on this texture object.
                self.float_render_texture.invalidate_egl_binding();
            }

            self.convert_fbo.bind();
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                render_tex_id,
                0,
            );
            if let Err(fbo_status) = super::super::core::check_framebuffer_complete() {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-PBO: FBO incomplete (0x{fbo_status:x}) for {path:?}; \
                     using CPU fallback"
                )));
            }
        }

        // ── Render the full-screen quad with the float program (shared) ──
        let dst_image_size = match path {
            FloatRenderPath::PboF16Nchw => Some((dst_w as f32, dst_h as f32)),
            _ => None,
        };
        self.draw_float_quad(
            program_id,
            sampler_name,
            src_tex_id,
            packed_w as u32,
            packed_h as u32,
            src_rect_uv,
            dst_rect_px,
            pad_color,
            dst_image_size,
        )?;

        unsafe {
            // ── Readback into the destination PBO ──
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                packed_w as i32,
                packed_h as i32,
                client_fmt,
                gl_type,
                (dst_len * dst.dtype().size()) as i32,
                std::ptr::null_mut(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            // Wait for the readback into the destination PBO to complete before
            // returning (same contract as glFinish, scoped to a fence).
            finish_via_fence();
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Render an RGBA8 source into a DMA-backed F16 `PlanarRgb` destination,
    /// writing the NCHW-packed RGBA16F surface directly into the dma-buf via an
    /// EGLImage-backed renderbuffer (zero-copy — no `glReadPixels`).
    ///
    /// Mirrors the F16 PBO path ([`Self::convert_float_to_pbo`] /
    /// `FloatRenderPath::PboF16Nchw`): identical source upload, shader, crop
    /// uniforms, viewport `(W/4, 3*H)` and full-screen quad. The only
    /// difference is the render target — instead of a float texture read back
    /// into a PBO, the GPU renders straight into the destination dma-buf
    /// imported as a `DrmFourcc::Abgr16161616f` (bpp=8) EGLImage.
    ///
    /// This is the V3D / Mali zero-copy float path. On any driver that rejects
    /// the RGBA16F dma-buf import (e.g. Vivante, desktop NVIDIA) or returns an
    /// incomplete FBO, returns `Err(NotSupported)` so `convert()` degrades
    /// gracefully to the CPU. Never panics.
    pub(super) fn convert_float_to_zero_copy(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        // Rotation/flip are not implemented by the float shaders.
        if rotation != Rotation::None || flip != Flip::None {
            return Err(crate::Error::NotSupported(
                "GL float render-to-DMA: rotation/flip not supported on the float path; \
                 using CPU fallback"
                    .to_string(),
            ));
        }

        // Source must be RGBA8.
        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        if src_fmt != PixelFormat::Rgba {
            return Err(crate::Error::NotSupported(format!(
                "GL float render-to-DMA: source format must be Rgba, got {src_fmt:?}; \
                 using CPU fallback"
            )));
        }
        let src_w = src_u8.width().ok_or(Error::NotAnImage)?;
        let src_h = src_u8.height().ok_or(Error::NotAnImage)?;

        // Destination must be an F16 PlanarRgb DMA tensor.
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
        if dst_fmt != PixelFormat::PlanarRgb {
            return Err(crate::Error::NotSupported(format!(
                "GL float render-to-DMA: dst format must be PlanarRgb, got {dst_fmt:?}; \
                 using CPU fallback"
            )));
        }
        let dst_f16 = match dst {
            TensorDyn::F16(t) => t,
            other => {
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: dst dtype must be F16, got {:?}; using CPU fallback",
                    other.dtype()
                )));
            }
        };
        if dst_f16.memory() != edgefirst_tensor::TensorMemory::Dma {
            return Err(crate::Error::NotSupported(
                "GL float render-to-DMA: dst is not a zero-copy GPU buffer; using CPU fallback"
                    .to_string(),
            ));
        }

        // Packed RGBA16F surface geometry; requires W % 4 == 0.
        let (surface_w, surface_h, _pitch) = dma_f16_packed_layout(dst_w as u32, dst_h as u32)
            .ok_or_else(|| {
                crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: F16 NCHW requires width divisible by 4, \
                     got {dst_w}; using CPU fallback"
                ))
            })?;

        // Crop uniforms — identical contract to the F16 PBO path.
        let (src_rect_uv, dst_rect_px, pad_color) =
            float_crop_uniforms(&crop, src_w, src_h, dst_w, dst_h)?;

        let program_id = self.float_f16_nchw_program.id;
        let src_tex_id = self.camera_normal_texture.id;

        // PBO sources upload via GL_PIXEL_UNPACK_BUFFER (no map() — avoids a
        // GL-thread deadlock); non-PBO sources map to CPU directly.
        let src_pbo_id = src_u8.as_pbo().map(|p| p.buffer_id());
        let src_cpu_pixels = if src_pbo_id.is_none() {
            Some(src_u8.map()?)
        } else {
            None
        };
        let src_filter = gls::gl::LINEAR as i32;

        // ── Source RGBA8 texture upload (shared with the PBO F16 path) ──
        self.upload_float_src(
            src_tex_id,
            src_w,
            src_h,
            src_filter,
            src_pbo_id,
            src_cpu_pixels.as_deref(),
        );

        // ── Import the destination dma-buf as a renderable RGBA16F EGLImage ──
        // Packed surface (W/4, 3*H), bpp=8. The tensor's natural planar f16 row
        // stride (W * 2 bytes) equals the packed pitch ((W/4) * 8), so
        // create_egl_image_with_dims derives the correct pitch automatically.
        self.convert_fbo.bind();
        let dest_egl = self.get_or_create_egl_image_rgb(
            dst_f16,
            PixelFormat::PlanarRgb,
            surface_w as usize,
            surface_h as usize,
            super::super::platform::PackedImportFormat::Rgba16161616F,
        )?;

        // Attach the EGLImage (renderbuffer when supported, else texture) to the
        // FBO, mirroring the u8 packed-RGB DMA destination path.
        unsafe {
            match self.cached_dst_renderbuffer(dst_f16, PixelFormat::PlanarRgb) {
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
                    super::super::core::set_tex_filter(gls::gl::TEXTURE_2D, gls::gl::NEAREST);
                    // Platform attach (EGLImage target on Linux,
                    // eglBindTexImage on macOS — a raw OES call there
                    // silently no-ops on a pbuffer handle and leaves the
                    // FBO incomplete).
                    super::super::platform::Platform::attach_tex_image_2d(
                        &self.gl_context,
                        dest_egl,
                    )?;
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        self.render_texture.id,
                        0,
                    );
                    // The texture-path binding above mutated render_texture's GL state
                    // without going through bind_egl_image's cache; drop any stale
                    // cached binding so future convert calls re-bind correctly.
                    self.render_texture.invalidate_egl_binding();
                }
            }

            if let Err(fbo_status) = super::super::core::check_framebuffer_complete() {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: FBO incomplete (0x{fbo_status:x}) for the RGBA16F \
                     dma-buf import; using CPU fallback"
                )));
            }
            check_gl_error(function!(), line!())?;
        }

        // ── Shared F16 draw: viewport (W/4, 3*H), program, uniforms, quad ──
        self.draw_float_quad(
            program_id,
            c"src",
            src_tex_id,
            surface_w,
            surface_h,
            src_rect_uv,
            dst_rect_px,
            pad_color,
            Some((dst_w as f32, dst_h as f32)),
        )?;

        // Zero-copy: the GPU wrote straight into the dma-buf. No readback, but
        // we must still wait for the render to complete before returning so the
        // dma-buf is safe for the consumer to read (same contract as glFinish,
        // scoped to a fence rather than a full-queue drain).
        unsafe {
            finish_via_fence();
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }
}

// `float_pbo_buffer_id` is pure (no GL state), so it is unit-testable without a
// GPU. The GL draw/upload/readback methods above need a real V3D/Mali device and
// are covered on-target, not in CI. (`float_crop_uniforms` moved to `gl::core`
// and is tested there.)
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::float_pbo_buffer_id;
    use crate::Error;

    #[test]
    fn pbo_buffer_id_rejects_non_pbo_tensor() {
        // A plain Mem-backed tensor is not PBO-backed → OpenGl error rather
        // than a panic / bogus buffer id.
        let t = edgefirst_tensor::Tensor::<f32>::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let err = float_pbo_buffer_id(&t).unwrap_err();
        assert!(
            matches!(err, Error::OpenGl(_)),
            "expected OpenGl error, got {err:?}"
        );
    }
}
