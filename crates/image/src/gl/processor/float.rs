// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL float (F16/F32) preprocessing paths (Linux DMA-BUF + macOS IOSurface
//! zero-copy targets, Tegra PBO targets).
//!
//! This is a child module of [`super`] (`gl::processor`) so the
//! `impl GLProcessorST` block here can access the parent's PRIVATE fields
//! (`convert_fbo`, `float_render_texture`, the float programs, EGLImage caches,
//! capability flags, …) via the same-module-tree visibility rules. None of
//! those fields are promoted to `pub(super)` for this move.
//!
//! Items grouped here:
//! * [`float_render_support`] — reportable float-render capability.
//! * [`packed_planar_layout`] — packed zero-copy surface geometry for a
//!   planar float destination.
//! * The [`GLProcessorST`] float render methods: `convert_float_to_pbo`,
//!   `convert_float_to_zero_copy`, `render_float_to_zero_copy_tail` (the
//!   dst-import/draw half, shared with the fused NV→PlanarF16 two-pass —
//!   its source is the GPU-resident intermediate texture),
//!   `feed_float_src`, `draw_float_quad`.

use std::ffi::{c_void, CStr};

use super::super::cache::{BufferImportKey, CacheKind};
use super::super::core::float_crop_uniforms;
use super::super::platform::GlPlatform;
use super::{
    check_gl_error, dyn_to_u8_src, plan_pbo_readback, spread_rows, with_mapped_pbo, GLProcessorST,
    PboReadbackLayout,
};
use crate::{Error, Flip, ResolvedCrop, Rotation};
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorDyn, TensorMemory, TensorTrait};

// The float render-path decision (`FloatRenderPath` + `classify_float_render`)
// is defined once in the cfg-agnostic `gl::float_dispatch` module so the Linux
// and macOS backends share a single source of truth. Re-imported here so this
// module's call sites and the `gl::tests` unit tests keep using the
// `processor::{FloatRenderPath, classify_float_render}` paths unchanged.
pub(in super::super) use super::super::float_dispatch::{classify_float_render, FloatRenderPath};

/// How the float source reached the GPU this frame — the zero-copy
/// telemetry primitive (see `ConvertStats`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum FloatSrcFeed {
    /// Zero-copy: the source's Dma buffer is the texture's storage.
    Import,
    /// GL-internal copy from the source PBO (no CPU visit).
    Pbo,
    /// CPU map + TexImage upload — the copy fallback.
    Upload,
}

/// `EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT=1` disables the float-path zero-copy
/// source import (field escape hatch — a driver that mis-renders TEXTURE_2D
/// dma-buf attaches can fall back to the upload path without a rebuild).
fn float_src_import_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

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

/// Packed float surface geometry for a `channels`-plane NCHW zero-copy
/// destination.
///
/// The logical destination is `[C, H, W]` f16 or f32 (`PlanarRgb` /
/// `PlanarRgba`). It is packed into an RGBA16F or RGBA32F surface where each
/// texel holds four contiguous planar elements, giving a GL-visible surface of
/// `(W/4, C*H)` texels.
///
/// Returns `None` when `W` is not divisible by 4 (the packing requires whole
/// texels per row), signalling the caller to fall back to CPU.
///
/// The same geometry the allocators derive — `edgefirst_tensor`'s
/// `packed_rgba16f_layout` (F16 IOSurface/AHardwareBuffer) and
/// `d3d11_layout::image_d3d11_layout` (both dtypes) — so a destination
/// allocated there imports at exactly these dimensions.
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
/// Returns `None` unless `W*3` divides into whole texels, which for a
/// three-channel row means `W % 4 == 0` — the same rule
/// `d3d11_layout::image_d3d11_layout` applies to an `Rgb` float texture.
fn packed_interleaved_layout(w: u32, h: u32) -> Option<(u32, u32)> {
    let row = w.checked_mul(3)?;
    if !row.is_multiple_of(4) {
        return None;
    }
    Some((row / 4, h))
}

/// Fetch the PBO buffer id of a float PBO destination tensor.
///
/// Shared by the `TensorDyn::F32`/`F16` arms of `convert_float_to_pbo`:
/// resolves the tensor's PBO (`NotSupported`-equivalent `OpenGl` error when
/// the tensor is not PBO-backed) and rejects a currently-mapped PBO.
fn float_pbo_buffer_id<T>(t: &edgefirst_tensor::Tensor<T>) -> crate::Result<u32>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
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
    unsafe {
        // 1 second, in nanoseconds — generous; a healthy convert completes in well
        // under a millisecond and never reaches the timeout.
        const TIMEOUT_NS: u64 = 1_000_000_000;
        let sync = edgefirst_gl::gl::FenceSync(edgefirst_gl::gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if sync.is_null() {
            // Fence could not be created; preserve the completion guarantee.
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
            // Timeout expired or the wait failed: fall back to a blocking drain so
            // the caller never proceeds on an incomplete readback/render.
            _ => edgefirst_gl::gl::Finish(),
        }
    }
}

/// `glReadPixels` the packed float render target into the destination PBO,
/// placing each destination row at its own pitch.
///
/// The float counterpart of the u8 `read_pixels_into_pbo`, and the same three
/// routes: a tight destination takes exactly the read it always did, a pitch
/// that is a whole number of texels is handed to GL as `GL_PACK_ROW_LENGTH`
/// so the transfer stays CPU-free, and a pitch no texel size divides is read
/// tight and spread inside the PBO's own mapping. There is no
/// `direct_read_supported` question here and no BGRA swap: the pack format is
/// the render target's own, and the float shaders write the destination's
/// channel order already.
///
/// The spread route is reachable: an F16 NCHW destination reads 8 bytes to a
/// texel, so a plane-row pitch padded by 4 bytes is not expressible in texels.
///
/// # Safety
/// Must run on the GL thread with the float render target bound as a complete
/// read framebuffer. `layout` must describe this destination -- its `needed`
/// is the bound every write here stays inside, and nothing below can check it
/// (the read and the mapping both go through raw pointers).
unsafe fn read_float_pixels_into_pbo(
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
        // Plain ReadPixels — glReadnPixels is ES 3.2-only and rejected by
        // ANGLE/Metal's ES 3.0 contexts (see `readback_rendered`). The PBO
        // PACK binding bounds the write.
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
        // Wait for the readback into the destination PBO to complete before
        // returning (same contract as glFinish, scoped to a fence). The spread
        // below reads what it wrote, so this wait has to precede it.
        finish_via_fence();
    }
    check_gl_error(function!(), line!())?;
    if stride > tight_row && row_length.is_none() {
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

impl GLProcessorST {
    /// Feed the RGBA8 source into `camera_normal_texture` for the float
    /// render paths (PBO and zero-copy F16), preferring zero-copy.
    ///
    /// Feed order:
    /// 1. **Import** — a `TensorMemory::DmaBuf` source on a zero-copy backend
    ///    is attached as the texture's storage (EGLImage on Linux/Android,
    ///    IOSurface pbuffer bind on macOS) through the same
    ///    `get_or_create_egl_image(CacheKind::Src)` cache the u8 engine
    ///    uses. No map, no upload, no per-frame cache-maintenance on the
    ///    source buffer. NOTE the gate deliberately does NOT carry the u8
    ///    path's `!is_dma()` exclusion (`draw_src_texture`,
    ///    `zero_copy_attach`): on Linux `is_dma()` is true and the u8
    ///    engine samples imports via `GL_TEXTURE_EXTERNAL_OES`, but the
    ///    float shaders sample plain `sampler2D`, so the float path wants
    ///    the `TEXTURE_2D` attach on Linux too. Do not "fix" the gates to
    ///    match.
    /// 2. **PBO** — uploaded via `GL_PIXEL_UNPACK_BUFFER` (no CPU copy; a
    ///    PBO `map()` would deadlock the GL worker — see the caller note).
    /// 3. **Upload** — CPU map + `TexImage2D`/`TexSubImage2D` (the
    ///    fallback copy path; logged at debug when it follows a failed
    ///    import).
    ///
    /// `EDGEFIRST_GL_NO_FLOAT_SRC_IMPORT=1` disables arm 1 (field escape
    /// hatch, mirroring `EDGEFIRST_GL_NO_FLOAT_LINEAR`).
    ///
    /// Resets the texture swizzle to identity first (a prior Grey/planar
    /// conversion may have left it non-identity).
    ///
    /// Storage-state contract (the "poison", mirroring the u8 attach at
    /// `draw_src_texture`): after an Import, `camera_normal_texture.target`
    /// is set to 0 so a later upload frame must re-`TexImage2D` fresh
    /// storage — `TexSubImage2D` into an EGLImage-sibling texture would
    /// write through into the client's live buffer. In the reverse
    /// direction, every `TexImage2D` site already clears `bound_egl_key`
    /// (resources.rs / the PBO arm below), so a later import frame can
    /// never be skipped by the binding-skip cache.
    pub(super) fn feed_float_src(
        &mut self,
        src_u8: &Tensor<u8>,
        src_w: usize,
        src_h: usize,
        src_filter: i32,
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
            // Identity swizzle (a prior Grey/planar conversion may have left
            // TEXTURE_SWIZZLE_* in a non-identity state on this texture).
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
                    // SAFETY: camera_normal_texture is bound on the active
                    // unit (prologue above); the handle's import is owned by
                    // the src cache.
                    let attach = unsafe {
                        self.camera_normal_texture
                            .bind_egl_image(&self.gl_context, key, handle)
                    };
                    // Explicit glGetError: the Linux `attach_tex_image_2d`
                    // does not check it, and some drivers (Vivante) accept
                    // dma-buf EGLImages only on EXTERNAL_OES targets —
                    // GL_INVALID_OPERATION here must fall back to upload,
                    // not render garbage.
                    match attach.and_then(|_| check_gl_error(function!(), line!())) {
                        Ok(()) => {
                            // Upload-poison (see doc comment): a later
                            // upload frame must TexImage2D fresh storage.
                            self.camera_normal_texture.target = 0;
                            self.convert_stats.src_imports += 1;
                            tracing::Span::current().record("src_feed", "import");
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

        // ── Arm 2: PBO source (GL-internal copy, no CPU visit) ──
        if let Some(buffer_id) = src_u8.as_pbo().map(|p| p.buffer_id()) {
            unsafe {
                // Upload directly from the source PBO (zero CPU copy). The PBO
                // path allocates the same RGBA8 storage that `update_texture`
                // would (internalformat RGBA, GL_RGBA, UNSIGNED_BYTE), so the
                // two paths share `camera_normal_texture`'s size/format cache.
                // On the steady-state video path (fixed input size) reuse the
                // existing storage with `TexSubImage2D` (PBO-bound, NULL data)
                // instead of reallocating with `TexImage2D` every frame.
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, buffer_id);
                let cache = &mut self.camera_normal_texture;
                let needs_alloc = cache.target != edgefirst_gl::gl::TEXTURE_2D
                    || cache.width != src_w
                    || cache.height != src_h
                    || cache.format != edgefirst_gl::gl::RGBA;
                if needs_alloc {
                    edgefirst_gl::gl::TexImage2D(
                        edgefirst_gl::gl::TEXTURE_2D,
                        0,
                        edgefirst_gl::gl::RGBA as i32,
                        src_w as i32,
                        src_h as i32,
                        0,
                        edgefirst_gl::gl::RGBA,
                        edgefirst_gl::gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                    // Record the true storage state so the cache reflects
                    // reality: a later interleaved u8 `update_texture` (same
                    // dims/format) will correctly take its TexSubImage2D fast
                    // path rather than be forced to reallocate. TexImage2D
                    // reallocated storage, so any EGLImage binding is stale.
                    cache.target = edgefirst_gl::gl::TEXTURE_2D;
                    cache.width = src_w;
                    cache.height = src_h;
                    cache.format = edgefirst_gl::gl::RGBA;
                    cache.invalidate_egl_binding();
                } else {
                    edgefirst_gl::gl::TexSubImage2D(
                        edgefirst_gl::gl::TEXTURE_2D,
                        0,
                        0,
                        0,
                        src_w as i32,
                        src_h as i32,
                        edgefirst_gl::gl::RGBA,
                        edgefirst_gl::gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                }
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, 0);
            }
            self.convert_stats.src_pbo_uploads += 1;
            tracing::Span::current().record("src_feed", "pbo");
            return Ok(FloatSrcFeed::Pbo);
        }

        // ── Arm 3: CPU map + upload (the copy fallback) ──
        // The map happens ONLY here — a Dma source that imported above never
        // pays the per-frame lock/sync cache maintenance.
        self.convert_stats.src_uploads += 1;
        tracing::Span::current().record("src_feed", "upload");
        let pixels = src_u8.map_read()?;
        self.camera_normal_texture.update_texture(
            edgefirst_gl::gl::TEXTURE_2D,
            src_w,
            src_h,
            edgefirst_gl::gl::RGBA,
            &pixels,
        );
        Ok(FloatSrcFeed::Upload)
    }

    /// Run the shared float full-screen-quad draw used by both the PBO and DMA
    /// F16 render paths.
    ///
    /// The caller must have already (a) uploaded the source via
    /// [`Self::upload_float_src`] and (b) bound the render target (float
    /// texture for PBO, EGLImage renderbuffer for DMA) to the active FBO and
    /// confirmed it complete. This sets the viewport to `(packed_w, packed_h)`,
    /// binds the program, sets the crop uniforms, the sample clamp
    /// `src_extent` (`render::sample_clamp_rect`) and `dst_image_size` when
    /// it is `Some` (required by the F16 NCHW shader), binds the source to
    /// TEXTURE0, and draws the quad. No readback is performed.
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
        // Locations resolve once per program (the string lookups previously
        // ran on every float draw); render programs live for the
        // processor's life, so ids are never recycled under the cache.
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
            edgefirst_gl::gl::Uniform4f(
                locs.pad_color,
                pad_color[0],
                pad_color[1],
                pad_color[2],
                pad_color[3],
            );
            if let Some((w, h)) = dst_image_size {
                edgefirst_gl::gl::Uniform2f(locs.dst_image_size, w, h);
            }
            check_gl_error(function!(), line!())?;

            // Full-screen quad: NDC -1..1 mapped to the whole viewport with
            // 0..1 texcoords. The float shaders ignore the interpolated
            // texcoords (they derive sampling from gl_FragCoord + uniforms),
            // but a valid quad is still needed to rasterize every fragment.
            // The geometry is constant, so it lives in the two STATIC_DRAW
            // VBOs uploaded once by `ensure_float_quad_vbos` — the attrib
            // pointers are re-pointed at them for the draw and RESTORED to
            // the engine's dynamic buffers after (Buffer::new's init-time
            // contract: attrib N permanently reads vertex_buffer /
            // texture_buffer, which every other draw path relies on).
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
            // Restore the init-time attrib→buffer contract before any other
            // draw path runs.
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

    /// Lazily create + upload the constant full-screen quad VBOs used by
    /// [`Self::draw_float_quad`] (see the field docs on `GLProcessorST`).
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
        // Only the two PBO float paths are implemented here. `pixel_bytes` is
        // what one texel of the pack format occupies -- one f32 for the NHWC
        // path, four f16 for the NCHW one -- which is what turns the packed
        // surface's width into a tight row of bytes and what
        // `GL_PACK_ROW_LENGTH` counts.
        let (internal, client_fmt, gl_type, pixel_bytes) = match path {
            FloatRenderPath::PboF32Nhwc => (
                edgefirst_gl::gl::R32F,
                edgefirst_gl::gl::RED,
                edgefirst_gl::gl::FLOAT,
                4,
            ),
            FloatRenderPath::PboF16Nchw => (
                edgefirst_gl::gl::RGBA16F,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::HALF_FLOAT,
                8,
            ),
            // The zero-copy variants and `None` are not PBO paths;
            // `convert()` never routes them here (see the match in
            // `processor::convert`), but the match must stay exhaustive.
            FloatRenderPath::ZeroCopyF16Nchw
            | FloatRenderPath::ZeroCopyF32Nchw
            | FloatRenderPath::ZeroCopyFloatNhwc
            | FloatRenderPath::ZeroCopyFloatRgba
            | FloatRenderPath::None => {
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
            _ => unreachable!(),
        };

        // ── Destination pitch, before anything is drawn ──
        // The packed surface's rows ARE the destination's rows: `PboF32Nhwc`
        // renders `(W*3, H)` R32F texels, one texel per element and one
        // surface row per image row; `PboF16Nchw` renders `(W/4, C*H)`
        // RGBA16F texels, one texel per four planar elements and one surface
        // row per plane row. So surface row `y` is destination row `y`, the
        // read is tight at `packed_w * pixel_bytes`, and the destination
        // spaces its rows at `effective_row_stride()` -- `W * 2` for an F16
        // plane row, `W * 3 * 4` for an F32 image row, more for a pool tensor
        // narrowed with `configure_image` that kept the pool's pitch. Planned
        // here rather than after the draw so a destination this readback
        // cannot fill declines to the CPU backend without rendering first.
        let layout = plan_pbo_readback(
            dst.effective_row_stride().unwrap_or(packed_w * pixel_bytes),
            dst.view_origin().is_some(),
            dst.plane_offset().unwrap_or(0),
            dst.capacity_bytes(),
            packed_w,
            packed_h,
            pixel_bytes,
        )?;

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

        // Source sampling filter. Both float shaders sample at output-pixel
        // centers (they add +0.5 to the integer output index before mapping to
        // the source UV), so LINEAR gives a correct bilinear resize on both
        // paths. LINEAR on the RGBA8 source is unconditionally supported —
        // GL_OES_texture_float_linear is irrelevant here because we filter the
        // u8 source texture, not the float render target.
        let src_filter = edgefirst_gl::gl::LINEAR as i32;

        // ── Source RGBA8 feed (shared with the DMA F16 path): zero-copy
        // import when the source is Dma-backed, PBO upload, or CPU upload —
        // see `feed_float_src`. (PBO sources must NOT be `map()`ed on this
        // thread: a PBO map round-trips a message to this same GL worker,
        // deadlocking — the feed checks `as_pbo()` before mapping.)
        let feed = self.feed_float_src(src_u8, src_w, src_h, src_filter)?;
        // A zero-copy feed may have imported more of the texture than the
        // logical image; map the source rectangle onto it and clamp samples
        // to it.
        let (src_rect_uv, src_extent) =
            self.float_src_mapping_for_feed(feed, src_rect_uv, src_u8, src_w, src_h);

        unsafe {
            // ── Float render texture + FBO ──
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, render_tex_id);
            super::super::core::set_tex_filter(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::NEAREST,
            );
            // Only (re)spec the render-target storage when the packed dims or
            // internal format change (mirrors `proto_tex_dims` / Texture's
            // size cache). On the steady-state fixed-input video path this is
            // unchanged every frame, so we reuse the existing storage and skip
            // the per-frame `TexImage2D` reallocation entirely.
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
                // The float render texture storage was just (re)allocated; mark
                // the cached struct fields so a later EGLImage path won't assume
                // a stale binding/size on this texture object.
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
            src_extent,
            dst_rect_px,
            pad_color,
            dst_image_size,
        )?;

        // ── Readback into the destination PBO, row by row at its pitch ──
        // SAFETY: on the GL thread, with the float render target complete and
        // bound as the read framebuffer by the block above.
        unsafe {
            read_float_pixels_into_pbo(
                &layout,
                dst_buffer_id,
                packed_w,
                packed_h,
                client_fmt,
                gl_type,
            )
        }
    }

    /// Render an RGBA8 source into a zero-copy float destination, writing the
    /// packed surface straight into the platform's GPU buffer through an
    /// imported renderbuffer or texture (zero-copy — no `glReadPixels`).
    ///
    /// `path` selects the destination layout, which
    /// [`Self::render_float_to_zero_copy_tail`] turns into a surface geometry,
    /// an import format and a program:
    ///
    /// * [`FloatRenderPath::ZeroCopyF16Nchw`] / [`FloatRenderPath::ZeroCopyF32Nchw`]
    ///   — `PlanarRgb` / `PlanarRgba`, three or four planes packed into a
    ///   `(W/4, C*H)` RGBA16F / RGBA32F surface.
    /// * [`FloatRenderPath::ZeroCopyFloatNhwc`] — `Rgb`, interleaved into a
    ///   `(W*3/4, H)` surface.
    /// * [`FloatRenderPath::ZeroCopyFloatRgba`] — `Rgba`, one texel per pixel
    ///   into a `(W, H)` surface.
    ///
    /// Everything before the tail — source feed, crop uniforms, quad — is the
    /// F16 PBO path's ([`Self::convert_float_to_pbo`] /
    /// `FloatRenderPath::PboF16Nchw`); only the render target differs.
    ///
    /// This is the V3D / Mali / IOSurface / D3D11-texture zero-copy float
    /// path. On any driver that rejects the float import (e.g. Vivante,
    /// desktop NVIDIA dma-buf) or returns an incomplete FBO, returns
    /// `Err(NotSupported)` so `convert()` degrades gracefully to the CPU.
    /// Never panics.
    pub(super) fn convert_float_to_zero_copy(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        path: FloatRenderPath,
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

        // Destination geometry for the crop uniforms; the full destination
        // validation (format, dtype, zero-copy) happens in the shared tail.
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        // Crop uniforms — identical contract to the F16 PBO path.
        let (src_rect_uv, dst_rect_px, pad_color) =
            float_crop_uniforms(&crop, src_w, src_h, dst_w, dst_h)?;

        let src_tex_id = self.camera_normal_texture.id;
        let src_filter = edgefirst_gl::gl::LINEAR as i32;

        // ── Source RGBA8 feed (shared with the PBO F16 path): zero-copy
        // import when the source is Dma-backed, else PBO/CPU upload — see
        // `feed_float_src`.
        let feed = self.feed_float_src(src_u8, src_w, src_h, src_filter)?;
        // As above: the imported texture can be larger than the logical image.
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
        )
    }

    /// The destination half of the zero-copy float render: import the float
    /// destination as its packed render surface, attach it to the FBO, and
    /// draw `src_tex_id` (an RGBA8 texture already holding the source pixels)
    /// through the program that packs it.
    ///
    /// Shared by [`Self::convert_float_to_zero_copy`] (which feeds its tensor
    /// source first) and the fused NV→float two-pass (whose pass 1 rendered
    /// into an engine-internal texture — the pixels never visit the host).
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
    /// Any other `(path, format, dtype)` combination is `NotSupported` so
    /// `convert()` falls back to the CPU.
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

        // Surface geometry, program and the plane size the packed shaders need
        // (`dst_image_size`, which only the planar packer reads).
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

        // The float import format follows the destination's element width;
        // both are RGBA surfaces of four elements per texel.
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
        // Both arms run the same body — `render_float_dst` is generic over the
        // element type.
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
                )
            }
        }
    }

    /// Import `dst` as a `(surface_w, surface_h)` packed float render surface,
    /// attach it to the convert FBO and draw `src_tex_id` through
    /// `program_id`.
    ///
    /// Generic over the destination element type so the F16 and F32 arms of
    /// [`Self::render_float_to_zero_copy_tail`] share one body; the surface,
    /// import format and program are all the four zero-copy float paths
    /// differ in.
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
    ) -> crate::Result<()>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        // ── Import the destination as a renderable float surface ──
        // The tensor's natural row stride equals the packed pitch for every
        // layout in the table (four elements per texel), so the import derives
        // the correct pitch from the surface dimensions.
        self.convert_fbo.bind();
        let dest_egl = self.get_or_create_egl_image_rgb(
            dst,
            dst_fmt,
            surface_w as usize,
            surface_h as usize,
            packed,
        )?;

        // Attach the import (renderbuffer when supported, else texture) to the
        // FBO, mirroring the u8 packed-RGB DMA destination path.
        //
        // SAFETY: convert_fbo is bound above on the thread's current GL
        // context; rbo comes from this call's own cache (either just
        // inserted by get_or_create_egl_image_rgb or a prior hit for the
        // same buffer identity) and dest_egl/self.render_texture stay valid
        // for the FBO calls below.
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
                }
                None => {
                    edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                    edgefirst_gl::gl::BindTexture(
                        edgefirst_gl::gl::TEXTURE_2D,
                        self.render_texture.id,
                    );
                    super::super::core::set_tex_filter(
                        edgefirst_gl::gl::TEXTURE_2D,
                        edgefirst_gl::gl::NEAREST,
                    );
                    // Platform attach (EGLImage target on Linux and ANGLE,
                    // eglBindTexImage on macOS — a raw OES call there
                    // silently no-ops on a pbuffer handle and leaves the
                    // FBO incomplete).
                    super::super::platform::Platform::attach_tex_image_2d(
                        &self.gl_context,
                        dest_egl,
                    )?;
                    edgefirst_gl::gl::FramebufferTexture2D(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::TEXTURE_2D,
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
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                return Err(crate::Error::NotSupported(format!(
                    "GL float render-to-DMA: FBO incomplete (0x{fbo_status:x}) for the \
                     {packed:?} import; using CPU fallback"
                )));
            }
            check_gl_error(function!(), line!())?;
        }

        // ── Shared float draw: viewport = the packed surface, program,
        // uniforms, quad ──
        self.draw_float_quad(
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
        )?;

        // Zero-copy: the GPU wrote straight into the destination buffer. No
        // readback, but we must still wait for the render to complete before
        // returning so the buffer is safe for the consumer to read (same
        // contract as glFinish, scoped to a fence rather than a full-queue
        // drain).
        //
        // SAFETY: called on the thread that owns the current GL context, per
        // finish_via_fence's contract.
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
    use super::{float_pbo_buffer_id, packed_interleaved_layout, packed_planar_layout};
    use crate::Error;

    /// The interleaved rule is the less obvious of the two: three channels
    /// per pixel means a row is `W*3` elements, and a texel holds four, so a
    /// whole-texel row needs `W*3 % 4 == 0` -- which for three channels is
    /// exactly `W % 4 == 0`.
    #[test]
    fn packed_interleaved_layout_takes_whole_texels_only() {
        assert_eq!(packed_interleaved_layout(640, 480), Some((480, 480)));
        assert_eq!(packed_interleaved_layout(4, 1), Some((3, 1)));
        // W*3 for these six widths is 3, 6, 9, 15, 18 and 21 -- not one of
        // them a whole number of four-element texels, which is the same set
        // `W % 4 == 0` rejects.
        for w in [1u32, 2, 3, 5, 6, 7] {
            assert_eq!(
                packed_interleaved_layout(w, 16),
                None,
                "{w}x3 is not a whole number of four-element texels"
            );
        }
        // The planar rule beside it, for contrast: four elements of one
        // channel per texel, so it is the width itself that must divide.
        assert_eq!(packed_planar_layout(640, 480, 3), Some((160, 1440)));
        assert_eq!(packed_planar_layout(6, 16, 3), None);
    }

    #[test]
    fn pbo_buffer_id_rejects_non_pbo_tensor() {
        // A plain Mem-backed tensor is not PBO-backed → OpenGl error rather
        // than a panic / bogus buffer id.
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
