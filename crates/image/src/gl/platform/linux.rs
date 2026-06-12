// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Linux implementation of [`GlPlatform`]: EGL display bring-up via
//! GBM/PlatformDevice/Default and DMA-BUF buffer import.
//!
//! Pure delegation — the actual machinery predates this seam and lives in
//! [`super::super::context`] (display/context bring-up, transfer-backend
//! probe) and [`super::super::dma_import`] (DMA-BUF attribute assembly).
//! This file only binds it to the cross-platform contract; it must stay
//! free of logic so the platform trait remains the single source of truth
//! for *what* a platform provides, with `context.rs`/`dma_import.rs`
//! owning *how* Linux provides it.

use super::super::context::{egl_ext, Egl, GlContext};
use super::super::dma_import::DmaImportAttrs;
use super::super::EglDisplayKind;
use super::GlPlatform;
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;
use std::rc::Rc;

/// An owned EGLImage import over a DMA-BUF. Dropping destroys the
/// EGLImage under the same display-level lifecycle lock as creation.
/// This is the Linux [`GlPlatform::Import`]; the macOS analogue is
/// `angle::IoSurfacePbuffer`.
pub(in crate::opengl_headless) struct EglImage {
    pub(in crate::opengl_headless) egl_image: egl::Image,
    pub(in crate::opengl_headless) egl: Rc<Egl>,
    pub(in crate::opengl_headless) display: egl::Display,
}

impl Drop for EglImage {
    fn drop(&mut self) {
        if self.egl_image.as_ptr() == egl::NO_IMAGE {
            return;
        }

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Display-level EGL op — same dedicated lock as creation.
            let _image_guard = super::super::context::image_lifecycle_guard();
            let e =
                GlContext::egl_destroy_image_with_fallback(&self.egl, self.display, self.egl_image);
            if let Err(e) = e {
                log::error!("Could not destroy EGL image: {e:?}");
            }
        }));
    }
}

/// Marker type: Linux EGL + DMA-BUF platform. Stateless — all state
/// lives in the [`GlContext`] created by [`GlPlatform::init_display`].
pub(crate) struct LinuxEgl;

impl GlPlatform for LinuxEgl {
    type Display = GlContext;
    type Import = EglImage;
    type ImportHandle = egl::Image;

    // EGLImage targets persist on the texture object — the engine's
    // binding-skip cache (BufferImportKey on Texture) applies.
    const PERSISTENT_TEX_BINDINGS: bool = true;

    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<GlContext> {
        GlContext::new(kind)
    }

    fn load_gl_once(display: &GlContext) {
        static GL_LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        GL_LOADED.get_or_init(|| {
            gls::load_with(|s| {
                display
                    .egl
                    .get_proc_address(s)
                    .map_or(std::ptr::null(), |p| p as *const _)
            });
        });
    }

    fn import_handle(import: &EglImage) -> egl::Image {
        import.egl_image
    }

    unsafe fn attach_tex_image_2d(_display: &GlContext, handle: egl::Image) -> crate::Result<()> {
        gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, handle.as_ptr());
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &GlContext,
        handle: egl::Image,
    ) -> crate::Result<()> {
        gls::egl_image_target_texture_2d_oes(gls::gl::TEXTURE_EXTERNAL_OES, handle.as_ptr());
        Ok(())
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &GlContext,
        handle: egl::Image,
    ) -> crate::Result<()> {
        gls::gl::EGLImageTargetRenderbufferStorageOES(gls::gl::RENDERBUFFER, handle.as_ptr());
        Ok(())
    }

    fn end_gpu_pass(_display: &GlContext) {
        // EGLImage texture targets persist by design; nothing to release.
    }

    fn import_buffer_packed<T>(
        display: &GlContext,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> crate::Result<EglImage>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        use std::os::fd::AsRawFd;
        let drm_format = match fmt {
            super::PackedImportFormat::Rgba8888 => drm_fourcc::DrmFourcc::Abgr8888,
            super::PackedImportFormat::Rgba16161616F => drm_fourcc::DrmFourcc::Abgr16161616f,
        };
        let bpp = fmt.bytes_per_pixel();
        let dma = img.as_dma().ok_or_else(|| {
            crate::Error::NotImplemented("import_buffer_packed requires DMA tensor".to_string())
        })?;
        let fd = dma.fd.as_raw_fd();

        // Use the tensor's stored stride when available (externally
        // allocated buffers with row padding), otherwise compute the
        // tightly-packed pitch.
        let pitch = img.effective_row_stride().unwrap_or(width * bpp);
        let offset = img.plane_offset().unwrap_or(0);
        let egl_img_attr = [
            egl_ext::LINUX_DRM_FOURCC as egl::Attrib,
            drm_format as u32 as egl::Attrib,
            khronos_egl::WIDTH as egl::Attrib,
            width as egl::Attrib,
            khronos_egl::HEIGHT as egl::Attrib,
            height as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as egl::Attrib,
            pitch as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as egl::Attrib,
            offset as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as egl::Attrib,
            fd as egl::Attrib,
            egl::IMAGE_PRESERVED as egl::Attrib,
            egl::TRUE as egl::Attrib,
            khronos_egl::NONE as egl::Attrib,
        ];
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    fn import_buffer(
        display: &GlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> crate::Result<EglImage> {
        let attrs = DmaImportAttrs::from_tensor(img, fmt, for_dst)?;
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())
    }

    fn import_buffer_nv_r8(
        display: &GlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<EglImage> {
        let attrs = DmaImportAttrs::from_tensor_nv_r8(img, fmt)?;
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())
    }
}

/// Create an owned EGLImage from raw EGL import attributes.
///
/// Every EGLImage creation funnels through here (DMA-BUF, NV R8, and the
/// float path's RGB renderbuffer import), so one span = one actual
/// `eglCreateImageKHR`. Steady-state frame loops must show ZERO of these
/// after warmup — the span count is the observable for cache-behavior
/// equality gates.
///
/// Module-scoped (not a trait method): the attribute list is
/// EGL/DRM-specific, so only Linux-side code (the float path's
/// dims-override import in `processor/mod.rs`) may call it directly;
/// portable code goes through [`GlPlatform::import_buffer`].
pub(in crate::opengl_headless) fn new_egl_image_owned(
    display: &GlContext,
    target: egl::Enum,
    attrib_list: &[egl::Attrib],
) -> crate::Result<EglImage> {
    let _span = tracing::trace_span!("image.convert.gl.egl_import", target).entered();
    // EGLImage creation is a display-level EGL op shared by every
    // processor — serialized by a dedicated short lock (zero cost in
    // steady state: creations are cache-miss-only).
    let _image_guard = super::super::context::image_lifecycle_guard();
    let image = GlContext::egl_create_image_with_fallback(
        &display.egl,
        display.display,
        unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
        target,
        unsafe { egl::ClientBuffer::from_ptr(std::ptr::null_mut()) },
        attrib_list,
    )?;
    Ok(EglImage {
        egl_image: image,
        display: display.display,
        egl: Rc::clone(&display.egl),
    })
}
