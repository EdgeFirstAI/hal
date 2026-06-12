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

use super::super::context::{egl_ext, GlContext};
use super::super::dma_import::DmaImportAttrs;
use super::super::resources::EglImage;
use super::super::EglDisplayKind;
use super::GlPlatform;
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;
use std::rc::Rc;

/// Marker type: Linux EGL + DMA-BUF platform. Stateless — all state
/// lives in the [`GlContext`] created by [`GlPlatform::init_display`].
pub(crate) struct LinuxEgl;

impl GlPlatform for LinuxEgl {
    type Display = GlContext;
    type Import = EglImage;

    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<GlContext> {
        GlContext::new(kind)
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
