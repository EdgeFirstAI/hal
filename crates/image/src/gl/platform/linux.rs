// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Linux platform implementation of the OpenGL backend.
//!
//! This file is the seam for code that used to live directly in
//! `super::context` and `super::dma_import`. As the refactor progresses
//! (Task 28) the existing helpers move here. For now this is a stub that
//! returns `NotImplemented` from every method — the production code path
//! still goes through the original locations in `context.rs`.

use super::super::{Egl, TransferBackend};
use super::{GlPlatform, PlatformDisplay, PlatformGpuBuffer};
use crate::Error;
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;

pub(in super::super) struct LinuxPlatform;

impl GlPlatform for LinuxPlatform {
    fn load_egl_lib() -> Result<&'static libloading::Library, Error> {
        Err(Error::NotImplemented(
            "LinuxPlatform::load_egl_lib not yet wired \
             (Task 28 — code still in context.rs::get_egl_lib)"
                .into(),
        ))
    }

    fn create_display(_egl: &Egl) -> Result<(egl::Display, PlatformDisplay), Error> {
        Err(Error::NotImplemented(
            "LinuxPlatform::create_display not yet wired \
             (Task 28 — code still in context.rs::init_shared_display)"
                .into(),
        ))
    }

    fn probe_transfer_backend(_egl: &Egl, _dpy: egl::Display) -> TransferBackend {
        // Wired in Task 28 — meanwhile callers still hit the existing
        // path in context.rs::egl_check_support_dma. Returning Sync is a
        // safe never-reached default.
        TransferBackend::Sync
    }

    fn import_buffer(
        _egl: &Egl,
        _dpy: egl::Display,
        _cfg: Option<egl::Config>,
        _ctx: egl::Context,
        _tensor: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<PlatformGpuBuffer, Error> {
        Err(Error::NotImplemented(
            "LinuxPlatform::import_buffer not yet wired \
             (Task 29 — code still in processor.rs)"
                .into(),
        ))
    }

    fn bind_as_texture(
        _egl: &Egl,
        _dpy: egl::Display,
        _buf: &PlatformGpuBuffer,
        _tex_id: u32,
    ) -> Result<(), Error> {
        Err(Error::NotImplemented(
            "LinuxPlatform::bind_as_texture not yet wired".into(),
        ))
    }

    fn bind_as_render_target(
        _egl: &Egl,
        _dpy: egl::Display,
        _buf: &PlatformGpuBuffer,
        _tex_id: u32,
    ) -> Result<(), Error> {
        Err(Error::NotImplemented(
            "LinuxPlatform::bind_as_render_target not yet wired".into(),
        ))
    }

    fn release_buffer(_egl: &Egl, _dpy: egl::Display, _buf: PlatformGpuBuffer) {
        // The actual destroy logic still lives in EglImage::drop in
        // resources.rs. Once Task 29 moves cache entries to use
        // PlatformGpuBuffer this method becomes the canonical destroy.
    }
}
