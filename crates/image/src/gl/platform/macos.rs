// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS platform implementation of the OpenGL backend.
//!
//! Uses Google ANGLE (translating OpenGL ES 3.0 to Metal) as the libEGL
//! provider, and Apple IOSurface as the zero-copy buffer interchange.
//! The buffer import path is structurally different from Linux: Linux
//! creates an `EGLImage` from a DMA-BUF fd and binds it to a texture via
//! `glEGLImageTargetTexture2DOES`; macOS creates an EGL pbuffer from an
//! IOSurface via `EGL_ANGLE_iosurface_client_buffer` and binds it as a
//! texture source via `eglBindTexImage`.
//!
//! Method bodies are stubs in this skeleton commit. Real implementations
//! land in Task 36 (display bring-up) and Task 37 (IOSurface import).
//! See `spikes/angle_iosurface/` for the proof-of-concept that validates
//! this approach end-to-end at 7-30× speedup over naive upload/download.

use super::super::{Egl, TransferBackend};
use super::{GlPlatform, PlatformDisplay, PlatformGpuBuffer};
use crate::Error;
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;

pub(in super::super) struct MacosPlatform;

impl GlPlatform for MacosPlatform {
    fn load_egl_lib() -> Result<&'static libloading::Library, Error> {
        Err(Error::NotImplemented(
            "MacosPlatform::load_egl_lib not yet wired (Task 36)".into(),
        ))
    }

    fn create_display(_egl: &Egl) -> Result<(egl::Display, PlatformDisplay), Error> {
        Err(Error::NotImplemented(
            "MacosPlatform::create_display not yet wired (Task 36)".into(),
        ))
    }

    fn probe_transfer_backend(_egl: &Egl, _dpy: egl::Display) -> TransferBackend {
        // Wired in Task 36. macOS will return TransferBackend::IOSurface
        // when EGL_ANGLE_iosurface_client_buffer is advertised, else Pbo.
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
            "MacosPlatform::import_buffer not yet wired (Task 37)".into(),
        ))
    }

    fn bind_as_texture(
        _egl: &Egl,
        _dpy: egl::Display,
        _buf: &PlatformGpuBuffer,
        _tex_id: u32,
    ) -> Result<(), Error> {
        Err(Error::NotImplemented(
            "MacosPlatform::bind_as_texture not yet wired (Task 37)".into(),
        ))
    }

    fn bind_as_render_target(
        _egl: &Egl,
        _dpy: egl::Display,
        _buf: &PlatformGpuBuffer,
        _tex_id: u32,
    ) -> Result<(), Error> {
        Err(Error::NotImplemented(
            "MacosPlatform::bind_as_render_target not yet wired (Task 37)".into(),
        ))
    }

    fn release_buffer(_egl: &Egl, _dpy: egl::Display, _buf: PlatformGpuBuffer) {
        // Wired in Task 37 — will call egl.destroy_surface on the pbuf.
    }
}
