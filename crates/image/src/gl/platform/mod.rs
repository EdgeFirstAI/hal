// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform abstraction for the OpenGL backend.
//!
//! Linux uses GBM + EGL + DMA-BUF; macOS uses ANGLE + IOSurface. Both
//! routes drive the same GLES 3.0 shaders via the same `Egl` instance and
//! the same `processor.rs` / `resources.rs` plumbing. The platform-
//! specific pieces — libEGL discovery, display bring-up, buffer import,
//! texture/render-target binding — live behind this trait.
//!
//! `CurrentPlatform` is a compile-time type alias resolved per target.
//! Zero overhead vs hand-written cfg branches.
//!
//! See `crates/image/ARCHITECTURE.md § OpenGL Backend Platform
//! Abstraction` for the design rationale.

use super::{Egl, TransferBackend};
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;

#[cfg(target_os = "linux")]
pub(super) mod linux;
#[cfg(target_os = "macos")]
pub(super) mod macos;

/// Selected platform implementation for the current build target.
#[cfg(target_os = "linux")]
#[allow(dead_code)] // wired in Task 29
pub(super) type CurrentPlatform = linux::LinuxPlatform;
#[cfg(target_os = "macos")]
#[allow(dead_code)] // wired in Task 29
pub(super) type CurrentPlatform = macos::MacosPlatform;

/// Platform-specific display state. Carries any resources that must
/// outlive the EGL display (e.g. GBM device on Linux). The display
/// pointer itself is held separately.
#[allow(dead_code)] // populated by Linux/macOS variants
pub(super) enum PlatformDisplay {
    /// Linux GBM-backed display — owns the GBM device + DRM fd.
    #[cfg(target_os = "linux")]
    Gbm,
    /// Linux PlatformDevice EGL extension — no extra state.
    #[cfg(target_os = "linux")]
    PlatformDevice,
    /// Linux default display — fallback path.
    #[cfg(target_os = "linux")]
    Default,
    /// macOS ANGLE Metal-backed display — no extra state.
    #[cfg(target_os = "macos")]
    AngleMetal,
}

/// Platform-specific zero-copy GPU buffer handle.
///
/// On Linux this is an EGLImage created from a DMA-BUF fd. On macOS this
/// is an EGL pbuffer wrapping an IOSurface via
/// `EGL_ANGLE_iosurface_client_buffer`. The two have different EGL flows
/// (`glEGLImageTargetTexture2DOES` vs `eglBindTexImage`), which the trait
/// hides from `processor.rs`.
#[allow(dead_code)] // variants used per-platform
pub(super) enum PlatformGpuBuffer {
    /// Linux DMA-BUF → EGLImage path. Stored as an opaque ptr because
    /// the `EglImage` type lives in `super::resources` and is Linux-only.
    #[cfg(target_os = "linux")]
    EglImage(egl::Image),
    /// macOS IOSurface → EGL pbuffer path.
    #[cfg(target_os = "macos")]
    IoSurface { pbuf: egl::Surface },
}

/// The platform trait. Compile-time selected via `CurrentPlatform`.
///
/// All methods take an `&Egl` rather than holding their own state so the
/// trait can be used as a namespace (`CurrentPlatform::method()`) without
/// requiring an instance. The platform's only persistent state lives in
/// `SharedEglDisplay` (in `context.rs`).
#[allow(dead_code)] // methods called once platform/{linux,macos}.rs are wired in
pub(super) trait GlPlatform: Sized {
    /// Load the platform's `libEGL` dynamic library. Returned reference
    /// is leaked to give the rest of the code a `'static` handle (matches
    /// the existing `EGL_LIB` pattern in `context.rs`).
    fn load_egl_lib() -> Result<&'static libloading::Library, crate::Error>;

    /// Bring up the EGL display for the platform. Returns the display
    /// pointer plus any platform-specific state that must outlive it.
    fn create_display(egl: &Egl) -> Result<(egl::Display, PlatformDisplay), crate::Error>;

    /// Probe what transfer backend is available on this display.
    ///
    /// Linux: returns `DmaBuf` if `EGL_EXT_image_dma_buf_import` is
    /// advertised, else `Pbo`, else `Sync`.
    /// macOS: returns `IOSurface` if `EGL_ANGLE_iosurface_client_buffer`
    /// is advertised, else `Pbo` as fallback.
    fn probe_transfer_backend(egl: &Egl, dpy: egl::Display) -> TransferBackend;

    /// Import an existing tensor's underlying buffer as a GPU-accessible
    /// surface/image. The returned handle is cached by tensor identity in
    /// `super::cache::EglImageCache` so repeated frames from the same
    /// buffer don't re-import.
    fn import_buffer(
        egl: &Egl,
        dpy: egl::Display,
        cfg: Option<egl::Config>,
        ctx: egl::Context,
        tensor: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> Result<PlatformGpuBuffer, crate::Error>;

    /// Bind an imported buffer as the source texture for the next draw.
    ///
    /// Linux: calls `glEGLImageTargetTexture2DOES`.
    /// macOS: calls `eglBindTexImage(EGL_BACK_BUFFER)`.
    fn bind_as_texture(
        egl: &Egl,
        dpy: egl::Display,
        buf: &PlatformGpuBuffer,
        tex_id: u32,
    ) -> Result<(), crate::Error>;

    /// Bind an imported buffer as the FBO color attachment for rendering.
    fn bind_as_render_target(
        egl: &Egl,
        dpy: egl::Display,
        buf: &PlatformGpuBuffer,
        tex_id: u32,
    ) -> Result<(), crate::Error>;

    /// Release the GPU resources held by an imported buffer. Called by
    /// the cache when an entry is evicted.
    fn release_buffer(egl: &Egl, dpy: egl::Display, buf: PlatformGpuBuffer);
}
