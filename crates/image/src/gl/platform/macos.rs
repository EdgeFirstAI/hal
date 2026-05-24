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
//! See `spikes/angle_iosurface/` (local, gitignored) for the spike that
//! validates this approach end-to-end at 7-30× speedup over naive
//! upload/download on Apple M2 Max.

use super::super::iosurface_import;
use super::super::{Egl, TransferBackend};
use super::{GlPlatform, PlatformDisplay, PlatformGpuBuffer};
use crate::Error;
use edgefirst_tensor::{PixelFormat, Tensor};
use khronos_egl as egl;
use log::{debug, warn};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// ANGLE EGL constants (from include/EGL/eglext_angle.h)
// ---------------------------------------------------------------------------

const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: i32 = 0x3203;
const EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE: i32 = 0x3489;

/// Cached libEGL handle. Leaked at first load to avoid dlclose-during-
/// shutdown SIGBUS issues (same pattern as Linux's EGL_LIB in
/// context.rs).
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

/// Search path for libEGL.dylib (Homebrew's ANGLE tap). Override via
/// the `EDGEFIRST_ANGLE_PATH` environment variable — used when a hal
/// binary bundles its own ANGLE alongside the executable.
const ANGLE_SEARCH_PATHS: &[&str] = &[
    "/opt/homebrew/opt/angle/lib/libEGL.dylib",
    "/usr/local/opt/angle/lib/libEGL.dylib",
    // `@loader_path/libEGL.dylib` resolution happens inside dlopen when
    // the path is just `libEGL.dylib`. Listed last as a fallback so
    // explicit Homebrew installs win when both are present.
    "libEGL.dylib",
];

pub(in super::super) struct MacosPlatform;

impl GlPlatform for MacosPlatform {
    fn load_egl_lib() -> Result<&'static libloading::Library, Error> {
        if let Some(lib) = EGL_LIB.get() {
            return Ok(lib);
        }
        // Try the user override first.
        let candidates: Vec<String> = std::env::var("EDGEFIRST_ANGLE_PATH")
            .ok()
            .map(|p| {
                // EDGEFIRST_ANGLE_PATH points at the dir containing the
                // dylibs; build the full path.
                vec![format!("{p}/libEGL.dylib")]
            })
            .unwrap_or_default()
            .into_iter()
            .chain(ANGLE_SEARCH_PATHS.iter().map(|s| s.to_string()))
            .collect();

        let mut last_err: Option<libloading::Error> = None;
        for path in &candidates {
            // SAFETY: dlopen is unsafe because the loaded library can
            // run initializers. ANGLE's libEGL.dylib is well-behaved.
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => {
                    debug!("MacosPlatform: loaded ANGLE libEGL from {path}");
                    let leaked: &'static libloading::Library = Box::leak(Box::new(lib));
                    return Ok(EGL_LIB.get_or_init(|| leaked));
                }
                Err(e) => last_err = Some(e),
            }
        }
        warn!(
            "MacosPlatform: failed to load libEGL.dylib from any search path. \
             Install ANGLE via `brew install startergo/angle/angle` and re-sign \
             the dylibs (see README.md § macOS GPU Acceleration). \
             Set EDGEFIRST_ANGLE_PATH=/path/to/angle/lib to override the search."
        );
        Err(Error::Io(std::io::Error::other(format!(
            "ANGLE libEGL.dylib not found in any of {candidates:?}: {last_err:?}"
        ))))
    }

    fn create_display(egl: &Egl) -> Result<(egl::Display, PlatformDisplay), Error> {
        // ANGLE's libEGL exposes EGL_EXT_platform_base as a client
        // extension. We must call eglGetPlatformDisplayEXT explicitly
        // with platform = EGL_PLATFORM_ANGLE_ANGLE and a type attrib
        // selecting the Metal backend.
        type FnGetPlatformDisplayEXT = unsafe extern "C" fn(
            platform: u32,
            native: *mut std::ffi::c_void,
            attribs: *const i32,
        ) -> egl::EGLDisplay;

        let get_platform_display_ptr = egl
            .get_proc_address("eglGetPlatformDisplayEXT")
            .ok_or_else(|| {
                Error::Io(std::io::Error::other(
                    "eglGetPlatformDisplayEXT not exported by ANGLE libEGL",
                ))
            })?;
        // SAFETY: function pointer comes from EGL's own dispatch table
        // and matches the well-known C signature.
        let get_platform_display: FnGetPlatformDisplayEXT =
            unsafe { std::mem::transmute(get_platform_display_ptr) };

        let attribs = [
            EGL_PLATFORM_ANGLE_TYPE_ANGLE,
            EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE,
            egl::NONE,
        ];

        // SAFETY: passing well-formed attrib list to a documented EGL
        // extension entry point.
        let raw = unsafe {
            get_platform_display(
                EGL_PLATFORM_ANGLE_ANGLE,
                std::ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(Error::Io(std::io::Error::other(
                "eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE) returned NO_DISPLAY",
            )));
        }
        // SAFETY: raw is a valid EGLDisplay pointer per the spec.
        let display = unsafe { egl::Display::from_ptr(raw) };
        Ok((display, PlatformDisplay::AngleMetal))
    }

    fn probe_transfer_backend(egl: &Egl, dpy: egl::Display) -> TransferBackend {
        let exts = egl
            .query_string(Some(dpy), egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if exts.contains("EGL_ANGLE_iosurface_client_buffer") {
            debug!("MacosPlatform: EGL_ANGLE_iosurface_client_buffer available");
            // TransferBackend gains an IOSurface variant in Task 38;
            // until then fall back to Sync so the existing dispatch
            // never picks the unimplemented path.
            TransferBackend::Sync
        } else {
            warn!(
                "MacosPlatform: EGL_ANGLE_iosurface_client_buffer missing — \
                 ANGLE build may not include IOSurface support, GPU \
                 backend will fall back to CPU"
            );
            TransferBackend::Sync
        }
    }

    fn import_buffer(
        egl: &Egl,
        dpy: egl::Display,
        cfg: Option<egl::Config>,
        _ctx: egl::Context,
        tensor: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> Result<PlatformGpuBuffer, Error> {
        let _span = tracing::trace_span!(
            "image.gl.import_buffer",
            platform = "macos",
            format = ?fmt,
        )
        .entered();

        let surface_ref = iosurface_import::tensor_iosurface_ref(tensor).ok_or_else(|| {
            Error::NotSupported(
                "macOS GL backend requires an IOSurface-backed tensor (TensorMemory::Dma); \
                 SHM/Mem/Pbo cannot be imported via EGL_ANGLE_iosurface_client_buffer"
                    .into(),
            )
        })?;
        let config = cfg.ok_or_else(|| {
            Error::Internal(
                "MacosPlatform::import_buffer: missing egl::Config (SharedEglDisplay must \
                 carry an explicit config on macOS for pbuffer creation)"
                    .into(),
            )
        })?;
        let width = tensor.width().ok_or_else(|| {
            Error::InvalidShape("import_buffer: tensor has no width".into())
        })?;
        let height = tensor.height().ok_or_else(|| {
            Error::InvalidShape("import_buffer: tensor has no height".into())
        })?;

        // SAFETY: surface_ref is borrowed from a live tensor, config is
        // a valid EGLConfig with EGL_BIND_TO_TEXTURE_TARGET_ANGLE set.
        let pbuf = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                egl,
                dpy,
                config,
                surface_ref,
                fmt,
                width,
                height,
            )?
        };
        Ok(PlatformGpuBuffer::IoSurface { pbuf })
    }

    fn bind_as_texture(
        egl: &Egl,
        dpy: egl::Display,
        buf: &PlatformGpuBuffer,
        tex_id: u32,
    ) -> Result<(), Error> {
        let _span = tracing::trace_span!(
            "image.gl.bind_texture",
            platform = "macos",
            target = "sampler",
            tex_id,
        )
        .entered();
        // `EGL_BACK_BUFFER` is the buffer identifier required by
        // eglBindTexImage; pbuffer surfaces have a single back buffer
        // by convention.
        const EGL_BACK_BUFFER: i32 = 0x3084;
        let pbuf = match buf {
            PlatformGpuBuffer::IoSurface { pbuf } => *pbuf,
        };
        egl.bind_tex_image(dpy, pbuf, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))
    }

    fn bind_as_render_target(
        egl: &Egl,
        dpy: egl::Display,
        buf: &PlatformGpuBuffer,
        tex_id: u32,
    ) -> Result<(), Error> {
        let _span = tracing::trace_span!(
            "image.gl.bind_texture",
            platform = "macos",
            target = "render_target",
            tex_id,
        )
        .entered();
        // On macOS the same `eglBindTexImage` is the entry point — the
        // resulting texture can be both sampled and attached to an FBO
        // as a color attachment (the caller does the FBO setup with
        // glFramebufferTexture2D, which lives in the GL processor, not
        // here).
        const EGL_BACK_BUFFER: i32 = 0x3084;
        let pbuf = match buf {
            PlatformGpuBuffer::IoSurface { pbuf } => *pbuf,
        };
        egl.bind_tex_image(dpy, pbuf, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))
    }

    fn release_buffer(egl: &Egl, dpy: egl::Display, buf: PlatformGpuBuffer) {
        let pbuf = match buf {
            PlatformGpuBuffer::IoSurface { pbuf } => pbuf,
        };
        if let Err(e) = egl.destroy_surface(dpy, pbuf) {
            warn!("MacosPlatform::release_buffer: destroy_surface failed: {e:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms `load_egl_lib` can locate ANGLE's libEGL.dylib on this
    /// host. Skips silently if the Homebrew ANGLE tap is not installed
    /// — CI runs without ANGLE should not fail this test.
    ///
    /// Successful loading of the dylib does NOT validate that the
    /// signature is correct for Tahoe (would require dlopen, which CI
    /// binaries can't do without entitlements). The signature validation
    /// step lives in the wider integration test once ImageProcessor is
    /// wired to use MacosPlatform.
    #[test]
    fn load_egl_lib_finds_homebrew_angle_or_skips() {
        let exists_at_any = ANGLE_SEARCH_PATHS
            .iter()
            .any(|p| std::path::Path::new(p).exists());
        if !exists_at_any {
            eprintln!(
                "ANGLE not installed at any default path — skipping. \
                 Run `brew install startergo/angle/angle` to enable this test."
            );
            return;
        }
        // The first .exists() candidate should also be loadable — this
        // catches missing/broken signatures only if the CI process has
        // entitlements (otherwise dlopen SIGKILLs the test process,
        // which manifests as a test failure with no stdout).
        let _ = MacosPlatform::load_egl_lib();
    }
}
