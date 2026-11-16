// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS and iOS platform helpers for the OpenGL backend.
//!
//! Uses Google ANGLE (translating OpenGL ES 3.0 to Metal) as the libEGL
//! provider, and Apple IOSurface as the zero-copy buffer interchange: an EGL
//! pbuffer is created from an IOSurface via
//! `EGL_ANGLE_iosurface_client_buffer` and bound as a texture source via
//! `eglBindTexImage`.
//!
//! The macOS and iOS bring-up flows differ in how the libEGL handle is
//! obtained (see [`ApplePlatform::load_egl_lib`]): macOS `dlopen`s a
//! `libEGL.dylib` from the search paths; iOS `dlopen`s the `libEGL` image
//! dyld has loaded.
//!
//! * [`ApplePlatform::load_egl_lib`]: locate and dlopen ANGLE's libEGL.
//! * [`ApplePlatform::create_display`]: bring up an ANGLE Metal display via
//!   `eglGetPlatformDisplayEXT`.
//!
//! Pbuffer import lives in [`super::super::iosurface_import`], and the
//! `GlPlatform` implementation for macOS/iOS is `AngleClientBuffer` in the
//! sibling `angle` module.

use super::super::Egl;
use crate::Error;
use edgefirst_egl as egl;
use log::debug;
#[cfg(target_os = "macos")]
use log::warn;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// ANGLE EGL constants (from include/EGL/eglext_angle.h)
// ---------------------------------------------------------------------------

const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: i32 = 0x3203;
const EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE: i32 = 0x3489;

/// Cached libEGL handle. Leaked at first load.
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

/// Default search paths for `libEGL.dylib`, in order:
///
/// 1. Homebrew installs (Apple Silicon, then Intel).
/// 2. `@loader_path` — alongside the loading binary.
/// 3. `@executable_path` — alongside the main executable.
/// 4. Bare `libEGL.dylib` — the dyld search path (`DYLD_LIBRARY_PATH`,
///    etc.).
///
/// `EDGEFIRST_ANGLE_PATH` (env var) is prepended at runtime — see
/// [`load_egl_lib`].
const ANGLE_SEARCH_PATHS: &[&str] = &[
    "/opt/homebrew/opt/angle/lib/libEGL.dylib",
    "/usr/local/opt/angle/lib/libEGL.dylib",
    "@loader_path/libEGL.dylib",
    "@loader_path/../Frameworks/libEGL.dylib",
    "@executable_path/libEGL.dylib",
    "@executable_path/../Frameworks/libEGL.dylib",
    "libEGL.dylib",
];

/// Apple-platform (macOS + iOS) helpers.
pub(in super::super) struct ApplePlatform;

impl ApplePlatform {
    /// Obtain the ANGLE libEGL handle, cached for the process lifetime.
    /// Returns a leaked `&'static` reference on first successful load; the
    /// handle is never closed.
    ///
    /// The acquisition is OS-specific ([`Self::load_egl_lib_inner`]): macOS
    /// `dlopen`s a `libEGL.dylib` from the search paths; iOS locates the
    /// loaded `libEGL` image in dyld's image list and `dlopen`s it by exact
    /// path.
    pub(in super::super) fn load_egl_lib() -> Result<&'static libloading::Library, Error> {
        if let Some(lib) = EGL_LIB.get() {
            return Ok(lib);
        }
        let lib = Self::load_egl_lib_inner()?;
        let leaked: &'static libloading::Library = Box::leak(Box::new(lib));
        Ok(EGL_LIB.get_or_init(|| leaked))
    }

    /// iOS: locate the loaded `libEGL` image in dyld's image list and
    /// `dlopen` it by its exact path. Returns an error when ANGLE is not
    /// linked.
    #[cfg(target_os = "ios")]
    fn load_egl_lib_inner() -> Result<libloading::Library, Error> {
        unsafe extern "C" {
            fn _dyld_image_count() -> u32;
            fn _dyld_get_image_name(image_index: u32) -> *const std::os::raw::c_char;
        }
        // SAFETY: `_dyld_image_count` / `_dyld_get_image_name` take no
        // pointers in and return an index bound or a NUL-terminated string
        // owned by dyld.
        let count = unsafe { _dyld_image_count() };
        for index in 0..count {
            // SAFETY: `index < count`; a NULL return is skipped.
            let name = unsafe { _dyld_get_image_name(index) };
            if name.is_null() {
                continue;
            }
            // SAFETY: dyld guarantees a valid NUL-terminated C string.
            let path = unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy();
            if path.ends_with("/libEGL.framework/libEGL") || path.ends_with("/libEGL.dylib") {
                debug!("ApplePlatform: dlopen already-loaded ANGLE libEGL at {path}");
                // SAFETY: the image is loaded; this dlopen bumps its reference count.
                return unsafe { libloading::Library::new(path.as_ref()) }.map_err(|e| {
                    Error::Io(std::io::Error::other(format!(
                        "dlopen of loaded ANGLE libEGL image {path} failed: {e}"
                    )))
                });
            }
        }
        Err(Error::Io(std::io::Error::other(
            "ANGLE libEGL is not loaded in this process: link AND embed \
             EGL.xcframework + GLESv2.xcframework into the app or test bundle \
             (an embed-only copy, or an unused-framework strip at link time, \
             leaves the image list without libEGL)",
        )))
    }

    /// macOS: locate and `dlopen` ANGLE's libEGL. Search order:
    /// `EDGEFIRST_ANGLE_PATH` env var (if set) → entries in
    /// [`ANGLE_SEARCH_PATHS`] (Homebrew, `@loader_path`, `@executable_path`,
    /// bare name).
    ///
    /// `EDGEFIRST_ANGLE_PATH` points at a directory containing flat
    /// `libEGL.dylib` + `libGLESv2.dylib` siblings.
    #[cfg(target_os = "macos")]
    fn load_egl_lib_inner() -> Result<libloading::Library, Error> {
        // Try the user override first.
        let candidates: Vec<String> = std::env::var("EDGEFIRST_ANGLE_PATH")
            .ok()
            .map(|p| {
                // EDGEFIRST_ANGLE_PATH names the directory containing the dylibs.
                vec![format!("{p}/libEGL.dylib")]
            })
            .unwrap_or_default()
            .into_iter()
            .chain(ANGLE_SEARCH_PATHS.iter().map(|s| s.to_string()))
            .collect();

        let mut last_err: Option<libloading::Error> = None;
        for path in &candidates {
            // SAFETY: ANGLE's libEGL initializers are sound to run.
            match unsafe { libloading::Library::new(path) } {
                Ok(lib) => {
                    debug!("ApplePlatform: loaded ANGLE libEGL from {path}");
                    return Ok(lib);
                }
                Err(e) => last_err = Some(e),
            }
        }
        warn!(
            "ApplePlatform: failed to load ANGLE libEGL from any search path. \
             Install ANGLE via `scripts/fetch-angle.sh` (the signed \
             EdgeFirst angle-package release) or `brew install \
             startergo/angle/angle` (re-sign the dylibs — see README.md § \
             macOS GPU Acceleration). Set EDGEFIRST_ANGLE_PATH to the \
             directory containing libEGL.dylib to override the search."
        );
        Err(Error::Io(std::io::Error::other(format!(
            "ANGLE libEGL not found in any of {candidates:?}: {last_err:?}"
        ))))
    }

    /// Bring up an ANGLE Metal-backed EGL display.
    ///
    /// `egl` must wrap a libEGL handle obtained from [`load_egl_lib`].
    pub(in super::super) fn create_display(egl: &Egl) -> Result<egl::Display, Error> {
        // Call eglGetPlatformDisplayEXT with platform =
        // EGL_PLATFORM_ANGLE_ANGLE and a type attrib selecting the Metal
        // backend.
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
        Ok(unsafe { egl::Display::from_ptr(raw) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `load_egl_lib` locates ANGLE's libEGL.dylib on this host, or the test skips.
    #[test]
    fn load_egl_lib_finds_homebrew_angle_or_skips() {
        let exists_at_any = ANGLE_SEARCH_PATHS
            .iter()
            .any(|p| std::path::Path::new(p).exists());
        if !exists_at_any {
            crate::test_support::report_skip(
                "ANGLE not installed at any default path — run `brew install \
                 startergo/angle/angle` to enable this test",
            );
            return;
        }

        // The dlopen runs only when `HAL_TEST_ALLOW_DLOPEN_ANGLE` is set.
        if std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            crate::test_support::report_skip(
                "HAL_TEST_ALLOW_DLOPEN_ANGLE unset — run via \
                 scripts/test-macos.sh to exercise the ANGLE dlopen probe",
            );
            return;
        }
        let _ = ApplePlatform::load_egl_lib();
    }
}
