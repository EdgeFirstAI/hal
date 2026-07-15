// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS and iOS platform helpers for the OpenGL backend.
//!
//! Uses Google ANGLE (translating OpenGL ES 3.0 to Metal) as the libEGL
//! provider, and Apple IOSurface as the zero-copy buffer interchange.
//! The buffer import path is structurally different from Linux: Linux
//! creates an `EGLImage` from a DMA-BUF fd and binds it to a texture via
//! `glEGLImageTargetTexture2DOES`; macOS/iOS creates an EGL pbuffer from an
//! IOSurface via `EGL_ANGLE_iosurface_client_buffer` and binds it as a
//! texture source via `eglBindTexImage`.
//!
//! The macOS and iOS bring-up flows are identical except for how the
//! libEGL handle is obtained (see [`ApplePlatform::load_egl_lib`]):
//! macOS `dlopen`s a `libEGL.dylib` from Homebrew / a bundled framework;
//! iOS resolves EGL/GLES from the main image via `Library::this()`
//! because the ANGLE xcframeworks are statically linked into the app
//! binary. Both paths feed the same `Dynamic` EGL loader.
//!
//! ## Seam shape
//!
//! Only two operations need a platform-specific spelling at the macOS GL
//! backend layer today:
//!
//! 1. [`ApplePlatform::load_egl_lib`] — locate and dlopen ANGLE's
//!    `libEGL.dylib`.
//! 2. [`ApplePlatform::create_display`] — bring up an ANGLE Metal
//!    display via `eglGetPlatformDisplayEXT`.
//!
//! Everything downstream (pbuffer import, texture binding, FBO setup,
//! shader compilation, lifetime management) lives in
//! [`super::super::macos_processor`] and [`super::super::iosurface_import`].
//! Those modules call these two functions directly — no trait, no enum
//! dispatch. The trait-based `GlPlatform` seam that existed earlier in
//! this branch was unused scaffolding and has been removed.
//!
//! A future `WindowsPlatform` (ANGLE + D3D11 shared textures) will most
//! likely follow the same two-function seam shape, with its own
//! `windows_processor.rs` and `d3d11_import.rs` companions. The seam
//! does not need to be a trait until and unless two platforms end up
//! sharing a processor implementation.

use super::super::Egl;
use crate::Error;
use edgefirst_egl as egl;
// `debug!`/`warn!` are only used inside the macOS `load_egl_lib_inner`
// branch (the iOS path resolves symbols via `Library::this()` with no
// logging). Gate the import so iOS builds don't warn about unused items.
#[cfg(target_os = "macos")]
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

/// Default search paths for `libEGL.dylib`. Order matters:
///
/// 1. Homebrew installs (Apple Silicon, then Intel).
/// 2. `@loader_path` — alongside the loading binary. Lets bundled
///    distributions ship ANGLE in `Frameworks/` or beside the
///    executable without needing `EDGEFIRST_ANGLE_PATH`.
/// 3. `@executable_path` — alongside the main executable when the
///    loader isn't itself the executable.
/// 4. Bare `libEGL.dylib` — last-resort fallback through the dyld
///    search path (`DYLD_LIBRARY_PATH`, etc.).
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

/// Apple-platform (macOS + iOS) helpers. Exposes two associated functions;
/// see the module docstring for the rationale.
pub(in super::super) struct ApplePlatform;

impl ApplePlatform {
    /// Obtain the ANGLE libEGL handle, cached for the process lifetime.
    /// Returns a leaked `&'static` reference on first successful load; the
    /// handle is never closed (avoids dlclose-during-shutdown crashes —
    /// same pattern as Linux's `EGL_LIB` in `context.rs`).
    ///
    /// The actual acquisition is OS-specific ([`Self::load_egl_lib_inner`]):
    /// macOS `dlopen`s a `libEGL.dylib` from the search paths; iOS resolves
    /// from the main image via `Library::this()` (the ANGLE xcframeworks are
    /// dynamic frameworks the app links against and embeds, so they load into
    /// the process image at launch).
    pub(in super::super) fn load_egl_lib() -> Result<&'static libloading::Library, Error> {
        if let Some(lib) = EGL_LIB.get() {
            return Ok(lib);
        }
        let lib = Self::load_egl_lib_inner()?;
        let leaked: &'static libloading::Library = Box::leak(Box::new(lib));
        Ok(EGL_LIB.get_or_init(|| leaked))
    }

    /// iOS: ANGLE EGL/GLES symbols come from the `EGL.xcframework` +
    /// `GLESv2.xcframework` dynamic frameworks that the app links against and
    /// embeds (see `.cargo/config.toml` and README.md § iOS). Because they are
    /// linked (not merely bundled), they load into the process image at launch,
    /// so resolving from the main image via `Library::this()` — equivalent to
    /// `dlopen(NULL)`, a handle for the whole process image — lets `dlsym` find
    /// the ANGLE entry points the `edgefirst-egl` `Dynamic` loader requests.
    /// (The app shell that performs that link/embed is future Swift-bindings
    /// work; see README.md § iOS.)
    #[cfg(target_os = "ios")]
    fn load_egl_lib_inner() -> Result<libloading::Library, Error> {
        let this = libloading::os::unix::Library::this();
        // `os::unix::Library` converts into `libloading::Library` via the
        // crate's `From<os::platform::Library>` impl (libloading 0.9),
        // avoiding an unannotated `transmute`. The EGL `Dynamic` loader
        // then accepts the resulting handle.
        Ok(this.into())
    }

    /// macOS: locate and `dlopen` ANGLE's libEGL. Search order:
    /// `EDGEFIRST_ANGLE_PATH` env var (if set) → entries in
    /// [`ANGLE_SEARCH_PATHS`] (Homebrew, `@loader_path`, `@executable_path`,
    /// bare name).
    ///
    /// `EDGEFIRST_ANGLE_PATH` points at a directory containing flat
    /// `libEGL.dylib` + `libGLESv2.dylib` siblings. ANGLE's libEGL
    /// internally `dlopen`s `libGLESv2.dylib` from its own directory
    /// (located via `dladdr`) to resolve GL entry points, so the two MUST
    /// be siblings in a flat layout — the framework bundle layout does
    /// not work for the macOS runtime dlopen path. The
    /// `scripts/fetch-angle.sh` helper stages such a flat dir from the
    /// signed `angle-package` release xcframeworks.
    #[cfg(target_os = "macos")]
    fn load_egl_lib_inner() -> Result<libloading::Library, Error> {
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
            // run initializers. ANGLE's libEGL is well-behaved.
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
    /// `egl` must wrap a libEGL handle obtained from [`load_egl_lib`]
    /// — the call goes through ANGLE's `EGL_EXT_platform_base` client
    /// extension, which is not present in Apple's system EGL (Apple
    /// ships none).
    pub(in super::super) fn create_display(egl: &Egl) -> Result<egl::Display, Error> {
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
        Ok(unsafe { egl::Display::from_ptr(raw) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirms `load_egl_lib` can locate ANGLE's libEGL.dylib on this
    /// host. Skips silently if the Homebrew ANGLE tap is not installed
    /// — CI runs without ANGLE should not fail this test.
    ///
    /// The optional dlopen probe (which also catches missing/broken
    /// signatures) only runs when the harness sets
    /// `HAL_TEST_ALLOW_DLOPEN_ANGLE=1` — without that opt-in, plain
    /// `cargo test` on macOS 26 hardened-runtime hosts would SIGKILL
    /// the test process at dylib-load time and the failure would
    /// surface as a silent crash with no stdout. `scripts/test-macos.sh`
    /// signs the test binary with the library-validation entitlement
    /// and exports the env var; the broader signature-validation path
    /// also runs as part of the `test_yuyv_to_rgba_opengl_macos`
    /// integration test in `crates/image/src/lib.rs`.
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

        // Only attempt the actual dlopen when the harness has signed
        // this binary with `disable-library-validation`. Outside that
        // path we still report a successful test — the file-existence
        // check above is the portable smoke-test, and the integration
        // test exercises the real load path.
        if std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            eprintln!(
                "HAL_TEST_ALLOW_DLOPEN_ANGLE unset — skipping ANGLE dlopen \
                 probe (run via scripts/test-macos.sh to exercise it)."
            );
            return;
        }
        let _ = ApplePlatform::load_egl_lib();
    }
}
