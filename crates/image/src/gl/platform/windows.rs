// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Windows implementation of [`GlPlatform`]: ANGLE (GLES → Direct3D 11) on
//! the tensor crate's D3D11 device, with zero-copy texture transfers.
//!
//! Structurally this is the macOS leaf (`platform/macos.rs` loader +
//! `platform/angle.rs` display) with three differences:
//!
//! 1. **Loader** — `libEGL.dll` is found through `EDGEFIRST_ANGLE_PATH`,
//!    then next to the module containing this code (the `@loader_path`
//!    analogue: a wheel's `_image.pyd` or the C archive's
//!    `edgefirst_image.dll` can carry the ANGLE DLLs as siblings), then next
//!    to the executable, then the default DLL search path. Absolute
//!    candidates load with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` so ANGLE's
//!    `libEGL.dll` resolves its sibling `libGLESv2.dll`.
//! 2. **Display** — the display is built on the device the tensor crate
//!    already created: `eglCreateDeviceANGLE` wraps that `ID3D11Device` as
//!    an `EGLDeviceEXT` and `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT)`
//!    turns it into the display. ANGLE imports textures only from its own
//!    device, and every texture tensor lives on the tensor crate's, so the
//!    two must be one device. Which adapter that is belongs to the tensor
//!    crate (`EDGEFIRST_D3D11_ADAPTER`, alias `EDGEFIRST_ANGLE_ADAPTER`);
//!    this leaf selects nothing.
//! 3. **Transfer** — the display reports [`TransferBackend::D3d11Texture`]:
//!    a tensor's `ID3D11Texture2D` is imported as an EGLImage through
//!    `EGL_ANGLE_image_d3d11_texture` and bound with
//!    `glEGLImageTargetTexture2DOES`. The attachment outlives the pass, so
//!    `end_gpu_pass` has nothing to release, but it is re-issued on every
//!    use (`PERSISTENT_TEX_BINDINGS = false`) because ANGLE does not see
//!    writes made to the texture outside GL — see the constant's comment.
//!
//! The shared-display / per-processor-context code is laid out
//! function-for-function like `angle.rs` (as the Android leaf is) so a
//! later `angle_common` extraction is a move rather than a rewrite.
//!
//! No `windows` / `windows-sys` dependency: the three kernel32 calls are
//! declared here, and the D3D11 device and textures arrive as raw
//! `*mut c_void` from the tensor crate's accessors.

use super::super::{CompletionFence, Egl, EglDisplayKind, TransferBackend};
use super::GlPlatform;
use crate::{Error, Result};
use edgefirst_egl as egl;
use edgefirst_tensor::{PixelFormat, Tensor, TensorDyn};
use log::{debug, info, warn};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

/// The EGL context that issued the most recent GL commands on the shared
/// display (see [`GlPlatform::begin_gpu_pass`] below). Touched only under
/// the dispatch wrapper's process-wide message lock (ANGLE takes the Full
/// serialization policy), so a plain atomic is enough.
static LAST_ACTIVE_CONTEXT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// Forget which context issued the last GL commands, so the next message on
/// ANY processor re-makes its own context current
/// ([`GlPlatform::begin_gpu_pass`]).
///
/// Every context creation and destruction on the shared display calls this;
/// destruction goes through [`destroy_context`] so the bring-up failure arms
/// cannot forget. ANGLE's D3D11 backend keeps one state manager per display,
/// and bringing a context up or tearing one down disturbs it: without the
/// invalidation a processor whose context was already the last active one
/// skips its re-sync and draws through state the lifecycle event invalidated,
/// which faults inside `glDrawArrays` (`0xC0000005`, reproduced in ~6% of
/// four-processor convert-then-teardown runs). Clearing it also stops a
/// successor context allocated at a destroyed one's address from being
/// mistaken for it.
fn invalidate_active_context() {
    LAST_ACTIVE_CONTEXT.store(std::ptr::null_mut(), Ordering::Release);
}

/// `eglDestroyContext` plus the invalidation every context destruction on the
/// shared display owes. The one place the leaf destroys a context, including
/// the bring-up failure arms, so [`invalidate_active_context`]'s contract
/// cannot drift away from the call sites.
fn destroy_context(egl: &Egl, display: egl::Display, context: egl::Context) {
    invalidate_active_context();
    let _ = egl.destroy_context(display, context);
}

// ---------------------------------------------------------------------------
// EGL constants (EGL/egl.h + EGL/eglext_angle.h; edgefirst_egl does not
// export the ANGLE ones).
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;

/// `EGL_ANGLE_device_d3d`: the `ID3D11Device*` an `EGLDeviceEXT` wraps.
const EGL_D3D11_DEVICE_ANGLE: i32 = 0x33A1;
/// `EGL_EXT_platform_device`: a display over an `EGLDeviceEXT`.
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
/// `EGL_EXT_device_query`: the `EGLDeviceEXT` behind a display.
const EGL_DEVICE_EXT: i32 = 0x322C;

// ---------------------------------------------------------------------------
// Loader: locate and load ANGLE's libEGL.dll.
// ---------------------------------------------------------------------------

/// Cached libEGL handle. Leaked at first load and never freed (the same
/// policy as Linux's `EGL_LIB` and macOS's: drivers keep state past explicit
/// cleanup, and unloading during process exit crashes).
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
}
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

/// Windows-platform helpers: the two operations with a Windows-specific
/// spelling (locate + load libEGL; bring up a D3D11 display). Everything
/// downstream is the shared engine.
pub(in super::super) struct WindowsPlatform;

impl WindowsPlatform {
    /// Obtain the ANGLE libEGL handle, cached for the process lifetime.
    pub(in super::super) fn load_egl_lib() -> Result<&'static libloading::Library> {
        if let Some(lib) = EGL_LIB.get() {
            return Ok(lib);
        }
        let lib = Self::load_egl_lib_inner()?;
        let leaked: &'static libloading::Library = Box::leak(Box::new(lib));
        Ok(EGL_LIB.get_or_init(|| leaked))
    }

    /// Directory of the module (DLL / `.pyd` / executable) that contains
    /// this code — the `@loader_path` analogue. `None` if either call fails.
    fn module_dir() -> Option<PathBuf> {
        use std::os::windows::ffi::OsStringExt as _;
        let anchor = Self::module_dir as *const () as *const u16;
        let mut module: *mut c_void = std::ptr::null_mut();
        // SAFETY: `anchor` is an address inside this module's image (the
        // documented use of GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS);
        // `module` is a valid out-pointer. UNCHANGED_REFCOUNT means no
        // reference is taken, so nothing needs freeing.
        let ok = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                anchor,
                &mut module,
            )
        };
        if ok == 0 || module.is_null() {
            return None;
        }
        // Long-path aware: 32 K UTF-16 units is the documented maximum.
        let mut buf = vec![0u16; 32 * 1024];
        // SAFETY: `module` is a valid module handle (obtained above) and
        // `buf` is a writable buffer of the passed length.
        let len =
            unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 || len >= buf.len() {
            return None;
        }
        let path = PathBuf::from(std::ffi::OsString::from_wide(&buf[..len]));
        path.parent().map(Path::to_path_buf)
    }

    /// Candidate directories for `libEGL.dll`, in search order.
    fn candidate_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(p) = std::env::var_os("EDGEFIRST_ANGLE_PATH") {
            if !p.is_empty() {
                dirs.push(PathBuf::from(p));
            }
        }
        if let Some(d) = Self::module_dir() {
            dirs.push(d);
        }
        if let Some(d) = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(Path::to_path_buf))
        {
            if !dirs.contains(&d) {
                dirs.push(d);
            }
        }
        dirs
    }

    /// Locate and load ANGLE's libEGL. Search order: `EDGEFIRST_ANGLE_PATH`
    /// → next to the loading module → next to the executable → bare
    /// `libEGL.dll` on the default DLL search path. `libGLESv2.dll` must be
    /// a flat sibling: ANGLE's libEGL loads it from its own directory.
    fn load_egl_lib_inner() -> Result<libloading::Library> {
        use libloading::os::windows::{
            Library as WinLibrary, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        // One line per candidate: "<path>: <why it was rejected>".
        let mut tried: Vec<String> = Vec::new();
        for dir in Self::candidate_dirs() {
            let path = dir.join("libEGL.dll");
            if !path.is_file() {
                tried.push(format!("{}: not found", path.display()));
                continue;
            }
            // SAFETY: LoadLibrary runs the DLL's initializers; ANGLE's
            // libEGL has no initializer side effects of concern.
            // DLL_LOAD_DIR lets it find its sibling libGLESv2.dll;
            // DEFAULT_DIRS covers System32 (d3d11, dxgi, d3dcompiler_47).
            match unsafe {
                WinLibrary::load_with_flags(
                    &path,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                )
            } {
                Ok(lib) => {
                    debug!(
                        "WindowsPlatform: loaded ANGLE libEGL from {}",
                        path.display()
                    );
                    return Ok(lib.into());
                }
                Err(e) => {
                    debug!("WindowsPlatform: {} failed to load: {e}", path.display());
                    tried.push(format!("{}: {e}", path.display()));
                }
            }
        }
        // SAFETY: as above.
        match unsafe { libloading::Library::new("libEGL.dll") } {
            Ok(lib) => {
                debug!("WindowsPlatform: loaded ANGLE libEGL from the default DLL search path");
                return Ok(lib);
            }
            Err(e) => tried.push(format!("libEGL.dll (default DLL search path): {e}")),
        }
        warn!(
            "WindowsPlatform: ANGLE libEGL.dll not found. Fetch the Windows ANGLE \
             package with `scripts/fetch-angle.sh` and set EDGEFIRST_ANGLE_PATH to \
             the directory containing libEGL.dll + libGLESv2.dll, or place the two \
             DLLs next to the executable / edgefirst_image.dll / _image.pyd — see \
             README.md § Windows GPU Acceleration. Falling back to the CPU backend."
        );
        Err(Error::Io(std::io::Error::other(format!(
            "ANGLE libEGL.dll not found: {}",
            tried.join("; ")
        ))))
    }

    /// Hands the tensor crate's device to ANGLE. Every texture tensor lives
    /// on that device, and ANGLE imports textures only from its own device.
    ///
    /// Returns the display and the `EGLDeviceEXT` it was built from. The
    /// device is deliberately never released once a display exists: the
    /// shared display itself is leaked for the process's life, and ANGLE
    /// tears both down at exit.
    ///
    /// `egl` must wrap a libEGL handle from [`Self::load_egl_lib`]: both
    /// calls go through ANGLE client extensions, which Windows has no
    /// system EGL for.
    pub(in super::super) fn create_display(
        egl: &Egl,
        device: &edgefirst_tensor::d3d11::D3d11Device,
    ) -> Result<(egl::Display, *mut c_void)> {
        type FnCreateDeviceANGLE = unsafe extern "C" fn(
            device_type: i32,
            native: *mut c_void,
            attribs: *const isize,
        ) -> *mut c_void;
        type FnGetPlatformDisplayEXT = unsafe extern "C" fn(
            platform: u32,
            native: *mut c_void,
            attribs: *const i32,
        ) -> egl::EGLDisplay;

        let create_device = egl
            .get_proc_address("eglCreateDeviceANGLE")
            .ok_or_else(|| {
                Error::Io(std::io::Error::other(
                    "eglCreateDeviceANGLE not exported by ANGLE libEGL",
                ))
            })?;
        let get_platform_display = egl
            .get_proc_address("eglGetPlatformDisplayEXT")
            .ok_or_else(|| {
                Error::Io(std::io::Error::other(
                    "eglGetPlatformDisplayEXT not exported by ANGLE libEGL",
                ))
            })?;
        // SAFETY: both pointers were resolved by name from libEGL and have
        // the extension's documented signatures.
        let create_device: FnCreateDeviceANGLE = unsafe { std::mem::transmute(create_device) };
        // SAFETY: as above.
        let get_platform_display: FnGetPlatformDisplayEXT =
            unsafe { std::mem::transmute(get_platform_display) };

        // SAFETY: `device.raw()` is the live process device; ANGLE AddRefs it.
        let egl_device =
            unsafe { create_device(EGL_D3D11_DEVICE_ANGLE, device.raw(), std::ptr::null()) };
        if egl_device.is_null() {
            return Err(Error::Io(std::io::Error::other(format!(
                "eglCreateDeviceANGLE failed: {}",
                egl_error_name(egl)
            ))));
        }
        // SAFETY: `egl_device` is the EGLDeviceEXT just created; no attributes.
        let raw =
            unsafe { get_platform_display(EGL_PLATFORM_DEVICE_EXT, egl_device, std::ptr::null()) };
        if raw.is_null() {
            let why = egl_error_name(egl);
            release_egl_device(egl, egl_device);
            return Err(Error::Io(std::io::Error::other(format!(
                "eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT) returned NO_DISPLAY: {why}"
            ))));
        }
        // SAFETY: `raw` is a valid EGLDisplay per the spec.
        Ok((unsafe { egl::Display::from_ptr(raw) }, egl_device))
    }
}

/// Drop an `EGLDeviceEXT` created by `eglCreateDeviceANGLE`. Only used when
/// display creation fails afterwards: a display that came up owns its device
/// for the process's life (see [`SharedD3d11Display::egl_device`]).
fn release_egl_device(egl: &Egl, egl_device: *mut c_void) {
    type FnReleaseDeviceANGLE = unsafe extern "C" fn(device: *mut c_void) -> u32;
    let Some(release) = egl.get_proc_address("eglReleaseDeviceANGLE") else {
        return;
    };
    // SAFETY: the pointer was resolved by name from libEGL and has the
    // extension's documented signature.
    let release: FnReleaseDeviceANGLE = unsafe { std::mem::transmute(release) };
    // SAFETY: `egl_device` came from `eglCreateDeviceANGLE` and no display
    // was built on it, so nothing else references it. The boolean result is
    // ignored: this already runs on a failure path.
    let _ = unsafe { release(egl_device) };
}

/// The EGL error left by the last call, for a message. `get_error` also
/// clears it, so call this once per failure.
fn egl_error_name(egl: &Egl) -> String {
    match egl.get_error() {
        Some(e) => format!("{e:?}"),
        None => "no EGL error reported".to_string(),
    }
}

/// The `ID3D11Device*` ANGLE reports for `display`, through
/// `EGL_EXT_device_query` + `EGL_ANGLE_device_d3d`. Null when either
/// extension is missing or either query fails — the caller compares it
/// against the device it injected, so null reads as "not the same device".
fn query_angle_d3d11_device(egl: &Egl, display: egl::Display) -> *mut c_void {
    type FnQueryDisplayAttribEXT =
        unsafe extern "C" fn(display: egl::EGLDisplay, attribute: i32, value: *mut isize) -> u32;
    type FnQueryDeviceAttribEXT =
        unsafe extern "C" fn(device: *mut c_void, attribute: i32, value: *mut isize) -> u32;

    let (Some(query_display), Some(query_device)) = (
        egl.get_proc_address("eglQueryDisplayAttribEXT"),
        egl.get_proc_address("eglQueryDeviceAttribEXT"),
    ) else {
        return std::ptr::null_mut();
    };
    // SAFETY: both pointers were resolved by name from libEGL and have the
    // extension's documented signatures.
    let query_display: FnQueryDisplayAttribEXT = unsafe { std::mem::transmute(query_display) };
    // SAFETY: as above.
    let query_device: FnQueryDeviceAttribEXT = unsafe { std::mem::transmute(query_device) };

    let mut device: isize = 0;
    // SAFETY: `display` is an initialized EGLDisplay and `device` a valid
    // out-parameter for the one attribute asked for.
    if unsafe { query_display(display.as_ptr(), EGL_DEVICE_EXT, &mut device) } == 0 {
        return std::ptr::null_mut();
    }
    let mut d3d11: isize = 0;
    // SAFETY: `device` is the EGLDeviceEXT the display just reported, and
    // `d3d11` a valid out-parameter.
    if unsafe { query_device(device as *mut c_void, EGL_D3D11_DEVICE_ANGLE, &mut d3d11) } == 0 {
        return std::ptr::null_mut();
    }
    d3d11 as *mut c_void
}

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table (display-global, loaded once).
// ---------------------------------------------------------------------------

static GL_LOADED: OnceLock<()> = OnceLock::new();

fn load_gl_once(egl: &Egl) {
    GL_LOADED.get_or_init(|| {
        edgefirst_gl::load_with(|name| match egl.get_proc_address(name) {
            Some(ptr) => ptr as *const c_void,
            None => std::ptr::null(),
        });
    });
}

/// Create an OpenGL ES context: 3.1 first when `try_31` (compute shaders),
/// else / on failure 3.0. Returns the context and whether it is 3.1 — the
/// Linux `GlContext` pattern, so every context on one display agrees.
fn create_es_context(
    egl: &Egl,
    display: egl::Display,
    config: egl::Config,
    try_31: bool,
) -> Result<(egl::Context, bool)> {
    if try_31 {
        let attribs_31 = [
            egl::CONTEXT_MAJOR_VERSION,
            3,
            egl::CONTEXT_MINOR_VERSION,
            1,
            egl::NONE,
        ];
        if let Ok(ctx) = egl.create_context(display, config, None, &attribs_31) {
            return Ok((ctx, true));
        }
        debug!("GLES 3.1 context unavailable on this D3D11 device; falling back to 3.0");
    }
    let attribs_30 = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
    let ctx = egl
        .create_context(display, config, None, &attribs_30)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglCreateContext: {e:?}"))))?;
    Ok((ctx, false))
}

// ---------------------------------------------------------------------------
// Process-global ANGLE D3D11 display.
// ---------------------------------------------------------------------------

/// All process-global EGL state. Use [`shared_display`] to access.
pub(in crate::opengl_headless) struct SharedD3d11Display {
    /// Static-lifetime EGL handle over the leaked libEGL.dll.
    pub(in crate::opengl_headless) egl: Egl,
    pub(in crate::opengl_headless) display: egl::Display,
    pub(in crate::opengl_headless) config: egl::Config,
    /// Probe context + pbuffer used once at init to load GL and read the
    /// extension / version strings. Kept alive (never made current again)
    /// so the display's D3D device is not torn down while idle.
    probe_context: egl::Context,
    probe_pbuffer: egl::Surface,
    /// The probe context came up as GLES 3.1 (compute shaders available).
    pub(in crate::opengl_headless) has_compute: bool,
    /// `GL_EXT_color_buffer_float` — gates F32 PBO destinations.
    pub(in crate::opengl_headless) supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` — gates F16 PBO destinations.
    pub(in crate::opengl_headless) supports_f16_color: bool,
    /// The tensor crate's process device, which this display renders on and
    /// whose fence carries convert completions.
    pub(in crate::opengl_headless) device: &'static edgefirst_tensor::d3d11::D3d11Device,
    /// The `EGLDeviceEXT` wrapping [`Self::device`]. Never released: this
    /// display is process-global and leaked, so the device it was built on
    /// must outlive it. Kept here so the pairing is inspectable.
    #[allow(dead_code)]
    egl_device: *mut c_void,
    /// Human-readable name of the adapter the tensor crate chose.
    pub(in crate::opengl_headless) adapter: String,
}

// SAFETY: every member is either a leaked static, an EGL handle (ANGLE
// synchronizes display-level entry points internally), the process device
// (itself `Send + Sync`), or plain data. The `EGLDeviceEXT` is only ever
// read. The probe context is never made current after init.
unsafe impl Send for SharedD3d11Display {}
unsafe impl Sync for SharedD3d11Display {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedD3d11Display, String>> = OnceLock::new();

/// Acquire the process-global ANGLE D3D11 display, initialising it on the
/// first call. The error is cached too, so a failed ANGLE load is not
/// retried (and re-logged) for every processor.
pub(in crate::opengl_headless) fn shared_display() -> Result<&'static SharedD3d11Display> {
    SHARED_DISPLAY
        .get_or_init(|| init_shared_display().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|s| Error::Io(std::io::Error::other(s.clone())))
}

fn gl_string(name: u32) -> String {
    // SAFETY: a GL context is current on this thread (caller contract);
    // glGetString returns NULL or a static NUL-terminated string.
    unsafe {
        let ptr = edgefirst_gl::gl::GetString(name);
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        }
    }
}

fn init_shared_display() -> Result<SharedD3d11Display> {
    let _span =
        tracing::info_span!("image.gl_init", platform = "windows", backend = "d3d11").entered();

    // 1. The tensor crate's device (it owns adapter selection), then ANGLE
    //    libEGL.
    let device = edgefirst_tensor::d3d11::device()
        .map_err(|e| Error::Io(std::io::Error::other(format!("D3D11 device: {e}"))))?;
    let egl_lib = WindowsPlatform::load_egl_lib()
        .map_err(|e| Error::Io(std::io::Error::other(format!("ANGLE libEGL: {e}"))))?;
    // SAFETY: `egl_lib` is a live, leaked library handle; the loader
    // resolves every EGL 1.4 entry point or fails.
    let egl: Egl = unsafe {
        edgefirst_egl::Instance::<
            edgefirst_egl::Dynamic<&'static libloading::Library, edgefirst_egl::EGL1_4>,
        >::load_required_from(egl_lib)
    }
    .map_err(|e| Error::Io(std::io::Error::other(format!("EGL load: {e:?}"))))?;
    debug!("EGL dynamic instance loaded, version = {:?}", egl.version());

    // 2. D3D11 display on that device.
    let (display, egl_device) = WindowsPlatform::create_display(&egl, device)?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    let egl_version = egl
        .query_string(Some(display), egl::VERSION)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    debug!("ANGLE EGL {maj}.{min} initialised (process-global shared display): {egl_version}");
    // The two pointers must be the same object: a mismatch means ANGLE built
    // the display on a device of its own, and no texture tensor could ever be
    // imported into it. Fail here rather than at the first convert.
    let reported = query_angle_d3d11_device(&egl, display);
    debug!(
        "D3D11 device injected: {:?}, reported back by ANGLE: {reported:?}",
        device.raw()
    );
    if reported != device.raw() {
        return Err(Error::Io(std::io::Error::other(format!(
            "ANGLE did not adopt the injected D3D11 device: injected {:?}, display \
             reports {reported:?} (EGL_EXT_device_query / EGL_ANGLE_device_d3d)",
            device.raw()
        ))));
    }

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. GLES3 + pbuffer config. No bind-to-texture attribute: imports are
    //    EGLImages, and the pbuffer here only exists so a context can be
    //    made current.
    let cfg_attribs = [
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &cfg_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglChooseConfig: {e:?}"))))?
        .ok_or_else(|| {
            Error::NotSupported("no EGL config with GLES3 + PBUFFER on the D3D11 display".into())
        })?;

    // 4. Probe context (3.1 preferred) + a 16×16 pbuffer so it can be current.
    let (probe_context, has_compute) = create_es_context(&egl, display, config, true)?;
    let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
    let probe_pbuffer = egl
        .create_pbuffer_surface(display, config, &dummy_attribs)
        .map_err(|e| {
            destroy_context(&egl, display, probe_context);
            Error::Io(std::io::Error::other(format!(
                "eglCreatePbufferSurface(probe): {e:?}"
            )))
        })?;
    if let Err(e) = egl.make_current(
        display,
        Some(probe_pbuffer),
        Some(probe_pbuffer),
        Some(probe_context),
    ) {
        let _ = egl.destroy_surface(display, probe_pbuffer);
        destroy_context(&egl, display, probe_context);
        return Err(Error::Io(std::io::Error::other(format!(
            "eglMakeCurrent(probe): {e:?}"
        ))));
    }

    // 5. GL function table (once per process) + capability strings.
    load_gl_once(&egl);
    let extensions = gl_string(edgefirst_gl::gl::EXTENSIONS);
    let supports_f32_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_float");
    let supports_f16_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_half_float");
    let gl_version = gl_string(edgefirst_gl::gl::VERSION);
    let gl_renderer = gl_string(edgefirst_gl::gl::RENDERER);
    info!(
        "ANGLE D3D11 display ready on {}: {gl_renderer} — {gl_version} \
         (compute={has_compute}, f32_color={supports_f32_color}, f16_color={supports_f16_color})",
        device.adapter_label()
    );
    let _ = egl.make_current(display, None, None, None);

    Ok(SharedD3d11Display {
        egl,
        display,
        config,
        probe_context,
        probe_pbuffer,
        has_compute,
        supports_f32_color,
        supports_f16_color,
        device,
        egl_device,
        adapter: device.adapter_label().to_owned(),
    })
}

// ---------------------------------------------------------------------------
// Per-processor display: a private context on the shared D3D11 display.
// ---------------------------------------------------------------------------

/// One processor's GL bring-up state: a private EGL context (plus dummy
/// pbuffer) on the process-global shared ANGLE display. Created on the
/// processor's worker thread, made current there once, and held for the
/// thread's life. Not `Send`: dropped on the creating thread (the dispatch
/// wrapper guarantees both).
pub(in crate::opengl_headless) struct D3d11Display {
    pub(in crate::opengl_headless) shared: &'static SharedD3d11Display,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Duck-typed counterparts of the `GlContext` members the portable
    /// engine reads. D3D11 textures are the transfer backend on Windows.
    pub(in crate::opengl_headless) transfer_backend: TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
}

impl D3d11Display {
    /// This context's raw pointer, the value `begin_gpu_pass` records in
    /// `LAST_ACTIVE_CONTEXT`. Only the leaf's own tests read it.
    #[cfg(test)]
    fn context_ptr(&self) -> *mut c_void {
        self.context.as_ptr()
    }
}

impl Drop for D3d11Display {
    fn drop(&mut self) {
        // Runs on the owning worker thread: release, then destroy.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        destroy_context(&d.egl, d.display, self.context);
    }
}

/// An owned EGLImage over a tensor's `ID3D11Texture2D`. A binding made from
/// it needs no release at the end of a pass (unlike the macOS pbuffer
/// route), but is re-issued per use — see `PERSISTENT_TEX_BINDINGS`.
/// Dropping destroys the image; the texture stays the tensor's.
pub(in crate::opengl_headless) struct D3d11EglImage {
    shared: &'static SharedD3d11Display,
    pub(in crate::opengl_headless) image: egl::Image,
    /// The imported texture's own geometry, which a tensor narrowed by
    /// `configure_image` no longer fills — reported as the import extent so
    /// the engine samples the logical image rather than the whole texture.
    extent: (u32, u32),
}

impl Drop for D3d11EglImage {
    fn drop(&mut self) {
        // The image was created through the EGL 1.5 entry points, so the
        // upcast that made it is available again here.
        let Some(egl15) = self.shared.egl.upcast::<egl::EGL1_5>() else {
            log::error!("eglDestroyImage(D3D11 texture): EGL 1.5 no longer available");
            return;
        };
        if let Err(e) = egl15.destroy_image(self.shared.display, self.image) {
            log::error!("eglDestroyImage(D3D11 texture): {e:?}");
        }
    }
}

/// Wrap `texture` as an EGLImage on the shared display. `what` names the
/// role in the error message.
fn import_texture(
    display: &D3d11Display,
    texture: *mut c_void,
    layout: &edgefirst_tensor::d3d11_layout::D3d11ImageLayout,
    what: &str,
) -> Result<D3d11EglImage> {
    let _span =
        tracing::trace_span!("image.convert.gl.egl_import", target = "d3d11_texture").entered();
    let shared = display.shared;
    let egl15 = shared.egl.upcast::<egl::EGL1_5>().ok_or_else(|| {
        Error::NotSupported(
            "ANGLE libEGL does not expose EGL 1.5 (eglCreateImage), which the D3D11 \
             texture import needs"
                .into(),
        )
    })?;
    let attribs = super::super::d3d11_import::d3d11_image_attribs(layout.gl_internal_format, None);
    // The image ANGLE returns holds its own reference to `texture`: that is
    // what makes a cached import outliving its tensor sound, and what the
    // pool-recycle tier tests exercise every run.
    let image = egl15
        .create_image(
            shared.display,
            // SAFETY: EGL_NO_CONTEXT is the context a client-buffer image
            // target takes; the image is not tied to any GL context.
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            super::super::d3d11_import::EGL_D3D11_TEXTURE_ANGLE,
            // SAFETY: `texture` is the tensor's live `ID3D11Texture2D`. It
            // is live for the call, and ANGLE takes its own reference to it
            // for the image's life, so the import stays valid even after the
            // tensor drops -- which the cache relies on, since an entry
            // deliberately outlives the tensor it was built from
            // (`cache.rs`).
            unsafe { egl::ClientBuffer::from_ptr(texture) },
            &attribs,
        )
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "eglCreateImage(EGL_D3D11_TEXTURE_ANGLE) for {what}: {e:?}"
            )))
        })?;
    Ok(D3d11EglImage {
        shared,
        image,
        extent: (layout.texture_width as u32, layout.texture_height as u32),
    })
}

/// The tensor's texture and the geometry the HAL gave it, or an error
/// naming the role that is not texture-backed.
fn texture_of<T>(
    img: &Tensor<T>,
    what: &str,
) -> Result<(
    *mut c_void,
    edgefirst_tensor::d3d11_layout::D3d11ImageLayout,
)>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    match (img.d3d11_texture(), img.d3d11_layout()) {
        (Some(t), Some(l)) => {
            check_identity_names_texture(
                edgefirst_tensor::TensorTrait::buffer_identity(img).kind(),
                edgefirst_tensor::TensorTrait::buffer_identity(img).id(),
                t,
                what,
            )?;
            Ok((t, l))
        }
        _ => Err(Error::NotSupported(format!(
            "GL convert: {what} is not a D3D11 texture tensor"
        ))),
    }
}

/// The key `IdentityKind::D3d11Texture` must carry for `texture`: the
/// discriminant in the high bits, the texture's own address below it. Exactly
/// what `BufferIdentity::derived` builds and what both tensor backends derive
/// (`d3d11::texture::tex_key` in the static one, `ef_tensor_d3d11_texture` in
/// the dynamic one).
fn d3d11_identity_key(texture: *mut c_void) -> u64 {
    ((edgefirst_tensor::IdentityKind::D3d11Texture as u64) << 56) ^ (texture as usize as u64)
}

/// Refuse an identity that does not name `texture`.
///
/// The import cache keys on the identity and outlives the tensor that produced
/// it, so the identity has to name something the cached EGLImage itself keeps
/// alive -- otherwise a later, unrelated buffer arrives under the same key and
/// is served the old import. Only `D3d11Texture` over this texture's own
/// address qualifies: ANGLE's EGLImage holds a reference to the texture, so
/// that address stays taken while the entry lives. An `ef_tensor` handle
/// address does not qualify, and keying on one rendered into the previous
/// texture and left the new one blank.
///
/// Refused in every build rather than asserted in debug: the failure this
/// catches is silent wrong pixels. One enum compare and one XOR, and the error
/// falls the frame back to the CPU converter -- right instead of fast.
///
/// Split out from its two callers ([`texture_of`], which runs on a cache miss,
/// and [`GlPlatform::validate_import_identity`], which runs on every import)
/// so the rule has one home, and so the decision is testable without a tensor
/// whose identity disagrees with its texture -- which is not constructible
/// through any public API.
fn check_identity_names_texture(
    kind: edgefirst_tensor::IdentityKind,
    id: u64,
    texture: *mut c_void,
    what: &str,
) -> Result<()> {
    let expected = d3d11_identity_key(texture);
    if kind != edgefirst_tensor::IdentityKind::D3d11Texture || id != expected {
        return Err(Error::NotSupported(format!(
            "GL convert: {what} is a D3D11 texture but its buffer identity is {kind:?} \
             ({id:#x}), not this texture's own address ({expected:#x}); the EGLImage cache \
             outlives the tensor and cannot key on anything the import does not keep alive \
             -- see TensorDyn::derive_identity"
        )));
    }
    Ok(())
}

/// Refuse a texture whose DXGI format is not the one the HAL's layout table
/// picks for `fmt` — the engine samples the import as `fmt`, so a texture of
/// another format would be read with the wrong channel layout. Dimensions are
/// deliberately not compared: a view's own width and height are smaller than
/// the parent texture it imports. The check is skipped when the table has no
/// entry for `fmt` at these dimensions (a packed-RGB view whose width breaks
/// the 4-byte texel packing), which says nothing about the texture.
fn check_dxgi_format<T>(
    layout: &edgefirst_tensor::d3d11_layout::D3d11ImageLayout,
    img: &Tensor<T>,
    fmt: PixelFormat,
) -> Result<()>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    let (Some(w), Some(h)) = (img.width(), img.height()) else {
        return Ok(());
    };
    let Some(expected) =
        edgefirst_tensor::d3d11_layout::image_d3d11_layout(fmt, edgefirst_tensor::DType::U8, w, h)
    else {
        log::debug!("GL convert: no D3D11 layout for {fmt:?} at {w}x{h}; format check skipped");
        return Ok(());
    };
    if layout.dxgi_format != expected.dxgi_format {
        return Err(Error::NotSupported(format!(
            "GL convert: texture is DXGI format {} but {fmt:?} needs {}",
            layout.dxgi_format, expected.dxgi_format
        )));
    }
    Ok(())
}

/// Marker type: Windows ANGLE + Direct3D 11 platform.
pub(crate) struct AngleD3d11;

impl GlPlatform for AngleD3d11 {
    type Display = D3d11Display;
    type Import = D3d11EglImage;
    type ImportHandle = egl::Image;

    // The attachment survives the pass (there is nothing for
    // `end_gpu_pass` to release), but the engine must still re-attach on
    // every use: ANGLE's D3D11 texture storage keeps its own view of the
    // image and does not observe writes made to the underlying
    // `ID3D11Texture2D` outside GL, which is how a CPU producer refreshes a
    // recycled source (staging `CopyResource`). Skipping the re-attach then
    // samples the previous frame — `dma_recycle_grey_stale_read` fails on
    // WARP with the skip enabled and passes without it. The cost is one
    // `glEGLImageTargetTexture2DOES` per draw; the EGLImage import cache
    // itself is unaffected.
    const PERSISTENT_TEX_BINDINGS: bool = false;

    /// Every zero-copy float destination: planar F16 and F32, interleaved
    /// `Rgb`, and `Rgba`. Each is covered by
    /// `windows_float_destinations_match_the_cpu_converter_for_every_layout`
    /// on both adapters.
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        super::super::float_dispatch::ZeroCopyFloatSet::All;
    const EXTERNAL_OES: bool = false;

    fn load_gl_once(_display: &D3d11Display) {
        // Loaded once at shared-display init, before any context exists.
    }

    fn init_display(kind: Option<EglDisplayKind>) -> Result<D3d11Display> {
        if let Some(kind) = kind {
            debug!("EglDisplayKind::{kind} ignored on Windows — ANGLE/D3D11 is the only display");
        }
        let shared = shared_display()?;
        // Same ES version as the probe context so every context on the
        // display agrees on `has_compute`.
        let (context, has_compute) = create_es_context(
            &shared.egl,
            shared.display,
            shared.config,
            shared.has_compute,
        )?;
        let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
        let dummy_pbuffer = shared
            .egl
            .create_pbuffer_surface(shared.display, shared.config, &dummy_attribs)
            .map_err(|e| {
                destroy_context(&shared.egl, shared.display, context);
                Error::Io(std::io::Error::other(format!(
                    "eglCreatePbufferSurface (per-processor dummy): {e:?}"
                )))
            })?;
        // Made current once on the calling (worker) thread and held for the
        // thread's life.
        if let Err(e) = shared.egl.make_current(
            shared.display,
            Some(dummy_pbuffer),
            Some(dummy_pbuffer),
            Some(context),
        ) {
            let _ = shared.egl.destroy_surface(shared.display, dummy_pbuffer);
            destroy_context(&shared.egl, shared.display, context);
            return Err(Error::Io(std::io::Error::other(format!(
                "eglMakeCurrent (per-processor): {e:?}"
            ))));
        }
        // This context is now current on this thread, which disturbs the
        // display's shared state manager the same way a teardown does.
        invalidate_active_context();
        debug!(
            "Windows GL context up on {} (GLES {}, transfer=D3d11Texture)",
            shared.adapter,
            if has_compute { "3.1" } else { "3.0" }
        );
        Ok(D3d11Display {
            shared,
            context,
            dummy_pbuffer,
            transfer_backend: TransferBackend::D3d11Texture,
            has_compute,
        })
    }

    /// Windows keys its imports on a raw `ID3D11Texture2D` address that the
    /// tensor crate derives independently in each of its two backends, so the
    /// identity and the texture can disagree without anything else noticing.
    /// They did: the dynamic backend identified texture tensors by their
    /// recyclable `ef_tensor` handle address, and converts rendered into
    /// dropped textures. Checked here, before the cache lookup, so a cache
    /// **hit** cannot serve a stale image either -- `import_buffer` runs only
    /// on a miss and would never see it.
    ///
    /// A tensor with no texture is not this check's business: the import
    /// functions refuse it by name, with a message about what it is instead.
    fn validate_import_identity<T>(img: &Tensor<T>, what: &str) -> Result<()>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        let Some(texture) = img.d3d11_texture() else {
            return Ok(());
        };
        let identity = edgefirst_tensor::TensorTrait::buffer_identity(img);
        check_identity_names_texture(identity.kind(), identity.id(), texture, what)
    }

    fn import_buffer(
        display: &D3d11Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        _for_dst: bool,
    ) -> Result<D3d11EglImage> {
        // `EGL_ANGLE_image_d3d11_texture` has no sub-extent attribute, so
        // the image always covers the whole texture. That is what a
        // destination view wants (the tile offset is viewport state) and
        // what `d3d11_texture()` returns for a view either way, so there is
        // no branch here. A source whose logical image is smaller than its
        // texture -- a pool buffer narrowed by `configure_image` -- is
        // sampled through the extent reported by `import_extent`, which the
        // engine folds into the source rectangle.
        //
        // `for_dst` is unused for the same reason there is no bind-flag
        // refusal here: the tensor crate's layout ABI is frozen and carries
        // no bind flags, so this leaf cannot ask whether an externally
        // wrapped texture was created with `D3D11_BIND_RENDER_TARGET`. Such
        // a texture is imported and attached, and rejected at
        // `CheckFramebufferStatus`, which falls the frame back to the CPU
        // converter -- correct, at the cost of one `eglCreateImage` and one
        // FBO probe per frame. Textures the HAL allocates always carry the
        // flag.
        let (texture, layout) = texture_of(img, "source/destination")?;
        check_dxgi_format(&layout, img, fmt)?;
        import_texture(display, texture, &layout, "image")
    }

    fn import_buffer_nv_r8(
        display: &D3d11Display,
        img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<D3d11EglImage> {
        let (texture, layout) = texture_of(img, "NV source")?;
        // The HAL's own semi-planar allocations are always R8, and so is any
        // externally wrapped one the tensor crate accepted. That acceptance
        // is a contract this whole path rests on: `d3d11/texture.rs`'s
        // `validate_external` refuses a semi-planar texture whose width is
        // not its own staging row pitch, which is what makes the combined
        // plane's rows and the texel grid one number here. Checking the
        // format anyway costs nothing and makes the leaf self-defending.
        if layout.dxgi_format != edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R8_UNORM {
            return Err(Error::UnsupportedFormat(format!(
                "NV source texture is DXGI format {}, but the combined-plane path needs R8_UNORM ({})",
                layout.dxgi_format,
                edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R8_UNORM
            )));
        }
        import_texture(display, texture, &layout, "NV combined plane")
    }

    fn import_buffer_packed<T>(
        display: &D3d11Display,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> Result<D3d11EglImage>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        let (texture, layout) = texture_of(img, "packed destination")?;
        let expected = match fmt {
            super::PackedImportFormat::Rgba8888 => {
                edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R8G8B8A8_UNORM
            }
            super::PackedImportFormat::Rgba16161616F => {
                edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R16G16B16A16_FLOAT
            }
            super::PackedImportFormat::Rgba32323232F => {
                edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R32G32B32A32_FLOAT
            }
        };
        if layout.dxgi_format != expected
            || layout.texture_width != width
            || layout.texture_height != height
        {
            return Err(Error::NotSupported(format!(
                "GL convert: texture layout {layout:?} does not match the packed render \
                 surface {width}x{height} {fmt:?}"
            )));
        }
        import_texture(display, texture, &layout, "packed surface")
    }

    fn import_handle(import: &D3d11EglImage) -> egl::Image {
        import.image
    }

    fn import_extent(import: &D3d11EglImage) -> Option<(u32, u32)> {
        // Always `Some`: the image covers the whole `ID3D11Texture2D`, which
        // is larger than the logical image whenever `configure_image` narrowed
        // a pool tensor.
        Some(import.extent)
    }

    unsafe fn attach_tex_image_2d(_display: &D3d11Display, handle: egl::Image) -> Result<()> {
        // SAFETY: the caller has the intended texture bound on the active
        // unit and keeps the import alive for this call.
        unsafe {
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                edgefirst_gl::gl::TEXTURE_2D,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &D3d11Display,
        _handle: egl::Image,
    ) -> Result<()> {
        // Unreachable in practice: `EXTERNAL_OES` is false, so path
        // selection never picks the external sampler.
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES sampling is not used on ANGLE/D3D11".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &D3d11Display,
        handle: egl::Image,
    ) -> Result<()> {
        // SAFETY: the caller has the intended renderbuffer bound and keeps
        // the import alive for this call.
        unsafe {
            edgefirst_gl::gl::EGLImageTargetRenderbufferStorageOES(
                edgefirst_gl::gl::RENDERBUFFER,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    fn begin_gpu_pass(display: &D3d11Display) {
        // ANGLE's D3D11 backend keeps one `StateManager11` per display and
        // re-syncs a context's GL state onto the shared D3D device only from
        // `eglMakeCurrent` (`Context11::onMakeCurrent`). With every
        // processor's context permanently current on its own thread,
        // alternating processors, even fully serialized by the dispatch
        // wrapper, render with the previous context's applied state
        // (viewport, bindings, uniforms). So whenever a different context
        // issued the last commands, release and re-make this one current on
        // this thread, which makes ANGLE mark all state dirty. No cost when
        // consecutive messages come from the same processor.
        let ctx = display.context.as_ptr();
        if LAST_ACTIVE_CONTEXT.swap(ctx, Ordering::AcqRel) == ctx {
            return;
        }
        let d = display.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        if let Err(e) = d.egl.make_current(
            d.display,
            Some(display.dummy_pbuffer),
            Some(display.dummy_pbuffer),
            Some(display.context),
        ) {
            warn!("eglMakeCurrent (per-message context re-sync) failed: {e:?}");
            // The swap above already recorded this context as active, but it
            // is not: leaving the marker would make the next message skip its
            // own re-sync, which is the exact state this invalidation exists
            // to prevent.
            invalidate_active_context();
        }
    }

    fn end_gpu_pass(_display: &D3d11Display) {
        // EGLImage texture targets persist by design; nothing to release.
    }

    fn native_fence_sync(display: &D3d11Display) -> bool {
        display.shared.device.signal_supported()
    }

    /// A manual-reset event set when the device fence reaches the value
    /// covering this convert.
    ///
    /// `recorded` is that value when `record_completion` has already taken
    /// it for the same convert, which is the fenced path: the event and
    /// `dst.gpu_completion()` then name one value, at the cost of one flush
    /// and one signal for the whole call rather than two of each. Without
    /// one -- a caller exporting a fence over work no destination recorded --
    /// this submits and signals itself.
    fn export_completion_fence(
        display: &D3d11Display,
        recorded: Option<u64>,
    ) -> Result<Option<CompletionFence>> {
        let dev = display.shared.device;
        if !dev.signal_supported() {
            return Ok(None);
        }
        let value = match recorded {
            Some(v) => v,
            None => {
                // SAFETY: a GL context is current on this thread (the worker
                // holds it for its life); glFlush takes no arguments.
                unsafe { edgefirst_gl::gl::Flush() };
                let Some(v) = dev.signal() else {
                    return Ok(None);
                };
                v
            }
        };
        dev.event_for(value).map(Some).map_err(Error::Io)
    }

    /// Queues a fence signal behind the render -- a convert or a mask draw --
    /// and records its value on the destination so CUDA, D3D12 and
    /// other-device consumers can wait on it. The `glFlush` first: the value
    /// must cover this render's GL work even when the terminal `glFinish` is
    /// deferred to `flush`.
    ///
    /// `submit == false` keeps the D3D11 command buffer unsubmitted, so a
    /// batch of deferred converts costs one submission at its flush instead
    /// of one per tile. The values are still allocated under the device lock,
    /// so they order the tiles among themselves; what they do not carry until
    /// `submit_recorded` runs is the promise that a waiter sees them arrive.
    fn record_completion(display: &D3d11Display, dst: &mut TensorDyn, submit: bool) -> Option<u64> {
        if dst.memory() != edgefirst_tensor::TensorMemory::DmaBuf
            || !display.shared.device.signal_supported()
        {
            return None;
        }
        // SAFETY: as in `export_completion_fence` — a context is current on
        // this thread.
        unsafe { edgefirst_gl::gl::Flush() };
        let dev = display.shared.device;
        let value = if submit {
            dev.signal()
        } else {
            dev.signal_deferred()
        }?;
        if let Err(e) = dst.set_gpu_write(value) {
            log::debug!("record_completion: {e}");
        }
        Some(value)
    }

    fn submit_recorded(display: &D3d11Display) {
        display.shared.device.flush();
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use edgefirst_tensor::{TensorMapTrait as _, TensorTrait as _};
    use std::os::windows::io::AsRawHandle as _;

    /// The import-identity rule, decided directly.
    ///
    /// There is no honest end-to-end test for the failure it catches: a tensor
    /// whose `BufferIdentity` disagrees with its own `d3d11_texture()` is not
    /// constructible through any public API — both tensor backends derive the
    /// identity from the texture at construction and neither exposes a way to
    /// swap one without the other. That is the property the fix established,
    /// so the only way to exercise the refusal is to hand the decision the
    /// disagreement it is meant to reject. Hence the split: the rule lives in
    /// a function over `(kind, id, texture)` that both callers share.
    #[test]
    fn an_identity_that_does_not_name_the_texture_is_refused() {
        use edgefirst_tensor::IdentityKind;

        // A plausible texture address; never dereferenced.
        let texture = 0x2000_usize as *mut c_void;
        let good = d3d11_identity_key(texture);

        assert!(
            check_identity_names_texture(IdentityKind::D3d11Texture, good, texture, "dst").is_ok(),
            "the key both tensor backends derive for this texture must be accepted"
        );

        // The defect: right kind is not enough if the key names another
        // object. This is what a cache HIT would have served a stale image
        // for, since the miss path never runs.
        let other = d3d11_identity_key(0x3000_usize as *mut c_void);
        let err = check_identity_names_texture(IdentityKind::D3d11Texture, other, texture, "dst")
            .expect_err("a D3d11Texture key naming a different texture must be refused");
        assert!(
            matches!(err, Error::NotSupported(_)),
            "a refusal falls the frame back to the CPU converter: {err:?}"
        );

        // The shipped defect's actual shape: the dynamic backend's `HostPtr`
        // over a recyclable `ef_tensor` handle address.
        assert!(
            check_identity_names_texture(IdentityKind::HostPtr, 0x2000, texture, "src").is_err(),
            "a HostPtr identity must be refused even when its numeric key \
             happens to equal the texture address"
        );

        // Every other kind is equally wrong here, including the ones that are
        // legitimate cache keys on their own platforms.
        for kind in [
            IdentityKind::DmaBuf,
            IdentityKind::Shm,
            IdentityKind::IoSurface,
            IdentityKind::AHardwareBufferId,
            IdentityKind::AHardwareBufferPtr,
            IdentityKind::Pbo,
        ] {
            assert!(
                check_identity_names_texture(kind, good, texture, "src").is_err(),
                "{kind:?} is not a D3D11 texture identity"
            );
        }
    }

    /// Confirms `load_egl_lib` can locate ANGLE's libEGL.dll when
    /// `EDGEFIRST_ANGLE_PATH` points at one; skips otherwise (CI runs
    /// without ANGLE must not fail this test).
    #[test]
    fn load_egl_lib_finds_angle_or_skips() {
        let Some(dir) = std::env::var_os("EDGEFIRST_ANGLE_PATH") else {
            eprintln!("EDGEFIRST_ANGLE_PATH unset — skipping ANGLE load probe");
            return;
        };
        if !Path::new(&dir).join("libEGL.dll").is_file() {
            eprintln!("no libEGL.dll under EDGEFIRST_ANGLE_PATH — skipping ANGLE load probe");
            return;
        }
        WindowsPlatform::load_egl_lib()
            .expect("ANGLE libEGL.dll should load from EDGEFIRST_ANGLE_PATH");
    }

    /// A context teardown must forget the last-active context for EVERY
    /// processor, not just its own: ANGLE's per-display state manager is
    /// disturbed by the destruction, so a processor that was already the last
    /// active one must still re-sync on its next message. Skipping that
    /// re-sync faulted inside `glDrawArrays` in about one four-processor
    /// convert-then-teardown run in fifteen.
    #[test]
    fn destroying_a_context_forgets_another_processors_last_active_context() {
        // Serializes this display/context work with the GL workers of
        // other tests in this binary; see `lifecycle_guard`.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(keeper) = AngleD3d11::init_display(None) else {
            eprintln!("SKIP: no ANGLE");
            return;
        };
        let Ok(doomed) = AngleD3d11::init_display(None) else {
            eprintln!("SKIP: no ANGLE");
            return;
        };
        // `keeper` issues the last commands, so it is the context the
        // per-message re-sync would skip.
        AngleD3d11::begin_gpu_pass(&keeper);
        assert_eq!(
            LAST_ACTIVE_CONTEXT.load(Ordering::Acquire),
            keeper.context_ptr()
        );
        drop(doomed);
        assert!(
            LAST_ACTIVE_CONTEXT.load(Ordering::Acquire).is_null(),
            "another context's teardown must invalidate the re-sync cache"
        );
        // And the invalidation makes `keeper`'s next pass re-make it current.
        AngleD3d11::begin_gpu_pass(&keeper);
        assert_eq!(
            LAST_ACTIVE_CONTEXT.load(Ordering::Acquire),
            keeper.context_ptr()
        );
    }

    /// The display ANGLE brings up is the tensor crate's device, both as the
    /// leaf recorded it and as ANGLE reports it back through
    /// `EGL_ANGLE_device_d3d`.
    #[test]
    fn shared_display_runs_on_the_tensor_crates_device() {
        // Serializes this display/context work with the GL workers of
        // other tests in this binary; see `lifecycle_guard`.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(shared) = shared_display() else {
            eprintln!("SKIP: no ANGLE");
            return;
        };
        let dev = edgefirst_tensor::d3d11::device().expect("device");
        assert_eq!(shared.device.raw(), dev.raw());
        assert_eq!(
            query_angle_d3d11_device(&shared.egl, shared.display),
            dev.raw()
        );
    }

    /// A texture tensor imported as an EGLImage, attached to a GL texture
    /// and rendered into, carries the rendered pixels back to the CPU map:
    /// the zero-copy round trip the whole leaf exists for.
    #[test]
    fn rgba8_texture_imports_renders_and_reads_back() {
        // Serializes this display/context work with the GL workers of
        // other tests in this binary; see `lifecycle_guard`.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(display) = AngleD3d11::init_display(None) else {
            eprintln!("SKIP: no ANGLE");
            return;
        };
        let t = edgefirst_tensor::Tensor::<u8>::image(
            64,
            32,
            PixelFormat::Rgba,
            Some(edgefirst_tensor::TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::Read,
        )
        .unwrap();
        let import = AngleD3d11::import_buffer(&display, &t, PixelFormat::Rgba, true).unwrap();
        let handle = AngleD3d11::import_handle(&import);
        let mut tex = 0;
        let mut fbo = 0;
        // SAFETY: `init_display` left a context current on this thread, and
        // every GL name below was generated in this block. The import is
        // alive for the whole block.
        unsafe {
            edgefirst_gl::gl::GenTextures(1, &mut tex);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, tex);
            AngleD3d11::attach_tex_image_2d(&display, handle).unwrap();
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                tex,
                0,
            );
            assert_eq!(
                edgefirst_gl::gl::CheckFramebufferStatus(edgefirst_gl::gl::FRAMEBUFFER),
                edgefirst_gl::gl::FRAMEBUFFER_COMPLETE
            );
            edgefirst_gl::gl::ClearColor(1.0, 0.5, 0.0, 1.0);
            edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            edgefirst_gl::gl::Finish();
        }
        let m = t.map_read().unwrap();
        let px = &m.as_slice()[..4];
        assert_eq!(px[0], 255, "red {px:?}");
        // 0.5 lands on 127 or 128 depending on the driver's rounding.
        assert!(px[1] == 128 || px[1] == 127, "green {px:?}");
        assert_eq!(px[2], 0, "blue {px:?}");
        assert_eq!(px[3], 255, "alpha {px:?}");
    }

    /// The exported completion fence is an event on the tensor crate's
    /// device fence; after a `glFinish` the value it waits for is reached.
    #[test]
    fn export_completion_fence_yields_a_set_event_after_finish() {
        // Serializes this display/context work with the GL workers of
        // other tests in this binary; see `lifecycle_guard`.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(display) = AngleD3d11::init_display(None) else {
            eprintln!("SKIP: no ANGLE");
            return;
        };
        assert!(AngleD3d11::native_fence_sync(&display));
        let fence = AngleD3d11::export_completion_fence(&display, None)
            .unwrap()
            .expect("fence");
        // SAFETY: `fence` owns a live manual-reset event handle.
        assert_eq!(
            unsafe { WaitForSingleObject(fence.as_raw_handle() as _, 5000) },
            0
        );
    }

    /// Per-processor context bring-up latency (budget <50 ms each, as on
    /// macOS). Ignored: needs ANGLE + a D3D11 device; run on demand:
    /// `cargo test -p edgefirst-image --release --lib windows::tests -- --ignored --nocapture`
    #[test]
    #[ignore = "needs ANGLE + a D3D11 device; run on demand"]
    fn per_processor_context_bring_up_latency() {
        use std::time::Instant;
        // Serializes this display/context work with the GL workers of
        // other tests in this binary; see `lifecycle_guard`.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let t0 = Instant::now();
        let first = AngleD3d11::init_display(None).expect("first display");
        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(first);
        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let t0 = Instant::now();
                    let d = AngleD3d11::init_display(None).expect("display");
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    drop(d);
                    (i, ms)
                })
            })
            .collect();
        for h in handles {
            let (i, ms) = h.join().expect("thread");
            println!("per-processor context {i}: {ms:.1} ms");
            assert!(
                ms < 50.0,
                "context bring-up {ms:.1} ms exceeds 50 ms budget"
            );
        }
        println!("first (incl. shared display init): {first_ms:.1} ms");
    }
}
