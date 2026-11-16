// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Windows implementation of [`GlPlatform`]: ANGLE (GLES → Direct3D 11) on
//! the tensor crate's D3D11 device, with zero-copy texture transfers.
//!
//! * Loader: `libEGL.dll` is found through `EDGEFIRST_ANGLE_PATH`, then next
//!   to the module containing this code, then next to the executable, then
//!   on the default DLL search path. Absolute candidates load with
//!   `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`.
//! * Display: `eglCreateDeviceANGLE` wraps the tensor crate's `ID3D11Device`
//!   as an `EGLDeviceEXT` and
//!   `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT)` turns it into the
//!   display. The tensor crate selects the adapter
//!   (`EDGEFIRST_D3D11_ADAPTER`, alias `EDGEFIRST_ANGLE_ADAPTER`).
//! * Transfer: the display reports [`TransferBackend::D3d11Texture`]. A
//!   tensor's `ID3D11Texture2D` is imported as an EGLImage through
//!   `EGL_ANGLE_image_d3d11_texture` and bound with
//!   `glEGLImageTargetTexture2DOES` on every use.
//!
//! The three kernel32 calls are declared here, and the D3D11 device and
//! textures arrive as raw `*mut c_void` from the tensor crate's accessors.

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
/// display (see [`GlPlatform::begin_gpu_pass`]).
static LAST_ACTIVE_CONTEXT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

/// Forgets which context issued the last GL commands. The next message on
/// any processor re-makes its own context current
/// ([`GlPlatform::begin_gpu_pass`]).
fn invalidate_active_context() {
    LAST_ACTIVE_CONTEXT.store(std::ptr::null_mut(), Ordering::Release);
}

/// Calls [`invalidate_active_context`], then `eglDestroyContext`.
fn destroy_context(egl: &Egl, display: egl::Display, context: egl::Context) {
    invalidate_active_context();
    let _ = egl.destroy_context(display, context);
}

// ---------------------------------------------------------------------------
// EGL constants (EGL/egl.h + EGL/eglext_angle.h).
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

/// Cached libEGL handle. Leaked at first load and never freed.
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
}
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

/// Windows-platform helpers: locate and load libEGL, and bring up a D3D11
/// display.
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
    /// this code. `None` if either call fails.
    fn module_dir() -> Option<PathBuf> {
        use std::os::windows::ffi::OsStringExt as _;
        let anchor = Self::module_dir as *const () as *const u16;
        let mut module: *mut c_void = std::ptr::null_mut();
        // SAFETY: `anchor` is an address inside this module's image and
        // `module` is a valid out-pointer. No reference is taken.
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
        // 32 K UTF-16 units, the maximum path length.
        let mut buf = vec![0u16; 32 * 1024];
        // SAFETY: `module` is a valid module handle and `buf` is a writable
        // buffer of the passed length.
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
    /// a flat sibling of `libEGL.dll`.
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
            // SAFETY: loading runs the DLL's initializers; ANGLE's libEGL
            // has none with side effects.
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

    /// Builds an ANGLE display on the tensor crate's D3D11 device.
    ///
    /// Returns the display and the `EGLDeviceEXT` it was built from. The
    /// device is not released once a display exists.
    ///
    /// `egl` must wrap a libEGL handle from [`Self::load_egl_lib`].
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

/// Drop an `EGLDeviceEXT` created by `eglCreateDeviceANGLE`. Used when
/// display creation fails afterwards.
fn release_egl_device(egl: &Egl, egl_device: *mut c_void) {
    type FnReleaseDeviceANGLE = unsafe extern "C" fn(device: *mut c_void) -> u32;
    let Some(release) = egl.get_proc_address("eglReleaseDeviceANGLE") else {
        return;
    };
    // SAFETY: the pointer was resolved by name from libEGL and has the
    // extension's documented signature.
    let release: FnReleaseDeviceANGLE = unsafe { std::mem::transmute(release) };
    // SAFETY: `egl_device` came from `eglCreateDeviceANGLE` and no display
    // was built on it, so nothing else references it.
    let _ = unsafe { release(egl_device) };
}

/// The EGL error left by the last call, for a message. Reading it clears
/// it.
fn egl_error_name(egl: &Egl) -> String {
    match egl.get_error() {
        Some(e) => format!("{e:?}"),
        None => "no EGL error reported".to_string(),
    }
}

/// The `ID3D11Device*` ANGLE reports for `display`, through
/// `EGL_EXT_device_query` + `EGL_ANGLE_device_d3d`. Null when either
/// extension is missing or either query fails.
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

/// Create an OpenGL ES context: 3.1 first when `try_31`, else or on failure
/// 3.0. Returns the context and whether it is 3.1.
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
    /// extension / version strings. Kept alive and never made current again.
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
    /// The `EGLDeviceEXT` wrapping [`Self::device`]. Never released.
    #[allow(dead_code)]
    egl_device: *mut c_void,
    /// Human-readable name of the adapter the tensor crate chose.
    pub(in crate::opengl_headless) adapter: String,
}

// SAFETY: every member is a leaked static, an EGL handle, the process device
// (`Send + Sync`), or plain data. The `EGLDeviceEXT` is only read and the
// probe context is never made current after init.
unsafe impl Send for SharedD3d11Display {}
unsafe impl Sync for SharedD3d11Display {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedD3d11Display, String>> = OnceLock::new();

/// Acquire the process-global ANGLE D3D11 display, initialising it on the
/// first call. A failed initialisation is cached and returned to every later
/// caller.
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

    // 1. The tensor crate's device, then ANGLE libEGL.
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
    // ANGLE must report the device it was given.
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

    // 3. GLES3 + pbuffer config.
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

    // 4. Probe context (3.1 preferred) + a 16×16 pbuffer to make it current on.
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
/// thread's life. Not `Send`: dropped on the creating thread.
pub(in crate::opengl_headless) struct D3d11Display {
    pub(in crate::opengl_headless) shared: &'static SharedD3d11Display,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Counterparts of the `GlContext` members the portable engine reads.
    pub(in crate::opengl_headless) transfer_backend: TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
}

impl D3d11Display {
    /// This context's raw pointer, the value `begin_gpu_pass` records in
    /// `LAST_ACTIVE_CONTEXT`.
    #[cfg(test)]
    fn context_ptr(&self) -> *mut c_void {
        self.context.as_ptr()
    }
}

impl Drop for D3d11Display {
    fn drop(&mut self) {
        // Release the context, then destroy it.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        destroy_context(&d.egl, d.display, self.context);
    }
}

/// An owned EGLImage over a tensor's `ID3D11Texture2D`. Dropping destroys
/// the image; the texture stays the tensor's.
pub(in crate::opengl_headless) struct D3d11EglImage {
    shared: &'static SharedD3d11Display,
    pub(in crate::opengl_headless) image: egl::Image,
    /// The imported texture's own geometry, reported as the import extent.
    extent: (u32, u32),
}

impl Drop for D3d11EglImage {
    fn drop(&mut self) {
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
    import_plane(
        display,
        texture,
        layout.gl_internal_format,
        None,
        (layout.texture_width as u32, layout.texture_height as u32),
        what,
    )
}

/// Wrap plane `plane` of a native NV12 texture: plane 0 is its luma as
/// `GL_RED_EXT` at the full grid, plane 1 its interleaved chroma as
/// `GL_RG_EXT` at half the width and height.
fn import_nv12_plane(
    display: &D3d11Display,
    texture: *mut c_void,
    layout: &edgefirst_tensor::d3d11_layout::D3d11ImageLayout,
    plane: u32,
) -> Result<D3d11EglImage> {
    let (w, h) = (layout.texture_width as u32, layout.texture_height as u32);
    let (format, extent, what) = match plane {
        0 => (
            edgefirst_tensor::d3d11_layout::GL_RED_EXT,
            (w, h),
            "NV12 luma plane",
        ),
        _ => (
            edgefirst_tensor::d3d11_layout::GL_RG_EXT,
            (w / 2, h / 2),
            "NV12 chroma plane",
        ),
    };
    import_plane(display, texture, format, Some(plane), extent, what)
}

/// Wrap `texture`, or one plane of it, as an EGLImage sampled as
/// `gl_internal_format` over `extent` texels.
fn import_plane(
    display: &D3d11Display,
    texture: *mut c_void,
    gl_internal_format: u32,
    plane: Option<u32>,
    extent: (u32, u32),
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
    let attribs = super::super::d3d11_import::d3d11_image_attribs(gl_internal_format, plane);
    // The image ANGLE returns holds its own reference to `texture`.
    let image = egl15
        .create_image(
            shared.display,
            // SAFETY: a client-buffer image target takes EGL_NO_CONTEXT.
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            super::super::d3d11_import::EGL_D3D11_TEXTURE_ANGLE,
            // SAFETY: `texture` is the tensor's live `ID3D11Texture2D`, and
            // ANGLE takes its own reference to it for the image's life.
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
        extent,
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

/// The key `IdentityKind::D3d11Texture` carries for `texture`: the
/// discriminant in the high bits, the texture's own address below it.
fn d3d11_identity_key(texture: *mut c_void) -> u64 {
    ((edgefirst_tensor::IdentityKind::D3d11Texture as u64) << 56) ^ (texture as usize as u64)
}

/// Refuse an identity that does not name `texture`: its kind must be
/// `D3d11Texture` and its id [`d3d11_identity_key`] of `texture`.
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
/// picks for `fmt`. Dimensions are not compared. The check is skipped when
/// the table has no entry for `fmt` at these dimensions.
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

    // The engine re-attaches the EGLImage on every use.
    const PERSISTENT_TEX_BINDINGS: bool = false;

    /// Every zero-copy float destination: planar F16 and F32, interleaved
    /// `Rgb`, and `Rgba`.
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        super::super::float_dispatch::ZeroCopyFloatSet::All;
    const EXTERNAL_OES: bool = false;

    fn load_gl_once(_display: &D3d11Display) {
        // Loaded once at shared-display init.
    }

    fn init_display(kind: Option<EglDisplayKind>) -> Result<D3d11Display> {
        if let Some(kind) = kind {
            debug!("EglDisplayKind::{kind} ignored on Windows — ANGLE/D3D11 is the only display");
        }
        let shared = shared_display()?;
        // Same ES version as the probe context.
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
        // Made current once on the calling thread and held for the thread's
        // life.
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

    /// A destination that does not start at its texture's origin is placed
    /// only when it carries a `view_origin`.
    fn dst_import_places<T>(img: &Tensor<T>) -> bool
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        super::unplaced_destination(img).is_none()
    }

    /// Checks that a texture tensor's buffer identity names its own
    /// `ID3D11Texture2D`. A tensor with no texture passes.
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
        for_dst: bool,
    ) -> Result<D3d11EglImage> {
        // The image covers the whole texture. A source whose pixels start at
        // an offset is refused.
        if !for_dst {
            super::refuse_offset_source(img, "source", "EGL_ANGLE_image_d3d11_texture")?;
        }
        let (texture, layout) = texture_of(img, "source/destination")?;
        check_dxgi_format(&layout, img, fmt)?;
        import_texture(display, texture, &layout, "image")
    }

    fn import_buffer_nv_r8(
        display: &D3d11Display,
        img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<D3d11EglImage> {
        super::refuse_offset_source(img, "NV source", "EGL_ANGLE_image_d3d11_texture")?;
        let (texture, layout) = texture_of(img, "NV source")?;
        // A native NV12 texture's luma import is its plane 0.
        if layout.is_native_nv12() {
            return import_nv12_plane(display, texture, &layout, 0);
        }
        // A combined-plane semi-planar texture must be R8.
        if layout.dxgi_format != edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R8_UNORM {
            return Err(Error::UnsupportedFormat(format!(
                "NV source texture is DXGI format {}, but the combined-plane path needs R8_UNORM ({})",
                layout.dxgi_format,
                edgefirst_tensor::d3d11_layout::DXGI_FORMAT_R8_UNORM
            )));
        }
        import_texture(display, texture, &layout, "NV combined plane")
    }

    /// Imports plane 1 of a native NV12 texture. A combined-plane texture
    /// has no chroma plane of its own and is refused.
    fn import_buffer_nv_chroma(
        display: &D3d11Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> Result<D3d11EglImage> {
        super::refuse_offset_source(img, "NV source", "EGL_ANGLE_image_d3d11_texture")?;
        let (texture, layout) = texture_of(img, "NV chroma source")?;
        if !layout.is_native_nv12() {
            return Err(Error::NotSupported(format!(
                "{fmt:?} in a combined-plane texture has no chroma plane of its own"
            )));
        }
        import_nv12_plane(display, texture, &layout, 1)
    }

    fn nv_native_planes(img: &Tensor<u8>) -> Option<(usize, usize)> {
        img.d3d11_layout()
            .filter(|l| l.is_native_nv12())
            .map(|l| (l.texture_width, l.texture_height))
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
        // Always `Some`: the image covers the whole `ID3D11Texture2D`.
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
        // When a different context issued the last commands, release this
        // one and make it current again.
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
            // This context did not become current; clear the marker.
            invalidate_active_context();
        }
    }

    fn end_gpu_pass(_display: &D3d11Display) {
        // Nothing to release.
    }

    fn native_fence_sync(display: &D3d11Display) -> bool {
        display.shared.device.signal_supported()
    }

    /// A manual-reset event set when the device fence reaches the value
    /// covering this convert.
    ///
    /// `recorded` is that value when `record_completion` has taken it for
    /// the same convert. Without one this flushes and signals the fence
    /// itself.
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
                // SAFETY: a GL context is current on this thread.
                unsafe { edgefirst_gl::gl::Flush() };
                let Some(v) = dev.signal() else {
                    return Ok(None);
                };
                v
            }
        };
        dev.event_for(value).map(Some).map_err(Error::Io)
    }

    /// Flushes, queues a fence signal behind the render and records its
    /// value on the destination. Returns the value, or `None` when none was
    /// recorded.
    ///
    /// `submit == false` leaves the D3D11 command buffer unsubmitted until
    /// `submit_recorded` runs.
    fn record_completion(display: &D3d11Display, dst: &mut TensorDyn, submit: bool) -> Option<u64> {
        if dst.memory() != edgefirst_tensor::TensorMemory::DmaBuf
            || !display.shared.device.signal_supported()
        {
            return None;
        }
        // SAFETY: a GL context is current on this thread.
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

    /// `check_identity_names_texture` accepts only a `D3d11Texture` identity whose id names the texture.
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

        // The right kind with an id naming another object.
        let other = d3d11_identity_key(0x3000_usize as *mut c_void);
        let err = check_identity_names_texture(IdentityKind::D3d11Texture, other, texture, "dst")
            .expect_err("a D3d11Texture key naming a different texture must be refused");
        assert!(
            matches!(err, Error::NotSupported(_)),
            "a refusal falls the frame back to the CPU converter: {err:?}"
        );

        // A `HostPtr` identity over the same address.
        assert!(
            check_identity_names_texture(IdentityKind::HostPtr, 0x2000, texture, "src").is_err(),
            "a HostPtr identity must be refused even when its numeric key \
             happens to equal the texture address"
        );

        // Every other kind.
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

    /// `load_egl_lib` locates ANGLE's libEGL.dll when `EDGEFIRST_ANGLE_PATH` points at one, else skips.
    #[test]
    fn load_egl_lib_finds_angle_or_skips() {
        let Some(dir) = std::env::var_os("EDGEFIRST_ANGLE_PATH") else {
            crate::test_support::report_skip("EDGEFIRST_ANGLE_PATH unset — no ANGLE load probe");
            return;
        };
        if !Path::new(&dir).join("libEGL.dll").is_file() {
            crate::test_support::report_skip(
                "no libEGL.dll under EDGEFIRST_ANGLE_PATH — no ANGLE load probe",
            );
            return;
        }
        WindowsPlatform::load_egl_lib()
            .expect("ANGLE libEGL.dll should load from EDGEFIRST_ANGLE_PATH");
    }

    /// Destroying a context clears the last-active context another processor recorded.
    #[test]
    fn destroying_a_context_forgets_another_processors_last_active_context() {
        // Serializes with the GL workers of other tests in this binary.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(keeper) = AngleD3d11::init_display(None) else {
            crate::test_support::report_skip("no ANGLE");
            return;
        };
        let Ok(doomed) = AngleD3d11::init_display(None) else {
            crate::test_support::report_skip("no ANGLE");
            return;
        };
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
        AngleD3d11::begin_gpu_pass(&keeper);
        assert_eq!(
            LAST_ACTIVE_CONTEXT.load(Ordering::Acquire),
            keeper.context_ptr()
        );
    }

    /// The shared display runs on the tensor crate's device, as recorded and as ANGLE reports it.
    #[test]
    fn shared_display_runs_on_the_tensor_crates_device() {
        // Serializes with the GL workers of other tests in this binary.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(shared) = shared_display() else {
            crate::test_support::report_skip("no ANGLE");
            return;
        };
        let dev = edgefirst_tensor::d3d11::device().expect("device");
        assert_eq!(shared.device.raw(), dev.raw());
        assert_eq!(
            query_angle_d3d11_device(&shared.egl, shared.display),
            dev.raw()
        );
    }

    /// A texture tensor imported as an EGLImage and rendered into reads back the rendered pixels.
    #[test]
    fn rgba8_texture_imports_renders_and_reads_back() {
        // Serializes with the GL workers of other tests in this binary.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(display) = AngleD3d11::init_display(None) else {
            crate::test_support::report_skip("no ANGLE");
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
        // SAFETY: `init_display` left a context current on this thread, every
        // GL name below was generated in this block, and the import is alive
        // for the whole block.
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

    /// The exported completion fence is an event on the device fence, set after a `glFinish`.
    #[test]
    fn export_completion_fence_yields_a_set_event_after_finish() {
        // Serializes with the GL workers of other tests in this binary.
        let _lifecycle = super::super::super::threaded::lifecycle_guard();
        let Ok(display) = AngleD3d11::init_display(None) else {
            crate::test_support::report_skip("no ANGLE");
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

    /// Per-processor context bring-up takes under 50 ms each.
    #[test]
    #[ignore = "needs ANGLE + a D3D11 device; run on demand"]
    fn per_processor_context_bring_up_latency() {
        use std::time::Instant;
        // Serializes with the GL workers of other tests in this binary.
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
