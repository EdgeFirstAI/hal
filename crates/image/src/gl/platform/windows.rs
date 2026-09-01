// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Windows implementation of [`GlPlatform`]: ANGLE (GLES → Direct3D 11)
//! display bring-up with PBO transfers.
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
//! 2. **Display** — `eglGetPlatformDisplayEXT` selects the D3D11 backend
//!    and, via `EDGEFIRST_ANGLE_ADAPTER`, the adapter: ANGLE's default
//!    hardware adapter, WARP (software, for CI), or a specific adapter by
//!    LUID / description substring resolved through DXGI enumeration.
//! 3. **Transfer** — there is no zero-copy buffer kind on Windows yet, so
//!    the display reports [`TransferBackend::Pbo`]: `Mem` sources upload
//!    through `glTexImage2D` and destinations are PBO tensors read back
//!    through `GL_PIXEL_PACK_BUFFER` — the same path desktop Linux takes on
//!    NVIDIA where DMA-BUF import is unavailable. The `Import` type
//!    ([`D3dTexturePbuffer`]) is the shape the D3D11 shared-texture
//!    follow-on fills via `EGL_ANGLE_d3d_texture_client_buffer`; today the
//!    three `import_*` methods return `NotSupported` and are unreachable
//!    because no tensor on Windows is zero-copy-backed.
//!
//! The shared-display / per-processor-context shape is deliberately laid
//! out function-for-function like `angle.rs` (the Android leaf did the
//! same) so a later `angle_common` extraction is a pure move.
//!
//! No `windows-sys` dependency: the two kernel32 calls are declared here
//! and the DXGI enumeration uses hand-written COM vtable slots over a
//! `dxgi.dll` loaded with `libloading` — the HAL's dlopen convention for
//! optional platform libraries.

use super::super::{CompletionFence, Egl, EglDisplayKind, TransferBackend};
use super::GlPlatform;
use crate::{Error, Result};
use edgefirst_egl as egl;
use edgefirst_tensor::{PixelFormat, Tensor};
use log::{debug, info, warn};
use std::cell::RefCell;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

/// The EGL context that issued the most recent GL commands on the shared
/// display (see [`GlPlatform::begin_gpu_pass`] below). Touched only under
/// the dispatch wrapper's process-wide message lock (ANGLE takes the Full
/// serialization policy), so a plain atomic is enough.
static LAST_ACTIVE_CONTEXT: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

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
const EGL_BACK_BUFFER: i32 = 0x3084;

/// `EGL_ANGLE_platform_angle`.
const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: i32 = 0x3203;
/// `EGL_ANGLE_platform_angle_d3d`.
const EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE: i32 = 0x3208;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE: i32 = 0x3209;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE: i32 = 0x320A;
const EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE: i32 = 0x320B;
/// `EGL_ANGLE_platform_angle_d3d_luid`.
const EGL_PLATFORM_ANGLE_D3D_LUID_HIGH_ANGLE: i32 = 0x34A0;
const EGL_PLATFORM_ANGLE_D3D_LUID_LOW_ANGLE: i32 = 0x34A1;

/// Environment variable selecting the D3D11 adapter (see
/// [`AdapterSelection`]).
pub(crate) const ADAPTER_ENV: &str = "EDGEFIRST_ANGLE_ADAPTER";

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
    /// this code — the `@loader_path` analogue. `None` if the OS refuses.
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
            // libEGL is well-behaved. DLL_LOAD_DIR lets it find its sibling
            // libGLESv2.dll; DEFAULT_DIRS covers System32 (d3d11, dxgi,
            // d3dcompiler_47).
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

    /// Bring up an ANGLE Direct3D 11 display on the selected adapter.
    ///
    /// `egl` must wrap a libEGL handle from [`Self::load_egl_lib`]: the call
    /// goes through ANGLE's `EGL_EXT_platform_base` client extension
    /// (`eglGetPlatformDisplayEXT`), which Windows has no system EGL for.
    pub(in super::super) fn create_display(
        egl: &Egl,
        sel: &AdapterSelection,
    ) -> Result<egl::Display> {
        type FnGetPlatformDisplayEXT = unsafe extern "C" fn(
            platform: u32,
            native: *mut c_void,
            attribs: *const i32,
        ) -> egl::EGLDisplay;

        let ptr = egl
            .get_proc_address("eglGetPlatformDisplayEXT")
            .ok_or_else(|| {
                Error::Io(std::io::Error::other(
                    "eglGetPlatformDisplayEXT not exported by ANGLE libEGL",
                ))
            })?;
        // SAFETY: the pointer comes from EGL's own dispatch table and
        // matches the documented C signature.
        let get_platform_display: FnGetPlatformDisplayEXT = unsafe { std::mem::transmute(ptr) };

        let mut attribs = vec![
            EGL_PLATFORM_ANGLE_TYPE_ANGLE,
            EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
        ];
        match sel {
            AdapterSelection::Hardware => attribs.extend([
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE,
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE,
            ]),
            AdapterSelection::Warp => attribs.extend([
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE,
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE,
            ]),
            AdapterSelection::Luid { high, low } => attribs.extend([
                EGL_PLATFORM_ANGLE_D3D_LUID_HIGH_ANGLE,
                *high,
                EGL_PLATFORM_ANGLE_D3D_LUID_LOW_ANGLE,
                *low as i32,
            ]),
            // Resolved to Luid/Hardware by `resolve_adapter` before we get here.
            AdapterSelection::Discrete | AdapterSelection::Match(_) => attribs.extend([
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE,
                EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE,
            ]),
        }
        attribs.push(egl::NONE);

        // SAFETY: well-formed attrib list passed to a documented EGL
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
                "eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, D3D11) returned NO_DISPLAY",
            )));
        }
        // SAFETY: `raw` is a valid EGLDisplay per the spec.
        Ok(unsafe { egl::Display::from_ptr(raw) })
    }
}

// ---------------------------------------------------------------------------
// Adapter selection (EDGEFIRST_ANGLE_ADAPTER) + DXGI enumeration.
// ---------------------------------------------------------------------------

/// Which D3D11 adapter ANGLE should create its device on.
///
/// Parsed from `EDGEFIRST_ANGLE_ADAPTER`:
///
/// | Value | Meaning |
/// |---|---|
/// | unset / `hardware` | ANGLE's default hardware adapter (DXGI adapter 0) |
/// | `warp` | Microsoft Basic Render Driver (software; classified as a software renderer, so it also needs `EDGEFIRST_ALLOW_SOFTWARE_GL=1`) |
/// | `discrete` | the non-software adapter with the most dedicated video memory |
/// | `<high>:<low>` | an explicit adapter LUID (decimal or `0x` hex) |
/// | anything else | case-insensitive substring of the adapter description (e.g. `RTX 3070`) |
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AdapterSelection {
    Hardware,
    Warp,
    Discrete,
    Luid { high: i32, low: u32 },
    Match(String),
}

fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else {
        s.parse().ok()
    }
}

/// Parse an `EDGEFIRST_ANGLE_ADAPTER` value. Pure.
pub(crate) fn parse_adapter_selection(raw: Option<&str>) -> AdapterSelection {
    let raw = raw.map(str::trim).unwrap_or("");
    match raw.to_ascii_lowercase().as_str() {
        "" | "hardware" | "hw" | "default" => AdapterSelection::Hardware,
        "warp" | "software" => AdapterSelection::Warp,
        "discrete" | "dgpu" => AdapterSelection::Discrete,
        _ => {
            if let Some((h, l)) = raw.split_once(':') {
                if let (Some(high), Some(low)) = (parse_int(h), parse_int(l)) {
                    if let (Ok(high), Ok(low)) = (i32::try_from(high), u32::try_from(low)) {
                        return AdapterSelection::Luid { high, low };
                    }
                }
            }
            AdapterSelection::Match(raw.to_string())
        }
    }
}

/// One DXGI adapter as enumerated by `IDXGIFactory1::EnumAdapters1`.
#[derive(Debug, Clone)]
pub(crate) struct DxgiAdapter {
    pub(crate) index: u32,
    pub(crate) description: String,
    pub(crate) luid_high: i32,
    pub(crate) luid_low: u32,
    pub(crate) software: bool,
    pub(crate) dedicated_video_memory: u64,
}

// Minimal hand-written COM surface: the DXGI vtables are ABI-stable and we
// need three slots, which is not worth a `windows-sys` dependency.
type ComPtr = *mut c_void;
type HResult = i32;
#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}
/// `IID_IDXGIFactory1` = {770aae78-f26f-4dba-a829-253c83d1b387}.
const IID_IDXGI_FACTORY1: Guid = Guid {
    data1: 0x770a_ae78,
    data2: 0xf26f,
    data3: 0x4dba,
    data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
};
#[repr(C)]
struct Luid {
    low: u32,
    high: i32,
}
/// `DXGI_ADAPTER_DESC1`.
#[repr(C)]
struct DxgiAdapterDesc1 {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    subsys_id: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    adapter_luid: Luid,
    flags: u32,
}
const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;
const DXGI_ERROR_NOT_FOUND: HResult = 0x887A_0002_u32 as i32;
/// Vtable slots: IUnknown (0-2), IDXGIObject (3-6), IDXGIFactory (7-11),
/// IDXGIFactory1 (12-13); IDXGIAdapter (7-9), IDXGIAdapter1 (10).
const SLOT_RELEASE: usize = 2;
const SLOT_FACTORY1_ENUM_ADAPTERS1: usize = 12;
const SLOT_ADAPTER1_GET_DESC1: usize = 10;
type FnCreateDxgiFactory1 =
    unsafe extern "system" fn(riid: *const Guid, out: *mut ComPtr) -> HResult;
type FnRelease = unsafe extern "system" fn(this: ComPtr) -> u32;
type FnEnumAdapters1 =
    unsafe extern "system" fn(this: ComPtr, index: u32, out: *mut ComPtr) -> HResult;
type FnGetDesc1 = unsafe extern "system" fn(this: ComPtr, desc: *mut DxgiAdapterDesc1) -> HResult;

/// Read vtable slot `index` of a live COM object.
///
/// # Safety
/// `this` must be a live COM object pointer whose vtable has at least
/// `index + 1` entries.
unsafe fn com_slot(this: ComPtr, index: usize) -> *const c_void {
    // SAFETY: per the contract, `this` points at an object whose first
    // field is the vtable pointer, an array of function pointers.
    unsafe {
        let vtbl = *(this as *const *const *const c_void);
        *vtbl.add(index)
    }
}

/// Enumerate the DXGI adapters (hardware and software) on this host.
/// Errors when `dxgi.dll` or the factory is unavailable (Server Core, a
/// sandbox); callers degrade to ANGLE's default adapter.
pub(crate) fn enumerate_dxgi_adapters() -> Result<Vec<DxgiAdapter>> {
    let io = |m: String| Error::Io(std::io::Error::other(m));
    // SAFETY: dxgi.dll is a Windows system library with no initializer
    // side effects of concern.
    let dxgi = unsafe { libloading::Library::new("dxgi.dll") }
        .map_err(|e| io(format!("dxgi.dll: {e}")))?;
    // SAFETY: the symbol has the documented `CreateDXGIFactory1` signature.
    let create: libloading::Symbol<FnCreateDxgiFactory1> =
        unsafe { dxgi.get(b"CreateDXGIFactory1\0") }
            .map_err(|e| io(format!("CreateDXGIFactory1: {e}")))?;
    let mut factory: ComPtr = std::ptr::null_mut();
    // SAFETY: valid IID and out-pointer.
    let hr = unsafe { create(&IID_IDXGI_FACTORY1, &mut factory) };
    if hr < 0 || factory.is_null() {
        return Err(io(format!("CreateDXGIFactory1 failed: HRESULT {hr:#010x}")));
    }
    // SAFETY: `factory` is a live IDXGIFactory1; slot indices per the
    // interface layout documented on the constants.
    let (factory_release, enum_adapters1): (FnRelease, FnEnumAdapters1) = unsafe {
        (
            std::mem::transmute::<*const c_void, FnRelease>(com_slot(factory, SLOT_RELEASE)),
            std::mem::transmute::<*const c_void, FnEnumAdapters1>(com_slot(
                factory,
                SLOT_FACTORY1_ENUM_ADAPTERS1,
            )),
        )
    };
    let mut adapters = Vec::new();
    let mut index = 0u32;
    loop {
        let mut adapter: ComPtr = std::ptr::null_mut();
        // SAFETY: live factory, valid out-pointer.
        let hr = unsafe { enum_adapters1(factory, index, &mut adapter) };
        if hr == DXGI_ERROR_NOT_FOUND || hr < 0 || adapter.is_null() {
            break;
        }
        // SAFETY: `adapter` is a live IDXGIAdapter1.
        let (adapter_release, get_desc1): (FnRelease, FnGetDesc1) = unsafe {
            (
                std::mem::transmute::<*const c_void, FnRelease>(com_slot(adapter, SLOT_RELEASE)),
                std::mem::transmute::<*const c_void, FnGetDesc1>(com_slot(
                    adapter,
                    SLOT_ADAPTER1_GET_DESC1,
                )),
            )
        };
        let mut desc = std::mem::MaybeUninit::<DxgiAdapterDesc1>::zeroed();
        // SAFETY: live adapter, valid out-pointer to a correctly laid-out struct.
        let hr = unsafe { get_desc1(adapter, desc.as_mut_ptr()) };
        if hr >= 0 {
            // SAFETY: GetDesc1 succeeded and filled the struct.
            let desc = unsafe { desc.assume_init() };
            let end = desc
                .description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.description.len());
            adapters.push(DxgiAdapter {
                index,
                description: String::from_utf16_lossy(&desc.description[..end]),
                luid_high: desc.adapter_luid.high,
                luid_low: desc.adapter_luid.low,
                software: desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0,
                dedicated_video_memory: desc.dedicated_video_memory as u64,
            });
        }
        // SAFETY: balances the reference EnumAdapters1 handed out.
        unsafe { adapter_release(adapter) };
        index += 1;
    }
    // SAFETY: balances CreateDXGIFactory1's reference.
    unsafe { factory_release(factory) };
    Ok(adapters)
}

/// Turn `Discrete` / `Match` into a concrete LUID (or `Hardware` when
/// nothing matches, with a warning). Pure over the adapter list.
pub(crate) fn resolve_adapter(sel: AdapterSelection, adapters: &[DxgiAdapter]) -> AdapterSelection {
    let luid = |a: &DxgiAdapter| AdapterSelection::Luid {
        high: a.luid_high,
        low: a.luid_low,
    };
    match sel {
        AdapterSelection::Discrete => match adapters
            .iter()
            .filter(|a| !a.software)
            .max_by_key(|a| a.dedicated_video_memory)
        {
            Some(a) => luid(a),
            None => {
                warn!("{ADAPTER_ENV}=discrete: no hardware adapter enumerated; using ANGLE's default adapter");
                AdapterSelection::Hardware
            }
        },
        AdapterSelection::Match(needle) => {
            let n = needle.to_ascii_lowercase();
            match adapters
                .iter()
                .find(|a| a.description.to_ascii_lowercase().contains(&n))
            {
                Some(a) => luid(a),
                None => {
                    warn!(
                        "{ADAPTER_ENV}={needle:?} matches no DXGI adapter description \
                         ({:?}); using ANGLE's default adapter",
                        adapters
                            .iter()
                            .map(|a| a.description.as_str())
                            .collect::<Vec<_>>()
                    );
                    AdapterSelection::Hardware
                }
            }
        }
        other => other,
    }
}

/// Read `EDGEFIRST_ANGLE_ADAPTER`, enumerate DXGI, log the adapters, and
/// return the selection to hand to [`WindowsPlatform::create_display`]
/// plus a human-readable name of the chosen adapter.
fn select_adapter() -> (AdapterSelection, String) {
    let raw = std::env::var(ADAPTER_ENV).ok();
    let sel = parse_adapter_selection(raw.as_deref());
    let adapters = match enumerate_dxgi_adapters() {
        Ok(a) => a,
        Err(e) => {
            debug!("DXGI adapter enumeration unavailable ({e}); using ANGLE's default adapter");
            Vec::new()
        }
    };
    for a in &adapters {
        debug!(
            "dxgi adapter #{}: {:?} luid={:#x}:{:#x} vram={} MiB software={}",
            a.index,
            a.description,
            a.luid_high,
            a.luid_low,
            a.dedicated_video_memory >> 20,
            a.software
        );
    }
    let resolved = resolve_adapter(sel, &adapters);
    let name = match &resolved {
        AdapterSelection::Warp => "WARP (Microsoft Basic Render Driver)".to_string(),
        AdapterSelection::Luid { high, low } => adapters
            .iter()
            .find(|a| a.luid_high == *high && a.luid_low == *low)
            .map(|a| a.description.clone())
            .unwrap_or_else(|| format!("LUID {high:#x}:{low:#x}")),
        _ => adapters
            .iter()
            .find(|a| !a.software)
            .map(|a| format!("{} (ANGLE default)", a.description))
            .unwrap_or_else(|| "ANGLE default adapter".to_string()),
    };
    if matches!(resolved, AdapterSelection::Hardware)
        && !adapters.is_empty()
        && adapters.iter().all(|a| a.software)
    {
        warn!(
            "no hardware D3D11 adapter is enumerated (only {:?}); ANGLE will run on \
             the software renderer, which the GL backend rejects unless \
             EDGEFIRST_ALLOW_SOFTWARE_GL=1",
            adapters
                .iter()
                .map(|a| a.description.as_str())
                .collect::<Vec<_>>()
        );
    }
    info!(
        "ANGLE D3D11 adapter: {name} ({ADAPTER_ENV}={})",
        raw.as_deref().unwrap_or("unset")
    );
    (resolved, name)
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
    /// so the display's D3D device is never idle-torn-down.
    probe_context: egl::Context,
    probe_pbuffer: egl::Surface,
    /// The probe context came up as GLES 3.1 (compute shaders available).
    pub(in crate::opengl_headless) has_compute: bool,
    /// `GL_EXT_color_buffer_float` — gates F32 PBO destinations.
    pub(in crate::opengl_headless) supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` — gates F16 PBO destinations.
    pub(in crate::opengl_headless) supports_f16_color: bool,
    /// Human-readable name of the adapter ANGLE was pointed at.
    pub(in crate::opengl_headless) adapter: String,
}

// SAFETY: every member is either a leaked static, an EGL handle (ANGLE
// synchronizes display-level entry points internally), or plain data.
// The probe context is never made current after init.
unsafe impl Send for SharedD3d11Display {}
unsafe impl Sync for SharedD3d11Display {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedD3d11Display, String>> = OnceLock::new();

/// Acquire the process-global ANGLE D3D11 display, initialising it on the
/// first call. The error case is cached too — once ANGLE fails to load we
/// don't keep retrying (and re-warning) per processor.
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
        tracing::info_span!("image.gl_init", platform = "windows", backend = "pbo").entered();

    // 1. Adapter selection (env + DXGI enumeration), then ANGLE libEGL.
    let (selection, adapter) = select_adapter();
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

    // 2. D3D11 display on the selected adapter.
    let display = WindowsPlatform::create_display(&egl, &selection)?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    let egl_version = egl
        .query_string(Some(display), egl::VERSION)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    debug!("ANGLE EGL {maj}.{min} initialised (process-global shared display): {egl_version}");

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. GLES3 + pbuffer config. No EGL_BIND_TO_TEXTURE_TARGET_ANGLE here —
    //    that attribute belongs to the IOSurface extension and D3D11
    //    rejects it. The D3D11 client-buffer follow-on will add
    //    EGL_BIND_TO_TEXTURE_RGBA when pbuffer imports arrive.
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

    // 4. Probe context (3.1 preferred) + a tiny pbuffer so it can be current.
    let (probe_context, has_compute) = create_es_context(&egl, display, config, true)?;
    let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
    let probe_pbuffer = egl
        .create_pbuffer_surface(display, config, &dummy_attribs)
        .map_err(|e| {
            let _ = egl.destroy_context(display, probe_context);
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
        let _ = egl.destroy_context(display, probe_context);
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
        "ANGLE D3D11 display ready: {gl_renderer} — {gl_version} \
         (compute={has_compute}, f32_color={supports_f32_color}, f16_color={supports_f16_color})"
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
        adapter,
    })
}

// ---------------------------------------------------------------------------
// Per-processor display: a private context on the shared D3D11 display.
// ---------------------------------------------------------------------------

/// One processor's GL bring-up state: a PRIVATE EGL context (plus dummy
/// pbuffer) on the process-global shared ANGLE display. Created on the
/// processor's worker thread, made current there once, and held for the
/// thread's life. NOT `Send`: dropped on the creating thread (the dispatch
/// wrapper guarantees both).
pub(in crate::opengl_headless) struct D3d11Display {
    pub(in crate::opengl_headless) shared: &'static SharedD3d11Display,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Duck-typed counterparts of the `GlContext` members the portable
    /// engine reads. PBO is the only transfer backend on Windows.
    pub(in crate::opengl_headless) transfer_backend: TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
    /// Texture attachments made via `eglBindTexImage` since the last
    /// `GlPlatform::end_gpu_pass` (released there, after the engine's sync
    /// point). Unused until the D3D11 client-buffer follow-on; kept so the
    /// leaf already has the pbuffer-binding shape.
    active_binds: RefCell<Vec<egl::Surface>>,
}

impl Drop for D3d11Display {
    fn drop(&mut self) {
        // Runs on the owning worker thread: release, then destroy. Forget
        // this context as the last active one so a successor allocated at
        // the same address is not mistaken for it.
        let d = self.shared;
        let ctx = self.context.as_ptr();
        let _ = LAST_ACTIVE_CONTEXT.compare_exchange(
            ctx,
            std::ptr::null_mut(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        let _ = d.egl.destroy_context(d.display, self.context);
    }
}

/// An owned D3D11-texture→EGL-pbuffer import. Dropping destroys the
/// pbuffer. Never constructed today (no zero-copy tensor exists on
/// Windows); the D3D11 shared-texture follow-on creates these via
/// `eglCreatePbufferFromClientBuffer(EGL_D3D_TEXTURE_ANGLE, ...)`.
pub(in crate::opengl_headless) struct D3dTexturePbuffer {
    shared: &'static SharedD3d11Display,
    pub(in crate::opengl_headless) surface: egl::Surface,
}

impl Drop for D3dTexturePbuffer {
    fn drop(&mut self) {
        let _ = self
            .shared
            .egl
            .destroy_surface(self.shared.display, self.surface);
    }
}

/// Marker type: Windows ANGLE + Direct3D 11 platform.
pub(crate) struct AngleD3d11;

impl GlPlatform for AngleD3d11 {
    type Display = D3d11Display;
    type Import = D3dTexturePbuffer;
    type ImportHandle = egl::Surface;

    // eglBindTexImage attachments are released at end_gpu_pass — the
    // engine's binding-skip cache must stay cold, as on macOS.
    const PERSISTENT_TEX_BINDINGS: bool = false;
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
                let _ = shared.egl.destroy_context(shared.display, context);
                Error::Io(std::io::Error::other(format!(
                    "eglCreatePbufferSurface (per-processor dummy): {e:?}"
                )))
            })?;
        // Made current ONCE on the calling (worker) thread and held for the
        // thread's life.
        if let Err(e) = shared.egl.make_current(
            shared.display,
            Some(dummy_pbuffer),
            Some(dummy_pbuffer),
            Some(context),
        ) {
            let _ = shared.egl.destroy_surface(shared.display, dummy_pbuffer);
            let _ = shared.egl.destroy_context(shared.display, context);
            return Err(Error::Io(std::io::Error::other(format!(
                "eglMakeCurrent (per-processor): {e:?}"
            ))));
        }
        debug!(
            "Windows GL context up on {} (GLES {}, transfer=Pbo)",
            shared.adapter,
            if has_compute { "3.1" } else { "3.0" }
        );
        Ok(D3d11Display {
            shared,
            context,
            dummy_pbuffer,
            transfer_backend: TransferBackend::Pbo,
            has_compute,
            active_binds: RefCell::new(Vec::new()),
        })
    }

    fn import_buffer(
        _display: &D3d11Display,
        _img: &Tensor<u8>,
        _fmt: PixelFormat,
        _for_dst: bool,
    ) -> Result<D3dTexturePbuffer> {
        Err(Error::NotSupported(
            "Windows GL backend has no zero-copy buffer import yet (D3D11 shared-texture \
             import is a follow-on); sources are Mem tensors and destinations PBO tensors"
                .into(),
        ))
    }

    fn import_buffer_nv_r8(
        _display: &D3d11Display,
        _img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<D3dTexturePbuffer> {
        Err(Error::NotSupported(
            "Windows GL backend has no zero-copy NV import yet (D3D11 shared-texture import is a follow-on)".into(),
        ))
    }

    fn import_buffer_packed<T>(
        _display: &D3d11Display,
        _img: &Tensor<T>,
        _width: usize,
        _height: usize,
        _fmt: super::PackedImportFormat,
    ) -> Result<D3dTexturePbuffer>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        Err(Error::NotSupported(
            "Windows GL backend has no zero-copy packed import yet (D3D11 shared-texture import is a follow-on)".into(),
        ))
    }

    fn import_handle(import: &D3dTexturePbuffer) -> egl::Surface {
        import.surface
    }

    unsafe fn attach_tex_image_2d(display: &D3d11Display, handle: egl::Surface) -> Result<()> {
        display
            .shared
            .egl
            .bind_tex_image(display.shared.display, handle, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))?;
        display.active_binds.borrow_mut().push(handle);
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &D3d11Display,
        _handle: egl::Surface,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES is not available on ANGLE/D3D11".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &D3d11Display,
        _handle: egl::Surface,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "renderbuffer import targets are not available on ANGLE/D3D11 \
             (EDGEFIRST_OPENGL_RENDERSURFACE has no effect on Windows)"
                .into(),
        ))
    }

    fn begin_gpu_pass(display: &D3d11Display) {
        // ANGLE's D3D11 backend keeps ONE `StateManager11` per display and
        // re-syncs a context's GL state onto the shared D3D device only from
        // `eglMakeCurrent` (`Context11::onMakeCurrent`). With every
        // processor's context permanently current on its own thread,
        // alternating processors — even fully serialized by the dispatch
        // wrapper — rendered with the PREVIOUS context's applied state
        // (viewport, bindings, uniforms): the parallel-processor tests
        // diverged on ~55 % of their bytes. So whenever a different context
        // issued the last commands, release and re-make this one current on
        // this thread, which makes ANGLE mark everything dirty. Free when
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
        }
    }

    fn end_gpu_pass(display: &D3d11Display) {
        for surface in display.active_binds.borrow_mut().drain(..) {
            let _ = display.shared.egl.release_tex_image(
                display.shared.display,
                surface,
                EGL_BACK_BUFFER,
            );
        }
    }

    fn native_fence_sync(_display: &D3d11Display) -> bool {
        // No EGL_ANDROID_native_fence_sync on ANGLE/D3D11; consumers rely on
        // convert() returning ⇒ GPU done. A D3D11 fence handle is the
        // follow-on's job (the `CompletionFence` alias is an OwnedHandle here).
        false
    }

    fn export_completion_fence(_display: &D3d11Display) -> Result<Option<CompletionFence>> {
        Ok(None)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn adapter_selection_parses_env_values() {
        assert_eq!(parse_adapter_selection(None), AdapterSelection::Hardware);
        assert_eq!(
            parse_adapter_selection(Some("")),
            AdapterSelection::Hardware
        );
        assert_eq!(
            parse_adapter_selection(Some(" Hardware ")),
            AdapterSelection::Hardware
        );
        assert_eq!(
            parse_adapter_selection(Some("WARP")),
            AdapterSelection::Warp
        );
        assert_eq!(
            parse_adapter_selection(Some("discrete")),
            AdapterSelection::Discrete
        );
        assert_eq!(
            parse_adapter_selection(Some("0x1234:0xabcd")),
            AdapterSelection::Luid {
                high: 0x1234,
                low: 0xabcd
            }
        );
        assert_eq!(
            parse_adapter_selection(Some("0:74901")),
            AdapterSelection::Luid {
                high: 0,
                low: 74901
            }
        );
        assert_eq!(
            parse_adapter_selection(Some("RTX 3070")),
            AdapterSelection::Match("RTX 3070".into())
        );
        // A colon that is not a LUID stays a substring match.
        assert_eq!(
            parse_adapter_selection(Some("Intel: Arc")),
            AdapterSelection::Match("Intel: Arc".into())
        );
    }

    #[test]
    fn resolve_adapter_prefers_largest_hardware_adapter_and_matches_substrings() {
        let adapters = vec![
            DxgiAdapter {
                index: 0,
                description: "Intel(R) UHD Graphics 630".into(),
                luid_high: 0,
                luid_low: 1,
                software: false,
                dedicated_video_memory: 128 << 20,
            },
            DxgiAdapter {
                index: 1,
                description: "NVIDIA GeForce RTX 3070".into(),
                luid_high: 0,
                luid_low: 2,
                software: false,
                dedicated_video_memory: 8 << 30,
            },
            DxgiAdapter {
                index: 2,
                description: "Microsoft Basic Render Driver".into(),
                luid_high: 0,
                luid_low: 3,
                software: true,
                dedicated_video_memory: 0,
            },
        ];
        assert_eq!(
            resolve_adapter(AdapterSelection::Discrete, &adapters),
            AdapterSelection::Luid { high: 0, low: 2 }
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Match("intel".into()), &adapters),
            AdapterSelection::Luid { high: 0, low: 1 }
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Match("no such gpu".into()), &adapters),
            AdapterSelection::Hardware
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Warp, &adapters),
            AdapterSelection::Warp
        );
        assert_eq!(
            resolve_adapter(AdapterSelection::Discrete, &adapters[2..]),
            AdapterSelection::Hardware
        );
    }

    /// DXGI enumeration is a system facility; on a host where dxgi.dll is
    /// missing (Server Core) the function errors and the caller degrades,
    /// so this only asserts the happy path when it applies.
    #[test]
    fn dxgi_enumeration_lists_adapters_or_skips() {
        match enumerate_dxgi_adapters() {
            Ok(adapters) => {
                assert!(!adapters.is_empty(), "DXGI enumerated zero adapters");
                for a in &adapters {
                    assert!(!a.description.is_empty());
                }
            }
            Err(e) => eprintln!("DXGI enumeration unavailable — skipping: {e}"),
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

    /// Per-processor context bring-up latency (budget <50 ms each, as on
    /// macOS). Ignored: needs ANGLE + a D3D11 device; run on demand:
    /// `cargo test -p edgefirst-image --release --lib windows::tests -- --ignored --nocapture`
    #[test]
    #[ignore = "needs ANGLE + a D3D11 device; run on demand"]
    fn per_processor_context_bring_up_latency() {
        use std::time::Instant;
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
