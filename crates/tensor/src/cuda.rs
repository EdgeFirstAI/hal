// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional CUDA Runtime interop, loaded via dlopen (no link-time dependency).
//! Absent libcudart ⇒ every CUDA path degrades to `None`.
//!
//! The intended client pattern is a fast-fail + host fallback: call
//! [`Tensor::cuda_map`](crate::Tensor::cuda_map) (or [`TensorDyn::cuda_map`](crate::TensorDyn::cuda_map))
//! first; if it returns `None` (no CUDA handle attached or libcudart absent),
//! fall back to the host mapping via `Tensor::map`. This keeps hot paths
//! zero-copy on CUDA-capable hardware while remaining correct on CPU-only targets.
use libloading::Library;
#[cfg(target_os = "windows")]
use std::ffi::c_char;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
#[cfg(all(target_os = "windows", feature = "static"))]
use std::os::windows::io::{AsRawHandle, OwnedHandle, RawHandle};
use std::sync::{Arc, OnceLock};
#[cfg(target_os = "windows")]
use windows::core::PCWSTR;
#[cfg(target_os = "windows")]
use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};

pub(crate) type CudaError = c_int; // cudaSuccess == 0
pub type GraphicsResource = *mut c_void; // cudaGraphicsResource_t
pub(crate) type ExternalMemory = *mut c_void; // cudaExternalMemory_t
pub type CudaStream = *mut c_void; // cudaStream_t
#[cfg(target_os = "windows")]
pub(crate) type MipmappedArray = *mut c_void; // cudaMipmappedArray_t
#[cfg(target_os = "windows")]
pub(crate) type CudaArray = *mut c_void; // cudaArray_t
#[cfg(target_os = "windows")]
pub(crate) type ExternalSemaphore = *mut c_void; // cudaExternalSemaphore_t

#[allow(non_snake_case, dead_code)]
pub(crate) struct CudaTable {
    // Owns the loaded module so it stays mapped for as long as the function
    // pointers below are callable. `CudaTable` only ever lives inside the
    // `'static TABLE` OnceLock (never dropped for the life of the process),
    // so this need not be leaked separately.
    _lib: Library,
    /// The runtime `load()` actually opened: a `CUDA_PATH\bin\...` candidate
    /// is recorded verbatim (loading a full path is unambiguous); a bare
    /// name (Windows) is resolved to the absolute path Windows mapped via
    /// `GetModuleHandleW`/`GetModuleFileNameW`, falling back to the bare
    /// name if that resolution fails. On Linux this is the soname that
    /// opened. Several toolkits on a dev box can share a bare DLL/soname, so
    /// this is what makes "which runtime is CUDA actually using" answerable.
    pub runtime_path: std::path::PathBuf,
    pub graphics_gl_register_buffer:
        unsafe extern "C" fn(*mut GraphicsResource, c_uint, c_uint) -> CudaError,
    pub graphics_map_resources:
        unsafe extern "C" fn(c_int, *mut GraphicsResource, *mut c_void) -> CudaError,
    pub graphics_get_mapped_pointer:
        unsafe extern "C" fn(*mut *mut c_void, *mut usize, GraphicsResource) -> CudaError,
    pub graphics_unmap_resources:
        unsafe extern "C" fn(c_int, *mut GraphicsResource, *mut c_void) -> CudaError,
    pub graphics_unregister_resource: unsafe extern "C" fn(GraphicsResource) -> CudaError,
    pub import_external_memory:
        unsafe extern "C" fn(*mut ExternalMemory, *const c_void) -> CudaError,
    pub external_memory_get_mapped_buffer:
        unsafe extern "C" fn(*mut *mut c_void, ExternalMemory, *const c_void) -> CudaError,
    pub destroy_external_memory: unsafe extern "C" fn(ExternalMemory) -> CudaError,
    pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> CudaError,
    pub free: unsafe extern "C" fn(*mut c_void) -> CudaError,
    pub stream_create: unsafe extern "C" fn(*mut CudaStream) -> CudaError,
    pub stream_synchronize: unsafe extern "C" fn(CudaStream) -> CudaError,
    pub stream_destroy: unsafe extern "C" fn(CudaStream) -> CudaError,
    // D3D11 texture interop (Task C2 consumes these; only meaningful where D3D11
    // exists, so Windows-only both here and in `load_table`).
    #[cfg(target_os = "windows")]
    pub d3d11_get_device: unsafe extern "C" fn(*mut c_int, *mut c_void) -> CudaError,
    #[cfg(target_os = "windows")]
    pub set_device: unsafe extern "C" fn(c_int) -> CudaError,
    #[cfg(target_os = "windows")]
    pub external_memory_get_mapped_mipmapped_array:
        unsafe extern "C" fn(*mut MipmappedArray, ExternalMemory, *const c_void) -> CudaError,
    #[cfg(target_os = "windows")]
    pub get_mipmapped_array_level:
        unsafe extern "C" fn(*mut CudaArray, MipmappedArray, c_uint) -> CudaError,
    #[cfg(target_os = "windows")]
    pub free_mipmapped_array: unsafe extern "C" fn(MipmappedArray) -> CudaError,
    // The async spellings, not the plain ones: a device-to-device
    // `cudaMemcpy2D*` runs on the legacy default stream, and synchronising
    // that is a device-wide barrier that also waits behind every other stream
    // this crate hands out (the codec's, for one). These take the handle's own
    // stream, so the wait, the copy and the synchronisation are one sequence
    // on one stream.
    #[cfg(target_os = "windows")]
    pub memcpy_2d_from_array_async: unsafe extern "C" fn(
        *mut c_void,
        usize,
        CudaArray,
        usize,
        usize,
        usize,
        usize,
        c_int,
        CudaStream,
    ) -> CudaError,
    #[cfg(target_os = "windows")]
    pub memcpy_2d_to_array_async: unsafe extern "C" fn(
        CudaArray,
        usize,
        usize,
        *const c_void,
        usize,
        usize,
        usize,
        c_int,
        CudaStream,
    ) -> CudaError,
    #[cfg(target_os = "windows")]
    pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> CudaError,
    #[cfg(target_os = "windows")]
    pub import_external_semaphore:
        unsafe extern "C" fn(*mut ExternalSemaphore, *const c_void) -> CudaError,
    #[cfg(target_os = "windows")]
    pub wait_external_semaphores_async: unsafe extern "C" fn(
        *const ExternalSemaphore,
        *const c_void,
        c_uint,
        CudaStream,
    ) -> CudaError,
    #[cfg(target_os = "windows")]
    pub destroy_external_semaphore: unsafe extern "C" fn(ExternalSemaphore) -> CudaError,
    #[cfg(target_os = "windows")]
    pub get_error_string: unsafe extern "C" fn(CudaError) -> *const c_char,
}

static TABLE: OnceLock<Option<CudaTable>> = OnceLock::new();

// Miri cannot execute `dlopen` -- it is a foreign function Miri's
// interpreter refuses outright ("can't call foreign function `dlopen`"),
// not merely a syscall it emulates differently. `Tensor::new` calls
// `try_init_dma_cuda` unconditionally on Linux (`lib.rs`), which calls
// `is_cuda_available` -> `table` -> `load` before it even checks whether
// the tensor is DMA-backed, so this fires on every tensor construction
// under Miri, DMA or not. Reporting "unavailable" here is not suppressing
// a finding: this module's own contract is "absent libcudart => every CUDA
// path degrades to `None`" (see the module doc), and under Miri libcudart
// is, honestly, never going to be found.
#[cfg(miri)]
fn load() -> Option<CudaTable> {
    None
}

#[cfg(all(not(miri), not(target_os = "windows")))]
fn load() -> Option<CudaTable> {
    for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"] {
        // SAFETY: `name` is one of the fixed soname literals above. Loading
        // a shared object runs its init routines, an unavoidable side effect
        // of dynamic loading that this whole module's design accepts (see
        // the module doc: "absent libcudart => every CUDA path degrades to
        // `None`" is the only contract offered).
        let Ok(lib) = (unsafe { Library::new(name) }) else {
            continue;
        };
        // `load_table` takes ownership of `lib`; if a required symbol is
        // missing it returns `None` and its local `lib` parameter drops
        // (unloading this soname), so the next candidate is tried rather
        // than disabling CUDA outright for a stale library earlier in the
        // search order.
        if let Some(table) = load_table(lib, std::path::PathBuf::from(name)) {
            log::debug!("CUDA runtime loaded from {}", table.runtime_path.display());
            return Some(table);
        }
    }
    None
}

#[cfg(all(not(miri), target_os = "windows"))]
fn load() -> Option<CudaTable> {
    // Newest runtime first; consumers ship their own, the HAL only needs one
    // that resolves the interop entry points. `cudart64_110.dll` is not on
    // PATH/System32 by default, so also try CUDA_PATH\bin (where the
    // installer puts it) for each candidate name. `is_full_path` marks the
    // `CUDA_PATH\bin` entries: loading a full path is unambiguous, so their
    // `runtime_path` is the candidate itself; a bare name resolves through
    // the OS's DLL search order and needs `resolve_loaded_module_path` to
    // learn which physical file actually opened.
    let names = ["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"];
    let mut candidates: Vec<(std::path::PathBuf, bool)> = names
        .iter()
        .map(|n| (std::path::PathBuf::from(n), false))
        .collect();
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        for n in names {
            let full = std::path::Path::new(&cuda_path).join("bin").join(n);
            candidates.push((full, true));
        }
    }
    for (candidate, is_full_path) in &candidates {
        // SAFETY: `candidate` is a `.dll` filename or full path built above.
        // Loading a module runs its init routines, an unavoidable side
        // effect of dynamic loading that this whole module's design accepts
        // (see the module doc: "absent libcudart => every CUDA path
        // degrades to `None`" is the only contract offered).
        let Ok(lib) = (unsafe { Library::new(candidate) }) else {
            continue;
        };
        let runtime_path = if *is_full_path {
            candidate.clone()
        } else {
            resolve_loaded_module_path(candidate.as_os_str()).unwrap_or_else(|| candidate.clone())
        };
        // `load_table` takes ownership of `lib`; if a required symbol is
        // missing it returns `None` and its local `lib` parameter drops
        // (unloading this candidate), so the next one is tried rather than
        // disabling CUDA outright for a stale toolkit earlier on PATH.
        if let Some(table) = load_table(lib, runtime_path) {
            log::debug!("CUDA runtime loaded from {}", table.runtime_path.display());
            return Some(table);
        }
    }
    None
}

/// Resolves a bare DLL name (no directory) to the absolute path Windows
/// actually mapped, via `GetModuleHandleW` + `GetModuleFileNameW`. Callers
/// only use this immediately after `Library::new` already succeeded for the
/// same name, so the module is already loaded: `GetModuleHandleW` looks up
/// its existing handle, it does not load a second copy or re-run init
/// routines. Returns `None` on any resolution failure; the caller falls back
/// to the bare name it tried.
#[cfg(target_os = "windows")]
fn resolve_loaded_module_path(name: &std::ffi::OsStr) -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let wide: Vec<u16> = name.encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 buffer kept alive for the
    // call. `GetModuleHandleW` only resolves an already-loaded module's
    // handle by name -- it neither loads nor initializes anything.
    let handle = unsafe { GetModuleHandleW(PCWSTR(wide.as_ptr())) }.ok()?;
    let mut buf = [0u16; 1024];
    // SAFETY: `handle` was just returned by a successful `GetModuleHandleW`
    // for a module already loaded in this process; `buf` is a correctly
    // sized mutable out-buffer for `GetModuleFileNameW` to write into.
    let len = unsafe { GetModuleFileNameW(Some(handle), &mut buf) } as usize;
    // `len == buf.len()` is truncation: the path did not fit and Windows did
    // not terminate it, so the prefix would name a different file.
    (len > 0 && len < buf.len())
        .then(|| std::path::PathBuf::from(std::ffi::OsString::from_wide(&buf[..len])))
}

/// Resolves every interop symbol shared by both platforms' `load()`. Returns
/// `None` and logs the missing symbol's name if any required entry point is
/// absent from `lib` -- including the D3D11 entries on Windows, where they
/// are not optional. Takes ownership of `lib`: on success it is stored in
/// the returned `CudaTable`; on failure it is simply dropped here (unloading
/// the module), so the caller is free to try its next candidate.
#[cfg(not(miri))]
fn load_table(lib: Library, runtime_path: std::path::PathBuf) -> Option<CudaTable> {
    macro_rules! sym {
        ($n:literal) => {{
            // SAFETY: `$n` is one of the fixed `cuda*` symbol-name literals
            // below; `lib` is the module `load()` just opened, and the
            // resulting function pointer is only ever called through the
            // `unsafe extern "C" fn` field type it is stored into, at each
            // call site's own risk.
            match unsafe { lib.get(concat!($n, "\0").as_bytes()) } {
                Ok(s) => *s,
                Err(_) => {
                    log::debug!("CUDA runtime missing required symbol {}", $n);
                    return None;
                }
            }
        }};
    }
    // `cudaWaitExternalSemaphoresAsync` gained a `_v2` export when the ABI of
    // `cudaExternalSemaphoreWaitParams` changed (the `reserved` tail this
    // module's `CudaExternalSemaphoreWaitParams` already carries); prefer it
    // and fall back to the unsuffixed name for older runtimes that only have
    // the v1 ABI symbol under the original name. The unsuffixed symbol on a
    // pre-11.2 runtime expects the shorter v1 params layout
    // (`cudaExternalSemaphoreWaitParams_v1`), whose `flags` sits at offset 32
    // where v2 has `params_reserved[0]`. A v1 runtime therefore reads a
    // different zeroed word as `flags` rather than omitting a field, and both
    // words are zero here: `fence.value` is at offset 0 in either layout and
    // is the only field this module sets.
    #[cfg(target_os = "windows")]
    macro_rules! sym_versioned {
        ($v2:literal, $v1:literal) => {{
            // SAFETY: `$v2`/`$v1` are fixed symbol-name literals; `lib` is
            // the module `load()` just opened. Same call-site-risk contract
            // as `sym!` above.
            let found = unsafe { lib.get(concat!($v2, "\0").as_bytes()) }
                .or_else(|_| unsafe { lib.get(concat!($v1, "\0").as_bytes()) });
            match found {
                Ok(s) => *s,
                Err(_) => {
                    log::debug!("CUDA runtime missing required symbol {} (or {})", $v2, $v1);
                    return None;
                }
            }
        }};
    }
    Some(CudaTable {
        runtime_path,
        graphics_gl_register_buffer: sym!("cudaGraphicsGLRegisterBuffer"),
        graphics_map_resources: sym!("cudaGraphicsMapResources"),
        graphics_get_mapped_pointer: sym!("cudaGraphicsResourceGetMappedPointer"),
        graphics_unmap_resources: sym!("cudaGraphicsUnmapResources"),
        graphics_unregister_resource: sym!("cudaGraphicsUnregisterResource"),
        import_external_memory: sym!("cudaImportExternalMemory"),
        external_memory_get_mapped_buffer: sym!("cudaExternalMemoryGetMappedBuffer"),
        destroy_external_memory: sym!("cudaDestroyExternalMemory"),
        memcpy: sym!("cudaMemcpy"),
        free: sym!("cudaFree"),
        stream_create: sym!("cudaStreamCreate"),
        stream_synchronize: sym!("cudaStreamSynchronize"),
        stream_destroy: sym!("cudaStreamDestroy"),
        #[cfg(target_os = "windows")]
        d3d11_get_device: sym!("cudaD3D11GetDevice"),
        #[cfg(target_os = "windows")]
        set_device: sym!("cudaSetDevice"),
        #[cfg(target_os = "windows")]
        external_memory_get_mapped_mipmapped_array: sym!(
            "cudaExternalMemoryGetMappedMipmappedArray"
        ),
        #[cfg(target_os = "windows")]
        get_mipmapped_array_level: sym!("cudaGetMipmappedArrayLevel"),
        #[cfg(target_os = "windows")]
        free_mipmapped_array: sym!("cudaFreeMipmappedArray"),
        #[cfg(target_os = "windows")]
        memcpy_2d_from_array_async: sym!("cudaMemcpy2DFromArrayAsync"),
        #[cfg(target_os = "windows")]
        memcpy_2d_to_array_async: sym!("cudaMemcpy2DToArrayAsync"),
        #[cfg(target_os = "windows")]
        malloc: sym!("cudaMalloc"),
        #[cfg(target_os = "windows")]
        import_external_semaphore: sym!("cudaImportExternalSemaphore"),
        #[cfg(target_os = "windows")]
        wait_external_semaphores_async: sym_versioned!(
            "cudaWaitExternalSemaphoresAsync_v2",
            "cudaWaitExternalSemaphoresAsync"
        ),
        #[cfg(target_os = "windows")]
        destroy_external_semaphore: sym!("cudaDestroyExternalSemaphore"),
        #[cfg(target_os = "windows")]
        get_error_string: sym!("cudaGetErrorString"),
        // Moved last: every `sym!`/`sym_versioned!` call above borrows
        // `lib`, so the field that takes ownership of it must come after
        // all of them in the struct literal (fields evaluate in the order
        // written, not declaration order).
        _lib: lib,
    })
}

/// `cudaGetErrorString(rc)` with the numeric code kept alongside it, for the
/// D3D11 arm's diagnostics. Falls back to the bare code if the runtime hands
/// back no string.
#[cfg(all(target_os = "windows", feature = "static"))]
fn err_str(t: &CudaTable, rc: CudaError) -> String {
    // SAFETY: `cudaGetErrorString` returns a pointer to a NUL-terminated
    // string the runtime owns for the life of the process, or null for a code
    // it does not know.
    let p = unsafe { (t.get_error_string)(rc) };
    if p.is_null() {
        return format!("cudaError {rc}");
    }
    // SAFETY: `p` is the non-null NUL-terminated string from the call above.
    let text = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
    format!("{text} (cudaError {rc})")
}

pub(crate) fn table() -> Option<&'static CudaTable> {
    TABLE.get_or_init(load).as_ref()
}

/// True iff libcudart loaded and all interop symbols resolved. Cached, cheap.
pub fn is_cuda_available() -> bool {
    table().is_some()
}

/// The runtime `load()` opened, if any -- see `CudaTable::runtime_path` for
/// what this points to on each platform. `None` iff no runtime loaded.
pub fn runtime_path() -> Option<&'static std::path::Path> {
    table().map(|t| t.runtime_path.as_path())
}

/// `cudaMemcpyDeviceToHost` — copies from device (GPU) to host (CPU).
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// `cudaMemcpyHostToDevice` — copies from host (CPU) to device (GPU).
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;

/// Copy `count` bytes from a CUDA device pointer to host. Returns `false` on
/// failure or if libcudart is unavailable.
///
/// # Safety
///
/// The caller must ensure:
/// - `host` points to at least `count` bytes of writable memory.
/// - `device` points to at least `count` bytes of valid CUDA device memory
///   (i.e. obtained from a `CudaMap::device_ptr()` while the map is live).
pub unsafe fn memcpy_device_to_host(
    host: *mut c_void,
    device: *const c_void,
    count: usize,
) -> bool {
    match table() {
        Some(t) => unsafe { (t.memcpy)(host, device, count, CUDA_MEMCPY_DEVICE_TO_HOST) == 0 },
        None => false,
    }
}

/// Copy `count` bytes from host to a CUDA device pointer. Returns `false` on
/// failure or if libcudart is unavailable.
///
/// The counterpart of [`memcpy_device_to_host`], for filling a writable
/// mapping ([`CudaHandle::map_mut`]) from host bytes.
///
/// # Safety
///
/// The caller must ensure:
/// - `host` points to at least `count` bytes of readable memory.
/// - `device` points to at least `count` bytes of valid CUDA device memory
///   (i.e. obtained from a `CudaMap::device_ptr()` while the map is live).
pub unsafe fn memcpy_host_to_device(
    device: *mut c_void,
    host: *const c_void,
    count: usize,
) -> bool {
    match table() {
        Some(t) => unsafe { (t.memcpy)(device, host, count, CUDA_MEMCPY_HOST_TO_DEVICE) == 0 },
        None => false,
    }
}

/// Create a CUDA stream. Returns `None` if libcudart is unavailable or stream
/// creation fails. The returned stream must be released with
/// [`stream_destroy`]. Intended for clients (e.g. the codec's nvJPEG backend)
/// that submit async device work and synchronise on it.
pub fn stream_create() -> Option<CudaStream> {
    let t = table()?;
    let mut stream: CudaStream = std::ptr::null_mut();
    if unsafe { (t.stream_create)(&mut stream) } != 0 {
        return None;
    }
    Some(stream)
}

/// Block until all work submitted to `stream` completes. Returns `false` on
/// failure or if libcudart is unavailable.
///
/// # Safety
///
/// `stream` must be a live stream returned by [`stream_create`] and not yet
/// destroyed.
pub unsafe fn stream_synchronize(stream: CudaStream) -> bool {
    match table() {
        Some(t) => unsafe { (t.stream_synchronize)(stream) == 0 },
        None => false,
    }
}

/// Destroy a CUDA stream. No-op if libcudart is unavailable.
///
/// # Safety
///
/// `stream` must be a live stream returned by [`stream_create`] and must not be
/// used after this call.
pub unsafe fn stream_destroy(stream: CudaStream) {
    if let Some(t) = table() {
        let _ = unsafe { (t.stream_destroy)(stream) };
    }
}

/// Register a GL buffer (PBO) with CUDA. Returns the resource as `usize`
/// (pointer) or `None`. MUST be called on the thread where the GL context is
/// current.
pub fn gl_register_buffer(buffer_id: u32) -> Option<usize> {
    let t = table()?;
    let mut res: GraphicsResource = std::ptr::null_mut();
    // cudaGraphicsRegisterFlagsNone = 0
    let rc = unsafe { (t.graphics_gl_register_buffer)(&mut res, buffer_id, 0) };
    if rc != 0 {
        log::debug!("cudaGraphicsGLRegisterBuffer(buffer={buffer_id}) failed: cudaError {rc}");
        return None;
    }
    Some(res as usize)
}

/// Map a registered resource → `(device ptr as usize, size)`. GL-thread only.
pub fn gl_map_resource(resource: usize) -> Option<(usize, usize)> {
    let t = table()?;
    let mut res = resource as GraphicsResource;
    if unsafe { (t.graphics_map_resources)(1, &mut res, std::ptr::null_mut()) } != 0 {
        return None;
    }
    let (mut ptr, mut size) = (std::ptr::null_mut::<c_void>(), 0usize);
    if unsafe { (t.graphics_get_mapped_pointer)(&mut ptr, &mut size, res) } != 0 {
        unsafe {
            (t.graphics_unmap_resources)(1, &mut res, std::ptr::null_mut());
        }
        return None;
    }
    Some((ptr as usize, size))
}

/// Unmap a previously mapped resource. GL-thread only.
pub fn gl_unmap_resource(resource: usize) {
    if let Some(t) = table() {
        let mut r = resource as GraphicsResource;
        unsafe {
            (t.graphics_unmap_resources)(1, &mut r, std::ptr::null_mut());
        }
    }
}

/// Unregister a previously registered resource. GL-thread only.
pub fn gl_unregister_resource(resource: usize) {
    if let Some(t) = table() {
        unsafe {
            (t.graphics_unregister_resource)(resource as GraphicsResource);
        }
    }
}

// =============================================================================
// DMA-BUF → CUDA external memory import (thread-independent; no GL context).
//
// ABI verified against CUDA 12.6 driver_types.h, LP64, stable across CUDA
// 11/12.  The structs are layout-asserted in the `ext_mem_layout` test module
// below — no host with both /dev/dma_heap and CUDA is available in CI, so
// runtime validation is deferred to on-target testing (orin-nano, gpu-probe O5
// already confirmed cudaImportExternalMemory(OpaqueFd) works on Orin).
// =============================================================================

/// `cudaExternalMemoryHandleTypeOpaqueFd` — the only handle type used for
/// Linux DMA-BUF fds. Value verified vs. driver_types.h for CUDA 11/12.
#[allow(dead_code)] // only reached on Linux DMA tensors; kept cross-platform + ABI-tested
pub(crate) const CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_uint = 1;

/// FFI mirror of `cudaExternalMemoryHandleDesc` (driver_types.h, LP64).
///
/// Layout (size 40, align 8):
/// - `type_`       → int       @ 0
/// - `_pad0`       → u32       @ 4  (align the union to 8)
/// - `handle_fd`   → int       @ 8  (first member of the 16-byte union)
/// - `_union_rest` → [u32; 3]  @ 12 (pads union to 16 bytes; ends at 24)
/// - `size`        → u64       @ 24
/// - `flags`       → c_uint    @ 32
/// - `_tail`       → u32       @ 36 (struct size 40)
#[allow(dead_code)] // only reached on Linux DMA tensors; kept cross-platform + ABI-tested
#[repr(C)]
pub(crate) struct CudaExternalMemoryHandleDesc {
    pub type_: c_int,
    pub _pad0: u32,
    pub handle_fd: c_int,
    pub _union_rest: [u32; 3],
    pub size: u64,
    pub flags: c_uint,
    pub _tail: u32,
}

/// FFI mirror of `cudaExternalMemoryBufferDesc` (driver_types.h, LP64).
///
/// Layout (size 24, align 8):
/// - `offset` → u64    @ 0
/// - `size`   → u64    @ 8
/// - `flags`  → c_uint @ 16
/// - `_tail`  → u32    @ 20 (struct size 24)
#[allow(dead_code)] // only reached on Linux DMA tensors; kept cross-platform + ABI-tested
#[repr(C)]
pub(crate) struct CudaExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub _tail: u32,
}

/// Import a DMA-BUF fd as CUDA external memory and map it to a device pointer.
///
/// Thread-independent — no GL context is required. `cudaImportExternalMemory`
/// with OpaqueFd takes ownership of the fd it is given, so this function dups
/// the caller's `fd` and hands CUDA the dup; the caller's `fd` is therefore
/// untouched and remains owned by the caller. Returns `(ext_mem_handle,
/// device_ptr)` on success, or `None` on any failure (missing libcudart,
/// unsupported platform, dup failure, or driver error). The returned handle
/// must be destroyed via `cudaDestroyExternalMemory` (done by [`CudaHandle`]
/// drop), which also closes the dup'd fd.
///
/// # RUNTIME-UNVALIDATED
/// No test platform has both `/dev/dma_heap` and a CUDA device. ABI is
/// layout-asserted vs. CUDA 12.6 `driver_types.h`; the mechanism is proven
/// by gpu-probe O5 on Orin. Best-effort: returns `None` on failure.
// `static`-only: the sole caller, `Tensor::try_init_dma_cuda` (`lib.rs`), is
// itself `#[cfg(feature = "static")]` (it reads `self.storage:
// TensorStorage`, a `static`-only field). The rest of this module
// (`CudaHandle`, `CudaMap`, `gl_register_buffer`, ...) stays available under
// `dynamic` -- `dynamic_tensor::Tensor::cuda`/`cuda_map` return their types.
#[cfg(all(target_os = "linux", feature = "static"))]
pub(crate) fn import_dma_fd(fd: i32, size: usize) -> Option<(ExternalMemory, *mut c_void)> {
    use std::os::fd::{AsRawFd, BorrowedFd, IntoRawFd};
    let t = table()?;
    // cudaExternalMemoryHandleTypeOpaqueFd TAKES OWNERSHIP of the fd on a
    // successful import (CUDA closes it at cudaDestroyExternalMemory). The
    // caller's fd is owned by TensorStorage::Dma and closed on tensor drop,
    // so hand CUDA a dup to avoid a double-close.
    // Keep the dup as OwnedFd until CUDA tells us whether it consumed it.
    // `into_raw_fd` before the call left a window where a failed import that
    // still closed the fd (observed on discrete NVIDIA + /dev/dma_heap) made
    // `OwnedFd::from_raw_fd` + drop trip Rust's IO-safety abort.
    let dup = unsafe { BorrowedFd::borrow_raw(fd) }
        .try_clone_to_owned()
        .ok()?;
    let mut desc: CudaExternalMemoryHandleDesc = unsafe { std::mem::zeroed() };
    desc.type_ = CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD as c_int;
    desc.handle_fd = dup.as_raw_fd();
    desc.size = size as u64;
    let mut ext: ExternalMemory = std::ptr::null_mut();
    let rc = unsafe { (t.import_external_memory)(&mut ext, &desc as *const _ as *const c_void) };
    if rc != 0 {
        log::debug!("cudaImportExternalMemory(OpaqueFd, size={size}) failed: cudaError {rc}");
        // CUDA docs: a failed import does not consume the fd. Discrete NVIDIA
        // drivers have been observed to close it anyway; reclaim via libc so
        // an already-closed dup does not trip Rust's IO-safety abort.
        let raw = dup.into_raw_fd();
        if unsafe { libc::close(raw) } != 0 {
            log::debug!(
                "reclaim of CUDA-import dup fd {raw} after failed import: {}",
                std::io::Error::last_os_error()
            );
        }
        return None;
    }
    // Success: CUDA now owns the dup; it is closed by cudaDestroyExternalMemory.
    let _ = dup.into_raw_fd();
    let bdesc = CudaExternalMemoryBufferDesc {
        offset: 0,
        size: size as u64,
        flags: 0,
        _tail: 0,
    };
    let mut dptr: *mut c_void = std::ptr::null_mut();
    if unsafe {
        (t.external_memory_get_mapped_buffer)(&mut dptr, ext, &bdesc as *const _ as *const c_void)
    } != 0
    {
        unsafe { (t.destroy_external_memory)(ext) };
        return None;
    }
    Some((ext, dptr))
}

// =============================================================================
// D3D11 texture → CUDA external memory import (Windows only; Task C2 is the
// consumer -- `import_d3d11_texture`, the fence wait, and the array copies).
//
// Field offsets verified against this box's CUDA 11.8 `driver_types.h` by a
// compiled offsetof program (`cl /nologo /I"%CUDA_PATH%\include" layouts.c`,
// MSVC/VS 2026, x64); see the `d3d11_desc_layout` test module below for the
// printed numbers.
//
// Layout rule: the three structs below (all but `cudaExternalSemaphoreWaitParams`,
// which already matches 11.8) are laid out as the newest known toolkit
// defines them, not as 11.8 defines them, because `load()` tries
// `cudart64_13.dll` and `cudart64_12.dll` before `cudart64_110.dll` and a
// newer runtime reads the struct size and trailing fields it expects, not
// the ones 11.8 has. Each struct's own doc comment gives its exact offsets,
// size and padding.
// =============================================================================

/// `cudaExternalMemoryHandleTypeD3D11Resource`.
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
pub(crate) const CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE: c_int = 5;
/// `cudaExternalSemaphoreHandleTypeD3D11Fence`.
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
pub(crate) const CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE: c_int = 4;
/// `cudaExternalMemoryDedicated` — the imported resource is a dedicated
/// allocation (required for D3D11 textures, as opposed to a suballocated
/// buffer region).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
pub(crate) const CUDA_EXTERNAL_MEMORY_DEDICATED: c_uint = 1;
/// `cudaMemcpyDeviceToDevice` — both endpoints of the copy are CUDA device
/// memory (the imported array and the tensor's linear device buffer).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
/// `cudaChannelFormatKindUnsigned` — integer texture formats (RGBA8, R8).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
pub(crate) const CUDA_CHANNEL_FORMAT_KIND_UNSIGNED: c_int = 1;
/// `cudaChannelFormatKindFloat` — float texture formats (F16, F32).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
pub(crate) const CUDA_CHANNEL_FORMAT_KIND_FLOAT: c_int = 2;

/// `cudaExternalMemoryHandleDesc` with the `win32` arm of the handle union
/// populated (`type_`, `win32_handle`/`win32_name`, `size`, `flags`), laid
/// out as CUDA 12.x/13.x's `driver_types.h` defines it (see the section
/// comment above).
///
/// Field offsets `type_` @0, `win32_handle` @8, `win32_name` @16, `size`
/// @24, `flags` @32 are measured against this box's CUDA 11.8
/// `driver_types.h` (identical in 12.x/13.x, which only append `reserved`).
/// Total size 104, align 8: `reserved[16]` @36..100, tail padding to 104
/// (12.x/13.x add this trailing array; 11.8 does not, but ignores it since
/// 11.8 only reads its own 40-byte prefix).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
#[repr(C)]
pub(crate) struct CudaExternalMemoryHandleDescWin32 {
    pub type_: c_int,
    pub _pad0: u32,
    pub win32_handle: *mut c_void,
    pub win32_name: *const c_void,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [u32; 16],
    pub _tail: u32,
}

/// `cudaChannelFormatDesc` (driver_types.h; size 20, align 4). Generic across
/// platforms; only reached via the Windows mipmapped-array desc for now.
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaChannelFormatDesc {
    pub x: c_int,
    pub y: c_int,
    pub z: c_int,
    pub w: c_int,
    pub f: c_int,
}

/// `cudaExtent` (driver_types.h; size 24, align 8). Generic across platforms;
/// only reached via the Windows mipmapped-array desc for now.
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

/// `cudaExternalMemoryMipmappedArrayDesc`, laid out as CUDA 12.x/13.x's
/// `driver_types.h` defines it (see the section comment above).
///
/// Field offsets `offset` @0, `format_desc` @8, `extent` @32, `flags` @56,
/// `num_levels` @60 are measured against this box's CUDA 11.8
/// `driver_types.h` (identical in 12.x/13.x, which only append `reserved`).
/// Total size 128, align 8: `reserved[16]` @64..128, no further padding
/// needed (12.x/13.x add this trailing array; 11.8 does not, but ignores it
/// since 11.8 only reads its own 64-byte prefix).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
#[repr(C)]
pub(crate) struct CudaExternalMemoryMipmappedArrayDesc {
    pub offset: u64,
    pub format_desc: CudaChannelFormatDesc,
    pub _pad0: u32,
    pub extent: CudaExtent,
    pub flags: c_uint,
    pub num_levels: c_uint,
    pub reserved: [u32; 16],
}

/// `cudaExternalSemaphoreHandleDesc` with the `win32` arm of the handle
/// union populated, laid out as CUDA 12.x/13.x's `driver_types.h` defines it
/// (see the section comment above).
///
/// Field offsets `type_` @0, `win32_handle` @8, `win32_name` @16, `flags`
/// @24 are measured against this box's CUDA 11.8 `driver_types.h` (identical
/// in 12.x/13.x, which only append `reserved`). Total size 96, align 8:
/// `reserved[16]` @28..92, tail padding to 96 (12.x/13.x add this trailing
/// array; 11.8 does not, but ignores it since 11.8 only reads its own
/// 32-byte prefix).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
#[repr(C)]
pub(crate) struct CudaExternalSemaphoreHandleDesc {
    pub type_: c_int,
    pub _pad0: u32,
    pub win32_handle: *mut c_void,
    pub win32_name: *const c_void,
    pub flags: c_uint,
    pub reserved: [u32; 16],
    pub _tail: u32,
}

/// `cudaExternalSemaphoreWaitParams`.
///
/// Layout verified against CUDA 11.8 `driver_types.h` (size 144, align 8):
/// `params.fence.value` @0, `params.nvSciSync` @8, `params.keyedMutex.key`
/// @16, `params.keyedMutex.timeoutMs` @24, `params.reserved[10]` @32,
/// `flags` @72, `reserved[16]` @76, tail padding to 144. This struct already
/// carries the reserved arrays in CUDA 11.8, so its size and offsets match
/// the task brief's guess exactly (unlike the three structs above).
#[allow(dead_code)] // unused under the dynamic backend (the D3D11 arm is static-only)
#[cfg(target_os = "windows")]
#[repr(C)]
pub(crate) struct CudaExternalSemaphoreWaitParams {
    pub fence_value: u64,
    pub nv_sci_sync: u64,
    pub keyed_mutex_key: u64,
    pub keyed_mutex_timeout_ms: c_uint,
    pub _pad0: u32,
    pub params_reserved: [u32; 10],
    pub flags: c_uint,
    pub reserved: [u32; 16],
    pub _tail: u32,
}

/// The CUDA side of one D3D11 texture tensor: the imported external memory,
/// the level-0 array it maps to, and the linear device buffer every map copies
/// through.
///
/// A CUDA array is opaque, tiled memory -- no consumer can index it as a
/// device pointer -- so a map turns it into the tensor's tight rows with one
/// `cudaMemcpy2DFromArray`, and a writable map's release turns them back with
/// one `cudaMemcpy2DToArray`. The linear buffer is allocated on the first map
/// and then reused, so a per-frame consumer pays one `cudaMalloc` in total.
///
/// `static`-only for the same reason `d3d11::texture` is: `dynamic` forwards
/// every tensor call to the C ABI and never constructs a texture storage, so
/// nothing there can produce this backing.
#[cfg(all(target_os = "windows", feature = "static"))]
struct D3d11External {
    ext_mem: ExternalMemory,
    mipmapped: MipmappedArray,
    level0: CudaArray,
    /// The linear buffer, allocated on first use: `map` takes `&self`, so the
    /// slot is behind a lock, which also serialises two concurrent maps that
    /// would otherwise copy into one buffer at once.
    linear: std::sync::Mutex<Option<*mut c_void>>,
    bytes: usize,
    row_bytes: usize,
    rows: usize,
    stream: CudaStream,
    /// The CUDA device the D3D11 adapter maps to. Set again before every copy
    /// and before the release: `cudaSetDevice` is per-thread state, so a
    /// consumer thread that last used another GPU would otherwise allocate the
    /// linear buffer on it and fail to use this handle's stream.
    device: c_int,
    /// The process fence imported as an external semaphore, shared by every
    /// D3D11 import in this copy of the crate (see [`shared_semaphore`]).
    ///
    /// `None` when the device has no shared fence or the import failed. A map
    /// of a tensor with **no** recorded write then copies without waiting,
    /// which is correct -- there is nothing to wait for. A map of a tensor
    /// that *does* carry a recorded write declines instead: it has something
    /// to wait for and no way to, and the caller's host-map fallback is
    /// ordered by the D3D11 immediate context.
    semaphore: Option<&'static SharedSemaphore>,
    /// The newest D3D11 fence value recorded on the tensor, shared with the
    /// storage's own slot: a map waits for it before reading the array.
    last_write: Arc<std::sync::atomic::AtomicU64>,
}

/// Imports a D3D11 texture's shared NT handle as CUDA external memory, maps
/// its level-0 array and imports `fence_handle` as an external semaphore so
/// maps can order their copies after the D3D11 work a producer signalled.
///
/// `handle` stays owned by the caller: CUDA duplicates a Win32 handle it
/// imports rather than taking it over, unlike the OpaqueFd path in
/// `import_dma_fd`. `fence_handle` is a duplicate this backing takes over and
/// closes on drop, for the same reason -- nothing else holds it.
///
/// Returns `None` on any failure (no CUDA device for the adapter, as on WARP;
/// a DXGI format with no channel description; a driver error), releasing
/// everything it had already created. The tensor is then simply CUDA-less.
#[cfg(all(target_os = "windows", feature = "static"))]
pub(crate) fn import_d3d11_texture(
    handle: RawHandle,
    layout: crate::d3d11_layout::D3d11ImageLayout,
    adapter: *mut c_void,
    fence_handle: Option<OwnedHandle>,
    last_write: Arc<std::sync::atomic::AtomicU64>,
) -> Option<CudaHandle> {
    let t = table()?;
    // Every texture tensor is created on the one process D3D11 device, so
    // every import resolves the same ordinal and selects the same CUDA
    // device; caching it keeps `cudaD3D11GetDevice` off the allocation path.
    // The ordinal itself is cached, not just the verdict, because every map
    // and every release sets it again on its own thread.
    static CUDA_DEVICE_SELECTED: OnceLock<Option<c_int>> = OnceLock::new();
    let device = (*CUDA_DEVICE_SELECTED.get_or_init(|| {
        let mut dev = -1;
        // SAFETY: `adapter` is the process device's live `IDXGIAdapter*` and
        // `dev` is a valid out-parameter.
        let rc = unsafe { (t.d3d11_get_device)(&mut dev, adapter) };
        if rc != 0 {
            log::debug!(
                "cudaD3D11GetDevice for the process adapter: {}; D3D11 texture tensors \
                 will have no CUDA handle",
                err_str(t, rc)
            );
            return None;
        }
        // SAFETY: `dev` is the ordinal the call above filled in.
        let rc = unsafe { (t.set_device)(dev) };
        if rc != 0 {
            log::debug!("cudaSetDevice({dev}): {}", err_str(t, rc));
            return None;
        }
        Some(dev)
    }))?;
    // The channel description of the texture's DXGI format: one component's
    // width in bits and how many components a texel carries. Only the formats
    // `crate::d3d11_layout` allocates are listed; anything else declines the
    // import rather than guessing a description the driver would reject.
    let (bits, kind, channels) = match layout.dxgi_format {
        crate::d3d11_layout::DXGI_FORMAT_R8_UNORM => (8, CUDA_CHANNEL_FORMAT_KIND_UNSIGNED, 1),
        crate::d3d11_layout::DXGI_FORMAT_R8G8_UNORM => (8, CUDA_CHANNEL_FORMAT_KIND_UNSIGNED, 2),
        crate::d3d11_layout::DXGI_FORMAT_R8G8B8A8_UNORM
        | crate::d3d11_layout::DXGI_FORMAT_B8G8R8A8_UNORM => {
            (8, CUDA_CHANNEL_FORMAT_KIND_UNSIGNED, 4)
        }
        crate::d3d11_layout::DXGI_FORMAT_R16G16B16A16_FLOAT => {
            (16, CUDA_CHANNEL_FORMAT_KIND_FLOAT, 4)
        }
        crate::d3d11_layout::DXGI_FORMAT_R32G32B32A32_FLOAT => {
            (32, CUDA_CHANNEL_FORMAT_KIND_FLOAT, 4)
        }
        other => {
            log::debug!("no CUDA channel format for DXGI format {other}");
            return None;
        }
    };
    let ch = |i: c_int| if i < channels { bits } else { 0 };
    // SAFETY: every field is an integer, a nested POD struct or an array of
    // those, so all-zero is a valid bit pattern; the named fields below are
    // then set. Zeroing is what makes the 12.x/13.x `reserved` tail
    // well-defined when an 11.8 runtime is the one loaded.
    let mut md: CudaExternalMemoryMipmappedArrayDesc = unsafe { std::mem::zeroed() };
    md.format_desc = CudaChannelFormatDesc {
        x: ch(0),
        y: ch(1),
        z: ch(2),
        w: ch(3),
        f: kind,
    };
    md.extent = CudaExtent {
        width: layout.texture_width,
        height: layout.texture_height,
        depth: 0,
    };
    md.num_levels = 1;
    let (ext, mm) = import_and_map_array(t, handle, &md, layout)?;
    let mut level0: CudaArray = std::ptr::null_mut();
    // SAFETY: `mm` is the array just mapped, which has the one level `md`
    // asked for, and `level0` is a valid out-parameter.
    let rc = unsafe { (t.get_mipmapped_array_level)(&mut level0, mm, 0) };
    let stream = if rc == 0 && !level0.is_null() {
        stream_create()
    } else {
        None
    };
    let Some(stream) = stream else {
        if rc != 0 {
            log::debug!("cudaGetMipmappedArrayLevel(0): {}", err_str(t, rc));
        } else if level0.is_null() {
            log::debug!("cudaGetMipmappedArrayLevel(0) reported success but left a null array");
        } else {
            log::debug!("cudaStreamCreate for an imported D3D11 texture failed");
        }
        // SAFETY: both objects were created above and nothing else holds them;
        // the array is released before the memory it was mapped from.
        unsafe {
            (t.free_mipmapped_array)(mm);
            (t.destroy_external_memory)(ext);
        }
        return None;
    };
    let semaphore = fence_handle.and_then(|fh| shared_semaphore(t, fh));
    Some(CudaHandle {
        size: layout.tight_bytes(),
        kind: CudaBacking::D3d11(D3d11External {
            ext_mem: ext,
            mipmapped: mm,
            level0,
            linear: std::sync::Mutex::new(None),
            bytes: layout.tight_bytes(),
            row_bytes: layout.tight_row_bytes(),
            rows: layout.texture_height,
            stream,
            device,
            semaphore,
            last_write,
        }),
    })
}

/// The process fence imported as a CUDA external semaphore, with the
/// duplicate of its NT handle CUDA needs kept open beside it.
#[cfg(all(target_os = "windows", feature = "static"))]
struct SharedSemaphore {
    sem: ExternalSemaphore,
    /// CUDA duplicates a Win32 handle it imports rather than taking it over,
    /// but the source handle must stay valid for the import; held for the
    /// process lifetime because the semaphore is.
    _handle: OwnedHandle,
}

// SAFETY: a CUDA external-semaphore handle is an opaque runtime object bound
// to a device, not to a thread, and every use here passes it to a runtime call
// that is itself thread-safe. `OwnedHandle` is `Send + Sync`. The struct is
// written once inside a `OnceLock` initialiser and only read afterwards.
#[cfg(all(target_os = "windows", feature = "static"))]
unsafe impl Send for SharedSemaphore {}
#[cfg(all(target_os = "windows", feature = "static"))]
unsafe impl Sync for SharedSemaphore {}

/// Imports the process D3D11 fence as an external semaphore, once.
///
/// Every texture tensor in this copy of the crate records its completions on
/// the one process fence, so importing per tensor would duplicate the same NT
/// handle and import the same object again for every allocation -- a 30-frame
/// pool pays 30 of each for no benefit. The semaphore and its handle live for
/// the process, like the device and the fence they name, so neither is
/// destroyed on a handle's drop.
///
/// The CUDA *stream* is deliberately not hoisted with it: it stays per handle
/// so a map never waits behind another consumer's unrelated work.
///
/// `fence_handle` is consumed by the first caller and dropped by the rest.
#[cfg(all(target_os = "windows", feature = "static"))]
fn shared_semaphore(t: &CudaTable, fence_handle: OwnedHandle) -> Option<&'static SharedSemaphore> {
    static IMPORTED: OnceLock<Option<SharedSemaphore>> = OnceLock::new();
    IMPORTED
        .get_or_init(move || {
            // SAFETY: same all-zero reasoning as the descriptors above.
            let mut sd: CudaExternalSemaphoreHandleDesc = unsafe { std::mem::zeroed() };
            sd.type_ = CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE;
            sd.win32_handle = fence_handle.as_raw_handle();
            let mut sem: ExternalSemaphore = std::ptr::null_mut();
            // SAFETY: `sd` names the live shared handle `fence_handle` owns,
            // which outlives the call, and `sem` is a valid out-parameter.
            let rc = unsafe {
                (t.import_external_semaphore)(&mut sem, &sd as *const _ as *const c_void)
            };
            if rc != 0 {
                // The duplicate closes with `fence_handle` here. Tensors that
                // carry a recorded write then decline their CUDA map rather
                // than copy unordered; see `D3d11External::semaphore`.
                log::debug!(
                    "cudaImportExternalSemaphore(D3D11Fence): {}",
                    err_str(t, rc)
                );
                return None;
            }
            Some(SharedSemaphore {
                sem,
                _handle: fence_handle,
            })
        })
        .as_ref()
}

/// Whether a copy issued now would be ordered behind the producer's recorded
/// D3D11 write: either nothing is recorded, or a wait for it was queued on the
/// same stream.
///
/// A recorded write with no wait behind it is the one case that must not
/// proceed. The copy would run against a texture the producer's GPU work is
/// still writing -- timing-dependent, silent, and invisible below `warn`. It is
/// the hazard `validate_descriptor` refuses a descriptor for, so a map declines
/// here for the same reason: the documented fallback is the host map, which
/// goes through the D3D11 immediate context and *is* ordered.
#[cfg(all(target_os = "windows", feature = "static"))]
fn d3d11_copy_is_ordered(recorded: u64, wait_queued: bool) -> bool {
    recorded == 0 || wait_queued
}

/// Set once a D3D11 allocation has proved that this adapter's driver pads its
/// textures past their padding-free size, so later imports skip the tight rung
/// of [`import_and_map_array`]'s ladder instead of paying a doomed import each
/// time.
#[cfg(all(target_os = "windows", feature = "static"))]
static CUDA_IMPORT_SIZE_PADDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Imports the texture as external memory and maps `md` out of it, returning
/// the pair on success.
///
/// The declared size is the one thing here that cannot be read off the
/// texture: D3D11 has no allocation-size query, and the driver pads a
/// texture's rows and its row count for tiling, so a texture's real
/// allocation is larger than its padding-free bytes at most geometries. This
/// box's RTX 3070 maps a 256x128 RGBA8 texture from the tight 131072 bytes but
/// rejects a 1920x1080 one (`cudaErrorInvalidValue` from
/// `cudaExternalMemoryGetMappedMipmappedArray`, not from the import, which
/// accepts any non-zero size) until the declaration covers the padding.
///
/// So the declaration climbs a short ladder -- the exact bytes, then 2x, then
/// 4x -- and stops at the first rung the driver maps. Over-declaring is what
/// the probe's S5.4 measured as harmless (a 2x declaration of a 256x128
/// texture mapped and copied the same 0-mismatch bytes as the tight one), and
/// it is safe by construction: with `cudaExternalMemoryDedicated` the mapping
/// covers the whole resource, no address is derived from this number, and the
/// array's own extent bounds every copy.
///
/// The first rung is floored at one 4 KiB page: no GPU allocation is smaller
/// than a page, so a 1x1 RGBA texture's four bytes are provably too few to
/// describe one, and 4x of too few is still too few. Measured on this box --
/// 1x1 and 4x4 RGBA and 7x3 R8 map at the floor or its doublings and at
/// nothing the unfloored ladder reaches; every image-sized texture is far
/// above the floor and unaffected by it.
///
/// Whether the tight rung works is a property of the adapter, not of the
/// texture, so the first allocation that has to climb records it in
/// [`CUDA_IMPORT_SIZE_PADDED`] and later ones start one rung up.
#[cfg(all(target_os = "windows", feature = "static"))]
fn import_and_map_array(
    t: &CudaTable,
    handle: RawHandle,
    md: &CudaExternalMemoryMipmappedArrayDesc,
    layout: crate::d3d11_layout::D3d11ImageLayout,
) -> Option<(ExternalMemory, MipmappedArray)> {
    /// The ladder's floor, one page.
    const PAGE: usize = 4096;
    let tight = layout.tight_bytes();
    let start = usize::from(CUDA_IMPORT_SIZE_PADDED.load(std::sync::atomic::Ordering::Relaxed));
    for multiple in [1usize, 2, 4].into_iter().skip(start) {
        let size = tight.max(PAGE).saturating_mul(multiple);
        // SAFETY: every field is an integer or a raw pointer, so all-zero is a
        // valid bit pattern; the named fields below are then set.
        let mut hd: CudaExternalMemoryHandleDescWin32 = unsafe { std::mem::zeroed() };
        hd.type_ = CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE;
        hd.win32_handle = handle;
        hd.size = size as u64;
        hd.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        let mut ext: ExternalMemory = std::ptr::null_mut();
        // SAFETY: `hd` is a fully initialised descriptor naming a live shared
        // NT handle of a D3D11 texture, and `ext` is a valid out-parameter.
        let rc = unsafe { (t.import_external_memory)(&mut ext, &hd as *const _ as *const c_void) };
        if rc != 0 {
            log::debug!(
                "cudaImportExternalMemory(D3D11Resource, size {size}): {}",
                err_str(t, rc)
            );
            continue;
        }
        let mut mm: MipmappedArray = std::ptr::null_mut();
        // SAFETY: `ext` is the live import, `md` a fully initialised
        // descriptor of the texture's own geometry, and `mm` a valid
        // out-parameter.
        let rc = unsafe {
            (t.external_memory_get_mapped_mipmapped_array)(
                &mut mm,
                ext,
                md as *const _ as *const c_void,
            )
        };
        if rc == 0 {
            // Logged once, by the allocation that discovers the padding.
            if multiple > 1
                && !CUDA_IMPORT_SIZE_PADDED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                log::debug!(
                    "this adapter pads D3D11 allocations: a {tight}-byte texture mapped \
                     only when declared at {multiple}x ({size} bytes), so later imports \
                     start there"
                );
            }
            return Some((ext, mm));
        }
        log::debug!(
            "cudaExternalMemoryGetMappedMipmappedArray over {size} declared bytes: {}",
            err_str(t, rc)
        );
        // SAFETY: `ext` is the import made just above and nothing else holds it.
        unsafe { (t.destroy_external_memory)(ext) };
    }
    None
}

/// Routes `cudaGraphicsMapResources`/Unmap/Unregister through the GL worker
/// thread (the GL context must be current there). Implemented by the image crate.
pub trait CudaGlOps: Send + Sync {
    fn map(&self, resource: GraphicsResource) -> Option<(*mut c_void, usize)>;
    fn unmap(&self, resource: GraphicsResource);
    fn unregister(&self, resource: GraphicsResource);
}

enum CudaBacking {
    #[allow(dead_code)] // consumed by C3/C4
    GlBuffer {
        resource: GraphicsResource,
        ops: Arc<dyn CudaGlOps>,
    },
    #[allow(dead_code)] // consumed by C3/C4
    ExternalMem {
        ext_mem: ExternalMemory,
        dptr: *mut c_void,
    },
    #[cfg(all(target_os = "windows", feature = "static"))]
    D3d11(D3d11External),
}

// SAFETY: CUDA handles/ptrs are process-global; GlBuffer routes to the GL
// worker; ExternalMem ptr is valid via the per-device primary context.
unsafe impl Send for CudaBacking {}
unsafe impl Sync for CudaBacking {}

/// CUDA registration for a GPU-backed tensor. Held as `Option` on the tensor.
pub struct CudaHandle {
    kind: CudaBacking,
    size: usize,
}

impl std::fmt::Debug for CudaHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.kind {
            CudaBacking::GlBuffer { .. } => "GlBuffer",
            CudaBacking::ExternalMem { .. } => "ExternalMem",
            #[cfg(all(target_os = "windows", feature = "static"))]
            CudaBacking::D3d11(_) => "D3d11",
        };
        f.debug_struct("CudaHandle")
            .field("kind", &kind)
            .field("size", &self.size)
            .finish()
    }
}

impl CudaHandle {
    /// Construct a GL-buffer-backed CUDA handle. `resource` is the
    /// `cudaGraphicsResource_t` returned by [`gl_register_buffer`]; `ops`
    /// routes map/unmap/unregister back to the GL-context thread.
    pub fn new_gl(resource: GraphicsResource, size: usize, ops: Arc<dyn CudaGlOps>) -> Self {
        Self {
            kind: CudaBacking::GlBuffer { resource, ops },
            size,
        }
    }

    #[allow(dead_code)] // consumed by C3/C4
    pub(crate) fn new_external(ext_mem: ExternalMemory, dptr: *mut c_void, size: usize) -> Self {
        Self {
            kind: CudaBacking::ExternalMem { ext_mem, dptr },
            size,
        }
    }

    /// Map to a device pointer. `GlBuffer` routes to the GL worker;
    /// `ExternalMem` is persistent (no per-call map/unmap); `D3d11` copies the
    /// texture's array into its linear buffer, after waiting for the newest
    /// fence value recorded on the tensor.
    ///
    /// The `D3d11` mapping is the texture's *tight* rows: `len()` is
    /// `width * bytes_per_texel * height` and row `n` starts at
    /// `n * width * bytes_per_texel`, whatever the tensor's
    /// [`row_stride`](crate::Tensor::row_stride) reports -- that is the D3D11
    /// staging pitch a CPU map sees, a different number on a padded backing.
    /// A consumer that strides the device pointer by `row_stride()` walks off
    /// the end of the mapping.
    ///
    /// Semi-planar textures (NV12, NV16, NV24) are the exception: the texture
    /// is as wide as its pitch by construction, so the mapping equals
    /// `row_stride() x combined_height` bytes and striding by `row_stride()`
    /// is exactly right.
    ///
    /// One `D3d11` handle has one linear buffer, so two live maps of the same
    /// tensor hand out the same device pointer and the second map's copy
    /// refreshes what the first is reading. That is the same aliasing the GL
    /// and external-memory backings have always had -- one buffer per handle,
    /// not one per map. A `map()` taken while a [`map_mut`](Self::map_mut)
    /// guard is live is the case that also loses data: it refreshes the buffer
    /// under the writer, and the writer's release then publishes the array's
    /// pre-write contents back into the texture. Callers serialise the two.
    pub fn map(&self) -> Option<CudaMap<'_>> {
        match &self.kind {
            CudaBacking::GlBuffer { resource, ops } => {
                let (ptr, len) = ops.map(*resource)?;
                Some(CudaMap {
                    ptr,
                    len,
                    release: Release::GlUnmap(ops.clone(), *resource),
                    _marker: std::marker::PhantomData,
                })
            }
            CudaBacking::ExternalMem { dptr, .. } => Some(CudaMap {
                ptr: *dptr,
                len: self.size,
                release: Release::None,
                _marker: std::marker::PhantomData,
            }),
            #[cfg(all(target_os = "windows", feature = "static"))]
            CudaBacking::D3d11(x) => Some(CudaMap {
                ptr: self.d3d11_copy_in(x)?,
                len: x.bytes,
                release: Release::None,
                _marker: std::marker::PhantomData,
            }),
        }
    }

    /// Map to a device pointer for writing.
    ///
    /// The GL and external-memory backings alias the same device memory for
    /// reads and writes -- a GL buffer's CUDA mapping and an imported
    /// external-memory allocation are each one address the caller may write
    /// through -- so for them this is [`map`](Self::map). A `D3d11` texture is
    /// the backing that does distinguish the two: its linear buffer is a copy
    /// of an opaque array, so the map's release copies it back into the
    /// texture and synchronises, and later D3D11 and GL work on the immediate
    /// context is ordered after that write.
    ///
    /// The `D3d11` mapping is the texture's *tight* rows: `len()` is
    /// `width * bytes_per_texel * height` and row `n` starts at
    /// `n * width * bytes_per_texel`, whatever the tensor's
    /// [`row_stride`](crate::Tensor::row_stride) reports -- that is the D3D11
    /// staging pitch a CPU map sees, a different number on a padded backing.
    /// A consumer that strides the device pointer by `row_stride()` walks off
    /// the end of the mapping.
    ///
    /// Semi-planar textures (NV12, NV16, NV24) are the exception: the texture
    /// is as wide as its pitch by construction, so the mapping equals
    /// `row_stride() x combined_height` bytes and striding by `row_stride()`
    /// is exactly right.
    pub fn map_mut(&self) -> Option<CudaMap<'_>> {
        #[cfg(all(target_os = "windows", feature = "static"))]
        if let CudaBacking::D3d11(x) = &self.kind {
            return Some(CudaMap {
                ptr: self.d3d11_copy_in(x)?,
                len: x.bytes,
                release: Release::D3d11Writeback {
                    level0: x.level0,
                    row_bytes: x.row_bytes,
                    rows: x.rows,
                    stream: x.stream,
                    device: x.device,
                },
                _marker: std::marker::PhantomData,
            });
        }
        self.map()
    }

    /// Waits for the tensor's newest recorded D3D11 write, then refreshes the
    /// linear buffer from the texture's array and returns its device pointer.
    ///
    /// The wait, the copy and the synchronisation all run on this backing's
    /// own stream, so the copy is ordered behind the D3D11 signal (the
    /// sequence gpu-probe S5.6 measured as the one that never reads a stale
    /// frame) without a device-wide barrier: a legacy-stream synchronisation
    /// would also wait behind streams this crate hands to other consumers.
    ///
    /// A recorded fence value the producer never signals blocks the
    /// synchronisation below until it is. That is inherent to waiting on a
    /// fence: a value is a promise of work, and `set_gpu_write` is the
    /// producer's statement that it queued it.
    ///
    /// Answers `None` when the wait cannot be issued at all
    /// ([`d3d11_copy_is_ordered`]), like every other failure here.
    #[cfg(all(target_os = "windows", feature = "static"))]
    fn d3d11_copy_in(&self, x: &D3d11External) -> Option<*mut c_void> {
        let t = table()?;
        // Per-thread state: a consumer thread that last used another GPU would
        // otherwise allocate on it and be unable to use this handle's stream.
        // SAFETY: `x.device` is the ordinal `import_d3d11_texture` cached for
        // the process D3D11 adapter.
        let rc = unsafe { (t.set_device)(x.device) };
        if rc != 0 {
            log::debug!("cudaSetDevice({}): {}", x.device, err_str(t, rc));
            return None;
        }
        let mut guard = x.linear.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let mut p: *mut c_void = std::ptr::null_mut();
            // SAFETY: `p` is a valid out-parameter and `x.bytes` the texture's
            // padding-free byte count.
            let rc = unsafe { (t.malloc)(&mut p, x.bytes) };
            if rc != 0 {
                log::debug!("cudaMalloc({}): {}", x.bytes, err_str(t, rc));
                return None;
            }
            *guard = Some(p);
        }
        let dptr = guard.unwrap();
        let value = x.last_write.load(std::sync::atomic::Ordering::Acquire);
        let mut queued = false;
        if let (Some(shared), true) = (x.semaphore, value != 0) {
            // SAFETY: every field is an integer or an array of integers, so
            // all-zero is valid; only `fence_value` is then set.
            let mut wp: CudaExternalSemaphoreWaitParams = unsafe { std::mem::zeroed() };
            wp.fence_value = value;
            // SAFETY: `shared.sem` is the process semaphore, live for the
            // process, `wp` a fully initialised wait parameter for the one
            // semaphore passed, and `x.stream` this backing's live stream.
            let rc = unsafe {
                (t.wait_external_semaphores_async)(
                    &shared.sem,
                    &wp as *const _ as *const c_void,
                    1,
                    x.stream,
                )
            };
            queued = rc == 0;
            if rc != 0 {
                log::warn!(
                    "cudaWaitExternalSemaphoresAsync({value}): {}",
                    err_str(t, rc)
                );
            }
        }
        if !d3d11_copy_is_ordered(value, queued) {
            log::warn!(
                "CUDA map of a D3D11 texture with a recorded write ({value}) this handle cannot \
                 wait for ({}); declining the device mapping so the caller falls back to the \
                 host map, which the immediate context orders",
                if x.semaphore.is_some() {
                    "the wait could not be queued"
                } else {
                    "no external semaphore was imported"
                }
            );
            return None;
        }
        // One device copy turns the tiled array into the tensor's tight rows,
        // on the same stream as the wait above so it runs behind it.
        // SAFETY: `dptr` holds `x.rows * x.row_bytes == x.bytes` bytes,
        // `x.level0` is the live level-0 array of the imported texture, which
        // is `x.row_bytes` wide and `x.rows` tall, and `x.stream` is this
        // backing's live stream.
        let rc = unsafe {
            (t.memcpy_2d_from_array_async)(
                dptr,
                x.row_bytes,
                x.level0,
                0,
                0,
                x.row_bytes,
                x.rows,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                x.stream,
            )
        };
        if rc != 0 {
            log::warn!("cudaMemcpy2DFromArrayAsync: {}", err_str(t, rc));
            return None;
        }
        // SAFETY: `x.stream` is this backing's live stream.
        let rc = unsafe { (t.stream_synchronize)(x.stream) };
        if rc != 0 {
            log::warn!(
                "cudaStreamSynchronize after the array copy: {}",
                err_str(t, rc)
            );
            return None;
        }
        Some(dptr)
    }
}

impl Drop for CudaHandle {
    fn drop(&mut self) {
        match &self.kind {
            CudaBacking::GlBuffer { resource, ops } => ops.unregister(*resource),
            CudaBacking::ExternalMem { ext_mem, dptr: _ } => {
                // The device pointer comes from `cudaExternalMemoryGetMappedBuffer`,
                // which CUDA frees together with the external-memory object. Calling
                // `cudaFree` on such a pointer is explicitly disallowed and corrupts
                // the driver's bookkeeping (risking a double-free when the handle is
                // destroyed), so only destroy the external-memory object here.
                if let Some(t) = table() {
                    unsafe {
                        (t.destroy_external_memory)(*ext_mem);
                    }
                }
            }
            // Everything `import_d3d11_texture` created for *this* handle,
            // released in reverse: the linear buffer (this one IS a
            // `cudaMalloc` pointer, unlike `ExternalMem`'s), then the stream,
            // then the array before the external memory it was mapped from.
            // The external semaphore is not here: it is the process import
            // (`shared_semaphore`) and outlives every handle.
            #[cfg(all(target_os = "windows", feature = "static"))]
            CudaBacking::D3d11(x) => {
                let Some(t) = table() else { return };
                // The handle can be dropped on any thread; the objects below
                // all belong to the D3D11 adapter's device.
                // SAFETY: `x.device` is the ordinal `import_d3d11_texture`
                // cached for that adapter.
                let rc = unsafe { (t.set_device)(x.device) };
                if rc != 0 {
                    log::debug!(
                        "cudaSetDevice({}) before releasing a D3D11 import: {}",
                        x.device,
                        err_str(t, rc)
                    );
                }
                let linear = x.linear.lock().unwrap_or_else(|e| e.into_inner()).take();
                // SAFETY: every object below was created by
                // `import_d3d11_texture` for this handle, which is being
                // dropped, so nothing else can hold or use them.
                unsafe {
                    if let Some(p) = linear {
                        (t.free)(p);
                    }
                    (t.stream_destroy)(x.stream);
                    (t.free_mipmapped_array)(x.mipmapped);
                    (t.destroy_external_memory)(x.ext_mem);
                }
            }
        }
    }
}

/// What a [`CudaMap`] does when it drops: nothing for a persistent mapping,
/// a GL unmap, a write-back into the D3D11 texture's array, or -- on the
/// `dynamic` backend -- the C library's own unmap of a mapping it handed out.
enum Release {
    None,
    GlUnmap(Arc<dyn CudaGlOps>, GraphicsResource),
    #[cfg(feature = "dynamic")]
    FfiUnmap(*mut c_void),
    #[cfg(all(target_os = "windows", feature = "static"))]
    D3d11Writeback {
        level0: CudaArray,
        row_bytes: usize,
        rows: usize,
        stream: CudaStream,
        device: c_int,
    },
}

/// Scoped CUDA device-pointer mapping. `Drop` unmaps a `GlBuffer` (so GL may
/// reuse the PBO for the next `convert()` call) and copies a writable `D3d11`
/// mapping back into the texture. `ExternalMem` mappings are persistent —
/// `Drop` is a no-op.
///
/// A `D3d11` mapping is the texture's tight rows: [`len`](Self::len) is
/// `width * bytes_per_texel * height` and rows are `width * bytes_per_texel`
/// apart, which is *not* the tensor's `row_stride()` (the D3D11 staging pitch
/// a CPU map sees) on a padded backing. A semi-planar texture is as wide as
/// its own pitch, so there the two numbers agree and the mapping is
/// `row_stride() x combined_height` bytes.
pub struct CudaMap<'a> {
    ptr: *mut c_void,
    len: usize,
    release: Release,
    _marker: std::marker::PhantomData<&'a ()>,
}

// SAFETY: the mapped device pointer is process-global and valid cross-thread
// via the per-device CUDA primary context; the routed CudaGlOps is Send+Sync.
// A `Release::FfiUnmap` handle is likewise cross-thread: it owns a retain on a
// refcounted `ef_tensor`, whose retain and free are atomic and whose accessors
// are documented safe to call from any thread (`crates/tensor-capi/src/
// handle.rs`, `ef_tensor_retain` and `ef_tensor_set_colorimetry`'s
// Concurrency note). Required so callers can hold the guard on a separate
// inference thread.
unsafe impl Send for CudaMap<'_> {}
unsafe impl Sync for CudaMap<'_> {}

impl CudaMap<'_> {
    /// Raw device pointer to the mapped buffer.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Length of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the mapping covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adopt the opaque map `ef_tensor_cuda_map`/`_map_mut` returned.
    ///
    /// The `dynamic` backend's tensors are allocated inside
    /// `libedgefirst_tensor`, which attaches their [`CudaHandle`] to its own
    /// `Tensor` -- an object the Rust-side wrapper does not hold -- so the
    /// mapping is taken and released across the ABI instead. The C map holds
    /// its own retain on the tensor handle, so this guard stays valid for as
    /// long as it lives, and dropping it releases both.
    ///
    /// Returns `None` (having released `raw`) when the map carries no device
    /// pointer, so a caller never sees a mapping it cannot read.
    ///
    /// # Safety
    /// `raw` must be a live, non-NULL map from `ef_tensor_cuda_map` or
    /// `ef_tensor_cuda_map_mut` that no one else will release: ownership
    /// passes to the guard returned here, which calls `ef_tensor_cuda_unmap`
    /// on it exactly once -- immediately when there is no device pointer,
    /// otherwise on drop. A second release is a double free of the map and of
    /// the tensor retain it holds.
    #[cfg(feature = "dynamic")]
    pub(crate) unsafe fn from_ffi(raw: *mut c_void) -> Option<Self> {
        let mut len = 0usize;
        // SAFETY: `raw` is a live map by this function's contract, and `len`
        // is a live local; the accessor takes either kind of map.
        let ptr = unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_device_ptr(raw, &mut len) };
        if ptr.is_null() {
            // SAFETY: as above -- releasing, exactly once, a map whose pointer
            // is unusable, before ownership reaches any caller.
            unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_unmap(raw) };
            return None;
        }
        Some(CudaMap {
            ptr,
            len,
            release: Release::FfiUnmap(raw),
            _marker: std::marker::PhantomData,
        })
    }
}

impl Drop for CudaMap<'_> {
    fn drop(&mut self) {
        // Taken, not borrowed, so the release happens exactly once even if a
        // future `CudaMap` gains a way to be released early.
        match std::mem::replace(&mut self.release, Release::None) {
            Release::None => {}
            Release::GlUnmap(ops, r) => ops.unmap(r),
            // The write-back a writable D3D11 map owes happens inside the C
            // library, on the `CudaMap` this handle wraps there.
            #[cfg(feature = "dynamic")]
            Release::FfiUnmap(h) => {
                // SAFETY: the handle came from `ef_tensor_cuda_map`/`_map_mut`
                // and is released exactly once -- `replace` above took it.
                unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_unmap(h) }
            }
            #[cfg(all(target_os = "windows", feature = "static"))]
            Release::D3d11Writeback {
                level0,
                row_bytes,
                rows,
                stream,
                device,
            } => {
                let Some(t) = table() else { return };
                // The guard can be dropped on any thread; the stream and the
                // array belong to the D3D11 adapter's device.
                // SAFETY: `device` is the ordinal `import_d3d11_texture`
                // cached for that adapter.
                let rc = unsafe { (t.set_device)(device) };
                if rc != 0 {
                    // The write-back is what publishes `cuda_map_mut`'s
                    // buffer; `Drop` cannot propagate, so this line is the
                    // whole report that the caller's writes were lost.
                    log::warn!(
                        "cudaSetDevice({device}) before the D3D11 write-back: {}",
                        err_str(t, rc)
                    );
                    return;
                }
                // SAFETY: `self.ptr` is the linear buffer this map was built
                // over, holding `rows * row_bytes` bytes, `level0` the live
                // level-0 array of the same geometry, and `stream` the
                // backing's live stream, which outlives this map.
                let rc = unsafe {
                    (t.memcpy_2d_to_array_async)(
                        level0,
                        0,
                        0,
                        self.ptr,
                        row_bytes,
                        row_bytes,
                        rows,
                        CUDA_MEMCPY_DEVICE_TO_DEVICE,
                        stream,
                    )
                };
                if rc != 0 {
                    log::warn!("cudaMemcpy2DToArrayAsync: {}", err_str(t, rc));
                }
                // Synchronised here, not left in flight: the copy above is
                // asynchronous, and later D3D11 and GL work on the immediate
                // context has no other ordering against this write.
                // SAFETY: `stream` is the backing's live stream.
                let rc = unsafe { (t.stream_synchronize)(stream) };
                if rc != 0 {
                    log::warn!(
                        "cudaStreamSynchronize after the write-back: {}",
                        err_str(t, rc)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod ext_mem_layout {
    use super::*;
    #[test]
    fn external_memory_desc_abi() {
        assert_eq!(std::mem::size_of::<CudaExternalMemoryHandleDesc>(), 40);
        assert_eq!(std::mem::align_of::<CudaExternalMemoryHandleDesc>(), 8);
        // size field at offset 24, flags at 32 (verified vs driver_types.h)
        let d: CudaExternalMemoryHandleDesc = unsafe { std::mem::zeroed() };
        let base = &d as *const _ as usize;
        assert_eq!((&d.size as *const _ as usize) - base, 24);
        assert_eq!((&d.flags as *const _ as usize) - base, 32);
        assert_eq!(std::mem::size_of::<CudaExternalMemoryBufferDesc>(), 24);
        let b: CudaExternalMemoryBufferDesc = unsafe { std::mem::zeroed() };
        let bb = &b as *const _ as usize;
        assert_eq!((&b.size as *const _ as usize) - bb, 8);
    }
}

#[cfg(all(test, target_os = "windows", feature = "static"))]
mod d3d11_ordering {
    use super::d3d11_copy_is_ordered;

    /// The rule the D2D copy is gated on. A recorded write with no wait behind
    /// it is the one combination that must decline: copying would read a
    /// texture the producer is still writing, silently and only sometimes,
    /// while `None` sends the caller to the host map the immediate context
    /// orders.
    #[test]
    fn a_recorded_write_with_no_wait_behind_it_declines_the_copy() {
        assert!(d3d11_copy_is_ordered(0, false), "nothing to wait for");
        assert!(d3d11_copy_is_ordered(0, true));
        assert!(d3d11_copy_is_ordered(42, true), "the wait was queued");
        assert!(
            !d3d11_copy_is_ordered(42, false),
            "no semaphore, or the wait failed to queue"
        );
    }
}

#[cfg(all(test, target_os = "windows"))]
mod d3d11_desc_layout {
    use super::*;
    fn off<T, F>(base: &T, field: &F) -> usize {
        (field as *const F as usize) - (base as *const T as usize)
    }

    // Field offsets measured against this box's CUDA 11.8 driver_types.h
    // with a compiled offsetof program (`cl /nologo /I"%CUDA_PATH%\include"
    // layouts.c`, MSVC from VS 2026's dev shell, x64):
    //   handle_desc 40 handle@8 size@24 flags@32
    //   mip_desc 64 format@8 extent@32 flags@56 levels@60
    //   sem_desc 32 handle@8 flags@24
    //   wait_params 144 value@0 flags@72
    // These offsets are asserted below and hold in both 11.8 and 12.x/13.x
    // (the newer headers only append fields after them).
    //
    // The sizes asserted below (104/128/96/144) are the CUDA 12.x/13.x
    // sizes, not the 11.8 sizes just measured (40/64/32/144) -- see the
    // struct docs and the section comment above `CudaExternalMemoryHandleDescWin32`
    // for why: the loader prefers a 12.x/13.x runtime when present, and that
    // runtime reads `unsigned int reserved[16]` fields these three structs
    // don't have in 11.8's header. No 12.x/13.x driver_types.h exists on this
    // box to run the offsetof program against, so 104/128/96 are derived
    // arithmetically here: each struct's measured 11.8 layout, plus 64 bytes
    // for `reserved[16]`, plus padding to the next multiple of 8 (this
    // module's structs are all 8-aligned, from their `u64`/pointer fields):
    //   handle_desc: 36 (fields end) + 64 (reserved) = 100 -> pad to 104
    //   mip_desc:    64 (fields end) + 64 (reserved) = 128 -> already aligned
    //   sem_desc:    28 (fields end) + 64 (reserved) = 92  -> pad to 96
    // `cudaExternalSemaphoreWaitParams` already carries its reserved arrays
    // in 11.8, so 144 is both the 11.8 and the 12.x/13.x size.
    #[test]
    fn external_memory_and_semaphore_descs_match_driver_types_h() {
        assert_eq!(
            std::mem::size_of::<CudaExternalMemoryHandleDescWin32>(),
            104
        );
        // SAFETY: every field of `CudaExternalMemoryHandleDescWin32` is an
        // integer, raw pointer or fixed-size array of those -- all-zero is a
        // valid bit pattern for each, so a zeroed instance is well-defined;
        // this test only reads field offsets and the struct's size, never
        // dereferences the null `win32_handle`/`win32_name` pointers.
        let d: CudaExternalMemoryHandleDescWin32 = unsafe { std::mem::zeroed() };
        assert_eq!(off(&d, &d.win32_handle), 8);
        assert_eq!(off(&d, &d.size), 24);
        assert_eq!(off(&d, &d.flags), 32);
        assert_eq!(
            std::mem::size_of::<CudaExternalMemoryMipmappedArrayDesc>(),
            128
        );
        // SAFETY: same reasoning as `d` above -- every field is an integer,
        // nested POD struct or array of those, so all-zero is valid.
        let m: CudaExternalMemoryMipmappedArrayDesc = unsafe { std::mem::zeroed() };
        assert_eq!(off(&m, &m.format_desc), 8);
        assert_eq!(off(&m, &m.extent), 32);
        assert_eq!(off(&m, &m.flags), 56);
        assert_eq!(off(&m, &m.num_levels), 60);
        assert_eq!(std::mem::size_of::<CudaExternalSemaphoreHandleDesc>(), 96);
        // SAFETY: same reasoning as `d` above; the null `win32_handle`/
        // `win32_name` pointers are never dereferenced.
        let s: CudaExternalSemaphoreHandleDesc = unsafe { std::mem::zeroed() };
        assert_eq!(off(&s, &s.win32_handle), 8);
        assert_eq!(off(&s, &s.flags), 24);
        assert_eq!(std::mem::size_of::<CudaExternalSemaphoreWaitParams>(), 144);
        // SAFETY: same reasoning as `d` above -- every field is an integer
        // or fixed-size array of integers, so all-zero is valid.
        let w: CudaExternalSemaphoreWaitParams = unsafe { std::mem::zeroed() };
        assert_eq!(off(&w, &w.fence_value), 0);
        assert_eq!(off(&w, &w.flags), 72);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_table_loads_when_libcudart_present() {
        let avail = is_cuda_available();
        if avail {
            assert!(table().is_some(), "table present when available");
            let path = runtime_path().expect("runtime_path set whenever the table loaded");
            assert!(
                !path.as_os_str().is_empty(),
                "runtime_path must name the DLL/soname the loader opened"
            );
            // Printed so the report's "which DLL did the loader open" claim
            // is reproducible straight from this test's output.
            eprintln!("[cuda] runtime_path = {}", path.display());
        }
        // total + non-panicking either way
    }

    #[test]
    fn pub_primitives_degrade_without_libcudart() {
        // On hosts without libcudart (all CI coverage lanes), the public CUDA
        // primitives must degrade cleanly — None / false / no-op, never panic.
        // Skip on a CUDA host, where these would touch the real driver without
        // a current GL context.
        if is_cuda_available() {
            return;
        }
        assert!(gl_register_buffer(0).is_none());
        assert!(gl_map_resource(0).is_none());
        gl_unmap_resource(0); // no-op without a table — must not panic
        gl_unregister_resource(0); // no-op without a table — must not panic
                                   // SAFETY: with no libcudart, memcpy_device_to_host returns false before
                                   // touching the pointers, so null/zero args are sound here.
        assert!(!unsafe { memcpy_device_to_host(std::ptr::null_mut(), std::ptr::null(), 0) });
    }
}

#[cfg(test)]
mod handle_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct MockOps {
        unmaps: Arc<AtomicUsize>,
        unregisters: Arc<AtomicUsize>,
    }
    impl CudaGlOps for MockOps {
        fn map(&self, _r: GraphicsResource) -> Option<(*mut std::ffi::c_void, usize)> {
            Some((0x1000usize as *mut _, 4096))
        }
        fn unmap(&self, _r: GraphicsResource) {
            self.unmaps.fetch_add(1, Ordering::SeqCst);
        }
        fn unregister(&self, _r: GraphicsResource) {
            self.unregisters.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn cudamap_guard_unmaps_on_drop_for_glbuffer() {
        let unmaps = Arc::new(AtomicUsize::new(0));
        let unregisters = Arc::new(AtomicUsize::new(0));
        {
            let h = CudaHandle::new_gl(
                0x1usize as GraphicsResource,
                4096,
                Arc::new(MockOps {
                    unmaps: unmaps.clone(),
                    unregisters: unregisters.clone(),
                }),
            );
            {
                let m = h.map().expect("map");
                assert_eq!(m.device_ptr() as usize, 0x1000);
                assert_eq!(m.len(), 4096);
                assert!(!m.is_empty());
            }
            // CudaMap dropped → exactly one unmap; handle still alive → no unregister yet.
            assert_eq!(
                unmaps.load(Ordering::SeqCst),
                1,
                "Drop must unmap a GlBuffer"
            );
            assert_eq!(unregisters.load(Ordering::SeqCst), 0);
        }
        // CudaHandle dropped → exactly one unregister.
        assert_eq!(
            unregisters.load(Ordering::SeqCst),
            1,
            "Dropping a GlBuffer handle must unregister"
        );
    }

    /// A GlBuffer handle whose ops.map() fails yields None from CudaHandle::map.
    struct NoneOps;
    impl CudaGlOps for NoneOps {
        fn map(&self, _r: GraphicsResource) -> Option<(*mut std::ffi::c_void, usize)> {
            None
        }
        fn unmap(&self, _r: GraphicsResource) {}
        fn unregister(&self, _r: GraphicsResource) {}
    }
    #[test]
    fn glbuffer_map_returns_none_when_ops_map_fails() {
        let h = CudaHandle::new_gl(0x9usize as GraphicsResource, 4096, Arc::new(NoneOps));
        assert!(
            h.map().is_none(),
            "GlBuffer map propagates ops.map() failure"
        );
    }

    #[test]
    fn glbuffer_handle_debug_and_empty_map() {
        let unmaps = Arc::new(AtomicUsize::new(0));
        let unregisters = Arc::new(AtomicUsize::new(0));
        let h = CudaHandle::new_gl(
            0x2usize as GraphicsResource,
            0,
            Arc::new(MockOps {
                unmaps: unmaps.clone(),
                unregisters: unregisters.clone(),
            }),
        );
        let dbg = format!("{h:?}");
        assert!(
            dbg.contains("GlBuffer"),
            "debug names the backing kind: {dbg}"
        );
        assert!(dbg.contains("size"), "debug includes size: {dbg}");
    }

    #[test]
    fn external_mem_map_is_persistent_and_debug_names_kind() {
        // ExternalMem handle: map() returns the persistent device ptr directly
        // (no GL routing, unmap is a no-op). Construct with a synthetic ptr.
        let dptr = 0xCAFE_0000usize as *mut std::ffi::c_void;
        let h = CudaHandle::new_external(std::ptr::null_mut(), dptr, 8192);
        let dbg = format!("{h:?}");
        assert!(dbg.contains("ExternalMem"), "debug names the kind: {dbg}");
        {
            let m = h.map().expect("ExternalMem map is always Some");
            assert_eq!(m.device_ptr(), dptr, "persistent device ptr passthrough");
            assert_eq!(m.len(), 8192);
            assert!(!m.is_empty());
            // CudaMap drops here: ExternalMem mapping has unmap=None → no-op, safe.
        }
        // HOST-SAFETY: dropping `h` would call the real cudaDestroyExternalMemory
        // on this synthetic handle (libcudart is present on dev hosts). Forget it.
        std::mem::forget(h);
    }

    #[test]
    fn external_mem_zero_len_map_is_empty() {
        let h = CudaHandle::new_external(std::ptr::null_mut(), std::ptr::null_mut(), 0);
        {
            let m = h.map().expect("map");
            assert_eq!(m.len(), 0);
            assert!(m.is_empty(), "zero-length mapping is empty");
            assert!(m.device_ptr().is_null());
        }
        std::mem::forget(h); // HOST-SAFETY: avoid real cudaDestroyExternalMemory
    }
}
