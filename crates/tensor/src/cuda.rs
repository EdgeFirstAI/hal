// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional CUDA Runtime interop, loaded via dlopen (no link-time dependency).
//! Absent libcudart ⇒ every CUDA path degrades to `None`.
//!
//! Client pattern: call [`Tensor::cuda_map`](crate::Tensor::cuda_map) (or
//! [`TensorDyn::cuda_map`](crate::TensorDyn::cuda_map)) first; if it returns
//! `None` (no CUDA handle attached or libcudart absent), fall back to the host
//! mapping via `Tensor::map`.
use edgefirst_tensor_abi::{EfClientState, EfCudaMapFn, EfCudaUnmapFn, EfCudaUnregisterFn};
use libloading::Library;
#[cfg(target_os = "windows")]
use std::ffi::c_char;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
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
    // Owns the loaded module so it stays mapped while the function pointers
    // below are callable.
    _lib: Library,
    /// The runtime `load()` opened: a `CUDA_PATH\bin\...` candidate is
    /// recorded verbatim; a bare name (Windows) is resolved to the absolute
    /// path Windows mapped via `GetModuleHandleW`/`GetModuleFileNameW`,
    /// falling back to the bare name if that resolution fails. On Linux this
    /// is the soname that opened.
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
    // D3D11 texture interop, Windows-only.
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
    /// `cudaArrayGetPlane`, CUDA 11.2 and later: one plane of a multi-planar
    /// (NV12) array. `None` on an older runtime, which then declines native
    /// NV12 textures.
    #[cfg(target_os = "windows")]
    pub array_get_plane:
        Option<unsafe extern "C" fn(*mut CudaArray, CudaArray, c_uint) -> CudaError>,
    // The async spellings, which take the handle's own stream.
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

// Under Miri no runtime is loaded.
#[cfg(miri)]
fn load() -> Option<CudaTable> {
    None
}

#[cfg(all(not(miri), not(target_os = "windows")))]
fn load() -> Option<CudaTable> {
    for name in [
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ] {
        // SAFETY: `name` is one of the fixed soname literals above.
        let Ok(lib) = (unsafe { Library::new(name) }) else {
            continue;
        };
        // `load_table` takes ownership of `lib` and unloads it when a required
        // symbol is missing; the next candidate is then tried.
        if let Some(table) = load_table(lib, std::path::PathBuf::from(name)) {
            log::debug!("CUDA runtime loaded from {}", table.runtime_path.display());
            return Some(table);
        }
    }
    None
}

#[cfg(all(not(miri), target_os = "windows"))]
fn load() -> Option<CudaTable> {
    // Newest runtime first. Each name is also tried under CUDA_PATH\bin;
    // `is_full_path` marks those entries, whose `runtime_path` is the
    // candidate itself. A bare name is resolved with
    // `resolve_loaded_module_path`.
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
        let Ok(lib) = (unsafe { Library::new(candidate) }) else {
            continue;
        };
        let runtime_path = if *is_full_path {
            candidate.clone()
        } else {
            resolve_loaded_module_path(candidate.as_os_str()).unwrap_or_else(|| candidate.clone())
        };
        // `load_table` takes ownership of `lib` and unloads it when a required
        // symbol is missing; the next candidate is then tried.
        if let Some(table) = load_table(lib, runtime_path) {
            log::debug!("CUDA runtime loaded from {}", table.runtime_path.display());
            return Some(table);
        }
    }
    None
}

/// Resolves a bare DLL name (no directory) to the absolute path Windows
/// mapped, via `GetModuleHandleW` + `GetModuleFileNameW`. The module must be
/// loaded. Returns `None` on any resolution failure.
#[cfg(target_os = "windows")]
fn resolve_loaded_module_path(name: &std::ffi::OsStr) -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let wide: Vec<u16> = name.encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 buffer kept alive for the call.
    let handle = unsafe { GetModuleHandleW(PCWSTR(wide.as_ptr())) }.ok()?;
    let mut buf = [0u16; 1024];
    // SAFETY: `handle` is a loaded module's handle and `buf` is a correctly
    // sized mutable out-buffer.
    let len = unsafe { GetModuleFileNameW(Some(handle), &mut buf) } as usize;
    // `len == buf.len()` is truncation.
    (len > 0 && len < buf.len())
        .then(|| std::path::PathBuf::from(std::ffi::OsString::from_wide(&buf[..len])))
}

/// Resolves every interop symbol shared by both platforms' `load()`. Returns
/// `None` and logs the missing symbol's name if any required entry point is
/// absent from `lib`, including the D3D11 entries on Windows. Takes ownership
/// of `lib`: on success it is stored in the returned `CudaTable`; on failure
/// it is dropped, unloading the module.
#[cfg(not(miri))]
fn load_table(lib: Library, runtime_path: std::path::PathBuf) -> Option<CudaTable> {
    macro_rules! sym {
        ($n:literal) => {{
            // SAFETY: `$n` is one of the fixed `cuda*` symbol-name literals
            // below and `lib` is the module `load()` opened.
            match unsafe { lib.get(concat!($n, "\0").as_bytes()) } {
                Ok(s) => *s,
                Err(_) => {
                    log::debug!("CUDA runtime missing required symbol {}", $n);
                    return None;
                }
            }
        }};
    }
    // Resolves the `_v2` export and falls back to the unsuffixed name.
    #[cfg(target_os = "windows")]
    macro_rules! sym_versioned {
        ($v2:literal, $v1:literal) => {{
            // SAFETY: `$v2`/`$v1` are fixed symbol-name literals; `lib` is
            // the module `load()` opened.
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
        // SAFETY: a fixed symbol-name literal looked up in the module `load()`
        // just opened, read as the function type its CUDA declaration has.
        array_get_plane: unsafe { lib.get(b"cudaArrayGetPlane\0") }.ok().map(|s| *s),
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
        _lib: lib,
    })
}

/// `cudaGetErrorString(rc)` with the numeric code kept alongside it. Falls
/// back to the bare code if the runtime hands back no string.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
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

/// True iff libcudart loaded and all interop symbols resolved. Cached.
pub fn is_cuda_available() -> bool {
    table().is_some()
}

/// The runtime `load()` opened, if any; see `CudaTable::runtime_path`.
/// `None` iff no runtime loaded.
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
/// [`stream_destroy`].
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
/// (pointer) or `None`. Must be called on the thread where the GL context is
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
// =============================================================================

/// `cudaExternalMemoryHandleTypeOpaqueFd`: the handle type used for Linux
/// DMA-BUF fds.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
#[repr(C)]
pub(crate) struct CudaExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub _tail: u32,
}

/// Import a DMA-BUF fd as CUDA external memory and map it to a device pointer.
///
/// No GL context is required. This function dups the caller's `fd` and hands
/// CUDA the dup; the caller's `fd` remains owned by the caller. Returns
/// `(ext_mem_handle, device_ptr)` on success, or `None` on any failure
/// (missing libcudart, unsupported platform, dup failure, or driver error).
/// The returned handle must be destroyed via `cudaDestroyExternalMemory`
/// (done by [`CudaHandle`] drop), which also closes the dup'd fd.
#[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic)))]
pub(crate) fn import_dma_fd(fd: i32, size: usize) -> Option<(ExternalMemory, *mut c_void)> {
    use std::os::fd::{AsRawFd, BorrowedFd, IntoRawFd};
    let t = table()?;
    // Hands CUDA a dup of `fd`, kept as `OwnedFd` until the import returns.
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
        // Closes the dup through libc; a dup the driver closed is tolerated.
        let raw = dup.into_raw_fd();
        if unsafe { libc::close(raw) } != 0 {
            log::debug!(
                "reclaim of CUDA-import dup fd {raw} after failed import: {}",
                std::io::Error::last_os_error()
            );
        }
        return None;
    }
    // CUDA owns the dup; it is closed by cudaDestroyExternalMemory.
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
// D3D11 texture → CUDA external memory import (Windows only), used by
// `import_d3d11_texture`, the fence wait, and the array copies.
//
// The structs below, except `cudaExternalSemaphoreWaitParams`, are laid out
// as CUDA 12.x/13.x define them.
// =============================================================================

/// `cudaExternalMemoryHandleTypeD3D11Resource`.
#[allow(dead_code)]
#[cfg(target_os = "windows")]
pub(crate) const CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE: c_int = 5;
/// `cudaExternalSemaphoreHandleTypeD3D11Fence`.
#[allow(dead_code)]
#[cfg(target_os = "windows")]
pub(crate) const CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE: c_int = 4;
/// `cudaExternalMemoryDedicated`: the imported resource is a dedicated
/// allocation.
#[allow(dead_code)]
pub(crate) const CUDA_EXTERNAL_MEMORY_DEDICATED: c_uint = 1;
/// `cudaMemcpyDeviceToDevice` — both endpoints of the copy are CUDA device
/// memory (the imported array and the tensor's linear device buffer).
#[allow(dead_code)]
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
/// `cudaChannelFormatKindUnsigned` — integer texture formats (RGBA8, R8).
#[allow(dead_code)]
pub(crate) const CUDA_CHANNEL_FORMAT_KIND_UNSIGNED: c_int = 1;
/// `cudaChannelFormatKindFloat` — float texture formats (F16, F32).
#[allow(dead_code)]
pub(crate) const CUDA_CHANNEL_FORMAT_KIND_FLOAT: c_int = 2;
/// `cudaChannelFormatKindNV12`: an 8-bit 4:2:0 array of two planes.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
pub(crate) const CUDA_CHANNEL_FORMAT_KIND_NV12: c_int = 4;

/// `cudaExternalMemoryHandleDesc` with the `win32` arm of the handle union
/// populated (`type_`, `win32_handle`/`win32_name`, `size`, `flags`), laid
/// out as CUDA 12.x/13.x's `driver_types.h` defines it.
///
/// Field offsets: `type_` @0, `win32_handle` @8, `win32_name` @16, `size`
/// @24, `flags` @32, `reserved[16]` @36..100. Total size 104, align 8.
#[allow(dead_code)]
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

/// `cudaChannelFormatDesc` (driver_types.h; size 20, align 4).
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaChannelFormatDesc {
    pub x: c_int,
    pub y: c_int,
    pub z: c_int,
    pub w: c_int,
    pub f: c_int,
}

/// `cudaExtent` (driver_types.h; size 24, align 8).
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

/// `cudaExternalMemoryMipmappedArrayDesc`, laid out as CUDA 12.x/13.x's
/// `driver_types.h` defines it.
///
/// Field offsets: `offset` @0, `format_desc` @8, `extent` @32, `flags` @56,
/// `num_levels` @60, `reserved[16]` @64..128. Total size 128, align 8.
#[allow(dead_code)]
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
/// union populated, laid out as CUDA 12.x/13.x's `driver_types.h` defines it.
///
/// Field offsets: `type_` @0, `win32_handle` @8, `win32_name` @16, `flags`
/// @24, `reserved[16]` @28..92. Total size 96, align 8.
#[allow(dead_code)]
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
/// Layout (size 144, align 8): `params.fence.value` @0, `params.nvSciSync`
/// @8, `params.keyedMutex.key` @16, `params.keyedMutex.timeoutMs` @24,
/// `params.reserved[10]` @32, `flags` @72, `reserved[16]` @76, tail padding
/// to 144.
#[allow(dead_code)]
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

#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
/// One CUDA array the linear buffer is copied from and back to: the level-0
/// array of a single-plane texture, or one plane of a native NV12 one.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
#[derive(Clone, Copy)]
struct ArrayPlane {
    array: CudaArray,
    /// Bytes of one row of the plane, which is also its row in the linear
    /// buffer.
    row_bytes: usize,
    rows: usize,
    /// Where the plane's rows start in the linear buffer.
    offset: usize,
}

/// The arrays of an imported texture in the order its linear buffer lays
/// them out: one plane, or NV12's luma then its interleaved chroma, which is
/// the combined-plane layout a CPU map of the texture has.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
#[derive(Clone, Copy)]
struct ArrayPlanes {
    planes: [ArrayPlane; 2],
    count: usize,
}

#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
impl ArrayPlanes {
    fn iter(&self) -> impl Iterator<Item = &ArrayPlane> {
        self.planes[..self.count].iter()
    }
}

/// The CUDA side of one D3D11 texture tensor: the imported external memory,
/// the arrays it maps to, and the linear device buffer every map copies
/// through.
///
/// A map copies the arrays into the linear buffer as the tensor's tight rows,
/// and a writable map's release copies them back. The linear buffer is
/// allocated on the first map and then reused.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
struct D3d11External {
    ext_mem: ExternalMemory,
    mipmapped: MipmappedArray,
    planes: ArrayPlanes,
    /// The linear buffer, allocated on first use, behind a lock that
    /// serialises concurrent maps.
    linear: std::sync::Mutex<Option<*mut c_void>>,
    bytes: usize,
    stream: CudaStream,
    /// The CUDA device the D3D11 adapter maps to. Set again before every copy
    /// and before the release.
    device: c_int,
    /// The process fence imported as an external semaphore, shared by every
    /// D3D11 import in this copy of the crate (see [`shared_semaphore`]).
    ///
    /// `None` when the device has no shared fence or the import failed. A map
    /// of a tensor with no recorded write then copies without waiting; a map
    /// of a tensor that carries a recorded write declines.
    semaphore: Option<&'static SharedSemaphore>,
    /// The newest D3D11 fence value recorded on the tensor, shared with the
    /// storage's own slot: a map waits for it before reading the array.
    last_write: Arc<std::sync::atomic::AtomicU64>,
}

/// Imports a D3D11 texture's shared NT handle as CUDA external memory, maps
/// its level-0 array and imports `fence_handle` as an external semaphore so
/// maps can order their copies after the D3D11 work a producer signalled.
///
/// `handle` stays owned by the caller. `fence_handle` is a duplicate this
/// backing takes over.
///
/// Returns `None` on any failure (no CUDA device for the adapter, as on WARP;
/// a DXGI format with no channel description; a driver error), releasing
/// everything it had already created.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
pub(crate) fn import_d3d11_texture(
    handle: RawHandle,
    layout: crate::d3d11_layout::D3d11ImageLayout,
    adapter: *mut c_void,
    fence_handle: Option<OwnedHandle>,
    last_write: Arc<std::sync::atomic::AtomicU64>,
) -> Option<CudaHandle> {
    let t = table()?;
    // The CUDA device ordinal for the process D3D11 device, resolved once and
    // cached.
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
    // width in bits and how many components a texel carries. A format
    // `crate::d3d11_layout` does not allocate declines the import.
    let (bits, kind, channels) = match layout.dxgi_format {
        // An NV12 array is described as three 8-bit channels of kind NV12;
        // its planes are split out below.
        crate::d3d11_layout::DXGI_FORMAT_NV12 if t.array_get_plane.is_some() => {
            (8, CUDA_CHANNEL_FORMAT_KIND_NV12, 3)
        }
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
    // those, so all-zero is a valid bit pattern.
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
    let Some(planes) = array_planes(t, level0, layout) else {
        // SAFETY: as in the failure arm above: created here, held by nothing
        // else, the stream first and the array before its memory.
        unsafe {
            (t.stream_destroy)(stream);
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
            planes,
            linear: std::sync::Mutex::new(None),
            bytes: layout.tight_bytes(),
            stream,
            device,
            semaphore,
            last_write,
        }),
    })
}

/// The arrays the linear buffer of `layout`'s texture is copied through:
/// `level0` itself, or for a native NV12 texture its luma plane (one 8-bit
/// channel, the full grid) and its chroma plane (two 8-bit channels at half
/// the width and height, so as many bytes per row as luma).
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
fn array_planes(
    t: &CudaTable,
    level0: CudaArray,
    layout: crate::d3d11_layout::D3d11ImageLayout,
) -> Option<ArrayPlanes> {
    let row_bytes = layout.tight_row_bytes();
    if !layout.is_native_nv12() {
        let only = ArrayPlane {
            array: level0,
            row_bytes,
            rows: layout.texture_height,
            offset: 0,
        };
        return Some(ArrayPlanes {
            planes: [only; 2],
            count: 1,
        });
    }
    let get_plane = t.array_get_plane?;
    let mut planes = [std::ptr::null_mut(); 2];
    for (i, plane) in planes.iter_mut().enumerate() {
        // SAFETY: `level0` is the live NV12 array just mapped, which has two
        // planes, and `plane` is a valid out-parameter.
        let rc = unsafe { get_plane(plane, level0, i as c_uint) };
        if rc != 0 || plane.is_null() {
            log::debug!(
                "cudaArrayGetPlane({i}) of an NV12 texture: {}",
                err_str(t, rc)
            );
            return None;
        }
    }
    let luma_rows = layout.texture_height;
    Some(ArrayPlanes {
        planes: [
            ArrayPlane {
                array: planes[0],
                row_bytes,
                rows: luma_rows,
                offset: 0,
            },
            ArrayPlane {
                array: planes[1],
                row_bytes,
                rows: luma_rows / 2,
                offset: row_bytes * luma_rows,
            },
        ],
        count: 2,
    })
}

/// The process fence imported as a CUDA external semaphore, with the
/// duplicate of its NT handle CUDA needs kept open beside it.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
struct SharedSemaphore {
    sem: ExternalSemaphore,
    /// The source handle of the import, held for the process lifetime.
    _handle: OwnedHandle,
}

// SAFETY: a CUDA external-semaphore handle is bound to a device, not to a
// thread, and is only passed to thread-safe runtime calls. `OwnedHandle` is
// `Send + Sync`. The struct is written once inside a `OnceLock` initialiser.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
unsafe impl Send for SharedSemaphore {}
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
unsafe impl Sync for SharedSemaphore {}

/// Imports the process D3D11 fence as an external semaphore, once. The
/// semaphore and its handle live for the process.
///
/// `fence_handle` is consumed by the first caller and dropped by the rest.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
fn shared_semaphore(t: &CudaTable, fence_handle: OwnedHandle) -> Option<&'static SharedSemaphore> {
    static IMPORTED: OnceLock<Option<SharedSemaphore>> = OnceLock::new();
    IMPORTED
        .get_or_init(move || {
            // SAFETY: all-zero is a valid bit pattern for this descriptor.
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
                // The duplicate closes with `fence_handle` here.
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
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
fn d3d11_copy_is_ordered(recorded: u64, wait_queued: bool) -> bool {
    recorded == 0 || wait_queued
}

/// Set once a D3D11 allocation has found that this adapter's driver pads its
/// textures past their padding-free size; later imports then skip the tight
/// rung of [`import_and_map_array`]'s ladder.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
static CUDA_IMPORT_SIZE_PADDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Imports the texture as external memory and maps `md` out of it, returning
/// the pair on success.
///
/// The declared size climbs a ladder (the exact bytes, then 2x, then 4x) and
/// stops at the first rung the driver maps. The first rung is floored at one
/// 4 KiB page. The first allocation that has to climb records it in
/// [`CUDA_IMPORT_SIZE_PADDED`] and later ones start one rung up.
#[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
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
    #[allow(dead_code)]
    GlBuffer {
        resource: GraphicsResource,
        ops: Arc<dyn CudaGlOps>,
    },
    #[allow(dead_code)]
    ExternalMem {
        ext_mem: ExternalMemory,
        dptr: *mut c_void,
    },
    #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
    D3d11(D3d11External),
    /// State after [`CudaHandle::into_client_parts`] moved a GL registration
    /// across the ABI; Drop does not unregister.
    Detached,
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
            CudaBacking::Detached => "Detached",
            #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
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

    /// Decompose a GL-buffer registration into the callback channel
    /// [`ef_tensor_cuda_attach`](crate::TensorDyn::set_cuda_handle) takes.
    ///
    /// Consumes `self` without unregistering: the library tensor's
    /// `CudaHandle` becomes the owner that will unregister on drop.
    pub fn into_client_parts(self) -> crate::Result<CudaClientParts> {
        let mut this = self;
        let kind = std::mem::replace(&mut this.kind, CudaBacking::Detached);
        let size = this.size;
        match kind {
            CudaBacking::GlBuffer { resource, ops } => {
                let channel = Arc::new(CudaGlChannel { ops });
                let state = EfClientState {
                    ctx: Arc::as_ptr(&channel) as *const c_void,
                    retain: Some(cuda_vt_retain),
                    release: Some(cuda_vt_release),
                };
                Ok(CudaClientParts {
                    state,
                    resource,
                    size,
                    map_fn: cuda_vt_map,
                    unmap_fn: cuda_vt_unmap,
                    unregister_fn: cuda_vt_unregister,
                    _channel: channel,
                })
            }
            other => {
                this.kind = other;
                drop(this);
                Err(crate::Error::InvalidArgument(
                    "cuda attach is for GL-buffer registrations only".into(),
                ))
            }
        }
    }

    #[allow(dead_code)]
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
    /// The `D3d11` mapping is the texture's tight rows: `len()` is
    /// `width * bytes_per_texel * height` and row `n` starts at
    /// `n * width * bytes_per_texel`, not at the tensor's
    /// [`row_stride`](crate::Tensor::row_stride), which is the D3D11 staging
    /// pitch a CPU map sees.
    ///
    /// For semi-planar textures (NV12, NV16, NV24) the texture is as wide as
    /// its pitch, so the mapping equals `row_stride() x combined_height`
    /// bytes.
    ///
    /// One `D3d11` handle has one linear buffer: two live maps of the same
    /// tensor hand out the same device pointer and the second map's copy
    /// refreshes what the first is reading. Callers serialise a `map()` with
    /// a live [`map_mut`](Self::map_mut) guard.
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
            #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
            CudaBacking::D3d11(x) => Some(CudaMap {
                ptr: self.d3d11_copy_in(x)?,
                len: x.bytes,
                release: Release::None,
                _marker: std::marker::PhantomData,
            }),
            CudaBacking::Detached => None,
        }
    }

    /// Map to a device pointer for writing.
    ///
    /// For the GL and external-memory backings this is [`map`](Self::map).
    /// For a `D3d11` texture the map's release copies the linear buffer back
    /// into the texture and synchronises.
    ///
    /// The `D3d11` mapping layout is the one [`map`](Self::map) describes.
    pub fn map_mut(&self) -> Option<CudaMap<'_>> {
        #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
        if let CudaBacking::D3d11(x) = &self.kind {
            return Some(CudaMap {
                ptr: self.d3d11_copy_in(x)?,
                len: x.bytes,
                release: Release::D3d11Writeback {
                    planes: x.planes,
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
    /// own stream. A recorded fence value the producer never signals blocks
    /// the synchronisation.
    ///
    /// Answers `None` when the wait cannot be issued
    /// ([`d3d11_copy_is_ordered`]) and on every other failure.
    #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
    fn d3d11_copy_in(&self, x: &D3d11External) -> Option<*mut c_void> {
        let t = table()?;
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
        // One device copy per plane, on the same stream as the wait above.
        for plane in x.planes.iter() {
            // SAFETY: `dptr` holds `x.bytes` bytes, which covers every plane's
            // `offset + rows * row_bytes`; `plane.array` is a live array of the
            // imported texture, `row_bytes` wide and `rows` tall; and
            // `x.stream` is this backing's live stream.
            let rc = unsafe {
                (t.memcpy_2d_from_array_async)(
                    dptr.cast::<u8>().add(plane.offset).cast(),
                    plane.row_bytes,
                    plane.array,
                    0,
                    0,
                    plane.row_bytes,
                    plane.rows,
                    CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    x.stream,
                )
            };
            if rc != 0 {
                log::warn!("cudaMemcpy2DFromArrayAsync: {}", err_str(t, rc));
                return None;
            }
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
            CudaBacking::Detached => {}
            CudaBacking::GlBuffer { resource, ops } => ops.unregister(*resource),
            CudaBacking::ExternalMem { ext_mem, dptr: _ } => {
                // Destroys the external-memory object, which frees the mapped device
                // pointer.
                if let Some(t) = table() {
                    unsafe {
                        (t.destroy_external_memory)(*ext_mem);
                    }
                }
            }
            // Releases the linear buffer, the stream, the array and the external
            // memory, in that order.
            #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
            CudaBacking::D3d11(x) => {
                let Some(t) = table() else { return };
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

struct CudaGlChannel {
    ops: Arc<dyn CudaGlOps>,
}

/// The frozen callback channel a GL-buffer CUDA registration becomes when
/// it crosses `ef_tensor_cuda_attach`.
pub struct CudaClientParts {
    pub state: EfClientState,
    pub resource: GraphicsResource,
    pub size: usize,
    pub map_fn: EfCudaMapFn,
    pub unmap_fn: EfCudaUnmapFn,
    pub unregister_fn: EfCudaUnregisterFn,
    _channel: Arc<CudaGlChannel>,
}

unsafe extern "C" fn cuda_vt_retain(ctx: *const c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        Arc::increment_strong_count(ctx as *const CudaGlChannel);
    }));
}

unsafe extern "C" fn cuda_vt_release(ctx: *const c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        Arc::decrement_strong_count(ctx as *const CudaGlChannel);
    }));
}

unsafe extern "C" fn cuda_vt_map(
    ctx: *const c_void,
    resource: *mut c_void,
    out_ptr: *mut *mut c_void,
    out_len: *mut usize,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let channel = unsafe { &*(ctx as *const CudaGlChannel) };
        match channel.ops.map(resource) {
            Some((ptr, len)) => {
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                0
            }
            None => 1,
        }
    }))
    .unwrap_or(1)
}

unsafe extern "C" fn cuda_vt_unmap(ctx: *const c_void, resource: *mut c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let channel = unsafe { &*(ctx as *const CudaGlChannel) };
        channel.ops.unmap(resource);
    }));
}

unsafe extern "C" fn cuda_vt_unregister(ctx: *const c_void, resource: *mut c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let channel = unsafe { &*(ctx as *const CudaGlChannel) };
        channel.ops.unregister(resource);
    }));
}

struct ImportedCudaGlOps {
    state: EfClientState,
    map_fn: EfCudaMapFn,
    unmap_fn: EfCudaUnmapFn,
    unregister_fn: EfCudaUnregisterFn,
}

unsafe impl Send for ImportedCudaGlOps {}
unsafe impl Sync for ImportedCudaGlOps {}

impl Drop for ImportedCudaGlOps {
    fn drop(&mut self) {
        if let Some(release) = self.state.release {
            unsafe { release(self.state.ctx) };
        }
    }
}

impl CudaGlOps for ImportedCudaGlOps {
    fn map(&self, resource: GraphicsResource) -> Option<(*mut c_void, usize)> {
        let mut ptr = std::ptr::null_mut();
        let mut len = 0usize;
        let rc = unsafe { (self.map_fn)(self.state.ctx, resource, &mut ptr, &mut len) };
        (rc == 0).then_some((ptr, len))
    }

    fn unmap(&self, resource: GraphicsResource) {
        unsafe { (self.unmap_fn)(self.state.ctx, resource) };
    }

    fn unregister(&self, resource: GraphicsResource) {
        unsafe { (self.unregister_fn)(self.state.ctx, resource) };
    }
}

/// Build `CudaGlOps` from a client's frozen `EfClientState` and the three
/// CUDA op functions. Retains the channel; [`Drop`] of the returned `Arc`
/// releases it.
///
/// # Safety
/// `state.ctx` must be the context the supplied function pointers understand.
pub unsafe fn client_state_cuda_ops(
    state: EfClientState,
    map_fn: EfCudaMapFn,
    unmap_fn: EfCudaUnmapFn,
    unregister_fn: EfCudaUnregisterFn,
) -> crate::Result<Arc<dyn CudaGlOps>> {
    if state.ctx.is_null() || state.retain.is_none() || state.release.is_none() {
        return Err(crate::Error::InvalidArgument(
            "CUDA client state must carry a non-NULL ctx, retain and release".into(),
        ));
    }
    let retain = state.retain.expect("checked above");
    unsafe { retain(state.ctx) };
    Ok(Arc::new(ImportedCudaGlOps {
        state,
        map_fn,
        unmap_fn,
        unregister_fn,
    }))
}

/// What a [`CudaMap`] does when it drops: nothing for a persistent mapping,
/// a GL unmap, a write-back into the D3D11 texture's array, or -- on the
/// `dynamic` backend -- the C library's own unmap of a mapping it handed out.
enum Release {
    None,
    GlUnmap(Arc<dyn CudaGlOps>, GraphicsResource),
    #[cfg(edgefirst_tensor_dynamic)]
    FfiUnmap(*mut c_void),
    #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
    D3d11Writeback {
        planes: ArrayPlanes,
        stream: CudaStream,
        device: c_int,
    },
}

/// Scoped CUDA device-pointer mapping. `Drop` unmaps a `GlBuffer` and copies a
/// writable `D3d11` mapping back into the texture. `ExternalMem` mappings are
/// persistent: `Drop` is a no-op.
///
/// A `D3d11` mapping is the texture's tight rows: [`len`](Self::len) is
/// `width * bytes_per_texel * height` and rows are `width * bytes_per_texel`
/// apart, not the tensor's `row_stride()`. A semi-planar texture is as wide
/// as its own pitch, so there the mapping is
/// `row_stride() x combined_height` bytes.
pub struct CudaMap<'a> {
    ptr: *mut c_void,
    len: usize,
    release: Release,
    _marker: std::marker::PhantomData<&'a ()>,
}

// SAFETY: the mapped device pointer is process-global and valid cross-thread
// via the per-device CUDA primary context; the routed CudaGlOps is Send+Sync.
// A `Release::FfiUnmap` handle owns a retain on a refcounted `ef_tensor`, whose
// retain and free are atomic.
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
    /// The C map holds its own retain on the tensor handle; dropping this
    /// guard releases both.
    ///
    /// Returns `None` (having released `raw`) when the map carries no device
    /// pointer.
    ///
    /// # Safety
    /// `raw` must be a live, non-NULL map from `ef_tensor_cuda_map` or
    /// `ef_tensor_cuda_map_mut` that no one else will release: ownership
    /// passes to the guard returned here, which calls `ef_tensor_cuda_unmap`
    /// on it exactly once.
    #[cfg(edgefirst_tensor_dynamic)]
    pub(crate) unsafe fn from_ffi(raw: *mut c_void) -> Option<Self> {
        let mut len = 0usize;
        // SAFETY: `raw` is a live map by this function's contract, and `len`
        // is a live local; the accessor takes either kind of map.
        let ptr = unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_device_ptr(raw, &mut len) };
        if ptr.is_null() {
            // SAFETY: as above; the map is released exactly once.
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
        // Takes the release so it runs exactly once.
        match std::mem::replace(&mut self.release, Release::None) {
            Release::None => {}
            Release::GlUnmap(ops, r) => ops.unmap(r),
            // The write-back of a writable D3D11 map happens inside the C library.
            #[cfg(edgefirst_tensor_dynamic)]
            Release::FfiUnmap(h) => {
                // SAFETY: the handle came from `ef_tensor_cuda_map`/`_map_mut`
                // and is released exactly once -- `replace` above took it.
                unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_unmap(h) }
            }
            #[cfg(all(target_os = "windows", not(edgefirst_tensor_dynamic)))]
            Release::D3d11Writeback {
                planes,
                stream,
                device,
            } => {
                let Some(t) = table() else { return };
                // SAFETY: `device` is the ordinal `import_d3d11_texture`
                // cached for that adapter.
                let rc = unsafe { (t.set_device)(device) };
                if rc != 0 {
                    log::warn!(
                        "cudaSetDevice({device}) before the D3D11 write-back: {}",
                        err_str(t, rc)
                    );
                    return;
                }
                for plane in planes.iter() {
                    // SAFETY: `self.ptr` is the linear buffer this map was
                    // built over, which holds every plane's rows at its
                    // `offset`; `plane.array` is a live array of the same
                    // geometry; and `stream` is the backing's live stream,
                    // which outlives this map.
                    let rc = unsafe {
                        (t.memcpy_2d_to_array_async)(
                            plane.array,
                            0,
                            0,
                            self.ptr.cast::<u8>().add(plane.offset).cast_const().cast(),
                            plane.row_bytes,
                            plane.row_bytes,
                            plane.rows,
                            CUDA_MEMCPY_DEVICE_TO_DEVICE,
                            stream,
                        )
                    };
                    if rc != 0 {
                        log::warn!("cudaMemcpy2DToArrayAsync: {}", err_str(t, rc));
                    }
                }
                // Synchronises the asynchronous copy above.
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
        // size field at offset 24, flags at 32
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

#[cfg(all(test, target_os = "windows", not(edgefirst_tensor_dynamic)))]
mod d3d11_ordering {
    use super::d3d11_copy_is_ordered;

    /// A recorded write with no wait behind it declines the copy.
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

    /// The D3D11 descriptor structs have the CUDA 12.x/13.x field offsets and
    /// sizes.
    #[test]
    fn external_memory_and_semaphore_descs_match_driver_types_h() {
        assert_eq!(
            std::mem::size_of::<CudaExternalMemoryHandleDescWin32>(),
            104
        );
        // SAFETY: every field is an integer, raw pointer or fixed-size array of
        // those, so all-zero is a valid bit pattern.
        let d: CudaExternalMemoryHandleDescWin32 = unsafe { std::mem::zeroed() };
        assert_eq!(off(&d, &d.win32_handle), 8);
        assert_eq!(off(&d, &d.size), 24);
        assert_eq!(off(&d, &d.flags), 32);
        assert_eq!(
            std::mem::size_of::<CudaExternalMemoryMipmappedArrayDesc>(),
            128
        );
        // SAFETY: every field is an integer, nested POD struct or array of those,
        // so all-zero is valid.
        let m: CudaExternalMemoryMipmappedArrayDesc = unsafe { std::mem::zeroed() };
        assert_eq!(off(&m, &m.format_desc), 8);
        assert_eq!(off(&m, &m.extent), 32);
        assert_eq!(off(&m, &m.flags), 56);
        assert_eq!(off(&m, &m.num_levels), 60);
        assert_eq!(std::mem::size_of::<CudaExternalSemaphoreHandleDesc>(), 96);
        // SAFETY: every field is an integer, raw pointer or array of integers, so
        // all-zero is valid.
        let s: CudaExternalSemaphoreHandleDesc = unsafe { std::mem::zeroed() };
        assert_eq!(off(&s, &s.win32_handle), 8);
        assert_eq!(off(&s, &s.flags), 24);
        assert_eq!(std::mem::size_of::<CudaExternalSemaphoreWaitParams>(), 144);
        // SAFETY: every field is an integer or fixed-size array of integers, so
        // all-zero is valid.
        let w: CudaExternalSemaphoreWaitParams = unsafe { std::mem::zeroed() };
        assert_eq!(off(&w, &w.fence_value), 0);
        assert_eq!(off(&w, &w.flags), 72);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pub_primitives_degrade_without_libcudart() {
        if is_cuda_available() {
            crate::test_support::report_skip(
                "pub_primitives_degrade_without_libcudart - a CUDA runtime is loaded",
            );
            return;
        }
        assert!(table().is_none());
        assert!(runtime_path().is_none());
        assert!(stream_create().is_none());
        assert!(gl_register_buffer(0).is_none());
        assert!(gl_map_resource(0).is_none());
        gl_unmap_resource(0); // no-op without a table
        gl_unregister_resource(0); // no-op without a table
                                   // SAFETY: with no libcudart, memcpy_device_to_host returns false before
                                   // touching the pointers.
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

    /// A writable mapping of a GlBuffer maps and unmaps like a read one.
    #[test]
    fn map_mut_of_a_glbuffer_maps_and_unmaps_like_map() {
        let unmaps = Arc::new(AtomicUsize::new(0));
        let h = CudaHandle::new_gl(
            0x3usize as GraphicsResource,
            4096,
            Arc::new(MockOps {
                unmaps: unmaps.clone(),
                unregisters: Arc::new(AtomicUsize::new(0)),
            }),
        );
        {
            let m = h.map_mut().expect("map_mut");
            assert_eq!(m.device_ptr() as usize, 0x1000);
            assert_eq!(m.len(), 4096);
        }
        assert_eq!(unmaps.load(Ordering::SeqCst), 1);
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
        // ExternalMem handle: map() returns the persistent device ptr directly.
        let dptr = 0xCAFE_0000usize as *mut std::ffi::c_void;
        let h = CudaHandle::new_external(std::ptr::null_mut(), dptr, 8192);
        let dbg = format!("{h:?}");
        assert!(dbg.contains("ExternalMem"), "debug names the kind: {dbg}");
        {
            let m = h.map().expect("ExternalMem map is always Some");
            assert_eq!(m.device_ptr(), dptr, "persistent device ptr passthrough");
            assert_eq!(m.len(), 8192);
            assert!(!m.is_empty());
            // CudaMap drops here; an ExternalMem mapping has no unmap.
        }
        // Forgets `h`, whose drop would call cudaDestroyExternalMemory on the
        // synthetic handle.
        std::mem::forget(h);
    }

    /// Calls a client's channel received, keyed by the `ctx` it was given.
    #[derive(Default)]
    struct ClientCalls {
        retains: AtomicUsize,
        releases: AtomicUsize,
        maps: AtomicUsize,
        unmaps: AtomicUsize,
        unregisters: AtomicUsize,
    }

    fn calls<'a>(ctx: *const std::ffi::c_void) -> &'a ClientCalls {
        // SAFETY: every `ctx` below is a `&ClientCalls` that outlives the ops.
        unsafe { &*(ctx as *const ClientCalls) }
    }

    unsafe extern "C" fn fake_retain(ctx: *const std::ffi::c_void) {
        calls(ctx).retains.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn fake_release(ctx: *const std::ffi::c_void) {
        calls(ctx).releases.fetch_add(1, Ordering::SeqCst);
    }

    /// Maps resource `1` to 64 bytes at `0x1000`; fails for any other.
    unsafe extern "C" fn fake_map(
        ctx: *const std::ffi::c_void,
        resource: *mut std::ffi::c_void,
        out_ptr: *mut *mut std::ffi::c_void,
        out_len: *mut usize,
    ) -> std::ffi::c_int {
        calls(ctx).maps.fetch_add(1, Ordering::SeqCst);
        if resource as usize != 1 {
            return 7;
        }
        // SAFETY: the out-params are valid for this call.
        unsafe {
            *out_ptr = 0x1000usize as *mut std::ffi::c_void;
            *out_len = 64;
        }
        0
    }

    unsafe extern "C" fn fake_unmap(ctx: *const std::ffi::c_void, _r: *mut std::ffi::c_void) {
        calls(ctx).unmaps.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn fake_unregister(ctx: *const std::ffi::c_void, _r: *mut std::ffi::c_void) {
        calls(ctx).unregisters.fetch_add(1, Ordering::SeqCst);
    }

    /// A client channel's ops forward each call through its function pointers
    /// and hold one reference on the channel.
    #[test]
    fn client_channel_ops_forward_through_the_clients_functions() {
        let log = ClientCalls::default();
        let state = EfClientState {
            ctx: &log as *const ClientCalls as *const std::ffi::c_void,
            retain: Some(fake_retain),
            release: Some(fake_release),
        };
        // SAFETY: `ctx` is a live `ClientCalls` the fake functions understand.
        let ops = unsafe { client_state_cuda_ops(state, fake_map, fake_unmap, fake_unregister) }
            .expect("a complete client state");
        assert_eq!(log.retains.load(Ordering::SeqCst), 1);

        assert_eq!(
            ops.map(1usize as GraphicsResource),
            Some((0x1000usize as *mut std::ffi::c_void, 64))
        );
        assert_eq!(ops.map(2usize as GraphicsResource), None);
        assert_eq!(log.maps.load(Ordering::SeqCst), 2);

        ops.unmap(1usize as GraphicsResource);
        ops.unregister(1usize as GraphicsResource);
        assert_eq!(log.unmaps.load(Ordering::SeqCst), 1);
        assert_eq!(log.unregisters.load(Ordering::SeqCst), 1);

        assert_eq!(log.releases.load(Ordering::SeqCst), 0);
        drop(ops);
        assert_eq!(log.releases.load(Ordering::SeqCst), 1);
        assert_eq!(log.retains.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_client_state_missing_a_part_is_refused_without_a_retain() {
        let log = ClientCalls::default();
        let ctx = &log as *const ClientCalls as *const std::ffi::c_void;
        let incomplete = [
            EfClientState {
                ctx: std::ptr::null(),
                retain: Some(fake_retain),
                release: Some(fake_release),
            },
            EfClientState {
                ctx,
                retain: None,
                release: Some(fake_release),
            },
            EfClientState {
                ctx,
                retain: Some(fake_retain),
                release: None,
            },
        ];
        for state in incomplete {
            // SAFETY: refused before any function pointer is called.
            let r = unsafe { client_state_cuda_ops(state, fake_map, fake_unmap, fake_unregister) };
            assert!(matches!(r, Err(crate::Error::InvalidArgument(_))));
        }
        assert_eq!(log.retains.load(Ordering::SeqCst), 0);
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
        std::mem::forget(h); // skips cudaDestroyExternalMemory
    }
}
