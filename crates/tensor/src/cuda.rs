// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional CUDA Runtime interop, loaded via dlopen (no link-time dependency).
//! Absent libcudart ⇒ every CUDA path degrades to None.
use libloading::Library;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
use std::sync::{Arc, OnceLock};

pub(crate) type CudaError = c_int; // cudaSuccess == 0
pub(crate) type GraphicsResource = *mut c_void; // cudaGraphicsResource_t
pub(crate) type ExternalMemory = *mut c_void; // cudaExternalMemory_t

#[allow(non_snake_case, dead_code)]
pub(crate) struct CudaTable {
    _lib: &'static Library,
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
}

static TABLE: OnceLock<Option<CudaTable>> = OnceLock::new();

fn load() -> Option<CudaTable> {
    let lib = ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        .iter()
        .find_map(|n| unsafe { Library::new(n) }.ok())?;
    let lib: &'static Library = Box::leak(Box::new(lib));
    macro_rules! sym {
        ($n:literal) => {{
            *unsafe { lib.get(concat!($n, "\0").as_bytes()) }.ok()?
        }};
    }
    Some(CudaTable {
        _lib: lib,
        graphics_gl_register_buffer: sym!("cudaGraphicsGLRegisterBuffer"),
        graphics_map_resources: sym!("cudaGraphicsMapResources"),
        graphics_get_mapped_pointer: sym!("cudaGraphicsResourceGetMappedPointer"),
        graphics_unmap_resources: sym!("cudaGraphicsUnmapResources"),
        graphics_unregister_resource: sym!("cudaGraphicsUnregisterResource"),
        import_external_memory: sym!("cudaImportExternalMemory"),
        external_memory_get_mapped_buffer: sym!("cudaExternalMemoryGetMappedBuffer"),
        destroy_external_memory: sym!("cudaDestroyExternalMemory"),
    })
}

pub(crate) fn table() -> Option<&'static CudaTable> {
    TABLE.get_or_init(load).as_ref()
}

/// True iff libcudart loaded and all interop symbols resolved. Cached, cheap.
pub fn cuda_available() -> bool {
    table().is_some()
}

/// Register a GL buffer (PBO) with CUDA. Returns the resource as `usize`
/// (pointer) or `None`. MUST be called on the thread where the GL context is
/// current.
pub fn gl_register_buffer(buffer_id: u32) -> Option<usize> {
    let t = table()?;
    let mut res: GraphicsResource = std::ptr::null_mut();
    // cudaGraphicsRegisterFlagsNone = 0
    if unsafe { (t.graphics_gl_register_buffer)(&mut res, buffer_id, 0) } != 0 {
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
#[repr(C)]
pub(crate) struct CudaExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub _tail: u32,
}

/// Import a DMA-BUF fd as CUDA external memory and map it to a device pointer.
///
/// Thread-independent — no GL context is required. CUDA dups the fd internally,
/// so the caller's `fd` remains valid after this call returns. Returns
/// `(ext_mem_handle, device_ptr)` on success, or `None` on any failure (missing
/// libcudart, unsupported platform, or driver error).
///
/// # Safety contract (caller)
/// - `fd` must be a valid, open DMA-BUF file descriptor for the lifetime of the
///   returned `ExternalMemory` handle.
/// - The caller must call `cudaDestroyExternalMemory` (via [`CudaHandle`] drop)
///   before closing `fd`.
///
/// # RUNTIME-UNVALIDATED
/// No test platform has both `/dev/dma_heap` and a CUDA device. ABI is
/// layout-asserted vs. CUDA 12.6 `driver_types.h`; the mechanism is proven
/// by gpu-probe O5 on Orin. Best-effort: returns `None` on failure.
pub(crate) fn import_dma_fd(fd: i32, size: usize) -> Option<(ExternalMemory, *mut c_void)> {
    let t = table()?;
    let mut desc: CudaExternalMemoryHandleDesc = unsafe { std::mem::zeroed() };
    desc.type_ = CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD as c_int;
    desc.handle_fd = fd;
    desc.size = size as u64;
    let mut ext: ExternalMemory = std::ptr::null_mut();
    if unsafe { (t.import_external_memory)(&mut ext, &desc as *const _ as *const c_void) } != 0 {
        return None;
    }
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
    /// `ExternalMem` is persistent (no per-call map/unmap).
    pub fn map(&self) -> Option<CudaMap<'_>> {
        match &self.kind {
            CudaBacking::GlBuffer { resource, ops } => {
                let (ptr, len) = ops.map(*resource)?;
                Some(CudaMap {
                    ptr,
                    len,
                    unmap: Some((ops.clone(), *resource)),
                    _marker: std::marker::PhantomData,
                })
            }
            CudaBacking::ExternalMem { dptr, .. } => Some(CudaMap {
                ptr: *dptr,
                len: self.size,
                unmap: None,
                _marker: std::marker::PhantomData,
            }),
        }
    }
}

impl Drop for CudaHandle {
    fn drop(&mut self) {
        match &self.kind {
            CudaBacking::GlBuffer { resource, ops } => ops.unregister(*resource),
            CudaBacking::ExternalMem { ext_mem, .. } => {
                if let Some(t) = table() {
                    unsafe { (t.destroy_external_memory)(*ext_mem) };
                }
            }
        }
    }
}

/// Scoped CUDA device-pointer mapping. `Drop` unmaps a `GlBuffer` (so GL may
/// reuse the PBO for the next `convert()` call). `ExternalMem` mappings are
/// persistent — `Drop` is a no-op.
pub struct CudaMap<'a> {
    ptr: *mut c_void,
    len: usize,
    unmap: Option<(Arc<dyn CudaGlOps>, GraphicsResource)>,
    _marker: std::marker::PhantomData<&'a ()>,
}

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
}

impl Drop for CudaMap<'_> {
    fn drop(&mut self) {
        if let Some((ops, r)) = self.unmap.take() {
            ops.unmap(r);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_table_loads_when_libcudart_present() {
        let avail = cuda_available();
        if avail {
            assert!(table().is_some(), "table present when available");
        }
        // total + non-panicking either way
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
    }
    impl CudaGlOps for MockOps {
        fn map(&self, _r: GraphicsResource) -> Option<(*mut std::ffi::c_void, usize)> {
            Some((0x1000usize as *mut _, 4096))
        }
        fn unmap(&self, _r: GraphicsResource) {
            self.unmaps.fetch_add(1, Ordering::SeqCst);
        }
        fn unregister(&self, _r: GraphicsResource) {}
    }
    #[test]
    fn cudamap_guard_unmaps_on_drop_for_glbuffer() {
        let unmaps = Arc::new(AtomicUsize::new(0));
        let h = CudaHandle::new_gl(
            0x1usize as GraphicsResource,
            4096,
            Arc::new(MockOps {
                unmaps: unmaps.clone(),
            }),
        );
        {
            let m = h.map().expect("map");
            assert_eq!(m.device_ptr() as usize, 0x1000);
            assert_eq!(m.len(), 4096);
        }
        assert_eq!(
            unmaps.load(Ordering::SeqCst),
            1,
            "Drop must unmap a GlBuffer"
        );
    }
}
