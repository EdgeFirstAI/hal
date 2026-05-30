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
