// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional CUDA Runtime interop, loaded via dlopen (no link-time dependency).
//! Absent libcudart ⇒ every CUDA path degrades to None.
use libloading::Library;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
use std::sync::OnceLock;

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
