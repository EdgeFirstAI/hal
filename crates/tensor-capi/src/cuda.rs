// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Zero-copy CUDA mapping of a tensor.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::handle::{ef_tensor_free, ef_tensor_retain, tensor_of, EfTensor};

/// Heap object behind the opaque `void *` returned by [`ef_tensor_cuda_map`].
///
/// `map` is transmuted to `'static` for FFI storage. That is only sound
/// while the tensor allocation stays alive, so this wrapper retains `t`
/// for the map's lifetime and releases it in [`ef_tensor_cuda_unmap`].
struct CudaMapHandle {
    map: edgefirst_tensor::CudaMap<'static>,
    tensor: *mut EfTensor,
}

/// Map `t` for CUDA use. Returns an opaque map, or NULL if CUDA is unavailable.
///
/// The map retains `t`. The caller may `ef_tensor_free` their own handle
/// while the map is outstanding; [`ef_tensor_cuda_unmap`] releases that
/// retain.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_map(t: *const EfTensor) -> *mut c_void {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(inner) = tensor_of(t) else {
                return std::ptr::null_mut();
            };
            match inner.cuda_map() {
                Some(m) => {
                    let t_mut = t.cast_mut();
                    if ef_tensor_retain(t_mut) != 0 {
                        return std::ptr::null_mut();
                    }
                    // SAFETY: the retain above keeps the tensor allocation
                    // alive until `ef_tensor_cuda_unmap` drops this handle
                    // and calls `ef_tensor_free`.
                    let m_static: edgefirst_tensor::CudaMap<'static> = std::mem::transmute(m);
                    Box::into_raw(Box::new(CudaMapHandle {
                        map: m_static,
                        tensor: t_mut,
                    })) as *mut c_void
                }
                None => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Device pointer from a map returned by [`ef_tensor_cuda_map`].
///
/// # Safety
/// `map` must be `NULL` or a live map. `out_size` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_device_ptr(
    map: *const c_void,
    out_size: *mut usize,
) -> *mut c_void {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if map.is_null() {
                if !out_size.is_null() {
                    *out_size = 0;
                }
                return std::ptr::null_mut();
            }
            let h = &*(map as *const CudaMapHandle);
            if !out_size.is_null() {
                *out_size = h.map.len();
            }
            h.map.device_ptr()
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Release a map from [`ef_tensor_cuda_map`]. NULL is a no-op.
///
/// Drops the CUDA mapping first, then releases the retain taken at map.
///
/// # Safety
/// `map` must be `NULL` or have come from [`ef_tensor_cuda_map`].
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_unmap(map: *mut c_void) {
    unsafe {
        if map.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let CudaMapHandle { map, tensor } = *Box::from_raw(map as *mut CudaMapHandle);
            drop(map);
            ef_tensor_free(tensor);
        }));
    }
}
