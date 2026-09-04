// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Zero-copy CUDA mapping of a tensor.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::TensorDyn;

use crate::handle::{ef_tensor_free, ef_tensor_retain, tensor_of, EfTensor};
use crate::last_error::{ensure_hook_installed, set_errno};

/// Heap object behind the opaque `void *` returned by [`ef_tensor_cuda_map`].
///
/// `map` is transmuted to `'static` for FFI storage. That is only sound
/// while the tensor allocation stays alive, so this wrapper retains `t`
/// for the map's lifetime and releases it in [`ef_tensor_cuda_unmap`].
struct CudaMapHandle {
    map: edgefirst_tensor::CudaMap<'static>,
    tensor: *mut EfTensor,
}

/// The body both map entry points share: take the mapping `mk` returns for
/// `t`, retain `t` for as long as that mapping lives, and box the pair as
/// the opaque handle the caller receives. One body so the retain that makes
/// the transmute sound cannot be added to one entry point and forgotten in
/// the other.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
unsafe fn boxed_map(
    t: *const EfTensor,
    mk: fn(&TensorDyn) -> Option<edgefirst_tensor::CudaMap<'_>>,
) -> *mut c_void {
    // SAFETY: `t` is NULL or a live handle by this function's contract, which
    // is what `tensor_of` and `ef_tensor_retain` below require of it.
    ensure_hook_installed();
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(inner) = tensor_of(t) else {
                return std::ptr::null_mut();
            };
            match mk(inner) {
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
        .unwrap_or_else(|_| {
            // An unwind must not leave a stale errno behind the NULL it returns.
            set_errno(libc::EINVAL);
            std::ptr::null_mut()
        })
    }
}

/// Platforms: any host with a CUDA driver.
///
/// Map `t` for CUDA use. Returns an opaque map, or NULL if CUDA is unavailable.
///
/// The map retains `t`. The caller may `ef_tensor_free` their own handle
/// while the map is outstanding; [`ef_tensor_cuda_unmap`] releases that
/// retain.
///
/// On Windows the mapping of a D3D11 texture tensor is tight rows of
/// `width * bytes_per_texel`, and the size [`ef_tensor_cuda_device_ptr`]
/// reports is their sum, whatever `ef_tensor_row_stride` says -- that is the
/// D3D11 staging pitch a CPU map sees, a larger number on a padded backing.
/// A consumer must not stride the device pointer by `ef_tensor_row_stride`.
///
/// A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch, so
/// there the mapping is `ef_tensor_row_stride` times the combined height and
/// striding by it is correct.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_map(t: *const EfTensor) -> *mut c_void {
    // SAFETY: `t` carries this function's own contract into the shared body.
    unsafe { boxed_map(t, TensorDyn::cuda_map) }
}

/// Platforms: any host with a CUDA driver.
///
/// Writable mapping; `ef_tensor_cuda_unmap` writes the device buffer back
/// into the tensor on backings that do not alias (Windows D3D11 textures).
///
/// On Windows the mapping of a D3D11 texture tensor is tight rows of
/// `width * bytes_per_texel`, and the size [`ef_tensor_cuda_device_ptr`]
/// reports is their sum, whatever `ef_tensor_row_stride` says -- that is the
/// D3D11 staging pitch a CPU map sees, a larger number on a padded backing.
/// A consumer must not stride the device pointer by `ef_tensor_row_stride`;
/// writing at that pitch scrambles the image rather than failing.
///
/// A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch, so
/// there the mapping is `ef_tensor_row_stride` times the combined height and
/// striding by it is correct.
///
/// Retains `t` and returns NULL when CUDA is unavailable, exactly as
/// [`ef_tensor_cuda_map`].
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_map_mut(t: *const EfTensor) -> *mut c_void {
    // SAFETY: `t` carries this function's own contract into the shared body.
    unsafe { boxed_map(t, TensorDyn::cuda_map_mut) }
}

/// Platforms: any host with a CUDA driver.
///
/// Device pointer from a map returned by [`ef_tensor_cuda_map`] or
/// [`ef_tensor_cuda_map_mut`].
///
/// `*out_size` is the mapping's length in bytes. On Windows that is the
/// D3D11 texture's tight rows of `width * bytes_per_texel` summed over the
/// rows, not `ef_tensor_row_stride` times the row count -- the stride is the
/// staging pitch a CPU map sees, and striding this pointer by it walks off
/// the end of the mapping.
///
/// A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch, so
/// there the mapping is `ef_tensor_row_stride` times the combined height and
/// striding by it is correct.
///
/// # Safety
/// `map` must be `NULL` or a live map. `out_size` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cuda_device_ptr(
    map: *const c_void,
    out_size: *mut usize,
) -> *mut c_void {
    ensure_hook_installed();
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
        .unwrap_or_else(|_| {
            // An unwind must not leave a stale errno behind the NULL it returns.
            set_errno(libc::EINVAL);
            std::ptr::null_mut()
        })
    }
}

/// Platforms: any host with a CUDA driver.
///
/// Release a map from [`ef_tensor_cuda_map`] or [`ef_tensor_cuda_map_mut`].
/// NULL is a no-op.
///
/// Drops the CUDA mapping first, then releases the retain taken at map. A
/// writable map's drop is where the write-back happens: the device buffer is
/// copied into the tensor and the copy is synchronized before this returns.
///
/// # Safety
/// `map` must be `NULL` or have come from [`ef_tensor_cuda_map`] or
/// [`ef_tensor_cuda_map_mut`].
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{ef_tensor_free, ef_tensor_ndim, ef_tensor_new, ef_tensor_retain};

    #[test]
    fn cuda_map_keeps_the_tensor_alive_after_the_caller_frees() {
        // Copilot review of this file: transmute-to-'static is only sound if
        // the map retains `t`. `ef_tensor_cuda_map` already does that; this
        // test is the contract, including the common NULL-map (no CUDA) path
        // which must *not* take a retain.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        // Extra owner so we can observe liveness after dropping the mint.
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);

        let map = unsafe { ef_tensor_cuda_map(t) };
        if map.is_null() {
            unsafe { ef_tensor_free(t) };
            assert_eq!(
                unsafe { ef_tensor_ndim(t) },
                1,
                "a NULL cuda map must not consume a retain"
            );
            unsafe { ef_tensor_free(t) };
            return;
        }

        unsafe { ef_tensor_free(t) };
        unsafe { ef_tensor_free(t) };
        let mut size = 0usize;
        let _ptr = unsafe { ef_tensor_cuda_device_ptr(map, &mut size) };
        assert_eq!(
            unsafe { ef_tensor_ndim(t) },
            1,
            "the map's retain must keep the handle alive after the caller frees"
        );
        unsafe { ef_tensor_cuda_unmap(map) };
    }

    #[test]
    fn cuda_map_mut_takes_the_same_retain_as_cuda_map() {
        // The writable export shares `boxed_map` with the read-only one, so
        // this pins the shared body's retain from the second entry point:
        // a NULL map consumes no retain, a live one keeps the handle alive.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);

        let map = unsafe { ef_tensor_cuda_map_mut(t) };
        if map.is_null() {
            unsafe { ef_tensor_free(t) };
            assert_eq!(
                unsafe { ef_tensor_ndim(t) },
                1,
                "a NULL writable cuda map must not consume a retain"
            );
            unsafe { ef_tensor_free(t) };
            return;
        }

        unsafe { ef_tensor_free(t) };
        unsafe { ef_tensor_free(t) };
        assert_eq!(
            unsafe { ef_tensor_ndim(t) },
            1,
            "the writable map's retain must keep the handle alive after the caller frees"
        );
        unsafe { ef_tensor_cuda_unmap(map) };
    }

    #[test]
    fn cuda_unmap_of_null_is_a_noop() {
        unsafe { ef_tensor_cuda_unmap(std::ptr::null_mut()) };
    }
}
