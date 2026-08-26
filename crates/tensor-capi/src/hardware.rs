// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! AHardwareBuffer and IOSurface accessors.

use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[cfg(target_os = "android")]
use crate::handle::into_handle;
#[cfg(any(target_os = "android", target_os = "macos", target_os = "ios"))]
use crate::handle::tensor_of;
#[cfg(target_os = "android")]
use edgefirst_tensor::{DType, TensorDyn};
#[cfg(target_os = "android")]
use std::ffi::CStr;

use crate::handle::EfTensor;

fn set_errno(code: i32) {
    errno::set_errno(errno::Errno(code));
}

/// Wrap an AHardwareBuffer. `NULL` / `ENOTSUP` off Android.
///
/// # Safety
/// `buffer` and `dims` must be valid when non-NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_from_hardware_buffer(
    dtype: u32,
    buffer: *mut c_void,
    dims: *const u64,
    ndim: u32,
    name: *const c_char,
) -> *mut EfTensor {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(not(target_os = "android"))]
        {
            let _ = (dtype, buffer, dims, ndim, name);
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
        #[cfg(target_os = "android")]
        unsafe {
            if buffer.is_null() || dims.is_null() || ndim == 0 || ndim > 8 {
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            }
            let Some(dt) = DType::from_code(dtype) else {
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            let shape: Vec<usize> = std::slice::from_raw_parts(dims, ndim as usize)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let name_opt = if name.is_null() {
                None
            } else {
                CStr::from_ptr(name).to_str().ok()
            };
            match TensorDyn::from_hardware_buffer(buffer, &shape, dt, name_opt) {
                Ok(t) => into_handle(t),
                Err(_) => {
                    set_errno(libc::EIO);
                    std::ptr::null_mut()
                }
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Borrowed AHardwareBuffer pointer, or NULL / `ENOTSUP`.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_hardware_buffer_ptr(t: *const EfTensor) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(not(target_os = "android"))]
        {
            let _ = t;
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
        #[cfg(target_os = "android")]
        {
            let Some(inner) = tensor_of(t) else {
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            match inner.hardware_buffer_ptr() {
                Some(p) => p,
                None => {
                    set_errno(libc::ENOTSUP);
                    std::ptr::null_mut()
                }
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Physical AHardwareBuffer dimensions in texels.
///
/// # Safety
/// `width` and `height` must be writable when non-NULL.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_hardware_buffer_physical_dims(
    t: *const EfTensor,
    width: *mut usize,
    height: *mut usize,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(not(target_os = "android"))]
        {
            let _ = (t, width, height);
            set_errno(libc::ENOTSUP);
            libc::ENOTSUP
        }
        #[cfg(target_os = "android")]
        unsafe {
            if width.is_null() || height.is_null() {
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            };
            match inner.hardware_buffer_physical_dims() {
                Some((w, h)) => {
                    *width = w;
                    *height = h;
                    0
                }
                None => {
                    set_errno(libc::ENOTSUP);
                    libc::ENOTSUP
                }
            }
        }
    }))
    .unwrap_or(libc::EINVAL)
}

/// Borrowed IOSurfaceRef, or NULL / `ENOTSUP` off Apple.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_iosurface_ref(t: *const EfTensor) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let _ = t;
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let Some(inner) = tensor_of(t) else {
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            match inner.iosurface_ref() {
                Some(p) => p,
                None => {
                    set_errno(libc::ENOTSUP);
                    std::ptr::null_mut()
                }
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}
