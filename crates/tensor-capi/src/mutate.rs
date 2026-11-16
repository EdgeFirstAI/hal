// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Live-handle geometry mutators: `ef_tensor_set_format`,
//! `ef_tensor_set_row_stride`, `ef_tensor_set_row_stride_unchecked`,
//! `ef_tensor_set_plane_offset`, `ef_tensor_configure_image` -- plus
//! `ef_tensor_plane_offset`, the reader for [`ef_tensor_set_plane_offset`].
//!
//! `ef_tensor_set_row_stride_unchecked` sets a row stride on a tensor that
//! carries no format, such as a multiplane chroma sub-tensor.
//! `ef_tensor_plane_at` reports that stride for such a tensor.
//!
//! Each of these wraps the `TensorDyn` method of the same name, adding null
//! checks, `PixelFormat` string parsing, errno translation, and a refresh of
//! [`crate::handle::EfTensorImpl`]'s cached `shape_u64`/`strides_i64`/
//! `format_c`.
//!
//! # Concurrency
//!
//! Every setter here goes through [`crate::handle::imp_mut`] and mutates
//! `inner`'s fields directly: the caller must ensure no other thread is
//! calling any `tensor-capi` function on the same handle while one of these
//! is in flight.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, PixelFormat};

use crate::handle::{imp_mut, refresh_caches, tensor_of, EfTensor};
use crate::last_error::{set_last_error, shield_int};
use crate::map::errno_for;

/// Parse a NUL-terminated `PixelFormat` wire string. Returns `None` for
/// `NULL`, invalid UTF-8 or an unrecognized format.
unsafe fn parse_format(f: *const c_char) -> Option<PixelFormat> {
    unsafe {
        if f.is_null() {
            return None;
        }
        let s = CStr::from_ptr(f).to_str().ok()?;
        PixelFormat::from_str_code(s)
    }
}

/// Attach pixel format metadata to a live handle, validating that its shape
/// is compatible with the format's layout. See
/// [`edgefirst_tensor::TensorDyn::set_format`].
///
/// @retval 0 success; the format is attached and `ef_tensor_format` /
///         `ef_tensor_plane_count` / `ef_tensor_plane_at` reflect it.
/// @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
///         the tensor's current shape does not match the format's layout
///         (packed expects `[H, W, C]`, planar `[C, H, W]`, semi-planar
///         `[H*k, W]`).
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `format` must be `NULL` or a
/// NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_format(t: *mut EfTensor, format: *const c_char) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("set_format: null tensor");
                return libc::EINVAL;
            }
            let Some(fmt) = parse_format(format) else {
                set_last_error("set_format: null, non-UTF8, or unrecognized format string");
                return libc::EINVAL;
            };
            let Some(imp) = imp_mut(t) else {
                set_last_error("set_format: could not resolve handle");
                return libc::EINVAL;
            };
            match imp.inner.set_format(fmt) {
                Ok(()) => {
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("set_format: {e}"));
                    errno
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Set the row stride in bytes for a live handle with padded rows (e.g. a
/// V4L2/GStreamer allocator's buffer). Must be called after
/// [`ef_tensor_set_format`]. See
/// [`edgefirst_tensor::TensorDyn::set_row_stride`].
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`, no pixel format is set on this tensor yet,
///         or `stride` is smaller than the format's minimum row size at the
///         tensor's current width.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_row_stride(t: *mut EfTensor, stride: usize) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("set_row_stride: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = imp_mut(t) else {
                set_last_error("set_row_stride: could not resolve handle");
                return libc::EINVAL;
            };
            match imp.inner.set_row_stride(stride) {
                Ok(()) => {
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("set_row_stride: {e}"));
                    errno
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Set the row stride in bytes without format validation. See
/// [`edgefirst_tensor::TensorDyn::set_row_stride_unchecked`].
///
/// Unlike [`ef_tensor_set_row_stride`], this requires no pixel format and
/// does not validate `stride` against one. The caller is responsible for
/// the stride being valid.
///
/// For a tensor with no format, `ef_tensor_plane_at` reports this stride as
/// plane 0's stride.
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_row_stride_unchecked(
    t: *mut EfTensor,
    stride: usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("set_row_stride_unchecked: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = imp_mut(t) else {
                set_last_error("set_row_stride_unchecked: could not resolve handle");
                return libc::EINVAL;
            };
            imp.inner.set_row_stride_unchecked(stride);
            refresh_caches(imp);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Set the byte offset within the backing buffer where image data starts.
/// Format-independent, unlike [`ef_tensor_set_row_stride`]. See
/// [`edgefirst_tensor::TensorDyn::set_plane_offset`].
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_plane_offset(t: *mut EfTensor, offset: usize) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("set_plane_offset: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = imp_mut(t) else {
                set_last_error("set_plane_offset: could not resolve handle");
                return libc::EINVAL;
            };
            imp.inner.set_plane_offset(offset);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Read back the byte offset within the backing buffer where image data
/// starts, as set by [`ef_tensor_set_plane_offset`] (or a producer's
/// `ef_tensor_builder_add_plane` at construction). See
/// [`edgefirst_tensor::TensorDyn::plane_offset`].
///
/// @retval `>= 0` the current plane offset in bytes.
/// @retval `-1` `t` is `NULL`, or no offset has been set (the default).
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_plane_offset(t: *const EfTensor) -> i64 {
    catch_unwind(AssertUnwindSafe(|| {
        if t.is_null() {
            return -1;
        }
        let Some(inner) = tensor_of(t) else {
            return -1;
        };
        inner.plane_offset().map(|o| o as i64).unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

/// Change a tensor's logical shape, keeping the same element count.
///
/// The product of `dims` must equal the current element count. See
/// [`ef_tensor_set_logical_shape`] for the capacity-based variant.
///
/// @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides` reflect the new
///         geometry.
/// @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
///         out of range for this host.
/// @retval ERANGE the new shape's element count differs from the current
///         one -- the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `dims` must point to `ndim`
/// readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_reshape(t: *mut EfTensor, dims: *const u64, ndim: u32) -> c_int {
    unsafe { reshape_impl(t, dims, ndim, false) }
}

/// Change a tensor's logical shape to anything its allocation can hold.
///
/// The capacity-based counterpart to [`ef_tensor_reshape`]: the new shape's
/// element count may differ from the current one.
///
/// @retval 0 success.
/// @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
///         out of range for this host.
/// @retval ERANGE the new shape does not fit the existing allocation --
///         the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `dims` must point to `ndim`
/// readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_logical_shape(
    t: *mut EfTensor,
    dims: *const u64,
    ndim: u32,
) -> c_int {
    unsafe { reshape_impl(t, dims, ndim, true) }
}

/// The shared body of [`ef_tensor_reshape`] and
/// [`ef_tensor_set_logical_shape`].
///
/// # Safety
/// `t` must be `NULL` or a live handle; `dims` must point to `ndim`
/// readable `uint64_t`.
unsafe fn reshape_impl(t: *mut EfTensor, dims: *const u64, ndim: u32, by_capacity: bool) -> c_int {
    let what = if by_capacity {
        "set_logical_shape"
    } else {
        "reshape"
    };
    {
        shield_int(|| {
            if t.is_null() || dims.is_null() || ndim == 0 {
                set_last_error(&format!("{what}: null tensor/dims or zero ndim"));
                return libc::EINVAL;
            }
            // SAFETY: the caller contracts `dims` is readable for `ndim`.
            let raw = unsafe { std::slice::from_raw_parts(dims, ndim as usize) };
            let Some(shape) = raw
                .iter()
                .map(|d| usize::try_from(*d).ok())
                .collect::<Option<Vec<usize>>>()
            else {
                set_last_error(&format!(
                    "{what}: a dimension is out of range for this host's usize"
                ));
                return libc::EINVAL;
            };
            // SAFETY: `t` is checked non-null and contracted live.
            let Some(imp) = (unsafe { imp_mut(t) }) else {
                set_last_error(&format!("{what}: could not resolve handle"));
                return libc::EINVAL;
            };
            let r = if by_capacity {
                imp.inner.set_logical_shape(&shape)
            } else {
                imp.inner.reshape(&shape)
            };
            match r {
                Ok(()) => {
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    let errno = match &e {
                        edgefirst_tensor::Error::ShapeMismatch(_)
                        | edgefirst_tensor::Error::InsufficientCapacity { .. } => libc::ERANGE,
                        other => errno_for(other),
                    };
                    set_last_error(&format!("{what}: {e}"));
                    errno
                }
            }
        })
    }
}

/// Platforms: Linux, macOS, iOS, Android.
///
/// Duplicate the file descriptor backing this tensor, for any storage kind
/// that has one: a dma-buf fd on Linux, a shared-memory fd wherever `Shm`
/// allocates.
///
/// @retval `>= 0` a new file descriptor the caller owns and must `close()`.
/// @retval a negative errno: `-EINVAL` for a `NULL`/invalid handle,
///         `-ENOTSUP` for a backing with no file descriptor at all,
///         or another negative errno from the underlying `dup`.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_clone_fd(t: *const EfTensor) -> c_int {
    {
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("clone_fd: null tensor");
                return -libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_last_error("clone_fd: could not resolve handle");
                return -libc::EINVAL;
            };
            #[cfg(not(unix))]
            {
                let _ = inner;
                set_last_error("clone_fd: not supported on this platform");
                -libc::ENOTSUP
            }
            #[cfg(unix)]
            match inner.clone_fd() {
                // `into_raw_fd` transfers ownership of the fd to the C caller.
                Ok(fd) => {
                    use std::os::fd::IntoRawFd;
                    fd.into_raw_fd()
                }
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("clone_fd: {e}"));
                    -errno
                }
            }
        }))
        .unwrap_or(-libc::EINVAL)
    }
}

/// Retag a tensor's element type without touching its bytes.
///
/// The recorded dtype is metadata over the same allocation: `EF_DTYPE_U8`
/// and `EF_DTYPE_I8` address identical bytes and differ only in how a
/// consumer reads them. A dtype of a different width is refused.
///
/// @retval 0 success; `ef_tensor_dtype` reports `dtype`.
/// @retval EINVAL `t` is `NULL`, or `dtype` is not a recognized code.
/// @retval ERANGE `dtype` has a different element width than the current one
///         -- the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_dtype(t: *mut EfTensor, dtype: u32) -> c_int {
    unsafe {
        shield_int(|| {
            if t.is_null() {
                set_last_error("set_dtype: null tensor");
                return libc::EINVAL;
            }
            let Some(dt) = DType::from_code(dtype) else {
                set_last_error(&format!("set_dtype: unknown dtype code {dtype}"));
                return libc::EINVAL;
            };
            let Some(imp) = imp_mut(t) else {
                set_last_error("set_dtype: could not resolve handle");
                return libc::EINVAL;
            };
            match imp.inner.set_dtype(dt) {
                Ok(()) => {
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    set_last_error(&format!("set_dtype: {e}"));
                    libc::ERANGE
                }
            }
        })
    }
}

/// Reconfigure a live handle's logical dimensions and pixel format, reusing
/// its existing allocation. See
/// [`edgefirst_tensor::TensorDyn::configure_image`].
///
/// @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`
///         /`ef_tensor_plane_at` reflect the new geometry.
/// @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
///         `width`x`height` is not a valid size for `format`.
/// @retval ERANGE the existing allocation cannot hold `width`x`height` in
///         `format`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `format` must be `NULL` or a
/// NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_configure_image(
    t: *mut EfTensor,
    width: usize,
    height: usize,
    format: *const c_char,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("configure_image: null tensor");
                return libc::EINVAL;
            }
            let Some(fmt) = parse_format(format) else {
                set_last_error("configure_image: null, non-UTF8, or unrecognized format string");
                return libc::EINVAL;
            };
            let Some(imp) = imp_mut(t) else {
                set_last_error("configure_image: could not resolve handle");
                return libc::EINVAL;
            };
            match imp.inner.configure_image(width, height, fmt) {
                Ok(()) => {
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("configure_image: {e}"));
                    errno
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{
        ef_tensor_free, ef_tensor_new, ef_tensor_plane_at, inner_of, EfTensorPlane,
    };
    use edgefirst_tensor::PixelFormat as Fmt;

    fn nv12(w: u64, h: u64) -> *mut EfTensor {
        // NV12 combined-plane shape: [H + H/2, W].
        let dims = [h + h / 2, w];
        unsafe { ef_tensor_new(0, dims.as_ptr(), 2) }
    }

    #[test]
    fn set_format_attaches_and_validates_shape() {
        let t = nv12(64, 48);
        let nv12_c = std::ffi::CString::new("NV12").unwrap();
        assert_eq!(unsafe { ef_tensor_set_format(t, nv12_c.as_ptr()) }, 0);
        assert_eq!(inner_of(t).format(), Some(Fmt::Nv12));
        // The cached format string is refreshed.
        let fmt_ptr = unsafe { crate::handle::ef_tensor_format(t) };
        let got = unsafe { std::ffi::CStr::from_ptr(fmt_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(got, "NV12");
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_format_rejects_incompatible_shape() {
        // A 3x3 tensor cannot be RGB (needs [H, W, 3]).
        let dims = [3u64, 3];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let rgb = std::ffi::CString::new("rgb8").unwrap();
        assert_eq!(
            unsafe { ef_tensor_set_format(t, rgb.as_ptr()) },
            libc::EINVAL
        );
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_row_stride_requires_format_first() {
        let t = nv12(64, 48);
        assert_eq!(unsafe { ef_tensor_set_row_stride(t, 128) }, libc::EINVAL);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_row_stride_updates_the_cached_strides() {
        let t = nv12(64, 48);
        let nv12_c = std::ffi::CString::new("NV12").unwrap();
        assert_eq!(unsafe { ef_tensor_set_format(t, nv12_c.as_ptr()) }, 0);
        assert_eq!(unsafe { ef_tensor_set_row_stride(t, 128) }, 0);
        let strides = unsafe { crate::handle::ef_tensor_strides(t) };
        // Row (dim 0) stride is the padded 128 bytes.
        assert_eq!(unsafe { *strides }, 128);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_row_stride_on_planar_pads_rows_and_planes() {
        // PlanarRgb [C, H, W] = [3, 48, 64]: a padded pitch moves the row
        // stride (dim 1) and makes each plane `pitch * H` bytes (dim 0).
        let dims = [3u64, 48, 64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 3) };
        let planar_c = std::ffi::CString::new(Fmt::PlanarRgb.as_str()).unwrap();
        assert_eq!(unsafe { ef_tensor_set_format(t, planar_c.as_ptr()) }, 0);
        assert_eq!(unsafe { ef_tensor_set_row_stride(t, 128) }, 0);
        let strides = unsafe { std::slice::from_raw_parts(crate::handle::ef_tensor_strides(t), 3) };
        assert_eq!(strides, [128 * 48, 128, 1]);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_plane_offset_is_read_back_through_ef_tensor_plane_offset() {
        let t = nv12(64, 48);
        let nv12_c = std::ffi::CString::new("NV12").unwrap();
        unsafe { ef_tensor_set_format(t, nv12_c.as_ptr()) };
        assert_eq!(inner_of(t).plane_offset(), None);
        assert_eq!(unsafe { ef_tensor_plane_offset(t) }, -1);
        assert_eq!(unsafe { ef_tensor_set_plane_offset(t, 4096) }, 0);
        assert_eq!(inner_of(t).plane_offset(), Some(4096));
        assert_eq!(unsafe { ef_tensor_plane_offset(t) }, 4096);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn configure_image_reconfigures_shape_and_is_read_back() {
        // An oversized buffer, reconfigured down to a smaller frame.
        let t = nv12(128, 128);
        let grey = std::ffi::CString::new("mono8").unwrap();
        assert_eq!(
            unsafe { ef_tensor_configure_image(t, 64, 64, grey.as_ptr()) },
            0
        );
        assert_eq!(inner_of(t).format(), Some(Fmt::Grey));
        assert_eq!(inner_of(t).width(), Some(64));
        assert_eq!(inner_of(t).height(), Some(64));
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn configure_image_too_large_for_the_allocation_is_erange() {
        let t = nv12(16, 16);
        let grey = std::ffi::CString::new("mono8").unwrap();
        assert_eq!(
            unsafe { ef_tensor_configure_image(t, 4096, 4096, grey.as_ptr()) },
            libc::ERANGE
        );
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn every_mutator_rejects_a_null_handle() {
        let fmt = std::ffi::CString::new("NV12").unwrap();
        assert_eq!(
            unsafe { ef_tensor_set_format(std::ptr::null_mut(), fmt.as_ptr()) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_set_row_stride(std::ptr::null_mut(), 128) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_set_row_stride_unchecked(std::ptr::null_mut(), 128) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_set_plane_offset(std::ptr::null_mut(), 0) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_configure_image(std::ptr::null_mut(), 1, 1, fmt.as_ptr()) },
            libc::EINVAL
        );
    }

    #[test]
    fn set_row_stride_unchecked_needs_no_format() {
        let dims = [24u64, 64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        assert_eq!(inner_of(t).format(), None);
        assert_eq!(unsafe { ef_tensor_set_row_stride_unchecked(t, 96) }, 0);
        assert_eq!(inner_of(t).row_stride(), Some(96));
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_row_stride_unchecked_is_read_back_through_plane_at() {
        let dims = [24u64, 64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut plane = EfTensorPlane::default();
        assert_eq!(unsafe { ef_tensor_plane_at(t, 0, &mut plane) }, 0);
        let capacity = plane.stride; // no stride set: the whole-buffer fallback
        assert!(capacity > 96, "sanity: allocation is bigger than one row");
        assert_eq!(unsafe { ef_tensor_set_row_stride_unchecked(t, 96) }, 0);
        assert_eq!(unsafe { ef_tensor_plane_at(t, 0, &mut plane) }, 0);
        assert_eq!(
            plane.stride, 96,
            "must read back the real stride, not capacity_bytes()"
        );
        unsafe { ef_tensor_free(t) };
    }
}
