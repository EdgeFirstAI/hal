// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `ef_tensor_image_desc` — the full-featured image request.
//!
//! `ef_image_processor_create_image` (in `edgefirst-image`) covers the common
//! case in one call. This request is the only route to compression.
//!
//! The builder constructs an [`edgefirst_tensor::ImageDesc`].
//! `ef_image_processor_create_image_desc` in `edgefirst-image` allocates from
//! a `struct ef_tensor_image_desc *` built here.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{Compression, CpuAccess, DType, ImageDesc, PixelFormat, TensorMemory};
use edgefirst_tensor_abi::{EfCompressionRequest, EfImageDescView};

/// An image request, built up field by field.
pub struct EfTensorImageDesc {
    pub(crate) inner: ImageDesc,
}

/// Create a request for a `width`×`height` image.
///
/// `format` is the wire descriptor (`"NV12"`, `"rgb8"`). `dtype` is an
/// `ef_dtype` code. CPU access defaults to `EF_CPU_ACCESS_READ_WRITE`.
///
/// @return `NULL` for an unknown format or dtype, or zero dimensions.
///
/// # Safety
/// `format` must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_new(
    width: usize,
    height: usize,
    format: *const c_char,
    dtype: u32,
) -> *mut EfTensorImageDesc {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if format.is_null() || width == 0 || height == 0 {
                return std::ptr::null_mut();
            }
            let Ok(f) = CStr::from_ptr(format).to_str() else {
                return std::ptr::null_mut();
            };
            let (Some(fmt), Some(dt)) = (PixelFormat::from_str_code(f), DType::from_code(dtype))
            else {
                return std::ptr::null_mut();
            };
            Box::into_raw(Box::new(EfTensorImageDesc {
                inner: ImageDesc::new(width, height, fmt, dt),
            }))
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free a request. Freeing `NULL` is a no-op.
///
/// # Safety
/// `d` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_free(d: *mut EfTensorImageDesc) {
    unsafe {
        if d.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(d))));
    }
}

/// Apply a builder step in place: swap the value out for a 1x1 grey
/// placeholder, run `body`, and store its result. Returns `EINVAL` for a
/// `NULL` request, or when `body` returns `None`, which leaves the
/// placeholder in place.
unsafe fn with_desc<F>(d: *mut EfTensorImageDesc, body: F) -> c_int
where
    F: FnOnce(ImageDesc) -> Option<ImageDesc>,
{
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                return libc::EINVAL;
            }
            let slot = &mut (*d).inner;
            let taken = std::mem::replace(slot, ImageDesc::new(1, 1, PixelFormat::Grey, DType::U8));
            match body(taken) {
                Some(next) => {
                    *slot = next;
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Request a specific backing store, by `ef_storage_kind` code.
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_set_memory(
    d: *mut EfTensorImageDesc,
    kind: u32,
) -> c_int {
    unsafe {
        with_desc(d, |desc| {
            TensorMemory::from_code(kind).map(|m| desc.with_memory(Some(m)))
        })
    }
}

/// Declare CPU access, by `ef_cpu_access` code (`EF_CPU_ACCESS_*`).
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_set_access(
    d: *mut EfTensorImageDesc,
    access: u32,
) -> c_int {
    unsafe {
        with_desc(d, |desc| {
            Some(desc.with_access(CpuAccess::from_code(access)?))
        })
    }
}

/// Request compression, by `ef_compression_request` code:
/// `EF_COMPRESSION_REQUEST_NONE` or `EF_COMPRESSION_REQUEST_ANY` (any scheme
/// the platform offers). Any other code is `EINVAL`.
///
/// `Any` allocates linear when the format is not eligible and counts the
/// fallback.
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_set_compression(
    d: *mut EfTensorImageDesc,
    compression: u32,
) -> c_int {
    unsafe {
        with_desc(d, |desc| {
            if compression == EfCompressionRequest::None as u32 {
                Some(desc)
            } else if compression == EfCompressionRequest::Any as u32 {
                Some(desc.with_compression(Compression::Any))
            } else {
                None
            }
        })
    }
}

/// Require physically contiguous memory: 0 = no, 1 = yes.
///
/// A contiguous request is allocated from the Linux CMA DMA heap only:
/// `ef_tensor_image_desc_alloc` fails, naming the heap, when that heap
/// cannot serve it.
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_set_contiguous(
    d: *mut EfTensorImageDesc,
    contiguous: c_int,
) -> c_int {
    unsafe {
        with_desc(d, |desc| match contiguous {
            0 => Some(desc.with_contiguous(false)),
            1 => Some(desc.with_contiguous(true)),
            _ => None,
        })
    }
}

/// Whether the request requires physically contiguous memory: 1 if it does,
/// 0 if not or `d` is `NULL`.
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_contiguous(d: *const EfTensorImageDesc) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            d.as_ref().map_or(0, |d| c_int::from(d.inner.contiguous()))
        }))
        .unwrap_or(0)
    }
}

/// Flatten an `ImageDesc` into the `#[repr(C)]` view a foreign library
/// reads.
fn view_of(d: &ImageDesc) -> EfImageDescView {
    let (memory, has_memory) = match d.memory() {
        Some(m) => (m.code(), 1),
        None => (0, 0),
    };
    // The compression request is encoded as an `EfCompressionRequest` code.
    let (compression, has_compression) = match d.compression() {
        None => (EfCompressionRequest::None as u32, 0),
        Some(Compression::Any) => (EfCompressionRequest::Any as u32, 1),
        // Every other variant is reported as `Scheme`.
        Some(_) => (EfCompressionRequest::Scheme as u32, 1),
    };
    let access = d.access().code();
    EfImageDescView {
        width: d.width() as u64,
        height: d.height() as u64,
        format: d.format().code(),
        dtype: d.dtype().code(),
        access,
        memory,
        has_memory,
        compression,
        has_compression,
    }
}

/// Read a request's fields into `out`.
///
/// @return 0 on success, `EINVAL` for a `NULL` argument.
///
/// # Safety
/// `d` must be `NULL` or a live request. `out` must be a valid pointer to a
/// writable `ef_image_desc_view`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_get(
    d: *const EfTensorImageDesc,
    out: *mut EfImageDescView,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            *out = view_of(&(*d).inner);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Allocate an image tensor from a finished request. See
/// [`edgefirst_tensor::TensorDyn::image_desc`].
///
/// @retval a new tensor the caller must free with `ef_tensor_free`, on
///         success.
/// @retval `NULL` for a `NULL` request, or if the underlying allocation
///         fails (an incompatible `CpuAccess`/compression combination, an
///         unsupported compression request, or an invalid `width`x`height`
///         for the format) -- `ef_tensor_last_error_message` carries the
///         reason.
///
/// # Safety
/// `d` must be `NULL` or a live request from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_alloc(
    d: *const EfTensorImageDesc,
) -> *mut crate::handle::EfTensor {
    unsafe {
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                crate::last_error::set_last_error("image_desc_alloc: null request");
                return std::ptr::null_mut();
            }
            match edgefirst_tensor::TensorDyn::image_desc(&(*d).inner) {
                Ok(t) => crate::handle::into_handle(t),
                Err(e) => {
                    crate::last_error::set_last_error_classified(
                        crate::last_error::class_of(&e),
                        &format!("image_desc_alloc: {e}"),
                    );
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc() -> *mut EfTensorImageDesc {
        let f = std::ffi::CString::new("rgb8").unwrap();
        let d = unsafe { ef_tensor_image_desc_new(64, 48, f.as_ptr(), 0) };
        assert!(!d.is_null());
        d
    }

    #[test]
    fn a_request_can_be_built_and_freed() {
        unsafe {
            let d = desc();
            assert_eq!(
                ef_tensor_image_desc_set_memory(d, TensorMemory::Mem.code()),
                0
            );
            assert_eq!(
                ef_tensor_image_desc_set_access(d, CpuAccess::ReadWrite.code()),
                0
            );
            assert_eq!(
                ef_tensor_image_desc_set_compression(d, EfCompressionRequest::None as u32),
                0
            );
            ef_tensor_image_desc_free(d);
            ef_tensor_image_desc_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn bad_construction_arguments_are_refused() {
        unsafe {
            let f = std::ffi::CString::new("rgb8").unwrap();
            let junk = std::ffi::CString::new("not-a-format").unwrap();
            assert!(ef_tensor_image_desc_new(0, 48, f.as_ptr(), 0).is_null());
            assert!(ef_tensor_image_desc_new(64, 0, f.as_ptr(), 0).is_null());
            assert!(ef_tensor_image_desc_new(64, 48, std::ptr::null(), 0).is_null());
            assert!(ef_tensor_image_desc_new(64, 48, junk.as_ptr(), 0).is_null());
            assert!(ef_tensor_image_desc_new(64, 48, f.as_ptr(), 9999).is_null());
        }
    }

    #[test]
    fn every_setter_rejects_an_unknown_code_and_a_null_handle() {
        unsafe {
            let d = desc();
            assert_eq!(ef_tensor_image_desc_set_memory(d, 9999), libc::EINVAL);
            assert_eq!(ef_tensor_image_desc_set_access(d, 9999), libc::EINVAL);
            assert_eq!(ef_tensor_image_desc_set_compression(d, 9999), libc::EINVAL);
            let n = std::ptr::null_mut();
            assert_eq!(ef_tensor_image_desc_set_memory(n, 0), libc::EINVAL);
            assert_eq!(ef_tensor_image_desc_set_access(n, 0), libc::EINVAL);
            assert_eq!(ef_tensor_image_desc_set_compression(n, 0), libc::EINVAL);
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn the_contiguous_request_round_trips_and_rejects_bad_input() {
        unsafe {
            let d = desc();
            assert_eq!(ef_tensor_image_desc_contiguous(d), 0);
            assert_eq!(ef_tensor_image_desc_set_contiguous(d, 1), 0);
            assert_eq!(ef_tensor_image_desc_contiguous(d), 1);
            assert!((*d).inner.contiguous());
            assert_eq!(ef_tensor_image_desc_set_contiguous(d, 0), 0);
            assert_eq!(ef_tensor_image_desc_contiguous(d), 0);
            assert_eq!(ef_tensor_image_desc_set_contiguous(d, 2), libc::EINVAL);
            assert_eq!(
                ef_tensor_image_desc_set_contiguous(std::ptr::null_mut(), 1),
                libc::EINVAL
            );
            assert_eq!(ef_tensor_image_desc_contiguous(std::ptr::null()), 0);
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn a_rejected_setter_leaves_the_request_usable() {
        // A rejected setter leaves the 1x1 grey placeholder `with_desc`
        // swapped in.
        unsafe {
            let d = desc();
            assert_eq!(ef_tensor_image_desc_set_memory(d, 9999), libc::EINVAL);
            assert_eq!((*d).inner.width(), 1);
            assert_eq!((*d).inner.height(), 1);
            assert_eq!((*d).inner.format(), PixelFormat::Grey);
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn a_new_request_declares_readwrite() {
        unsafe {
            let d = desc();
            let mut view = EfImageDescView::default();
            assert_eq!(ef_tensor_image_desc_get(d, &mut view), 0);
            assert_eq!(
                CpuAccess::from_code(view.access),
                Some(CpuAccess::ReadWrite)
            );
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn the_view_round_trips_every_field() {
        unsafe {
            let d = desc(); // 64x48 rgb8 u8, no memory/compression request
            assert_eq!(
                ef_tensor_image_desc_set_memory(d, TensorMemory::Shm.code()),
                0
            );
            assert_eq!(
                ef_tensor_image_desc_set_access(d, CpuAccess::Read.code()),
                0
            );
            assert_eq!(
                ef_tensor_image_desc_set_compression(d, EfCompressionRequest::Any as u32),
                0
            );

            let mut view = EfImageDescView::default();
            assert_eq!(ef_tensor_image_desc_get(d, &mut view), 0);

            assert_eq!(view.width, 64);
            assert_eq!(view.height, 48);
            assert_eq!(PixelFormat::from_code(view.format), Some(PixelFormat::Rgb));
            assert_eq!(DType::from_code(view.dtype), Some(DType::U8));
            assert_eq!(CpuAccess::from_code(view.access), Some(CpuAccess::Read));
            assert_eq!(view.has_memory, 1);
            assert_eq!(
                TensorMemory::from_code(view.memory),
                Some(TensorMemory::Shm)
            );
            assert_eq!(view.has_compression, 1);
            assert_eq!(view.compression, EfCompressionRequest::Any as u32);

            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn the_view_reports_absence_by_flag_not_by_a_zero_that_collides() {
        // A fresh request has no memory or compression preference.
        unsafe {
            let d = desc();
            let mut view = EfImageDescView::default();
            assert_eq!(ef_tensor_image_desc_get(d, &mut view), 0);
            assert_eq!(view.has_memory, 0);
            assert_eq!(view.has_compression, 0);
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn image_desc_alloc_mints_a_real_tensor_matching_the_request() {
        let d = desc(); // 64x48 rgb8 u8, no memory/compression request
        let t = unsafe { ef_tensor_image_desc_alloc(d) };
        assert!(!t.is_null());
        let inner = crate::handle::inner_of(t);
        assert_eq!(inner.format(), Some(PixelFormat::Rgb));
        assert_eq!(inner.width(), Some(64));
        assert_eq!(inner.height(), Some(48));
        unsafe {
            crate::handle::ef_tensor_free(t);
            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn image_desc_alloc_rejects_a_null_request() {
        assert!(unsafe { ef_tensor_image_desc_alloc(std::ptr::null()) }.is_null());
    }

    #[test]
    fn the_getter_rejects_null_arguments() {
        unsafe {
            let mut view = EfImageDescView::default();
            assert_eq!(
                ef_tensor_image_desc_get(std::ptr::null(), &mut view),
                libc::EINVAL
            );
            let d = desc();
            assert_eq!(
                ef_tensor_image_desc_get(d, std::ptr::null_mut()),
                libc::EINVAL
            );
            ef_tensor_image_desc_free(d);
        }
    }
}
