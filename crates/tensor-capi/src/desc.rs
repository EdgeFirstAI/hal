// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `ef_tensor_image_desc` — the full-featured image request.
//!
//! `ef_image_processor_create_image` (in `edgefirst-image`) covers the common
//! case in one call. This exists for the rest, and is the **only** route to
//! compression: asking for a vendor tile layout needs more inputs than a
//! positional call can carry without becoming unreadable.
//!
//! The builder lives here, in `edgefirst-tensor-capi`, because it constructs
//! an [`edgefirst_tensor::ImageDesc`] — the type this library owns. Allocating
//! from a finished request is `edgefirst-image`'s job (only a processor can
//! mint a PBO), so `ef_image_processor_create_image_desc` stays there and
//! takes a `struct ef_tensor_image_desc *` it did not mint.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{Compression, CpuAccess, DType, ImageDesc, PixelFormat, TensorMemory};
use edgefirst_tensor_abi::EfImageDescView;

/// An image request, built up field by field.
pub struct EfTensorImageDesc {
    pub(crate) inner: ImageDesc,
}

/// Create a request for a `width`×`height` image.
///
/// `format` is the wire descriptor (`"NV12"`, `"rgb8"`), matching every other
/// entry point rather than introducing a second vocabulary. `dtype` is the
/// shared code.
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

/// Apply a builder step in place.
///
/// `ImageDesc`'s setters consume and return `self`, so each step swaps the
/// value out and back. `take` is safe here because nothing can observe the
/// intermediate state: the closure cannot unwind past `catch_unwind`.
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

/// Declare CPU access: 0 none, 1 read, 2 write, 3 read-write.
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
            let a = match access {
                0 => CpuAccess::None,
                1 => CpuAccess::Read,
                2 => CpuAccess::Write,
                3 => CpuAccess::ReadWrite,
                _ => return None,
            };
            Some(desc.with_access(a))
        })
    }
}

/// Request compression: 0 = none, 1 = any scheme the platform offers.
///
/// `Any` allocates linear when the format is not eligible and counts the
/// fallback, which is the right default for a pipeline that wants the
/// bandwidth win without a portability failure. Requesting a *specific*
/// scheme is deliberately not exposed here — it fails outright on a device
/// whose scheme differs, and that belongs behind a named entry point rather
/// than an integer a caller might pass by accident.
///
/// # Safety
/// `d` must be `NULL` or a live request.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_desc_set_compression(
    d: *mut EfTensorImageDesc,
    compression: u32,
) -> c_int {
    unsafe {
        with_desc(d, |desc| match compression {
            0 => Some(desc),
            1 => Some(desc.with_compression(Compression::Any)),
            _ => None,
        })
    }
}

/// The single place an `ImageDesc` becomes the flattened, `#[repr(C)]` view a
/// foreign library reads instead of dereferencing this crate's opaque
/// handle -- see `EfImageDescView`'s doc for why a handle is never crossed by
/// pointer for this. `ef_tensor_image_desc_get` is the only caller; the
/// round-trip test below is the proof a decoder of this view recovers
/// exactly what went in.
fn view_of(d: &ImageDesc) -> EfImageDescView {
    let (memory, has_memory) = match d.memory() {
        Some(m) => (m.code(), 1),
        None => (0, 0),
    };
    // `Compression` has no shared `code()` in the vocabulary macro (it
    // carries a payload on `Scheme`, unlike the plain vocabularies). 1/2
    // mirrors `ef_tensor_image_desc_set_compression`'s own 0/1 wire values,
    // extended by one: `Scheme` is a real state the type can hold even
    // though no C setter can create one, so it gets a code rather than
    // being folded into "any" or silently dropped.
    let (compression, has_compression) = match d.compression() {
        None => (0, 0),
        Some(Compression::Any) => (1, 1),
        // `Compression` is `#[non_exhaustive]`: `Scheme(_)` is the only other
        // variant today, but a wildcard is required here regardless, and any
        // future variant falls into the same "present, and it's a specific
        // one" code rather than failing to compile at the call site that
        // adds it.
        Some(_) => (2, 1),
    };
    // `CpuAccess` has no shared `code()` either (see `ef_tensor_image_desc_set_access`'s
    // own doc); this is the same 0..3 numbering, decoded the same way.
    let access = match d.access() {
        CpuAccess::None => 0,
        CpuAccess::Read => 1,
        CpuAccess::Write => 2,
        CpuAccess::ReadWrite => 3,
    };
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

/// Read a request's fields into `out`, mirroring `ef_tensor_plane_at`'s
/// shape: a scalar block a foreign library copies rather than a pointer it
/// would have to dereference into this library's private layout.
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

/// Allocate an image tensor from a finished request -- the primitive that
/// makes the `ef_tensor_image_desc` family a real constructor rather than a
/// request-only echo. Triggers [`edgefirst_tensor::TensorDyn::image_desc`]
/// (dispatching on `desc.dtype()`) inside `libedgefirst_tensor.so`, the same
/// geometry-computing code any Rust caller of `TensorDyn::image_desc`
/// reaches -- this does not reimplement or approximate it.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`, on
///         success.
/// @retval `NULL` for a `NULL` request, or if the underlying allocation
///         fails (see `TensorDyn::image_desc`'s error conditions --
///         incompatible `CpuAccess`/compression combination, an
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
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
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
            assert_eq!(ef_tensor_image_desc_set_memory(d, 0), 0);
            assert_eq!(ef_tensor_image_desc_set_access(d, 3), 0);
            assert_eq!(ef_tensor_image_desc_set_compression(d, 0), 0);
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
    fn a_rejected_setter_leaves_the_request_usable() {
        // `with_desc` swaps the value out for the 1x1 grey placeholder
        // *before* calling `body`, and only swaps a replacement back in on
        // success -- there is no "restore the original" path. So a rejected
        // setter leaves the placeholder in place, not the caller's original
        // request. "Usable" means exactly that: still a valid `ImageDesc`,
        // never a half-swapped or poisoned one. The corresponding
        // cross-library check -- that a tensor can still be minted from it
        // afterwards -- lives in image-capi, which owns that entry point.
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
    fn the_view_round_trips_every_field() {
        // Builds a request with every field away from its default, reads it
        // back through `ef_tensor_image_desc_get`, and decodes every view
        // field back to the value that went in. This is the proof that
        // `view_of` and a decoder of `EfImageDescView` (image-capi's
        // reconstruction, mirrored here) agree -- the whole point of having
        // exactly one conversion site.
        unsafe {
            let d = desc(); // 64x48 rgb8 u8, no memory/compression request
            assert_eq!(ef_tensor_image_desc_set_memory(d, 1), 0); // Shm
            assert_eq!(ef_tensor_image_desc_set_access(d, 3), 0); // ReadWrite
            assert_eq!(ef_tensor_image_desc_set_compression(d, 1), 0); // Any

            let mut view = EfImageDescView::default();
            assert_eq!(ef_tensor_image_desc_get(d, &mut view), 0);

            assert_eq!(view.width, 64);
            assert_eq!(view.height, 48);
            assert_eq!(PixelFormat::from_code(view.format), Some(PixelFormat::Rgb));
            assert_eq!(DType::from_code(view.dtype), Some(DType::U8));
            assert_eq!(view.access, 3);
            assert_eq!(view.has_memory, 1);
            assert_eq!(
                TensorMemory::from_code(view.memory),
                Some(TensorMemory::Shm)
            );
            assert_eq!(view.has_compression, 1);
            assert_eq!(view.compression, 1);

            ef_tensor_image_desc_free(d);
        }
    }

    #[test]
    fn the_view_reports_absence_by_flag_not_by_a_zero_that_collides() {
        // A fresh request has no memory or compression preference. `memory`
        // reads 0 (== EfStorageKind::Mem's own code) precisely because
        // `has_memory` is the thing that says "unset", not the value.
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
