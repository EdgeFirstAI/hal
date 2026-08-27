// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image construction and identity-sharing sub-views:
//! `ef_tensor_image_alloc`, `ef_tensor_image_with_stride_alloc`,
//! `ef_tensor_view_region`, `ef_tensor_from_planes`.
//!
//! Every constructor here forwards to the real, already-implemented
//! `TensorDyn`/`Tensor<T>` method (`edgefirst-tensor`'s `static` backend,
//! the only backend this library ever ships): the ~250 lines of
//! platform-specific image geometry (macOS IOSurface tiling, Android
//! AHardwareBuffer, Linux DMA-BUF 64-byte pitch alignment, odd-dimension
//! chroma handling) live there, not here. Adding these exports does not
//! reimplement or approximate that geometry -- it makes the code that
//! already computes it correctly reachable from the dynamic backend.
//!
//! `ef_tensor_image_desc_alloc` (the `EfTensorImageDesc`-driven variant)
//! lives in `desc.rs`, next to the request type it consumes.
//!
//! # Naming
//!
//! [`ef_tensor_view_region`] is named that, not `ef_tensor_view`, because
//! `ef_tensor_view` already names a **struct** in `tensor.h`
//! (`EfTensorView`, the mapped-window scalar block `ef_tensor_map` fills) --
//! colliding with it would break every consumer including a C header at
//! all.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;

use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, Tensor, TensorDyn, TensorMemory};

use crate::handle::{impl_of, into_handle, tensor_of, EfTensor, EfTensorImpl};
use crate::last_error::{class_of, set_last_error, set_last_error_classified};

/// Parse a `NUL`-terminated `PixelFormat` wire string.
unsafe fn parse_format(f: *const c_char) -> Option<PixelFormat> {
    unsafe {
        if f.is_null() {
            return None;
        }
        PixelFormat::from_str_code(CStr::from_ptr(f).to_str().ok()?)
    }
}

/// Decode a 0..=3 CPU access code, `None` included (unlike
/// [`crate::codes::cpu_access_from_code`], which reserves 0 as "not a
/// mappable direction" -- image allocation legitimately wants
/// `CpuAccess::None`, e.g. a GPU-only render target).
fn parse_access(access: u32) -> Option<CpuAccess> {
    match access {
        0 => Some(CpuAccess::None),
        1 => Some(CpuAccess::Read),
        2 => Some(CpuAccess::Write),
        3 => Some(CpuAccess::ReadWrite),
        _ => None,
    }
}

/// Decode the `(has_memory, memory)` pair every `EfImageDescView`-adjacent
/// entry point uses for "a specific backing, or auto-select" -- `Err(())`
/// for an unrecognized `memory` code when `has_memory != 0`.
fn parse_memory(has_memory: c_int, memory: u32) -> Result<Option<TensorMemory>, ()> {
    if has_memory == 0 {
        return Ok(None);
    }
    TensorMemory::from_code(memory).map(Some).ok_or(())
}

/// Allocate an image tensor of `width` x `height` in `format`/`dtype`. See
/// [`edgefirst_tensor::TensorDyn::image`].
///
/// @retval a new tensor the caller must free with `ef_tensor_free`, on
///         success.
/// @retval `NULL` for a `NULL`/unrecognized `format`, an unknown `dtype` or
///         `memory` code, an unrecognized `access` code, or if the
///         underlying allocation fails (invalid `width`x`height` for
///         `format`, or the requested `memory` is unavailable) --
///         `ef_tensor_last_error_message` carries the reason.
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_alloc(
    width: usize,
    height: usize,
    format: *const c_char,
    dtype: u32,
    has_memory: c_int,
    memory: u32,
    access: u32,
) -> *mut EfTensor {
    unsafe {
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(fmt) = parse_format(format) else {
                set_last_error("image_alloc: null, non-UTF8, or unrecognized format string");
                return std::ptr::null_mut();
            };
            let Some(dt) = DType::from_code(dtype) else {
                set_last_error("image_alloc: unknown dtype code");
                return std::ptr::null_mut();
            };
            let Ok(mem) = parse_memory(has_memory, memory) else {
                set_last_error("image_alloc: unknown memory code");
                return std::ptr::null_mut();
            };
            let Some(acc) = parse_access(access) else {
                set_last_error("image_alloc: unknown access code");
                return std::ptr::null_mut();
            };
            match TensorDyn::image(width, height, fmt, dt, mem, acc) {
                Ok(t) => into_handle(t),
                Err(e) => {
                    set_last_error_classified(class_of(&e), &format!("image_alloc: {e}"));
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Allocate a DMA-backed image tensor with an explicit row stride, for a
/// pitch wider than the format's natural `width * channels * sizeof(dtype)`
/// (GPU pitch alignment). See
/// [`edgefirst_tensor::TensorDyn::image_with_stride`].
///
/// @retval a new tensor the caller must free with `ef_tensor_free`, on
///         success.
/// @retval `NULL` for the same argument reasons as [`ef_tensor_image_alloc`],
///         plus a `row_stride_bytes` smaller than the format's minimum row
///         size, a non-packed `format` (only packed layouts support a
///         padded stride), or non-DMA `memory` --
///         `ef_tensor_last_error_message` carries the reason.
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_image_with_stride_alloc(
    width: usize,
    height: usize,
    format: *const c_char,
    dtype: u32,
    row_stride_bytes: usize,
    has_memory: c_int,
    memory: u32,
    access: u32,
) -> *mut EfTensor {
    unsafe {
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(fmt) = parse_format(format) else {
                set_last_error(
                    "image_with_stride_alloc: null, non-UTF8, or unrecognized format string",
                );
                return std::ptr::null_mut();
            };
            let Some(dt) = DType::from_code(dtype) else {
                set_last_error("image_with_stride_alloc: unknown dtype code");
                return std::ptr::null_mut();
            };
            let Ok(mem) = parse_memory(has_memory, memory) else {
                set_last_error("image_with_stride_alloc: unknown memory code");
                return std::ptr::null_mut();
            };
            let Some(acc) = parse_access(access) else {
                set_last_error("image_with_stride_alloc: unknown access code");
                return std::ptr::null_mut();
            };
            match TensorDyn::image_with_stride(width, height, fmt, dt, row_stride_bytes, mem, acc) {
                Ok(t) => into_handle(t),
                Err(e) => {
                    set_last_error_classified(
                        class_of(&e),
                        &format!("image_with_stride_alloc: {e}"),
                    );
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Borrow a rectangular `(x, y, width, height)` pixel sub-region of `t` as a
/// new, independent handle that shares `t`'s underlying allocation and
/// identity (zero-copy) -- never a new allocation. See
/// [`edgefirst_tensor::TensorDyn::view`].
///
/// @retval a new tensor the caller must free with `ef_tensor_free`
///         (independently of `t`; both stay valid, sharing the same
///         backing), on success.
/// @retval `NULL` for a `NULL`/invalid `t`, or a region that does not fit
///         within `t`'s frame -- `ef_tensor_last_error_message` carries the
///         reason.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_view_region(
    t: *const EfTensor,
    x: u64,
    y: u64,
    width: u64,
    height: u64,
) -> *mut EfTensor {
    // The quiet hook, before the catch: a caught panic must WRITE the
    // thread-local, or a consumer reading `ef_tensor_last_error_class`
    // after this returns NULL gets a class left behind by an earlier
    // failure and reports it as this call's. See `ensure_hook_installed`.
    crate::last_error::ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        if t.is_null() {
            set_last_error("view_region: null tensor");
            return std::ptr::null_mut();
        }
        let Some(inner) = tensor_of(t) else {
            set_last_error("view_region: could not resolve handle");
            return std::ptr::null_mut();
        };
        let region = Region::new(x as usize, y as usize, width as usize, height as usize);
        match inner.view(region) {
            Ok(v) => into_handle(v),
            Err(e) => {
                set_last_error_classified(class_of(&e), &format!("view_region: {e}"));
                std::ptr::null_mut()
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Borrow batch element `n` of a batched tensor (leading `N` dimension) as a
/// new, independent handle that shares `t`'s underlying allocation and
/// identity (zero-copy) -- never a new allocation. See
/// [`edgefirst_tensor::TensorDyn::batch`].
///
/// Distinct from `ef_tensor_view_region`, which crops a *spatial*
/// rectangle within one image: this indexes the leading dimension, and the
/// result has `t`'s shape with that dimension dropped. A tensor whose
/// leading dimension is not a batch axis has no meaningful element `n`, and
/// the underlying call refuses.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`
///         (independently of `t`; both stay valid, sharing the same
///         backing), on success.
/// @retval `NULL` for a `NULL`/invalid `t`, or an `n` outside the leading
///         dimension -- `ef_tensor_last_error_message` carries the reason.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_batch(t: *const EfTensor, n: u64) -> *mut EfTensor {
    // The quiet hook, before the catch: a caught panic must WRITE the
    // thread-local, or a consumer reading `ef_tensor_last_error_class`
    // after this returns NULL gets a class left behind by an earlier
    // failure and reports it as this call's. See `ensure_hook_installed`.
    crate::last_error::ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        if t.is_null() {
            set_last_error("batch: null tensor");
            return std::ptr::null_mut();
        }
        let Some(inner) = tensor_of(t) else {
            set_last_error("batch: could not resolve handle");
            return std::ptr::null_mut();
        };
        // A 64-bit index on a 32-bit host cannot address an element that
        // exists, and truncating would silently return a *different*
        // element's view -- refuse instead.
        let Ok(n) = usize::try_from(n) else {
            set_last_error("batch: index is out of range for this host's usize");
            return std::ptr::null_mut();
        };
        match inner.batch(n) {
            Ok(v) => into_handle(v),
            Err(e) => {
                set_last_error_classified(class_of(&e), &format!("batch: {e}"));
                std::ptr::null_mut()
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Combine separate luma and chroma plane tensors into one semi-planar
/// (NV12/NV16) tensor. See [`edgefirst_tensor::Tensor::from_planes`].
///
/// **Ownership: consumes both `luma` and `chroma`, but only past this
/// function's precondition checks.** A real Rust caller of
/// `Tensor::<T>::from_planes` always passes both by value, with no "give
/// them back on error" path -- but every precondition that call requires
/// (matching element types chief among them, since the C boundary is
/// type-erased and the compiler cannot enforce it the way it does for a
/// real Rust caller) is checked here *first*, before either handle is
/// reclaimed. Consuming happens once, right before the underlying
/// `Tensor::from_planes` call; from that point on both outcomes (success or
/// a validation failure *inside* `Tensor::from_planes`, e.g. an
/// incompatible format/shape) leave `luma`/`chroma` invalidated, matching
/// Rust's own by-value semantics. See the `@retval` list below for exactly
/// which failures consume and which do not.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`, on
///         success (`luma`/`chroma` consumed).
/// @retval `NULL`, **`luma`/`chroma` left valid and unconsumed** -- a
///         `NULL`/invalid `luma` or `chroma`, an unrecognized `format`, a
///         `luma`/`chroma` element-type mismatch, or an
///         outstanding `ef_tensor_retain`/`ef_tensor_map` on either handle
///         (consuming a handle another reference still points at, or that
///         has a live map guard, would dangle it).
/// @retval `NULL`, **`luma`/`chroma` consumed regardless** -- every
///         precondition above passed but `Tensor::from_planes` itself
///         refused (see its constraints: only NV12/NV16, matching
///         luma/chroma widths, and the format-specific height ratio).
///
/// Every `NULL` case sets `ef_tensor_last_error_message` with the reason.
///
/// # Safety
/// `luma` and `chroma` must each be `NULL` or a live handle; `format` must
/// be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_from_planes(
    luma: *mut EfTensor,
    chroma: *mut EfTensor,
    format: *const c_char,
) -> *mut EfTensor {
    unsafe {
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            if luma.is_null() || chroma.is_null() {
                set_last_error("from_planes: null or invalid luma/chroma handle");
                return std::ptr::null_mut();
            }
            let Some(fmt) = parse_format(format) else {
                set_last_error("from_planes: null, non-UTF8, or unrecognized format string");
                return std::ptr::null_mut();
            };
            let (Some(limp), Some(cimp)) = (impl_of(luma), impl_of(chroma)) else {
                set_last_error("from_planes: could not resolve luma/chroma handle");
                return std::ptr::null_mut();
            };
            // Precondition, checked before consuming anything: neither handle
            // may have an outstanding retain or map. This function reclaims
            // both `Box<EfTensorImpl>`s outright; doing that while another
            // reference (a retain, or a live map guard's keepalive) still
            // points at the same allocation would dangle it, the same hazard
            // `ef_tensor_map`'s own exclusive-write gate (`map.rs`) guards
            // against for a narrower case.
            if limp.refs.load(Ordering::Acquire) != 1 || cimp.refs.load(Ordering::Acquire) != 1 {
                set_last_error(
                    "from_planes: luma and chroma must each have no outstanding ef_tensor_retain",
                );
                return std::ptr::null_mut();
            }
            let outstanding_map = |imp: &EfTensorImpl| {
                imp.map_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            };
            if outstanding_map(limp) || outstanding_map(cimp) {
                set_last_error(
                    "from_planes: luma and chroma must each have no outstanding ef_tensor_map",
                );
                return std::ptr::null_mut();
            }
            // Another precondition checked before consuming: a real Rust caller
            // of `Tensor::<T>::from_planes` can never pass mismatched element
            // types (the compiler enforces `T` is the same for both arguments);
            // that guarantee does not exist at this type-erased C boundary, so
            // it is checked explicitly here, before either handle is reclaimed
            // -- not inside the match below, where by then both would already
            // be gone and a caller "retrying" with the free'd pointers would be
            // a use-after-free.
            if limp.inner.dtype() != cimp.inner.dtype() {
                set_last_error(
                    "from_planes: luma and chroma tensors must have the same element type",
                );
                return std::ptr::null_mut();
            }

            // Consume: from this point on `luma`/`chroma` are invalidated on
            // every path, matching Rust's own `from_planes(luma, chroma, ..)`
            // taking both by value regardless of outcome (see this function's
            // Doxygen).
            let luma_box = Box::from_raw(luma as *mut EfTensorImpl);
            let chroma_box = Box::from_raw(chroma as *mut EfTensorImpl);
            let result = match (luma_box.inner, chroma_box.inner) {
                (TensorDyn::U8(l), TensorDyn::U8(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::I8(l), TensorDyn::I8(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::U16(l), TensorDyn::U16(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::I16(l), TensorDyn::I16(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::U32(l), TensorDyn::U32(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::I32(l), TensorDyn::I32(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::U64(l), TensorDyn::U64(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::I64(l), TensorDyn::I64(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::F16(l), TensorDyn::F16(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::F32(l), TensorDyn::F32(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                (TensorDyn::F64(l), TensorDyn::F64(c)) => {
                    Tensor::from_planes(l, c, fmt).map(TensorDyn::from)
                }
                _ => Err(edgefirst_tensor::Error::InvalidArgument(
                    "from_planes: luma and chroma tensors must have the same element type".into(),
                )),
            };
            match result {
                Ok(t) => into_handle(t),
                Err(e) => {
                    set_last_error_classified(class_of(&e), &format!("from_planes: {e}"));
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
    use crate::handle::{ef_tensor_free, ef_tensor_new, inner_of};

    fn u8_c(s: &str) -> std::ffi::CString {
        std::ffi::CString::new(s).unwrap()
    }

    #[test]
    fn image_alloc_allocates_a_real_image_tensor() {
        let fmt = u8_c("mono8");
        let t = unsafe { ef_tensor_image_alloc(64, 48, fmt.as_ptr(), 0, 0, 0, 0) }; // U8, auto memory, CpuAccess::None
        if t.is_null() {
            let msg = unsafe {
                std::ffi::CStr::from_ptr(crate::last_error::ef_tensor_last_error_message())
            };
            panic!("image_alloc failed: {}", msg.to_str().unwrap());
        }
        assert_eq!(inner_of(t).format(), Some(PixelFormat::Grey));
        assert_eq!(inner_of(t).width(), Some(64));
        assert_eq!(inner_of(t).height(), Some(48));
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn image_alloc_rejects_unknown_format() {
        let fmt = u8_c("not-a-format");
        let t = unsafe { ef_tensor_image_alloc(64, 48, fmt.as_ptr(), 0, 0, 0, 0) };
        assert!(t.is_null());
    }

    #[test]
    fn image_with_stride_alloc_reports_the_padded_pitch() {
        if !edgefirst_tensor::is_dma_available() {
            eprintln!("SKIPPED: DMA not available on this host");
            return;
        }
        let fmt = u8_c("Rgba");
        // Natural pitch is 64*4=256; pad to 320.
        let t = unsafe { ef_tensor_image_with_stride_alloc(64, 64, fmt.as_ptr(), 0, 320, 1, 2, 3) };
        if t.is_null() {
            eprintln!("SKIPPED: image_with_stride_alloc unavailable on this host");
            return;
        }
        assert_eq!(inner_of(t).effective_row_stride(), Some(320));
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn view_region_shares_identity_with_its_parent() {
        let dims = [8u64, 8, 1]; // [H, W, C] -- Grey/mono8 is a packed format
        let parent = unsafe { ef_tensor_new(0, dims.as_ptr(), 3) };
        let grey = u8_c("mono8");
        assert_eq!(
            unsafe { crate::mutate::ef_tensor_set_format(parent, grey.as_ptr()) },
            0
        );
        let view = unsafe { ef_tensor_view_region(parent, 0, 0, 4, 4) };
        assert!(!view.is_null());
        assert_eq!(
            inner_of(parent).buffer_identity().id(),
            inner_of(view).buffer_identity().id(),
            "a view must share its parent's BufferIdentity, never mint a fresh one"
        );
        unsafe {
            ef_tensor_free(view);
            ef_tensor_free(parent);
        }
    }

    #[test]
    fn view_region_out_of_bounds_is_rejected() {
        let dims = [4u64, 4, 1];
        let parent = unsafe { ef_tensor_new(0, dims.as_ptr(), 3) };
        let grey = u8_c("mono8");
        unsafe { crate::mutate::ef_tensor_set_format(parent, grey.as_ptr()) };
        let view = unsafe { ef_tensor_view_region(parent, 0, 0, 100, 100) };
        assert!(view.is_null());
        unsafe { ef_tensor_free(parent) };
    }

    #[test]
    fn from_planes_combines_two_own_mint_handles() {
        // NV12: chroma height is luma height / 2, same width.
        let luma_dims = [8u64, 8];
        let chroma_dims = [4u64, 8];
        let luma = unsafe { ef_tensor_new(0, luma_dims.as_ptr(), 2) };
        let chroma = unsafe { ef_tensor_new(0, chroma_dims.as_ptr(), 2) };
        let nv12 = u8_c("NV12");
        let combined = unsafe { ef_tensor_from_planes(luma, chroma, nv12.as_ptr()) };
        assert!(!combined.is_null());
        assert_eq!(inner_of(combined).format(), Some(PixelFormat::Nv12));
        unsafe { ef_tensor_free(combined) };
    }

    #[test]
    fn from_planes_rejects_a_retained_luma() {
        let luma_dims = [8u64, 8];
        let chroma_dims = [4u64, 8];
        let luma = unsafe { ef_tensor_new(0, luma_dims.as_ptr(), 2) };
        let chroma = unsafe { ef_tensor_new(0, chroma_dims.as_ptr(), 2) };
        assert_eq!(unsafe { crate::handle::ef_tensor_retain(luma) }, 0);
        let nv12 = u8_c("NV12");
        let combined = unsafe { ef_tensor_from_planes(luma, chroma, nv12.as_ptr()) };
        assert!(combined.is_null(), "a retained luma must not be consumed");
        // Both handles remain valid: release the retain, then free normally.
        unsafe {
            ef_tensor_free(luma);
            ef_tensor_free(luma);
            ef_tensor_free(chroma);
        }
    }

    #[test]
    fn from_planes_rejects_mismatched_element_types() {
        let luma_dims = [8u64, 8];
        let chroma_dims = [4u64, 8];
        let luma = unsafe { ef_tensor_new(0, luma_dims.as_ptr(), 2) }; // U8
        let chroma = unsafe { ef_tensor_new(1, chroma_dims.as_ptr(), 2) }; // I8
        let nv12 = u8_c("NV12");
        let combined = unsafe { ef_tensor_from_planes(luma, chroma, nv12.as_ptr()) };
        assert!(combined.is_null());
        unsafe {
            ef_tensor_free(luma);
            ef_tensor_free(chroma);
        }
    }
}
