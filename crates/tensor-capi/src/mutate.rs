// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Live-handle geometry mutators: `ef_tensor_set_format`,
//! `ef_tensor_set_row_stride`, `ef_tensor_set_row_stride_unchecked`,
//! `ef_tensor_set_plane_offset`, `ef_tensor_configure_image` -- plus
//! `ef_tensor_plane_offset`, the reader [`ef_tensor_set_plane_offset`] had
//! none of, found missing while proving the setter (task 15):
//! `Tensor::plane_offset()` reads it back from `ef_tensor_plane_at`'s
//! plane-0 geometry, but that geometry is always 0 for plane 0 by
//! construction (`plane_table` computes *intra-buffer* layout, not the
//! DMA-BUF-level start offset `set_plane_offset` writes), so the existing
//! reader silently reported the wrong value for any `dynamic` tensor with a
//! real offset -- exactly the "answer that looks plausible but is not the
//! real one" this whole task exists to close, so it is fixed alongside the
//! family it belongs to rather than left for a future task to rediscover.
//!
//! `ef_tensor_set_row_stride_unchecked` (task 17) is the same kind of gap,
//! found while auditing predicates that mean different things across
//! backends: `Tensor::set_row_stride_unchecked` (`static`'s raw,
//! format-independent stride setter, needed for a multiplane chroma
//! sub-tensor which by contract carries no format) had a `dynamic`-side
//! stub that only ever panicked, because no primitive backed it. Adding one
//! also required fixing `ef_tensor_plane_at`'s formatless-tensor fallback
//! (see its own doc comment) -- otherwise the newly-real setter would
//! silently write a stride `ef_tensor_plane_at`/`effective_row_stride`
//! continued to ignore.
//!
//! Each of these is a thin wrapper over the real, already-implemented
//! `TensorDyn` method of the same name (`edgefirst-tensor`'s `static`
//! backend, the only backend this library is ever built with) -- the
//! geometry, alignment, and shape-validation logic all live there, not
//! here. What this file adds is the C-ABI plumbing: null checks,
//! `PixelFormat` string parsing, errno translation, and refreshing
//! [`crate::handle::EfTensorImpl`]'s cached `shape_u64`/`strides_i64`/
//! `format_c` afterward (they are derived from `inner` once at construction
//! and would otherwise go stale the moment one of these setters changes it).
//!
//! # Concurrency: narrower than `ef_tensor_set_colorimetry`
//!
//! Every setter here goes through [`crate::handle::imp_mut`], whose doc
//! comment is the authority; read it before adding another one. In short:
//! unlike colorimetry (an `AtomicU32`, safe under concurrent access from any
//! thread by construction), these mutate `inner`'s own fields directly, so
//! **the caller must ensure no other thread is calling any `tensor-capi`
//! function on the same handle while one of these is in flight.** Every real
//! caller in this workspace already fits that shape: `import_image` calls
//! `set_format`/`set_row_stride`/`set_plane_offset` once, immediately after
//! wrapping a freshly-imported fd, before the tensor is retained or shared;
//! `configure_image`'s pool-reuse callers reconfigure a slot they alone hold
//! between frames.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, PixelFormat};

use crate::handle::{imp_mut, refresh_caches, tensor_of, EfTensor};
use crate::last_error::{set_last_error, shield_int};
use crate::map::errno_for;

/// Parse a `NUL`-terminated `PixelFormat` wire string, `NULL`/invalid UTF-8/
/// unrecognized all folding to `None` -- the caller turns that into `EINVAL`.
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
///         `ef_tensor_plane_count` / `ef_tensor_plane_at` now reflect it.
/// @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
///         the tensor's current shape does not match the format's layout
///         (packed expects `[H, W, C]`, planar `[C, H, W]`, semi-planar
///         `[H*k, W]`).
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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
/// on the same handle from another thread -- see this file's module docs.
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
/// Unlike [`ef_tensor_set_row_stride`], this never requires a pixel format
/// and never validates `stride` against one -- for a raw sub-tensor that by
/// contract carries no format (the multiplane chroma plane
/// [`crate::image::ef_tensor_from_planes`] combines, or the standalone plane
/// tensors `ef_tensor_builder_wrap` produces before a format is attached),
/// there is no minimum to check the caller's stride against. Same escape
/// hatch as `static`'s `Tensor::set_row_stride_unchecked` (`lib.rs`): the
/// caller is responsible for the stride being valid for whatever it goes on
/// to describe.
///
/// Updates `ef_tensor_plane_at`'s cached geometry the same way
/// [`ef_tensor_set_row_stride`] does (`refresh_caches`) -- unlike that
/// setter, a formatless tensor's plane-0 geometry falls back to reporting
/// this stride directly (see `ef_tensor_plane_at`'s own doc comment for why
/// that fallback exists).
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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
/// Does not touch `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`'s
/// cached values -- `plane_offset` is read live from `inner` by
/// `ef_tensor_plane_at`, not cached, so no refresh is needed here (contrast
/// [`ef_tensor_set_format`]/[`ef_tensor_set_row_stride`]).
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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
/// `-1` is a genuine sentinel here, not a presence-flag substitute: a byte
/// offset can never be negative, so it is unambiguous, matching
/// `ef_tensor_plane`'s own `handle: -1` convention for "none" -- unlike
/// axis or scale (where every value including 0 is legitimate), there is no
/// real offset this collides with.
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
/// [`ef_tensor_set_logical_shape`] for the capacity-based sibling a pool
/// tensor needs.
///
/// @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides` now reflect the
///         new geometry.
/// @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
///         out of range for this host.
/// @retval ERANGE the new shape's element count differs from the current
///         one -- the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `dims` must point to `ndim`
/// readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_reshape(t: *mut EfTensor, dims: *const u64, ndim: u32) -> c_int {
    unsafe { reshape_impl(t, dims, ndim, false) }
}

/// Change a tensor's logical shape to anything its **allocation** can hold.
///
/// The capacity-based counterpart to [`ef_tensor_reshape`]
/// (`TensorTrait::set_logical_shape`): an oversized reusable pool tensor
/// reconfigured to a smaller image without reallocating, which
/// `ef_tensor_reshape`'s equal-count rule refuses.
///
/// Two entry points rather than one with a flag, because they are two
/// contracts a caller picks between deliberately -- the same reason
/// `ef_tensor_sync_for_cpu` and `_sync_for_device` are two entries.
///
/// Task P2b wrote this, found nothing called it, and **deleted it before
/// committing** -- unreferenced ABI surface drifts unwatched. Task P2e gave
/// it a caller: `Tensor<T>` never overrode
/// `TensorTrait::set_logical_shape`, so both backends silently applied
/// `reshape`'s strict rule under a name promising the opposite, and fixing
/// that on the `dynamic` side needs exactly this primitive. Added back on
/// the strength of the caller, not of the idea.
///
/// @retval 0 success.
/// @retval EINVAL `t` or `dims` is `NULL`, `ndim` is 0, or a dimension is
///         out of range for this host.
/// @retval ERANGE the new shape does not fit the existing allocation --
///         the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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

/// The body of [`ef_tensor_reshape`], kept separate only so the entry point
/// itself stays a one-liner alongside its siblings in this file.
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
                    // `ef_tensor_shape`/`_strides` read the cached copies,
                    // not `inner` live, so a reshape that did not refresh
                    // them would report the OLD geometry immediately after
                    // succeeding.
                    refresh_caches(imp);
                    0
                }
                Err(e) => {
                    // ERANGE for "this shape does not fit", EINVAL for a
                    // malformed argument -- chosen here rather than taken
                    // from `errno_for`, which maps `ShapeMismatch` and
                    // `InvalidShape` alike to EINVAL and so cannot tell the
                    // caller which kind of refusal this was. The errno is
                    // the only signal the ABI carries (the message is
                    // advisory), so it has to make the one distinction a
                    // caller acts on.
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

/// Duplicate the file descriptor backing this tensor, for any storage kind
/// that has one.
///
/// Deliberately **not** derivable from `ef_tensor_plane_at`: that reports a
/// plane's *native handle*, which is a dma-buf fd on Linux and an IOSurface
/// id on Apple and `-1` for everything else -- so a consumer deriving
/// "clone this tensor's fd" from it refuses SHM-backed tensors, which do
/// have a real fd. The library owns the storage and knows which kinds have
/// one; asking it is the whole point of the split.
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
                return -libc::ENOTSUP;
            }
            #[cfg(unix)]
            match inner.clone_fd() {
                // `into_raw_fd` hands ownership to the C caller, who closes
                // it. Returning the fd itself (not an out-param) matches
                // POSIX `dup`, which is what a C caller expects here.
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
/// consumer reads them. `edgefirst-image` allocates a PBO or DMA buffer as
/// `u8` and hands it back as `i8`, with the int8 shader applying an XOR 0x80
/// bias over the same buffer -- this is the primitive that makes the handle
/// agree with that.
///
/// A dtype of a different width is refused. That is not a retag but a
/// reinterpretation: `ef_tensor_shape` times the element width is what a
/// consumer multiplies out, so widening or narrowing here would silently
/// change the element count over an allocation whose size did not change.
///
/// @retval 0 success; `ef_tensor_dtype` now reports `dtype`.
/// @retval EINVAL `t` is `NULL`, or `dtype` is not a recognized code.
/// @retval ERANGE `dtype` has a different element width than the current one
///         -- the tensor is left untouched.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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
                    // `derive_caches` computes `strides_i64` from the element
                    // width. A same-width retag leaves them identical, so this
                    // is belt-and-braces rather than load-bearing today --
                    // but the caches are derived state and every other
                    // mutator in this file refreshes them, so not doing it
                    // here would be the odd one out for a reader to trip on.
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
/// its existing allocation -- the pool-reuse primitive a JPEG
/// decode-into-pool destination tensor needs before each decode. See
/// [`edgefirst_tensor::TensorDyn::configure_image`].
///
/// @retval 0 success; `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format`
///         /`ef_tensor_plane_at` now reflect the new geometry.
/// @retval EINVAL `t` is `NULL`, `format` is `NULL`/non-UTF8/unrecognized, or
///         `width`x`height` is not a valid size for `format`.
/// @retval ERANGE the existing allocation cannot hold `width`x`height` in
///         `format` (a pool tensor sized too small for this request).
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this file's module docs.
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
        // Cached format string must have been refreshed.
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
        // Row (dim 0) stride is now the padded 128 bytes, not the tight 64.
        assert_eq!(unsafe { *strides }, 128);
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_plane_offset_is_read_back_through_ef_tensor_plane_offset() {
        // `ef_tensor_plane_at`'s `offset` is `plane_table`'s intra-buffer
        // plane layout (e.g. where NV12's chroma plane starts within the
        // combined buffer) -- a different concept from the DMA-BUF-level
        // offset this setter writes. `ef_tensor_plane_offset` is the
        // dedicated reader for that; `inner_of(t).plane_offset()` proves
        // the Rust-level state directly as a cross-check.
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
        // Oversized pool buffer, reconfigured down to a smaller NV12 frame.
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
        // Unlike `ef_tensor_set_row_stride`, a raw formatless tensor (the
        // shape a multiplane chroma plane has by contract -- see
        // `Tensor::from_planes`) accepts an unchecked stride directly.
        let dims = [24u64, 64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        assert_eq!(inner_of(t).format(), None);
        assert_eq!(unsafe { ef_tensor_set_row_stride_unchecked(t, 96) }, 0);
        assert_eq!(inner_of(t).row_stride(), Some(96));
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn set_row_stride_unchecked_is_read_back_through_plane_at() {
        // `ef_tensor_plane_at`'s formatless fallback used to report the
        // *whole allocation's* byte count as plane 0's stride unconditionally
        // (`capacity_bytes()`), because no primitive could ever set a
        // formatless tensor's `row_stride` before this task -- once one
        // does, the fallback must prefer it, or a caller reading plane
        // geometry back would see the wrong pitch. See `ef_tensor_plane_at`'s
        // own doc comment.
        let dims = [24u64, 64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 2) };
        let mut plane = EfTensorPlane::default();
        assert_eq!(unsafe { ef_tensor_plane_at(t, 0, &mut plane) }, 0);
        let capacity = plane.stride; // no explicit stride yet: whole-buffer fallback
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
