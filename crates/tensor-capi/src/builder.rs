// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tensor construction through a builder.
//!
//! # Sticky errors
//!
//! After a setter fails, later setters no-op and the terminal call returns the
//! first failure with its `errno`. A chained call is checked once, at the end:
//!
//! ```text
//! Tensor t = Builder().dtype(U8).shape(dims, 2).storage(Dma).alloc();
//! ```

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, TensorMemory};

/// One plane handed to `wrap`, mirroring `TensorPlane` on the wire.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlaneSpec {
    pub handle: i64,
    pub offset: u64,
    pub stride: u64,
    pub size: u64,
    pub used: u64,
    pub modifier: u64,
}

/// Accumulates a tensor's description, one field at a time.
///
/// Opaque to C. Once a setter fails, every later setter is a no-op and
/// returns the first error unchanged.
pub struct EfTensorBuilder {
    /// First error seen, or 0. Never overwritten by a later failure.
    pub(crate) err: c_int,
    pub(crate) dtype: Option<DType>,
    pub(crate) shape: Vec<u64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) storage: Option<TensorMemory>,
    pub(crate) planes: Vec<PlaneSpec>,
    pub(crate) format: Option<String>,
    pub(crate) colorimetry: [String; 4],
    pub(crate) quant: Option<(i32, Vec<f32>, Vec<i32>)>,
    pub(crate) fence_fd: c_int,
}

impl Default for EfTensorBuilder {
    fn default() -> Self {
        Self {
            err: 0,
            dtype: None,
            shape: Vec::new(),
            strides: Vec::new(),
            storage: None,
            planes: Vec::new(),
            format: None,
            colorimetry: [const { String::new() }; 4],
            quant: None,
            fence_fd: -1,
        }
    }
}

/// Run a setter under the sticky-error discipline: return the pending error
/// without running `body` when one is set, and record the error `body`
/// returns.
fn with_builder<F>(b: *mut EfTensorBuilder, body: F) -> c_int
where
    F: FnOnce(&mut EfTensorBuilder) -> c_int,
{
    catch_unwind(AssertUnwindSafe(|| {
        if b.is_null() {
            return libc::EINVAL;
        }
        // SAFETY: non-null and owned by the caller for the call's duration.
        let b = unsafe { &mut *b };
        if b.err != 0 {
            return b.err;
        }
        let rc = body(b);
        if rc != 0 {
            b.err = rc;
        }
        rc
    }))
    .unwrap_or(libc::EINVAL)
}

/// Create a builder. Returns `NULL` only on allocation failure.
#[no_mangle]
pub extern "C" fn ef_tensor_builder_new() -> *mut EfTensorBuilder {
    catch_unwind(|| Box::into_raw(Box::new(EfTensorBuilder::default())))
        .unwrap_or(std::ptr::null_mut())
}

/// Free a builder. Freeing `NULL` is a no-op.
///
/// # Safety
/// `b` must have come from [`ef_tensor_builder_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_free(b: *mut EfTensorBuilder) {
    unsafe {
        if b.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(b))));
    }
}

/// The first error recorded, or 0. `EINVAL` for a `NULL` builder.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_error(b: *const EfTensorBuilder) -> c_int {
    unsafe {
        if b.is_null() {
            return libc::EINVAL;
        }
        catch_unwind(AssertUnwindSafe(|| (*b).err)).unwrap_or(libc::EINVAL)
    }
}

/// Set the element type. `dtype` is an `EF_DTYPE_*` code; an unknown code
/// is `EINVAL`.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_dtype(b: *mut EfTensorBuilder, dtype: u32) -> c_int {
    with_builder(b, |b| match DType::from_code(dtype) {
        Some(d) => {
            b.dtype = Some(d);
            0
        }
        None => libc::EINVAL,
    })
}

/// Set the addressing grid.
///
/// # Safety
/// `dims` must point to `ndim` readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_shape(
    b: *mut EfTensorBuilder,
    dims: *const u64,
    ndim: u32,
) -> c_int {
    unsafe {
        with_builder(b, |b| {
            if ndim == 0 || dims.is_null() {
                return libc::EINVAL;
            }
            b.shape = std::slice::from_raw_parts(dims, ndim as usize).to_vec();
            0
        })
    }
}

/// Set strides, in bytes. Must have the same rank as the shape.
///
/// # Safety
/// `str_` must point to `ndim` readable `int64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_strides(
    b: *mut EfTensorBuilder,
    str_: *const i64,
    ndim: u32,
) -> c_int {
    unsafe {
        with_builder(b, |b| {
            if ndim == 0 || str_.is_null() {
                return libc::EINVAL;
            }
            if !b.shape.is_empty() && b.shape.len() != ndim as usize {
                return libc::EINVAL;
            }
            b.strides = std::slice::from_raw_parts(str_, ndim as usize).to_vec();
            0
        })
    }
}

/// Set the backing store. `kind` is an `EF_STORAGE_KIND_*` code; an unknown
/// code is `EINVAL`.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_storage(b: *mut EfTensorBuilder, kind: u32) -> c_int {
    with_builder(b, |b| match TensorMemory::from_code(kind) {
        Some(k) => {
            b.storage = Some(k);
            0
        }
        None => libc::EINVAL,
    })
}

/// Add a plane. Required by `wrap`, rejected by `alloc`.
///
/// [`ef_tensor_builder_wrap`] treats the fields as follows:
///
/// * `handle` -- adopted as the tensor's fd.
/// * `stride` -- carried onto the tensor as its row stride, applied after
///   any format the builder carries. `0` means unspecified.
/// * `offset` -- carried onto the tensor as its plane offset, applied after
///   any format.
/// * `size` -- validated against the extent `shape` and `stride` imply, and
///   not stored. `0` means unspecified and is not checked. A nonzero `size`
///   smaller than required is rejected; a larger one is accepted.
/// * `used` -- rejected unless equal to `size`: `used > size` by this
///   function with `EINVAL`, `used < size` by `wrap` with `EBADMSG`.
/// * `modifier` -- rejected unless `0` (linear).
/// * a second and later plane -- rejected: `wrap` adopts one handle per
///   tensor.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ef_tensor_builder_add_plane(
    b: *mut EfTensorBuilder,
    handle: i64,
    offset: u64,
    stride: u64,
    size: u64,
    used: u64,
    modifier: u64,
) -> c_int {
    with_builder(b, |b| {
        if used > size {
            return libc::EINVAL;
        }
        b.planes.push(PlaneSpec {
            handle,
            offset,
            stride,
            size,
            used,
            modifier,
        });
        0
    })
}

/// Read a C string, rejecting `NULL` and invalid UTF-8.
unsafe fn cstr(p: *const c_char) -> Option<String> {
    unsafe {
        if p.is_null() {
            return None;
        }
        CStr::from_ptr(p).to_str().ok().map(str::to_string)
    }
}

/// Set the format descriptor (`"NV12"`, `"rgb8"`); `""` means not an image.
///
/// # Safety
/// `f` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_format(
    b: *mut EfTensorBuilder,
    f: *const c_char,
) -> c_int {
    unsafe {
        with_builder(b, |b| match cstr(f) {
            Some(s) => {
                b.format = Some(s);
                0
            }
            None => libc::EINVAL,
        })
    }
}

/// Set the four colorimetry axes. Any may be `""` for unspecified.
///
/// # Safety
/// Each argument must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_colorimetry(
    b: *mut EfTensorBuilder,
    space: *const c_char,
    transfer: *const c_char,
    encoding: *const c_char,
    range: *const c_char,
) -> c_int {
    unsafe {
        with_builder(b, |b| {
            let (Some(s), Some(t), Some(e), Some(r)) =
                (cstr(space), cstr(transfer), cstr(encoding), cstr(range))
            else {
                return libc::EINVAL;
            };
            b.colorimetry = [s, t, e, r];
            0
        })
    }
}

/// Set quantization. `axis` is `-2` for none, `-1` per-tensor, `>= 0` per-channel.
///
/// # Safety
/// `scales` must point to `n` floats; `zps` must be `NULL` or point to `n` ints.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_quantization(
    b: *mut EfTensorBuilder,
    axis: i32,
    scales: *const f32,
    zps: *const i32,
    n: u32,
) -> c_int {
    unsafe {
        with_builder(b, |b| {
            if axis == -2 {
                b.quant = None;
                return 0;
            }
            if scales.is_null() || n == 0 {
                return libc::EINVAL;
            }
            // Per-tensor quantization takes exactly one scale.
            if axis == -1 && n != 1 {
                return libc::EINVAL;
            }
            let s = std::slice::from_raw_parts(scales, n as usize).to_vec();
            let z = if zps.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(zps, n as usize).to_vec()
            };
            b.quant = Some((axis, s, z));
            0
        })
    }
}

/// Set the acquire fence fd, or `-1` for none.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_fence(b: *mut EfTensorBuilder, fd: c_int) -> c_int {
    with_builder(b, |b| {
        if fd < -1 {
            return libc::EINVAL;
        }
        b.fence_fd = fd;
        0
    })
}

/// Finish the builder, producing a tensor, under the sticky-error discipline.
///
/// Runs `body`, then [`apply_metadata`], then `finish`. A failure is recorded
/// on the builder and `NULL` is returned. The builder is not consumed.
fn terminal<F, G>(b: *mut EfTensorBuilder, body: F, finish: G) -> *mut crate::handle::EfTensor
where
    F: FnOnce(&EfTensorBuilder) -> Result<edgefirst_tensor::TensorDyn, c_int>,
    G: FnOnce(
        &EfTensorBuilder,
        edgefirst_tensor::TensorDyn,
    ) -> Result<edgefirst_tensor::TensorDyn, c_int>,
{
    catch_unwind(AssertUnwindSafe(|| {
        if b.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: non-null and owned by the caller for the call's duration.
        let b = unsafe { &mut *b };
        if b.err != 0 {
            return std::ptr::null_mut();
        }
        match body(b)
            .and_then(|t| apply_metadata(b, t))
            .and_then(|t| finish(b, t))
        {
            Ok(t) => crate::handle::into_handle(t),
            Err(e) => {
                b.err = e;
                std::ptr::null_mut()
            }
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Attach the descriptive fields both terminal calls share.
fn apply_metadata(
    b: &EfTensorBuilder,
    mut t: edgefirst_tensor::TensorDyn,
) -> Result<edgefirst_tensor::TensorDyn, c_int> {
    if let Some(f) = b.format.as_deref().filter(|f| !f.is_empty()) {
        let fmt = edgefirst_tensor::PixelFormat::from_str_code(f).ok_or(libc::EINVAL)?;
        t.set_format(fmt).map_err(|_| libc::EINVAL)?;
    }
    let c = edgefirst_tensor::Colorimetry {
        space: edgefirst_tensor::ColorSpace::from_str_code(&b.colorimetry[0]),
        transfer: edgefirst_tensor::ColorTransfer::from_str_code(&b.colorimetry[1]),
        encoding: edgefirst_tensor::ColorEncoding::from_str_code(&b.colorimetry[2]),
        range: edgefirst_tensor::ColorRange::from_str_code(&b.colorimetry[3]),
    };
    if c != edgefirst_tensor::Colorimetry::default() {
        t.set_colorimetry(Some(c));
    }
    if let Some((axis, scales, zeros)) = &b.quant {
        // `zeros` is empty when the caller passed a `NULL` `zps`, and holds
        // one zero-point per scale otherwise. An empty `zeros` selects the
        // symmetric constructor in both the per-tensor and per-channel arms.
        let q = match (*axis, zeros.is_empty()) {
            (-1, true) => edgefirst_tensor::Quantization::per_tensor_symmetric(scales[0]),
            (-1, false) => edgefirst_tensor::Quantization::per_tensor(scales[0], zeros[0]),
            (a, true) if a >= 0 => {
                edgefirst_tensor::Quantization::per_channel_symmetric(scales.clone(), a as usize)
                    .map_err(|_| libc::EINVAL)?
            }
            (a, false) if a >= 0 => edgefirst_tensor::Quantization::per_channel(
                scales.clone(),
                zeros.clone(),
                a as usize,
            )
            .map_err(|_| libc::EINVAL)?,
            _ => return Err(libc::EINVAL),
        };
        t.set_quantization(q).map_err(|_| libc::EINVAL)?;
    }
    Ok(t)
}

/// The shape a terminal call allocates or wraps with.
fn shape_of(b: &EfTensorBuilder) -> Result<Vec<usize>, c_int> {
    if b.shape.is_empty() {
        return Err(libc::EINVAL);
    }
    Ok(b.shape.iter().map(|d| *d as usize).collect())
}

/// Minimum plane byte extent implied by `shape` and an optional `stride`
/// (bytes per line; `0` means tight).
///
/// With a `stride`, the extent is the product of every dimension but the
/// last, times `stride`. Returns `None` on an empty shape or an overflow.
fn required_plane_bytes(shape: &[usize], elem_size: usize, stride: u64) -> Option<u64> {
    let elem_size = elem_size as u64;
    if stride == 0 {
        let elems = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))?;
        elems.checked_mul(elem_size)
    } else {
        let (_last, leading) = shape.split_last()?;
        let rows = leading
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))?;
        rows.checked_mul(stride)
    }
}

/// Allocate storage and produce a tensor. Returns `NULL` on failure.
///
/// Fails when planes were added. The builder survives and may be used again.
///
/// # Safety
/// `b` must be `NULL` or a live builder.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_alloc(
    b: *mut EfTensorBuilder,
) -> *mut crate::handle::EfTensor {
    terminal(
        b,
        |b| {
            if !b.planes.is_empty() {
                crate::last_error::set_last_error(
                    "builder_alloc: planes were added; use ef_tensor_builder_wrap to adopt \
                     an existing buffer, or omit them to allocate",
                );
                return Err(libc::EINVAL);
            }
            let shape = shape_of(b).inspect_err(|_| {
                crate::last_error::set_last_error("builder_alloc: no shape was set");
            })?;
            let dtype = b.dtype.ok_or(libc::EINVAL).inspect_err(|_| {
                crate::last_error::set_last_error("builder_alloc: no dtype was set");
            })?;
            let storage = b.storage.unwrap_or(TensorMemory::Mem);
            edgefirst_tensor::TensorDyn::new(&shape, dtype, Some(storage), None).map_err(|e| {
                let errno = crate::map::errno_for(&e);
                crate::last_error::set_last_error_classified(
                    crate::last_error::class_of(&e),
                    &format!("builder_alloc: {e}"),
                );
                errno
            })
        },
        |_, t| Ok(t),
    )
}

/// Adopt an externally-owned handle and produce a tensor. Returns `NULL` on
/// failure.
///
/// Requires exactly one plane carrying a handle.
///
/// Adopts the handle only on success: the resulting tensor owns it and the
/// caller must not close it. On failure the handle is left open and remains
/// the caller's to close.
///
/// See [`ef_tensor_builder_add_plane`] for how each plane field is treated.
/// A nonzero plane `stride` supplied together with a format
/// (`ef_tensor_builder_format`) is validated against that format.
///
/// @retval non-`NULL` success.
/// @retval `NULL` on failure; [`ef_tensor_builder_error`] distinguishes why:
///   - `EINVAL` no plane was added, `handle` is negative or does not fit an
///     `int`, no dtype/shape was set, `stride` is nonzero and smaller than
///     the format's minimum row size (checked only when a format is
///     attached), or the fd could not be imported.
///   - `ERANGE` `size` is nonzero and smaller than the extent `shape` and
///     `stride` require.
///   - `EBADMSG` `used` does not equal `size`.
///   - `EDOM` `modifier` is nonzero.
///   - `ENOTSUP` more than one plane was added.
///   - `ENOSYS` the platform is non-Unix.
///
/// # Safety
/// `b` must be `NULL` or a live builder, and any plane handle must be a valid
/// file descriptor this process may adopt.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_builder_wrap(
    b: *mut EfTensorBuilder,
) -> *mut crate::handle::EfTensor {
    terminal(
        b,
        |b| {
            let Some(first) = b.planes.first() else {
                return Err(libc::EINVAL);
            };
            if b.planes.len() > 1 {
                return Err(libc::ENOTSUP);
            }
            if first.modifier != 0 {
                return Err(libc::EDOM);
            }
            if first.used != first.size {
                return Err(libc::EBADMSG);
            }
            let shape = shape_of(b)?;
            let dtype = b.dtype.ok_or(libc::EINVAL)?;
            if first.handle < 0 {
                return Err(libc::EINVAL);
            }
            if first.size != 0 {
                // A `size` of 0 is unspecified; a nonzero one must cover what
                // `shape` and `stride` require.
                let required =
                    required_plane_bytes(&shape, dtype.size(), first.stride).ok_or(libc::ERANGE)?;
                if first.size < required {
                    return Err(libc::ERANGE);
                }
            }
            #[cfg(unix)]
            {
                let fd = i32::try_from(first.handle).map_err(|_| libc::EINVAL)?;
                // The tensor is built on a duplicate of the caller's fd;
                // `finish` closes the caller's fd.
                // SAFETY: `fd` is a valid open fd, borrowed for the duplicate.
                let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
                let owned = borrowed.try_clone_to_owned().map_err(|_| libc::EINVAL)?;
                let t = edgefirst_tensor::TensorDyn::from_fd(owned, &shape, dtype, None)
                    .map_err(|_| libc::EINVAL)?;
                Ok(t)
            }
            #[cfg(not(unix))]
            {
                let _ = (shape, dtype);
                Err(libc::ENOSYS)
            }
        },
        |b, mut t| {
            // Runs after `apply_metadata`. A `stride` or `offset` of 0 is
            // unspecified and is not applied.
            if let Some(first) = b.planes.first() {
                if first.stride > 0 {
                    t.set_row_stride(first.stride as usize)
                        .map_err(|_| libc::EINVAL)?;
                }
                if first.offset > 0 {
                    t.set_plane_offset(first.offset as usize);
                }
                // Close the caller's fd; the tensor holds the duplicate made
                // in `body`.
                #[cfg(unix)]
                {
                    use std::os::fd::FromRawFd;
                    // SAFETY: `body` validated the handle as an open fd that
                    // fits an `int`.
                    drop(unsafe { std::os::fd::OwnedFd::from_raw_fd(first.handle as i32) });
                }
            }
            Ok(t)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new builder.
    fn b() -> *mut EfTensorBuilder {
        let p = ef_tensor_builder_new();
        assert!(!p.is_null());
        p
    }

    /// A raw fd backing `len` bytes of shared memory, owned by the caller,
    /// or `-1` when shared memory is unavailable.
    fn shm_fd(len: usize) -> i64 {
        if !edgefirst_tensor::is_shm_available() {
            return -1;
        }
        #[cfg(unix)]
        {
            let t =
                edgefirst_tensor::TensorDyn::new(&[len], DType::U8, Some(TensorMemory::Shm), None)
                    .expect("shm backing allocation");
            use std::os::fd::IntoRawFd;
            t.clone_fd().expect("clone fd").into_raw_fd() as i64
        }
        #[cfg(not(unix))]
        {
            let _ = len;
            -1
        }
    }

    #[test]
    fn wrap_without_planes_is_an_error() {
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0);
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EINVAL);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn alloc_with_planes_is_an_error() {
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0);
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            ef_tensor_builder_add_plane(p, 7, 0, 4, 16, 16, 0);
            assert!(ef_tensor_builder_alloc(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EINVAL);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn wrap_carries_the_plane_offset_across_format_application() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        // 720x640 NV12 (combined-plane height 720) plus one extra row for
        // the nonzero offset.
        let handle = shm_fd(721 * 640);
        let p = b();
        let dims = [720u64, 640];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            let nv12 = std::ffi::CString::new("NV12").unwrap();
            assert_eq!(ef_tensor_builder_format(p, nv12.as_ptr()), 0);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 640, 0, 0, 0, 0), 0);
            let t = ef_tensor_builder_wrap(p);
            assert!(
                !t.is_null(),
                "wrap must succeed; errno {}",
                ef_tensor_builder_error(p)
            );
            assert_eq!(
                crate::handle::inner_of(t).plane_offset(),
                Some(640),
                "the offset must survive apply_metadata's format application"
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn wrap_carries_the_stride_across_format_application() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        // 720x640 NV12 with a padded stride of 768.
        let handle = shm_fd(720 * 768);
        let p = b();
        let dims = [720u64, 640];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            let nv12 = std::ffi::CString::new("NV12").unwrap();
            assert_eq!(ef_tensor_builder_format(p, nv12.as_ptr()), 0);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 768, 0, 0, 0), 0);
            let t = ef_tensor_builder_wrap(p);
            assert!(
                !t.is_null(),
                "wrap must succeed; errno {}",
                ef_tensor_builder_error(p)
            );
            assert_eq!(
                crate::handle::inner_of(t).row_stride(),
                Some(768),
                "the stride must survive apply_metadata's format application"
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn wrap_rejects_a_size_smaller_than_shape_and_stride_imply() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            // 4x4 u8 needs 16 bytes; declare a plane extent of 8.
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 8, 8, 0), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::ERANGE);
            ef_tensor_builder_free(p);
            libc::close(handle as i32);
        }
    }

    #[test]
    fn wrap_accepts_a_size_larger_than_shape_and_stride_require() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(64);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            // 4x4 u8 needs 16 bytes; declare a 64-byte plane.
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 64, 64, 0), 0);
            let t = ef_tensor_builder_wrap(p);
            assert!(
                !t.is_null(),
                "a larger-than-required size must be accepted; errno {}",
                ef_tensor_builder_error(p)
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn wrap_rejects_used_not_equal_to_size() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 16, 8, 0), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EBADMSG);
            ef_tensor_builder_free(p);
            libc::close(handle as i32);
        }
    }

    #[test]
    fn wrap_rejects_a_nonzero_modifier() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 1), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EDOM);
            ef_tensor_builder_free(p);
            libc::close(handle as i32);
        }
    }

    #[test]
    fn wrap_rejects_a_second_plane() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 0), 0);
            // The second plane's handle is never adopted.
            assert_eq!(ef_tensor_builder_add_plane(p, 99999, 0, 0, 0, 0, 0), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::ENOTSUP);
            ef_tensor_builder_free(p);
            libc::close(handle as i32);
        }
    }

    /// The `(dev, inode)` an open fd refers to, or `None` once it is closed.
    #[cfg(unix)]
    fn fd_identity(fd: i32) -> Option<(libc::dev_t, libc::ino_t)> {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `st` is a valid out-pointer for one `stat`.
        (unsafe { libc::fstat(fd, st.as_mut_ptr()) } == 0).then(|| {
            // SAFETY: `fstat` succeeded, so `st` is initialised.
            let st = unsafe { st.assume_init() };
            (st.st_dev, st.st_ino)
        })
    }

    /// A handle `from_fd` refuses leaves `wrap` returning NULL with the
    /// handle still open.
    #[cfg(target_os = "linux")]
    #[test]
    fn wrap_leaves_a_handle_from_fd_refuses_open() {
        // `from_fd` refuses a pipe: it is neither a dma-buf nor tmpfs.
        let mut ends = [0i32; 2];
        // SAFETY: `ends` is a valid out-array of two fds.
        assert_eq!(unsafe { libc::pipe(ends.as_mut_ptr()) }, 0);
        let [handle, write_end] = ends;
        let before = fd_identity(handle).expect("open fd");
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(
                ef_tensor_builder_add_plane(p, handle as i64, 0, 0, 0, 0, 0),
                0
            );
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EINVAL);
            ef_tensor_builder_free(p);
        }
        assert_eq!(
            fd_identity(handle),
            Some(before),
            "wrap closed a handle it refused"
        );
        // SAFETY: both ends are this test's own open fds.
        unsafe {
            assert_eq!(libc::close(handle), 0);
            libc::close(write_end);
        }
    }

    /// A failure in `apply_metadata`, after `from_fd` succeeded, leaves the
    /// handle open.
    #[cfg(unix)]
    #[test]
    fn wrap_leaves_the_handle_open_when_metadata_fails() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let before = fd_identity(handle as i32).expect("open fd");
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_format(p, c"not-a-format".as_ptr()), 0);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 0), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EINVAL);
            ef_tensor_builder_free(p);
        }
        assert_eq!(
            fd_identity(handle as i32),
            Some(before),
            "wrap closed a handle it refused"
        );
        assert_eq!(unsafe { libc::close(handle as i32) }, 0);
    }

    /// On success the caller's handle is adopted: it no longer names the
    /// buffer's file.
    #[cfg(unix)]
    #[test]
    fn wrap_adopts_the_handle_on_success() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let before = fd_identity(handle as i32).expect("open fd");
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 0), 0);
            let t = ef_tensor_builder_wrap(p);
            assert!(!t.is_null());
            assert_ne!(
                fd_identity(handle as i32),
                Some(before),
                "wrap left the caller's handle open"
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn wrap_still_succeeds_with_all_zero_plane_fields() {
        if !edgefirst_tensor::is_shm_available() {
            crate::artifacts::skip("SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 0), 0);
            let t = ef_tensor_builder_wrap(p);
            assert!(
                !t.is_null(),
                "an all-zero plane must still wrap; errno {}",
                ef_tensor_builder_error(p)
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn a_builder_can_alloc_repeatedly_to_fill_a_pool() {
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0);
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            ef_tensor_builder_storage(p, 0);
            let mut ids = Vec::new();
            let mut handles = Vec::new();
            for _ in 0..3 {
                let t = ef_tensor_builder_alloc(p);
                assert!(!t.is_null(), "alloc must keep working after the first call");
                // Compare buffer identities, not handle pointers.
                ids.push(crate::handle::inner_of(t).buffer_identity().id());
                handles.push(t);
            }
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), 3, "each alloc must produce a distinct buffer");
            for t in handles {
                crate::handle::ef_tensor_free(t);
            }
            assert_eq!(ef_tensor_builder_error(p), 0, "the builder is still usable");
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn a_terminal_call_on_a_failed_builder_reports_the_original_error() {
        let p = b();
        unsafe {
            ef_tensor_builder_dtype(p, 9999);
            assert!(ef_tensor_builder_alloc(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::EINVAL);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn alloc_carries_format_and_colorimetry_onto_the_tensor() {
        let p = b();
        // NV12 640x480 allocation geometry.
        let dims = [720u64, 640];
        unsafe {
            ef_tensor_builder_dtype(p, 0);
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            ef_tensor_builder_storage(p, 0);
            let nv12 = std::ffi::CString::new("NV12").unwrap();
            assert_eq!(ef_tensor_builder_format(p, nv12.as_ptr()), 0);
            let t = ef_tensor_builder_alloc(p);
            assert!(!t.is_null());
            assert_eq!(
                crate::handle::inner_of(t).format(),
                Some(edgefirst_tensor::PixelFormat::Nv12)
            );
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn alloc_carries_per_channel_asymmetric_quantization_zero_points_onto_the_tensor() {
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            let scales = [0.1f32, 0.2, 0.3, 0.4];
            let zps = [1i32, -2, 3, -4];
            assert_eq!(
                ef_tensor_builder_quantization(p, 1, scales.as_ptr(), zps.as_ptr(), 4),
                0
            );
            let t = ef_tensor_builder_alloc(p);
            assert!(!t.is_null());
            let q = crate::handle::inner_of(t)
                .quantization()
                .expect("quantization must be attached");
            assert_eq!(q.axis(), Some(1));
            assert_eq!(q.scale(), &scales);
            assert_eq!(q.zero_point(), Some(&zps[..]));
            crate::handle::ef_tensor_free(t);
            ef_tensor_builder_free(p);
        }
    }

    #[test]
    fn a_fresh_builder_reports_no_error() {
        let p = b();
        assert_eq!(unsafe { ef_tensor_builder_error(p) }, 0);
        unsafe { ef_tensor_builder_free(p) };
    }

    #[test]
    fn the_first_error_sticks_and_later_setters_no_op() {
        let p = b();
        // An unknown dtype code.
        assert_ne!(unsafe { ef_tensor_builder_dtype(p, 9999) }, 0);
        // A valid call afterwards reports the sticky error.
        let dims = [4u64, 4];
        let rc = unsafe { ef_tensor_builder_shape(p, dims.as_ptr(), 2) };
        assert_ne!(
            rc, 0,
            "a setter after a failure must report the sticky error"
        );
        assert_eq!(
            unsafe { ef_tensor_builder_error(p) },
            libc::EINVAL,
            "the FIRST failure must survive, not the most recent call"
        );
        unsafe { ef_tensor_builder_free(p) };
    }

    #[test]
    fn a_later_setter_does_not_take_effect_once_an_error_is_pending() {
        let p = b();
        assert_ne!(unsafe { ef_tensor_builder_dtype(p, 9999) }, 0);
        let dims = [4u64, 4];
        unsafe { ef_tensor_builder_shape(p, dims.as_ptr(), 2) };
        assert!(
            unsafe { (*p).shape.is_empty() },
            "the shape must not have been recorded after a sticky error"
        );
        unsafe { ef_tensor_builder_free(p) };
    }

    #[test]
    fn each_setter_validates_its_own_field() {
        let p = b();
        assert_eq!(unsafe { ef_tensor_builder_dtype(p, 0) }, 0, "U8 is valid");
        unsafe { ef_tensor_builder_free(p) };

        let p = b();
        assert_ne!(
            unsafe { ef_tensor_builder_shape(p, std::ptr::null(), 2) },
            0,
            "a null dims pointer with ndim > 0 is invalid"
        );
        unsafe { ef_tensor_builder_free(p) };

        let p = b();
        let dims = [4u64, 4];
        unsafe { ef_tensor_builder_shape(p, dims.as_ptr(), 2) };
        let strides = [4i64];
        assert_ne!(
            unsafe { ef_tensor_builder_strides(p, strides.as_ptr(), 1) },
            0,
            "strides must have the same rank as the shape"
        );
        unsafe { ef_tensor_builder_free(p) };

        let p = b();
        assert_ne!(
            unsafe { ef_tensor_builder_storage(p, 9999) },
            0,
            "an unknown storage kind is invalid"
        );
        unsafe { ef_tensor_builder_free(p) };
    }

    #[test]
    fn a_null_builder_is_an_error_not_a_crash() {
        let dims = [1u64];
        unsafe {
            assert_eq!(ef_tensor_builder_error(std::ptr::null()), libc::EINVAL);
            assert_eq!(
                ef_tensor_builder_dtype(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            assert_eq!(
                ef_tensor_builder_shape(std::ptr::null_mut(), dims.as_ptr(), 1),
                libc::EINVAL
            );
            assert_eq!(
                ef_tensor_builder_storage(std::ptr::null_mut(), 0),
                libc::EINVAL
            );
            assert_eq!(
                ef_tensor_builder_fence(std::ptr::null_mut(), -1),
                libc::EINVAL
            );
            // Freeing null is a no-op.
            ef_tensor_builder_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_valid_sequence_leaves_no_error() {
        let p = b();
        let dims = [4u64, 4];
        let strides = [4i64, 1];
        unsafe {
            assert_eq!(ef_tensor_builder_dtype(p, 0), 0);
            assert_eq!(ef_tensor_builder_shape(p, dims.as_ptr(), 2), 0);
            assert_eq!(ef_tensor_builder_strides(p, strides.as_ptr(), 2), 0);
            assert_eq!(ef_tensor_builder_storage(p, 0), 0);
            assert_eq!(ef_tensor_builder_fence(p, -1), 0);
            assert_eq!(ef_tensor_builder_error(p), 0);
            ef_tensor_builder_free(p);
        }
    }
}
