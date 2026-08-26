// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Construction: a builder, not a descriptor struct.
//!
//! A `ef_tensor_desc` struct passed by value would fix its layout forever and
//! give one error for the whole call. A builder gives **per-field error
//! handling**, keeps the opaque-pointer extensibility the rest of this API
//! relies on, is friendlier to FFI generators, and supports a header-only C++
//! wrapper that chains.
//!
//! # Sticky errors
//!
//! After a setter fails, later setters no-op and the terminal call returns the
//! **first** failure with its `errno`. That is what lets a C++ wrapper write
//!
//! ```text
//! Tensor t = Builder().dtype(U8).shape(dims, 2).storage(Dma).alloc();
//! ```
//!
//! without a check per line, and still learn precisely which field was wrong.
//! Reporting the *last* error instead would name whichever call happened to
//! come after the real fault.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, TensorMemory};

/// One plane handed to `wrap`, mirroring `TensorPlane` on the wire.
///
/// Not every field is honoured equally by [`ef_tensor_builder_wrap`] -- see
/// [`ef_tensor_builder_add_plane`]'s doc for the field-by-field disposition
/// (`handle`/`stride`/`offset` carried; `size`/`used`/`modifier` validated or
/// rejected, never stored; a second plane rejected). They are all still
/// recorded here regardless, because a multi-handle wrap needs every one of
/// them the moment it lands.
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
/// Opaque to C. The `err` field is the sticky-error state: once non-zero every
/// setter becomes a no-op and returns it unchanged.
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

/// Run a setter under the sticky-error discipline.
///
/// Returns the pending error without running `body` when one is already set,
/// which is what makes "later setters no-op" true rather than merely
/// "later setters also report an error" -- the distinction matters, because a
/// terminal call must not build from a half-populated builder.
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

/// Free a builder. Freeing `NULL` is a no-op, matching `free(3)`.
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

/// Set the element type.
///
/// Takes the integer rather than `ef_dtype` deliberately: a C caller can pass
/// any value, and transmuting an out-of-range one into a Rust enum is undefined
/// behaviour, while validating an integer is not. Pass an `EF_DTYPE_*`
/// enumerator; an unknown code is `EINVAL`.
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

/// Set strides, **in bytes**.
///
/// Must have the same rank as the shape: a partial stride array has no
/// meaning, matching the blob format's all-or-nothing rule.
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

/// Set the backing store.
///
/// Takes the integer rather than `ef_storage_kind`, for the same reason as
/// [`ef_tensor_builder_dtype`]. Pass an `EF_STORAGE_KIND_*` enumerator.
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
/// Every field is recorded here, but [`ef_tensor_builder_wrap`] cannot honour
/// all of them equally -- the underlying `Tensor` can only represent a
/// subset of what a real V4L2/DRM plane carries:
///
/// * `handle` -- adopted as the tensor's fd.
/// * `stride` -- carried onto the tensor as its row stride, applied after any
///   format the builder also carries (`ef_tensor_builder_format`) is
///   attached -- `set_row_stride` itself requires a format already be set,
///   so applying it any earlier would reject every `wrap` call that supplied
///   both a format and a nonzero stride, unconditionally, regardless of
///   whether the stride was actually valid.
/// * `offset` -- carried onto the tensor as its plane offset, for the same
///   reason applied after any format (this is the one field `wrap` used to
///   silently drop).
/// * `size` -- validated against the extent `shape` and `stride` imply, never
///   stored: the tensor derives its own extent from shape and stride, and a
///   caller-supplied `size` is only ever a sanity bound. `0` means
///   "unspecified" and is not checked (matching `stride`'s own convention). A
///   nonzero `size` **smaller** than required is rejected; **larger** is
///   accepted -- a padded or over-allocated buffer is a normal thing to hand
///   a wrapper.
/// * `used` -- rejected unless equal to `size`. The tensor has no
///   partial-fill/`bytes_used` concept, so any other value is unrepresentable.
///   `used > size` is rejected right here, by this function, with `EINVAL`;
///   `used < size` is rejected by `wrap` itself, with `EBADMSG` -- same
///   underlying fact caught at two different points in the two functions'
///   own validation order, not an accidental split.
/// * `modifier` -- rejected unless `0` (linear). The `Tensor` type has no
///   representation for a DRM format modifier: adopting a tiled or
///   compressed buffer under a nonzero modifier would read it back as
///   linear, which is silently wrong data in every pixel, so `wrap` refuses
///   it instead.
/// * a second and later plane -- rejected: `wrap` adopts exactly one handle
///   per tensor. Combined multi-plane geometry (e.g. NV12's Y and UV planes
///   at different offsets within one dma-buf) is not something this builder
///   composes; wrap each plane as its own single-plane tensor instead.
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
        CStr::from_ptr(p).to_str().ok().map(|s| s.to_string())
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
            // Per-tensor means exactly one scale; anything else is a caller bug
            // that would otherwise surface much later as a shape mismatch.
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
/// Records a failure on the builder and returns `NULL`, so a chained caller
/// that checks only [`ef_tensor_builder_error`] at the end still learns what
/// went wrong. The builder is **not** consumed: `alloc` in a loop with one
/// builder is how a V4L2 buffer pool is filled, and the caller frees it
/// explicitly.
///
/// `finish` runs *after* [`apply_metadata`], not alongside `body` -- that
/// ordering matters for anything that `apply_metadata`'s `set_format` call
/// clears (`plane_offset`, notably: `tensor/src/lib.rs` documents `set_format`
/// wiping it). A step that must survive a format application belongs in
/// `finish`, not `body`, or it is silently undone the instant a caller also
/// supplies a format.
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
        // `zeros` is `Vec::new()` when the caller passed a `NULL` `zps`
        // (see `ef_tensor_builder_quantization`'s own null-handling), never
        // partially filled -- so `.is_empty()` is the correct symmetric
        // test for both the per-tensor and per-channel arms below. An
        // earlier version matched only on `zeros.first()` for the
        // per-tensor arms and unconditionally built `per_channel_symmetric`
        // for every per-channel request regardless of `zeros` -- silently
        // discarding real per-channel zero-points a caller supplied. Fixed
        // while proving family 2's `ef_tensor_quantization_get` primitive
        // (task 15): a round-trip through this exact path is how a real C
        // producer attaches per-channel asymmetric quantization.
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
/// (bytes per line; `0` means "tight", matching [`ef_tensor_builder_wrap`]'s
/// own convention for an unset `stride`).
///
/// Used only to validate a caller-supplied `size` in `wrap` -- never to
/// derive storage. Row-major: a `stride` wider than the tight last-dimension
/// row overrides that dimension's contribution and scales every dimension
/// before it as whole rows, the same relationship [`edgefirst_tensor::Tensor`]'s
/// own `set_row_stride` validates for the packed/planar/semi-planar cases,
/// generalized here without format-layout knowledge (`wrap`'s body runs
/// before [`apply_metadata`] attaches one). `None` on an empty shape or an
/// element-count/byte-count overflow.
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
/// Requires **no** planes: `alloc` derives storage from the format, shape and
/// alignment. Supplying planes means the caller wanted [`ef_tensor_builder_wrap`],
/// so it is rejected rather than silently ignored.
///
/// The builder survives and may be called again.
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
            // Was `map_err(|_| libc::ENOMEM)`, which reported "out of
            // memory" for every refusal and threw the reason away with it.
            // `TensorMemory::IoSurface` on any platform, or `Pbo`/`Cuda`
            // anywhere, is a backing this build does not serve -- not a
            // shortage of memory -- and a caller cannot tell "try a smaller
            // allocation" from "ask for a different backing" when both
            // arrive as ENOMEM. `errno_for` is the shared conversion whose
            // own doc comment asks callers to use it rather than collapse
            // to one hardcoded code; this was the last site still
            // collapsing.
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

/// Adopt externally-owned handles and produce a tensor. Returns `NULL` on failure.
///
/// Requires **at least one** plane carrying a real handle — that is the
/// difference from [`ef_tensor_builder_alloc`], and the reason misuse is a
/// per-field error rather than a convention.
///
/// Adopts the handle: the resulting tensor owns it and the caller must not
/// close it.
///
/// **Behaviour change**: earlier versions of this function silently ignored
/// `offset`, `size`, `used`, `modifier`, and every plane past the first --
/// a caller who supplied any of them got a tensor that looked valid but read
/// from the wrong place, or under the wrong layout. They are now carried,
/// validated, or rejected; see [`ef_tensor_builder_add_plane`]'s doc for the
/// field-by-field disposition. A third-party caller that was previously
/// getting a silently-wrong tensor now gets one of the errors below instead
/// -- the correct trade, but a real behaviour change for anyone linking this
/// library. Also fixed in the same pass: a `wrap` call that supplied both a
/// format (`ef_tensor_builder_format`) and a nonzero plane `stride` used to
/// fail unconditionally with `EINVAL` regardless of whether the stride was
/// valid, because the stride was applied before the format was; that
/// combination now succeeds when the stride is actually valid for the format.
///
/// @retval non-`NULL` success.
/// @retval `NULL` on failure; [`ef_tensor_builder_error`] distinguishes why:
///   - `EINVAL` no plane was added, `handle` is negative or does not fit an
///     `int`, no dtype/shape was set, `stride` is nonzero and smaller than
///     the format's minimum row size (only checked once a format is
///     attached; same check as [`crate::mutate::ef_tensor_set_row_stride`]),
///     `from_fd` itself failed (e.g. an unrecognized fd type), or the
///     platform is non-Unix.
///   - `ERANGE` `size` is nonzero and smaller than the extent `shape` and
///     `stride` require.
///   - `EBADMSG` `used` does not equal `size`.
///   - `EDOM` `modifier` is nonzero -- only linear (`0`) is representable.
///   - `ENOTSUP` more than one plane was added -- `wrap` adopts a single
///     handle per tensor.
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
                // Nothing to adopt. Falling back to an allocation here would
                // hide the caller's mistake behind a tensor that looks right.
                return Err(libc::EINVAL);
            };
            if b.planes.len() > 1 {
                // One handle per tensor: see `add_plane`'s doc for why this
                // is a refusal rather than a silent truncation to `first`.
                return Err(libc::ENOTSUP);
            }
            if first.modifier != 0 {
                // The type has no representation for a DRM format modifier;
                // adopting it as linear would be silently wrong data.
                return Err(libc::EDOM);
            }
            if first.used != first.size {
                // No partial-fill/`bytes_used` concept to carry this in.
                // `used > size` is already rejected at `add_plane`; this is
                // the `used < size` half of the same "unrepresentable" fact.
                return Err(libc::EBADMSG);
            }
            let shape = shape_of(b)?;
            let dtype = b.dtype.ok_or(libc::EINVAL)?;
            if first.handle < 0 {
                return Err(libc::EINVAL);
            }
            if first.size != 0 {
                // `0` means "unspecified", matching `stride`'s convention
                // below. A caller-supplied `size` is a sanity bound, not a
                // stored value: reject only when it is too small to hold
                // what `shape`/`stride` require. Larger is fine -- a padded
                // or over-allocated buffer is a normal thing to wrap.
                let required =
                    required_plane_bytes(&shape, dtype.size(), first.stride).ok_or(libc::ERANGE)?;
                if first.size < required {
                    return Err(libc::ERANGE);
                }
            }
            #[cfg(unix)]
            {
                use std::os::fd::FromRawFd;
                let fd = i32::try_from(first.handle).map_err(|_| libc::EINVAL)?;
                // SAFETY: the caller contracts that this is a valid fd to adopt.
                let owned = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
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
            // Both run after `apply_metadata`, not inside `body` above:
            // `set_format` clears `plane_offset` when it runs, and
            // `Tensor::set_row_stride` itself requires a format already be
            // attached -- setting either before the builder's format is
            // applied meant a `wrap` call that supplied both a format AND a
            // nonzero `stride`/`offset` either lost the value (offset) or
            // failed outright with `EINVAL` (stride, unconditionally, even
            // though both fields were valid). `0` means "unspecified" for
            // both fields, so there is nothing to carry when either is 0.
            if let Some(first) = b.planes.first() {
                // A wrapped V4L2 or libcamera buffer usually has a hardware
                // pitch wider than width * bpp. Dropping it here would make
                // every later row address wrong, so it is carried onto the
                // tensor rather than recomputed from the shape.
                if first.stride > 0 {
                    t.set_row_stride(first.stride as usize)
                        .map_err(|_| libc::EINVAL)?;
                }
                if first.offset > 0 {
                    t.set_plane_offset(first.offset as usize);
                }
            }
            Ok(t)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A builder that is freed when the test ends.
    fn b() -> *mut EfTensorBuilder {
        let p = ef_tensor_builder_new();
        assert!(!p.is_null());
        p
    }

    /// A real, adoptable fd backing `shape` bytes of `u8`, for tests that
    /// need `wrap` to actually reach `from_fd` rather than fail earlier.
    /// Leaked as a raw fd deliberately: the caller either hands it to
    /// `wrap` (which adopts and eventually frees it via the returned
    /// tensor) or must close it itself on a path where `wrap` rejects the
    /// plane before adoption.
    fn shm_fd(len: usize) -> i64 {
        if !edgefirst_tensor::is_shm_available() {
            return -1;
        }
        let t = edgefirst_tensor::TensorDyn::new(&[len], DType::U8, Some(TensorMemory::Shm), None)
            .expect("shm backing allocation");
        use std::os::fd::IntoRawFd;
        t.clone_fd().expect("clone fd").into_raw_fd() as i64
    }

    #[test]
    fn wrap_without_planes_is_an_error() {
        // `wrap` adopts external handles; with no planes there is nothing to
        // adopt, and silently allocating instead would hide the caller's bug.
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
        // The mirror: `alloc` derives storage itself, so supplied planes mean
        // the caller wanted `wrap`. Misuse is a per-field error, not a
        // documented convention nobody reads.
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
        // Ordering hazard: `set_format` clears `plane_offset` (see the clear
        // sites `tensor/src/lib.rs` documents next to `set_plane_offset`).
        // If `wrap` set the offset before `apply_metadata` applies the
        // builder's format, the format application immediately after would
        // silently wipe it. Proves the offset survives a `wrap` call that
        // ALSO carries a format, not just one that omits it.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
            return;
        }
        // 720x640 NV12 (480p, combined-plane height 720) plus one extra row
        // of slack so a nonzero offset stays inside the backing allocation.
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
        // Mirror of `wrap_carries_the_plane_offset_across_format_application`:
        // `Tensor::set_row_stride` requires a format to already be attached
        // (`tensor/src/lib.rs`), so applying the stride inside `wrap`'s
        // `body` -- before `apply_metadata` applies the builder's format --
        // meant a `wrap` call that supplied both a format AND a nonzero
        // plane `stride` failed unconditionally with `EINVAL`, even when the
        // stride was perfectly valid for that format.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
            return;
        }
        // 720x640 NV12 with a padded stride (768, wider than the tight 640
        // minimum for this width).
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
            eprintln!("SKIPPED: SHM not available");
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
        // The mirror of the previous test: a generous/padded/over-allocated
        // buffer is a normal thing to hand a wrapper and must not be
        // rejected merely for being bigger than the tight requirement.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
            return;
        }
        let handle = shm_fd(64);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            // 4x4 u8 needs 16 bytes; declare a generous 64-byte plane.
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
        // `add_plane` already rejects `used > size`; this closes the rest --
        // the tensor has no partial-fill/`bytes_used` concept, so `used <
        // size` cannot be represented either.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
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
        // The type cannot express a DRM format modifier; a tiled or
        // compressed buffer silently reinterpreted as linear is exactly the
        // "wrong data disguised as an answer" shape this task exists to fix.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
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
        // `wrap` adopts exactly one handle per tensor -- combined-plane
        // geometry (e.g. NV12's Y/UV at different offsets in one dma-buf) is
        // not something this builder composes; an honest refusal beats a
        // silent truncation to the first plane.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
            return;
        }
        let handle = shm_fd(16);
        let p = b();
        let dims = [4u64, 4];
        unsafe {
            ef_tensor_builder_dtype(p, 0); // U8
            ef_tensor_builder_shape(p, dims.as_ptr(), 2);
            assert_eq!(ef_tensor_builder_add_plane(p, handle, 0, 0, 0, 0, 0), 0);
            // The second plane's handle is never adopted -- any value works.
            assert_eq!(ef_tensor_builder_add_plane(p, 99999, 0, 0, 0, 0, 0), 0);
            assert!(ef_tensor_builder_wrap(p).is_null());
            assert_eq!(ef_tensor_builder_error(p), libc::ENOTSUP);
            ef_tensor_builder_free(p);
            libc::close(handle as i32);
        }
    }

    #[test]
    fn wrap_still_succeeds_with_all_zero_plane_fields() {
        // Blast-radius regression: the only in-tree caller of `wrap`
        // (`TensorDyn::from_fd` in `tensor_dyn/dynamic_backend.rs`) calls
        // `add_plane` with offset/stride/size/used/modifier all zero. None
        // of this task's new rejections may fire on that shape of call.
        if !edgefirst_tensor::is_shm_available() {
            eprintln!("SKIPPED: SHM not available");
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
        // The builder survives its terminal call: `alloc` in a loop with one
        // builder is how a V4L2 buffer pool gets filled.
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
                // Distinct BUFFERS, not merely distinct handle pointers: three
                // wrappers around one allocation would also give three
                // pointers, and would be wrong.
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
        // Sticky errors must survive into the terminal call, or a chained
        // caller checking only at the end learns nothing.
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
        // NV12 640x480 allocation geometry, so the format validates.
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
        // Regression: `apply_metadata` used to build `per_channel_symmetric`
        // unconditionally for any `axis >= 0` request, silently discarding
        // real zero-points a caller supplied -- found while proving family
        // 2's `ef_tensor_quantization_get` primitive round-trips a builder-
        // attached quantization faithfully (task 15).
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
        // Bad dtype code: this is the fault the caller must eventually see.
        assert_ne!(unsafe { ef_tensor_builder_dtype(p, 9999) }, 0);
        // A perfectly good call afterwards must NOT clear it, and must not
        // report success either -- a chaining wrapper checks only at the end.
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
        // "No-op" has to mean it, or the terminal call would build from a
        // half-populated builder that looks valid.
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
        // Per-field validation is the reason this is a builder and not a
        // struct: one bad field is attributable to that field.
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
        // Every entry point is reachable from C with a null pointer.
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
            // Freeing null must be a no-op, matching free(3).
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
