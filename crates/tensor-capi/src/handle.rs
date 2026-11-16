// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The tensor handle and its accessors.
//!
//! [`EfTensor`] is an opaque zero-sized marker: C holds a
//! `*mut EfTensor`/`*const EfTensor` and passes it back through an exported
//! accessor. The allocation behind it is [`EfTensorImpl`], reached by
//! reinterpreting the same pointer.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, TensorDyn, TensorMemory};
use edgefirst_tensor_abi::EfErrorClass;
pub use edgefirst_tensor_abi::{EfCompression, EfContiguity, EfTensorPlane, EfViewOrigin};

use crate::last_error::{set_errno, set_last_error_classified};

/// An opaque tensor handle.
///
/// C holds a pointer to this and passes it back through an exported
/// accessor.
#[repr(C)]
pub struct EfTensor {
    _private: [u8; 0],
}

/// The allocation behind an [`EfTensor`].
///
/// `shape_u64`, `strides_i64` and `format_c` are the C-visible forms of the
/// tensor's shape, byte strides and format. `map_state` holds the
/// outstanding `ef_tensor_map` guard, if any.
#[repr(C)]
pub(crate) struct EfTensorImpl {
    pub(crate) map_state: std::sync::Mutex<Option<crate::map::MapState>>,
    pub(crate) inner: TensorDyn,
    shape_u64: Vec<u64>,
    strides_i64: Vec<i64>,
    format_c: CString,
    /// CPU-side handle count.
    pub(crate) refs: std::sync::atomic::AtomicUsize,
    /// Packed colorimetry (`Colorimetry::pack`'s wire form), read by
    /// [`ef_tensor_colorimetry`] and written by [`ef_tensor_set_colorimetry`].
    /// Seeded from `inner` in [`into_handle`].
    colorimetry: std::sync::atomic::AtomicU32,
}

/// Recover the implementation behind a handle.
///
/// # Safety
/// `t` must be a live handle produced by [`into_handle`].
unsafe fn imp<'a>(t: *const EfTensor) -> Option<&'a EfTensorImpl> {
    unsafe {
        if t.is_null() {
            return None;
        }
        Some(&*(t as *const EfTensorImpl))
    }
}

/// Recover the implementation behind a handle for exclusive mutation.
///
/// # Safety
/// `t` must be a live handle produced by [`into_handle`]. No other
/// `tensor-capi` accessor may run on the same handle, from any thread, while
/// the returned reference is live.
pub(crate) unsafe fn imp_mut<'a>(t: *mut EfTensor) -> Option<&'a mut EfTensorImpl> {
    unsafe {
        if t.is_null() {
            return None;
        }
        Some(&mut *(t as *mut EfTensorImpl))
    }
}

// Each accessor catches panics and returns a benign value.

/// Number of dimensions in the addressing grid; 0 for an invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_ndim(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => 0,
            Some(i) => i.shape_u64.len() as u32,
        }))
        .unwrap_or(0)
    }
}

/// Borrowed pointer to `ndim` dimension extents, valid while `t` lives.
///
/// Returns NULL for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_shape(t: *const EfTensor) -> *const u64 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => std::ptr::null(),
            Some(i) => i.shape_u64.as_ptr(),
        }))
        .unwrap_or(std::ptr::null())
    }
}

/// Borrowed pointer to `ndim` strides in bytes, valid while `t` lives.
///
/// Returns NULL for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_strides(t: *const EfTensor) -> *const i64 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => std::ptr::null(),
            Some(i) => i.strides_i64.as_ptr(),
        }))
        .unwrap_or(std::ptr::null())
    }
}

/// The `ef_dtype` code of the addressing grid's element type.
///
/// Returns 0 for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_dtype(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => 0,
            Some(i) => i.inner.dtype().code(),
        }))
        .unwrap_or(0)
    }
}

/// The `ef_storage_kind` code of the backing store.
///
/// Returns 0 for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_storage_kind(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => 0,
            Some(i) => i.inner.memory().code(),
        }))
        .unwrap_or(0)
    }
}

/// Number of planes; 1 for a bare (formatless) tensor.
///
/// Returns 0 for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_plane_count(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => 0,
            Some(i) => match i.inner.format() {
                None => 1,
                Some(f) => {
                    let stride = i.inner.effective_row_stride().unwrap_or(0);
                    let shape = i.inner.shape();
                    let (w, h) = if shape.len() >= 2 {
                        (shape[1], shape[0])
                    } else {
                        (0, 0)
                    };
                    f.plane_table(w, h, stride)
                        .map(|p| p.len() as u32)
                        .unwrap_or(1)
                }
            },
        }))
        .unwrap_or(0)
    }
}

/// Borrowed format descriptor, "" when this is not an image.
///
/// Returns NULL for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_format(t: *const EfTensor) -> *const c_char {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => std::ptr::null(),
            Some(i) => i.format_c.as_ptr(),
        }))
        .unwrap_or(std::ptr::null())
    }
}

/// Packed colorimetry (`space | transfer<<8 | encoding<<16 | range<<24`),
/// or 0 when undefined. See `edgefirst_tensor::Colorimetry::pack`.
///
/// Returns 0 for an invalid or NULL handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_colorimetry(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| match imp(t) {
            None => 0,
            Some(i) => i.colorimetry.load(std::sync::atomic::Ordering::Relaxed),
        }))
        .unwrap_or(0)
    }
}

/// Describe plane `index`. Returns 0 on success, EINVAL otherwise.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_plane_at(
    t: *const EfTensor,
    index: u32,
    out: *mut EfTensorPlane,
) -> std::ffi::c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let (Some(i), false) = (imp(t), out.is_null()) else {
                return libc::EINVAL;
            };
            let planes = match i.inner.format() {
                Some(f) => {
                    let shape = i.inner.shape();
                    let (w, h) = if shape.len() >= 2 {
                        (shape[1], shape[0])
                    } else {
                        (0, 0)
                    };
                    let stride = i.inner.effective_row_stride().unwrap_or(w);
                    match f.plane_table(w, h, stride) {
                        Some(p) => p,
                        None => return libc::EINVAL,
                    }
                }
                // A bare (formatless) tensor is one plane spanning its whole
                // allocation: `size` is the allocation's byte count, and
                // `stride` is the recorded row stride or, when none is set,
                // the same byte count.
                None => {
                    let cap = i.inner.capacity_bytes() as u64;
                    vec![edgefirst_tensor::PlaneGeometry {
                        offset: 0,
                        stride: i.inner.row_stride().map(|s| s as u64).unwrap_or(cap),
                        size: cap,
                    }]
                }
            };
            let Some(g) = planes.get(index as usize) else {
                return libc::EINVAL;
            };
            *out = EfTensorPlane {
                handle: native_handle(&i.inner),
                offset: g.offset,
                stride: g.stride,
                size: g.size,
                used: g.size,
                modifier: 0,
            };
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Bytes of the underlying allocation, which is `>=` the tensor's logical
/// size.
///
/// @retval `>= 0` the allocation's byte count.
/// @retval `-1` `t` is `NULL` or invalid.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_capacity_bytes(t: *const EfTensor) -> i64 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp(t) else {
                return -1;
            };
            i64::try_from(i.inner.capacity_bytes()).unwrap_or(-1)
        }))
        .unwrap_or(-1)
    }
}

/// The recorded row stride in bytes, or `-1` when none is recorded
/// (tightly packed).
///
/// `ef_tensor_plane_at`'s `stride` is the effective pitch: the recorded one
/// when there is one, else a pitch computed from the format and width.
///
/// @retval `>= 0` the recorded row stride in bytes.
/// @retval `-1` `t` is `NULL`/invalid, or no row stride is recorded.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_row_stride(t: *const EfTensor) -> i64 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp(t) else {
                return -1;
            };
            i.inner
                .row_stride()
                .and_then(|s| i64::try_from(s).ok())
                .unwrap_or(-1)
        }))
        .unwrap_or(-1)
    }
}

/// The vendor tile-compression scheme this tensor's allocation resolved to,
/// as an `ef_compression` code; `EF_COMPRESSION_NONE` (0) for a linear
/// layout. Non-zero only for an Android AHardwareBuffer allocation that
/// requested compression and received it.
///
/// Returns `EF_COMPRESSION_NONE` (0) for a `NULL`/invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_compression(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp(t) else {
                return EfCompression::None as u32;
            };
            crate::codes::compression_code(i.inner.compression())
        }))
        .unwrap_or(EfCompression::None as u32)
    }
}

/// The CPU access declared for this tensor, as an `ef_cpu_access` code: set
/// at allocation, possibly narrowed by `ef_tensor_restrict_cpu_access`. A
/// map, copy or sync bracket whose direction it does not cover fails with
/// `EACCES`.
///
/// Returns `EF_CPU_ACCESS_NONE` (0) for a `NULL`/invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_cpu_access(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp(t) else {
                return edgefirst_tensor::CpuAccess::None.code();
            };
            i.inner.cpu_access().code()
        }))
        .unwrap_or(edgefirst_tensor::CpuAccess::None.code())
    }
}

/// Narrow this tensor's declared CPU access to the directions both it and
/// `access` (an `ef_cpu_access` code) cover, on the chroma plane too.
/// Never widens: a code wider than the declaration leaves it unchanged and
/// is not an error.
///
/// @return 0, or `EINVAL` for a `NULL`/invalid handle or an unknown code.
///
/// # Safety
/// `t` must be `NULL` or a live handle with no concurrent `tensor-capi`
/// calls on it.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_restrict_cpu_access(
    t: *mut EfTensor,
    access: u32,
) -> std::ffi::c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(access) = edgefirst_tensor::CpuAccess::from_code(access) else {
                crate::last_error::set_last_error("restrict_cpu_access: unknown access code");
                return libc::EINVAL;
            };
            let Some(i) = imp_mut(t) else {
                crate::last_error::set_last_error("restrict_cpu_access: could not resolve handle");
                return libc::EINVAL;
            };
            i.inner.restrict_cpu_access(access);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Whether this tensor's memory is known to be physically contiguous, as an
/// `ef_contiguity` code: contiguous for DMA-BUF allocated from the CMA heap,
/// non-contiguous for the system heap, unknown for an imported fd and every
/// other kind of memory.
///
/// Returns unknown (0) for a `NULL`/invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_contiguity(t: *const EfTensor) -> u32 {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp(t) else {
                return EfContiguity::Unknown as u32;
            };
            i.inner.contiguity().code()
        }))
        .unwrap_or(EfContiguity::Unknown as u32)
    }
}

/// Describe this tensor's parent-region snapshot, if it is a `view`/`batch`
/// sub-region. `out->has_origin` is 0 for a whole tensor, in which case the
/// rest of `out` is zeroed and unused.
///
/// Returns 0 on success (`out` always written), EINVAL for a NULL/invalid
/// handle or a NULL `out`.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_view_origin(
    t: *const EfTensor,
    out: *mut EfViewOrigin,
) -> std::ffi::c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let (Some(i), false) = (imp(t), out.is_null()) else {
                return libc::EINVAL;
            };
            *out = match i.inner.view_origin() {
                Some(vo) => EfViewOrigin {
                    parent_width: vo.parent_width as u64,
                    parent_height: vo.parent_height as u64,
                    parent_row_stride: vo.parent_row_stride as u64,
                    x: vo.x as u64,
                    y: vo.y as u64,
                    has_origin: 1,
                },
                None => EfViewOrigin::default(),
            };
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Whether this tensor addresses part of a larger buffer: a `view`/`batch`
/// sub-region, or a tensor rebuilt from one's descriptor and marked with
/// `ef_tensor_mark_view`. A CPU write into such a tensor maps read-write.
///
/// Returns 1 for a view, 0 for a whole tensor or a `NULL`/invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_is_view(t: *const EfTensor) -> std::ffi::c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            imp(t).map_or(0, |i| std::ffi::c_int::from(i.inner.is_view()))
        }))
        .unwrap_or(0)
    }
}

/// Record that this tensor addresses part of a larger buffer, which
/// `ef_tensor_is_view` then reports. Never cleared.
///
/// @return 0, or `EINVAL` for a `NULL`/invalid handle.
///
/// # Safety
/// `t` must be `NULL` or a live handle with no concurrent `tensor-capi`
/// calls on it.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_mark_view(t: *mut EfTensor) -> std::ffi::c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(i) = imp_mut(t) else {
                crate::last_error::set_last_error("mark_view: could not resolve handle");
                return libc::EINVAL;
            };
            i.inner.mark_view();
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// This tensor's shareable native handle, or -1 when it has none.
fn native_handle(t: &TensorDyn) -> i64 {
    match t.memory() {
        #[cfg(target_os = "linux")]
        edgefirst_tensor::TensorMemory::DmaBuf => {
            use std::os::fd::AsRawFd;
            t.dmabuf().map(|fd| fd.as_raw_fd() as i64).unwrap_or(-1)
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        edgefirst_tensor::TensorMemory::DmaBuf => {
            t.iosurface_id().map(|id| id as i64).unwrap_or(-1)
        }
        _ => -1,
    }
}

/// Drop one reference; destroy at zero.
unsafe fn release_own(t: *mut EfTensor) {
    unsafe {
        let imp = &*(t as *const EfTensorImpl);
        if imp.refs.fetch_sub(1, std::sync::atomic::Ordering::Release) == 1 {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            // Drop an outstanding `ef_tensor_map` guard and log a warning.
            let mut slot = imp
                .map_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.take().is_some() {
                #[cfg(test)]
                crate::map::test_support::FREED_WITH_OUTSTANDING_MAP
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                log::warn!(
                    "ef_tensor_free: tensor freed with an outstanding \
                 ef_tensor_map still held -- the caller leaked the map; \
                 dropping the mapping guard now so its sync bracket \
                 still runs"
                );
            }
            drop(slot);
            drop(Box::from_raw(t as *mut EfTensorImpl));
        }
    }
}

/// Derive the three caches [`EfTensorImpl`] keeps alongside `inner`
/// (`shape_u64`, `strides_i64`, `format_c`) from its current state.
fn derive_caches(inner: &TensorDyn) -> (Vec<u64>, Vec<i64>, CString) {
    let shape_u64: Vec<u64> = inner.shape().iter().map(|d| *d as u64).collect();
    let strides_i64 = edgefirst_tensor::protocol::c_byte_strides(
        &shape_u64,
        inner.dtype().size() as i64,
        inner.format().map(|f| f.layout()),
        inner.row_stride(),
    );
    let format_c =
        CString::new(inner.format().map(|f| f.as_str()).unwrap_or("")).unwrap_or_default();
    (shape_u64, strides_i64, format_c)
}

/// Wrap a Rust tensor in a C handle. The caller owns the result.
pub(crate) fn into_handle(inner: TensorDyn) -> *mut EfTensor {
    let (shape_u64, strides_i64, format_c) = derive_caches(&inner);
    let colorimetry = inner.colorimetry().map(|c| c.pack()).unwrap_or(0);
    let boxed = Box::new(EfTensorImpl {
        map_state: std::sync::Mutex::new(None),
        inner,
        shape_u64,
        strides_i64,
        format_c,
        refs: std::sync::atomic::AtomicUsize::new(1),
        colorimetry: std::sync::atomic::AtomicU32::new(colorimetry),
    });
    Box::into_raw(boxed) as *mut EfTensor
}

/// Re-derive `shape_u64`/`strides_i64`/`format_c` from `imp.inner`'s
/// current shape, row stride and format. Does not touch `colorimetry`.
pub(crate) fn refresh_caches(imp: &mut EfTensorImpl) {
    let (shape_u64, strides_i64, format_c) = derive_caches(&imp.inner);
    imp.shape_u64 = shape_u64;
    imp.strides_i64 = strides_i64;
    imp.format_c = format_c;
}

/// The tensor behind a handle, for in-crate tests.
#[cfg(test)]
pub(crate) fn inner_of<'a>(t: *const EfTensor) -> &'a TensorDyn {
    // SAFETY: `t` came from `into_handle` in the same test.
    unsafe { &(*(t as *const EfTensorImpl)).inner }
}

/// The tensor behind a handle, for this crate's own entry points.
pub(crate) fn tensor_of<'a>(t: *const EfTensor) -> Option<&'a TensorDyn> {
    // SAFETY: `imp` checks `t` for null; `t` came from `into_handle`.
    unsafe { imp(t).map(|i| &i.inner) }
}

/// The full implementation behind a handle, for this crate's own entry
/// points.
///
/// # Safety
/// `t` must be `NULL` or a live handle produced by [`into_handle`].
pub(crate) unsafe fn impl_of<'a>(t: *const EfTensor) -> Option<&'a EfTensorImpl> {
    unsafe { imp(t) }
}

/// Allocate a host-memory tensor.
///
/// Returns `NULL` on failure.
///
/// # Safety
/// `dims` must point to `ndim` readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_new(dtype: u32, dims: *const u64, ndim: u32) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if dims.is_null() || ndim == 0 {
                return std::ptr::null_mut();
            }
            let Some(dt) = DType::from_code(dtype) else {
                return std::ptr::null_mut();
            };
            let shape: Vec<usize> = std::slice::from_raw_parts(dims, ndim as usize)
                .iter()
                .map(|d| *d as usize)
                .collect();
            match TensorDyn::new(&shape, dt, Some(TensorMemory::Mem), None) {
                Ok(t) => into_handle(t),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Caller-owned debug name. Free with `free(3)`.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_name(t: *const EfTensor) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(inner) = tensor_of(t) else {
            return std::ptr::null_mut();
        };
        let name = inner.name().replace('\0', "?");
        CString::new(name)
            .map(CString::into_raw)
            .unwrap_or(std::ptr::null_mut())
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Wrap a caller-owned host allocation as a tensor that aliases it, without
/// copying or owning it.
///
/// `capacity` is the allocation's size in bytes, which is `>=` the tight
/// footprint `dims` implies. Pass 0 for the tight footprint.
///
/// The returned tensor does not keep `ptr` alive: it is valid only while
/// the caller keeps that memory alive.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` for a `NULL` `ptr`/`dims`, `ndim == 0`, or an
///         unrecognized `dtype` -- `ef_tensor_last_error_message` carries
///         the reason, `ef_tensor_last_error_class` is
///         `EF_ERROR_CLASS_INVALID_ARGUMENT`, and `errno` is `EINVAL`.
///
/// # Safety
/// `ptr` must be non-null, aligned for `dtype`, and valid for
/// `max(capacity, product(dims) * sizeof(dtype))` bytes for as long as the
/// returned tensor and every view/map sharing its backing is used. `dims`
/// must point to `ndim` readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_wrap_host(
    ptr: *mut u8,
    capacity: usize,
    dtype: u32,
    dims: *const u64,
    ndim: u32,
) -> *mut EfTensor {
    unsafe {
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shape) = read_dims(dims, ndim, "wrap_host") else {
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            if ptr.is_null() {
                set_errno(libc::EINVAL);
                set_last_error_classified(EfErrorClass::InvalidArgument, "wrap_host: null pointer");
                return std::ptr::null_mut();
            }
            let Some(dt) = DType::from_code(dtype) else {
                set_errno(libc::EINVAL);
                set_last_error_classified(
                    EfErrorClass::InvalidArgument,
                    &format!("wrap_host: unknown dtype code {dtype}"),
                );
                return std::ptr::null_mut();
            };
            // A `capacity` of 0 selects the tight footprint.
            let capacity = if capacity == 0 {
                shape.iter().product::<usize>() * dt.size()
            } else {
                capacity
            };
            match TensorDyn::from_raw_host_with_capacity(ptr, &shape, capacity, dt, None) {
                Ok(t) => into_handle(t),
                Err(e) => {
                    set_errno(libc::EINVAL);
                    set_last_error_classified(
                        crate::last_error::class_of(&e),
                        &format!("wrap_host: {e}"),
                    );
                    std::ptr::null_mut()
                }
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Platforms: macOS, iOS.
///
/// Wrap a live IOSurface, named by its cross-process `IOSurfaceID`, as a
/// tensor.
///
/// The symbol is exported on every platform and returns `NULL` off Apple
/// platforms.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`. It
///         holds its own retain on the surface.
/// @retval `NULL` off Apple platforms, for a `NULL` `dims`/`ndim == 0`/
///         unrecognized `dtype`, or for an `id` no live surface has --
///         `ef_tensor_last_error_message` carries the reason.
///
/// # Safety
/// `dims` must point to `ndim` readable `uint64_t`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_from_iosurface_id(
    id: u32,
    dtype: u32,
    dims: *const u64,
    ndim: u32,
) -> *mut EfTensor {
    unsafe {
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shape) = read_dims(dims, ndim, "from_iosurface_id") else {
                return std::ptr::null_mut();
            };
            let Some(dt) = DType::from_code(dtype) else {
                crate::last_error::set_last_error(&format!(
                    "from_iosurface_id: unknown dtype code {dtype}"
                ));
                return std::ptr::null_mut();
            };
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                match TensorDyn::from_iosurface_id(id, &shape, dt, None) {
                    Ok(t) => into_handle(t),
                    Err(e) => {
                        crate::last_error::set_last_error_classified(
                            crate::last_error::class_of(&e),
                            &format!("from_iosurface_id: {e}"),
                        );
                        std::ptr::null_mut()
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                let _ = (id, dt, shape);
                crate::last_error::set_last_error(
                    "from_iosurface_id: IOSurface import is Apple-platform only",
                );
                std::ptr::null_mut()
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Read a `(dims, ndim)` pair into a shape, or set the last error and
/// return `None`.
///
/// # Safety
/// `dims` must be `NULL` or point to `ndim` readable `uint64_t`.
pub(crate) unsafe fn read_dims(dims: *const u64, ndim: u32, what: &str) -> Option<Vec<usize>> {
    if dims.is_null() || ndim == 0 {
        crate::last_error::set_errno(libc::EINVAL);
        crate::last_error::set_last_error_classified(
            edgefirst_tensor_abi::EfErrorClass::InvalidArgument,
            &format!("{what}: null dims or zero ndim"),
        );
        return None;
    }
    // SAFETY: the caller contracts `dims` is readable for `ndim` entries.
    let raw = unsafe { std::slice::from_raw_parts(dims, ndim as usize) };
    raw.iter()
        .map(|d| usize::try_from(*d).ok())
        .collect::<Option<Vec<usize>>>()
        .or_else(|| {
            crate::last_error::set_errno(libc::EINVAL);
            crate::last_error::set_last_error_classified(
                edgefirst_tensor_abi::EfErrorClass::InvalidArgument,
                &format!("{what}: a dimension is out of range for this host's usize"),
            );
            None
        })
}

/// Release one reference to a tensor handle. Freeing `NULL` is a no-op.
///
/// `ef_tensor` is refcounted: `ef_tensor_retain` adds a reference and this
/// function releases one. The tensor is destroyed when the last reference
/// is released. A handle the caller holds a reference to remains valid
/// after another reference's `ef_tensor_free` call returns.
///
/// Accepts a handle from any EdgeFirst library.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library. `t` must
/// not be used after the release of the caller's own last reference to it.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_free(t: *mut EfTensor) {
    unsafe {
        if t.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| {
            release_own(t);
        }));
    }
}

/// Add one reference, keeping the tensor alive until a matching
/// `ef_tensor_free` releases it.
///
/// # Safety
/// `t` must be NULL or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_retain(t: *mut EfTensor) -> std::ffi::c_int {
    unsafe {
        crate::last_error::shield_int(|| {
            if t.is_null() {
                crate::last_error::set_last_error("retain: null tensor");
                return libc::EINVAL;
            }
            (*(t as *const EfTensorImpl))
                .refs
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            0
        })
    }
}

/// Set or clear colorimetry metadata on a live handle. `packed` is
/// `Colorimetry::pack`'s wire form; 0 clears it.
///
/// This function and [`ef_tensor_colorimetry`] may be called concurrently
/// on the same handle from any number of threads, with no external locking.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_colorimetry(
    t: *mut EfTensor,
    packed: u32,
) -> std::ffi::c_int {
    unsafe {
        crate::last_error::shield_int(|| {
            if t.is_null() {
                crate::last_error::set_last_error("set_colorimetry: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = imp(t) else {
                crate::last_error::set_last_error("set_colorimetry: null tensor");
                return libc::EINVAL;
            };
            imp.colorimetry
                .store(packed, std::sync::atomic::Ordering::Relaxed);
            0
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn freeing_a_null_handle_is_survivable() {
        unsafe { ef_tensor_free(std::ptr::null_mut()) };
    }

    #[test]
    fn restrict_cpu_access_never_widens_and_rejects_bad_codes() {
        use edgefirst_tensor::CpuAccess;
        let t = edgefirst_tensor::TensorDyn::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            Some(edgefirst_tensor::TensorMemory::Mem),
            CpuAccess::Read,
        )
        .expect("Read-declared image");
        let h = into_handle(t);
        unsafe {
            assert_eq!(
                ef_tensor_restrict_cpu_access(h, CpuAccess::ReadWrite.code()),
                0
            );
            assert_eq!(ef_tensor_cpu_access(h), CpuAccess::Read.code());
            assert_eq!(ef_tensor_restrict_cpu_access(h, 4), libc::EINVAL);
            assert_eq!(ef_tensor_cpu_access(h), CpuAccess::Read.code());
            assert_eq!(
                ef_tensor_restrict_cpu_access(std::ptr::null_mut(), CpuAccess::Read.code()),
                libc::EINVAL
            );
            assert_eq!(ef_tensor_restrict_cpu_access(h, CpuAccess::Write.code()), 0);
            assert_eq!(ef_tensor_cpu_access(h), CpuAccess::None.code());
            ef_tensor_free(h);
        }
    }

    #[test]
    fn mark_view_is_reported_and_rejects_a_null_handle() {
        let t = edgefirst_tensor::TensorDyn::image(
            4,
            4,
            edgefirst_tensor::PixelFormat::Rgba,
            edgefirst_tensor::DType::U8,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::Write,
        )
        .expect("Write-declared image");
        let h = into_handle(t);
        unsafe {
            assert_eq!(ef_tensor_is_view(h), 0);
            assert_eq!(ef_tensor_mark_view(h), 0);
            assert_eq!(ef_tensor_is_view(h), 1);
            assert_eq!(ef_tensor_is_view(std::ptr::null()), 0);
            assert_eq!(ef_tensor_mark_view(std::ptr::null_mut()), libc::EINVAL);
            ef_tensor_free(h);
        }
    }

    #[test]
    fn retain_extends_a_tensors_life_and_free_releases() {
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);
        // The first free releases the retained reference.
        unsafe { ef_tensor_free(t) };
        assert_eq!(unsafe { ef_tensor_ndim(t) }, 1);
        // The second free destroys the tensor.
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn concurrent_colorimetry_read_and_write_do_not_race() {
        // One thread sets colorimetry while another reads it through a
        // second reference to the same handle.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);

        let addr = t as usize;
        const ITERS: u32 = 200_000;

        let writer = std::thread::spawn(move || {
            let t = addr as *mut EfTensor;
            for i in 0..ITERS {
                unsafe { ef_tensor_set_colorimetry(t, i) };
            }
        });
        let reader = std::thread::spawn(move || {
            let t = addr as *const EfTensor;
            for _ in 0..ITERS {
                let _ = unsafe { ef_tensor_colorimetry(t) };
            }
        });

        writer.join().expect("writer thread must not panic");
        reader.join().expect("reader thread must not panic");

        // Release both references.
        unsafe { ef_tensor_free(t) };
        unsafe { ef_tensor_free(t) };
    }
}
