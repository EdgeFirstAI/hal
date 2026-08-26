// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The tensor handle and its accessors.
//!
//! Sibling `-capi` leaves link `libedgefirst_tensor.so` rather than embedding
//! a private copy of the implementation, so every `ef_tensor` in the process
//! is the same [`EfTensorImpl`] layout. Accessors cast the opaque handle and
//! read it; there is no dispatch table. Sibling libraries export their own
//! constructors (`ef_image_processor_create_image` and so on); everything
//! past construction goes through this library.
//!
//! Layout that crosses the boundary — [`EfTensorPlane`] plus the
//! dtype/storage-kind vocabularies — is declared in `edgefirst-tensor-abi`.
//! The handle representation here is internal and is never declared in
//! `tensor.h`.
//!
//! # Layout
//!
//! [`EfTensor`] is a fully opaque zero-sized marker: C never dereferences it
//! directly, only ever holds a `*mut EfTensor`/`*const EfTensor` and passes
//! it back through an exported accessor. The real allocation is
//! [`EfTensorImpl`], reached by reinterpreting the same pointer.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{DType, TensorDyn, TensorMemory};
pub use edgefirst_tensor_abi::{EfCompression, EfTensorPlane, EfViewOrigin};

/// An opaque tensor handle.
///
/// Zero-sized on purpose: C only ever holds a pointer to this and passes it
/// back through an exported accessor, which reinterprets the same address as
/// [`EfTensorImpl`]. There is exactly one implementation library, so no
/// vtable or type tag is needed to tell handles apart.
#[repr(C)]
pub struct EfTensor {
    _private: [u8; 0],
}

/// The real allocation behind an [`EfTensor`].
///
/// The cached vectors exist because the accessors hand C a borrowed pointer:
/// `shape()` returns `&[usize]`, which is neither `u64` nor guaranteed to
/// outlive the call, so the C-visible representation is materialised once at
/// construction.
///
/// `map_state` holds the outstanding `ef_tensor_map` guard, if any (see
/// `crate::map`). It borrows nothing from `inner` -- `TensorDyn::map_bytes`
/// returns a genuinely `'static` `HostView`, sharing ownership through its
/// pin's keepalive `Arc` rather than referencing this struct -- so, unlike a
/// self-referential field, its declaration order relative to `inner` carries
/// no soundness requirement. `release_own` still drains it before dropping
/// the `Box`, but that is drop hygiene and a leak diagnostic (a caller who
/// forgot to `ef_tensor_unmap`), not a lifetime obligation.
#[repr(C)]
pub(crate) struct EfTensorImpl {
    pub(crate) map_state: std::sync::Mutex<Option<crate::map::MapState>>,
    pub(crate) inner: TensorDyn,
    shape_u64: Vec<u64>,
    strides_i64: Vec<i64>,
    format_c: CString,
    /// CPU-side handle count. `crate::map`'s exclusive-write gate reads this
    /// directly, hence `pub(crate)` rather than private to this module.
    pub(crate) refs: std::sync::atomic::AtomicUsize,
    /// Packed colorimetry (`Colorimetry::pack`'s wire form), the sole
    /// authority for [`ef_tensor_colorimetry`]/[`ef_tensor_set_colorimetry`]
    /// -- **not** a cache of `inner`'s own `colorimetry` field. Seeded from
    /// it once in [`into_handle`] and never read back through `inner`
    /// afterward, which is what lets both C entry points use `&self`-only
    /// atomic ops instead of ever taking `&mut EfTensorImpl`: a `&mut` to
    /// this struct while another thread holds a live `&EfTensorImpl` (e.g.
    /// any other accessor's `imp(t)`, or a concurrent reader/writer on the
    /// same retained handle -- `ef_tensor` is refcounted and explicitly
    /// designed to cross threads, see `dynamic_backend.rs`'s module docs)
    /// is undefined behaviour regardless of what a lock's *timing*
    /// enforces, because the aliasing violation is about the reference
    /// itself existing, not about whether the racing accesses happen to
    /// touch the same bytes. `inner.colorimetry()`/`set_colorimetry()`
    /// stay the source of truth for every purely-Rust caller (there are
    /// none in this crate today -- confirmed by grep, `serialize.rs`'s
    /// blob export never touches colorimetry), so nothing here needs those
    /// two views kept in sync after construction.
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
/// Used by `mutate.rs`/`quant.rs` for the handful of setters that change
/// `inner`'s own fields (format, row stride, plane offset, shape,
/// quantization) rather than a side field of `EfTensorImpl` -- unlike
/// [`ef_tensor_set_colorimetry`], which stores into a dedicated `AtomicU32`
/// specifically so it never needs `&mut EfTensorImpl` (see that function's
/// doc comment and the F12 regression it fixed), these mutate `inner` in
/// place because a live "geometry" primitive (shape, stride, format
/// validation) genuinely does not fit a single atomic scalar the way a
/// packed colorimetry value does.
///
/// # Safety
/// `t` must be a live handle produced by [`into_handle`].
///
/// **This establishes exclusivity against nothing but the compiler's own
/// aliasing check at this call site -- it does NOT make mutation through the
/// returned reference safe under concurrent access from another thread.**
/// `ef_tensor` handles are refcounted and designed to cross threads
/// (`ef_tensor_retain`), and every *read-only* accessor in this file
/// (`ef_tensor_dtype`, `ef_tensor_plane_at`, `ef_tensor_view_origin`, ...)
/// takes only `&EfTensorImpl` and can be called from another thread at any
/// time. A `&mut EfTensorImpl` obtained here that overlaps in time with any
/// of those is the same class of aliasing violation F12 fixed for
/// colorimetry -- this helper does not fix that class of bug, it only
/// avoids widening it beyond the specific setters that call it. Those
/// setters' own doc comments state the resulting constraint plainly: the
/// caller must not call any other `tensor-capi` accessor on the same handle,
/// from any thread, while one of these mutators is in flight. This is a
/// real, narrower guarantee than [`ef_tensor_set_colorimetry`]'s, not an
/// oversight -- every real caller of these setters (`import_image`,
/// `configure_image`'s pool-reuse path, quantization setup at model load)
/// calls them once, before the tensor is retained or shared with another
/// thread, which is the shape of use this constraint fits.
pub(crate) unsafe fn imp_mut<'a>(t: *mut EfTensor) -> Option<&'a mut EfTensorImpl> {
    unsafe {
        if t.is_null() {
            return None;
        }
        Some(&mut *(t as *mut EfTensorImpl))
    }
}

// Every accessor is reachable from C with a null or foreign pointer, and an
// unwind across the FFI boundary is undefined behaviour. Each one therefore
// catches panics and returns a benign value rather than propagating.
// Written out literally, not via a `macro_rules!` helper: cbindgen parses
// this file's syntax directly and does not expand user macros, so a
// macro-generated `#[no_mangle]` function is invisible to it -- the header
// would silently stop declaring every accessor. (Confirmed the hard way:
// routing these through a macro first passed `cargo build` clean but made
// every one of these vanish from tensor.h.)

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

/// Borrowed pointer to `ndim` strides in BYTES, valid while `t` lives.
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
/// Returns 0 for an invalid or NULL handle -- indistinguishable from a
/// genuinely undefined colorimetry, which is the same ambiguity every other
/// "0 means absent" wire value in this ABI already accepts.
///
/// # Safety
/// `t` must be `NULL` or a live handle from an EdgeFirst library.
// This reads the dedicated `colorimetry` atomic (see `EfTensorImpl`'s field
// doc), not `inner`, and `Relaxed` suffices because nothing else is
// synchronized through this value -- it is an independent packed scalar,
// not a guard for any other memory.
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
                // allocation -- `size` is always that allocation's real byte
                // count. `stride` used to be the same `capacity_bytes()` value
                // unconditionally, which was never observably wrong before this
                // task: no `ef_tensor_*` primitive could set a formatless
                // tensor's `row_stride` field (`Tensor::set_row_stride` itself
                // requires a format), so `row_stride()` was always `None` here
                // and the two fell back to the same value. Task 17 added
                // `ef_tensor_set_row_stride_unchecked`, precisely so a raw
                // multiplane chroma plane (which by contract carries no format;
                // see `Tensor::from_planes`) can record its own pitch -- once
                // that primitive can set it, this accessor must read it back, or
                // every caller of `ef_tensor_plane_at`/`effective_row_stride` on
                // such a tensor would see the *whole buffer size* as its row
                // pitch instead of the real one, the same "answer that looks
                // plausible but is not the real one" class of bug `plane_offset`
                // turned out to be in task 15.
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
/// size -- a pool tensor holding a smaller decoded image, or a
/// pitch-aligned image whose padding the shape alone cannot express.
///
/// The producer side of [`edgefirst_tensor::TensorDesc::capacity`]: a
/// consumer re-importing this tensor's memory needs the real allocation
/// size, not the size the shape implies. Not derivable from
/// `ef_tensor_plane_at`: for a *formatted* tensor that reports per-plane
/// geometry over a computed plane table, whose sum is the logical image
/// size and not the allocation's.
///
/// `-1` is a genuine sentinel, matching `ef_tensor_plane_offset`'s: a byte
/// count can never be negative.
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

/// The *recorded* row stride in bytes, or `-1` when none is recorded
/// (tightly packed).
///
/// Deliberately distinct from `ef_tensor_plane_at`'s `stride`, which is the
/// *effective* pitch -- the recorded one when there is one, else a pitch
/// computed from the format and width. The difference is load-bearing for
/// [`edgefirst_tensor::TensorDyn::descriptor`]: the cross-package protocol
/// carries `None` for a tight tensor and lets the consumer recompute, and
/// baking a computed pitch in instead would turn "no stride recorded" into
/// "this exact stride is required" across a package boundary.
///
/// `-1` is a genuine sentinel, matching `ef_tensor_plane_offset`'s: a byte
/// pitch can never be negative.
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

/// The vendor tile-compression scheme this tensor's allocation actually
/// resolved to; `EF_COMPRESSION_NONE` (0) for a linear layout, which is the
/// answer everywhere except an Android AHardwareBuffer allocation that both
/// requested compression and got it.
///
/// Returns 0 rather than an error code for a `NULL`/invalid handle: the
/// return type is the vocabulary itself, with no spare bit pattern to carry
/// a failure, and "linear" is the conservative answer -- a consumer that
/// treats an unreadable handle as linear reads plausible bytes, whereas one
/// that treated it as tiled would decode garbage. Callers who need to
/// distinguish an invalid handle have `ef_tensor_dtype`/`ef_tensor_ndim`
/// for that.
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
            // An outstanding `ef_tensor_map` at last-reference time is a caller
            // bug (map without a matching unmap), but the platform sync bracket
            // — the mmap, the IOSurface lock, whatever the guard's Drop runs —
            // still has to fire before `inner` goes away, or it leaks silently
            // inside the freed Box instead. Loud, not silent: this is the
            // "released with a loud warning" the map-window spec calls for.
            // Poison recovery mirrors the map/unmap sites: the slot is a
            // single-assignment `Option`, so a lock poisoned by a shielded
            // panic still holds a coherent value and the drain must not be
            // skipped because of it.
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
///
/// Shared by [`into_handle`] (first derivation) and [`refresh_caches`]
/// (re-derivation after a mutator changes `inner`'s shape, stride, or
/// format) so the two never drift apart.
fn derive_caches(inner: &TensorDyn) -> (Vec<u64>, Vec<i64>, CString) {
    let shape_u64: Vec<u64> = inner.shape().iter().map(|d| *d as u64).collect();
    let esz = inner.dtype().size() as i64;
    let mut strides_i64 = vec![0i64; shape_u64.len()];
    let mut acc = esz;
    for i in (0..shape_u64.len()).rev() {
        strides_i64[i] = acc;
        acc *= shape_u64[i] as i64;
    }
    if let (Some(rs), true) = (inner.row_stride(), shape_u64.len() >= 2) {
        strides_i64[0] = rs as i64;
    }
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

/// Re-derive `shape_u64`/`strides_i64`/`format_c` after a `mutate.rs`
/// setter changes `imp.inner`'s shape, row stride, or format -- otherwise
/// `ef_tensor_shape`/`ef_tensor_strides`/`ef_tensor_format` (which read the
/// cached copies, not `inner` live) would report stale geometry the moment
/// after, e.g., `ef_tensor_configure_image` genuinely changed it. Does not
/// touch `colorimetry`: no mutator this task adds changes it.
pub(crate) fn refresh_caches(imp: &mut EfTensorImpl) {
    let (shape_u64, strides_i64, format_c) = derive_caches(&imp.inner);
    imp.shape_u64 = shape_u64;
    imp.strides_i64 = strides_i64;
    imp.format_c = format_c;
}

/// The tensor behind a handle, for in-crate tests.
///
/// Not exported: C reaches the tensor only through the exported accessors.
#[cfg(test)]
pub(crate) fn inner_of<'a>(t: *const EfTensor) -> &'a TensorDyn {
    // SAFETY: `t` came from `into_handle` in the same test.
    unsafe { &(*(t as *const EfTensorImpl)).inner }
}

/// The tensor behind a handle, for this crate's own entry points.
pub(crate) fn tensor_of<'a>(t: *const EfTensor) -> Option<&'a TensorDyn> {
    // SAFETY: `t` is checked non-null; the caller contracts it came from here.
    unsafe { imp(t).map(|i| &i.inner) }
}

/// The full implementation behind a handle, for this crate's own entry
/// points that need more than [`tensor_of`] alone -- `crate::map` needs
/// `map_state` alongside `inner`.
///
/// # Safety
/// `t` must be `NULL` or a live handle produced by [`into_handle`].
pub(crate) unsafe fn impl_of<'a>(t: *const EfTensor) -> Option<&'a EfTensorImpl> {
    unsafe { imp(t) }
}

/// Allocate a host-memory tensor.
///
/// The simplest constructor: `edgefirst-tensor` handles `mem` allocation
/// itself, with no `ImageProcessor` involved. Returns `NULL` on failure.
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
            .map(|s| s.into_raw())
            .unwrap_or(std::ptr::null_mut())
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Wrap a caller-owned host allocation as a tensor, aliasing it rather than
/// copying or owning it -- the consumer half of the cross-package capsule
/// protocol's `HOST` kind.
///
/// `capacity` is the producer's real allocation size, which is `>=` the
/// tight footprint `dims` implies: a pool tensor, or one padded to a
/// decoder's pitch alignment, is larger than the shape it currently
/// reports, and without carrying it the alias would be clamped to today's
/// shape and unable to grow back into memory the producer actually has.
/// Pass 0 to mean "exactly the tight footprint".
///
/// **The returned tensor does not keep `ptr` alive.** It is valid only
/// while the producer keeps that memory alive, which for the capsule
/// protocol is the capsule keepalive's job; nothing here takes a reference
/// to extend it. See
/// [`edgefirst_tensor::TensorDyn::import_descriptor`].
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` for a `NULL` `ptr`/`dims`, `ndim == 0`, or an
///         unrecognized `dtype` -- `ef_tensor_last_error_message` carries
///         the reason.
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
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
        crate::last_error::ensure_hook_installed();
        catch_unwind(AssertUnwindSafe(|| {
            let Some(shape) = read_dims(dims, ndim, "wrap_host") else {
                return std::ptr::null_mut();
            };
            if ptr.is_null() {
                crate::last_error::set_last_error("wrap_host: null pointer");
                return std::ptr::null_mut();
            }
            let Some(dt) = DType::from_code(dtype) else {
                crate::last_error::set_last_error(&format!(
                    "wrap_host: unknown dtype code {dtype}"
                ));
                return std::ptr::null_mut();
            };
            // 0 means "no separate capacity declared": fall back to the
            // tight footprint, which is what `from_raw_host` (the
            // capacity-less constructor) records. A sentinel rather than a
            // presence flag because a zero-byte allocation has nothing to
            // alias in the first place -- `ptr` would have to be dangling.
            let capacity = if capacity == 0 {
                shape.iter().product::<usize>() * dt.size()
            } else {
                capacity
            };
            match TensorDyn::from_raw_host_with_capacity(ptr, &shape, capacity, dt, None) {
                Ok(t) => into_handle(t),
                Err(e) => {
                    crate::last_error::set_last_error_classified(
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

/// Wrap a live IOSurface, named by its cross-process `IOSurfaceID`, as a
/// tensor (macOS/iOS only) -- the consumer half of the capsule protocol's
/// `IOSURFACE` kind.
///
/// Declared on every platform and refused at runtime off Apple, rather than
/// existing only in an Apple build: this library's ABI surface is the same
/// set of symbols everywhere, the same rule `ef_storage_kind` follows for
/// naming `IO_SURFACE` on Linux. A platform-conditional symbol would make
/// "does this build have it" a link-time question for every consumer.
///
/// IDs are reused after a surface is freed, so a stale one fails rather
/// than resolving to an unrelated buffer.
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
        // The quiet hook, before the catch: a caught panic must WRITE the
        // thread-local, or a consumer reading `ef_tensor_last_error_class`
        // after this returns NULL gets a class left behind by an earlier
        // failure and reports it as this call's. See `ensure_hook_installed`.
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
/// Shared by the constructors above rather than repeated: the null/zero
/// check and the `u64 -> usize` narrowing are exactly the places two
/// hand-written copies drift into accepting different arguments.
///
/// # Safety
/// `dims` must be `NULL` or point to `ndim` readable `uint64_t`.
unsafe fn read_dims(dims: *const u64, ndim: u32, what: &str) -> Option<Vec<usize>> {
    if dims.is_null() || ndim == 0 {
        crate::last_error::set_last_error(&format!("{what}: null dims or zero ndim"));
        return None;
    }
    // SAFETY: the caller contracts `dims` is readable for `ndim` entries.
    let raw = unsafe { std::slice::from_raw_parts(dims, ndim as usize) };
    raw.iter()
        .map(|d| usize::try_from(*d).ok())
        .collect::<Option<Vec<usize>>>()
        .or_else(|| {
            crate::last_error::set_last_error(&format!(
                "{what}: a dimension is out of range for this host's usize"
            ));
            None
        })
}

/// Release one reference to a tensor handle. Freeing `NULL` is a no-op.
///
/// `ef_tensor` is refcounted: `ef_tensor_retain` adds a reference and this
/// function is the release. The tensor is destroyed only when the *last*
/// reference is released -- whoever still holds a reference keeps the tensor
/// alive, the `GstBuffer`/`CVPixelBuffer` convention. A handle you hold a
/// reference to remains valid after another reference's `ef_tensor_free`
/// call returns; it is a use-after-free only once *your own* last reference
/// has been released.
///
/// Works on a handle from **any** EdgeFirst library: every library links
/// this one's shared implementation, so every handle is the exact same
/// layout, allocated by the exact same allocator.
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

/// Attach/clear colorimetry metadata on a live handle. `packed` is
/// `Colorimetry::pack`'s wire form; 0 clears it (matching `pack`'s own
/// all-`None`-maps-to-0 convention, so a caller can round-trip
/// [`ef_tensor_colorimetry`] straight back through this without special-
/// casing "no colorimetry").
///
/// **Concurrency.** `ef_tensor` is refcounted and its handles are designed
/// to cross threads (`ef_tensor_retain` is exactly how two threads come to
/// legitimately hold the same handle). This function and
/// [`ef_tensor_colorimetry`] are safe to call concurrently -- from any
/// number of threads, holding valid references to the same handle, in any
/// interleaving -- with no external locking required around either call.
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
            // Deliberately `&EfTensorImpl` (via `imp`, the same shared-reference
            // helper every read-only accessor uses), never `&mut EfTensorImpl`:
            // an earlier version took `&mut` here while `ef_tensor_colorimetry`
            // (this function's read counterpart) takes `&` with no
            // synchronization between them -- real aliasing UB per Rust's
            // memory model the moment two threads hold the same retained
            // handle, which the ABI explicitly allows. Synchronizing only the
            // timing (e.g. a lock around the old `&mut` cast) would not have
            // fixed that: the violation is the reference itself existing while
            // another is live, independent of whether the racing accesses land
            // on the same bytes. The actual fix is that this never takes
            // `&mut EfTensorImpl` at all -- it stores straight into the
            // dedicated `colorimetry: AtomicU32` field (see that field's doc
            // comment for why it, not `inner`, is the sole authority for this
            // value) through the same `&EfTensorImpl` shape every other
            // accessor already uses. `Relaxed` suffices on both the load
            // (`ef_tensor_colorimetry`) and this store: colorimetry is an
            // independent scalar with no documented ordering requirement
            // against any other ABI-exposed state (format is immutable after
            // construction; `map_state` and `refs` are unrelated), so there is
            // no happens-before relationship for a stronger ordering to buy.
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
    fn retain_extends_a_tensors_life_and_free_releases() {
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        assert_eq!(unsafe { ef_tensor_retain(t) }, 0);
        // First free releases the retain; the handle must still answer.
        unsafe { ef_tensor_free(t) };
        assert_eq!(unsafe { ef_tensor_ndim(t) }, 1);
        // Second free destroys.
        unsafe { ef_tensor_free(t) };
    }

    #[test]
    fn concurrent_colorimetry_read_and_write_do_not_race() {
        // Regression for F12: an earlier version's `ef_tensor_set_colorimetry`
        // took `&mut EfTensorImpl` (the whole struct) while every other
        // accessor, including `ef_tensor_colorimetry`, took `&EfTensorImpl`
        // unsynchronized with it -- real aliasing UB per Rust's memory
        // model, not a benign timing race, and exactly the scenario two
        // threads sharing one handle via `ef_tensor_retain` produce (the
        // handle is refcounted and explicitly designed to cross threads).
        //
        // Ideally this runs under Miri (`cargo +nightly miri test`), whose
        // data-race detector catches the aliasing violation deterministically
        // rather than depending on hardware timing. That was attempted here
        // and blocked by an unrelated environment issue: `cargo miri test`
        // re-resolves against a newer `rustix` than the pinned lockfile
        // uses, and `dma-heap 0.4.1` (an unconditional Linux dependency of
        // `edgefirst-tensor`) fails to compile against it (`Opcode` changed
        // from `u32` to `u64`) -- present with or without `--locked`, so it
        // is a real toolchain/dependency mismatch, not something specific to
        // this test. Not chased further; still worth running under Miri once
        // that is fixed. Until then this loop is the plain-hardware fallback:
        // it does not reliably fail on every run (small structs can tear
        // without an observable wrong value on a given CPU/build), so treat
        // a pass here as "no crash observed," not proof of absence -- the
        // real guarantee is the code no longer takes `&mut EfTensorImpl`
        // while a live `&EfTensorImpl` can exist elsewhere, checked by
        // inspection above and in the SAFETY comments this change rewrote.
        let dims = [4u64];
        let t = unsafe { ef_tensor_new(0, dims.as_ptr(), 1) };
        assert!(!t.is_null());
        // Two legitimate owners of the same handle -- the scenario the bug
        // needs: `ef_tensor_retain` is exactly how a second thread comes to
        // hold the same pointer.
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

        // Two references outstanding (the initial one plus the retain
        // above): release both.
        unsafe { ef_tensor_free(t) };
        unsafe { ef_tensor_free(t) };
    }
}
