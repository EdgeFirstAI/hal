// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The cross-package tensor protocol.
//!
//! Independently-linked EdgeFirst packages cannot share a Rust type: each
//! Python extension module statically links its own copy of this crate, and
//! PyO3 caches every `#[pyclass]`'s type object in a static belonging to the
//! module that defined it. Two modules therefore see two distinct Python
//! classes even when the Rust types are identical.
//!
//! So the contract between packages is a **descriptor**, not a type. A producer
//! exposes `__edgefirst_tensor__()` returning a `PyCapsule` named
//! `edgefirst_tensor_v1` wrapping [`TensorDesc`]; a consumer reads the
//! descriptor and never performs an `isinstance` check. That duck typing is
//! what makes the protocol survive independent release cadences, and it is the
//! same pattern numpy, pyarrow and DLPack use.
//!
//! DLPack cannot serve this role alone: it has no device type for dma-buf,
//! IOSurface or PBO, which are precisely the zero-copy cases this library
//! exists for.

/// Protocol version, checked by `TensorDyn::import_descriptor` after the
/// payload has already been read as this build's `TensorDesc`.
///
/// This is the **second line of defense, not the gate**: it cannot catch a
/// producer whose `TensorDesc` is a different *size*, since the
/// out-of-bounds/misaligned read has already happened by the time this
/// field could be inspected. The capsule name (`edgefirst_tensor_v1`, see
/// `INTEROP.md`'s Versioning section) is what actually gates that --
/// checked by `PyCapsule::pointer_checked` before any byte of the payload
/// is read. DLPack reached the same conclusion after shipping an
/// unversioned `dltensor` capsule and finding no upgrade path: its v1.0 fix
/// added both a version field *and* a new capsule name, because the field
/// alone is unreadable until the layout is already known.
///
/// This field stays useful for the case the name rule does not cover on its
/// own: a future change to what a *same-sized, same-layout* `TensorDesc`'s
/// fields mean. It has not needed to move past `1`.
pub const ABI_VERSION: u32 = 1;

/// Element type, as reported in [`TensorDesc::dtype`].
///
/// These codes cross a package boundary and are read by value, so they are
/// part of the ABI: append, never reorder. Re-exported from the wire module
/// [`crate::DType`] itself generates (via `ef_vocabulary!`, see
/// `vocabulary.rs`), so this and [`crate::DType::code`] are two views of the
/// same declared literal per variant, not two hand-kept tables that could
/// drift apart.
///
/// `protocol::dtype` is the canonical, documented path -- the source module,
/// `crate::dtype_wire`, is `#[doc(hidden)]` because it exists only so this
/// re-export has something public to alias; it is emission plumbing, not a
/// second API for the same codes.
pub use crate::dtype_wire as dtype;

/// Map a [`crate::DType`] to its wire code.
///
/// Compiled into both backends: [`from_parts`] is reached from
/// `TensorDyn::descriptor_pinned`, which now exists on the `dynamic`
/// backend too (task P2a). Nothing here touches storage -- it is the wire
/// encoding of facts both backends can already answer -- so there was never
/// a backend-specific reason for the gate this replaces, only a
/// dead-code-warning one.
pub(crate) fn dtype_of(d: crate::DType) -> u32 {
    d.code()
}

/// Map a wire [`dtype`] code back to a [`crate::DType`]. Inverse of
/// [`dtype_of`]. `None` for a code this build does not recognise (e.g. a
/// dtype added by a newer producer this build predates) — the caller
/// rejects rather than guessing.
pub fn dtype_to_dtype(code: u32) -> Option<crate::DType> {
    crate::DType::from_code(code)
}

/// HAL pixel format, as reported in [`TensorDesc::format`].
///
/// Same ABI concern as [`dtype`]: these codes cross a package boundary, so
/// they are appended, never reordered. [`crate::PixelFormat`] itself
/// generates this module's per-variant constants (via `ef_vocabulary!`, see
/// `vocabulary.rs`), so this and [`crate::PixelFormat::code`] are two views
/// of the same declared literal per variant, not two hand-kept tables that
/// could drift apart. [`format::NONE`] is the one constant this module adds
/// beyond what the macro emits -- see its own doc.
///
/// This exists because [`TensorDesc::fourcc`] cannot represent every HAL
/// format: `PlanarRgb`/`PlanarRgba` have no standard FourCC and both encode
/// as `fourcc = 0`, the same sentinel a non-image tensor uses, so `fourcc`
/// alone cannot tell "Planar RGB" apart from "no format". `fourcc` stays for
/// third-party/DRM interop; a HAL-aware consumer prefers `format` and
/// falls back to `fourcc` only when `format` is `0`.
///
/// `protocol::format` is the canonical, documented path -- the source
/// module, `crate::pixel_format_wire`, is `#[doc(hidden)]` because it exists
/// only so this re-export has something public to alias; it is emission
/// plumbing, not a second API for the same codes.
pub mod format {
    /// No format (non-image tensor), or a format this build predates. Not a
    /// [`crate::PixelFormat`] variant -- every code `1..=11` below is one of
    /// those, generated by the same `ef_vocabulary!` declaration as
    /// [`crate::PixelFormat::code`]; `0` is a sentinel this module alone
    /// defines, since "no format" has no corresponding Rust variant.
    pub const NONE: u32 = 0;
    pub use crate::format::pixel_format_wire::*;
}

/// Map a [`crate::PixelFormat`] to its [`mod@format`] wire code.
///
/// Compiled into both backends: same reason as [`dtype_of`].
pub(crate) fn format_of(f: crate::PixelFormat) -> u32 {
    f.code()
}

/// Map a [`mod@format`] wire code back to a [`crate::PixelFormat`]. `None` for
/// [`format::NONE`] or a code this build does not recognise (e.g. a format
/// added by a newer producer).
pub fn format_from_code(code: u32) -> Option<crate::PixelFormat> {
    crate::PixelFormat::from_code(code)
}

/// Bit flags, as reported in [`TensorDesc::flags`].
///
/// A bitfield rather than more discrete fields: the additions this protocol
/// is most likely to need are booleans, and each one spent as its own `u32`
/// would grow the descriptor -- which after release costs a capsule-name
/// bump. Spending one `u32` now buys 32 of them for free.
pub mod flags {
    /// [`TensorDesc::sync`] carries a real handle. When clear, `sync` is
    /// meaningless and must not be waited on.
    ///
    /// A flag rather than a sentinel because every plausible sentinel is a
    /// legal handle somewhere: `0` is a valid `GLsync`-adjacent value and a
    /// valid fd number is never negative, so `-1` and `0` cannot both be
    /// spare across the kinds this field spans.
    pub const SYNC_PRESENT: u32 = 1 << 0;
}

/// Backing store kind, as reported in [`TensorDesc::kind`].
pub mod kind {
    /// Host memory: `Mem` or `Shm`. `ptr` is valid; `handle` is -1.
    pub const HOST: u32 = 0;
    /// Linux dma-buf. `handle` is the fd.
    pub const DMABUF: u32 = 1;
    /// Apple IOSurface. `handle` is the surface id.
    pub const IOSURFACE: u32 = 2;
    /// OpenGL pixel buffer object. `handle` is the buffer id.
    pub const PBO: u32 = 3;
    /// CUDA device memory. `ptr` is a device pointer, not host-addressable.
    pub const CUDA_DEVICE: u32 = 4;
}

/// A raw pointer that may cross a `PyCapsule`.
///
/// `PyCapsule::new` requires `T: Send`, which `*mut u8` is not. The producer
/// discharges that obligation by keeping the owning tensor alive for the
/// capsule's lifetime via the capsule's keepalive; see the module docs.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct SendPtr(pub *mut u8);

// SAFETY: the pointer is only dereferenced by a consumer that received it
// through a capsule whose keepalive holds the producing tensor alive, so the
// memory outlives every use.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// A null pointer, for descriptors whose memory is not host-addressable.
    pub const fn null() -> Self {
        SendPtr(std::ptr::null_mut())
    }
    /// True when the descriptor carries no host address.
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// The cross-package descriptor. Layout is part of the contract: it is
/// `#[repr(C)]` and its size is pinned by test.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorDesc {
    /// [`ABI_VERSION`]. Checked, never assumed.
    pub version: u32,
    /// One of the [`kind`] constants.
    pub kind: u32,
    /// dma-buf fd, IOSurface id or GL buffer id; `-1` when unused.
    pub handle: i64,
    /// Host pointer when addressable, else null -- **except under
    /// [`kind::PBO`]**, where it instead carries a `*const
    /// crate::pbo::PboOpsVtable`, or null if the producer attached none.
    /// Interpretation is keyed by [`Self::kind`], the same established
    /// pattern [`Self::sync`]'s own doc comment already documents for that
    /// field: a PBO tensor never takes a `HostPin` (`PboTensor` refuses
    /// `pin_host` — the address cannot outlive the map), so this field is
    /// otherwise permanently unused for that kind, which is exactly why it
    /// is the field that carries the vtable instead of a new one requiring
    /// a layout change. Valid under the exact same borrow contract as the
    /// host-pointer case: only while the producer's own keepalive (for the
    /// capsule protocol, the capsule's) keeps the owning tensor alive.
    pub ptr: SendPtr,
    /// Number of significant entries in `shape` and `strides`.
    pub ndim: u32,
    /// Element type, one of the [`dtype`] constants.
    ///
    /// Deliberately placed here: `ndim` leaves a 4-byte hole before the
    /// 8-aligned `shape`, so this field costs nothing. Without it `strides`
    /// — documented in *elements* — cannot be converted to bytes by a
    /// consumer, and a `Tensor<f32>` is indistinguishable from a
    /// `Tensor<u8>` of the same shape.
    pub dtype: u32,
    /// Logical shape, most-significant dimension first.
    pub shape: [u64; 8],
    /// Strides **in bytes**, signed so a consumer can express flips.
    ///
    /// Bytes rather than elements so a hardware row pitch is always
    /// representable: a pitch need not be a whole number of `dtype` elements
    /// (and a sub-byte dtype has no element size to divide by). Multiply a
    /// coordinate by the stride directly; do not scale by `dtype` size.
    pub strides: [i64; 8],
    /// DRM FourCC of the pixel format, or 0 for a non-image tensor. Kept for
    /// third-party/DRM interop; a HAL-aware consumer should prefer
    /// [`Self::format`] and fall back to this only when `format` is 0
    /// (see [`mod@format`]'s doc for why `fourcc` alone is insufficient).
    pub fourcc: u32,
    /// One of the [`mod@format`] constants, or 0 for a non-image tensor.
    pub format: u32,
    /// Packed colorimetry, or 0 when unspecified.
    pub colorimetry: u32,
    /// Bit flags; see [`mod@flags`]. Zero when nothing applies.
    ///
    /// Occupies what was a 4-byte alignment hole before `capacity`, so it
    /// cost nothing to add.
    pub flags: u32,
    /// Bytes available in the producer's underlying allocation, from this
    /// tensor's window start (its [`TensorTrait::capacity_bytes`
    /// ](crate::TensorTrait::capacity_bytes)) -- may exceed the byte size
    /// `shape`/`strides` imply when the producer over-allocated: a pool
    /// tensor sized for its largest expected image but holding a smaller
    /// one today, or row-pitch/MCU-alignment padding a decoder needs beyond
    /// the logical width. Without this, a consumer importing a `HOST`
    /// descriptor has no way to learn that headroom exists and must treat
    /// the declared shape as the allocation's exact size -- which starves a
    /// later `configure_image`/`set_logical_shape` of capacity the producer
    /// actually has. `DMABUF`/`IOSURFACE` consumers query their real
    /// capacity from the handle itself and do not depend on this field, but
    /// it is populated for every kind for consistency.
    pub capacity: u64,
    /// Completion handle for work still in flight against this tensor, or 0
    /// when [`flags::SYNC_PRESENT`] is clear.
    ///
    /// **Reserved: producers in this build always leave it clear.** It is
    /// declared now because the field has to exist before the first release
    /// or adding it later costs a capsule-name bump, and it is the one gap
    /// the synchronous-at-the-boundary contract cannot close: a *foreign*
    /// producer with queued GPU work has no other way to say "wait on this
    /// first".
    ///
    /// Interpretation is keyed by [`Self::kind`], since each backing already
    /// implies its own fence flavour and a separate discriminant would be
    /// redundant:
    ///
    /// * [`kind::DMABUF`] -- a `sync_file` fd from
    ///   `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`. The consumer owns it and must
    ///   close it.
    /// * [`kind::PBO`] -- a `GLsync` from `glFenceSync`, valid only in the
    ///   producer's share group.
    /// * [`kind::CUDA_DEVICE`] -- a `cudaEvent_t`.
    /// * [`kind::HOST`], [`kind::IOSURFACE`] -- no fence flavour defined;
    ///   `SYNC_PRESENT` must be clear.
    pub sync: u64,
}

impl TensorDesc {
    /// Maximum rank the fixed-size `shape`/`strides` arrays can carry.
    pub const MAX_NDIM: usize = 8;

    /// The logical shape as a slice.
    pub fn shape(&self) -> &[u64] {
        &self.shape[..(self.ndim as usize).min(Self::MAX_NDIM)]
    }

    /// The strides as a slice, in bytes.
    pub fn strides(&self) -> &[i64] {
        &self.strides[..(self.ndim as usize).min(Self::MAX_NDIM)]
    }

    /// Total element count.
    pub fn len(&self) -> u64 {
        self.shape().iter().product()
    }

    /// True when the descriptor addresses no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Producer side
// ---------------------------------------------------------------------------

use crate::TensorMemory;

/// Map a backing store to its [`kind`] constant.
///
/// `TensorMemory::DmaBuf` is a deliberately shared discriminant covering DMA-BUF,
/// IOSurface and AHardwareBuffer (it is ABI-stable across platforms), so the
/// platform decides which protocol kind it reports.
pub(crate) fn kind_of(memory: TensorMemory) -> u32 {
    match memory {
        // Both are a host-addressable pointer with no handle, which is all
        // `kind` records -- it is a narrower vocabulary than `TensorMemory`
        // on purpose, not a copy of it that lost a variant.
        TensorMemory::Mem | TensorMemory::Shm => kind::HOST,
        TensorMemory::Pbo => kind::PBO,
        TensorMemory::Cuda => kind::CUDA_DEVICE,
        // Named specifically rather than through `DmaBuf`. Unreachable
        // today (no backend reports it) but correct the day one does, which
        // is cheaper than the panic the obvious `unreachable!()` becomes.
        TensorMemory::IoSurface => kind::IOSURFACE,
        TensorMemory::DmaBuf => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                kind::IOSURFACE
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                kind::DMABUF
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Consumer side: the per-kind field checks `TensorDyn::import_storage` needs
// ---------------------------------------------------------------------------
//
// Each of these validates one kind's handle/pointer field and produces the
// value a constructor takes. They live here, next to the descriptor they
// read, rather than in either backend: a descriptor is an *untrusted*
// cross-package payload, and two copies of "is this fd representable" or
// "is this surface id in range" are two places for the refusals to drift.

/// `dup` a `kind::DMABUF` descriptor's fd into one this process owns.
///
/// The producer still owns the original and will close it when its own
/// tensor drops, so an import must have its own -- `from_fd` adopts what it
/// is given.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor carries no fd, or
/// one not representable as a file descriptor. `as i32` would wrap a handle
/// above `i32::MAX` onto an unrelated -- possibly live -- descriptor, which
/// the `dup` would then adopt as the buffer.
#[cfg(target_os = "linux")]
pub(crate) fn dup_descriptor_fd(desc: &TensorDesc) -> crate::Result<std::os::fd::OwnedFd> {
    use std::os::fd::BorrowedFd;
    if desc.handle < 0 {
        return Err(crate::Error::InvalidArgument(
            "dma-buf descriptor carries no fd".into(),
        ));
    }
    let raw_fd = i32::try_from(desc.handle).map_err(|_| {
        crate::Error::InvalidArgument(format!(
            "dma-buf descriptor fd {} is not representable as a file descriptor",
            desc.handle
        ))
    })?;
    // SAFETY: the producer owns `desc.handle` and the capsule keepalive
    // holds it open for the duration of this call, so it is a valid fd to
    // borrow (not adopt) for the dup below.
    let borrowed = unsafe { BorrowedFd::borrow_raw(raw_fd) };
    Ok(nix::unistd::dup(borrowed)?)
}

/// Read a `kind::IOSURFACE` descriptor's surface id.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor carries no id, or
/// one out of `u32` range. Liveness is *not* checked here -- that belongs
/// with the lookup, which happens inside the constructor (and, under the
/// `dynamic` backend, inside `libedgefirst_tensor.so`) so no window exists
/// between "found it" and "retained it".
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn descriptor_surface_id(desc: &TensorDesc) -> crate::Result<u32> {
    if desc.handle < 0 {
        return Err(crate::Error::InvalidArgument(
            "IOSurface descriptor carries no surface id".into(),
        ));
    }
    u32::try_from(desc.handle).map_err(|_| {
        crate::Error::InvalidArgument(format!(
            "IOSurface descriptor id {} is out of range",
            desc.handle
        ))
    })
}

/// Refuse a `kind::HOST` descriptor that carries no address.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`], naming the producer-side fix: a
/// descriptive-only capsule was requested, so no address was pinned.
pub(crate) fn check_descriptor_host_ptr(desc: &TensorDesc) -> crate::Result<()> {
    if desc.ptr.is_null() {
        return Err(crate::Error::InvalidArgument(
            "host descriptor has no address: the producer was asked \
             for a descriptive-only capsule (request access=\"read\" \
             or \"readwrite\")"
                .into(),
        ));
    }
    Ok(())
}

/// Read a `kind::PBO` descriptor's GL buffer id, and refuse one carrying no
/// ops vtable.
///
/// Both checks together, unlike the other kinds' one apiece, because the
/// `PBO` kind is the one that needs *two* of the descriptor's fields to be
/// present before a constructor can be called at all -- `handle` names the
/// buffer, `ptr` carries the vtable that can reach it, and either alone is
/// useless.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the buffer id is absent or out of
/// range, or when no vtable address is carried.
pub(crate) fn descriptor_pbo_buffer_id(desc: &TensorDesc) -> crate::Result<u32> {
    if desc.handle < 0 {
        return Err(crate::Error::InvalidArgument(
            "PBO descriptor carries no buffer id".into(),
        ));
    }
    let buffer_id = u32::try_from(desc.handle).map_err(|_| {
        crate::Error::InvalidArgument(format!(
            "PBO descriptor buffer id {} is out of range",
            desc.handle
        ))
    })?;
    if desc.ptr.is_null() {
        return Err(crate::Error::InvalidArgument(
            "PBO descriptor carries no ops vtable -- the producer either \
             predates this build's cross-cdylib PBO support or its own \
             PboTensor could not build one"
                .into(),
        ));
    }
    Ok(buffer_id)
}

/// Inputs to [`from_parts`], grouped into one named-field struct rather than
/// a positional argument list.
///
/// This crossed nine parameters once the pixel-format and capacity inputs
/// were added, several of them same-typed (`fourcc: u32`, a `u32`
/// handle-adjacent value, `colorimetry: u32`) — a positional call at that
/// width compiles happily with two arguments transposed and silently
/// corrupts the descriptor. Named fields make that a compile error instead. Internal to
/// this crate: `TensorDyn::descriptor_pinned` is presently the only caller --
/// on both backends now, not just `static` (task P2a).
pub(crate) struct DescParts<'a, 'p> {
    pub dims: &'a [usize],
    pub memory: TensorMemory,
    pub dtype: crate::DType,
    /// DRM FourCC for [`TensorDesc::fourcc`]; 0 when the format has none.
    pub fourcc: u32,
    /// The HAL format, source for both the row-dimension decision below and
    /// [`TensorDesc::format`] (via [`format_of`]).
    pub format: Option<crate::PixelFormat>,
    /// Row pitch in bytes; see [`from_parts`]'s doc for how it maps to a
    /// stride index.
    pub row_stride: Option<usize>,
    pub handle: i64,
    pub colorimetry: u32,
    /// Producer's [`crate::TensorTrait::capacity_bytes`] for
    /// [`TensorDesc::capacity`]; see that field's docs.
    pub capacity: u64,
    pub pin: Option<&'a crate::HostPin<'p>>,
    /// The producer's `PboOpsVtable` address, for [`TensorDesc::ptr`] under
    /// [`kind::PBO`] — see that field's own doc comment. `None` for every
    /// other kind (mutually exclusive with `pin`: a PBO tensor never has
    /// one, since `PboTensor` refuses `pin_host`).
    pub pbo_vtable_ptr: Option<*const std::ffi::c_void>,
}

/// Build a descriptor from its parts.
///
/// The single construction site: `TensorDyn::descriptor` and any future
/// `Tensor<T>::descriptor` both route here, so the layout and the stride
/// convention are defined once.
///
/// `ptr` is null unless a [`HostPin`](crate::HostPin) is supplied, or the
/// tensor is PBO-backed and `pbo_vtable_ptr` is supplied instead — see
/// [`TensorDesc::ptr`]'s own doc comment for why that one field carries two
/// different meanings by [`TensorDesc::kind`]. A raw address is only
/// meaningful while something guarantees the memory (or, for PBO, the
/// `PboHandle` the vtable addresses) stays put, so the descriptor carries
/// one only when the producer has pinned or built a vtable — and the
/// capsule's keepalive then holds the relevant tensor alive for the
/// consumer's benefit either way.
///
/// `row_stride` is the producer's [`TensorDyn::row_stride`](crate::TensorDyn::row_stride)
/// **in bytes**, or `None` for a tightly-packed tensor. A pitch-aligned image
/// carries padding between rows that the shape alone cannot express, so when
/// present it overrides the stride of the dimension that is actually the row
/// dimension for `format`'s layout. `format` is also the source of
/// [`TensorDesc::format`] (via [`format_of`]) and is taken separately
/// from `fourcc` rather than derived from it: `PlanarRgb`/`PlanarRgba` both
/// encode as `fourcc = 0` (no standard FourCC exists for them), the same
/// sentinel a non-image tensor uses, so `fourcc` alone cannot tell Planar
/// apart from "no format" — for either purpose.
///
/// * Packed (`[H, W, C]`) / SemiPlanar (`[combined_H, W]`): dimension 0.
/// * Planar (`[C, H, W]`, e.g. `PlanarRgb`): dimension 1 — dimension 0 there
///   is the *plane* count, and clobbering it with the row pitch would
///   corrupt the plane stride while leaving the true per-row stride
///   (dimension 1) unfixed. `Tensor` tracks only one row-pitch scalar (no
///   separate per-plane stride), so dimension 0 keeps its packed value here
///   exactly as it did before any padding was known — unchanged, not
///   recomputed.
///
/// `row_stride` is a byte pitch and is carried verbatim. It need not be a
/// whole number of `dtype` elements: `set_row_stride` validates only
/// `stride >= minimum`, not element alignment, and since strides are bytes
/// there is no division to lose a remainder to. The element-stride
/// representation this replaced could not express such a pitch at all — it
/// fell back to the packed stride and reported a pitch the buffer does not
/// have, across a package/ABI boundary.
///
/// `handle`, `colorimetry` and `capacity` are carried through verbatim; `-1`
/// and `0` are the "unused"/"undefined" sentinels for the first two.
pub(crate) fn from_parts(parts: DescParts) -> TensorDesc {
    let DescParts {
        dims,
        memory,
        dtype,
        fourcc,
        format,
        row_stride,
        handle,
        colorimetry,
        capacity,
        pin,
        pbo_vtable_ptr,
    } = parts;
    // `ndim` carries the TRUE rank even when it exceeds the eight slots the
    // descriptor can hold, so a consumer can detect the case and refuse.
    // Reporting the clamped value instead would ship a descriptor whose
    // `ndim` disagrees with its own `dims`, and the consumer would address
    // `product(dims[..8])` elements of a larger allocation -- silently wrong
    // data across a package boundary. This mirrors the `row_stride` override
    // below, which likewise refuses to truncate quietly.
    let ndim = dims.len();
    let filled = ndim.min(TensorDesc::MAX_NDIM);
    if ndim > TensorDesc::MAX_NDIM {
        log::warn!(
            "tensor of rank {ndim} exceeds the descriptor's {} shape slots; \
             the descriptor will be rejected on import",
            TensorDesc::MAX_NDIM
        );
    }
    let mut shape = [0u64; TensorDesc::MAX_NDIM];
    let mut strides = [0i64; TensorDesc::MAX_NDIM];
    for (i, d) in dims.iter().take(filled).enumerate() {
        shape[i] = *d as u64;
    }
    // Row-major and contiguous, in BYTES: stride[i] = dtype size * product of
    // trailing dims. Bytes, not elements, because a hardware pitch need not be
    // a whole number of elements (and sub-byte dtypes have no element size at
    // all) -- see the `row_stride` override below.
    let mut acc: i64 = dtype.size() as i64;
    // `filled`, not `ndim`: the arrays hold MAX_NDIM slots and `ndim` may now
    // legitimately exceed that.
    for i in (0..filled).rev() {
        strides[i] = acc;
        acc *= shape[i] as i64;
    }
    if ndim >= 2 {
        if let Some(rs) = row_stride {
            // Planar stacks channels ahead of rows ([C, H, W]): the row
            // dimension is 1, not 0. Every other layout this protocol
            // carries (Packed [H, W, C], SemiPlanar [combined_H, W]) has the
            // row dimension at 0. `format` is `None` for a non-image tensor,
            // in which case `row_stride` should not have been supplied in
            // the first place; dimension 0 is a harmless default for that
            // case since it matches the pre-padding behaviour.
            let row_dim = match format.map(|f| f.layout()) {
                Some(crate::PixelLayout::Planar) if ndim >= 3 => 1,
                _ => 0,
            };
            // Bytes in, bytes out. The element-stride representation had to
            // divide here, and a remainder made the pitch unrepresentable --
            // it fell back to the packed stride and reported a pitch the
            // buffer does not have. There is nothing left to get wrong.
            strides[row_dim] = rs as i64;
        }
    }
    TensorDesc {
        version: ABI_VERSION,
        kind: kind_of(memory),
        handle,
        // `pin` and `pbo_vtable_ptr` are mutually exclusive by construction
        // (see `DescParts::pbo_vtable_ptr`'s own doc comment) -- `pin`
        // takes precedence purely because it is the older, more general
        // case; a real caller only ever supplies one.
        ptr: pin
            .map(|p| SendPtr(p.as_mut_ptr()))
            .or_else(|| pbo_vtable_ptr.map(|p| SendPtr(p as *mut u8)))
            .unwrap_or_else(SendPtr::null),
        ndim: ndim as u32,
        dtype: dtype_of(dtype),
        shape,
        strides,
        fourcc,
        format: format.map(format_of).unwrap_or(format::NONE),
        colorimetry,
        // Reserved; no producer in this build sets either. See the field docs.
        flags: 0,
        capacity,
        sync: 0,
    }
}
