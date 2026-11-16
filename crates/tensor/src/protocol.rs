// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The cross-package tensor protocol.
//!
//! The contract between independently-linked EdgeFirst packages is a
//! descriptor, not a type. A producer exposes `__edgefirst_tensor__()`
//! returning a `PyCapsule` named `edgefirst_tensor_v2` whose payload leads
//! with [`TensorDesc`] (followed by the quantization metadata, see
//! `edgefirst-python-common`'s `QuantDesc`); a consumer reads the descriptor
//! and performs no `isinstance` check.

/// Protocol version, checked by `TensorDyn::import_descriptor` after the
/// payload has been read as this build's `TensorDesc`.
///
/// The capsule name (`edgefirst_tensor_v2`) gates the descriptor's size and
/// layout; this field covers a change to what a same-sized, same-layout
/// `TensorDesc`'s fields mean. Under version `2` [`TensorDesc::kind`] carries
/// the [`crate::TensorMemory`] codes; a version-1 descriptor is refused.
pub const ABI_VERSION: u32 = 2;

/// Element type, as reported in [`TensorDesc::dtype`].
///
/// These codes cross a package boundary and are part of the ABI. They are
/// the [`crate::DType::code`] values, re-exported from the wire module
/// [`crate::DType`] generates (via `ef_vocabulary!`, see `vocabulary.rs`).
pub use crate::dtype_wire as dtype;

/// Map a [`crate::DType`] to its wire code.
pub(crate) fn dtype_of(d: crate::DType) -> u32 {
    d.code()
}

/// Map a wire [`dtype`] code back to a [`crate::DType`]. Inverse of
/// [`dtype_of`]. `None` for a code this build does not recognise.
pub fn dtype_to_dtype(code: u32) -> Option<crate::DType> {
    crate::DType::from_code(code)
}

/// HAL pixel format, as reported in [`TensorDesc::format`].
///
/// These codes cross a package boundary and are part of the ABI. They are
/// the [`crate::PixelFormat::code`] values (generated via `ef_vocabulary!`,
/// see `vocabulary.rs`), plus [`format::NONE`].
///
/// [`TensorDesc::fourcc`] cannot represent every HAL format:
/// `PlanarRgb`/`PlanarRgba` have no standard FourCC and both encode as
/// `fourcc = 0`, the same value a non-image tensor uses. A HAL-aware
/// consumer reads `format` and falls back to `fourcc` only when `format` is
/// `0`.
pub mod format {
    /// No format (non-image tensor), or a format this build does not
    /// recognise. Not a [`crate::PixelFormat`] variant.
    pub const NONE: u32 = 0;
    pub use crate::format::pixel_format_wire::*;
}

/// Map a [`crate::PixelFormat`] to its [`mod@format`] wire code.
pub(crate) fn format_of(f: crate::PixelFormat) -> u32 {
    f.code()
}

/// Map a [`mod@format`] wire code back to a [`crate::PixelFormat`]. `None` for
/// [`format::NONE`] or a code this build does not recognise.
pub fn format_from_code(code: u32) -> Option<crate::PixelFormat> {
    crate::PixelFormat::from_code(code)
}

/// Bit flags, as reported in [`TensorDesc::flags`].
pub mod flags {
    /// [`TensorDesc::sync`] carries a real handle. When clear, `sync` is
    /// meaningless and must not be waited on.
    ///
    /// Under [`kind::D3D11_TEXTURE`] `sync` is not a handle: it is a value on
    /// the timeline of the fence whose NT handle [`TensorDesc::ptr`] carries,
    /// so a producer that sets this flag must also fill in `ptr`.
    ///
    /// Bit 0.
    pub const SYNC_PRESENT: u32 = 1;

    /// The producer reports its declared [`crate::CpuAccess`] in
    /// [`CPU_READ`] and [`CPU_WRITE`]. When clear the declaration is unknown
    /// and an importer keeps its own default. Bit 1.
    pub const CPU_ACCESS_DECLARED: u32 = 1 << 1;
    /// The declaration includes CPU reads. Bit 2.
    pub const CPU_READ: u32 = 1 << 2;
    /// The declaration includes CPU writes. Bit 3.
    pub const CPU_WRITE: u32 = 1 << 3;
    /// The tensor addresses part of a larger buffer (a `view`/`batch`
    /// sub-region, at any offset). Clear means a whole tensor. Bit 4.
    pub const VIEW: u32 = 1 << 4;

    /// The flag bits reporting `access`.
    pub fn from_cpu_access(access: crate::CpuAccess) -> u32 {
        let mut bits = CPU_ACCESS_DECLARED;
        if access.reads() {
            bits |= CPU_READ;
        }
        if access.writes() {
            bits |= CPU_WRITE;
        }
        bits
    }

    /// The declared access `flags` reports, or `None` when the producer did
    /// not report one.
    pub fn cpu_access(flags: u32) -> Option<crate::CpuAccess> {
        if flags & CPU_ACCESS_DECLARED == 0 {
            return None;
        }
        Some(crate::CpuAccess::from_directions(
            flags & CPU_READ != 0,
            flags & CPU_WRITE != 0,
        ))
    }
}

/// Backing store kind, as reported in [`TensorDesc::kind`].
///
/// These are the [`crate::TensorMemory`] codes, re-exported from the wire
/// module that `TensorMemory` generates (via `ef_vocabulary!`, see
/// `vocabulary.rs`): `DMABUF` is `2` on every surface. What each kind means
/// for the descriptor's fields:
///
/// * `MEM`, `SHM` -- host memory. `ptr` is valid; `handle` is -1. An
///   importer treats both the same way (see [`TensorDesc::is_host`]).
/// * `DMABUF` -- Linux dma-buf. `handle` is the fd.
/// * `IOSURFACE` -- Apple IOSurface. `handle` is the surface id. A macOS/iOS
///   producer reports this for its `DmaBuf`-spelled tensors.
/// * `PBO` -- OpenGL pixel buffer object. `handle` is the buffer id.
/// * `CUDA` -- CUDA device memory. `ptr` is a device pointer, not
///   host-addressable.
/// * `D3D11_TEXTURE` -- Windows D3D11 texture. `handle` is the texture's NT
///   shared handle and `ptr` the device fence's NT shared handle, both valid
///   in the producing process (other modules of the same process included);
///   `sync` is the fence value of the last recorded GPU write when
///   `SYNC_PRESENT` is set. A Windows producer reports this for its
///   `DmaBuf`-spelled tensors.
pub use crate::tensor_memory_wire as kind;

/// A raw pointer that may cross a `PyCapsule`.
///
/// The producer keeps the owning tensor alive for the capsule's lifetime via
/// the capsule's keepalive.
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

/// The cross-package descriptor. `#[repr(C)]`; its layout is part of the
/// contract.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TensorDesc {
    /// [`ABI_VERSION`].
    pub version: u32,
    /// One of the [`mod@kind`] codes, which are the [`crate::TensorMemory`]
    /// codes.
    pub kind: u32,
    /// dma-buf fd, IOSurface id, GL buffer id or D3D11 texture NT handle
    /// value; `-1` when unused.
    pub handle: i64,
    /// Host pointer when addressable, else null. Under [`kind::PBO`] it
    /// carries a `*const crate::pbo::PboOpsVtable`, or null if the producer
    /// attached none; under [`kind::D3D11_TEXTURE`] it carries the process
    /// device fence's NT handle value, or null when [`Self::sync`] carries
    /// nothing to wait on.
    ///
    /// Valid only while the producer's keepalive (for the capsule protocol,
    /// the capsule's) keeps the owning tensor alive.
    pub ptr: SendPtr,
    /// Number of significant entries in `shape` and `strides`.
    pub ndim: u32,
    /// Element type, one of the [`dtype`] constants.
    pub dtype: u32,
    /// Logical shape, most-significant dimension first.
    pub shape: [u64; 8],
    /// Strides in bytes, signed so a consumer can express flips. Multiply a
    /// coordinate by the stride directly, without scaling by `dtype` size.
    pub strides: [i64; 8],
    /// V4L2 pixel format code ([`crate::PixelFormat::to_fourcc`]) of the
    /// pixel format, or 0 for a non-image tensor or a format with no V4L2
    /// code. A HAL-aware consumer reads [`Self::format`] first and falls back
    /// to this only when `format` is 0.
    pub fourcc: u32,
    /// One of the [`mod@format`] constants, or 0 for a non-image tensor.
    pub format: u32,
    /// Packed colorimetry, or 0 when unspecified.
    pub colorimetry: u32,
    /// Bit flags; see [`mod@flags`]. Zero when nothing applies.
    pub flags: u32,
    /// Bytes available in the producer's underlying allocation, from this
    /// tensor's window start (its [`TensorTrait::capacity_bytes`
    /// ](crate::TensorTrait::capacity_bytes)). May exceed the byte size
    /// `shape`/`strides` imply when the producer over-allocated. Populated
    /// for every kind.
    pub capacity: u64,
    /// Completion handle for work still in flight against this tensor, or 0
    /// when [`flags::SYNC_PRESENT`] is clear.
    ///
    /// Filled in by [`kind::D3D11_TEXTURE`] producers, and left clear by
    /// every other kind in this build.
    ///
    /// Interpretation is keyed by [`Self::kind`]:
    ///
    /// * [`kind::DMABUF`] -- a `sync_file` fd from
    ///   `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`. The consumer owns it and must
    ///   close it.
    /// * [`kind::PBO`] -- a `GLsync` from `glFenceSync`, valid only in the
    ///   producer's share group.
    /// * [`kind::CUDA`] -- a `cudaEvent_t`.
    /// * [`kind::D3D11_TEXTURE`] -- a value on the timeline of the
    ///   `ID3D11Fence` [`Self::ptr`] names, not a handle of its own: the
    ///   handle and the value together are what a consumer waits on.
    /// * [`kind::MEM`], [`kind::SHM`], [`kind::IOSURFACE`] -- no fence
    ///   flavour defined; `SYNC_PRESENT` must be clear.
    pub sync: u64,
}

impl TensorDesc {
    /// Maximum rank the fixed-size `shape`/`strides` arrays can carry.
    pub const MAX_NDIM: usize = 8;

    /// True when [`Self::kind`] is a host-memory kind ([`kind::MEM`] or
    /// [`kind::SHM`]): `ptr` is a host address and `handle` is unused.
    pub fn is_host(&self) -> bool {
        matches!(self.kind, kind::MEM | kind::SHM)
    }

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

/// Map a backing store to its [`kind`] code.
///
/// The identity on [`TensorMemory::code`] except for `DmaBuf`, which maps to
/// the platform's concrete backing: `IOSURFACE` on macOS/iOS,
/// `D3D11_TEXTURE` on Windows, `DMABUF` elsewhere.
pub(crate) fn kind_of(memory: TensorMemory) -> u32 {
    match memory {
        TensorMemory::DmaBuf => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                kind::IOSURFACE
            }
            #[cfg(target_os = "windows")]
            {
                kind::D3D11_TEXTURE
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "windows")))]
            {
                kind::DMABUF
            }
        }
        other => other.code(),
    }
}

// ---------------------------------------------------------------------------
// Consumer side: the per-kind field checks `TensorDyn::import_storage` needs
// ---------------------------------------------------------------------------

/// `dup` a `kind::DMABUF` descriptor's fd into one this process owns.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor carries no fd, or
/// one not representable as a file descriptor.
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
    // SAFETY: the producer owns `desc.handle` and the capsule keepalive holds
    // it open for the duration of this call. It is borrowed, not adopted.
    let borrowed = unsafe { BorrowedFd::borrow_raw(raw_fd) };
    Ok(nix::unistd::dup(borrowed)?)
}

/// Read a `kind::IOSURFACE` descriptor's surface id.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor carries no id, or
/// one out of `u32` range. Liveness is not checked here.
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

/// Refuse a host-memory (`kind::MEM`/`kind::SHM`) descriptor that carries no address.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`].
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

/// Read a `kind::D3D11_TEXTURE` descriptor's texture handle and, when the
/// producer recorded a GPU write, the fence handle plus value an import
/// waits on before reading.
///
/// Both handles stay the producer's: they are valid only while the capsule
/// keepalive holds the producing tensor alive.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor carries no texture
/// handle.
#[cfg(target_os = "windows")]
pub(crate) fn descriptor_d3d11_handles(
    desc: &TensorDesc,
) -> crate::Result<(
    std::os::windows::io::RawHandle,
    Option<(std::os::windows::io::RawHandle, u64)>,
)> {
    if desc.handle <= 0 {
        return Err(crate::Error::InvalidArgument(
            "D3D11_TEXTURE descriptor without a texture handle".into(),
        ));
    }
    let tex = desc.handle as usize as std::os::windows::io::RawHandle;
    let completion = (desc.flags & flags::SYNC_PRESENT != 0 && !desc.ptr.is_null())
        .then_some((desc.ptr.0 as std::os::windows::io::RawHandle, desc.sync));
    Ok((tex, completion))
}

/// The pixel format and image dimensions a `kind::D3D11_TEXTURE` import is
/// built at, with the descriptor's shape checked against them.
///
/// The dimensions come from the texture, except a semi-planar image's width,
/// which comes from `shape`. `shape` may be the allocation shape, the
/// addressing shape, or a packed window of the texture (`[h', w', c]` with
/// `h'` and `w'` no larger than the texture's, and no smaller than one).
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the descriptor names no pixel
/// format, or when its shape is none of the three spellings of the geometry
/// the texture reports. Propagates whatever opening and describing the
/// texture reports.
///
/// # Safety
///
/// `texture` must be an NT shared handle of a D3D11 texture, valid in this
/// process. It stays owned by the caller.
#[cfg(target_os = "windows")]
pub(crate) unsafe fn descriptor_d3d11_geometry(
    desc: &TensorDesc,
    texture: std::os::windows::io::RawHandle,
    shape: &[usize],
) -> crate::Result<(crate::PixelFormat, usize, usize)> {
    let format = crate::PixelFormat::from_code(desc.format).ok_or_else(|| {
        crate::Error::InvalidArgument(
            "D3D11_TEXTURE descriptor without a pixel format: the texture's own \
             description cannot say which format it was allocated for"
                .into(),
        )
    })?;
    // SAFETY: the caller guarantees `texture` is a shared NT handle valid in
    // this process.
    let (width, height) =
        unsafe { d3d11_geometry_with_rule(format, texture, shape, ShapeRule::OrWindow) }?;
    Ok((format, width, height))
}

/// Which spellings of a texture's geometry a `shape` may take.
#[cfg(target_os = "windows")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShapeRule {
    /// The allocation shape or the addressing shape, exactly.
    #[cfg_attr(edgefirst_tensor_dynamic, allow(dead_code))]
    Exact,
    /// `Exact`, or a packed window of the texture (a `view()`).
    OrWindow,
}

/// Whether `shape` is a packed sub-rectangle of a `width` x `height` image
/// in `format`: the allocation spelling with a smaller height or width. A
/// window is never a semi-planar or planar image and never empty.
#[cfg(target_os = "windows")]
fn is_packed_window_of(
    format: crate::PixelFormat,
    shape: &[usize],
    width: usize,
    height: usize,
) -> bool {
    if format.layout() != crate::PixelLayout::Packed {
        return false;
    }
    let [h, w, c] = shape else {
        return false;
    };
    *c == format.channels() && (1..=height).contains(h) && (1..=width).contains(w)
}

/// The image dimensions a shared D3D11 texture reports, with `shape` checked
/// against them.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when `shape` is neither spelling of the
/// geometry the texture reports. Propagates whatever opening and describing
/// the texture reports.
///
/// # Safety
///
/// `texture` must be an NT shared handle of a D3D11 texture, valid in this
/// process. It stays owned by the caller.
#[cfg(target_os = "windows")]
#[cfg_attr(edgefirst_tensor_dynamic, allow(dead_code))]
pub(crate) unsafe fn d3d11_geometry_checked(
    format: crate::PixelFormat,
    texture: std::os::windows::io::RawHandle,
    shape: &[usize],
) -> crate::Result<(usize, usize)> {
    // SAFETY: forwarded unchanged.
    unsafe { d3d11_geometry_with_rule(format, texture, shape, ShapeRule::Exact) }
}

#[cfg(target_os = "windows")]
unsafe fn d3d11_geometry_with_rule(
    format: crate::PixelFormat,
    texture: std::os::windows::io::RawHandle,
    shape: &[usize],
    rule: ShapeRule,
) -> crate::Result<(usize, usize)> {
    // SAFETY: the caller guarantees `texture` is a shared NT handle valid in
    // this process; the helper opens its own texture and drops it on return.
    let (width, height) =
        unsafe { crate::d3d11_shared_handle_geometry(texture, format, Some(shape)) }?;
    let allocation = format.allocation_shape(width, height);
    let addressing = format.addressing_shape(width, height);
    if allocation.as_deref() == Some(shape) || addressing.as_deref() == Some(shape) {
        return Ok((width, height));
    }
    if rule == ShapeRule::OrWindow && is_packed_window_of(format, shape, width, height) {
        return Ok((width, height));
    }
    let window = match rule {
        ShapeRule::OrWindow => " nor a packed window",
        ShapeRule::Exact => "",
    };
    Err(crate::Error::InvalidArgument(format!(
        "D3D11 texture shape {shape:?} is neither the {format:?} allocation \
         shape {allocation:?} nor the addressing shape {addressing:?}{window} of the \
         {width}x{height} texture it names"
    )))
}

/// Inputs to [`from_parts`].
pub(crate) struct DescParts<'a, 'p> {
    pub dims: &'a [usize],
    pub memory: TensorMemory,
    pub dtype: crate::DType,
    /// V4L2 pixel format code for [`TensorDesc::fourcc`]; 0 when the format
    /// has none.
    pub fourcc: u32,
    /// The HAL format, source for both the row-dimension decision and
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
    /// [`kind::PBO`]. `None` for every other kind.
    pub pbo_vtable_ptr: Option<*const std::ffi::c_void>,
    /// The process device fence's NT handle value, for [`TensorDesc::ptr`]
    /// under [`kind::D3D11_TEXTURE`]. `0` for every other kind and for a
    /// device with no fence.
    pub fence_handle: usize,
    /// The fence value of the producer's last recorded GPU write, for
    /// [`TensorDesc::sync`] under [`kind::D3D11_TEXTURE`]. `None` for every
    /// other kind and when nothing has been recorded.
    pub sync: Option<u64>,
    /// The producer's declared CPU access, reported in [`TensorDesc::flags`].
    pub cpu_access: crate::CpuAccess,
    /// Whether the producer is a view, reported as [`flags::VIEW`].
    pub view: bool,
}

/// Row-major byte strides for `shape`, with the row dimension replaced by
/// `row_stride` when one is recorded.
///
/// The stride convention of the descriptor ([`from_parts`]), the blob
/// (`blob::export`) and the C API's `ef_tensor_strides`. See [`from_parts`]
/// for which dimension is the row dimension per layout.
pub fn c_byte_strides(
    shape: &[u64],
    esz: i64,
    layout: Option<crate::PixelLayout>,
    row_stride: Option<usize>,
) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut acc = esz;
    for (stride, dim) in strides.iter_mut().zip(shape).rev() {
        *stride = acc;
        acc *= *dim as i64;
    }
    let Some(rs) = row_stride.map(|rs| rs as i64) else {
        return strides;
    };
    // A formatted tensor one rank above its image rank is a batch `[N, ...]`
    // of images stacked at the padded pitch: the row override applies to the
    // per-image dims and `N` steps over one padded image.
    let image_rank = match layout {
        Some(crate::PixelLayout::SemiPlanar) => Some(2),
        Some(crate::PixelLayout::Packed | crate::PixelLayout::Planar) => Some(3),
        None => None,
    };
    let batch_dims = match image_rank {
        Some(rank) if shape.len() == rank + 1 => 1,
        _ => 0,
    };
    let (lead, image) = strides.split_at_mut(batch_dims);
    let image_bytes = match (layout, image, &shape[batch_dims..]) {
        (Some(crate::PixelLayout::Planar), [plane, row, _, ..], [planes, rows, ..]) => {
            *row = rs;
            *plane = rs * *rows as i64;
            *plane * *planes as i64
        }
        // A rank-1 tensor has no row to pad.
        (_, [row, _, ..], [rows, ..]) => {
            *row = rs;
            rs * *rows as i64
        }
        _ => return strides,
    };
    if let [n] = lead {
        *n = image_bytes;
    }
    strides
}

/// Build a descriptor from its parts.
///
/// `ptr` is null unless a [`HostPin`](crate::HostPin) is supplied, or the
/// tensor is PBO-backed and `pbo_vtable_ptr` is supplied; see
/// [`TensorDesc::ptr`].
///
/// `row_stride` is the producer's [`TensorDyn::row_stride`](crate::TensorDyn::row_stride)
/// in bytes, or `None` for a tightly-packed tensor. When present it overrides
/// the stride of the row dimension for `format`'s layout:
///
/// * Packed (`[H, W, C]`) / SemiPlanar (`[combined_H, W]`): dimension 0.
/// * Planar (`[C, H, W]`, e.g. `PlanarRgb`): dimension 1. Dimension 0, the
///   plane stride, becomes `row_stride * H`: a strided planar tensor stores
///   its planes as `C * H` rows of one pitch.
///
/// `row_stride` is a byte pitch and is carried verbatim; it need not be a
/// whole number of `dtype` elements.
///
/// `format` is also the source of [`TensorDesc::format`] (via [`format_of`]).
/// `handle`, `colorimetry` and `capacity` are carried through verbatim; `-1`
/// and `0` are the "unused"/"undefined" values for the first two.
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
        fence_handle,
        sync,
        cpu_access,
        view,
    } = parts;
    let desc_kind = kind_of(memory);
    // Both fields are set only under `D3D11_TEXTURE`, and only when the device
    // has a fence.
    let completion = (desc_kind == kind::D3D11_TEXTURE && fence_handle != 0)
        .then_some(sync)
        .flatten()
        .map(|value| (fence_handle, value));
    // `ndim` carries the true rank even when it exceeds the eight slots the
    // descriptor can hold.
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
    // `filled`, not `ndim`: the arrays hold MAX_NDIM slots.
    let byte_strides = c_byte_strides(
        &shape[..filled],
        dtype.size() as i64,
        format.map(|f| f.layout()),
        row_stride,
    );
    strides[..filled].copy_from_slice(&byte_strides);
    TensorDesc {
        version: ABI_VERSION,
        kind: desc_kind,
        handle,
        // Keyed by kind: under `D3D11_TEXTURE` the fence handle or null; otherwise
        // the pinned host address, else the PBO vtable address, else null.
        ptr: if desc_kind == kind::D3D11_TEXTURE {
            completion.map_or_else(SendPtr::null, |(h, _)| SendPtr(h as *mut u8))
        } else {
            pin.map(|p| SendPtr(p.as_mut_ptr()))
                .or_else(|| pbo_vtable_ptr.map(|p| SendPtr(p as *mut u8)))
                .unwrap_or_else(SendPtr::null)
        },
        ndim: ndim as u32,
        dtype: dtype_of(dtype),
        shape,
        strides,
        fourcc,
        format: format.map(format_of).unwrap_or(format::NONE),
        colorimetry,
        flags: flags::from_cpu_access(cpu_access)
            | if completion.is_some() {
                flags::SYNC_PRESENT
            } else {
                0
            }
            | if view { flags::VIEW } else { 0 },
        capacity,
        sync: completion.map_or(0, |(_, value)| value),
    }
}

// ---------------------------------------------------------------------------
// Quantization, as it crosses the capsule.
// ---------------------------------------------------------------------------

/// A `*const T` that may cross a `PyCapsule`. The producer keeps the
/// pointed-to arrays alive for the capsule's lifetime, via
/// `TensorCapsulePayload::quant_keepalive`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct SendConstPtr<T>(pub *const T);

// SAFETY: the pointer is only dereferenced by a consumer that received it
// through a capsule whose `quant_keepalive` owns the pointed-to arrays, so
// the memory outlives every use. See `QuantDesc`.
unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}

impl<T> SendConstPtr<T> {
    /// A null pointer, for a descriptor that carries no array.
    pub const fn null() -> Self {
        SendConstPtr(std::ptr::null())
    }
}

/// Quantization metadata, in a form that can cross an `.so` boundary. It
/// rides in the capsule payload beside [`crate::TensorDesc`].
///
/// The arrays are borrowed, not inlined: `scales`/`zero_points` point into
/// the `Quantization` that `TensorCapsulePayload::quant_keepalive` owns. A
/// consumer copies the values out during import (see
/// [`QuantDesc::to_quantization`]) while the capsule is alive, and does not
/// retain the pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QuantDesc {
    /// Number of entries in `scales` (and in `zero_points` when non-null).
    /// Zero means the tensor carries no quantization.
    pub len: u64,
    /// Channel axis for per-channel quantization, plus one; `0` means
    /// per-tensor.
    pub axis_plus_one: u64,
    /// `len` scale factors. Null iff `len == 0`.
    pub scales: SendConstPtr<f32>,
    /// `len` zero-points, or null for symmetric quantization.
    pub zero_points: SendConstPtr<i32>,
}

impl QuantDesc {
    /// The descriptor for a tensor with no quantization metadata.
    pub const fn absent() -> Self {
        Self {
            len: 0,
            axis_plus_one: 0,
            scales: SendConstPtr::null(),
            zero_points: SendConstPtr::null(),
        }
    }

    /// Describe `quant`, borrowing its arrays.
    ///
    /// The caller must keep `quant` alive for as long as the descriptor is
    /// readable -- in the capsule protocol, by storing the same `Arc` in
    /// `TensorCapsulePayload::quant_keepalive`.
    pub fn borrowing(quant: &crate::Quantization) -> Self {
        Self {
            len: quant.scale().len() as u64,
            axis_plus_one: quant.axis().map_or(0, |a| a as u64 + 1),
            scales: SendConstPtr(quant.scale().as_ptr()),
            zero_points: quant
                .zero_point()
                .map_or(SendConstPtr::null(), |zp| SendConstPtr(zp.as_ptr())),
        }
    }

    /// Copy the borrowed arrays into an owned [`crate::Quantization`], or `None`
    /// when the producer declared no quantization. A descriptor that declares
    /// quantization but cannot be read back as a valid
    /// [`crate::Quantization`] is an `Err`.
    ///
    /// # Safety
    ///
    /// `self` must have come from a capsule that is still alive, so the
    /// arrays `scales`/`zero_points` point at are still owned by the
    /// producer's `quant_keepalive`.
    pub unsafe fn to_quantization(&self) -> crate::Result<Option<crate::Quantization>> {
        if self.len == 0 {
            return Ok(None);
        }
        if self.scales.0.is_null() {
            return Err(crate::Error::InvalidArgument(format!(
                "malformed tensor capsule: quantization declares {} scale(s) but \
                 carries a null scale array",
                self.len
            )));
        }
        let len = self.len as usize;
        // SAFETY: the producer's `quant_keepalive` owns these arrays and the
        // capsule holding it is alive; `len` came from the same producer as the
        // pointers.
        let scales = unsafe { std::slice::from_raw_parts(self.scales.0, len) }.to_vec();
        let zero_points = if self.zero_points.0.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(self.zero_points.0, len) }.to_vec())
        };
        // Rebuilt through the named constructors, which validate the descriptor.
        let axis = self.axis_plus_one.checked_sub(1).map(|a| a as usize);
        let quant = match (axis, zero_points) {
            (None, None) if len == 1 => Ok(crate::Quantization::per_tensor_symmetric(scales[0])),
            (None, Some(zp)) if len == 1 => Ok(crate::Quantization::per_tensor(scales[0], zp[0])),
            (Some(a), None) => crate::Quantization::per_channel_symmetric(scales, a),
            (Some(a), Some(zp)) => crate::Quantization::per_channel(scales, zp, a),
            // `len > 1` with no axis: per-channel scales that name no
            // channel dimension.
            (None, _) => {
                return Err(crate::Error::InvalidArgument(format!(
                    "malformed tensor capsule: {len} quantization scales but no \
                     channel axis; per-channel quantization must name one"
                )));
            }
        };
        quant.map(Some).map_err(|e| {
            crate::Error::InvalidArgument(format!(
                "malformed tensor capsule: quantization metadata is not valid: {e}"
            ))
        })
    }
}

#[cfg(test)]
mod quant_desc_tests {
    use super::QuantDesc;
    use crate::Quantization;

    /// Borrow `q`, then copy back out of the descriptor while `q` is alive.
    fn round_trip(q: &Quantization) -> Option<Quantization> {
        let desc = QuantDesc::borrowing(q);
        // SAFETY: `q` outlives `desc`.
        unsafe { desc.to_quantization() }.expect("a well-formed descriptor must import")
    }

    #[test]
    fn absent_descriptor_is_none_not_an_error() {
        // SAFETY: an all-zero descriptor names no arrays to read.
        let imported = unsafe { QuantDesc::absent().to_quantization() };
        assert!(imported
            .expect("absent is well-formed, not an error")
            .is_none());
    }

    #[test]
    fn per_tensor_round_trips() {
        for q in [
            Quantization::per_tensor_symmetric(0.5),
            Quantization::per_tensor(0.125, -7),
        ] {
            assert_eq!(round_trip(&q).as_ref(), Some(&q));
        }
    }

    /// A per-channel symmetric quantization round-trips with its length and
    /// axis.
    #[test]
    fn per_channel_symmetric_round_trips_with_its_length_and_axis() {
        let q = Quantization::per_channel_symmetric(vec![0.5, 0.25, 0.125], 0)
            .expect("per-channel symmetric");
        let imported = round_trip(&q).expect("per-channel must survive the descriptor");
        assert_eq!(imported, q);
        assert_eq!(imported.scale(), &[0.5, 0.25, 0.125]);
        assert_eq!(imported.axis(), Some(0));
        assert_eq!(imported.zero_point(), None);
    }

    #[test]
    fn per_channel_asymmetric_round_trips_with_its_zero_points() {
        let q = Quantization::per_channel(vec![0.5, 0.25, 0.125], vec![1, 2, 3], 2)
            .expect("per-channel asymmetric");
        let imported = round_trip(&q).expect("per-channel must survive the descriptor");
        assert_eq!(imported, q);
        assert_eq!(imported.scale().len(), 3);
        assert_eq!(imported.axis(), Some(2));
        assert_eq!(imported.zero_point(), Some(&[1, 2, 3][..]));
    }

    /// A non-zero axis survives the `+1` encoding.
    #[test]
    fn the_axis_plus_one_encoding_is_reversible() {
        for axis in 0..4usize {
            let q =
                Quantization::per_channel_symmetric(vec![0.5, 0.25], axis).expect("per-channel");
            assert_eq!(QuantDesc::borrowing(&q).axis_plus_one, axis as u64 + 1);
            assert_eq!(round_trip(&q).and_then(|i| i.axis()), Some(axis));
        }
    }

    /// Per-channel scales with no axis are an error.
    #[test]
    fn per_channel_without_an_axis_is_an_error_not_a_silent_none() {
        let q = Quantization::per_channel_symmetric(vec![0.5, 0.25], 0).expect("per-channel");
        let mut desc = QuantDesc::borrowing(&q);
        desc.axis_plus_one = 0;
        // SAFETY: `q` still owns the arrays `desc` points at.
        let err = unsafe { desc.to_quantization() }
            .expect_err("per-channel scales naming no axis is malformed");
        assert!(
            err.to_string().contains("no channel axis"),
            "the error must say what is wrong with the descriptor, got: {err}"
        );
    }

    /// A non-zero count with a null scale array is an error.
    #[test]
    fn a_null_scale_array_with_a_non_zero_length_is_an_error() {
        let mut desc = QuantDesc::absent();
        desc.len = 3;
        // SAFETY: nothing is dereferenced; the null check precedes it.
        let err =
            unsafe { desc.to_quantization() }.expect_err("a length with a null array is malformed");
        assert!(err.to_string().contains("null scale array"), "got: {err}");
    }
}

#[cfg(test)]
mod desc_tests {
    use super::{c_byte_strides, from_parts, DescParts};
    use crate::{DType, PixelFormat, PixelLayout, TensorMemory};

    #[test]
    fn a_padded_batch_steps_n_over_one_padded_image() {
        // [N, H, W, C] = [2, 2, 8, 3] at a 32 B pitch: each tile is 2 rows.
        assert_eq!(
            c_byte_strides(&[2, 2, 8, 3], 1, Some(PixelLayout::Packed), Some(32)),
            [64, 32, 3, 1]
        );
        // [N, C, H, W] = [2, 3, 4, 8] at 16 B: 3 planes of 4 rows per image.
        assert_eq!(
            c_byte_strides(&[2, 3, 4, 8], 1, Some(PixelLayout::Planar), Some(16)),
            [192, 64, 16, 1]
        );
        // [N, H*k, W] = [2, 6, 8] NV12 at 32 B: 6 combined rows per image.
        assert_eq!(
            c_byte_strides(&[2, 6, 8], 1, Some(PixelLayout::SemiPlanar), Some(32)),
            [192, 32, 1]
        );
        // Unformatted rank 4 has no batch reading: the pitch lands on dim 0.
        assert_eq!(
            c_byte_strides(&[2, 2, 8, 3], 1, None, Some(64)),
            [64, 24, 3, 1]
        );
    }

    #[test]
    fn a_tight_tensor_gets_row_major_byte_strides() {
        assert_eq!(c_byte_strides(&[3, 8], 4, None, None), [32, 4]);
        assert_eq!(
            c_byte_strides(&[48, 64, 3], 1, Some(PixelLayout::Packed), None),
            [192, 3, 1]
        );
    }

    #[test]
    fn a_recorded_pitch_replaces_the_row_stride_of_a_packed_or_semi_planar_image() {
        assert_eq!(
            c_byte_strides(&[48, 64, 3], 1, Some(PixelLayout::Packed), Some(256)),
            [256, 3, 1]
        );
        assert_eq!(
            c_byte_strides(&[48, 64], 1, Some(PixelLayout::SemiPlanar), Some(128)),
            [128, 1]
        );
    }

    /// Planes of a strided planar image are `H` rows of the pitch apart.
    #[test]
    fn a_recorded_pitch_moves_both_the_row_and_the_plane_stride_of_a_planar_image() {
        assert_eq!(
            c_byte_strides(&[3, 48, 64], 1, Some(PixelLayout::Planar), Some(128)),
            [128 * 48, 128, 1]
        );
        // Bytes, not elements: an f32 planar image at a 320-byte pitch.
        assert_eq!(
            c_byte_strides(&[3, 48, 64], 4, Some(PixelLayout::Planar), Some(320)),
            [320 * 48, 320, 4]
        );
    }

    #[test]
    fn a_rank_one_tensor_has_no_row_to_pad() {
        assert_eq!(c_byte_strides(&[16], 1, None, Some(64)), [1]);
    }

    fn parts(
        memory: TensorMemory,
        fence_handle: usize,
        sync: Option<u64>,
    ) -> DescParts<'static, 'static> {
        DescParts {
            dims: &[48, 64, 3],
            memory,
            dtype: DType::U8,
            fourcc: 0,
            format: Some(PixelFormat::Rgb),
            row_stride: None,
            handle: -1,
            colorimetry: 0,
            capacity: 48 * 64 * 3,
            pin: None,
            pbo_vtable_ptr: None,
            fence_handle,
            sync,
            cpu_access: crate::CpuAccess::ReadWrite,
            view: false,
        }
    }

    /// The `VIEW` bit reports exactly the producer's `view`.
    #[test]
    fn the_view_bit_reports_the_producer() {
        let whole = from_parts(parts(TensorMemory::Mem, 0, None));
        assert_eq!(whole.flags & super::flags::VIEW, 0);
        let view = from_parts(DescParts {
            view: true,
            ..parts(TensorMemory::Mem, 0, None)
        });
        assert_eq!(view.flags & super::flags::VIEW, super::flags::VIEW);
    }

    /// A completion handed to a kind other than a D3D11 texture is not
    /// advertised as `SYNC_PRESENT`.
    #[test]
    fn a_completion_on_a_kind_with_no_fence_is_not_advertised() {
        let d = from_parts(parts(TensorMemory::Mem, 5, Some(9)));
        assert_eq!(d.flags & super::flags::SYNC_PRESENT, 0);
        assert_eq!(d.sync, 0);
        assert!(d.ptr.is_null());
    }

    /// A texture completion needs both the fence handle and a recorded value.
    #[cfg(target_os = "windows")]
    #[test]
    fn a_texture_completion_needs_a_fence_handle() {
        use super::flags;
        let d = from_parts(parts(TensorMemory::DmaBuf, 5, Some(9)));
        assert_eq!(d.flags & flags::SYNC_PRESENT, flags::SYNC_PRESENT);
        assert_eq!(d.sync, 9);
        assert_eq!(d.ptr.0 as usize, 5);

        let d = from_parts(parts(TensorMemory::DmaBuf, 0, Some(9)));
        assert_eq!(
            d.flags & flags::SYNC_PRESENT,
            0,
            "a value with no fence handle cannot be waited on"
        );
        assert_eq!(d.sync, 0);
        assert!(d.ptr.is_null());
    }
}
