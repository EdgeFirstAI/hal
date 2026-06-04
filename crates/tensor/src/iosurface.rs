// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IOSurface-backed tensor storage for macOS.
//!
//! `IoSurfaceTensor<T>` is the macOS counterpart to `DmaTensor<T>` on
//! Linux: a zero-copy GPU↔CPU buffer that the OpenGL backend (ANGLE on
//! macOS) can import directly via `EGL_ANGLE_iosurface_client_buffer`.
//! Both fit into the `TensorMemory::Dma` slot — the variant name is
//! shared, the inner storage type differs per platform.
//!
//! ## Bindings approach
//!
//! Raw FFI to the IOSurface and CoreFoundation frameworks (linked via
//! `#[link]`). The `objc2-io-surface` crate's Obj-C wrappers want
//! `NSDictionary` for properties and don't accept CFDictionary cleanly
//! through their `initWithProperties` API; the C IOSurface API takes
//! `CFDictionaryRef` directly, which is what we have to hand and what
//! the spike at `spikes/angle_iosurface/` validates.
//!
//! ## CPU access
//!
//! `map()` returns `IoSurfaceMap<T>` which holds an `IOSurfaceLock` and
//! exposes the base address as a slice. `unmap()`/`Drop` calls the
//! matching unlock. IOSurface handles GPU↔CPU cache coherency
//! implicitly — no separate `DMA_BUF_IOCTL_SYNC` analog needed.
//!
//! ## Cross-process sharing
//!
//! IOSurfaces are identified by `IOSurfaceID` (u32) within a host. The
//! GL backend uses this id as part of the buffer-identity cache key so
//! repeated frames for the same IOSurface reuse the EGL pbuffer import.
//! Full Mach port passing is deferred.

#![cfg(target_os = "macos")]

use crate::{
    error::{Error, Result},
    packed_rgba16f_layout, BufferIdentity, DType, PixelFormat, TensorMap, TensorMapTrait,
    TensorMemory, TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

// ---------------------------------------------------------------------------
// Raw FFI to IOSurface + CoreFoundation
// ---------------------------------------------------------------------------

type IOSurfaceRef = *mut c_void;
type CFDictionaryRef = *mut c_void;
type CFStringRef = *mut c_void;
type CFNumberRef = *mut c_void;

const K_CF_NUMBER_LONG_TYPE: i32 = 10; // kCFNumberLongType
const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;

// IOSurface lock options.
const K_IOSURFACE_LOCK_READ_ONLY: u32 = 0x01;
#[allow(dead_code)]
const K_IOSURFACE_LOCK_AVOID_SYNC: u32 = 0x02;

// IOSurface depends on CoreFoundation for the CFDictionary properties
// dict and CFNumber values. Both frameworks must be linked explicitly:
// clippy warns about the duplicate `kind` attribute but it's required —
// each framework is its own dylib.
#[allow(clippy::duplicated_attributes)]
#[link(name = "IOSurface", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerRow(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetWidth(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetHeight(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;

    fn CFRetain(cf: *const c_void) -> *const c_void;
    fn CFRelease(cf: *const c_void);

    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFDictionarySetValue(dict: CFDictionaryRef, key: *const c_void, value: *const c_void);
    fn CFStringCreateWithCString(
        allocator: *const c_void,
        cstr: *const i8,
        encoding: u32,
    ) -> CFStringRef;
    fn CFNumberCreate(allocator: *const c_void, ty: i32, value_ptr: *const c_void) -> CFNumberRef;

    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

/// Owned IOSurface handle. Releases on Drop via `CFRelease`. Cloneable
/// via Arc — every clone shares the same underlying surface.
#[derive(Debug, Clone)]
pub(crate) struct OwnedIoSurface {
    inner: Arc<IoSurfaceHandle>,
}

/// Inner wrapper that handles the actual CFRelease in Drop. Wrapping in
/// Arc means multiple clones of `OwnedIoSurface` share the same retain
/// count and the surface is released exactly once when the last clone
/// drops.
#[derive(Debug)]
struct IoSurfaceHandle(IOSurfaceRef);

unsafe impl Send for IoSurfaceHandle {}
unsafe impl Sync for IoSurfaceHandle {}

impl Drop for IoSurfaceHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0 as *const c_void) };
        }
    }
}

impl OwnedIoSurface {
    /// Take ownership of an `IOSurfaceRef` returned by `IOSurfaceCreate`.
    /// The caller must not call `CFRelease` on the ref after this.
    fn from_created(ptr: IOSurfaceRef) -> Result<Self> {
        if ptr.is_null() {
            return Err(Error::IoError(std::io::Error::other(
                "IOSurfaceCreate returned null",
            )));
        }
        Ok(Self {
            inner: Arc::new(IoSurfaceHandle(ptr)),
        })
    }

    /// Wrap an externally-owned `IOSurfaceRef`. Calls `CFRetain` so the
    /// surface is kept alive while this `OwnedIoSurface` exists; the
    /// caller's reference is independent.
    pub(crate) fn from_external(ptr: IOSurfaceRef) -> Result<Self> {
        if ptr.is_null() {
            return Err(Error::InvalidArgument(
                "from_external: null IOSurfaceRef".into(),
            ));
        }
        unsafe { CFRetain(ptr as *const c_void) };
        Ok(Self {
            inner: Arc::new(IoSurfaceHandle(ptr)),
        })
    }

    pub(crate) fn as_ptr(&self) -> IOSurfaceRef {
        self.inner.0
    }
}

// ---------------------------------------------------------------------------
// IoSurfaceTensor — fits into TensorStorage::Dma on macOS.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub(crate) surface: OwnedIoSurface,
    pub shape: Vec<usize>,
    pub _marker: PhantomData<T>,
    identity: BufferIdentity,
    /// Total bytes allocated by the IOSurface (from `IOSurfaceGetAllocSize`).
    pub(crate) buf_size: usize,
    /// Row pitch in bytes (from `IOSurfaceGetBytesPerRow`). IOSurface rounds
    /// this up to 64-byte alignment, so for image-formatted surfaces it can
    /// exceed the natural `width × bytes_per_pixel`. Image-formatted tensors
    /// carry this as their row stride so CPU consumers iterate rows correctly;
    /// raw byte surfaces (`new_with_byte_size`, a single padded row) leave the
    /// tensor stride natural.
    pub(crate) bytes_per_row: usize,
    /// Whether this tensor was constructed from an externally-provided
    /// IOSurface via `from_surface`. Mirrors `DmaTensor::is_imported`
    /// and is reserved for diagnostic and C-API parity uses. Not yet
    /// consumed by any decision logic on macOS — IOSurface lifecycle is
    /// CFRetain/CFRelease symmetric regardless of import origin.
    #[allow(dead_code)]
    pub(crate) is_imported: bool,
}

unsafe impl<T> Send for IoSurfaceTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for IoSurfaceTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> TensorTrait<T> for IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let byte_size = shape
            .iter()
            .product::<usize>()
            .saturating_mul(std::mem::size_of::<T>());
        Self::new_with_byte_size(shape, byte_size, name)
    }

    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "IoSurfaceTensor::from_fd: IOSurface is not fd-backed; use from_surface()".into(),
        ))
    }

    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        Err(Error::NotImplemented(
            "IoSurfaceTensor::clone_fd: use surface_id() for cross-process sharing".into(),
        ))
    }

    fn memory(&self) -> TensorMemory {
        // Unified variant: macOS reports Dma, same as Linux. The variant
        // name is shared; the inner storage type differs per platform.
        TensorMemory::Dma
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        let new_elems: usize = shape.iter().product();
        let cur_elems: usize = self.shape.iter().product();
        if new_elems != cur_elems {
            return Err(Error::InvalidShape(format!(
                "reshape: element count mismatch ({cur_elems} → {new_elems})",
            )));
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        let _span = tracing::trace_span!("tensor.map", memory = "iosurface",).entered();
        let m = IoSurfaceMap::new(self.surface.clone(), self.shape.clone(), self.buf_size)?;
        Ok(TensorMap::IoSurface(m))
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    fn capacity_bytes(&self) -> usize {
        self.buf_size
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if needed > self.buf_size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.buf_size,
            });
        }
        self.shape = shape.to_vec();
        Ok(())
    }
}

impl<T> IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate a new IOSurface large enough to hold `byte_size` bytes
    /// arranged as `shape`. The surface is created as a 1-row,
    /// 1-byte-per-element layout — the GL backend separately allocates
    /// properly-shaped image IOSurfaces (with YUYV/NV12/BGRA FOURCC) via
    /// `crates/image/src/gl/iosurface_import.rs`.
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        // Span name follows the project convention `<crate>.<function>`
        // (see `ARCHITECTURE.md § Span naming conventions`). The
        // `memory = "iosurface"` field tags the variant so traces can
        // filter macOS-specific allocations.
        let _span =
            tracing::trace_span!("tensor.alloc", memory = "iosurface", byte_size,).entered();

        // SAFETY: dict is created and consumed within this block; the
        // CFDictionary stays alive across the IOSurfaceCreate call.
        let (dict, ptr) = unsafe {
            let dict = build_props(byte_size.max(1), 1, 1, FOURCC_L008)?;
            let ptr = IOSurfaceCreate(dict);
            (dict, ptr)
        };
        unsafe { CFRelease(dict as *const c_void) };
        let surface = OwnedIoSurface::from_created(ptr)?;
        let alloc = unsafe { IOSurfaceGetAllocSize(surface.as_ptr()) };
        let bytes_per_row = unsafe { IOSurfaceGetBytesPerRow(surface.as_ptr()) };

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("iosurface-{}", uuid::Uuid::new_v4()),
        };

        trace!("IoSurfaceTensor::new: name={name} bytes={alloc} shape={shape:?}",);

        Ok(Self {
            name,
            surface,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size: alloc,
            bytes_per_row,
            is_imported: false,
        })
    }

    /// Allocate an image-formatted IOSurface — proper FourCC + per-pixel
    /// byte count + 2D dimensions, suitable for binding directly via
    /// `EGL_ANGLE_iosurface_client_buffer` on the GL backend.
    ///
    /// Unlike `new_with_byte_size`, the returned IOSurface has the
    /// format ANGLE expects when the GL backend later wraps it in a
    /// pbuffer. Used by `Tensor::image()` on macOS when the caller
    /// requests `TensorMemory::Dma`.
    ///
    /// For planar pixel formats (`PlanarRgb`, `PlanarRgba`) the
    /// IOSurface is allocated as a single-channel float surface sized
    /// `(width, channels * height)` — the channel planes stack
    /// vertically and the byte layout matches the tensor's
    /// `[channels, H, W]` shape.
    pub(crate) fn new_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        let _span =
            tracing::trace_span!("tensor.alloc", memory = "iosurface", width, height, ?format,)
                .entered();

        let (fourcc, bpe) = image_fourcc_and_bpe(format, dtype).ok_or_else(|| {
            Error::NotImplemented(format!(
                "IoSurfaceTensor::new_image: ({format:?}, {dtype:?}) has no IOSurface FourCC mapping"
            ))
        })?;

        // IOSurface geometry per (format, dtype):
        //
        //   * Packed formats (Rgba/Bgra/Yuyv, any supported dtype):
        //     surface dimensions = (width, height).
        //
        //   * PlanarRgb[a] u8: not currently supported (no FourCC).
        //
        //   * PlanarRgb[a] F16: ANGLE only accepts RGBA16F (FourCC
        //     'RGhA') for float IOSurfaces. We pack 4 contiguous f16
        //     elements of the planar `[C, H, W]` byte stream into
        //     each RGBA16F pixel — surface dimensions become
        //     `(W/4, C*H)` via `packed_rgba16f_layout`. The byte
        //     layout is identical to a (nonexistent) R16F `(W, C*H)`
        //     surface and ORT consumes the locked base address as
        //     `&[f16]` with shape `[1, C, H, W]` without
        //     rearrangement. W must be a multiple of 4 for the
        //     packing to align — validated by `packed_rgba16f_layout`.
        let (surface_width, surface_height) = match (format, dtype) {
            (PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
                // Delegate W%4 check and (W/4, C*H) computation to the
                // canonical single source of truth. The only failure mode
                // here is misaligned width (packed_rgba16f_layout returns
                // None); overflow is not possible at these dimensions.
                let layout =
                    packed_rgba16f_layout(format, dtype, width, height).ok_or_else(|| {
                        Error::InvalidShape(format!(
                            "{format:?} F16 IOSurface requires width%4==0 for RGBA16F packing \
                             (got width={width})"
                        ))
                    })?;
                (layout.surface_w, layout.surface_h)
            }
            (PixelFormat::PlanarRgb, _) => {
                let sh = height.checked_mul(3).ok_or_else(|| {
                    Error::InvalidShape(format!("PlanarRgb height overflow (height={height})"))
                })?;
                (width, sh)
            }
            (PixelFormat::PlanarRgba, _) => {
                let sh = height.checked_mul(4).ok_or_else(|| {
                    Error::InvalidShape(format!("PlanarRgba height overflow (height={height})"))
                })?;
                (width, sh)
            }
            // Semi-planar YUV (NV12/NV16/NV24): bind the whole contiguous
            // combined-plane buffer as one R8 texture, sized to the 64-aligned
            // row pitch (see `PixelFormat::semi_planar_surface_dims` for the
            // ANGLE width==pitch rationale).
            (PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24, _) => format
                .semi_planar_surface_dims(width, height, bpe)
                .ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "{format:?} has no semi-planar surface dims for {width}x{height}"
                    ))
                })?,
            _ => (width, height),
        };

        let dict = unsafe { build_props(surface_width, surface_height, bpe, fourcc) }?;
        let ptr = unsafe { IOSurfaceCreate(dict) };
        unsafe { CFRelease(dict) };
        let surface = OwnedIoSurface::from_created(ptr)?;
        let alloc = unsafe { IOSurfaceGetAllocSize(surface.as_ptr()) };
        let bytes_per_row = unsafe { IOSurfaceGetBytesPerRow(surface.as_ptr()) };

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("iosurface-img-{}", uuid::Uuid::new_v4()),
        };

        trace!(
            "IoSurfaceTensor::new_image: name={name} surface={surface_width}x{surface_height} \
             logical={width}x{height} {format:?}/{dtype:?} fourcc=0x{fourcc:08x} bytes={alloc} \
             bytes_per_row={bytes_per_row}",
        );

        Ok(Self {
            name,
            surface,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size: alloc,
            bytes_per_row,
            is_imported: false,
        })
    }

    /// Wrap an existing IOSurface as a tensor. Used by the GL backend
    /// when importing an externally-allocated surface (e.g. from
    /// VideoToolbox or cross-process `IOSurfaceID` lookup).
    ///
    /// `surface_ref` must be a valid `IOSurfaceRef`. The pointer is
    /// retained for the tensor's lifetime; the external owner keeps its
    /// own reference and must release it independently.
    ///
    /// The type-erased public entry point is [`crate::TensorDyn::from_iosurface`]
    /// (and [`crate::Tensor::from_iosurface`] for the typed wrapper);
    /// most callers should prefer those over calling this inner
    /// constructor directly.
    ///
    /// # Safety
    ///
    /// The caller must ensure `surface_ref` is a valid live
    /// `IOSurfaceRef`. Passing a stale or invalid pointer is UB.
    ///
    /// The shape footprint
    /// (`shape.iter().product::<usize>() * std::mem::size_of::<T>()`) is
    /// validated against the IOSurface's allocated byte size
    /// (`IOSurfaceGetAllocSize`) and the constructor returns
    /// `Err(InvalidShape)` if it does not fit. This catches accidental
    /// mismatches that would otherwise cause out-of-bounds reads/writes
    /// in [`crate::Tensor::map`]; it does not relax the pointer-validity
    /// requirement above.
    pub unsafe fn from_surface(
        surface_ref: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        let surface = OwnedIoSurface::from_external(surface_ref)?;
        let alloc = IOSurfaceGetAllocSize(surface.as_ptr());
        let bytes_per_row = IOSurfaceGetBytesPerRow(surface.as_ptr());

        let elem_size = std::mem::size_of::<T>();
        let elems: usize = shape.iter().product();
        let requested = elems.checked_mul(elem_size).ok_or_else(|| {
            Error::InvalidShape(format!(
                "from_surface: shape footprint overflows usize (shape={shape:?}, sizeof T={elem_size})",
            ))
        })?;
        if requested > alloc {
            return Err(Error::InvalidShape(format!(
                "from_surface: shape requires {requested} bytes but IOSurface only \
                 has {alloc} (shape={shape:?}, sizeof T={elem_size})",
            )));
        }

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("iosurface-imported-{}", uuid::Uuid::new_v4()),
        };
        Ok(Self {
            name,
            surface,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size: alloc,
            bytes_per_row,
            is_imported: true,
        })
    }

    /// Row pitch in bytes as reported by `IOSurfaceGetBytesPerRow` (64-byte
    /// aligned). For image-formatted surfaces this is the authoritative row
    /// stride; CPU consumers must iterate rows by this value, not by
    /// `width × bytes_per_pixel`.
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Physical IOSurface dimensions in *texels* (`IOSurfaceGetWidth` /
    /// `IOSurfaceGetHeight`), independent of the tensor's current logical shape.
    ///
    /// A reused pool surface keeps these fixed for its whole life while
    /// `configure_image` changes the logical frame shape per decode. The GL
    /// backend binds the EGL pbuffer at these physical dims so one cached
    /// pbuffer serves every frame, and `texelFetch(col, row)` resolves to memory
    /// `row * bytes_per_row + col` (the row pitch is carried by the surface, not
    /// the declared texel width) — matching the codec's fixed-grid byte layout.
    pub(crate) fn physical_surface_dims(&self) -> (usize, usize) {
        let w = unsafe { IOSurfaceGetWidth(self.surface.as_ptr()) };
        let h = unsafe { IOSurfaceGetHeight(self.surface.as_ptr()) };
        (w, h)
    }

    /// Row pitch to honour as an *image* stride when a tensor is reconfigured
    /// to a smaller image (`Tensor::configure_image`), or `None` for a generic
    /// byte-bag surface.
    ///
    /// A byte-bag (`new_with_byte_size`) is a single `height == 1` row whose
    /// `bytes_per_row` equals the entire allocation — meaningless as a per-row
    /// image pitch. Adopting it would set a row stride spanning the whole buffer
    /// and blow up the strided map's size check. Only genuine 2D image-formatted
    /// surfaces (`new_image`, height > 1) carry a real row pitch to preserve.
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        let (_w, h) = self.physical_surface_dims();
        (h > 1).then_some(self.bytes_per_row)
    }

    /// Lock and map exposing `byte_size` bytes via `as_slice()` for strided
    /// row iteration. The caller (`Tensor::map`) validates
    /// `byte_size <= buf_size` first. The IOSurface lock yields the full
    /// surface base address, so the strided view is genuinely zero-copy — no
    /// staging buffer, just a wider slice over the same locked allocation.
    pub(crate) fn map_with_byte_size(&self, byte_size: usize) -> Result<TensorMap<T>> {
        let m = IoSurfaceMap::new_with_byte_size(
            self.surface.clone(),
            self.shape.clone(),
            self.buf_size,
            byte_size,
        )?;
        Ok(TensorMap::IoSurface(m))
    }

    /// Raw `IOSurfaceID` for cross-process sharing or GL backend import.
    pub fn surface_id(&self) -> u32 {
        unsafe { IOSurfaceGetID(self.surface.as_ptr()) }
    }

    /// Raw `IOSurfaceRef` for the GL backend to pass to
    /// `eglCreatePbufferFromClientBuffer(EGL_IOSURFACE_ANGLE, ...)`.
    pub fn surface_ref(&self) -> *mut c_void {
        self.surface.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// Tensor<T>::iosurface_ref accessor — macOS-only.
//
// The GL backend (image crate) calls this when importing an
// IOSurface-backed tensor as an EGL pbuffer. Returns None for tensors
// backed by SHM/Mem/Pbo since they have no associated IOSurface.
// ---------------------------------------------------------------------------

impl<T> crate::Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Borrow the underlying `IOSurfaceRef` for this tensor (macOS only).
    ///
    /// Returns `Some(ptr)` when the tensor is backed by IOSurface (i.e.
    /// `TensorMemory::Dma` on macOS), `None` otherwise. The pointer is
    /// borrowed — its lifetime is tied to the tensor.
    pub fn iosurface_ref(&self) -> Option<*mut c_void> {
        match &self.storage {
            crate::TensorStorage::Dma(io_tensor) => Some(io_tensor.surface_ref()),
            _ => None,
        }
    }

    /// Return the IOSurfaceID for cross-process IOSurface lookup
    /// (macOS only). Returns `None` when the tensor is not
    /// IOSurface-backed.
    ///
    /// The ID is stable for the lifetime of the IOSurface and can be
    /// passed across process boundaries; the receiver recovers the
    /// `IOSurfaceRef` via `IOSurfaceLookup(id)`.
    pub fn iosurface_id(&self) -> Option<u32> {
        match &self.storage {
            crate::TensorStorage::Dma(io_tensor) => Some(io_tensor.surface_id()),
            _ => None,
        }
    }

    /// Physical IOSurface dimensions in texels (`IOSurfaceGetWidth` /
    /// `IOSurfaceGetHeight`), independent of the logical shape. `None` when not
    /// IOSurface-backed.
    ///
    /// The GL backend binds the EGL pbuffer at these dims so one cached pbuffer
    /// serves every frame a reused pool surface holds; `texelFetch` then
    /// resolves to `row * bytesPerRow + col` against the surface's real pitch.
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        match &self.storage {
            crate::TensorStorage::Dma(io_tensor) => Some(io_tensor.physical_surface_dims()),
            _ => None,
        }
    }

    /// Wrap an externally-allocated IOSurface as a tensor (macOS only).
    ///
    /// Used to import IOSurfaces from VideoToolbox, AVFoundation, or
    /// other producers, and to recover a tensor from an IOSurfaceID
    /// received over a Mach port or XPC connection. The surface is
    /// retained for the tensor's lifetime; the external owner keeps
    /// its own reference and must release it independently.
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`. Passing a
    /// stale or invalid pointer is UB.
    ///
    /// HAL validates that
    /// `shape.iter().product::<usize>() * std::mem::size_of::<T>()` fits
    /// within the IOSurface's allocated byte size
    /// (`IOSurfaceGetAllocSize`) and returns `Err(InvalidShape)`
    /// otherwise. The pointer-validity requirement above is the
    /// caller's responsibility.
    pub unsafe fn from_iosurface(
        surface_ref: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self>
    where
        T: num_traits::Num,
    {
        let inner = unsafe { IoSurfaceTensor::<T>::from_surface(surface_ref, shape, name)? };
        Ok(crate::Tensor::wrap(crate::TensorStorage::Dma(inner)))
    }
}

// ---------------------------------------------------------------------------
// IoSurfaceMap — locked-for-CPU view.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    surface: OwnedIoSurface,
    shape: Vec<usize>,
    base_ptr: NonNull<c_void>,
    buf_size: usize,
    /// When `Some(bytes)`, `as_slice()` exposes `bytes / sizeof(T)` elements
    /// (the full row-padded surface) instead of `shape.product()`. Mirrors
    /// `DmaMap`/`MemMap`/`ShmMap`: used for strided IOSurface tensors so CPU
    /// callers iterate rows via `effective_row_stride()` (= `bytes_per_row`)
    /// without running past the locked region.
    byte_size_override: Option<usize>,
    _marker: PhantomData<T>,
    /// Lock options used at map time, replayed in unmap for symmetry.
    lock_options: u32,
    locked: bool,
}

unsafe impl<T> Send for IoSurfaceMap<T> where T: Num + Clone + fmt::Debug {}
unsafe impl<T> Sync for IoSurfaceMap<T> where T: Num + Clone + fmt::Debug {}

impl<T> IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn new(surface: OwnedIoSurface, shape: Vec<usize>, buf_size: usize) -> Result<Self> {
        Self::new_inner(surface, shape, buf_size, None)
    }

    /// Lock the surface and expose `byte_size` bytes via `as_slice()` rather
    /// than the shape-derived element count — for strided IOSurface tensors
    /// whose rows are `bytes_per_row`-padded. The caller (`Tensor::map`)
    /// validates `byte_size <= buf_size` first.
    fn new_with_byte_size(
        surface: OwnedIoSurface,
        shape: Vec<usize>,
        buf_size: usize,
        byte_size: usize,
    ) -> Result<Self> {
        Self::new_inner(surface, shape, buf_size, Some(byte_size))
    }

    fn new_inner(
        surface: OwnedIoSurface,
        shape: Vec<usize>,
        buf_size: usize,
        byte_size_override: Option<usize>,
    ) -> Result<Self> {
        // Default to read-write (options = 0). The read-only path
        // (K_IOSURFACE_LOCK_READ_ONLY) skips a CPU cache flush when the
        // caller only reads — a measurable savings if it becomes a hot
        // path. Left as a future enhancement once we have call-site
        // information about read-vs-write intent.
        let options: u32 = 0;
        let mut seed: u32 = 0;
        let lock_rc = unsafe { IOSurfaceLock(surface.as_ptr(), options, &mut seed) };
        if lock_rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "IOSurfaceLock failed (rc={lock_rc})"
            ))));
        }
        let base = unsafe { IOSurfaceGetBaseAddress(surface.as_ptr()) };
        let base_ptr = NonNull::new(base).ok_or_else(|| {
            Error::IoError(std::io::Error::other(
                "IOSurfaceGetBaseAddress returned null after lock",
            ))
        })?;
        Ok(Self {
            surface,
            shape,
            base_ptr,
            buf_size,
            byte_size_override,
            _marker: PhantomData,
            lock_options: options,
            locked: true,
        })
    }

    fn elem_count(&self) -> usize {
        match self.byte_size_override {
            Some(bytes) => bytes / std::mem::size_of::<T>(),
            None => self.shape.iter().product(),
        }
    }
}

impl<T> TensorMapTrait<T> for IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        self.elem_count()
    }

    fn unmap(&mut self) {
        if self.locked {
            let mut seed: u32 = 0;
            unsafe {
                IOSurfaceUnlock(self.surface.as_ptr(), self.lock_options, &mut seed);
            }
            self.locked = false;
        }
    }

    fn as_slice(&self) -> &[T] {
        self.deref()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<T> Deref for IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        let ptr = self.base_ptr.as_ptr() as *const T;
        let len = self.elem_count();
        debug_assert!(
            len * std::mem::size_of::<T>() <= self.buf_size,
            "IoSurfaceMap deref: {} elems × {} bytes > buf_size {}",
            len,
            std::mem::size_of::<T>(),
            self.buf_size,
        );
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl<T> DerefMut for IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        let ptr = self.base_ptr.as_ptr() as *mut T;
        let len = self.elem_count();
        // Symmetric with `Deref::deref` — without this an oversized
        // mutable write proceeds silently in release builds even
        // though the read path would have caught the same mismatch.
        debug_assert!(
            len * std::mem::size_of::<T>() <= self.buf_size,
            "IoSurfaceMap deref_mut: {} elems × {} bytes > buf_size {}",
            len,
            std::mem::size_of::<T>(),
            self.buf_size,
        );
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

impl<T> Drop for IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        self.unmap();
    }
}

// ---------------------------------------------------------------------------
// CFDictionary builder for IOSurfaceCreate
// ---------------------------------------------------------------------------

/// `L008` (kCVPixelFormatType_OneComponent8) — single-channel 8-bit
/// layout used for raw-byte allocations from
/// `IoSurfaceTensor::new_with_byte_size`.
const FOURCC_L008: u32 = u32::from_be_bytes(*b"L008");

/// IOSurface FourCC + bytes-per-element mapping for image-formatted
/// IOSurfaces, keyed on `(PixelFormat, DType)`. The GL backend's
/// `EGL_ANGLE_iosurface_client_buffer` import requires the IOSurface
/// pixel format to match the GL internal format / type combination —
/// ANGLE validates `IOSurfaceGetBytesPerElement` against the requested
/// `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` and rejects mismatches with
/// `EGL_BAD_ATTRIBUTE`. **This function is the single source of truth
/// for the `(PixelFormat, DType) → (FourCC, bpe)` mapping** — the image
/// crate's macOS GL backend reads it via [`image_iosurface_layout`]
/// when constructing the EGL pbuffer attribute list. Keep the two
/// layers in sync by not duplicating this table.
///
/// FourCC codes follow Apple's CoreVideo `kCVPixelFormatType_*`
/// constants because ANGLE's Metal backend recognizes those for
/// `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` mapping.
///
/// **ANGLE float-format constraint** (verified against
/// `EGL_ANGLE_iosurface_client_buffer.txt`): the extension's accepted
/// `(type, internal_format)` allowlist contains exactly **one** float
/// entry — `GL_HALF_FLOAT + GL_RGBA` (RGBA16F). There is no
/// `GL_FLOAT` entry, no single-channel float, no RGBA32F. R32F and
/// R16F single-channel bindings produce `EGL_BAD_ATTRIBUTE` at
/// `eglCreatePbufferFromClientBuffer` time even though the
/// extension-presence query (`GL_EXT_color_buffer_float` /
/// `_half_float`) reports them as available. Until the spec changes
/// our only viable float path is RGBA16F + 4-element pixel packing.
///
/// Combinations not listed are not supported by the GL backend on
/// macOS; callers fall back to SHM/Mem and a CPU code path.
fn image_fourcc_and_bpe(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        // YUYV is 4:2:2 packed (2 bytes/pixel); sampled as GL_RG via
        // FourCC '2C08' (kCVPixelFormatType_TwoComponent8).
        (PixelFormat::Yuyv, DType::U8) => Some((u32::from_be_bytes(*b"2C08"), 2)),
        // The FourCC matches the in-memory byte order: 'RGBA' for Rgba
        // tensors, 'BGRA' for Bgra. ANGLE supports both via
        // `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE = GL_RGBA` / `GL_BGRA_EXT`
        // and produces the matching shader output. Mapping both to
        // 'BGRA' would put the IOSurface bytes in BGRA order, which is
        // wrong for the Rgba contract.
        (PixelFormat::Rgba, DType::U8) => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        (PixelFormat::Bgra, DType::U8) => Some((u32::from_be_bytes(*b"BGRA"), 4)),
        // Single-channel 8-bit (`L008` = kCVPixelFormatType_OneComponent8),
        // sampled as `GL_RED`. Used for GREY images and as the raw byte plane
        // for the semi-planar YUV formats (NV12/NV16/NV24): the GPU binds the
        // whole contiguous `[total_h, W]` buffer as one R8 texture and the
        // YUV→RGB shader computes the luma/chroma texel positions itself
        // (portable across ANGLE/Metal, Mali/EGL, and embedded GLES).
        (
            PixelFormat::Grey | PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24,
            DType::U8,
        ) => Some((u32::from_be_bytes(*b"L008"), 1)),
        // ── F16 IOSurface for zero-copy preprocessing (CoreML / ANE) ──
        // The only ANGLE-supported float (type, internal_format) pair
        // is `(GL_HALF_FLOAT, GL_RGBA)` = RGBA16F, FourCC 'RGhA'
        // (kCVPixelFormatType_64RGBAHalf), 8 bytes per pixel.
        //
        // For Rgba destinations: 1 RGBA16F pixel = 1 image pixel of 4
        // half-floats.
        //
        // For PlanarRgb / PlanarRgba destinations: we pack 4 contiguous
        // half-floats of the planar `[C, H, W]` byte stream into each
        // RGBA16F pixel. The IOSurface is then sized `(W/4, C*H)` —
        // see `new_image` for the geometry. The byte layout is
        // identical to a (nonexistent) R16F `(W, C*H)` surface so ORT
        // can consume the locked base address as `&[f16]` with shape
        // `[1, C, H, W]` without any rearrangement.
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((u32::from_be_bytes(*b"RGhA"), 8))
        }
        _ => None,
    }
}

/// Public re-export of the `(PixelFormat, DType) → (FourCC,
/// bytes-per-element)` mapping for callers in other crates
/// (specifically the `edgefirst-image` macOS GL backend). The FourCC
/// is the cross-crate identifier — the GL backend maps it to the
/// matching `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` internally.
///
/// The image crate must use this function rather than duplicating the
/// table; a drift between the allocation and import sides produced a
/// silent R↔B swap during macOS bring-up (mapping Rgba to `'BGRA'`),
/// which is why the table now lives in one place.
///
/// Returns `None` when the (format, dtype) pair does not have a
/// defined IOSurface FourCC mapping in HAL (NV12, U8 planar, etc).
pub fn image_iosurface_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    image_fourcc_and_bpe(format, dtype)
}

unsafe fn build_props(
    width: usize,
    height: usize,
    bytes_per_element: usize,
    fourcc: u32,
) -> Result<CFDictionaryRef> {
    // Checked arithmetic (mirrors the image crate's `build_image_props`): an
    // overflowing pitch or allocation size would describe an under-sized
    // IOSurface that the GL import then treats as a valid buffer — a memory
    // hazard. Fail loudly instead of wrapping.
    let bytes_per_row = width
        .checked_mul(bytes_per_element)
        .and_then(|b| b.checked_add(63))
        .map(|b| b & !63)
        .ok_or_else(|| {
            Error::InvalidShape(format!(
                "IOSurface bytes-per-row overflow (width={width}, bpe={bytes_per_element})"
            ))
        })?;
    let alloc_size = bytes_per_row.checked_mul(height).ok_or_else(|| {
        Error::InvalidShape(format!(
            "IOSurface allocation size overflow (bytes_per_row={bytes_per_row}, height={height})"
        ))
    })?;

    let dict = CFDictionaryCreateMutable(
        std::ptr::null(),
        0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks,
    );
    if dict.is_null() {
        return Err(Error::IoError(std::io::Error::other(
            "CFDictionaryCreateMutable returned null",
        )));
    }

    let set_num = |key: &str, value: i64| -> Result<()> {
        let key_c = std::ffi::CString::new(key)
            .map_err(|e| Error::InvalidArgument(format!("CString: {e}")))?;
        let key_cf =
            CFStringCreateWithCString(std::ptr::null(), key_c.as_ptr(), K_CF_STRING_ENCODING_UTF8);
        if key_cf.is_null() {
            return Err(Error::IoError(std::io::Error::other(
                "CFStringCreateWithCString returned null",
            )));
        }
        let value_cf = CFNumberCreate(
            std::ptr::null(),
            K_CF_NUMBER_LONG_TYPE,
            &value as *const i64 as *const c_void,
        );
        if value_cf.is_null() {
            CFRelease(key_cf as *const c_void);
            return Err(Error::IoError(std::io::Error::other(
                "CFNumberCreate returned null",
            )));
        }
        CFDictionarySetValue(dict, key_cf as *const c_void, value_cf as *const c_void);
        CFRelease(key_cf as *const c_void);
        CFRelease(value_cf as *const c_void);
        Ok(())
    };

    let result = (|| -> Result<()> {
        set_num("IOSurfaceWidth", width as i64)?;
        set_num("IOSurfaceHeight", height as i64)?;
        set_num("IOSurfaceBytesPerElement", bytes_per_element as i64)?;
        set_num("IOSurfacePixelFormat", fourcc as i64)?;
        set_num("IOSurfaceBytesPerRow", bytes_per_row as i64)?;
        set_num("IOSurfaceAllocSize", alloc_size as i64)?;
        Ok(())
    })();

    if let Err(e) = result {
        CFRelease(dict as *const c_void);
        return Err(e);
    }
    let _ = K_IOSURFACE_LOCK_READ_ONLY; // silence unused on bring-up
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_map_write_read_roundtrip() {
        let t = IoSurfaceTensor::<u8>::new(&[256], None).expect("alloc");
        assert!(t.buf_size >= 256, "buf_size should accommodate shape");
        assert_eq!(t.memory(), TensorMemory::Dma);

        // Write via map
        {
            let mut m = t.map().expect("map");
            let slice = m.as_mut_slice();
            assert!(slice.len() >= 256);
            for (i, b) in slice.iter_mut().take(256).enumerate() {
                *b = (i & 0xff) as u8;
            }
        }
        // Read back via fresh map
        {
            let m = t.map().expect("remap");
            let slice = m.as_slice();
            for (i, b) in slice.iter().take(256).enumerate() {
                assert_eq!(*b, (i & 0xff) as u8, "byte {i} mismatch");
            }
        }
    }

    #[test]
    fn image_surface_strided_map_honours_bytes_per_row() {
        use crate::Tensor;

        // Packed RGBA U8 at width 17: natural row = 68 B, which IOSurface pads
        // up to a 64-aligned `bytes_per_row` (128 B). Previously `image()`
        // rejected this (forcing an SHM fallback); now it allocates a padded
        // IOSurface and records the stride, so CPU access is correct + zero-copy.
        let h = 3usize;
        let w = 17usize;
        let t = Tensor::<u8>::image(w, h, PixelFormat::Rgba, Some(TensorMemory::Dma))
            .expect("non-aligned packed RGBA should allocate a padded IOSurface");

        let stride = t.effective_row_stride().expect("stride");
        assert!(stride >= w * 4, "stride {stride} >= natural {}", w * 4);
        assert_eq!(stride % 64, 0, "IOSurface pads bytes_per_row to 64");
        assert!(stride > w * 4, "width 17 RGBA must be padded (68 -> 128)");
        assert_eq!(t.width(), Some(w));
        assert_eq!(t.height(), Some(h));

        // Write a distinct value per logical pixel, iterating rows by `stride`.
        {
            let mut m = t.map().expect("strided IOSurface map");
            let buf = m.as_mut_slice();
            assert_eq!(buf.len(), stride * h, "map exposes the full padded surface");
            for row in 0..h {
                for col in 0..w * 4 {
                    buf[row * stride + col] = (row * 37 + col) as u8;
                }
            }
        }
        // Read back via a fresh lock and verify the padded layout round-trips.
        {
            let m = t.map().expect("remap");
            let buf = m.as_slice();
            for row in 0..h {
                for col in 0..w * 4 {
                    assert_eq!(
                        buf[row * stride + col],
                        (row * 37 + col) as u8,
                        "pixel byte ({row},{col}) mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn surface_id_is_nonzero() {
        let t = IoSurfaceTensor::<u8>::new(&[64], None).expect("alloc");
        assert!(t.surface_id() != 0, "IOSurface IDs should be nonzero");
    }

    #[test]
    fn shape_reshape_roundtrip() {
        let mut t = IoSurfaceTensor::<u8>::new(&[16, 16], None).expect("alloc");
        assert_eq!(t.shape(), &[16, 16]);
        t.reshape(&[256]).expect("flatten");
        assert_eq!(t.shape(), &[256]);
        // Element count mismatch rejected
        assert!(t.reshape(&[100]).is_err());
    }

    #[test]
    fn fourcc_layout_u8_packed_unchanged() {
        // U8 packed formats keep the legacy mappings (regression
        // guard for the refactor that added dtype to the lookup).
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::Rgba, DType::U8),
            Some((u32::from_be_bytes(*b"RGBA"), 4))
        );
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::Bgra, DType::U8),
            Some((u32::from_be_bytes(*b"BGRA"), 4))
        );
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::Yuyv, DType::U8),
            Some((u32::from_be_bytes(*b"2C08"), 2))
        );
    }

    #[test]
    fn fourcc_layout_f16_packed_rgba16f() {
        // ANGLE's iosurface_client_buffer extension supports exactly
        // one float (type, internal_format) pair: (GL_HALF_FLOAT,
        // GL_RGBA) = RGBA16F, FourCC 'RGhA'. All HAL F16 IOSurface
        // destinations (packed Rgba or planar) use this mapping.
        let expected = (u32::from_be_bytes(*b"RGhA"), 8);
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::Rgba, DType::F16),
            Some(expected)
        );
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::PlanarRgb, DType::F16),
            Some(expected)
        );
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::PlanarRgba, DType::F16),
            Some(expected)
        );
    }

    #[test]
    fn fourcc_layout_unsupported_combinations_return_none() {
        // PlanarRgb / PlanarRgba u8 not supported on the IOSurface
        // path — those tensors stay on SHM/Mem storage.
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::PlanarRgb, DType::U8),
            None
        );
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::PlanarRgba, DType::U8),
            None
        );
        // F32 isn't accepted by ANGLE's IOSurface extension at all —
        // no GL_FLOAT entry on the allowlist. Any F32 combination
        // must return None so consumers fall back to staging.
        assert_eq!(image_fourcc_and_bpe(PixelFormat::Rgba, DType::F32), None);
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::PlanarRgb, DType::F32),
            None
        );
        // NV12/NV16/NV24/GREY (u8) now map to 'L008' (R8) so the GPU can sample
        // the contiguous semi-planar buffer as a single-channel texture.
        assert_eq!(
            image_fourcc_and_bpe(PixelFormat::Nv12, DType::U8),
            Some((u32::from_be_bytes(*b"L008"), 1))
        );
        // Float for YUYV / Grey isn't a meaningful combination.
        assert_eq!(image_fourcc_and_bpe(PixelFormat::Yuyv, DType::F16), None);
        assert_eq!(image_fourcc_and_bpe(PixelFormat::Grey, DType::F16), None);
    }

    #[test]
    fn new_image_planar_f16_packs_into_rgba16f() {
        // PlanarRgb F16: logical 64×32 image gets a packed RGBA16F
        // IOSurface sized (64/4, 3*32) = (16, 96). 8 B/pixel,
        // bytes_per_row rounded to 64-byte alignment (16*8 = 128,
        // already aligned) = 128, * 96 rows = 12288 bytes total. The
        // byte payload exactly matches an [3, 32, 64] f16 tensor
        // (3*32*64*2 = 12288 bytes).
        let shape = [3, 32, 64];
        let t = IoSurfaceTensor::<half::f16>::new_image(
            64,
            32,
            PixelFormat::PlanarRgb,
            DType::F16,
            &shape,
            None,
        )
        .expect("PlanarRgb F16 IOSurface allocation");
        assert!(t.buf_size >= 3 * 32 * 64 * 2);
    }

    #[test]
    fn new_image_planar_f16_rejects_misaligned_width() {
        // W=63 isn't divisible by 4 — packing would mis-align the
        // last column. HAL must reject at allocation time rather than
        // silently truncate.
        let shape = [3, 32, 63];
        let err = IoSurfaceTensor::<half::f16>::new_image(
            63,
            32,
            PixelFormat::PlanarRgb,
            DType::F16,
            &shape,
            None,
        )
        .expect_err("misaligned width must be rejected");
        match err {
            Error::InvalidShape(msg) => assert!(
                msg.contains("width%4==0") || msg.contains("RGBA16F"),
                "unexpected message: {msg}"
            ),
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    #[test]
    fn new_image_packed_rgba_f16_alloc_size_matches_dtype() {
        // Packed Rgba F16 at 64×32 → 64 pixels × 8 bytes/pixel = 512 B/row
        // (8-byte aligned, padded up to 64-byte stride), × 32 rows
        // ≥ 16384 bytes. (Equals 32*64*8 = 16384 bytes.)
        let shape = [32, 64, 4];
        let t = IoSurfaceTensor::<half::f16>::new_image(
            64,
            32,
            PixelFormat::Rgba,
            DType::F16,
            &shape,
            None,
        )
        .expect("Rgba F16 IOSurface allocation");
        assert!(t.buf_size >= 32 * 64 * 8);
    }

    #[test]
    fn from_surface_rejects_shape_overflowing_alloc() {
        // Allocate a small backing surface and try to import it under a
        // shape whose footprint is much larger than the allocation.
        let src = IoSurfaceTensor::<u8>::new(&[64], None).expect("alloc");
        let alloc = src.buf_size;
        let surface_ref = src.surface.as_ptr();

        // u32 element type: requested bytes = (alloc + 1) * 4 ≫ alloc.
        let bad_shape = [alloc + 1];
        let err = unsafe { IoSurfaceTensor::<u32>::from_surface(surface_ref, &bad_shape, None) }
            .expect_err("oversized shape must be rejected");
        match err {
            Error::InvalidShape(msg) => assert!(
                msg.contains("IOSurface only has"),
                "unexpected message: {msg}"
            ),
            other => panic!("expected InvalidShape, got {other:?}"),
        }

        // Sanity check: the same surface accepts a shape that does fit.
        let ok_shape = [alloc / std::mem::size_of::<u32>()];
        unsafe { IoSurfaceTensor::<u32>::from_surface(surface_ref, &ok_shape, None) }
            .expect("fitting shape should succeed");
    }
}
