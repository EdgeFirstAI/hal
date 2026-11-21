// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! IOSurface-backed tensor storage for macOS.
//!
//! `IoSurfaceTensor<T>` is the macOS counterpart to `DmaTensor<T>` on Linux: a
//! zero-copy GPU↔CPU buffer that the OpenGL backend (ANGLE on macOS) can import
//! directly via `EGL_ANGLE_iosurface_client_buffer`. Both fit into the
//! `TensorMemory::DmaBuf` slot.
//!
//! The bindings are raw FFI to the IOSurface and CoreFoundation frameworks.
//!
//! ## CPU access
//!
//! A map holds an `IOSurfaceLock` for its lifetime and exposes the base address
//! as a slice; dropping it calls the matching unlock.
//!
//! ## Planar surfaces
//!
//! A CoreVideo `420v` or `420f` surface has two planes at their own offsets
//! and row pitches. `TensorDyn::from_planar_iosurface` wraps it as a multiplane
//! NV12 tensor whose luma and chroma each wrap the surface at their plane's
//! offset and row stride, whether or not the planes are contiguous. A planar
//! surface records no single backing pitch.
//!
//! ## Cross-process sharing
//!
//! IOSurfaces are identified by `IOSurfaceID` (u32) within a host. The GL
//! backend uses this id as part of the buffer-identity cache key.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::{
    error::{Error, Result},
    packed_rgba16f_layout, BufferIdentity, DType, IdentityKind, PixelFormat, TensorMemory,
    TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{ffi::c_void, fmt, marker::PhantomData, ptr::NonNull, sync::Arc};

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

// IOSurface and CoreFoundation are separate frameworks; both are linked.
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
    fn IOSurfaceLookup(csid: u32) -> IOSurfaceRef;

    fn CFRetain(cf: *const c_void) -> *const c_void;
    fn CFRelease(cf: *const c_void);
    // Called only from `#[cfg(test)]` retain-count assertions.
    #[cfg_attr(not(test), expect(dead_code))]
    fn CFGetRetainCount(cf: *const c_void) -> isize;

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

/// Derive a [`BufferIdentity`] from `IOSurfaceGetID`, the system's identifier
/// for the surface, stable across every wrap of it within a host.
fn identity_from_surface(surface: IOSurfaceRef) -> BufferIdentity {
    let id = unsafe { IOSurfaceGetID(surface) };
    BufferIdentity::derived(IdentityKind::IoSurface, id as u64)
}

/// Owned IOSurface handle. Releases on Drop via `CFRelease`. Cloneable
/// via Arc — every clone shares the same underlying surface.
#[derive(Debug, Clone)]
pub(crate) struct OwnedIoSurface {
    inner: Arc<IoSurfaceHandle>,
}

/// Inner wrapper that calls `CFRelease` on drop. The surface is released once,
/// when the last clone of `OwnedIoSurface` drops.
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

/// Recover a live `IOSurfaceRef` from an IOSurfaceID: the identifier
/// `Tensor::iosurface_id` reports and that
/// [`crate::protocol::TensorDesc::handle`] carries for
/// [`crate::protocol::kind::IOSURFACE`] descriptors.
///
/// The result is a new (+1) reference the caller owns and gives back with
/// [`release`]. Returns `None` when no live surface has that id.
pub(crate) fn lookup_by_id(id: u32) -> Option<*mut c_void> {
    let ptr = unsafe { IOSurfaceLookup(id) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr)
    }
}

/// Release a `+1` `IOSurfaceRef` obtained from [`lookup_by_id`].
pub(crate) fn release(ptr: *mut c_void) {
    unsafe { CFRelease(ptr as *const c_void) };
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
    /// carry this as their row stride; raw byte surfaces (`new_with_byte_size`,
    /// a single padded row) leave the tensor stride natural. Zero for a planar
    /// surface, which has no single row pitch.
    pub(crate) bytes_per_row: usize,
    /// Whether this tensor was constructed from an externally-provided
    /// IOSurface via `from_surface`.
    #[allow(dead_code)]
    pub(crate) is_imported: bool,
    /// Byte offset of this tensor's window into the shared IOSurface. Non-zero
    /// only for sub-views (`view`/`batch`), which share the surface and
    /// identity. CPU maps add it to the locked base address.
    pub(crate) view_offset: usize,
}

unsafe impl<T> Send for IoSurfaceTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for IoSurfaceTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

/// Keepalive holding an IOSurface's CPU lock for a map's lifetime. Locks on
/// construction and unlocks on drop with the same options.
pub(crate) struct IoSurfaceMapLock {
    surface: OwnedIoSurface,
    options: u32,
}

impl Drop for IoSurfaceMapLock {
    fn drop(&mut self) {
        let mut seed: u32 = 0;
        let rc = unsafe { IOSurfaceUnlock(self.surface.as_ptr(), self.options, &mut seed) };
        if rc != 0 {
            log::error!("IOSurfaceUnlock failed (rc={rc})");
        }
    }
}

/// Lock options for a CPU access direction: read-only unless the access writes.
/// `IOSurfaceUnlock` must be passed the same options as `IOSurfaceLock`.
fn lock_options(access: crate::CpuAccess) -> u32 {
    if access.writes() {
        0
    } else {
        K_IOSURFACE_LOCK_READ_ONLY
    }
}

impl<T> IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Establish a persistent host address for this surface. Takes no lock; the
    /// surface is retained as the pin's keepalive.
    pub(crate) fn host_pin<'a>(&self) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let base = unsafe { IOSurfaceGetBaseAddress(self.surface.as_ptr()) };
        let base_ptr = NonNull::new(base).ok_or_else(|| {
            Error::IoError(std::io::Error::other(
                "IOSurfaceGetBaseAddress returned null; surface is not host-addressable",
            ))
        })?;
        // The surface from the view offset on; `Tensor::pin_host` narrows to
        // the logical extent.
        let offset = self.view_offset.min(self.buf_size);
        // SAFETY: `offset` is at most `buf_size`, the surface's allocation.
        let data = unsafe { (base_ptr.as_ptr() as *mut u8).add(offset) };
        Ok(crate::pin::HostPin::new(
            std::sync::Arc::new(self.surface.clone()),
            data,
            self.buf_size - offset,
        ))
    }
}

impl<T> IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Shared map constructor: lock via `scoped_pin` (released by the pin's
    /// keepalive on drop, with the same options at unlock as at lock) and wrap
    /// the result in the collapsed view.
    fn map_inner<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let exposed = match byte_size_override {
            Some(bytes) => bytes,
            None => self.shape.iter().product::<usize>() * std::mem::size_of::<T>(),
        };
        let end = self
            .view_offset
            .checked_add(exposed)
            .ok_or(Error::InvalidSize(exposed))?;
        if end > self.buf_size {
            return Err(Error::InsufficientCapacity {
                needed: end,
                capacity: self.buf_size,
            });
        }
        Ok(crate::view::HostView::new(
            self.scoped_pin(access)?,
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }

    /// Lock the surface and pin its address for the resulting view's lifetime.
    /// The lock is held until the pin's keepalive drops.
    pub(crate) fn scoped_pin<'a>(&self, access: crate::CpuAccess) -> Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let options = lock_options(access);
        let mut seed: u32 = 0;
        let rc = unsafe { IOSurfaceLock(self.surface.as_ptr(), options, &mut seed) };
        if rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "IOSurfaceLock failed (rc={rc})"
            ))));
        }
        let keepalive = std::sync::Arc::new(IoSurfaceMapLock {
            surface: self.surface.clone(),
            options,
        });
        let base = unsafe { IOSurfaceGetBaseAddress(self.surface.as_ptr()) };
        let base_ptr = NonNull::new(base).ok_or_else(|| {
            Error::IoError(std::io::Error::other(
                "IOSurfaceGetBaseAddress returned null after lock",
            ))
        })?;
        // SAFETY: the sole caller, `map_inner`, refuses `view_offset + exposed`
        // past `buf_size`, so this stays inside the surface's allocation.
        let data = unsafe { (base_ptr.as_ptr() as *mut u8).add(self.view_offset) };
        let len = self.buf_size.saturating_sub(self.view_offset);
        Ok(crate::pin::HostPin::new(keepalive, data, len))
    }

    /// `IOSurfaceLock` — the CPU is about to access the surface.
    pub(crate) fn sync_for_cpu(&self, access: crate::CpuAccess) -> crate::Result<()> {
        let mut seed: u32 = 0;
        let rc = unsafe { IOSurfaceLock(self.surface.as_ptr(), lock_options(access), &mut seed) };
        if rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "IOSurfaceLock failed (rc={rc})"
            ))));
        }
        Ok(())
    }

    /// `IOSurfaceUnlock` — the CPU is done. Options must match the lock.
    pub(crate) fn sync_for_device(&self, access: crate::CpuAccess) -> crate::Result<()> {
        let mut seed: u32 = 0;
        let rc = unsafe { IOSurfaceUnlock(self.surface.as_ptr(), lock_options(access), &mut seed) };
        if rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "IOSurfaceUnlock failed (rc={rc})"
            ))));
        }
        Ok(())
    }
}

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
        TensorMemory::DmaBuf
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
        self.view_offset = 0;
        Ok(())
    }

    fn map_with<'a>(&self, access: crate::CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let _span = tracing::trace_span!("tensor.map", memory = "iosurface", ?access).entered();
        self.map_inner(None, access)
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

    /// Zero-copy sub-region view sharing this surface and [`BufferIdentity`],
    /// positioned at `offset_bytes` from this tensor's own window with logical
    /// `shape`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the surface.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "IoSurfaceTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .view_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > self.buf_size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.buf_size,
            });
        }
        Ok(Self {
            name: self.name.clone(),
            surface: self.surface.clone(),
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: self.identity.clone(),
            buf_size: self.buf_size,
            bytes_per_row: self.bytes_per_row,
            is_imported: self.is_imported,
            view_offset: abs_offset,
        })
    }
}

impl<T> IoSurfaceTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate a new IOSurface large enough to hold `byte_size` bytes arranged
    /// as `shape`. The surface is created as a 1-row, 1-byte-per-element
    /// layout.
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
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
        let identity = identity_from_surface(surface.as_ptr());
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
            identity,
            buf_size: alloc,
            bytes_per_row,
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Allocate an image-formatted IOSurface (FourCC, per-pixel byte count and
    /// 2D dimensions) that the GL backend can bind via
    /// `EGL_ANGLE_iosurface_client_buffer`.
    ///
    /// For planar pixel formats (`PlanarRgb`, `PlanarRgba`) the channel planes
    /// stack vertically and the byte layout matches the tensor's `[channels, H,
    /// W]` shape.
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

        let (fourcc, bpe) = crate::iosurface_layout::image_iosurface_layout(format, dtype).ok_or_else(|| {
            Error::NotImplemented(format!(
                "IoSurfaceTensor::new_image: ({format:?}, {dtype:?}) has no IOSurface FourCC mapping"
            ))
        })?;

        // IOSurface geometry per (format, dtype):
        //
        //   * Packed formats (Rgba/Bgra/Yuyv, any supported dtype):
        //     surface dimensions = (width, height).
        //
        //   * PlanarRgb[a] u8: not supported (no FourCC).
        //
        //   * PlanarRgb[a] F16: an RGBA16F surface (FourCC 'RGhA'). Each
        //     RGBA16F pixel holds 4 contiguous f16 elements of the planar
        //     `[C, H, W]` byte stream, giving surface dimensions
        //     `(W/4, C*H)` via `packed_rgba16f_layout`. W must be a
        //     multiple of 4.
        let (surface_width, surface_height) = match (format, dtype) {
            (PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
                let layout =
                    packed_rgba16f_layout(format, dtype, width, height).ok_or_else(|| {
                        Error::InvalidShape(format!(
                            "{format:?} F16 IOSurface requires width%4==0 for RGBA16F packing \
                             (got width={width})"
                        ))
                    })?;
                (layout.surface_w, layout.surface_h)
            }
            // Packed RGB u8/i8 uses an RGBA8888 surface at (W*3/4, H).
            (PixelFormat::Rgb, DType::U8 | DType::I8) => {
                let layout = crate::packed_rgb888_layout(width, height).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Rgb u8 IOSurface requires width%4==0 for RGBA8888 packing \
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
            // Semi-planar YUV (NV12/NV16/NV24): the whole contiguous
            // combined-plane buffer as one R8 surface, sized to the 64-aligned
            // row pitch.
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
        let identity = identity_from_surface(surface.as_ptr());
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
            identity,
            buf_size: alloc,
            bytes_per_row,
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Wrap an existing IOSurface as a tensor.
    ///
    /// The pointer is retained for the tensor's lifetime; the external owner
    /// keeps its own reference and releases it independently. The type-erased
    /// entry point is [`crate::TensorDyn::from_iosurface`], and
    /// [`crate::Tensor::from_iosurface`] is the typed wrapper.
    ///
    /// # Errors
    ///
    /// `InvalidShape` if the shape footprint (`shape.iter().product::<usize>()
    /// * std::mem::size_of::<T>()`) exceeds the IOSurface's allocated byte size
    /// (`IOSurfaceGetAllocSize`).
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`.
    pub unsafe fn from_surface(
        surface_ref: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        unsafe {
            let surface = OwnedIoSurface::from_external(surface_ref)?;
            let identity = identity_from_surface(surface.as_ptr());
            let alloc = IOSurfaceGetAllocSize(surface.as_ptr());
            let planar = !crate::iosurface_planes::plane_geometry(surface.as_ptr()).is_empty();
            let bytes_per_row = if planar {
                0
            } else {
                IOSurfaceGetBytesPerRow(surface.as_ptr())
            };

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
                identity,
                buf_size: alloc,
                bytes_per_row,
                is_imported: true,
                view_offset: 0,
            })
        }
    }

    /// Row pitch in bytes as reported by `IOSurfaceGetBytesPerRow` (64-byte
    /// aligned). For image-formatted surfaces this is the row stride.
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Physical IOSurface dimensions in texels (`IOSurfaceGetWidth` /
    /// `IOSurfaceGetHeight`), independent of the tensor's current logical
    /// shape.
    pub(crate) fn physical_surface_dims(&self) -> (usize, usize) {
        let w = unsafe { IOSurfaceGetWidth(self.surface.as_ptr()) };
        let h = unsafe { IOSurfaceGetHeight(self.surface.as_ptr()) };
        (w, h)
    }

    /// Row pitch to honour as an image stride when a tensor is reconfigured to
    /// a smaller image (`Tensor::configure_image`): `bytes_per_row` for a 2D
    /// image-formatted surface (height > 1), `None` for a single-row byte
    /// surface (`new_with_byte_size`).
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        let (_w, h) = self.physical_surface_dims();
        (h > 1 && self.bytes_per_row != 0).then_some(self.bytes_per_row)
    }

    /// Lock and map exposing `byte_size` bytes via `as_slice()` for strided row
    /// iteration. The caller (`Tensor::map`) validates `byte_size <= buf_size`
    /// first.
    pub(crate) fn map_with_byte_size<'a>(
        &self,
        byte_size: usize,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_inner(Some(byte_size), access)
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
// Tensor<T> IOSurface accessors (macOS only).
// ---------------------------------------------------------------------------

impl<T> crate::Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Borrow the underlying `IOSurfaceRef` for this tensor (macOS only).
    ///
    /// Returns `Some(ptr)` when the tensor is backed by IOSurface (i.e.
    /// `TensorMemory::DmaBuf` on macOS), `None` otherwise. The pointer is
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
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        match &self.storage {
            crate::TensorStorage::Dma(io_tensor) => Some(io_tensor.physical_surface_dims()),
            _ => None,
        }
    }

    /// Wrap an externally-allocated IOSurface as a tensor (macOS only).
    ///
    /// The surface is retained for the tensor's lifetime; the external owner
    /// keeps its own reference and must release it independently.
    ///
    /// # Errors
    ///
    /// `InvalidShape` if `shape.iter().product::<usize>() *
    /// std::mem::size_of::<T>()` exceeds the IOSurface's allocated byte size
    /// (`IOSurfaceGetAllocSize`).
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`.
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
// CFDictionary builder for IOSurfaceCreate
// ---------------------------------------------------------------------------

/// `L008` (kCVPixelFormatType_OneComponent8) — single-channel 8-bit
/// layout used for raw-byte allocations from
/// `IoSurfaceTensor::new_with_byte_size`.
const FOURCC_L008: u32 = u32::from_be_bytes(*b"L008");

unsafe fn build_props(
    width: usize,
    height: usize,
    bytes_per_element: usize,
    fourcc: u32,
) -> Result<CFDictionaryRef> {
    unsafe {
        // Checked arithmetic: an overflowing pitch or allocation size is
        // reported as an error.
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
            let key_cf = CFStringCreateWithCString(
                std::ptr::null(),
                key_c.as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
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
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;

    #[test]
    fn alloc_map_write_read_roundtrip() {
        let t = IoSurfaceTensor::<u8>::new(&[256], None).expect("alloc");
        assert!(t.buf_size >= 256, "buf_size should accommodate shape");
        assert_eq!(t.memory(), TensorMemory::DmaBuf);

        {
            let mut m = t.map().expect("map");
            let slice = m.as_mut_slice();
            assert!(slice.len() >= 256);
            for (i, b) in slice.iter_mut().take(256).enumerate() {
                *b = (i & 0xff) as u8;
            }
        }
        {
            let m = t.map().expect("remap");
            let slice = m.as_slice();
            for (i, b) in slice.iter().take(256).enumerate() {
                assert_eq!(*b, (i & 0xff) as u8, "byte {i} mismatch");
            }
        }
    }

    #[test]
    fn wrap_clone_drop_moves_cf_retain_count_by_one() {
        let t = IoSurfaceTensor::<u8>::new(&[64], None).expect("alloc");
        let ptr = t.surface.as_ptr() as *const c_void;
        let base = unsafe { CFGetRetainCount(ptr) };
        assert!(base >= 1, "create must leave a live retain, got {base}");

        let clone = t.surface.clone();
        assert_eq!(
            unsafe { CFGetRetainCount(ptr) },
            base,
            "Arc clone of OwnedIoSurface must not CFRetain"
        );
        drop(clone);
        assert_eq!(unsafe { CFGetRetainCount(ptr) }, base);

        let extra = OwnedIoSurface::from_external(t.surface.as_ptr()).expect("retain");
        assert_eq!(
            unsafe { CFGetRetainCount(ptr) },
            base + 1,
            "from_external must CFRetain"
        );
        drop(extra);
        assert_eq!(
            unsafe { CFGetRetainCount(ptr) },
            base,
            "dropping from_external must CFRelease"
        );
    }

    #[test]
    fn image_surface_strided_map_honours_bytes_per_row() {
        use crate::Tensor;

        // RGBA u8 at width 17: a 68-byte natural row, padded by IOSurface to
        // 128 bytes.
        let h = 3usize;
        let w = 17usize;
        let t = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("non-aligned packed RGBA should allocate a padded IOSurface");

        let stride = t.effective_row_stride().expect("stride");
        assert!(stride >= w * 4, "stride {stride} >= natural {}", w * 4);
        assert_eq!(stride % 64, 0, "IOSurface pads bytes_per_row to 64");
        assert!(stride > w * 4, "width 17 RGBA must be padded (68 -> 128)");
        assert_eq!(t.width(), Some(w));
        assert_eq!(t.height(), Some(h));

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
    fn copy_to_flat_compacts_padded_rows() {
        use crate::{Tensor, TensorDyn};

        // RGBA u8 at width 17: a 68-byte natural row, padded by IOSurface to
        // 128 bytes.
        let (w, h) = (17usize, 3usize);
        let t = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("padded IOSurface alloc");
        let stride = t.effective_row_stride().expect("stride");
        assert!(stride > w * 4, "test requires a padded stride");
        {
            let mut m = t.map().expect("map");
            let buf = m.as_mut_slice();
            for row in 0..h {
                for col in 0..w * 4 {
                    buf[row * stride + col] = (row * 91 + col) as u8;
                }
            }
        }
        let mut flat = vec![0u8; h * w * 4];
        t.copy_to_flat(&mut flat).expect("copy_to_flat");
        for row in 0..h {
            for col in 0..w * 4 {
                assert_eq!(
                    flat[row * w * 4 + col],
                    (row * 91 + col) as u8,
                    "flat byte ({row},{col})"
                );
            }
        }

        let mut short = vec![0u8; h * w * 4 - 1];
        assert!(t.copy_to_flat(&mut short).is_err());

        let tight = TensorDyn::image(
            64,
            4,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            crate::CpuAccess::ReadWrite,
        )
        .expect("tight Mem alloc");
        {
            let tu8 = tight.as_u8().unwrap();
            let mut m = tu8.map().expect("map");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        }
        let mut flat2 = vec![0u8; 64 * 4 * 4];
        tight.copy_to_flat(&mut flat2).expect("tight copy_to_flat");
        for (i, b) in flat2.iter().enumerate() {
            assert_eq!(*b, (i % 251) as u8, "tight byte {i}");
        }
    }

    #[test]
    fn explicit_dma_unmapped_image_format_errors() {
        use crate::Tensor;

        for (name, result) in [
            (
                "PlanarRgb u8",
                Tensor::<u8>::image(
                    64,
                    8,
                    PixelFormat::PlanarRgb,
                    Some(TensorMemory::DmaBuf),
                    crate::CpuAccess::ReadWrite,
                )
                .map(drop),
            ),
            (
                "Rgb f32",
                Tensor::<f32>::image(
                    64,
                    8,
                    PixelFormat::Rgb,
                    Some(TensorMemory::DmaBuf),
                    crate::CpuAccess::ReadWrite,
                )
                .map(drop),
            ),
        ] {
            match result {
                Err(crate::Error::InvalidArgument(msg)) => {
                    assert!(
                        msg.contains("no zero-copy IOSurface mapping"),
                        "{name}: error must name the missing mapping: {msg}"
                    );
                }
                other => panic!("{name}: expected InvalidArgument, got {other:?}"),
            }
        }

        let nv12 = Tensor::<u8>::image(
            64,
            8,
            PixelFormat::Nv12,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("NV12 u8 keeps its designed L008 IOSurface mapping");
        assert_eq!(nv12.memory(), TensorMemory::DmaBuf);
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
        assert!(t.reshape(&[100]).is_err());
    }

    #[test]
    fn fourcc_layout_u8_packed_unchanged() {
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Rgba, DType::U8),
            Some((u32::from_be_bytes(*b"RGBA"), 4))
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Bgra, DType::U8),
            Some((u32::from_be_bytes(*b"BGRA"), 4))
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Yuyv, DType::U8),
            Some((u32::from_be_bytes(*b"2C08"), 2))
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Rgb, DType::U8),
            Some((u32::from_be_bytes(*b"RGBA"), 4))
        );
    }

    #[test]
    fn fourcc_layout_f16_packed_rgba16f() {
        let expected = (u32::from_be_bytes(*b"RGhA"), 8);
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Rgba, DType::F16),
            Some(expected)
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::PlanarRgb, DType::F16),
            Some(expected)
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::PlanarRgba, DType::F16),
            Some(expected)
        );
    }

    #[test]
    fn fourcc_layout_unsupported_combinations_return_none() {
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::PlanarRgb, DType::U8),
            None
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::PlanarRgba, DType::U8),
            None
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Rgba, DType::F32),
            None
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::PlanarRgb, DType::F32),
            None
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Nv12, DType::U8),
            Some((u32::from_be_bytes(*b"L008"), 1))
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Yuyv, DType::F16),
            None
        );
        assert_eq!(
            crate::iosurface_layout::image_iosurface_layout(PixelFormat::Grey, DType::F16),
            None
        );
    }

    #[test]
    fn new_image_planar_f16_packs_into_rgba16f() {
        // PlanarRgb F16 64×32: an RGBA16F surface of (64/4, 3*32) = (16, 96)
        // texels at 8 bytes per texel, 128 bytes per row, 12288 bytes in total.
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
        // Rgba F16 64×32: 64 pixels × 8 bytes = 512 bytes per row, × 32 rows =
        // 16384 bytes.
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
        let src = IoSurfaceTensor::<u8>::new(&[64], None).expect("alloc");
        let alloc = src.buf_size;
        let surface_ref = src.surface.as_ptr();

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

        let ok_shape = [alloc / std::mem::size_of::<u32>()];
        unsafe { IoSurfaceTensor::<u32>::from_surface(surface_ref, &ok_shape, None) }
            .expect("fitting shape should succeed");
    }

    /// A `subview` of an IOSurface shares the surface and identity and reads
    /// its own window through the CPU map.
    #[test]
    fn subview_iosurface_shares_identity_and_offsets_map() {
        use crate::{Tensor, TensorTrait};

        let parent = Tensor::<u8>::new(&[256], Some(TensorMemory::DmaBuf), None).expect("alloc");
        assert_eq!(parent.memory(), TensorMemory::DmaBuf);
        {
            let mut m = parent.map().expect("map parent");
            for (i, b) in m.as_mut_slice().iter_mut().take(256).enumerate() {
                *b = (i & 0xff) as u8;
            }
        }

        let view = parent
            .subview(64, &[64])
            .expect("iosurface subview must succeed");
        assert_eq!(
            view.buffer_identity().id(),
            parent.buffer_identity().id(),
            "subview shares the parent BufferIdentity (one GL import)"
        );

        let m = view.map().expect("map iosurface subview");
        let s = m.as_slice();
        assert_eq!(s.len(), 64, "window exposes exactly its logical length");
        for (i, b) in s.iter().enumerate() {
            assert_eq!(
                *b,
                ((i + 64) & 0xff) as u8,
                "view byte {i} must be parent[{}]",
                i + 64
            );
        }

        drop(m);
        {
            let mut mw = view.map().expect("remap for write");
            mw.as_mut_slice()[0] = 0xAB;
        }
        let mp = parent.map().expect("remap parent");
        assert_eq!(
            mp.as_slice()[64],
            0xAB,
            "write through view is visible in parent"
        );
    }

    /// `map_inner` refuses a plane offset that leaves too little of the surface
    /// for the image.
    #[test]
    fn iosurface_map_refuses_a_plane_offset_that_cannot_hold_the_image() {
        use crate::{DType, Tensor, TensorDyn, TensorTrait};

        const PARENT: usize = 256;
        const WINDOW: usize = 64;

        let parent = Tensor::<u8>::new(&[PARENT], Some(TensorMemory::DmaBuf), None).expect("alloc");
        assert_eq!(parent.memory(), TensorMemory::DmaBuf);
        let parent_dyn = TensorDyn::from(parent);
        let id = parent_dyn
            .iosurface_id()
            .expect("parent is IOSurface-backed");
        let capacity = parent_dyn
            .map_bytes(crate::CpuAccess::Read)
            .expect("map parent")
            .as_slice()
            .len();

        let mut rebuilt = TensorDyn::from_iosurface_id(id, &[WINDOW], DType::U8, None)
            .expect("reconstruct at the view's shape");

        rebuilt.set_plane_offset(capacity - WINDOW);
        assert_eq!(
            rebuilt
                .map_bytes(crate::CpuAccess::Read)
                .expect("the last window that fits")
                .as_slice()
                .len(),
            WINDOW,
            "a window ending exactly at the surface's end must still map"
        );

        rebuilt.set_plane_offset(capacity - WINDOW + 1);
        let err = rebuilt
            .map_bytes(crate::CpuAccess::Read)
            .expect_err("a window one byte short of its image");
        assert!(
            matches!(err, Error::InsufficientCapacity { .. }),
            "expected InsufficientCapacity, got {err}"
        );

        rebuilt.set_plane_offset(capacity);
        assert!(
            rebuilt.map_bytes(crate::CpuAccess::Read).is_err(),
            "an empty window for a non-empty tensor must not map"
        );

        rebuilt.set_plane_offset(usize::MAX);
        assert!(
            rebuilt.map_bytes(crate::CpuAccess::Read).is_err(),
            "usize::MAX must not wrap into range"
        );
    }

    /// Restoring a plane offset onto a reconstructed IOSurface tensor moves
    /// where `map()` starts.
    #[test]
    fn set_plane_offset_moves_the_iosurface_map_window() {
        use crate::{DType, Tensor, TensorDyn, TensorTrait};

        const PARENT: usize = 256;
        const OFFSET: usize = 64;
        const WINDOW: usize = 64;

        let parent = Tensor::<u8>::new(&[PARENT], Some(TensorMemory::DmaBuf), None).expect("alloc");
        assert_eq!(parent.memory(), TensorMemory::DmaBuf);
        {
            let mut m = parent.map().expect("map parent");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }
        let parent_dyn = TensorDyn::from(parent);
        let id = parent_dyn
            .iosurface_id()
            .expect("parent is IOSurface-backed");

        let mut rebuilt = TensorDyn::from_iosurface_id(id, &[WINDOW], DType::U8, None)
            .expect("reconstruct at the view's shape");
        {
            let m = rebuilt
                .map_bytes(crate::CpuAccess::Read)
                .expect("map rebuilt before the offset");
            assert_eq!(
                m.as_slice()[0],
                0,
                "precondition: a bare import starts at the parent's origin"
            );
        }

        rebuilt.set_plane_offset(OFFSET);
        assert_eq!(
            rebuilt.plane_offset(),
            Some(OFFSET),
            "the wrapper field records the offset on every backing"
        );

        let m = rebuilt
            .map_bytes(crate::CpuAccess::Read)
            .expect("map rebuilt after the offset");
        let s = m.as_slice();
        assert_eq!(s.len(), WINDOW, "the window exposes its logical length");
        assert_ne!(
            s[0], 0,
            "map() still starts at the parent's origin — set_plane_offset was \
             a no-op on IOSurface storage"
        );
        for (i, b) in s.iter().enumerate() {
            assert_eq!(
                *b,
                ((i + OFFSET) & 0xff) as u8,
                "window byte {i} must be parent[{}]",
                i + OFFSET
            );
        }
    }

    /// A nested `subview` does not compound its offset.
    #[test]
    fn nested_iosurface_subviews_do_not_compound_their_plane_offset() {
        use crate::{Tensor, TensorTrait};

        let parent = Tensor::<u8>::new(&[256], Some(TensorMemory::DmaBuf), None).expect("alloc");
        {
            let mut m = parent.map().expect("map parent");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }

        let outer = parent.subview(64, &[128]).expect("outer subview");
        assert_eq!(outer.plane_offset(), Some(64));
        let inner = outer.subview(32, &[64]).expect("inner subview");
        assert_eq!(
            inner.plane_offset(),
            Some(96),
            "offsets add once, not twice (64 + 32)"
        );

        let m = inner.map().expect("map nested subview");
        for (i, b) in m.as_slice().iter().enumerate() {
            assert_eq!(
                *b,
                ((i + 96) & 0xff) as u8,
                "nested window byte {i} must be parent[{}], not a compounded offset",
                i + 96
            );
        }
    }

    /// `set_format` clears the IOSurface storage offset as well as the
    /// wrapper field, so `map()` starts at the surface origin afterwards.
    #[test]
    fn set_format_clears_the_iosurface_map_window() {
        use crate::{PixelFormat, Tensor, TensorTrait};

        const PARENT: usize = 256;
        const OFFSET: usize = 64;

        let mut t = Tensor::<u8>::new(&[PARENT], Some(TensorMemory::DmaBuf), None).expect("alloc");
        {
            let mut m = t.map().expect("map parent");
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i & 0xff) as u8;
            }
        }

        t.set_logical_shape(&[8, 8, 1]).expect("logical shape");
        t.set_format(PixelFormat::Grey).expect("format");
        t.set_plane_offset(OFFSET);
        assert_eq!(
            t.map().expect("map before").as_slice()[0],
            OFFSET as u8,
            "precondition: the offset is live before the format change"
        );

        t.set_logical_shape(&[4, 8, 3]).expect("rgb logical shape");
        t.set_format(PixelFormat::Rgb).expect("new format");
        assert_eq!(t.plane_offset(), None, "the wrapper field is cleared");
        assert_eq!(
            t.map().expect("map after").as_slice()[0],
            0,
            "map() still starts at the old offset — set_format cleared the \
             wrapper field but not the IOSurface storage"
        );
    }

    /// A descriptor round trip restores the surface's pitch onto an IOSurface
    /// view import.
    #[test]
    fn descriptor_import_restores_the_iosurface_pitch_onto_a_view() {
        use crate::{Region, Tensor, TensorDyn, TensorTrait};

        // Width 50 RGBA: a 200-byte natural row, padded by IOSurface to a
        // 256-byte pitch.
        let parent = Tensor::<u8>::image(
            50,
            8,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            crate::CpuAccess::ReadWrite,
        )
        .expect("alloc a pitched RGBA surface");
        assert_eq!(parent.memory(), TensorMemory::DmaBuf);
        let pitch = parent.effective_row_stride().expect("surface pitch");
        assert_ne!(pitch, 50 * 4, "precondition: the pitch is padded");

        let view = parent.view(Region::new(4, 2, 16, 4)).expect("view");
        assert_eq!(
            view.effective_row_stride(),
            Some(pitch),
            "a multi-row view carries its parent's pitch"
        );
        let desc = TensorDyn::from(view).descriptor_pinned(None);
        assert_eq!(
            desc.strides()[0],
            pitch as i64,
            "the descriptor reports the pitch in bytes"
        );

        let rebuilt = TensorDyn::import_descriptor(&desc).expect("import the view's descriptor");
        assert_eq!(
            rebuilt.shape(),
            &[4, 16, 4],
            "imported at the window's shape"
        );
        assert_eq!(
            rebuilt.effective_row_stride(),
            Some(pitch),
            "restore_imported_row_stride dropped the IOSurface pitch: a 16-wide \
             window over a {pitch}-byte-pitched surface would be read at 64-byte \
             rows"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn plane_geometry_reads_a_corevideo_nv12_surface() {
        use crate::iosurface_planes::{plane_geometry, PlaneGeometry};

        let buf = crate::test_support::corevideo_nv12(1920, 1080, false);
        let planes = unsafe { plane_geometry(buf.surface()) };
        assert_eq!(
            planes,
            vec![
                PlaneGeometry {
                    offset: 512,
                    bytes_per_row: 1920,
                    width: 1920,
                    height: 1080,
                    bytes_per_element: 1,
                },
                PlaneGeometry {
                    offset: 2_074_112,
                    bytes_per_row: 1920,
                    width: 960,
                    height: 540,
                    bytes_per_element: 2,
                },
            ]
        );

        let buf = crate::test_support::corevideo_nv12(1000, 562, false);
        let planes = unsafe { plane_geometry(buf.surface()) };
        assert_eq!(planes.len(), 2);
        assert_eq!(
            planes[0],
            PlaneGeometry {
                offset: 512,
                bytes_per_row: 1024,
                width: 1000,
                height: 562,
                bytes_per_element: 1,
            }
        );
        assert_eq!(planes[1].offset, 582_144);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_planar_wrap_records_no_backing_pitch() {
        let buf = crate::test_support::corevideo_nv12(1920, 1080, false);
        let t = unsafe { IoSurfaceTensor::<u8>::from_surface(buf.surface(), &[1080, 1920], None) }
            .expect("wrap planar surface");
        assert_eq!(t.image_backing_row_stride(), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn host_pin_starts_at_the_plane_offset() {
        let buf = crate::test_support::corevideo_nv12(1920, 1080, false);
        buf.fill(0x10, 0x80);
        let mut t =
            unsafe { IoSurfaceTensor::<u8>::from_surface(buf.surface(), &[1080, 1920], None) }
                .expect("wrap planar surface");
        t.view_offset = 2_074_112;
        let pin = t.host_pin().expect("pin");
        assert_eq!(unsafe { *pin.as_ptr() }, 0x80);
        assert_eq!(pin.len(), t.buf_size - 2_074_112);
    }
}
