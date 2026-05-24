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
    BufferIdentity, PixelFormat, TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
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
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;

    fn CFRetain(cf: *const c_void) -> *const c_void;
    fn CFRelease(cf: *const c_void);

    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFDictionarySetValue(
        dict: CFDictionaryRef,
        key: *const c_void,
        value: *const c_void,
    );
    fn CFStringCreateWithCString(
        allocator: *const c_void,
        cstr: *const i8,
        encoding: u32,
    ) -> CFStringRef;
    fn CFNumberCreate(
        allocator: *const c_void,
        ty: i32,
        value_ptr: *const c_void,
    ) -> CFNumberRef;

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

    fn from_fd(
        _fd: std::os::fd::OwnedFd,
        _shape: &[usize],
        _name: Option<&str>,
    ) -> Result<Self> {
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
        let _span = tracing::trace_span!(
            "tensor.map",
            memory = "iosurface",
        )
        .entered();
        let m = IoSurfaceMap::new(self.surface.clone(), self.shape.clone(), self.buf_size)?;
        Ok(TensorMap::IoSurface(m))
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
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
        let _span = tracing::info_span!(
            "tensor.iosurface.create",
            byte_size,
        )
        .entered();

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

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("iosurface-{}", uuid::Uuid::new_v4()),
        };

        trace!(
            "IoSurfaceTensor::new: name={name} bytes={alloc} shape={shape:?}",
        );

        Ok(Self {
            name,
            surface,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size: alloc,
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
    pub(crate) fn new_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        let _span = tracing::info_span!(
            "tensor.iosurface.create",
            width,
            height,
            ?format,
        )
        .entered();

        let (fourcc, bpe) = image_fourcc_and_bpe(format).ok_or_else(|| {
            Error::NotImplemented(format!(
                "IoSurfaceTensor::new_image: format {format:?} has no IOSurface FourCC mapping"
            ))
        })?;
        let dict = unsafe { build_props(width, height, bpe, fourcc) }?;
        let ptr = unsafe { IOSurfaceCreate(dict) };
        unsafe { CFRelease(dict) };
        let surface = OwnedIoSurface::from_created(ptr)?;
        let alloc = unsafe { IOSurfaceGetAllocSize(surface.as_ptr()) };

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("iosurface-img-{}", uuid::Uuid::new_v4()),
        };

        trace!(
            "IoSurfaceTensor::new_image: name={name} {width}x{height} {format:?} fourcc=0x{fourcc:08x} bytes={alloc}",
        );

        Ok(Self {
            name,
            surface,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size: alloc,
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
    /// # Safety
    ///
    /// The caller must ensure `surface_ref` is a valid live
    /// `IOSurfaceRef`. Passing a stale or invalid pointer is UB.
    pub unsafe fn from_surface(
        surface_ref: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        let surface = OwnedIoSurface::from_external(surface_ref)?;
        let alloc = IOSurfaceGetAllocSize(surface.as_ptr());
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
            is_imported: true,
        })
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
            crate::TensorStorage::Dma(io_tensor) => Some(io_tensor.surface.as_ptr()),
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
    /// stale or invalid pointer is UB. `shape` must match the
    /// IOSurface's pixel dimensions and element count.
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
            _marker: PhantomData,
            lock_options: options,
            locked: true,
        })
    }

    fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<T> TensorMapTrait<T> for IoSurfaceMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
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
/// IOSurfaces. The GL backend's
/// `EGL_ANGLE_iosurface_client_buffer` import requires the IOSurface
/// pixel format to match the GL internal format / type combination —
/// ANGLE validates `IOSurfaceGetBytesPerElement` against the requested
/// `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` and rejects mismatches with
/// `EGL_BAD_ATTRIBUTE`. The mapping here is the authoritative table
/// keyed by the HAL's `PixelFormat`.
///
/// Formats not listed are not supported by the GL backend on macOS;
/// callers fall back to SHM/Mem and a CPU code path.
fn image_fourcc_and_bpe(format: PixelFormat) -> Option<(u32, usize)> {
    match format {
        // YUYV is 4:2:2 packed (2 bytes/pixel); sampled as GL_RG via
        // FourCC '2C08' (kCVPixelFormatType_TwoComponent8).
        PixelFormat::Yuyv => Some((u32::from_be_bytes(*b"2C08"), 2)),
        // The FourCC matches the in-memory byte order: 'RGBA' for Rgba
        // tensors, 'BGRA' for Bgra. ANGLE supports both via
        // `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE = GL_RGBA` / `GL_BGRA_EXT`
        // and produces the matching shader output. Mapping both to
        // 'BGRA' would put the IOSurface bytes in BGRA order, which is
        // wrong for the Rgba contract.
        PixelFormat::Rgba => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        PixelFormat::Bgra => Some((u32::from_be_bytes(*b"BGRA"), 4)),
        _ => None,
    }
}

unsafe fn build_props(
    width: usize,
    height: usize,
    bytes_per_element: usize,
    fourcc: u32,
) -> Result<CFDictionaryRef> {
    let bytes_per_row = (width * bytes_per_element + 63) & !63;
    let alloc_size = bytes_per_row * height;

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
}
