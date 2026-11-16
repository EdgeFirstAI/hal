// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! AHardwareBuffer-backed tensor storage for Android.
//!
//! `AHardwareBufferTensor<T>` is the Android counterpart to `DmaTensor<T>` on
//! Linux and `IoSurfaceTensor<T>` on macOS/iOS: a zero-copy GPU↔CPU buffer that
//! the OpenGL backend imports directly via `EGL_ANDROID_image_native_buffer`
//! (`eglGetNativeClientBufferANDROID` → `eglCreateImageKHR`). All three fit
//! into the `TensorMemory::DmaBuf` slot.
//!
//! The bindings are raw FFI to `libnativewindow.so`.
//!
//! ## CPU access
//!
//! A map holds the buffer locked (`AHardwareBuffer_lock`) and exposes the base
//! address as a slice; dropping it performs the matching unlock. The lock
//! invalidates CPU caches and the unlock flushes them. The lock does not wait
//! for GPU completion.
//!
//! The shared handle serializes CPU locks: the first live map issues the FFI
//! lock, further maps of the same buffer (e.g. a parent tensor and a `view()`
//! sibling) reuse the same base address, and the last unmap issues the FFI
//! unlock.
//!
//! The tensor records its allocation-time CPU usage flags. At lock time the
//! map's requested [`CpuAccess`](crate::CpuAccess) selects the direction and
//! the recorded RARELY/OFTEN values are replayed. A buffer allocated
//! `CpuAccess::None` refuses `map()` with `CpuAccessDenied`, and a read-only
//! map or read-only import panics on `as_mut_slice()`.
//!
//! ## Geometry
//!
//! `AHardwareBuffer_Desc.stride` is an output parameter in pixels: gralloc
//! chooses the row pitch at allocation. The tensor caches the derived byte
//! pitch and allocation size at construction.
//!
//! Generic byte allocations use `AHARDWAREBUFFER_FORMAT_BLOB` (a 1-row byte
//! buffer). Image-formatted allocations use the real pixel formats so the GL
//! backend can bind them as textures; the `(PixelFormat, DType) → format` table
//! is [`image_ahardwarebuffer_layout`].

#![cfg(target_os = "android")]

use crate::{
    ahardwarebuffer_layout::{
        checked_shape_bytes, desc_layout, image_format_and_bpe, AHardwareBufferDesc, FORMAT_BLOB,
        USAGE_CPU_READ_MASK, USAGE_CPU_READ_OFTEN, USAGE_CPU_WRITE_MASK, USAGE_CPU_WRITE_OFTEN,
    },
    error::{Error, Result},
    packed_rgba16f_layout, BufferIdentity, DType, PixelFormat, TensorMemory, TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

// ---------------------------------------------------------------------------
// Raw FFI to libnativewindow (AHardwareBuffer, stable NDK ABI since API 26)
// ---------------------------------------------------------------------------

/// Opaque AHardwareBuffer handle (`struct AHardwareBuffer` in the NDK).
pub(crate) type AHardwareBufferRef = *mut c_void;

/// `ARect` for `AHardwareBuffer_lock` (null = lock the whole buffer).
#[repr(C)]
struct ARect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

// The `AHardwareBuffer_Desc` struct, format constants, and the layout math live
// in `ahardwarebuffer_layout.rs`.

// AHardwareBuffer_UsageFlags GPU bits (subset used by the HAL).
/// Sampleable as a GPU texture (`glEGLImageTargetTexture2DOES`).
const USAGE_GPU_SAMPLED_IMAGE: u64 = 0x100;
/// Renderable as a GPU color attachment (FBO render target):
/// `AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT`, of which `GPU_FRAMEBUFFER` is an
/// alias.
const USAGE_GPU_FRAMEBUFFER: u64 = 0x200;
/// Usable as a GPU data buffer; set on BLOB allocations.
const USAGE_GPU_DATA_BUFFER: u64 = 0x1000000;

/// CPU usage bits for a declared [`CpuAccess`](crate::CpuAccess).
fn cpu_usage_bits_for(access: crate::CpuAccess) -> u64 {
    let mut bits = 0;
    if access.reads() {
        bits |= USAGE_CPU_READ_OFTEN;
    }
    if access.writes() {
        bits |= USAGE_CPU_WRITE_OFTEN;
    }
    bits
}

#[link(name = "nativewindow")]
extern "C" {
    fn AHardwareBuffer_allocate(
        desc: *const AHardwareBufferDesc,
        out_buffer: *mut AHardwareBufferRef,
    ) -> i32;
    fn AHardwareBuffer_acquire(buffer: AHardwareBufferRef);
    fn AHardwareBuffer_release(buffer: AHardwareBufferRef);
    fn AHardwareBuffer_describe(buffer: AHardwareBufferRef, out_desc: *mut AHardwareBufferDesc);
    fn AHardwareBuffer_lock(
        buffer: AHardwareBufferRef,
        usage: u64,
        fence: i32,
        rect: *const ARect,
        out_virtual_address: *mut *mut c_void,
    ) -> i32;
    fn AHardwareBuffer_unlock(buffer: AHardwareBufferRef, fence: *mut i32) -> i32;
    /// Bionic system-property read (`<sys/system_properties.h>`); writes
    /// at most `PROP_VALUE_MAX` (92) bytes into `value` and returns the
    /// string length.
    fn __system_property_get(
        name: *const std::os::raw::c_char,
        value: *mut std::os::raw::c_char,
    ) -> i32;
}

/// `AHardwareBuffer_getId` (API 31+): the system's stable 64-bit id for a
/// buffer, identical across every wrap/import of the same allocation in the
/// process. Resolved lazily via `dlsym`; `None` on API 26–30, where the symbol
/// is absent.
fn ahardwarebuffer_get_id(buffer: AHardwareBufferRef) -> Option<u64> {
    type GetIdFn = unsafe extern "C" fn(AHardwareBufferRef, *mut u64) -> i32;
    static GET_ID: std::sync::OnceLock<Option<GetIdFn>> = std::sync::OnceLock::new();
    let get_id = (*GET_ID.get_or_init(|| {
        extern "C" {
            fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
        }
        // Bionic RTLD_DEFAULT: null on LP64, 0xffffffff on LP32.
        #[cfg(target_pointer_width = "64")]
        let rtld_default = std::ptr::null_mut();
        #[cfg(target_pointer_width = "32")]
        let rtld_default = 0xffff_ffffusize as *mut c_void;
        // SAFETY: dlsym with a NUL-terminated name; a null result means
        // the symbol is absent (API < 31).
        let sym = unsafe { dlsym(rtld_default, c"AHardwareBuffer_getId".as_ptr()) };
        if sym.is_null() {
            log::debug!("AHardwareBuffer_getId unavailable (API < 31): identities stay fresh");
            None
        } else {
            // SAFETY: the NDK declares this exact signature for the symbol.
            Some(unsafe { std::mem::transmute::<*mut c_void, GetIdFn>(sym) })
        }
    }))?;
    let mut id = 0u64;
    // SAFETY: buffer is a live acquired AHardwareBuffer; out param is valid.
    (unsafe { get_id(buffer, &mut id) } == 0).then_some(id)
}

/// Derive the [`BufferIdentity`] for `buffer`: from `AHardwareBuffer_getId` on
/// API 31+, and from the buffer pointer before API 31.
fn buffer_identity(buffer: &OwnedAHardwareBuffer) -> BufferIdentity {
    crate::ahardwarebuffer_layout::identity_from_parts(
        ahardwarebuffer_get_id(buffer.as_ptr()),
        buffer.as_ptr() as usize as u64,
    )
}

/// The device's native tile-compression scheme, classified from the
/// `ro.hardware.egl` system property (`scheme_for_egl_vendor`) and cached.
/// `None` on emulators and unrecognized vendors.
pub(crate) fn device_compression_scheme() -> Option<crate::CompressionScheme> {
    static SCHEME: std::sync::OnceLock<Option<crate::CompressionScheme>> =
        std::sync::OnceLock::new();
    *SCHEME.get_or_init(|| {
        const PROP_VALUE_MAX: usize = 92;
        let mut buf = [0u8; PROP_VALUE_MAX];
        // SAFETY: bionic writes at most PROP_VALUE_MAX bytes (including
        // the NUL terminator) and returns the value length.
        let len = unsafe {
            __system_property_get(
                c"ro.hardware.egl".as_ptr(),
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
            )
        };
        let vendor = usize::try_from(len)
            .ok()
            .filter(|&n| n <= PROP_VALUE_MAX)
            .and_then(|n| std::str::from_utf8(&buf[..n]).ok())
            .unwrap_or("");
        let scheme = crate::ahardwarebuffer_layout::scheme_for_egl_vendor(vendor);
        log::debug!("ro.hardware.egl={vendor:?} → compression scheme {scheme:?}");
        scheme
    })
}

/// Owned AHardwareBuffer handle. Releases on Drop via
/// `AHardwareBuffer_release`. Cloneable via Arc: every clone shares the same
/// underlying buffer.
#[derive(Debug, Clone)]
pub(crate) struct OwnedAHardwareBuffer {
    inner: Arc<AhbHandle>,
}

/// Keepalive holding an AHardwareBuffer's CPU lock for a [`HostPin`]'s life.
/// The lock is released when the last clone of the pin drops.
#[allow(dead_code)]
pub(crate) struct AhbMapLock {
    buffer: OwnedAHardwareBuffer,
}

impl Drop for AhbMapLock {
    fn drop(&mut self) {
        self.buffer.unlock_shared();
    }
}

/// Inner wrapper that releases the buffer on drop, once, when the last clone
/// drops.
///
/// Also owns the CPU-lock state: the first live map issues the FFI lock,
/// further concurrent maps of the same buffer reuse the recorded base address,
/// and the last unmap issues the FFI unlock. A nested map reuses the live lock
/// only when its usage direction covers the request.
#[derive(Debug)]
struct AhbHandle {
    ptr: AHardwareBufferRef,
    lock: Mutex<LockState>,
}

#[derive(Debug, Default)]
struct LockState {
    /// Live `AHardwareBufferMap`s on this buffer.
    count: usize,
    /// Base address returned by the (single) FFI lock while `count > 0`.
    base: *mut c_void,
    /// CPU usage bits of the live FFI lock. A nested map may reuse the lock
    /// only when its requested direction is covered by these bits.
    usage: u64,
}

// SAFETY: AHardwareBuffer is a thread-safe, refcounted kernel object; the
// pointer may be shared and released from any thread. The lock state is
// Mutex-guarded.
unsafe impl Send for AhbHandle {}
unsafe impl Sync for AhbHandle {}

impl AhbHandle {
    fn new(ptr: AHardwareBufferRef) -> Self {
        Self {
            ptr,
            lock: Mutex::new(LockState::default()),
        }
    }
}

impl Drop for AhbHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { AHardwareBuffer_release(self.ptr) };
        }
    }
}

impl OwnedAHardwareBuffer {
    /// Allocate a buffer for `desc`, taking ownership of the initial
    /// reference. Returns the descriptor as filled in by the allocator
    /// (`stride` populated by gralloc).
    fn allocate(mut desc: AHardwareBufferDesc) -> Result<(Self, AHardwareBufferDesc)> {
        let mut ptr: AHardwareBufferRef = std::ptr::null_mut();
        let rc = unsafe { AHardwareBuffer_allocate(&desc, &mut ptr) };
        if rc != 0 || ptr.is_null() {
            return Err(Error::IoError(std::io::Error::other(format!(
                "AHardwareBuffer_allocate failed (rc={rc}, format=0x{:x}, {}x{})",
                desc.format, desc.width, desc.height
            ))));
        }
        // Re-describe to pick up the allocator-chosen stride.
        unsafe { AHardwareBuffer_describe(ptr, &mut desc) };
        Ok((
            Self {
                inner: Arc::new(AhbHandle::new(ptr)),
            },
            desc,
        ))
    }

    /// Wrap an externally-owned buffer (e.g. from CameraX/ImageReader via
    /// JNI). Calls `AHardwareBuffer_acquire` so the buffer stays alive for
    /// this handle's lifetime; the caller's reference is independent.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid live AHardwareBuffer pointer.
    unsafe fn from_external(ptr: AHardwareBufferRef) -> Result<(Self, AHardwareBufferDesc)> {
        unsafe {
            if ptr.is_null() {
                return Err(Error::InvalidArgument(
                    "from_external: null AHardwareBuffer".into(),
                ));
            }
            AHardwareBuffer_acquire(ptr);
            let mut desc = AHardwareBufferDesc {
                width: 0,
                height: 0,
                layers: 0,
                format: 0,
                usage: 0,
                stride: 0,
                rfu0: 0,
                rfu1: 0,
            };
            AHardwareBuffer_describe(ptr, &mut desc);
            Ok((
                Self {
                    inner: Arc::new(AhbHandle::new(ptr)),
                },
                desc,
            ))
        }
    }

    pub(crate) fn as_ptr(&self) -> AHardwareBufferRef {
        self.inner.ptr
    }

    /// Acquire the refcounted CPU lock (see [`AhbHandle`]). Returns the
    /// buffer's base address. Every successful call must be balanced by exactly
    /// one [`Self::unlock_shared`].
    fn lock_shared(&self, cpu_usage: u64) -> Result<*mut c_void> {
        let mut st = self.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        if st.count > 0 {
            // A nested map reuses the live lock only when its direction covers
            // the request.
            if cpu_usage & st.usage != cpu_usage {
                return Err(Error::InvalidOperation(format!(
                    "concurrent AHardwareBuffer map with wider access \
                     (live lock usage {:#x}, requested {:#x}) — take the wider \
                     map first or drop the narrower one",
                    st.usage, cpu_usage
                )));
            }
            st.count += 1;
            return Ok(st.base);
        }
        let mut base: *mut c_void = std::ptr::null_mut();
        // fence = -1: no explicit acquire fence.
        let rc = unsafe {
            AHardwareBuffer_lock(self.inner.ptr, cpu_usage, -1, std::ptr::null(), &mut base)
        };
        if rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "AHardwareBuffer_lock failed (rc={rc})"
            ))));
        }
        if base.is_null() {
            // Balance the successful lock before erroring.
            unsafe { AHardwareBuffer_unlock(self.inner.ptr, std::ptr::null_mut()) };
            return Err(Error::IoError(std::io::Error::other(
                "AHardwareBuffer_lock returned a null address",
            )));
        }
        st.base = base;
        st.count = 1;
        st.usage = cpu_usage;
        Ok(base)
    }

    /// Release one refcount taken by [`Self::lock_shared`]; the last
    /// release performs the FFI unlock (null fence pointer = synchronous:
    /// flushes CPU caches before returning).
    fn unlock_shared(&self) {
        let mut st = self.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        match st.count {
            0 => log::error!("unbalanced AHardwareBuffer unlock (count already 0)"),
            1 => {
                let rc = unsafe { AHardwareBuffer_unlock(self.inner.ptr, std::ptr::null_mut()) };
                if rc != 0 {
                    log::error!("AHardwareBuffer_unlock failed (rc={rc})");
                }
                st.count = 0;
                st.base = std::ptr::null_mut();
            }
            _ => st.count -= 1,
        }
    }
}

#[derive(Debug)]
pub struct AHardwareBufferTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub(crate) buffer: OwnedAHardwareBuffer,
    pub shape: Vec<usize>,
    pub _marker: PhantomData<T>,
    identity: BufferIdentity,
    /// CPU-accessible allocation size in bytes: `bytes_per_row × height` (BLOB:
    /// the requested length). Cached at construction from the allocator-filled
    /// descriptor.
    pub(crate) buf_size: usize,
    /// Row pitch in bytes (`desc.stride` pixels × bytes-per-element). For BLOB
    /// buffers this equals the whole allocation.
    pub(crate) bytes_per_row: usize,
    /// Physical descriptor dimensions in texels, fixed for the buffer's life.
    physical_dims: (usize, usize),
    /// CPU usage flags (`CPU_READ_*`/`CPU_WRITE_*` values) the buffer was
    /// allocated/imported with; replayed by `AHardwareBuffer_lock`.
    cpu_usage: u64,
    /// True when wrapping an externally-allocated buffer (camera frame).
    pub(crate) is_imported: bool,
    /// Byte offset of a sub-region `view()` into the buffer.
    pub(crate) view_offset: usize,
}

impl<T> AHardwareBufferTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Acquire a host address valid for as long as the returned pin lives.
    #[allow(dead_code)]
    pub(crate) fn scoped_pin<'a>(&self, access: crate::CpuAccess) -> Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let base = self.buffer.lock_shared(cpu_usage_bits_for(access))?;
        let data = unsafe { (base as *mut u8).add(self.view_offset) };
        let len = self.buf_size.saturating_sub(self.view_offset);
        Ok(crate::pin::HostPin::new(
            std::sync::Arc::new(AhbMapLock {
                buffer: self.buffer.clone(),
            }),
            data,
            len,
        ))
    }
}

impl<T> TensorTrait<T> for AHardwareBufferTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let byte_size = checked_shape_bytes::<T>(shape)?;
        Self::new_with_byte_size(shape, byte_size, name)
    }

    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "AHardwareBufferTensor::from_fd: AHardwareBuffer is not fd-backed; \
             use from_hardware_buffer()"
                .into(),
        ))
    }

    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        Err(Error::NotImplemented(
            "AHardwareBufferTensor::clone_fd: share the AHardwareBuffer handle \
             (hardware_buffer_ptr) via binder instead"
                .into(),
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
        // Overflow-checked byte footprints.
        let new_bytes = checked_shape_bytes::<T>(shape)?;
        let cur_bytes = checked_shape_bytes::<T>(&self.shape)?;
        if new_bytes != cur_bytes {
            return Err(Error::InvalidShape(format!(
                "reshape: byte footprint mismatch ({cur_bytes} → {new_bytes})",
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
        let _span =
            tracing::trace_span!("tensor.map", memory = "ahardwarebuffer", ?access).entered();
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
        let needed = checked_shape_bytes::<T>(shape)?;
        // A viewed tensor's capacity is what remains past its offset.
        let capacity = self.buf_size.saturating_sub(self.view_offset);
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    /// Zero-copy sub-region view sharing this buffer and [`BufferIdentity`],
    /// positioned at `offset_bytes` from this tensor's own window with logical
    /// `shape`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the buffer.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "AHardwareBufferTensor::view: offset {offset_bytes} not aligned to \
                 align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .view_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = checked_shape_bytes::<T>(shape)?;
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
            buffer: self.buffer.clone(),
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: self.identity.clone(),
            buf_size: self.buf_size,
            bytes_per_row: self.bytes_per_row,
            physical_dims: self.physical_dims,
            cpu_usage: self.cpu_usage,
            is_imported: self.is_imported,
            view_offset: abs_offset,
        })
    }
}

impl<T> AHardwareBufferTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate a BLOB buffer large enough to hold `byte_size` bytes arranged
    /// as `shape`. BLOB buffers carry `GPU_DATA_BUFFER` usage and cannot bind
    /// as GL textures.
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let _span =
            tracing::trace_span!("tensor.alloc", memory = "ahardwarebuffer", byte_size,).entered();

        let len: u32 = byte_size
            .max(1)
            .try_into()
            .map_err(|_| Error::InvalidSize(byte_size))?;
        let desc = AHardwareBufferDesc {
            width: len,
            height: 1,
            layers: 1,
            format: FORMAT_BLOB,
            usage: USAGE_CPU_READ_OFTEN | USAGE_CPU_WRITE_OFTEN | USAGE_GPU_DATA_BUFFER,
            stride: 0,
            rfu0: 0,
            rfu1: 0,
        };
        let (buffer, desc) = OwnedAHardwareBuffer::allocate(desc)?;
        let (bytes_per_row, buf_size) = desc_layout(&desc).expect("BLOB layout is always known");

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("ahardwarebuffer-{}", uuid::Uuid::new_v4()),
        };

        trace!("AHardwareBufferTensor::new: name={name} bytes={buf_size} shape={shape:?}",);

        Ok(Self {
            name,
            identity: buffer_identity(&buffer),
            buffer,
            shape: shape.to_vec(),
            _marker: PhantomData,
            buf_size,
            bytes_per_row,
            physical_dims: (desc.width as usize, desc.height as usize),
            cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Allocate an image-formatted AHardwareBuffer (pixel format and 2D
    /// dimensions) for EGLImage import on the GL backend
    /// (`eglGetNativeClientBufferANDROID` → `eglCreateImageKHR`).
    ///
    /// For planar F16 formats (`PlanarRgb`, `PlanarRgba`) the buffer is an
    /// RGBA16F surface sized `(W/4, C*H)` via `packed_rgba16f_layout`: 4
    /// contiguous f16 elements of the planar `[C, H, W]` byte stream per pixel.
    /// When the allocator pads the pixel stride, `Tensor::image` records the
    /// padded pitch.
    ///
    /// Buffers are allocated with GPU sample and render usage plus the CPU
    /// usage of `access`.
    pub(crate) fn new_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
        access: crate::CpuAccess,
    ) -> Result<Self> {
        let _span = tracing::trace_span!(
            "tensor.alloc",
            memory = "ahardwarebuffer",
            width,
            height,
            ?format,
        )
        .entered();

        let (ahb_format, bpe) = image_format_and_bpe(format, dtype).ok_or_else(|| {
            Error::NotImplemented(format!(
                "AHardwareBufferTensor::new_image: ({format:?}, {dtype:?}) has no \
                 AHardwareBuffer format mapping"
            ))
        })?;

        // Surface geometry per (format, dtype): packed formats use (width,
        // height); planar F16 packs into RGBA16F at `(W/4, C*H)` via
        // `packed_rgba16f_layout`; packed RGB u8 packs into RGBA8888 at
        // `(W*3/4, H)` via `packed_rgb888_layout`.
        let (surface_width, surface_height) = match (format, dtype) {
            (PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
                let layout =
                    packed_rgba16f_layout(format, dtype, width, height).ok_or_else(|| {
                        Error::InvalidShape(format!(
                            "{format:?} F16 AHardwareBuffer requires width%4==0 for RGBA16F \
                             packing (got width={width})"
                        ))
                    })?;
                (layout.surface_w, layout.surface_h)
            }
            (PixelFormat::Rgb, DType::U8 | DType::I8) => {
                let layout = crate::packed_rgb888_layout(width, height).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Rgb u8/i8 AHardwareBuffer requires width%4==0 for RGBA8888 \
                         packing (got width={width})"
                    ))
                })?;
                (layout.surface_w, layout.surface_h)
            }
            _ => (width, height),
        };

        let desc = AHardwareBufferDesc {
            width: surface_width
                .try_into()
                .map_err(|_| Error::InvalidSize(surface_width))?,
            height: surface_height
                .try_into()
                .map_err(|_| Error::InvalidSize(surface_height))?,
            layers: 1,
            format: ahb_format,
            // GPU sample and render usage, plus the CPU usage bits of the
            // declared `CpuAccess`.
            usage: USAGE_GPU_SAMPLED_IMAGE | USAGE_GPU_FRAMEBUFFER | cpu_usage_bits_for(access),
            stride: 0,
            rfu0: 0,
            rfu1: 0,
        };
        let (buffer, desc) = match OwnedAHardwareBuffer::allocate(desc) {
            Ok(ok) => ok,
            Err(e) if access == crate::CpuAccess::None => {
                // Retry once with CPU read/write usage bits.
                log::warn!(
                    "AHardwareBuffer hardware-only allocation failed for \
                     {format:?}/{dtype:?} {width}x{height} ({e:?}); retrying \
                     with CPU access bits (linear layout)"
                );
                let retry = AHardwareBufferDesc {
                    usage: USAGE_GPU_SAMPLED_IMAGE
                        | USAGE_GPU_FRAMEBUFFER
                        | USAGE_CPU_READ_OFTEN
                        | USAGE_CPU_WRITE_OFTEN,
                    ..desc
                };
                OwnedAHardwareBuffer::allocate(retry)?
            }
            Err(e) => return Err(e),
        };
        let (bytes_per_row, buf_size) = desc_layout(&desc).ok_or_else(|| {
            Error::InvalidShape(format!(
                "AHardwareBuffer layout overflow ({surface_width}x{surface_height}, bpe={bpe})"
            ))
        })?;

        let name = match name {
            Some(s) => s.to_owned(),
            None => format!("ahardwarebuffer-img-{}", uuid::Uuid::new_v4()),
        };

        trace!(
            "AHardwareBufferTensor::new_image: name={name} surface={surface_width}x\
             {surface_height} logical={width}x{height} {format:?}/{dtype:?} \
             format=0x{ahb_format:x} stride_px={} bytes={buf_size} bytes_per_row={bytes_per_row}",
            desc.stride,
        );

        Ok(Self {
            name,
            identity: buffer_identity(&buffer),
            buffer,
            shape: shape.to_vec(),
            _marker: PhantomData,
            buf_size,
            bytes_per_row,
            physical_dims: (desc.width as usize, desc.height as usize),
            cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Wrap an existing AHardwareBuffer as a tensor.
    ///
    /// The buffer is acquired for the tensor's lifetime; the external owner
    /// keeps its own reference and must release it independently. The public
    /// entry point is [`crate::Tensor::from_hardware_buffer`].
    ///
    /// # Errors
    ///
    /// `InvalidShape` if the shape footprint exceeds the buffer's derived
    /// allocation size. `NotImplemented` for multi-planar YUV formats (camera
    /// NV12/YUV_420_888).
    ///
    /// # Safety
    ///
    /// `buffer_ptr` must be a valid live AHardwareBuffer pointer.
    pub unsafe fn from_hardware_buffer(
        buffer_ptr: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        unsafe {
            let (buffer, desc) = OwnedAHardwareBuffer::from_external(buffer_ptr)?;
            let (bytes_per_row, buf_size) = desc_layout(&desc).ok_or_else(|| {
                Error::NotImplemented(format!(
                    "from_hardware_buffer: AHardwareBuffer format 0x{:x} has no linear CPU \
                 layout the HAL supports (multi-planar YUV import is a planned follow-up)",
                    desc.format
                ))
            })?;

            // Overflow-checked byte footprint.
            let requested = checked_shape_bytes::<T>(shape)?;
            if requested > buf_size {
                return Err(Error::InvalidShape(format!(
                    "from_hardware_buffer: shape requires {requested} bytes but the buffer only \
                 has {buf_size} (shape={shape:?}, sizeof T={})",
                    std::mem::size_of::<T>(),
                )));
            }

            let name = match name {
                Some(s) => s.to_owned(),
                None => format!("ahardwarebuffer-imported-{}", uuid::Uuid::new_v4()),
            };
            Ok(Self {
                name,
                identity: buffer_identity(&buffer),
                buffer,
                shape: shape.to_vec(),
                _marker: PhantomData,
                buf_size,
                bytes_per_row,
                physical_dims: (desc.width as usize, desc.height as usize),
                cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
                is_imported: true,
                view_offset: 0,
            })
        }
    }

    /// Row pitch in bytes derived from the allocator-chosen `desc.stride`
    /// (pixels × bytes-per-element). For image-formatted buffers this is the
    /// row stride.
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Physical buffer dimensions in texels (`AHardwareBuffer_Desc`
    /// width/height), independent of the tensor's current logical shape and
    /// fixed for the buffer's life.
    pub(crate) fn physical_surface_dims(&self) -> (usize, usize) {
        self.physical_dims
    }

    /// Row pitch to honour as an image stride when a tensor is reconfigured to
    /// a smaller image (`Tensor::configure_image`), or `None` for a BLOB buffer
    /// whose single row is the whole allocation.
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        let (_w, h) = self.physical_dims;
        (h > 1).then_some(self.bytes_per_row)
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

    /// Shared map constructor: checks the window and acquires the CPU lock
    /// through `scoped_pin`, whose keepalive unlocks on drop.
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

    /// Raw AHardwareBuffer pointer for the GL backend
    /// (`eglGetNativeClientBufferANDROID`) or an NPU runtime handoff.
    pub fn hardware_buffer_ptr(&self) -> *mut c_void {
        self.buffer.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// Tensor<T> accessors — Android-only.
// ---------------------------------------------------------------------------

impl<T> crate::Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Borrow the underlying AHardwareBuffer pointer for this tensor
    /// (Android only). Returns `Some(ptr)` when the tensor is backed by an
    /// AHardwareBuffer (i.e. `TensorMemory::DmaBuf` on Android), `None`
    /// otherwise. The pointer is borrowed — its lifetime is tied to the
    /// tensor.
    pub fn hardware_buffer_ptr(&self) -> Option<*mut c_void> {
        match &self.storage {
            crate::TensorStorage::Dma(ahb) => Some(ahb.hardware_buffer_ptr()),
            _ => None,
        }
    }

    /// Physical AHardwareBuffer dimensions in texels, independent of the
    /// logical shape. `None` when not AHardwareBuffer-backed.
    pub fn hardware_buffer_physical_dims(&self) -> Option<(usize, usize)> {
        match &self.storage {
            crate::TensorStorage::Dma(ahb) => Some(ahb.physical_surface_dims()),
            _ => None,
        }
    }

    /// Wrap an externally-allocated AHardwareBuffer as a tensor (Android only).
    /// The buffer is acquired for the tensor's lifetime; the external owner
    /// keeps its own reference and must release it independently.
    ///
    /// # Errors
    ///
    /// `InvalidShape` if the shape footprint exceeds the buffer's allocation.
    ///
    /// # Safety
    ///
    /// `buffer_ptr` must be a valid live AHardwareBuffer pointer.
    pub unsafe fn from_hardware_buffer(
        buffer_ptr: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self>
    where
        T: num_traits::Num,
    {
        let inner =
            unsafe { AHardwareBufferTensor::<T>::from_hardware_buffer(buffer_ptr, shape, name)? };
        // Derive the declared CPU access from the imported buffer's usage bits.
        let declared = match (
            inner.cpu_usage & USAGE_CPU_READ_MASK != 0,
            inner.cpu_usage & USAGE_CPU_WRITE_MASK != 0,
        ) {
            (true, true) => crate::CpuAccess::ReadWrite,
            (true, false) => crate::CpuAccess::Read,
            (false, true) => crate::CpuAccess::Write,
            (false, false) => crate::CpuAccess::None,
        };
        let mut t = crate::Tensor::wrap(crate::TensorStorage::Dma(inner));
        t.set_cpu_access_unchecked(declared);
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// (PixelFormat, DType) → AHardwareBuffer format mapping
// ---------------------------------------------------------------------------

/// The `(PixelFormat, DType) → (AHardwareBuffer format, bytes-per-element)`
/// mapping, for callers in other crates. Returns `None` when the pair has no
/// AHardwareBuffer mapping. The table lives in the
/// [`ahardwarebuffer_layout`](crate::ahardwarebuffer_layout) module.
pub fn image_ahardwarebuffer_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    image_format_and_bpe(format, dtype)
}
