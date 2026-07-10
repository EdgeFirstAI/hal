// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! AHardwareBuffer-backed tensor storage for Android.
//!
//! `AHardwareBufferTensor<T>` is the Android counterpart to `DmaTensor<T>`
//! on Linux and `IoSurfaceTensor<T>` on macOS/iOS: a zero-copy GPU↔CPU
//! buffer that the OpenGL backend imports directly via
//! `EGL_ANDROID_image_native_buffer` (`eglGetNativeClientBufferANDROID` →
//! `eglCreateImageKHR`). All three fit into the `TensorMemory::Dma` slot —
//! the variant name is shared, the inner storage type differs per platform.
//!
//! ## Bindings approach
//!
//! Raw FFI to `libnativewindow.so`, the public NDK library whose
//! AHardwareBuffer C ABI is stable since API 26 (the HAL's Android floor).
//! This mirrors `iosurface.rs`'s direct `#[link]` to the IOSurface
//! framework — no `ndk`/`ndk-sys` crate dependency.
//!
//! ## CPU access
//!
//! `map()` returns `AHardwareBufferMap<T>` which holds the buffer locked
//! (`AHardwareBuffer_lock`) and exposes the base address as a slice;
//! `unmap()`/`Drop` performs the matching unlock. The lock/unlock pair
//! carries the CPU **cache-coherency** contract (invalidate on lock,
//! flush on unlock) — like IOSurface, and unlike Linux DMA-BUF's explicit
//! `DMA_BUF_IOCTL_SYNC`.
//!
//! **Completion is NOT the lock's job.** For GLES-produced content the
//! lock's implicit fencing (`fence = -1`) is driver-dependent — plain GL
//! rendering into an EGLImage sibling does not reliably register a
//! release fence with gralloc. GPU completion is guaranteed upstream: the
//! GL worker issues `finish_via_fence()`/`glFinish` before `convert()`
//! returns, and only then may the CPU map the destination. Do not remove
//! that GL-side barrier on the assumption that the lock waits.
//!
//! The NDK lock contract does not document refcounted nesting (unlike
//! `IOSurfaceLock`), and locking an already-locked buffer is undefined —
//! so the shared handle serializes CPU locks itself: the first live map
//! issues the FFI lock, further maps of the same buffer (e.g. a parent
//! tensor and a `view()` sibling) reuse the same base address, and only
//! the last unmap issues the FFI unlock.
//!
//! The lock usage flags must match what the buffer was allocated with
//! (they are enum values under `CPU_READ_MASK`/`CPU_WRITE_MASK`, not
//! independent bits), so the tensor records its allocation-time CPU flags
//! and replays them at lock time. A buffer allocated without CPU usage
//! (a future NPU-direct render target) refuses `map()`, and a read-only
//! buffer (e.g. an imported camera frame allocated without `CPU_WRITE_*`)
//! maps for reading but panics on `as_mut_slice()` — writing through a
//! read-only lock is undefined behaviour per the NDK contract.
//!
//! ## Geometry
//!
//! `AHardwareBuffer_Desc.stride` is an **output** parameter in **pixels**:
//! gralloc chooses the row pitch at allocation. The tensor caches the
//! derived byte pitch and allocation size at construction — CPU consumers
//! must never recompute a row as `width × bpe`.
//!
//! Generic byte-bag allocations use `AHARDWAREBUFFER_FORMAT_BLOB` (a
//! 1-"row" byte buffer, the layout NNAPI itself uses for tensors and the
//! analog of `iosurface.rs`'s 1-row `L008` surface). Image-formatted
//! allocations use the real pixel formats so the GL backend can bind them
//! as textures; the `(PixelFormat, DType) → format` table is
//! [`image_ahardwarebuffer_layout`], the single source of truth shared
//! with the image crate's import path.

#![cfg(target_os = "android")]

use crate::{
    ahardwarebuffer_layout::{
        checked_shape_bytes, desc_layout, image_format_and_bpe, AHardwareBufferDesc, FORMAT_BLOB,
    },
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

// The AHardwareBuffer_Desc struct, format constants, and the pure layout
// math (format table, descriptor geometry, checked shape footprints) live
// in `ahardwarebuffer_layout.rs` — compiled and unit-tested on every host
// (this FFI module is cfg(android) and no CI lane executes it).

// AHardwareBuffer_UsageFlags (subset used by the HAL). The CPU flags are
// enum VALUES within their masks (NEVER=0, RARELY=2, OFTEN=3), not
// independent bits — lock() must replay the allocated values.
const USAGE_CPU_READ_OFTEN: u64 = 0x3;
const USAGE_CPU_READ_MASK: u64 = 0xF;
const USAGE_CPU_WRITE_OFTEN: u64 = 0x30;
const USAGE_CPU_WRITE_MASK: u64 = 0xF0;
/// Sampleable as a GPU texture (`glEGLImageTargetTexture2DOES`).
const USAGE_GPU_SAMPLED_IMAGE: u64 = 0x100;
/// Renderable as a GPU color attachment (FBO render target). Named
/// `AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT` in current NDK headers;
/// `GPU_FRAMEBUFFER` is the original (still-valid) alias, same value.
const USAGE_GPU_FRAMEBUFFER: u64 = 0x200;
/// Usable as a GPU data buffer; kept on BLOB allocations for the future
/// NNAPI/LiteRT zero-copy handoff.
const USAGE_GPU_DATA_BUFFER: u64 = 0x1000000;

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
}

/// Owned AHardwareBuffer handle. Releases on Drop via
/// `AHardwareBuffer_release`. Cloneable via Arc — every clone shares the
/// same underlying buffer (mirrors `OwnedIoSurface`).
#[derive(Debug, Clone)]
pub(crate) struct OwnedAHardwareBuffer {
    inner: Arc<AhbHandle>,
}

/// Inner wrapper running the actual release in Drop. The Arc means the
/// buffer is released exactly once when the last clone drops.
///
/// Also owns the process-side CPU-lock state: the NDK lock contract does
/// not document refcounted nesting (unlike `IOSurfaceLock`), and locking
/// an already-locked buffer is undefined — so the first live map issues
/// the FFI lock, further concurrent maps of the same buffer (a parent
/// tensor and its `view()` siblings share this handle) reuse the recorded
/// base address, and only the last unmap issues the FFI unlock. All maps
/// of one buffer use the same allocation-time CPU usage flags, so the
/// first lock's flags are valid for every nested map.
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
}

// SAFETY: AHardwareBuffer is a thread-safe, refcounted kernel object; the
// pointer may be shared and released from any thread (same contract as
// `IoSurfaceHandle`). The lock state is Mutex-guarded.
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

    pub(crate) fn as_ptr(&self) -> AHardwareBufferRef {
        self.inner.ptr
    }

    /// Acquire the (refcounted) CPU lock — see [`AhbHandle`]. Returns the
    /// buffer's base address. Every successful call must be balanced by
    /// exactly one [`Self::unlock_shared`].
    fn lock_shared(&self, cpu_usage: u64) -> Result<*mut c_void> {
        let mut st = self.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        if st.count > 0 {
            st.count += 1;
            return Ok(st.base);
        }
        let mut base: *mut c_void = std::ptr::null_mut();
        // fence = -1: no explicit acquire fence. This is a cache-coherency
        // barrier only — GPU completion for GLES-produced content is
        // guaranteed by the GL worker's finish/fence before convert()
        // returns, NOT by this lock (see the module docs).
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
    /// CPU-accessible allocation size in bytes: `bytes_per_row × height`
    /// (BLOB: the requested length). Cached at construction from the
    /// allocator-filled descriptor — never re-derived per map.
    pub(crate) buf_size: usize,
    /// Row pitch in bytes (`desc.stride` pixels × bytes-per-element), the
    /// authoritative stride for CPU row iteration. For BLOB byte-bags this
    /// equals the whole allocation (single-row semantics, like the
    /// IOSurface `L008` byte-bag).
    pub(crate) bytes_per_row: usize,
    /// Physical descriptor dimensions in texels, fixed for the buffer's
    /// life (a reused pool buffer keeps these while `configure_image`
    /// changes the logical shape).
    physical_dims: (usize, usize),
    /// CPU usage flags (`CPU_READ_*`/`CPU_WRITE_*` values) the buffer was
    /// allocated/imported with; replayed by `AHardwareBuffer_lock`.
    cpu_usage: u64,
    /// True when wrapping an externally-allocated buffer (camera frame).
    pub(crate) is_imported: bool,
    /// Byte offset of a sub-region `view()` into the buffer.
    pub(crate) view_offset: usize,
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
        // Unified variant: Android reports Dma, same as Linux DMA-BUF and
        // macOS IOSurface. The variant name is shared; the inner storage
        // type differs per platform.
        TensorMemory::Dma
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        // Overflow-checked byte footprints: a wrapped product that happens
        // to collide with the current count must not smuggle in a bogus
        // shape (same rule as `set_logical_shape`/`view`).
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

    fn map(&self) -> Result<TensorMap<T>> {
        let _span = tracing::trace_span!("tensor.map", memory = "ahardwarebuffer",).entered();
        let m = AHardwareBufferMap::new(
            self.buffer.clone(),
            self.shape.clone(),
            self.buf_size,
            self.cpu_usage,
            self.view_offset,
        )?;
        Ok(TensorMap::HardwareBuffer(m))
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
        // A viewed tensor's window is what remains past its offset — a
        // shape that fits the whole buffer but not the window would then
        // fail at map() time; reject it here instead.
        let capacity = self.buf_size.saturating_sub(self.view_offset);
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    /// Zero-copy sub-region view sharing this buffer (acquired via the
    /// `Arc`-backed `OwnedAHardwareBuffer`) and [`BufferIdentity`],
    /// positioned at `offset_bytes` from this tensor's own window with
    /// logical `shape`. Mirrors [`IoSurfaceTensor::view`] /
    /// [`DmaTensor::view`](crate::TensorTrait::view).
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
    /// Allocate a BLOB buffer large enough to hold `byte_size` bytes
    /// arranged as `shape`. The GL backend separately allocates
    /// properly-formatted image buffers via [`Self::new_image`]; BLOB
    /// buffers carry `GPU_DATA_BUFFER` for the NNAPI-style tensor handoff
    /// but cannot bind as GL textures.
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
            buffer,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size,
            bytes_per_row,
            physical_dims: (desc.width as usize, desc.height as usize),
            cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Allocate an image-formatted AHardwareBuffer — real pixel format +
    /// 2D dimensions, suitable for EGLImage import on the GL backend
    /// (`eglGetNativeClientBufferANDROID` → `eglCreateImageKHR`).
    ///
    /// For planar F16 formats (`PlanarRgb`, `PlanarRgba`) the buffer is
    /// an RGBA16F surface sized `(W/4, C*H)` via `packed_rgba16f_layout`
    /// — 4 contiguous f16 elements of the planar `[C, H, W]` byte stream
    /// per pixel, the same geometry as the macOS IOSurface path. Whether
    /// the locked base address is consumable FLAT as `&[f16]` shaped
    /// `[1, C, H, W]` depends on gralloc: when the allocator pads the
    /// pixel stride (Qualcomm SnapAlloc does, observed on the S26 Ultra),
    /// `Tensor::image` records the padded pitch and CPU consumers must
    /// iterate rows via `effective_row_stride()`; flat consumers (the
    /// future NPU handoff) must check `row_stride()` and repack when set.
    ///
    /// Buffers are allocated with GPU sample + render + CPU read/write
    /// usage (the combination validated on-device by the Phase-1 probe).
    /// Role-tuned usage (e.g. dropping CPU flags for NPU-direct targets)
    /// is a planned follow-up alongside fence export.
    pub(crate) fn new_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
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

        // Surface geometry per (format, dtype) — same scheme as the macOS
        // IOSurface path (see `iosurface.rs::new_image`): packed formats
        // use (width, height); planar F16 packs into RGBA16F at
        // `(W/4, C*H)` via the canonical `packed_rgba16f_layout`; packed
        // RGB u8 packs into RGBA8888 at `(W*3/4, H)` via
        // `packed_rgb888_layout`.
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
            (PixelFormat::Rgb, DType::U8) => {
                let layout = crate::packed_rgb888_layout(width, height).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Rgb u8 AHardwareBuffer requires width%4==0 for RGBA8888 \
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
            usage: USAGE_GPU_SAMPLED_IMAGE
                | USAGE_GPU_FRAMEBUFFER
                | USAGE_CPU_READ_OFTEN
                | USAGE_CPU_WRITE_OFTEN,
            stride: 0,
            rfu0: 0,
            rfu1: 0,
        };
        let (buffer, desc) = OwnedAHardwareBuffer::allocate(desc)?;
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
            buffer,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size,
            bytes_per_row,
            physical_dims: (desc.width as usize, desc.height as usize),
            cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
            is_imported: false,
            view_offset: 0,
        })
    }

    /// Wrap an existing AHardwareBuffer as a tensor. Used to import
    /// externally-allocated buffers (CameraX/ImageReader frames handed
    /// through JNI, NNAPI blobs, cross-process buffers received via
    /// binder).
    ///
    /// The buffer is acquired for the tensor's lifetime; the external
    /// owner keeps its own reference and must release it independently.
    ///
    /// The type-erased/typed public entry point is
    /// [`crate::Tensor::from_hardware_buffer`]; most callers should
    /// prefer that over this inner constructor.
    ///
    /// # Safety
    ///
    /// The caller must ensure `buffer_ptr` is a valid live AHardwareBuffer
    /// pointer. Passing a stale or invalid pointer is UB.
    ///
    /// The shape footprint is validated against the buffer's derived
    /// allocation size and the constructor returns `Err(InvalidShape)` if
    /// it does not fit. Multi-planar YUV formats (camera NV12/YUV_420_888)
    /// are refused with `NotImplemented` for now: their CPU mapping needs
    /// `AHardwareBuffer_lockPlanes` (API 29) and their GL sampling needs
    /// the external-OES path — both planned follow-ups.
    pub unsafe fn from_hardware_buffer(
        buffer_ptr: *mut c_void,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Self> {
        let (buffer, desc) = OwnedAHardwareBuffer::from_external(buffer_ptr)?;
        let (bytes_per_row, buf_size) = desc_layout(&desc).ok_or_else(|| {
            Error::NotImplemented(format!(
                "from_hardware_buffer: AHardwareBuffer format 0x{:x} has no linear CPU \
                 layout the HAL supports (multi-planar YUV import is a planned follow-up)",
                desc.format
            ))
        })?;

        // Overflow-checked: the element product itself can wrap before a
        // checked byte multiply would notice.
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
            buffer,
            shape: shape.to_vec(),
            _marker: PhantomData,
            identity: BufferIdentity::new(),
            buf_size,
            bytes_per_row,
            physical_dims: (desc.width as usize, desc.height as usize),
            cpu_usage: desc.usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK),
            is_imported: true,
            view_offset: 0,
        })
    }

    /// Row pitch in bytes derived from the allocator-chosen `desc.stride`
    /// (pixels × bytes-per-element). For image-formatted buffers this is
    /// the authoritative row stride; CPU consumers must iterate rows by
    /// this value, never `width × bpe`.
    pub(crate) fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }

    /// Physical buffer dimensions in texels (`AHardwareBuffer_Desc`
    /// width/height), independent of the tensor's current logical shape.
    /// Fixed for the buffer's life — a reused pool buffer keeps these
    /// while `configure_image` changes the logical frame shape per decode.
    pub(crate) fn physical_surface_dims(&self) -> (usize, usize) {
        self.physical_dims
    }

    /// Row pitch to honour as an *image* stride when a tensor is
    /// reconfigured to a smaller image (`Tensor::configure_image`), or
    /// `None` for a BLOB byte-bag whose single "row" is the whole
    /// allocation (same rule as the IOSurface `L008` byte-bag).
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        let (_w, h) = self.physical_dims;
        (h > 1).then_some(self.bytes_per_row)
    }

    /// Lock and map exposing `byte_size` bytes via `as_slice()` for
    /// strided row iteration. The caller (`Tensor::map`) validates
    /// `byte_size <= buf_size` first. The lock yields the full buffer base
    /// address, so the strided view is genuinely zero-copy.
    pub(crate) fn map_with_byte_size(&self, byte_size: usize) -> Result<TensorMap<T>> {
        let m = AHardwareBufferMap::new_with_byte_size(
            self.buffer.clone(),
            self.shape.clone(),
            self.buf_size,
            byte_size,
            self.cpu_usage,
            self.view_offset,
        )?;
        Ok(TensorMap::HardwareBuffer(m))
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
    /// AHardwareBuffer (i.e. `TensorMemory::Dma` on Android), `None`
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

    /// Wrap an externally-allocated AHardwareBuffer as a tensor (Android
    /// only). Used to import buffers from CameraX/ImageReader (via JNI),
    /// NNAPI, or cross-process binder transfers. The buffer is acquired
    /// for the tensor's lifetime; the external owner keeps its own
    /// reference and must release it independently.
    ///
    /// # Safety
    ///
    /// `buffer_ptr` must be a valid live AHardwareBuffer pointer. Passing
    /// a stale or invalid pointer is UB.
    ///
    /// HAL validates that the shape footprint fits within the buffer's
    /// allocation and returns `Err(InvalidShape)` otherwise. The
    /// pointer-validity requirement above is the caller's responsibility.
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
        Ok(crate::Tensor::wrap(crate::TensorStorage::Dma(inner)))
    }
}

// ---------------------------------------------------------------------------
// AHardwareBufferMap — locked-for-CPU view.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AHardwareBufferMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    buffer: OwnedAHardwareBuffer,
    shape: Vec<usize>,
    base_ptr: NonNull<c_void>,
    buf_size: usize,
    /// When `Some(bytes)`, `as_slice()` exposes `bytes / sizeof(T)`
    /// elements (the full row-padded buffer) instead of `shape.product()`.
    /// Mirrors `DmaMap`/`IoSurfaceMap`: used for strided tensors so CPU
    /// callers iterate rows via `effective_row_stride()` without running
    /// past the locked region.
    byte_size_override: Option<usize>,
    _marker: PhantomData<T>,
    locked: bool,
    /// Whether the buffer's CPU usage includes a write flag. Writing
    /// through a lock without CPU_WRITE is undefined behaviour per the
    /// NDK contract, so the mutable accessors panic instead (a read-only
    /// imported camera buffer maps fine for reading).
    writable: bool,
}

// SAFETY: the locked base address stays valid until unlock, and
// AHardwareBuffer itself is thread-safe (same contract as `IoSurfaceMap`).
unsafe impl<T> Send for AHardwareBufferMap<T> where T: Num + Clone + fmt::Debug {}
unsafe impl<T> Sync for AHardwareBufferMap<T> where T: Num + Clone + fmt::Debug {}

impl<T> AHardwareBufferMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn new(
        buffer: OwnedAHardwareBuffer,
        shape: Vec<usize>,
        buf_size: usize,
        cpu_usage: u64,
        view_offset: usize,
    ) -> Result<Self> {
        Self::new_inner(buffer, shape, buf_size, None, cpu_usage, view_offset)
    }

    fn new_with_byte_size(
        buffer: OwnedAHardwareBuffer,
        shape: Vec<usize>,
        buf_size: usize,
        byte_size: usize,
        cpu_usage: u64,
        view_offset: usize,
    ) -> Result<Self> {
        Self::new_inner(
            buffer,
            shape,
            buf_size,
            Some(byte_size),
            cpu_usage,
            view_offset,
        )
    }

    fn new_inner(
        buffer: OwnedAHardwareBuffer,
        shape: Vec<usize>,
        buf_size: usize,
        byte_size_override: Option<usize>,
        cpu_usage: u64,
        view_offset: usize,
    ) -> Result<Self> {
        // The lock usage must replay the allocation-time CPU flags (they
        // are enum values under the CPU masks — see module docs). A buffer
        // without CPU usage cannot be mapped; refuse instead of invoking
        // undefined behaviour.
        if cpu_usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK) == 0 {
            return Err(Error::InvalidOperation(
                "AHardwareBufferMap: buffer was allocated without CPU usage flags".into(),
            ));
        }
        // Validate the exposed window BEFORE locking: the slice handed out
        // by `deref` is `elem_count()` elements starting `view_offset`
        // bytes in, so both the override and the shape-derived length must
        // fit `buf_size − view_offset`. Callers pre-validate, but the
        // release-mode `deref` has only a `debug_assert` — this is the
        // load-bearing bounds check (mirrors `DmaMap`).
        if view_offset > buf_size {
            return Err(Error::InsufficientCapacity {
                needed: view_offset,
                capacity: buf_size,
            });
        }
        let window = buf_size - view_offset;
        let exposed_bytes = match byte_size_override {
            Some(bytes) => bytes,
            None => checked_shape_bytes::<T>(&shape)?,
        };
        if exposed_bytes > window {
            return Err(Error::InsufficientCapacity {
                needed: exposed_bytes,
                capacity: window,
            });
        }
        // Refcounted lock (see `AhbHandle`): the first live map issues the
        // FFI lock; nested maps of the same buffer reuse its base address.
        let base = buffer.lock_shared(cpu_usage)?;
        let base_ptr = NonNull::new(base).expect("lock_shared validates the base address");
        // A sub-view window starts `view_offset` bytes into the locked
        // buffer (bounds-checked above). Advance the exposed base and
        // shrink the remaining-window bound so the `deref` length check is
        // against the window.
        let base_ptr = NonNull::new(unsafe { base_ptr.as_ptr().byte_add(view_offset) })
            .expect("offset base within locked buffer is non-null");
        Ok(Self {
            buffer,
            shape,
            base_ptr,
            buf_size: buf_size - view_offset,
            byte_size_override,
            _marker: PhantomData,
            locked: true,
            writable: cpu_usage & USAGE_CPU_WRITE_MASK != 0,
        })
    }

    fn elem_count(&self) -> usize {
        match self.byte_size_override {
            Some(bytes) => bytes / std::mem::size_of::<T>(),
            None => self.shape.iter().product(),
        }
    }
}

impl<T> TensorMapTrait<T> for AHardwareBufferMap<T>
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
            // Releases one refcount; the last live map performs the
            // synchronous FFI unlock (flushing CPU caches).
            self.buffer.unlock_shared();
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

impl<T> Deref for AHardwareBufferMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        let ptr = self.base_ptr.as_ptr() as *const T;
        let len = self.elem_count();
        debug_assert!(
            len * std::mem::size_of::<T>() <= self.buf_size,
            "AHardwareBufferMap deref: {} elems × {} bytes > buf_size {}",
            len,
            std::mem::size_of::<T>(),
            self.buf_size,
        );
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl<T> DerefMut for AHardwareBufferMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        // Writing through a lock whose buffer lacks CPU_WRITE usage is
        // undefined behaviour per the NDK contract — panic with a clear
        // message instead (read-only imported camera buffers hit this).
        assert!(
            self.writable,
            "AHardwareBufferMap: buffer was allocated/imported without CPU_WRITE usage; \
             mutable access would be undefined behaviour"
        );
        let ptr = self.base_ptr.as_ptr() as *mut T;
        let len = self.elem_count();
        // Symmetric with `Deref::deref` — an oversized mutable write must
        // be caught the same way an oversized read would be.
        debug_assert!(
            len * std::mem::size_of::<T>() <= self.buf_size,
            "AHardwareBufferMap deref_mut: {} elems × {} bytes > buf_size {}",
            len,
            std::mem::size_of::<T>(),
            self.buf_size,
        );
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

impl<T> Drop for AHardwareBufferMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        self.unmap();
    }
}

// ---------------------------------------------------------------------------
// (PixelFormat, DType) → AHardwareBuffer format mapping
// ---------------------------------------------------------------------------

/// Public re-export of the `(PixelFormat, DType) → (AHardwareBuffer
/// format, bytes-per-element)` mapping for callers in other crates
/// (specifically the `edgefirst-image` Android GL backend). Returns `None`
/// when the pair has no AHardwareBuffer mapping. The table itself lives in
/// the host-tested [`ahardwarebuffer_layout`](crate::ahardwarebuffer_layout)
/// module (single source of truth — see its docs for the per-format
/// rationale).
pub fn image_ahardwarebuffer_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    image_format_and_bpe(format, dtype)
}
