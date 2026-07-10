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
//! (`AHardwareBuffer_lock` with `fence = -1`, blocking on any pending GPU
//! work) and exposes the base address as a slice; `unmap()`/`Drop` calls
//! the matching blocking unlock. Like IOSurface — and unlike Linux
//! DMA-BUF's explicit `DMA_BUF_IOCTL_SYNC` — the lock/unlock pair carries
//! the GPU↔CPU cache-coherency contract.
//!
//! The lock usage flags must match what the buffer was allocated with
//! (they are enum values under `CPU_READ_MASK`/`CPU_WRITE_MASK`, not
//! independent bits), so the tensor records its allocation-time CPU flags
//! and replays them at lock time. A buffer allocated without CPU usage
//! (a future NPU-direct render target) refuses `map()` instead of
//! triggering undefined behaviour.
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
// Raw FFI to libnativewindow (AHardwareBuffer, stable NDK ABI since API 26)
// ---------------------------------------------------------------------------

/// Opaque AHardwareBuffer handle (`struct AHardwareBuffer` in the NDK).
pub(crate) type AHardwareBufferRef = *mut c_void;

/// `AHardwareBuffer_Desc` from `<android/hardware_buffer.h>`. Layout must
/// match the NDK header exactly; `stride` is filled by
/// `AHardwareBuffer_allocate`/`_describe` (in pixels, not bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AHardwareBufferDesc {
    width: u32,
    height: u32,
    layers: u32,
    format: u32,
    usage: u64,
    stride: u32,
    rfu0: u32,
    rfu1: u64,
}

/// `ARect` for `AHardwareBuffer_lock` (null = lock the whole buffer).
#[repr(C)]
struct ARect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

// AHardwareBuffer_Format values (subset used by the HAL).
/// RGBA 8:8:8:8 UNORM — API 26+.
const FORMAT_R8G8B8A8_UNORM: u32 = 1;
/// RGBA 16:16:16:16 half-float — API 26+. The F16 NCHW render target.
const FORMAT_R16G16B16A16_FLOAT: u32 = 0x16;
/// Opaque byte buffer (`width` = length, `height = layers = 1`) — API 26+.
/// The layout NNAPI uses for tensor blobs; the byte-bag allocation format.
const FORMAT_BLOB: u32 = 0x21;

// AHardwareBuffer_UsageFlags (subset used by the HAL). The CPU flags are
// enum VALUES within their masks (READ_OFTEN = 3 = RARELY | 2), not
// independent bits — lock() must replay the allocated values.
const USAGE_CPU_READ_OFTEN: u64 = 0x3;
const USAGE_CPU_READ_MASK: u64 = 0xF;
const USAGE_CPU_WRITE_OFTEN: u64 = 0x30;
const USAGE_CPU_WRITE_MASK: u64 = 0xF0;
/// Sampleable as a GPU texture (`glEGLImageTargetTexture2DOES`).
const USAGE_GPU_SAMPLED_IMAGE: u64 = 0x100;
/// Renderable as a GPU color attachment (FBO render target).
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
#[derive(Debug)]
struct AhbHandle(AHardwareBufferRef);

// SAFETY: AHardwareBuffer is a thread-safe, refcounted kernel object; the
// pointer may be shared and released from any thread (same contract as
// `IoSurfaceHandle`).
unsafe impl Send for AhbHandle {}
unsafe impl Sync for AhbHandle {}

impl Drop for AhbHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { AHardwareBuffer_release(self.0) };
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
                inner: Arc::new(AhbHandle(ptr)),
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
                inner: Arc::new(AhbHandle(ptr)),
            },
            desc,
        ))
    }

    pub(crate) fn as_ptr(&self) -> AHardwareBufferRef {
        self.inner.0
    }
}

/// Bytes-per-element for a known AHardwareBuffer pixel format, or `None`
/// for formats the HAL cannot CPU-map linearly (multi-planar YUV etc.).
fn format_bpe(format: u32) -> Option<usize> {
    match format {
        FORMAT_R8G8B8A8_UNORM => Some(4),
        FORMAT_R16G16B16A16_FLOAT => Some(8),
        FORMAT_BLOB => Some(1),
        _ => None,
    }
}

/// Byte pitch + allocation size derived from a (post-allocation)
/// descriptor. BLOB buffers are `width` bytes with a single "row".
fn desc_layout(desc: &AHardwareBufferDesc) -> Option<(usize, usize)> {
    let bpe = format_bpe(desc.format)?;
    if desc.format == FORMAT_BLOB {
        let len = desc.width as usize;
        return Some((len, len));
    }
    let bytes_per_row = (desc.stride as usize).checked_mul(bpe)?;
    let buf_size = bytes_per_row.checked_mul(desc.height as usize)?;
    Some((bytes_per_row, buf_size))
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
        let byte_size = shape
            .iter()
            .product::<usize>()
            .saturating_mul(std::mem::size_of::<T>());
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
        if needed > self.buf_size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.buf_size,
            });
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

/// Byte footprint of `shape` for element type `T`, with overflow-checked
/// arithmetic — a shape whose element product (or its byte size) wraps
/// `usize` must be rejected, not allowed to slip past a capacity check and
/// produce an out-of-bounds map later.
fn checked_shape_bytes<T>(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| {
            Error::InvalidShape(format!(
                "shape footprint overflows usize (shape={shape:?}, sizeof T={})",
                std::mem::size_of::<T>()
            ))
        })
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
    /// per pixel, identical to the macOS IOSurface geometry, so NPU
    /// runtimes consume the locked base address as `&[f16]` with shape
    /// `[1, C, H, W]` without rearrangement.
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
        // `(W/4, C*H)` via the canonical `packed_rgba16f_layout`.
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
        let mut base: *mut c_void = std::ptr::null_mut();
        // fence = -1: no acquire fence to wait on beyond the buffer's own
        // implicit fencing — the lock blocks until pending GPU work that
        // wrote the buffer completes (the coherency contract, mirroring
        // IOSurfaceLock).
        let rc = unsafe {
            AHardwareBuffer_lock(buffer.as_ptr(), cpu_usage, -1, std::ptr::null(), &mut base)
        };
        if rc != 0 {
            return Err(Error::IoError(std::io::Error::other(format!(
                "AHardwareBuffer_lock failed (rc={rc})"
            ))));
        }
        let base_ptr = NonNull::new(base).ok_or_else(|| {
            // Balance the successful lock before erroring.
            unsafe { AHardwareBuffer_unlock(buffer.as_ptr(), std::ptr::null_mut()) };
            Error::IoError(std::io::Error::other(
                "AHardwareBuffer_lock returned a null address",
            ))
        })?;
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
            // Null fence pointer = synchronous unlock: flushes CPU caches
            // and blocks until the buffer is safe for GPU consumption.
            let rc = unsafe { AHardwareBuffer_unlock(self.buffer.as_ptr(), std::ptr::null_mut()) };
            if rc != 0 {
                log::error!("AHardwareBuffer_unlock failed (rc={rc})");
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

/// AHardwareBuffer format + bytes-per-element mapping for image-formatted
/// buffers, keyed on `(PixelFormat, DType)`. **This function is the single
/// source of truth for the mapping** — the image crate's Android GL
/// backend reads it via [`image_ahardwarebuffer_layout`]; keep the two
/// layers in sync by not duplicating this table (the same rule that
/// prevents the macOS R↔B drift documented in `iosurface.rs`).
///
/// Combinations not listed have no zero-copy path on Android today and
/// fall back to SHM/Mem + CPU conversion:
///
/// * `Grey`/`Nv12`/`Nv16`/`Nv24` u8 — the single-channel
///   `AHARDWAREBUFFER_FORMAT_R8_UNORM` requires API 29; the HAL floor is
///   26. Zero-copy Grey/NV on 29+ is a planned follow-up together with
///   the external-OES YUV sampling path.
/// * `Bgra` u8 — AHardwareBuffer has no BGRA format; mapping it to RGBA
///   would silently swap R↔B (the exact macOS footgun).
/// * `Yuyv` u8 — no packed-4:2:2 AHardwareBuffer format exists.
fn image_format_and_bpe(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        (PixelFormat::Rgba, DType::U8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // The F16 zero-copy path: RGBA16F, both as a packed RGBA image and
        // as the 4-elements-per-pixel packing of planar `[C, H, W]` f16
        // streams (surface sized via `packed_rgba16f_layout`).
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((FORMAT_R16G16B16A16_FLOAT, 8))
        }
        _ => None,
    }
}

/// Public re-export of the `(PixelFormat, DType) → (AHardwareBuffer
/// format, bytes-per-element)` mapping for callers in other crates
/// (specifically the `edgefirst-image` Android GL backend). Returns `None`
/// when the pair has no AHardwareBuffer mapping (see
/// [`image_format_and_bpe`] for the rationale per format).
pub fn image_ahardwarebuffer_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    image_format_and_bpe(format, dtype)
}
