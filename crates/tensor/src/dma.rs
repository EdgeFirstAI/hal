// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
};
use log::{trace, warn};
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    num::NonZero,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, OwnedFd},
    ptr::NonNull,
    sync::{Arc, Mutex},
};

/// A tensor backed by DMA (Direct Memory Access) memory.
///
/// On Linux, for self-allocated (dma_heap) buffers a DRM PRIME attachment is
/// created to enable CPU cache coherency via `DMA_BUF_IOCTL_SYNC`. Without an
/// active attachment, sync ioctls are no-ops on cached CMA heaps.
///
/// For imported (foreign) DMA-BUF fds — e.g. those exported by the Neutron
/// NPU driver — no DRM attachment is created. Cache coherency for foreign
/// buffers is the responsibility of the buffer owner (the kernel driver).
#[derive(Debug)]
pub struct DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub fd: OwnedFd,
    pub shape: Vec<usize>,
    pub _marker: std::marker::PhantomData<T>,
    #[cfg(target_os = "linux")]
    _drm_attachment: Option<crate::dmabuf::DrmAttachment>,
    identity: crate::BufferIdentity,
    /// Actual buffer size in bytes (from fstat at creation time).
    /// May be larger than shape.product() * sizeof(T) for externally
    /// allocated buffers with row padding.
    pub(crate) buf_size: usize,
    /// Byte offset into the DMA buffer where the tensor data begins.
    /// Set via `Tensor::set_plane_offset` for sub-region imports.
    pub(crate) mmap_offset: usize,
    /// Whether this tensor was created via `from_fd()` (imported from an
    /// external allocator).  Propagated through `try_clone()` so that DRM
    /// PRIME import failures are logged at DEBUG rather than WARN, and
    /// used to gate CPU mapping of strided tensors: self-allocated DMA
    /// tensors with pitch padding (via `new_with_byte_size`) are
    /// mappable because HAL owns the layout, but foreign V4L2/GStreamer
    /// strided imports are not — the external allocator defines the
    /// layout and HAL cannot validate what the caller expects.
    #[cfg(target_os = "linux")]
    pub(crate) is_imported: bool,
}

unsafe impl<T> Send for DmaTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for DmaTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> TensorTrait<T> for DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    #[cfg(target_os = "linux")]
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        use log::debug;
        use nix::sys::stat::fstat;

        let logical_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let name = match name {
            Some(name) => name.to_owned(),
            None => {
                let uuid = uuid::Uuid::new_v4().as_simple().to_string();
                format!("/{}", &uuid[..16])
            }
        };

        let heap = match dma_heap::Heap::new(dma_heap::HeapKind::Cma) {
            Ok(heap) => heap,
            Err(_) => dma_heap::Heap::new(dma_heap::HeapKind::System)?,
        };

        let dma_fd = heap.allocate(logical_size)?;
        let stat = fstat(&dma_fd)?;
        debug!("DMA memory stat: {stat:?}");
        let buf_size = if stat.st_size > 0 {
            std::cmp::max(stat.st_size as usize, logical_size)
        } else {
            logical_size
        };

        let drm_attachment = crate::dmabuf::DrmAttachment::new(&dma_fd, false);

        Ok(DmaTensor::<T> {
            name: name.to_owned(),
            fd: dma_fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            _drm_attachment: drm_attachment,
            identity: crate::BufferIdentity::new(),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "DMA tensors are not supported on this platform".to_owned(),
        ))
    }

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let logical_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if logical_size == 0 {
            return Err(Error::InvalidSize(0));
        }

        // fstat may return st_size=0 for DMA-BUF fds on some kernels;
        // fall back to logical_size in that case.
        let buf_size = {
            #[cfg(target_os = "linux")]
            {
                use nix::sys::stat::fstat;
                match fstat(&fd) {
                    Ok(stat) if stat.st_size > 0 && stat.st_size as usize >= logical_size => {
                        stat.st_size as usize
                    }
                    _ => logical_size,
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                logical_size
            }
        };

        // Do NOT attempt a DRM attachment for foreign (imported) DMA-BUF fds.
        // DRM PRIME import is only meaningful for DMA-BUF fds that were
        // allocated by the same DRM device (e.g. via the CMA/system heap).
        // For fds owned by other kernel drivers (e.g. Neutron NPU), the
        // PRIME_FD_TO_HANDLE ioctl will fail and the resulting no-op
        // attachment attempt adds unnecessary ioctl overhead on every import.
        // DMA_BUF_IOCTL_SYNC coherency for foreign buffers is the
        // responsibility of the buffer owner (the NPU driver in this case).
        #[cfg(target_os = "linux")]
        let drm_attachment = None;

        Ok(DmaTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            #[cfg(target_os = "linux")]
            _drm_attachment: drm_attachment,
            identity: crate::BufferIdentity::new(),
            buf_size,
            mmap_offset: 0,
            #[cfg(target_os = "linux")]
            is_imported: true,
        })
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        Ok(self.fd.try_clone()?)
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Dma
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let new_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if new_size != self.size() {
            return Err(Error::ShapeMismatch(format!(
                "Cannot reshape incompatible shape: {:?} to {:?}",
                self.shape, shape
            )));
        }

        self.shape = shape.to_vec();
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        Ok(TensorMap::Dma(DmaMap::new(
            self.fd.try_clone()?,
            &self.shape,
            self.buf_size,
            self.mmap_offset,
        )?))
    }

    fn buffer_identity(&self) -> &crate::BufferIdentity {
        &self.identity
    }
}

impl<T> AsRawFd for DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.fd.as_raw_fd()
    }
}

impl<T> DmaTensor<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug + Send + Sync,
{
    /// Allocate a DMA-BUF with an explicit byte size that may exceed
    /// `shape.product() * sizeof(T)`.
    ///
    /// Used for image tensors that need a row-padded layout so the
    /// resulting DMA-BUF satisfies a downstream consumer's pitch
    /// alignment requirement (e.g. Mali Valhall's 64-byte EGLImage
    /// import rule). The `shape` field stores the **logical** dimensions
    /// `[height, width, channels]`, so `Tensor::width()` / `height()` /
    /// `shape()` continue to report the user-requested values; the
    /// padding is carried separately by `Tensor::row_stride` and is
    /// visible to the CPU mapping (which spans the full `byte_size`
    /// bytes) but not to the logical shape.
    ///
    /// Errors:
    /// - `InvalidArgument` if `byte_size < shape.product() * sizeof(T)`
    ///   (the request would lose data)
    /// - `IoError` if the DMA-heap allocation fails
    #[cfg(target_os = "linux")]
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        use log::debug;
        use nix::sys::stat::fstat;

        // Compute the logical byte size with checked arithmetic. A caller
        // passing an absurdly large shape (or sizeof::<T> × product) must
        // not silently wrap — the comparison below would then accept an
        // allocation that's actually smaller than the logical size.
        let logical_elems = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "DmaTensor::new_with_byte_size: shape.product() overflows usize \
                     (shape={shape:?})"
                ))
            })?;
        let logical_size = logical_elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "DmaTensor::new_with_byte_size: logical_elems {logical_elems} × \
                     sizeof::<T>={} overflows usize (shape={shape:?})",
                    std::mem::size_of::<T>()
                ))
            })?;
        if byte_size < logical_size {
            return Err(Error::InvalidArgument(format!(
                "DmaTensor::new_with_byte_size: byte_size {byte_size} < logical {logical_size} \
                 (shape={shape:?}, sizeof::<T>={})",
                std::mem::size_of::<T>()
            )));
        }
        let name = match name {
            Some(name) => name.to_owned(),
            None => {
                let uuid = uuid::Uuid::new_v4().as_simple().to_string();
                format!("/{}", &uuid[..16])
            }
        };

        let heap = match dma_heap::Heap::new(dma_heap::HeapKind::Cma) {
            Ok(heap) => heap,
            Err(_) => dma_heap::Heap::new(dma_heap::HeapKind::System)?,
        };

        let dma_fd = heap.allocate(byte_size)?;
        let stat = fstat(&dma_fd)?;
        debug!("DMA padded memory stat: {stat:?}");
        let buf_size = if stat.st_size > 0 {
            std::cmp::max(stat.st_size as usize, byte_size)
        } else {
            byte_size
        };

        let drm_attachment = crate::dmabuf::DrmAttachment::new(&dma_fd, false);

        Ok(DmaTensor::<T> {
            name,
            fd: dma_fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            _drm_attachment: drm_attachment,
            identity: crate::BufferIdentity::new(),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) fn new_with_byte_size(
        _shape: &[usize],
        _byte_size: usize,
        _name: Option<&str>,
    ) -> Result<Self> {
        Err(Error::NotImplemented(
            "DMA tensors are not supported on this platform".to_owned(),
        ))
    }

    /// Map this DMA tensor with an explicit total byte size.
    ///
    /// Used by `Tensor::map()` for self-allocated strided tensors — the
    /// returned `DmaMap` exposes the full `byte_size` bytes via
    /// `as_slice()`/`as_mut_slice()`, not just the shape-derived logical
    /// count. Callers are expected to iterate rows with
    /// `Tensor::effective_row_stride()` so they don't read past the end.
    pub(crate) fn map_with_byte_size(&self, byte_size: usize) -> Result<DmaMap<T>> {
        DmaMap::new_with_byte_size(
            self.fd.try_clone()?,
            &self.shape,
            self.buf_size,
            self.mmap_offset,
            byte_size,
        )
    }

    pub fn try_clone(&self) -> Result<Self> {
        let fd = self.clone_fd()?;
        // Preserve the imported/owned distinction: imported fds never get a
        // DRM attachment (consistent with from_fd()).
        #[cfg(target_os = "linux")]
        let drm_attachment = if self.is_imported {
            None
        } else {
            crate::dmabuf::DrmAttachment::new(&fd, false)
        };
        Ok(Self {
            name: self.name.clone(),
            fd,
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
            #[cfg(target_os = "linux")]
            _drm_attachment: drm_attachment,
            identity: self.identity.clone(),
            buf_size: self.buf_size,
            mmap_offset: self.mmap_offset,
            #[cfg(target_os = "linux")]
            is_imported: self.is_imported,
        })
    }
}

#[derive(Debug)]
pub struct DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    ptr: Arc<Mutex<DmaPtr>>,
    fd: OwnedFd,
    shape: Vec<usize>,
    /// Actual mmap'd size (may be > shape.product() * sizeof(T) for padded buffers).
    mmap_size: usize,
    /// Byte offset into the mmap'd region where tensor data begins.
    offset: usize,
    /// Optional override for `as_slice().len() * sizeof(T)`. When `None`,
    /// `as_slice()` returns `shape.product()` elements (the traditional
    /// logical view). When `Some(bytes)`, `as_slice()` returns `bytes /
    /// sizeof(T)` elements, exposing the full padded buffer. Used for
    /// self-allocated strided DMA tensors where the mmap'd region has
    /// row-padding between logical rows and callers need to iterate via
    /// `row_stride` rather than a packed `width * bpp` layout.
    byte_size_override: Option<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    pub fn new(fd: OwnedFd, shape: &[usize], buf_size: usize, offset: usize) -> Result<Self> {
        Self::new_internal(fd, shape, buf_size, offset, None)
    }

    /// Construct a DmaMap whose `as_slice()` exposes the full padded
    /// buffer rather than the shape-derived logical byte count. Used by
    /// `Tensor::map()` for self-allocated strided DMA tensors so CPU
    /// iteration can respect `row_stride` without going past the end
    /// of the returned slice.
    pub fn new_with_byte_size(
        fd: OwnedFd,
        shape: &[usize],
        buf_size: usize,
        offset: usize,
        byte_size: usize,
    ) -> Result<Self> {
        Self::new_internal(fd, shape, buf_size, offset, Some(byte_size))
    }

    fn new_internal(
        fd: OwnedFd,
        shape: &[usize],
        buf_size: usize,
        offset: usize,
        byte_size_override: Option<usize>,
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let logical_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if logical_size == 0 {
            return Err(Error::InvalidSize(0));
        }

        // Use the buffer's actual size (from fstat at DmaTensor creation).
        // as_slice() uses the logical element count from shape.
        // When an offset is present (sub-region of a larger DMA-BUF), verify
        // that offset + logical_size fits within the allocated buffer — mapping
        // beyond buf_size would cause SIGBUS on access.
        let total_needed = offset
            .checked_add(logical_size)
            .ok_or(Error::InvalidSize(0))?;
        if total_needed > buf_size {
            warn!(
                "DmaMap: offset={} + logical_size={} = {} exceeds buf_size={} (fd={})",
                offset,
                logical_size,
                total_needed,
                buf_size,
                fd.as_raw_fd()
            );
            return Err(Error::InvalidSize(total_needed));
        }
        if std::mem::size_of::<T>() > 1 && !offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "DmaMap: offset {} is not aligned to align_of::<T>()={}",
                offset,
                std::mem::align_of::<T>()
            )));
        }
        let mmap_size = buf_size;

        #[cfg(target_os = "linux")]
        {
            trace!("DmaMap: sync start fd={} size={mmap_size}", fd.as_raw_fd());
            if let Err(e) = crate::dmabuf::start_readwrite(&fd) {
                warn!(
                    "DmaMap: DMA_BUF_IOCTL_SYNC(START) failed fd={}: {e}",
                    fd.as_raw_fd()
                );
                return Err(Error::NixError(e));
            }
        }

        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                NonZero::new(mmap_size).ok_or(Error::InvalidSize(mmap_size))?,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &fd,
                0,
            )?
        };

        trace!("Mapping DMA memory: {ptr:?}");
        let dma_ptr = DmaPtr(NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?);
        Ok(DmaMap {
            ptr: Arc::new(Mutex::new(dma_ptr)),
            fd,
            shape: shape.to_vec(),
            mmap_size,
            offset,
            byte_size_override,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T> Deref for DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[derive(Debug)]
struct DmaPtr(NonNull<c_void>);
impl Deref for DmaPtr {
    type Target = NonNull<c_void>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for DmaPtr {}

impl<T> TensorMapTrait<T> for DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");

        if let Err(e) = unsafe { nix::sys::mman::munmap(**ptr, self.mmap_size) } {
            warn!("Failed to unmap DMA memory: {e}");
        }

        #[cfg(target_os = "linux")]
        if let Err(e) = crate::dmabuf::end_readwrite(&self.fd) {
            warn!("Failed to end read/write on DMA memory: {e}");
        }
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        let base = unsafe { (ptr.as_ptr() as *const u8).add(self.offset) as *const T };
        unsafe { std::slice::from_raw_parts(base, self.slice_len_elems()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        let base = unsafe { (ptr.as_ptr() as *mut u8).add(self.offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(base, self.slice_len_elems()) }
    }
}

impl<T> DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    /// Number of `T` elements exposed by `as_slice()`. Honours
    /// `byte_size_override` when set (for strided tensors the caller
    /// wants the full padded mmap exposed, not just `shape.product()`).
    /// Falls back to the shape-derived logical element count.
    fn slice_len_elems(&self) -> usize {
        match self.byte_size_override {
            Some(bytes) => bytes / std::mem::size_of::<T>(),
            None => self.shape.iter().product(),
        }
    }
}

impl<T> Drop for DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        trace!("DmaMap dropped, unmapping memory");
        self.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a valid fd backed by /dev/null.  The new error paths in
    /// DmaMap::new() all fire before any fd-specific syscall (mmap,
    /// DMA_BUF_IOCTL_SYNC), so any readable fd is sufficient.
    #[cfg(target_os = "linux")]
    fn dummy_fd() -> std::os::fd::OwnedFd {
        use std::os::fd::FromRawFd;
        use std::os::unix::io::IntoRawFd;
        let f = std::fs::File::open("/dev/null").expect("open /dev/null");
        unsafe { std::os::fd::OwnedFd::from_raw_fd(f.into_raw_fd()) }
    }

    /// offset + logical_size exceeds buf_size — must return InvalidSize.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_offset_exceeds_buf_size() {
        let fd = dummy_fd();
        // shape=[4096] u8 → logical_size=4096; offset=4096 → total_needed=8192
        // buf_size=4096 < 8192 → error
        let result = DmaMap::<u8>::new(fd, &[4096], 4096, 4096);
        match result {
            Err(Error::InvalidSize(n)) => assert_eq!(n, 8192),
            other => panic!("expected InvalidSize(8192), got {:?}", other),
        }
    }

    /// Offset not aligned to align_of::<T>() — must return InvalidOperation.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_misaligned_offset() {
        let fd = dummy_fd();
        // shape=[1024] u32 → logical_size=4096; offset=3 (not aligned to 4)
        // buf_size=8192 so total_needed check passes; alignment check fires
        let result = DmaMap::<u32>::new(fd, &[1024], 8192, 3);
        assert!(
            matches!(result, Err(Error::InvalidOperation(_))),
            "expected InvalidOperation for misaligned offset, got {:?}",
            result
        );
    }

    /// offset + logical_size overflows usize — must return InvalidSize(0).
    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_offset_overflow() {
        let fd = dummy_fd();
        // offset=usize::MAX, shape=[1] u8 → checked_add overflows
        let result = DmaMap::<u8>::new(fd, &[1], usize::MAX, usize::MAX);
        assert!(
            matches!(result, Err(Error::InvalidSize(0))),
            "expected InvalidSize(0) on overflow, got {:?}",
            result
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_with_offset() {
        use crate::{Tensor, TensorMapTrait, TensorMemory, TensorTrait};

        // Skip if DMA heap not available
        let total_size: usize = 4096 * 4; // 16KB
        let offset: usize = 4096; // 4KB offset
        let data_size: usize = 4096; // 4KB of data after offset

        let large_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::Dma), None) {
            Ok(buf) => buf,
            Err(_) => {
                eprintln!("SKIPPED: DMA not available");
                return;
            }
        };

        // Fill entire buffer with sentinel
        {
            let mut map = large_buf.map().unwrap();
            map.as_mut_slice().fill(0xAA);
        }

        // Import at offset as a smaller tensor using clone_fd + set_plane_offset
        let fd = large_buf.clone_fd().unwrap();
        let mut offset_tensor = Tensor::<u8>::from_fd(fd, &[data_size], None).unwrap();
        offset_tensor.set_plane_offset(offset);

        // Map the offset tensor — should succeed (not rejected)
        let mut map = offset_tensor.map().unwrap();
        let slice = map.as_mut_slice();

        // Should see the sentinel at the offset position
        assert_eq!(slice.len(), data_size);
        assert!(
            slice.iter().all(|&b| b == 0xAA),
            "Offset tensor map should see sentinel data at offset"
        );

        // Write different data at offset
        slice.fill(0xBB);
        drop(map);

        // Verify via the original buffer: bytes before offset unchanged,
        // bytes at offset are 0xBB
        {
            let map = large_buf.map().unwrap();
            let buf = map.as_slice();
            assert!(
                buf[..offset].iter().all(|&b| b == 0xAA),
                "Data before offset should be unchanged"
            );
            assert!(
                buf[offset..offset + data_size].iter().all(|&b| b == 0xBB),
                "Data at offset should be 0xBB"
            );
        }
    }
}
