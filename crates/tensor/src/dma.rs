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
/// On Linux, a DRM PRIME attachment is created to enable proper CPU cache
/// coherency via `DMA_BUF_IOCTL_SYNC`. Without this attachment, sync ioctls
/// are no-ops on cached CMA heaps.
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
    buf_size: usize,
    /// Byte offset into the DMA buffer where the tensor data begins.
    /// Set via `Tensor::set_plane_offset` for sub-region imports.
    pub(crate) mmap_offset: usize,
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

        #[cfg(target_os = "linux")]
        let drm_attachment = crate::dmabuf::DrmAttachment::new(&fd, true);

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
    pub fn try_clone(&self) -> Result<Self> {
        let fd = self.clone_fd()?;
        #[cfg(target_os = "linux")]
        let drm_attachment = crate::dmabuf::DrmAttachment::new(&fd, false);
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
    _marker: std::marker::PhantomData<T>,
}

impl<T> DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    pub fn new(fd: OwnedFd, shape: &[usize], buf_size: usize, offset: usize) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let logical_size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if logical_size == 0 {
            return Err(Error::InvalidSize(0));
        }

        // Use the buffer's actual size (from fstat at DmaTensor creation) to ensure
        // mmap covers the full allocation, including any row padding from external
        // allocators. as_slice() still uses the logical element count from shape.
        // When an offset is present (sub-region of a larger DMA-BUF), ensure
        // the mmap covers offset + logical_size so the slice can start at the
        // offset position.
        let total_needed = offset
            .checked_add(logical_size)
            .ok_or(Error::InvalidSize(0))?;
        let mmap_size = std::cmp::max(buf_size, total_needed);

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
        unsafe { std::slice::from_raw_parts(base, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        let base = unsafe { (ptr.as_ptr() as *mut u8).add(self.offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(base, self.len()) }
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
