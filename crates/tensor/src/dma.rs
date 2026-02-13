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

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
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

        let dma_fd = heap.allocate(size)?;
        let stat = fstat(&dma_fd)?;
        debug!("DMA memory stat: {stat:?}");

        let drm_attachment = crate::dmabuf::DrmAttachment::new(&dma_fd);

        Ok(DmaTensor::<T> {
            name: name.to_owned(),
            fd: dma_fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            _drm_attachment: drm_attachment,
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

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        #[cfg(target_os = "linux")]
        let drm_attachment = crate::dmabuf::DrmAttachment::new(&fd);

        Ok(DmaTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            #[cfg(target_os = "linux")]
            _drm_attachment: drm_attachment,
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
        )?))
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
        let drm_attachment = crate::dmabuf::DrmAttachment::new(&fd);
        Ok(Self {
            name: self.name.clone(),
            fd,
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
            #[cfg(target_os = "linux")]
            _drm_attachment: drm_attachment,
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
    _marker: std::marker::PhantomData<T>,
}

impl<T> DmaMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    pub fn new(fd: OwnedFd, shape: &[usize]) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        #[cfg(target_os = "linux")]
        crate::dmabuf::start_readwrite(&fd)?;

        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                NonZero::new(size).ok_or(Error::InvalidSize(size))?,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &fd,
                0,
            )?
        };

        trace!("Mapping DMA memory: {ptr:?}");
        let dma_ptr = DmaPtr(NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(size))?);
        Ok(DmaMap {
            ptr: Arc::new(Mutex::new(dma_ptr)),
            fd,
            shape: shape.to_vec(),
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

        if let Err(e) = unsafe { nix::sys::mman::munmap(**ptr, self.size()) } {
            warn!("Failed to unmap DMA memory: {e}");
        }

        #[cfg(target_os = "linux")]
        if let Err(e) = crate::dmabuf::end_readwrite(&self.fd) {
            warn!("Failed to end read/write on DMA memory: {e}");
        }
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
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
