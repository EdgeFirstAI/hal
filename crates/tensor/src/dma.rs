// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMemory, TensorTrait,
};
use num_traits::Num;
use std::{
    fmt,
    num::NonZero,
    os::fd::{AsRawFd, OwnedFd},
};

/// Derive a [`crate::BufferIdentity`] from an fd's `(st_dev, st_ino)`. `dup`
/// preserves the inode, so every process holding the buffer derives the same
/// identity.
#[allow(clippy::unnecessary_cast)]
fn identity_from_stat(stat: &nix::sys::stat::FileStat) -> crate::BufferIdentity {
    let key = ((stat.st_dev as u64) << 32) ^ (stat.st_ino as u64);
    crate::BufferIdentity::derived(crate::IdentityKind::DmaBuf, key)
}

/// A tensor backed by DMA (Direct Memory Access) memory.
///
/// On Linux, a self-allocated (dma_heap) buffer gets a DRM PRIME attachment,
/// with which `DMA_BUF_IOCTL_SYNC` maintains CPU cache coherency. An imported
/// DMA-BUF fd gets no DRM attachment.
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
    /// external allocator). Propagated through `try_clone()`. DRM PRIME import
    /// failures for an imported tensor are logged at DEBUG rather than WARN. It
    /// does not gate CPU mapping.
    #[cfg(target_os = "linux")]
    pub(crate) is_imported: bool,
    /// Which heap the memory came from, as far as this crate knows: CMA,
    /// the system heap, or `Unknown` for an imported fd.
    pub(crate) contiguity: crate::Contiguity,
}

unsafe impl<T> Send for DmaTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
/// Path of the CMA heap, the physically contiguous DMA heap the CPU maps
/// cached.
#[cfg(all(target_os = "linux", not(miri)))]
const CMA_HEAP_PATH: &str = "/dev/dma_heap/linux,cma";

/// Path of the CMA heap the CPU maps uncached (write-combined).
#[cfg(all(target_os = "linux", not(miri)))]
const UNCACHED_CMA_HEAP_PATH: &str = "/dev/dma_heap/linux,cma-uncached";

/// Which heaps a DMA-BUF allocation may come from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct HeapPolicy {
    /// Physically contiguous memory only: the CMA heaps, never the system
    /// heap.
    pub(crate) contiguous: bool,
    /// Prefer the uncached CMA heap. For buffers the CPU never reads.
    pub(crate) uncached: bool,
}

impl HeapPolicy {
    /// The policy for a buffer declared with `access`: uncached when the
    /// declaration excludes CPU reads (`None`, `Write`).
    pub(crate) fn for_access(access: crate::CpuAccess, contiguous: bool) -> Self {
        Self {
            contiguous,
            uncached: !access.reads(),
        }
    }
}

/// Allocate `size` bytes of DMA-BUF: from the uncached CMA heap first when
/// `policy.uncached` and the system has one, then from the CMA heap when there
/// is one, and from the system heap when there is none or the CMA pool cannot
/// fit the buffer. With `policy.contiguous`, from the CMA heaps only. When both
/// heaps fail, the CMA heap's error is the one reported.
#[cfg(all(target_os = "linux", not(miri)))]
fn heap_allocate(
    size: usize,
    policy: HeapPolicy,
) -> Result<(std::os::fd::OwnedFd, crate::Contiguity)> {
    heap_allocate_from(size, policy, UNCACHED_CMA_HEAP_PATH)
}

/// [`heap_allocate`] with the uncached heap at `uncached_path`.
#[cfg(all(target_os = "linux", not(miri)))]
fn heap_allocate_from(
    size: usize,
    policy: HeapPolicy,
    uncached_path: &str,
) -> Result<(std::os::fd::OwnedFd, crate::Contiguity)> {
    use crate::Contiguity;
    let contiguous = policy.contiguous;

    if policy.uncached {
        let uncached = dma_heap::HeapKind::Custom(uncached_path.into());
        match dma_heap::Heap::new(uncached).and_then(|heap| heap.allocate(size)) {
            Ok(fd) => {
                log::debug!("allocated {size} bytes from {uncached_path}");
                return Ok((fd, Contiguity::Contiguous));
            }
            Err(e) => log::debug!(
                "{uncached_path} cannot allocate {size} bytes ({e}); trying the cached heaps"
            ),
        }
    }
    let cma_error = match dma_heap::Heap::new(dma_heap::HeapKind::Cma) {
        Ok(cma) => match cma.allocate(size) {
            Ok(fd) => {
                log::debug!("allocated {size} bytes from {CMA_HEAP_PATH}");
                return Ok((fd, Contiguity::Contiguous));
            }
            Err(e) => Some(e),
        },
        Err(e) if contiguous => {
            return Err(std::io::Error::new(
                e.kind(),
                format!(
                    "physically contiguous memory requires the CMA heap {CMA_HEAP_PATH}, \
                     which cannot be opened: {e}"
                ),
            )
            .into())
        }
        Err(_) => None,
    };
    if contiguous {
        let e = cma_error.expect("the CMA heap opened, so its allocation failed");
        return Err(std::io::Error::new(
            e.kind(),
            format!(
                "the CMA heap {CMA_HEAP_PATH} cannot allocate {size} bytes of physically \
                 contiguous memory: {e}"
            ),
        )
        .into());
    }
    let system = dma_heap::Heap::new(dma_heap::HeapKind::System);
    match (system, cma_error) {
        (Ok(system), None) => {
            let fd = system.allocate(size)?;
            log::debug!("allocated {size} bytes from the system heap");
            Ok((fd, Contiguity::NonContiguous))
        }
        (Ok(system), Some(cma_error)) => match system.allocate(size) {
            Ok(fd) => {
                log::debug!(
                    "CMA heap could not fit {size} bytes ({cma_error}); allocated from the \
                     system heap, which is not physically contiguous"
                );
                Ok((fd, Contiguity::NonContiguous))
            }
            Err(_) => Err(cma_error.into()),
        },
        (Err(e), None) => Err(e.into()),
        (Err(_), Some(cma_error)) => Err(cma_error.into()),
    }
}

unsafe impl<T> Sync for DmaTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> TensorTrait<T> for DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    #[cfg(all(target_os = "linux", not(miri)))]
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

        let (dma_fd, contiguity) = heap_allocate(logical_size, HeapPolicy::default())?;
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
            identity: identity_from_stat(&stat),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
            contiguity,
        })
    }

    #[cfg(not(all(target_os = "linux", not(miri))))]
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "DMA tensor allocation is unavailable (not Linux, or running \
             under Miri, which cannot execute the real ioctls)"
                .to_owned(),
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

        // One `fstat` provides both `buf_size` and the identity.
        let stat = nix::sys::stat::fstat(&fd)?;
        let buf_size = if stat.st_size > 0 && stat.st_size as usize >= logical_size {
            stat.st_size as usize
        } else {
            logical_size
        };

        // An imported DMA-BUF fd gets no DRM attachment.
        #[cfg(target_os = "linux")]
        let drm_attachment = None;

        Ok(DmaTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
            #[cfg(target_os = "linux")]
            _drm_attachment: drm_attachment,
            identity: identity_from_stat(&stat),
            buf_size,
            mmap_offset: 0,
            #[cfg(target_os = "linux")]
            is_imported: true,
            contiguity: crate::Contiguity::Unknown,
        })
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        Ok(self.fd.try_clone()?)
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

    fn map_with<'a>(&self, access: crate::CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_inner(None, access)
    }

    fn buffer_identity(&self) -> &crate::BufferIdentity {
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

    /// Zero-copy sub-region view sharing this buffer's fd and
    /// [`BufferIdentity`](crate::BufferIdentity), positioned at `offset_bytes`
    /// from this tensor's own window with logical `shape`. The view maps
    /// `[abs_offset, abs_offset + shape.product()*size_of::<T>())` of the
    /// shared DMA-BUF.
    ///
    /// Linux only; on other platforms the trait's `NotImplemented` default
    /// applies.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the buffer.
    #[cfg(target_os = "linux")]
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "DmaTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .mmap_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = shape.iter().product::<usize>() * elem;
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > self.buf_size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.buf_size,
            });
        }
        // `try_clone` preserves fd, identity and `buf_size`; set the logical
        // shape and absolute offset for this window.
        let mut v = self.try_clone()?;
        v.shape = shape.to_vec();
        v.mmap_offset = abs_offset;
        Ok(v)
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

/// Keepalive that owns a DMA-BUF map's cache-coherency bracket:
/// `DMA_BUF_SYNC_START` on acquire and the matching `DMA_BUF_SYNC_END` on drop.
pub(crate) struct DmaSyncBracket {
    fd: std::os::fd::OwnedFd,
    access: crate::CpuAccess,
    _mapping: std::sync::Arc<crate::pin::MmapOwner>,
}

impl Drop for DmaSyncBracket {
    fn drop(&mut self) {
        // Same direction as the START sync.
        if let Err(e) = crate::dmabuf::sync_access(&self.fd, false, self.access) {
            log::error!("DMA_BUF_SYNC_END failed: {e}");
        }
    }
}

impl<T> DmaTensor<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug + Send + Sync,
{
    /// Allocate a DMA-BUF with an explicit byte size that may exceed
    /// `shape.product() * sizeof(T)`.
    ///
    /// `shape` stores the logical dimensions `[height, width, channels]`; the
    /// row padding is carried by `Tensor::row_stride` and is visible to the
    /// CPU mapping, which spans the full `byte_size` bytes.
    ///
    /// Errors:
    /// - `InvalidArgument` if `byte_size < shape.product() * sizeof(T)`
    /// - `IoError` if the DMA-heap allocation fails
    #[cfg(all(target_os = "linux", not(edgefirst_tensor_dynamic), not(miri)))]
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
        policy: HeapPolicy,
    ) -> Result<Self> {
        use log::debug;
        use nix::sys::stat::fstat;

        // Logical byte size, with checked arithmetic.
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

        let (dma_fd, contiguity) = heap_allocate(byte_size, policy)?;
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
            identity: identity_from_stat(&stat),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
            contiguity,
        })
    }

    #[cfg(not(all(target_os = "linux", not(miri))))]
    pub(crate) fn new_with_byte_size(
        _shape: &[usize],
        _byte_size: usize,
        _name: Option<&str>,
        _policy: HeapPolicy,
    ) -> Result<Self> {
        Err(Error::NotImplemented(
            "DMA tensor allocation is unavailable (not Linux, or running \
             under Miri, which cannot execute the real ioctls)"
                .to_owned(),
        ))
    }

    /// Map this DMA tensor with an explicit total byte size: the returned view
    /// exposes the full `byte_size` bytes via `as_slice()`/`as_mut_slice()`,
    /// not just the shape-derived logical count.
    #[cfg(not(edgefirst_tensor_dynamic))]
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

    pub fn try_clone(&self) -> Result<Self> {
        let fd = self.clone_fd()?;
        // An imported fd gets no DRM attachment.
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
            contiguity: self.contiguity,
        })
    }
}

impl<T> DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Validate the window a map would expose, before any syscall.
    fn check_map_window(&self, byte_size_override: Option<usize>) -> Result<()> {
        let t_size = std::mem::size_of::<T>();
        let logical_size = self.shape.iter().product::<usize>() * t_size;
        if logical_size == 0 {
            return Err(Error::InvalidSize(0));
        }
        let total_needed = self
            .mmap_offset
            .checked_add(logical_size)
            .ok_or(Error::InvalidSize(0))?;
        if total_needed > self.buf_size {
            return Err(Error::InvalidSize(total_needed));
        }
        if !self.mmap_offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "DMA map: offset {} not aligned to align_of::<T>()={}",
                self.mmap_offset,
                std::mem::align_of::<T>()
            )));
        }
        if let Some(byte_size) = byte_size_override {
            if byte_size == 0 {
                return Err(Error::InvalidSize(0));
            }
            if !byte_size.is_multiple_of(t_size) {
                return Err(Error::InvalidOperation(format!(
                    "DMA map: byte_size {byte_size} is not a multiple of size_of::<T>()={t_size}"
                )));
            }
            if self.mmap_offset.saturating_add(byte_size) > self.buf_size {
                return Err(Error::InvalidSize(byte_size));
            }
        }
        Ok(())
    }

    /// Shared map constructor: [`check_map_window`](Self::check_map_window),
    /// then a sync-bracketed pin.
    fn map_inner<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.check_map_window(byte_size_override)?;
        Ok(crate::view::HostView::new(
            self.scoped_pin(access)?,
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }

    /// Map for CPU access: pin the address and open the coherency window.
    /// Issues `DMA_BUF_SYNC_START` now and the matching `DMA_BUF_SYNC_END` when
    /// the returned pin's keepalive drops.
    pub(crate) fn scoped_pin<'a>(
        &self,
        access: crate::CpuAccess,
    ) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let mmap_size = self.buf_size;
        let _mapping_guard = crate::pin::cpu_mapping_shared();
        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                NonZero::new(mmap_size).ok_or(Error::InvalidSize(mmap_size))?,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &self.fd,
                0,
            )?
        };
        let base = std::ptr::NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?;
        let mapping = std::sync::Arc::new(crate::pin::MmapOwner::new(base, mmap_size));

        let fd = self.fd.try_clone()?;
        crate::dmabuf::sync_access(&fd, true, access).map_err(Error::NixError)?;

        let data = unsafe { mapping.base().add(self.mmap_offset) };
        let len = mmap_size.saturating_sub(self.mmap_offset);
        let keepalive = std::sync::Arc::new(DmaSyncBracket {
            fd,
            access,
            _mapping: mapping,
        });
        Ok(crate::pin::HostPin::new(keepalive, data, len))
    }

    /// Establish a persistent host mapping over this DMA-BUF. Performs no sync
    /// bracketing; callers bracket with
    /// [`Tensor::sync_for_cpu`](crate::Tensor::sync_for_cpu).
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn host_pin<'a>(&self) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let mmap_size = self.buf_size;
        let _mapping_guard = crate::pin::cpu_mapping_shared();
        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                NonZero::new(mmap_size).ok_or(Error::InvalidSize(mmap_size))?,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &self.fd,
                0,
            )?
        };
        let base = std::ptr::NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?;
        let mapping = std::sync::Arc::new(crate::pin::MmapOwner::new(base, mmap_size));

        // The mmap base plus this tensor's offset.
        let data = unsafe { mapping.base().add(self.mmap_offset) };
        // Everything addressable from this tensor's offset; `Tensor::pin_host`
        // narrows to the logical extent.
        let len = mmap_size.saturating_sub(self.mmap_offset);
        Ok(crate::pin::HostPin::new(mapping, data, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuAccess, TensorMapTrait};
    use std::os::fd::FromRawFd;

    /// A fd backed by /dev/null: `fstat` reports `st_size == 0`, and every
    /// window check fires before any fd-specific syscall.
    fn dummy_fd() -> OwnedFd {
        OwnedFd::from(std::fs::File::open("/dev/null").expect("open /dev/null"))
    }

    /// An anonymous memory file of `len` bytes: mmap-able, a real `st_size`,
    /// but not a dma-buf, so sync ioctls on it fail with `ENOTTY`.
    fn memfd(len: usize, fill: impl Fn(usize) -> u8) -> OwnedFd {
        use std::io::Write;
        let raw = unsafe { libc::memfd_create(c"dma-test".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(
            raw >= 0,
            "memfd_create: {}",
            std::io::Error::last_os_error()
        );
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let bytes: Vec<u8> = (0..len).map(fill).collect();
        std::fs::File::from(fd.try_clone().expect("dup"))
            .write_all(&bytes)
            .expect("fill memfd");
        fd
    }

    fn zeroed_stat() -> nix::sys::stat::FileStat {
        // SAFETY: `libc::stat` is plain old data; all-zero is a valid value.
        unsafe { std::mem::zeroed() }
    }

    /// `st_dev` lands in the high half and `st_ino` is XORed in: an inode
    /// whose bit 32 overlaps the device's bit 0 cancels it.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn identity_from_stat_folds_the_device_into_the_high_half() {
        let mut st = zeroed_stat();
        st.st_dev = 1;
        st.st_ino = (1 << 32) | 5;
        let id = identity_from_stat(&st);
        assert_eq!(id.kind(), crate::IdentityKind::DmaBuf);
        assert_eq!(
            id.id(),
            crate::BufferIdentity::derived(crate::IdentityKind::DmaBuf, 5).id()
        );

        st.st_dev = 2;
        st.st_ino = 3;
        assert_eq!(
            identity_from_stat(&st).id(),
            crate::BufferIdentity::derived(crate::IdentityKind::DmaBuf, (2 << 32) | 3).id()
        );
    }

    #[test]
    fn from_fd_records_name_shape_capacity_and_fd() {
        let fd = dummy_fd();
        let raw = fd.as_raw_fd();
        let t = DmaTensor::<u8>::from_fd(fd, &[4096], Some("imported")).expect("import");
        assert_eq!(t.name(), "imported");
        assert_eq!(t.shape(), &[4096]);
        assert_eq!(t.capacity_bytes(), 4096);
        assert_eq!(t.as_raw_fd(), raw);
        assert_eq!(t.memory(), TensorMemory::DmaBuf);
    }

    #[test]
    fn from_fd_rejects_empty_and_zero_sized_shapes() {
        assert!(matches!(
            DmaTensor::<u8>::from_fd(dummy_fd(), &[], None),
            Err(Error::InvalidSize(0))
        ));
        assert!(matches!(
            DmaTensor::<u32>::from_fd(dummy_fd(), &[4, 0], None),
            Err(Error::InvalidSize(0))
        ));
    }

    /// Capacity is the fd's `st_size` when that covers the logical size, and
    /// the logical size otherwise (including kernels reporting 0).
    #[test]
    fn from_fd_capacity_is_the_fstat_size_only_when_it_covers_the_shape() {
        let bigger = DmaTensor::<u32>::from_fd(memfd(8192, |_| 0), &[1024], None).unwrap();
        assert_eq!(bigger.capacity_bytes(), 8192);

        let smaller = DmaTensor::<u32>::from_fd(memfd(1000, |_| 0), &[1024], None).unwrap();
        assert_eq!(smaller.capacity_bytes(), 4096);

        let unsized_fd = DmaTensor::<u32>::from_fd(dummy_fd(), &[1024], None).unwrap();
        assert_eq!(unsized_fd.capacity_bytes(), 4096);
    }

    #[test]
    fn distinct_segments_get_distinct_identities_and_a_dup_keeps_its_own() {
        let a_fd = memfd(64, |_| 0);
        let a = DmaTensor::<u8>::from_fd(a_fd.try_clone().unwrap(), &[64], None).unwrap();
        let again = DmaTensor::<u8>::from_fd(a_fd, &[64], None).unwrap();
        let b = DmaTensor::<u8>::from_fd(memfd(64, |_| 0), &[64], None).unwrap();
        assert_eq!(a.buffer_identity().id(), again.buffer_identity().id());
        assert_ne!(a.buffer_identity().id(), b.buffer_identity().id());
    }

    #[test]
    fn reshape_keeps_the_byte_size_of_a_multi_byte_element() {
        let mut t = DmaTensor::<u32>::from_fd(dummy_fd(), &[1024], None).unwrap();
        t.reshape(&[32, 32]).expect("same element count");
        assert_eq!(t.shape(), &[32, 32]);
        assert!(matches!(t.reshape(&[7]), Err(Error::ShapeMismatch(_))));
        assert!(matches!(t.reshape(&[]), Err(Error::InvalidSize(0))));
        assert_eq!(t.shape(), &[32, 32], "a refused reshape leaves the shape");
    }

    #[test]
    fn set_logical_shape_is_bounded_by_capacity_in_bytes() {
        let mut t = DmaTensor::<u32>::from_fd(dummy_fd(), &[1024], None).unwrap();
        t.set_logical_shape(&[512]).expect("shrink");
        t.set_logical_shape(&[1024]).expect("exact fit");
        match t.set_logical_shape(&[1025]) {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), (4100, 4096));
            }
            other => panic!("expected InsufficientCapacity, got {other:?}"),
        }
        assert_eq!(t.shape(), &[1024]);
        assert!(matches!(
            t.set_logical_shape(&[]),
            Err(Error::InvalidSize(0))
        ));
    }

    #[test]
    fn view_checks_alignment_and_the_byte_window() {
        let t = DmaTensor::<u32>::from_fd(dummy_fd(), &[1024], Some("p")).unwrap();

        let v = t.view(4, &[1023]).expect("exact fit at an aligned offset");
        assert_eq!(v.shape(), &[1023]);
        assert_eq!(v.mmap_offset, 4);
        assert_eq!(v.capacity_bytes(), 4096);
        assert_eq!(v.name(), "p");
        assert_eq!(v.buffer_identity().id(), t.buffer_identity().id());
        let nested = v.view(4, &[1022]).expect("nested view composes");
        assert_eq!(nested.mmap_offset, 8);

        assert!(matches!(t.view(2, &[1]), Err(Error::InvalidOperation(_))));
        match t.view(4, &[1024]) {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), (4100, 4096));
            }
            other => panic!("expected InsufficientCapacity, got {other:?}"),
        }
    }

    #[test]
    fn map_window_checks_for_a_multi_byte_element() {
        let mut t = DmaTensor::<u32>::from_fd(dummy_fd(), &[1024], None).unwrap();
        assert!(t.check_map_window(None).is_ok(), "exact fit");
        assert!(t.check_map_window(Some(4096)).is_ok(), "override exact fit");
        assert!(matches!(
            t.check_map_window(Some(0)),
            Err(Error::InvalidSize(0))
        ));
        assert!(matches!(
            t.check_map_window(Some(6)),
            Err(Error::InvalidOperation(_))
        ));
        assert!(matches!(
            t.check_map_window(Some(4100)),
            Err(Error::InvalidSize(4100))
        ));

        t.shape = vec![1023];
        t.mmap_offset = 4;
        assert!(
            t.check_map_window(None).is_ok(),
            "aligned offset, exact fit"
        );
        assert!(t.check_map_window(Some(4092)).is_ok());
        assert!(matches!(
            t.check_map_window(Some(4096)),
            Err(Error::InvalidSize(4096))
        ));

        t.mmap_offset = 2;
        assert!(matches!(
            t.check_map_window(None),
            Err(Error::InvalidOperation(_))
        ));

        t.mmap_offset = 0;
        t.shape = vec![1024];
        t.buf_size = 1024;
        assert!(matches!(
            t.check_map_window(None),
            Err(Error::InvalidSize(4096))
        ));
    }

    #[test]
    fn map_window_checks_accept_any_offset_for_a_byte_element() {
        let mut t = DmaTensor::<u8>::from_fd(dummy_fd(), &[4095], None).unwrap();
        t.buf_size = 4096;
        t.mmap_offset = 1;
        assert!(t.check_map_window(None).is_ok());
        assert!(t.check_map_window(Some(3)).is_ok());
    }

    /// offset + logical_size exceeds buf_size — must return InvalidSize.
    #[test]
    fn test_dma_map_offset_exceeds_buf_size() {
        let fd = dummy_fd();
        let mut t = DmaTensor::<u8>::from_fd(fd, &[4096], None).expect("import");
        t.buf_size = 4096;
        t.mmap_offset = 4096;
        let result = t.map_inner(None, CpuAccess::ReadWrite);
        match result {
            Err(Error::InvalidSize(n)) => assert_eq!(n, 8192),
            other => panic!("expected InvalidSize(8192), got {other:?}"),
        }
    }

    /// Offset not aligned to align_of::<T>() — must return InvalidOperation.
    #[test]
    fn test_dma_map_misaligned_offset() {
        let fd = dummy_fd();
        let mut t = DmaTensor::<u32>::from_fd(fd, &[1024], None).expect("import");
        t.buf_size = 8192;
        t.mmap_offset = 3;
        let result = t.map_inner(None, CpuAccess::ReadWrite);
        assert!(
            matches!(result, Err(Error::InvalidOperation(_))),
            "expected InvalidOperation for misaligned offset, got {result:?}"
        );
    }

    /// offset + logical_size overflows usize — must return InvalidSize(0).
    #[test]
    fn test_dma_map_offset_overflow() {
        let fd = dummy_fd();
        let mut t = DmaTensor::<u8>::from_fd(fd, &[1], None).expect("import");
        t.buf_size = 4096;
        t.mmap_offset = usize::MAX;
        let result = t.map_inner(None, CpuAccess::ReadWrite);
        match result {
            Err(Error::InvalidSize(n)) => assert_eq!(n, 0),
            other => panic!("expected InvalidSize(0), got {other:?}"),
        }
    }

    /// `host_pin` over a memfd starts at the tensor's offset and spans the rest
    /// of the buffer.
    #[test]
    fn host_pin_of_an_imported_segment_is_offset_adjusted_and_readable() {
        let fd = memfd(8192, |i| (i % 251) as u8);
        let mut t = DmaTensor::<u8>::from_fd(fd, &[8192], None).unwrap();
        t.mmap_offset = 16;
        let pin = t.host_pin().expect("pin");
        assert_eq!(pin.len(), 8192 - 16);
        let bytes = unsafe { pin.as_slice() };
        assert_eq!(bytes[0], 16);
        assert_eq!(bytes[300 - 16], (300 % 251) as u8);
    }

    /// A map over a fd that is not a dma-buf fails on the START sync.
    #[test]
    fn map_of_a_fd_that_is_not_a_dma_buf_fails_on_the_start_sync() {
        let t = DmaTensor::<u8>::from_fd(memfd(4096, |_| 0), &[4096], None).unwrap();
        let before = crate::dmabuf::SYNC_CALLS.with(|c| c.get());
        match t.map_with(CpuAccess::Read) {
            Err(Error::NixError(nix::errno::Errno::ENOTTY)) => {}
            other => panic!("expected NixError(ENOTTY), got {other:?}"),
        }
        let after = crate::dmabuf::SYNC_CALLS.with(|c| c.get());
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (1, 0),
            "a failed START must not be followed by an END"
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn new_with_byte_size_refuses_a_byte_size_below_the_shape() {
        assert!(matches!(
            DmaTensor::<u32>::new_with_byte_size(&[10], 39, None, HeapPolicy::default()),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            DmaTensor::<u8>::new_with_byte_size(&[usize::MAX, 2], 1, None, HeapPolicy::default()),
            Err(Error::InvalidArgument(_))
        ));
    }

    /// A buffer larger than the CMA pool is allocated from the system heap.
    #[test]
    #[cfg(not(miri))]
    fn a_buffer_larger_than_the_cma_pool_comes_from_the_system_heap() {
        const NAME: &str = "a_buffer_larger_than_the_cma_pool_comes_from_the_system_heap";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let heaps = [dma_heap::HeapKind::Cma, dma_heap::HeapKind::System]
            .map(|kind| dma_heap::Heap::new(kind).is_ok());
        if heaps != [true, true] {
            crate::test_support::report_skip(&format!(
                "{NAME} - needs both the CMA and the system heap (CMA, system: {heaps:?})"
            ));
            return;
        }
        let Some(cma_total) = small_cma_pool(NAME) else {
            return;
        };
        let size = cma_total + (4 << 20);
        let t = DmaTensor::<u8>::new(&[size], None)
            .unwrap_or_else(|e| panic!("{size} B, past the {cma_total} B CMA pool: {e}"));
        assert!(fstat_size(&t.fd) >= size);
        assert_eq!(t.contiguity, crate::Contiguity::NonContiguous);
    }

    /// The CMA pool size from `/proc/meminfo`, when it is between 1 and
    /// 256 MB so a test can allocate past it; otherwise reports a skip.
    fn small_cma_pool(name: &str) -> Option<usize> {
        let cma_total = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| {
                m.lines()
                    .find_map(|l| l.strip_prefix("CmaTotal:"))
                    .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<usize>().ok())
            })
            .map_or(0, |kb| kb * 1024);
        if cma_total == 0 || cma_total > 256 << 20 {
            crate::test_support::report_skip(&format!(
                "{name} - CMA pool is {} MB; the test exceeds pools of 1-256 MB",
                cma_total >> 20
            ));
            return None;
        }
        Some(cma_total)
    }

    fn cma_heap_present() -> bool {
        dma_heap::Heap::new(dma_heap::HeapKind::Cma).is_ok()
    }

    const CONTIGUOUS: HeapPolicy = HeapPolicy {
        contiguous: true,
        uncached: false,
    };

    /// The exporter a dma-buf fd reports in `/proc/self/fdinfo` (the heap's
    /// name), or `None` on a kernel that does not report one.
    fn exporter(fd: &OwnedFd) -> Option<String> {
        use std::os::fd::AsRawFd;
        let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.as_raw_fd())).ok()?;
        info.lines()
            .find_map(|l| l.strip_prefix("exp_name:"))
            .map(|n| n.trim().to_owned())
    }

    /// Hardware-only and write-only declarations select the uncached heap;
    /// declarations that read do not.
    #[test]
    fn only_declarations_without_reads_are_uncached() {
        for (access, uncached) in [
            (CpuAccess::None, true),
            (CpuAccess::Write, true),
            (CpuAccess::Read, false),
            (CpuAccess::ReadWrite, false),
        ] {
            assert_eq!(
                HeapPolicy::for_access(access, false),
                HeapPolicy {
                    contiguous: false,
                    uncached
                },
                "{access:?}"
            );
        }
        assert!(HeapPolicy::for_access(CpuAccess::None, true).contiguous);
    }

    /// An uncached request is served from the uncached CMA heap when it can
    /// allocate and from the cached heaps otherwise; a cached request never
    /// comes from the uncached heap.
    #[test]
    #[cfg(not(miri))]
    fn uncached_requests_are_served_and_cached_ones_never_uncached() {
        const NAME: &str = "uncached_requests_are_served_and_cached_ones_never_uncached";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let uncached = HeapPolicy {
            contiguous: false,
            uncached: true,
        };
        let (fd, contiguity) = heap_allocate(4096, uncached).expect("uncached request");
        match exporter(&fd) {
            Some(name) if name.ends_with("cma-uncached") => {
                assert_eq!(contiguity, crate::Contiguity::Contiguous)
            }
            Some(name) => crate::test_support::report_skip(&format!(
                "{NAME} - the uncached heap did not serve the request ({name}), \
                 so only the fallback ran"
            )),
            None => crate::test_support::report_skip(&format!(
                "{NAME} - the kernel names no dma-buf exporter"
            )),
        }
        let (fd, _) = heap_allocate(4096, HeapPolicy::default()).expect("cached request");
        if let Some(name) = exporter(&fd) {
            assert!(!name.ends_with("-uncached"), "exporter {name}");
        }
    }

    /// Without a usable uncached heap an uncached request falls back to the
    /// cached heaps, keeping the contiguous rule.
    #[test]
    #[cfg(not(miri))]
    fn uncached_requests_fall_back_to_the_cached_heaps() {
        const NAME: &str = "uncached_requests_fall_back_to_the_cached_heaps";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let missing = "/dev/dma_heap/no-such-uncached-heap";
        let uncached = HeapPolicy {
            contiguous: false,
            uncached: true,
        };
        let (fd, _) = heap_allocate_from(4096, uncached, missing).expect("fallback");
        if let Some(name) = exporter(&fd) {
            assert!(!name.ends_with("-uncached"), "exporter {name}");
        }
        let contiguous = HeapPolicy {
            contiguous: true,
            uncached: true,
        };
        match heap_allocate_from(4096, contiguous, missing) {
            Ok((fd, contiguity)) => {
                assert!(cma_heap_present(), "contiguous memory without a CMA heap");
                assert_eq!(contiguity, crate::Contiguity::Contiguous);
                if let Some(name) = exporter(&fd) {
                    assert_eq!(name, "linux,cma");
                }
            }
            Err(e) => assert!(
                e.to_string().contains(CMA_HEAP_PATH),
                "error does not name the CMA heap: {e}"
            ),
        }
    }

    /// A contiguous request never falls back to the system heap: past the
    /// CMA pool it fails, and the error names the CMA heap.
    #[test]
    #[cfg(not(miri))]
    fn a_contiguous_buffer_larger_than_the_cma_pool_fails() {
        const NAME: &str = "a_contiguous_buffer_larger_than_the_cma_pool_fails";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        if !cma_heap_present() {
            crate::test_support::report_skip(&format!("{NAME} - no CMA heap"));
            return;
        }
        let Some(cma_total) = small_cma_pool(NAME) else {
            return;
        };
        let size = cma_total + (4 << 20);
        let err = DmaTensor::<u8>::new_with_byte_size(&[size], size, None, CONTIGUOUS)
            .expect_err("a contiguous buffer past the CMA pool must not be allocated");
        assert!(
            err.to_string().contains(CMA_HEAP_PATH),
            "error does not name the CMA heap: {err}"
        );
    }

    /// A contiguous request is served by the CMA heap, or fails naming it.
    #[test]
    #[cfg(not(miri))]
    fn a_contiguous_request_comes_from_the_cma_heap_or_fails() {
        const NAME: &str = "a_contiguous_request_comes_from_the_cma_heap_or_fails";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        match DmaTensor::<u8>::new_with_byte_size(&[4096], 4096, None, CONTIGUOUS) {
            Ok(t) => {
                assert!(cma_heap_present(), "contiguous memory without a CMA heap");
                assert_eq!(t.contiguity, crate::Contiguity::Contiguous);
            }
            Err(e) => assert!(
                e.to_string().contains(CMA_HEAP_PATH),
                "error does not name the CMA heap: {e}"
            ),
        }
    }

    /// Self-allocated memory reports the heap that served it, a clone keeps
    /// it, and the same buffer imported through its fd is `Unknown`.
    #[test]
    #[cfg(not(miri))]
    fn dma_allocations_report_their_heap_and_imports_are_unknown() {
        const NAME: &str = "dma_allocations_report_their_heap_and_imports_are_unknown";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let t = DmaTensor::<u8>::new(&[4096], None).expect("alloc");
        if cma_heap_present() {
            assert_ne!(t.contiguity, crate::Contiguity::Unknown);
        } else {
            assert_eq!(t.contiguity, crate::Contiguity::NonContiguous);
        }
        assert_eq!(t.try_clone().expect("clone").contiguity, t.contiguity);
        let imported = DmaTensor::<u8>::from_fd(t.clone_fd().expect("clone_fd"), &[4096], None)
            .expect("import");
        assert_eq!(imported.contiguity, crate::Contiguity::Unknown);
    }

    /// The public request: `ImageDesc::with_contiguous` through
    /// `Tensor::image_desc`, and its rejection with non-DMA memory.
    #[test]
    #[cfg(not(miri))]
    fn dma_image_desc_contiguous_request() {
        use crate::{DType, ImageDesc, PixelFormat, Tensor, TensorMemory};
        const NAME: &str = "dma_image_desc_contiguous_request";

        let host = ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8)
            .with_memory(Some(TensorMemory::Mem))
            .with_contiguous(true);
        assert!(matches!(
            Tensor::<u8>::image_desc(&host),
            Err(Error::InvalidArgument(_))
        ));

        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let desc = ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8).with_contiguous(true);
        match Tensor::<u8>::image_desc(&desc) {
            Ok(t) => {
                assert!(cma_heap_present(), "contiguous memory without a CMA heap");
                assert_eq!(t.memory(), TensorMemory::DmaBuf);
                assert_eq!(t.contiguity(), crate::Contiguity::Contiguous);
            }
            Err(e) => assert!(e.to_string().contains(CMA_HEAP_PATH), "{e}"),
        }
    }

    fn fstat_size(fd: &OwnedFd) -> usize {
        nix::sys::stat::fstat(fd).expect("fstat").st_size as usize
    }

    /// Capacity reports the page-rounded size the kernel allocated, not the
    /// request.
    #[test]
    #[cfg(not(miri))]
    fn heap_allocation_capacity_is_the_fstat_size() {
        if !crate::test_support::dma_or_skip("heap_allocation_capacity_is_the_fstat_size") {
            return;
        }
        let t = DmaTensor::<u8>::new(&[1000], Some("heap")).expect("alloc");
        assert_eq!(t.name(), "heap");
        assert_eq!(t.capacity_bytes(), fstat_size(&t.fd));
        assert!(t.capacity_bytes() >= 1000);

        let padded =
            DmaTensor::<u32>::new_with_byte_size(&[10, 25], 1000, None, HeapPolicy::default())
                .expect("alloc");
        assert_eq!(padded.shape(), &[10, 25]);
        assert_eq!(padded.capacity_bytes(), fstat_size(&padded.fd));
        assert!(padded.capacity_bytes() > 1000, "page rounding");
        assert!(padded.name().starts_with('/'), "generated name");
    }

    /// Every map issues exactly one START and, when the view drops, exactly
    /// one END.
    #[test]
    #[cfg(not(miri))]
    fn a_map_brackets_with_one_start_and_one_end_sync() {
        if !crate::test_support::dma_or_skip("a_map_brackets_with_one_start_and_one_end_sync") {
            return;
        }
        let t = DmaTensor::<u32>::new(&[256], None).expect("alloc");
        let before = crate::dmabuf::SYNC_CALLS.with(|c| c.get());
        {
            let mut m = t.map_with(CpuAccess::ReadWrite).expect("map");
            assert_eq!(m.as_slice().len(), 256);
            m.as_mut_slice()[255] = 0xDEAD_BEEF;
            let mid = crate::dmabuf::SYNC_CALLS.with(|c| c.get());
            assert_eq!((mid.0 - before.0, mid.1 - before.1), (1, 0));
        }
        let after = crate::dmabuf::SYNC_CALLS.with(|c| c.get());
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 1));

        let wide = t
            .map_with_byte_size(t.capacity_bytes(), CpuAccess::Read)
            .expect("map the whole buffer");
        assert_eq!(wide.as_slice().len(), t.capacity_bytes() / 4);
        assert_eq!(wide.as_slice()[255], 0xDEAD_BEEF);
    }

    #[test]
    fn test_dma_map_with_offset() {
        use crate::{Tensor, TensorMemory, TensorTrait};

        if !crate::test_support::dma_or_skip("test_dma_map_with_offset") {
            return;
        }
        let total_size: usize = 4096 * 4;
        let offset: usize = 4096;
        let data_size: usize = 4096;

        let large_buf = Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None)
            .expect("DMA allocation");

        {
            let mut map = large_buf.map().unwrap();
            map.as_mut_slice().fill(0xAA);
        }

        let fd = large_buf.clone_fd().unwrap();
        let mut offset_tensor = Tensor::<u8>::from_fd(fd, &[data_size], None).unwrap();
        offset_tensor.set_plane_offset(offset);

        let mut map = offset_tensor.map().unwrap();
        let slice = map.as_mut_slice();

        assert_eq!(slice.len(), data_size);
        assert!(
            slice.iter().all(|&b| b == 0xAA),
            "Offset tensor map should see sentinel data at offset"
        );

        slice.fill(0xBB);
        drop(map);

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
