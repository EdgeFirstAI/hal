// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
};
use log::{debug, trace, warn};
use nix::{sys::stat::fstat, unistd::ftruncate};
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
#[derive(Debug)]
pub struct ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub fd: OwnedFd,
    pub shape: Vec<usize>,
    /// Byte offset into the shared segment where this tensor's logical window
    /// begins. `0` for whole-segment tensors; non-zero for `view()` sub-regions
    /// (mirrors `DmaTensor::mmap_offset` / `MemTensor::offset`). Applied in
    /// `ShmMap::as_slice`, which maps the whole segment and indexes from here.
    offset: usize,
    /// Logical byte length of the backing segment (the size requested at
    /// `ftruncate`), as opposed to the physical `fstat().st_size` exposed by
    /// [`capacity_bytes`](Self::capacity_bytes). These diverge on macOS, where
    /// POSIX shm segments are rounded up to a page, so bounds checks must use
    /// this logical length to stay platform-consistent. Inherited unchanged by
    /// `view()` sub-regions (they share the same segment).
    byte_len: usize,
    pub _marker: std::marker::PhantomData<T>,
    identity: crate::BufferIdentity,
}

unsafe impl<T> Send for ShmTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for ShmTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate the anonymous POSIX shared-memory segment backing a new
    /// tensor: `shm_open` with a random name, then immediately `shm_unlink`
    /// so nothing outlives the process — the library's sharing model is
    /// file descriptors (`clone_fd`), never names.
    ///
    /// Android's bionic libc has no POSIX shared memory (`shm_open` /
    /// `shm_unlink` do not exist), so allocation reports `NotImplemented`
    /// there. Receiving a segment created elsewhere still works on Android:
    /// [`from_fd`](TensorTrait::from_fd) needs only `mmap`/`fstat`, which
    /// bionic provides. (A `memfd_create`-backed allocator is the planned
    /// replacement: the syscall exists on all Android kernels, but the
    /// bionic wrapper appears at API 30 and the HAL floor is 26.)
    #[cfg(not(target_os = "android"))]
    fn alloc_anon_fd(name: &str) -> Result<OwnedFd> {
        use nix::fcntl::OFlag;
        let shm_fd = nix::sys::mman::shm_open(
            name,
            OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_RDWR,
            nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
        )?;
        if let Err(e) = nix::sys::mman::shm_unlink(name) {
            warn!("Failed to unlink shared memory: {e}");
        }
        Ok(shm_fd)
    }

    #[cfg(target_os = "android")]
    fn alloc_anon_fd(_name: &str) -> Result<OwnedFd> {
        Err(Error::NotImplemented(
            "TensorMemory::Shm allocation is not available on Android (bionic has no \
             POSIX shm_open); import an existing segment via from_fd instead"
                .to_owned(),
        ))
    }

    /// Create a shared-memory tensor with a logical `shape` but a physical
    /// allocation of `byte_size` bytes (which must be `>= shape.product() *
    /// sizeof(T)`).  Used for image tensors with a 64-byte-aligned row stride
    /// that exceeds the logical shape product.
    pub(crate) fn new_with_byte_size(
        shape: &[usize],
        byte_size: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        let logical = shape.iter().product::<usize>() * elem;
        if byte_size < logical {
            return Err(Error::InsufficientCapacity {
                needed: logical,
                capacity: byte_size,
            });
        }
        let name = match name {
            Some(n) => n.to_owned(),
            None => {
                let uuid = uuid::Uuid::new_v4().as_simple().to_string();
                format!("/{}", &uuid[..16])
            }
        };
        let shm_fd = Self::alloc_anon_fd(name.as_str())?;
        ftruncate(&shm_fd, byte_size as i64)?;
        Ok(ShmTensor::<T> {
            name,
            fd: shm_fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len: byte_size,
            _marker: std::marker::PhantomData,
            identity: crate::BufferIdentity::new(),
        })
    }

    /// Map exposing `byte_size` bytes via `as_slice()` for self-allocated
    /// strided tensors whose rows are padded. The caller (`Tensor::map`)
    /// validates `byte_size <= capacity_bytes()` first.
    pub(crate) fn map_with_byte_size(
        &self,
        byte_size: usize,
        access: crate::CpuAccess,
    ) -> Result<TensorMap<T>> {
        self.map_inner(Some(byte_size), access)
    }

    fn map_inner(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<TensorMap<T>> {
        let exposed = byte_size_override.unwrap_or_else(|| self.size());
        // Map the whole segment from fd offset 0 and apply `self.offset` in
        // `ShmMap::as_slice` — mmap cannot take a non-page-aligned fd offset,
        // and the segment is small (mirrors `DmaMap`, which maps `buf_size`).
        let mmap_size = self.capacity_bytes();
        let end = self
            .offset
            .checked_add(exposed)
            .ok_or(Error::InvalidSize(exposed))?;
        if end > mmap_size {
            return Err(Error::InsufficientCapacity {
                needed: end,
                capacity: mmap_size,
            });
        }
        if std::mem::size_of::<T>() > 1 && !self.offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "ShmMap: offset {} not aligned to align_of::<T>()={}",
                self.offset,
                std::mem::align_of::<T>()
            )));
        }
        let size = NonZero::new(mmap_size).ok_or(Error::InvalidSize(mmap_size))?;
        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                size,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &self.fd,
                0,
            )?
        };

        trace!("Mapping shared memory: {ptr:?}");
        let shm_ptr = ShmPtr(NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?);
        Ok(TensorMap::Shm(ShmMap {
            ptr: Arc::new(Mutex::new(shm_ptr)),
            shape: self.shape.clone(),
            offset: self.offset,
            mmap_size,
            byte_size_override,
            writable: access.writes(),
            _marker: std::marker::PhantomData,
        }))
    }
}

impl<T> TensorTrait<T> for ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let name = match name {
            Some(name) => name.to_owned(),
            None => {
                let uuid = uuid::Uuid::new_v4().as_simple().to_string();
                format!("/{}", &uuid[..16])
            }
        };

        let shm_fd = Self::alloc_anon_fd(name.as_str())?;

        trace!("Creating shared memory: {name}");

        ftruncate(&shm_fd, size as i64)?;
        let stat = fstat(&shm_fd)?;
        debug!("Shared memory stat: {stat:?}");

        Ok(ShmTensor::<T> {
            name: name.to_owned(),
            fd: shm_fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len: size,
            _marker: std::marker::PhantomData,
            identity: crate::BufferIdentity::new(),
        })
    }

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        // The true logical length of an externally shared segment is unknown
        // here; `fstat` is the only signal (exact on Linux, page-rounded on
        // macOS). Fall back to the declared `size` when it is unavailable.
        let byte_len = fstat(&fd).map(|s| s.st_size as usize).unwrap_or(size);

        Ok(ShmTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len,
            _marker: std::marker::PhantomData,
            identity: crate::BufferIdentity::new(),
        })
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        Ok(self.fd.try_clone()?)
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Shm
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

    fn map_with(&self, access: crate::CpuAccess) -> Result<TensorMap<T>> {
        self.map_inner(None, access)
    }

    fn buffer_identity(&self) -> &crate::BufferIdentity {
        &self.identity
    }

    fn capacity_bytes(&self) -> usize {
        fstat(&self.fd)
            .map(|s| s.st_size as usize)
            .unwrap_or_else(|_| self.size())
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let capacity = self.capacity_bytes();
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    /// Zero-copy sub-region view sharing this tensor's segment via a cloned fd
    /// (the SHM sharing model is fd-based, like `clone_fd`/`from_fd`) and its
    /// [`BufferIdentity`](crate::BufferIdentity).
    ///
    /// The view maps `[offset_bytes, offset_bytes + logical_size)` measured from
    /// this tensor's own window (`logical_size = shape.product() *
    /// size_of::<T>()`), so a sub-view of a sub-view composes. N such views into
    /// one parent share the segment (no copy) and write independently.
    ///
    /// # Errors
    /// - [`Error::InvalidOperation`] if `offset_bytes` is not aligned to
    ///   `align_of::<T>()` (required for the `ShmMap` pointer cast).
    /// - [`Error::InsufficientCapacity`] if the window exceeds the segment.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "ShmTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = shape.iter().product::<usize>() * elem;
        // Bound against the segment's *logical* byte length, not the physical
        // `capacity_bytes()` (`st_size`): the latter is page-rounded on macOS,
        // which would let an out-of-bounds window slip through (see `byte_len`).
        let capacity = self.byte_len;
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        Ok(ShmTensor {
            name: self.name.clone(),
            fd: self.fd.try_clone()?,
            shape: shape.to_vec(),
            offset: abs_offset,
            byte_len: self.byte_len,
            _marker: std::marker::PhantomData,
            // A sub-view is the *same* segment: share the parent's identity so
            // identity-keyed logic treats the windows as one buffer at distinct
            // offsets, not unrelated allocations.
            identity: self.identity.clone(),
        })
    }
}

impl<T> AsRawFd for ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.fd.as_raw_fd()
    }
}

#[derive(Debug)]
pub struct ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    ptr: Arc<Mutex<ShmPtr>>,
    shape: Vec<usize>,
    /// Byte offset into the mmap'd segment where this map's window begins
    /// (non-zero for sub-region views). `as_slice()` returns `base + offset`.
    offset: usize,
    /// Bytes actually mmap'd (the whole segment). `unmap()` munmaps exactly
    /// this, independent of the logical `offset`/`len()` window.
    mmap_size: usize,
    /// When `Some(bytes)`, `as_slice()` exposes `bytes / sizeof(T)` elements
    /// (the full padded window) instead of `shape.product()`.
    byte_size_override: Option<usize>,
    /// Whether mutable access is permitted (`map_read()` maps are not).
    /// The mmap itself stays PROT_READ|PROT_WRITE (protection narrowing is
    /// a follow-up); this enforces the API contract uniformly.
    writable: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Deref for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[derive(Debug)]
struct ShmPtr(NonNull<c_void>);
impl Deref for ShmPtr {
    type Target = NonNull<c_void>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for ShmPtr {}

impl<T> TensorMapTrait<T> for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        match self.byte_size_override {
            Some(bytes) => bytes / std::mem::size_of::<T>(),
            None => self.shape.iter().product(),
        }
    }

    fn unmap(&mut self) {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        // Munmap the whole mmap'd segment (`mmap_size`), not the logical window
        // — `offset` only shifts where `as_slice()` reads, not the mapping.
        let err = unsafe { nix::sys::mman::munmap(**ptr, self.mmap_size) };
        if let Err(e) = err {
            warn!("Failed to unmap shared memory: {e}");
        }
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        let base = unsafe { (ptr.as_ptr() as *const u8).add(self.offset) as *const T };
        unsafe { std::slice::from_raw_parts(base, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        crate::assert_map_writable(self.writable, "Shm");
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        let base = unsafe { (ptr.as_ptr() as *mut u8).add(self.offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(base, self.len()) }
    }
}

impl<T> Drop for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        trace!("ShmMap dropped, unmapping memory");
        self.unmap();
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use crate::{TensorMapTrait, TensorMemory, TensorTrait};

    #[test]
    fn test_new_valid_shape() {
        let tensor = ShmTensor::<u8>::new(&[2, 3, 4], None).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.memory(), TensorMemory::Shm);
        assert_eq!(tensor.len(), 24);
        assert_eq!(tensor.size(), 24);
    }

    #[test]
    fn test_map_read_write() {
        let tensor = ShmTensor::<u8>::new(&[4, 4], None).unwrap();
        let mut map = tensor.map().unwrap();
        map.as_mut_slice()[0] = 10;
        map.as_mut_slice()[5] = 20;
        assert_eq!(map.as_slice()[0], 10);
        assert_eq!(map.as_slice()[5], 20);
        assert_eq!(map.as_slice()[1], 0);
    }

    #[test]
    fn test_from_fd_roundtrip() {
        // Create tensor A and write data into it.
        let tensor_a = ShmTensor::<u8>::new(&[2, 4], None).unwrap();
        {
            let mut map_a = tensor_a.map().unwrap();
            map_a.as_mut_slice()[0] = 0xAB;
            map_a.as_mut_slice()[7] = 0xCD;
        }

        // Clone A's fd and create tensor B from it.
        let fd = tensor_a.clone_fd().unwrap();
        let tensor_b = ShmTensor::<u8>::from_fd(fd, &[2, 4], Some("clone")).unwrap();

        // Verify B sees the same data (shared memory).
        let map_b = tensor_b.map().unwrap();
        assert_eq!(map_b.as_slice()[0], 0xAB);
        assert_eq!(map_b.as_slice()[7], 0xCD);
    }

    #[test]
    fn test_reshape() {
        let mut tensor = ShmTensor::<u8>::new(&[3, 4], None).unwrap();
        tensor.reshape(&[12]).unwrap();
        assert_eq!(tensor.shape(), &[12]);
        assert_eq!(tensor.len(), 12);

        // Incompatible reshape should fail.
        let result = tensor.reshape(&[7]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ShapeMismatch(_)));
    }
}
