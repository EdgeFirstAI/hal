// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMemory, TensorTrait,
};
use log::{debug, trace};
use nix::{sys::stat::fstat, unistd::ftruncate};
use num_traits::Num;
use std::{
    fmt,
    num::NonZero,
    os::fd::{AsRawFd, OwnedFd},
    ptr::NonNull,
};

/// Derive a [`crate::BufferIdentity`] for a POSIX shm segment from its fd's
/// `(st_dev, st_ino)`, mirroring dma-buf's treatment: it survives `dup` (so
/// `clone_fd`/`from_fd` importing a producer's fd yields the same identity),
/// which is what lets a downstream GL import cache recognize the same buffer
/// handed off between independently-linked copies of this crate.
///
/// That key is only usable where the platform actually populates it. Probed
/// directly on this host (macOS 27, both with and without an intervening
/// `shm_unlink`): `fstat` on a `shm_open` fd always reports `(st_dev,
/// st_ino) = (0, 0)` -- Darwin's POSIX shm objects are not real vnodes with
/// stable inode numbers. Using that pair unconditionally would collapse
/// every live shm segment to ONE identity there, which is the exact defect
/// this derivation exists to prevent. When the stat pair is all-zero, fall
/// back to the raw fd number: it carries no cross-process meaning, but it
/// IS unique among this process's currently-open fds (the same
/// close/reuse hazard as any process-local key -- a pointer, a GL name --
/// mitigated the same way per `IdentityKind`'s doc), which is what a
/// same-process import cache actually needs.
// `st_dev`/`st_ino` are `i32`/`u64` on Darwin but `u64`/`u64` on Linux, so
// exactly one of these casts is a clippy::unnecessary_cast on any given
// platform; both are needed for the expression to compile on both.
#[allow(clippy::unnecessary_cast)]
fn identity_from_stat(fd: &OwnedFd, stat: &nix::sys::stat::FileStat) -> crate::BufferIdentity {
    if stat.st_dev == 0 && stat.st_ino == 0 {
        return crate::BufferIdentity::derived(crate::IdentityKind::HostPtr, fd.as_raw_fd() as u64);
    }
    let key = ((stat.st_dev as u64) << 32) ^ (stat.st_ino as u64);
    crate::BufferIdentity::derived(crate::IdentityKind::Shm, key)
}

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
            log::warn!("Failed to unlink shared memory: {e}");
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
        let stat = fstat(&shm_fd)?;
        let identity = identity_from_stat(&shm_fd, &stat);
        Ok(ShmTensor::<T> {
            name,
            fd: shm_fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len: byte_size,
            _marker: std::marker::PhantomData,
            identity,
        })
    }

    /// Map exposing `byte_size` bytes via `as_slice()` for self-allocated
    /// strided tensors whose rows are padded. The caller (`Tensor::map`)
    /// validates `byte_size <= capacity_bytes()` first.
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

    fn map_inner<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
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
        let base = NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?;
        let owner = std::sync::Arc::new(crate::pin::MmapOwner::new(base, mmap_size));
        let data = unsafe { owner.base().add(self.offset) };
        let len = mmap_size - self.offset;
        Ok(crate::view::HostView::new(
            crate::pin::HostPin::new(owner, data, len),
            self.shape.clone(),
            byte_size_override,
            access,
        ))
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
        let identity = identity_from_stat(&shm_fd, &stat);

        Ok(ShmTensor::<T> {
            name: name.to_owned(),
            fd: shm_fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len: size,
            _marker: std::marker::PhantomData,
            identity,
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

        // One fstat serves two purposes below: byte_len (the true logical
        // length of an externally shared segment is otherwise unknown --
        // exact on Linux, page-rounded on macOS) and this tensor's identity,
        // mirroring dma-buf's from_fd.
        let stat = fstat(&fd)?;
        let byte_len = stat.st_size as usize;
        let identity = identity_from_stat(&fd, &stat);

        Ok(ShmTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            offset: 0,
            byte_len,
            _marker: std::marker::PhantomData,
            identity,
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

impl<T> ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Establish a persistent host mapping over this segment.
    ///
    /// Unlike [`map_with`](TensorTrait::map_with) the mapping is owned by the
    /// returned pin rather than by a guard, so the address stays valid for as
    /// long as the pin lives. SHM is plain CPU memory, so there is no sync to
    /// bracket — [`Tensor::sync_for_cpu`](crate::Tensor::sync_for_cpu) is a
    /// documented no-op here.
    pub(crate) fn host_pin<'a>(&self) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        // Map the whole segment from fd offset 0 and apply `self.offset`
        // afterwards: mmap cannot take a non-page-aligned fd offset, and this
        // mirrors what map_with already does.
        let mmap_size = self.capacity_bytes();
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
        let base = NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(mmap_size))?;
        let owner = std::sync::Arc::new(crate::pin::MmapOwner::new(base, mmap_size));

        // Offset-adjusted, so a sub-region view hands back its own window
        // rather than the raw mmap base.
        if self.offset > mmap_size {
            return Err(Error::InsufficientCapacity {
                needed: self.offset,
                capacity: mmap_size,
            });
        }
        let data = unsafe { owner.base().add(self.offset) };
        // Everything addressable from this tensor's offset; Tensor::pin_host
        // narrows to the logical extent.
        let len = mmap_size - self.offset;
        Ok(crate::pin::HostPin::new(owner, data, len))
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

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;
    use crate::{TensorMemory, TensorTrait};

    /// Android's bionic libc has no POSIX shared memory, so `ShmTensor::new`
    /// reports `NotImplemented` there (see the module docs). These tests
    /// allocate, so they cannot run on that platform — skip loudly rather
    /// than unwrapping a documented failure.
    ///
    /// Probing beats `cfg(target_os = "android")`: it also covers a host where
    /// /dev/shm is absent or unwritable, which looks identical from here.
    fn shm_or_skip(what: &str) -> bool {
        if crate::is_shm_available() {
            return true;
        }
        log::warn!("SKIPPED: {what} - SHM allocation unavailable on this platform");
        false
    }

    #[test]
    fn test_new_valid_shape() {
        if !shm_or_skip("test_new_valid_shape") {
            return;
        }
        let tensor = ShmTensor::<u8>::new(&[2, 3, 4], None).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.memory(), TensorMemory::Shm);
        assert_eq!(tensor.len(), 24);
        assert_eq!(tensor.size(), 24);
    }

    #[test]
    fn test_map_read_write() {
        if !shm_or_skip("test_map_read_write") {
            return;
        }
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
        if !shm_or_skip("test_from_fd_roundtrip") {
            return;
        }
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
        if !shm_or_skip("test_reshape") {
            return;
        }
        let mut tensor = ShmTensor::<u8>::new(&[3, 4], None).unwrap();
        tensor.reshape(&[12]).unwrap();
        assert_eq!(tensor.shape(), &[12]);
        assert_eq!(tensor.len(), 12);

        // Incompatible reshape should fail.
        let result = tensor.reshape(&[7]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ShapeMismatch(_)));
    }

    #[test]
    fn two_shm_tensors_do_not_share_an_identity() {
        if !shm_or_skip("two_shm_tensors_do_not_share_an_identity") {
            return;
        }
        let a = ShmTensor::<u8>::new(&[8], None).unwrap();
        let b = ShmTensor::<u8>::new(&[8], None).unwrap();
        // `Shm` where the platform's fstat gives a real inode (Linux); the
        // `HostPtr`-tagged fd-number fallback where it does not (confirmed
        // on macOS -- see `identity_from_stat`'s doc). Either way two
        // distinct segments must not collide.
        assert!(matches!(
            a.buffer_identity().kind(),
            crate::IdentityKind::Shm | crate::IdentityKind::HostPtr
        ));
        assert_ne!(a.buffer_identity().id(), b.buffer_identity().id());
    }

    #[test]
    fn a_dup_of_the_same_shm_segment_has_the_same_identity() {
        if !shm_or_skip("a_dup_of_the_same_shm_segment_has_the_same_identity") {
            return;
        }
        // `from_fd` on a dup'd fd is what a cross-library import does. If dup
        // changed the identity, every such import would miss the GL cache --
        // the measured blocker this derivation exists to fix (mirrors the
        // dma-buf test in `crates/tensor/tests/identity.rs`). This property
        // only holds where the identity is keyed on the real `(st_dev,
        // st_ino)` -- `dup` preserves the inode, but not the fd number the
        // `HostPtr` fallback keys on where the platform's fstat gives no
        // usable inode (see `identity_from_stat`'s doc). Skip loudly rather
        // than asserting a property this platform cannot provide.
        let a = ShmTensor::<u8>::new(&[8], None).unwrap();
        if a.buffer_identity().kind() != crate::IdentityKind::Shm {
            println!(
                "SKIP: a_dup_of_the_same_shm_segment_has_the_same_identity - \
                 this platform's fstat gives no usable shm inode, so identity \
                 falls back to the fd number (see identity_from_stat)"
            );
            return;
        }
        let dup_fd = a.clone_fd().unwrap();
        let dup = ShmTensor::<u8>::from_fd(dup_fd, &[8], None).unwrap();
        assert_eq!(a.buffer_identity().id(), dup.buffer_identity().id());
    }
}
