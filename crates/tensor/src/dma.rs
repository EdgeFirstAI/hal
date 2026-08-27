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

/// Derive a [`crate::BufferIdentity`] from an fd's `(st_dev, st_ino)`.
///
/// `dup` (and therefore `from_fd` importing a producer's cloned fd) preserves
/// the inode, so this yields the same identity in every process that holds
/// the buffer -- the property a downstream GL import cache keys on. dma-buf
/// inodes all live on one anonymous mount, so `st_ino` alone would already be
/// unique; folding in `st_dev` costs nothing and drops the assumption that
/// stays true.
// `st_dev`/`st_ino` are `i32`/`u64` on Darwin but `u64`/`u64` on Linux, so
// exactly one of these casts is a clippy::unnecessary_cast on any given
// platform; both are needed for the expression to compile on both. Note this
// file is `target_os = "linux"`-gated, so a macOS `cargo clippy` never sees it
// at all -- the Linux lane is the only one that catches a lint here.
#[allow(clippy::unnecessary_cast)]
fn identity_from_stat(stat: &nix::sys::stat::FileStat) -> crate::BufferIdentity {
    let key = ((stat.st_dev as u64) << 32) ^ (stat.st_ino as u64);
    crate::BufferIdentity::derived(crate::IdentityKind::DmaBuf, key)
}

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
    // `dma-heap` is excluded from the dependency graph under Miri (see
    // Cargo.toml's `[target.'cfg(all(target_os = "linux", not(miri)))'.
    // dependencies]`): `dma-heap 0.4.1` fails to type-check at all under
    // Miri (its hardcoded `u32` ioctl opcode constant conflicts with
    // `rustix`'s Miri-forced libc backend, whose `Opcode` is `u64` on this
    // target), so this real-allocation branch must agree -- every normal
    // (non-Miri) build is unaffected, `not(miri)` is only ever true when
    // `cargo miri` is doing the compiling.
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
            identity: identity_from_stat(&stat),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
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

        // One fstat serves two purposes below: buf_size (with a fallback for
        // kernels that report st_size=0 on DMA-BUF fds) and, further down,
        // this tensor's identity. Calling it twice risked the two derived
        // values disagreeing if the fd's backing changed between calls.
        let stat = nix::sys::stat::fstat(&fd)?;
        let buf_size = if stat.st_size > 0 && stat.st_size as usize >= logical_size {
            stat.st_size as usize
        } else {
            logical_size
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
            // dup() preserves (st_dev, st_ino), so an imported tensor gets
            // the same identity as the producer's -- what lets a GL import
            // cache hit across a library boundary instead of missing every
            // frame on a fresh counter value.
            identity: identity_from_stat(&stat),
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
    /// `[abs_offset, abs_offset + shape.product()*size_of::<T>())` of the shared
    /// DMA-BUF.
    ///
    /// DMA-BUF backing is Linux-only; on other platforms `DmaTensor` falls back
    /// to the trait's `NotImplemented` default (the type only exists as a stub).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the buffer.
    #[cfg(target_os = "linux")]
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        // Alignment depends on `align_of::<T>()`, not element size (a
        // `size_of == 1`, `align_of > 1` type would otherwise skip the check).
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
        // try_clone preserves fd, identity and buf_size; override the logical
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

/// Keepalive that owns a DMA-BUF map's **cache-coherency bracket**.
///
/// `host_pin` deliberately performs no sync — that is the whole point of
/// separating a pin's lifetime from its coherency window. A *map* is the
/// opposite: it promises coherent CPU access for its duration, so it issues
/// `DMA_BUF_SYNC_START` on acquire and the matching `..._END` on drop, exactly
/// as the deleted `DmaMap` did.
///
/// Getting this wrong does not crash: a missing invalidate yields
/// stale-but-plausible pixels, which is how it reached hardware unnoticed and
/// showed up as accuracy deltas rather than failures.
pub(crate) struct DmaSyncBracket {
    fd: std::os::fd::OwnedFd,
    access: crate::CpuAccess,
    _mapping: std::sync::Arc<crate::pin::MmapOwner>,
}

impl Drop for DmaSyncBracket {
    fn drop(&mut self) {
        // Direction must match the START, or the kernel skips the writeback.
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
    // `static`-only: called from `Tensor::image_with_stride` (`lib.rs`,
    // `impl<T> Tensor<T>`), which is itself `#[cfg(feature = "static")]`.
    // `dynamic`'s `image_with_stride` drives `ef_tensor_image_with_stride_alloc`
    // instead, never this method directly.
    #[cfg(all(target_os = "linux", feature = "static", not(miri)))]
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
            identity: identity_from_stat(&stat),
            buf_size,
            mmap_offset: 0,
            is_imported: false,
        })
    }

    #[cfg(not(all(target_os = "linux", not(miri))))]
    pub(crate) fn new_with_byte_size(
        _shape: &[usize],
        _byte_size: usize,
        _name: Option<&str>,
    ) -> Result<Self> {
        Err(Error::NotImplemented(
            "DMA tensor allocation is unavailable (not Linux, or running \
             under Miri, which cannot execute the real ioctls)"
                .to_owned(),
        ))
    }

    /// Map this DMA tensor with an explicit total byte size.
    ///
    /// Used by `Tensor::map()` for self-allocated strided tensors — the
    /// returned view exposes the full `byte_size` bytes via
    /// `as_slice()`/`as_mut_slice()`, not just the shape-derived logical
    /// count. Callers are expected to iterate rows with
    /// `Tensor::effective_row_stride()` so they don't read past the end.
    ///
    /// `static`-only: the sole caller, `impl<T> TensorMapTrait<T> for
    /// Tensor<T>` in `lib.rs`, is itself `#[cfg(feature = "static")]`.
    #[cfg(feature = "static")]
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

impl<T> DmaTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Shared map constructor.
    ///
    /// Carries forward every check the deleted `DmaMap::new_internal`
    /// performed, with the same error variants — these are asserted by the
    /// tests below, and a mapping that slipped past them would SIGBUS on
    /// access rather than fail cleanly.
    fn map_inner<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
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
        if t_size > 1 && !self.mmap_offset.is_multiple_of(std::mem::align_of::<T>()) {
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
            if t_size > 1 && !byte_size.is_multiple_of(t_size) {
                return Err(Error::InvalidOperation(format!(
                    "DMA map: byte_size {byte_size} is not a multiple of size_of::<T>()={t_size}"
                )));
            }
            if self.mmap_offset.saturating_add(byte_size) > self.buf_size {
                return Err(Error::InvalidSize(byte_size));
            }
        }
        Ok(crate::view::HostView::new(
            self.scoped_pin(access)?,
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }

    /// Map for CPU access: pin the address **and** open the coherency window.
    ///
    /// This is what `map_with` uses. Unlike [`host_pin`](Self::host_pin) it
    /// issues `DMA_BUF_SYNC_START` now and the matching `..._END` when the
    /// returned pin's keepalive drops, so the bracket lasts exactly as long as
    /// the view does.
    pub(crate) fn scoped_pin<'a>(
        &self,
        access: crate::CpuAccess,
    ) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let mmap_size = self.buf_size;
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

    /// Establish a persistent host mapping over this DMA-BUF.
    ///
    /// Unlike [`map_with`](TensorTrait::map_with) this performs **no** sync
    /// bracketing: the mapping's lifetime and the coherency window are now
    /// separate concerns, which is the whole point. Callers bracket with
    /// [`Tensor::sync_for_cpu`](crate::Tensor::sync_for_cpu).
    ///
    /// `static`-only: called only from `TensorStorage::pin_host`
    /// (`lib.rs`), itself `#[cfg(feature = "static")]`.
    #[cfg(feature = "static")]
    pub(crate) fn host_pin<'a>(&self) -> crate::Result<crate::pin::HostPin<'a>>
    where
        T: 'a,
    {
        let mmap_size = self.buf_size;
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

        // Offset-adjusted, so a strided or sub-region tensor does not hand back
        // the raw mmap base -- requirement 2 of issue #134.
        let data = unsafe { mapping.base().add(self.mmap_offset) };
        // Everything addressable from this tensor's offset. Tensor::pin_host
        // narrows to the logical extent; a map guard keeps this wider window
        // so it can expose stride-padded rows.
        let len = mmap_size.saturating_sub(self.mmap_offset);
        Ok(crate::pin::HostPin::new(mapping, data, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;

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
        let mut t = DmaTensor::<u8>::from_fd(fd, &[4096], None).expect("import");
        t.buf_size = 4096;
        t.mmap_offset = 4096;
        let result = t.map_inner(None, crate::CpuAccess::ReadWrite);
        match result {
            Err(Error::InvalidSize(n)) => assert_eq!(n, 8192),
            other => panic!("expected InvalidSize(8192), got {other:?}"),
        }
    }

    /// Offset not aligned to align_of::<T>() — must return InvalidOperation.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_misaligned_offset() {
        let fd = dummy_fd();
        let mut t = DmaTensor::<u32>::from_fd(fd, &[1024], None).expect("import");
        t.buf_size = 8192;
        t.mmap_offset = 3;
        let result = t.map_inner(None, crate::CpuAccess::ReadWrite);
        assert!(
            matches!(result, Err(Error::InvalidOperation(_))),
            "expected InvalidOperation for misaligned offset, got {result:?}"
        );
    }

    /// offset + logical_size overflows usize — must return InvalidSize(0).
    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_offset_overflow() {
        let fd = dummy_fd();
        let mut t = DmaTensor::<u8>::from_fd(fd, &[1], None).expect("import");
        t.buf_size = 4096;
        t.mmap_offset = usize::MAX;
        let result = t.map_inner(None, crate::CpuAccess::ReadWrite);
        match result {
            Err(Error::InvalidSize(n)) => assert_eq!(n, 0),
            other => panic!("expected InvalidSize(0), got {other:?}"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_map_with_offset() {
        use crate::{Tensor, TensorMemory, TensorTrait};

        // Skip if DMA heap not available
        let total_size: usize = 4096 * 4; // 16KB
        let offset: usize = 4096; // 4KB offset
        let data_size: usize = 4096; // 4KB of data after offset

        let large_buf = match Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None) {
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
