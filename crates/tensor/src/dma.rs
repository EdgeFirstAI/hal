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
    /// Validate the window a map would expose, before any syscall.
    ///
    /// A mapping that slipped past these checks would SIGBUS on access rather
    /// than fail cleanly. The alignment and multiple-of checks are no-ops for
    /// single-byte `T` (`align_of` and `size_of` are both 1).
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

    /// `host_pin` needs no sync, so it works over any mmap-able fd: the pin
    /// starts at the tensor's offset and spans the rest of the buffer.
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

    /// A map opens a coherency bracket; over a fd that is not a dma-buf the
    /// START sync fails, and the map must report it rather than hand back an
    /// unbracketed view.
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
            DmaTensor::<u32>::new_with_byte_size(&[10], 39, None),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            DmaTensor::<u8>::new_with_byte_size(&[usize::MAX, 2], 1, None),
            Err(Error::InvalidArgument(_))
        ));
    }

    fn fstat_size(fd: &OwnedFd) -> usize {
        nix::sys::stat::fstat(fd).expect("fstat").st_size as usize
    }

    /// Heap allocations are page-rounded; capacity must report what the
    /// kernel actually allocated, not the request.
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

        let padded = DmaTensor::<u32>::new_with_byte_size(&[10, 25], 1000, None).expect("alloc");
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
        let total_size: usize = 4096 * 4; // 16KB
        let offset: usize = 4096; // 4KB offset
        let data_size: usize = 4096; // 4KB of data after offset

        let large_buf = Tensor::<u8>::new(&[total_size], Some(TensorMemory::DmaBuf), None)
            .expect("DMA allocation");

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
