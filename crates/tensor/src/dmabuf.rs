// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
use nix::{ioctl_read, ioctl_write_ptr};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::OnceLock;

const DMA_BUF_BASE: u8 = b'b';
const DMA_BUF_IOCTL_SYNC: u8 = 0;
const DMA_BUF_IOCTL_PHYS: u8 = 10;

const DMA_BUF_SYNC_READ: u64 = 1 << 0;
const DMA_BUF_SYNC_WRITE: u64 = 1 << 1;
const DMA_BUF_SYNC_START: u64 = 0 << 2;
const DMA_BUF_SYNC_END: u64 = 1 << 2;

#[derive(Default)]
#[repr(C)]
struct DmaBufSync {
    flags: u64,
}

ioctl_write_ptr!(
    ioctl_dma_buf_sync,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_SYNC,
    DmaBufSync
);

ioctl_read!(
    ioctl_dma_buf_phys,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_PHYS,
    std::ffi::c_ulong
);

fn sync(fd: &OwnedFd, flags: u64) -> nix::Result<()> {
    let sync = DmaBufSync { flags };
    unsafe { ioctl_dma_buf_sync(fd.as_raw_fd(), &sync) }?;
    Ok(())
}

pub(crate) fn start_read(fd: &OwnedFd) -> nix::Result<()> {
    sync(fd, DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START)
}

pub(crate) fn end_read(fd: &OwnedFd) -> nix::Result<()> {
    sync(fd, DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END)
}

pub(crate) fn start_write(fd: &OwnedFd) -> nix::Result<()> {
    sync(fd, DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_START)
}

pub(crate) fn end_write(fd: &OwnedFd) -> nix::Result<()> {
    sync(fd, DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_END)
}

pub(crate) fn start_readwrite(fd: &OwnedFd) -> nix::Result<()> {
    sync(
        fd,
        DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_START,
    )
}

pub(crate) fn end_readwrite(fd: &OwnedFd) -> nix::Result<()> {
    sync(
        fd,
        DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_END,
    )
}

pub(crate) fn phys(fd: &OwnedFd) -> nix::Result<u64> {
    let mut phys: u64 = 0;
    unsafe { ioctl_dma_buf_phys(fd.as_raw_fd(), &mut phys) }?;
    Ok(phys)
}

// =============================================================================
// DRM PRIME import — creates persistent dma_buf_attach for cache maintenance
// =============================================================================
//
// The CMA heap's begin_cpu_access iterates over buffer->attachments to perform
// cache maintenance via dma_sync_sgtable_for_cpu(). Without any active
// attachments, DMA_BUF_IOCTL_SYNC is a no-op on cached CMA heaps.
//
// By importing the DMA-buf fd through the DRM/GPU driver
// (DRM_IOCTL_PRIME_FD_TO_HANDLE), a persistent dma_buf_attach() is created.
// This makes DMA_BUF_IOCTL_SYNC actually perform cache invalidation/flush.

const DRM_IOCTL_BASE: u8 = b'd';

#[repr(C)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

// DRM_IOCTL_PRIME_FD_TO_HANDLE = _IOWR('d', 0x2e, struct drm_prime_handle)
const DRM_IOCTL_PRIME_FD_TO_HANDLE: libc::c_ulong = (3 << 30) // _IOWR
    | ((std::mem::size_of::<DrmPrimeHandle>() as libc::c_ulong) << 16)
    | ((DRM_IOCTL_BASE as libc::c_ulong) << 8)
    | 0x2e;

// DRM_IOCTL_GEM_CLOSE = _IOW('d', 0x09, struct drm_gem_close)
#[repr(C)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

const DRM_IOCTL_GEM_CLOSE: libc::c_ulong = (1 << 30) // _IOW
    | ((std::mem::size_of::<DrmGemClose>() as libc::c_ulong) << 16)
    | ((DRM_IOCTL_BASE as libc::c_ulong) << 8)
    | 0x09;

/// Shared DRM render node fd — opened once, reused for all PRIME imports.
///
/// Opening `/dev/dri/renderD128` per tensor can deadlock on Vivante DRM drivers
/// when v4l2 decoders (VPU) are concurrently using DMA-BUFs. A single shared fd
/// avoids the contention by routing all PRIME imports through one DRM file instance.
static SHARED_DRM_FD: OnceLock<Option<OwnedFd>> = OnceLock::new();

fn shared_drm_fd() -> Option<BorrowedFd<'static>> {
    SHARED_DRM_FD
        .get_or_init(|| {
            let path = b"/dev/dri/renderD128\0";
            let raw_fd = unsafe {
                libc::open(
                    path.as_ptr() as *const libc::c_char,
                    libc::O_RDWR | libc::O_CLOEXEC,
                )
            };
            if raw_fd < 0 {
                log::debug!(
                    "DrmAttachment: /dev/dri/renderD128 not available: {}",
                    std::io::Error::last_os_error()
                );
                None
            } else {
                log::debug!("DrmAttachment: opened shared /dev/dri/renderD128");
                Some(unsafe { OwnedFd::from_raw_fd(raw_fd) })
            }
        })
        .as_ref()
        .map(|fd| unsafe { BorrowedFd::borrow_raw(fd.as_raw_fd()) })
}

/// Holds a DRM GEM handle that keeps a persistent `dma_buf_attach` alive.
///
/// When the DMA-buf fd is imported through the GPU DRM driver via
/// `DRM_IOCTL_PRIME_FD_TO_HANDLE`, the driver creates a persistent
/// `dma_buf_attach()`. This attachment is required for `DMA_BUF_IOCTL_SYNC`
/// to perform actual cache maintenance on cached CMA heaps.
///
/// Uses a shared DRM render node fd to avoid deadlocks with concurrent
/// V4L2/VPU DMA-BUF usage on Vivante-based SoCs.
///
/// The attachment is released when the GEM handle is closed on drop.
#[derive(Debug)]
pub(crate) struct DrmAttachment {
    gem_handle: u32,
}

impl DrmAttachment {
    /// Import a DMA-buf fd through the GPU DRM driver to create a persistent
    /// `dma_buf_attach`. Returns `None` if `/dev/dri/renderD128` is not
    /// available (e.g. on non-GPU systems or in containers).
    pub(crate) fn new(dma_buf_fd: &OwnedFd) -> Option<Self> {
        let drm_fd = shared_drm_fd()?;

        let mut prime = DrmPrimeHandle {
            handle: 0,
            flags: 0,
            fd: dma_buf_fd.as_raw_fd(),
        };

        let ret =
            unsafe { libc::ioctl(drm_fd.as_raw_fd(), DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut prime) };
        if ret == -1 {
            log::debug!(
                "DrmAttachment: PRIME_FD_TO_HANDLE failed: {}",
                std::io::Error::last_os_error()
            );
            return None;
        }

        log::trace!("DrmAttachment: imported as GEM handle {}", prime.handle);

        Some(Self {
            gem_handle: prime.handle,
        })
    }
}

impl Drop for DrmAttachment {
    fn drop(&mut self) {
        if let Some(drm_fd) = shared_drm_fd() {
            let close = DrmGemClose {
                handle: self.gem_handle,
                pad: 0,
            };
            unsafe { libc::ioctl(drm_fd.as_raw_fd(), DRM_IOCTL_GEM_CLOSE, &close) };
        }
    }
}
