// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use nix::ioctl_write_ptr;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::OnceLock;

const DMA_BUF_BASE: u8 = b'b';
const DMA_BUF_IOCTL_SYNC: u8 = 0;

// Values of `DMA_BUF_SYNC_*` from `<linux/dma-buf.h>`, written as literals:
// READ = 1 << 0, WRITE = 1 << 1, START = 0 << 2, END = 1 << 2.
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_WRITE: u64 = 2;
const DMA_BUF_SYNC_RW: u64 = 3;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 4;

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

fn sync(fd: &OwnedFd, flags: u64) -> nix::Result<()> {
    let sync = DmaBufSync { flags };
    unsafe { ioctl_dma_buf_sync(fd.as_raw_fd(), &sync) }?;
    Ok(())
}

/// The `DMA_BUF_IOCTL_SYNC` flags for one direction and one half of the
/// bracket.
///
/// `CpuAccess::None` maps to read-write: callers reject it before reaching
/// here, but a zero direction would issue a no-op ioctl that silently
/// maintains nothing.
fn sync_flags(start: bool, access: crate::CpuAccess) -> u64 {
    let mut flags = 0;
    if access.reads() {
        flags |= DMA_BUF_SYNC_READ;
    }
    if access.writes() {
        flags |= DMA_BUF_SYNC_WRITE;
    }
    if flags == 0 {
        flags = DMA_BUF_SYNC_RW;
    }
    flags |= if start {
        DMA_BUF_SYNC_START
    } else {
        DMA_BUF_SYNC_END
    };
    flags
}

#[cfg(test)]
mod sync_counter {
    thread_local! {
        /// `(start, end)` sync calls issued on this thread, so a test can
        /// assert that a map's bracket is balanced. Per-thread so concurrently
        /// running tests do not see each other's calls.
        pub(crate) static SYNC_CALLS: std::cell::Cell<(usize, usize)> =
            const { std::cell::Cell::new((0, 0)) };
    }
}
#[cfg(test)]
pub(crate) use sync_counter::SYNC_CALLS;

/// Issue `DMA_BUF_IOCTL_SYNC` for one direction and one half of the bracket.
///
/// The direction matters: a read-only bracket lets the kernel skip the
/// writeback at END, a write-only one skips the invalidate at START — so
/// `start` and `end` MUST be called with the same `access`, or the
/// maintenance one half performed is undone by the other half's omission.
pub(crate) fn sync_access(fd: &OwnedFd, start: bool, access: crate::CpuAccess) -> nix::Result<()> {
    #[cfg(test)]
    SYNC_CALLS.with(|c| {
        let (s, e) = c.get();
        c.set(if start { (s + 1, e) } else { (s, e + 1) });
    });
    sync(fd, sync_flags(start, access))
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
const DRM_IOCTL_PRIME_FD_TO_HANDLE: nix::sys::ioctl::ioctl_num_type =
    nix::request_code_readwrite!(DRM_IOCTL_BASE, 0x2e, std::mem::size_of::<DrmPrimeHandle>());

#[repr(C)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

// DRM_IOCTL_GEM_CLOSE = _IOW('d', 0x09, struct drm_gem_close)
const DRM_IOCTL_GEM_CLOSE: nix::sys::ioctl::ioctl_num_type =
    nix::request_code_write!(DRM_IOCTL_BASE, 0x09, std::mem::size_of::<DrmGemClose>());

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
    /// If `imported` is true the fd came from an external source (e.g. a
    /// Neutron NPU kernel driver) and a PRIME import failure is expected —
    /// logged at DEBUG.  For self-allocated dma_heap buffers, failure is
    /// unexpected and logged at WARN.
    pub(crate) fn new(dma_buf_fd: &OwnedFd, imported: bool) -> Option<Self> {
        let drm_fd = shared_drm_fd()?;

        let mut prime = DrmPrimeHandle {
            handle: 0,
            flags: 0,
            fd: dma_buf_fd.as_raw_fd(),
        };

        let ret =
            unsafe { libc::ioctl(drm_fd.as_raw_fd(), DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut prime) };
        if ret == -1 {
            let err = std::io::Error::last_os_error();
            if imported {
                log::debug!("DrmAttachment: PRIME_FD_TO_HANDLE failed (imported fd): {err}");
            } else {
                log::warn!("DrmAttachment: PRIME_FD_TO_HANDLE failed: {err}");
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAccess;

    fn dev_null() -> OwnedFd {
        OwnedFd::from(std::fs::File::open("/dev/null").expect("open /dev/null"))
    }

    #[test]
    fn sync_flags_match_the_dma_buf_uapi_for_every_direction() {
        // <linux/dma-buf.h>: READ = 1, WRITE = 2, START = 0, END = 4.
        let table = [
            (true, CpuAccess::Read, 1),
            (true, CpuAccess::Write, 2),
            (true, CpuAccess::ReadWrite, 3),
            (true, CpuAccess::None, 3),
            (false, CpuAccess::Read, 5),
            (false, CpuAccess::Write, 6),
            (false, CpuAccess::ReadWrite, 7),
            (false, CpuAccess::None, 7),
        ];
        for (start, access, want) in table {
            assert_eq!(
                sync_flags(start, access),
                want,
                "start={start} access={access:?}"
            );
        }
    }

    /// The request codes the kernel's `_IOWR`/`_IOW` produce for these two
    /// structs on the generic ioctl encoding (x86, Arm, RISC-V).
    #[test]
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv64"
    ))]
    fn drm_ioctl_request_codes_match_the_kernel_uapi() {
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE as u32, 0xC00C_642E);
        assert_eq!(DRM_IOCTL_GEM_CLOSE as u32, 0x4008_6409);
    }

    #[test]
    fn sync_on_a_fd_that_is_not_a_dma_buf_reports_enotty() {
        let fd = dev_null();
        for start in [true, false] {
            assert_eq!(
                sync_access(&fd, start, CpuAccess::ReadWrite),
                Err(nix::errno::Errno::ENOTTY),
                "start={start}"
            );
        }
    }

    #[test]
    fn sync_calls_are_counted_per_bracket_half() {
        let fd = dev_null();
        let before = SYNC_CALLS.with(|c| c.get());
        let _ = sync_access(&fd, true, CpuAccess::Read);
        let _ = sync_access(&fd, false, CpuAccess::Read);
        let _ = sync_access(&fd, false, CpuAccess::Read);
        let after = SYNC_CALLS.with(|c| c.get());
        assert_eq!((after.0 - before.0, after.1 - before.1), (1, 2));
    }

    /// Whether the shared render node opened must agree with whether this
    /// process can open `/dev/dri/renderD128` itself, and the shared fd must
    /// be read-write and must not leak into exec'd children.
    #[test]
    fn shared_drm_fd_opens_the_render_node_read_write_close_on_exec() {
        let openable = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .is_ok();
        let shared = shared_drm_fd();
        assert_eq!(shared.is_some(), openable);
        if let Some(fd) = shared {
            let flags = nix::fcntl::fcntl(fd, nix::fcntl::FcntlArg::F_GETFD).expect("F_GETFD");
            assert_ne!(
                flags & libc::FD_CLOEXEC,
                0,
                "shared DRM fd lacks FD_CLOEXEC"
            );
            let fl = nix::fcntl::fcntl(fd, nix::fcntl::FcntlArg::F_GETFL).expect("F_GETFL");
            assert_eq!(
                fl & libc::O_ACCMODE,
                libc::O_RDWR,
                "shared DRM fd is not O_RDWR"
            );
        }
    }

    /// Dropping an attachment closes its GEM handle: closing the same handle
    /// again afterwards must fail with `EINVAL`.
    #[test]
    #[cfg(not(miri))]
    fn dropping_a_drm_attachment_closes_its_gem_handle() {
        const NAME: &str = "dropping_a_drm_attachment_closes_its_gem_handle";
        if !crate::test_support::dma_or_skip(NAME) {
            return;
        }
        let heap = dma_heap::Heap::new(dma_heap::HeapKind::Cma)
            .or_else(|_| dma_heap::Heap::new(dma_heap::HeapKind::System))
            .expect("dma heap");
        let buf = heap.allocate(4096).expect("allocate");
        let Some(attachment) = DrmAttachment::new(&buf, true) else {
            crate::test_support::report_skip(&format!(
                "{NAME} - the render node does not accept a PRIME import of a heap buffer"
            ));
            return;
        };
        let handle = attachment.gem_handle;
        drop(attachment);

        let drm_fd = shared_drm_fd().expect("an attachment implies a render node");
        let close = DrmGemClose { handle, pad: 0 };
        let ret = unsafe { libc::ioctl(drm_fd.as_raw_fd(), DRM_IOCTL_GEM_CLOSE, &close) };
        let err = std::io::Error::last_os_error();
        assert_eq!(ret, -1, "GEM handle {handle} was still open after drop");
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
    }
}
