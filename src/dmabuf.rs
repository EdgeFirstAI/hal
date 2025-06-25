use nix::ioctl_write_ptr;
use std::os::fd::{AsRawFd, OwnedFd};

const DMA_BUF_BASE: u8 = b'b';
const DMA_BUF_IOCTL_SYNC: u8 = 0;
const DMA_BUF_IOCTL_PHYS: u8 = 10;

const DMA_BUF_SYNC_READ: u64 = 1 << 0;
const DMA_BUF_SYNC_WRITE: u64 = 1 << 1;
const DMA_BUF_SYNC_START: u64 = 0 << 2;
const DMA_BUF_SYNC_END: u64 = 1 << 2;

#[derive(Default)]
#[repr(C)]
struct dma_buf_sync {
    flags: u64,
}

ioctl_write_ptr!(
    ioctl_dma_buf_sync,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_SYNC,
    dma_buf_sync
);

ioctl_write_ptr!(
    ioctl_dma_buf_phys,
    DMA_BUF_BASE,
    DMA_BUF_IOCTL_PHYS,
    std::ffi::c_ulong
);

fn sync(fd: &OwnedFd, flags: u64) -> nix::Result<()> {
    let mut sync = dma_buf_sync { flags };
    unsafe { ioctl_dma_buf_sync(fd.as_raw_fd(), &mut sync) }?;
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
