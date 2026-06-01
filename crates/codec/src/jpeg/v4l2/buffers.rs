// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! RAII wrapper for an `mmap`'d V4L2 buffer plane.

use std::ffi::c_void;
use std::num::NonZero;
use std::os::fd::BorrowedFd;
use std::ptr::NonNull;

use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};

/// A memory-mapped V4L2 buffer plane. Unmaps on drop.
pub(crate) struct Mmap {
    ptr: NonNull<c_void>,
    len: usize,
}

impl Mmap {
    /// Map `len` bytes of the device buffer at `offset` (the `mem_offset`
    /// reported by `VIDIOC_QUERYBUF`).
    pub fn new(fd: BorrowedFd, len: usize, offset: i64) -> nix::Result<Self> {
        let nz = NonZero::new(len).ok_or(nix::errno::Errno::EINVAL)?;
        // SAFETY: mapping a driver buffer the kernel just described via
        // QUERYBUF; `len`/`offset` come straight from that query.
        let ptr = unsafe {
            mmap(
                None,
                nz,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                fd,
                offset,
            )?
        };
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/len describe a valid mapping owned by this Mmap.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr/len describe a valid mapping uniquely owned here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: unmapping our own mapping exactly once.
        let _ = unsafe { munmap(self.ptr, self.len) };
    }
}
