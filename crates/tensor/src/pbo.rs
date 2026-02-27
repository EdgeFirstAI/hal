// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;

/// Raw mapped pointer from a PBO. CPU-accessible while the buffer is mapped.
/// The pointer is only valid between map and unmap calls.
pub struct PboMapping {
    pub ptr: *mut u8,
    pub size: usize,
}

// SAFETY: PboMapping is only created by PboOps::map_buffer which runs on the
// GL thread, but the resulting pointer is used on the caller's thread. This is
// safe because glMapBufferRange returns a CPU-visible pointer that can be
// accessed from any thread while the buffer remains mapped.
unsafe impl Send for PboMapping {}

/// Trait for PBO GL operations, implemented by the image crate.
///
/// All methods are blocking — they send commands to the GL thread
/// and wait for completion. Implementations must ensure GL context
/// is current on the thread that executes the actual GL calls.
pub trait PboOps: Send + Sync {
    /// Map the PBO for CPU read/write access.
    /// The returned PboMapping is valid until `unmap_buffer` is called.
    fn map_buffer(&self, buffer_id: u32, size: usize) -> Result<PboMapping>;

    /// Unmap a previously mapped PBO. Must be called before GL operations
    /// on this buffer (GLES 3.0 requirement).
    fn unmap_buffer(&self, buffer_id: u32) -> Result<()>;

    /// Delete the PBO. Fire-and-forget — no reply needed.
    /// Called from PboTensor's Drop impl.
    fn delete_buffer(&self, buffer_id: u32);
}
