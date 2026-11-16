// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Persistent host mapping, decoupled from cache coherency.
//!
//! [`HostPin`] is a stable host address that carries no borrow of the tensor:
//! it holds an `Arc` keepalive over the memory.

use std::sync::Arc;

/// Opaque, type-erased keepalive holding whatever owns the pinned memory alive.
type Keepalive<'a> = Arc<dyn Send + Sync + 'a>;

/// A stable host address for a tensor's data, valid until the pin is dropped.
/// Cloning is cheap and shares the keepalive.
#[derive(Clone)]
pub struct HostPin<'a> {
    /// Held, never read: dropping it releases the producer's allocation.
    #[allow(dead_code)]
    keepalive: Keepalive<'a>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the keepalive holds the owning allocation alive for the pin's life,
// and the base pointer is stable for that allocation's lifetime. Synchronising
// access is the caller's job via sync_for_cpu / sync_for_device.
unsafe impl Send for HostPin<'_> {}
unsafe impl Sync for HostPin<'_> {}

impl<'a> HostPin<'a> {
    pub(crate) fn new(keepalive: Keepalive<'a>, ptr: *mut u8, len: usize) -> Self {
        HostPin {
            keepalive,
            ptr,
            len,
        }
    }

    /// Narrow this pin to `len` bytes, keeping the same address and keepalive.
    /// A `len` larger than the current one is clamped.
    #[cfg(not(edgefirst_tensor_dynamic))]
    pub(crate) fn narrowed(mut self, len: usize) -> Self {
        self.len = self.len.min(len);
        self
    }

    /// The pinned host address. Stable until this pin is dropped.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// The pinned host address, mutable.
    ///
    /// # Safety contract
    ///
    /// The caller must not write while a device may be reading, and must
    /// bracket CPU access with the sync calls on non-coherent backends.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// The tensor's logical byte length, offset-adjusted for sub-region views;
    /// not the backing allocation's capacity.
    ///
    /// For a stride-padded image the pinned window covers `shape.product()`
    /// bytes, which is less than `row_stride * height`. A map guard exposes the
    /// padded extent.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the pinned window addresses no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment of the pinned address, in bytes: the largest power of two
    /// dividing it, or 0 for a null address.
    pub fn alignment(&self) -> usize {
        let addr = self.ptr as usize;
        if addr == 0 {
            return 0;
        }
        1usize << addr.trailing_zeros()
    }

    /// The pinned window as a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure no device write is in flight, and must have
    /// bracketed with `sync_for_cpu` on non-coherent backends.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Owner of an `mmap` whose lifetime is independent of any map guard. Shared by
/// every fd-backed backend (DMA-BUF, SHM).
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
pub(crate) struct MmapOwner {
    ptr: std::ptr::NonNull<std::ffi::c_void>,
    len: usize,
}

// SAFETY: the mapping is valid for `len` bytes until Drop, and MAP_SHARED
// mappings are shareable across threads.
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
unsafe impl Send for MmapOwner {}
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
unsafe impl Sync for MmapOwner {}

#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
impl MmapOwner {
    pub(crate) fn new(ptr: std::ptr::NonNull<std::ffi::c_void>, len: usize) -> Self {
        MmapOwner { ptr, len }
    }

    /// Base address of the whole mapping, before any tensor offset.
    pub(crate) fn base(&self) -> *mut u8 {
        self.ptr.as_ptr() as *mut u8
    }
}

/// Orders CPU mappings against GPU-driver calls that unmap an address range
/// twice: mappings are established under the shared side and such driver calls
/// run under the exclusive side.
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
static CPU_MAPPING_GUARD: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Shared side of [`CPU_MAPPING_GUARD`], held across each `mmap`.
#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
pub(crate) fn cpu_mapping_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    CPU_MAPPING_GUARD.read().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` while no tensor CPU mapping can be created in this process.
///
/// For a GPU-driver call that may unmap a range twice (Adreno
/// `eglDestroyImage`). `f` must not map a tensor, or it deadlocks.
#[doc(hidden)]
#[cfg(not(edgefirst_tensor_dynamic))]
pub fn with_cpu_mappings_excluded<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(unix)]
    let _guard = CPU_MAPPING_GUARD.write().unwrap_or_else(|e| e.into_inner());
    f()
}

/// The `dynamic` backend's [`with_cpu_mappings_excluded`]: excludes the
/// library's mappings, which are the only ones this build creates.
#[doc(hidden)]
#[cfg(edgefirst_tensor_dynamic)]
pub fn with_cpu_mappings_excluded<F: FnOnce() -> R, R>(f: F) -> R {
    struct Call<F, R> {
        f: Option<F>,
        out: Option<std::thread::Result<R>>,
    }
    unsafe extern "C" fn trampoline<F: FnOnce() -> R, R>(ctx: *mut std::ffi::c_void) {
        // SAFETY: `ctx` is the `Call` below, alive and exclusively borrowed
        // for the duration of the export's synchronous callback.
        let call = unsafe { &mut *ctx.cast::<Call<F, R>>() };
        if let Some(f) = call.f.take() {
            // Catch a panic so it does not unwind through the C frame.
            call.out = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
        }
    }
    let mut call = Call {
        f: Some(f),
        out: None,
    };
    // SAFETY: the callback and its context stay valid for the whole call,
    // and the export invokes the callback at most once, synchronously.
    unsafe {
        edgefirst_tensor_ffi::ef_tensor_with_cpu_mappings_excluded(
            Some(trampoline::<F, R>),
            (&mut call as *mut Call<F, R>).cast(),
        )
    };
    match call
        .out
        .expect("ef_tensor_with_cpu_mappings_excluded skipped its callback")
    {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[cfg(all(unix, not(edgefirst_tensor_dynamic)))]
impl Drop for MmapOwner {
    #[cfg_attr(test, mutants::skip)]
    fn drop(&mut self) {
        unsafe {
            let _ = nix::sys::mman::munmap(self.ptr, self.len);
        }
    }
}

impl std::fmt::Debug for HostPin<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostPin")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("alignment", &self.alignment())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pin over a synthetic address; nothing here dereferences it.
    fn pin_at(addr: usize, len: usize) -> HostPin<'static> {
        HostPin::new(Arc::new(()), addr as *mut u8, len)
    }

    #[test]
    fn is_empty_tracks_the_length() {
        assert!(pin_at(0x1000, 0).is_empty());
        assert!(!pin_at(0x1000, 1).is_empty());
    }

    #[test]
    fn alignment_is_the_largest_power_of_two_dividing_the_address() {
        assert_eq!(pin_at(0, 8).alignment(), 0);
        assert_eq!(pin_at(0x6, 8).alignment(), 2);
        assert_eq!(pin_at(0x1040, 8).alignment(), 64);
        assert_eq!(pin_at(0x1000, 8).alignment(), 4096);
        let top = 1usize << (usize::BITS - 1);
        assert_eq!(pin_at(top, 8).alignment(), top);
    }

    #[test]
    fn debug_names_the_type_and_its_extent() {
        let text = format!("{:?}", pin_at(0x1040, 7));
        assert!(text.contains("HostPin"), "{text}");
        assert!(text.contains("len: 7"), "{text}");
        assert!(text.contains("alignment: 64"), "{text}");
    }
}
