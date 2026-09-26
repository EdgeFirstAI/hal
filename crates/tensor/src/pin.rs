// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Persistent host mapping, decoupled from cache coherency.
//!
//! `map()` fuses two separate concerns: establishing a host address, and
//! bracketing CPU access for coherency. Fusing them makes a whole class of
//! consumer impossible to express — TFLite's `SetCustomAllocationForTensor`
//! and ONNX Runtime's external tensors both retain a raw pointer for the
//! interpreter's lifetime, far beyond any single access window
//! ([#134](https://github.com/EdgeFirstAI/hal/issues/134)).
//!
//! The blocker is a borrow, not a lifetime: a map guard borrows the tensor,
//! while `ImageProcessor::convert(&mut self, src, dst)` needs `&mut` on that
//! same tensor every frame, and the two cannot coexist. So [`HostPin`] carries
//! **no borrow of the tensor at all** — it holds an `Arc` keepalive over the
//! memory instead, which is what lets a caller pin once and still mutate the
//! tensor in the frame loop.
//!
//! IOSurface is the reference implementation this generalises: it already
//! separates "address of the surface" (`IOSurfaceGetBaseAddress`, stable) from
//! "CPU access window" (`IOSurfaceLock`), which is why the Apple path already
//! works and the DMA-BUF one does not.

use std::sync::Arc;

/// Opaque keepalive holding whatever owns the pinned memory alive.
///
/// Type-erased so a pin does not leak the element type or the backend, which
/// is what allows `HostPin` to be a single type across every backend.
///
/// Deliberately **not** `dyn Any`: `Any` carries a `'static` bound, and this is
/// never downcast — it exists only so its `Drop` runs. Requiring `'static`
/// would assert the allocation lives for the whole program, which is false for
/// a tensor; the keepalive only has to outlive the pin, which `'a` says
/// exactly.
type Keepalive<'a> = Arc<dyn Send + Sync + 'a>;

/// A stable host address for a tensor's data, valid until the pin is dropped.
///
/// Deliberately carries **no lifetime parameter and no reference to the
/// tensor**: see the module docs. Cloning is cheap and shares the keepalive.
#[derive(Clone)]
pub struct HostPin<'a> {
    /// Held, never read: dropping this is what releases the producer's
    /// allocation, so its lifetime IS the guarantee the pinned address stays
    /// valid. Clippy cannot see that a field's only purpose is its Drop.
    #[allow(dead_code)]
    keepalive: Keepalive<'a>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the keepalive holds the owning allocation alive for the pin's life,
// and every backend that yields a pin has a base pointer that is stable for
// that allocation's lifetime. Sharing the address across threads is therefore
// sound; synchronising *access* is the caller's job via sync_for_cpu /
// sync_for_device, exactly as it is for the underlying buffer.
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
    ///
    /// Backends produce a pin covering everything addressable from the tensor's
    /// offset (capacity, including any pitch padding); the public
    /// [`pin_host`](crate::Tensor::pin_host) narrows that to the tensor's
    /// logical extent so a consumer cannot mistake padding for data, while a
    /// map guard keeps the wider window it needs to expose padded rows.
    ///
    /// Narrowing only ever shrinks: a `len` larger than the current one is
    /// clamped, so this cannot widen a window beyond what was mapped.
    ///
    /// `static`-only: the sole caller, `Tensor::pin_host` (`impl<T>
    /// Tensor<T>` in `lib.rs`), is itself `#[cfg(feature = "static")]`.
    /// `dynamic`'s own `pin_host` (`dynamic_backend.rs`/`dynamic_tensor.rs`)
    /// does not call this -- it hands back whatever extent `ef_tensor_map`
    /// reports directly, unnarrowed.
    #[cfg(feature = "static")]
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
    /// Takes `&self` rather than `&mut self` on purpose: a raw pointer handed
    /// out here can legitimately coexist with a `&mut` derived elsewhere (that
    /// is the entire point of the pin), so pretending exclusivity via `&mut
    /// self` would be a lie. Writing through it is the caller's obligation to
    /// sequence — see [`crate::Tensor::sync_for_cpu`].
    ///
    /// # Safety contract
    ///
    /// The caller must not write while a device may be reading, and must
    /// bracket CPU access with the sync calls on non-coherent backends.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// The tensor's **logical** byte length, offset-adjusted for sub-region
    /// views — not the backing allocation's capacity.
    ///
    /// Backends round allocations up (page size, pitch alignment), and every
    /// backend reports the logical length so a consumer cannot mistake padding
    /// for data. This is what TFLite's `SetCustomAllocationForTensor` wants.
    ///
    /// One consequence: for a **stride-padded** image the pinned window covers
    /// `shape.product()` bytes, which is less than `row_stride * height`. Use a
    /// map guard, which exposes the padded extent, when you need to write whole
    /// padded rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the pinned window addresses no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignment of the pinned address, in bytes.
    ///
    /// TFLite requires 64-byte alignment (`kDefaultTensorAlignment`) unless the
    /// caller opts out, and upstream warns that opting out can crash in
    /// `Invoke()`. Page-backed mappings satisfy it, but a non-zero tensor
    /// offset can break it — so callers can check rather than assume.
    pub fn alignment(&self) -> usize {
        let addr = self.ptr as usize;
        if addr == 0 {
            return 0;
        }
        // A non-zero address has at most BITS - 1 trailing zeros.
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

/// Owner of an `mmap` whose lifetime is independent of any map guard.
///
/// Shared by every fd-backed backend (DMA-BUF, SHM): the address a pin hands
/// out is a property of the mapping, so the mapping has to be owned by
/// something with its own lifetime rather than by the guard that happened to
/// create it. That single change is what lets a host pointer outlive a guard
/// at all — see the module docs and issue #134.
///
/// `static`-gated as well as unix-gated: both backends that own an mmap
/// (DMA-BUF and SHM) are storage, and only the `static` backend has
/// storage. Under `dynamic` the mapping is owned on the far side of the
/// ABI and reached through `ef_tensor_map`, so nothing here constructs one.
#[cfg(all(unix, feature = "static"))]
pub(crate) struct MmapOwner {
    ptr: std::ptr::NonNull<std::ffi::c_void>,
    len: usize,
}

// SAFETY: the mapping is valid for `len` bytes until Drop, and MAP_SHARED
// mappings are shareable across threads. Coherency is the caller's obligation
// via sync_for_cpu / sync_for_device, exactly as for any map guard.
#[cfg(all(unix, feature = "static"))]
unsafe impl Send for MmapOwner {}
#[cfg(all(unix, feature = "static"))]
unsafe impl Sync for MmapOwner {}

#[cfg(all(unix, feature = "static"))]
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
/// twice.
///
/// Adreno's proprietary EGL maps every imported DMA-BUF into the process at
/// `eglCreateImage` and, inside a single `eglDestroyImage`, unmaps that range
/// twice. A mapping another thread creates between the two `munmap`s can land
/// in the freed range and is destroyed by the second one, so that thread's
/// next access faults. Mappings are established under the shared side and
/// such driver calls run under the exclusive side, which keeps this crate's
/// own mappings out of that window.
#[cfg(all(unix, feature = "static"))]
static CPU_MAPPING_GUARD: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Shared side of [`CPU_MAPPING_GUARD`], held across each `mmap`.
#[cfg(all(unix, feature = "static"))]
pub(crate) fn cpu_mapping_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    CPU_MAPPING_GUARD.read().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` while no tensor CPU mapping can be created in this process.
///
/// For a GPU-driver call that may unmap a range twice (Adreno
/// `eglDestroyImage`). `f` must not map a tensor, or it deadlocks.
#[doc(hidden)]
#[cfg(feature = "static")]
pub fn with_cpu_mappings_excluded<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(unix)]
    let _guard = CPU_MAPPING_GUARD.write().unwrap_or_else(|e| e.into_inner());
    f()
}

/// The `dynamic` backend's [`with_cpu_mappings_excluded`]: excludes the
/// library's mappings, which are the only ones this build creates.
#[doc(hidden)]
#[cfg(feature = "dynamic")]
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
            // A panic must not unwind through the C frame.
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

#[cfg(all(unix, feature = "static"))]
impl Drop for MmapOwner {
    // munmap has no in-process observer that is not racy: probing the range
    // afterwards can see another thread's new mapping at the same address.
    #[cfg_attr(test, mutants::skip)]
    fn drop(&mut self) {
        // Nothing useful to do on failure at Drop time; the address space is
        // reclaimed at process exit regardless.
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
