// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One CPU-access window, for every backend.
//!
//! Replaces the six per-backend `*Map<T>` types. They differed in how they
//! *obtained* an address — `mmap`, `IOSurfaceLock`, `glMapBufferRange`,
//! `AHardwareBuffer_lock` — but once obtained they all carried the same five
//! things: a base pointer, the shape, an offset, an optional padded extent, and
//! whether writes are allowed. [`HostPin`] already absorbs the first and third
//! (its address is offset-adjusted, its keepalive releases whatever the backend
//! acquired), which leaves this type holding the other three.
//!
//! The acquisition difference does not need a type per backend, only a
//! constructor per backend: persistent ones hand over a real pin, lock-scoped
//! ones (PBO, AHardwareBuffer) hand over a pin whose keepalive unlocks on drop.
//! Either way what the caller gets is the same window.

use crate::{pin::HostPin, Num, TensorMapTrait};
use std::fmt;
use std::marker::PhantomData;

/// A CPU-accessible window onto a tensor's memory.
pub struct HostView<'a, T> {
    /// `None` after [`unmap`](TensorMapTrait::unmap): dropping the pin is what
    /// releases the backend's mapping or lock, so clearing it *is* the unmap.
    pin: Option<HostPin<'a>>,
    shape: Vec<usize>,
    /// When `Some(bytes)`, expose `bytes / size_of::<T>()` elements instead of
    /// the shape's product — the padded window a stride-aligned image needs so
    /// CPU row iteration can honour `row_stride` without running past the end.
    byte_size_override: Option<usize>,
    /// Whether `as_mut_slice` is permitted. A read-only map keeps the mapping
    /// itself readable/writable; this enforces the API contract uniformly, as
    /// the per-backend maps did.
    writable: bool,
    _marker: PhantomData<T>,
}

impl<'a, T> HostView<'a, T>
where
    T: Num + Clone + fmt::Debug,
{
    #[allow(dead_code)] // wired when the first backend migrates
    pub(crate) fn new(
        pin: HostPin<'a>,
        shape: Vec<usize>,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Self {
        HostView {
            pin: Some(pin),
            shape,
            byte_size_override,
            writable: access.writes(),
            _marker: PhantomData,
        }
    }

    /// Elements exposed by `as_slice`.
    ///
    /// Clamped to what the pin actually maps, so a `byte_size_override` larger
    /// than the mapping cannot produce an out-of-bounds slice — the override is
    /// a request to expose padding, never permission to invent it.
    fn len_elems(&self) -> usize {
        let Some(pin) = self.pin.as_ref() else {
            return 0;
        };
        let elem = std::mem::size_of::<T>();
        if elem == 0 {
            return 0;
        }
        let requested = match self.byte_size_override {
            Some(bytes) => bytes / elem,
            None => self.shape.iter().product::<usize>(),
        };
        requested.min(pin.len() / elem)
    }

    /// Re-view this mapping as raw bytes, type-erased.
    ///
    /// Keeps the same pin (so the same platform sync bracket still runs on
    /// [`Drop`]) and the same writability; only the shape bookkeeping changes,
    /// from "N elements of `T`" to "this many bytes". Computed directly in
    /// bytes rather than by calling [`len_elems`](Self::len_elems) and
    /// multiplying back by `size_of::<T>()`, which would round a
    /// non-multiple-of-`size_of::<T>()` `byte_size_override` down twice.
    ///
    /// `pub(crate)`: this is plumbing for `static`'s `TensorDyn::map_bytes`
    /// (`tensor_dyn/static_backend.rs`), the type-erased byte-level guard a
    /// C ABI consumer needs (it does not know `T` and cannot call the typed
    /// `map_with`); it is not meant as a public entry point in its own
    /// right. It was `static`-only on the reasoning that `dynamic` never has
    /// a typed `HostView<T>` to re-view, since its own `TensorDyn::map_bytes`
    /// builds a byte view straight from `ef_tensor_map`. That stopped being
    /// true when the dynamic backend learned to map a PBO-backed tensor
    /// (`dynamic_backend.rs::map_pin_pbo`): a PBO's real bytes live in the GL
    /// buffer, so that path goes through `PboTensor<T>::map_with` and gets a
    /// typed guard it must erase.
    pub(crate) fn into_bytes(self) -> HostView<'a, u8> {
        let elem = std::mem::size_of::<T>().max(1);
        let requested_bytes = match self.byte_size_override {
            Some(bytes) => bytes,
            None => self.shape.iter().product::<usize>() * elem,
        };
        let byte_len = match self.pin.as_ref() {
            Some(pin) => requested_bytes.min(pin.len()),
            None => 0,
        };
        HostView {
            pin: self.pin,
            shape: vec![byte_len],
            byte_size_override: None,
            writable: self.writable,
            _marker: PhantomData,
        }
    }
}

impl<T> TensorMapTrait<T> for HostView<'_, T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        // Idempotent: the pin's keepalive releases exactly once, when the last
        // clone drops, so a second unmap is a no-op rather than a double free.
        self.pin = None;
    }

    fn is_writable(&self) -> bool {
        self.writable
    }

    fn as_slice(&self) -> &[T] {
        let Some(pin) = self.pin.as_ref() else {
            // Unmapped. The per-backend maps dereferenced a stale pointer here;
            // an empty slice is the safe reading of "there is no window".
            return &[];
        };
        let len = self.len_elems();
        if len == 0 {
            return &[];
        }
        // SAFETY: the pin's keepalive holds the mapping open for at least this
        // borrow, and `len_elems` is clamped to the mapped extent.
        unsafe { std::slice::from_raw_parts(pin.as_ptr() as *const T, len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        crate::assert_map_writable(self.writable, "HostView");
        let len = self.len_elems();
        let Some(pin) = self.pin.as_ref() else {
            return &mut [];
        };
        if len == 0 {
            return &mut [];
        }
        // SAFETY: as above, plus `&mut self` gives exclusive access to this
        // window for the borrow.
        unsafe { std::slice::from_raw_parts_mut(pin.as_mut_ptr() as *mut T, len) }
    }
}

/// Slice access by deref, as the per-backend maps offered: `map.fill(0)` and
/// `map.iter()` keep working without an explicit `as_slice()`.
impl<T> std::ops::Deref for HostView<'_, T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for HostView<'_, T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> fmt::Debug for HostView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostView")
            .field("mapped", &self.pin.is_some())
            .field("shape", &self.shape)
            .field("byte_size_override", &self.byte_size_override)
            .field("writable", &self.writable)
            .finish()
    }
}
