// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One CPU-access window, for every backend.
//!
//! [`HostView`] pairs a [`HostPin`] (the offset-adjusted base address, and the
//! keepalive that releases the backend's mapping or lock) with the shape, an
//! optional padded extent, and whether writes are allowed.

use crate::{pin::HostPin, Num, TensorMapTrait};
use std::fmt;
use std::marker::PhantomData;

/// A CPU-accessible window onto a tensor's memory.
pub struct HostView<'a, T> {
    /// `None` after [`unmap`](TensorMapTrait::unmap); dropping the pin releases
    /// the backend's mapping or lock.
    pin: Option<HostPin<'a>>,
    shape: Vec<usize>,
    /// When `Some(bytes)`, expose `bytes / size_of::<T>()` elements instead of
    /// the shape's product: the padded window of a stride-aligned image.
    byte_size_override: Option<usize>,
    /// Whether `as_mut_slice` is permitted.
    writable: bool,
    _marker: PhantomData<T>,
}

impl<'a, T> HostView<'a, T>
where
    T: Num + Clone + fmt::Debug,
{
    #[allow(dead_code)]
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

    /// Elements exposed by `as_slice`, clamped to what the pin maps.
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

    /// Re-view this mapping as raw bytes, type-erased. Keeps the same pin and
    /// the same writability; the shape becomes the byte count.
    #[cfg(not(edgefirst_tensor_dynamic))]
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
        // Idempotent: a second unmap is a no-op.
        self.pin = None;
    }

    fn is_writable(&self) -> bool {
        self.writable
    }

    fn as_slice(&self) -> &[T] {
        let Some(pin) = self.pin.as_ref() else {
            // Unmapped: an empty slice.
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

/// Slice access by deref: `map.fill(0)` and `map.iter()` work without an
/// explicit `as_slice()`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A view over `pin_len` bytes of a 64-element backing.
    fn view_u32(
        pin_len: usize,
        shape: Vec<usize>,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> HostView<'static, u32> {
        let backing: Arc<Vec<u32>> = Arc::new((0..64).collect());
        let ptr = backing.as_ptr() as *mut u8;
        HostView::new(
            HostPin::new(backing, ptr, pin_len),
            shape,
            byte_size_override,
            access,
        )
    }

    #[test]
    fn a_byte_override_exposes_that_many_bytes_as_elements() {
        let v = view_u32(256, vec![1], Some(8), crate::CpuAccess::Read);
        assert_eq!(v.as_slice(), &[0, 1]);
    }

    #[test]
    fn the_exposed_length_is_clamped_to_the_pinned_bytes() {
        let v = view_u32(16, vec![4], Some(64), crate::CpuAccess::Read);
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
        let v = view_u32(16, vec![8], None, crate::CpuAccess::Read);
        assert_eq!(v.as_slice().len(), 4);
    }

    #[test]
    fn unmap_empties_the_view() {
        let mut v = view_u32(256, vec![4], None, crate::CpuAccess::Read);
        assert_eq!(v.as_slice().len(), 4);
        v.unmap();
        assert!(v.as_slice().is_empty());
        assert!(format!("{v:?}").contains("mapped: false"));
    }

    #[test]
    fn debug_names_the_type_and_its_writability() {
        let v = view_u32(256, vec![4], Some(16), crate::CpuAccess::ReadWrite);
        let text = format!("{v:?}");
        assert!(text.contains("HostView"), "{text}");
        assert!(text.contains("mapped: true"), "{text}");
        assert!(text.contains("shape: [4]"), "{text}");
        assert!(text.contains("byte_size_override: Some(16)"), "{text}");
        assert!(text.contains("writable: true"), "{text}");
    }
}
