// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{
    cell::UnsafeCell,
    fmt,
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// Interior-mutable backing allocation for `MemTensor`.
///
/// The elements live in `UnsafeCell`s so that sub-region views — which share
/// one allocation through a cloned `Arc` — can each hand out `&mut [T]` over
/// their own window with **correct write provenance**. A `&mut [T]` derived
/// from a shared `Arc<Vec<T>>` via `Vec::as_ptr` carries read-only provenance
/// (the pointer is born of a shared borrow), so writing through it is undefined
/// behaviour; `UnsafeCell` is what makes interior mutation through the shared
/// `Arc` sound.
///
/// Disjointness is the caller's contract: distinct `subview` windows must not
/// overlap if mapped mutably at the same time (the same contract as `DmaMap`,
/// whose `mmap` likewise hands out independent `&mut` windows of one buffer).
/// `UnsafeCell` makes the *non-overlapping* case sound; overlapping mutable
/// windows held simultaneously remain UB by the documented `subview` contract.
struct MemBacking<T> {
    cells: Box<[UnsafeCell<T>]>,
}

// SAFETY: `UnsafeCell<T>` is `!Sync`, but a `MemBacking` is only mutated through
// the disjoint-window contract documented above (and was already guarded by the
// `unsafe impl Send/Sync for MemTensor`). The cells add no thread-safety hazard
// beyond what `MemTensor` already asserts; they only fix pointer provenance.
unsafe impl<T: Send> Send for MemBacking<T> {}
unsafe impl<T: Sync> Sync for MemBacking<T> {}

impl<T: fmt::Debug> fmt::Debug for MemBacking<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemBacking")
            .field("len", &self.cells.len())
            .finish()
    }
}

impl<T> MemBacking<T> {
    /// Number of elements in the allocation.
    fn len(&self) -> usize {
        self.cells.len()
    }

    /// Base pointer with write provenance (the cells are `UnsafeCell`, so
    /// interior mutation through the shared `Arc` is sound). `UnsafeCell<T>`
    /// is `#[repr(transparent)]`, so the address equals the `T` element's.
    fn base_ptr(&self) -> *mut T {
        self.cells.as_ptr() as *mut T
    }

    /// Build a zeroed backing of `n` elements.
    fn zeroed(n: usize) -> Self
    where
        T: Num + Clone,
    {
        let cells: Vec<UnsafeCell<T>> = (0..n).map(|_| UnsafeCell::new(T::zero())).collect();
        Self {
            cells: cells.into_boxed_slice(),
        }
    }
}

#[derive(Debug)]
pub struct MemTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub shape: Vec<usize>,
    /// Shared, fixed-size backing allocation. An owned tensor holds a
    /// refcount-1 `Arc`; zero-copy sub-region views clone it to co-own the
    /// parent's buffer. The allocation is never resized after creation
    /// (`reshape`/`set_logical_shape` only adjust the logical `shape` within
    /// capacity), so the base pointer is stable for the lifetime of the `Arc`.
    data: Arc<MemBacking<T>>,
    /// Byte offset into `data` where this tensor's logical window begins. `0`
    /// for whole-buffer tensors; non-zero for sub-region views. Mirrors
    /// `DmaTensor::mmap_offset`.
    offset: usize,
    identity: crate::BufferIdentity,
}

unsafe impl<T> Send for MemTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for MemTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> MemTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate `byte_capacity` bytes and set an initial logical `shape`
    /// (whose byte size must fit the capacity).
    pub fn with_capacity_bytes(
        shape: &[usize],
        byte_capacity: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        let cap_elems = byte_capacity / elem;
        let logical: usize = shape.iter().product();
        if logical > cap_elems {
            return Err(Error::InsufficientCapacity {
                needed: logical * elem,
                capacity: byte_capacity,
            });
        }
        Ok(MemTensor {
            name: name.unwrap_or("mem_tensor").to_owned(),
            shape: shape.to_vec(),
            data: Arc::new(MemBacking::zeroed(cap_elems)),
            offset: 0,
            identity: crate::BufferIdentity::new(),
        })
    }

    /// Create a zero-copy sub-region view that shares `parent`'s allocation.
    ///
    /// The view maps the window `[offset_bytes, offset_bytes + logical_size)`
    /// measured from `parent`'s own logical start, where
    /// `logical_size = shape.product() * size_of::<T>()`. N such views into one
    /// parent share the buffer (no copy) and can be written independently.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is not aligned to
    ///   `align_of::<T>()` (required for the pointer cast in `MemMap`).
    /// - [`Error::InsufficientCapacity`] if the window exceeds the parent
    ///   allocation.
    pub fn view(parent: &MemTensor<T>, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        // Alignment depends on `align_of::<T>()`, not size (a `size_of == 1`,
        // `align_of > 1` type would otherwise skip this and make the `MemMap`
        // pointer cast UB). `is_multiple_of(1)` is always true, so this is a
        // no-op for the common align-1 element types.
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "MemTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let logical: usize = shape.iter().product::<usize>() * elem;
        // The view is positioned relative to this tensor's own window, so a
        // sub-view of a sub-view composes correctly.
        let abs_offset = parent
            .offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let total = parent.data.len() * elem;
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > total {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: total,
            });
        }
        Ok(MemTensor {
            name: parent.name.clone(),
            shape: shape.to_vec(),
            data: Arc::clone(&parent.data),
            offset: abs_offset,
            // A sub-view is the *same* buffer: share the parent's identity so
            // identity-keyed caches (e.g. the GL EGLImage cache) treat the
            // windows as one buffer distinguished by offset, not as unrelated
            // allocations.
            identity: parent.identity.clone(),
        })
    }

    /// Set the byte offset of the logical window into the backing allocation.
    /// Validated against the allocation at `map()` time.
    pub(crate) fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
}

impl<T> TensorTrait<T> for MemTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        // Zero-element shapes (e.g. `[0, num_protos]`) are permitted — they
        // represent genuine "no detections this frame" sentinels produced by
        // the tracker path. DMA-backed storage cannot represent them; Mem
        // storage can (empty Vec).
        let element_count: usize = shape.iter().product();

        let name = name.unwrap_or("mem_tensor").to_owned();
        Ok(MemTensor {
            name,
            shape: shape.to_vec(),
            data: Arc::new(MemBacking::zeroed(element_count)),
            offset: 0,
            identity: crate::BufferIdentity::new(),
        })
    }

    #[cfg(unix)]
    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "MemTensor does not support from_fd".to_owned(),
        ))
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        Err(Error::NotImplemented(
            "MemTensor does not support clone_fd".to_owned(),
        ))
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Mem
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

    fn map(&self) -> Result<TensorMap<T>> {
        let elem = std::mem::size_of::<T>();
        // Validate the offset window fits the backing allocation (mirrors
        // DmaMap::new_internal's bounds + alignment checks).
        let logical = self.shape.iter().product::<usize>() * elem;
        let total = self.data.len() * elem;
        let end = self
            .offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if end > total {
            return Err(Error::InsufficientCapacity {
                needed: end,
                capacity: total,
            });
        }
        // Alignment depends on `align_of::<T>()`, not element size.
        if !self.offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "MemMap: offset {} not aligned to align_of::<T>()={}",
                self.offset,
                std::mem::align_of::<T>()
            )));
        }
        Ok(TensorMap::Mem(MemMap {
            // Keep the backing allocation alive for the map's lifetime so the
            // map stays valid even if the source tensor is dropped first.
            backing: Arc::clone(&self.data),
            offset: self.offset,
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
        }))
    }

    fn buffer_identity(&self) -> &crate::BufferIdentity {
        &self.identity
    }

    fn capacity_bytes(&self) -> usize {
        // Capacity available to this tensor is the allocation minus its window
        // start, so `set_logical_shape` stays bounded for sub-region views.
        // Saturating: `set_plane_offset`/`set_offset` are not validated against
        // the allocation, so an out-of-range offset must report 0 capacity
        // (rejecting any further shape) rather than underflowing to a huge value
        // that `set_logical_shape` would wrongly accept.
        self.data
            .len()
            .saturating_mul(std::mem::size_of::<T>())
            .saturating_sub(self.offset)
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let capacity = self.capacity_bytes();
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        self.shape = shape.to_vec();
        Ok(())
    }
}

#[derive(Debug)]
pub struct MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    /// Keep-alive of the backing allocation. Owning the `Arc` (rather than a
    /// raw pointer) ensures the buffer outlives the map even if the source
    /// tensor is dropped first.
    backing: Arc<MemBacking<T>>,
    /// Byte offset of the mapped window into `backing`.
    offset: usize,
    shape: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Deref for MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> TensorMapTrait<T> for MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        trace!("Unmapping MemMap memory");
    }

    fn as_slice(&self) -> &[T] {
        // SAFETY: the window `[offset, offset + len*size_of::<T>())` was bounds-
        // and alignment-checked in `MemTensor::map()`; `backing` keeps the
        // allocation alive; the base pointer is stable (never reallocated).
        let base = unsafe { (self.backing.base_ptr() as *const u8).add(self.offset) as *const T };
        unsafe { std::slice::from_raw_parts(base, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as `as_slice`, plus: the backing elements are `UnsafeCell`, so
        // `base_ptr()` carries write provenance and interior mutation through
        // the shared `Arc` is sound. Distinct sub-region views address
        // non-overlapping windows (the documented `subview` disjointness
        // contract, matching `DmaMap`), so the `&mut` windows never alias.
        let base = unsafe { (self.backing.base_ptr() as *const u8).add(self.offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(base, self.len()) }
    }
}

impl<T> Drop for MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        self.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TensorMapTrait, TensorMemory, TensorTrait};

    #[test]
    fn test_new_valid_shape() {
        let tensor = MemTensor::<u8>::new(&[2, 3, 4], Some("test")).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.memory(), TensorMemory::Mem);
        assert_eq!(tensor.name(), "test");
        assert_eq!(tensor.size(), 24);
        assert_eq!(tensor.len(), 24);
    }

    #[test]
    fn test_new_empty_shape_error() {
        let result = MemTensor::<u8>::new(&[], Some("test"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidSize(_)));
    }

    #[test]
    fn test_new_zero_dim_is_accepted() {
        // Zero-element shapes are intentionally permitted on Mem-backed
        // tensors: they represent "empty collection" sentinels (e.g.
        // `[0, num_protos]` when a tracker emits no fresh detections).
        // DMA-backed storage still rejects them.
        let result = MemTensor::<u8>::new(&[2, 0, 4], Some("test")).unwrap();
        assert_eq!(result.shape(), &[2, 0, 4]);
        assert_eq!(result.size(), 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_map_read_write() {
        let tensor = MemTensor::<u8>::new(&[2, 3], Some("rw")).unwrap();
        let mut map = tensor.map().unwrap();
        map.as_mut_slice()[0] = 42;
        map.as_mut_slice()[1] = 99;
        assert_eq!(map.as_slice()[0], 42);
        assert_eq!(map.as_slice()[1], 99);
        // Remaining elements should still be zero-initialized.
        assert_eq!(map.as_slice()[2], 0);
    }

    #[test]
    fn test_reshape_compatible() {
        let mut tensor = MemTensor::<u8>::new(&[2, 3], None).unwrap();
        tensor.reshape(&[6]).unwrap();
        assert_eq!(tensor.shape(), &[6]);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn test_reshape_incompatible() {
        let mut tensor = MemTensor::<u8>::new(&[2, 3], None).unwrap();
        let result = tensor.reshape(&[7]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ShapeMismatch(_)));
    }

    #[test]
    fn mem_capacity_and_logical_shape() {
        let mut t = MemTensor::<u8>::with_capacity_bytes(&[480, 640, 3], 921_600, None).unwrap();
        assert_eq!(t.capacity_bytes(), 921_600);
        t.set_logical_shape(&[240, 320, 3]).unwrap();
        assert_eq!(t.shape(), &[240, 320, 3]);
        assert!(t.set_logical_shape(&[480, 640, 4]).is_err());
    }

    #[test]
    fn mem_capacity_saturates_on_oversize_offset() {
        // `set_offset`/`set_plane_offset` are not validated against the
        // allocation, so an out-of-range offset must report 0 capacity (a
        // saturating subtract) instead of underflowing to a huge value that
        // `set_logical_shape` would wrongly accept. The over-range shape is then
        // rejected up front, and `map()` also refuses (bounds check).
        let mut t = MemTensor::<u8>::with_capacity_bytes(&[100], 100, None).unwrap();
        assert_eq!(t.capacity_bytes(), 100);
        t.set_offset(200); // beyond the 100-byte allocation
        assert_eq!(
            t.capacity_bytes(),
            0,
            "capacity must saturate at 0, not underflow"
        );
        assert!(t.set_logical_shape(&[1]).is_err());
        assert!(t.map().is_err());
    }

    #[test]
    fn mem_subview_disjoint_writes_are_independent() {
        // Two disjoint sub-views of one parent are written through `as_mut_slice`
        // (the interior-mutable `MemBacking` write path); each window must see
        // only its own bytes and the parent must reflect both — proving the
        // shared-`Arc` mutable mapping is sound for non-overlapping windows.
        let parent = MemTensor::<u8>::with_capacity_bytes(&[8], 8, None).unwrap();
        let a = MemTensor::view(&parent, 0, &[4]).unwrap();
        let b = MemTensor::view(&parent, 4, &[4]).unwrap();
        a.map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);
        b.map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[5, 6, 7, 8]);
        assert_eq!(a.map().unwrap().as_slice(), &[1, 2, 3, 4]);
        assert_eq!(b.map().unwrap().as_slice(), &[5, 6, 7, 8]);
        assert_eq!(parent.map().unwrap().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn mem_map_rejects_unaligned_offset() {
        // `view()` enforces alignment at creation, but `set_offset` bypasses it,
        // so `map()` keeps a defensive guard: a `u32` window (align 4) starting
        // at byte 2 is in-bounds yet misaligned and must be refused.
        let mut t = MemTensor::<u32>::with_capacity_bytes(&[1], 16, None).unwrap();
        t.set_offset(2); // within the 16-byte allocation but not 4-byte aligned
                         // `TensorMap` isn't `Debug`, so match rather than `unwrap_err()`.
        match t.map() {
            Err(Error::InvalidOperation(_)) => {}
            Err(other) => panic!("expected InvalidOperation, got {other:?}"),
            Ok(_) => panic!("misaligned offset must be rejected by map()"),
        }
    }

    #[test]
    fn mem_backing_debug_reports_len_only() {
        // `MemBacking`'s Debug prints just the element count (never the cell
        // contents), reached via the derived `MemTensor` Debug chain.
        let t = MemTensor::<u8>::with_capacity_bytes(&[4], 4, None).unwrap();
        let s = format!("{t:?}");
        assert!(
            s.contains("MemBacking"),
            "debug should name MemBacking: {s}"
        );
        assert!(s.contains("len"), "debug should report len: {s}");
    }
}
