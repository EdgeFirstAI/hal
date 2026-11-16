// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMemory, TensorTrait,
};
use num_traits::Num;
use std::{cell::UnsafeCell, fmt, sync::Arc};

/// Derive a process-local [`crate::BufferIdentity`] from a host allocation's
/// base address.
fn identity_from_ptr<T>(data: &MemBacking<T>) -> crate::BufferIdentity {
    crate::BufferIdentity::derived(crate::IdentityKind::HostPtr, data.base_ptr() as u64)
}

/// Interior-mutable backing allocation for `MemTensor`.
///
/// The elements live in `UnsafeCell`s: sub-region views, which share one
/// allocation through a cloned `Arc`, each hand out `&mut [T]` over their own
/// window. Distinct `subview` windows must not overlap if mapped mutably at the
/// same time.
enum MemBacking<T> {
    /// HAL-owned allocation; the `Box` frees it when the last `Arc` drops.
    Owned(Box<[UnsafeCell<T>]>),
    /// Externally-owned memory the HAL borrows but never frees. `ptr`/`len`
    /// describe the borrowed region (`len` in elements); `_owner`, when
    /// present, is an opaque keep-alive whose `Drop` releases the source (e.g.
    /// `cudaFreeHost`, a NumPy ref decrement). Dropping this arm runs
    /// `_owner`'s destructor but leaves the foreign pointer untouched.
    Foreign {
        ptr: *mut UnsafeCell<T>,
        len: usize,
        _owner: Option<crate::ForeignOwner>,
    },
}

// SAFETY: a `MemBacking` is mutated only through disjoint windows, per the
// `subview` contract.
unsafe impl<T: Send> Send for MemBacking<T> {}
unsafe impl<T: Sync> Sync for MemBacking<T> {}

impl<T: fmt::Debug> fmt::Debug for MemBacking<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.len();
        f.debug_struct("MemBacking").field("len", &len).finish()
    }
}

impl<T> MemBacking<T> {
    /// Number of elements in the allocation.
    fn len(&self) -> usize {
        match self {
            MemBacking::Owned(cells) => cells.len(),
            MemBacking::Foreign { len, .. } => *len,
        }
    }

    /// Base pointer with write provenance. `UnsafeCell<T>` is
    /// `#[repr(transparent)]`, so the address equals the `T` element's.
    fn base_ptr(&self) -> *mut T {
        match self {
            MemBacking::Owned(cells) => cells.as_ptr() as *mut T,
            MemBacking::Foreign { ptr, .. } => *ptr as *mut T,
        }
    }

    /// Build a zeroed backing of `n` elements by writing `T::zero()` into each
    /// cell.
    fn zeroed(n: usize) -> Self
    where
        T: Num + Clone,
    {
        let cells: Vec<UnsafeCell<T>> = (0..n).map(|_| UnsafeCell::new(T::zero())).collect();
        MemBacking::Owned(cells.into_boxed_slice())
    }

    /// Build a zeroed backing of `n` elements, using the allocator's zeroed
    /// (calloc) path when `T` is one of HAL's primitive numeric element types
    /// (`crate::dtype_of` is `Some`). Any other `T` falls back to the
    /// per-element [`zeroed`](Self::zeroed) loop.
    fn zeroed_fast(n: usize) -> Self
    where
        T: Num + Clone + 'static,
    {
        if n == 0 {
            return MemBacking::Owned(Vec::new().into_boxed_slice());
        }
        if crate::dtype_of::<T>().is_some() {
            // SAFETY: `dtype_of` is `Some` only for HAL's primitive numeric
            // types, whose `T::zero()` is the all-zeros bit pattern, so a
            // zero-filled allocation is a valid `[UnsafeCell<T>; n]`; `n > 0`.
            return unsafe { Self::alloc_zeroed_unchecked(n) };
        }
        Self::zeroed(n)
    }

    /// Allocate `n` zeroed `UnsafeCell<T>` via the global allocator's zeroed
    /// (calloc) path.
    ///
    /// # Safety
    ///
    /// The caller must ensure `T::zero()` is the all-zeros bit pattern (so a
    /// zero-filled block is a valid `[UnsafeCell<T>; n]`) and that `n > 0`.
    unsafe fn alloc_zeroed_unchecked(n: usize) -> Self {
        use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
        // `UnsafeCell<T>` is `#[repr(transparent)]` over `T`, so its layout
        // equals `T`'s.
        let layout = Layout::array::<UnsafeCell<T>>(n).expect("allocation layout overflow");
        // SAFETY: `layout` is non-zero-sized (`n > 0` per the caller
        // contract above).
        let ptr = unsafe { alloc_zeroed(layout) } as *mut UnsafeCell<T>;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        // SAFETY: `ptr` is a freshly-allocated, suitably-aligned block of `n`
        // zeroed `UnsafeCell<T>`; the zero pattern is a valid `T::zero()` per
        // the contract above. `Box::from_raw` over a slice built from the same
        // `(ptr, n)` and element type deallocates with the matching
        // `Layout::array::<UnsafeCell<T>>(n)` on drop.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
        MemBacking::Owned(unsafe { Box::from_raw(slice) })
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
    /// parent's buffer. The allocation is never resized after creation, so the
    /// base pointer is stable for the lifetime of the `Arc`.
    data: Arc<MemBacking<T>>,
    /// Byte offset into `data` where this tensor's logical window begins. `0`
    /// for whole-buffer tensors; non-zero for sub-region views.
    offset: usize,
    identity: crate::BufferIdentity,
}

unsafe impl<T> Send for MemTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for MemTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> Clone for MemTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            data: Arc::clone(&self.data),
            offset: self.offset,
            identity: self.identity.clone(),
        }
    }
}

impl<T> MemTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocate `byte_capacity` bytes and set an initial logical `shape` (whose
    /// byte size must fit the capacity).
    pub(crate) fn with_capacity_bytes(
        shape: &[usize],
        byte_capacity: usize,
        name: Option<&str>,
    ) -> Result<Self>
    where
        T: 'static,
    {
        let elem = std::mem::size_of::<T>();
        let cap_elems = byte_capacity / elem;
        let logical: usize = shape.iter().product();
        if logical > cap_elems {
            return Err(Error::InsufficientCapacity {
                needed: logical * elem,
                capacity: byte_capacity,
            });
        }
        let data = Arc::new(MemBacking::zeroed_fast(cap_elems));
        let identity = identity_from_ptr(&data);
        Ok(MemTensor {
            name: name.unwrap_or("mem_tensor").to_owned(),
            shape: shape.to_vec(),
            data,
            offset: 0,
            identity,
        })
    }

    /// Map exposing `byte_size` bytes via `as_slice()` rather than the
    /// shape-derived logical byte count, for strided tensors whose rows are
    /// padded. The caller (`Tensor::map`) validates `byte_size <=
    /// capacity_bytes()` first; `map_inner` re-checks against the backing
    /// allocation.
    pub(crate) fn map_with_byte_size<'a>(
        &self,
        byte_size: usize,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_inner(Some(byte_size), access)
    }

    /// Shared map constructor. When `byte_size_override` is `Some(bytes)`,
    /// `as_slice()` exposes `bytes / size_of::<T>()` elements; otherwise the
    /// shape-derived logical length is used. Validates that the exposed window
    /// `[offset, offset + exposed)` fits the backing allocation and that
    /// `offset` is aligned to `align_of::<T>()`.
    fn map_inner<'a>(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let elem = std::mem::size_of::<T>();
        let exposed = match byte_size_override {
            Some(bytes) => bytes,
            None => self.shape.iter().product::<usize>() * elem,
        };
        let total = self.data.len() * elem;
        let end = self
            .offset
            .checked_add(exposed)
            .ok_or(Error::InvalidSize(exposed))?;
        if end > total {
            return Err(Error::InsufficientCapacity {
                needed: end,
                capacity: total,
            });
        }
        if !self.offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "MemMap: offset {} not aligned to align_of::<T>()={}",
                self.offset,
                std::mem::align_of::<T>()
            )));
        }
        // The pin keeps the backing allocation alive for the view's lifetime.
        Ok(crate::view::HostView::new(
            self.host_pin(),
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }

    /// Wrap externally-owned memory as a `MemTensor` without taking ownership
    /// of the allocation (no copy). The tensor borrows `[ptr, ptr +
    /// capacity_elems.max(shape.product()) * size_of::<T>())` but reports the
    /// logical `shape` given; `owner`, when `Some`, co-owns the source so the
    /// borrowed memory outlives this tensor (and any [`view`](Self::view) or
    /// map derived from it). The foreign pointer is never freed by this type;
    /// only `owner`'s destructor runs on drop.
    ///
    /// `capacity_elems` records headroom beyond the tight `shape.product()`
    /// footprint; pass `0` for none. A value below the shape's element count is
    /// clamped up to it.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned to `align_of::<T>()`, and valid for
    /// `capacity_elems` elements of `T` (or `shape.product()`, whichever is
    /// larger) for as long as this tensor, and every view or map sharing its
    /// backing, is alive.
    pub(crate) unsafe fn from_foreign_with_capacity(
        ptr: *mut T,
        shape: &[usize],
        capacity_elems: usize,
        owner: Option<crate::ForeignOwner>,
        name: Option<&str>,
    ) -> Self {
        let len: usize = shape.iter().product();
        let data = Arc::new(MemBacking::Foreign {
            ptr: ptr as *mut UnsafeCell<T>,
            len: capacity_elems.max(len),
            _owner: owner,
        });
        let identity = identity_from_ptr(&data);
        MemTensor {
            name: name.unwrap_or("foreign_mem_tensor").to_owned(),
            shape: shape.to_vec(),
            data,
            offset: 0,
            identity,
        }
    }

    /// Host pin over this tensor's logical window. The pin's keepalive is a
    /// clone of the backing `Arc`.
    pub(crate) fn host_pin<'a>(&self) -> crate::pin::HostPin<'a>
    where
        T: 'a,
    {
        let base = unsafe { (self.data.base_ptr() as *mut u8).add(self.offset) };
        // `capacity_bytes()` is relative to this tensor's window start.
        let len = self.capacity_bytes();
        crate::pin::HostPin::new(self.data.clone(), base, len)
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

        // Zero-element shapes (e.g. `[0, num_protos]`) are permitted.
        let element_count: usize = shape.iter().product();

        let name = name.unwrap_or("mem_tensor").to_owned();
        let data = Arc::new(MemBacking::zeroed(element_count));
        let identity = identity_from_ptr(&data);
        Ok(MemTensor {
            name,
            shape: shape.to_vec(),
            data,
            offset: 0,
            identity,
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

    fn map_with<'a>(&self, access: crate::CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.map_inner(None, access)
    }

    fn buffer_identity(&self) -> &crate::BufferIdentity {
        &self.identity
    }

    fn capacity_bytes(&self) -> usize {
        // The allocation minus this tensor's window start, saturating at 0.
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

    /// Zero-copy sub-region view sharing this tensor's allocation and
    /// [`BufferIdentity`](crate::BufferIdentity).
    ///
    /// The view maps `[offset_bytes, offset_bytes + logical_size)` measured from
    /// this tensor's own logical start, where `logical_size = shape.product() *
    /// size_of::<T>()`. N such views into one parent share the buffer (no copy)
    /// and can be written independently.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is not aligned to
    ///   `align_of::<T>()`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the allocation.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        let elem = std::mem::size_of::<T>();
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "MemTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let logical: usize = shape.iter().product::<usize>() * elem;
        // The view is positioned relative to this tensor's own window.
        let abs_offset = self
            .offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let total = self.data.len() * elem;
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
            name: self.name.clone(),
            shape: shape.to_vec(),
            data: Arc::clone(&self.data),
            offset: abs_offset,
            // A sub-view shares the parent's identity.
            identity: self.identity.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;
    use crate::{TensorMemory, TensorTrait};

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
        assert_eq!(map.as_slice()[2], 0);
    }

    #[test]
    fn zeroed_backing_reads_all_zero_across_types() {
        fn assert_all_zero<T>(n: usize)
        where
            T: Num + Clone + fmt::Debug + Send + Sync + Copy + PartialEq + 'static,
        {
            assert!(
                crate::dtype_of::<T>().is_some(),
                "{} should be alloc_zeroed fast-path eligible",
                std::any::type_name::<T>()
            );
            let bytes = n * std::mem::size_of::<T>();
            let t = MemTensor::<T>::with_capacity_bytes(&[n], bytes, Some("z")).unwrap();
            let map = t.map().unwrap();
            assert!(
                map.as_slice().iter().all(|v| *v == T::zero()),
                "zeroed {} backing not all-zero",
                std::any::type_name::<T>()
            );
        }
        let n = 200_000; // spans many pages
        assert_all_zero::<u8>(n);
        assert_all_zero::<u16>(n);
        assert_all_zero::<i32>(n);
        assert_all_zero::<f32>(n);
        assert_all_zero::<f64>(n);
        assert_all_zero::<u8>(0);
        assert_all_zero::<f32>(1);
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
        let parent = MemTensor::<u8>::with_capacity_bytes(&[8], 8, None).unwrap();
        let a = parent.view(0, &[4]).unwrap();
        let b = parent.view(4, &[4]).unwrap();
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
        let mut t = MemTensor::<u32>::with_capacity_bytes(&[1], 16, None).unwrap();
        t.set_offset(2); // within the 16-byte allocation but not 4-byte aligned
        match t.map() {
            Err(Error::InvalidOperation(_)) => {}
            Err(other) => panic!("expected InvalidOperation, got {other:?}"),
            Ok(_) => panic!("misaligned offset must be rejected by map()"),
        }
    }

    #[test]
    fn mem_backing_debug_reports_len_only() {
        let t = MemTensor::<u8>::with_capacity_bytes(&[4], 4, None).unwrap();
        let s = format!("{t:?}");
        assert!(
            s.contains("MemBacking"),
            "debug should name MemBacking: {s}"
        );
        assert!(s.contains("len"), "debug should report len: {s}");
    }

    #[test]
    fn from_foreign_without_owner_borrows_caller_memory() {
        let mut vec: Vec<i32> = vec![0; 4];
        let ptr = vec.as_mut_ptr();
        let t = unsafe { MemTensor::<i32>::from_foreign_with_capacity(ptr, &[4], 0, None, None) };
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.memory(), TensorMemory::Mem);
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&[10, 20, 30, 40]);
        }
        drop(t);
        assert_eq!(vec, vec![10, 20, 30, 40]);
    }

    #[test]
    fn from_foreign_views_share_the_borrowed_buffer() {
        let mut vec: Vec<u8> = vec![0u8; 6];
        let ptr = vec.as_mut_ptr();
        let owner: crate::ForeignOwner = Box::new(vec); // keep the heap buffer alive
        let t = unsafe {
            MemTensor::<u8>::from_foreign_with_capacity(
                ptr,
                &[2, 3],
                0,
                Some(owner),
                Some("foreign"),
            )
        };
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        }
        let v = t.view(3, &[3]).unwrap();
        assert_eq!(v.map().unwrap().as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn from_foreign_owner_drop_fires_on_last_reference() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct DropFlag {
            _buf: Box<[u8]>,
            flag: Arc<AtomicBool>,
        }
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.flag.store(true, Ordering::SeqCst);
            }
        }

        let flag = Arc::new(AtomicBool::new(false));
        let mut buf = vec![7u8; 4].into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        let owner: crate::ForeignOwner = Box::new(DropFlag {
            _buf: buf,
            flag: flag.clone(),
        });
        let t =
            unsafe { MemTensor::<u8>::from_foreign_with_capacity(ptr, &[4], 0, Some(owner), None) };

        let m = t.map().unwrap();
        assert_eq!(m.as_slice()[0], 7);
        drop(t);
        assert!(
            !flag.load(Ordering::SeqCst),
            "owner must stay alive while a map still references the backing"
        );
        drop(m);
        assert!(
            flag.load(Ordering::SeqCst),
            "owner Drop (the free) must fire on the last reference drop"
        );
    }

    #[test]
    fn two_mem_tensors_do_not_share_an_identity() {
        let a = MemTensor::<u8>::new(&[8], None).unwrap();
        let b = MemTensor::<u8>::new(&[8], None).unwrap();
        assert_eq!(a.buffer_identity().kind(), crate::IdentityKind::HostPtr);
        assert_ne!(a.buffer_identity().id(), b.buffer_identity().id());
    }

    fn expect_insufficient<V: fmt::Debug>(r: Result<V>, want: (usize, usize)) {
        match r {
            Err(Error::InsufficientCapacity { needed, capacity }) => {
                assert_eq!((needed, capacity), want);
            }
            other => panic!("expected InsufficientCapacity{want:?}, got {other:?}"),
        }
    }

    #[test]
    fn with_capacity_bytes_counts_bytes_of_a_multi_byte_element() {
        let t = MemTensor::<u32>::with_capacity_bytes(&[4], 16, None).expect("exact fit");
        assert_eq!(t.capacity_bytes(), 16);
        assert_eq!(t.shape(), &[4]);
        expect_insufficient(
            MemTensor::<u32>::with_capacity_bytes(&[5], 16, None),
            (20, 16),
        );
    }

    #[test]
    fn map_bounds_an_offset_window_in_bytes() {
        let mut t = MemTensor::<u32>::with_capacity_bytes(&[4], 16, None).unwrap();
        t.set_offset(4);
        expect_insufficient(t.map(), (20, 16));
        t.set_logical_shape(&[3])
            .expect("the window's own capacity");
        assert_eq!(t.map().expect("fits after the offset").as_slice().len(), 3);
    }

    #[test]
    fn reshape_and_logical_shape_count_bytes_of_a_multi_byte_element() {
        let mut t = MemTensor::<u32>::with_capacity_bytes(&[2, 3], 24, None).unwrap();
        t.reshape(&[6]).expect("same element count");
        assert_eq!(t.shape(), &[6]);

        let mut t = MemTensor::<u32>::with_capacity_bytes(&[4], 16, None).unwrap();
        expect_insufficient(t.set_logical_shape(&[5]), (20, 16));
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn view_bounds_the_window_in_bytes() {
        let t = MemTensor::<u32>::with_capacity_bytes(&[4], 16, None).unwrap();
        expect_insufficient(t.view(8, &[3]), (20, 16));
        let v = t.view(4, &[3]).expect("exact fit");
        assert_eq!(v.map().unwrap().as_slice().len(), 3);
    }
}
