// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    TensorMemory, TensorTrait,
};
use num_traits::Num;
use std::{cell::UnsafeCell, fmt, sync::Arc};

/// Derive a [`crate::BufferIdentity`] from a host allocation's base address.
///
/// Process-local, which is exactly `Mem`'s documented scope: two views of the
/// same allocation within this process see the same pointer, but the key is
/// never looked at across a process boundary, so pointer reuse after `free`
/// carries no cross-process ABA hazard the way a reused inode would.
fn identity_from_ptr<T>(data: &MemBacking<T>) -> crate::BufferIdentity {
    crate::BufferIdentity::derived(crate::IdentityKind::HostPtr, data.base_ptr() as u64)
}

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
enum MemBacking<T> {
    /// HAL-owned allocation; the `Box` frees it when the last `Arc` drops.
    Owned(Box<[UnsafeCell<T>]>),
    /// Externally-owned memory the HAL borrows but never frees. `ptr`/`len`
    /// describe the borrowed region (`len` in elements); `_owner`, when
    /// present, is an opaque keep-alive whose `Drop` releases the source (e.g.
    /// `cudaFreeHost`, a NumPy ref decrement). Dropping this arm runs
    /// `_owner`'s destructor but leaves the foreign pointer untouched. The
    /// elements are still `UnsafeCell` so the shared-`Arc` write-provenance
    /// contract documented above holds for foreign buffers too.
    Foreign {
        ptr: *mut UnsafeCell<T>,
        len: usize,
        _owner: Option<crate::ForeignOwner>,
    },
}

// SAFETY: `UnsafeCell<T>` is `!Sync`, but a `MemBacking` is only mutated through
// the disjoint-window contract documented above (and was already guarded by the
// `unsafe impl Send/Sync for MemTensor`). The cells add no thread-safety hazard
// beyond what `MemTensor` already asserts; they only fix pointer provenance.
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

    /// Base pointer with write provenance (the cells are `UnsafeCell`, so
    /// interior mutation through the shared `Arc` is sound). `UnsafeCell<T>`
    /// is `#[repr(transparent)]`, so the address equals the `T` element's.
    fn base_ptr(&self) -> *mut T {
        match self {
            MemBacking::Owned(cells) => cells.as_ptr() as *mut T,
            MemBacking::Foreign { ptr, .. } => *ptr as *mut T,
        }
    }

    /// Build a zeroed backing of `n` elements by writing `T::zero()` into each
    /// cell. Type-agnostic; used by the `TensorTrait::new` path. The
    /// image/capacity path uses [`zeroed_fast`](Self::zeroed_fast) for the
    /// `alloc_zeroed` win.
    fn zeroed(n: usize) -> Self
    where
        T: Num + Clone,
    {
        let cells: Vec<UnsafeCell<T>> = (0..n).map(|_| UnsafeCell::new(T::zero())).collect();
        MemBacking::Owned(cells.into_boxed_slice())
    }

    /// Build a zeroed backing of `n` elements, using the allocator's zeroed
    /// (calloc) path when `T` is one of HAL's primitive numeric element types.
    ///
    /// For those types `T::zero()` is the all-zeros bit pattern, so the kernel
    /// can hand back lazily-zeroed pages instead of eagerly writing every
    /// element — a `(0..n).map(UnsafeCell::new).collect()` forces a full memset
    /// and commits every page up front. This backs every `Tensor::image(.., Mem)`
    /// and the per-call image intermediates the CPU converter allocates. Any
    /// other `T` falls back to the per-element [`zeroed`](Self::zeroed) loop.
    ///
    /// `crate::dtype_of` is the single source for "is `T` a HAL numeric
    /// primitive" — a safe `TypeId` allowlist, with no inspection of
    /// (possibly padded/uninitialised) bytes.
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
        // equals `T`'s; allocating an array of it round-trips on `Box` drop.
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
    ///
    /// Used for image tensors with a 64-byte-aligned row stride that may
    /// exceed `shape.product() * sizeof(T)`: production callers allocate via
    /// `Tensor::image()` (which routes through this constructor), and tests use
    /// it directly when they need spare capacity (e.g. an offset window past
    /// the logical end).
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
    /// shape-derived logical byte count, for self-allocated strided tensors
    /// whose rows are padded. The caller (`Tensor::map`) validates
    /// `byte_size <= capacity_bytes()` first; `map_inner` re-checks against the
    /// backing allocation. `Mem` is always HAL-owned, so the backing covers the
    /// full requested range.
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
    /// `as_slice()` exposes the full padded allocation (`bytes / size_of::<T>()`
    /// elements); otherwise the shape-derived logical length is used. Validates
    /// that the exposed window `[offset, offset + exposed)` fits the backing
    /// allocation and that `offset` is aligned to `align_of::<T>()` (mirrors
    /// `DmaMap`'s bounds + alignment checks).
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
        // Alignment depends on `align_of::<T>()`, not element size.
        if !self.offset.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "MemMap: offset {} not aligned to align_of::<T>()={}",
                self.offset,
                std::mem::align_of::<T>()
            )));
        }
        // The pin keeps the backing allocation alive for the view's lifetime,
        // so the view stays valid even if the source tensor drops first --
        // previously MemMap's own Arc::clone of `data`.
        Ok(crate::view::HostView::new(
            self.host_pin(),
            self.shape.clone(),
            byte_size_override,
            access,
        ))
    }

    /// Wrap externally-owned memory as a `MemTensor` without taking ownership
    /// of the allocation (no copy). The tensor borrows
    /// `[ptr, ptr + capacity_elems.max(shape.product()) * size_of::<T>())`
    /// but reports the logical `shape` given; `owner`, when `Some`, co-owns
    /// the *source* so the borrowed memory outlives this tensor (and any
    /// [`view`](Self::view) or map derived from it). The foreign pointer is
    /// never freed by this type — only `owner`'s destructor runs on drop.
    ///
    /// `capacity_elems` records headroom beyond the tight `shape.product()`
    /// footprint -- pass `0` for none (the ordinary case: exactly
    /// `shape.product()` elements). This is the consumer half of the
    /// cross-package capsule protocol's `capacity` field
    /// (`Tensor::from_foreign_with_capacity`'s docs): a producer's pool
    /// tensor, or one padded to a decoder's pitch/MCU alignment, is larger
    /// than the shape it currently reports, and without recording that here
    /// the imported alias would be clamped to today's shape and unable to
    /// grow back into memory the producer actually has. `capacity_elems`
    /// below the shape's element count is clamped up to it, never down.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned to `align_of::<T>()`, and valid for
    /// `capacity_elems` elements of `T` (or `shape.product()`, whichever is
    /// larger) for as long as this tensor — and every view/map sharing its
    /// backing — is alive. Passing an `owner` that co-owns the source (e.g.
    /// an `Arc` whose `Drop` frees it) is the supported way to uphold that
    /// lifetime contract.
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

    /// Set the byte offset of the logical window into the backing allocation.
    /// Validated against the allocation at `map()` time.
    /// Host pin over this tensor's logical window.
    ///
    /// `Mem` needs no mapping step: the backing allocation is never resized
    /// after creation, so `base_ptr` is already stable for the life of the
    /// `Arc`. Cloning that `Arc` IS the keepalive, which is why pinning host
    /// memory costs an refcount bump and nothing else.
    pub(crate) fn host_pin<'a>(&self) -> crate::pin::HostPin<'a>
    where
        T: 'a,
    {
        let base = unsafe { (self.data.base_ptr() as *mut u8).add(self.offset) };
        // capacity_bytes() is ALREADY relative to this tensor's window start
        // ("the allocation minus its window start"), so subtracting `offset`
        // again would zero a subview whose window is exactly its capacity.
        let len = self.capacity_bytes();
        crate::pin::HostPin::new(self.data.clone(), base, len)
    }

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
    ///   `align_of::<T>()` (required for the pointer cast in `MemMap`).
    /// - [`Error::InsufficientCapacity`] if the window exceeds the allocation.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
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
            // A sub-view is the *same* buffer: share the parent's identity so
            // identity-keyed caches (e.g. the GL EGLImage cache) treat the
            // windows as one buffer distinguished by offset, not as unrelated
            // allocations.
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
    fn zeroed_backing_reads_all_zero_across_types() {
        // Every HAL primitive numeric T is alloc_zeroed-fast-path eligible
        // (`dtype_of` is `Some`). A large allocation spanning many pages, built
        // through the capacity path (`with_capacity_bytes` → `zeroed_fast`),
        // must still read back entirely zero.
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
        // n == 0 and n == 1 cover the empty-box and single-element edges.
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

    #[test]
    fn from_foreign_without_owner_borrows_caller_memory() {
        // No owner: the caller guarantees the buffer outlives the tensor. Writes
        // through the tensor must land in the caller's own allocation (zero-copy).
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
        // The writes are visible in the caller's Vec — same physical buffer.
        assert_eq!(vec, vec![10, 20, 30, 40]);
    }

    #[test]
    fn from_foreign_views_share_the_borrowed_buffer() {
        // A `view()` shares the backing `Arc`, so it addresses the same foreign
        // memory as the parent (with an offset) — the zero-copy export path.
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
        // Second row [4,5,6] via a sub-region view of the same buffer.
        let v = t.view(3, &[3]).unwrap();
        assert_eq!(v.map().unwrap().as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn from_foreign_owner_drop_fires_on_last_reference() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // The owner co-owns the source; its `Drop` (the "free") must fire exactly
        // once, after the last tensor/map sharing the backing is gone — regardless
        // of whether the source tensor or a derived map outlives the other.
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

        // `map()` returns an owned `Arc` clone, so the map outlives the tensor.
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
        // A `HostPtr` key derived from a shared counter would be
        // indistinguishable from one derived from a real, distinct
        // allocation -- this pins the derivation to the actual base pointer
        // of each backing.
        let a = MemTensor::<u8>::new(&[8], None).unwrap();
        let b = MemTensor::<u8>::new(&[8], None).unwrap();
        assert_eq!(a.buffer_identity().kind(), crate::IdentityKind::HostPtr);
        assert_ne!(a.buffer_identity().id(), b.buffer_identity().id());
    }
}
