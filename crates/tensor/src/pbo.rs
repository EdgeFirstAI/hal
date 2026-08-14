// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{Error, Result},
    BufferIdentity, TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
};
use log::trace;
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex},
};

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
///
/// # Safety
///
/// Implementations must ensure:
/// - `map_buffer` returns a valid, aligned pointer to `size` bytes of
///   CPU-accessible memory that remains valid until `unmap_buffer` is called.
/// - `unmap_buffer` invalidates the pointer and releases the mapping.
/// - `delete_buffer` frees the GL buffer resources.
pub unsafe trait PboOps: Send + Sync {
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

/// Opaque handle to a PBO's GL resources.
struct PboHandle {
    ops: Arc<dyn PboOps>,
    buffer_id: u32,
    size: usize,
    map_state: Mutex<MapState>,
    /// Wakes threads waiting out an in-flight GL map/unmap. Needed because
    /// `map_state` is deliberately NOT held across `PboOps` calls — see
    /// [`PboHandle::acquire_map`].
    map_cv: Condvar,
}

/// CPU-map state of a PBO's single GL buffer.
///
/// A GL buffer has exactly one mapping at a time, so these describe that
/// one mapping's ownership rather than per-[`PboMap`] state:
///
/// * `Unmapped` — no CPU mapping exists.
/// * `Exclusive` — one writable map holds it. Nothing else may map until
///   that map drops.
/// * `Shared` — `readers` read-only maps over ONE mapping, unmapped when
///   the last of them drops.
///
/// Read sharing exists because tiled (SAHI) pre-processing has several
/// worker threads convert different crops of the SAME source tensor
/// concurrently, each taking `map_read()`. Read-only holders cannot observe
/// one another's writes because there are none, so one mapping serves them
/// all. Under a single-map rule every reader but one failed with
/// [`Error::PboMapped`], which left callers no option but to serialize the
/// workers that share a source.
enum MapState {
    Unmapped,
    /// A thread is inside `PboOps::map_buffer` for this buffer right now.
    /// Others must wait it out rather than start a second GL map.
    Mapping,
    Exclusive,
    Shared {
        readers: usize,
        ptr: PboPtr,
    },
    /// A thread is inside `PboOps::unmap_buffer` for this buffer right now.
    Unmapping,
}

impl PboHandle {
    /// Acquire a CPU mapping of this buffer, returning the base pointer.
    ///
    /// Read-only acquisitions join an existing read-only mapping (bumping
    /// its refcount) or create one. Writable acquisitions are exclusive.
    /// Returns [`Error::PboMapped`] whenever the request cannot share what
    /// is already held: anything against an exclusive map, or a writable
    /// request against readers.
    ///
    /// # Locking
    ///
    /// `map_state` is NEVER held across a [`PboOps`] call. Those calls are
    /// blocking round-trips to the GL thread, and the work the GL thread
    /// runs (a convert, say) maps tensors itself — so a thread that waited
    /// on GL while holding this mutex would deadlock against the very GL
    /// thread it is waiting for. The in-flight GL call is published as
    /// [`MapState::Mapping`] / [`MapState::Unmapping`] instead, and other
    /// threads wait on `map_cv` until it resolves.
    fn acquire_map(&self, writable: bool) -> Result<PboPtr> {
        loop {
            let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
            // Wait out any GL map/unmap another thread is mid-way through.
            while matches!(*guard, MapState::Mapping | MapState::Unmapping) {
                guard = self
                    .map_cv
                    .wait(guard)
                    .expect("PBO map state mutex poisoned");
            }
            match &mut *guard {
                MapState::Shared { readers, ptr } if !writable => {
                    *readers += 1;
                    return Ok(PboPtr(ptr.0));
                }
                MapState::Shared { .. } | MapState::Exclusive => return Err(Error::PboMapped),
                // Ruled out by the wait above; re-check rather than assume.
                MapState::Mapping | MapState::Unmapping => continue,
                MapState::Unmapped => *guard = MapState::Mapping,
            }
            drop(guard);

            // No lock held here — see this function's Locking note.
            let mapped = self.ops.map_buffer(self.buffer_id, self.size);
            let base = match mapped {
                Ok(mapping) => NonNull::new(mapping.ptr as *mut c_void),
                Err(e) => {
                    self.finish_map(None, writable);
                    return Err(e);
                }
            };
            let Some(base) = base else {
                // GL reported success but handed back a null pointer. It
                // considers the buffer mapped, so release it rather than
                // strand it mapped forever.
                //
                // Order matters: the cleanup unmap runs while this thread
                // still holds the `Mapping` claim, so waiters stay parked
                // and cannot start a fresh `map_buffer` that would race
                // this unmap on the GL queue. Only once it has completed is
                // `Unmapped` published.
                let _ = self.ops.unmap_buffer(self.buffer_id);
                self.finish_map(None, writable);
                return Err(Error::InvalidSize(self.size));
            };
            self.finish_map(Some(base), writable);
            return Ok(PboPtr(base));
        }
    }

    /// Publish the outcome of a GL map this thread had claimed via
    /// [`MapState::Mapping`], waking anyone who waited on it. `None` means
    /// the map failed and the buffer is unmapped again.
    fn finish_map(&self, base: Option<NonNull<c_void>>, writable: bool) {
        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        *guard = match base {
            None => MapState::Unmapped,
            Some(_) if writable => MapState::Exclusive,
            Some(base) => MapState::Shared {
                readers: 1,
                ptr: PboPtr(base),
            },
        };
        self.map_cv.notify_all();
    }

    /// Release one acquisition, unmapping the GL buffer once the last
    /// holder is gone. Same locking rule as [`Self::acquire_map`]: the GL
    /// unmap runs with no lock held.
    fn release_map(&self) {
        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        let last_holder = match &mut *guard {
            MapState::Shared { readers, .. } => {
                *readers = readers.saturating_sub(1);
                *readers == 0
            }
            MapState::Exclusive => true,
            // Nothing held — a release with no matching acquire. (A caller
            // holding an acquisition cannot observe Mapping/Unmapping:
            // only the last holder enters Unmapping, and it is this one.)
            MapState::Unmapped | MapState::Mapping | MapState::Unmapping => false,
        };
        if !last_holder {
            return;
        }
        *guard = MapState::Unmapping;
        drop(guard);

        trace!("Unmapping PBO buffer_id={}", self.buffer_id);
        if let Err(e) = self.ops.unmap_buffer(self.buffer_id) {
            log::warn!("Failed to unmap PBO buffer {}: {e}", self.buffer_id);
        }

        let mut guard = self.map_state.lock().expect("PBO map state mutex poisoned");
        *guard = MapState::Unmapped;
        self.map_cv.notify_all();
    }

    /// Whether any CPU mapping is currently held or being established.
    fn is_mapped(&self) -> bool {
        !matches!(
            *self.map_state.lock().expect("PBO map state mutex poisoned"),
            MapState::Unmapped
        )
    }
}

impl Drop for PboHandle {
    fn drop(&mut self) {
        self.ops.delete_buffer(self.buffer_id);
    }
}

/// A tensor backed by an OpenGL Pixel Buffer Object.
pub struct PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub shape: Vec<usize>,
    handle: Arc<PboHandle>,
    identity: BufferIdentity,
    /// Byte offset of this tensor's window into the shared GL buffer. Non-zero
    /// only for sub-views (`view`/`batch`), which share the `Arc<PboHandle>` and
    /// `BufferIdentity` and address a sub-region by this offset — mirrors
    /// `DmaTensor::mmap_offset` / `IoSurfaceTensor::view_offset`.
    pub(crate) view_offset: usize,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PboTensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("buffer_id", &self.handle.buffer_id)
            .field("size", &self.handle.size)
            .finish()
    }
}

unsafe impl<T> Send for PboTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for PboTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Create a new PBO tensor from an already-allocated GL buffer.
    ///
    /// Called by the image crate after creating the PBO on the GL thread.
    /// Users should not call this directly — use `ImageProcessor::create_image()`.
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if `size` does not equal
    /// `shape.iter().product::<usize>() * std::mem::size_of::<T>()`.
    /// Returns `Error::InvalidSize` if `size` is zero.
    pub fn from_pbo(
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        name: Option<&str>,
        ops: Arc<dyn PboOps>,
    ) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }
        let expected = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        // Allow `size >= expected`: PBOs allocated with a 64-byte-aligned row
        // stride may be larger than the shape product.  Reject only if the
        // allocation is strictly smaller than the logical content.
        if size < expected {
            return Err(Error::ShapeMismatch(format!(
                "PBO size {size} is smaller than shape {shape:?} * sizeof({}) = {expected}",
                std::any::type_name::<T>(),
            )));
        }
        let name = name.unwrap_or("pbo_tensor").to_owned();
        Ok(Self {
            name,
            shape: shape.to_vec(),
            handle: Arc::new(PboHandle {
                ops,
                buffer_id,
                size,
                map_state: Mutex::new(MapState::Unmapped),
                map_cv: Condvar::new(),
            }),
            identity: BufferIdentity::new(),
            view_offset: 0,
            _marker: PhantomData,
        })
    }

    /// Returns the GL buffer ID for this PBO.
    pub fn buffer_id(&self) -> u32 {
        self.handle.buffer_id
    }

    /// Returns true unless the PBO is fully unmapped.
    ///
    /// That covers an exclusive map, one or more read-only holders, AND a
    /// map or unmap another thread currently has in flight — during those
    /// windows the GL buffer is (or is about to be) mapped, and no other
    /// mapping may begin, so reporting `false` would be misleading. Use it
    /// as "is this buffer free for GL operations?", not as "does a CPU
    /// pointer exist right now".
    pub fn is_mapped(&self) -> bool {
        self.handle.is_mapped()
    }
}

impl<T> TensorTrait<T> for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "PboTensor cannot be created directly — use ImageProcessor::create_image()".to_owned(),
        ))
    }

    #[cfg(unix)]
    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "PboTensor does not support from_fd".to_owned(),
        ))
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        Err(Error::NotImplemented(
            "PboTensor does not support clone_fd".to_owned(),
        ))
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Pbo
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
        if new_size != self.handle.size {
            return Err(Error::ShapeMismatch(format!(
                "Cannot reshape incompatible shape: {:?} to {:?}",
                self.shape, shape
            )));
        }
        self.shape = shape.to_vec();
        self.view_offset = 0;
        Ok(())
    }

    fn capacity_bytes(&self) -> usize {
        self.handle.size
    }

    /// Capacity-based reconfigure (mirrors Mem/Shm/DMA/IOSurface): allow any
    /// shape whose byte size fits the PBO allocation, so an oversized reusable
    /// pool can be `configure_image`d to a smaller image. Without this PBO fell
    /// back to the strict-`reshape` default and rejected pool reuse.
    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if needed > self.handle.size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.handle.size,
            });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    fn map_with(&self, access: crate::CpuAccess) -> Result<TensorMap<T>> {
        self.map_internal(None, access)
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    /// Zero-copy sub-region view sharing this PBO's GL buffer (via the
    /// `Arc<PboHandle>`) and [`BufferIdentity`], positioned at `offset_bytes`
    /// from this tensor's own window with logical `shape`. The GL backend keys
    /// the import on the shared identity and addresses the window via
    /// `glViewport` / the staged copy; a CPU map adds `view_offset` to the
    /// mapped base. Mirrors [`DmaTensor::view`](crate::TensorTrait::view).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOperation`] if `offset_bytes` is mis-aligned for `T`.
    /// - [`Error::InsufficientCapacity`] if the window exceeds the allocation.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "PboTensor::view: offset {offset_bytes} not aligned to align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .view_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let logical = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > self.handle.size {
            return Err(Error::InsufficientCapacity {
                needed,
                capacity: self.handle.size,
            });
        }
        Ok(Self {
            name: self.name.clone(),
            shape: shape.to_vec(),
            handle: Arc::clone(&self.handle),
            identity: self.identity.clone(),
            view_offset: abs_offset,
            _marker: PhantomData,
        })
    }
}

impl<T> PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Map the PBO so `as_slice()` exposes the full padded buffer (`byte_size`
    /// bytes) rather than the shape-derived logical count. Mirrors
    /// [`DmaTensor::map_with_byte_size`]: a CPU producer (e.g. the JPEG decoder)
    /// or a strided convert source iterates rows via `effective_row_stride()`
    /// without running past the slice. Crate-private; the only caller is
    /// `Tensor::map()`, which already checks `byte_size <= capacity_bytes()`.
    pub(crate) fn map_with_byte_size(
        &self,
        byte_size: usize,
        access: crate::CpuAccess,
    ) -> Result<TensorMap<T>> {
        self.map_internal(Some(byte_size), access)
    }

    fn map_internal(
        &self,
        byte_size_override: Option<usize>,
        access: crate::CpuAccess,
    ) -> Result<TensorMap<T>> {
        // Always map the full GL allocation (`handle.size`); the slice length is
        // narrowed by `byte_size_override` (or the logical shape) at access time.
        let writable = access.writes();
        let ptr = self.handle.acquire_map(writable)?;
        Ok(TensorMap::Pbo(PboMap {
            ptr: Arc::new(Mutex::new(ptr)),
            shape: self.shape.clone(),
            handle: Arc::clone(&self.handle),
            byte_size_override,
            view_offset: self.view_offset,
            writable,
            released: false,
            _marker: PhantomData,
        }))
    }
}

// -- PboMap --

#[derive(Debug)]
struct PboPtr(NonNull<c_void>);

impl Deref for PboPtr {
    type Target = NonNull<c_void>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for PboPtr {}

pub struct PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    ptr: Arc<Mutex<PboPtr>>,
    shape: Vec<usize>,
    handle: Arc<PboHandle>,
    /// Optional override for `as_slice().len()`. `None` → `shape.product()`
    /// elements (logical view). `Some(bytes)` → `bytes / sizeof(T)` elements,
    /// exposing the full padded GL allocation so callers can iterate rows via
    /// `row_stride` (set by `Tensor::map()` for strided PBO tensors). Mirrors
    /// `DmaMap::byte_size_override`.
    byte_size_override: Option<usize>,
    /// Byte offset of the sub-view window into the mapped GL buffer. `as_slice`
    /// advances the base pointer by this many bytes before exposing the slice.
    view_offset: usize,
    /// Whether mutable access is permitted (`map_read()` maps are not).
    /// The GL mapping itself stays MAP_READ|MAP_WRITE (bit narrowing is a
    /// follow-up); this enforces the API contract uniformly.
    writable: bool,
    /// Whether this map has already released its acquisition. `unmap()` is
    /// public on the trait AND called from `Drop`, so without this an
    /// explicit unmap followed by the drop would release twice — which,
    /// now that read-only maps are refcounted, would drop a *sibling*
    /// reader's mapping out from under it rather than being the harmless
    /// double-store it used to be.
    released: bool,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PboMap")
            .field("shape", &self.shape)
            .field("buffer_id", &self.handle.buffer_id)
            .finish()
    }
}

impl<T> TensorMapTrait<T> for PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        self.handle.release_map();
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock PboMap pointer");
        let base = unsafe { (ptr.as_ptr() as *const u8).add(self.view_offset) as *const T };
        unsafe { std::slice::from_raw_parts(base, self.slice_len_elems()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        crate::assert_map_writable(self.writable, "Pbo");
        let ptr = self.ptr.lock().expect("Failed to lock PboMap pointer");
        let base = unsafe { (ptr.as_ptr() as *mut u8).add(self.view_offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(base, self.slice_len_elems()) }
    }
}

impl<T> PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    /// Number of `T` elements exposed by `as_slice()`. Honours
    /// `byte_size_override` (the full padded GL allocation) when set; otherwise
    /// the shape-derived logical count. Mirrors `DmaMap::slice_len_elems`.
    fn slice_len_elems(&self) -> usize {
        match self.byte_size_override {
            Some(bytes) => bytes / std::mem::size_of::<T>(),
            None => self.shape.iter().product(),
        }
    }
}

impl<T> Deref for PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Drop for PboMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        self.unmap();
    }
}

impl<T> Clone for PboTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            shape: self.shape.clone(),
            handle: Arc::clone(&self.handle),
            identity: self.identity.clone(),
            view_offset: self.view_offset,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock PboOps that uses a Vec<u8> as backing storage instead of GL.
    ///
    /// Counts map/unmap calls so tests can assert that N read-only holders
    /// share ONE GL mapping rather than each taking their own.
    struct MockPboOps {
        storage: Mutex<Vec<u8>>,
        maps: std::sync::atomic::AtomicUsize,
        unmaps: std::sync::atomic::AtomicUsize,
    }

    impl MockPboOps {
        fn new(size: usize) -> Arc<Self> {
            Arc::new(Self {
                storage: Mutex::new(vec![0u8; size]),
                maps: std::sync::atomic::AtomicUsize::new(0),
                unmaps: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn map_count(&self) -> usize {
            self.maps.load(std::sync::atomic::Ordering::Acquire)
        }

        fn unmap_count(&self) -> usize {
            self.unmaps.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    // SAFETY: the returned pointer addresses a `Vec<u8>` that is allocated
    // once in `new` and never resized, so it stays valid for the whole life
    // of the `MockPboOps` — which outlives every map handed out, since the
    // tensor holds an `Arc` to it. The storage mutex is NOT what keeps the
    // pointer alive (`map_buffer` releases it before returning); it only
    // guards the length assertion.
    unsafe impl PboOps for MockPboOps {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            self.maps.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            let storage = self.storage.lock().expect("lock");
            assert_eq!(storage.len(), size);
            Ok(PboMapping {
                ptr: storage.as_ptr() as *mut u8,
                size,
            })
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            self.unmaps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            Ok(())
        }

        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    #[test]
    fn test_pbo_tensor_create_and_metadata() {
        let ops = MockPboOps::new(24);
        let tensor = PboTensor::<u8>::from_pbo(42, 24, &[2, 3, 4], Some("test_pbo"), ops).unwrap();
        assert_eq!(tensor.memory(), TensorMemory::Pbo);
        assert_eq!(tensor.name(), "test_pbo");
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.buffer_id(), 42);
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_map_write_read() {
        let ops = MockPboOps::new(12);
        let tensor = PboTensor::<u8>::from_pbo(1, 12, &[3, 4], Some("rw_test"), ops).unwrap();
        {
            let mut map = tensor.map().expect("map should succeed");
            assert_eq!(map.shape(), &[3, 4]);
            assert!(tensor.is_mapped());
            map.as_mut_slice().fill(0xAB);
            assert!(map.as_slice().iter().all(|&b| b == 0xAB));
        }
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_double_map_fails() {
        let ops = MockPboOps::new(8);
        let tensor = PboTensor::<u8>::from_pbo(2, 8, &[8], None, ops).unwrap();
        let _map1 = tensor.map().expect("first map should succeed");
        assert!(tensor.is_mapped());
        let result = tensor.map();
        assert!(result.is_err(), "second map while mapped should fail");
    }

    /// Mock whose `map_buffer` reports success but hands back a null
    /// pointer — the one path where a claimed map has to be released again.
    #[derive(Default)]
    struct NullMappingOps {
        unmaps: std::sync::atomic::AtomicUsize,
    }

    impl NullMappingOps {
        fn unmap_count(&self) -> usize {
            self.unmaps.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    // SAFETY: never produces a usable pointer, so no caller can dereference
    // one; the mapping it returns is rejected before reaching a slice.
    unsafe impl PboOps for NullMappingOps {
        fn map_buffer(&self, _buffer_id: u32, size: usize) -> Result<PboMapping> {
            Ok(PboMapping {
                ptr: std::ptr::null_mut(),
                size,
            })
        }

        fn unmap_buffer(&self, _buffer_id: u32) -> Result<()> {
            self.unmaps
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            Ok(())
        }

        fn delete_buffer(&self, _buffer_id: u32) {}
    }

    /// A map that "succeeds" with a null pointer leaves GL considering the
    /// buffer mapped, so it must be released and the state returned to
    /// unmapped — otherwise the buffer is stranded and every later map
    /// fails. The release happens before `Unmapped` is published, so no
    /// waiter can start a map that races the cleanup unmap.
    #[test]
    fn null_mapping_releases_the_buffer_and_resets_state() {
        let ops = Arc::new(NullMappingOps::default());
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(12, 8, &[8], None, dyn_ops).unwrap();

        let Err(err) = tensor.map_read() else {
            panic!("a null mapping must not succeed");
        };
        assert!(
            matches!(err, Error::InvalidSize(_)),
            "unexpected error: {err:?}"
        );
        assert_eq!(
            ops.unmap_count(),
            1,
            "GL still holds the buffer mapped — the failed map must release it"
        );
        assert!(
            !tensor.is_mapped(),
            "state must return to Unmapped so later maps can proceed"
        );

        // Not stranded: the buffer can be claimed again.
        let Err(err) = tensor.map_read() else {
            panic!("still null, so it must still fail");
        };
        assert!(matches!(err, Error::InvalidSize(_)));
        assert_eq!(ops.unmap_count(), 2, "and released again");
    }

    /// Several read-only holders share ONE GL mapping. This is what lets
    /// tiled pre-processing run several workers over one source tensor;
    /// before read sharing, every reader after the first failed.
    #[test]
    fn read_maps_share_one_gl_mapping() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(9, 8, &[8], None, dyn_ops).unwrap();

        let r1 = tensor.map_read().expect("first read map");
        let r2 = tensor
            .map_read()
            .expect("second read map must share, not fail");
        let r3 = tensor
            .map_read()
            .expect("third read map must share, not fail");

        assert_eq!(ops.map_count(), 1, "one GL mapping serves every reader");
        assert_eq!(ops.unmap_count(), 0, "nothing unmapped while readers live");
        assert!(tensor.is_mapped());
        // All readers address the same bytes.
        assert_eq!(r1.as_slice(), r2.as_slice());
        assert_eq!(r2.as_slice(), r3.as_slice());

        drop(r1);
        drop(r2);
        assert_eq!(
            ops.unmap_count(),
            0,
            "the mapping outlives every reader but the last"
        );
        assert!(tensor.is_mapped());

        drop(r3);
        assert_eq!(ops.unmap_count(), 1, "last reader out unmaps, exactly once");
        assert!(!tensor.is_mapped());
    }

    /// Read sharing must not weaken write exclusion in either direction.
    #[test]
    fn writers_and_readers_still_exclude_each_other() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(10, 8, &[8], None, dyn_ops).unwrap();

        let reader = tensor.map_read().expect("read map");
        assert!(
            tensor.map().is_err(),
            "a writer must not join an existing reader set"
        );
        drop(reader);

        let writer = tensor.map().expect("write map after readers drained");
        assert!(
            tensor.map_read().is_err(),
            "a reader must not join an exclusive writer"
        );
        assert!(
            tensor.map().is_err(),
            "a second writer must not join an exclusive writer"
        );
        drop(writer);

        // Fully released — a fresh map of either kind succeeds again.
        assert!(tensor.map_read().is_ok());
    }

    /// `unmap()` is public on the trait and also runs from `Drop`. With
    /// refcounted readers a double release would unmap a SIBLING reader's
    /// mapping, so each map must release at most once.
    #[test]
    fn explicit_unmap_then_drop_releases_only_once() {
        let ops = MockPboOps::new(8);
        let dyn_ops: Arc<dyn PboOps> = ops.clone();
        let tensor = PboTensor::<u8>::from_pbo(11, 8, &[8], None, dyn_ops).unwrap();

        let keeper = tensor.map_read().expect("reader that must survive");
        let mut early = tensor.map_read().expect("reader unmapped by hand");
        early.unmap();
        drop(early); // Drop must NOT release a second time.

        assert_eq!(
            ops.unmap_count(),
            0,
            "one reader still holds the mapping — it must not be unmapped"
        );
        assert!(tensor.is_mapped(), "the surviving reader still holds it");
        // The survivor's pointer is still valid and readable.
        assert_eq!(keeper.as_slice().len(), 8);

        drop(keeper);
        assert_eq!(ops.unmap_count(), 1);
        assert!(!tensor.is_mapped());
    }

    #[test]
    fn test_pbo_tensor_reshape() {
        let ops = MockPboOps::new(24);
        let mut tensor = PboTensor::<u8>::from_pbo(3, 24, &[2, 3, 4], None, ops).unwrap();
        tensor
            .reshape(&[4, 6])
            .expect("compatible reshape should succeed");
        assert_eq!(tensor.shape(), &[4, 6]);
        let result = tensor.reshape(&[100]);
        assert!(result.is_err(), "incompatible reshape should fail");
    }

    #[test]
    fn test_pbo_tensor_set_logical_shape_capacity_based() {
        // A 24-byte PBO can be reconfigured to any shape that fits (unlike the
        // strict `reshape`), so an oversized reusable pool can be
        // `configure_image`d to a smaller image (the native-chroma decode pool).
        let ops = MockPboOps::new(24);
        let mut tensor = PboTensor::<u8>::from_pbo(7, 24, &[24], None, ops).unwrap();
        // Smaller-than-capacity logical shape is accepted (reshape would reject).
        tensor
            .set_logical_shape(&[4, 5])
            .expect("shape within capacity should succeed");
        assert_eq!(tensor.shape(), &[4, 5]);
        // Exactly-capacity is fine.
        tensor.set_logical_shape(&[24]).unwrap();
        // Over-capacity is rejected.
        assert!(
            tensor.set_logical_shape(&[5, 5]).is_err(),
            "shape exceeding PBO capacity must be rejected"
        );
    }

    #[test]
    fn test_pbo_tensor_buffer_identity() {
        let ops1 = MockPboOps::new(8);
        let ops2 = MockPboOps::new(8);
        let t1 = PboTensor::<u8>::from_pbo(1, 8, &[8], None, ops1).unwrap();
        let t2 = PboTensor::<u8>::from_pbo(2, 8, &[8], None, ops2).unwrap();
        assert_ne!(t1.buffer_identity().id(), t2.buffer_identity().id());
    }

    #[test]
    fn test_pbo_tensor_new_returns_error() {
        let result = PboTensor::<u8>::new(&[8], None);
        assert!(result.is_err(), "PboTensor::new() should fail");
    }

    #[cfg(unix)]
    #[test]
    fn test_pbo_tensor_fd_ops_return_error() {
        let ops = MockPboOps::new(8);
        let tensor = PboTensor::<u8>::from_pbo(1, 8, &[8], None, ops).unwrap();
        assert!(tensor.clone_fd().is_err());
    }

    #[test]
    fn test_pbo_tensor_from_pbo_size_mismatch() {
        let ops = MockPboOps::new(24);
        let result = PboTensor::<u8>::from_pbo(1, 24, &[2, 3, 5], None, ops);
        assert!(result.is_err(), "mismatched size/shape should fail");
    }

    #[test]
    fn test_pbo_tensor_from_pbo_zero_size() {
        let ops = MockPboOps::new(0);
        let result = PboTensor::<u8>::from_pbo(1, 0, &[0], None, ops);
        assert!(result.is_err(), "zero size should fail");
    }

    #[test]
    fn test_pbo_via_tensor_enum() {
        let ops = MockPboOps::new(12);
        let pbo = PboTensor::<u8>::from_pbo(10, 12, &[3, 4], Some("enum_test"), ops).unwrap();
        let tensor = crate::Tensor::wrap(crate::TensorStorage::Pbo(pbo));
        assert_eq!(tensor.memory(), TensorMemory::Pbo);
        assert_eq!(tensor.name(), "enum_test");
        assert_eq!(tensor.shape(), &[3, 4]);
        let mut map = tensor.map().expect("map via enum");
        map.as_mut_slice().fill(42);
        assert!(map.as_slice().iter().all(|&b| b == 42));
    }
}
