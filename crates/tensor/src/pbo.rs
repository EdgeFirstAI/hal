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
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
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

/// Opaque handle to a PBO's GL resources.
struct PboHandle {
    ops: Arc<dyn PboOps>,
    buffer_id: u32,
    size: usize,
    mapped: AtomicBool,
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
    pub fn from_pbo(
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        name: Option<&str>,
        ops: Arc<dyn PboOps>,
    ) -> Self {
        let name = name.unwrap_or("pbo_tensor").to_owned();
        Self {
            name,
            shape: shape.to_vec(),
            handle: Arc::new(PboHandle {
                ops,
                buffer_id,
                size,
                mapped: AtomicBool::new(false),
            }),
            identity: BufferIdentity::new(),
            _marker: PhantomData,
        }
    }

    /// Returns the GL buffer ID for this PBO.
    pub fn buffer_id(&self) -> u32 {
        self.handle.buffer_id
    }

    /// Returns true if the PBO is currently mapped for CPU access.
    pub fn is_mapped(&self) -> bool {
        self.handle.mapped.load(Ordering::Acquire)
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
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        if self.handle.mapped.swap(true, Ordering::AcqRel) {
            return Err(Error::PboMapped);
        }
        match self
            .handle
            .ops
            .map_buffer(self.handle.buffer_id, self.handle.size)
        {
            Ok(mapping) => {
                let pbo_ptr = PboPtr(
                    NonNull::new(mapping.ptr as *mut c_void)
                        .ok_or(Error::InvalidSize(self.handle.size))?,
                );
                Ok(TensorMap::Pbo(PboMap {
                    ptr: Arc::new(Mutex::new(pbo_ptr)),
                    shape: self.shape.clone(),
                    handle: Arc::clone(&self.handle),
                    _marker: PhantomData,
                }))
            }
            Err(e) => {
                self.handle.mapped.store(false, Ordering::Release);
                Err(e)
            }
        }
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
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
        trace!("Unmapping PboMap buffer_id={}", self.handle.buffer_id);
        if let Err(e) = self.handle.ops.unmap_buffer(self.handle.buffer_id) {
            log::warn!("Failed to unmap PBO buffer {}: {e}", self.handle.buffer_id);
        }
        self.handle.mapped.store(false, Ordering::Release);
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock PboMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock PboMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
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
