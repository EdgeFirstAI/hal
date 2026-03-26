// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!
EdgeFirst HAL - Tensor Module

The `edgefirst_tensor` crate provides a unified interface for managing multi-dimensional arrays (tensors)
with support for different memory types, including Direct Memory Access (DMA), POSIX Shared Memory (Shm),
and system memory. The crate defines traits and structures for creating, reshaping, and mapping tensors into memory.

## Examples
```rust
use edgefirst_tensor::{Error, Tensor, TensorMemory, TensorTrait};
# fn main() -> Result<(), Error> {
let tensor = Tensor::<f32>::new(&[2, 3, 4], Some(TensorMemory::Mem), Some("test_tensor"))?;
assert_eq!(tensor.memory(), TensorMemory::Mem);
assert_eq!(tensor.name(), "test_tensor");
#    Ok(())
# }
```

## Overview
The main structures and traits provided by the `edgefirst_tensor` crate are `TensorTrait` and `TensorMapTrait`,
which define the behavior of Tensors and their memory mappings, respectively.
The `Tensor<T>` struct wraps a backend-specific storage with optional image format metadata (`PixelFormat`),
while the `TensorMap` enum provides access to the underlying data. The `TensorDyn` type-erased enum
wraps `Tensor<T>` for runtime element-type dispatch.
 */
#[cfg(target_os = "linux")]
mod dma;
#[cfg(target_os = "linux")]
mod dmabuf;
mod error;
mod format;
mod mem;
mod pbo;
#[cfg(unix)]
mod shm;
mod tensor_dyn;

#[cfg(target_os = "linux")]
pub use crate::dma::{DmaMap, DmaTensor};
pub use crate::mem::{MemMap, MemTensor};
pub use crate::pbo::{PboMap, PboMapping, PboOps, PboTensor};
#[cfg(unix)]
pub use crate::shm::{ShmMap, ShmTensor};
pub use error::{Error, Result};
pub use format::{PixelFormat, PixelLayout};
use num_traits::Num;
use serde::{Deserialize, Serialize};
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
};
pub use tensor_dyn::TensorDyn;

/// Per-plane DMA-BUF descriptor for external buffer import.
///
/// Owns a duplicated file descriptor plus optional stride and offset metadata.
/// The fd is duplicated eagerly in [`new()`](Self::new) so that a bad fd is
/// caught immediately. `import_image` consumes the descriptor and takes
/// ownership of the duped fd — no further cleanup is needed by the caller.
///
/// # Examples
///
/// ```rust,no_run
/// use edgefirst_tensor::PlaneDescriptor;
/// use std::os::fd::BorrowedFd;
///
/// // SAFETY: fd 42 is hypothetical; real code must pass a valid fd.
/// let pd = unsafe { PlaneDescriptor::new(BorrowedFd::borrow_raw(42)) }
///     .unwrap()
///     .with_stride(2048)
///     .with_offset(0);
/// ```
#[cfg(unix)]
pub struct PlaneDescriptor {
    fd: OwnedFd,
    stride: Option<usize>,
    offset: Option<usize>,
}

#[cfg(unix)]
impl PlaneDescriptor {
    /// Create a new plane descriptor by duplicating the given file descriptor.
    ///
    /// The fd is duped immediately — a bad fd fails here rather than inside
    /// `import_image`. The caller retains ownership of the original fd.
    ///
    /// # Errors
    ///
    /// Returns an error if the `dup()` syscall fails (e.g. invalid fd or
    /// fd limit reached).
    pub fn new(fd: std::os::fd::BorrowedFd<'_>) -> Result<Self> {
        let owned = fd.try_clone_to_owned()?;
        Ok(Self {
            fd: owned,
            stride: None,
            offset: None,
        })
    }

    /// Set the row stride in bytes (consuming builder).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Set the plane offset in bytes (consuming builder).
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Consume the descriptor and return the owned file descriptor.
    pub fn into_fd(self) -> OwnedFd {
        self.fd
    }

    /// Row stride in bytes, if set.
    pub fn stride(&self) -> Option<usize> {
        self.stride
    }

    /// Plane offset in bytes, if set.
    pub fn offset(&self) -> Option<usize> {
        self.offset
    }
}

/// Element type discriminant for runtime type identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Short type name (e.g., "u8", "f32", "f16").
    pub const fn name(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Monotonic counter for buffer identity IDs.
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identity for a tensor's underlying buffer.
///
/// Created fresh on every buffer allocation or import. The `id` is a monotonic
/// u64 used as a cache key. The `guard` is an `Arc<()>` whose weak references
/// allow downstream caches to detect when the buffer has been dropped.
#[derive(Debug, Clone)]
pub struct BufferIdentity {
    id: u64,
    guard: Arc<()>,
}

impl BufferIdentity {
    /// Create a new unique buffer identity.
    pub fn new() -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            guard: Arc::new(()),
        }
    }

    /// Unique identifier for this buffer. Changes when the buffer changes.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns a weak reference to the buffer guard. Goes dead when the
    /// owning Tensor is dropped (and no clones remain).
    pub fn weak(&self) -> Weak<()> {
        Arc::downgrade(&self.guard)
    }
}

impl Default for BufferIdentity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "linux")]
use nix::sys::stat::{major, minor};

pub trait TensorTrait<T>: Send + Sync
where
    T: Num + Clone + fmt::Debug,
{
    /// Create a new tensor with the given shape and optional name. If no name
    /// is given, a random name will be generated.
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    #[cfg(unix)]
    /// Create a new tensor using the given file descriptor, shape, and optional
    /// name. If no name is given, a random name will be generated.
    ///
    /// On Linux: Inspects the fd to determine DMA vs SHM based on device major/minor.
    /// On other Unix (macOS): Always creates SHM tensor.
    fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    #[cfg(unix)]
    /// Clone the file descriptor associated with this tensor.
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd>;

    /// Get the memory type of this tensor.
    fn memory(&self) -> TensorMemory;

    /// Get the name of this tensor.
    fn name(&self) -> String;

    /// Get the number of elements in this tensor.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size in bytes of this tensor.
    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Get the shape of this tensor.
    fn shape(&self) -> &[usize];

    /// Reshape this tensor to the given shape. The total number of elements
    /// must remain the same.
    fn reshape(&mut self, shape: &[usize]) -> Result<()>;

    /// Map the tensor into memory and return a TensorMap for accessing the
    /// data.
    fn map(&self) -> Result<TensorMap<T>>;

    /// Get the buffer identity for cache keying and liveness tracking.
    fn buffer_identity(&self) -> &BufferIdentity;
}

pub trait TensorMapTrait<T>
where
    T: Num + Clone + fmt::Debug,
{
    /// Get the shape of this tensor map.
    fn shape(&self) -> &[usize];

    /// Unmap the tensor from memory.
    fn unmap(&mut self);

    /// Get the number of elements in this tensor map.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size in bytes of this tensor map.
    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Get a slice to the data in this tensor map.
    fn as_slice(&self) -> &[T];

    /// Get a mutable slice to the data in this tensor map.
    fn as_mut_slice(&mut self) -> &mut [T];

    #[cfg(feature = "ndarray")]
    /// Get an ndarray ArrayView of the tensor data.
    fn view(&'_ self) -> Result<ndarray::ArrayView<'_, T, ndarray::Dim<ndarray::IxDynImpl>>> {
        Ok(ndarray::ArrayView::from_shape(
            self.shape(),
            self.as_slice(),
        )?)
    }

    #[cfg(feature = "ndarray")]
    /// Get an ndarray ArrayViewMut of the tensor data.
    fn view_mut(
        &'_ mut self,
    ) -> Result<ndarray::ArrayViewMut<'_, T, ndarray::Dim<ndarray::IxDynImpl>>> {
        let shape = self.shape().to_vec();
        Ok(ndarray::ArrayViewMut::from_shape(
            shape,
            self.as_mut_slice(),
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMemory {
    #[cfg(target_os = "linux")]
    /// Direct Memory Access (DMA) allocation. Incurs additional
    /// overhead for memory reading/writing with the CPU.  Allows for
    /// hardware acceleration when supported.
    Dma,
    #[cfg(unix)]
    /// POSIX Shared Memory allocation. Suitable for inter-process
    /// communication, but not suitable for hardware acceleration.
    Shm,

    /// Regular system memory allocation
    Mem,

    /// OpenGL Pixel Buffer Object memory. Created by ImageProcessor
    /// when DMA-buf is unavailable but OpenGL is present.
    Pbo,
}

impl From<TensorMemory> for String {
    fn from(memory: TensorMemory) -> Self {
        match memory {
            #[cfg(target_os = "linux")]
            TensorMemory::Dma => "dma".to_owned(),
            #[cfg(unix)]
            TensorMemory::Shm => "shm".to_owned(),
            TensorMemory::Mem => "mem".to_owned(),
            TensorMemory::Pbo => "pbo".to_owned(),
        }
    }
}

impl TryFrom<&str> for TensorMemory {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            #[cfg(target_os = "linux")]
            "dma" => Ok(TensorMemory::Dma),
            #[cfg(unix)]
            "shm" => Ok(TensorMemory::Shm),
            "mem" => Ok(TensorMemory::Mem),
            "pbo" => Ok(TensorMemory::Pbo),
            _ => Err(Error::InvalidMemoryType(s.to_owned())),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Variants are constructed by downstream crates via pub(crate) helpers
pub(crate) enum TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    #[cfg(target_os = "linux")]
    Dma(DmaTensor<T>),
    #[cfg(unix)]
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
    Pbo(PboTensor<T>),
}

impl<T> TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Create a new tensor storage with the given shape, memory type, and
    /// optional name. If no name is given, a random name will be generated.
    /// If no memory type is given, the best available memory type will be
    /// chosen based on the platform and environment variables.
    fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Dma) => {
                DmaTensor::<T>::new(shape, name).map(TensorStorage::Dma)
            }
            #[cfg(unix)]
            Some(TensorMemory::Shm) => {
                ShmTensor::<T>::new(shape, name).map(TensorStorage::Shm)
            }
            Some(TensorMemory::Mem) => {
                MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
            }
            Some(TensorMemory::Pbo) => Err(crate::error::Error::NotImplemented(
                "PboTensor cannot be created via Tensor::new() — use ImageProcessor::create_image()".to_owned(),
            )),
            None => {
                if std::env::var("EDGEFIRST_TENSOR_FORCE_MEM")
                    .is_ok_and(|x| x != "0" && x.to_lowercase() != "false")
                {
                    MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                } else {
                    #[cfg(target_os = "linux")]
                    {
                        // Linux: Try DMA -> SHM -> Mem
                        match DmaTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Dma(tensor)),
                            Err(_) => {
                                match ShmTensor::<T>::new(shape, name)
                                    .map(TensorStorage::Shm)
                                {
                                    Ok(tensor) => Ok(tensor),
                                    Err(_) => MemTensor::<T>::new(shape, name)
                                        .map(TensorStorage::Mem),
                                }
                            }
                        }
                    }
                    #[cfg(all(unix, not(target_os = "linux")))]
                    {
                        // macOS/BSD: Try SHM -> Mem (no DMA)
                        match ShmTensor::<T>::new(shape, name) {
                            Ok(tensor) => Ok(TensorStorage::Shm(tensor)),
                            Err(_) => {
                                MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                            }
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        // Windows/other: Mem only
                        MemTensor::<T>::new(shape, name).map(TensorStorage::Mem)
                    }
                }
            }
        }
    }

    /// Create a new tensor storage using the given file descriptor, shape,
    /// and optional name.
    #[cfg(unix)]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use nix::sys::stat::fstat;

            let stat = fstat(&fd)?;
            let major = major(stat.st_dev);
            let minor = minor(stat.st_dev);

            log::debug!("Creating tensor from fd: major={major}, minor={minor}");

            if major != 0 {
                // Dma and Shm tensors are expected to have major number 0
                return Err(Error::UnknownDeviceType(major, minor));
            }

            match minor {
                9 | 10 => {
                    // minor number 9 & 10 indicates DMA memory
                    DmaTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Dma)
                }
                _ => {
                    // other minor numbers are assumed to be shared memory
                    ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm)
                }
            }
        }
        #[cfg(all(unix, not(target_os = "linux")))]
        {
            // On macOS/BSD, always use SHM (no DMA support)
            ShmTensor::<T>::from_fd(fd, shape, name).map(TensorStorage::Shm)
        }
    }
}

impl<T> TensorTrait<T> for TensorStorage<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, None, name)
    }

    #[cfg(unix)]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::from_fd(fd, shape, name)
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<OwnedFd> {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.clone_fd(),
            TensorStorage::Shm(t) => t.clone_fd(),
            TensorStorage::Mem(t) => t.clone_fd(),
            TensorStorage::Pbo(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(_) => TensorMemory::Dma,
            #[cfg(unix)]
            TensorStorage::Shm(_) => TensorMemory::Shm,
            TensorStorage::Mem(_) => TensorMemory::Mem,
            TensorStorage::Pbo(_) => TensorMemory::Pbo,
        }
    }

    fn name(&self) -> String {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.name(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.name(),
            TensorStorage::Mem(t) => t.name(),
            TensorStorage::Pbo(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.shape(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.shape(),
            TensorStorage::Mem(t) => t.shape(),
            TensorStorage::Pbo(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.reshape(shape),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.reshape(shape),
            TensorStorage::Mem(t) => t.reshape(shape),
            TensorStorage::Pbo(t) => t.reshape(shape),
        }
    }

    fn map(&self) -> Result<TensorMap<T>> {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.map(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.map(),
            TensorStorage::Mem(t) => t.map(),
            TensorStorage::Pbo(t) => t.map(),
        }
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        match self {
            #[cfg(target_os = "linux")]
            TensorStorage::Dma(t) => t.buffer_identity(),
            #[cfg(unix)]
            TensorStorage::Shm(t) => t.buffer_identity(),
            TensorStorage::Mem(t) => t.buffer_identity(),
            TensorStorage::Pbo(t) => t.buffer_identity(),
        }
    }
}

/// Multi-backend tensor with optional image format metadata.
///
/// When `format` is `Some`, this tensor represents an image. Width, height,
/// and channels are derived from `shape` + `format`. When `format` is `None`,
/// this is a raw tensor (identical to the pre-refactoring behavior).
#[derive(Debug)]
pub struct Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub(crate) storage: TensorStorage<T>,
    format: Option<PixelFormat>,
    chroma: Option<Box<Tensor<T>>>,
    /// Row stride in bytes for externally allocated buffers with row padding.
    /// `None` means tightly packed (stride == width * bytes_per_pixel).
    row_stride: Option<usize>,
    /// Byte offset within the DMA-BUF where image data starts.
    /// `None` means offset 0 (data starts at the beginning of the buffer).
    plane_offset: Option<usize>,
}

impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Wrap a TensorStorage in a Tensor with no image metadata.
    pub(crate) fn wrap(storage: TensorStorage<T>) -> Self {
        Self {
            storage,
            format: None,
            chroma: None,
            row_stride: None,
            plane_offset: None,
        }
    }

    /// Create a new tensor with the given shape, memory type, and optional
    /// name. If no name is given, a random name will be generated. If no
    /// memory type is given, the best available memory type will be chosen
    /// based on the platform and environment variables.
    ///
    /// On Linux platforms, the order of preference is: Dma -> Shm -> Mem.
    /// On other Unix platforms (macOS), the order is: Shm -> Mem.
    /// On non-Unix platforms, only Mem is available.
    ///
    /// # Environment Variables
    /// - `EDGEFIRST_TENSOR_FORCE_MEM`: If set to a non-zero and non-false
    ///   value, forces the use of regular system memory allocation
    ///   (`TensorMemory::Mem`) regardless of platform capabilities.
    ///
    /// # Example
    /// ```rust
    /// use edgefirst_tensor::{Error, Tensor, TensorMemory, TensorTrait};
    /// # fn main() -> Result<(), Error> {
    /// let tensor = Tensor::<f32>::new(&[2, 3, 4], Some(TensorMemory::Mem), Some("test_tensor"))?;
    /// assert_eq!(tensor.memory(), TensorMemory::Mem);
    /// assert_eq!(tensor.name(), "test_tensor");
    /// #    Ok(())
    /// # }
    /// ```
    pub fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        TensorStorage::new(shape, memory, name).map(Self::wrap)
    }

    /// Create an image tensor with the given format.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let shape = match format.layout() {
            PixelLayout::Packed => vec![height, width, format.channels()],
            PixelLayout::Planar => vec![format.channels(), height, width],
            PixelLayout::SemiPlanar => {
                // Contiguous semi-planar: luma + interleaved chroma in one allocation.
                // NV12 (4:2:0): H lines luma + H/2 lines chroma = H * 3/2 total
                // NV16 (4:2:2): H lines luma + H lines chroma = H * 2 total
                let total_h = match format {
                    PixelFormat::Nv12 => {
                        if !height.is_multiple_of(2) {
                            return Err(Error::InvalidArgument(format!(
                                "NV12 requires even height, got {height}"
                            )));
                        }
                        height * 3 / 2
                    }
                    PixelFormat::Nv16 => height * 2,
                    _ => {
                        return Err(Error::InvalidArgument(format!(
                            "unknown semi-planar height multiplier for {format:?}"
                        )))
                    }
                };
                vec![total_h, width]
            }
        };
        let mut t = Self::new(&shape, memory, None)?;
        t.format = Some(format);
        Ok(t)
    }

    /// Attach format metadata to an existing tensor.
    ///
    /// # Arguments
    ///
    /// * `format` - The pixel format to attach
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, with the format stored as metadata on the tensor.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidShape` if the tensor shape is incompatible with
    /// the format's layout (packed expects `[H, W, C]`, planar expects
    /// `[C, H, W]`, semi-planar expects `[H*k, W]` with format-specific
    /// height constraints).
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        let shape = self.shape();
        match format.layout() {
            PixelLayout::Packed => {
                if shape.len() != 3 || shape[2] != format.channels() {
                    return Err(Error::InvalidShape(format!(
                        "packed format {format:?} expects [H, W, {}], got {shape:?}",
                        format.channels()
                    )));
                }
            }
            PixelLayout::Planar => {
                if shape.len() != 3 || shape[0] != format.channels() {
                    return Err(Error::InvalidShape(format!(
                        "planar format {format:?} expects [{}, H, W], got {shape:?}",
                        format.channels()
                    )));
                }
            }
            PixelLayout::SemiPlanar => {
                if shape.len() != 2 {
                    return Err(Error::InvalidShape(format!(
                        "semi-planar format {format:?} expects [H*k, W], got {shape:?}"
                    )));
                }
                match format {
                    PixelFormat::Nv12 if !shape[0].is_multiple_of(3) => {
                        return Err(Error::InvalidShape(format!(
                            "NV12 contiguous shape[0] must be divisible by 3, got {}",
                            shape[0]
                        )));
                    }
                    PixelFormat::Nv16 if !shape[0].is_multiple_of(2) => {
                        return Err(Error::InvalidShape(format!(
                            "NV16 contiguous shape[0] must be even, got {}",
                            shape[0]
                        )));
                    }
                    _ => {}
                }
            }
        }
        // Clear stored stride/offset when format changes — they may be invalid
        // for the new format. Caller must re-set after changing format.
        if self.format != Some(format) {
            self.row_stride = None;
            self.plane_offset = None;
        }
        self.format = Some(format);
        Ok(())
    }

    /// Pixel format (None if not an image).
    pub fn format(&self) -> Option<PixelFormat> {
        self.format
    }

    /// Image width (None if not an image).
    pub fn width(&self) -> Option<usize> {
        let fmt = self.format?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => Some(shape[1]),
            PixelLayout::Planar => Some(shape[2]),
            PixelLayout::SemiPlanar => Some(shape[1]),
        }
    }

    /// Image height (None if not an image).
    pub fn height(&self) -> Option<usize> {
        let fmt = self.format?;
        let shape = self.shape();
        match fmt.layout() {
            PixelLayout::Packed => Some(shape[0]),
            PixelLayout::Planar => Some(shape[1]),
            PixelLayout::SemiPlanar => {
                if self.is_multiplane() {
                    Some(shape[0])
                } else {
                    match fmt {
                        PixelFormat::Nv12 => Some(shape[0] * 2 / 3),
                        PixelFormat::Nv16 => Some(shape[0] / 2),
                        _ => None,
                    }
                }
            }
        }
    }

    /// Create from separate Y and UV planes (multiplane NV12/NV16).
    pub fn from_planes(luma: Tensor<T>, chroma: Tensor<T>, format: PixelFormat) -> Result<Self> {
        if format.layout() != PixelLayout::SemiPlanar {
            return Err(Error::InvalidArgument(format!(
                "from_planes requires a semi-planar format, got {format:?}"
            )));
        }
        if chroma.format.is_some() || chroma.chroma.is_some() {
            return Err(Error::InvalidArgument(
                "chroma tensor must be a raw tensor (no format or chroma metadata)".into(),
            ));
        }
        let luma_shape = luma.shape();
        let chroma_shape = chroma.shape();
        if luma_shape.len() != 2 || chroma_shape.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "from_planes expects 2D shapes, got luma={luma_shape:?} chroma={chroma_shape:?}"
            )));
        }
        if luma_shape[1] != chroma_shape[1] {
            return Err(Error::InvalidArgument(format!(
                "luma width {} != chroma width {}",
                luma_shape[1], chroma_shape[1]
            )));
        }
        match format {
            PixelFormat::Nv12 => {
                if luma_shape[0] % 2 != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "NV12 requires even luma height, got {}",
                        luma_shape[0]
                    )));
                }
                if chroma_shape[0] != luma_shape[0] / 2 {
                    return Err(Error::InvalidArgument(format!(
                        "NV12 chroma height {} != luma height / 2 ({})",
                        chroma_shape[0],
                        luma_shape[0] / 2
                    )));
                }
            }
            PixelFormat::Nv16 => {
                if chroma_shape[0] != luma_shape[0] {
                    return Err(Error::InvalidArgument(format!(
                        "NV16 chroma height {} != luma height {}",
                        chroma_shape[0], luma_shape[0]
                    )));
                }
            }
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "from_planes only supports NV12 and NV16, got {format:?}"
                )));
            }
        }

        Ok(Tensor {
            storage: luma.storage,
            format: Some(format),
            chroma: Some(Box::new(chroma)),
            row_stride: luma.row_stride,
            plane_offset: luma.plane_offset,
        })
    }

    /// Whether this tensor uses separate plane allocations.
    pub fn is_multiplane(&self) -> bool {
        self.chroma.is_some()
    }

    /// Access the chroma plane for multiplane semi-planar images.
    pub fn chroma(&self) -> Option<&Tensor<T>> {
        self.chroma.as_deref()
    }

    /// Mutable access to the chroma plane for multiplane semi-planar images.
    pub fn chroma_mut(&mut self) -> Option<&mut Tensor<T>> {
        self.chroma.as_deref_mut()
    }

    /// Row stride in bytes (`None` = tightly packed).
    pub fn row_stride(&self) -> Option<usize> {
        self.row_stride
    }

    /// Effective row stride in bytes: the stored stride if set, otherwise the
    /// minimum stride computed from the format, width, and element size.
    /// Returns `None` only when no format is set and no explicit stride was
    /// stored via [`set_row_stride`](Self::set_row_stride).
    pub fn effective_row_stride(&self) -> Option<usize> {
        if let Some(s) = self.row_stride {
            return Some(s);
        }
        let fmt = self.format?;
        let w = self.width()?;
        let elem = std::mem::size_of::<T>();
        Some(match fmt.layout() {
            PixelLayout::Packed => w * fmt.channels() * elem,
            PixelLayout::Planar | PixelLayout::SemiPlanar => w * elem,
        })
    }

    /// Set the row stride in bytes for externally allocated buffers with
    /// row padding (e.g. V4L2 or GStreamer allocators).
    ///
    /// The stride is propagated to the EGL DMA-BUF import attributes so
    /// the GPU interprets the padded buffer layout correctly. Must be
    /// called after [`set_format`](Self::set_format) and before the tensor
    /// is first passed to [`ImageProcessor::convert`]. The stored stride
    /// is cleared automatically if the pixel format is later changed.
    ///
    /// No stride-vs-buffer-size validation is performed because the
    /// backing allocation size is not reliably known: external DMA-BUFs
    /// may be over-allocated by the allocator, and internal tensors store
    /// a logical (unpadded) shape. An incorrect stride will be caught by
    /// the EGL driver at import time.
    ///
    /// # Arguments
    ///
    /// * `stride` - Row stride in bytes. Must be >= the minimum stride for
    ///   the format (width * channels * sizeof(T) for packed,
    ///   width * sizeof(T) for planar/semi-planar).
    ///
    /// # Errors
    ///
    /// * `InvalidArgument` if no pixel format is set on this tensor
    /// * `InvalidArgument` if `stride` is less than the minimum for the
    ///   format and width
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        let fmt = self.format.ok_or_else(|| {
            Error::InvalidArgument("cannot set row_stride without a pixel format".into())
        })?;
        let w = self.width().ok_or_else(|| {
            Error::InvalidArgument("cannot determine width for row_stride validation".into())
        })?;
        let elem = std::mem::size_of::<T>();
        let min_stride = match fmt.layout() {
            PixelLayout::Packed => w * fmt.channels() * elem,
            PixelLayout::Planar | PixelLayout::SemiPlanar => w * elem,
        };
        if stride < min_stride {
            return Err(Error::InvalidArgument(format!(
                "row_stride {stride} < minimum {min_stride} for {fmt:?} at width {w}"
            )));
        }
        self.row_stride = Some(stride);
        Ok(())
    }

    /// Set the row stride without format validation.
    ///
    /// Use this for raw sub-tensors (e.g. chroma planes) that don't carry
    /// format metadata. The caller is responsible for ensuring the stride
    /// is valid.
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        self.row_stride = Some(stride);
    }

    /// Builder-style variant of [`set_row_stride`](Self::set_row_stride),
    /// consuming and returning `self`.
    ///
    /// # Errors
    ///
    /// Same conditions as [`set_row_stride`](Self::set_row_stride).
    pub fn with_row_stride(mut self, stride: usize) -> Result<Self> {
        self.set_row_stride(stride)?;
        Ok(self)
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    pub fn plane_offset(&self) -> Option<usize> {
        self.plane_offset
    }

    /// Set the byte offset within the DMA-BUF where image data starts.
    ///
    /// Propagated to `EGL_DMA_BUF_PLANE0_OFFSET_EXT` on GPU import.
    /// Unlike [`set_row_stride`](Self::set_row_stride), no format is required
    /// since the offset is format-independent.
    pub fn set_plane_offset(&mut self, offset: usize) {
        self.plane_offset = Some(offset);
        #[cfg(target_os = "linux")]
        if let TensorStorage::Dma(ref mut dma) = self.storage {
            dma.mmap_offset = offset;
        }
    }

    /// Builder-style variant of [`set_plane_offset`](Self::set_plane_offset),
    /// consuming and returning `self`.
    pub fn with_plane_offset(mut self, offset: usize) -> Self {
        self.set_plane_offset(offset);
        self
    }

    /// Downcast to PBO tensor reference (for GL backends).
    pub fn as_pbo(&self) -> Option<&PboTensor<T>> {
        match &self.storage {
            TensorStorage::Pbo(p) => Some(p),
            _ => None,
        }
    }

    /// Downcast to DMA tensor reference (for EGL import, G2D).
    #[cfg(target_os = "linux")]
    pub fn as_dma(&self) -> Option<&DmaTensor<T>> {
        match &self.storage {
            TensorStorage::Dma(d) => Some(d),
            _ => None,
        }
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor.
    ///
    /// # Returns
    ///
    /// A borrowed reference to the DMA-BUF file descriptor, tied to `self`'s
    /// lifetime.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotImplemented` if the tensor is not DMA-backed.
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> Result<std::os::fd::BorrowedFd<'_>> {
        use std::os::fd::AsFd;
        match &self.storage {
            TensorStorage::Dma(dma) => Ok(dma.fd.as_fd()),
            _ => Err(Error::NotImplemented(format!(
                "dmabuf requires DMA-backed tensor, got {:?}",
                self.storage.memory()
            ))),
        }
    }

    /// Construct a Tensor from a PBO tensor (for GL backends that allocate PBOs).
    pub fn from_pbo(pbo: PboTensor<T>) -> Self {
        Self {
            storage: TensorStorage::Pbo(pbo),
            format: None,
            chroma: None,
            row_stride: None,
            plane_offset: None,
        }
    }
}

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new(shape, None, name)
    }

    #[cfg(unix)]
    fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::wrap(TensorStorage::from_fd(fd, shape, name)?))
    }

    #[cfg(unix)]
    fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        self.storage.clone_fd()
    }

    fn memory(&self) -> TensorMemory {
        self.storage.memory()
    }

    fn name(&self) -> String {
        self.storage.name()
    }

    fn shape(&self) -> &[usize] {
        self.storage.shape()
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        if self.chroma.is_some() {
            return Err(Error::InvalidOperation(
                "cannot reshape a multiplane tensor — decompose planes first".into(),
            ));
        }
        self.storage.reshape(shape)?;
        self.format = None;
        self.row_stride = None;
        self.plane_offset = None;
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        if self.row_stride.is_some() {
            return Err(Error::InvalidOperation(
                "CPU mapping of strided tensors is not supported; use GPU path only".into(),
            ));
        }
        // Offset tensors are supported for DMA storage — DmaMap adjusts the
        // mmap range and slice start position.  Non-DMA offset tensors are
        // not meaningful (offset only applies to DMA-BUF sub-regions).
        if self.plane_offset.is_some_and(|o| o > 0) {
            #[cfg(target_os = "linux")]
            if !matches!(self.storage, TensorStorage::Dma(_)) {
                return Err(Error::InvalidOperation(
                    "plane offset only supported for DMA tensors".into(),
                ));
            }
            #[cfg(not(target_os = "linux"))]
            return Err(Error::InvalidOperation(
                "plane offset only supported for DMA tensors".into(),
            ));
        }
        self.storage.map()
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        self.storage.buffer_identity()
    }
}

pub enum TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    #[cfg(target_os = "linux")]
    Dma(DmaMap<T>),
    #[cfg(unix)]
    Shm(ShmMap<T>),
    Mem(MemMap<T>),
    Pbo(PboMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.shape(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.shape(),
            TensorMap::Mem(map) => map.shape(),
            TensorMap::Pbo(map) => map.shape(),
        }
    }

    fn unmap(&mut self) {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.unmap(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.unmap(),
            TensorMap::Mem(map) => map.unmap(),
            TensorMap::Pbo(map) => map.unmap(),
        }
    }

    fn as_slice(&self) -> &[T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_slice(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.as_slice(),
            TensorMap::Mem(map) => map.as_slice(),
            TensorMap::Pbo(map) => map.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_mut_slice(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.as_mut_slice(),
            TensorMap::Mem(map) => map.as_mut_slice(),
            TensorMap::Pbo(map) => map.as_mut_slice(),
        }
    }
}

impl<T> Deref for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.deref(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.deref(),
            TensorMap::Mem(map) => map.deref(),
            TensorMap::Pbo(map) => map.deref(),
        }
    }
}

impl<T> DerefMut for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.deref_mut(),
            #[cfg(unix)]
            TensorMap::Shm(map) => map.deref_mut(),
            TensorMap::Mem(map) => map.deref_mut(),
            TensorMap::Pbo(map) => map.deref_mut(),
        }
    }
}

// ============================================================================
// Platform availability helpers
// ============================================================================

/// Check if DMA memory allocation is available on this system.
///
/// Returns `true` only on Linux systems with DMA-BUF heap access (typically
/// requires running as root or membership in a video/render group).
/// Always returns `false` on non-Linux platforms (macOS, Windows, etc.).
///
/// This function caches its result after the first call for efficiency.
#[cfg(target_os = "linux")]
static DMA_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if DMA memory allocation is available on this system.
#[cfg(target_os = "linux")]
pub fn is_dma_available() -> bool {
    *DMA_AVAILABLE.get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::Dma), None).is_ok())
}

/// Check if DMA memory allocation is available on this system.
///
/// Always returns `false` on non-Linux platforms since DMA-BUF is Linux-specific.
#[cfg(not(target_os = "linux"))]
pub fn is_dma_available() -> bool {
    false
}

/// Check if POSIX shared memory allocation is available on this system.
///
/// Returns `true` on Unix systems (Linux, macOS, BSD) where POSIX shared memory
/// is supported. Always returns `false` on non-Unix platforms (Windows).
///
/// This function caches its result after the first call for efficiency.
#[cfg(unix)]
static SHM_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Check if POSIX shared memory allocation is available on this system.
#[cfg(unix)]
pub fn is_shm_available() -> bool {
    *SHM_AVAILABLE.get_or_init(|| Tensor::<u8>::new(&[64], Some(TensorMemory::Shm), None).is_ok())
}

/// Check if POSIX shared memory allocation is available on this system.
///
/// Always returns `false` on non-Unix platforms since POSIX SHM is Unix-specific.
#[cfg(not(unix))]
pub fn is_shm_available() -> bool {
    false
}

#[cfg(test)]
mod dtype_tests {
    use super::*;

    #[test]
    fn dtype_size() {
        assert_eq!(DType::U8.size(), 1);
        assert_eq!(DType::I8.size(), 1);
        assert_eq!(DType::U16.size(), 2);
        assert_eq!(DType::I16.size(), 2);
        assert_eq!(DType::U32.size(), 4);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::U64.size(), 8);
        assert_eq!(DType::I64.size(), 8);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F64.size(), 8);
    }

    #[test]
    fn dtype_name() {
        assert_eq!(DType::U8.name(), "u8");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::F32.name(), "f32");
    }

    #[test]
    fn dtype_serde_roundtrip() {
        use serde_json;
        let dt = DType::F16;
        let json = serde_json::to_string(&dt).unwrap();
        let back: DType = serde_json::from_str(&json).unwrap();
        assert_eq!(dt, back);
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn raw_tensor_has_no_format() {
        let t = Tensor::<u8>::new(&[480, 640, 3], None, None).unwrap();
        assert!(t.format().is_none());
        assert!(t.width().is_none());
        assert!(t.height().is_none());
        assert!(!t.is_multiplane());
        assert!(t.chroma().is_none());
    }

    #[test]
    fn image_tensor_packed() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::Rgba, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[480, 640, 4]);
        assert!(!t.is_multiplane());
    }

    #[test]
    fn image_tensor_planar() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::PlanarRgb, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        assert_eq!(t.shape(), &[3, 480, 640]);
    }

    #[test]
    fn image_tensor_semi_planar_contiguous() {
        let t = Tensor::<u8>::image(640, 480, PixelFormat::Nv12, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
        // NV12: H*3/2 = 720
        assert_eq!(t.shape(), &[720, 640]);
        assert!(!t.is_multiplane());
    }

    #[test]
    fn set_format_valid() {
        let mut t = Tensor::<u8>::new(&[480, 640, 3], None, None).unwrap();
        assert!(t.format().is_none());
        t.set_format(PixelFormat::Rgb).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgb));
        assert_eq!(t.width(), Some(640));
        assert_eq!(t.height(), Some(480));
    }

    #[test]
    fn set_format_invalid_shape() {
        let mut t = Tensor::<u8>::new(&[480, 640, 4], None, None).unwrap();
        // RGB expects 3 channels, not 4
        let err = t.set_format(PixelFormat::Rgb);
        assert!(err.is_err());
        // Original tensor is unmodified
        assert!(t.format().is_none());
    }

    #[test]
    fn reshape_clears_format() {
        let mut t = Tensor::<u8>::image(640, 480, PixelFormat::Rgba, None).unwrap();
        assert_eq!(t.format(), Some(PixelFormat::Rgba));
        // Reshape to flat — format cleared
        t.reshape(&[480 * 640 * 4]).unwrap();
        assert!(t.format().is_none());
    }

    #[test]
    fn from_planes_nv12() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let img = Tensor::from_planes(y, uv, PixelFormat::Nv12).unwrap();
        assert_eq!(img.format(), Some(PixelFormat::Nv12));
        assert!(img.is_multiplane());
        assert!(img.chroma().is_some());
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
    }

    #[test]
    fn from_planes_rejects_non_semiplanar() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let err = Tensor::from_planes(y, uv, PixelFormat::Rgb);
        assert!(err.is_err());
    }

    #[test]
    fn reshape_multiplane_errors() {
        let y = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let uv = Tensor::<u8>::new(&[240, 640], None, None).unwrap();
        let mut img = Tensor::from_planes(y, uv, PixelFormat::Nv12).unwrap();
        let err = img.reshape(&[480 * 640 + 240 * 640]);
        assert!(err.is_err());
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use nix::unistd::{access, AccessFlags};
    #[cfg(target_os = "linux")]
    use std::io::Write as _;
    use std::sync::RwLock;

    use super::*;

    #[ctor::ctor]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    /// Macro to get the current function name for logging in tests.
    #[cfg(target_os = "linux")]
    macro_rules! function {
        () => {{
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);

            // Find and cut the rest of the path
            match &name[..name.len() - 3].rfind(':') {
                Some(pos) => &name[pos + 1..name.len() - 3],
                None => &name[..name.len() - 3],
            }
        }};
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_tensor() {
        let _lock = FD_LOCK.read().unwrap();
        let shape = vec![1];
        let tensor = DmaTensor::<f32>::new(&shape, Some("dma_tensor"));
        let dma_enabled = tensor.is_ok();

        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        match dma_enabled {
            true => assert_eq!(tensor.memory(), TensorMemory::Dma),
            false => assert_eq!(tensor.memory(), TensorMemory::Shm),
        }
    }

    #[test]
    #[cfg(all(unix, not(target_os = "linux")))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        // On macOS/BSD, auto-detection tries SHM first, falls back to Mem
        assert!(
            tensor.memory() == TensorMemory::Shm || tensor.memory() == TensorMemory::Mem,
            "Expected SHM or Mem on macOS, got {:?}",
            tensor.memory()
        );
    }

    #[test]
    #[cfg(not(unix))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        assert_eq!(tensor.memory(), TensorMemory::Mem);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_tensor() {
        let _lock = FD_LOCK.read().unwrap();
        match access(
            "/dev/dma_heap/linux,cma",
            AccessFlags::R_OK | AccessFlags::W_OK,
        ) {
            Ok(_) => println!("/dev/dma_heap/linux,cma is available"),
            Err(_) => match access(
                "/dev/dma_heap/system",
                AccessFlags::R_OK | AccessFlags::W_OK,
            ) {
                Ok(_) => println!("/dev/dma_heap/system is available"),
                Err(e) => {
                    writeln!(
                        &mut std::io::stdout(),
                        "[WARNING] DMA Heap is unavailable: {e}"
                    )
                    .unwrap();
                    return;
                }
            },
        }

        let shape = vec![2, 3, 4];
        let tensor =
            DmaTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");

        const DUMMY_VALUE: f32 = 12.34;

        assert_eq!(tensor.memory(), TensorMemory::Dma);
        assert_eq!(tensor.name(), "test_tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.len(), 2 * 3 * 4);

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        {
            let shared = Tensor::<f32>::from_fd(
                tensor
                    .clone_fd()
                    .expect("Failed to duplicate tensor file descriptor"),
                &shape,
                Some("test_tensor_shared"),
            )
            .expect("Failed to create tensor from fd");

            assert_eq!(shared.memory(), TensorMemory::Dma);
            assert_eq!(shared.name(), "test_tensor_shared");
            assert_eq!(shared.shape(), &shape);

            let mut tensor_map = shared.map().expect("Failed to map DMA memory from fd");
            tensor_map.fill(DUMMY_VALUE);
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map DMA memory");
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        let mut tensor = DmaTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map DMA memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_shm_tensor() {
        let _lock = FD_LOCK.read().unwrap();
        let shape = vec![2, 3, 4];
        let tensor =
            ShmTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.name(), "test_tensor");

        const DUMMY_VALUE: f32 = 12.34;
        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        {
            let shared = Tensor::<f32>::from_fd(
                tensor
                    .clone_fd()
                    .expect("Failed to duplicate tensor file descriptor"),
                &shape,
                Some("test_tensor_shared"),
            )
            .expect("Failed to create tensor from fd");

            assert_eq!(shared.memory(), TensorMemory::Shm);
            assert_eq!(shared.name(), "test_tensor_shared");
            assert_eq!(shared.shape(), &shape);

            let mut tensor_map = shared.map().expect("Failed to map shared memory from fd");
            tensor_map.fill(DUMMY_VALUE);
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map shared memory");
            assert!(tensor_map.iter().all(|&x| x == DUMMY_VALUE));
        }

        let mut tensor = ShmTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map shared memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    fn test_mem_tensor() {
        let shape = vec![2, 3, 4];
        let tensor =
            MemTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.name(), "test_tensor");

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map.fill(42.0);
            assert!(tensor_map.iter().all(|&x| x == 42.0));
        }

        let mut tensor = MemTensor::<u8>::new(&shape, None).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        let new_shape = vec![3, 4, 4];
        assert!(
            tensor.reshape(&new_shape).is_err(),
            "Reshape should fail due to size mismatch"
        );
        assert_eq!(tensor.shape(), &shape, "Shape should remain unchanged");

        let new_shape = vec![2, 3, 4];
        tensor.reshape(&new_shape).expect("Reshape should succeed");
        assert_eq!(
            tensor.shape(),
            &new_shape,
            "Shape should be updated after successful reshape"
        );

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map.fill(1);
            assert!(tensor_map.iter().all(|&x| x == 1));
        }

        {
            let mut tensor_map = tensor.map().expect("Failed to map memory");
            tensor_map[2] = 42;
            assert_eq!(tensor_map[1], 1, "Value at index 1 should be 1");
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_dma_available() {
            log::warn!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Dma), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let end_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_from_fd_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_dma_available() {
            log::warn!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Dma), None).unwrap();

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }
        drop(orig);

        let end_open_fds = proc.fd_count().unwrap();

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_shm_available() {
            log::warn!(
                "SKIPPED: {} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        for _ in 0..100 {
            let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
                .expect("Failed to create tensor");
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }

        let end_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_shm_from_fd_no_fd_leaks() {
        let _lock = FD_LOCK.write().unwrap();
        if !is_shm_available() {
            log::warn!(
                "SKIPPED: {} - SHM memory allocation not available (permission denied or no SHM support)",
                function!()
            );
            return;
        }

        let proc = procfs::process::Process::myself()
            .expect("Failed to get current process using /proc/self");

        let start_open_fds = proc
            .fd_count()
            .expect("Failed to get open file descriptor count");

        let orig = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None).unwrap();

        for _ in 0..100 {
            let tensor =
                Tensor::<u8>::from_fd(orig.clone_fd().unwrap(), orig.shape(), None).unwrap();
            let mut map = tensor.map().unwrap();
            map.as_mut_slice().fill(233);
        }
        drop(orig);

        let end_open_fds = proc.fd_count().unwrap();

        assert_eq!(
            start_open_fds, end_open_fds,
            "File descriptor leak detected: {} -> {}",
            start_open_fds, end_open_fds
        );
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray() {
        let _lock = FD_LOCK.read().unwrap();
        let shape = vec![2, 3, 4];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");

        let mut tensor_map = tensor.map().expect("Failed to map tensor memory");
        tensor_map.fill(1.0);

        let view = tensor_map.view().expect("Failed to get ndarray view");
        assert_eq!(view.shape(), &[2, 3, 4]);
        assert!(view.iter().all(|&x| x == 1.0));

        let mut view_mut = tensor_map
            .view_mut()
            .expect("Failed to get mutable ndarray view");
        view_mut[[0, 0, 0]] = 42.0;
        assert_eq!(view_mut[[0, 0, 0]], 42.0);
        assert_eq!(tensor_map[0], 42.0, "Value at index 0 should be 42");
    }

    #[test]
    fn test_buffer_identity_unique() {
        let id1 = BufferIdentity::new();
        let id2 = BufferIdentity::new();
        assert_ne!(
            id1.id(),
            id2.id(),
            "Two identities should have different ids"
        );
    }

    #[test]
    fn test_buffer_identity_clone_shares_guard() {
        let id1 = BufferIdentity::new();
        let weak = id1.weak();
        assert!(
            weak.upgrade().is_some(),
            "Weak should be alive while original exists"
        );

        let id2 = id1.clone();
        assert_eq!(id1.id(), id2.id(), "Cloned identity should have same id");

        drop(id1);
        assert!(
            weak.upgrade().is_some(),
            "Weak should still be alive (clone holds Arc)"
        );

        drop(id2);
        assert!(
            weak.upgrade().is_none(),
            "Weak should be dead after all clones dropped"
        );
    }

    #[test]
    fn test_tensor_buffer_identity() {
        let t1 = Tensor::<u8>::new(&[100], Some(TensorMemory::Mem), Some("t1")).unwrap();
        let t2 = Tensor::<u8>::new(&[100], Some(TensorMemory::Mem), Some("t2")).unwrap();
        assert_ne!(
            t1.buffer_identity().id(),
            t2.buffer_identity().id(),
            "Different tensors should have different buffer ids"
        );
    }

    // Any test that cares about the fd count must grab it exclusively.
    // Any tests which modifies the fd count by opening or closing fds must grab it
    // shared.
    pub static FD_LOCK: RwLock<()> = RwLock::new(());

    /// Test that DMA is NOT available on non-Linux platforms.
    /// This verifies the cross-platform behavior of is_dma_available().
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn test_dma_not_available_on_non_linux() {
        assert!(
            !is_dma_available(),
            "DMA memory allocation should NOT be available on non-Linux platforms"
        );
    }

    /// Test that SHM memory allocation is available and usable on Unix systems.
    /// This is a basic functional test; Linux has additional FD leak tests using procfs.
    #[test]
    #[cfg(unix)]
    fn test_shm_available_and_usable() {
        assert!(
            is_shm_available(),
            "SHM memory allocation should be available on Unix systems"
        );

        // Create a tensor with SHM backing
        let tensor = Tensor::<u8>::new(&[100, 100], Some(TensorMemory::Shm), None)
            .expect("Failed to create SHM tensor");

        // Verify we can map and write to it
        let mut map = tensor.map().expect("Failed to map SHM tensor");
        map.as_mut_slice().fill(0xAB);

        // Verify the data was written correctly
        assert!(
            map.as_slice().iter().all(|&b| b == 0xAB),
            "SHM tensor data should be writable and readable"
        );
    }
}
