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
The main structures and traits provided by the `edgefirst_tensor` crate is the `TensorTrait` and `TensorMapTrait` traits,
which define the behavior of Tensors and their memory mappings, respectively.
The `Tensor` enum encapsulates different tensor implementations based on the memory type, while the `TensorMap` enum
provides access to the underlying data.
```
 */
#[cfg(target_os = "linux")]
mod dma;
#[cfg(target_os = "linux")]
mod dmabuf;
mod error;
mod mem;
#[cfg(target_os = "linux")]
mod shm;

pub use crate::mem::{MemMap, MemTensor};
#[cfg(target_os = "linux")]
pub use crate::{
    dma::{DmaMap, DmaTensor},
    shm::{ShmMap, ShmTensor},
};
pub use error::{Error, Result};
use num_traits::Num;
#[cfg(target_os = "linux")]
use std::os::fd::OwnedFd;
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

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

    #[cfg(target_os = "linux")]
    /// Create a new tensor using the given file descriptor, shape, and optional
    /// name. If no name is given, a random name will be generated.
    fn from_fd(fd: std::os::fd::OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    #[cfg(target_os = "linux")]
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
    /// Direct Memory Access (DMA) allocation allocation. Incurs additional
    /// overhead for memory reading/writing with the CPU.  Allows for
    /// hardware acceleration when suppported.
    Dma,
    #[cfg(target_os = "linux")]
    /// POSIX Shared Memory allocation. Suitable for inter-process
    /// communication, but not suitable for hardware acceleration.
    Shm,

    /// Regular system memory allocation
    Mem,
}

impl From<TensorMemory> for String {
    fn from(memory: TensorMemory) -> Self {
        match memory {
            #[cfg(target_os = "linux")]
            TensorMemory::Dma => "dma".to_owned(),
            #[cfg(target_os = "linux")]
            TensorMemory::Shm => "shm".to_owned(),
            TensorMemory::Mem => "mem".to_owned(),
        }
    }
}

impl TryFrom<&str> for TensorMemory {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            #[cfg(target_os = "linux")]
            "dma" => Ok(TensorMemory::Dma),
            #[cfg(target_os = "linux")]
            "shm" => Ok(TensorMemory::Shm),
            "mem" => Ok(TensorMemory::Mem),
            _ => Err(Error::InvalidMemoryType(s.to_owned())),
        }
    }
}

#[derive(Debug)]
pub enum Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    #[cfg(target_os = "linux")]
    Dma(DmaTensor<T>),
    #[cfg(target_os = "linux")]
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
}

impl<T> Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Create a new tensor with the given shape, memory type, and optional
    /// name. If no name is given, a random name will be generated. If no
    /// memory type is given, the best available memory type will be chosen
    /// based on the platform and environment variables.
    ///
    /// On Linux platforms, the order of preference is: Dma -> Shm -> Mem.
    /// On non-Linux platforms, only Mem is available.
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
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Dma) => DmaTensor::<T>::new(shape, name).map(Tensor::Dma),
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Shm) => ShmTensor::<T>::new(shape, name).map(Tensor::Shm),
            Some(TensorMemory::Mem) => MemTensor::<T>::new(shape, name).map(Tensor::Mem),
            None => {
                if std::env::var("EDGEFIRST_TENSOR_FORCE_MEM")
                    .is_ok_and(|x| x != "0" && x.to_lowercase() != "false")
                {
                    MemTensor::<T>::new(shape, name).map(Tensor::Mem)
                } else {
                    #[cfg(target_os = "linux")]
                    match DmaTensor::<T>::new(shape, name) {
                        Ok(tensor) => Ok(Tensor::Dma(tensor)),
                        Err(_) => match ShmTensor::<T>::new(shape, name).map(Tensor::Shm) {
                            Ok(tensor) => Ok(tensor),
                            Err(_) => MemTensor::<T>::new(shape, name).map(Tensor::Mem),
                        },
                    }
                    #[cfg(not(target_os = "linux"))]
                    MemTensor::<T>::new(shape, name).map(Tensor::Mem)
                }
            }
        }
    }
}

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, None, name)
    }

    #[cfg(target_os = "linux")]
    /// Create a new tensor using the given file descriptor, shape, and optional
    /// name. If no name is given, a random name will be generated.
    ///
    /// Inspects the file descriptor to determine the appropriate tensor type
    /// (Dma or Shm) based on the device major and minor numbers.
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
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
                DmaTensor::<T>::from_fd(fd, shape, name).map(Tensor::Dma)
            }
            _ => {
                // other minor numbers are assumed to be shared memory
                ShmTensor::<T>::from_fd(fd, shape, name).map(Tensor::Shm)
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn clone_fd(&self) -> Result<OwnedFd> {
        match self {
            Tensor::Dma(t) => t.clone_fd(),
            Tensor::Shm(t) => t.clone_fd(),
            Tensor::Mem(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            #[cfg(target_os = "linux")]
            Tensor::Dma(_) => TensorMemory::Dma,
            #[cfg(target_os = "linux")]
            Tensor::Shm(_) => TensorMemory::Shm,
            Tensor::Mem(_) => TensorMemory::Mem,
        }
    }

    fn name(&self) -> String {
        match self {
            #[cfg(target_os = "linux")]
            Tensor::Dma(t) => t.name(),
            #[cfg(target_os = "linux")]
            Tensor::Shm(t) => t.name(),
            Tensor::Mem(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(target_os = "linux")]
            Tensor::Dma(t) => t.shape(),
            #[cfg(target_os = "linux")]
            Tensor::Shm(t) => t.shape(),
            Tensor::Mem(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            #[cfg(target_os = "linux")]
            Tensor::Dma(t) => t.reshape(shape),
            #[cfg(target_os = "linux")]
            Tensor::Shm(t) => t.reshape(shape),
            Tensor::Mem(t) => t.reshape(shape),
        }
    }

    fn map(&self) -> Result<TensorMap<T>> {
        match self {
            #[cfg(target_os = "linux")]
            Tensor::Dma(t) => t.map(),
            #[cfg(target_os = "linux")]
            Tensor::Shm(t) => t.map(),
            Tensor::Mem(t) => t.map(),
        }
    }
}

pub enum TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    #[cfg(target_os = "linux")]
    Dma(DmaMap<T>),
    #[cfg(target_os = "linux")]
    Shm(ShmMap<T>),
    Mem(MemMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.shape(),
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.shape(),
            TensorMap::Mem(map) => map.shape(),
        }
    }

    fn unmap(&mut self) {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.unmap(),
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.unmap(),
            TensorMap::Mem(map) => map.unmap(),
        }
    }

    fn as_slice(&self) -> &[T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_slice(),
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.as_slice(),
            TensorMap::Mem(map) => map.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            #[cfg(target_os = "linux")]
            TensorMap::Dma(map) => map.as_mut_slice(),
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.as_mut_slice(),
            TensorMap::Mem(map) => map.as_mut_slice(),
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
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.deref(),
            TensorMap::Mem(map) => map.deref(),
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
            #[cfg(target_os = "linux")]
            TensorMap::Shm(map) => map.deref_mut(),
            TensorMap::Mem(map) => map.deref_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use nix::unistd::{AccessFlags, access};
    use std::io::Write as _;

    use super::*;

    #[ctor::ctor]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_tensor() {
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
    #[cfg(not(target_os = "linux"))]
    fn test_tensor() {
        let shape = vec![1];
        let tensor = Tensor::<f32>::new(&shape, None, None).expect("Failed to create tensor");
        assert_eq!(tensor.memory(), TensorMemory::Mem);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_dma_tensor() {
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
    #[cfg(target_os = "linux")]
    fn test_shm_tensor() {
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

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray() {
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
}
