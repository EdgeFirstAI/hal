pub use error::{Error, Result};

use crate::{
    dma::{DmaMap, DmaTensor},
    mem::{MemMap, MemTensor},
    shm::{ShmMap, ShmTensor},
};
use nix::sys::stat::fstat;
use num_traits::Num;
use std::{
    ops::{Deref, DerefMut},
    os::fd::OwnedFd,
};

mod dma;
mod dmabuf;
mod error;
mod mem;
mod shm;

#[cfg(target_os = "linux")]
use nix::sys::stat::{major, minor};

pub trait TensorTrait<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self>
    where
        Self: Sized;

    fn clone_fd(&self) -> Result<OwnedFd>;

    fn memory(&self) -> TensorMemory;

    fn name(&self) -> String;

    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn shape(&self) -> &[usize];

    fn reshape(&mut self, shape: &[usize]) -> Result<()>;

    fn map(&self) -> Result<TensorMap<T>>;
}

pub trait TensorMapTrait<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    fn shape(&self) -> &[usize];

    fn unmap(&mut self);

    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    fn size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMemory {
    Dma,
    Shm,
    Mem,
}

impl From<TensorMemory> for String {
    fn from(memory: TensorMemory) -> Self {
        match memory {
            TensorMemory::Dma => "dma".to_owned(),
            TensorMemory::Shm => "shm".to_owned(),
            TensorMemory::Mem => "mem".to_owned(),
        }
    }
}

impl TryFrom<&str> for TensorMemory {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s {
            "dma" => Ok(TensorMemory::Dma),
            "shm" => Ok(TensorMemory::Shm),
            "mem" => Ok(TensorMemory::Mem),
            _ => Err(Error::InvalidMemoryType(s.to_owned())),
        }
    }
}

pub enum Tensor<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    Dma(DmaTensor<T>),
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
}

impl<T> Tensor<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    pub fn new(shape: &[usize], memory: Option<TensorMemory>, name: Option<&str>) -> Result<Self> {
        match memory {
            Some(TensorMemory::Dma) => DmaTensor::<T>::new(shape, name).map(Tensor::Dma),
            Some(TensorMemory::Shm) => ShmTensor::<T>::new(shape, name).map(Tensor::Shm),
            Some(TensorMemory::Mem) => MemTensor::<T>::new(shape, name).map(Tensor::Mem),
            None => match DmaTensor::<T>::new(shape, name) {
                Ok(tensor) => Ok(Tensor::Dma(tensor)),
                Err(_) => match ShmTensor::<T>::new(shape, name).map(Tensor::Shm) {
                    Ok(tensor) => Ok(tensor),
                    Err(_) => MemTensor::<T>::new(shape, name).map(Tensor::Mem),
                },
            },
        }
    }
}

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, None, name)
    }

    #[cfg(target_os = "linux")]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        use log::debug;

        let stat = fstat(&fd)?;
        let major = major(stat.st_dev);
        let minor = minor(stat.st_dev);

        debug!("Creating tensor from fd: major={major}, minor={minor}");

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

    #[cfg(not(target_os = "linux"))]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        // Default to shared memory for non-Linux platforms
        ShmTensor::<T>::from_fd(fd, shape, name).map(Tensor::Shm)
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        match self {
            Tensor::Dma(t) => t.clone_fd(),
            Tensor::Shm(t) => t.clone_fd(),
            Tensor::Mem(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            Tensor::Dma(_) => TensorMemory::Dma,
            Tensor::Shm(_) => TensorMemory::Shm,
            Tensor::Mem(_) => TensorMemory::Mem,
        }
    }

    fn name(&self) -> String {
        match self {
            Tensor::Dma(t) => t.name(),
            Tensor::Shm(t) => t.name(),
            Tensor::Mem(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Tensor::Dma(t) => t.shape(),
            Tensor::Shm(t) => t.shape(),
            Tensor::Mem(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            Tensor::Dma(t) => t.reshape(shape),
            Tensor::Shm(t) => t.reshape(shape),
            Tensor::Mem(t) => t.reshape(shape),
        }
    }

    fn map(&self) -> Result<TensorMap<T>> {
        match self {
            Tensor::Dma(t) => t.map(),
            Tensor::Shm(t) => t.map(),
            Tensor::Mem(t) => t.map(),
        }
    }
}

pub enum TensorMap<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    Dma(DmaMap<T>),
    Shm(ShmMap<T>),
    Mem(MemMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            TensorMap::Dma(map) => map.shape(),
            TensorMap::Shm(map) => map.shape(),
            TensorMap::Mem(map) => map.shape(),
        }
    }

    fn unmap(&mut self) {
        match self {
            TensorMap::Dma(map) => map.unmap(),
            TensorMap::Shm(map) => map.unmap(),
            TensorMap::Mem(map) => map.unmap(),
        }
    }
}

impl<T> Deref for TensorMap<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            TensorMap::Dma(map) => map.deref(),
            TensorMap::Shm(map) => map.deref(),
            TensorMap::Mem(map) => map.deref(),
        }
    }
}

impl<T> DerefMut for TensorMap<T>
where
    T: Num + Clone + Send + Sync + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            TensorMap::Dma(map) => map.deref_mut(),
            TensorMap::Shm(map) => map.deref_mut(),
            TensorMap::Mem(map) => map.deref_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use nix::unistd::{AccessFlags, access};
    use std::io::Write as _;

    use super::*;

    #[ctor::ctor]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    #[test]
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
            tensor_map.fill(3.14);
            assert!(tensor_map.iter().all(|&x| x == 3.14));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map DMA memory");
            assert!(tensor_map.iter().all(|&x| x == 3.14));
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
    fn test_shm_tensor() {
        let shape = vec![2, 3, 4];
        let tensor =
            ShmTensor::<f32>::new(&shape, Some("test_tensor")).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 2 * 3 * 4 * std::mem::size_of::<f32>());
        assert_eq!(tensor.name(), "test_tensor");

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
            tensor_map.fill(3.14);
            assert!(tensor_map.iter().all(|&x| x == 3.14));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map shared memory");
            assert!(tensor_map.iter().all(|&x| x == 3.14));
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
}
