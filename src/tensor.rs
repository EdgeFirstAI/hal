use crate::error::{Error, Result};
use log::warn;
use nix::{fcntl::OFlag, unistd::ftruncate};
use num_traits::Num;
use std::{
    ffi::c_void,
    num::NonZero,
    ops::{Deref, DerefMut},
    os::fd::OwnedFd,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

pub trait TensorTrait<T>
where
    T: Num + Clone + std::fmt::Debug,
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
    T: Num + Clone + std::fmt::Debug,
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
}

pub enum Tensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    Shm(ShmTensor<T>),
}

impl<T> Tensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    pub fn new(shape: &[usize], memory: TensorMemory, name: Option<&str>) -> Result<Self> {
        match memory {
            TensorMemory::Dma => Err(Error::NotImplemented(
                "DMA tensor not implemented".to_string(),
            )),
            TensorMemory::Shm => ShmTensor::<T>::new(shape, name).map(Tensor::Shm),
        }
    }
}

impl<T> TensorTrait<T> for Tensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, TensorMemory::Shm, name)
    }

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        ShmTensor::<T>::from_fd(fd, shape, name).map(Tensor::Shm)
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        match self {
            Tensor::Shm(t) => t.clone_fd(),
        }
    }

    fn memory(&self) -> TensorMemory {
        match self {
            Tensor::Shm(t) => t.memory(),
        }
    }

    fn name(&self) -> String {
        match self {
            Tensor::Shm(t) => t.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Tensor::Shm(t) => t.shape(),
        }
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        match self {
            Tensor::Shm(t) => t.reshape(shape),
        }
    }

    fn map(&self) -> Result<TensorMap<T>> {
        match self {
            Tensor::Shm(t) => t.map(),
        }
    }
}

pub enum TensorMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    Shm(ShmMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            TensorMap::Shm(map) => &map.shape,
        }
    }

    fn unmap(&mut self) {
        match self {
            TensorMap::Shm(map) => map.unmap(),
        }
    }
}

impl<T> Deref for TensorMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            TensorMap::Shm(map) => map.deref(),
        }
    }
}

impl<T> DerefMut for TensorMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            TensorMap::Shm(map) => map.deref_mut(),
        }
    }
}

pub struct ShmTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    pub name: String,
    pub fd: OwnedFd,
    pub shape: Vec<usize>,
    pub _marker: std::marker::PhantomData<T>,
}

impl<T> TensorTrait<T> for ShmTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let name = match name {
            Some(name) => name.to_owned(),
            None => format!(
                "/{}",
                random_string::generate(16, random_string::charsets::ALPHANUMERIC)
            ),
        };

        let shm_fd = nix::sys::mman::shm_open(
            name.as_str(),
            OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_RDWR,
            nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
        )?;

        println!("Creating shared memory: {}", name);

        // We drop the shared memory object name after creating it to avoid
        // leaving it in the system after the program exits.  The sharing model
        // for the library is through file descriptors, not names.
        let err = nix::sys::mman::shm_unlink(name.as_str());
        if let Err(e) = err {
            warn!("Failed to unlink shared memory: {}", e);
        }

        ftruncate(&shm_fd, size as i64)?;

        Ok(ShmTensor::<T> {
            name: name.to_owned(),
            fd: shm_fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
        })
    }

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        Ok(ShmTensor {
            name: name.unwrap_or("").to_owned(),
            fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
        })
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        Ok(self.fd.try_clone()?)
    }

    fn memory(&self) -> TensorMemory {
        TensorMemory::Shm
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
            return Err(Error::ShapeVolumeMismatch);
        }

        self.shape = shape.to_vec();
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        let size = NonZero::new(self.size()).ok_or(Error::InvalidSize(self.size()))?;
        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                size,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &self.fd,
                0,
            )?
        };

        println!("Mapping shared memory: {:?}", ptr);

        Ok(TensorMap::Shm(ShmMap {
            ptr: Arc::new(Mutex::new(
                NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(self.size()))?,
            )),
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
        }))
    }
}

pub struct ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    ptr: Arc<Mutex<NonNull<c_void>>>,
    shape: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

unsafe impl<T> Send for ShmMap<T> where T: Num + Clone + std::fmt::Debug {}
unsafe impl<T> Sync for ShmMap<T> where T: Num + Clone + std::fmt::Debug {}

impl<T> Deref for ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        let ptr = self
            .ptr
            .lock()
            .expect("Failed to lock ShmMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }
}

impl<T> DerefMut for ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        let ptr = self
            .ptr
            .lock()
            .expect("Failed to lock ShmMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
    }
}

impl<T> TensorMapTrait<T> for ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        println!("Unmapping shared memory...");
        let ptr = self
            .ptr
            .lock()
            .expect("Failed to lock ShmMap pointer");
        let err = unsafe { nix::sys::mman::munmap(*ptr, self.size()) };
        if let Err(e) = err {
            warn!("Failed to unmap shared memory: {}", e);
        }
    }
}

impl<T> Drop for ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn drop(&mut self) {
        println!("ShmMap dropped, unmapping memory: {:?}", self.to_vec());
        self.unmap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let shared = ShmTensor::<f32>::from_fd(
                tensor
                    .clone_fd()
                    .expect("Failed to duplicate tensor file descriptor"),
                &shape,
                Some("test_tensor_shared"),
            )
            .expect("Failed to create tensor from fd");

            let mut tensor_map = shared.map().expect("Failed to map shared memory from fd");
            tensor_map.fill(3.14);
            assert!(tensor_map.iter().all(|&x| x == 3.14));
        }

        {
            let tensor_map = tensor.map().expect("Failed to map shared memory");
            println!("Mapped tensor: {:?}", tensor_map.to_vec());
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
            assert_eq!(tensor_map[2], 42, "Value at index 2 should be 42");
            println!("Mapped tensor: {:?}", tensor_map.to_vec());
        }
    }
}
