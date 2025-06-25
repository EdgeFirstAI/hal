use crate::error::{Error, Result};
use log::{debug, trace, warn};
use nix::{fcntl::OFlag, sys::stat::fstat, unistd::ftruncate};
use num_traits::Num;
use std::{
    ffi::c_void,
    num::NonZero,
    ops::{Deref, DerefMut},
    os::fd::OwnedFd,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

#[cfg(target_os = "linux")]
use nix::sys::stat::{major, minor};

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
    Mem,
}

pub enum Tensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    Dma(DmaTensor<T>),
    Shm(ShmTensor<T>),
    Mem(MemTensor<T>),
}

impl<T> Tensor<T>
where
    T: Num + Clone + std::fmt::Debug,
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
    T: Num + Clone + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        Self::new(shape, None, name)
    }

    #[cfg(target_os = "linux")]
    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        let stat = fstat(&fd)?;
        let major = major(stat.st_dev);
        let minor = minor(stat.st_dev);

        if major != 0 {
            // Dma and Shm tensors are expected to have major number 0
            return Err(Error::UnknownDeviceType(major, minor));
        }

        match minor {
            10 => {
                // minor number 10 indicates DMA memory
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
    T: Num + Clone + std::fmt::Debug,
{
    Dma(DmaMap<T>),
    Shm(ShmMap<T>),
    Mem(MemMap<T>),
}

impl<T> TensorMapTrait<T> for TensorMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        match self {
            TensorMap::Dma(map) => &map.shape,
            TensorMap::Shm(map) => &map.shape,
            TensorMap::Mem(map) => &map.shape,
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
    T: Num + Clone + std::fmt::Debug,
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
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            TensorMap::Dma(map) => map.deref_mut(),
            TensorMap::Shm(map) => map.deref_mut(),
            TensorMap::Mem(map) => map.deref_mut(),
        }
    }
}

pub struct DmaTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    pub name: String,
    pub fd: OwnedFd,
    pub shape: Vec<usize>,
    pub _marker: std::marker::PhantomData<T>,
}

impl<T> TensorTrait<T> for DmaTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    #[cfg(target_os = "linux")]
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let name = match name {
            Some(name) => name.to_owned(),
            None => format!(
                "/{}",
                random_string::generate(16, random_string::charsets::ALPHANUMERIC)
            ),
        };

        let heap = match dma_heap::Heap::new(dma_heap::HeapKind::Cma) {
            Ok(heap) => heap,
            Err(_) => dma_heap::Heap::new(dma_heap::HeapKind::System)?,
        };

        let dma_fd = heap.allocate(size)?;
        let stat = fstat(&dma_fd)?;
        debug!("DMA memory stat: {:?}", stat);

        Ok(DmaTensor::<T> {
            name: name.to_owned(),
            fd: dma_fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::UnsupportedOperation(
            "DMA tensors are not supported on this platform".to_owned(),
        ))
    }

    fn from_fd(fd: OwnedFd, shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        Ok(DmaTensor {
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
        TensorMemory::Dma
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
        Ok(TensorMap::Dma(DmaMap::new(
            self.fd.try_clone()?,
            &self.shape,
        )?))
    }
}

pub struct DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    ptr: Arc<Mutex<NonNull<c_void>>>,
    fd: OwnedFd,
    shape: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

unsafe impl<T> Send for DmaMap<T> where T: Num + Clone + std::fmt::Debug {}
unsafe impl<T> Sync for DmaMap<T> where T: Num + Clone + std::fmt::Debug {}

impl<T> DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    pub fn new(fd: OwnedFd, shape: &[usize]) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        #[cfg(target_os = "linux")]
        crate::dmabuf::start_readwrite(&fd)?;

        let ptr = unsafe {
            nix::sys::mman::mmap(
                None,
                NonZero::new(size).ok_or(Error::InvalidSize(size))?,
                nix::sys::mman::ProtFlags::PROT_READ | nix::sys::mman::ProtFlags::PROT_WRITE,
                nix::sys::mman::MapFlags::MAP_SHARED,
                &fd,
                0,
            )?
        };

        trace!("Mapping DMA memory: {:?}", ptr);

        Ok(DmaMap {
            ptr: Arc::new(Mutex::new(
                NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(size))?,
            )),
            fd,
            shape: shape.to_vec(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T> Deref for DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }
}

impl<T> DerefMut for DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
    }
}

impl<T> TensorMapTrait<T> for DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        let ptr = self.ptr.lock().expect("Failed to lock DmaMap pointer");

        if let Err(e) = unsafe { nix::sys::mman::munmap(*ptr, self.size()) } {
            warn!("Failed to unmap DMA memory: {}", e);
        }

        #[cfg(target_os = "linux")]
        if let Err(e) = crate::dmabuf::end_readwrite(&self.fd) {
            warn!("Failed to end read/write on DMA memory: {}", e);
        }
    }
}

impl<T> Drop for DmaMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn drop(&mut self) {
        trace!("DmaMap dropped, unmapping memory: {:?}", self.to_vec());
        self.unmap();
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

        trace!("Creating shared memory: {}", name);

        // We drop the shared memory object name after creating it to avoid
        // leaving it in the system after the program exits.  The sharing model
        // for the library is through file descriptors, not names.
        let err = nix::sys::mman::shm_unlink(name.as_str());
        if let Err(e) = err {
            warn!("Failed to unlink shared memory: {}", e);
        }

        ftruncate(&shm_fd, size as i64)?;
        let stat = fstat(&shm_fd)?;
        debug!("Shared memory stat: {:?}", stat);

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

        trace!("Mapping shared memory: {:?}", ptr);

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
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }
}

impl<T> DerefMut for ShmMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
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
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
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
        trace!("ShmMap dropped, unmapping memory: {:?}", self.to_vec());
        self.unmap();
    }
}

pub struct MemTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> TensorTrait<T> for MemTensor<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }

        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if size == 0 {
            return Err(Error::InvalidSize(0));
        }

        let name = name.unwrap_or("mem_tensor").to_owned();
        let data = vec![T::zero(); size / std::mem::size_of::<T>()];

        Ok(MemTensor {
            name,
            shape: shape.to_vec(),
            data,
        })
    }

    fn from_fd(_fd: OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::UnsupportedOperation(
            "MemTensor does not support from_fd".to_owned(),
        ))
    }

    fn clone_fd(&self) -> Result<OwnedFd> {
        Err(Error::UnsupportedOperation(
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
            return Err(Error::ShapeVolumeMismatch);
        }

        self.shape = shape.to_vec();
        Ok(())
    }

    fn map(&self) -> Result<TensorMap<T>> {
        Ok(TensorMap::Mem(MemMap {
            ptr: Arc::new(Mutex::new(
                NonNull::new(self.data.as_ptr() as *mut c_void)
                    .ok_or(Error::InvalidSize(self.size()))?,
            )),
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
        }))
    }
}

pub struct MemMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    ptr: Arc<Mutex<NonNull<c_void>>>,
    shape: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

unsafe impl<T> Send for MemMap<T> where T: Num + Clone + std::fmt::Debug {}
unsafe impl<T> Sync for MemMap<T> where T: Num + Clone + std::fmt::Debug {}

impl<T> Deref for MemMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock MemMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }
}

impl<T> DerefMut for MemMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock MemMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
    }
}

impl<T> TensorMapTrait<T> for MemMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        trace!("Unmapping MemMap memory: {:?}", self.to_vec());
    }
}

impl<T> Drop for MemMap<T>
where
    T: Num + Clone + std::fmt::Debug,
{
    fn drop(&mut self) {
        self.unmap();
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
                    write!(
                        &mut std::io::stdout(),
                        "[WARNING] DMA Heap is unavailable: {}\n",
                        e
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
