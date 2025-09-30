use crate::{
    TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
    error::{Error, Result},
};
use log::trace;
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::{Arc, Mutex},
};

pub struct MemTensor<T>
where
    T: Num + Clone + fmt::Debug,
{
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> TensorTrait<T> for MemTensor<T>
where
    T: Num + Clone + fmt::Debug,
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

    #[cfg(target_os = "linux")]
    fn from_fd(_fd: std::os::fd::OwnedFd, _shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "MemTensor does not support from_fd".to_owned(),
        ))
    }

    #[cfg(target_os = "linux")]
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
        let mem_ptr = MemPtr(
            NonNull::new(self.data.as_ptr() as *mut c_void)
                .ok_or(Error::InvalidSize(self.size()))?,
        );
        Ok(TensorMap::Mem(MemMap {
            ptr: Arc::new(Mutex::new(mem_ptr)),
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
        }))
    }
}

pub struct MemMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    ptr: Arc<Mutex<MemPtr>>,
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

struct MemPtr(NonNull<c_void>);
impl Deref for MemPtr {
    type Target = NonNull<c_void>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for MemPtr {}

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
        let ptr = self.ptr.lock().expect("Failed to lock MemMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock MemMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
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
