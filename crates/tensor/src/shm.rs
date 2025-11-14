// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    TensorMap, TensorMapTrait, TensorMemory, TensorTrait,
    error::{Error, Result},
};
use log::{debug, trace, warn};
use nix::{fcntl::OFlag, sys::stat::fstat, unistd::ftruncate};
use num_traits::Num;
use std::{
    ffi::c_void,
    fmt,
    num::NonZero,
    ops::{Deref, DerefMut},
    os::fd::{AsRawFd, OwnedFd},
    ptr::NonNull,
    sync::{Arc, Mutex},
};

pub struct ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    pub name: String,
    pub fd: OwnedFd,
    pub shape: Vec<usize>,
    pub _marker: std::marker::PhantomData<T>,
}

unsafe impl<T> Send for ShmTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
unsafe impl<T> Sync for ShmTensor<T> where T: Num + Clone + fmt::Debug + Send + Sync {}
impl<T> TensorTrait<T> for ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(shape: &[usize], name: Option<&str>) -> Result<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let name = match name {
            Some(name) => name.to_owned(),
            None => {
                let uuid = uuid::Uuid::new_v4().as_simple().to_string();
                format!("/{}", &uuid[..16])
            }
        };

        let shm_fd = nix::sys::mman::shm_open(
            name.as_str(),
            OFlag::O_CREAT | OFlag::O_EXCL | OFlag::O_RDWR,
            nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
        )?;

        trace!("Creating shared memory: {name}");

        // We drop the shared memory object name after creating it to avoid
        // leaving it in the system after the program exits.  The sharing model
        // for the library is through file descriptors, not names.
        let err = nix::sys::mman::shm_unlink(name.as_str());
        if let Err(e) = err {
            warn!("Failed to unlink shared memory: {e}");
        }

        ftruncate(&shm_fd, size as i64)?;
        let stat = fstat(&shm_fd)?;
        debug!("Shared memory stat: {stat:?}");

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
            return Err(Error::ShapeMismatch(format!(
                "Cannot reshape incompatible shape: {:?} to {:?}",
                self.shape, shape
            )));
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

        trace!("Mapping shared memory: {ptr:?}");
        let shm_ptr = ShmPtr(NonNull::new(ptr.as_ptr()).ok_or(Error::InvalidSize(self.size()))?);
        Ok(TensorMap::Shm(ShmMap {
            ptr: Arc::new(Mutex::new(shm_ptr)),
            shape: self.shape.clone(),
            _marker: std::marker::PhantomData,
        }))
    }
}

impl<T> AsRawFd for ShmTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.fd.as_raw_fd()
    }
}

pub struct ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    ptr: Arc<Mutex<ShmPtr>>,
    shape: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Deref for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

struct ShmPtr(NonNull<c_void>);
impl Deref for ShmPtr {
    type Target = NonNull<c_void>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for ShmPtr {}

impl<T> TensorMapTrait<T> for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn unmap(&mut self) {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        let err = unsafe { nix::sys::mman::munmap(**ptr, self.size()) };
        if let Err(e) = err {
            warn!("Failed to unmap shared memory: {e}");
        }
    }

    fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.ptr.lock().expect("Failed to lock ShmMap pointer");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, self.len()) }
    }
}

impl<T> Drop for ShmMap<T>
where
    T: Num + Clone + fmt::Debug,
{
    fn drop(&mut self) {
        trace!("ShmMap dropped, unmapping memory");
        self.unmap();
    }
}
