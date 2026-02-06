// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tensor C API - Type-generic tensor operations with DMA/SHM support.
//!
//! This module provides a type-erased tensor API that supports multiple data types
//! through a single set of functions. Callers query the tensor's dtype and cast
//! the void* data pointer appropriately.

use crate::error::{c_str_to_option, set_error, set_error_null, str_to_c_string};
use crate::{check_null, check_null_ret_null, try_or_null};
use edgefirst_tensor::{Tensor, TensorMap, TensorMapTrait, TensorMemory, TensorTrait};
use libc::{c_char, c_int, size_t};

#[cfg(target_os = "linux")]
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

/// Data type of tensor elements.
///
/// Used to query the type of elements stored in a tensor and interpret
/// the void* data pointer returned by hal_tensor_map_data().
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalDtype {
    /// Unsigned 8-bit integer (uint8_t)
    U8 = 0,
    /// Signed 8-bit integer (int8_t)
    I8 = 1,
    /// Unsigned 16-bit integer (uint16_t)
    U16 = 2,
    /// Signed 16-bit integer (int16_t)
    I16 = 3,
    /// Unsigned 32-bit integer (uint32_t)
    U32 = 4,
    /// Signed 32-bit integer (int32_t)
    I32 = 5,
    /// Unsigned 64-bit integer (uint64_t)
    U64 = 6,
    /// Signed 64-bit integer (int64_t)
    I64 = 7,
    /// 32-bit floating point (float)
    F32 = 9,
    /// 64-bit floating point (double)
    F64 = 10,
}

impl HalDtype {
    /// Returns the size in bytes of this data type
    pub fn size(&self) -> usize {
        match self {
            HalDtype::U8 | HalDtype::I8 => 1,
            HalDtype::U16 | HalDtype::I16 => 2,
            HalDtype::U32 | HalDtype::I32 | HalDtype::F32 => 4,
            HalDtype::U64 | HalDtype::I64 | HalDtype::F64 => 8,
        }
    }
}

/// Memory allocation type for tensors.
///
/// Controls how tensor memory is allocated. DMA is recommended for hardware
/// acceleration, while MEM is the fallback for all platforms.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalTensorMemory {
    /// Regular system memory allocation (available on all platforms)
    Mem = 0,
    /// Direct Memory Access allocation (Linux only, enables hardware acceleration)
    Dma = 1,
    /// POSIX Shared Memory allocation (Linux only, for IPC)
    Shm = 2,
}

impl From<HalTensorMemory> for Option<TensorMemory> {
    fn from(mem: HalTensorMemory) -> Self {
        match mem {
            HalTensorMemory::Mem => Some(TensorMemory::Mem),
            #[cfg(target_os = "linux")]
            HalTensorMemory::Dma => Some(TensorMemory::Dma),
            #[cfg(target_os = "linux")]
            HalTensorMemory::Shm => Some(TensorMemory::Shm),
            #[cfg(not(target_os = "linux"))]
            _ => Some(TensorMemory::Mem),
        }
    }
}

impl From<TensorMemory> for HalTensorMemory {
    fn from(mem: TensorMemory) -> Self {
        match mem {
            TensorMemory::Mem => HalTensorMemory::Mem,
            #[cfg(target_os = "linux")]
            TensorMemory::Dma => HalTensorMemory::Dma,
            #[cfg(target_os = "linux")]
            TensorMemory::Shm => HalTensorMemory::Shm,
        }
    }
}

/// Type-erased tensor that can hold any supported data type.
///
/// This is an opaque type - use hal_tensor_* functions to interact with it.
pub enum HalTensor {
    U8(Tensor<u8>),
    I8(Tensor<i8>),
    U16(Tensor<u16>),
    I16(Tensor<i16>),
    U32(Tensor<u32>),
    I32(Tensor<i32>),
    U64(Tensor<u64>),
    I64(Tensor<i64>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
}

impl HalTensor {
    pub fn dtype(&self) -> HalDtype {
        match self {
            HalTensor::U8(_) => HalDtype::U8,
            HalTensor::I8(_) => HalDtype::I8,
            HalTensor::U16(_) => HalDtype::U16,
            HalTensor::I16(_) => HalDtype::I16,
            HalTensor::U32(_) => HalDtype::U32,
            HalTensor::I32(_) => HalDtype::I32,
            HalTensor::U64(_) => HalDtype::U64,
            HalTensor::I64(_) => HalDtype::I64,
            HalTensor::F32(_) => HalDtype::F32,
            HalTensor::F64(_) => HalDtype::F64,
        }
    }
}

/// Macro to dispatch a method call to all tensor variants
macro_rules! dispatch_tensor {
    ($self:expr, |$t:ident| $body:expr) => {
        match $self {
            HalTensor::U8($t) => $body,
            HalTensor::I8($t) => $body,
            HalTensor::U16($t) => $body,
            HalTensor::I16($t) => $body,
            HalTensor::U32($t) => $body,
            HalTensor::I32($t) => $body,
            HalTensor::U64($t) => $body,
            HalTensor::I64($t) => $body,
            HalTensor::F32($t) => $body,
            HalTensor::F64($t) => $body,
        }
    };
}

/// Type-erased tensor map for CPU access to tensor data.
///
/// This is an opaque type - use hal_tensor_map_* functions to interact with it.
pub enum HalTensorMap {
    U8(TensorMap<u8>),
    I8(TensorMap<i8>),
    U16(TensorMap<u16>),
    I16(TensorMap<i16>),
    U32(TensorMap<u32>),
    I32(TensorMap<i32>),
    U64(TensorMap<u64>),
    I64(TensorMap<i64>),
    F32(TensorMap<f32>),
    F64(TensorMap<f64>),
}

/// Macro to dispatch a method call to all tensor map variants
macro_rules! dispatch_map {
    ($self:expr, |$m:ident| $body:expr) => {
        match $self {
            HalTensorMap::U8($m) => $body,
            HalTensorMap::I8($m) => $body,
            HalTensorMap::U16($m) => $body,
            HalTensorMap::I16($m) => $body,
            HalTensorMap::U32($m) => $body,
            HalTensorMap::I32($m) => $body,
            HalTensorMap::U64($m) => $body,
            HalTensorMap::I64($m) => $body,
            HalTensorMap::F32($m) => $body,
            HalTensorMap::F64($m) => $body,
        }
    };
}

// ============================================================================
// Tensor Lifecycle Functions
// ============================================================================

/// Create a new tensor with the given data type, shape, and memory type.
///
/// @param dtype Data type of tensor elements (HAL_DTYPE_*)
/// @param shape Array of dimension sizes (ndim elements)
/// @param ndim Number of dimensions (1-8)
/// @param memory Memory allocation type (HAL_TENSOR_DMA recommended)
/// @param name Optional tensor name for debugging (can be NULL)
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL shape, ndim is 0, invalid dtype)
/// - ENOMEM: Memory allocation failed
/// - EIO: DMA heap not available (falls back to SHM/MEM)
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_new(
    dtype: HalDtype,
    shape: *const size_t,
    ndim: size_t,
    memory: HalTensorMemory,
    name: *const c_char,
) -> *mut HalTensor {
    check_null_ret_null!(shape);
    if ndim == 0 || ndim > 8 {
        return set_error_null(libc::EINVAL);
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let name_opt = unsafe { c_str_to_option(name) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let tensor = match dtype {
        HalDtype::U8 => try_or_null!(
            Tensor::<u8>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::U8),
        HalDtype::I8 => try_or_null!(
            Tensor::<i8>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::I8),
        HalDtype::U16 => try_or_null!(
            Tensor::<u16>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::U16),
        HalDtype::I16 => try_or_null!(
            Tensor::<i16>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::I16),
        HalDtype::U32 => try_or_null!(
            Tensor::<u32>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::U32),
        HalDtype::I32 => try_or_null!(
            Tensor::<i32>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::I32),
        HalDtype::U64 => try_or_null!(
            Tensor::<u64>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::U64),
        HalDtype::I64 => try_or_null!(
            Tensor::<i64>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::I64),
        HalDtype::F32 => try_or_null!(
            Tensor::<f32>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::F32),
        HalDtype::F64 => try_or_null!(
            Tensor::<f64>::new(shape_slice, mem_opt, name_opt),
            libc::ENOMEM
        )
        .pipe(HalTensor::F64),
    };

    Box::into_raw(Box::new(tensor))
}

/// Create a new tensor from an existing file descriptor (Linux only).
///
/// Takes ownership of the file descriptor - caller must NOT close it.
///
/// @param dtype Data type of tensor elements (HAL_DTYPE_*)
/// @param fd File descriptor for DMA/SHM buffer (ownership transferred)
/// @param shape Array of dimension sizes (ndim elements)
/// @param ndim Number of dimensions (1-8)
/// @param name Optional tensor name for debugging (can be NULL)
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL shape, ndim is 0, invalid fd)
/// - ENOMEM: Memory allocation failed
/// - ENOTSUP: Not supported on this platform (non-Linux)
#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn hal_tensor_from_fd(
    dtype: HalDtype,
    fd: c_int,
    shape: *const size_t,
    ndim: size_t,
    name: *const c_char,
) -> *mut HalTensor {
    check_null_ret_null!(shape);
    if ndim == 0 || ndim > 8 || fd < 0 {
        return set_error_null(libc::EINVAL);
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let name_opt = unsafe { c_str_to_option(name) };
    let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };

    let tensor = match dtype {
        HalDtype::U8 => try_or_null!(
            Tensor::<u8>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::U8),
        HalDtype::I8 => try_or_null!(
            Tensor::<i8>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::I8),
        HalDtype::U16 => try_or_null!(
            Tensor::<u16>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::U16),
        HalDtype::I16 => try_or_null!(
            Tensor::<i16>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::I16),
        HalDtype::U32 => try_or_null!(
            Tensor::<u32>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::U32),
        HalDtype::I32 => try_or_null!(
            Tensor::<i32>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::I32),
        HalDtype::U64 => try_or_null!(
            Tensor::<u64>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::U64),
        HalDtype::I64 => try_or_null!(
            Tensor::<i64>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::I64),
        HalDtype::F32 => try_or_null!(
            Tensor::<f32>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::F32),
        HalDtype::F64 => try_or_null!(
            Tensor::<f64>::from_fd(owned_fd, shape_slice, name_opt),
            libc::EIO
        )
        .pipe(HalTensor::F64),
    };

    Box::into_raw(Box::new(tensor))
}

/// Create a new tensor from an existing file descriptor (stub for non-Linux).
#[no_mangle]
#[cfg(not(target_os = "linux"))]
pub unsafe extern "C" fn hal_tensor_from_fd(
    _dtype: HalDtype,
    _fd: c_int,
    _shape: *const size_t,
    _ndim: size_t,
    _name: *const c_char,
) -> *mut HalTensor {
    set_error_null(libc::ENOTSUP)
}

/// Free a tensor and release its resources.
///
/// After calling this function, the tensor pointer becomes invalid.
///
/// @param tensor Tensor handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_free(tensor: *mut HalTensor) {
    if !tensor.is_null() {
        drop(unsafe { Box::from_raw(tensor) });
    }
}

// ============================================================================
// Tensor Property Functions
// ============================================================================

/// Get the data type of a tensor.
///
/// @param tensor Tensor handle
/// @return Data type enum value, or HAL_DTYPE_U8 if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_dtype(tensor: *const HalTensor) -> HalDtype {
    if tensor.is_null() {
        return HalDtype::U8;
    }
    unsafe { &*tensor }.dtype()
}

/// Get the size in bytes of the tensor's data type.
///
/// @param tensor Tensor handle
/// @return Size of one element in bytes, or 0 if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_dtype_size(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    unsafe { &*tensor }.dtype().size()
}

/// Get the memory allocation type of a tensor.
///
/// @param tensor Tensor handle
/// @return Memory type enum value, or HAL_TENSOR_MEM if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_memory_type(tensor: *const HalTensor) -> HalTensorMemory {
    if tensor.is_null() {
        return HalTensorMemory::Mem;
    }
    dispatch_tensor!(unsafe { &*tensor }, |t| t.memory().into())
}

/// Get the name of a tensor.
///
/// The returned string is owned by the caller and must be freed with free().
///
/// @param tensor Tensor handle
/// @return Newly allocated C string with tensor name, or NULL on error
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_name(tensor: *const HalTensor) -> *mut c_char {
    if tensor.is_null() {
        return std::ptr::null_mut();
    }
    let name = dispatch_tensor!(unsafe { &*tensor }, |t| t.name());
    str_to_c_string(&name)
}

/// Get the shape of a tensor.
///
/// The returned pointer is borrowed and valid only during the tensor's lifetime.
///
/// @param tensor Tensor handle
/// @param out_ndim Output parameter for number of dimensions (can be NULL)
/// @return Pointer to shape array, or NULL if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_shape(
    tensor: *const HalTensor,
    out_ndim: *mut size_t,
) -> *const size_t {
    if tensor.is_null() {
        if !out_ndim.is_null() {
            unsafe { *out_ndim = 0 };
        }
        return std::ptr::null();
    }

    let shape = dispatch_tensor!(unsafe { &*tensor }, |t| t.shape());
    if !out_ndim.is_null() {
        unsafe { *out_ndim = shape.len() };
    }
    shape.as_ptr()
}

/// Get the total number of elements in a tensor.
///
/// @param tensor Tensor handle
/// @return Number of elements (product of all dimensions), or 0 if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_len(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    dispatch_tensor!(unsafe { &*tensor }, |t| t.len())
}

/// Get the total size in bytes of a tensor's data.
///
/// @param tensor Tensor handle
/// @return Size in bytes (len * dtype_size), or 0 if tensor is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_size(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    dispatch_tensor!(unsafe { &*tensor }, |t| t.size())
}

/// Clone the file descriptor associated with a tensor (Linux only).
///
/// Creates a new owned file descriptor that the caller must close().
///
/// @param tensor Tensor handle
/// @return New file descriptor on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL tensor
/// - ENOTSUP: Tensor memory type doesn't support file descriptors, or non-Linux
/// - EIO: Failed to clone file descriptor
#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn hal_tensor_clone_fd(tensor: *const HalTensor) -> c_int {
    check_null!(tensor);
    let result = dispatch_tensor!(unsafe { &*tensor }, |t| t.clone_fd());
    match result {
        Ok(fd) => fd.into_raw_fd(),
        Err(_) => set_error(libc::EIO),
    }
}

/// Clone file descriptor stub for non-Linux platforms.
#[no_mangle]
#[cfg(not(target_os = "linux"))]
pub unsafe extern "C" fn hal_tensor_clone_fd(_tensor: *const HalTensor) -> c_int {
    set_error(libc::ENOTSUP)
}

/// Reshape a tensor to a new shape.
///
/// The total number of elements must remain the same.
///
/// @param tensor Tensor handle
/// @param shape New shape array
/// @param ndim Number of dimensions in new shape
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL tensor/shape, ndim is 0, or element count mismatch
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_reshape(
    tensor: *mut HalTensor,
    shape: *const size_t,
    ndim: size_t,
) -> c_int {
    check_null!(tensor, shape);
    if ndim == 0 || ndim > 8 {
        return set_error(libc::EINVAL);
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };

    let result = dispatch_tensor!(unsafe { &mut *tensor }, |t| t.reshape(shape_slice));
    match result {
        Ok(()) => 0,
        Err(_) => set_error(libc::EINVAL),
    }
}

// ============================================================================
// Tensor Map Functions
// ============================================================================

/// Map a tensor for CPU access.
///
/// This function maps the tensor's memory for CPU read/write operations.
/// For DMA tensors, this includes automatic cache synchronization.
/// The returned map must be unmapped with hal_tensor_map_unmap() when done.
///
/// @param tensor Tensor handle
/// @return Tensor map handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL tensor
/// - EIO: Failed to map tensor memory
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_create(tensor: *const HalTensor) -> *mut HalTensorMap {
    check_null_ret_null!(tensor);

    let map = match unsafe { &*tensor } {
        HalTensor::U8(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::U8),
        HalTensor::I8(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::I8),
        HalTensor::U16(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::U16),
        HalTensor::I16(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::I16),
        HalTensor::U32(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::U32),
        HalTensor::I32(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::I32),
        HalTensor::U64(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::U64),
        HalTensor::I64(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::I64),
        HalTensor::F32(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::F32),
        HalTensor::F64(t) => try_or_null!(t.map(), libc::EIO).pipe(HalTensorMap::F64),
    };

    Box::into_raw(Box::new(map))
}

/// Get a mutable pointer to the mapped tensor data.
///
/// The returned pointer is valid only while the map exists.
/// Cast the void* to the appropriate type based on hal_tensor_dtype().
///
/// @param map Tensor map handle
/// @return Pointer to data, or NULL if map is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_data(map: *mut HalTensorMap) -> *mut libc::c_void {
    if map.is_null() {
        return std::ptr::null_mut();
    }
    dispatch_map!(unsafe { &mut *map }, |m| m.as_mut_slice().as_mut_ptr()
        as *mut libc::c_void)
}

/// Get a const pointer to the mapped tensor data.
///
/// The returned pointer is valid only while the map exists.
/// Cast the void* to the appropriate type based on hal_tensor_dtype().
///
/// @param map Tensor map handle
/// @return Pointer to data, or NULL if map is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_data_const(
    map: *const HalTensorMap,
) -> *const libc::c_void {
    if map.is_null() {
        return std::ptr::null();
    }
    dispatch_map!(unsafe { &*map }, |m| m.as_slice().as_ptr()
        as *const libc::c_void)
}

/// Get the shape of a mapped tensor.
///
/// @param map Tensor map handle
/// @param out_ndim Output parameter for number of dimensions (can be NULL)
/// @return Pointer to shape array, or NULL if map is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_shape(
    map: *const HalTensorMap,
    out_ndim: *mut size_t,
) -> *const size_t {
    if map.is_null() {
        if !out_ndim.is_null() {
            unsafe { *out_ndim = 0 };
        }
        return std::ptr::null();
    }

    let shape = dispatch_map!(unsafe { &*map }, |m| m.shape());
    if !out_ndim.is_null() {
        unsafe { *out_ndim = shape.len() };
    }
    shape.as_ptr()
}

/// Get the total number of elements in a mapped tensor.
///
/// @param map Tensor map handle
/// @return Number of elements, or 0 if map is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_len(map: *const HalTensorMap) -> size_t {
    if map.is_null() {
        return 0;
    }
    dispatch_map!(unsafe { &*map }, |m| m.len())
}

/// Get the total size in bytes of mapped tensor data.
///
/// @param map Tensor map handle
/// @return Size in bytes, or 0 if map is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_size(map: *const HalTensorMap) -> size_t {
    if map.is_null() {
        return 0;
    }
    dispatch_map!(unsafe { &*map }, |m| m.size())
}

/// Unmap a tensor and release the mapping.
///
/// For DMA tensors, this includes automatic cache synchronization.
/// After calling this function, the map pointer becomes invalid.
///
/// @param map Tensor map handle to unmap (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_map_unmap(map: *mut HalTensorMap) {
    if !map.is_null() {
        let mut boxed = unsafe { Box::from_raw(map) };
        dispatch_map!(&mut *boxed, |m| m.unmap());
        // Box is dropped here, releasing the map
    }
}

// ============================================================================
// Helper trait for method chaining
// ============================================================================

trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_create_and_free() {
        unsafe {
            let shape: [size_t; 3] = [2, 3, 4];
            let tensor = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            assert_eq!(hal_tensor_dtype(tensor), HalDtype::F32);
            assert_eq!(hal_tensor_dtype_size(tensor), 4);
            assert_eq!(hal_tensor_len(tensor), 24);
            assert_eq!(hal_tensor_size(tensor), 96);

            let mut ndim: size_t = 0;
            let shape_ptr = hal_tensor_shape(tensor, &mut ndim);
            assert_eq!(ndim, 3);
            assert_eq!(*shape_ptr, 2);
            assert_eq!(*shape_ptr.add(1), 3);
            assert_eq!(*shape_ptr.add(2), 4);

            hal_tensor_free(tensor);
        }
    }

    #[test]
    fn test_tensor_map() {
        unsafe {
            let shape: [size_t; 2] = [10, 10];
            let tensor = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            let map = hal_tensor_map_create(tensor);
            assert!(!map.is_null());

            let data = hal_tensor_map_data(map) as *mut f32;
            assert!(!data.is_null());
            assert_eq!(hal_tensor_map_len(map), 100);
            assert_eq!(hal_tensor_map_size(map), 400);

            // Write some data
            *data = 42.0;
            assert_eq!(*data, 42.0);

            hal_tensor_map_unmap(map);
            hal_tensor_free(tensor);
        }
    }

    #[test]
    fn test_tensor_reshape() {
        unsafe {
            let shape: [size_t; 2] = [6, 4];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            let new_shape: [size_t; 3] = [2, 3, 4];
            let result = hal_tensor_reshape(tensor, new_shape.as_ptr(), new_shape.len());
            assert_eq!(result, 0);

            let mut ndim: size_t = 0;
            let shape_ptr = hal_tensor_shape(tensor, &mut ndim);
            assert_eq!(ndim, 3);
            assert_eq!(*shape_ptr, 2);

            hal_tensor_free(tensor);
        }
    }

    #[test]
    fn test_null_handling() {
        unsafe {
            // These should not crash
            assert_eq!(hal_tensor_dtype(std::ptr::null()), HalDtype::U8);
            assert_eq!(hal_tensor_len(std::ptr::null()), 0);
            hal_tensor_free(std::ptr::null_mut());
            hal_tensor_map_unmap(std::ptr::null_mut());
        }
    }
}
