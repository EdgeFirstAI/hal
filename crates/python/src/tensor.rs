use edgefirst::{
    tensor,
    tensor::{TensorMapTrait as _, TensorTrait as _},
};
use pyo3::{exceptions::PyBufferError, ffi::PyMemoryView_FromMemory, prelude::*};

#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use pyo3::ffi::Py_buffer;

#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use std::ffi::{CString, c_int, c_void};
use std::{fmt, os::raw::c_char};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Tensor(tensor::Error),
    UnsupportedMemoryType(String),
    UnsupportedDataType(String),
    TensorMap(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tensor(e) => write!(f, "Tensor error: {e:?}"),
            Error::UnsupportedMemoryType(msg) => write!(f, "Invalid memory type: {msg}"),
            Error::UnsupportedDataType(msg) => write!(f, "Invalid data type: {msg}"),
            Error::TensorMap(msg) => write!(f, "Tensor map error: {msg}"),
        }
    }
}

impl From<tensor::Error> for Error {
    fn from(err: tensor::Error) -> Self {
        Error::Tensor(err)
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}"))
    }
}

pub enum TensorT {
    TensorU8(tensor::Tensor<u8>),
    TensorI8(tensor::Tensor<i8>),
    TensorU16(tensor::Tensor<u16>),
    TensorI16(tensor::Tensor<i16>),
    TensorU32(tensor::Tensor<u32>),
    TensorI32(tensor::Tensor<i32>),
    TensorU64(tensor::Tensor<u64>),
    TensorI64(tensor::Tensor<i64>),
    TensorF32(tensor::Tensor<f32>),
    TensorF64(tensor::Tensor<f64>),
}

impl TensorT {
    pub fn dtype(&self) -> String {
        match self {
            TensorT::TensorU8(_) => "uint8".to_string(),
            TensorT::TensorI8(_) => "int8".to_string(),
            TensorT::TensorU16(_) => "uint16".to_string(),
            TensorT::TensorI16(_) => "int16".to_string(),
            TensorT::TensorU32(_) => "uint32".to_string(),
            TensorT::TensorI32(_) => "int32".to_string(),
            TensorT::TensorU64(_) => "uint64".to_string(),
            TensorT::TensorI64(_) => "int64".to_string(),
            TensorT::TensorF32(_) => "float32".to_string(),
            TensorT::TensorF64(_) => "float64".to_string(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorT::TensorU8(t) => t.size(),
            TensorT::TensorI8(t) => t.size(),
            TensorT::TensorU16(t) => t.size(),
            TensorT::TensorI16(t) => t.size(),
            TensorT::TensorU32(t) => t.size(),
            TensorT::TensorI32(t) => t.size(),
            TensorT::TensorU64(t) => t.size(),
            TensorT::TensorI64(t) => t.size(),
            TensorT::TensorF32(t) => t.size(),
            TensorT::TensorF64(t) => t.size(),
        }
    }

    pub fn memory(&self) -> tensor::TensorMemory {
        match self {
            TensorT::TensorU8(t) => t.memory(),
            TensorT::TensorI8(t) => t.memory(),
            TensorT::TensorU16(t) => t.memory(),
            TensorT::TensorI16(t) => t.memory(),
            TensorT::TensorU32(t) => t.memory(),
            TensorT::TensorI32(t) => t.memory(),
            TensorT::TensorU64(t) => t.memory(),
            TensorT::TensorI64(t) => t.memory(),
            TensorT::TensorF32(t) => t.memory(),
            TensorT::TensorF64(t) => t.memory(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            TensorT::TensorU8(t) => t.name(),
            TensorT::TensorI8(t) => t.name(),
            TensorT::TensorU16(t) => t.name(),
            TensorT::TensorI16(t) => t.name(),
            TensorT::TensorU32(t) => t.name(),
            TensorT::TensorI32(t) => t.name(),
            TensorT::TensorU64(t) => t.name(),
            TensorT::TensorI64(t) => t.name(),
            TensorT::TensorF32(t) => t.name(),
            TensorT::TensorF64(t) => t.name(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            TensorT::TensorU8(t) => t.shape(),
            TensorT::TensorI8(t) => t.shape(),
            TensorT::TensorU16(t) => t.shape(),
            TensorT::TensorI16(t) => t.shape(),
            TensorT::TensorU32(t) => t.shape(),
            TensorT::TensorI32(t) => t.shape(),
            TensorT::TensorU64(t) => t.shape(),
            TensorT::TensorI64(t) => t.shape(),
            TensorT::TensorF32(t) => t.shape(),
            TensorT::TensorF64(t) => t.shape(),
        }
    }

    pub fn reshape(&mut self, shape: &[usize]) -> tensor::Result<()> {
        match self {
            TensorT::TensorU8(t) => t.reshape(shape),
            TensorT::TensorI8(t) => t.reshape(shape),
            TensorT::TensorU16(t) => t.reshape(shape),
            TensorT::TensorI16(t) => t.reshape(shape),
            TensorT::TensorU32(t) => t.reshape(shape),
            TensorT::TensorI32(t) => t.reshape(shape),
            TensorT::TensorU64(t) => t.reshape(shape),
            TensorT::TensorI64(t) => t.reshape(shape),
            TensorT::TensorF32(t) => t.reshape(shape),
            TensorT::TensorF64(t) => t.reshape(shape),
        }
    }

    pub fn map(&self) -> tensor::Result<TensorMapT> {
        match self {
            TensorT::TensorU8(t) => t.map().map(TensorMapT::TensorU8),
            TensorT::TensorI8(t) => t.map().map(TensorMapT::TensorI8),
            TensorT::TensorU16(t) => t.map().map(TensorMapT::TensorU16),
            TensorT::TensorI16(t) => t.map().map(TensorMapT::TensorI16),
            TensorT::TensorU32(t) => t.map().map(TensorMapT::TensorU32),
            TensorT::TensorI32(t) => t.map().map(TensorMapT::TensorI32),
            TensorT::TensorU64(t) => t.map().map(TensorMapT::TensorU64),
            TensorT::TensorI64(t) => t.map().map(TensorMapT::TensorI64),
            TensorT::TensorF32(t) => t.map().map(TensorMapT::TensorF32),
            TensorT::TensorF64(t) => t.map().map(TensorMapT::TensorF64),
        }
    }
}

pub enum TensorMapT {
    TensorU8(tensor::TensorMap<u8>),
    TensorI8(tensor::TensorMap<i8>),
    TensorU16(tensor::TensorMap<u16>),
    TensorI16(tensor::TensorMap<i16>),
    TensorU32(tensor::TensorMap<u32>),
    TensorI32(tensor::TensorMap<i32>),
    TensorU64(tensor::TensorMap<u64>),
    TensorI64(tensor::TensorMap<i64>),
    TensorF32(tensor::TensorMap<f32>),
    TensorF64(tensor::TensorMap<f64>),
}

impl TensorMapT {
    pub fn unmap(&mut self) {
        match self {
            TensorMapT::TensorU8(map) => map.unmap(),
            TensorMapT::TensorI8(map) => map.unmap(),
            TensorMapT::TensorU16(map) => map.unmap(),
            TensorMapT::TensorI16(map) => map.unmap(),
            TensorMapT::TensorU32(map) => map.unmap(),
            TensorMapT::TensorI32(map) => map.unmap(),
            TensorMapT::TensorU64(map) => map.unmap(),
            TensorMapT::TensorI64(map) => map.unmap(),
            TensorMapT::TensorF32(map) => map.unmap(),
            TensorMapT::TensorF64(map) => map.unmap(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            TensorMapT::TensorU8(map) => map.shape(),
            TensorMapT::TensorI8(map) => map.shape(),
            TensorMapT::TensorU16(map) => map.shape(),
            TensorMapT::TensorI16(map) => map.shape(),
            TensorMapT::TensorU32(map) => map.shape(),
            TensorMapT::TensorI32(map) => map.shape(),
            TensorMapT::TensorU64(map) => map.shape(),
            TensorMapT::TensorI64(map) => map.shape(),
            TensorMapT::TensorF32(map) => map.shape(),
            TensorMapT::TensorF64(map) => map.shape(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorMapT::TensorU8(map) => map.size(),
            TensorMapT::TensorI8(map) => map.size(),
            TensorMapT::TensorU16(map) => map.size(),
            TensorMapT::TensorI16(map) => map.size(),
            TensorMapT::TensorU32(map) => map.size(),
            TensorMapT::TensorI32(map) => map.size(),
            TensorMapT::TensorU64(map) => map.size(),
            TensorMapT::TensorI64(map) => map.size(),
            TensorMapT::TensorF32(map) => map.size(),
            TensorMapT::TensorF64(map) => map.size(),
        }
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    pub fn element_size(&self) -> usize {
        match self {
            TensorMapT::TensorU8(_) => std::mem::size_of::<u8>(),
            TensorMapT::TensorI8(_) => std::mem::size_of::<i8>(),
            TensorMapT::TensorU16(_) => std::mem::size_of::<u16>(),
            TensorMapT::TensorI16(_) => std::mem::size_of::<i16>(),
            TensorMapT::TensorU32(_) => std::mem::size_of::<u32>(),
            TensorMapT::TensorI32(_) => std::mem::size_of::<i32>(),
            TensorMapT::TensorU64(_) => std::mem::size_of::<u64>(),
            TensorMapT::TensorI64(_) => std::mem::size_of::<i64>(),
            TensorMapT::TensorF32(_) => std::mem::size_of::<f32>(),
            TensorMapT::TensorF64(_) => std::mem::size_of::<f64>(),
        }
    }
}

#[pyclass(name = "Tensor")]
pub struct PyTensor(TensorT);

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (shape, dtype = "float32", memory = None, name = None))]
    fn __init__(
        shape: Vec<usize>,
        dtype: &str,
        memory: Option<&str>,
        name: Option<&str>,
    ) -> Result<Self> {
        let memory = match memory {
            #[cfg(target_os = "linux")]
            Some("dma") => Some(tensor::TensorMemory::Dma),
            #[cfg(target_os = "linux")]
            Some("shm") => Some(tensor::TensorMemory::Shm),
            Some(inv) => return Err(Error::UnsupportedMemoryType(inv.to_string())),
            None => None,
        };

        let tensor = match dtype {
            "uint8" => TensorT::TensorU8(tensor::Tensor::new(&shape, memory, name)?),
            "int8" => TensorT::TensorI8(tensor::Tensor::new(&shape, memory, name)?),
            "uint16" => TensorT::TensorU16(tensor::Tensor::new(&shape, memory, name)?),
            "int16" => TensorT::TensorI16(tensor::Tensor::new(&shape, memory, name)?),
            "uint32" => TensorT::TensorU32(tensor::Tensor::new(&shape, memory, name)?),
            "int32" => TensorT::TensorI32(tensor::Tensor::new(&shape, memory, name)?),
            "uint64" => TensorT::TensorU64(tensor::Tensor::new(&shape, memory, name)?),
            "int64" => TensorT::TensorI64(tensor::Tensor::new(&shape, memory, name)?),
            "float32" => TensorT::TensorF32(tensor::Tensor::new(&shape, memory, name)?),
            "float64" => TensorT::TensorF64(tensor::Tensor::new(&shape, memory, name)?),
            _ => return Err(Error::UnsupportedDataType(dtype.to_string())),
        };

        Ok(PyTensor(tensor))
    }

    #[getter]
    fn dtype(&self) -> String {
        self.0.dtype()
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size()
    }

    #[getter]
    fn memory(&self) -> String {
        match self.0.memory() {
            #[cfg(target_os = "linux")]
            tensor::TensorMemory::Shm => "shm".to_string(),
            #[cfg(target_os = "linux")]
            tensor::TensorMemory::Dma => "dma".to_string(),
            tensor::TensorMemory::Mem => "mem".to_string(),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }

    #[getter]
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        Ok(self.0.reshape(&shape)?)
    }

    fn map(&self) -> Result<TensorMap> {
        Ok(TensorMap {
            mapped: Some(self.0.map()?),
        })
    }
}

#[pyclass]
struct TensorMap {
    mapped: Option<TensorMapT>,
}

unsafe impl Send for TensorMap {}
unsafe impl Sync for TensorMap {}

#[pymethods]
impl TensorMap {
    fn unmap(&mut self) {
        if let Some(map) = &mut self.mapped {
            map.unmap();
            self.mapped = None;
        }
    }

    fn __repr__(&self) -> String {
        match &self.mapped {
            Some(TensorMapT::TensorU8(map)) => {
                format!(
                    "TensorMap(dtype=uint8, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorI8(map)) => {
                format!(
                    "TensorMap(dtype=int8, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorU16(map)) => {
                format!(
                    "TensorMap(dtype=uint16, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorI16(map)) => {
                format!(
                    "TensorMap(dtype=int16, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorU32(map)) => {
                format!(
                    "TensorMap(dtype=uint32, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorI32(map)) => {
                format!(
                    "TensorMap(dtype=int32, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorU64(map)) => {
                format!(
                    "TensorMap(dtype=uint64, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorI64(map)) => {
                format!(
                    "TensorMap(dtype=int64, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorF32(map)) => {
                format!(
                    "TensorMap(dtype=float32, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            Some(TensorMapT::TensorF64(map)) => {
                format!(
                    "TensorMap(dtype=float64, shape={:?}) => {:?}",
                    map.shape(),
                    map.to_vec()
                )
            }
            None => "Unmapped Tensor".to_string(),
        }
    }

    fn __len__(&self) -> usize {
        if let Some(map) = &self.mapped {
            map.shape().iter().product()
        } else {
            0
        }
    }

    fn __getitem__(&self, index: usize, py: Python) -> PyResult<PyObject> {
        if let Some(map) = &self.mapped {
            match map {
                TensorMapT::TensorU8(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI8(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU16(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI16(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU32(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI32(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU64(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI64(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorF32(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorF64(m) => {
                    if index < m.size() {
                        Ok(m.as_ref()[index].into_pyobject(py)?.into())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
            }
        } else {
            Err(PyBufferError::new_err("Buffer not mapped"))
        }
    }

    fn __setitem__(&mut self, index: usize, value: PyObject, py: Python) -> PyResult<()> {
        if let Some(map) = &mut self.mapped {
            match map {
                TensorMapT::TensorU8(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<u8>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI8(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<i8>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU16(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<u16>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI16(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<i16>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU32(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<u32>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI32(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<i32>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorU64(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<u64>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorI64(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<i64>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorF32(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<f32>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
                TensorMapT::TensorF64(m) => {
                    if index < m.size() {
                        m.as_mut()[index] = value.extract::<f64>(py)?;
                        Ok(())
                    } else {
                        Err(PyBufferError::new_err("Index out of bounds"))
                    }
                }
            }
        } else {
            Err(PyBufferError::new_err("Buffer not mapped"))
        }
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut Py_buffer,
        _flags: c_int,
    ) -> PyResult<()> {
        let slf2 = slf.borrow();

        if slf2.mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        let mapped = slf2.mapped.as_ref().unwrap();
        let mut shape = mapped
            .shape()
            .iter()
            .map(|&s| s as isize)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Create a memory view from the mapped buffer
        let ptr = match mapped {
            TensorMapT::TensorU8(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI8(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU16(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI16(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU64(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI64(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorF32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorF64(m) => m.as_ref().as_ptr() as *mut c_void,
        };

        let format = match mapped {
            TensorMapT::TensorU8(_) => CString::new("B").unwrap(),
            TensorMapT::TensorI8(_) => CString::new("b").unwrap(),
            TensorMapT::TensorU16(_) => CString::new("H").unwrap(),
            TensorMapT::TensorI16(_) => CString::new("h").unwrap(),
            TensorMapT::TensorU32(_) => CString::new("I").unwrap(),
            TensorMapT::TensorI32(_) => CString::new("i").unwrap(),
            TensorMapT::TensorU64(_) => CString::new("Q").unwrap(),
            TensorMapT::TensorI64(_) => CString::new("q").unwrap(),
            TensorMapT::TensorF32(_) => CString::new("f").unwrap(),
            TensorMapT::TensorF64(_) => CString::new("d").unwrap(),
        };

        unsafe {
            (*view).buf = ptr;
            (*view).len = mapped.size() as isize;
            (*view).itemsize = mapped.element_size() as isize;
            (*view).readonly = 0;

            (*view).format = format.into_raw(); // dropped in __releasebuffer__

            (*view).ndim = shape.len() as i32;
            (*view).shape = shape.as_mut_ptr();
            std::mem::forget(shape); // dropped in __releasebuffer__

            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();

            (*view).obj = slf.into_ptr();
        }

        Ok(())
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __releasebuffer__(&mut self, view: *mut Py_buffer) {
        drop(unsafe { CString::from_raw((*view).format) });
        drop(unsafe { Box::from_raw((*view).shape) });
    }

    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        if slf.borrow().mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        Ok(slf)
    }

    fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.mapped = None; // Release the mapped buffer
    }

    fn view(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if self.mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        let mapped = self.mapped.as_ref().unwrap();

        // Create a memory view from the mapped buffer
        let ptr = match mapped {
            TensorMapT::TensorU8(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorI8(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorU16(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorI16(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorU32(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorI32(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorU64(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorI64(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorF32(m) => m.as_ref().as_ptr() as *mut c_char,
            TensorMapT::TensorF64(m) => m.as_ref().as_ptr() as *mut c_char,
        };

        let mem = unsafe {
            PyMemoryView_FromMemory(
                ptr,
                self.mapped.as_ref().unwrap().size() as isize,
                0x200, // ffi::PyBUF_WRITE,
            )
        };

        unsafe { PyObject::from_owned_ptr_or_err(py, mem) }
    }
}
