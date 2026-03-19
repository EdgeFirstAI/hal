// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_hal::tensor::{self, DType, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use pyo3::ffi::Py_buffer;
use pyo3::{exceptions::PyBufferError, ffi::PyMemoryView_FromMemory, prelude::*};

use std::ffi::c_void;
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use std::ffi::{c_int, CString};
#[cfg(unix)]
use std::os::fd::{IntoRawFd, RawFd};

use std::{
    fmt::{self, Display},
    os::raw::c_char,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Tensor(tensor::Error),
    UnsupportedMemoryType(String),
    UnsupportedDataType(String),
    TensorMap(String),
    Format(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tensor(e) => write!(f, "Tensor error: {e:?}"),
            Error::UnsupportedMemoryType(msg) => write!(f, "Invalid memory type: {msg}"),
            Error::UnsupportedDataType(msg) => write!(f, "Invalid data type: {msg}"),
            Error::TensorMap(msg) => write!(f, "Tensor map error: {msg}"),
            Error::Format(msg) => write!(f, "Format error: {msg}"),
            Error::Io(e) => write!(f, "IO error: {e:?}"),
        }
    }
}

impl From<tensor::Error> for Error {
    fn from(err: tensor::Error) -> Self {
        Error::Tensor(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<edgefirst_hal::image::Error> for Error {
    fn from(err: edgefirst_hal::image::Error) -> Self {
        Error::Format(format!("{err:?}"))
    }
}

impl From<crate::image::Error> for Error {
    fn from(err: crate::image::Error) -> Self {
        Error::Format(format!("{err}"))
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

#[pyclass(name = "TensorMemory", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTensorMemory {
    #[cfg(target_os = "linux")]
    DMA,
    #[cfg(unix)]
    SHM,
    PBO,
    MEM,
}

impl From<PyTensorMemory> for TensorMemory {
    fn from(value: PyTensorMemory) -> Self {
        match value {
            #[cfg(target_os = "linux")]
            PyTensorMemory::DMA => TensorMemory::Dma,
            #[cfg(unix)]
            PyTensorMemory::SHM => TensorMemory::Shm,
            PyTensorMemory::PBO => TensorMemory::Pbo,
            PyTensorMemory::MEM => TensorMemory::Mem,
        }
    }
}

impl From<TensorMemory> for PyTensorMemory {
    fn from(value: TensorMemory) -> Self {
        match value {
            #[cfg(target_os = "linux")]
            TensorMemory::Dma => PyTensorMemory::DMA,
            #[cfg(unix)]
            TensorMemory::Shm => PyTensorMemory::SHM,
            TensorMemory::Pbo => PyTensorMemory::PBO,
            TensorMemory::Mem => PyTensorMemory::MEM,
        }
    }
}

/// Parse a Python dtype string (e.g. "float32", "uint8") into a `DType`.
fn parse_dtype(dtype: &str) -> Result<DType> {
    match dtype {
        "uint8" => Ok(DType::U8),
        "int8" => Ok(DType::I8),
        "uint16" => Ok(DType::U16),
        "int16" => Ok(DType::I16),
        "uint32" => Ok(DType::U32),
        "int32" => Ok(DType::I32),
        "uint64" => Ok(DType::U64),
        "int64" => Ok(DType::I64),
        "float32" => Ok(DType::F32),
        "float64" => Ok(DType::F64),
        _ => Err(Error::UnsupportedDataType(dtype.to_string())),
    }
}

/// Convert a `DType` to a Python dtype string.
fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::I8 => "int8",
        DType::U16 => "uint16",
        DType::I16 => "int16",
        DType::U32 => "uint32",
        DType::I32 => "int32",
        DType::U64 => "uint64",
        DType::I64 => "int64",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        _ => "unknown",
    }
}

// ─── Type-erased TensorMap ──────────────────────────────────────────────────
// Needed for Python buffer protocol — must dispatch per dtype to get typed
// pointers, format strings, and per-element operations.

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

/// Dispatch a method call across all TensorMapT variants.
macro_rules! map_dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            TensorMapT::TensorU8(m) => m.$method($($arg),*),
            TensorMapT::TensorI8(m) => m.$method($($arg),*),
            TensorMapT::TensorU16(m) => m.$method($($arg),*),
            TensorMapT::TensorI16(m) => m.$method($($arg),*),
            TensorMapT::TensorU32(m) => m.$method($($arg),*),
            TensorMapT::TensorI32(m) => m.$method($($arg),*),
            TensorMapT::TensorU64(m) => m.$method($($arg),*),
            TensorMapT::TensorI64(m) => m.$method($($arg),*),
            TensorMapT::TensorF32(m) => m.$method($($arg),*),
            TensorMapT::TensorF64(m) => m.$method($($arg),*),
        }
    };
}

impl TensorMapT {
    pub fn unmap(&mut self) {
        map_dispatch!(self, unmap);
    }

    pub fn shape(&self) -> &[usize] {
        map_dispatch!(self, shape)
    }

    pub fn size(&self) -> usize {
        map_dispatch!(self, size)
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

    pub fn get_value_at(&self, index: usize, py: Python) -> PyResult<Py<PyAny>> {
        if index >= self.size() {
            return Err(PyBufferError::new_err("Index out of bounds"));
        }
        match self {
            TensorMapT::TensorU8(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI8(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU16(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI16(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorF32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorF64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
        }
    }

    pub fn set_value_at(&mut self, index: usize, value: Py<PyAny>, py: Python) -> PyResult<()> {
        if index >= self.size() {
            return Err(PyBufferError::new_err("Index out of bounds"));
        }
        match self {
            TensorMapT::TensorU8(m) => m.as_mut()[index] = value.extract::<u8>(py)?,
            TensorMapT::TensorI8(m) => m.as_mut()[index] = value.extract::<i8>(py)?,
            TensorMapT::TensorU16(m) => m.as_mut()[index] = value.extract::<u16>(py)?,
            TensorMapT::TensorI16(m) => m.as_mut()[index] = value.extract::<i16>(py)?,
            TensorMapT::TensorU32(m) => m.as_mut()[index] = value.extract::<u32>(py)?,
            TensorMapT::TensorI32(m) => m.as_mut()[index] = value.extract::<i32>(py)?,
            TensorMapT::TensorU64(m) => m.as_mut()[index] = value.extract::<u64>(py)?,
            TensorMapT::TensorI64(m) => m.as_mut()[index] = value.extract::<i64>(py)?,
            TensorMapT::TensorF32(m) => m.as_mut()[index] = value.extract::<f32>(py)?,
            TensorMapT::TensorF64(m) => m.as_mut()[index] = value.extract::<f64>(py)?,
        }
        Ok(())
    }

    /// Get a raw pointer to the mapped data (for Python buffer protocol).
    fn data_ptr(&self) -> *mut c_void {
        match self {
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
        }
    }

    /// Get the struct format character for Python buffer protocol.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    fn format_str(&self) -> &'static str {
        match self {
            TensorMapT::TensorU8(_) => "B",
            TensorMapT::TensorI8(_) => "b",
            TensorMapT::TensorU16(_) => "H",
            TensorMapT::TensorI16(_) => "h",
            TensorMapT::TensorU32(_) => "I",
            TensorMapT::TensorI32(_) => "i",
            TensorMapT::TensorU64(_) => "Q",
            TensorMapT::TensorI64(_) => "q",
            TensorMapT::TensorF32(_) => "f",
            TensorMapT::TensorF64(_) => "d",
        }
    }

    fn dtype_name(&self) -> &'static str {
        match self {
            TensorMapT::TensorU8(_) => "uint8",
            TensorMapT::TensorI8(_) => "int8",
            TensorMapT::TensorU16(_) => "uint16",
            TensorMapT::TensorI16(_) => "int16",
            TensorMapT::TensorU32(_) => "uint32",
            TensorMapT::TensorI32(_) => "int32",
            TensorMapT::TensorU64(_) => "uint64",
            TensorMapT::TensorI64(_) => "int64",
            TensorMapT::TensorF32(_) => "float32",
            TensorMapT::TensorF64(_) => "float64",
        }
    }
}

/// Map a `TensorDyn` to a `TensorMapT`.
fn map_tensor_dyn(t: &TensorDyn) -> tensor::Result<TensorMapT> {
    match t {
        TensorDyn::U8(t) => t.map().map(TensorMapT::TensorU8),
        TensorDyn::I8(t) => t.map().map(TensorMapT::TensorI8),
        TensorDyn::U16(t) => t.map().map(TensorMapT::TensorU16),
        TensorDyn::I16(t) => t.map().map(TensorMapT::TensorI16),
        TensorDyn::U32(t) => t.map().map(TensorMapT::TensorU32),
        TensorDyn::I32(t) => t.map().map(TensorMapT::TensorI32),
        TensorDyn::U64(t) => t.map().map(TensorMapT::TensorU64),
        TensorDyn::I64(t) => t.map().map(TensorMapT::TensorI64),
        TensorDyn::F16(_) => Err(tensor::Error::InvalidArgument(
            "float16 tensor mapping not supported in Python".to_string(),
        )),
        TensorDyn::F32(t) => t.map().map(TensorMapT::TensorF32),
        TensorDyn::F64(t) => t.map().map(TensorMapT::TensorF64),
        _ => Err(tensor::Error::InvalidArgument(
            "unsupported dtype for tensor mapping".to_string(),
        )),
    }
}

// ─── PyTensor ───────────────────────────────────────────────────────────────

#[pyclass(name = "Tensor", str)]
pub struct PyTensor(pub(crate) TensorDyn);

impl Display for PyTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(dtype={}, shape={:?}, memory={:?})",
            dtype_to_str(self.0.dtype()),
            self.0.shape(),
            self.0.memory(),
        )
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (shape, dtype = "float32", mem = None, name = None))]
    fn __init__(
        shape: Vec<usize>,
        dtype: &str,
        mem: Option<PyTensorMemory>,
        name: Option<&str>,
    ) -> Result<Self> {
        let dt = parse_dtype(dtype)?;
        let memory = mem.map(|x| x.into());
        let tensor = TensorDyn::new(&shape, dt, memory, name)?;
        Ok(PyTensor(tensor))
    }

    #[cfg(unix)]
    #[staticmethod]
    #[pyo3(signature = (fd, shape, dtype = "float32", name = None))]
    fn from_fd(fd: RawFd, shape: Vec<usize>, dtype: &str, name: Option<&str>) -> Result<Self> {
        use std::os::fd::FromRawFd;
        if fd < 0 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file descriptor",
            )));
        }
        let dt = parse_dtype(dtype)?;
        let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
        let tensor = TensorDyn::from_fd(fd, &shape, dt, name)?;
        Ok(PyTensor(tensor))
    }

    #[getter]
    fn dtype(&self) -> String {
        dtype_to_str(self.0.dtype()).to_string()
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size()
    }

    #[getter]
    fn memory(&self) -> PyTensorMemory {
        self.0.memory().into()
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    #[cfg(unix)]
    #[getter]
    fn fd(&self) -> Result<RawFd> {
        let owned = self.0.clone_fd()?;
        Ok(owned.into_raw_fd())
    }

    fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        Ok(self.0.reshape(&shape)?)
    }

    fn map(&self) -> Result<PyTensorMap> {
        Ok(PyTensorMap {
            mapped: Some(map_tensor_dyn(&self.0)?),
        })
    }

    // ── Image-specific methods ──────────────────────────────────────────

    /// Create an image tensor with the given dimensions and pixel format.
    #[staticmethod]
    #[pyo3(signature = (width, height, format, mem = None))]
    fn image(
        width: usize,
        height: usize,
        format: crate::image::PyPixelFormat,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        use edgefirst_hal::tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        let memory = mem.map(|x| x.into());
        let tensor = TensorDyn::image(width, height, fmt, DType::U8, memory)?;
        Ok(PyTensor(tensor))
    }

    /// Load an image from a file, decoding JPEG/PNG automatically.
    #[staticmethod]
    #[pyo3(signature = (filename, format = None, mem = None))]
    fn load(
        filename: &str,
        format: Option<crate::image::PyPixelFormat>,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        use edgefirst_hal::image;
        let fmt = format.map(|f| f.into());
        let data = std::fs::read(filename)?;
        let memory = mem.map(|x| x.into());
        let tensor = image::load_image(&data, fmt, memory)?;
        Ok(PyTensor(tensor))
    }

    /// Load an image from raw bytes, decoding JPEG/PNG automatically.
    #[staticmethod]
    #[pyo3(signature = (data, format = None, mem = None))]
    fn load_from_bytes(
        data: &[u8],
        format: Option<crate::image::PyPixelFormat>,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        use edgefirst_hal::image;
        let fmt = format.map(|f| f.into());
        let memory = mem.map(|x| x.into());
        let tensor = image::load_image(data, fmt, memory)?;
        Ok(PyTensor(tensor))
    }

    /// Save this image tensor as a JPEG file.
    #[pyo3(signature = (filename, quality=80))]
    fn save_jpeg(&self, filename: &str, quality: u8) -> Result<()> {
        use edgefirst_hal::image;
        image::save_jpeg(&self.0, filename, quality)?;
        Ok(())
    }

    /// Pixel format of this tensor (None if not an image tensor).
    #[getter]
    fn format(&self) -> Option<crate::image::PyPixelFormat> {
        self.0
            .format()
            .and_then(|f| crate::image::PyPixelFormat::try_from(f).ok())
    }

    /// Image width in pixels (None if not an image tensor).
    #[getter]
    fn width(&self) -> Option<usize> {
        self.0.width()
    }

    /// Image height in pixels (None if not an image tensor).
    #[getter]
    fn height(&self) -> Option<usize> {
        self.0.height()
    }

    /// Whether this image uses a planar pixel layout.
    #[getter]
    fn is_planar(&self) -> bool {
        use edgefirst_hal::tensor::PixelLayout;
        self.0
            .format()
            .map(|f| f.layout() == PixelLayout::Planar)
            .unwrap_or(false)
    }

    /// Normalize image data and write to a numpy array.
    #[pyo3(signature = (dst, normalization=crate::image::Normalization::DEFAULT, zero_point=None))]
    fn normalize_to_numpy(
        &self,
        dst: crate::image::ImageDest3,
        normalization: crate::image::Normalization,
        zero_point: Option<i64>,
    ) -> Result<()> {
        Ok(crate::image::normalize_tensor_to_numpy(
            &self.0,
            dst,
            normalization,
            zero_point,
        )?)
    }

    /// Copy data from a numpy array into this tensor.
    fn copy_from_numpy(&mut self, src: numpy::PyArrayLike3<u8>) -> Result<()> {
        Ok(crate::image::copy_numpy_to_tensor(&self.0, src)?)
    }
}

// ─── PyTensorMap ────────────────────────────────────────────────────────────

#[pyclass(name = "TensorMap")]
pub struct PyTensorMap {
    pub(crate) mapped: Option<TensorMapT>,
}

unsafe impl Send for PyTensorMap {}
unsafe impl Sync for PyTensorMap {}

#[pymethods]
impl PyTensorMap {
    fn unmap(&mut self) {
        if let Some(map) = &mut self.mapped {
            map.unmap();
            self.mapped = None;
        }
    }

    fn __repr__(&self) -> String {
        match &self.mapped {
            Some(m) => format!("TensorMap(dtype={}, shape={:?})", m.dtype_name(), m.shape(),),
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

    fn __getitem__(&self, index: usize, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(map) = &self.mapped {
            map.get_value_at(index, py)
        } else {
            Err(PyBufferError::new_err("Buffer not mapped"))
        }
    }

    fn __setitem__(&mut self, index: usize, value: Py<PyAny>, py: Python) -> PyResult<()> {
        if let Some(map) = &mut self.mapped {
            map.set_value_at(index, value, py)
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
        let shape: Vec<isize> = mapped.shape().iter().map(|&s| s as isize).collect();
        let ndim = shape.len();

        // Compute C-contiguous strides: strides[i] = itemsize * product(shape[i+1..])
        let itemsize = mapped.element_size() as isize;
        let mut strides = vec![0isize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = itemsize;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        // Box both arrays together so we can recover the length in __releasebuffer__.
        // Store (shape_ptr, strides_ptr, ndim) using view.internal.
        let mut shape = shape.into_boxed_slice();
        let mut strides = strides.into_boxed_slice();

        let ptr = mapped.data_ptr();
        let format = CString::new(mapped.format_str()).unwrap();

        unsafe {
            (*view).buf = ptr;
            (*view).len = mapped.size() as isize;
            (*view).itemsize = itemsize;
            (*view).readonly = 0;

            (*view).format = format.into_raw(); // dropped in __releasebuffer__

            (*view).ndim = ndim as i32;
            (*view).shape = shape.as_mut_ptr();
            (*view).strides = strides.as_mut_ptr();
            // Store ndim in internal so __releasebuffer__ can reconstruct the slices.
            (*view).internal = ndim as *mut c_void;
            std::mem::forget(shape); // dropped in __releasebuffer__
            std::mem::forget(strides); // dropped in __releasebuffer__

            (*view).suboffsets = std::ptr::null_mut();

            (*view).obj = slf.into_ptr();
        }

        Ok(())
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __releasebuffer__(&mut self, view: *mut Py_buffer) {
        drop(unsafe { CString::from_raw((*view).format) });
        let ndim = unsafe { (*view).internal } as usize;
        if ndim > 0 {
            // Reconstruct the boxed slices with the correct length.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut((*view).shape, ndim)) });
            drop(unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut((*view).strides, ndim))
            });
        }
    }

    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        if slf.borrow().mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        Ok(slf)
    }

    fn __exit__(&mut self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.mapped = None; // Release the mapped buffer
    }

    fn view(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        let mapped = self.mapped.as_ref().unwrap();
        let ptr = mapped.data_ptr() as *mut c_char;

        let mem = unsafe {
            PyMemoryView_FromMemory(
                ptr,
                self.mapped.as_ref().unwrap().size() as isize,
                0x200, // ffi::PyBUF_WRITE,
            )
        };

        unsafe { Py::<PyAny>::from_owned_ptr_or_err(py, mem) }
    }
}
