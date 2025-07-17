use edgefirst::image;
use four_char_code::FourCharCode;
use pyo3::prelude::*;
use std::{
    fmt,
    sync::{Arc, Mutex},
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    ImageError(image::Error),
    FormatError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ImageError(e) => write!(f, "Tensor error: {e:?}"),
            Error::FormatError(msg) => write!(f, "Format error: {msg}"),
        }
    }
}

impl From<image::Error> for Error {
    fn from(err: image::Error) -> Self {
        Error::ImageError(err)
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}"))
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourCC {
    YUYV,
    RGBA,
    RGB,
}

impl TryFrom<&str> for FourCC {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "YUYV" => Ok(FourCC::YUYV),
            "RGBA" => Ok(FourCC::RGBA),
            "RGB" => Ok(FourCC::RGB),
            _ => Err(Error::FormatError(value.to_string())),
        }
    }
}

impl Into<FourCharCode> for FourCC {
    fn into(self) -> FourCharCode {
        match self {
            FourCC::YUYV => image::YUYV,
            FourCC::RGBA => image::RGBA,
            FourCC::RGB => image::RGB,
        }
    }
}

impl TryFrom<FourCharCode> for FourCC {
    type Error = Error;

    fn try_from(value: FourCharCode) -> Result<Self, Self::Error> {
        match value.to_string().as_str() {
            "YUYV" => Ok(FourCC::YUYV),
            "RGBA" => Ok(FourCC::RGBA),
            "RGB " => Ok(FourCC::RGB),
            _ => Err(Error::FormatError(value.to_string())),
        }
    }
}

#[pyclass]
pub struct PyTensorImage(image::TensorImage);

#[pymethods]
impl PyTensorImage {
    #[new]
    #[pyo3(signature = (width, height, fourcc = FourCC::RGB))]
    pub fn new(width: usize, height: usize, fourcc: FourCC) -> Result<Self> {
        let fourcc: FourCharCode = fourcc.into();
        let tensor_image = image::TensorImage::new(width, height, fourcc, None)?;
        Ok(PyTensorImage(tensor_image))
    }

    #[getter]
    pub fn format(&self) -> Result<FourCC> {
        self.0.fourcc().try_into()
    }

    #[getter]
    pub fn width(&self) -> usize {
        self.0.width()
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.0.height()
    }

    #[getter]
    pub fn is_planar(&self) -> bool {
        self.0.is_planar()
    }
}

#[pyclass]
pub struct PyImageConverter(Mutex<image::ImageConverter>);

unsafe impl Send for PyImageConverter {}
unsafe impl Sync for PyImageConverter {}
