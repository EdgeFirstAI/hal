use edgefirst::image::{self, ImageConverterTrait, Rect, Rotation};
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
#[allow(clippy::upper_case_acronyms)]
pub enum FourCC {
    YUYV,
    RGBA,
    RGB,
    NV12,
    GREY,
}

impl TryFrom<&str> for FourCC {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_uppercase().as_str() {
            "YUYV" => Ok(FourCC::YUYV),
            "RGBA" => Ok(FourCC::RGBA),
            "RGB" | "RGB " => Ok(FourCC::RGB),
            "NV12" => Ok(FourCC::RGB),
            "Y800" | "GREY" | "GRAY" => Ok(FourCC::GREY),
            _ => Err(Error::FormatError(value.to_string())),
        }
    }
}

impl From<FourCC> for FourCharCode {
    fn from(val: FourCC) -> Self {
        match val {
            FourCC::YUYV => image::YUYV,
            FourCC::RGBA => image::RGBA,
            FourCC::RGB => image::RGB,
            FourCC::NV12 => image::NV12,
            FourCC::GREY => image::GREY,
        }
    }
}

impl TryFrom<FourCharCode> for FourCC {
    type Error = Error;

    fn try_from(value: FourCharCode) -> Result<Self, Self::Error> {
        Self::try_from(value.to_string().to_uppercase().as_str())
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

#[pymethods]
impl PyImageConverter {
    #[pyo3(signature = (dst, src, rotation = PyRotation::None, crop = None))]
    fn convert(
        &mut self,
        dst: &mut PyTensorImage,
        src: &PyTensorImage,
        rotation: PyRotation,
        crop: Option<PyRect>,
    ) -> Result<()> {
        if let Ok(mut l) = self.0.lock() {
            l.convert(&mut dst.0, &src.0, rotation.into(), crop.map(|x| x.into()))?
        };
        Ok(())
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyRotation {
    None = 0,
    Rotate90Clockwise = 1,
    Rotate180 = 2,
    Rotate90CounterClockwise = 3,
}
impl PyRotation {
    pub fn from_degrees_clockwise(angle: usize) -> Rotation {
        match angle.rem_euclid(90) {
            0 => Rotation::None,
            90 => Rotation::Rotate90Clockwise,
            180 => Rotation::Rotate180,
            270 => Rotation::Rotate90CounterClockwise,
            _ => panic!("rotation angle is not a multiple of 90"),
        }
    }
}

impl From<PyRotation> for Rotation {
    fn from(val: PyRotation) -> Self {
        match val {
            PyRotation::None => Rotation::None,
            PyRotation::Rotate90Clockwise => Rotation::Rotate90Clockwise,
            PyRotation::Rotate180 => Rotation::Rotate180,
            PyRotation::Rotate90CounterClockwise => Rotation::Rotate90CounterClockwise,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyRect {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl From<PyRect> for Rect {
    fn from(val: PyRect) -> Self {
        Rect {
            left: val.left,
            top: val.top,
            width: val.width,
            height: val.height,
        }
    }
}
