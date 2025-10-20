use edgefirst::{
    image::{self, Crop, Flip, ImageConverterTrait, RGBA, Rect, Rotation},
    tensor::{self, TensorMapTrait, TensorMemory, TensorTrait},
};
use four_char_code::FourCharCode;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use numpy::{IntoPyArray, PyArray3, PyArrayLike3, PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::{
    fmt::{self},
    sync::Mutex,
};

use crate::tensor::{PyTensorMap, TensorMapT};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Image(image::Error),
    Tensor(edgefirst::tensor::Error),
    NdArrayShape(ndarray::ShapeError),
    Io(std::io::Error),
    Format(String),
    Shape(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Image(e) => write!(f, "Image error: {e:?}"),
            Error::Tensor(e) => write!(f, "Tensor error: {e:?}"),
            Error::NdArrayShape(e) => write!(f, "Ndarray shape error: {e:?}"),
            Error::Io(e) => write!(f, "Io error: {e:?}"),
            Error::Format(msg) => write!(f, "Format error: {msg}"),
            Error::Shape(msg) => write!(f, "Shape error: {msg}"),
        }
    }
}

impl From<image::Error> for Error {
    fn from(err: image::Error) -> Self {
        Error::Image(err)
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

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::NdArrayShape(err)
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}"))
    }
}

#[derive(FromPyObject)]
pub enum ImageDest3<'py> {
    UInt8(PyReadwriteArray3<'py, u8>),
    Int8(PyReadwriteArray3<'py, i8>),
    Float32(PyReadwriteArray3<'py, f32>),
    Float64(PyReadwriteArray3<'py, f64>),
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

#[pymethods]
impl FourCC {
    #[new]
    pub fn new(fourcc: &str) -> Result<Self> {
        Self::try_from(fourcc)
    }
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
            _ => Err(Error::Format(value.to_string())),
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

#[pyclass(name = "TensorMemory")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTensorMemory {
    #[cfg(target_os = "linux")]
    DMA,
    #[cfg(target_os = "linux")]
    SHM,
    MEM,
}

impl From<PyTensorMemory> for TensorMemory {
    fn from(value: PyTensorMemory) -> Self {
        match value {
            #[cfg(target_os = "linux")]
            PyTensorMemory::DMA => TensorMemory::Dma,
            #[cfg(target_os = "linux")]
            PyTensorMemory::SHM => TensorMemory::Shm,
            PyTensorMemory::MEM => TensorMemory::Mem,
        }
    }
}

#[pyclass(name = "TensorImage")]
pub struct PyTensorImage(image::TensorImage);

#[pymethods]
impl PyTensorImage {
    #[new]
    #[pyo3(signature = (width, height, fourcc = FourCC::RGB, mem = None))]
    pub fn new(
        width: usize,
        height: usize,
        fourcc: FourCC,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        let fourcc: FourCharCode = fourcc.into();
        let mem = mem.map(|x| x.into());
        let tensor_image = image::TensorImage::new(width, height, fourcc, mem)?;
        Ok(PyTensorImage(tensor_image))
    }

    #[staticmethod]
    #[pyo3(signature = (data, fourcc = Some(FourCC::RGB), mem = None))]
    pub fn load_from_bytes(
        data: &[u8],
        fourcc: Option<FourCC>,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        let fourcc = fourcc.map(|f| f.into());
        let mem = mem.map(|x| x.into());
        let tensor_image = image::TensorImage::load(data, fourcc, mem)?;
        Ok(PyTensorImage(tensor_image))
    }

    #[staticmethod]
    #[pyo3(signature = (filename, fourcc = Some(FourCC::RGB), mem = None))]
    pub fn load(
        filename: &str,
        fourcc: Option<FourCC>,
        mem: Option<PyTensorMemory>,
    ) -> Result<Self> {
        let fourcc = fourcc.map(|f| f.into());
        let data = std::fs::read(filename)?;
        let mem = mem.map(|x| x.into());
        let tensor_image = image::TensorImage::load(&data, fourcc, mem)?;
        Ok(PyTensorImage(tensor_image))
    }

    #[pyo3(signature = (filename, quality=80))]
    pub fn save_jpeg(&self, filename: &str, quality: u8) -> Result<()> {
        self.0.save_jpeg(filename, quality)?;
        Ok(())
    }

    pub fn to_numpy<'py>(self_: PyRef<'py, Self>) -> Result<Bound<'py, PyArray3<u8>>> {
        let map = self_.0.tensor().map()?;
        let data = map.as_slice().to_vec();
        let ndarray = Array3::from_shape_vec(
            [self_.0.height(), self_.0.width(), self_.0.channels()],
            data,
        )?;
        Ok(ndarray.into_pyarray(self_.py()))
    }

    pub fn copy_into_numpy(&self, dst: ImageDest3) -> Result<()> {
        let tensor = &self.0;
        let shape = [tensor.height(), tensor.width(), tensor.channels()];
        let dst_shape = match &dst {
            ImageDest3::UInt8(dst) => dst.shape(),
            ImageDest3::Int8(dst) => dst.shape(),
            ImageDest3::Float32(dst) => dst.shape(),
            ImageDest3::Float64(dst) => dst.shape(),
        }
        .to_vec();

        if dst_shape[..2] != shape[..2] {
            return Err(Error::Format(format!(
                "Shape Mismatch: Expected {:?} but got {:?}",
                shape, dst_shape
            )));
        }

        match self.0.fourcc() {
            RGBA => {
                if dst_shape[2] != 4 && dst_shape[2] != 3 {
                    return Err(Error::Format(format!(
                        "Shape Mismatch: Expected {:?} but got {:?}",
                        shape, dst_shape
                    )));
                }
            }
            _ => {
                if dst_shape[2] != shape[2] {
                    return Err(Error::Format(format!(
                        "Shape Mismatch: Expected {:?} but got {:?}",
                        shape, dst_shape
                    )));
                }
            }
        }

        match dst {
            ImageDest3::UInt8(mut dst) => {
                let mut dst = dst.as_array_mut();
                let map = tensor.tensor().map()?;
                let data = map.as_slice();
                let ndarray = ArrayView3::from_shape(shape, data)?;

                if self.0.fourcc() == RGBA
                    && dst_shape[2] == 3
                    && let Some(dst) = dst.as_slice_mut()
                {
                    let dst = dst.as_chunks_mut::<3>().0;
                    let src = data.as_chunks::<4>().0;
                    dst.iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                    });
                    return Ok(());
                }
                dst.assign(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]));
            }
            ImageDest3::Int8(mut dst) => {
                let mut dst = dst.as_array_mut();
                let map = tensor.tensor().map()?;
                let data = map.as_slice();
                let ndarray = ArrayView3::from_shape(shape, data)?;
                if self.0.fourcc() == RGBA
                    && dst_shape[2] == 3
                    && let Some(dst) = dst.as_slice_mut()
                {
                    let dst = dst.as_chunks_mut::<3>().0;
                    let src = data.as_chunks::<4>().0;
                    dst.iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = (s[0] as i16 - 128) as i8;
                        d[1] = (s[1] as i16 - 128) as i8;
                        d[2] = (s[2] as i16 - 128) as i8;
                    });
                    return Ok(());
                }
                dst.zip_mut_with(
                    &ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]),
                    |x, y| *x = (*y as i16 - 128) as i8,
                );
            }
            ImageDest3::Float32(_dst) => todo!(),
            ImageDest3::Float64(_dst) => todo!(),
        }

        Ok(())
    }

    pub fn copy_from_numpy(&mut self, src: PyArrayLike3<u8>) -> Result<()> {
        let src = src.as_array();
        let tensor = &self.0;
        let shape = [tensor.height(), tensor.width(), tensor.channels()];
        if src.shape() != shape {
            return Err(Error::Format(format!(
                "Shape Mismatch: Expected {:?} but got {:?}",
                shape,
                src.shape()
            )));
        }

        let mut map = tensor.tensor().map()?;
        let data = map.as_mut_slice();
        let mut ndarray = ArrayViewMut3::from_shape(shape, data)?;
        ndarray.assign(&src);
        Ok(())
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

    pub fn map(&self) -> Result<PyTensorMap> {
        Ok(PyTensorMap {
            mapped: Some(self.0.tensor().map().map(TensorMapT::TensorU8)?),
        })
    }
}

#[pyclass(name = "ImageConverter")]
pub struct PyImageConverter(Mutex<image::ImageConverter>);

unsafe impl Send for PyImageConverter {}
unsafe impl Sync for PyImageConverter {}

#[pymethods]
impl PyImageConverter {
    #[new]
    pub fn new() -> Result<Self> {
        let converter = image::ImageConverter::new()?;
        Ok(PyImageConverter(Mutex::new(converter)))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, src_crop = None, dst_crop = None, dst_color = None))]
    pub fn convert(
        &mut self,
        src: &PyTensorImage,
        dst: &mut PyTensorImage,
        rotation: PyRotation,
        flip: PyFlip,
        src_crop: Option<PyRect>,
        dst_crop: Option<PyRect>,
        dst_color: Option<[u8; 4]>,
    ) -> Result<()> {
        if let Ok(mut l) = self.0.lock() {
            l.convert(
                &src.0,
                &mut dst.0,
                rotation.into(),
                flip.into(),
                Crop {
                    src_rect: src_crop.map(|x| x.into()),
                    dst_rect: dst_crop.map(|x| x.into()),
                    dst_color,
                },
            )?
        };
        Ok(())
    }
}

#[pyclass(name = "Rotation")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyRotation {
    Rotate0 = 0,
    Clockwise90 = 1,
    Rotate180 = 2,
    CounterClockwise90 = 3,
}

#[pymethods]
impl PyRotation {
    #[staticmethod]
    pub fn degrees_clockwise(angle: usize) -> PyRotation {
        match angle.rem_euclid(90) {
            0 => PyRotation::Rotate0,
            90 => PyRotation::Clockwise90,
            180 => PyRotation::Rotate180,
            270 => PyRotation::CounterClockwise90,
            _ => panic!("rotation angle is not a multiple of 90"),
        }
    }
}

impl From<PyRotation> for Rotation {
    fn from(val: PyRotation) -> Self {
        match val {
            PyRotation::Rotate0 => Rotation::None,
            PyRotation::Clockwise90 => Rotation::Clockwise90,
            PyRotation::Rotate180 => Rotation::Rotate180,
            PyRotation::CounterClockwise90 => Rotation::CounterClockwise90,
        }
    }
}

#[pyclass(name = "Flip")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyFlip {
    NoFlip = 0,
    Horizontal = 1,
    Vertical = 2,
}

impl From<PyFlip> for Flip {
    fn from(val: PyFlip) -> Self {
        match val {
            PyFlip::NoFlip => Flip::None,
            PyFlip::Horizontal => Flip::Horizontal,
            PyFlip::Vertical => Flip::Vertical,
        }
    }
}

#[pyclass(name = "Rect")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyRect {
    #[pyo3(get, set)]
    pub left: usize,
    #[pyo3(get, set)]
    pub top: usize,
    #[pyo3(get, set)]
    pub width: usize,
    #[pyo3(get, set)]
    pub height: usize,
}

#[pymethods]
impl PyRect {
    #[new]
    pub fn new(left: usize, top: usize, width: usize, height: usize) -> PyRect {
        PyRect {
            left,
            top,
            width,
            height,
        }
    }
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
