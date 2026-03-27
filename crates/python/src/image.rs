// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::decoder::{convert_detect_box, PyDecoder};
use crate::tensor::PyTensor;
use crate::tracker::PyTrackInfo;
use edgefirst_hal::{
    decoder::{BoundingBox, DetectBox, Segmentation},
    image::{self, Crop, Flip, ImageProcessorConfig, ImageProcessorTrait, Rect, Rotation},
    tensor::{self as tensor, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait},
};

use ndarray::{
    parallel::prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    ArrayView1, ArrayView2, ArrayView3, ArrayViewMut3, Zip,
};
use numpy::{
    PyArrayLike3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use std::{
    fmt::{self},
    sync::Mutex,
};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Image(image::Error),
    Tensor(tensor::Error),
    NdArrayShape(ndarray::ShapeError),
    Io(std::io::Error),
    Format(String),
    Shape(String),
    InvalidArg(String),
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
            Error::InvalidArg(msg) => write!(f, "Invalid Argument: {msg}"),
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
    Float16(PyReadwriteArray3<'py, half::f16>),
    Float32(PyReadwriteArray3<'py, f32>),
    Float64(PyReadwriteArray3<'py, f64>),
}

/// Pixel format for image tensors.
///
/// Each variant maps directly to an `edgefirst_tensor::PixelFormat` value.
#[pyclass(name = "PixelFormat", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPixelFormat {
    Rgb = 1,
    Rgba = 2,
    Bgra = 3,
    Grey = 4,
    Yuyv = 5,
    Vyuy = 6,
    Nv12 = 7,
    Nv16 = 8,
    PlanarRgb = 9,
    PlanarRgba = 10,
}

#[pymethods]
impl PyPixelFormat {
    #[new]
    pub fn new(name: &str) -> Result<Self> {
        Self::try_from(name)
    }
}

impl TryFrom<&str> for PyPixelFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_uppercase().as_str() {
            "YUYV" => Ok(PyPixelFormat::Yuyv),
            "VYUY" => Ok(PyPixelFormat::Vyuy),
            "RGBA" => Ok(PyPixelFormat::Rgba),
            "BGRA" => Ok(PyPixelFormat::Bgra),
            "RGB" | "RGB " => Ok(PyPixelFormat::Rgb),
            "NV12" => Ok(PyPixelFormat::Nv12),
            "NV16" => Ok(PyPixelFormat::Nv16),
            "Y800" | "GREY" | "GRAY" => Ok(PyPixelFormat::Grey),
            "8BPS" | "PLANAR_RGB" | "PLANARRGB" => Ok(PyPixelFormat::PlanarRgb),
            "PLANAR_RGBA" | "PLANARRGBA" => Ok(PyPixelFormat::PlanarRgba),
            _ => Err(Error::Format(value.to_string())),
        }
    }
}

impl From<PyPixelFormat> for PixelFormat {
    fn from(val: PyPixelFormat) -> Self {
        match val {
            PyPixelFormat::Rgb => PixelFormat::Rgb,
            PyPixelFormat::Rgba => PixelFormat::Rgba,
            PyPixelFormat::Bgra => PixelFormat::Bgra,
            PyPixelFormat::Grey => PixelFormat::Grey,
            PyPixelFormat::Yuyv => PixelFormat::Yuyv,
            PyPixelFormat::Vyuy => PixelFormat::Vyuy,
            PyPixelFormat::Nv12 => PixelFormat::Nv12,
            PyPixelFormat::Nv16 => PixelFormat::Nv16,
            PyPixelFormat::PlanarRgb => PixelFormat::PlanarRgb,
            PyPixelFormat::PlanarRgba => PixelFormat::PlanarRgba,
        }
    }
}

impl TryFrom<PixelFormat> for PyPixelFormat {
    type Error = Error;

    fn try_from(val: PixelFormat) -> Result<Self, Self::Error> {
        match val {
            PixelFormat::Rgb => Ok(PyPixelFormat::Rgb),
            PixelFormat::Rgba => Ok(PyPixelFormat::Rgba),
            PixelFormat::Bgra => Ok(PyPixelFormat::Bgra),
            PixelFormat::Grey => Ok(PyPixelFormat::Grey),
            PixelFormat::Yuyv => Ok(PyPixelFormat::Yuyv),
            PixelFormat::Vyuy => Ok(PyPixelFormat::Vyuy),
            PixelFormat::Nv12 => Ok(PyPixelFormat::Nv12),
            PixelFormat::Nv16 => Ok(PyPixelFormat::Nv16),
            PixelFormat::PlanarRgb => Ok(PyPixelFormat::PlanarRgb),
            PixelFormat::PlanarRgba => Ok(PyPixelFormat::PlanarRgba),
            _ => Err(Error::Format(format!("unsupported pixel format: {val:?}"))),
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Normalization {
    DEFAULT,
    SIGNED,
    UNSIGNED,
    RAW,
}

/// Normalize image tensor data and write to a numpy array.
/// Called from PyTensor.normalize_to_numpy() in tensor.rs.
pub(crate) fn normalize_tensor_to_numpy(
    tensor_dyn: &TensorDyn,
    dst: ImageDest3,
    normalization: Normalization,
    zero_point: Option<i64>,
) -> Result<()> {
    let _timer = crate::FunctionTimer::new("normalize_to_numpy".to_string());

    let tensor_u8 = tensor_dyn
        .as_u8()
        .ok_or_else(|| Error::Format("Tensor is not U8".to_string()))?;
    let shape = tensor_u8.shape();
    let shape = [shape[0], shape[1], shape[2]];
    let dst_shape = match &dst {
        ImageDest3::UInt8(dst) => dst.shape(),
        ImageDest3::Int8(dst) => dst.shape(),
        ImageDest3::Float16(dst) => dst.shape(),
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

    let fmt = tensor_dyn.format();
    if fmt == Some(PixelFormat::Rgba) {
        if dst_shape[2] != 4 && dst_shape[2] != 3 {
            return Err(Error::Format(format!(
                "Shape Mismatch: Expected {:?} but got {:?}",
                shape, dst_shape
            )));
        }
    } else if dst_shape[2] != shape[2] {
        return Err(Error::Format(format!(
            "Shape Mismatch: Expected {:?} but got {:?}",
            shape, dst_shape
        )));
    }

    let is_rgba = fmt == Some(PixelFormat::Rgba);

    match dst {
        ImageDest3::UInt8(mut dst) => normalize_to_uint8(
            tensor_u8,
            shape,
            &mut dst,
            [dst_shape[0], dst_shape[1], dst_shape[2]],
            normalization,
            zero_point,
            is_rgba,
        ),
        ImageDest3::Int8(mut dst) => normalize_to_int8(
            tensor_u8,
            shape,
            &mut dst,
            [dst_shape[0], dst_shape[1], dst_shape[2]],
            normalization,
            zero_point,
            is_rgba,
        ),
        ImageDest3::Float16(mut dst) => normalize_to_float_16(
            tensor_u8,
            shape,
            &mut dst,
            [dst_shape[0], dst_shape[1], dst_shape[2]],
            normalization,
            zero_point,
            is_rgba,
        ),
        ImageDest3::Float32(mut dst) => normalize_to_float_32(
            tensor_u8,
            shape,
            &mut dst,
            [dst_shape[0], dst_shape[1], dst_shape[2]],
            normalization,
            zero_point,
            is_rgba,
        ),
        ImageDest3::Float64(mut dst) => normalize_to_float_64(
            tensor_u8,
            shape,
            &mut dst,
            [dst_shape[0], dst_shape[1], dst_shape[2]],
            normalization,
            zero_point,
            is_rgba,
        ),
    }
}

/// Copy data from a numpy array into a tensor.
/// Called from PyTensor.copy_from_numpy() in tensor.rs.
pub(crate) fn copy_numpy_to_tensor(tensor_dyn: &TensorDyn, src: PyArrayLike3<u8>) -> Result<()> {
    let src = src.as_array();
    let tensor_u8 = tensor_dyn
        .as_u8()
        .ok_or_else(|| Error::Format("Tensor is not U8".to_string()))?;
    let w = tensor_u8
        .width()
        .ok_or_else(|| Error::Format("not an image".to_string()))?;
    let h = tensor_u8
        .height()
        .ok_or_else(|| Error::Format("not an image".to_string()))?;
    let fmt = tensor_u8
        .format()
        .ok_or_else(|| Error::Format("not an image".to_string()))?;
    let shape = [h, w, fmt.channels()];
    if src.shape() != shape {
        return Err(Error::Format(format!(
            "Shape Mismatch: Expected {:?} but got {:?}",
            shape,
            src.shape()
        )));
    }

    let mut map = tensor_u8.map()?;
    let data = map.as_mut_slice();
    let mut ndarray = ArrayViewMut3::from_shape(shape, data)?;
    ndarray.assign(&src);
    Ok(())
}

#[inline(always)]
fn normalize_to_uint8<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, u8>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    if !matches!(normalization, Normalization::RAW | Normalization::DEFAULT) {
        return Err(Error::InvalidArg(
            "UInt8 destination only supports RAW normalization".to_string(),
        ));
    }
    if zero_point.is_some_and(|zp| zp != 0) {
        return Err(Error::InvalidArg(
            "RAW normalization does not support setting zero point".to_string(),
        ));
    }
    let mut dst = dst.as_array_mut();
    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;

    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let dst = dst.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;
            dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            });

            return Ok(());
        }
    }

    Zip::from(dst)
        .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
        .into_par_iter()
        .for_each(|(x, y)| *x = *y);

    Ok(())
}

#[inline(always)]
fn normalize_to_int8<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, i8>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    if !matches!(
        normalization,
        Normalization::SIGNED | Normalization::DEFAULT
    ) {
        return Err(Error::InvalidArg(
            "Int8 destination only supports SIGNED normalization".to_string(),
        ));
    }

    let zp = if let Some(zp) = zero_point {
        if !(0..=255).contains(&zp) {
            return Err(Error::InvalidArg(format!(
                "zero point out of range expected 0-255, got {zp}"
            )));
        }
        zp as i16
    } else {
        128
    };
    let mut dst = dst.as_array_mut();
    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;
    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let dst = dst.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;
            if zp == 128 {
                dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                    d[0] = (s[0] as i16 - 128) as i8;
                    d[1] = (s[1] as i16 - 128) as i8;
                    d[2] = (s[2] as i16 - 128) as i8;
                });
                return Ok(());
            }
            dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                d[0] = (s[0] as i16 - zp).clamp(-128, 127) as i8;
                d[1] = (s[1] as i16 - zp).clamp(-128, 127) as i8;
                d[2] = (s[2] as i16 - zp).clamp(-128, 127) as i8;
            });
            return Ok(());
        }
    }

    if zp == 128 {
        Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = (*y as i16 - 128) as i8);
        return Ok(());
    }
    Zip::from(dst)
        .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
        .into_par_iter()
        .for_each(|(x, y)| *x = (*y as i16 - zp).clamp(-128, 127) as i8);
    Ok(())
}

// High-performance native f16 implementation for nightly Rust
#[inline(always)]
#[cfg(nightly)]
fn normalize_to_float_16<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, half::f16>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    let dst: ArrayViewMut3<half::f16> = dst.as_array_mut();
    // SAFETY: half::f16 has the same memory layout as native f16
    // This allows us to use the native f16 arithmetic which is much faster
    let mut dst: ArrayViewMut3<f16> = unsafe { std::mem::transmute(dst) };

    let zp = if let Some(zp) = zero_point {
        if !(0..=255).contains(&zp) {
            return Err(Error::InvalidArg(format!(
                "zero point out of range expected 0-255, got {zp}"
            )));
        }
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => zp as f32 / 127.5,
            Normalization::UNSIGNED | Normalization::RAW if zp != 0 => {
                return Err(Error::InvalidArg(
                    "RAW or UNSIGNED normalization does not support setting zero point".to_string(),
                ));
            }
            _ => 0.0,
        }
    } else {
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => 1.0,
            Normalization::UNSIGNED | Normalization::RAW => 0.0,
        }
    };

    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;
    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let dst = dst.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;

            match normalization {
                Normalization::SIGNED | Normalization::DEFAULT => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = (s[0] as f32 / 127.5 - zp) as f16;
                        d[1] = (s[1] as f32 / 127.5 - zp) as f16;
                        d[2] = (s[2] as f32 / 127.5 - zp) as f16;
                    });
                }
                Normalization::UNSIGNED => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = (s[0] as f32 / 255.0) as f16;
                        d[1] = (s[1] as f32 / 255.0) as f16;
                        d[2] = (s[2] as f32 / 255.0) as f16;
                    });
                }
                Normalization::RAW => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = (s[0] as f32) as f16;
                        d[1] = (s[1] as f32) as f16;
                        d[2] = (s[2] as f32) as f16;
                    });
                }
            }

            return Ok(());
        }
    }

    match normalization {
        Normalization::DEFAULT | Normalization::SIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = (*y as f32 / 127.5 - zp) as f16),

        Normalization::UNSIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = (*y as f32 / 255.0) as f16),

        Normalization::RAW => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = (*y) as f16),
    }
    Ok(())
}

// Stable fallback using half crate (slower but works everywhere)
#[inline(always)]
#[cfg(not(nightly))]
fn normalize_to_float_16<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, half::f16>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    use half::slice::HalfFloatSliceExt;
    let mut dst: ArrayViewMut3<half::f16> = dst.as_array_mut();
    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;

    let zp = if let Some(zp) = zero_point {
        if !(0..=255).contains(&zp) {
            return Err(Error::InvalidArg(format!(
                "zero point out of range expected 0-255, got {zp}"
            )));
        }
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => zp as f32 / 127.5,
            Normalization::UNSIGNED | Normalization::RAW if zp != 0 => {
                return Err(Error::InvalidArg(
                    "RAW or UNSIGNED normalization does not support setting zero point".to_string(),
                ));
            }
            _ => 0.0,
        }
    } else {
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => 1.0,
            Normalization::UNSIGNED | Normalization::RAW => 0.0,
        }
    };

    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let mut tmp = vec![0.0; dst.len()];
            let tmp_ = tmp.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;

            match normalization {
                Normalization::SIGNED | Normalization::DEFAULT => {
                    tmp_.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32 / 127.5 - zp;
                        d[1] = s[1] as f32 / 127.5 - zp;
                        d[2] = s[2] as f32 / 127.5 - zp;
                    });
                }
                Normalization::UNSIGNED => {
                    tmp_.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32 / 255.0;
                        d[1] = s[1] as f32 / 255.0;
                        d[2] = s[2] as f32 / 255.0;
                    });
                }
                Normalization::RAW => {
                    tmp_.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32;
                        d[1] = s[1] as f32;
                        d[2] = s[2] as f32;
                    });
                }
            }
            // split into chunks of 256
            let dst = dst.as_chunks_mut::<256>();
            let tmp_ = tmp.as_chunks_mut::<256>();
            dst.0.par_iter_mut().zip(tmp_.0).for_each(|(d, s)| {
                d.convert_from_f32_slice(s);
            });
            dst.1.convert_from_f32_slice(tmp_.1);
            return Ok(());
        }
    }
    match normalization {
        Normalization::SIGNED | Normalization::DEFAULT => {
            Zip::from(dst)
                .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
                .into_par_iter()
                .for_each(|(d, s)| {
                    *d = half::f16::from_f32(*s as f32 / 127.5 - zp);
                });
        }
        Normalization::UNSIGNED => {
            Zip::from(dst)
                .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
                .into_par_iter()
                .for_each(|(d, s)| {
                    *d = half::f16::from_f32(*s as f32 / 255.0);
                });
        }
        Normalization::RAW => {
            Zip::from(dst)
                .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
                .into_par_iter()
                .for_each(|(d, s)| {
                    *d = half::f16::from(*s);
                });
        }
    }

    Ok(())
}

#[inline(always)]
fn normalize_to_float_32<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, f32>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    let mut dst = dst.as_array_mut();
    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;

    let zp = if let Some(zp) = zero_point {
        if !(0..=255).contains(&zp) {
            return Err(Error::InvalidArg(format!(
                "zero point out of range expected 0-255, got {zp}"
            )));
        }
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => zp as f32 / 127.5,
            Normalization::UNSIGNED | Normalization::RAW if zp != 0 => {
                return Err(Error::InvalidArg(
                    "RAW or UNSIGNED normalization does not support setting zero point".to_string(),
                ));
            }
            _ => 0.0,
        }
    } else {
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => 1.0,
            Normalization::UNSIGNED | Normalization::RAW => 0.0,
        }
    };

    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let dst = dst.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;
            match normalization {
                Normalization::SIGNED | Normalization::DEFAULT => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32 / 127.5 - zp;
                        d[1] = s[1] as f32 / 127.5 - zp;
                        d[2] = s[2] as f32 / 127.5 - zp;
                    });
                }
                Normalization::UNSIGNED => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32 / 255.0;
                        d[1] = s[1] as f32 / 255.0;
                        d[2] = s[2] as f32 / 255.0;
                    });
                }
                Normalization::RAW => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f32;
                        d[1] = s[1] as f32;
                        d[2] = s[2] as f32;
                    });
                }
            }
            return Ok(());
        }
    }
    match normalization {
        Normalization::DEFAULT | Normalization::SIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f32 / 127.5 - zp),

        Normalization::UNSIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f32 / 255.0),

        Normalization::RAW => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f32),
    }
    Ok(())
}

#[inline(always)]
fn normalize_to_float_64<'py>(
    tensor: &tensor::Tensor<u8>,
    shape: [usize; 3],
    dst: &mut PyReadwriteArray3<'py, f64>,
    dst_shape: [usize; 3],
    normalization: Normalization,
    zero_point: Option<i64>,
    is_rgba: bool,
) -> Result<()> {
    let mut dst = dst.as_array_mut();
    let map = tensor.map()?;
    let data = map.as_slice();
    let ndarray = ArrayView3::from_shape(shape, data)?;

    let zp = if let Some(zp) = zero_point {
        if !(0..=255).contains(&zp) {
            return Err(Error::InvalidArg(format!(
                "zero point out of range expected 0-255, got {zp}"
            )));
        }
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => zp as f64 / 127.5,
            Normalization::UNSIGNED | Normalization::RAW if zp != 0 => {
                return Err(Error::InvalidArg(
                    "RAW or UNSIGNED normalization does not support setting zero point".to_string(),
                ));
            }
            _ => 0.0,
        }
    } else {
        match normalization {
            Normalization::SIGNED | Normalization::DEFAULT => 1.0,
            Normalization::UNSIGNED | Normalization::RAW => 0.0,
        }
    };

    if is_rgba && dst_shape[2] == 3 {
        if let Some(dst) = dst.as_slice_mut() {
            let dst = dst.as_chunks_mut::<3>().0;
            let src = data.as_chunks::<4>().0;
            match normalization {
                Normalization::SIGNED | Normalization::DEFAULT => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f64 / 127.5 - zp;
                        d[1] = s[1] as f64 / 127.5 - zp;
                        d[2] = s[2] as f64 / 127.5 - zp;
                    });
                }
                Normalization::UNSIGNED => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f64 / 255.0;
                        d[1] = s[1] as f64 / 255.0;
                        d[2] = s[2] as f64 / 255.0;
                    });
                }
                Normalization::RAW => {
                    dst.par_iter_mut().zip(src).for_each(|(d, s)| {
                        d[0] = s[0] as f64;
                        d[1] = s[1] as f64;
                        d[2] = s[2] as f64;
                    });
                }
            }
            return Ok(());
        }
    }
    match normalization {
        Normalization::DEFAULT | Normalization::SIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f64 / 127.5 - zp),

        Normalization::UNSIGNED => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f64 / 255.0),

        Normalization::RAW => Zip::from(dst)
            .and(&ndarray.slice(ndarray::s![.., .., ..dst_shape[2]]))
            .into_par_iter()
            .for_each(|(x, y)| *x = *y as f64),
    }
    Ok(())
}

/// Identifies the type of EGL display used for headless OpenGL ES rendering.
#[pyclass(name = "EglDisplayKind", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyEglDisplayKind {
    Gbm,
    PlatformDevice,
    Default,
}

#[cfg(target_os = "linux")]
impl From<PyEglDisplayKind> for image::EglDisplayKind {
    fn from(val: PyEglDisplayKind) -> Self {
        match val {
            PyEglDisplayKind::Gbm => image::EglDisplayKind::Gbm,
            PyEglDisplayKind::PlatformDevice => image::EglDisplayKind::PlatformDevice,
            PyEglDisplayKind::Default => image::EglDisplayKind::Default,
        }
    }
}

#[cfg(target_os = "linux")]
impl From<image::EglDisplayKind> for PyEglDisplayKind {
    fn from(val: image::EglDisplayKind) -> Self {
        match val {
            image::EglDisplayKind::Gbm => PyEglDisplayKind::Gbm,
            image::EglDisplayKind::PlatformDevice => PyEglDisplayKind::PlatformDevice,
            image::EglDisplayKind::Default => PyEglDisplayKind::Default,
        }
    }
}

/// A validated, available EGL display discovered by probe_egl_displays().
#[cfg(target_os = "linux")]
#[pyclass(name = "EglDisplayInfo")]
pub struct PyEglDisplayInfo {
    info: image::EglDisplayInfo,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl PyEglDisplayInfo {
    #[getter]
    fn kind(&self) -> PyEglDisplayKind {
        self.info.kind.into()
    }

    #[getter]
    fn description(&self) -> &str {
        &self.info.description
    }

    fn __repr__(&self) -> String {
        format!(
            "EglDisplayInfo(kind={}, description='{}')",
            self.info.kind, self.info.description
        )
    }
}

/// Probe for available EGL displays supporting headless OpenGL ES 3.0.
#[cfg(target_os = "linux")]
#[pyfunction]
pub fn probe_egl_displays() -> Result<Vec<PyEglDisplayInfo>> {
    let displays = image::probe_egl_displays()?;
    Ok(displays
        .into_iter()
        .map(|info| PyEglDisplayInfo { info })
        .collect())
}

#[pyclass(name = "ImageProcessor")]
pub struct PyImageProcessor(pub(crate) Mutex<image::ImageProcessor>);

unsafe impl Send for PyImageProcessor {}
unsafe impl Sync for PyImageProcessor {}

#[pymethods]
impl PyImageProcessor {
    #[new]
    #[pyo3(signature = (egl_display=None))]
    pub fn new(egl_display: Option<PyEglDisplayKind>) -> Result<Self> {
        let mut _config = ImageProcessorConfig::default();
        #[cfg(target_os = "linux")]
        {
            _config.egl_display = egl_display.map(|k| k.into());
        }
        #[cfg(not(target_os = "linux"))]
        let _ = egl_display;
        let converter = image::ImageProcessor::with_config(_config)?;
        Ok(PyImageProcessor(Mutex::new(converter)))
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, src_crop = None, dst_crop = None, dst_color = None))]
    pub fn convert(
        &mut self,
        src: &PyTensor,
        dst: &mut PyTensor,
        rotation: PyRotation,
        flip: PyFlip,
        src_crop: Option<PyRect>,
        dst_crop: Option<PyRect>,
        dst_color: Option<[u8; 4]>,
    ) -> Result<()> {
        let rotation = rotation.into();
        let flip = flip.into();
        let crop = Crop {
            src_rect: src_crop.map(|x| x.into()),
            dst_rect: dst_crop.map(|x| x.into()),
            dst_color,
        };
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        l.convert(&src.0, &mut dst.0, rotation, flip, crop)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (dst, bbox, scores, classes, seg=vec![], background=None, opacity=1.0))]
    pub fn draw_decoded_masks(
        &mut self,
        dst: &mut PyTensor,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        seg: Vec<PyReadonlyArray3<u8>>,
        background: Option<&PyTensor>,
        opacity: f32,
    ) -> Result<()> {
        if bbox.shape()[1] != 4 {
            return Err(Error::InvalidArg("bbox shape must be (N, 4)".to_string()));
        }
        if bbox.shape()[0] != scores.shape()[0] || bbox.shape()[0] != classes.shape()[0] {
            return Err(Error::InvalidArg(
                "bbox, scores, classes must have the same length".to_string(),
            ));
        }
        let bbox: ArrayView2<f32> = bbox.as_array();
        let scores: ArrayView1<f32> = scores.as_array();
        let classes: ArrayView1<usize> = classes.as_array();
        let detect = Zip::from(bbox.rows())
            .and(scores)
            .and(classes)
            .into_par_iter()
            .map(|(b, s, c)| DetectBox {
                bbox: BoundingBox::new(b[0], b[1], b[2], b[3]),
                score: *s,
                label: *c,
            })
            .collect::<Vec<_>>();

        let mut is_instance = false;
        for s in &seg {
            if s.shape()[2] == 1 {
                is_instance = true;
                break;
            }
        }

        if is_instance && !seg.is_empty() && seg.len() > detect.len() {
            return Err(Error::InvalidArg(
                "instance segmentation masks length must be less than or equal to detections length"
                    .to_string(),
            ));
        }

        let seg = seg
            .into_iter()
            .enumerate()
            .map(|(ind, s)| {
                let arr: ArrayView3<u8> = s.as_array();
                let (xmin, ymin, xmax, ymax) = if arr.shape()[2] == 1 {
                    (
                        detect[ind].bbox.xmin,
                        detect[ind].bbox.ymin,
                        detect[ind].bbox.xmax,
                        detect[ind].bbox.ymax,
                    )
                } else {
                    (0.0, 0.0, 1.0, 1.0)
                };
                Segmentation {
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    segmentation: arr.to_owned(),
                }
            })
            .collect::<Vec<_>>();
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        if let Some(bg) = &background {
            if std::ptr::eq(&bg.0 as *const _, &dst.0 as *const _) {
                return Err(Error::InvalidArg(
                    "background must not be the same tensor as dst".to_string(),
                ));
            }
        }
        let overlay = image::MaskOverlay {
            background: background.map(|b| &b.0),
            opacity: opacity.clamp(0.0, 1.0),
        };
        l.draw_decoded_masks(&mut dst.0, &detect, &seg, overlay)?;
        Ok(())
    }

    /// Decode model outputs and draw masks directly onto the destination
    /// image in a single call. Masks never leave Rust, eliminating the
    /// Python round-trip overhead of `decode()` + `draw_decoded_masks()`.
    ///
    /// For segmentation models, prototype data is passed directly to the
    /// renderer without materializing intermediate mask arrays in Python.
    /// For detection-only models, this falls back to the standard rendering
    /// path.
    ///
    /// When `tracker` is provided, object tracking is performed and the
    /// return value includes a `tracks` list:
    /// `(boxes, scores, classes, tracks)`.
    ///
    /// Without a tracker the return value is `(boxes, scores, classes)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (decoder, model_output, dst, tracker=None, timestamp=None, background=None, opacity=1.0))]
    pub fn draw_masks<'py>(
        &mut self,
        decoder: &PyDecoder,
        model_output: Vec<PyRef<'py, crate::tensor::PyTensor>>,
        dst: &mut PyTensor,
        tracker: Option<&mut crate::tracker::PyByteTrack>,
        timestamp: Option<u64>,
        background: Option<&PyTensor>,
        opacity: f32,
        py: Python<'py>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_refs: Vec<&edgefirst_hal::tensor::TensorDyn> =
            model_output.iter().map(|t| &t.0).collect();

        if let Some(bg) = &background {
            if std::ptr::eq(&bg.0 as *const _, &dst.0 as *const _) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "background must not be the same tensor as dst",
                ));
            }
        }
        let overlay = image::MaskOverlay {
            background: background.map(|b| &b.0),
            opacity: opacity.clamp(0.0, 1.0),
        };
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;

        if let Some(t) = tracker {
            let ts = timestamp.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            });
            let (output_boxes, output_tracks) = l
                .draw_masks_tracked(&decoder.decoder, t, ts, &tensor_refs, &mut dst.0, overlay)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("draw_masks_tracked: {e:#?}"))
                })?;

            let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
            let tracks: Vec<PyTrackInfo> = output_tracks.into_iter().map(|ti| ti.into()).collect();
            Ok((boxes, scores, classes, tracks).into_pyobject(py)?.into())
        } else {
            let output_boxes = l
                .draw_masks(&decoder.decoder, &tensor_refs, &mut dst.0, overlay)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("draw_masks: {e:#?}"))
                })?;

            let (boxes, scores, classes) = convert_detect_box(py, &output_boxes);
            Ok((boxes, scores, classes).into_pyobject(py)?.into())
        }
    }

    /// Create an image with the processor's optimal memory backend.
    ///
    /// Selects the best available backing storage based on hardware capabilities:
    /// DMA-buf > PBO (GPU buffer) > system memory. Images created this way benefit
    /// from zero-copy GPU paths when used with this processor's convert().
    #[pyo3(signature = (width, height, format = PyPixelFormat::Rgba, dtype = "uint8"))]
    pub fn create_image(
        &self,
        width: usize,
        height: usize,
        format: PyPixelFormat,
        dtype: &str,
    ) -> Result<PyTensor> {
        let fmt: PixelFormat = format.into();
        let dt = crate::tensor::parse_dtype(dtype).map_err(|e| Error::InvalidArg(e.to_string()))?;
        let dyn_tensor = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .create_image(width, height, fmt, dt, None)?;
        Ok(PyTensor(dyn_tensor))
    }

    /// Import an external DMA-BUF image.
    ///
    /// The GPU renders directly into this buffer via EGL DMA-BUF import —
    /// no CPU copy is needed after ``convert()``. The caller retains ownership
    /// of the underlying buffer; the fd is ``dup()``'d immediately.
    ///
    /// The optional ``stride`` and ``offset`` parameters specify the row stride
    /// in bytes and the byte offset within the DMA-BUF where pixel data starts.
    /// Use these when importing buffers with row padding (e.g. V4L2 ``bytesperline``
    /// > width * bytes_per_pixel). When omitted, rows are assumed tightly packed
    /// starting at byte 0.
    ///
    /// For multiplane NV12, pass ``chroma_fd`` for the UV plane, with optional
    /// ``chroma_stride`` and ``chroma_offset`` for the UV plane layout.
    ///
    /// The caller must ensure the DMA-BUF allocation is large enough for the
    /// specified dimensions, format, and any stride/offset values. No buffer-size
    /// validation is performed.
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (fd, width, height, format, dtype = "uint8", stride = None, offset = None, chroma_fd = None, chroma_stride = None, chroma_offset = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn import_image(
        &self,
        fd: std::os::fd::RawFd,
        width: usize,
        height: usize,
        format: PyPixelFormat,
        dtype: &str,
        stride: Option<usize>,
        offset: Option<usize>,
        chroma_fd: Option<std::os::fd::RawFd>,
        chroma_stride: Option<usize>,
        chroma_offset: Option<usize>,
    ) -> Result<PyTensor> {
        use std::os::fd::BorrowedFd;
        use tensor::PlaneDescriptor;

        if fd < 0 {
            return Err(Error::InvalidArg("Invalid file descriptor".to_string()));
        }
        let fmt: PixelFormat = format.into();
        let dt = crate::tensor::parse_dtype(dtype).map_err(|e| Error::InvalidArg(e.to_string()))?;

        // Build image plane descriptor (dups fd eagerly)
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        let mut image_pd = PlaneDescriptor::new(borrowed)?;
        if let Some(s) = stride {
            image_pd = image_pd.with_stride(s);
        }
        if let Some(o) = offset {
            image_pd = image_pd.with_offset(o);
        }

        // Build optional chroma plane descriptor
        let chroma_pd = if let Some(c_fd) = chroma_fd {
            if c_fd < 0 {
                return Err(Error::InvalidArg(
                    "Invalid chroma file descriptor".to_string(),
                ));
            }
            let c_borrowed = unsafe { BorrowedFd::borrow_raw(c_fd) };
            let mut cpd = PlaneDescriptor::new(c_borrowed)?;
            if let Some(s) = chroma_stride {
                cpd = cpd.with_stride(s);
            }
            if let Some(o) = chroma_offset {
                cpd = cpd.with_offset(o);
            }
            Some(cpd)
        } else {
            None
        };

        let proc = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        let dyn_tensor = proc.import_image(image_pd, chroma_pd, width, height, fmt, dt)?;
        Ok(PyTensor(dyn_tensor))
    }

    pub fn set_class_colors(&mut self, colors: Vec<[u8; 4]>) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .set_class_colors(&colors)?;
        Ok(())
    }

    /// Sets the interpolation mode for int8 proto textures.
    ///
    /// Accepts "nearest", "bilinear", or "twopass". Default is "bilinear".
    /// Only affects rendering of quantized (int8) proto segmentation masks.
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (mode))]
    pub fn set_int8_interpolation(&mut self, mode: &str) -> Result<()> {
        let mode = match mode {
            "nearest" => image::Int8InterpolationMode::Nearest,
            "bilinear" => image::Int8InterpolationMode::Bilinear,
            "twopass" => image::Int8InterpolationMode::TwoPass,
            _ => {
                return Err(Error::InvalidArg(format!(
                "Unknown interpolation mode '{mode}'. Expected 'nearest', 'bilinear', or 'twopass'"
            )))
            }
        };
        self.0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .set_int8_interpolation_mode(mode)?;
        Ok(())
    }
}

#[pyclass(name = "Rotation", eq, eq_int)]
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
        match angle.rem_euclid(360) {
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

#[pyclass(name = "Flip", eq, eq_int)]
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

#[pyclass(name = "Rect", eq)]
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
