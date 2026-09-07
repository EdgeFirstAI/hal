// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::detect_boxes::{convert_seg_mask, numpy_to_detect_boxes};
use crate::tensor::PyTensor;
pub use crate::tensor::{PyPixelFormat, PyRegion};
use edgefirst_image::{
    self as image, Crop, Fit, Flip, ImageProcessorConfig, ImageProcessorTrait, MaskResolution,
    Rotation,
};
use edgefirst_tensor::Segmentation;
use edgefirst_tensor::{self as tensor, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};

use ndarray::{
    parallel::prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    ArrayView3, ArrayViewMut3, Zip,
};
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray3, PyUntypedArrayMethods,
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
    /// A `PyErr` raised directly by the cross-package interop layer
    /// (`crate::interop::TensorArg::extract`). Carried through unchanged
    /// rather than flattened into `InvalidArg`/`RuntimeError` so its
    /// exception type (`TypeError` for a protocol violation, `ValueError`
    /// for a bad `access` argument, ...) survives to the caller.
    Py(PyErr),
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
            Error::Py(e) => write!(f, "{e}"),
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
        match err {
            // Already the right exception type -- pass through unchanged.
            Error::Py(e) => e,
            other => pyo3::exceptions::PyRuntimeError::new_err(format!("{other:?}")),
        }
    }
}

impl From<PyErr> for Error {
    fn from(err: PyErr) -> Self {
        Error::Py(err)
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

#[pyclass(eq, eq_int, from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Normalization {
    DEFAULT,
    SIGNED,
    UNSIGNED,
    RAW,
}

/// Single-package; see `PyEglDisplayKind`'s `__hash__` comment.
#[pymethods]
impl Normalization {
    fn __hash__(&self) -> isize {
        *self as isize
    }
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

/// Build a stride-aware [`ArrayView3`] over a mapped `u8` image tensor.
///
/// `map().as_slice()` exposes the full backing buffer, which for a
/// pitch-aligned (DMA / GPU) image tensor is **row-padded**: each row spans
/// `effective_row_stride()` bytes, wider than the logical `W*C`. A naive
/// `ArrayView3::from_shape(shape, data)` assumes a tight `W*C` row and so
/// reads every row after the first at the wrong offset — a progressive shear
/// that silently corrupts `normalize_to_numpy` output on i.MX (DMA dst) while
/// passing on headless x86 (tight Mem/Shm dst).
///
/// Returns `(view, tight)`; `tight == true` means the buffer has no row
/// padding, so callers may keep their flat (`as_chunks`) fast paths.
fn src_view_strided<'a>(
    tensor: &tensor::Tensor<u8>,
    data: &'a [u8],
    shape: [usize; 3],
) -> Result<(ArrayView3<'a, u8>, bool)> {
    use ndarray::ShapeBuilder;
    // Planar layouts (PlanarRgb / PlanarRgba) are already rejected upstream
    // before reaching this helper; only packed [H, W, C] tensors arrive here.
    debug_assert!(
        !matches!(
            tensor.format().map(|f| f.layout()),
            Some(tensor::PixelLayout::Planar)
        ),
        "src_view_strided: planar tensors must be handled upstream"
    );
    let tight_stride = shape[1] * shape[2];
    let row_stride = tensor.effective_row_stride().unwrap_or(tight_stride);
    if row_stride == tight_stride {
        let view = ArrayView3::from_shape(shape, &data[..tight_stride * shape[0]])?;
        return Ok((view, true));
    }
    // Rows sit at `row_stride` bytes; columns/channels remain tightly packed.
    let view = ArrayView3::from_shape(
        (shape[0], shape[1], shape[2]).strides((row_stride, shape[2], 1)),
        data,
    )?;
    Ok((view, false))
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;

    if is_rgba && dst_shape[2] == 3 && tight {
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;
    if is_rgba && dst_shape[2] == 3 && tight {
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;
    if is_rgba && dst_shape[2] == 3 && tight {
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;

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

    if is_rgba && dst_shape[2] == 3 && tight {
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;

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

    if is_rgba && dst_shape[2] == 3 && tight {
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
    let (ndarray, tight) = src_view_strided(tensor, data, shape)?;

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

    if is_rgba && dst_shape[2] == 3 && tight {
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
#[pyclass(
    name = "EglDisplayKind",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.image"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyEglDisplayKind {
    Gbm,
    PlatformDevice,
    Default,
}

/// Single-package (only `edgefirst.image` registers this type), so
/// `eq_int`'s native-or-bare-int richcmp has no cross-package identity
/// problem. Still worth fixing the unhashability `eq` without `hash`
/// leaves behind: hash the discriminant, matching this enum's own
/// int-comparability.
#[pymethods]
impl PyEglDisplayKind {
    fn __hash__(&self) -> isize {
        *self as isize
    }
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
#[pyclass(name = "EglDisplayInfo", module = "edgefirst.image")]
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

/// Round `width` (in pixels) up so that the resulting row stride
/// (`width * bpp` bytes) satisfies the GPU's DMA-BUF EGLImage import
/// alignment requirement.
///
/// Use this when allocating a DMA-BUF that will later be imported as an
/// EGLImage by HAL's GL backend (or by any GLES driver that requires
/// 64-byte aligned pitches — currently Mali Valhall on i.MX 95).
///
/// Pre-aligned widths (640, 1280, 1920, 3008, 3840, ...) round-trip
/// unchanged. Misaligned widths are bumped up to the next valid value.
///
/// :param width: Image width in pixels
/// :param bpp:   Bytes per pixel for the primary plane (4 for RGBA8/BGRA8,
///               3 for RGB888, 1 for Grey/NV12-luma)
/// :return:      Aligned width in pixels (always >= ``width``). Returns
///               ``width`` unchanged if ``bpp == 0``, ``width == 0``, or
///               if the rounded value would overflow.
///
/// :Example:
///
/// .. code-block:: python
///
///     from edgefirst_hal import align_width_for_gpu_pitch
///     # crowd.png canvas: 3004 × 4 = 12016 bytes pitch, NOT 64-aligned.
///     # Round up to 3008 → 12032 bytes pitch (64-aligned).
///     aligned = align_width_for_gpu_pitch(3004, 4)
///     assert aligned == 3008
#[pyfunction]
pub fn align_width_for_gpu_pitch(width: usize, bpp: usize) -> usize {
    image::align_width_for_gpu_pitch(width, bpp)
}

/// Convenience wrapper that derives bytes-per-pixel from a pixel format and
/// dtype, then calls :func:`align_width_for_gpu_pitch`.
///
/// Use this when you have a :class:`PixelFormat` already.
///
/// :param width:  Image width in pixels
/// :param format: Pixel format (e.g. ``PixelFormat.Rgba``)
/// :param dtype:  Element data type as a string. Same set of names accepted
///                by :meth:`ImageProcessor.create_image` and the rest of
///                the HAL Python API — ``"uint8"``, ``"int8"``, ``"uint16"``,
///                ``"int16"``, ``"float16"``, ``"float32"``, etc.
/// :return:       Aligned width in pixels (always >= ``width``)
#[pyfunction]
#[pyo3(signature = (width, format, dtype = "uint8"))]
pub fn align_width_for_pixel_format(
    width: usize,
    format: PyPixelFormat,
    dtype: &str,
) -> Result<usize> {
    let pf: PixelFormat = format.into();
    let dt = crate::tensor::parse_dtype(dtype).map_err(|e| Error::InvalidArg(e.to_string()))?;
    let elem = dt.size();
    Ok(match image::primary_plane_bpp(pf, elem) {
        Some(bpp) => image::align_width_for_gpu_pitch(width, bpp),
        None => width,
    })
}

/// Required DMA-BUF row pitch alignment in bytes for GL backend imports
/// (currently 64). External callers that need to allocate their own
/// DMA-BUFs should size them so each row pitch is a multiple of this value.
///
/// :return: Required pitch alignment in bytes (currently 64)
#[pyfunction]
pub fn gpu_dma_buf_pitch_alignment_bytes() -> usize {
    image::GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES
}

#[pyclass(name = "ImageProcessor", module = "edgefirst.image")]
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
            _config.egl_display = egl_display.map(Into::into);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = egl_display;
        let converter = image::ImageProcessor::with_config(_config)?;
        Ok(PyImageProcessor(Mutex::new(converter)))
    }

    /// Convert ``src`` into ``dst``, scaling, converting colour, rotating
    /// and flipping as needed, and wait for the GPU before returning.
    ///
    /// Args:
    ///     src: Source image, from this or any other ``edgefirst.*``
    ///         package (see ``crates/python-common/INTEROP.md``).
    ///     dst: Destination image, written in place.
    ///     rotation: Applied to the converted image.
    ///     flip: Applied to the converted image.
    ///     source: Sub-region of ``src`` to read, in source pixels.
    ///         ``None`` reads the whole image.
    ///     letterbox: ``(r, g, b, a)`` pad colour. Given, the source is
    ///         fitted into the destination preserving aspect ratio and the
    ///         remainder is padded; ``None`` stretches.
    ///
    /// On Windows the destination's ``gpu_completion()`` reflects this
    /// convert afterwards.
    // `&self`, not `&mut self`: `PyImageProcessor` is a bare `Mutex` whose
    // whole point is to let multiple callers coordinate through the lock
    // rather than through Rust-level exclusive access. `&mut self` would
    // still compile (the pyo3 macro happily borrow-checks it), but it adds
    // a SECOND, outer exclusivity gate on top of the mutex: pyo3 takes a
    // runtime-checked `&mut` borrow of the whole `PyImageProcessor` object
    // for the entire method call, including the `py.detach` region below.
    // Two Python threads calling `convert()` on the *same* `ImageProcessor`
    // -- exactly the pattern this GIL release exists to make concurrent --
    // would then have their second call fail outright with pyo3's "Already
    // borrowed", never reaching the mutex at all. `&self` leaves the mutex
    // as the only serialization point, so a second concurrent call blocks
    // (or proceeds, once GL/CPU backends allow it) instead of erroring.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, source = None, letterbox = None))]
    pub fn convert<'py>(
        &self,
        py: Python<'py>,
        src: &Bound<'py, PyAny>,
        dst: &Bound<'py, PyAny>,
        rotation: PyRotation,
        flip: PyFlip,
        source: Option<PyRegion>,
        letterbox: Option<[u8; 4]>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("python.convert").entered();
        // `access=None`: this runs on the GPU/DMA path and must not force a
        // host pin -- the descriptor's native handle is all a zero-copy
        // consumer needs.
        let src = crate::interop::TensorArg::extract(src, None)?;
        let dst_arg = crate::interop::TensorArg::extract_mut(dst, None)?;
        let rotation = rotation.into();
        let flip = flip.into();
        // Destination placement is the destination tensor (use `tensor.view`/
        // `tensor.batch` for a sub-region); `letterbox` is the pad colour for an
        // aspect-preserving fit.
        let crop = Crop {
            source: source.map(|x| x.into()),
            fit: match letterbox {
                Some(pad) => Fit::Letterbox { pad },
                None => Fit::Stretch,
            },
        };
        if src.can_detach() && dst_arg.can_detach() {
            // Every Python guard is released by `into_raw_access` before
            // this point -- `src` and the reconstructed destination are now
            // plain `Send` data, so the actual convert (GPU or CPU, and
            // rayon-parallel on the CPU path) can run with the GIL released.
            let src = src.into_raw_access()?;
            let mut rendered = dst_arg.into_raw_access()?;
            // Borrowed into the detached region rather than moved, so the
            // completion the backend recorded on it can be published back
            // onto the caller's object once the GIL is held again.
            let target = &mut rendered;
            py.detach(move || -> Result<()> {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                l.convert(src.as_ref(), target.as_mut(), rotation, flip, crop)?;
                Ok(())
            })?;
            // Read before the reconstructed tensor is dropped, published
            // after: `publish_gpu_write` calls into Python, which must not
            // happen while a borrow of the destination is alive.
            let recorded = crate::interop::recorded_gpu_write(rendered.as_ref());
            drop(rendered);
            crate::interop::publish_gpu_write(dst, recorded);
        } else {
            // A GL-PBO-backed argument: `TensorArg::can_detach` -- see its
            // docs -- cannot reconstruct an independent `TensorDyn` for it,
            // so this call keeps the GIL held for its whole duration,
            // exactly this crate's behaviour before GIL release existed
            // for tensors at all.
            let mut dst_arg = dst_arg;
            {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                l.convert(src.as_ref(), dst_arg.as_mut(), rotation, flip, crop)?;
            }
            // `dst_arg` holds a `PyRefMut` of a native destination, and
            // `set_gpu_write` takes `&self`: the borrow has to end before
            // the publish or pyo3 refuses the call as "Already borrowed".
            let recorded = crate::interop::recorded_gpu_write(dst_arg.as_ref());
            drop(dst_arg);
            crate::interop::publish_gpu_write(dst, recorded);
        }
        Ok(())
    }

    /// Convert without blocking on the GPU; returns a completion handle.
    ///
    /// Same ``src``/``dst``/``rotation``/``flip``/``source``/``letterbox``
    /// arguments as an ordinary convert; this call differs only in what it
    /// returns. The handle is the GL to NPU handoff primitive: hand it to a
    /// consumer, or wait on it, instead of paying a blocking GPU sync.
    ///
    /// Returns:
    ///     A sync-file descriptor on Linux and Android, an event handle
    ///     (as an integer) on Windows, or ``None`` when the convert
    ///     completed synchronously (no native fence on this display). The
    ///     caller owns the handle -- close it (``os.close`` for the fd,
    ///     ``ctypes.windll.kernel32.CloseHandle`` for the Windows handle).
    ///
    /// On Windows the destination's ``gpu_completion()`` reflects this
    /// convert afterwards, so a device consumer can be handed the fence
    /// value instead of this event.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, source = None, letterbox = None))]
    pub fn convert_with_fence<'py>(
        &self,
        py: Python<'py>,
        src: &Bound<'py, PyAny>,
        dst: &Bound<'py, PyAny>,
        rotation: PyRotation,
        flip: PyFlip,
        source: Option<PyRegion>,
        letterbox: Option<[u8; 4]>,
    ) -> Result<Option<usize>> {
        let _span = tracing::trace_span!("python.convert_with_fence").entered();
        // See `convert`: GPU/DMA path, no host pin forced.
        let src = crate::interop::TensorArg::extract(src, None)?;
        let dst_arg = crate::interop::TensorArg::extract_mut(dst, None)?;
        let rotation = rotation.into();
        let flip = flip.into();
        let crop = Crop {
            source: source.map(|x| x.into()),
            fit: match letterbox {
                Some(pad) => Fit::Letterbox { pad },
                None => Fit::Stretch,
            },
        };
        // Assigned in both arms below: the publish has to run inside each
        // arm, where the tensor the convert wrote is still in scope.
        let fence;
        if src.can_detach() && dst_arg.can_detach() {
            // See `convert`: every Python guard is released by this point,
            // so the actual convert can run with the GIL released.
            let src = src.into_raw_access()?;
            let mut rendered = dst_arg.into_raw_access()?;
            // See `convert`: borrowed rather than moved, so the recorded
            // completion can be published afterwards.
            let target = &mut rendered;
            fence = py.detach(move || -> Result<Option<image::CompletionFence>> {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                Ok(l.convert_with_fence(src.as_ref(), target.as_mut(), rotation, flip, crop)?)
            })?;
            // Read before the reconstructed tensor is dropped, published
            // after: `publish_gpu_write` calls into Python, which must not
            // happen while a borrow of the destination is alive.
            let recorded = crate::interop::recorded_gpu_write(rendered.as_ref());
            drop(rendered);
            crate::interop::publish_gpu_write(dst, recorded);
        } else {
            // See `convert`: a GL-PBO-backed argument keeps the GIL held.
            let mut dst_arg = dst_arg;
            {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                fence =
                    l.convert_with_fence(src.as_ref(), dst_arg.as_mut(), rotation, flip, crop)?;
            }
            // `dst_arg` holds a `PyRefMut` of a native destination, and
            // `set_gpu_write` takes `&self`: the borrow has to end before
            // the publish or pyo3 refuses the call as "Already borrowed".
            let recorded = crate::interop::recorded_gpu_write(dst_arg.as_ref());
            drop(dst_arg);
            crate::interop::publish_gpu_write(dst, recorded);
        }
        // The caller now owns the handle: it crosses into a bare Python int,
        // so ownership can no longer be tracked by a Rust drop.
        Ok(fence.map(|f| {
            #[cfg(unix)]
            {
                use std::os::fd::IntoRawFd;
                f.into_raw_fd() as usize
            }
            #[cfg(windows)]
            {
                use std::os::windows::io::IntoRawHandle;
                f.into_raw_handle() as usize
            }
        }))
    }

    /// Convert without waiting for the GPU — the batch-preprocessing primitive.
    ///
    /// Same arguments as :meth:`convert`, but the OpenGL backend skips the
    /// per-call ``glFinish()``. Render N model inputs by looping this over
    /// ``dst.batch(n)`` / ``dst.view(region)`` row-bands of one batched
    /// destination, then call :meth:`flush` once: the backend imports the
    /// destination a single time and renders each tile as a viewport band,
    /// syncing once at flush. A deferred destination is not safe to read
    /// (or ``cuda_map``) until :meth:`flush` returns. Non-GL backends
    /// complete synchronously and :meth:`flush` is a no-op.
    ///
    /// On Windows the destination's ``gpu_completion()`` reflects this
    /// convert afterwards — the value covers the queued render, so a device
    /// consumer waits on that fence rather than on :meth:`flush`.
    // `&self`: see `convert`'s comment on the same choice.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, source = None, letterbox = None))]
    pub fn convert_deferred<'py>(
        &self,
        py: Python<'py>,
        src: &Bound<'py, PyAny>,
        dst: &Bound<'py, PyAny>,
        rotation: PyRotation,
        flip: PyFlip,
        source: Option<PyRegion>,
        letterbox: Option<[u8; 4]>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("python.convert_deferred").entered();
        // See `convert`: GPU/DMA path, no host pin forced.
        let src = crate::interop::TensorArg::extract(src, None)?;
        let dst_arg = crate::interop::TensorArg::extract_mut(dst, None)?;
        let rotation = rotation.into();
        let flip = flip.into();
        let crop = Crop {
            source: source.map(|x| x.into()),
            fit: match letterbox {
                Some(pad) => Fit::Letterbox { pad },
                None => Fit::Stretch,
            },
        };
        if src.can_detach() && dst_arg.can_detach() {
            // See `convert`: every Python guard is gone by this point.
            let src = src.into_raw_access()?;
            let mut rendered = dst_arg.into_raw_access()?;
            // See `convert`: borrowed rather than moved, so the recorded
            // completion can be published afterwards. The value covers the
            // queued render even though nothing has waited on the GPU yet,
            // which is what a device consumer waits on instead of `flush`.
            let target = &mut rendered;
            py.detach(move || -> Result<()> {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                l.convert_deferred(src.as_ref(), target.as_mut(), rotation, flip, crop)?;
                Ok(())
            })?;
            // Read before the reconstructed tensor is dropped, published
            // after: `publish_gpu_write` calls into Python, which must not
            // happen while a borrow of the destination is alive.
            let recorded = crate::interop::recorded_gpu_write(rendered.as_ref());
            drop(rendered);
            crate::interop::publish_gpu_write(dst, recorded);
        } else {
            // See `convert`: a GL-PBO-backed argument keeps the GIL held.
            let mut dst_arg = dst_arg;
            {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                l.convert_deferred(src.as_ref(), dst_arg.as_mut(), rotation, flip, crop)?;
            }
            // `dst_arg` holds a `PyRefMut` of a native destination, and
            // `set_gpu_write` takes `&self`: the borrow has to end before
            // the publish or pyo3 refuses the call as "Already borrowed".
            let recorded = crate::interop::recorded_gpu_write(dst_arg.as_ref());
            drop(dst_arg);
            crate::interop::publish_gpu_write(dst, recorded);
        }
        Ok(())
    }

    /// Complete all deferred converts since the last flush with a single GPU
    /// sync. After this returns, every destination written by
    /// [`convert_deferred`](Self::convert_deferred) is finished and safe to read
    /// back or `cuda_map`. Non-GL backends return immediately.
    // `&self`: see `convert`'s comment. Also load-bearing for `flush` itself
    // -- it must be callable while another thread's `convert_deferred` is
    // still inside its own `py.detach` region on the same `ImageProcessor`,
    // without `&mut self`'s outer borrow-check turning that into an error.
    pub fn flush(&self) -> Result<()> {
        let _span = tracing::trace_span!("python.flush").entered();
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        l.flush()?;
        Ok(())
    }

    /// Draw detection boxes and optional segmentation masks onto ``dst``.
    ///
    /// ``dst`` is always fully overwritten: ``background`` plus the masks
    /// when a background is given, otherwise cleared to transparent and
    /// drawn on. See the stub for the full argument list.
    ///
    /// On Windows the destination's ``gpu_completion()`` reflects this draw
    /// afterwards, as it does after ``convert``.
    ///
    /// A ``BGRA`` ``background=`` onto a GPU-backed destination raises: the
    /// OpenGL base-layer draw has no ``BGRA`` arm and the CPU backend renders
    /// only ``RGBA``/``RGB``. It previously returned without writing ``dst``.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (dst, bbox, scores, classes, seg=vec![], background=None, opacity=1.0, letterbox=None, color_mode=PyColorMode::Class))]
    pub fn draw_decoded_masks<'py>(
        &mut self,
        dst: &Bound<'py, PyAny>,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        seg: Vec<PyReadonlyArray3<u8>>,
        background: Option<&Bound<'py, PyAny>>,
        opacity: f32,
        letterbox: Option<[f32; 4]>,
        color_mode: PyColorMode,
    ) -> Result<()> {
        // GPU render target, same as `convert`'s dst: no host pin forced.
        // The caller's object is kept under its own name: the extraction
        // shadows `dst`, and the completion publish below needs the object,
        // not the resolved tensor.
        let dst_obj = dst;
        let mut dst = crate::interop::TensorArg::extract_mut(dst_obj, None)?;
        // Read-only compositing input.
        let background = background
            .map(|b| crate::interop::TensorArg::extract(b, None))
            .transpose()?;
        let detect = numpy_to_detect_boxes(&bbox, &scores, &classes)?;

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
                    segmentation: edgefirst_tensor::Tensor::from_arrayview3(arr)
                        .expect("mask -> TensorDyn")
                        .into(),
                }
            })
            .collect::<Vec<_>>();
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        // Buffer-aliasing is validated inside draw_decoded_masks via
        // TensorDyn::aliases (catches same-buffer across separate PyTensor
        // wrappers — pointer-identity on the wrapper would miss that).
        let overlay = image::MaskOverlay {
            background: background.as_ref().map(|b| b.as_ref()),
            opacity: opacity.clamp(0.0, 1.0),
            letterbox,
            color_mode: color_mode.into(),
        };
        l.draw_decoded_masks(dst.as_mut(), &detect, &seg, overlay)?;
        drop(l);
        // Same as the convert variants: a draw writes the tensor resolved
        // from the destination's descriptor, not the caller's, so the
        // completion the GL backend recorded on it (a D3D11 texture on
        // Windows) has to be published back. Read before the borrow ends,
        // published after, as in `convert`.
        let recorded = crate::interop::recorded_gpu_write(dst.as_ref());
        drop(dst);
        crate::interop::publish_gpu_write(dst_obj, recorded);
        Ok(())
    }

    /// Materialize per-instance segmentation masks from prototype data.
    ///
    /// Computes ``mask_coeff @ protos`` for each detection, producing compact
    /// binary masks at prototype resolution (e.g., 160×160 crops). Mask values
    /// are **binary** ``uint8 {0, 255}`` — pixels where the dot product is
    /// positive are foreground (255), otherwise background (0).
    ///
    /// The returned masks can be:
    ///
    /// - Inspected or exported for analytics, IoU computation, etc.
    /// - Passed directly to :meth:`draw_decoded_masks` for GPU-interpolated
    ///   rendering.
    ///
    /// .. note::
    ///
    ///     Calling ``materialize_masks`` + ``draw_decoded_masks`` separately
    ///     prevents the HAL from using its internal fused optimization. For
    ///     render-only use cases, prefer :meth:`Decoder.draw_onto` which is 1.6–27×
    ///     faster on tested platforms.
    ///
    /// :param bbox: detection boxes as (N, 4) float32 array (normalized xyxy)
    /// :param scores: detection scores as (N,) float32 array
    /// :param classes: class indices as (N,) array
    /// :param proto_data: prototype data from :meth:`Decoder.decode_proto`.
    ///     May come from another ``edgefirst.*`` package (e.g.
    ///     ``edgefirst.decoder``) -- see the ``__edgefirst_protodata__``
    ///     capsule protocol.
    /// :param letterbox: optional letterbox region ``(x0, y0, x1, y1)`` in
    ///     normalized coordinates, or ``None`` if no letterboxing was applied
    /// :param resolution: optional mask materialization mode. When ``None`` or
    ///     :class:`MaskResolution.Proto`, returns per-detection tiles at
    ///     proto-plane resolution with binary mask values ``{0, 255}``. When
    ///     :class:`MaskResolution.Scaled` ``(width, height)``, HAL upsamples
    ///     the full proto plane once and returns per-detection tiles at the
    ///     target resolution with binary mask values ``{0, 255}``.
    ///     Both modes use ``> 127`` as the threshold convention.
    /// :returns: list of ``(H, W, 1)`` uint8 numpy arrays with binary
    ///     ``{0, 255}`` mask values.
    /// :rtype: list[numpy.ndarray]
    #[pyo3(signature = (bbox, scores, classes, proto_data, letterbox=None, resolution=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn materialize_masks<'py>(
        &self,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        proto_data: &Bound<'py, PyAny>,
        letterbox: Option<[f32; 4]>,
        resolution: Option<PyMaskResolution>,
        py: Python<'py>,
    ) -> Result<Vec<Bound<'py, numpy::PyArray3<u8>>>> {
        let _span = tracing::trace_span!("python.materialize_masks").entered();
        // `numpy_to_detect_boxes` already copies out of the numpy arrays
        // into an owned `Vec<DetectBox>` (it has to, to build `DetectBox`
        // values at all) -- `detect` needs no further resolution before a
        // detached region.
        let detect = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
        let proto_data = crate::interop::ProtoDataArg::extract(proto_data)?;
        let resolution = resolution.map(|r| r.0).unwrap_or(MaskResolution::Proto);
        let masks = if proto_data.can_detach() {
            // Reconstructs `proto_data`'s two tensors independently -- see
            // `RawProtoDataAccess`'s doc comment for why that (not holding
            // `proto_data`'s guard alive the way `Decoder`'s methods do) is
            // the right model for a value that is just tensors and an enum.
            // Every Python guard is gone by this point, so the actual mask
            // upsample -- over full-resolution buffers, per this method's
            // own doc comment -- can run with the GIL released.
            let proto_data = proto_data.into_raw_access()?;
            py.detach(move || -> Result<Vec<Segmentation>> {
                let mut l = self
                    .0
                    .lock()
                    .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
                Ok(l.materialize_masks(&detect, proto_data.as_ref(), letterbox, resolution)?)
            })?
        } else {
            // A GL-PBO-backed proto tensor: `ProtoDataArg::can_detach` (see
            // its docs) cannot reconstruct an independent value for it, so
            // this call keeps the GIL held for its whole duration, exactly
            // this crate's behaviour before GIL release existed for tensors
            // at all.
            let mut l = self
                .0
                .lock()
                .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
            l.materialize_masks(&detect, proto_data.as_ref(), letterbox, resolution)?
        };
        Ok(convert_seg_mask(py, &masks))
    }

    /// Draw prototype masks onto ``dst`` without materialising per-instance
    /// mask arrays in Python. ``proto_data`` may come from another
    /// ``edgefirst.*`` package via the ``__edgefirst_protodata__`` capsule.
    ///
    /// On Windows the destination's ``gpu_completion()`` reflects this draw
    /// afterwards, as it does after ``convert``. The same ``BGRA``
    /// ``background=`` restriction as :meth:`draw_decoded_masks` applies.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (dst, bbox, scores, classes, proto_data, background=None, opacity=1.0, letterbox=None, color_mode=PyColorMode::Class))]
    pub fn draw_proto_masks<'py>(
        &self,
        dst: &Bound<'py, PyAny>,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        proto_data: &Bound<'py, PyAny>,
        background: Option<&Bound<'py, PyAny>>,
        opacity: f32,
        letterbox: Option<[f32; 4]>,
        color_mode: PyColorMode,
    ) -> Result<()> {
        // The caller's object is kept under its own name: the extraction
        // shadows `dst`, and the completion publish below needs the object,
        // not the resolved tensor.
        let dst_obj = dst;
        let mut dst = crate::interop::TensorArg::extract_mut(dst_obj, None)?;
        let background = background
            .map(|b| crate::interop::TensorArg::extract(b, None))
            .transpose()?;
        let detect = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
        let proto_data = crate::interop::ProtoDataArg::extract(proto_data)?;
        let overlay = image::MaskOverlay {
            background: background.as_ref().map(|b| b.as_ref()),
            opacity: opacity.clamp(0.0, 1.0),
            letterbox,
            color_mode: color_mode.into(),
        };
        let mut l = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        l.draw_proto_masks(dst.as_mut(), &detect, proto_data.as_ref(), overlay)?;
        drop(l);
        // As `draw_decoded_masks`: a draw writes the tensor resolved from the
        // destination's descriptor, not the caller's, so the completion the
        // GL backend recorded on it (a D3D11 texture on Windows) has to be
        // published back. Read before the borrow ends, published after, as in
        // `convert`.
        let recorded = crate::interop::recorded_gpu_write(dst.as_ref());
        drop(dst);
        crate::interop::publish_gpu_write(dst_obj, recorded);
        Ok(())
    }

    /// Create an image with the processor's optimal memory backend.
    ///
    /// Selects the best available backing storage based on hardware capabilities:
    /// DMA-buf > PBO (GPU buffer) > system memory. Images created this way benefit
    /// from zero-copy GPU paths when used with this processor's convert().
    ///
    /// ``access`` declares CPU involvement (``"none"`` — the default,
    /// hardware-only — ``"read"``, ``"write"``, or ``"readwrite"``).
    /// Scripts that touch the pixels from Python (``map()``, numpy
    /// interop) must pass ``access="readwrite"`` (or ``"read"``/
    /// ``"write"``): the strict ``"none"`` default keeps hardware
    /// pipelines eligible for tile compression and skips CPU cache
    /// maintenance, but mapping such a tensor is best-effort and counted
    /// as unplanned.
    ///
    /// ``compression`` requests a vendor tile-compressed layout:
    /// ``None`` (default, linear), ``"any"`` (native scheme when
    /// eligible, counted linear fallback), or a specific scheme
    /// (``"ubwc"``/``"afbc"``/``"pvric"``/``"dcc"`` — allocation fails
    /// unless the device's native scheme matches). Requires
    /// ``access="none"``. Read the outcome via ``Tensor.compression``.
    #[pyo3(signature = (width, height, format = PyPixelFormat::Rgba, dtype = "uint8", access = "none", compression = None))]
    pub fn create_image(
        &self,
        width: usize,
        height: usize,
        format: PyPixelFormat,
        dtype: &str,
        access: &str,
        compression: Option<&str>,
    ) -> Result<PyTensor> {
        let fmt: PixelFormat = format.into();
        let dt = crate::tensor::parse_dtype(dtype).map_err(|e| Error::InvalidArg(e.to_string()))?;
        let acc = crate::tensor::parse_cpu_access(access)
            .map_err(|e| Error::InvalidArg(e.to_string()))?;
        let proc = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?;
        let dyn_tensor = match crate::tensor::parse_compression(compression)
            .map_err(|e| Error::InvalidArg(e.to_string()))?
        {
            Some(request) => {
                let desc = tensor::ImageDesc::new(width, height, fmt, dt)
                    .with_access(acc)
                    .with_compression(request);
                proc.create_image_desc(&desc)?
            }
            None => proc.create_image(width, height, fmt, dt, None, acc)?,
        };
        Ok(PyTensor(dyn_tensor))
    }

    /// Allocate the tall packed batch destination ``[tile_w, n*tile_h]`` — a
    /// single GL-importable parent that stacks ``n`` tiles vertically. Uses
    /// the same DMA-pitch alignment and ``Dma>Pbo>Mem`` selection as
    /// :meth:`create_image`. Caller-owned for pool reuse.
    #[pyo3(signature = (n, cfg, format = PyPixelFormat::Rgba, dtype = "uint8", memory = None, access = "none"))]
    pub fn alloc_tile_batch(
        &self,
        n: usize,
        cfg: &crate::tiling::PyTilingConfig,
        format: PyPixelFormat,
        dtype: &str,
        memory: Option<crate::tensor::PyTensorMemory>,
        access: &str,
    ) -> Result<PyTensor> {
        let fmt: PixelFormat = format.into();
        let dt = crate::tensor::parse_dtype(dtype).map_err(|e| Error::InvalidArg(e.to_string()))?;
        let acc = crate::tensor::parse_cpu_access(access)
            .map_err(|e| Error::InvalidArg(e.to_string()))?;
        let dyn_tensor = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .alloc_tile_batch(n, &cfg.0, fmt, dt, memory.map(Into::into), acc)?;
        Ok(PyTensor(dyn_tensor))
    }

    /// Compute the tile grid and per-tile :class:`TilePlacement` metadata for a
    /// ``width``×``height`` frame **without touching the GPU**. Call once per
    /// frame to size pools and drive a tile stream.
    #[pyo3(signature = (width, height, cfg))]
    pub fn plan_tiles(
        &self,
        width: usize,
        height: usize,
        cfg: &crate::tiling::PyTilingConfig,
    ) -> Result<Vec<crate::tiling::PyTilePlacement>> {
        let placements = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .plan_tiles(width, height, &cfg.0)?;
        Ok(placements
            .into_iter()
            .map(crate::tiling::PyTilePlacement)
            .collect())
    }

    /// Render every tile of ``src`` into ``dst_batched`` (a tall packed parent
    /// from :meth:`alloc_tile_batch`), one deferred convert per tile, then a
    /// single ``flush``. Returns the per-tile :class:`TilePlacement` list in
    /// tile-index order.
    #[pyo3(signature = (src, dst_batched, cfg))]
    pub fn tile_into<'py>(
        &mut self,
        src: &Bound<'py, PyAny>,
        dst_batched: &Bound<'py, PyAny>,
        cfg: &crate::tiling::PyTilingConfig,
    ) -> Result<Vec<crate::tiling::PyTilePlacement>> {
        let _span = tracing::trace_span!("python.tile_into").entered();
        // GPU/DMA path, same as `convert`: no host pin forced.
        let src = crate::interop::TensorArg::extract(src, None)?;
        let mut dst_batched = crate::interop::TensorArg::extract_mut(dst_batched, None)?;
        let placements = self
            .0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .tile_into(src.as_ref(), dst_batched.as_mut(), &cfg.0)?;
        Ok(placements
            .into_iter()
            .map(crate::tiling::PyTilePlacement)
            .collect())
    }

    /// Render exactly one tile of ``src`` into ``dst_slot`` (a single
    /// model-input sized destination). Deferred — call :meth:`flush` on your
    /// own cadence so tiles overlap with inference. All geometry rides in
    /// ``placement`` (from :meth:`plan_tiles`).
    #[pyo3(signature = (src, dst_slot, placement, cfg))]
    pub fn tile_one<'py>(
        &mut self,
        src: &Bound<'py, PyAny>,
        dst_slot: &Bound<'py, PyAny>,
        placement: &crate::tiling::PyTilePlacement,
        cfg: &crate::tiling::PyTilingConfig,
    ) -> Result<()> {
        let _span = tracing::trace_span!("python.tile_one").entered();
        let src = crate::interop::TensorArg::extract(src, None)?;
        let mut dst_slot = crate::interop::TensorArg::extract_mut(dst_slot, None)?;
        self.0
            .lock()
            .map_err(|_| Error::InvalidArg("ImageProcessor lock poisoned".to_string()))?
            .tile_one(src.as_ref(), dst_slot.as_mut(), &placement.0, &cfg.0)?;
        Ok(())
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
    #[pyo3(signature = (fd, width, height, format, dtype = "uint8", stride = None, offset = None, chroma_fd = None, chroma_stride = None, chroma_offset = None, colorimetry = None))]
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
        colorimetry: Option<crate::colorimetry::PyColorimetry>,
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
        let dyn_tensor = proc.import_image(
            image_pd,
            chroma_pd,
            width,
            height,
            fmt,
            dt,
            colorimetry.map(Into::into),
        )?;
        Ok(PyTensor(dyn_tensor))
    }

    /// Set the colours used to render segmentation masks by class label.
    ///
    /// The palette holds 20 entries and a class index wraps around it. Only
    /// the leading entries this call supplies are replaced; the rest keep the
    /// defaults, and anything past the twentieth is ignored.
    ///
    /// Args:
    ///     colors: ``(r, g, b, a)`` tuples, indexed by class label.
    ///
    /// Raises:
    ///     RuntimeError: If the backend rejects the palette.
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
    // Mirrors the cfg on `ImageProcessor::set_int8_interpolation_mode` in
    // crates/image/src/lib.rs, which is every OS with a GL backend — NOT
    // Linux. This binding was gated to Linux alone, matching the EGL-probe
    // and DMA-BUF-import neighbours above rather than the API it wraps, which
    // silently removed a working method from macOS, iOS and Android callers
    // (macOS reaches the same GL path through ANGLE). `feature = "opengl"` is
    // still a default feature of edgefirst-image itself (its own `default =
    // ["opengl", "static"]` is unchanged) -- but that no longer guarantees it
    // is active HERE. Two different lines, two different jobs: the root
    // Cargo.toml's edgefirst-image entry is what sets `default-features =
    // false` for real (single-tensor-home, Python side, task P2) and stops
    // image's defaults reaching this edge at all -- this crate's OWN
    // `default-features = false` on the same edge (its Cargo.toml) is
    // redundant with root's and does nothing by itself: for an inherited
    // `workspace = true` dependency, a member's `default-features = false`
    // is silently ignored unless the workspace entry sets one too, which is
    // exactly the mechanic that let this default keep reaching every
    // consumer unnoticed before task P2. Because the suppression is
    // workspace-wide, `"opengl"` had to be restored explicitly in this
    // crate's own feature list, and THAT explicit entry -- not either
    // `default-features = false` -- is the sole reason it is active here.
    // Still not re-exported as a feature of this crate itself, so there is
    // still no feature to test at this layer -- but if that explicit
    // `"opengl"` is ever trimmed back out of python-common's Cargo.toml,
    // this layer DOES need a gate, and none exists.
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
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

#[pyclass(
    name = "Rotation",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.image"
)]
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

    /// Single-package; see `PyEglDisplayKind`'s `__hash__` comment.
    fn __hash__(&self) -> isize {
        *self as isize
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

#[pyclass(name = "Flip", eq, eq_int, from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyFlip {
    NoFlip = 0,
    Horizontal = 1,
    Vertical = 2,
}

/// Single-package; see `PyEglDisplayKind`'s `__hash__` comment.
#[pymethods]
impl PyFlip {
    fn __hash__(&self) -> isize {
        *self as isize
    }
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

/// Controls how mask colors are assigned to detections.
///
/// - ``Class`` — color is chosen by class label (default, correct for semantic
///   segmentation where colors carry class meaning)
/// - ``Instance`` — color is chosen by detection index (each detected object
///   gets a unique color regardless of class)
/// - ``Track`` — color is chosen by track ID (use with object tracking)
#[pyclass(
    name = "ColorMode",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.image"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyColorMode {
    Class = 0,
    Instance = 1,
    Track = 2,
}

/// Single-package; see `PyEglDisplayKind`'s `__hash__` comment.
#[pymethods]
impl PyColorMode {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyColorMode> for image::ColorMode {
    fn from(val: PyColorMode) -> Self {
        match val {
            PyColorMode::Class => image::ColorMode::Class,
            PyColorMode::Instance => image::ColorMode::Instance,
            PyColorMode::Track => image::ColorMode::Track,
        }
    }
}

/// Controls the resolution and coordinate frame of masks produced by
/// :meth:`ImageProcessor.materialize_masks`.
///
/// Construct via classmethods:
///
/// - ``MaskResolution.Proto()`` — per-detection tiles at proto-plane
///   resolution (historical default). Mask values are binary ``uint8 {0, 255}``.
/// - ``MaskResolution.Scaled(width, height)`` — per-detection tiles at
///   caller-specified pixel resolution, produced by upsampling the full
///   proto plane once (correct edge-clamp bilinear) and cropping by bbox.
///   Mask values are binary ``uint8 {0, 255}``.
///   Both modes use ``> 127`` as the threshold convention. If a ``letterbox``
///   is also passed to ``materialize_masks``, ``(width, height)`` are
///   interpreted as original-content pixel dims and the inverse letterbox
///   transform is applied during the upsample.
#[pyclass(name = "MaskResolution", from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy)]
pub struct PyMaskResolution(pub(crate) MaskResolution);

#[pymethods]
impl PyMaskResolution {
    /// Per-detection tile at proto-plane resolution (default).
    #[classmethod]
    #[allow(non_snake_case)]
    fn Proto(_cls: &Bound<'_, pyo3::types::PyType>) -> Self {
        Self(MaskResolution::Proto)
    }

    /// Per-detection tile at ``(width, height)`` pixel resolution.
    #[classmethod]
    #[allow(non_snake_case)]
    #[pyo3(signature = (width, height))]
    fn Scaled(_cls: &Bound<'_, pyo3::types::PyType>, width: u32, height: u32) -> Self {
        Self(MaskResolution::Scaled { width, height })
    }

    fn __repr__(&self) -> String {
        match self.0 {
            MaskResolution::Proto => "MaskResolution.Proto()".into(),
            MaskResolution::Scaled { width, height } => {
                format!("MaskResolution.Scaled(width={width}, height={height})")
            }
        }
    }
}
