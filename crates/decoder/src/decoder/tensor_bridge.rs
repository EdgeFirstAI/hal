// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between `TensorDyn` (the HAL's fundamental type-erased tensor) and
//! the decoder's internal `ArrayViewD`/`ArrayViewDQuantized` representations.
//!
//! This module maps `TensorDyn` outputs into memory and converts them into
//! ndarray views that the existing decode methods consume.

use edgefirst_tensor::{HostView, PixelLayout, TensorDyn, TensorMapTrait, TensorTrait};
use half::f16;
use ndarray::{ArrayViewD, IxDyn, ShapeBuilder};

use super::ArrayViewDQuantized;
use crate::DecoderError;

/// Element strides addressing `t` through its padded row pitch, or `None`
/// when the tensor is tightly packed.
///
/// A row spans every dimension after the leading one for a raw tensor,
/// `W * C` for a packed image and `W` for a planar or semi-planar one, the
/// same rows a strided map of the tensor exposes.
///
/// # Errors
///
/// Returns `DecoderError::InvalidShape` if the pitch is shorter than a row,
/// is not a whole number of elements, or the shape has no row dimension.
pub(crate) fn padded_strides(t: &TensorDyn) -> Result<Option<Vec<usize>>, DecoderError> {
    let Some(pitch) = t.row_stride() else {
        return Ok(None);
    };
    let shape = t.shape();
    let ndim = shape.len();
    let row_start = match t.format().map(|f| f.layout()) {
        None => Some(1),
        Some(PixelLayout::Packed) => ndim.checked_sub(2),
        Some(PixelLayout::Planar | PixelLayout::SemiPlanar) => ndim.checked_sub(1),
        Some(_) => None,
    }
    .filter(|&k| k >= 1 && k <= ndim)
    .ok_or_else(|| {
        DecoderError::InvalidShape(format!(
            "row stride {pitch} cannot apply to tensor shape {shape:?}"
        ))
    })?;
    let elem = t.dtype().size();
    let row_elems: usize = shape[row_start..].iter().product();
    let row_bytes = row_elems.saturating_mul(elem);
    if pitch == row_bytes {
        return Ok(None);
    }
    if elem == 0 || pitch < row_bytes || pitch % elem != 0 {
        return Err(DecoderError::InvalidShape(format!(
            "row stride {pitch} is not a valid pitch for rows of {row_elems} \
             {elem}-byte elements (shape {shape:?})"
        )));
    }
    let mut strides = vec![0usize; ndim];
    let mut acc = 1usize;
    for i in (row_start..ndim).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    acc = pitch / elem;
    for i in (0..row_start).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    Ok(Some(strides))
}

/// An `ArrayViewD` of `shape` over `slice`, addressed through `strides` when
/// the rows are padded and as a standard-layout view otherwise.
pub(crate) fn strided_view<'a, T>(
    shape: &[usize],
    strides: Option<&[usize]>,
    slice: &'a [T],
) -> Result<ArrayViewD<'a, T>, DecoderError> {
    match strides {
        None => ArrayViewD::from_shape(shape, slice),
        Some(s) => ArrayViewD::from_shape(IxDyn(shape).strides(IxDyn(s)), slice),
    }
    .map_err(|e| DecoderError::InvalidConfig(format!("tensor shape: {e}")))
}

/// A read-only map of one output with the strides that address its rows.
///
/// The `HostView` borrows from the original `TensorDyn`; views created from
/// it must not outlive the `MappedOutputs` that owns it.
pub(super) struct Mapped<T> {
    map: HostView<'static, T>,
    strides: Option<Vec<usize>>,
}

impl<T> Mapped<T>
where
    T: num_traits::Num + Clone + std::fmt::Debug,
    HostView<'static, T>: TensorMapTrait<T>,
{
    /// View the mapped data as an `ArrayViewD`.
    fn view(&self) -> Result<ArrayViewD<'_, T>, DecoderError> {
        strided_view(
            self.map.shape(),
            self.strides.as_deref(),
            self.map.as_slice(),
        )
    }
}

/// Map `$t` (already dtype-checked as `$ty`) read-only into a `Mapped`.
macro_rules! map_input {
    ($t:expr, $ty:ty) => {
        Mapped {
            strides: padded_strides($t)?,
            map: $t
                .as_typed::<$ty>()
                .ok_or_else(|| {
                    DecoderError::Internal(format!("{:?} is not {}", $t.dtype(), stringify!($ty)))
                })?
                .map_read()
                .map_err(|e| DecoderError::InvalidConfig(format!("tensor map failed: {e}")))?,
        }
    };
}

/// Mapped tensor outputs, grouped by dtype category.
pub(super) enum MappedOutputs {
    /// All outputs are integer types (u8, i8, u16, i16, u32, i32).
    Quantized(Vec<QuantizedMap>),
    /// All outputs are f16 (TensorRT fp16, GPU half-float models).
    Float16(Vec<Mapped<f16>>),
    /// All outputs are f32.
    Float32(Vec<Mapped<f32>>),
    /// All outputs are f64.
    Float64(Vec<Mapped<f64>>),
}

/// A mapped quantized tensor preserving the concrete integer type.
pub(super) enum QuantizedMap {
    U8(Mapped<u8>),
    I8(Mapped<i8>),
    U16(Mapped<u16>),
    I16(Mapped<i16>),
    U32(Mapped<u32>),
    I32(Mapped<i32>),
}

impl QuantizedMap {
    /// Create an `ArrayViewDQuantized` borrowing from the mapped data.
    pub(super) fn as_view(&self) -> Result<ArrayViewDQuantized<'_>, DecoderError> {
        Ok(match self {
            Self::U8(m) => ArrayViewDQuantized::UInt8(m.view()?),
            Self::I8(m) => ArrayViewDQuantized::Int8(m.view()?),
            Self::U16(m) => ArrayViewDQuantized::UInt16(m.view()?),
            Self::I16(m) => ArrayViewDQuantized::Int16(m.view()?),
            Self::U32(m) => ArrayViewDQuantized::UInt32(m.view()?),
            Self::I32(m) => ArrayViewDQuantized::Int32(m.view()?),
        })
    }
}

/// Map `TensorDyn` outputs into memory, detecting whether they are quantized
/// (integer) or floating-point.
///
/// Outputs are mapped read-only, and a padded row pitch is carried into the
/// views as explicit strides.
///
/// All integer types (u8, i8, u16, i16, u32, i32) are grouped as quantized.
/// Float types (f16, f32, f64) are grouped by precision. Mixed float/integer
/// inputs are an error, except that i32 tensors mixed with a float dtype are
/// allowed (some models produce mixed outputs where shape/count tensors are
/// i32).
///
/// # Errors
///
/// Returns `DecoderError::InvalidConfig` if:
/// - The output slice is empty
/// - Tensor memory mapping fails
/// - Tensor types are mixed in an unsupported way
/// - An unsupported dtype is encountered (u64, i64)
///
/// Returns `DecoderError::InvalidShape` if an output's row pitch is invalid.
pub(super) fn map_tensors(outputs: &[&TensorDyn]) -> Result<MappedOutputs, DecoderError> {
    if outputs.is_empty() {
        return Err(DecoderError::InvalidConfig("no outputs".to_string()));
    }

    // Determine the category
    let mut float_dtype = None;
    for &t in outputs {
        if matches!(
            t.dtype(),
            edgefirst_tensor::DType::F16
                | edgefirst_tensor::DType::F32
                | edgefirst_tensor::DType::F64
        ) {
            float_dtype = Some(t.dtype());
            break;
        }
    }

    if let Some(float_dtype) = float_dtype {
        map_float_tensors(outputs, float_dtype)
    } else {
        map_quantized_tensors(outputs)
    }
}

/// Map all outputs as float tensors (f16, f32, or f64).
fn map_float_tensors(
    outputs: &[&TensorDyn],
    first_dtype: edgefirst_tensor::DType,
) -> Result<MappedOutputs, DecoderError> {
    match first_dtype {
        edgefirst_tensor::DType::F16 => {
            let mut maps = Vec::with_capacity(outputs.len());
            for &t in outputs {
                match t.dtype() {
                    edgefirst_tensor::DType::F16 => maps.push(map_input!(t, f16)),
                    // Mixed f16 + i32 is allowed: skip i32 tensors silently
                    // (matches the f32 path's handling of count/shape tensors).
                    edgefirst_tensor::DType::I32 => continue,
                    _ => {
                        return Err(DecoderError::InvalidConfig(format!(
                            "mixed tensor types: expected f16, got {:?}",
                            t.dtype()
                        )));
                    }
                }
            }
            Ok(MappedOutputs::Float16(maps))
        }
        edgefirst_tensor::DType::F32 => {
            let mut maps = Vec::with_capacity(outputs.len());
            for &t in outputs {
                match t.dtype() {
                    edgefirst_tensor::DType::F32 => maps.push(map_input!(t, f32)),
                    // Some models have mixed f32 + i32 outputs (e.g. count tensors).
                    // Skip i32 tensors silently; the decoder indexes only f32 outputs.
                    edgefirst_tensor::DType::I32 => continue,
                    _ => {
                        return Err(DecoderError::InvalidConfig(format!(
                            "mixed tensor types: expected f32, got {:?}",
                            t.dtype()
                        )));
                    }
                }
            }
            Ok(MappedOutputs::Float32(maps))
        }
        edgefirst_tensor::DType::F64 => {
            let mut maps = Vec::with_capacity(outputs.len());
            for &t in outputs {
                match t.dtype() {
                    edgefirst_tensor::DType::F64 => maps.push(map_input!(t, f64)),
                    _ => {
                        return Err(DecoderError::InvalidConfig(format!(
                            "mixed tensor types: expected f64, got {:?}",
                            t.dtype()
                        )));
                    }
                }
            }
            Ok(MappedOutputs::Float64(maps))
        }
        _ => unreachable!("map_float_tensors called with non-float dtype {first_dtype:?}"),
    }
}

/// Map all outputs as quantized (integer) tensors.
fn map_quantized_tensors(outputs: &[&TensorDyn]) -> Result<MappedOutputs, DecoderError> {
    let mut maps = Vec::with_capacity(outputs.len());
    for &t in outputs {
        let qmap = match t.dtype() {
            edgefirst_tensor::DType::U8 => QuantizedMap::U8(map_input!(t, u8)),
            edgefirst_tensor::DType::I8 => QuantizedMap::I8(map_input!(t, i8)),
            edgefirst_tensor::DType::U16 => QuantizedMap::U16(map_input!(t, u16)),
            edgefirst_tensor::DType::I16 => QuantizedMap::I16(map_input!(t, i16)),
            edgefirst_tensor::DType::U32 => QuantizedMap::U32(map_input!(t, u32)),
            edgefirst_tensor::DType::I32 => QuantizedMap::I32(map_input!(t, i32)),
            _ => {
                return Err(DecoderError::InvalidConfig(format!(
                    "unsupported tensor dtype for quantized decode: {:?}",
                    t.dtype()
                )));
            }
        };
        maps.push(qmap);
    }
    Ok(MappedOutputs::Quantized(maps))
}

/// Convert a slice of `QuantizedMap` into `ArrayViewDQuantized` views.
pub(super) fn quantized_views(
    maps: &[QuantizedMap],
) -> Result<Vec<ArrayViewDQuantized<'_>>, DecoderError> {
    maps.iter().map(QuantizedMap::as_view).collect()
}

/// Convert a slice of mapped f16 outputs into `ArrayViewD<f16>` views.
pub(super) fn f16_views(maps: &[Mapped<f16>]) -> Result<Vec<ArrayViewD<'_, f16>>, DecoderError> {
    maps.iter().map(Mapped::view).collect()
}

/// Convert a slice of mapped f32 outputs into `ArrayViewD<f32>` views.
pub(super) fn f32_views(maps: &[Mapped<f32>]) -> Result<Vec<ArrayViewD<'_, f32>>, DecoderError> {
    maps.iter().map(Mapped::view).collect()
}

/// Convert a slice of mapped f64 outputs into `ArrayViewD<f64>` views.
pub(super) fn f64_views(maps: &[Mapped<f64>]) -> Result<Vec<ArrayViewD<'_, f64>>, DecoderError> {
    maps.iter().map(Mapped::view).collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tensor_bridge_tests {
    use edgefirst_tensor::{DType, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

    use crate::decoder::tensor_bridge::{self, MappedOutputs, QuantizedMap};
    use crate::decoder::ArrayViewDQuantized;

    macro_rules! make_tensor_fn {
        ($name:ident, $ty:ty, $variant:ident) => {
            fn $name(shape: &[usize], values: &[$ty]) -> TensorDyn {
                let t = Tensor::<$ty>::new(shape, Some(TensorMemory::Mem), None).unwrap();
                let mut m = t.map().unwrap();
                m.as_mut_slice()[..values.len()].copy_from_slice(values);
                drop(m);
                t.into()
            }
        };
    }

    make_tensor_fn!(make_u8, u8, U8);
    make_tensor_fn!(make_i8, i8, I8);
    make_tensor_fn!(make_u16, u16, U16);
    make_tensor_fn!(make_i16, i16, I16);
    make_tensor_fn!(make_u32, u32, U32);
    make_tensor_fn!(make_i32, i32, I32);
    make_tensor_fn!(make_f32, f32, F32);
    make_tensor_fn!(make_f64, f64, F64);

    fn make_f16(shape: &[usize], values: &[half::f16]) -> TensorDyn {
        let t = Tensor::<half::f16>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        let mut m = t.map().unwrap();
        m.as_mut_slice()[..values.len()].copy_from_slice(values);
        drop(m);
        t.into()
    }

    // ─── row pitch tests ─────────────────────────────────

    fn grey_region_view() -> TensorDyn {
        let parent = Tensor::<u8>::image(
            4,
            2,
            edgefirst_tensor::PixelFormat::Grey,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        parent
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 99, 4, 5, 6, 99]);
        parent
            .view(edgefirst_tensor::Region::new(0, 0, 3, 2))
            .unwrap()
            .into()
    }

    #[test]
    fn padded_strides_none_for_packed_tensor() {
        let t = make_u8(&[2, 3, 1], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(tensor_bridge::padded_strides(&t).unwrap(), None);
    }

    #[test]
    fn padded_strides_follow_the_row_pitch() {
        let t = grey_region_view();
        assert_eq!(t.shape(), &[2, 3, 1]);
        assert_eq!(t.row_stride(), Some(4));
        assert_eq!(
            tensor_bridge::padded_strides(&t).unwrap(),
            Some(vec![4, 1, 1])
        );
    }

    #[test]
    fn padded_input_views_skip_the_padding() {
        let t = grey_region_view();
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        let MappedOutputs::Quantized(maps) = &mapped else {
            panic!("expected Quantized");
        };
        let ArrayViewDQuantized::UInt8(view) = maps[0].as_view().unwrap() else {
            panic!("expected UInt8");
        };
        assert_eq!(view.shape(), &[2, 3, 1]);
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn padded_strides_reject_a_pitch_shorter_than_a_row() {
        let mut t = make_u8(&[2, 3, 1], &[1, 2, 3, 4, 5, 6]);
        t.set_row_stride_unchecked(2);
        assert!(matches!(
            tensor_bridge::padded_strides(&t),
            Err(crate::DecoderError::InvalidShape(_))
        ));
        assert!(tensor_bridge::map_tensors(&[&t]).is_err());
    }

    // ─── map_tensors tests ───────────────────────────────

    #[test]
    fn test_map_tensors_empty_slice() {
        let result = tensor_bridge::map_tensors(&[]);
        assert!(result.is_err());
    }

    macro_rules! test_map_tensors_single {
        ($test_name:ident, $make_fn:ident, $variant:pat) => {
            #[test]
            fn $test_name() {
                let t = $make_fn(&[2, 2], &[1, 2, 3, 4]);
                let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
                assert!(matches!(mapped, MappedOutputs::Quantized(_)));
                if let MappedOutputs::Quantized(maps) = &mapped {
                    assert_eq!(maps.len(), 1);
                    assert!(matches!(&maps[0], $variant));
                }
            }
        };
    }

    test_map_tensors_single!(test_map_tensors_single_u8, make_u8, QuantizedMap::U8(_));
    test_map_tensors_single!(test_map_tensors_single_i8, make_i8, QuantizedMap::I8(_));
    test_map_tensors_single!(test_map_tensors_single_u16, make_u16, QuantizedMap::U16(_));
    test_map_tensors_single!(test_map_tensors_single_i16, make_i16, QuantizedMap::I16(_));
    test_map_tensors_single!(test_map_tensors_single_u32, make_u32, QuantizedMap::U32(_));
    test_map_tensors_single!(test_map_tensors_single_i32, make_i32, QuantizedMap::I32(_));

    #[test]
    fn test_map_tensors_single_f32() {
        let t = make_f32(&[1, 3], &[1.0, 2.5, 3.75]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            assert_eq!(maps.len(), 1);
        } else {
            panic!("expected MappedOutputs::Float32");
        }
    }

    #[test]
    fn test_map_tensors_single_f64() {
        let t = make_f64(&[1, 3], &[1.0, 2.5, 3.75]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float64(maps) = &mapped {
            assert_eq!(maps.len(), 1);
        } else {
            panic!("expected MappedOutputs::Float64");
        }
    }

    #[test]
    fn test_map_tensors_multiple_quantized() {
        let t1 = make_u8(&[1, 4], &[10, 20, 30, 40]);
        let t2 = make_i8(&[1, 3], &[-1, 0, 1]);
        let t3 = make_u16(&[1, 2], &[500, 600]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2, &t3]).unwrap();
        if let MappedOutputs::Quantized(maps) = &mapped {
            assert_eq!(maps.len(), 3);
            assert!(matches!(&maps[0], QuantizedMap::U8(_)));
            assert!(matches!(&maps[1], QuantizedMap::I8(_)));
            assert!(matches!(&maps[2], QuantizedMap::U16(_)));
        } else {
            panic!("expected MappedOutputs::Quantized");
        }
    }

    #[test]
    fn test_map_tensors_multiple_f32() {
        let t1 = make_f32(&[1, 2], &[1.0, 2.0]);
        let t2 = make_f32(&[1, 3], &[3.0, 4.0, 5.0]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            assert_eq!(maps.len(), 2);
        } else {
            panic!("expected MappedOutputs::Float32");
        }
    }

    #[test]
    fn test_map_tensors_f32_with_i32_skipped() {
        // Mixed f32 + i32 is allowed: i32 tensors are silently skipped.
        let t1 = make_f32(&[1, 2], &[1.0, 2.0]);
        let t2 = make_f32(&[1, 3], &[3.0, 4.0, 5.0]);
        let t3 = make_i32(&[1, 1], &[42]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2, &t3]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            // The i32 tensor should be skipped
            assert_eq!(maps.len(), 2);
        } else {
            panic!("expected MappedOutputs::Float32");
        }

        // check that order doesn't matter
        let mapped = tensor_bridge::map_tensors(&[&t3, &t2, &t1]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            // The i32 tensor should be skipped
            assert_eq!(maps.len(), 2);
        } else {
            panic!("expected MappedOutputs::Float32");
        }
    }

    #[test]
    fn test_map_tensors_mixed_f32_u8_rejected() {
        let t1 = make_f32(&[1, 2], &[1.0, 2.0]);
        let t2 = make_u8(&[1, 2], &[10, 20]);
        let result = tensor_bridge::map_tensors(&[&t1, &t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_map_tensors_mixed_f64_f32_rejected() {
        let t1 = make_f64(&[1, 2], &[1.0, 2.0]);
        let t2 = make_f32(&[1, 2], &[3.0, 4.0]);
        let result = tensor_bridge::map_tensors(&[&t1, &t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_views_u8_data() {
        let t = make_u8(&[1, 2, 3], &[10, 20, 30, 40, 50, 60]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Quantized(maps) = &mapped {
            let views = tensor_bridge::quantized_views(maps).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[1, 2, 3]);
        } else {
            panic!("expected MappedOutputs::Quantized");
        }
    }

    #[test]
    fn test_quantized_views_i8_data() {
        let t = make_i8(&[2, 3], &[-10, 20, -30, 40, -50, 60]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Quantized(maps) = &mapped {
            let views = tensor_bridge::quantized_views(maps).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[2, 3]);
        } else {
            panic!("expected MappedOutputs::Quantized");
        }
    }

    #[test]
    fn test_quantized_views_multiple_types() {
        let t1 = make_u8(&[1, 4], &[1, 2, 3, 4]);
        let t2 = make_i16(&[2, 2], &[-1, 2, -3, 4]);
        let t3 = make_u32(&[1, 3], &[100, 200, 300]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2, &t3]).unwrap();
        if let MappedOutputs::Quantized(maps) = &mapped {
            let views = tensor_bridge::quantized_views(maps).unwrap();
            assert_eq!(views.len(), 3);
            assert_eq!(views[0].shape(), &[1, 4]);
            assert_eq!(views[1].shape(), &[2, 2]);
            assert_eq!(views[2].shape(), &[1, 3]);
        } else {
            panic!("expected MappedOutputs::Quantized");
        }
    }

    #[test]
    fn test_f32_views_data() {
        let t = make_f32(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            let views = tensor_bridge::f32_views(maps).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[1, 2, 3]);
            assert_eq!(views[0][[0, 0, 0]], 1.0);
            assert_eq!(views[0][[0, 0, 1]], 2.0);
            assert_eq!(views[0][[0, 0, 2]], 3.0);
            assert_eq!(views[0][[0, 1, 0]], 4.0);
            assert_eq!(views[0][[0, 1, 1]], 5.0);
            assert_eq!(views[0][[0, 1, 2]], 6.0);
        } else {
            panic!("expected Float32");
        }
    }

    #[test]
    fn test_f32_views_multiple() {
        let t1 = make_f32(&[1, 2], &[1.5, 2.5]);
        let t2 = make_f32(&[2, 2], &[3.0, 4.0, 5.0, 6.0]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2]).unwrap();
        if let MappedOutputs::Float32(maps) = &mapped {
            let views = tensor_bridge::f32_views(maps).unwrap();
            assert_eq!(views.len(), 2);
            assert_eq!(views[0][[0, 0]], 1.5);
            assert_eq!(views[1][[1, 1]], 6.0);
        } else {
            panic!("expected Float32");
        }
    }

    // ─── f64_views tests ────────────────────────────────

    #[test]
    fn test_f64_views_data() {
        let t = make_f64(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float64(maps) = &mapped {
            let views = tensor_bridge::f64_views(maps).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[1, 2, 3]);
            assert_eq!(views[0][[0, 0, 0]], 1.0);
            assert_eq!(views[0][[0, 1, 2]], 6.0);
        } else {
            panic!("expected Float64");
        }
    }

    macro_rules! test_quantized_map_as_view {
        ($name:ident, $make_fn:ident, $variant:pat) => {
            #[test]
            fn $name() {
                let t = $make_fn(&[1, 2, 3], &[6, 5, 4, 3, 2, 1]);
                let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
                if let MappedOutputs::Quantized(maps) = &mapped {
                    let view = maps[0].as_view().unwrap();
                    assert!(matches!(&view, $variant));
                    assert_eq!(view.shape(), &[1, 2, 3]);
                } else {
                    panic!("expected Quantized");
                }
            }
        };
    }

    test_quantized_map_as_view!(test_as_view_u8, make_u8, ArrayViewDQuantized::UInt8(_));
    test_quantized_map_as_view!(test_as_view_i8, make_i8, ArrayViewDQuantized::Int8(_));
    test_quantized_map_as_view!(test_as_view_u16, make_u16, ArrayViewDQuantized::UInt16(_));
    test_quantized_map_as_view!(test_as_view_i16, make_i16, ArrayViewDQuantized::Int16(_));
    test_quantized_map_as_view!(test_as_view_u32, make_u32, ArrayViewDQuantized::UInt32(_));
    test_quantized_map_as_view!(test_as_view_i32, make_i32, ArrayViewDQuantized::Int32(_));

    // ─── Unsupported dtype tests ────────────────────────

    macro_rules! test_map_tensors_rejected {
        ($test_name:ident, $dtype:expr) => {
            #[test]
            fn $test_name() {
                let t = TensorDyn::new(&[1, 4], $dtype, Some(TensorMemory::Mem), None).unwrap();
                let result = tensor_bridge::map_tensors(&[&t]);
                assert!(result.is_err());
            }
        };
    }

    test_map_tensors_rejected!(test_map_tensors_u64_rejected, DType::U64);
    test_map_tensors_rejected!(test_map_tensors_i64_rejected, DType::I64);

    // ─── f16 acceptance tests ───────────────────────────

    #[test]
    fn test_map_tensors_single_f16() {
        let t = make_f16(
            &[1, 3],
            &[
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.5),
                half::f16::from_f32(3.75),
            ],
        );
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float16(maps) = &mapped {
            assert_eq!(maps.len(), 1);
        } else {
            panic!("expected MappedOutputs::Float16");
        }
    }

    #[test]
    fn test_map_tensors_multiple_f16() {
        let t1 = make_f16(
            &[1, 2],
            &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        );
        let t2 = make_f16(
            &[1, 3],
            &[
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
                half::f16::from_f32(5.0),
            ],
        );
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2]).unwrap();
        if let MappedOutputs::Float16(maps) = &mapped {
            assert_eq!(maps.len(), 2);
        } else {
            panic!("expected MappedOutputs::Float16");
        }
    }

    #[test]
    fn test_map_tensors_f16_with_i32_skipped() {
        // Mixed f16 + i32 is allowed: i32 tensors are silently skipped,
        // mirroring the f32 path (some models emit count/shape tensors as i32).
        let t1 = make_f16(
            &[1, 2],
            &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        );
        let t2 = make_i32(&[1, 1], &[42]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2]).unwrap();
        if let MappedOutputs::Float16(maps) = &mapped {
            assert_eq!(maps.len(), 1);
        } else {
            panic!("expected MappedOutputs::Float16");
        }
    }

    #[test]
    fn test_map_tensors_mixed_f16_f32_rejected() {
        let t1 = make_f16(
            &[1, 2],
            &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        );
        let t2 = make_f32(&[1, 2], &[3.0, 4.0]);
        let result = tensor_bridge::map_tensors(&[&t1, &t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_map_tensors_mixed_f16_u8_rejected() {
        let t1 = make_f16(
            &[1, 2],
            &[half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        );
        let t2 = make_u8(&[1, 2], &[10, 20]);
        let result = tensor_bridge::map_tensors(&[&t1, &t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_f16_views_data() {
        let t = make_f16(
            &[1, 2, 3],
            &[
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
                half::f16::from_f32(5.0),
                half::f16::from_f32(6.0),
            ],
        );
        let mapped = tensor_bridge::map_tensors(&[&t]).unwrap();
        if let MappedOutputs::Float16(maps) = &mapped {
            let views = tensor_bridge::f16_views(maps).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].shape(), &[1, 2, 3]);
            assert_eq!(views[0][[0, 0, 0]], half::f16::from_f32(1.0));
            assert_eq!(views[0][[0, 1, 2]], half::f16::from_f32(6.0));
        } else {
            panic!("expected Float16");
        }
    }
}
