// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between `TensorDyn` (the HAL's fundamental type-erased tensor) and
//! the decoder's internal `ArrayViewD`/`ArrayViewDQuantized` representations.
//!
//! This module maps `TensorDyn` outputs into memory and converts them into
//! ndarray views that the existing decode methods consume.

use edgefirst_tensor::{TensorDyn, TensorMap, TensorMapTrait, TensorTrait};
use ndarray::ArrayViewD;

use super::ArrayViewDQuantized;
use crate::DecoderError;

/// Mapped tensor outputs, grouped by dtype category.
///
/// The `TensorMap` values borrow from the original `TensorDyn` tensors. The
/// ndarray views created from these maps must not outlive the `MappedOutputs`.
pub(super) enum MappedOutputs {
    /// All outputs are integer types (u8, i8, u16, i16, u32, i32).
    Quantized(Vec<QuantizedMap>),
    /// All outputs are f32.
    Float32(Vec<TensorMap<f32>>),
    /// All outputs are f64.
    Float64(Vec<TensorMap<f64>>),
}

/// A mapped quantized tensor preserving the concrete integer type.
pub(super) enum QuantizedMap {
    U8(TensorMap<u8>),
    I8(TensorMap<i8>),
    U16(TensorMap<u16>),
    I16(TensorMap<i16>),
    U32(TensorMap<u32>),
    I32(TensorMap<i32>),
}

impl QuantizedMap {
    /// Create an `ArrayViewDQuantized` borrowing from the mapped data.
    pub(super) fn as_view(&self) -> Result<ArrayViewDQuantized<'_>, DecoderError> {
        macro_rules! make_view {
            ($map:expr, $variant:ident) => {{
                let shape = $map.shape().to_vec();
                let slice = $map.as_slice();
                ArrayViewD::from_shape(shape.as_slice(), slice)
                    .map(|v| ArrayViewDQuantized::$variant(v))
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor shape: {e}")))
            }};
        }
        match self {
            Self::U8(m) => make_view!(m, UInt8),
            Self::I8(m) => make_view!(m, Int8),
            Self::U16(m) => make_view!(m, UInt16),
            Self::I16(m) => make_view!(m, Int16),
            Self::U32(m) => make_view!(m, UInt32),
            Self::I32(m) => make_view!(m, Int32),
        }
    }
}

/// Map `TensorDyn` outputs into memory, detecting whether they are quantized
/// (integer) or floating-point.
///
/// All integer types (u8, i8, u16, i16, u32, i32) are grouped as quantized.
/// Float types (f32, f64) are grouped by precision. Mixed float/integer
/// inputs are an error, except that i32 tensors mixed with f32 are allowed
/// (some models produce mixed outputs where shape/count tensors are i32).
///
/// # Errors
///
/// Returns `DecoderError::InvalidConfig` if:
/// - The output slice is empty
/// - Tensor memory mapping fails
/// - Tensor types are mixed in an unsupported way
/// - An unsupported dtype is encountered (u64, i64, f16)
pub(super) fn map_tensors(outputs: &[&TensorDyn]) -> Result<MappedOutputs, DecoderError> {
    if outputs.is_empty() {
        return Err(DecoderError::InvalidConfig("no outputs".to_string()));
    }

    // Determine the category from the first tensor
    let first_dtype = outputs[0].dtype();
    let is_float = matches!(
        first_dtype,
        edgefirst_tensor::DType::F32 | edgefirst_tensor::DType::F64
    );

    if is_float {
        map_float_tensors(outputs, first_dtype)
    } else {
        map_quantized_tensors(outputs)
    }
}

/// Map all outputs as float tensors (f32 or f64).
fn map_float_tensors(
    outputs: &[&TensorDyn],
    first_dtype: edgefirst_tensor::DType,
) -> Result<MappedOutputs, DecoderError> {
    if first_dtype == edgefirst_tensor::DType::F32 {
        let mut maps = Vec::with_capacity(outputs.len());
        for &t in outputs {
            match t {
                TensorDyn::F32(tensor) => {
                    maps.push(tensor.map().map_err(|e| {
                        DecoderError::InvalidConfig(format!("tensor map failed: {e}"))
                    })?);
                }
                // Some models have mixed f32 + i32 outputs (e.g. count tensors).
                // Skip i32 tensors silently; the decoder indexes only f32 outputs.
                TensorDyn::I32(_) => continue,
                _ => {
                    return Err(DecoderError::InvalidConfig(format!(
                        "mixed tensor types: expected f32, got {:?}",
                        t.dtype()
                    )));
                }
            }
        }
        Ok(MappedOutputs::Float32(maps))
    } else {
        // f64
        let mut maps = Vec::with_capacity(outputs.len());
        for &t in outputs {
            match t {
                TensorDyn::F64(tensor) => {
                    maps.push(tensor.map().map_err(|e| {
                        DecoderError::InvalidConfig(format!("tensor map failed: {e}"))
                    })?);
                }
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
}

/// Map all outputs as quantized (integer) tensors.
fn map_quantized_tensors(outputs: &[&TensorDyn]) -> Result<MappedOutputs, DecoderError> {
    let mut maps = Vec::with_capacity(outputs.len());
    for &t in outputs {
        let qmap = match t {
            TensorDyn::U8(tensor) => QuantizedMap::U8(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
            TensorDyn::I8(tensor) => QuantizedMap::I8(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
            TensorDyn::U16(tensor) => QuantizedMap::U16(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
            TensorDyn::I16(tensor) => QuantizedMap::I16(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
            TensorDyn::U32(tensor) => QuantizedMap::U32(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
            TensorDyn::I32(tensor) => QuantizedMap::I32(
                tensor
                    .map()
                    .map_err(|e| DecoderError::InvalidConfig(format!("tensor map: {e}")))?,
            ),
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
    maps.iter().map(|m| m.as_view()).collect()
}

/// Convert a slice of `TensorMap<f32>` into `ArrayViewD<f32>` views.
pub(super) fn f32_views(maps: &[TensorMap<f32>]) -> Result<Vec<ArrayViewD<'_, f32>>, DecoderError> {
    maps.iter()
        .map(|m| {
            let shape = m.shape().to_vec();
            ArrayViewD::from_shape(shape.as_slice(), m.as_slice())
                .map_err(|e| DecoderError::InvalidConfig(format!("tensor shape: {e}")))
        })
        .collect()
}

/// Convert a slice of `TensorMap<f64>` into `ArrayViewD<f64>` views.
pub(super) fn f64_views(maps: &[TensorMap<f64>]) -> Result<Vec<ArrayViewD<'_, f64>>, DecoderError> {
    maps.iter()
        .map(|m| {
            let shape = m.shape().to_vec();
            ArrayViewD::from_shape(shape.as_slice(), m.as_slice())
                .map_err(|e| DecoderError::InvalidConfig(format!("tensor shape: {e}")))
        })
        .collect()
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
                TensorDyn::$variant(t)
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
        let t2 = make_i32(&[1, 1], &[42]);
        let t3 = make_f32(&[1, 3], &[3.0, 4.0, 5.0]);
        let mapped = tensor_bridge::map_tensors(&[&t1, &t2, &t3]).unwrap();
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

    test_map_tensors_rejected!(test_map_tensors_f16_rejected, DType::F16);
    test_map_tensors_rejected!(test_map_tensors_u64_rejected, DType::U64);
    test_map_tensors_rejected!(test_map_tensors_i64_rejected, DType::I64);
}
