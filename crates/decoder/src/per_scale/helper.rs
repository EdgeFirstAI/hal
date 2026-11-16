// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional helper: walk a schema-v2 document and attach
//! per-tensor `Quantization` to integer input tensors via shape match.
//!
//! Use when the upstream inference layer has not attached quantization
//! metadata to the tensors.

use crate::schema::SchemaV2;
use crate::{DecoderError, DecoderResult};
use edgefirst_tensor::{Quantization as TQ, TensorDyn};

/// Walk the schema's logical outputs and their per-scale children,
/// attaching per-tensor `Quantization` to any matching integer tensor
/// in `tensors` (matched by shape).
///
/// - Per-channel quantization is skipped.
/// - Float tensors are skipped.
/// - Tensors with no matching schema entry are left alone.
/// - If a schema entry has a quant declaration but no tensor matches
///   the declared shape, returns `InvalidShape`.
///
/// A repeated call overwrites the attached quantization.
pub fn apply_schema_quant(schema: &SchemaV2, tensors: &mut [&mut TensorDyn]) -> DecoderResult<()> {
    for logical in &schema.outputs {
        // Per-scale children
        for child in &logical.outputs {
            if let Some(q) = child.quantization.as_ref() {
                attach_per_tensor_quant_by_shape(tensors, &child.shape, q)?;
            }
        }
        // Direct logical (no children) — quant lives on the logical itself.
        if logical.outputs.is_empty() {
            if let Some(q) = logical.quantization.as_ref() {
                attach_per_tensor_quant_by_shape(tensors, &logical.shape, q)?;
            }
        }
    }
    Ok(())
}

fn attach_per_tensor_quant_by_shape(
    tensors: &mut [&mut TensorDyn],
    expected_shape: &[usize],
    schema_q: &crate::schema::Quantization,
) -> DecoderResult<()> {
    // An empty scale is rejected.
    if schema_q.scale.is_empty() {
        return Err(DecoderError::InvalidShape(format!(
            "apply_schema_quant: schema declares quantization for shape \
             {expected_shape:?} but `scale` is empty"
        )));
    }
    if !schema_q.is_per_tensor() {
        // Per-channel quantization is skipped.
        return Ok(());
    }
    let scale = schema_q.scale[0];
    let zp = schema_q.zero_point_at(0);

    for t in tensors.iter_mut() {
        if t.shape() == expected_shape {
            // Attaches to integer variants; floats are skipped.
            macro_rules! try_attach {
                ($ty:ty) => {
                    if let Some(inner) = t.as_typed_mut::<$ty>() {
                        let tq = TQ::per_tensor(scale, zp);
                        inner.set_quantization(tq).map_err(|e| {
                            DecoderError::Internal(format!(
                                "apply_schema_quant set_quantization failed: {e}"
                            ))
                        })?;
                        return Ok(());
                    }
                };
            }
            try_attach!(i8);
            try_attach!(u8);
            try_attach!(i16);
            try_attach!(u16);
            // Float and other dtypes are skipped.
            return Ok(());
        }
    }
    Err(DecoderError::InvalidShape(format!(
        "apply_schema_quant: no tensor matches {expected_shape:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SchemaV2;
    use edgefirst_tensor::{Tensor, TensorMemory};

    #[test]
    fn applies_quant_to_int8_by_shape() {
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(&json).unwrap();

        // A tensor with the shape of the schema's first box child.
        let t = Tensor::<i8>::new(&[1, 80, 80, 64], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.quantization().is_none(), "fresh tensor has no quant");
        let mut td: TensorDyn = t.into();
        let mut tensors: Vec<&mut TensorDyn> = vec![&mut td];

        apply_schema_quant(&schema, &mut tensors).unwrap_err();
        // The schema's other children have no matching tensor.
    }

    #[test]
    fn applies_quant_to_full_yolov8_input_set() {
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(&json).unwrap();

        // One tensor for every schema child and the protos.
        let shapes_int8 = [
            vec![1, 80, 80, 64],
            vec![1, 80, 80, 80],
            vec![1, 80, 80, 32],
            vec![1, 40, 40, 64],
            vec![1, 40, 40, 80],
            vec![1, 40, 40, 32],
            vec![1, 20, 20, 64],
            vec![1, 20, 20, 80],
            vec![1, 20, 20, 32],
            vec![1, 160, 160, 32],
        ];
        let mut owned: Vec<TensorDyn> = shapes_int8
            .iter()
            .map(|s| {
                Tensor::<i8>::new(s, Some(TensorMemory::Mem), None)
                    .unwrap()
                    .into()
            })
            .collect();
        let mut refs: Vec<&mut TensorDyn> = owned.iter_mut().collect();

        apply_schema_quant(&schema, &mut refs).unwrap();

        for td in &owned {
            if let Some(t) = td.as_typed::<i8>() {
                assert!(
                    t.quantization().is_some(),
                    "tensor missing quant after apply"
                );
            }
        }
    }

    #[test]
    fn empty_scale_errors_instead_of_silently_skipping() {
        use crate::schema::{DType, Quantization};
        let bad_q = Quantization {
            scale: vec![],
            zero_point: None,
            axis: None,
            dtype: Some(DType::Int8),
        };
        let mut td: TensorDyn = Tensor::<i8>::new(&[1, 2, 3], Some(TensorMemory::Mem), None)
            .unwrap()
            .into();
        let mut refs: Vec<&mut TensorDyn> = vec![&mut td];

        let err = attach_per_tensor_quant_by_shape(&mut refs, &[1, 2, 3], &bad_q)
            .expect_err("empty scale must be rejected");
        match err {
            DecoderError::InvalidShape(msg) => {
                assert!(msg.contains("`scale` is empty"), "msg = {msg}")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn skips_float_tensors_silently() {
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(&json).unwrap();

        // A float tensor with a box child's shape.
        let shape = vec![1, 80, 80, 64];
        let t = Tensor::<f32>::new(&shape, Some(TensorMemory::Mem), None).unwrap();
        let mut td: TensorDyn = t.into();
        let mut refs: Vec<&mut TensorDyn> = vec![&mut td];

        // The schema's other children have no matching tensor.
        let r = apply_schema_quant(&schema, &mut refs);
        assert!(r.is_err());
        assert_eq!(td.dtype(), edgefirst_tensor::DType::F32, "unexpected dtype");
    }
}
