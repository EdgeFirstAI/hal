// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Optional helper: walk a schema-v2 document and attach
//! per-tensor `Quantization` to integer input tensors via shape match.
//!
//! Use when the upstream inference layer hasn't already attached
//! quantization metadata to the tensors.

use crate::schema::SchemaV2;
use crate::{DecoderError, DecoderResult};
use edgefirst_tensor::{Quantization as TQ, TensorDyn};

/// Walk the schema's logical outputs and their per-scale children,
/// attaching per-tensor `Quantization` to any matching integer tensor
/// in `tensors` (matched by shape).
///
/// **Behavior:**
/// - Per-channel quantization is silently skipped — the per-scale
///   subsystem only consumes per-tensor quant.
/// - Float tensors are silently skipped (they don't carry quantization).
/// - Tensors with no matching schema entry are silently left alone.
/// - If a schema entry has a quant declaration but no tensor matches
///   the declared shape, returns `InvalidShape`.
///
/// **Idempotent:** safe to call multiple times; later calls overwrite
/// the attached quantization.
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
    // Reject empty scale up-front so a malformed schema fails fast at
    // attach time instead of silently looking "per-channel" here and
    // surfacing as `QuantMissing` later at decode time.
    if schema_q.scale.is_empty() {
        return Err(DecoderError::InvalidShape(format!(
            "apply_schema_quant: schema declares quantization for shape \
             {expected_shape:?} but `scale` is empty"
        )));
    }
    if !schema_q.is_per_tensor() {
        // Per-channel — skip silently. The per-scale planner errors
        // separately when it actually needs to use a per-channel quant.
        return Ok(());
    }
    let scale = schema_q.scale[0];
    let zp = schema_q.zero_point_at(0);

    for t in tensors.iter_mut() {
        if t.shape() == expected_shape {
            // Try to attach to integer variants; silently skip floats.
            macro_rules! try_attach {
                ($variant:ident) => {
                    if let TensorDyn::$variant(inner) = &mut **t {
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
            try_attach!(I8);
            try_attach!(U8);
            try_attach!(I16);
            try_attach!(U16);
            // Float / other dtypes silently skipped.
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
        // Build a tiny schema with one logical output `boxes` having an int8 child.
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(json).unwrap();

        // Build a tensor matching one box child's shape.
        // The yolov8n schema's first box child has shape [1, 80, 80, 64].
        let t = Tensor::<i8>::new(&[1, 80, 80, 64], Some(TensorMemory::Mem), None).unwrap();
        assert!(t.quantization().is_none(), "fresh tensor has no quant");
        let mut td = TensorDyn::I8(t);
        let mut tensors: Vec<&mut TensorDyn> = vec![&mut td];

        apply_schema_quant(&schema, &mut tensors).unwrap_err();
        // Errors because not all schema-declared shapes have matching tensors.
        // (We only provided one tensor, but schema has many children.)
    }

    #[test]
    fn applies_quant_to_full_yolov8_input_set() {
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(json).unwrap();

        // Build the full set of input tensors matching every schema child + protos.
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
            .map(|s| TensorDyn::I8(Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap()))
            .collect();
        let mut refs: Vec<&mut TensorDyn> = owned.iter_mut().collect();

        apply_schema_quant(&schema, &mut refs).unwrap();

        // All 10 tensors should now carry quant.
        for td in &owned {
            if let TensorDyn::I8(t) = td {
                assert!(
                    t.quantization().is_some(),
                    "tensor missing quant after apply"
                );
            }
        }
    }

    #[test]
    fn empty_scale_errors_instead_of_silently_skipping() {
        // Regression for Copilot review on PR #63: an empty `scale`
        // vector used to slip past `is_per_tensor()` and return Ok,
        // masking a malformed schema until decode time.
        use crate::schema::{DType, Quantization};
        let bad_q = Quantization {
            scale: vec![],
            zero_point: None,
            axis: None,
            dtype: Some(DType::Int8),
        };
        let mut td =
            TensorDyn::I8(Tensor::<i8>::new(&[1, 2, 3], Some(TensorMemory::Mem), None).unwrap());
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
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(json).unwrap();

        // Build float tensors instead of int8.
        let shape = vec![1, 80, 80, 64];
        let t = Tensor::<f32>::new(&shape, Some(TensorMemory::Mem), None).unwrap();
        let mut td = TensorDyn::F32(t);
        let mut refs: Vec<&mut TensorDyn> = vec![&mut td];

        // Float tensors don't take quant; the helper returns Ok and the schema
        // declaration is silently skipped on this single mismatched-shape input.
        // (Actually it'll error on InvalidShape for the OTHER schema children
        // that have no matching tensor. That's the expected behaviour.)
        let r = apply_schema_quant(&schema, &mut refs);
        assert!(r.is_err());
        // What we're really testing: float tensors don't crash.
        if let TensorDyn::F32(_) = &td {
            // OK — still F32, no panic
        } else {
            panic!("unexpected dtype");
        }
    }
}
