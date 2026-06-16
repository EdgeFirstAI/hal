// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Physical-to-logical tensor merge path for schema v2.
//!
//! When a v2 metadata document declares that a logical output has been
//! split into physical children (for quantization-error minimisation),
//! the HAL must reassemble the logical tensor before handing it to the
//! existing float-path decoders.
//!
//! A [`DecodeProgram`] captures the per-logical recipe derived from the
//! schema at build time. At decode time the program runs against the
//! actual input tensors and produces one `ArrayD<f32>` per logical
//! output, ordered to match the schema's `outputs[]`.
//!
//! # Supported merges
//!
//! | Strategy | Trigger | Operation |
//! |---|---|---|
//! | [`LogicalMerge::Direct`] | logical has no children | find tensor by shape, dequantize to f32 |
//! | [`LogicalMerge::ChannelConcat`] | children lack `stride` (e.g. ARA-2 `boxes_xy` + `boxes_wh`) | dequantize each, concat along channel axis |
//!
//! Per-scale FPN splits (children carrying `stride`, e.g. Hailo
//! YOLOv8/v11 with `encoding: dfl`) are NOT handled here: the
//! `per_scale` subsystem ([`crate::per_scale::PerScalePlan`]) claims
//! those schemas in full at builder time, so the merge program is only
//! ever built for flat or channel-sub-split schemas.
//!
//! # Not yet supported
//!
//! - **Per-channel quantization** on split children — the HAL
//!   currently only consumes per-tensor scalar `(scale, zero_point)`.
//! - **Activation flags on floats** — when a child carries
//!   `activation_required: sigmoid` the HAL does not yet apply it
//!   during merge. For split ARA-2 typical usage the NPU applies
//!   sigmoid on-chip (`activation_applied`), so this is not blocking.

use ndarray::{Array, ArrayD, ArrayViewD, Axis, IxDyn};

use crate::configs::DimName;
use crate::schema::{self, padding_axes, squeeze_padding_dims, LogicalOutput, SchemaV2};
// `squeeze_padding_dims` is re-used by plan_channel_concat below.
use crate::{dequantize_cpu_chunked, DecoderError, DecoderResult, Quantization};
use edgefirst_tensor::{TensorDyn, TensorMapTrait, TensorTrait};

/// Compiled merge program for one schema v2 document.
///
/// The program holds one [`LogicalMerge`] per entry in
/// [`SchemaV2::outputs`], in schema order. [`DecodeProgram::execute`]
/// produces the matching vector of merged `f32` logical tensors.
#[derive(Debug, Clone)]
pub(crate) struct DecodeProgram {
    merges: Vec<LogicalMerge>,
}

/// One merge recipe per logical output.
#[derive(Debug, Clone)]
enum LogicalMerge {
    /// Logical IS the physical tensor. No merge step required beyond
    /// optional dequantization to `f32` and an optional squeeze of
    /// `padding` axes the v2 schema declares explicitly but the v1
    /// legacy dispatch does not understand.
    Direct {
        #[allow(dead_code)] // Reserved for future name-keyed decoding
        name: Option<String>,
        /// Physical tensor shape — used to bind the input tensor.
        shape: Vec<usize>,
        /// Axes (in the physical shape) to drop after dequant. Ordered
        /// descending so `remove_axis` calls can apply sequentially
        /// without index-shift bookkeeping.
        padding_axes: Vec<usize>,
        quant: Option<Quantization>,
    },
    /// Children share a common anchor axis and differ along the channel
    /// dimension (e.g. ARA-2 xy + wh). Dequantize each and concat along
    /// the channel axis.
    ChannelConcat {
        children: Vec<PhysicalBinding>,
        logical_shape: Vec<usize>,
        channel_axis: usize,
        /// Padding axis indices (descending) to squeeze from the
        /// merged tensor before emitting.
        padding_axes: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
struct PhysicalBinding {
    shape: Vec<usize>,
    quant: Option<Quantization>,
}

impl DecodeProgram {
    /// Build a decode program for a schema v2 document, if any logical
    /// output has physical children.
    ///
    /// Returns `Ok(None)` when the schema is flat (no children
    /// anywhere) — callers can then feed the input tensors through the
    /// legacy decode path directly. Returns an error if the schema
    /// uses unsupported features (per-channel quantization, missing
    /// dshape, DFL encoding on split boxes).
    ///
    /// Typeless logical outputs are skipped: they're carried in the
    /// schema as user-managed / auxiliary tensors and do not participate
    /// in decoder dispatch. The returned program emits one merged
    /// tensor per *typed* logical output in schema order — this keeps
    /// it aligned with [`SchemaV2::to_legacy_config_outputs`], which
    /// applies the same filter.
    pub fn try_from_schema(schema: &SchemaV2) -> DecoderResult<Option<Self>> {
        let needs_merge = schema
            .outputs
            .iter()
            .any(|l| l.type_.is_some() && !l.outputs.is_empty());
        if !needs_merge {
            return Ok(None);
        }
        let merges = schema
            .outputs
            .iter()
            .filter(|l| l.type_.is_some())
            .map(plan_logical)
            .collect::<DecoderResult<Vec<_>>>()?;
        Ok(Some(Self { merges }))
    }

    /// Run the program against the provided tensors, producing one
    /// merged `ArrayD<f32>` per logical output in schema order.
    ///
    /// Tensor binding: each `Direct` or physical child consumes one
    /// input from `inputs` by shape match. When multiple children share
    /// a shape (e.g. ARA-2 xy/wh), the first un-consumed tensor is used
    /// — callers must pass inputs in the declared schema-child order.
    pub fn execute(&self, inputs: &[&TensorDyn]) -> DecoderResult<Vec<ArrayD<f32>>> {
        let mut used: Vec<usize> = Vec::new();
        self.merges
            .iter()
            .map(|m| execute_merge(m, inputs, &mut used))
            .collect()
    }
}

fn plan_logical(logical: &LogicalOutput) -> DecoderResult<LogicalMerge> {
    if logical.outputs.is_empty() {
        let pad = padding_axes(&logical.dshape);
        return Ok(LogicalMerge::Direct {
            name: logical.name.clone(),
            shape: logical.shape.clone(),
            padding_axes: pad,
            quant: logical
                .quantization
                .as_ref()
                .map(schema_quant_to_runtime)
                .transpose()?,
        });
    }

    let first_has_stride = logical.outputs[0].stride.is_some();
    if first_has_stride {
        // Per-scale FPN splits are owned end-to-end by the `per_scale`
        // subsystem, which claims such schemas at builder time before the
        // merge program is built (see `DecoderBuilder::build_from_schema`).
        // Reaching here means a schema mixes a per-scale logical with
        // flat/channel-sub-split siblings such that `PerScalePlan` declined
        // it — a decomposition the merge path does not support.
        Err(DecoderError::NotSupported(format!(
            "logical `{}` has per-scale (strided) children but the schema was \
             not claimed by the per-scale subsystem; mixed per-scale + \
             channel-sub-split schemas are not supported by the merge path",
            logical.name.as_deref().unwrap_or("<anonymous>")
        )))
    } else {
        plan_channel_concat(logical)
    }
}

fn plan_channel_concat(logical: &LogicalOutput) -> DecoderResult<LogicalMerge> {
    let children = logical
        .outputs
        .iter()
        .map(physical_binding)
        .collect::<DecoderResult<Vec<_>>>()?;
    let channel_axis = channel_axis_in_logical(logical)?;
    // Padding axes (always 1) are squeezed out of the merged output so
    // the shape matches what the legacy decoder's `verify_yolo_*`
    // expects. Record their positions (descending) so the executor can
    // drop them deterministically.
    let pad = padding_axes(&logical.dshape);
    let (squeezed_shape, _) = squeeze_padding_dims(logical.shape.clone(), logical.dshape.clone());
    Ok(LogicalMerge::ChannelConcat {
        children,
        logical_shape: squeezed_shape,
        channel_axis,
        padding_axes: pad,
    })
}

fn physical_binding(p: &schema::PhysicalOutput) -> DecoderResult<PhysicalBinding> {
    let quant = p
        .quantization
        .as_ref()
        .map(schema_quant_to_runtime)
        .transpose()?;
    Ok(PhysicalBinding {
        shape: p.shape.clone(),
        quant,
    })
}

fn schema_quant_to_runtime(q: &schema::Quantization) -> DecoderResult<Quantization> {
    if q.is_per_channel() {
        return Err(DecoderError::NotSupported(format!(
            "per-channel quantization (axis {:?}, {} scales) is not yet \
             supported by the HAL merge path",
            q.axis,
            q.scale.len(),
        )));
    }
    Ok(Quantization::new(
        *q.scale.first().unwrap_or(&0.0),
        q.zero_point_at(0),
    ))
}

fn channel_axis_in_logical(logical: &LogicalOutput) -> DecoderResult<usize> {
    if !logical.dshape.is_empty() {
        for (i, (name, _)) in logical.dshape.iter().enumerate() {
            if matches!(
                name,
                DimName::BoxCoords
                    | DimName::NumFeatures
                    | DimName::NumClasses
                    | DimName::NumProtos
                    | DimName::NumAnchorsXFeatures
            ) {
                return Ok(i);
            }
        }
    }
    Err(DecoderError::InvalidConfig(format!(
        "logical `{}` has channel-sub-split children; `dshape` must name \
         a channel axis (box_coords, num_features, num_classes, num_protos)",
        logical.name.as_deref().unwrap_or("<anonymous>")
    )))
}

// =============================================================================
// Execution
// =============================================================================

fn execute_merge(
    merge: &LogicalMerge,
    inputs: &[&TensorDyn],
    used: &mut Vec<usize>,
) -> DecoderResult<ArrayD<f32>> {
    match merge {
        LogicalMerge::Direct {
            shape,
            padding_axes,
            quant,
            ..
        } => {
            let mut arr = find_and_dequantize(inputs, shape, *quant, used)?;
            for &ax in padding_axes {
                if ax < arr.ndim() && arr.shape()[ax] == 1 {
                    arr = arr.remove_axis(Axis(ax));
                }
            }
            Ok(arr)
        }
        LogicalMerge::ChannelConcat {
            children,
            logical_shape,
            channel_axis,
            padding_axes,
        } => execute_channel_concat(
            inputs,
            children,
            logical_shape,
            *channel_axis,
            padding_axes,
            used,
        ),
    }
}

/// Locate the first un-consumed input tensor with the given shape.
/// Consumed indices (already bound to an earlier child) are excluded,
/// which correctly handles channel sub-splits where siblings share a
/// shape (e.g. ARA-2 `boxes_xy` and `boxes_wh` at `[1, 2, 8400, 1]`).
/// The first match is recorded via `used`.
fn find_unused_tensor_by_shape<'a>(
    inputs: &'a [&'a TensorDyn],
    shape: &[usize],
    used: &mut Vec<usize>,
) -> DecoderResult<&'a TensorDyn> {
    for (i, t) in inputs.iter().enumerate() {
        if used.contains(&i) {
            continue;
        }
        if t.shape() == shape {
            used.push(i);
            return Ok(*t);
        }
    }
    Err(DecoderError::InvalidShape(format!(
        "no remaining input tensor matches shape {shape:?} (already \
         bound tensors are excluded; pass inputs in schema child order, \
         or use name-keyed decode once available)"
    )))
}

/// Find tensor by shape, dequantize to f32, and transpose to match the
/// expected shape if needed.
///
/// Tries exact shape match first. When that fails (common in GStreamer /
/// NNStreamer pipelines where the framework reports dimensions in
/// innermost-first order while the v2 schema uses standard row-major
/// shapes), falls back to element-count matching with axis permutation.
///
/// The permutation is derived by matching dimension sizes between the
/// tensor's natural shape and the expected shape. When the tensor has
/// fewer dimensions (e.g. trailing unit dims were stripped), it is
/// padded with trailing 1s before matching.
fn find_and_dequantize(
    inputs: &[&TensorDyn],
    expected_shape: &[usize],
    quant: Option<Quantization>,
    used: &mut Vec<usize>,
) -> DecoderResult<ArrayD<f32>> {
    // Fast path: exact shape match
    if let Ok(t) = find_unused_tensor_by_shape(inputs, expected_shape, used) {
        return tensor_to_f32(t, quant);
    }

    // Fallback: element-count match with axis permutation
    let expected_count: usize = expected_shape.iter().product();
    for (i, t) in inputs.iter().enumerate() {
        if used.contains(&i) {
            continue;
        }
        let count: usize = t.shape().iter().product();
        if count != expected_count {
            continue;
        }
        // Pad tensor shape with trailing 1s to match expected ndim
        let mut padded_shape = t.shape().to_vec();
        while padded_shape.len() < expected_shape.len() {
            padded_shape.push(1);
        }
        if let Some(perm) = find_axis_permutation(&padded_shape, expected_shape) {
            used.push(i);
            let arr = tensor_to_f32(t, quant)?;
            // Reshape to padded ndim (adds trailing unit dims, no data move)
            let mut arr = if arr.ndim() < expected_shape.len() {
                arr.into_shape_with_order(IxDyn(&padded_shape))
                    .map_err(DecoderError::NDArrayShape)?
            } else {
                arr
            };
            // Transpose axes to match expected shape. permuted_axes
            // reorders strides (zero-copy view), then as_standard_layout
            // produces a C-contiguous copy with data in the new order.
            arr = arr.permuted_axes(IxDyn(&perm));
            arr = arr.as_standard_layout().to_owned();
            debug_assert_eq!(arr.shape(), expected_shape);
            return Ok(arr);
        }
    }

    Err(DecoderError::InvalidShape(format!(
        "no remaining input tensor matches shape {expected_shape:?} \
         (tried exact match and element-count + permutation; already \
         bound tensors are excluded)"
    )))
}

/// Find the axis permutation that transforms `from` into `to`.
///
/// Returns `None` if no permutation exists (different multiset of
/// dimension sizes or different lengths). When repeated dimension
/// sizes occur (e.g. 160×160 spatial dims), they are matched in order
/// which preserves their relative axis positions.
fn find_axis_permutation(from: &[usize], to: &[usize]) -> Option<Vec<usize>> {
    if from.len() != to.len() {
        return None;
    }
    let n = from.len();
    let mut perm = vec![0usize; n];
    let mut bound = vec![false; n];
    for (i, &target_dim) in to.iter().enumerate() {
        let mut found = false;
        for (j, &source_dim) in from.iter().enumerate() {
            if !bound[j] && source_dim == target_dim {
                perm[i] = j;
                bound[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }
    }
    Some(perm)
}

/// Dequantize (if integer) or cast (if float) a TensorDyn to an owned
/// `ArrayD<f32>` matching the tensor's shape.
///
/// Schema-v2 merge always widens to f32 because logical outputs may
/// concatenate children of heterogeneous dtypes (e.g. an i8 physical child
/// with a f32 sibling). This is the one place in the decoder where f16 is
/// intentionally promoted rather than kept native.
fn tensor_to_f32(t: &TensorDyn, quant: Option<Quantization>) -> DecoderResult<ArrayD<f32>> {
    let shape = t.shape().to_vec();
    match t {
        TensorDyn::F16(tensor) => {
            let m = tensor
                .map()
                .map_err(|e| DecoderError::Internal(format!("tensor map: {e}")))?;
            // Vectorized f16 → f32 widening via the `half` crate.
            use half::slice::HalfFloatSliceExt;
            let total: usize = shape.iter().product();
            let mut out = vec![0.0_f32; total];
            m.as_slice().convert_to_f32_slice(&mut out);
            Ok(Array::from_shape_vec(IxDyn(&shape), out)?)
        }
        TensorDyn::F32(tensor) => {
            let m = tensor
                .map()
                .map_err(|e| DecoderError::Internal(format!("tensor map: {e}")))?;
            let view = ArrayViewD::from_shape(IxDyn(&shape), m.as_slice())?;
            Ok(view.to_owned())
        }
        TensorDyn::F64(tensor) => {
            let m = tensor
                .map()
                .map_err(|e| DecoderError::Internal(format!("tensor map: {e}")))?;
            let view = ArrayViewD::from_shape(IxDyn(&shape), m.as_slice())?;
            Ok(view.mapv(|v| v as f32))
        }
        TensorDyn::U8(_)
        | TensorDyn::I8(_)
        | TensorDyn::U16(_)
        | TensorDyn::I16(_)
        | TensorDyn::U32(_)
        | TensorDyn::I32(_) => dequantize_integer_tensor(t, quant, &shape),
        other => Err(DecoderError::NotSupported(format!(
            "merge: unsupported tensor dtype {:?}",
            other.dtype()
        ))),
    }
}

fn dequantize_integer_tensor(
    t: &TensorDyn,
    quant: Option<Quantization>,
    shape: &[usize],
) -> DecoderResult<ArrayD<f32>> {
    let quant = quant.unwrap_or(Quantization::new(1.0, 0));
    let total: usize = shape.iter().product();
    let mut out = vec![0.0_f32; total];
    macro_rules! dq {
        ($tensor:expr) => {{
            let m = $tensor
                .map()
                .map_err(|e| DecoderError::Internal(format!("tensor map: {e}")))?;
            dequantize_cpu_chunked(m.as_slice(), quant, &mut out);
        }};
    }
    match t {
        TensorDyn::U8(tensor) => dq!(tensor),
        TensorDyn::I8(tensor) => dq!(tensor),
        TensorDyn::U16(tensor) => dq!(tensor),
        TensorDyn::I16(tensor) => dq!(tensor),
        TensorDyn::U32(tensor) => dq!(tensor),
        TensorDyn::I32(tensor) => dq!(tensor),
        _ => unreachable!("dequantize_integer_tensor called on non-integer dtype"),
    }
    let arr = Array::from_shape_vec(IxDyn(shape), out)?;
    Ok(arr)
}

/// Concatenate children along a channel axis.
///
/// For ARA-2: children `boxes_xy` and `boxes_wh` each shape `[1, 2,
/// 8400, 1]`; logical shape `[1, 4, 8400, 1]`; `channel_axis = 1`.
fn execute_channel_concat(
    inputs: &[&TensorDyn],
    children: &[PhysicalBinding],
    logical_shape: &[usize],
    channel_axis: usize,
    padding_axes: &[usize],
    used: &mut Vec<usize>,
) -> DecoderResult<ArrayD<f32>> {
    let mut parts = Vec::with_capacity(children.len());
    for child in children {
        parts.push(find_and_dequantize(
            inputs,
            &child.shape,
            child.quant,
            used,
        )?);
    }
    let views: Vec<_> = parts.iter().map(|a| a.view()).collect();
    let mut merged =
        ndarray::concatenate(Axis(channel_axis), &views).map_err(DecoderError::NDArrayShape)?;
    // Drop padding axes (descending order, so earlier removes do not
    // shift later indices).
    for &ax in padding_axes {
        if ax < merged.ndim() && merged.shape()[ax] == 1 {
            merged = merged.remove_axis(Axis(ax));
        }
    }
    if merged.shape() != logical_shape {
        return Err(DecoderError::InvalidShape(format!(
            "channel-concat produced shape {:?} but logical expected {:?}",
            merged.shape(),
            logical_shape
        )));
    }
    Ok(merged)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::schema::{
        BoxEncoding, DType, DecoderKind, LogicalOutput, LogicalType, PhysicalOutput, PhysicalType,
        Quantization as SchemaQuant, SchemaV2, ScoreFormat,
    };
    use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

    fn make_u8_tensor(shape: &[usize], values: &[u8]) -> TensorDyn {
        let t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        let mut m = t.map().unwrap();
        let slice = m.as_mut_slice();
        slice[..values.len()].copy_from_slice(values);
        drop(m);
        TensorDyn::U8(t)
    }

    fn make_i16_tensor(shape: &[usize], values: &[i16]) -> TensorDyn {
        let t = Tensor::<i16>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        let mut m = t.map().unwrap();
        let slice = m.as_mut_slice();
        slice[..values.len()].copy_from_slice(values);
        drop(m);
        TensorDyn::I16(t)
    }

    fn make_f32_tensor(shape: &[usize], values: &[f32]) -> TensorDyn {
        let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        let mut m = t.map().unwrap();
        let slice = m.as_mut_slice();
        slice[..values.len()].copy_from_slice(values);
        drop(m);
        TensorDyn::F32(t)
    }

    fn make_f16_tensor(shape: &[usize], values: &[f32]) -> TensorDyn {
        let t = Tensor::<half::f16>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        let mut m = t.map().unwrap();
        let slice = m.as_mut_slice();
        for (dst, &src) in slice.iter_mut().zip(values.iter()) {
            *dst = half::f16::from_f32(src);
        }
        drop(m);
        TensorDyn::F16(t)
    }

    #[test]
    fn tensor_to_f32_widens_f16_natively() {
        // Direct check of the schema-v2 merge helper: f16 values must be
        // exactly widened to f32 via `half` crate conversion. Picks values
        // representable exactly in both f16 and f32 so the comparison is
        // bit-exact (no tolerance needed).
        let t = make_f16_tensor(&[2, 3], &[1.0, -2.0, 0.5, 0.25, -0.125, 4.0]);
        let arr = super::tensor_to_f32(&t, None).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        let flat: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(flat, vec![1.0, -2.0, 0.5, 0.25, -0.125, 4.0]);
    }

    fn per_tensor_q(scale: f32, zp: i32, dt: DType) -> SchemaQuant {
        SchemaQuant {
            scale: vec![scale],
            zero_point: Some(vec![zp]),
            axis: None,
            dtype: Some(dt),
        }
    }

    /// Typeless logical outputs are filtered from the DecodeProgram,
    /// keeping it aligned with `to_legacy_config_outputs` (which applies
    /// the same filter). Without this, `decode()` would pass N merged
    /// tensors to a decode path expecting N-k — corrupting downstream
    /// decoding when a user adds a custom output alongside a decoded
    /// YOLO model.
    #[test]
    fn typeless_logical_not_included_in_decode_program() {
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![
                // User-managed: no type, flat.
                LogicalOutput {
                    name: Some("user_custom".into()),
                    type_: None,
                    shape: vec![1, 32],
                    dshape: vec![],
                    decoder: None,
                    encoding: None,
                    score_format: None,
                    normalized: None,
                    anchors: None,
                    stride: None,
                    dtype: None,
                    quantization: None,
                    outputs: vec![],
                    activation_applied: None,
                    activation_required: None,
                },
                // Typed boxes with a single merged child (triggers program).
                LogicalOutput {
                    name: Some("boxes".into()),
                    type_: Some(LogicalType::Boxes),
                    shape: vec![1, 4, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 3),
                    ],
                    decoder: Some(DecoderKind::Ultralytics),
                    encoding: Some(BoxEncoding::Direct),
                    score_format: None,
                    normalized: Some(true),
                    anchors: None,
                    stride: None,
                    dtype: None,
                    quantization: None,
                    outputs: vec![PhysicalOutput {
                        name: "boxes_raw".into(),
                        type_: Some(PhysicalType::Boxes),
                        shape: vec![1, 4, 3],
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::BoxCoords, 4),
                            (DimName::NumBoxes, 3),
                        ],
                        dtype: DType::Float32,
                        quantization: None,
                        stride: None,
                        scale_index: None,
                        activation_applied: None,
                        activation_required: None,
                    }],
                    activation_applied: None,
                    activation_required: None,
                },
            ],
            ..Default::default()
        };

        let program = DecodeProgram::try_from_schema(&schema)
            .unwrap()
            .expect("typed boxes has children → program must be Some");

        // Legacy config filters typeless outputs; the program must
        // apply the identical filter so positional alignment holds.
        let legacy = schema.to_legacy_config_outputs().unwrap();
        assert_eq!(legacy.outputs.len(), 1);

        // Program emits one merged tensor (for the typed `boxes`), not
        // two (which would include the user_custom placeholder).
        let boxes_tensor = make_f32_tensor(&[1, 4, 3], &[1.0; 12]);
        let inputs: Vec<&TensorDyn> = vec![&boxes_tensor];
        let merged = program.execute(&inputs).unwrap();
        assert_eq!(
            merged.len(),
            1,
            "decode program must emit one tensor per typed logical, not \
             per schema-order logical (would otherwise misalign with \
             legacy ConfigOutputs passed to decode_float)"
        );
    }

    #[test]
    fn flat_schema_has_no_decode_program() {
        // Logical outputs with no children => no merge needed.
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![LogicalOutput {
                name: Some("boxes".into()),
                type_: Some(LogicalType::Boxes),
                shape: vec![1, 4, 8400],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::BoxCoords, 4),
                    (DimName::NumBoxes, 8400),
                ],
                decoder: Some(DecoderKind::Ultralytics),
                encoding: Some(BoxEncoding::Direct),
                score_format: None,
                normalized: Some(true),
                anchors: None,
                stride: None,
                dtype: Some(DType::Float32),
                quantization: None,
                outputs: vec![],
                activation_applied: None,
                activation_required: None,
            }],
            ..Default::default()
        };
        let program = DecodeProgram::try_from_schema(&schema).unwrap();
        assert!(program.is_none());
    }

    #[test]
    fn channel_concat_merges_xy_and_wh_to_logical_shape() {
        // ARA-2 style: boxes_xy + boxes_wh → [1, 4, 3]
        let boxes_logical = LogicalOutput {
            name: Some("boxes".into()),
            type_: Some(LogicalType::Boxes),
            shape: vec![1, 4, 3],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::BoxCoords, 4),
                (DimName::NumBoxes, 3),
            ],
            decoder: Some(DecoderKind::Ultralytics),
            encoding: Some(BoxEncoding::Direct),
            score_format: None,
            normalized: Some(true),
            anchors: None,
            stride: None,
            dtype: None,
            quantization: None,
            outputs: vec![
                PhysicalOutput {
                    name: "xy".into(),
                    type_: Some(PhysicalType::BoxesXy),
                    shape: vec![1, 2, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 2),
                        (DimName::NumBoxes, 3),
                    ],
                    dtype: DType::Int16,
                    quantization: Some(per_tensor_q(0.01, 0, DType::Int16)),
                    stride: None,
                    scale_index: None,
                    activation_applied: None,
                    activation_required: None,
                },
                PhysicalOutput {
                    name: "wh".into(),
                    type_: Some(PhysicalType::BoxesWh),
                    shape: vec![1, 2, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 2),
                        (DimName::NumBoxes, 3),
                    ],
                    dtype: DType::Int16,
                    quantization: Some(per_tensor_q(0.02, 0, DType::Int16)),
                    stride: None,
                    scale_index: None,
                    activation_applied: None,
                    activation_required: None,
                },
            ],
            activation_applied: None,
            activation_required: None,
        };
        // Two i16 tensors with the same shape: the merge path binds by
        // shape, so the first match is used for the first child. To keep
        // the test deterministic we give the two children distinct shapes
        // by using 3 vs 2 columns below. Update shapes for the test
        // only:
        let mut schema = SchemaV2::default();
        schema.outputs.push(LogicalOutput {
            shape: vec![1, 3, 3],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::BoxCoords, 3),
                (DimName::NumBoxes, 3),
            ],
            outputs: vec![
                PhysicalOutput {
                    name: "xy".into(),
                    type_: Some(PhysicalType::BoxesXy),
                    shape: vec![1, 1, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 1),
                        (DimName::NumBoxes, 3),
                    ],
                    dtype: DType::Int16,
                    quantization: Some(per_tensor_q(0.01, 0, DType::Int16)),
                    stride: None,
                    scale_index: None,
                    activation_applied: None,
                    activation_required: None,
                },
                PhysicalOutput {
                    name: "wh".into(),
                    type_: Some(PhysicalType::BoxesWh),
                    shape: vec![1, 2, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 2),
                        (DimName::NumBoxes, 3),
                    ],
                    dtype: DType::Int16,
                    quantization: Some(per_tensor_q(0.02, 0, DType::Int16)),
                    stride: None,
                    scale_index: None,
                    activation_applied: None,
                    activation_required: None,
                },
            ],
            ..boxes_logical
        });

        let program = DecodeProgram::try_from_schema(&schema).unwrap().unwrap();

        // Synthetic inputs: xy is [1, 1, 3] i16 with values [100, 200, 300]
        // wh is [1, 2, 3] i16 with values [10, 20, 30, 40, 50, 60]
        let xy = make_i16_tensor(&[1, 1, 3], &[100, 200, 300]);
        let wh = make_i16_tensor(&[1, 2, 3], &[10, 20, 30, 40, 50, 60]);
        let inputs: Vec<&TensorDyn> = vec![&xy, &wh];
        let merged = program.execute(&inputs).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].shape(), &[1, 3, 3]);
        // xy dequant: 0.01 * value → [1.0, 2.0, 3.0]
        // wh dequant: 0.02 * value → [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        // Merged along channel axis (1):
        //   channel 0: xy → [1.0, 2.0, 3.0]
        //   channel 1: wh[0] → [0.2, 0.4, 0.6]
        //   channel 2: wh[1] → [0.8, 1.0, 1.2]
        let arr = &merged[0];
        assert!((arr[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((arr[[0, 0, 2]] - 3.0).abs() < 1e-5);
        assert!((arr[[0, 1, 0]] - 0.2).abs() < 1e-5);
        assert!((arr[[0, 2, 2]] - 1.2).abs() < 1e-5);
    }

    #[test]
    fn direct_logical_with_float_tensor_pass_through() {
        // A flat logical (no children) with a float32 tensor.
        // try_from_schema returns None (no merge); we directly call the
        // Direct path via a single-element program.
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![
                LogicalOutput {
                    name: Some("boxes".into()),
                    type_: Some(LogicalType::Boxes),
                    shape: vec![1, 4, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 3),
                    ],
                    decoder: Some(DecoderKind::Ultralytics),
                    encoding: Some(BoxEncoding::Direct),
                    score_format: None,
                    normalized: Some(true),
                    anchors: None,
                    stride: None,
                    dtype: Some(DType::Float32),
                    quantization: None,
                    outputs: vec![],
                    activation_applied: None,
                    activation_required: None,
                },
                // Force at least one split to enable DecodeProgram
                LogicalOutput {
                    name: Some("scores".into()),
                    type_: Some(LogicalType::Scores),
                    shape: vec![1, 2, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 2),
                        (DimName::NumBoxes, 3),
                    ],
                    decoder: Some(DecoderKind::Ultralytics),
                    encoding: None,
                    score_format: Some(ScoreFormat::PerClass),
                    normalized: None,
                    anchors: None,
                    stride: None,
                    dtype: None,
                    quantization: None,
                    outputs: vec![
                        PhysicalOutput {
                            name: "s0".into(),
                            type_: Some(PhysicalType::Scores),
                            shape: vec![1, 1, 3],
                            dshape: vec![
                                (DimName::Batch, 1),
                                (DimName::NumClasses, 1),
                                (DimName::NumBoxes, 3),
                            ],
                            dtype: DType::Uint8,
                            quantization: Some(per_tensor_q(0.5, 0, DType::Uint8)),
                            stride: None,
                            scale_index: None,
                            activation_applied: None,
                            activation_required: None,
                        },
                        PhysicalOutput {
                            name: "s1".into(),
                            type_: Some(PhysicalType::Scores),
                            shape: vec![1, 1, 3],
                            dshape: vec![
                                (DimName::Batch, 1),
                                (DimName::NumClasses, 1),
                                (DimName::NumBoxes, 3),
                            ],
                            dtype: DType::Uint8,
                            quantization: Some(per_tensor_q(0.25, 0, DType::Uint8)),
                            stride: None,
                            scale_index: None,
                            activation_applied: None,
                            activation_required: None,
                        },
                    ],
                    activation_applied: None,
                    activation_required: None,
                },
            ],
            ..Default::default()
        };

        let program = DecodeProgram::try_from_schema(&schema).unwrap().unwrap();
        let boxes = make_f32_tensor(
            &[1, 4, 3],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        );
        let s0 = make_u8_tensor(&[1, 1, 3], &[2, 4, 6]);
        let s1 = make_u8_tensor(&[1, 1, 3], &[8, 16, 24]);
        let inputs: Vec<&TensorDyn> = vec![&boxes, &s0, &s1];
        let merged = program.execute(&inputs).unwrap();
        assert_eq!(merged.len(), 2);
        // Direct boxes: pass-through f32.
        assert!((merged[0][[0, 0, 0]] - 0.1).abs() < 1e-6);
        assert!((merged[0][[0, 3, 2]] - 1.2).abs() < 1e-6);
        // ChannelConcat scores:
        //   s0 dequant 0.5 → [1.0, 2.0, 3.0]
        //   s1 dequant 0.25 → [2.0, 4.0, 6.0]
        //   merged on axis=1: channel 0 from s0, channel 1 from s1
        assert!((merged[1][[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((merged[1][[0, 0, 2]] - 3.0).abs() < 1e-6);
        assert!((merged[1][[0, 1, 0]] - 2.0).abs() < 1e-6);
        assert!((merged[1][[0, 1, 2]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dequantize_affine_reference_values_match_validator() {
        // Validator-parity: `(q - zp) × scale` with scale=0.130,
        // zp=70 produces (0 - 70)×0.130 = -9.10, (70 - 70)×0.130 = 0,
        // (255 - 70)×0.130 = 24.05. Exercised end-to-end through the
        // `Direct` merge path (non-DFL) so the assertion holds against
        // the same dequant kernel the DFL path uses.
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![
                LogicalOutput {
                    name: Some("scores".into()),
                    type_: Some(LogicalType::Scores),
                    shape: vec![1, 1, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 1),
                        (DimName::NumBoxes, 3),
                    ],
                    decoder: Some(DecoderKind::Ultralytics),
                    encoding: None,
                    score_format: Some(ScoreFormat::PerClass),
                    normalized: None,
                    anchors: None,
                    stride: None,
                    dtype: Some(DType::Uint8),
                    quantization: Some(per_tensor_q(0.130, 70, DType::Uint8)),
                    outputs: vec![],
                    activation_applied: None,
                    activation_required: None,
                },
                LogicalOutput {
                    // Second logical forces a DecodeProgram so the Direct
                    // branch actually runs (try_from_schema short-circuits
                    // when nothing needs merging).
                    name: Some("boxes".into()),
                    type_: Some(LogicalType::Boxes),
                    shape: vec![1, 4, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 3),
                    ],
                    decoder: Some(DecoderKind::Ultralytics),
                    encoding: Some(BoxEncoding::Direct),
                    score_format: None,
                    normalized: Some(true),
                    anchors: None,
                    stride: None,
                    dtype: None,
                    quantization: None,
                    outputs: vec![
                        PhysicalOutput {
                            name: "b0".into(),
                            type_: Some(PhysicalType::BoxesXy),
                            shape: vec![1, 2, 3],
                            dshape: vec![
                                (DimName::Batch, 1),
                                (DimName::BoxCoords, 2),
                                (DimName::NumBoxes, 3),
                            ],
                            dtype: DType::Float32,
                            quantization: None,
                            stride: None,
                            scale_index: None,
                            activation_applied: None,
                            activation_required: None,
                        },
                        PhysicalOutput {
                            name: "b1".into(),
                            type_: Some(PhysicalType::BoxesWh),
                            shape: vec![1, 2, 3],
                            dshape: vec![
                                (DimName::Batch, 1),
                                (DimName::BoxCoords, 2),
                                (DimName::NumBoxes, 3),
                            ],
                            dtype: DType::Float32,
                            quantization: None,
                            stride: None,
                            scale_index: None,
                            activation_applied: None,
                            activation_required: None,
                        },
                    ],
                    activation_applied: None,
                    activation_required: None,
                },
            ],
            ..Default::default()
        };
        let program = DecodeProgram::try_from_schema(&schema).unwrap().unwrap();
        let scores = make_u8_tensor(&[1, 1, 3], &[0, 70, 255]);
        let xy = make_f32_tensor(&[1, 2, 3], &[0.0f32; 6]);
        let wh = make_f32_tensor(&[1, 2, 3], &[0.0f32; 6]);
        let inputs: Vec<&TensorDyn> = vec![&scores, &xy, &wh];
        let merged = program.execute(&inputs).unwrap();
        let scores_out = &merged[0];
        assert!(
            (scores_out[[0, 0, 0]] - (-9.10)).abs() < 1e-4,
            "{}",
            scores_out[[0, 0, 0]]
        );
        assert!(
            (scores_out[[0, 0, 1]] - 0.00).abs() < 1e-4,
            "{}",
            scores_out[[0, 0, 1]]
        );
        assert!(
            (scores_out[[0, 0, 2]] - 24.05).abs() < 1e-3,
            "{}",
            scores_out[[0, 0, 2]]
        );
    }

    // ------------------------------------------------------------------
    // find_and_dequantize: permutation-aware fallback tests
    // ------------------------------------------------------------------

    #[test]
    fn find_and_dequantize_exact_match_preferred() {
        // When the tensor shape matches exactly, no permutation is applied.
        let t = make_f32_tensor(&[1, 3, 4], &[1.0; 12]);
        let inputs: Vec<&TensorDyn> = vec![&t];
        let mut used = Vec::new();
        let arr = find_and_dequantize(&inputs, &[1, 3, 4], None, &mut used).unwrap();
        assert_eq!(arr.shape(), &[1, 3, 4]);
        assert_eq!(used, vec![0]);
    }

    #[test]
    fn find_and_dequantize_permuted_nchw_to_nhwc() {
        // Tensor is NCHW [1, 3, 2, 2] but expected is NHWC [1, 2, 2, 3].
        // The fallback should detect the axis permutation and transpose.
        //
        // Data in C-contiguous NCHW order (channel-major):
        //   C0: [[1, 2], [3, 4]]
        //   C1: [[5, 6], [7, 8]]
        //   C2: [[9, 10], [11, 12]]
        let t = make_f32_tensor(
            &[1, 3, 2, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let inputs: Vec<&TensorDyn> = vec![&t];
        let mut used = Vec::new();
        let arr = find_and_dequantize(&inputs, &[1, 2, 2, 3], None, &mut used).unwrap();
        assert_eq!(arr.shape(), &[1, 2, 2, 3]);
        // After NCHW→NHWC, pixel (0,0) should have channels [1, 5, 9],
        // pixel (0,1) should be [2, 6, 10], etc.
        assert_eq!(arr[[0, 0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 0, 1]], 5.0);
        assert_eq!(arr[[0, 0, 0, 2]], 9.0);
        assert_eq!(arr[[0, 0, 1, 0]], 2.0);
        assert_eq!(arr[[0, 0, 1, 1]], 6.0);
        assert_eq!(arr[[0, 0, 1, 2]], 10.0);
        assert_eq!(arr[[0, 1, 0, 0]], 3.0);
        assert_eq!(arr[[0, 1, 0, 1]], 7.0);
        assert_eq!(arr[[0, 1, 0, 2]], 11.0);
    }

    #[test]
    fn find_and_dequantize_stripped_trailing_unit_dim() {
        // Tensor reported as [1, 6] (trailing unit dim stripped) but
        // expected shape is [1, 6, 1]. The fallback should pad with
        // trailing 1s and find an identity permutation.
        let t = make_f32_tensor(&[1, 6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let inputs: Vec<&TensorDyn> = vec![&t];
        let mut used = Vec::new();
        let arr = find_and_dequantize(&inputs, &[1, 6, 1], None, &mut used).unwrap();
        assert_eq!(arr.shape(), &[1, 6, 1]);
        assert_eq!(arr[[0, 0, 0]], 10.0);
        assert_eq!(arr[[0, 5, 0]], 60.0);
    }

    #[test]
    fn find_and_dequantize_skips_already_used() {
        // Two tensors with same element count; the first is already
        // consumed so the fallback must select the second.
        let t0 = make_f32_tensor(&[2, 3], &[0.0; 6]);
        let t1 = make_f32_tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let inputs: Vec<&TensorDyn> = vec![&t0, &t1];
        let mut used = vec![0]; // t0 already consumed
        let arr = find_and_dequantize(&inputs, &[2, 3], None, &mut used).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert!(used.contains(&1));
        // t1 is [3, 2] permuted to [2, 3]: rows of t1 become columns
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 3.0);
        assert_eq!(arr[[0, 2]], 5.0);
        assert_eq!(arr[[1, 0]], 2.0);
        assert_eq!(arr[[1, 1]], 4.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn find_and_dequantize_no_match_returns_error() {
        // Element count doesn't match any candidate.
        let t = make_f32_tensor(&[2, 5], &[0.0; 10]);
        let inputs: Vec<&TensorDyn> = vec![&t];
        let mut used = Vec::new();
        let result = find_and_dequantize(&inputs, &[3, 4], None, &mut used);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // find_axis_permutation unit tests
    // ------------------------------------------------------------------

    #[test]
    fn find_axis_permutation_identity() {
        let perm = find_axis_permutation(&[1, 3, 4], &[1, 3, 4]);
        assert_eq!(perm, Some(vec![0, 1, 2]));
    }

    #[test]
    fn find_axis_permutation_nchw_to_nhwc() {
        // NCHW [1, 3, 2, 2] → NHWC [1, 2, 2, 3]
        // Perm should map: out[0]=in[0], out[1]=in[2], out[2]=in[3], out[3]=in[1]
        let perm = find_axis_permutation(&[1, 3, 2, 2], &[1, 2, 2, 3]);
        assert_eq!(perm, Some(vec![0, 2, 3, 1]));
    }

    #[test]
    fn find_axis_permutation_no_match() {
        // Different dimension multisets
        assert_eq!(find_axis_permutation(&[1, 3, 4], &[1, 4, 5]), None);
    }

    #[test]
    fn find_axis_permutation_different_lengths() {
        assert_eq!(find_axis_permutation(&[1, 3], &[1, 3, 1]), None);
    }
}
