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
//! | [`LogicalMerge::PerScale`] | children carry `stride` (e.g. Hailo FPN) | sort by stride ascending, dequantize, flatten H×W, concat along spatial axis, reshape to logical shape |
//!
//! # Not yet supported
//!
//! - **DFL decoding** — per-scale children with `encoding: dfl` (e.g.
//!   Hailo YOLOv8 boxes) merge to `(1, reg_max×4, total_anchors)`
//!   which still carries the DFL distribution. Decoding that to
//!   `(1, 4, total_anchors)` xcycwh pixel coords requires a per-level
//!   softmax + weighted sum + dist2bbox kernel. Tracked separately.
//! - **Per-channel quantization** on split children — the HAL
//!   currently only consumes per-tensor scalar `(scale, zero_point)`.
//! - **Activation flags on floats** — when a child carries
//!   `activation_required: sigmoid` the HAL does not yet apply it
//!   during merge. For split ARA-2 / Hailo typical usage the NPU
//!   applies sigmoid on-chip (`activation_applied`), so this is not
//!   blocking.

use ndarray::{Array, Array3, ArrayD, ArrayViewD, Axis, IxDyn};

use crate::configs::DimName;
use crate::schema::{
    self, padding_axes, squeeze_padding_dims, Activation, LogicalOutput, LogicalType, SchemaV2,
    Stride,
};
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
    /// Children share a common feature axis and differ in spatial
    /// extent. Concatenate along the logical anchor / box axis after
    /// flattening each child's H×W.
    PerScale {
        children: Vec<PhysicalBinding>,
        logical_shape: Vec<usize>,
        feature_axis_logical: usize,
        box_axis_logical: usize,
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
    name: String,
    shape: Vec<usize>,
    dshape: Vec<(DimName, usize)>,
    quant: Option<Quantization>,
    stride: Option<Stride>,
    #[allow(dead_code)] // applied by the NPU; currently informational
    activation_applied: Option<Activation>,
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
    pub fn try_from_schema(schema: &SchemaV2) -> DecoderResult<Option<Self>> {
        let needs_merge = schema.outputs.iter().any(|l| !l.outputs.is_empty());
        if !needs_merge {
            return Ok(None);
        }
        let merges = schema
            .outputs
            .iter()
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

    if logical.type_ == LogicalType::Boxes && logical.encoding == Some(schema::BoxEncoding::Dfl) {
        return Err(DecoderError::NotSupported(format!(
            "logical `{}` has encoding=dfl with physical children; \
             DFL decode is not yet implemented in the HAL merge path",
            logical.name.as_deref().unwrap_or("<anonymous>")
        )));
    }

    let first_has_stride = logical.outputs[0].stride.is_some();
    if first_has_stride {
        plan_per_scale(logical)
    } else {
        plan_channel_concat(logical)
    }
}

fn plan_per_scale(logical: &LogicalOutput) -> DecoderResult<LogicalMerge> {
    let mut children = logical
        .outputs
        .iter()
        .map(physical_binding)
        .collect::<DecoderResult<Vec<_>>>()?;
    children.sort_by_key(|c| c.stride.map(|s| s.x()).unwrap_or(0));

    // Identify feature and box axes in the logical dshape. If dshape is
    // absent we cannot merge unambiguously — fall back to heuristics
    // based on common YOLO conventions (batch, features, boxes).
    let (feature_axis_logical, box_axis_logical) = logical_per_scale_axes(logical)?;
    Ok(LogicalMerge::PerScale {
        children,
        logical_shape: logical.shape.clone(),
        feature_axis_logical,
        box_axis_logical,
    })
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
        name: p.name.clone(),
        shape: p.shape.clone(),
        dshape: p.dshape.clone(),
        quant,
        stride: p.stride,
        activation_applied: p.activation_applied,
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

fn logical_per_scale_axes(logical: &LogicalOutput) -> DecoderResult<(usize, usize)> {
    // Require a dshape so we can name axes unambiguously.
    if logical.dshape.is_empty() {
        return Err(DecoderError::InvalidConfig(format!(
            "logical `{}` has per-scale children but no `dshape`; cannot \
             infer feature / box axes for merge",
            logical.name.as_deref().unwrap_or("<anonymous>")
        )));
    }
    let feature = logical.dshape.iter().position(|(n, _)| {
        matches!(
            n,
            DimName::NumFeatures
                | DimName::NumClasses
                | DimName::NumProtos
                | DimName::BoxCoords
                | DimName::NumAnchorsXFeatures
        )
    });
    let boxes = logical
        .dshape
        .iter()
        .position(|(n, _)| matches!(n, DimName::NumBoxes));
    match (feature, boxes) {
        (Some(f), Some(b)) => Ok((f, b)),
        _ => Err(DecoderError::InvalidConfig(format!(
            "logical `{}` dshape {:?} must name both a feature dim and `num_boxes`",
            logical.name.as_deref().unwrap_or("<anonymous>"),
            logical.dshape,
        ))),
    }
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
            let t = find_unused_tensor_by_shape(inputs, shape, used)?;
            let mut arr = tensor_to_f32(t, *quant)?;
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
        LogicalMerge::PerScale {
            children,
            logical_shape,
            feature_axis_logical,
            box_axis_logical,
        } => execute_per_scale(
            inputs,
            children,
            logical_shape,
            *feature_axis_logical,
            *box_axis_logical,
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

/// Dequantize (if integer) or cast (if float) a TensorDyn to an owned
/// `ArrayD<f32>` matching the tensor's shape.
fn tensor_to_f32(t: &TensorDyn, quant: Option<Quantization>) -> DecoderResult<ArrayD<f32>> {
    let shape = t.shape().to_vec();
    match t {
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
        let t = find_unused_tensor_by_shape(inputs, &child.shape, used)?;
        parts.push(tensor_to_f32(t, child.quant)?);
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

/// Merge per-scale children into a `(batch, features, total_anchors)`
/// logical tensor.
///
/// For each child:
/// 1. Resolve batch / height / width / feature axes from its `dshape`.
/// 2. Dequantize to `f32`.
/// 3. Reorder to `(batch, features, H*W)` regardless of NHWC/NCHW.
///
/// Concatenate parts along the anchor axis (stride-ascending order was
/// set at plan time), then reshape to `logical_shape`.
fn execute_per_scale(
    inputs: &[&TensorDyn],
    children: &[PhysicalBinding],
    logical_shape: &[usize],
    feature_axis_logical: usize,
    box_axis_logical: usize,
    used: &mut Vec<usize>,
) -> DecoderResult<ArrayD<f32>> {
    if children.is_empty() {
        return Err(DecoderError::InvalidConfig(
            "per-scale merge with zero children".into(),
        ));
    }

    let mut per_scale_parts: Vec<Array3<f32>> = Vec::with_capacity(children.len());
    let mut feature_count: Option<usize> = None;
    let mut batch: Option<usize> = None;
    for child in children {
        let t = find_unused_tensor_by_shape(inputs, &child.shape, used)?;
        let arr = tensor_to_f32(t, child.quant)?;
        let (b, features, part) = child_to_batch_feature_spatial(arr, child)?;
        match batch {
            None => batch = Some(b),
            Some(prev) if prev != b => {
                return Err(DecoderError::InvalidShape(format!(
                    "per-scale children have inconsistent batch: {prev} vs {b}"
                )));
            }
            _ => {}
        }
        match feature_count {
            None => feature_count = Some(features),
            Some(prev) if prev != features => {
                return Err(DecoderError::InvalidShape(format!(
                    "per-scale children have inconsistent feature count: {prev} vs {features}"
                )));
            }
            _ => {}
        }
        per_scale_parts.push(part);
    }

    let views: Vec<_> = per_scale_parts.iter().map(|a| a.view()).collect();
    let merged = ndarray::concatenate(Axis(2), &views).map_err(DecoderError::NDArrayShape)?;

    // merged is (batch, features, total_anchors). Reshape to logical shape
    // by moving features and anchors to their logical positions.
    reshape_to_logical(
        merged.into_dyn(),
        logical_shape,
        feature_axis_logical,
        box_axis_logical,
    )
}

/// Permute a physical child's dequantized array to a canonical
/// `(batch, features, H*W)` layout regardless of whether the child was
/// declared NCHW or NHWC.
///
/// Returns the materialised `Array3<f32>` along with the batch and
/// feature dimensions for cross-child consistency checks.
fn child_to_batch_feature_spatial(
    arr: ArrayD<f32>,
    child: &PhysicalBinding,
) -> DecoderResult<(usize, usize, Array3<f32>)> {
    if child.dshape.is_empty() {
        return Err(DecoderError::InvalidConfig(format!(
            "per-scale child `{}` must declare `dshape` for layout \
             disambiguation (NCHW vs NHWC)",
            child.name
        )));
    }
    let shape = arr.shape().to_vec();
    if shape.len() != child.dshape.len() {
        return Err(DecoderError::InvalidShape(format!(
            "per-scale child `{}` shape rank {} does not match dshape rank {}",
            child.name,
            shape.len(),
            child.dshape.len()
        )));
    }

    let mut batch = 1usize;
    let mut height = 1usize;
    let mut width = 1usize;
    let mut features = 1usize;
    for (i, (name, _)) in child.dshape.iter().enumerate() {
        let size = shape[i];
        match name {
            DimName::Batch => batch = size,
            DimName::Height => height = size,
            DimName::Width => width = size,
            DimName::NumFeatures
            | DimName::NumClasses
            | DimName::NumProtos
            | DimName::BoxCoords
            | DimName::NumAnchorsXFeatures => features = size,
            DimName::NumBoxes | DimName::Padding => {}
        }
    }

    let b_axis = axis_index(&child.dshape, &[DimName::Batch]).ok_or_else(|| {
        DecoderError::InvalidConfig(format!(
            "per-scale child `{}` dshape {:?} lacks a `batch` axis",
            child.name, child.dshape
        ))
    })?;
    let f_axis = axis_index(
        &child.dshape,
        &[
            DimName::NumFeatures,
            DimName::NumClasses,
            DimName::NumProtos,
            DimName::BoxCoords,
            DimName::NumAnchorsXFeatures,
        ],
    )
    .ok_or_else(|| {
        DecoderError::InvalidConfig(format!(
            "per-scale child `{}` dshape {:?} lacks a feature axis",
            child.name, child.dshape
        ))
    })?;
    let h_axis = axis_index(&child.dshape, &[DimName::Height]);
    let w_axis = axis_index(&child.dshape, &[DimName::Width]);

    // Permute to canonical order: batch, feature, then spatial (H, W)
    // when present. Remaining axes are appended so the permutation is
    // a valid permutation of 0..rank.
    let mut perm: Vec<usize> = vec![b_axis, f_axis];
    if let Some(h) = h_axis {
        perm.push(h);
    }
    if let Some(w) = w_axis {
        perm.push(w);
    }
    for i in 0..child.dshape.len() {
        if !perm.contains(&i) {
            perm.push(i);
        }
    }
    debug_assert_eq!(perm.len(), child.dshape.len());

    // Permute then materialise (contiguous (batch, feature, H, W, ...)).
    let permuted = arr.permuted_axes(IxDyn(&perm));
    let contiguous = permuted
        .as_standard_layout()
        .to_owned()
        .into_dimensionality::<IxDyn>()
        .map_err(DecoderError::NDArrayShape)?;

    let spatial = height * width;
    // Reshape into (batch, feature, spatial) Array3. This is a pure
    // view reshape of the contiguous permuted data.
    let reshaped = contiguous
        .into_shape_with_order(ndarray::Ix3(batch, features, spatial))
        .map_err(|e| {
            DecoderError::InvalidShape(format!(
                "per-scale: failed to reshape permuted child `{}` to \
                 (batch,features,spatial)=({batch},{features},{spatial}): {e}",
                child.name
            ))
        })?;
    Ok((batch, features, reshaped))
}

fn axis_index(dshape: &[(DimName, usize)], any_of: &[DimName]) -> Option<usize> {
    dshape.iter().position(|(n, _)| any_of.contains(n))
}

/// Reshape a `(batch, features, anchors)` tensor into the logical shape
/// declared by the schema, moving features/anchors to their logical
/// positions.
fn reshape_to_logical(
    merged: ArrayD<f32>,
    logical_shape: &[usize],
    feature_axis_logical: usize,
    box_axis_logical: usize,
) -> DecoderResult<ArrayD<f32>> {
    // Common case: logical is [batch, features, boxes] in the same order
    // as our intermediate representation — this is the YOLO convention.
    if logical_shape.len() == 3
        && feature_axis_logical == 1
        && box_axis_logical == 2
        && merged.shape() == logical_shape
    {
        return Ok(merged);
    }
    // General case: reshape if the element count matches, and transpose
    // batch/feature/boxes into the logical order.
    let total: usize = logical_shape.iter().product();
    if merged.len() != total {
        return Err(DecoderError::InvalidShape(format!(
            "merged shape {:?} has {} elements; logical shape {:?} \
             expects {} elements",
            merged.shape(),
            merged.len(),
            logical_shape,
            total
        )));
    }

    // Build a permutation that takes (batch=0, features=1, boxes=2) to
    // (batch_pos, feature_pos, box_pos) inside logical_shape. We assume
    // batch is at axis 0 in logical_shape unless otherwise declared.
    let batch_pos_logical = (0..logical_shape.len())
        .find(|&i| i != feature_axis_logical && i != box_axis_logical)
        .unwrap_or(0);
    let mut target = vec![0usize; logical_shape.len()];
    target[batch_pos_logical] = 0; // batch
    target[feature_axis_logical] = 1;
    target[box_axis_logical] = 2;

    // If logical has more than 3 dims (padding etc.), put remaining
    // logical dims past 3. We don't currently support padding dims in
    // per-scale merges — reject if present.
    if logical_shape.len() != 3 {
        return Err(DecoderError::NotSupported(format!(
            "per-scale merge into logical rank {} is not yet supported \
             (only rank-3 [batch, features, boxes] today)",
            logical_shape.len()
        )));
    }

    let inv_perm: Vec<usize> = (0..3)
        .map(|src| target.iter().position(|&t| t == src).unwrap())
        .collect();
    let out = merged.permuted_axes(inv_perm);
    Ok(out.as_standard_layout().to_owned())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::schema::{
        BoxEncoding, DType, DecoderKind, LogicalOutput, LogicalType, PhysicalOutput, PhysicalType,
        Quantization as SchemaQuant, SchemaV2, ScoreFormat, Stride as SchemaStride,
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

    fn per_tensor_q(scale: f32, zp: i32, dt: DType) -> SchemaQuant {
        SchemaQuant {
            scale: vec![scale],
            zero_point: Some(vec![zp]),
            axis: None,
            dtype: Some(dt),
        }
    }

    #[test]
    fn flat_schema_has_no_decode_program() {
        // Logical outputs with no children => no merge needed.
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![LogicalOutput {
                name: Some("boxes".into()),
                type_: LogicalType::Boxes,
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
            type_: LogicalType::Boxes,
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
                    type_: PhysicalType::BoxesXy,
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
                    type_: PhysicalType::BoxesWh,
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
                    type_: PhysicalType::BoxesXy,
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
                    type_: PhysicalType::BoxesWh,
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
    fn per_scale_merge_nhwc_to_nchw() {
        // Boxes split per-scale, NHWC children → (1, 4, total) logical NCHW.
        // Strides 8 + 16: spatial sizes 4 + 1 = 5 anchors, 4 features each.
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![LogicalOutput {
                name: Some("boxes".into()),
                type_: LogicalType::Boxes,
                shape: vec![1, 4, 5],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 4),
                    (DimName::NumBoxes, 5),
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
                        type_: PhysicalType::Boxes,
                        // NHWC [1, 2, 2, 4]: 4 anchors at stride 8
                        shape: vec![1, 2, 2, 4],
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 2),
                            (DimName::Width, 2),
                            (DimName::NumFeatures, 4),
                        ],
                        dtype: DType::Uint8,
                        quantization: Some(per_tensor_q(1.0, 0, DType::Uint8)),
                        stride: Some(SchemaStride::Square(8)),
                        scale_index: Some(0),
                        activation_applied: None,
                        activation_required: None,
                    },
                    PhysicalOutput {
                        name: "b1".into(),
                        type_: PhysicalType::Boxes,
                        // NHWC [1, 1, 1, 4]: 1 anchor at stride 16
                        shape: vec![1, 1, 1, 4],
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 1),
                            (DimName::Width, 1),
                            (DimName::NumFeatures, 4),
                        ],
                        dtype: DType::Uint8,
                        quantization: Some(per_tensor_q(1.0, 0, DType::Uint8)),
                        stride: Some(SchemaStride::Square(16)),
                        scale_index: Some(1),
                        activation_applied: None,
                        activation_required: None,
                    },
                ],
            }],
            ..Default::default()
        };

        let program = DecodeProgram::try_from_schema(&schema).unwrap().unwrap();

        // b0: NHWC [1, 2, 2, 4]. Anchor-major, feature-minor.
        //   Values by (h, w, c):
        //     (0,0,*) = [1, 2, 3, 4]
        //     (0,1,*) = [5, 6, 7, 8]
        //     (1,0,*) = [9, 10, 11, 12]
        //     (1,1,*) = [13, 14, 15, 16]
        // b1: NHWC [1, 1, 1, 4]. Values = [100, 101, 102, 103]
        let b0 = make_u8_tensor(
            &[1, 2, 2, 4],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        let b1 = make_u8_tensor(&[1, 1, 1, 4], &[100, 101, 102, 103]);
        let inputs: Vec<&TensorDyn> = vec![&b0, &b1];
        let merged = program.execute(&inputs).unwrap();

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].shape(), &[1, 4, 5]);
        let arr = &merged[0];
        // Anchors 0..4 are from stride-8 (4 anchors). Anchor 4 is from stride-16.
        // Canonical per-scale ordering: row-major over H then W, so (0,0),
        // (0,1), (1,0), (1,1) for stride 8.
        // Feature 0 across all anchors = [1, 5, 9, 13, 100]
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 1]], 5.0);
        assert_eq!(arr[[0, 0, 2]], 9.0);
        assert_eq!(arr[[0, 0, 3]], 13.0);
        assert_eq!(arr[[0, 0, 4]], 100.0);
        // Feature 3 across all anchors = [4, 8, 12, 16, 103]
        assert_eq!(arr[[0, 3, 0]], 4.0);
        assert_eq!(arr[[0, 3, 4]], 103.0);
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
                    type_: LogicalType::Boxes,
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
                },
                // Force at least one split to enable DecodeProgram
                LogicalOutput {
                    name: Some("scores".into()),
                    type_: LogicalType::Scores,
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
                            type_: PhysicalType::Scores,
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
                            type_: PhysicalType::Scores,
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
    fn rejects_dfl_split() {
        let schema = SchemaV2 {
            schema_version: 2,
            outputs: vec![LogicalOutput {
                name: Some("boxes".into()),
                type_: LogicalType::Boxes,
                shape: vec![1, 64, 8400],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 64),
                    (DimName::NumBoxes, 8400),
                ],
                decoder: Some(DecoderKind::Ultralytics),
                encoding: Some(BoxEncoding::Dfl),
                score_format: None,
                normalized: Some(true),
                anchors: None,
                stride: None,
                dtype: None,
                quantization: None,
                outputs: vec![PhysicalOutput {
                    name: "b0".into(),
                    type_: PhysicalType::Boxes,
                    shape: vec![1, 80, 80, 64],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 80),
                        (DimName::Width, 80),
                        (DimName::NumFeatures, 64),
                    ],
                    dtype: DType::Uint8,
                    quantization: Some(per_tensor_q(0.01, 128, DType::Uint8)),
                    stride: Some(SchemaStride::Square(8)),
                    scale_index: Some(0),
                    activation_applied: None,
                    activation_required: None,
                }],
            }],
            ..Default::default()
        };
        let err = DecodeProgram::try_from_schema(&schema).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("DFL"), "got: {msg}");
    }
}
