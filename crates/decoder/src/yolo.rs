// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use ndarray::{
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s, Array2, Array3, ArrayView1, ArrayView2, ArrayView3,
};
use num_traits::{AsPrimitive, Float, PrimInt, Signed};

use crate::{
    byte::{
        nms_class_aware_int, nms_extra_class_aware_int, nms_extra_int, nms_int,
        postprocess_boxes_index_quant, postprocess_boxes_quant, quantize_score_threshold,
    },
    configs::Nms,
    dequant_detect_box,
    float::{
        nms_class_aware_float, nms_extra_class_aware_float, nms_extra_float, nms_float,
        postprocess_boxes_float, postprocess_boxes_index_float,
    },
    BBoxTypeTrait, BoundingBox, DetectBox, DetectBoxQuantized, ProtoData, ProtoLayout,
    Quantization, Segmentation, XYWH, XYXY,
};

/// Maximum number of above-threshold candidates fed to NMS.
///
/// At very low score thresholds (e.g., t=0.01 on YOLOv8 with
/// 8400 anchors × 80 classes), the number of survivors approaches
/// the full 672 000-entry score grid. NMS is O(n²) and the
/// downstream mask matmul runs once per survivor, so an
/// unbounded set produces minutes-per-frame decode times.
///
/// `MAX_NMS_CANDIDATES` matches the Ultralytics `max_nms` default
/// and is applied as a top-K-by-score truncation immediately
/// before NMS. Values above the cap are silently dropped — at the
/// score thresholds where the cap activates the bottom of the
/// candidate list is dominated by noise that NMS would discard
/// anyway.
pub const MAX_NMS_CANDIDATES: usize = 30_000;

/// Truncate `boxes` to the highest-scoring `top_k` entries in-place when the
/// input exceeds the cap. Uses partial sort (O(N)) via `select_nth_unstable_by`
/// to avoid full O(N log N) sort. No-op when input length ≤ `top_k`.
fn truncate_to_top_k_by_score<E: Send>(boxes: &mut Vec<(DetectBox, E)>, top_k: usize) {
    if boxes.len() > top_k {
        boxes.select_nth_unstable_by(top_k, |a, b| b.0.score.total_cmp(&a.0.score));
        boxes.truncate(top_k);
    }
}

/// Quantized counterpart of [`truncate_to_top_k_by_score`]. Sorts on
/// the raw quantized score (which preserves order under monotonic
/// dequantization). Uses partial sort (O(N)) via `select_nth_unstable_by`.
fn truncate_to_top_k_by_score_quant<S: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send>(
    boxes: &mut Vec<(DetectBoxQuantized<S>, E)>,
    top_k: usize,
) {
    if boxes.len() > top_k {
        boxes.select_nth_unstable_by(top_k, |a, b| b.0.score.cmp(&a.0.score));
        boxes.truncate(top_k);
    }
}

/// Dispatches to the appropriate NMS function based on mode for float boxes.
fn dispatch_nms_float(nms: Option<Nms>, iou: f32, boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    match nms {
        Some(Nms::ClassAgnostic) => nms_float(iou, boxes),
        Some(Nms::ClassAware) => nms_class_aware_float(iou, boxes),
        None => boxes, // bypass NMS
    }
}

/// Dispatches to the appropriate NMS function based on mode for float boxes
/// with extra data.
pub(super) fn dispatch_nms_extra_float<E: Send + Sync>(
    nms: Option<Nms>,
    iou: f32,
    boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    match nms {
        Some(Nms::ClassAgnostic) => nms_extra_float(iou, boxes),
        Some(Nms::ClassAware) => nms_extra_class_aware_float(iou, boxes),
        None => boxes, // bypass NMS
    }
}

/// Dispatches to the appropriate NMS function based on mode for quantized
/// boxes.
fn dispatch_nms_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    nms: Option<Nms>,
    iou: f32,
    boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    match nms {
        Some(Nms::ClassAgnostic) => nms_int(iou, boxes),
        Some(Nms::ClassAware) => nms_class_aware_int(iou, boxes),
        None => boxes, // bypass NMS
    }
}

/// Dispatches to the appropriate NMS function based on mode for quantized boxes
/// with extra data.
fn dispatch_nms_extra_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send + Sync>(
    nms: Option<Nms>,
    iou: f32,
    boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    match nms {
        Some(Nms::ClassAgnostic) => nms_extra_int(iou, boxes),
        Some(Nms::ClassAware) => nms_extra_class_aware_int(iou, boxes),
        None => boxes, // bypass NMS
    }
}

/// Decodes YOLO detection outputs from quantized tensors into detection boxes.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - output: (4 + num_classes, num_boxes)
pub fn decode_yolo_det<BOX: PrimInt + AsPrimitive<f32> + Send + Sync>(
    output: (ArrayView2<BOX>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<BOX>,
{
    impl_yolo_quant::<XYWH, _>(output, score_threshold, iou_threshold, nms, output_boxes);
}

/// Decodes YOLO detection outputs from float tensors into detection boxes.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - output: (4 + num_classes, num_boxes)
pub fn decode_yolo_det_float<T>(
    output: ArrayView2<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    impl_yolo_float::<XYWH, _>(output, score_threshold, iou_threshold, nms, output_boxes);
}

/// Decodes YOLO detection and segmentation outputs from quantized tensors into
/// detection boxes and segmentation masks.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes: (4 + num_classes + num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
pub fn decode_yolo_segdet_quant<
    BOX: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<BOX>,
{
    impl_yolo_segdet_quant::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
        output_masks,
    )
}

/// Decodes YOLO detection and segmentation outputs from float tensors into
/// detection boxes and segmentation masks.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes: (4 + num_classes + num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
pub fn decode_yolo_segdet_float<T>(
    boxes: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    impl_yolo_segdet_float::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
        output_masks,
    )
}

/// Decodes YOLO split detection outputs from quantized tensors into detection
/// boxes.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes: (4, num_boxes)
/// - scores: (num_classes, num_boxes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_yolo_split_det_quant<
    BOX: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    impl_yolo_split_quant::<XYWH, _, _>(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
    );
}

/// Decodes YOLO split detection outputs from float tensors into detection
/// boxes.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes: (4, num_boxes)
/// - scores: (num_classes, num_boxes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_yolo_split_det_float<T>(
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    impl_yolo_split_float::<XYWH, _, _>(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
    );
}

/// Decodes YOLO split detection segmentation outputs from quantized tensors
/// into detection boxes and segmentation masks.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes_tensor: (4, num_boxes)
/// - scores_tensor: (num_classes, num_boxes)
/// - mask_tensor: (num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet<
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    MASK: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    mask_coeff: (ArrayView2<MASK>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<SCORE>,
{
    impl_yolo_split_segdet_quant::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
        output_masks,
    )
}

/// Decodes YOLO split detection segmentation outputs from float tensors
/// into detection boxes and segmentation masks.
///
/// Boxes are expected to be in XYWH format.
///
/// Expected shapes of inputs:
/// - boxes_tensor: (4, num_boxes)
/// - scores_tensor: (num_classes, num_boxes)
/// - mask_tensor: (num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet_float<T>(
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    mask_coeff: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    impl_yolo_split_segdet_float::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        nms,
        output_boxes,
        output_masks,
    )
}

/// Decodes end-to-end YOLO detection outputs (post-NMS from model).
/// Expects an array of shape `(6, N)`, where the first dimension (rows)
/// corresponds to the 6 per-detection features
/// `[x1, y1, x2, y2, conf, class]` and the second dimension (columns)
/// indexes the `N` detections.
/// Boxes are output directly without NMS (the model already applied NMS).
///
/// Coordinates may be normalized `[0, 1]` or absolute pixel values depending
/// on the model configuration. The caller should check
/// `decoder.normalized_boxes()` to determine which.
///
/// # Errors
///
/// Returns `DecoderError::InvalidShape` if `output` has fewer than 6 rows.
pub fn decode_yolo_end_to_end_det_float<T>(
    output: ArrayView2<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) -> Result<(), crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    // Validate input shape: need at least 6 rows (x1, y1, x2, y2, conf, class)
    if output.shape()[0] < 6 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "End-to-end detection output requires at least 6 rows, got {}",
            output.shape()[0]
        )));
    }

    // Input shape: (6, N) -> transpose to (N, 4) for boxes and (N, 1) for scores
    let boxes = output.slice(s![0..4, ..]).reversed_axes();
    let scores = output.slice(s![4..5, ..]).reversed_axes();
    let classes = output.slice(s![5, ..]);
    let mut boxes =
        postprocess_boxes_index_float::<XYXY, _, _>(score_threshold.as_(), boxes, scores);
    boxes.truncate(output_boxes.capacity());
    output_boxes.clear();
    for (mut b, i) in boxes.into_iter() {
        b.label = classes[i].as_() as usize;
        output_boxes.push(b);
    }
    // No NMS — model output is already post-NMS
    Ok(())
}

/// Decodes end-to-end YOLO detection + segmentation outputs (post-NMS from
/// model).
///
/// Input shapes:
/// - detection: (6 + num_protos, N) where rows are [x1, y1, x2, y2, conf,
///   class, mask_coeff_0, ..., mask_coeff_31]
/// - protos: (proto_height, proto_width, num_protos)
///
/// Boxes are output directly without NMS (model already applied NMS).
/// Coordinates may be normalized [0,1] or pixel values depending on model
/// config.
///
/// # Errors
///
/// Returns `DecoderError::InvalidShape` if:
/// - output has fewer than 7 rows (6 base + at least 1 mask coefficient)
/// - protos shape doesn't match mask coefficients count
pub fn decode_yolo_end_to_end_segdet_float<T>(
    output: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<crate::Segmentation>,
) -> Result<(), crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    let (boxes, scores, classes, mask_coeff) =
        postprocess_yolo_end_to_end_segdet(&output, protos.dim().2)?;
    let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
        boxes,
        scores,
        classes,
        score_threshold,
        output_boxes.capacity(),
    );

    // No NMS — model output is already post-NMS

    impl_yolo_split_segdet_process_masks(boxes, mask_coeff, protos, output_boxes, output_masks)
}

/// Decodes split end-to-end YOLO detection outputs (post-NMS from model).
///
/// Input shapes (after batch dim removed):
/// - boxes: (4, N) — xyxy pixel coordinates
/// - scores: (1, N) — confidence of the top class
/// - classes: (1, N) — class index of the top class
///
/// Boxes are output directly without NMS (model already applied NMS).
pub fn decode_yolo_split_end_to_end_det_float<T: Float + AsPrimitive<f32>>(
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    classes: ArrayView2<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) -> Result<(), crate::DecoderError> {
    let n = boxes.shape()[1];

    output_boxes.clear();

    let (boxes, scores, classes) = postprocess_yolo_split_end_to_end_det(boxes, scores, &classes)?;

    for i in 0..n {
        let score: f32 = scores[[i, 0]].as_();
        if score < score_threshold {
            continue;
        }
        if output_boxes.len() >= output_boxes.capacity() {
            break;
        }
        output_boxes.push(DetectBox {
            bbox: BoundingBox {
                xmin: boxes[[i, 0]].as_(),
                ymin: boxes[[i, 1]].as_(),
                xmax: boxes[[i, 2]].as_(),
                ymax: boxes[[i, 3]].as_(),
            },
            score,
            label: classes[i].as_() as usize,
        });
    }
    Ok(())
}

/// Decodes split end-to-end YOLO detection + segmentation outputs.
///
/// Input shapes (after batch dim removed):
/// - boxes: (4, N) — xyxy pixel coordinates
/// - scores: (1, N) — confidence
/// - classes: (1, N) — class index
/// - mask_coeff: (num_protos, N) — mask coefficients per detection
/// - protos: (proto_h, proto_w, num_protos) — prototype masks
#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_end_to_end_segdet_float<T>(
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    classes: ArrayView2<T>,
    mask_coeff: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<crate::Segmentation>,
) -> Result<(), crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
    f32: AsPrimitive<T>,
{
    let (boxes, scores, classes, mask_coeff) =
        postprocess_yolo_split_end_to_end_segdet(boxes, scores, &classes, mask_coeff)?;
    let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
        boxes,
        scores,
        classes,
        score_threshold,
        output_boxes.capacity(),
    );

    impl_yolo_split_segdet_process_masks(boxes, mask_coeff, protos, output_boxes, output_masks)
}

#[allow(clippy::type_complexity)]
pub(crate) fn postprocess_yolo_end_to_end_segdet<'a, T>(
    output: &'a ArrayView2<'_, T>,
    num_protos: usize,
) -> Result<
    (
        ArrayView2<'a, T>,
        ArrayView2<'a, T>,
        ArrayView1<'a, T>,
        ArrayView2<'a, T>,
    ),
    crate::DecoderError,
> {
    // Validate input shape: need at least 7 rows (6 base + at least 1 mask coeff)
    if output.shape()[0] < 7 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "End-to-end segdet output requires at least 7 rows, got {}",
            output.shape()[0]
        )));
    }

    let num_mask_coeffs = output.shape()[0] - 6;
    if num_mask_coeffs != num_protos {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Mask coefficients count ({}) doesn't match protos count ({})",
            num_mask_coeffs, num_protos
        )));
    }

    // Input shape: (6+num_protos, N) -> transpose for postprocessing
    let boxes = output.slice(s![0..4, ..]).reversed_axes();
    let scores = output.slice(s![4..5, ..]).reversed_axes();
    let classes = output.slice(s![5, ..]);
    let mask_coeff = output.slice(s![6.., ..]).reversed_axes();
    Ok((boxes, scores, classes, mask_coeff))
}

/// Postprocess yolo split end to end det by reversing axes of boxes,
/// scores, and flattening the class tensor.
/// Expected input shapes:
/// - boxes: (4, N)
/// - scores: (1, N)
/// - classes: (1, N)
#[allow(clippy::type_complexity)]
pub(crate) fn postprocess_yolo_split_end_to_end_det<'a, 'b, 'c, BOXES, SCORES, CLASS>(
    boxes: ArrayView2<'a, BOXES>,
    scores: ArrayView2<'b, SCORES>,
    classes: &'c ArrayView2<CLASS>,
) -> Result<
    (
        ArrayView2<'a, BOXES>,
        ArrayView2<'b, SCORES>,
        ArrayView1<'c, CLASS>,
    ),
    crate::DecoderError,
> {
    let num_boxes = boxes.shape()[1];
    if boxes.shape()[0] != 4 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end box_coords must be 4, got {}",
            boxes.shape()[0]
        )));
    }

    if scores.shape()[0] != 1 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end scores num_classes must be 1, got {}",
            scores.shape()[0]
        )));
    }

    if classes.shape()[0] != 1 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end classes num_classes must be 1, got {}",
            classes.shape()[0]
        )));
    }

    if scores.shape()[1] != num_boxes {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end scores must have same num_boxes as boxes ({}), got {}",
            num_boxes,
            scores.shape()[1]
        )));
    }

    if classes.shape()[1] != num_boxes {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end classes must have same num_boxes as boxes ({}), got {}",
            num_boxes,
            classes.shape()[1]
        )));
    }

    let boxes = boxes.reversed_axes();
    let scores = scores.reversed_axes();
    let classes = classes.slice(s![0, ..]);
    Ok((boxes, scores, classes))
}

/// Postprocess yolo split end to end segdet by reversing axes of boxes,
/// scores, mask tensors and flattening the class tensor.
#[allow(clippy::type_complexity)]
pub(crate) fn postprocess_yolo_split_end_to_end_segdet<
    'a,
    'b,
    'c,
    'd,
    BOXES,
    SCORES,
    CLASS,
    MASK,
>(
    boxes: ArrayView2<'a, BOXES>,
    scores: ArrayView2<'b, SCORES>,
    classes: &'c ArrayView2<CLASS>,
    mask_coeff: ArrayView2<'d, MASK>,
) -> Result<
    (
        ArrayView2<'a, BOXES>,
        ArrayView2<'b, SCORES>,
        ArrayView1<'c, CLASS>,
        ArrayView2<'d, MASK>,
    ),
    crate::DecoderError,
> {
    let num_boxes = boxes.shape()[1];
    if boxes.shape()[0] != 4 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end box_coords must be 4, got {}",
            boxes.shape()[0]
        )));
    }

    if scores.shape()[0] != 1 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end scores num_classes must be 1, got {}",
            scores.shape()[0]
        )));
    }

    if classes.shape()[0] != 1 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end classes num_classes must be 1, got {}",
            classes.shape()[0]
        )));
    }

    if scores.shape()[1] != num_boxes {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end scores must have same num_boxes as boxes ({}), got {}",
            num_boxes,
            scores.shape()[1]
        )));
    }

    if classes.shape()[1] != num_boxes {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end classes must have same num_boxes as boxes ({}), got {}",
            num_boxes,
            classes.shape()[1]
        )));
    }

    if mask_coeff.shape()[1] != num_boxes {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Split end-to-end mask_coeff must have same num_boxes as boxes ({}), got {}",
            num_boxes,
            mask_coeff.shape()[1]
        )));
    }

    let boxes = boxes.reversed_axes();
    let scores = scores.reversed_axes();
    let classes = classes.slice(s![0, ..]);
    let mask_coeff = mask_coeff.reversed_axes();
    Ok((boxes, scores, classes, mask_coeff))
}
/// Internal implementation of YOLO decoding for quantized tensors.
///
/// Expected shapes of inputs:
/// - output: (4 + num_classes, num_boxes)
pub(crate) fn impl_yolo_quant<B: BBoxTypeTrait, T: PrimInt + AsPrimitive<f32> + Send + Sync>(
    output: (ArrayView2<T>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<T>,
{
    let (boxes, quant_boxes) = output;
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&boxes);

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
        postprocess_boxes_quant::<B, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant_boxes,
        )
    };

    let boxes = dispatch_nms_int(nms, iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant_boxes));
    }
}

/// Internal implementation of YOLO decoding for float tensors.
///
/// Expected shapes of inputs:
/// - output: (4 + num_classes, num_boxes)
pub(crate) fn impl_yolo_float<B: BBoxTypeTrait, T: Float + AsPrimitive<f32> + Send + Sync>(
    output: ArrayView2<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<T>,
{
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes =
        postprocess_boxes_float::<B, _, _>(score_threshold.as_(), boxes_tensor, scores_tensor);
    let boxes = dispatch_nms_float(nms, iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

/// Internal implementation of YOLO split detection decoding for quantized
/// tensors.
///
/// Expected shapes of inputs:
/// - boxes: (4, num_boxes)
/// - scores: (num_classes, num_boxes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub(crate) fn impl_yolo_split_quant<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;

    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
        postprocess_boxes_quant::<B, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant_boxes,
        )
    };

    let boxes = dispatch_nms_int(nms, iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant_scores));
    }
}

/// Internal implementation of YOLO split detection decoding for float tensors.
///
/// Expected shapes of inputs:
/// - boxes: (4, num_boxes)
/// - scores: (num_classes, num_boxes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub(crate) fn impl_yolo_split_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let boxes =
        postprocess_boxes_float::<B, _, _>(score_threshold.as_(), boxes_tensor, scores_tensor);
    let boxes = dispatch_nms_float(nms, iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

/// Internal implementation of YOLO detection segmentation decoding for
/// quantized tensors.
///
/// Expected shapes of inputs:
/// - boxes: (4 + num_classes + num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
pub(crate) fn impl_yolo_segdet_quant<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<BOX>,
{
    let (boxes, quant_boxes) = boxes;
    let num_protos = protos.0.dim().2;

    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes, num_protos);
    let boxes = impl_yolo_split_segdet_quant_get_boxes::<B, _, _>(
        (boxes_tensor, quant_boxes),
        (scores_tensor, quant_boxes),
        score_threshold,
        iou_threshold,
        nms,
        MAX_NMS_CANDIDATES,
        output_boxes.capacity(),
    );

    impl_yolo_split_segdet_quant_process_masks::<_, _>(
        boxes,
        (mask_tensor, quant_boxes),
        protos,
        output_boxes,
        output_masks,
    )
}

/// Internal implementation of YOLO detection segmentation decoding for
/// float tensors.
///
/// Expected shapes of inputs:
/// - boxes: (4 + num_classes + num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub(crate) fn impl_yolo_segdet_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes: ArrayView2<BOX>,
    protos: ArrayView3<PROTO>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<BOX>,
{
    let num_protos = protos.dim().2;
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes, num_protos);
    let boxes = impl_yolo_segdet_get_boxes::<B, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        nms,
        MAX_NMS_CANDIDATES,
        output_boxes.capacity(),
    );
    impl_yolo_split_segdet_process_masks(boxes, mask_tensor, protos, output_boxes, output_masks)
}

pub(crate) fn impl_yolo_segdet_get_boxes<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
) -> Vec<(DetectBox, usize)>
where
    f32: AsPrimitive<SCORE>,
{
    let mut boxes = postprocess_boxes_index_float::<B, _, _>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
    );
    truncate_to_top_k_by_score(&mut boxes, pre_nms_top_k);
    let mut boxes = dispatch_nms_extra_float(nms, iou_threshold, boxes);
    boxes.truncate(max_det);
    boxes
}

pub(crate) fn impl_yolo_end_to_end_segdet_get_boxes<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
    CLASS: AsPrimitive<f32> + Send + Sync,
>(
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
    classes: ArrayView1<CLASS>,
    score_threshold: f32,
    max_boxes: usize,
) -> Vec<(DetectBox, usize)>
where
    f32: AsPrimitive<SCORE>,
{
    let mut boxes = postprocess_boxes_index_float::<B, _, _>(score_threshold.as_(), boxes, scores);
    boxes.truncate(max_boxes);
    for (b, ind) in &mut boxes {
        b.label = classes[*ind].as_().round() as usize;
    }
    boxes
}

pub(crate) fn impl_yolo_split_segdet_process_masks<
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes: Vec<(DetectBox, usize)>,
    masks_tensor: ArrayView2<MASK>,
    protos_tensor: ArrayView3<PROTO>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError> {
    let boxes = decode_segdet_f32(boxes, masks_tensor, protos_tensor)?;
    output_boxes.clear();
    output_masks.clear();
    for (b, m) in boxes.into_iter() {
        output_boxes.push(b);
        output_masks.push(Segmentation {
            xmin: b.bbox.xmin,
            ymin: b.bbox.ymin,
            xmax: b.bbox.xmax,
            ymax: b.bbox.ymax,
            segmentation: m,
        });
    }
    Ok(())
}
/// Expected input shapes:
/// - boxes_tensor: (num_boxes, 4)
/// - scores_tensor: (num_boxes, num_classes)
pub(crate) fn impl_yolo_split_segdet_quant_get_boxes<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
) -> Vec<(DetectBox, usize)>
where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;

    let span = tracing::trace_span!(
        "decode",
        n_candidates = tracing::field::Empty,
        n_after_topk = tracing::field::Empty,
        n_after_nms = tracing::field::Empty,
        n_detections = tracing::field::Empty,
    );
    let _guard = span.enter();

    let mut boxes = {
        let _s = tracing::trace_span!("score_filter").entered();
        let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
        postprocess_boxes_index_quant::<B, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant_boxes,
        )
    };
    span.record("n_candidates", boxes.len());

    {
        let _s = tracing::trace_span!("top_k", k = pre_nms_top_k).entered();
        truncate_to_top_k_by_score_quant(&mut boxes, pre_nms_top_k);
    }
    span.record("n_after_topk", boxes.len());

    let mut boxes = {
        let _s = tracing::trace_span!("nms").entered();
        dispatch_nms_extra_int(nms, iou_threshold, boxes)
    };
    span.record("n_after_nms", boxes.len());

    boxes.truncate(max_det);
    let result: Vec<_> = {
        let _s = tracing::trace_span!("box_dequant", n = boxes.len()).entered();
        boxes
            .into_iter()
            .map(|(b, i)| (dequant_detect_box(&b, quant_scores), i))
            .collect()
    };
    span.record("n_detections", result.len());

    result
}

pub(crate) fn impl_yolo_split_segdet_quant_process_masks<
    MASK: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: Vec<(DetectBox, usize)>,
    mask_coeff: (ArrayView2<MASK>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError> {
    let (masks, quant_masks) = mask_coeff;
    let (protos, quant_protos) = protos;

    let boxes = decode_segdet_quant(boxes, masks, protos, quant_masks, quant_protos)?;
    output_boxes.clear();
    output_masks.clear();
    for (b, m) in boxes.into_iter() {
        output_boxes.push(b);
        output_masks.push(Segmentation {
            xmin: b.bbox.xmin,
            ymin: b.bbox.ymin,
            xmax: b.bbox.xmax,
            ymax: b.bbox.ymax,
            segmentation: m,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Internal implementation of YOLO split detection segmentation decoding for
/// quantized tensors.
///
/// Expected shapes of inputs:
/// - boxes_tensor: (4, num_boxes)
/// - scores_tensor: (num_classes, num_boxes)
/// - mask_tensor: (num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
pub(crate) fn impl_yolo_split_segdet_quant<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    MASK: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    mask_coeff: (ArrayView2<MASK>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_, scores_, mask_coeff_) =
        postprocess_yolo_split_segdet(boxes.0, scores.0, mask_coeff.0);
    let boxes = (boxes_, boxes.1);
    let scores = (scores_, scores.1);
    let mask_coeff = (mask_coeff_, mask_coeff.1);

    let boxes = impl_yolo_split_segdet_quant_get_boxes::<B, _, _>(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        nms,
        MAX_NMS_CANDIDATES,
        output_boxes.capacity(),
    );

    impl_yolo_split_segdet_quant_process_masks(
        boxes,
        mask_coeff,
        protos,
        output_boxes,
        output_masks,
    )
}

#[allow(clippy::too_many_arguments)]
/// Internal implementation of YOLO split detection segmentation decoding for
/// float tensors.
///
/// Expected shapes of inputs:
/// - boxes_tensor: (4, num_boxes)
/// - scores_tensor: (num_classes, num_boxes)
/// - mask_tensor: (num_protos, num_boxes)
/// - protos: (proto_height, proto_width, num_protos)
///
/// # Errors
/// Returns `DecoderError::InvalidShape` if bounding boxes are not normalized.
pub(crate) fn impl_yolo_split_segdet_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    mask_tensor: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) -> Result<(), crate::DecoderError>
where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, scores_tensor, mask_tensor) =
        postprocess_yolo_split_segdet(boxes_tensor, scores_tensor, mask_tensor);

    let boxes = impl_yolo_segdet_get_boxes::<B, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        nms,
        MAX_NMS_CANDIDATES,
        output_boxes.capacity(),
    );
    impl_yolo_split_segdet_process_masks(boxes, mask_tensor, protos, output_boxes, output_masks)
}

// ---------------------------------------------------------------------------
// Proto-extraction variants: return ProtoData instead of materialized masks
// ---------------------------------------------------------------------------

/// Proto-extraction variant of `impl_yolo_segdet_quant`.
/// Runs NMS but returns raw `ProtoData` instead of materialized masks.
#[allow(clippy::too_many_arguments)]
pub fn impl_yolo_segdet_quant_proto<
    B: BBoxTypeTrait,
    BOX: PrimInt
        + AsPrimitive<i64>
        + AsPrimitive<i128>
        + AsPrimitive<f32>
        + AsPrimitive<i8>
        + Send
        + Sync,
    PROTO: PrimInt
        + AsPrimitive<i64>
        + AsPrimitive<i128>
        + AsPrimitive<f32>
        + AsPrimitive<i8>
        + Send
        + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData
where
    f32: AsPrimitive<BOX>,
{
    let (boxes_arr, quant_boxes) = boxes;
    let (protos_arr, quant_protos) = protos;
    let num_protos = protos_arr.dim().2;

    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes_arr, num_protos);

    let det_indices = impl_yolo_split_segdet_quant_get_boxes::<B, _, _>(
        (boxes_tensor, quant_boxes),
        (scores_tensor, quant_boxes),
        score_threshold,
        iou_threshold,
        nms,
        pre_nms_top_k,
        max_det,
    );

    extract_proto_data_quant(
        det_indices,
        mask_tensor,
        quant_boxes,
        protos_arr,
        quant_protos,
        output_boxes,
    )
}

/// Proto-extraction variant of `impl_yolo_segdet_float`.
/// Runs NMS but returns raw `ProtoData` instead of materialized masks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn impl_yolo_segdet_float_proto<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
    PROTO: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
>(
    boxes: ArrayView2<BOX>,
    protos: ArrayView3<PROTO>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData
where
    f32: AsPrimitive<BOX>,
{
    let num_protos = protos.dim().2;
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes, num_protos);

    let boxes = impl_yolo_segdet_get_boxes::<B, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        nms,
        pre_nms_top_k,
        max_det,
    );

    extract_proto_data_float(boxes, mask_tensor, protos, output_boxes)
}

/// Proto-extraction variant of `impl_yolo_split_segdet_float`.
/// Runs NMS but returns raw `ProtoData` instead of materialized masks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn impl_yolo_split_segdet_float_proto<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
    MASK: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
    PROTO: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    mask_tensor: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData
where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, scores_tensor, mask_tensor) =
        postprocess_yolo_split_segdet(boxes_tensor, scores_tensor, mask_tensor);
    let det_indices = impl_yolo_segdet_get_boxes::<B, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        nms,
        pre_nms_top_k,
        max_det,
    );

    extract_proto_data_float(det_indices, mask_tensor, protos, output_boxes)
}

/// Proto-extraction variant of `decode_yolo_end_to_end_segdet_float`.
pub fn decode_yolo_end_to_end_segdet_float_proto<T>(
    output: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) -> Result<ProtoData, crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
    f32: AsPrimitive<T>,
{
    let (boxes, scores, classes, mask_coeff) =
        postprocess_yolo_end_to_end_segdet(&output, protos.dim().2)?;
    let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
        boxes,
        scores,
        classes,
        score_threshold,
        output_boxes.capacity(),
    );

    Ok(extract_proto_data_float(
        boxes,
        mask_coeff,
        protos,
        output_boxes,
    ))
}

/// Proto-extraction variant of `decode_yolo_split_end_to_end_segdet_float`.
#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_end_to_end_segdet_float_proto<T>(
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    classes: ArrayView2<T>,
    mask_coeff: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) -> Result<ProtoData, crate::DecoderError>
where
    T: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
    f32: AsPrimitive<T>,
{
    let (boxes, scores, classes, mask_coeff) =
        postprocess_yolo_split_end_to_end_segdet(boxes, scores, &classes, mask_coeff)?;
    let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
        boxes,
        scores,
        classes,
        score_threshold,
        output_boxes.capacity(),
    );

    Ok(extract_proto_data_float(
        boxes,
        mask_coeff,
        protos,
        output_boxes,
    ))
}

/// Helper: extract ProtoData from float mask coefficients + protos.
///
/// Builds [`ProtoData`] with both `protos` and `mask_coefficients` as
/// [`edgefirst_tensor::TensorDyn`]. Preserves the native element type for
/// `f16` and `f32`; narrows `f64` to `f32` (there is no native f64 kernel
/// path). `mask_coefficients` shape is `[num_detections, num_protos]`.
pub(super) fn extract_proto_data_float<
    MASK: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
    PROTO: Float + AsPrimitive<f32> + Copy + Send + Sync + FloatProtoElem,
>(
    det_indices: Vec<(DetectBox, usize)>,
    mask_tensor: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData {
    let num_protos = mask_tensor.ncols();
    let n = det_indices.len();

    // Per-detection coefficients packed row-major into a contiguous buffer,
    // preserving the source dtype. Shape: [N, num_protos] — N=0 is permitted
    // (tracker path emits no detections this frame) since Mem-backed tensors
    // accept zero-element shapes as "empty collection" sentinels.
    let mut coeff_rows: Vec<MASK> = Vec::with_capacity(n * num_protos);
    output_boxes.clear();
    for (det, idx) in det_indices {
        output_boxes.push(det);
        let row = mask_tensor.row(idx);
        coeff_rows.extend(row.iter().copied());
    }

    let mask_coefficients = MASK::slice_into_tensor_dyn(&coeff_rows, &[n, num_protos])
        .expect("allocating mask_coefficients TensorDyn");
    let protos_tensor =
        PROTO::arrayview3_into_tensor_dyn(protos).expect("allocating protos TensorDyn");

    ProtoData {
        mask_coefficients,
        protos: protos_tensor,
        layout: ProtoLayout::Nhwc,
    }
}

/// Helper: extract ProtoData from quantized mask coefficients + protos.
///
/// Dequantizes mask coefficients to f32 at extraction (one-time cost on a
/// `num_detections * num_protos` slice) and keeps protos in raw i8,
/// attaching the dequantization params as
/// [`edgefirst_tensor::Quantization::per_tensor`] metadata on the proto
/// tensor. The GPU shader / CPU kernel reads `protos.quantization()` and
/// dequantizes per-texel.
pub(crate) fn extract_proto_data_quant<
    MASK: PrimInt + AsPrimitive<f32> + AsPrimitive<i8> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<f32> + AsPrimitive<i8> + Send + Sync + 'static,
>(
    det_indices: Vec<(DetectBox, usize)>,
    mask_tensor: ArrayView2<MASK>,
    quant_masks: Quantization,
    protos: ArrayView3<PROTO>,
    quant_protos: Quantization,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData {
    use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

    let span = tracing::trace_span!(
        "extract_proto",
        n = det_indices.len(),
        num_protos = tracing::field::Empty,
        layout = tracing::field::Empty,
    );
    let _guard = span.enter();

    let num_protos = mask_tensor.ncols();
    let n = det_indices.len();
    span.record("num_protos", num_protos);

    // Keep mask coefficients in raw i8 with quantization metadata attached.
    // Consumers that need f32 can dequantize on the fly; the scaled-path
    // integer kernel uses raw i8 directly for i8×i8→i32 dot products.
    let mut coeff_i8 = Vec::<i8>::with_capacity(n * num_protos);
    output_boxes.clear();
    for (det, idx) in det_indices {
        output_boxes.push(det);
        let row = mask_tensor.row(idx);
        coeff_i8.extend(row.iter().map(|v| {
            let v_i8: i8 = v.as_();
            v_i8
        }));
    }

    // Shape `[n, num_protos]` with n=0 is permitted (tracker path emits no
    // fresh detections this frame) via the Mem-backed zero-size allowance.
    let coeff_tensor = Tensor::<i8>::new(&[n, num_protos], Some(TensorMemory::Mem), None)
        .expect("allocating mask_coefficients tensor");
    if n > 0 {
        let mut m = coeff_tensor
            .map()
            .expect("mapping mask_coefficients tensor");
        m.as_mut_slice().copy_from_slice(&coeff_i8);
    }
    let coeff_quant =
        edgefirst_tensor::Quantization::per_tensor(quant_masks.scale, quant_masks.zero_point);
    let coeff_tensor = coeff_tensor
        .with_quantization(coeff_quant)
        .expect("per-tensor quantization on mask coefficients");
    let mask_coefficients = TensorDyn::I8(coeff_tensor);

    // Keep protos in raw i8 — consumers dequantize via protos.quantization().
    // When PROTO is already i8, detect layout and copy efficiently without
    // transposing. The mask materialisation kernels dispatch on the layout.
    let (h, w, k) = protos.dim();

    // Lazy extraction: if no detections survived NMS, return a minimal
    // ProtoData without copying the 819KB proto tensor at all.
    if n == 0 {
        let tensor_quant =
            edgefirst_tensor::Quantization::per_tensor(quant_protos.scale, quant_protos.zero_point);
        // Empty protos with valid shape metadata so consumers can still
        // inspect proto_h/proto_w/num_protos without panicking.
        let empty_tensor = Tensor::<i8>::new(&[h, w, k], Some(TensorMemory::Mem), None)
            .expect("allocating empty protos tensor");
        let empty_tensor = empty_tensor
            .with_quantization(tensor_quant)
            .expect("per-tensor quantization on empty protos");
        return ProtoData {
            mask_coefficients,
            protos: TensorDyn::I8(empty_tensor),
            layout: ProtoLayout::Nhwc,
        };
    }

    // Determine physical layout and copy strategy.
    let (proto_shape, proto_layout) =
        if std::any::TypeId::of::<PROTO>() == std::any::TypeId::of::<i8>() {
            if protos.is_standard_layout() {
                // Already NHWC [H, W, K] in contiguous memory.
                (&[h, w, k][..], ProtoLayout::Nhwc)
            } else if protos.ndim() == 3 && protos.strides()[2] > 1 {
                // NCHW reinterpreted as NHWC via stride swap. Physical storage
                // is [K, H, W] contiguous. Keep in NCHW — eliminates the costly
                // 3.1ms transpose entirely.
                (&[k, h, w][..], ProtoLayout::Nchw)
            } else {
                // Unknown layout — fall back to iter copy as NHWC.
                (&[h, w, k][..], ProtoLayout::Nhwc)
            }
        } else {
            (&[h, w, k][..], ProtoLayout::Nhwc)
        };

    let protos_tensor = Tensor::<i8>::new(proto_shape, Some(TensorMemory::Mem), None)
        .expect("allocating protos tensor");
    {
        let mut m = protos_tensor.map().expect("mapping protos tensor");
        let dst = m.as_mut_slice();
        if std::any::TypeId::of::<PROTO>() == std::any::TypeId::of::<i8>() {
            // SAFETY: PROTO == i8 checked via TypeId; cast slice view is
            // size/alignment-compatible by construction.
            if protos.is_standard_layout() {
                let src: &[i8] = unsafe {
                    std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len())
                };
                dst.copy_from_slice(src);
            } else if protos.ndim() == 3 && protos.strides()[2] > 1 {
                // NCHW physical layout — sequential copy WITHOUT transpose.
                // This saves ~3.1ms on A53/A55 by avoiding the tiled
                // NCHW→NHWC transpose of the 819KB proto buffer.
                let total = h * w * k;
                // SAFETY: ArrayView was constructed from a contiguous slice of
                // `total` elements. as_ptr() points to the base of that slice.
                let src: &[i8] =
                    unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, total) };
                dst.copy_from_slice(src);
            } else {
                for (d, s) in dst.iter_mut().zip(protos.iter()) {
                    let v_i8: i8 = s.as_();
                    *d = v_i8;
                }
            }
        } else {
            for (d, s) in dst.iter_mut().zip(protos.iter()) {
                let v_i8: i8 = s.as_();
                *d = v_i8;
            }
        }
    }
    let tensor_quant =
        edgefirst_tensor::Quantization::per_tensor(quant_protos.scale, quant_protos.zero_point);
    let protos_tensor = protos_tensor
        .with_quantization(tensor_quant)
        .expect("per-tensor quantization on new Tensor<i8>");

    span.record("layout", format!("{:?}", proto_layout).as_str());

    ProtoData {
        mask_coefficients,
        protos: TensorDyn::I8(protos_tensor),
        layout: proto_layout,
    }
}

/// Per-float-dtype construction of a [`TensorDyn`] from a flat slice / 3-D
/// `ArrayView`. Replaces the old `IntoProtoTensor` trait. Each implementor
/// either passes its element type straight to `Tensor::from_slice` /
/// `Tensor::from_arrayview3`, or narrows `f64` to `f32` (no native f64 kernel
/// path exists).
pub trait FloatProtoElem: Copy + 'static {
    fn slice_into_tensor_dyn(
        values: &[Self],
        shape: &[usize],
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn>;

    fn arrayview3_into_tensor_dyn(
        view: ArrayView3<'_, Self>,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn>;
}

impl FloatProtoElem for f32 {
    fn slice_into_tensor_dyn(
        values: &[f32],
        shape: &[usize],
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        edgefirst_tensor::Tensor::<f32>::from_slice(values, shape)
            .map(edgefirst_tensor::TensorDyn::F32)
    }
    fn arrayview3_into_tensor_dyn(
        view: ArrayView3<'_, f32>,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        edgefirst_tensor::Tensor::<f32>::from_arrayview3(view).map(edgefirst_tensor::TensorDyn::F32)
    }
}

impl FloatProtoElem for half::f16 {
    fn slice_into_tensor_dyn(
        values: &[half::f16],
        shape: &[usize],
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        edgefirst_tensor::Tensor::<half::f16>::from_slice(values, shape)
            .map(edgefirst_tensor::TensorDyn::F16)
    }
    fn arrayview3_into_tensor_dyn(
        view: ArrayView3<'_, half::f16>,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        edgefirst_tensor::Tensor::<half::f16>::from_arrayview3(view)
            .map(edgefirst_tensor::TensorDyn::F16)
    }
}

impl FloatProtoElem for f64 {
    fn slice_into_tensor_dyn(
        values: &[f64],
        shape: &[usize],
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        // Narrow to f32 — no native f64 kernel path.
        let narrowed: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        edgefirst_tensor::Tensor::<f32>::from_slice(&narrowed, shape)
            .map(edgefirst_tensor::TensorDyn::F32)
    }
    fn arrayview3_into_tensor_dyn(
        view: ArrayView3<'_, f64>,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::TensorDyn> {
        let narrowed: ndarray::Array3<f32> = view.mapv(|v| v as f32);
        edgefirst_tensor::Tensor::<f32>::from_arrayview3(narrowed.view())
            .map(edgefirst_tensor::TensorDyn::F32)
    }
}

fn postprocess_yolo<'a, T>(
    output: &'a ArrayView2<'_, T>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>) {
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4.., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor)
}

pub(crate) fn postprocess_yolo_seg<'a, T>(
    output: &'a ArrayView2<'_, T>,
    num_protos: usize,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>, ArrayView2<'a, T>) {
    assert!(
        output.shape()[0] > num_protos + 4,
        "Output shape is too short: {} <= {} + 4",
        output.shape()[0],
        num_protos
    );
    let num_classes = output.shape()[0] - 4 - num_protos;
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]).reversed_axes();
    let mask_tensor = output.slice(s![(num_classes + 4).., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor, mask_tensor)
}

pub(crate) fn postprocess_yolo_split_segdet<'a, 'b, 'c, BOX, SCORE, MASK>(
    boxes_tensor: ArrayView2<'a, BOX>,
    scores_tensor: ArrayView2<'b, SCORE>,
    mask_tensor: ArrayView2<'c, MASK>,
) -> (
    ArrayView2<'a, BOX>,
    ArrayView2<'b, SCORE>,
    ArrayView2<'c, MASK>,
) {
    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let mask_tensor = mask_tensor.reversed_axes();
    (boxes_tensor, scores_tensor, mask_tensor)
}

fn decode_segdet_f32<
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes: Vec<(DetectBox, usize)>,
    masks: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
) -> Result<Vec<(DetectBox, Array3<u8>)>, crate::DecoderError> {
    if boxes.is_empty() {
        return Ok(Vec::new());
    }
    if masks.shape()[1] != protos.shape()[2] {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Mask coefficients count ({}) doesn't match protos channel count ({})",
            masks.shape()[1],
            protos.shape()[2],
        )));
    }
    boxes
        .into_par_iter()
        .map(|mut b| {
            let ind = b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox)?;
            b.0.bbox = roi;
            Ok((b.0, make_segmentation(masks.row(ind), protos.view())))
        })
        .collect()
}

pub(crate) fn decode_segdet_quant<
    MASK: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + Send + Sync,
>(
    boxes: Vec<(DetectBox, usize)>,
    masks: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
    quant_masks: Quantization,
    quant_protos: Quantization,
) -> Result<Vec<(DetectBox, Array3<u8>)>, crate::DecoderError> {
    if boxes.is_empty() {
        return Ok(Vec::new());
    }
    if masks.shape()[1] != protos.shape()[2] {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Mask coefficients count ({}) doesn't match protos channel count ({})",
            masks.shape()[1],
            protos.shape()[2],
        )));
    }

    let total_bits = MASK::zero().count_zeros() + PROTO::zero().count_zeros() + 5; // 32 protos is 2^5
    boxes
        .into_iter()
        .map(|mut b| {
            let i = b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox.to_canonical())?;
            b.0.bbox = roi;
            let seg = match total_bits {
                0..=64 => make_segmentation_quant::<MASK, PROTO, i64>(
                    masks.row(i),
                    protos.view(),
                    quant_masks,
                    quant_protos,
                ),
                65..=128 => make_segmentation_quant::<MASK, PROTO, i128>(
                    masks.row(i),
                    protos.view(),
                    quant_masks,
                    quant_protos,
                ),
                _ => {
                    return Err(crate::DecoderError::NotSupported(format!(
                        "Unsupported bit width ({total_bits}) for segmentation computation"
                    )));
                }
            };
            Ok((b.0, seg))
        })
        .collect()
}

fn protobox<'a, T>(
    protos: &'a ArrayView3<T>,
    roi: &BoundingBox,
) -> Result<(ArrayView3<'a, T>, BoundingBox), crate::DecoderError> {
    let width = protos.dim().1 as f32;
    let height = protos.dim().0 as f32;

    // Detect un-normalized bounding boxes (pixel-space coordinates).
    // protobox expects normalized coordinates in [0, 1]. ONNX models output
    // pixel-space boxes (e.g. 0-640) which must be normalized before calling
    // decode(). Without this check, pixel-space coordinates silently clamp to
    // the proto boundary, producing empty (0, 0, C) masks for every detection.
    //
    // The limit is set to 2.0 (not 1.01) because YOLO models legitimately
    // predict coordinates slightly > 1.0 for objects near frame edges.
    // Any value > 2.0 is clearly pixel-space (even the smallest practical
    // model input of 32×32 would produce coordinates >> 2.0).
    const NORM_LIMIT: f32 = 2.0;
    if roi.xmin > NORM_LIMIT
        || roi.ymin > NORM_LIMIT
        || roi.xmax > NORM_LIMIT
        || roi.ymax > NORM_LIMIT
    {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Bounding box coordinates appear un-normalized (pixel-space). \
             Got bbox=({:.2}, {:.2}, {:.2}, {:.2}) but expected values in [0, 1]. \
             ONNX models output pixel-space boxes — normalize them by dividing by \
             the input dimensions before calling decode().",
            roi.xmin, roi.ymin, roi.xmax, roi.ymax,
        )));
    }

    let roi = [
        (roi.xmin * width).clamp(0.0, width) as usize,
        (roi.ymin * height).clamp(0.0, height) as usize,
        (roi.xmax * width).clamp(0.0, width).ceil() as usize,
        (roi.ymax * height).clamp(0.0, height).ceil() as usize,
    ];

    let roi_norm = [
        roi[0] as f32 / width,
        roi[1] as f32 / height,
        roi[2] as f32 / width,
        roi[3] as f32 / height,
    ]
    .into();

    let cropped = protos.slice(s![roi[1]..roi[3], roi[0]..roi[2], ..]);

    Ok((cropped, roi_norm))
}

/// Compute a single instance segmentation mask from mask coefficients and
/// proto maps (float path).
///
/// Computes `sigmoid(coefficients · protos)` and maps to `[0, 255]`.
/// Returns an `(H, W, 1)` u8 array.
fn make_segmentation<
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    mask: ArrayView1<MASK>,
    protos: ArrayView3<PROTO>,
) -> Array3<u8> {
    let shape = protos.shape();

    // Safe to unwrap since the shapes will always be compatible
    let mask = mask.to_shape((1, mask.len())).unwrap();
    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();
    let mask = mask.map(|x| x.as_());
    let protos = protos.map(|x| x.as_());

    // Safe to unwrap since the shapes will always be compatible
    let mask = mask
        .dot(&protos)
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    mask.map(|x| {
        let sigmoid = 1.0 / (1.0 + (-*x).exp());
        (sigmoid * 255.0).round() as u8
    })
}

/// Compute a single instance segmentation mask from quantized mask
/// coefficients and proto maps.
///
/// Dequantizes both inputs (subtracting zero-points), computes the dot
/// product, applies sigmoid, and maps to `[0, 255]`.
/// Returns an `(H, W, 1)` u8 array.
fn make_segmentation_quant<
    MASK: PrimInt + AsPrimitive<DEST> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<DEST> + Send + Sync,
    DEST: PrimInt + 'static + Signed + AsPrimitive<f32> + Debug,
>(
    mask: ArrayView1<MASK>,
    protos: ArrayView3<PROTO>,
    quant_masks: Quantization,
    quant_protos: Quantization,
) -> Array3<u8>
where
    i32: AsPrimitive<DEST>,
    f32: AsPrimitive<DEST>,
{
    let shape = protos.shape();

    // Safe to unwrap since the shapes will always be compatible
    let mask = mask.to_shape((1, mask.len())).unwrap();

    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();

    let zp = quant_masks.zero_point.as_();

    let mask = mask.mapv(|x| x.as_() - zp);

    let zp = quant_protos.zero_point.as_();
    let protos = protos.mapv(|x| x.as_() - zp);

    // Safe to unwrap since the shapes will always be compatible
    let segmentation = mask
        .dot(&protos)
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    let combined_scale = quant_masks.scale * quant_protos.scale;
    segmentation.map(|x| {
        let val: f32 = (*x).as_() * combined_scale;
        let sigmoid = 1.0 / (1.0 + (-val).exp());
        (sigmoid * 255.0).round() as u8
    })
}

/// Converts Yolo Instance Segmentation into a 2D mask.
///
/// The input segmentation is expected to have shape (H, W, 1).
///
/// The output mask will have shape (H, W), with values 0 or 1 based on the
/// threshold.
///
/// # Errors
///
/// Returns `DecoderError::InvalidShape` if the input segmentation does not
/// have shape (H, W, 1).
pub fn yolo_segmentation_to_mask(
    segmentation: ArrayView3<u8>,
    threshold: u8,
) -> Result<Array2<u8>, crate::DecoderError> {
    if segmentation.shape()[2] != 1 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "Yolo Instance Segmentation should have shape (H, W, 1), got (H, W, {})",
            segmentation.shape()[2]
        )));
    }
    Ok(segmentation
        .slice(s![.., .., 0])
        .map(|x| if *x >= threshold { 1 } else { 0 }))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ========================================================================
    // Tests for decode_yolo_end_to_end_det_float
    // ========================================================================

    #[test]
    fn test_end_to_end_det_basic_filtering() {
        // Create synthetic end-to-end detection output: (6, N) where rows are
        // [x1, y1, x2, y2, conf, class]
        // 3 detections: one above threshold, two below
        let data: Vec<f32> = vec![
            // Detection 0: high score (0.9)
            0.1, 0.2, 0.3, // x1 values
            0.1, 0.2, 0.3, // y1 values
            0.5, 0.6, 0.7, // x2 values
            0.5, 0.6, 0.7, // y2 values
            0.9, 0.1, 0.2, // confidence scores
            0.0, 1.0, 2.0, // class indices
        ];
        let output = Array2::from_shape_vec((6, 3), data).unwrap();

        let mut boxes = Vec::with_capacity(10);
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        // Only 1 detection should pass threshold of 0.5
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].label, 0);
        assert!((boxes[0].score - 0.9).abs() < 0.01);
        assert!((boxes[0].bbox.xmin - 0.1).abs() < 0.01);
        assert!((boxes[0].bbox.ymin - 0.1).abs() < 0.01);
        assert!((boxes[0].bbox.xmax - 0.5).abs() < 0.01);
        assert!((boxes[0].bbox.ymax - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_end_to_end_det_all_pass_threshold() {
        // All detections above threshold
        let data: Vec<f32> = vec![
            10.0, 20.0, // x1
            10.0, 20.0, // y1
            50.0, 60.0, // x2
            50.0, 60.0, // y2
            0.8, 0.7, // conf (both above 0.5)
            1.0, 2.0, // class
        ];
        let output = Array2::from_shape_vec((6, 2), data).unwrap();

        let mut boxes = Vec::with_capacity(10);
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].label, 1);
        assert_eq!(boxes[1].label, 2);
    }

    #[test]
    fn test_end_to_end_det_none_pass_threshold() {
        // All detections below threshold
        let data: Vec<f32> = vec![
            10.0, 20.0, // x1
            10.0, 20.0, // y1
            50.0, 60.0, // x2
            50.0, 60.0, // y2
            0.1, 0.2, // conf (both below 0.5)
            1.0, 2.0, // class
        ];
        let output = Array2::from_shape_vec((6, 2), data).unwrap();

        let mut boxes = Vec::with_capacity(10);
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        assert_eq!(boxes.len(), 0);
    }

    #[test]
    fn test_end_to_end_det_capacity_limit() {
        // Test that output is truncated to capacity
        let data: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, // x1
            0.1, 0.2, 0.3, 0.4, 0.5, // y1
            0.5, 0.6, 0.7, 0.8, 0.9, // x2
            0.5, 0.6, 0.7, 0.8, 0.9, // y2
            0.9, 0.9, 0.9, 0.9, 0.9, // conf (all pass)
            0.0, 1.0, 2.0, 3.0, 4.0, // class
        ];
        let output = Array2::from_shape_vec((6, 5), data).unwrap();

        let mut boxes = Vec::with_capacity(2); // Only allow 2 boxes
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn test_end_to_end_det_empty_output() {
        // Test with zero detections
        let output = Array2::<f32>::zeros((6, 0));

        let mut boxes = Vec::with_capacity(10);
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        assert_eq!(boxes.len(), 0);
    }

    #[test]
    fn test_end_to_end_det_pixel_coordinates() {
        // Test with pixel coordinates (non-normalized)
        let data: Vec<f32> = vec![
            100.0, // x1
            200.0, // y1
            300.0, // x2
            400.0, // y2
            0.95,  // conf
            5.0,   // class
        ];
        let output = Array2::from_shape_vec((6, 1), data).unwrap();

        let mut boxes = Vec::with_capacity(10);
        decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes).unwrap();

        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].label, 5);
        assert!((boxes[0].bbox.xmin - 100.0).abs() < 0.01);
        assert!((boxes[0].bbox.ymin - 200.0).abs() < 0.01);
        assert!((boxes[0].bbox.xmax - 300.0).abs() < 0.01);
        assert!((boxes[0].bbox.ymax - 400.0).abs() < 0.01);
    }

    #[test]
    fn test_end_to_end_det_invalid_shape() {
        // Test with too few rows (needs at least 6)
        let output = Array2::<f32>::zeros((5, 3));

        let mut boxes = Vec::with_capacity(10);
        let result = decode_yolo_end_to_end_det_float(output.view(), 0.5, &mut boxes);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(crate::DecoderError::InvalidShape(s)) if s.contains("at least 6 rows")
        ));
    }

    // ========================================================================
    // Tests for decode_yolo_end_to_end_segdet_float
    // ========================================================================

    #[test]
    fn test_end_to_end_segdet_basic() {
        // Create synthetic segdet output: (6 + num_protos, N)
        // Detection format: [x1, y1, x2, y2, conf, class, mask_coeff_0..31]
        let num_protos = 32;
        let num_detections = 2;
        let num_features = 6 + num_protos;

        // Build detection tensor
        let mut data = vec![0.0f32; num_features * num_detections];
        // Detection 0: passes threshold
        data[0] = 0.1; // x1[0]
        data[1] = 0.5; // x1[1]
        data[num_detections] = 0.1; // y1[0]
        data[num_detections + 1] = 0.5; // y1[1]
        data[2 * num_detections] = 0.4; // x2[0]
        data[2 * num_detections + 1] = 0.9; // x2[1]
        data[3 * num_detections] = 0.4; // y2[0]
        data[3 * num_detections + 1] = 0.9; // y2[1]
        data[4 * num_detections] = 0.9; // conf[0] - passes
        data[4 * num_detections + 1] = 0.3; // conf[1] - fails
        data[5 * num_detections] = 1.0; // class[0]
        data[5 * num_detections + 1] = 2.0; // class[1]
                                            // Fill mask coefficients with small values
        for i in 6..num_features {
            data[i * num_detections] = 0.1;
            data[i * num_detections + 1] = 0.1;
        }

        let output = Array2::from_shape_vec((num_features, num_detections), data).unwrap();

        // Create protos tensor: (proto_height, proto_width, num_protos)
        let protos = Array3::<f32>::zeros((16, 16, num_protos));

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        // Only detection 0 should pass
        assert_eq!(boxes.len(), 1);
        assert_eq!(masks.len(), 1);
        assert_eq!(boxes[0].label, 1);
        assert!((boxes[0].score - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_end_to_end_segdet_mask_coordinates() {
        // Test that mask coordinates match box coordinates
        let num_protos = 32;
        let num_features = 6 + num_protos;

        let mut data = vec![0.0f32; num_features];
        data[0] = 0.2; // x1
        data[1] = 0.2; // y1
        data[2] = 0.8; // x2
        data[3] = 0.8; // y2
        data[4] = 0.95; // conf
        data[5] = 3.0; // class

        let output = Array2::from_shape_vec((num_features, 1), data).unwrap();
        let protos = Array3::<f32>::zeros((16, 16, num_protos));

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        assert_eq!(boxes.len(), 1);
        assert_eq!(masks.len(), 1);

        // Verify mask coordinates match box coordinates
        assert!((masks[0].xmin - boxes[0].bbox.xmin).abs() < 0.01);
        assert!((masks[0].ymin - boxes[0].bbox.ymin).abs() < 0.01);
        assert!((masks[0].xmax - boxes[0].bbox.xmax).abs() < 0.01);
        assert!((masks[0].ymax - boxes[0].bbox.ymax).abs() < 0.01);
    }

    #[test]
    fn test_end_to_end_segdet_empty_output() {
        let num_protos = 32;
        let output = Array2::<f32>::zeros((6 + num_protos, 0));
        let protos = Array3::<f32>::zeros((16, 16, num_protos));

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        assert_eq!(boxes.len(), 0);
        assert_eq!(masks.len(), 0);
    }

    #[test]
    fn test_end_to_end_segdet_capacity_limit() {
        let num_protos = 32;
        let num_detections = 5;
        let num_features = 6 + num_protos;

        let mut data = vec![0.0f32; num_features * num_detections];
        // All detections pass threshold
        for i in 0..num_detections {
            data[i] = 0.1 * (i as f32); // x1
            data[num_detections + i] = 0.1 * (i as f32); // y1
            data[2 * num_detections + i] = 0.1 * (i as f32) + 0.2; // x2
            data[3 * num_detections + i] = 0.1 * (i as f32) + 0.2; // y2
            data[4 * num_detections + i] = 0.9; // conf
            data[5 * num_detections + i] = i as f32; // class
        }

        let output = Array2::from_shape_vec((num_features, num_detections), data).unwrap();
        let protos = Array3::<f32>::zeros((16, 16, num_protos));

        let mut boxes = Vec::with_capacity(2); // Limit to 2
        let mut masks = Vec::with_capacity(2);
        decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        assert_eq!(boxes.len(), 2);
        assert_eq!(masks.len(), 2);
    }

    #[test]
    fn test_end_to_end_segdet_invalid_shape_too_few_rows() {
        // Test with too few rows (needs at least 7: 6 base + 1 mask coeff)
        let output = Array2::<f32>::zeros((6, 3));
        let protos = Array3::<f32>::zeros((16, 16, 32));

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        let result = decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        );

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(crate::DecoderError::InvalidShape(s)) if s.contains("at least 7 rows")
        ));
    }

    #[test]
    fn test_end_to_end_segdet_invalid_shape_protos_mismatch() {
        // Test with mismatched mask coefficients and protos count
        let num_protos = 32;
        let output = Array2::<f32>::zeros((6 + 16, 3)); // 16 mask coeffs
        let protos = Array3::<f32>::zeros((16, 16, num_protos)); // 32 protos

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        let result = decode_yolo_end_to_end_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        );

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(crate::DecoderError::InvalidShape(s)) if s.contains("doesn't match protos count")
        ));
    }

    // ========================================================================
    // Tests for decode_yolo_split_end_to_end_segdet_float
    // ========================================================================

    #[test]
    fn test_split_end_to_end_segdet_basic() {
        // Create synthetic segdet output: (6 + num_protos, N)
        // Detection format: [x1, y1, x2, y2, conf, class, mask_coeff_0..31]
        let num_protos = 32;
        let num_detections = 2;
        let num_features = 6 + num_protos;

        // Build detection tensor
        let mut data = vec![0.0f32; num_features * num_detections];
        // Detection 0: passes threshold
        data[0] = 0.1; // x1[0]
        data[1] = 0.5; // x1[1]
        data[num_detections] = 0.1; // y1[0]
        data[num_detections + 1] = 0.5; // y1[1]
        data[2 * num_detections] = 0.4; // x2[0]
        data[2 * num_detections + 1] = 0.9; // x2[1]
        data[3 * num_detections] = 0.4; // y2[0]
        data[3 * num_detections + 1] = 0.9; // y2[1]
        data[4 * num_detections] = 0.9; // conf[0] - passes
        data[4 * num_detections + 1] = 0.3; // conf[1] - fails
        data[5 * num_detections] = 1.0; // class[0]
        data[5 * num_detections + 1] = 2.0; // class[1]
                                            // Fill mask coefficients with small values
        for i in 6..num_features {
            data[i * num_detections] = 0.1;
            data[i * num_detections + 1] = 0.1;
        }

        let output = Array2::from_shape_vec((num_features, num_detections), data).unwrap();
        let box_coords = output.slice(s![..4, ..]);
        let scores = output.slice(s![4..5, ..]);
        let classes = output.slice(s![5..6, ..]);
        let mask_coeff = output.slice(s![6.., ..]);
        // Create protos tensor: (proto_height, proto_width, num_protos)
        let protos = Array3::<f32>::zeros((16, 16, num_protos));

        let mut boxes = Vec::with_capacity(10);
        let mut masks = Vec::with_capacity(10);
        decode_yolo_split_end_to_end_segdet_float(
            box_coords,
            scores,
            classes,
            mask_coeff,
            protos.view(),
            0.5,
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        // Only detection 0 should pass
        assert_eq!(boxes.len(), 1);
        assert_eq!(masks.len(), 1);
        assert_eq!(boxes[0].label, 1);
        assert!((boxes[0].score - 0.9).abs() < 0.01);
    }

    // ========================================================================
    // Tests for yolo_segmentation_to_mask
    // ========================================================================

    #[test]
    fn test_segmentation_to_mask_basic() {
        // Create a 4x4x1 segmentation with values above and below threshold
        let data: Vec<u8> = vec![
            100, 200, 50, 150, // row 0
            10, 255, 128, 64, // row 1
            0, 127, 128, 255, // row 2
            64, 64, 192, 192, // row 3
        ];
        let segmentation = Array3::from_shape_vec((4, 4, 1), data).unwrap();

        let mask = yolo_segmentation_to_mask(segmentation.view(), 128).unwrap();

        // Values >= 128 should be 1, others 0
        assert_eq!(mask[[0, 0]], 0); // 100 < 128
        assert_eq!(mask[[0, 1]], 1); // 200 >= 128
        assert_eq!(mask[[0, 2]], 0); // 50 < 128
        assert_eq!(mask[[0, 3]], 1); // 150 >= 128
        assert_eq!(mask[[1, 1]], 1); // 255 >= 128
        assert_eq!(mask[[1, 2]], 1); // 128 >= 128
        assert_eq!(mask[[2, 0]], 0); // 0 < 128
        assert_eq!(mask[[2, 1]], 0); // 127 < 128
    }

    #[test]
    fn test_segmentation_to_mask_all_above() {
        let segmentation = Array3::from_elem((4, 4, 1), 255u8);
        let mask = yolo_segmentation_to_mask(segmentation.view(), 128).unwrap();
        assert!(mask.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_segmentation_to_mask_all_below() {
        let segmentation = Array3::from_elem((4, 4, 1), 64u8);
        let mask = yolo_segmentation_to_mask(segmentation.view(), 128).unwrap();
        assert!(mask.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_segmentation_to_mask_invalid_shape() {
        let segmentation = Array3::from_elem((4, 4, 3), 128u8);
        let result = yolo_segmentation_to_mask(segmentation.view(), 128);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(crate::DecoderError::InvalidShape(s)) if s.contains("(H, W, 1)")
        ));
    }

    // ========================================================================
    // Tests for protobox / NORM_LIMIT regression
    // ========================================================================

    #[test]
    fn test_protobox_clamps_edge_coordinates() {
        // bbox with xmax=1.0 should not panic (OOB guard)
        let protos = Array3::<f32>::zeros((16, 16, 4));
        let view = protos.view();
        let roi = BoundingBox {
            xmin: 0.5,
            ymin: 0.5,
            xmax: 1.0,
            ymax: 1.0,
        };
        let result = protobox(&view, &roi);
        assert!(result.is_ok(), "protobox should accept xmax=1.0");
        let (cropped, _roi_norm) = result.unwrap();
        // Cropped region must have non-zero spatial dimensions
        assert!(cropped.shape()[0] > 0);
        assert!(cropped.shape()[1] > 0);
        assert_eq!(cropped.shape()[2], 4);
    }

    #[test]
    fn test_protobox_rejects_wildly_out_of_range() {
        // bbox with coords > NORM_LIMIT (e.g. 3.0) returns error
        let protos = Array3::<f32>::zeros((16, 16, 4));
        let view = protos.view();
        let roi = BoundingBox {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 3.0,
            ymax: 3.0,
        };
        let result = protobox(&view, &roi);
        assert!(
            matches!(result, Err(crate::DecoderError::InvalidShape(s)) if s.contains("un-normalized")),
            "protobox should reject coords > NORM_LIMIT"
        );
    }

    #[test]
    fn test_protobox_accepts_slightly_over_one() {
        // bbox with coords at 1.5 (within NORM_LIMIT=2.0) succeeds
        let protos = Array3::<f32>::zeros((16, 16, 4));
        let view = protos.view();
        let roi = BoundingBox {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.5,
            ymax: 1.5,
        };
        let result = protobox(&view, &roi);
        assert!(
            result.is_ok(),
            "protobox should accept coords <= NORM_LIMIT (2.0)"
        );
        let (cropped, _roi_norm) = result.unwrap();
        // Entire proto map should be selected when coords > 1.0 (clamped to boundary)
        assert_eq!(cropped.shape()[0], 16);
        assert_eq!(cropped.shape()[1], 16);
    }

    #[test]
    fn test_segdet_float_proto_no_panic() {
        // Simulates YOLOv8n-seg: output0 = [116, 8400] (4 box + 80 class + 32 mask coeff)
        // output1 (protos) = [32, 160, 160]
        let num_proposals = 100; // enough to produce idx >= 32
        let num_classes = 80;
        let num_mask_coeffs = 32;
        let rows = 4 + num_classes + num_mask_coeffs; // 116

        // Fill boxes with valid xywh data so some detections pass the threshold.
        // Layout is [116, num_proposals] row-major: row 0=cx, 1=cy, 2=w, 3=h,
        // rows 4..84=class scores, rows 84..116=mask coefficients.
        let mut data = vec![0.0f32; rows * num_proposals];
        for i in 0..num_proposals {
            let row = |r: usize| r * num_proposals + i;
            data[row(0)] = 320.0; // cx
            data[row(1)] = 320.0; // cy
            data[row(2)] = 50.0; // w
            data[row(3)] = 50.0; // h
            data[row(4)] = 0.9; // class-0 score
        }
        let boxes = ndarray::Array2::from_shape_vec((rows, num_proposals), data).unwrap();

        // Protos must be in HWC order. Under the HAL physical-order
        // contract, callers declare shape+dshape matching producer memory
        // and swap_axes_if_needed permutes the stride tuple into canonical
        // [batch, height, width, num_protos] before this function sees it.
        let protos = ndarray::Array3::<f32>::zeros((160, 160, num_mask_coeffs));

        let mut output_boxes = Vec::with_capacity(300);

        // This panicked before fix: mask_tensor.row(idx) with idx >= 32
        let proto_data = impl_yolo_segdet_float_proto::<XYWH, _, _>(
            boxes.view(),
            protos.view(),
            0.5,
            0.7,
            Some(Nms::default()),
            MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );

        // Should produce detections (NMS will collapse many overlapping boxes)
        assert!(!output_boxes.is_empty());
        let coeffs_shape = proto_data.mask_coefficients.shape();
        assert_eq!(coeffs_shape[0], output_boxes.len());
        // Each mask coefficient vector should have 32 elements
        assert_eq!(coeffs_shape[1], num_mask_coeffs);
    }

    // ========================================================================
    // Pre-NMS top-K cap (MAX_NMS_CANDIDATES)
    // ========================================================================

    /// At very low score thresholds (e.g., t=0.01 on YOLOv8 with 8400×80
    /// candidates) almost every score passes the filter, feeding O(n²)
    /// NMS and a per-survivor mask matmul. The decoder caps the
    /// candidate set fed to NMS at `MAX_NMS_CANDIDATES` (Ultralytics
    /// default 30 000) to bound worst-case decode time.
    ///
    /// This regression test pumps 50 000 above-threshold candidates
    /// into `impl_yolo_segdet_get_boxes` with NMS bypassed (Nms=None)
    /// and a generous post-NMS cap. Before the fix, the function
    /// returned all 50 000; after the fix, exactly 30 000.
    #[test]
    fn test_pre_nms_cap_truncates_excess_candidates() {
        let n: usize = 50_000;
        let num_classes = 1;

        // Identical valid boxes. Distinct scores (descending) so the
        // top-K cap keeps the highest-scoring ones in deterministic
        // order — letting us assert *which* ones survived.
        let mut boxes_data = Vec::with_capacity(n * 4);
        let mut scores_data = Vec::with_capacity(n * num_classes);
        for i in 0..n {
            boxes_data.extend_from_slice(&[0.1f32, 0.1, 0.5, 0.5]);
            // score_i = 0.99 - i * 1e-7 keeps everything well above 0.1
            // threshold but strictly decreasing.
            scores_data.push(0.99 - (i as f32) * 1e-7);
        }
        let boxes = Array2::from_shape_vec((n, 4), boxes_data).unwrap();
        let scores = Array2::from_shape_vec((n, num_classes), scores_data).unwrap();

        let result = impl_yolo_segdet_get_boxes::<XYXY, _, _>(
            boxes.view(),
            scores.view(),
            0.1,
            1.0,
            None,                            // bypass NMS so we measure the cap, not suppression
            crate::yolo::MAX_NMS_CANDIDATES, // pre_nms_top_k
            usize::MAX,                      // no post-NMS truncation
        );

        assert_eq!(
            result.len(),
            crate::yolo::MAX_NMS_CANDIDATES,
            "pre-NMS cap should truncate to MAX_NMS_CANDIDATES; got {}",
            result.len()
        );
        // Top-K survivors: highest scores were the first n indices,
        // so survivor 0 must have score ~0.99.
        let top_score = result[0].0.score;
        assert!(
            top_score > 0.98,
            "highest-ranked survivor should have the largest score, got {top_score}"
        );
    }

    /// Counterpart for the quantized split path. Same contract: feed
    /// more than `MAX_NMS_CANDIDATES` survivors above the quantized
    /// threshold, confirm `impl_yolo_split_segdet_quant_get_boxes`
    /// truncates before NMS.
    #[test]
    fn test_pre_nms_cap_truncates_excess_candidates_quant() {
        use crate::Quantization;
        let n: usize = 50_000;
        let num_classes = 1;

        // i8 boxes with simple scale/zp; the box value 50 dequantizes
        // to 0.5 with scale=0.01, zp=0 — fine for a flat box set.
        let boxes_data = (0..n).flat_map(|_| [10i8, 10, 50, 50]).collect::<Vec<_>>();
        let boxes = Array2::from_shape_vec((n, 4), boxes_data).unwrap();
        let quant_boxes = Quantization {
            scale: 0.01,
            zero_point: 0,
        };

        // u8 scores: distinct descending values, all well above threshold.
        // value 250 → 0.98 with scale 0.00392, zp 0.
        // value (250 - i % 200) keeps a wide spread above the dequant
        // threshold of 0.5.
        let scores_data: Vec<u8> = (0..n)
            .map(|i| 250u8.saturating_sub((i % 200) as u8))
            .collect();
        let scores = Array2::from_shape_vec((n, num_classes), scores_data).unwrap();
        let quant_scores = Quantization {
            scale: 0.00392,
            zero_point: 0,
        };

        let result = impl_yolo_split_segdet_quant_get_boxes::<XYXY, _, _>(
            (boxes.view(), quant_boxes),
            (scores.view(), quant_scores),
            0.1,
            1.0,
            None,
            crate::yolo::MAX_NMS_CANDIDATES, // pre_nms_top_k
            usize::MAX,                      // no post-NMS truncation
        );

        assert_eq!(
            result.len(),
            crate::yolo::MAX_NMS_CANDIDATES,
            "quant path pre-NMS cap should truncate to MAX_NMS_CANDIDATES; got {}",
            result.len()
        );
    }

    /// Regression test for HAILORT_BUG.md — the YoloSegDet path
    /// (combined `(4 + nc + nm, N)` detection tensor + separate protos)
    /// must pair each surviving detection with the mask coefficient
    /// row at the SAME anchor index the box came from. The validator
    /// sees this path miss the pairing under schema-v2 Hailo inputs
    /// (mAP collapse from 46.8 → 3.65 while mask IoU stays at 66.9,
    /// the fingerprint of mask-to-detection misalignment).
    ///
    /// Construction: three anchors with distinct mask-coef signatures
    /// that, after dot(coefs, protos) + sigmoid, produce HIGH vs LOW
    /// mask pixel values. Two anchors survive (one high, one low); if
    /// the mask row is looked up at the wrong index, the per-detection
    /// mean mask value would cross the threshold and we catch it.
    #[test]
    fn segdet_combined_tensor_pairs_detection_with_matching_mask_row() {
        let nc = 2; // num_classes
        let nm = 2; // num_protos
        let n = 3; // num_anchors
        let feat = 4 + nc + nm; // 8

        // Tensor layout: (8, 3) rows=features, cols=anchors.
        // Row indices:  0..4 = xywh, 4..6 = scores, 6..8 = mask_coefs.
        //
        //         anchor 0 | anchor 1 | anchor 2
        // xc       0.2      | 0.5      | 0.8
        // yc       0.2      | 0.5      | 0.8
        // w        0.1      | 0.1      | 0.1
        // h        0.1      | 0.1      | 0.1
        // s[0]     0.9      | 0.0      | 0.8   (class 0)
        // s[1]     0.0      | 0.0      | 0.0   (class 1 — always loses)
        // m[0]     3.0      | 0.0      | -3.0  (high for a0, low for a2)
        // m[1]     3.0      | 0.0      | -3.0
        //
        // Proto[0] = Proto[1] = all-ones (8x8), so
        //   mask(a0) = sigmoid(3 + 3) ≈ 0.9975 → 254
        //   mask(a2) = sigmoid(-3 + -3) ≈ 0.0025 → 1
        // 250-point gap makes any misalignment trivially detectable.
        let mut data = vec![0.0f32; feat * n];
        let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * n + c] = v;
        set(&mut data, 0, 0, 0.2);
        set(&mut data, 1, 0, 0.2);
        set(&mut data, 2, 0, 0.1);
        set(&mut data, 3, 0, 0.1);
        set(&mut data, 0, 1, 0.5);
        set(&mut data, 1, 1, 0.5);
        set(&mut data, 2, 1, 0.1);
        set(&mut data, 3, 1, 0.1);
        set(&mut data, 0, 2, 0.8);
        set(&mut data, 1, 2, 0.8);
        set(&mut data, 2, 2, 0.1);
        set(&mut data, 3, 2, 0.1);
        set(&mut data, 4, 0, 0.9);
        set(&mut data, 4, 2, 0.8);
        set(&mut data, 6, 0, 3.0);
        set(&mut data, 7, 0, 3.0);
        set(&mut data, 6, 2, -3.0);
        set(&mut data, 7, 2, -3.0);

        let output = Array2::from_shape_vec((feat, n), data).unwrap();
        let protos = Array3::<f32>::from_elem((8, 8, nm), 1.0);

        let mut boxes: Vec<DetectBox> = Vec::with_capacity(10);
        let mut masks: Vec<Segmentation> = Vec::with_capacity(10);
        decode_yolo_segdet_float(
            output.view(),
            protos.view(),
            0.5,
            0.5,
            Some(Nms::ClassAgnostic),
            &mut boxes,
            &mut masks,
        )
        .unwrap();

        assert_eq!(
            boxes.len(),
            2,
            "two anchors above threshold should survive (a0 score=0.9, a2 score=0.8); got {}",
            boxes.len()
        );

        // Build a (anchor_index → mask_mean) mapping from the results.
        // Anchor 0 has centre (0.2, 0.2), anchor 2 has centre (0.8,
        // 0.8). The DetectBox bbox is the post-XYWH-to-XYXY conversion
        // of the original xywh; cropping inside protobox may shrink it,
        // so match by centre (0.2 vs 0.8) rather than exact bbox.
        for (b, m) in boxes.iter().zip(masks.iter()) {
            let cx = (b.bbox.xmin + b.bbox.xmax) * 0.5;
            let mean = {
                let s = &m.segmentation;
                let total: u32 = s.iter().map(|&v| v as u32).sum();
                total as f32 / s.len() as f32
            };
            if cx < 0.3 {
                // anchor 0 — expect HIGH mask values ≈ 254
                assert!(
                    mean > 200.0,
                    "anchor 0 detection (centre {cx:.2}) should have high-value mask; got mean {mean}"
                );
            } else if cx > 0.7 {
                // anchor 2 — expect LOW mask values ≈ 1
                assert!(
                    mean < 50.0,
                    "anchor 2 detection (centre {cx:.2}) should have low-value mask; got mean {mean}"
                );
            } else {
                panic!("unexpected detection centre {cx:.2}");
            }
        }
    }
}
