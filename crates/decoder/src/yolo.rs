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
    BBoxTypeTrait, BoundingBox, DetectBox, DetectBoxQuantized, ProtoData, ProtoTensor,
    Quantization, Segmentation, XYWH, XYXY,
};

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
    max_boxes: usize,
) -> Vec<(DetectBox, usize)>
where
    f32: AsPrimitive<SCORE>,
{
    let boxes = postprocess_boxes_index_float::<B, _, _>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
    );
    let mut boxes = dispatch_nms_extra_float(nms, iou_threshold, boxes);
    boxes.truncate(max_boxes);
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
    max_boxes: usize,
) -> Vec<(DetectBox, usize)>
where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
        postprocess_boxes_index_quant::<B, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant_boxes,
        )
    };
    let mut boxes = dispatch_nms_extra_int(nms, iou_threshold, boxes);
    boxes.truncate(max_boxes);
    boxes
        .into_iter()
        .map(|(b, i)| (dequant_detect_box(&b, quant_scores), i))
        .collect()
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
        output_boxes.capacity(),
    );
    impl_yolo_split_segdet_process_masks(boxes, mask_tensor, protos, output_boxes, output_masks)
}

// ---------------------------------------------------------------------------
// Proto-extraction variants: return ProtoData instead of materialized masks
// ---------------------------------------------------------------------------

/// Proto-extraction variant of `impl_yolo_segdet_quant`.
/// Runs NMS but returns raw `ProtoData` instead of materialized masks.
pub fn impl_yolo_segdet_quant_proto<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + AsPrimitive<f32> + Send + Sync,
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
        output_boxes.capacity(),
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
pub(crate) fn impl_yolo_segdet_float_proto<
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
        output_boxes.capacity(),
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
        output_boxes.capacity(),
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

    Ok(extract_proto_data_float(
        boxes,
        mask_coeff,
        protos,
        output_boxes,
    ))
}

/// Helper: extract ProtoData from float mask coefficients + protos.
pub(super) fn extract_proto_data_float<
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    det_indices: Vec<(DetectBox, usize)>,
    mask_tensor: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData {
    let mut mask_coefficients = Vec::with_capacity(det_indices.len());
    output_boxes.clear();
    for (det, idx) in det_indices {
        output_boxes.push(det);
        let row = mask_tensor.row(idx);
        mask_coefficients.push(row.iter().map(|v| v.as_()).collect());
    }
    let protos_f32 = protos.map(|v| v.as_());
    ProtoData {
        mask_coefficients,
        protos: ProtoTensor::Float(protos_f32),
    }
}

/// Helper: extract ProtoData from quantized mask coefficients + protos.
///
/// Dequantizes mask coefficients to f32 (small — per-detection) but keeps
/// protos in raw int8 form wrapped in `ProtoTensor::Quantized` so the GPU
/// shader can dequantize per-texel without CPU overhead.
pub(crate) fn extract_proto_data_quant<
    MASK: PrimInt + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<f32> + AsPrimitive<i8> + Send + Sync + 'static,
>(
    det_indices: Vec<(DetectBox, usize)>,
    mask_tensor: ArrayView2<MASK>,
    quant_masks: Quantization,
    protos: ArrayView3<PROTO>,
    quant_protos: Quantization,
    output_boxes: &mut Vec<DetectBox>,
) -> ProtoData {
    let mut mask_coefficients = Vec::with_capacity(det_indices.len());
    output_boxes.clear();
    for (det, idx) in det_indices {
        output_boxes.push(det);
        let row = mask_tensor.row(idx);
        mask_coefficients.push(
            row.iter()
                .map(|v| (v.as_() - quant_masks.zero_point as f32) * quant_masks.scale)
                .collect(),
        );
    }
    // Keep protos in raw int8 — GPU shader will dequantize per-texel.
    // When PROTO is already i8, use to_owned() for a flat memcpy instead of
    // per-element as_() conversion.
    let protos_i8 = if std::any::TypeId::of::<PROTO>() == std::any::TypeId::of::<i8>() {
        // SAFETY: PROTO and i8 have identical size and layout when TypeId matches.
        let view_i8 =
            unsafe { &*(&protos as *const ArrayView3<'_, PROTO> as *const ArrayView3<'_, i8>) };
        view_i8.to_owned()
    } else {
        protos.map(|v| {
            let v_i8: i8 = v.as_();
            v_i8
        })
    };
    ProtoData {
        mask_coefficients,
        protos: ProtoTensor::Quantized {
            protos: protos_i8,
            quantization: quant_protos,
        },
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

        // Protos must be in HWC order (decoder.rs protos_to_hwc converts
        // before calling into these functions).
        let protos = ndarray::Array3::<f32>::zeros((160, 160, num_mask_coeffs));

        let mut output_boxes = Vec::with_capacity(300);

        // This panicked before fix: mask_tensor.row(idx) with idx >= 32
        let proto_data = impl_yolo_segdet_float_proto::<XYWH, _, _>(
            boxes.view(),
            protos.view(),
            0.5,
            0.7,
            Some(Nms::default()),
            &mut output_boxes,
        );

        // Should produce detections (NMS will collapse many overlapping boxes)
        assert!(!output_boxes.is_empty());
        assert_eq!(proto_data.mask_coefficients.len(), output_boxes.len());
        // Each mask coefficient vector should have 32 elements
        for coeffs in &proto_data.mask_coefficients {
            assert_eq!(coeffs.len(), num_mask_coeffs);
        }
    }
}
