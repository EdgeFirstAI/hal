// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use ndarray::{
    Array2, Array3, ArrayView1, ArrayView2, ArrayView3,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s,
};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float, PrimInt, Signed};

use crate::{
    BBoxTypeTrait, BoundingBox, DetectBox, DetectBoxQuantized, Quantization, Segmentation, XYWH,
    XYXY,
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
fn dispatch_nms_extra_float<E: Send + Sync>(
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
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
) where
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
    );
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_yolo_segdet_float<T>(
    boxes: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<Nms>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) where
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
    );
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
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
) where
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
    );
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
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
) where
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
    );
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
    // Validate input shape: need at least 7 rows (6 base + at least 1 mask coeff)
    if output.shape()[0] < 7 {
        return Err(crate::DecoderError::InvalidShape(format!(
            "End-to-end segdet output requires at least 7 rows, got {}",
            output.shape()[0]
        )));
    }

    let num_mask_coeffs = output.shape()[0] - 6;
    let num_protos = protos.shape()[2];
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
    let mut boxes =
        postprocess_boxes_index_float::<XYXY, _, _>(score_threshold.as_(), boxes, scores);
    boxes.truncate(output_boxes.capacity());

    for (b, ind) in &mut boxes {
        b.label = classes[*ind].as_() as usize;
    }

    // No NMS — model output is already post-NMS

    let boxes = decode_segdet_f32(boxes, mask_coeff, protos);

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
/// Internal implementation of YOLO decoding for quantized tensors.
///
/// Expected shapes of inputs:
/// - output: (4 + num_classes, num_boxes)
pub fn impl_yolo_quant<B: BBoxTypeTrait, T: PrimInt + AsPrimitive<f32> + Send + Sync>(
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
pub fn impl_yolo_float<B: BBoxTypeTrait, T: Float + AsPrimitive<f32> + Send + Sync>(
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
pub fn impl_yolo_split_quant<
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
pub fn impl_yolo_split_float<
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn impl_yolo_segdet_quant<
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
) where
    f32: AsPrimitive<BOX>,
{
    let (boxes, quant_boxes) = boxes;
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = impl_yolo_split_segdet_quant_get_boxes::<B, _, _>(
        (boxes_tensor.reversed_axes(), quant_boxes),
        (scores_tensor.reversed_axes(), quant_boxes),
        score_threshold,
        iou_threshold,
        nms,
        output_boxes.capacity(),
    );

    impl_yolo_split_segdet_quant_process_masks::<_, _>(
        boxes,
        (mask_tensor.reversed_axes(), quant_boxes),
        protos,
        output_boxes,
        output_masks,
    );
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
pub fn impl_yolo_segdet_float<
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
) where
    f32: AsPrimitive<BOX>,
{
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = postprocess_boxes_index_float::<B, _, _>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
    );
    let mut boxes = dispatch_nms_extra_float(nms, iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = decode_segdet_f32(boxes, mask_tensor, protos);
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
}

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

    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();

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
) {
    let (masks, quant_masks) = mask_coeff;
    let (protos, quant_protos) = protos;

    let masks = masks.reversed_axes();

    let boxes = decode_segdet_quant(boxes, masks, protos, quant_masks, quant_protos);
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn impl_yolo_split_segdet_quant<
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
) where
    f32: AsPrimitive<SCORE>,
{
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
    );
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
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn impl_yolo_split_segdet_float<
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
) where
    f32: AsPrimitive<SCORE>,
{
    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let mask_tensor = mask_tensor.reversed_axes();

    let boxes = postprocess_boxes_index_float::<B, _, _>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
    );
    let mut boxes = dispatch_nms_extra_float(nms, iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = decode_segdet_f32(boxes, mask_tensor, protos);
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
}

fn postprocess_yolo<'a, T>(
    output: &'a ArrayView2<'_, T>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>) {
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4.., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor)
}

fn postprocess_yolo_seg<'a, T>(
    output: &'a ArrayView2<'_, T>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>, ArrayView2<'a, T>) {
    assert!(output.shape()[0] > 32 + 4, "Output shape is too short");
    let num_classes = output.shape()[0] - 4 - 32;
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]).reversed_axes();
    let mask_tensor = output.slice(s![(num_classes + 4).., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor, mask_tensor)
}

fn decode_segdet_f32<
    MASK: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes: Vec<(DetectBox, usize)>,
    masks: ArrayView2<MASK>,
    protos: ArrayView3<PROTO>,
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    assert!(masks.shape()[1] == protos.shape()[2]);
    boxes
        .into_par_iter()
        .map(|mut b| {
            let ind = b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox);
            b.0.bbox = roi;
            (b.0, make_segmentation(masks.row(ind), protos.view()))
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
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    assert!(masks.shape()[1] == protos.shape()[2]);

    let total_bits = MASK::zero().count_zeros() + PROTO::zero().count_zeros() + 5; // 32 protos is 2^5
    boxes
        .into_iter()
        .map(|mut b| {
            let i = b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox.to_canonical());
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
                _ => panic!("Unsupported bit width for segmentation computation"),
            };
            (b.0, seg)
        })
        .collect()
}

fn protobox<'a, T>(
    protos: &'a ArrayView3<T>,
    roi: &BoundingBox,
) -> (ArrayView3<'a, T>, BoundingBox) {
    let width = protos.dim().1 as f32;
    let height = protos.dim().0 as f32;

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

    (cropped, roi_norm)
}

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

    let min = *mask.min().unwrap_or(&0.0);
    let max = *mask.max().unwrap_or(&1.0);
    let max = max.max(-min);
    let min = -max;
    let u8_max = 256.0;
    mask.map(|x| ((*x - min) / (max - min) * u8_max) as u8)
}

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

    let min = *segmentation.min().unwrap_or(&DEST::zero());
    let max = *segmentation.max().unwrap_or(&DEST::one());
    let max = max.max(-min);
    let min = -max;
    segmentation.map(|x| ((*x - min).as_() / (max - min).as_() * 256.0) as u8)
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
}
