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

/// Dispatches to the appropriate NMS function based on mode for float boxes with extra data.
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

/// Dispatches to the appropriate NMS function based on mode for quantized boxes.
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

/// Dispatches to the appropriate NMS function based on mode for quantized boxes with extra data.
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
///
/// Input shape: (N, 6+) where columns are [x1, y1, x2, y2, conf, class, ...]
/// Boxes are output directly without NMS (model already applied NMS).
///
/// Coordinates may be normalized [0,1] or pixel values depending on model config.
/// The caller should check `decoder.normalized_boxes()` to determine which.
pub fn decode_yolo_end_to_end_det_float<T>(
    output: ArrayView2<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
{
    let num_cols = output.ncols();
    if num_cols < 6 {
        // Invalid shape, need at least [x1, y1, x2, y2, conf, class]
        return;
    }

    output_boxes.clear();
    for row in output.rows() {
        let conf: f32 = row[4].as_();
        if conf < score_threshold {
            continue;
        }
        if output_boxes.len() >= output_boxes.capacity() {
            break;
        }
        output_boxes.push(crate::DetectBox {
            bbox: crate::BoundingBox {
                xmin: row[0].as_(),
                ymin: row[1].as_(),
                xmax: row[2].as_(),
                ymax: row[3].as_(),
            },
            score: conf,
            label: row[5].as_() as usize,
        });
    }
    // No NMS — model output is already post-NMS
}

/// Decodes end-to-end YOLO detection + segmentation outputs (post-NMS from model).
///
/// Input shapes:
/// - detection: (N, 6 + num_protos) where columns are
///   [x1, y1, x2, y2, conf, class, mask_coeff_0, ..., mask_coeff_31]
/// - protos: (proto_height, proto_width, num_protos)
///
/// Boxes are output directly without NMS (model already applied NMS).
/// Coordinates may be normalized [0,1] or pixel values depending on model config.
pub fn decode_yolo_end_to_end_segdet_float<T>(
    output: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<crate::Segmentation>,
) where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
{
    let num_protos = protos.dim().2;
    let min_cols = 6 + num_protos;
    let num_cols = output.ncols();
    if num_cols < min_cols {
        // Invalid shape
        return;
    }

    output_boxes.clear();
    output_masks.clear();

    // Collect boxes with mask coefficients
    let mut boxes_with_masks: Vec<(crate::DetectBox, Vec<f32>)> = Vec::new();

    for row in output.rows() {
        let conf: f32 = row[4].as_();
        if conf < score_threshold {
            continue;
        }
        if boxes_with_masks.len() >= output_boxes.capacity() {
            break;
        }

        let bbox = crate::BoundingBox {
            xmin: row[0].as_(),
            ymin: row[1].as_(),
            xmax: row[2].as_(),
            ymax: row[3].as_(),
        };

        let detect_box = crate::DetectBox {
            bbox,
            score: conf,
            label: row[5].as_() as usize,
        };

        // Extract mask coefficients
        let mask_coeffs: Vec<f32> = (6..6 + num_protos).map(|i| row[i].as_()).collect();

        boxes_with_masks.push((detect_box, mask_coeffs));
    }

    // Generate segmentation masks
    for (detect_box, mask_coeffs) in boxes_with_masks {
        let mask_arr = ndarray::Array1::from_vec(mask_coeffs);
        let (protos_crop, roi) = protobox(&protos, &detect_box.bbox);

        let seg = make_segmentation_from_coeffs(mask_arr.view(), protos_crop.view());

        let final_box = crate::DetectBox {
            bbox: roi,
            score: detect_box.score,
            label: detect_box.label,
        };

        output_boxes.push(final_box);
        output_masks.push(crate::Segmentation {
            xmin: roi.xmin,
            ymin: roi.ymin,
            xmax: roi.xmax,
            ymax: roi.ymax,
            segmentation: seg,
        });
    }
    // No NMS — model output is already post-NMS
}

/// Helper to compute segmentation mask from coefficients and protos.
fn make_segmentation_from_coeffs<T>(
    mask_coeffs: ArrayView1<f32>,
    protos: ArrayView3<T>,
) -> Array3<u8>
where
    T: Float + AsPrimitive<f32> + Send + Sync,
{
    let shape = protos.shape();
    let mask = mask_coeffs.to_shape((1, mask_coeffs.len())).unwrap();
    let protos_2d = protos
        .to_shape([shape[0] * shape[1], shape[2]])
        .unwrap();
    let protos_2d = protos_2d.reversed_axes();
    let protos_f32 = protos_2d.map(|x| x.as_());

    let result = mask
        .dot(&protos_f32)
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    let min = *result.min().unwrap_or(&0.0);
    let max = *result.max().unwrap_or(&1.0);
    let max = max.max(-min);
    let min_val = -max;
    let range = max - min_val;
    // Guard against division by zero for uniform masks
    if range == 0.0 {
        return result.map(|_| 128u8);
    }
    let u8_max = 256.0;
    result.map(|x| ((*x - min_val) / range * u8_max) as u8)
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
/// # Panics
/// Panics if the input segmentation does not have shape (H, W, 1).
pub fn yolo_segmentation_to_mask(segmentation: ArrayView3<u8>, threshold: u8) -> Array2<u8> {
    assert_eq!(
        segmentation.shape()[2],
        1,
        "Yolo Instance Segmentation should have shape (H, W, 1)"
    );
    segmentation
        .slice(s![.., .., 0])
        .map(|x| if *x >= threshold { 1 } else { 0 })
}
