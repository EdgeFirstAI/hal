use std::fmt::Debug;

use ndarray::{
    Array2, Array3, ArrayView1, ArrayView2, ArrayView3,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s,
};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float, PrimInt, Signed};

use crate::{
    BBoxTypeTrait, BoundingBox, DetectBox, Quantization, Segmentation, XYWH,
    byte::{
        nms_extra_int, nms_int, postprocess_boxes_8bit, postprocess_boxes_extra_8bit,
        quantize_score_threshold,
    },
    dequant_detect_box,
    float::{nms_extra_f32, nms_f32, postprocess_boxes_extra_float, postprocess_boxes_float},
};

pub fn decode_yolo_u8(
    output: (ArrayView2<u8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_8bit::<XYWH, _>(output, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_i8(
    output: (ArrayView2<i8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_8bit::<XYWH, _>(output, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_f32(
    output: ArrayView2<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_float::<XYWH, _>(output, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_f64(
    output: ArrayView2<f64>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_float::<XYWH, _>(output, score_threshold, iou_threshold, output_boxes);
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_segdet_i8(
    boxes: (ArrayView2<i8>, Quantization),
    protos: (ArrayView3<i8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_segdet_8bit::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_segdet_u8(
    boxes: (ArrayView2<u8>, Quantization),
    protos: (ArrayView3<u8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_segdet_8bit::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn decode_yolo_segdet_f32(
    boxes: ArrayView2<f32>,
    protos: ArrayView3<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_segdet_float::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn decode_yolo_segdet_f64(
    boxes: ArrayView2<f64>,
    protos: ArrayView3<f64>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_segdet_float::<XYWH, _, _>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn decode_yolo_split_det_u8(
    boxes: (ArrayView2<u8>, Quantization),
    scores: (ArrayView2<u8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_split_8bit::<XYWH, _, _>(boxes, scores, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_split_det_i8(
    boxes: (ArrayView2<i8>, Quantization),
    scores: (ArrayView2<i8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_split_8bit::<XYWH, _, _>(boxes, scores, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_split_det_f32(
    boxes: ArrayView2<f32>,
    scores: ArrayView2<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_split_float::<XYWH, _, _>(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        output_boxes,
    );
}

pub fn decode_yolo_split_det_f64(
    boxes: ArrayView2<f64>,
    scores: ArrayView2<f64>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_split_float::<XYWH, _, _>(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        output_boxes,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet_u8(
    boxes: (ArrayView2<u8>, Quantization),
    scores: (ArrayView2<u8>, Quantization),
    mask_coeff: (ArrayView2<u8>, Quantization),
    protos: (ArrayView3<u8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_split_segdet_8bit::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet_i8(
    boxes: (ArrayView2<i8>, Quantization),
    scores: (ArrayView2<i8>, Quantization),
    mask_coeff: (ArrayView2<i8>, Quantization),
    protos: (ArrayView3<i8>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_split_segdet_8bit::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet_f32(
    boxes: ArrayView2<f32>,
    scores: ArrayView2<f32>,
    mask_coeff: ArrayView2<f32>,
    protos: ArrayView3<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_split_segdet_float::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_split_segdet_f64(
    boxes: ArrayView2<f64>,
    scores: ArrayView2<f64>,
    mask_coeff: ArrayView2<f64>,
    protos: ArrayView3<f64>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    impl_yolo_split_segdet_float::<XYWH, _, _, _, _>(
        boxes,
        scores,
        mask_coeff,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn impl_yolo_8bit<
    B: BBoxTypeTrait,
    T: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
>(
    output: (ArrayView2<T>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes, quant_boxes) = output;
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&boxes);

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
        postprocess_boxes_8bit::<B, _, _>(score_threshold, boxes_tensor, scores_tensor, quant_boxes)
    };

    let boxes = nms_int::<_, _, i32>(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant_boxes, quant_boxes));
    }
}

pub fn impl_yolo_float<B: BBoxTypeTrait, T: Float + AsPrimitive<f32> + Send + Sync>(
    output: ArrayView2<T>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<T>,
{
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes =
        postprocess_boxes_float::<B, _, _>(score_threshold.as_(), boxes_tensor, scores_tensor);
    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn impl_yolo_split_8bit<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;

    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
        postprocess_boxes_8bit::<B, _, _>(score_threshold, boxes_tensor, scores_tensor, quant_boxes)
    };

    let boxes = nms_int::<_, _, i32>(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant_boxes, quant_boxes));
    }
}

pub fn impl_yolo_split_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let boxes =
        postprocess_boxes_float::<B, _, _>(score_threshold.as_(), boxes_tensor, scores_tensor);
    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn impl_yolo_segdet_8bit<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    let (boxes, quant_boxes) = boxes;
    let (protos, quant_protos) = protos;

    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
        postprocess_boxes_extra_8bit::<B, _, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            &mask_tensor,
            quant_boxes,
        )
    };

    let mut boxes = nms_extra_int::<_, _, i32, _>(iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = boxes
        .into_iter()
        .map(|(b, m)| (dequant_detect_box(&b, quant_boxes, quant_boxes), m))
        .collect();
    let boxes = decode_segdet_8bit(boxes, protos, quant_boxes, quant_protos);
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

pub fn impl_yolo_segdet_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    PROTO: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes: ArrayView2<BOX>,
    protos: ArrayView3<PROTO>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) where
    f32: AsPrimitive<BOX>,
{
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = postprocess_boxes_extra_float::<B, _, _, _>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
        &mask_tensor,
    );
    let mut boxes = nms_extra_f32(iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = decode_segdet_f32(boxes, protos);
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
pub fn impl_yolo_split_segdet_8bit<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    MASK: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    mask_coeff: (ArrayView2<MASK>, Quantization),
    protos: (ArrayView3<PROTO>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) {
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;
    let (mask_tensor, quant_masks) = mask_coeff;
    let (protos, quant_protos) = protos;

    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let mask_tensor = mask_tensor.reversed_axes();

    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
        postprocess_boxes_extra_8bit::<B, _, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            &mask_tensor,
            quant_boxes,
        )
    };

    let mut boxes = nms_extra_int::<_, _, i32, _>(iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = boxes
        .into_iter()
        .map(|(b, m)| (dequant_detect_box(&b, quant_boxes, quant_scores), m))
        .collect();
    let boxes = decode_segdet_8bit(boxes, protos, quant_masks, quant_protos);
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
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
) where
    f32: AsPrimitive<SCORE>,
{
    let boxes_tensor = boxes_tensor.reversed_axes();
    let scores_tensor = scores_tensor.reversed_axes();
    let mask_tensor = mask_tensor.reversed_axes();

    let boxes = postprocess_boxes_extra_float::<B, BOX, SCORE, MASK>(
        score_threshold.as_(),
        boxes_tensor,
        scores_tensor,
        &mask_tensor,
    );
    let mut boxes = nms_extra_f32(iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = decode_segdet_f32(boxes, protos);
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
    boxes: Vec<(DetectBox, ArrayView1<MASK>)>,
    protos: ArrayView3<PROTO>,
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    boxes
        .into_par_iter()
        .map(|mut b| {
            let mask = &b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox);
            b.0.bbox = roi;
            (b.0, make_segmentation(mask.view(), protos.view()))
        })
        .collect()
}

fn decode_segdet_8bit<
    MASK: PrimInt + AsPrimitive<i32> + Send + Sync,
    PROTO: PrimInt + AsPrimitive<i32> + Send + Sync,
>(
    boxes: Vec<(DetectBox, ArrayView1<MASK>)>,
    protos: ArrayView3<PROTO>,
    quant_masks: Quantization,
    quant_protos: Quantization,
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    boxes
        .into_par_iter()
        .map(|mut b| {
            let mask = &b.1;
            let (protos, roi) = protobox(&protos, &b.0.bbox.to_canonical());
            b.0.bbox = roi;
            (
                b.0,
                make_segmentation_8bit(mask.view(), protos.view(), quant_masks, quant_protos),
            )
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
    let mask = mask.to_shape((1, mask.len())).unwrap();
    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();
    let mask = mask.map(|x| x.as_());
    let protos = protos.map(|x| x.as_());
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

fn make_segmentation_8bit<
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
{
    let shape = protos.shape();
    let mask = mask.to_shape((1, mask.len())).unwrap();

    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();

    let zp = quant_masks.zero_point.as_();
    // This cannot overflow because in the dot product, we have u8 * u8 and then 32
    // proto layers are summed together. Which means in the result of the matrix
    // multiply, the maximum value of a single element is 256 * 256 * 32 = 2^21
    let mask = mask.mapv(|x| x.as_() - zp);

    let zp = quant_protos.zero_point.as_();
    let protos = protos.mapv(|x| x.as_() - zp);

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
