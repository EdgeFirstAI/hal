use ndarray::{
    Array3, ArrayView1, ArrayView2, ArrayView3,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s,
};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float, PrimInt};

use crate::{
    BBoxTypeTrait, DetectBox, Quantization, SegmentationMask, XYWH,
    byte::{
        nms_extra_i16, nms_i16, postprocess_boxes_8bit, postprocess_boxes_extra_8bit,
        quantize_score_threshold,
    },
    dequant_detect_box,
    float::{nms_extra_f32, nms_f32, postprocess_boxes_extra_float, postprocess_boxes_float},
};

pub fn decode_yolo_u8(
    output: ArrayView2<u8>,
    quant: &Quantization<u8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_8bit::<XYWH, u8>(output, quant, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_i8(
    output: ArrayView2<i8>,
    quant: &Quantization<i8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_8bit::<XYWH, i8>(output, quant, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_f32(
    output: ArrayView2<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_float::<XYWH, f32>(output, score_threshold, iou_threshold, output_boxes);
}

pub fn decode_yolo_f64(
    output: ArrayView2<f64>,
    score_threshold: f64,
    iou_threshold: f64,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_yolo_float::<XYWH, f64>(output, score_threshold, iou_threshold, output_boxes);
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_masks_i8(
    boxes: ArrayView2<i8>,
    protos: ArrayView3<i8>,
    quant_boxes: &Quantization<i8>,
    quant_protos: &Quantization<i8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    impl_yolo_masks_8bit::<XYWH, i8>(
        boxes,
        protos,
        quant_boxes,
        quant_protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn decode_yolo_masks_u8(
    boxes: ArrayView2<u8>,
    protos: ArrayView3<u8>,
    quant_boxes: &Quantization<u8>,
    quant_protos: &Quantization<u8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    impl_yolo_masks_8bit::<XYWH, u8>(
        boxes,
        protos,
        quant_boxes,
        quant_protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn decode_yolo_masks_f32(
    boxes: ArrayView2<f32>,
    protos: ArrayView3<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    impl_yolo_masks_float::<XYWH, f32>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn decode_yolo_masks_f64(
    boxes: ArrayView2<f64>,
    protos: ArrayView3<f64>,
    score_threshold: f64,
    iou_threshold: f64,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    impl_yolo_masks_float::<XYWH, f64>(
        boxes,
        protos,
        score_threshold,
        iou_threshold,
        output_boxes,
        output_masks,
    );
}

pub fn impl_yolo_8bit<
    B: BBoxTypeTrait,
    T: PrimInt + AsPrimitive<i16> + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
>(
    output: ArrayView2<T>,
    quant: &Quantization<T>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = quantize_score_threshold(score_threshold, quant);
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_8bit::<B, _>(score_threshold, boxes_tensor, scores_tensor, quant);
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant, quant));
    }
}

pub fn impl_yolo_float<B: BBoxTypeTrait, T: Float + AsPrimitive<f32> + Send + Sync>(
    output: ArrayView2<T>,
    score_threshold: T,
    iou_threshold: T,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_float::<B, T>(score_threshold, boxes_tensor, scores_tensor);
    let boxes = nms_f32(iou_threshold.as_(), boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn impl_yolo_masks_8bit<
    B: BBoxTypeTrait,
    T: PrimInt + AsPrimitive<i16> + AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync,
>(
    boxes: ArrayView2<T>,
    protos: ArrayView3<T>,
    quant_boxes: &Quantization<T>,
    quant_protos: &Quantization<T>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = postprocess_boxes_extra_8bit::<B, T, T>(
        score_threshold,
        boxes_tensor,
        scores_tensor,
        &mask_tensor,
        quant_boxes,
    );
    let mut boxes = nms_extra_i16(iou_threshold, boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = boxes
        .into_iter()
        .map(|(b, m)| (dequant_detect_box(&b, quant_boxes, quant_boxes), m))
        .collect();
    let boxes = decode_masks_8bit(boxes, protos, quant_boxes, quant_protos);
    output_boxes.clear();
    output_masks.clear();
    for (b, m) in boxes.into_iter() {
        output_boxes.push(b);
        output_masks.push(SegmentationMask {
            xmin: b.xmin,
            ymin: b.ymin,
            xmax: b.xmax,
            ymax: b.ymax,
            mask: m,
        });
    }
}

pub fn impl_yolo_masks_float<
    B: BBoxTypeTrait,
    T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync,
>(
    boxes: ArrayView2<T>,
    protos: ArrayView3<T>,
    score_threshold: T,
    iou_threshold: T,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&boxes);

    let boxes = postprocess_boxes_extra_float::<B, T, T>(
        score_threshold,
        boxes_tensor,
        scores_tensor,
        &mask_tensor,
    );
    let mut boxes = nms_extra_f32(iou_threshold.as_(), boxes);
    boxes.truncate(output_boxes.capacity());
    let boxes = decode_masks_f32(boxes, protos);
    output_boxes.clear();
    output_masks.clear();
    for (b, m) in boxes.into_iter() {
        output_boxes.push(b);
        output_masks.push(SegmentationMask {
            xmin: b.xmin,
            ymin: b.ymin,
            xmax: b.xmax,
            ymax: b.ymax,
            mask: m,
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

fn decode_masks_f32<T: Float + Send + Sync + AsPrimitive<u8>>(
    boxes: Vec<(DetectBox, ArrayView1<T>)>,
    protos: ArrayView3<T>,
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    boxes
        .into_par_iter()
        .map(|mut b| {
            let mask = &b.1;
            let (protos, roi) = protobox(&protos, &[b.0.xmin, b.0.ymin, b.0.xmax, b.0.ymax]);
            b.0.xmin = roi[0];
            b.0.ymin = roi[1];
            b.0.xmax = roi[2];
            b.0.ymax = roi[3];
            (b.0, make_mask(mask.view(), protos.view()))
        })
        .collect()
}

fn decode_masks_8bit<T: AsPrimitive<i32> + AsPrimitive<f32> + Send + Sync>(
    boxes: Vec<(DetectBox, ArrayView1<T>)>,
    protos: ArrayView3<T>,
    quant_boxes: &Quantization<T>,
    quant_protos: &Quantization<T>,
) -> Vec<(DetectBox, Array3<u8>)> {
    if boxes.is_empty() {
        return Vec::new();
    }
    boxes
        .into_par_iter()
        .map(|mut b| {
            let mask = &b.1;
            let (protos, roi) = protobox(&protos, &[b.0.xmin, b.0.ymin, b.0.xmax, b.0.ymax]);
            b.0.xmin = roi[0];
            b.0.ymin = roi[1];
            b.0.xmax = roi[2];
            b.0.ymax = roi[3];
            (
                b.0,
                mask_mask_8bit(mask.view(), protos.view(), quant_boxes, quant_protos),
            )
        })
        .collect()
}

fn protobox<'a, T>(protos: &'a ArrayView3<T>, roi: &[f32; 4]) -> (ArrayView3<'a, T>, [f32; 4]) {
    let width = protos.dim().1 as f32;
    let height = protos.dim().0 as f32;

    let roi = [
        (roi[0] * width - 0.5).clamp(0.0, width) as usize,
        (roi[1] * height - 0.5).clamp(0.0, height) as usize,
        (roi[2] * width + 0.5).clamp(0.0, width).ceil() as usize,
        (roi[3] * height + 0.5).clamp(0.0, height).ceil() as usize,
    ];

    let roi_norm = [
        roi[0] as f32 / width,
        roi[1] as f32 / height,
        roi[2] as f32 / width,
        roi[3] as f32 / height,
    ];

    let shape = [(roi[3] - roi[1]), (roi[2] - roi[0]), protos.dim().2];
    if shape[0] * shape[1] * shape[2] == 0 {
        return (protos.slice(s![0..0, 0..0, ..]), roi_norm);
    }

    let cropped = protos.slice(s![roi[1]..roi[3], roi[0]..roi[2], ..]);

    (cropped, roi_norm)
}

fn make_mask<T: Float + Send + Sync + AsPrimitive<u8>>(
    mask: ArrayView1<T>,
    protos: ArrayView3<T>,
) -> Array3<u8> {
    let shape = protos.shape();
    let mask = mask.to_shape((1, mask.len())).unwrap();
    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();
    let mask = mask
        .dot(&protos)
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    let min = *mask.min().unwrap_or(&T::zero());
    let max = *mask.max().unwrap_or(&T::one());
    let u8_max = T::from(255.0).unwrap();
    mask.map(|x| ((*x - min) / max * u8_max).as_())
}

fn mask_mask_8bit<T: AsPrimitive<i32> + AsPrimitive<f32>>(
    mask: ArrayView1<T>,
    protos: ArrayView3<T>,
    quant_boxes: &Quantization<T>,
    quant_protos: &Quantization<T>,
) -> Array3<u8> {
    let shape = protos.shape();
    let mask = mask.to_shape((1, mask.len())).unwrap();

    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();

    let to_i32 = |x| <T as num_traits::AsPrimitive<i32>>::as_(x);
    // This cannot overflow because in the dot product, we have u8 * u8 and then 32
    // proto layers are summed together. Which means in the result of the matrix
    // multiply, the maximum value of a single element is 256 * 256 * 32 = 2^21
    let mask = mask
        .mapv(|x| to_i32(x) - to_i32(quant_boxes.zero_point))
        .dot(&protos.mapv(|x| to_i32(x) - to_i32(quant_protos.zero_point)))
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    let min = mask.min().unwrap();
    let max = mask.max().unwrap();

    mask.map(|x| ((x - min) as f32 / *max as f32 * 255.0) as u8)
}
