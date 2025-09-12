use ndarray::{
    Array3, ArrayView1, ArrayView2, ArrayView3,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s,
};
use ndarray_stats::QuantileExt;

use crate::{
    BBoxTypeTrait, DetectBox, DetectBoxF64, Quantization, SegmentationMask,
    bits8::{nms_i16, postprocess_boxes_i8, postprocess_boxes_u8},
    dequant_detect_box,
    float::{
        nms_extra_f32, nms_f32, nms_f64, postprocess_boxes_extra_f32, postprocess_boxes_f32,
        postprocess_boxes_f64,
    },
};

pub fn decode_yolo_u8<T: BBoxTypeTrait>(
    output: ArrayView2<u8>,
    quant: &Quantization<u8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as u8;
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_u8::<T>(score_threshold, boxes_tensor, scores_tensor, quant);
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant, quant));
    }
}

pub fn decode_yolo_i8<T: BBoxTypeTrait>(
    output: ArrayView2<i8>,
    quant: &Quantization<i8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_i8::<T>(score_threshold, boxes_tensor, scores_tensor, quant);
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant, quant));
    }
}

pub fn decode_yolo_f32<T: BBoxTypeTrait>(
    output: ArrayView2<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_f32::<T>(score_threshold, boxes_tensor, scores_tensor);
    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn decode_yolo_f64<T: BBoxTypeTrait>(
    output: ArrayView2<f64>,
    score_threshold: f64,
    iou_threshold: f64,
    output_boxes: &mut Vec<DetectBoxF64>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_yolo(&output);
    let boxes = postprocess_boxes_f64::<T>(score_threshold, boxes_tensor, scores_tensor);

    let boxes = nms_f64(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn postprocess_yolo<'a, T>(
    output: &'a ArrayView2<'_, T>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>) {
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4.., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor)
}

pub fn decode_yolo_masks_f32<T: BBoxTypeTrait>(
    output: ArrayView2<f32>,
    protos: ArrayView3<f32>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<SegmentationMask>,
) {
    let (boxes_tensor, scores_tensor, mask_tensor) = postprocess_yolo_seg(&output);

    let boxes = postprocess_boxes_extra_f32::<T>(
        score_threshold,
        boxes_tensor,
        scores_tensor,
        &mask_tensor,
    );
    let mut boxes = nms_extra_f32(iou_threshold, boxes);
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

pub fn postprocess_yolo_seg<'a, T>(
    output: &'a ArrayView2<'_, T>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>, ArrayView2<'a, T>) {
    assert!(output.shape()[0] > 32 + 4, "Output shape is too short");
    let num_classes = output.shape()[0] - 4 - 32;
    let boxes_tensor = output.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]).reversed_axes();
    let mask_tensor = output.slice(s![(num_classes + 4).., ..,]).reversed_axes();
    (boxes_tensor, scores_tensor, mask_tensor)
}

pub fn decode_masks_f32(
    boxes: Vec<(DetectBox, ArrayView1<f32>)>,
    protos: ArrayView3<f32>,
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

pub fn protobox<'a>(
    protos: &'a ArrayView3<f32>,
    roi: &[f32; 4],
) -> (ArrayView3<'a, f32>, [f32; 4]) {
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

pub fn make_mask(mask: ArrayView1<f32>, protos: ArrayView3<f32>) -> Array3<u8> {
    let shape = protos.shape();
    let mask = mask.to_shape((1, mask.len())).unwrap();
    let protos = protos.to_shape([shape[0] * shape[1], shape[2]]).unwrap();
    let protos = protos.reversed_axes();
    let mask = mask
        .dot(&protos)
        .into_shape_with_order((shape[0], shape[1], 1))
        .unwrap();

    let min = mask.min_skipnan();
    let max = mask.max_skipnan();

    mask.map(|x| ((x - min) / max * 255.0) as u8)
}
