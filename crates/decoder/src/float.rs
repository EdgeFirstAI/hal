use crate::{BBoxTypeTrait, DetectBox, DetectBoxF64};
use ndarray::{
    ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    s,
};

pub fn decode_f32<T: BBoxTypeTrait>(
    output: ArrayView2<f32>,
    num_classes: usize,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);

    let boxes = decode_boxes_f32::<T>(score_threshold, scores_tensor, boxes_tensor, num_classes);

    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn decode_boxes_f32<T: BBoxTypeTrait>(
    threshold: f32,
    scores: ArrayView2<f32>,
    boxes: ArrayView2<f32>,
    num_classes: usize,
) -> Vec<DetectBox> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    Zip::from(scores.columns())
        .and(boxes.columns())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) =
                score
                    .iter()
                    .enumerate()
                    .fold((score[0], 0), |(max, arg_max), (ind, s)| {
                        if max > *s { (max, arg_max) } else { (*s, ind) }
                    });
            if score_ < threshold {
                return None;
            }

            let bbox = T::ndarray_to_xyxy_float(bbox);
            Some(DetectBox {
                label,
                score: score_,
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
            })
        })
        .collect()
}

pub fn nms_f32(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_f32(&boxes[j], &boxes[i]) > iou {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

fn jaccard_f32(a: &DetectBox, b: &DetectBox) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.0000001);

    intersection / union
}

pub fn decode_f64<T: BBoxTypeTrait>(
    output: ArrayView2<f64>,
    num_classes: usize,
    score_threshold: f64,
    iou_threshold: f64,
    output_boxes: &mut Vec<DetectBoxF64>,
) {
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);

    let boxes = decode_boxes_f64::<T>(score_threshold, scores_tensor, boxes_tensor, num_classes);

    let boxes = nms_f64(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn decode_boxes_f64<T: BBoxTypeTrait>(
    threshold: f64,
    scores: ArrayView2<f64>,
    boxes: ArrayView2<f64>,
    num_classes: usize,
) -> Vec<DetectBoxF64> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    Zip::from(scores.columns())
        .and(boxes.columns())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) =
                score
                    .iter()
                    .enumerate()
                    .fold((score[0], 0), |(max, arg_max), (ind, s)| {
                        if max > *s { (max, arg_max) } else { (*s, ind) }
                    });
            if score_ < threshold {
                return None;
            }
            let bbox = T::ndarray_to_xyxy_float(bbox);
            Some(DetectBoxF64 {
                label,
                score: score_,
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
            })
        })
        .collect()
}

pub fn nms_f64(iou: f64, mut boxes: Vec<DetectBoxF64>) -> Vec<DetectBoxF64> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_f64(&boxes[j], &boxes[i]) > iou {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

fn jaccard_f64(a: &DetectBoxF64, b: &DetectBoxF64) -> f64 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.000000001);

    intersection / union
}
