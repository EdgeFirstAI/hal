use crate::{DetectBox, DetectBoxQuantized, Quantization, dequant_detect_box};
use ndarray::{
    ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    s,
};

pub fn decode_u8(
    output: ArrayView2<u8>,
    num_classes: usize,
    quant: &Quantization<u8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as u8;
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);
    let boxes = decode_boxes_u8(
        score_threshold,
        scores_tensor,
        boxes_tensor,
        num_classes,
        quant,
    );
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant));
    }
}

pub fn decode_boxes_u8(
    threshold: u8,
    scores: ArrayView2<u8>,
    boxes: ArrayView2<u8>,
    num_classes: usize,
    quant: &Quantization<u8>,
) -> Vec<DetectBoxQuantized<i16>> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    let zp = quant.zero_point as i16;
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
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_ as i16,
                xmin: 2 * (bbox[0] as i16 - zp) - (bbox[2] as i16 - zp),
                ymin: 2 * (bbox[1] as i16 - zp) - (bbox[3] as i16 - zp),
                xmax: 2 * (bbox[0] as i16 - zp) + (bbox[2] as i16 - zp),
                ymax: 2 * (bbox[1] as i16 - zp) + (bbox[3] as i16 - zp),
            })
        })
        .collect()
}

pub fn decode_i8(
    output: ArrayView2<i8>,
    num_classes: usize,
    quant: &Quantization<i8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);
    let boxes = decode_boxes_i8(
        score_threshold,
        scores_tensor,
        boxes_tensor,
        num_classes,
        quant,
    );
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant));
    }
}

pub fn decode_boxes_i8(
    threshold: i8,
    scores: ArrayView2<i8>,
    boxes: ArrayView2<i8>,
    num_classes: usize,
    quant: &Quantization<i8>,
) -> Vec<DetectBoxQuantized<i16>> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    let zp = quant.zero_point as i16;
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
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_ as i16,
                xmin: 2 * (bbox[0] as i16 - zp) - (bbox[2] as i16 - zp),
                ymin: 2 * (bbox[1] as i16 - zp) - (bbox[3] as i16 - zp),
                xmax: 2 * (bbox[0] as i16 - zp) + (bbox[2] as i16 - zp),
                ymax: 2 * (bbox[1] as i16 - zp) + (bbox[3] as i16 - zp),
            })
        })
        .collect()
}

pub fn nms_i16(iou: f32, mut boxes: Vec<DetectBoxQuantized<i16>>) -> Vec<DetectBoxQuantized<i16>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.cmp(&a.score));
    let min_val = i16::MIN;
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= min_val {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= min_val {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_i16(&boxes[j], &boxes[i]) > iou {
                // suppress this box
                boxes[j].score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

fn jaccard_i16(a: &DetectBoxQuantized<i16>, b: &DetectBoxQuantized<i16>) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0) as i32 * (bottom - top).max(0) as i32;
    let area_a = (a.xmax - a.xmin) as i32 * (a.ymax - a.ymin) as i32;
    let area_b = (b.xmax - b.xmin) as i32 * (b.ymax - b.ymin) as i32;

    // need to make sure we are not dividing by zero
    let union = area_a + area_b - intersection;
    if union <= 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}
