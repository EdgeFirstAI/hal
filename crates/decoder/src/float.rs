use crate::{BBoxTypeTrait, DetectBox, arg_max};
use ndarray::{
    ArrayView1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, Float};

pub fn postprocess_boxes_float<B: BBoxTypeTrait, T: Float + AsPrimitive<f32> + Send + Sync>(
    threshold: T,
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
) -> Vec<DetectBox> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    Zip::from(scores.rows())
        .and(boxes.rows())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox = B::ndarray_to_xyxy_float(bbox);
            Some(DetectBox {
                label,
                score: score_.as_(),
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
            })
        })
        .collect()
}

pub fn postprocess_boxes_extra_float<
    'a,
    B: BBoxTypeTrait,
    T: Float + AsPrimitive<f32> + Send + Sync,
    E: Send + Sync,
>(
    threshold: T,
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    extra: &'a ArrayView2<T>,
) -> Vec<(DetectBox, ArrayView1<'a, T>)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(extra.rows())
        .into_par_iter()
        .filter_map(|(score, bbox, mask)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox = B::ndarray_to_xyxy_float(bbox);
            Some((
                DetectBox {
                    label,
                    score: score_.as_(),
                    xmin: bbox[0],
                    ymin: bbox[1],
                    xmax: bbox[2],
                    ymax: bbox[3],
                },
                mask,
            ))
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
            if jaccard_f32(&boxes[j], &boxes[i], iou) {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

pub fn nms_extra_f32<E: Send + Sync>(
    iou: f32,
    mut boxes: Vec<(DetectBox, ArrayView1<E>)>,
) -> Vec<(DetectBox, ArrayView1<E>)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].0.score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].0.score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_f32(&boxes[j].0, &boxes[i].0, iou) {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].0.score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.0.score > 0.0).collect()
}

// Returns true if the IOU of the given boxes are greater than the iou threshold
fn jaccard_f32(a: &DetectBox, b: &DetectBox, iou: f32) -> bool {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.0000001);

    intersection / union > iou
}
