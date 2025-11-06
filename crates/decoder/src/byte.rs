use crate::{BBoxTypeTrait, BoundingBoxQuantized, DetectBoxQuantized, Quantization, arg_max};
use ndarray::{
    Array1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, ConstZero, PrimInt, Signed};

pub fn postprocess_boxes_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<i32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant: Quantization,
) -> Vec<DetectBoxQuantized<i32, Scores>> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let zp = quant.zero_point.as_();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox_quant = B::ndarray_to_xyxy_quant(bbox.map(|x| x.as_()).view(), zp);
            Some(DetectBoxQuantized {
                label,
                score: score_,
                bbox: BoundingBoxQuantized::from_array(&bbox_quant),
            })
        })
        .collect()
}

pub fn postprocess_boxes_index<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<i32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<i32, Scores>, usize)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let indices: Array1<usize> = (0..boxes.dim().0).collect();
    let zp = quant_boxes.zero_point;
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(&indices)
        .into_par_iter()
        .filter_map(|(score, bbox, index)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox_quant = B::ndarray_to_xyxy_quant(bbox.map(|x| x.as_()).view(), zp);

            Some((
                DetectBoxQuantized {
                    label,
                    score: score_,
                    bbox: BoundingBoxQuantized::from_array(&bbox_quant),
                },
                *index,
            ))
        })
        .collect()
}

pub fn nms_int<
    BOX: Signed + PrimInt + AsPrimitive<DEST> + AsPrimitive<f32>,
    SCORE: PrimInt + AsPrimitive<f32>,
    DEST: PrimInt + 'static + AsPrimitive<f32>,
>(
    iou: f32,
    mut boxes: Vec<DetectBoxQuantized<BOX, SCORE>>,
) -> Vec<DetectBoxQuantized<BOX, SCORE>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.cmp(&a.score));
    let min_val = SCORE::min_value();
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
            if jaccard_int::<BOX, DEST>(&boxes[j].bbox, &boxes[i].bbox, iou) {
                // suppress this box
                boxes[j].score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

pub fn nms_extra_int<
    BOX: ConstZero + Signed + PrimInt + AsPrimitive<DEST> + AsPrimitive<f32>,
    SCORE: PrimInt + AsPrimitive<f32>,
    DEST: ConstZero + PrimInt + 'static + AsPrimitive<f32>,
    E,
>(
    iou: f32,
    mut boxes: Vec<(DetectBoxQuantized<BOX, SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<BOX, SCORE>, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.0.score.cmp(&a.0.score));
    let min_val = SCORE::min_value();
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].0.score <= min_val {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].0.score <= min_val {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_int::<BOX, DEST>(&boxes[j].0.bbox, &boxes[i].0.bbox, iou) {
                // suppress this box
                boxes[j].0.score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.0.score > min_val).collect()
}

// Returns true if the IOU of the given boxes are greater than the iou threshold
fn jaccard_int<
    BOX: Signed + PrimInt + AsPrimitive<DEST> + AsPrimitive<f32>,
    DEST: PrimInt + 'static + AsPrimitive<f32>,
>(
    a: &BoundingBoxQuantized<BOX>,
    b: &BoundingBoxQuantized<BOX>,
    iou: f32,
) -> bool {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);
    let zero = BOX::zero();
    let as_dst = num_traits::AsPrimitive::<DEST>::as_;

    let intersection = as_dst((right - left).max(zero)) * as_dst((bottom - top).max(zero));

    let area_a = as_dst(a.xmax - a.xmin) * as_dst(a.ymax - a.ymin);
    let area_b = as_dst(b.xmax - b.xmin) * as_dst(b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = area_a + area_b - intersection;
    if union <= DEST::zero() {
        return 0.0 > iou;
    }
    intersection.as_() / union.as_() > iou
}

pub(crate) fn quantize_score_threshold<T: PrimInt + AsPrimitive<f32>>(
    score: f32,
    quant: Quantization,
) -> T {
    if quant.scale == 0.0 {
        return T::max_value();
    }
    let v = (score / quant.scale + quant.zero_point as f32).ceil();
    let v = v.clamp(T::min_value().as_(), T::max_value().as_());
    T::from(v).unwrap()
}
