use crate::{BBoxTypeTrait, DetectBoxQuantized, Quantization, arg_max};
use ndarray::{
    ArrayView1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, PrimInt};

pub fn postprocess_boxes_8bit<
    B: BBoxTypeTrait,
    T: PrimInt + AsPrimitive<i16> + AsPrimitive<f32> + Send + Sync,
>(
    threshold: T,
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    quant: &Quantization<T>,
) -> Vec<DetectBoxQuantized<i16>> {
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

            let bbox_quant = B::ndarray_to_xyxy_quant(bbox, zp);
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_.as_(),
                xmin: bbox_quant[0],
                ymin: bbox_quant[1],
                xmax: bbox_quant[2],
                ymax: bbox_quant[3],
            })
        })
        .collect()
}

pub fn postprocess_boxes_extra_8bit<
    'a,
    B: BBoxTypeTrait,
    T: PrimInt + AsPrimitive<i16> + AsPrimitive<f32> + Send + Sync,
    E: Send + Sync,
>(
    threshold: T,
    boxes: ArrayView2<T>,
    scores: ArrayView2<T>,
    extra: &'a ArrayView2<E>,
    quant_boxes: &Quantization<T>,
) -> Vec<(DetectBoxQuantized<i16>, ArrayView1<'a, E>)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let zp = quant_boxes.zero_point.as_();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(extra.rows())
        .into_par_iter()
        .filter_map(|(score, bbox, mask)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }
            let bbox_quant = B::ndarray_to_xyxy_quant(bbox, zp);
            Some((
                DetectBoxQuantized::<i16> {
                    label,
                    score: score_.as_(),
                    xmin: bbox_quant[0],
                    ymin: bbox_quant[1],
                    xmax: bbox_quant[2],
                    ymax: bbox_quant[3],
                },
                mask,
            ))
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
            if jaccard_i16(&boxes[j], &boxes[i], iou) {
                // suppress this box
                boxes[j].score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

pub fn nms_extra_i16<E>(
    iou: f32,
    mut boxes: Vec<(DetectBoxQuantized<i16>, E)>,
) -> Vec<(DetectBoxQuantized<i16>, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.0.score.cmp(&a.0.score));
    let min_val = i16::MIN;
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
            if jaccard_i16(&boxes[j].0, &boxes[i].0, iou) {
                // suppress this box
                boxes[j].0.score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.0.score > min_val).collect()
}

// Returns true if the IOU of the given boxes are greater than the iou threshold
fn jaccard_i16(a: &DetectBoxQuantized<i16>, b: &DetectBoxQuantized<i16>, iou: f32) -> bool {
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
        return 0.0 > iou;
    }
    intersection as f32 / union as f32 > iou
}

pub(crate) fn quantize_score_threshold<T: AsPrimitive<f32> + PrimInt>(
    score: f32,
    quant: &Quantization<T>,
) -> T {
    if quant.scale == 0.0 {
        return T::max_value();
    }
    let v = (score / quant.scale + quant.zero_point.as_()).ceil();
    let v = v.clamp(T::min_value().as_(), T::max_value().as_());
    T::from(v).unwrap()
}
