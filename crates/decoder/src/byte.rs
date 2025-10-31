use crate::{
    BBoxTypeTrait, BoundingBoxQuantized, DetectBoxQuantized, Quantization, ReinterpretSigns,
    arg_max,
};
use ndarray::{
    ArrayView1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, PrimInt};

pub fn postprocess_boxes_8bit<
    B: BBoxTypeTrait,
    Boxes: PrimInt + Send + Sync + ReinterpretSigns<Signed = i8, Unsigned = u8>,
    Scores: PrimInt + AsPrimitive<i16> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant: Quantization,
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

            let bbox_quant = if quant.signed {
                B::ndarray_to_xyxy_quant(bbox.map(|x| x.reinterp_signed()).view(), zp)
            } else {
                B::ndarray_to_xyxy_quant(bbox.map(|x| x.reinterp_unsigned()).view(), zp)
            };
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_.as_(),
                bbox: BoundingBoxQuantized::from_array(&bbox_quant),
            })
        })
        .collect()
}

pub fn postprocess_boxes_extra_8bit<
    'a,
    B: BBoxTypeTrait,
    Boxes: PrimInt + Send + Sync + ReinterpretSigns<Signed = i8, Unsigned = u8>,
    Scores: PrimInt + AsPrimitive<i16> + Send + Sync,
    E: Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    extra: &'a ArrayView2<E>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<i16>, ArrayView1<'a, E>)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let zp = quant_boxes.zero_point as i16;
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(extra.rows())
        .into_par_iter()
        .filter_map(|(score, bbox, mask)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox_quant = if quant_boxes.signed {
                B::ndarray_to_xyxy_quant(bbox.map(|x| x.reinterp_signed()).view(), zp)
            } else {
                B::ndarray_to_xyxy_quant(bbox.map(|x| x.reinterp_unsigned()).view(), zp)
            };

            Some((
                DetectBoxQuantized::<i16> {
                    label,
                    score: score_.as_(),
                    bbox: BoundingBoxQuantized::from_array(&bbox_quant),
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
            if jaccard_i16(&boxes[j].bbox, &boxes[i].bbox, iou) {
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
            if jaccard_i16(&boxes[j].0.bbox, &boxes[i].0.bbox, iou) {
                // suppress this box
                boxes[j].0.score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.0.score > min_val).collect()
}

// Returns true if the IOU of the given boxes are greater than the iou threshold
fn jaccard_i16(a: &BoundingBoxQuantized<i16>, b: &BoundingBoxQuantized<i16>, iou: f32) -> bool {
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
