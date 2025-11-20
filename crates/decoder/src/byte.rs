// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    BBoxTypeTrait, BoundingBox, DetectBoxQuantized, Quantization, arg_max, float::jaccard,
};
use ndarray::{
    Array1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, PrimInt};
use rayon::slice::ParallelSliceMut;

/// Post processes boxes and scores tensors into quantized detection boxes,
/// filtering out any boxes below the score threshold. The boxes tensor
/// is converted to XYXY using the given BBoxTypeTrait. The order of the boxes
/// is preserved.
#[doc(hidden)]
pub fn postprocess_boxes_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<DetectBoxQuantized<Scores>> {
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

            let bbox_quant = B::ndarray_to_xyxy_dequant(bbox.view(), quant_boxes);
            Some(DetectBoxQuantized {
                label,
                score: score_,
                bbox: BoundingBox::from(bbox_quant),
            })
        })
        .collect()
}

/// Post processes boxes and scores tensors into quantized detection boxes,
/// filtering out any boxes below the score threshold. The boxes tensor
/// is converted to XYXY using the given BBoxTypeTrait. The order of the boxes
/// is preserved.
///
/// This function is very similar to `postprocess_boxes_quant` but will also
/// return the index of the box. The boxes will be in ascending index order.
#[doc(hidden)]
pub fn postprocess_boxes_index_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<Scores>, usize)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let indices: Array1<usize> = (0..boxes.dim().0).collect();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(&indices)
        .into_par_iter()
        .filter_map(|(score, bbox, index)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox_quant = B::ndarray_to_xyxy_dequant(bbox.view(), quant_boxes);

            Some((
                DetectBoxQuantized {
                    label,
                    score: score_,
                    bbox: BoundingBox::from(bbox_quant),
                },
                *index,
            ))
        })
        .collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
#[doc(hidden)]
pub fn nms_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    iou: f32,
    mut boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.

    boxes.par_sort_by(|a, b| b.score.cmp(&a.score));

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

            if jaccard(&boxes[j].bbox, &boxes[i].bbox, iou) {
                // suppress this box
                boxes[j].score = min_val;
            }
        }
    }
    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// This is same as `nms_int` but will also include extra information along
/// with each box, such as the index
#[doc(hidden)]
pub fn nms_extra_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send + Sync>(
    iou: f32,
    mut boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.0.score.cmp(&a.0.score));
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
            if jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou) {
                // suppress this box
                boxes[j].0.score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.0.score > min_val).collect()
}

/// Quantizes a score from f32 to the given integer type, using the following
/// formula `(score/quant.scale + quant.zero_point).ceil()`, then clamping to
/// the min and max value of the given integer type
///
/// # Examples
/// ```rust
/// use edgefirst_decoder::{Quantization, byte::quantize_score_threshold};
/// let quant = Quantization {
///     scale: 0.1,
///     zero_point: 128,
/// };
/// let q: u8 = quantize_score_threshold::<u8>(0.5, quant);
/// assert_eq!(q, 128 + 5);
/// ```
#[doc(hidden)]
pub fn quantize_score_threshold<T: PrimInt + AsPrimitive<f32>>(score: f32, quant: Quantization) -> T
where
    f32: AsPrimitive<T>,
{
    if quant.scale == 0.0 {
        return T::max_value();
    }
    let v = (score / quant.scale + quant.zero_point as f32).ceil();
    let v = v.clamp(T::min_value().as_(), T::max_value().as_());
    v.as_()
}
