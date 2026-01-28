// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{BBoxTypeTrait, BoundingBox, DetectBox, arg_max};
use ndarray::{
    Array1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
};
use num_traits::{AsPrimitive, Float};
use rayon::slice::ParallelSliceMut;

/// Post processes boxes and scores tensors into detection boxes, filtering out
/// any boxes below the score threshold. The boxes tensor is converted to XYXY
/// using the given BBoxTypeTrait. The order of the boxes is preserved.
pub fn postprocess_boxes_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
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
                bbox: bbox.into(),
            })
        })
        .collect()
}

/// Post processes boxes and scores tensors into detection boxes, filtering out
/// any boxes below the score threshold. The boxes tensor is converted to XYXY
/// using the given BBoxTypeTrait. The order of the boxes is preserved.
///
/// This function is very similar to `postprocess_boxes_float` but will also
/// return the index of the box. The boxes will be in ascending index order.
pub fn postprocess_boxes_index_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
) -> Vec<(DetectBox, usize)> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    let indices: Array1<usize> = (0..boxes.dim().0).collect();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(&indices)
        .into_par_iter()
        .filter_map(|(score, bbox, i)| {
            let (score_, label) = arg_max(score);
            if score_ < threshold {
                return None;
            }

            let bbox = B::ndarray_to_xyxy_float(bbox);
            Some((
                DetectBox {
                    label,
                    score: score_.as_(),
                    bbox: bbox.into(),
                },
                *i,
            ))
        })
        .collect()
}




/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
#[must_use]
pub fn nms_float(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.score.total_cmp(&a.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score < 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score < 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard(&boxes[j].bbox, &boxes[i].bbox, iou) {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = -1.0;
            }
        }
    }
    // Filter out suppressed boxes.
    boxes.into_iter().filter(|b| b.score >= 0.0).collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// This is same as `nms_float` but will also include extra information along
/// with each box, such as the index
#[must_use]
pub fn nms_extra_float<E: Send + Sync>(
    iou: f32,
    mut boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.0.score.total_cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

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
            if jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou) {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].0.score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.0.score > 0.0).collect()
}

/// Class-aware NMS: only suppress boxes with the same label.
///
/// Sorts boxes by score, then greedily selects a subset of boxes in descending
/// order of score. Unlike class-agnostic NMS, boxes are only suppressed if they
/// have the same class label AND overlap above the IoU threshold.
///
/// # Example
/// ```
/// # use edgefirst_decoder::{BoundingBox, DetectBox, float::nms_class_aware_float};
/// let boxes = vec![
///     DetectBox {
///         bbox: BoundingBox::new(0.0, 0.0, 0.5, 0.5),
///         score: 0.9,
///         label: 0,
///     },
///     DetectBox {
///         bbox: BoundingBox::new(0.1, 0.1, 0.6, 0.6),
///         score: 0.8,
///         label: 1,
///     }, // different class
/// ];
/// // Both boxes survive because they have different labels
/// let result = nms_class_aware_float(0.3, boxes);
/// assert_eq!(result.len(), 2);
/// ```
#[must_use]
pub fn nms_class_aware_float(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    boxes.par_sort_by(|a, b| b.score.total_cmp(&a.score));

    if iou >= 1.0 {
        return boxes;
    }

    for i in 0..boxes.len() {
        if boxes[i].score < 0.0 {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if boxes[j].score < 0.0 {
                continue;
            }
            // Only suppress if same class AND overlapping
            if boxes[j].label == boxes[i].label && jaccard(&boxes[j].bbox, &boxes[i].bbox, iou) {
                boxes[j].score = -1.0;
            }
        }
    }
    boxes.into_iter().filter(|b| b.score >= 0.0).collect()
}

/// Class-aware NMS with extra data: only suppress boxes with the same label.
///
/// This is same as `nms_class_aware_float` but will also include extra
/// information along with each box, such as the index.
#[must_use]
pub fn nms_extra_class_aware_float<E: Send + Sync>(
    iou: f32,
    mut boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    boxes.par_sort_by(|a, b| b.0.score.total_cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

    for i in 0..boxes.len() {
        if boxes[i].0.score <= 0.0 {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if boxes[j].0.score <= 0.0 {
                continue;
            }
            // Only suppress if same class AND overlapping
            if boxes[j].0.label == boxes[i].0.label
                && jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou)
            {
                boxes[j].0.score = 0.0;
            }
        }
    }
    boxes.into_iter().filter(|b| b.0.score > 0.0).collect()
}

/// Returns true if the IOU of the given bounding boxes is greater than the iou
/// threshold
///
/// # Example
/// ```
/// # use edgefirst_decoder::{BoundingBox, float::jaccard};
/// let a = BoundingBox::new(0.0, 0.0, 0.2, 0.2);
/// let b = BoundingBox::new(0.1, 0.1, 0.3, 0.3);
/// let iou_threshold = 0.1;
/// let result = jaccard(&a, &b, iou_threshold);
/// assert!(result);
/// ```
pub fn jaccard(a: &BoundingBox, b: &BoundingBox, iou: f32) -> bool {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = area_a + area_b - intersection;

    intersection > iou * union
}
