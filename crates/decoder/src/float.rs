// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{arg_max, BBoxTypeTrait, BoundingBox, DetectBox};
use ndarray::{
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    Array1, ArrayView2, Zip,
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

/// Multi-label variant of [`postprocess_boxes_index_float`].
///
/// For each anchor row, emits one `(DetectBox, anchor_idx)` per class whose
/// score meets `threshold` — every class, not just the argmax.  The same
/// `anchor_idx` is returned for all per-class entries of a given anchor so
/// that downstream mask-coefficient lookup can reuse the shared coefficient
/// row.
///
/// The bbox is computed once per anchor (via `B::ndarray_to_xyxy_float`) and
/// reused across all emitted classes, avoiding redundant work.
///
/// Intended for **validation/mAP evaluation only** (the Ultralytics `val`
/// convention).  Deployment must use the argmax variant
/// [`postprocess_boxes_index_float`] so trackers see at most one box per
/// anchor.
pub fn postprocess_boxes_multilabel_index_float<
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
    let n = boxes.dim().0;
    let indices: Array1<usize> = (0..n).collect();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(&indices)
        .into_par_iter()
        .flat_map(|(score_row, bbox_row, &anchor_idx)| {
            // Compute bbox once; clone into each per-class candidate.
            let bbox = B::ndarray_to_xyxy_float(bbox_row);
            let bbox: crate::BoundingBox = bbox.into();
            score_row
                .iter()
                .enumerate()
                .filter(|(_, &s)| s >= threshold)
                .map(move |(c, &s)| {
                    (
                        DetectBox {
                            label: c,
                            score: s.as_(),
                            bbox,
                        },
                        anchor_idx,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// If `max_det` is `Some(n)`, the greedy loop stops as soon as `n` survivors
/// have been confirmed. Because the input is sorted descending, the first `n`
/// survivors are the highest-scoring `n`, so the post-NMS top-`n` is preserved
/// without iterating the full O(N²) suppression loop.
#[must_use]
pub fn nms_float(iou: f32, max_det: Option<usize>, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.score.total_cmp(&a.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return match max_det {
            Some(n) => {
                boxes.truncate(n);
                boxes
            }
            None => boxes,
        };
    }

    let cap = max_det.unwrap_or(usize::MAX);
    let mut survivors: usize = 0;

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

        // NOTE: jaccard_batch4_neon is available for callers that can
        // batch unsuppressed candidates externally. It is not used
        // inline here because score=-1 marking creates sparse gaps
        // that prevent contiguous 4-box batching.
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    // Filter out suppressed boxes; cap at `max_det` because boxes after the
    // break may still hold positive scores but score lower than every survivor.
    boxes
        .into_iter()
        .filter(|b| b.score >= 0.0)
        .take(cap)
        .collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// This is same as `nms_float` but will also include extra information along
/// with each box, such as the index
#[must_use]
pub fn nms_extra_float<E: Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.0.score.total_cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return match max_det {
            Some(n) => {
                boxes.truncate(n);
                boxes
            }
            None => boxes,
        };
    }

    let cap = max_det.unwrap_or(usize::MAX);
    let mut survivors: usize = 0;

    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].0.score < 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].0.score < 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou) {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].0.score = -1.0;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }

    // Filter out suppressed boxes; cap at `max_det` for the same reason as
    // `nms_float`.
    boxes
        .into_iter()
        .filter(|b| b.0.score >= 0.0)
        .take(cap)
        .collect()
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
/// let result = nms_class_aware_float(0.3, None, boxes);
/// assert_eq!(result.len(), 2);
/// ```
#[must_use]
pub fn nms_class_aware_float(
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<DetectBox>,
) -> Vec<DetectBox> {
    boxes.par_sort_by(|a, b| b.score.total_cmp(&a.score));

    if iou >= 1.0 {
        return match max_det {
            Some(n) => {
                boxes.truncate(n);
                boxes
            }
            None => boxes,
        };
    }

    let cap = max_det.unwrap_or(usize::MAX);
    let mut survivors: usize = 0;

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
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    boxes
        .into_iter()
        .filter(|b| b.score >= 0.0)
        .take(cap)
        .collect()
}

/// Class-aware NMS with extra data: only suppress boxes with the same label.
///
/// This is same as `nms_class_aware_float` but will also include extra
/// information along with each box, such as the index.
#[must_use]
pub fn nms_extra_class_aware_float<E: Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    boxes.par_sort_by(|a, b| b.0.score.total_cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return match max_det {
            Some(n) => {
                boxes.truncate(n);
                boxes
            }
            None => boxes,
        };
    }

    let cap = max_det.unwrap_or(usize::MAX);
    let mut survivors: usize = 0;

    for i in 0..boxes.len() {
        if boxes[i].0.score < 0.0 {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if boxes[j].0.score < 0.0 {
                continue;
            }
            // Only suppress if same class AND overlapping
            if boxes[j].0.label == boxes[i].0.label
                && jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou)
            {
                boxes[j].0.score = -1.0;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    boxes
        .into_iter()
        .filter(|b| b.0.score >= 0.0)
        .take(cap)
        .collect()
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

/// Batch IoU check: test one reference box `a` against 4 candidate boxes.
///
/// Returns a 4-element array of booleans: `result[i]` is true if
/// `jaccard(a, boxes[i], iou)` would return true.
///
/// On aarch64, uses NEON `vmaxq_f32`/`vminq_f32` for vectorized
/// intersection computation. On other architectures falls back to
/// 4 scalar `jaccard` calls.
#[inline]
pub fn jaccard_batch4(a: &BoundingBox, boxes: &[BoundingBox; 4], iou: f32) -> [bool; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { jaccard_batch4_neon(a, boxes, iou) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            jaccard(a, &boxes[0], iou),
            jaccard(a, &boxes[1], iou),
            jaccard(a, &boxes[2], iou),
            jaccard(a, &boxes[3], iou),
        ]
    }
}

/// NEON-vectorized batch IoU for 4 candidate boxes against one reference.
///
/// Loads xmin/ymin/xmax/ymax of the 4 candidates into separate NEON
/// registers (AoS→SoA transpose), then computes intersection, union,
/// and the `intersection > iou * union` test in 4-wide SIMD.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn jaccard_batch4_neon(a: &BoundingBox, boxes: &[BoundingBox; 4], iou: f32) -> [bool; 4] {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let iou_v = vdupq_n_f32(iou);

    // Reference box broadcast.
    let a_xmin = vdupq_n_f32(a.xmin);
    let a_ymin = vdupq_n_f32(a.ymin);
    let a_xmax = vdupq_n_f32(a.xmax);
    let a_ymax = vdupq_n_f32(a.ymax);
    let area_a = vmulq_f32(vsubq_f32(a_xmax, a_xmin), vsubq_f32(a_ymax, a_ymin));

    // Load 4 boxes (each BoundingBox is [xmin, ymin, xmax, ymax]).
    let b0 = vld1q_f32(&boxes[0].xmin as *const f32);
    let b1 = vld1q_f32(&boxes[1].xmin as *const f32);
    let b2 = vld1q_f32(&boxes[2].xmin as *const f32);
    let b3 = vld1q_f32(&boxes[3].xmin as *const f32);

    // AoS → SoA transpose (4×4).
    let t01_lo = vtrn1q_f32(b0, b1); // xmin0,xmin1,xmax0,xmax1
    let t01_hi = vtrn2q_f32(b0, b1); // ymin0,ymin1,ymax0,ymax1
    let t23_lo = vtrn1q_f32(b2, b3);
    let t23_hi = vtrn2q_f32(b2, b3);

    let b_xmin = vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(t01_lo),
        vreinterpretq_f64_f32(t23_lo),
    ));
    let b_ymin = vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(t01_hi),
        vreinterpretq_f64_f32(t23_hi),
    ));
    let b_xmax = vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(t01_lo),
        vreinterpretq_f64_f32(t23_lo),
    ));
    let b_ymax = vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(t01_hi),
        vreinterpretq_f64_f32(t23_hi),
    ));

    // Intersection.
    let left = vmaxq_f32(a_xmin, b_xmin);
    let top = vmaxq_f32(a_ymin, b_ymin);
    let right = vminq_f32(a_xmax, b_xmax);
    let bottom = vminq_f32(a_ymax, b_ymax);
    let w = vmaxq_f32(vsubq_f32(right, left), zero);
    let h = vmaxq_f32(vsubq_f32(bottom, top), zero);
    let intersection = vmulq_f32(w, h);

    // Area B.
    let area_b = vmulq_f32(vsubq_f32(b_xmax, b_xmin), vsubq_f32(b_ymax, b_ymin));

    // Union = area_a + area_b - intersection.
    let union = vsubq_f32(vaddq_f32(area_a, area_b), intersection);

    // Test: intersection > iou * union (equivalent to IoU > threshold).
    let iou_union = vmulq_f32(iou_v, union);
    let mask = vcgtq_f32(intersection, iou_union);

    // Extract per-lane results.
    [
        vgetq_lane_u32(mask, 0) != 0,
        vgetq_lane_u32(mask, 1) != 0,
        vgetq_lane_u32(mask, 2) != 0,
        vgetq_lane_u32(mask, 3) != 0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundingBox;

    /// Helper: create `n` non-overlapping boxes with descending f32 scores.
    fn make_nms_boxes_float(n: usize) -> Vec<DetectBox> {
        (0..n)
            .map(|i| DetectBox {
                bbox: BoundingBox {
                    xmin: i as f32 * 100.0,
                    ymin: 0.0,
                    xmax: i as f32 * 100.0 + 10.0,
                    ymax: 10.0,
                },
                label: 0,
                score: 1.0 - i as f32 * 0.01,
            })
            .collect()
    }

    #[test]
    fn nms_float_max_det_matches_full_truncated() {
        let boxes = make_nms_boxes_float(20);
        let n = 5;
        let full = nms_float(0.5, None, boxes.clone());
        let capped = nms_float(0.5, Some(n), boxes);
        assert_eq!(capped.len(), n);
        for (f, c) in full[..n].iter().zip(capped.iter()) {
            assert_eq!(f.bbox, c.bbox);
            assert_eq!(f.score, c.score);
        }
    }

    #[test]
    fn nms_float_max_det_zero_returns_empty() {
        let boxes = make_nms_boxes_float(10);
        let result = nms_float(0.5, Some(0), boxes);
        assert!(result.is_empty());
    }

    #[test]
    fn nms_float_max_det_iou_ge_1_returns_sorted_truncated() {
        let boxes = make_nms_boxes_float(10);
        let result = nms_float(1.0, Some(3), boxes);
        assert_eq!(result.len(), 3);
        assert!(result[0].score >= result[1].score);
        assert!(result[1].score >= result[2].score);
    }

    #[test]
    fn nms_float_max_det_larger_than_input() {
        let boxes = make_nms_boxes_float(5);
        let full = nms_float(0.5, None, boxes.clone());
        let capped = nms_float(0.5, Some(100), boxes);
        assert_eq!(full.len(), capped.len());
    }

    #[test]
    fn jaccard_batch4_matches_scalar() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let boxes = [
            BoundingBox::new(5.0, 5.0, 15.0, 15.0),   // overlap
            BoundingBox::new(20.0, 20.0, 30.0, 30.0), // no overlap
            BoundingBox::new(0.0, 0.0, 10.0, 10.0),   // identical
            BoundingBox::new(8.0, 8.0, 18.0, 18.0),   // small overlap
        ];
        let iou_threshold = 0.1;
        let batch = jaccard_batch4(&a, &boxes, iou_threshold);
        for (i, b) in boxes.iter().enumerate() {
            let scalar = jaccard(&a, b, iou_threshold);
            assert_eq!(
                batch[i], scalar,
                "batch4 mismatch at {i}: batch={} scalar={}",
                batch[i], scalar
            );
        }
    }
}
