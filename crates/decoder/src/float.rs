// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Float-domain NMS and box geometry.
//!
//! Two groups of primitives, both reusable outside the decoder:
//!
//! - **Suppression** — [`nms_float`], [`nms_class_aware_float`], and the
//!   `_extra` variants that carry a per-box payload (mask coefficients, say)
//!   through suppression so the survivors stay paired with their data.
//! - **Geometry** — [`intersection_area`], [`box_area`], [`iou_value`], and
//!   [`ios_value`]. The tiled-inference merge in [`crate::tiling`] shares
//!   these, which is what keeps its IoS metric numerically identical to the
//!   IoU used by ordinary NMS.
//!
//! [`jaccard`] is the threshold test the suppression loops use: it answers "do
//! these overlap past `iou`?" without materializing the ratio. [`jaccard_batch4`]
//! is the public batch form of it, one box against four.

use crate::neon_dispatch::{
    column_argmax, column_hits, ge_indices_kernel, jaccard4_kernel, keep_unflagged, nms_dead,
    nms_isa, row_arg_max, score_columns, Isa, Ran, PAR_MIN_SCORES, ROWS_PER_TASK,
};
use crate::{BBoxTypeTrait, BoundingBox, DetectBox};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::{AsPrimitive, Float};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// `f` over every `(index, score row, box row)`, keeping the `Some`s in row
/// order, on the rayon pool when the score array is large.
fn filter_rows<BOX, SCORE, T, F>(boxes: ArrayView2<BOX>, scores: ArrayView2<SCORE>, f: F) -> Vec<T>
where
    BOX: Sync,
    SCORE: Sync,
    T: Send,
    F: Fn(usize, ArrayView1<SCORE>, ArrayView1<BOX>) -> Option<T> + Sync + Send,
{
    let (n, nc) = scores.dim();
    let row = |i: usize| f(i, scores.row(i), boxes.row(i));
    if n * nc < PAR_MIN_SCORES {
        (0..n).filter_map(row).collect()
    } else {
        (0..n)
            .into_par_iter()
            .with_min_len(ROWS_PER_TASK)
            .filter_map(row)
            .collect()
    }
}

/// Post processes boxes and scores tensors into detection boxes, filtering out
/// any boxes below the score threshold. The boxes tensor is converted to XYXY
/// using the given BBoxTypeTrait. The order of the boxes is preserved.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
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
    argmax_candidates::<B, _, _>(Isa::current_or_scalar(), threshold, boxes, scores)
        .into_iter()
        .map(|(b, _)| b)
        .collect()
}

/// Post processes boxes and scores tensors into detection boxes, filtering out
/// any boxes below the score threshold. The boxes tensor is converted to XYXY
/// using the given BBoxTypeTrait. The order of the boxes is preserved.
///
/// This function is very similar to `postprocess_boxes_float` but will also
/// return the index of the box. The boxes will be in ascending index order.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
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
    argmax_candidates::<B, _, _>(Isa::current_or_scalar(), threshold, boxes, scores)
}

/// [`postprocess_boxes_index_float`] on `isa`.
pub(crate) fn argmax_candidates<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
) -> Vec<(DetectBox, usize)> {
    let detect = |i: usize, score: SCORE, label: usize| {
        let bbox = B::ndarray_to_xyxy_float(boxes.row(i));
        (
            DetectBox {
                label,
                score: score.as_(),
                bbox: bbox.into(),
            },
            i,
        )
    };
    let columns = (!isa.is_scalar()).then(|| score_columns(&scores)).flatten();
    if let Some(kept) = columns.and_then(|cols| column_argmax(isa, &cols, threshold)) {
        return kept
            .into_iter()
            .filter_map(|(i, c)| scores.get((i, c)).map(|&s| detect(i, s, c)))
            .collect();
    }
    filter_rows(boxes, scores, |i, score, _| {
        let (score_, label) = row_arg_max(isa, score);
        if score_ < threshold {
            return None;
        }
        Some(detect(i, score_, label))
    })
}

/// The argmax or multi-label candidates of a decode on `isa`, with an error instead of a panic
/// for mismatched shapes.
pub(crate) fn decode_candidates<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    multi_label: bool,
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
) -> crate::DecoderResult<Vec<(DetectBox, usize)>> {
    crate::byte::check_candidate_shapes(boxes.dim(), scores.dim())?;
    Ok(if multi_label {
        multilabel_candidates::<B, _, _>(isa, threshold, boxes, scores)
    } else {
        argmax_candidates::<B, _, _>(isa, threshold, boxes, scores)
    })
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
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
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
    multilabel_candidates::<B, _, _>(Isa::current_or_scalar(), threshold, boxes, scores)
}

/// [`postprocess_boxes_multilabel_index_float`] on `isa`.
pub(crate) fn multilabel_candidates<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
) -> Vec<(DetectBox, usize)> {
    if !isa.is_scalar() {
        if let Some(hits) =
            score_columns(&scores).and_then(|cols| column_hits(isa, &cols, threshold))
        {
            let mut out = Vec::with_capacity(hits.len());
            let mut current: Option<(usize, BoundingBox)> = None;
            for (anchor, label) in hits {
                let bbox = match current {
                    Some((a, bbox)) if a == anchor => bbox,
                    _ => {
                        let bbox = B::ndarray_to_xyxy_float(boxes.row(anchor)).into();
                        current = Some((anchor, bbox));
                        bbox
                    }
                };
                if let Some(&score) = scores.get((anchor, label)) {
                    out.push((
                        DetectBox {
                            label,
                            score: score.as_(),
                            bbox,
                        },
                        anchor,
                    ));
                }
            }
            return out;
        }
    }
    filter_rows(boxes, scores, |anchor_idx, score_row, bbox_row| {
        let mut classes = Vec::new();
        let row = score_row.as_slice();
        let ran = row.map_or(Ran::Scalar, |r| {
            ge_indices_kernel(isa, r, threshold, |c| classes.push(c))
        });
        if ran == Ran::Scalar {
            classes.extend(
                score_row
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s >= threshold)
                    .map(|(c, _)| c),
            );
        }
        if classes.is_empty() {
            return None;
        }
        // Compute bbox once; clone into each per-class candidate.
        let bbox = B::ndarray_to_xyxy_float(bbox_row);
        let bbox: crate::BoundingBox = bbox.into();
        Some(
            classes
                .into_iter()
                .filter_map(|c| {
                    let &s = score_row.get(c)?;
                    Some((
                        DetectBox {
                            label: c,
                            score: s.as_(),
                            bbox,
                        },
                        anchor_idx,
                    ))
                })
                .collect::<Vec<_>>(),
        )
    })
    .into_iter()
    .flatten()
    .collect()
}

/// [`postprocess_boxes_multilabel_index_float`] without anchor indices.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
///
/// # Errors
/// [`crate::DecoderError::InvalidShape`] unless boxes are `[N, 4]` and scores
/// `[N, C]`.
pub fn postprocess_boxes_multilabel_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    threshold: SCORE,
    boxes: ArrayView2<BOX>,
    scores: ArrayView2<SCORE>,
) -> crate::DecoderResult<Vec<DetectBox>> {
    crate::byte::check_candidate_shapes(boxes.dim(), scores.dim())?;
    Ok(
        postprocess_boxes_multilabel_index_float::<B, _, _>(threshold, boxes, scores)
            .into_iter()
            .map(|(b, _)| b)
            .collect(),
    )
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// If `max_det` is `Some(n)`, the greedy loop stops as soon as `n` survivors
/// have been confirmed. Because the input is sorted descending, the first `n`
/// survivors are the highest-scoring `n`, so the post-NMS top-`n` is preserved
/// without iterating the full O(N²) suppression loop.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[must_use]
pub fn nms_float(iou: f32, max_det: Option<usize>, boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    nms_float_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_float`] on `isa`.
#[must_use]
pub(crate) fn nms_float_on(
    isa: Isa,
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<DetectBox>,
) -> Vec<DetectBox> {
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
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.bbox,
        false,
        |b| b.label,
        |b| b.score < 0.0,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.score >= 0.0, cap);
    }
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
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[must_use]
pub fn nms_extra_float<E: Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    nms_extra_float_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_extra_float`] on `isa`.
#[must_use]
pub(crate) fn nms_extra_float_on<E: Send + Sync>(
    isa: Isa,
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
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.0.bbox,
        false,
        |b| b.0.label,
        |b| b.0.score < 0.0,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.0.score >= 0.0, cap);
    }
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
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[must_use]
pub fn nms_class_aware_float(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<DetectBox>,
) -> Vec<DetectBox> {
    nms_class_aware_float_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_class_aware_float`] on `isa`.
#[must_use]
pub(crate) fn nms_class_aware_float_on(
    isa: Isa,
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
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.bbox,
        true,
        |b| b.label,
        |b| b.score < 0.0,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.score >= 0.0, cap);
    }
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
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[must_use]
pub fn nms_extra_class_aware_float<E: Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<(DetectBox, E)>,
) -> Vec<(DetectBox, E)> {
    nms_extra_class_aware_float_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_extra_class_aware_float`] on `isa`.
#[must_use]
pub(crate) fn nms_extra_class_aware_float_on<E: Send + Sync>(
    isa: Isa,
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
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.0.bbox,
        true,
        |b| b.0.label,
        |b| b.0.score < 0.0,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.0.score >= 0.0, cap);
    }
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

/// Area of the axis-aligned intersection of two boxes, clamped to `>= 0` per
/// side. Mirrors the `inter` term in ModelPack `metrics/tiled.py::_match`.
#[must_use]
#[inline]
pub fn intersection_area(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);
    (right - left).max(0.0) * (bottom - top).max(0.0)
}

/// Area of a box, clamped to `>= 0` per side. Mirrors ModelPack `_areas`.
#[must_use]
#[inline]
pub fn box_area(b: &BoundingBox) -> f32 {
    (b.xmax - b.xmin).max(0.0) * (b.ymax - b.ymin).max(0.0)
}

/// Intersection-over-Union value in `[0, 1]`. Mirrors ModelPack
/// `_match(metric='iou')`: `inter / max(area_a + area_b - inter, 1e-9)`.
#[must_use]
#[inline]
pub fn iou_value(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter = intersection_area(a, b);
    let union = (box_area(a) + box_area(b) - inter).max(1e-9);
    inter / union
}

/// Intersection-over-Smaller value in `[0, 1]`. Mirrors ModelPack
/// `_match(metric='ios')`: `inter / max(min(area_a, area_b), 1e-9)`. Used by the
/// tiled-detection merge: a seam-split object has low IoU but high IoS.
#[must_use]
#[inline]
pub fn ios_value(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter = intersection_area(a, b);
    let denom = box_area(a).min(box_area(b)).max(1e-9);
    inter / denom
}

/// Returns true if the IOU of the given bounding boxes is greater than the iou
/// threshold
///
/// Kept as a standalone inline (NOT routed through [`iou_value`]) so this hot
/// NMS primitive stays bit-identical to the NEON NMS kernel and
/// [`jaccard_batch4`]; the value helpers above are additive and used by the
/// tiled-detection merge.
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
/// Returns a 4-element array of booleans: `result[i]` is `jaccard(a, boxes[i], iou)`, on every
/// input; on NEON tiers the four tests run as one vector.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[inline]
pub fn jaccard_batch4(a: &BoundingBox, boxes: &[BoundingBox; 4], iou: f32) -> [bool; 4] {
    jaccard4_kernel(Isa::current_or_scalar(), a, boxes, iou).unwrap_or_else(|| {
        [
            jaccard(a, &boxes[0], iou),
            jaccard(a, &boxes[1], iou),
            jaccard(a, &boxes[2], iou),
            jaccard(a, &boxes[3], iou),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundingBox;

    #[test]
    fn multilabel_float_drops_indices_only() {
        let boxes = ndarray::arr2(&[[0.1f32, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]);
        let scores = ndarray::arr2(&[[0.6f32, 0.1, 0.7], [0.05, 0.9, 0.05]]);
        let with_idx = postprocess_boxes_multilabel_index_float::<crate::XYXY, _, _>(
            0.5,
            boxes.view(),
            scores.view(),
        );
        let without = postprocess_boxes_multilabel_float::<crate::XYXY, _, _>(
            0.5,
            boxes.view(),
            scores.view(),
        )
        .unwrap();
        let stripped: Vec<_> = with_idx.into_iter().map(|(b, _)| b).collect();
        assert_eq!(stripped, without);
        assert_eq!(without.len(), 3);
    }

    #[test]
    fn multilabel_float_rejects_non_four_column_boxes() {
        let boxes = ndarray::arr2(&[[0.1f32, 0.1, 0.2, 0.2, 0.0]]);
        let scores = ndarray::arr2(&[[0.6f32]]);
        let r = postprocess_boxes_multilabel_float::<crate::XYXY, _, _>(
            0.5,
            boxes.view(),
            scores.view(),
        );
        assert!(matches!(r, Err(crate::DecoderError::InvalidShape(_))));
    }

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

    /// Two boxes at the same place with the same score. Sorting is stable
    /// enough that the first stays and the second is suppressed; a score of
    /// exactly 0.0 must not be mistaken for the `-1.0` suppression marker at
    /// either end of the comparison.
    fn overlapping_pair(score: f32) -> Vec<DetectBox> {
        vec![
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 10.0,
                    ymax: 10.0,
                },
                label: 0,
                score,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 1.0,
                    ymin: 1.0,
                    xmax: 11.0,
                    ymax: 11.0,
                },
                label: 0,
                score,
            },
        ]
    }

    /// A box scoring exactly 0.0 is a real detection that passed the
    /// threshold, not a suppressed one. Treating `< 0.0` as `<= 0.0` would
    /// skip it and leave its overlapping neighbour unsuppressed.
    #[test]
    fn nms_float_suppresses_overlapping_zero_score_boxes() {
        let result = nms_float(0.5, None, overlapping_pair(0.0));
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].score, 0.0);
    }

    /// Three boxes in a chain: A overlaps B, B overlaps C, A does not reach
    /// C. Once B is suppressed it must take no further part, or it suppresses
    /// C on behalf of a box that is not in the output.
    ///
    /// The offsets carry the whole test. At 10 wide and stepping 2, A-B and
    /// B-C are both 8/12 = 0.667 and A-C is 6/14 = 0.429, so only the
    /// adjacent pairs clear the 0.5 threshold. Step further apart -- 0, 1 and
    /// 9.5, say -- and B-C falls to 0.081, B cannot reach C whatever the
    /// implementation does, and all three tests below pass with the bug they
    /// exist to catch.
    fn suppression_chain() -> Vec<DetectBox> {
        let mk = |xmin: f32, xmax: f32, score: f32| DetectBox {
            bbox: BoundingBox {
                xmin,
                ymin: 0.0,
                xmax,
                ymax: 10.0,
            },
            label: 0,
            score,
        };
        vec![mk(0.0, 10.0, 0.9), mk(2.0, 12.0, 0.8), mk(4.0, 14.0, 0.7)]
    }

    #[test]
    fn nms_float_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let result = nms_float(0.5, None, suppression_chain());
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].score, 0.9);
        assert_eq!(result[1].score, 0.7);
    }

    #[test]
    fn nms_extra_float_suppresses_overlapping_zero_score_boxes() {
        let boxes = overlapping_pair(0.0)
            .into_iter()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let result = nms_extra_float(0.5, None, boxes);
        assert_eq!(result.len(), 1, "{result:?}");
        // Both boxes score 0.0 and par_sort_by is an unstable sort, so which
        // of the two survives is not fixed; that it carries its own payload
        // is.
        assert!(result[0].1 == 0 || result[0].1 == 1, "{result:?}");
    }

    #[test]
    fn nms_extra_float_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let boxes = suppression_chain()
            .into_iter()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        let result = nms_extra_float(0.5, None, boxes);
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].1, 0);
        assert_eq!(result[1].1, 2);
    }

    /// Class-aware NMS with no cap still has to run the whole suppression
    /// loop. Breaking out of it early leaves overlapping same-class boxes in
    /// the output.
    #[test]
    fn nms_class_aware_float_suppresses_same_class_without_a_cap() {
        let mut boxes = overlapping_pair(0.9);
        boxes[1].score = 0.8;
        let result = nms_class_aware_float(0.5, None, boxes);
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].score, 0.9);
    }

    /// The same two boxes with different labels are both kept: class-aware
    /// NMS suppresses within a class, never across.
    #[test]
    fn nms_class_aware_float_keeps_overlapping_boxes_of_other_classes() {
        let mut boxes = overlapping_pair(0.9);
        boxes[1].score = 0.8;
        boxes[1].label = 1;
        let result = nms_class_aware_float(0.5, None, boxes);
        assert_eq!(result.len(), 2, "{result:?}");
    }

    #[test]
    fn nms_class_aware_float_suppresses_overlapping_zero_score_boxes() {
        let result = nms_class_aware_float(0.5, None, overlapping_pair(0.0));
        assert_eq!(result.len(), 1, "{result:?}");
    }

    #[test]
    fn nms_class_aware_float_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let result = nms_class_aware_float(0.5, None, suppression_chain());
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[1].score, 0.7);
    }

    /// A box that overlaps nothing, then a pair that overlaps each other. The
    /// first box confirms a survivor without suppressing anything, so a loop
    /// that stops at the first survivor leaves the pair un-deduplicated while
    /// the three-box chain above, where the first box does the suppressing,
    /// looks identical either way.
    fn isolated_then_overlapping_pair() -> Vec<DetectBox> {
        let mk = |xmin: f32, xmax: f32, score: f32| DetectBox {
            bbox: BoundingBox {
                xmin,
                ymin: 0.0,
                xmax,
                ymax: 10.0,
            },
            label: 0,
            score,
        };
        vec![
            mk(100.0, 110.0, 0.9),
            mk(0.0, 10.0, 0.8),
            mk(1.0, 11.0, 0.7),
        ]
    }

    #[test]
    fn nms_class_aware_float_keeps_suppressing_after_the_first_survivor() {
        let result = nms_class_aware_float(0.5, None, isolated_then_overlapping_pair());
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].score, 0.9);
        assert_eq!(result[1].score, 0.8);
    }

    fn with_index(boxes: Vec<DetectBox>) -> Vec<(DetectBox, usize)> {
        boxes.into_iter().enumerate().map(|(i, b)| (b, i)).collect()
    }

    #[test]
    fn nms_extra_class_aware_float_keeps_suppressing_after_the_first_survivor() {
        let result =
            nms_extra_class_aware_float(0.5, None, with_index(isolated_then_overlapping_pair()));
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[1].1, 1);
    }

    #[test]
    fn nms_extra_class_aware_float_suppresses_same_class_without_a_cap() {
        let mut boxes = overlapping_pair(0.9);
        boxes[1].score = 0.8;
        let result = nms_extra_class_aware_float(0.5, None, with_index(boxes));
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].0.score, 0.9);
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn nms_extra_class_aware_float_keeps_overlapping_boxes_of_other_classes() {
        let mut boxes = overlapping_pair(0.9);
        boxes[1].score = 0.8;
        boxes[1].label = 1;
        let result = nms_extra_class_aware_float(0.5, None, with_index(boxes));
        assert_eq!(result.len(), 2, "{result:?}");
    }

    #[test]
    fn nms_extra_class_aware_float_suppresses_overlapping_zero_score_boxes() {
        let result = nms_extra_class_aware_float(0.5, None, with_index(overlapping_pair(0.0)));
        assert_eq!(result.len(), 1, "{result:?}");
    }

    #[test]
    fn nms_extra_class_aware_float_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let result = nms_extra_class_aware_float(0.5, None, with_index(suppression_chain()));
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[1].1, 2);
    }

    /// The threshold is a floor, not a cut: a score equal to it is kept.
    #[test]
    fn postprocess_boxes_float_keeps_a_score_equal_to_the_threshold() {
        let boxes = ndarray::array![[0.0_f32, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]];
        let scores = ndarray::array![[0.5_f32], [0.25]];

        let kept = postprocess_boxes_float::<crate::XYXY, _, _>(0.5, boxes.view(), scores.view());
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].score, 0.5);

        let kept =
            postprocess_boxes_index_float::<crate::XYXY, _, _>(0.5, boxes.view(), scores.view());
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].1, 0);
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

    // Parity literals from ModelPack metrics/tiled.py::_match for
    // A=[100,100,400,300], B=[350,100,400,300] (B fully inside A):
    // inter=10000, area_a=60000, area_b=10000 => IoS=1.0, IoU=0.16667.
    #[test]
    fn metric_values_match_modelpack_match() {
        let a = BoundingBox::new(100.0, 100.0, 400.0, 300.0);
        let b = BoundingBox::new(350.0, 100.0, 400.0, 300.0);
        assert!((intersection_area(&a, &b) - 10000.0).abs() < 1e-3);
        assert!((box_area(&a) - 60000.0).abs() < 1e-3);
        assert!((box_area(&b) - 10000.0).abs() < 1e-3);
        assert!((ios_value(&a, &b) - 1.0).abs() < 1e-6);
        assert!((iou_value(&a, &b) - (1.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn metric_values_disjoint_are_zero() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(20.0, 20.0, 30.0, 30.0);
        assert_eq!(intersection_area(&a, &b), 0.0);
        assert_eq!(ios_value(&a, &b), 0.0);
        assert_eq!(iou_value(&a, &b), 0.0);
    }

    #[test]
    fn metric_values_zero_area_box_no_panic() {
        // Degenerate (zero-area) box: denom floored to 1e-9, result finite.
        let degenerate = BoundingBox::new(10.0, 10.0, 10.0, 20.0);
        let other = BoundingBox::new(0.0, 0.0, 30.0, 30.0);
        assert_eq!(box_area(&degenerate), 0.0);
        assert!(ios_value(&degenerate, &other).is_finite());
        assert!(iou_value(&degenerate, &other).is_finite());
    }

    #[test]
    fn ios_high_where_iou_low_for_contained_box() {
        // The SAHI seam case: a small box fully inside a large one.
        let big = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let small = BoundingBox::new(10.0, 10.0, 20.0, 20.0);
        assert!((ios_value(&big, &small) - 1.0).abs() < 1e-6); // fully contained
        assert!(iou_value(&big, &small) < 0.05); // but IoU tiny
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
