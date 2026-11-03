// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Quantized-domain NMS, the int8 / uint8 mirror of [`crate::float`].
//!
//! These kernels suppress and post-process candidates while the scores are
//! still in their raw integer units, which is the point: dequantizing every
//! anchor costs more than dequantizing the handful that survive. The threshold
//! crosses the other way instead — [`quantize_score_threshold`] maps a float
//! score threshold into the tensor's quantized domain so the comparison stays
//! an integer one.
//!
//! [`postprocess_boxes_quant`] and [`postprocess_boxes_index_quant`] handle
//! candidate selection; [`nms_int`], [`nms_class_aware_int`], and their
//! `_extra` payload-carrying variants handle suppression. Boxes are
//! dequantized only after suppression.

use crate::float::jaccard;
use crate::neon_dispatch::{
    column_argmax, column_hits, ge_indices_kernel, keep_unflagged, nms_dead, nms_isa, row_arg_max,
    score_columns, Isa, Ran, ROWS_PER_TASK,
};
use crate::{
    BBoxTypeTrait, BoundingBox, DecoderError, DecoderResult, DetectBoxQuantized, Quantization,
};
use ndarray::{
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    ArrayView2, Zip,
};
use num_traits::{AsPrimitive, PrimInt};
use rayon::slice::ParallelSliceMut;

/// Post processes boxes and scores tensors into quantized detection boxes,
/// filtering out any boxes below the score threshold. The boxes tensor
/// is converted to XYXY using the given BBoxTypeTrait. The order of the boxes
/// is preserved.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
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
    let isa = Isa::current_or_scalar();
    argmax_candidates_quant_with::<B, _, _, _>(isa, threshold, boxes, scores, quant_boxes, |b, _| b)
}

/// Post processes boxes and scores tensors into quantized detection boxes,
/// filtering out any boxes below the score threshold. The boxes tensor
/// is converted to XYXY using the given BBoxTypeTrait. The order of the boxes
/// is preserved.
///
/// This function is very similar to `postprocess_boxes_quant` but will also
/// return the index of the box. The boxes will be in ascending index order.
///
/// When scores originate from a transposed DMA-BUF view (stride-1 along axis 0),
/// an optimized column-major scan is used to avoid catastrophic strided reads on
/// uncacheable memory.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
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
    argmax_candidates_quant::<B, _, _>(
        Isa::current_or_scalar(),
        threshold,
        boxes,
        scores,
        quant_boxes,
    )
}

/// [`postprocess_boxes_index_quant`] on `isa`.
pub(crate) fn argmax_candidates_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<Scores>, usize)> {
    argmax_candidates_quant_with::<B, _, _, _>(
        isa,
        threshold,
        boxes,
        scores,
        quant_boxes,
        |b, i| (b, i),
    )
}

/// [`argmax_candidates_quant`] with each candidate and its anchor index mapped through `emit`.
pub(crate) fn argmax_candidates_quant_with<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
    O: Send,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
    emit: impl Fn(DetectBoxQuantized<Scores>, usize) -> O + Sync + Send,
) -> Vec<O> {
    // Detect transposed C-contiguous layout (e.g., from reversed_axes() on DMA-BUF).
    // In this layout columns are contiguous (stride-1 along axis 0) but rows are not.
    // The column-major path reads memory sequentially, avoiding cache-hostile strides.
    if scores.strides()[0] == 1 && scores.as_slice().is_none() {
        return argmax_candidates_quant_column_major::<B, _, _, _>(
            isa,
            threshold,
            boxes,
            scores,
            quant_boxes,
            emit,
        );
    }
    argmax_candidates_quant_rows::<B, _, _, _>(isa, threshold, boxes, scores, quant_boxes, emit)
}

/// The row-major scan of [`argmax_candidates_quant_with`].
fn argmax_candidates_quant_rows<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
    O: Send,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
    emit: impl Fn(DetectBoxQuantized<Scores>, usize) -> O + Sync + Send,
) -> Vec<O> {
    Zip::indexed(scores.rows())
        .and(boxes.rows())
        .into_par_iter()
        .with_min_len(ROWS_PER_TASK)
        .filter_map(|(index, score, bbox)| {
            let (score_, label) = row_arg_max(isa, score);
            if score_ < threshold {
                return None;
            }

            let bbox_quant = B::ndarray_to_xyxy_dequant(bbox.view(), quant_boxes);

            Some(emit(
                DetectBoxQuantized {
                    label,
                    score: score_,
                    bbox: BoundingBox::from(bbox_quant),
                },
                index,
            ))
        })
        .collect()
}

/// The argmax or multi-label candidates of a decode on `isa`, each with its anchor index mapped
/// through `emit`; an error for mismatched shapes.
pub(crate) fn decode_candidates_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
    O: Send,
>(
    isa: Isa,
    multi_label: bool,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
    emit: impl Fn(DetectBoxQuantized<Scores>, usize) -> O + Sync + Send,
) -> DecoderResult<Vec<O>> {
    check_candidate_shapes(boxes.dim(), scores.dim())?;
    Ok(if multi_label {
        multilabel_candidates_quant::<B, _, _>(isa, threshold, boxes, scores, quant_boxes)
            .into_iter()
            .map(|(b, i)| emit(b, i))
            .collect()
    } else {
        argmax_candidates_quant_with::<B, _, _, _>(isa, threshold, boxes, scores, quant_boxes, emit)
    })
}

/// Multi-label variant of [`postprocess_boxes_index_quant`]: one entry per
/// class at or above `threshold`, sharing the anchor's bbox and index.
///
/// Intended for validation decode; deployment uses the argmax variant.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
///
/// # Errors
/// [`DecoderError::InvalidShape`] unless boxes are `[N, 4]` and scores `[N, C]`.
pub fn postprocess_boxes_multilabel_index_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> DecoderResult<Vec<(DetectBoxQuantized<Scores>, usize)>> {
    check_candidate_shapes(boxes.dim(), scores.dim())?;
    Ok(multilabel_candidates_quant::<B, _, _>(
        Isa::current_or_scalar(),
        threshold,
        boxes,
        scores,
        quant_boxes,
    ))
}

/// [`postprocess_boxes_multilabel_index_quant`] on `isa`, shapes already checked.
pub(crate) fn multilabel_candidates_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<Scores>, usize)> {
    if scores.strides()[0] == 1 && scores.as_slice().is_none() {
        return multilabel_candidates_quant_column_major::<B, _, _>(
            isa,
            threshold,
            boxes,
            scores,
            quant_boxes,
        );
    }
    Zip::indexed(scores.rows())
        .and(boxes.rows())
        .into_par_iter()
        .with_min_len(ROWS_PER_TASK)
        .flat_map(|(anchor_idx, score_row, bbox_row)| {
            let mut classes = Vec::new();
            let ran = score_row.as_slice().map_or(Ran::Scalar, |r| {
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
                return Vec::new();
            }
            let bbox = BoundingBox::from(B::ndarray_to_xyxy_dequant(bbox_row.view(), quant_boxes));
            classes
                .into_iter()
                .filter_map(|label| {
                    let &score = score_row.get(label)?;
                    Some((DetectBoxQuantized { label, score, bbox }, anchor_idx))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// [`postprocess_boxes_multilabel_index_quant`] without anchor indices.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
///
/// # Errors
/// [`DecoderError::InvalidShape`] unless boxes are `[N, 4]` and scores `[N, C]`.
pub fn postprocess_boxes_multilabel_quant<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> DecoderResult<Vec<DetectBoxQuantized<Scores>>> {
    Ok(
        postprocess_boxes_multilabel_index_quant::<B, _, _>(threshold, boxes, scores, quant_boxes)?
            .into_iter()
            .map(|(b, _)| b)
            .collect(),
    )
}

/// Column-major path for [`postprocess_boxes_multilabel_index_quant`].
///
/// Reads each class column contiguously (the transposed DMA-BUF layout),
/// then orders hits by `(anchor, class)` to match the row-major output and
/// dequantizes each anchor's bbox once.
fn multilabel_candidates_quant_column_major<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<Scores>, usize)> {
    let neon = (!isa.is_scalar())
        .then(|| score_columns(&scores))
        .flatten()
        .and_then(|cols| column_hits(isa, &cols, threshold));
    let hits: Vec<(usize, usize, Scores)> = match neon {
        Some(hits) => hits
            .into_iter()
            .filter_map(|(a, c)| scores.get((a, c)).map(|&s| (a, c, s)))
            .collect(),
        None => {
            let mut hits: Vec<(usize, usize, Scores)> = (0..scores.dim().1)
                .into_par_iter()
                .flat_map_iter(|class_idx| {
                    let col = scores.column(class_idx);
                    let above =
                        |(i, &s): (usize, &Scores)| (s >= threshold).then_some((i, class_idx, s));
                    match col.as_slice() {
                        Some(slice) => slice
                            .iter()
                            .enumerate()
                            .filter_map(above)
                            .collect::<Vec<_>>(),
                        None => col.iter().enumerate().filter_map(above).collect(),
                    }
                })
                .collect();
            hits.par_sort_unstable_by_key(|&(anchor, class, _)| (anchor, class));
            hits
        }
    };

    let mut out = Vec::with_capacity(hits.len());
    let mut current: Option<(usize, BoundingBox)> = None;
    for (anchor, label, score) in hits {
        let bbox = match current {
            Some((a, bbox)) if a == anchor => bbox,
            _ => {
                let bbox =
                    BoundingBox::from(B::ndarray_to_xyxy_dequant(boxes.row(anchor), quant_boxes));
                current = Some((anchor, bbox));
                bbox
            }
        };
        out.push((DetectBoxQuantized { label, score, bbox }, anchor));
    }
    out
}

/// Candidate generators need `[N, 4]` boxes and `[N, C]` scores.
pub(crate) fn check_candidate_shapes(
    boxes: (usize, usize),
    scores: (usize, usize),
) -> DecoderResult<()> {
    if boxes.1 != 4 || boxes.0 != scores.0 {
        return Err(DecoderError::InvalidShape(format!(
            "candidate selection needs boxes [N, 4] and scores [N, C]; got boxes {boxes:?}, scores {scores:?}"
        )));
    }
    Ok(())
}

/// Column-major path for [`argmax_candidates_quant`].
///
/// When scores come from a transposed DMA-BUF view ([N_candidates, N_classes]
/// with strides [1, N_candidates]), row iteration causes N_classes reads each
/// N_candidates bytes apart — catastrophic on uncacheable memory. Instead, this
/// iterates over classes (columns, which are contiguous): on NEON tiers in
/// register tiles that keep each anchor's running argmax across every class,
/// otherwise with a running argmax per candidate in heap buffers.
fn argmax_candidates_quant_column_major<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
    O: Send,
>(
    isa: Isa,
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
    emit: impl Fn(DetectBoxQuantized<Scores>, usize) -> O + Sync + Send,
) -> Vec<O> {
    let (n_candidates, n_classes) = scores.dim();

    // Step 1: the anchors that pass, with their best class.
    let columns = (!isa.is_scalar()).then(|| score_columns(&scores)).flatten();
    let neon = columns.and_then(|cols| column_argmax(isa, &cols, threshold));
    let mut kept: Vec<(usize, usize, Scores)> = neon
        .iter()
        .flatten()
        .filter_map(|&(i, c)| scores.get((i, c)).map(|&s| (i, c, s)))
        .collect();
    if neon.is_none() {
        // u8 class indices; beyond 255 classes the row-major scan runs instead.
        if n_classes > 255 {
            return argmax_candidates_quant_rows::<B, _, _, _>(
                isa,
                threshold,
                boxes,
                scores,
                quant_boxes,
                emit,
            );
        }
        let mut max_scores = vec![Scores::min_value(); n_candidates];
        let mut max_classes = vec![0u8; n_candidates];
        for class_idx in 0..n_classes {
            let col = scores.column(class_idx);
            if let Some(slice) = col.as_slice() {
                for (i, &val) in slice.iter().enumerate() {
                    if val >= max_scores[i] {
                        max_scores[i] = val;
                        max_classes[i] = class_idx as u8;
                    }
                }
            } else {
                for (i, &val) in col.iter().enumerate() {
                    if val >= max_scores[i] {
                        max_scores[i] = val;
                        max_classes[i] = class_idx as u8;
                    }
                }
            }
        }
        kept = (0..n_candidates)
            .filter(|&i| max_scores[i] >= threshold)
            .map(|i| (i, usize::from(max_classes[i]), max_scores[i]))
            .collect();
    }

    // Step 2: Copy boxes column-by-column into contiguous heap buffer.
    // Boxes view is also transposed [N_candidates, 4] with strides [1, N_candidates],
    // so column reads are sequential while row reads are strided.
    let copy = !kept.is_empty() && boxes.strides()[0] == 1 && boxes.as_slice().is_none();
    let boxes_buf: [Vec<Boxes>; 4] = if copy {
        let mut cols: [Vec<Boxes>; 4] = [
            vec![Boxes::zero(); n_candidates],
            vec![Boxes::zero(); n_candidates],
            vec![Boxes::zero(); n_candidates],
            vec![Boxes::zero(); n_candidates],
        ];
        for (dim, col_buf) in cols.iter_mut().enumerate() {
            let col = boxes.column(dim);
            if let Some(slice) = col.as_slice() {
                col_buf.copy_from_slice(slice);
            } else {
                for (i, &val) in col.iter().enumerate() {
                    col_buf[i] = val;
                }
            }
        }
        cols
    } else {
        // Boxes are contiguous or differently strided — read per-candidate below.
        [vec![], vec![], vec![], vec![]]
    };
    let boxes_copied = !boxes_buf[0].is_empty();

    // Step 3: build the candidates that pass.
    let mut result = Vec::with_capacity(kept.len());
    for (i, label, score) in kept {
        let bbox_quant = if boxes_copied {
            let raw = [
                boxes_buf[0][i],
                boxes_buf[1][i],
                boxes_buf[2][i],
                boxes_buf[3][i],
            ];
            B::to_xyxy_dequant(&raw, quant_boxes)
        } else {
            B::ndarray_to_xyxy_dequant(boxes.row(i), quant_boxes)
        };
        result.push(emit(
            DetectBoxQuantized {
                label,
                score,
                bbox: BoundingBox::from(bbox_quant),
            },
            i,
        ));
    }

    result
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[doc(hidden)]
#[must_use]
pub fn nms_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    nms_int_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_int`] on `isa`.
#[must_use]
pub(crate) fn nms_int_on<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    isa: Isa,
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.

    boxes.par_sort_by(|a, b| b.score.cmp(&a.score));

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

    let min_val = SCORE::min_value();
    let cap = max_det.unwrap_or(usize::MAX);
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.bbox,
        false,
        |b| b.label,
        |b| b.score <= min_val,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.score > min_val, cap);
    }
    let mut survivors: usize = 0;
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
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    // Filter out boxes that were suppressed; cap because boxes after the
    // break may still hold positive scores but are all lower than survivors.
    boxes
        .into_iter()
        .filter(|b| b.score > min_val)
        .take(cap)
        .collect()
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
///
/// This is same as `nms_int` but will also include extra information along
/// with each box, such as the index
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[doc(hidden)]
#[must_use]
pub fn nms_extra_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    nms_extra_int_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_extra_int`] on `isa`.
#[must_use]
pub(crate) fn nms_extra_int_on<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send + Sync>(
    isa: Isa,
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.0.score.cmp(&a.0.score));

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

    let min_val = SCORE::min_value();
    let cap = max_det.unwrap_or(usize::MAX);
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.0.bbox,
        false,
        |b| b.0.label,
        |b| b.0.score <= min_val,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.0.score > min_val, cap);
    }
    let mut survivors: usize = 0;
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
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }

    // Filter out boxes that were suppressed; cap at `max_det`.
    boxes
        .into_iter()
        .filter(|b| b.0.score > min_val)
        .take(cap)
        .collect()
}

/// Class-aware NMS for quantized boxes: only suppress boxes with the same
/// label.
///
/// Sorts boxes by score, then greedily selects a subset of boxes in descending
/// order of score. Unlike class-agnostic NMS, boxes are only suppressed if they
/// have the same class label AND overlap above the IoU threshold.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[doc(hidden)]
#[must_use]
pub fn nms_class_aware_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    nms_class_aware_int_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_class_aware_int`] on `isa`.
#[must_use]
pub(crate) fn nms_class_aware_int_on<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    isa: Isa,
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    boxes.par_sort_by(|a, b| b.score.cmp(&a.score));

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

    let min_val = SCORE::min_value();
    let cap = max_det.unwrap_or(usize::MAX);
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.bbox,
        true,
        |b| b.label,
        |b| b.score <= min_val,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.score > min_val, cap);
    }
    let mut survivors: usize = 0;
    for i in 0..boxes.len() {
        if boxes[i].score <= min_val {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if boxes[j].score <= min_val {
                continue;
            }
            // Only suppress if same class AND overlapping
            if boxes[j].label == boxes[i].label && jaccard(&boxes[j].bbox, &boxes[i].bbox, iou) {
                boxes[j].score = min_val;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    boxes
        .into_iter()
        .filter(|b| b.score > min_val)
        .take(cap)
        .collect()
}

/// Class-aware NMS for quantized boxes with extra data: only suppress boxes
/// with the same label.
///
/// This is same as `nms_class_aware_int` but will also include extra
/// information along with each box, such as the index.
///
/// An unavailable or unknown tier forced by `EDGEFIRST_DECODER_FORCE_KERNEL` runs the scalar
/// code, which gives the same result.
#[doc(hidden)]
#[must_use]
pub fn nms_extra_class_aware_int<
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    E: Send + Sync,
>(
    iou: f32,
    max_det: Option<usize>,
    boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    nms_extra_class_aware_int_on(nms_isa(boxes.len()), iou, max_det, boxes)
}

/// [`nms_extra_class_aware_int`] on `isa`.
#[must_use]
pub(crate) fn nms_extra_class_aware_int_on<
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    E: Send + Sync,
>(
    isa: Isa,
    iou: f32,
    max_det: Option<usize>,
    mut boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    boxes.par_sort_by(|a, b| b.0.score.cmp(&a.0.score));

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

    let min_val = SCORE::min_value();
    let cap = max_det.unwrap_or(usize::MAX);
    if let Some(dead) = nms_dead(
        isa,
        &boxes,
        |b| &b.0.bbox,
        true,
        |b| b.0.label,
        |b| b.0.score <= min_val,
        iou,
        cap,
    ) {
        return keep_unflagged(boxes, dead, |b| b.0.score > min_val, cap);
    }
    let mut survivors: usize = 0;
    for i in 0..boxes.len() {
        if boxes[i].0.score <= min_val {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if boxes[j].0.score <= min_val {
                continue;
            }
            // Only suppress if same class AND overlapping
            if boxes[j].0.label == boxes[i].0.label
                && jaccard(&boxes[j].0.bbox, &boxes[i].0.bbox, iou)
            {
                boxes[j].0.score = min_val;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    boxes
        .into_iter()
        .filter(|b| b.0.score > min_val)
        .take(cap)
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{XYWH, XYXY};
    use ndarray::Array2;

    fn unit_quant() -> Quantization {
        Quantization {
            scale: 0.1,
            zero_point: 0,
        }
    }

    fn sorted_by_anchor_label(
        mut v: Vec<(DetectBoxQuantized<i8>, usize)>,
    ) -> Vec<(DetectBoxQuantized<i8>, usize)> {
        v.sort_by_key(|(b, i)| (*i, b.label));
        v
    }

    #[test]
    fn multilabel_index_quant_emits_every_class_above_threshold() {
        let boxes = ndarray::arr2(&[[10i8, 10, 20, 20], [30, 30, 40, 40]]);
        let scores = ndarray::arr2(&[[60i8, 10, 70], [5, 90, 5]]);
        let got = sorted_by_anchor_label(
            postprocess_boxes_multilabel_index_quant::<XYXY, _, _>(
                50,
                boxes.view(),
                scores.view(),
                unit_quant(),
            )
            .unwrap(),
        );
        let labels: Vec<(usize, usize)> = got.iter().map(|(b, i)| (*i, b.label)).collect();
        assert_eq!(labels, vec![(0, 0), (0, 2), (1, 1)]);
        assert_eq!(got[0].0.bbox, got[1].0.bbox);
        assert_eq!(got[1].0.score, 70);
    }

    #[test]
    fn multilabel_index_quant_column_major_matches_row_major() {
        let boxes = ndarray::arr2(&[[10i8, 10, 20, 20], [30, 30, 40, 40]]);
        let scores_t = ndarray::arr2(&[[60i8, 5], [10, 90], [70, 5]]);
        let scores_cm = scores_t.view().reversed_axes();
        let scores_rm = scores_cm.to_owned();
        let a = postprocess_boxes_multilabel_index_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores_cm,
            unit_quant(),
        )
        .unwrap();
        let b = postprocess_boxes_multilabel_index_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores_rm.view(),
            unit_quant(),
        )
        .unwrap();
        assert_eq!(sorted_by_anchor_label(a), sorted_by_anchor_label(b));
    }

    #[test]
    fn multilabel_index_quant_rejects_mismatched_shapes() {
        let boxes = ndarray::arr2(&[[10i8, 10, 20, 20]]);
        let scores = ndarray::arr2(&[[60i8, 10], [5, 90]]);
        let r = postprocess_boxes_multilabel_index_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores.view(),
            unit_quant(),
        );
        assert!(matches!(r, Err(crate::DecoderError::InvalidShape(_))));
    }

    #[test]
    fn multilabel_quant_rejects_non_four_column_boxes() {
        let boxes = ndarray::arr2(&[[10i8, 10, 20, 20, 0]]);
        let scores = ndarray::arr2(&[[60i8]]);
        let r = postprocess_boxes_multilabel_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores.view(),
            unit_quant(),
        );
        assert!(matches!(r, Err(crate::DecoderError::InvalidShape(_))));
    }

    #[test]
    fn multilabel_quant_drops_indices_only() {
        let boxes = ndarray::arr2(&[[10i8, 10, 20, 20], [30, 30, 40, 40]]);
        let scores = ndarray::arr2(&[[60i8, 10, 70], [5, 90, 5]]);
        let with_idx = postprocess_boxes_multilabel_index_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores.view(),
            unit_quant(),
        )
        .unwrap();
        let without = postprocess_boxes_multilabel_quant::<XYXY, _, _>(
            50,
            boxes.view(),
            scores.view(),
            unit_quant(),
        )
        .unwrap();
        let stripped: Vec<_> = with_idx.into_iter().map(|(b, _)| b).collect();
        assert_eq!(stripped, without);
    }

    /// Verify that the column-major path produces identical results to the
    /// row-major path for a transposed (non-contiguous) score array.
    /// Multi-label on a transposed (DMA-BUF style) score view must match the
    /// contiguous row-major result exactly, including order.
    fn multilabel_layouts_agree<S>(n_classes: usize, n_candidates: usize, threshold: S)
    where
        S: PrimInt + AsPrimitive<f32> + Send + Sync + std::fmt::Debug,
        usize: AsPrimitive<S>,
    {
        let mut scores_physical = Array2::<S>::zeros((n_classes, n_candidates));
        for c in 0..n_classes {
            for i in 0..n_candidates {
                scores_physical[[c, i]] = ((c * 3 + i * 7) % 100).as_();
            }
        }
        let mut boxes_physical = Array2::<i16>::zeros((4, n_candidates));
        for i in 0..n_candidates {
            boxes_physical[[0, i]] = (i * 10) as i16;
            boxes_physical[[1, i]] = (i * 20) as i16;
            boxes_physical[[2, i]] = (i * 10 + 50) as i16;
            boxes_physical[[3, i]] = (i * 20 + 100) as i16;
        }
        let quant = Quantization {
            scale: 0.00390625,
            zero_point: 0,
        };
        let scores_rm = scores_physical.clone().reversed_axes().to_owned();
        let boxes_rm = boxes_physical.clone().reversed_axes().to_owned();
        let row = postprocess_boxes_multilabel_index_quant::<XYWH, _, _>(
            threshold,
            boxes_rm.view(),
            scores_rm.view(),
            quant,
        )
        .unwrap();
        let scores_cm = scores_physical.view().reversed_axes();
        assert!(scores_cm.as_slice().is_none() && scores_cm.strides()[0] == 1);
        let col = postprocess_boxes_multilabel_index_quant::<XYWH, _, _>(
            threshold,
            boxes_physical.view().reversed_axes(),
            scores_cm,
            quant,
        )
        .unwrap();
        assert!(!row.is_empty());
        assert_eq!(row, col);
    }

    #[test]
    fn multilabel_column_major_matches_row_major_u8() {
        multilabel_layouts_agree::<u8>(80, 100, 60);
    }

    #[test]
    fn multilabel_column_major_matches_row_major_i8_many_classes() {
        multilabel_layouts_agree::<i8>(300, 64, 90);
    }

    #[test]
    fn candidate_shape_error_names_candidate_selection() {
        let err = check_candidate_shapes((3, 5), (3, 2)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("candidate selection"), "{msg}");
        assert!(!msg.contains("multi-label"), "{msg}");
    }

    #[test]
    fn column_major_matches_row_major() {
        // Create scores in "model output" layout: [num_classes, num_candidates]
        let n_classes = 80usize;
        let n_candidates = 100usize;
        let mut scores_physical = Array2::<u8>::zeros((n_classes, n_candidates));
        // Fill with known pattern: class c, candidate i → (c * 3 + i * 7) % 256
        for c in 0..n_classes {
            for i in 0..n_candidates {
                scores_physical[[c, i]] = ((c * 3 + i * 7) % 256) as u8;
            }
        }

        // Create boxes: [4, num_candidates] i16
        let mut boxes_physical = Array2::<i16>::zeros((4, n_candidates));
        for i in 0..n_candidates {
            boxes_physical[[0, i]] = (i * 10) as i16; // x
            boxes_physical[[1, i]] = (i * 20) as i16; // y
            boxes_physical[[2, i]] = (i * 10 + 50) as i16; // w
            boxes_physical[[3, i]] = (i * 20 + 100) as i16; // h
        }

        let quant = Quantization {
            scale: 0.00390625,
            zero_point: 0,
        };

        let threshold: u8 = 10;

        // Row-major path: contiguous [n_candidates, n_classes] array
        let scores_contiguous = scores_physical.clone().reversed_axes().to_owned();
        let boxes_contiguous = boxes_physical.clone().reversed_axes().to_owned();
        let row_result = postprocess_boxes_index_quant::<XYWH, _, _>(
            threshold,
            boxes_contiguous.view(),
            scores_contiguous.view(),
            quant,
        );

        // Column-major path: non-contiguous reversed view
        let scores_view = scores_physical.view().reversed_axes();
        let boxes_view = boxes_physical.view().reversed_axes();
        assert!(scores_view.as_slice().is_none(), "should be non-contiguous");
        assert_eq!(scores_view.strides()[0], 1);
        let col_result =
            postprocess_boxes_index_quant::<XYWH, _, _>(threshold, boxes_view, scores_view, quant);

        // Both paths should produce the same results
        assert_eq!(
            row_result.len(),
            col_result.len(),
            "different number of results: row={}, col={}",
            row_result.len(),
            col_result.len()
        );
        for (i, (row, col)) in row_result.iter().zip(col_result.iter()).enumerate() {
            assert_eq!(
                row.0.label, col.0.label,
                "candidate {i}: label mismatch row={} col={}",
                row.0.label, col.0.label
            );
            assert_eq!(row.0.score, col.0.score, "candidate {i}: score mismatch");
            assert_eq!(row.1, col.1, "candidate {i}: index mismatch");
            assert_eq!(row.0.bbox, col.0.bbox, "candidate {i}: bbox mismatch");
        }
    }

    /// Test column-major path with i8 scores (signed, matches NEON argmax path).
    #[test]
    fn column_major_matches_row_major_i8() {
        let n_classes = 80usize;
        let n_candidates = 50usize;
        let mut scores_physical = Array2::<i8>::zeros((n_classes, n_candidates));
        for c in 0..n_classes {
            for i in 0..n_candidates {
                scores_physical[[c, i]] = ((c as i16 * 3 + i as i16 * 7) % 256 - 128) as i8;
            }
        }

        let mut boxes_physical = Array2::<i16>::zeros((4, n_candidates));
        for i in 0..n_candidates {
            boxes_physical[[0, i]] = (i * 10) as i16;
            boxes_physical[[1, i]] = (i * 20) as i16;
            boxes_physical[[2, i]] = (i * 10 + 50) as i16;
            boxes_physical[[3, i]] = (i * 20 + 100) as i16;
        }

        let quant = Quantization {
            scale: 0.0256,
            zero_point: -116,
        };
        let threshold: i8 = -100;

        let scores_contiguous = scores_physical.clone().reversed_axes().to_owned();
        let boxes_contiguous = boxes_physical.clone().reversed_axes().to_owned();
        let row_result = postprocess_boxes_index_quant::<XYWH, _, _>(
            threshold,
            boxes_contiguous.view(),
            scores_contiguous.view(),
            quant,
        );

        let scores_view = scores_physical.view().reversed_axes();
        let boxes_view = boxes_physical.view().reversed_axes();
        let col_result =
            postprocess_boxes_index_quant::<XYWH, _, _>(threshold, boxes_view, scores_view, quant);

        assert_eq!(row_result.len(), col_result.len());
        for (i, (row, col)) in row_result.iter().zip(col_result.iter()).enumerate() {
            assert_eq!(row.0.label, col.0.label, "i8 candidate {i}: label mismatch");
            assert_eq!(row.0.score, col.0.score, "i8 candidate {i}: score mismatch");
            assert_eq!(row.1, col.1, "i8 candidate {i}: index mismatch");
        }
    }

    /// Helper: create `n` non-overlapping boxes with descending u8 scores.
    fn make_nms_boxes_int(n: usize) -> Vec<DetectBoxQuantized<u8>> {
        (0..n)
            .map(|i| DetectBoxQuantized {
                bbox: BoundingBox {
                    xmin: i as f32 * 100.0,
                    ymin: 0.0,
                    xmax: i as f32 * 100.0 + 10.0,
                    ymax: 10.0,
                },
                label: 0,
                score: (200 - i as u32).min(255) as u8,
            })
            .collect()
    }

    /// Four candidates' worth of boxes, one row each, values distinct enough
    /// that a mis-indexed read is visible.
    fn probe_boxes(n: usize) -> Array2<i16> {
        let mut boxes = Array2::<i16>::zeros((n, 4));
        for i in 0..n {
            boxes[[i, 0]] = i as i16 * 10;
            boxes[[i, 1]] = i as i16 * 10 + 1;
            boxes[[i, 2]] = i as i16 * 10 + 2;
            boxes[[i, 3]] = i as i16 * 10 + 3;
        }
        boxes
    }

    fn probe_quant() -> Quantization {
        Quantization {
            scale: 1.0,
            zero_point: 0,
        }
    }

    /// The threshold is a floor: a score equal to it is a detection. Both
    /// entry points share the rule and neither had a test that sat exactly on
    /// it.
    #[test]
    fn postprocess_boxes_quant_keeps_a_score_equal_to_the_threshold() {
        let boxes = probe_boxes(2);
        let scores = ndarray::array![[200_u8, 0], [199, 0]];

        let kept = postprocess_boxes_quant::<XYXY, _, _>(
            200_u8,
            boxes.view(),
            scores.view(),
            probe_quant(),
        );
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].score, 200);

        let kept = postprocess_boxes_index_quant::<XYXY, _, _>(
            200_u8,
            boxes.view(),
            scores.view(),
            probe_quant(),
        );
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert_eq!(kept[0].1, 0);
    }

    /// Class indices in the scalar column-major scan are `u8`, so beyond 255
    /// classes it has to hand over to the row-major scan. Running the column
    /// scan anyway wraps the label: class 260 comes back as 4.
    #[test]
    fn column_major_falls_back_beyond_255_classes() {
        let n_classes = 300usize;
        let boxes = probe_boxes(2);
        let mut scores = Array2::<u8>::zeros((2, n_classes));
        scores[[0, 260]] = 200;
        scores[[1, 3]] = 10;

        for (_, isa) in crate::neon_dispatch::tiers() {
            let kept = argmax_candidates_quant_column_major::<XYXY, _, _, _>(
                isa,
                200_u8,
                boxes.view(),
                scores.view(),
                probe_quant(),
                |b, i| (b, i),
            );
            assert_eq!(kept.len(), 1, "{isa:?} {kept:?}");
            assert_eq!(kept[0].0.label, 260, "{isa:?}");
            assert_eq!(kept[0].0.score, 200, "{isa:?}");
            assert_eq!(kept[0].1, 0, "{isa:?}");
        }
    }

    /// Driving the column-major scan with an ordinary row-major array: its
    /// columns are not contiguous, so the running argmax takes its strided
    /// branch, and the boxes are read in place rather than from the
    /// column-copy buffers.
    #[test]
    fn column_major_scan_handles_row_major_input() {
        let boxes = probe_boxes(3);
        // Row-major (3 candidates, 4 classes); the winner differs per row and
        // is never class 0, so a comparison that keeps the minimum is visible.
        let scores = ndarray::array![[10_u8, 90, 30, 40], [5, 6, 7, 200], [1, 2, 3, 4]];

        for (_, isa) in crate::neon_dispatch::tiers() {
            let kept = argmax_candidates_quant_column_major::<XYXY, _, _, _>(
                isa,
                50_u8,
                boxes.view(),
                scores.view(),
                probe_quant(),
                |b, i| (b, i),
            );
            assert_eq!(kept.len(), 2, "{isa:?} {kept:?}");
            assert_eq!((kept[0].0.label, kept[0].0.score), (1, 90));
            assert_eq!((kept[1].0.label, kept[1].0.score), (3, 200));
            // Boxes were read in place, so row 1's values must line up with it.
            assert_eq!(kept[1].0.bbox.xmin, 10.0);
            assert_eq!((kept[0].1, kept[1].1), (0, 1));
        }
    }

    /// Two overlapping same-class boxes. `SCORE::min_value()` is the
    /// suppression marker, so for `u8` a score of 0 already means suppressed
    /// and the lower box here is given a real score.
    fn overlapping_pair_int() -> Vec<DetectBoxQuantized<u8>> {
        vec![
            DetectBoxQuantized {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 10.0,
                    ymax: 10.0,
                },
                label: 0,
                score: 200,
            },
            DetectBoxQuantized {
                bbox: BoundingBox {
                    xmin: 1.0,
                    ymin: 1.0,
                    xmax: 11.0,
                    ymax: 11.0,
                },
                label: 0,
                score: 180,
            },
        ]
    }

    /// A overlaps B, B overlaps C, A does not reach C.
    fn suppression_chain_int() -> Vec<DetectBoxQuantized<u8>> {
        let mk = |xmin: f32, xmax: f32, score: u8| DetectBoxQuantized {
            bbox: BoundingBox {
                xmin,
                ymin: 0.0,
                xmax,
                ymax: 10.0,
            },
            label: 0,
            score,
        };
        vec![mk(0.0, 10.0, 200), mk(2.0, 12.0, 180), mk(4.0, 14.0, 160)]
    }

    #[test]
    fn nms_class_aware_int_suppresses_same_class_without_a_cap() {
        let result = nms_class_aware_int(0.5, None, overlapping_pair_int());
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].score, 200);
    }

    #[test]
    fn nms_class_aware_int_keeps_overlapping_boxes_of_other_classes() {
        let mut boxes = overlapping_pair_int();
        boxes[1].label = 1;
        let result = nms_class_aware_int(0.5, None, boxes);
        assert_eq!(result.len(), 2, "{result:?}");
    }

    #[test]
    fn nms_class_aware_int_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let result = nms_class_aware_int(0.5, None, suppression_chain_int());
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].score, 200);
        assert_eq!(result[1].score, 160);
    }

    /// A suppressed box is marked with `min_value()`, and the final filter
    /// drops it. Keeping `>= min_value` instead would hand every suppressed
    /// box back to the caller.
    #[test]
    fn nms_class_aware_int_drops_boxes_already_at_the_minimum_score() {
        let mut boxes = overlapping_pair_int();
        boxes.push(DetectBoxQuantized {
            bbox: BoundingBox {
                xmin: 50.0,
                ymin: 50.0,
                xmax: 60.0,
                ymax: 60.0,
            },
            label: 0,
            score: u8::MIN,
        });
        let result = nms_class_aware_int(0.5, None, boxes);
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].score, 200);
    }

    /// A box that overlaps nothing, then a pair that overlaps each other. The
    /// first box confirms a survivor without suppressing anything, so a loop
    /// that stops at the first survivor leaves the pair un-deduplicated while
    /// the chain above, where the first box does the suppressing, looks
    /// identical either way.
    fn isolated_then_overlapping_pair_int() -> Vec<DetectBoxQuantized<u8>> {
        let mk = |xmin: f32, xmax: f32, score: u8| DetectBoxQuantized {
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
            mk(100.0, 110.0, 200),
            mk(0.0, 10.0, 180),
            mk(1.0, 11.0, 160),
        ]
    }

    #[test]
    fn nms_class_aware_int_keeps_suppressing_after_the_first_survivor() {
        let result = nms_class_aware_int(0.5, None, isolated_then_overlapping_pair_int());
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].score, 200);
        assert_eq!(result[1].score, 180);
    }

    fn with_index_int(boxes: Vec<DetectBoxQuantized<u8>>) -> Vec<(DetectBoxQuantized<u8>, usize)> {
        boxes.into_iter().enumerate().map(|(i, b)| (b, i)).collect()
    }

    #[test]
    fn nms_extra_class_aware_int_keeps_suppressing_after_the_first_survivor() {
        let result = nms_extra_class_aware_int(
            0.5,
            None,
            with_index_int(isolated_then_overlapping_pair_int()),
        );
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[1].1, 1);
    }

    #[test]
    fn nms_extra_class_aware_int_suppresses_same_class_without_a_cap() {
        let result = nms_extra_class_aware_int(0.5, None, with_index_int(overlapping_pair_int()));
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].0.score, 200);
        assert_eq!(result[0].1, 0, "the surviving box keeps its extra data");
    }

    #[test]
    fn nms_extra_class_aware_int_keeps_overlapping_boxes_of_other_classes() {
        let mut boxes = overlapping_pair_int();
        boxes[1].label = 1;
        let result = nms_extra_class_aware_int(0.5, None, with_index_int(boxes));
        assert_eq!(result.len(), 2, "{result:?}");
    }

    #[test]
    fn nms_extra_class_aware_int_does_not_suppress_on_behalf_of_a_suppressed_box() {
        let result = nms_extra_class_aware_int(0.5, None, with_index_int(suppression_chain_int()));
        assert_eq!(result.len(), 2, "{result:?}");
        assert_eq!(result[0].1, 0);
        assert_eq!(result[1].1, 2);
    }

    #[test]
    fn nms_extra_class_aware_int_drops_boxes_already_at_the_minimum_score() {
        let mut boxes = overlapping_pair_int();
        boxes.push(DetectBoxQuantized {
            bbox: BoundingBox {
                xmin: 50.0,
                ymin: 50.0,
                xmax: 60.0,
                ymax: 60.0,
            },
            label: 0,
            score: u8::MIN,
        });
        let result = nms_extra_class_aware_int(0.5, None, with_index_int(boxes));
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].0.score, 200);
    }

    /// A single box is never compared with itself: the inner loop starts one
    /// past the outer index. Starting at `i` instead makes every box suppress
    /// itself, since a box's IoU with itself is 1.
    #[test]
    fn nms_extra_class_aware_int_keeps_a_lone_box() {
        let boxes = with_index_int(vec![overlapping_pair_int().remove(0)]);
        let result = nms_extra_class_aware_int(0.5, None, boxes);
        assert_eq!(result.len(), 1, "{result:?}");
        assert_eq!(result[0].0.score, 200);
    }

    #[test]
    fn nms_int_max_det_matches_full_truncated() {
        let boxes = make_nms_boxes_int(20);
        let n = 5;
        let full = nms_int(0.5, None, boxes.clone());
        let capped = nms_int(0.5, Some(n), boxes);
        assert_eq!(capped.len(), n);
        assert_eq!(&full[..n], &capped[..]);
    }

    #[test]
    fn nms_int_max_det_zero_returns_empty() {
        let boxes = make_nms_boxes_int(10);
        let result = nms_int(0.5, Some(0), boxes);
        assert!(result.is_empty());
    }

    #[test]
    fn nms_int_max_det_iou_ge_1_returns_sorted_truncated() {
        let boxes = make_nms_boxes_int(10);
        let result = nms_int(1.0, Some(3), boxes);
        assert_eq!(result.len(), 3);
        // Scores should be in descending order (sorted).
        assert!(result[0].score >= result[1].score);
        assert!(result[1].score >= result[2].score);
    }

    #[test]
    fn nms_int_max_det_larger_than_input() {
        let boxes = make_nms_boxes_int(5);
        let full = nms_int(0.5, None, boxes.clone());
        let capped = nms_int(0.5, Some(100), boxes);
        assert_eq!(full.len(), capped.len());
    }
}
