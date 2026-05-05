// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(target_arch = "aarch64")]
use crate::arg_max_i8;
use crate::{
    arg_max, float::jaccard, BBoxTypeTrait, BoundingBox, DetectBoxQuantized, Quantization,
};
use ndarray::{
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    Array1, ArrayView1, ArrayView2, Zip,
};
use num_traits::{AsPrimitive, PrimInt};
use rayon::slice::ParallelSliceMut;

/// NEON-accelerated column max update for the column-major argmax path.
///
/// Processes 16 elements per iteration using SIMD max + bitwise select.
/// Handles both unsigned (u8) and signed (i8) comparison semantics.
///
/// # Safety
///
/// - `col_ptr` must point to at least `n` valid bytes.
/// - `max_ptr` must point to at least `n` valid mutable bytes.
/// - `class_ptr` must point to at least `n` valid mutable bytes.
#[cfg(target_arch = "aarch64")]
unsafe fn column_max_update_neon(
    col_ptr: *const u8,
    max_ptr: *mut u8,
    class_ptr: *mut u8,
    n: usize,
    class_idx: u8,
    signed: bool,
) {
    use std::arch::aarch64::*;

    let class_vec = vdupq_n_u8(class_idx);
    let chunks = n / 16;
    let remainder = n % 16;

    if signed {
        // Signed i8 comparison: interpret bytes as i8.
        for chunk in 0..chunks {
            let offset = chunk * 16;
            let col = vld1q_s8(col_ptr.add(offset) as *const i8);
            let cur_max = vld1q_s8(max_ptr.add(offset) as *const i8);
            // mask[i] = 0xFF where col[i] >= cur_max[i], else 0x00
            let mask = vcgeq_s8(col, cur_max);
            // new_max = max(col, cur_max)
            let new_max = vmaxq_s8(col, cur_max);
            vst1q_s8(max_ptr.add(offset) as *mut i8, new_max);
            // Select class_idx where mask is set, keep old class otherwise.
            let cur_class = vld1q_u8(class_ptr.add(offset));
            let new_class = vbslq_u8(mask, class_vec, cur_class);
            vst1q_u8(class_ptr.add(offset), new_class);
        }
        // Scalar tail.
        for i in (chunks * 16)..n {
            let val = *(col_ptr.add(i) as *const i8);
            let cur = *(max_ptr.add(i) as *const i8);
            if val >= cur {
                *(max_ptr.add(i) as *mut i8) = val;
                *class_ptr.add(i) = class_idx;
            }
        }
    } else {
        // Unsigned u8 comparison.
        for chunk in 0..chunks {
            let offset = chunk * 16;
            let col = vld1q_u8(col_ptr.add(offset));
            let cur_max = vld1q_u8(max_ptr.add(offset));
            let mask = vcgeq_u8(col, cur_max);
            let new_max = vmaxq_u8(col, cur_max);
            vst1q_u8(max_ptr.add(offset), new_max);
            let cur_class = vld1q_u8(class_ptr.add(offset));
            let new_class = vbslq_u8(mask, class_vec, cur_class);
            vst1q_u8(class_ptr.add(offset), new_class);
        }
        // Scalar tail.
        for i in (chunks * 16)..n {
            let val = *col_ptr.add(i);
            let cur = *max_ptr.add(i);
            if val >= cur {
                *max_ptr.add(i) = val;
                *class_ptr.add(i) = class_idx;
            }
        }
    }
    let _ = remainder; // suppress unused warning
}

/// Fast argmax dispatching to NEON-optimized path for i8 on aarch64.
#[inline(always)]
fn fast_arg_max<T: PrimInt + Copy>(score: ArrayView1<T>) -> (T, usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // Check if this is an i8 slice and contiguous.
        if std::mem::size_of::<T>() == 1 && score.as_slice().is_some() {
            let slice = score.as_slice().unwrap();
            // Safety: T is i8 when size_of::<T>() == 1 and PrimInt.
            // PrimInt covers i8, u8, i16, etc. We only want to use the
            // i8 NEON path for signed i8.
            let ptr = slice.as_ptr() as *const i8;
            let i8_slice = unsafe { std::slice::from_raw_parts(ptr, slice.len()) };
            // Only valid for signed i8 (not u8). Check sign bit behavior:
            // PrimInt for i8 means min_value() is negative.
            if T::min_value() < T::zero() {
                let (max_val, idx) = arg_max_i8(i8_slice);
                // Safety: transmute i8 back to T (they have the same size and
                // representation for i8).
                let result: T = unsafe { std::mem::transmute_copy(&max_val) };
                return (result, idx);
            }
        }
    }
    arg_max(score)
}

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

    // Use column-major path for transposed DMA-BUF views (see postprocess_boxes_index_quant).
    if scores.strides()[0] == 1 && scores.as_slice().is_none() {
        return postprocess_boxes_quant_column_major::<B, _, _>(
            threshold,
            boxes,
            scores,
            quant_boxes,
        );
    }

    Zip::from(scores.rows())
        .and(boxes.rows())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) = fast_arg_max(score);
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

/// Column-major optimized path for `postprocess_boxes_quant`.
fn postprocess_boxes_quant_column_major<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<DetectBoxQuantized<Scores>> {
    let (n_candidates, n_classes) = scores.dim();

    // Column-major NEON path uses u8 class indices; fall back for >255 classes.
    if n_classes > 255 {
        return Zip::from(scores.rows())
            .and(boxes.rows())
            .into_par_iter()
            .filter_map(|(score, bbox)| {
                let (score_, label) = fast_arg_max(score);
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
            .collect();
    }
    let mut max_scores = vec![Scores::min_value(); n_candidates];
    let mut max_classes = vec![0u8; n_candidates];

    for class_idx in 0..n_classes {
        let col = scores.column(class_idx);
        if let Some(slice) = col.as_slice() {
            #[cfg(target_arch = "aarch64")]
            {
                if std::mem::size_of::<Scores>() == 1 {
                    unsafe {
                        column_max_update_neon(
                            slice.as_ptr() as *const u8,
                            max_scores.as_mut_ptr() as *mut u8,
                            max_classes.as_mut_ptr(),
                            n_candidates,
                            class_idx as u8,
                            Scores::min_value() < Scores::zero(),
                        );
                    }
                    continue;
                }
            }
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

    // Copy boxes column-by-column if also transposed.
    let boxes_buf: [Vec<Boxes>; 4] = if boxes.strides()[0] == 1 && boxes.as_slice().is_none() {
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
        [vec![], vec![], vec![], vec![]]
    };
    let boxes_copied = !boxes_buf[0].is_empty();

    let mut result = Vec::new();
    for i in 0..n_candidates {
        if max_scores[i] >= threshold {
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
            result.push(DetectBoxQuantized {
                label: max_classes[i] as usize,
                score: max_scores[i],
                bbox: BoundingBox::from(bbox_quant),
            });
        }
    }

    result
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

    // Detect transposed C-contiguous layout (e.g., from reversed_axes() on DMA-BUF).
    // In this layout columns are contiguous (stride-1 along axis 0) but rows are not.
    // The column-major path reads memory sequentially, avoiding cache-hostile strides.
    if scores.strides()[0] == 1 && scores.as_slice().is_none() {
        return postprocess_boxes_index_quant_column_major::<B, _, _>(
            threshold,
            boxes,
            scores,
            quant_boxes,
        );
    }

    let indices: Array1<usize> = (0..boxes.dim().0).collect();
    Zip::from(scores.rows())
        .and(boxes.rows())
        .and(&indices)
        .into_par_iter()
        .filter_map(|(score, bbox, index)| {
            let (score_, label) = fast_arg_max(score);
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

/// Column-major optimized path for `postprocess_boxes_index_quant`.
///
/// When scores come from a transposed DMA-BUF view ([N_candidates, N_classes]
/// with strides [1, N_candidates]), row iteration causes N_classes reads each
/// N_candidates bytes apart — catastrophic on uncacheable memory. Instead, this
/// iterates over classes (columns, which are contiguous), maintaining a running
/// argmax per candidate in cacheable heap buffers.
fn postprocess_boxes_index_quant_column_major<
    B: BBoxTypeTrait,
    Boxes: PrimInt + AsPrimitive<f32> + Send + Sync,
    Scores: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    threshold: Scores,
    boxes: ArrayView2<Boxes>,
    scores: ArrayView2<Scores>,
    quant_boxes: Quantization,
) -> Vec<(DetectBoxQuantized<Scores>, usize)> {
    let (n_candidates, n_classes) = scores.dim();

    // Phase 1: Column-based argmax — sequential reads over contiguous columns.
    // Use u8 for class indices (max 255 classes) for optimal NEON vectorization.
    // Fall back to row-major path for models with >255 classes.
    if n_classes > 255 {
        let indices: Array1<usize> = (0..n_candidates).collect();
        return Zip::from(scores.rows())
            .and(boxes.rows())
            .and(&indices)
            .into_par_iter()
            .filter_map(|(score, bbox, index)| {
                let (score_, label) = fast_arg_max(score);
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
            .collect();
    }
    let mut max_scores = vec![Scores::min_value(); n_candidates];
    let mut max_classes = vec![0u8; n_candidates];
    for class_idx in 0..n_classes {
        let col = scores.column(class_idx);
        if let Some(slice) = col.as_slice() {
            // Use NEON-accelerated column max update on aarch64 for u8 scores.
            #[cfg(target_arch = "aarch64")]
            {
                if std::mem::size_of::<Scores>() == 1 {
                    // SAFETY: Scores is u8 or i8 (size == 1). We transmute the
                    // slice pointers to the concrete byte type for NEON processing.
                    unsafe {
                        column_max_update_neon(
                            slice.as_ptr() as *const u8,
                            max_scores.as_mut_ptr() as *mut u8,
                            max_classes.as_mut_ptr(),
                            n_candidates,
                            class_idx as u8,
                            Scores::min_value() < Scores::zero(), // signed flag
                        );
                    }
                    continue;
                }
            }
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

    // Phase 2: Copy boxes column-by-column into contiguous heap buffer.
    // Boxes view is also transposed [N_candidates, 4] with strides [1, N_candidates],
    // so column reads are sequential while row reads are strided.
    let boxes_buf: [Vec<Boxes>; 4] = if boxes.strides()[0] == 1 && boxes.as_slice().is_none() {
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

    // Phase 3: Threshold filter — collect candidates that pass.
    let mut result = Vec::new();
    for i in 0..n_candidates {
        if max_scores[i] >= threshold {
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
            result.push((
                DetectBoxQuantized {
                    label: max_classes[i] as usize,
                    score: max_scores[i],
                    bbox: BoundingBox::from(bbox_quant),
                },
                i,
            ));
        }
    }

    result
}

/// Uses NMS to filter boxes based on the score and iou. Sorts boxes by score,
/// then greedily selects a subset of boxes in descending order of score.
#[doc(hidden)]
#[must_use]
pub fn nms_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    iou: f32,
    mut boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.

    boxes.par_sort_by(|a, b| b.score.cmp(&a.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

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
#[must_use]
pub fn nms_extra_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync, E: Send + Sync>(
    iou: f32,
    mut boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.par_sort_by(|a, b| b.0.score.cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

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

/// Class-aware NMS for quantized boxes: only suppress boxes with the same
/// label.
///
/// Sorts boxes by score, then greedily selects a subset of boxes in descending
/// order of score. Unlike class-agnostic NMS, boxes are only suppressed if they
/// have the same class label AND overlap above the IoU threshold.
#[doc(hidden)]
#[must_use]
pub fn nms_class_aware_int<SCORE: PrimInt + AsPrimitive<f32> + Send + Sync>(
    iou: f32,
    mut boxes: Vec<DetectBoxQuantized<SCORE>>,
) -> Vec<DetectBoxQuantized<SCORE>> {
    boxes.par_sort_by(|a, b| b.score.cmp(&a.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

    let min_val = SCORE::min_value();
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
    }
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

/// Class-aware NMS for quantized boxes with extra data: only suppress boxes
/// with the same label.
///
/// This is same as `nms_class_aware_int` but will also include extra
/// information along with each box, such as the index.
#[doc(hidden)]
#[must_use]
pub fn nms_extra_class_aware_int<
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
    E: Send + Sync,
>(
    iou: f32,
    mut boxes: Vec<(DetectBoxQuantized<SCORE>, E)>,
) -> Vec<(DetectBoxQuantized<SCORE>, E)> {
    boxes.par_sort_by(|a, b| b.0.score.cmp(&a.0.score));

    // When the iou is 1.0 or larger, no boxes will be filtered so we just return
    // immediately
    if iou >= 1.0 {
        return boxes;
    }

    let min_val = SCORE::min_value();
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
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::XYWH;
    use ndarray::Array2;

    /// Verify that the column-major path produces identical results to the
    /// row-major path for a transposed (non-contiguous) score array.
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
}
