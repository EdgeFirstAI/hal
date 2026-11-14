// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between the per-scale fast path's `DecodedOutputsRef` and the
//! existing post-NMS pipeline (score filter → top-K → NMS → optional mask
//! materialisation).
//!
//! Boxes and scores reach the candidate filter in the dtype the per-scale
//! pipeline decoded them in: f16 scores are compared as f16, only the
//! candidates' boxes and scores are converted to f32 (exactly), and on a CPU
//! with FP16 arithmetic the NMS overlap tests run in FP16. Mask coefficients
//! and protos are always f32, the type the mask kernels take.

use half::f16;
use ndarray::{ArrayView2, ArrayView3, Axis};
use tracing::trace_span;

use crate::neon_dispatch::Isa;
use crate::per_scale::outputs::{BufferRef, DecodedOutputsRef};
use crate::yolo::{
    extract_proto_data_float, impl_yolo_segdet_get_boxes_on, impl_yolo_split_segdet_process_masks,
};
use crate::{DecoderError, DetectBox, Nms, ProtoData, Segmentation, XYWH};

/// The lowest f16 at or above `t`: an f16 score passes it exactly when its
/// value passes `t`.
fn f16_threshold(t: f32) -> f16 {
    let h = f16::from_f32(t);
    if h.to_f32() >= t || h.is_nan() {
        return h;
    }
    let bits = h.to_bits();
    f16::from_bits(match bits {
        0x8000 => 0x0001,
        b if b & 0x8000 != 0 => b - 1,
        b => b + 1,
    })
}

/// Score filter, pre-NMS cap and NMS over the decoded boxes `(xc, yc, w, h)`
/// and scores, in their own dtype.
#[allow(clippy::too_many_arguments)]
fn get_boxes(
    decoded: &DecodedOutputsRef<'_>,
    score_threshold: f32,
    iou_threshold: f32,
    nms_mode: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    multi_label: bool,
) -> Result<Vec<(DetectBox, usize)>, DecoderError> {
    let (n, nc) = (decoded.total_anchors, decoded.num_classes);
    let shape = |e: ndarray::ShapeError| DecoderError::Internal(format!("per_scale view: {e}"));
    let isa = Isa::current()?;
    match (&decoded.boxes, &decoded.scores) {
        (BufferRef::F32(boxes), BufferRef::F32(scores)) => {
            impl_yolo_segdet_get_boxes_on::<XYWH, _, _>(
                isa,
                ArrayView2::from_shape((n, 4), boxes).map_err(shape)?,
                ArrayView2::from_shape((n, nc), scores).map_err(shape)?,
                score_threshold,
                iou_threshold,
                nms_mode,
                pre_nms_top_k,
                max_det,
                multi_label,
            )
        }
        (BufferRef::F16(boxes), BufferRef::F16(scores)) => {
            impl_yolo_segdet_get_boxes_on::<XYWH, _, _>(
                isa.with_f16_nms(),
                ArrayView2::from_shape((n, 4), boxes).map_err(shape)?,
                ArrayView2::from_shape((n, nc), scores).map_err(shape)?,
                f16_threshold(score_threshold),
                iou_threshold,
                nms_mode,
                pre_nms_top_k,
                max_det,
                multi_label,
            )
        }
        _ => Err(DecoderError::Internal(
            "per_scale boxes and scores differ in dtype".into(),
        )),
    }
}

/// The mask coefficients `[N, NM]` and protos `(H, W, NM)` of a segmentation
/// model; `None` for a detection-only model.
type MaskViews<'a> = Option<(ArrayView2<'a, f32>, ArrayView3<'a, f32>)>;

fn mask_views<'a>(decoded: &'a DecodedOutputsRef<'a>) -> Result<MaskViews<'a>, DecoderError> {
    let (Some(mask_coefs), Some(protos)) = (decoded.mask_coefs, decoded.protos.as_ref()) else {
        return Ok(None);
    };
    let mc = ArrayView2::from_shape((decoded.total_anchors, decoded.num_mask_coefs), mask_coefs)
        .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
    // Protos are `[1, H, W, NM]` (NHWC); the mask kernels take `(H, W, NM)`.
    Ok(Some((mc, protos.index_axis(Axis(0), 0))))
}

/// Bridge: per-scale outputs → optional `ProtoData`, with `output_boxes`
/// populated post-NMS. Mirrors the contract of `decode_proto`.
#[allow(clippy::too_many_arguments)]
pub(super) fn per_scale_to_proto_data<'a>(
    decoded: &'a DecodedOutputsRef<'a>,
    output_boxes: &mut Vec<DetectBox>,
    iou_threshold: f32,
    score_threshold: f32,
    nms_mode: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    normalized: Option<bool>,
    input_dims: Option<(usize, usize)>,
    multi_label: bool,
) -> Result<Option<ProtoData>, DecoderError> {
    let _span = trace_span!("decoder.decode_proto.per_scale_to_proto_data").entered();
    output_boxes.clear();

    let mut det_indices = {
        let _s = trace_span!("decoder.decode_proto.per_scale_to_proto_data.get_boxes").entered();
        get_boxes(
            decoded,
            score_threshold,
            iou_threshold,
            nms_mode,
            pre_nms_top_k,
            max_det,
            multi_label,
        )?
    };
    // Per-scale `dist2bbox_anchor_*` emits pixel-space coords by design.
    // Apply EDGEAI-1303 normalization so output bboxes land in the
    // canonical [0, 1] range expected by Decoder callers.
    crate::yolo::maybe_normalize_boxes_in_place(&mut det_indices, normalized, input_dims);

    match mask_views(decoded)? {
        Some((mc_view, pr_view)) => {
            let _s = trace_span!("decoder.decode_proto.per_scale_to_proto_data.extract").entered();
            let proto_data = extract_proto_data_float(det_indices, mc_view, pr_view, output_boxes);
            Ok(Some(proto_data))
        }
        None => {
            // Detection-only: copy boxes into output and report no proto data.
            for (db, _) in det_indices {
                output_boxes.push(db);
            }
            Ok(None)
        }
    }
}

/// Bridge: per-scale outputs → materialised masks (`Segmentation`),
/// with `output_boxes` populated post-NMS. Mirrors the contract of
/// `decode`.
#[allow(clippy::too_many_arguments)]
pub(super) fn per_scale_to_masks<'a>(
    decoded: &'a DecodedOutputsRef<'a>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
    iou_threshold: f32,
    score_threshold: f32,
    nms_mode: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
    normalized: Option<bool>,
    input_dims: Option<(usize, usize)>,
    multi_label: bool,
) -> Result<(), DecoderError> {
    let _span = trace_span!("decoder.decode.per_scale_to_masks").entered();
    output_boxes.clear();
    output_masks.clear();

    let mut det_indices = {
        let _s = trace_span!("decoder.decode.per_scale_to_masks.get_boxes").entered();
        get_boxes(
            decoded,
            score_threshold,
            iou_threshold,
            nms_mode,
            pre_nms_top_k,
            max_det,
            multi_label,
        )?
    };
    // Per-scale `dist2bbox_anchor_*` emits pixel-space coords by design.
    // Normalize before mask processing so `protobox` sees [0, 1] coords
    // and the returned bboxes match the contract of `Decoder::decode`.
    crate::yolo::maybe_normalize_boxes_in_place(&mut det_indices, normalized, input_dims);

    match mask_views(decoded)? {
        Some((mc_view, pr_view)) => {
            let _s = trace_span!("decoder.decode.per_scale_to_masks.process_masks").entered();
            impl_yolo_split_segdet_process_masks(
                det_indices,
                mc_view,
                pr_view,
                output_boxes,
                output_masks,
            )
        }
        None => {
            // Detection-only: emit boxes; no masks.
            for (db, _) in det_indices {
                output_boxes.push(db);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Every f16 value passes `f16_threshold(t)` exactly when its f32 value passes `t`.
    #[test]
    fn f16_threshold_keeps_the_f32_decision_for_every_f16() {
        let thresholds = [
            0.001,
            0.01,
            0.1,
            0.25,
            0.3,
            0.5,
            0.7,
            0.999,
            1.0,
            0.0,
            -0.0,
            -0.25,
            1e-9,
            -1e-9,
            65504.0,
            65520.0,
            1e9,
            -1e9,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for t in thresholds {
            let t16 = f16_threshold(t);
            for bits in 0..=u16::MAX {
                let score = f16::from_bits(bits);
                assert_eq!(
                    score >= t16,
                    score.to_f32() >= t,
                    "score {score} threshold {t}"
                );
            }
        }
        assert!(f16_threshold(f32::NAN).is_nan());
    }
}
