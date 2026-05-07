// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between the per-scale fast path's `DecodedOutputsRef` and the
//! existing post-NMS pipeline (score filter → top-K → NMS → optional mask
//! materialisation).
//!
//! Phase 1 simplification: always widen `f16` outputs to `f32` here.
//! The per-scale kernels honour `DecodeDtype::F16` for the dequant /
//! softmax / sigmoid stages; widening at the bridge only affects NMS and
//! mask materialisation, where the existing float path is f32-native.
//! Phase 2 may revisit if the bandwidth savings warrant a native-f16 NMS
//! / mask kernel.

use ndarray::{ArrayView2, ArrayView3};

use crate::per_scale::outputs::{BufferRef, DecodedOutputsRef, ProtosView};
use crate::yolo::{
    extract_proto_data_float, impl_yolo_segdet_get_boxes, impl_yolo_split_segdet_process_masks,
};
use crate::{DecoderError, DetectBox, Nms, ProtoData, Segmentation, XYWH};

/// Materialise the per-scale buffers as f32 contiguous `Vec`s and views
/// suitable for the legacy NMS / mask kernels. Boxes are emitted in
/// `xc, yc, w, h` (XYWH centre-point) order — see
/// `per_scale::kernels::box_primitives::dist2bbox_anchor_f32`.
struct WidenedF32 {
    boxes: Vec<f32>,
    scores: Vec<f32>,
    mask_coefs: Option<Vec<f32>>,
    protos: Option<ndarray::Array3<f32>>,
    n: usize,
    nc: usize,
    nm: usize,
}

fn widen_to_f32(decoded: &DecodedOutputsRef<'_>) -> WidenedF32 {
    let n = decoded.total_anchors;
    let nc = decoded.num_classes;
    let nm = decoded.num_mask_coefs;

    let boxes: Vec<f32> = match &decoded.boxes {
        BufferRef::F32(s) => s.to_vec(),
        BufferRef::F16(s) => s.iter().map(|v| v.to_f32()).collect(),
    };
    let scores: Vec<f32> = match &decoded.scores {
        BufferRef::F32(s) => s.to_vec(),
        BufferRef::F16(s) => s.iter().map(|v| v.to_f32()).collect(),
    };
    let mask_coefs = decoded.mask_coefs.as_ref().map(|b| match b {
        BufferRef::F32(s) => s.to_vec(),
        BufferRef::F16(s) => s.iter().map(|v| v.to_f32()).collect(),
    });
    // Protos are owned `Array4<F>` shape `[1, H, W, NM]` (NHWC). The
    // legacy yolo path expects `ArrayView3<PROTO>` of shape `(H, W,
    // NM)`, so we drop the leading batch axis here.
    let protos = decoded.protos.as_ref().map(|p| match p {
        ProtosView::F32(a) => a.index_axis(ndarray::Axis(0), 0).to_owned(),
        ProtosView::F16(a) => a.index_axis(ndarray::Axis(0), 0).mapv(|v| v.to_f32()),
    });

    WidenedF32 {
        boxes,
        scores,
        mask_coefs,
        protos,
        n,
        nc,
        nm,
    }
}

/// Bridge: per-scale outputs → optional `ProtoData`, with `output_boxes`
/// populated post-NMS. Mirrors the contract of `decode_proto`.
pub(super) fn per_scale_to_proto_data(
    decoded: &DecodedOutputsRef<'_>,
    output_boxes: &mut Vec<DetectBox>,
    iou_threshold: f32,
    score_threshold: f32,
    nms_mode: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
) -> Result<Option<ProtoData>, DecoderError> {
    let WidenedF32 {
        boxes,
        scores,
        mask_coefs,
        protos,
        n,
        nc,
        nm,
    } = widen_to_f32(decoded);

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), &boxes)
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), &scores)
        .map_err(|e| DecoderError::Internal(format!("per_scale scores view: {e}")))?;

    output_boxes.clear();

    let det_indices = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
        boxes_view,
        scores_view,
        score_threshold,
        iou_threshold,
        nms_mode,
        pre_nms_top_k,
        max_det,
    );

    match (mask_coefs.as_ref(), protos.as_ref()) {
        (Some(mc), Some(pr)) => {
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc)
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
            let pr_view: ArrayView3<f32> = pr.view();
            let proto_data = extract_proto_data_float(det_indices, mc_view, pr_view, output_boxes);
            Ok(Some(proto_data))
        }
        _ => {
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
pub(super) fn per_scale_to_masks(
    decoded: &DecodedOutputsRef<'_>,
    output_boxes: &mut Vec<DetectBox>,
    output_masks: &mut Vec<Segmentation>,
    iou_threshold: f32,
    score_threshold: f32,
    nms_mode: Option<Nms>,
    pre_nms_top_k: usize,
    max_det: usize,
) -> Result<(), DecoderError> {
    let WidenedF32 {
        boxes,
        scores,
        mask_coefs,
        protos,
        n,
        nc,
        nm,
    } = widen_to_f32(decoded);

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), &boxes)
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), &scores)
        .map_err(|e| DecoderError::Internal(format!("per_scale scores view: {e}")))?;

    output_boxes.clear();
    output_masks.clear();

    let det_indices = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
        boxes_view,
        scores_view,
        score_threshold,
        iou_threshold,
        nms_mode,
        pre_nms_top_k,
        max_det,
    );

    match (mask_coefs.as_ref(), protos.as_ref()) {
        (Some(mc), Some(pr)) => {
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc)
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
            let pr_view: ArrayView3<f32> = pr.view();
            impl_yolo_split_segdet_process_masks(
                det_indices,
                mc_view,
                pr_view,
                output_boxes,
                output_masks,
            )
        }
        _ => {
            // Detection-only: emit boxes; no masks.
            for (db, _) in det_indices {
                output_boxes.push(db);
            }
            Ok(())
        }
    }
}
