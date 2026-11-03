// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between the per-scale fast path's `DecodedOutputsRef` and the
//! existing post-NMS pipeline (score filter → top-K → NMS → optional mask
//! materialisation).
//!
//! The downstream NMS and mask kernels are f32-native, so this layer is
//! responsible for widening `DecodeDtype::F16` outputs to f32:
//!
//! - **`DecodeDtype::F32`** (default): zero-copy. The widened views
//!   borrow directly from the per-scale buffers.
//! - **`DecodeDtype::F16`**: widen into the decoder's
//!   [`WidenScratch`], whose storage is reused frame to frame, on NEON
//!   where the plan's CPU features allow it.

use ndarray::{ArrayView2, ArrayView3, Axis};
use tracing::trace_span;

use crate::per_scale::kernels::CpuFeatures;
use crate::per_scale::outputs::{
    staging, widen_f16_into, BufferRef, DecodedOutputsRef, ProtosView, WidenScratch,
};
use crate::yolo::{
    extract_proto_data_float, impl_yolo_segdet_get_boxes, impl_yolo_split_segdet_process_masks,
};
use crate::{DecoderError, DetectBox, Nms, ProtoData, Segmentation, XYWH};

/// f32-typed views of the per-scale outputs ready for the legacy NMS/mask
/// kernels, borrowed from the per-scale buffers (f32 outputs) or from the
/// widen scratch (f16 outputs). Boxes are in `(xc, yc, w, h)` order — see
/// `edgefirst_kernels::prim::box_primitives::dist2bbox_anchor_f32`.
struct WidenedF32<'a> {
    boxes: &'a [f32],
    scores: &'a [f32],
    mask_coefs: Option<&'a [f32]>,
    protos: Option<ArrayView3<'a, f32>>,
    n: usize,
    nc: usize,
    nm: usize,
}

/// `buf` as f32: borrowed, or widened into `store`.
fn widen_buffer<'a>(f: &CpuFeatures, buf: &BufferRef<'a>, store: &'a mut Vec<f32>) -> &'a [f32] {
    match buf {
        BufferRef::F32(s) => s,
        BufferRef::F16(s) => {
            let out = staging(store, s.len());
            widen_f16_into(f, s, out);
            out
        }
    }
}

fn widen_to_f32<'a>(
    decoded: &'a DecodedOutputsRef<'a>,
    scratch: &'a mut WidenScratch,
) -> Result<WidenedF32<'a>, DecoderError> {
    // Trace stage so the f32 zero-copy vs f16 widen split is visible to
    // callers profiling decode latency. The `kind` attribute also makes
    // it easy to spot accidental f16 selection on f32-only hardware.
    let kind = match &decoded.boxes {
        BufferRef::F32(_) => "f32_borrow",
        BufferRef::F16(_) => "f16_widen",
    };
    let _span = trace_span!("decoder.per_scale_run.widen_f32", kind = kind).entered();

    let WidenScratch {
        features,
        boxes: boxes_store,
        scores: scores_store,
        mask_coefs: mc_store,
        protos: protos_store,
    } = scratch;
    let f = *features;
    let boxes = widen_buffer(&f, &decoded.boxes, boxes_store);
    let scores = widen_buffer(&f, &decoded.scores, scores_store);
    let mask_coefs = decoded
        .mask_coefs
        .as_ref()
        .map(|b| widen_buffer(&f, b, mc_store));
    // Protos are `Array4<F>` shape `[1, H, W, NM]` (NHWC). The legacy
    // yolo path expects `ArrayView3<PROTO>` of shape `(H, W, NM)`, so
    // drop the leading batch axis.
    let protos = match decoded.protos.as_ref() {
        None => None,
        Some(ProtosView::F32(a)) => Some(a.index_axis(Axis(0), 0)),
        Some(ProtosView::F16(a)) => {
            let v = a.index_axis(Axis(0), 0);
            let out = staging(protos_store, v.len());
            match v.as_slice() {
                Some(s) => widen_f16_into(&f, s, out),
                None => {
                    for (o, x) in out.iter_mut().zip(v.iter()) {
                        *o = x.to_f32();
                    }
                }
            }
            let view = ArrayView3::from_shape(v.raw_dim(), &*out)
                .map_err(|e| DecoderError::Internal(format!("per_scale protos view: {e}")))?;
            Some(view)
        }
    };

    Ok(WidenedF32 {
        boxes,
        scores,
        mask_coefs,
        protos,
        n: decoded.total_anchors,
        nc: decoded.num_classes,
        nm: decoded.num_mask_coefs,
    })
}

/// Bridge: per-scale outputs → optional `ProtoData`, with `output_boxes`
/// populated post-NMS. Mirrors the contract of `decode_proto`.
#[allow(clippy::too_many_arguments)]
pub(super) fn per_scale_to_proto_data<'a>(
    decoded: &'a DecodedOutputsRef<'a>,
    widen: &'a mut WidenScratch,
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
    let widened = widen_to_f32(decoded, widen)?;
    let n = widened.n;
    let nc = widened.nc;
    let nm = widened.nm;

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), widened.boxes)
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), widened.scores)
        .map_err(|e| DecoderError::Internal(format!("per_scale scores view: {e}")))?;

    output_boxes.clear();

    let mut det_indices = {
        let _s = trace_span!("decoder.decode_proto.per_scale_to_proto_data.get_boxes").entered();
        impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_view,
            scores_view,
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

    match (widened.mask_coefs, widened.protos) {
        (Some(mc), Some(pr_view)) => {
            let _s = trace_span!("decoder.decode_proto.per_scale_to_proto_data.extract").entered();
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc)
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
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
pub(super) fn per_scale_to_masks<'a>(
    decoded: &'a DecodedOutputsRef<'a>,
    widen: &'a mut WidenScratch,
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
    let widened = widen_to_f32(decoded, widen)?;
    let n = widened.n;
    let nc = widened.nc;
    let nm = widened.nm;

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), widened.boxes)
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), widened.scores)
        .map_err(|e| DecoderError::Internal(format!("per_scale scores view: {e}")))?;

    output_boxes.clear();
    output_masks.clear();

    let mut det_indices = {
        let _s = trace_span!("decoder.decode.per_scale_to_masks.get_boxes").entered();
        impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_view,
            scores_view,
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
    // and the returned bboxes match the contract of `Decoder::decode`
    // (EDGEAI-1303 + Copilot review feedback on PR #63).
    crate::yolo::maybe_normalize_boxes_in_place(&mut det_indices, normalized, input_dims);

    match (widened.mask_coefs, widened.protos) {
        (Some(mc), Some(pr_view)) => {
            let _s = trace_span!("decoder.decode.per_scale_to_masks.process_masks").entered();
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc)
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
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
