// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bridge between the per-scale fast path's `DecodedOutputsRef` and the
//! existing post-NMS pipeline (score filter → top-K → NMS → optional mask
//! materialisation).
//!
//! The downstream NMS and mask kernels are f32-native, so this layer is
//! responsible for widening `DecodeDtype::F16` outputs to f32. The widen
//! is hybrid:
//!
//! - **`DecodeDtype::F32`** (default): zero-copy. The widened views
//!   borrow directly from the per-scale buffers — no allocation, no
//!   memcpy. For a yolov8-seg model that's ~7 MB/decode of allocation
//!   we used to do unconditionally.
//! - **`DecodeDtype::F16`**: allocate-and-convert. There is no f32
//!   storage to borrow; we materialise into a fresh `Vec<f32>` (or
//!   `Array3<f32>`) for each decode. A future revision can hoist these
//!   buffers into the [`crate::per_scale::PerScaleDecoder`] for reuse
//!   across calls.

use std::borrow::Cow;

use ndarray::{Array3, ArrayView2, ArrayView3, Axis};
use tracing::trace_span;

use crate::per_scale::outputs::{BufferRef, DecodedOutputsRef, ProtosView};
use crate::yolo::{
    extract_proto_data_float, impl_yolo_segdet_get_boxes, impl_yolo_split_segdet_process_masks,
};
use crate::{DecoderError, DetectBox, Nms, ProtoData, Segmentation, XYWH};

/// f32 protos: borrowed view (zero-copy) or owned array (f16→f32 widen).
enum ProtosF32<'a> {
    Borrowed(ArrayView3<'a, f32>),
    Owned(Array3<f32>),
}

impl ProtosF32<'_> {
    fn view(&self) -> ArrayView3<'_, f32> {
        match self {
            Self::Borrowed(v) => v.reborrow(),
            Self::Owned(a) => a.view(),
        }
    }
}

/// f32-typed views of the per-scale outputs ready for the legacy NMS/mask
/// kernels. Each field is either a zero-copy borrow (the per-scale path
/// already produced f32) or a freshly-widened owned buffer (the per-scale
/// path produced f16). Boxes are in `(xc, yc, w, h)` order — see
/// `per_scale::kernels::box_primitives::dist2bbox_anchor_f32`.
struct WidenedF32<'a> {
    boxes: Cow<'a, [f32]>,
    scores: Cow<'a, [f32]>,
    mask_coefs: Option<Cow<'a, [f32]>>,
    protos: Option<ProtosF32<'a>>,
    n: usize,
    nc: usize,
    nm: usize,
}

fn widen_to_f32<'a>(decoded: &'a DecodedOutputsRef<'a>) -> WidenedF32<'a> {
    // Trace stage so the f32 zero-copy vs f16 widen split is visible to
    // callers profiling decode latency. The `kind` attribute also makes
    // it easy to spot accidental f16 selection on f32-only hardware.
    let kind = match &decoded.boxes {
        BufferRef::F32(_) => "f32_borrow",
        BufferRef::F16(_) => "f16_widen",
    };
    let _span = trace_span!("decoder.per_scale_run.widen_f32", kind = kind).entered();

    let n = decoded.total_anchors;
    let nc = decoded.num_classes;
    let nm = decoded.num_mask_coefs;

    let boxes: Cow<'a, [f32]> = match &decoded.boxes {
        BufferRef::F32(s) => Cow::Borrowed(*s),
        BufferRef::F16(s) => Cow::Owned(s.iter().map(|v| v.to_f32()).collect()),
    };
    let scores: Cow<'a, [f32]> = match &decoded.scores {
        BufferRef::F32(s) => Cow::Borrowed(*s),
        BufferRef::F16(s) => Cow::Owned(s.iter().map(|v| v.to_f32()).collect()),
    };
    let mask_coefs = decoded.mask_coefs.as_ref().map(|b| match b {
        BufferRef::F32(s) => Cow::Borrowed(*s),
        BufferRef::F16(s) => Cow::Owned(s.iter().map(|v| v.to_f32()).collect()),
    });
    // Protos are `Array4<F>` shape `[1, H, W, NM]` (NHWC). The legacy
    // yolo path expects `ArrayView3<PROTO>` of shape `(H, W, NM)`, so
    // drop the leading batch axis.
    let protos = decoded.protos.as_ref().map(|p| match p {
        ProtosView::F32(a) => ProtosF32::Borrowed(a.index_axis(Axis(0), 0)),
        ProtosView::F16(a) => ProtosF32::Owned(a.index_axis(Axis(0), 0).mapv(|v| v.to_f32())),
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
    let widened = widen_to_f32(decoded);
    let n = widened.n;
    let nc = widened.nc;
    let nm = widened.nm;

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), widened.boxes.as_ref())
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), widened.scores.as_ref())
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
        )
    };
    // Per-scale `dist2bbox_anchor_*` emits pixel-space coords by design.
    // Apply EDGEAI-1303 normalization so output bboxes land in the
    // canonical [0, 1] range expected by Decoder callers.
    crate::yolo::maybe_normalize_boxes_in_place(&mut det_indices, normalized, input_dims);

    match (widened.mask_coefs.as_ref(), widened.protos.as_ref()) {
        (Some(mc), Some(pr)) => {
            let _s = trace_span!("decoder.decode_proto.per_scale_to_proto_data.extract").entered();
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc.as_ref())
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
            let pr_view = pr.view();
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
    let widened = widen_to_f32(decoded);
    let n = widened.n;
    let nc = widened.nc;
    let nm = widened.nm;

    let boxes_view = ArrayView2::<f32>::from_shape((n, 4), widened.boxes.as_ref())
        .map_err(|e| DecoderError::Internal(format!("per_scale boxes view: {e}")))?;
    let scores_view = ArrayView2::<f32>::from_shape((n, nc), widened.scores.as_ref())
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
        )
    };
    // Per-scale `dist2bbox_anchor_*` emits pixel-space coords by design.
    // Normalize before mask processing so `protobox` sees [0, 1] coords
    // and the returned bboxes match the contract of `Decoder::decode`
    // (EDGEAI-1303 + Copilot review feedback on PR #63).
    crate::yolo::maybe_normalize_boxes_in_place(&mut det_indices, normalized, input_dims);

    match (widened.mask_coefs.as_ref(), widened.protos.as_ref()) {
        (Some(mc), Some(pr)) => {
            let _s = trace_span!("decoder.decode.per_scale_to_masks.process_masks").entered();
            let mc_view = ArrayView2::<f32>::from_shape((n, nm), mc.as_ref())
                .map_err(|e| DecoderError::Internal(format!("per_scale mc view: {e}")))?;
            let pr_view = pr.view();
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
