// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-frame tensor binding (shape match) and live quantization
//! reading from `TensorDyn`.

use crate::per_scale::plan::PerScalePlan;
use crate::{DecoderError, DecoderResult, Quantization};
use edgefirst_tensor::{DType, TensorDyn};

#[derive(Debug)]
#[allow(dead_code)] // Wired by Task 21.
pub(crate) struct LevelBindings<'a> {
    pub(crate) boxes: &'a TensorDyn,
    pub(crate) scores: &'a TensorDyn,
    pub(crate) mask_coefs: Option<&'a TensorDyn>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct FrameBindings<'a> {
    pub(crate) levels: Vec<LevelBindings<'a>>,
    pub(crate) protos: Option<&'a TensorDyn>,
}

/// Read quantization from the bound tensor for an integer dtype, or
/// return `Quantization::identity()` for float dtypes.
///
/// This is the **single source of truth** for runtime quantization in
/// the per-scale subsystem: tensors carry their own quantization
/// metadata via `Tensor::quantization()`, and the decoder reads it
/// live each frame. The schema declares the intended quant; the
/// upstream inference layer (or `apply_schema_quant` helper) is
/// responsible for attaching it before calling the decoder.
#[allow(dead_code)] // Wired by Task 21.
pub(crate) fn quant_from_tensor(
    t: &TensorDyn,
    role: &'static str,
    level: usize,
) -> DecoderResult<Quantization> {
    fn convert_quant(q: &edgefirst_tensor::Quantization) -> Quantization {
        // Per-tensor quant: scale is length-1 and zero_point is `Some([zp])`
        // (asymmetric) or `None` (symmetric). Per-channel quant (rejected
        // upstream by the plan) would have longer slices; this conversion
        // would silently use [0] for it, but the planner doesn't allow
        // per-channel for per-scale, so we only see per-tensor here.
        Quantization {
            scale: q.scale().first().copied().unwrap_or(1.0),
            zero_point: q.zero_point().and_then(|z| z.first().copied()).unwrap_or(0),
        }
    }
    macro_rules! quant {
        ($ty:ty, $dtype:ident) => {
            t.as_typed::<$ty>()
                .expect("dtype checked")
                .quantization()
                .map(convert_quant)
                .ok_or(DecoderError::QuantMissing {
                    dtype: DType::$dtype,
                    role,
                    level,
                })
        };
    }
    match t.dtype() {
        DType::I8 => quant!(i8, I8),
        DType::U8 => quant!(u8, U8),
        DType::I16 => quant!(i16, I16),
        DType::U16 => quant!(u16, U16),
        DType::F16 | DType::F32 => Ok(Quantization::identity()),
        other => Err(DecoderError::DtypeMismatch {
            expected: DType::I8,
            actual: other,
            role,
            level,
        }),
    }
}

/// Read per-channel quantization from the bound box tensor for DFL decode.
///
/// Returns `Ok(None)` when the tensor carries per-tensor quant (caller falls
/// back to the per-tensor scalar kernel). Returns `Ok(Some((scales, zps)))`
/// — borrowed from the tensor's quantization, no allocation — when per-channel
/// quant is present; an **empty `zps`** slice means "all zero-points are 0"
/// (the per-channel kernels treat a missing entry as 0). Returns
/// `Err(QuantMissing)` when no quantization is attached to an integer tensor.
///
/// Borrowing (rather than `to_vec`) keeps this allocation-free on the per-frame
/// decode hot path, where it runs once per FPN level per frame.
///
/// Float tensors (`F16`, `F32`) never carry per-channel quant from NPUs and
/// always return `Ok(None)` so the per-tensor (identity) fast path is used.
pub(crate) fn box_per_channel_quant_from_tensor(
    t: &TensorDyn,
    level: usize,
) -> DecoderResult<Option<(&[f32], &[i32])>> {
    /// Borrow per-channel `(scales, zps)` from a `Quantization` if it is
    /// per-channel, otherwise return `None` (caller uses per-tensor path). An
    /// empty `zps` is the "all zero-points = 0" convention (symmetric quant).
    fn extract(q: &edgefirst_tensor::Quantization) -> Option<(&[f32], &[i32])> {
        if !q.is_per_channel() {
            return None;
        }
        Some((q.scale(), q.zero_point().unwrap_or(&[])))
    }
    macro_rules! quant {
        ($ty:ty, $dtype:ident) => {
            match t.as_typed::<$ty>().expect("dtype checked").quantization() {
                Some(q) => Ok(extract(q)),
                None => Err(DecoderError::QuantMissing {
                    dtype: DType::$dtype,
                    role: "boxes",
                    level,
                }),
            }
        };
    }
    match t.dtype() {
        DType::I8 => quant!(i8, I8),
        DType::U8 => quant!(u8, U8),
        DType::I16 => quant!(i16, I16),
        DType::U16 => quant!(u16, U16),
        // Float tensors: no per-channel quant from NPUs — use per-tensor (identity).
        DType::F16 | DType::F32 => Ok(None),
        other => Err(DecoderError::DtypeMismatch {
            expected: DType::I8,
            actual: other,
            role: "boxes",
            level,
        }),
    }
}

/// Find the first un-consumed input tensor matching `expected` shape.
/// Marks the matched tensor's index in `used` so subsequent calls skip it.
#[allow(dead_code)]
pub(crate) fn bind_one<'a>(
    inputs: &'a [&'a TensorDyn],
    used: &mut [bool],
    expected: &[usize],
    role: &'static str,
    level: usize,
) -> DecoderResult<&'a TensorDyn> {
    for (i, t) in inputs.iter().enumerate() {
        if !used[i] && t.shape() == expected {
            used[i] = true;
            return Ok(*t);
        }
    }
    Err(DecoderError::InvalidShape(format!(
        "per-scale {role} (level {level}): no remaining tensor matches {expected:?}"
    )))
}

/// Resolve all per-frame bindings for a given plan.
#[allow(dead_code)] // Wired by Task 21.
pub(crate) fn resolve_bindings<'a>(
    plan: &PerScalePlan,
    inputs: &'a [&'a TensorDyn],
) -> DecoderResult<FrameBindings<'a>> {
    let mut used = vec![false; inputs.len()];
    let mut levels = Vec::with_capacity(plan.levels.len());
    for (li, lvl) in plan.levels.iter().enumerate() {
        let boxes = bind_one(inputs, &mut used, &lvl.box_shape, "boxes", li)?;
        let scores = bind_one(inputs, &mut used, &lvl.score_shape, "scores", li)?;
        let mask_coefs = match &lvl.mc_shape {
            Some(s) => Some(bind_one(inputs, &mut used, s, "mask_coefs", li)?),
            None => None,
        };
        levels.push(LevelBindings {
            boxes,
            scores,
            mask_coefs,
        });
    }
    let protos = if let Some(s) = &plan.proto_shape {
        Some(bind_one(inputs, &mut used, s, "protos", 0)?)
    } else {
        None
    };
    Ok(FrameBindings { levels, protos })
}

use crate::per_scale::kernels::dfl_lut::DflExpTable;
use crate::per_scale::kernels::dispatch::{DstSliceMut, InputView};
use crate::per_scale::kernels::level_box::{
    decode_box_level_dfl_i16_to_f16_per_channel, decode_box_level_dfl_i16_to_f32_per_channel,
    decode_box_level_dfl_i8_to_f16_per_channel, decode_box_level_dfl_i8_to_f32_per_channel,
    decode_box_level_dfl_u16_to_f16_per_channel, decode_box_level_dfl_u16_to_f32_per_channel,
    decode_box_level_dfl_u8_to_f16_per_channel, decode_box_level_dfl_u8_to_f32_per_channel,
};
use crate::per_scale::kernels::transpose::nchw_to_nhwc;
use crate::per_scale::outputs::{
    boxes_level_slice_of, mask_coefs_level_slice_of, scores_level_slice_of, Buffer, BufferRef,
    DecodedOutputBuffers, DecodedOutputsRef, ProtoStorage, ProtosView,
};
use crate::per_scale::plan::Layout;
use crate::per_scale::prefilter::{
    gather_rows, raw_score_floor, select_anchors, LevelScratch, PACK_ANCHORS,
};
use edgefirst_tensor::{CpuAccess, TensorMapTrait, TensorTrait};

/// Run the per-scale decode pipeline for one frame's inputs. Writes
/// into `buffers` and returns a borrowed view.
///
/// With a `score_threshold` the prefilter decodes only the anchors that
/// can reach it (see [`crate::per_scale::prefilter`]); frames it cannot
/// bound, such as float score tensors, are decoded densely.
#[allow(dead_code)] // Wired by Task 24.
pub(crate) fn run<'a>(
    plan: &PerScalePlan,
    buffers: &'a mut DecodedOutputBuffers,
    inputs: &[&TensorDyn],
    score_threshold: Option<f32>,
) -> DecoderResult<DecodedOutputsRef<'a>> {
    // Top-level span. The `n_levels` field gives at-a-glance shape in
    // Perfetto's flame graph; per-level / per-role child spans below
    // give the per-stage decomposition we need to find hotspots.
    let _outer = tracing::trace_span!(
        "decoder.per_scale_run",
        n_levels = plan.levels.len(),
        encoding = ?plan.box_encoding,
        nc = plan.num_classes,
        nm = plan.num_mask_coefs,
    )
    .entered();
    let bind = {
        let _s = tracing::trace_span!("decoder.per_scale_run.resolve_bindings").entered();
        resolve_bindings(plan, inputs)?
    };

    // Anchors are independent: each writes disjoint ranges of `boxes` /
    // `scores` / `mask_coefs`. The dense NHWC decode (the common TFLite
    // case) splits the levels into pieces for the rayon pool. NCHW levels
    // share `layout_scratch`, which would need per-level scratch buffers
    // to parallelize safely, so they decode sequentially.
    let mut tables = std::mem::take(&mut buffers.dfl_tables);
    let levels = update_dfl_tables(plan, &bind, &mut tables).and_then(|()| {
        let kept = match score_threshold {
            Some(t) => run_levels_prefiltered(plan, &bind, buffers, &tables, t)?,
            None => None,
        };
        Ok(match kept {
            Some(n) => n,
            None => {
                let all_nhwc = plan.levels.iter().all(|l| l.layout == Layout::Nhwc);
                if all_nhwc {
                    run_levels_parallel(plan, &bind, buffers, &tables)?;
                } else {
                    run_levels_sequential(plan, &bind, buffers, &tables)?;
                }
                plan.total_anchors
            }
        })
    });
    buffers.dfl_tables = tables;
    let total_anchors = levels?;

    // Protos — single tensor, no per-level structure. Always sequential.
    // When the schema declares NCHW protos, we transpose into the NHWC
    // scratch before dequant, matching the per-level NCHW strategy.
    if let (Some(proto_input), Some(proto_dispatch)) = (bind.protos, plan.proto_dispatch) {
        let _s = tracing::trace_span!("decoder.per_scale_run.protos").entered();
        let q = quant_from_tensor(proto_input, "protos", 0)?;
        let proto_layout = plan.proto_layout.unwrap_or(Layout::Nhwc);
        // Destructure to get disjoint borrows on `protos` and
        // `layout_scratch`, same pattern used by run_levels_sequential.
        let DecodedOutputBuffers {
            protos: ref mut proto_buf,
            layout_scratch,
            ..
        } = buffers;
        let dst = match proto_buf.as_mut() {
            Some(ProtoStorage::F32(a)) => a.as_slice_mut().map(DstSliceMut::F32),
            Some(ProtoStorage::F16(a)) => a.as_slice_mut().map(DstSliceMut::F16),
            None => None,
        }
        .ok_or_else(|| {
            DecoderError::Internal("protos buffer absent but plan declared proto_dispatch".into())
        })?;
        if proto_layout == Layout::Nchw {
            let nhwc = plan.proto_nhwc_shape.as_ref().ok_or_else(|| {
                DecoderError::Internal("NCHW protos but proto_nhwc_shape is None".into())
            })?;
            let (h, w, c) = (nhwc[1], nhwc[2], nhwc[3]);
            with_mapped_or_transposed_input(
                proto_input,
                "protos",
                0,
                proto_layout,
                h,
                w,
                c,
                layout_scratch,
                |input| proto_dispatch.run(input, q, dst),
            )?;
        } else {
            with_mapped_input(proto_input, "protos", 0, |input| {
                proto_dispatch.run(input, q, dst)
            })?;
        }
    }

    Ok(make_outputs_ref(buffers, plan, total_anchors))
}

/// Point each level's DFL exp table at its box tensor's scale, when the box
/// kernel weighs bins through one. A table is rebuilt only when the scale
/// changes, so a steady stream of frames builds each one once.
fn update_dfl_tables(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    tables: &mut Vec<DflExpTable>,
) -> DecoderResult<()> {
    if !plan.box_dispatch.uses_exp_table() {
        return Ok(());
    }
    for (li, lvl_bind) in bind.levels.iter().enumerate() {
        let scale = quant_from_tensor(lvl_bind.boxes, "boxes", li)?.scale;
        match tables.get_mut(li) {
            Some(t) => t.update(scale),
            None => tables.push(DflExpTable::new(scale)),
        }
    }
    Ok(())
}

/// Dispatch the per-channel DFL kernel for the given `(input, dst)` dtype pair.
/// Does NOT use `BoxLevelDispatch` (which carries only per-tensor `Quantization`).
/// Only integer input types are matched — float inputs fall through to a
/// `KernelDispatchUnreachable` error because NPUs never emit per-channel quant
/// on float outputs.
fn dispatch_dfl_per_channel(
    input: InputView<'_>,
    scales: &[f32],
    zps: &[i32],
    level: &crate::per_scale::plan::LevelPlan,
    dst: DstSliceMut<'_>,
) -> DecoderResult<()> {
    match (input, dst) {
        (InputView::I8(i), DstSliceMut::F32(d)) => {
            decode_box_level_dfl_i8_to_f32_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::U8(i), DstSliceMut::F32(d)) => {
            decode_box_level_dfl_u8_to_f32_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::I16(i), DstSliceMut::F32(d)) => {
            decode_box_level_dfl_i16_to_f32_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::U16(i), DstSliceMut::F32(d)) => {
            decode_box_level_dfl_u16_to_f32_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::I8(i), DstSliceMut::F16(d)) => {
            decode_box_level_dfl_i8_to_f16_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::U8(i), DstSliceMut::F16(d)) => {
            decode_box_level_dfl_u8_to_f16_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::I16(i), DstSliceMut::F16(d)) => {
            decode_box_level_dfl_i16_to_f16_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        (InputView::U16(i), DstSliceMut::F16(d)) => {
            decode_box_level_dfl_u16_to_f16_per_channel(i, scales, zps, level, d);
            Ok(())
        }
        _ => Err(DecoderError::KernelDispatchUnreachable(
            "dispatch_dfl_per_channel: float inputs do not support per-channel quant".into(),
        )),
    }
}

/// Run the box kernel for one level, choosing the per-channel DFL kernel
/// when the box tensor carries per-channel quantization.
fn run_box_kernel(
    plan: &PerScalePlan,
    level: &crate::per_scale::plan::LevelPlan,
    boxes: &TensorDyn,
    li: usize,
    table: Option<&DflExpTable>,
    input: InputView<'_>,
    dst: DstSliceMut<'_>,
) -> DecoderResult<()> {
    if plan.box_encoding == crate::schema::BoxEncoding::Dfl {
        if let Some((scales, zps)) = box_per_channel_quant_from_tensor(boxes, li)? {
            return dispatch_dfl_per_channel(input, scales, zps, level, dst);
        }
    }
    let q = quant_from_tensor(boxes, "boxes", li)?;
    plan.box_dispatch.run(input, q, level, table, dst)
}

/// Prefiltered per-level loop: select the anchors whose raw scores can
/// reach `threshold`, gather their rows, and decode only those, packing
/// each level's survivors after the previous level's in the output
/// buffers. A level's tail (the anchors after its last whole
/// [`crate::per_scale::prefilter::PACK_ANCHORS`] group) is decoded whole after its packed survivors.
/// Returns the number of anchors decoded, or `None` (with the outputs
/// untouched) when the threshold cannot be mapped onto every level's raw
/// scores and the caller must decode densely.
///
/// Runs on the calling thread: the kept set is a few percent of the
/// anchors, too little work to repay a rayon scope.
fn run_levels_prefiltered(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    buffers: &mut DecodedOutputBuffers,
    tables: &[DflExpTable],
    threshold: f32,
) -> DecoderResult<Option<usize>> {
    if buffers.prefilter.len() != plan.levels.len() {
        buffers.prefilter = plan.levels.iter().map(LevelScratch::for_level).collect();
    }
    let mut floors = Vec::with_capacity(bind.levels.len());
    for (li, lvl_bind) in bind.levels.iter().enumerate() {
        let dtype = lvl_bind.scores.dtype();
        if !matches!(dtype, DType::I8 | DType::U8 | DType::I16 | DType::U16) {
            return Ok(None);
        }
        let q = quant_from_tensor(lvl_bind.scores, "scores", li)?;
        let floor = raw_score_floor(
            &mut buffers.prefilter[li],
            &plan.score_dispatch,
            dtype,
            plan.out_dtype,
            q,
            plan.score_activation,
            threshold,
        )?;
        match floor {
            Some(floor) => floors.push((q, floor)),
            None => return Ok(None),
        }
    }
    // A level with no survivors never maps its boxes or mask coefficients,
    // so check their declared access up front, as the dense decode's maps do.
    for lvl_bind in &bind.levels {
        lvl_bind.boxes.cpu_access().require(CpuAccess::Read)?;
        if let Some(mc) = lvl_bind.mask_coefs {
            mc.cpu_access().require(CpuAccess::Read)?;
        }
    }

    let _span = tracing::trace_span!(
        "decoder.per_scale_run.prefilter",
        threshold,
        kept = tracing::field::Empty,
    )
    .entered();
    let DecodedOutputBuffers {
        boxes,
        scores,
        mask_coefs,
        layout_scratch,
        prefilter,
        ..
    } = buffers;

    let nc = plan.num_classes;
    let nm = plan.num_mask_coefs;
    let mut offset = 0;
    for (li, lvl_bind) in bind.levels.iter().enumerate() {
        let lvl = &plan.levels[li];
        let scratch = &mut prefilter[li];
        let (score_q, floor) = floors[li];
        let body = scratch.body;
        let tail = scratch.tail.w;
        let _level_span = tracing::trace_span!(
            "decoder.per_scale_run.level",
            li = li,
            stride = lvl.stride,
            anchors = lvl.h * lvl.w,
            kept = tracing::field::Empty,
        )
        .entered();

        with_mapped_or_transposed_input(
            lvl_bind.scores,
            "scores",
            li,
            lvl.layout,
            lvl.h,
            lvl.w,
            nc,
            layout_scratch,
            |input| {
                select_anchors(&input.slice(0..body * nc), nc, floor, &mut scratch.anchors)?;
                let kept = scratch.pack(lvl);
                if kept > 0 {
                    let packed = gather_rows(&input, nc, &scratch.anchors, &mut scratch.scores);
                    let dst = scores_level_slice_of(scores, offset, scratch.anchors.len(), nc);
                    plan.score_dispatch.run(
                        packed,
                        score_q,
                        nc,
                        &scratch.plan,
                        plan.score_activation,
                        dst,
                    )?;
                }
                if tail > 0 {
                    let dst = scores_level_slice_of(scores, offset + kept, tail, nc);
                    plan.score_dispatch.run(
                        input.slice(body * nc..(body + tail) * nc),
                        score_q,
                        nc,
                        &scratch.tail,
                        plan.score_activation,
                        dst,
                    )?;
                }
                Ok(())
            },
        )?;
        let kept = scratch.kept;
        let k = kept + tail;
        _level_span.record("kept", k);
        if k == 0 {
            continue;
        }
        let packed_len = scratch.anchors.len();

        let box_channels = box_channel_count_for_level(lvl);
        with_mapped_or_transposed_input(
            lvl_bind.boxes,
            "boxes",
            li,
            lvl.layout,
            lvl.h,
            lvl.w,
            box_channels,
            layout_scratch,
            |input| {
                if kept > 0 {
                    let packed =
                        gather_rows(&input, box_channels, &scratch.anchors, &mut scratch.boxes);
                    let dst = boxes_level_slice_of(boxes, offset, packed_len);
                    let table = tables.get(li);
                    run_box_kernel(plan, &scratch.plan, lvl_bind.boxes, li, table, packed, dst)?;
                }
                if tail > 0 {
                    let rows = input.slice(body * box_channels..(body + tail) * box_channels);
                    let dst = boxes_level_slice_of(boxes, offset + kept, tail);
                    let table = tables.get(li);
                    run_box_kernel(plan, &scratch.tail, lvl_bind.boxes, li, table, rows, dst)?;
                }
                Ok(())
            },
        )?;

        if let (Some(mc_input), Some(mc_dispatch), Some(mc_buf)) =
            (lvl_bind.mask_coefs, plan.mc_dispatch, mask_coefs.as_mut())
        {
            if nm > 0 {
                let q = quant_from_tensor(mc_input, "mask_coefs", li)?;
                with_mapped_or_transposed_input(
                    mc_input,
                    "mask_coefs",
                    li,
                    lvl.layout,
                    lvl.h,
                    lvl.w,
                    nm,
                    layout_scratch,
                    |input| {
                        if kept > 0 {
                            let packed =
                                gather_rows(&input, nm, &scratch.anchors, &mut scratch.mask_coefs);
                            let dst = mask_coefs_level_slice_of(mc_buf, offset, packed_len, nm);
                            mc_dispatch.run(packed, q, nm, &scratch.plan, dst)?;
                        }
                        if tail > 0 {
                            let rows = input.slice(body * nm..(body + tail) * nm);
                            let dst = mask_coefs_level_slice_of(mc_buf, offset + kept, tail, nm);
                            mc_dispatch.run(rows, q, nm, &scratch.tail, dst)?;
                        }
                        Ok(())
                    },
                )?;
            }
        }
        offset += k;
    }
    _span.record("kept", offset);
    Ok(Some(offset))
}

/// Sequential per-level loop (NCHW path: `layout_scratch` is shared).
fn run_levels_sequential(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    buffers: &mut DecodedOutputBuffers,
    tables: &[DflExpTable],
) -> DecoderResult<()> {
    // Iterate levels by index so we can carve disjoint mutable slices
    // out of the buffers per level. Each iteration maps the bound
    // tensors fresh — `TensorMap` lifetime is tied to the borrow it
    // takes, which the `with_mapped_input` helper encapsulates.
    //
    // For NCHW-declared children, `with_mapped_or_transposed_input`
    // first transposes the bound tensor into the per-dtype scratch in
    // `buffers.layout_scratch`, then passes the NHWC scratch view to
    // the closure. Disjoint field borrows: dst slices come from the
    // per-role buffers (boxes / scores / mask_coefs); the scratch lives
    // in `layout_scratch`. We use the free `*_level_slice_of` helpers
    // so each `&mut buffers.<field>` borrow is tracked independently.
    for (li, lvl_bind) in bind.levels.iter().enumerate() {
        let lvl = &plan.levels[li];
        // Per-level span groups all role work (box / score / mc) for one
        // FPN scale. `h*w` is the dominant cost factor.
        let _level_span = tracing::trace_span!(
            "decoder.per_scale_run.level",
            li = li,
            stride = lvl.stride,
            h = lvl.h,
            w = lvl.w,
            anchors = lvl.h * lvl.w,
            layout = ?lvl.layout,
        )
        .entered();
        let DecodedOutputBuffers {
            boxes,
            scores,
            mask_coefs,
            layout_scratch,
            ..
        } = buffers;

        // Boxes — DFL or LTRB depending on plan.box_encoding. The DFL
        // path's softmax + weighted-sum is currently the per-anchor
        // bottleneck on Cortex-A53; this span isolates the cost from
        // score / mc.
        {
            let _s = tracing::trace_span!("decoder.per_scale_run.level.boxes", encoding = ?plan.box_encoding)
                .entered();
            let dst = boxes_level_slice_of(boxes, lvl.anchor_offset, lvl.h * lvl.w);
            let box_channels = box_channel_count_for_level(lvl);
            if plan.box_encoding == crate::schema::BoxEncoding::Dfl {
                match box_per_channel_quant_from_tensor(lvl_bind.boxes, li)? {
                    Some((scales, zps)) => {
                        with_mapped_or_transposed_input(
                            lvl_bind.boxes,
                            "boxes",
                            li,
                            lvl.layout,
                            lvl.h,
                            lvl.w,
                            box_channels,
                            layout_scratch,
                            |input| dispatch_dfl_per_channel(input, scales, zps, lvl, dst),
                        )?;
                    }
                    None => {
                        // Per-tensor fallback (tensor has per-tensor quant or is float).
                        let q = quant_from_tensor(lvl_bind.boxes, "boxes", li)?;
                        with_mapped_or_transposed_input(
                            lvl_bind.boxes,
                            "boxes",
                            li,
                            lvl.layout,
                            lvl.h,
                            lvl.w,
                            box_channels,
                            layout_scratch,
                            |input| plan.box_dispatch.run(input, q, lvl, tables.get(li), dst),
                        )?;
                    }
                }
            } else {
                let q = quant_from_tensor(lvl_bind.boxes, "boxes", li)?;
                with_mapped_or_transposed_input(
                    lvl_bind.boxes,
                    "boxes",
                    li,
                    lvl.layout,
                    lvl.h,
                    lvl.w,
                    box_channels,
                    layout_scratch,
                    |input| plan.box_dispatch.run(input, q, lvl, tables.get(li), dst),
                )?;
            }
        }

        // Scores — dequant + optional sigmoid. ~32% of decode time per
        // the per-stage estimate; isolating it lets us see whether the
        // dequant or sigmoid dominates after NEON wiring.
        {
            let _s = tracing::trace_span!("decoder.per_scale_run.level.scores", activation = ?plan.score_activation).entered();
            let q = quant_from_tensor(lvl_bind.scores, "scores", li)?;
            let dst =
                scores_level_slice_of(scores, lvl.anchor_offset, lvl.h * lvl.w, plan.num_classes);
            with_mapped_or_transposed_input(
                lvl_bind.scores,
                "scores",
                li,
                lvl.layout,
                lvl.h,
                lvl.w,
                plan.num_classes,
                layout_scratch,
                |input| {
                    plan.score_dispatch.run(
                        input,
                        q,
                        plan.num_classes,
                        lvl,
                        plan.score_activation,
                        dst,
                    )
                },
            )?;
        }

        // Mask coefs — pure dequant; smaller channel count (32) than
        // scores (80) but same anchor count.
        if let (Some(mc_input), Some(mc_dispatch), Some(num_mc_gt0)) = (
            lvl_bind.mask_coefs,
            plan.mc_dispatch,
            Some(plan.num_mask_coefs).filter(|&n| n > 0),
        ) {
            let _s = tracing::trace_span!("decoder.per_scale_run.level.mask_coefs").entered();
            let q = quant_from_tensor(mc_input, "mask_coefs", li)?;
            let mc_buf = mask_coefs.as_mut().ok_or_else(|| {
                DecoderError::Internal(
                    "mask_coefs buffer absent but plan declared mc_dispatch".into(),
                )
            })?;
            let dst =
                mask_coefs_level_slice_of(mc_buf, lvl.anchor_offset, lvl.h * lvl.w, num_mc_gt0);
            with_mapped_or_transposed_input(
                mc_input,
                "mask_coefs",
                li,
                lvl.layout,
                lvl.h,
                lvl.w,
                num_mc_gt0,
                layout_scratch,
                |input| mc_dispatch.run(input, q, num_mc_gt0, lvl, dst),
            )?;
        }
    }
    Ok(())
}

/// Frames with fewer anchors than this decode on the calling thread: waking
/// the rayon workers, and their spinning once the work runs out, costs more
/// than the split saves.
const PARALLEL_MIN_ANCHORS: usize = 2048;

/// A run of one level's anchors, decoded as one task of the dense parallel
/// path. `start` is a multiple of [`PACK_ANCHORS`], so each element takes
/// the same kernel path as in a decode of the whole level.
#[derive(Debug)]
pub(crate) struct Piece {
    level: usize,
    start: usize,
    /// The level's plan cut down to the piece's anchors.
    plan: crate::per_scale::plan::LevelPlan,
}

/// The pieces of every level for `threads` rayon threads, in anchor order.
/// Each level is cut into pieces of about `total / (2 * threads)` anchors,
/// so the workers share the frame evenly however its anchors are spread
/// over the levels; a single thread takes each level whole.
pub(crate) fn split_into_pieces(plan: &PerScalePlan, threads: usize) -> Vec<Piece> {
    let target = plan
        .total_anchors
        .div_ceil(2 * threads.max(1))
        .next_multiple_of(PACK_ANCHORS);
    let mut pieces = Vec::new();
    for (li, lvl) in plan.levels.iter().enumerate() {
        let n = lvl.h * lvl.w;
        let count = if threads > 1 {
            (n + target / 2) / target
        } else {
            1
        };
        let len = n.div_ceil(count.max(1)).next_multiple_of(PACK_ANCHORS);
        let mut start = 0;
        while start < n {
            let end = (start + len).min(n);
            pieces.push(Piece {
                level: li,
                start,
                plan: crate::per_scale::plan::LevelPlan {
                    stride: lvl.stride,
                    h: 1,
                    w: end - start,
                    reg_max: lvl.reg_max,
                    anchor_offset: lvl.anchor_offset + start,
                    grid_x: lvl.grid_x[start..end].into(),
                    grid_y: lvl.grid_y[start..end].into(),
                    box_shape: lvl.box_shape.clone(),
                    score_shape: lvl.score_shape.clone(),
                    mc_shape: lvl.mc_shape.clone(),
                    layout: Layout::Nhwc,
                },
            });
            start = end;
        }
    }
    pieces
}

/// One level's inputs, mapped for the duration of a frame.
struct LevelMaps<'a> {
    boxes: MappedInput<'a>,
    scores: MappedInput<'a>,
    mask_coefs: Option<MappedInput<'a>>,
}

/// One piece's work: its outputs and the inputs it reads.
struct PieceWork<'p, 'o> {
    piece: &'p Piece,
    boxes: DstSliceMut<'o>,
    scores: DstSliceMut<'o>,
    mask_coefs: Option<DstSliceMut<'o>>,
}

/// Dense loop for NHWC levels. Every input is mapped once on the calling
/// thread and the levels are decoded in [`Piece`]s, on the rayon pool when
/// the frame has at least [`PARALLEL_MIN_ANCHORS`] anchors.
fn run_levels_parallel(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    buffers: &mut DecodedOutputBuffers,
    tables: &[DflExpTable],
) -> DecoderResult<()> {
    let threads = if plan.total_anchors < PARALLEL_MIN_ANCHORS {
        1
    } else {
        rayon::current_num_threads()
    };
    if buffers.pieces.0 != threads {
        buffers.pieces = (threads, split_into_pieces(plan, threads));
    }
    let maps = bind
        .levels
        .iter()
        .enumerate()
        .map(|(li, lvl_bind)| {
            Ok(LevelMaps {
                boxes: map_input(lvl_bind.boxes, "boxes", li)?,
                scores: map_input(lvl_bind.scores, "scores", li)?,
                mask_coefs: match (lvl_bind.mask_coefs, plan.num_mask_coefs) {
                    (Some(mc), nm) if nm > 0 => Some(map_input(mc, "mask_coefs", li)?),
                    _ => None,
                },
            })
        })
        .collect::<DecoderResult<Vec<_>>>()?;

    let DecodedOutputBuffers {
        boxes,
        scores,
        mask_coefs,
        pieces,
        ..
    } = buffers;
    let pieces = &pieces.1;
    let spans = |per_anchor: usize| -> Vec<(usize, usize)> {
        pieces
            .iter()
            .map(|p| (p.plan.anchor_offset * per_anchor, p.plan.w * per_anchor))
            .collect()
    };
    let nm = plan.num_mask_coefs;
    let box_dsts = split_buffer_into_levels(boxes, &spans(4));
    let score_dsts = split_buffer_into_levels(scores, &spans(plan.num_classes));
    let mut mc_dsts = match mask_coefs.as_mut() {
        Some(mc) if nm > 0 => split_buffer_into_levels(mc, &spans(nm))
            .into_iter()
            .map(Some)
            .collect(),
        _ => Vec::new(),
    };
    mc_dsts.resize_with(pieces.len(), || None);
    let work: Vec<PieceWork<'_, '_>> = pieces
        .iter()
        .zip(box_dsts)
        .zip(score_dsts)
        .zip(mc_dsts)
        .map(|(((piece, boxes), scores), mask_coefs)| PieceWork {
            piece,
            boxes,
            scores,
            mask_coefs,
        })
        .collect();

    let run = |w: PieceWork<'_, '_>| {
        let li = w.piece.level;
        decode_piece(plan, &bind.levels[li], &maps[li], tables.get(li), w)
    };
    if threads == 1 {
        work.into_iter().try_for_each(run)
    } else {
        use rayon::prelude::*;
        work.into_par_iter().try_for_each(run)
    }
}

/// Box, score and mask-coefficient decode of one [`Piece`].
fn decode_piece(
    plan: &PerScalePlan,
    lvl_bind: &LevelBindings<'_>,
    maps: &LevelMaps<'_>,
    table: Option<&DflExpTable>,
    work: PieceWork<'_, '_>,
) -> DecoderResult<()> {
    let PieceWork {
        piece,
        boxes,
        scores,
        mask_coefs,
    } = work;
    let (li, lvl) = (piece.level, &piece.plan);
    fn rows<'v>(view: InputView<'v>, piece: &Piece, per_anchor: usize) -> InputView<'v> {
        let start = piece.start * per_anchor;
        view.slice(start..start + piece.plan.w * per_anchor)
    }
    let _span = tracing::trace_span!(
        "decoder.per_scale_run.level",
        li = li,
        start = piece.start,
        anchors = lvl.w,
    )
    .entered();
    {
        let _s = tracing::trace_span!("decoder.per_scale_run.level.boxes", encoding = ?plan.box_encoding)
            .entered();
        let input = rows(maps.boxes.view(), piece, box_channel_count_for_level(lvl));
        run_box_kernel(plan, lvl, lvl_bind.boxes, li, table, input, boxes)?;
    }
    {
        let _s = tracing::trace_span!("decoder.per_scale_run.level.scores", activation = ?plan.score_activation)
            .entered();
        let q = quant_from_tensor(lvl_bind.scores, "scores", li)?;
        let nc = plan.num_classes;
        let input = rows(maps.scores.view(), piece, nc);
        plan.score_dispatch
            .run(input, q, nc, lvl, plan.score_activation, scores)?;
    }
    if let (Some(mc_input), Some(mc_map), Some(mc_dispatch), Some(dst)) = (
        lvl_bind.mask_coefs,
        maps.mask_coefs.as_ref(),
        plan.mc_dispatch,
        mask_coefs,
    ) {
        let _s = tracing::trace_span!("decoder.per_scale_run.level.mask_coefs").entered();
        let q = quant_from_tensor(mc_input, "mask_coefs", li)?;
        let nm = plan.num_mask_coefs;
        mc_dispatch.run(rows(mc_map.view(), piece, nm), q, nm, lvl, dst)?;
    }
    Ok(())
}

/// Split a `Buffer` into N disjoint sub-views, one per level. The
/// `(start, len)` spans must be in increasing order and must lie
/// fully inside the buffer (caller's responsibility).
fn split_buffer_into_levels<'a>(
    buf: &'a mut crate::per_scale::outputs::Buffer,
    spans: &[(usize, usize)],
) -> Vec<crate::per_scale::kernels::dispatch::DstSliceMut<'a>> {
    use crate::per_scale::kernels::dispatch::DstSliceMut;
    use crate::per_scale::outputs::Buffer;
    let mut out = Vec::with_capacity(spans.len());
    match buf {
        Buffer::F32(v) => {
            let mut remaining: &mut [f32] = v.as_mut_slice();
            let mut cursor = 0usize;
            for &(start, len) in spans {
                let skip = start - cursor;
                let (_, after_skip) = remaining.split_at_mut(skip);
                let (chunk, rest) = after_skip.split_at_mut(len);
                out.push(DstSliceMut::F32(chunk));
                remaining = rest;
                cursor = start + len;
            }
        }
        Buffer::F16(v) => {
            let mut remaining: &mut [half::f16] = v.as_mut_slice();
            let mut cursor = 0usize;
            for &(start, len) in spans {
                let skip = start - cursor;
                let (_, after_skip) = remaining.split_at_mut(skip);
                let (chunk, rest) = after_skip.split_at_mut(len);
                out.push(DstSliceMut::F16(chunk));
                remaining = rest;
                cursor = start + len;
            }
        }
    }
    out
}

/// Compute box-channel count for a level for the transpose dispatcher.
///
/// DFL: `4 * reg_max` (boxes carry the raw DFL bin distributions).
/// LTRB / Direct: 4 (boxes carry the four LTRB grid-unit distances directly).
///
/// Used to size the per-level scratch buffer when an NCHW box child is
/// transposed to NHWC. The count must match the actual channel-axis
/// length of the bound tensor; a mismatch indicates schema-vs-runtime
/// drift and would corrupt the transpose.
fn box_channel_count_for_level(lvl: &crate::per_scale::plan::LevelPlan) -> usize {
    // For both DFL (`reg_max > 1`) and LTRB (`reg_max == 1`), the
    // channel count is `4 * reg_max`: 4 for LTRB (since reg_max == 1)
    // and `4 * 16` (or whatever reg_max is) for DFL.
    4 * lvl.reg_max
}

/// Map a `TensorDyn`'s underlying slice. If the level's layout is NCHW,
/// transpose it into the appropriate per-dtype scratch in
/// `layout_scratch` and pass an NHWC `InputView` over the scratch to
/// the closure. NHWC inputs pass through unchanged (zero overhead).
///
/// The transpose runs the scalar `nchw_to_nhwc` helper from
/// `kernels::transpose`; NEON tile-transpose can lift the
/// `tiled_proto_transpose_nchw_to_nhwc` pattern from `main`'s mask
/// pipeline as a drop-in replacement.
#[allow(clippy::too_many_arguments)]
fn with_mapped_or_transposed_input<F>(
    t: &TensorDyn,
    role: &'static str,
    level: usize,
    layout: Layout,
    h: usize,
    w: usize,
    c: usize,
    scratch: &mut crate::per_scale::outputs::LayoutScratch,
    f: F,
) -> DecoderResult<()>
where
    F: FnOnce(InputView<'_>) -> DecoderResult<()>,
{
    if layout == Layout::Nhwc {
        return with_mapped_input(t, role, level, f);
    }

    // NCHW path: read source bytes via TensorMap, transpose into the
    // per-dtype scratch, and pass an InputView over the scratch.
    reject_padded_input(t, role, level)?;
    let n = h * w * c;
    let map_fail = |e: edgefirst_tensor::Error| map_failure(e, role, level);
    macro_rules! nchw {
        ($ty:ty, $ensure:ident, $variant:ident) => {{
            let tensor = t.as_typed::<$ty>().expect("dtype checked");
            let m = tensor.map_read().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.$ensure(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::$variant(dst))
        }};
    }
    match t.dtype() {
        DType::I8 => nchw!(i8, ensure_i8, I8),
        DType::U8 => nchw!(u8, ensure_u8, U8),
        DType::I16 => nchw!(i16, ensure_i16, I16),
        DType::U16 => nchw!(u16, ensure_u16, U16),
        DType::F16 => nchw!(half::f16, ensure_f16, F16),
        DType::F32 => nchw!(f32, ensure_f32, F32),
        other => Err(DecoderError::DtypeMismatch {
            expected: edgefirst_tensor::DType::I8,
            actual: other,
            role,
            level,
        }),
    }
}

/// Refuse an input whose rows carry a padded pitch: the per-scale kernels
/// read their inputs as tightly packed slices.
fn reject_padded_input(t: &TensorDyn, role: &'static str, level: usize) -> DecoderResult<()> {
    match crate::decoder::tensor_bridge::padded_strides(t)? {
        None => Ok(()),
        Some(_) => Err(DecoderError::NotSupported(format!(
            "{role} (level {level}): per-scale decode requires a tightly packed input, \
             got row stride {:?} for shape {:?}",
            t.row_stride(),
            t.shape()
        ))),
    }
}

/// The error for a failed input map of `role` at `level`.
fn map_failure(e: edgefirst_tensor::Error, role: &str, level: usize) -> DecoderError {
    DecoderError::from_map(e, |e| {
        DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
    })
}

/// A tensor mapped read-only, holding its map for as long as it lives.
enum MappedInput<'a> {
    I8(edgefirst_tensor::HostView<'a, i8>),
    U8(edgefirst_tensor::HostView<'a, u8>),
    I16(edgefirst_tensor::HostView<'a, i16>),
    U16(edgefirst_tensor::HostView<'a, u16>),
    F16(edgefirst_tensor::HostView<'a, half::f16>),
    F32(edgefirst_tensor::HostView<'a, f32>),
}

impl MappedInput<'_> {
    fn view(&self) -> InputView<'_> {
        match self {
            MappedInput::I8(m) => InputView::I8(m.as_slice()),
            MappedInput::U8(m) => InputView::U8(m.as_slice()),
            MappedInput::I16(m) => InputView::I16(m.as_slice()),
            MappedInput::U16(m) => InputView::U16(m.as_slice()),
            MappedInput::F16(m) => InputView::F16(m.as_slice()),
            MappedInput::F32(m) => InputView::F32(m.as_slice()),
        }
    }
}

/// Map a `TensorDyn` read-only as an [`InputView`] source.
fn map_input<'a>(
    t: &'a TensorDyn,
    role: &'static str,
    level: usize,
) -> DecoderResult<MappedInput<'a>> {
    reject_padded_input(t, role, level)?;
    macro_rules! mapped {
        ($ty:ty, $variant:ident) => {{
            let tensor = t.as_typed::<$ty>().expect("dtype checked");
            let m = tensor.map_read().map_err(|e| map_failure(e, role, level))?;
            Ok(MappedInput::$variant(m))
        }};
    }
    match t.dtype() {
        DType::I8 => mapped!(i8, I8),
        DType::U8 => mapped!(u8, U8),
        DType::I16 => mapped!(i16, I16),
        DType::U16 => mapped!(u16, U16),
        DType::F16 => mapped!(half::f16, F16),
        DType::F32 => mapped!(f32, F32),
        other => Err(DecoderError::DtypeMismatch {
            expected: edgefirst_tensor::DType::I8, // arbitrary; we just want to flag mismatch
            actual: other,
            role,
            level,
        }),
    }
}

/// Map a `TensorDyn`'s underlying slice read-only and call the closure with
/// an `InputView`. The map is dropped when the closure returns.
fn with_mapped_input<F>(t: &TensorDyn, role: &'static str, level: usize, f: F) -> DecoderResult<()>
where
    F: FnOnce(InputView<'_>) -> DecoderResult<()>,
{
    f(map_input(t, role, level)?.view())
}

/// Build a `DecodedOutputsRef` borrowing the first `total_anchors`
/// anchors of the buffers.
fn make_outputs_ref<'a>(
    buffers: &'a DecodedOutputBuffers,
    plan: &PerScalePlan,
    total_anchors: usize,
) -> DecodedOutputsRef<'a> {
    let prefix = |b: &'a Buffer, per_anchor: usize| match b {
        Buffer::F32(v) => BufferRef::F32(&v[..total_anchors * per_anchor]),
        Buffer::F16(v) => BufferRef::F16(&v[..total_anchors * per_anchor]),
    };
    let boxes = prefix(&buffers.boxes, 4);
    let scores = prefix(&buffers.scores, plan.num_classes);
    let mask_coefs = buffers
        .mask_coefs
        .as_ref()
        .map(|b| prefix(b, plan.num_mask_coefs));
    let protos = buffers.protos.as_ref().map(|p| match p {
        ProtoStorage::F32(a) => ProtosView::F32(a.view()),
        ProtoStorage::F16(a) => ProtosView::F16(a.view()),
    });
    DecodedOutputsRef {
        boxes,
        scores,
        mask_coefs,
        protos,
        total_anchors,
        num_classes: plan.num_classes,
        num_mask_coefs: plan.num_mask_coefs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_tensor::{Tensor, TensorMemory};

    /// Build a small TensorDyn with the given shape and dtype.
    /// Helper for tests.
    fn mk_i8_tensor(shape: &[usize]) -> TensorDyn {
        let t = Tensor::<i8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        t.into()
    }

    fn mk_f32_tensor(shape: &[usize]) -> TensorDyn {
        let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        t.into()
    }

    #[test]
    fn quant_from_tensor_returns_identity_for_float() {
        let t = mk_f32_tensor(&[1, 2, 2, 4]);
        let q = quant_from_tensor(&t, "boxes", 0).unwrap();
        assert_eq!(q, Quantization::identity());
    }

    #[test]
    fn quant_from_tensor_errors_for_unattached_int8() {
        let t = mk_i8_tensor(&[1, 2, 2, 4]);
        let r = quant_from_tensor(&t, "boxes", 0);
        assert!(matches!(
            r,
            Err(DecoderError::QuantMissing {
                dtype: DType::I8,
                ..
            })
        ));
    }

    #[test]
    fn quant_from_tensor_reads_attached_int8() {
        use edgefirst_tensor::Quantization as TQ;
        let mut t = Tensor::<i8>::new(&[1, 2, 2, 4], Some(TensorMemory::Mem), None).unwrap();
        t.set_quantization(TQ::per_tensor(0.1, -10)).unwrap();
        let dyn_t: TensorDyn = t.into();
        let q = quant_from_tensor(&dyn_t, "boxes", 0).unwrap();
        assert!((q.scale - 0.1).abs() < 1e-7);
        assert_eq!(q.zero_point, -10);
    }

    /// `bind_one` errors when no remaining tensor matches the requested shape.
    #[test]
    fn bind_one_errors_on_no_shape_match() {
        let t1 = mk_f32_tensor(&[1, 1, 1, 4]);
        let inputs = vec![&t1];
        let mut used = vec![false];
        let r = bind_one(&inputs, &mut used, &[1, 99, 99, 4], "boxes", 0);
        assert!(r.is_err());
    }

    /// `bind_one` consumes the first match; the second call won't reuse it.
    #[test]
    fn bind_one_consumes_match_via_used_array() {
        let t1 = mk_f32_tensor(&[1, 2, 2, 4]);
        let t2 = mk_f32_tensor(&[1, 2, 2, 4]); // same shape as t1
        let inputs = vec![&t1, &t2];
        let mut used = vec![false; 2];

        let r1 = bind_one(&inputs, &mut used, &[1, 2, 2, 4], "boxes", 0);
        assert!(r1.is_ok());
        assert_eq!(used, vec![true, false]);

        let r2 = bind_one(&inputs, &mut used, &[1, 2, 2, 4], "scores", 0);
        assert!(r2.is_ok());
        assert_eq!(used, vec![true, true]);

        let r3 = bind_one(&inputs, &mut used, &[1, 2, 2, 4], "mc", 0);
        assert!(r3.is_err()); // none left
    }

    #[test]
    fn run_smoke_test_constructs_decoder_with_correctly_sized_buffers() {
        use crate::per_scale::DecodeDtype;
        use crate::per_scale::PerScaleDecoder;
        use crate::schema::SchemaV2;

        // Use the synthetic yolov8n schema to build a plan, then verify
        // PerScaleDecoder::new() produces buffers of the right size.
        //
        // Note: the plan was built expecting i8 tensors (per the schema's
        // quantization), so the box_dispatch is selected as DflI8ToF32Lut.
        // Feeding f32 tensors would hit the dispatch's mismatched-arm path
        // → KernelDispatchUnreachable. End-to-end run() integration is
        // exercised separately via the int8-with-attached-quant test below.
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(&json).unwrap();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();

        let decoder = PerScaleDecoder::new(plan);
        // After new, buffers should be sized.
        match &decoder.buffers.boxes {
            crate::per_scale::outputs::Buffer::F32(v) => {
                assert_eq!(
                    v.len(),
                    4 * (8400 + crate::per_scale::prefilter::PACK_ANCHORS - 1)
                )
            }
            _ => panic!("expected F32 boxes buffer"),
        }
    }

    #[test]
    fn run_with_int8_inputs_and_attached_quant() {
        use crate::per_scale::outputs::BufferRef;
        use crate::per_scale::DecodeDtype;
        use crate::per_scale::PerScaleDecoder;
        use crate::schema::SchemaV2;
        use edgefirst_tensor::Quantization as TQ;
        use edgefirst_tensor::{Tensor, TensorMemory};

        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(&json).unwrap();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();

        // Build i8 inputs for all 10 physical tensors with attached quant.
        let mut inputs_owned: Vec<TensorDyn> = Vec::new();
        for lvl in &plan.levels {
            // boxes
            let mut t = Tensor::<i8>::new(&lvl.box_shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(t.into());
            // scores
            let mut t = Tensor::<i8>::new(&lvl.score_shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(t.into());
            // mask_coefs
            if let Some(s) = &lvl.mc_shape {
                let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
                t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
                inputs_owned.push(t.into());
            }
        }
        // protos
        if let Some(s) = &plan.proto_shape {
            let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(t.into());
        }
        let inputs: Vec<&TensorDyn> = inputs_owned.iter().collect();

        let mut decoder = PerScaleDecoder::new(plan);
        let result = decoder.run(&inputs, None);
        assert!(
            result.is_ok(),
            "run should succeed on zero-i8 inputs: {result:?}"
        );
        let outputs = result.unwrap();
        assert_eq!(outputs.total_anchors, 8400);
        assert_eq!(outputs.num_classes, 80);
        assert_eq!(outputs.num_mask_coefs, 32);

        // Boxes have finite, non-NaN values (zero-input + zero-zp + 0.1-scale →
        // softmax uniform → distance 7.5 grid units).
        if let BufferRef::F32(v) = &outputs.boxes {
            for (i, &b) in v.iter().enumerate() {
                assert!(b.is_finite(), "box[{i}] = {b} not finite");
            }
        }
    }

    /// Pieces tile every level in order from pack-aligned starts, and cut
    /// a 960x544 frame's 8160/2040/510 levels into pieces of similar size.
    #[test]
    fn pieces_tile_each_level_from_pack_aligned_starts() {
        use crate::per_scale::DecodeDtype;
        use crate::schema::SchemaV2;
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for logical in v["outputs"].as_array_mut().unwrap() {
            if let Some(children) = logical.get_mut("outputs").and_then(|c| c.as_array_mut()) {
                for child in children {
                    let stride = child["stride"].as_u64().unwrap() as usize;
                    child["shape"][1] = (544 / stride).into();
                    child["shape"][2] = (960 / stride).into();
                    child["dshape"][1]["height"] = (544 / stride).into();
                    child["dshape"][2]["width"] = (960 / stride).into();
                }
            }
        }
        let schema: SchemaV2 = serde_json::from_value(v).unwrap();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        assert_eq!(plan.total_anchors, 10710);
        for threads in [1, 2, 6, 8] {
            let pieces = split_into_pieces(&plan, threads);
            let mut next = 0;
            for p in &pieces {
                let lvl = &plan.levels[p.level];
                assert_eq!(p.start % PACK_ANCHORS, 0);
                assert_eq!(p.plan.anchor_offset, next, "{threads} threads");
                assert_eq!(p.plan.anchor_offset, lvl.anchor_offset + p.start);
                assert_eq!(p.plan.grid_x[..], lvl.grid_x[p.start..p.start + p.plan.w]);
                assert_eq!(p.plan.grid_y[..], lvl.grid_y[p.start..p.start + p.plan.w]);
                next += p.plan.w;
            }
            assert_eq!(next, plan.total_anchors);
            let largest = pieces.iter().map(|p| p.plan.w).max().unwrap();
            if threads == 1 {
                assert_eq!(pieces.len(), plan.levels.len());
            } else {
                assert!(
                    pieces.len() >= threads,
                    "{threads} threads: {}",
                    pieces.len()
                );
                assert!(largest <= 2 * plan.total_anchors / pieces.len());
            }
        }
    }

    #[test]
    fn padded_row_input_is_rejected_on_both_layouts() {
        let parent = edgefirst_tensor::Tensor::<i8>::image(
            4,
            2,
            edgefirst_tensor::PixelFormat::Grey,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let view: TensorDyn = parent
            .view(edgefirst_tensor::Region::new(0, 0, 3, 2))
            .unwrap()
            .into();
        assert_eq!(view.row_stride(), Some(4));

        let r = with_mapped_input(&view, "boxes", 0, |_| Ok(()));
        assert!(matches!(r, Err(DecoderError::NotSupported(_))), "{r:?}");

        let mut scratch = crate::per_scale::outputs::LayoutScratch::default();
        let r = with_mapped_or_transposed_input(
            &view,
            "boxes",
            0,
            Layout::Nchw,
            3,
            1,
            2,
            &mut scratch,
            |_| Ok(()),
        );
        assert!(matches!(r, Err(DecoderError::NotSupported(_))), "{r:?}");

        let packed = mk_i8_tensor(&[2, 3, 1]);
        assert!(with_mapped_input(&packed, "boxes", 0, |_| Ok(())).is_ok());
    }
}
