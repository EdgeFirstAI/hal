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
    match t {
        TensorDyn::I8(t) => t
            .quantization()
            .map(convert_quant)
            .ok_or(DecoderError::QuantMissing {
                dtype: DType::I8,
                role,
                level,
            }),
        TensorDyn::U8(t) => t
            .quantization()
            .map(convert_quant)
            .ok_or(DecoderError::QuantMissing {
                dtype: DType::U8,
                role,
                level,
            }),
        TensorDyn::I16(t) => {
            t.quantization()
                .map(convert_quant)
                .ok_or(DecoderError::QuantMissing {
                    dtype: DType::I16,
                    role,
                    level,
                })
        }
        TensorDyn::U16(t) => {
            t.quantization()
                .map(convert_quant)
                .ok_or(DecoderError::QuantMissing {
                    dtype: DType::U16,
                    role,
                    level,
                })
        }
        TensorDyn::F16(_) | TensorDyn::F32(_) => Ok(Quantization::identity()),
        other => Err(DecoderError::DtypeMismatch {
            expected: DType::I8,
            actual: other.dtype(),
            role,
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

use crate::per_scale::kernels::dispatch::InputView;
use crate::per_scale::kernels::transpose::nchw_to_nhwc;
use crate::per_scale::outputs::{
    boxes_level_slice_of, mask_coefs_level_slice_of, scores_level_slice_of, Buffer, BufferRef,
    DecodedOutputBuffers, DecodedOutputsRef, ProtoStorage, ProtosView,
};
use crate::per_scale::plan::Layout;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};

/// Run the per-scale decode pipeline for one frame's inputs. Writes
/// into `buffers` and returns a borrowed view.
#[allow(dead_code)] // Wired by Task 24.
pub(crate) fn run<'a>(
    plan: &PerScalePlan,
    buffers: &'a mut DecodedOutputBuffers,
    inputs: &[&TensorDyn],
) -> DecoderResult<DecodedOutputsRef<'a>> {
    // Top-level span. The `n_levels` field gives at-a-glance shape in
    // Perfetto's flame graph; per-level / per-role child spans below
    // give the per-stage decomposition we need to find Phase 2 hotspots.
    let _outer = tracing::trace_span!(
        "per_scale_decode",
        n_levels = plan.levels.len(),
        encoding = ?plan.box_encoding,
        nc = plan.num_classes,
        nm = plan.num_mask_coefs,
    )
    .entered();
    let bind = {
        let _s = tracing::trace_span!("resolve_bindings").entered();
        resolve_bindings(plan, inputs)?
    };

    // Per-level work is independent: each level writes to disjoint
    // anchor ranges of `boxes` / `scores` / `mask_coefs`. We
    // parallelize across levels via rayon::scope when all levels are
    // NHWC (the common TFLite case). NCHW levels share `layout_scratch`
    // which would need per-level scratch buffers to parallelize safely;
    // for now we keep the sequential path for that case.
    let all_nhwc = plan.levels.iter().all(|l| l.layout == Layout::Nhwc);
    if all_nhwc {
        run_levels_parallel(plan, &bind, buffers)?;
    } else {
        run_levels_sequential(plan, &bind, buffers)?;
    }

    // Protos — single tensor, no per-level structure. Always sequential.
    if let (Some(proto_input), Some(proto_dispatch)) = (bind.protos, plan.proto_dispatch) {
        let _s = tracing::trace_span!("protos").entered();
        let q = quant_from_tensor(proto_input, "protos", 0)?;
        let dst = buffers.protos_dst().ok_or_else(|| {
            DecoderError::Internal("protos buffer absent but plan declared proto_dispatch".into())
        })?;
        with_mapped_input(proto_input, "protos", 0, |input| {
            proto_dispatch.run(input, q, dst)
        })?;
    }

    Ok(make_outputs_ref(buffers, plan))
}

/// Sequential per-level loop (NCHW path: `layout_scratch` is shared).
fn run_levels_sequential(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    buffers: &mut DecodedOutputBuffers,
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
            "level",
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
            let _s = tracing::trace_span!("box", encoding = ?plan.box_encoding).entered();
            let q = quant_from_tensor(lvl_bind.boxes, "boxes", li)?;
            let dst = boxes_level_slice_of(boxes, lvl.anchor_offset, lvl.h * lvl.w);
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
                |input| plan.box_dispatch.run(input, q, lvl, dst),
            )?;
        }

        // Scores — dequant + optional sigmoid. ~32% of decode time per
        // the per-stage estimate; isolating it lets us see whether the
        // dequant or sigmoid dominates after NEON wiring.
        {
            let _s = tracing::trace_span!("score", activation = ?plan.score_activation).entered();
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
            let _s = tracing::trace_span!("mc").entered();
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

/// Parallel per-level loop (NHWC-only path). Disjoint mutable slices
/// of `boxes` / `scores` / `mask_coefs` are pre-split per level so
/// each rayon worker can write independently. NCHW would need
/// per-level `layout_scratch` to parallelize; that's deferred.
fn run_levels_parallel(
    plan: &PerScalePlan,
    bind: &FrameBindings<'_>,
    buffers: &mut DecodedOutputBuffers,
) -> DecoderResult<()> {
    let n_levels = plan.levels.len();
    if n_levels == 0 {
        return Ok(());
    }

    // Per-level (start_elem, len_elem) spans for each output buffer.
    // Levels are contiguous and disjoint by anchor_offset construction.
    let box_spans: Vec<(usize, usize)> = plan
        .levels
        .iter()
        .map(|l| (l.anchor_offset * 4, l.h * l.w * 4))
        .collect();
    let score_spans: Vec<(usize, usize)> = plan
        .levels
        .iter()
        .map(|l| {
            (
                l.anchor_offset * plan.num_classes,
                l.h * l.w * plan.num_classes,
            )
        })
        .collect();
    let mc_spans: Vec<(usize, usize)> = plan
        .levels
        .iter()
        .map(|l| {
            (
                l.anchor_offset * plan.num_mask_coefs,
                l.h * l.w * plan.num_mask_coefs,
            )
        })
        .collect();

    let DecodedOutputBuffers {
        boxes,
        scores,
        mask_coefs,
        ..
    } = buffers;

    let box_dsts = split_buffer_into_levels(boxes, &box_spans);
    let score_dsts = split_buffer_into_levels(scores, &score_spans);
    let mc_dsts = if plan.num_mask_coefs > 0 {
        mask_coefs
            .as_mut()
            .map(|mc| split_buffer_into_levels(mc, &mc_spans))
    } else {
        None
    };

    // Collect first error from any worker. Spawn closures can't return
    // Result, so we funnel errors through a Mutex<Option<_>>.
    let first_error: std::sync::Mutex<Option<DecoderError>> = std::sync::Mutex::new(None);

    // Pre-collect all per-level dst slices into Option holders indexed by
    // level. We later `take()` them out one at a time (in a non-sequential
    // order: spawn levels 1..N first, then process level 0 on the
    // calling thread). Levels share a vec but each `take()` is disjoint.
    let mut box_dst_opts: Vec<Option<_>> = box_dsts.into_iter().map(Some).collect();
    let mut score_dst_opts: Vec<Option<_>> = score_dsts.into_iter().map(Some).collect();
    let mut mc_dst_opts: Option<Vec<Option<_>>> =
        mc_dsts.map(|v| v.into_iter().map(Some).collect());

    // Process the heaviest level (index 0 in YOLO-style FPN, 80x80 grid)
    // on the calling thread; spawn the smaller levels onto rayon
    // workers. This eliminates one barrier-wait round-trip vs spawning
    // every level — rayon's scope() overhead is ~4 ms / frame on
    // imx95-evk (Cortex-A55), so saving even one trip matters.
    rayon::scope(|s| {
        let bind_levels = &bind.levels;
        let plan_levels = &plan.levels;
        let plan_ref = plan;
        let first_error = &first_error;

        // Spawn levels 1..N onto workers.
        for li in 1..n_levels {
            let lvl_bind = &bind_levels[li];
            let lvl = &plan_levels[li];
            let box_dst = box_dst_opts[li].take().unwrap();
            let score_dst = score_dst_opts[li].take().unwrap();
            let mc_dst = mc_dst_opts.as_mut().and_then(|v| v[li].take());

            s.spawn(move |_| {
                if let Err(e) =
                    process_one_level_nhwc(plan_ref, lvl, li, lvl_bind, box_dst, score_dst, mc_dst)
                {
                    let mut g = first_error.lock().unwrap();
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            });
        }

        // Process level 0 in the calling thread. This overlaps with the
        // spawned workers' execution; the scope's join point waits for
        // both to finish.
        let lvl_bind = &bind_levels[0];
        let lvl = &plan_levels[0];
        let box_dst = box_dst_opts[0].take().unwrap();
        let score_dst = score_dst_opts[0].take().unwrap();
        let mc_dst = mc_dst_opts.as_mut().and_then(|v| v[0].take());
        if let Err(e) =
            process_one_level_nhwc(plan_ref, lvl, 0, lvl_bind, box_dst, score_dst, mc_dst)
        {
            let mut g = first_error.lock().unwrap();
            if g.is_none() {
                *g = Some(e);
            }
        }
    });

    if let Some(e) = first_error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

/// Per-level box + score + mc work for the NHWC parallel path. Mirrors
/// the body of the sequential loop but takes pre-split disjoint
/// destination slices instead of slicing live from the buffers struct.
fn process_one_level_nhwc(
    plan: &PerScalePlan,
    lvl: &crate::per_scale::plan::LevelPlan,
    li: usize,
    lvl_bind: &LevelBindings<'_>,
    box_dst: crate::per_scale::kernels::dispatch::DstSliceMut<'_>,
    score_dst: crate::per_scale::kernels::dispatch::DstSliceMut<'_>,
    mc_dst: Option<crate::per_scale::kernels::dispatch::DstSliceMut<'_>>,
) -> DecoderResult<()> {
    let _level_span = tracing::trace_span!(
        "level",
        li = li,
        stride = lvl.stride,
        h = lvl.h,
        w = lvl.w,
        anchors = lvl.h * lvl.w,
        layout = ?lvl.layout,
    )
    .entered();

    {
        let _s = tracing::trace_span!("box", encoding = ?plan.box_encoding).entered();
        let q = quant_from_tensor(lvl_bind.boxes, "boxes", li)?;
        with_mapped_input(lvl_bind.boxes, "boxes", li, |input| {
            plan.box_dispatch.run(input, q, lvl, box_dst)
        })?;
    }

    {
        let _s = tracing::trace_span!("score", activation = ?plan.score_activation).entered();
        let q = quant_from_tensor(lvl_bind.scores, "scores", li)?;
        with_mapped_input(lvl_bind.scores, "scores", li, |input| {
            plan.score_dispatch.run(
                input,
                q,
                plan.num_classes,
                lvl,
                plan.score_activation,
                score_dst,
            )
        })?;
    }

    if let (Some(mc_input), Some(mc_dispatch), Some(num_mc_gt0), Some(mc_dst)) = (
        lvl_bind.mask_coefs,
        plan.mc_dispatch,
        Some(plan.num_mask_coefs).filter(|&n| n > 0),
        mc_dst,
    ) {
        let _s = tracing::trace_span!("mc").entered();
        let q = quant_from_tensor(mc_input, "mask_coefs", li)?;
        with_mapped_input(mc_input, "mask_coefs", li, |input| {
            mc_dispatch.run(input, q, num_mc_gt0, lvl, mc_dst)
        })?;
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
/// `kernels::transpose`; Phase 2 NEON tile-transpose can lift the
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
    let n = h * w * c;
    let map_fail = |e: edgefirst_tensor::Error| {
        DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
    };
    match t {
        TensorDyn::I8(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_i8(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::I8(dst))
        }
        TensorDyn::U8(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_u8(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::U8(dst))
        }
        TensorDyn::I16(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_i16(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::I16(dst))
        }
        TensorDyn::U16(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_u16(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::U16(dst))
        }
        TensorDyn::F16(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_f16(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::F16(dst))
        }
        TensorDyn::F32(tensor) => {
            let m = tensor.map().map_err(map_fail)?;
            let src = m.as_slice();
            if src.len() != n {
                return Err(DecoderError::InvalidShape(format!(
                    "{role} (level {level}): NCHW src len {} != h*w*c {n}",
                    src.len()
                )));
            }
            let dst = scratch.ensure_f32(n);
            nchw_to_nhwc(src, h, w, c, dst);
            f(InputView::F32(dst))
        }
        other => Err(DecoderError::DtypeMismatch {
            expected: edgefirst_tensor::DType::I8,
            actual: other.dtype(),
            role,
            level,
        }),
    }
}

/// Map a `TensorDyn`'s underlying slice and call the closure with an
/// `InputView`. The map is dropped when the closure returns.
fn with_mapped_input<F>(t: &TensorDyn, role: &'static str, level: usize, f: F) -> DecoderResult<()>
where
    F: FnOnce(InputView<'_>) -> DecoderResult<()>,
{
    match t {
        TensorDyn::I8(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::I8(m.as_slice()))
        }
        TensorDyn::U8(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::U8(m.as_slice()))
        }
        TensorDyn::I16(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::I16(m.as_slice()))
        }
        TensorDyn::U16(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::U16(m.as_slice()))
        }
        TensorDyn::F16(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::F16(m.as_slice()))
        }
        TensorDyn::F32(tensor) => {
            let m = tensor.map().map_err(|e| {
                DecoderError::Internal(format!("tensor map failed for {role} (level {level}): {e}"))
            })?;
            f(InputView::F32(m.as_slice()))
        }
        other => Err(DecoderError::DtypeMismatch {
            expected: edgefirst_tensor::DType::I8, // arbitrary; we just want to flag mismatch
            actual: other.dtype(),
            role,
            level,
        }),
    }
}

/// Build a `DecodedOutputsRef` borrowing the buffers.
fn make_outputs_ref<'a>(
    buffers: &'a DecodedOutputBuffers,
    plan: &PerScalePlan,
) -> DecodedOutputsRef<'a> {
    let boxes = match &buffers.boxes {
        Buffer::F32(v) => BufferRef::F32(v),
        Buffer::F16(v) => BufferRef::F16(v),
    };
    let scores = match &buffers.scores {
        Buffer::F32(v) => BufferRef::F32(v),
        Buffer::F16(v) => BufferRef::F16(v),
    };
    let mask_coefs = buffers.mask_coefs.as_ref().map(|b| match b {
        Buffer::F32(v) => BufferRef::F32(v),
        Buffer::F16(v) => BufferRef::F16(v),
    });
    let protos = buffers.protos.as_ref().map(|p| match p {
        ProtoStorage::F32(a) => ProtosView::F32(a.view()),
        ProtoStorage::F16(a) => ProtosView::F16(a.view()),
    });
    DecodedOutputsRef {
        boxes,
        scores,
        mask_coefs,
        protos,
        total_anchors: plan.total_anchors,
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
        TensorDyn::I8(t)
    }

    fn mk_f32_tensor(shape: &[usize]) -> TensorDyn {
        let t = Tensor::<f32>::new(shape, Some(TensorMemory::Mem), None).unwrap();
        TensorDyn::F32(t)
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
        let dyn_t = TensorDyn::I8(t);
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
        // quantization), so the box_dispatch is selected as DflI8ToF32Scalar.
        // Feeding f32 tensors would hit the dispatch's mismatched-arm path
        // → KernelDispatchUnreachable. End-to-end run() integration is
        // exercised separately via the int8-with-attached-quant test below.
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(json).unwrap();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();

        let decoder = PerScaleDecoder::new(plan);
        // After new, buffers should be sized.
        match &decoder.buffers.boxes {
            crate::per_scale::outputs::Buffer::F32(v) => assert_eq!(v.len(), 4 * 8400),
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

        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        let schema: SchemaV2 = serde_json::from_str(json).unwrap();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();

        // Build i8 inputs for all 10 physical tensors with attached quant.
        let mut inputs_owned: Vec<TensorDyn> = Vec::new();
        for lvl in &plan.levels {
            // boxes
            let mut t = Tensor::<i8>::new(&lvl.box_shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(TensorDyn::I8(t));
            // scores
            let mut t = Tensor::<i8>::new(&lvl.score_shape, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(TensorDyn::I8(t));
            // mask_coefs
            if let Some(s) = &lvl.mc_shape {
                let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
                t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
                inputs_owned.push(TensorDyn::I8(t));
            }
        }
        // protos
        if let Some(s) = &plan.proto_shape {
            let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            inputs_owned.push(TensorDyn::I8(t));
        }
        let inputs: Vec<&TensorDyn> = inputs_owned.iter().collect();

        let mut decoder = PerScaleDecoder::new(plan);
        let result = decoder.run(&inputs);
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
}
