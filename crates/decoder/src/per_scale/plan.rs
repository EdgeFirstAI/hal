// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale decode plan — built once at DecoderBuilder time.
//!
//! Holds only what is invariant per build AND not already on the input
//! tensors at frame time. Quantization parameters live on the
//! `TensorDyn` itself; the plan does not cache them. Dtype is uniform
//! per role across all FPN levels, so the plan stores one dispatch per
//! role, not one per level. Dispatch fields are added in Task 15.

use super::kernels::grids::make_anchor_grid;
use super::kernels::CpuFeatures;
use super::{Activation, DecodeDtype};
use crate::configs::DimName;
use crate::schema::BoxEncoding;
use crate::schema::{LogicalOutput, LogicalType, PhysicalOutput, SchemaV2};
use crate::{DecoderError, DecoderResult};

/// Memory layout of a per-scale child tensor, derived from its dshape.
///
/// The per-scale level kernels (`level_box`, `level_score`, `level_mc`)
/// operate on flat slices that assume **NHWC** memory order — each
/// anchor's channel block is contiguous (`anchor * channels` indexing).
/// When a child arrives in **NCHW** order, the pipeline transposes the
/// tensor into a scratch buffer before invoking the kernel; the kernel
/// itself stays NHWC. See [`pipeline::run`] for the routing.
///
/// Detection is name-based: the position of the channel-axis dimension
/// (one of `BoxCoords` / `NumClasses` / `NumFeatures` / `NumProtos`) in
/// the child's dshape determines the layout — last position is NHWC,
/// second position (after `Batch`) is NCHW. Schemas without a named
/// channel axis fall through to NHWC for back-compat with the existing
/// positional-fallback contract in `extract_hw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// `[batch, height, width, channel]` — channels contiguous in memory.
    /// All HAL canonical fixtures and the TFLite-converter output use this.
    Nhwc,
    /// `[batch, channel, height, width]` — channels strided by `h*w`.
    /// Produced by the ara2-converter (NCHW NPU output) and likely by
    /// some Hailo / Neutron paths.
    Nchw,
}

#[derive(Debug)]
#[allow(dead_code)] // Consumed by Phase 1 try_from_schema + run() in later tasks.
pub(crate) struct PerScalePlan {
    /// Per-FPN-level geometry, pre-computed grids, expected shapes.
    /// Sorted stride-ascending by `try_from_schema`.
    pub(crate) levels: Vec<LevelPlan>,

    /// Sum of h*w across levels (e.g. 8400 for 640×640 yolo).
    pub(crate) total_anchors: usize,

    /// Number of classes — uniform across levels, validated at plan time.
    pub(crate) num_classes: usize,

    /// Mask-coefficient channel count. Zero if detection-only.
    pub(crate) num_mask_coefs: usize,

    /// Box encoding selects the box-level kernel orchestrator at run
    /// time (DFL: dequant → softmax → weighted-sum → dist2bbox;
    /// LTRB / Direct: dequant → dist2bbox).
    pub(crate) box_encoding: BoxEncoding,

    /// Activation declared on score logits. Uniform across levels in
    /// every observed model; if a future schema breaks this we move it
    /// back into LevelPlan.
    pub(crate) score_activation: Activation,

    /// Builder choice + probed CPU features, captured once.
    pub(crate) out_dtype: DecodeDtype,
    pub(crate) cpu_features: CpuFeatures,

    // Pre-selected dispatch — one per role. Reused across all levels
    // because dtype is uniform per role.
    pub(crate) box_dispatch: super::kernels::dispatch::BoxLevelDispatch,
    pub(crate) score_dispatch: super::kernels::dispatch::ScoreLevelDispatch,
    pub(crate) mc_dispatch: Option<super::kernels::dispatch::MaskCoefDispatch>,
    pub(crate) proto_dispatch: Option<super::kernels::dispatch::ProtoDispatch>,

    /// Protos: a single tensor (no per-scale structure). Only the
    /// expected shape is retained — for binding by shape match. The
    /// dtype is captured implicitly in the proto dispatch's variant
    /// (added Task 15); quant comes live from the bound tensor.
    pub(crate) proto_shape: Option<Box<[usize]>>,
}

#[derive(Debug)]
#[allow(dead_code)] // Consumed by Phase 1 try_from_schema + level kernels.
pub(crate) struct LevelPlan {
    /// FPN spatial stride in pixels (8.0, 16.0, 32.0 typically).
    pub(crate) stride: f32,

    /// Spatial dimensions of this level's tensors.
    pub(crate) h: usize,
    pub(crate) w: usize,

    /// DFL: number of distribution bins (typically 16). Box channels
    /// are then `4 * reg_max`.
    /// LTRB: 1 (degenerate; not consulted by the LTRB kernel). Box
    /// channels are 4.
    pub(crate) reg_max: usize,

    /// Cumulative sum of prior levels' `h * w`. Used to address into
    /// flat output buffers without recomputing per frame.
    pub(crate) anchor_offset: usize,

    /// Pre-computed at plan time (saves per-frame allocation).
    /// `len() == h * w` each, row-major.
    pub(crate) grid_x: Box<[f32]>,
    pub(crate) grid_y: Box<[f32]>,

    /// Expected tensor shapes used at frame time to bind inputs via
    /// shape-match (frames may arrive in any order). The shape is the
    /// raw schema-declared shape; whether it represents NHWC or NCHW is
    /// recorded in `layout`. The level kernel itself is NHWC-only — the
    /// pipeline transposes NCHW children into a scratch buffer before
    /// dispatch.
    ///
    /// DFL boxes  (NHWC): `[1, h, w, 4 * reg_max]`.
    /// DFL boxes  (NCHW): `[1, 4 * reg_max, h, w]`.
    /// LTRB boxes (NHWC): `[1, h, w, 4]`.
    /// LTRB boxes (NCHW): `[1, 4, h, w]`.
    pub(crate) box_shape: Box<[usize]>,

    /// Score tensor shape: `[1, h, w, num_classes]` (NHWC) or
    /// `[1, num_classes, h, w]` (NCHW).
    pub(crate) score_shape: Box<[usize]>,

    /// Mask-coef tensor shape: `[1, h, w, num_mask_coefs]` (NHWC) or
    /// `[1, num_mask_coefs, h, w]` (NCHW). None if detection-only model.
    pub(crate) mc_shape: Option<Box<[usize]>>,

    /// Memory layout of this level's child tensors. Determined at plan
    /// time from the box child's dshape (scores and mc are validated to
    /// match in `try_from_schema`). NHWC is the canonical case;
    /// NCHW triggers a per-frame transpose to scratch in `pipeline::run`.
    pub(crate) layout: Layout,
}

impl PerScalePlan {
    /// Build a per-scale plan from a schema-v2 document, if and only if
    /// the schema describes a per-scale decomposition this subsystem
    /// can decode.
    ///
    /// Returns `Ok(None)` for non-per-scale schemas (the caller falls
    /// through to the legacy decode path). Returns an error for
    /// per-scale schemas this subsystem cannot handle (per-channel
    /// quant, reg_max > 64, unsupported encoding, etc.).
    #[allow(dead_code)] // Wired into DecoderBuilder in Task 16.
    pub(crate) fn try_from_schema(
        schema: &SchemaV2,
        out_dtype: DecodeDtype,
    ) -> DecoderResult<Option<Self>> {
        let boxes = match find_logical_by_type(schema, LogicalType::Boxes) {
            Some(b) if !b.outputs.is_empty() => b,
            _ => return Ok(None),
        };

        let scores = match find_logical_by_type(schema, LogicalType::Scores) {
            Some(s) if !s.outputs.is_empty() => s,
            _ => return Ok(None),
        };

        let mc = find_logical_by_type(schema, LogicalType::MaskCoefs);
        let protos = find_logical_by_type(schema, LogicalType::Protos);

        // Encoding sanity (only Dfl + Direct/Ltrb are in scope for Phase 1).
        let box_encoding = boxes.encoding.unwrap_or(BoxEncoding::Dfl);
        if !matches!(box_encoding, BoxEncoding::Dfl | BoxEncoding::Direct) {
            return Ok(None); // Anchor-encoded falls through to legacy.
        }

        let levels = build_levels(boxes, scores, mc, box_encoding)?;

        // Plan-time validation.
        for lvl in &levels {
            if box_encoding == BoxEncoding::Dfl && lvl.reg_max > 64 {
                return Err(DecoderError::NotSupported(format!(
                    "DFL reg_max={} exceeds Phase 1 stack-scratch cap of 64",
                    lvl.reg_max
                )));
            }
        }

        let total_anchors: usize = levels.iter().map(|l| l.h * l.w).sum();
        let num_classes = scores
            .outputs
            .first()
            .and_then(|s| s.shape.last())
            .copied()
            .unwrap_or(0);
        let num_mask_coefs = mc
            .and_then(|m| m.outputs.first())
            .and_then(|c| c.shape.last())
            .copied()
            .unwrap_or(0);

        // Activation declaration site, in priority order:
        //   1. `scores` logical parent's `activation_required` — this is
        //      where the EdgeFirst tflite-converter actually writes the
        //      sigmoid annotation when stripping the score activation
        //      from the per-scale graph (one entry, not duplicated per
        //      physical child).
        //   2. The first physical child's `activation_required` — kept as
        //      a fallback for synthetic test fixtures and any future
        //      converter that puts the annotation on the children.
        // Per-scale physical children always inherit the parent's
        // declaration; mixing per-child variants isn't a valid
        // configuration.
        let score_activation = Activation::from_schema(
            scores
                .activation_required
                .or_else(|| scores.outputs.first().and_then(|s| s.activation_required)),
        );

        // Protos shape: prefer the logical's own shape; if it has children,
        // fall back to the first child's shape.
        let proto_shape = protos.map(|p| {
            if p.outputs.is_empty() {
                p.shape.clone().into_boxed_slice()
            } else {
                p.outputs[0].shape.clone().into_boxed_slice()
            }
        });

        let cpu_features = CpuFeatures::from_env_or_probe()?;

        // All children of a given role share dtype (by construction in the
        // TFLite-converter). Read the first child to pre-select dispatches.
        let box_dtype = boxes
            .outputs
            .first()
            .and_then(|b| b.quantization.as_ref())
            .and_then(|q| q.dtype)
            .ok_or_else(|| {
                DecoderError::InvalidConfig(
                    "per-scale boxes child missing quantization dtype".into(),
                )
            })?;
        let box_dtype = schema_dtype_to_tensor_dtype(box_dtype)?;

        let score_dtype = scores
            .outputs
            .first()
            .and_then(|s| s.quantization.as_ref())
            .and_then(|q| q.dtype)
            .ok_or_else(|| {
                DecoderError::InvalidConfig(
                    "per-scale scores child missing quantization dtype".into(),
                )
            })?;
        let score_dtype = schema_dtype_to_tensor_dtype(score_dtype)?;

        let box_dispatch = super::kernels::dispatch::BoxLevelDispatch::select(
            box_encoding,
            box_dtype,
            out_dtype,
            &cpu_features,
        )?;
        let score_dispatch = super::kernels::dispatch::ScoreLevelDispatch::select(
            score_dtype,
            out_dtype,
            &cpu_features,
        )?;

        let mc_dispatch = if let Some(m) = mc {
            let mc_dtype = m
                .outputs
                .first()
                .and_then(|c| c.quantization.as_ref())
                .and_then(|q| q.dtype)
                .ok_or_else(|| {
                    DecoderError::InvalidConfig(
                        "per-scale mask_coefs child missing quantization dtype".into(),
                    )
                })?;
            let mc_dtype = schema_dtype_to_tensor_dtype(mc_dtype)?;
            Some(super::kernels::dispatch::MaskCoefDispatch::select(
                mc_dtype,
                out_dtype,
                &cpu_features,
            )?)
        } else {
            None
        };

        let proto_dispatch = if let Some(p) = protos {
            // Protos quant lives on the logical itself when no children, on
            // the first child if it has children.
            let proto_dtype = if p.outputs.is_empty() {
                p.quantization.as_ref().and_then(|q| q.dtype)
            } else {
                p.outputs
                    .first()
                    .and_then(|c| c.quantization.as_ref())
                    .and_then(|q| q.dtype)
            }
            .ok_or_else(|| {
                DecoderError::InvalidConfig("per-scale protos missing quantization dtype".into())
            })?;
            let proto_dtype = schema_dtype_to_tensor_dtype(proto_dtype)?;
            Some(super::kernels::dispatch::ProtoDispatch::select(
                proto_dtype,
                out_dtype,
                &cpu_features,
            )?)
        } else {
            None
        };

        Ok(Some(Self {
            levels,
            total_anchors,
            num_classes,
            num_mask_coefs,
            box_encoding,
            score_activation,
            out_dtype,
            cpu_features,
            box_dispatch,
            score_dispatch,
            mc_dispatch,
            proto_dispatch,
            proto_shape,
        }))
    }
}

/// Map a schema-level [`crate::schema::DType`] (which uses `Int8`/`Float32`
/// style names) to the runtime [`edgefirst_tensor::DType`] (which uses
/// `I8`/`F32` style names). Phase 1 supports the integer + float types
/// covered by the dispatch matrix; other variants surface as
/// `NotSupported`.
fn schema_dtype_to_tensor_dtype(d: crate::schema::DType) -> DecoderResult<edgefirst_tensor::DType> {
    use crate::schema::DType as S;
    use edgefirst_tensor::DType as T;
    Ok(match d {
        S::Int8 => T::I8,
        S::Uint8 => T::U8,
        S::Int16 => T::I16,
        S::Uint16 => T::U16,
        S::Int32 => T::I32,
        S::Uint32 => T::U32,
        S::Float16 => T::F16,
        S::Float32 => T::F32,
    })
}

fn find_logical_by_type(schema: &SchemaV2, t: LogicalType) -> Option<&LogicalOutput> {
    schema.outputs.iter().find(|l| l.type_ == Some(t))
}

fn build_levels(
    boxes: &LogicalOutput,
    scores: &LogicalOutput,
    mc: Option<&LogicalOutput>,
    encoding: BoxEncoding,
) -> DecoderResult<Vec<LevelPlan>> {
    use std::collections::BTreeMap;

    type ChildTriple<'a> = (
        Option<&'a PhysicalOutput>,
        Option<&'a PhysicalOutput>,
        Option<&'a PhysicalOutput>,
    );
    let mut by_stride: BTreeMap<u32, ChildTriple<'_>> = BTreeMap::new();

    for child in &boxes.outputs {
        let s = child.stride.map(|s| s.x()).ok_or_else(|| {
            DecoderError::InvalidConfig(format!("box child {:?} missing stride", child.name))
        })?;
        by_stride.entry(s).or_insert((None, None, None)).0 = Some(child);
    }
    for child in &scores.outputs {
        let s = child.stride.map(|s| s.x()).ok_or_else(|| {
            DecoderError::InvalidConfig(format!("score child {:?} missing stride", child.name))
        })?;
        by_stride.entry(s).or_insert((None, None, None)).1 = Some(child);
    }
    if let Some(mc) = mc {
        for child in &mc.outputs {
            let s = child.stride.map(|s| s.x()).ok_or_else(|| {
                DecoderError::InvalidConfig(format!("mc child {:?} missing stride", child.name))
            })?;
            by_stride.entry(s).or_insert((None, None, None)).2 = Some(child);
        }
    }

    let mut levels = Vec::with_capacity(by_stride.len());
    let mut anchor_offset = 0;
    for (stride, (b, s, m)) in by_stride {
        let b = b.ok_or_else(|| {
            DecoderError::InvalidConfig(format!("stride {stride}: missing box child"))
        })?;
        let s = s.ok_or_else(|| {
            DecoderError::InvalidConfig(format!("stride {stride}: missing score child"))
        })?;
        let (h, w) = extract_hw(b)?;

        // Layout is derived from the box child's dshape and is required
        // to match across the level's box / score / mc children. We
        // validate score and mc match boxes here; mismatched layouts are
        // rejected at plan time (an unsupported schema variant).
        let layout = detect_layout(b)?;
        let s_layout = detect_layout(s)?;
        if s_layout != layout {
            return Err(DecoderError::InvalidConfig(format!(
                "stride {stride}: score child layout ({s_layout:?}) differs from box \
                 child layout ({layout:?}); per-level mixed layouts are not supported"
            )));
        }
        if let Some(mc_child) = m {
            let mc_layout = detect_layout(mc_child)?;
            if mc_layout != layout {
                return Err(DecoderError::InvalidConfig(format!(
                    "stride {stride}: mask_coefs child layout ({mc_layout:?}) differs \
                     from box child layout ({layout:?}); per-level mixed layouts are \
                     not supported"
                )));
            }
        }

        let reg_max = match encoding {
            BoxEncoding::Dfl => {
                // DFL channel count is the box child's channel-axis size,
                // NOT positional `shape.last()`. Channel-axis position
                // varies with layout: NHWC last, NCHW second.
                let feat = box_channel_count(b)?;
                if feat == 0 || feat % 4 != 0 {
                    return Err(DecoderError::InvalidConfig(format!(
                        "DFL box feature count {feat} is not a positive multiple of 4"
                    )));
                }
                feat / 4
            }
            BoxEncoding::Direct => 1, // LTRB: 4 channels, no bins
            BoxEncoding::Anchor => unreachable!("filtered above"),
        };
        let (gx, gy) = make_anchor_grid(h, w);
        levels.push(LevelPlan {
            stride: stride as f32,
            h,
            w,
            reg_max,
            anchor_offset,
            grid_x: gx,
            grid_y: gy,
            box_shape: b.shape.clone().into_boxed_slice(),
            score_shape: s.shape.clone().into_boxed_slice(),
            mc_shape: m.map(|c| c.shape.clone().into_boxed_slice()),
            layout,
        });
        anchor_offset += h * w;
    }
    Ok(levels)
}

/// Determine the memory layout of a per-scale child from its dshape by
/// finding the position of its channel-axis dimension.
///
/// - Channel axis at position `len-1` → NHWC.
/// - Channel axis at position 1 (after batch) → NCHW.
/// - Channel axis elsewhere in 4-D → unsupported (rejected).
/// - No channel axis named at all → NHWC default (matches the existing
///   positional fallback in `extract_hw`).
///
/// Channel-axis dim names: `BoxCoords`, `NumClasses`, `NumFeatures`,
/// `NumProtos`.
fn detect_layout(p: &PhysicalOutput) -> DecoderResult<Layout> {
    use DimName::*;
    let channel_pos = p
        .dshape
        .iter()
        .position(|(name, _)| matches!(name, BoxCoords | NumClasses | NumFeatures | NumProtos));
    let rank = p.shape.len();
    match channel_pos {
        Some(idx) if idx == rank - 1 => Ok(Layout::Nhwc),
        Some(idx) if idx == 1 && rank >= 4 => Ok(Layout::Nchw),
        Some(idx) if idx == 1 && rank == 3 => Ok(Layout::Nhwc),
        Some(idx) => Err(DecoderError::InvalidConfig(format!(
            "child {:?}: channel axis at position {idx} of {rank}-D dshape; \
             expected last position (NHWC) or position 1 (NCHW for 4-D)",
            p.name
        ))),
        None => Ok(Layout::Nhwc),
    }
}

/// Channel-axis size for a box child, found by name (not position).
/// Used by the DFL `reg_max` derivation, which must work for both NHWC
/// and NCHW children.
fn box_channel_count(p: &PhysicalOutput) -> DecoderResult<usize> {
    use DimName::*;
    for (i, (name, _)) in p.dshape.iter().enumerate() {
        if matches!(name, BoxCoords | NumFeatures) {
            return Ok(p.shape[i]);
        }
    }
    // Back-compat fallback for schemas with no named channel axis: assume
    // NHWC and use the trailing dim (matches the prior unconditional
    // `shape.last()` behavior).
    p.shape
        .last()
        .copied()
        .ok_or_else(|| DecoderError::InvalidConfig(format!("box child {:?}: empty shape", p.name)))
}

fn extract_hw(p: &PhysicalOutput) -> DecoderResult<(usize, usize)> {
    let mut h = None;
    let mut w = None;
    for (i, (name, _)) in p.dshape.iter().enumerate() {
        match name {
            DimName::Height => h = Some(p.shape[i]),
            DimName::Width => w = Some(p.shape[i]),
            _ => {}
        }
    }
    if h.is_none() && w.is_none() && p.shape.len() == 4 {
        // NHWC fallback (shape [N, H, W, C]).
        h = Some(p.shape[1]);
        w = Some(p.shape[2]);
    }
    Ok((
        h.ok_or_else(|| {
            DecoderError::InvalidConfig(format!("child {:?}: missing height", p.name))
        })?,
        w.ok_or_else(|| DecoderError::InvalidConfig(format!("child {:?}: missing width", p.name)))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_yolov8n_schema() -> SchemaV2 {
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolov8n_schema.json");
        serde_json::from_str(json).expect("yolov8n fixture must parse")
    }

    fn fixture_yolo26n_schema() -> SchemaV2 {
        let json = include_str!("../../../../testdata/per_scale/synthetic_yolo26n_schema.json");
        serde_json::from_str(json).expect("yolo26n fixture must parse")
    }

    fn fixture_flat_schema() -> SchemaV2 {
        let json = include_str!("../../../../testdata/per_scale/synthetic_flat_schema.json");
        serde_json::from_str(json).expect("flat fixture must parse")
    }

    #[test]
    fn try_from_schema_yolov8n_returns_some() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .expect("should plan successfully")
            .expect("yolov8n schema is per-scale");
        assert_eq!(plan.levels.len(), 3);
        assert_eq!(plan.total_anchors, 80 * 80 + 40 * 40 + 20 * 20);
        assert_eq!(plan.num_classes, 80);
        assert_eq!(plan.num_mask_coefs, 32);
        assert_eq!(plan.box_encoding, BoxEncoding::Dfl);
        assert_eq!(plan.score_activation, Activation::Sigmoid);
        assert!(plan.proto_shape.is_some());
    }

    #[test]
    fn try_from_schema_yolo26n_returns_some_with_ltrb_encoding() {
        let schema = fixture_yolo26n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .expect("should plan successfully")
            .expect("yolo26n schema is per-scale");
        assert_eq!(plan.levels.len(), 3);
        assert_eq!(plan.box_encoding, BoxEncoding::Direct);
        // LTRB box channels = 4, so reg_max conceptually = 1 (degenerate;
        // not consulted by the LTRB kernel, but must be set somewhere finite).
        for lvl in &plan.levels {
            assert_eq!(lvl.box_shape.last().copied(), Some(4));
        }
    }

    #[test]
    fn try_from_schema_returns_none_for_flat_schema() {
        let schema = fixture_flat_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32).unwrap();
        assert!(plan.is_none(), "flat schema should fall through to legacy");
    }

    #[test]
    fn try_from_schema_strides_sorted_ascending() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        let strides: Vec<f32> = plan.levels.iter().map(|l| l.stride).collect();
        assert_eq!(strides, vec![8.0, 16.0, 32.0]);
    }

    #[test]
    fn try_from_schema_anchor_offsets_cumulative() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        assert_eq!(plan.levels[0].anchor_offset, 0);
        assert_eq!(plan.levels[1].anchor_offset, 80 * 80);
        assert_eq!(plan.levels[2].anchor_offset, 80 * 80 + 40 * 40);
    }

    #[test]
    fn try_from_schema_grids_pre_computed() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        for lvl in &plan.levels {
            assert_eq!(lvl.grid_x.len(), lvl.h * lvl.w);
            assert_eq!(lvl.grid_y.len(), lvl.h * lvl.w);
        }
    }

    #[test]
    fn try_from_schema_dfl_reg_max_is_16() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        for lvl in &plan.levels {
            assert_eq!(lvl.reg_max, 16);
        }
    }

    #[test]
    fn try_from_schema_yolov8n_selects_dfl_i8_to_f32_dispatch() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        use crate::per_scale::kernels::dispatch::BoxLevelDispatch;
        // Tier-aware: scalar host picks `DflI8ToF32Scalar`, aarch64
        // hosts pick `DflI8ToF32NeonBase` (or `*NeonFp16` on A55+).
        // The test asserts shape (DFL i8 → f32), not the SIMD tier.
        // NEON variants are aarch64-only; gate the arms accordingly.
        let ok = match plan.box_dispatch {
            BoxLevelDispatch::DflI8ToF32Scalar => true,
            #[cfg(target_arch = "aarch64")]
            BoxLevelDispatch::DflI8ToF32NeonBase | BoxLevelDispatch::DflI8ToF32NeonFp16 => true,
            _ => false,
        };
        assert!(ok, "unexpected dispatch: {:?}", plan.box_dispatch);
        assert!(plan.mc_dispatch.is_some());
        assert!(plan.proto_dispatch.is_some());
    }

    #[test]
    fn try_from_schema_yolo26n_selects_ltrb_dispatch() {
        let schema = fixture_yolo26n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        use crate::per_scale::kernels::dispatch::BoxLevelDispatch;
        // Tier-aware: see sibling test above. NEON variants are
        // aarch64-only; LTRB has no FP16 variant by design.
        let ok = match plan.box_dispatch {
            BoxLevelDispatch::LtrbI8ToF32Scalar => true,
            #[cfg(target_arch = "aarch64")]
            BoxLevelDispatch::LtrbI8ToF32NeonBase => true,
            _ => false,
        };
        assert!(ok, "unexpected dispatch: {:?}", plan.box_dispatch);
    }
}
