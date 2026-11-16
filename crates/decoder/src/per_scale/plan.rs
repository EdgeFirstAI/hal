// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale decode plan, built once at DecoderBuilder time.
//!
//! Holds what is invariant per build and not carried by the input tensors
//! at frame time. Quantization parameters are read from the `TensorDyn`;
//! the plan does not cache them. The plan stores one dispatch per role.

use super::kernels::CpuFeatures;
use super::{Activation, DecodeDtype};
use crate::configs::DimName;
use crate::schema::BoxEncoding;
use crate::schema::{last_feature_axis, LogicalOutput, LogicalType, PhysicalOutput, SchemaV2};
use crate::{DecoderError, DecoderResult};
use edgefirst_kernels::prim::dfl_lut::DflGeom;
use edgefirst_kernels::prim::grids::make_anchor_grid;

/// Memory layout of a per-scale child tensor, derived from its dshape.
///
/// The per-scale level kernels (`level_box`, `level_score`, `level_mc`)
/// operate on flat slices in NHWC memory order: each anchor's channel block
/// is contiguous (`anchor * channels` indexing). The pipeline transposes an
/// NCHW child into a scratch buffer before invoking the kernel. See
/// [`pipeline::run`] for the routing.
///
/// The position of the channel-axis dimension (one of `BoxCoords` /
/// `NumClasses` / `NumFeatures` / `NumProtos`) in the child's dshape
/// determines the layout: last position is NHWC, second position (after
/// `Batch`) is NCHW. A schema without a named channel axis is NHWC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// `[batch, height, width, channel]` — channels contiguous in memory.
    Nhwc,
    /// `[batch, channel, height, width]` — channels strided by `h*w`.
    Nchw,
}

#[derive(Debug)]
#[allow(dead_code)]
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

    /// Activation declared on score logits, uniform across levels.
    pub(crate) score_activation: Activation,

    /// Builder choice + probed CPU features, captured once.
    pub(crate) out_dtype: DecodeDtype,
    pub(crate) cpu_features: CpuFeatures,

    // Pre-selected dispatch, one per role.
    pub(crate) box_dispatch: super::kernels::dispatch::BoxLevelDispatch,
    /// The box kernel of levels whose box tensor carries per-channel quantization.
    pub(crate) box_per_channel: super::kernels::dispatch::DflPerChannelDispatch,
    pub(crate) score_dispatch: super::kernels::dispatch::ScoreLevelDispatch,
    pub(crate) mc_dispatch: Option<super::kernels::dispatch::MaskCoefDispatch>,
    pub(crate) proto_dispatch: Option<super::kernels::dispatch::ProtoDispatch>,
    /// The transpose of NCHW inputs.
    pub(crate) transpose: super::kernels::dispatch::TransposeDispatch,

    /// Expected shape of the protos tensor, used to bind it by shape match.
    /// This is the raw schema shape, not the buffer layout: for NCHW protos
    /// the buffer is allocated in NHWC order (`proto_nhwc_shape`) and the
    /// pipeline transposes at decode time.
    pub(crate) proto_shape: Option<Box<[usize]>>,

    /// NHWC-normalised proto shape for buffer allocation. Identical to
    /// `proto_shape` when the schema is NHWC; transposed from `[N,C,H,W]`
    /// to `[N,H,W,C]` when the schema declares NCHW.
    pub(crate) proto_nhwc_shape: Option<Box<[usize]>>,

    /// Memory layout of the proto tensor, detected from the schema's
    /// dshape. The pipeline transposes NCHW proto data into the NHWC buffer.
    pub(crate) proto_layout: Option<Layout>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct LevelPlan {
    /// FPN spatial stride in pixels (8.0, 16.0, 32.0 typically).
    pub(crate) stride: f32,

    /// Spatial dimensions of this level's tensors.
    pub(crate) h: usize,
    pub(crate) w: usize,

    /// DFL: number of distribution bins (typically 16). Box channels
    /// are then `4 * reg_max`.
    /// LTRB: 1 (not read by the LTRB kernel). Box channels are 4.
    pub(crate) reg_max: usize,

    /// Cumulative sum of prior levels' `h * w`: this level's offset into
    /// the flat output buffers.
    pub(crate) anchor_offset: usize,

    /// Anchor grid coordinates, computed at plan time. `len() == h * w`
    /// each, row-major.
    pub(crate) grid_x: Box<[f32]>,
    pub(crate) grid_y: Box<[f32]>,

    /// Expected tensor shapes used at frame time to bind inputs by shape
    /// match. The shape is the raw schema-declared shape; `layout` records
    /// whether it is NHWC or NCHW.
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

    /// Memory layout of this level's child tensors, taken from the box
    /// child's dshape; `try_from_schema` checks that scores and mask
    /// coefficients match. NCHW children are transposed to scratch per frame
    /// in `pipeline::run`.
    pub(crate) layout: Layout,
}

impl LevelPlan {
    /// This level's anchor geometry for the exp-table DFL decode.
    #[inline]
    pub(crate) fn dfl_geom(&self) -> DflGeom<'_> {
        DflGeom {
            reg_max: self.reg_max,
            anchors: self.h * self.w,
            grid_x: &self.grid_x,
            grid_y: &self.grid_y,
            stride: self.stride,
        }
    }
}

impl PerScalePlan {
    /// Build a per-scale plan from a schema-v2 document, if and only if
    /// the schema describes a per-scale decomposition this subsystem
    /// can decode.
    ///
    /// Returns `Ok(None)` for non-per-scale schemas. Returns an error for
    /// per-scale schemas this subsystem cannot handle (reg_max > 64,
    /// unsupported encoding, etc.).
    #[allow(dead_code)]
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

        // Only the Dfl and Direct (LTRB) encodings get a plan.
        let box_encoding = boxes.encoding.unwrap_or(BoxEncoding::Dfl);
        if !matches!(box_encoding, BoxEncoding::Dfl | BoxEncoding::Direct) {
            return Ok(None);
        }

        let levels = build_levels(boxes, scores, mc, box_encoding)?;

        // Plan-time validation.
        for lvl in &levels {
            if box_encoding == BoxEncoding::Dfl && lvl.reg_max > 64 {
                return Err(DecoderError::NotSupported(format!(
                    "DFL reg_max={} exceeds stack-scratch cap of 64",
                    lvl.reg_max
                )));
            }
        }

        let total_anchors: usize = levels.iter().map(|l| l.h * l.w).sum();
        let num_classes = scores
            .outputs
            .first()
            .and_then(last_feature_axis)
            .unwrap_or(0);
        let num_mask_coefs = mc
            .and_then(|m| m.outputs.first())
            .and_then(last_feature_axis)
            .unwrap_or(0);

        // The score activation is the `scores` logical's `activation_required`,
        // else its first physical child's.
        let score_activation = Activation::from_schema(
            scores
                .activation_required
                .or_else(|| scores.outputs.first().and_then(|s| s.activation_required)),
        );

        // Protos shape: the logical's own shape, or its first child's when it
        // has children.
        let proto_shape = protos.map(|p| {
            if p.outputs.is_empty() {
                p.shape.clone().into_boxed_slice()
            } else {
                p.outputs[0].shape.clone().into_boxed_slice()
            }
        });

        // Proto layout from the dshape, and the NHWC-normalised shape used for
        // buffer allocation.
        let proto_layout = protos.map(|p| {
            let dshape = if p.outputs.is_empty() {
                &p.dshape
            } else {
                &p.outputs[0].dshape
            };
            detect_layout_from_dshape(dshape, proto_shape.as_ref().map_or(0, |s| s.len()))
        });
        let proto_nhwc_shape = match (&proto_shape, proto_layout) {
            (Some(s), Some(Layout::Nchw)) if s.len() == 4 => {
                // [N, C, H, W] → [N, H, W, C]
                Some(vec![s[0], s[2], s[3], s[1]].into_boxed_slice())
            }
            _ => proto_shape.clone(),
        };

        let cpu_features = super::kernels::decoder_cpu_features()?;

        // The first child's dtype of each role pre-selects the dispatches.
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
        let box_per_channel =
            super::kernels::dispatch::DflPerChannelDispatch::select(out_dtype, &cpu_features);
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
            // Mask coefficients decode to f32, whatever the box and score dtype.
            Some(super::kernels::dispatch::MaskCoefDispatch::select(
                mc_dtype,
                DecodeDtype::F32,
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
                DecodeDtype::F32,
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
            box_per_channel,
            score_dispatch,
            mc_dispatch,
            proto_dispatch,
            transpose: super::kernels::dispatch::TransposeDispatch::select(&cpu_features),
            proto_shape,
            proto_nhwc_shape,
            proto_layout,
        }))
    }

    /// DFL `reg_max` for the boxes role, or `None` when boxes are not
    /// DFL-encoded. Taken from the first level.
    #[cfg(test)]
    pub(crate) fn boxes_reg_max(&self) -> Option<usize> {
        if self.box_encoding != BoxEncoding::Dfl {
            return None;
        }
        self.levels.first().map(|lvl| lvl.reg_max)
    }
}

/// Map a schema-level [`crate::schema::DType`] (`Int8`/`Float32` style
/// names) to the runtime [`edgefirst_tensor::DType`] (`I8`/`F32` style
/// names). Variants outside the dispatch matrix return `NotSupported`.
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
        S::Int64 => T::I64,
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

        // The layout comes from the box child's dshape; the score and mc
        // children must match it.
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
                // DFL channel count is the size of the box child's channel
                // axis, found by name.
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

/// Determine memory layout from a dshape slice and tensor rank.
fn detect_layout_from_dshape(dshape: &[(DimName, usize)], rank: usize) -> Layout {
    use DimName::*;
    let channel_pos = dshape
        .iter()
        .position(|(name, _)| matches!(name, BoxCoords | NumClasses | NumFeatures | NumProtos));
    match channel_pos {
        Some(idx) if idx == rank - 1 => Layout::Nhwc,
        Some(idx) if idx == 1 && rank >= 4 => Layout::Nchw,
        Some(idx) if idx == 1 && rank == 3 => Layout::Nhwc,
        _ => Layout::Nhwc,
    }
}

/// Determine the memory layout of a per-scale child from its dshape by
/// finding the position of its channel-axis dimension.
///
/// - Channel axis at position `len-1` → NHWC.
/// - Channel axis at position 1 (after batch) → NCHW.
/// - Channel axis elsewhere in 4-D → unsupported (rejected).
/// - No channel axis named at all → NHWC.
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

/// Channel-axis size for a box child, found by name.
fn box_channel_count(p: &PhysicalOutput) -> DecoderResult<usize> {
    use DimName::*;
    for (i, (name, _)) in p.dshape.iter().enumerate() {
        if matches!(name, BoxCoords | NumFeatures) {
            return Ok(p.shape[i]);
        }
    }
    // A schema with no named channel axis is taken as NHWC: the trailing dim.
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
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolov8n_schema.json");
        serde_json::from_str(&json).expect("yolov8n fixture must parse")
    }

    fn fixture_yolo26n_schema() -> SchemaV2 {
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_yolo26n_schema.json");
        serde_json::from_str(&json).expect("yolo26n fixture must parse")
    }

    fn fixture_flat_schema() -> SchemaV2 {
        let json =
            edgefirst_bench::testdata::read_to_string("per_scale/synthetic_flat_schema.json");
        serde_json::from_str(&json).expect("flat fixture must parse")
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
        // Accepts the scalar and the NEON DFL i8 → f32 variants.
        let ok = match plan.box_dispatch {
            BoxLevelDispatch::DflI8ToF32Lut => true,
            #[cfg(target_arch = "aarch64")]
            BoxLevelDispatch::DflI8ToF32LutNeon => true,
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
        // Accepts the scalar and the NEON LTRB variants.
        let ok = match plan.box_dispatch {
            BoxLevelDispatch::LtrbI8ToF32Scalar => true,
            #[cfg(target_arch = "aarch64")]
            BoxLevelDispatch::LtrbI8ToF32NeonBase => true,
            _ => false,
        };
        assert!(ok, "unexpected dispatch: {:?}", plan.box_dispatch);
    }

    #[test]
    fn nhwc_protos_layout_detected_correctly() {
        let schema = fixture_yolov8n_schema();
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        // Fixture protos: [1, 160, 160, 32] NHWC
        assert_eq!(plan.proto_layout, Some(Layout::Nhwc));
        assert_eq!(plan.proto_shape.as_deref(), Some(&[1, 160, 160, 32][..]));
        assert_eq!(plan.proto_nhwc_shape, plan.proto_shape);
    }

    #[test]
    fn nchw_protos_layout_normalize_and_transpose() {
        // Build NCHW protos by mutating the NHWC fixture
        let mut schema = fixture_yolo26n_schema();
        for out in &mut schema.outputs {
            if out.type_ == Some(LogicalType::Protos) {
                out.shape = vec![1, 32, 160, 160];
                out.dshape = vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 32),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ];
            }
        }
        let plan = PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
            .unwrap()
            .unwrap();
        assert_eq!(plan.proto_layout, Some(Layout::Nchw));
        // Binding shape stays as-is for tensor shape-match
        assert_eq!(plan.proto_shape.as_deref(), Some(&[1, 32, 160, 160][..]));
        // Buffer shape normalized to NHWC
        assert_eq!(
            plan.proto_nhwc_shape.as_deref(),
            Some(&[1, 160, 160, 32][..])
        );
    }

    /// An NCHW plan transposes with NEON on NEON tiers and with the element loop on the scalar tier.
    #[test]
    fn nchw_transpose_follows_the_forced_tier() {
        use crate::per_scale::kernels::dispatch::TransposeDispatch;
        use crate::per_scale::kernels::forced_for_test;
        let mut schema = fixture_yolo26n_schema();
        for out in &mut schema.outputs {
            if out.type_ == Some(LogicalType::Protos) {
                out.shape = vec![1, 32, 160, 160];
                out.dshape = vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 32),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ];
            }
        }
        for (name, isa) in crate::neon_dispatch::tiers() {
            let plan = forced_for_test::with(name, || {
                PerScalePlan::try_from_schema(&schema, DecodeDtype::F32)
                    .unwrap()
                    .unwrap()
            });
            assert_eq!(plan.proto_layout, Some(Layout::Nchw), "{name}");
            let neon = plan.transpose == TransposeDispatch::Neon;
            assert_eq!(neon, !isa.is_scalar(), "{name}: {:?}", plan.transpose);
        }
    }
}
