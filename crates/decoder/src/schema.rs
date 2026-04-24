// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Schema v2 metadata types for EdgeFirst model configuration.
//!
//! Schema v2 introduces a two-layer output model that separates the logical
//! contract (what the model produces semantically) from the physical
//! realization (what tensors the converter emitted). Each entry in the
//! top-level [`SchemaV2::outputs`] array is a [`LogicalOutput`]. A logical
//! output either IS a physical tensor (when the converter did not split it
//! further) or contains a `outputs` array of [`PhysicalOutput`] children
//! that realize it.
//!
//! # Example — YOLOv8 detection, flat (TFLite)
//!
//! ```
//! use edgefirst_decoder::schema::SchemaV2;
//!
//! let json = r#"{
//!   "schema_version": 2,
//!   "outputs": [
//!     {
//!       "name": "boxes", "type": "boxes",
//!       "shape": [1, 64, 8400],
//!       "dshape": [{"batch": 1}, {"num_features": 64}, {"num_boxes": 8400}],
//!       "encoding": "dfl", "decoder": "ultralytics", "normalized": true,
//!       "dtype": "int8",
//!       "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"}
//!     },
//!     {
//!       "name": "scores", "type": "scores",
//!       "shape": [1, 80, 8400],
//!       "dshape": [{"batch": 1}, {"num_classes": 80}, {"num_boxes": 8400}],
//!       "decoder": "ultralytics", "score_format": "per_class",
//!       "dtype": "int8",
//!       "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"}
//!     }
//!   ]
//! }"#;
//!
//! let schema: SchemaV2 = serde_json::from_str(json).unwrap();
//! assert_eq!(schema.schema_version, 2);
//! assert_eq!(schema.outputs.len(), 2);
//! ```

use crate::configs::{self, deserialize_dshape, DimName, QuantTuple};
use crate::{ConfigOutput, ConfigOutputs, DecoderError, DecoderResult};
use serde::{Deserialize, Serialize};

/// Highest `schema_version` this parser accepts. Files with a higher
/// version are rejected rather than silently parsed against the wrong
/// grammar.
pub const MAX_SUPPORTED_SCHEMA_VERSION: u32 = 2;

/// Root of the edgefirst.json schema v2 metadata.
///
/// All fields except [`SchemaV2::schema_version`] are optional, so
/// third-party integrations can include only the sections relevant to
/// their use case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaV2 {
    /// Schema version. Always 2 for v2 metadata.
    pub schema_version: u32,

    /// Input tensor specification (shape, named dims, camera adaptor).
    ///
    /// Required for decoders that need to know the model input resolution:
    /// DFL dist2bbox scaling, box normalization against input dimensions,
    /// and per-scale anchor grid generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<InputSpec>,

    /// Logical outputs describing the model's output tensors and their
    /// semantic roles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<LogicalOutput>,

    /// HAL NMS mode. Omitted for end-to-end models with embedded NMS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nms: Option<NmsMode>,

    /// YOLO architecture version for Ultralytics decoders.
    ///
    /// Values: `yolov5`, `yolov8`, `yolo11`, `yolo26`. `yolo26` is
    /// end-to-end (embedded NMS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<DecoderVersion>,
}

impl Default for SchemaV2 {
    fn default() -> Self {
        Self {
            schema_version: 2,
            input: None,
            outputs: Vec::new(),
            nms: None,
            decoder_version: None,
        }
    }
}

/// Input tensor specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputSpec {
    /// Input tensor shape in the model's native layout (NCHW or NHWC).
    pub shape: Vec<usize>,

    /// Named dimensions ordered to match `shape`. Empty means the layout
    /// is unspecified and consumers must infer from the format.
    #[serde(
        default,
        deserialize_with = "deserialize_dshape",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub dshape: Vec<(DimName, usize)>,

    /// Camera adaptor input format (`rgb`, `rgba`, `bgr`, `bgra`, `grey`,
    /// `yuyv`). Free-form string rather than an enum because new adaptor
    /// formats can appear without breaking parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cameraadaptor: Option<String>,
}

/// Logical output: the semantic contract the model exposes.
///
/// When `outputs` is empty, the logical output IS the physical tensor
/// (`dtype` and `quantization` carry the tensor-level fields directly).
/// When `outputs` contains one or more [`PhysicalOutput`] entries, those
/// children are the real physical tensors and the logical `shape` is
/// the reconstructed shape produced by the fallback merge path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogicalOutput {
    /// Logical output name (optional at the logical level).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Semantic type.
    #[serde(rename = "type")]
    pub type_: LogicalType,

    /// Reconstructed logical shape (what the fallback dequant+merge path
    /// produces).
    pub shape: Vec<usize>,

    /// Named dimensions ordered to match `shape`.
    #[serde(
        default,
        deserialize_with = "deserialize_dshape",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub dshape: Vec<(DimName, usize)>,

    /// Decoder to use for post-processing. Omitted for outputs consumed
    /// directly (e.g. `protos`) where no decode step is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder: Option<DecoderKind>,

    /// Box encoding. Required on `boxes` logical outputs in v2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<BoxEncoding>,

    /// Score format. Scores only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_format: Option<ScoreFormat>,

    /// Coordinate format. `true` means `[0, 1]` normalized; `false` means
    /// pixel coordinates relative to the letterboxed model input. `None`
    /// means unspecified (decoder must infer). `boxes` and `detections`
    /// only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized: Option<bool>,

    /// Anchor boxes for anchor-encoded logical outputs. Required when
    /// `encoding: anchor`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchors: Option<Vec<[f32; 2]>>,

    /// Spatial stride. For non-split logical outputs this is a spatial
    /// hint (e.g. `protos` at stride 4). For per-scale splits each child
    /// carries its own `stride` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stride: Option<Stride>,

    /// Tensor dtype. Present when `outputs` is empty (this logical IS the
    /// physical tensor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<DType>,

    /// Quantization parameters. Present when `outputs` is empty. `None`
    /// means the tensor is not quantized (float model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,

    /// Physical children that realize this logical output. Empty when the
    /// logical IS the physical tensor. At most one level of nesting is
    /// permitted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<PhysicalOutput>,
}

impl LogicalOutput {
    /// Returns `true` if this logical output has been split into physical
    /// children by the converter.
    pub fn is_split(&self) -> bool {
        !self.outputs.is_empty()
    }
}

/// Physical output: a concrete tensor produced by the converter.
///
/// Physical outputs carry only tensor-level fields (`dtype`,
/// `quantization`, `stride`, `activation_applied`/`activation_required`).
/// Semantic fields live on the [`LogicalOutput`] parent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalOutput {
    /// Physical tensor name as produced by the converter. This name is
    /// used to bind the metadata to the tensor returned by the inference
    /// runtime.
    pub name: String,

    /// Semantic type. Matches the parent's type or declares a sub-split
    /// such as `boxes_xy` or `boxes_wh`.
    #[serde(rename = "type")]
    pub type_: PhysicalType,

    /// Physical tensor shape.
    pub shape: Vec<usize>,

    /// Named dimensions ordered to match `shape`. Disambiguates NHWC vs
    /// NCHW per-child rather than assuming a model-wide layout.
    #[serde(
        default,
        deserialize_with = "deserialize_dshape",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub dshape: Vec<(DimName, usize)>,

    /// Quantized data type.
    pub dtype: DType,

    /// Quantization parameters. Always present in v2; `null` means float
    /// (no quantization).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,

    /// FPN stride. Present on per-scale splits; absent on channel
    /// sub-splits (e.g. `boxes_xy`/`boxes_wh`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stride: Option<Stride>,

    /// Zero-based index into the parent's strides array. Used for
    /// parallel iteration with precomputed per-scale state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_index: Option<usize>,

    /// Activation already applied by the NPU. The HAL must NOT re-apply
    /// an activation declared here (e.g. Hailo applies sigmoid to score
    /// tensors on-chip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_applied: Option<Activation>,

    /// Activation NOT yet applied. The HAL MUST apply the declared
    /// activation before consuming the tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_required: Option<Activation>,
}

/// Quantization parameters for a quantized tensor.
///
/// Supports per-tensor (scalar `scale`) and per-channel (array `scale`)
/// quantization. Symmetric quantization is indicated by an absent or
/// all-zero `zero_point`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    /// Scale factor(s). One element for per-tensor quantization, or one
    /// element per slice for per-channel quantization.
    #[serde(deserialize_with = "deserialize_scalar_or_vec_f32")]
    pub scale: Vec<f32>,

    /// Zero point offset(s). Omit or set to all-zero for symmetric
    /// quantization. For per-channel quantization, length must match
    /// `scale` length.
    #[serde(
        default,
        deserialize_with = "deserialize_opt_scalar_or_vec_i32",
        skip_serializing_if = "Option::is_none"
    )]
    pub zero_point: Option<Vec<i32>>,

    /// Tensor dimension index that `scale`/`zero_point` arrays correspond
    /// to. Required when per-channel; ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<usize>,

    /// Quantized data type. Required on v2 metadata files; may be absent
    /// on programmatically-constructed configurations where the dtype is
    /// resolved at decode time from the actual tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<DType>,
}

impl Quantization {
    /// Returns `true` when per-tensor (scalar scale).
    pub fn is_per_tensor(&self) -> bool {
        self.scale.len() == 1
    }

    /// Returns `true` when per-channel (array scale of length > 1).
    pub fn is_per_channel(&self) -> bool {
        self.scale.len() > 1
    }

    /// Returns `true` when all zero points are 0 (or absent).
    pub fn is_symmetric(&self) -> bool {
        match &self.zero_point {
            None => true,
            Some(zps) => zps.iter().all(|&z| z == 0),
        }
    }

    /// Returns the zero point for the given channel index, or 0 when the
    /// quantization is symmetric.
    pub fn zero_point_at(&self, channel: usize) -> i32 {
        match &self.zero_point {
            None => 0,
            Some(zps) if zps.len() == 1 => zps[0],
            Some(zps) => zps.get(channel).copied().unwrap_or(0),
        }
    }

    /// Returns the scale for the given channel index.
    pub fn scale_at(&self, channel: usize) -> f32 {
        if self.scale.len() == 1 {
            self.scale[0]
        } else {
            self.scale.get(channel).copied().unwrap_or(0.0)
        }
    }
}

/// Accept a scalar or a JSON array when deserializing a `Vec<f32>`.
fn deserialize_scalar_or_vec_f32<'de, D>(de: D) -> Result<Vec<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(f32),
        Many(Vec<f32>),
    }
    match OneOrMany::deserialize(de)? {
        OneOrMany::One(v) => Ok(vec![v]),
        OneOrMany::Many(vs) => Ok(vs),
    }
}

/// Accept a scalar or a JSON array when deserializing an `Option<Vec<i32>>`.
fn deserialize_opt_scalar_or_vec_i32<'de, D>(de: D) -> Result<Option<Vec<i32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(i32),
        Many(Vec<i32>),
    }
    match Option::<OneOrMany>::deserialize(de)? {
        None => Ok(None),
        Some(OneOrMany::One(v)) => Ok(Some(vec![v])),
        Some(OneOrMany::Many(vs)) => Ok(Some(vs)),
    }
}

/// FPN stride. `Square(s)` means `(s, s)`; `Rect(sx, sy)` supports
/// non-square inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stride {
    Square(u32),
    Rect([u32; 2]),
}

impl Stride {
    /// Horizontal stride.
    pub fn x(self) -> u32 {
        match self {
            Stride::Square(s) => s,
            Stride::Rect([sx, _]) => sx,
        }
    }

    /// Vertical stride.
    pub fn y(self) -> u32 {
        match self {
            Stride::Square(s) => s,
            Stride::Rect([_, sy]) => sy,
        }
    }
}

/// Semantic type of a logical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalType {
    /// Bounding box coordinates.
    Boxes,
    /// Per-class or class-aggregate scores.
    Scores,
    /// Objectness scores (YOLOv5-style `obj_x_class`).
    Objectness,
    /// End-to-end class indices.
    Classes,
    /// Mask coefficients for instance segmentation.
    MaskCoefs,
    /// Instance segmentation prototypes.
    Protos,
    /// Facial / keypoint landmarks.
    Landmarks,
    /// Fully decoded post-NMS detections (end-to-end models).
    Detections,
    /// Semantic segmentation output (ModelPack).
    Segmentation,
    /// Semantic segmentation masks (ModelPack).
    Masks,
    /// ModelPack anchor-grid raw output requiring anchor decode.
    Detection,
}

/// Semantic type of a physical output.
///
/// Physical outputs either share their parent's type (per-scale splits
/// carry the parent's name) or declare a channel sub-split such as
/// `boxes_xy` / `boxes_wh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalType {
    Boxes,
    Scores,
    Objectness,
    Classes,
    MaskCoefs,
    Protos,
    Landmarks,
    Detections,
    Segmentation,
    Masks,
    Detection,
    /// ARA-2 xy channel sub-split.
    BoxesXy,
    /// ARA-2 wh channel sub-split.
    BoxesWh,
}

/// Box encoding for `boxes` logical outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoxEncoding {
    /// Distribution Focal Loss: `reg_max × 4` channels, softmax +
    /// weighted sum recovers 4 coordinates (YOLOv8, YOLO11).
    Dfl,
    /// Direct 4-channel coordinates, already decoded (YOLO26,
    /// ARA-2 post-split).
    Direct,
    /// Anchor-based grid offsets with sigmoid + anchor-scale transform
    /// per cell (YOLOv5, SSD MobileNet, ModelPack).
    Anchor,
}

/// Score format for `scores` logical outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreFormat {
    /// Each anchor outputs `[nc]` class probabilities directly
    /// (YOLOv8, YOLO11, YOLO26).
    PerClass,
    /// Each anchor outputs `[nc]` class probabilities; final confidence
    /// is `objectness × class_score` via a separate `objectness` logical
    /// output (YOLOv5).
    ObjXClass,
}

/// Activation function applied to or required by a physical tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Sigmoid,
    Softmax,
    Tanh,
}

/// Decoder framework for a logical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecoderKind {
    /// Au-Zone ModelPack anchor-based YOLO decoder.
    #[serde(rename = "modelpack")]
    ModelPack,
    /// Ultralytics anchor-free DFL decoder (YOLOv5/v8/v11/v26).
    #[serde(rename = "ultralytics")]
    Ultralytics,
}

/// YOLO architecture version for Ultralytics decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderVersion {
    Yolov5,
    Yolov8,
    Yolo11,
    Yolo26,
}

impl DecoderVersion {
    /// Returns `true` for architectures with embedded NMS (YOLO26).
    pub fn is_end_to_end(self) -> bool {
        matches!(self, DecoderVersion::Yolo26)
    }
}

/// HAL NMS mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NmsMode {
    /// Suppress overlapping boxes regardless of class label.
    ClassAgnostic,
    /// Only suppress boxes sharing a class label and overlapping above
    /// the IoU threshold.
    ClassAware,
}

/// Quantized or floating-point data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float16,
    Float32,
}

impl DType {
    /// Returns the tensor's byte width per element.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::Int8 | DType::Uint8 => 1,
            DType::Int16 | DType::Uint16 | DType::Float16 => 2,
            DType::Int32 | DType::Uint32 | DType::Float32 => 4,
        }
    }

    /// Returns `true` for integer dtypes (quantized tensors).
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Uint8
                | DType::Int16
                | DType::Uint16
                | DType::Int32
                | DType::Uint32
        )
    }

    /// Returns `true` for floating-point dtypes.
    pub fn is_float(self) -> bool {
        matches!(self, DType::Float16 | DType::Float32)
    }
}

// =============================================================================
// Parsing entry points + legacy v1 compatibility shim.
// =============================================================================

impl SchemaV2 {
    /// Parse schema metadata from a JSON string.
    ///
    /// Auto-detects the schema version from the `schema_version` field.
    /// Absent or `1` → legacy v1 metadata converted to v2 in memory.
    /// `2` → parsed as v2 directly. Any version higher than
    /// [`MAX_SUPPORTED_SCHEMA_VERSION`] is rejected with
    /// [`DecoderError::NotSupported`].
    pub fn parse_json(s: &str) -> DecoderResult<Self> {
        let value: serde_json::Value = serde_json::from_str(s)?;
        Self::from_json_value(value)
    }

    /// Parse schema metadata from a YAML string.
    ///
    /// Same version-detection logic as [`SchemaV2::parse_json`].
    pub fn parse_yaml(s: &str) -> DecoderResult<Self> {
        let value: serde_yaml::Value = serde_yaml::from_str(s)?;
        let json = serde_json::to_value(value)
            .map_err(|e| DecoderError::InvalidConfig(format!("yaml→json bridge failed: {e}")))?;
        Self::from_json_value(json)
    }

    /// Parse schema metadata from a file, auto-detecting JSON vs YAML
    /// from the file extension. Unknown extensions are parsed as JSON
    /// first then as YAML as a fallback.
    pub fn parse_file(path: impl AsRef<std::path::Path>) -> DecoderResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| DecoderError::InvalidConfig(format!("read {}: {e}", path.display())))?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("json") => Self::parse_json(&content),
            Some("yaml") | Some("yml") => Self::parse_yaml(&content),
            _ => Self::parse_json(&content).or_else(|_| Self::parse_yaml(&content)),
        }
    }

    /// Parse from an already-deserialized `serde_json::Value`. Useful for
    /// callers that have already done the initial deserialization step.
    pub fn from_json_value(value: serde_json::Value) -> DecoderResult<Self> {
        let version = value
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(1);

        if version > MAX_SUPPORTED_SCHEMA_VERSION {
            return Err(DecoderError::NotSupported(format!(
                "schema_version {version} is not supported by this HAL \
                 (maximum supported version is {MAX_SUPPORTED_SCHEMA_VERSION}); \
                 upgrade the HAL or downgrade the metadata"
            )));
        }

        if version >= 2 {
            serde_json::from_value(value).map_err(DecoderError::Json)
        } else {
            let v1: ConfigOutputs = serde_json::from_value(value).map_err(DecoderError::Json)?;
            Self::from_v1(&v1)
        }
    }

    /// Convert a legacy v1 [`ConfigOutputs`] to an equivalent v2
    /// [`SchemaV2`] in memory.
    ///
    /// The conversion preserves:
    /// - Output order and types (mapped to their v2 [`LogicalType`]).
    /// - Quantization (v1 `QuantTuple(scale, zp)` → v2 [`Quantization`] with
    ///   a single scalar scale/zero_point and unspecified dtype).
    /// - `dshape`, `shape`, `anchors`, `normalized` fields.
    /// - Root-level `nms` and `decoder_version`.
    ///
    /// Fields v1 does not carry (tensor `dtype`, per-channel quant, box
    /// encoding, score format, activation metadata, stride on physical
    /// children) are left as `None`. The v2 decoder is expected to infer
    /// these from the runtime tensor type and the legacy decoder
    /// dispatch rules.
    pub fn from_v1(v1: &ConfigOutputs) -> DecoderResult<Self> {
        let outputs = v1
            .outputs
            .iter()
            .map(logical_from_v1)
            .collect::<DecoderResult<Vec<_>>>()?;
        Ok(SchemaV2 {
            schema_version: 2,
            input: None,
            outputs,
            nms: v1.nms.as_ref().map(NmsMode::from_v1),
            decoder_version: v1.decoder_version.as_ref().map(DecoderVersion::from_v1),
        })
    }
}

impl SchemaV2 {
    /// Downconvert a v2 schema to a legacy [`ConfigOutputs`] for the
    /// v1 decoder dispatch path.
    ///
    /// Each [`LogicalOutput`] maps to one [`ConfigOutput`] variant
    /// selected by [`LogicalType`]; per-tensor scalar quantization
    /// becomes a `QuantTuple(scale, zp)`; and `decoder`, `anchors`,
    /// `normalized` are copied verbatim.
    ///
    /// This conversion does **not** reject logical outputs that also
    /// declare physical children (per-scale FPN splits, or channel
    /// sub-splits such as ARA-2 `boxes_xy` / `boxes_wh`). The returned
    /// legacy config captures the logical-level metadata, while the
    /// physical-to-logical merge is handled separately by the
    /// [`DecodeProgram`](crate::decoder::merge::DecodeProgram) that
    /// [`DecoderBuilder::build`](crate::decoder::builder::DecoderBuilder::build)
    /// compiles alongside this legacy config.
    ///
    /// Returns [`DecoderError::NotSupported`] when the schema uses
    /// features the v1 decoder cannot express at the logical level:
    /// - Per-channel quantization arrays on a logical output.
    /// - `encoding: dfl` on a **flat** logical output (no physical
    ///   children). DFL combined with per-scale children is handled by
    ///   the merge path (see [`crate::decoder::merge::DecodeProgram`])
    ///   which decodes the distribution before producing the merged
    ///   post-decode `(1, 4, total_anchors)` tensor the legacy decoder
    ///   consumes.
    pub fn to_legacy_config_outputs(&self) -> DecoderResult<ConfigOutputs> {
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for logical in &self.outputs {
            // Flat DFL (no children) remains unsupported — the HAL has
            // no path that applies softmax + dist2bbox to a single
            // `(1, 4·reg_max, anchors)` tensor yet. DFL with per-scale
            // children is decoded by the merge path, so we let it
            // through here and rely on the merged logical shape (post-
            // decode 4 channels) being valid for the legacy dispatch.
            if logical.type_ == LogicalType::Boxes
                && logical.encoding == Some(BoxEncoding::Dfl)
                && logical.outputs.is_empty()
            {
                return Err(DecoderError::NotSupported(format!(
                    "`boxes` output `{}` has `encoding: dfl` on a flat \
                     logical (no per-scale children); the HAL's DFL \
                     decode kernel only runs inside the per-scale merge \
                     path. Split the boxes output into per-FPN-level \
                     children (Hailo convention) or pre-decode to 4 \
                     channels in the model graph (TFLite convention).",
                    logical.name.as_deref().unwrap_or("<anonymous>"),
                )));
            }
            if let Some(q) = &logical.quantization {
                if q.is_per_channel() {
                    return Err(DecoderError::NotSupported(format!(
                        "logical `{}` uses per-channel quantization \
                         (axis {:?}, {} scales); the v1 decoder only \
                         supports per-tensor quantization",
                        logical.name.as_deref().unwrap_or("<anonymous>"),
                        q.axis,
                        q.scale.len(),
                    )));
                }
            }
            outputs.push(logical_to_legacy_config_output(logical)?);
        }
        Ok(ConfigOutputs {
            outputs,
            nms: self.nms.map(NmsMode::to_v1),
            decoder_version: self.decoder_version.map(|v| v.to_v1()),
        })
    }

    /// Validate the schema against the rules the HAL enforces.
    ///
    /// Rules checked:
    /// - `schema_version` in `[1, MAX_SUPPORTED_SCHEMA_VERSION]`.
    /// - Physical children, when present, carry non-empty `name` fields
    ///   so tensor binding by name is unambiguous.
    /// - All physical children of a given logical output have pairwise
    ///   distinct shapes (shape-based binding safety, per HailoRT spec).
    /// - For a `boxes` logical output with `encoding: dfl`, every
    ///   physical child shape has a `num_features` (or last) dimension
    ///   divisible by 4 (the box-coordinate count).
    /// - Per-scale splits carry `stride` on every child.
    /// - Mixed per-scale + channel-sub-split decompositions are
    ///   rejected (the spec permits only one merge strategy per logical).
    /// - `end2end` models (decoder_version=yolo26 with `detections`
    ///   output) do not also declare per-scale split children on that
    ///   output.
    pub fn validate(&self) -> DecoderResult<()> {
        if self.schema_version == 0 || self.schema_version > MAX_SUPPORTED_SCHEMA_VERSION {
            return Err(DecoderError::InvalidConfig(format!(
                "schema_version {} outside supported range [1, {MAX_SUPPORTED_SCHEMA_VERSION}]",
                self.schema_version
            )));
        }

        for logical in &self.outputs {
            validate_logical(logical)?;
        }

        Ok(())
    }
}

fn validate_logical(logical: &LogicalOutput) -> DecoderResult<()> {
    if logical.outputs.is_empty() {
        return Ok(());
    }

    // All children must carry a name for unambiguous tensor binding.
    for child in &logical.outputs {
        if child.name.is_empty() {
            return Err(DecoderError::InvalidConfig(format!(
                "physical child of logical `{}` is missing `name`; name is \
                 required for tensor binding",
                logical.name.as_deref().unwrap_or("<anonymous>")
            )));
        }
    }

    // Uniqueness of physical child shapes *within the same type* — two
    // children with distinct types (e.g. ARA-2 `boxes_xy` + `boxes_wh`)
    // may legitimately share shape, since type disambiguates the binding.
    for (i, a) in logical.outputs.iter().enumerate() {
        for b in &logical.outputs[i + 1..] {
            if a.shape == b.shape && a.type_ == b.type_ {
                return Err(DecoderError::InvalidConfig(format!(
                    "physical children `{}` and `{}` share shape {:?} and \
                     type; tensor binding cannot be resolved",
                    a.name, b.name, a.shape
                )));
            }
        }
    }

    // Merge strategy must be uniform: either all children carry `stride`
    // (per-scale split) or none (channel sub-split). Mixed decompositions
    // are ill-defined.
    let strided: Vec<_> = logical.outputs.iter().map(|c| c.stride.is_some()).collect();
    let all_strided = strided.iter().all(|&b| b);
    let none_strided = strided.iter().all(|&b| !b);
    if !(all_strided || none_strided) {
        return Err(DecoderError::InvalidConfig(format!(
            "logical `{}` mixes per-scale children (with stride) and \
             channel sub-split children (without stride); decomposition \
             must be uniform",
            logical.name.as_deref().unwrap_or("<anonymous>")
        )));
    }

    // DFL boxes: every child's feature axis must be divisible by 4
    // (otherwise `reg_max` is not an integer).
    if logical.type_ == LogicalType::Boxes && logical.encoding == Some(BoxEncoding::Dfl) {
        for child in &logical.outputs {
            if let Some(feat) = last_feature_axis(child) {
                if feat % 4 != 0 {
                    return Err(DecoderError::InvalidConfig(format!(
                        "DFL boxes child `{}` feature axis {feat} is not \
                         divisible by 4 (reg_max×4)",
                        child.name
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Resolve the channel / feature count from a physical child's dshape
/// when present, otherwise from the last dimension of its shape.
fn last_feature_axis(child: &PhysicalOutput) -> Option<usize> {
    // Prefer explicit named dimensions: NumFeatures, NumClasses,
    // NumProtos, BoxCoords, NumAnchorsXFeatures.
    for (name, size) in &child.dshape {
        if matches!(
            name,
            DimName::NumFeatures
                | DimName::NumClasses
                | DimName::NumProtos
                | DimName::BoxCoords
                | DimName::NumAnchorsXFeatures
        ) {
            return Some(*size);
        }
    }
    child.shape.last().copied()
}

fn quantization_from_v1(q: Option<QuantTuple>) -> Option<Quantization> {
    q.map(|QuantTuple(scale, zp)| Quantization {
        scale: vec![scale],
        zero_point: Some(vec![zp]),
        axis: None,
        dtype: None,
    })
}

fn logical_from_v1(v1: &ConfigOutput) -> DecoderResult<LogicalOutput> {
    match v1 {
        ConfigOutput::Detection(d) => {
            // v1 Detection covers two semantic cases that v2 separates:
            //   - ModelPack anchor-grid (decoder=modelpack, anchors present)
            //   - YOLO legacy combined (decoder=ultralytics, no anchors)
            // Both map to v2 LogicalType::Detection; the decoder dispatch
            // still differentiates via the `decoder` + `anchors` fields.
            let encoding = match (d.decoder, d.anchors.is_some()) {
                (configs::DecoderType::ModelPack, true) => Some(BoxEncoding::Anchor),
                (configs::DecoderType::Ultralytics, _) => Some(BoxEncoding::Direct),
                // ModelPack without anchors — keep encoding unset; the
                // decoder may not need it.
                (configs::DecoderType::ModelPack, false) => None,
            };
            Ok(LogicalOutput {
                name: None,
                type_: LogicalType::Detection,
                shape: d.shape.clone(),
                dshape: d.dshape.clone(),
                decoder: Some(DecoderKind::from_v1(d.decoder)),
                encoding,
                score_format: None,
                normalized: d.normalized,
                anchors: d.anchors.clone(),
                stride: None,
                dtype: None,
                quantization: quantization_from_v1(d.quantization),
                outputs: Vec::new(),
            })
        }
        ConfigOutput::Boxes(b) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Boxes,
            shape: b.shape.clone(),
            dshape: b.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(b.decoder)),
            // v1 boxes are always pre-decoded 4-channel (the legacy
            // convention). Explicitly declare Direct so the v2 dispatch
            // doesn't try DFL decoding on them.
            encoding: Some(BoxEncoding::Direct),
            score_format: None,
            normalized: b.normalized,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(b.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::Scores(s) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Scores,
            shape: s.shape.clone(),
            dshape: s.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(s.decoder)),
            encoding: None,
            // v1 does not declare score format explicitly; assume per_class
            // (YOLOv8-style). YOLOv5 users must migrate to v2 to get
            // obj_x_class behaviour.
            score_format: Some(ScoreFormat::PerClass),
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(s.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::Protos(p) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Protos,
            shape: p.shape.clone(),
            dshape: p.dshape.clone(),
            // protos are consumed directly; decoder field is informational.
            decoder: Some(DecoderKind::from_v1(p.decoder)),
            encoding: None,
            score_format: None,
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(p.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::MaskCoefficients(m) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::MaskCoefs,
            shape: m.shape.clone(),
            dshape: m.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(m.decoder)),
            encoding: None,
            score_format: None,
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(m.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::Segmentation(seg) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Segmentation,
            shape: seg.shape.clone(),
            dshape: seg.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(seg.decoder)),
            encoding: None,
            score_format: None,
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(seg.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::Mask(m) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Masks,
            shape: m.shape.clone(),
            dshape: m.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(m.decoder)),
            encoding: None,
            score_format: None,
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(m.quantization),
            outputs: Vec::new(),
        }),
        ConfigOutput::Classes(c) => Ok(LogicalOutput {
            name: None,
            type_: LogicalType::Classes,
            shape: c.shape.clone(),
            dshape: c.dshape.clone(),
            decoder: Some(DecoderKind::from_v1(c.decoder)),
            encoding: None,
            score_format: None,
            normalized: None,
            anchors: None,
            stride: None,
            dtype: None,
            quantization: quantization_from_v1(c.quantization),
            outputs: Vec::new(),
        }),
    }
}

impl DecoderKind {
    /// Convert a legacy v1 [`configs::DecoderType`] to a v2 [`DecoderKind`].
    pub fn from_v1(v: configs::DecoderType) -> Self {
        match v {
            configs::DecoderType::ModelPack => DecoderKind::ModelPack,
            configs::DecoderType::Ultralytics => DecoderKind::Ultralytics,
        }
    }

    /// Convert back to the legacy v1 [`configs::DecoderType`].
    pub fn to_v1(self) -> configs::DecoderType {
        match self {
            DecoderKind::ModelPack => configs::DecoderType::ModelPack,
            DecoderKind::Ultralytics => configs::DecoderType::Ultralytics,
        }
    }
}

impl DecoderVersion {
    /// Convert a legacy v1 [`configs::DecoderVersion`] to a v2 [`DecoderVersion`].
    pub fn from_v1(v: &configs::DecoderVersion) -> Self {
        match v {
            configs::DecoderVersion::Yolov5 => DecoderVersion::Yolov5,
            configs::DecoderVersion::Yolov8 => DecoderVersion::Yolov8,
            configs::DecoderVersion::Yolo11 => DecoderVersion::Yolo11,
            configs::DecoderVersion::Yolo26 => DecoderVersion::Yolo26,
        }
    }

    /// Convert back to the legacy v1 [`configs::DecoderVersion`].
    pub fn to_v1(self) -> configs::DecoderVersion {
        match self {
            DecoderVersion::Yolov5 => configs::DecoderVersion::Yolov5,
            DecoderVersion::Yolov8 => configs::DecoderVersion::Yolov8,
            DecoderVersion::Yolo11 => configs::DecoderVersion::Yolo11,
            DecoderVersion::Yolo26 => configs::DecoderVersion::Yolo26,
        }
    }
}

impl NmsMode {
    /// Convert a legacy v1 [`configs::Nms`] to a v2 [`NmsMode`].
    pub fn from_v1(v: &configs::Nms) -> Self {
        match v {
            configs::Nms::ClassAgnostic => NmsMode::ClassAgnostic,
            configs::Nms::ClassAware => NmsMode::ClassAware,
        }
    }

    /// Convert back to the legacy v1 [`configs::Nms`].
    pub fn to_v1(self) -> configs::Nms {
        match self {
            NmsMode::ClassAgnostic => configs::Nms::ClassAgnostic,
            NmsMode::ClassAware => configs::Nms::ClassAware,
        }
    }
}

/// Convert a quantized v2 [`Quantization`] to a v1 [`QuantTuple`]. Only
/// valid for per-tensor scalar quantization.
fn quantization_to_legacy(q: &Quantization) -> DecoderResult<QuantTuple> {
    if q.is_per_channel() {
        return Err(DecoderError::NotSupported(
            "per-channel quantization cannot be expressed as a v1 QuantTuple".into(),
        ));
    }
    let scale = *q.scale.first().unwrap_or(&0.0);
    let zp = q.zero_point_at(0);
    Ok(QuantTuple(scale, zp))
}

/// Drop axes named `padding` (always size 1 per spec) from the given
/// shape/dshape pair. ARA-2 emits logical shapes like
/// `[1, 4, 8400, 1]` with a trailing `padding=1` dim to satisfy the
/// converter's rank requirements — the decoder only cares about the
/// semantic axes, so squeezing is safe.
pub(crate) fn squeeze_padding_dims(
    shape: Vec<usize>,
    dshape: Vec<(DimName, usize)>,
) -> (Vec<usize>, Vec<(DimName, usize)>) {
    // dshape is `#[serde(default)]`; a logical output without named dims
    // arrives here with an empty dshape. `zip` would stop at the shorter
    // iterator and silently truncate shape to `[]`, so short-circuit.
    if dshape.is_empty() {
        return (shape, dshape);
    }
    let keep: Vec<bool> = dshape
        .iter()
        .map(|(n, _)| !matches!(n, DimName::Padding))
        .collect();
    let shape = shape
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(s, &k)| k.then_some(s))
        .collect();
    let dshape = dshape
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(d, &k)| k.then_some(d))
        .collect();
    (shape, dshape)
}

/// Return the list of axis indices in `dshape` that carry the
/// `padding` dim name. Indices are returned in descending order so
/// that `remove_axis` calls can be applied directly without tracking
/// index shifts.
pub(crate) fn padding_axes(dshape: &[(DimName, usize)]) -> Vec<usize> {
    let mut v: Vec<usize> = dshape
        .iter()
        .enumerate()
        .filter_map(|(i, (n, _))| matches!(n, DimName::Padding).then_some(i))
        .collect();
    v.sort_by(|a, b| b.cmp(a));
    v
}

fn logical_to_legacy_config_output(logical: &LogicalOutput) -> DecoderResult<ConfigOutput> {
    let decoder = logical
        .decoder
        .map(|d| d.to_v1())
        .unwrap_or(configs::DecoderType::Ultralytics);
    let quantization = logical
        .quantization
        .as_ref()
        .map(quantization_to_legacy)
        .transpose()?;
    // Squeeze explicit `padding` dims before handing to the legacy
    // dispatch: the v1 decoder's `verify_yolo_*` helpers require rank-3
    // shapes, but v2 metadata often carries an explicit `padding: 1`
    // axis (ARA-2).
    let (shape, dshape) = squeeze_padding_dims(logical.shape.clone(), logical.dshape.clone());

    Ok(match logical.type_ {
        LogicalType::Boxes => ConfigOutput::Boxes(configs::Boxes {
            decoder,
            quantization,
            shape,
            dshape,
            normalized: logical.normalized,
        }),
        LogicalType::Scores => ConfigOutput::Scores(configs::Scores {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        LogicalType::Protos => ConfigOutput::Protos(configs::Protos {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        LogicalType::MaskCoefs => ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        LogicalType::Segmentation => ConfigOutput::Segmentation(configs::Segmentation {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        LogicalType::Masks => ConfigOutput::Mask(configs::Mask {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        LogicalType::Classes => ConfigOutput::Classes(configs::Classes {
            decoder,
            quantization,
            shape,
            dshape,
        }),
        // Detection covers ModelPack anchor-grid and legacy YOLO combined.
        // Detections (plural) is end-to-end; maps to Detection with the
        // appropriate dimension layout.
        LogicalType::Detection | LogicalType::Detections => {
            ConfigOutput::Detection(configs::Detection {
                anchors: logical.anchors.clone(),
                decoder,
                quantization,
                shape,
                dshape,
                normalized: logical.normalized,
            })
        }
        // Objectness / Landmarks have no direct v1 equivalent; the v1
        // YOLOv5 path embedded objectness in the combined Detection.
        LogicalType::Objectness | LogicalType::Landmarks => {
            return Err(DecoderError::NotSupported(format!(
                "logical type {:?} has no legacy v1 equivalent; use the \
                 native v2 decoder path",
                logical.type_
            )));
        }
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn schema_default_is_v2() {
        let s = SchemaV2::default();
        assert_eq!(s.schema_version, 2);
        assert!(s.outputs.is_empty());
    }

    #[test]
    fn dtype_roundtrip() {
        for d in [
            DType::Int8,
            DType::Uint8,
            DType::Int16,
            DType::Uint16,
            DType::Float16,
            DType::Float32,
        ] {
            let j = serde_json::to_string(&d).unwrap();
            let back: DType = serde_json::from_str(&j).unwrap();
            assert_eq!(back, d);
        }
    }

    #[test]
    fn dtype_widths() {
        assert_eq!(DType::Int8.size_bytes(), 1);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float32.size_bytes(), 4);
    }

    #[test]
    fn stride_accepts_scalar_or_pair() {
        let a: Stride = serde_json::from_str("8").unwrap();
        let b: Stride = serde_json::from_str("[8, 16]").unwrap();
        assert_eq!(a, Stride::Square(8));
        assert_eq!(b, Stride::Rect([8, 16]));
        assert_eq!(a.x(), 8);
        assert_eq!(a.y(), 8);
        assert_eq!(b.x(), 8);
        assert_eq!(b.y(), 16);
    }

    #[test]
    fn quantization_scalar_scale() {
        let j = r#"{"scale": 0.00392, "zero_point": 0, "dtype": "int8"}"#;
        let q: Quantization = serde_json::from_str(j).unwrap();
        assert!(q.is_per_tensor());
        assert!(q.is_symmetric());
        assert_eq!(q.scale_at(0), 0.00392);
        assert_eq!(q.scale_at(5), 0.00392);
        assert_eq!(q.zero_point_at(0), 0);
    }

    #[test]
    fn quantization_per_channel() {
        let j = r#"{"scale": [0.054, 0.089, 0.195], "axis": 0, "dtype": "int8"}"#;
        let q: Quantization = serde_json::from_str(j).unwrap();
        assert!(q.is_per_channel());
        assert!(q.is_symmetric());
        assert_eq!(q.axis, Some(0));
        assert_eq!(q.scale_at(0), 0.054);
        assert_eq!(q.scale_at(2), 0.195);
    }

    #[test]
    fn quantization_asymmetric_per_tensor() {
        let j = r#"{"scale": 0.176, "zero_point": 198, "dtype": "uint8"}"#;
        let q: Quantization = serde_json::from_str(j).unwrap();
        assert!(!q.is_symmetric());
        assert_eq!(q.zero_point_at(0), 198);
        assert_eq!(q.zero_point_at(10), 198);
    }

    #[test]
    fn quantization_symmetric_default_zero_point() {
        let j = r#"{"scale": 0.00392, "dtype": "int8"}"#;
        let q: Quantization = serde_json::from_str(j).unwrap();
        assert!(q.is_symmetric());
        assert_eq!(q.zero_point_at(0), 0);
    }

    #[test]
    fn logical_output_flat_tflite_boxes() {
        // Example 3 from the spec: TFLite YOLOv8 detection, flat boxes
        let j = r#"{
          "name": "boxes", "type": "boxes",
          "shape": [1, 64, 8400],
          "dshape": [{"batch": 1}, {"num_features": 64}, {"num_boxes": 8400}],
          "dtype": "int8",
          "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
          "decoder": "ultralytics",
          "encoding": "dfl",
          "normalized": true
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert_eq!(lo.type_, LogicalType::Boxes);
        assert_eq!(lo.encoding, Some(BoxEncoding::Dfl));
        assert_eq!(lo.normalized, Some(true));
        assert!(!lo.is_split());
        assert_eq!(lo.dtype, Some(DType::Int8));
    }

    #[test]
    fn logical_output_hailo_per_scale_split() {
        // Example 5 from the spec: Hailo YOLOv8 boxes, per-scale split
        let j = r#"{
          "name": "boxes", "type": "boxes",
          "shape": [1, 64, 8400],
          "encoding": "dfl", "decoder": "ultralytics", "normalized": true,
          "outputs": [
            {
              "name": "boxes_0", "type": "boxes",
              "stride": 8, "scale_index": 0,
              "shape": [1, 80, 80, 64],
              "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_features": 64}],
              "dtype": "uint8",
              "quantization": {"scale": 0.0234, "zero_point": 128, "dtype": "uint8"}
            }
          ]
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert!(lo.is_split());
        assert_eq!(lo.outputs.len(), 1);
        let child = &lo.outputs[0];
        assert_eq!(child.name, "boxes_0");
        assert_eq!(child.type_, PhysicalType::Boxes);
        assert_eq!(child.stride, Some(Stride::Square(8)));
        assert_eq!(child.scale_index, Some(0));
        assert_eq!(child.dtype, DType::Uint8);
    }

    #[test]
    fn logical_output_ara2_xy_wh_channel_split() {
        // Example 4 from the spec: ARA-2 boxes split into xy/wh
        let j = r#"{
          "name": "boxes", "type": "boxes",
          "shape": [1, 4, 8400, 1],
          "encoding": "direct", "decoder": "ultralytics", "normalized": true,
          "outputs": [
            {
              "name": "_model_22_Div_1_output_0", "type": "boxes_xy",
              "shape": [1, 2, 8400, 1],
              "dshape": [{"batch": 1}, {"box_coords": 2}, {"num_boxes": 8400}, {"padding": 1}],
              "dtype": "int16",
              "quantization": {"scale": 3.129e-5, "zero_point": 0, "dtype": "int16"}
            },
            {
              "name": "_model_22_Sub_1_output_0", "type": "boxes_wh",
              "shape": [1, 2, 8400, 1],
              "dshape": [{"batch": 1}, {"box_coords": 2}, {"num_boxes": 8400}, {"padding": 1}],
              "dtype": "int16",
              "quantization": {"scale": 3.149e-5, "zero_point": 0, "dtype": "int16"}
            }
          ]
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert_eq!(lo.encoding, Some(BoxEncoding::Direct));
        assert_eq!(lo.outputs.len(), 2);
        assert_eq!(lo.outputs[0].type_, PhysicalType::BoxesXy);
        assert_eq!(lo.outputs[1].type_, PhysicalType::BoxesWh);
        assert!(lo.outputs[0].stride.is_none());
        assert!(lo.outputs[1].stride.is_none());
    }

    #[test]
    fn logical_output_hailo_scores_sigmoid_applied() {
        let j = r#"{
          "name": "scores", "type": "scores",
          "shape": [1, 80, 8400],
          "decoder": "ultralytics", "score_format": "per_class",
          "outputs": [
            {
              "name": "scores_0", "type": "scores",
              "stride": 8, "scale_index": 0,
              "shape": [1, 80, 80, 80],
              "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_classes": 80}],
              "dtype": "uint8",
              "quantization": {"scale": 0.003922, "dtype": "uint8"},
              "activation_applied": "sigmoid"
            }
          ]
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert_eq!(lo.score_format, Some(ScoreFormat::PerClass));
        let child = &lo.outputs[0];
        assert_eq!(child.activation_applied, Some(Activation::Sigmoid));
        assert!(child.activation_required.is_none());
    }

    #[test]
    fn yolo26_end_to_end_detections() {
        let j = r#"{
          "schema_version": 2,
          "decoder_version": "yolo26",
          "outputs": [{
            "name": "output0", "type": "detections",
            "shape": [1, 100, 6],
            "dshape": [{"batch": 1}, {"num_boxes": 100}, {"num_features": 6}],
            "dtype": "int8",
            "quantization": {"scale": 0.0078, "zero_point": 0, "dtype": "int8"},
            "normalized": false,
            "decoder": "ultralytics"
          }]
        }"#;
        let s: SchemaV2 = serde_json::from_str(j).unwrap();
        assert_eq!(s.decoder_version, Some(DecoderVersion::Yolo26));
        assert!(s.decoder_version.unwrap().is_end_to_end());
        assert_eq!(s.outputs[0].type_, LogicalType::Detections);
        assert_eq!(s.outputs[0].normalized, Some(false));
        assert!(s.nms.is_none());
    }

    #[test]
    fn modelpack_anchor_detection_with_rect_stride() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name": "output_0", "type": "detection",
            "shape": [1, 40, 40, 54],
            "dshape": [{"batch": 1}, {"height": 40}, {"width": 40}, {"num_anchors_x_features": 54}],
            "dtype": "uint8",
            "quantization": {"scale": 0.176, "zero_point": 198, "dtype": "uint8"},
            "decoder": "modelpack",
            "encoding": "anchor",
            "stride": [16, 16],
            "anchors": [[0.054, 0.065], [0.089, 0.139], [0.195, 0.196]]
          }]
        }"#;
        let s: SchemaV2 = serde_json::from_str(j).unwrap();
        let lo = &s.outputs[0];
        assert_eq!(lo.encoding, Some(BoxEncoding::Anchor));
        assert_eq!(lo.stride, Some(Stride::Rect([16, 16])));
        assert_eq!(lo.anchors.as_ref().map(|a| a.len()), Some(3));
    }

    #[test]
    fn yolov5_obj_x_class_objectness_logical() {
        let j = r#"{
          "name": "objectness", "type": "objectness",
          "shape": [1, 3, 8400],
          "decoder": "ultralytics",
          "outputs": [{
            "name": "objectness_0", "type": "objectness",
            "stride": 8, "scale_index": 0,
            "shape": [1, 80, 80, 3],
            "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_features": 3}],
            "dtype": "uint8",
            "quantization": {"scale": 0.0039, "zero_point": 0, "dtype": "uint8"},
            "activation_applied": "sigmoid"
          }]
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert_eq!(lo.type_, LogicalType::Objectness);
        assert_eq!(lo.outputs[0].activation_applied, Some(Activation::Sigmoid));
    }

    #[test]
    fn direct_protos_no_decoder() {
        // protos are consumed directly — no `decoder` field
        let j = r#"{
          "name": "protos", "type": "protos",
          "shape": [1, 32, 160, 160],
          "dshape": [{"batch": 1}, {"num_protos": 32}, {"height": 160}, {"width": 160}],
          "dtype": "uint8",
          "quantization": {"scale": 0.0203, "zero_point": 45, "dtype": "uint8"},
          "stride": 4
        }"#;
        let lo: LogicalOutput = serde_json::from_str(j).unwrap();
        assert_eq!(lo.type_, LogicalType::Protos);
        assert!(lo.decoder.is_none());
        assert_eq!(lo.stride, Some(Stride::Square(4)));
    }

    #[test]
    fn full_yolov8_tflite_flat_detection() {
        // Example 3: complete two-output YOLOv8 detection schema
        let j = r#"{
          "schema_version": 2,
          "decoder_version": "yolov8",
          "nms": "class_agnostic",
          "input": { "shape": [1, 640, 640, 3], "cameraadaptor": "rgb" },
          "outputs": [
            {
              "name": "boxes", "type": "boxes",
              "shape": [1, 64, 8400],
              "dshape": [{"batch": 1}, {"num_features": 64}, {"num_boxes": 8400}],
              "dtype": "int8",
              "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
              "decoder": "ultralytics",
              "encoding": "dfl",
              "normalized": true
            },
            {
              "name": "scores", "type": "scores",
              "shape": [1, 80, 8400],
              "dshape": [{"batch": 1}, {"num_classes": 80}, {"num_boxes": 8400}],
              "dtype": "int8",
              "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
              "decoder": "ultralytics",
              "score_format": "per_class"
            }
          ]
        }"#;
        let s: SchemaV2 = serde_json::from_str(j).unwrap();
        assert_eq!(s.schema_version, 2);
        assert_eq!(s.decoder_version, Some(DecoderVersion::Yolov8));
        assert_eq!(s.nms, Some(NmsMode::ClassAgnostic));
        assert_eq!(s.input.as_ref().unwrap().shape, vec![1, 640, 640, 3]);
        assert_eq!(s.outputs.len(), 2);
    }

    #[test]
    fn schema_unknown_version_parses_without_validation() {
        // Parser accepts any u32; the decoder is responsible for rejecting
        // unsupported versions with a useful error.
        let j = r#"{"schema_version": 99, "outputs": []}"#;
        let s: SchemaV2 = serde_json::from_str(j).unwrap();
        assert_eq!(s.schema_version, 99);
    }

    #[test]
    fn serde_roundtrip_preserves_fields() {
        let original = SchemaV2 {
            schema_version: 2,
            input: Some(InputSpec {
                shape: vec![1, 3, 640, 640],
                dshape: vec![],
                cameraadaptor: Some("rgb".into()),
            }),
            outputs: vec![LogicalOutput {
                name: Some("boxes".into()),
                type_: LogicalType::Boxes,
                shape: vec![1, 4, 8400],
                dshape: vec![],
                decoder: Some(DecoderKind::Ultralytics),
                encoding: Some(BoxEncoding::Direct),
                score_format: None,
                normalized: Some(true),
                anchors: None,
                stride: None,
                dtype: Some(DType::Float32),
                quantization: None,
                outputs: vec![],
            }],
            nms: Some(NmsMode::ClassAgnostic),
            decoder_version: Some(DecoderVersion::Yolov8),
        };
        let j = serde_json::to_string(&original).unwrap();
        let parsed: SchemaV2 = serde_json::from_str(&j).unwrap();
        assert_eq!(parsed, original);
    }

    // ─── v1 → v2 shim tests ─────────────────────────────────

    #[test]
    fn parse_v1_yaml_yolov8_seg_testdata() {
        let yaml = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_seg.yaml"
        ));
        let schema = SchemaV2::parse_yaml(yaml).expect("parse v1 yaml");
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.outputs.len(), 2);
        // First output: Detection [1, 116, 8400]
        let det = &schema.outputs[0];
        assert_eq!(det.type_, LogicalType::Detection);
        assert_eq!(det.shape, vec![1, 116, 8400]);
        assert_eq!(det.decoder, Some(DecoderKind::Ultralytics));
        assert_eq!(det.encoding, Some(BoxEncoding::Direct));
        let q = det.quantization.as_ref().unwrap();
        assert_eq!(q.scale.len(), 1);
        assert!((q.scale[0] - 0.021_287_762).abs() < 1e-6);
        assert_eq!(q.zero_point, Some(vec![31]));
        // Second output: Protos [1, 160, 160, 32]
        let protos = &schema.outputs[1];
        assert_eq!(protos.type_, LogicalType::Protos);
        assert_eq!(protos.shape, vec![1, 160, 160, 32]);
    }

    #[test]
    fn parse_v1_json_modelpack_split_testdata() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split.json"
        ));
        let schema = SchemaV2::parse_json(json).expect("parse v1 json");
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.outputs.len(), 2);
        // Both are ModelPack anchor detection with anchors
        for out in &schema.outputs {
            assert_eq!(out.type_, LogicalType::Detection);
            assert_eq!(out.decoder, Some(DecoderKind::ModelPack));
            assert_eq!(out.encoding, Some(BoxEncoding::Anchor));
            assert_eq!(out.anchors.as_ref().map(|a| a.len()), Some(3));
        }
    }

    #[test]
    fn parse_v2_json_direct_when_schema_version_present() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name": "boxes", "type": "boxes",
            "shape": [1, 4, 8400],
            "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}],
            "dtype": "float32",
            "decoder": "ultralytics",
            "encoding": "direct",
            "normalized": true
          }]
        }"#;
        let schema = SchemaV2::parse_json(j).unwrap();
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.outputs[0].type_, LogicalType::Boxes);
    }

    #[test]
    fn parse_rejects_future_schema_version() {
        let j = r#"{"schema_version": 99, "outputs": []}"#;
        let err = SchemaV2::parse_json(j).unwrap_err();
        matches!(err, DecoderError::NotSupported(_));
    }

    #[test]
    fn parse_absent_schema_version_treats_as_v1() {
        // No schema_version field — classic v1 yolov8 split
        let j = r#"{
          "outputs": [
            {
              "type": "boxes", "decoder": "ultralytics",
              "shape": [1, 4, 8400],
              "quantization": [0.00392, 0]
            },
            {
              "type": "scores", "decoder": "ultralytics",
              "shape": [1, 80, 8400],
              "quantization": [0.00392, 0]
            }
          ]
        }"#;
        let schema = SchemaV2::parse_json(j).expect("v1 legacy parse");
        assert_eq!(schema.schema_version, 2); // converted
        assert_eq!(schema.outputs.len(), 2);
        assert_eq!(schema.outputs[0].type_, LogicalType::Boxes);
        assert_eq!(schema.outputs[1].type_, LogicalType::Scores);
        // default score_format assumed on v1→v2
        assert_eq!(schema.outputs[1].score_format, Some(ScoreFormat::PerClass));
    }

    #[test]
    fn from_v1_preserves_nms_and_decoder_version() {
        let v1 = ConfigOutputs {
            outputs: vec![ConfigOutput::Boxes(crate::configs::Boxes {
                decoder: crate::configs::DecoderType::Ultralytics,
                quantization: Some(crate::configs::QuantTuple(0.01, 5)),
                shape: vec![1, 4, 8400],
                dshape: vec![],
                normalized: Some(true),
            })],
            nms: Some(crate::configs::Nms::ClassAware),
            decoder_version: Some(crate::configs::DecoderVersion::Yolo11),
        };
        let v2 = SchemaV2::from_v1(&v1).unwrap();
        assert_eq!(v2.nms, Some(NmsMode::ClassAware));
        assert_eq!(v2.decoder_version, Some(DecoderVersion::Yolo11));
        assert_eq!(v2.outputs[0].normalized, Some(true));
        let q = v2.outputs[0].quantization.as_ref().unwrap();
        assert_eq!(q.scale, vec![0.01]);
        assert_eq!(q.zero_point, Some(vec![5]));
        assert_eq!(q.dtype, None); // v1 did not carry dtype
    }

    #[test]
    fn from_v1_modelpack_anchor_detection_maps_encoding() {
        let v1 = ConfigOutputs {
            outputs: vec![ConfigOutput::Detection(crate::configs::Detection {
                anchors: Some(vec![[0.1, 0.2], [0.3, 0.4]]),
                decoder: crate::configs::DecoderType::ModelPack,
                quantization: Some(crate::configs::QuantTuple(0.176, 198)),
                shape: vec![1, 40, 40, 54],
                dshape: vec![],
                normalized: None,
            })],
            nms: None,
            decoder_version: None,
        };
        let v2 = SchemaV2::from_v1(&v1).unwrap();
        assert_eq!(v2.outputs[0].encoding, Some(BoxEncoding::Anchor));
        assert_eq!(v2.outputs[0].decoder, Some(DecoderKind::ModelPack));
        assert_eq!(v2.outputs[0].anchors.as_ref().map(|a| a.len()), Some(2));
    }

    // ─── validate() tests ──────────────────────────────────

    #[test]
    fn validate_accepts_flat_v2_yolov8_detection() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [
            {"name":"boxes","type":"boxes","shape":[1,64,8400],
             "dtype":"int8","decoder":"ultralytics","encoding":"dfl"},
            {"name":"scores","type":"scores","shape":[1,80,8400],
             "dtype":"int8","decoder":"ultralytics","score_format":"per_class"}
          ]
        }"#;
        SchemaV2::parse_json(j).unwrap().validate().unwrap();
    }

    #[test]
    fn validate_rejects_unnamed_physical_child() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,64,8400],
            "encoding":"dfl","decoder":"ultralytics",
            "outputs": [{
              "name":"","type":"boxes","stride":8,
              "shape":[1,80,80,64],"dtype":"uint8"
            }]
          }]
        }"#;
        let err = SchemaV2::parse_json(j).unwrap().validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing `name`"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_duplicate_physical_shapes() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,64,8400],
            "encoding":"dfl","decoder":"ultralytics",
            "outputs": [
              {"name":"a","type":"boxes","stride":8,"shape":[1,80,80,64],"dtype":"uint8"},
              {"name":"b","type":"boxes","stride":16,"shape":[1,80,80,64],"dtype":"uint8"}
            ]
          }]
        }"#;
        let err = SchemaV2::parse_json(j).unwrap().validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("share shape"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_mixed_decomposition() {
        // one child carries stride, the other does not — ill-defined merge
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,4,8400,1],
            "encoding":"direct","decoder":"ultralytics",
            "outputs": [
              {"name":"xy","type":"boxes_xy","shape":[1,2,8400,1],"dtype":"int16"},
              {"name":"p0","type":"boxes","stride":8,"shape":[1,80,80,64],"dtype":"uint8"}
            ]
          }]
        }"#;
        let err = SchemaV2::parse_json(j).unwrap().validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("uniform"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_dfl_boxes_feature_not_divisible_by_4() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,63,8400],
            "encoding":"dfl","decoder":"ultralytics",
            "outputs": [{
              "name":"b0","type":"boxes","stride":8,
              "shape":[1,80,80,63],
              "dshape":[{"batch":1},{"height":80},{"width":80},{"num_features":63}],
              "dtype":"uint8"
            }]
          }]
        }"#;
        let err = SchemaV2::parse_json(j).unwrap().validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not"), "got: {msg}");
        assert!(msg.contains("divisible by 4"), "got: {msg}");
    }

    #[test]
    fn validate_accepts_hailo_per_scale_yolov8() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,64,8400],
            "encoding":"dfl","decoder":"ultralytics","normalized":true,
            "outputs": [
              {"name":"b0","type":"boxes","stride":8,
               "shape":[1,80,80,64],
               "dshape":[{"batch":1},{"height":80},{"width":80},{"num_features":64}],
               "dtype":"uint8",
               "quantization":{"scale":0.0234,"zero_point":128,"dtype":"uint8"}},
              {"name":"b1","type":"boxes","stride":16,
               "shape":[1,40,40,64],
               "dshape":[{"batch":1},{"height":40},{"width":40},{"num_features":64}],
               "dtype":"uint8",
               "quantization":{"scale":0.0198,"zero_point":130,"dtype":"uint8"}},
              {"name":"b2","type":"boxes","stride":32,
               "shape":[1,20,20,64],
               "dshape":[{"batch":1},{"height":20},{"width":20},{"num_features":64}],
               "dtype":"uint8",
               "quantization":{"scale":0.0312,"zero_point":125,"dtype":"uint8"}}
            ]
          }]
        }"#;
        let s = SchemaV2::parse_json(j).unwrap();
        s.validate().unwrap();
    }

    #[test]
    fn validate_accepts_ara2_xy_wh() {
        let j = r#"{
          "schema_version": 2,
          "outputs": [{
            "name":"boxes","type":"boxes","shape":[1,4,8400,1],
            "encoding":"direct","decoder":"ultralytics","normalized":true,
            "outputs": [
              {"name":"xy","type":"boxes_xy","shape":[1,2,8400,1],
               "dshape":[{"batch":1},{"box_coords":2},{"num_boxes":8400},{"padding":1}],
               "dtype":"int16",
               "quantization":{"scale":3.1e-5,"zero_point":0,"dtype":"int16"}},
              {"name":"wh","type":"boxes_wh","shape":[1,2,8400,1],
               "dshape":[{"batch":1},{"box_coords":2},{"num_boxes":8400},{"padding":1}],
               "dtype":"int16",
               "quantization":{"scale":3.2e-5,"zero_point":0,"dtype":"int16"}}
            ]
          }]
        }"#;
        SchemaV2::parse_json(j).unwrap().validate().unwrap();
    }

    #[test]
    fn parse_file_auto_detects_json() {
        let tmp = std::env::temp_dir().join(format!("schema_v2_test_{}.json", std::process::id()));
        std::fs::write(&tmp, r#"{"schema_version":2,"outputs":[]}"#).unwrap();
        let s = SchemaV2::parse_file(&tmp).unwrap();
        assert_eq!(s.schema_version, 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn parse_file_auto_detects_yaml() {
        let tmp = std::env::temp_dir().join(format!("schema_v2_test_{}.yaml", std::process::id()));
        std::fs::write(&tmp, "schema_version: 2\noutputs: []\n").unwrap();
        let s = SchemaV2::parse_file(&tmp).unwrap();
        assert_eq!(s.schema_version, 2);
        let _ = std::fs::remove_file(&tmp);
    }

    // ─── Real ARA-2 DVM fixtures ────────────────────────────

    #[test]
    fn parse_real_ara2_int8_dvm_metadata() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/ara2_int8_edgefirst.json"
        ));
        let schema = SchemaV2::parse_json(json).expect("ARA-2 int8 parse");
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.decoder_version, Some(DecoderVersion::Yolov8));
        assert_eq!(schema.nms, Some(NmsMode::ClassAgnostic));
        assert_eq!(schema.input.as_ref().unwrap().shape, vec![1, 3, 640, 640]);

        // Four logical outputs: boxes (split xy/wh), scores, mask_coefs, protos.
        assert_eq!(schema.outputs.len(), 4);
        let boxes = &schema.outputs[0];
        assert_eq!(boxes.type_, LogicalType::Boxes);
        assert_eq!(boxes.encoding, Some(BoxEncoding::Direct));
        assert_eq!(boxes.normalized, Some(true));
        assert_eq!(boxes.shape, vec![1, 4, 8400, 1]); // 4D with padding
        assert_eq!(boxes.outputs.len(), 2);
        assert_eq!(boxes.outputs[0].type_, PhysicalType::BoxesXy);
        assert_eq!(boxes.outputs[1].type_, PhysicalType::BoxesWh);
        // xy quant: scale 0.004177791997790337, zp -122, int8
        let q_xy = boxes.outputs[0].quantization.as_ref().unwrap();
        assert_eq!(q_xy.dtype, Some(DType::Int8));
        assert!((q_xy.scale[0] - 0.004_177_792).abs() < 1e-6);
        assert_eq!(q_xy.zero_point_at(0), -122);

        let scores = &schema.outputs[1];
        assert_eq!(scores.type_, LogicalType::Scores);
        assert_eq!(scores.score_format, Some(ScoreFormat::PerClass));
        assert_eq!(scores.shape, vec![1, 80, 8400, 1]);

        let mask_coefs = &schema.outputs[2];
        assert_eq!(mask_coefs.type_, LogicalType::MaskCoefs);
        assert_eq!(mask_coefs.shape, vec![1, 32, 8400, 1]);

        let protos = &schema.outputs[3];
        assert_eq!(protos.type_, LogicalType::Protos);
        assert_eq!(protos.shape, vec![1, 32, 160, 160]);

        // Schema-level validation passes.
        schema.validate().expect("ARA-2 int8 validate");
    }

    #[test]
    fn parse_real_ara2_int16_dvm_metadata() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/ara2_int16_edgefirst.json"
        ));
        let schema = SchemaV2::parse_json(json).expect("ARA-2 int16 parse");
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.outputs.len(), 4);
        let boxes = &schema.outputs[0];
        assert_eq!(boxes.outputs.len(), 2);
        let q_xy = boxes.outputs[0].quantization.as_ref().unwrap();
        assert_eq!(q_xy.dtype, Some(DType::Int16));
        assert!((q_xy.scale[0] - 3.211_570_6e-5).abs() < 1e-10);
        assert_eq!(q_xy.zero_point_at(0), 0);
        // Mask coefs and protos too are INT16 in this build.
        let mc_q = schema.outputs[2].quantization.as_ref().unwrap();
        assert_eq!(mc_q.dtype, Some(DType::Int16));
        schema.validate().expect("ARA-2 int16 validate");
    }

    #[test]
    fn parse_yaml_with_explicit_schema_version_2() {
        let yaml = r#"
schema_version: 2
outputs:
  - name: scores
    type: scores
    shape: [1, 80, 8400]
    dtype: int8
    quantization:
      scale: 0.00392
      dtype: int8
    decoder: ultralytics
    score_format: per_class
"#;
        let schema = SchemaV2::parse_yaml(yaml).unwrap();
        assert_eq!(schema.schema_version, 2);
        assert_eq!(schema.outputs[0].score_format, Some(ScoreFormat::PerClass));
    }

    // ─── squeeze_padding_dims / to_legacy_config_outputs regressions ────

    #[test]
    fn squeeze_padding_dims_preserves_shape_when_dshape_absent() {
        // Empty dshape must pass shape through untouched. The previous
        // `zip` implementation silently truncated to `[]`, which made
        // every v2 logical output without named dims arrive at the legacy
        // verifier with `shape: []` and fail rank checks.
        let (shape, dshape) = squeeze_padding_dims(vec![1, 4, 8400], vec![]);
        assert_eq!(shape, vec![1, 4, 8400]);
        assert!(dshape.is_empty());
    }

    #[test]
    fn to_legacy_preserves_shape_for_v2_split_boxes_without_dshape() {
        // Regression: `Decoder({...v2 split boxes, shape:[1,4,8400], no dshape...})`
        // used to fail with `Invalid Yolo Split Boxes shape []` because
        // `squeeze_padding_dims` truncated shape when dshape was empty.
        let j = r#"{
          "schema_version": 2,
          "outputs": [
            {"name":"boxes","type":"boxes","shape":[1,4,8400],
             "dtype":"float32","decoder":"ultralytics","encoding":"direct"},
            {"name":"scores","type":"scores","shape":[1,80,8400],
             "dtype":"float32","decoder":"ultralytics","score_format":"per_class"}
          ]
        }"#;
        let schema = SchemaV2::parse_json(j).unwrap();
        let legacy = schema.to_legacy_config_outputs().expect("lowers cleanly");
        let boxes = match &legacy.outputs[0] {
            crate::ConfigOutput::Boxes(b) => b,
            other => panic!("expected Boxes, got {other:?}"),
        };
        assert_eq!(boxes.shape, vec![1, 4, 8400]);
        let scores = match &legacy.outputs[1] {
            crate::ConfigOutput::Scores(s) => s,
            other => panic!("expected Scores, got {other:?}"),
        };
        assert_eq!(scores.shape, vec![1, 80, 8400]);
    }
}
