// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Ultralytics YOLO schema inference from raw model I/O signals.
//!
//! Model exports (ONNX, TFLite, CoreML) carry Ultralytics-authored
//! metadata (class names, task, input size) alongside the tensor shapes
//! and dtypes the runtime reports. This module turns those signals into
//! a [`crate::schema::SchemaV2`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::configs::DimName;
use crate::schema::{
    BoxEncoding, DType, DecoderKind, DecoderVersion, InputSpec, LogicalOutput, LogicalType,
    NmsMode, Quantization, SchemaV2, ScoreFormat,
};

/// A single input or output tensor as reported by the inference runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name as reported by the runtime.
    pub name: String,
    /// Tensor shape in the model's native layout.
    pub shape: Vec<usize>,
    /// Tensor element dtype.
    pub dtype: DType,
    /// Quantization parameters, if the tensor is quantized.
    pub quantization: Option<Quantization>,
}

edgefirst_tensor::ef_vocabulary! {
    /// Container format of the model the signals were read from. Drives
    /// Ultralytics per-format conventions (box normalization, tensor order).
    ///
    /// The code is the one the C `EF_MODEL_SOURCE_*` enumerators use; the
    /// string form is the one the Python bindings accept.
    pub enum ModelSource {
        /// An ONNX export. Ultralytics emits pixel-space box coordinates;
        /// the inferred schema sets `normalized: false`.
        Onnx = 0, "onnx", ONNX,
        /// A TFLite/LiteRT export. Ultralytics emits boxes normalized to
        /// `[0, 1]`; the inferred schema sets `normalized: true`.
        TfLite = 1, "tflite", TFLITE,
        /// A CoreML export (`.mlpackage` / `.mlmodelc`). A no-NMS export
        /// emits pixel-space box coordinates; the inferred schema sets
        /// `normalized: false`, as for [`ModelSource::Onnx`].
        ///
        /// A `nms=True` export (`image`/`iouThreshold`/`confidenceThreshold`
        /// inputs, dynamically-shaped `confidence`/`coordinates` outputs) is
        /// not classified: inference returns an error for it.
        CoreMl = 3, "coreml", COREML,
        /// Any other container. Inference refuses this with
        /// [`InferError::UnknownBoxConvention`].
        Other = 2, "other", OTHER,
    }
    #[doc(hidden)]
    pub mod model_source_code;
}

/// Raw model I/O signals: tensor shapes/dtypes plus unparsed metadata, as
/// reported by the inference runtime that loaded the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSignals {
    /// Container format the signals were read from.
    pub source: ModelSource,
    /// Input tensors.
    pub inputs: Vec<TensorInfo>,
    /// Output tensors.
    pub outputs: Vec<TensorInfo>,
    /// Raw model metadata key/values (ONNX metadata_props, TFLite
    /// metadata entries, CoreML's own metadata alongside Ultralytics'
    /// props). Values are passed verbatim; parsing happens here.
    pub metadata: BTreeMap<String, String>,
}

/// Result of inferring a schema from [`ModelSignals`].
#[derive(Debug, Clone, PartialEq)]
pub struct InferredSchema {
    /// The inferred schema.
    pub schema: SchemaV2,
    /// Class names, in index order.
    pub labels: Vec<String>,
    /// Human-readable summary, e.g. "Ultralytics YOLO26 segment, 80 classes".
    pub description: String,
}

/// Errors that can occur while inferring an Ultralytics schema.
#[derive(Debug)]
pub enum InferError {
    /// Model metadata carries no Ultralytics signature.
    NotUltralytics(String),
    /// The Ultralytics `names` metadata field could not be parsed.
    BadNames(String),
    /// A metadata field is present but carries a value that cannot be
    /// interpreted.
    BadMetadata(String),
    /// The Ultralytics `task` metadata field names an unsupported task.
    UnsupportedTask(String),
    /// The number of classes in `names` does not match what the output
    /// tensor shapes imply.
    ClassCountMismatch { expected: usize, found: usize },
    /// The output tensor layout does not match any supported Ultralytics
    /// convention.
    UnsupportedLayout(String),
    /// More than one output matches the same layout.
    AmbiguousLayout(String),
    /// A boundary tensor carries quantization the decoder cannot consume:
    /// per-channel quantization, no scale, or zero points that do not pair
    /// with the scale.
    UnsupportedQuantization(String),
    /// The signals came from [`ModelSource::Other`], which has no known
    /// box coordinate convention. The known conventions are ONNX
    /// (pixel-space), TFLite (`[0, 1]`) and CoreML (pixel-space).
    UnknownBoxConvention,
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::NotUltralytics(s) => {
                write!(f, "model metadata carries no Ultralytics signature: {s}")
            }
            InferError::BadNames(s) => {
                write!(f, "cannot parse Ultralytics `names` metadata: {s}")
            }
            InferError::BadMetadata(s) => {
                write!(f, "malformed Ultralytics metadata: {s}")
            }
            InferError::UnsupportedTask(s) => write!(
                f,
                "unsupported Ultralytics task `{s}` (supported: detect, segment)"
            ),
            InferError::ClassCountMismatch { expected, found } => write!(
                f,
                "class-count mismatch: names give {expected} classes but output features imply {found}"
            ),
            InferError::UnsupportedLayout(s) => write!(f, "unsupported output layout: {s}"),
            InferError::AmbiguousLayout(s) => write!(f, "ambiguous output layout: {s}"),
            InferError::UnsupportedQuantization(s) => {
                write!(f, "unsupported quantization: {s}")
            }
            InferError::UnknownBoxConvention => write!(
                f,
                "box coordinate convention is unknown for this container; only \
                 onnx (pixel-space), tflite ([0,1]), and coreml (pixel-space) \
                 are characterized"
            ),
        }
    }
}

impl std::error::Error for InferError {}

/// Supported Ultralytics export tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    Detect,
    Segment,
}

impl Task {
    fn parse(s: &str) -> Result<Self, InferError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "detect" => Ok(Task::Detect),
            "segment" => Ok(Task::Segment),
            other => Err(InferError::UnsupportedTask(other.to_string())),
        }
    }
}

/// Parsed Ultralytics export metadata, in whichever raw form (ONNX flat
/// string props, or TFLite `metadata.json` envelope) it was captured.
#[derive(Debug, Clone, PartialEq)]
struct UltralyticsMeta {
    names: Vec<String>,
    task: Option<Task>,
    /// YOLO26's built-in NMS-free end-to-end head signal. `Some(true)` for
    /// YOLO26 exports, `Some(false)` for NMS-required heads (v8/v11),
    /// `None` when the metadata omits the field entirely.
    end2end: Option<bool>,
}

/// Returns an error when `author` or `docs` is present, non-empty and
/// does not name Ultralytics. Absent fields are accepted.
fn reject_foreign_vendor(author: Option<&String>, docs: Option<&String>) -> Result<(), InferError> {
    let names_ultralytics = |v: &String| v.to_ascii_lowercase().contains("ultralytics");
    for (field, value) in [("author", author), ("docs", docs)] {
        if let Some(v) = value {
            if !v.trim().is_empty() && !names_ultralytics(v) {
                return Err(InferError::NotUltralytics(format!(
                    "metadata `{field}` is `{v}`, not Ultralytics"
                )));
            }
        }
    }
    Ok(())
}

impl UltralyticsMeta {
    /// Extracts Ultralytics metadata from a model's raw metadata map.
    ///
    /// Handles two forms: flat string props (ONNX `metadata_props`, one
    /// key per field) and the TFLite `metadata.json` envelope (a single
    /// entry whose *value* is a JSON document carrying the same fields).
    fn from_metadata(meta: &BTreeMap<String, String>) -> Result<Self, InferError> {
        if let Some(names_raw) = meta.get("names") {
            reject_foreign_vendor(meta.get("author"), meta.get("docs"))?;
            return Self::from_flat_props(meta, names_raw);
        }

        // TFLite form: scan every metadata value for a JSON object that
        // carries a `names` field.
        for value in meta.values() {
            if let Ok(serde_json::Value::Object(obj)) = serde_json::from_str(value) {
                if let Some(names) = obj.get("names") {
                    let author = obj
                        .get("author")
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    let docs = obj.get("docs").and_then(|v| v.as_str()).map(str::to_string);
                    reject_foreign_vendor(author.as_ref(), docs.as_ref())?;
                    return Self::from_json_envelope(&obj, names);
                }
            }
        }

        Err(InferError::NotUltralytics(
            "no `names` metadata field or metadata.json envelope found".into(),
        ))
    }

    fn from_flat_props(
        meta: &BTreeMap<String, String>,
        names_raw: &str,
    ) -> Result<Self, InferError> {
        let names = parse_names(names_raw)?;
        let task = meta.get("task").map(|t| Task::parse(t)).transpose()?;
        let end2end = meta.get("end2end").map(|s| parse_bool_str(s)).transpose()?;
        Ok(UltralyticsMeta {
            names,
            task,
            end2end,
        })
    }

    /// Builds the metadata from a JSON envelope object and its `names`
    /// value.
    fn from_json_envelope(
        obj: &serde_json::Map<String, serde_json::Value>,
        names_value: &serde_json::Value,
    ) -> Result<Self, InferError> {
        let names = names_from_json_value(names_value)?;
        let task = obj
            .get("task")
            .and_then(|v| v.as_str())
            .map(Task::parse)
            .transpose()?;
        // A non-boolean `end2end` is an error.
        let end2end = match obj.get("end2end") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Bool(b)) => Some(*b),
            Some(other) => {
                return Err(InferError::BadMetadata(format!(
                    "`end2end` is {other}, expected a boolean"
                )))
            }
        };
        Ok(UltralyticsMeta {
            names,
            task,
            end2end,
        })
    }
}

/// Parses an Ultralytics `end2end` metadata string value (`"True"` /
/// `"False"`, case-insensitively).
///
/// Any other value is an error.
fn parse_bool_str(s: &str) -> Result<bool, InferError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(InferError::BadMetadata(format!(
            "`end2end` is `{other}`, expected true or false"
        ))),
    }
}

/// Parses the Ultralytics `names` metadata field, which appears in two
/// forms: a stringified Python dict (`"{0: 'person', 1: 'bicycle'}"`, as
/// captured verbatim from ONNX `metadata_props`) or a JSON object with
/// string keys (`{"0": "person", "1": "bicycle"}`).
fn parse_names(s: &str) -> Result<Vec<String>, InferError> {
    let trimmed = s.trim();

    if let Ok(map) = serde_json::from_str::<BTreeMap<String, String>>(trimmed) {
        return names_from_str_keyed_map(map);
    }

    let body = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| InferError::BadNames(format!("not a dict: {trimmed}")))?;

    let mut indexed: BTreeMap<usize, String> = BTreeMap::new();
    for item in split_top_level_commas(body) {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (key, value) = item
            .split_once(':')
            .ok_or_else(|| InferError::BadNames(format!("malformed entry `{item}`")))?;
        let idx: usize = key
            .trim()
            .parse()
            .map_err(|_| InferError::BadNames(format!("non-numeric key `{key}`")))?;
        let value = unquote(value.trim())
            .ok_or_else(|| InferError::BadNames(format!("unquoted value `{value}`")))?;
        indexed.insert(idx, value);
    }
    contiguous_values(indexed)
}

/// Converts a `names` JSON value (a JSON object keyed by stringified
/// index) into ordered class names.
fn names_from_json_value(v: &serde_json::Value) -> Result<Vec<String>, InferError> {
    let obj = v
        .as_object()
        .ok_or_else(|| InferError::BadNames("`names` is not a JSON object".into()))?;
    let mut map = BTreeMap::new();
    for (k, val) in obj {
        let s = val
            .as_str()
            .ok_or_else(|| InferError::BadNames(format!("non-string name for key `{k}`")))?;
        map.insert(k.clone(), s.to_string());
    }
    names_from_str_keyed_map(map)
}

/// Converts a string-keyed names map (JSON form) into ordered class names
/// by parsing keys as indices.
fn names_from_str_keyed_map(map: BTreeMap<String, String>) -> Result<Vec<String>, InferError> {
    let mut indexed: BTreeMap<usize, String> = BTreeMap::new();
    for (k, v) in map {
        let idx: usize = k
            .parse()
            .map_err(|_| InferError::BadNames(format!("non-numeric key `{k}`")))?;
        indexed.insert(idx, v);
    }
    contiguous_values(indexed)
}

/// Requires a `0..n` contiguous index range and returns the values in
/// index order.
fn contiguous_values(indexed: BTreeMap<usize, String>) -> Result<Vec<String>, InferError> {
    if indexed.is_empty() {
        return Err(InferError::BadNames("class map is empty".into()));
    }
    let n = indexed.len();
    for i in 0..n {
        if !indexed.contains_key(&i) {
            return Err(InferError::BadNames("non-contiguous class indices".into()));
        }
    }
    Ok(indexed.into_values().collect())
}

/// Splits a dict body on top-level commas; commas inside `'...'` and
/// `"..."` spans do not split.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for c in s.chars() {
        match quote {
            Some(q) => {
                current.push(c);
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' => {
                    quote = Some(c);
                    current.push(c);
                }
                ',' => parts.push(std::mem::take(&mut current)),
                _ => current.push(c),
            },
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Strips matching outer `'` or `"` quotes from a value, returning `None`
/// if the value isn't quoted.
fn unquote(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if s.len() >= 2 {
        let first = bytes[0];
        let last = bytes[s.len() - 1];
        if (first == b'\'' && last == b'\'') || (first == b'"' && last == b'"') {
            return Some(s[1..s.len() - 1].to_string());
        }
    }
    None
}

/// Classifies a rank-4 model input's spatial layout.
///
/// Channel dimension is wherever the shape holds a `3`: index 1 (NCHW) or
/// index 3 (NHWC). [`ModelSource`] is not consulted.
///
/// Batch must be 1.
#[allow(clippy::type_complexity)]
fn classify_input_layout(
    input: &TensorInfo,
) -> Result<(usize, usize, Vec<(DimName, usize)>), InferError> {
    debug_assert_eq!(
        input.shape.len(),
        4,
        "caller selects the input by rank-4; see infer_ultralytics_schema"
    );
    if input.shape[0] != 1 {
        return Err(InferError::UnsupportedLayout(format!(
            "model input `{}` has batch {} but every supported output layout \
             is batch-1; shape {:?}",
            input.name, input.shape[0], input.shape
        )));
    }
    if input.shape[1] == 3 {
        let (h, w) = (input.shape[2], input.shape[3]);
        Ok((
            h,
            w,
            vec![
                (DimName::Batch, input.shape[0]),
                (DimName::NumFeatures, 3),
                (DimName::Height, h),
                (DimName::Width, w),
            ],
        ))
    } else if input.shape[3] == 3 {
        let (h, w) = (input.shape[1], input.shape[2]);
        Ok((
            h,
            w,
            vec![
                (DimName::Batch, input.shape[0]),
                (DimName::Height, h),
                (DimName::Width, w),
                (DimName::NumFeatures, 3),
            ],
        ))
    } else {
        Err(InferError::UnsupportedLayout(format!(
            "cannot locate the channel dimension (=3) in input shape {:?}",
            input.shape
        )))
    }
}

/// If `shape` is rank-3 with a unit leading (batch) dim and one of its
/// remaining two dims equals `expected_anchors`, returns the other dim's
/// value and its index (1 or 2) within `shape`. `None` if no dim matches
/// or the leading dim isn't 1.
fn find_anchors_dim(shape: &[usize], expected_anchors: usize) -> Option<(usize, usize)> {
    if shape.len() != 3 || shape[0] != 1 {
        return None;
    }
    if shape[2] == expected_anchors {
        Some((shape[1], 1)) // features candidate at index 1: features-first
    } else if shape[1] == expected_anchors {
        Some((shape[2], 2)) // features candidate at index 2: anchors-first
    } else {
        None
    }
}

/// Classifies the combined pre-NMS detection layout: a rank-3 output whose
/// dims contain both `expected_anchors` and `feat` (`4 + nc`).
///
/// Every rank-3 output is scanned. A tensor that carries the anchor count
/// but the wrong feature width yields [`InferError::ClassCountMismatch`]
/// only when no output matches; two matching outputs yield
/// [`InferError::AmbiguousLayout`].
#[allow(clippy::type_complexity)]
fn classify_pre_nms<'a>(
    rank3_outputs: &[&'a TensorInfo],
    expected_anchors: usize,
    feat: usize,
    nc: usize,
) -> Result<(&'a TensorInfo, Vec<(DimName, usize)>), InferError> {
    let mut mismatch: Option<InferError> = None;
    let mut matched: Option<(&'a TensorInfo, Vec<(DimName, usize)>)> = None;
    for out in rank3_outputs {
        let Some((other_dim, feat_idx)) = find_anchors_dim(&out.shape, expected_anchors) else {
            continue;
        };
        if other_dim == feat {
            let dshape = if feat_idx == 1 {
                vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, feat),
                    (DimName::NumBoxes, expected_anchors),
                ]
            } else {
                vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, expected_anchors),
                    (DimName::NumFeatures, feat),
                ]
            };
            // Two outputs of the same detection shape are ambiguous.
            if let Some((first, _)) = &matched {
                return Err(InferError::AmbiguousLayout(format!(
                    "outputs `{}` and `{}` both match the pre-NMS detection layout \
                     ({:?}); cannot tell which carries the detections",
                    first.name, out.name, out.shape
                )));
            }
            matched = Some((out, dshape));
            continue;
        }
        mismatch.get_or_insert_with(|| InferError::ClassCountMismatch {
            expected: nc,
            // `feat - nc` is the non-class base width: `4` for detection,
            // `4 + k` for segmentation (proto channels).
            found: other_dim.saturating_sub(feat.saturating_sub(nc)),
        });
    }
    if let Some(found) = matched {
        return Ok(found);
    }
    Err(mismatch.unwrap_or_else(|| {
        InferError::UnsupportedLayout(format!(
            "no output matches the Ultralytics pre-NMS layout (expected a dim == \
             {expected_anchors} anchors and a dim == {feat} features [4 + {nc} classes]); \
             outputs seen: {:?}",
            rank3_outputs.iter().map(|o| &o.shape).collect::<Vec<_>>()
        ))
    }))
}

/// Finds an end-to-end (YOLO26) candidate: a rank-3 output shaped
/// `[1, N, feat]`, where `feat` is `6` for detection or `6 + k` for
/// segmentation (`k` being the proto channel count) and
/// `0 < N <= expected_anchors`. `N` is the export's `max_det`. More than
/// one match is [`InferError::AmbiguousLayout`].
fn find_e2e_candidate<'a>(
    rank3_outputs: &[&'a TensorInfo],
    expected_anchors: usize,
    feat: usize,
) -> Result<Option<(&'a TensorInfo, usize)>, InferError> {
    let matches: Vec<(&'a TensorInfo, usize)> = rank3_outputs
        .iter()
        .filter_map(|o| {
            let s = &o.shape;
            debug_assert_eq!(s.len(), 3, "caller filters to rank-3 outputs");
            (s[0] == 1 && s[2] == feat && s[1] > 0 && s[1] <= expected_anchors)
                .then_some((*o, s[1]))
        })
        .collect();
    // Two outputs of the same end-to-end shape are ambiguous.
    if matches.len() > 1 {
        return Err(InferError::AmbiguousLayout(format!(
            "{} outputs match the end-to-end layout [1, N, {feat}]: {:?}; \
             cannot tell which carries the detections",
            matches.len(),
            matches.iter().map(|(o, _)| &o.name).collect::<Vec<_>>()
        )));
    }
    Ok(matches.into_iter().next())
}

/// Classifies a rank-4 proto tensor: the mask-prototype output paired with
/// a segmentation detection tensor. Expected spatial dims are `(H/4, W/4)`
/// (Ultralytics proto stride 4); the remaining non-batch dim is the proto
/// channel count `k`. Supports both `[1, k, H/4, W/4]` (NCHW) and
/// `[1, H/4, W/4, k]` (NHWC).
///
/// The NCHW check runs first: a shape that fits both layouts (e.g.
/// `[1, 160, 160, 160]` on a 640x640 input) resolves as NCHW.
fn classify_proto(
    proto: &TensorInfo,
    h: usize,
    w: usize,
) -> Result<(usize, Vec<(DimName, usize)>), InferError> {
    let (eh, ew) = (h / 4, w / 4);
    let s = &proto.shape;
    if s.len() != 4 || s[0] != 1 {
        return Err(InferError::UnsupportedLayout(format!(
            "expected a rank-4 proto tensor with batch 1, found shape {s:?}"
        )));
    }
    if s[2] == eh && s[3] == ew {
        let k = s[1];
        Ok((
            k,
            vec![
                (DimName::Batch, 1),
                (DimName::NumProtos, k),
                (DimName::Height, eh),
                (DimName::Width, ew),
            ],
        ))
    } else if s[1] == eh && s[2] == ew {
        let k = s[3];
        Ok((
            k,
            vec![
                (DimName::Batch, 1),
                (DimName::Height, eh),
                (DimName::Width, ew),
                (DimName::NumProtos, k),
            ],
        ))
    } else {
        Err(InferError::UnsupportedLayout(format!(
            "proto tensor shape {s:?} doesn't match the expected ({eh}, {ew}) spatial dims"
        )))
    }
}

/// Default field values for a [`LogicalOutput`] whose semantic fields are
/// filled in by the caller; every field not overridden is `None` or empty.
fn default_logical() -> LogicalOutput {
    LogicalOutput {
        name: None,
        type_: None,
        shape: Vec::new(),
        dshape: Vec::new(),
        decoder: None,
        encoding: None,
        score_format: None,
        normalized: None,
        anchors: None,
        stride: None,
        dtype: None,
        quantization: None,
        outputs: Vec::new(),
        activation_applied: None,
        activation_required: None,
        semantic: Default::default(),
    }
}

/// Rejects an integer-dtype boundary tensor that carries no quantization
/// parameters, and a quantization that is per-channel, has no scale, or
/// has zero points that do not pair with the scale.
fn check_quantized_boundary(tensor: &TensorInfo) -> Result<(), InferError> {
    if tensor.dtype.is_integer() && tensor.quantization.is_none() {
        return Err(InferError::UnsupportedLayout(format!(
            "output `{}` has integer dtype {:?} but no quantization params",
            tensor.name, tensor.dtype
        )));
    }
    if let Some(q) = &tensor.quantization {
        if q.scale.len() > 1 {
            return Err(InferError::UnsupportedQuantization(format!(
                "output `{}` carries per-channel quantization ({} scales); \
                 the decoder supports per-tensor quantization only",
                tensor.name,
                q.scale.len()
            )));
        }
        if q.scale.is_empty() {
            return Err(InferError::UnsupportedQuantization(format!(
                "output `{}` is quantized but carries no scale",
                tensor.name
            )));
        }
        // Zero points, when present, pair with the scale.
        if let Some(zp) = &q.zero_point {
            if zp.len() != q.scale.len() {
                return Err(InferError::UnsupportedQuantization(format!(
                    "output `{}` has {} scale(s) but {} zero point(s)",
                    tensor.name,
                    q.scale.len(),
                    zp.len()
                )));
            }
        }
    }
    Ok(())
}

/// Infers an Ultralytics YOLO [`SchemaV2`] from raw model I/O signals and
/// metadata.
///
/// Detection and segmentation are both supported. Segmentation is
/// recognised either by metadata (`task == segment`) or by the presence of
/// a rank-4 proto tensor; a `segment` task with no proto, or a `detect`
/// task with a proto present, is an error.
///
/// # Examples
///
/// A two-class YOLOv8 detection export, as an inference runtime would
/// report it — a rank-4 input, one `[1, 4 + nc, anchors]` output, and the
/// metadata the exporter embedded:
///
/// ```
/// use std::collections::BTreeMap;
///
/// use edgefirst_decoder::schema::DType;
/// use edgefirst_decoder::{
///     infer_ultralytics_schema, ModelSignals, ModelSource, TensorInfo,
/// };
///
/// let tensor = |name: &str, shape: &[usize]| TensorInfo {
///     name: name.to_string(),
///     shape: shape.to_vec(),
///     dtype: DType::Float32,
///     quantization: None,
/// };
///
/// let mut metadata = BTreeMap::new();
/// metadata.insert("task".to_string(), "detect".to_string());
/// metadata.insert("end2end".to_string(), "False".to_string());
/// metadata.insert("names".to_string(), "{0: 'person', 1: 'bicycle'}".to_string());
///
/// let signals = ModelSignals {
///     source: ModelSource::Onnx,
///     inputs: vec![tensor("images", &[1, 3, 640, 640])],
///     // 8400 anchors for a 640x640 input (80² + 40² + 20²);
///     // 6 features = 4 box coordinates + 2 classes.
///     outputs: vec![tensor("output0", &[1, 6, 8400])],
///     metadata,
/// };
///
/// let inferred = infer_ultralytics_schema(&signals)?;
/// assert_eq!(inferred.labels, ["person", "bicycle"]);
/// assert_eq!(inferred.description, "Ultralytics YOLOv8/11 detect, 2 classes");
///
/// // Hand the schema to `DecoderBuilder` as `edgefirst.json` v2 JSON.
/// let config_json = serde_json::to_string(&inferred.schema).unwrap();
/// assert!(config_json.contains("\"decoder_version\":\"yolov8\""));
/// # Ok::<(), edgefirst_decoder::InferError>(())
/// ```
///
/// Signals that carry no Ultralytics metadata are refused with a typed
/// error:
///
/// ```
/// use std::collections::BTreeMap;
///
/// use edgefirst_decoder::{
///     infer_ultralytics_schema, InferError, ModelSignals, ModelSource,
/// };
///
/// let signals = ModelSignals {
///     source: ModelSource::Onnx,
///     inputs: Vec::new(),
///     outputs: Vec::new(),
///     metadata: BTreeMap::new(),
/// };
///
/// assert!(matches!(
///     infer_ultralytics_schema(&signals),
///     Err(InferError::NotUltralytics(_))
/// ));
/// ```
pub fn infer_ultralytics_schema(signals: &ModelSignals) -> Result<InferredSchema, InferError> {
    infer_from_meta(signals, UltralyticsMeta::from_metadata(&signals.metadata)?)
}

/// Infers the schema, with no labels, from a class count other than the one
/// `names` gives, and returns that count. `classes`, the count the caller
/// states, is tried first; otherwise each anchor-shaped output proposes a
/// count and exactly one must lay out, else the layout is ambiguous.
pub(crate) fn infer_ultralytics_schema_ignoring_names(
    signals: &ModelSignals,
    classes: Option<usize>,
) -> Result<(InferredSchema, usize), InferError> {
    let meta = UltralyticsMeta::from_metadata(&signals.metadata)?;
    let with_count = |nc: usize| {
        let names = (0..nc).map(|i| format!("class{i}")).collect();
        let candidate = UltralyticsMeta {
            names,
            task: meta.task,
            end2end: meta.end2end,
        };
        infer_from_meta(signals, candidate)
    };
    const REMEDY: &str = "supply a labels override naming every class, or describe the \
                          outputs in an edgefirst.json beside the model";
    let stated = classes
        .filter(|&nc| nc > 0)
        .and_then(|nc| with_count(nc).ok());
    let mut found: Option<InferredSchema> = stated;
    if found.is_none() {
        for nc in candidate_class_counts(signals)? {
            let inferred = match with_count(nc) {
                Ok(inferred) => inferred,
                Err(InferError::AmbiguousLayout(why)) => {
                    return Err(InferError::AmbiguousLayout(format!("{why}; {REMEDY}")))
                }
                Err(_) => continue,
            };
            if let Some(first) = &found {
                return Err(InferError::AmbiguousLayout(format!(
                    "`names` miscounts the classes and the outputs allow both {} and {nc}; \
                     {REMEDY}",
                    first.labels.len()
                )));
            }
            found = Some(inferred);
        }
    }
    let mut inferred = found.ok_or_else(|| {
        InferError::UnsupportedLayout("no output carries a usable class count".into())
    })?;
    let nc = std::mem::take(&mut inferred.labels).len();
    Ok((inferred, nc))
}

/// The class counts the anchor-shaped outputs imply, ascending.
fn candidate_class_counts(signals: &ModelSignals) -> Result<BTreeSet<usize>, InferError> {
    let input = signals
        .inputs
        .iter()
        .find(|t| t.shape.len() == 4)
        .ok_or_else(|| InferError::UnsupportedLayout("no rank-4 model input found".into()))?;
    let (h, w, _) = classify_input_layout(input)?;
    let k = match signals.outputs.iter().find(|o| o.shape.len() == 4) {
        Some(proto) => classify_proto(proto, h, w)?.0,
        None => 0,
    };
    Ok(signals
        .outputs
        .iter()
        .filter_map(|o| find_anchors_dim(&o.shape, anchor_count(h, w)))
        .filter_map(|(features, _)| features.checked_sub(4 + k))
        .filter(|&nc| nc > 0)
        .collect())
}

/// Anchors over the stride 8, 16 and 32 grids.
fn anchor_count(h: usize, w: usize) -> usize {
    (h / 8) * (w / 8) + (h / 16) * (w / 16) + (h / 32) * (w / 32)
}

fn infer_from_meta(
    signals: &ModelSignals,
    meta: UltralyticsMeta,
) -> Result<InferredSchema, InferError> {
    let proto = signals.outputs.iter().find(|o| o.shape.len() == 4);
    match (meta.task, proto.is_some()) {
        (Some(Task::Segment), false) => {
            return Err(InferError::UnsupportedLayout(
                "segment task but no proto tensor".into(),
            ));
        }
        (Some(Task::Detect), true) => {
            return Err(InferError::UnsupportedLayout(
                "detect task but proto tensor present".into(),
            ));
        }
        _ => {}
    }
    let is_segment = meta.task == Some(Task::Segment) || proto.is_some();

    let input = signals
        .inputs
        .iter()
        .find(|t| t.shape.len() == 4)
        .ok_or_else(|| {
            InferError::UnsupportedLayout(format!(
                "no rank-4 model input found; inputs seen: {:?}",
                signals.inputs.iter().map(|t| &t.shape).collect::<Vec<_>>()
            ))
        })?;
    let (h, w, input_dshape) = classify_input_layout(input)?;

    let expected_anchors = anchor_count(h, w);
    let nc = meta.names.len();

    // `k` is the proto channel count (0 without a proto): detection
    // features are `4 + nc + k` wide, end-to-end features `6 + k`.
    let (k, proto_dshape) = match proto {
        Some(proto) => {
            let (k, dshape) = classify_proto(proto, h, w)?;
            (k, Some(dshape))
        }
        None => (0, None),
    };
    let feat = 4 + nc + k;
    let e2e_feat = 6 + k;

    let rank3_outputs: Vec<&TensorInfo> = signals
        .outputs
        .iter()
        .filter(|o| o.shape.len() == 3)
        .collect();

    // The `end2end` metadata flag selects the layout; when it is absent
    // the shapes decide.
    let (out, dshape, version) = match meta.end2end {
        Some(true) => {
            let (out, n) = find_e2e_candidate(&rank3_outputs, expected_anchors, e2e_feat)?
                .ok_or_else(|| {
                    InferError::UnsupportedLayout(format!(
                        "metadata declares end2end=true but no output matches the YOLO26 \
                         end-to-end layout [1, N<={expected_anchors}, {e2e_feat}]; \
                         outputs seen: {:?}",
                        rank3_outputs.iter().map(|o| &o.shape).collect::<Vec<_>>()
                    ))
                })?;
            (
                out,
                vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, n),
                    (DimName::NumFeatures, e2e_feat),
                ],
                DecoderVersion::Yolo26,
            )
        }
        Some(false) => {
            let (out, dshape) = classify_pre_nms(&rank3_outputs, expected_anchors, feat, nc)?;
            (out, dshape, DecoderVersion::Yolov8)
        }
        None => match classify_pre_nms(&rank3_outputs, expected_anchors, feat, nc) {
            Ok((out, dshape)) => (out, dshape, DecoderVersion::Yolov8),
            Err(err @ InferError::ClassCountMismatch { .. }) => return Err(err),
            Err(_) => {
                let (out, n) = find_e2e_candidate(&rank3_outputs, expected_anchors, e2e_feat)?
                    .ok_or_else(|| {
                        InferError::UnsupportedLayout(format!(
                            "no output matches a known Ultralytics detection layout \
                             (pre-NMS or end-to-end); outputs seen: {:?}",
                            rank3_outputs.iter().map(|o| &o.shape).collect::<Vec<_>>()
                        ))
                    })?;
                (
                    out,
                    vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, n),
                        (DimName::NumFeatures, e2e_feat),
                    ],
                    DecoderVersion::Yolo26,
                )
            }
        },
    };

    check_quantized_boundary(out)?;
    if let Some(proto) = proto {
        check_quantized_boundary(proto)?;
    }

    // Box normalization follows the container: `[0, 1]` for TFLite,
    // pixel-space for ONNX and CoreML; `Other` is refused.
    let normalized = match signals.source {
        ModelSource::TfLite => true,
        ModelSource::Onnx | ModelSource::CoreMl => false,
        ModelSource::Other => {
            return Err(InferError::UnknownBoxConvention);
        }
    };
    let det = LogicalOutput {
        name: Some(out.name.clone()),
        // `Detections` is the decoded post-NMS type; `Detection` is the
        // anchor-grid output.
        type_: Some(match version {
            DecoderVersion::Yolo26 => LogicalType::Detections,
            _ => LogicalType::Detection,
        }),
        shape: out.shape.clone(),
        dshape,
        decoder: Some(DecoderKind::Ultralytics),
        encoding: Some(BoxEncoding::Direct), // DFL is decoded in the graph
        score_format: Some(ScoreFormat::PerClass),
        normalized: Some(normalized),
        anchors: None,
        stride: None,
        dtype: Some(out.dtype),
        quantization: out.quantization.clone(),
        outputs: Vec::new(),
        activation_applied: None,
        activation_required: None,
        semantic: Default::default(),
    };

    let mut outputs = vec![det];
    if let (Some(proto), Some(proto_dshape)) = (proto, proto_dshape) {
        outputs.push(LogicalOutput {
            name: Some(proto.name.clone()),
            type_: Some(LogicalType::Protos),
            shape: proto.shape.clone(),
            dshape: proto_dshape,
            decoder: Some(DecoderKind::Ultralytics),
            dtype: Some(proto.dtype),
            quantization: proto.quantization.clone(),
            ..default_logical()
        });
    }

    let schema = SchemaV2 {
        schema_version: 2,
        input: Some(InputSpec {
            shape: input.shape.clone(),
            dshape: input_dshape,
            cameraadaptor: Some("rgb".into()),
            preprocessing: None,
        }),
        outputs,
        // Class-aware NMS for pre-NMS heads; none for YOLO26 end-to-end
        // heads, which run NMS in-graph.
        nms: match version {
            DecoderVersion::Yolo26 => None,
            _ => Some(NmsMode::ClassAware),
        },
        nms_multi_label: None,
        decoder_version: Some(version),
        model: None,
    };

    let version_label = if version == DecoderVersion::Yolo26 {
        "YOLO26"
    } else {
        "YOLOv8/11"
    };
    let task_label = if is_segment { "segment" } else { "detect" };
    let description = format!("Ultralytics {version_label} {task_label}, {nc} classes");

    Ok(InferredSchema {
        schema,
        labels: meta.names,
        description,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Root of the `testdata/infer` fixture directory:
    /// `$EDGEFIRST_TESTDATA_DIR/infer` when the variable is set, else the
    /// manifest-relative path.
    fn infer_fixture_dir() -> std::path::PathBuf {
        std::env::var("EDGEFIRST_TESTDATA_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata")
            })
            .join("infer")
    }

    #[test]
    fn parse_names_python_dict_repr() {
        let s = "{0: 'person', 1: 'bicycle', 2: \"fire hydrant\"}";
        assert_eq!(
            parse_names(s).unwrap(),
            vec!["person", "bicycle", "fire hydrant"]
        );
    }

    #[test]
    fn parse_names_json_object() {
        let s = r#"{"0": "person", "1": "bicycle"}"#;
        assert_eq!(parse_names(s).unwrap(), vec!["person", "bicycle"]);
    }

    #[test]
    fn parse_names_orders_by_index_not_insertion() {
        let s = "{1: 'b', 0: 'a'}";
        assert_eq!(parse_names(s).unwrap(), vec!["a", "b"]);
    }

    #[test]
    fn parse_names_rejects_garbage() {
        assert!(matches!(
            parse_names("not a dict"),
            Err(InferError::BadNames(_))
        ));
    }

    #[test]
    fn parse_names_rejects_non_contiguous_indices() {
        // Python dict-repr form: index 1 is missing.
        match parse_names("{0: 'a', 2: 'b'}") {
            Err(InferError::BadNames(msg)) => {
                assert!(
                    msg.contains("non-contiguous"),
                    "unexpected BadNames message: {msg}"
                );
            }
            other => panic!("expected BadNames, got {other:?}"),
        }

        // JSON-object form: same gap.
        match parse_names(r#"{"0": "a", "2": "b"}"#) {
            Err(InferError::BadNames(msg)) => {
                assert!(
                    msg.contains("non-contiguous"),
                    "unexpected BadNames message: {msg}"
                );
            }
            other => panic!("expected BadNames, got {other:?}"),
        }
    }

    #[test]
    fn meta_from_onnx_props() {
        let mut m = BTreeMap::new();
        m.insert("names".into(), "{0: 'person'}".into());
        m.insert("task".into(), "detect".into());
        let u = UltralyticsMeta::from_metadata(&m).unwrap();
        assert_eq!(u.names, vec!["person"]);
        assert_eq!(u.task, Some(Task::Detect));
    }

    #[test]
    fn meta_task_pose_rejected() {
        let mut m = BTreeMap::new();
        m.insert("names".into(), "{0: 'person'}".into());
        m.insert("task".into(), "pose".into());
        assert!(matches!(
            UltralyticsMeta::from_metadata(&m),
            Err(InferError::UnsupportedTask(_))
        ));
    }

    #[test]
    fn meta_missing_names_is_not_ultralytics() {
        assert!(matches!(
            UltralyticsMeta::from_metadata(&BTreeMap::new()),
            Err(InferError::NotUltralytics(_))
        ));
    }

    #[test]
    fn meta_end2end_true_or_false_from_onnx_flat_props() {
        let mut m = BTreeMap::new();
        m.insert("names".into(), "{0: 'person'}".into());
        m.insert("end2end".into(), "True".into());
        assert_eq!(
            UltralyticsMeta::from_metadata(&m).unwrap().end2end,
            Some(true)
        );

        m.insert("end2end".into(), "False".into());
        assert_eq!(
            UltralyticsMeta::from_metadata(&m).unwrap().end2end,
            Some(false)
        );
    }

    /// Loads a captured fixture's `metadata` map as raw string key/values.
    fn load_fixture_metadata(name: &str) -> BTreeMap<String, String> {
        let path = infer_fixture_dir().join(format!("{name}.signals.json"));
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_str(&content).expect("fixture is valid JSON");
        json["metadata"]
            .as_object()
            .expect("fixture has a metadata object")
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().expect("metadata value is a string").to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn meta_from_real_tflite_metadata_json_envelope() {
        let meta = load_fixture_metadata("yolov8n_float32");
        let u = UltralyticsMeta::from_metadata(&meta).unwrap();
        assert_eq!(u.names.len(), 80);
        assert_eq!(u.task, Some(Task::Detect));
    }

    #[test]
    fn meta_end2end_true_on_real_yolo26_tflite_fixture() {
        let meta = load_fixture_metadata("yolo26n_float32");
        let u = UltralyticsMeta::from_metadata(&meta).unwrap();
        assert_eq!(u.end2end, Some(true));
    }

    #[test]
    fn meta_end2end_false_on_real_yolov8_onnx_fixture() {
        let meta = load_fixture_metadata("yolov8n");
        let u = UltralyticsMeta::from_metadata(&meta).unwrap();
        assert_eq!(u.end2end, Some(false));
    }

    /// Loads a captured fixture into `ModelSignals`: tensor
    /// names/shapes/dtypes plus the raw metadata map.
    fn signals_from_fixture(name: &str) -> ModelSignals {
        let path = infer_fixture_dir().join(format!("{name}.signals.json"));
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
        let json: serde_json::Value =
            serde_json::from_str(&content).expect("fixture is valid JSON");

        let source = match json["source"]
            .as_str()
            .expect("fixture `source` is a string")
        {
            "onnx" => ModelSource::Onnx,
            "tflite" => ModelSource::TfLite,
            "coreml" => ModelSource::CoreMl,
            other => panic!("fixture {name}: unknown source `{other}`"),
        };

        fn parse_dtype(s: &str) -> DType {
            match s {
                "float32" => DType::Float32,
                "float16" => DType::Float16,
                "int8" => DType::Int8,
                "uint8" => DType::Uint8,
                "int16" => DType::Int16,
                "uint16" => DType::Uint16,
                "int32" => DType::Int32,
                "uint32" => DType::Uint32,
                other => panic!("unknown dtype `{other}`"),
            }
        }

        fn parse_tensor(v: &serde_json::Value) -> TensorInfo {
            TensorInfo {
                name: v["name"]
                    .as_str()
                    .expect("tensor `name` is a string")
                    .to_string(),
                shape: v["shape"]
                    .as_array()
                    .expect("tensor `shape` is an array")
                    .iter()
                    .map(|d| d.as_u64().expect("shape dim is a number") as usize)
                    .collect(),
                dtype: parse_dtype(v["dtype"].as_str().expect("tensor `dtype` is a string")),
                // Every captured fixture reports `"quantization": null`.
                quantization: None,
            }
        }

        let inputs = json["inputs"]
            .as_array()
            .expect("fixture `inputs` is an array")
            .iter()
            .map(parse_tensor)
            .collect();
        let outputs = json["outputs"]
            .as_array()
            .expect("fixture `outputs` is an array")
            .iter()
            .map(parse_tensor)
            .collect();
        let metadata = json["metadata"]
            .as_object()
            .expect("fixture `metadata` is an object")
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().expect("metadata value is a string").to_string(),
                )
            })
            .collect();

        ModelSignals {
            source,
            inputs,
            outputs,
            metadata,
        }
    }

    /// Builds a synthetic Ultralytics-style metadata map (Python dict-repr
    /// `names`, `task`, `end2end`).
    fn synthetic_metadata(nc: usize, task: &str, end2end: &str) -> BTreeMap<String, String> {
        let names = (0..nc)
            .map(|i| format!("{i}: 'c{i}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut m = BTreeMap::new();
        m.insert("names".into(), format!("{{{names}}}"));
        m.insert("task".into(), task.into());
        m.insert("end2end".into(), end2end.into());
        m
    }

    #[test]
    fn infer_transposed_det_layout_classified() {
        // Anchors-first layout ([1, A, 4+nc]).
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 8400, 84],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        let o = &r.schema.outputs[0];
        assert_eq!(
            o.dshape,
            vec![
                (DimName::Batch, 1),
                (DimName::NumBoxes, 8400),
                (DimName::NumFeatures, 84),
            ]
        );
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolov8));
    }

    #[test]
    fn infer_yolo26n_onnx_end_to_end() {
        let s = signals_from_fixture("yolo26n");
        let r = infer_ultralytics_schema(&s).unwrap();
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolo26));
        assert_eq!(r.schema.outputs[0].shape[2], 6);
    }

    #[test]
    fn infer_class_count_mismatch_rejected() {
        let mut s = signals_from_fixture("yolov8n");
        s.metadata.insert("names".into(), "{0: 'only-one'}".into()); // nc=1, f=84
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::ClassCountMismatch {
                expected: 1,
                found: 80
            })
        ));
    }

    #[test]
    fn infer_seg_class_count_mismatch_rejected() {
        // The yolov8n-seg det tensor is [1, 116, 8400] (feat = 4+80+32).
        // With `names` cut to 1 class the expected feat is 4+1+32=37, and
        // the mismatch reports found = 116-(37-1) = 80.
        let mut s = signals_from_fixture("yolov8n-seg");
        s.metadata.insert("names".into(), "{0: 'only-one'}".into()); // nc=1
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::ClassCountMismatch {
                expected: 1,
                found: 80
            })
        ));
    }

    #[test]
    fn infer_end2end_flag_shape_conflict_rejected() {
        // A pre-NMS [1, 84, 8400] output with `end2end` forced to "True".
        let mut s = signals_from_fixture("yolov8n");
        s.metadata.insert("end2end".into(), "True".into());
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }

    #[test]
    fn infer_yolov8n_seg_onnx() {
        let r = infer_ultralytics_schema(&signals_from_fixture("yolov8n-seg")).unwrap();
        assert_eq!(r.schema.outputs.len(), 2);
        let proto = r
            .schema
            .outputs
            .iter()
            .find(|o| o.type_ == Some(LogicalType::Protos))
            .unwrap();
        assert_eq!(proto.dshape[1].0, DimName::NumProtos); // NCHW
        let det = r
            .schema
            .outputs
            .iter()
            .find(|o| o.type_ == Some(LogicalType::Detection))
            .unwrap();
        assert!(det.shape.contains(&116)); // 4+80+32
    }

    #[test]
    fn infer_yolov8n_seg_tflite_proto_nchw() {
        // The yolov8n-seg_float32 TFLite export's proto is
        // [1, 32, 160, 160], NCHW-ordered.
        let r = infer_ultralytics_schema(&signals_from_fixture("yolov8n-seg_float32")).unwrap();
        let proto = r
            .schema
            .outputs
            .iter()
            .find(|o| o.type_ == Some(LogicalType::Protos))
            .unwrap();
        assert_eq!(proto.dshape[1].0, DimName::NumProtos); // NCHW
        assert_eq!(proto.dshape[1].1, 32);
    }

    #[test]
    fn infer_yolo26n_seg_end_to_end() {
        let r = infer_ultralytics_schema(&signals_from_fixture("yolo26n-seg")).unwrap();
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolo26));
        assert_eq!(r.schema.outputs.len(), 2);
    }

    #[test]
    fn infer_nhwc_proto_layout_classified() {
        // TFLite source, NHWC input, features-first det tensor
        // (4+80+32=116), NHWC proto ([1, 160, 160, 32]).
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "output0".into(),
                    shape: vec![1, 116, 8400],
                    dtype: DType::Float32,
                    quantization: None,
                },
                TensorInfo {
                    name: "output1".into(),
                    shape: vec![1, 160, 160, 32],
                    dtype: DType::Float32,
                    quantization: None,
                },
            ],
            metadata: synthetic_metadata(80, "segment", "False"),
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        let proto = r
            .schema
            .outputs
            .iter()
            .find(|o| o.type_ == Some(LogicalType::Protos))
            .unwrap();
        assert_eq!(proto.dshape[3].0, DimName::NumProtos); // NHWC
        assert_eq!(proto.dshape[3].1, 32);
    }

    /// The proto output carries its name, decoder, dtype and quantization.
    #[test]
    fn infer_proto_output_carries_its_name_decoder_dtype_and_quantization() {
        let quant = Quantization {
            scale: vec![0.0125],
            zero_point: Some(vec![-7]),
            axis: None,
            dtype: Some(DType::Int8),
        };
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Int8,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "output0".into(),
                    shape: vec![1, 116, 8400],
                    dtype: DType::Int8,
                    quantization: Some(Quantization {
                        scale: vec![0.003_921_6],
                        zero_point: Some(vec![0]),
                        axis: None,
                        dtype: Some(DType::Int8),
                    }),
                },
                TensorInfo {
                    name: "protos".into(),
                    shape: vec![1, 160, 160, 32],
                    dtype: DType::Int8,
                    quantization: Some(quant.clone()),
                },
            ],
            metadata: synthetic_metadata(80, "segment", "False"),
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        let proto = r
            .schema
            .outputs
            .iter()
            .find(|o| o.type_ == Some(LogicalType::Protos))
            .unwrap();
        assert_eq!(proto.name.as_deref(), Some("protos"));
        assert_eq!(proto.decoder, Some(DecoderKind::Ultralytics));
        assert_eq!(proto.dtype, Some(DType::Int8));
        assert_eq!(proto.quantization.as_ref(), Some(&quant));
    }

    /// A proto tensor makes the model a segmentation model whether or not the
    /// metadata says `task: segment`.
    #[test]
    fn infer_treats_a_proto_tensor_without_a_task_as_segmentation() {
        let mut metadata = synthetic_metadata(80, "segment", "False");
        metadata.remove("task");
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "output0".into(),
                    shape: vec![1, 116, 8400],
                    dtype: DType::Float32,
                    quantization: None,
                },
                TensorInfo {
                    name: "output1".into(),
                    shape: vec![1, 160, 160, 32],
                    dtype: DType::Float32,
                    quantization: None,
                },
            ],
            metadata,
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        assert!(
            r.schema
                .outputs
                .iter()
                .any(|o| o.type_ == Some(LogicalType::Protos)),
            "proto tensor present but no protos output was emitted"
        );
        assert!(r.description.contains("segment"), "{}", r.description);
    }

    fn proto_tensor(shape: Vec<usize>) -> TensorInfo {
        TensorInfo {
            name: "protos".into(),
            shape,
            dtype: DType::Float32,
            quantization: None,
        }
    }

    /// A wrong rank and a wrong batch are each rejected.
    #[test]
    fn classify_proto_rejects_wrong_rank_or_batch() {
        for shape in [vec![1, 32, 160], vec![2, 32, 160, 160], vec![1, 160, 160]] {
            let err = classify_proto(&proto_tensor(shape.clone()), 640, 640).unwrap_err();
            assert!(
                matches!(err, InferError::UnsupportedLayout(_)),
                "shape {shape:?} gave {err:?}"
            );
        }
    }

    /// The NHWC branch requires both spatial dims to match.
    #[test]
    fn classify_proto_rejects_a_shape_matching_only_one_spatial_dim() {
        // h/4 = w/4 = 160. s[1] matches, s[2] does not.
        let err = classify_proto(&proto_tensor(vec![1, 160, 7, 9]), 640, 640).unwrap_err();
        assert!(matches!(err, InferError::UnsupportedLayout(_)), "{err:?}");

        // The NHWC shape classifies, with the proto count last.
        let (k, dshape) = classify_proto(&proto_tensor(vec![1, 160, 160, 32]), 640, 640).unwrap();
        assert_eq!(k, 32);
        assert_eq!(dshape[3], (DimName::NumProtos, 32));
    }

    /// `unquote` strips a matching pair of quotes only.
    #[test]
    fn unquote_requires_a_matching_pair_of_quotes() {
        assert_eq!(unquote("'abc'").as_deref(), Some("abc"));
        assert_eq!(unquote("\"abc\"").as_deref(), Some("abc"));
        assert_eq!(unquote("'abc"), None);
        assert_eq!(unquote("abc'"), None);
        assert_eq!(unquote("\"abc"), None);
        assert_eq!(unquote("abc\""), None);
        assert_eq!(unquote("abc"), None);
    }

    /// A comma inside a quoted value does not split.
    #[test]
    fn split_top_level_commas_keeps_quoted_commas_together() {
        let parts = split_top_level_commas("a='x,y', b=2");
        assert_eq!(parts, vec!["a='x,y'".to_string(), " b=2".to_string()]);

        let parts = split_top_level_commas("a=\"x,y\",b=\"p,q\"");
        assert_eq!(
            parts,
            vec!["a=\"x,y\"".to_string(), "b=\"p,q\"".to_string()]
        );
    }

    /// `InferError`'s `Display` states the reason.
    #[test]
    fn infer_error_display_states_the_reason() {
        let err = InferError::NotUltralytics("no names key".into());
        let shown = err.to_string();
        assert!(shown.contains("no names key"), "{shown}");
        assert!(shown.contains("Ultralytics"), "{shown}");

        let err = InferError::UnsupportedLayout("rank-3 proto".into());
        assert!(err.to_string().contains("rank-3 proto"));
    }

    #[test]
    fn infer_segment_task_without_proto_rejected() {
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "segment", "False"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }

    #[test]
    fn infer_detect_task_with_proto_rejected() {
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "output0".into(),
                    shape: vec![1, 84, 8400],
                    dtype: DType::Float32,
                    quantization: None,
                },
                TensorInfo {
                    name: "output1".into(),
                    shape: vec![1, 32, 160, 160],
                    dtype: DType::Float32,
                    quantization: None,
                },
            ],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }

    #[test]
    fn infer_int8_tflite_classifies_like_float32() {
        // Ultralytics int8 TFLite exports expose float32 I/O with no
        // boundary quantization; this fixture classifies like its float32
        // counterpart.
        let r = infer_ultralytics_schema(&signals_from_fixture("yolov8n_int8")).unwrap();
        let det = &r.schema.outputs[0];
        assert_eq!(det.type_, Some(LogicalType::Detection));
        assert_eq!(det.dtype, Some(DType::Float32));
        assert_eq!(det.quantization, None);
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolov8));
        assert_eq!(det.normalized, Some(true));
    }

    #[test]
    fn infer_quantized_boundary_carries_quantization() {
        // A quantized boundary output: `TensorInfo.quantization` passes
        // through onto the schema's detection output.
        let quant = Quantization {
            scale: vec![0.02],
            zero_point: Some(vec![-5]),
            axis: None,
            dtype: Some(DType::Int8),
        };
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Int8,
                quantization: Some(quant.clone()),
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        let det = &r.schema.outputs[0];
        assert_eq!(det.dtype, Some(DType::Int8));
        assert_eq!(det.quantization, Some(quant));
    }

    #[test]
    fn infer_rejects_batched_input() {
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![4, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(msg.contains("batch 4"), "unexpected message: {msg}");
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    #[test]
    fn infer_rejects_input_without_a_channel_dim() {
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 4, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(msg.contains("channel dimension"), "unexpected: {msg}");
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    #[test]
    fn infer_rejects_model_with_no_rank4_input() {
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(msg.contains("no rank-4 model input"), "unexpected: {msg}");
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    /// Builds detection signals for a 640x640 model with one rank-3 output.
    fn det_signals(nc: usize, out_shape: Vec<usize>, end2end: &str) -> ModelSignals {
        ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: out_shape,
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(nc, "detect", end2end),
        }
    }

    #[test]
    fn infer_end_to_end_accepts_any_max_det_up_to_the_anchor_count() {
        // N is the export's `max_det`, at most the anchor count: 300 is the
        // Ultralytics default and 8400 equals `expected_anchors`.
        for n in [1, 300, 2000, 8400] {
            let r = infer_ultralytics_schema(&det_signals(80, vec![1, n, 6], "True"))
                .unwrap_or_else(|e| panic!("max_det={n} must infer: {e}"));
            assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolo26));
            assert_eq!(r.schema.outputs[0].type_, Some(LogicalType::Detections));
            assert_eq!(
                r.schema.outputs[0].dshape,
                vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, n),
                    (DimName::NumFeatures, 6),
                ],
                "max_det={n}"
            );
            assert_eq!(r.schema.nms, None, "max_det={n}");
        }
    }

    #[test]
    fn infer_end_to_end_rejects_more_rows_than_anchors() {
        // N above the anchor count is not an end-to-end output.
        match infer_ultralytics_schema(&det_signals(80, vec![1, 8401, 6], "True")) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(msg.contains("N<=8400"), "unexpected message: {msg}");
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    #[test]
    fn infer_legacy_nms_model_is_unaffected_by_max_det() {
        // The pre-NMS head emits every anchor and classifies by the anchor
        // count.
        let r = infer_ultralytics_schema(&det_signals(80, vec![1, 84, 8400], "False")).unwrap();
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolov8));
        assert_eq!(r.schema.outputs[0].type_, Some(LogicalType::Detection));
        assert_eq!(
            r.schema.outputs[0].dshape,
            vec![
                (DimName::Batch, 1),
                (DimName::NumFeatures, 84),
                (DimName::NumBoxes, 8400),
            ]
        );
        assert_eq!(r.schema.nms, Some(NmsMode::ClassAware));

        // Anchors-first classifies identically.
        let t = infer_ultralytics_schema(&det_signals(80, vec![1, 8400, 84], "False")).unwrap();
        assert_eq!(t.schema.decoder_version, Some(DecoderVersion::Yolov8));
        assert_eq!(
            t.schema.outputs[0].dshape,
            vec![
                (DimName::Batch, 1),
                (DimName::NumBoxes, 8400),
                (DimName::NumFeatures, 84),
            ]
        );
    }

    #[test]
    fn infer_pre_nms_wins_when_a_shape_reads_as_both() {
        // With metadata stripped and nc == 2, feat == 4 + nc == 6 equals the
        // end-to-end feature width, so [1, A, 6] fits both layouts; pre-NMS
        // is tried first.
        let mut m = synthetic_metadata(2, "detect", "False");
        m.remove("end2end");
        let s = ModelSignals {
            metadata: m,
            ..det_signals(2, vec![1, 8400, 6], "False")
        };
        let r = infer_ultralytics_schema(&s).unwrap();
        assert_eq!(r.schema.decoder_version, Some(DecoderVersion::Yolov8));
    }

    #[test]
    fn meta_empty_names_rejected() {
        let mut m = BTreeMap::new();
        m.insert("names".into(), "{}".into());
        match UltralyticsMeta::from_metadata(&m) {
            Err(InferError::BadNames(msg)) => assert!(msg.contains("empty"), "{msg}"),
            other => panic!("expected BadNames, got {other:?}"),
        }
    }

    #[test]
    fn meta_malformed_end2end_is_an_error_not_an_absent_key() {
        let mut m = synthetic_metadata(80, "detect", "False");
        m.insert("end2end".into(), "yes".into());
        match UltralyticsMeta::from_metadata(&m) {
            Err(InferError::BadMetadata(msg)) => assert!(msg.contains("end2end"), "{msg}"),
            other => panic!("expected BadMetadata, got {other:?}"),
        }

        m.remove("end2end");
        assert_eq!(UltralyticsMeta::from_metadata(&m).unwrap().end2end, None);
    }

    #[test]
    fn meta_foreign_vendor_rejected_but_missing_provenance_allowed() {
        let mut m = synthetic_metadata(80, "detect", "False");
        m.insert("author".into(), "SomeOtherVendor".into());
        assert!(matches!(
            UltralyticsMeta::from_metadata(&m),
            Err(InferError::NotUltralytics(_))
        ));

        // Provenance that names Ultralytics passes, in either field.
        m.insert("author".into(), "Ultralytics".into());
        m.insert("docs".into(), "https://docs.ultralytics.com".into());
        assert!(UltralyticsMeta::from_metadata(&m).is_ok());

        // Absent provenance passes.
        m.remove("author");
        m.remove("docs");
        assert!(UltralyticsMeta::from_metadata(&m).is_ok());
    }

    #[test]
    fn infer_zero_row_end_to_end_rejected() {
        // A zero-row [1, 0, 6] tensor is refused.
        assert!(matches!(
            infer_ultralytics_schema(&det_signals(80, vec![1, 0, 6], "True")),
            Err(InferError::UnsupportedLayout(_))
        ));
    }

    #[test]
    fn infer_duplicate_matching_outputs_are_ambiguous() {
        let input = TensorInfo {
            name: "images".into(),
            shape: vec![1, 3, 640, 640],
            dtype: DType::Float32,
            quantization: None,
        };
        let t = |name: &str, shape: Vec<usize>| TensorInfo {
            name: name.into(),
            shape,
            dtype: DType::Float32,
            quantization: None,
        };

        // Two identical pre-NMS detection tensors.
        let pre = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![input.clone()],
            outputs: vec![t("a", vec![1, 84, 8400]), t("b", vec![1, 84, 8400])],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&pre) {
            Err(InferError::AmbiguousLayout(msg)) => {
                assert!(msg.contains("`a`") && msg.contains("`b`"), "{msg}");
            }
            other => panic!("expected AmbiguousLayout, got {other:?}"),
        }

        // Two end-to-end candidates.
        let e2e = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![input],
            outputs: vec![t("a", vec![1, 300, 6]), t("b", vec![1, 200, 6])],
            metadata: synthetic_metadata(80, "detect", "True"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&e2e),
            Err(InferError::AmbiguousLayout(_))
        ));
    }

    #[test]
    fn infer_malformed_per_tensor_quantization_rejected() {
        let build = |q: Quantization| ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Int8,
                quantization: Some(q),
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };

        // A quantization with no scale is refused.
        match infer_ultralytics_schema(&build(Quantization {
            scale: vec![],
            zero_point: Some(vec![]),
            axis: None,
            dtype: Some(DType::Int8),
        })) {
            Err(InferError::UnsupportedQuantization(msg)) => {
                assert!(msg.contains("no scale"), "{msg}");
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }

        // Zero points that do not pair with the scale are refused.
        match infer_ultralytics_schema(&build(Quantization {
            scale: vec![0.02],
            zero_point: Some(vec![-5, -7]),
            axis: None,
            dtype: Some(DType::Int8),
        })) {
            Err(InferError::UnsupportedQuantization(msg)) => {
                assert!(msg.contains("zero point"), "{msg}");
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }

        // Symmetric quantization (no zero points) is accepted.
        assert!(infer_ultralytics_schema(&build(Quantization {
            scale: vec![0.02],
            zero_point: None,
            axis: None,
            dtype: Some(DType::Int8),
        }))
        .is_ok());
    }

    #[test]
    fn infer_pre_nms_is_independent_of_output_order() {
        // Several anchor-shaped rank-3 outputs: a `[1, 4, A]` box tensor
        // beside the `[1, 4+nc, A]` detection tensor, or a `[1, k, A]`
        // mask-coefficient tensor.
        let det = TensorInfo {
            name: "output0".into(),
            shape: vec![1, 84, 8400],
            dtype: DType::Float32,
            quantization: None,
        };
        let aux = TensorInfo {
            name: "boxes".into(),
            shape: vec![1, 4, 8400],
            dtype: DType::Float32,
            quantization: None,
        };
        let build = |outputs: Vec<TensorInfo>| ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs,
            metadata: synthetic_metadata(80, "detect", "False"),
        };

        let det_first = infer_ultralytics_schema(&build(vec![det.clone(), aux.clone()]))
            .expect("detection tensor first must classify");
        let aux_first = infer_ultralytics_schema(&build(vec![aux, det]))
            .expect("an auxiliary anchor-shaped tensor first must not shadow the detection tensor");
        assert_eq!(det_first.schema.outputs[0].name, Some("output0".into()));
        assert_eq!(aux_first.schema.outputs[0].name, Some("output0".into()));
        assert_eq!(det_first.schema, aux_first.schema);
    }

    #[test]
    fn infer_class_count_mismatch_survives_a_full_scan() {
        // When nothing matches, the anchor-shaped near-miss reports a
        // class-count mismatch.
        let s = ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            // 10 classes would need feat == 14, not the 84 present.
            metadata: synthetic_metadata(10, "detect", "False"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::ClassCountMismatch {
                expected: 10,
                found: 80
            })
        ));
    }

    /// What a captured fixture infers to.
    struct FixtureExpectation {
        name: &'static str,
        version: DecoderVersion,
        det_type: LogicalType,
        /// Detection tensor `dshape`, in physical order.
        det_dshape: &'static [(DimName, usize)],
        normalized: bool,
        nms: Option<NmsMode>,
        /// Proto channel count and spatial dims, for segmentation fixtures.
        proto: Option<(usize, usize, usize)>,
    }

    const A: usize = 8400; // 640x640 anchors: 80² + 40² + 20²
    const NC: usize = 80;

    fn fixture_expectations() -> Vec<FixtureExpectation> {
        use DimName::{Batch, NumBoxes, NumFeatures};
        // Every captured export is features-first ([1, 4+nc, A]).
        const DET: &[(DimName, usize)] = &[(Batch, 1), (NumFeatures, 4 + NC), (NumBoxes, A)];
        const SEG: &[(DimName, usize)] = &[(Batch, 1), (NumFeatures, 4 + NC + 32), (NumBoxes, A)];
        // End-to-end heads emit [1, max_det, 6] (+32 mask coefficients).
        const E2E: &[(DimName, usize)] = &[(Batch, 1), (NumBoxes, 300), (NumFeatures, 6)];
        const E2E_SEG: &[(DimName, usize)] = &[(Batch, 1), (NumBoxes, 300), (NumFeatures, 6 + 32)];

        let pre_nms = |name, det_dshape, normalized, proto| FixtureExpectation {
            name,
            version: DecoderVersion::Yolov8,
            det_type: LogicalType::Detection,
            det_dshape,
            normalized,
            nms: Some(NmsMode::ClassAware),
            proto,
        };
        let e2e = |name, det_dshape, normalized, proto| FixtureExpectation {
            name,
            version: DecoderVersion::Yolo26,
            det_type: LogicalType::Detections,
            det_dshape,
            normalized,
            nms: None,
            proto,
        };
        // Ultralytics ONNX exports are pixel-space, TFLite exports [0,1];
        // proto is (k, H/4, W/4).
        const P: Option<(usize, usize, usize)> = Some((32, 160, 160));
        vec![
            pre_nms("yolov8n", DET, false, None),
            pre_nms("yolo11n", DET, false, None),
            pre_nms("yolov8n-seg", SEG, false, P),
            pre_nms("yolo11n-seg", SEG, false, P),
            pre_nms("yolov8n_float32", DET, true, None),
            pre_nms("yolov8n_int8", DET, true, None),
            pre_nms("yolov8n-seg_float32", SEG, true, P),
            pre_nms("yolov8n-seg_int8", SEG, true, P),
            e2e("yolo26n", E2E, false, None),
            e2e("yolo26n-seg", E2E_SEG, false, P),
            e2e("yolo26n_float32", E2E, true, None),
            // A YOLO26 checkpoint exported with the one2many anchor-grid
            // head: a pre-NMS row.
            pre_nms("yolo26n_coreml", DET, false, None),
        ]
    }

    #[test]
    fn every_captured_fixture_infers_the_expected_schema() {
        for want in fixture_expectations() {
            let name = want.name;
            let r = infer_ultralytics_schema(&signals_from_fixture(name))
                .unwrap_or_else(|e| panic!("{name}: inference failed: {e}"));

            assert_eq!(r.labels.len(), NC, "{name}: class count");
            assert_eq!(
                r.labels[0], "person",
                "{name}: labels must be index-ordered"
            );
            assert_eq!(r.labels[NC - 1], "toothbrush", "{name}: labels index order");
            assert_eq!(
                r.schema.decoder_version,
                Some(want.version),
                "{name}: version"
            );
            assert_eq!(r.schema.nms, want.nms, "{name}: nms mode");

            let det = &r.schema.outputs[0];
            assert_eq!(det.type_, Some(want.det_type), "{name}: detection type");
            assert_eq!(det.dshape, want.det_dshape, "{name}: detection dshape");
            assert_eq!(det.normalized, Some(want.normalized), "{name}: normalized");
            assert_eq!(
                det.decoder,
                Some(DecoderKind::Ultralytics),
                "{name}: decoder"
            );
            assert_eq!(det.encoding, Some(BoxEncoding::Direct), "{name}: encoding");
            assert_eq!(
                det.score_format,
                Some(ScoreFormat::PerClass),
                "{name}: scores"
            );

            match want.proto {
                None => {
                    assert_eq!(r.schema.outputs.len(), 1, "{name}: detection-only");
                }
                Some((k, ph, pw)) => {
                    assert_eq!(r.schema.outputs.len(), 2, "{name}: detection + proto");
                    let proto = &r.schema.outputs[1];
                    assert_eq!(proto.type_, Some(LogicalType::Protos), "{name}: proto type");
                    assert_eq!(
                        proto.dshape,
                        vec![
                            (DimName::Batch, 1),
                            (DimName::NumProtos, k),
                            (DimName::Height, ph),
                            (DimName::Width, pw),
                        ],
                        "{name}: proto dshape"
                    );
                }
            }

            let task = if want.proto.is_some() {
                "segment"
            } else {
                "detect"
            };
            assert!(
                r.description.contains(task) && r.description.contains("Ultralytics"),
                "{name}: description was `{}`",
                r.description
            );
        }
    }

    #[test]
    fn every_fixture_is_asserted() {
        // Every `*.signals.json` on disk has a row in `fixture_expectations`.
        let dir = infer_fixture_dir();
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{} exists: {e}", dir.display()))
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                name.strip_suffix(".signals.json").map(str::to_string)
            })
            .collect();
        on_disk.sort();

        let mut asserted: Vec<String> = fixture_expectations()
            .iter()
            .map(|f| f.name.to_string())
            .collect();
        asserted.sort();

        assert_eq!(
            on_disk, asserted,
            "every *.signals.json fixture must appear in fixture_expectations()"
        );
    }

    #[test]
    fn infer_other_source_refuses_rather_than_guessing_normalization() {
        // An uncharacterized container is refused with a typed error.
        let s = ModelSignals {
            source: ModelSource::Other,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float32,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnknownBoxConvention)
        ));

        // The same signals under a known source succeed.
        let ok = ModelSignals {
            source: ModelSource::Onnx,
            ..s
        };
        assert_eq!(
            infer_ultralytics_schema(&ok).unwrap().schema.outputs[0].normalized,
            Some(false)
        );
    }

    #[test]
    fn infer_per_channel_quantization_rejected() {
        // Per-channel quantization on a boundary output is refused.
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Int8,
                quantization: Some(Quantization {
                    scale: vec![0.02, 0.03],
                    zero_point: Some(vec![-5, -7]),
                    axis: Some(1),
                    dtype: Some(DType::Int8),
                }),
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedQuantization(msg)) => {
                assert!(msg.contains("per-channel"), "unexpected message: {msg}");
                assert!(
                    msg.contains("output0"),
                    "message must name the tensor: {msg}"
                );
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }
    }

    #[test]
    fn infer_per_channel_proto_quantization_rejected() {
        // The proto tensor is checked as well as the detection output.
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "output0".into(),
                    shape: vec![1, 116, 8400],
                    dtype: DType::Float32,
                    quantization: None,
                },
                TensorInfo {
                    name: "output1".into(),
                    shape: vec![1, 32, 160, 160],
                    dtype: DType::Int8,
                    quantization: Some(Quantization {
                        scale: vec![0.02, 0.03],
                        zero_point: Some(vec![-5, -7]),
                        axis: Some(1),
                        dtype: Some(DType::Int8),
                    }),
                },
            ],
            metadata: synthetic_metadata(80, "segment", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedQuantization(msg)) => {
                assert!(
                    msg.contains("output1"),
                    "message must name the proto: {msg}"
                );
            }
            other => panic!("expected UnsupportedQuantization, got {other:?}"),
        }
    }

    #[test]
    fn infer_integer_output_without_quant_rejected() {
        // An integer-dtype boundary output with `quantization: None` is
        // refused.
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 640, 640, 3],
                dtype: DType::Float32,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Int8,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        match infer_ultralytics_schema(&s) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(
                    msg.contains("quantization"),
                    "unexpected UnsupportedLayout message: {msg}"
                );
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    #[test]
    fn infer_pre_nms_rejects_non_unit_batch() {
        // A [2, 84, 8400] output has a matching anchors/feature pair but a
        // non-unit leading (batch) dim.
        let mut s = signals_from_fixture("yolov8n");
        s.outputs[0].shape = vec![2, 84, 8400];
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }

    #[test]
    fn coreml_source_resolves_a_box_convention() {
        // A CoreML source is not refused as `Other`.
        let s = ModelSignals {
            source: ModelSource::CoreMl,
            inputs: vec![TensorInfo {
                name: "images".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float16,
                quantization: None,
            }],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 84, 8400],
                dtype: DType::Float16,
                quantization: None,
            }],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        let r = infer_ultralytics_schema(&s)
            .expect("CoreML signals must infer, not raise UnknownBoxConvention");
        let o = &r.schema.outputs[0];
        assert_eq!(
            o.normalized,
            Some(false),
            "Ultralytics CoreML exports trace the raw model, so boxes are \
             pixel-space exactly as in the ONNX export"
        );
    }

    #[test]
    fn coreml_nms_pipeline_export_is_refused_not_classified() {
        // A `nms=True` CoreML export: `confidence` and `coordinates` are
        // float32 outputs with an empty shape, neither of them the
        // `[1, 4+nc, A]` layout.
        let s = ModelSignals {
            source: ModelSource::CoreMl,
            inputs: vec![TensorInfo {
                name: "image".into(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::Float16,
                quantization: None,
            }],
            outputs: vec![
                TensorInfo {
                    name: "confidence".into(),
                    shape: vec![],
                    dtype: DType::Float32,
                    quantization: None,
                },
                TensorInfo {
                    name: "coordinates".into(),
                    shape: vec![],
                    dtype: DType::Float32,
                    quantization: None,
                },
            ],
            metadata: synthetic_metadata(80, "detect", "False"),
        };
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }
}
