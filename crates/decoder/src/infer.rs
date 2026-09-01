// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Ultralytics YOLO schema inference from raw model I/O signals.
//!
//! Model export pipelines (ONNX, TFLite) carry Ultralytics-authored
//! metadata (class names, task, input size) alongside the tensor shapes
//! and dtypes the runtime reports. This module turns that raw signal into
//! a [`crate::schema::SchemaV2`] the decoder can act on, without requiring
//! a hand-written `edgefirst.json` for every Ultralytics export.

use std::collections::BTreeMap;
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

/// Container format of the model the signals were read from. Drives
/// Ultralytics per-format conventions (box normalization, tensor order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// An ONNX export. Ultralytics emits pixel-space box coordinates here,
    /// so the inferred schema sets `normalized: false`.
    Onnx,
    /// A TFLite/LiteRT export. Ultralytics emits boxes normalized to
    /// `[0, 1]` here, so the inferred schema sets `normalized: true`.
    TfLite,
    /// Any other container. Inference **refuses** this with
    /// [`InferError::UnknownBoxConvention`] rather than assuming a box
    /// convention: whether coordinates are pixel-space or `[0, 1]` follows
    /// the exporter, is not derivable from tensor shapes, and getting it
    /// wrong scales every box by the input size. The variant exists so a
    /// signal collector can report honestly what it read; add a measured
    /// container here rather than routing it through `Other`.
    Other,
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
    /// metadata entries). Values are passed verbatim; parsing happens here.
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
    /// interpreted. Distinct from an absent field, which is allowed to
    /// select a fallback.
    BadMetadata(String),
    /// The Ultralytics `task` metadata field names an unsupported task.
    UnsupportedTask(String),
    /// The number of classes in `names` does not match what the output
    /// tensor shapes imply.
    ClassCountMismatch { expected: usize, found: usize },
    /// The output tensor layout does not match any supported Ultralytics
    /// convention.
    UnsupportedLayout(String),
    /// More than one output matches the same layout, so which tensor
    /// carries the detections depends on the order the runtime reported
    /// them. Reported rather than resolved by position.
    AmbiguousLayout(String),
    /// A boundary tensor carries quantization the decoder cannot consume.
    /// Currently this means per-channel quantization: the decoder supports
    /// per-tensor only, so such a schema would build a decoder that fails.
    UnsupportedQuantization(String),
    /// The signals came from [`ModelSource::Other`], for which the box
    /// coordinate convention is unmeasured. Only the ONNX (pixel-space)
    /// and TFLite (`[0, 1]`) conventions are characterized, and picking
    /// the wrong one silently scales every box by the input size.
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
                 onnx (pixel-space) and tflite ([0,1]) are characterized"
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

/// Refuses metadata that positively identifies a *different* vendor.
///
/// A `names` map alone is not an Ultralytics signature — any exporter can
/// write class names — and this function then applies Ultralytics box,
/// score and normalization conventions to whatever it was given. Real
/// exports carry `author: "Ultralytics"` and a `docs` URL, so when either
/// is present and names someone else, that is a positive contradiction and
/// an error.
///
/// Absence is not treated as a contradiction: intermediate tools
/// (quantizers, format converters) routinely preserve `names` while
/// dropping provenance fields, and those exports are still the ones this
/// module exists to read. The layout cross-checks that follow remain the
/// substantive guard — a model whose shapes don't fit is rejected whatever
/// its metadata claims.
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

    /// Takes the `names` value directly rather than re-looking it up:
    /// [`Self::from_metadata`] only reaches here after finding the key, so
    /// a second lookup would add an error arm that can never fire.
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
        // The envelope types it properly, so anything non-boolean is
        // corrupt rather than merely absent.
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
/// An unparseable value is an error rather than `None`: `None` means "the
/// key is absent", which selects the shape-only fallback. Letting a
/// corrupt value collapse into that would classify by shape while the
/// model was actively claiming something else.
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
/// by parsing keys as indices, rather than relying on lexicographic string
/// order (which would misorder e.g. "10" before "2").
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
    // Zero classes is not a degenerate-but-valid model: `feat` would be
    // `4 + 0`, and the decoder requires at least one class, so a schema
    // built from it parses and then fails at `DecoderBuilder::build`.
    // Refuse here, where the message can say what is actually missing.
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

/// Splits a dict body on top-level commas, treating `'...'` and `"..."`
/// spans as opaque so commas inside class names never split an entry.
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
/// index 3 (NHWC). Layout is never assumed from [`ModelSource`] — some
/// TFLite exports are still NCHW at the tensor level.
///
/// Batch must be 1. Every output layout this module classifies is
/// batch-1 (`classify_pre_nms`, `find_e2e_candidate` and `classify_proto`
/// all require it), so accepting a batched input would emit a `dshape`
/// whose `Batch` disagrees with the detection tensor it is paired with.
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
/// value and its index (1 or 2) within `shape`. `None` if no dim matches,
/// including when the leading dim isn't 1 — same requirement as
/// [`find_e2e_candidate`], so a batch > 1 tensor is never silently
/// mislabeled batch-1.
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

/// Classifies the combined pre-NMS detection layout (rule 4): a rank-3
/// output whose dims contain both `expected_anchors` and `feat` (`4 + nc`).
///
/// Every rank-3 output is scanned to completion. A tensor that carries the
/// anchor count but the wrong feature width is remembered as a fallback
/// [`InferError::ClassCountMismatch`] rather than returned immediately,
/// because an export can legitimately publish more than one anchor-shaped
/// output — a `[1, 4, 8400]` box tensor beside the `[1, 4+nc, 8400]`
/// detection tensor, or a `[1, k, 8400]` mask-coefficient tensor — and
/// returning on the first near-miss made the result depend on the order the
/// runtime happened to report its outputs in. The mismatch is only
/// surfaced when no output matched, where it is the more useful diagnostic:
/// a dim equal to the anchor count unambiguously signals pre-NMS intent, so
/// a feature-width mismatch there is a class-count problem rather than a
/// layout question.
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
            // Two tensors of the identical detection shape leave nothing to
            // choose between them but the order the runtime listed them in.
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
/// segmentation (`k` being the proto channel count).
///
/// `N` is the export's `max_det`, which Ultralytics bakes into the graph
/// rather than applying afterwards: the head's `postprocess` runs a TopK
/// with `k = min(max_det, anchors)` (`nn/modules/head.py`), so the tensor
/// carries exactly that many rows and no padding. The bound here is
/// upstream's own clamp, `m.max_det = min(args.max_det, available)` where
/// `available` is the anchor count summed over strides
/// (`engine/exporter.py`) — the identical quantity this module computes as
/// `expected_anchors`. So `N == expected_anchors` is a legitimate
/// end-to-end shape (any `max_det >= anchors` produces it, verified by
/// export), not a pre-NMS tensor to be excluded.
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
            // `N > 0`: a zero-row tensor satisfies every other condition but
            // produces a schema the decoder rejects for a zero dimension.
            (s[0] == 1 && s[2] == feat && s[1] > 0 && s[1] <= expected_anchors)
                .then_some((*o, s[1]))
        })
        .collect();
    // Two tensors of the same end-to-end shape leave nothing to choose
    // between them but the order the runtime listed them in. This module
    // reports ambiguity rather than picking.
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

/// Classifies a rank-4 proto tensor (rule 1): the mask-prototype output
/// paired with a segmentation detection tensor. Expected spatial dims are
/// `(H/4, W/4)` (Ultralytics proto stride 4); the remaining non-batch dim
/// is the proto channel count `k`, never hardcoded. Supports both
/// `[1, k, H/4, W/4]` (NCHW) and `[1, H/4, W/4, k]` (NHWC).
///
/// The NCHW check runs first, so a shape where `k` happens to equal one of
/// the spatial dims (e.g. `[1, 160, 160, 160]` on a 640x640 input, where
/// `k == H/4 == W/4`) is shape-undecidable and deliberately resolves NCHW
/// rather than erroring — no real Ultralytics export produces a proto with
/// that many channels, so this tie-break is a pragmatic default, not a
/// verified convention.
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
    }
}

/// Rejects an integer-dtype boundary tensor that carries no quantization
/// parameters — there would be no way to dequantize it, so this is always
/// an error rather than a value silently passed through as raw integers.
///
/// Real Ultralytics int8 TFLite exports never hit this: the int8 graph is
/// wrapped in dequant/quant ops, so the boundary tensors the runtime
/// reports are float32 (see `infer_int8_tflite_classifies_like_float32`).
/// This guard only fires for a genuinely quantized boundary, which no
/// captured fixture exhibits but a future export convention might.
/// Per-channel quantization is refused here for the same reason: the
/// decoder supports per-tensor quantization only, so a per-channel boundary
/// would produce a schema that parses and then fails at
/// `DecoderBuilder::build` with an error naming the decoder rather than the
/// signal that caused it. Refusing at the boundary names the real problem
/// while the caller still has the tensor in hand.
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
        // Exactly one scale, not merely "not per-channel". An empty scale
        // list passed the check above and produced `{"scale": []}`, which
        // the decoder reads as a zero scale rather than rejecting.
        if q.scale.is_empty() {
            return Err(InferError::UnsupportedQuantization(format!(
                "output `{}` is quantized but carries no scale",
                tensor.name
            )));
        }
        // Zero points are optional (symmetric quantization), but when
        // present must pair with the scale; a longer list was silently
        // truncated to its first element.
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
/// a rank-4 proto tensor; the two signals are cross-checked (rule 5) so a
/// `segment` task with no proto, or a `detect` task with a proto present,
/// is always an error rather than a guess.
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
/// error rather than a best-effort guess:
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
    let meta = UltralyticsMeta::from_metadata(&signals.metadata)?;

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

    let expected_anchors = (h / 8) * (w / 8) + (h / 16) * (w / 16) + (h / 32) * (w / 32);
    let nc = meta.names.len();

    // Segmentation detection features carry `k` extra mask coefficients
    // (`4 + nc + k`); the end-to-end feature width grows by the same `k`
    // (`6 + k`). `k` comes from the proto tensor itself (rule 1), never
    // hardcoded, so both widths fall back to the plain detection case when
    // there's no proto.
    // `is_segment` is already true whenever a proto is present (the two
    // are cross-checked above), so the proto's presence alone decides here.
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

    // The `end2end` metadata flag is the primary end-to-end signal (present
    // on every real export). The shape-based e2e fallback applies only
    // when the flag is absent (stripped metadata).
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

    // Box normalization is the one field shape cannot reveal -- it follows
    // the exporter, and only the ONNX and TFLite conventions have been
    // measured (see testdata/infer/NOTES.md answer 5: 637.25 px vs 0.9957 on
    // the same image). Guessing it wrong scales every box by the input size,
    // which is why `Other` is refused rather than defaulted: everywhere else
    // this module errors on ambiguity, and this is the field whose
    // corruption `tests/infer_builder.rs` exists to pin.
    let normalized = match signals.source {
        ModelSource::TfLite => true,
        ModelSource::Onnx => false,
        ModelSource::Other => {
            return Err(InferError::UnknownBoxConvention);
        }
    };
    let det = LogicalOutput {
        name: Some(out.name.clone()),
        // `Detections` (plural) is the schema vocabulary's fully-decoded
        // post-NMS type; `Detection` is the anchor-grid output that still
        // needs decoding. Both collapse to the same legacy path today
        // (schema.rs `logical_to_legacy`), but this JSON now leaves the
        // process through the C and Python bindings, so it has to name the
        // tensor it actually describes.
        type_: Some(match version {
            DecoderVersion::Yolo26 => LogicalType::Detections,
            _ => LogicalType::Detection,
        }),
        shape: out.shape.clone(),
        dshape,
        decoder: Some(DecoderKind::Ultralytics),
        encoding: Some(BoxEncoding::Direct), // graph already decoded DFL
        score_format: Some(ScoreFormat::PerClass),
        normalized: Some(normalized),
        anchors: None,
        stride: None,
        dtype: Some(out.dtype),
        quantization: out.quantization.clone(),
        outputs: Vec::new(),
        activation_applied: None,
        activation_required: None,
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
        }),
        outputs,
        // Ultralytics runs NMS with `agnostic=False`, i.e. class-aware, so
        // that is what a schema describing an Ultralytics model must say.
        // Leaving it unset is not neutral: the builder's `Nms::Auto` default
        // resolves an unset config to ClassAgnostic (builder.rs `resolve_auto`),
        // which suppresses a box against an overlapping box of a *different*
        // class and silently loses recall against the model's own reference
        // output. An explicit `.with_nms(..)` on the builder still overrides.
        //
        // YOLO26 end-to-end heads do their NMS in-graph, so they carry none:
        // this field describes what the *decoder* must add.
        nms: match version {
            DecoderVersion::Yolo26 => None,
            _ => Some(NmsMode::ClassAware),
        },
        decoder_version: Some(version),
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

    /// Loads a captured Task-0 fixture's `metadata` map, exactly as
    /// `ModelSignals::metadata` would carry it: raw string key/values,
    /// regardless of whether the source was ONNX (flat props) or TFLite
    /// (single `metadata.json` envelope entry).
    fn load_fixture_metadata(name: &str) -> BTreeMap<String, String> {
        let path = format!(
            "{}/testdata/infer/{name}.signals.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
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

    /// Loads a captured Task-0 fixture into `ModelSignals`, exactly as the
    /// inference runtime would report it: tensor names/shapes/dtypes plus
    /// the raw metadata map.
    fn signals_from_fixture(name: &str) -> ModelSignals {
        let path = format!(
            "{}/testdata/infer/{name}.signals.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
        let json: serde_json::Value =
            serde_json::from_str(&content).expect("fixture is valid JSON");

        let source = match json["source"]
            .as_str()
            .expect("fixture `source` is a string")
        {
            "onnx" => ModelSource::Onnx,
            "tflite" => ModelSource::TfLite,
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
                // Every captured fixture reports `"quantization": null`
                // (see testdata/infer/NOTES.md); real quantized signals
                // are out of scope for this task.
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
    /// `names`, `task`, `end2end`) for hand-built layout tests that don't
    /// need a full captured fixture.
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
        // Anchors-first layout ([1, A, 4+nc]), as produced by older
        // Ultralytics TFLite exporters. Rule 4 must still classify this
        // correctly from tensor shape alone, and the builder's dshape-order
        // e2e-inference pitfall must be guarded by an explicit
        // decoder_version.
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
        // Covers the generalized ClassCountMismatch math for segmentation,
        // where the non-class base width is `4 + k` (proto channels), not
        // the plain-detection `4`. Real yolov8n-seg det tensor is
        // [1, 116, 8400] (features-first, feat = 4+80+32); truncating
        // `names` to 1 class makes the expected feat 4+1+32=37, so the
        // real 116-feature tensor mismatches with found = 116-(37-1) = 80
        // — the true class count the features imply, unaffected by the
        // metadata truncation.
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
        // The end2end flag is the primary signal: a real yolov8n ONNX
        // export with its pre-NMS [1, 84, 8400] output, but with `end2end`
        // forced to "True", must never be guessed into an end-to-end
        // classification — flag/shape disagreement is always an error.
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
        // The real yolov8n-seg_float32 TFLite export's proto is
        // [1, 32, 160, 160], NCHW-ordered — same as its ONNX counterpart.
        // An earlier draft of this test assumed TFLite seg exports carry a
        // NHWC-transposed proto; no captured fixture confirms that (see
        // infer_nhwc_proto_layout_classified for the synthetic NHWC coverage
        // the shape-driven classifier still supports).
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
        // Hand-built signals: TFLite source, NHWC input, features-first det
        // tensor (4+80+32=116), NHWC proto ([1, 160, 160, 32]) — the other
        // valid proto channel ordering rule 1 must also classify correctly.
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
        // Real Ultralytics int8 TFLite exports expose float32 I/O with NO
        // boundary quantization — the int8 graph is wrapped in
        // dequant/quant ops (Task 0 finding). The classifier must treat
        // this fixture exactly like its float32 counterpart, not attempt
        // to recover quantization that isn't there.
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
        // Hand-built signals for a model that DOES expose a quantized
        // boundary output (unlike any captured fixture) — the classifier
        // must pass `TensorInfo.quantization` through onto the schema's
        // detection output untouched.
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
        // Every output layout here is batch-1, so a batched input would
        // emit an input dshape whose Batch disagrees with the detection
        // tensor beside it. Previously this succeeded and emitted
        // `(Batch, 4)`.
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
        // Ultralytics bakes max_det into the graph: the head's postprocess
        // is a TopK with k = min(max_det, anchors), so N is whatever the
        // export chose. The exporter clamps it to the anchor count
        // (exporter.py: `m.max_det = min(args.max_det, available)`), which
        // is the same quantity as `expected_anchors` here -- verified by
        // exporting yolo26n at max_det = 100/1000/2000/8400/10000, the last
        // two both yielding [1, 8400, 6].
        //
        // 300 is the default; 2000 is what a crowded-scene export uses and
        // was rejected outright before; 8400 == expected_anchors is what any
        // max_det >= anchors produces and is equally legitimate.
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
            // End-to-end ran NMS in-graph, so the schema adds none.
            assert_eq!(r.schema.nms, None, "max_det={n}");
        }
    }

    #[test]
    fn infer_end_to_end_rejects_more_rows_than_anchors() {
        // Upstream cannot emit this: TopK's k is clamped to the anchor
        // count, so a longer tensor is not an Ultralytics end-to-end output
        // and the bound stays a real check rather than a formality.
        match infer_ultralytics_schema(&det_signals(80, vec![1, 8401, 6], "True")) {
            Err(InferError::UnsupportedLayout(msg)) => {
                assert!(msg.contains("N<=8400"), "unexpected message: {msg}");
            }
            other => panic!("expected UnsupportedLayout, got {other:?}"),
        }
    }

    #[test]
    fn infer_legacy_nms_model_is_unaffected_by_max_det() {
        // The pre-NMS (NMS-required) head has no max_det in its graph at
        // all: it emits every anchor, and the detection count is decided
        // later by the decoder's own NMS. So the schema must classify it by
        // the anchor count regardless, and must NOT acquire a NumBoxes
        // dimension that looks like an end-to-end row count.
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
        // Class-aware NMS, matching Ultralytics' own agnostic=False. The
        // schema names the mode only; thresholds and max_det stay the
        // caller's, set on the builder per frame-rate/recall budget.
        assert_eq!(r.schema.nms, Some(NmsMode::ClassAware));

        // Anchors-first is classified identically -- neither orientation
        // may be mistaken for an end-to-end [1, N, feat] tensor.
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
        // With metadata stripped and nc == 2, feat == 4 + nc == 6 == the
        // end-to-end feature width, so [1, A, 6] is genuinely both layouts
        // at once. pre-NMS is tried first and wins; this is the one place a
        // guess remains, so it is pinned rather than left to drift.
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
        // nc == 0 makes feat == 4, so a [1,4,A] output would infer a schema
        // the decoder then refuses ("Yolo num_features 4 must be > 4").
        let mut m = BTreeMap::new();
        m.insert("names".into(), "{}".into());
        match UltralyticsMeta::from_metadata(&m) {
            Err(InferError::BadNames(msg)) => assert!(msg.contains("empty"), "{msg}"),
            other => panic!("expected BadNames, got {other:?}"),
        }
    }

    #[test]
    fn meta_malformed_end2end_is_an_error_not_an_absent_key() {
        // `None` means "key absent", which selects the shape-only fallback.
        // A corrupt value must not collapse into that -- the model is
        // actively claiming something, just not something parseable.
        let mut m = synthetic_metadata(80, "detect", "False");
        m.insert("end2end".into(), "yes".into());
        match UltralyticsMeta::from_metadata(&m) {
            Err(InferError::BadMetadata(msg)) => assert!(msg.contains("end2end"), "{msg}"),
            other => panic!("expected BadMetadata, got {other:?}"),
        }

        // Absent is still fine, and still means "fall back to shapes".
        m.remove("end2end");
        assert_eq!(UltralyticsMeta::from_metadata(&m).unwrap().end2end, None);
    }

    #[test]
    fn meta_foreign_vendor_rejected_but_missing_provenance_allowed() {
        // A `names` map alone is not an Ultralytics signature; another
        // exporter's model would otherwise be given Ultralytics box, score
        // and normalization conventions.
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

        // Absence is not a contradiction: quantizers and format converters
        // routinely keep `names` and drop provenance.
        m.remove("author");
        m.remove("docs");
        assert!(UltralyticsMeta::from_metadata(&m).is_ok());
    }

    #[test]
    fn infer_zero_row_end_to_end_rejected() {
        // [1, 0, 6] satisfies every other end-to-end condition but yields a
        // schema the decoder refuses for a zero dimension.
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

        // Two identical pre-NMS detection tensors: nothing distinguishes
        // them but the order the runtime listed them in.
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

        // Same for two end-to-end candidates.
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

        // No scale at all: previously emitted `{"scale": []}`, which the
        // decoder reads as a zero scale instead of refusing.
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

        // Zero points that do not pair with the scale were silently
        // truncated to the first element.
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

        // Symmetric quantization (no zero points) stays valid.
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
        // An export may publish more than one anchor-shaped rank-3 output:
        // a `[1, 4, A]` box tensor beside the `[1, 4+nc, A]` detection
        // tensor, or a `[1, k, A]` mask-coefficient tensor. Scanning must
        // not stop at the first near-miss, or the result depends on the
        // order the runtime happened to report its outputs in.
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
        // When nothing matches, the anchor-shaped near-miss is still the
        // more useful diagnostic than "no output matches the layout".
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

    /// What every captured fixture must infer to, asserted field by field.
    ///
    /// `yolo11n-seg` and `yolov8n-seg_int8` previously appeared only in
    /// `tests/infer_builder.rs`'s "did the builder accept it" loop, which
    /// passes for any schema that parses -- so the only v11 segmentation
    /// fixture and the only int8 *segmentation* fixture had no field-level
    /// coverage at all. A table keeps every fixture honest at once, and
    /// `every_fixture_is_asserted` below makes an unasserted one impossible
    /// to add.
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
        // Every captured export is features-first ([1, 4+nc, A]); no real
        // exporter transposes to anchors-first, despite an early draft of
        // these tests assuming TFLite did.
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
            // End-to-end heads ran NMS in-graph; the schema adds none.
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
        // Without this, a fixture can be added (or renamed) and silently
        // never asserted on -- which is exactly how `yolo11n-seg` and
        // `yolov8n-seg_int8` ended up with no field-level coverage.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/infer");
        let mut on_disk: Vec<String> = std::fs::read_dir(dir)
            .expect("testdata/infer exists")
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
        // `normalized` follows the exporter and cannot be read off the
        // shapes. ONNX (pixel-space) and TFLite ([0,1]) are the two measured
        // conventions; an uncharacterized container gets a typed refusal,
        // because guessing scales every box by the input size and the
        // resulting schema looks perfectly valid.
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

        // The same signals under a characterized source succeed, so the
        // refusal is about the container and nothing else.
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
        // Per-channel quantization on a boundary output is refused here
        // rather than passed through. The decoder supports per-tensor only
        // (`the v1 decoder only supports per-tensor quantization`), and it
        // refuses per-channel with OR without an `axis` -- so emitting the
        // schema anyway would defer the failure to `DecoderBuilder::build`,
        // where the message names the decoder instead of the tensor that
        // caused it. No captured fixture exhibits this; a future export
        // convention might.
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
        // The same guard runs on the proto tensor, not just the detection
        // output -- `check_quantized_boundary` is called for both.
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
        // Same synthetic signals as above but with `quantization: None` —
        // an integer-dtype boundary output with no quantization params is
        // never silently accepted (there'd be no way to dequantize it).
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
        // non-unit leading (batch) dim. It must never be silently accepted
        // and mislabeled batch-1 (find_e2e_candidate already requires
        // shape[0] == 1; find_anchors_dim must match it).
        let mut s = signals_from_fixture("yolov8n");
        s.outputs[0].shape = vec![2, 84, 8400];
        assert!(matches!(
            infer_ultralytics_schema(&s),
            Err(InferError::UnsupportedLayout(_))
        ));
    }
}
