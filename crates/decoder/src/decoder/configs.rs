// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The vocabulary types a model config is built from.
//!
//! Per-role output descriptors ([`Detection`], [`Boxes`], [`Scores`],
//! [`Classes`], [`Segmentation`], [`Protos`], [`MaskCoefficients`],
//! [`Mask`]), the axis names used to declare physical layout ([`DimName`]),
//! quantization pairs ([`QuantTuple`]), and the enums that steer decoding:
//! [`DecoderType`], [`DecoderVersion`], [`Nms`], and the resolved
//! [`ModelType`].
//!
//! [`Nms`] is worth a note: it has no `None` variant. Bypassing suppression is
//! expressed as `Option<Nms>::None` on the decoder configuration, and
//! [`Nms::Auto`] means "take the mode from the config document", falling back
//! to [`Nms::ClassAware`] when the document is silent.

use std::collections::HashMap;
use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// Deserialize dshape from either array-of-tuples or array-of-single-key-dicts.
///
/// The metadata spec produces `[{"batch": 1}, {"num_features": 84}]` (dict format),
/// while serde's default `Vec<(A, B)>` expects `[["batch", 1]]` (tuple format).
/// This deserializer accepts both.
pub fn deserialize_dshape<'de, D>(deserializer: D) -> Result<Vec<(DimName, usize)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DShapeItem {
        Tuple(DimName, usize),
        Map(HashMap<DimName, usize>),
    }

    let items: Vec<DShapeItem> = Vec::deserialize(deserializer)?;
    items
        .into_iter()
        .map(|item| match item {
            DShapeItem::Tuple(name, size) => Ok((name, size)),
            DShapeItem::Map(map) => {
                if map.len() != 1 {
                    return Err(serde::de::Error::custom(
                        "dshape map entry must have exactly one key",
                    ));
                }
                let (name, size) = map.into_iter().next().unwrap();
                Ok((name, size))
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub struct QuantTuple(pub f32, pub i32);
impl From<QuantTuple> for (f32, i32) {
    fn from(value: QuantTuple) -> Self {
        (value.0, value.1)
    }
}

impl From<(f32, i32)> for QuantTuple {
    fn from(value: (f32, i32)) -> Self {
        QuantTuple(value.0, value.1)
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Segmentation {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Protos {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct MaskCoefficients {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Mask {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Detection {
    #[serde(default)]
    pub anchors: Option<Vec<[f32; 2]>>,
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
    /// Whether box coordinates are normalized to `[0,1]` range.
    /// - `Some(true)`: Coordinates in `[0,1]` range relative to model input
    /// - `Some(false)`: Pixel coordinates relative to model input
    ///   (letterboxed)
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate
    ///   > 1.0)
    #[serde(default)]
    pub normalized: Option<bool>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Scores {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Boxes {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
    /// Whether box coordinates are normalized to `[0,1]` range.
    /// - `Some(true)`: Coordinates in `[0,1]` range relative to model input
    /// - `Some(false)`: Pixel coordinates relative to model input
    ///   (letterboxed)
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate
    ///   > 1.0)
    #[serde(default)]
    pub normalized: Option<bool>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct Classes {
    #[serde(default)]
    pub decoder: DecoderType,
    #[serde(default)]
    pub quantization: Option<QuantTuple>,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(default, deserialize_with = "deserialize_dshape")]
    pub dshape: Vec<(DimName, usize)>,
}

edgefirst_tensor::ef_vocabulary! {
    /// A named tensor axis. The code is the one the C `EF_DIM_NAME_*`
    /// enumerators and the Python `DimName` enum use.
    #[derive(Serialize, Deserialize)]
    pub enum DimName {
        #[serde(rename = "batch")]
        Batch = 0, "batch", BATCH,
        #[serde(rename = "height")]
        Height = 1, "height", HEIGHT,
        #[serde(rename = "width")]
        Width = 2, "width", WIDTH,
        #[serde(rename = "num_classes")]
        NumClasses = 3, "num_classes", NUM_CLASSES,
        #[serde(rename = "num_features")]
        NumFeatures = 4, "num_features", NUM_FEATURES,
        #[serde(rename = "num_boxes")]
        NumBoxes = 5, "num_boxes", NUM_BOXES,
        #[serde(rename = "num_protos")]
        NumProtos = 6, "num_protos", NUM_PROTOS,
        #[serde(rename = "num_anchors_x_features")]
        NumAnchorsXFeatures = 7, "num_anchors_x_features", NUM_ANCHORS_X_FEATURES,
        #[serde(rename = "padding")]
        Padding = 8, "padding", PADDING,
        #[serde(rename = "box_coords")]
        BoxCoords = 9, "box_coords", BOX_COORDS,
        /// Any axis name the HAL does not recognise (e.g. a producer's
        /// `channels` on the input dshape). Preserved so the dshape length still
        /// matches the shape and the axis sorts to the canonical tail in
        /// `swap_axes_if_needed`, but it never satisfies a required-dimension
        /// check. Keeps metadata parsing tolerant of unknown axis names instead
        /// of failing the whole decoder build.
        #[serde(other)]
        Unknown = 10, "unknown", UNKNOWN,
    }
    #[doc(hidden)]
    pub mod dim_name_code;
}

impl Display for DimName {
    /// Formats the DimName for display
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::configs::DimName;
    /// let dim = DimName::Height;
    /// assert_eq!(format!("{}", dim), "height");
    /// # let s = format!("{} {} {} {} {} {} {} {} {} {}", DimName::Batch, DimName::Height, DimName::Width, DimName::NumClasses, DimName::NumFeatures, DimName::NumBoxes, DimName::NumProtos, DimName::NumAnchorsXFeatures, DimName::Padding, DimName::BoxCoords);
    /// # assert_eq!(s, "batch height width num_classes num_features num_boxes num_protos num_anchors_x_features padding box_coords");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

edgefirst_tensor::ef_vocabulary! {
    /// The post-processing family. The code is the one the C
    /// `EF_DECODER_TYPE_*` enumerators and the Python `DecoderType` enum use.
    #[derive(Serialize, Deserialize, Default)]
    pub enum DecoderType {
        #[default]
        #[serde(rename = "ultralytics", alias = "yolov8")]
        Ultralytics = 0, "ultralytics", ULTRALYTICS,
        #[serde(rename = "modelpack")]
        ModelPack = 1, "modelpack", MODELPACK,
    }
    #[doc(hidden)]
    pub mod decoder_type_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// Decoder version for Ultralytics models.
    ///
    /// Specifies the YOLO architecture version, which determines the decoding
    /// strategy:
    /// - `Yolov5`, `Yolov8`, `Yolo11`: Traditional models requiring external
    ///   NMS
    /// - `Yolo26`: End-to-end models with NMS embedded in the model
    ///   architecture
    ///
    /// When `decoder_version` is set to `Yolo26`, the decoder uses end-to-end
    /// model types regardless of the `nms` setting.
    ///
    /// The code is the one the Python `DecoderVersion` enum uses.
    #[derive(Serialize, Deserialize)]
    pub enum DecoderVersion {
        /// YOLOv5 - anchor-free DFL decoder, requires external NMS
        #[serde(rename = "yolov5")]
        Yolov5 = 0, "yolov5", YOLOV5,
        /// YOLOv8 - anchor-free DFL decoder, requires external NMS
        #[serde(rename = "yolov8")]
        Yolov8 = 1, "yolov8", YOLOV8,
        /// YOLO11 - anchor-free DFL decoder, requires external NMS
        #[serde(rename = "yolo11")]
        Yolo11 = 2, "yolo11", YOLO11,
        /// YOLO26 - end-to-end model with embedded NMS (one-to-one matching
        /// heads)
        #[serde(rename = "yolo26")]
        Yolo26 = 3, "yolo26", YOLO26,
    }
    #[doc(hidden)]
    pub mod decoder_version_code;
}

impl DecoderVersion {
    /// Returns true if this version uses end-to-end inference (embedded
    /// NMS).
    pub fn is_end_to_end(&self) -> bool {
        matches!(self, DecoderVersion::Yolo26)
    }
}

edgefirst_tensor::ef_vocabulary! {
    /// NMS (Non-Maximum Suppression) mode for filtering overlapping detections.
    ///
    /// This enum is used with `Option<Nms>`:
    /// - `Some(Nms::Auto)` — resolve from config or fall back to `ClassAware`
    /// - `Some(Nms::ClassAgnostic)` — class-agnostic NMS: suppress overlapping
    ///   boxes regardless of class label
    /// - `Some(Nms::ClassAware)` — class-aware NMS: only suppress boxes that
    ///   share the same class label AND overlap above the IoU threshold
    /// - `None` — bypass NMS entirely (for end-to-end models with embedded NMS)
    ///
    /// The code is the one the C `EF_NMS_*` enumerators and the Python `Nms`
    /// enum use. Code 0 is reserved for `None` (NMS off), which has no
    /// variant here; see [`NMS_OFF_CODE`].
    #[derive(Serialize, Deserialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub enum Nms {
        /// Let the builder resolve NMS mode from the model config (e.g.
        /// `edgefirst.json`).  Falls back to [`Nms::ClassAware`] when no
        /// config specifies a mode.  This is the builder default — callers
        /// should only use an explicit variant when they need to override
        /// the config.
        Auto = 1, "auto", AUTO,
        /// Only suppress boxes with the same class label that overlap (default
        /// concrete behavior; matches trainer and COCO evaluation).
        #[default]
        ClassAware = 2, "class_aware", CLASS_AWARE,
        /// Suppress overlapping boxes regardless of class label.
        ClassAgnostic = 3, "class_agnostic", CLASS_AGNOSTIC,
    }
    #[doc(hidden)]
    pub mod nms_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// The kind of a model output, without its configuration: the tag of
    /// [`crate::ConfigOutput`]. The string form is that enum's serialized
    /// `type`; the code is the one the C `EF_OUTPUT_TYPE_*` enumerators use.
    pub enum OutputType {
        Detection = 0, "detection", DETECTION,
        Boxes = 1, "boxes", BOXES,
        Scores = 2, "scores", SCORES,
        Protos = 3, "protos", PROTOS,
        Segmentation = 4, "segmentation", SEGMENTATION,
        MaskCoefficients = 5, "mask_coefs", MASK_COEFFICIENTS,
        Mask = 6, "masks", MASK,
        Classes = 7, "classes", CLASSES,
    }
    #[doc(hidden)]
    pub mod output_type_code;
}

/// The code for "NMS off" (`Option::<Nms>::None`). No [`Nms`] variant uses it.
pub const NMS_OFF_CODE: u32 = 0;

impl Nms {
    /// The code for an optional mode, `None` (NMS off) included.
    pub const fn option_code(nms: Option<Nms>) -> u32 {
        match nms {
            Some(n) => n.code(),
            None => NMS_OFF_CODE,
        }
    }

    /// Decode an optional mode, [`NMS_OFF_CODE`] included. The outer `None`
    /// is an unknown code; `Some(None)` is NMS off.
    pub fn from_option_code(code: u32) -> Option<Option<Nms>> {
        if code == NMS_OFF_CODE {
            Some(None)
        } else {
            Nms::from_code(code).map(Some)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    ModelPackSegDet {
        boxes: Boxes,
        scores: Scores,
        segmentation: Segmentation,
    },
    ModelPackSegDetSplit {
        detection: Vec<Detection>,
        segmentation: Segmentation,
    },
    ModelPackDet {
        boxes: Boxes,
        scores: Scores,
    },
    ModelPackDetSplit {
        detection: Vec<Detection>,
    },
    ModelPackSeg {
        segmentation: Segmentation,
    },
    YoloDet {
        boxes: Detection,
    },
    YoloSegDet {
        boxes: Detection,
        protos: Protos,
    },
    YoloSplitDet {
        boxes: Boxes,
        scores: Scores,
    },
    YoloSplitSegDet {
        boxes: Boxes,
        scores: Scores,
        mask_coeff: MaskCoefficients,
        protos: Protos,
    },
    /// 2-way split YOLO segmentation detection.
    /// Combined detection tensor (boxes + scores) with separate mask
    /// coefficients and prototype masks.
    /// - detection: [1, nc+4, N] — boxes and scores combined
    /// - mask_coeff: [1, 32, N] — mask coefficients (separate tensor)
    /// - protos: [1, H/4, W/4, 32] — prototype masks
    YoloSegDet2Way {
        boxes: Detection,
        mask_coeff: MaskCoefficients,
        protos: Protos,
    },
    /// End-to-end YOLO detection (post-NMS output from model)
    /// Input shape: (1, N, 6+) where columns are [x1, y1, x2, y2, conf,
    /// class, ...]
    YoloEndToEndDet {
        boxes: Detection,
    },
    /// End-to-end YOLO detection + segmentation (post-NMS output from
    /// model) Input shape: (1, N, 6 + num_protos) where columns are
    /// [x1, y1, x2, y2, conf, class, mask_coeff_0, ..., mask_coeff_31]
    YoloEndToEndSegDet {
        boxes: Detection,
        protos: Protos,
    },
    /// Split end-to-end YOLO detection (onnx2tf splits `[1,N,6]` into 3
    /// tensors) boxes: [batch, N, 4] xyxy, scores: [batch, N, 1],
    /// classes: [batch, N, 1]
    YoloSplitEndToEndDet {
        boxes: Boxes,
        scores: Scores,
        classes: Classes,
    },
    /// Split end-to-end YOLO seg detection (onnx2tf splits into 5
    /// tensors)
    YoloSplitEndToEndSegDet {
        boxes: Boxes,
        scores: Scores,
        classes: Classes,
        mask_coeff: MaskCoefficients,
        protos: Protos,
    },
    /// Per-scale (physical-output-decomposition) YOLO model. The
    /// per-scale subsystem (`crates/decoder/src/per_scale/`) owns
    /// model decoding entirely; this variant exists as a marker so the
    /// `Decoder::model_type` field has a sensible value for per-scale
    /// Decoders that bypass the legacy `ModelType`-driven dispatch.
    PerScale,
}

#[cfg(test)]
mod vocabulary_tests {
    use super::*;

    #[test]
    fn nms_off_is_code_zero_and_no_variant_claims_it() {
        assert_eq!(Nms::from_code(NMS_OFF_CODE), None);
        assert_eq!(Nms::option_code(None), NMS_OFF_CODE);
        assert_eq!(Nms::from_option_code(NMS_OFF_CODE), Some(None));
        for &n in Nms::all() {
            assert_ne!(n.code(), NMS_OFF_CODE);
            assert_eq!(Nms::from_option_code(n.code()), Some(Some(n)));
        }
        assert_eq!(Nms::from_option_code(99), None);
    }

    #[test]
    fn serde_names_match_the_vocabulary_strings() {
        for &n in Nms::all() {
            assert_eq!(serde_json::to_value(n).unwrap(), n.as_str());
        }
        for &d in DecoderType::all() {
            assert_eq!(serde_json::to_value(d).unwrap(), d.as_str());
        }
        for &v in DecoderVersion::all() {
            assert_eq!(serde_json::to_value(v).unwrap(), v.as_str());
        }
        for &d in DimName::all() {
            assert_eq!(d.to_string(), d.as_str());
            if d != DimName::Unknown {
                assert_eq!(serde_json::to_value(d).unwrap(), d.as_str());
            }
        }
    }
}
