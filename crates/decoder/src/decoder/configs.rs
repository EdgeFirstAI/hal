// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

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
    /// Whether box coordinates are normalized to [0,1] range.
    /// - `Some(true)`: Coordinates in [0,1] range relative to model input
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
    /// Whether box coordinates are normalized to [0,1] range.
    /// - `Some(true)`: Coordinates in [0,1] range relative to model input
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

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy, Hash, Eq)]
pub enum DimName {
    #[serde(rename = "batch")]
    Batch,
    #[serde(rename = "height")]
    Height,
    #[serde(rename = "width")]
    Width,
    #[serde(rename = "num_classes")]
    NumClasses,
    #[serde(rename = "num_features")]
    NumFeatures,
    #[serde(rename = "num_boxes")]
    NumBoxes,
    #[serde(rename = "num_protos")]
    NumProtos,
    #[serde(rename = "num_anchors_x_features")]
    NumAnchorsXFeatures,
    #[serde(rename = "padding")]
    Padding,
    #[serde(rename = "box_coords")]
    BoxCoords,
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
        match self {
            DimName::Batch => write!(f, "batch"),
            DimName::Height => write!(f, "height"),
            DimName::Width => write!(f, "width"),
            DimName::NumClasses => write!(f, "num_classes"),
            DimName::NumFeatures => write!(f, "num_features"),
            DimName::NumBoxes => write!(f, "num_boxes"),
            DimName::NumProtos => write!(f, "num_protos"),
            DimName::NumAnchorsXFeatures => write!(f, "num_anchors_x_features"),
            DimName::Padding => write!(f, "padding"),
            DimName::BoxCoords => write!(f, "box_coords"),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy, Hash, Eq, Default)]
pub enum DecoderType {
    #[serde(rename = "modelpack")]
    ModelPack,
    #[default]
    #[serde(rename = "ultralytics", alias = "yolov8")]
    Ultralytics,
}

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
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy, Hash, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DecoderVersion {
    /// YOLOv5 - anchor-free DFL decoder, requires external NMS
    #[serde(rename = "yolov5")]
    Yolov5,
    /// YOLOv8 - anchor-free DFL decoder, requires external NMS
    #[serde(rename = "yolov8")]
    Yolov8,
    /// YOLO11 - anchor-free DFL decoder, requires external NMS
    #[serde(rename = "yolo11")]
    Yolo11,
    /// YOLO26 - end-to-end model with embedded NMS (one-to-one matching
    /// heads)
    #[serde(rename = "yolo26")]
    Yolo26,
}

impl DecoderVersion {
    /// Returns true if this version uses end-to-end inference (embedded
    /// NMS).
    pub fn is_end_to_end(&self) -> bool {
        matches!(self, DecoderVersion::Yolo26)
    }
}

/// NMS (Non-Maximum Suppression) mode for filtering overlapping detections.
///
/// This enum is used with `Option<Nms>`:
/// - `Some(Nms::ClassAgnostic)` — class-agnostic NMS (default): suppress
///   overlapping boxes regardless of class label
/// - `Some(Nms::ClassAware)` — class-aware NMS: only suppress boxes that
///   share the same class label AND overlap above the IoU threshold
/// - `None` — bypass NMS entirely (for end-to-end models with embedded NMS)
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy, Hash, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Nms {
    /// Suppress overlapping boxes regardless of class label (default HAL
    /// behavior)
    #[default]
    ClassAgnostic,
    /// Only suppress boxes with the same class label that overlap
    ClassAware,
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
    /// Split end-to-end YOLO detection (onnx2tf splits [1,N,6] into 3
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Raw = 0,
    Int8 = 1,
    UInt8 = 2,
    Int16 = 3,
    UInt16 = 4,
    Float16 = 5,
    Int32 = 6,
    UInt32 = 7,
    Float32 = 8,
    Int64 = 9,
    UInt64 = 10,
    Float64 = 11,
    String = 12,
}
