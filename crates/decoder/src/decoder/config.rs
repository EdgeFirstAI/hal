// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::configs::{self, DimName, QuantTuple};
use serde::{Deserialize, Serialize};

/// Used to represent the outputs in the model configuration.
/// # Examples
/// ```rust
/// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, ConfigOutputs};
/// # fn main() -> DecoderResult<()> {
/// let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json"));
/// let config: ConfigOutputs = serde_json::from_str(config_json)?;
/// let decoder = DecoderBuilder::new().with_config(config).build()?;
///
/// # Ok(())
/// # }
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub struct ConfigOutputs {
    #[serde(default)]
    pub outputs: Vec<ConfigOutput>,
    /// NMS mode from config file. When present, overrides the builder's NMS
    /// setting.
    /// - `Some(Nms::ClassAgnostic)` — class-agnostic NMS: suppress overlapping
    ///   boxes regardless of class
    /// - `Some(Nms::ClassAware)` — class-aware NMS: only suppress boxes with
    ///   the same class
    /// - `None` — use builder default or skip NMS (user handles it externally)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nms: Option<configs::Nms>,
    /// Decoder version for Ultralytics models. Determines the decoding
    /// strategy.
    /// - `Some(Yolo26)` — end-to-end model with embedded NMS
    /// - `Some(Yolov5/Yolov8/Yolo11)` — traditional models requiring external
    ///   NMS
    /// - `None` — infer from other settings (legacy behavior)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_version: Option<configs::DecoderVersion>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ConfigOutput {
    #[serde(rename = "detection")]
    Detection(configs::Detection),
    #[serde(rename = "masks")]
    Mask(configs::Mask),
    #[serde(rename = "segmentation")]
    Segmentation(configs::Segmentation),
    #[serde(rename = "protos")]
    Protos(configs::Protos),
    #[serde(rename = "scores")]
    Scores(configs::Scores),
    #[serde(rename = "boxes")]
    Boxes(configs::Boxes),
    #[serde(rename = "mask_coefs", alias = "mask_coefficients")]
    MaskCoefficients(configs::MaskCoefficients),
    #[serde(rename = "classes")]
    Classes(configs::Classes),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ConfigOutputRef<'a> {
    Detection(&'a configs::Detection),
    Mask(&'a configs::Mask),
    Segmentation(&'a configs::Segmentation),
    Protos(&'a configs::Protos),
    Scores(&'a configs::Scores),
    Boxes(&'a configs::Boxes),
    MaskCoefficients(&'a configs::MaskCoefficients),
    Classes(&'a configs::Classes),
}

impl<'a> ConfigOutputRef<'a> {
    pub(super) fn decoder(&self) -> configs::DecoderType {
        match self {
            ConfigOutputRef::Detection(v) => v.decoder,
            ConfigOutputRef::Mask(v) => v.decoder,
            ConfigOutputRef::Segmentation(v) => v.decoder,
            ConfigOutputRef::Protos(v) => v.decoder,
            ConfigOutputRef::Scores(v) => v.decoder,
            ConfigOutputRef::Boxes(v) => v.decoder,
            ConfigOutputRef::MaskCoefficients(v) => v.decoder,
            ConfigOutputRef::Classes(v) => v.decoder,
        }
    }

    pub(super) fn dshape(&self) -> &[(DimName, usize)] {
        match self {
            ConfigOutputRef::Detection(v) => &v.dshape,
            ConfigOutputRef::Mask(v) => &v.dshape,
            ConfigOutputRef::Segmentation(v) => &v.dshape,
            ConfigOutputRef::Protos(v) => &v.dshape,
            ConfigOutputRef::Scores(v) => &v.dshape,
            ConfigOutputRef::Boxes(v) => &v.dshape,
            ConfigOutputRef::MaskCoefficients(v) => &v.dshape,
            ConfigOutputRef::Classes(v) => &v.dshape,
        }
    }
}

impl<'a> From<&'a configs::Detection> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let detection_config = configs::Detection {
    ///     anchors: None,
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 84, 8400],
    ///     dshape: Vec::new(),
    ///     normalized: Some(true),
    /// };
    /// let output: ConfigOutputRef = (&detection_config).into();
    /// ```
    fn from(v: &'a configs::Detection) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Detection(v)
    }
}

impl<'a> From<&'a configs::Mask> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let mask = configs::Mask {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: None,
    ///     shape: vec![1, 160, 160, 1],
    ///     dshape: Vec::new(),
    /// };
    /// let output: ConfigOutputRef = (&mask).into();
    /// ```
    fn from(v: &'a configs::Mask) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Mask(v)
    }
}

impl<'a> From<&'a configs::Segmentation> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let seg = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: None,
    ///     shape: vec![1, 160, 160, 3],
    ///     dshape: Vec::new(),
    /// };
    /// let output: ConfigOutputRef = (&seg).into();
    /// ```
    fn from(v: &'a configs::Segmentation) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Segmentation(v)
    }
}

impl<'a> From<&'a configs::Protos> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let protos = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 160, 160, 32],
    ///     dshape: Vec::new(),
    /// };
    /// let output: ConfigOutputRef = (&protos).into();
    /// ```
    fn from(v: &'a configs::Protos) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Protos(v)
    }
}

impl<'a> From<&'a configs::Scores> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let scores = configs::Scores {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 40, 8400],
    ///     dshape: Vec::new(),
    /// };
    /// let output: ConfigOutputRef = (&scores).into();
    /// ```
    fn from(v: &'a configs::Scores) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Scores(v)
    }
}

impl<'a> From<&'a configs::Boxes> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let boxes = configs::Boxes {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 4, 8400],
    ///     dshape: Vec::new(),
    ///     normalized: Some(true),
    /// };
    /// let output: ConfigOutputRef = (&boxes).into();
    /// ```
    fn from(v: &'a configs::Boxes) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Boxes(v)
    }
}

impl<'a> From<&'a configs::MaskCoefficients> for ConfigOutputRef<'a> {
    /// Converts from references of config structs to ConfigOutputRef
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutputRef};
    /// let mask_coefficients = configs::MaskCoefficients {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 32, 8400],
    ///     dshape: Vec::new(),
    /// };
    /// let output: ConfigOutputRef = (&mask_coefficients).into();
    /// ```
    fn from(v: &'a configs::MaskCoefficients) -> ConfigOutputRef<'a> {
        ConfigOutputRef::MaskCoefficients(v)
    }
}

impl<'a> From<&'a configs::Classes> for ConfigOutputRef<'a> {
    fn from(v: &'a configs::Classes) -> ConfigOutputRef<'a> {
        ConfigOutputRef::Classes(v)
    }
}

impl ConfigOutput {
    /// Returns the shape of the output.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutput};
    /// let detection_config = configs::Detection {
    ///     anchors: None,
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 84, 8400],
    ///     dshape: Vec::new(),
    ///     normalized: Some(true),
    /// };
    /// let output = ConfigOutput::Detection(detection_config);
    /// assert_eq!(output.shape(), &[1, 84, 8400]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            ConfigOutput::Detection(detection) => &detection.shape,
            ConfigOutput::Mask(mask) => &mask.shape,
            ConfigOutput::Segmentation(segmentation) => &segmentation.shape,
            ConfigOutput::Scores(scores) => &scores.shape,
            ConfigOutput::Boxes(boxes) => &boxes.shape,
            ConfigOutput::Protos(protos) => &protos.shape,
            ConfigOutput::MaskCoefficients(mask_coefficients) => &mask_coefficients.shape,
            ConfigOutput::Classes(classes) => &classes.shape,
        }
    }

    /// Returns the decoder type of the output.
    ///    
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutput};
    /// let detection_config = configs::Detection {
    ///     anchors: None,
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: None,
    ///     shape: vec![1, 84, 8400],
    ///     dshape: Vec::new(),
    ///     normalized: Some(true),
    /// };
    /// let output = ConfigOutput::Detection(detection_config);
    /// assert_eq!(output.decoder(), &configs::DecoderType::Ultralytics);
    /// ```
    pub fn decoder(&self) -> &configs::DecoderType {
        match self {
            ConfigOutput::Detection(detection) => &detection.decoder,
            ConfigOutput::Mask(mask) => &mask.decoder,
            ConfigOutput::Segmentation(segmentation) => &segmentation.decoder,
            ConfigOutput::Scores(scores) => &scores.decoder,
            ConfigOutput::Boxes(boxes) => &boxes.decoder,
            ConfigOutput::Protos(protos) => &protos.decoder,
            ConfigOutput::MaskCoefficients(mask_coefficients) => &mask_coefficients.decoder,
            ConfigOutput::Classes(classes) => &classes.decoder,
        }
    }

    /// Returns the quantization of the output.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutput};
    /// let detection_config = configs::Detection {
    ///   anchors: None,
    ///   decoder: configs::DecoderType::Ultralytics,
    ///   quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///   shape: vec![1, 84, 8400],
    ///   dshape: Vec::new(),
    ///   normalized: Some(true),
    /// };
    /// let output = ConfigOutput::Detection(detection_config);
    /// assert_eq!(output.quantization(),
    /// Some(configs::QuantTuple(0.012345,26))); ```
    pub fn quantization(&self) -> Option<QuantTuple> {
        match self {
            ConfigOutput::Detection(detection) => detection.quantization,
            ConfigOutput::Mask(mask) => mask.quantization,
            ConfigOutput::Segmentation(segmentation) => segmentation.quantization,
            ConfigOutput::Scores(scores) => scores.quantization,
            ConfigOutput::Boxes(boxes) => boxes.quantization,
            ConfigOutput::Protos(protos) => protos.quantization,
            ConfigOutput::MaskCoefficients(mask_coefficients) => mask_coefficients.quantization,
            ConfigOutput::Classes(classes) => classes.quantization,
        }
    }
}
