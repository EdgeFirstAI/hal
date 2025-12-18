// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array3, ArrayView, ArrayViewD, Dimension, s};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::{
    DecoderError, DetectBox, Quantization, Segmentation, XYWH,
    configs::{DecoderType, ModelType, QuantTuple},
    dequantize_ndarray,
    modelpack::{
        ModelPackDetectionConfig, decode_modelpack_det, decode_modelpack_float,
        decode_modelpack_split_float,
    },
    yolo::{
        decode_yolo_det, decode_yolo_det_float, decode_yolo_segdet_float, decode_yolo_segdet_quant,
        decode_yolo_split_det_float, decode_yolo_split_det_quant, decode_yolo_split_segdet_float,
        impl_yolo_split_segdet_quant_get_boxes, impl_yolo_split_segdet_quant_process_masks,
    },
};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
/// Used to represent the outputs in the model configuration.
/// # Examples
/// ```rust
/// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, ConfigOutputs};
/// # fn main() -> DecoderResult<()> {
/// let config_json = include_str!("../../../testdata/modelpack_split.json");
/// let config: ConfigOutputs = serde_json::from_str(config_json)?;
/// let decoder = DecoderBuilder::new().with_config(config).build()?;
///
/// # Ok(())
/// # }
pub struct ConfigOutputs {
    pub outputs: Vec<ConfigOutput>,
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
    #[serde(rename = "mask_coefficients")]
    MaskCoefficients(configs::MaskCoefficients),
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
    ///     channels_first: false,
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
    ///     channels_first: false,
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
        }
    }

    /// Returns the quantization of the output.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{configs, ConfigOutput};
    /// let detection_config = configs::Detection {
    ///    anchors: None,
    ///   decoder: configs::DecoderType::Ultralytics,
    ///   quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///  shape: vec![1, 84, 8400],
    ///   channels_first: false,
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
        }
    }
}

pub mod configs {
    use serde::{Deserialize, Serialize};

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

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Segmentation {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Protos {
        pub decoder: DecoderType,
        #[serde(flatten)]
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct MaskCoefficients {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Mask {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Detection {
        pub anchors: Option<Vec<[f32; 2]>>,
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Scores {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Boxes {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub enum DecoderType {
        #[serde(rename = "modelpack")]
        ModelPack,
        #[serde(rename = "ultralytics")]
        Ultralytics,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderBuilder {
    config_src: Option<ConfigSource>,
    iou_threshold: f32,
    score_threshold: f32,
}

#[derive(Debug, Clone, PartialEq)]
enum ConfigSource {
    Yaml(String),
    Json(String),
    Config(ConfigOutputs),
}

impl Default for DecoderBuilder {
    /// Creates a default DecoderBuilder with no configuration and 0.5 score
    /// threshold and 0.5 OU threshold.
    ///
    /// A valid confguration must be provided before building the Decoder.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #  let config_yaml = include_str!("../../../testdata/modelpack_split.yaml").to_string();
    /// let decoder = DecoderBuilder::default()
    ///     .with_config_yaml_str(config_yaml)
    ///     .build()?;
    /// assert_eq!(decoder.score_threshold, 0.5);
    /// assert_eq!(decoder.iou_threshold, 0.5);
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn default() -> Self {
        Self {
            config_src: None,
            iou_threshold: 0.5,
            score_threshold: 0.5,
        }
    }
}

impl DecoderBuilder {
    /// Creates a default DecoderBuilder with no configuration and 0.5 score
    /// threshold and 0.5 OU threshold.
    ///
    /// A valid confguration must be provided before building the Decoder.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #  let config_yaml = include_str!("../../../testdata/modelpack_split.yaml").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yaml_str(config_yaml)
    ///     .build()?;
    /// assert_eq!(decoder.score_threshold, 0.5);
    /// assert_eq!(decoder.iou_threshold, 0.5);
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a model configuration in YAML format. Does not check if the string
    /// is a correct configuration file. Use `DecoderBuilder.build()` to
    /// deserialize the YAML and parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// let config_yaml = include_str!("../../../testdata/modelpack_split.yaml").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yaml_str(config_yaml)
    ///     .build()?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yaml_str(mut self, yaml_str: String) -> Self {
        self.config_src.replace(ConfigSource::Yaml(yaml_str));
        self
    }

    /// Loads a model configuration in JSON format. Does not check if the string
    /// is a correct configuration file. Use `DecoderBuilder.build()` to
    /// deserialize the JSON and parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// let config_json = include_str!("../../../testdata/modelpack_split.json").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .build()?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_json_str(mut self, json_str: String) -> Self {
        self.config_src.replace(ConfigSource::Json(json_str));
        self
    }

    /// Loads a model configuration. Does not check if the configuration is
    /// correct. Intended to be used when the user needs control over the
    /// deserialize of the configuration information. Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// let config_json = include_str!("../../../testdata/modelpack_split.json");
    /// let config = serde_json::from_str(config_json)?;
    /// let decoder = DecoderBuilder::new().with_config(config).build()?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config(mut self, config: ConfigOutputs) -> Self {
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a YOLO detection model configuration.  Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yolo_det(configs::Detection {
    ///         anchors: None,
    ///         decoder: configs::DecoderType::Ultralytics,
    ///         quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///         shape: vec![1, 84, 8400],
    ///         channels_first: false,
    ///     })
    ///     .build()?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_det(mut self, boxes: configs::Detection) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Detection(boxes)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a YOLO split detection model configuration.  Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let boxes_config = configs::Boxes {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 4, 8400],
    ///     channels_first: false,
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 80, 8400],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yolo_split_det(boxes_config, scores_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_split_det(
        mut self,
        boxes: configs::Boxes,
        scores: configs::Scores,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Boxes(boxes), ConfigOutput::Scores(scores)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a YOLO segmentation model configuration.  Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let seg_config = configs::Detection {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 116, 8400],
    ///     channels_first: false,
    ///     anchors: None,
    /// };
    /// let protos_config = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 160, 160, 32],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yolo_segdet(seg_config, protos_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_segdet(
        mut self,
        boxes: configs::Detection,
        protos: configs::Protos,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Detection(boxes), ConfigOutput::Protos(protos)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a YOLO split segmentation model configuration.  Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let boxes_config = configs::Boxes {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 4, 8400],
    ///     channels_first: false,
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.012345, 14)),
    ///     shape: vec![1, 80, 8400],
    ///     channels_first: false,
    /// };
    /// let mask_config = configs::MaskCoefficients {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, 125)),
    ///     shape: vec![1, 32, 8400],
    ///     channels_first: false,
    /// };
    /// let protos_config = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 160, 160, 32],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yolo_split_segdet(boxes_config, scores_config, mask_config, protos_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_split_segdet(
        mut self,
        boxes: configs::Boxes,
        scores: configs::Scores,
        mask_coefficients: configs::MaskCoefficients,
        protos: configs::Protos,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![
                ConfigOutput::Boxes(boxes),
                ConfigOutput::Scores(scores),
                ConfigOutput::MaskCoefficients(mask_coefficients),
                ConfigOutput::Protos(protos),
            ],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a ModelPack detection model configuration.  Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let boxes_config = configs::Boxes {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 8400, 1, 4],
    ///     channels_first: false,
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 8400, 3],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_modelpack_det(boxes_config, scores_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_modelpack_det(
        mut self,
        boxes: configs::Boxes,
        scores: configs::Scores,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Boxes(boxes), ConfigOutput::Scores(scores)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a ModelPack split detection model configuration. Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let config0 = configs::Detection {
    ///     anchors: Some(vec![
    ///         [0.13750000298023224, 0.2074074000120163],
    ///         [0.2541666626930237, 0.21481481194496155],
    ///         [0.23125000298023224, 0.35185185074806213],
    ///     ]),
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 17, 30, 18],
    ///     channels_first: false,
    /// };
    /// let config1 = configs::Detection {
    ///     anchors: Some(vec![
    ///         [0.36666667461395264, 0.31481480598449707],
    ///         [0.38749998807907104, 0.4740740656852722],
    ///         [0.5333333611488342, 0.644444465637207],
    ///     ]),
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 9, 15, 18],
    ///     channels_first: false,
    /// };
    ///
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_modelpack_det_split(vec![config0, config1])
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_modelpack_det_split(mut self, boxes: Vec<configs::Detection>) -> Self {
        let outputs = boxes.into_iter().map(ConfigOutput::Detection).collect();
        let config = ConfigOutputs { outputs };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a ModelPack segmentation detection model configuration. Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let boxes_config = configs::Boxes {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///     shape: vec![1, 8400, 1, 4],
    ///     channels_first: false,
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 8400, 2],
    ///     channels_first: false,
    /// };
    /// let seg_config = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 640, 640, 3],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_modelpack_segdet(boxes_config, scores_config, seg_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_modelpack_segdet(
        mut self,
        boxes: configs::Boxes,
        scores: configs::Scores,
        segmentation: configs::Segmentation,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![
                ConfigOutput::Boxes(boxes),
                ConfigOutput::Scores(scores),
                ConfigOutput::Segmentation(segmentation),
            ],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a ModelPack segmentation split detection model configuration. Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let config0 = configs::Detection {
    ///     anchors: Some(vec![
    ///         [0.36666667461395264, 0.31481480598449707],
    ///         [0.38749998807907104, 0.4740740656852722],
    ///         [0.5333333611488342, 0.644444465637207],
    ///     ]),
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.08547406643629074, 174)),
    ///     shape: vec![1, 9, 15, 18],
    ///     channels_first: false,
    /// };
    /// let config1 = configs::Detection {
    ///     anchors: Some(vec![
    ///         [0.13750000298023224, 0.2074074000120163],
    ///         [0.2541666626930237, 0.21481481194496155],
    ///         [0.23125000298023224, 0.35185185074806213],
    ///     ]),
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.09929127991199493, 183)),
    ///     shape: vec![1, 17, 30, 18],
    ///     channels_first: false,
    /// };
    /// let seg_config = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 640, 640, 2],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_modelpack_segdet_split(vec![config0, config1], seg_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_modelpack_segdet_split(
        mut self,
        boxes: Vec<configs::Detection>,
        segmentation: configs::Segmentation,
    ) -> Self {
        let mut outputs = boxes
            .into_iter()
            .map(ConfigOutput::Detection)
            .collect::<Vec<_>>();
        outputs.push(ConfigOutput::Segmentation(segmentation));
        let config = ConfigOutputs { outputs };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Loads a ModelPack segmentation model configuration. Use
    /// `DecoderBuilder.build()` to parse the model configuration.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
    /// # fn main() -> DecoderResult<()> {
    /// let seg_config = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 640, 640, 3],
    ///     channels_first: false,
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_modelpack_seg(seg_config)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_modelpack_seg(mut self, segmentation: configs::Segmentation) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Segmentation(segmentation)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Sets the scores threshold of the decoder
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!("../../../testdata/modelpack_split.json").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_score_threshold(0.654)
    ///     .build()?;
    /// assert_eq!(decoder.score_threshold, 0.654);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = score_threshold;
        self
    }

    /// Sets the IOU threshold of the decoder
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!("../../../testdata/modelpack_split.json").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_iou_threshold(0.654)
    ///     .build()?;
    /// assert_eq!(decoder.iou_threshold, 0.654);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    /// Builds the decoder with the given settings. If the config is a JSON or
    /// YAML string, this will deserialize the JSON or YAML and then parse the
    /// configuration information.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!("../../../testdata/modelpack_split.json").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_score_threshold(0.654)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<Decoder, DecoderError> {
        let config = match self.config_src {
            Some(ConfigSource::Json(s)) => serde_json::from_str(&s)?,
            Some(ConfigSource::Yaml(s)) => serde_yaml::from_str(&s)?,
            Some(ConfigSource::Config(c)) => c,
            None => return Err(DecoderError::NoConfig),
        };
        let model_type = Self::get_model_type(config.outputs)?;
        Ok(Decoder {
            model_type,
            iou_threshold: self.iou_threshold,
            score_threshold: self.score_threshold,
        })
    }

    fn get_model_type(configs: Vec<ConfigOutput>) -> Result<ModelType, DecoderError> {
        // yolo or modelpack
        let mut yolo = false;
        let mut modelpack = false;
        for c in &configs {
            match c.decoder() {
                DecoderType::ModelPack => modelpack = true,
                DecoderType::Ultralytics => yolo = true,
            }
        }
        match (modelpack, yolo) {
            (true, true) => Err(DecoderError::InvalidConfig(
                "Both ModelPack and Yolo outputs found in config".to_string(),
            )),
            (true, false) => Self::get_model_type_modelpack(configs),
            (false, true) => Self::get_model_type_yolo(configs),
            (false, false) => Err(DecoderError::InvalidConfig(
                "No outputs found in config".to_string(),
            )),
        }
    }

    fn get_model_type_yolo(configs: Vec<ConfigOutput>) -> Result<ModelType, DecoderError> {
        let mut boxes = None;
        let mut protos = None;
        let mut split_boxes = None;
        let mut split_scores = None;
        let mut split_mask_coeff = None;
        for c in configs {
            match c {
                ConfigOutput::Detection(detection) => boxes = Some(detection),
                ConfigOutput::Segmentation(_) => {
                    return Err(DecoderError::InvalidConfig(
                        "Invalid Segmentation output with Yolo decoder".to_string(),
                    ));
                }
                ConfigOutput::Protos(protos_) => protos = Some(protos_),
                ConfigOutput::Mask(_) => {
                    return Err(DecoderError::InvalidConfig(
                        "Invalid Mask output with Yolo decoder".to_string(),
                    ));
                }
                ConfigOutput::Scores(scores) => split_scores = Some(scores),
                ConfigOutput::Boxes(boxes) => split_boxes = Some(boxes),
                ConfigOutput::MaskCoefficients(mask_coeff) => split_mask_coeff = Some(mask_coeff),
            }
        }

        if let Some(boxes) = boxes {
            if let Some(protos) = protos {
                Self::verify_yolo_seg_det(&boxes, &protos)?;
                Ok(ModelType::YoloSegDet { boxes, protos })
            } else {
                Self::verify_yolo_det(&boxes)?;
                Ok(ModelType::YoloDet { boxes })
            }
        } else if let Some(boxes) = split_boxes
            && let Some(scores) = split_scores
        {
            if let Some(mask_coeff) = split_mask_coeff
                && let Some(protos) = protos
            {
                Self::verify_yolo_split_segdet(&boxes, &scores, &mask_coeff, &protos)?;
                Ok(ModelType::YoloSplitSegDet {
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                })
            } else {
                Self::verify_yolo_split_det(&boxes, &scores)?;
                Ok(ModelType::YoloSplitDet { boxes, scores })
            }
        } else {
            Err(DecoderError::InvalidConfig(
                "Invalid Yolo model outputs".to_string(),
            ))
        }
    }

    fn verify_yolo_det(boxes: &configs::Detection) -> Result<(), DecoderError> {
        if boxes.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Detection shape {:?}",
                boxes.shape
            )));
        }
        Ok(())
    }

    fn verify_yolo_seg_det(
        detection: &configs::Detection,
        protos: &configs::Protos,
    ) -> Result<(), DecoderError> {
        if detection.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Detection shape {:?}",
                detection.shape
            )));
        }
        if protos.shape.len() != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Protos shape {:?}",
                protos.shape
            )));
        }

        let seg_channels = if detection.channels_first {
            detection.shape[2]
        } else {
            detection.shape[1]
        };
        let protos_channels = if protos.channels_first {
            protos.shape[1]
        } else {
            protos.shape[3]
        };

        if protos_channels + 4 >= seg_channels {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo Protos channels {} incompatible with Detection channels {}",
                protos_channels, seg_channels
            )));
        }
        Ok(())
    }

    fn verify_yolo_split_det(
        boxes: &configs::Boxes,
        scores: &configs::Scores,
    ) -> Result<(), DecoderError> {
        if boxes.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Boxes shape {:?}",
                boxes.shape
            )));
        }
        if scores.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Scores shape {:?}",
                scores.shape
            )));
        }

        let boxes_dim = if boxes.channels_first {
            boxes.shape[2]
        } else {
            boxes.shape[1]
        };

        if boxes_dim != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Boxes dimension {}, expected 4",
                boxes_dim
            )));
        }

        let boxes_num = if boxes.channels_first {
            boxes.shape[1]
        } else {
            boxes.shape[2]
        };
        let scores_num = if scores.channels_first {
            scores.shape[1]
        } else {
            scores.shape[2]
        };

        if boxes_num != scores_num {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo Split Detection Boxes num {} incompatible with Scores num {}",
                boxes_num, scores_num
            )));
        }

        Ok(())
    }

    fn verify_yolo_split_segdet(
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
    ) -> Result<(), DecoderError> {
        if boxes.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Boxes shape {:?}",
                boxes.shape
            )));
        }
        if scores.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Scores shape {:?}",
                scores.shape
            )));
        }

        if mask_coeff.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Mask Coefficients shape {:?}",
                mask_coeff.shape
            )));
        }

        if protos.shape.len() != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Protos shape {:?}",
                mask_coeff.shape
            )));
        }

        let boxes_num = if boxes.channels_first {
            boxes.shape[1]
        } else {
            boxes.shape[2]
        };

        let scores_num = if scores.channels_first {
            scores.shape[1]
        } else {
            scores.shape[2]
        };

        let mask_num = if mask_coeff.channels_first {
            mask_coeff.shape[1]
        } else {
            mask_coeff.shape[2]
        };

        let mask_channels = if mask_coeff.channels_first {
            mask_coeff.shape[2]
        } else {
            mask_coeff.shape[1]
        };

        let proto_channels = if protos.channels_first {
            protos.shape[1]
        } else {
            protos.shape[3]
        };

        if boxes_num != scores_num {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo Split Detection Boxes num {} incompatible with Scores num {}",
                boxes_num, scores_num
            )));
        }

        if boxes_num != mask_num {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo Split Detection Boxes num {} incompatible with Mask Coefficients num {}",
                boxes_num, mask_num
            )));
        }

        if proto_channels != mask_channels {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo Protos channels {} incompatible with Mask Coefficients channels {}",
                proto_channels, mask_channels
            )));
        }

        Ok(())
    }

    fn get_model_type_modelpack(configs: Vec<ConfigOutput>) -> Result<ModelType, DecoderError> {
        let mut split_decoders = Vec::new();
        let mut segment_ = None;
        let mut scores_ = None;
        let mut boxes_ = None;
        for c in configs {
            match c {
                ConfigOutput::Detection(detection) => split_decoders.push(detection),
                ConfigOutput::Segmentation(segmentation) => segment_ = Some(segmentation),
                ConfigOutput::Mask(_) => {}
                ConfigOutput::Protos(_) => {
                    return Err(DecoderError::InvalidConfig(
                        "ModelPack should not have protos".to_string(),
                    ));
                }
                ConfigOutput::Scores(scores) => scores_ = Some(scores),
                ConfigOutput::Boxes(boxes) => boxes_ = Some(boxes),
                ConfigOutput::MaskCoefficients(_) => {
                    return Err(DecoderError::InvalidConfig(
                        "ModelPack should not have mask coefficients".to_string(),
                    ));
                }
            }
        }

        if let Some(segmentation) = segment_ {
            if !split_decoders.is_empty() {
                let classes = Self::verify_modelpack_split_det(&split_decoders)?;
                Self::verify_modelpack_seg(&segmentation, Some(classes))?;
                Ok(ModelType::ModelPackSegDetSplit {
                    detection: split_decoders,
                    segmentation,
                })
            } else if let Some(scores) = scores_
                && let Some(boxes) = boxes_
            {
                let classes = Self::verify_modelpack_det(&boxes, &scores)?;
                Self::verify_modelpack_seg(&segmentation, Some(classes))?;
                Ok(ModelType::ModelPackSegDet {
                    boxes,
                    scores,
                    segmentation,
                })
            } else {
                Self::verify_modelpack_seg(&segmentation, None)?;
                Ok(ModelType::ModelPackSeg { segmentation })
            }
        } else if !split_decoders.is_empty() {
            Self::verify_modelpack_split_det(&split_decoders)?;
            Ok(ModelType::ModelPackDetSplit {
                detection: split_decoders,
            })
        } else if let Some(scores) = scores_
            && let Some(boxes) = boxes_
        {
            Self::verify_modelpack_det(&boxes, &scores)?;
            Ok(ModelType::ModelPackDet { boxes, scores })
        } else {
            Err(DecoderError::InvalidConfig(
                "Invalid ModelPack model outputs".to_string(),
            ))
        }
    }

    fn verify_modelpack_det(
        boxes: &configs::Boxes,
        scores: &configs::Scores,
    ) -> Result<usize, DecoderError> {
        if boxes.shape.len() != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Boxes shape {:?}",
                boxes.shape
            )));
        }
        if scores.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Scores shape {:?}",
                scores.shape
            )));
        }

        if boxes.shape[2] != 1 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Boxes dimension 2: {}, expected 1",
                boxes.shape[2]
            )));
        }

        let boxes_dim = if boxes.channels_first {
            boxes.shape[1]
        } else {
            boxes.shape[3]
        };

        if boxes_dim != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Boxes dimension {}, expected 4",
                boxes_dim
            )));
        }

        let boxes_num = if boxes.channels_first {
            boxes.shape[3]
        } else {
            boxes.shape[1]
        };
        let scores_num = if scores.channels_first {
            scores.shape[2]
        } else {
            scores.shape[1]
        };

        if boxes_num != scores_num {
            return Err(DecoderError::InvalidConfig(format!(
                "ModelPack Detection Boxes num {} incompatible with Scores num {}",
                boxes_num, scores_num
            )));
        }

        let num_classes = if scores.channels_first {
            scores.shape[1]
        } else {
            scores.shape[2]
        };

        Ok(num_classes)
    }

    fn verify_modelpack_split_det(boxes: &[configs::Detection]) -> Result<usize, DecoderError> {
        let mut num_classes = None;
        for b in boxes {
            let Some(num_anchors) = b.anchors.as_ref().map(|a| a.len()) else {
                return Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection missing anchors".to_string(),
                ));
            };

            if num_anchors == 0 {
                return Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection has zero anchors".to_string(),
                ));
            }

            if b.shape.len() != 4 {
                return Err(DecoderError::InvalidConfig(format!(
                    "Invalid ModelPack Split Detection shape {:?}",
                    b.shape
                )));
            };

            let boxes_channels = if b.channels_first {
                b.shape[1]
            } else {
                b.shape[3]
            };

            if boxes_channels <= num_anchors * 5 {
                return Err(DecoderError::InvalidConfig(format!(
                    "Invalid ModelPack Split Detection shape {:?}: channels {} not greater than number of anchors * 5 = {}",
                    b.shape,
                    boxes_channels,
                    num_anchors * 5,
                )));
            };

            if !boxes_channels.is_multiple_of(num_anchors) {
                return Err(DecoderError::InvalidConfig(format!(
                    "Invalid ModelPack Split Detection shape {:?}: channels {} not a multiple of number of anchors {}",
                    b.shape, boxes_channels, num_anchors
                )));
            };

            match num_classes {
                Some(n) => {
                    let this_n = (boxes_channels / num_anchors) - 5;
                    if n != this_n {
                        return Err(DecoderError::InvalidConfig(format!(
                            "ModelPack Split Detection inconsistent number of classes: previous {}, current {}",
                            n, this_n
                        )));
                    }
                }
                None => {
                    num_classes = Some((boxes_channels / num_anchors) - 5);
                }
            }
        }

        Ok(num_classes.unwrap_or(0))
    }

    fn verify_modelpack_seg(
        segmentation: &configs::Segmentation,
        classes: Option<usize>,
    ) -> Result<(), DecoderError> {
        if segmentation.shape.len() != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Segmentation shape {:?}",
                segmentation.shape
            )));
        }

        if let Some(classes) = classes {
            let seg_channels = if segmentation.channels_first {
                segmentation.shape[1]
            } else {
                segmentation.shape[3]
            };

            if seg_channels != classes + 1 {
                return Err(DecoderError::InvalidConfig(format!(
                    "ModelPack Segmentation channels {} incompatible with number of classes {}",
                    seg_channels, classes
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decoder {
    model_type: ModelType,
    pub iou_threshold: f32,
    pub score_threshold: f32,
}

#[derive(Debug)]
pub enum ArrayViewDQuantized<'a> {
    UInt8(ArrayViewD<'a, u8>),
    Int8(ArrayViewD<'a, i8>),
    UInt16(ArrayViewD<'a, u16>),
    Int16(ArrayViewD<'a, i16>),
    UInt32(ArrayViewD<'a, u32>),
    Int32(ArrayViewD<'a, i32>),
}

impl<'a, D> From<ArrayView<'a, u8, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, u8, D>) -> Self {
        Self::UInt8(arr.into_dyn())
    }
}

impl<'a, D> From<ArrayView<'a, i8, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, i8, D>) -> Self {
        Self::Int8(arr.into_dyn())
    }
}

impl<'a, D> From<ArrayView<'a, u16, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, u16, D>) -> Self {
        Self::UInt16(arr.into_dyn())
    }
}

impl<'a, D> From<ArrayView<'a, i16, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, i16, D>) -> Self {
        Self::Int16(arr.into_dyn())
    }
}

impl<'a, D> From<ArrayView<'a, u32, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, u32, D>) -> Self {
        Self::UInt32(arr.into_dyn())
    }
}

impl<'a, D> From<ArrayView<'a, i32, D>> for ArrayViewDQuantized<'a>
where
    D: Dimension,
{
    fn from(arr: ArrayView<'a, i32, D>) -> Self {
        Self::Int32(arr.into_dyn())
    }
}

impl<'a> ArrayViewDQuantized<'a> {
    /// Returns the shape of the underlying array.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::ArrayViewDQuantized;
    /// # use ndarray::Array2;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let arr = Array2::from_shape_vec((2, 3), vec![1u8, 2, 3, 4, 5, 6])?;
    /// let view = ArrayViewDQuantized::from(arr.view().into_dyn());
    /// assert_eq!(view.shape(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayViewDQuantized::UInt8(a) => a.shape(),
            ArrayViewDQuantized::Int8(a) => a.shape(),
            ArrayViewDQuantized::UInt16(a) => a.shape(),
            ArrayViewDQuantized::Int16(a) => a.shape(),
            ArrayViewDQuantized::UInt32(a) => a.shape(),
            ArrayViewDQuantized::Int32(a) => a.shape(),
        }
    }
}

macro_rules! with_quantized {
    ($x:expr, $var:ident, $body:expr) => {
        match $x {
            ArrayViewDQuantized::UInt8(x) => {
                let $var = x;
                $body
            }
            ArrayViewDQuantized::Int8(x) => {
                let $var = x;
                $body
            }
            ArrayViewDQuantized::UInt16(x) => {
                let $var = x;
                $body
            }
            ArrayViewDQuantized::Int16(x) => {
                let $var = x;
                $body
            }
            ArrayViewDQuantized::UInt32(x) => {
                let $var = x;
                $body
            }
            ArrayViewDQuantized::Int32(x) => {
                let $var = x;
                $body
            }
        }
    };
}

impl Decoder {
    /// This function returns the parsed model type of the decoder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, configs::ModelType};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = include_str!("../../../testdata/modelpack_split.yaml").to_string();
    ///     let decoder = DecoderBuilder::default()
    ///         .with_config_yaml_str(config_yaml)
    ///         .build()?;
    ///     assert!(matches!(
    ///         decoder.model_type(),
    ///         ModelType::ModelPackDetSplit { .. }
    ///     ));
    /// #    Ok(())
    /// # }
    /// ```
    pub fn model_type(&self) -> &ModelType {
        &self.model_type
    }

    /// This function decodes quantized model outputs into detection boxes and
    /// segmentation masks. The quantized outputs can be of u8, i8, u16, i16,
    /// u32, or i32 types. Up to `output_boxes.capacity()` boxes and masks
    /// will be decoded. The function clears the provided output vectors
    /// before populating them with the decoded results.
    ///
    /// This function returns a `DecoderError` if the the provided outputs don't
    /// match the configuration provided by the user when building the decoder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult};
    /// # use ndarray::Array4;
    /// # fn main() -> DecoderResult<()> {
    /// #    let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
    /// #    let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec())?;
    /// #
    /// #    let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
    /// #    let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec())?;
    /// #    let model_output = vec![
    /// #        detect1.view().into_dyn().into(),
    /// #        detect0.view().into_dyn().into(),
    /// #    ];
    /// let decoder = DecoderBuilder::default()
    ///     .with_config_yaml_str(include_str!("../../../testdata/modelpack_split.yaml").to_string())
    ///     .with_score_threshold(0.45)
    ///     .with_iou_threshold(0.45)
    ///     .build()?;
    ///
    /// let mut output_boxes: Vec<_> = Vec::with_capacity(10);
    /// let mut output_masks: Vec<_> = Vec::with_capacity(10);
    /// decoder.decode_quantized(&model_output, &mut output_boxes, &mut output_masks)?;
    /// assert!(output_boxes[0].equal_within_delta(
    ///     &DetectBox {
    ///         bbox: BoundingBox {
    ///             xmin: 0.43171933,
    ///             ymin: 0.68243736,
    ///             xmax: 0.5626645,
    ///             ymax: 0.808863,
    ///         },
    ///         score: 0.99240804,
    ///         label: 0
    ///     },
    ///     1e-6
    /// ));
    /// #    Ok(())
    /// # }
    /// ```
    pub fn decode_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_quantized(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split_quantized(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_quantized(outputs, boxes, scores, output_boxes)
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_quantized(outputs, detection, output_boxes)
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_quantized(outputs, boxes, output_boxes)
            }
            ModelType::YoloSegDet { boxes, protos } => self.decode_yolo_segdet_quantized(
                outputs,
                boxes,
                protos,
                output_boxes,
                output_masks,
            ),
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_quantized(outputs, boxes, scores, output_boxes)
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => self.decode_yolo_split_segdet_quantized(
                outputs,
                boxes,
                scores,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
            ),
        }
    }

    /// This function decodes floating point model outputs into detection boxes
    /// and segmentation masks. Up to `output_boxes.capacity()` boxes and
    /// masks will be decoded. The function clears the provided output
    /// vectors before populating them with the decoded results.
    ///
    /// This function returns an `Error` if the the provided outputs don't
    /// match the configuration provided by the user when building the decoder.
    ///
    /// Any quantization information in the configuration will be ignored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult, configs::Boxes, configs::DecoderType, configs::Detection, dequantize_cpu, Quantization};
    /// # use ndarray::Array3;
    /// # fn main() -> DecoderResult<()> {
    /// #   let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    /// #   let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    /// #   let mut out_dequant = vec![0.0_f64; 84 * 8400];
    /// #   let quant = Quantization::new(0.0040811873, -123);
    /// #   dequantize_cpu(out, quant, &mut out_dequant);
    /// #   let model_output_f64 = Array3::from_shape_vec((1, 84, 8400), out_dequant)?.into_dyn();
    ///    let decoder = DecoderBuilder::default()
    ///     .with_config_yolo_det(Detection {
    ///         decoder: DecoderType::Ultralytics,
    ///         quantization: None,
    ///         shape: vec![1, 84, 8400],
    ///         channels_first: false,
    ///         anchors: None,
    ///     })
    ///     .with_score_threshold(0.25)
    ///     .with_iou_threshold(0.7)
    ///     .build()?;
    ///
    /// let mut output_boxes: Vec<_> = Vec::with_capacity(10);
    /// let mut output_masks: Vec<_> = Vec::with_capacity(10);
    /// let model_output_f64 = vec![model_output_f64.view().into()];
    /// decoder.decode_float(&model_output_f64, &mut output_boxes, &mut output_masks)?;    
    /// assert!(output_boxes[0].equal_within_delta(
    ///        &DetectBox {
    ///            bbox: BoundingBox {
    ///                xmin: 0.5285137,
    ///                ymin: 0.05305544,
    ///                xmax: 0.87541467,
    ///                ymax: 0.9998909,
    ///            },
    ///            score: 0.5591227,
    ///            label: 0
    ///        },
    ///        1e-6
    ///    ));
    ///
    /// #    Ok(())
    /// # }
    pub fn decode_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_float(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_float(outputs, boxes, protos, output_boxes, output_masks)?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_float(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_segdet_float(
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                )?;
            }
        }
        Ok(())
    }

    fn decode_modelpack_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (scores_tensor, _) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &[ind])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let mut boxes_tensor = b.slice(s![0, .., 0, ..]);
                Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

                let mut scores_tensor = s.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);
                decode_modelpack_det(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            });
        });

        Ok(())
    }

    fn decode_modelpack_seg_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (seg, _) = Self::find_outputs_with_shape_quantized(&segmentation.shape, outputs, &[])?;

        macro_rules! modelpack_seg {
            ($seg:expr) => {{
                let mut seg = $seg.slice(s![0, .., .., ..]);
                Self::swap_axes_if_needed(&mut seg, segmentation.channels_first);
                seg
            }};
        }
        use ArrayViewDQuantized::*;
        let seg = match seg {
            UInt8(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| x)
            }
            Int8(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| (x as i16 + 128) as u8)
            }
            UInt16(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| (x >> 8) as u8)
            }
            Int16(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| ((x as i32 + 32768) >> 8) as u8)
            }
            UInt32(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| (x >> 24) as u8)
            }
            Int32(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| ((x as i64 + 2147483648) >> 24) as u8)
            }
        };

        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_split_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let new_detection = detection
            .iter()
            .map(|x| match &x.anchors {
                None => Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection missing anchors".to_string(),
                )),
                Some(a) => Ok(ModelPackDetectionConfig {
                    anchors: a.clone(),
                    quantization: None,
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let new_outputs = Self::match_outputs_to_detect_quantized(detection, outputs)?;

        macro_rules! dequant_output {
            ($det_tensor:expr, $detection:expr) => {{
                let mut det_tensor = $det_tensor.slice(s![0, .., .., ..]);
                Self::swap_axes_if_needed(&mut det_tensor, $detection.channels_first);
                if let Some(q) = $detection.quantization {
                    dequantize_ndarray(det_tensor, q.into())
                } else {
                    det_tensor.map(|x| *x as f32)
                }
            }};
        }

        let new_outputs = new_outputs
            .iter()
            .zip(detection)
            .map(|(det_tensor, detection)| {
                with_quantized!(det_tensor, d, dequant_output!(d, detection))
            })
            .collect::<Vec<_>>();

        let new_outputs_view = new_outputs
            .iter()
            .map(|d: &Array3<f32>| d.view())
            .collect::<Vec<_>>();
        decode_modelpack_split_float(
            &new_outputs_view,
            &new_detection,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, _) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            let mut boxes_tensor = b.slice(s![0, .., ..]);
            if boxes.channels_first {
                boxes_tensor.swap_axes(0, 1);
            };
            Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);
            decode_yolo_det(
                (boxes_tensor, quant_boxes),
                self.score_threshold,
                self.iou_threshold,
                output_boxes,
            );
        });

        Ok(())
    }

    fn decode_yolo_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &[ind])?;

        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(protos_tensor, p, {
                let mut box_tensor = b.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut box_tensor, boxes.channels_first);

                let mut protos_tensor = p.slice(s![0, .., .., ..]);
                Self::swap_axes_if_needed(&mut protos_tensor, protos.channels_first);

                decode_yolo_segdet_quant(
                    (box_tensor, quant_boxes),
                    (protos_tensor, quant_protos),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                    output_masks,
                );
            });
        });

        Ok(())
    }

    fn decode_yolo_split_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (scores_tensor, _) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &[ind])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let mut boxes_tensor = b.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

                let mut scores_tensor = s.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);
                decode_yolo_split_det_quant(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            });
        });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &skip)?;
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        let boxes = with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let mut boxes_tensor = b.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

                let mut scores_tensor = s.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);
                impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes.capacity(),
                )
            })
        });

        with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mut mask_tensor = m.slice(s![0, .., ..]);
                Self::swap_axes_if_needed(&mut mask_tensor, mask_coeff.channels_first);

                let mut protos_tensor = p.slice(s![0, .., .., ..]);
                Self::swap_axes_if_needed(&mut protos_tensor, protos.channels_first);

                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    boxes,
                    (mask_tensor, quant_masks),
                    (protos_tensor, quant_protos),
                    output_boxes,
                    output_masks,
                )
            })
        });

        Ok(())
    }

    fn decode_modelpack_det_split_float<D>(
        &self,
        outputs: &[ArrayViewD<D>],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        D: AsPrimitive<f32>,
    {
        let new_detection = detection
            .iter()
            .map(|x| match &x.anchors {
                None => Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection missing anchors".to_string(),
                )),
                Some(a) => Ok(ModelPackDetectionConfig {
                    anchors: a.clone(),
                    quantization: None,
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let new_outputs = Self::match_outputs_to_detect(detection, outputs)?;
        let new_outputs = new_outputs
            .into_iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>();

        decode_modelpack_split_float(
            &new_outputs,
            &new_detection,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_modelpack_seg_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;
        let mut seg = seg.slice(s![0, .., .., ..]);
        Self::swap_axes_if_needed(&mut seg, segmentation.channels_first);

        let u8_max = 255.0_f32.as_();
        let max = *seg.max().unwrap_or(&u8_max);
        let min = *seg.min().unwrap_or(&0.0_f32.as_());
        let seg = seg.mapv(|x| ((x - min) / (max - min) * u8_max).as_());
        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);

        decode_modelpack_float(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

        decode_yolo_det_float(
            boxes_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        Self::swap_axes_if_needed(&mut protos_tensor, protos.channels_first);

        decode_yolo_segdet_float(
            boxes_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_yolo_split_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);

        decode_yolo_split_det_float(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut boxes_tensor, boxes.channels_first);
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut scores_tensor, scores.channels_first);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mut mask_tensor = mask_tensor.slice(s![0, .., ..]);
        Self::swap_axes_if_needed(&mut mask_tensor, mask_coeff.channels_first);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        Self::swap_axes_if_needed(&mut protos_tensor, protos.channels_first);

        decode_yolo_split_segdet_float(
            boxes_tensor,
            scores_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn match_outputs_to_detect<'a, 'b, T>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewD<'b, T>],
    ) -> Result<Vec<&'a ArrayViewD<'b, T>>, DecoderError> {
        let mut new_output_order = Vec::new();
        for c in configs {
            let mut found = false;
            for o in outputs {
                if o.shape() == c.shape {
                    new_output_order.push(o);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(DecoderError::InvalidShape(format!(
                    "Did not find output with shape {:?}",
                    c.shape
                )));
            }
        }
        Ok(new_output_order)
    }

    fn find_outputs_with_shape<'a, 'b, T>(
        shape: &[usize],
        outputs: &'a [ArrayViewD<'b, T>],
        skip: &[usize],
    ) -> Result<(&'a ArrayViewD<'b, T>, usize), DecoderError> {
        for (ind, o) in outputs.iter().enumerate() {
            if skip.contains(&ind) {
                continue;
            }
            if o.shape() == shape {
                return Ok((o, ind));
            }
        }
        Err(DecoderError::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }

    fn find_outputs_with_shape_quantized<'a, 'b>(
        shape: &[usize],
        outputs: &'a [ArrayViewDQuantized<'b>],
        skip: &[usize],
    ) -> Result<(&'a ArrayViewDQuantized<'b>, usize), DecoderError> {
        for (ind, o) in outputs.iter().enumerate() {
            if skip.contains(&ind) {
                continue;
            }
            if o.shape() == shape {
                return Ok((o, ind));
            }
        }
        Err(DecoderError::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }

    fn swap_axes_if_needed<T, D: Dimension>(array: &mut ArrayView<T, D>, channels_first: bool) {
        if channels_first {
            match array.ndim() {
                2 => array.swap_axes(0, 1),
                3 => {
                    array.swap_axes(0, 1);
                    array.swap_axes(1, 2);
                }
                _ => {}
            }
        }
    }

    fn match_outputs_to_detect_quantized<'a, 'b>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewDQuantized<'b>],
    ) -> Result<Vec<&'a ArrayViewDQuantized<'b>>, DecoderError> {
        let mut new_output_order = Vec::new();
        for c in configs {
            let mut found = false;
            for o in outputs {
                if o.shape() == c.shape {
                    new_output_order.push(o);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(DecoderError::InvalidShape(format!(
                    "Did not find output with shape {:?}",
                    c.shape
                )));
            }
        }
        Ok(new_output_order)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod decoder_builder_tests {
    use super::*;

    #[test]
    fn test_decoder_builder_no_config() {
        use crate::DecoderBuilder;
        let result = DecoderBuilder::default().build();
        assert!(matches!(result, Err(DecoderError::NoConfig)));
    }

    #[test]
    fn test_decoder_builder_empty_config() {
        use crate::DecoderBuilder;
        let result = DecoderBuilder::default()
            .with_config(ConfigOutputs { outputs: vec![] })
            .build();
        assert!(
            matches!(result, Err(DecoderError::InvalidConfig(s)) if s == "No outputs found in config")
        );
    }

    #[test]
    fn test_malformed_config_yaml() {
        let malformed_yaml = "
        model_type: yolov8_det
        outputs:
          - shape: [1, 84, 8400]
            channels_first: false
        "
        .to_owned();
        let result = DecoderBuilder::new()
            .with_config_yaml_str(malformed_yaml)
            .build();
        assert!(matches!(result, Err(DecoderError::Yaml(_))));
    }

    #[test]
    fn test_malformed_config_json() {
        let malformed_yaml = "
        {
            \"model_type\": \"yolov8_det\",
            \"outputs\": [
                {
                    \"shape\": [1, 84, 8400]
                    \"channels_first\": false
                }
            ]
        }"
        .to_owned();
        let result = DecoderBuilder::new()
            .with_config_json_str(malformed_yaml)
            .build();
        assert!(matches!(result, Err(DecoderError::Json(_))));
    }

    #[test]
    fn test_modelpack_and_yolo_config_error() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400],
                    channels_first: false,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::ModelPack,
                    shape: vec![1, 80, 8400],
                    channels_first: false,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Both ModelPack and Yolo outputs found in config"
        ));
    }

    #[test]
    fn test_yolo_invalid_seg_shape() {
        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400, 1], // Invalid shape
                    channels_first: false,
                    quantization: None,
                    anchors: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")
        ));
    }

    #[test]
    fn test_yolo_invalid_mask() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Mask(configs::Mask {
                    channels_first: false,
                    shape: vec![1, 160, 160, 1],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                })],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Mask output with Yolo decoder")
        ));
    }

    #[test]
    fn test_yolo_invalid_outputs() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Segmentation(configs::Segmentation {
                    channels_first: false,
                    shape: vec![1, 84, 8400],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                })],
            })
            .build();

        assert!(
            matches!(result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid Segmentation output with Yolo decoder")
        );
    }

    #[test]
    fn test_yolo_invalid_det() {
        let result = DecoderBuilder::new()
            .with_config_yolo_det(configs::Detection {
                anchors: None,
                decoder: DecoderType::Ultralytics,
                quantization: None,
                shape: vec![1, 84, 8400, 1], // Invalid shape
                channels_first: false,
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")));
    }

    #[test]
    fn test_yolo_invalid_segdet() {
        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400, 1], // Invalid shape
                    channels_first: false,
                    quantization: None,
                    anchors: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400],
                    channels_first: false,
                    quantization: None,
                    anchors: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1], // Invalid shape
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Protos shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 36], // too few classes
                    channels_first: true,
                    quantization: None,
                    anchors: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Yolo Protos channels 32 incompatible with Detection channels 36"));
    }

    #[test]
    fn test_yolo_invalid_split_det() {
        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400, 1], // Invalid shape
                    channels_first: false,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    channels_first: false,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400],
                    channels_first: false,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400, 1], // Invalid shape
                    channels_first: false,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400 + 1, 80], // Invalid number of boxes
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Scores num 8401")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 5, 8400], // Invalid boxes dimensions
                    channels_first: false,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    channels_first: false,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Boxes dimension 5, expected 4")));
    }

    #[test]
    fn test_yolo_invalid_split_segdet() {
        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4, 1],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80, 1],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32, 1],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Mask Coefficients shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Protos shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Scores num 8401")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(ref s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Mask Coefficients num 8401")));
        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    channels_first: true,
                    quantization: None,
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    channels_first: true,
                    quantization: None,
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    channels_first: true,
                    quantization: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 31, 160, 160],
                    channels_first: true,
                    quantization: None,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(ref s)) if s.starts_with( "Yolo Protos channels 31 incompatible with Mask Coefficients channels 32")));
    }

    #[test]
    fn test_modelpack_invalid_config() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Boxes(configs::Boxes {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 1, 4],
                        channels_first: false,
                        quantization: None,
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        channels_first: false,
                        quantization: None,
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        channels_first: false,
                        quantization: None,
                    }),
                ],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack should not have protos"));

        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Boxes(configs::Boxes {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 1, 4],
                        channels_first: false,
                        quantization: None,
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        channels_first: false,
                        quantization: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        channels_first: false,
                        quantization: None,
                    }),
                ],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack should not have mask coefficients"));

        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Boxes(configs::Boxes {
                    decoder: configs::DecoderType::ModelPack,
                    shape: vec![1, 8400, 1, 4],
                    channels_first: false,
                    quantization: None,
                })],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid ModelPack model outputs"));
    }

    #[test]
    fn test_modelpack_invalid_det() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    channels_first: true,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400, 1],
                    channels_first: true,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 2, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    channels_first: true,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid ModelPack Boxes dimension 2: 2, expected 1"));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 5, 1, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    channels_first: true,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid ModelPack Boxes dimension 5, expected 4"));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8401],
                    channels_first: true,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Detection Boxes num 8400 incompatible with Scores num 8401"));
    }

    #[test]
    fn test_modelpack_invalid_det_split() {
        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: None,
                    quantization: None,
                    channels_first: false,
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 9, 15, 18],
                    anchors: None,
                    quantization: None,
                    channels_first: false,
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection missing anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18],
                anchors: Some(vec![]),
                quantization: None,
                channels_first: false,
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection has zero anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18, 1],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                channels_first: false,
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Split Detection shape")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 15, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                channels_first: true,
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not greater than number of anchors * 5 =")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 16, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                channels_first: true,
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not a multiple of number of anchors")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    channels_first: false,
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 21],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    channels_first: false,
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("ModelPack Split Detection inconsistent number of classes:")));
    }

    #[test]
    fn test_modelpack_invalid_seg() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_seg(configs::Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: None,
                shape: vec![1, 160, 106, 3, 1],
                channels_first: false,
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Segmentation shape")));
    }

    #[test]
    fn test_modelpack_invalid_segdet() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_segdet(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    channels_first: true,
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 8400],
                    channels_first: true,
                },
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    channels_first: false,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("incompatible with number of classes")));
    }

    #[test]
    fn test_modelpack_invalid_segdet_split() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_segdet_split(
                vec![configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    channels_first: false,
                }],
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    channels_first: false,
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("incompatible with number of classes")));
    }
}
