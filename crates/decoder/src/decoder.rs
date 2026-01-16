// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use edgefirst_tracker::{AsTracker, TrackInfo};
use ndarray::{Array3, ArrayView, ArrayViewD, Dimension, s};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::{
    DecoderError, DetectBox, Quantization, Segmentation, XYWH,
    configs::{DecoderType, DimName, ModelType, QuantTuple},
    dequantize_ndarray,
    modelpack::{
        ModelPackDetectionConfig, decode_modelpack_det, decode_modelpack_float,
        decode_modelpack_split_float,
    },
    yolo::{
        decode_yolo_det, decode_yolo_det_float, decode_yolo_split_det_float,
        decode_yolo_split_det_quant, impl_yolo_segdet_get_boxes,
        impl_yolo_split_segdet_process_masks, impl_yolo_split_segdet_quant_get_boxes,
        impl_yolo_split_segdet_quant_process_masks, postprocess_yolo_seg,
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

type TrackerSettings<'a> = (
    &'a mut Box<dyn AsTracker<DetectBox>>,
    &'a mut Vec<TrackInfo<DetectBox>>,
    u64,
);

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

#[derive(Debug, PartialEq, Clone)]
pub enum ConfigOutputRef<'a> {
    Detection(&'a configs::Detection),
    Mask(&'a configs::Mask),
    Segmentation(&'a configs::Segmentation),
    Protos(&'a configs::Protos),
    Scores(&'a configs::Scores),
    Boxes(&'a configs::Boxes),
    MaskCoefficients(&'a configs::MaskCoefficients),
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
    ///     dshape: Vec::new(),
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
    ///   anchors: None,
    ///   decoder: configs::DecoderType::Ultralytics,
    ///   quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///   shape: vec![1, 84, 8400],
    ///   dshape: Vec::new(),
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
    use std::fmt::Display;

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
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Protos {
        pub decoder: DecoderType,
        #[serde(flatten)]
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct MaskCoefficients {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Mask {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Detection {
        pub anchors: Option<Vec<[f32; 2]>>,
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Scores {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Boxes {
        pub decoder: DecoderType,
        pub quantization: Option<QuantTuple>,
        pub shape: Vec<usize>,
        // #[serde(default)]
        // pub channels_first: bool,
        #[serde(default)]
        pub dshape: Vec<(DimName, usize)>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
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
    }

    impl Display for DimName {
        /// Formats the DimName for display
        /// # Examples
        /// ```rust
        /// # use edgefirst_decoder::configs::DimName;
        /// let dim = DimName::Height;
        /// assert_eq!(format!("{}", dim), "height");
        /// # let s = format!("{} {} {} {} {} {} {} {} {}", DimName::Batch, DimName::Height, DimName::Width, DimName::NumClasses, DimName::NumFeatures, DimName::NumBoxes, DimName::NumProtos, DimName::NumAnchorsXFeatures, DimName::Padding);
        /// # assert_eq!(s, "batch height width num_classes num_features num_boxes num_protos num_anchors_x_features padding");
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
            }
        }
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

pub struct DecoderBuilder {
    config_src: Option<ConfigSource>,
    iou_threshold: f32,
    score_threshold: f32,
    tracker: Option<Box<dyn AsTracker<DetectBox>>>,
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
            tracker: None,
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
    ///         dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 80, 8400],
    ///     dshape: Vec::new(),
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
    ///     anchors: None,
    ///     dshape: Vec::new(),
    /// };
    /// let protos_config = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 160, 160, 32],
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.012345, 14)),
    ///     shape: vec![1, 80, 8400],
    ///     dshape: Vec::new(),
    /// };
    /// let mask_config = configs::MaskCoefficients {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, 125)),
    ///     shape: vec![1, 32, 8400],
    ///     dshape: Vec::new(),
    /// };
    /// let protos_config = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 160, 160, 32],
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 8400, 3],
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
    /// };
    /// let scores_config = configs::Scores {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 8400, 2],
    ///     dshape: Vec::new(),
    /// };
    /// let seg_config = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 640, 640, 3],
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
    /// };
    /// let seg_config = configs::Segmentation {
    ///     decoder: configs::DecoderType::ModelPack,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 640, 640, 2],
    ///     dshape: Vec::new(),
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
    ///     dshape: Vec::new(),
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

    /// Sets a tracker for the decoder to use with the
    /// `decode_quantized_tracked()` and `decode_float_tracked()` functions
    ///
    /// The tracker will not be used for the standard `decode_quantized()` and
    /// `decode_float()` functions.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!("../../../testdata/modelpack_split.json").to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_tracker(edgefirst_tracker::ByteTrackBuilder::new().build())
    ///     .build()?;
    /// assert_eq!(decoder.iou_threshold, 0.5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tracker<T: AsTracker<DetectBox> + 'static>(mut self, tracker: T) -> Self {
        self.tracker = Some(Box::new(tracker));
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
            tracker: self.tracker,
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

    fn verify_yolo_det(detect: &configs::Detection) -> Result<(), DecoderError> {
        if detect.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Detection shape {:?}",
                detect.shape
            )));
        }

        Self::verify_dshapes(&detect.dshape, &detect.shape, "Detection")?;
        if !detect.dshape.is_empty() {
            Self::get_class_count(&detect.dshape, None, None)?;
        } else {
            Self::get_class_count_no_dshape(detect.into(), None)?;
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

        Self::verify_dshapes(&detection.dshape, &detection.shape, "Detection")?;
        Self::verify_dshapes(&protos.dshape, &protos.shape, "Protos")?;

        let protos_count = Self::get_protos_count(&protos.dshape).unwrap_or(protos.shape[3]);
        let classes = if !detection.dshape.is_empty() {
            Self::get_class_count(&detection.dshape, Some(protos_count), None)?
        } else {
            Self::get_class_count_no_dshape(detection.into(), Some(protos_count))?
        };

        if classes == 0 {
            return Err(DecoderError::InvalidConfig(
                "Yolo Segmentation Detection has zero classes".to_string(),
            ));
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

        Self::verify_dshapes(&boxes.dshape, &boxes.shape, "Boxes")?;
        Self::verify_dshapes(&scores.dshape, &scores.shape, "Scores")?;

        let boxes_dim = if boxes.dshape.is_empty() {
            boxes.shape[1]
        } else {
            boxes
                .dshape
                .iter()
                .filter_map(|(name, dim)| {
                    if *name == DimName::NumFeatures {
                        Some(*dim)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap_or(boxes.shape[1])
        };

        if boxes_dim != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Split Boxes dimension {}, expected 4",
                boxes_dim
            )));
        }
        let boxes_num = Self::get_box_count(&boxes.dshape).unwrap_or(boxes.shape[2]);
        let scores_num = Self::get_box_count(&scores.dshape).unwrap_or(scores.shape[2]);

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

        Self::verify_dshapes(&boxes.dshape, &boxes.shape, "Boxes")?;
        Self::verify_dshapes(&scores.dshape, &scores.shape, "Scores")?;
        Self::verify_dshapes(&mask_coeff.dshape, &mask_coeff.shape, "Mask Coefficients")?;
        Self::verify_dshapes(&protos.dshape, &protos.shape, "Protos")?;

        let boxes_num = Self::get_box_count(&boxes.dshape).unwrap_or(boxes.shape[2]);
        let scores_num = Self::get_box_count(&scores.dshape).unwrap_or(scores.shape[2]);
        let mask_num = Self::get_box_count(&mask_coeff.dshape).unwrap_or(mask_coeff.shape[2]);

        let mask_channels = if !mask_coeff.dshape.is_empty() {
            Self::get_protos_count(&mask_coeff.dshape).ok_or_else(|| {
                DecoderError::InvalidConfig("Could not find num_protos in config".to_string())
            })?
        } else {
            mask_coeff.shape[1]
        };
        let proto_channels = if !protos.dshape.is_empty() {
            Self::get_protos_count(&protos.dshape).ok_or_else(|| {
                DecoderError::InvalidConfig("Could not find num_protos in config".to_string())
            })?
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

        Self::verify_dshapes(&boxes.dshape, &boxes.shape, "Boxes")?;
        Self::verify_dshapes(&scores.dshape, &scores.shape, "Scores")?;

        let boxes_dim = Self::get_features_count(&boxes.dshape).unwrap_or(boxes.shape[3]);

        if boxes_dim != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid ModelPack Boxes dimension {}, expected 4",
                boxes_dim
            )));
        }

        let boxes_num = Self::get_box_count(&boxes.dshape).unwrap_or(boxes.shape[1]);
        let scores_num = Self::get_box_count(&scores.dshape).unwrap_or(scores.shape[1]);

        if boxes_num != scores_num {
            return Err(DecoderError::InvalidConfig(format!(
                "ModelPack Detection Boxes num {} incompatible with Scores num {}",
                boxes_num, scores_num
            )));
        }

        let num_classes = if !scores.dshape.is_empty() {
            Self::get_class_count(&scores.dshape, None, None)?
        } else {
            Self::get_class_count_no_dshape(scores.into(), None)?
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

            Self::verify_dshapes(&b.dshape, &b.shape, "Split Detection")?;
            let classes = if !b.dshape.is_empty() {
                Self::get_class_count(&b.dshape, None, Some(num_anchors))?
            } else {
                Self::get_class_count_no_dshape(b.into(), None)?
            };

            match num_classes {
                Some(n) => {
                    if n != classes {
                        return Err(DecoderError::InvalidConfig(format!(
                            "ModelPack Split Detection inconsistent number of classes: previous {}, current {}",
                            n, classes
                        )));
                    }
                }
                None => {
                    num_classes = Some(classes);
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
        Self::verify_dshapes(&segmentation.dshape, &segmentation.shape, "Segmentation")?;

        if let Some(classes) = classes {
            let seg_classes = if !segmentation.dshape.is_empty() {
                Self::get_class_count(&segmentation.dshape, None, None)?
            } else {
                Self::get_class_count_no_dshape(segmentation.into(), None)?
            };

            if seg_classes != classes + 1 {
                return Err(DecoderError::InvalidConfig(format!(
                    "ModelPack Segmentation channels {} incompatible with number of classes {}",
                    seg_classes, classes
                )));
            }
        }
        Ok(())
    }

    // verifies that dshapes match the given shape
    fn verify_dshapes(
        dshape: &[(DimName, usize)],
        shape: &[usize],
        name: &str,
    ) -> Result<(), DecoderError> {
        for s in shape {
            if *s == 0 {
                return Err(DecoderError::InvalidConfig(format!(
                    "{} shape has zero dimension",
                    name
                )));
            }
        }

        if dshape.is_empty() {
            return Ok(());
        }
        // check the dshape lengths match the shape lengths
        if dshape.len() != shape.len() {
            return Err(DecoderError::InvalidConfig(format!(
                "{} dshape length does not match shape length",
                name
            )));
        }

        // check the dshape values match the shape values
        for ((dim_name, dim_size), shape_size) in dshape.iter().zip(shape) {
            if dim_size != shape_size {
                return Err(DecoderError::InvalidConfig(format!(
                    "{} dshape dimension {} size {} does not match shape size {}",
                    name, dim_name, dim_size, shape_size
                )));
            }
            if *dim_name == DimName::Padding && *dim_size != 1 {
                return Err(DecoderError::InvalidConfig(
                    "Padding dimension size must be 1".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn get_box_count(dshape: &[(DimName, usize)]) -> Option<usize> {
        for (dim_name, dim_size) in dshape {
            if *dim_name == DimName::NumBoxes {
                return Some(*dim_size);
            }
        }
        None
    }

    fn get_class_count_no_dshape(
        config: ConfigOutputRef,
        protos: Option<usize>,
    ) -> Result<usize, DecoderError> {
        match config {
            ConfigOutputRef::Detection(detection) => match detection.decoder {
                DecoderType::Ultralytics => {
                    if detection.shape[1] <= 4 + protos.unwrap_or(0) {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid shape: Yolo num_features {} must be greater than {}",
                            detection.shape[1],
                            4 + protos.unwrap_or(0),
                        )));
                    };
                    Ok(detection.shape[1] - 4 - protos.unwrap_or(0))
                }
                DecoderType::ModelPack => {
                    let Some(num_anchors) = detection.anchors.as_ref().map(|a| a.len()) else {
                        return Err(DecoderError::Internal(
                            "ModelPack Detection missing anchors".to_string(),
                        ));
                    };
                    let anchors_x_features = detection.shape[3];
                    if anchors_x_features <= num_anchors * 5 {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not greater than number of anchors * 5 = {}",
                            anchors_x_features,
                            num_anchors * 5,
                        )));
                    };

                    if !anchors_x_features.is_multiple_of(num_anchors) {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not a multiple of number of anchors {}",
                            anchors_x_features, num_anchors
                        )));
                    };
                    Ok(anchors_x_features / num_anchors - 5)
                }
            },

            ConfigOutputRef::Scores(scores) => match scores.decoder {
                DecoderType::Ultralytics => Ok(scores.shape[1]),
                DecoderType::ModelPack => Ok(scores.shape[2]),
            },
            ConfigOutputRef::Segmentation(seg) => Ok(seg.shape[3]),
            _ => Err(DecoderError::Internal(
                "Attempted to get class count from unsupported config output".to_owned(),
            )),
        }
    }

    // get the class count from dshape or calculate from num_features
    fn get_class_count(
        dshape: &[(DimName, usize)],
        protos: Option<usize>,
        anchors: Option<usize>,
    ) -> Result<usize, DecoderError> {
        if dshape.is_empty() {
            return Ok(0);
        }
        // if it has num_classes in dshape, return it
        for (dim_name, dim_size) in dshape {
            if *dim_name == DimName::NumClasses {
                return Ok(*dim_size);
            }
        }

        // number of classes can be calculated from num_features - 4 for yolo.  If the
        // model has protos, we also subtract the number of protos.
        for (dim_name, dim_size) in dshape {
            if *dim_name == DimName::NumFeatures {
                let protos = protos.unwrap_or(0);
                if protos + 4 >= *dim_size {
                    return Err(DecoderError::InvalidConfig(format!(
                        "Invalid shape: Yolo num_features {} must be greater than {}",
                        *dim_size,
                        protos + 4,
                    )));
                };
                return Ok(*dim_size - 4 - protos);
            }
        }

        // number of classes can be calculated from number of anchors for modelpack
        // split detection
        if let Some(num_anchors) = anchors {
            for (dim_name, dim_size) in dshape {
                if *dim_name == DimName::NumAnchorsXFeatures {
                    let anchors_x_features = *dim_size;
                    if anchors_x_features <= num_anchors * 5 {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not greater than number of anchors * 5 = {}",
                            anchors_x_features,
                            num_anchors * 5,
                        )));
                    };

                    if !anchors_x_features.is_multiple_of(num_anchors) {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not a multiple of number of anchors {}",
                            anchors_x_features, num_anchors
                        )));
                    };
                    return Ok((anchors_x_features / num_anchors) - 5);
                }
            }
        }
        Err(DecoderError::InvalidConfig(
            "Cannot determine number of classes from dshape".to_owned(),
        ))
    }

    fn get_protos_count(dshape: &[(DimName, usize)]) -> Option<usize> {
        for (dim_name, dim_size) in dshape {
            if *dim_name == DimName::NumProtos {
                return Some(*dim_size);
            }
        }
        None
    }

    fn get_features_count(dshape: &[(DimName, usize)]) -> Option<usize> {
        for (dim_name, dim_size) in dshape {
            if *dim_name == DimName::NumFeatures {
                return Some(*dim_size);
            }
        }
        None
    }
}

pub struct Decoder {
    model_type: ModelType,
    pub iou_threshold: f32,
    pub score_threshold: f32,
    tracker: Option<Box<dyn AsTracker<DetectBox>>>,
}

impl Debug for Decoder {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::result::Result<(), std::fmt::Error> {
        formatter
            .debug_struct("Decoder")
            .field("model_type", &self.model_type)
            .field("iou_threshold", &self.iou_threshold)
            .field("score_threshold", &self.score_threshold)
            // .field(
            //     "tracker",
            //     &self.tracker.as_ref().map(|x| format!("{:?}", x.update())),
            // )
            .finish()
    }
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
        let tracking = None;
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                Self::decode_modelpack_det_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                Self::decode_modelpack_det_split_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackDet { boxes, scores } => Self::decode_modelpack_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                output_boxes,
                tracking,
            ),
            ModelType::ModelPackDetSplit { detection } => {
                Self::decode_modelpack_det_split_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )
            }
            ModelType::ModelPackSeg { segmentation } => {
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::YoloDet { boxes } => Self::decode_yolo_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                output_boxes,
                tracking,
            ),
            ModelType::YoloSegDet { boxes, protos } => Self::decode_yolo_segdet_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                protos,
                output_boxes,
                output_masks,
                tracking,
            ),
            ModelType::YoloSplitDet { boxes, scores } => Self::decode_yolo_split_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                output_boxes,
                tracking,
            ),
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => Self::decode_yolo_split_segdet_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
                tracking,
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
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult, configs, configs::DecoderType, dequantize_cpu, Quantization};
    /// # use ndarray::Array3;
    /// # fn main() -> DecoderResult<()> {
    /// #   let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    /// #   let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    /// #   let mut out_dequant = vec![0.0_f64; 84 * 8400];
    /// #   let quant = Quantization::new(0.0040811873, -123);
    /// #   dequantize_cpu(out, quant, &mut out_dequant);
    /// #   let model_output_f64 = Array3::from_shape_vec((1, 84, 8400), out_dequant)?.into_dyn();
    ///    let decoder = DecoderBuilder::default()
    ///     .with_config_yolo_det(configs::Detection {
    ///         decoder: DecoderType::Ultralytics,
    ///         quantization: None,
    ///         shape: vec![1, 84, 8400],
    ///         anchors: None,
    ///         dshape: Vec::new(),
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
        let output_tracks = None;
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                Self::decode_modelpack_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    output_tracks,
                )?;
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                Self::decode_modelpack_det_split_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    output_tracks,
                )?;
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                Self::decode_modelpack_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    output_tracks,
                )?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                Self::decode_modelpack_det_split_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    output_tracks,
                )?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                Self::decode_yolo_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    output_boxes,
                    output_tracks,
                )?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                Self::decode_yolo_segdet_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                Self::decode_yolo_split_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    output_tracks,
                )?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                Self::decode_yolo_split_segdet_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
        }
        Ok(())
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
    pub fn decode_quantized_tracked(
        &mut self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo<DetectBox>>,
        timestamp: u64,
    ) -> Result<(), DecoderError> {
        let tracking = if let Some(tracker) = self.tracker.as_mut() {
            Some((tracker, output_tracks, timestamp))
        } else {
            return Err(DecoderError::NoTracker);
        };
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                Self::decode_modelpack_det_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                Self::decode_modelpack_det_split_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::ModelPackDet { boxes, scores } => Self::decode_modelpack_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                output_boxes,
                tracking,
            ),
            ModelType::ModelPackDetSplit { detection } => {
                Self::decode_modelpack_det_split_quantized(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )
            }
            ModelType::ModelPackSeg { segmentation } => {
                Self::decode_modelpack_seg_quantized(outputs, segmentation, output_masks)
            }
            ModelType::YoloDet { boxes } => Self::decode_yolo_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                output_boxes,
                tracking,
            ),
            ModelType::YoloSegDet { boxes, protos } => Self::decode_yolo_segdet_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                protos,
                output_boxes,
                output_masks,
                tracking,
            ),
            ModelType::YoloSplitDet { boxes, scores } => Self::decode_yolo_split_det_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                output_boxes,
                tracking,
            ),
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => Self::decode_yolo_split_segdet_quantized(
                self.score_threshold,
                self.iou_threshold,
                outputs,
                boxes,
                scores,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
                tracking,
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
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult, configs, configs::DecoderType, dequantize_cpu, Quantization};
    /// # use ndarray::Array3;
    /// # fn main() -> DecoderResult<()> {
    /// #   let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    /// #   let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    /// #   let mut out_dequant = vec![0.0_f64; 84 * 8400];
    /// #   let quant = Quantization::new(0.0040811873, -123);
    /// #   dequantize_cpu(out, quant, &mut out_dequant);
    /// #   let model_output_f64 = Array3::from_shape_vec((1, 84, 8400), out_dequant)?.into_dyn();
    ///    let decoder = DecoderBuilder::default()
    ///     .with_config_yolo_det(configs::Detection {
    ///         decoder: DecoderType::Ultralytics,
    ///         quantization: None,
    ///         shape: vec![1, 84, 8400],
    ///         anchors: None,
    ///         dshape: Vec::new(),
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
    pub fn decode_float_tracked<T>(
        &mut self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo<DetectBox>>,
        timestamp: u64,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let tracking = if let Some(tracker) = self.tracker.as_mut() {
            Some((tracker, output_tracks, timestamp))
        } else {
            return Err(DecoderError::NoTracker);
        };
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                Self::decode_modelpack_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                Self::decode_modelpack_det_split_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )?;
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                Self::decode_modelpack_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    tracking,
                )?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                Self::decode_modelpack_det_split_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    detection,
                    output_boxes,
                    tracking,
                )?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                Self::decode_modelpack_seg_float(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                Self::decode_yolo_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    output_boxes,
                    tracking,
                )?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                Self::decode_yolo_segdet_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                    tracking,
                )?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                Self::decode_yolo_split_det_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    output_boxes,
                    tracking,
                )?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                Self::decode_yolo_split_segdet_float(
                    self.score_threshold,
                    self.iou_threshold,
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                    tracking,
                )?;
            }
        }
        Ok(())
    }

    fn decode_modelpack_det_quantized(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
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
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                decode_modelpack_det(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    score_threshold,
                    iou_threshold,
                    output_boxes,
                );
            });
        });

        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    fn decode_modelpack_seg_quantized(
        outputs: &[ArrayViewDQuantized],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (seg, _) = Self::find_outputs_with_shape_quantized(&segmentation.shape, outputs, &[])?;

        macro_rules! modelpack_seg {
            ($seg:expr, $body:expr) => {{
                let seg = Self::swap_axes_if_needed($seg, segmentation.into());
                let seg = seg.slice(s![0, .., .., ..]);
                seg.mapv($body)
            }};
        }
        use ArrayViewDQuantized::*;
        let seg = match seg {
            UInt8(s) => {
                modelpack_seg!(s, |x| x)
            }
            Int8(s) => {
                modelpack_seg!(s, |x| (x as i16 + 128) as u8)
            }
            UInt16(s) => {
                modelpack_seg!(s, |x| (x >> 8) as u8)
            }
            Int16(s) => {
                modelpack_seg!(s, |x| ((x as i32 + 32768) >> 8) as u8)
            }
            UInt32(s) => {
                modelpack_seg!(s, |x| (x >> 24) as u8)
            }
            Int32(s) => {
                modelpack_seg!(s, |x| ((x as i64 + 2147483648) >> 24) as u8)
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
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
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
                let det_tensor = Self::swap_axes_if_needed($det_tensor, $detection.into());
                let det_tensor = det_tensor.slice(s![0, .., .., ..]);
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
            score_threshold,
            iou_threshold,
            output_boxes,
        );
        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    fn decode_yolo_det_quantized(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, _) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
            let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
            decode_yolo_det(
                (boxes_tensor, quant_boxes),
                score_threshold,
                iou_threshold,
                output_boxes,
            );
        });

        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_segdet_quantized(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        tracking: Option<TrackerSettings>,
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

        // with_quantized!(boxes_tensor, b, {
        //     with_quantized!(protos_tensor, p, {
        //         let box_tensor = Self::swap_axes_if_needed(b, boxes.into());
        //         let box_tensor = box_tensor.slice(s![0, .., ..]);

        //         let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
        //         let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        //         decode_yolo_segdet_quant(
        //             (box_tensor, quant_boxes),
        //             (protos_tensor, quant_protos),
        //             score_threshold,
        //             iou_threshold,
        //             output_boxes,
        //             output_masks,
        //         );
        //     });
        // });

        with_quantized!(boxes_tensor, b, {
            let box_tensor = Self::swap_axes_if_needed(b, boxes.into());
            let box_tensor = box_tensor.slice(s![0, .., ..]);

            let (boxes_tensor, scores_tensor, masks_tensor) = postprocess_yolo_seg(&box_tensor);

            let boxes = impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                (boxes_tensor, quant_boxes),
                (scores_tensor, quant_boxes),
                score_threshold,
                iou_threshold,
                output_boxes.capacity(),
            );

            let (boxes, old_boxes) = if let Some((tracker, track_infos, timestamp)) = tracking {
                let tracker = tracker.as_tracker_mut();
                Self::update_tracker_yolo_segdet(tracker, boxes, track_infos, timestamp)
            } else {
                (boxes, Vec::new())
            };

            with_quantized!(protos_tensor, p, {
                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    boxes,
                    (masks_tensor, quant_boxes),
                    (protos_tensor, quant_protos),
                    output_boxes,
                    output_masks,
                )
            });
            output_boxes.extend(old_boxes);
        });

        Ok(())
    }

    fn decode_yolo_split_det_quantized(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
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
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                decode_yolo_split_det_quant(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    score_threshold,
                    iou_threshold,
                    output_boxes,
                );
            });
        });
        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_quantized(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        tracking: Option<TrackerSettings>,
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

        let (masks_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        let boxes = with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor.reversed_axes(), quant_boxes),
                    (scores_tensor.reversed_axes(), quant_scores),
                    score_threshold,
                    iou_threshold,
                    output_boxes.capacity(),
                )
            })
        });

        let (boxes, old_boxes) = if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker_yolo_segdet(tracker, boxes, track_infos, timestamp)
        } else {
            (boxes, Vec::new())
        };

        with_quantized!(masks_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let masks_tensor = Self::swap_axes_if_needed(m, mask_coeff.into());
                let masks_tensor = masks_tensor.slice(s![0, .., ..]);

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    boxes,
                    (masks_tensor.reversed_axes(), quant_masks),
                    (protos_tensor, quant_protos),
                    output_boxes,
                    output_masks,
                )
            })
        });

        output_boxes.extend(old_boxes);

        Ok(())
    }

    fn decode_modelpack_det_split_float<D>(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<D>],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
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
            score_threshold,
            iou_threshold,
            output_boxes,
        );
        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    fn decode_modelpack_seg_float<T>(
        outputs: &[ArrayViewD<T>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;

        let seg = Self::swap_axes_if_needed(seg, segmentation.into());
        let seg = seg.slice(s![0, .., .., ..]);
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
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);

        decode_modelpack_float(
            boxes_tensor,
            scores_tensor,
            score_threshold,
            iou_threshold,
            output_boxes,
        );
        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    fn decode_yolo_det_float<T>(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        decode_yolo_det_float(boxes_tensor, score_threshold, iou_threshold, output_boxes);
        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_segdet_float<T>(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;

        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);

        let (boxes_tensor, scores_tensor, masks_tensor) = postprocess_yolo_seg(&boxes_tensor);

        let boxes = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_tensor,
            scores_tensor,
            score_threshold,
            iou_threshold,
            output_boxes.capacity(),
        );

        let (boxes, old_boxes) = if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker_yolo_segdet(tracker, boxes, track_infos, timestamp)
        } else {
            (boxes, Vec::new())
        };

        impl_yolo_split_segdet_process_masks(
            boxes,
            masks_tensor,
            protos_tensor,
            output_boxes,
            output_masks,
        );
        output_boxes.extend(old_boxes);

        Ok(())
    }

    fn decode_yolo_split_det_float<T>(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;

        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);

        decode_yolo_split_det_float(
            boxes_tensor,
            scores_tensor,
            score_threshold,
            iou_threshold,
            output_boxes,
        );

        if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker(tracker, output_boxes, track_infos, timestamp);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_float<T>(
        score_threshold: f32,
        iou_threshold: f32,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        tracking: Option<TrackerSettings>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;

        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (masks_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let masks_tensor = Self::swap_axes_if_needed(masks_tensor, mask_coeff.into());
        let masks_tensor = masks_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);

        let boxes = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_tensor.reversed_axes(),
            scores_tensor.reversed_axes(),
            score_threshold,
            iou_threshold,
            output_boxes.capacity(),
        );

        let (boxes, old_boxes) = if let Some((tracker, track_infos, timestamp)) = tracking {
            let tracker = tracker.as_tracker_mut();
            Self::update_tracker_yolo_segdet(tracker, boxes, track_infos, timestamp)
        } else {
            (boxes, Vec::new())
        };

        impl_yolo_split_segdet_process_masks(
            boxes,
            masks_tensor.reversed_axes(),
            protos_tensor,
            output_boxes,
            output_masks,
        );
        output_boxes.extend(old_boxes);

        Ok(())
    }

    fn update_tracker_yolo_segdet(
        tracker: &mut dyn edgefirst_tracker::Tracker<DetectBox>,
        boxes: Vec<(DetectBox, usize)>,
        track_infos: &mut Vec<TrackInfo<DetectBox>>,
        timestamp: u64,
    ) -> (Vec<(DetectBox, usize)>, Vec<DetectBox>) {
        // custom tracking since we can only use live boxes for masks. We
        // can still output "old" boxes but they will not have masks.
        // let live_tracks = tracker.update(output_boxes, timestamp);
        let (new_boxes, boxes_indices): (Vec<_>, Vec<_>) = boxes.into_iter().unzip();
        let live_tracks = tracker.update(&new_boxes, timestamp);
        let old_tracks = tracker
            .get_active_tracks()
            .into_iter()
            .filter(|x| x.last_updated != timestamp)
            .collect::<Vec<_>>();

        let live_boxes = live_tracks
            .iter()
            .zip(new_boxes)
            .zip(boxes_indices)
            .filter_map(|((t, mut b), ind)| {
                if let Some(t) = t {
                    b.bbox = t.tracked_location.into();
                    Some((b, ind))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let old_boxes = old_tracks
            .iter()
            .map(|t| {
                let mut b = t.last_box;
                b.bbox = t.tracked_location.into();
                b
            })
            .collect::<Vec<_>>();

        track_infos.clear();
        track_infos.extend(live_tracks.into_iter().flatten());
        track_infos.extend(old_tracks);
        (live_boxes, old_boxes)
    }

    fn update_tracker(
        tracker: &mut dyn edgefirst_tracker::Tracker<DetectBox>,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo<DetectBox>>,
        timestamp: u64,
    ) {
        tracker.update(output_boxes, timestamp);
        let tracks = tracker.get_active_tracks();

        let (new_tracks, new_boxes): (Vec<_>, Vec<_>) = tracks
            .into_iter()
            .map(|t| {
                let mut box_ = t.last_box;
                box_.bbox = t.tracked_location.into();
                (t, box_)
            })
            .unzip();
        output_boxes.clear();
        output_boxes.extend(new_boxes);
        output_tracks.clear();
        output_tracks.extend(new_tracks);
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

    fn swap_axes_if_needed<'a, T, D: Dimension>(
        array: &ArrayView<'a, T, D>,
        config: ConfigOutputRef,
    ) -> ArrayView<'a, T, D> {
        let mut array = array.clone();
        let mut current_order = match config {
            ConfigOutputRef::Detection(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                // This is split detection, need to swap axes to batch, height,
                // width, num_anchors_x_features,
                match s.decoder {
                    DecoderType::ModelPack => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::Height => 1,
                            DimName::Width => 2,
                            DimName::NumAnchorsXFeatures => 3,
                            _ => 1000, // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                    DecoderType::Ultralytics => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::NumFeatures => 1,
                            DimName::NumBoxes => 2,
                            _ => 1000, // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                }
            }
            ConfigOutputRef::Boxes(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                match s.decoder {
                    DecoderType::ModelPack => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::NumBoxes => 1,
                            DimName::Padding => 2,
                            DimName::NumFeatures => 3, // TODO: what is the 4 going to  be called?
                            _ => 1000,                 // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                    DecoderType::Ultralytics => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::NumFeatures => 1, // TODO: what is the 4 going to be called?
                            DimName::NumBoxes => 2,
                            _ => 1000, // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                }
            }
            ConfigOutputRef::Scores(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                match s.decoder {
                    DecoderType::ModelPack => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::NumBoxes => 1,
                            DimName::NumClasses => 2,
                            _ => 1000, // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                    DecoderType::Ultralytics => s
                        .dshape
                        .iter()
                        .map(|x| match x.0 {
                            DimName::Batch => 0,
                            DimName::NumClasses => 1,
                            DimName::NumBoxes => 2,
                            _ => 1000, // this should be unreachable
                        })
                        .collect::<Vec<_>>(),
                }
            }
            ConfigOutputRef::Segmentation(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                s.dshape
                    .iter()
                    .map(|x| match x.0 {
                        DimName::Batch => 0,
                        DimName::Height => 1,
                        DimName::Width => 2,
                        DimName::NumClasses => 3,
                        _ => 1000, // this should be unreachable
                    })
                    .collect::<Vec<_>>()
            }
            ConfigOutputRef::Mask(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                s.dshape
                    .iter()
                    .map(|x| match x.0 {
                        DimName::Batch => 0,
                        DimName::Height => 1,
                        DimName::Width => 2,
                        DimName::NumClasses => 3,
                        _ => 1000, // this should be unreachable
                    })
                    .collect::<Vec<_>>()
            }
            ConfigOutputRef::Protos(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                s.dshape
                    .iter()
                    .map(|x| match x.0 {
                        DimName::Batch => 0,
                        DimName::Height => 1,
                        DimName::Width => 2,
                        DimName::NumProtos => 3,

                        _ => 1000, // this should be unreachable
                    })
                    .collect::<Vec<_>>()
            }
            ConfigOutputRef::MaskCoefficients(s) => {
                if s.dshape.is_empty() {
                    return array;
                }
                s.dshape
                    .iter()
                    .map(|x| match x.0 {
                        DimName::Batch => 0,
                        DimName::NumProtos => 1,
                        DimName::NumBoxes => 2,
                        _ => 1000, // this should be unreachable
                    })
                    .collect::<Vec<_>>()
            }
        };

        assert_eq!(array.shape().len(), current_order.len());
        // do simple bubble sort as swap_axes is inexpensive and the
        // number of dimensions is small
        for i in 0..current_order.len() {
            let mut swapped = false;
            for j in 0..current_order.len() - 1 - i {
                if current_order[j] > current_order[j + 1] {
                    array.swap_axes(j, j + 1);
                    current_order.swap(j, j + 1);
                    swapped = true;
                }
            }
            if !swapped {
                break;
            }
        }
        array
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::ModelPack,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    shape: vec![1, 160, 160, 1],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumFeatures, 1),
                    ],
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
                    shape: vec![1, 84, 8400],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 84),
                        (DimName::NumBoxes, 8400),
                    ],
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
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 84),
                    (DimName::NumBoxes, 8400),
                    (DimName::Batch, 1),
                ],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_det(configs::Detection {
                anchors: None,
                decoder: DecoderType::Ultralytics,
                quantization: None,
                shape: vec![1, 8400, 3], // Invalid shape
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumFeatures, 3),
                ],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid shape: Yolo num_features 3 must be greater than 4")));

        let result = DecoderBuilder::new()
            .with_config_yolo_det(configs::Detection {
                anchors: None,
                decoder: DecoderType::Ultralytics,
                quantization: None,
                shape: vec![1, 3, 8400], // Invalid shape
                dshape: Vec::new(),
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid shape: Yolo num_features 3 must be greater than 4")));
    }

    #[test]
    fn test_yolo_invalid_segdet() {
        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400, 1], // Invalid shape
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1], // Invalid shape
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::Batch, 1),
                    ],
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
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 36),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();
        println!("{:?}", result);
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid shape: Yolo num_features 36 must be greater than 36"));
    }

    #[test]
    fn test_yolo_invalid_split_det() {
        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400, 1], // Invalid shape
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400, 1], // Invalid shape
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400 + 1, 80], // Invalid number of boxes
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumClasses, 80),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 5),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],

                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                        (DimName::Batch, 1),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::Batch, 1),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 32],

                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 31, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 31),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();
        println!("{:?}", result);
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
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::Padding, 1),
                            (DimName::NumFeatures, 4),
                        ],
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumClasses, 3),
                        ],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumFeatures, 3),
                        ],
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
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::Padding, 1),
                            (DimName::NumFeatures, 4),
                        ],
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumClasses, 3),
                        ],
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumProtos, 3),
                        ],
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
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::Padding, 1),
                        (DimName::NumFeatures, 4),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                        (DimName::Padding, 1),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::Padding, 2),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Padding dimension size must be 1"));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 5, 1, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 5),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8401],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8401),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 9, 15, 18],
                    anchors: None,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 9),
                        (DimName::Width, 15),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection missing anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18],
                anchors: None,
                quantization: None,
                dshape: Vec::new(),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection missing anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18],
                anchors: Some(vec![]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                    (DimName::NumAnchorsXFeatures, 18),
                ],
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
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                    (DimName::NumAnchorsXFeatures, 18),
                    (DimName::Padding, 1),
                ],
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
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumAnchorsXFeatures, 15),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not greater than number of anchors * 5 =")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 15],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: Vec::new(),
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
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumAnchorsXFeatures, 16),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not a multiple of number of anchors")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 16],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: Vec::new(),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not a multiple of number of anchors")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 18, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 18),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
            }])
            .build();
        println!("{:?}", result);
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("Cannot determine number of classes from dshape")));

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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 21),
                    ],
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("ModelPack Split Detection inconsistent number of classes:")));

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
                    dshape: vec![],
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
                    dshape: vec![],
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
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 106),
                    (DimName::NumClasses, 3),
                    (DimName::Padding, 1),
                ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 106),
                        (DimName::NumClasses, 3),
                    ],
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
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                }],
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 106),
                        (DimName::NumClasses, 3),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("incompatible with number of classes")));
    }

    #[test]
    fn test_decode_bad_shapes() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let quant = (0.0040811873, -123);
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let out_float: Array3<f32> = dequantize_ndarray(out.view(), quant.into());

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(configs::Detection {
                decoder: DecoderType::Ultralytics,
                shape: vec![1, 85, 8400],
                anchors: None,
                quantization: Some(quant.into()),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 85),
                    (DimName::NumBoxes, 8400),
                ],
            })
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);

        let result =
            decoder.decode_quantized(&[out.view().into()], &mut output_boxes, &mut output_masks);

        assert!(matches!(
            result, Err(DecoderError::InvalidShape(s)) if s == "Did not find output with shape [1, 85, 8400]"));

        let result = decoder.decode_float(
            &[out_float.view().into_dyn()],
            &mut output_boxes,
            &mut output_masks,
        );

        assert!(matches!(
            result, Err(DecoderError::InvalidShape(s)) if s == "Did not find output with shape [1, 85, 8400]"));
    }

    #[test]
    fn test_config_outputs() {
        let outputs = [
            ConfigOutput::Detection(configs::Detection {
                decoder: configs::DecoderType::Ultralytics,
                anchors: None,
                shape: vec![1, 8400, 85],
                quantization: Some(QuantTuple(0.123, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumFeatures, 85),
                ],
            }),
            ConfigOutput::Mask(configs::Mask {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 160, 160, 1],
                quantization: Some(QuantTuple(0.223, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                    (DimName::NumFeatures, 1),
                ],
            }),
            ConfigOutput::Segmentation(configs::Segmentation {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 160, 160, 80],
                quantization: Some(QuantTuple(0.323, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                    (DimName::NumClasses, 80),
                ],
            }),
            ConfigOutput::Scores(configs::Scores {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 80],
                quantization: Some(QuantTuple(0.423, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumClasses, 80),
                ],
            }),
            ConfigOutput::Boxes(configs::Boxes {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 4],
                quantization: Some(QuantTuple(0.523, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumFeatures, 4),
                ],
            }),
            ConfigOutput::Protos(configs::Protos {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 32, 160, 160],
                quantization: Some(QuantTuple(0.623, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 32),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ],
            }),
            ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 32],
                quantization: Some(QuantTuple(0.723, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumProtos, 32),
                ],
            }),
        ];

        let shapes = outputs.clone().map(|x| x.shape().to_vec());
        assert_eq!(
            shapes,
            [
                vec![1, 8400, 85],
                vec![1, 160, 160, 1],
                vec![1, 160, 160, 80],
                vec![1, 8400, 80],
                vec![1, 8400, 4],
                vec![1, 32, 160, 160],
                vec![1, 8400, 32],
            ]
        );

        let quants: [Option<(f32, i32)>; 7] = outputs.map(|x| x.quantization().map(|q| q.into()));
        assert_eq!(
            quants,
            [
                Some((0.123, 0)),
                Some((0.223, 0)),
                Some((0.323, 0)),
                Some((0.423, 0)),
                Some((0.523, 0)),
                Some((0.623, 0)),
                Some((0.723, 0)),
            ]
        );
    }
}
