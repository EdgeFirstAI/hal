// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::config::ConfigOutputRef;
use super::configs::{self, DecoderType, DecoderVersion, DimName, ModelType};
use super::{ConfigOutput, ConfigOutputs, Decoder};
use crate::DecoderError;

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderBuilder {
    config_src: Option<ConfigSource>,
    iou_threshold: f32,
    score_threshold: f32,
    /// NMS mode: Some(mode) applies NMS, None bypasses NMS (for end-to-end
    /// models)
    nms: Option<configs::Nms>,
}

#[derive(Debug, Clone, PartialEq)]
enum ConfigSource {
    Yaml(String),
    Json(String),
    Config(ConfigOutputs),
}

impl Default for DecoderBuilder {
    /// Creates a default DecoderBuilder with no configuration and 0.5 score
    /// threshold and 0.5 IoU threshold.
    ///
    /// A valid configuration must be provided before building the Decoder.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #  let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string();
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
            nms: Some(configs::Nms::ClassAgnostic),
        }
    }
}

impl DecoderBuilder {
    /// Creates a default DecoderBuilder with no configuration and 0.5 score
    /// threshold and 0.5 IoU threshold.
    ///
    /// A valid configuration must be provided before building the Decoder.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #  let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string();
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
    /// let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string();
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
    /// let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json")).to_string();
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
    /// let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json"));
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
    ///     .with_config_yolo_det(
    ///         configs::Detection {
    ///             anchors: None,
    ///             decoder: configs::DecoderType::Ultralytics,
    ///             quantization: Some(configs::QuantTuple(0.012345, 26)),
    ///             shape: vec![1, 84, 8400],
    ///             dshape: Vec::new(),
    ///             normalized: Some(true),
    ///         },
    ///         None,
    ///     )
    ///     .build()?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_det(
        mut self,
        boxes: configs::Detection,
        version: Option<DecoderVersion>,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Detection(boxes)],
            decoder_version: version,
            ..Default::default()
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
    ///     normalized: Some(true),
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
            ..Default::default()
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
    ///     normalized: Some(true),
    /// };
    /// let protos_config = configs::Protos {
    ///     decoder: configs::DecoderType::Ultralytics,
    ///     quantization: Some(configs::QuantTuple(0.0064123, -31)),
    ///     shape: vec![1, 160, 160, 32],
    ///     dshape: Vec::new(),
    /// };
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_yolo_segdet(
    ///         seg_config,
    ///         protos_config,
    ///         Some(configs::DecoderVersion::Yolov8),
    ///     )
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config_yolo_segdet(
        mut self,
        boxes: configs::Detection,
        protos: configs::Protos,
        version: Option<DecoderVersion>,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Detection(boxes), ConfigOutput::Protos(protos)],
            decoder_version: version,
            ..Default::default()
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
    ///     normalized: Some(true),
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
            ..Default::default()
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
    ///     normalized: Some(true),
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
            ..Default::default()
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
    ///     normalized: Some(true),
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
    ///     normalized: Some(true),
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
        let config = ConfigOutputs {
            outputs,
            ..Default::default()
        };
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
    ///     normalized: Some(true),
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
            ..Default::default()
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
    ///     normalized: Some(true),
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
    ///     normalized: Some(true),
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
        let config = ConfigOutputs {
            outputs,
            ..Default::default()
        };
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
            ..Default::default()
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    /// Add an output to the decoder configuration.
    ///
    /// Incrementally builds the model configuration by adding outputs one at
    /// a time. The decoder resolves the model type from the combination of
    /// outputs during `build()`.
    ///
    /// If `dshape` is non-empty on the output, `shape` is automatically
    /// derived from it (the size component of each named dimension). This
    /// prevents conflicts between `shape` and `dshape`.
    ///
    /// This uses the programmatic config path. Calling this after
    /// `with_config_json_str()` or `with_config_yaml_str()` replaces the
    /// string-based config source.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, ConfigOutput, configs};
    /// # fn main() -> DecoderResult<()> {
    /// let decoder = DecoderBuilder::new()
    ///     .add_output(ConfigOutput::Scores(configs::Scores {
    ///         decoder: configs::DecoderType::Ultralytics,
    ///         dshape: vec![
    ///             (configs::DimName::Batch, 1),
    ///             (configs::DimName::NumClasses, 80),
    ///             (configs::DimName::NumBoxes, 8400),
    ///         ],
    ///         ..Default::default()
    ///     }))
    ///     .add_output(ConfigOutput::Boxes(configs::Boxes {
    ///         decoder: configs::DecoderType::Ultralytics,
    ///         dshape: vec![
    ///             (configs::DimName::Batch, 1),
    ///             (configs::DimName::BoxCoords, 4),
    ///             (configs::DimName::NumBoxes, 8400),
    ///         ],
    ///         ..Default::default()
    ///     }))
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_output(mut self, output: ConfigOutput) -> Self {
        if !matches!(self.config_src, Some(ConfigSource::Config(_))) {
            self.config_src = Some(ConfigSource::Config(ConfigOutputs::default()));
        }
        if let Some(ConfigSource::Config(ref mut config)) = self.config_src {
            config.outputs.push(Self::normalize_output(output));
        }
        self
    }

    /// Sets the decoder version for Ultralytics models.
    ///
    /// This is used with `add_output()` to specify the YOLO architecture
    /// version when it cannot be inferred from the output shapes alone.
    ///
    /// - `Yolov5`, `Yolov8`, `Yolo11`: Traditional models requiring external
    ///   NMS
    /// - `Yolo26`: End-to-end models with NMS embedded in the model graph
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, ConfigOutput, configs};
    /// # fn main() -> DecoderResult<()> {
    /// let decoder = DecoderBuilder::new()
    ///     .add_output(ConfigOutput::Detection(configs::Detection {
    ///         decoder: configs::DecoderType::Ultralytics,
    ///         dshape: vec![
    ///             (configs::DimName::Batch, 1),
    ///             (configs::DimName::NumBoxes, 100),
    ///             (configs::DimName::NumFeatures, 6),
    ///         ],
    ///         ..Default::default()
    ///     }))
    ///     .with_decoder_version(configs::DecoderVersion::Yolo26)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_decoder_version(mut self, version: configs::DecoderVersion) -> Self {
        if !matches!(self.config_src, Some(ConfigSource::Config(_))) {
            self.config_src = Some(ConfigSource::Config(ConfigOutputs::default()));
        }
        if let Some(ConfigSource::Config(ref mut config)) = self.config_src {
            config.decoder_version = Some(version);
        }
        self
    }

    /// Normalize an output: if dshape is non-empty, derive shape from it.
    fn normalize_output(mut output: ConfigOutput) -> ConfigOutput {
        fn normalize_shape(shape: &mut Vec<usize>, dshape: &[(configs::DimName, usize)]) {
            if !dshape.is_empty() {
                *shape = dshape.iter().map(|(_, size)| *size).collect();
            }
        }
        match &mut output {
            ConfigOutput::Detection(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Boxes(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Scores(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Protos(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Segmentation(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::MaskCoefficients(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Mask(c) => normalize_shape(&mut c.shape, &c.dshape),
            ConfigOutput::Classes(c) => normalize_shape(&mut c.shape, &c.dshape),
        }
        output
    }

    /// Sets the scores threshold of the decoder
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json")).to_string();
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

    /// Sets the IOU threshold of the decoder. Has no effect when NMS is set to
    /// `None`
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json")).to_string();
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

    /// Sets the NMS mode for the decoder.
    ///
    /// - `Some(Nms::ClassAgnostic)` — class-agnostic NMS (default): suppress
    ///   overlapping boxes regardless of class label
    /// - `Some(Nms::ClassAware)` — class-aware NMS: only suppress boxes that
    ///   share the same class label AND overlap above the IoU threshold
    /// - `None` — bypass NMS entirely (for end-to-end models with embedded NMS)
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, configs::Nms};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json")).to_string();
    /// let decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_nms(Some(Nms::ClassAware))
    ///     .build()?;
    /// assert_eq!(decoder.nms, Some(Nms::ClassAware));
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_nms(mut self, nms: Option<configs::Nms>) -> Self {
        self.nms = nms;
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
    /// # let config_json = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.json")).to_string();
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

        // Extract normalized flag from config outputs
        let normalized = Self::get_normalized(&config.outputs);

        // Use NMS from config if present, otherwise use builder's NMS setting
        let nms = config.nms.or(self.nms);
        let model_type = Self::get_model_type(config)?;

        Ok(Decoder {
            model_type,
            iou_threshold: self.iou_threshold,
            score_threshold: self.score_threshold,
            nms,
            normalized,
        })
    }

    /// Extracts the normalized flag from config outputs.
    /// - `Some(true)`: Boxes are in normalized [0,1] coordinates
    /// - `Some(false)`: Boxes are in pixel coordinates
    /// - `None`: Unknown (not specified in config), caller must infer
    fn get_normalized(outputs: &[ConfigOutput]) -> Option<bool> {
        for output in outputs {
            match output {
                ConfigOutput::Detection(det) => return det.normalized,
                ConfigOutput::Boxes(boxes) => return boxes.normalized,
                _ => {}
            }
        }
        None // not specified
    }

    fn get_model_type(configs: ConfigOutputs) -> Result<ModelType, DecoderError> {
        // yolo or modelpack
        let mut yolo = false;
        let mut modelpack = false;
        for c in &configs.outputs {
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

    fn get_model_type_yolo(configs: ConfigOutputs) -> Result<ModelType, DecoderError> {
        let mut boxes = None;
        let mut protos = None;
        let mut split_boxes = None;
        let mut split_scores = None;
        let mut split_mask_coeff = None;
        let mut split_classes = None;
        for c in configs.outputs {
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
                ConfigOutput::Classes(classes) => split_classes = Some(classes),
            }
        }

        // Use end-to-end model types when:
        // 1. decoder_version is explicitly set to Yolo26 (definitive), OR
        //    decoder_version is not set but the dshapes are (batch, num_boxes,
        //    num_features)
        let is_end_to_end_dshape = boxes.as_ref().is_some_and(|b| {
            let dims = b.dshape.iter().map(|(d, _)| *d).collect::<Vec<_>>();
            dims == vec![DimName::Batch, DimName::NumBoxes, DimName::NumFeatures]
        });

        let is_end_to_end = configs
            .decoder_version
            .map(|v| v.is_end_to_end())
            .unwrap_or(is_end_to_end_dshape);

        if is_end_to_end {
            if let Some(boxes) = boxes {
                if let Some(protos) = protos {
                    Self::verify_yolo_seg_det_26(&boxes, &protos)?;
                    return Ok(ModelType::YoloEndToEndSegDet { boxes, protos });
                } else {
                    Self::verify_yolo_det_26(&boxes)?;
                    return Ok(ModelType::YoloEndToEndDet { boxes });
                }
            } else if let (Some(split_boxes), Some(split_scores), Some(split_classes)) =
                (split_boxes, split_scores, split_classes)
            {
                if let (Some(split_mask_coeff), Some(protos)) = (split_mask_coeff, protos) {
                    Self::verify_yolo_split_end_to_end_segdet(
                        &split_boxes,
                        &split_scores,
                        &split_classes,
                        &split_mask_coeff,
                        &protos,
                    )?;
                    return Ok(ModelType::YoloSplitEndToEndSegDet {
                        boxes: split_boxes,
                        scores: split_scores,
                        classes: split_classes,
                        mask_coeff: split_mask_coeff,
                        protos,
                    });
                }
                Self::verify_yolo_split_end_to_end_det(
                    &split_boxes,
                    &split_scores,
                    &split_classes,
                )?;
                return Ok(ModelType::YoloSplitEndToEndDet {
                    boxes: split_boxes,
                    scores: split_scores,
                    classes: split_classes,
                });
            } else {
                return Err(DecoderError::InvalidConfig(
                    "Invalid Yolo end-to-end model outputs".to_string(),
                ));
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
        } else if let (Some(boxes), Some(scores)) = (split_boxes, split_scores) {
            if let (Some(mask_coeff), Some(protos)) = (split_mask_coeff, protos) {
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

        Self::verify_dshapes(
            &detect.dshape,
            &detect.shape,
            "Detection",
            &[DimName::Batch, DimName::NumFeatures, DimName::NumBoxes],
        )?;
        if !detect.dshape.is_empty() {
            Self::get_class_count(&detect.dshape, None, None)?;
        } else {
            Self::get_class_count_no_dshape(detect.into(), None)?;
        }

        Ok(())
    }

    fn verify_yolo_det_26(detect: &configs::Detection) -> Result<(), DecoderError> {
        if detect.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid Yolo Detection shape {:?}",
                detect.shape
            )));
        }

        Self::verify_dshapes(
            &detect.dshape,
            &detect.shape,
            "Detection",
            &[DimName::Batch, DimName::NumFeatures, DimName::NumBoxes],
        )?;

        if !detect.shape.contains(&6) {
            return Err(DecoderError::InvalidConfig(
                "Yolo26 Detection must have 6 features".to_string(),
            ));
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

        Self::verify_dshapes(
            &detection.dshape,
            &detection.shape,
            "Detection",
            &[DimName::Batch, DimName::NumFeatures, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &protos.dshape,
            &protos.shape,
            "Protos",
            &[
                DimName::Batch,
                DimName::Height,
                DimName::Width,
                DimName::NumProtos,
            ],
        )?;

        let protos_count = Self::get_protos_count(&protos.dshape)
            .unwrap_or_else(|| protos.shape[1].min(protos.shape[3]));
        log::debug!("Protos count: {}", protos_count);
        log::debug!("Detection dshape: {:?}", detection.dshape);
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

    fn verify_yolo_seg_det_26(
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

        Self::verify_dshapes(
            &detection.dshape,
            &detection.shape,
            "Detection",
            &[DimName::Batch, DimName::NumFeatures, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &protos.dshape,
            &protos.shape,
            "Protos",
            &[
                DimName::Batch,
                DimName::Height,
                DimName::Width,
                DimName::NumProtos,
            ],
        )?;

        let protos_count = Self::get_protos_count(&protos.dshape)
            .unwrap_or_else(|| protos.shape[1].min(protos.shape[3]));
        log::debug!("Protos count: {}", protos_count);
        log::debug!("Detection dshape: {:?}", detection.dshape);

        if !detection.shape.contains(&(6 + protos_count)) {
            return Err(DecoderError::InvalidConfig(format!(
                "Yolo26 Segmentation Detection must have num_features be 6 + num_protos = {}",
                6 + protos_count
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

        Self::verify_dshapes(
            &boxes.dshape,
            &boxes.shape,
            "Boxes",
            &[DimName::Batch, DimName::BoxCoords, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &scores.dshape,
            &scores.shape,
            "Scores",
            &[DimName::Batch, DimName::NumClasses, DimName::NumBoxes],
        )?;

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

        Self::verify_dshapes(
            &boxes.dshape,
            &boxes.shape,
            "Boxes",
            &[DimName::Batch, DimName::BoxCoords, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &scores.dshape,
            &scores.shape,
            "Scores",
            &[DimName::Batch, DimName::NumClasses, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &mask_coeff.dshape,
            &mask_coeff.shape,
            "Mask Coefficients",
            &[DimName::Batch, DimName::NumProtos, DimName::NumBoxes],
        )?;
        Self::verify_dshapes(
            &protos.dshape,
            &protos.shape,
            "Protos",
            &[
                DimName::Batch,
                DimName::Height,
                DimName::Width,
                DimName::NumProtos,
            ],
        )?;

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
            protos.shape[1].min(protos.shape[3])
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

    fn verify_yolo_split_end_to_end_det(
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        classes: &configs::Classes,
    ) -> Result<(), DecoderError> {
        if boxes.shape.len() != 3 || !boxes.shape.contains(&4) {
            return Err(DecoderError::InvalidConfig(format!(
                "Split end-to-end boxes must be [batch, N, 4], got {:?}",
                boxes.shape
            )));
        }
        if scores.shape.len() != 3 || !scores.shape.contains(&1) {
            return Err(DecoderError::InvalidConfig(format!(
                "Split end-to-end scores must be [batch, N, 1], got {:?}",
                scores.shape
            )));
        }
        if classes.shape.len() != 3 || !classes.shape.contains(&1) {
            return Err(DecoderError::InvalidConfig(format!(
                "Split end-to-end classes must be [batch, N, 1], got {:?}",
                classes.shape
            )));
        }
        Ok(())
    }

    fn verify_yolo_split_end_to_end_segdet(
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        classes: &configs::Classes,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
    ) -> Result<(), DecoderError> {
        Self::verify_yolo_split_end_to_end_det(boxes, scores, classes)?;
        if mask_coeff.shape.len() != 3 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid split end-to-end mask coefficients shape {:?}",
                mask_coeff.shape
            )));
        }
        if protos.shape.len() != 4 {
            return Err(DecoderError::InvalidConfig(format!(
                "Invalid protos shape {:?}",
                protos.shape
            )));
        }
        Ok(())
    }

    fn get_model_type_modelpack(configs: ConfigOutputs) -> Result<ModelType, DecoderError> {
        let mut split_decoders = Vec::new();
        let mut segment_ = None;
        let mut scores_ = None;
        let mut boxes_ = None;
        for c in configs.outputs {
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
                ConfigOutput::Classes(_) => {
                    return Err(DecoderError::InvalidConfig(
                        "ModelPack should not have classes output".to_string(),
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
            } else if let (Some(scores), Some(boxes)) = (scores_, boxes_) {
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
        } else if let (Some(scores), Some(boxes)) = (scores_, boxes_) {
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

        Self::verify_dshapes(
            &boxes.dshape,
            &boxes.shape,
            "Boxes",
            &[
                DimName::Batch,
                DimName::NumBoxes,
                DimName::Padding,
                DimName::BoxCoords,
            ],
        )?;
        Self::verify_dshapes(
            &scores.dshape,
            &scores.shape,
            "Scores",
            &[DimName::Batch, DimName::NumBoxes, DimName::NumClasses],
        )?;

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
            }

            Self::verify_dshapes(
                &b.dshape,
                &b.shape,
                "Split Detection",
                &[
                    DimName::Batch,
                    DimName::Height,
                    DimName::Width,
                    DimName::NumAnchorsXFeatures,
                ],
            )?;
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
        Self::verify_dshapes(
            &segmentation.dshape,
            &segmentation.shape,
            "Segmentation",
            &[
                DimName::Batch,
                DimName::Height,
                DimName::Width,
                DimName::NumClasses,
            ],
        )?;

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
        dims: &[DimName],
    ) -> Result<(), DecoderError> {
        for s in shape {
            if *s == 0 {
                return Err(DecoderError::InvalidConfig(format!(
                    "{} shape has zero dimension",
                    name
                )));
            }
        }

        if shape.len() != dims.len() {
            return Err(DecoderError::InvalidConfig(format!(
                "{} shape length {} does not match expected dims length {}",
                name,
                shape.len(),
                dims.len()
            )));
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

            if *dim_name == DimName::BoxCoords && *dim_size != 4 {
                return Err(DecoderError::InvalidConfig(
                    "BoxCoords dimension size must be 4".to_string(),
                ));
            }
        }

        let dims_present = HashSet::<DimName>::from_iter(dshape.iter().map(|(name, _)| *name));
        for dim in dims {
            if !dims_present.contains(dim) {
                return Err(DecoderError::InvalidConfig(format!(
                    "{} dshape missing required dimension {:?}",
                    name, dim
                )));
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
                    }
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
                    }

                    if !anchors_x_features.is_multiple_of(num_anchors) {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not a multiple of number of anchors {}",
                            anchors_x_features, num_anchors
                        )));
                    }
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
                }
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
                    }

                    if !anchors_x_features.is_multiple_of(num_anchors) {
                        return Err(DecoderError::InvalidConfig(format!(
                            "Invalid ModelPack Split Detection shape: anchors_x_features {} not a multiple of number of anchors {}",
                            anchors_x_features, num_anchors
                        )));
                    }
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
}
