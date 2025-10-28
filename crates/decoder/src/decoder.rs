use ndarray::{Array, ArrayViewD, s};
use ndarray_stats::QuantileExt;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    DetectBox, Error, Quantization, Segmentation,
    configs::{DecoderType, ModelType},
    modelpack::{
        ModelPackDetectionConfig, decode_modelpack_f32, decode_modelpack_f64, decode_modelpack_i8,
        decode_modelpack_split, decode_modelpack_split_float, decode_modelpack_u8,
    },
    yolo::{
        decode_yolo_f32, decode_yolo_f64, decode_yolo_i8, decode_yolo_segdet_f32,
        decode_yolo_segdet_f64, decode_yolo_segdet_i8, decode_yolo_segdet_u8,
        decode_yolo_split_det_f32, decode_yolo_split_det_f64, decode_yolo_split_det_i8,
        decode_yolo_split_det_u8, decode_yolo_split_segdet_f32, decode_yolo_split_segdet_f64,
        decode_yolo_split_segdet_i8, decode_yolo_split_segdet_u8, decode_yolo_u8,
    },
};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
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

    pub fn quantization(&self) -> &Option<(f64, i64)> {
        match self {
            ConfigOutput::Detection(detection) => &detection.quantization,
            ConfigOutput::Mask(mask) => &mask.quantization,
            ConfigOutput::Segmentation(segmentation) => &segmentation.quantization,
            ConfigOutput::Scores(scores) => &scores.quantization,
            ConfigOutput::Boxes(boxes) => &boxes.quantization,
            ConfigOutput::Protos(protos) => &protos.quantization,
            ConfigOutput::MaskCoefficients(mask_coefficients) => &mask_coefficients.quantization,
        }
    }
}

pub mod configs {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Segmentation {
        pub decode: bool,
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Protos {
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct MaskCoefficients {
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Mask {
        pub decode: bool,
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Detection {
        pub anchors: Option<Vec<[f32; 2]>>,
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Scores {
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub struct Boxes {
        pub decoder: DecoderType,
        pub quantization: Option<(f64, i64)>,
        pub shape: Vec<usize>,
        #[serde(default)]
        pub channels_first: bool,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
    pub enum DecoderType {
        #[serde(rename = "modelpack")]
        ModelPack,
        #[serde(rename = "yolov8")]
        Yolov8,
    }

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
            boxes: Segmentation,
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
}

enum ConfigSource {
    Yaml(String),
    Json(String),
    Config(ConfigOutputs),
}

impl Default for DecoderBuilder {
    fn default() -> Self {
        Self {
            config_src: None,
            iou_threshold: 0.5,
            score_threshold: 0.5,
        }
    }
}

impl DecoderBuilder {
    pub fn with_config_yaml_str(mut self, yaml_str: String) -> Self {
        self.config_src.replace(ConfigSource::Yaml(yaml_str));
        self
    }

    pub fn with_config_json_str(mut self, json_str: String) -> Self {
        self.config_src.replace(ConfigSource::Json(json_str));
        self
    }

    pub fn with_config(mut self, config: ConfigOutputs) -> Self {
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    pub fn with_config_yolo_det(mut self, boxes: configs::Boxes) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Boxes(boxes)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

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

    pub fn with_config_yolo_segdet(
        mut self,
        boxes: configs::Segmentation,
        protos: configs::Protos,
    ) -> Self {
        let config = ConfigOutputs {
            outputs: vec![
                ConfigOutput::Segmentation(boxes),
                ConfigOutput::Protos(protos),
            ],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

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

    pub fn with_config_modelpack_det_split(mut self, boxes: Vec<configs::Detection>) -> Self {
        let outputs = boxes.into_iter().map(ConfigOutput::Detection).collect();
        let config = ConfigOutputs { outputs };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

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

    pub fn with_config_modelpack_seg(mut self, segmentation: configs::Segmentation) -> Self {
        let config = ConfigOutputs {
            outputs: vec![ConfigOutput::Segmentation(segmentation)],
        };
        self.config_src.replace(ConfigSource::Config(config));
        self
    }

    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = score_threshold;
        self
    }

    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    pub fn build(self) -> Result<Decoder, Error> {
        let config = match self.config_src {
            Some(ConfigSource::Json(s)) => serde_json::from_str(&s)?,
            Some(ConfigSource::Yaml(s)) => serde_yaml::from_str(&s)?,
            Some(ConfigSource::Config(c)) => c,
            None => return Err(Error::NoConfig),
        };
        let model_type = Self::get_model_type(config.outputs)?;
        Ok(Decoder {
            model_type,
            iou_threshold: self.iou_threshold,
            score_threshold: self.score_threshold,
        })
    }

    fn get_model_type(configs: Vec<ConfigOutput>) -> Result<ModelType, Error> {
        // yolo or modelpack
        let mut yolo = false;
        let mut modelpack = false;
        for c in &configs {
            match c.decoder() {
                DecoderType::ModelPack => modelpack = true,
                DecoderType::Yolov8 => yolo = true,
            }
        }
        match (modelpack, yolo) {
            (true, true) => Err(Error::InvalidConfig(
                "Both ModelPack and Yolo outputs found in config".to_string(),
            )),
            (true, false) => Self::get_model_type_modelpack(configs),
            (false, true) => Self::get_model_type_yolo(configs),
            (false, false) => Err(Error::InvalidConfig(
                "No outputs found in config".to_string(),
            )),
        }
    }

    fn get_model_type_yolo(configs: Vec<ConfigOutput>) -> Result<ModelType, Error> {
        let mut boxes = None;
        let mut seg_boxes = None;
        let mut protos = None;
        let mut split_boxes = None;
        let mut split_scores = None;
        let mut split_mask_coeff = None;
        for c in configs {
            match c {
                ConfigOutput::Detection(detection) => boxes = Some(detection),
                ConfigOutput::Segmentation(segmentation) => {
                    if segmentation.shape.len() == 3 {
                        seg_boxes = Some(segmentation)
                    } else {
                        return Err(Error::InvalidConfig(format!(
                            "Invalid Yolo Segmentation shape {:?}",
                            segmentation.shape
                        )));
                    }
                }
                ConfigOutput::Protos(protos_) => protos = Some(protos_),
                ConfigOutput::Mask(_) => {
                    return Err(Error::InvalidConfig(
                        "Invalid Mask output with Yolo decoder".to_string(),
                    ));
                }
                ConfigOutput::Scores(scores) => split_scores = Some(scores),
                ConfigOutput::Boxes(boxes) => split_boxes = Some(boxes),
                ConfigOutput::MaskCoefficients(mask_coeff) => split_mask_coeff = Some(mask_coeff),
            }
        }

        if let Some(boxes) = seg_boxes
            && let Some(protos) = protos
        {
            Ok(ModelType::YoloSegDet { boxes, protos })
        } else if let Some(boxes) = boxes {
            Ok(ModelType::YoloDet { boxes })
        } else if let Some(boxes) = split_boxes
            && let Some(scores) = split_scores
        {
            if let Some(mask_coeff) = split_mask_coeff
                && let Some(protos) = protos
            {
                Ok(ModelType::YoloSplitSegDet {
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                })
            } else {
                Ok(ModelType::YoloSplitDet { boxes, scores })
            }
        } else {
            Err(Error::InvalidConfig(
                "Invalid Yolo model outputs".to_string(),
            ))
        }
    }

    fn get_model_type_modelpack(configs: Vec<ConfigOutput>) -> Result<ModelType, Error> {
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
                    return Err(Error::InvalidConfig(
                        "ModelPack should not have protos".to_string(),
                    ));
                }
                ConfigOutput::Scores(scores) => scores_ = Some(scores),
                ConfigOutput::Boxes(boxes) => boxes_ = Some(boxes),
                ConfigOutput::MaskCoefficients(_) => {
                    return Err(Error::InvalidConfig(
                        "ModelPack should not have mask coefficients".to_string(),
                    ));
                }
            }
        }

        if let Some(segmentation) = segment_ {
            if !split_decoders.is_empty() {
                Ok(ModelType::ModelPackSegDetSplit {
                    detection: split_decoders,
                    segmentation,
                })
            } else if let Some(scores) = scores_
                && let Some(boxes) = boxes_
            {
                Ok(ModelType::ModelPackSegDet {
                    boxes,
                    scores,
                    segmentation,
                })
            } else {
                Ok(ModelType::ModelPackSeg { segmentation })
            }
        } else if !split_decoders.is_empty() {
            Ok(ModelType::ModelPackDetSplit {
                detection: split_decoders,
            })
        } else if let Some(scores) = scores_
            && let Some(boxes) = boxes_
        {
            Ok(ModelType::ModelPackDet { boxes, scores })
        } else {
            Err(Error::InvalidConfig(
                "Invalid ModelPack model outputs".to_string(),
            ))
        }
    }
}

pub struct Decoder {
    model_type: ModelType,
    pub iou_threshold: f32,
    pub score_threshold: f32,
}

impl Decoder {
    pub fn model_type(&self) -> &ModelType {
        &self.model_type
    }

    pub fn decode_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_u8(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_u8(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_u8(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_u8(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split(outputs, detection, output_boxes)?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_u8(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_u8(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_u8(outputs, boxes, protos, output_boxes, output_masks)?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_u8(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_segdet_u8(
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

    pub fn decode_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_i8(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_i8(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_i8(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_i8(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split(outputs, detection, output_boxes)?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_i8(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_i8(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_i8(outputs, boxes, protos, output_boxes, output_masks)?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_i8(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_segdet_i8(
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

    pub fn decode_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_f32(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_f32(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_f32(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_f32(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_f32(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_f32(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_f32(outputs, boxes, protos, output_boxes, output_masks)?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_f32(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_segdet_f32(
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

    pub fn decode_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                self.decode_modelpack_det_f64(outputs, boxes, scores, output_boxes)?;
                self.decode_modelpack_seg_f64(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
                self.decode_modelpack_seg_f64(outputs, segmentation, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_f64(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_f64(outputs, segmentation, output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_f64(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_f64(outputs, boxes, protos, output_boxes, output_masks)?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_f64(outputs, boxes, scores, output_boxes)?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_segdet_f64(
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

    fn decode_modelpack_det_split<D>(
        &self,
        outputs: &[ArrayViewD<D>],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error>
    where
        D: AsPrimitive<f32>,
        i64: AsPrimitive<D>,
    {
        let new_outputs = Self::match_outputs_to_detect(detection, outputs)?;
        let new_outputs = new_outputs
            .into_iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>();
        let new_detection = detection
            .iter()
            .map(|x| ModelPackDetectionConfig {
                anchors: x.anchors.clone().unwrap(),
                quantization: x.quantization.map(Quantization::from_tuple_truncate),
            })
            .collect::<Vec<_>>();
        decode_modelpack_split(
            &new_outputs,
            &new_detection,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_modelpack_det_split_float<D>(
        &self,
        outputs: &[ArrayViewD<D>],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error>
    where
        D: AsPrimitive<f32>,
        i64: AsPrimitive<D>,
    {
        let new_outputs = Self::match_outputs_to_detect(detection, outputs)?;
        let new_outputs = new_outputs
            .into_iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>();
        let new_detection = detection
            .iter()
            .map(|x| ModelPackDetectionConfig {
                anchors: x.anchors.clone().unwrap(),
                quantization: None,
            })
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

    fn decode_modelpack_seg_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;
        let mut seg = seg.slice(s![0, .., .., ..]);
        if segmentation.channels_first {
            seg.swap_axes(0, 1);
            seg.swap_axes(1, 2);
        };
        let seg = seg.mapv(|x| (x as i16 + 128) as u8);

        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_modelpack_i8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        decode_yolo_i8(
            (boxes_tensor, quant_boxes),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &configs::Segmentation,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (box_output, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut box_tensor = box_output.slice(s![0, .., ..]);
        if boxes.channels_first {
            box_tensor.swap_axes(0, 1);
        }

        let quant_protos = protos
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        }

        decode_yolo_segdet_i8(
            (box_tensor, quant_boxes),
            (protos_tensor, quant_protos),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_yolo_split_det_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        decode_yolo_split_det_i8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let mut skip = vec![];
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mut mask_tensor = mask_tensor.slice(s![0, .., ..]);
        if mask_coeff.channels_first {
            mask_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_protos = protos
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        }

        decode_yolo_split_segdet_i8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            (mask_tensor, quant_masks),
            (protos_tensor, quant_protos),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_modelpack_seg_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;
        let mut seg = seg.slice(s![0, .., .., ..]);
        if segmentation.channels_first {
            seg.swap_axes(0, 1);
            seg.swap_axes(1, 2);
        }

        // TODO: Adjust signatures so this doesn't need to clone the entire backing
        // array?
        let seg = if let Some(slc) = seg.as_slice() {
            Array::from_shape_vec(seg.raw_dim(), slc.to_vec())?
        } else {
            Array::from_shape_vec(seg.raw_dim(), seg.iter().cloned().collect())?
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

    fn decode_modelpack_det_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_modelpack_u8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        decode_yolo_u8(
            (boxes_tensor, quant_boxes),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &configs::Segmentation,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let quant_protos = protos
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        };
        decode_yolo_segdet_u8(
            (boxes_tensor, quant_boxes),
            (protos_tensor, quant_protos),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_yolo_split_det_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        decode_yolo_split_det_u8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let mut skip = vec![];
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_scores = scores
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mut mask_tensor = mask_tensor.slice(s![0, .., ..]);
        if mask_coeff.channels_first {
            mask_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let quant_protos = protos
            .quantization
            .map(Quantization::from_tuple_truncate)
            .unwrap_or_default();
        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        }

        decode_yolo_split_segdet_u8(
            (boxes_tensor, quant_boxes),
            (scores_tensor, quant_scores),
            (mask_tensor, quant_masks),
            (protos_tensor, quant_protos),
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_modelpack_seg_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;
        let mut seg = seg.slice(s![0, .., .., ..]);
        if segmentation.channels_first {
            seg.swap_axes(0, 1);
            seg.swap_axes(1, 2);
        };

        let max = seg.max().unwrap_or(&255.0);
        let min = seg.min().unwrap_or(&0.0);
        let seg = seg.mapv(|x| ((x - min) / (max - min) * 255.0) as u8);
        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_modelpack_f32(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        decode_yolo_f32(
            boxes_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        boxes: &configs::Segmentation,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        };

        decode_yolo_segdet_f32(
            boxes_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_yolo_split_det_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_yolo_split_det_f32(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_f32(
        &self,
        outputs: &[ArrayViewD<f32>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mut mask_tensor = mask_tensor.slice(s![0, .., ..]);
        if mask_coeff.channels_first {
            mask_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        }

        decode_yolo_split_segdet_f32(
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

    fn decode_modelpack_seg_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;
        let mut seg = seg.slice(s![0, .., .., ..]);
        if segmentation.channels_first {
            seg.swap_axes(0, 1);
            seg.swap_axes(1, 2);
        };

        let max = seg.max().unwrap_or(&255.0);
        let min = seg.min().unwrap_or(&0.0);
        let seg = seg.mapv(|x| ((x - min) / (max - min) * 255.0) as u8);
        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_modelpack_f64(
            boxes_tensor,
            scores_tensor,
            self.score_threshold as f64,
            self.iou_threshold as f64,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        decode_yolo_f64(
            boxes_tensor,
            self.score_threshold as f64,
            self.iou_threshold as f64,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        boxes: &configs::Segmentation,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        };

        decode_yolo_segdet_f64(
            boxes_tensor,
            protos_tensor,
            self.score_threshold as f64,
            self.iou_threshold as f64,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn decode_yolo_split_det_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };

        skip.push(ind);
        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };

        decode_yolo_split_det_f64(
            boxes_tensor,
            scores_tensor,
            self.score_threshold as f64,
            self.iou_threshold as f64,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_yolo_split_segdet_f64(
        &self,
        outputs: &[ArrayViewD<f64>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let mut boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        if boxes.channels_first {
            boxes_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let mut scores_tensor = scores_tensor.slice(s![0, .., ..]);
        if scores.channels_first {
            scores_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mut mask_tensor = mask_tensor.slice(s![0, .., ..]);
        if mask_coeff.channels_first {
            mask_tensor.swap_axes(0, 1);
        };
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let mut protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        if protos.channels_first {
            protos_tensor.swap_axes(0, 1);
            protos_tensor.swap_axes(1, 2);
        }

        decode_yolo_split_segdet_f64(
            boxes_tensor,
            scores_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold as f64,
            self.iou_threshold as f64,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn match_outputs_to_detect<'a, 'b, T>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewD<'b, T>],
    ) -> Result<Vec<&'a ArrayViewD<'b, T>>, Error> {
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
                return Err(Error::InvalidShape(format!(
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
    ) -> Result<(&'a ArrayViewD<'b, T>, usize), Error> {
        for (ind, o) in outputs.iter().enumerate() {
            if skip.contains(&ind) {
                continue;
            }
            if o.shape() == shape {
                return Ok((o, ind));
            }
        }
        Err(Error::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }
}
