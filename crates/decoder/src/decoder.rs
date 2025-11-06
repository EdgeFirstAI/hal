use ndarray::{Array, Array3, ArrayViewD, s};
use ndarray_stats::QuantileExt;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use serde_yaml::with;

use crate::{
    DetectBox, Error, Quantization, Segmentation, XYWH,
    configs::{DecoderType, ModelType, QuantTuple},
    dequantize_ndarray,
    modelpack::{
        ModelPackDetectionConfig, decode_modelpack_det, decode_modelpack_f32, decode_modelpack_f64,
        decode_modelpack_split, decode_modelpack_split_float,
    },
    yolo::{
        decode_segdet_8bit, decode_yolo_det, decode_yolo_f32, decode_yolo_f64, decode_yolo_segdet,
        decode_yolo_segdet_f32, decode_yolo_segdet_f64, decode_yolo_split_det,
        decode_yolo_split_det_f32, decode_yolo_split_det_f64, decode_yolo_split_segdet,
        decode_yolo_split_segdet_f32, decode_yolo_split_segdet_f64,
        impl_yolo_split_segdet_8bit_get_boxes,
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

#[derive(Debug)]
pub enum ArrayViewDQuantized<'a> {
    UInt8_(ArrayViewD<'a, u8>),
    Int8__(ArrayViewD<'a, i8>),
    UInt16(ArrayViewD<'a, u16>),
    Int16_(ArrayViewD<'a, i16>),
}

impl<'a> From<ArrayViewD<'a, u8>> for ArrayViewDQuantized<'a> {
    fn from(arr: ArrayViewD<'a, u8>) -> Self {
        Self::UInt8_(arr)
    }
}

impl<'a> From<ArrayViewD<'a, i8>> for ArrayViewDQuantized<'a> {
    fn from(arr: ArrayViewD<'a, i8>) -> Self {
        Self::Int8__(arr)
    }
}

impl<'a> From<ArrayViewD<'a, u16>> for ArrayViewDQuantized<'a> {
    fn from(arr: ArrayViewD<'a, u16>) -> Self {
        Self::UInt16(arr)
    }
}

impl<'a> From<ArrayViewD<'a, i16>> for ArrayViewDQuantized<'a> {
    fn from(arr: ArrayViewD<'a, i16>) -> Self {
        Self::Int16_(arr)
    }
}

impl<'a> ArrayViewDQuantized<'a> {
    fn shape(&self) -> &[usize] {
        match self {
            ArrayViewDQuantized::UInt8_(a) => a.shape(),
            ArrayViewDQuantized::Int8__(a) => a.shape(),
            ArrayViewDQuantized::UInt16(a) => a.shape(),
            ArrayViewDQuantized::Int16_(a) => a.shape(),
        }
    }
}

impl Decoder {
    pub fn model_type(&self) -> &ModelType {
        &self.model_type
    }

    pub fn decode_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
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

    fn decode_modelpack_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
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
        macro_rules! yolo_split_quant {
            ($boxes_tensor:expr, $scores_tensor:expr) => {{
                let mut boxes_tensor = $boxes_tensor.slice(s![0, .., 0, ..]);
                if boxes.channels_first {
                    boxes_tensor.swap_axes(0, 1);
                };

                let mut scores_tensor = $scores_tensor.slice(s![0, .., ..]);
                if scores.channels_first {
                    scores_tensor.swap_axes(0, 1);
                };
                decode_modelpack_det(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            }};
        }
        use ArrayViewDQuantized::*;
        match (boxes_tensor, scores_tensor) {
            (UInt8_(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), Int8__(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), UInt16(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), Int16_(s)) => yolo_split_quant!(b, s),

            (Int8__(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (Int8__(b), Int8__(s)) => yolo_split_quant!(b, s),
            (Int8__(b), UInt16(s)) => yolo_split_quant!(b, s),
            (Int8__(b), Int16_(s)) => yolo_split_quant!(b, s),

            (UInt16(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (UInt16(b), Int8__(s)) => yolo_split_quant!(b, s),
            (UInt16(b), UInt16(s)) => yolo_split_quant!(b, s),
            (UInt16(b), Int16_(s)) => yolo_split_quant!(b, s),

            (Int16_(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (Int16_(b), Int8__(s)) => yolo_split_quant!(b, s),
            (Int16_(b), UInt16(s)) => yolo_split_quant!(b, s),
            (Int16_(b), Int16_(s)) => yolo_split_quant!(b, s),
        }

        Ok(())
    }

    fn decode_modelpack_seg_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
        let (seg, _) = Self::find_outputs_with_shape_quantized(&segmentation.shape, outputs, &[])?;

        macro_rules! modelpack_seg {
            ($seg:expr) => {{
                let mut seg = $seg.slice(s![0, .., .., ..]);
                if segmentation.channels_first {
                    seg.swap_axes(0, 1);
                    seg.swap_axes(1, 2);
                }
                seg
            }};
        }
        use ArrayViewDQuantized::*;
        let seg = match seg {
            UInt8_(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| x)
            }
            Int8__(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| (x as i16 + 128) as u8)
            }
            UInt16(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| (x >> 8) as u8)
            }
            Int16_(s) => {
                let seg = modelpack_seg!(s);
                seg.mapv(|x| ((x as i32 + 32768) >> 8) as u8)
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
    ) -> Result<(), Error> {
        let new_detection = detection
            .iter()
            .map(|x| ModelPackDetectionConfig {
                anchors: x.anchors.clone().unwrap(),
                quantization: x.quantization.map(Quantization::from),
            })
            .collect::<Vec<_>>();
        let new_outputs = Self::match_outputs_to_detect_quantized(detection, outputs)?;

        macro_rules! dequant_output {
            ($det_tensor:expr, $detection:expr) => {{
                let mut det_tensor = $det_tensor.slice(s![0, .., .., ..]);
                if $detection.channels_first {
                    det_tensor.swap_axes(0, 1);
                    det_tensor.swap_axes(1, 2);
                }
                if let Some(q) = $detection.quantization {
                    dequantize_ndarray(det_tensor, q.into())
                } else {
                    det_tensor.map(|x| *x as f32)
                }
            }};
        }
        use ArrayViewDQuantized::*;
        let new_outputs = new_outputs
            .iter()
            .zip(detection)
            .map(|(det_tensor, detection)| match det_tensor {
                UInt8_(d) => dequant_output!(d, detection),
                Int8__(d) => dequant_output!(d, detection),
                UInt16(d) => dequant_output!(d, detection),
                Int16_(d) => dequant_output!(d, detection),
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
    ) -> Result<(), Error> {
        let (boxes_tensor, _) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;

        macro_rules! yolo_quant {
            ($boxes_tensor:expr) => {{
                let quant_boxes = boxes
                    .quantization
                    .map(Quantization::from)
                    .unwrap_or_default();

                let mut boxes_tensor = $boxes_tensor.slice(s![0, .., ..]);
                if boxes.channels_first {
                    boxes_tensor.swap_axes(0, 1);
                };
                decode_yolo_det(
                    (boxes_tensor, quant_boxes),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            }};
        }
        use ArrayViewDQuantized::*;
        match boxes_tensor {
            UInt8_(b) => yolo_quant!(b),
            Int8__(b) => yolo_quant!(b),
            UInt16(b) => yolo_quant!(b),
            Int16_(b) => yolo_quant!(b),
        }

        Ok(())
    }

    fn decode_yolo_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Segmentation,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), Error> {
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
        macro_rules! yolo_quant {
            ($boxes_tensor:expr, $protos_tensor:expr) => {{
                let mut box_tensor = $boxes_tensor.slice(s![0, .., ..]);
                if boxes.channels_first {
                    box_tensor.swap_axes(0, 1);
                }

                let mut protos_tensor = $protos_tensor.slice(s![0, .., .., ..]);
                if protos.channels_first {
                    protos_tensor.swap_axes(0, 1);
                    protos_tensor.swap_axes(1, 2);
                }

                decode_yolo_segdet(
                    (box_tensor, quant_boxes),
                    (protos_tensor, quant_protos),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                    output_masks,
                );
            }};
        }
        use ArrayViewDQuantized::*;
        match (boxes_tensor, protos_tensor) {
            (UInt8_(b), UInt8_(p)) => yolo_quant!(b, p),
            (UInt8_(b), Int8__(p)) => yolo_quant!(b, p),
            (UInt8_(b), UInt16(p)) => yolo_quant!(b, p),
            (UInt8_(b), Int16_(p)) => yolo_quant!(b, p),
            (Int8__(b), UInt8_(p)) => yolo_quant!(b, p),
            (Int8__(b), Int8__(p)) => yolo_quant!(b, p),
            (Int8__(b), UInt16(p)) => yolo_quant!(b, p),
            (Int8__(b), Int16_(p)) => yolo_quant!(b, p),
            (UInt16(b), UInt8_(p)) => yolo_quant!(b, p),
            (UInt16(b), Int8__(p)) => yolo_quant!(b, p),
            (UInt16(b), UInt16(p)) => yolo_quant!(b, p),
            (UInt16(b), Int16_(p)) => yolo_quant!(b, p),
            (Int16_(b), UInt8_(p)) => yolo_quant!(b, p),
            (Int16_(b), Int8__(p)) => yolo_quant!(b, p),
            (Int16_(b), UInt16(p)) => yolo_quant!(b, p),
            (Int16_(b), Int16_(p)) => yolo_quant!(b, p),
        }

        Ok(())
    }

    fn decode_yolo_split_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
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
        macro_rules! yolo_split_quant {
            ($boxes_tensor:expr, $scores_tensor:expr) => {{
                let mut boxes_tensor = $boxes_tensor.slice(s![0, .., ..]);
                if boxes.channels_first {
                    boxes_tensor.swap_axes(0, 1);
                };

                let mut scores_tensor = $scores_tensor.slice(s![0, .., ..]);
                if scores.channels_first {
                    scores_tensor.swap_axes(0, 1);
                };
                decode_yolo_split_det(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            }};
        }
        use ArrayViewDQuantized::*;
        match (boxes_tensor, scores_tensor) {
            (UInt8_(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), Int8__(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), UInt16(s)) => yolo_split_quant!(b, s),
            (UInt8_(b), Int16_(s)) => yolo_split_quant!(b, s),

            (Int8__(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (Int8__(b), Int8__(s)) => yolo_split_quant!(b, s),
            (Int8__(b), UInt16(s)) => yolo_split_quant!(b, s),
            (Int8__(b), Int16_(s)) => yolo_split_quant!(b, s),

            (UInt16(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (UInt16(b), Int8__(s)) => yolo_split_quant!(b, s),
            (UInt16(b), UInt16(s)) => yolo_split_quant!(b, s),
            (UInt16(b), Int16_(s)) => yolo_split_quant!(b, s),

            (Int16_(b), UInt8_(s)) => yolo_split_quant!(b, s),
            (Int16_(b), Int8__(s)) => yolo_split_quant!(b, s),
            (Int16_(b), UInt16(s)) => yolo_split_quant!(b, s),
            (Int16_(b), Int16_(s)) => yolo_split_quant!(b, s),
        }

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
    ) -> Result<(), Error> {
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

        macro_rules! with_quantized {
            ($x:expr, $var:ident, $body:expr) => {
                match $x {
                    ArrayViewDQuantized::UInt8_(x) => {
                        let $var = x;
                        $body
                    }
                    ArrayViewDQuantized::Int8__(x) => {
                        let $var = x;
                        $body
                    }
                    ArrayViewDQuantized::UInt16(x) => {
                        let $var = x;
                        $body
                    }
                    ArrayViewDQuantized::Int16_(x) => {
                        let $var = x;
                        $body
                    }
                }
            };
        }

        // This works but compiles really slowly (1 min+ for cargo clippy)
        // macro_rules! yolo_split_segdet_quant {
        //     ($boxes_tensor:expr, $scores_tensor:expr, $mask_tensor:expr,
        // $protos_tensor:expr) => {{
        //         let mut boxes_tensor = $boxes_tensor.slice(s![0, .., ..]);
        //         if boxes.channels_first {
        //             boxes_tensor.swap_axes(0, 1);
        //         };

        //         let mut scores_tensor = $scores_tensor.slice(s![0, .., ..]);
        //         if scores.channels_first {
        //             scores_tensor.swap_axes(0, 1);
        //         };

        //         let mut mask_tensor = $mask_tensor.slice(s![0, .., ..]);
        //         if mask_coeff.channels_first {
        //             mask_tensor.swap_axes(0, 1);
        //         };

        //         let mut protos_tensor = $protos_tensor.slice(s![0, .., .., ..]);
        //         if protos.channels_first {
        //             protos_tensor.swap_axes(0, 1);
        //             protos_tensor.swap_axes(1, 2);
        //         }

        //         decode_yolo_split_segdet(
        //             (boxes_tensor, quant_boxes),
        //             (scores_tensor, quant_scores),
        //             (mask_tensor, quant_masks),
        //             (protos_tensor, quant_protos),
        //             self.score_threshold,
        //             self.iou_threshold,
        //             output_boxes,
        //             output_masks,
        //         );
        //     }};
        // }
        // with_quantized!(boxes_tensor, b, {
        //     with_quantized!(scores_tensor, s, {
        //         with_quantized!(mask_tensor, m, {
        //             with_quantized!(protos_tensor, p, {
        //                 yolo_split_segdet_quant!(b, s, m, p);
        //             });
        //         });
        //     });
        // });

        macro_rules! yolo_split_segdet_get_boxes {
            ($boxes_tensor:expr, $scores_tensor:expr, $mask_tensor:expr) => {{
                let mut boxes_tensor = $boxes_tensor.slice(s![0, .., ..]);
                if boxes.channels_first {
                    boxes_tensor.swap_axes(0, 1);
                };

                let mut scores_tensor = $scores_tensor.slice(s![0, .., ..]);
                if scores.channels_first {
                    scores_tensor.swap_axes(0, 1);
                };

                impl_yolo_split_segdet_8bit_get_boxes::<XYWH, _, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    ($mask_tensor, quant_masks),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes.capacity(),
                )
            }};
        }

        with_quantized!(mask_tensor, m, {
            let mut m = m.slice(s![0, .., ..]);
            if mask_coeff.channels_first {
                m.swap_axes(0, 1);
            };

            let boxes = with_quantized!(boxes_tensor, b, {
                with_quantized!(scores_tensor, s, yolo_split_segdet_get_boxes!(b, s, m))
            });

            with_quantized!(protos_tensor, p, {
                let mut protos_tensor = p.slice(s![0, .., .., ..]);
                if protos.channels_first {
                    protos_tensor.swap_axes(0, 1);
                    protos_tensor.swap_axes(1, 2);
                }

                let boxes = decode_segdet_8bit(boxes, protos_tensor, quant_masks, quant_protos);
                output_boxes.clear();
                output_masks.clear();
                for (b, m) in boxes.into_iter() {
                    output_boxes.push(b);
                    output_masks.push(Segmentation {
                        xmin: b.bbox.xmin,
                        ymin: b.bbox.ymin,
                        xmax: b.bbox.xmax,
                        ymax: b.bbox.ymax,
                        segmentation: m,
                    });
                }
            })
        });

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
            self.score_threshold,
            self.iou_threshold,
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
            self.score_threshold,
            self.iou_threshold,
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
            self.score_threshold,
            self.iou_threshold,
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

    fn find_outputs_with_shape_quantized<'a, 'b>(
        shape: &[usize],
        outputs: &'a [ArrayViewDQuantized<'b>],
        skip: &[usize],
    ) -> Result<(&'a ArrayViewDQuantized<'b>, usize), Error> {
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

    fn match_outputs_to_detect_quantized<'a, 'b>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewDQuantized<'b>],
    ) -> Result<Vec<&'a ArrayViewDQuantized<'b>>, Error> {
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
}
