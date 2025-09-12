use ndarray::{ArrayViewD, s};
use num_traits::{AsPrimitive, FromPrimitive};
use serde::{Deserialize, Serialize};

use crate::{
    DetectBox, Error, Quantization, SegmentationMask, XYWH, XYXY, dequantize_ndarray,
    modelpack::{
        ModelPackDetectionConfig, decode_modelpack_i8, decode_modelpack_split, decode_modelpack_u8,
    },
    yolo::{decode_yolo_i8, decode_yolo_masks_f32, decode_yolo_u8},
};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ConfigOutputs {
    pub outputs: Vec<ConfigOutput>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ConfigOutput {
    #[serde(rename = "detection")]
    Detection(Detection),
    #[serde(rename = "masks")]
    Mask(Mask),
    #[serde(rename = "segmentation")]
    Segmentation(Segmentation),
    #[serde(rename = "scores")]
    Scores(Scores),
    #[serde(rename = "boxes")]
    Boxes(Boxes),
}

impl ConfigOutput {
    pub fn shape(&self) -> &[usize] {
        match self {
            ConfigOutput::Detection(detection) => &detection.shape,
            ConfigOutput::Mask(mask) => &mask.shape,
            ConfigOutput::Segmentation(segmentation) => &segmentation.shape,
            ConfigOutput::Scores(scores) => &scores.shape,
            ConfigOutput::Boxes(boxes) => &boxes.shape,
        }
    }

    pub fn decoder(&self) -> &DecoderType {
        match self {
            ConfigOutput::Detection(detection) => &detection.decoder,
            ConfigOutput::Mask(mask) => &mask.decoder,
            ConfigOutput::Segmentation(segmentation) => &segmentation.decoder,
            ConfigOutput::Scores(scores) => &scores.decoder,
            ConfigOutput::Boxes(boxes) => &boxes.decoder,
        }
    }

    pub fn dtype(&self) -> &DataType {
        match self {
            ConfigOutput::Detection(detection) => &detection.dtype,
            ConfigOutput::Mask(mask) => &mask.dtype,
            ConfigOutput::Segmentation(segmentation) => &segmentation.dtype,
            ConfigOutput::Scores(scores) => &scores.dtype,
            ConfigOutput::Boxes(boxes) => &boxes.dtype,
        }
    }

    pub fn quantization(&self) -> &Option<[f32; 2]> {
        match self {
            ConfigOutput::Detection(detection) => &detection.quantization,
            ConfigOutput::Mask(mask) => &mask.quantization,
            ConfigOutput::Segmentation(segmentation) => &segmentation.quantization,
            ConfigOutput::Scores(scores) => &scores.quantization,
            ConfigOutput::Boxes(boxes) => &boxes.quantization,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Segmentation {
    pub decode: bool,
    #[serde(rename = "decoder")]
    pub decoder: DecoderType,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Mask {
    pub decode: bool,
    pub decoder: DecoderType,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Detection {
    pub anchors: Option<Vec<[f32; 2]>>,
    pub decode: bool,
    pub decoder: DecoderType,
    pub dtype: DataType,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Scores {
    pub decoder: DecoderType,
    pub dtype: DataType,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Boxes {
    pub decoder: DecoderType,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
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
        protos: Segmentation,
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
        for c in configs {
            match c {
                ConfigOutput::Detection(detection) => boxes = Some(detection),
                ConfigOutput::Segmentation(segmentation) => {
                    if segmentation.shape.len() == 4 {
                        protos = Some(segmentation)
                    } else if segmentation.shape.len() == 3 && segmentation.shape[1] > 4 + 32 {
                        seg_boxes = Some(segmentation)
                    } else {
                        return Err(Error::InvalidConfig(format!(
                            "Invalid Yolo Segmentation shape {:?}",
                            segmentation.shape
                        )));
                    }
                }
                ConfigOutput::Mask(_) => {
                    return Err(Error::InvalidConfig(
                        "Invalid Mask output with Yolo decoder".to_string(),
                    ));
                }
                ConfigOutput::Scores(_) => {
                    return Err(Error::InvalidConfig(
                        "Invalid Scores output with Yolo decoder".to_string(),
                    ));
                }
                ConfigOutput::Boxes(_) => {
                    return Err(Error::InvalidConfig(
                        "Invalid Boxes output with Yolo decoder".to_string(),
                    ));
                }
            }
        }
        if let Some(protos) = protos
            && let Some(boxes) = seg_boxes
        {
            Ok(ModelType::YoloSegDet { boxes, protos })
        } else if let Some(boxes) = boxes {
            Ok(ModelType::YoloDet { boxes })
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
                ConfigOutput::Scores(scores) => scores_ = Some(scores),
                ConfigOutput::Boxes(boxes) => boxes_ = Some(boxes),
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
    iou_threshold: f32,
    score_threshold: f32,
}

impl Decoder {
    pub fn decode_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<SegmentationMask>,
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
        }
        Ok(())
    }

    pub fn decode_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<SegmentationMask>,
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
        }
        Ok(())
    }

    fn decode_modelpack_det_split<D: AsPrimitive<f32> + FromPrimitive>(
        &self,
        outputs: &[ArrayViewD<D>],
        detection: &[Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let new_outputs = Self::match_outputs_to_detect(detection, outputs)?;
        let new_outputs = new_outputs
            .into_iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>();
        let new_detection = detection
            .iter()
            .map(|x| ModelPackDetectionConfig {
                anchors: x.anchors.clone().unwrap(),
                quantization: x.quantization.map(|x| Quantization {
                    scale: x[0],
                    zero_point: D::from_f32(x[1]).unwrap(),
                }),
            })
            .collect::<Vec<_>>();
        decode_modelpack_split::<XYWH, D>(
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
        segmentation: &Segmentation,
        output_masks: &mut Vec<SegmentationMask>,
    ) -> Result<(), Error> {
        let seg = Self::find_outputs_with_shape(&segmentation.shape, outputs)?;
        let seg = seg.slice(s![0, .., .., ..]);
        let seg = seg.mapv(|x| (x as i16 + 128) as u8);
        output_masks.push(SegmentationMask {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            mask: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &Boxes,
        scores: &Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let boxes = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let boxes = boxes.slice(s![0, .., 0, ..]);

        let scores_quant = scores.quantization.unwrap_or([1.0, 0.0]).into();
        let scores = Self::find_outputs_with_shape(&scores.shape, outputs)?;
        let scores = scores.slice(s![0, .., ..]);

        decode_modelpack_i8::<XYXY>(
            boxes,
            scores,
            &box_quant,
            &scores_quant,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let box_output = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let box_output = box_output.slice(s![0, .., ..]);

        decode_yolo_i8::<XYWH>(
            box_output,
            &box_quant,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_i8(
        &self,
        outputs: &[ArrayViewD<i8>],
        boxes: &Segmentation,
        protos: &Segmentation,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<SegmentationMask>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let box_output: &ndarray::ArrayBase<
            ndarray::ViewRepr<&i8>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let box_output = dequantize_ndarray(&box_quant, box_output.into());
        let box_output = box_output.slice(s![0, .., ..]);

        let protos_quant = protos.quantization.unwrap_or([1.0, 0.0]).into();
        let protos = Self::find_outputs_with_shape(&protos.shape, outputs)?;
        let protos = dequantize_ndarray(&protos_quant, protos.into());
        let protos = protos.slice(s![0, .., .., ..]);

        decode_yolo_masks_f32::<XYWH>(
            box_output,
            protos,
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
        segmentation: &Segmentation,
        output_masks: &mut Vec<SegmentationMask>,
    ) -> Result<(), Error> {
        let seg = Self::find_outputs_with_shape(&segmentation.shape, outputs)?;
        let seg = seg.slice(s![0, .., .., ..]);

        // TODO: Adjust signatures so this doesn't need to clone the entire backing
        // array?
        let seg = seg.to_owned();
        output_masks.push(SegmentationMask {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            mask: seg,
        });
        Ok(())
    }

    fn decode_modelpack_det_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &Boxes,
        scores: &Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let boxes = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let boxes = boxes.slice(s![0, .., 0, ..]);

        let scores_quant = scores.quantization.unwrap_or([1.0, 0.0]).into();
        let scores = Self::find_outputs_with_shape(&scores.shape, outputs)?;
        let scores = scores.slice(s![0, .., ..]);

        decode_modelpack_u8::<XYXY>(
            boxes,
            scores,
            &box_quant,
            &scores_quant,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_det_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let box_output = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let box_output = box_output.slice(s![0, .., ..]);

        decode_yolo_u8::<XYWH>(
            box_output,
            &box_quant,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    fn decode_yolo_segdet_u8(
        &self,
        outputs: &[ArrayViewD<u8>],
        boxes: &Segmentation,
        protos: &Segmentation,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<SegmentationMask>,
    ) -> Result<(), Error> {
        let box_quant = boxes.quantization.unwrap_or([1.0, 0.0]).into();
        let box_output = Self::find_outputs_with_shape(&boxes.shape, outputs)?;
        let box_output = dequantize_ndarray(&box_quant, box_output.into());
        let box_output = box_output.slice(s![0, .., ..]);

        let protos_quant = protos.quantization.unwrap_or([1.0, 0.0]).into();
        let protos = Self::find_outputs_with_shape(&protos.shape, outputs)?;
        let protos = dequantize_ndarray(&protos_quant, protos.into());
        let protos = protos.slice(s![0, .., .., ..]);

        decode_yolo_masks_f32::<XYWH>(
            box_output,
            protos,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
            output_masks,
        );
        Ok(())
    }

    fn match_outputs_to_detect<'a, 'b, T>(
        configs: &[Detection],
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
    ) -> Result<&'a ArrayViewD<'b, T>, Error> {
        for o in outputs {
            if o.shape() == shape {
                return Ok(o);
            }
        }
        Err(Error::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }
}
