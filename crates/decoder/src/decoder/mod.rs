// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayView, ArrayViewD, Dimension};
use num_traits::{AsPrimitive, Float};

use crate::{DecoderError, DetectBox, ProtoData, Segmentation};

pub mod config;
pub mod configs;

use configs::ModelType;

#[derive(Debug, Clone, PartialEq)]
pub struct Decoder {
    model_type: ModelType,
    pub iou_threshold: f32,
    pub score_threshold: f32,
    /// NMS mode: Some(mode) applies NMS, None bypasses NMS (for end-to-end
    /// models)
    pub nms: Option<configs::Nms>,
    /// Whether decoded boxes are in normalized [0,1] coordinates.
    /// - `Some(true)`: Coordinates in [0,1] range
    /// - `Some(false)`: Pixel coordinates
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate >
    ///   1.0)
    normalized: Option<bool>,
}

#[doc(hidden)]
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

/// WARNING: Do NOT nest `with_quantized!` calls. Each level multiplies
/// monomorphized code paths by 6 (one per integer variant), so nesting
/// N levels deep produces 6^N instantiations.
///
/// Instead, dequantize each tensor sequentially with `dequant_3d!`/`dequant_4d!`
/// (6*N paths) or split into independent phases that each nest at most 2 levels.
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

mod builder;
mod helpers;
mod postprocess;
mod tensor_bridge;
mod tests;

pub use builder::DecoderBuilder;
pub use config::{ConfigOutput, ConfigOutputRef, ConfigOutputs};

impl Decoder {
    /// This function returns the parsed model type of the decoder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, configs::ModelType};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string();
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

    /// Returns the box coordinate format if known from the model config.
    ///
    /// - `Some(true)`: Boxes are in normalized [0,1] coordinates
    /// - `Some(false)`: Boxes are in pixel coordinates relative to model input
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate >
    ///   1.0)
    ///
    /// This is determined by the model config's `normalized` field, not the NMS
    /// mode. When coordinates are in pixels or unknown, the caller may need
    /// to normalize using the model input dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string();
    ///     let decoder = DecoderBuilder::default()
    ///         .with_config_yaml_str(config_yaml)
    ///         .build()?;
    ///     // Config doesn't specify normalized, so it's None
    ///     assert!(decoder.normalized_boxes().is_none());
    /// #    Ok(())
    /// # }
    /// ```
    pub fn normalized_boxes(&self) -> Option<bool> {
        self.normalized
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
    /// #    let detect0 = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split_9x15x18.bin"));
    /// #    let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec())?;
    /// #
    /// #    let detect1 = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split_17x30x18.bin"));
    /// #    let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec())?;
    /// #    let model_output = vec![
    /// #        detect1.view().into_dyn().into(),
    /// #        detect0.view().into_dyn().into(),
    /// #    ];
    /// let decoder = DecoderBuilder::default()
    ///     .with_config_yaml_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string())
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
    #[doc(hidden)]
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
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => self.decode_yolo_segdet_2way_quantized(
                outputs,
                boxes,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
            ),
            ModelType::YoloEndToEndDet { boxes } => {
                self.decode_yolo_end_to_end_det_quantized(outputs, boxes, output_boxes)
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => self
                .decode_yolo_end_to_end_segdet_quantized(
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                ),
            ModelType::YoloSplitEndToEndDet {
                boxes,
                scores,
                classes,
            } => self.decode_yolo_split_end_to_end_det_quantized(
                outputs,
                boxes,
                scores,
                classes,
                output_boxes,
            ),
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => self.decode_yolo_split_end_to_end_segdet_quantized(
                outputs,
                boxes,
                scores,
                classes,
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
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult, configs, configs::{DecoderType, DecoderVersion}, dequantize_cpu, Quantization};
    /// # use ndarray::Array3;
    /// # fn main() -> DecoderResult<()> {
    /// #   let out = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/yolov8s_80_classes.bin"));
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
    ///         normalized: Some(true),
    ///     },
    ///     Some(DecoderVersion::Yolo11))
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
    #[doc(hidden)]
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
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_segdet_2way_float(
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                )?;
            }
            ModelType::YoloEndToEndDet { boxes } => {
                self.decode_yolo_end_to_end_det_float(outputs, boxes, output_boxes)?;
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                self.decode_yolo_end_to_end_segdet_float(
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                )?;
            }
            ModelType::YoloSplitEndToEndDet {
                boxes,
                scores,
                classes,
            } => {
                self.decode_yolo_split_end_to_end_det_float(
                    outputs,
                    boxes,
                    scores,
                    classes,
                    output_boxes,
                )?;
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                self.decode_yolo_split_end_to_end_segdet_float(
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                )?;
            }
        }
        Ok(())
    }

    /// Decodes quantized model outputs into detection boxes, returning raw
    /// `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (use
    /// `decode_quantized` for those). Returns `Ok(Some(ProtoData))` for
    /// YOLO segmentation models.
    #[doc(hidden)]
    pub fn decode_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only and ModelPack variants: no proto data
            ModelType::ModelPackSegDet { .. }
            | ModelType::ModelPackSegDetSplit { .. }
            | ModelType::ModelPackDet { .. }
            | ModelType::ModelPackDetSplit { .. }
            | ModelType::ModelPackSeg { .. }
            | ModelType::YoloDet { .. }
            | ModelType::YoloSplitDet { .. }
            | ModelType::YoloEndToEndDet { .. }
            | ModelType::YoloSplitEndToEndDet { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto =
                    self.decode_yolo_segdet_quantized_proto(outputs, boxes, protos, output_boxes)?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_split_segdet_quantized_proto(
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_segdet_2way_quantized_proto(
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                let proto = self.decode_yolo_end_to_end_segdet_quantized_proto(
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_split_end_to_end_segdet_quantized_proto(
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
        }
    }

    /// Decodes floating-point model outputs into detection boxes, returning
    /// raw `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models. Returns
    /// `Ok(Some(ProtoData))` for YOLO segmentation models.
    #[doc(hidden)]
    pub fn decode_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only and ModelPack variants: no proto data
            ModelType::ModelPackSegDet { .. }
            | ModelType::ModelPackSegDetSplit { .. }
            | ModelType::ModelPackDet { .. }
            | ModelType::ModelPackDetSplit { .. }
            | ModelType::ModelPackSeg { .. }
            | ModelType::YoloDet { .. }
            | ModelType::YoloSplitDet { .. }
            | ModelType::YoloEndToEndDet { .. }
            | ModelType::YoloSplitEndToEndDet { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto =
                    self.decode_yolo_segdet_float_proto(outputs, boxes, protos, output_boxes)?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_split_segdet_float_proto(
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_segdet_2way_float_proto(
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                let proto = self.decode_yolo_end_to_end_segdet_float_proto(
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_yolo_split_end_to_end_segdet_float_proto(
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                )?;
                Ok(Some(proto))
            }
        }
    }

    // ========================================================================
    // TensorDyn-based public API
    // ========================================================================

    /// Decode model outputs into detection boxes and segmentation masks.
    ///
    /// This is the primary decode API. Accepts `TensorDyn` outputs directly
    /// from model inference. Automatically dispatches to quantized or float
    /// paths based on the tensor dtype.
    ///
    /// # Arguments
    ///
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    /// * `output_masks` - Destination for decoded segmentation masks (cleared first)
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if tensor mapping fails, dtypes are unsupported,
    /// or the outputs don't match the decoder's model configuration.
    pub fn decode(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let mapped = tensor_bridge::map_tensors(outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized(&views, output_boxes, output_masks)
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_float(&views, output_boxes, output_masks)
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_float(&views, output_boxes, output_masks)
            }
        }
    }

    /// Decode model outputs into detection boxes, returning raw proto data
    /// for segmentation models instead of materialized masks.
    ///
    /// Accepts `TensorDyn` outputs directly from model inference.
    /// Returns `Ok(None)` for detection-only and ModelPack models.
    /// Returns `Ok(Some(ProtoData))` for YOLO segmentation models.
    ///
    /// # Arguments
    ///
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if tensor mapping fails, dtypes are unsupported,
    /// or the outputs don't match the decoder's model configuration.
    pub fn decode_proto(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        let mapped = tensor_bridge::map_tensors(outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized_proto(&views, output_boxes)
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_float_proto(&views, output_boxes)
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_float_proto(&views, output_boxes)
            }
        }
    }
}

#[cfg(feature = "tracker")]
pub use edgefirst_tracker::TrackInfo;

#[cfg(feature = "tracker")]
pub use edgefirst_tracker::Tracker;

#[cfg(feature = "tracker")]
impl Decoder {
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
    /// #    let detect0 = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split_9x15x18.bin"));
    /// #    let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec())?;
    /// #
    /// #    let detect1 = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split_17x30x18.bin"));
    /// #    let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec())?;
    /// #    let model_output = vec![
    /// #        detect1.view().into_dyn().into(),
    /// #        detect0.view().into_dyn().into(),
    /// #    ];
    /// let decoder = DecoderBuilder::default()
    ///     .with_config_yaml_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/modelpack_split.yaml")).to_string())
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
    #[doc(hidden)]
    pub fn decode_tracked_quantized<TR: edgefirst_tracker::Tracker<DetectBox>>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<(), DecoderError> {
        output_boxes.clear();
        output_masks.clear();
        output_tracks.clear();

        // yolo segdet variants require special handling to separate boxes that come from decoding vs active tracks.
        // Only boxes that come from decoding can be used for proto/mask generation.
        match &self.model_type {
            ModelType::YoloSegDet { boxes, protos } => self.decode_tracked_yolo_segdet_quantized(
                tracker,
                timestamp,
                outputs,
                boxes,
                protos,
                output_boxes,
                output_masks,
                output_tracks,
            ),
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => self.decode_tracked_yolo_split_segdet_quantized(
                tracker,
                timestamp,
                outputs,
                boxes,
                scores,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
                output_tracks,
            ),
            ModelType::YoloEndToEndSegDet { boxes, protos } => self
                .decode_tracked_yolo_end_to_end_segdet_quantized(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                ),
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => self.decode_tracked_yolo_split_end_to_end_segdet_quantized(
                tracker,
                timestamp,
                outputs,
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
                output_tracks,
            ),
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => self.decode_tracked_yolo_segdet_2way_quantized(
                tracker,
                timestamp,
                outputs,
                boxes,
                mask_coeff,
                protos,
                output_boxes,
                output_masks,
                output_tracks,
            ),
            _ => {
                self.decode_quantized(outputs, output_boxes, output_masks)?;
                Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
                Ok(())
            }
        }
    }

    /// This function decodes floating point model outputs into detection boxes
    /// and segmentation masks. Up to `output_boxes.capacity()` boxes and
    /// masks will be decoded. The function clears the provided output
    /// vectors before populating them with the decoded results.
    ///
    /// This function returns an `Error` if the provided outputs don't
    /// match the configuration provided by the user when building the decoder.
    ///
    /// Any quantization information in the configuration will be ignored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{BoundingBox, DecoderBuilder, DetectBox, DecoderResult, configs, configs::{DecoderType, DecoderVersion}, dequantize_cpu, Quantization};
    /// # use ndarray::Array3;
    /// # fn main() -> DecoderResult<()> {
    /// #   let out = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/yolov8s_80_classes.bin"));
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
    ///         normalized: Some(true),
    ///     },
    ///     Some(DecoderVersion::Yolo11))
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
    #[doc(hidden)]
    pub fn decode_tracked_float<TR: edgefirst_tracker::Tracker<DetectBox>, T>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        output_masks.clear();
        output_tracks.clear();
        match &self.model_type {
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_tracked_yolo_segdet_float(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                self.decode_tracked_yolo_split_segdet_float(
                    tracker,
                    timestamp,
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
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                self.decode_tracked_yolo_end_to_end_segdet_float(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                self.decode_tracked_yolo_split_end_to_end_segdet_float(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                self.decode_tracked_yolo_segdet_2way_float(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )?;
            }
            _ => {
                self.decode_float(outputs, output_boxes, output_masks)?;
                Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
            }
        }
        Ok(())
    }

    /// Decodes quantized model outputs into detection boxes, returning raw
    /// `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (use
    /// `decode_quantized` for those). Returns `Ok(Some(ProtoData))` for
    /// YOLO segmentation models.
    #[doc(hidden)]
    pub fn decode_tracked_quantized_proto<TR: edgefirst_tracker::Tracker<DetectBox>>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        output_boxes.clear();
        output_tracks.clear();
        match &self.model_type {
            // Detection-only and ModelPack variants: no proto data
            ModelType::ModelPackSegDet { .. }
            | ModelType::ModelPackSegDetSplit { .. }
            | ModelType::ModelPackDet { .. }
            | ModelType::ModelPackDetSplit { .. }
            | ModelType::ModelPackSeg { .. }
            | ModelType::YoloDet { .. }
            | ModelType::YoloSplitDet { .. }
            | ModelType::YoloEndToEndDet { .. }
            | ModelType::YoloSplitEndToEndDet { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto = self.decode_tracked_yolo_segdet_quantized_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_split_segdet_quantized_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_segdet_2way_quantized_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                let proto = self.decode_tracked_yolo_end_to_end_segdet_quantized_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_split_end_to_end_segdet_quantized_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
        }
    }

    /// Decodes floating-point model outputs into detection boxes, returning
    /// raw `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models. Returns
    /// `Ok(Some(ProtoData))` for YOLO segmentation models.
    #[doc(hidden)]
    pub fn decode_tracked_float_proto<TR: edgefirst_tracker::Tracker<DetectBox>, T>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        output_tracks.clear();
        match &self.model_type {
            // Detection-only and ModelPack variants: no proto data
            ModelType::ModelPackSegDet { .. }
            | ModelType::ModelPackSegDetSplit { .. }
            | ModelType::ModelPackDet { .. }
            | ModelType::ModelPackDetSplit { .. }
            | ModelType::ModelPackSeg { .. }
            | ModelType::YoloDet { .. }
            | ModelType::YoloSplitDet { .. }
            | ModelType::YoloEndToEndDet { .. }
            | ModelType::YoloSplitEndToEndDet { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto = self.decode_tracked_yolo_segdet_float_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitSegDet {
                boxes,
                scores,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_split_segdet_float_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    scores,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSegDet2Way {
                boxes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_segdet_2way_float_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloEndToEndSegDet { boxes, protos } => {
                let proto = self.decode_tracked_yolo_end_to_end_segdet_float_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
            ModelType::YoloSplitEndToEndSegDet {
                boxes,
                scores,
                classes,
                mask_coeff,
                protos,
            } => {
                let proto = self.decode_tracked_yolo_split_end_to_end_segdet_float_proto(
                    tracker,
                    timestamp,
                    outputs,
                    boxes,
                    scores,
                    classes,
                    mask_coeff,
                    protos,
                    output_boxes,
                    output_tracks,
                )?;
                Ok(Some(proto))
            }
        }
    }

    // ========================================================================
    // TensorDyn-based tracked public API
    // ========================================================================

    /// Decode model outputs with tracking.
    ///
    /// Accepts `TensorDyn` outputs directly from model inference. Automatically
    /// dispatches to quantized or float paths based on the tensor dtype, then
    /// updates the tracker with the decoded boxes.
    ///
    /// # Arguments
    ///
    /// * `tracker` - The tracker instance to update
    /// * `timestamp` - Current frame timestamp
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    /// * `output_masks` - Destination for decoded segmentation masks (cleared first)
    /// * `output_tracks` - Destination for track info (cleared first)
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if tensor mapping fails, dtypes are unsupported,
    /// or the outputs don't match the decoder's model configuration.
    pub fn decode_tracked<TR: edgefirst_tracker::Tracker<DetectBox>>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<(), DecoderError> {
        let mapped = tensor_bridge::map_tensors(outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_tracked_quantized(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_tracked_float(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_tracked_float(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_masks,
                    output_tracks,
                )
            }
        }
    }

    /// Decode model outputs with tracking, returning raw proto data for
    /// segmentation models.
    ///
    /// Accepts `TensorDyn` outputs directly from model inference.
    /// Returns `Ok(None)` for detection-only and ModelPack models.
    /// Returns `Ok(Some(ProtoData))` for YOLO segmentation models.
    ///
    /// # Arguments
    ///
    /// * `tracker` - The tracker instance to update
    /// * `timestamp` - Current frame timestamp
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    /// * `output_tracks` - Destination for track info (cleared first)
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if tensor mapping fails, dtypes are unsupported,
    /// or the outputs don't match the decoder's model configuration.
    pub fn decode_proto_tracked<TR: edgefirst_tracker::Tracker<DetectBox>>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        let mapped = tensor_bridge::map_tensors(outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_tracked_quantized_proto(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_tracks,
                )
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_tracked_float_proto(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_tracks,
                )
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_tracked_float_proto(
                    tracker,
                    timestamp,
                    &views,
                    output_boxes,
                    output_tracks,
                )
            }
        }
    }
}
