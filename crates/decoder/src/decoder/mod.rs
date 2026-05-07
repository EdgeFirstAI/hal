// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayView, ArrayViewD, Dimension};
use num_traits::{AsPrimitive, Float};

use crate::{DecoderError, DetectBox, ProtoData, Segmentation};

pub mod config;
pub mod configs;

use configs::ModelType;

#[derive(Debug)]
pub struct Decoder {
    model_type: ModelType,
    pub iou_threshold: f32,
    pub score_threshold: f32,
    /// NMS mode: Some(mode) applies NMS, None bypasses NMS (for end-to-end
    /// models)
    pub nms: Option<configs::Nms>,
    /// Maximum number of candidate boxes fed into NMS after score filtering.
    /// Reduces O(N²) NMS cost when many low-confidence proposals pass the
    /// threshold (common during COCO mAP evaluation with threshold ≈ 0.001).
    /// Candidates are ranked by score; only the top `pre_nms_top_k` proceed
    /// to NMS.  Default: 300.  Ignored when `nms` is `None`.
    pub pre_nms_top_k: usize,
    /// Maximum number of detections returned after NMS. Matches the
    /// Ultralytics `max_det` parameter.  Default: 300.
    ///
    /// This bound applies uniformly across all segmentation and detection
    /// decode paths reached via [`Decoder::decode`] / [`Decoder::decode_proto`].
    /// The output `Vec`'s capacity is only an allocation hint; the post-NMS
    /// detection count is bounded solely by `max_det` (EDGEAI-1302). The
    /// `pub fn decode_yolo_*` free convenience wrappers use a separate
    /// constant ([`crate::yolo::DEFAULT_MAX_DETECTIONS`], also 300) and are
    /// not affected by this field.
    pub max_det: usize,
    /// Whether decoded boxes are in normalized [0,1] coordinates.
    /// - `Some(true)`: Coordinates in [0,1] range
    /// - `Some(false)`: Pixel coordinates
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate >
    ///   1.0)
    normalized: Option<bool>,
    /// Model input spatial dimensions `(width, height)`, captured from
    /// the schema's `input.shape` / `input.dshape` at builder time.
    /// Required to honour `normalized: false`: pixel-space box coords
    /// emitted by the model are divided by these dimensions before NMS
    /// so the post-NMS bbox is in `[0, 1]`. `None` when no schema input
    /// spec is available — the legacy >2.0 reject in `protobox` then
    /// preserves the previous safety net (EDGEAI-1303).
    input_dims: Option<(usize, usize)>,
    /// Schema v2 merge program. Present when the decoder was built from
    /// a [`crate::schema::SchemaV2`] whose logical outputs carry
    /// physical children. Absent for flat configurations (v1 and
    /// flat-v2).
    pub(crate) decode_program: Option<merge::DecodeProgram>,
    /// Per-scale fast path. Constructed at build time from a schema-v2
    /// document with per-scale children. Wrapped in `Mutex` because
    /// `Decoder::decode_proto` and `Decoder::decode` are `&self` but
    /// the per-scale buffers are mutated per-frame.
    pub(crate) per_scale: Option<std::sync::Mutex<crate::per_scale::PerScaleDecoder>>,
}

impl PartialEq for Decoder {
    fn eq(&self, other: &Self) -> bool {
        // DecodeProgram and PerScaleDecoder have non-comparable embedded
        // data; compare by the config-derived fields only.
        self.model_type == other.model_type
            && self.iou_threshold == other.iou_threshold
            && self.score_threshold == other.score_threshold
            && self.nms == other.nms
            && self.pre_nms_top_k == other.pre_nms_top_k
            && self.max_det == other.max_det
            && self.normalized == other.normalized
            && self.input_dims == other.input_dims
            && self.decode_program.is_some() == other.decode_program.is_some()
            && self.per_scale.is_some() == other.per_scale.is_some()
    }
}

impl Clone for Decoder {
    /// Cloning a `Decoder` preserves the legacy decode path
    /// (`decode_program`) but drops the per-scale fast path:
    /// `PerScaleDecoder` owns mutable per-frame scratch buffers and is
    /// not `Clone`. Decoders built from a per-scale schema should be
    /// rebuilt via [`DecoderBuilder`] rather than cloned to preserve the
    /// fast path; cloning is intended for tests and rare configs.
    fn clone(&self) -> Self {
        Self {
            model_type: self.model_type.clone(),
            iou_threshold: self.iou_threshold,
            score_threshold: self.score_threshold,
            nms: self.nms,
            pre_nms_top_k: self.pre_nms_top_k,
            max_det: self.max_det,
            normalized: self.normalized,
            input_dims: self.input_dims,
            decode_program: self.decode_program.clone(),
            per_scale: None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum ArrayViewDQuantized<'a> {
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
    pub(crate) fn shape(&self) -> &[usize] {
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
mod dfl;
mod helpers;
mod merge;
mod per_scale_bridge;
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

    /// Model input dimensions `(width, height)` captured from the
    /// schema's `input.shape` / `input.dshape`, or `None` when the
    /// schema did not declare an input spec (e.g. flat YAML configs
    /// or `DecoderBuilder::add_output(...)` programmatic builds).
    ///
    /// Used together with [`normalized_boxes`](Self::normalized_boxes):
    /// when the decoder reports `normalized_boxes() == Some(false)` and
    /// `input_dims()` is `Some((w, h))`, the decoder divides post-NMS
    /// bbox coordinates by `(w, h)` so they enter the canonical `[0, 1]`
    /// range before mask cropping (EDGEAI-1303). When `input_dims()` is
    /// `None`, the decoder cannot perform the division and the existing
    /// `protobox` `> 2.0` reject acts as a safety net.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    ///     let json = r#"{
    ///         "schema_version": 2,
    ///         "nms": "class_agnostic",
    ///         "input": {
    ///             "shape": [1, 640, 640, 3],
    ///             "dshape": [{"batch": 1}, {"height": 640}, {"width": 640}, {"num_features": 3}]
    ///         },
    ///         "outputs": [{
    ///             "name": "out", "type": "detection",
    ///             "shape": [1, 38, 256],
    ///             "dshape": [{"batch": 1}, {"num_features": 38}, {"num_boxes": 256}],
    ///             "decoder": "ultralytics", "encoding": "direct", "normalized": false
    ///         }]
    ///     }"#;
    ///     let schema: SchemaV2 = serde_json::from_str(json).unwrap();
    ///     let decoder = DecoderBuilder::default().with_schema(schema).build()?;
    ///     assert_eq!(decoder.input_dims(), Some((640, 640)));
    /// #   Ok(())
    /// # }
    /// ```
    pub fn input_dims(&self) -> Option<(usize, usize)> {
        self.input_dims
    }

    /// Decode quantized model outputs into detection boxes and segmentation
    /// masks. The quantized outputs can be of u8, i8, u16, i16, u32, or i32
    /// types. Clears the provided output vectors before populating them.
    pub(crate) fn decode_quantized(
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
            ModelType::PerScale => Err(DecoderError::Internal(
                "per-scale path must be intercepted before ModelType dispatch".into(),
            )),
        }
    }

    /// Decode floating point model outputs into detection boxes and
    /// segmentation masks. Clears the provided output vectors before
    /// populating them.
    pub(crate) fn decode_float<T>(
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
            ModelType::PerScale => {
                return Err(DecoderError::Internal(
                    "per-scale path must be intercepted before ModelType dispatch".into(),
                ));
            }
        }
        Ok(())
    }

    /// Decodes quantized model outputs into detection boxes, returning raw
    /// `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (detections
    /// are still decoded into `output_boxes`). Returns `Ok(Some(ProtoData))`
    /// for YOLO segmentation models.
    pub(crate) fn decode_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only variants: decode boxes, return None for proto data.
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_quantized(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_quantized(outputs, detection, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_quantized(outputs, boxes, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_quantized(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloEndToEndDet { boxes } => {
                self.decode_yolo_end_to_end_det_quantized(outputs, boxes, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloSplitEndToEndDet {
                boxes,
                scores,
                classes,
            } => {
                self.decode_yolo_split_end_to_end_det_quantized(
                    outputs,
                    boxes,
                    scores,
                    classes,
                    output_boxes,
                )?;
                Ok(None)
            }
            // ModelPack seg/segdet variants have no YOLO proto data.
            ModelType::ModelPackSegDet { boxes, scores, .. } => {
                self.decode_modelpack_det_quantized(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackSegDetSplit { detection, .. } => {
                self.decode_modelpack_det_split_quantized(outputs, detection, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackSeg { .. } => Ok(None),

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
            ModelType::PerScale => Err(DecoderError::Internal(
                "per-scale path must be intercepted before ModelType dispatch".into(),
            )),
        }
    }

    /// Decodes floating-point model outputs into detection boxes, returning
    /// raw `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (detections
    /// are still decoded into `output_boxes`). Returns `Ok(Some(ProtoData))`
    /// for YOLO segmentation models.
    pub(crate) fn decode_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + crate::yolo::FloatProtoElem,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only variants: decode boxes, return None for proto data.
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_float(outputs, boxes, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_float(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloEndToEndDet { boxes } => {
                self.decode_yolo_end_to_end_det_float(outputs, boxes, output_boxes)?;
                Ok(None)
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
                Ok(None)
            }
            // ModelPack seg/segdet variants have no YOLO proto data.
            ModelType::ModelPackSegDet { boxes, scores, .. } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackSegDetSplit { detection, .. } => {
                self.decode_modelpack_det_split_float(outputs, detection, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackSeg { .. } => Ok(None),

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
            ModelType::PerScale => Err(DecoderError::Internal(
                "per-scale path must be intercepted before ModelType dispatch".into(),
            )),
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
    /// # `output_boxes` / `output_masks` capacity
    ///
    /// The capacity of the supplied `Vec`s is **only** an allocation hint —
    /// it is **not** a cap on the number of detections returned. The
    /// post-NMS detection count is bounded by [`Decoder::max_det`] (set
    /// via [`DecoderBuilder::with_max_det`], default `300`). Passing
    /// `Vec::new()` (capacity 0) returns up to `max_det` detections;
    /// pre-allocating with [`Vec::with_capacity`] only avoids the
    /// reallocation when the decoder grows the buffer.
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
        // Per-scale fast path — selected at builder time when the schema
        // declares per-scale children with DFL or LTRB encoding.
        if let Some(per_scale_mutex) = &self.per_scale {
            let mut ps = per_scale_mutex
                .lock()
                .map_err(|e| DecoderError::Internal(format!("per_scale mutex poisoned: {e}")))?;
            let decoded = ps.run(outputs)?;
            return per_scale_bridge::per_scale_to_masks(
                &decoded,
                output_boxes,
                output_masks,
                self.iou_threshold,
                self.score_threshold,
                self.nms,
                self.pre_nms_top_k,
                self.max_det,
            );
        }

        // Schema v2 merge path: dequantize physical children into
        // logical float32 tensors, then feed through the float dispatch.
        if let Some(program) = &self.decode_program {
            let merged = program.execute(outputs)?;
            let views: Vec<_> = merged.iter().map(|a| a.view()).collect();
            return self.decode_float(&views, output_boxes, output_masks);
        }

        let mapped = tensor_bridge::map_tensors(outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized(&views, output_boxes, output_masks)
            }
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_float(&views, output_boxes, output_masks)
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
    /// Detections are always decoded into `output_boxes` regardless of model type.
    /// Returns `Ok(None)` for detection-only and ModelPack models.
    /// Returns `Ok(Some(ProtoData))` for YOLO segmentation models.
    ///
    /// # Arguments
    ///
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    ///
    /// # `output_boxes` capacity
    ///
    /// The capacity of `output_boxes` is **only** an allocation hint — it
    /// is **not** a cap on the number of detections returned. The
    /// post-NMS detection count is bounded by [`Decoder::max_det`] (set
    /// via [`DecoderBuilder::with_max_det`], default `300`). Passing
    /// `Vec::new()` (capacity 0) returns up to `max_det` detections.
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
        // Per-scale fast path — selected at builder time when the schema
        // declares per-scale children with DFL or LTRB encoding.
        if let Some(per_scale_mutex) = &self.per_scale {
            let mut ps = per_scale_mutex
                .lock()
                .map_err(|e| DecoderError::Internal(format!("per_scale mutex poisoned: {e}")))?;
            let decoded = ps.run(outputs)?;
            return per_scale_bridge::per_scale_to_proto_data(
                &decoded,
                output_boxes,
                self.iou_threshold,
                self.score_threshold,
                self.nms,
                self.pre_nms_top_k,
                self.max_det,
            );
        }

        // Schema v2 merge path: dequantize physical children into
        // logical float32 tensors, then feed through the float dispatch.
        if let Some(program) = &self.decode_program {
            let merged = program.execute(outputs)?;
            let views: Vec<_> = merged.iter().map(|a| a.view()).collect();
            return self.decode_float_proto(&views, output_boxes);
        }

        let mapped = tensor_bridge::map_tensors(outputs)?;
        let result = match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized_proto(&views, output_boxes)
            }
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_float_proto(&views, output_boxes)
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_float_proto(&views, output_boxes)
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_float_proto(&views, output_boxes)
            }
        };
        result
    }

    /// Run the per-scale pipeline and return pre-NMS buffers as owned f32.
    ///
    /// Test-only entry point used by the parity-fixture tests to compare
    /// HAL stage output against the NumPy reference's stage output
    /// without NMS ordering noise. Returns an error if the decoder
    /// isn't configured for per-scale decoding.
    #[doc(hidden)]
    pub fn _testing_run_per_scale_pre_nms(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
    ) -> Result<crate::per_scale::PreNmsCapture, crate::error::DecoderError> {
        let mutex = self.per_scale.as_ref().ok_or_else(|| {
            crate::error::DecoderError::Internal("decoder not configured for per-scale".into())
        })?;
        let mut ps = mutex.lock().map_err(|e| {
            crate::error::DecoderError::Internal(format!("per_scale mutex poisoned: {e}"))
        })?;
        // Drop the borrowed view immediately so we can reborrow buffers below.
        {
            ps.run(outputs)?;
        }
        let total_anchors = ps.plan.total_anchors;
        let num_classes = ps.plan.num_classes;
        let num_mc = ps.plan.num_mask_coefs;
        Ok(ps
            .buffers
            .snapshot_owned_f32(total_anchors, num_classes, num_mc))
    }
}

#[cfg(feature = "tracker")]
pub use edgefirst_tracker::TrackInfo;

#[cfg(feature = "tracker")]
pub use edgefirst_tracker::Tracker;

#[cfg(feature = "tracker")]
impl Decoder {
    /// Decode quantized model outputs into detection boxes and segmentation
    /// masks with tracking. Clears the provided output vectors before
    /// populating them.
    pub(crate) fn decode_tracked_quantized<TR: edgefirst_tracker::Tracker<DetectBox>>(
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
    pub(crate) fn decode_tracked_float<TR: edgefirst_tracker::Tracker<DetectBox>, T>(
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
    pub(crate) fn decode_tracked_quantized_proto<TR: edgefirst_tracker::Tracker<DetectBox>>(
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
            // Non-seg variants: decode boxes via the non-proto path, then track.
            _ => {
                let mut masks = Vec::new();
                self.decode_quantized(outputs, output_boxes, &mut masks)?;
                Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
                Ok(None)
            }
        }
    }

    /// Decodes floating-point model outputs into detection boxes, returning
    /// raw `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Detections are always decoded into `output_boxes` regardless of model type.
    /// Returns `Ok(None)` for detection-only and ModelPack models. Returns
    /// `Ok(Some(ProtoData))` for YOLO segmentation models.
    pub(crate) fn decode_tracked_float_proto<TR: edgefirst_tracker::Tracker<DetectBox>, T>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + crate::yolo::FloatProtoElem,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        output_tracks.clear();
        match &self.model_type {
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
            // Non-seg variants: decode boxes via the non-proto path, then track.
            _ => {
                let mut masks = Vec::new();
                self.decode_float(outputs, output_boxes, &mut masks)?;
                Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
                Ok(None)
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
        // Per-scale fast path: route via the basic decode then update the
        // tracker. Phase 1 keeps the tracker integration simple; per-frame
        // decoupling between detection and tracking is preserved.
        if self.per_scale.is_some() {
            output_tracks.clear();
            self.decode(outputs, output_boxes, output_masks)?;
            Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
            return Ok(());
        }

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
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_tracked_float(
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
        // Per-scale fast path: route via the basic decode_proto then
        // update the tracker on the resulting boxes.
        if self.per_scale.is_some() {
            output_tracks.clear();
            let proto = self.decode_proto(outputs, output_boxes)?;
            Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
            return Ok(proto);
        }

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
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_tracked_float_proto(
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
