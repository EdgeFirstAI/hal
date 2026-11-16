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
    /// NMS mode, always a concrete variant after build:
    /// - `Some(ClassAgnostic)` — class-agnostic NMS
    /// - `Some(ClassAware)` — class-aware NMS
    /// - `None` — NMS bypassed (end-to-end models)
    ///
    /// `Nms::Auto` is resolved by
    /// [`DecoderBuilder::build`](crate::DecoderBuilder::build). When
    /// `Some(Nms::Auto)` is assigned to this field, debug builds panic at
    /// decode and release builds treat it as class-aware.
    pub nms: Option<configs::Nms>,
    /// Maximum number of candidate boxes fed into NMS after score filtering.
    /// Candidates are ranked by score, equal scores by anchor index; only the
    /// top `pre_nms_top_k` proceed to NMS. Applies to every NMS decode path;
    /// ignored when `nms` is `None`.
    ///
    /// - `None` (the default unless the builder was given a cap): resolved
    ///   per decode by [`Decoder::pre_nms_top_k_for`] —
    ///   [`MULTI_LABEL_PRE_NMS_TOP_K`] (30 000) for a multi-label decode,
    ///   [`DEFAULT_PRE_NMS_TOP_K`] (300) for an argmax decode, including the
    ///   tracking entry points of a multi-label decoder.
    /// - `Some(n)`: an explicit cap used by every decode as given; `Some(0)`
    ///   is unbounded.
    pub pre_nms_top_k: Option<usize>,
    /// Maximum number of detections returned after NMS. Default: 300.
    ///
    /// Applies to every decode path reached via [`Decoder::decode`] /
    /// [`Decoder::decode_proto`]. The output `Vec`'s capacity does not bound
    /// the detection count.
    pub max_det: usize,
    /// Whether decoded boxes are in normalized [0,1] coordinates.
    /// - `Some(true)`: Coordinates in [0,1] range
    /// - `Some(false)`: Pixel coordinates
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate >
    ///   1.0)
    normalized: Option<bool>,
    /// Model input spatial dimensions `(width, height)`, captured from
    /// the schema's `input.shape` / `input.dshape` at builder time.
    /// With `normalized: false`, pixel-space box coords are divided by
    /// these dimensions before NMS. `None` when no schema input spec is
    /// available; `protobox` then rejects coordinates above 2.0.
    input_dims: Option<(usize, usize)>,
    /// Schema v2 merge program. Present when the decoder was built from
    /// a [`crate::schema::SchemaV2`] whose logical outputs carry
    /// physical children. Absent for flat configurations (v1 and
    /// flat-v2).
    pub(crate) decode_program: Option<merge::DecodeProgram>,
    /// How roles find their tensors when the model's outputs share shapes;
    /// `None` when lookup by shape is exact. Flat configurations only.
    binding: Option<binding::OutputBinding>,
    /// The order `decode_program`'s logical outputs are handed to the
    /// decode functions in, when it is not the schema's.
    merged_order: Option<Vec<usize>>,
    /// When `true`, emit one detection per (anchor, class) pair for every
    /// class whose score meets the threshold. Off by default.
    /// `Decoder::decode_tracked` ignores this and decodes one label per box.
    pub(crate) multi_label: bool,
    /// Where `multi_label` came from; named in the tracked-decode warning.
    pub(crate) multi_label_source: MultiLabelSource,
    /// Set once this decoder has logged that tracked decode ignores
    /// multi-label.
    tracked_multi_label_warned: std::sync::atomic::AtomicBool,
    /// Per-scale path. Constructed at build time from a schema-v2 document
    /// with per-scale children. The `Mutex` guards the per-frame buffers,
    /// which the `&self` decode calls mutate.
    pub(crate) per_scale: Option<std::sync::Mutex<crate::per_scale::PerScaleDecoder>>,
    /// Semantic segmentation output: a `decoder: semantic` output, or the
    /// segmentation output of a ModelPack model.
    pub(crate) semantic: Option<crate::semantic::SemanticConfig>,
}

/// Where a decoder's multi-label setting came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MultiLabelSource {
    /// Neither the caller nor the model config set it.
    #[default]
    Default,
    /// Declared by the model config's `nms_multi_label`.
    Metadata,
    /// Set by the caller through [`DecoderBuilder::with_multi_label`].
    Explicit,
}

fn semantic_only() -> DecoderError {
    DecoderError::NotSupported("semantic segmentation models are decoded by decode_semantic".into())
}

impl PartialEq for Decoder {
    fn eq(&self, other: &Self) -> bool {
        // Compares the config-derived fields only.
        self.model_type == other.model_type
            && self.iou_threshold == other.iou_threshold
            && self.score_threshold == other.score_threshold
            && self.nms == other.nms
            && self.pre_nms_top_k == other.pre_nms_top_k
            && self.max_det == other.max_det
            && self.normalized == other.normalized
            && self.input_dims == other.input_dims
            && self.multi_label == other.multi_label
            && self.decode_program.is_some() == other.decode_program.is_some()
            && self.binding == other.binding
            && self.merged_order == other.merged_order
            && self.per_scale.is_some() == other.per_scale.is_some()
            && self.semantic == other.semantic
    }
}

impl Clone for Decoder {
    /// The clone keeps `decode_program` and has no per-scale path
    /// (`per_scale` is `None`).
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
            multi_label: self.multi_label,
            multi_label_source: self.multi_label_source,
            tracked_multi_label_warned: std::sync::atomic::AtomicBool::new(false),
            decode_program: self.decode_program.clone(),
            binding: self.binding.clone(),
            merged_order: self.merged_order.clone(),
            per_scale: None,
            semantic: self.semantic.clone(),
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

/// Binds `$var` to the typed array of an [`ArrayViewDQuantized`] and expands
/// `$body` once per integer variant.
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

mod binding;
mod builder;
mod helpers;
mod merge;
mod per_scale_bridge;
mod postprocess;
pub(crate) mod tensor_bridge;
mod tests;

pub use builder::{DecoderBuilder, DEFAULT_PRE_NMS_TOP_K, MULTI_LABEL_PRE_NMS_TOP_K};
pub use config::{ConfigOutput, ConfigOutputRef, ConfigOutputs};

impl Decoder {
    /// Static label naming the dispatch path `decode` / `decode_proto` take:
    /// `per_scale`, `decode_program` (schema-v2 merge) or `legacy`
    /// (config-driven). Used as a tracing-span attribute.
    fn decode_path_label(&self) -> &'static str {
        if self.per_scale.is_some() {
            "per_scale"
        } else if self.decode_program.is_some() {
            "decode_program"
        } else {
            "legacy"
        }
    }

    /// Whether [`Decoder::decode`] and [`Decoder::decode_proto`] emit one box
    /// per class above the score threshold instead of one per anchor.
    ///
    /// [`Self::decode_for_tracking`], and `Decoder::decode_tracked` /
    /// `Decoder::decode_proto_tracked` (behind the `tracker` feature), ignore
    /// this and always use one label per box.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = edgefirst_bench::testdata::read_to_string("modelpack_split.yaml").to_string();
    ///     let decoder = DecoderBuilder::default()
    ///         .with_config_yaml_str(config_yaml)
    ///         .with_multi_label(true)
    ///         .build()?;
    ///     assert!(decoder.multi_label());
    /// #    Ok(())
    /// # }
    /// ```
    pub fn multi_label(&self) -> bool {
        self.multi_label
    }

    /// The pre-NMS cap a decode uses: the explicit
    /// [`pre_nms_top_k`](Self::pre_nms_top_k) when set, else
    /// [`MULTI_LABEL_PRE_NMS_TOP_K`] for a multi-label decode and
    /// [`DEFAULT_PRE_NMS_TOP_K`] for an argmax one. `0` means unbounded.
    ///
    /// Pass [`multi_label()`](Self::multi_label) for what `decode` /
    /// `decode_proto` use, and `false` for the tracking entry points, which
    /// always decode argmax.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// # let config_json = edgefirst_bench::testdata::read_to_string("modelpack_split.json").to_string();
    /// let mut decoder = DecoderBuilder::new()
    ///     .with_config_json_str(config_json)
    ///     .with_multi_label(true)
    ///     .build()?;
    /// assert_eq!(decoder.pre_nms_top_k, None);
    /// assert_eq!(decoder.pre_nms_top_k_for(true), 30_000);
    /// assert_eq!(decoder.pre_nms_top_k_for(false), 300);
    /// decoder.pre_nms_top_k = Some(30_000);
    /// assert_eq!(decoder.pre_nms_top_k_for(false), 30_000);
    /// # Ok(())
    /// # }
    /// ```
    pub fn pre_nms_top_k_for(&self, multi_label: bool) -> usize {
        match self.pre_nms_top_k {
            Some(k) => k,
            None if multi_label => MULTI_LABEL_PRE_NMS_TOP_K,
            None => DEFAULT_PRE_NMS_TOP_K,
        }
    }

    /// Log, the first time this decoder decodes for tracking with
    /// multi-label enabled, that tracking ignores it. Returns whether this
    /// call logged.
    pub(crate) fn warn_once_tracked_multi_label(&self) -> bool {
        if !self.multi_label
            || self
                .tracked_multi_label_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return false;
        }
        tracing::warn!(
            source = ?self.multi_label_source,
            "multi_label is ignored when decoding for tracking; one label per box"
        );
        true
    }

    /// This function returns the parsed model type of the decoder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult, configs::ModelType};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = edgefirst_bench::testdata::read_to_string("modelpack_split.yaml").to_string();
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

    /// Returns the coordinate format of the boxes the decoder emits to
    /// the caller.
    ///
    /// - `Some(true)`: Boxes are in normalized `[0, 1]` coordinates
    /// - `Some(false)`: Boxes are in pixel coordinates relative to the
    ///   model input
    /// - `None`: Unknown, caller must infer (e.g., check if any coordinate
    ///   > 1.0)
    ///
    /// This describes the post-decode coordinate space. The per-scale path,
    /// [`ModelType::YoloSegDet`](crate::configs::ModelType::YoloSegDet),
    /// [`ModelType::YoloSplitSegDet`](crate::configs::ModelType::YoloSplitSegDet)
    /// and [`ModelType::YoloSegDet2Way`](crate::configs::ModelType::YoloSegDet2Way)
    /// divide box coordinates by [`input_dims`](Self::input_dims) on every
    /// entry point; for those, a schema that declares `normalized: false`
    /// with valid `input_dims` reports `Some(true)`.
    ///
    /// Every other model type returns the schema's flag unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use edgefirst_decoder::{DecoderBuilder, DecoderResult};
    /// # fn main() -> DecoderResult<()> {
    /// #    let config_yaml = edgefirst_bench::testdata::read_to_string("modelpack_split.yaml").to_string();
    ///     let decoder = DecoderBuilder::default()
    ///         .with_config_yaml_str(config_yaml)
    ///         .build()?;
    ///     // Config doesn't specify normalized, so it's None
    ///     assert!(decoder.normalized_boxes().is_none());
    /// #    Ok(())
    /// # }
    /// ```
    pub fn normalized_boxes(&self) -> Option<bool> {
        // The per-scale path and the model types that normalize on every entry
        // point report `Some(true)` for `normalized == Some(false)` with valid
        // `input_dims`. The other paths return the schema flag.
        if self.per_scale.is_some() || self.legacy_path_normalizes_uniformly() {
            match (self.normalized, self.input_dims) {
                (Some(true), _) => Some(true),
                (Some(false), Some((w, h))) if w != 0 && h != 0 => Some(true),
                (Some(false), _) => Some(false),
                (None, _) => None,
            }
        } else {
            self.normalized
        }
    }

    /// Returns true for the `ModelType` dispatch paths that call
    /// `yolo::maybe_normalize_boxes_in_place` on every entry point
    /// (`decode`, `decode_proto`, `decode_tracked`, `decode_tracked_proto`,
    /// both quantized and float variants).
    fn legacy_path_normalizes_uniformly(&self) -> bool {
        matches!(
            self.model_type,
            ModelType::YoloSegDet { .. }
                | ModelType::YoloSplitSegDet { .. }
                | ModelType::YoloSegDet2Way { .. }
        )
    }

    /// Model input dimensions `(width, height)` captured from the
    /// schema's `input.shape` / `input.dshape`, or `None` when the
    /// schema did not declare an input spec (e.g. flat YAML configs
    /// or `DecoderBuilder::add_output(...)` programmatic builds).
    ///
    /// When the schema declares pixel-space outputs and `input_dims()` is
    /// `Some((w, h))`, the per-scale path and the `ModelType::YoloSegDet`,
    /// `ModelType::YoloSplitSegDet` and `ModelType::YoloSegDet2Way` paths
    /// divide post-NMS box coordinates by `(w, h)`. The other model types
    /// do not; see [`normalized_boxes`](Self::normalized_boxes).
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

    /// The dtype the per-scale pipeline decodes boxes and scores in: the one
    /// given to [`DecoderBuilder::with_decode_dtype`], else
    /// [`DecodeDtype::native`](crate::DecodeDtype::native). `None` for a
    /// model that does not take the per-scale path.
    pub fn decode_dtype(&self) -> Option<crate::DecodeDtype> {
        let per_scale = self.per_scale.as_ref()?;
        let per_scale = per_scale.lock().unwrap_or_else(|e| e.into_inner());
        Some(per_scale.plan.out_dtype)
    }

    /// Decode quantized model outputs into detection boxes and segmentation
    /// masks. The quantized outputs can be of u8, i8, u16, i16, u32, or i32
    /// types. Clears the provided output vectors before populating them.
    ///
    /// Test-only: calls [`Self::decode_quantized_impl`] with the decoder's
    /// own multi-label setting.
    #[cfg(test)]
    pub(crate) fn decode_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        self.decode_quantized_impl(outputs, output_boxes, output_masks, self.multi_label)
    }

    /// Dispatch with an explicit multi-label setting; tracked decode passes `false`.
    pub(crate) fn decode_quantized_impl(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        multi_label: bool,
    ) -> Result<(), DecoderError> {
        output_boxes.clear();
        output_masks.clear();
        match &self.model_type {
            ModelType::ModelPackSegDet {
                boxes,
                scores,
                segmentation,
            } => {
                let detected = self.decode_modelpack_det_quantized(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
                self.decode_modelpack_seg_quantized(outputs, segmentation, &detected, output_masks)
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                let detected = self.decode_modelpack_det_split_quantized(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                self.decode_modelpack_seg_quantized(outputs, segmentation, &detected, output_masks)
            }
            ModelType::ModelPackDet { boxes, scores } => self
                .decode_modelpack_det_quantized(outputs, boxes, scores, multi_label, output_boxes)
                .map(drop),
            ModelType::ModelPackDetSplit { detection } => self
                .decode_modelpack_det_split_quantized(outputs, detection, multi_label, output_boxes)
                .map(drop),
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_quantized(outputs, segmentation, &[], output_masks)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_quantized(outputs, boxes, multi_label, output_boxes)
            }
            ModelType::YoloSegDet { boxes, protos } => self.decode_yolo_segdet_quantized(
                outputs,
                boxes,
                protos,
                multi_label,
                output_boxes,
                output_masks,
            ),
            ModelType::YoloSplitDet { boxes, scores } => self.decode_yolo_split_det_quantized(
                outputs,
                boxes,
                scores,
                multi_label,
                output_boxes,
            ),
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
                multi_label,
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
                multi_label,
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
            ModelType::Semantic => Err(semantic_only()),
        }
    }

    /// Decode floating point model outputs into detection boxes and
    /// segmentation masks. Clears the provided output vectors before
    /// populating them.
    ///
    /// Test-only: calls [`Self::decode_float_impl`] with the decoder's own
    /// multi-label setting.
    #[cfg(test)]
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
        self.decode_float_impl(outputs, output_boxes, output_masks, self.multi_label)
    }

    /// Dispatch with an explicit multi-label setting; tracked decode passes `false`.
    pub(crate) fn decode_float_impl<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        multi_label: bool,
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
                let detected = self.decode_modelpack_det_float(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
                self.decode_modelpack_seg_float(outputs, segmentation, &detected, output_masks)?;
            }
            ModelType::ModelPackSegDetSplit {
                detection,
                segmentation,
            } => {
                let detected = self.decode_modelpack_det_split_float(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                self.decode_modelpack_seg_float(outputs, segmentation, &detected, output_masks)?;
            }
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, multi_label, output_boxes)?;
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
            }
            ModelType::ModelPackSeg { segmentation } => {
                self.decode_modelpack_seg_float(outputs, segmentation, &[], output_masks)?;
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_float(outputs, boxes, multi_label, output_boxes)?;
            }
            ModelType::YoloSegDet { boxes, protos } => {
                self.decode_yolo_segdet_float(
                    outputs,
                    boxes,
                    protos,
                    multi_label,
                    output_boxes,
                    output_masks,
                )?;
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_float(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
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
                    multi_label,
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
                    multi_label,
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
            ModelType::Semantic => return Err(semantic_only()),
        }
        Ok(())
    }

    /// Decodes quantized model outputs into detection boxes, returning raw
    /// `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (detections
    /// are still decoded into `output_boxes`). Returns `Ok(Some(ProtoData))`
    /// for YOLO segmentation models.
    ///
    /// Test-only: calls [`Self::decode_quantized_proto_impl`] with the
    /// decoder's own multi-label setting.
    #[cfg(test)]
    pub(crate) fn decode_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError> {
        self.decode_quantized_proto_impl(outputs, output_boxes, self.multi_label)
    }

    /// Dispatch with an explicit multi-label setting; tracked decode passes `false`.
    pub(crate) fn decode_quantized_proto_impl(
        &self,
        outputs: &[ArrayViewDQuantized],
        output_boxes: &mut Vec<DetectBox>,
        multi_label: bool,
    ) -> Result<Option<ProtoData>, DecoderError> {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only variants: decode boxes, return None for proto data.
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_quantized(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_quantized(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_quantized(outputs, boxes, multi_label, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_quantized(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
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
                self.decode_modelpack_det_quantized(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::ModelPackSegDetSplit { detection, .. } => {
                self.decode_modelpack_det_split_quantized(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::ModelPackSeg { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto = self.decode_yolo_segdet_quantized_proto(
                    outputs,
                    boxes,
                    protos,
                    multi_label,
                    output_boxes,
                )?;
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
                    multi_label,
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
                    multi_label,
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
            ModelType::Semantic => Err(semantic_only()),
        }
    }

    /// Decodes floating-point model outputs into detection boxes, returning
    /// raw `ProtoData` for segmentation models instead of materialized masks.
    ///
    /// Returns `Ok(None)` for detection-only and ModelPack models (detections
    /// are still decoded into `output_boxes`). Returns `Ok(Some(ProtoData))`
    /// for YOLO segmentation models.
    ///
    /// Test-only: calls [`Self::decode_float_proto_impl`] with the decoder's
    /// own multi-label setting.
    #[cfg(test)]
    pub(crate) fn decode_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + crate::yolo::FloatProtoElem,
        f32: AsPrimitive<T>,
    {
        self.decode_float_proto_impl(outputs, output_boxes, self.multi_label)
    }

    /// Dispatch with an explicit multi-label setting; tracked decode passes `false`.
    pub(crate) fn decode_float_proto_impl<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        output_boxes: &mut Vec<DetectBox>,
        multi_label: bool,
    ) -> Result<Option<ProtoData>, DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + crate::yolo::FloatProtoElem,
        f32: AsPrimitive<T>,
    {
        output_boxes.clear();
        match &self.model_type {
            // Detection-only variants: decode boxes, return None for proto data.
            ModelType::ModelPackDet { boxes, scores } => {
                self.decode_modelpack_det_float(outputs, boxes, scores, multi_label, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackDetSplit { detection } => {
                self.decode_modelpack_det_split_float(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::YoloDet { boxes } => {
                self.decode_yolo_det_float(outputs, boxes, multi_label, output_boxes)?;
                Ok(None)
            }
            ModelType::YoloSplitDet { boxes, scores } => {
                self.decode_yolo_split_det_float(
                    outputs,
                    boxes,
                    scores,
                    multi_label,
                    output_boxes,
                )?;
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
                self.decode_modelpack_det_float(outputs, boxes, scores, multi_label, output_boxes)?;
                Ok(None)
            }
            ModelType::ModelPackSegDetSplit { detection, .. } => {
                self.decode_modelpack_det_split_float(
                    outputs,
                    detection,
                    multi_label,
                    output_boxes,
                )?;
                Ok(None)
            }
            ModelType::ModelPackSeg { .. } => Ok(None),

            ModelType::YoloSegDet { boxes, protos } => {
                let proto = self.decode_yolo_segdet_float_proto(
                    outputs,
                    boxes,
                    protos,
                    multi_label,
                    output_boxes,
                )?;
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
                    multi_label,
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
                    multi_label,
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
            ModelType::Semantic => Err(semantic_only()),
        }
    }

    // ========================================================================
    // TensorDyn-based public API
    // ========================================================================

    /// Decode model outputs into detection boxes and segmentation masks.
    ///
    /// Accepts `TensorDyn` outputs directly from model inference and
    /// dispatches to quantized or float paths based on the tensor dtype.
    ///
    /// # Arguments
    ///
    /// * `outputs` - Tensor outputs from model inference
    /// * `output_boxes` - Destination for decoded detection boxes (cleared first)
    /// * `output_masks` - Destination for decoded segmentation masks (cleared first)
    ///
    /// # `output_boxes` / `output_masks` capacity
    ///
    /// The capacity of the supplied `Vec`s is an allocation hint and does not
    /// cap the number of detections returned. The post-NMS detection count is
    /// bounded by [`Decoder::max_det`] (set via
    /// [`DecoderBuilder::with_max_det`], default `300`).
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
        self.decode_impl(outputs, output_boxes, output_masks, self.multi_label)
    }

    /// Decode with one label per box, whatever [`Self::multi_label`] says,
    /// for boxes that feed an external tracker.
    ///
    /// Logs a warning once per decoder when the decoder has multi-label
    /// enabled. `Decoder::decode_tracked` and `Decoder::decode_proto_tracked`
    /// (behind the `tracker` feature) apply the same rule. Uses
    /// [`DEFAULT_PRE_NMS_TOP_K`] unless [`Self::pre_nms_top_k`] holds an
    /// explicit cap.
    ///
    /// # Errors
    ///
    /// The same as [`Self::decode`].
    pub fn decode_for_tracking(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        self.warn_once_tracked_multi_label();
        self.decode_impl(outputs, output_boxes, output_masks, false)
    }

    /// The model's semantic segmentation output: a `decoder: semantic`
    /// output, or the segmentation output of a ModelPack model. `None` when
    /// the model has neither.
    pub fn semantic(&self) -> Option<&crate::semantic::SemanticConfig> {
        self.semantic.as_ref()
    }

    /// Decodes the model's semantic segmentation output into `dst`, an
    /// `[H, W]` class map of dtype `u8` (up to 256 classes) or `u16`, at any
    /// resolution.
    ///
    /// `confidence`, when given, receives the winning class's probability as
    /// an `[H, W]` map of `f32` or `u8` (scaled to 255); it is approximate.
    /// See [`crate::semantic`] for the decode modes and
    /// [`crate::semantic::SemanticOptions`] for the letterbox and the
    /// runtime mode override.
    ///
    /// # Errors
    ///
    /// [`DecoderError::NotSupported`] if the model has no semantic output;
    /// otherwise the errors of [`crate::semantic::decode_semantic`].
    pub fn decode_semantic(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        dst: &mut edgefirst_tensor::TensorDyn,
        confidence: Option<&mut edgefirst_tensor::TensorDyn>,
        options: &crate::semantic::SemanticOptions,
    ) -> Result<(), DecoderError> {
        let config = self.semantic.as_ref().ok_or_else(|| {
            DecoderError::NotSupported("the model has no semantic segmentation output".into())
        })?;
        let _span = tracing::trace_span!("decoder.decode_semantic").entered();
        let input = crate::semantic::select_output(outputs, config)?;
        crate::semantic::decode_semantic(input, config, dst, confidence, options)
    }

    /// `outputs` as the decode functions look them up: each role's tensor,
    /// in lookup order, when the model's outputs share shapes; as passed
    /// otherwise.
    fn bound<'a, 'b>(
        &self,
        outputs: &'b [&'a edgefirst_tensor::TensorDyn],
    ) -> Result<std::borrow::Cow<'b, [&'a edgefirst_tensor::TensorDyn]>, DecoderError> {
        Ok(match &self.binding {
            Some(binding) => binding.bind(outputs)?.into(),
            None => outputs.into(),
        })
    }

    /// Views of the merge program's logical outputs, in lookup order.
    fn merged_views<'a>(&self, merged: &'a [ndarray::ArrayD<f32>]) -> Vec<ArrayViewD<'a, f32>> {
        match &self.merged_order {
            Some(order) => order.iter().map(|&index| merged[index].view()).collect(),
            None => merged.iter().map(|a| a.view()).collect(),
        }
    }

    /// [`Self::decode`] with an explicit multi-label setting.
    fn decode_impl(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        multi_label: bool,
    ) -> Result<(), DecoderError> {
        let path = self.decode_path_label();
        let _span = tracing::trace_span!("decoder.decode", path = path, n_outputs = outputs.len())
            .entered();
        // Per-scale path.
        if let Some(per_scale_mutex) = &self.per_scale {
            let mut ps = per_scale_mutex
                .lock()
                .map_err(|e| DecoderError::Internal(format!("per_scale mutex poisoned: {e}")))?;
            let decoded = ps.run(outputs, Some(self.score_threshold))?;
            return per_scale_bridge::per_scale_to_masks(
                &decoded,
                output_boxes,
                output_masks,
                self.iou_threshold,
                self.score_threshold,
                self.nms,
                self.pre_nms_top_k_for(multi_label),
                self.max_det,
                self.normalized,
                self.input_dims,
                multi_label,
            );
        }

        // Schema v2 merge path: dequantize physical children into
        // logical float32 tensors, then feed through the float dispatch.
        if let Some(program) = &self.decode_program {
            let merged = program.execute(outputs)?;
            let views = self.merged_views(&merged);
            return self.decode_float_impl(&views, output_boxes, output_masks, multi_label);
        }

        let outputs = self.bound(outputs)?;
        let mapped = tensor_bridge::map_tensors(&outputs)?;
        match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized_impl(&views, output_boxes, output_masks, multi_label)
            }
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_float_impl(&views, output_boxes, output_masks, multi_label)
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_float_impl(&views, output_boxes, output_masks, multi_label)
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_float_impl(&views, output_boxes, output_masks, multi_label)
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
    /// The capacity of `output_boxes` is an allocation hint and does not cap
    /// the number of detections returned. The post-NMS detection count is
    /// bounded by [`Decoder::max_det`] (set via
    /// [`DecoderBuilder::with_max_det`], default `300`).
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
        self.decode_proto_impl(outputs, output_boxes, self.multi_label)
    }

    /// [`Self::decode_proto`] with an explicit multi-label setting.
    fn decode_proto_impl(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        output_boxes: &mut Vec<DetectBox>,
        multi_label: bool,
    ) -> Result<Option<ProtoData>, DecoderError> {
        let path = self.decode_path_label();
        let _span = tracing::trace_span!(
            "decoder.decode_proto",
            path = path,
            n_outputs = outputs.len()
        )
        .entered();
        // Per-scale path.
        if let Some(per_scale_mutex) = &self.per_scale {
            let mut ps = per_scale_mutex
                .lock()
                .map_err(|e| DecoderError::Internal(format!("per_scale mutex poisoned: {e}")))?;
            let decoded = ps.run(outputs, Some(self.score_threshold))?;
            return per_scale_bridge::per_scale_to_proto_data(
                &decoded,
                output_boxes,
                self.iou_threshold,
                self.score_threshold,
                self.nms,
                self.pre_nms_top_k_for(multi_label),
                self.max_det,
                self.normalized,
                self.input_dims,
                multi_label,
            );
        }

        // Schema v2 merge path: dequantize physical children into
        // logical float32 tensors, then feed through the float dispatch.
        if let Some(program) = &self.decode_program {
            let merged = program.execute(outputs)?;
            let views = self.merged_views(&merged);
            return self.decode_float_proto_impl(&views, output_boxes, multi_label);
        }

        let outputs = self.bound(outputs)?;
        let mapped = tensor_bridge::map_tensors(&outputs)?;
        let result = match &mapped {
            tensor_bridge::MappedOutputs::Quantized(maps) => {
                let views = tensor_bridge::quantized_views(maps)?;
                self.decode_quantized_proto_impl(&views, output_boxes, multi_label)
            }
            tensor_bridge::MappedOutputs::Float16(maps) => {
                let views = tensor_bridge::f16_views(maps)?;
                self.decode_float_proto_impl(&views, output_boxes, multi_label)
            }
            tensor_bridge::MappedOutputs::Float32(maps) => {
                let views = tensor_bridge::f32_views(maps)?;
                self.decode_float_proto_impl(&views, output_boxes, multi_label)
            }
            tensor_bridge::MappedOutputs::Float64(maps) => {
                let views = tensor_bridge::f64_views(maps)?;
                self.decode_float_proto_impl(&views, output_boxes, multi_label)
            }
        };
        result
    }

    /// Run only the per-scale pipeline (no candidate filter, no NMS) and
    /// return the number of anchors it decoded.
    ///
    /// Benchmark entry point: `Some(threshold)` runs the score prefilter
    /// as [`Decoder::decode`] does, `None` decodes every anchor.
    #[doc(hidden)]
    pub fn _bench_run_per_scale(
        &self,
        outputs: &[&edgefirst_tensor::TensorDyn],
        score_threshold: Option<f32>,
    ) -> Result<usize, crate::error::DecoderError> {
        let mutex = self.per_scale.as_ref().ok_or_else(|| {
            crate::error::DecoderError::Internal("decoder not configured for per-scale".into())
        })?;
        let mut ps = mutex.lock().map_err(|e| {
            crate::error::DecoderError::Internal(format!("per_scale mutex poisoned: {e}"))
        })?;
        Ok(ps.run(outputs, score_threshold)?.total_anchors)
    }

    /// Run the per-scale pipeline and return pre-NMS buffers as owned f32.
    ///
    /// Test-only entry point. Returns an error if the decoder isn't
    /// configured for per-scale decoding.
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
        {
            ps.run(outputs, None)?;
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

        // The yolo segdet variants keep the boxes that come from decoding apart
        // from active tracks; only decoded boxes get protos and masks.
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
                self.decode_quantized_impl(outputs, output_boxes, output_masks, false)?;
                Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
                Ok(())
            }
        }
    }

    /// This function decodes floating point model outputs into detection boxes
    /// and segmentation masks. Up to `max_det` boxes and
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
                self.decode_float_impl(outputs, output_boxes, output_masks, false)?;
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
                self.decode_quantized_impl(outputs, output_boxes, &mut masks, false)?;
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
                self.decode_float_impl(outputs, output_boxes, &mut masks, false)?;
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
    /// Always decodes one label per box, regardless of [`Self::multi_label`].
    /// Logs a warning once per decoder when the decoder has multi-label
    /// enabled. Like [`Self::decode_for_tracking`], it uses the argmax
    /// default pre-NMS cap unless the cap was set explicitly.
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
        self.warn_once_tracked_multi_label();
        // Per-scale path: decodes, then updates the tracker.
        if self.per_scale.is_some() {
            output_tracks.clear();
            self.decode_impl(outputs, output_boxes, output_masks, false)?;
            Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
            return Ok(());
        }

        let outputs = self.bound(outputs)?;
        let mapped = tensor_bridge::map_tensors(&outputs)?;
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
    /// Always decodes one label per box, regardless of [`Self::multi_label`].
    /// Logs a warning once per decoder when the decoder has multi-label
    /// enabled. Like [`Self::decode_for_tracking`], it uses the argmax
    /// default pre-NMS cap unless the cap was set explicitly.
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
        self.warn_once_tracked_multi_label();
        // Per-scale path: runs `decode_proto`, then updates the tracker on the
        // resulting boxes.
        if self.per_scale.is_some() {
            output_tracks.clear();
            let proto = self.decode_proto_impl(outputs, output_boxes, false)?;
            Self::update_tracker(tracker, timestamp, output_boxes, output_tracks);
            return Ok(proto);
        }

        let outputs = self.bound(outputs)?;
        let mapped = tensor_bridge::map_tensors(&outputs)?;
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
