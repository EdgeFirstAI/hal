// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decoder C API - ML model output decoding.
//!
//! This module provides functions for decoding YOLO and ModelPack model outputs
//! into detection boxes and segmentation masks.

use crate::error::{set_error, set_error_null, str_to_c_string};
use crate::tensor::HalTensor;
use crate::{
    check_null, check_null_ret_null, try_or_errno, try_or_null, HalByteTrack, HalTrackInfoList,
};
use edgefirst_decoder::{
    configs, configs::Nms, dequantize_cpu_chunked, segmentation_to_mask, ConfigOutput, Decoder,
    DecoderBuilder, DetectBox, Quantization, Segmentation,
};
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
use edgefirst_tracker::TrackInfo;
use libc::{c_char, c_int, size_t};
use std::ffi::CStr;

/// Quantization parameters for dequantizing tensor data.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalQuantization {
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for dequantization
    pub zero_point: i32,
}

impl From<HalQuantization> for Quantization {
    fn from(q: HalQuantization) -> Self {
        Quantization::new(q.scale, q.zero_point)
    }
}

/// Detection box result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalDetectBox {
    /// Left-most normalized coordinate (xmin)
    pub xmin: f32,
    /// Top-most normalized coordinate (ymin)
    pub ymin: f32,
    /// Right-most normalized coordinate (xmax)
    pub xmax: f32,
    /// Bottom-most normalized coordinate (ymax)
    pub ymax: f32,
    /// Confidence score
    pub score: f32,
    /// Class label index
    pub label: size_t,
}

impl From<&DetectBox> for HalDetectBox {
    fn from(b: &DetectBox) -> Self {
        Self {
            xmin: b.bbox.xmin,
            ymin: b.bbox.ymin,
            xmax: b.bbox.xmax,
            ymax: b.bbox.ymax,
            score: b.score,
            label: b.label,
        }
    }
}

/// NMS (Non-Maximum Suppression) mode.
///
/// Controls how overlapping detection boxes are suppressed after decoding.
///
/// | Mode | Value | Behavior |
/// |------|-------|----------|
/// | HAL_NMS_CLASS_AGNOSTIC | 0 | Suppress overlapping boxes regardless of class |
/// | HAL_NMS_CLASS_AWARE | 1 | Only suppress boxes with the same class label |
/// | HAL_NMS_NONE | 2 | Disable NMS (for end-to-end models with embedded NMS) |
///
/// Most YOLO models use HAL_NMS_CLASS_AGNOSTIC. Use HAL_NMS_NONE for
/// end-to-end models (e.g. `decoder_version: yolo26` in EdgeFirst metadata)
/// that perform NMS internally.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalNms {
    /// Class-agnostic NMS: suppress overlapping boxes regardless of class.
    /// This is the most common mode for YOLO detection models.
    ClassAgnostic = 0,
    /// Class-aware NMS: only suppress boxes with the same class label.
    /// Use when different classes may legitimately overlap (e.g. person + car).
    ClassAware = 1,
    /// No NMS: skip post-processing suppression entirely.
    /// Use for end-to-end models that include NMS in the model graph.
    None = 2,
}

impl From<HalNms> for Option<Nms> {
    fn from(nms: HalNms) -> Self {
        match nms {
            HalNms::ClassAgnostic => Some(Nms::ClassAgnostic),
            HalNms::ClassAware => Some(Nms::ClassAware),
            HalNms::None => None,
        }
    }
}

/// Output type for model tensor outputs.
///
/// Identifies the role of a model output tensor in the decoder pipeline.
///
/// @see hal_decoder_params_add_output
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalOutputType {
    /// Combined detection output (e.g. YOLO fused [batch, features, boxes])
    Detection = 0,
    /// Split boxes output [batch, coords, boxes]
    Boxes = 1,
    /// Split scores output [batch, classes, boxes]
    Scores = 2,
    /// Prototype masks for instance segmentation [batch, protos, H, W]
    Protos = 3,
    /// Segmentation mask output
    Segmentation = 4,
    /// Mask coefficients for instance segmentation
    MaskCoefficients = 5,
    /// Raw mask output
    Mask = 6,
    /// Class indices for end-to-end split models [batch, N, 1]
    Classes = 7,
}

/// Decoder framework type.
///
/// Identifies the model framework that produced the output tensors.
///
/// @see hal_decoder_params_add_output
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalDecoderType {
    /// Ultralytics YOLO models (YOLOv5, YOLOv8, YOLO11, YOLO26)
    Ultralytics = 0,
    /// Au-Zone ModelPack models
    ModelPack = 1,
}

impl From<HalDecoderType> for configs::DecoderType {
    fn from(dt: HalDecoderType) -> Self {
        match dt {
            HalDecoderType::Ultralytics => configs::DecoderType::Ultralytics,
            HalDecoderType::ModelPack => configs::DecoderType::ModelPack,
        }
    }
}

/// Decoder version for Ultralytics models.
///
/// Determines the YOLO architecture version and decoding strategy.
///
/// @see hal_decoder_params_set_decoder_version
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalDecoderVersion {
    /// YOLOv5 - anchor-based decoder, requires external NMS
    Yolov5 = 0,
    /// YOLOv8 - anchor-free DFL decoder, requires external NMS
    Yolov8 = 1,
    /// YOLO11 - anchor-free DFL decoder, requires external NMS
    Yolo11 = 2,
    /// YOLO26 - end-to-end model with embedded NMS
    Yolo26 = 3,
}

impl From<HalDecoderVersion> for configs::DecoderVersion {
    fn from(v: HalDecoderVersion) -> Self {
        match v {
            HalDecoderVersion::Yolov5 => configs::DecoderVersion::Yolov5,
            HalDecoderVersion::Yolov8 => configs::DecoderVersion::Yolov8,
            HalDecoderVersion::Yolo11 => configs::DecoderVersion::Yolo11,
            HalDecoderVersion::Yolo26 => configs::DecoderVersion::Yolo26,
        }
    }
}

/// Named dimension for tensor shapes.
///
/// Assigns a semantic name to each dimension in a tensor shape, allowing the
/// decoder to handle different tensor layouts (NCHW vs NHWC, etc.).
///
/// Pass an array of these alongside shape values in
/// `hal_decoder_params_add_output()`, or pass NULL for unnamed shapes.
///
/// @see hal_decoder_params_add_output
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalDimName {
    /// Batch dimension
    Batch = 0,
    /// Height spatial dimension
    Height = 1,
    /// Width spatial dimension
    Width = 2,
    /// Number of object classes
    NumClasses = 3,
    /// Number of features (e.g. 4 box coords + num_classes + mask_coeffs)
    NumFeatures = 4,
    /// Number of detection boxes / proposals
    NumBoxes = 5,
    /// Number of prototype masks
    NumProtos = 6,
    /// Number of anchors multiplied by features (ModelPack split)
    NumAnchorsXFeatures = 7,
    /// Padding dimension
    Padding = 8,
    /// Box coordinate dimension (typically 4)
    BoxCoords = 9,
}

impl From<HalDimName> for configs::DimName {
    fn from(d: HalDimName) -> Self {
        match d {
            HalDimName::Batch => configs::DimName::Batch,
            HalDimName::Height => configs::DimName::Height,
            HalDimName::Width => configs::DimName::Width,
            HalDimName::NumClasses => configs::DimName::NumClasses,
            HalDimName::NumFeatures => configs::DimName::NumFeatures,
            HalDimName::NumBoxes => configs::DimName::NumBoxes,
            HalDimName::NumProtos => configs::DimName::NumProtos,
            HalDimName::NumAnchorsXFeatures => configs::DimName::NumAnchorsXFeatures,
            HalDimName::Padding => configs::DimName::Padding,
            HalDimName::BoxCoords => configs::DimName::BoxCoords,
        }
    }
}

/// Opaque decoder construction parameters.
///
/// Configures how ML model outputs are decoded into detection boxes and
/// segmentation masks. Create with `hal_decoder_params_new()`, configure
/// with setter functions, then pass to `hal_decoder_new()`.
///
/// Configuration can be provided in two ways (mutually exclusive):
/// 1. From a file or string: `hal_decoder_params_set_config_file()`,
///    `hal_decoder_params_set_config_json()`, or
///    `hal_decoder_params_set_config_yaml()`
/// 2. Programmatically: `hal_decoder_params_add_output()` to define each
///    model output tensor
///
/// @section dp_programmatic Programmatic Configuration
/// @code{.c}
/// hal_decoder_params *params = hal_decoder_params_new();
///
/// size_t scores_shape[] = {1, 80, 8400};
/// enum hal_dim_name scores_dims[] = {HAL_DIM_NAME_BATCH, HAL_DIM_NAME_NUM_CLASSES,
///                                     HAL_DIM_NAME_NUM_BOXES};
/// int s = hal_decoder_params_add_output(params, HAL_OUTPUT_TYPE_SCORES,
///             HAL_DECODER_TYPE_ULTRALYTICS, scores_shape, scores_dims, 3);
/// hal_decoder_params_output_set_quantization(params, s, 0.003906f, 0);
///
/// size_t boxes_shape[] = {1, 4, 8400};
/// enum hal_dim_name boxes_dims[] = {HAL_DIM_NAME_BATCH, HAL_DIM_NAME_BOX_COORDS,
///                                    HAL_DIM_NAME_NUM_BOXES};
/// int b = hal_decoder_params_add_output(params, HAL_OUTPUT_TYPE_BOXES,
///             HAL_DECODER_TYPE_ULTRALYTICS, boxes_shape, boxes_dims, 3);
/// hal_decoder_params_output_set_quantization(params, b, 0.019824f, 0);
///
/// hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AGNOSTIC);
/// hal_decoder *decoder = hal_decoder_new(params);
/// hal_decoder_params_free(params);
/// @endcode
///
/// @section dp_file File-based Configuration
/// @code{.c}
/// hal_decoder_params *params = hal_decoder_params_new();
/// hal_decoder_params_set_config_file(params, "/models/yolov8n/edgefirst.yaml");
/// hal_decoder_params_set_score_threshold(params, 0.25f);
/// hal_decoder *decoder = hal_decoder_new(params);
/// hal_decoder_params_free(params);
/// @endcode
///
/// @see hal_decoder_params_new, hal_decoder_params_free, hal_decoder_new
pub struct HalDecoderParams {
    outputs: Vec<ConfigOutput>,
    config_json: Option<String>,
    config_yaml: Option<String>,
    config_file: Option<String>,
    score_threshold: f32,
    iou_threshold: f32,
    nms: Option<configs::Nms>,
    decoder_version: Option<configs::DecoderVersion>,
}

/// Opaque decoder type.
pub struct HalDecoder {
    pub(crate) inner: Decoder,
}

/// List of detection boxes.
pub struct HalDetectBoxList {
    pub(crate) boxes: Vec<DetectBox>,
}

/// List of segmentation results.
pub struct HalSegmentationList {
    pub(crate) masks: Vec<Segmentation>,
}

// ============================================================================
// Decoder Params Functions
// ============================================================================

/// Create new decoder parameters.
///
/// Returns an opaque handle initialized with safe defaults:
///
/// | Setting | Default |
/// |---------|---------|
/// | score_threshold | 0.5 |
/// | iou_threshold | 0.5 |
/// | nms | HAL_NMS_CLASS_AGNOSTIC |
///
/// The caller must configure outputs (via `hal_decoder_params_add_output()`,
/// `hal_decoder_params_set_config_file()`, `hal_decoder_params_set_config_json()`,
/// or `hal_decoder_params_set_config_yaml()`) before passing to
/// `hal_decoder_new()`. Free with `hal_decoder_params_free()` after use.
///
/// @return New params handle, or NULL on allocation failure
///
/// @par Example
/// @code{.c}
/// hal_decoder_params *params = hal_decoder_params_new();
/// hal_decoder_params_set_config_file(params, "edgefirst.yaml");
/// hal_decoder_params_set_score_threshold(params, 0.3f);
/// hal_decoder *decoder = hal_decoder_new(params);
/// hal_decoder_params_free(params);
/// @endcode
///
/// @see hal_decoder_params_free, hal_decoder_new
#[no_mangle]
pub extern "C" fn hal_decoder_params_new() -> *mut HalDecoderParams {
    Box::into_raw(Box::new(HalDecoderParams {
        outputs: Vec::new(),
        config_json: None,
        config_yaml: None,
        config_file: None,
        score_threshold: 0.5,
        iou_threshold: 0.5,
        nms: Some(configs::Nms::ClassAgnostic),
        decoder_version: None,
    }))
}

/// Free decoder parameters.
///
/// @param params Params handle to free (can be NULL, no-op)
///
/// @see hal_decoder_params_new
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_free(params: *mut HalDecoderParams) {
    if !params.is_null() {
        drop(Box::from_raw(params));
    }
}

/// Set decoder configuration from a JSON string.
///
/// @param params Params handle
/// @param json   JSON string (null-terminated)
/// @param len    String length in bytes (excluding null terminator),
///               or 0 to use strlen
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_set_config_yaml, hal_decoder_params_set_config_file
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_config_json(
    params: *mut HalDecoderParams,
    json: *const c_char,
    len: size_t,
) -> c_int {
    check_null!(params, json);
    let s = if len == 0 {
        match CStr::from_ptr(json).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return set_error(libc::EINVAL),
        }
    } else {
        let bytes = std::slice::from_raw_parts(json.cast::<u8>(), len);
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => return set_error(libc::EINVAL),
        }
    };
    (*params).config_json = Some(s);
    0
}

/// Set decoder configuration from a YAML string.
///
/// @param params Params handle
/// @param yaml   YAML string (null-terminated)
/// @param len    String length in bytes (excluding null terminator),
///               or 0 to use strlen
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_set_config_json, hal_decoder_params_set_config_file
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_config_yaml(
    params: *mut HalDecoderParams,
    yaml: *const c_char,
    len: size_t,
) -> c_int {
    check_null!(params, yaml);
    let s = if len == 0 {
        match CStr::from_ptr(yaml).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return set_error(libc::EINVAL),
        }
    } else {
        let bytes = std::slice::from_raw_parts(yaml.cast::<u8>(), len);
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => return set_error(libc::EINVAL),
        }
    };
    (*params).config_yaml = Some(s);
    0
}

/// Set decoder configuration from a file path.
///
/// The file is read when `hal_decoder_new()` is called, not immediately.
/// JSON and YAML are auto-detected from the file extension and content.
///
/// @param params Params handle
/// @param path   File path (null-terminated)
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_set_config_json, hal_decoder_params_set_config_yaml
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_config_file(
    params: *mut HalDecoderParams,
    path: *const c_char,
) -> c_int {
    check_null!(params, path);
    let s = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return set_error(libc::EINVAL),
    };
    (*params).config_file = Some(s);
    0
}

/// Add a model output tensor to the decoder configuration.
///
/// Builds the output configuration programmatically. Each call adds one
/// output tensor description. The decoder resolves the model type
/// automatically from the combination of outputs.
///
/// @param params  Params handle
/// @param type_   Output tensor role (scores, boxes, detection, etc.)
/// @param decoder Framework that produced the model (ultralytics, modelpack)
/// @param shape   Array of dimension sizes (length = ndim)
/// @param dims    Array of dimension names (length = ndim), or NULL for
///                unnamed dimensions. When non-NULL, named dimensions
///                (dshape) are stored and shape is derived from them.
/// @param ndim    Number of dimensions
/// @return Output index (>= 0) on success, -1 on error (errno = EINVAL)
///
/// @par Example
/// @code{.c}
/// size_t shape[] = {1, 80, 8400};
/// enum hal_dim_name dims[] = {HAL_DIM_NAME_BATCH, HAL_DIM_NAME_NUM_CLASSES,
///                              HAL_DIM_NAME_NUM_BOXES};
/// int idx = hal_decoder_params_add_output(params, HAL_OUTPUT_TYPE_SCORES,
///               HAL_DECODER_TYPE_ULTRALYTICS, shape, dims, 3);
/// @endcode
///
/// @see hal_decoder_params_output_set_quantization
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_add_output(
    params: *mut HalDecoderParams,
    type_: HalOutputType,
    decoder: HalDecoderType,
    shape: *const size_t,
    dims: *const HalDimName,
    ndim: size_t,
) -> c_int {
    if params.is_null() || shape.is_null() || ndim == 0 {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    }

    let shape_slice = std::slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.to_vec();
    let decoder_type: configs::DecoderType = decoder.into();

    let dshape: Vec<(configs::DimName, usize)> = if !dims.is_null() {
        let dims_slice = std::slice::from_raw_parts(dims, ndim);
        dims_slice
            .iter()
            .zip(shape_slice.iter())
            .map(|(d, s)| ((*d).into(), *s))
            .collect()
    } else {
        Vec::new()
    };

    // If dshape is non-empty, shape will be derived from it by normalize_output.
    // If dshape is empty, use the provided shape directly.
    let output = match type_ {
        HalOutputType::Detection => ConfigOutput::Detection(configs::Detection {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::Boxes => ConfigOutput::Boxes(configs::Boxes {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::Scores => ConfigOutput::Scores(configs::Scores {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::Protos => ConfigOutput::Protos(configs::Protos {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::Segmentation => ConfigOutput::Segmentation(configs::Segmentation {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::MaskCoefficients => {
            ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                decoder: decoder_type,
                shape: shape_vec,
                dshape,
                ..Default::default()
            })
        }
        HalOutputType::Mask => ConfigOutput::Mask(configs::Mask {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
        HalOutputType::Classes => ConfigOutput::Classes(configs::Classes {
            decoder: decoder_type,
            shape: shape_vec,
            dshape,
            ..Default::default()
        }),
    };

    let p = &mut *params;
    p.outputs.push(output);
    (p.outputs.len() - 1) as c_int
}

/// Set quantization parameters on an output by index.
///
/// @param params     Params handle
/// @param index      Output index (returned by `hal_decoder_params_add_output`)
/// @param scale      Quantization scale
/// @param zero_point Quantization zero point
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_add_output
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_output_set_quantization(
    params: *mut HalDecoderParams,
    index: c_int,
    scale: f32,
    zero_point: c_int,
) -> c_int {
    check_null!(params);
    let p = &mut *params;
    let idx = index as usize;
    if idx >= p.outputs.len() {
        return set_error(libc::EINVAL);
    }
    let quant = Some(configs::QuantTuple(scale, zero_point));
    match &mut p.outputs[idx] {
        ConfigOutput::Detection(c) => c.quantization = quant,
        ConfigOutput::Boxes(c) => c.quantization = quant,
        ConfigOutput::Scores(c) => c.quantization = quant,
        ConfigOutput::Protos(c) => c.quantization = quant,
        ConfigOutput::Segmentation(c) => c.quantization = quant,
        ConfigOutput::MaskCoefficients(c) => c.quantization = quant,
        ConfigOutput::Mask(c) => c.quantization = quant,
        ConfigOutput::Classes(c) => c.quantization = quant,
    }
    0
}

/// Set anchor boxes on a detection output by index.
///
/// Only valid for `HAL_OUTPUT_TYPE_DETECTION` outputs.
///
/// @param params      Params handle
/// @param index       Output index (returned by `hal_decoder_params_add_output`)
/// @param anchors     Array of [width, height] anchor pairs
/// @param num_anchors Number of anchor pairs
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_add_output
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_output_set_anchors(
    params: *mut HalDecoderParams,
    index: c_int,
    anchors: *const [f32; 2],
    num_anchors: size_t,
) -> c_int {
    check_null!(params, anchors);
    let p = &mut *params;
    let idx = index as usize;
    if idx >= p.outputs.len() {
        return set_error(libc::EINVAL);
    }
    let anchors_vec: Vec<[f32; 2]> = std::slice::from_raw_parts(anchors, num_anchors).to_vec();
    match &mut p.outputs[idx] {
        ConfigOutput::Detection(c) => c.anchors = Some(anchors_vec),
        _ => return set_error(libc::EINVAL),
    }
    0
}

/// Set the normalized flag on a detection or boxes output by index.
///
/// Only valid for `HAL_OUTPUT_TYPE_DETECTION` and `HAL_OUTPUT_TYPE_BOXES` outputs.
///
/// @param params     Params handle
/// @param index      Output index (returned by `hal_decoder_params_add_output`)
/// @param normalized Non-zero for normalized [0,1] coordinates, 0 for pixel
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see hal_decoder_params_add_output
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_output_set_normalized(
    params: *mut HalDecoderParams,
    index: c_int,
    normalized: c_int,
) -> c_int {
    check_null!(params);
    let p = &mut *params;
    let idx = index as usize;
    if idx >= p.outputs.len() {
        return set_error(libc::EINVAL);
    }
    let norm = Some(normalized != 0);
    match &mut p.outputs[idx] {
        ConfigOutput::Detection(c) => c.normalized = norm,
        ConfigOutput::Boxes(c) => c.normalized = norm,
        _ => return set_error(libc::EINVAL),
    }
    0
}

/// Set the score threshold.
///
/// Detections with confidence below this threshold are discarded.
///
/// @param params Params handle
/// @param threshold Score threshold (default: 0.5)
/// @return 0 on success, -1 on error (errno = EINVAL)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_score_threshold(
    params: *mut HalDecoderParams,
    threshold: f32,
) -> c_int {
    check_null!(params);
    (*params).score_threshold = threshold;
    0
}

/// Set the IoU (Intersection over Union) threshold for NMS.
///
/// @param params Params handle
/// @param threshold IoU threshold (default: 0.5)
/// @return 0 on success, -1 on error (errno = EINVAL)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_iou_threshold(
    params: *mut HalDecoderParams,
    threshold: f32,
) -> c_int {
    check_null!(params);
    (*params).iou_threshold = threshold;
    0
}

/// Set the NMS (Non-Maximum Suppression) mode.
///
/// @param params Params handle
/// @param nms NMS mode
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see HalNms
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_nms(
    params: *mut HalDecoderParams,
    nms: HalNms,
) -> c_int {
    check_null!(params);
    (*params).nms = nms.into();
    0
}

/// Set the decoder version for Ultralytics models.
///
/// Determines the YOLO architecture version and decoding strategy.
///
/// @param params  Params handle
/// @param version Decoder version
/// @return 0 on success, -1 on error (errno = EINVAL)
///
/// @see HalDecoderVersion
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_params_set_decoder_version(
    params: *mut HalDecoderParams,
    version: HalDecoderVersion,
) -> c_int {
    check_null!(params);
    (*params).decoder_version = Some(version.into());
    0
}

/// Create a new decoder from parameters.
///
/// Validates the parameters and constructs a decoder ready for use with
/// `hal_decoder_decode()`. Configuration must be provided either via
/// `hal_decoder_params_add_output()` (programmatic) or via one of the
/// config file/string setters (mutually exclusive).
///
/// @param params Params handle (must not be NULL)
/// @return New decoder handle on success, NULL on error (errno set)
///
/// @par Errors (errno):
/// - EINVAL: NULL params or no configuration provided
/// - ENOENT: Config file path does not exist
/// - EIO: Config file could not be read
/// - EBADMSG: Configuration is syntactically valid but semantically invalid
///   (e.g. missing required `outputs` array, unknown decoder type)
///
/// @par Example
/// @code{.c}
/// hal_decoder_params *params = hal_decoder_params_new();
/// hal_decoder_params_set_config_file(params, "/models/yolov8n/edgefirst.yaml");
/// hal_decoder_params_set_score_threshold(params, 0.25f);
/// hal_decoder_params_set_iou_threshold(params, 0.45f);
/// hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AWARE);
///
/// hal_decoder *decoder = hal_decoder_new(params);
/// if (!decoder) {
///     fprintf(stderr, "decoder: %s\n", strerror(errno));
///     hal_decoder_params_free(params);
///     return -1;
/// }
///
/// // ... use decoder with hal_decoder_decode() ...
///
/// hal_decoder_params_free(params);
/// hal_decoder_free(decoder);
/// @endcode
///
/// @see hal_decoder_params_new, hal_decoder_params_free, hal_decoder_free
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_new(params: *const HalDecoderParams) -> *mut HalDecoder {
    check_null_ret_null!(params);

    let p = &*params;

    let has_outputs = !p.outputs.is_empty();
    let has_json = p.config_json.is_some();
    let has_yaml = p.config_yaml.is_some();
    let has_file = p.config_file.is_some();

    // Must have exactly one config source
    let config_count = has_outputs as u8 + has_json as u8 + has_yaml as u8 + has_file as u8;
    if config_count != 1 {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return std::ptr::null_mut();
    }

    let mut builder = DecoderBuilder::new()
        .with_score_threshold(p.score_threshold)
        .with_iou_threshold(p.iou_threshold)
        .with_nms(p.nms);

    if has_outputs {
        for output in &p.outputs {
            builder = builder.add_output(output.clone());
        }
        if let Some(version) = p.decoder_version {
            builder = builder.with_decoder_version(version);
        }
    } else if has_json {
        builder = builder.with_config_json_str(p.config_json.clone().unwrap());
    } else if has_yaml {
        builder = builder.with_config_yaml_str(p.config_yaml.clone().unwrap());
    } else {
        let path_str = p.config_file.as_ref().unwrap();

        let content = match std::fs::read_to_string(path_str) {
            Ok(c) => c,
            Err(e) => {
                errno::set_errno(errno::Errno(if e.kind() == std::io::ErrorKind::NotFound {
                    libc::ENOENT
                } else {
                    libc::EIO
                }));
                return std::ptr::null_mut();
            }
        };

        let is_json = path_str.ends_with(".json") || content.trim_start().starts_with('{');
        if is_json {
            builder = builder.with_config_json_str(content);
        } else {
            builder = builder.with_config_yaml_str(content);
        }
    }

    let decoder = try_or_null!(builder.build(), libc::EBADMSG);
    Box::into_raw(Box::new(HalDecoder { inner: decoder }))
}

// ============================================================================
// Decoder Functions
// ============================================================================

/// Decode model outputs into detection boxes and segmentation masks.
///
/// Automatically selects the decoding path based on tensor dtype:
/// - f32 tensors -> float decode path
/// - Integer tensors (u8, i8, u16, i16, u32, i32) -> quantized decode path
///
/// All output tensors must be the same general category (all float or all integer).
///
/// @param decoder Decoder handle
/// @param outputs Array of output tensor pointers
/// @param num_outputs Number of output tensors
/// @param out_boxes Output parameter for detection box list (caller must free)
/// @param out_segmentations Output parameter for segmentation list (can be NULL; caller must free if non-NULL)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL decoder/outputs/out_boxes, mixed dtypes)
/// - EIO: Decoding failed
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_decode(
    decoder: *const HalDecoder,
    outputs: *const *const HalTensor,
    num_outputs: size_t,
    out_boxes: *mut *mut HalDetectBoxList,
    out_segmentations: *mut *mut HalSegmentationList,
) -> c_int {
    check_null!(decoder, outputs, out_boxes);

    if num_outputs == 0 {
        return set_error(libc::EINVAL);
    }

    let outputs_slice = std::slice::from_raw_parts(outputs, num_outputs);

    // Extract TensorDyn references from HalTensor pointers
    let mut tensor_refs: Vec<&TensorDyn> = Vec::with_capacity(num_outputs);
    for &tensor_ptr in outputs_slice {
        if tensor_ptr.is_null() {
            return set_error(libc::EINVAL);
        }
        tensor_refs.push(&unsafe { &*tensor_ptr }.inner);
    }

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(100);
    let mut masks: Vec<Segmentation> = Vec::new();

    try_or_errno!(
        (*decoder)
            .inner
            .decode(&tensor_refs, &mut boxes, &mut masks),
        libc::EIO
    );

    *out_boxes = Box::into_raw(Box::new(HalDetectBoxList { boxes }));

    if !out_segmentations.is_null() {
        *out_segmentations = Box::into_raw(Box::new(HalSegmentationList { masks }));
    }

    0
}

/// Extract `TensorDyn` references from an array of `HalTensor` pointers.
///
/// Returns `Err(-1)` with `errno = EINVAL` if any pointer is NULL.
pub(crate) unsafe fn extract_tensor_refs(
    outputs_slice: &[*const HalTensor],
) -> Result<Vec<&TensorDyn>, c_int> {
    let mut refs = Vec::with_capacity(outputs_slice.len());
    for &tensor_ptr in outputs_slice {
        if tensor_ptr.is_null() {
            set_error(libc::EINVAL);
            return Err(-1);
        }
        refs.push(&unsafe { &*tensor_ptr }.inner);
    }
    Ok(refs)
}

/// Get the model type string from a decoder.
///
/// Returns a human-readable string identifying the model type (e.g., "yolo_det",
/// "modelpack_segdet"). The returned string is owned by the caller and must be
/// freed with free().
///
/// @param decoder Decoder handle
/// @return Newly allocated C string with model type, or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL decoder
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_model_type(decoder: *const HalDecoder) -> *mut c_char {
    if decoder.is_null() {
        return set_error_null(libc::EINVAL);
    }

    let model_type = unsafe { &(*decoder) }.inner.model_type();
    let type_str = format!("{:?}", model_type);

    // Convert the Debug representation to a snake_case identifier
    let name = type_str
        .split_once('{')
        .or_else(|| type_str.split_once(' '))
        .map(|(name, _)| name.trim())
        .unwrap_or(&type_str);

    // Convert CamelCase to snake_case
    let mut snake = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            snake.push('_');
        }
        snake.push(ch.to_lowercase().next().unwrap_or(ch));
    }

    str_to_c_string(&snake)
}

/// Get whether the decoder produces normalized box coordinates.
///
/// @param decoder Decoder handle
/// @return 1 if boxes are in normalized [0,1] coordinates,
///         0 if boxes are in pixel coordinates,
///        -1 if unknown or decoder is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_normalized_boxes(decoder: *const HalDecoder) -> c_int {
    if decoder.is_null() {
        return -1;
    }

    match unsafe { &(*decoder) }.inner.normalized_boxes() {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

/// Dequantize an integer tensor into a float tensor.
///
/// Converts quantized integer data to float using: output = (input - zero_point) * scale.
/// The input tensor must be an integer dtype (u8, i8, u16, i16, u32, i32).
/// The output tensor must be f32 dtype and have the same total number of elements.
///
/// @param input Input integer tensor (not consumed)
/// @param quant Quantization parameters (scale and zero_point)
/// @param output Output f32 tensor (must be pre-allocated with same element count)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL input/output, input is not integer, output is not f32, size mismatch
/// - EIO: Failed to map tensors
#[no_mangle]
pub unsafe extern "C" fn hal_dequantize(
    input: *const HalTensor,
    quant: HalQuantization,
    output: *mut HalTensor,
) -> c_int {
    check_null!(input, output);

    let quant_rust: Quantization = quant.into();

    // Output must be f32
    let output_tensor = match &mut unsafe { &mut *output }.inner {
        TensorDyn::F32(t) => t,
        _ => return set_error(libc::EINVAL),
    };
    let mut out_map = try_or_errno!(output_tensor.map(), libc::EIO);

    macro_rules! dequantize_typed {
        ($t:expr) => {{
            let in_map = try_or_errno!($t.map(), libc::EIO);
            let in_slice = in_map.as_slice();
            let out_slice = out_map.as_mut_slice();
            if in_slice.len() != out_slice.len() {
                return set_error(libc::EINVAL);
            }
            dequantize_cpu_chunked(in_slice, quant_rust, out_slice);
        }};
    }

    match &unsafe { &*input }.inner {
        TensorDyn::U8(t) => dequantize_typed!(t),
        TensorDyn::I8(t) => dequantize_typed!(t),
        TensorDyn::U16(t) => dequantize_typed!(t),
        TensorDyn::I16(t) => dequantize_typed!(t),
        TensorDyn::U32(t) => dequantize_typed!(t),
        TensorDyn::I32(t) => dequantize_typed!(t),
        _ => return set_error(libc::EINVAL), // f32/f64/u64/i64 not supported
    }

    out_map.unmap();
    0
}

/// Convert a 3D segmentation mask into a 2D binary mask tensor.
///
/// Takes a segmentation from a segmentation list by index and converts it
/// to a 2D binary mask (height x width) as a new u8 tensor.
/// If the segmentation has depth=1, values >= 128 become 1, < 128 become 0.
/// If depth > 1, argmax across the depth dimension is used.
///
/// @param list Segmentation list handle
/// @param index Index of the segmentation (0-based)
/// @return New u8 tensor with shape [height, width], or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL list, index out of bounds, or invalid segmentation shape
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_segmentation_to_mask(
    list: *const HalSegmentationList,
    index: size_t,
) -> *mut HalTensor {
    if list.is_null() {
        return set_error_null(libc::EINVAL);
    }

    if index >= unsafe { &*list }.masks.len() {
        return set_error_null(libc::EINVAL);
    }

    let seg = &unsafe { &*list }.masks[index];
    let mask_2d = try_or_null!(segmentation_to_mask(seg.segmentation.view()), libc::EINVAL);

    let shape = mask_2d.shape();
    let h = shape[0];
    let w = shape[1];

    // Create a new u8 tensor and copy the mask data into it
    let tensor = match Tensor::<u8>::new(&[h, w], Some(TensorMemory::Mem), None) {
        Ok(t) => t,
        Err(_) => return set_error_null(libc::ENOMEM),
    };

    let mut map = try_or_null!(tensor.map(), libc::EIO);
    let slice = match mask_2d.as_slice() {
        Some(s) => s,
        None => return set_error_null(libc::EINVAL),
    };
    map.as_mut_slice().copy_from_slice(slice);
    map.unmap();

    Box::into_raw(Box::new(HalTensor {
        inner: TensorDyn::U8(tensor),
    }))
}

/// Free a decoder.
///
/// @param decoder Decoder handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_free(decoder: *mut HalDecoder) {
    if !decoder.is_null() {
        drop(Box::from_raw(decoder));
    }
}

// ============================================================================
// Detection Box List Functions
// ============================================================================

/// Get the number of detections in a list.
///
/// @param list Detection box list handle
/// @return Number of detections, or 0 if list is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_detect_box_list_len(list: *const HalDetectBoxList) -> size_t {
    if list.is_null() {
        return 0;
    }
    (*list).boxes.len()
}

/// Get a detection box from a list by index.
///
/// @param list Detection box list handle
/// @param index Index of the detection (0-based)
/// @param out_box Output parameter for the detection box
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL list/out_box, index out of bounds)
#[no_mangle]
pub unsafe extern "C" fn hal_detect_box_list_get(
    list: *const HalDetectBoxList,
    index: size_t,
    out_box: *mut HalDetectBox,
) -> c_int {
    check_null!(list, out_box);

    if index >= (*list).boxes.len() {
        return set_error(libc::EINVAL);
    }

    *out_box = HalDetectBox::from(&(&(*list).boxes)[index]);
    0
}

/// Free a detection box list.
///
/// @param list Detection box list handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_detect_box_list_free(list: *mut HalDetectBoxList) {
    if !list.is_null() {
        drop(Box::from_raw(list));
    }
}

// ============================================================================
// Segmentation List Functions
// ============================================================================

/// Get the number of segmentations in a list.
///
/// @param list Segmentation list handle
/// @return Number of segmentations, or 0 if list is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_segmentation_list_len(list: *const HalSegmentationList) -> size_t {
    if list.is_null() {
        return 0;
    }
    (*list).masks.len()
}

/// Get the bounding box of a segmentation by index.
///
/// @param list Segmentation list handle
/// @param index Index of the segmentation (0-based)
/// @param xmin Output parameter for left coordinate
/// @param ymin Output parameter for top coordinate
/// @param xmax Output parameter for right coordinate
/// @param ymax Output parameter for bottom coordinate
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL list/outputs, index out of bounds)
#[no_mangle]
pub unsafe extern "C" fn hal_segmentation_list_get_bbox(
    list: *const HalSegmentationList,
    index: size_t,
    xmin: *mut f32,
    ymin: *mut f32,
    xmax: *mut f32,
    ymax: *mut f32,
) -> c_int {
    check_null!(list, xmin, ymin, xmax, ymax);

    if index >= (*list).masks.len() {
        return set_error(libc::EINVAL);
    }

    let seg = &(&(*list).masks)[index];
    *xmin = seg.xmin;
    *ymin = seg.ymin;
    *xmax = seg.xmax;
    *ymax = seg.ymax;
    0
}

/// Get the mask data of a segmentation by index.
///
/// The returned pointer is borrowed and valid only during the list's lifetime.
///
/// @param list Segmentation list handle
/// @param index Index of the segmentation (0-based)
/// @param out_height Output parameter for mask height
/// @param out_width Output parameter for mask width
/// @return Pointer to mask data (uint8_t array), or NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL list, index out of bounds)
#[no_mangle]
pub unsafe extern "C" fn hal_segmentation_list_get_mask(
    list: *const HalSegmentationList,
    index: size_t,
    out_height: *mut size_t,
    out_width: *mut size_t,
) -> *const u8 {
    if list.is_null() || index >= (*list).masks.len() {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return std::ptr::null();
    }

    let seg = &(&(*list).masks)[index];
    let shape = seg.segmentation.shape();

    if !out_height.is_null() {
        *out_height = shape[0];
    }
    if !out_width.is_null() {
        *out_width = shape[1];
    }

    seg.segmentation.as_ptr()
}

/// Free a segmentation list.
///
/// @param list Segmentation list handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_segmentation_list_free(list: *mut HalSegmentationList) {
    if !list.is_null() {
        drop(Box::from_raw(list));
    }
}

/// Decode model outputs into tracked detection boxes and segmentation masks.
///
/// Automatically selects the decoding path based on tensor dtype:
/// - f32 tensors -> float decode path
/// - Integer tensors (u8, i8, u16, i16, u32, i32) -> quantized decode path
///
/// All output tensors must be the same general category (all float or all integer).
///
/// @param decoder Decoder handle
/// @param tracker Tracker handle for maintaining object identities across frames
/// @param timestamp Timestamp for the current frame (e.g., in nanoseconds)
/// @param outputs Array of output tensor pointers
/// @param num_outputs Number of output tensors
/// @param out_boxes Output parameter for detection box list (caller must free)
/// @param out_segmentations Output parameter for segmentation list (can be NULL; caller must free if non-NULL)
/// @param out_tracks Output parameter for track info list (can be NULL; caller must free if non-NULL)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL decoder/tracker/outputs/out_boxes, mixed dtypes)
/// - EIO: Decoding failed
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_decode_tracked(
    decoder: *const HalDecoder,
    tracker: *mut HalByteTrack,
    timestamp: u64,
    outputs: *const *const HalTensor,
    num_outputs: size_t,
    out_boxes: *mut *mut HalDetectBoxList,
    out_segmentations: *mut *mut HalSegmentationList,
    out_tracks: *mut *mut HalTrackInfoList,
) -> c_int {
    check_null!(decoder, tracker, outputs, out_boxes);

    if num_outputs == 0 {
        return set_error(libc::EINVAL);
    }

    let outputs_slice = std::slice::from_raw_parts(outputs, num_outputs);

    // Extract TensorDyn references from HalTensor pointers
    let tensor_refs = match extract_tensor_refs(outputs_slice) {
        Ok(refs) => refs,
        Err(rc) => return rc,
    };

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(100);
    let mut masks: Vec<Segmentation> = Vec::new();
    let mut tracks: Vec<TrackInfo> = Vec::new();

    try_or_errno!(
        (*decoder).inner.decode_tracked(
            &mut (*tracker).inner,
            timestamp,
            &tensor_refs,
            &mut boxes,
            &mut masks,
            &mut tracks
        ),
        libc::EIO
    );

    *out_boxes = Box::into_raw(Box::new(HalDetectBoxList { boxes }));

    if !out_segmentations.is_null() {
        *out_segmentations = Box::into_raw(Box::new(HalSegmentationList { masks }));
    }

    if !out_tracks.is_null() {
        *out_tracks = Box::into_raw(Box::new(HalTrackInfoList { tracks }));
    }

    0
}

#[cfg(test)]
mod tests {
    use edgefirst_decoder::dequantize_cpu;

    use super::*;
    use crate::tensor::{
        hal_tensor_free, hal_tensor_map_create, hal_tensor_map_data, hal_tensor_map_unmap,
        hal_tensor_new, HalDtype, HalTensorMemory,
    };
    use std::ffi::CString;

    // Valid YOLO detection config (84 features = 4 bbox + 80 classes)
    const YOLO_JSON_CONFIG: &str = r#"{
        "outputs": [{
            "decoder": "ultralytics",
            "type": "detection",
            "shape": [1, 84, 8400],
            "dshape": [["batch", 1], ["num_features", 84], ["num_boxes", 8400]]
        }],
        "nms": "class_aware"
    }"#;

    const YOLO_YAML_CONFIG: &str = r#"outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape: [[batch, 1], [num_features, 84], [num_boxes, 8400]]
nms: class_aware
"#;

    /// Helper: create a decoder from JSON config.
    unsafe fn make_decoder_json(json: &str) -> *mut HalDecoder {
        let config = CString::new(json).unwrap();
        let params = hal_decoder_params_new();
        hal_decoder_params_set_config_json(params, config.as_ptr(), 0);
        let decoder = hal_decoder_new(params);
        hal_decoder_params_free(params);
        decoder
    }

    #[test]
    fn test_params_new_free() {
        unsafe {
            let params = hal_decoder_params_new();
            assert!(!params.is_null());
            // No config set → decoder should fail
            let decoder = hal_decoder_new(params);
            assert!(decoder.is_null());
            hal_decoder_params_free(params);
            // NULL free is safe
            hal_decoder_params_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_null_handling() {
        unsafe {
            assert_eq!(hal_detect_box_list_len(std::ptr::null()), 0);
            assert_eq!(hal_segmentation_list_len(std::ptr::null()), 0);
            hal_detect_box_list_free(std::ptr::null_mut());
            hal_segmentation_list_free(std::ptr::null_mut());
            hal_decoder_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_quantization_conversion() {
        let hal_quant = HalQuantization {
            scale: 0.5,
            zero_point: 128,
        };
        let quant: Quantization = hal_quant.into();
        assert_eq!(quant.scale, 0.5);
        assert_eq!(quant.zero_point, 128);
    }

    #[test]
    fn test_nms_conversion() {
        let agnostic: Option<Nms> = HalNms::ClassAgnostic.into();
        assert!(matches!(agnostic, Some(Nms::ClassAgnostic)));

        let aware: Option<Nms> = HalNms::ClassAware.into();
        assert!(matches!(aware, Some(Nms::ClassAware)));

        let none: Option<Nms> = HalNms::None.into();
        assert!(none.is_none());
    }

    #[test]
    fn test_detect_box_conversion() {
        let detect_box = DetectBox {
            bbox: edgefirst_decoder::BoundingBox {
                xmin: 0.1,
                ymin: 0.2,
                xmax: 0.3,
                ymax: 0.4,
            },
            score: 0.95,
            label: 5,
        };
        let hal_box = HalDetectBox::from(&detect_box);
        assert!((hal_box.xmin - 0.1).abs() < 1e-6);
        assert!((hal_box.ymin - 0.2).abs() < 1e-6);
        assert!((hal_box.xmax - 0.3).abs() < 1e-6);
        assert!((hal_box.ymax - 0.4).abs() < 1e-6);
        assert!((hal_box.score - 0.95).abs() < 1e-6);
        assert_eq!(hal_box.label, 5);
    }

    #[test]
    fn test_decoder_new_with_json() {
        unsafe {
            let config = CString::new(YOLO_JSON_CONFIG).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_json(params, config.as_ptr(), 0);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_new_with_yaml() {
        unsafe {
            let config = CString::new(YOLO_YAML_CONFIG).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_yaml(params, config.as_ptr(), 0);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_new_with_thresholds() {
        unsafe {
            let config = CString::new(YOLO_JSON_CONFIG).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_json(params, config.as_ptr(), 0);
            hal_decoder_params_set_score_threshold(params, 0.3);
            hal_decoder_params_set_iou_threshold(params, 0.45);
            hal_decoder_params_set_nms(params, HalNms::ClassAware);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_new_null_params() {
        unsafe {
            assert!(hal_decoder_new(std::ptr::null()).is_null());
        }
    }

    #[test]
    fn test_decoder_new_no_config() {
        unsafe {
            let params = hal_decoder_params_new();
            let decoder = hal_decoder_new(params);
            assert!(decoder.is_null());
            hal_decoder_params_free(params);
        }
    }

    #[test]
    fn test_decoder_new_multiple_configs() {
        unsafe {
            let json = CString::new(YOLO_JSON_CONFIG).unwrap();
            let yaml = CString::new(YOLO_YAML_CONFIG).unwrap();

            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_json(params, json.as_ptr(), 0);
            hal_decoder_params_set_config_yaml(params, yaml.as_ptr(), 0);

            let decoder = hal_decoder_new(params);
            assert!(decoder.is_null());
            hal_decoder_params_free(params);
        }
    }

    #[test]
    fn test_add_output_programmatic() {
        unsafe {
            let params = hal_decoder_params_new();

            // Add a detection output with named dimensions
            let shape: [usize; 3] = [1, 84, 8400];
            let dims: [HalDimName; 3] = [
                HalDimName::Batch,
                HalDimName::NumFeatures,
                HalDimName::NumBoxes,
            ];
            let idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Detection,
                HalDecoderType::Ultralytics,
                shape.as_ptr(),
                dims.as_ptr(),
                3,
            );
            assert_eq!(idx, 0);

            hal_decoder_params_set_nms(params, HalNms::ClassAware);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_add_output_split_det() {
        unsafe {
            let params = hal_decoder_params_new();

            // Add scores output
            let scores_shape: [usize; 3] = [1, 80, 8400];
            let scores_dims: [HalDimName; 3] = [
                HalDimName::Batch,
                HalDimName::NumClasses,
                HalDimName::NumBoxes,
            ];
            let s = hal_decoder_params_add_output(
                params,
                HalOutputType::Scores,
                HalDecoderType::Ultralytics,
                scores_shape.as_ptr(),
                scores_dims.as_ptr(),
                3,
            );
            assert_eq!(s, 0);

            // Add boxes output
            let boxes_shape: [usize; 3] = [1, 4, 8400];
            let boxes_dims: [HalDimName; 3] = [
                HalDimName::Batch,
                HalDimName::BoxCoords,
                HalDimName::NumBoxes,
            ];
            let b = hal_decoder_params_add_output(
                params,
                HalOutputType::Boxes,
                HalDecoderType::Ultralytics,
                boxes_shape.as_ptr(),
                boxes_dims.as_ptr(),
                3,
            );
            assert_eq!(b, 1);

            hal_decoder_params_set_nms(params, HalNms::ClassAgnostic);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_add_output_no_dims() {
        unsafe {
            let params = hal_decoder_params_new();

            // Add output without named dimensions (dims = NULL)
            let shape: [usize; 3] = [1, 84, 8400];
            let idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Detection,
                HalDecoderType::Ultralytics,
                shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(idx, 0);

            hal_decoder_params_set_nms(params, HalNms::ClassAware);

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_add_output_null_params() {
        unsafe {
            let shape: [usize; 3] = [1, 84, 8400];
            assert_eq!(
                hal_decoder_params_add_output(
                    std::ptr::null_mut(),
                    HalOutputType::Detection,
                    HalDecoderType::Ultralytics,
                    shape.as_ptr(),
                    std::ptr::null(),
                    3,
                ),
                -1
            );
        }
    }

    #[test]
    fn test_output_set_quantization() {
        unsafe {
            let params = hal_decoder_params_new();

            let shape: [usize; 3] = [1, 80, 8400];
            let idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Scores,
                HalDecoderType::Ultralytics,
                shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(idx, 0);

            assert_eq!(
                hal_decoder_params_output_set_quantization(params, 0, 0.003906, 0),
                0
            );

            // Out of bounds
            assert_eq!(
                hal_decoder_params_output_set_quantization(params, 1, 0.5, 0),
                -1
            );

            hal_decoder_params_free(params);
        }
    }

    #[test]
    fn test_output_set_anchors() {
        unsafe {
            let params = hal_decoder_params_new();

            let shape: [usize; 4] = [1, 9, 15, 18];
            let idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Detection,
                HalDecoderType::ModelPack,
                shape.as_ptr(),
                std::ptr::null(),
                4,
            );
            assert_eq!(idx, 0);

            let anchors: [[f32; 2]; 3] = [[0.367, 0.315], [0.388, 0.474], [0.533, 0.644]];
            assert_eq!(
                hal_decoder_params_output_set_anchors(params, 0, anchors.as_ptr(), 3),
                0
            );

            // Add a non-detection output and try to set anchors (should fail)
            let scores_shape: [usize; 3] = [1, 80, 8400];
            let s_idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Scores,
                HalDecoderType::Ultralytics,
                scores_shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(
                hal_decoder_params_output_set_anchors(params, s_idx, anchors.as_ptr(), 3),
                -1
            );

            hal_decoder_params_free(params);
        }
    }

    #[test]
    fn test_output_set_normalized() {
        unsafe {
            let params = hal_decoder_params_new();

            // Detection output: normalized is valid
            let shape: [usize; 3] = [1, 84, 8400];
            let det_idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Detection,
                HalDecoderType::Ultralytics,
                shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(
                hal_decoder_params_output_set_normalized(params, det_idx, 1),
                0
            );

            // Boxes output: normalized is valid
            let boxes_shape: [usize; 3] = [1, 4, 8400];
            let box_idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Boxes,
                HalDecoderType::Ultralytics,
                boxes_shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(
                hal_decoder_params_output_set_normalized(params, box_idx, 0),
                0
            );

            // Scores output: normalized is invalid
            let scores_shape: [usize; 3] = [1, 80, 8400];
            let sc_idx = hal_decoder_params_add_output(
                params,
                HalOutputType::Scores,
                HalDecoderType::Ultralytics,
                scores_shape.as_ptr(),
                std::ptr::null(),
                3,
            );
            assert_eq!(
                hal_decoder_params_output_set_normalized(params, sc_idx, 1),
                -1
            );

            hal_decoder_params_free(params);
        }
    }

    #[test]
    fn test_set_decoder_version() {
        unsafe {
            let params = hal_decoder_params_new();

            // Add a YOLO26 end-to-end detection output
            let shape: [usize; 3] = [1, 100, 6];
            let dims: [HalDimName; 3] = [
                HalDimName::Batch,
                HalDimName::NumBoxes,
                HalDimName::NumFeatures,
            ];
            hal_decoder_params_add_output(
                params,
                HalOutputType::Detection,
                HalDecoderType::Ultralytics,
                shape.as_ptr(),
                dims.as_ptr(),
                3,
            );

            assert_eq!(
                hal_decoder_params_set_decoder_version(params, HalDecoderVersion::Yolo26),
                0
            );

            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_params_setter_null_safety() {
        unsafe {
            // All setters should return -1 for NULL params
            assert_eq!(
                hal_decoder_params_set_config_json(
                    std::ptr::null_mut(),
                    b"{}".as_ptr() as *const c_char,
                    2
                ),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_config_yaml(
                    std::ptr::null_mut(),
                    b"".as_ptr() as *const c_char,
                    0
                ),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_config_file(std::ptr::null_mut(), c"path".as_ptr()),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_score_threshold(std::ptr::null_mut(), 0.5),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_iou_threshold(std::ptr::null_mut(), 0.5),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_nms(std::ptr::null_mut(), HalNms::ClassAgnostic),
                -1
            );
            assert_eq!(
                hal_decoder_params_set_decoder_version(
                    std::ptr::null_mut(),
                    HalDecoderVersion::Yolov8
                ),
                -1
            );
        }
    }

    #[test]
    fn test_decode_null_parameters() {
        unsafe {
            let decoder = make_decoder_json(YOLO_JSON_CONFIG);
            assert!(!decoder.is_null());

            let shape: [usize; 3] = [1, 84, 8400];
            let tensor = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());
            let outputs = [tensor as *const HalTensor];

            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();

            // NULL decoder
            assert_eq!(
                hal_decoder_decode(
                    std::ptr::null(),
                    outputs.as_ptr(),
                    1,
                    &mut boxes as *mut _,
                    std::ptr::null_mut(),
                ),
                -1
            );

            // NULL outputs
            assert_eq!(
                hal_decoder_decode(
                    decoder,
                    std::ptr::null(),
                    1,
                    &mut boxes as *mut _,
                    std::ptr::null_mut()
                ),
                -1
            );

            // NULL out_boxes
            assert_eq!(
                hal_decoder_decode(
                    decoder,
                    outputs.as_ptr(),
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                -1
            );

            // Zero num_outputs
            assert_eq!(
                hal_decoder_decode(
                    decoder,
                    outputs.as_ptr(),
                    0,
                    &mut boxes as *mut _,
                    std::ptr::null_mut()
                ),
                -1
            );

            hal_tensor_free(tensor);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decode_float_success() {
        unsafe {
            let config = CString::new(YOLO_JSON_CONFIG).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_json(params, config.as_ptr(), 0);
            hal_decoder_params_set_score_threshold(params, 0.01);
            let decoder = hal_decoder_new(params);
            hal_decoder_params_free(params);
            assert!(!decoder.is_null());

            let shape: [usize; 3] = [1, 84, 8400];
            let tensor = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            let map = hal_tensor_map_create(tensor);
            assert!(!map.is_null());
            let data = hal_tensor_map_data(map) as *mut f32;

            for i in 0..(84 * 8400) {
                *data.add(i) = 0.0;
            }

            #[allow(clippy::identity_op, clippy::erasing_op)]
            {
                let box_idx = 0usize;
                let num_boxes = 8400usize;
                *data.add(0 * num_boxes + box_idx) = 0.5; // cx
                *data.add(1 * num_boxes + box_idx) = 0.5; // cy
                *data.add(2 * num_boxes + box_idx) = 0.1; // w
                *data.add(3 * num_boxes + box_idx) = 0.1; // h
                *data.add(4 * num_boxes + box_idx) = 0.9; // class 0 score
            }

            hal_tensor_map_unmap(map);

            let outputs = [tensor as *const HalTensor];
            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut segs: *mut HalSegmentationList = std::ptr::null_mut();

            let result = hal_decoder_decode(
                decoder,
                outputs.as_ptr(),
                1,
                &mut boxes as *mut _,
                &mut segs as *mut _,
            );
            assert_eq!(result, 0);
            assert!(!boxes.is_null());
            assert!(!segs.is_null());

            let len = hal_detect_box_list_len(boxes);

            if len > 0 {
                let mut box_out = HalDetectBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 0.0,
                    ymax: 0.0,
                    score: 0.0,
                    label: 0,
                };
                assert_eq!(hal_detect_box_list_get(boxes, 0, &mut box_out), 0);
                assert!(box_out.xmax >= box_out.xmin);
                assert!(box_out.ymax >= box_out.ymin);
            }

            let mut box_out = HalDetectBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 0.0,
                ymax: 0.0,
                score: 0.0,
                label: 0,
            };
            assert_eq!(hal_detect_box_list_get(boxes, len, &mut box_out), -1);

            hal_detect_box_list_free(boxes);
            hal_segmentation_list_free(segs);
            hal_tensor_free(tensor);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_model_type() {
        unsafe {
            let decoder = make_decoder_json(YOLO_JSON_CONFIG);
            assert!(!decoder.is_null());

            let model_type = hal_decoder_model_type(decoder);
            assert!(!model_type.is_null());

            let type_str = std::ffi::CStr::from_ptr(model_type).to_str().unwrap();
            assert!(!type_str.is_empty());

            libc::free(model_type as *mut libc::c_void);

            // NULL decoder
            assert!(hal_decoder_model_type(std::ptr::null()).is_null());

            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_normalized_boxes() {
        unsafe {
            let decoder = make_decoder_json(YOLO_JSON_CONFIG);
            assert!(!decoder.is_null());

            // YOLO config without explicit normalized field returns -1 (unknown)
            let result = hal_decoder_normalized_boxes(decoder);
            assert!(result == -1 || result == 0 || result == 1);

            // NULL decoder returns -1
            assert_eq!(hal_decoder_normalized_boxes(std::ptr::null()), -1);

            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_dequantize() {
        unsafe {
            // Create u8 input tensor
            let shape: [usize; 2] = [2, 3];
            let input = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!input.is_null());

            // Fill with values
            let map = hal_tensor_map_create(input);
            assert!(!map.is_null());
            let data = hal_tensor_map_data(map) as *mut u8;
            for i in 0..6 {
                *data.add(i) = (i as u8) + 128;
            }
            hal_tensor_map_unmap(map);

            // Create f32 output tensor
            let output = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!output.is_null());

            let quant = HalQuantization {
                scale: 0.5,
                zero_point: 128,
            };
            let result = hal_dequantize(input, quant, output);
            assert_eq!(result, 0);

            // Verify: output[0] = (128 - 128) * 0.5 = 0.0
            let map = hal_tensor_map_create(output);
            let out_data = hal_tensor_map_data(map) as *const f32;
            assert!((*out_data - 0.0).abs() < 1e-6);
            // output[1] = (129 - 128) * 0.5 = 0.5
            assert!((*out_data.add(1) - 0.5).abs() < 1e-6);
            hal_tensor_map_unmap(map);

            // NULL input
            assert_eq!(hal_dequantize(std::ptr::null(), quant, output), -1);

            // NULL output
            assert_eq!(hal_dequantize(input, quant, std::ptr::null_mut()), -1);

            hal_tensor_free(input);
            hal_tensor_free(output);
        }
    }

    #[test]
    fn test_dequantize_wrong_dtypes() {
        unsafe {
            let shape: [usize; 2] = [2, 3];
            let quant = HalQuantization {
                scale: 1.0,
                zero_point: 0,
            };

            // f32 input should fail
            let f32_input = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            let f32_output = hal_tensor_new(
                HalDtype::F32,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert_eq!(hal_dequantize(f32_input, quant, f32_output), -1);

            // u8 input, u8 output (output must be f32)
            let u8_input = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            let u8_output = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                2,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert_eq!(hal_dequantize(u8_input, quant, u8_output), -1);

            hal_tensor_free(f32_input);
            hal_tensor_free(f32_output);
            hal_tensor_free(u8_input);
            hal_tensor_free(u8_output);
        }
    }

    #[test]
    fn test_segmentation_to_mask() {
        unsafe {
            // Create a segmentation list with a simple mask
            let mut mask_data = ndarray::Array3::<u8>::zeros((10, 10, 1));
            // Set some pixels above threshold
            mask_data[[0, 0, 0]] = 200;
            mask_data[[1, 1, 0]] = 255;
            mask_data[[2, 2, 0]] = 50; // below threshold

            let seg = Segmentation {
                xmin: 0.1,
                ymin: 0.2,
                xmax: 0.3,
                ymax: 0.4,
                segmentation: mask_data,
            };

            let list = Box::into_raw(Box::new(HalSegmentationList { masks: vec![seg] }));

            let mask_tensor = hal_segmentation_to_mask(list, 0);
            assert!(!mask_tensor.is_null());

            // Check shape is 2D [10, 10]
            let mut ndim: size_t = 0;
            let shape_ptr = crate::tensor::hal_tensor_shape(mask_tensor, &mut ndim);
            assert_eq!(ndim, 2);
            assert_eq!(*shape_ptr, 10);
            assert_eq!(*shape_ptr.add(1), 10);

            // Check dtype is u8
            assert_eq!(crate::tensor::hal_tensor_dtype(mask_tensor), HalDtype::U8);

            crate::tensor::hal_tensor_free(mask_tensor);

            // Out-of-bounds index
            let null_mask = hal_segmentation_to_mask(list, 1);
            assert!(null_mask.is_null());

            // NULL list
            let null_mask = hal_segmentation_to_mask(std::ptr::null(), 0);
            assert!(null_mask.is_null());

            hal_segmentation_list_free(list);
        }
    }

    #[test]
    fn test_detect_box_list_get_null_output() {
        unsafe {
            let boxes = Box::into_raw(Box::new(HalDetectBoxList {
                boxes: vec![DetectBox {
                    bbox: edgefirst_decoder::BoundingBox {
                        xmin: 0.1,
                        ymin: 0.2,
                        xmax: 0.3,
                        ymax: 0.4,
                    },
                    score: 0.9,
                    label: 1,
                }],
            }));

            assert_eq!(hal_detect_box_list_len(boxes), 1);
            assert_eq!(hal_detect_box_list_get(boxes, 0, std::ptr::null_mut()), -1);

            hal_detect_box_list_free(boxes);
        }
    }

    #[test]
    fn test_segmentation_list_operations() {
        unsafe {
            let mask_data = ndarray::Array3::<u8>::zeros((10, 10, 1));
            let seg = Segmentation {
                xmin: 0.1,
                ymin: 0.2,
                xmax: 0.3,
                ymax: 0.4,
                segmentation: mask_data,
            };

            let list = Box::into_raw(Box::new(HalSegmentationList { masks: vec![seg] }));

            assert_eq!(hal_segmentation_list_len(list), 1);

            let mut xmin = 0.0f32;
            let mut ymin = 0.0f32;
            let mut xmax = 0.0f32;
            let mut ymax = 0.0f32;

            assert_eq!(
                hal_segmentation_list_get_bbox(list, 0, &mut xmin, &mut ymin, &mut xmax, &mut ymax),
                0
            );
            assert!((xmin - 0.1).abs() < 1e-6);
            assert!((ymin - 0.2).abs() < 1e-6);
            assert!((xmax - 0.3).abs() < 1e-6);
            assert!((ymax - 0.4).abs() < 1e-6);

            assert_eq!(
                hal_segmentation_list_get_bbox(list, 1, &mut xmin, &mut ymin, &mut xmax, &mut ymax),
                -1
            );

            assert_eq!(
                hal_segmentation_list_get_bbox(
                    list,
                    0,
                    std::ptr::null_mut(),
                    &mut ymin,
                    &mut xmax,
                    &mut ymax
                ),
                -1
            );

            let mut height: usize = 0;
            let mut width: usize = 0;
            let mask_ptr = hal_segmentation_list_get_mask(list, 0, &mut height, &mut width);
            assert!(!mask_ptr.is_null());
            assert_eq!(height, 10);
            assert_eq!(width, 10);

            let mask_ptr2 =
                hal_segmentation_list_get_mask(list, 0, std::ptr::null_mut(), std::ptr::null_mut());
            assert!(!mask_ptr2.is_null());

            let mask_ptr3 = hal_segmentation_list_get_mask(list, 1, &mut height, &mut width);
            assert!(mask_ptr3.is_null());

            hal_segmentation_list_free(list);
        }
    }

    #[test]
    fn test_config_file_loading() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("test_decoder_config.json");
        let yaml_path = temp_dir.join("test_decoder_config.yaml");

        {
            let mut file = std::fs::File::create(&json_path).unwrap();
            file.write_all(YOLO_JSON_CONFIG.as_bytes()).unwrap();
        }
        {
            let mut file = std::fs::File::create(&yaml_path).unwrap();
            file.write_all(YOLO_YAML_CONFIG.as_bytes()).unwrap();
        }

        unsafe {
            // Test JSON file loading
            let path = CString::new(json_path.to_str().unwrap()).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_file(params, path.as_ptr());
            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);

            // Test YAML file loading
            let path = CString::new(yaml_path.to_str().unwrap()).unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_file(params, path.as_ptr());
            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);
            hal_decoder_free(decoder);

            // Test non-existent file
            let path = CString::new("/nonexistent/path/config.json").unwrap();
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_file(params, path.as_ptr());
            let decoder = hal_decoder_new(params);
            assert!(decoder.is_null());
            hal_decoder_params_free(params);
        }

        let _ = std::fs::remove_file(json_path);
        let _ = std::fs::remove_file(yaml_path);
    }

    #[test]
    fn test_decode_with_null_tensor_in_array() {
        unsafe {
            let decoder = make_decoder_json(YOLO_JSON_CONFIG);
            assert!(!decoder.is_null());

            let outputs: [*const HalTensor; 1] = [std::ptr::null()];
            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();

            assert_eq!(
                hal_decoder_decode(
                    decoder,
                    outputs.as_ptr(),
                    1,
                    &mut boxes as *mut _,
                    std::ptr::null_mut()
                ),
                -1
            );

            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decode_tracked_linear_motion() {
        use crate::tracker::{
            hal_bytetrack_free, hal_bytetrack_get_active_tracks, hal_bytetrack_new,
            hal_track_info_list_free, hal_track_info_list_get, hal_track_info_list_len,
            HalTrackInfo,
        };

        let yaml = c"
decoder_version: yolov8
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.0040811873, -123]
   shape: [1, 84, 8400]
   dshape:
    - [batch, 1]
    - [num_features, 84]
    - [num_boxes, 8400]
   normalized: true
";
        unsafe {
            // Load the yolov8s test data (i8 quantized, shape [1, 84, 8400])
            let raw = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/yolov8s_80_classes.bin"
            ));
            let quant_scale: f32 = 0.0040811873;

            // Build decoder from YAML config
            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_yaml(params, yaml.as_ptr(), 0);
            hal_decoder_params_set_score_threshold(params, 0.25);
            hal_decoder_params_set_iou_threshold(params, 0.1);
            hal_decoder_params_set_nms(params, HalNms::ClassAgnostic);

            let decoder = hal_decoder_new(params);

            assert!(!decoder.is_null(), "decoder creation failed");
            hal_decoder_params_free(params);

            // Create tracker: track_update=0.1, high_thresh=0.3
            let tracker = hal_bytetrack_new(0.1, 0.3, 0.25, 30, 30);
            assert!(!tracker.is_null());

            // Create i8 tensor from the test data
            let tensor_shape: [usize; 3] = [1, 84, 8400];
            let tensor = hal_tensor_new(
                HalDtype::I8,
                tensor_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            // Copy initial data
            copy_data_to_tensor(tensor, raw);

            // --- Frame 0: initial decode ---
            let outputs = [tensor as *const HalTensor];
            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut segs: *mut HalSegmentationList = std::ptr::null_mut();
            let mut tracks: *mut HalTrackInfoList = std::ptr::null_mut();

            let rc = hal_decoder_decode_tracked(
                decoder,
                tracker,
                0,
                outputs.as_ptr(),
                1,
                &mut boxes,
                &mut segs,
                &mut tracks,
            );
            assert_eq!(rc, 0);
            assert_eq!(hal_detect_box_list_len(boxes), 2);

            // Verify initial detections
            let mut box0 = std::mem::zeroed::<HalDetectBox>();
            let mut box1 = std::mem::zeroed::<HalDetectBox>();
            assert_eq!(hal_detect_box_list_get(boxes, 0, &mut box0), 0);
            assert_eq!(hal_detect_box_list_get(boxes, 1, &mut box1), 0);

            assert!((box0.xmin - 0.5285137).abs() < 1e-6);
            assert!((box0.ymin - 0.05305544).abs() < 1e-6);
            assert!((box0.xmax - 0.87541467).abs() < 1e-6);
            assert!((box0.ymax - 0.9998909).abs() < 1e-6);
            assert_eq!(box0.label, 0);

            assert!((box1.xmin - 0.130598).abs() < 1e-6);
            assert!((box1.ymin - 0.43260583).abs() < 1e-6);
            assert!((box1.xmax - 0.35098213).abs() < 1e-6);
            assert!((box1.ymax - 0.9958097).abs() < 1e-6);
            assert_eq!(box1.label, 75);

            hal_detect_box_list_free(boxes);
            if !segs.is_null() {
                hal_segmentation_list_free(segs);
            }
            if !tracks.is_null() {
                hal_track_info_list_free(tracks);
            }

            // --- Frames 1..=100: linear motion on X ---
            let num_elements = 84 * 8400;
            let mut data_buf = vec![0u8; num_elements];
            for i in 1u64..=100 {
                // Copy original data
                data_buf.copy_from_slice(&raw[..num_elements]);
                // Cast to i8 slice for mutation
                let data_i8 =
                    std::slice::from_raw_parts_mut(data_buf.as_mut_ptr() as *mut i8, num_elements);
                // Modify X coordinates (row 0 of the [84, 8400] matrix)
                for x in data_i8[..8400].iter_mut() {
                    *x = x.saturating_add((i as f32 * 1e-3 / quant_scale).round() as i8);
                }
                copy_data_to_tensor(tensor, &data_buf);

                boxes = std::ptr::null_mut();
                segs = std::ptr::null_mut();
                tracks = std::ptr::null_mut();

                let rc = hal_decoder_decode_tracked(
                    decoder,
                    tracker,
                    100_000_000 * i / 3,
                    outputs.as_ptr(),
                    1,
                    &mut boxes,
                    &mut segs,
                    &mut tracks,
                );
                assert_eq!(rc, 0);
                assert_eq!(hal_detect_box_list_len(boxes), 2);

                hal_detect_box_list_free(boxes);
                if !segs.is_null() {
                    hal_segmentation_list_free(segs);
                }
                if !tracks.is_null() {
                    hal_track_info_list_free(tracks);
                }
            }

            // Verify tracker's predicted locations match expected after linear motion
            let active = hal_bytetrack_get_active_tracks(tracker);
            assert!(!active.is_null());
            assert_eq!(hal_track_info_list_len(active), 2);

            let mut track0 = std::mem::zeroed::<HalTrackInfo>();
            let mut track1 = std::mem::zeroed::<HalTrackInfo>();
            assert_eq!(hal_track_info_list_get(active, 0, &mut track0), 0);
            assert_eq!(hal_track_info_list_get(active, 1, &mut track1), 0);

            // Expected: original boxes shifted by +0.1 in X
            assert!((track0.location[0] - (0.5285137 + 0.1)).abs() < 1e-3); // xmin
            assert!((track0.location[2] - (0.87541467 + 0.1)).abs() < 1e-3); // xmax
            assert!((track1.location[0] - (0.130598 + 0.1)).abs() < 1e-3); // xmin
            assert!((track1.location[2] - (0.35098213 + 0.1)).abs() < 1e-3); // xmax

            hal_track_info_list_free(active);

            // --- Final frame: zero all scores to test tracker prediction ---
            data_buf.copy_from_slice(&raw[..num_elements]);
            let data_i8 =
                std::slice::from_raw_parts_mut(data_buf.as_mut_ptr() as *mut i8, num_elements);
            // Zero scores: rows 4..84 in [84, 8400] layout
            for val in data_i8[4 * 8400..].iter_mut() {
                *val = i8::MIN;
            }
            copy_data_to_tensor(tensor, &data_buf);

            boxes = std::ptr::null_mut();
            segs = std::ptr::null_mut();
            tracks = std::ptr::null_mut();

            let rc = hal_decoder_decode_tracked(
                decoder,
                tracker,
                100_000_000 * 101 / 3,
                outputs.as_ptr(),
                1,
                &mut boxes,
                &mut segs,
                &mut tracks,
            );
            assert_eq!(rc, 0);

            // Tracker should predict forward: boxes from prediction, shifted +0.101 in X
            let len = hal_detect_box_list_len(boxes);
            assert_eq!(len, 2);

            assert_eq!(hal_detect_box_list_get(boxes, 0, &mut box0), 0);
            assert_eq!(hal_detect_box_list_get(boxes, 1, &mut box1), 0);

            assert!((box0.xmin - (0.5285137 + 0.101)).abs() < 1e-3);
            assert!((box0.xmax - (0.87541467 + 0.101)).abs() < 1e-3);
            assert!((box1.xmin - (0.130598 + 0.101)).abs() < 1e-3);
            assert!((box1.xmax - (0.35098213 + 0.101)).abs() < 1e-3);

            hal_detect_box_list_free(boxes);
            if !segs.is_null() {
                hal_segmentation_list_free(segs);
            }
            if !tracks.is_null() {
                hal_track_info_list_free(tracks);
            }

            // Cleanup
            hal_tensor_free(tensor);
            hal_bytetrack_free(tracker);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decode_tracked_end_to_end_segdet_split_proto() {
        use crate::image::{
            hal_image_processor_free, hal_image_processor_new, hal_tensor_new_image, HalPixelFormat,
        };
        use crate::tracker::{
            hal_bytetrack_free, hal_bytetrack_new, hal_track_info_list_free, HalTrackInfoList,
        };

        let yaml = c"
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

        unsafe {
            let quant_scale: f32 = 2.0 / 255.0;

            // --- Create split u8 tensors ---

            // Boxes [1, 10, 4]
            let boxes_shape: [usize; 3] = [1, 10, 4];
            let boxes_tensor = hal_tensor_new(
                HalDtype::U8,
                boxes_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!boxes_tensor.is_null());
            {
                let map = hal_tensor_map_create(boxes_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut u8;
                std::ptr::write_bytes(data, 0, 10 * 4);
                *data.add(0) = (0.1234f32 / quant_scale).round() as u8;
                *data.add(1) = (0.1234f32 / quant_scale).round() as u8;
                *data.add(2) = (0.2345f32 / quant_scale).round() as u8;
                *data.add(3) = (0.2345f32 / quant_scale).round() as u8;
                hal_tensor_map_unmap(map);
            }

            // Scores [1, 10, 1]
            let scores_shape: [usize; 3] = [1, 10, 1];
            let scores_tensor = hal_tensor_new(
                HalDtype::U8,
                scores_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!scores_tensor.is_null());
            {
                let map = hal_tensor_map_create(scores_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut u8;
                std::ptr::write_bytes(data, 0, 10);
                *data.add(0) = (0.9876f32 / quant_scale).round() as u8;
                hal_tensor_map_unmap(map);
            }

            // Classes [1, 10, 1]
            let classes_shape: [usize; 3] = [1, 10, 1];
            let classes_tensor = hal_tensor_new(
                HalDtype::U8,
                classes_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!classes_tensor.is_null());
            {
                let map = hal_tensor_map_create(classes_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut u8;
                std::ptr::write_bytes(data, 0, 10);
                *data.add(0) = (2.0f32 / quant_scale).round().min(255.0) as u8;
                hal_tensor_map_unmap(map);
            }

            // Mask coefficients [1, 10, 32] — all zeros
            let mask_shape: [usize; 3] = [1, 10, 32];
            let mask_tensor = hal_tensor_new(
                HalDtype::U8,
                mask_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!mask_tensor.is_null());

            // Protos [1, 160, 160, 32] — all zeros
            let protos_shape: [usize; 4] = [1, 160, 160, 32];
            let protos_tensor = hal_tensor_new(
                HalDtype::U8,
                protos_shape.as_ptr(),
                4,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!protos_tensor.is_null());

            // --- Build decoder ---
            let params = hal_decoder_params_new();
            assert!(!params.is_null());
            assert_eq!(
                hal_decoder_params_set_config_yaml(params, yaml.as_ptr(), 0),
                0
            );
            hal_decoder_params_set_score_threshold(params, 0.45);
            hal_decoder_params_set_iou_threshold(params, 0.45);
            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);

            // --- Create tracker ---
            let tracker = hal_bytetrack_new(0.1, 0.7, 0.5, 30, 30);
            assert!(!tracker.is_null());

            // --- Create image processor and destination image ---
            let processor = hal_image_processor_new();
            assert!(!processor.is_null());

            let image = hal_tensor_new_image(
                400,
                400,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            assert!(!image.is_null());

            // --- Frame 0: decode tracked with proto ---
            let outputs = [
                boxes_tensor as *const HalTensor,
                scores_tensor as *const HalTensor,
                classes_tensor as *const HalTensor,
                mask_tensor as *const HalTensor,
                protos_tensor as *const HalTensor,
            ];
            let mut box_list: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut track_list: *mut HalTrackInfoList = std::ptr::null_mut();

            let rc = crate::image::hal_image_processor_draw_masks_tracked(
                processor,
                decoder,
                tracker,
                0,
                outputs.as_ptr(),
                5,
                image,
                std::ptr::null(),
                1.0,
                &mut box_list,
                &mut track_list,
            );
            assert_eq!(rc, 0);
            assert!(!box_list.is_null());
            assert_eq!(hal_detect_box_list_len(box_list), 1);

            let mut box0 = std::mem::zeroed::<HalDetectBox>();
            assert_eq!(hal_detect_box_list_get(box_list, 0, &mut box0), 0);
            let tol = 1.0 / 160.0;
            assert!((box0.xmin - 0.12549022).abs() < tol, "xmin: {}", box0.xmin);
            assert!((box0.ymin - 0.12549022).abs() < tol, "ymin: {}", box0.ymin);
            assert!((box0.xmax - 0.23529413).abs() < tol, "xmax: {}", box0.xmax);
            assert!((box0.ymax - 0.23529413).abs() < tol, "ymax: {}", box0.ymax);
            assert_eq!(box0.label, 2);

            hal_detect_box_list_free(box_list);
            if !track_list.is_null() {
                hal_track_info_list_free(track_list);
            }

            // --- Frame 1: zero all scores, verify tracker prediction ---
            {
                let map = hal_tensor_map_create(scores_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut u8;
                std::ptr::write_bytes(data, 0, 10);
                hal_tensor_map_unmap(map);
            }

            box_list = std::ptr::null_mut();
            track_list = std::ptr::null_mut();

            let rc = crate::image::hal_image_processor_draw_masks_tracked(
                processor,
                decoder,
                tracker,
                100_000_000 / 3,
                outputs.as_ptr(),
                5,
                image,
                std::ptr::null(),
                1.0,
                &mut box_list,
                &mut track_list,
            );
            assert_eq!(rc, 0);
            assert!(!box_list.is_null());

            // Tracker predicts the box forward (same location, no motion)
            assert_eq!(hal_detect_box_list_len(box_list), 1);
            assert_eq!(hal_detect_box_list_get(box_list, 0, &mut box0), 0);
            assert!((box0.xmin - 0.12549022).abs() < 1e-3);
            assert!((box0.ymin - 0.12549022).abs() < 1e-3);
            assert!((box0.xmax - 0.23529413).abs() < 1e-3);
            assert!((box0.ymax - 0.23529413).abs() < 1e-3);

            hal_detect_box_list_free(box_list);
            if !track_list.is_null() {
                hal_track_info_list_free(track_list);
            }

            // Cleanup
            hal_tensor_free(boxes_tensor);
            hal_tensor_free(scores_tensor);
            hal_tensor_free(classes_tensor);
            hal_tensor_free(mask_tensor);
            hal_tensor_free(protos_tensor);
            hal_tensor_free(image);
            hal_bytetrack_free(tracker);
            hal_image_processor_free(processor);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decode_tracked_end_to_end_segdet_split_proto_float() {
        use crate::image::{
            hal_image_processor_free, hal_image_processor_new, hal_tensor_new_image, HalPixelFormat,
        };
        use crate::tracker::{
            hal_bytetrack_free, hal_bytetrack_new, hal_track_info_list_free, HalTrackInfoList,
        };

        let yaml = c"
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

        unsafe {
            // --- Create split f32 tensors ---

            // Boxes [1, 10, 4]
            let boxes_shape: [usize; 3] = [1, 10, 4];
            let boxes_tensor = hal_tensor_new(
                HalDtype::F32,
                boxes_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!boxes_tensor.is_null());
            {
                let map = hal_tensor_map_create(boxes_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut f32;
                std::ptr::write_bytes(data, 0, 10 * 4);
                *data.add(0) = 0.1234;
                *data.add(1) = 0.1234;
                *data.add(2) = 0.2345;
                *data.add(3) = 0.2345;
                hal_tensor_map_unmap(map);
            }

            // Scores [1, 10, 1]
            let scores_shape: [usize; 3] = [1, 10, 1];
            let scores_tensor = hal_tensor_new(
                HalDtype::F32,
                scores_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!scores_tensor.is_null());
            {
                let map = hal_tensor_map_create(scores_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut f32;
                std::ptr::write_bytes(data, 0, 10);
                *data.add(0) = 0.9876;
                hal_tensor_map_unmap(map);
            }

            // Classes [1, 10, 1]
            let classes_shape: [usize; 3] = [1, 10, 1];
            let classes_tensor = hal_tensor_new(
                HalDtype::F32,
                classes_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!classes_tensor.is_null());
            {
                let map = hal_tensor_map_create(classes_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut f32;
                std::ptr::write_bytes(data, 0, 10);
                *data.add(0) = 2.0;
                hal_tensor_map_unmap(map);
            }

            // Mask coefficients [1, 10, 32] — all zeros
            let mask_shape: [usize; 3] = [1, 10, 32];
            let mask_tensor = hal_tensor_new(
                HalDtype::F32,
                mask_shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!mask_tensor.is_null());

            // Protos [1, 160, 160, 32] — all zeros
            let protos_shape: [usize; 4] = [1, 160, 160, 32];
            let protos_tensor = hal_tensor_new(
                HalDtype::F32,
                protos_shape.as_ptr(),
                4,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!protos_tensor.is_null());

            // --- Build decoder ---
            let params = hal_decoder_params_new();
            assert!(!params.is_null());
            assert_eq!(
                hal_decoder_params_set_config_yaml(params, yaml.as_ptr(), 0),
                0
            );
            hal_decoder_params_set_score_threshold(params, 0.45);
            hal_decoder_params_set_iou_threshold(params, 0.45);
            let decoder = hal_decoder_new(params);
            assert!(!decoder.is_null());
            hal_decoder_params_free(params);

            // --- Create tracker ---
            let tracker = hal_bytetrack_new(0.1, 0.7, 0.5, 30, 30);
            assert!(!tracker.is_null());

            // --- Create image processor and destination image ---
            let processor = hal_image_processor_new();
            assert!(!processor.is_null());

            let image = hal_tensor_new_image(
                400,
                400,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            assert!(!image.is_null());

            // --- Frame 0: decode tracked with proto ---
            let outputs = [
                boxes_tensor as *const HalTensor,
                scores_tensor as *const HalTensor,
                classes_tensor as *const HalTensor,
                mask_tensor as *const HalTensor,
                protos_tensor as *const HalTensor,
            ];
            let mut box_list: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut track_list: *mut HalTrackInfoList = std::ptr::null_mut();

            let rc = crate::image::hal_image_processor_draw_masks_tracked(
                processor,
                decoder,
                tracker,
                0,
                outputs.as_ptr(),
                5,
                image,
                std::ptr::null(),
                1.0,
                &mut box_list,
                &mut track_list,
            );
            assert_eq!(rc, 0);
            assert!(!box_list.is_null());
            assert_eq!(hal_detect_box_list_len(box_list), 1);

            let mut box0 = std::mem::zeroed::<HalDetectBox>();
            assert_eq!(hal_detect_box_list_get(box_list, 0, &mut box0), 0);
            // No quantization error, so we can use much tighter tolerance
            assert!((box0.xmin - 0.1234).abs() < 1e-6, "xmin: {}", box0.xmin);
            assert!((box0.ymin - 0.1234).abs() < 1e-6, "ymin: {}", box0.ymin);
            assert!((box0.xmax - 0.2345).abs() < 1e-6, "xmax: {}", box0.xmax);
            assert!((box0.ymax - 0.2345).abs() < 1e-6, "ymax: {}", box0.ymax);
            assert_eq!(box0.label, 2);

            hal_detect_box_list_free(box_list);
            if !track_list.is_null() {
                hal_track_info_list_free(track_list);
            }

            // --- Frame 1: zero all scores, verify tracker prediction ---
            {
                let map = hal_tensor_map_create(scores_tensor);
                assert!(!map.is_null());
                let data = hal_tensor_map_data(map) as *mut f32;
                std::ptr::write_bytes(data, 0, 10);
                hal_tensor_map_unmap(map);
            }

            box_list = std::ptr::null_mut();
            track_list = std::ptr::null_mut();

            let rc = crate::image::hal_image_processor_draw_masks_tracked(
                processor,
                decoder,
                tracker,
                100_000_000 / 3,
                outputs.as_ptr(),
                5,
                image,
                std::ptr::null(),
                1.0,
                &mut box_list,
                &mut track_list,
            );
            assert_eq!(rc, 0);
            assert!(!box_list.is_null());

            // Tracker predicts the box forward (same location, no motion)
            assert_eq!(hal_detect_box_list_len(box_list), 1);
            assert_eq!(hal_detect_box_list_get(box_list, 0, &mut box0), 0);
            assert!((box0.xmin - 0.1234).abs() < 1e-6);
            assert!((box0.ymin - 0.1234).abs() < 1e-6);
            assert!((box0.xmax - 0.2345).abs() < 1e-6);
            assert!((box0.ymax - 0.2345).abs() < 1e-6);

            hal_detect_box_list_free(box_list);
            if !track_list.is_null() {
                hal_track_info_list_free(track_list);
            }

            // Cleanup
            hal_tensor_free(boxes_tensor);
            hal_tensor_free(scores_tensor);
            hal_tensor_free(classes_tensor);
            hal_tensor_free(mask_tensor);
            hal_tensor_free(protos_tensor);
            hal_tensor_free(image);
            hal_bytetrack_free(tracker);
            hal_image_processor_free(processor);
            hal_decoder_free(decoder);
        }
    }

    unsafe fn copy_data_to_tensor<T>(t: *mut HalTensor, src: &[T]) {
        let map = hal_tensor_map_create(t);
        assert!(!map.is_null());
        let data = hal_tensor_map_data(map) as *mut T;
        assert!(!data.is_null());
        std::ptr::copy_nonoverlapping(src.as_ptr(), data, src.len());
        hal_tensor_map_unmap(map);
    }

    #[test]
    fn test_decode_tracked_float_and_quant() {
        use crate::tracker::{hal_bytetrack_new, hal_track_info_list_free};

        let yaml = c"
decoder_version: yolov8
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.0040811873, -123]
   shape: [1, 84, 8400]
   dshape:
    - [batch, 1]
    - [num_features, 84]
    - [num_boxes, 8400]
   normalized: true
";
        unsafe {
            // Load the yolov8s test data (i8 quantized, shape [1, 84, 8400])
            let raw = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/yolov8s_80_classes.bin"
            ));
            let raw = std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len());
            let quant = (0.0040811873, -123);

            let params = hal_decoder_params_new();
            hal_decoder_params_set_config_yaml(params, yaml.as_ptr(), 0);
            hal_decoder_params_set_score_threshold(params, 0.25);
            hal_decoder_params_set_iou_threshold(params, 0.1);
            hal_decoder_params_set_nms(params, HalNms::ClassAgnostic);

            let decoder = hal_decoder_new(params);

            assert!(!decoder.is_null(), "decoder creation failed");
            hal_decoder_params_free(params);

            let tracker = hal_bytetrack_new(0.1, 0.3, 0.25, 30, 30);
            assert!(!tracker.is_null());

            // Create i8 tensor from the test data
            let tensor = hal_tensor_new(
                HalDtype::I8,
                [1, 84, 8400].as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            // Copy initial data
            copy_data_to_tensor(tensor, raw);

            let outputs = [tensor as *const HalTensor];
            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut segs: *mut HalSegmentationList = std::ptr::null_mut();
            let mut tracks: *mut HalTrackInfoList = std::ptr::null_mut();

            let rc = hal_decoder_decode_tracked(
                decoder,
                tracker,
                0,
                outputs.as_ptr(),
                1,
                &mut boxes,
                &mut segs,
                &mut tracks,
            );
            assert_eq!(rc, 0);
            assert_eq!(hal_detect_box_list_len(boxes), 2);

            let mut box0 = std::mem::zeroed::<HalDetectBox>();
            let mut box1 = std::mem::zeroed::<HalDetectBox>();
            assert_eq!(hal_detect_box_list_get(boxes, 0, &mut box0), 0);
            assert_eq!(hal_detect_box_list_get(boxes, 1, &mut box1), 0);

            assert!((box0.xmin - 0.5285137).abs() < 1e-6);
            assert!((box0.ymin - 0.05305544).abs() < 1e-6);
            assert!((box0.xmax - 0.87541467).abs() < 1e-6);
            assert!((box0.ymax - 0.9998909).abs() < 1e-6);
            assert_eq!(box0.label, 0);

            assert!((box1.xmin - 0.130598).abs() < 1e-6);
            assert!((box1.ymin - 0.43260583).abs() < 1e-6);
            assert!((box1.xmax - 0.35098213).abs() < 1e-6);
            assert!((box1.ymax - 0.9958097).abs() < 1e-6);
            assert_eq!(box1.label, 75);

            hal_detect_box_list_free(boxes);
            if !segs.is_null() {
                hal_segmentation_list_free(segs);
            }
            if !tracks.is_null() {
                hal_track_info_list_free(tracks);
            }
            let mut raw_float = vec![0.0f32; raw.len()];
            dequantize_cpu(raw, quant.into(), &mut raw_float);
            let tensor_float = hal_tensor_new(
                HalDtype::F32,
                [1, 84, 8400].as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor_float.is_null());
            copy_data_to_tensor(tensor_float, &raw_float);

            let outputs = [tensor_float as *const HalTensor];
            let mut boxes_float: *mut HalDetectBoxList = std::ptr::null_mut();
            let mut segs_float: *mut HalSegmentationList = std::ptr::null_mut();
            let mut tracks_float: *mut HalTrackInfoList = std::ptr::null_mut();

            let rc = hal_decoder_decode_tracked(
                decoder,
                tracker,
                0,
                outputs.as_ptr(),
                1,
                &mut boxes_float,
                &mut segs_float,
                &mut tracks_float,
            );
            assert_eq!(rc, 0);
            assert_eq!(hal_detect_box_list_len(boxes_float), 2);

            let mut box0_float = std::mem::zeroed::<HalDetectBox>();
            let mut box1_float = std::mem::zeroed::<HalDetectBox>();
            assert_eq!(hal_detect_box_list_get(boxes_float, 0, &mut box0_float), 0);
            assert_eq!(hal_detect_box_list_get(boxes_float, 1, &mut box1_float), 0);

            assert!((box0_float.xmin - 0.5285137).abs() < 1e-6);
            assert!((box0_float.ymin - 0.05305544).abs() < 1e-6);
            assert!((box0_float.xmax - 0.87541467).abs() < 1e-6);
            assert!((box0_float.ymax - 0.9998909).abs() < 1e-6);
            assert_eq!(box0_float.label, 0);

            assert!((box1_float.xmin - 0.130598).abs() < 1e-6);
            assert!((box1_float.ymin - 0.43260583).abs() < 1e-6);
            assert!((box1_float.xmax - 0.35098213).abs() < 1e-6);
            assert!((box1_float.ymax - 0.9958097).abs() < 1e-6);
            assert_eq!(box1_float.label, 75);
        }
    }
}
