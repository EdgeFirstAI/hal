// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decoder C API - ML model output decoding.
//!
//! This module provides functions for decoding YOLO and ModelPack model outputs
//! into detection boxes and segmentation masks.

use crate::error::set_error;
use crate::tensor::HalTensor;
use crate::{check_null, check_null_ret_null, try_or_errno, try_or_null};
use edgefirst_decoder::{
    configs::Nms, Decoder, DecoderBuilder, DetectBox, Quantization, Segmentation,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use libc::{c_char, c_int, size_t};
use ndarray::ArrayViewD;
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

/// Decoder construction parameters.
///
/// Configures how ML model outputs are decoded into detection boxes and
/// segmentation masks. Obtain defaults with `hal_decoder_params_default()`,
/// set the desired fields, then pass to `hal_decoder_new()`.
///
/// @section dp_config Configuration Source (exactly one required)
///
/// Exactly one of `config_json`, `config_yaml`, or `config_file` must be
/// non-NULL. These are mutually exclusive; setting more than one causes
/// `hal_decoder_new()` to fail with EINVAL. The configuration describes the
/// model output layout so the decoder knows how to interpret raw tensors.
///
/// @section dp_thresholds Threshold Tuning
///
/// `score_threshold` and `iou_threshold` control post-processing filtering:
/// - **score_threshold** (default 0.5): detections with confidence below this
///   value are discarded. Lower values yield more detections (higher recall),
///   higher values yield fewer but more confident detections (higher precision).
/// - **iou_threshold** (default 0.5): during NMS, box pairs with
///   Intersection-over-Union above this value are candidates for suppression.
///   Lower values suppress more aggressively.
///
/// @section dp_usage Basic Usage
/// @code{.c}
/// // Minimal: create a decoder from an inline JSON config
/// struct hal_decoder_params params = hal_decoder_params_default();
/// params.config_json = "{\"outputs\":[{\"decoder\":\"ultralytics\","
///                       "\"type\":\"detection\",\"shape\":[1,84,8400],"
///                       "\"dshape\":[[\"batch\",1],[\"num_features\",84],"
///                       "[\"num_boxes\",8400]]}],\"nms\":\"class_aware\"}";
/// params.score_threshold = 0.25f;
/// struct hal_decoder* decoder = hal_decoder_new(&params);
/// @endcode
///
/// @section dp_file Loading from a File
/// @code{.c}
/// // Load decoder config from an edgefirst.json or edgefirst.yaml file
/// struct hal_decoder_params params = hal_decoder_params_default();
/// params.config_file = "/path/to/edgefirst.json";
/// params.score_threshold = 0.3f;
/// params.iou_threshold  = 0.45f;
/// params.nms = HAL_NMS_CLASS_AWARE;
/// struct hal_decoder* decoder = hal_decoder_new(&params);
/// @endcode
///
/// @section dp_metadata EdgeFirst Model Metadata
///
/// EdgeFirst models ship with an `edgefirst.json` (or `edgefirst.yaml`)
/// metadata file that fully describes the model's output layout, decoder
/// type, and recommended post-processing settings. Pass this file directly
/// via `config_file` to auto-configure the decoder:
///
/// @code{.c}
/// // Auto-configure from EdgeFirst model metadata.
/// // The metadata file contains the outputs array, decoder type
/// // (ultralytics / modelpack), shapes, and NMS settings.
/// //
/// // Example edgefirst.yaml:
/// //   outputs:
/// //     - decoder: ultralytics
/// //       type: detection
/// //       shape: [1, 84, 8400]
/// //       dshape: [[batch, 1], [num_features, 84], [num_boxes, 8400]]
/// //   nms: class_agnostic
/// //   validation:
/// //     iou: 0.7
/// //     score: 0.001
/// //
/// struct hal_decoder_params params = hal_decoder_params_default();
/// params.config_file = "/models/yolov8n/edgefirst.yaml";
///
/// // Optionally override the metadata's recommended thresholds
/// // for your application's precision/recall needs:
/// params.score_threshold = 0.4f;
///
/// struct hal_decoder* decoder = hal_decoder_new(&params);
/// if (!decoder) {
///     fprintf(stderr, "decoder init failed: %s\n", strerror(errno));
/// }
/// @endcode
///
/// @see hal_decoder_params_default, hal_decoder_new, hal_nms
#[repr(C)]
pub struct HalDecoderParams {
    /// JSON configuration string (NUL-terminated).
    ///
    /// Mutually exclusive with `config_yaml` and `config_file`.
    /// The string is copied internally; the caller may free the buffer
    /// after `hal_decoder_new()` returns.
    ///
    /// This should contain the EdgeFirst model metadata JSON, or at minimum
    /// the `outputs` array describing the model's output tensors. Example:
    /// @code{.c}
    /// params.config_json = "{\"outputs\":[{\"decoder\":\"ultralytics\","
    ///                       "\"type\":\"detection\",\"shape\":[1,84,8400],"
    ///                       "\"dshape\":[[\"batch\",1],[\"num_features\",84],"
    ///                       "[\"num_boxes\",8400]]}]}";
    /// @endcode
    pub config_json: *const c_char,
    /// YAML configuration string (NUL-terminated).
    ///
    /// Mutually exclusive with `config_json` and `config_file`.
    /// The string is copied internally; the caller may free the buffer
    /// after `hal_decoder_new()` returns.
    pub config_yaml: *const c_char,
    /// Path to a configuration file (NUL-terminated).
    ///
    /// Mutually exclusive with `config_json` and `config_yaml`.
    /// The file is read and the path string copied during `hal_decoder_new()`;
    /// both may be freed after the call returns.
    ///
    /// The format is auto-detected: files ending in `.json` or starting with
    /// `{` are parsed as JSON; everything else is parsed as YAML. This makes
    /// it straightforward to point at an EdgeFirst model's `edgefirst.json`
    /// or `edgefirst.yaml` metadata file.
    pub config_file: *const c_char,
    /// Score threshold for filtering detections (default: 0.5).
    ///
    /// Detections with confidence scores below this value are discarded
    /// before NMS. Range: 0.0 to 1.0.
    ///
    /// Typical values:
    /// - 0.001 -- 0.01: evaluation / mAP benchmarking (keep nearly all boxes)
    /// - 0.25  -- 0.5:  general-purpose inference
    /// - 0.5   -- 0.8:  high-confidence-only applications
    pub score_threshold: f32,
    /// IOU (Intersection-over-Union) threshold for NMS (default: 0.5).
    ///
    /// During NMS, if two boxes overlap with IOU above this threshold the
    /// lower-scoring box is suppressed. Range: 0.0 to 1.0.
    ///
    /// Typical values:
    /// - 0.45 -- 0.5: standard object detection
    /// - 0.6  -- 0.7: when objects frequently overlap (e.g. dense crowds)
    pub iou_threshold: f32,
    /// NMS (Non-Maximum Suppression) mode (default: HAL_NMS_CLASS_AGNOSTIC).
    ///
    /// @see hal_nms for available modes and when to use each one.
    pub nms: HalNms,
}

/// Opaque decoder type.
pub struct HalDecoder {
    inner: Decoder,
}

/// List of detection boxes.
pub struct HalDetectBoxList {
    pub(crate) boxes: Vec<DetectBox>,
}

/// List of segmentation results.
pub struct HalSegmentationList {
    masks: Vec<Segmentation>,
}

// ============================================================================
// Decoder Params Functions
// ============================================================================

/// Create default decoder parameters.
///
/// Returns a `hal_decoder_params` struct initialized with safe defaults:
///
/// | Field | Default |
/// |-------|---------|
/// | config_json | NULL |
/// | config_yaml | NULL |
/// | config_file | NULL |
/// | score_threshold | 0.5 |
/// | iou_threshold | 0.5 |
/// | nms | HAL_NMS_CLASS_AGNOSTIC |
///
/// The caller must set exactly one configuration source (`config_json`,
/// `config_yaml`, or `config_file`) before passing to `hal_decoder_new()`.
/// All other fields may be left at their defaults or overridden.
///
/// @return Default decoder parameters (by value)
///
/// @par Example
/// @code{.c}
/// struct hal_decoder_params params = hal_decoder_params_default();
/// params.config_file = "edgefirst.yaml";
/// params.score_threshold = 0.3f;
/// struct hal_decoder* decoder = hal_decoder_new(&params);
/// @endcode
#[no_mangle]
pub extern "C" fn hal_decoder_params_default() -> HalDecoderParams {
    HalDecoderParams {
        config_json: std::ptr::null(),
        config_yaml: std::ptr::null(),
        config_file: std::ptr::null(),
        score_threshold: 0.5,
        iou_threshold: 0.5,
        nms: HalNms::ClassAgnostic,
    }
}

/// Create a new decoder from parameters.
///
/// Validates the parameters and constructs a decoder ready for use with
/// `hal_decoder_decode_float()`. Exactly one of `params->config_json`,
/// `params->config_yaml`, or `params->config_file` must be non-NULL.
///
/// All strings are copied internally; the caller may free their buffers
/// immediately after this call returns.
///
/// @param params Pointer to decoder parameters (must not be NULL)
/// @return New decoder handle on success, NULL on error (errno set)
///
/// @par Errors (errno):
/// - EINVAL: NULL params, no config source set, or multiple config sources set
/// - ENOENT: Config file path does not exist
/// - EIO: Config file could not be read
/// - EBADMSG: Configuration is syntactically valid but semantically invalid
///   (e.g. missing required `outputs` array, unknown decoder type)
///
/// @par Example
/// @code{.c}
/// struct hal_decoder_params params = hal_decoder_params_default();
/// params.config_file = "/models/yolov8n/edgefirst.yaml";
/// params.score_threshold = 0.25f;
/// params.iou_threshold  = 0.45f;
/// params.nms = HAL_NMS_CLASS_AWARE;
///
/// struct hal_decoder* decoder = hal_decoder_new(&params);
/// if (!decoder) {
///     fprintf(stderr, "decoder: %s\n", strerror(errno));
///     return -1;
/// }
///
/// // ... use decoder with hal_decoder_decode_float() ...
///
/// hal_decoder_free(decoder);
/// @endcode
///
/// @see hal_decoder_params_default, hal_decoder_free, hal_decoder_decode_float
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_new(params: *const HalDecoderParams) -> *mut HalDecoder {
    check_null_ret_null!(params);

    let p = &*params;

    let has_json = !p.config_json.is_null();
    let has_yaml = !p.config_yaml.is_null();
    let has_file = !p.config_file.is_null();

    // Exactly one config source must be set
    let config_count = has_json as u8 + has_yaml as u8 + has_file as u8;
    if config_count != 1 {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return std::ptr::null_mut();
    }

    let mut builder = DecoderBuilder::new()
        .with_score_threshold(p.score_threshold)
        .with_iou_threshold(p.iou_threshold)
        .with_nms(p.nms.into());

    if has_json {
        let json = match CStr::from_ptr(p.config_json).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return std::ptr::null_mut();
            }
        };
        builder = builder.with_config_json_str(json);
    } else if has_yaml {
        let yaml = match CStr::from_ptr(p.config_yaml).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return std::ptr::null_mut();
            }
        };
        builder = builder.with_config_yaml_str(yaml);
    } else {
        let path_str = match CStr::from_ptr(p.config_file).to_str() {
            Ok(s) => s,
            Err(_) => {
                errno::set_errno(errno::Errno(libc::EINVAL));
                return std::ptr::null_mut();
            }
        };

        let content = match std::fs::read_to_string(path_str) {
            Ok(c) => c,
            Err(e) => {
                errno::set_errno(errno::Errno(
                    if e.kind() == std::io::ErrorKind::NotFound {
                        libc::ENOENT
                    } else {
                        libc::EIO
                    },
                ));
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

/// Decode model outputs into detection boxes.
///
/// This is a simplified API for detection-only models with float outputs.
///
/// @param decoder Decoder handle
/// @param outputs Array of output tensor pointers
/// @param num_outputs Number of output tensors
/// @param out_boxes Output parameter for detection box list (caller must free)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL decoder/outputs/out_boxes)
/// - EIO: Decoding failed
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_decode_float(
    decoder: *const HalDecoder,
    outputs: *const *const HalTensor,
    num_outputs: size_t,
    out_boxes: *mut *mut HalDetectBoxList,
) -> c_int {
    check_null!(decoder, outputs, out_boxes);

    if num_outputs == 0 {
        return set_error(libc::EINVAL);
    }

    let outputs_slice = std::slice::from_raw_parts(outputs, num_outputs);

    // First, collect all tensor maps to keep them alive
    let mut maps = Vec::with_capacity(num_outputs);
    for &tensor_ptr in outputs_slice {
        if tensor_ptr.is_null() {
            return set_error(libc::EINVAL);
        }

        let tensor = &*tensor_ptr;

        // Only handle f32 tensors for this API
        match tensor {
            HalTensor::F32(t) => {
                let map = try_or_errno!(t.map(), libc::EIO);
                maps.push(map);
            }
            _ => return set_error(libc::EINVAL),
        }
    }

    // Now build views from the maps (maps remain alive)
    let mut views: Vec<ArrayViewD<'_, f32>> = Vec::with_capacity(num_outputs);
    for map in &maps {
        let shape = map.shape().to_vec();
        let slice = map.as_slice();
        let view = try_or_errno!(
            ArrayViewD::from_shape(shape.as_slice(), slice),
            libc::EINVAL
        );
        views.push(view);
    }

    // Decode
    let mut boxes: Vec<DetectBox> = Vec::with_capacity(100);
    let mut masks: Vec<Segmentation> = Vec::new();

    try_or_errno!(
        (*decoder)
            .inner
            .decode_float(&views, &mut boxes, &mut masks),
        libc::EIO
    );

    *out_boxes = Box::into_raw(Box::new(HalDetectBoxList { boxes }));
    0
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

#[cfg(test)]
mod tests {
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

    /// Helper: create a decoder from JSON config with optional overrides.
    unsafe fn make_decoder_json(json: &str) -> *mut HalDecoder {
        let config = CString::new(json).unwrap();
        let mut params = hal_decoder_params_default();
        params.config_json = config.as_ptr();
        hal_decoder_new(&params)
    }

    #[test]
    fn test_params_default() {
        let params = hal_decoder_params_default();
        assert!(params.config_json.is_null());
        assert!(params.config_yaml.is_null());
        assert!(params.config_file.is_null());
        assert!((params.score_threshold - 0.5).abs() < f32::EPSILON);
        assert!((params.iou_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.nms, HalNms::ClassAgnostic);
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
            let mut params = hal_decoder_params_default();
            params.config_json = config.as_ptr();

            let decoder = hal_decoder_new(&params);
            assert!(!decoder.is_null());
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_new_with_yaml() {
        unsafe {
            let config = CString::new(YOLO_YAML_CONFIG).unwrap();
            let mut params = hal_decoder_params_default();
            params.config_yaml = config.as_ptr();

            let decoder = hal_decoder_new(&params);
            assert!(!decoder.is_null());
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decoder_new_with_thresholds() {
        unsafe {
            let config = CString::new(YOLO_JSON_CONFIG).unwrap();
            let mut params = hal_decoder_params_default();
            params.config_json = config.as_ptr();
            params.score_threshold = 0.3;
            params.iou_threshold = 0.45;
            params.nms = HalNms::ClassAware;

            let decoder = hal_decoder_new(&params);
            assert!(!decoder.is_null());
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
            // All config pointers NULL → EINVAL
            let params = hal_decoder_params_default();
            let decoder = hal_decoder_new(&params);
            assert!(decoder.is_null());
        }
    }

    #[test]
    fn test_decoder_new_multiple_configs() {
        unsafe {
            let json = CString::new(YOLO_JSON_CONFIG).unwrap();
            let yaml = CString::new(YOLO_YAML_CONFIG).unwrap();

            let mut params = hal_decoder_params_default();
            params.config_json = json.as_ptr();
            params.config_yaml = yaml.as_ptr();

            let decoder = hal_decoder_new(&params);
            assert!(decoder.is_null());
        }
    }

    #[test]
    fn test_decode_float_null_parameters() {
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
                hal_decoder_decode_float(
                    std::ptr::null(),
                    outputs.as_ptr(),
                    1,
                    &mut boxes as *mut _
                ),
                -1
            );

            // NULL outputs
            assert_eq!(
                hal_decoder_decode_float(decoder, std::ptr::null(), 1, &mut boxes as *mut _),
                -1
            );

            // NULL out_boxes
            assert_eq!(
                hal_decoder_decode_float(decoder, outputs.as_ptr(), 1, std::ptr::null_mut()),
                -1
            );

            // Zero num_outputs
            assert_eq!(
                hal_decoder_decode_float(decoder, outputs.as_ptr(), 0, &mut boxes as *mut _),
                -1
            );

            hal_tensor_free(tensor);
            hal_decoder_free(decoder);
        }
    }

    #[test]
    fn test_decode_float_wrong_dtype() {
        unsafe {
            let decoder = make_decoder_json(YOLO_JSON_CONFIG);
            assert!(!decoder.is_null());

            let shape: [usize; 3] = [1, 84, 8400];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());
            let outputs = [tensor as *const HalTensor];
            let mut boxes: *mut HalDetectBoxList = std::ptr::null_mut();

            assert_eq!(
                hal_decoder_decode_float(decoder, outputs.as_ptr(), 1, &mut boxes as *mut _),
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
            let mut params = hal_decoder_params_default();
            params.config_json = config.as_ptr();
            params.score_threshold = 0.01;
            let decoder = hal_decoder_new(&params);
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

            let result =
                hal_decoder_decode_float(decoder, outputs.as_ptr(), 1, &mut boxes as *mut _);
            assert_eq!(result, 0);
            assert!(!boxes.is_null());

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
            hal_tensor_free(tensor);
            hal_decoder_free(decoder);
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
            let mut params = hal_decoder_params_default();
            params.config_file = path.as_ptr();
            let decoder = hal_decoder_new(&params);
            assert!(!decoder.is_null());
            hal_decoder_free(decoder);

            // Test YAML file loading
            let path = CString::new(yaml_path.to_str().unwrap()).unwrap();
            let mut params = hal_decoder_params_default();
            params.config_file = path.as_ptr();
            let decoder = hal_decoder_new(&params);
            assert!(!decoder.is_null());
            hal_decoder_free(decoder);

            // Test non-existent file
            let path = CString::new("/nonexistent/path/config.json").unwrap();
            let mut params = hal_decoder_params_default();
            params.config_file = path.as_ptr();
            let decoder = hal_decoder_new(&params);
            assert!(decoder.is_null());
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
                hal_decoder_decode_float(decoder, outputs.as_ptr(), 1, &mut boxes as *mut _),
                -1
            );

            hal_decoder_free(decoder);
        }
    }
}
