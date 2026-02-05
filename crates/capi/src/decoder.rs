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
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalNms {
    /// Class-agnostic NMS: suppress overlapping boxes regardless of class
    ClassAgnostic = 0,
    /// Class-aware NMS: only suppress boxes with the same class
    ClassAware = 1,
    /// No NMS (for end-to-end models with embedded NMS)
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

/// Opaque decoder builder type.
pub struct HalDecoderBuilder {
    inner: DecoderBuilder,
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
// Decoder Builder Functions
// ============================================================================

/// Create a new decoder builder.
///
/// The builder starts with default values:
/// - Score threshold: 0.5
/// - IOU threshold: 0.5
/// - NMS: Class-agnostic
///
/// A configuration must be set before building the decoder.
///
/// @return New decoder builder handle on success, NULL on error
/// @par Errors (errno):
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub extern "C" fn hal_decoder_builder_new() -> *mut HalDecoderBuilder {
    Box::into_raw(Box::new(HalDecoderBuilder {
        inner: DecoderBuilder::new(),
    }))
}

/// Set decoder configuration from a file (JSON or YAML).
///
/// @param builder Decoder builder handle
/// @param path Path to configuration file
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder/path)
/// - ENOENT: File not found
/// - EBADMSG: Failed to parse configuration file
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_config_file(
    builder: *mut HalDecoderBuilder,
    path: *const c_char,
) -> c_int {
    check_null!(builder, path);

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return set_error(libc::EINVAL),
    };

    let content = match std::fs::read_to_string(path_str) {
        Ok(c) => c,
        Err(e) => {
            return set_error(if e.kind() == std::io::ErrorKind::NotFound {
                libc::ENOENT
            } else {
                libc::EIO
            })
        }
    };

    // Try to detect format by extension or content
    let is_json = path_str.ends_with(".json") || content.trim_start().starts_with('{');

    if is_json {
        (*builder).inner = std::mem::take(&mut (*builder).inner).with_config_json_str(content);
    } else {
        (*builder).inner = std::mem::take(&mut (*builder).inner).with_config_yaml_str(content);
    }

    0
}

/// Set decoder configuration from a JSON string.
///
/// @param builder Decoder builder handle
/// @param json_str JSON configuration string
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder/json_str)
/// - EBADMSG: Invalid JSON format
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_config_json(
    builder: *mut HalDecoderBuilder,
    json_str: *const c_char,
) -> c_int {
    check_null!(builder, json_str);

    let json = match CStr::from_ptr(json_str).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return set_error(libc::EINVAL),
    };

    (*builder).inner = std::mem::take(&mut (*builder).inner).with_config_json_str(json);
    0
}

/// Set decoder configuration from a YAML string.
///
/// @param builder Decoder builder handle
/// @param yaml_str YAML configuration string
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder/yaml_str)
/// - EBADMSG: Invalid YAML format
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_config_yaml(
    builder: *mut HalDecoderBuilder,
    yaml_str: *const c_char,
) -> c_int {
    check_null!(builder, yaml_str);

    let yaml = match CStr::from_ptr(yaml_str).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return set_error(libc::EINVAL),
    };

    (*builder).inner = std::mem::take(&mut (*builder).inner).with_config_yaml_str(yaml);
    0
}

/// Set the score threshold for filtering detections.
///
/// Detections with scores below this threshold are discarded.
///
/// @param builder Decoder builder handle
/// @param threshold Score threshold (0.0 to 1.0)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_score_threshold(
    builder: *mut HalDecoderBuilder,
    threshold: f32,
) -> c_int {
    check_null!(builder);
    (*builder).inner = std::mem::take(&mut (*builder).inner).with_score_threshold(threshold);
    0
}

/// Set the IOU threshold for NMS.
///
/// Detection pairs with IOU above this threshold are candidates for suppression.
///
/// @param builder Decoder builder handle
/// @param threshold IOU threshold (0.0 to 1.0)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_iou_threshold(
    builder: *mut HalDecoderBuilder,
    threshold: f32,
) -> c_int {
    check_null!(builder);
    (*builder).inner = std::mem::take(&mut (*builder).inner).with_iou_threshold(threshold);
    0
}

/// Set the NMS (Non-Maximum Suppression) mode.
///
/// @param builder Decoder builder handle
/// @param nms NMS mode (HAL_NMS_*)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_with_nms(
    builder: *mut HalDecoderBuilder,
    nms: HalNms,
) -> c_int {
    check_null!(builder);
    (*builder).inner = std::mem::take(&mut (*builder).inner).with_nms(nms.into());
    0
}

/// Build the decoder from the builder configuration.
///
/// Consumes the builder - it cannot be used after this call.
///
/// @param builder Decoder builder handle (consumed)
/// @return New decoder handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL builder) or no configuration set
/// - EBADMSG: Invalid configuration
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_build(
    builder: *mut HalDecoderBuilder,
) -> *mut HalDecoder {
    check_null_ret_null!(builder);

    let boxed = Box::from_raw(builder);
    let decoder = try_or_null!(boxed.inner.build(), libc::EBADMSG);

    Box::into_raw(Box::new(HalDecoder { inner: decoder }))
}

/// Free a decoder builder.
///
/// @param builder Decoder builder handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_decoder_builder_free(builder: *mut HalDecoderBuilder) {
    if !builder.is_null() {
        drop(Box::from_raw(builder));
    }
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

    #[test]
    fn test_builder_create_and_free() {
        unsafe {
            let builder = hal_decoder_builder_new();
            assert!(!builder.is_null());
            hal_decoder_builder_free(builder);
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
            hal_decoder_builder_free(std::ptr::null_mut());
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
}
