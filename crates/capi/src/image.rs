// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image C API - Hardware-accelerated image processing.
//!
//! This module provides image conversion and manipulation functions with
//! support for hardware acceleration (G2D, OpenGL) when available.

use crate::decoder::{HalDecoder, HalDetectBoxList, HalProtoData, HalSegmentationList};
use crate::error::{set_error, set_error_null};
use crate::tensor::{HalDtype, HalTensor, HalTensorMemory};
use crate::{check_null, try_or_errno, try_or_null, HalByteTrack, HalTrackInfoList};
use edgefirst_codec::{CodecError, ImageDecoder, ImageLoad};
use edgefirst_decoder::{DetectBox, Segmentation};
#[allow(deprecated)]
use edgefirst_image::{
    save_jpeg, ComputeBackend, Crop, Fit, Flip, ImageProcessor, ImageProcessorConfig,
    ImageProcessorTrait, MaskResolution, Region, Rotation,
};
use edgefirst_tensor::{PixelFormat, PixelLayout, TensorDyn, TensorMemory};
use edgefirst_tracker::TrackInfo;
use libc::{c_char, c_int, size_t};
use std::{cell::RefCell, ffi::CStr};

thread_local! {
    static CODEC_DECODER: RefCell<ImageDecoder> = RefCell::new(ImageDecoder::new());
}

/// Parse an optional letterbox pointer into `Option<[f32; 4]>`.
///
/// # Safety
/// `ptr` must be NULL or point to at least 4 contiguous f32 values.
unsafe fn parse_letterbox(ptr: *const f32) -> Option<[f32; 4]> {
    if ptr.is_null() {
        None
    } else {
        let lb = std::slice::from_raw_parts(ptr, 4);
        Some([lb[0], lb[1], lb[2], lb[3]])
    }
}

/// Build a `MaskOverlay` from raw C parameters.
///
/// # Safety
/// - `background` must be NULL or a valid `HalTensor` pointer.
/// - `letterbox` must be NULL or point to at least 4 contiguous f32 values.
unsafe fn build_overlay<'a>(
    background: *const HalTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: HalColorMode,
) -> edgefirst_image::MaskOverlay<'a> {
    let bg = if background.is_null() {
        None
    } else {
        Some(&(*background).inner)
    };
    edgefirst_image::MaskOverlay {
        background: bg,
        opacity: opacity.clamp(0.0, 1.0),
        letterbox: parse_letterbox(letterbox),
        color_mode: color_mode.into(),
    }
}

/// Map a draw-masks error to a POSIX errno. Aliased `background`/`dst`
/// surfaces as `EINVAL`; everything else as `EIO`.
fn draw_err_to_errno(e: &edgefirst_image::Error) -> libc::c_int {
    match e {
        edgefirst_image::Error::AliasedBuffers(_) => libc::EINVAL,
        _ => libc::EIO,
    }
}

fn decode_err_to_errno(err: &CodecError) -> libc::c_int {
    match err {
        CodecError::InsufficientCapacity { .. } => libc::ENOSPC,
        _ => libc::EBADMSG,
    }
}

/// Image pixel format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalPixelFormat {
    /// 8-bit interleaved YUV422, limited range (YUYV)
    Yuyv = 0,
    /// 8-bit semi-planar YUV 4:2:0 (NV12): Y plane + interleaved CbCr.
    /// JPEG-decoded content is BT.601 full-range (interim).
    Nv12 = 1,
    /// 8-bit semi-planar YUV 4:2:2 (NV16): Y plane + interleaved CbCr.
    /// JPEG-decoded content is BT.601 full-range (interim).
    Nv16 = 2,
    /// 8-bit RGBA (4 channels)
    Rgba = 3,
    /// 8-bit RGB (3 channels)
    Rgb = 4,
    /// 8-bit grayscale, full range (Y800)
    Grey = 5,
    /// 8-bit planar RGB (3 planes)
    PlanarRgb = 6,
    /// 8-bit planar RGBA (4 planes)
    PlanarRgba = 7,
    /// 8-bit BGRA (4 channels, blue first)
    Bgra = 8,
    /// 8-bit interleaved YUV422, limited range (VYUY byte order)
    ///
    /// @note On Vivante GPUs (i.MX 8M Plus), VYUY import uses a 2D texture
    /// fallback instead of the EGL external texture path.  The HAL detects
    /// this automatically; no caller action is needed.  Quality is validated
    /// but may differ slightly from the external-texture path used on other
    /// GPUs.
    Vyuy = 9,
    /// 8-bit semi-planar YUV 4:4:4 (NV24): full-resolution chroma.
    ///
    /// Y plane (`H` rows) followed by an interleaved Cb/Cr plane at full
    /// resolution (`2*W` bytes per image row, stored as `2H` buffer rows).
    /// Emitted by the JPEG decoder for 4:4:4 sources; BT.601 full-range (interim).
    Nv24 = 10,
}

impl HalPixelFormat {
    fn to_pixel_format(self) -> PixelFormat {
        match self {
            HalPixelFormat::Yuyv => PixelFormat::Yuyv,
            HalPixelFormat::Nv12 => PixelFormat::Nv12,
            HalPixelFormat::Nv16 => PixelFormat::Nv16,
            HalPixelFormat::Rgba => PixelFormat::Rgba,
            HalPixelFormat::Rgb => PixelFormat::Rgb,
            HalPixelFormat::Grey => PixelFormat::Grey,
            HalPixelFormat::PlanarRgb => PixelFormat::PlanarRgb,
            HalPixelFormat::PlanarRgba => PixelFormat::PlanarRgba,
            HalPixelFormat::Bgra => PixelFormat::Bgra,
            HalPixelFormat::Vyuy => PixelFormat::Vyuy,
            HalPixelFormat::Nv24 => PixelFormat::Nv24,
        }
    }

    fn from_pixel_format(fmt: PixelFormat) -> Self {
        match fmt {
            PixelFormat::Rgb => HalPixelFormat::Rgb,
            PixelFormat::Rgba => HalPixelFormat::Rgba,
            PixelFormat::Grey => HalPixelFormat::Grey,
            PixelFormat::Yuyv => HalPixelFormat::Yuyv,
            PixelFormat::Nv12 => HalPixelFormat::Nv12,
            PixelFormat::Nv16 => HalPixelFormat::Nv16,
            PixelFormat::PlanarRgb => HalPixelFormat::PlanarRgb,
            PixelFormat::PlanarRgba => HalPixelFormat::PlanarRgba,
            PixelFormat::Bgra => HalPixelFormat::Bgra,
            PixelFormat::Vyuy => HalPixelFormat::Vyuy,
            PixelFormat::Nv24 => HalPixelFormat::Nv24,
            // TODO: make this a compile error when new PixelFormat variants are added
            other => {
                log::warn!("PixelFormat {other:?} has no C API mapping, defaulting to Rgb");
                HalPixelFormat::Rgb
            }
        }
    }
}

/// Image rotation angles.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalRotation {
    /// No rotation
    None = 0,
    /// 90 degrees clockwise
    Rotate90 = 1,
    /// 180 degrees
    Rotate180 = 2,
    /// 270 degrees clockwise (90 degrees counter-clockwise)
    Rotate270 = 3,
}

impl From<HalRotation> for Rotation {
    fn from(rot: HalRotation) -> Self {
        match rot {
            HalRotation::None => Rotation::None,
            HalRotation::Rotate90 => Rotation::Clockwise90,
            HalRotation::Rotate180 => Rotation::Rotate180,
            HalRotation::Rotate270 => Rotation::CounterClockwise90,
        }
    }
}

/// Image flip modes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalFlip {
    /// No flip
    None = 0,
    /// Vertical flip (top to bottom)
    Vertical = 1,
    /// Horizontal flip (left to right)
    Horizontal = 2,
}

impl From<HalFlip> for Flip {
    fn from(flip: HalFlip) -> Self {
        match flip {
            HalFlip::None => Flip::None,
            HalFlip::Vertical => Flip::Vertical,
            HalFlip::Horizontal => Flip::Horizontal,
        }
    }
}

/// Controls how mask colors are assigned to detections.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalColorMode {
    /// Color chosen by class label (default, correct for semantic segmentation)
    Class = 0,
    /// Color chosen by detection index (unique color per instance)
    Instance = 1,
    /// Color chosen by track ID (use with object tracking)
    Track = 2,
}

impl From<HalColorMode> for edgefirst_image::ColorMode {
    fn from(mode: HalColorMode) -> Self {
        match mode {
            HalColorMode::Class => edgefirst_image::ColorMode::Class,
            HalColorMode::Instance => edgefirst_image::ColorMode::Instance,
            HalColorMode::Track => edgefirst_image::ColorMode::Track,
        }
    }
}

/// Rectangular region (pixels) for a source crop.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HalRegion {
    /// X coordinate (left edge)
    pub x: size_t,
    /// Y coordinate (top edge)
    pub y: size_t,
    /// Width in pixels
    pub width: size_t,
    /// Height in pixels
    pub height: size_t,
}

impl From<HalRegion> for Region {
    fn from(r: HalRegion) -> Self {
        Region::new(r.x, r.y, r.width, r.height)
    }
}

/// Stretch fit (fill the destination).
pub const HAL_FIT_STRETCH: i32 = 0;
/// Letterbox fit (preserve source aspect, pad with `HalCrop::pad`).
pub const HAL_FIT_LETTERBOX: i32 = 1;

/// Crop configuration for image conversion — **source-side only**.
///
/// Destination placement is the destination tensor itself: pass a
/// `hal_tensor_view` / `hal_tensor_batch` sub-region as the `convert`
/// destination to render into a tile. This struct selects the source
/// sub-rectangle and the fit mode.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalCrop {
    /// Source rectangle to sample from
    pub source: HalRegion,
    /// Whether `source` is set (else the whole source is sampled)
    pub has_source: bool,
    /// Fit mode: `HAL_FIT_STRETCH` or `HAL_FIT_LETTERBOX`
    pub fit: i32,
    /// Letterbox pad colour (RGBA), used when `fit == HAL_FIT_LETTERBOX`
    pub pad: [u8; 4],
}

impl Default for HalCrop {
    fn default() -> Self {
        Self {
            source: HalRegion::default(),
            has_source: false,
            fit: HAL_FIT_STRETCH,
            pad: [0, 0, 0, 255],
        }
    }
}

impl From<HalCrop> for Crop {
    fn from(crop: HalCrop) -> Self {
        let source = crop.has_source.then(|| crop.source.into());
        let fit = if crop.fit == HAL_FIT_LETTERBOX {
            Fit::Letterbox { pad: crop.pad }
        } else {
            Fit::Stretch
        };
        Crop { source, fit }
    }
}

/// Opaque image processor type.
///
/// The ImageProcessor handles format conversion with hardware acceleration when available.
pub struct HalImageProcessor {
    /// Accessible to other modules for draw_decoded_masks / draw_proto_masks operations.
    pub(crate) inner: ImageProcessor,
}

// ============================================================================
// Rect and Crop Helper Functions
// ============================================================================

/// Create a new region.
///
/// @param x Left edge (x coordinate)
/// @param y Top edge (y coordinate)
/// @param width Width of the region
/// @param height Height of the region
/// @return New region structure
#[no_mangle]
pub extern "C" fn hal_region_new(x: size_t, y: size_t, width: size_t, height: size_t) -> HalRegion {
    HalRegion {
        x,
        y,
        width,
        height,
    }
}

/// Create a new default crop configuration (whole source, stretch fit).
///
/// @return New crop structure
#[no_mangle]
pub extern "C" fn hal_crop_new() -> HalCrop {
    HalCrop::default()
}

/// Set the source sampling region for a crop configuration.
///
/// @param crop Crop configuration to modify
/// @param region Source region (can be NULL to clear → whole source)
#[no_mangle]
pub unsafe extern "C" fn hal_crop_set_source(crop: *mut HalCrop, region: *const HalRegion) {
    if crop.is_null() {
        return;
    }
    unsafe {
        if region.is_null() {
            (*crop).has_source = false;
        } else {
            (*crop).source = *region;
            (*crop).has_source = true;
        }
    }
}

/// Configure letterbox fit with the given pad colour (RGBA).
///
/// @param crop Crop configuration to modify
/// @param r Red component (0-255)
/// @param g Green component (0-255)
/// @param b Blue component (0-255)
/// @param a Alpha component (0-255)
#[no_mangle]
pub unsafe extern "C" fn hal_crop_set_letterbox(crop: *mut HalCrop, r: u8, g: u8, b: u8, a: u8) {
    if crop.is_null() {
        return;
    }
    unsafe {
        (*crop).fit = HAL_FIT_LETTERBOX;
        (*crop).pad = [r, g, b, a];
    }
}

// ============================================================================
// Tensor Image Lifecycle Functions
// ============================================================================

/// Create a new empty image tensor.
///
/// @note When an image processor is available, prefer
/// hal_image_processor_create_image() which selects the optimal memory
/// backend (DMA-buf, PBO, or system memory) automatically. Direct
/// allocation via this function bypasses PBO-backed GPU zero-copy paths.
///
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param format Pixel format (HAL_PIXEL_FORMAT_*)
/// @param dtype Data type of tensor elements (HAL_DTYPE_*)
/// @param memory Memory allocation type (HAL_TENSOR_DMA recommended)
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (zero dimensions, unsupported format)
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_new_image(
    width: size_t,
    height: size_t,
    format: HalPixelFormat,
    dtype: HalDtype,
    memory: HalTensorMemory,
    access: crate::tensor::HalCpuAccess,
) -> *mut HalTensor {
    if width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    let mem_opt: Option<TensorMemory> = memory.into();
    let dyn_tensor = try_or_null!(
        TensorDyn::image(
            width,
            height,
            format.to_pixel_format(),
            dtype.into(),
            mem_opt,
            access.into(),
        ),
        libc::ENOMEM
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Decode image data (JPEG/PNG) into a pre-allocated tensor.
///
/// The tensor must have sufficient capacity for the decoded image.
/// Returns 0 on success, -1 on error (errno set).
///
/// The image is decoded to its native pixel format (JPEG → NV12/NV16/NV24 by
/// chroma subsampling (4:2:0/4:2:2/4:4:4) for colour or GREY for greyscale;
/// PNG → RGB/RGBA/GREY) and the tensor is configured
/// with that format and the decoded dimensions. Use the tensor's pixel format
/// accessor (`hal_tensor_pixel_format()`) to inspect the result, and the image
/// processor convert API (`hal_image_processor_convert()`) if a different
/// format such as RGB is required.
///
/// @note EXIF orientation is reported but never applied. The decoder writes
/// the source's native (unrotated) pixels and dimensions; callers that need an
/// upright image must apply the reported orientation themselves downstream
/// (e.g. via `hal_image_processor_convert()`). `out_rotation_degrees` and
/// `out_flip_horizontal` describe the transform the caller should apply: rotate
/// clockwise by the given degrees (0/90/180/270), then flip horizontally if
/// requested. Both are `0`/`false` when the image has no EXIF orientation.
///
/// @param tensor Pre-allocated tensor to decode into
/// @param data Pointer to encoded image data (JPEG or PNG)
/// @param len Length of image data in bytes
/// @param out_width If non-NULL, receives the decoded image width
/// @param out_height If non-NULL, receives the decoded image height
/// @param out_rotation_degrees If non-NULL, receives the EXIF clockwise
///        rotation in degrees the caller should apply (0/90/180/270)
/// @param out_flip_horizontal If non-NULL, receives whether the caller should
///        also flip the image horizontally
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL tensor/data, zero length)
/// - EBADMSG: Failed to decode image
/// - ENOSPC: Tensor capacity insufficient for decoded image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_decode_image(
    tensor: *mut HalTensor,
    data: *const u8,
    len: size_t,
    out_width: *mut size_t,
    out_height: *mut size_t,
    out_rotation_degrees: *mut u16,
    out_flip_horizontal: *mut bool,
) -> c_int {
    check_null!(tensor);
    check_null!(data);
    if len == 0 {
        return set_error(libc::EINVAL);
    }

    let tensor = unsafe { &mut *tensor };
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };

    let info = CODEC_DECODER.with(|cell| {
        let mut decoder = cell.borrow_mut();
        tensor.inner.load_image(&mut decoder, data_slice)
    });

    let info = match info {
        Ok(info) => info,
        Err(err) => return set_error(decode_err_to_errno(&err)),
    };

    if !out_width.is_null() {
        unsafe { *out_width = info.width };
    }
    if !out_height.is_null() {
        unsafe { *out_height = info.height };
    }
    if !out_rotation_degrees.is_null() {
        unsafe { *out_rotation_degrees = info.rotation_degrees };
    }
    if !out_flip_horizontal.is_null() {
        unsafe { *out_flip_horizontal = info.flip_horizontal };
    }

    0
}

/// Decode an image file (JPEG/PNG) into a pre-allocated tensor.
///
/// The image is decoded to its native pixel format and the tensor is
/// configured accordingly; see `hal_tensor_decode_image()` for details.
///
/// @param tensor Pre-allocated tensor to decode into
/// @param path Path to the image file
/// @param out_width If non-NULL, receives the decoded image width
/// @param out_height If non-NULL, receives the decoded image height
/// @param out_rotation_degrees If non-NULL, receives the EXIF clockwise
///        rotation in degrees the caller should apply (0/90/180/270)
/// @param out_flip_horizontal If non-NULL, receives whether the caller should
///        also flip the image horizontally
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL tensor/path, invalid UTF-8)
/// - ENOENT: File not found
/// - EIO: Failed to read file
/// - EBADMSG: Failed to decode image
/// - ENOSPC: Tensor capacity insufficient
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_decode_image_file(
    tensor: *mut HalTensor,
    path: *const c_char,
    out_width: *mut size_t,
    out_height: *mut size_t,
    out_rotation_degrees: *mut u16,
    out_flip_horizontal: *mut bool,
) -> c_int {
    check_null!(tensor);
    check_null!(path);

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return set_error(libc::EINVAL),
    };

    let data = match std::fs::read(path_str) {
        Ok(d) => d,
        Err(e) => {
            return set_error(if e.kind() == std::io::ErrorKind::NotFound {
                libc::ENOENT
            } else {
                libc::EIO
            });
        }
    };

    unsafe {
        hal_tensor_decode_image(
            tensor,
            data.as_ptr(),
            data.len(),
            out_width,
            out_height,
            out_rotation_degrees,
            out_flip_horizontal,
        )
    }
}

/// Save an image tensor as JPEG.
///
/// @param tensor Image tensor to save
/// @param path Output file path
/// @param quality JPEG quality (1-100, 0 for default)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL tensor/path)
/// - EIO: Failed to write file
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_save_jpeg(
    tensor: *const HalTensor,
    path: *const c_char,
    quality: c_int,
) -> c_int {
    check_null!(tensor, path);

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return set_error(libc::EINVAL),
    };

    let quality = if quality <= 0 || quality > 100 {
        80
    } else {
        quality as u8
    };

    let dyn_ref = &unsafe { &(*tensor) }.inner;

    try_or_errno!(save_jpeg(dyn_ref, path_str, quality), libc::EIO);
    0
}

// ============================================================================
// Tensor Image Property Functions
// ============================================================================

/// Get the width of an image tensor.
///
/// @param tensor Image tensor handle
/// @return Width in pixels, or 0 if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_width(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    unsafe { &(*tensor) }.inner.width().unwrap_or(0)
}

/// Get the height of an image tensor.
///
/// @param tensor Image tensor handle
/// @return Height in pixels, or 0 if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_height(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    unsafe { &(*tensor) }.inner.height().unwrap_or(0)
}

/// Get the pixel format of an image tensor.
///
/// @param tensor Image tensor handle
/// @return Pixel format, or HAL_PIXEL_FORMAT_RGB if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_pixel_format(tensor: *const HalTensor) -> HalPixelFormat {
    if tensor.is_null() {
        return HalPixelFormat::Rgb;
    }
    match unsafe { &(*tensor) }.inner.format() {
        Some(fmt) => HalPixelFormat::from_pixel_format(fmt),
        None => HalPixelFormat::Rgb,
    }
}

/// Attach pixel format metadata to a tensor.
///
/// Validates that the tensor's shape is compatible with the format's
/// layout (packed, planar, or semi-planar). This enables tensors
/// created via `hal_tensor_from_fd()` to be used as image conversion
/// destinations.
///
/// @param tensor Tensor handle
/// @param format Pixel format to attach (HAL_PIXEL_FORMAT_*)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL tensor or shape doesn't match format layout
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_set_format(
    tensor: *mut HalTensor,
    format: HalPixelFormat,
) -> c_int {
    check_null!(tensor);
    let fmt = format.to_pixel_format();
    match unsafe { &mut *tensor }.inner.set_format(fmt) {
        Ok(()) => 0,
        Err(_) => set_error(libc::EINVAL),
    }
}

/// Check if an image tensor uses a planar pixel format.
///
/// Planar formats store each color channel in a separate plane (e.g., NV12,
/// NV16, PLANAR_RGB, PLANAR_RGBA), while interleaved formats store channels
/// together per pixel (e.g., RGB, RGBA, YUYV).
///
/// @param tensor Image tensor handle
/// @return true if the tensor uses a planar format, false if interleaved or NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_is_planar(tensor: *const HalTensor) -> bool {
    if tensor.is_null() {
        return false;
    }
    match unsafe { &(*tensor) }.inner.format() {
        Some(fmt) => !matches!(fmt.layout(), PixelLayout::Packed),
        None => false,
    }
}

/// Get the number of channels in an image tensor.
///
/// Returns the number of color channels (e.g., 3 for RGB, 4 for RGBA,
/// 1 for GREY/NV12/NV16 (luma plane)).
///
/// @param tensor Image tensor handle
/// @return Number of channels, or 0 if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_channels(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    match unsafe { &(*tensor) }.inner.format() {
        Some(fmt) => fmt.channels(),
        None => 0,
    }
}

/// Get the effective row stride of an image tensor in bytes.
///
/// If an explicit stride was set (e.g. via `hal_plane_descriptor_set_stride`),
/// that value is returned. Otherwise the minimum stride is computed from the
/// format, width, and element size: `width * channels * element_size` for
/// packed formats, `width * element_size` for planar/semi-planar.
///
/// @param tensor Image tensor handle
/// @return Effective row stride in bytes: the explicit stride if set, or the
///         minimum stride computed from format, width, and element size.
///         Returns 0 only when no pixel format is set (and no explicit stride
///         was stored) or tensor is NULL.
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_row_stride(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    let dyn_ref = &unsafe { &(*tensor) }.inner;
    dyn_ref.effective_row_stride().unwrap_or(0)
}

// ============================================================================
// ImageProcessor Functions
// ============================================================================

/// Create a new image processor.
///
/// Automatically selects the best available backend (G2D, OpenGL, or CPU).
///
/// @return New image processor handle on success, NULL on error
/// @par Errors (errno):
/// - ENOMEM: Memory allocation failed
/// - ENOTSUP: No suitable image processing backend available
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_new() -> *mut HalImageProcessor {
    let processor = try_or_null!(ImageProcessor::new(), libc::ENOTSUP);
    Box::into_raw(Box::new(HalImageProcessor { inner: processor }))
}

/// Compute backend selection for image processing.
///
/// @see hal_image_processor_new_with_backend
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalComputeBackend {
    /// Auto-detect based on hardware and environment variables.
    Auto = 0,
    /// CPU-only processing.
    Cpu = 1,
    /// Prefer G2D hardware blitter (with CPU fallback).
    G2d = 2,
    /// Prefer OpenGL ES (with CPU fallback).
    #[allow(non_camel_case_types)]
    Opengl = 3,
}

impl From<HalComputeBackend> for ComputeBackend {
    fn from(b: HalComputeBackend) -> Self {
        match b {
            HalComputeBackend::Auto => ComputeBackend::Auto,
            HalComputeBackend::Cpu => ComputeBackend::Cpu,
            HalComputeBackend::G2d => ComputeBackend::G2d,
            HalComputeBackend::Opengl => ComputeBackend::OpenGl,
        }
    }
}

/// Create a new image processor with a specific compute backend.
///
/// When `backend` is not `HAL_COMPUTE_BACKEND_AUTO`, the processor
/// initializes the requested backend plus CPU as a fallback chain.
/// Environment variables (`EDGEFIRST_FORCE_BACKEND`, `EDGEFIRST_DISABLE_*`)
/// are ignored in this case.
///
/// @param backend Preferred compute backend
/// @return New image processor handle on success, NULL on error
/// @par Errors (errno):
/// - ENOMEM: Memory allocation failed
/// - ENOTSUP: No suitable image processing backend available
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_new_with_backend(
    backend: HalComputeBackend,
) -> *mut HalImageProcessor {
    // `needless_update` fires on macOS where `ImageProcessorConfig` has only
    // `backend`, but on Linux the struct also has the cfg-gated `egl_display`.
    #[allow(clippy::needless_update)]
    let config = ImageProcessorConfig {
        backend: backend.into(),
        ..Default::default()
    };
    let processor = try_or_null!(ImageProcessor::with_config(config), libc::ENOTSUP);
    Box::into_raw(Box::new(HalImageProcessor { inner: processor }))
}

/// Convert an image to a different format/size.
///
/// Performs format conversion, scaling, rotation, flip, and crop operations.
/// When the OpenGL backend is active, the function issues a `glFinish()` before
/// returning, guaranteeing that all GPU reads of `src` and writes to `dst` are
/// complete. The caller must not write to the `src` DMA-BUF while this function
/// is in progress.
///
/// **EGL image cache**: The OpenGL backend maintains separate LRU caches for
/// source and destination EGLImages, keyed by the tensor's `BufferIdentity.id`.
/// Reusing the same tensor objects across frames yields cache hits on both
/// sides; creating new tensors from the same fds every frame yields cache
/// misses. See `hal_tensor_from_fd()` and `hal_import_image()` for
/// details on `BufferIdentity` assignment.
///
/// **Content updates between calls**: Because EGLImage is a handle to live
/// physical memory, content written into a DMA-BUF between calls (e.g., by a
/// video decoder) is visible to the GPU on the next call. The EGL image and
/// tensor wrapper remain valid and do not need to be recreated.
///
/// **Chaining convert() calls**: It is safe to chain calls where the `dst` of
/// one call becomes the `src` of the next (e.g., NV12 → RGBA → PlanarRgb).
/// The `glFinish()` issued at the end of each call ensures GPU coherency.
/// The same DMA-BUF used as both `dst` in one call and `src` in the next will
/// have two separate EGLImage cache entries (one per cache) without collision.
/// In-place conversion where `src` and `dst` are the same tensor in a single
/// call is undefined behavior.
///
/// @param processor Image processor handle
/// @param src Source image tensor
/// @param dst Destination image tensor (must be pre-allocated with desired dimensions)
/// @param rotation Rotation to apply
/// @param flip Flip to apply
/// @param crop Crop configuration (can be NULL for no crop)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/src/dst)
/// - EIO: Conversion failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_convert(
    processor: *mut HalImageProcessor,
    src: *const HalTensor,
    dst: *mut HalTensor,
    rotation: HalRotation,
    flip: HalFlip,
    crop: *const HalCrop,
) -> c_int {
    check_null!(processor, src, dst);

    let crop_config = if crop.is_null() {
        Crop::default()
    } else {
        unsafe { *crop }.into()
    };

    try_or_errno!(
        unsafe { &mut (*processor) }.inner.convert(
            &unsafe { &(*src) }.inner,
            &mut unsafe { &mut (*dst) }.inner,
            rotation.into(),
            flip.into(),
            crop_config,
        ),
        libc::EIO
    );
    0
}

/// Convert and export a native sync-fence fd instead of blocking on the
/// GPU — the GL→NPU handoff (`EGL_ANDROID_native_fence_sync`, Android).
///
/// Identical semantics to [`hal_image_processor_convert`] (including the
/// GL→CPU fallback chain), except that when the platform supports native
/// fences the call returns as soon as the GPU commands are submitted and
/// writes the fence fd to `*fence_fd`. The destination buffer must not
/// be read until the fence signals — hand the fd to the NPU runtime
/// (`ANeuralNetworksExecution_startComputeWithDependencies`) or `poll()`
/// it. When no fence is available (`*fence_fd == -1`) the convert
/// completed with the blocking contract and the destination is already
/// safe to read.
///
/// The returned fd is owned by the caller: `close()` it when done (the
/// NPU runtime duplicates what it needs).
///
/// @param processor Image processor handle
/// @param src Source image tensor
/// @param dst Destination image tensor (pre-allocated; reuse across frames)
/// @param rotation Rotation to apply
/// @param flip Flip to apply
/// @param crop Crop configuration (can be NULL for no crop)
/// @param fence_fd Out: the sync-fence fd, or -1 when the convert
///                 completed synchronously (must not be NULL)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/src/dst/fence_fd)
/// - EIO: Conversion failed
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_image_processor_convert_fence(
    processor: *mut HalImageProcessor,
    src: *const HalTensor,
    dst: *mut HalTensor,
    rotation: HalRotation,
    flip: HalFlip,
    crop: *const HalCrop,
    fence_fd: *mut c_int,
) -> c_int {
    check_null!(processor, src, dst, fence_fd);

    let crop_config = if crop.is_null() {
        Crop::default()
    } else {
        unsafe { *crop }.into()
    };

    let fd = try_or_errno!(
        unsafe { &mut (*processor) }.inner.convert_with_fence(
            &unsafe { &(*src) }.inner,
            &mut unsafe { &mut (*dst) }.inner,
            rotation.into(),
            flip.into(),
            crop_config,
        ),
        libc::EIO
    );
    unsafe {
        *fence_fd = fd.map_or(-1, |owned| {
            use std::os::fd::IntoRawFd;
            owned.into_raw_fd()
        });
    }
    0
}

/// Convert without waiting for the GPU — the batch-preprocessing primitive.
///
/// Identical to [`hal_image_processor_convert`](hal_image_processor_convert)
/// except the OpenGL backend does **not** issue the per-call `glFinish()`. Render
/// `N` model inputs by looping this over row-band views of one batched
/// destination (each `dst` a `hal_tensor_view` / `hal_tensor_batch` sub-region of
/// the same buffer) and then call [`hal_image_processor_flush`] **once**: the
/// backend imports the destination a single time and renders each tile as a
/// `glViewport` band, syncing once at flush. The result of a deferred convert is
/// **not** safe to read on the CPU (or map via CUDA) until `flush` returns
/// (CUDA map auto-flushes). Non-GL backends complete synchronously and `flush`
/// is a no-op.
///
/// @param processor Image processor handle
/// @param src Source image tensor
/// @param dst Destination image tensor (typically a `hal_tensor_view`/`batch` tile)
/// @param rotation Rotation to apply
/// @param flip Flip to apply
/// @param crop Crop configuration (can be NULL for no crop)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/src/dst)
/// - EIO: Conversion failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_convert_deferred(
    processor: *mut HalImageProcessor,
    src: *const HalTensor,
    dst: *mut HalTensor,
    rotation: HalRotation,
    flip: HalFlip,
    crop: *const HalCrop,
) -> c_int {
    check_null!(processor, src, dst);

    let crop_config = if crop.is_null() {
        Crop::default()
    } else {
        unsafe { *crop }.into()
    };

    try_or_errno!(
        unsafe { &mut (*processor) }.inner.convert_deferred(
            &unsafe { &(*src) }.inner,
            &mut unsafe { &mut (*dst) }.inner,
            rotation.into(),
            flip.into(),
            crop_config,
        ),
        libc::EIO
    );
    0
}

/// Complete all deferred converts since the last flush with a single GPU sync.
///
/// After this returns, every destination written by
/// [`hal_image_processor_convert_deferred`] is finished and safe to read back or
/// `cuda_map`. Backends with no deferred path return success immediately.
///
/// @param processor Image processor handle
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor)
/// - EIO: Flush failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_flush(processor: *mut HalImageProcessor) -> c_int {
    check_null!(processor);
    try_or_errno!(unsafe { &mut (*processor) }.inner.flush(), libc::EIO);
    0
}

/// Draw detection boxes and segmentation masks onto an image.
///
/// Draws bounding boxes (with labels) and segmentation overlays on the
/// destination image tensor. Uses hardware acceleration (OpenGL) when available,
/// falling back to CPU rendering.
///
/// **Output contract:** `dst` is always fully written by this call — its prior
/// contents are discarded.  The four cases are:
///
/// | detections | background | output                            |
/// |------------|------------|-----------------------------------|
/// | NULL       | NULL       | dst cleared to 0x00000000         |
/// | NULL       | set        | dst <- background                 |
/// | set        | NULL       | masks drawn over cleared dst      |
/// | set        | set        | masks drawn over background       |
///
/// @note **Migrating from v0.16.3 or earlier:** if you populated `dst` with
/// an image before calling this function, you must now pass that image as
/// `background` instead; pre-loading `dst` is no longer effective.
///
/// @param processor Image processor handle
/// @param dst Output image tensor (always fully written; prior contents discarded)
/// @param detections Detection box list (NULL for no detections)
/// @param segmentations Segmentation list (NULL for detection-only or no masks)
/// @param background Optional compositing source — `dst` is written as
///        `background + masks`. Pass NULL to clear `dst` to transparent.
///        Must have the same dimensions and format as `dst`.
/// @param opacity Mask opacity in [0.0, 1.0] (1.0 = fully opaque, clamped)
/// @param letterbox Optional letterbox coordinates [x0, y0, x1, y1] in normalized
///        coordinates for mapping model-space boxes back to image space (NULL = no letterbox)
/// @param color_mode How to assign colors to detections (HAL_COLOR_MODE_CLASS by default)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor or dst, or `background` aliases `dst` — they must reference distinct buffers)
/// - EIO: Drawing failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_draw_decoded_masks(
    processor: *mut HalImageProcessor,
    dst: *mut HalTensor,
    detections: *const HalDetectBoxList,
    segmentations: *const HalSegmentationList,
    background: *const HalTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: HalColorMode,
) -> c_int {
    check_null!(processor, dst);

    // Reject same-HalTensor aliasing BEFORE `build_overlay` borrows
    // background — otherwise we'd form `&(*bg).inner` and `&mut (*dst).inner`
    // over the same object, which is UB at the Rust reference level before
    // any runtime check can fire. `TensorDyn::aliases` inside
    // `draw_decoded_masks` still catches the distinct-wrapper same-buffer
    // case (separate HalTensor handles over one dmabuf fd).
    if !background.is_null() && std::ptr::eq(background, dst as *const _) {
        return set_error(libc::EINVAL);
    }

    let detect_slice = if detections.is_null() {
        &[]
    } else {
        unsafe { &(*detections).boxes }.as_slice()
    };

    let seg_slice = if segmentations.is_null() {
        &[]
    } else {
        unsafe { &(*segmentations).masks }.as_slice()
    };

    let overlay = build_overlay(background, opacity, letterbox, color_mode);

    if let Err(e) = unsafe { &mut (*processor) }.inner.draw_decoded_masks(
        &mut unsafe { &mut (*dst) }.inner,
        detect_slice,
        seg_slice,
        overlay,
    ) {
        return set_error(draw_err_to_errno(&e));
    }
    0
}

/// Materialize per-instance segmentation masks from prototype data.
///
/// Computes `mask_coeff @ protos` for each detection, producing compact
/// binary masks at prototype resolution (e.g., 160×160 crops).
/// Mask values are binary uint8 {0, 255} — pixels where the dot product
/// is positive are foreground (255), otherwise background (0).
///
/// The returned segmentation list can be:
/// - Inspected via `hal_segmentation_list_get_mask()` for analytics, IoU, etc.
/// - Passed to `hal_image_processor_draw_decoded_masks()` for GPU rendering.
///
/// @note Calling `hal_decoder_decode_proto()` + `hal_image_processor_materialize_masks()`
///       + `hal_image_processor_draw_decoded_masks()` separately prevents the HAL from
///       using its internal fused optimization. For render-only use cases, prefer
///       `hal_image_processor_draw_masks()` which is 1.6–27× faster on tested platforms.
///
/// @param processor Image processor handle
/// @param detections Detection box list from `hal_decoder_decode_proto()`
/// @param proto Prototype data from `hal_decoder_decode_proto()`
/// @param letterbox Optional letterbox coordinates [x0, y0, x1, y1] in normalized
///        coordinates (NULL = no letterbox correction)
/// @return Segmentation list (caller must free with `hal_segmentation_list_free()`),
///         or NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/detections/proto)
/// - ENOTSUP: CPU backend not available (required for mask materialization)
/// - EIO: Materialization failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_materialize_masks(
    processor: *mut HalImageProcessor,
    detections: *const HalDetectBoxList,
    proto: *const HalProtoData,
    letterbox: *const f32,
) -> *mut HalSegmentationList {
    if processor.is_null() || detections.is_null() || proto.is_null() {
        set_error(libc::EINVAL);
        return std::ptr::null_mut();
    }

    let masks = match (*processor).inner.materialize_masks(
        &(*detections).boxes,
        &(*proto).inner,
        parse_letterbox(letterbox),
        MaskResolution::Proto,
    ) {
        Ok(m) => m,
        Err(edgefirst_image::Error::NoConverter) => {
            log::error!("hal_image_processor_materialize_masks: CPU backend not available");
            set_error(libc::ENOTSUP);
            return std::ptr::null_mut();
        }
        Err(e) => {
            log::error!("hal_image_processor_materialize_masks: {e:#?}");
            set_error(libc::EIO);
            return std::ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(HalSegmentationList { masks }))
}

/// Decode model outputs and draw masks directly onto a destination image.
///
/// This is the fused path: for segmentation models, prototype data is passed
/// directly to the renderer without materializing intermediate mask arrays.
/// For detection-only models, this falls back to decode + draw_decoded_masks.
///
/// **Output contract:** `dst` is always fully written by this call — its prior
/// contents are discarded.  See `hal_image_processor_draw_decoded_masks` for
/// the four-case detections × background matrix.
///
/// @note **Migrating from v0.16.3 or earlier:** if you populated `dst` before
/// calling this function, pass that image as `background` instead.
///
/// @param processor Image processor handle
/// @param decoder Decoder handle
/// @param outputs Array of output tensor pointers
/// @param num_outputs Number of output tensors
/// @param dst Output image tensor (always fully written; prior contents discarded)
/// @param background Optional compositing source — `dst` is written as
///        `background + masks`. Pass NULL to clear `dst` to transparent.
///        Must have the same dimensions and format as `dst`.
/// @param opacity Mask opacity, clamped to [0.0, 1.0] (1.0 = fully opaque)
/// @param letterbox Optional letterbox coordinates [x0, y0, x1, y1] in normalized
///        coordinates (NULL = no letterbox correction)
/// @param color_mode How to assign colors to detections (HAL_COLOR_MODE_CLASS by default)
/// @param out_boxes Output parameter for detection box list (caller must free)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/decoder/outputs/dst/out_boxes, or `background` aliases `dst` — they must reference distinct buffers)
/// - EIO: Decoding or drawing failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_draw_masks(
    processor: *mut HalImageProcessor,
    decoder: *const HalDecoder,
    outputs: *const *const HalTensor,
    num_outputs: size_t,
    dst: *mut HalTensor,
    background: *const HalTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: HalColorMode,
    out_boxes: *mut *mut HalDetectBoxList,
) -> c_int {
    check_null!(processor, decoder, outputs, dst, out_boxes);

    if num_outputs == 0 {
        return set_error(libc::EINVAL);
    }

    // Reject same-HalTensor aliasing before borrowing through build_overlay
    // (UB guard — see hal_image_processor_draw_decoded_masks). TensorDyn::aliases
    // in draw_{decoded,proto}_masks still catches the distinct-wrapper case.
    if !background.is_null() && std::ptr::eq(background, dst as *const _) {
        return set_error(libc::EINVAL);
    }

    let overlay = build_overlay(background, opacity, letterbox, color_mode);

    let outputs_slice = std::slice::from_raw_parts(outputs, num_outputs);

    // Extract TensorDyn references from HalTensor pointers
    let tensor_refs = match crate::decoder::extract_tensor_refs(outputs_slice) {
        Ok(refs) => refs,
        Err(rc) => return rc,
    };

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(100);

    // Try proto decode first (returns ProtoData for seg models, None for det-only)
    let proto_result = try_or_errno!(
        (*decoder).inner.decode_proto(&tensor_refs, &mut boxes),
        libc::EIO
    );

    if let Some(proto_data) = proto_result {
        // Fused path: render directly from proto data
        if let Err(e) =
            (*processor)
                .inner
                .draw_proto_masks(&mut (*dst).inner, &boxes, &proto_data, overlay)
        {
            return set_error(draw_err_to_errno(&e));
        }
    } else {
        // Detection-only fallback: full decode + draw_decoded_masks
        let mut masks: Vec<Segmentation> = Vec::new();
        try_or_errno!(
            (*decoder)
                .inner
                .decode(&tensor_refs, &mut boxes, &mut masks),
            libc::EIO
        );
        if let Err(e) =
            (*processor)
                .inner
                .draw_decoded_masks(&mut (*dst).inner, &boxes, &masks, overlay)
        {
            return set_error(draw_err_to_errno(&e));
        }
    }

    *out_boxes = Box::into_raw(Box::new(HalDetectBoxList { boxes }));
    0
}

/// Draw segmentation masks from pre-decoded boxes and prototype data.
///
/// This is the split version of `hal_image_processor_draw_masks()`: the caller
/// has already called `hal_decoder_decode_proto()` to obtain boxes and proto data,
/// and now wants to render masks onto a destination image using the GPU-accelerated
/// proto path.
///
/// This avoids re-decoding when the same inference results need to be rendered
/// onto multiple video frames (e.g., in a dual-pad overlay element where tensors
/// and video arrive asynchronously at different rates).
///
/// **Output contract:** `dst` is always fully written by this call — its prior
/// contents are discarded.  See `hal_image_processor_draw_decoded_masks` for
/// the four-case detections × background matrix.
///
/// @param processor Image processor handle
/// @param dst Destination image (RGBA, must not alias `background`)
/// @param detections Detection box list from `hal_decoder_decode_proto()`
/// @param proto Prototype data from `hal_decoder_decode_proto()`
/// @param background Optional source image to composite under masks (NULL = clear to transparent black)
/// @param opacity Mask opacity (0.0 = transparent, 1.0 = opaque)
/// @param letterbox Optional letterbox coordinates [x0, y0, x1, y1] in normalized
///        coordinates (NULL = no letterbox correction)
/// @param color_mode How to assign colors to detections (HAL_COLOR_MODE_CLASS by default)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/dst/detections/proto, or `background` aliases `dst`)
/// - EIO: Rendering failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_draw_proto_masks(
    processor: *mut HalImageProcessor,
    dst: *mut HalTensor,
    detections: *const HalDetectBoxList,
    proto: *const HalProtoData,
    background: *const HalTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: HalColorMode,
) -> c_int {
    check_null!(processor, dst, detections, proto);

    // Reject same-HalTensor aliasing before borrowing through build_overlay
    if !background.is_null() && std::ptr::eq(background, dst as *const _) {
        return set_error(libc::EINVAL);
    }

    let overlay = build_overlay(background, opacity, letterbox, color_mode);

    if let Err(e) = (*processor).inner.draw_proto_masks(
        &mut (*dst).inner,
        &(*detections).boxes,
        &(*proto).inner,
        overlay,
    ) {
        return set_error(draw_err_to_errno(&e));
    }
    0
}

/// Decode tracked model outputs and draw masks directly onto a destination image.
///
/// This is the fused tracked path: for segmentation models, prototype data is
/// passed directly to the renderer without materializing intermediate mask
/// arrays. Object tracking is applied to maintain identities across frames.
/// For detection-only models, this falls back to tracked decode + draw_decoded_masks.
///
/// **Output contract:** `dst` is always fully written by this call — its prior
/// contents are discarded.  See `hal_image_processor_draw_decoded_masks` for
/// the four-case detections × background matrix.
///
/// @note **Migrating from v0.16.3 or earlier:** if you populated `dst` before
/// calling this function, pass that image as `background` instead.
///
/// @param processor Image processor handle
/// @param decoder Decoder handle
/// @param tracker Tracker handle for maintaining object identities across frames
/// @param timestamp Timestamp for the current frame (e.g., in nanoseconds)
/// @param outputs Array of output tensor pointers
/// @param num_outputs Number of output tensors
/// @param dst Output image tensor (always fully written; prior contents discarded)
/// @param background Optional compositing source — `dst` is written as
///        `background + masks`. Pass NULL to clear `dst` to transparent.
///        Must have the same dimensions and format as `dst`.
/// @param opacity Mask opacity in [0.0, 1.0] (1.0 = fully opaque, clamped)
/// @param letterbox Optional letterbox coordinates [x0, y0, x1, y1] in normalized
///        coordinates (NULL = no letterbox correction)
/// @param color_mode How to assign colors to detections (HAL_COLOR_MODE_CLASS by default)
/// @param out_boxes Output parameter for detection box list (caller must free)
/// @param out_tracks Output parameter for track info list (can be NULL; caller must free if non-NULL)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/decoder/tracker/outputs/dst/out_boxes, or `background` aliases `dst` — they must reference distinct buffers)
/// - EIO: Decoding or drawing failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_draw_masks_tracked(
    processor: *mut HalImageProcessor,
    decoder: *const HalDecoder,
    tracker: *mut HalByteTrack,
    timestamp: u64,
    outputs: *const *const HalTensor,
    num_outputs: size_t,
    dst: *mut HalTensor,
    background: *const HalTensor,
    opacity: f32,
    letterbox: *const f32,
    color_mode: HalColorMode,
    out_boxes: *mut *mut HalDetectBoxList,
    out_tracks: *mut *mut HalTrackInfoList,
) -> c_int {
    check_null!(processor, decoder, tracker, outputs, dst, out_boxes);

    if num_outputs == 0 {
        return set_error(libc::EINVAL);
    }

    // Reject same-HalTensor aliasing before borrowing through build_overlay
    // (UB guard — see hal_image_processor_draw_decoded_masks). TensorDyn::aliases
    // in draw_{decoded,proto}_masks still catches the distinct-wrapper case.
    if !background.is_null() && std::ptr::eq(background, dst as *const _) {
        return set_error(libc::EINVAL);
    }
    let overlay = build_overlay(background, opacity, letterbox, color_mode);

    let outputs_slice = std::slice::from_raw_parts(outputs, num_outputs);

    // Extract TensorDyn references from HalTensor pointers
    let tensor_refs = match crate::decoder::extract_tensor_refs(outputs_slice) {
        Ok(refs) => refs,
        Err(rc) => return rc,
    };

    let mut boxes: Vec<DetectBox> = Vec::with_capacity(100);
    let mut tracks: Vec<TrackInfo> = Vec::new();

    // Try tracked proto decode first (returns ProtoData for seg models, None for det-only)
    let proto_result = try_or_errno!(
        (*decoder).inner.decode_proto_tracked(
            &mut (*tracker).inner,
            timestamp,
            &tensor_refs,
            &mut boxes,
            &mut tracks
        ),
        libc::EIO
    );

    if let Some(proto_data) = proto_result {
        // Fused path: render directly from proto data
        if let Err(e) =
            (*processor)
                .inner
                .draw_proto_masks(&mut (*dst).inner, &boxes, &proto_data, overlay)
        {
            return set_error(draw_err_to_errno(&e));
        }
    } else {
        // Detection-only fallback: full tracked decode + draw_decoded_masks
        let mut masks: Vec<Segmentation> = Vec::new();
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
        if let Err(e) =
            (*processor)
                .inner
                .draw_decoded_masks(&mut (*dst).inner, &boxes, &masks, overlay)
        {
            return set_error(draw_err_to_errno(&e));
        }
    }

    *out_boxes = Box::into_raw(Box::new(HalDetectBoxList { boxes }));

    if !out_tracks.is_null() {
        *out_tracks = Box::into_raw(Box::new(HalTrackInfoList { tracks }));
    }

    0
}

/// Set class colors for segmentation rendering.
///
/// Colors are used when drawing segmentation masks via
/// hal_image_processor_draw_decoded_masks(). Each color is an RGBA tuple.
///
/// @param processor Image processor handle
/// @param colors Pointer to array of RGBA color tuples ([u8; 4] per color)
/// @param num_colors Number of colors in the array
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor or colors)
/// - EIO: Failed to set colors
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_set_class_colors(
    processor: *mut HalImageProcessor,
    colors: *const [u8; 4],
    num_colors: size_t,
) -> c_int {
    check_null!(processor, colors);

    if num_colors == 0 {
        return set_error(libc::EINVAL);
    }

    let colors_slice = unsafe { std::slice::from_raw_parts(colors, num_colors) };

    try_or_errno!(
        unsafe { &mut (*processor) }
            .inner
            .set_class_colors(colors_slice),
        libc::EIO
    );
    0
}

/// Create a new image tensor using the processor's optimal memory backend.
///
/// Selects the best available backing storage based on hardware capabilities:
/// DMA-buf > PBO (GPU buffer) > system memory. Images created this way benefit
/// from zero-copy GPU paths when used with the same processor's convert().
///
/// @param processor Image processor handle
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param format Pixel format (HAL_PIXEL_FORMAT_*)
/// @param dtype Data type of tensor elements (HAL_DTYPE_*)
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor, zero dimensions, invalid shape)
/// - ENOTSUP: Unsupported format or operation
/// - EIO: I/O error during DMA or GPU buffer allocation
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_create_image(
    processor: *mut HalImageProcessor,
    width: size_t,
    height: size_t,
    format: HalPixelFormat,
    dtype: HalDtype,
    access: crate::tensor::HalCpuAccess,
) -> *mut HalTensor {
    if processor.is_null() || width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    let dyn_tensor = match unsafe { &(*processor) }.inner.create_image(
        width,
        height,
        format.to_pixel_format(),
        dtype.into(),
        None,
        access.into(),
    ) {
        Ok(t) => t,
        Err(e) => {
            return set_error_null(match &e {
                edgefirst_image::Error::Tensor(te) => match te {
                    edgefirst_tensor::Error::InvalidArgument(_)
                    | edgefirst_tensor::Error::InvalidShape(_)
                    | edgefirst_tensor::Error::ShapeMismatch(_) => libc::EINVAL,
                    edgefirst_tensor::Error::IoError(io) => io.raw_os_error().unwrap_or(libc::EIO),
                    edgefirst_tensor::Error::NotImplemented(_) => libc::ENOTSUP,
                    _ => libc::ENOMEM,
                },
                edgefirst_image::Error::InvalidShape(_) | edgefirst_image::Error::NotAnImage => {
                    libc::EINVAL
                }
                edgefirst_image::Error::UnsupportedFormat(_)
                | edgefirst_image::Error::NotSupported(_)
                | edgefirst_image::Error::NotImplemented(_) => libc::ENOTSUP,
                edgefirst_image::Error::Io(io) => io.raw_os_error().unwrap_or(libc::EIO),
                _ => libc::ENOMEM,
            });
        }
    };
    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

// ============================================================================
// GPU pitch alignment helpers
// ============================================================================

/// Pitch alignment in bytes that DMA-BUF EGLImage imports require on the
/// GL backend.
///
/// Mali Valhall (e.g. Mali-G310 on i.MX 95) rejects `eglCreateImageKHR` with
/// `EGL_BAD_ALLOC` for any DMA-BUF whose row pitch is not a multiple of this
/// value. Vivante GC7000UL (i.MX 8MP) accepts arbitrary pitches but the
/// requirement is harmless on that path.
///
/// External callers that allocate their own DMA-BUFs (GStreamer plugins,
/// V4L2 ring buffers, custom video pipelines) should size those buffers so
/// that `width * bytes_per_pixel` is a multiple of this constant — see
/// hal_align_width_for_gpu_pitch() for a helper.
///
/// `hal_image_processor_create_image()` applies this alignment automatically
/// when allocating DMA-backed image tensors; the constant is only relevant
/// when the caller manages its own DMA-BUF allocations.
///
/// Returned as a function rather than a `#define` so the value stays in
/// sync with the Rust source if the alignment requirement ever changes.
///
/// @return Required DMA-BUF row pitch alignment in bytes (currently 64)
#[no_mangle]
pub extern "C" fn hal_gpu_dma_buf_pitch_alignment_bytes() -> size_t {
    edgefirst_image::GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES
}

/// Round `width` (in pixels) up so that `width * bpp` is a multiple of
/// the value returned by hal_gpu_dma_buf_pitch_alignment_bytes()
/// (currently 64) **and** an integer pixel count for the given
/// bytes-per-pixel.
///
/// Use this when allocating a DMA-BUF that will later be imported as an
/// EGLImage by HAL's GL backend (or by any GLES driver that requires
/// 64-byte aligned pitches, which currently includes Mali Valhall on
/// i.MX 95 / G310).
///
/// Pre-aligned widths (640, 1280, 1920, 3008, 3840, …) round-trip
/// unchanged. Misaligned widths are bumped up to the next valid value.
///
/// `bpp` is the bytes-per-pixel for the primary plane:
///  - RGBA8 / BGRA8: 4
///  - RGB888:        3
///  - Grey / NV12 luma: 1
///
/// @param width Image width in pixels
/// @param bpp   Bytes per pixel for the primary plane
/// @return Aligned width in pixels (always >= `width`). Returns `width`
///         unchanged if `bpp == 0`, `width == 0`, or if the rounded value
///         would overflow `size_t`.
///
/// @par Example
/// @code{.c}
/// // Allocate an RGBA8 DMA-BUF for a 3004×1688 canvas.
/// // 3004 × 4 = 12016 bytes pitch, NOT 64-aligned, would fail on Mali.
/// // After alignment: width = 3008, pitch = 12032 bytes (64-aligned).
/// size_t aligned_w = hal_align_width_for_gpu_pitch(3004, 4);  // 3008
/// size_t pitch = aligned_w * 4;                                // 12032
/// size_t bytes = pitch * 1688;
/// // ... allocate DMA-BUF of `bytes` bytes, then import via EGL.
/// @endcode
#[no_mangle]
pub extern "C" fn hal_align_width_for_gpu_pitch(width: size_t, bpp: size_t) -> size_t {
    edgefirst_image::align_width_for_gpu_pitch(width, bpp)
}

/// Convenience wrapper that derives `bpp` from a HAL pixel format and dtype,
/// then calls hal_align_width_for_gpu_pitch().
///
/// Use this when you have a HalPixelFormat already (the common case for
/// GStreamer plugins and other HAL clients). The wrapper handles the
/// channels × element-size multiplication so callers don't need to remember
/// per-format BPPs.
///
/// For semi-planar formats (NV12, NV16) this returns the alignment for the
/// luma plane; the chroma plane has the same row pitch in bytes.
///
/// @param width  Image width in pixels
/// @param format Pixel format
/// @param dtype  Element data type
/// @return Aligned width in pixels (always >= `width`). Returns `width`
///         unchanged if the format / dtype combination has no defined BPP.
#[no_mangle]
pub extern "C" fn hal_align_width_for_pixel_format(
    width: size_t,
    format: HalPixelFormat,
    dtype: HalDtype,
) -> size_t {
    let pf = format.to_pixel_format();
    let elem = match dtype {
        HalDtype::U8 | HalDtype::I8 => 1,
        HalDtype::U16 | HalDtype::I16 | HalDtype::F16 => 2,
        HalDtype::U32 | HalDtype::I32 | HalDtype::F32 => 4,
        HalDtype::U64 | HalDtype::I64 | HalDtype::F64 => 8,
    };
    match edgefirst_image::primary_plane_bpp(pf, elem) {
        Some(bpp) => edgefirst_image::align_width_for_gpu_pitch(width, bpp),
        None => width,
    }
}

// ============================================================================
// Plane Descriptor + Image Import
// ============================================================================

/// Per-plane DMA-BUF descriptor for external buffer import.
///
/// The fd is duplicated eagerly in `hal_plane_descriptor_new()` so that a bad
/// fd fails immediately. `hal_import_image()` **consumes** the descriptor —
/// do NOT call `hal_plane_descriptor_free()` after a successful import.
///
/// @code{.c}
/// // Single-plane RGBA
/// struct hal_plane_descriptor *pd = hal_plane_descriptor_new(fd);
/// hal_plane_descriptor_set_stride(pd, bytesperline);  // optional
/// struct hal_tensor *src = hal_import_image(proc, pd, NULL,
///                                            1920, 1080,
///                                            HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
/// // pd is consumed — do NOT free it
///
/// // Multi-plane NV12
/// struct hal_plane_descriptor *y_pd = hal_plane_descriptor_new(y_fd);
/// struct hal_plane_descriptor *uv_pd = hal_plane_descriptor_new(uv_fd);
/// struct hal_tensor *src = hal_import_image(proc, y_pd, uv_pd,
///                                            1920, 1080,
///                                            HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
/// @endcode
///
/// @see hal_plane_descriptor_new, hal_import_image
#[cfg(unix)]
pub struct HalPlaneDescriptor {
    inner: Option<edgefirst_tensor::PlaneDescriptor>,
}

/// Create a new plane descriptor by duplicating a DMA-BUF file descriptor.
///
/// The fd is `dup()`'d immediately — the caller retains ownership of the
/// original fd. A bad fd will cause this call to fail.
///
/// @param fd DMA-BUF file descriptor (caller retains ownership)
/// @return New plane descriptor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: fd is negative
/// - EMFILE/ENFILE: fd dup failed (fd limit reached)
/// - EIO: other dup failure
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_plane_descriptor_new(fd: c_int) -> *mut HalPlaneDescriptor {
    if fd < 0 {
        return set_error_null(libc::EINVAL);
    }
    use std::os::fd::BorrowedFd;
    let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
    let pd = match edgefirst_tensor::PlaneDescriptor::new(borrowed) {
        Ok(pd) => pd,
        Err(e) => {
            let code = match e {
                edgefirst_tensor::Error::IoError(ref io) => io.raw_os_error().unwrap_or(libc::EIO),
                _ => libc::EIO,
            };
            return set_error_null(code);
        }
    };
    Box::into_raw(Box::new(HalPlaneDescriptor { inner: Some(pd) }))
}

/// Free a plane descriptor.
///
/// Only call this if the descriptor was NOT consumed by `hal_import_image()`.
/// Passing NULL is a safe no-op.
///
/// @param pd Plane descriptor handle to free (can be NULL)
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_plane_descriptor_free(pd: *mut HalPlaneDescriptor) {
    if !pd.is_null() {
        drop(unsafe { Box::from_raw(pd) });
    }
}

/// Set the row stride (bytes per row) on a plane descriptor.
///
/// Only needed when the external buffer has row padding. If not set,
/// the buffer is assumed tightly packed.
///
/// @note The caller is responsible for ensuring the DMA-BUF allocation
/// is large enough for the given stride, height, and pixel format.
/// No buffer-size validation is performed because external DMA-BUF sizes
/// are not reliably queryable; an incorrect stride is caught by the EGL
/// driver at import time.
///
/// @param pd     Plane descriptor handle
/// @param stride Row stride in bytes (must be > 0)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL pd, stride is zero, or descriptor already consumed
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_plane_descriptor_set_stride(
    pd: *mut HalPlaneDescriptor,
    stride: size_t,
) -> c_int {
    check_null!(pd);
    if stride == 0 {
        set_error(libc::EINVAL);
        return -1;
    }
    let pd = unsafe { &mut *pd };
    if let Some(inner) = pd.inner.take() {
        pd.inner = Some(inner.with_stride(stride));
        0
    } else {
        set_error(libc::EINVAL);
        -1
    }
}

/// Set the byte offset within the DMA-BUF where image data starts.
///
/// @param pd     Plane descriptor handle
/// @param offset Byte offset
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL pd or descriptor already consumed
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_plane_descriptor_set_offset(
    pd: *mut HalPlaneDescriptor,
    offset: size_t,
) -> c_int {
    check_null!(pd);
    let pd = unsafe { &mut *pd };
    if let Some(inner) = pd.inner.take() {
        pd.inner = Some(inner.with_offset(offset));
        0
    } else {
        set_error(libc::EINVAL);
        -1
    }
}

/// Import an external DMA-BUF image using plane descriptors.
///
/// Both plane descriptors are **always consumed** by this call, whether it
/// succeeds or fails — do NOT call `hal_plane_descriptor_free()` on them
/// after calling `hal_import_image()`.
///
/// The `chroma` parameter is NULL for single-plane formats. For multiplane
/// NV12, pass a second descriptor for the UV plane.
///
/// @note The caller must ensure the DMA-BUF allocation is large enough for
/// the specified width, height, format, and any stride/offset set on the
/// plane descriptors. No buffer-size validation is performed; an
/// undersized buffer may cause GPU faults or EGL import failure.
///
/// @param processor Image processor handle
/// @param image     Plane descriptor for the primary (Y or only) plane (consumed)
/// @param chroma    Plane descriptor for the UV chroma plane, or NULL (consumed)
/// @param width     Image width in pixels
/// @param height    Image height in pixels
/// @param format    Pixel format (HAL_PIXEL_FORMAT_*)
/// @param dtype     Data type (HAL_DTYPE_*)
/// @return New tensor handle on success (free with `hal_tensor_free()`), NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL processor/image, zero dimensions, consumed descriptor
/// - EIO:    Failed to create tensor
/// - ENOTSUP: Not supported on this platform or format not supported
#[no_mangle]
#[cfg(target_os = "linux")]
pub unsafe extern "C" fn hal_import_image(
    processor: *mut HalImageProcessor,
    image: *mut HalPlaneDescriptor,
    chroma: *mut HalPlaneDescriptor,
    width: size_t,
    height: size_t,
    format: HalPixelFormat,
    dtype: HalDtype,
    colorimetry: *const crate::colorimetry::hal_colorimetry,
) -> *mut HalTensor {
    // Guard against double-free: if the caller passes the same pointer for
    // both image and chroma, a second Box::from_raw would free already-freed
    // memory.  Treat same-pointer as "no chroma" and return EINVAL below.
    let same_ptr = !chroma.is_null() && std::ptr::eq(chroma, image);

    // Consume descriptors unconditionally so the "always consumed" contract
    // holds on every code path (including early EINVAL returns).  The Boxes
    // are dropped at end-of-scope, freeing the OwnedFd inside.
    let image_box = if image.is_null() {
        None
    } else {
        Some(unsafe { Box::from_raw(image) })
    };
    let chroma_box = if chroma.is_null() || same_ptr {
        None
    } else {
        Some(unsafe { Box::from_raw(chroma) })
    };

    if same_ptr {
        return set_error_null(libc::EINVAL);
    }

    if processor.is_null() || image_box.is_none() || width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    // Extract the inner PlaneDescriptor from each consumed box
    let image_pd = match image_box.unwrap().inner {
        Some(pd) => pd,
        None => return set_error_null(libc::EINVAL),
    };

    let chroma_pd = match chroma_box {
        None => None,
        Some(cb) => match cb.inner {
            Some(pd) => Some(pd),
            None => return set_error_null(libc::EINVAL),
        },
    };

    let colorimetry_opt = if colorimetry.is_null() {
        None
    } else {
        Some(edgefirst_tensor::Colorimetry::from(unsafe { *colorimetry }))
    };

    let proc_inner = &unsafe { &(*processor) }.inner;
    let dyn_tensor = match proc_inner.import_image(
        image_pd,
        chroma_pd,
        width,
        height,
        format.to_pixel_format(),
        dtype.into(),
        colorimetry_opt,
    ) {
        Ok(t) => t,
        Err(e) => {
            return set_error_null(match &e {
                edgefirst_image::Error::Tensor(te) => match te {
                    edgefirst_tensor::Error::InvalidArgument(_)
                    | edgefirst_tensor::Error::InvalidShape(_)
                    | edgefirst_tensor::Error::ShapeMismatch(_) => libc::EINVAL,
                    edgefirst_tensor::Error::IoError(io) => io.raw_os_error().unwrap_or(libc::EIO),
                    edgefirst_tensor::Error::NotImplemented(_) => libc::ENOTSUP,
                    _ => libc::EIO,
                },
                edgefirst_image::Error::InvalidShape(_) | edgefirst_image::Error::NotAnImage => {
                    libc::EINVAL
                }
                edgefirst_image::Error::UnsupportedFormat(_)
                | edgefirst_image::Error::NotSupported(_)
                | edgefirst_image::Error::NotImplemented(_) => libc::ENOTSUP,
                edgefirst_image::Error::Io(io) => io.raw_os_error().unwrap_or(libc::EIO),
                _ => libc::EIO,
            });
        }
    };

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// cbindgen:ignore
#[no_mangle]
#[cfg(all(unix, not(target_os = "linux")))]
pub unsafe extern "C" fn hal_import_image(
    _processor: *mut HalImageProcessor,
    _image: *mut HalPlaneDescriptor,
    _chroma: *mut HalPlaneDescriptor,
    _width: size_t,
    _height: size_t,
    _format: HalPixelFormat,
    _dtype: HalDtype,
    _colorimetry: *const crate::colorimetry::hal_colorimetry,
) -> *mut HalTensor {
    let same_ptr = !_chroma.is_null() && std::ptr::eq(_chroma, _image);
    // Consume descriptors to uphold the "always consumed" contract.
    if !_image.is_null() {
        drop(unsafe { Box::from_raw(_image) });
    }
    if !_chroma.is_null() && !same_ptr {
        drop(unsafe { Box::from_raw(_chroma) });
    }
    set_error_null(libc::ENOTSUP)
}

/// Free an image processor.
///
/// @param processor Image processor handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_free(processor: *mut HalImageProcessor) {
    if !processor.is_null() {
        drop(unsafe { Box::from_raw(processor) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{hal_tensor_dtype, hal_tensor_free};
    use std::ffi::CString;

    #[test]
    fn test_image_create_and_free() {
        unsafe {
            let image = hal_tensor_new_image(
                640,
                480,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!image.is_null());

            assert_eq!(hal_tensor_width(image), 640);
            assert_eq!(hal_tensor_height(image), 480);
            assert_eq!(hal_tensor_pixel_format(image), HalPixelFormat::Rgb);

            hal_tensor_free(image);
        }
    }

    #[test]
    fn test_image_all_pixel_formats() {
        unsafe {
            let formats = [
                HalPixelFormat::Rgb,
                HalPixelFormat::Rgba,
                HalPixelFormat::Bgra,
                HalPixelFormat::Grey,
                HalPixelFormat::Nv12,
                HalPixelFormat::Nv16,
                HalPixelFormat::Nv24,
                HalPixelFormat::Yuyv,
                HalPixelFormat::Vyuy,
                HalPixelFormat::PlanarRgb,
                HalPixelFormat::PlanarRgba,
            ];

            for fmt in formats {
                // Use dimensions that work for all formats (divisible by 2 for YUV)
                let image = hal_tensor_new_image(
                    320,
                    240,
                    fmt,
                    HalDtype::U8,
                    HalTensorMemory::Mem,
                    crate::tensor::HalCpuAccess::ReadWrite,
                );
                assert!(
                    !image.is_null(),
                    "Failed to create image with format {:?}",
                    fmt
                );

                assert_eq!(hal_tensor_width(image), 320);
                assert_eq!(hal_tensor_height(image), 240);
                assert_eq!(hal_tensor_pixel_format(image), fmt);

                hal_tensor_free(image);
            }
        }
    }

    #[test]
    fn test_image_pixel_format_null() {
        unsafe {
            // NULL image returns Rgb default
            assert_eq!(
                hal_tensor_pixel_format(std::ptr::null()),
                HalPixelFormat::Rgb
            );
        }
    }

    #[test]
    fn test_tensor_set_format() {
        use crate::tensor::{hal_tensor_free, hal_tensor_new, HalTensorMemory};
        unsafe {
            // Create a raw [480, 640, 3] tensor — no format yet
            let shape: [size_t; 3] = [480, 640, 3];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            // Before set_format: no image metadata
            assert_eq!(hal_tensor_width(tensor), 0);

            // Set format to RGB
            let ret = hal_tensor_set_format(tensor, HalPixelFormat::Rgb);
            assert_eq!(ret, 0);
            assert_eq!(hal_tensor_pixel_format(tensor), HalPixelFormat::Rgb);
            assert_eq!(hal_tensor_width(tensor), 640);
            assert_eq!(hal_tensor_height(tensor), 480);

            // Wrong format should fail (4-channel format on 3-channel tensor)
            let ret = hal_tensor_set_format(tensor, HalPixelFormat::Rgba);
            assert_eq!(ret, -1);

            // NULL tensor
            let ret = hal_tensor_set_format(std::ptr::null_mut(), HalPixelFormat::Rgb);
            assert_eq!(ret, -1);

            hal_tensor_free(tensor);
        }
    }

    #[test]
    fn test_region_and_crop() {
        let region = hal_region_new(10, 20, 100, 200);
        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.width, 100);
        assert_eq!(region.height, 200);

        unsafe {
            let mut crop = hal_crop_new();
            assert!(!crop.has_source);
            assert_eq!(crop.fit, HAL_FIT_STRETCH);

            hal_crop_set_source(&mut crop, &region);
            assert!(crop.has_source);
            assert_eq!(crop.source.x, 10);

            hal_crop_set_letterbox(&mut crop, 255, 128, 0, 255);
            assert_eq!(crop.fit, HAL_FIT_LETTERBOX);
            assert_eq!(crop.pad, [255, 128, 0, 255]);
        }
    }

    #[test]
    fn test_crop_set_null_source() {
        unsafe {
            let mut crop = hal_crop_new();
            // Setting NULL source is a no-op (whole source).
            hal_crop_set_source(&mut crop, std::ptr::null());
            assert!(!crop.has_source);
        }
    }

    #[test]
    fn test_crop_conversion_to_rust() {
        let mut crop = hal_crop_new();
        let region = hal_region_new(10, 20, 100, 200);

        unsafe {
            hal_crop_set_source(&mut crop, &region);
            hal_crop_set_letterbox(&mut crop, 255, 0, 0, 255);
        }

        let rust_crop: Crop = crop.into();
        assert!(rust_crop.source.is_some());
        assert!(matches!(rust_crop.fit, Fit::Letterbox { pad } if pad == [255, 0, 0, 255]));
    }

    #[test]
    fn test_rotation_conversion() {
        let none: Rotation = HalRotation::None.into();
        assert!(matches!(none, Rotation::None));

        let rot90: Rotation = HalRotation::Rotate90.into();
        assert!(matches!(rot90, Rotation::Clockwise90));

        let rot180: Rotation = HalRotation::Rotate180.into();
        assert!(matches!(rot180, Rotation::Rotate180));

        let rot270: Rotation = HalRotation::Rotate270.into();
        assert!(matches!(rot270, Rotation::CounterClockwise90));
    }

    #[test]
    fn test_flip_conversion() {
        let none: Flip = HalFlip::None.into();
        assert!(matches!(none, Flip::None));

        let horizontal: Flip = HalFlip::Horizontal.into();
        assert!(matches!(horizontal, Flip::Horizontal));

        let vertical: Flip = HalFlip::Vertical.into();
        assert!(matches!(vertical, Flip::Vertical));
    }

    #[test]
    fn test_processor_create_and_free() {
        unsafe {
            let processor = hal_image_processor_new();
            // Processor may be NULL if no backend is available
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_processor_convert_null_params() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                // No processor available, skip test
                return;
            }

            let src = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            let dst = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!src.is_null());
            assert!(!dst.is_null());

            // NULL processor
            assert_eq!(
                hal_image_processor_convert(
                    std::ptr::null_mut(),
                    src,
                    dst,
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            // NULL src
            assert_eq!(
                hal_image_processor_convert(
                    processor,
                    std::ptr::null(),
                    dst,
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            // NULL dst
            assert_eq!(
                hal_image_processor_convert(
                    processor,
                    src,
                    std::ptr::null_mut(),
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            hal_tensor_free(src);
            hal_tensor_free(dst);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_save_jpeg_null_params() {
        unsafe {
            let image = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!image.is_null());

            let path = CString::new("/tmp/test.jpg").unwrap();

            // NULL image
            assert_eq!(
                hal_tensor_save_jpeg(std::ptr::null(), path.as_ptr(), 80),
                -1
            );

            // NULL path
            assert_eq!(hal_tensor_save_jpeg(image, std::ptr::null(), 80), -1);

            hal_tensor_free(image);
        }
    }

    #[test]
    fn test_null_handling() {
        unsafe {
            assert_eq!(hal_tensor_width(std::ptr::null()), 0);
            assert_eq!(hal_tensor_height(std::ptr::null()), 0);
            hal_image_processor_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_image_is_planar() {
        unsafe {
            let rgb = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!rgb.is_null());
            assert!(!hal_tensor_is_planar(rgb));
            hal_tensor_free(rgb);

            let nv12 = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Nv12,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!nv12.is_null());
            assert!(hal_tensor_is_planar(nv12));
            hal_tensor_free(nv12);

            let planar = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!planar.is_null());
            assert!(hal_tensor_is_planar(planar));
            hal_tensor_free(planar);

            // NULL image returns false
            assert!(!hal_tensor_is_planar(std::ptr::null()));
        }
    }

    #[test]
    fn test_image_channels() {
        unsafe {
            let rgb = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_channels(rgb), 3);
            hal_tensor_free(rgb);

            let rgba = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!rgba.is_null());
            assert_eq!(hal_tensor_channels(rgba), 4);
            hal_tensor_free(rgba);

            let grey = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Grey,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!grey.is_null());
            assert_eq!(hal_tensor_channels(grey), 1);
            hal_tensor_free(grey);

            // NULL image returns 0
            assert_eq!(hal_tensor_channels(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_image_row_stride() {
        unsafe {
            let rgb = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_row_stride(rgb), 300); // 100 * 3
            hal_tensor_free(rgb);

            let planar = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!planar.is_null());
            assert_eq!(hal_tensor_row_stride(planar), 100); // planar: width
            hal_tensor_free(planar);

            // NULL image returns 0
            assert_eq!(hal_tensor_row_stride(std::ptr::null()), 0);
        }
    }
    #[test]
    fn test_image_processor_draw_decoded_masks_null_params() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let dst = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!dst.is_null());

            // NULL processor
            assert_eq!(
                hal_image_processor_draw_decoded_masks(
                    std::ptr::null_mut(),
                    dst,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1.0,
                    std::ptr::null(),
                    HalColorMode::Class,
                ),
                -1
            );

            // NULL dst
            assert_eq!(
                hal_image_processor_draw_decoded_masks(
                    processor,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1.0,
                    std::ptr::null(),
                    HalColorMode::Class,
                ),
                -1
            );

            hal_tensor_free(dst);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_set_class_colors_null() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            // NULL colors
            assert_eq!(
                hal_image_processor_set_class_colors(processor, std::ptr::null(), 3),
                -1
            );

            // Zero count
            let colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
            assert_eq!(
                hal_image_processor_set_class_colors(processor, colors.as_ptr(), 0),
                -1
            );

            // NULL processor
            assert_eq!(
                hal_image_processor_set_class_colors(std::ptr::null_mut(), colors.as_ptr(), 3),
                -1
            );

            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_map_create_null() {
        use crate::tensor::hal_tensor_map_create;
        unsafe {
            let map = hal_tensor_map_create(std::ptr::null());
            assert!(map.is_null());
        }
    }

    #[test]
    fn test_image_map_create_rgb() {
        use crate::tensor::{
            hal_tensor_map_create, hal_tensor_map_data_const, hal_tensor_map_size,
            hal_tensor_map_unmap,
        };

        unsafe {
            let image = hal_tensor_new_image(
                100,
                50,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!image.is_null());

            let map = hal_tensor_map_create(image);
            assert!(!map.is_null());

            // RGB: 100 * 50 * 3 = 15000 bytes
            assert_eq!(hal_tensor_map_size(map), 15000);

            let data = hal_tensor_map_data_const(map);
            assert!(!data.is_null());

            hal_tensor_map_unmap(map);
            hal_tensor_free(image);
        }
    }

    #[test]
    fn test_image_map_create_planar_rgb() {
        use crate::tensor::{
            hal_tensor_map_create, hal_tensor_map_data_const, hal_tensor_map_size,
            hal_tensor_map_unmap,
        };

        unsafe {
            let image = hal_tensor_new_image(
                64,
                64,
                HalPixelFormat::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!image.is_null());

            let map = hal_tensor_map_create(image);
            assert!(!map.is_null());

            // PLANAR_RGB: 3 * 64 * 64 = 12288 bytes
            assert_eq!(hal_tensor_map_size(map), 12288);

            let data = hal_tensor_map_data_const(map);
            assert!(!data.is_null());

            hal_tensor_map_unmap(map);
            hal_tensor_free(image);
        }
    }

    #[test]
    fn test_pixel_format_roundtrip() {
        // Test all PixelFormat conversions roundtrip
        let formats = [
            HalPixelFormat::Yuyv,
            HalPixelFormat::Nv12,
            HalPixelFormat::Nv16,
            HalPixelFormat::Rgba,
            HalPixelFormat::Rgb,
            HalPixelFormat::Grey,
            HalPixelFormat::PlanarRgb,
            HalPixelFormat::PlanarRgba,
            HalPixelFormat::Bgra,
            HalPixelFormat::Vyuy,
            HalPixelFormat::Nv24,
        ];

        for format in formats {
            let pf = format.to_pixel_format();
            let back = HalPixelFormat::from_pixel_format(pf);
            assert_eq!(back, format, "Roundtrip failed for {:?}", format);
        }
        // Every core PixelFormat must have a distinct C-API mapping (no silent
        // fallback to Rgb). Add new core variants here so a missing
        // HalPixelFormat counterpart is caught instead of silently mis-mapping.
        let core = [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
            PixelFormat::Grey,
            PixelFormat::Yuyv,
            PixelFormat::Vyuy,
            PixelFormat::Nv12,
            PixelFormat::Nv16,
            PixelFormat::Nv24,
            PixelFormat::PlanarRgb,
            PixelFormat::PlanarRgba,
        ];
        for pf in core {
            let hal = HalPixelFormat::from_pixel_format(pf);
            assert_eq!(
                hal.to_pixel_format(),
                pf,
                "PixelFormat {pf:?} has no distinct C-API mapping (fell back to Rgb?)"
            );
        }
    }

    #[test]
    fn test_image_processor_create_image_null_params() {
        unsafe {
            // NULL processor
            let img = hal_image_processor_create_image(
                std::ptr::null_mut(),
                640,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(img.is_null());

            // Zero width
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }
            let img = hal_image_processor_create_image(
                processor,
                0,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(img.is_null());

            // Zero height
            let img = hal_image_processor_create_image(
                processor,
                640,
                0,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(img.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_create_image() {
        use crate::tensor::{
            hal_tensor_map_create, hal_tensor_map_data_const, hal_tensor_map_size,
            hal_tensor_map_unmap,
        };

        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                eprintln!("SKIPPED: test_image_processor_create_image — no processor available");
                return;
            }

            let formats = [
                (HalPixelFormat::Rgb, 3),
                (HalPixelFormat::Rgba, 4),
                (HalPixelFormat::Grey, 1),
            ];

            for (fmt, channels) in formats {
                let img = hal_image_processor_create_image(
                    processor,
                    320,
                    240,
                    fmt,
                    HalDtype::U8,
                    crate::tensor::HalCpuAccess::ReadWrite,
                );
                assert!(!img.is_null(), "create_image failed for {:?}", fmt);

                assert_eq!(hal_tensor_width(img), 320);
                assert_eq!(hal_tensor_height(img), 240);
                assert_eq!(hal_tensor_pixel_format(img), fmt);

                // Verify the image is mappable
                let map = hal_tensor_map_create(img);
                assert!(!map.is_null(), "map failed for {:?}", fmt);
                assert_eq!(hal_tensor_map_size(map), 320 * 240 * channels);
                assert!(!hal_tensor_map_data_const(map).is_null());
                hal_tensor_map_unmap(map);

                hal_tensor_free(img);
            }

            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_create_image_i8() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                eprintln!("SKIPPED: test_image_processor_create_image_i8 — no processor available");
                return;
            }

            // I8 image via create_image
            let img = hal_image_processor_create_image(
                processor,
                320,
                240,
                HalPixelFormat::Rgb,
                HalDtype::I8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!img.is_null(), "create_image with I8 dtype failed");
            assert_eq!(hal_tensor_dtype(img), HalDtype::I8);
            assert_eq!(hal_tensor_width(img), 320);
            assert_eq!(hal_tensor_height(img), 240);

            hal_tensor_free(img);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_create_image_convert() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                eprintln!(
                    "SKIPPED: test_image_processor_create_image_convert — no processor available"
                );
                return;
            }

            // Create source image via create_image (may be PBO or Mem)
            let src = hal_image_processor_create_image(
                processor,
                320,
                240,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!src.is_null());

            // Create destination image via create_image
            let dst = hal_image_processor_create_image(
                processor,
                160,
                120,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                crate::tensor::HalCpuAccess::ReadWrite,
            );
            assert!(!dst.is_null());

            // Convert should succeed regardless of backing (PBO, DMA, or Mem)
            let ret = hal_image_processor_convert(
                processor,
                src,
                dst,
                HalRotation::None,
                HalFlip::None,
                std::ptr::null(),
            );
            assert_eq!(ret, 0, "convert() with create_image tensors failed");

            hal_tensor_free(src);
            hal_tensor_free(dst);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_tensor_set_format_planar() {
        use crate::tensor::{hal_tensor_free, hal_tensor_new, HalTensorMemory};
        unsafe {
            // Create a raw [3, 480, 640] tensor — no format yet
            let shape: [size_t; 3] = [3, 480, 640];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            // Set format to PlanarRgb
            let ret = hal_tensor_set_format(tensor, HalPixelFormat::PlanarRgb);
            assert_eq!(ret, 0);
            assert_eq!(hal_tensor_pixel_format(tensor), HalPixelFormat::PlanarRgb);
            assert_eq!(hal_tensor_width(tensor), 640);
            assert_eq!(hal_tensor_height(tensor), 480);

            hal_tensor_free(tensor);
        }
    }

    #[test]
    fn test_tensor_set_format_nv12() {
        use crate::tensor::{hal_tensor_free, hal_tensor_new, HalTensorMemory};
        unsafe {
            // Create a raw [720, 640] tensor — NV12: 480 * 3/2 = 720
            let shape: [size_t; 2] = [720, 640];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                shape.len(),
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            // Set format to Nv12
            let ret = hal_tensor_set_format(tensor, HalPixelFormat::Nv12);
            assert_eq!(ret, 0);
            assert_eq!(hal_tensor_pixel_format(tensor), HalPixelFormat::Nv12);
            assert_eq!(hal_tensor_width(tensor), 640);
            assert_eq!(hal_tensor_height(tensor), 480);

            hal_tensor_free(tensor);
        }
    }

    /// Helper: create a valid fd via memfd_create (Linux) or pipe (other Unix).
    #[cfg(target_os = "linux")]
    fn make_test_fd() -> c_int {
        unsafe { libc::memfd_create(c"test".as_ptr(), 0) }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    fn make_test_fd() -> c_int {
        let mut fds: [c_int; 2] = [0; 2];
        unsafe { libc::pipe(fds.as_mut_ptr()) };
        unsafe { libc::close(fds[1]) };
        fds[0]
    }

    #[test]
    fn test_plane_descriptor_bad_fd() {
        unsafe {
            let pd = hal_plane_descriptor_new(-1);
            assert!(pd.is_null());
        }
    }

    #[test]
    fn test_plane_descriptor_free_null() {
        unsafe {
            // NULL is a safe no-op
            hal_plane_descriptor_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_plane_descriptor_set_stride() {
        let fd = make_test_fd();
        assert!(fd >= 0, "make_test_fd failed");
        unsafe {
            let pd = hal_plane_descriptor_new(fd);
            libc::close(fd); // original fd — descriptor holds its own dup
            assert!(!pd.is_null());

            // stride = 0 should fail
            assert_eq!(hal_plane_descriptor_set_stride(pd, 0), -1);

            // valid stride
            assert_eq!(hal_plane_descriptor_set_stride(pd, 2048), 0);

            hal_plane_descriptor_free(pd);
        }
    }

    #[test]
    fn test_plane_descriptor_set_offset() {
        let fd = make_test_fd();
        assert!(fd >= 0);
        unsafe {
            let pd = hal_plane_descriptor_new(fd);
            libc::close(fd);
            assert!(!pd.is_null());

            assert_eq!(hal_plane_descriptor_set_offset(pd, 4096), 0);
            // offset = 0 is also valid
            assert_eq!(hal_plane_descriptor_set_offset(pd, 0), 0);

            hal_plane_descriptor_free(pd);
        }
    }

    #[test]
    fn test_plane_descriptor_null_set_stride() {
        unsafe {
            assert_eq!(
                hal_plane_descriptor_set_stride(std::ptr::null_mut(), 2048),
                -1
            );
        }
    }

    #[test]
    fn test_plane_descriptor_null_set_offset() {
        unsafe {
            assert_eq!(hal_plane_descriptor_set_offset(std::ptr::null_mut(), 0), -1);
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_import_image_null_processor() {
        let fd = make_test_fd();
        assert!(fd >= 0);
        unsafe {
            let pd = hal_plane_descriptor_new(fd);
            libc::close(fd);
            assert!(!pd.is_null());

            // NULL processor — pd is still consumed (always-consume contract)
            let result = hal_import_image(
                std::ptr::null_mut(),
                pd,
                std::ptr::null_mut(),
                640,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                std::ptr::null(),
            );
            assert!(result.is_null());
            // pd is consumed — do NOT free
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_import_image_null_image() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let result = hal_import_image(
                processor,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                640,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                std::ptr::null(),
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_import_image_zero_width() {
        let fd = make_test_fd();
        assert!(fd >= 0);
        unsafe {
            let pd = hal_plane_descriptor_new(fd);
            libc::close(fd);
            assert!(!pd.is_null());

            let processor = hal_image_processor_new();
            if processor.is_null() {
                hal_plane_descriptor_free(pd);
                return;
            }

            let result = hal_import_image(
                processor,
                pd,
                std::ptr::null_mut(),
                0,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                std::ptr::null(),
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_import_image_zero_height() {
        let fd = make_test_fd();
        assert!(fd >= 0);
        unsafe {
            let pd = hal_plane_descriptor_new(fd);
            libc::close(fd);
            assert!(!pd.is_null());

            let processor = hal_image_processor_new();
            if processor.is_null() {
                hal_plane_descriptor_free(pd);
                return;
            }

            let result = hal_import_image(
                processor,
                pd,
                std::ptr::null_mut(),
                640,
                0,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                std::ptr::null(),
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_import_image_chroma_with_packed_format() {
        // Passing a chroma descriptor with a non-semi-planar format should fail
        let fd1 = make_test_fd();
        let fd2 = make_test_fd();
        assert!(fd1 >= 0 && fd2 >= 0);
        unsafe {
            let pd = hal_plane_descriptor_new(fd1);
            let cpd = hal_plane_descriptor_new(fd2);
            libc::close(fd1);
            libc::close(fd2);
            assert!(!pd.is_null() && !cpd.is_null());

            let processor = hal_image_processor_new();
            if processor.is_null() {
                hal_plane_descriptor_free(pd);
                hal_plane_descriptor_free(cpd);
                return;
            }

            // RGBA is packed — chroma descriptor should cause ENOTSUP
            let result = hal_import_image(
                processor,
                pd,
                cpd,
                640,
                480,
                HalPixelFormat::Rgba,
                HalDtype::U8,
                std::ptr::null(),
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_row_stride_with_explicit_stride() {
        unsafe {
            // Create an RGBA image and set an explicit stride via the Rust API
            let mut t = edgefirst_tensor::TensorDyn::image(
                100,
                100,
                edgefirst_tensor::PixelFormat::Rgba,
                edgefirst_tensor::DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            t.set_row_stride(512).unwrap();

            let tensor = Box::into_raw(Box::new(HalTensor { inner: t }));

            // hal_tensor_row_stride should return the explicit stride, not width*channels
            let stride = hal_tensor_row_stride(tensor);
            assert_eq!(stride, 512);

            // Verify width*channels would be 400 (different from 512)
            let width = crate::image::hal_tensor_width(tensor);
            let channels = crate::image::hal_tensor_channels(tensor);
            assert_eq!(width * channels, 400);

            hal_tensor_free(tensor);
        }
    }
}
