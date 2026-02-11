// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image C API - Hardware-accelerated image processing.
//!
//! This module provides image conversion and manipulation functions with
//! support for hardware acceleration (G2D, OpenGL) when available.

use crate::decoder::{HalDetectBoxList, HalSegmentationList};
use crate::error::{set_error, set_error_null};
use crate::tensor::{HalTensor, HalTensorMemory};
use crate::{check_null, check_null_ret_null, try_or_errno, try_or_null};
use edgefirst_image::{
    Crop, Flip, ImageProcessor, ImageProcessorTrait, Rect, Rotation, TensorImage, TensorImageRef,
    GREY, NV12, NV16, PLANAR_RGB, PLANAR_RGBA, RGB, RGBA, YUYV,
};
use edgefirst_tensor::{TensorMemory, TensorTrait};
use libc::{c_char, c_int, c_void, size_t};
use std::ffi::CStr;

/// Image pixel format (FourCC codes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalFourcc {
    /// 8-bit interleaved YUV422, limited range (YUYV)
    Yuyv = 0,
    /// 8-bit planar YUV420, limited range (NV12)
    Nv12 = 1,
    /// 8-bit planar YUV422, limited range (NV16)
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
}

impl HalFourcc {
    fn to_fourcc(self) -> four_char_code::FourCharCode {
        match self {
            HalFourcc::Yuyv => YUYV,
            HalFourcc::Nv12 => NV12,
            HalFourcc::Nv16 => NV16,
            HalFourcc::Rgba => RGBA,
            HalFourcc::Rgb => RGB,
            HalFourcc::Grey => GREY,
            HalFourcc::PlanarRgb => PLANAR_RGB,
            HalFourcc::PlanarRgba => PLANAR_RGBA,
        }
    }

    fn from_fourcc(fourcc: four_char_code::FourCharCode) -> Option<Self> {
        if fourcc == YUYV {
            Some(HalFourcc::Yuyv)
        } else if fourcc == NV12 {
            Some(HalFourcc::Nv12)
        } else if fourcc == NV16 {
            Some(HalFourcc::Nv16)
        } else if fourcc == RGBA {
            Some(HalFourcc::Rgba)
        } else if fourcc == RGB {
            Some(HalFourcc::Rgb)
        } else if fourcc == GREY {
            Some(HalFourcc::Grey)
        } else if fourcc == PLANAR_RGB {
            Some(HalFourcc::PlanarRgb)
        } else if fourcc == PLANAR_RGBA {
            Some(HalFourcc::PlanarRgba)
        } else {
            None
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

/// Rectangle structure for defining regions.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct HalRect {
    /// Left edge (x coordinate)
    pub left: size_t,
    /// Top edge (y coordinate)
    pub top: size_t,
    /// Width of the rectangle
    pub width: size_t,
    /// Height of the rectangle
    pub height: size_t,
}

impl From<HalRect> for Rect {
    fn from(rect: HalRect) -> Self {
        Rect::new(rect.left, rect.top, rect.width, rect.height)
    }
}

/// Crop configuration for image conversion.
///
/// Specifies source crop region, destination placement, and background color.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalCrop {
    /// Source rectangle to crop from
    pub src_rect: HalRect,
    /// Destination rectangle to place crop into
    pub dst_rect: HalRect,
    /// Background color (RGBA)
    pub dst_color: [u8; 4],
    /// Whether src_rect is set
    pub has_src_rect: bool,
    /// Whether dst_rect is set
    pub has_dst_rect: bool,
    /// Whether dst_color is set
    pub has_dst_color: bool,
}

impl Default for HalCrop {
    fn default() -> Self {
        Self {
            src_rect: HalRect::default(),
            dst_rect: HalRect::default(),
            dst_color: [0, 0, 0, 255],
            has_src_rect: false,
            has_dst_rect: false,
            has_dst_color: false,
        }
    }
}

impl From<HalCrop> for Crop {
    fn from(crop: HalCrop) -> Self {
        let mut result = Crop::default();
        if crop.has_src_rect {
            result = result.with_src_rect(Some(crop.src_rect.into()));
        }
        if crop.has_dst_rect {
            result = result.with_dst_rect(Some(crop.dst_rect.into()));
        }
        if crop.has_dst_color {
            result = result.with_dst_color(Some(crop.dst_color));
        }
        result
    }
}

/// Opaque tensor image type.
///
/// A TensorImage combines a tensor with image format metadata (width, height, fourcc).
pub struct HalTensorImage {
    inner: TensorImage,
}

/// Opaque image processor type.
///
/// The ImageProcessor handles format conversion with hardware acceleration when available.
pub struct HalImageProcessor {
    inner: ImageProcessor,
}

// ============================================================================
// Rect and Crop Helper Functions
// ============================================================================

/// Create a new rectangle.
///
/// @param left Left edge (x coordinate)
/// @param top Top edge (y coordinate)
/// @param width Width of the rectangle
/// @param height Height of the rectangle
/// @return New rectangle structure
#[no_mangle]
pub extern "C" fn hal_rect_new(
    left: size_t,
    top: size_t,
    width: size_t,
    height: size_t,
) -> HalRect {
    HalRect {
        left,
        top,
        width,
        height,
    }
}

/// Create a new default crop configuration.
///
/// @return New crop structure with all fields unset
#[no_mangle]
pub extern "C" fn hal_crop_new() -> HalCrop {
    HalCrop::default()
}

/// Set the source rectangle for a crop configuration.
///
/// @param crop Crop configuration to modify
/// @param rect Source rectangle (can be NULL to clear)
#[no_mangle]
pub unsafe extern "C" fn hal_crop_set_src_rect(crop: *mut HalCrop, rect: *const HalRect) {
    if crop.is_null() {
        return;
    }
    unsafe {
        if rect.is_null() {
            (*crop).has_src_rect = false;
        } else {
            (*crop).src_rect = *rect;
            (*crop).has_src_rect = true;
        }
    }
}

/// Set the destination rectangle for a crop configuration.
///
/// @param crop Crop configuration to modify
/// @param rect Destination rectangle (can be NULL to clear)
#[no_mangle]
pub unsafe extern "C" fn hal_crop_set_dst_rect(crop: *mut HalCrop, rect: *const HalRect) {
    if crop.is_null() {
        return;
    }
    unsafe {
        if rect.is_null() {
            (*crop).has_dst_rect = false;
        } else {
            (*crop).dst_rect = *rect;
            (*crop).has_dst_rect = true;
        }
    }
}

/// Set the background color for a crop configuration.
///
/// @param crop Crop configuration to modify
/// @param r Red component (0-255)
/// @param g Green component (0-255)
/// @param b Blue component (0-255)
/// @param a Alpha component (0-255)
#[no_mangle]
pub unsafe extern "C" fn hal_crop_set_dst_color(crop: *mut HalCrop, r: u8, g: u8, b: u8, a: u8) {
    if crop.is_null() {
        return;
    }
    unsafe {
        (*crop).dst_color = [r, g, b, a];
        (*crop).has_dst_color = true;
    }
}

// ============================================================================
// TensorImage Lifecycle Functions
// ============================================================================

/// Create a new empty tensor image.
///
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param fourcc Pixel format (HAL_FOURCC_*)
/// @param memory Memory allocation type (HAL_TENSOR_DMA recommended)
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (zero dimensions, unsupported format)
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_new(
    width: size_t,
    height: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensorImage {
    if width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    let mem_opt: Option<TensorMemory> = memory.into();
    let image = try_or_null!(
        TensorImage::new(width, height, fourcc.to_fourcc(), mem_opt),
        libc::ENOMEM
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Load an image from memory (JPEG or PNG).
///
/// Automatically detects format and decodes the image.
///
/// @param data Pointer to image data
/// @param len Length of image data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode image
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_load(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensorImage {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let image = try_or_null!(
        TensorImage::load(data_slice, Some(fourcc.to_fourcc()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Load an image from a file (JPEG or PNG).
///
/// @param path Path to the image file
/// @param fourcc Output pixel format
/// @param memory Memory allocation type
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL path)
/// - ENOENT: File not found
/// - EBADMSG: Failed to decode image
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_load_file(
    path: *const c_char,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensorImage {
    check_null_ret_null!(path);

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return set_error_null(libc::EINVAL),
    };

    let data = match std::fs::read(path_str) {
        Ok(d) => d,
        Err(e) => {
            return set_error_null(if e.kind() == std::io::ErrorKind::NotFound {
                libc::ENOENT
            } else {
                libc::EIO
            })
        }
    };

    let mem_opt: Option<TensorMemory> = memory.into();
    let image = try_or_null!(
        TensorImage::load(&data, Some(fourcc.to_fourcc()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Create a tensor image from an existing u8 tensor.
///
/// Takes ownership of the tensor. The tensor must be u8 dtype with a 3D shape
/// matching the specified pixel format.
///
/// @param tensor u8 tensor to take ownership of (must not be used after this call)
/// @param fourcc Pixel format describing the tensor layout
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL tensor, tensor is not u8 dtype, or shape is invalid for format
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_from_tensor(
    tensor: *mut HalTensor,
    fourcc: HalFourcc,
) -> *mut HalTensorImage {
    check_null_ret_null!(tensor);

    // Take ownership of the tensor
    let boxed = unsafe { Box::from_raw(tensor) };

    // Must be u8 dtype
    let u8_tensor = match *boxed {
        HalTensor::U8(t) => t,
        _ => return set_error_null(libc::EINVAL),
    };

    let image = try_or_null!(
        TensorImage::from_tensor(u8_tensor, fourcc.to_fourcc()),
        libc::EINVAL
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Load a JPEG image from memory.
///
/// @param data Pointer to JPEG data
/// @param len Length of JPEG data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode JPEG
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_load_jpeg(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensorImage {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let image = try_or_null!(
        TensorImage::load_jpeg(data_slice, Some(fourcc.to_fourcc()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Load a PNG image from memory.
///
/// @param data Pointer to PNG data
/// @param len Length of PNG data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor image handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode PNG
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_load_png(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensorImage {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let image = try_or_null!(
        TensorImage::load_png(data_slice, Some(fourcc.to_fourcc()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensorImage { inner: image }))
}

/// Save a tensor image as JPEG.
///
/// @param image Tensor image to save
/// @param path Output file path
/// @param quality JPEG quality (1-100, 0 for default)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL image/path)
/// - EIO: Failed to write file
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_save_jpeg(
    image: *const HalTensorImage,
    path: *const c_char,
    quality: c_int,
) -> c_int {
    check_null!(image, path);

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return set_error(libc::EINVAL),
    };

    let quality = if quality <= 0 || quality > 100 {
        80
    } else {
        quality as u8
    };

    try_or_errno!(
        unsafe { &(*image) }.inner.save_jpeg(path_str, quality),
        libc::EIO
    );
    0
}

/// Free a tensor image.
///
/// @param image Tensor image handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_free(image: *mut HalTensorImage) {
    if !image.is_null() {
        drop(unsafe { Box::from_raw(image) });
    }
}

// ============================================================================
// TensorImage Property Functions
// ============================================================================

/// Get the width of a tensor image.
///
/// @param image Tensor image handle
/// @return Width in pixels, or 0 if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_width(image: *const HalTensorImage) -> size_t {
    if image.is_null() {
        return 0;
    }
    unsafe { &(*image) }.inner.width()
}

/// Get the height of a tensor image.
///
/// @param image Tensor image handle
/// @return Height in pixels, or 0 if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_height(image: *const HalTensorImage) -> size_t {
    if image.is_null() {
        return 0;
    }
    unsafe { &(*image) }.inner.height()
}

/// Get the pixel format of a tensor image.
///
/// @param image Tensor image handle
/// @return Pixel format, or HAL_FOURCC_RGB if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_fourcc(image: *const HalTensorImage) -> HalFourcc {
    if image.is_null() {
        return HalFourcc::Rgb;
    }
    HalFourcc::from_fourcc(unsafe { &(*image) }.inner.fourcc()).unwrap_or(HalFourcc::Rgb)
}

/// Check if a tensor image uses a planar pixel format.
///
/// Planar formats store each color channel in a separate plane (e.g., NV12,
/// NV16, PLANAR_RGB, PLANAR_RGBA), while interleaved formats store channels
/// together per pixel (e.g., RGB, RGBA, YUYV).
///
/// @param image Tensor image handle
/// @return true if the image uses a planar format, false if interleaved or NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_is_planar(image: *const HalTensorImage) -> bool {
    if image.is_null() {
        return false;
    }
    unsafe { &(*image) }.inner.is_planar()
}

/// Get the number of channels in a tensor image.
///
/// Returns the number of color channels (e.g., 3 for RGB, 4 for RGBA,
/// 1 for GREY, 2 for NV12).
///
/// @param image Tensor image handle
/// @return Number of channels, or 0 if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_channels(image: *const HalTensorImage) -> size_t {
    if image.is_null() {
        return 0;
    }
    unsafe { &(*image) }.inner.channels()
}

/// Get the row stride of a tensor image in bytes.
///
/// For planar formats this is equal to the width. For interleaved formats
/// this is width * channels.
///
/// @param image Tensor image handle
/// @return Row stride in bytes, or 0 if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_row_stride(image: *const HalTensorImage) -> size_t {
    if image.is_null() {
        return 0;
    }
    unsafe { &(*image) }.inner.row_stride()
}

/// Clone the file descriptor associated with a tensor image (Linux only).
///
/// Creates a new owned file descriptor that the caller must close().
///
/// @param image Tensor image handle
/// @return New file descriptor on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL image
/// - ENOTSUP: Image memory type doesn't support file descriptors
/// - EIO: Failed to clone file descriptor
#[no_mangle]
#[cfg(unix)]
pub unsafe extern "C" fn hal_tensor_image_clone_fd(image: *const HalTensorImage) -> c_int {
    use std::os::fd::IntoRawFd;

    check_null!(image);
    match unsafe { &(*image) }.inner.tensor().clone_fd() {
        Ok(fd) => fd.into_raw_fd(),
        Err(_) => set_error(libc::EIO),
    }
}

/// Clone file descriptor stub for non-Unix platforms.
#[no_mangle]
#[cfg(not(unix))]
pub unsafe extern "C" fn hal_tensor_image_clone_fd(_image: *const HalTensorImage) -> c_int {
    set_error(libc::ENOTSUP)
}

/// Get the underlying tensor of a tensor image.
///
/// The returned tensor is borrowed and valid only during the image's lifetime.
/// Note: This returns an opaque pointer that provides type-erased access.
///
/// @param image Tensor image handle
/// @return Pointer to the underlying tensor data info, or NULL if image is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_image_tensor(image: *const HalTensorImage) -> *const c_void {
    if image.is_null() {
        return std::ptr::null();
    }
    unsafe { &(*image) }.inner.tensor() as *const _ as *const c_void
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

/// Convert an image to a different format/size.
///
/// Performs format conversion, scaling, rotation, flip, and crop operations.
///
/// @param processor Image processor handle
/// @param src Source image
/// @param dst Destination image (must be pre-allocated with desired dimensions)
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
    src: *const HalTensorImage,
    dst: *mut HalTensorImage,
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

/// Convert an image into a borrowed tensor reference.
///
/// This enables zero-copy preprocessing directly into a model's pre-allocated
/// input buffer. The destination tensor must be u8 dtype with a 3D shape
/// matching the specified output format.
///
/// @param processor Image processor handle
/// @param src Source image
/// @param dst_tensor Destination u8 tensor (not consumed, must remain valid)
/// @param dst_fourcc Output pixel format for the destination
/// @param rotation Rotation to apply
/// @param flip Flip to apply
/// @param crop Crop configuration (can be NULL for no crop)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor/src/dst_tensor, wrong dtype, invalid shape)
/// - EIO: Conversion failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_convert_ref(
    processor: *mut HalImageProcessor,
    src: *const HalTensorImage,
    dst_tensor: *mut HalTensor,
    dst_fourcc: HalFourcc,
    rotation: HalRotation,
    flip: HalFlip,
    crop: *const HalCrop,
) -> c_int {
    check_null!(processor, src, dst_tensor);

    // Get a mutable reference to the u8 tensor
    let tensor_ref = match unsafe { &mut *dst_tensor } {
        HalTensor::U8(t) => t,
        _ => return set_error(libc::EINVAL),
    };

    let mut dst_ref = try_or_errno!(
        TensorImageRef::from_borrowed_tensor(tensor_ref, dst_fourcc.to_fourcc()),
        libc::EINVAL
    );

    let crop_config = if crop.is_null() {
        Crop::default()
    } else {
        unsafe { *crop }.into()
    };

    try_or_errno!(
        unsafe { &mut (*processor) }.inner.convert_ref(
            &unsafe { &(*src) }.inner,
            &mut dst_ref,
            rotation.into(),
            flip.into(),
            crop_config,
        ),
        libc::EIO
    );
    0
}

/// Render detection boxes and segmentation masks onto an image.
///
/// Draws bounding boxes (with labels) and segmentation overlays on the
/// destination image. Uses hardware acceleration (OpenGL) when available,
/// falling back to CPU rendering.
///
/// @param processor Image processor handle
/// @param dst Destination image to render onto
/// @param detections Detection box list (can be NULL for segmentation-only)
/// @param segmentations Segmentation list (can be NULL for detection-only)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor or dst)
/// - EIO: Rendering failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_render_to_image(
    processor: *mut HalImageProcessor,
    dst: *mut HalTensorImage,
    detections: *const HalDetectBoxList,
    segmentations: *const HalSegmentationList,
) -> c_int {
    check_null!(processor, dst);

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

    try_or_errno!(
        unsafe { &mut (*processor) }.inner.render_to_image(
            &mut unsafe { &mut (*dst) }.inner,
            detect_slice,
            seg_slice,
        ),
        libc::EIO
    );
    0
}

/// Set class colors for segmentation rendering.
///
/// Colors are used when rendering segmentation masks via
/// hal_image_processor_render_to_image(). Each color is an RGBA tuple.
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
    use std::ffi::CString;

    #[test]
    fn test_image_create_and_free() {
        unsafe {
            let image = hal_tensor_image_new(640, 480, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!image.is_null());

            assert_eq!(hal_tensor_image_width(image), 640);
            assert_eq!(hal_tensor_image_height(image), 480);
            assert_eq!(hal_tensor_image_fourcc(image), HalFourcc::Rgb);

            hal_tensor_image_free(image);
        }
    }

    #[test]
    fn test_image_all_fourcc_formats() {
        unsafe {
            let formats = [
                HalFourcc::Rgb,
                HalFourcc::Rgba,
                HalFourcc::Grey,
                HalFourcc::Nv12,
                HalFourcc::Nv16,
                HalFourcc::Yuyv,
                HalFourcc::PlanarRgb,
                HalFourcc::PlanarRgba,
            ];

            for fourcc in formats {
                // Use dimensions that work for all formats (divisible by 2 for YUV)
                let image = hal_tensor_image_new(320, 240, fourcc, HalTensorMemory::Mem);
                assert!(
                    !image.is_null(),
                    "Failed to create image with fourcc {:?}",
                    fourcc
                );

                assert_eq!(hal_tensor_image_width(image), 320);
                assert_eq!(hal_tensor_image_height(image), 240);
                assert_eq!(hal_tensor_image_fourcc(image), fourcc);

                hal_tensor_image_free(image);
            }
        }
    }

    #[test]
    fn test_image_tensor() {
        unsafe {
            let image = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!image.is_null());

            let tensor = hal_tensor_image_tensor(image);
            assert!(!tensor.is_null());

            // NULL image returns NULL
            assert!(hal_tensor_image_tensor(std::ptr::null()).is_null());

            hal_tensor_image_free(image);
        }
    }

    #[test]
    fn test_image_fourcc_null() {
        unsafe {
            // NULL image returns Rgb default
            assert_eq!(hal_tensor_image_fourcc(std::ptr::null()), HalFourcc::Rgb);
        }
    }

    #[test]
    fn test_rect_and_crop() {
        let rect = hal_rect_new(10, 20, 100, 200);
        assert_eq!(rect.left, 10);
        assert_eq!(rect.top, 20);
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 200);

        unsafe {
            let mut crop = hal_crop_new();
            assert!(!crop.has_src_rect);
            assert!(!crop.has_dst_rect);
            assert!(!crop.has_dst_color);

            hal_crop_set_src_rect(&mut crop, &rect);
            assert!(crop.has_src_rect);
            assert_eq!(crop.src_rect.left, 10);

            hal_crop_set_dst_color(&mut crop, 255, 128, 0, 255);
            assert!(crop.has_dst_color);
            assert_eq!(crop.dst_color, [255, 128, 0, 255]);
        }
    }

    #[test]
    fn test_crop_dst_rect() {
        unsafe {
            let rect = hal_rect_new(50, 60, 200, 300);
            let mut crop = hal_crop_new();

            hal_crop_set_dst_rect(&mut crop, &rect);
            assert!(crop.has_dst_rect);
            assert_eq!(crop.dst_rect.left, 50);
            assert_eq!(crop.dst_rect.top, 60);
            assert_eq!(crop.dst_rect.width, 200);
            assert_eq!(crop.dst_rect.height, 300);
        }
    }

    #[test]
    fn test_crop_set_null_rect() {
        unsafe {
            let mut crop = hal_crop_new();

            // Setting NULL rect should be no-op
            hal_crop_set_src_rect(&mut crop, std::ptr::null());
            assert!(!crop.has_src_rect);

            hal_crop_set_dst_rect(&mut crop, std::ptr::null());
            assert!(!crop.has_dst_rect);
        }
    }

    #[test]
    fn test_crop_conversion_to_rust() {
        let mut crop = hal_crop_new();
        let rect = hal_rect_new(10, 20, 100, 200);

        unsafe {
            hal_crop_set_src_rect(&mut crop, &rect);
            hal_crop_set_dst_rect(&mut crop, &rect);
            hal_crop_set_dst_color(&mut crop, 255, 0, 0, 255);
        }

        // Convert to Rust Crop and check
        let rust_crop: Crop = crop.into();
        assert!(rust_crop.src_rect.is_some());
        assert!(rust_crop.dst_rect.is_some());
        assert!(rust_crop.dst_color.is_some());
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

            let src = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            let dst = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
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

            hal_tensor_image_free(src);
            hal_tensor_image_free(dst);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_load_null_data() {
        unsafe {
            let image =
                hal_tensor_image_load(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_zero_length() {
        unsafe {
            let data = [0u8; 10];
            let image =
                hal_tensor_image_load(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_null_path() {
        unsafe {
            let image =
                hal_tensor_image_load_file(std::ptr::null(), HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_nonexistent() {
        unsafe {
            let path = CString::new("/nonexistent/path/image.jpg").unwrap();
            let image =
                hal_tensor_image_load_file(path.as_ptr(), HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_save_jpeg_null_params() {
        unsafe {
            let image = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!image.is_null());

            let path = CString::new("/tmp/test.jpg").unwrap();

            // NULL image
            assert_eq!(
                hal_tensor_image_save_jpeg(std::ptr::null(), path.as_ptr(), 80),
                -1
            );

            // NULL path
            assert_eq!(hal_tensor_image_save_jpeg(image, std::ptr::null(), 80), -1);

            hal_tensor_image_free(image);
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_image_clone_fd_mem_image() {
        unsafe {
            let image = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!image.is_null());

            // MEM images don't support fd, should return -1
            let fd = hal_tensor_image_clone_fd(image);
            assert_eq!(fd, -1);

            // NULL image should return -1
            assert_eq!(hal_tensor_image_clone_fd(std::ptr::null()), -1);

            hal_tensor_image_free(image);
        }
    }

    #[test]
    fn test_null_handling() {
        unsafe {
            assert_eq!(hal_tensor_image_width(std::ptr::null()), 0);
            assert_eq!(hal_tensor_image_height(std::ptr::null()), 0);
            hal_tensor_image_free(std::ptr::null_mut());
            hal_image_processor_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_image_is_planar() {
        unsafe {
            let rgb = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert!(!hal_tensor_image_is_planar(rgb));
            hal_tensor_image_free(rgb);

            let nv12 = hal_tensor_image_new(100, 100, HalFourcc::Nv12, HalTensorMemory::Mem);
            assert!(!nv12.is_null());
            assert!(hal_tensor_image_is_planar(nv12));
            hal_tensor_image_free(nv12);

            let planar = hal_tensor_image_new(100, 100, HalFourcc::PlanarRgb, HalTensorMemory::Mem);
            assert!(!planar.is_null());
            assert!(hal_tensor_image_is_planar(planar));
            hal_tensor_image_free(planar);

            // NULL image returns false
            assert!(!hal_tensor_image_is_planar(std::ptr::null()));
        }
    }

    #[test]
    fn test_image_channels() {
        unsafe {
            let rgb = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_image_channels(rgb), 3);
            hal_tensor_image_free(rgb);

            let rgba = hal_tensor_image_new(100, 100, HalFourcc::Rgba, HalTensorMemory::Mem);
            assert!(!rgba.is_null());
            assert_eq!(hal_tensor_image_channels(rgba), 4);
            hal_tensor_image_free(rgba);

            let grey = hal_tensor_image_new(100, 100, HalFourcc::Grey, HalTensorMemory::Mem);
            assert!(!grey.is_null());
            assert_eq!(hal_tensor_image_channels(grey), 1);
            hal_tensor_image_free(grey);

            // NULL image returns 0
            assert_eq!(hal_tensor_image_channels(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_image_row_stride() {
        unsafe {
            let rgb = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_image_row_stride(rgb), 300); // 100 * 3
            hal_tensor_image_free(rgb);

            let planar = hal_tensor_image_new(100, 100, HalFourcc::PlanarRgb, HalTensorMemory::Mem);
            assert!(!planar.is_null());
            assert_eq!(hal_tensor_image_row_stride(planar), 100); // planar: width
            hal_tensor_image_free(planar);

            // NULL image returns 0
            assert_eq!(hal_tensor_image_row_stride(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_image_from_tensor() {
        use crate::tensor::{hal_tensor_new, HalDtype};

        unsafe {
            // Create a u8 tensor with shape [100, 100, 3] for RGB
            let shape: [size_t; 3] = [100, 100, 3];
            let tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor.is_null());

            let image = hal_tensor_image_from_tensor(tensor, HalFourcc::Rgb);
            assert!(!image.is_null());
            assert_eq!(hal_tensor_image_width(image), 100);
            assert_eq!(hal_tensor_image_height(image), 100);
            assert_eq!(hal_tensor_image_fourcc(image), HalFourcc::Rgb);
            hal_tensor_image_free(image);
            // tensor is consumed, do NOT free it

            // f32 tensor should fail
            let shape_f32: [size_t; 3] = [100, 100, 3];
            let tensor_f32 = hal_tensor_new(
                HalDtype::F32,
                shape_f32.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!tensor_f32.is_null());
            let image_f32 = hal_tensor_image_from_tensor(tensor_f32, HalFourcc::Rgb);
            assert!(image_f32.is_null());
            // tensor_f32 was consumed by the function (even on failure, Box::from_raw took it)

            // NULL tensor should fail
            let image_null = hal_tensor_image_from_tensor(std::ptr::null_mut(), HalFourcc::Rgb);
            assert!(image_null.is_null());
        }
    }

    #[test]
    fn test_image_load_jpeg_null() {
        unsafe {
            let image =
                hal_tensor_image_load_jpeg(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());

            let data = [0u8; 10];
            let image =
                hal_tensor_image_load_jpeg(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_png_null() {
        unsafe {
            let image =
                hal_tensor_image_load_png(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());

            let data = [0u8; 10];
            let image =
                hal_tensor_image_load_png(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_processor_convert_ref_null_params() {
        use crate::tensor::{hal_tensor_free, hal_tensor_new, HalDtype};

        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let src = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!src.is_null());

            let shape: [size_t; 3] = [3, 100, 100];
            let dst_tensor = hal_tensor_new(
                HalDtype::U8,
                shape.as_ptr(),
                3,
                HalTensorMemory::Mem,
                std::ptr::null(),
            );
            assert!(!dst_tensor.is_null());

            // NULL processor
            assert_eq!(
                hal_image_processor_convert_ref(
                    std::ptr::null_mut(),
                    src,
                    dst_tensor,
                    HalFourcc::PlanarRgb,
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            // NULL src
            assert_eq!(
                hal_image_processor_convert_ref(
                    processor,
                    std::ptr::null(),
                    dst_tensor,
                    HalFourcc::PlanarRgb,
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            // NULL dst_tensor
            assert_eq!(
                hal_image_processor_convert_ref(
                    processor,
                    src,
                    std::ptr::null_mut(),
                    HalFourcc::PlanarRgb,
                    HalRotation::None,
                    HalFlip::None,
                    std::ptr::null()
                ),
                -1
            );

            hal_tensor_free(dst_tensor);
            hal_tensor_image_free(src);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_render_to_image_null_params() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let dst = hal_tensor_image_new(100, 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(!dst.is_null());

            // NULL processor
            assert_eq!(
                hal_image_processor_render_to_image(
                    std::ptr::null_mut(),
                    dst,
                    std::ptr::null(),
                    std::ptr::null()
                ),
                -1
            );

            // NULL dst
            assert_eq!(
                hal_image_processor_render_to_image(
                    processor,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null()
                ),
                -1
            );

            hal_tensor_image_free(dst);
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
    fn test_fourcc_to_from_fourcc() {
        // Test all fourcc conversions roundtrip
        let formats = [
            HalFourcc::Yuyv,
            HalFourcc::Nv12,
            HalFourcc::Nv16,
            HalFourcc::Rgba,
            HalFourcc::Rgb,
            HalFourcc::Grey,
            HalFourcc::PlanarRgb,
            HalFourcc::PlanarRgba,
        ];

        for format in formats {
            let fourcc = format.to_fourcc();
            let back = HalFourcc::from_fourcc(fourcc);
            assert_eq!(back, Some(format), "Roundtrip failed for {:?}", format);
        }
    }
}
