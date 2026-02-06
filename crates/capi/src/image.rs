// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image C API - Hardware-accelerated image processing.
//!
//! This module provides image conversion and manipulation functions with
//! support for hardware acceleration (G2D, OpenGL) when available.

use crate::error::{set_error, set_error_null};
use crate::tensor::HalTensorMemory;
use crate::{check_null, check_null_ret_null, try_or_errno, try_or_null};
use edgefirst_image::{
    Crop, Flip, ImageProcessor, ImageProcessorTrait, Rect, Rotation, TensorImage, GREY, NV12, NV16,
    PLANAR_RGB, PLANAR_RGBA, RGB, RGBA, YUYV,
};
use edgefirst_tensor::TensorMemory;
#[cfg(unix)]
use edgefirst_tensor::TensorTrait;
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
    fn test_processor_create_and_free() {
        unsafe {
            let processor = hal_image_processor_new();
            // Processor may be NULL if no backend is available
            hal_image_processor_free(processor);
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
}
