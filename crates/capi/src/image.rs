// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image C API - Hardware-accelerated image processing.
//!
//! This module provides image conversion and manipulation functions with
//! support for hardware acceleration (G2D, OpenGL) when available.

use crate::decoder::{HalDetectBoxList, HalSegmentationList};
use crate::error::{set_error, set_error_null};
use crate::tensor::{HalDtype, HalTensor, HalTensorMemory};
use crate::{check_null, check_null_ret_null, try_or_errno, try_or_null};
use edgefirst_image::{
    load_image, save_jpeg, ComputeBackend, Crop, Flip, ImageProcessor, ImageProcessorConfig,
    ImageProcessorTrait, Rect, Rotation,
};
use edgefirst_tensor::{PixelFormat, PixelLayout, TensorDyn, TensorMemory, TensorTrait};
use libc::{c_char, c_int, size_t};
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
    /// 8-bit BGRA (4 channels, blue first)
    Bgra = 8,
    /// 8-bit interleaved YUV422, limited range (VYUY byte order)
    Vyuy = 9,
}

impl HalFourcc {
    fn to_pixel_format(self) -> PixelFormat {
        match self {
            HalFourcc::Yuyv => PixelFormat::Yuyv,
            HalFourcc::Nv12 => PixelFormat::Nv12,
            HalFourcc::Nv16 => PixelFormat::Nv16,
            HalFourcc::Rgba => PixelFormat::Rgba,
            HalFourcc::Rgb => PixelFormat::Rgb,
            HalFourcc::Grey => PixelFormat::Grey,
            HalFourcc::PlanarRgb => PixelFormat::PlanarRgb,
            HalFourcc::PlanarRgba => PixelFormat::PlanarRgba,
            HalFourcc::Bgra => PixelFormat::Bgra,
            HalFourcc::Vyuy => PixelFormat::Vyuy,
        }
    }

    fn from_pixel_format(fmt: PixelFormat) -> Self {
        match fmt {
            PixelFormat::Rgb => HalFourcc::Rgb,
            PixelFormat::Rgba => HalFourcc::Rgba,
            PixelFormat::Grey => HalFourcc::Grey,
            PixelFormat::Yuyv => HalFourcc::Yuyv,
            PixelFormat::Nv12 => HalFourcc::Nv12,
            PixelFormat::Nv16 => HalFourcc::Nv16,
            PixelFormat::PlanarRgb => HalFourcc::PlanarRgb,
            PixelFormat::PlanarRgba => HalFourcc::PlanarRgba,
            PixelFormat::Bgra => HalFourcc::Bgra,
            PixelFormat::Vyuy => HalFourcc::Vyuy,
            _ => HalFourcc::Rgb,
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

/// Opaque image processor type.
///
/// The ImageProcessor handles format conversion with hardware acceleration when available.
pub struct HalImageProcessor {
    /// Accessible to `decoder.rs` for `hal_decoder_draw_masks()`.
    pub(crate) inner: ImageProcessor,
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
/// @param fourcc Pixel format (HAL_FOURCC_*)
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
    fourcc: HalFourcc,
    dtype: HalDtype,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    if width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    let mem_opt: Option<TensorMemory> = memory.into();
    let dyn_tensor = try_or_null!(
        TensorDyn::image(
            width,
            height,
            fourcc.to_pixel_format(),
            dtype.into(),
            mem_opt,
        ),
        libc::ENOMEM
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load an image from memory (JPEG or PNG).
///
/// Automatically detects format and decodes the image.
///
/// @param data Pointer to image data
/// @param len Length of image data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode image
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_load_image(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(fourcc.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load an image from a file (JPEG or PNG).
///
/// @param path Path to the image file
/// @param fourcc Output pixel format
/// @param memory Memory allocation type
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL path)
/// - ENOENT: File not found
/// - EBADMSG: Failed to decode image
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_load_image_file(
    path: *const c_char,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensor {
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
    let dyn_tensor = try_or_null!(
        load_image(&data, Some(fourcc.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Create a multiplane image tensor from separate Y and UV DMA-BUF file descriptors.
///
/// This is used for V4L2 multi-planar NV12 (`V4L2_PIX_FMT_NV12M`) where the
/// Y and UV planes are in separate DMA-BUF allocations.
///
/// **Ownership**: Ownership is transferred once the function validates that
/// both FDs are non-negative, non-zero dimensions, and distinct. After that
/// point, the HAL closes them on any error. If validation of the FD values
/// themselves fails (negative FD, zero dimensions, or `y_fd == uv_fd`), the
/// caller retains ownership.
///
/// @param y_fd    DMA-BUF file descriptor for the Y (luma) plane
/// @param width   Image width in pixels
/// @param height  Image height in pixels
/// @param uv_fd   DMA-BUF file descriptor for the UV (chroma) plane
/// @param fourcc  Pixel format (must be HAL_FOURCC_NV12 or HAL_FOURCC_NV16)
/// @param out     Receives the new tensor handle on success
/// @return 0 on success, -1 on error (errno set)
/// @par Errors (errno):
/// - EINVAL: Invalid argument or unsupported fourcc
/// - EIO:    Failed to wrap DMA-BUF file descriptor
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_from_planes(
    y_fd: c_int,
    width: u32,
    height: u32,
    uv_fd: c_int,
    fourcc: HalFourcc,
    out: *mut *mut HalTensor,
) -> c_int {
    check_null!(out);
    if y_fd < 0 || uv_fd < 0 || width == 0 || height == 0 {
        set_error(libc::EINVAL);
        return -1;
    }
    if y_fd == uv_fd {
        set_error(libc::EINVAL);
        return -1;
    }

    // Take ownership of the file descriptors early so they auto-close on any
    // subsequent error return, making the documented ownership contract truthful.
    use std::os::unix::io::FromRawFd;
    let y_owned = unsafe { std::os::unix::io::OwnedFd::from_raw_fd(y_fd) };
    let uv_owned = unsafe { std::os::unix::io::OwnedFd::from_raw_fd(uv_fd) };

    let fmt = fourcc.to_pixel_format();
    let w = width as usize;
    let h = height as usize;

    // Determine chroma height based on format
    let chroma_h = match fmt {
        PixelFormat::Nv12 => h / 2,
        PixelFormat::Nv16 => h,
        _ => {
            set_error(libc::EINVAL);
            return -1; // y_owned and uv_owned drop here, closing FDs
        }
    };

    let luma = match edgefirst_tensor::Tensor::<u8>::from_fd(y_owned, &[h, w], Some("luma")) {
        Ok(t) => t,
        Err(_) => {
            set_error(libc::EIO);
            return -1;
        }
    };

    let chroma =
        match edgefirst_tensor::Tensor::<u8>::from_fd(uv_owned, &[chroma_h, w], Some("chroma")) {
            Ok(t) => t,
            Err(_) => {
                set_error(libc::EIO);
                return -1;
            }
        };

    let dyn_tensor = match edgefirst_tensor::Tensor::<u8>::from_planes(luma, chroma, fmt) {
        Ok(t) => TensorDyn::U8(t),
        Err(_) => {
            set_error(libc::EINVAL);
            return -1;
        }
    };

    unsafe { *out = Box::into_raw(Box::new(HalTensor { inner: dyn_tensor })) };
    0
}

/// Load a JPEG image from memory.
///
/// @param data Pointer to JPEG data
/// @param len Length of JPEG data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode JPEG
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_load_jpeg(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(fourcc.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load a PNG image from memory.
///
/// @param data Pointer to PNG data
/// @param len Length of PNG data in bytes
/// @param fourcc Output pixel format (HAL_FOURCC_RGB, HAL_FOURCC_RGBA, or HAL_FOURCC_GREY)
/// @param memory Memory allocation type
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL data, zero length)
/// - EBADMSG: Failed to decode PNG
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_load_png(
    data: *const u8,
    len: size_t,
    fourcc: HalFourcc,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(fourcc.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
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
/// @return Pixel format, or HAL_FOURCC_RGB if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_fourcc(tensor: *const HalTensor) -> HalFourcc {
    if tensor.is_null() {
        return HalFourcc::Rgb;
    }
    match unsafe { &(*tensor) }.inner.format() {
        Some(fmt) => HalFourcc::from_pixel_format(fmt),
        None => HalFourcc::Rgb,
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

/// Get the row stride of an image tensor in bytes.
///
/// For planar formats this is equal to the width. For interleaved formats
/// this is width * channels.
///
/// @param tensor Image tensor handle
/// @return Row stride in bytes, or 0 if tensor is NULL or not an image
#[no_mangle]
pub unsafe extern "C" fn hal_tensor_row_stride(tensor: *const HalTensor) -> size_t {
    if tensor.is_null() {
        return 0;
    }
    let dyn_ref = &unsafe { &(*tensor) }.inner;
    match (dyn_ref.format(), dyn_ref.width()) {
        (Some(fmt), Some(w)) => match fmt.layout() {
            PixelLayout::Packed => w * fmt.channels(),
            _ => w,
        },
        _ => 0,
    }
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

/// Convert an image into a borrowed tensor reference.
///
/// This enables zero-copy preprocessing directly into a model's pre-allocated
/// input buffer. The destination tensor must be u8 dtype with a 3D shape
/// matching the specified output format.
///
/// @param processor Image processor handle
/// @param src Source image tensor
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
    src: *const HalTensor,
    dst_tensor: *mut HalTensor,
    dst_fourcc: HalFourcc,
    rotation: HalRotation,
    flip: HalFlip,
    crop: *const HalCrop,
) -> c_int {
    check_null!(processor, src, dst_tensor);

    // Validate that the tensor is u8 before modifying format
    if !matches!(unsafe { &*dst_tensor }.inner, TensorDyn::U8(_)) {
        return set_error(libc::EINVAL);
    }

    // Move the inner TensorDyn out via ptr::read so we can modify the
    // format on the inner Tensor<u8> and pass it to convert. We wrap the
    // source in ManuallyDrop to prevent a double-free if convert() panics.
    let dst_dyn = unsafe { std::ptr::read(&(*dst_tensor).inner) };
    let mut dst_dyn = std::mem::ManuallyDrop::new(dst_dyn);

    if let TensorDyn::U8(ref mut t) = *dst_dyn {
        if t.set_format(dst_fourcc.to_pixel_format()).is_err() {
            unsafe {
                std::ptr::write(
                    &mut (*dst_tensor).inner,
                    std::mem::ManuallyDrop::into_inner(dst_dyn),
                )
            };
            return set_error(libc::EINVAL);
        }
    }

    let crop_config = if crop.is_null() {
        Crop::default()
    } else {
        unsafe { *crop }.into()
    };

    let result = unsafe { &mut (*processor) }.inner.convert(
        &unsafe { &(*src) }.inner,
        &mut dst_dyn,
        rotation.into(),
        flip.into(),
        crop_config,
    );

    // Put the tensor back regardless of success/failure
    unsafe {
        std::ptr::write(
            &mut (*dst_tensor).inner,
            std::mem::ManuallyDrop::into_inner(dst_dyn),
        )
    };

    try_or_errno!(result, libc::EIO);
    0
}

/// Draw detection boxes and segmentation masks onto an image.
///
/// Draws bounding boxes (with labels) and segmentation overlays on the
/// destination image tensor. Uses hardware acceleration (OpenGL) when available,
/// falling back to CPU rendering.
///
/// @param processor Image processor handle
/// @param dst Destination image tensor to draw onto
/// @param detections Detection box list (can be NULL for segmentation-only)
/// @param segmentations Segmentation list (can be NULL for detection-only)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL processor or dst)
/// - EIO: Drawing failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_draw_masks(
    processor: *mut HalImageProcessor,
    dst: *mut HalTensor,
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
        unsafe { &mut (*processor) }.inner.draw_masks(
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
/// Colors are used when drawing segmentation masks via
/// hal_image_processor_draw_masks(). Each color is an RGBA tuple.
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
/// @param fourcc Pixel format (HAL_FOURCC_*)
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
    fourcc: HalFourcc,
    dtype: HalDtype,
) -> *mut HalTensor {
    if processor.is_null() || width == 0 || height == 0 {
        return set_error_null(libc::EINVAL);
    }

    let dyn_tensor = match unsafe { &(*processor) }.inner.create_image(
        width,
        height,
        fourcc.to_pixel_format(),
        dtype.into(),
        None,
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
            let image =
                hal_tensor_new_image(640, 480, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            assert!(!image.is_null());

            assert_eq!(hal_tensor_width(image), 640);
            assert_eq!(hal_tensor_height(image), 480);
            assert_eq!(hal_tensor_fourcc(image), HalFourcc::Rgb);

            hal_tensor_free(image);
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
                let image =
                    hal_tensor_new_image(320, 240, fourcc, HalDtype::U8, HalTensorMemory::Mem);
                assert!(
                    !image.is_null(),
                    "Failed to create image with fourcc {:?}",
                    fourcc
                );

                assert_eq!(hal_tensor_width(image), 320);
                assert_eq!(hal_tensor_height(image), 240);
                assert_eq!(hal_tensor_fourcc(image), fourcc);

                hal_tensor_free(image);
            }
        }
    }

    #[test]
    fn test_image_fourcc_null() {
        unsafe {
            // NULL image returns Rgb default
            assert_eq!(hal_tensor_fourcc(std::ptr::null()), HalFourcc::Rgb);
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

            let src =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            let dst =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
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
    fn test_image_load_null_data() {
        unsafe {
            let image =
                hal_tensor_load_image(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_zero_length() {
        unsafe {
            let data = [0u8; 10];
            let image =
                hal_tensor_load_image(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_null_path() {
        unsafe {
            let image =
                hal_tensor_load_image_file(std::ptr::null(), HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_nonexistent() {
        unsafe {
            let path = CString::new("/nonexistent/path/image.jpg").unwrap();
            let image =
                hal_tensor_load_image_file(path.as_ptr(), HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_save_jpeg_null_params() {
        unsafe {
            let image =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
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
            let rgb =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert!(!hal_tensor_is_planar(rgb));
            hal_tensor_free(rgb);

            let nv12 = hal_tensor_new_image(
                100,
                100,
                HalFourcc::Nv12,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            assert!(!nv12.is_null());
            assert!(hal_tensor_is_planar(nv12));
            hal_tensor_free(nv12);

            let planar = hal_tensor_new_image(
                100,
                100,
                HalFourcc::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
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
            let rgb =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_channels(rgb), 3);
            hal_tensor_free(rgb);

            let rgba = hal_tensor_new_image(
                100,
                100,
                HalFourcc::Rgba,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            assert!(!rgba.is_null());
            assert_eq!(hal_tensor_channels(rgba), 4);
            hal_tensor_free(rgba);

            let grey = hal_tensor_new_image(
                100,
                100,
                HalFourcc::Grey,
                HalDtype::U8,
                HalTensorMemory::Mem,
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
            let rgb =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            assert!(!rgb.is_null());
            assert_eq!(hal_tensor_row_stride(rgb), 300); // 100 * 3
            hal_tensor_free(rgb);

            let planar = hal_tensor_new_image(
                100,
                100,
                HalFourcc::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            assert!(!planar.is_null());
            assert_eq!(hal_tensor_row_stride(planar), 100); // planar: width
            hal_tensor_free(planar);

            // NULL image returns 0
            assert_eq!(hal_tensor_row_stride(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_image_load_jpeg_null() {
        unsafe {
            let image =
                hal_tensor_load_jpeg(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());

            let data = [0u8; 10];
            let image =
                hal_tensor_load_jpeg(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_png_null() {
        unsafe {
            let image =
                hal_tensor_load_png(std::ptr::null(), 100, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());

            let data = [0u8; 10];
            let image = hal_tensor_load_png(data.as_ptr(), 0, HalFourcc::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_processor_convert_ref_null_params() {
        use crate::tensor::{hal_tensor_new, HalDtype};

        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let src =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
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
            hal_tensor_free(src);
            hal_image_processor_free(processor);
        }
    }

    #[test]
    fn test_image_processor_draw_masks_null_params() {
        unsafe {
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }

            let dst =
                hal_tensor_new_image(100, 100, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
            assert!(!dst.is_null());

            // NULL processor
            assert_eq!(
                hal_image_processor_draw_masks(
                    std::ptr::null_mut(),
                    dst,
                    std::ptr::null(),
                    std::ptr::null()
                ),
                -1
            );

            // NULL dst
            assert_eq!(
                hal_image_processor_draw_masks(
                    processor,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null()
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
            let image =
                hal_tensor_new_image(100, 50, HalFourcc::Rgb, HalDtype::U8, HalTensorMemory::Mem);
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
                HalFourcc::PlanarRgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
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
            HalFourcc::Yuyv,
            HalFourcc::Nv12,
            HalFourcc::Nv16,
            HalFourcc::Rgba,
            HalFourcc::Rgb,
            HalFourcc::Grey,
            HalFourcc::PlanarRgb,
            HalFourcc::PlanarRgba,
            HalFourcc::Bgra,
            HalFourcc::Vyuy,
        ];

        for format in formats {
            let pf = format.to_pixel_format();
            let back = HalFourcc::from_pixel_format(pf);
            assert_eq!(back, format, "Roundtrip failed for {:?}", format);
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
                HalFourcc::Rgba,
                HalDtype::U8,
            );
            assert!(img.is_null());

            // Zero width
            let processor = hal_image_processor_new();
            if processor.is_null() {
                return;
            }
            let img =
                hal_image_processor_create_image(processor, 0, 480, HalFourcc::Rgba, HalDtype::U8);
            assert!(img.is_null());

            // Zero height
            let img =
                hal_image_processor_create_image(processor, 640, 0, HalFourcc::Rgba, HalDtype::U8);
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
                (HalFourcc::Rgb, 3),
                (HalFourcc::Rgba, 4),
                (HalFourcc::Grey, 1),
            ];

            for (fourcc, channels) in formats {
                let img =
                    hal_image_processor_create_image(processor, 320, 240, fourcc, HalDtype::U8);
                assert!(!img.is_null(), "create_image failed for {:?}", fourcc);

                assert_eq!(hal_tensor_width(img), 320);
                assert_eq!(hal_tensor_height(img), 240);
                assert_eq!(hal_tensor_fourcc(img), fourcc);

                // Verify the image is mappable
                let map = hal_tensor_map_create(img);
                assert!(!map.is_null(), "map failed for {:?}", fourcc);
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
            let img =
                hal_image_processor_create_image(processor, 320, 240, HalFourcc::Rgb, HalDtype::I8);
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
                HalFourcc::Rgba,
                HalDtype::U8,
            );
            assert!(!src.is_null());

            // Create destination image via create_image
            let dst = hal_image_processor_create_image(
                processor,
                160,
                120,
                HalFourcc::Rgba,
                HalDtype::U8,
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
}
