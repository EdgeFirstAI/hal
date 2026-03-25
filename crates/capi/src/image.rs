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
use edgefirst_tensor::{PixelFormat, PixelLayout, TensorDyn, TensorMemory};
use libc::{c_char, c_int, size_t};
use std::ffi::CStr;

/// Image pixel format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalPixelFormat {
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
    ///
    /// @note On Vivante GPUs (i.MX 8M Plus), VYUY import uses a 2D texture
    /// fallback instead of the EGL external texture path.  The HAL detects
    /// this automatically; no caller action is needed.  Quality is validated
    /// but may differ slightly from the external-texture path used on other
    /// GPUs.
    Vyuy = 9,
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
/// @param format Output pixel format (HAL_PIXEL_FORMAT_RGB, HAL_PIXEL_FORMAT_RGBA, or HAL_PIXEL_FORMAT_GREY)
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
    format: HalPixelFormat,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(format.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load an image from a file (JPEG or PNG).
///
/// @param path Path to the image file
/// @param format Output pixel format
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
    format: HalPixelFormat,
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
        load_image(&data, Some(format.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load a JPEG image from memory.
///
/// @param data Pointer to JPEG data
/// @param len Length of JPEG data in bytes
/// @param format Output pixel format (HAL_PIXEL_FORMAT_RGB, HAL_PIXEL_FORMAT_RGBA, or HAL_PIXEL_FORMAT_GREY)
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
    format: HalPixelFormat,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(format.to_pixel_format()), mem_opt),
        libc::EBADMSG
    );

    Box::into_raw(Box::new(HalTensor { inner: dyn_tensor }))
}

/// Load a PNG image from memory.
///
/// @param data Pointer to PNG data
/// @param len Length of PNG data in bytes
/// @param format Output pixel format (HAL_PIXEL_FORMAT_RGB, HAL_PIXEL_FORMAT_RGBA, or HAL_PIXEL_FORMAT_GREY)
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
    format: HalPixelFormat,
    memory: HalTensorMemory,
) -> *mut HalTensor {
    check_null_ret_null!(data);
    if len == 0 {
        return set_error_null(libc::EINVAL);
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mem_opt: Option<TensorMemory> = memory.into();

    let dyn_tensor = try_or_null!(
        load_image(data_slice, Some(format.to_pixel_format()), mem_opt),
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
            edgefirst_image::MaskOverlay::default(),
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

    let proc_inner = &unsafe { &(*processor) }.inner;
    let dyn_tensor = match proc_inner.import_image(
        image_pd,
        chroma_pd,
        width,
        height,
        format.to_pixel_format(),
        dtype.into(),
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
#[cfg(not(target_os = "linux"))]
pub unsafe extern "C" fn hal_import_image(
    _processor: *mut HalImageProcessor,
    _image: *mut HalPlaneDescriptor,
    _chroma: *mut HalPlaneDescriptor,
    _width: size_t,
    _height: size_t,
    _format: HalPixelFormat,
    _dtype: HalDtype,
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
                HalPixelFormat::Grey,
                HalPixelFormat::Nv12,
                HalPixelFormat::Nv16,
                HalPixelFormat::Yuyv,
                HalPixelFormat::PlanarRgb,
                HalPixelFormat::PlanarRgba,
            ];

            for fmt in formats {
                // Use dimensions that work for all formats (divisible by 2 for YUV)
                let image = hal_tensor_new_image(320, 240, fmt, HalDtype::U8, HalTensorMemory::Mem);
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

            let src = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
            );
            let dst = hal_tensor_new_image(
                100,
                100,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
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
    fn test_image_load_null_data() {
        unsafe {
            let image = hal_tensor_load_image(
                std::ptr::null(),
                100,
                HalPixelFormat::Rgb,
                HalTensorMemory::Mem,
            );
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_zero_length() {
        unsafe {
            let data = [0u8; 10];
            let image =
                hal_tensor_load_image(data.as_ptr(), 0, HalPixelFormat::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_null_path() {
        unsafe {
            let image = hal_tensor_load_image_file(
                std::ptr::null(),
                HalPixelFormat::Rgb,
                HalTensorMemory::Mem,
            );
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_file_nonexistent() {
        unsafe {
            let path = CString::new("/nonexistent/path/image.jpg").unwrap();
            let image = hal_tensor_load_image_file(
                path.as_ptr(),
                HalPixelFormat::Rgb,
                HalTensorMemory::Mem,
            );
            assert!(image.is_null());
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
            let image = hal_tensor_load_jpeg(
                std::ptr::null(),
                100,
                HalPixelFormat::Rgb,
                HalTensorMemory::Mem,
            );
            assert!(image.is_null());

            let data = [0u8; 10];
            let image =
                hal_tensor_load_jpeg(data.as_ptr(), 0, HalPixelFormat::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_load_png_null() {
        unsafe {
            let image = hal_tensor_load_png(
                std::ptr::null(),
                100,
                HalPixelFormat::Rgb,
                HalTensorMemory::Mem,
            );
            assert!(image.is_null());

            let data = [0u8; 10];
            let image =
                hal_tensor_load_png(data.as_ptr(), 0, HalPixelFormat::Rgb, HalTensorMemory::Mem);
            assert!(image.is_null());
        }
    }

    #[test]
    fn test_image_processor_draw_masks_null_params() {
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
            );
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
            let image = hal_tensor_new_image(
                100,
                50,
                HalPixelFormat::Rgb,
                HalDtype::U8,
                HalTensorMemory::Mem,
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
        ];

        for format in formats {
            let pf = format.to_pixel_format();
            let back = HalPixelFormat::from_pixel_format(pf);
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
                HalPixelFormat::Rgba,
                HalDtype::U8,
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
            );
            assert!(img.is_null());

            // Zero height
            let img = hal_image_processor_create_image(
                processor,
                640,
                0,
                HalPixelFormat::Rgba,
                HalDtype::U8,
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
                let img = hal_image_processor_create_image(processor, 320, 240, fmt, HalDtype::U8);
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
            );
            assert!(!src.is_null());

            // Create destination image via create_image
            let dst = hal_image_processor_create_image(
                processor,
                160,
                120,
                HalPixelFormat::Rgba,
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
    #[cfg(target_os = "linux")]
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
            );
            assert!(result.is_null());
            // pd is consumed — do NOT free
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
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
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
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
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
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
            );
            assert!(result.is_null());

            hal_image_processor_free(processor);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
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
