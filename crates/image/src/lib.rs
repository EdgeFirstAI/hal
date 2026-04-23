// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!

## EdgeFirst HAL - Image Converter

The `edgefirst_image` crate is part of the EdgeFirst Hardware Abstraction
Layer (HAL) and provides functionality for converting images between
different formats and sizes.  The crate is designed to work with hardware
acceleration when available, but also provides a CPU-based fallback for
environments where hardware acceleration is not present or not suitable.

The main features of the `edgefirst_image` crate include:
- Support for various image formats, including YUYV, RGB, RGBA, and GREY.
- Support for source crop, destination crop, rotation, and flipping.
- Image conversion using hardware acceleration (G2D, OpenGL) when available.
- CPU-based image conversion as a fallback option.

The crate uses [`TensorDyn`] from `edgefirst_tensor` to represent images,
with [`PixelFormat`] metadata describing the pixel layout. The
[`ImageProcessor`] struct manages the conversion process, selecting
the appropriate conversion method based on the available hardware.

## Examples

```rust
# use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait, load_image};
# use edgefirst_tensor::{PixelFormat, DType, TensorDyn};
# fn main() -> Result<(), edgefirst_image::Error> {
let image = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
let src = load_image(image, Some(PixelFormat::Rgba), None)?;
let mut converter = ImageProcessor::new()?;
let mut dst = converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None)?;
converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
# Ok(())
# }
```

## Environment Variables
The behavior of the `edgefirst_image::ImageProcessor` struct can be influenced by the
following environment variables:
- `EDGEFIRST_FORCE_BACKEND`: When set to `cpu`, `g2d`, or `opengl` (case-insensitive),
  only that single backend is initialized and no fallback chain is used. If the
  forced backend fails to initialize, an error is returned immediately. This is
  useful for benchmarking individual backends in isolation. When this variable is
  set, the `EDGEFIRST_DISABLE_*` variables are ignored.
- `EDGEFIRST_DISABLE_GL`: If set to `1`, disables the use of OpenGL for image
  conversion, forcing the use of CPU or other available hardware methods.
- `EDGEFIRST_DISABLE_G2D`: If set to `1`, disables the use of G2D for image
  conversion, forcing the use of CPU or other available hardware methods.
- `EDGEFIRST_DISABLE_CPU`: If set to `1`, disables the use of CPU for image
  conversion, forcing the use of hardware acceleration methods. If no hardware
  acceleration methods are available, an error will be returned when attempting
  to create an `ImageProcessor`.

Additionally the TensorMemory used by default allocations can be controlled using the
`EDGEFIRST_TENSOR_FORCE_MEM` environment variable. If set to `1`, default tensor memory
uses system memory. This will disable the use of specialized memory regions for tensors
and hardware acceleration. However, this will increase the performance of the CPU converter.
*/
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

/// Pitch alignment requirement for DMA-BUF tensors that may be imported as
/// EGLImages by the GL backend. Mali Valhall (i.MX 95 / G310) rejects
/// `eglCreateImageKHR` with `EGL_BAD_ALLOC` for any DMA-BUF whose row pitch
/// is not a multiple of 64 bytes; Vivante GC7000UL (i.MX 8MP) accepts any
/// pitch so the constant is harmless on that path. 64 is the smallest
/// alignment that satisfies every embedded ARM GPU we ship to.
///
/// Applied automatically inside [`ImageProcessor::create_image`] when the
/// allocation lands on `TensorMemory::Dma`. External callers that allocate
/// their own DMA-BUF tensors (e.g. GStreamer plugins, video pipelines) can
/// use [`align_width_for_gpu_pitch`] to compute a width whose resulting row
/// stride satisfies this requirement.
pub const GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES: usize = 64;

/// Round `width` (in pixels) up so the resulting row stride
/// `width * bpp` is a multiple of [`GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`]
/// AND a multiple of `bpp` (so the rounded width is an integer pixel count).
///
/// `bpp` must be the per-pixel byte count for the image's primary plane
/// (e.g. 4 for RGBA8/BGRA8, 3 for RGB888, 1 for Grey/NV12-luma).
///
/// External callers — GStreamer plugins, video pipelines, anyone wrapping a
/// foreign DMA-BUF — should call this when sizing the destination so that
/// `eglCreateImageKHR` doesn't reject the import on Mali. Pre-aligned widths
/// (640, 1280, 1920, 3008, 3840 …) round-trip unchanged; misaligned widths
/// are bumped up to the next valid value.
///
/// # Overflow behaviour
///
/// All arithmetic is checked. If the alignment computation or the rounded
/// width would overflow `usize`, the function logs a warning and returns the
/// original `width` unchanged rather than wrapping or producing a smaller
/// value. Callers can rely on the returned width being **at least** the
/// requested width.
///
/// `bpp == 0` and `width == 0` short-circuit to return the input unchanged.
///
/// # Examples
///
/// ```
/// use edgefirst_image::align_width_for_gpu_pitch;
///
/// // RGBA8 (bpp=4): width must round to a multiple of 16 pixels (64-byte stride).
/// assert_eq!(align_width_for_gpu_pitch(1920, 4), 1920); // already aligned
/// assert_eq!(align_width_for_gpu_pitch(3004, 4), 3008); // crowd.png case: +4 px
/// assert_eq!(align_width_for_gpu_pitch(1281, 4), 1296); // +15 px
///
/// // RGB888 (bpp=3): width must round to a multiple of 64 pixels (192-byte stride).
/// assert_eq!(align_width_for_gpu_pitch(640, 3), 640);
/// assert_eq!(align_width_for_gpu_pitch(641, 3), 704);
/// ```
pub fn align_width_for_gpu_pitch(width: usize, bpp: usize) -> usize {
    if bpp == 0 || width == 0 {
        return width;
    }

    // The minimum aligned stride must be a common multiple of both the
    // GPU's pitch alignment and the per-pixel byte count. Using the LCM
    // guarantees the rounded stride is an integer multiple of `bpp`, so
    // converting back to a pixel count is exact.
    //
    // Compute the alignment in pixels (`width_alignment`) so we never need
    // to multiply `width * bpp`, which is the only operation that could
    // realistically overflow for large caller-supplied widths.
    let Some(lcm_alignment) = checked_num_integer_lcm(GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES, bpp)
    else {
        log::warn!(
            "align_width_for_gpu_pitch: lcm({GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES}, {bpp}) \
             overflows usize, returning unaligned width {width}"
        );
        return width;
    };
    if lcm_alignment == 0 {
        return width;
    }

    debug_assert_eq!(lcm_alignment % bpp, 0);
    let width_alignment = lcm_alignment / bpp;
    if width_alignment == 0 {
        return width;
    }

    let remainder = width % width_alignment;
    if remainder == 0 {
        return width;
    }

    let pad = width_alignment - remainder;
    match width.checked_add(pad) {
        Some(aligned) => aligned,
        None => {
            log::warn!(
                "align_width_for_gpu_pitch: width {width} + pad {pad} overflows usize, \
                 returning unaligned (caller should use a smaller width or pre-aligned size)"
            );
            width
        }
    }
}

/// Round `min_pitch_bytes` up to the next multiple of
/// [`GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`]. Returns `None` if the rounded
/// value would overflow `usize`. Returns `Some(0)` for input 0.
///
/// Used internally by [`ImageProcessor::create_image`] to compute the
/// padded row stride for DMA-backed image allocations. External callers
/// that need pixel-counted alignment (instead of raw byte pitch) should
/// use [`align_width_for_gpu_pitch`] instead.
#[cfg(target_os = "linux")]
pub(crate) fn align_pitch_bytes_to_gpu_alignment(min_pitch_bytes: usize) -> Option<usize> {
    let alignment = GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES;
    if min_pitch_bytes == 0 {
        return Some(0);
    }
    let remainder = min_pitch_bytes % alignment;
    if remainder == 0 {
        return Some(min_pitch_bytes);
    }
    min_pitch_bytes.checked_add(alignment - remainder)
}

/// Overflow-safe least common multiple. Returns `None` when `(a / gcd) * b`
/// would wrap.
fn checked_num_integer_lcm(a: usize, b: usize) -> Option<usize> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    let g = num_integer_gcd(a, b);
    // a / g is exact (g divides a by definition) and at most a, so this
    // division never panics. Only the subsequent multiply can overflow.
    (a / g).checked_mul(b)
}

fn num_integer_gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        num_integer_gcd(b, a % b)
    }
}

/// Bytes-per-pixel for the primary plane of `format` at element size `elem`.
/// Returns `None` for formats that don't have a single packed BPP (semi-planar
/// chroma is handled separately, returning the luma-plane bpp).
///
/// External callers can use this together with [`align_width_for_gpu_pitch`]
/// to size their own DMA-BUFs without having to remember per-format BPPs:
///
/// ```
/// use edgefirst_image::{align_width_for_gpu_pitch, primary_plane_bpp};
/// use edgefirst_tensor::PixelFormat;
///
/// let bpp = primary_plane_bpp(PixelFormat::Rgba, 1).unwrap();
/// let aligned = align_width_for_gpu_pitch(3004, bpp);
/// assert_eq!(aligned, 3008);
/// ```
pub fn primary_plane_bpp(format: PixelFormat, elem: usize) -> Option<usize> {
    use edgefirst_tensor::PixelLayout;
    match format.layout() {
        PixelLayout::Packed => Some(format.channels() * elem),
        PixelLayout::Planar => Some(elem),
        // For NV12/NV16 the luma plane is single-channel so the pitch
        // matches `elem`; the chroma plane uses the same pitch in bytes
        // (UV is half-width but two interleaved channels = same pitch).
        PixelLayout::SemiPlanar => Some(elem),
        // `PixelLayout` is non-exhaustive — fall through unaligned for
        // any future variant we don't yet recognise.
        _ => None,
    }
}

use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{
    DType, PixelFormat, PixelLayout, Tensor, TensorDyn, TensorMemory, TensorTrait as _,
};
use enum_dispatch::enum_dispatch;
use std::{fmt::Display, time::Instant};
use zune_jpeg::{
    zune_core::{colorspace::ColorSpace, options::DecoderOptions},
    JpegDecoder,
};
use zune_png::PngDecoder;

pub use cpu::CPUProcessor;
pub use error::{Error, Result};
#[cfg(target_os = "linux")]
pub use g2d::G2DProcessor;
#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
pub use opengl_headless::GLProcessorThreaded;
#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
pub use opengl_headless::Int8InterpolationMode;
#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
pub use opengl_headless::{probe_egl_displays, EglDisplayInfo, EglDisplayKind};

mod cpu;
mod error;
mod g2d;
#[path = "gl/mod.rs"]
mod opengl_headless;

// Use `edgefirst_tensor::PixelFormat` variants (Rgb, Rgba, Grey, etc.) and
// `TensorDyn` / `Tensor<u8>` with `.format()` metadata instead.

/// Flips the image data, then rotates it. Returns a new `TensorDyn`.
fn rotate_flip_to_dyn(
    src: &Tensor<u8>,
    src_fmt: PixelFormat,
    rotation: Rotation,
    flip: Flip,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn, Error> {
    let src_w = src.width().unwrap();
    let src_h = src.height().unwrap();
    let channels = src_fmt.channels();

    let (dst_w, dst_h) = match rotation {
        Rotation::None | Rotation::Rotate180 => (src_w, src_h),
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => (src_h, src_w),
    };

    let dst = Tensor::<u8>::image(dst_w, dst_h, src_fmt, memory)?;
    let src_map = src.map()?;
    let mut dst_map = dst.map()?;

    CPUProcessor::flip_rotate_ndarray_pf(
        &src_map,
        &mut dst_map,
        dst_w,
        dst_h,
        channels,
        rotation,
        flip,
    )?;
    drop(dst_map);
    drop(src_map);

    Ok(TensorDyn::from(dst))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    None = 0,
    Clockwise90 = 1,
    Rotate180 = 2,
    CounterClockwise90 = 3,
}
impl Rotation {
    /// Creates a Rotation enum from an angle in degrees. The angle must be a
    /// multiple of 90.
    ///
    /// # Panics
    /// Panics if the angle is not a multiple of 90.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_image::Rotation;
    /// let rotation = Rotation::from_degrees_clockwise(270);
    /// assert_eq!(rotation, Rotation::CounterClockwise90);
    /// ```
    pub fn from_degrees_clockwise(angle: usize) -> Rotation {
        match angle.rem_euclid(360) {
            0 => Rotation::None,
            90 => Rotation::Clockwise90,
            180 => Rotation::Rotate180,
            270 => Rotation::CounterClockwise90,
            _ => panic!("rotation angle is not a multiple of 90"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flip {
    None = 0,
    Vertical = 1,
    Horizontal = 2,
}

/// Controls how the color palette index is chosen for each detected object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorMode {
    /// Color is chosen by object class label (`det.label`). Default.
    ///
    /// Preserves backward compatibility and is correct for semantic
    /// segmentation where colors carry class meaning.
    #[default]
    Class,
    /// Color is chosen by instance order (loop index, zero-based).
    ///
    /// Each detected object gets a unique color regardless of class,
    /// useful for instance segmentation.
    Instance,
    /// Color is chosen by track ID (future use; currently behaves like
    /// [`Instance`](Self::Instance)).
    Track,
}

impl ColorMode {
    /// Return the palette index for a detection given its loop index and label.
    #[inline]
    pub fn index(self, idx: usize, label: usize) -> usize {
        match self {
            ColorMode::Class => label,
            ColorMode::Instance | ColorMode::Track => idx,
        }
    }
}

/// Controls the resolution and coordinate frame of masks produced by
/// [`ImageProcessor::materialize_masks`].
///
/// - [`Proto`](Self::Proto) returns per-detection tiles at proto-plane
///   resolution (e.g. 48×32 u8 for a typical COCO bbox on a 160×160 proto
///   plane). This is the historical behavior of `materialize_masks` and the
///   fastest path because no upsample runs inside HAL. Mask values are
///   continuous sigmoid output quantized to `uint8 [0, 255]`.
/// - [`Scaled`](Self::Scaled) returns per-detection tiles at caller-specified
///   pixel resolution by upsampling the full proto plane once and cropping by
///   bbox after sigmoid. The upsample uses bilinear interpolation with
///   edge-clamp sampling — semantically equivalent to Ultralytics'
///   `process_masks_retina` reference. When a `letterbox` is also passed to
///   [`materialize_masks`], the inverse letterbox transform is applied during
///   the upsample so mask pixels land in original-content coordinates
///   (drop-in for overlay on the original image). Mask values are binary
///   `uint8 {0, 255}` after thresholding sigmoid > 0.5 — interchangeable
///   with `Proto` output via the same `> 127` test.
///
/// [`materialize_masks`]: ImageProcessor::materialize_masks
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MaskResolution {
    /// Per-detection tile at proto-plane resolution (default).
    #[default]
    Proto,
    /// Per-detection tile at `(width, height)` pixel resolution in the
    /// coordinate frame determined by the `letterbox` parameter of
    /// [`ImageProcessor::materialize_masks`].
    Scaled {
        /// Target pixel width of the output coordinate frame.
        width: u32,
        /// Target pixel height of the output coordinate frame.
        height: u32,
    },
}

/// Options for mask overlay rendering.
///
/// Controls how segmentation masks are composited onto the destination image:
/// - `background`: when set, the background image is drawn first and masks
///   are composited over it (result written to `dst`). When `None`, `dst` is
///   cleared to `0x00000000` (fully transparent) before masks are drawn.
///   **`dst` is always fully overwritten — its prior contents are never
///   preserved.** Callers who used to pre-load an image into `dst` before
///   calling `draw_decoded_masks` / `draw_proto_masks` must now supply that
///   image via `background` instead (behaviour changed in v0.16.4).
/// - `opacity`: scales the alpha of rendered mask colors. `1.0` (default)
///   preserves the class color's alpha unchanged; `0.5` makes masks
///   semi-transparent.
/// - `color_mode`: controls whether colors are assigned by class label,
///   instance index, or track ID. Defaults to [`ColorMode::Class`].
#[derive(Debug, Clone, Copy)]
pub struct MaskOverlay<'a> {
    /// Compositing source image. Must have the same dimensions and pixel
    /// format as `dst`. When `Some`, the output is `background + masks`.
    /// When `None`, `dst` is cleared to `0x00000000` before masks are drawn.
    pub background: Option<&'a TensorDyn>,
    pub opacity: f32,
    /// Normalized letterbox region `[xmin, ymin, xmax, ymax]` in model-input
    /// space that contains actual image content (the rest is padding).
    ///
    /// When set, bounding boxes and mask coordinates from the decoder (which
    /// are in model-input normalized space) are mapped back to the original
    /// image coordinate space before rendering.
    ///
    /// Use [`with_letterbox_crop`](Self::with_letterbox_crop) to compute this
    /// from the [`Crop`] that was used in the model input [`convert`](crate::ImageProcessorTrait::convert) call.
    pub letterbox: Option<[f32; 4]>,
    pub color_mode: ColorMode,
}

impl Default for MaskOverlay<'_> {
    fn default() -> Self {
        Self {
            background: None,
            opacity: 1.0,
            letterbox: None,
            color_mode: ColorMode::Class,
        }
    }
}

impl<'a> MaskOverlay<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compositing source image.
    ///
    /// `bg` must have the same dimensions and pixel format as the `dst` passed
    /// to [`draw_decoded_masks`](crate::ImageProcessorTrait::draw_decoded_masks) /
    /// [`draw_proto_masks`](crate::ImageProcessorTrait::draw_proto_masks).
    /// The output will be `bg + masks`. Without a background, `dst` is cleared
    /// to `0x00000000`.
    pub fn with_background(mut self, bg: &'a TensorDyn) -> Self {
        self.background = Some(bg);
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn with_color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    /// Set the letterbox transform from the [`Crop`] used when preparing the
    /// model input, so that bounding boxes and masks are correctly mapped back
    /// to the original image coordinate space during rendering.
    ///
    /// Pass the same `crop` that was given to
    /// [`convert`](crate::ImageProcessorTrait::convert) along with the model
    /// input dimensions (`model_w` × `model_h`).
    ///
    /// Has no effect when `crop.dst_rect` is `None` (no letterbox applied).
    pub fn with_letterbox_crop(mut self, crop: &Crop, model_w: usize, model_h: usize) -> Self {
        if let Some(r) = crop.dst_rect {
            self.letterbox = Some([
                r.left as f32 / model_w as f32,
                r.top as f32 / model_h as f32,
                (r.left + r.width) as f32 / model_w as f32,
                (r.top + r.height) as f32 / model_h as f32,
            ]);
        }
        self
    }
}

/// Apply the inverse letterbox transform to a bounding box.
///
/// `letterbox` is `[lx0, ly0, lx1, ly1]` — the normalized region of the model
/// input that contains actual image content (output of
/// [`MaskOverlay::with_letterbox_crop`]).
///
/// Converts model-input-normalized coords to output-image-normalized coords,
/// clamped to `[0.0, 1.0]`. Also canonicalises the bbox (ensures xmin ≤ xmax).
#[inline]
fn unletter_bbox(bbox: DetectBox, lb: [f32; 4]) -> DetectBox {
    let b = bbox.bbox.to_canonical();
    let [lx0, ly0, lx1, ly1] = lb;
    let inv_w = if lx1 > lx0 { 1.0 / (lx1 - lx0) } else { 1.0 };
    let inv_h = if ly1 > ly0 { 1.0 / (ly1 - ly0) } else { 1.0 };
    DetectBox {
        bbox: edgefirst_decoder::BoundingBox {
            xmin: ((b.xmin - lx0) * inv_w).clamp(0.0, 1.0),
            ymin: ((b.ymin - ly0) * inv_h).clamp(0.0, 1.0),
            xmax: ((b.xmax - lx0) * inv_w).clamp(0.0, 1.0),
            ymax: ((b.ymax - ly0) * inv_h).clamp(0.0, 1.0),
        },
        ..bbox
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crop {
    pub src_rect: Option<Rect>,
    pub dst_rect: Option<Rect>,
    pub dst_color: Option<[u8; 4]>,
}

impl Default for Crop {
    fn default() -> Self {
        Crop::new()
    }
}
impl Crop {
    // Creates a new Crop with default values (no cropping).
    pub fn new() -> Self {
        Crop {
            src_rect: None,
            dst_rect: None,
            dst_color: None,
        }
    }

    // Sets the source rectangle for cropping.
    pub fn with_src_rect(mut self, src_rect: Option<Rect>) -> Self {
        self.src_rect = src_rect;
        self
    }

    // Sets the destination rectangle for cropping.
    pub fn with_dst_rect(mut self, dst_rect: Option<Rect>) -> Self {
        self.dst_rect = dst_rect;
        self
    }

    // Sets the destination color for areas outside the cropped region.
    pub fn with_dst_color(mut self, dst_color: Option<[u8; 4]>) -> Self {
        self.dst_color = dst_color;
        self
    }

    // Creates a new Crop with no cropping.
    pub fn no_crop() -> Self {
        Crop::new()
    }

    /// Validate crop rectangles against explicit dimensions.
    pub(crate) fn check_crop_dims(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<(), Error> {
        let src_ok = self
            .src_rect
            .is_none_or(|r| r.left + r.width <= src_w && r.top + r.height <= src_h);
        let dst_ok = self
            .dst_rect
            .is_none_or(|r| r.left + r.width <= dst_w && r.top + r.height <= dst_h);
        match (src_ok, dst_ok) {
            (true, true) => Ok(()),
            (true, false) => Err(Error::CropInvalid(format!(
                "Dest crop invalid: {:?}",
                self.dst_rect
            ))),
            (false, true) => Err(Error::CropInvalid(format!(
                "Src crop invalid: {:?}",
                self.src_rect
            ))),
            (false, false) => Err(Error::CropInvalid(format!(
                "Dest and Src crop invalid: {:?} {:?}",
                self.dst_rect, self.src_rect
            ))),
        }
    }

    /// Validate crop rectangles against TensorDyn source and destination.
    pub fn check_crop_dyn(
        &self,
        src: &edgefirst_tensor::TensorDyn,
        dst: &edgefirst_tensor::TensorDyn,
    ) -> Result<(), Error> {
        self.check_crop_dims(
            src.width().unwrap_or(0),
            src.height().unwrap_or(0),
            dst.width().unwrap_or(0),
            dst.height().unwrap_or(0),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    // Creates a new Rect with the specified left, top, width, and height.
    pub fn new(left: usize, top: usize, width: usize, height: usize) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    // Checks if the rectangle is valid for the given TensorDyn image.
    pub fn check_rect_dyn(&self, image: &TensorDyn) -> bool {
        let w = image.width().unwrap_or(0);
        let h = image.height().unwrap_or(0);
        self.left + self.width <= w && self.top + self.height <= h
    }
}

#[enum_dispatch(ImageProcessor)]
pub trait ImageProcessorTrait {
    /// Converts the source image to the destination image format and size. The
    /// image is cropped first, then flipped, then rotated
    ///
    /// # Arguments
    ///
    /// * `dst` - The destination image to be converted to.
    /// * `src` - The source image to convert from.
    /// * `rotation` - The rotation to apply to the destination image.
    /// * `flip` - Flips the image
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the conversion.
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()>;

    /// Draw pre-decoded detection boxes and segmentation masks onto `dst`.
    ///
    /// Supports two segmentation modes based on the mask channel count:
    /// - **Instance segmentation** (`C=1`): one `Segmentation` per detection,
    ///   `segmentation` and `detect` are zipped.
    /// - **Semantic segmentation** (`C>1`): a single `Segmentation` covering
    ///   all classes; only the first element is used.
    ///
    /// # Format requirements
    ///
    /// - CPU backend: `dst` must be `RGBA` or `RGB`.
    /// - OpenGL backend: `dst` must be `RGBA`, `BGRA`, or `RGB`.
    /// - G2D backend: only produces the base frame (empty detections);
    ///   returns `NotImplemented` when any detection or segmentation is
    ///   supplied.
    ///
    /// # Output contract
    ///
    /// This function always fully writes `dst` — it never relies on the
    /// caller having pre-cleared the destination. The four cases are:
    ///
    /// | detections | background | output                              |
    /// |------------|------------|-------------------------------------|
    /// | none       | none       | dst cleared to `0x00000000`         |
    /// | none       | set        | dst ← background                    |
    /// | set        | none       | masks drawn over cleared dst        |
    /// | set        | set        | masks drawn over background         |
    ///
    /// Each backend implements this with its native primitives: G2D uses
    /// `g2d_clear` / `g2d_blit`, OpenGL uses `glClear` / DMA-BUF GPU blit
    /// plus the mask program, and CPU uses direct buffer fill / memcpy as
    /// the terminal fallback. CPU-memcpy of DMA buffers is avoided on the
    /// accelerated paths.
    ///
    /// An empty `segmentation` slice is valid — only bounding boxes are drawn.
    ///
    /// `overlay` controls compositing: `background` is the compositing source
    /// (must match `dst` in size and format); `opacity` scales mask alpha.
    ///
    /// # Buffer aliasing
    ///
    /// `dst` and `overlay.background` must reference **distinct underlying
    /// buffers**. An aliased pair returns [`Error::AliasedBuffers`] without
    /// dispatching to any backend — the GL path would otherwise read and
    /// write the same texture in a single draw, which is undefined behaviour
    /// on most drivers. Aliasing is detected via
    /// [`TensorDyn::aliases`](edgefirst_tensor::TensorDyn::aliases), which
    /// catches both shared-allocation clones and separate imports over the
    /// same dmabuf fd.
    ///
    /// # Migration from v0.16.3 and earlier
    ///
    /// Prior to v0.16.4 the call silently preserved `dst`'s contents on empty
    /// detections. That invariant no longer holds — `dst` is always fully
    /// written. Callers who pre-loaded an image into `dst` before calling this
    /// function must now pass that image via `overlay.background` instead.
    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: MaskOverlay<'_>,
    ) -> Result<()>;

    /// Draw masks from proto data onto image (fused decode+draw).
    ///
    /// For YOLO segmentation models, this avoids materializing intermediate
    /// `Array3<u8>` masks. The `ProtoData` contains mask coefficients and the
    /// prototype tensor; the renderer computes `mask_coeff @ protos` directly
    /// at the output resolution using bilinear sampling.
    ///
    /// `detect` and `proto_data.mask_coefficients` must have the same length
    /// (enforced by zip — excess entries are silently ignored). An empty
    /// `detect` slice is valid and produces the base frame — cleared or
    /// background-blitted — via the selected backend's native primitive.
    ///
    /// # Format requirements and output contract
    ///
    /// Same as [`draw_decoded_masks`](Self::draw_decoded_masks), including
    /// the "always fully writes dst" guarantee across all four
    /// detection/background combinations.
    ///
    /// `overlay` controls compositing — see [`draw_decoded_masks`](Self::draw_decoded_masks).
    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: MaskOverlay<'_>,
    ) -> Result<()>;

    /// Sets the colors used for rendering segmentation masks. Up to 20 colors
    /// can be set.
    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<()>;
}

/// Configuration for [`ImageProcessor`] construction.
///
/// Use with [`ImageProcessor::with_config`] to override the default EGL
/// display auto-detection and backend selection. The default configuration
/// preserves the existing auto-detection behaviour.
#[derive(Debug, Clone, Default)]
pub struct ImageProcessorConfig {
    /// Force OpenGL to use this EGL display type instead of auto-detecting.
    ///
    /// When `None`, the processor probes displays in priority order: GBM,
    /// PlatformDevice, Default. Use [`probe_egl_displays`] to discover
    /// which displays are available on the current system.
    ///
    /// Ignored when `EDGEFIRST_DISABLE_GL=1` is set.
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    pub egl_display: Option<EglDisplayKind>,

    /// Preferred compute backend.
    ///
    /// When set to a specific backend (not [`ComputeBackend::Auto`]), the
    /// processor initializes that backend with no fallback — returns an error if the conversion is not supported.
    /// This takes precedence over `EDGEFIRST_FORCE_BACKEND` and the
    /// `EDGEFIRST_DISABLE_*` environment variables.
    ///
    /// - [`ComputeBackend::OpenGl`]: init OpenGL + CPU, skip G2D
    /// - [`ComputeBackend::G2d`]: init G2D + CPU, skip OpenGL
    /// - [`ComputeBackend::Cpu`]: init CPU only
    /// - [`ComputeBackend::Auto`]: existing env-var-driven selection
    pub backend: ComputeBackend,
}

/// Compute backend selection for [`ImageProcessor`].
///
/// Use with [`ImageProcessorConfig::backend`] to select which backend the
/// processor should prefer. When a specific backend is selected, the
/// processor initializes that backend plus CPU as a fallback. When `Auto`
/// is used, the existing environment-variable-driven selection applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ComputeBackend {
    /// Auto-detect based on available hardware and environment variables.
    #[default]
    Auto,
    /// CPU-only processing (no hardware acceleration).
    Cpu,
    /// Prefer G2D hardware blitter (+ CPU fallback).
    G2d,
    /// Prefer OpenGL ES (+ CPU fallback).
    OpenGl,
}

/// Backend forced via the `EDGEFIRST_FORCE_BACKEND` environment variable
/// or [`ImageProcessorConfig::backend`].
///
/// When set, the [`ImageProcessor`] only initializes and dispatches to the
/// selected backend — no fallback chain is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForcedBackend {
    Cpu,
    G2d,
    OpenGl,
}

/// Image converter that uses available hardware acceleration or CPU as a
/// fallback.
#[derive(Debug)]
pub struct ImageProcessor {
    /// CPU-based image converter as a fallback. This is only None if the
    /// EDGEFIRST_DISABLE_CPU environment variable is set.
    pub cpu: Option<CPUProcessor>,

    #[cfg(target_os = "linux")]
    /// G2D-based image converter for Linux systems. This is only available if
    /// the EDGEFIRST_DISABLE_G2D environment variable is not set and libg2d.so
    /// is available.
    pub g2d: Option<G2DProcessor>,
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    /// OpenGL-based image converter for Linux systems. This is only available
    /// if the EDGEFIRST_DISABLE_GL environment variable is not set and OpenGL
    /// ES is available.
    pub opengl: Option<GLProcessorThreaded>,

    /// When set, only the specified backend is used — no fallback chain.
    pub(crate) forced_backend: Option<ForcedBackend>,
}

unsafe impl Send for ImageProcessor {}
unsafe impl Sync for ImageProcessor {}

impl ImageProcessor {
    /// Creates a new `ImageProcessor` instance, initializing available
    /// hardware converters based on the system capabilities and environment
    /// variables.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait, load_image};
    /// # use edgefirst_tensor::{PixelFormat, DType, TensorDyn};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let image = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
    /// let src = load_image(image, Some(PixelFormat::Rgba), None)?;
    /// let mut converter = ImageProcessor::new()?;
    /// let mut dst = converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None)?;
    /// converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        Self::with_config(ImageProcessorConfig::default())
    }

    /// Creates a new `ImageProcessor` with the given configuration.
    ///
    /// When [`ImageProcessorConfig::backend`] is set to a specific backend,
    /// environment variables are ignored and the processor initializes the
    /// requested backend plus CPU as a fallback.
    ///
    /// When `Auto`, the existing `EDGEFIRST_FORCE_BACKEND` and
    /// `EDGEFIRST_DISABLE_*` environment variables apply.
    #[allow(unused_variables)]
    pub fn with_config(config: ImageProcessorConfig) -> Result<Self> {
        // ── Config-driven backend selection ──────────────────────────
        // When the caller explicitly requests a backend via the config,
        // skip all environment variable logic.
        match config.backend {
            ComputeBackend::Cpu => {
                log::info!("ComputeBackend::Cpu — CPU only");
                return Ok(Self {
                    cpu: Some(CPUProcessor::new()),
                    #[cfg(target_os = "linux")]
                    g2d: None,
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    opengl: None,
                    forced_backend: None,
                });
            }
            ComputeBackend::G2d => {
                log::info!("ComputeBackend::G2d — G2D + CPU fallback");
                #[cfg(target_os = "linux")]
                {
                    let g2d = match G2DProcessor::new() {
                        Ok(g) => Some(g),
                        Err(e) => {
                            log::warn!("G2D requested but failed to initialize: {e:?}");
                            None
                        }
                    };
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        g2d,
                        #[cfg(feature = "opengl")]
                        opengl: None,
                        forced_backend: None,
                    });
                }
                #[cfg(not(target_os = "linux"))]
                {
                    log::warn!("G2D requested but not available on this platform, using CPU");
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        forced_backend: None,
                    });
                }
            }
            ComputeBackend::OpenGl => {
                log::info!("ComputeBackend::OpenGl — OpenGL + CPU fallback");
                #[cfg(target_os = "linux")]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::new(config.egl_display) {
                        Ok(gl) => Some(gl),
                        Err(e) => {
                            log::warn!("OpenGL requested but failed to initialize: {e:?}");
                            None
                        }
                    };
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        g2d: None,
                        #[cfg(feature = "opengl")]
                        opengl,
                        forced_backend: None,
                    });
                }
                #[cfg(not(target_os = "linux"))]
                {
                    log::warn!("OpenGL requested but not available on this platform, using CPU");
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        forced_backend: None,
                    });
                }
            }
            ComputeBackend::Auto => { /* fall through to env-var logic below */ }
        }

        // ── EDGEFIRST_FORCE_BACKEND ──────────────────────────────────
        // When set, only the requested backend is initialised and no
        // fallback chain is used. Accepted values (case-insensitive):
        //   "cpu", "g2d", "opengl"
        if let Ok(val) = std::env::var("EDGEFIRST_FORCE_BACKEND") {
            let val_lower = val.to_lowercase();
            let forced = match val_lower.as_str() {
                "cpu" => ForcedBackend::Cpu,
                "g2d" => ForcedBackend::G2d,
                "opengl" => ForcedBackend::OpenGl,
                other => {
                    return Err(Error::ForcedBackendUnavailable(format!(
                        "unknown EDGEFIRST_FORCE_BACKEND value: {other:?} (expected cpu, g2d, or opengl)"
                    )));
                }
            };

            log::info!("EDGEFIRST_FORCE_BACKEND={val} — only initializing {val_lower} backend");

            return match forced {
                ForcedBackend::Cpu => Ok(Self {
                    cpu: Some(CPUProcessor::new()),
                    #[cfg(target_os = "linux")]
                    g2d: None,
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    opengl: None,
                    forced_backend: Some(ForcedBackend::Cpu),
                }),
                ForcedBackend::G2d => {
                    #[cfg(target_os = "linux")]
                    {
                        let g2d = G2DProcessor::new().map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "g2d forced but failed to initialize: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            g2d: Some(g2d),
                            #[cfg(feature = "opengl")]
                            opengl: None,
                            forced_backend: Some(ForcedBackend::G2d),
                        })
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        Err(Error::ForcedBackendUnavailable(
                            "g2d backend is only available on Linux".into(),
                        ))
                    }
                }
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::new(config.egl_display).map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced but failed to initialize: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            g2d: None,
                            opengl: Some(opengl),
                            forced_backend: Some(ForcedBackend::OpenGl),
                        })
                    }
                    #[cfg(not(all(target_os = "linux", feature = "opengl")))]
                    {
                        Err(Error::ForcedBackendUnavailable(
                            "opengl backend requires Linux with the 'opengl' feature enabled"
                                .into(),
                        ))
                    }
                }
            };
        }

        // ── Existing DISABLE logic (unchanged) ──────────────────────
        #[cfg(target_os = "linux")]
        let g2d = if std::env::var("EDGEFIRST_DISABLE_G2D")
            .map(|x| x != "0" && x.to_lowercase() != "false")
            .unwrap_or(false)
        {
            log::debug!("EDGEFIRST_DISABLE_G2D is set");
            None
        } else {
            match G2DProcessor::new() {
                Ok(g2d_converter) => Some(g2d_converter),
                Err(err) => {
                    log::warn!("Failed to initialize G2D converter: {err:?}");
                    None
                }
            }
        };

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        let opengl = if std::env::var("EDGEFIRST_DISABLE_GL")
            .map(|x| x != "0" && x.to_lowercase() != "false")
            .unwrap_or(false)
        {
            log::debug!("EDGEFIRST_DISABLE_GL is set");
            None
        } else {
            match GLProcessorThreaded::new(config.egl_display) {
                Ok(gl_converter) => Some(gl_converter),
                Err(err) => {
                    log::warn!("Failed to initialize GL converter: {err:?}");
                    None
                }
            }
        };

        let cpu = if std::env::var("EDGEFIRST_DISABLE_CPU")
            .map(|x| x != "0" && x.to_lowercase() != "false")
            .unwrap_or(false)
        {
            log::debug!("EDGEFIRST_DISABLE_CPU is set");
            None
        } else {
            Some(CPUProcessor::new())
        };
        Ok(Self {
            cpu,
            #[cfg(target_os = "linux")]
            g2d,
            #[cfg(target_os = "linux")]
            #[cfg(feature = "opengl")]
            opengl,
            forced_backend: None,
        })
    }

    /// Sets the interpolation mode for int8 proto textures on the OpenGL
    /// backend. No-op if OpenGL is not available.
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) -> Result<()> {
        if let Some(ref mut gl) = self.opengl {
            gl.set_int8_interpolation_mode(mode)?;
        }
        Ok(())
    }

    /// Create a [`TensorDyn`] image with the best available memory backend.
    ///
    /// Priority: DMA-buf → PBO (byte-sized types: u8, i8) → system memory.
    ///
    /// Use this method instead of [`TensorDyn::image()`] when the tensor will
    /// be used with [`ImageProcessor::convert()`]. It selects the optimal
    /// memory backing (including PBO for GPU zero-copy) which direct
    /// allocation cannot achieve.
    ///
    /// This method is on [`ImageProcessor`] rather than [`ImageProcessorTrait`]
    /// because optimal allocation requires knowledge of the active compute
    /// backends (e.g. the GL context handle for PBO allocation). Individual
    /// backend implementations ([`CPUProcessor`], etc.) do not have this
    /// cross-backend visibility.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `format` - Pixel format
    /// * `dtype` - Element data type (e.g. `DType::U8`, `DType::I8`)
    /// * `memory` - Optional memory type override; when `None`, the best
    ///   available backend is selected automatically.
    ///
    /// # Returns
    ///
    /// A [`TensorDyn`] backed by the highest-performance memory type
    /// available on this system.
    ///
    /// # Pitch alignment for DMA-backed allocations
    ///
    /// DMA-BUF imports into the GL backend (Mali Valhall on i.MX 95
    /// specifically) require every row pitch to be a multiple of
    /// [`GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`] (currently 64). When this
    /// method lands on `TensorMemory::Dma`, the underlying allocation is
    /// silently padded so the row stride satisfies that requirement.
    ///
    /// **The user-requested `width` is preserved** — `tensor.width()`
    /// returns the same value you passed in. The padding is carried by
    /// [`TensorDyn::row_stride`] / `effective_row_stride()`, which the
    /// GL backend reads when importing the buffer as an EGLImage.
    /// Callers that compute byte offsets from the tensor must use the
    /// stride, not `width × bytes_per_pixel`; the CPU mapping spans the
    /// full `stride × height` bytes.
    ///
    /// Pre-aligned widths (640, 1280, 1920, 3008, 3840 …) allocate
    /// exactly `width × bpp × height` bytes with no padding. PBO and
    /// Mem fallbacks never pad — they don't go through EGLImage import.
    ///
    /// See also [`align_width_for_gpu_pitch`] for an advisory helper
    /// that external callers (GStreamer plugins, video pipelines) can
    /// use to size their own DMA-BUFs for GL compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error if all allocation strategies fail.
    pub fn create_image(
        &self,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
    ) -> Result<TensorDyn> {
        // Compute the GPU-aligned row stride in bytes for this image.
        // `None` means either the format has no defined primary-plane bpp
        // (unknown future layout) or the stride calculation would overflow
        // — in both cases we fall back to the natural layout via the plain
        // `TensorDyn::image` constructor, and the slow-path warning inside
        // `draw_*_masks` will fire if the subsequent GL import fails.
        //
        // DMA allocation is Linux-only (see `TensorMemory::Dma` cfg gate),
        // so both the stride computation and the helper closure are gated
        // accordingly — the callers below are already Linux-only.
        #[cfg(target_os = "linux")]
        let dma_stride_bytes: Option<usize> = primary_plane_bpp(format, dtype.size())
            .and_then(|bpp| width.checked_mul(bpp))
            .and_then(align_pitch_bytes_to_gpu_alignment);

        // Helper: allocate a DMA image, using the padded-stride constructor
        // when the computed stride exceeds the natural pitch, otherwise the
        // plain constructor (byte-identical result in the common case).
        #[cfg(target_os = "linux")]
        let try_dma = || -> Result<TensorDyn> {
            // Stride padding is only meaningful for packed pixel layouts
            // (RGBA8, BGRA8, RGB888, Grey) — the formats the GL backend
            // renders into. Semi-planar (NV12, NV16) and planar (PlanarRgb,
            // PlanarRgba) tensors go through `TensorDyn::image(...)` with
            // their natural layout; they're imported from camera capture
            // via `from_fd` far more often than allocated here, and
            // `Tensor::image_with_stride` explicitly rejects them.
            let packed = format.layout() == edgefirst_tensor::PixelLayout::Packed;
            match dma_stride_bytes {
                Some(stride)
                    if packed
                        && primary_plane_bpp(format, dtype.size())
                            .and_then(|bpp| width.checked_mul(bpp))
                            .is_some_and(|natural| stride > natural) =>
                {
                    log::debug!(
                        "create_image: padding row stride for {format:?} {width}x{height} \
                         from natural pitch to {stride} bytes for GPU alignment"
                    );
                    Ok(TensorDyn::image_with_stride(
                        width,
                        height,
                        format,
                        dtype,
                        stride,
                        Some(edgefirst_tensor::TensorMemory::Dma),
                    )?)
                }
                _ => Ok(TensorDyn::image(
                    width,
                    height,
                    format,
                    dtype,
                    Some(edgefirst_tensor::TensorMemory::Dma),
                )?),
            }
        };

        // If an explicit memory type is requested, honour it directly.
        // On Linux, `TensorMemory::Dma` gets the padded-stride treatment;
        // other memory types take the user-requested width verbatim.
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Dma) => {
                return try_dma();
            }
            Some(mem) => {
                return Ok(TensorDyn::image(width, height, format, dtype, Some(mem))?);
            }
            None => {}
        }

        // Try DMA first on Linux — skip only when GL has explicitly selected PBO
        // as the preferred transfer path (PBO is better than DMA in that case).
        #[cfg(target_os = "linux")]
        {
            #[cfg(feature = "opengl")]
            let gl_uses_pbo = self
                .opengl
                .as_ref()
                .is_some_and(|gl| gl.transfer_backend() == opengl_headless::TransferBackend::Pbo);
            #[cfg(not(feature = "opengl"))]
            let gl_uses_pbo = false;

            if !gl_uses_pbo {
                if let Ok(img) = try_dma() {
                    return Ok(img);
                }
            }
        }

        // Try PBO (if GL available).
        // PBO buffers are u8-sized; the int8 shader emulates i8 output via
        // XOR 0x80 on the same underlying buffer, so both U8 and I8 work.
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if dtype.size() == 1 {
            if let Some(gl) = &self.opengl {
                match gl.create_pbo_image(width, height, format) {
                    Ok(t) => {
                        if dtype == DType::I8 {
                            // SAFETY: Tensor<u8> and Tensor<i8> are layout-
                            // identical (same element size, no T-dependent
                            // drop glue). The int8 shader applies XOR 0x80
                            // on the same PBO buffer. Same rationale as
                            // gl::processor::tensor_i8_as_u8_mut.
                            // Invariant: PBO tensors never have chroma
                            // (create_pbo_image → Tensor::wrap sets it None).
                            debug_assert!(
                                t.chroma().is_none(),
                                "PBO i8 transmute requires chroma == None"
                            );
                            let t_i8: Tensor<i8> = unsafe { std::mem::transmute(t) };
                            return Ok(TensorDyn::from(t_i8));
                        }
                        return Ok(TensorDyn::from(t));
                    }
                    Err(e) => log::debug!("PBO image creation failed, falling back to Mem: {e:?}"),
                }
            }
        }

        // Fallback to Mem
        Ok(TensorDyn::image(
            width,
            height,
            format,
            dtype,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )?)
    }

    /// Import an external DMA-BUF image.
    ///
    /// Each [`PlaneDescriptor`] owns an already-duped fd; this method
    /// consumes the descriptors and takes ownership of those fds (whether
    /// the call succeeds or fails).
    ///
    /// The caller must ensure the DMA-BUF allocation is large enough for the
    /// specified width, height, format, and any stride/offset on the plane
    /// descriptors. No buffer-size validation is performed; an undersized
    /// buffer may cause GPU faults or EGL import failure.
    ///
    /// # Arguments
    ///
    /// * `image` - Plane descriptor for the primary (or only) plane
    /// * `chroma` - Optional plane descriptor for the UV chroma plane
    ///   (required for multiplane NV12)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `format` - Pixel format of the buffer
    /// * `dtype` - Element data type (e.g. `DType::U8`)
    ///
    /// # Returns
    ///
    /// A `TensorDyn` configured as an image.
    ///
    /// # Errors
    ///
    /// * [`Error::NotSupported`] if `chroma` is `Some` for a non-semi-planar
    ///   format, or multiplane NV16 (not yet supported), or the fd is not
    ///   DMA-backed
    /// * [`Error::InvalidShape`] if NV12 height is odd
    ///
    /// # Platform
    ///
    /// Linux only.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use edgefirst_tensor::PlaneDescriptor;
    ///
    /// // Single-plane RGBA
    /// let pd = PlaneDescriptor::new(fd.as_fd())?;
    /// let src = proc.import_image(pd, None, 1920, 1080, PixelFormat::Rgba, DType::U8)?;
    ///
    /// // Multi-plane NV12 with stride
    /// let y_pd = PlaneDescriptor::new(y_fd.as_fd())?.with_stride(2048);
    /// let uv_pd = PlaneDescriptor::new(uv_fd.as_fd())?.with_stride(2048);
    /// let src = proc.import_image(y_pd, Some(uv_pd), 1920, 1080,
    ///                             PixelFormat::Nv12, DType::U8)?;
    /// ```
    #[cfg(target_os = "linux")]
    pub fn import_image(
        &self,
        image: edgefirst_tensor::PlaneDescriptor,
        chroma: Option<edgefirst_tensor::PlaneDescriptor>,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
    ) -> Result<TensorDyn> {
        use edgefirst_tensor::{Tensor, TensorMemory};

        // Capture stride/offset from descriptors before consuming them
        let image_stride = image.stride();
        let image_offset = image.offset();
        let chroma_stride = chroma.as_ref().and_then(|c| c.stride());
        let chroma_offset = chroma.as_ref().and_then(|c| c.offset());

        if let Some(chroma_pd) = chroma {
            // ── Multiplane path ──────────────────────────────────────
            // Multiplane tensors are backed by Tensor<u8> (or transmuted to
            // Tensor<i8>). Reject other dtypes to avoid silently returning a
            // tensor with the wrong element type.
            if dtype != DType::U8 && dtype != DType::I8 {
                return Err(Error::NotSupported(format!(
                    "multiplane import only supports U8/I8, got {dtype:?}"
                )));
            }
            if format.layout() != PixelLayout::SemiPlanar {
                return Err(Error::NotSupported(format!(
                    "import_image with chroma requires a semi-planar format, got {format:?}"
                )));
            }

            let chroma_h = match format {
                PixelFormat::Nv12 => {
                    if !height.is_multiple_of(2) {
                        return Err(Error::InvalidShape(format!(
                            "NV12 requires even height, got {height}"
                        )));
                    }
                    height / 2
                }
                // NV16 multiplane will be supported in a future release;
                // the GL backend currently only handles NV12 plane1 attributes.
                PixelFormat::Nv16 => {
                    return Err(Error::NotSupported(
                        "multiplane NV16 is not yet supported; use contiguous NV16 instead".into(),
                    ))
                }
                _ => {
                    return Err(Error::NotSupported(format!(
                        "unsupported semi-planar format: {format:?}"
                    )))
                }
            };

            let luma = Tensor::<u8>::from_fd(image.into_fd(), &[height, width], Some("luma"))?;
            if luma.memory() != TensorMemory::Dma {
                return Err(Error::NotSupported(format!(
                    "luma fd must be DMA-backed, got {:?}",
                    luma.memory()
                )));
            }

            let chroma_tensor =
                Tensor::<u8>::from_fd(chroma_pd.into_fd(), &[chroma_h, width], Some("chroma"))?;
            if chroma_tensor.memory() != TensorMemory::Dma {
                return Err(Error::NotSupported(format!(
                    "chroma fd must be DMA-backed, got {:?}",
                    chroma_tensor.memory()
                )));
            }

            // from_planes creates the combined tensor with format set,
            // preserving luma's row_stride (currently None since luma was raw).
            let mut tensor = Tensor::<u8>::from_planes(luma, chroma_tensor, format)?;

            // Apply stride/offset to the combined tensor (luma plane)
            if let Some(s) = image_stride {
                tensor.set_row_stride(s)?;
            }
            if let Some(o) = image_offset {
                tensor.set_plane_offset(o);
            }

            // Apply stride/offset to the chroma sub-tensor.
            // The chroma tensor is a raw 2D [chroma_h, width] tensor without
            // format metadata, so we validate stride manually rather than
            // using set_row_stride (which requires format).
            if let Some(chroma_ref) = tensor.chroma_mut() {
                if let Some(s) = chroma_stride {
                    if s < width {
                        return Err(Error::InvalidShape(format!(
                            "chroma stride {s} < minimum {width} for {format:?}"
                        )));
                    }
                    chroma_ref.set_row_stride_unchecked(s);
                }
                if let Some(o) = chroma_offset {
                    chroma_ref.set_plane_offset(o);
                }
            }

            if dtype == DType::I8 {
                // SAFETY: Tensor<u8> and Tensor<i8> have identical layout because
                // the struct contains only type-erased storage (OwnedFd, shape, name),
                // no inline T values. This assertion catches layout drift at compile time.
                const {
                    assert!(std::mem::size_of::<Tensor<u8>>() == std::mem::size_of::<Tensor<i8>>());
                    assert!(
                        std::mem::align_of::<Tensor<u8>>() == std::mem::align_of::<Tensor<i8>>()
                    );
                }
                let tensor_i8: Tensor<i8> = unsafe { std::mem::transmute(tensor) };
                return Ok(TensorDyn::from(tensor_i8));
            }
            Ok(TensorDyn::from(tensor))
        } else {
            // ── Single-plane path ────────────────────────────────────
            let shape = match format.layout() {
                PixelLayout::Packed => vec![height, width, format.channels()],
                PixelLayout::Planar => vec![format.channels(), height, width],
                PixelLayout::SemiPlanar => {
                    let total_h = match format {
                        PixelFormat::Nv12 => {
                            if !height.is_multiple_of(2) {
                                return Err(Error::InvalidShape(format!(
                                    "NV12 requires even height, got {height}"
                                )));
                            }
                            height * 3 / 2
                        }
                        PixelFormat::Nv16 => height * 2,
                        _ => {
                            return Err(Error::InvalidShape(format!(
                                "unknown semi-planar height multiplier for {format:?}"
                            )))
                        }
                    };
                    vec![total_h, width]
                }
                _ => {
                    return Err(Error::NotSupported(format!(
                        "unsupported pixel layout for import_image: {:?}",
                        format.layout()
                    )));
                }
            };
            let tensor = TensorDyn::from_fd(image.into_fd(), &shape, dtype, None)?;
            if tensor.memory() != TensorMemory::Dma {
                return Err(Error::NotSupported(format!(
                    "import_image requires DMA-backed fd, got {:?}",
                    tensor.memory()
                )));
            }
            let mut tensor = tensor.with_format(format)?;
            if let Some(s) = image_stride {
                tensor.set_row_stride(s)?;
            }
            if let Some(o) = image_offset {
                tensor.set_plane_offset(o);
            }
            Ok(tensor)
        }
    }

    /// Decode model outputs and draw segmentation masks onto `dst`.
    ///
    /// This is the primary mask rendering API. The processor decodes via the
    /// provided [`Decoder`], selects the optimal rendering path (hybrid
    /// CPU+GL or fused GPU), and composites masks onto `dst`.
    ///
    /// Returns the detected bounding boxes.
    pub fn draw_masks(
        &mut self,
        decoder: &edgefirst_decoder::Decoder,
        outputs: &[&TensorDyn],
        dst: &mut TensorDyn,
        overlay: MaskOverlay<'_>,
    ) -> Result<Vec<DetectBox>> {
        let mut output_boxes = Vec::with_capacity(100);

        // Try proto path first (fused rendering without materializing masks)
        let proto_result = decoder
            .decode_proto(outputs, &mut output_boxes)
            .map_err(|e| Error::Internal(format!("decode_proto: {e:#?}")))?;

        if let Some(proto_data) = proto_result {
            self.draw_proto_masks(dst, &output_boxes, &proto_data, overlay)?;
        } else {
            // Detection-only or unsupported model: full decode + render
            let mut output_masks = Vec::with_capacity(100);
            decoder
                .decode(outputs, &mut output_boxes, &mut output_masks)
                .map_err(|e| Error::Internal(format!("decode: {e:#?}")))?;
            self.draw_decoded_masks(dst, &output_boxes, &output_masks, overlay)?;
        }
        Ok(output_boxes)
    }

    /// Decode tracked model outputs and draw segmentation masks onto `dst`.
    ///
    /// Like [`draw_masks`](Self::draw_masks) but integrates a tracker for
    /// maintaining object identities across frames. The tracker runs after
    /// NMS but before mask extraction.
    ///
    /// Returns detected boxes and track info.
    #[cfg(feature = "tracker")]
    pub fn draw_masks_tracked<TR: edgefirst_tracker::Tracker<DetectBox>>(
        &mut self,
        decoder: &edgefirst_decoder::Decoder,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[&TensorDyn],
        dst: &mut TensorDyn,
        overlay: MaskOverlay<'_>,
    ) -> Result<(Vec<DetectBox>, Vec<edgefirst_tracker::TrackInfo>)> {
        let mut output_boxes = Vec::with_capacity(100);
        let mut output_tracks = Vec::new();

        let proto_result = decoder
            .decode_proto_tracked(
                tracker,
                timestamp,
                outputs,
                &mut output_boxes,
                &mut output_tracks,
            )
            .map_err(|e| Error::Internal(format!("decode_proto_tracked: {e:#?}")))?;

        if let Some(proto_data) = proto_result {
            self.draw_proto_masks(dst, &output_boxes, &proto_data, overlay)?;
        } else {
            // Note: decode_proto_tracked returns None for detection-only/ModelPack
            // models WITHOUT calling the tracker. The else branch below is the
            // first (and only) tracker call for those model types.
            let mut output_masks = Vec::with_capacity(100);
            decoder
                .decode_tracked(
                    tracker,
                    timestamp,
                    outputs,
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
                .map_err(|e| Error::Internal(format!("decode_tracked: {e:#?}")))?;
            self.draw_decoded_masks(dst, &output_boxes, &output_masks, overlay)?;
        }
        Ok((output_boxes, output_tracks))
    }

    /// Materialize per-instance segmentation masks from raw prototype data.
    ///
    /// Computes `mask_coeff @ protos` with sigmoid activation for each detection,
    /// producing compact masks at prototype resolution (e.g., 160×160 crops).
    /// Mask values are continuous sigmoid confidence outputs quantized to u8
    /// (0 = background, 255 = full confidence), NOT binary thresholded.
    ///
    /// The returned [`Vec<Segmentation>`] can be:
    /// - Inspected or exported for analytics, IoU computation, etc.
    /// - Passed directly to [`ImageProcessorTrait::draw_decoded_masks`] for
    ///   GPU-interpolated rendering.
    ///
    /// # Performance Note
    ///
    /// Calling `materialize_masks` + `draw_decoded_masks` separately prevents
    /// the HAL from using its internal fused optimization path. For render-only
    /// use cases, prefer [`ImageProcessorTrait::draw_proto_masks`] which selects
    /// the fastest path automatically (currently 1.6×–27× faster on tested
    /// platforms). Use this method when you need access to the intermediate masks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoConverter`] if the CPU backend is not available.
    pub fn materialize_masks(
        &self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        letterbox: Option<[f32; 4]>,
        resolution: MaskResolution,
    ) -> Result<Vec<Segmentation>> {
        let cpu = self.cpu.as_ref().ok_or(Error::NoConverter)?;
        match resolution {
            MaskResolution::Proto => cpu.materialize_segmentations(detect, proto_data, letterbox),
            MaskResolution::Scaled { width, height } => {
                cpu.materialize_scaled_segmentations(detect, proto_data, letterbox, width, height)
            }
        }
    }
}

impl ImageProcessorTrait for ImageProcessor {
    /// Converts the source image to the destination image format and size. The
    /// image is cropped first, then flipped, then rotated
    ///
    /// Prefer hardware accelerators when available, falling back to CPU if
    /// necessary.
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let start = Instant::now();
        let src_fmt = src.format();
        let dst_fmt = dst.format();
        log::trace!(
            "convert: {src_fmt:?}({:?}/{:?}) → {dst_fmt:?}({:?}/{:?}), \
             rotation={rotation:?}, flip={flip:?}, backend={:?}",
            src.dtype(),
            src.memory(),
            dst.dtype(),
            dst.memory(),
            self.forced_backend,
        );

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        let r = cpu.convert(src, dst, rotation, flip, crop);
                        log::trace!(
                            "convert: forced=cpu result={} ({:?})",
                            if r.is_ok() { "ok" } else { "err" },
                            start.elapsed()
                        );
                        return r;
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => {
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        let r = g2d.convert(src, dst, rotation, flip, crop);
                        log::trace!(
                            "convert: forced=g2d result={} ({:?})",
                            if r.is_ok() { "ok" } else { "err" },
                            start.elapsed()
                        );
                        return r;
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        let r = opengl.convert(src, dst, rotation, flip, crop);
                        log::trace!(
                            "convert: forced=opengl result={} ({:?})",
                            if r.is_ok() { "ok" } else { "err" },
                            start.elapsed()
                        );
                        return r;
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Auto fallback chain: OpenGL → G2D → CPU ──────────────────
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            match opengl.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!(
                        "convert: auto selected=opengl for {src_fmt:?}→{dst_fmt:?} ({:?})",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("convert: auto opengl declined {src_fmt:?}→{dst_fmt:?}: {e}");
                }
            }
        }

        #[cfg(target_os = "linux")]
        if let Some(g2d) = self.g2d.as_mut() {
            match g2d.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!(
                        "convert: auto selected=g2d for {src_fmt:?}→{dst_fmt:?} ({:?})",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("convert: auto g2d declined {src_fmt:?}→{dst_fmt:?}: {e}");
                }
            }
        }

        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!(
                        "convert: auto selected=cpu for {src_fmt:?}→{dst_fmt:?} ({:?})",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("convert: auto cpu failed {src_fmt:?}→{dst_fmt:?}: {e}");
                    return Err(e);
                }
            }
        }
        Err(Error::NoConverter)
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        let start = Instant::now();

        if let Some(bg) = overlay.background {
            if bg.aliases(dst) {
                return Err(Error::AliasedBuffers(
                    "background must not reference the same buffer as dst".to_string(),
                ));
            }
        }

        // Un-letterbox detect boxes and segmentation bboxes for rendering when
        // a letterbox was applied to prepare the model input.
        let lb_boxes: Vec<DetectBox>;
        let lb_segs: Vec<Segmentation>;
        let (detect, segmentation) = if let Some(lb) = overlay.letterbox {
            lb_boxes = detect.iter().map(|&d| unletter_bbox(d, lb)).collect();
            // Keep segmentation bboxes in sync with the transformed detect boxes
            // when we have a 1:1 correspondence (instance segmentation).
            lb_segs = if segmentation.len() == lb_boxes.len() {
                segmentation
                    .iter()
                    .zip(lb_boxes.iter())
                    .map(|(s, d)| Segmentation {
                        xmin: d.bbox.xmin,
                        ymin: d.bbox.ymin,
                        xmax: d.bbox.xmax,
                        ymax: d.bbox.ymax,
                        segmentation: s.segmentation.clone(),
                    })
                    .collect()
            } else {
                segmentation.to_vec()
            };
            (lb_boxes.as_slice(), lb_segs.as_slice())
        } else {
            (detect, segmentation)
        };
        #[cfg(target_os = "linux")]
        let is_empty_frame = detect.is_empty() && segmentation.is_empty();

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.draw_decoded_masks(dst, detect, segmentation, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => {
                    // G2D can only produce empty frames (clear / bg blit).
                    // For populated frames it has no rasterizer — fail loudly.
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        return g2d.draw_decoded_masks(dst, detect, segmentation, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    // GL handles background natively via GPU blit, and now
                    // actively clears when there is no background.
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_decoded_masks(dst, detect, segmentation, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Auto dispatch ──────────────────────────────────────────
        // Empty frames prefer G2D when available — a single g2d_clear or
        // g2d_blit is the cheapest HW path to produce the correct output
        // and avoids spinning up the GL pipeline every zero-detection
        // frame in a triple-buffered display loop.
        #[cfg(target_os = "linux")]
        if is_empty_frame {
            if let Some(g2d) = self.g2d.as_mut() {
                match g2d.draw_decoded_masks(dst, detect, segmentation, overlay) {
                    Ok(_) => {
                        log::trace!(
                            "draw_decoded_masks empty frame via g2d in {:?}",
                            start.elapsed()
                        );
                        return Ok(());
                    }
                    Err(e) => log::trace!("g2d empty-frame path unavailable: {e:?}"),
                }
            }
        }

        // Populated frames (or G2D unavailable): GL first, CPU fallback.
        // Both backends now own their own base-layer handling (bg blit
        // or clear), so we hand the overlay through untouched.
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            log::trace!(
                "draw_decoded_masks started with opengl in {:?}",
                start.elapsed()
            );
            match opengl.draw_decoded_masks(dst, detect, segmentation, overlay) {
                Ok(_) => {
                    log::trace!("draw_decoded_masks with opengl in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_decoded_masks didn't work with opengl: {e:?}")
                }
            }
        }

        log::trace!(
            "draw_decoded_masks started with cpu in {:?}",
            start.elapsed()
        );
        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.draw_decoded_masks(dst, detect, segmentation, overlay) {
                Ok(_) => {
                    log::trace!("draw_decoded_masks with cpu in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_decoded_masks didn't work with cpu: {e:?}");
                    return Err(e);
                }
            }
        }
        Err(Error::NoConverter)
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        let start = Instant::now();

        if let Some(bg) = overlay.background {
            if bg.aliases(dst) {
                return Err(Error::AliasedBuffers(
                    "background must not reference the same buffer as dst".to_string(),
                ));
            }
        }

        // Un-letterbox detect boxes for rendering when a letterbox was applied
        // to prepare the model input.  The original `detect` coords are still
        // passed to `materialize_segmentations` (which needs model-space coords
        // to correctly crop the proto tensor) alongside `overlay.letterbox` so
        // it can emit `Segmentation` structs in output-image space.
        let lb_boxes: Vec<DetectBox>;
        let render_detect = if let Some(lb) = overlay.letterbox {
            lb_boxes = detect.iter().map(|&d| unletter_bbox(d, lb)).collect();
            lb_boxes.as_slice()
        } else {
            detect
        };
        #[cfg(target_os = "linux")]
        let is_empty_frame = detect.is_empty();

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.draw_proto_masks(dst, render_detect, proto_data, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => {
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        return g2d.draw_proto_masks(dst, render_detect, proto_data, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_proto_masks(dst, render_detect, proto_data, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Auto dispatch ──────────────────────────────────────────
        // Empty frames: prefer G2D — cheapest HW path (clear or bg blit).
        #[cfg(target_os = "linux")]
        if is_empty_frame {
            if let Some(g2d) = self.g2d.as_mut() {
                match g2d.draw_proto_masks(dst, render_detect, proto_data, overlay) {
                    Ok(_) => {
                        log::trace!(
                            "draw_proto_masks empty frame via g2d in {:?}",
                            start.elapsed()
                        );
                        return Ok(());
                    }
                    Err(e) => log::trace!("g2d empty-frame path unavailable: {e:?}"),
                }
            }
        }

        // Hybrid path: CPU materialize + GL overlay (benchmarked faster than
        // full-GPU draw_proto_masks on all tested platforms: 27× on imx8mp,
        // 4× on imx95, 2.5× on rpi5, 1.6× on x86).
        // GL owns its own bg-blit / glClear — we pass the overlay through.
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            let Some(cpu) = self.cpu.as_ref() else {
                return Err(Error::Internal(
                    "draw_proto_masks requires CPU backend for hybrid path".into(),
                ));
            };
            log::trace!(
                "draw_proto_masks started with hybrid (cpu+opengl) in {:?}",
                start.elapsed()
            );
            let segmentation =
                cpu.materialize_segmentations(detect, proto_data, overlay.letterbox)?;
            match opengl.draw_decoded_masks(dst, render_detect, &segmentation, overlay) {
                Ok(_) => {
                    log::trace!(
                        "draw_proto_masks with hybrid (cpu+opengl) in {:?}",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_proto_masks hybrid path failed, falling back to cpu: {e:?}");
                }
            }
        }

        let Some(cpu) = self.cpu.as_mut() else {
            return Err(Error::Internal(
                "draw_proto_masks requires CPU backend for fallback path".into(),
            ));
        };
        log::trace!("draw_proto_masks started with cpu in {:?}", start.elapsed());
        cpu.draw_proto_masks(dst, render_detect, proto_data, overlay)
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<()> {
        let start = Instant::now();

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.set_class_colors(colors);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => Err(Error::NotSupported(
                    "g2d does not support set_class_colors".into(),
                )),
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.set_class_colors(colors);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // skip G2D as it doesn't support rendering to image

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            log::trace!("image started with opengl in {:?}", start.elapsed());
            match opengl.set_class_colors(colors) {
                Ok(_) => {
                    log::trace!("colors set with opengl in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("colors didn't set with opengl: {e:?}")
                }
            }
        }
        log::trace!("image started with cpu in {:?}", start.elapsed());
        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.set_class_colors(colors) {
                Ok(_) => {
                    log::trace!("colors set with cpu in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("colors didn't set with cpu: {e:?}");
                    return Err(e);
                }
            }
        }
        Err(Error::NoConverter)
    }
}

// ---------------------------------------------------------------------------
// Image loading / saving helpers
// ---------------------------------------------------------------------------

/// Read EXIF orientation from raw EXIF bytes and return (Rotation, Flip).
fn read_exif_orientation(exif_bytes: &[u8]) -> (Rotation, Flip) {
    let exifreader = exif::Reader::new();
    let Ok(exif_) = exifreader.read_raw(exif_bytes.to_vec()) else {
        return (Rotation::None, Flip::None);
    };
    let Some(orientation) = exif_.get_field(exif::Tag::Orientation, exif::In::PRIMARY) else {
        return (Rotation::None, Flip::None);
    };
    match orientation.value.get_uint(0) {
        Some(1) => (Rotation::None, Flip::None),
        Some(2) => (Rotation::None, Flip::Horizontal),
        Some(3) => (Rotation::Rotate180, Flip::None),
        Some(4) => (Rotation::Rotate180, Flip::Horizontal),
        Some(5) => (Rotation::Clockwise90, Flip::Horizontal),
        Some(6) => (Rotation::Clockwise90, Flip::None),
        Some(7) => (Rotation::CounterClockwise90, Flip::Horizontal),
        Some(8) => (Rotation::CounterClockwise90, Flip::None),
        Some(v) => {
            log::warn!("broken orientation EXIF value: {v}");
            (Rotation::None, Flip::None)
        }
        None => (Rotation::None, Flip::None),
    }
}

/// Map a [`PixelFormat`] to the zune-jpeg `ColorSpace` for decoding.
/// Returns `None` for formats that the JPEG decoder cannot output directly.
fn pixelfmt_to_colorspace(fmt: PixelFormat) -> Option<ColorSpace> {
    match fmt {
        PixelFormat::Rgb => Some(ColorSpace::RGB),
        PixelFormat::Rgba => Some(ColorSpace::RGBA),
        PixelFormat::Grey => Some(ColorSpace::Luma),
        _ => None,
    }
}

/// Map a zune-jpeg `ColorSpace` to a [`PixelFormat`].
fn colorspace_to_pixelfmt(cs: ColorSpace) -> Option<PixelFormat> {
    match cs {
        ColorSpace::RGB => Some(PixelFormat::Rgb),
        ColorSpace::RGBA => Some(PixelFormat::Rgba),
        ColorSpace::Luma => Some(PixelFormat::Grey),
        _ => None,
    }
}

/// Load a JPEG image from raw bytes and return a [`TensorDyn`].
fn load_jpeg(
    image: &[u8],
    format: Option<PixelFormat>,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    let colour = match format {
        Some(f) => pixelfmt_to_colorspace(f)
            .ok_or_else(|| Error::NotSupported(format!("Unsupported image format {f:?}")))?,
        None => ColorSpace::RGB,
    };
    let options = DecoderOptions::default().jpeg_set_out_colorspace(colour);
    let mut decoder = JpegDecoder::new_with_options(image, options);
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or(Error::Internal(
        "JPEG did not return decoded image info".to_string(),
    ))?;

    let converted_cs = decoder
        .get_output_colorspace()
        .ok_or(Error::Internal("No output colorspace".to_string()))?;

    let converted_fmt = colorspace_to_pixelfmt(converted_cs).ok_or(Error::NotSupported(
        "Unsupported JPEG decoder output".to_string(),
    ))?;

    let dest_fmt = format.unwrap_or(converted_fmt);

    let (rotation, flip) = decoder
        .exif()
        .map(|x| read_exif_orientation(x))
        .unwrap_or((Rotation::None, Flip::None));

    let w = image_info.width as usize;
    let h = image_info.height as usize;

    if (rotation, flip) == (Rotation::None, Flip::None) {
        let mut img = Tensor::<u8>::image(w, h, dest_fmt, memory)?;

        if converted_fmt != dest_fmt {
            let tmp = Tensor::<u8>::image(w, h, converted_fmt, Some(TensorMemory::Mem))?;
            decoder.decode_into(&mut tmp.map()?)?;
            CPUProcessor::convert_format_pf(&tmp, &mut img, converted_fmt, dest_fmt)?;
            return Ok(TensorDyn::from(img));
        }
        decoder.decode_into(&mut img.map()?)?;
        return Ok(TensorDyn::from(img));
    }

    let mut tmp = Tensor::<u8>::image(w, h, dest_fmt, Some(TensorMemory::Mem))?;

    if converted_fmt != dest_fmt {
        let tmp2 = Tensor::<u8>::image(w, h, converted_fmt, Some(TensorMemory::Mem))?;
        decoder.decode_into(&mut tmp2.map()?)?;
        CPUProcessor::convert_format_pf(&tmp2, &mut tmp, converted_fmt, dest_fmt)?;
    } else {
        decoder.decode_into(&mut tmp.map()?)?;
    }

    rotate_flip_to_dyn(&tmp, dest_fmt, rotation, flip, memory)
}

/// Load a PNG image from raw bytes and return a [`TensorDyn`].
fn load_png(
    image: &[u8],
    format: Option<PixelFormat>,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    let fmt = format.unwrap_or(PixelFormat::Rgb);
    let alpha = match fmt {
        PixelFormat::Rgb => false,
        PixelFormat::Rgba => true,
        _ => {
            return Err(Error::NotImplemented(
                "Unsupported image format".to_string(),
            ));
        }
    };

    let options = DecoderOptions::default()
        .png_set_add_alpha_channel(alpha)
        .png_set_decode_animated(false);
    let mut decoder = PngDecoder::new_with_options(image, options);
    decoder.decode_headers()?;
    let image_info = decoder.get_info().ok_or(Error::Internal(
        "PNG did not return decoded image info".to_string(),
    ))?;

    let (rotation, flip) = image_info
        .exif
        .as_ref()
        .map(|x| read_exif_orientation(x))
        .unwrap_or((Rotation::None, Flip::None));

    if (rotation, flip) == (Rotation::None, Flip::None) {
        let img = Tensor::<u8>::image(image_info.width, image_info.height, fmt, memory)?;
        decoder.decode_into(&mut img.map()?)?;
        return Ok(TensorDyn::from(img));
    }

    let tmp = Tensor::<u8>::image(
        image_info.width,
        image_info.height,
        fmt,
        Some(TensorMemory::Mem),
    )?;
    decoder.decode_into(&mut tmp.map()?)?;

    rotate_flip_to_dyn(&tmp, fmt, rotation, flip, memory)
}

/// Load an image from raw bytes (JPEG or PNG) and return a [`TensorDyn`].
///
/// The optional `format` specifies the desired output pixel format (e.g.,
/// [`PixelFormat::Rgb`], [`PixelFormat::Rgba`]); if `None`, the native
/// format of the file is used (typically RGB for JPEG).
///
/// # Examples
/// ```rust
/// use edgefirst_image::load_image;
/// use edgefirst_tensor::PixelFormat;
/// # fn main() -> Result<(), edgefirst_image::Error> {
/// let jpeg = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
/// let img = load_image(jpeg, Some(PixelFormat::Rgb), None)?;
/// assert_eq!(img.width(), Some(1280));
/// assert_eq!(img.height(), Some(720));
/// # Ok(())
/// # }
/// ```
pub fn load_image(
    image: &[u8],
    format: Option<PixelFormat>,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    if let Ok(i) = load_jpeg(image, format, memory) {
        return Ok(i);
    }
    if let Ok(i) = load_png(image, format, memory) {
        return Ok(i);
    }
    Err(Error::NotSupported(
        "Could not decode as jpeg or png".to_string(),
    ))
}

/// Save a [`TensorDyn`] image as a JPEG file.
///
/// Only packed RGB and RGBA formats are supported.
pub fn save_jpeg(tensor: &TensorDyn, path: impl AsRef<std::path::Path>, quality: u8) -> Result<()> {
    let t = tensor.as_u8().ok_or(Error::UnsupportedFormat(
        "save_jpeg requires u8 tensor".to_string(),
    ))?;
    let fmt = t.format().ok_or(Error::NotAnImage)?;
    if fmt.layout() != PixelLayout::Packed {
        return Err(Error::NotImplemented(
            "Saving planar images is not supported".to_string(),
        ));
    }

    let colour = match fmt {
        PixelFormat::Rgb => jpeg_encoder::ColorType::Rgb,
        PixelFormat::Rgba => jpeg_encoder::ColorType::Rgba,
        _ => {
            return Err(Error::NotImplemented(
                "Unsupported image format for saving".to_string(),
            ));
        }
    };

    let w = t.width().ok_or(Error::NotAnImage)?;
    let h = t.height().ok_or(Error::NotAnImage)?;
    let encoder = jpeg_encoder::Encoder::new_file(path, quality)?;
    let tensor_map = t.map()?;

    encoder.encode(&tensor_map, w as u16, h as u16, colour)?;

    Ok(())
}

pub(crate) struct FunctionTimer<T: Display> {
    name: T,
    start: std::time::Instant,
}

impl<T: Display> FunctionTimer<T> {
    pub fn new(name: T) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl<T: Display> Drop for FunctionTimer<T> {
    fn drop(&mut self) {
        log::trace!("{} elapsed: {:?}", self.name, self.start.elapsed())
    }
}

const DEFAULT_COLORS: [[f32; 4]; 20] = [
    [0., 1., 0., 0.7],
    [1., 0.5568628, 0., 0.7],
    [0.25882353, 0.15294118, 0.13333333, 0.7],
    [0.8, 0.7647059, 0.78039216, 0.7],
    [0.3137255, 0.3137255, 0.3137255, 0.7],
    [0.1411765, 0.3098039, 0.1215686, 0.7],
    [1., 0.95686275, 0.5137255, 0.7],
    [0.3529412, 0.32156863, 0., 0.7],
    [0.4235294, 0.6235294, 0.6509804, 0.7],
    [0.5098039, 0.5098039, 0.7294118, 0.7],
    [0.00784314, 0.18823529, 0.29411765, 0.7],
    [0.0, 0.2706, 1.0, 0.7],
    [0.0, 0.0, 0.0, 0.7],
    [0.0, 0.5, 0.0, 0.7],
    [1.0, 0.0, 0.0, 0.7],
    [0.0, 0.0, 1.0, 0.7],
    [1.0, 0.5, 0.5, 0.7],
    [0.1333, 0.5451, 0.1333, 0.7],
    [0.1176, 0.4118, 0.8235, 0.7],
    [1., 1., 1., 0.7],
];

const fn denorm<const M: usize, const N: usize>(a: [[f32; M]; N]) -> [[u8; M]; N] {
    let mut result = [[0; M]; N];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < M {
            result[i][j] = (a[i][j] * 255.0).round() as u8;
            j += 1;
        }
        i += 1;
    }
    result
}

const DEFAULT_COLORS_U8: [[u8; 4]; 20] = denorm(DEFAULT_COLORS);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod alignment_tests {
    use super::*;

    #[test]
    fn align_width_rgba8_common_widths() {
        // RGBA8 (bpp=4, lcm(64,4)=64, so width must round to multiple of 16 px).
        assert_eq!(align_width_for_gpu_pitch(640, 4), 640); // 2560 byte pitch — already aligned
        assert_eq!(align_width_for_gpu_pitch(1280, 4), 1280); // 5120
        assert_eq!(align_width_for_gpu_pitch(1920, 4), 1920); // 7680
        assert_eq!(align_width_for_gpu_pitch(3840, 4), 3840); // 15360
                                                              // crowd.png case from the imx95 investigation:
        assert_eq!(align_width_for_gpu_pitch(3004, 4), 3008); // 12016 → 12032
        assert_eq!(align_width_for_gpu_pitch(3000, 4), 3008); // 12000 → 12032
        assert_eq!(align_width_for_gpu_pitch(17, 4), 32); // 68 → 128
        assert_eq!(align_width_for_gpu_pitch(1, 4), 16); // 4 → 64
    }

    #[test]
    fn align_width_rgb888_packed() {
        // RGB888 (bpp=3, lcm(64,3)=192, so width must round to multiple of 64 px).
        assert_eq!(align_width_for_gpu_pitch(64, 3), 64); // 192 byte pitch
        assert_eq!(align_width_for_gpu_pitch(640, 3), 640); // 1920
        assert_eq!(align_width_for_gpu_pitch(1, 3), 64); // 3 → 192
        assert_eq!(align_width_for_gpu_pitch(65, 3), 128); // 195 → 384
                                                           // Verify the rounded width × bpp is a clean multiple of the LCM.
        for w in [3004usize, 1281, 100, 17] {
            let padded = align_width_for_gpu_pitch(w, 3);
            assert!(padded >= w);
            assert_eq!((padded * 3) % 64, 0);
            assert_eq!((padded * 3) % 3, 0);
        }
    }

    #[test]
    fn align_width_grey_u8() {
        // Grey (bpp=1, lcm(64,1)=64, so width must round to multiple of 64 px).
        assert_eq!(align_width_for_gpu_pitch(64, 1), 64);
        assert_eq!(align_width_for_gpu_pitch(640, 1), 640);
        assert_eq!(align_width_for_gpu_pitch(1, 1), 64);
        assert_eq!(align_width_for_gpu_pitch(65, 1), 128);
    }

    #[test]
    fn align_width_zero_inputs() {
        assert_eq!(align_width_for_gpu_pitch(0, 4), 0);
        assert_eq!(align_width_for_gpu_pitch(640, 0), 640);
    }

    #[test]
    fn align_width_never_returns_smaller_than_input() {
        // Spot-check the "returned width >= input width" contract across a
        // range of values that would previously have hit `width * bpp`
        // overflow paths.
        for &bpp in &[1usize, 2, 3, 4, 8] {
            for &w in &[
                1usize,
                17,
                64,
                65,
                100,
                1280,
                1281,
                1920,
                3004,
                3072,
                3840,
                usize::MAX / 8,
                usize::MAX / 4,
                usize::MAX / 2,
                usize::MAX - 1,
                usize::MAX,
            ] {
                let aligned = align_width_for_gpu_pitch(w, bpp);
                assert!(
                    aligned >= w,
                    "align_width_for_gpu_pitch({w}, {bpp}) = {aligned} < {w}"
                );
            }
        }
    }

    #[test]
    fn align_width_overflow_returns_unaligned_not_smaller() {
        // For width values close to usize::MAX, padding up would wrap. The
        // function must return the original width rather than wrapping or
        // panicking. A pre-aligned width round-trips unchanged even at the
        // extreme.
        let aligned_extreme = usize::MAX - 15; // 16-pixel boundary for RGBA8
        assert_eq!(
            align_width_for_gpu_pitch(aligned_extreme, 4),
            aligned_extreme
        );
        // A misaligned extreme value cannot be rounded up — the function
        // returns the original.
        let misaligned_extreme = usize::MAX - 1;
        let result = align_width_for_gpu_pitch(misaligned_extreme, 4);
        assert!(
            result == misaligned_extreme || result >= misaligned_extreme,
            "extreme misaligned width must not be rounded down to {result}"
        );
    }

    #[test]
    fn checked_lcm_basic_and_overflow() {
        assert_eq!(checked_num_integer_lcm(64, 4), Some(64));
        assert_eq!(checked_num_integer_lcm(64, 3), Some(192));
        assert_eq!(checked_num_integer_lcm(64, 1), Some(64));
        assert_eq!(checked_num_integer_lcm(0, 4), Some(0));
        assert_eq!(checked_num_integer_lcm(64, 0), Some(0));
        // Coprime values whose product exceeds usize::MAX must return None.
        assert_eq!(
            checked_num_integer_lcm(usize::MAX, usize::MAX - 1),
            None,
            "coprime extreme values must overflow detect, not panic"
        );
    }

    #[test]
    fn primary_plane_bpp_known_formats() {
        // Packed formats use channels × elem_size.
        assert_eq!(primary_plane_bpp(PixelFormat::Rgba, 1), Some(4));
        assert_eq!(primary_plane_bpp(PixelFormat::Bgra, 1), Some(4));
        assert_eq!(primary_plane_bpp(PixelFormat::Rgb, 1), Some(3));
        assert_eq!(primary_plane_bpp(PixelFormat::Grey, 1), Some(1));
        // Semi-planar (NV12) reports the luma plane's bpp.
        assert_eq!(primary_plane_bpp(PixelFormat::Nv12, 1), Some(1));
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod image_tests {
    use super::*;
    use crate::{CPUProcessor, Rotation};
    #[cfg(target_os = "linux")]
    use edgefirst_tensor::is_dma_available;
    use edgefirst_tensor::{TensorMapTrait, TensorMemory, TensorTrait};
    use image::buffer::ConvertBuffer;

    /// Test helper: call `ImageProcessorTrait::convert()` on two `TensorDyn`s
    /// by going through the `TensorDyn` API.
    ///
    /// Returns the `(src_image, dst_image)` reconstructed from the TensorDyn
    /// round-trip so the caller can feed them to `compare_images` etc.
    fn convert_img(
        proc: &mut dyn ImageProcessorTrait,
        src: TensorDyn,
        dst: TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> (Result<()>, TensorDyn, TensorDyn) {
        let src_fourcc = src.format().unwrap();
        let dst_fourcc = dst.format().unwrap();
        let src_dyn = src;
        let mut dst_dyn = dst;
        let result = proc.convert(&src_dyn, &mut dst_dyn, rotation, flip, crop);
        let src_back = {
            let mut __t = src_dyn.into_u8().unwrap();
            __t.set_format(src_fourcc).unwrap();
            TensorDyn::from(__t)
        };
        let dst_back = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(dst_fourcc).unwrap();
            TensorDyn::from(__t)
        };
        (result, src_back, dst_back)
    }

    #[ctor::ctor]
    fn init() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    macro_rules! function {
        () => {{
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);

            // Find and cut the rest of the path
            match &name[..name.len() - 3].rfind(':') {
                Some(pos) => &name[pos + 1..name.len() - 3],
                None => &name[..name.len() - 3],
            }
        }};
    }

    #[test]
    fn test_invalid_crop() {
        let src = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        let dst = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();

        let crop = Crop::new()
            .with_src_rect(Some(Rect::new(50, 50, 60, 60)))
            .with_dst_rect(Some(Rect::new(0, 0, 150, 150)));

        let result = crop.check_crop_dyn(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Dest and Src crop invalid")
        ));

        let crop = crop.with_src_rect(Some(Rect::new(0, 0, 10, 10)));
        let result = crop.check_crop_dyn(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Dest crop invalid")
        ));

        let crop = crop
            .with_src_rect(Some(Rect::new(50, 50, 60, 60)))
            .with_dst_rect(Some(Rect::new(0, 0, 50, 50)));
        let result = crop.check_crop_dyn(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Src crop invalid")
        ));

        let crop = crop.with_src_rect(Some(Rect::new(50, 50, 50, 50)));

        let result = crop.check_crop_dyn(&src, &dst);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_tensor_format() -> Result<(), Error> {
        // 4D tensor cannot be set to a 3-channel pixel format
        let mut tensor = Tensor::<u8>::new(&[720, 1280, 4, 1], None, None)?;
        let result = tensor.set_format(PixelFormat::Rgb);
        assert!(result.is_err(), "4D tensor should reject set_format");

        // Tensor with wrong channel count for the format
        let mut tensor = Tensor::<u8>::new(&[720, 1280, 4], None, None)?;
        let result = tensor.set_format(PixelFormat::Rgb);
        assert!(result.is_err(), "4-channel tensor should reject RGB format");

        Ok(())
    }

    #[test]
    fn test_invalid_image_file() -> Result<(), Error> {
        let result = crate::load_image(&[123; 5000], None, None);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e == "Could not decode as jpeg or png"));

        Ok(())
    }

    #[test]
    fn test_invalid_jpeg_format() -> Result<(), Error> {
        let result = crate::load_image(&[123; 5000], Some(PixelFormat::Yuyv), None);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e == "Could not decode as jpeg or png"));

        Ok(())
    }

    #[test]
    fn test_load_resize_save() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ));
        let img = crate::load_image(file, Some(PixelFormat::Rgba), None).unwrap();
        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));

        let dst = TensorDyn::image(640, 360, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut converter = CPUProcessor::new();
        let (result, _img, dst) = convert_img(
            &mut converter,
            img,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();
        assert_eq!(dst.width(), Some(640));
        assert_eq!(dst.height(), Some(360));

        crate::save_jpeg(&dst, "zidane_resized.jpg", 80).unwrap();

        let file = std::fs::read("zidane_resized.jpg").unwrap();
        let img = crate::load_image(&file, None, None).unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(360));
        assert_eq!(img.format().unwrap(), PixelFormat::Rgb);
    }

    #[test]
    fn test_from_tensor_planar() -> Result<(), Error> {
        let mut tensor = Tensor::new(&[3, 720, 1280], None, None)?;
        tensor.map()?.copy_from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.8bps"
        )));
        let planar = {
            tensor
                .set_format(PixelFormat::PlanarRgb)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(tensor)
        };

        let rbga = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )?;
        compare_images_convert_to_rgb(&planar, &rbga, 0.98, function!());

        Ok(())
    }

    #[test]
    fn test_from_tensor_invalid_format() {
        // PixelFormat::from_fourcc_str returns None for unknown FourCC codes.
        // Since there's no "TEST" pixel format, this validates graceful handling.
        assert!(PixelFormat::from_fourcc(u32::from_le_bytes(*b"TEST")).is_none());
    }

    #[test]
    #[should_panic(expected = "Failed to save planar RGB image")]
    fn test_save_planar() {
        let planar_img = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::PlanarRgb,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.8bps"
            )),
        )
        .unwrap();

        let save_path = "/tmp/planar_rgb.jpg";
        crate::save_jpeg(&planar_img, save_path, 90).expect("Failed to save planar RGB image");
    }

    #[test]
    #[should_panic(expected = "Failed to save YUYV image")]
    fn test_save_yuyv() {
        let planar_img = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let save_path = "/tmp/yuyv.jpg";
        crate::save_jpeg(&planar_img, save_path, 90).expect("Failed to save YUYV image");
    }

    #[test]
    fn test_rotation_angle() {
        assert_eq!(Rotation::from_degrees_clockwise(0), Rotation::None);
        assert_eq!(Rotation::from_degrees_clockwise(90), Rotation::Clockwise90);
        assert_eq!(Rotation::from_degrees_clockwise(180), Rotation::Rotate180);
        assert_eq!(
            Rotation::from_degrees_clockwise(270),
            Rotation::CounterClockwise90
        );
        assert_eq!(Rotation::from_degrees_clockwise(360), Rotation::None);
        assert_eq!(Rotation::from_degrees_clockwise(450), Rotation::Clockwise90);
        assert_eq!(Rotation::from_degrees_clockwise(540), Rotation::Rotate180);
        assert_eq!(
            Rotation::from_degrees_clockwise(630),
            Rotation::CounterClockwise90
        );
    }

    #[test]
    #[should_panic(expected = "rotation angle is not a multiple of 90")]
    fn test_rotation_angle_panic() {
        Rotation::from_degrees_clockwise(361);
    }

    #[test]
    fn test_disable_env_var() -> Result<(), Error> {
        // EDGEFIRST_FORCE_BACKEND takes precedence over EDGEFIRST_DISABLE_*,
        // so clear it for the duration of this test to avoid races with
        // test_force_backend_cpu running in parallel.
        let saved_force = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };

        #[cfg(target_os = "linux")]
        {
            let original = std::env::var("EDGEFIRST_DISABLE_G2D").ok();
            unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", "1") };
            let converter = ImageProcessor::new()?;
            match original {
                Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", s) },
                None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_G2D") },
            }
            assert!(converter.g2d.is_none());
        }

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        {
            let original = std::env::var("EDGEFIRST_DISABLE_GL").ok();
            unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
            let converter = ImageProcessor::new()?;
            match original {
                Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", s) },
                None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_GL") },
            }
            assert!(converter.opengl.is_none());
        }

        let original = std::env::var("EDGEFIRST_DISABLE_CPU").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        let converter = ImageProcessor::new()?;
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_CPU") },
        }
        assert!(converter.cpu.is_none());

        let original_cpu = std::env::var("EDGEFIRST_DISABLE_CPU").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        let original_gl = std::env::var("EDGEFIRST_DISABLE_GL").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
        let original_g2d = std::env::var("EDGEFIRST_DISABLE_G2D").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", "1") };
        let mut converter = ImageProcessor::new()?;

        let src = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        let dst = TensorDyn::image(640, 360, PixelFormat::Rgba, DType::U8, None)?;
        let (result, _src, _dst) = convert_img(
            &mut converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(matches!(result, Err(Error::NoConverter)));

        match original_cpu {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_CPU") },
        }
        match original_gl {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_GL") },
        }
        match original_g2d {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_G2D") },
        }
        match saved_force {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }

        Ok(())
    }

    #[test]
    fn test_unsupported_conversion() {
        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        let dst = TensorDyn::image(640, 360, PixelFormat::Nv12, DType::U8, None).unwrap();
        let mut converter = ImageProcessor::new().unwrap();
        let (result, _src, _dst) = convert_img(
            &mut converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        log::debug!("result: {:?}", result);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e.starts_with("Conversion from NV12 to NV12")
        ));
    }

    #[test]
    fn test_load_grey() {
        let grey_img = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        let grey_but_rgb_img = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey-rgb.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        compare_images(&grey_img, &grey_but_rgb_img, 0.99, function!());
    }

    #[test]
    fn test_new_nv12() {
        let nv12 = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        assert_eq!(nv12.height(), Some(720));
        assert_eq!(nv12.width(), Some(1280));
        assert_eq!(nv12.format().unwrap(), PixelFormat::Nv12);
        // PixelFormat::Nv12.channels() returns 1 (luma plane channel count)
        assert_eq!(nv12.format().unwrap().channels(), 1);
        assert!(nv12.format().is_some_and(
            |f| f.layout() == PixelLayout::Planar || f.layout() == PixelLayout::SemiPlanar
        ))
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_new_image_converter() {
        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mut converter = ImageProcessor::new().unwrap();
        let converter_dst = converter
            .create_image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        let (result, src, converter_dst) = convert_img(
            &mut converter,
            src,
            converter_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&converter_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_create_image_dtype_i8() {
        let mut converter = ImageProcessor::new().unwrap();

        // I8 image should allocate successfully via create_image
        let dst = converter
            .create_image(320, 240, PixelFormat::Rgb, DType::I8, None)
            .unwrap();
        assert_eq!(dst.dtype(), DType::I8);
        assert!(dst.width() == Some(320));
        assert!(dst.height() == Some(240));
        assert_eq!(dst.format(), Some(PixelFormat::Rgb));

        // U8 for comparison
        let dst_u8 = converter
            .create_image(320, 240, PixelFormat::Rgb, DType::U8, None)
            .unwrap();
        assert_eq!(dst_u8.dtype(), DType::U8);

        // Convert into I8 dst should succeed
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();
        let mut dst_i8 = converter
            .create_image(320, 240, PixelFormat::Rgb, DType::I8, None)
            .unwrap();
        converter
            .convert(
                &src,
                &mut dst_i8,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_create_image_nv12_dma_non_aligned_width() {
        // Regression for C2: create_image must not apply stride padding to
        // non-packed formats. NV12 is semi-planar (PixelLayout::SemiPlanar),
        // so the try_dma path should fall through to the plain
        // TensorDyn::image allocation for any width, regardless of the
        // 64-byte GPU pitch alignment.
        let converter = ImageProcessor::new().unwrap();

        // 100 is intentionally not a multiple of 64 (the Mali pitch
        // alignment) to prove that non-packed layouts do not take the
        // stride-padded branch.
        let result = converter.create_image(
            100,
            64,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Dma),
        );

        match result {
            Ok(img) => {
                assert_eq!(img.width(), Some(100));
                assert_eq!(img.height(), Some(64));
                assert_eq!(img.format(), Some(PixelFormat::Nv12));
                // Non-packed formats must never carry a row_stride override.
                assert!(
                    img.row_stride().is_none(),
                    "NV12 must not be stride-padded by create_image",
                );
            }
            Err(e) => {
                // Accept skip on hosts without a dma-heap, but never the
                // "NotImplemented" we used to return for non-packed layouts.
                let msg = format!("{e}");
                assert!(
                    !msg.contains("image_with_stride"),
                    "NV12 should not hit the stride-padded path: {msg}",
                );
            }
        }
    }

    #[test]
    #[ignore] // Hangs on desktop platforms where DMA-buf is unavailable and PBO
              // fallback triggers a GPU driver hang during SHM→texture upload (e.g.,
              // NVIDIA without /dev/dma_heap permissions). Works on embedded targets.
    fn test_crop_skip() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mut converter = ImageProcessor::new().unwrap();
        let converter_dst = converter
            .create_image(1280, 720, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        let crop = Crop::new()
            .with_src_rect(Some(Rect::new(0, 0, 640, 640)))
            .with_dst_rect(Some(Rect::new(0, 0, 640, 640)));
        let (result, src, converter_dst) = convert_img(
            &mut converter,
            src,
            converter_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images(&converter_dst, &cpu_dst, 0.99999, function!());
    }

    #[test]
    fn test_invalid_pixel_format() {
        // PixelFormat::from_fourcc returns None for unknown formats,
        // so TensorDyn::image cannot be called with an invalid format.
        assert!(PixelFormat::from_fourcc(u32::from_le_bytes(*b"TEST")).is_none());
    }

    // Helper function to check if G2D library is available (Linux/i.MX8 only)
    #[cfg(target_os = "linux")]
    static G2D_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(target_os = "linux")]
    fn is_g2d_available() -> bool {
        *G2D_AVAILABLE.get_or_init(|| G2DProcessor::new().is_ok())
    }

    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    // Helper function to check if OpenGL is available
    fn is_opengl_available() -> bool {
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        {
            *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
        }

        #[cfg(not(all(target_os = "linux", feature = "opengl")))]
        {
            false
        }
    }

    #[test]
    fn test_load_jpeg_with_exif() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane_rotated_exif.jpg"
        ))
        .to_vec();
        let loaded = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        assert_eq!(loaded.height(), Some(1280));
        assert_eq!(loaded.width(), Some(720));

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let cpu_src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let (dst_width, dst_height) = (cpu_src.height().unwrap(), cpu_src.width().unwrap());

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _cpu_src, cpu_dst) = convert_img(
            &mut cpu_converter,
            cpu_src,
            cpu_dst,
            Rotation::Clockwise90,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&loaded, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_load_png_with_exif() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane_rotated_exif_180.png"
        ))
        .to_vec();
        let loaded = crate::load_png(&file, Some(PixelFormat::Rgba), None).unwrap();

        assert_eq!(loaded.height(), Some(720));
        assert_eq!(loaded.width(), Some(1280));

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let cpu_src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _cpu_src, cpu_dst) = convert_img(
            &mut cpu_converter,
            cpu_src,
            cpu_dst,
            Rotation::Rotate180,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&loaded, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_resize() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_g2d_resize - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_g2d_resize - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src =
            crate::load_image(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma)).unwrap();

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_resize() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let mut src = src;
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        for _ in 0..5 {
            let gl_dst =
                TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None)
                    .unwrap();
            let (result, src_back, gl_dst) = convert_img(
                &mut gl_converter,
                src,
                gl_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            );
            result.unwrap();
            src = src_back;

            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_10_threads() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let handles: Vec<_> = (0..10)
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("Thread {i}"))
                    .spawn(test_opengl_resize)
                    .unwrap()
            })
            .collect();
        handles.into_iter().for_each(|h| {
            if let Err(e) = h.join() {
                std::panic::resume_unwind(e)
            }
        });
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_grey() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let img = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey.jpg"
            )),
            Some(PixelFormat::Grey),
            None,
        )
        .unwrap();

        let gl_dst = TensorDyn::image(640, 640, PixelFormat::Grey, DType::U8, None).unwrap();
        let cpu_dst = TensorDyn::image(640, 640, PixelFormat::Grey, DType::U8, None).unwrap();

        let mut converter = CPUProcessor::new();

        let (result, img, cpu_dst) = convert_img(
            &mut converter,
            img,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let (result, _img, gl_dst) = convert_img(
            &mut gl,
            img,
            gl_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&gl_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_src_crop() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_g2d_src_crop - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_g2d_src_crop - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop {
            src_rect: Some(Rect {
                left: 0,
                top: 0,
                width: 640,
                height: 360,
            }),
            dst_rect: None,
            dst_color: None,
        };
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let g2d_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        let (result, _src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_dst_crop() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_g2d_dst_crop - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_g2d_dst_crop - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop {
            src_rect: None,
            dst_rect: Some(Rect::new(100, 100, 512, 288)),
            dst_color: None,
        };
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let g2d_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        let (result, _src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_all_rgba() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_g2d_all_rgba - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_g2d_all_rgba - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();
        let src_dyn = src;

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let mut g2d_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let crop = Crop {
            src_rect: Some(Rect::new(50, 120, 1024, 576)),
            dst_rect: Some(Rect::new(100, 100, 512, 288)),
            dst_color: None,
        };

        for rot in [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            cpu_dst
                .as_u8()
                .unwrap()
                .map()
                .unwrap()
                .as_mut_slice()
                .fill(114);
            g2d_dst
                .as_u8()
                .unwrap()
                .map()
                .unwrap()
                .as_mut_slice()
                .fill(114);
            for flip in [Flip::None, Flip::Horizontal, Flip::Vertical] {
                let mut cpu_dst_dyn = cpu_dst;
                cpu_converter
                    .convert(&src_dyn, &mut cpu_dst_dyn, Rotation::None, Flip::None, crop)
                    .unwrap();
                cpu_dst = {
                    let mut __t = cpu_dst_dyn.into_u8().unwrap();
                    __t.set_format(PixelFormat::Rgba).unwrap();
                    TensorDyn::from(__t)
                };

                let mut g2d_dst_dyn = g2d_dst;
                g2d_converter
                    .convert(&src_dyn, &mut g2d_dst_dyn, Rotation::None, Flip::None, crop)
                    .unwrap();
                g2d_dst = {
                    let mut __t = g2d_dst_dyn.into_u8().unwrap();
                    __t.set_format(PixelFormat::Rgba).unwrap();
                    TensorDyn::from(__t)
                };

                compare_images(
                    &g2d_dst,
                    &cpu_dst,
                    0.98,
                    &format!("{} {:?} {:?}", function!(), rot, flip),
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_src_crop() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();
        let crop = Crop {
            src_rect: Some(Rect {
                left: 320,
                top: 180,
                width: 1280 - 320,
                height: 720 - 180,
            }),
            dst_rect: None,
            dst_color: None,
        };

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let gl_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();
        let (result, _src, gl_dst) = convert_img(
            &mut gl_converter,
            src,
            gl_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images(&gl_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_dst_crop() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop {
            src_rect: None,
            dst_rect: Some(Rect::new(100, 100, 512, 288)),
            dst_color: None,
        };
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let gl_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();
        let (result, _src, gl_dst) = convert_img(
            &mut gl_converter,
            src,
            gl_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images(&gl_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_all_rgba() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();

        let mut cpu_converter = CPUProcessor::new();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let mut mem = vec![None, Some(TensorMemory::Mem), Some(TensorMemory::Shm)];
        if is_dma_available() {
            mem.push(Some(TensorMemory::Dma));
        }
        let crop = Crop {
            src_rect: Some(Rect::new(50, 120, 1024, 576)),
            dst_rect: Some(Rect::new(100, 100, 512, 288)),
            dst_color: None,
        };
        for m in mem {
            let src = crate::load_image(&file, Some(PixelFormat::Rgba), m).unwrap();
            let src_dyn = src;

            for rot in [
                Rotation::None,
                Rotation::Clockwise90,
                Rotation::Rotate180,
                Rotation::CounterClockwise90,
            ] {
                for flip in [Flip::None, Flip::Horizontal, Flip::Vertical] {
                    let cpu_dst =
                        TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, m)
                            .unwrap();
                    let gl_dst =
                        TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, m)
                            .unwrap();
                    cpu_dst
                        .as_u8()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_mut_slice()
                        .fill(114);
                    gl_dst
                        .as_u8()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_mut_slice()
                        .fill(114);

                    let mut cpu_dst_dyn = cpu_dst;
                    cpu_converter
                        .convert(&src_dyn, &mut cpu_dst_dyn, Rotation::None, Flip::None, crop)
                        .unwrap();
                    let cpu_dst = {
                        let mut __t = cpu_dst_dyn.into_u8().unwrap();
                        __t.set_format(PixelFormat::Rgba).unwrap();
                        TensorDyn::from(__t)
                    };

                    let mut gl_dst_dyn = gl_dst;
                    gl_converter
                        .convert(&src_dyn, &mut gl_dst_dyn, Rotation::None, Flip::None, crop)
                        .map_err(|e| {
                            log::error!("error mem {m:?} rot {rot:?} error: {e:?}");
                            e
                        })
                        .unwrap();
                    let gl_dst = {
                        let mut __t = gl_dst_dyn.into_u8().unwrap();
                        __t.set_format(PixelFormat::Rgba).unwrap();
                        TensorDyn::from(__t)
                    };

                    compare_images(
                        &gl_dst,
                        &cpu_dst,
                        0.98,
                        &format!("{} {:?} {:?}", function!(), rot, flip),
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_cpu_rotate() {
        for rot in [
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            test_cpu_rotate_(rot);
        }
    }

    #[cfg(target_os = "linux")]
    fn test_cpu_rotate_(rot: Rotation) {
        // This test rotates the image 4 times and checks that the image was returned to
        // be the same Currently doesn't check if rotations actually rotated in
        // right direction
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();

        let unchanged_src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => (src.width().unwrap(), src.height().unwrap()),
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
                (src.height().unwrap(), src.width().unwrap())
            }
        };

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        // After rotating 4 times, the image should be the same as the original

        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let (result, cpu_dst, src) = convert_img(
            &mut cpu_converter,
            cpu_dst,
            src,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let (result, _cpu_dst, src) = convert_img(
            &mut cpu_converter,
            cpu_dst,
            src,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&src, &unchanged_src, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_rotate() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let size = (1280, 720);
        let mut mem = vec![None, Some(TensorMemory::Shm), Some(TensorMemory::Mem)];

        if is_dma_available() {
            mem.push(Some(TensorMemory::Dma));
        }
        for m in mem {
            for rot in [
                Rotation::Clockwise90,
                Rotation::Rotate180,
                Rotation::CounterClockwise90,
            ] {
                test_opengl_rotate_(size, rot, m);
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_rotate_(
        size: (usize, usize),
        rot: Rotation,
        tensor_memory: Option<TensorMemory>,
    ) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (size.1, size.0),
        };

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), tensor_memory).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, mut src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        for _ in 0..5 {
            let gl_dst = TensorDyn::image(
                dst_width,
                dst_height,
                PixelFormat::Rgba,
                DType::U8,
                tensor_memory,
            )
            .unwrap();
            let (result, src_back, gl_dst) = convert_img(
                &mut gl_converter,
                src,
                gl_dst,
                rot,
                Flip::None,
                Crop::no_crop(),
            );
            result.unwrap();
            src = src_back;
            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_rotate() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_g2d_rotate - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_g2d_rotate - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let size = (1280, 720);
        for rot in [
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            test_g2d_rotate_(size, rot);
        }
    }

    #[cfg(target_os = "linux")]
    fn test_g2d_rotate_(size: (usize, usize), rot: Rotation) {
        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => size,
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (size.1, size.0),
        };

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src =
            crate::load_image(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma)).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, _src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            rot,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_rgba_to_yuyv_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Yuyv, DType::U8, None).unwrap();

        let dst_through_yuyv =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let dst_direct =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();

        let mut cpu_converter = CPUProcessor::new();

        let (result, src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let (result, _dst, dst_through_yuyv) = convert_img(
            &mut cpu_converter,
            dst,
            dst_through_yuyv,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let (result, _src, dst_direct) = convert_img(
            &mut cpu_converter,
            src,
            dst_direct,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&dst_through_yuyv, &dst_direct, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[ignore = "opengl doesn't support rendering to PixelFormat::Yuyv texture"]
    fn test_rgba_to_yuyv_resize_opengl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        if !is_dma_available() {
            eprintln!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let (result, src, dst) = convert_img(
            &mut gl_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::new()
                .with_dst_rect(Some(Rect::new(100, 100, 100, 100)))
                .with_dst_color(Some([255, 255, 255, 255])),
        );
        result.unwrap();

        std::fs::write(
            "rgba_to_yuyv_opengl.yuyv",
            dst.as_u8().unwrap().map().unwrap().as_slice(),
        )
        .unwrap();
        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let (result, _src, cpu_dst) = convert_img(
            &mut CPUProcessor::new(),
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images_convert_to_rgb(&dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_rgba_to_yuyv_resize_g2d() {
        if !is_g2d_available() {
            eprintln!(
                "SKIPPED: test_rgba_to_yuyv_resize_g2d - G2D library (libg2d.so.2) not available"
            );
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_rgba_to_yuyv_resize_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (1280, 720);

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();

        let mut g2d_converter = G2DProcessor::new().unwrap();
        let crop = Crop {
            src_rect: None,
            dst_rect: Some(Rect::new(100, 100, 2, 2)),
            dst_color: None,
        };

        g2d_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let cpu_dst_img = cpu_dst;
        cpu_dst_img
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        let (result, _src, cpu_dst) = convert_img(
            &mut CPUProcessor::new(),
            src,
            cpu_dst_img,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        compare_images_convert_to_rgb(&cpu_dst, &g2d_dst, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgba_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.yuyv"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Yuyv, DType::U8, None).unwrap();
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgb_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.yuyv"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Yuyv, DType::U8, None).unwrap();
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/camera720p.rgba"
                ))
                .as_chunks::<4>()
                .0,
            )
            .for_each(|(dst, src)| *dst = [src[0], src[1], src[2]]);

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_g2d() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_yuyv_to_rgba_g2d - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_rgba_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, _src, dst) = convert_img(
            &mut g2d_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let (result, _src, dst) = convert_img(
            &mut gl_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgb_g2d() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_yuyv_to_rgb_g2d - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_rgb_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_yuyv_resize_g2d() {
        if !is_g2d_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_yuyv_resize_g2d - G2D library (libg2d.so.2) not available"
            );
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_yuyv_resize_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            600,
            400,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let cpu_dst = TensorDyn::image(600, 400, PixelFormat::Yuyv, DType::U8, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        // TODO: compare PixelFormat::Yuyv and PixelFormat::Yuyv images without having to convert them to PixelFormat::Rgb
        compare_images_convert_to_rgb(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgba_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (960, 540);

        let dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let dst_target =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let src_target = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();
        let (result, _src_target, dst_target) = convert_img(
            &mut cpu_converter,
            src_target,
            dst_target,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&dst, &dst_target, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_crop_flip_g2d() {
        if !is_g2d_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_rgba_crop_flip_g2d - G2D library (libg2d.so.2) not available"
            );
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_yuyv_to_rgba_crop_flip_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let dst_g2d = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        let crop = Crop {
            src_rect: Some(Rect {
                left: 20,
                top: 15,
                width: 400,
                height: 300,
            }),
            dst_rect: None,
            dst_color: None,
        };

        let (result, src, dst_g2d) = convert_img(
            &mut g2d_converter,
            src,
            dst_g2d,
            Rotation::None,
            Flip::Horizontal,
            crop,
        );
        result.unwrap();

        let dst_cpu = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst_cpu) = convert_img(
            &mut cpu_converter,
            src,
            dst_cpu,
            Rotation::None,
            Flip::Horizontal,
            crop,
        );
        result.unwrap();
        compare_images(&dst_g2d, &dst_cpu, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_crop_flip_opengl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        if !is_dma_available() {
            eprintln!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let dst_gl = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();
        let crop = Crop {
            src_rect: Some(Rect {
                left: 20,
                top: 15,
                width: 400,
                height: 300,
            }),
            dst_rect: None,
            dst_color: None,
        };

        let (result, src, dst_gl) = convert_img(
            &mut gl_converter,
            src,
            dst_gl,
            Rotation::None,
            Flip::Horizontal,
            crop,
        );
        result.unwrap();

        let dst_cpu = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst_cpu) = convert_img(
            &mut cpu_converter,
            src,
            dst_cpu,
            Rotation::None,
            Flip::Horizontal,
            crop,
        );
        result.unwrap();
        compare_images(&dst_gl, &dst_cpu, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgba_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.vyuy"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Vyuy, DType::U8, None).unwrap();
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgb_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.vyuy"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Vyuy, DType::U8, None).unwrap();
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/camera720p.rgba"
                ))
                .as_chunks::<4>()
                .0,
            )
            .for_each(|(dst, src)| *dst = [src[0], src[1], src[2]]);

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "G2D does not support VYUY; re-enable when hardware support is added"]
    fn test_vyuy_to_rgba_g2d() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_vyuy_to_rgba_g2d - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_vyuy_to_rgba_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Vyuy,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, _src, dst) = convert_img(
            &mut g2d_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        match result {
            Err(Error::G2D(_)) => {
                eprintln!("SKIPPED: test_vyuy_to_rgba_g2d - G2D does not support PixelFormat::Vyuy format");
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "G2D does not support VYUY; re-enable when hardware support is added"]
    fn test_vyuy_to_rgb_g2d() {
        if !is_g2d_available() {
            eprintln!("SKIPPED: test_vyuy_to_rgb_g2d - G2D library (libg2d.so.2) not available");
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_vyuy_to_rgb_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)"
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Vyuy,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        match result {
            Err(Error::G2D(_)) => {
                eprintln!(
                    "SKIPPED: test_vyuy_to_rgb_g2d - G2D does not support PixelFormat::Vyuy format"
                );
                return;
            }
            r => r.unwrap(),
        }

        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_vyuy_to_rgba_opengl() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Vyuy,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let (result, _src, dst) = convert_img(
            &mut gl_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        match result {
            Err(Error::NotSupported(_)) => {
                eprintln!(
                    "SKIPPED: {} - OpenGL does not support PixelFormat::Vyuy DMA format",
                    function!()
                );
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        target_image
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_rgba_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.nv12"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_rgb_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.nv12"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(PixelFormat::Rgb),
            None,
        )
        .unwrap();

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_grey_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.nv12"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Grey, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(PixelFormat::Grey),
            None,
        )
        .unwrap();

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_yuyv_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.nv12"
        ))
        .to_vec();
        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(1280, 720, PixelFormat::Yuyv, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, dst) = convert_img(
            &mut cpu_converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let target_image = crate::load_image(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(PixelFormat::Rgb),
            None,
        )
        .unwrap();

        compare_images_convert_to_rgb(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_cpu_resize_planar_rgb() {
        let src = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None).unwrap();
        #[rustfmt::skip]
        let src_image = [
                    255, 0, 0, 255,     0, 255, 0, 255,     0, 0, 255, 255,     255, 255, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
                    0, 0, 255, 0,       0, 255, 255, 255,   255, 255, 0, 0,     0, 0, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
        ];
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&src_image);

        let cpu_dst = TensorDyn::image(5, 5, PixelFormat::PlanarRgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::new()
                .with_dst_rect(Some(Rect {
                    left: 1,
                    top: 1,
                    width: 4,
                    height: 4,
                }))
                .with_dst_color(Some([114, 114, 114, 255])),
        );
        result.unwrap();

        #[rustfmt::skip]
        let expected_dst = [
            114, 114, 114, 114, 114,    114, 255, 0, 0, 255,    114, 255, 0, 255, 255,      114, 0, 0, 255, 0,        114, 255, 0, 255, 255,
            114, 114, 114, 114, 114,    114, 0, 255, 0, 255,    114, 0, 0, 0, 0,            114, 0, 255, 255, 0,      114, 0, 0, 0, 0,
            114, 114, 114, 114, 114,    114, 0, 0, 255, 0,      114, 0, 0, 255, 255,        114, 255, 255, 0, 0,      114, 0, 0, 255, 255,
        ];

        assert_eq!(
            cpu_dst.as_u8().unwrap().map().unwrap().as_slice(),
            &expected_dst
        );
    }

    #[test]
    fn test_cpu_resize_planar_rgba() {
        let src = TensorDyn::image(4, 4, PixelFormat::Rgba, DType::U8, None).unwrap();
        #[rustfmt::skip]
        let src_image = [
                    255, 0, 0, 255,     0, 255, 0, 255,     0, 0, 255, 255,     255, 255, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
                    0, 0, 255, 0,       0, 255, 255, 255,   255, 255, 0, 0,     0, 0, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
        ];
        src.as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&src_image);

        let cpu_dst = TensorDyn::image(5, 5, PixelFormat::PlanarRgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::new()
                .with_dst_rect(Some(Rect {
                    left: 1,
                    top: 1,
                    width: 4,
                    height: 4,
                }))
                .with_dst_color(Some([114, 114, 114, 255])),
        );
        result.unwrap();

        #[rustfmt::skip]
        let expected_dst = [
            114, 114, 114, 114, 114,    114, 255, 0, 0, 255,        114, 255, 0, 255, 255,      114, 0, 0, 255, 0,        114, 255, 0, 255, 255,
            114, 114, 114, 114, 114,    114, 0, 255, 0, 255,        114, 0, 0, 0, 0,            114, 0, 255, 255, 0,      114, 0, 0, 0, 0,
            114, 114, 114, 114, 114,    114, 0, 0, 255, 0,          114, 0, 0, 255, 255,        114, 255, 255, 0, 0,      114, 0, 0, 255, 255,
            255, 255, 255, 255, 255,    255, 255, 255, 255, 255,    255, 0, 255, 0, 255,        255, 0, 255, 0, 255,      255, 0, 255, 0, 255,
        ];

        assert_eq!(
            cpu_dst.as_u8().unwrap().map().unwrap().as_slice(),
            &expected_dst
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_opengl_resize_planar_rgb() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        if !is_dma_available() {
            eprintln!(
                "SKIPPED: {} - DMA memory allocation not available (permission denied or no DMA-BUF support)",
                function!()
            );
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/test_image.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
        )
        .unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();
        let crop_letterbox = Crop::new()
            .with_dst_rect(Some(Rect {
                left: 102,
                top: 102,
                width: 440,
                height: 440,
            }))
            .with_dst_color(Some([114, 114, 114, 114]));
        let (result, src, cpu_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            crop_letterbox,
        );
        result.unwrap();

        let gl_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
        )
        .unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let (result, _src, gl_dst) = convert_img(
            &mut gl_converter,
            src,
            gl_dst,
            Rotation::None,
            Flip::None,
            crop_letterbox,
        );
        result.unwrap();
        compare_images(&gl_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_cpu_resize_nv16() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_nv16_dst = TensorDyn::image(640, 640, PixelFormat::Nv16, DType::U8, None).unwrap();
        let cpu_rgb_dst = TensorDyn::image(640, 640, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::new()
            .with_dst_rect(Some(Rect {
                left: 20,
                top: 140,
                width: 600,
                height: 360,
            }))
            .with_dst_color(Some([255, 128, 0, 255]));

        let (result, src, cpu_nv16_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_nv16_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();

        let (result, _src, cpu_rgb_dst) = convert_img(
            &mut cpu_converter,
            src,
            cpu_rgb_dst,
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();
        compare_images_convert_to_rgb(&cpu_nv16_dst, &cpu_rgb_dst, 0.99, function!());
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, Error> {
        let src = TensorDyn::image(width, height, format, DType::U8, memory)?;
        src.as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(bytes);
        Ok(src)
    }

    fn compare_images(img1: &TensorDyn, img2: &TensorDyn, threshold: f64, name: &str) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(
            img1.format().unwrap(),
            img2.format().unwrap(),
            "PixelFormat differ"
        );
        assert!(
            matches!(
                img1.format().unwrap(),
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey | PixelFormat::PlanarRgb
            ),
            "format must be Rgb or Rgba for comparison"
        );

        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                (img1.height().unwrap() * 3) as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                (img2.height().unwrap() * 3) as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    fn compare_images_convert_to_rgb(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 = TensorDyn::image(
            img1.width().unwrap(),
            img1.height().unwrap(),
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let mut img_rgb2 = TensorDyn::image(
            img1.width().unwrap(),
            img1.height().unwrap(),
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let mut __cv = CPUProcessor::default();
        let r1 = __cv.convert(
            img1,
            &mut img_rgb1,
            crate::Rotation::None,
            crate::Flip::None,
            crate::Crop::default(),
        );
        let r2 = __cv.convert(
            img2,
            &mut img_rgb2,
            crate::Rotation::None,
            crate::Flip::None,
            crate::Crop::default(),
        );
        if r1.is_err() || r2.is_err() {
            // Fallback: compare raw bytes as greyscale strip
            let w = img1.width().unwrap() as u32;
            let data1 = img1.as_u8().unwrap().map().unwrap().to_vec();
            let data2 = img2.as_u8().unwrap().map().unwrap().to_vec();
            let h1 = (data1.len() as u32) / w;
            let h2 = (data2.len() as u32) / w;
            let g1 = image::GrayImage::from_vec(w, h1, data1).unwrap();
            let g2 = image::GrayImage::from_vec(w, h2, data2).unwrap();
            let similarity = image_compare::gray_similarity_structure(
                &image_compare::Algorithm::RootMeanSquared,
                &g1,
                &g2,
            )
            .expect("Image Comparison failed");
            if similarity.score < threshold {
                panic!(
                    "{name}: converted image and target image have similarity score too low: {} < {}",
                    similarity.score, threshold
                )
            }
            return;
        }

        let image1 = image::RgbImage::from_vec(
            img_rgb1.width().unwrap() as u32,
            img_rgb1.height().unwrap() as u32,
            img_rgb1.as_u8().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width().unwrap() as u32,
            img_rgb2.height().unwrap() as u32,
            img_rgb2.as_u8().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    // =========================================================================
    // PixelFormat::Nv12 Format Tests
    // =========================================================================

    #[test]
    fn test_nv12_image_creation() {
        let width = 640;
        let height = 480;
        let img = TensorDyn::image(width, height, PixelFormat::Nv12, DType::U8, None).unwrap();

        assert_eq!(img.width(), Some(width));
        assert_eq!(img.height(), Some(height));
        assert_eq!(img.format().unwrap(), PixelFormat::Nv12);
        // PixelFormat::Nv12 uses shape [H*3/2, W] to store Y plane + UV plane
        assert_eq!(img.as_u8().unwrap().shape(), &[height * 3 / 2, width]);
    }

    #[test]
    fn test_nv12_channels() {
        let img = TensorDyn::image(640, 480, PixelFormat::Nv12, DType::U8, None).unwrap();
        // PixelFormat::Nv12.channels() returns 1 (luma plane)
        assert_eq!(img.format().unwrap().channels(), 1);
    }

    // =========================================================================
    // Tensor Format Metadata Tests
    // =========================================================================

    #[test]
    fn test_tensor_set_format_planar() {
        let mut tensor = Tensor::<u8>::new(&[3, 480, 640], None, None).unwrap();
        tensor.set_format(PixelFormat::PlanarRgb).unwrap();
        assert_eq!(tensor.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(tensor.width(), Some(640));
        assert_eq!(tensor.height(), Some(480));
    }

    #[test]
    fn test_tensor_set_format_interleaved() {
        let mut tensor = Tensor::<u8>::new(&[480, 640, 4], None, None).unwrap();
        tensor.set_format(PixelFormat::Rgba).unwrap();
        assert_eq!(tensor.format(), Some(PixelFormat::Rgba));
        assert_eq!(tensor.width(), Some(640));
        assert_eq!(tensor.height(), Some(480));
    }

    #[test]
    fn test_tensordyn_image_rgb() {
        let img = TensorDyn::image(640, 480, PixelFormat::Rgb, DType::U8, None).unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
        assert_eq!(img.format(), Some(PixelFormat::Rgb));
    }

    #[test]
    fn test_tensordyn_image_planar_rgb() {
        let img = TensorDyn::image(640, 480, PixelFormat::PlanarRgb, DType::U8, None).unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
        assert_eq!(img.format(), Some(PixelFormat::PlanarRgb));
    }

    #[test]
    fn test_rgb_int8_format() {
        // Int8 variant: same PixelFormat::Rgb but with DType::I8
        let img = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));
        assert_eq!(img.format(), Some(PixelFormat::Rgb));
        assert_eq!(img.dtype(), DType::I8);
    }

    #[test]
    fn test_planar_rgb_int8_format() {
        let img = TensorDyn::image(
            1280,
            720,
            PixelFormat::PlanarRgb,
            DType::I8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));
        assert_eq!(img.format(), Some(PixelFormat::PlanarRgb));
        assert_eq!(img.dtype(), DType::I8);
    }

    #[test]
    fn test_rgb_from_tensor() {
        let mut tensor = Tensor::<u8>::new(&[720, 1280, 3], None, None).unwrap();
        tensor.set_format(PixelFormat::Rgb).unwrap();
        let img = TensorDyn::from(tensor);
        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));
        assert_eq!(img.format(), Some(PixelFormat::Rgb));
    }

    #[test]
    fn test_planar_rgb_from_tensor() {
        let mut tensor = Tensor::<u8>::new(&[3, 720, 1280], None, None).unwrap();
        tensor.set_format(PixelFormat::PlanarRgb).unwrap();
        let img = TensorDyn::from(tensor);
        assert_eq!(img.width(), Some(1280));
        assert_eq!(img.height(), Some(720));
        assert_eq!(img.format(), Some(PixelFormat::PlanarRgb));
    }

    #[test]
    fn test_dtype_determines_int8() {
        // DType::I8 indicates int8 data
        let u8_img = TensorDyn::image(64, 64, PixelFormat::Rgb, DType::U8, None).unwrap();
        let i8_img = TensorDyn::image(64, 64, PixelFormat::Rgb, DType::I8, None).unwrap();
        assert_eq!(u8_img.dtype(), DType::U8);
        assert_eq!(i8_img.dtype(), DType::I8);
    }

    #[test]
    fn test_pixel_layout_packed_vs_planar() {
        // Packed vs planar layout classification
        assert_eq!(PixelFormat::Rgb.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Rgba.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::PlanarRgb.layout(), PixelLayout::Planar);
        assert_eq!(PixelFormat::Nv12.layout(), PixelLayout::SemiPlanar);
    }

    /// Integration test that exercises the PBO-to-PBO convert path.
    /// Uses ImageProcessor::create_image() to allocate PBO-backed tensors,
    /// then converts between them. Skipped when GL is unavailable or the
    /// backend is not PBO (e.g. DMA-buf systems).
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[test]
    fn test_convert_pbo_to_pbo() {
        let mut converter = ImageProcessor::new().unwrap();

        // Skip if GL is not available or backend is not PBO
        let is_pbo = converter
            .opengl
            .as_ref()
            .is_some_and(|gl| gl.transfer_backend() == opengl_headless::TransferBackend::Pbo);
        if !is_pbo {
            eprintln!("Skipping test_convert_pbo_to_pbo: backend is not PBO");
            return;
        }

        let src_w = 640;
        let src_h = 480;
        let dst_w = 320;
        let dst_h = 240;

        // Create PBO-backed source image
        let pbo_src = converter
            .create_image(src_w, src_h, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        assert_eq!(
            pbo_src.as_u8().unwrap().memory(),
            TensorMemory::Pbo,
            "create_image should produce a PBO tensor"
        );

        // Fill source PBO with test pattern: load JPEG then convert Mem→PBO
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let jpeg_src = crate::load_image(&file, Some(PixelFormat::Rgba), None).unwrap();

        // Resize JPEG into a Mem temp of the right size, then copy into PBO
        let mem_src = TensorDyn::image(
            src_w,
            src_h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let (result, _jpeg_src, mem_src) = convert_img(
            &mut CPUProcessor::new(),
            jpeg_src,
            mem_src,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        // Copy pixel data into the PBO source by mapping it
        {
            let src_data = mem_src.as_u8().unwrap().map().unwrap();
            let mut pbo_map = pbo_src.as_u8().unwrap().map().unwrap();
            pbo_map.copy_from_slice(&src_data);
        }

        // Create PBO-backed destination image
        let pbo_dst = converter
            .create_image(dst_w, dst_h, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        assert_eq!(pbo_dst.as_u8().unwrap().memory(), TensorMemory::Pbo);

        // Convert PBO→PBO (this exercises convert_pbo_to_pbo)
        let mut pbo_dst = pbo_dst;
        let result = converter.convert(
            &pbo_src,
            &mut pbo_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        // Verify: compare with CPU-only conversion of the same input
        let cpu_dst = TensorDyn::image(
            dst_w,
            dst_h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let (result, _mem_src, cpu_dst) = convert_img(
            &mut CPUProcessor::new(),
            mem_src,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let pbo_dst_img = {
            let mut __t = pbo_dst.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba).unwrap();
            TensorDyn::from(__t)
        };
        compare_images(&pbo_dst_img, &cpu_dst, 0.95, function!());
        log::info!("test_convert_pbo_to_pbo: PASS — PBO-to-PBO convert matches CPU reference");
    }

    #[test]
    fn test_image_bgra() {
        let img = TensorDyn::image(
            640,
            480,
            PixelFormat::Bgra,
            DType::U8,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
        assert_eq!(img.format().unwrap().channels(), 4);
        assert_eq!(img.format().unwrap(), PixelFormat::Bgra);
    }

    // ========================================================================
    // Tests for EDGEFIRST_FORCE_BACKEND env var
    // ========================================================================

    #[test]
    fn test_force_backend_cpu() {
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let result = ImageProcessor::new();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        let converter = result.unwrap();
        assert!(converter.cpu.is_some());
        assert_eq!(converter.forced_backend, Some(ForcedBackend::Cpu));
    }

    #[test]
    fn test_force_backend_invalid() {
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "invalid") };
        let result = ImageProcessor::new();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        assert!(
            matches!(&result, Err(Error::ForcedBackendUnavailable(s)) if s.contains("unknown")),
            "invalid backend value should return ForcedBackendUnavailable error: {result:?}"
        );
    }

    #[test]
    fn test_force_backend_unset() {
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };
        let result = ImageProcessor::new();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        let converter = result.unwrap();
        assert!(converter.forced_backend.is_none());
    }

    // ========================================================================
    // Tests for hybrid mask path error handling
    // ========================================================================

    #[test]
    fn test_draw_proto_masks_no_cpu_returns_error() {
        // Disable CPU backend to trigger the error path
        let original_cpu = std::env::var("EDGEFIRST_DISABLE_CPU").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        let original_gl = std::env::var("EDGEFIRST_DISABLE_GL").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
        let original_g2d = std::env::var("EDGEFIRST_DISABLE_G2D").ok();
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", "1") };

        let result = ImageProcessor::new();

        match original_cpu {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_CPU") },
        }
        match original_gl {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_GL") },
        }
        match original_g2d {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_DISABLE_G2D") },
        }

        let mut converter = result.unwrap();
        assert!(converter.cpu.is_none(), "CPU should be disabled");

        let dst = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let mut dst_dyn = dst;
        let det = [DetectBox {
            bbox: edgefirst_decoder::BoundingBox {
                xmin: 0.1,
                ymin: 0.1,
                xmax: 0.5,
                ymax: 0.5,
            },
            score: 0.9,
            label: 0,
        }];
        let proto_data = {
            use edgefirst_tensor::{Tensor, TensorDyn};
            let coeff_t = Tensor::<f32>::from_slice(&[0.5_f32; 4], &[1, 4]).unwrap();
            let protos_t =
                Tensor::<f32>::from_slice(&vec![0.0_f32; 8 * 8 * 4], &[8, 8, 4]).unwrap();
            ProtoData {
                mask_coefficients: TensorDyn::F32(coeff_t),
                protos: TensorDyn::F32(protos_t),
            }
        };
        let result =
            converter.draw_proto_masks(&mut dst_dyn, &det, &proto_data, Default::default());
        assert!(
            matches!(&result, Err(Error::Internal(s)) if s.contains("CPU backend")),
            "draw_proto_masks without CPU should return Internal error: {result:?}"
        );
    }

    #[test]
    fn test_draw_proto_masks_cpu_fallback_works() {
        // Force CPU-only backend to ensure the CPU fallback path executes
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let result = ImageProcessor::new();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }

        let mut converter = result.unwrap();
        assert!(converter.cpu.is_some());

        let dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let mut dst_dyn = dst;
        let det = [DetectBox {
            bbox: edgefirst_decoder::BoundingBox {
                xmin: 0.1,
                ymin: 0.1,
                xmax: 0.5,
                ymax: 0.5,
            },
            score: 0.9,
            label: 0,
        }];
        let proto_data = {
            use edgefirst_tensor::{Tensor, TensorDyn};
            let coeff_t = Tensor::<f32>::from_slice(&[0.5_f32; 4], &[1, 4]).unwrap();
            let protos_t =
                Tensor::<f32>::from_slice(&vec![0.0_f32; 8 * 8 * 4], &[8, 8, 4]).unwrap();
            ProtoData {
                mask_coefficients: TensorDyn::F32(coeff_t),
                protos: TensorDyn::F32(protos_t),
            }
        };
        let result =
            converter.draw_proto_masks(&mut dst_dyn, &det, &proto_data, Default::default());
        assert!(result.is_ok(), "CPU fallback path should work: {result:?}");
    }

    // ============================================================
    // draw_decoded_masks / draw_proto_masks — 4-scenario pixel-
    // verified tests. Exercises each backend against the full
    // output-contract matrix:
    //
    //   | detections | background | expected dst             |
    //   |------------|------------|--------------------------|
    //   | empty      | none       | fully cleared (0x00)     |
    //   | empty      | set        | fully equal to bg        |
    //   | set        | none       | cleared outside box +    |
    //   |            |            | mask-coloured inside     |
    //   | set        | set        | bg outside box + mask    |
    //   |            |            | blended inside           |
    //
    // Every test pre-fills dst with a non-zero "dirty" pattern so
    // that any silent `return Ok(())` leaks the pattern into the
    // asserted output and fails loudly.
    // ============================================================

    /// Run `body` with `EDGEFIRST_FORCE_BACKEND` temporarily set (or
    /// removed), restoring the prior value afterward. Tests are mutated
    /// env-serialized via the process-wide `FORCE_BACKEND_MUTEX`.
    fn with_force_backend<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        use std::sync::{Mutex, MutexGuard, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard: MutexGuard<()> = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        match value {
            Some(v) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", v) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        let r = body();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        r
    }

    /// Allocate an RGBA image tensor and pre-fill every byte with a
    /// distinctive non-zero pattern. Any test that relies on the old
    /// "dst is already cleared" assumption will see this pattern leak
    /// through to the output and fail.
    fn make_dirty_dst(w: usize, h: usize, mem: Option<TensorMemory>) -> TensorDyn {
        let dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, mem).unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_u8().unwrap();
            let mut map = u8t.map().unwrap();
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = 0xA0u8.wrapping_add((i as u8) & 0x3F);
            }
        }
        dst
    }

    /// Allocate an RGBA background filled with a constant colour.
    fn make_bg(w: usize, h: usize, mem: Option<TensorMemory>, rgba: [u8; 4]) -> TensorDyn {
        let bg = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, mem).unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = bg.as_u8().unwrap();
            let mut map = u8t.map().unwrap();
            for chunk in map.as_mut_slice().chunks_exact_mut(4) {
                chunk.copy_from_slice(&rgba);
            }
        }
        bg
    }

    fn pixel_at(dst: &TensorDyn, x: usize, y: usize) -> [u8; 4] {
        use edgefirst_tensor::TensorMapTrait;
        let w = dst.width().unwrap();
        let off = (y * w + x) * 4;
        let u8t = dst.as_u8().unwrap();
        let map = u8t.map().unwrap();
        let s = map.as_slice();
        [s[off], s[off + 1], s[off + 2], s[off + 3]]
    }

    fn assert_every_pixel_eq(dst: &TensorDyn, expected: [u8; 4], case: &str) {
        use edgefirst_tensor::TensorMapTrait;
        let u8t = dst.as_u8().unwrap();
        let map = u8t.map().unwrap();
        for (i, chunk) in map.as_slice().chunks_exact(4).enumerate() {
            assert_eq!(
                chunk, &expected,
                "{case}: pixel idx {i} = {chunk:?}, expected {expected:?}"
            );
        }
    }

    /// Scenario 1: empty detections, empty segmentation, no background
    /// → dst must be fully cleared to 0x00000000.
    fn scenario_empty_no_bg(processor: &mut ImageProcessor, case: &str) {
        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::default())
            .unwrap_or_else(|e| panic!("{case}/decoded_masks empty+no-bg failed: {e:?}"));
        assert_every_pixel_eq(&dst, [0, 0, 0, 0], &format!("{case}/decoded"));

        let mut dst = make_dirty_dst(64, 64, None);
        let proto = {
            use edgefirst_tensor::{Tensor, TensorDyn};
            // Placeholder (no detections); shape [1, 4] to keep the tensor well-formed.
            let coeff_t = Tensor::<f32>::from_slice(&[0.0_f32; 4], &[1, 4]).unwrap();
            let protos_t =
                Tensor::<f32>::from_slice(&vec![0.0_f32; 8 * 8 * 4], &[8, 8, 4]).unwrap();
            ProtoData {
                mask_coefficients: TensorDyn::F32(coeff_t),
                protos: TensorDyn::F32(protos_t),
            }
        };
        processor
            .draw_proto_masks(&mut dst, &[], &proto, MaskOverlay::default())
            .unwrap_or_else(|e| panic!("{case}/proto_masks empty+no-bg failed: {e:?}"));
        assert_every_pixel_eq(&dst, [0, 0, 0, 0], &format!("{case}/proto"));
    }

    /// Scenario 2: empty detections, empty segmentation, background set
    /// → dst must be fully equal to bg.
    fn scenario_empty_with_bg(processor: &mut ImageProcessor, case: &str) {
        let bg_color = [42, 99, 200, 255];
        let bg = make_bg(64, 64, None, bg_color);
        let overlay = MaskOverlay::new().with_background(&bg);

        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[], &[], overlay)
            .unwrap_or_else(|e| panic!("{case}/decoded_masks empty+bg failed: {e:?}"));
        assert_every_pixel_eq(&dst, bg_color, &format!("{case}/decoded bg blit"));

        let mut dst = make_dirty_dst(64, 64, None);
        let proto = {
            use edgefirst_tensor::{Tensor, TensorDyn};
            // Placeholder (no detections); shape [1, 4] to keep the tensor well-formed.
            let coeff_t = Tensor::<f32>::from_slice(&[0.0_f32; 4], &[1, 4]).unwrap();
            let protos_t =
                Tensor::<f32>::from_slice(&vec![0.0_f32; 8 * 8 * 4], &[8, 8, 4]).unwrap();
            ProtoData {
                mask_coefficients: TensorDyn::F32(coeff_t),
                protos: TensorDyn::F32(protos_t),
            }
        };
        processor
            .draw_proto_masks(&mut dst, &[], &proto, overlay)
            .unwrap_or_else(|e| panic!("{case}/proto_masks empty+bg failed: {e:?}"));
        assert_every_pixel_eq(&dst, bg_color, &format!("{case}/proto bg blit"));
    }

    /// Scenario 3: one detection with a fully-opaque segmentation fill,
    /// no background → outside the box dst must be 0x00, inside it must
    /// be a non-zero mask colour (the render_segmentation output).
    fn scenario_detect_no_bg(processor: &mut ImageProcessor, case: &str) {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;
        processor
            .set_class_colors(&[[200, 80, 40, 255]])
            .expect("set_class_colors");

        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.99,
            label: 0,
        };
        let seg_arr = Array3::from_shape_fn((4, 4, 1), |_| 255u8);
        let seg = Segmentation {
            segmentation: seg_arr,
            xmin: 0.25,
            ymin: 0.25,
            xmax: 0.75,
            ymax: 0.75,
        };

        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[detect], &[seg], MaskOverlay::default())
            .unwrap_or_else(|e| panic!("{case}/decoded_masks detect+no-bg failed: {e:?}"));

        // Outside the bbox (corner): must be cleared black.
        let corner = pixel_at(&dst, 2, 2);
        assert_eq!(
            corner,
            [0, 0, 0, 0],
            "{case}/decoded: corner (2,2) leaked dirty pattern: {corner:?}"
        );
        // Inside the bbox (center): the mask colour must be visible.
        // Any non-zero pixel is acceptable — exact rendering varies
        // between backends (GL smoothstep, CPU nearest).
        let center = pixel_at(&dst, 32, 32);
        assert!(
            center != [0, 0, 0, 0],
            "{case}/decoded: center (32,32) was not coloured: {center:?}"
        );
    }

    /// Scenario 4: detection + background. Outside the box must match
    /// bg; inside the box must NOT match bg (mask blended on top).
    fn scenario_detect_with_bg(processor: &mut ImageProcessor, case: &str) {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;
        processor
            .set_class_colors(&[[200, 80, 40, 255]])
            .expect("set_class_colors");
        let bg_color = [10, 20, 30, 255];
        let bg = make_bg(64, 64, None, bg_color);

        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.99,
            label: 0,
        };
        let seg_arr = Array3::from_shape_fn((4, 4, 1), |_| 255u8);
        let seg = Segmentation {
            segmentation: seg_arr,
            xmin: 0.25,
            ymin: 0.25,
            xmax: 0.75,
            ymax: 0.75,
        };

        let overlay = MaskOverlay::new().with_background(&bg);
        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[detect], &[seg], overlay)
            .unwrap_or_else(|e| panic!("{case}/decoded_masks detect+bg failed: {e:?}"));

        // Outside the bbox (corner): bg colour.
        let corner = pixel_at(&dst, 2, 2);
        assert_eq!(
            corner, bg_color,
            "{case}/decoded: corner (2,2) should show bg {bg_color:?} got {corner:?}"
        );
        // Inside the bbox (center): mask blended on bg, must differ from
        // pure bg (alpha-blend with mask colour produces a distinct shade).
        let center = pixel_at(&dst, 32, 32);
        assert!(
            center != bg_color,
            "{case}/decoded: center (32,32) should differ from bg {bg_color:?}, got {center:?}"
        );
    }

    /// Run all 4 scenarios against the processor. Skip gracefully if
    /// construction fails (backend unavailable on this host).
    fn run_all_scenarios(
        force_backend: Option<&'static str>,
        case: &'static str,
        require_dma_for_bg: bool,
    ) {
        if require_dma_for_bg && !edgefirst_tensor::is_dma_available() {
            eprintln!("SKIPPED: {case} — DMA not available on this host");
            return;
        }
        let processor_result = with_force_backend(force_backend, ImageProcessor::new);
        let mut processor = match processor_result {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {case} — backend init failed: {e:?}");
                return;
            }
        };
        scenario_empty_no_bg(&mut processor, case);
        scenario_empty_with_bg(&mut processor, case);
        scenario_detect_no_bg(&mut processor, case);
        scenario_detect_with_bg(&mut processor, case);
    }

    #[test]
    fn test_draw_masks_4_scenarios_cpu() {
        run_all_scenarios(Some("cpu"), "cpu", false);
    }

    #[test]
    fn test_draw_masks_4_scenarios_auto() {
        run_all_scenarios(None, "auto", false);
    }

    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[test]
    fn test_draw_masks_4_scenarios_opengl() {
        run_all_scenarios(Some("opengl"), "opengl", false);
    }

    /// G2D forced backend: exercises the zero-detection empty-frame
    /// paths via `g2d_clear` and `g2d_blit`. Scenarios 3 and 4 (with
    /// detections) expect `NotImplemented` since G2D has no rasterizer
    /// for boxes / masks.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_draw_masks_zero_detection_g2d_forced() {
        if !edgefirst_tensor::is_dma_available() {
            eprintln!("SKIPPED: g2d forced — DMA not available on this host");
            return;
        }
        let processor_result = with_force_backend(Some("g2d"), ImageProcessor::new);
        let mut processor = match processor_result {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: g2d forced — init failed: {e:?}");
                return;
            }
        };

        // Case 1: empty + no bg. G2D requires DMA-backed dst.
        let mut dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_u8_mut().unwrap();
            let mut map = u8t.map().unwrap();
            map.as_mut_slice().fill(0xBB);
        }
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::default())
            .expect("g2d empty+no-bg");
        assert_every_pixel_eq(&dst, [0, 0, 0, 0], "g2d/case1 cleared");

        // Case 2: empty + bg. Both surfaces DMA-backed for g2d_blit.
        let bg_color = [7, 11, 13, 255];
        let bg = {
            let t = TensorDyn::image(
                64,
                64,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Dma),
            )
            .unwrap();
            {
                use edgefirst_tensor::TensorMapTrait;
                let u8t = t.as_u8().unwrap();
                let mut map = u8t.map().unwrap();
                for chunk in map.as_mut_slice().chunks_exact_mut(4) {
                    chunk.copy_from_slice(&bg_color);
                }
            }
            t
        };
        let mut dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_u8_mut().unwrap();
            let mut map = u8t.map().unwrap();
            map.as_mut_slice().fill(0x55);
        }
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::new().with_background(&bg))
            .expect("g2d empty+bg");
        assert_every_pixel_eq(&dst, bg_color, "g2d/case2 bg blit");

        // Case 3 and 4: detect present — must return NotImplemented.
        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.9,
            label: 0,
        };
        let mut dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        let err = processor
            .draw_decoded_masks(&mut dst, &[detect], &[], MaskOverlay::default())
            .expect_err("g2d must reject detect-present draw_decoded_masks");
        assert!(
            matches!(err, Error::NotImplemented(_)),
            "g2d case3 wrong error: {err:?}"
        );
    }

    #[test]
    fn test_set_format_then_cpu_convert() {
        // Force CPU backend (save/restore to avoid leaking into other tests)
        let original = std::env::var("EDGEFIRST_FORCE_BACKEND").ok();
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let mut processor = ImageProcessor::new().unwrap();
        match original {
            Some(s) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", s) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }

        // Load a source image
        let image = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ));
        let src = load_image(image, Some(PixelFormat::Rgba), None).unwrap();

        // Create a raw tensor, then attach format — simulating the from_fd workflow
        let mut dst =
            TensorDyn::new(&[640, 640, 3], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        dst.set_format(PixelFormat::Rgb).unwrap();

        // Convert should work with the set_format-annotated tensor
        processor
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .unwrap();

        // Verify format survived conversion
        assert_eq!(dst.format(), Some(PixelFormat::Rgb));
        assert_eq!(dst.width(), Some(640));
        assert_eq!(dst.height(), Some(640));
    }

    /// Verify that creating multiple ImageProcessors on the same thread and
    /// performing a resize on each does not deadlock or error.
    ///
    /// Uses automatic memory allocation (DMA → PBO → Mem fallback) so that
    /// hardware backends (OpenGL, G2D) are exercised on capable targets.
    #[test]
    fn test_multiple_image_processors_same_thread() {
        let mut processors: Vec<ImageProcessor> = (0..4)
            .map(|_| ImageProcessor::new().expect("ImageProcessor::new() failed"))
            .collect();

        for proc in &mut processors {
            let src = proc
                .create_image(128, 128, PixelFormat::Rgb, DType::U8, None)
                .expect("create src failed");
            let mut dst = proc
                .create_image(64, 64, PixelFormat::Rgb, DType::U8, None)
                .expect("create dst failed");
            proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .expect("convert failed");
            assert_eq!(dst.width(), Some(64));
            assert_eq!(dst.height(), Some(64));
        }
    }

    /// Verify that creating ImageProcessors on separate threads and performing
    /// a resize on each does not deadlock or error.
    ///
    /// Uses automatic memory allocation (DMA → PBO → Mem fallback) so that
    /// hardware backends (OpenGL, G2D) are exercised on capable targets.
    /// A 60-second timeout prevents CI from hanging on deadlock regressions.
    #[test]
    fn test_multiple_image_processors_separate_threads() {
        use std::sync::mpsc;
        use std::time::Duration;

        const TIMEOUT: Duration = Duration::from_secs(60);

        let (tx, rx) = mpsc::channel::<()>();

        std::thread::spawn(move || {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    std::thread::spawn(move || {
                        let mut proc = ImageProcessor::new().unwrap_or_else(|e| {
                            panic!("ImageProcessor::new() failed on thread {i}: {e}")
                        });
                        let src = proc
                            .create_image(128, 128, PixelFormat::Rgb, DType::U8, None)
                            .unwrap_or_else(|e| panic!("create src failed on thread {i}: {e}"));
                        let mut dst = proc
                            .create_image(64, 64, PixelFormat::Rgb, DType::U8, None)
                            .unwrap_or_else(|e| panic!("create dst failed on thread {i}: {e}"));
                        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                            .unwrap_or_else(|e| panic!("convert failed on thread {i}: {e}"));
                        assert_eq!(dst.width(), Some(64));
                        assert_eq!(dst.height(), Some(64));
                    })
                })
                .collect();

            for (i, h) in handles.into_iter().enumerate() {
                h.join()
                    .unwrap_or_else(|e| panic!("thread {i} panicked: {e:?}"));
            }

            let _ = tx.send(());
        });

        rx.recv_timeout(TIMEOUT).unwrap_or_else(|_| {
            panic!("test_multiple_image_processors_separate_threads timed out after {TIMEOUT:?}")
        });
    }

    /// Verify that 4 fully-initialized ImageProcessors on separate threads can
    /// all operate concurrently without deadlocking each other.
    ///
    /// All processors are created first, then a barrier synchronizes them so
    /// they all start converting at the same instant — maximizing contention.
    /// A 60-second timeout prevents CI from hanging on deadlock regressions.
    #[test]
    fn test_image_processors_concurrent_operations() {
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        const N: usize = 4;
        const ROUNDS: usize = 10;
        const TIMEOUT: Duration = Duration::from_secs(60);

        let (tx, rx) = mpsc::channel::<()>();

        std::thread::spawn(move || {
            let barrier = Arc::new(Barrier::new(N));

            let handles: Vec<_> = (0..N)
                .map(|i| {
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        let mut proc = ImageProcessor::new().unwrap_or_else(|e| {
                            panic!("ImageProcessor::new() failed on thread {i}: {e}")
                        });

                        // All threads wait here until every processor is initialized.
                        barrier.wait();

                        // Now all 4 hammer the GPU concurrently.
                        for round in 0..ROUNDS {
                            let src = proc
                                .create_image(128, 128, PixelFormat::Rgb, DType::U8, None)
                                .unwrap_or_else(|e| {
                                    panic!("create src failed on thread {i} round {round}: {e}")
                                });
                            let mut dst = proc
                                .create_image(64, 64, PixelFormat::Rgb, DType::U8, None)
                                .unwrap_or_else(|e| {
                                    panic!("create dst failed on thread {i} round {round}: {e}")
                                });
                            proc.convert(
                                &src,
                                &mut dst,
                                Rotation::None,
                                Flip::None,
                                Crop::default(),
                            )
                            .unwrap_or_else(|e| {
                                panic!("convert failed on thread {i} round {round}: {e}")
                            });
                            assert_eq!(dst.width(), Some(64));
                            assert_eq!(dst.height(), Some(64));
                        }
                    })
                })
                .collect();

            for (i, h) in handles.into_iter().enumerate() {
                h.join()
                    .unwrap_or_else(|e| panic!("thread {i} panicked: {e:?}"));
            }

            let _ = tx.send(());
        });

        rx.recv_timeout(TIMEOUT).unwrap_or_else(|_| {
            panic!("test_image_processors_concurrent_operations timed out after {TIMEOUT:?}")
        });
    }
}
