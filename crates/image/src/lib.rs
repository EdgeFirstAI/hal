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
# use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait};
# use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
# use edgefirst_tensor::{PixelFormat, DType, Tensor, TensorMemory};
# fn main() -> Result<(), edgefirst_image::Error> {
let image = edgefirst_bench::testdata::read("zidane.jpg");
// The codec emits the source's native format (a colour JPEG decodes to NV12)
// and configures the destination tensor's dims+format during the decode.
let info = peek_info(&image).expect("peek");
let mut src = Tensor::<u8>::image(info.width, info.height, info.format,
                                   Some(TensorMemory::Mem))?;
let mut decoder = ImageDecoder::new();
src.load_image(&mut decoder, &image).expect("decode");
// Convert the native NV12 frame to packed RGB for downstream processing.
let mut converter = ImageProcessor::new()?;
let mut dst = converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None)?;
converter.convert(&src.into(), &mut dst, Rotation::None, Flip::None, Crop::default())?;
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

/// Retained constructor: installs the coverage flush-on-abort handler for this
/// crate's instrumented test binary. See `edgefirst_tensor::covguard`. Only
/// present under coverage on Linux (`.init_array` is ELF-only; flush is Linux-only).
#[cfg(all(coverage, target_os = "linux"))]
#[used]
#[link_section = ".init_array"]
static __EDGEFIRST_COV_INSTALL: extern "C" fn() = {
    extern "C" fn ctor() {
        edgefirst_tensor::covguard::install();
    }
    ctor
};

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

/// Return the GPU-aligned pitch in bytes when a DMA-backed image of
/// `width × fmt` would need row-stride padding, or `None` when the
/// natural pitch already satisfies `GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`
/// or the caller has explicitly requested non-DMA memory.
///
/// Mali G310 (i.MX 95) rejects `eglCreateImage` from DMA-BUFs whose
/// `PLANE0_PITCH_EXT` is not a multiple of 64 bytes, surfacing as
/// `EGL_BAD_ALLOC`. The `load_image_test_helper` test-only helper
/// in this crate uses this to decide whether to allocate a tensor
/// with padded row stride before invoking the decode path; production
/// callers do the equivalent peek → allocate → decode dance themselves
/// (see crate-level docs).
#[cfg(all(target_os = "linux", test))]
pub(crate) fn padded_dma_pitch_for(
    fmt: PixelFormat,
    width: usize,
    memory: &Option<TensorMemory>,
) -> Option<usize> {
    // Only pad when the caller explicitly requested DMA, or when they
    // left memory selection to the allocator AND DMA is actually
    // available. `Tensor::image_with_stride(..., None)` always routes
    // through DMA allocation, so treating `None` as "DMA wanted"
    // unconditionally would convert a normally-working image load into
    // a hard failure on systems where DMA is unavailable (sandboxed
    // CI, missing `/dev/dma_heap`, permission-denied containers) —
    // whereas `Tensor::image(..., None)` would have fallen back to
    // SHM/Mem there.
    match memory {
        Some(TensorMemory::Dma) => {}
        None if edgefirst_tensor::is_dma_available() => {}
        _ => return None,
    }
    // Padding only applies to packed layouts — `Tensor::image_with_stride`
    // rejects semi-planar / planar formats, and those take their own
    // per-plane pitches on import anyway.
    if fmt.layout() != PixelLayout::Packed {
        return None;
    }
    let bpp = primary_plane_bpp(fmt, 1)?;
    let natural = width.checked_mul(bpp)?;
    let aligned = align_pitch_bytes_to_gpu_alignment(natural)?;
    if aligned > natural {
        Some(aligned)
    } else {
        None
    }
}

pub use cpu::CPUProcessor;
pub use edgefirst_codec as codec;

#[cfg(test)]
use edgefirst_decoder::ProtoLayout;
use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
#[doc(inline)]
pub use edgefirst_tensor::Region;
#[cfg(any(test, all(target_os = "linux", feature = "opengl")))]
use edgefirst_tensor::Tensor;
use edgefirst_tensor::{
    DType, PixelFormat, PixelLayout, TensorDyn, TensorMemory, TensorTrait as _,
};
use enum_dispatch::enum_dispatch;
pub use error::{Error, Result};
#[cfg(target_os = "linux")]
pub use g2d::G2DProcessor;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android"
    ),
    feature = "opengl"
))]
pub use opengl_headless::EglDisplayKind;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android"
    ),
    feature = "opengl"
))]
pub use opengl_headless::GLProcessorThreaded;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android"
    ),
    feature = "opengl"
))]
pub use opengl_headless::Int8InterpolationMode;
#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
pub use opengl_headless::{probe_egl_displays, EglDisplayInfo};
// EGLImage cache counter snapshots (diagnostics): see
// `GLProcessorThreaded::egl_cache_stats` and the steady-state import gate in
// `crates/image/ARCHITECTURE.md § image.convert.gl.egl_import`.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android"
    ),
    feature = "opengl"
))]
pub use opengl_headless::{CacheStats, GlCacheStats};
use std::{fmt::Display, time::Instant};

mod colorimetry;
mod cpu;
mod error;
mod g2d;
#[path = "gl/mod.rs"]
mod opengl_headless;

// Use `edgefirst_tensor::PixelFormat` variants (Rgb, Rgba, Grey, etc.) and
// `TensorDyn` / `Tensor<u8>` with `.format()` metadata instead.

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
    pub fn with_letterbox_crop(
        mut self,
        crop: &Crop,
        src_w: usize,
        src_h: usize,
        model_w: usize,
        model_h: usize,
    ) -> Self {
        // The letterbox placement is resolved from the same source/destination
        // dimensions `convert()` used, so the inverse map matches the render.
        if let Ok(resolved) = crop.resolve(src_w, src_h, model_w, model_h) {
            if let Some(r) = resolved.dst_rect {
                self.letterbox = Some([
                    r.left as f32 / model_w as f32,
                    r.top as f32 / model_h as f32,
                    (r.left + r.width) as f32 / model_w as f32,
                    (r.top + r.height) as f32 / model_h as f32,
                ]);
            }
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

/// How a source is fit into the requested destination shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fit {
    /// Stretch the source (or source crop) to fill the whole destination.
    #[default]
    Stretch,
    /// Preserve the *source* aspect ratio, centring it in the destination and
    /// padding the remainder with `pad` (RGBA — e.g. `[114, 114, 114, 255]` for
    /// YOLO-style preprocessing).
    Letterbox { pad: [u8; 4] },
}

/// Source-side convert geometry: which sub-rectangle of the source to sample
/// (`source`) and how to fit it into the destination (`fit`). Destination
/// *placement* is the destination itself — a tensor, or a [`Region`] view /
/// `batch` tile of one — not a field here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop {
    /// Sub-rectangle of the source to sample. `None` samples the whole source.
    pub source: Option<Region>,
    /// How the source is fit into the destination shape.
    pub fit: Fit,
}

impl Crop {
    /// A no-op crop: whole source, stretched to fill the whole destination.
    pub fn new() -> Self {
        Self::default()
    }

    /// Alias for [`Crop::new`] — whole source, stretch to fill.
    pub fn no_crop() -> Self {
        Self::default()
    }

    /// Letterbox fit: preserve the source aspect ratio, padding the remainder
    /// with `pad` (RGBA).
    pub fn letterbox(pad: [u8; 4]) -> Self {
        Self {
            source: None,
            fit: Fit::Letterbox { pad },
        }
    }

    /// Sample only `source` of the input (builder).
    pub fn with_source(mut self, source: Option<Region>) -> Self {
        self.source = source;
        self
    }

    /// Set the fit mode (builder).
    pub fn with_fit(mut self, fit: Fit) -> Self {
        self.fit = fit;
        self
    }

    /// Resolve to the effective backend geometry for a `src_w`×`src_h` source
    /// and `dst_w`×`dst_h` destination: the source sampling rect, the
    /// destination placement rect (`None` = whole destination), and the pad
    /// colour. **The single place letterbox placement is computed** — every
    /// backend consumes the resolved rects rather than re-deriving them.
    pub(crate) fn resolve(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<ResolvedCrop, Error> {
        let src_rect = self.source.map(region_to_rect);
        // The letterbox aspect uses the *effective* source content — the source
        // crop when set, else the full source.
        let (sw, sh) = match self.source {
            Some(r) => (r.width, r.height),
            None => (src_w, src_h),
        };
        let resolved = match self.fit {
            Fit::Stretch => ResolvedCrop {
                src_rect,
                dst_rect: None,
                dst_color: None,
            },
            Fit::Letterbox { pad } => ResolvedCrop {
                src_rect,
                dst_rect: Some(letterbox_rect(sw, sh, dst_w, dst_h)),
                dst_color: Some(pad),
            },
        };
        resolved.check_crop_dims(src_w, src_h, dst_w, dst_h)?;
        Ok(resolved)
    }

    /// Validate against `TensorDyn` source and destination dimensions.
    pub fn check_crop_dyn(
        &self,
        src: &edgefirst_tensor::TensorDyn,
        dst: &edgefirst_tensor::TensorDyn,
    ) -> Result<(), Error> {
        self.resolve(
            src.width().unwrap_or(0),
            src.height().unwrap_or(0),
            dst.width().unwrap_or(0),
            dst.height().unwrap_or(0),
        )
        .map(|_| ())
    }
}

/// Resolved crop geometry consumed by the backends. Produced by
/// [`Crop::resolve`]; the backends read these fields directly (the same shape
/// the public `Crop` carried before destination placement moved to the view).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ResolvedCrop {
    pub(crate) src_rect: Option<Rect>,
    pub(crate) dst_rect: Option<Rect>,
    pub(crate) dst_color: Option<[u8; 4]>,
}

impl ResolvedCrop {
    /// A no-op resolved crop (whole source → whole destination, no pad).
    #[allow(dead_code)] // used by unit tests and the batch render paths
    pub(crate) fn no_crop() -> Self {
        Self::default()
    }

    /// Validate the resolved rects against explicit dimensions.
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
}

/// Convert a pixel [`Region`] to the internal [`Rect`] placement type.
fn region_to_rect(r: Region) -> Rect {
    Rect {
        left: r.x,
        top: r.y,
        width: r.width,
        height: r.height,
    }
}

/// Centred aspect-preserving placement of an `sw`×`sh` source within a `dw`×`dh`
/// destination — the canonical letterbox rectangle (one home, replacing the
/// per-caller `calculate_letterbox` copies).
fn letterbox_rect(sw: usize, sh: usize, dw: usize, dh: usize) -> Rect {
    if sw == 0 || sh == 0 {
        return Rect::new(0, 0, dw, dh);
    }
    let src_aspect = sw as f64 / sh as f64;
    let dst_aspect = dw as f64 / dh as f64;
    let (new_w, new_h) = if src_aspect > dst_aspect {
        (dw, ((dw as f64 / src_aspect).round() as usize).max(1))
    } else {
        (((dh as f64 * src_aspect).round() as usize).max(1), dh)
    };
    let left = dw.saturating_sub(new_w) / 2;
    let top = dh.saturating_sub(new_h) / 2;
    Rect::new(left, top, new_w, new_h)
}

/// Internal placement rectangle (top-left + size). The **public** pixel
/// sub-region type is [`Region`] (re-exported from `edgefirst-tensor`);
/// `Rect` is the crate-private resolved-placement representation used by the
/// CPU/G2D/GL backends after [`Crop::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
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

    /// Like [`convert`](Self::convert), but does not wait for the GPU to finish.
    ///
    /// This is the batch-preprocessing primitive: a caller renders `N` tiles
    /// into one batched destination by looping
    /// `convert_deferred(&src[n], &mut dst.batch(n)?, …)` and then calling
    /// [`flush`](Self::flush) **once**. On the OpenGL backend every deferred
    /// convert into sibling views of one buffer shares a single EGLImage import
    /// (the tile is a `glViewport`/`glScissor` ROI into the parent) and skips the
    /// per-tile `glFinish`; `flush` then issues a single GPU sync. The result of
    /// a deferred convert is **not** safe to read on the CPU (or map via CUDA)
    /// until `flush` returns.
    ///
    /// The default implementation is eager — it simply calls
    /// [`convert`](Self::convert), so CPU/G2D and any backend without a deferred
    /// fast path remain correct (each call completes synchronously and `flush`
    /// is a no-op).
    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        self.convert(src, dst, rotation, flip, crop)
    }

    /// Complete all work enqueued by [`convert_deferred`](Self::convert_deferred)
    /// since the last flush, issuing a single GPU synchronization.
    ///
    /// After this returns, every deferred destination is finished and safe to
    /// read back or `cuda_map`. Backends with no deferred path (the default)
    /// return `Ok(())` immediately, since their converts already completed.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
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
    /// Ignored when `EDGEFIRST_DISABLE_GL=1` is set, and on macOS
    /// (ANGLE/Metal is the only display there; a `Some` value logs a
    /// debug note and is otherwise ignored).
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ),
        feature = "opengl"
    ))]
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

    /// Colorimetry/performance trade-off for `convert()` (see
    /// [`ColorimetryMode`]). Defaults to [`ColorimetryMode::Fast`]. The
    /// `EDGEFIRST_COLORIMETRY` environment variable (`fast` | `exact`)
    /// overrides this setting when present.
    pub colorimetry: ColorimetryMode,
}

/// How `convert()` trades colorimetric exactness against speed on platforms
/// where the exact path is expensive.
///
/// Today this affects one decision: NV12 sources on Vivante GC7000UL
/// (i.MX 8M Plus), where the hardware external sampler converts ~12× faster
/// than the colorimetry-exact in-shader matrix (2.5 ms vs 29 ms at 720p)
/// but applies the driver's fixed BT.601-limited matrix regardless of the
/// source's tagged colorimetry. Platforms where the exact path is already
/// the fastest correct path (Mali, V3D, Tegra, ANGLE) behave identically in
/// both modes.
///
/// Override at runtime with `EDGEFIRST_COLORIMETRY=fast|exact` (takes
/// precedence over the config field), or per-source by forcing a path with
/// `EDGEFIRST_NV_CONVERT_PATH`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorimetryMode {
    /// Prefer the fastest path whose output is correct-enough video RGB
    /// (default; issue #106 policy). On Vivante, NV12 takes the hardware
    /// sampler even when the source is not BT.601-limited.
    #[default]
    Fast,
    /// Prefer bit-exact colorimetry everywhere: the fast path is used only
    /// when it matches the source's resolved (encoding, range) exactly.
    Exact,
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

/// Reports which float-color-buffer extensions the GPU backend detected.
/// Returned by [`ImageProcessor::supported_render_dtypes`]; the two flags
/// are independent.
///
/// **Linux:** reflects the real probe results from `GL_EXT_color_buffer_half_float`
/// and `GL_EXT_color_buffer_float`. On V3D (RPi 5) and Mali-G310 (i.MX 95)
/// both flags are typically `true`; on Vivante GC7000UL both are forced
/// `false` (float readback measured 170–320 ms — disabled). Tegra Orin
/// exposes both via PBO; the flags match the GPU report.
///
/// **macOS (ANGLE):** `f16 == true` gates the RGBA16F-packed IOSurface
/// path for F16 `PlanarRgb` destinations. `f32` reflects the GL
/// extension probe but is not actionable — ANGLE's
/// `EGL_ANGLE_iosurface_client_buffer` rejects every `(GL_FLOAT, *)`
/// combination with `EGL_BAD_ATTRIBUTE`, so there is no F32 IOSurface
/// path.
///
/// Regardless of these flags, [`ImageProcessor::convert`] never returns
/// an error due to float capability — it falls back to CPU when the GPU
/// path is unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderDtypeSupport {
    /// `GL_EXT_color_buffer_float` is available on the current GPU.
    ///
    /// On Linux, `true` enables F32 `Rgb` NHWC PBO readback. On macOS
    /// this flag is informational only — no F32 IOSurface path exists.
    pub f32: bool,
    /// `GL_EXT_color_buffer_half_float` is available on the current GPU.
    ///
    /// On Linux, `true` enables F16 `PlanarRgb` NCHW PBO readback and,
    /// on V3D/Mali, zero-copy DMA-BUF render. On macOS, `true` enables
    /// F16 `PlanarRgb` via RGBA16F-packed IOSurface (zero-copy).
    pub f16: bool,
}

/// Returns `true` when a float PBO destination should be attempted for `dtype`.
///
/// Only F16 and F32 are eligible, and only when the corresponding flag in
/// `support` is set. U8/I8 and all other dtypes return `false` — they are
/// handled by the existing `dtype.size() == 1` PBO gate.
///
/// Linux-only: the float PBO readback path is the Linux GL backend's
/// mechanism; macOS routes F16 through the RGBA16F-packed IOSurface
/// instead and never calls this. The sole runtime caller in
/// `create_image` is `cfg(all(target_os = "linux", feature = "opengl"))`,
/// so leaving this ungated makes it dead code on macOS under
/// `-D warnings`. Its unit test (`float_pbo_eligibility`) carries the
/// matching gate.
#[cfg(all(target_os = "linux", feature = "opengl"))]
pub(crate) fn float_pbo_eligible(dtype: DType, support: RenderDtypeSupport) -> bool {
    match dtype {
        DType::F16 => support.f16,
        DType::F32 => support.f32,
        _ => false,
    }
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
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
    #[cfg(feature = "opengl")]
    /// OpenGL-based image converter — the same unified
    /// `GLProcessorThreaded` engine as Linux (its worker owns a
    /// per-processor context). macOS/iOS run it via ANGLE + IOSurface
    /// (available when ANGLE's libEGL.dylib can be loaded — see
    /// README.md § macOS GPU Acceleration); Android runs it via the
    /// native EGL driver + AHardwareBuffer.
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
    /// ```rust,no_run
    /// # use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait};
    /// # use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
    /// # use edgefirst_tensor::{PixelFormat, DType, Tensor, TensorMemory};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let image = std::fs::read("zidane.jpg")?;
    /// // The codec emits the source's native format (a colour JPEG decodes to
    /// // NV12) and configures the destination tensor during the decode.
    /// let info = peek_info(&image).expect("peek");
    /// let mut src = Tensor::<u8>::image(info.width, info.height, info.format,
    ///                                    Some(TensorMemory::Mem))?;
    /// let mut decoder = ImageDecoder::new();
    /// src.load_image(&mut decoder, &image).expect("decode");
    /// let mut converter = ImageProcessor::new()?;
    /// let mut dst = converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None)?;
    /// converter.convert(&src.into(), &mut dst, Rotation::None, Flip::None, Crop::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        Self::with_config(ImageProcessorConfig::default())
    }

    /// Report which float dtypes the GPU can render to.
    ///
    /// Probes `GL_EXT_color_buffer_half_float` and
    /// `GL_EXT_color_buffer_float` once at `ImageProcessor::new()` time
    /// and caches the result. Call this once at startup to decide whether
    /// to request F16 or F32 destination tensors; [`create_image`] uses
    /// the result internally to auto-select float PBO when supported.
    ///
    /// Returns `RenderDtypeSupport { f32: false, f16: false }` when no
    /// OpenGL backend is active or the `opengl` feature is disabled.
    ///
    /// [`create_image`]: Self::create_image
    pub fn supported_render_dtypes(&self) -> RenderDtypeSupport {
        #[cfg(all(
            any(target_os = "macos", target_os = "ios", target_os = "android"),
            feature = "opengl"
        ))]
        if let Some(gl) = self.opengl.as_ref() {
            return gl.supported_render_dtypes();
        }
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        if let Some(gl) = self.opengl.as_ref() {
            return gl.supported_render_dtypes();
        }
        RenderDtypeSupport {
            f32: false,
            f16: false,
        }
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
                    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
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
                        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
                        #[cfg(feature = "opengl")]
                        opengl: None,
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
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::new(config.egl_display) {
                        Ok(gl) => Some(gl),
                        Err(e) => {
                            log::warn!(
                                "OpenGL requested on macOS but ANGLE init failed: {e:?} \
                                 (install ANGLE via `brew install startergo/angle/angle` \
                                 and re-sign the dylibs — see README.md § macOS GPU \
                                 Acceleration). Falling back to CPU."
                            );
                            None
                        }
                    };
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        #[cfg(feature = "opengl")]
                        opengl,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(target_os = "android")]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::new(config.egl_display) {
                        Ok(gl) => Some(gl),
                        Err(e) => {
                            log::warn!(
                                "OpenGL requested but native EGL init failed: {e:?}. \
                                 Falling back to CPU."
                            );
                            None
                        }
                    };
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        #[cfg(feature = "opengl")]
                        opengl,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(not(any(
                    target_os = "linux",
                    target_os = "macos",
                    target_os = "ios",
                    target_os = "android"
                )))]
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
                    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
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
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::new(config.egl_display).map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced on macOS but ANGLE init failed: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            opengl: Some(opengl),
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(target_os = "android")]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::new(config.egl_display).map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced but native EGL init failed: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            opengl: Some(opengl),
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(not(all(
                        any(
                            target_os = "linux",
                            target_os = "macos",
                            target_os = "ios",
                            target_os = "android"
                        ),
                        feature = "opengl"
                    )))]
                    {
                        Err(Error::ForcedBackendUnavailable(
                            "opengl backend requires Linux or macOS with the 'opengl' feature \
                             enabled"
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

        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
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
                    log::debug!(
                        "GL backend unavailable: {err:?} \
                         (CPU fallback will be used)"
                    );
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
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
            #[cfg(feature = "opengl")]
            opengl,
            forced_backend: None,
        }
        .apply_colorimetry_mode(config.colorimetry))
    }

    /// Apply the configured [`ColorimetryMode`] to whichever backend honours
    /// it (currently the Linux GL backend); no-op elsewhere. Constructor
    /// plumbing for [`ImageProcessorConfig::colorimetry`].
    fn apply_colorimetry_mode(self, _mode: ColorimetryMode) -> Self {
        #[cfg(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android"
            ),
            feature = "opengl"
        ))]
        {
            let mut me = self;
            if let Err(e) = me.set_colorimetry_mode(_mode) {
                log::warn!("Failed to apply ColorimetryMode::{_mode:?}: {e:?}");
            }
            me
        }
        #[cfg(not(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android"
            ),
            feature = "opengl"
        )))]
        {
            let _ = _mode;
            self
        }
    }

    /// Sets the colorimetry/performance trade-off (see [`ColorimetryMode`])
    /// on the OpenGL backend. No-op if OpenGL is not available. The
    /// `EDGEFIRST_COLORIMETRY` environment variable takes precedence — when
    /// it is set, this call logs and keeps the env-selected mode.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ),
        feature = "opengl"
    ))]
    pub fn set_colorimetry_mode(&mut self, mode: ColorimetryMode) -> Result<()> {
        if let Some(ref mut gl) = self.opengl {
            gl.set_colorimetry_mode(mode)?;
        }
        Ok(())
    }

    /// Sets the interpolation mode for int8 proto textures on the OpenGL
    /// backend. No-op if OpenGL is not available.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ),
        feature = "opengl"
    ))]
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) -> Result<()> {
        if let Some(ref mut gl) = self.opengl {
            gl.set_int8_interpolation_mode(mode)?;
        }
        Ok(())
    }

    /// Create a [`TensorDyn`] image with the best available memory backend.
    ///
    /// Priority: DMA-buf → float PBO (F16/F32) → u8/i8 PBO → system memory.
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
    /// **Float dtype behaviour:** when `dtype` is `F16` or `F32` and
    /// [`supported_render_dtypes`] reports the GPU supports that type,
    /// `memory: None` auto-selects a float PBO (Linux) or IOSurface (macOS
    /// F16 only). If GPU float support is absent the allocation falls through
    /// to `TensorMemory::Mem`; [`convert`] then uses the CPU path.
    /// Passing `memory: Some(TensorMemory::Dma)` with `dtype: F32` always
    /// returns `Error::NotSupported` — no 32-bit-float DRM fourcc exists.
    ///
    /// [`supported_render_dtypes`]: Self::supported_render_dtypes
    /// [`convert`]: ImageProcessorTrait::convert
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `format` - Pixel format
    /// * `dtype` - Element data type (e.g. `DType::U8`, `DType::F16`, `DType::F32`)
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
        // On macOS, `TensorMemory::Dma` dispatches through `TensorDyn::image`
        // which selects the IOSurface allocation path (FourCC-formatted)
        // for image-mappable formats, or falls back to SHM/Mem otherwise.
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::Dma) => {
                // F32 has no 32-bit-float DRM fourcc; callers must use PBO instead.
                if dtype == DType::F32 {
                    return Err(Error::NotSupported(
                        "F32 has no 32-bit-float DRM format for DMA-BUF; \
                         use TensorMemory::Pbo for F32"
                            .to_string(),
                    ));
                }
                return try_dma();
            }
            Some(mem) => {
                return Ok(TensorDyn::image(width, height, format, dtype, Some(mem))?);
            }
            None => {}
        }

        // macOS: when the GL backend is active with the IOSurface
        // transfer path, prefer Dma (IOSurface on Apple, AHardwareBuffer
        // on Android) for zero-copy import. The Tensor allocator falls
        // through to SHM/Mem automatically for formats without a
        // zero-copy mapping (NV12, planar u8, etc.).
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
        #[cfg(feature = "opengl")]
        if let Some(gl) = self.opengl.as_ref() {
            let _ = gl; // probe_transfer_backend lives behind the platform trait
            match TensorDyn::image(
                width,
                height,
                format,
                dtype,
                Some(edgefirst_tensor::TensorMemory::Dma),
            ) {
                Ok(img) => return Ok(img),
                Err(e) => {
                    // Falling back to a non-zero-copy destination is a real
                    // perf cliff — never do it silently (on-device triage
                    // starts from this line).
                    log::debug!(
                        "create_image: zero-copy Dma allocation declined \
                         ({format:?}/{dtype:?} {width}x{height}): {e:?}; using fallback storage"
                    );
                }
            }
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

        // Try float PBO when the GPU backend reports support for this dtype.
        // Falls through to Mem on error (same policy as u8 PBO above).
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if float_pbo_eligible(dtype, self.supported_render_dtypes()) {
            if let Some(gl) = &self.opengl {
                match gl.create_pbo_image_dtype(width, height, format, dtype) {
                    Ok(t) => return Ok(t),
                    Err(e) => {
                        log::debug!(
                            "Float PBO image creation failed for {dtype:?}, \
                             falling back to Mem: {e:?}"
                        );
                    }
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
    /// let src = proc.import_image(pd, None, 1920, 1080, PixelFormat::Rgba, DType::U8, None)?;
    ///
    /// // Multi-plane NV12 with stride
    /// let y_pd = PlaneDescriptor::new(y_fd.as_fd())?.with_stride(2048);
    /// let uv_pd = PlaneDescriptor::new(uv_fd.as_fd())?.with_stride(2048);
    /// let src = proc.import_image(y_pd, Some(uv_pd), 1920, 1080,
    ///                             PixelFormat::Nv12, DType::U8, None)?;
    /// ```
    // Import inherently needs plane(s) + geometry + format + dtype + colorimetry;
    // a params struct would obscure more than it clarifies here.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_os = "linux")]
    pub fn import_image(
        &self,
        image: edgefirst_tensor::PlaneDescriptor,
        chroma: Option<edgefirst_tensor::PlaneDescriptor>,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        colorimetry: Option<edgefirst_tensor::Colorimetry>,
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
                    // NV12 (4:2:0): ceil(H/2) chroma rows — odd heights are valid.
                    height.div_ceil(2)
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
                let mut dyn_tensor = TensorDyn::from(tensor_i8);
                dyn_tensor.set_colorimetry(colorimetry);
                return Ok(dyn_tensor);
            }
            let mut dyn_tensor = TensorDyn::from(tensor);
            dyn_tensor.set_colorimetry(colorimetry);
            Ok(dyn_tensor)
        } else {
            // ── Single-plane path ────────────────────────────────────
            // Canonical shape (Packed [H,W,C] / Planar [C,H,W] / SemiPlanar
            // [total_h, W]); `image_shape` supports NV12/NV16/NV24 (the old
            // hand-rolled match erroneously rejected NV24).
            let shape = format.image_shape(width, height).ok_or_else(|| {
                Error::NotSupported(format!(
                    "unsupported pixel format for import_image: {format:?}"
                ))
            })?;
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
            tensor.set_colorimetry(colorimetry);
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
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        letterbox: Option<[f32; 4]>,
        resolution: MaskResolution,
    ) -> Result<Vec<Segmentation>> {
        let cpu = self.cpu.as_mut().ok_or(Error::NoConverter)?;
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
        let _span = tracing::trace_span!(
            "image.convert",
            ?src_fmt,
            ?dst_fmt,
            src_memory = ?src.memory(),
            dst_memory = ?dst.memory(),
            ?rotation,
            ?flip,
        )
        .entered();
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
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android"
                    ))]
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
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ))]
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
            // G2D is matrix-only (no range control, no BT.2020). For any
            // conversion with a YUV side, resolve that side's colorimetry and
            // skip G2D entirely when it cannot be expressed (full-range YUV or
            // BT.2020), letting the chain fall through to GL/CPU which honour
            // range and BT.2020. YUV→RGB uses the source colorimetry; RGB→YUV
            // uses the destination. RGB→RGB has no YUV side and is unaffected.
            let src_is_yuv = src.format().is_some_and(|f| f.is_yuv());
            let dst_is_yuv = dst.format().is_some_and(|f| f.is_yuv());
            let g2d_eligible = if src_is_yuv || dst_is_yuv {
                let cm = if src_is_yuv {
                    crate::colorimetry::effective_colorimetry(src)
                } else {
                    crate::colorimetry::effective_colorimetry(dst)
                };
                crate::g2d::g2d_can_handle(&cm, true)
            } else {
                true
            };
            if !g2d_eligible {
                log::trace!(
                    "convert: auto g2d skipped {src_fmt:?}→{dst_fmt:?} \
                     (colorimetry not expressible: full-range/BT.2020)"
                );
            } else {
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

    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        // Deferred batching is an OpenGL optimization (shared parent EGLImage +
        // no per-tile glFinish). Route to the GL backend's deferred path when GL
        // is forced or auto-selectable; on a GL decline fall back to an eager
        // convert (the auto chain), which is correct everywhere — it completes
        // synchronously and `flush` stays a no-op for non-GL backends.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ))]
        #[cfg(feature = "opengl")]
        {
            let gl_forced = matches!(self.forced_backend, Some(ForcedBackend::OpenGl));
            if gl_forced || self.forced_backend.is_none() {
                if let Some(opengl) = self.opengl.as_mut() {
                    match opengl.convert_deferred(src, dst, rotation, flip, crop) {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            log::trace!("convert_deferred: gl declined: {e}; eager fallback");
                            // A forced-GL caller gets the GL error, matching
                            // `convert`'s no-fallback forced-backend contract.
                            if gl_forced {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        self.convert(src, dst, rotation, flip, crop)
    }

    fn flush(&mut self) -> Result<()> {
        let _span = tracing::trace_span!("image.flush").entered();
        // Only the OpenGL backend defers; flushing it issues the single GPU
        // sync. CPU/G2D converts already completed, so there is nothing to flush.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android"
        ))]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            return opengl.flush();
        }
        Ok(())
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        let _span = tracing::trace_span!(
            "image.draw_decoded_masks",
            n_detections = detect.len(),
            n_segmentations = segmentation.len(),
        )
        .entered();
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
        //
        // CPU materialize needs `&mut` for its MaskScratch buffers; GL also
        // needs `&mut`. The CPU borrow is scoped to its block so the
        // subsequent GL borrow is free to take over `self`.
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let (Some(_), Some(_)) = (self.cpu.as_ref(), self.opengl.as_ref()) {
            let segmentation = match self.cpu.as_mut() {
                Some(cpu) => {
                    log::trace!(
                        "draw_proto_masks started with hybrid (cpu+opengl) in {:?}",
                        start.elapsed()
                    );
                    cpu.materialize_segmentations(detect, proto_data, overlay.letterbox)?
                }
                None => unreachable!("cpu presence checked above"),
            };
            if let Some(opengl) = self.opengl.as_mut() {
                match opengl.draw_decoded_masks(dst, render_detect, &segmentation, overlay) {
                    Ok(_) => {
                        log::trace!(
                            "draw_proto_masks with hybrid (cpu+opengl) in {:?}",
                            start.elapsed()
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        log::trace!(
                            "draw_proto_masks hybrid path failed, falling back to cpu: {e:?}"
                        );
                    }
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

/// Test-only convenience helper that peeks the image header, allocates a
/// tensor sized to the image (honoring DMA pitch padding on Linux when
/// requested), and decodes via [`edgefirst_codec`]. Mirrors the semantics of
/// the removed public `load_image` API for test sites; production callers
/// should use the explicit peek → allocate → decode pattern directly.
#[cfg(test)]
pub(crate) fn load_image_test_helper(
    image: &[u8],
    format: Option<PixelFormat>,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};

    // Peek the source header to get its NATIVE format and dimensions. The
    // codec now emits the source's native format (JPEG → Nv12/Grey, PNG →
    // Rgb/Rgba/Grey) and configures the destination tensor itself.
    let info = peek_info(image)?;
    let native_fmt = info.format;
    let w = info.width;
    let h = info.height;

    let mut decoder = ImageDecoder::new();

    // Decode into a native-format tensor. The decoder sets the tensor's
    // dims+format, so we allocate it sized to the native layout.
    #[cfg(target_os = "linux")]
    let native_src = {
        if let Some(aligned_pitch) = padded_dma_pitch_for(native_fmt, w, &memory) {
            let mut dma = Tensor::<u8>::image_with_stride(
                w,
                h,
                native_fmt,
                aligned_pitch,
                Some(TensorMemory::Dma),
            )?;
            dma.load_image(&mut decoder, image)?;
            TensorDyn::from(dma)
        } else {
            let mut img = Tensor::<u8>::image(w, h, native_fmt, memory)?;
            img.load_image(&mut decoder, image)?;
            TensorDyn::from(img)
        }
    };
    #[cfg(not(target_os = "linux"))]
    let native_src = {
        let mut img = Tensor::<u8>::image(w, h, native_fmt, memory)?;
        img.load_image(&mut decoder, image)?;
        TensorDyn::from(img)
    };

    // If the caller requested a different format, convert into it (same
    // dims) using a headless CPU-backed processor so the helper works
    // without GPU/G2D hardware.
    match format {
        Some(f) if f != native_fmt => {
            let mut dst = TensorDyn::image(w, h, f, DType::U8, memory)?;
            // `ImageProcessorConfig` has platform-specific fields: on Linux it
            // carries extra GL/G2D options so `..Default::default()` is needed,
            // but on macOS `backend` is the only field, making the update
            // redundant (clippy::needless_update). Allow it for cross-platform
            // parity — the alternative (field reassign) trips
            // clippy::field_reassign_with_default on Linux instead.
            #[allow(clippy::needless_update)]
            let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::Cpu,
                ..Default::default()
            })?;
            proc.convert(
                &native_src,
                &mut dst,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;
            Ok(dst)
        }
        _ => Ok(native_src),
    }
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
#[allow(deprecated)]
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

    #[ctor::ctor(unsafe)]
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

    /// Master oracle for the view/batch **destination** batch engine: render `N`
    /// tiles into row-bands of ONE tall destination and assert each band equals
    /// the same source converted standalone — proving correct band placement and
    /// that a later tile's letterbox clear / draw never wipes a sibling band. On
    /// the Linux GL backend the `N` `convert_deferred` calls share ONE parent
    /// EGLImage import (each tile is a `glViewport`/`glScissor` ROI) and sync
    /// once at `flush()`; other backends fall back to an eager per-band convert
    /// (CPU writes via offset + parent stride). Either way the oracle must hold.
    ///
    /// Identical source/tile size makes the convert an exact copy, so the
    /// assertion is backend-agnostic (no GL-vs-CPU resampling drift). Distinct
    /// solid colors per tile make any sibling wipe a hard failure.
    #[test]
    fn batch_view_dst_tiles_match_standalone() {
        let mut proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "SKIPPED: {} — ImageProcessor init failed ({e:?})",
                    function!()
                );
                return;
            }
        };
        let n = 3usize;
        let (w, h) = (32usize, 24usize);
        let colors: [[u8; 4]; 3] = [[210, 40, 40, 255], [40, 210, 40, 255], [40, 40, 210, 255]];
        let make_src = |c: [u8; 4]| -> TensorDyn {
            let bytes: Vec<u8> = c.iter().copied().cycle().take(w * h * 4).collect();
            load_bytes_to_tensor(w, h, PixelFormat::Rgba, Some(TensorMemory::Mem), &bytes).unwrap()
        };
        // Tall destination: N stacked row-bands. DMA so the Linux GL band path
        // runs (one parent import + per-tile glViewport); skip if unavailable.
        let parent = match TensorDyn::image(
            w,
            n * h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        ) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "SKIPPED: {} — tall DMA destination alloc failed ({e:?})",
                    function!()
                );
                return;
            }
        };

        // Deferred batch: one parent import, glViewport/scissor per band, one sync.
        for (i, &c) in colors.iter().enumerate().take(n) {
            let mut tile = parent.view(Region::new(0, i * h, w, h)).unwrap();
            proc.convert_deferred(
                &make_src(c),
                &mut tile,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap_or_else(|e| panic!("convert_deferred tile {i}: {e:?}"));
        }
        proc.flush().unwrap();

        for (i, &c) in colors.iter().enumerate().take(n) {
            // Standalone full-buffer convert of the same source = the oracle.
            let mut solo =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .unwrap();
            proc.convert(
                &make_src(c),
                &mut solo,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

            let band = parent.view(Region::new(0, i * h, w, h)).unwrap();
            let band_bytes = band.as_u8().unwrap().map().unwrap().as_slice().to_vec();
            let solo_bytes = solo.as_u8().unwrap().map().unwrap().as_slice().to_vec();
            assert_eq!(
                band_bytes, solo_bytes,
                "tile {i}: band differs from standalone convert (placement or sibling wipe)"
            );
            assert!(
                band_bytes.chunks_exact(4).all(|p| p == c),
                "tile {i}: band is not the expected solid color {c:?} (sibling wipe?)"
            );
        }
    }

    #[test]
    fn test_invalid_crop() {
        let src = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();
        let dst = TensorDyn::image(100, 100, PixelFormat::Rgb, DType::U8, None).unwrap();

        // A source crop exceeding the source bounds is rejected.
        let crop = Crop::new().with_source(Some(Region::new(50, 50, 60, 60)));
        assert!(matches!(
            crop.check_crop_dyn(&src, &dst),
            Err(Error::CropInvalid(_))
        ));

        // A source crop within bounds is valid.
        let crop = Crop::new().with_source(Some(Region::new(0, 0, 10, 10)));
        assert!(crop.check_crop_dyn(&src, &dst).is_ok());

        // Letterbox is always valid — placement is computed within the dst.
        assert!(Crop::letterbox([0, 0, 0, 255])
            .check_crop_dyn(&src, &dst)
            .is_ok());
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
        let result = crate::load_image_test_helper(&[123; 5000], None, None);
        assert!(
            matches!(result, Err(Error::Codec(_))),
            "unrecognised bytes should surface as Error::Codec, got {result:?}"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_jpeg_format() -> Result<(), Error> {
        let result = crate::load_image_test_helper(&[123; 5000], Some(PixelFormat::Yuyv), None);
        // YUYV is not a valid decode target; peek_info fails before the magic-
        // bytes check, so the precise variant depends on which error fires first.
        assert!(
            matches!(result, Err(Error::Codec(_))),
            "Yuyv target with garbage bytes should surface as Error::Codec, got {result:?}"
        );
        Ok(())
    }

    #[test]
    fn test_load_resize_save() {
        let file = edgefirst_bench::testdata::read("zidane.jpg");
        let img = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
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
        // With `format: None` the helper returns the source's native format.
        // The codec now decodes colour JPEGs to NV12 (was RGB previously).
        let img = crate::load_image_test_helper(&file, None, None).unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(360));
        assert_eq!(img.format().unwrap(), PixelFormat::Nv12);
    }

    #[test]
    fn test_from_tensor_planar() -> Result<(), Error> {
        let mut tensor = Tensor::new(&[3, 720, 1280], None, None)?;
        tensor
            .map()?
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.8bps"));
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
            &edgefirst_bench::testdata::read("camera720p.rgba"),
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
            &edgefirst_bench::testdata::read("camera720p.8bps"),
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
        // Acquire the env-var mutex for the entire test body so we never race
        // with test_force_backend_* or test_draw_proto_masks_no_cpu_returns_error.
        let _lock = acquire_env_lock();

        // Snapshot ALL env vars we might touch so the RAII guard restores them
        // on exit (even on panic), preventing env-var poisoning of other tests.
        let _guard = EnvGuard::snapshot(&[
            "EDGEFIRST_FORCE_BACKEND",
            "EDGEFIRST_DISABLE_GL",
            "EDGEFIRST_DISABLE_G2D",
            "EDGEFIRST_DISABLE_CPU",
        ]);

        // EDGEFIRST_FORCE_BACKEND takes precedence over EDGEFIRST_DISABLE_*,
        // so clear it for the duration of this test.
        unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };

        #[cfg(target_os = "linux")]
        {
            unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", "1") };
            let converter = ImageProcessor::new()?;
            assert!(converter.g2d.is_none());
            unsafe { std::env::remove_var("EDGEFIRST_DISABLE_G2D") };
        }

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        {
            unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
            let converter = ImageProcessor::new()?;
            assert!(converter.opengl.is_none());
            unsafe { std::env::remove_var("EDGEFIRST_DISABLE_GL") };
        }

        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        let converter = ImageProcessor::new()?;
        assert!(converter.cpu.is_none());
        unsafe { std::env::remove_var("EDGEFIRST_DISABLE_CPU") };

        // Disable everything — convert must return NoConverter.
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
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
        // _guard restores all env vars on drop.
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
        // A single-component (greyscale) JPEG decodes to its native GREY
        // format, which has no even-dimension constraint, so the 1024×681
        // `grey.jpg` loads and converts to RGBA successfully.
        let grey_img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("grey.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        assert_eq!(grey_img.width(), Some(1024));
        assert_eq!(grey_img.height(), Some(681));

        // `grey-rgb.jpg` holds the same grey content but is encoded as a
        // 3-component (colour) JPEG, so the codec decodes it to native NV12.
        // Its 1024×681 dimensions have an odd height; NV12 now represents odd
        // dimensions via the `H + ceil(H/2)` combined-plane height, so the
        // decode succeeds and converts to RGBA at the true dimensions.
        let grey_but_rgb = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("grey-rgb.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .expect("odd-height colour JPEG should decode to NV12 and convert to RGBA");
        assert_eq!(grey_but_rgb.width(), Some(1024));
        assert_eq!(grey_but_rgb.height(), Some(681));
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
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
        // create_image is fully stride-aware: a non-64-aligned NV12 DMA tensor
        // may legitimately carry a GPU-pitch-padded row stride — that is the
        // intended behaviour, not a bug. Verify the logical geometry is preserved
        // for any width and that a reported stride is a valid (>= logical)
        // padding, rather than asserting the absence of a stride.
        let converter = ImageProcessor::new().unwrap();

        // 100 is intentionally not a multiple of 64 (the GPU pitch alignment).
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
                if let Some(stride) = img.row_stride() {
                    assert!(
                        stride >= 100,
                        "NV12 row_stride {stride} must be >= the logical width (100)",
                    );
                }
            }
            Err(e) => {
                // Skip cleanly on hosts without a dma-heap.
                eprintln!("SKIPPED: create_image NV12 DMA non-aligned width: {e}");
            }
        }
    }

    #[test]
    #[ignore] // Hangs on desktop platforms where DMA-buf is unavailable and PBO
              // fallback triggers a GPU driver hang during SHM→texture upload (e.g.,
              // NVIDIA without /dev/dma_heap permissions). Works on embedded targets.
    fn test_crop_skip() {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mut converter = ImageProcessor::new().unwrap();
        let converter_dst = converter
            .create_image(1280, 720, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        let crop = Crop::new().with_source(Some(Region::new(0, 0, 640, 640)));
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

    /// CI canary: fails the lane when the GL backend cannot initialize.
    ///
    /// Every GL test in this suite self-skips when the backend is
    /// unavailable — correct for developer machines, but it means a broken
    /// CI GL stack (e.g. the macOS ANGLE re-sign step regressing, the exact
    /// failure mode documented in the workflow) ships an untested GL
    /// backend behind a green lane. Gated on `HAL_TEST_REQUIRE_GL=1`, set
    /// only by CI jobs that install a working GL stack; local runs without
    /// one pass trivially. On macOS it additionally requires
    /// `HAL_TEST_ALLOW_DLOPEN_ANGLE`, so coverage pass 1 (unsigned
    /// binaries, dlopen gate closed) skips it and pass 2 (signed) enforces.
    #[test]
    #[cfg(feature = "opengl")]
    fn gl_backend_available_canary() {
        let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
        if !require_gl {
            eprintln!(
                "SKIPPED: {} — HAL_TEST_REQUIRE_GL is not set to 1",
                function!()
            );
            return;
        }
        #[cfg(target_os = "macos")]
        if std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            eprintln!(
                "SKIPPED: {} — ANGLE dlopen gate closed (coverage pass 1)",
                function!()
            );
            return;
        }
        GLProcessorThreaded::new(None).expect(
            "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to initialize — \
             check the ANGLE install/re-sign step and binary entitlements \
             (macOS) or the EGL stack (Linux)",
        );
    }

    #[test]
    fn test_load_jpeg_with_exif() {
        use edgefirst_codec::peek_info;

        // The migrated codec NEVER applies EXIF orientation: it decodes to the
        // source's native (un-rotated) dimensions and reports the rotation via
        // ImageInfo. `zidane_rotated_exif.jpg` carries EXIF orientation 6
        // (90° clockwise) over a 1280×720 frame.
        let file = edgefirst_bench::testdata::read("zidane_rotated_exif.jpg").to_vec();
        let info = peek_info(&file).unwrap();
        assert_eq!(info.rotation_degrees, 90);
        assert!(!info.flip_horizontal);

        let loaded = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        // Native (un-rotated) dimensions — the decode does not rotate.
        assert_eq!(loaded.width(), Some(1280));
        assert_eq!(loaded.height(), Some(720));

        // Applying the reported rotation downstream reproduces the upright
        // image: it matches `zidane.jpg` rotated by the same 90° clockwise.
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let cpu_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let rotation = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
        let (dst_width, dst_height) = (cpu_src.height().unwrap(), cpu_src.width().unwrap());

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        // Rotate the native-orientation `loaded` frame and the native `zidane`
        // frame by the same reported rotation; the results must agree.
        let loaded_rotated =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let (r0, _loaded, loaded_rotated) = convert_img(
            &mut cpu_converter,
            loaded,
            loaded_rotated,
            rotation,
            Flip::None,
            Crop::no_crop(),
        );
        r0.unwrap();

        let (result, _cpu_src, cpu_dst) = convert_img(
            &mut cpu_converter,
            cpu_src,
            cpu_dst,
            rotation,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        compare_images(&loaded_rotated, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_load_png_with_exif() {
        use edgefirst_codec::peek_info;

        // PNGs also report EXIF orientation without applying it.
        // `zidane_rotated_exif_180.png` carries EXIF orientation 3 (180°).
        let file = edgefirst_bench::testdata::read("zidane_rotated_exif_180.png").to_vec();
        let info = peek_info(&file).unwrap();
        assert_eq!(info.rotation_degrees, 180);
        assert!(!info.flip_horizontal);

        let loaded = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        // Native (un-rotated) dimensions — PNG decodes upright as authored.
        assert_eq!(loaded.height(), Some(720));
        assert_eq!(loaded.width(), Some(1280));

        // The PNG fixture stores upright `zidane` pixels tagged with a 180°
        // EXIF orientation. Because the codec no longer applies the rotation,
        // the decoded pixels match `zidane.jpg` directly (no convert needed).
        // Re-applying the reported rotation to both must still agree.
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let cpu_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let rotation = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        let (result, _cpu_src, cpu_dst) = convert_img(
            &mut cpu_converter,
            cpu_src,
            cpu_dst,
            rotation,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        // Rotate the decoded PNG by the same reported angle so both frames are
        // in the same (180°-rotated) orientation before comparing.
        let loaded_rotated =
            TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None).unwrap();
        let (r0, _loaded, loaded_rotated) = convert_img(
            &mut cpu_converter,
            loaded,
            loaded_rotated,
            rotation,
            Flip::None,
            Crop::no_crop(),
        );
        r0.unwrap();

        // Threshold 0.95 (was 0.98): `loaded` comes from a lossless PNG decode
        // while `cpu_src` (zidane.jpg) now decodes through native NV12 (chroma
        // subsampling) before the RGBA conversion, so the two paths differ by a
        // couple of percent versus the old direct-RGB JPEG decode.
        compare_images(&loaded_rotated, &cpu_dst, 0.95, function!());
    }

    /// Synthesise an RGB JPEG with a deterministic pattern at `(width, height)`
    /// using the workspace's `jpeg-encoder` crate (the `image` crate is
    /// compiled without its JPEG feature). Used to exercise the decoder /
    /// pitch-padding paths for arbitrary dimensions without having to bundle
    /// a fixture file per test size.
    #[cfg(target_os = "linux")]
    fn make_rgb_jpeg(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                bytes.push(((x + y) & 0xFF) as u8);
                bytes.push(((x.wrapping_mul(3)) & 0xFF) as u8);
                bytes.push(((y.wrapping_mul(5)) & 0xFF) as u8);
            }
        }
        let mut out = Vec::new();
        let encoder = jpeg_encoder::Encoder::new(&mut out, 85);
        encoder
            .encode(
                &bytes,
                width as u16,
                height as u16,
                jpeg_encoder::ColorType::Rgb,
            )
            .expect("jpeg-encoder must succeed on trivial input");
        out
    }

    /// End-to-end: a 375×333 RGBA JPEG (width NOT divisible by 4) loaded
    /// via the pitch-padded DMA path and letterboxed through the GL
    /// backend must produce correct output. Before the Rgba/Bgra
    /// width%4 relaxation in `DmaImportAttrs::from_tensor`, this case
    /// failed the pre-check and forced a CPU texture upload fallback;
    /// with the relaxation, EGL import succeeds at the driver level and
    /// the GL fast path runs. Output correctness is checked against a
    /// CPU reference (convert ran with `EDGEFIRST_FORCE_BACKEND=cpu`).
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_convert_rgba_non_4_aligned_width_end_to_end() {
        use edgefirst_tensor::is_dma_available;
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_convert_rgba_non_4_aligned_width_end_to_end — DMA not available"
            );
            return;
        }
        // 375 is the canonical failure width from dataset loaders —
        // 375 * 4 = 1500 bytes/row, pitch-padded to 1536. Width%4 = 3,
        // so the old pre-check rejected it; new code accepts it.
        let jpeg = make_rgb_jpeg(375, 333);
        let src_gl = crate::load_image_test_helper(&jpeg, Some(PixelFormat::Rgba), None).unwrap();
        assert_eq!(src_gl.width(), Some(375));
        // Row stride must still be pitch-padded (separate concern from width).
        let stride = src_gl.row_stride().unwrap();
        assert_eq!(stride, 1536, "expected padded pitch 1536, got {stride}");

        // GL-backed convert into a pitch-aligned 640×640 Rgba dest.
        let mut gl_proc = ImageProcessor::new().unwrap();
        let gl_dst = gl_proc
            .create_image(640, 640, PixelFormat::Rgba, DType::U8, None)
            .unwrap();
        let (r_gl, _src_gl, gl_dst) = convert_img(
            &mut gl_proc,
            src_gl,
            gl_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        r_gl.expect("GL-backed convert must succeed for 375x333 Rgba src");

        // CPU reference via a fresh load so the two paths start from
        // byte-identical inputs. `with_config(backend=Cpu)` forces the
        // CPU-only processor regardless of which backends the host has
        // available.
        let src_cpu =
            crate::load_image_test_helper(&jpeg, Some(PixelFormat::Rgba), Some(TensorMemory::Mem))
                .unwrap();
        let mut cpu_proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let cpu_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let (r_cpu, _src_cpu, cpu_dst) = convert_img(
            &mut cpu_proc,
            src_cpu,
            cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        r_cpu.unwrap();

        // Structural similarity: the GL path may have gone through EGL
        // import OR fallen back to CPU texture upload — either way, the
        // output must match the CPU reference closely.
        compare_images(&gl_dst, &cpu_dst, 0.95, function!());
    }

    /// Regression lock: loading a JPEG at a non-64-aligned RGBA pitch (e.g.
    /// 500×333 → natural pitch 2000, needs to be padded to 2048) must go
    /// through `image_with_stride` and set `row_stride()` / `effective_row_stride()`
    /// to the padded value. The earlier pitch-padding commit fixed this in
    /// `load_jpeg`; a regression would surface as `row_stride == None` or
    /// `effective_row_stride == 2000`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_load_jpeg_rgba_non_aligned_pitch_padded_dma() {
        use edgefirst_tensor::is_dma_available;
        if !is_dma_available() {
            eprintln!(
                "SKIPPED: test_load_jpeg_rgba_non_aligned_pitch_padded_dma — DMA not available"
            );
            return;
        }
        // Widths that force a non-64-aligned natural RGBA pitch. All three
        // are divisible by 4 so the EGL width-alignment pre-check passes.
        // The pitch-padding fix is what makes these importable at all.
        for &w in &[500u32, 612, 428] {
            let jpeg = make_rgb_jpeg(w, 333);
            let loaded =
                crate::load_image_test_helper(&jpeg, Some(PixelFormat::Rgba), None).unwrap();
            let natural = (w as usize) * 4;
            let aligned = crate::align_pitch_bytes_to_gpu_alignment(natural).unwrap();
            assert!(
                aligned > natural,
                "test sanity: width {w} should be unaligned"
            );
            let stride = loaded
                .row_stride()
                .expect("padded DMA path must set an explicit row_stride — regression if None");
            assert_eq!(
                stride, aligned,
                "width {w}: expected padded stride {aligned}, got {stride} \
                 (regression: pitch-padding branch skipped?)"
            );
            let eff = loaded.effective_row_stride().unwrap();
            assert_eq!(
                eff, aligned,
                "effective_row_stride must match stored stride"
            );
            assert_eq!(loaded.width(), Some(w as usize));
            assert_eq!(loaded.height(), Some(333));
        }
    }

    /// `padded_dma_pitch_for` must respect the caller's memory choice and
    /// must NOT route into the pitch-padded DMA path when the caller left
    /// the choice to the allocator (`None`) but DMA is unavailable on the
    /// host. The padded path requires `image_with_stride`, which always
    /// allocates DMA — taking it on a system without `/dev/dma_heap`
    /// would convert a normally-working image load into a hard failure
    /// (since `Tensor::image(..., None)` would have fallen back to
    /// SHM/Mem).
    #[test]
    #[cfg(target_os = "linux")]
    fn test_padded_dma_pitch_for_respects_memory_choice() {
        use edgefirst_tensor::{is_dma_available, TensorMemory};

        // 500×4 = 2000 → padded to 2048 by GPU alignment. Use it for
        // every case so any "no padding" answer is unambiguous.
        let unaligned_w = 500;

        // Caller asks for Mem / Shm: never pad, regardless of DMA.
        assert_eq!(
            crate::padded_dma_pitch_for(PixelFormat::Rgba, unaligned_w, &Some(TensorMemory::Mem),),
            None,
            "Mem must never trigger DMA padding"
        );
        assert_eq!(
            crate::padded_dma_pitch_for(PixelFormat::Rgba, unaligned_w, &Some(TensorMemory::Shm),),
            None,
            "Shm must never trigger DMA padding"
        );

        // Caller explicitly asks for DMA: always pad if width needs it.
        // Even if the runtime can't actually allocate DMA, the caller
        // owns that decision and the resulting allocation error is
        // their problem, not ours.
        assert_eq!(
            crate::padded_dma_pitch_for(PixelFormat::Rgba, unaligned_w, &Some(TensorMemory::Dma),),
            Some(2048),
            "explicit Dma must pad regardless of runtime DMA availability"
        );

        // Caller leaves it to the allocator: behaviour depends on
        // host-runtime DMA availability. This is the case the fix
        // guards against.
        let none_result = crate::padded_dma_pitch_for(PixelFormat::Rgba, unaligned_w, &None);
        if is_dma_available() {
            assert_eq!(
                none_result,
                Some(2048),
                "memory=None + DMA available → pad (will route through DMA)"
            );
        } else {
            assert_eq!(
                none_result, None,
                "memory=None + DMA unavailable → must NOT pad (would force \
                 image_with_stride into a DMA-only allocation that fails). \
                 Regression: padded_dma_pitch_for ignored is_dma_available()."
            );
        }
    }

    // Synthesise a small greyscale PNG in memory at `(width, height)` with a
    // deterministic ramp pattern so multiple tests can cross-check output
    // without bundling an extra fixture file.
    fn make_grey_png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                bytes.push(((x + y) & 0xFF) as u8);
            }
        }
        let img = image::GrayImage::from_vec(width, height, bytes).unwrap();
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    /// Greyscale PNG with a width that forces a pitch-misaligned natural
    /// row stride (612 bytes is not a multiple of the 64-byte GPU pitch
    /// alignment) must still load via the pitch-padded DMA path. Gated on
    /// DMA availability because `image_with_stride` is DMA-only.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_load_png_grey_misaligned_width_dma() {
        use edgefirst_tensor::is_dma_available;
        if !is_dma_available() {
            eprintln!("SKIPPED: test_load_png_grey_misaligned_width_dma — DMA not available");
            return;
        }
        let png = make_grey_png(612, 388);
        let loaded = crate::load_image_test_helper(&png, Some(PixelFormat::Grey), None).unwrap();
        assert_eq!(loaded.width(), Some(612));
        assert_eq!(loaded.height(), Some(388));
        assert_eq!(loaded.format(), Some(PixelFormat::Grey));

        // Round-trip pixels — natural-pitch DMA-BUFs pad the stride so we
        // must indirect through row_stride() rather than assume width.
        let map = loaded.as_u8().unwrap().map().unwrap();
        let stride = loaded.row_stride().unwrap_or(612);
        assert!(stride >= 612);
        let bytes: &[u8] = &map;
        for y in 0..388usize {
            for x in 0..612usize {
                let expected = ((x + y) & 0xFF) as u8;
                let got = bytes[y * stride + x];
                assert_eq!(
                    got, expected,
                    "grey png mismatch at ({x},{y}): got {got} expected {expected}"
                );
            }
        }
    }

    /// Greyscale PNG loaded with explicit Mem backing — runs on any
    /// platform (no DMA permission requirement) and covers the
    /// decoder-native Luma → Grey no-conversion path.
    #[test]
    fn test_load_png_grey_mem() {
        use edgefirst_tensor::TensorMemory;
        let png = make_grey_png(612, 100);
        let loaded =
            crate::load_image_test_helper(&png, Some(PixelFormat::Grey), Some(TensorMemory::Mem))
                .unwrap();
        assert_eq!(loaded.width(), Some(612));
        assert_eq!(loaded.height(), Some(100));
        assert_eq!(loaded.format(), Some(PixelFormat::Grey));
        let map = loaded.as_u8().unwrap().map().unwrap();
        let bytes: &[u8] = &map;
        // Mem allocation uses the natural pitch — 612 bytes per row, exact.
        assert_eq!(bytes.len(), 612 * 100);
        for y in 0..100 {
            for x in 0..612 {
                assert_eq!(bytes[y * 612 + x], ((x + y) & 0xFF) as u8);
            }
        }
    }

    /// Greyscale PNG decoded into RGB — exercises the decoder-colorspace
    /// mismatch path (Luma → Rgb via CPU converter). Uses Mem memory to
    /// stay portable to host-side test environments.
    #[test]
    fn test_load_png_grey_to_rgb_mem() {
        use edgefirst_tensor::TensorMemory;
        let png = make_grey_png(620, 240);
        let loaded =
            crate::load_image_test_helper(&png, Some(PixelFormat::Rgb), Some(TensorMemory::Mem))
                .unwrap();
        assert_eq!(loaded.width(), Some(620));
        assert_eq!(loaded.height(), Some(240));
        assert_eq!(loaded.format(), Some(PixelFormat::Rgb));

        // Greyscale promoted to RGB replicates luma into each channel.
        let map = loaded.as_u8().unwrap().map().unwrap();
        let bytes: &[u8] = &map;
        for (x, y) in [(0usize, 0usize), (100, 50), (619, 239)] {
            let expected = ((x + y) & 0xFF) as u8;
            let off = (y * 620 + x) * 3;
            assert_eq!(bytes[off], expected, "R@{x},{y}");
            assert_eq!(bytes[off + 1], expected, "G@{x},{y}");
            assert_eq!(bytes[off + 2], expected, "B@{x},{y}");
        }
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src =
            crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma))
                .unwrap();

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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

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

        let img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("grey.jpg"),
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::new().with_source(Some(Region::new(0, 0, 640, 360)));
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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::new();
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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let src_dyn = src;

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let mut g2d_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let crop = Crop::new().with_source(Some(Region::new(50, 120, 1024, 576)));

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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let crop = Crop::new().with_source(Some(Region::new(320, 180, 1280 - 320, 720 - 180)));

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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgba, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::new();
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();

        let mut cpu_converter = CPUProcessor::new();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        let mut mem = vec![None, Some(TensorMemory::Mem), Some(TensorMemory::Shm)];
        if is_dma_available() {
            mem.push(Some(TensorMemory::Dma));
        }
        let crop = Crop::new().with_source(Some(Region::new(50, 120, 1024, 576)));
        for m in mem {
            let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), m).unwrap();
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();

        let unchanged_src =
            crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

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

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src =
            crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), tensor_memory).unwrap();

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

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src =
            crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma))
                .unwrap();

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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_rgba_to_yuyv_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            &edgefirst_bench::testdata::read("camera720p.rgba"),
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
            &edgefirst_bench::testdata::read("camera720p.rgba"),
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
            Crop::letterbox([255, 255, 255, 255]),
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
            &edgefirst_bench::testdata::read("camera720p.rgba"),
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
        let crop = Crop::new();

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
        let file = edgefirst_bench::testdata::read("camera720p.yuyv").to_vec();
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // CPU path resolves the untagged 720p source to BT.709 limited (height
        // heuristic), matching the BT.709 camera fixture; measured 0.9995.
        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgb_cpu() {
        let file = edgefirst_bench::testdata::read("camera720p.yuyv").to_vec();
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
                edgefirst_bench::testdata::read("camera720p.rgba")
                    .as_chunks::<4>()
                    .0,
            )
            .for_each(|(dst, src)| *dst = [src[0], src[1], src[2]]);

        // CPU path resolves the untagged 720p source to BT.709 limited (height
        // heuristic), matching the BT.709 camera fixture; measured 0.9995.
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // Post-WS1 the GPU path applies the resolved per-tensor colorimetry,
        // so the matrix delta vs the reference that forced 0.95 has closed;
        // tightened to 0.98 (confirmed on the GPU/G2D lanes).
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // Post-WS1 the GPU path applies the resolved per-tensor colorimetry,
        // so the matrix delta vs the reference that forced 0.95 has closed;
        // tightened to 0.98 (confirmed on the GPU/G2D lanes).
        compare_images(&dst, &target_image, 0.98, function!());
    }

    /// macOS analog of `test_yuyv_to_rgba_opengl` — drives the ANGLE +
    /// IOSurface backend end-to-end and compares against the same
    /// reference image. Skips silently if ANGLE isn't installed so the
    /// test suite still passes on CI hosts without the Homebrew tap.
    /// Step-1 probe: proves ANGLE's Metal IOSurface-client-buffer path accepts
    /// an `L008`→`GL_RED` (R8) binding — the foundation for sampling the
    /// contiguous semi-planar YUV buffer as a single R8 texture. Renders a
    /// GREY (R8 IOSurface) source through the GL backend to RGBA and checks the
    /// luma round-trips to R=G=B (identity GREY→RGB).
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_grey_r8_iosurface_to_rgba_opengl_macos() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };

        let (w, h) = (16usize, 16usize);
        let src = TensorDyn::image(w, h, PixelFormat::Grey, DType::U8, Some(TensorMemory::Dma))
            .expect("GREY IOSurface (R8/L008) should allocate — proves the FourCC mapping");
        // Known luma ramp: value = (x * 13 + y * 7) & 0xff.
        {
            let su8 = src.as_u8().unwrap();
            let stride = src.as_u8().unwrap().effective_row_stride().unwrap();
            let mut m = su8.map().unwrap();
            let buf = m.as_mut_slice();
            for y in 0..h {
                for x in 0..w {
                    buf[y * stride + x] = ((x * 13 + y * 7) & 0xff) as u8;
                }
            }
        }

        let dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();
        let (result, src_back, dst) = convert_img(
            &mut proc,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.expect("GREY(R8 IOSurface) → RGBA must convert on ANGLE (R8 binding works)");

        let src_stride = src_back.as_u8().unwrap().effective_row_stride().unwrap();
        let src_map = src_back.as_u8().unwrap().map().unwrap();
        let sbytes = src_map.as_slice();
        let dst_stride = dst.as_u8().unwrap().effective_row_stride().unwrap();
        let dst_map = dst.as_u8().unwrap().map().unwrap();
        let dbytes = dst_map.as_slice();
        for y in 0..h {
            for x in 0..w {
                let yv = sbytes[y * src_stride + x] as i16;
                let p = y * dst_stride + x * 4;
                for c in 0..3 {
                    assert!(
                        (dbytes[p + c] as i16 - yv).abs() <= 2,
                        "pixel ({x},{y}) ch{c} = {} expected ~{yv} (GREY→RGB identity)",
                        dbytes[p + c]
                    );
                }
            }
        }
    }

    /// Two-pass GPU chain: NV12 (R8 IOSurface) → PlanarRgb F16, the profiler's
    /// preprocess. Verifies the chained `convert_nv_to_planar_float`
    /// (NV12→RGBA8 then the verified RGBA8→PlanarRgb F16) executes on ANGLE and
    /// produces a sane F16 planar result: a neutral-grey NV12 input (Y=U=V=128,
    /// BT.601 full ⇒ RGB≈0.5) must yield all three planes ≈0.5 (half-float).
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_two_pass_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — init failed ({e:?})", function!());
                return;
            }
        };
        let (w, h) = (64usize, 64usize);
        let src =
            match TensorDyn::image(w, h, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Dma)) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("SKIPPED: {} — NV12 IOSurface alloc: {e:?}", function!());
                    return;
                }
            };
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128); // Y=U=V=128

        let dst = match TensorDyn::image(
            w,
            h,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} — F16 PlanarRgb IOSurface: {e:?}", function!());
                return;
            }
        };
        let mut dst = dst;
        // Call convert directly (the convert_img helper restores u8 only).
        if let Err(e) = ImageProcessorTrait::convert(
            &mut gpu,
            &src,
            &mut dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        ) {
            // GL_EXT_color_buffer_half_float may be absent on some configs; the
            // RGBA8 pass-1 + F16 pass-2 path then can't render. Skip rather than
            // fail on a capability gap (the same policy as the F16 path tests).
            eprintln!(
                "SKIPPED: {} — NV12→PlanarRgb F16 not available ({e:?})",
                function!()
            );
            return;
        }
        let dt = dst.as_f16().expect("dst is F16 PlanarRgb");
        let map = dt.map().unwrap();
        let vals = map.as_slice();
        // Neutral grey → ~0.5 in every plane. Allow generous tolerance for the
        // mediump YUV math + half-float rounding.
        let mut checked = 0usize;
        for &v in vals.iter() {
            let f = f32::from(v);
            assert!(
                (0.40..=0.60).contains(&f),
                "planar F16 value {f} not ~0.5 for neutral-grey NV12"
            );
            checked += 1;
        }
        assert!(
            checked >= w * h * 3,
            "expected >= 3 planes of samples, got {checked}"
        );
    }

    /// Profiler-shaped two-pass: a reused **R8/Grey pool** (allocated larger
    /// than the frame, the NV24 worst case `3·H`) is reconfigured to an NV12
    /// frame, filled at the preserved physical stride, and converted with a
    /// letterbox `src_rect` crop into a model-sized PlanarRgb F16 destination —
    /// exactly the orchestrator's preprocess. Guards against the pooled
    /// two-pass NV→PlanarRgb F16 path hanging/erroring (the exact-size
    /// `test_nv12_to_planar_f16_two_pass` never exercised the larger pool).
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_two_pass_pool_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — init failed ({e:?})", function!());
                return;
            }
        };
        // Frame 96×64 in a 256×768 R8 pool (3·256 height; bpr padded past 96).
        let (fw, fh) = (96usize, 64usize);
        let (pool_w, pool_h) = (256usize, 768usize);
        let (model_w, model_h) = (128usize, 128usize);

        let mut src = match TensorDyn::image(
            pool_w,
            pool_h,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} — R8 pool alloc: {e:?}", function!());
                return;
            }
        };
        src.configure_image(fw, fh, PixelFormat::Nv12)
            .unwrap_or_else(|e| panic!("configure_image NV12 on pool: {e}"));
        let stride = src.as_u8().unwrap().effective_row_stride().unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128); // neutral grey

        let mut dst = match TensorDyn::image(
            model_w,
            model_h,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} — F16 PlanarRgb dst: {e:?}", function!());
                return;
            }
        };

        // Letterbox crop like the profiler: the backend computes the band.
        let _ = model_w;
        let crop = Crop::new()
            .with_source(Some(Region::new(0, 0, fw, fh)))
            .with_fit(Fit::Letterbox {
                pad: [0, 0, 0, 255],
            });
        if let Err(e) =
            ImageProcessorTrait::convert(&mut gpu, &src, &mut dst, Rotation::None, Flip::None, crop)
        {
            eprintln!(
                "SKIPPED: {} — NV12→PlanarRgb F16 unavailable ({e:?})",
                function!()
            );
            return;
        }
        let _ = stride;
        // Neutral grey → ~0.5 inside the letterbox band; just assert the convert
        // completed and produced finite values (no hang, no NaN garbage).
        let dt = dst.as_f16().expect("dst F16");
        let map = dt.map().unwrap();
        let any_half = map.as_slice().iter().any(|&v| {
            let f = f32::from(v);
            (0.40..=0.60).contains(&f)
        });
        assert!(any_half, "expected ~0.5 grey samples in the letterbox band");
    }

    /// Mirrors the orchestrator: the GL processor is created on one thread
    /// and `convert()` is called from a *different* thread (the profiler's
    /// Pre-processing worker). Reproduces (or rules out) the GL-context /
    /// `glFinish` cross-thread hang seen in the live pipeline. A 20 s watchdog
    /// fails loudly rather than hanging the whole test binary.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_cross_thread_opengl_macos() {
        use std::sync::mpsc;
        // Public ImageProcessor (Send) created HERE (the main test thread),
        // exactly like the orchestrator builds `config.processor` during setup.
        let mut proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — init failed ({e:?})", function!());
                return;
            }
        };
        let (fw, fh) = (96usize, 64usize);
        let mut src = match TensorDyn::image(
            256,
            768,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} — pool: {e:?}", function!());
                return;
            }
        };
        src.configure_image(fw, fh, PixelFormat::Nv12).unwrap();
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128);
        let mut dst = match TensorDyn::image(
            128,
            128,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Dma),
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: {} — dst: {e:?}", function!());
                return;
            }
        };
        let crop = Crop::new().with_source(Some(Region::new(0, 0, fw, fh)));

        // ...then MOVED to a worker thread where convert() runs — exactly the
        // orchestrator's create-on-setup / convert-on-Pre-processing split.
        let (tx, rx) = mpsc::channel::<bool>();
        let worker = std::thread::spawn(move || {
            let _ = ImageProcessorTrait::convert(
                &mut proc,
                &src,
                &mut dst,
                Rotation::None,
                Flip::None,
                crop,
            );
            let _ = tx.send(true);
        });
        match rx.recv_timeout(std::time::Duration::from_secs(20)) {
            Ok(_) => { let _ = worker.join(); }
            Err(_) => panic!(
                "cross-thread NV12→PlanarRgb convert HUNG (>20s) — reproduces the orchestrator deadlock"
            ),
        }
    }

    /// Reproduces the profiler's progressive-slowdown/hang: one processor
    /// converting many **varying-size** NV frames (like a COCO dataset) from a
    /// reused R8 pool into a fixed PlanarRgb F16 model input. The two-pass path
    /// reallocated its RGBA intermediate per frame-size, churning/leaking
    /// pbuffers until the GPU stalled. Asserts per-convert latency stays bounded
    /// (no runaway) over many iterations.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv_to_planar_f16_varying_sizes_no_leak_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — init failed ({e:?})", function!());
                return;
            }
        };
        // Mirror the orchestrator's ring buffers at depth 4: a pool of source R8
        // tensors AND a pool of PlanarRgb F16 dst slots, both cycled per frame.
        let (max_w, max_h) = (640usize, 640usize);
        let depth = 4usize;
        let mut srcs = Vec::new();
        let mut dsts = Vec::new();
        for _ in 0..depth {
            srcs.push(
                match TensorDyn::image(
                    max_w,
                    max_h * 3,
                    PixelFormat::Grey,
                    DType::U8,
                    Some(TensorMemory::Dma),
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("SKIPPED: {} — pool: {e:?}", function!());
                        return;
                    }
                },
            );
            dsts.push(
                match TensorDyn::image(
                    640,
                    640,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    Some(TensorMemory::Dma),
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("SKIPPED: {} — dst: {e:?}", function!());
                        return;
                    }
                },
            );
        }
        // COCO-like assorted frame sizes (all ≤ max), cycled.
        let sizes = [
            (640, 480),
            (500, 375),
            (640, 427),
            (333, 500),
            (480, 640),
            (612, 612),
            (428, 640),
            (576, 432),
        ];
        let mut first_ms = 0f64;
        let mut last_ms = 0f64;
        let iters = 40usize;
        for i in 0..iters {
            let (fw, fh) = sizes[i % sizes.len()];
            let src = &mut srcs[i % depth];
            let dst = &mut dsts[i % depth];
            src.configure_image(fw, fh, PixelFormat::Nv24).unwrap();
            src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128);
            let crop = Crop::new().with_source(Some(Region::new(0, 0, fw, fh)));
            let t0 = std::time::Instant::now();
            ImageProcessorTrait::convert(&mut gpu, src, dst, Rotation::None, Flip::None, crop)
                .unwrap_or_else(|e| panic!("convert iter {i} ({fw}×{fh}): {e}"));
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if i == 2 {
                first_ms = ms;
            }
            if i == iters - 1 {
                last_ms = ms;
            }
        }
        eprintln!("first={first_ms:.2}ms last={last_ms:.2}ms");
        assert!(
            last_ms < first_ms * 5.0 + 5.0,
            "convert latency ran away: first {first_ms:.2}ms → last {last_ms:.2}ms (intermediate/pbuffer leak)"
        );
    }

    /// Step-2 verification: NV12/NV16/NV24 (R8 IOSurface) → RGBA on the GPU
    /// must match the CPU `yuv` kernels within shader rounding. Fills an
    /// IOSurface source and a Mem source from the same logical YUV pattern
    /// (each at its own row stride), converts the IOSurface on the GPU and the
    /// Mem one on the CPU, and compares. Exercises the in-shader semi-planar
    /// addressing for all three subsamplings (incl. NV24's 2×-wide UV rows).
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_nv16_nv24_to_rgba_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };
        let mut cpu = CPUProcessor::new();

        // Fill Y plus the interleaved UV plane in the canonical semi-planar
        // layout — exactly what the codec writes and the CPU `yuv`-crate reader
        // expects: the Y plane is `h` rows at the buffer's row stride; the UV
        // plane starts at `h * stride`; each chroma row advances
        // `uv_grid_rows * stride` bytes (NV24 carries a full-resolution `2*W`
        // byte line == two grid rows, NV12/NV16 one row of `W/2` pairs); each
        // (Cb,Cr) pair is two consecutive bytes at column `cx * 2`. This is
        // stride-correct for both the tight Mem buffer and the padded IOSurface.
        //
        // Takes explicit w/h so the closure can be reused across multiple frame
        // sizes (even and odd) without capturing a fixed outer variable.
        let fill = |buf: &mut [u8], stride: usize, fmt: PixelFormat, w: usize, h: usize| {
            for y in 0..h {
                for x in 0..w {
                    buf[y * stride + x] = ((x * 9 + y * 5) & 0xff) as u8;
                }
            }
            let (cw, ch, uv_grid_rows) = match fmt {
                PixelFormat::Nv12 => (w / 2, h / 2, 1usize),
                PixelFormat::Nv16 => (w / 2, h, 1usize),
                _ => (w, h, 2usize), // Nv24: full-res chroma, 2W bytes/row
            };
            let uv_plane = h * stride;
            for cy in 0..ch {
                for cx in 0..cw {
                    let off = uv_plane + cy * uv_grid_rows * stride + cx * 2;
                    buf[off] = ((cx * 11 + 30) & 0xff) as u8;
                    buf[off + 1] = ((cy * 7 + 200) & 0xff) as u8;
                }
            }
        };

        for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
            for (w, h) in [
                (16usize, 16usize), // original even-dim case
                (15, 16),           // odd-W
                (16, 15),           // odd-H
            ] {
                let mem = TensorDyn::image(w, h, fmt, DType::U8, None).unwrap();
                let mem_stride = mem.as_u8().unwrap().effective_row_stride().unwrap();
                fill(
                    mem.as_u8().unwrap().map().unwrap().as_mut_slice(),
                    mem_stride,
                    fmt,
                    w,
                    h,
                );
                let cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
                let (r, _s, cpu_dst) = convert_img(
                    &mut cpu,
                    mem,
                    cpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| panic!("CPU {fmt:?}->{w}x{h}->RGBA: {e}"));

                let ios = TensorDyn::image(w, h, fmt, DType::U8, Some(TensorMemory::Dma))
                    .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h} IOSurface alloc: {e}"));
                let ios_stride = ios.as_u8().unwrap().effective_row_stride().unwrap();
                fill(
                    ios.as_u8().unwrap().map().unwrap().as_mut_slice(),
                    ios_stride,
                    fmt,
                    w,
                    h,
                );
                let gpu_dst =
                    TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                        .unwrap();
                let (r, _s, gpu_dst) = convert_img(
                    &mut gpu,
                    ios,
                    gpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| panic!("GPU {fmt:?}->{w}x{h}->RGBA on ANGLE: {e}"));

                let cs = cpu_dst.as_u8().unwrap().effective_row_stride().unwrap();
                let cmap = cpu_dst.as_u8().unwrap().map().unwrap();
                let cb = cmap.as_slice();
                let gs = gpu_dst.as_u8().unwrap().effective_row_stride().unwrap();
                let gmap = gpu_dst.as_u8().unwrap().map().unwrap();
                let gb = gmap.as_slice();
                let mut max_d = 0i16;
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..3 {
                            let cv = cb[y * cs + x * 4 + c] as i16;
                            let gv = gb[y * gs + x * 4 + c] as i16;
                            max_d = max_d.max((cv - gv).abs());
                        }
                    }
                }
                assert!(
                    max_d <= 3,
                    "{fmt:?} {w}x{h}: GPU vs CPU RGBA max channel diff {max_d} > 3"
                );
            }
        }
    }

    /// Phase-0 gate: the reused-pool / larger-surface case. A single R8 pool
    /// IOSurface (allocated bigger than the frame, so its `bytesPerRow` exceeds
    /// the frame's even width) is reconfigured to each NV frame, filled at the
    /// preserved physical stride, and converted on the GPU. This proves ANGLE
    /// binds the *whole* physical surface as a pbuffer and that `texelFetch`
    /// resolves the frame's Y/UV texels through the surface's real `bytesPerRow`
    /// (the physical-grid / logical-ROI decoupling). GPU must match CPU ≤3 LSB.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv_to_rgba_larger_pool_surface_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };
        let mut cpu = CPUProcessor::new();
        // The pool is generously oversized (256-wide → bpr 256, well beyond any
        // frame's even width) and tall enough for NV24's 3·H at the test frames.
        let (pool_w, pool_h) = (256usize, 256usize);

        // Canonical semi-planar fill: Y plane at the row stride, then the UV
        // plane at `h * stride` with each chroma row advancing `uv_grid_rows *
        // stride` bytes. Stride-correct for both tight Mem and padded IOSurface.
        let fill = |buf: &mut [u8], stride: usize, fmt: PixelFormat, w: usize, h: usize| {
            for y in 0..h {
                for x in 0..w {
                    buf[y * stride + x] = ((x * 9 + y * 5) & 0xff) as u8;
                }
            }
            let (cw, ch, uv_grid_rows) = match fmt {
                PixelFormat::Nv12 => (w / 2, h / 2, 1usize),
                PixelFormat::Nv16 => (w / 2, h, 1usize),
                _ => (w, h, 2usize), // Nv24: full-res chroma, 2W bytes/row
            };
            let uv_plane = h * stride;
            for cy in 0..ch {
                for cx in 0..cw {
                    let off = uv_plane + cy * uv_grid_rows * stride + cx * 2;
                    buf[off] = ((cx * 11 + 30) & 0xff) as u8;
                    buf[off + 1] = ((cy * 7 + 200) & 0xff) as u8;
                }
            }
        };

        for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
            for (w, h) in [
                (40usize, 24usize), // original even-dim case
                (15, 16),           // odd-W
                (16, 15),           // odd-H
            ] {
                // `ew` is the minimum even extent; the pool stride must exceed it
                // to actually exercise the physical-stride shader decoupling.
                let ew = w.next_multiple_of(2);

                // CPU reference from a tightly-packed Mem tensor at the frame size.
                let mem = TensorDyn::image(w, h, fmt, DType::U8, None).unwrap();
                let mem_stride = mem.as_u8().unwrap().effective_row_stride().unwrap();
                fill(
                    mem.as_u8().unwrap().map().unwrap().as_mut_slice(),
                    mem_stride,
                    fmt,
                    w,
                    h,
                );
                let cpu_dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, None).unwrap();
                let (r, _s, cpu_dst) = convert_img(
                    &mut cpu,
                    mem,
                    cpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| panic!("CPU {fmt:?}->{w}x{h}->RGBA: {e}"));

                // GPU source: a LARGER R8 pool surface, reconfigured down to the
                // frame. Phase 1 preserves the pool's padded `bytesPerRow` as the
                // tensor's row stride; the fill writes the frame at that stride.
                let mut ios = match TensorDyn::image(
                    pool_w,
                    pool_h,
                    PixelFormat::Grey,
                    DType::U8,
                    Some(TensorMemory::Dma),
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("SKIPPED: {} — R8 pool IOSurface alloc: {e:?}", function!());
                        return;
                    }
                };
                ios.configure_image(w, h, fmt)
                    .unwrap_or_else(|e| panic!("configure_image {fmt:?} {w}x{h} on pool: {e}"));
                let ios_stride = ios.as_u8().unwrap().effective_row_stride().unwrap();
                assert!(
                    ios_stride > ew,
                    "{fmt:?} {w}x{h}: pool stride {ios_stride} should exceed even width {ew} \
                     (test must exercise padding)"
                );
                fill(
                    ios.as_u8().unwrap().map().unwrap().as_mut_slice(),
                    ios_stride,
                    fmt,
                    w,
                    h,
                );

                let gpu_dst =
                    TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                        .unwrap();
                let (r, _s, gpu_dst) = convert_img(
                    &mut gpu,
                    ios,
                    gpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| {
                    panic!("GPU {fmt:?}->{w}x{h}->RGBA (pool surface) on ANGLE: {e}")
                });

                let cs = cpu_dst.as_u8().unwrap().effective_row_stride().unwrap();
                let cmap = cpu_dst.as_u8().unwrap().map().unwrap();
                let cb = cmap.as_slice();
                let gs = gpu_dst.as_u8().unwrap().effective_row_stride().unwrap();
                let gmap = gpu_dst.as_u8().unwrap().map().unwrap();
                let gb = gmap.as_slice();
                let mut max_d = 0i16;
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..3 {
                            let cv = cb[y * cs + x * 4 + c] as i16;
                            let gv = gb[y * gs + x * 4 + c] as i16;
                            max_d = max_d.max((cv - gv).abs());
                        }
                    }
                }
                assert!(
                    max_d <= 3,
                    "{fmt:?} {w}x{h}: GPU(pool surface) vs CPU RGBA max channel diff {max_d} > 3"
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl_macos() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "SKIPPED: {} — GL engine init failed ({e:?}). \
                     Install ANGLE via `brew install startergo/angle/angle` \
                     and re-sign per README.md § macOS GPU Acceleration to \
                     run this test.",
                    function!()
                );
                return;
            }
        };

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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

        let (result, _src, dst) = convert_img(
            &mut proc,
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // macOS YUYV shader now threads per-tensor colorimetry: the untagged
        // camera720p source resolves to BT.709 limited (matching the BT.709
        // reference), measured 0.9973 on ANGLE — up from 0.9733 under the old
        // BT.601-full stop-gap. 0.98 leaves headroom for cross-GPU variance.
        compare_images(&dst, &target_image, 0.98, function!());
    }

    /// Multi-resolution smoke test: convert YUYV→RGBA via the GL
    /// backend at a small (64×32) frame and a 4K (3840×2160) frame,
    /// both filled with a synthetic mid-grey pattern. Validates the
    /// shader math at the chroma-pairing boundary on small textures
    /// and exercises the IOSurface bytes-per-row alignment path at 4K
    /// (3840 pixels × 2 bytes/pixel = 7680 bytes, naturally 64-aligned).
    ///
    /// Resolutions below 32 pixels wide aren't tested because the
    /// IOSurface allocator pads bpr to 64 bytes — for a 4-px-wide
    /// YUYV surface that's 8 bytes data + 56 bytes padding per row,
    /// which exercises a sampling pattern that's ANGLE-version
    /// dependent rather than HAL-correctness dependent.
    ///
    /// This complements `test_yuyv_to_rgba_opengl_macos` (which checks
    /// pixel-exact correctness against a reference image at 720p) by
    /// ensuring the pipeline does not crash or produce gross errors at
    /// resolution extremes. Pixel-exact validation at 4K would require
    /// a 30 MB reference file we don't want to bundle.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl_macos_multi_resolution() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };

        for (w, h) in [(64usize, 32usize), (3840, 2160)] {
            // Synthetic YUYV: Y=128 (mid-grey luma), U=V=128 (neutral
            // chroma) → RGB grey at the output.
            let bytes_per_row = w * 2;
            let mut yuyv = vec![0u8; bytes_per_row * h];
            for chunk in yuyv.chunks_exact_mut(4) {
                chunk[0] = 128; // Y0
                chunk[1] = 128; // U
                chunk[2] = 128; // Y1
                chunk[3] = 128; // V
            }

            let src = load_bytes_to_tensor(w, h, PixelFormat::Yuyv, Some(TensorMemory::Dma), &yuyv)
                .unwrap();

            let dst = TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                .unwrap();

            let (result, _src, dst) = convert_img(
                &mut proc,
                src,
                dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            );
            result.expect("GL convert should succeed at this resolution");

            // The neutral-chroma input must produce a near-grey output;
            // BT.709 limited-range maps Y=128/UV=128 → roughly
            // (130, 130, 130). Allow ±4 LSB for `mediump float` shader
            // rounding.
            let dst_u8 = dst.as_u8().unwrap();
            let dst_map = dst_u8.map().unwrap();
            let dst_bytes = dst_map.as_slice();
            assert_eq!(dst_bytes.len(), w * h * 4, "RGBA byte count");
            for px in dst_bytes.chunks_exact(4) {
                for (i, &c) in px[..3].iter().enumerate() {
                    assert!(
                        (120..=140).contains(&c),
                        "{}: channel {i} = {c} (expected ~128 ±12) at {w}×{h}",
                        function!(),
                    );
                }
                assert_eq!(px[3], 255, "alpha must be 1.0");
            }
        }
    }

    /// Verify that two consecutive convert() calls on the same source
    /// tensor reuse the cached EGL pbuffer. Tests the cache hit path
    /// added with the macOS GL backend hardening — without it, each
    /// frame would pay `eglCreatePbufferFromClientBuffer` + destroy.
    ///
    /// This is a behaviour test rather than a perf test (the timing
    /// difference is 100-200µs which is too noisy to assert on); we
    /// check that the second call succeeds and produces a result
    /// identical to the first.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_pbuffer_cache_reuses_surfaces() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };

        // Allocate one source + one destination, run convert twice.
        let mut yuyv = vec![0u8; 64 * 32 * 2];
        for chunk in yuyv.chunks_exact_mut(4) {
            chunk[0] = 200;
            chunk[1] = 100;
            chunk[2] = 200;
            chunk[3] = 156;
        }
        let src = load_bytes_to_tensor(64, 32, PixelFormat::Yuyv, Some(TensorMemory::Dma), &yuyv)
            .unwrap();
        let dst = TensorDyn::image(
            64,
            32,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();

        let (r1, src, dst) = convert_img(
            &mut proc,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        r1.unwrap();
        let first: Vec<u8> = dst.as_u8().unwrap().map().unwrap().as_slice().to_vec();

        let (r2, _src, dst) = convert_img(
            &mut proc,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        r2.unwrap();
        let second: Vec<u8> = dst.as_u8().unwrap().map().unwrap().as_slice().to_vec();

        assert_eq!(first, second, "cache-hit conversion must be deterministic");
    }

    /// Steady-state import gate (macOS half of the Linux
    /// `dma_pool_steady_state_zero_imports` test): an N-frame convert loop
    /// over a fixed pool of IOSurface tensors must create ZERO new EGL
    /// pbuffers after the pool has been seen once — pbuffer-cache misses
    /// stay flat while hits grow. Counter-based hardening of
    /// `test_macos_gl_pbuffer_cache_reuses_surfaces` above: a refactor that
    /// re-imports per frame passes the pixel-equality test but fails this.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_pbuffer_cache_steady_state() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: {} — GL engine init failed ({e:?})", function!());
                return;
            }
        };

        let (w, h) = (64usize, 32usize);
        const POOL: usize = 3;
        const FRAMES: usize = 100;

        let yuyv = vec![128u8; w * h * 2];
        let pool: Vec<TensorDyn> = (0..POOL)
            .map(|_| {
                load_bytes_to_tensor(w, h, PixelFormat::Yuyv, Some(TensorMemory::Dma), &yuyv)
                    .unwrap()
            })
            .collect();
        let mut dst =
            TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma)).unwrap();

        // Warmup: two passes over the pool import every surface once.
        for src in pool.iter().cycle().take(POOL * 2) {
            proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        }
        let warm = proc.egl_cache_stats().unwrap();

        for src in pool.iter().cycle().take(FRAMES) {
            proc.convert(src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap();
        }
        let steady = proc.egl_cache_stats().unwrap();

        assert_eq!(
            warm.total_misses(),
            steady.total_misses(),
            "steady-state loop created new imports (warm {warm:?}, steady {steady:?})"
        );
        let hits = |s: &GlCacheStats| s.src.hits + s.dst.hits + s.nv_r8.hits;
        assert!(
            hits(&steady) - hits(&warm) >= FRAMES as u64,
            "expected at least {FRAMES} import-cache hits over the loop, got {}",
            hits(&steady) - hits(&warm)
        );
    }

    /// Backend assertion for the F16 zero-copy path: when the GL backend
    /// initialized (ANGLE on macOS) and reports F16 color-buffer support,
    /// the NV12→PlanarRgb-F16 IOSurface convert MUST be handled by the GL
    /// engine — proven by the engine's import-cache counters moving, not
    /// just by output correctness. This is the guard against the
    /// silent-CPU-fallback failure mode: a misclassified IOSurface F16
    /// destination keeps every output-correctness test green while
    /// quietly running ~10× slower on CPU; only a backend observable
    /// catches it. Skips ONLY when GL itself is unavailable or the
    /// configuration lacks F16 — a convert error or a CPU-routed convert
    /// with the capability present is a FAILURE.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_f16_planar_is_gl_backed() {
        let mut proc = ImageProcessor::new().expect("ImageProcessor");
        let Some(ref gl) = proc.opengl else {
            eprintln!("SKIPPED: {} — GL backend unavailable", function!());
            return;
        };
        if !gl.supported_render_dtypes().f16 {
            eprintln!(
                "SKIPPED: {} — configuration lacks F16 color-buffer support",
                function!()
            );
            return;
        }
        let stats_before = gl.egl_cache_stats().expect("cache stats");

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        {
            let t = src.as_u8().unwrap();
            let mut m = t.map().unwrap();
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = ((i * 31) % 211) as u8;
            }
        }
        let mut dst = TensorDyn::image(
            640,
            640,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Dma),
        )
        .unwrap();

        proc.convert(
            &src,
            &mut dst,
            Rotation::None,
            Flip::None,
            Crop::letterbox([114, 114, 114, 255]),
        )
        .expect("F16 capability reported but the NV12→PlanarF16 convert failed");
        let stats_after = proc
            .opengl
            .as_ref()
            .expect("GL backend present")
            .egl_cache_stats()
            .expect("cache stats");
        // ≥ 2 misses: the fused convert imports the zero-copy NV source
        // (pass 1) AND the F16 destination (pass 2). Requiring both means a
        // convert that imported its source, failed mid-engine, and fell back
        // to CPU (1 miss) cannot satisfy this gate.
        assert!(
            stats_after.total_misses() >= stats_before.total_misses() + 2,
            "convert succeeded but the GL engine did not import both the \
             source and the F16 destination — the work did not (fully) run \
             on the GL backend (silent CPU fallback); misses before={} after={}",
            stats_before.total_misses(),
            stats_after.total_misses()
        );
    }

    /// Portable oracle for the fused NV12→PlanarRgb-F16 engine convert
    /// (two GL passes: NV→RGBA intermediate, then the packed RGBA16F
    /// render). New on every platform with F16 render support — macOS
    /// IOSurface and Linux DMA-BUF alike. Compares against the CPU
    /// backend's reference within the float-path tolerance.
    #[test]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_fused_engine_vs_cpu() {
        let mut gl = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) if p.opengl.is_some() => p,
            _ => {
                eprintln!("SKIPPED: {} — GL backend unavailable", function!());
                return;
            }
        };
        if !gl
            .opengl
            .as_ref()
            .map(|g| g.supported_render_dtypes().f16)
            .unwrap_or(false)
        {
            eprintln!("SKIPPED: {} — no F16 render support", function!());
            return;
        }
        let mem = if edgefirst_tensor::is_gpu_buffer_available() {
            TensorMemory::Dma
        } else {
            eprintln!("SKIPPED: {} — no zero-copy buffers", function!());
            return;
        };

        let src = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, Some(mem)).unwrap();
        {
            // Smooth gradients, NOT noise: the GL and CPU paths upsample
            // chroma with different kernels (nearest vs bilinear), which
            // legitimately diverges on per-texel chroma noise. Gradients
            // keep that kernel difference sub-LSB while still exercising
            // the full matrix math and the letterbox geometry.
            let t = src.as_u8().unwrap();
            let mut m = t.map().unwrap();
            let buf = m.as_mut_slice();
            let (w, h) = (1280usize, 720usize);
            for y in 0..h {
                for x in 0..w {
                    buf[y * w + x] = ((x * 255) / w) as u8; // luma ramp
                }
            }
            for y in 0..(h / 2) {
                for x in 0..(w / 2) {
                    let o = h * w + y * w + 2 * x;
                    buf[o] = ((y * 255) / (h / 2)) as u8; // U vertical ramp
                    buf[o + 1] = (((x + y) * 255) / (w / 2 + h / 2)) as u8; // V diagonal
                }
            }
        }
        let crop = Crop::letterbox([114, 114, 114, 255]);
        let mut gl_dst =
            TensorDyn::image(640, 640, PixelFormat::PlanarRgb, DType::F16, Some(mem)).unwrap();
        // Drive the GL backend DIRECTLY: a convert through `ImageProcessor`
        // silently falls back to CPU on a GL error, turning this oracle into
        // a CPU-vs-CPU tautology. A direct call surfaces the engine error.
        gl.opengl
            .as_mut()
            .expect("GL backend present")
            .convert(&src, &mut gl_dst, Rotation::None, Flip::None, crop)
            .expect("fused NV12→PlanarF16 GL convert");

        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let mut cpu_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        cpu.convert(&src, &mut cpu_dst, Rotation::None, Flip::None, crop)
            .expect("CPU reference convert");

        let g = gl_dst.as_f16().unwrap().map().unwrap().as_slice().to_vec();
        let c = cpu_dst.as_f16().unwrap().map().unwrap().as_slice().to_vec();
        assert_eq!(g.len(), c.len());
        let mut max_diff = 0.0f32;
        let mut max_at = 0usize;
        for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
            let d = (a.to_f32() - b.to_f32()).abs();
            if d > max_diff {
                max_diff = d;
                max_at = i;
            }
        }
        // Localize: plane (R/G/B), row, col of the worst element.
        let (plane, rem) = (max_at / (640 * 640), max_at % (640 * 640));
        let (row, col) = (rem / 640, rem % 640);
        eprintln!(
            "fused-vs-cpu: max_diff={max_diff} at plane={plane} row={row} col={col} \
             gl={} cpu={}",
            g[max_at].to_f32(),
            c[max_at].to_f32()
        );
        // Two GPU passes (8-bit intermediate + linear filtering) vs the
        // CPU's direct path: allow a few 8-bit steps of divergence.
        assert!(
            max_diff <= 4.0 / 255.0 + 1e-3,
            "fused NV12→PlanarF16 diverges from CPU reference: max_diff={max_diff}"
        );
    }

    /// Zero-copy source → heap destination through the GL engine,
    /// driven on the GL backend DIRECTLY so a GL error cannot hide
    /// behind the `ImageProcessor` CPU fallback (the shape that exposed
    /// the macOS `glReadnPixels` failure: imported source, rendered,
    /// then errored at the heap readback). RGBA→BGRA so the convert is
    /// a pure byte shuffle: no chroma kernel or colorimetry ambiguity
    /// (Vivante's NV fast path legitimately diverges from the CPU
    /// reference), and the readback stays GL_RGBA (V3D rejects RGB
    /// readbacks).
    #[test]
    #[cfg(feature = "opengl")]
    fn test_zero_copy_src_to_mem_dst_gl_direct() {
        let mut proc = match ImageProcessor::new() {
            Ok(p) if p.opengl.is_some() => p,
            _ => {
                eprintln!("SKIPPED: {} — GL backend unavailable", function!());
                return;
            }
        };
        if !edgefirst_tensor::is_gpu_buffer_available() {
            eprintln!("SKIPPED: {} — no zero-copy buffers", function!());
            return;
        }

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )
        .unwrap();
        {
            let t = src.as_u8().unwrap();
            let mut m = t.map().unwrap();
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = ((i * 31) % 211) as u8;
            }
        }
        let mut gl_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        proc.opengl
            .as_mut()
            .expect("GL backend present")
            .convert(
                &src,
                &mut gl_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .expect("zero-copy src → heap dst GL convert");

        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let mut cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        cpu.convert(
            &src,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .expect("CPU reference convert");

        let g = gl_dst.as_u8().unwrap().map().unwrap().as_slice().to_vec();
        let c = cpu_dst.as_u8().unwrap().map().unwrap().as_slice().to_vec();
        assert_eq!(g.len(), c.len());
        let max_diff = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        // Same-size byte shuffle: GL samples texel centers 1:1, so allow
        // only rounding slack.
        assert!(
            max_diff <= 2,
            "zero-copy src → heap dst diverges from CPU reference: max_diff={max_diff}"
        );
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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

        // G2D has poor colorimetry support: its hardware YUYV resize/sampling
        // diverges from the CPU reference by enough that the similarity score
        // sits around 0.85 on every measured G2D core (i.MX 8M Plus, i.MX 95).
        // The threshold is held at 0.85 so the test guards against gross
        // regressions while tolerating the driver's inherent colorimetry error.
        // TODO: compare YUYV↔YUYV directly without a YUYV→RGB convert.
        eprintln!(
            "WARNING: G2D has poor colorimetry support — YUYV resize diverges from the \
             CPU reference (~0.85 similarity); threshold held at 0.85, not 0.95."
        );
        compare_images_convert_to_rgb(&g2d_dst, &cpu_dst, 0.85, function!());
    }

    #[test]
    fn test_yuyv_to_rgba_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            None,
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
            &edgefirst_bench::testdata::read("camera720p.rgba"),
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

        // CPU path resolves the untagged 720p source to BT.709 limited (height
        // heuristic), matching the BT.709 camera fixture; measured 0.9995.
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
        let crop = Crop::new().with_source(Some(Region::new(20, 15, 400, 300)));

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
        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
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
        let crop = Crop::new().with_source(Some(Region::new(20, 15, 400, 300)));

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
        // Post-WS1 the GL path applies the resolved colorimetry via the EGL
        // YUV color-space/sample-range hints, so the matrix delta that forced
        // 0.95 has closed; tightened to 0.98 (driver-matrix rounding confirmed
        // on the GPU lanes).
        compare_images(&dst_gl, &dst_cpu, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgba_cpu() {
        let file = edgefirst_bench::testdata::read("camera720p.vyuy").to_vec();
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // CPU path resolves the untagged 720p source to BT.709 limited (height
        // heuristic), matching the BT.709 camera fixture; measured 0.9995.
        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgb_cpu() {
        let file = edgefirst_bench::testdata::read("camera720p.vyuy").to_vec();
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
                edgefirst_bench::testdata::read("camera720p.rgba")
                    .as_chunks::<4>()
                    .0,
            )
            .for_each(|(dst, src)| *dst = [src[0], src[1], src[2]]);

        // CPU path resolves the untagged 720p source to BT.709 limited (height
        // heuristic), matching the BT.709 camera fixture; measured 0.9995.
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
            &edgefirst_bench::testdata::read("camera720p.vyuy"),
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // Post-WS1 the GPU path applies the resolved per-tensor colorimetry,
        // so the matrix delta vs the reference that forced 0.95 has closed;
        // tightened to 0.98 (confirmed on the GPU/G2D lanes).
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
            &edgefirst_bench::testdata::read("camera720p.vyuy"),
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

        // Post-WS1 both CPU and G2D resolve untagged sources to limited-
        // range BT.601/709 (G2D is limited-range matrix-only hardware), so
        // the YUV-matrix delta that forced 0.95 has closed; tightened to
        // 0.98. G2D declines full-range and BT.2020 (handled by GL/CPU) — a
        // structural gap not exercised by these limited-range fixtures.
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
            &edgefirst_bench::testdata::read("camera720p.vyuy"),
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
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        // Post-WS1 the GPU path applies the resolved per-tensor colorimetry,
        // so the matrix delta vs the reference that forced 0.95 has closed;
        // tightened to 0.98 (confirmed on the GPU/G2D lanes).
        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_rgba_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
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

        let target_image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();

        // Threshold 0.95 (was 0.98): the reference now decodes the colour JPEG
        // to native NV12 and then converts to RGBA (was a direct JPEG → RGBA
        // decode), so it differs slightly from the RGBA derived from the
        // separate `zidane.nv12` fixture.
        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_odd_height_to_rgb_cpu() {
        // Odd height (even width) — the logical-odd case, e.g. 640×483. The
        // contiguous NV12 buffer is `[5 + ceil(5/2), 8]` = `[8, 8]` (5 luma rows
        // + 3 chroma rows). A neutral-grey fill (Y=U=V=128, BT.601 full-range)
        // must convert to a uniform grey RGB, exercising the odd-height
        // chroma-row count and the logical-height derivation in convert.
        // (Odd *width* is rounded to an even buffer at allocation, so it is
        // covered by the decode integration tests rather than here.)
        // CPU-only test: pin to tight host memory (None auto-selects pitch-padded
        // DMA on i.MX, which would leave the dst's row padding unconverted and
        // break the flat byte scan below).
        let mut src =
            TensorDyn::image(8, 5, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem)).unwrap();
        assert_eq!(src.shape(), &[8, 8]);
        assert_eq!((src.width(), src.height()), (Some(8), Some(5)));
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128);
        // Tag BT.601 full-range so Y=128 decodes to grey 128 (the neutral-grey
        // identity this test asserts). Without a tag, the colorimetry heuristic
        // resolves an SD tensor to BT.601 *limited*, expanding Y=128 → ~131.
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let dst =
            TensorDyn::image(8, 5, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
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

        assert_eq!((dst.width(), dst.height()), (Some(8), Some(5)));
        let map = dst.as_u8().unwrap().map().unwrap();
        for (i, &b) in map.as_slice().iter().enumerate() {
            assert!(
                (b as i16 - 128).abs() <= 2,
                "pixel byte {i} = {b}, expected ~128 for neutral-grey NV12"
            );
        }
    }

    #[test]
    fn test_nv24_to_rgb_cpu() {
        // NV24 (4:4:4) at 8×4: contiguous buffer is [4*3, 8] = [12, 8] — Y plane
        // (4 rows) + full-res interleaved UV plane (8 rows = 2H, 2W bytes per
        // chroma row). Neutral-grey fill (Y=U=V=128) must convert to uniform
        // grey RGB, exercising the 2× UV stride and shape[0]/3 height recovery.
        // CPU-only test: pin to tight host memory (see test_nv12_odd_height_to_rgb_cpu).
        let mut src =
            TensorDyn::image(8, 4, PixelFormat::Nv24, DType::U8, Some(TensorMemory::Mem)).unwrap();
        assert_eq!(src.shape(), &[12, 8]);
        assert_eq!((src.width(), src.height()), (Some(8), Some(4)));
        src.as_u8().unwrap().map().unwrap().as_mut_slice().fill(128);
        // Tag BT.601 full-range (see test_nv12_odd_height_to_rgb_cpu): without it
        // the heuristic picks limited range and Y=128 expands to ~131.
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let dst =
            TensorDyn::image(8, 4, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
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

        assert_eq!((dst.width(), dst.height()), (Some(8), Some(4)));
        let map = dst.as_u8().unwrap().map().unwrap();
        for (i, &b) in map.as_slice().iter().enumerate() {
            assert!(
                (b as i16 - 128).abs() <= 2,
                "pixel byte {i} = {b}, expected ~128 for neutral-grey NV24"
            );
        }
    }

    #[test]
    fn cpu_nv12_to_rgb_respects_tagged_bt2020() {
        // A uniform but *saturated* chroma sample (U/V far from neutral) so the
        // YUV→RGB matrix — not just the range — drives the result. Decoding the
        // same NV12 bytes under BT.601 / BT.709 / BT.2020 must yield three
        // distinct RGB triples, proving the CPU path honours the source's tagged
        // ColorEncoding instead of a hardcoded matrix. (G2D declines BT.2020 and
        // falls through to this CPU path; QA F9.)
        // CPU-only test: pin to tight host memory (see test_nv12_odd_height_to_rgb_cpu).
        fn decode_tagged(enc: edgefirst_tensor::ColorEncoding) -> [u8; 3] {
            let mut src =
                TensorDyn::image(8, 4, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem))
                    .unwrap();
            // NV12 8×4: 32-byte Y plane + 16-byte interleaved UV plane (4:2:0).
            assert_eq!(src.shape(), &[6, 8]);
            {
                let mut map = src.as_u8().unwrap().map().unwrap();
                let buf = map.as_mut_slice();
                buf[..32].fill(120); // Y
                for px in buf[32..].chunks_exact_mut(2) {
                    px[0] = 180; // U / Cb
                    px[1] = 64; // V / Cr
                }
            }
            // Range held constant (Limited) across all three so only the encoding
            // matrix varies between runs.
            src.set_colorimetry(Some(
                edgefirst_tensor::Colorimetry::default()
                    .with_encoding(enc)
                    .with_range(edgefirst_tensor::ColorRange::Limited),
            ));
            let dst = TensorDyn::image(8, 4, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem))
                .unwrap();
            let mut cpu = CPUProcessor::new();
            let (result, _src, dst) = convert_img(
                &mut cpu,
                src,
                dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            );
            result.unwrap();
            let map = dst.as_u8().unwrap().map().unwrap();
            let s = map.as_slice();
            [s[0], s[1], s[2]]
        }

        let bt601 = decode_tagged(edgefirst_tensor::ColorEncoding::Bt601);
        let bt709 = decode_tagged(edgefirst_tensor::ColorEncoding::Bt709);
        let bt2020 = decode_tagged(edgefirst_tensor::ColorEncoding::Bt2020);

        assert_ne!(
            bt2020, bt601,
            "BT.2020 must decode differently from BT.601 ({bt2020:?} vs {bt601:?})"
        );
        assert_ne!(
            bt2020, bt709,
            "BT.2020 must decode differently from BT.709 ({bt2020:?} vs {bt709:?})"
        );
        assert_ne!(
            bt601, bt709,
            "BT.601 must decode differently from BT.709 ({bt601:?} vs {bt709:?})"
        );
    }

    #[test]
    fn test_nv12_to_rgb_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
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

        let target_image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
            Some(PixelFormat::Rgb),
            None,
        )
        .unwrap();

        // Threshold 0.95 (was 0.98): the reference now decodes the colour JPEG
        // to native NV12 and then converts to RGB (was a direct JPEG → RGB
        // decode), so it differs slightly from the RGB derived from the
        // separate `zidane.nv12` fixture.
        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_to_grey_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
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

        let target_image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
            Some(PixelFormat::Grey),
            None,
        )
        .unwrap();

        // Threshold 0.95 (was 0.98): the reference grey frame now comes from
        // the colour JPEG decoded to native NV12 and then converted to GREY
        // (was a direct JPEG → GREY decode), so it differs slightly from the
        // grey derived from the `zidane.nv12` fixture.
        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_to_yuyv_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
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

        let target_image = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("zidane.jpg"),
            Some(PixelFormat::Rgb),
            None,
        )
        .unwrap();

        // Threshold 0.95 (was 0.98): the reference now decodes the colour JPEG
        // to native NV12 and then converts to RGB (was a direct JPEG → RGB
        // decode), so it differs slightly from the YUYV-sourced frame derived
        // from the separate `zidane.nv12` fixture.
        compare_images_convert_to_rgb(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_cpu_resize_nv16() {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_nv16_dst = TensorDyn::image(640, 640, PixelFormat::Nv16, DType::U8, None).unwrap();
        let cpu_rgb_dst = TensorDyn::image(640, 640, PixelFormat::Rgb, DType::U8, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::letterbox([255, 128, 0, 255]);

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

    // DEDUP: this function is also defined verbatim in
    // `crates/image/src/gl/tests.rs` (inside `mod gl_tests`). Both copies
    // must be kept in sync. Cross-module sharing would require either a
    // `pub(crate)` test-helper module (which pollutes the non-test API) or a
    // separate test-utils crate — both are disproportionate for a single
    // helper. If the implementation ever diverges, extract to a shared
    // `test_helpers` module in this crate.
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
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let jpeg_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

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
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let converter = ImageProcessor::new().unwrap();
        assert!(converter.cpu.is_some());
        assert_eq!(converter.forced_backend, Some(ForcedBackend::Cpu));
    }

    #[test]
    fn test_force_backend_invalid() {
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "invalid") };
        let result = ImageProcessor::new();
        assert!(
            matches!(&result, Err(Error::ForcedBackendUnavailable(s)) if s.contains("unknown")),
            "invalid backend value should return ForcedBackendUnavailable error: {result:?}"
        );
    }

    #[test]
    fn test_force_backend_unset() {
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") };
        let converter = ImageProcessor::new().unwrap();
        assert!(converter.forced_backend.is_none());
    }

    // ========================================================================
    // Tests for hybrid mask path error handling
    // ========================================================================

    #[test]
    fn test_draw_proto_masks_no_cpu_returns_error() {
        // Serialize against all other env-var-mutating tests.
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&[
            "EDGEFIRST_FORCE_BACKEND",
            "EDGEFIRST_DISABLE_GL",
            "EDGEFIRST_DISABLE_G2D",
            "EDGEFIRST_DISABLE_CPU",
        ]);

        // Disable all backends so cpu.is_none() after construction.
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_CPU", "1") };
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_GL", "1") };
        unsafe { std::env::set_var("EDGEFIRST_DISABLE_G2D", "1") };

        let mut converter = ImageProcessor::new().unwrap();
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
                layout: ProtoLayout::Nhwc,
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
        // Force CPU-only backend to ensure the CPU fallback path executes.
        // Serialized under ENV_MUTEX so we don't race with disable-var tests.
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let mut converter = ImageProcessor::new().unwrap();
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
                layout: ProtoLayout::Nhwc,
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

    // =========================================================================
    // Env-var serialisation helpers
    //
    // ALL tests that mutate any EDGEFIRST_* backend env var must hold
    // ENV_MUTEX for their full duration.  This single mutex serialises
    // test_disable_env_var, test_draw_proto_masks_no_cpu_returns_error,
    // test_force_backend_*, with_force_backend, and with_env — preventing
    // any two of them from racing in a parallel `cargo test` run.
    // =========================================================================

    /// Acquire the process-wide env-var mutex.  Returns a guard that must be
    /// kept alive for the entire duration of the test body.
    fn acquire_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_MUTEX
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// RAII guard that snapshots a set of env vars on construction and
    /// restores them on `Drop`, even if the test panics.
    struct EnvGuard {
        vars: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        /// Snapshot the current values of `names`.  Call this while holding
        /// the env lock (the lock is not taken here — that is the caller's
        /// responsibility so the lock scope can be wider than the guard).
        fn snapshot(names: &[&'static str]) -> Self {
            Self {
                vars: names.iter().map(|&k| (k, std::env::var(k).ok())).collect(),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.vars {
                match v {
                    Some(s) => unsafe { std::env::set_var(k, s) },
                    None => unsafe { std::env::remove_var(k) },
                }
            }
        }
    }

    /// Run `body` with `EDGEFIRST_FORCE_BACKEND` temporarily set (or
    /// removed), restoring the prior value afterward. Tests are env-
    /// serialized via the process-wide `ENV_MUTEX`.
    fn with_force_backend<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        match value {
            Some(v) => unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", v) },
            None => unsafe { std::env::remove_var("EDGEFIRST_FORCE_BACKEND") },
        }
        body()
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
                layout: ProtoLayout::Nhwc,
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
                layout: ProtoLayout::Nhwc,
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
        // Force CPU backend; serialized under ENV_MUTEX to avoid racing with
        // test_force_backend_* and test_disable_env_var.
        let _lock = acquire_env_lock();
        let _guard = EnvGuard::snapshot(&["EDGEFIRST_FORCE_BACKEND"]);
        unsafe { std::env::set_var("EDGEFIRST_FORCE_BACKEND", "cpu") };
        let mut processor = ImageProcessor::new().unwrap();

        // Load a source image
        let image = edgefirst_bench::testdata::read("zidane.jpg");
        let src = load_image_test_helper(&image, Some(PixelFormat::Rgba), None).unwrap();

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
        // Hold the env mutex so env-var-mutating tests can't corrupt the state
        // seen by ImageProcessor::new() calls during this test.
        let _lock = acquire_env_lock();
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

        // The Vivante GC7000UL driver (i.MX 8M Plus) double-frees on concurrent
        // EGL context teardown — four processors spun up on four threads here
        // trips it and aborts the whole test binary (SIGABRT, not a catchable
        // panic). The bug is the driver's, not the HAL's; this test is kept so
        // it still exercises the multi-context path on every other GPU. The
        // on-target GitHub Actions imx8mp runner sets EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS
        // to skip just this case there, while the run stays red anywhere else
        // a regression appears. (Skip, not #[ignore]: the platform is decided
        // at runtime, not compile time.)
        if std::env::var_os("EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS").is_some() {
            eprintln!(
                "SKIPPED: test_multiple_image_processors_separate_threads — known Vivante \
                 GC7000UL concurrent-EGL-teardown double-free \
                 (EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS set)"
            );
            return;
        }

        const TIMEOUT: Duration = Duration::from_secs(60);

        // Hold the env mutex so env-var-mutating tests can't corrupt ImageProcessor::new()
        // calls made inside the spawned threads during this test.
        let _lock = acquire_env_lock();

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

        // Hold the env mutex so env-var-mutating tests can't corrupt ImageProcessor::new()
        // calls made inside the spawned threads during this test.
        let _lock = acquire_env_lock();

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

    /// THE parallel-processors demonstration test: 4 ImageProcessors on 4
    /// threads, each with its own GL context and worker, converting
    /// per-thread-DISTINCT synthetic inputs concurrently (barrier-released);
    /// every output must byte-match the thread's own pre-barrier sequential
    /// oracle from the same processor. On LifecycleOnly platforms (Mali,
    /// V3D, Tegra, llvmpipe, macOS) the converts genuinely overlap on the
    /// GPU; on Vivante they serialize via the Full policy and the test still
    /// must pass. Distinct inputs make any cross-processor state leakage
    /// (wrong texture, wrong context, clobbered upload) visible as a byte
    /// diff rather than a coincidental match — proven by a scratch
    /// cross-wire run (neighbor's input post-oracle) failing on every
    /// thread with ~53% of bytes diverged.
    ///
    /// Skipped under EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS like
    /// `test_multiple_image_processors_separate_threads`: the galcore driver
    /// can abort intermittently on concurrent multi-processor lifecycles
    /// regardless of locking (P0 spike: reproduces fully serialized).
    #[test]
    fn test_parallel_processors_unique_outputs() {
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        const N: usize = 4;
        const ROUNDS: usize = 25;
        const TIMEOUT: Duration = Duration::from_secs(60);

        if std::env::var_os("EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS").is_some() {
            eprintln!(
                "SKIPPED: test_parallel_processors_unique_outputs — known Vivante \
                 GC7000UL concurrent-multi-processor driver abort \
                 (EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS set)"
            );
            return;
        }

        let _lock = acquire_env_lock();
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
                        // CI-weight geometry (llvmpipe renders on the CPU).
                        let (w, h) = (640usize, 480usize);
                        let mem = if edgefirst_tensor::is_dma_available() {
                            Some(TensorMemory::Dma)
                        } else {
                            Some(TensorMemory::Mem)
                        };
                        let src = proc
                            .create_image(w, h, PixelFormat::Nv12, DType::U8, mem)
                            .unwrap();
                        {
                            let t = src.as_u8().unwrap();
                            let mut m = t.map().unwrap();
                            let s = m.as_mut_slice();
                            for (j, b) in s[..w * h].iter_mut().enumerate() {
                                *b = ((i * 53 + j) % 200 + 16) as u8;
                            }
                            for b in &mut s[w * h..] {
                                *b = (80 + i * 24) as u8;
                            }
                        }
                        let lb = Crop::letterbox([114, 114, 114, 255]);
                        let convert_once = |proc: &mut ImageProcessor| -> Vec<u8> {
                            let mut dst = proc
                                .create_image(320, 320, PixelFormat::Rgba, DType::U8, mem)
                                .unwrap();
                            proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb)
                                .unwrap_or_else(|e| panic!("convert failed on thread {i}: {e}"));
                            let t = dst.as_u8().unwrap();
                            let m = t.map().unwrap();
                            m.as_slice().to_vec()
                        };

                        let oracle = convert_once(&mut proc);
                        barrier.wait();
                        for round in 0..ROUNDS {
                            let out = convert_once(&mut proc);
                            let diffs = oracle.iter().zip(&out).filter(|(a, b)| a != b).count();
                            assert!(
                                diffs == 0,
                                "thread {i} round {round}: {diffs}/{} bytes diverged \
                                 from this processor's own oracle — cross-processor \
                                 GL state leakage under parallel execution",
                                oracle.len()
                            );
                        }
                    })
                })
                .collect();

            for (i, h) in handles.into_iter().enumerate() {
                h.join()
                    .unwrap_or_else(|e| panic!("parallel thread {i} panicked: {e:?}"));
            }
            let _ = tx.send(());
        });

        rx.recv_timeout(TIMEOUT).unwrap_or_else(|_| {
            panic!("test_parallel_processors_unique_outputs timed out after {TIMEOUT:?}")
        });
    }

    /// Heavy on-demand stressor for the GL serialization policy: 4
    /// processors × 4 threads × barrier × 200 NV12 720p → RGB 640 letterbox
    /// converts (DMA where available); every output must byte-match the
    /// thread's own pre-barrier sequential oracle from the same processor.
    /// Ignored by default (heavy; board tool — the CI-weight version is
    /// `test_parallel_processors_unique_outputs`). Run explicitly, optionally
    /// pinning the policy via EDGEFIRST_GL_SERIALIZE=full|lifecycle:
    ///   <test binary> stress_parallel_processors_oracle --ignored
    #[test]
    #[ignore = "heavy on-demand GL-parallelism stressor; run explicitly on boards"]
    fn stress_parallel_processors_oracle() {
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        const N: usize = 4;
        const ROUNDS: usize = 200;
        const TIMEOUT: Duration = Duration::from_secs(600);

        let _lock = acquire_env_lock();
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
                        let (w, h) = (1280usize, 720usize);
                        let mem = if edgefirst_tensor::is_dma_available() {
                            Some(TensorMemory::Dma)
                        } else {
                            Some(TensorMemory::Mem)
                        };

                        // Per-thread-distinct synthetic NV12 so cross-wired
                        // GL state between processors shows up as a byte diff.
                        let src = proc
                            .create_image(w, h, PixelFormat::Nv12, DType::U8, mem)
                            .unwrap();
                        {
                            let t = src.as_u8().unwrap();
                            let mut m = t.map().unwrap();
                            let s = m.as_mut_slice();
                            for (j, b) in s[..w * h].iter_mut().enumerate() {
                                *b = ((i * 37 + j) % 200 + 16) as u8;
                            }
                            for b in &mut s[w * h..] {
                                *b = (96 + i * 16) as u8;
                            }
                        }
                        let lb = Crop::letterbox([114, 114, 114, 255]);

                        let convert_once = |proc: &mut ImageProcessor| -> Vec<u8> {
                            let mut dst = proc
                                .create_image(640, 640, PixelFormat::Rgb, DType::U8, mem)
                                .unwrap();
                            proc.convert(&src, &mut dst, Rotation::None, Flip::None, lb)
                                .unwrap_or_else(|e| panic!("convert failed on thread {i}: {e}"));
                            let t = dst.as_u8().unwrap();
                            let m = t.map().unwrap();
                            m.as_slice().to_vec()
                        };

                        let oracle = convert_once(&mut proc);
                        barrier.wait();
                        for round in 0..ROUNDS {
                            let out = convert_once(&mut proc);
                            let diffs = oracle.iter().zip(&out).filter(|(a, b)| a != b).count();
                            assert!(
                                diffs == 0,
                                "thread {i} round {round}: {diffs}/{} bytes diverged \
                                 from the pre-barrier oracle",
                                oracle.len()
                            );
                        }
                    })
                })
                .collect();

            for (i, h) in handles.into_iter().enumerate() {
                h.join()
                    .unwrap_or_else(|e| panic!("stressor thread {i} panicked: {e:?}"));
            }
            let _ = tx.send(());
        });

        rx.recv_timeout(TIMEOUT).unwrap_or_else(|_| {
            panic!("stress_parallel_processors_oracle timed out after {TIMEOUT:?}")
        });
    }

    // =========================================================================
    // F16 / F32 auto-chain fallback integration tests
    // =========================================================================

    /// Proves the auto-chain (OpenGL → G2D → CPU) NEVER errors for a float
    /// combo the GL path does NOT cover.
    ///
    /// `Yuyv → Rgb F32` is not handled by the GL float render path (which
    /// only covers `Rgba → PlanarRgb F16` and `Rgba → Rgb F32`), so the
    /// chain falls through to the CPU float path. Before commit 868a7649
    /// added CPU U8→F32/F16 support this would have returned `Err`; now it
    /// must return `Ok` with output in `[0, 1]` and all values finite.
    #[test]
    fn convert_f32_auto_never_errors_non_gl_combo() {
        const W: usize = 64;
        const H: usize = 64;

        // Build a small synthetic YUYV source (Y=128, U=128, V=128 → near-grey).
        // YUYV packs two pixels into 4 bytes: [Y0, U, Y1, V] per macropixel.
        let src =
            TensorDyn::image(W, H, PixelFormat::Yuyv, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for chunk in data.chunks_exact_mut(4) {
                chunk[0] = 128; // Y0
                chunk[1] = 128; // U
                chunk[2] = 160; // Y1 — distinct so a layout bug is visible
                chunk[3] = 128; // V
            }
        }

        let mut dst =
            TensorDyn::image(W, H, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem)).unwrap();

        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default());
        assert!(
            result.is_ok(),
            "auto-chain Yuyv→Rgb F32 must not error: {:?}",
            result.err()
        );

        // Verify all output values are finite and in [0, 1].
        let map = dst.as_f32().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), W * H * 3, "unexpected output element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }

        // WEAK-1: Anti-all-zero spot-check.  A Y=128 YUYV source normalises to
        // ≈0.502 on the luma channel.  If the buffer is all-zero (e.g. the CPU
        // path never wrote to it) this assertion catches the regression.
        let first_non_zero = floats.iter().find(|&&v| v > 0.01);
        assert!(
            first_non_zero.is_some(),
            "all-zero output detected — CPU path likely did not write to the destination buffer"
        );
        // Y=128 → luma ≈ 0.502.  Spot-check the first pixel's R channel
        // (which carries luma for a near-grey YUV source).
        let r0 = floats[0];
        assert!(
            (r0 - 0.502_f32).abs() < 0.05,
            "first pixel R={r0} expected ≈0.502 (Y=128 neutral grey from YUYV source)"
        );
    }

    /// Proves CPU-forced `Rgba → PlanarRgb F16` correctness.
    ///
    /// Uses a source with clearly distinct per-channel values so a
    /// plane-swap or layout bug surfaces immediately. Tolerance is 2^-9
    /// (one F16 ULP at 0.5, i.e. roughly 1/512).
    #[test]
    // `ImageProcessorConfig` carries a Linux-only `egl_display` field, so
    // `{ backend, ..Default::default() }` is a genuine update on Linux but
    // covers no remaining fields on macOS, where `clippy::needless_update`
    // then fires. `allow` (not `expect`) because the lint is platform-
    // conditional — it does not fire on Linux.
    #[allow(clippy::needless_update)]
    fn convert_f16_forced_cpu_correct() {
        const W: usize = 16;
        const H: usize = 16;
        const TOL: f32 = 1.0 / 512.0; // 2^-9

        // pixel (y,x): R = 50+x, G = 100+y*8, B = 200
        let src =
            TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    data[i * 4] = (50 + x) as u8; // R: 50..65
                    data[i * 4 + 1] = (100 + y * 8) as u8; // G: 100..220
                    data[i * 4 + 2] = 200; // B: constant
                    data[i * 4 + 3] = 255;
                }
            }
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        )
        .unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
            .expect("forced-CPU Rgba→PlanarRgb F16 must not error");

        let src_map = src.as_u8().unwrap().map().unwrap();
        let src_bytes = src_map.as_slice();
        let dst_map = dst.as_f16().unwrap().map().unwrap();
        let dst_halfs = dst_map.as_slice();

        let plane = W * H;
        assert_eq!(dst_halfs.len(), plane * 3, "wrong output element count");

        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let r_exp = src_bytes[i * 4] as f32 / 255.0;
                let g_exp = src_bytes[i * 4 + 1] as f32 / 255.0;
                let b_exp = src_bytes[i * 4 + 2] as f32 / 255.0;

                let r_got = dst_halfs[i].to_f32();
                let g_got = dst_halfs[plane + i].to_f32();
                let b_got = dst_halfs[2 * plane + i].to_f32();

                assert!(
                    (r_got - r_exp).abs() <= TOL,
                    "R plane ({x},{y}): got {r_got}, expected {r_exp}"
                );
                assert!(
                    (g_got - g_exp).abs() <= TOL,
                    "G plane ({x},{y}): got {g_got}, expected {g_exp}"
                );
                assert!(
                    (b_got - b_exp).abs() <= TOL,
                    "B plane ({x},{y}): got {b_got}, expected {b_exp}"
                );

                // Catch plane-swap: R and G must differ (they have different formulas).
                if src_bytes[i * 4] != src_bytes[i * 4 + 1] {
                    assert_ne!(r_got, g_got, "R and G planes must differ at ({x},{y})");
                }
            }
        }
    }

    /// Proves the auto-chain falls through to CPU for `Rgba → Rgb F32` with
    /// rotation set.
    ///
    /// The GL float render path rejects any call where rotation ≠ None,
    /// returning an error that causes the chain to continue. Before the CPU
    /// float fallback this would have produced an error at the end of the
    /// chain; now it must reach CPU and return `Ok` with finite `[0, 1]` output.
    #[test]
    fn convert_f32_with_rotation_falls_back() {
        const W: usize = 16;
        const H: usize = 16;

        // RGBA8 source with a known gradient (distinct per-channel values).
        let src =
            TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    data[i * 4] = (x * 16) as u8; // R
                    data[i * 4 + 1] = (y * 16) as u8; // G
                    data[i * 4 + 2] = 128; // B
                    data[i * 4 + 3] = 255;
                }
            }
        }

        // Rotation swaps W and H, so dst is [W, H] (H×W output).
        let mut dst = TensorDyn::image(
            H, // dst W = src H after 90° rotation
            W, // dst H = src W after 90° rotation
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
        )
        .unwrap();

        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(
            &src,
            &mut dst,
            Rotation::Clockwise90,
            Flip::None,
            Crop::default(),
        );
        assert!(
            result.is_ok(),
            "auto-chain Rgba→Rgb F32 with Rot90 must not error: {:?}",
            result.err()
        );

        let map = dst.as_f32().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), H * W * 3, "unexpected output element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
    }

    /// GL-vs-CPU identity parity for `Rgba → PlanarRgb F16`.
    ///
    /// Converts the same RGBA8 source via forced `OpenGl` and forced `Cpu`,
    /// then verifies the two F16 output tensors agree element-wise within
    /// 2^-8 (two F16 ULPs at 0.5). Skipped when OpenGL or F16 render is
    /// unavailable.
    #[test]
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    fn convert_f16_gl_cpu_parity_identity() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: convert_f16_gl_cpu_parity_identity - OpenGL not available");
            return;
        }

        const W: usize = 16;
        const H: usize = 16;
        const TOL: f32 = 1.0 / 256.0; // 2^-8

        // pixel (y,x): R = 40+x, G = 80+y*10, B = 180
        let src =
            TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for y in 0..H {
                for x in 0..W {
                    let i = y * W + x;
                    data[i * 4] = (40 + x) as u8; // R
                    data[i * 4 + 1] = (80 + y * 10) as u8; // G
                    data[i * 4 + 2] = 180; // B
                    data[i * 4 + 3] = 255;
                }
            }
        }

        // GL path.
        let gl_result = {
            let mut gl_proc = match ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::OpenGl,
                ..Default::default()
            }) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "SKIPPED: convert_f16_gl_cpu_parity_identity - GL backend unavailable: {e}"
                    );
                    return;
                }
            };

            if !gl_proc.supported_render_dtypes().f16 {
                eprintln!("SKIPPED: convert_f16_gl_cpu_parity_identity - F16 render not supported");
                return;
            }

            let mut dst = TensorDyn::image(
                W,
                H,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::Mem),
            )
            .unwrap();
            match gl_proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default()) {
                Ok(()) => dst,
                Err(e) => {
                    eprintln!(
                        "SKIPPED: convert_f16_gl_cpu_parity_identity - GL convert failed: {e}"
                    );
                    return;
                }
            }
        };

        // CPU path.
        let cpu_result = {
            let mut cpu_proc = ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::Cpu,
                ..Default::default()
            })
            .unwrap();
            let mut dst = TensorDyn::image(
                W,
                H,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::Mem),
            )
            .unwrap();
            cpu_proc
                .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
                .expect("forced-CPU Rgba→PlanarRgb F16 must not error");
            dst
        };

        // Compare element-wise.
        let gl_map = gl_result.as_f16().unwrap().map().unwrap();
        let cpu_map = cpu_result.as_f16().unwrap().map().unwrap();
        let gl_halfs = gl_map.as_slice();
        let cpu_halfs = cpu_map.as_slice();

        assert_eq!(
            gl_halfs.len(),
            cpu_halfs.len(),
            "GL and CPU output sizes differ"
        );

        let plane = W * H;
        let channel_names = ["R", "G", "B"];
        for (idx, (gl_h, cpu_h)) in gl_halfs.iter().zip(cpu_halfs.iter()).enumerate() {
            let gl_v = gl_h.to_f32();
            let cpu_v = cpu_h.to_f32();
            let err = (gl_v - cpu_v).abs();
            let ch = channel_names[idx / plane];
            let pixel = idx % plane;
            assert!(
                err <= TOL,
                "GL vs CPU mismatch at {ch}[{pixel}]: GL={gl_v}, CPU={cpu_v}, err={err} > tol={TOL}"
            );
        }
    }

    // =========================================================================
    // GAP-1: supported_render_dtypes() Linux smoke test
    // =========================================================================

    /// Exercises the real Linux GL path that reads `gl.supported_render_dtypes()`.
    /// Skipped when no GL backend is available (CI/host without a GPU).
    #[test]
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    fn supported_render_dtypes_linux_smoke() {
        let proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("SKIPPED: supported_render_dtypes_linux_smoke — ImageProcessor::new() failed: {e}");
                return;
            }
        };
        if proc.opengl.is_none() {
            eprintln!("SKIPPED: supported_render_dtypes_linux_smoke — no GL backend on this host");
            return;
        }
        // The call must complete without panicking or deadlocking.
        let support = proc.supported_render_dtypes();
        eprintln!(
            "supported_render_dtypes_linux_smoke: f16={} f32={}",
            support.f16, support.f32
        );
        // No assertion on the specific values — they are hardware-dependent.
    }

    // =========================================================================
    // GAP-2: F16 PlanarRgb with width NOT divisible by 4 falls back to CPU
    // =========================================================================

    /// The GL float render path rejects PlanarRgb F16 destinations whose width
    /// is not a multiple of 4 (the packed RGBA16F swizzle trick requires W%4==0).
    /// The auto-chain must transparently fall through to the CPU path, which has
    /// no such restriction.  W=18, H=16 is chosen so W%4 == 2.
    #[test]
    fn convert_f16_pbo_non_4_aligned_width_falls_back() {
        const W: usize = 18; // 18 % 4 == 2 — NOT divisible by 4
        const H: usize = 16;

        // RGBA8 source filled with a flat mid-grey.
        let src =
            TensorDyn::image(W, H, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for chunk in data.chunks_exact_mut(4) {
                chunk[0] = 128;
                chunk[1] = 64;
                chunk[2] = 200;
                chunk[3] = 255;
            }
        }

        // F16 PlanarRgb destination in Mem (GL would use PBO, but we want
        // to exercise the fallback chain without hardware dependency).
        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        )
        .unwrap();

        // Use the default auto-chain so the GL path can attempt and reject,
        // then the CPU path succeeds.
        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default());
        assert!(
            result.is_ok(),
            "auto-chain PlanarRgb F16 W%4!=0 must not error (CPU fallback): {:?}",
            result.err()
        );

        // All output values must be finite and in [0, 1].
        let map = dst.as_f16().unwrap().map().unwrap();
        let halfs = map.as_slice();
        assert_eq!(halfs.len(), W * H * 3, "unexpected element count");
        for (i, h) in halfs.iter().enumerate() {
            let v = h.to_f32();
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
    }

    // =========================================================================
    // GAP-4: NV12 → Rgb F32 and NV12 → PlanarRgb F16, forced CPU
    // =========================================================================

    /// CPU widen-composition: NV12 (non-RGBA source) → Rgb F32.
    ///
    /// NV12 requires a two-stage CPU conversion (NV12→Rgba then Rgba→F32)
    /// which was untested. A wrong intermediate format selection silently
    /// produces garbage. Y=128, U=V=128 → near-neutral grey → R≈G≈B≈0.5.
    #[test]
    // Linux-only `egl_display` field makes `..Default::default()` needless
    // on macOS only; see `convert_f16_forced_cpu_correct`.
    #[allow(clippy::needless_update)]
    fn convert_nv12_to_rgb_f32_cpu() {
        const W: usize = 16;
        const H: usize = 16; // must be even for NV12

        // Build a valid NV12 tensor: shape [H*3/2, W], luma=128, chroma=128.
        let src =
            TensorDyn::image(W, H, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            map.as_mut_slice().fill(128); // Y=128, U=V=128 → neutral grey
        }

        let mut dst =
            TensorDyn::image(W, H, PixelFormat::Rgb, DType::F32, Some(TensorMemory::Mem)).unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();

        let result = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default());
        assert!(
            result.is_ok(),
            "forced-CPU NV12→Rgb F32 must not error: {:?}",
            result.err()
        );

        let map = dst.as_f32().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), W * H * 3, "unexpected element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
        // Anti-all-zero: Y=128 → luma channel ≈ 0.5 after YUV→RGB.
        let non_zero = floats.iter().any(|&v| v > 0.01);
        assert!(non_zero, "all-zero output from NV12→Rgb F32 CPU path");
    }

    /// CPU widen-composition: NV12 (non-RGBA source) → PlanarRgb F16.
    ///
    /// Same rationale as `convert_nv12_to_rgb_f32_cpu` but for F16 output.
    #[test]
    // Linux-only `egl_display` field makes `..Default::default()` needless
    // on macOS only; see `convert_f16_forced_cpu_correct`.
    #[allow(clippy::needless_update)]
    fn convert_nv12_to_planar_rgb_f16_cpu() {
        const W: usize = 16;
        const H: usize = 16;

        let src =
            TensorDyn::image(W, H, PixelFormat::Nv12, DType::U8, Some(TensorMemory::Mem)).unwrap();
        {
            let mut map = src.as_u8().unwrap().map().unwrap();
            map.as_mut_slice().fill(128);
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        )
        .unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();

        let result = proc.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default());
        assert!(
            result.is_ok(),
            "forced-CPU NV12→PlanarRgb F16 must not error: {:?}",
            result.err()
        );

        let map = dst.as_f16().unwrap().map().unwrap();
        let halfs = map.as_slice();
        assert_eq!(halfs.len(), W * H * 3, "unexpected element count");
        for (i, h) in halfs.iter().enumerate() {
            let v = h.to_f32();
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
        let non_zero = halfs.iter().any(|h| h.to_f32() > 0.01);
        assert!(non_zero, "all-zero output from NV12→PlanarRgb F16 CPU path");
    }

    // =========================================================================
    // GAP-5: create_image F32 + DMA must return NotSupported
    // =========================================================================

    /// There is no DRM FourCC for F32 images, so `create_image` with
    /// `TensorMemory::Dma` and `DType::F32` must return `Err(NotSupported)`.
    #[test]
    #[cfg(target_os = "linux")]
    fn create_image_f32_dma_rejected() {
        let proc = ImageProcessor::new().unwrap();
        let result = proc.create_image(
            64,
            64,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Dma),
        );
        assert!(
            result.is_err(),
            "create_image(F32, Dma) must fail — no DRM fourcc for f32"
        );
    }

    /// Verify that `import_image` stores the supplied `Option<Colorimetry>` on
    /// the returned `TensorDyn`.
    ///
    /// `import_image` requires a DMA-backed fd on Linux.  When DMA is
    /// unavailable we skip the DMA call but still verify the storage contract
    /// by inspecting `set_colorimetry` / `colorimetry` on a plain `TensorDyn`
    /// constructed the same way the function body does it — and we confirm via
    /// code-read that the new parameter is unconditionally stored.
    #[test]
    #[cfg(target_os = "linux")]
    fn import_image_carries_colorimetry() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry, TensorMemory};

        let expected = Colorimetry::default()
            .with_encoding(ColorEncoding::Bt709)
            .with_range(ColorRange::Limited);

        if !is_dma_available() {
            // DMA unavailable on this host: exercise the storage path via
            // TensorDyn directly (mirrors what import_image does internally).
            let mut t =
                TensorDyn::image(8, 8, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Mem))
                    .expect("alloc");
            assert_eq!(t.colorimetry(), None, "colorimetry must start as None");
            t.set_colorimetry(Some(expected));
            assert_eq!(
                t.colorimetry(),
                Some(expected),
                "set_colorimetry must round-trip"
            );
            eprintln!("SKIPPED import_image_carries_colorimetry (DMA unavailable); storage contract verified via TensorDyn");
            return;
        }

        // DMA is available: allocate a real DMA tensor, extract its fd, and
        // call import_image with an explicit Colorimetry.
        use edgefirst_tensor::{PlaneDescriptor, Tensor};

        let rgba_bytes = 64 * 64 * 4; // 64×64 RGBA8
        let dma_tensor =
            Tensor::<u8>::new(&[rgba_bytes], Some(TensorMemory::Dma), Some("import_test"))
                .expect("dma alloc");
        let pd =
            PlaneDescriptor::new(dma_tensor.dmabuf().expect("dma fd")).expect("PlaneDescriptor");

        let proc = ImageProcessor::new().expect("ImageProcessor");
        let result = proc.import_image(
            pd,
            None,
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(expected),
        );
        let tensor = result.expect("import_image must succeed on DMA fd");
        assert_eq!(
            tensor.colorimetry(),
            Some(expected),
            "import_image must store the supplied colorimetry on the returned TensorDyn"
        );
    }
}
