// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!

## EdgeFirst HAL - Image Converter

The `edgefirst_image` crate is part of the EdgeFirst Hardware Abstraction
Layer (HAL) and provides functionality for converting images between
different formats and sizes.  The crate is designed to work with hardware
acceleration when available, but also provides a CPU-based fallback for
environments where hardware acceleration is not present or not suitable.

What the crate covers:
- Packed and planar RGB families (`Rgb`, `Rgba`, `Bgra`, `Grey`,
  `PlanarRgb`, `PlanarRgba`), semi-planar YUV (`Nv12`, `Nv16`, `Nv24`), and
  packed YUV (`Yuyv`, `Vyuy`). `Yuyv`, `Vyuy`, `Nv16` and `Nv12` are also
  conversion destinations.
- Source crop, resize, letterbox, 90° rotation, and flipping in one call.
- Segmentation-mask rendering, including a fused GPU proto path.
- SAHI-style input tiling for small-object detection in large frames
  ([`TilingConfig`], [`ImageProcessor::tile_into`]).
- Hardware acceleration (OpenGL, NXP G2D) with a CPU fallback that is
  always available.

The crate uses [`TensorDyn`] from `edgefirst_tensor` to represent images,
with [`PixelFormat`] metadata describing the pixel layout. The
[`ImageProcessor`] struct manages the conversion process, selecting
the appropriate conversion method based on the available hardware.

Geometry is **source-side**: [`Crop`] says which sub-rectangle of the source
to sample and how to fit it into the destination's shape. Where the output
lands is the destination itself — pass a
[`view`](edgefirst_tensor::TensorDyn::view) or
[`batch`](edgefirst_tensor::TensorDyn::batch) of a larger tensor to render
into one tile of a batch.

## Examples

```rust
# use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait};
# use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
# use edgefirst_tensor::{CpuAccess, PixelFormat, DType, Tensor, TensorMemory};
# fn main() -> Result<(), edgefirst_image::Error> {
let image = edgefirst_bench::testdata::read("zidane.jpg");
// The codec emits the source's native format (a colour JPEG decodes to NV12)
// and configures the destination tensor's dims+format during the decode.
let info = peek_info(&image).expect("peek");
// Heap tensors are CPU-touched on both sides (codec decode writes, the
// engine's upload/CPU-fallback paths read), so declare ReadWrite. Pure
// hardware pipelines allocate their convert destinations with
// `CpuAccess::None` instead.
let mut src = Tensor::<u8>::image(info.width, info.height, info.format,
                                   Some(TensorMemory::Mem), edgefirst_tensor::CpuAccess::ReadWrite)?;
let mut decoder = ImageDecoder::new();
src.load_image(&mut decoder, &image).expect("decode");
// Convert the native NV12 frame to packed RGB for downstream processing.
let mut converter = ImageProcessor::new()?;
let mut dst =
    converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
converter.convert(&src.into(), &mut dst, None)?;
# Ok(())
# }
```

## Environment Variables
The behavior of the `edgefirst_image::ImageProcessor` struct can be influenced by the
following environment variables:
- `EDGEFIRST_FORCE_BACKEND`: When set to `cpu`, `g2d`, or `opengl` (case-insensitive),
  only that single backend is initialized and no fallback chain is used. If the
  forced backend fails to initialize, an error is returned immediately. When this variable is
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
uses system memory. This disables the use of specialized memory regions for tensors
and hardware acceleration.
*/
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

/// Installs the coverage flush-on-abort handler for this crate's
/// instrumented test binary. See `edgefirst_tensor::covguard`. Present
/// under coverage on Linux only.
#[cfg(all(coverage, target_os = "linux"))]
#[used]
#[link_section = ".init_array"]
static __EDGEFIRST_COV_INSTALL: extern "C" fn() = {
    extern "C" fn ctor() {
        edgefirst_tensor::covguard::install();
    }
    ctor
};

/// Row pitch alignment, in bytes, for DMA-BUF tensors that may be imported
/// as EGLImages by the GL backend.
///
/// Applied inside [`ImageProcessor::create_image`] when the allocation is
/// `TensorMemory::DmaBuf`. [`align_width_for_gpu_pitch`] computes a width
/// whose row stride satisfies this requirement.
pub const GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES: usize = 64;

/// Round `width` (in pixels) up so the resulting row stride
/// `width * bpp` is a multiple of [`GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`]
/// and a multiple of `bpp`.
///
/// `bpp` must be the per-pixel byte count for the image's primary plane
/// (e.g. 4 for RGBA8/BGRA8, 3 for RGB888, 1 for Grey/NV12-luma).
///
/// Pre-aligned widths (640, 1280, 1920, 3008, 3840 …) are returned
/// unchanged; misaligned widths are rounded up to the next valid value.
///
/// # Overflow behaviour
///
/// All arithmetic is checked. If the alignment computation or the rounded
/// width would overflow `usize`, the function logs a warning and returns the
/// original `width`. The returned width is at least the requested width.
///
/// `bpp == 0` and `width == 0` return the input unchanged.
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

    // The aligned stride is a multiple of lcm(pitch alignment, bpp). The
    // alignment is computed in pixels, without multiplying `width * bpp`.
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
    // `a / g` is exact; only the multiply can overflow.
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
/// Returns `None` for formats without a single packed BPP. Semi-planar
/// formats return the luma-plane bpp.
///
/// Use with [`align_width_for_gpu_pitch`] to size a DMA-BUF:
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
        // NV12/NV16: the luma plane is single-channel; the chroma plane has
        // the same pitch in bytes.
        PixelLayout::SemiPlanar => Some(elem),
        _ => None,
    }
}

/// Return the GPU-aligned pitch in bytes when a DMA-backed image of
/// `width × fmt` would need row-stride padding, or `None` when the
/// natural pitch already satisfies `GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`
/// or the caller has explicitly requested non-DMA memory.
#[cfg(all(target_os = "linux", test))]
pub(crate) fn padded_dma_pitch_for(
    fmt: PixelFormat,
    width: usize,
    memory: &Option<TensorMemory>,
) -> Option<usize> {
    // Pads when the caller requested DMA, or left the choice to the
    // allocator and DMA is available.
    match memory {
        Some(TensorMemory::DmaBuf) => {}
        None if edgefirst_tensor::is_dma_available() => {}
        _ => return None,
    }
    // Only packed layouts are padded here; `Tensor::image` pads semi-planar
    // and planar rows itself.
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

/// Whether a G2D initialisation failure only says the host has no G2D: the
/// loader found no file for its library. A library that is installed and
/// does not load, or that lacks a dependency, is a failure.
#[cfg(target_os = "linux")]
fn g2d_is_absent(err: &Error) -> bool {
    match err {
        Error::G2D(g2d_sys::Error::LibraryError(libloading::Error::DlOpen { source })) => {
            library_not_found(g2d::G2D_LIBRARY, &source.to_string())
        }
        _ => false,
    }
}

/// Whether the loader's `message` says it found no file for `library`
/// itself. The loader names the file it could not open before a colon, so a
/// missing dependency is reported under the dependency's name. A message in
/// any other form is not taken as a missing library.
#[cfg(target_os = "linux")]
fn library_not_found(library: &str, message: &str) -> bool {
    message.contains(&format!("{library}: ")) && message.ends_with("No such file or directory")
}

pub use cpu::CPUProcessor;
#[cfg(feature = "codec")]
pub use edgefirst_codec as codec;

#[cfg(test)]
use edgefirst_tensor::ProtoLayout;
#[cfg(any(
    test,
    all(any(target_os = "linux", target_os = "windows"), feature = "opengl")
))]
use edgefirst_tensor::Tensor;
#[doc(inline)]
pub use edgefirst_tensor::{
    fit_code, interpolation_code, letterbox_placement_code, normalization_code, ContentRegion,
    ContentRegionSource, Fit, FitMode, Interpolation, LetterboxPlacement, Normalization,
    NormalizationPreset, PadSpace, PreprocessSpec, Region,
};
use edgefirst_tensor::{
    DType, PixelFormat, PixelLayout, TensorDyn, TensorMemory, TensorTrait as _,
};
use edgefirst_tensor::{DetectBox, ProtoData, Segmentation};
use enum_dispatch::enum_dispatch;
pub use error::{Error, Result};
#[cfg(target_os = "linux")]
pub use g2d::G2DProcessor;
/// Handle returned by `convert_with_fence`: a sync_file fd on Unix, an
/// event handle on Windows that is set when the GPU work completes.
#[cfg(unix)]
pub type CompletionFence = std::os::fd::OwnedFd;
/// Handle returned by `convert_with_fence`: a sync_file fd on Unix, an
/// event handle on Windows that is set when the GPU work completes.
#[cfg(windows)]
pub type CompletionFence = std::os::windows::io::OwnedHandle;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
pub use opengl_headless::EglDisplayKind;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
pub use opengl_headless::GLProcessorThreaded;
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
pub use opengl_headless::Int8InterpolationMode;
#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
pub use opengl_headless::{probe_egl_displays, EglDisplayInfo};
// EGLImage cache counter snapshots (diagnostics).
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
pub use opengl_headless::{CacheStats, ConvertStats, GlCacheStats};
use std::{fmt::Display, time::Instant};

mod colorimetry;
mod cpu;
mod error;
mod fallback_log;
mod g2d;
#[path = "gl/mod.rs"]
mod opengl_headless;
mod options;
pub use options::ConvertOptions;
#[cfg(test)]
mod nv12_dst_tests;
#[cfg(test)]
mod test_support;
mod tiling;
pub use tiling::{tile_grid, TilePlacement, TileSpec, TilingConfig};

// The discriminant of each image option vocabulary is the code every
// surface uses (`code()`, the `EF_ROTATION_*` / `EF_FLIP_*` /
// `EF_COLOR_MODE_*` / `EF_FIT_*` C enumerators, and the Python enum values).
edgefirst_tensor::ef_vocabulary! {
    /// A quarter-turn rotation applied during [`ImageProcessorTrait::convert`].
    #[derive(Default)]
    pub enum Rotation {
        #[default]
        None = 0, "none", NONE,
        Clockwise90 = 1, "clockwise90", CLOCKWISE90,
        Rotate180 = 2, "rotate180", ROTATE180,
        CounterClockwise90 = 3, "counter_clockwise90", COUNTER_CLOCKWISE90,
    }
    #[doc(hidden)]
    pub mod rotation_code;
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

edgefirst_tensor::ef_vocabulary! {
    /// A mirror applied during [`ImageProcessorTrait::convert`].
    #[derive(Default)]
    pub enum Flip {
        #[default]
        None = 0, "none", NONE,
        /// Mirror top to bottom.
        Vertical = 1, "vertical", VERTICAL,
        /// Mirror left to right.
        Horizontal = 2, "horizontal", HORIZONTAL,
    }
    #[doc(hidden)]
    pub mod flip_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// Controls how the color palette index is chosen for each detected object.
    #[derive(Default)]
    pub enum ColorMode {
        /// Color is chosen by object class label (`det.label`). Default.
        #[default]
        Class = 0, "class", CLASS,
        /// Color is chosen by instance order (loop index, zero-based): each
        /// detected object gets a unique color regardless of class.
        Instance = 1, "instance", INSTANCE,
        /// Color is chosen by track ID. Behaves like
        /// [`Instance`](Self::Instance).
        Track = 2, "track", TRACK,
    }
    #[doc(hidden)]
    pub mod color_mode_code;
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
///   plane). Mask values are continuous sigmoid output quantized to
///   `uint8 [0, 255]`.
/// - [`Scaled`](Self::Scaled) returns per-detection tiles at caller-specified
///   pixel resolution by upsampling the full proto plane once and cropping by
///   bbox after sigmoid. The upsample uses bilinear interpolation with
///   edge-clamp sampling. When a `content_region` is also passed to
///   [`materialize_masks`], the inverse letterbox transform is applied during
///   the upsample so mask pixels land in original-content coordinates. Mask
///   values are binary `uint8 {0, 255}` after thresholding sigmoid > 0.5.
///
/// [`materialize_masks`]: ImageProcessor::materialize_masks
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MaskResolution {
    /// Per-detection tile at proto-plane resolution (default).
    #[default]
    Proto,
    /// Per-detection tile at `(width, height)` pixel resolution in the
    /// coordinate frame determined by the `content_region` parameter of
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
///   `dst` is always fully overwritten.
/// - `opacity`: scales the alpha of rendered mask and box colors. `1.0`
///   (default) preserves the class color's alpha unchanged; `0.5` makes
///   masks and boxes twice as transparent.
/// - `color_mode`: controls whether colors are assigned by class label,
///   instance index, or track ID. Defaults to [`ColorMode::Class`].
#[derive(Debug, Clone, Copy)]
pub struct MaskOverlay<'a> {
    /// Compositing source image. Must have the same dimensions and pixel
    /// format as `dst`. When `Some`, the output is `background + masks`.
    /// When `None`, `dst` is cleared to `0x00000000` before masks are drawn.
    pub background: Option<&'a TensorDyn>,
    pub opacity: f32,
    /// The image content within the model input, as normalized
    /// `[xmin, ymin, xmax, ymax]` ([`ContentRegion::normalized`]); the rest is
    /// letterbox padding.
    ///
    /// When set, bounding boxes and mask coordinates from the decoder (which
    /// are in model-input normalized space) are mapped back to the original
    /// image coordinate space before rendering.
    ///
    /// Use [`with_convert_options`](Self::with_convert_options) to compute this
    /// from the options of the model input [`convert`](crate::ImageProcessorTrait::convert) call.
    pub content_region: Option<[f32; 4]>,
    pub color_mode: ColorMode,
}

impl Default for MaskOverlay<'_> {
    fn default() -> Self {
        Self {
            background: None,
            opacity: 1.0,
            content_region: None,
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

    /// Set the content region from the [`ConvertOptions`] used when
    /// preparing the model input, so that bounding boxes and masks are
    /// mapped back to the original image when rendering.
    ///
    /// Pass the same options that were given to
    /// [`convert`](crate::ImageProcessorTrait::convert) along with the source
    /// and model input dimensions. Has no effect without a letterbox fit, or
    /// when the geometry does not resolve.
    pub fn with_convert_options(
        mut self,
        options: &ConvertOptions,
        src_w: usize,
        src_h: usize,
        model_w: usize,
        model_h: usize,
    ) -> Self {
        if let Ok(region) = options.content_region(src_w, src_h, model_w, model_h) {
            self.content_region = region.letterboxed();
        }
        self
    }
}

/// Copy a [`Segmentation`], replacing its bounds with `bbox`.
fn copy_segmentation(s: &Segmentation, bbox: [f32; 4]) -> Result<Segmentation> {
    use edgefirst_tensor::TensorMapTrait as _;
    let src = s
        .segmentation
        .as_typed::<u8>()
        .ok_or_else(|| Error::InvalidShape("segmentation must be a U8 tensor".into()))?;
    let map = src.map_read()?;
    let dst = edgefirst_tensor::Tensor::<u8>::new(
        s.segmentation.shape(),
        Some(edgefirst_tensor::TensorMemory::Mem),
        None,
    )
    .map_err(|e| Error::InvalidShape(format!("segmentation copy: {e}")))?;
    {
        let mut dmap = dst.map_mut()?;
        dmap.as_mut_slice().copy_from_slice(map.as_slice());
    }
    Ok(Segmentation {
        xmin: bbox[0],
        ymin: bbox[1],
        xmax: bbox[2],
        ymax: bbox[3],
        segmentation: dst.into(),
    })
}

/// Apply the inverse letterbox transform to a bounding box.
///
/// `lb` is `[lx0, ly0, lx1, ly1]` — the normalized region of the model input
/// that contains actual image content ([`MaskOverlay::content_region`]).
/// Converts model-input-normalized coords to output-image-normalized coords,
/// clamped to `[0.0, 1.0]`, and canonicalises the bbox (ensures xmin <= xmax).
///
/// Wraps [`edgefirst_tensor::unletter_norm`].
#[inline]
fn unletter_bbox(bbox: DetectBox, lb: [f32; 4]) -> DetectBox {
    DetectBox {
        bbox: edgefirst_tensor::unletter_norm(bbox.bbox, lb),
        ..bbox
    }
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
    /// colour.
    pub(crate) fn resolve(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
        placement: LetterboxPlacement,
    ) -> Result<ResolvedCrop, Error> {
        let src_rect = self.source.map(region_to_rect);
        // The letterbox aspect uses the source crop when set, else the full source.
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
                dst_rect: Some(letterbox_rect(sw, sh, dst_w, dst_h, placement)),
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
            LetterboxPlacement::Center,
        )
        .map(|_| ())
    }
}

/// Resolved crop geometry consumed by the backends. Produced by
/// [`Crop::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ResolvedCrop {
    pub(crate) src_rect: Option<Rect>,
    pub(crate) dst_rect: Option<Rect>,
    pub(crate) dst_color: Option<[u8; 4]>,
}

impl ResolvedCrop {
    /// A no-op resolved crop (whole source → whole destination, no pad).
    #[allow(dead_code)]
    pub(crate) fn no_crop() -> Self {
        Self::default()
    }

    /// Whether a convert from a `src`-sized source to a `dst`-sized
    /// destination with this crop changes the size of the sampled region. A
    /// quarter-turn rotation swaps the source region's axes.
    #[allow(dead_code)]
    pub(crate) fn resizes(
        &self,
        (src_w, src_h): (usize, usize),
        (dst_w, dst_h): (usize, usize),
        rotation: Rotation,
    ) -> bool {
        let (sw, sh) = self
            .src_rect
            .map_or((src_w, src_h), |r| (r.width, r.height));
        let (dw, dh) = self
            .dst_rect
            .map_or((dst_w, dst_h), |r| (r.width, r.height));
        let (sw, sh) = match rotation {
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (sh, sw),
            Rotation::None | Rotation::Rotate180 => (sw, sh),
        };
        (sw, sh) != (dw, dh)
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

/// [`edgefirst_tensor::letterbox_rect`] as the backends' [`Rect`].
fn letterbox_rect(
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    placement: LetterboxPlacement,
) -> Rect {
    let r = edgefirst_tensor::letterbox_rect(sw, sh, dw, dh, placement);
    Rect::new(r.x, r.y, r.width, r.height)
}

/// Placement rectangle (top-left + size): the crate-private resolved
/// placement the CPU/G2D/GL backends use after [`Crop::resolve`]. The
/// public pixel sub-region type is [`Region`].
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
    /// Convert `src` into `dst`'s format and size, applying the crop first,
    /// then the flip, then the rotation.
    ///
    /// The destination is normally an RGB-family colour — `Grey`, `Rgb`,
    /// `Rgba`/`Bgra`, packed `HWC` or planar `CHW`. The CPU backend also
    /// renders YUV destinations: `Yuyv` (from `Yuyv`, `Nv12`, `Rgb`, `Rgba`,
    /// `Grey`), `Vyuy` (from `Vyuy`), `Nv16` and `Nv12` (each from `Yuyv`,
    /// `Vyuy`, `Nv12`, `Rgb`, `Rgba` and `Grey`; `Nv12` also from `Nv16` and `Nv24`). The CPU backend
    /// converts a `Bgra` source to every destination it converts an `Rgba` source to. An `Nv12` destination's
    /// width and height must be even. An accelerator may render an `Nv12`
    /// destination; a destination whose [`CpuAccess`](edgefirst_tensor::CpuAccess)
    /// denies the CPU can be refused when the accelerator declines it.
    ///
    /// A YUV destination (`Nv12`, `Nv16`, `Yuyv`, `Vyuy`) is encoded with its
    /// own colorimetry. An untagged one is encoded with the colorimetry its
    /// height implies (720 rows or more: BT.709, otherwise BT.601; limited
    /// range), and after a successful convert carries that colorimetry as its
    /// tag. A failed convert leaves the tag unchanged. Tag the destination with
    /// [`TensorDyn::set_colorimetry`](edgefirst_tensor::TensorDyn::set_colorimetry)
    /// to choose another, for example to keep a source's colorimetry through a
    /// resize. Tagging it with the source's colorimetry keeps a YUV-to-YUV
    /// resize on engines that cannot re-encode, and
    /// [`ComputeBackend::G2d`](crate::ComputeBackend::G2d) writes the source's
    /// colorimetry into an untagged destination and tags it with it.
    ///
    /// A YUV source converted to a YUV destination with no resize, crop,
    /// placement, flip or rotation is copied without colour conversion when
    /// both resolve to the same matrix and range, and re-encoded otherwise. A
    /// `Vyuy` destination that would need re-encoding returns
    /// [`Error::NotSupported`].
    ///
    /// # Arguments
    ///
    /// * `src` - The source image to convert from.
    /// * `dst` - The destination image. Its shape and [`PixelFormat`] define
    ///   the output; pass a [`view`](edgefirst_tensor::TensorDyn::view) or
    ///   [`batch`](edgefirst_tensor::TensorDyn::batch) of a larger tensor to
    ///   render into one tile of a batch.
    /// * `options` - Rotation, flip, source crop, fit (stretch or letterbox),
    ///   normalization and pad ([`ConvertOptions`]); `None` is the default
    ///   convert. Destination placement lives on `dst`, not here.
    ///
    /// # Errors
    ///
    /// - [`Error::CropInvalid`] if the source crop falls outside `src`.
    /// - [`Error::NotAnImage`] if either tensor carries no `PixelFormat`.
    /// - [`Error::InvalidOptions`] if the options cannot apply to `dst`.
    /// - [`Error::NotSupported`] if an `Nv12` destination has an odd width or
    ///   height.
    /// - [`Error::NoConverter`] if no backend accepts the format pair; the
    ///   dispatch tries OpenGL, then G2D, then CPU.
    /// - [`Error::Tensor`] with [`CpuAccessDenied`](edgefirst_tensor::Error::CpuAccessDenied)
    ///   if a CPU pass needs access a tensor's declaration does not cover
    ///   (source `Read`; destination `Write`, or `ReadWrite` for a view or a
    ///   read-back pass).
    /// - [`Error::CpuFallbackDenied`] if GL or G2D declined the convert and
    ///   the CPU fallback was refused for access. [`Error::cpu_access_denied`]
    ///   finds the refusal in either form.
    ///
    /// Backend-specific failures (EGL import, G2D blit, allocation) surface as
    /// their own variants. A backend declining a pair is not an error while
    /// another remains in the chain;
    /// [`ImageProcessor::convert_fallback_count`] counts those.
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<ConvertOptions>,
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
    /// This function always fully writes `dst`. The four cases are:
    ///
    /// | detections | background | output                              |
    /// |------------|------------|-------------------------------------|
    /// | none       | none       | dst cleared to `0x00000000`         |
    /// | none       | set        | dst ← background                    |
    /// | set        | none       | masks drawn over cleared dst        |
    /// | set        | set        | masks drawn over background         |
    ///
    /// An empty `segmentation` slice is valid — only bounding boxes are drawn.
    ///
    /// `overlay` controls compositing: `background` is the compositing source
    /// (must match `dst` in size and format); `opacity` scales mask and box
    /// alpha. Box outlines are [`set_box_line_width`](Self::set_box_line_width)
    /// pixels thick and use the [`set_class_colors`](Self::set_class_colors)
    /// palette, alpha included.
    ///
    /// A `Bgra` background onto a zero-copy destination (a D3D11 texture, an
    /// IOSurface, a dma-buf) returns [`Error::NotSupported`]. A `Bgra`
    /// destination with no background is unaffected.
    ///
    /// # Buffer aliasing
    ///
    /// `dst` and `overlay.background` must reference **distinct underlying
    /// buffers**. An aliased pair returns [`Error::AliasedBuffers`] without
    /// dispatching to any backend. Aliasing is detected via
    /// [`TensorDyn::aliases`](edgefirst_tensor::TensorDyn::aliases), which
    /// catches both shared-allocation clones and separate imports over the
    /// same dmabuf fd.
    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: MaskOverlay<'_>,
    ) -> Result<()>;

    /// Draw masks from proto data onto image (fused decode+draw).
    ///
    /// The `ProtoData` contains mask coefficients and the prototype tensor;
    /// the renderer computes `mask_coeff @ protos` at the output resolution
    /// using bilinear sampling, without materializing intermediate masks.
    ///
    /// `detect` and `proto_data.mask_coefficients` must have the same length;
    /// excess entries are ignored. An empty `detect` slice is valid and
    /// produces the base frame, cleared or background-blitted.
    ///
    /// # Format requirements and output contract
    ///
    /// Same as [`draw_decoded_masks`](Self::draw_decoded_masks), including
    /// that `dst` is always fully written.
    ///
    /// `overlay` controls compositing — see [`draw_decoded_masks`](Self::draw_decoded_masks).
    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: MaskOverlay<'_>,
    ) -> Result<()>;

    /// Sets the RGBA colors used for rendering segmentation masks and
    /// bounding boxes. Up to 20 colors can be set; entries past the slice
    /// keep their previous value.
    ///
    /// The alpha channel is the coverage of both masks and box outlines:
    /// each is blended over the destination with `alpha / 255` scaled by
    /// [`MaskOverlay::opacity`], so `255` draws opaque boxes and lower
    /// values draw translucent ones. The destination's own alpha channel is
    /// left as the base layer set it. The default palette's alpha is `0.7`.
    ///
    /// On [`ImageProcessor`] with no forced backend this reaches every
    /// available backend. A forced backend receives the colors alone.
    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<()>;

    /// Sets the thickness, in destination pixels, of the bounding-box
    /// outlines drawn by [`draw_decoded_masks`](Self::draw_decoded_masks)
    /// and [`draw_proto_masks`](Self::draw_proto_masks). The default is
    /// [`DEFAULT_BOX_LINE_WIDTH`] (3 px).
    ///
    /// The outline is drawn outside the box edges and is clipped at the image
    /// border.
    ///
    /// Reaches every available backend, as
    /// [`set_class_colors`](Self::set_class_colors) does.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidOptions`] when `width` is 0, and
    /// [`Error::NotImplemented`] from a backend that does not draw (G2D).
    fn set_box_line_width(&mut self, width: u32) -> Result<()>;

    /// Like [`convert`](Self::convert), but does not wait for the GPU to finish.
    ///
    /// A caller renders `N` tiles into one batched destination by looping
    /// `convert_deferred(&src[n], &mut dst.batch(n)?, …)` and then calling
    /// [`flush`](Self::flush) once. On the OpenGL backend every deferred
    /// convert into sibling views of one buffer shares a single EGLImage import
    /// and skips the per-tile `glFinish`; `flush` then issues a single GPU
    /// sync. The result of a deferred convert is not safe to read on the CPU
    /// (or map via CUDA) until `flush` returns.
    ///
    /// The default implementation calls [`convert`](Self::convert): each call
    /// completes synchronously and `flush` is a no-op.
    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<ConvertOptions>,
    ) -> Result<()> {
        self.convert(src, dst, options)
    }

    /// Complete all work enqueued by [`convert_deferred`](Self::convert_deferred)
    /// since the last flush, issuing a single GPU synchronization.
    ///
    /// After this returns, every deferred destination is finished and safe to
    /// read back or `cuda_map`. Backends with no deferred path (the default)
    /// return `Ok(())` immediately.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Configuration for [`ImageProcessor`] construction.
///
/// Use with [`ImageProcessor::with_config`] to override the default EGL
/// display auto-detection and backend selection.
#[derive(Debug, Clone, Default)]
pub struct ImageProcessorConfig {
    /// Force OpenGL to use this EGL display type instead of auto-detecting.
    ///
    /// When `None`, the processor probes displays in priority order: GBM,
    /// PlatformDevice, Default. Use [`probe_egl_displays`] to discover
    /// which displays are available on the current system.
    ///
    /// Ignored when `EDGEFIRST_DISABLE_GL=1` is set, and on macOS and
    /// Windows (ANGLE is the only display there — Metal and Direct3D 11
    /// respectively; a `Some` value logs a debug note and is otherwise
    /// ignored).
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    pub egl_display: Option<EglDisplayKind>,

    /// Preferred compute backend.
    ///
    /// When set to a specific backend (not [`ComputeBackend::Auto`]), the
    /// processor initializes that backend with no fallback and returns an
    /// error if the conversion is not supported. This takes precedence over
    /// `EDGEFIRST_FORCE_BACKEND` and the `EDGEFIRST_DISABLE_*` environment
    /// variables.
    ///
    /// - [`ComputeBackend::OpenGl`]: init OpenGL + CPU, skip G2D; it renders
    ///   conversions that `Auto` leaves to the CPU for speed or exactness
    /// - [`ComputeBackend::G2d`]: init G2D + CPU, skip OpenGL
    /// - [`ComputeBackend::Cpu`]: init CPU only
    /// - [`ComputeBackend::Auto`]: env-var-driven selection
    pub backend: ComputeBackend,

    /// Colorimetry/performance trade-off for `convert()` (see
    /// [`ColorimetryMode`]). Defaults to [`ColorimetryMode::Fast`]. The
    /// `EDGEFIRST_COLORIMETRY` environment variable (`fast` | `exact`)
    /// overrides this setting when present.
    pub colorimetry: ColorimetryMode,

    /// How `convert()` resamples when the source (or source crop) and the
    /// destination differ in size (see [`Interpolation`]), for converts whose
    /// options do not choose their own ([`ConvertOptions::interpolation`]).
    /// Defaults to [`Interpolation::Bilinear`]. The `EDGEFIRST_INTERPOLATION`
    /// environment variable (`nearest` | `bilinear`), read when the processor
    /// is created, overrides this setting and every per-convert choice.
    pub interpolation: Interpolation,

    /// How many buffer imports the OpenGL backend keeps cached, per cache.
    ///
    /// `None` (the default) uses `EDGEFIRST_EGL_CACHE_CAPACITY` if set, and
    /// 16 otherwise. Precedence is:
    ///
    /// ```text
    /// egl_cache_capacity  >  EDGEFIRST_EGL_CACHE_CAPACITY  >  16
    /// ```
    ///
    /// `Some(0)` is rejected and falls through to the environment, then the
    /// default, as an unparseable env value does.
    ///
    /// # This is a buffer *count*, not a size
    ///
    /// A cached import pins its buffer: the underlying dma-buf or IOSurface
    /// cannot be freed while an entry names it. The memory held depends on
    /// the size of the caller's buffers:
    ///
    /// ```text
    /// worst-case pinned bytes                                (upper bound)
    ///     = capacity x 3 caches x frame size x live ImageProcessors
    ///
    /// capacity 16, 640x640 RGB   (1.2 MB)  ->   59 MB per processor
    /// capacity 16, 1080p NV12    (3.1 MB)  ->  149 MB per processor
    /// capacity 16, 4K NV12      (12.4 MB)  ->  597 MB per processor
    /// ```
    ///
    /// This is an upper bound: two entries naming one buffer pin it once.
    ///
    /// The bound is per [`ImageProcessor`], times three caches (source,
    /// destination, and the NV R8 source cache). A fourth cache, the
    /// chroma-plane views of hardware-filtered NV resizes, has the same
    /// capacity but imports only buffers the NV R8 cache also holds.
    ///
    /// [`CacheStats::peak_entries`] reports the working set reached, and
    /// [`CacheStats::evictions`] the entries the bound evicted.
    ///
    /// Ignored when the OpenGL backend is not in use.
    pub egl_cache_capacity: Option<usize>,
}

/// How `convert()` trades colorimetric exactness against speed.
///
/// Both modes select the same conversion paths on every backend: NV12
/// sources take the in-shader matrix wherever the GL backend can read them
/// with it.
///
/// Override at runtime with `EDGEFIRST_COLORIMETRY=fast|exact` (takes
/// precedence over the config field). `EDGEFIRST_NV_CONVERT_PATH` forces the
/// NV12 path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColorimetryMode {
    /// Prefer the fastest path whose output is correct-enough video RGB
    /// (default).
    #[default]
    Fast,
    /// Prefer bit-exact colorimetry everywhere: the fast path is used only
    /// when it matches the source's resolved (encoding, range) exactly.
    Exact,
}

/// `mode` if the backends run it, else [`Interpolation::supported`] with a
/// warning logged once per mode.
pub(crate) fn runnable_interpolation(mode: Interpolation) -> Interpolation {
    let run = mode.supported();
    if run != mode && warn_first_fallback(mode) {
        log::warn!(
            "{} interpolation is not implemented; using {}",
            mode.as_str(),
            run.as_str()
        );
    }
    run
}

/// Unsupported interpolations already warned about, one bit per code.
static INTERPOLATION_FALLBACKS_WARNED: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Whether this is the first fallback from `mode` in the process.
fn warn_first_fallback(mode: Interpolation) -> bool {
    let bit = 1u32 << (mode.code() % 32);
    INTERPOLATION_FALLBACKS_WARNED.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0
}

/// The environment variables an [`ImageProcessor`] reads when it is built.
#[derive(Debug, Clone, Default)]
pub(crate) struct ProcessorEnv {
    /// `EDGEFIRST_FORCE_BACKEND`.
    pub(crate) force_backend: Option<String>,
    /// `EDGEFIRST_DISABLE_GL` is set to a value other than `0` or `false`.
    #[cfg_attr(not(feature = "opengl"), allow(dead_code))]
    pub(crate) disable_gl: bool,
    /// `EDGEFIRST_DISABLE_G2D` is set to a value other than `0` or `false`.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) disable_g2d: bool,
    /// `EDGEFIRST_DISABLE_CPU` is set to a value other than `0` or `false`.
    pub(crate) disable_cpu: bool,
    /// `EDGEFIRST_INTERPOLATION`.
    pub(crate) interpolation: Option<String>,
}

impl ProcessorEnv {
    /// Reads the variables from the process environment.
    pub(crate) fn from_process() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Reads the variables through `lookup`, which returns a variable's value
    /// or `None` when it is unset.
    pub(crate) fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let disabled = |name: &str| {
            lookup(name)
                .map(|x| x != "0" && x.to_lowercase() != "false")
                .unwrap_or(false)
        };
        Self {
            force_backend: lookup("EDGEFIRST_FORCE_BACKEND"),
            disable_gl: disabled("EDGEFIRST_DISABLE_GL"),
            disable_g2d: disabled("EDGEFIRST_DISABLE_G2D"),
            disable_cpu: disabled("EDGEFIRST_DISABLE_CPU"),
            interpolation: lookup("EDGEFIRST_INTERPOLATION"),
        }
    }
}

/// The interpolation selected by an `EDGEFIRST_INTERPOLATION` value, if it
/// is a recognised one. An unrecognised value is logged and ignored.
pub(crate) fn interpolation_from_env(value: Option<&str>) -> Option<Interpolation> {
    let value = value?.trim().to_ascii_lowercase();
    if value.is_empty() {
        return None;
    }
    let mode = Interpolation::from_str_code(&value);
    if mode.is_none() {
        log::warn!(
            "EDGEFIRST_INTERPOLATION={value:?} is not recognised \
             (expected nearest or bilinear); ignoring it"
        );
    }
    mode
}

edgefirst_tensor::ef_vocabulary! {
    /// Compute backend selection for [`ImageProcessor`].
    ///
    /// Use with [`ImageProcessorConfig::backend`] to select which backend the
    /// processor should prefer. When a specific backend is selected, the
    /// processor initializes that backend plus CPU as a fallback. When `Auto`
    /// is used, the environment-variable-driven selection applies.
    ///
    /// The code is the one the C `EF_COMPUTE_BACKEND_*` enumerators use.
    #[derive(Default)]
    pub enum ComputeBackend {
        /// Auto-detect based on available hardware and environment variables.
        #[default]
        Auto = 0, "auto", AUTO,
        /// CPU-only processing (no hardware acceleration).
        Cpu = 1, "cpu", CPU,
        /// Prefer G2D hardware blitter (+ CPU fallback).
        G2d = 2, "g2d", G2D,
        /// Prefer OpenGL ES (+ CPU fallback).
        OpenGl = 3, "opengl", OPENGL,
    }
    #[doc(hidden)]
    pub mod compute_backend_code;
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

/// The CPU fallback's error, carrying the GPU decline reasons when the CPU
/// was refused for access. `declined` pairs each declining backend's name
/// with its error.
fn with_gpu_declines(e: Error, declined: Vec<(&'static str, Error)>) -> Error {
    match e {
        Error::Tensor(denied @ edgefirst_tensor::Error::CpuAccessDenied { .. })
            if !declined.is_empty() =>
        {
            let declined: Vec<String> = declined
                .iter()
                .map(|(backend, e)| format!("{backend}: {e}"))
                .collect();
            Error::CpuFallbackDenied {
                declined: declined.join("; "),
                denied,
            }
        }
        other => other,
    }
}

/// The CPU access an [`ImageProcessor`]'s own convert may need on a
/// destination, from what each present backend needs when it takes it
/// (`Some`) or declines it (`None`).
///
/// A forced backend is the only one that runs. Otherwise a GL render that
/// needs no CPU access ends the chain, while G2D and the CPU fallback may
/// each still run, so their needs are all included.
fn predicted_cpu_access(
    forced: Option<ForcedBackend>,
    gl: Option<Option<edgefirst_tensor::CpuAccess>>,
    g2d: Option<Option<edgefirst_tensor::CpuAccess>>,
    cpu: Option<Option<edgefirst_tensor::CpuAccess>>,
) -> edgefirst_tensor::CpuAccess {
    use edgefirst_tensor::CpuAccess;
    let only = |b: Option<Option<CpuAccess>>| b.flatten().unwrap_or(CpuAccess::None);
    match forced {
        Some(ForcedBackend::Cpu) => return only(cpu),
        Some(ForcedBackend::G2d) => return only(g2d),
        Some(ForcedBackend::OpenGl) => return only(gl),
        None => {}
    }
    let gl = gl.flatten();
    if gl == Some(CpuAccess::None) {
        return CpuAccess::None;
    }
    [gl, g2d.flatten(), cpu.flatten()]
        .into_iter()
        .flatten()
        .fold(CpuAccess::None, CpuAccess::union)
}

/// Reports which float-color-buffer extensions the GPU backend detected.
/// Returned by [`ImageProcessor::supported_render_dtypes`]; the two flags
/// are independent.
///
/// **Linux:** reflects the probe results from `GL_EXT_color_buffer_half_float`
/// and `GL_EXT_color_buffer_float`. On Vivante GC7000UL both are forced
/// `false`.
///
/// **macOS (ANGLE):** `f16 == true` gates the RGBA16F-packed IOSurface
/// path for F16 `PlanarRgb` destinations. `f32` reflects the GL extension
/// probe; there is no F32 IOSurface path.
///
/// **Windows (ANGLE/D3D11):** `f16` enables the RGBA16F texture paths and
/// `f32` the RGBA32F ones for every float layout: `PlanarRgb` and
/// `PlanarRgba` NCHW, interleaved `Rgb`, and `Rgba`.
///
/// Regardless of these flags, [`ImageProcessor::convert`] does not return
/// an error due to float capability: it falls back to CPU when the GPU
/// path is unavailable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderDtypeSupport {
    /// `GL_EXT_color_buffer_float` is available on the current GPU.
    ///
    /// On Linux, `true` enables F32 `Rgb` NHWC PBO readback. On macOS
    /// this flag is informational only — no F32 IOSurface path exists.
    /// On Windows, `true` enables the RGBA32F zero-copy texture render
    /// for every F32 layout.
    pub f32: bool,
    /// `GL_EXT_color_buffer_half_float` is available on the current GPU.
    ///
    /// On Linux, `true` enables F16 `PlanarRgb` NCHW PBO readback and,
    /// on V3D/Mali, zero-copy DMA-BUF render. On macOS, `true` enables
    /// F16 `PlanarRgb` via RGBA16F-packed IOSurface (zero-copy). On
    /// Windows, `true` enables the RGBA16F zero-copy texture render for
    /// every F16 layout.
    pub f16: bool,
}

/// Returns `true` when a float PBO destination should be attempted for `dtype`.
///
/// Only F16 and F32 are eligible, and only when the corresponding flag in
/// `support` is set. All other dtypes return `false`.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "ios",
        target_os = "android"
    ),
    feature = "opengl"
))]
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
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    /// OpenGL-based image converter. macOS/iOS run it via ANGLE + IOSurface
    /// (available when ANGLE's libEGL.dylib can be loaded), Android via the
    /// native EGL driver + AHardwareBuffer, Windows via ANGLE + Direct3D 11
    /// (available when ANGLE's libEGL.dll can be loaded).
    pub opengl: Option<GLProcessorThreaded>,

    /// When set, only the specified backend is used — no fallback chain.
    pub(crate) forced_backend: Option<ForcedBackend>,

    /// Converts where the GL backend declined and the auto chain fell
    /// through toward G2D/CPU. Read via
    /// [`convert_fallback_count`](Self::convert_fallback_count).
    pub(crate) convert_fallbacks: std::sync::atomic::AtomicU64,

    /// The shapes the auto chain has moved past a backend for.
    #[cfg_attr(not(any(target_os = "linux", feature = "opengl")), allow(dead_code))]
    pub(crate) fallback_log: fallback_log::FallbackLog,

    /// The interpolation a convert without its own uses, as the backends run it.
    pub(crate) interpolation: Interpolation,

    /// `EDGEFIRST_INTERPOLATION`, read once at construction; it pins every convert.
    pub(crate) env_interpolation: Option<Interpolation>,
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
    /// # use edgefirst_tensor::{CpuAccess, PixelFormat, DType, Tensor, TensorMemory};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let image = std::fs::read("zidane.jpg")?;
    /// // The codec emits the source's native format (a colour JPEG decodes to
    /// // NV12) and configures the destination tensor during the decode.
    /// let info = peek_info(&image).expect("peek");
    /// let mut src = Tensor::<u8>::image(info.width, info.height, info.format,
    ///                                    Some(TensorMemory::Mem), edgefirst_tensor::CpuAccess::ReadWrite)?;
    /// let mut decoder = ImageDecoder::new();
    /// src.load_image(&mut decoder, &image).expect("decode");
    /// let mut converter = ImageProcessor::new()?;
    /// let mut dst =
    ///     converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
    /// converter.convert(&src.into(), &mut dst, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self> {
        Self::with_config(ImageProcessorConfig::default())
    }

    /// Number of converts where the auto chain's GL backend declined and the
    /// frame fell through toward G2D/CPU. Always 0 under a forced backend.
    pub fn convert_fallback_count(&self) -> u64 {
        self.convert_fallbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of [`Compression::Any`](edgefirst_tensor::Compression)
    /// requests since process start that resolved to a linear layout: the
    /// process-wide [`edgefirst_tensor::compression_fallback_count`].
    pub fn compression_fallback_count(&self) -> u64 {
        edgefirst_tensor::compression_fallback_count()
    }

    /// Convert and return a native fence that signals when the GPU work
    /// completes, without blocking the CPU (`EGL_ANDROID_native_fence_sync`
    /// on Android). On Windows the handle is a manual-reset event set when
    /// the GPU work completes; wait on it with `WaitForSingleObject`.
    ///
    /// `Ok(Some(fence))`: the destination buffer is still in flight; hand
    /// the fence to the consumer (e.g.
    /// `ANeuralNetworksExecution_startComputeWithDependencies` on Android,
    /// or `WaitForSingleObject` on Windows) or poll it before reading.
    /// `Ok(None)`: the convert completed with the normal blocking contract
    /// (no native fence on this platform, or a non-GL backend handled the
    /// frame) — the destination is already safe. Semantics are otherwise
    /// identical to [`convert`](ImageProcessorTrait::convert), including
    /// the GL→G2D→CPU fallback chain.
    ///
    /// On Windows, for a texture destination, the event and the destination's
    /// own `Tensor::gpu_completion` name the same fence value, and a consumer
    /// may use either. A destination that is not a texture records nothing,
    /// and only the event can be waited on.
    pub fn convert_with_fence(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<ConvertOptions>,
    ) -> Result<Option<CompletionFence>> {
        let options = self.resolve_interpolation(options.unwrap_or_default());
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        {
            let gl_forced = matches!(self.forced_backend, Some(ForcedBackend::OpenGl));
            if self.forced_backend.is_none() || gl_forced {
                if let Some(opengl) = self.opengl.as_mut() {
                    match opengl.convert_with_fence(src, dst, Some(options)) {
                        Ok(fd) => return Ok(fd),
                        Err(e) if gl_forced => return Err(e),
                        Err(e) => {
                            self.fallback_log.note("opengl", src, dst, &e);
                        }
                    }
                } else if gl_forced {
                    return Err(Error::ForcedBackendUnavailable("opengl".into()));
                }
            }
        }
        // Blocking chain: G2D/CPU, forced non-GL, or non-GL builds.
        self.convert(src, dst, Some(options))?;
        Ok(None)
    }

    /// Report which float dtypes the GPU can render to.
    ///
    /// `GL_EXT_color_buffer_half_float` and `GL_EXT_color_buffer_float` are
    /// probed once at `ImageProcessor::new()` time and the result is cached.
    /// [`create_image`] uses it to select a float PBO when supported.
    ///
    /// Returns `RenderDtypeSupport { f32: false, f16: false }` when no
    /// OpenGL backend is active or the `opengl` feature is disabled.
    ///
    /// [`create_image`]: Self::create_image
    pub fn supported_render_dtypes(&self) -> RenderDtypeSupport {
        #[cfg(all(
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ),
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
    /// When `Auto`, the `EDGEFIRST_FORCE_BACKEND` and `EDGEFIRST_DISABLE_*`
    /// environment variables apply.
    pub fn with_config(config: ImageProcessorConfig) -> Result<Self> {
        Self::with_config_env(config, &ProcessorEnv::from_process())
    }

    /// [`ImageProcessor::with_config`] with the environment settings taken
    /// from `env` instead of the process environment.
    pub(crate) fn with_config_env(
        config: ImageProcessorConfig,
        env: &ProcessorEnv,
    ) -> Result<Self> {
        let interpolation = config.interpolation;
        let auto = config.backend == ComputeBackend::Auto;
        let mut processor = Self::with_config_backends(config, env)?;
        let soft_declines = processor.soft_declines_apply(auto);
        #[cfg(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ),
            feature = "opengl"
        ))]
        processor.set_gl_soft_declines(soft_declines)?;
        #[cfg(target_os = "linux")]
        processor.set_g2d_soft_declines(soft_declines);
        #[cfg(not(any(
            target_os = "linux",
            all(
                any(
                    target_os = "macos",
                    target_os = "ios",
                    target_os = "android",
                    target_os = "windows"
                ),
                feature = "opengl"
            )
        )))]
        let _ = soft_declines;
        processor.env_interpolation =
            interpolation_from_env(env.interpolation.as_deref()).map(runnable_interpolation);
        processor.set_interpolation(interpolation)?;
        Ok(processor)
    }

    /// Whether GL and G2D soft-decline: under `ComputeBackend::Auto` (`auto`) with no
    /// forced backend and a CPU backend present.
    fn soft_declines_apply(&self, auto: bool) -> bool {
        auto && self.forced_backend.is_none() && self.cpu.is_some()
    }

    /// Sets the G2D backend's soft declines.
    #[cfg(target_os = "linux")]
    fn set_g2d_soft_declines(&mut self, enabled: bool) {
        if let Some(g2d) = self.g2d.as_mut() {
            g2d.set_soft_declines(enabled);
        }
    }

    /// Sets the GL backend's soft declines. When the GL worker takes no message, drops the GL
    /// backend, or returns the error when GL is forced.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    fn set_gl_soft_declines(&mut self, enabled: bool) -> Result<()> {
        let Some(gl) = self.opengl.as_mut() else {
            return Ok(());
        };
        match gl.set_soft_declines(enabled) {
            Ok(()) => Ok(()),
            Err(e) if self.forced_backend == Some(ForcedBackend::OpenGl) => Err(e),
            Err(e) => {
                log::warn!("OpenGL backend disabled: its soft declines could not be set: {e}");
                self.opengl = None;
                Ok(())
            }
        }
    }

    /// The backend selection half of [`Self::with_config`].
    #[allow(unused_variables)]
    fn with_config_backends(config: ImageProcessorConfig, env: &ProcessorEnv) -> Result<Self> {
        // ── Config-driven backend selection ──────────────────────────
        // An explicit backend in the config skips the environment variables.
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
                    #[cfg(any(
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    opengl: None,
                    convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                    fallback_log: fallback_log::FallbackLog::default(),
                    interpolation: Interpolation::Bilinear,
                    env_interpolation: None,
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
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    });
                }
                #[cfg(not(target_os = "linux"))]
                {
                    log::warn!("G2D requested but not available on this platform, using CPU");
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        #[cfg(any(
                            target_os = "macos",
                            target_os = "ios",
                            target_os = "android",
                            target_os = "windows"
                        ))]
                        #[cfg(feature = "opengl")]
                        opengl: None,
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    });
                }
            }
            ComputeBackend::OpenGl => {
                log::info!("ComputeBackend::OpenGl — OpenGL + CPU fallback");
                #[cfg(target_os = "linux")]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::with_cache_capacity(
                        config.egl_display,
                        config.egl_cache_capacity,
                    ) {
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
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::with_cache_capacity(
                        config.egl_display,
                        config.egl_cache_capacity,
                    ) {
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
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(target_os = "android")]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::with_cache_capacity(
                        config.egl_display,
                        config.egl_cache_capacity,
                    ) {
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
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(target_os = "windows")]
                {
                    #[cfg(feature = "opengl")]
                    let opengl = match GLProcessorThreaded::with_cache_capacity(
                        config.egl_display,
                        config.egl_cache_capacity,
                    ) {
                        Ok(gl) => Some(gl),
                        Err(e) => {
                            log::warn!(
                                "OpenGL requested on Windows but ANGLE/D3D11 init failed: {e:?} \
                                 (fetch ANGLE with `scripts/fetch-angle.sh` and set \
                                 EDGEFIRST_ANGLE_PATH to the directory containing libEGL.dll + \
                                 libGLESv2.dll, or place the two DLLs next to the executable — \
                                 see README.md § Windows GPU Acceleration). Falling back to CPU."
                            );
                            None
                        }
                    };
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        #[cfg(feature = "opengl")]
                        opengl,
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
                        forced_backend: None,
                    }
                    .apply_colorimetry_mode(config.colorimetry));
                }
                #[cfg(not(any(
                    target_os = "linux",
                    target_os = "macos",
                    target_os = "ios",
                    target_os = "android",
                    target_os = "windows"
                )))]
                {
                    log::warn!("OpenGL requested but not available on this platform, using CPU");
                    return Ok(Self {
                        cpu: Some(CPUProcessor::new()),
                        convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                        fallback_log: fallback_log::FallbackLog::default(),
                        interpolation: Interpolation::Bilinear,
                        env_interpolation: None,
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
        if let Some(val) = env.force_backend.as_deref() {
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
                    #[cfg(any(
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    opengl: None,
                    convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                    fallback_log: fallback_log::FallbackLog::default(),
                    interpolation: Interpolation::Bilinear,
                    env_interpolation: None,
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
                            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                            fallback_log: fallback_log::FallbackLog::default(),
                            interpolation: Interpolation::Bilinear,
                            env_interpolation: None,
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
                        let opengl = GLProcessorThreaded::with_cache_capacity(
                            config.egl_display,
                            config.egl_cache_capacity,
                        )
                        .map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced but failed to initialize: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            g2d: None,
                            opengl: Some(opengl),
                            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                            fallback_log: fallback_log::FallbackLog::default(),
                            interpolation: Interpolation::Bilinear,
                            env_interpolation: None,
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::with_cache_capacity(
                            config.egl_display,
                            config.egl_cache_capacity,
                        )
                        .map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced on macOS but ANGLE init failed: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            opengl: Some(opengl),
                            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                            fallback_log: fallback_log::FallbackLog::default(),
                            interpolation: Interpolation::Bilinear,
                            env_interpolation: None,
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(target_os = "android")]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::with_cache_capacity(
                            config.egl_display,
                            config.egl_cache_capacity,
                        )
                        .map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced but native EGL init failed: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            opengl: Some(opengl),
                            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                            fallback_log: fallback_log::FallbackLog::default(),
                            interpolation: Interpolation::Bilinear,
                            env_interpolation: None,
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(target_os = "windows")]
                    #[cfg(feature = "opengl")]
                    {
                        let opengl = GLProcessorThreaded::with_cache_capacity(
                            config.egl_display,
                            config.egl_cache_capacity,
                        )
                        .map_err(|e| {
                            Error::ForcedBackendUnavailable(format!(
                                "opengl forced on Windows but ANGLE/D3D11 init failed: {e:?}"
                            ))
                        })?;
                        Ok(Self {
                            cpu: None,
                            opengl: Some(opengl),
                            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
                            fallback_log: fallback_log::FallbackLog::default(),
                            interpolation: Interpolation::Bilinear,
                            env_interpolation: None,
                            forced_backend: Some(ForcedBackend::OpenGl),
                        }
                        .apply_colorimetry_mode(config.colorimetry))
                    }
                    #[cfg(not(all(
                        any(
                            target_os = "linux",
                            target_os = "macos",
                            target_os = "ios",
                            target_os = "android",
                            target_os = "windows"
                        ),
                        feature = "opengl"
                    )))]
                    {
                        Err(Error::ForcedBackendUnavailable(
                            "opengl backend requires Linux, macOS, Android or Windows with the \
                             'opengl' feature enabled"
                                .into(),
                        ))
                    }
                }
            };
        }

        // ── EDGEFIRST_DISABLE_* ─────────────────────────────────────
        #[cfg(target_os = "linux")]
        let g2d = if env.disable_g2d {
            log::debug!("EDGEFIRST_DISABLE_G2D is set");
            None
        } else {
            match G2DProcessor::new() {
                Ok(g2d_converter) => Some(g2d_converter),
                Err(err) if g2d_is_absent(&err) => {
                    log::debug!("G2D is not available on this host: {err:?}");
                    None
                }
                Err(err) => {
                    log::warn!("Failed to initialize G2D converter: {err:?}");
                    None
                }
            }
        };

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        let opengl = if env.disable_gl {
            log::debug!("EDGEFIRST_DISABLE_GL is set");
            None
        } else {
            match GLProcessorThreaded::with_cache_capacity(
                config.egl_display,
                config.egl_cache_capacity,
            ) {
                Ok(gl_converter) => Some(gl_converter),
                Err(err) => {
                    log::warn!("Failed to initialize GL converter: {err:?}");
                    None
                }
            }
        };

        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        let opengl = if env.disable_gl {
            log::debug!("EDGEFIRST_DISABLE_GL is set");
            None
        } else {
            match GLProcessorThreaded::with_cache_capacity(
                config.egl_display,
                config.egl_cache_capacity,
            ) {
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

        let cpu = if env.disable_cpu {
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
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ))]
            #[cfg(feature = "opengl")]
            opengl,
            convert_fallbacks: std::sync::atomic::AtomicU64::new(0),
            fallback_log: fallback_log::FallbackLog::default(),
            interpolation: Interpolation::Bilinear,
            env_interpolation: None,
            forced_backend: None,
        }
        .apply_colorimetry_mode(config.colorimetry))
    }

    /// Applies the configured [`ColorimetryMode`] to the backend that honours
    /// it (the Linux GL backend); no-op elsewhere.
    fn apply_colorimetry_mode(self, _mode: ColorimetryMode) -> Self {
        #[cfg(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
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
                target_os = "android",
                target_os = "windows"
            ),
            feature = "opengl"
        )))]
        {
            let _ = _mode;
            self
        }
    }

    /// Sets the colorimetry/performance trade-off (see [`ColorimetryMode`])
    /// on the OpenGL backend. No-op if OpenGL is not available. When the
    /// `EDGEFIRST_COLORIMETRY` environment variable is set, this call logs
    /// and keeps the env-selected mode.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    pub fn set_colorimetry_mode(&mut self, mode: ColorimetryMode) -> Result<()> {
        if let Some(ref mut gl) = self.opengl {
            gl.set_colorimetry_mode(mode)?;
        }
        Ok(())
    }

    /// Sets how `convert()` resamples when it resizes (see [`Interpolation`])
    /// on every backend this processor holds, for converts whose options do
    /// not choose their own ([`ConvertOptions::interpolation`]). Bicubic and
    /// area run as [`Interpolation::supported`], with a warning logged once
    /// per mode. The `EDGEFIRST_INTERPOLATION` environment variable, read
    /// when the processor is created, takes precedence: when it is set, this
    /// call keeps the env-selected mode.
    pub fn set_interpolation(&mut self, mode: Interpolation) -> Result<()> {
        let mode = match self.env_interpolation {
            Some(env) => {
                if env != mode.supported() {
                    log::debug!(
                        "EDGEFIRST_INTERPOLATION pins Interpolation::{env:?}; \
                         ignoring Interpolation::{mode:?}"
                    );
                }
                env
            }
            None => runnable_interpolation(mode),
        };
        if mode == self.interpolation {
            return Ok(());
        }
        if let Some(cpu) = self.cpu.as_mut() {
            cpu.set_interpolation(mode);
        }
        #[cfg(target_os = "linux")]
        if let Some(g2d) = self.g2d.as_mut() {
            g2d.set_interpolation(mode);
        }
        #[cfg(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ),
            feature = "opengl"
        ))]
        if let Some(gl) = self.opengl.as_mut() {
            gl.set_interpolation(mode)?;
        }
        self.interpolation = mode;
        Ok(())
    }

    /// The interpolation the processor uses for converts that do not choose
    /// their own, as the backends run it.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// `options` with the interpolation this convert runs: the
    /// `EDGEFIRST_INTERPOLATION` pin, else the options' own (as
    /// [`runnable_interpolation`] maps it), else the processor's.
    fn resolve_interpolation(&self, options: ConvertOptions) -> ConvertOptions {
        let mode = match (self.env_interpolation, options.get_interpolation()) {
            (Some(env), _) => env,
            (None, Some(mode)) => runnable_interpolation(mode),
            (None, None) => self.interpolation,
        };
        options.with_interpolation(Some(mode))
    }

    /// Sets the interpolation mode for int8 proto textures on the OpenGL
    /// backend. No-op if OpenGL is not available.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) -> Result<()> {
        if let Some(ref mut gl) = self.opengl {
            gl.set_int8_interpolation_mode(mode)?;
        }
        Ok(())
    }

    /// The declaration [`create_image`](Self::create_image) allocates a
    /// `memory` destination with: the caller's `access` plus whatever this
    /// processor's own convert may need on the CPU (see
    /// [`predicted_cpu_access`]). `views` marks a destination written through
    /// views, where any CPU write is partial and so needs `ReadWrite`.
    fn producer_access(
        &self,
        access: edgefirst_tensor::CpuAccess,
        format: PixelFormat,
        dtype: DType,
        memory: TensorMemory,
        views: bool,
    ) -> edgefirst_tensor::CpuAccess {
        use edgefirst_tensor::CpuAccess;
        #[cfg(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ),
            feature = "opengl"
        ))]
        let gl = self
            .opengl
            .as_ref()
            .map(|gl| gl.dst_cpu_access(memory, format, dtype));
        #[cfg(not(all(
            any(
                target_os = "linux",
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows"
            ),
            feature = "opengl"
        )))]
        let gl = None;
        #[cfg(target_os = "linux")]
        let g2d = self
            .g2d
            .as_ref()
            .map(|_| g2d::dst_cpu_access(memory, format, dtype));
        #[cfg(not(target_os = "linux"))]
        let g2d = None;
        let cpu = self
            .cpu
            .as_ref()
            .map(|_| Some(cpu::convert_dst_access(None, format, dtype)));
        let needed = predicted_cpu_access(self.forced_backend, gl, g2d, cpu);
        let needed = if views && needed.writes() {
            CpuAccess::ReadWrite
        } else {
            needed
        };
        access.union(needed)
    }

    /// Allocate an image tensor from a declarative
    /// [`ImageDesc`](edgefirst_tensor::ImageDesc) request.
    ///
    /// Without a compression or contiguous request this is
    /// [`create_image`](Self::create_image). With one, the allocation goes
    /// through [`Tensor::image_desc`](edgefirst_tensor::Tensor::image_desc)
    /// without the processor's memory negotiation. A contiguous request is a
    /// CMA DMA-BUF at the same 64-byte GPU pitch `create_image` uses, or an
    /// error; it never falls back to a PBO or host memory. Its CPU access gets
    /// the same widening as `create_image`; a compression request's
    /// `CpuAccess::None` does not.
    pub fn create_image_desc(&self, desc: &edgefirst_tensor::ImageDesc) -> Result<TensorDyn> {
        // A compression request is allocated without widening.
        if desc.compression().is_some() {
            return Ok(TensorDyn::image_desc(desc)?);
        }
        if !desc.contiguous() {
            return self.create_image(
                desc.width(),
                desc.height(),
                desc.format(),
                desc.dtype(),
                desc.memory(),
                desc.access(),
            );
        }
        let access = self.producer_access(
            desc.access(),
            desc.format(),
            desc.dtype(),
            TensorMemory::DmaBuf,
            false,
        );
        Ok(TensorDyn::image_desc(&desc.clone().with_access(access))?)
    }

    /// Create a [`TensorDyn`] image with the best available memory backend.
    ///
    /// Priority: DMA-buf → float PBO (F16/F32) → u8/i8 PBO → system memory.
    ///
    /// **Float dtype behaviour:** when `dtype` is `F16` or `F32` and
    /// [`supported_render_dtypes`] reports the GPU supports that type,
    /// `memory: None` selects a float PBO (Linux) or IOSurface (macOS, F16
    /// only). Without GPU float support the allocation is `TensorMemory::Mem`
    /// and [`convert`] uses the CPU path. `memory: Some(TensorMemory::DmaBuf)`
    /// with `dtype: F32` returns `Error::NotSupported`.
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
    /// * `access` - The CPU access the caller plans on. Use
    ///   [`CpuAccess::None`](edgefirst_tensor::CpuAccess) for a destination
    ///   that only hardware reads (an NPU input). Hardware access is always
    ///   implied. The access this processor's own convert may need on the
    ///   image as a destination is added to the declaration: `Write` when it
    ///   may fill the whole destination on the CPU (the CPU backend, host
    ///   memory, a format or float dtype the GPU path does not render, or an
    ///   `Nv12` image),
    ///   `ReadWrite` when a pass reads back what was written or writes only
    ///   part of the buffer (an `I8` destination through the CPU or G2D, a
    ///   `Bgra` destination through the CPU, a G2D `Rgb` letterbox border).
    ///   The returned tensor's [`cpu_access`](edgefirst_tensor::TensorDyn::cpu_access)
    ///   may be wider than `access`. Any other CPU map beyond the declaration
    ///   fails with [`CpuAccessDenied`](edgefirst_tensor::Error::CpuAccessDenied).
    ///   The widening covers the destination role of convert only: a tensor
    ///   used as a convert source needs `Read` declared by the caller, a CPU
    ///   mask draw needs `ReadWrite` on its destination, and a destination
    ///   written through views (a batch) needs `ReadWrite` for a CPU convert,
    ///   which [`alloc_tile_batch`](Self::alloc_tile_batch) predicts. A
    ///   destination without CPU access (`None` on a GPU-rendered format) is
    ///   written only by the GPU; when the GPU declines a convert into it,
    ///   the CPU fallback is refused with [`Error::CpuFallbackDenied`]
    ///   carrying the GPU's reason. A PBO destination is always `ReadWrite`.
    ///
    /// # Returns
    ///
    /// A [`TensorDyn`] image in the selected memory type.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use edgefirst_image::ImageProcessor;
    /// # use edgefirst_tensor::{CpuAccess, DType, PixelFormat};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let processor = ImageProcessor::new()?;
    ///
    /// // Model input the NPU reads directly — no CPU mapping planned.
    /// let input = processor.create_image(
    ///     640, 640, PixelFormat::Rgb, DType::U8, None, CpuAccess::None,
    /// )?;
    /// # let _ = input;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// A display canvas the GPU renders and a Wayland compositor shows
    /// (linux-dmabuf), with a CPU-written overlay. Neither is read by the
    /// CPU; on Linux both come from uncached CMA where available.
    ///
    /// ```rust,no_run
    /// # use edgefirst_image::{ImageProcessor, ImageProcessorTrait, MaskOverlay};
    /// # use edgefirst_tensor::{CpuAccess, DType, DetectBox, PixelFormat, TensorDyn, TensorMemory, TensorMapTrait, TensorTrait};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// # let camera: TensorDyn = unimplemented!();
    /// # let boxes: Vec<DetectBox> = Vec::new();
    /// let mut processor = ImageProcessor::new()?;
    /// let dma = Some(TensorMemory::DmaBuf);
    /// let frame = processor.create_image(1920, 1080, PixelFormat::Rgba, DType::U8, dma, CpuAccess::None)?;
    /// let mut canvas = processor.create_image(1920, 1080, PixelFormat::Rgba, DType::U8, dma, CpuAccess::None)?;
    ///
    /// // Translucent orange boxes, 2 px wide.
    /// processor.set_class_colors(&[[255, 96, 0, 160]])?;
    /// processor.set_box_line_width(2)?;
    ///
    /// let mut frame = frame;
    /// processor.convert(&camera, &mut frame, None)?;
    /// processor.draw_decoded_masks(&mut canvas, &boxes, &[], MaskOverlay::new().with_background(&frame))?;
    ///
    /// // Hand the canvas to the compositor: its fd, pitch and format.
    /// let fd = canvas.clone_fd()?;
    /// let stride = canvas.effective_row_stride();
    /// # let _ = (fd, stride);
    ///
    /// // A label strip the CPU renders: written whole, never read back.
    /// let overlay = processor.create_image(640, 64, PixelFormat::Rgba, DType::U8, dma, CpuAccess::Write)?;
    /// overlay.as_u8().unwrap().map_for_write()?.as_mut_slice().fill(0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The returned image's [`cpu_access`](TensorDyn::cpu_access) may be
    /// wider than `access`; a wider declaration selects cached memory.
    ///
    /// # Pitch alignment for DMA-backed allocations
    ///
    /// A `TensorMemory::DmaBuf` allocation is padded so every row pitch is a
    /// multiple of [`GPU_DMA_BUF_PITCH_ALIGNMENT_BYTES`]. `tensor.width()`
    /// returns the requested `width`; the padding is carried by
    /// [`TensorDyn::row_stride`] / `effective_row_stride()`. Byte offsets are
    /// computed from the stride, not `width × bytes_per_pixel`; the CPU
    /// mapping spans `stride × height` bytes.
    ///
    /// A planar image pads each row of each plane: a plane is
    /// `stride × height` bytes and the next plane starts there. A 672×512
    /// `PlanarRgb` image has a 704-byte pitch and three 704×512 planes.
    /// [`TensorDyn::copy_to_flat`] copies out the tight `[C, H, W]` bytes.
    ///
    /// Pre-aligned widths (640, 1280, 1920, 3008, 3840 …) allocate exactly
    /// `width × bpp × height` bytes. PBO and Mem allocations are not padded.
    ///
    /// See also [`align_width_for_gpu_pitch`].
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
        access: edgefirst_tensor::CpuAccess,
    ) -> Result<TensorDyn> {
        self.create_image_with(width, height, format, dtype, memory, access, false)
    }

    /// A DMA image for [`create_image_with`](Self::create_image_with), its
    /// rows padded to the GPU pitch alignment when that exceeds the natural
    /// pitch. The tensor's `row_stride()` reports the pitch.
    #[cfg(target_os = "linux")]
    fn create_dma_image(
        &self,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        access: edgefirst_tensor::CpuAccess,
        views: bool,
    ) -> Result<TensorDyn> {
        let dma_access = self.producer_access(access, format, dtype, TensorMemory::DmaBuf, views);
        let natural =
            primary_plane_bpp(format, dtype.size()).and_then(|bpp| width.checked_mul(bpp));
        let padded = natural
            .and_then(align_pitch_bytes_to_gpu_alignment)
            .filter(|&stride| {
                format.layout() == edgefirst_tensor::PixelLayout::Packed
                    && natural.is_some_and(|n| stride > n)
            });
        let memory = Some(edgefirst_tensor::TensorMemory::DmaBuf);
        match padded {
            Some(stride) => {
                log::debug!(
                    "create_image: padding row stride for {format:?} {width}x{height} \
                     from natural pitch to {stride} bytes for GPU alignment"
                );
                Ok(TensorDyn::image_with_stride(
                    width, height, format, dtype, stride, memory, dma_access,
                )?)
            }
            None => Ok(TensorDyn::image(
                width, height, format, dtype, memory, dma_access,
            )?),
        }
    }

    /// A u8/i8 PBO image for [`create_image_with`](Self::create_image_with),
    /// or `None` when the dtype is wider or there is no PBO. The image is
    /// `ReadWrite`.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[cfg(feature = "opengl")]
    fn create_byte_pbo_image(
        &self,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
    ) -> Option<TensorDyn> {
        if dtype.size() != 1 {
            return None;
        }
        let gl = self.opengl.as_ref()?;
        let t = gl
            .create_pbo_image(width, height, format)
            .map_err(|e| log::debug!("PBO image creation failed, falling back to Mem: {e:?}"))
            .ok()?;
        if dtype != DType::I8 {
            return Some(TensorDyn::from(t));
        }
        debug_assert!(
            t.chroma().is_none(),
            "PBO i8 transmute requires chroma == None"
        );
        // SAFETY: `Tensor<u8>` and `Tensor<i8>` are layout-identical.
        let t_i8: Tensor<i8> = unsafe { std::mem::transmute(t) };
        Some(TensorDyn::from(t_i8))
    }

    /// [`create_image`](Self::create_image), for a destination written
    /// through views when `views` is set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_image_with(
        &self,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
        access: edgefirst_tensor::CpuAccess,
        views: bool,
    ) -> Result<TensorDyn> {
        #[cfg(target_os = "linux")]
        let try_dma = || self.create_dma_image(width, height, format, dtype, access, views);

        // An explicit memory type is honoured directly. On Linux
        // `TensorMemory::DmaBuf` gets a padded stride.
        match memory {
            #[cfg(target_os = "linux")]
            Some(TensorMemory::DmaBuf) => {
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
                return Ok(TensorDyn::image(
                    width,
                    height,
                    format,
                    dtype,
                    Some(mem),
                    self.producer_access(access, format, dtype, mem, views),
                )?);
            }
            None => {}
        }

        // macOS, Android, Windows: with the GL backend active, tries a zero-copy
        // allocation (IOSurface, AHardwareBuffer, D3D11 texture) and continues
        // to host storage when it is declined.
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        if let Some(gl) = self.opengl.as_ref() {
            let _ = gl;
            // Float textures only where the display can render into them.
            let support = self.supported_render_dtypes();
            let dtype_ok =
                !matches!(dtype, DType::F16 | DType::F32) || float_pbo_eligible(dtype, support);
            if dtype_ok {
                match TensorDyn::image(
                    width,
                    height,
                    format,
                    dtype,
                    Some(edgefirst_tensor::TensorMemory::DmaBuf),
                    self.producer_access(access, format, dtype, TensorMemory::DmaBuf, views),
                ) {
                    Ok(img) => return Ok(img),
                    Err(e) => {
                        log::debug!(
                            "create_image: zero-copy Dma allocation declined \
                             ({format:?}/{dtype:?} {width}x{height}): {e:?}; using fallback storage"
                        );
                    }
                }
            }
        }

        // Linux: tries DMA first unless GL selected PBO as its transfer path.
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

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        #[cfg(feature = "opengl")]
        if let Some(img) = self.create_byte_pbo_image(width, height, format, dtype) {
            return Ok(img);
        }

        // Tries a float PBO when the GPU backend reports support for this dtype.
        #[cfg(any(target_os = "linux", target_os = "windows"))]
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
            self.producer_access(access, format, dtype, TensorMemory::Mem, views),
        )?)
    }

    /// Import an external DMA-BUF image.
    ///
    /// Each [`PlaneDescriptor`] owns an already-duped fd; this method
    /// consumes the descriptors and takes ownership of those fds whether
    /// the call succeeds or fails.
    ///
    /// The caller must ensure the DMA-BUF allocation is large enough for the
    /// specified width, height, format, and any stride/offset on the plane
    /// descriptors. No buffer-size validation is performed.
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
    ///   format or for NV16, or the fd is not DMA-backed
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

        // Reads stride and offset from the descriptors before they are consumed.
        let image_stride = image.stride();
        let image_offset = image.offset();
        let chroma_stride = chroma.as_ref().and_then(|c| c.stride());
        let chroma_offset = chroma.as_ref().and_then(|c| c.offset());

        if let Some(chroma_pd) = chroma {
            // ── Multiplane path ──────────────────────────────────────
            // Multiplane tensors are `Tensor<u8>` or `Tensor<i8>`; other dtypes
            // are rejected.
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
                    // NV12 (4:2:0): ceil(H/2) chroma rows.
                    height.div_ceil(2)
                }
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
            if luma.memory() != TensorMemory::DmaBuf {
                return Err(Error::NotSupported(format!(
                    "luma fd must be DMA-backed, got {:?}",
                    luma.memory()
                )));
            }

            let mut chroma_tensor =
                Tensor::<u8>::from_fd(chroma_pd.into_fd(), &[chroma_h, width], Some("chroma"))?;
            if chroma_tensor.memory() != TensorMemory::DmaBuf {
                return Err(Error::NotSupported(format!(
                    "chroma fd must be DMA-backed, got {:?}",
                    chroma_tensor.memory()
                )));
            }

            // Sets the chroma plane's geometry before `from_planes` consumes it.
            if let Some(s) = chroma_stride {
                if s < width {
                    return Err(Error::InvalidShape(format!(
                        "chroma stride {s} < minimum {width} for {format:?}"
                    )));
                }
                chroma_tensor.set_row_stride_unchecked(s);
            }
            if let Some(o) = chroma_offset {
                chroma_tensor.set_plane_offset(o);
            }

            // `from_planes` creates the combined tensor with the format set.
            let mut tensor = Tensor::<u8>::from_planes(luma, chroma_tensor, format)?;

            // Apply stride/offset to the combined tensor (luma plane)
            if let Some(s) = image_stride {
                tensor.set_row_stride(s)?;
            }
            if let Some(o) = image_offset {
                tensor.set_plane_offset(o);
            }

            if dtype == DType::I8 {
                // SAFETY: `Tensor<u8>` and `Tensor<i8>` have identical layout; the
                // assertion checks their size and alignment at compile time.
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
            // Canonical shape: Packed [H,W,C], Planar [C,H,W], SemiPlanar
            // [total_h, W].
            let shape = format.allocation_shape(width, height).ok_or_else(|| {
                Error::NotSupported(format!(
                    "unsupported pixel format for import_image: {format:?}"
                ))
            })?;
            let tensor = TensorDyn::from_fd(image.into_fd(), &shape, dtype, None)?;
            if tensor.memory() != TensorMemory::DmaBuf {
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
    /// The processor decodes via the provided
    /// [`Decoder`](edgefirst_decoder::Decoder), selects the rendering path
    /// (hybrid CPU+GL or fused GPU), and composites masks onto `dst`.
    ///
    /// Returns the detected bounding boxes.
    ///
    /// **Requires the `decode` feature**, which is off by default. Without
    /// the feature, sequence the decode and the draw:
    ///
    /// ```ignore
    /// let mut boxes = Vec::new();
    /// match decoder.decode_proto(outputs, &mut boxes)? {
    ///     Some(protos) => proc.draw_proto_masks(dst, &boxes, &protos, overlay)?,
    ///     None => {
    ///         let mut masks = Vec::new();
    ///         decoder.decode(outputs, &mut boxes, &mut masks)?;
    ///         proc.draw_decoded_masks(dst, &boxes, &masks, overlay)?;
    ///     }
    /// }
    /// ```
    #[cfg(feature = "decode")]
    pub fn draw_masks(
        &mut self,
        decoder: &edgefirst_decoder::Decoder,
        outputs: &[&TensorDyn],
        dst: &mut TensorDyn,
        overlay: MaskOverlay<'_>,
    ) -> Result<Vec<DetectBox>> {
        let mut output_boxes = Vec::with_capacity(100);

        // Tries the proto path first.
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
            // `decode_proto_tracked` returns `None` for detection-only and
            // ModelPack models without calling the tracker; this branch calls it.
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
    /// (0 = background, 255 = full confidence), not binary thresholded.
    ///
    /// The returned [`Vec<Segmentation>`] can be passed to
    /// [`ImageProcessorTrait::draw_decoded_masks`] for rendering.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoConverter`] if the CPU backend is not available.
    pub fn materialize_masks(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        content_region: Option<[f32; 4]>,
        resolution: MaskResolution,
    ) -> Result<Vec<Segmentation>> {
        let cpu = self.cpu.as_mut().ok_or(Error::NoConverter)?;
        match resolution {
            MaskResolution::Proto => {
                cpu.materialize_segmentations(detect, proto_data, content_region)
            }
            MaskResolution::Scaled { width, height } => cpu.materialize_scaled_segmentations(
                detect,
                proto_data,
                content_region,
                width,
                height,
            ),
        }
    }
}

impl ImageProcessor {
    /// Converts `src` into each destination of `items` with its options and
    /// completes them all. On the OpenGL backend the converts and the GPU
    /// sync run as one worker message; a destination GL declines, or every
    /// destination when the batch itself fails, converts eagerly on the next
    /// backend, unless GL is forced.
    pub(crate) fn convert_batch(
        &mut self,
        src: &TensorDyn,
        items: &mut [(TensorDyn, ConvertOptions)],
    ) -> Result<()> {
        for (_, options) in items.iter_mut() {
            *options = self.resolve_interpolation(*options);
        }
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        {
            let gl_forced = matches!(self.forced_backend, Some(ForcedBackend::OpenGl));
            if gl_forced || self.forced_backend.is_none() {
                if let Some(opengl) = self.opengl.as_mut() {
                    match opengl.convert_batch(src, items) {
                        Ok(results) => {
                            for ((dst, options), result) in items.iter_mut().zip(results) {
                                if let Err(e) = result {
                                    log::trace!("convert_batch: gl declined: {e}; eager fallback");
                                    if gl_forced {
                                        return Err(e);
                                    }
                                    self.convert(src, dst, Some(*options))?;
                                }
                            }
                            return Ok(());
                        }
                        Err(e) => {
                            if gl_forced {
                                return Err(e);
                            }
                            if let Some((dst, _)) = items.first() {
                                self.fallback_log.note("opengl", src, dst, &e);
                            }
                        }
                    }
                }
            }
        }
        for (dst, options) in items.iter_mut() {
            self.convert(src, dst, Some(*options))?;
        }
        Ok(())
    }
}

impl ImageProcessorTrait for ImageProcessor {
    /// Converts the source image to the destination image format and size. The
    /// image is cropped first, then flipped, then rotated.
    ///
    /// Hardware accelerators are tried first, then the CPU.
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<ConvertOptions>,
    ) -> Result<()> {
        let options = self.resolve_interpolation(options.unwrap_or_default());
        let (rotation, flip) = (options.get_rotation(), options.get_flip());
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
                        let r = cpu.convert(src, dst, Some(options));
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
                        let r = g2d.convert(src, dst, Some(options));
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
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        let r = opengl.convert(src, dst, Some(options));
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
        #[allow(unused_mut)]
        let mut declined: Vec<(&'static str, Error)> = Vec::new();
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            match opengl.convert(src, dst, Some(options)) {
                Ok(_) => {
                    log::trace!(
                        "convert: auto selected=opengl for {src_fmt:?}→{dst_fmt:?} ({:?})",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    self.convert_fallbacks
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.fallback_log.note("opengl", src, dst, &e);
                    declined.push(("opengl", e));
                }
            }
        }

        #[cfg(target_os = "linux")]
        if let Some(g2d) = self.g2d.as_mut() {
            // Skips G2D when the colorimetry of the conversion's YUV side is
            // full-range or BT.2020. YUV→RGB uses the source colorimetry, RGB→YUV
            // the destination.
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
                self.fallback_log.note(
                    "g2d",
                    src,
                    dst,
                    &Error::NotSupported(
                        "G2D does not express full-range or BT.2020 colorimetry".into(),
                    ),
                );
            } else {
                match g2d.convert(src, dst, Some(options)) {
                    Ok(_) => {
                        log::trace!(
                            "convert: auto selected=g2d for {src_fmt:?}→{dst_fmt:?} ({:?})",
                            start.elapsed()
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        self.fallback_log.note("g2d", src, dst, &e);
                        declined.push(("g2d", e));
                    }
                }
            }
        }

        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.convert(src, dst, Some(options)) {
                Ok(_) => {
                    log::trace!(
                        "convert: auto selected=cpu for {src_fmt:?}→{dst_fmt:?} ({:?})",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("convert: auto cpu failed {src_fmt:?}→{dst_fmt:?}: {e}");
                    return Err(with_gpu_declines(e, declined));
                }
            }
        }
        Err(Error::NoConverter)
    }

    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<ConvertOptions>,
    ) -> Result<()> {
        let options = self.resolve_interpolation(options.unwrap_or_default());
        // Uses the GL backend's deferred path when GL is forced or
        // auto-selectable, and an eager convert when GL declines.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        {
            let gl_forced = matches!(self.forced_backend, Some(ForcedBackend::OpenGl));
            if gl_forced || self.forced_backend.is_none() {
                if let Some(opengl) = self.opengl.as_mut() {
                    match opengl.convert_deferred(src, dst, Some(options)) {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            log::trace!("convert_deferred: gl declined: {e}; eager fallback");
                            // A forced-GL caller gets the GL error.
                            if gl_forced {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        self.convert(src, dst, Some(options))
    }

    fn flush(&mut self) -> Result<()> {
        let _span = tracing::trace_span!("image.flush").entered();
        // Only the OpenGL backend defers.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
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

        // Un-letterboxes detect boxes and segmentation bboxes when the overlay
        // has a content region.
        let lb_boxes: Vec<DetectBox>;
        let lb_segs: Vec<Segmentation>;
        let (detect, segmentation) = if let Some(lb) = overlay.content_region {
            lb_boxes = detect.iter().map(|&d| unletter_bbox(d, lb)).collect();
            // With one segmentation per box, each segmentation takes its box's transformed bounds.
            if segmentation.len() == lb_boxes.len() {
                // Copies the mask bytes and replaces the four bounds.
                lb_segs = segmentation
                    .iter()
                    .zip(lb_boxes.iter())
                    .map(|(s, d)| copy_segmentation(s, d.bbox.into()))
                    .collect::<Result<Vec<_>>>()?;
                (lb_boxes.as_slice(), lb_segs.as_slice())
            } else {
                // Otherwise the segmentation bounds are left as they are.
                (lb_boxes.as_slice(), segmentation)
            }
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
                    // G2D draws empty frames only (clear or background blit).
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        return g2d.draw_decoded_masks(dst, detect, segmentation, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_decoded_masks(dst, detect, segmentation, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Auto dispatch ──────────────────────────────────────────
        // Empty frames go to G2D when available.
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

        // Populated frames, or G2D unavailable: GL first, then CPU.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
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

        // Un-letterboxes detect boxes for rendering when the overlay has a
        // content region. `materialize_segmentations` takes the original `detect`
        // and `overlay.content_region`.
        let lb_boxes: Vec<DetectBox>;
        let render_detect = if let Some(lb) = overlay.content_region {
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
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_proto_masks(dst, render_detect, proto_data, overlay);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Auto dispatch ──────────────────────────────────────────
        // Empty frames go to G2D when available.
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

        // Hybrid path: CPU materialize, then GL overlay.
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        if let (Some(_), Some(_)) = (self.cpu.as_ref(), self.opengl.as_ref()) {
            let segmentation = match self.cpu.as_mut() {
                Some(cpu) => {
                    log::trace!(
                        "draw_proto_masks started with hybrid (cpu+opengl) in {:?}",
                        start.elapsed()
                    );
                    cpu.materialize_segmentations(detect, proto_data, overlay.content_region)?
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
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.set_class_colors(colors);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // Sets the colours on every available backend.
        let mut backends = 0usize;
        let mut first_err = None;
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            backends += 1;
            match opengl.set_class_colors(colors) {
                Ok(()) => log::trace!("colors set with opengl in {:?}", start.elapsed()),
                Err(e) => {
                    log::trace!("colors didn't set with opengl: {e:?}");
                    first_err.get_or_insert(e);
                }
            }
        }
        if let Some(cpu) = self.cpu.as_mut() {
            backends += 1;
            match cpu.set_class_colors(colors) {
                Ok(()) => log::trace!("colors set with cpu in {:?}", start.elapsed()),
                Err(e) => {
                    log::trace!("colors didn't set with cpu: {e:?}");
                    first_err.get_or_insert(e);
                }
            }
        }
        // Returns the first error.
        match first_err {
            Some(e) => Err(e),
            None if backends > 0 => Ok(()),
            None => Err(Error::NoConverter),
        }
    }

    fn set_box_line_width(&mut self, width: u32) -> Result<()> {
        check_box_line_width(width)?;

        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.set_box_line_width(width);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => Err(Error::NotSupported(
                    "g2d does not support set_box_line_width".into(),
                )),
                ForcedBackend::OpenGl => {
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "macos",
                        target_os = "ios",
                        target_os = "android",
                        target_os = "windows"
                    ))]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.set_box_line_width(width);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // Sets the width on every drawing backend.
        let mut backends = 0usize;
        let mut first_err = None;
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ))]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            backends += 1;
            if let Err(e) = opengl.set_box_line_width(width) {
                log::trace!("box line width didn't set with opengl: {e:?}");
                first_err.get_or_insert(e);
            }
        }
        if let Some(cpu) = self.cpu.as_mut() {
            backends += 1;
            if let Err(e) = cpu.set_box_line_width(width) {
                log::trace!("box line width didn't set with cpu: {e:?}");
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None if backends > 0 => Ok(()),
            None => Err(Error::NoConverter),
        }
    }
}

// ---------------------------------------------------------------------------
// Image loading / saving helpers
// ---------------------------------------------------------------------------

/// Test helper: peeks the image header, allocates a tensor sized to the
/// image (with DMA pitch padding on Linux when requested), and decodes via
/// [`edgefirst_codec`].
#[cfg(all(test, feature = "codec"))]
pub(crate) fn load_image_test_helper(
    image: &[u8],
    format: Option<PixelFormat>,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};

    // Peeks the source header for its native format and dimensions.
    let info = peek_info(image)?;
    let native_fmt = info.format;
    let w = info.width;
    let h = info.height;

    let mut decoder = ImageDecoder::new();

    // Decodes into a tensor allocated for the native layout.
    #[cfg(target_os = "linux")]
    let native_src = {
        if let Some(aligned_pitch) = padded_dma_pitch_for(native_fmt, w, &memory) {
            let mut dma = Tensor::<u8>::image_with_stride(
                w,
                h,
                native_fmt,
                aligned_pitch,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            dma.load_image(&mut decoder, image)?;
            TensorDyn::from(dma)
        } else {
            let mut img = Tensor::<u8>::image(
                w,
                h,
                native_fmt,
                memory,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            img.load_image(&mut decoder, image)?;
            TensorDyn::from(img)
        }
    };
    #[cfg(not(target_os = "linux"))]
    let native_src = {
        let mut img = Tensor::<u8>::image(
            w,
            h,
            native_fmt,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        img.load_image(&mut decoder, image)?;
        TensorDyn::from(img)
    };

    // Converts to the requested format with a CPU-backed processor when it
    // differs from the native one.
    match format {
        Some(f) if f != native_fmt => {
            let mut dst = TensorDyn::image(
                w,
                h,
                f,
                DType::U8,
                memory,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )?;
            #[allow(clippy::needless_update)]
            let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::Cpu,
                ..Default::default()
            })?;
            proc.convert(&native_src, &mut dst, None)?;
            Ok(dst)
        }
        _ => Ok(native_src),
    }
}

/// Save a [`TensorDyn`] image as a JPEG file.
///
/// Only packed RGB and RGBA formats are supported.
pub fn save_jpeg(tensor: &TensorDyn, path: impl AsRef<std::path::Path>, quality: u8) -> Result<()> {
    let t = tensor.as_typed::<u8>().ok_or(Error::UnsupportedFormat(
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
    let tensor_map = t.map_read()?;

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

/// Default bounding-box outline thickness in pixels, used until
/// [`ImageProcessorTrait::set_box_line_width`] sets another.
pub const DEFAULT_BOX_LINE_WIDTH: u32 = 3;

/// Validates a [`ImageProcessorTrait::set_box_line_width`] argument.
pub(crate) fn check_box_line_width(width: u32) -> Result<()> {
    if width == 0 {
        return Err(Error::InvalidOptions(
            "box line width must be at least 1 pixel".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod alignment_tests {
    use super::*;

    #[test]
    fn align_width_rgba8_common_widths() {
        // RGBA8: bpp=4, lcm(64,4)=64, widths round to a multiple of 16 px.
        assert_eq!(align_width_for_gpu_pitch(640, 4), 640); // 2560 byte pitch
        assert_eq!(align_width_for_gpu_pitch(1280, 4), 1280); // 5120
        assert_eq!(align_width_for_gpu_pitch(1920, 4), 1920); // 7680
        assert_eq!(align_width_for_gpu_pitch(3840, 4), 3840); // 15360
        assert_eq!(align_width_for_gpu_pitch(3004, 4), 3008); // 12016 → 12032
        assert_eq!(align_width_for_gpu_pitch(3000, 4), 3008); // 12000 → 12032
        assert_eq!(align_width_for_gpu_pitch(17, 4), 32); // 68 → 128
        assert_eq!(align_width_for_gpu_pitch(1, 4), 16); // 4 → 64
    }

    #[test]
    fn align_width_rgb888_packed() {
        // RGB888: bpp=3, lcm(64,3)=192, widths round to a multiple of 64 px.
        assert_eq!(align_width_for_gpu_pitch(64, 3), 64); // 192 byte pitch
        assert_eq!(align_width_for_gpu_pitch(640, 3), 640); // 1920
        assert_eq!(align_width_for_gpu_pitch(1, 3), 64); // 3 → 192
        assert_eq!(align_width_for_gpu_pitch(65, 3), 128); // 195 → 384
        for w in [3004usize, 1281, 100, 17] {
            let padded = align_width_for_gpu_pitch(w, 3);
            assert!(padded >= w);
            assert_eq!((padded * 3) % 64, 0);
            assert_eq!((padded * 3) % 3, 0);
        }
    }

    #[test]
    fn align_width_grey_u8() {
        // Grey: bpp=1, lcm(64,1)=64, widths round to a multiple of 64 px.
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
        // Checks returned width >= input width, including where `width * bpp` overflows.
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
        // A pre-aligned width near usize::MAX is returned unchanged.
        let aligned_extreme = usize::MAX - 15; // 16-pixel boundary for RGBA8
        assert_eq!(
            align_width_for_gpu_pitch(aligned_extreme, 4),
            aligned_extreme
        );
        // A misaligned width near usize::MAX is returned as given.
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
        // Coprime values whose product exceeds usize::MAX return None.
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
    #[cfg(target_os = "linux")]
    use crate::test_support::is_dma_available;
    use crate::{CPUProcessor, Rotation};
    use edgefirst_tensor::{TensorMapTrait, TensorMemory, TensorTrait};
    use image::buffer::ConvertBuffer;

    /// Calls `ImageProcessorTrait::convert()` on two `TensorDyn`s and returns
    /// the result, the source and the destination.
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
        let result = proc.convert(
            &src_dyn,
            &mut dst_dyn,
            Some(
                ConvertOptions::new()
                    .rotation(rotation)
                    .flip(flip)
                    .crop(crop),
            ),
        );
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

    /// Tiles deferred into row-bands of one tall destination each equal the same source converted standalone.
    #[test]
    fn batch_view_dst_tiles_match_standalone() {
        let mut proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — ImageProcessor init failed ({e:?})",
                    function!()
                ));
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
        // Tall destination: N stacked row-bands.
        let Some(parent) = crate::test_support::dma_allocated(
            TensorDyn::image(
                w,
                n * h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "tall DMA destination",
        ) else {
            return;
        };

        for (i, &c) in colors.iter().enumerate().take(n) {
            let mut tile = parent.view(Region::new(0, i * h, w, h)).unwrap();
            proc.convert_deferred(&make_src(c), &mut tile, None)
                .unwrap_or_else(|e| panic!("convert_deferred tile {i}: {e:?}"));
        }
        proc.flush().unwrap();

        for (i, &c) in colors.iter().enumerate().take(n) {
            // The oracle: a standalone full-buffer convert of the same source.
            let mut solo = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            proc.convert(&make_src(c), &mut solo, None).unwrap();

            let band = parent.view(Region::new(0, i * h, w, h)).unwrap();
            let band_bytes = band
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            let solo_bytes = solo
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            assert_eq!(
                band_bytes, solo_bytes,
                "tile {i}: band differs from standalone convert (placement or sibling wipe)"
            );
            assert!(
                band_bytes.as_chunks::<4>().0.iter().all(|&p| p == c),
                "tile {i}: band is not the expected solid color {c:?} (sibling wipe?)"
            );
        }
    }

    /// Builds a `w`×`h` RGBA frame whose pixels encode their position.
    #[cfg(test)]
    fn gradient_frame(w: usize, h: usize) -> TensorDyn {
        let mut bytes = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                bytes[i] = x as u8;
                bytes[i + 1] = y as u8;
                bytes[i + 2] = (x ^ y) as u8;
                bytes[i + 3] = 255;
            }
        }
        load_bytes_to_tensor(w, h, PixelFormat::Rgba, Some(TensorMemory::Mem), &bytes).unwrap()
    }

    /// `set_box_line_width` validates its argument and reaches the default chain and a forced CPU backend.
    #[test]
    fn set_box_line_width_reaches_the_drawing_backend() {
        use crate::test_support::{box_drawn_columns, box_expected_span, BOX_DIM};
        let configs = [
            ImageProcessorConfig::default(),
            ImageProcessorConfig {
                backend: ComputeBackend::Cpu,
                ..Default::default()
            },
        ];
        for config in configs {
            let backend = config.backend;
            let mut proc = ImageProcessor::with_config(config).unwrap();
            let err = proc.set_box_line_width(0).unwrap_err();
            assert!(
                matches!(err, Error::InvalidOptions(_)),
                "{backend:?}: {err:?}"
            );
            let flat =
                crate::test_support::draw_box_fixture(&mut proc, [200, 100, 50, 255], Some(4), 1.0);
            assert_eq!(
                box_drawn_columns(&flat, BOX_DIM / 2),
                box_expected_span(4),
                "{backend:?}"
            );
        }
    }

    /// A `tile_into` band equals a standalone crop-convert of the same source region on the CPU backend.
    #[test]
    fn tile_into_cpu_distinct_content_parity() {
        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .expect("the CPU backend is always available");
        let (fw, fh) = (96usize, 64usize);
        let src = gradient_frame(fw, fh);
        let cfg = TilingConfig::new(32, 32).with_overlap(0.0); // exact 3×2 tiling
        let n = tile_grid(fh, fw, 32, 32, 0.0).len();
        assert_eq!(n, 6);

        let mut parent = proc
            .alloc_tile_batch(
                n,
                &cfg,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        let placements = proc.tile_into(&src, &mut parent, &cfg).unwrap();
        assert_eq!(placements.len(), n);

        for p in &placements {
            let source = Region::new(
                p.origin.0 as usize,
                p.origin.1 as usize,
                p.crop_size.0 as usize,
                p.crop_size.1 as usize,
            );
            let mut solo = TensorDyn::image(
                32,
                32,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            proc.convert(
                &src,
                &mut solo,
                Some(
                    crate::ConvertOptions::new().crop(
                        Crop::default()
                            .with_source(Some(source))
                            .with_fit(Fit::Stretch),
                    ),
                ),
            )
            .unwrap();
            let band = parent.view(Region::new(0, p.index * 32, 32, 32)).unwrap();
            let band_bytes = band
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            let solo_bytes = solo
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            assert_eq!(
                band_bytes, solo_bytes,
                "tile {} band differs from standalone crop-convert",
                p.index
            );
        }
    }

    /// `tile_one` into a single slot matches the corresponding `tile_into` band.
    #[test]
    fn tile_one_matches_tile_into_band() {
        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .expect("the CPU backend is always available");
        let (fw, fh) = (96usize, 64usize);
        let src = gradient_frame(fw, fh);
        let cfg = TilingConfig::new(32, 32).with_overlap(0.0);
        let n = tile_grid(fh, fw, 32, 32, 0.0).len();

        let mut parent = proc
            .alloc_tile_batch(
                n,
                &cfg,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        proc.tile_into(&src, &mut parent, &cfg).unwrap();

        let plan = proc.plan_tiles(fw, fh, &cfg).unwrap();
        for p in &plan {
            let mut slot = TensorDyn::image(
                32,
                32,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            proc.tile_one(&src, &mut slot, p, &cfg).unwrap();
            proc.flush().unwrap();
            let band = parent.view(Region::new(0, p.index * 32, 32, 32)).unwrap();
            let slot_bytes = slot
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            let band_bytes = band
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .to_vec();
            assert_eq!(
                slot_bytes, band_bytes,
                "tile {} stream != batch band",
                p.index
            );
        }
    }

    /// `tile_into` through the auto backend with a DMA parent matches standalone crop-converts.
    #[test]
    fn tile_into_auto_dma_parity() {
        let mut proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — ImageProcessor init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        let (fw, fh) = (96usize, 64usize);
        let src = gradient_frame(fw, fh);
        let cfg = TilingConfig::new(32, 32).with_overlap(0.0);
        let n = tile_grid(fh, fw, 32, 32, 0.0).len();

        let Some(mut parent) = crate::test_support::dma_allocated(
            proc.alloc_tile_batch(
                n,
                &cfg,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "tall DMA parent",
        ) else {
            return;
        };
        let placements = proc.tile_into(&src, &mut parent, &cfg).unwrap();

        for p in &placements {
            let source = Region::new(
                p.origin.0 as usize,
                p.origin.1 as usize,
                p.crop_size.0 as usize,
                p.crop_size.1 as usize,
            );
            let mut solo = TensorDyn::image(
                32,
                32,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            proc.convert(
                &src,
                &mut solo,
                Some(
                    crate::ConvertOptions::new().crop(
                        Crop::default()
                            .with_source(Some(source))
                            .with_fit(Fit::Stretch),
                    ),
                ),
            )
            .unwrap();
            let band = parent.view(Region::new(0, p.index * 32, 32, 32)).unwrap();
            compare_images(
                &band,
                &solo,
                0.98,
                &format!("{}_tile{}", function!(), p.index),
            );
        }
    }

    /// `tile_into` matches the CPU tile by tile while a second processor converts on another thread.
    #[test]
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    fn tile_into_matches_the_cpu_beside_a_converting_processor() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        const ROUNDS: usize = 20;
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

        let mut proc = match ImageProcessor::new() {
            Ok(p) if p.opengl.is_some() => p,
            Ok(_) => {
                crate::test_support::report_skip(&format!("{} — no OpenGL backend", function!()));
                return;
            }
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — ImageProcessor init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        let (fw, fh) = (96usize, 64usize);
        let cfg = TilingConfig::new(32, 32).with_overlap(0.0);
        let n = tile_grid(fh, fw, 32, 32, 0.0).len();
        let alloc = move |proc: &ImageProcessor| {
            crate::test_support::dma_allocated(
                proc.alloc_tile_batch(
                    n,
                    &cfg,
                    PixelFormat::Rgba,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                "tall DMA parent",
            )
        };
        if alloc(&proc).is_none() {
            return;
        }

        /// Sets the flag when dropped.
        struct StopOnDrop<'a>(&'a AtomicBool);
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let src = gradient_frame(fw, fh);
                let stop = AtomicBool::new(false);
                std::thread::scope(|scope| {
                    let (ready_tx, ready_rx) = mpsc::channel();
                    scope.spawn(|| {
                        let mut p = ImageProcessor::new().unwrap();
                        let other_src = gradient_frame(640, 480);
                        let crop = Crop::default()
                            .with_source(Some(Region::new(13, 7, 400, 300)))
                            .with_fit(Fit::Stretch);
                        let mut ready = Some(ready_tx);
                        while !stop.load(Ordering::Relaxed) {
                            let mut dst = TensorDyn::image(
                                320,
                                240,
                                PixelFormat::Rgba,
                                DType::U8,
                                Some(TensorMemory::DmaBuf),
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap();
                            p.convert(&other_src, &mut dst, Some(ConvertOptions::new().crop(crop)))
                                .unwrap();
                            if let Some(tx) = ready.take() {
                                let _ = tx.send(());
                            }
                        }
                    });
                    let _stop = StopOnDrop(&stop);
                    ready_rx
                        .recv()
                        .expect("the contending processor converts once before the rounds");

                    let mut failures = Vec::new();
                    for round in 0..ROUNDS {
                        let mut parent = alloc(&proc).unwrap();
                        let placements = proc.tile_into(&src, &mut parent, &cfg).unwrap();
                        for p in &placements {
                            let source = Region::new(
                                p.origin.0 as usize,
                                p.origin.1 as usize,
                                p.crop_size.0 as usize,
                                p.crop_size.1 as usize,
                            );
                            let mut want = TensorDyn::image(
                                32,
                                32,
                                PixelFormat::Rgba,
                                DType::U8,
                                Some(TensorMemory::Mem),
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap();
                            CPUProcessor::new()
                                .convert(
                                    &src,
                                    &mut want,
                                    Some(
                                        ConvertOptions::new().crop(
                                            Crop::default()
                                                .with_source(Some(source))
                                                .with_fit(Fit::Stretch),
                                        ),
                                    ),
                                )
                                .unwrap();
                            let band = parent.view(Region::new(0, p.index * 32, 32, 32)).unwrap();
                            let got = {
                                let t = band.as_u8().unwrap();
                                let stride = t.effective_row_stride().unwrap_or(32 * 4);
                                let map = t.map().unwrap();
                                (0..32)
                                    .flat_map(|r| map.as_slice()[r * stride..][..32 * 4].to_vec())
                                    .collect::<Vec<u8>>()
                            };
                            let want = want.as_u8().unwrap().map().unwrap().as_slice().to_vec();
                            let mean = got
                                .iter()
                                .zip(&want)
                                .map(|(a, b)| a.abs_diff(*b) as f64)
                                .sum::<f64>()
                                / got.len() as f64;
                            if mean > 4.0 {
                                failures.push(format!(
                                    "round {round} tile {}: mean diff {mean:.2}",
                                    p.index
                                ));
                            }
                        }
                    }
                    failures
                })
            }));
            let _ = done_tx.send(result);
        });
        match done_rx.recv_timeout(TIMEOUT) {
            Ok(Ok(failures)) => assert!(failures.is_empty(), "{}", failures.join("\n")),
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => panic!("{} timed out after {TIMEOUT:?}", function!()),
        }
    }

    /// `tile_into` rejects a destination too small to hold all tile bands.
    #[test]
    fn tile_into_undersized_dst_errors() {
        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .expect("the CPU backend is always available");
        let (fw, fh) = (96usize, 64usize);
        let src = gradient_frame(fw, fh);
        let cfg = TilingConfig::new(32, 32).with_overlap(0.0); // 6 tiles
        let mut small = TensorDyn::image(
            32,
            2 * 32,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let r = proc.tile_into(&src, &mut small, &cfg);
        assert!(
            matches!(r, Err(Error::InvalidShape(_))),
            "expected InvalidShape, got {r:?}"
        );
    }

    /// `alloc_tile_batch` / `plan_tiles` reject an invalid config (zero tile).
    #[test]
    fn tiling_alloc_rejects_invalid_config() {
        let proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .expect("the CPU backend is always available");
        let bad = TilingConfig::new(0, 640);
        assert!(proc.plan_tiles(1920, 1080, &bad).is_err());
        assert!(proc
            .alloc_tile_batch(
                4,
                &bad,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .is_err());
    }

    #[test]
    fn plan_tiles_metadata_4k() {
        let proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .expect("the CPU backend is always available");
        let cfg = TilingConfig::new(640, 640).with_overlap(0.2);
        let plan = proc.plan_tiles(3840, 2160, &cfg).unwrap();
        assert_eq!(plan.len(), 32);
        assert!(plan.iter().all(|p| p.count == 32));
        assert!(plan.iter().all(|p| p.crop_size == (640.0, 640.0)));
        assert!(plan.iter().all(|p| p.content_region.is_none())); // stretch fit
        assert!(plan.iter().all(|p| p.frame_dims == (3840.0, 2160.0)));
        assert_eq!(plan[0].origin, (0.0, 0.0));
    }

    #[test]
    fn test_invalid_crop() {
        let src = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let dst = TensorDyn::image(
            100,
            100,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        // A source crop exceeding the source bounds is rejected.
        let crop = Crop::new().with_source(Some(Region::new(50, 50, 60, 60)));
        assert!(matches!(
            crop.check_crop_dyn(&src, &dst),
            Err(Error::CropInvalid(_))
        ));

        // A source crop within bounds is valid.
        let crop = Crop::new().with_source(Some(Region::new(0, 0, 10, 10)));
        assert!(crop.check_crop_dyn(&src, &dst).is_ok());

        // A letterbox is valid.
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

        let dst = TensorDyn::image(
            640,
            360,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        // With `format: None` the helper returns the source's native format, NV12 for a colour JPEG.
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
        #[cfg(target_os = "linux")]
        {
            let converter = ImageProcessor::with_config_env(
                ImageProcessorConfig::default(),
                &ProcessorEnv {
                    disable_g2d: true,
                    ..Default::default()
                },
            )?;
            assert!(converter.g2d.is_none());
        }

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        #[cfg(feature = "opengl")]
        {
            let converter = ImageProcessor::with_config_env(
                ImageProcessorConfig::default(),
                &ProcessorEnv {
                    disable_gl: true,
                    ..Default::default()
                },
            )?;
            assert!(converter.opengl.is_none());
        }

        let converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_cpu: true,
                ..Default::default()
            },
        )?;
        assert!(converter.cpu.is_none());

        // With every backend disabled, convert returns `NoConverter`.
        let mut converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_gl: true,
                disable_g2d: true,
                disable_cpu: true,
                ..Default::default()
            },
        )?;

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let dst = TensorDyn::image(
            640,
            360,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        let (result, _src, _dst) = convert_img(
            &mut converter,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        assert!(matches!(result, Err(Error::NoConverter)));
        Ok(())
    }

    #[test]
    fn test_unsupported_conversion() {
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let dst = TensorDyn::image(
            640,
            360,
            PixelFormat::Nv24,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
            Err(Error::NotSupported(e)) if e.starts_with("Conversion from NV12 to NV24")
        ));
    }

    #[test]
    fn test_load_grey() {
        // A single-component JPEG decodes to GREY and converts to RGBA.
        let grey_img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("grey.jpg"),
            Some(PixelFormat::Rgba),
            None,
        )
        .unwrap();
        assert_eq!(grey_img.width(), Some(1024));
        assert_eq!(grey_img.height(), Some(681));

        // `grey-rgb.jpg` is a 3-component JPEG with an odd height: it decodes to
        // NV12 and converts to RGBA at its true dimensions.
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
        let nv12 = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(nv12.height(), Some(720));
        assert_eq!(nv12.width(), Some(1280));
        assert_eq!(nv12.format().unwrap(), PixelFormat::Nv12);
        assert_eq!(nv12.format().unwrap().channels(), 1);
        assert!(nv12.format().is_some_and(
            |f| f.layout() == PixelLayout::Planar || f.layout() == PixelLayout::SemiPlanar
        ))
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_new_image_converter() {
        let dst_width = 640;
        let dst_height = 360;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mut converter = ImageProcessor::new().unwrap();
        let converter_dst = converter
            .create_image(
                dst_width,
                dst_height,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
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

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn test_create_image_dtype_i8() {
        let mut converter = ImageProcessor::new().unwrap();

        let dst = converter
            .create_image(
                320,
                240,
                PixelFormat::Rgb,
                DType::I8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(dst.dtype(), DType::I8);
        assert!(dst.width() == Some(320));
        assert!(dst.height() == Some(240));
        assert_eq!(dst.format(), Some(PixelFormat::Rgb));

        let dst_u8 = converter
            .create_image(
                320,
                240,
                PixelFormat::Rgb,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(dst_u8.dtype(), DType::U8);

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let mut dst_i8 = converter
            .create_image(
                320,
                240,
                PixelFormat::Rgb,
                DType::I8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        converter.convert(&src, &mut dst_i8, None).unwrap();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_create_image_nv12_dma_non_aligned_width() {
        // Checks the logical geometry and that a reported stride is at least the logical row.
        let converter = ImageProcessor::new().unwrap();

        let Some(img) = crate::test_support::dma_allocated(
            converter.create_image(
                100,
                64,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "create_image NV12 DMA non-aligned width",
        ) else {
            return;
        };
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

    #[test]
    #[ignore]
    fn test_crop_skip() {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mut converter = ImageProcessor::new().unwrap();
        let converter_dst = converter
            .create_image(
                1280,
                720,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
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

        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        assert!(PixelFormat::from_fourcc(u32::from_le_bytes(*b"TEST")).is_none());
    }

    // Whether the G2D library is available.
    #[cfg(target_os = "linux")]
    static G2D_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(target_os = "linux")]
    fn g2d_skip_reason() -> Option<&'static str> {
        crate::test_support::g2d_skip_reason(
            *G2D_AVAILABLE.get_or_init(|| G2DProcessor::new().is_ok()),
        )
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    // Whether OpenGL is available.
    fn is_opengl_available() -> bool {
        *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
    }

    /// Fails when `HAL_TEST_REQUIRE_GL=1` and the GL backend cannot initialise.
    #[test]
    #[cfg(feature = "opengl")]
    fn gl_backend_available_canary() {
        let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
        if !require_gl {
            crate::test_support::report_skip(&format!(
                "{} — HAL_TEST_REQUIRE_GL is not set to 1",
                function!()
            ));
            return;
        }
        #[cfg(target_os = "macos")]
        if std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
            crate::test_support::report_skip(&format!(
                "{} — ANGLE dlopen gate closed",
                function!()
            ));
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

        // `zidane_rotated_exif.jpg` carries EXIF orientation 6 (90° clockwise) over a 1280×720 frame.
        let file = edgefirst_bench::testdata::read("zidane_rotated_exif.jpg").to_vec();
        let info = peek_info(&file).unwrap();
        assert_eq!(info.rotation_degrees, 90);
        assert!(!info.flip_horizontal);

        let loaded = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        // Native, un-rotated dimensions.
        assert_eq!(loaded.width(), Some(1280));
        assert_eq!(loaded.height(), Some(720));

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let cpu_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let rotation = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
        let (dst_width, dst_height) = (cpu_src.height().unwrap(), cpu_src.width().unwrap());

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut cpu_converter = CPUProcessor::new();

        // Rotates `loaded` and `zidane` by the reported rotation.
        let loaded_rotated = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        // `zidane_rotated_exif_180.png` carries EXIF orientation 3 (180°).
        let file = edgefirst_bench::testdata::read("zidane_rotated_exif_180.png").to_vec();
        let info = peek_info(&file).unwrap();
        assert_eq!(info.rotation_degrees, 180);
        assert!(!info.flip_horizontal);

        let loaded = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        // Native, un-rotated dimensions.
        assert_eq!(loaded.height(), Some(720));
        assert_eq!(loaded.width(), Some(1280));

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let cpu_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let rotation = Rotation::from_degrees_clockwise(info.rotation_degrees as usize);
        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        // Rotates the decoded PNG by the reported angle.
        let loaded_rotated = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let (r0, _loaded, loaded_rotated) = convert_img(
            &mut cpu_converter,
            loaded,
            loaded_rotated,
            rotation,
            Flip::None,
            Crop::no_crop(),
        );
        r0.unwrap();

        compare_images(&loaded_rotated, &cpu_dst, 0.95, function!());
    }

    /// Builds an RGB JPEG of `(width, height)` with a deterministic pattern using `jpeg-encoder`.
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

    /// A 375×333 RGBA JPEG loaded through the pitch-padded DMA path letterboxes on the GL backend matching the CPU.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    fn test_convert_rgba_non_4_aligned_width_end_to_end() {
        use crate::test_support::is_dma_available;
        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_convert_rgba_non_4_aligned_width_end_to_end — DMA not available",
            );
            return;
        }
        // 375 * 4 = 1500 bytes per row, pitch-padded to 1536; width % 4 = 3.
        let jpeg = make_rgb_jpeg(375, 333);
        let src_gl = crate::load_image_test_helper(&jpeg, Some(PixelFormat::Rgba), None).unwrap();
        assert_eq!(src_gl.width(), Some(375));
        let stride = src_gl.row_stride().unwrap();
        assert_eq!(stride, 1536, "expected padded pitch 1536, got {stride}");

        let mut gl_proc = ImageProcessor::new().unwrap();
        let gl_dst = gl_proc
            .create_image(
                640,
                640,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
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

        // CPU reference from a fresh load.
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
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        compare_images(&gl_dst, &cpu_dst, 0.95, function!());
    }

    /// A JPEG loaded at a non-64-aligned RGBA pitch reports the padded pitch from `row_stride()` and `effective_row_stride()`.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_load_jpeg_rgba_non_aligned_pitch_padded_dma() {
        use crate::test_support::is_dma_available;
        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_load_jpeg_rgba_non_aligned_pitch_padded_dma — DMA not available",
            );
            return;
        }
        // Widths with a non-64-aligned natural RGBA pitch, all divisible by 4.
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

    /// The host has no G2D only when the loader found no file for the library; a library that fails to load is a G2D failure.
    #[test]
    #[cfg(target_os = "linux")]
    fn only_a_g2d_library_that_was_not_found_counts_as_absent() {
        let dlopen = |message: &std::ffi::CStr| {
            Error::G2D(g2d_sys::Error::LibraryError(libloading::Error::DlOpen {
                source: message.into(),
            }))
        };
        assert!(crate::g2d_is_absent(&dlopen(
            c"libg2d.so.2: cannot open shared object file: No such file or directory"
        )));
        for failure in [
            c"libGAL.so: cannot open shared object file: No such file or directory",
            c"/usr/lib/libg2d.so.2: invalid ELF header",
            c"/usr/lib/libg2d.so.2: wrong ELF class: ELFCLASS32",
            c"libg2d.so.2: cannot open shared object file: Permission denied",
        ] {
            assert!(!crate::g2d_is_absent(&dlopen(failure)), "{failure:?}");
        }
        assert!(!crate::g2d_is_absent(&Error::G2D(
            g2d_sys::Error::LibraryError(libloading::Error::DlOpenUnknown)
        )));
        assert!(!crate::g2d_is_absent(&Error::G2D(g2d_sys::Error::IoError(
            std::io::Error::from_raw_os_error(libc::ENODEV)
        ))));
        assert!(!crate::g2d_is_absent(&Error::G2D(
            g2d_sys::Error::InvalidFormat("XR24".to_string())
        )));
    }

    /// The host's loader names a file it could not find and reports a file it could not load differently.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_loader_tells_a_missing_library_from_a_broken_one() {
        let message = |library: &std::ffi::OsStr| match unsafe { libloading::Library::new(library) }
        {
            Err(libloading::Error::DlOpen { source }) => source.to_string(),
            other => panic!("expected dlopen to fail: {other:?}"),
        };

        let missing = "libedgefirst-no-such-library.so.0";
        let reported = message(missing.as_ref());
        assert!(crate::library_not_found(missing, &reported), "{reported}");
        assert!(
            !crate::library_not_found("libg2d.so.2", &reported),
            "a message naming another library reads as G2D's: {reported}"
        );

        let broken =
            std::env::temp_dir().join(format!("libedgefirst-broken-{}.so", std::process::id()));
        std::fs::write(&broken, b"not a shared object").unwrap();
        let reported = message(broken.as_os_str());
        std::fs::remove_file(&broken).unwrap();
        assert!(
            !crate::library_not_found(broken.to_str().unwrap(), &reported),
            "{reported}"
        );
    }

    /// `padded_dma_pitch_for` pads for an explicit DMA request, never for `Mem` or `Shm`, and for `None` only when DMA is available.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_padded_dma_pitch_for_respects_memory_choice() {
        use crate::test_support::is_dma_available;
        use edgefirst_tensor::TensorMemory;

        // 500×4 = 2000 bytes pads to 2048.
        let unaligned_w = 500;

        // `Mem` and `Shm`: no padding.
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

        // An explicit DMA request pads.
        assert_eq!(
            crate::padded_dma_pitch_for(
                PixelFormat::Rgba,
                unaligned_w,
                &Some(TensorMemory::DmaBuf),
            ),
            Some(2048),
            "explicit Dma must pad regardless of runtime DMA availability"
        );

        // `None` pads only when DMA is available.
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

    // Builds a greyscale PNG of `(width, height)` with a deterministic ramp.
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

    /// A greyscale PNG whose 612-byte row is not pitch-aligned loads through the pitch-padded DMA path.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_load_png_grey_misaligned_width_dma() {
        use crate::test_support::is_dma_available;
        if !is_dma_available() {
            crate::test_support::report_skip(
                "test_load_png_grey_misaligned_width_dma — DMA not available",
            );
            return;
        }
        let png = make_grey_png(612, 388);
        let loaded = crate::load_image_test_helper(&png, Some(PixelFormat::Grey), None).unwrap();
        assert_eq!(loaded.width(), Some(612));
        assert_eq!(loaded.height(), Some(388));
        assert_eq!(loaded.format(), Some(PixelFormat::Grey));

        // Reads rows at `row_stride()`.
        let map = loaded.as_typed::<u8>().unwrap().map().unwrap();
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

    /// A greyscale PNG loads into `Mem` as `Grey` at its natural pitch.
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
        let map = loaded.as_typed::<u8>().unwrap().map().unwrap();
        let bytes: &[u8] = &map;
        assert_eq!(bytes.len(), 612 * 100);
        for y in 0..100 {
            for x in 0..612 {
                assert_eq!(bytes[y * 612 + x], ((x + y) & 0xFF) as u8);
            }
        }
    }

    /// A greyscale PNG loads as `Rgb` with the luma in each channel.
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

        let map = loaded.as_typed::<u8>().unwrap().map().unwrap();
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_g2d_resize - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_g2d_resize - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(
            &file,
            Some(PixelFormat::Rgba),
            Some(TensorMemory::DmaBuf),
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_resize() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        let mut src = src;
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        for _ in 0..5 {
            let gl_dst = TensorDyn::image(
                dst_width,
                dst_height,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_10_threads() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_grey() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let img = crate::load_image_test_helper(
            &edgefirst_bench::testdata::read("grey.jpg"),
            Some(PixelFormat::Grey),
            None,
        )
        .unwrap();

        let gl_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::Grey,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let cpu_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::Grey,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

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

    /// A zero-copy `Grey` source converts to `Rgba`, `Rgb` and `PlanarRgb` with its luma in every colour channel, matching the CPU.
    #[test]
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_grey_zero_copy_source_to_rgb() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - zero-copy GPU buffer not available on this host",
                function!()
            ));
            return;
        }
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        let h = 40usize;
        for w in [64usize, 62] {
            let grey = |memory| -> Option<TensorDyn> {
                let t = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Grey,
                    DType::U8,
                    Some(memory),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .ok()?;
                let stride = t.effective_row_stride().unwrap_or(w);
                {
                    let mut map = t.as_u8().unwrap().map().unwrap();
                    let buf = map.as_mut_slice();
                    for r in 0..h {
                        for c in 0..w {
                            buf[r * stride + c] = ((r * 13 + c * 7) as u8).wrapping_add(3);
                        }
                    }
                }
                Some(t)
            };
            let Some(src_gpu) = grey(TensorMemory::DmaBuf) else {
                crate::test_support::report_skip(&format!(
                    "{} - no zero-copy Grey {w}x{h} buffer on this host",
                    function!()
                ));
                return;
            };
            let src_mem = grey(TensorMemory::Mem).unwrap();
            for fmt in [PixelFormat::Rgba, PixelFormat::Rgb, PixelFormat::PlanarRgb] {
                if fmt == PixelFormat::Rgb && (w * 3) % 4 != 0 {
                    continue;
                }
                let dst = || {
                    TensorDyn::image(
                        w,
                        h,
                        fmt,
                        DType::U8,
                        None,
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap()
                };
                let mut cpu_dst = dst();
                CPUProcessor::new()
                    .convert(&src_mem, &mut cpu_dst, None)
                    .unwrap();
                let mut gl_dst = dst();
                let before = gl.convert_stats().unwrap();
                gl.convert(&src_gpu, &mut gl_dst, None).unwrap();
                let after = gl.convert_stats().unwrap();
                let imports = after.src_imports - before.src_imports;
                let uploads = after.src_uploads - before.src_uploads;
                log::info!(
                    "{}: {w}x{h} Grey -> {fmt:?} imports={imports} uploads={uploads}",
                    function!()
                );
                // A zero-copy transfer imports the Grey source once.
                #[cfg(target_os = "linux")]
                if gl.transfer_backend().is_zero_copy() {
                    assert_eq!(
                        imports, 1,
                        "{w}x{h} Grey -> {fmt:?}: the DMA-BUF source was not imported"
                    );
                }
                assert_grey_rgb_rows_match(&gl_dst, &cpu_dst, fmt, w, h);
            }
        }
    }

    /// Compares every image row of a GL and a CPU `Rgba`/`Rgb`/`PlanarRgb` destination within ±4.
    #[cfg(feature = "opengl")]
    fn assert_grey_rgb_rows_match(
        gl_dst: &TensorDyn,
        cpu_dst: &TensorDyn,
        fmt: PixelFormat,
        w: usize,
        h: usize,
    ) {
        let (rows, row_bytes) = match fmt {
            PixelFormat::PlanarRgb => (3 * h, w),
            _ => (h, w * fmt.channels()),
        };
        let gl_t = gl_dst.as_u8().unwrap();
        let cpu_t = cpu_dst.as_u8().unwrap();
        let gl_stride = gl_t.effective_row_stride().unwrap_or(row_bytes);
        let cpu_stride = cpu_t.effective_row_stride().unwrap_or(row_bytes);
        let gl_map = gl_t.map().unwrap();
        let cpu_map = cpu_t.map().unwrap();
        for r in 0..rows {
            let g = &gl_map.as_slice()[r * gl_stride..][..row_bytes];
            let c = &cpu_map.as_slice()[r * cpu_stride..][..row_bytes];
            if let Some(i) = (0..row_bytes).find(|&i| g[i].abs_diff(c[i]) > 4) {
                panic!(
                    "{w}x{h} Grey -> {fmt:?}: row {r} byte {i} is {} on GL, {} on CPU",
                    g[i], c[i]
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_src_crop() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_g2d_src_crop - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_g2d_src_crop - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_g2d_dst_crop - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_g2d_dst_crop - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_g2d_all_rgba - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_g2d_all_rgba - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let src_dyn = src;

        let mut cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let crop = Crop::new().with_source(Some(Region::new(50, 120, 1024, 576)));

        for rot in [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            cpu_dst
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_mut_slice()
                .fill(114);
            g2d_dst
                .as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_mut_slice()
                .fill(114);
            for flip in [Flip::None, Flip::Horizontal, Flip::Vertical] {
                let mut cpu_dst_dyn = cpu_dst;
                cpu_converter
                    .convert(
                        &src_dyn,
                        &mut cpu_dst_dyn,
                        Some(crate::ConvertOptions::new().crop(crop)),
                    )
                    .unwrap();
                cpu_dst = {
                    let mut __t = cpu_dst_dyn.into_u8().unwrap();
                    __t.set_format(PixelFormat::Rgba).unwrap();
                    TensorDyn::from(__t)
                };

                let mut g2d_dst_dyn = g2d_dst;
                g2d_converter
                    .convert(
                        &src_dyn,
                        &mut g2d_dst_dyn,
                        Some(crate::ConvertOptions::new().crop(crop)),
                    )
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_src_crop() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let dst_width = 640;
        let dst_height = 360;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let crop = Crop::new().with_source(Some(Region::new(320, 180, 1280 - 320, 720 - 180)));

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let gl_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_dst_crop() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let gl_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    /// A deterministic RGBA8 `Mem` source: `R = x`, `G = y`, `B = x + y`,
    /// `A = 255`, each `& 255`, written at the tensor's own row stride.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn synthetic_rgba(w: usize, h: usize) -> TensorDyn {
        let t = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_synthetic_rgba(&t);
        t
    }

    /// Asserts every pixel of `actual` equals `expected`, compared through the
    /// tensors' own row compaction.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn assert_pixels_identical(actual: &TensorDyn, expected: &TensorDyn, name: &str) {
        assert_eq!(actual.shape(), expected.shape(), "{name}: shapes differ");
        let flat = |t: &TensorDyn| {
            let typed = t.as_typed::<u8>().expect("u8 image");
            let mut out = vec![0u8; t.shape().iter().product::<usize>()];
            typed.copy_to_flat(&mut out).expect("compact the rows");
            out
        };
        let (a, e) = (flat(actual), flat(expected));
        let mismatches = a.iter().zip(&e).filter(|(x, y)| x != y).count();
        assert_eq!(
            mismatches,
            0,
            "{name}: {mismatches} of {} bytes differ (first rows: {:?} vs {:?})",
            a.len(),
            &a[..16.min(a.len())],
            &e[..16.min(e.len())]
        );
    }

    /// Writes the `synthetic_rgba` pattern into any writable RGBA8 tensor,
    /// a texture included, at its own row stride.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn fill_synthetic_rgba(t: &TensorDyn) {
        let w = t.width().unwrap();
        let h = t.height().unwrap();
        let bound = t.as_typed::<u8>().unwrap();
        let stride = bound.effective_row_stride().unwrap_or(w * 4);
        let mut m = bound.map().unwrap();
        let px = m.as_mut_slice();
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                px[o] = x as u8;
                px[o + 1] = y as u8;
                px[o + 2] = (x + y) as u8;
                px[o + 3] = 255;
            }
        }
    }

    /// Element-wise comparison of two float images of the same format and
    /// dtype, read through `copy_to_flat`. F16 values are widened with
    /// `half::f16::to_f64` before the comparison.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn compare_float_images(a: &TensorDyn, b: &TensorDyn, tol: f64, what: &str) {
        assert_eq!(a.format(), b.format(), "{what}: pixel formats differ");
        assert_eq!(a.dtype(), b.dtype(), "{what}: dtypes differ");
        assert_eq!(a.shape(), b.shape(), "{what}: shapes differ");
        let values = |t: &TensorDyn| -> Vec<f64> {
            let elems: usize = t.shape().iter().product();
            match t.dtype() {
                DType::F32 => {
                    let typed = t.as_typed::<f32>().expect("F32 image");
                    let mut bytes = vec![0u8; elems * 4];
                    typed
                        .copy_to_flat(&mut bytes)
                        .expect("compact the padded rows");
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c) as f64)
                        .collect()
                }
                DType::F16 => {
                    let typed = t.as_typed::<half::f16>().expect("F16 image");
                    let mut bytes = vec![0u8; elems * 2];
                    typed
                        .copy_to_flat(&mut bytes)
                        .expect("compact the padded rows");
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| half::f16::from_le_bytes(*c).to_f64())
                        .collect()
                }
                other => panic!("{what}: {other:?} is not a float dtype"),
            }
        };
        let (av, bv) = (values(a), values(b));
        assert_eq!(av.len(), bv.len(), "{what}: element counts differ");
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "{what}: element {i} differs by {} (gpu {x}, cpu {y}, tol {tol})",
                (x - y).abs()
            );
        }
    }

    /// The float PBO readback lands each destination row at the destination's own pitch.
    #[test]
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn windows_padded_float_pbo_rows_land_at_the_declared_stride() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        const W: usize = 64;
        const H: usize = 32;
        const POOL_W: usize = 96;

        let mut gpu = ImageProcessor::new().unwrap();
        let support = gpu.supported_render_dtypes();
        let src = synthetic_rgba(W, H);

        // (format, dtype, supported, the pitches to declare)
        let cases: [(PixelFormat, DType, bool, &[usize]); 2] = [
            (
                PixelFormat::PlanarRgb,
                DType::F16,
                support.f16,
                // Tight plane row, the pool's pitch, and a 4-byte pad.
                &[W * 2, POOL_W * 2, W * 2 + 4],
            ),
            (
                PixelFormat::Rgb,
                DType::F32,
                support.f32,
                &[W * 3 * 4, POOL_W * 3 * 4],
            ),
        ];

        for (fmt, dtype, supported, strides) in cases {
            if !supported {
                crate::test_support::report_skip(&format!(
                    "{} - {fmt:?}/{dtype:?} not reported",
                    function!()
                ));
                continue;
            }
            let Some(gl) = gpu.opengl.as_ref() else {
                crate::test_support::report_skip(&format!("{} - no GL processor", function!()));
                return;
            };
            // The oracle: a tight PBO of the target geometry.
            let mut reference = gl
                .create_pbo_image_dtype(W, H, fmt, dtype)
                .unwrap_or_else(|e| panic!("{fmt:?}/{dtype:?}: tight reference PBO: {e:?}"));
            assert_eq!(
                reference.memory(),
                TensorMemory::Pbo,
                "{fmt:?}/{dtype:?}: the destination must be a PBO to reach this readback"
            );
            gpu.convert(&src, &mut reference, None)
                .unwrap_or_else(|e| panic!("{fmt:?}/{dtype:?}: tight convert failed: {e}"));

            for &stride in strides {
                let label = format!("{fmt:?}/{dtype:?} at a {stride} B pitch");
                // Allocates at the pool width, then narrows the geometry and declares the pitch.
                let mut padded = gpu
                    .opengl
                    .as_ref()
                    .expect("GL processor")
                    .create_pbo_image_dtype(POOL_W, H, fmt, dtype)
                    .unwrap_or_else(|e| panic!("{label}: pool PBO: {e:?}"));
                padded
                    .configure_image(W, H, fmt)
                    .unwrap_or_else(|e| panic!("{label}: narrow the pool tensor: {e}"));
                padded
                    .set_row_stride(stride)
                    .unwrap_or_else(|e| panic!("{label}: declare the pitch: {e}"));
                assert_eq!(padded.effective_row_stride(), Some(stride), "{label}");

                let before = gpu.convert_fallback_count();
                gpu.convert(&src, &mut padded, None)
                    .unwrap_or_else(|e| panic!("{label}: padded convert failed: {e}"));
                assert_eq!(
                    gpu.convert_fallback_count(),
                    before,
                    "{label}: fell back to the CPU backend, so the float PBO readback under test never ran"
                );

                let tol = if dtype == DType::F16 {
                    1.0 / 1024.0
                } else {
                    1e-6
                };
                compare_float_images(&padded, &reference, tol, &label);
            }
        }
    }

    /// Every float destination layout the GL backend reports renders zero-copy into a texture and matches the CPU converter.
    #[test]
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn windows_float_destinations_match_the_cpu_converter_for_every_layout() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut gpu = ImageProcessor::new().unwrap();
        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let support = gpu.supported_render_dtypes();
        eprintln!("{}: supported_render_dtypes = {support:?}", function!());
        let src = synthetic_rgba(640, 480);
        for (fmt, dtype, need) in [
            (PixelFormat::PlanarRgb, DType::F16, support.f16),
            (PixelFormat::PlanarRgb, DType::F32, support.f32),
            (PixelFormat::PlanarRgba, DType::F16, support.f16),
            (PixelFormat::PlanarRgba, DType::F32, support.f32),
            (PixelFormat::Rgb, DType::F16, support.f16),
            (PixelFormat::Rgb, DType::F32, support.f32),
            (PixelFormat::Rgba, DType::F16, support.f16),
            (PixelFormat::Rgba, DType::F32, support.f32),
        ] {
            if !need {
                crate::test_support::report_skip(&format!(
                    "{} - {fmt:?}/{dtype:?} not reported",
                    function!()
                ));
                continue;
            }
            let mut dst_gpu = gpu
                .create_image(
                    640,
                    480,
                    fmt,
                    dtype,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
            assert_eq!(
                dst_gpu.memory(),
                TensorMemory::DmaBuf,
                "{fmt:?}/{dtype:?} must be a texture"
            );
            let mut dst_cpu = TensorDyn::image(
                640,
                480,
                fmt,
                dtype,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let before = gpu.convert_fallback_count();
            gpu.convert(&src, &mut dst_gpu, None).unwrap();
            cpu.convert(&src, &mut dst_cpu, None).unwrap();
            assert_eq!(
                gpu.convert_fallback_count(),
                before,
                "{fmt:?}/{dtype:?} fell back to CPU"
            );
            let tol = if dtype == DType::F16 {
                1.0 / 1024.0
            } else {
                1e-6
            };
            compare_float_images(&dst_gpu, &dst_cpu, tol, &format!("{fmt:?}/{dtype:?}"));
        }
    }

    /// A native two-plane NV12 texture source converts on the GL backend with no CPU fallback and matches the CPU converter.
    #[test]
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn windows_native_nv12_sources_convert_through_their_planes() {
        if !windows_gl_textures_or_skip(function!()) {
            return;
        }
        let mut gpu = ImageProcessor::new().unwrap();
        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        for (w, h) in [(640usize, 480usize), (1920, 1080), (64, 62)] {
            let src = TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            if !texture_or_skip(&src, function!()) {
                return;
            }
            assert!(
                src.as_typed::<u8>()
                    .unwrap()
                    .d3d11_layout()
                    .unwrap()
                    .is_native_nv12(),
                "{w}x{h} is a native NV12 texture"
            );
            let host = TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            fill_nv12_gradient(&src, w, h);
            fill_nv12_gradient(&host, w, h);
            let crop = Region::new((w / 8) & !1, (h / 8) & !1, w / 2, h / 2);
            // The 1:1 and crop draws are compared with the CPU converter, the
            // bilinear resize with the same convert of a host copy.
            for (label, (dw, dh), region, cpu_reference) in [
                ("1:1", (w, h), None, true),
                ("crop", (w / 2, h / 2), Some(crop), true),
                ("resize", (w / 2 + 6, h / 3 + 4), None, false),
            ] {
                let rgba = |memory| {
                    TensorDyn::image(
                        dw,
                        dh,
                        PixelFormat::Rgba,
                        DType::U8,
                        Some(memory),
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap()
                };
                let options = || ConvertOptions::default().source(region);
                let mut dst_gpu = rgba(TensorMemory::DmaBuf);
                let before = gpu.convert_fallback_count();
                gpu.convert(&src, &mut dst_gpu, Some(options())).unwrap();
                assert_eq!(
                    gpu.convert_fallback_count(),
                    before,
                    "{w}x{h} {label} fell back to the CPU"
                );
                let mut reference = rgba(TensorMemory::Mem);
                if cpu_reference {
                    cpu.convert(&host, &mut reference, Some(options())).unwrap();
                } else {
                    gpu.convert(&host, &mut reference, Some(options())).unwrap();
                }
                let (max, mean) = rgba_difference(&dst_gpu, &reference);
                assert!(
                    max <= 1 && mean <= 0.05,
                    "{w}x{h} {label}: max {max}, mean {mean:.3} against the {} reference",
                    if cpu_reference {
                        "CPU"
                    } else {
                        "combined-plane GL"
                    }
                );
            }
        }
    }

    /// A native NV12 pool texture reconfigured for a smaller frame samples inside its logical image.
    #[test]
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn windows_native_nv12_pool_reconfigured_for_a_smaller_frame_converts() {
        if !windows_gl_textures_or_skip(function!()) {
            return;
        }
        let mut gpu = ImageProcessor::new().unwrap();
        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let (w, h) = (640usize, 480usize);
        let mut src = TensorDyn::image(
            w,
            h,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let (iw, ih) = (w / 2, h / 2);
        src.configure_image(iw, ih, PixelFormat::Nv12).unwrap();
        let host = TensorDyn::image(
            iw,
            ih,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        fill_nv12_gradient(&src, iw, ih);
        fill_nv12_gradient(&host, iw, ih);
        let rgba = |memory| {
            TensorDyn::image(
                iw,
                ih,
                PixelFormat::Rgba,
                DType::U8,
                Some(memory),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let (mut dst_gpu, mut dst_cpu) = (rgba(TensorMemory::DmaBuf), rgba(TensorMemory::Mem));
        let before = gpu.convert_fallback_count();
        gpu.convert(&src, &mut dst_gpu, None).unwrap();
        assert_eq!(
            gpu.convert_fallback_count(),
            before,
            "narrowed source fell back"
        );
        cpu.convert(&host, &mut dst_cpu, None).unwrap();
        let (max, mean) = rgba_difference(&dst_gpu, &dst_cpu);
        assert!(
            max <= 1 && mean <= 0.05,
            "narrowed: max {max}, mean {mean:.3}"
        );
    }

    /// Fills a `w`x`h` NV12 image with a gradient: luma rows, then
    /// interleaved chroma rows, at the tensor's row stride.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn fill_nv12_gradient(t: &TensorDyn, w: usize, h: usize) {
        let stride = t.effective_row_stride().unwrap();
        let mut m = t.map_bytes(edgefirst_tensor::CpuAccess::ReadWrite).unwrap();
        let buf = m.as_mut_slice();
        for y in 0..h {
            for x in 0..w {
                buf[y * stride + x] = (16 + (x * 7 + y * 3) % 220) as u8;
            }
        }
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                let at = (h + y) * stride + 2 * x;
                buf[at] = (64 + (x * 5) % 128) as u8;
                buf[at + 1] = (64 + (y * 9) % 128) as u8;
            }
        }
    }

    /// The largest difference of any channel of two RGBA images, and the mean
    /// over all of them.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn rgba_difference(a: &TensorDyn, b: &TensorDyn) -> (u8, f64) {
        let (w, h) = (a.width().unwrap(), a.height().unwrap());
        let (sa, sb) = (
            a.effective_row_stride().unwrap(),
            b.effective_row_stride().unwrap(),
        );
        let ma = a.map_bytes(edgefirst_tensor::CpuAccess::Read).unwrap();
        let mb = b.map_bytes(edgefirst_tensor::CpuAccess::Read).unwrap();
        let (mut max, mut sum) = (0u8, 0u64);
        for y in 0..h {
            for x in 0..w * 4 {
                let d = ma.as_slice()[y * sa + x].abs_diff(mb.as_slice()[y * sb + x]);
                max = max.max(d);
                sum += u64::from(d);
            }
        }
        (max, sum as f64 / (w * h * 4) as f64)
    }

    /// The skip checks every Windows native NV12 GL test makes. Returns
    /// whether to run.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn windows_gl_textures_or_skip(test: &str) -> bool {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{test} - OpenGL not available"));
            return false;
        }
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!("{test} - no zero-copy GPU buffers"));
            return false;
        }
        true
    }

    // The Win32 wait call the fence tests use.
    #[cfg(target_os = "windows")]
    #[link(name = "kernel32")]
    extern "system" {
        fn WaitForSingleObject(handle: *mut std::ffi::c_void, milliseconds: u32) -> u32;
    }

    /// A batch of deferred converts records one distinct fence value per tile and submits once, at `flush`.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_deferred_batch_records_one_value_per_tile_and_submits_once() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - ImageProcessor init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        let (w, h, tiles) = (640usize, 480usize, 4usize);
        let src = synthetic_rgba(w, h);
        let mut dsts: Vec<TensorDyn> = (0..tiles)
            .map(|_| {
                p.create_image(w, h, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
                    .unwrap()
            })
            .collect();
        if dsts[0].memory() != TensorMemory::DmaBuf {
            crate::test_support::report_skip(&format!(
                "{} - destinations are not textures",
                function!()
            ));
            return;
        }
        for dst in &mut dsts {
            p.convert_deferred(&src, dst, None).unwrap();
        }
        let values: Vec<u64> = dsts
            .iter()
            .map(|d| {
                d.gpu_completion()
                    .unwrap()
                    .expect("each deferred convert records its own completion")
                    .value
            })
            .collect();
        for pair in values.windows(2) {
            assert!(
                pair[1] > pair[0],
                "the batch allocated {values:?}, which is not one increasing value per tile"
            );
        }
        p.flush().unwrap();
        let dev = edgefirst_tensor::d3d11::device().unwrap();
        let last = *values.last().unwrap();
        let ev = dev.event_for(last).unwrap();
        // SAFETY: `ev` owns a live Win32 event handle for the duration of
        // this call.
        assert_eq!(
            unsafe {
                WaitForSingleObject(
                    std::os::windows::io::AsRawHandle::as_raw_handle(&ev) as _,
                    5000,
                )
            },
            0,
            "the batch's newest value {last} never arrived after flush"
        );

        let mut solo = p
            .create_image(w, h, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
            .unwrap();
        p.convert(&src, &mut solo, None).unwrap();
        // Reads through the tensor's own compaction.
        let flat = |img: &TensorDyn| -> Vec<u8> {
            let t = img.as_typed::<u8>().expect("u8 image");
            let mut out = vec![0u8; t.shape().iter().product::<usize>()];
            t.copy_to_flat(&mut out).expect("compact the padded rows");
            out
        };
        let want = flat(&solo);
        for (i, dst) in dsts.iter().enumerate() {
            assert_eq!(
                flat(dst),
                want,
                "tile {i} does not match a standalone convert"
            );
        }
    }

    /// A processor dropped with a batch still open submits what that batch recorded.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_dropping_a_processor_submits_an_open_batch() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let src = synthetic_rgba(320, 240);
        let recorded = {
            let mut p = match ImageProcessor::new() {
                Ok(p) => p,
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "{} - init failed ({e:?})",
                        function!()
                    ));
                    return;
                }
            };
            let mut dst = p
                .create_image(
                    320,
                    240,
                    PixelFormat::Rgba,
                    DType::U8,
                    None,
                    CpuAccess::Read,
                )
                .unwrap();
            if dst.memory() != TensorMemory::DmaBuf {
                crate::test_support::report_skip(&format!(
                    "{} - destination is not a texture",
                    function!()
                ));
                return;
            }
            p.convert_deferred(&src, &mut dst, None).unwrap();
            dst.gpu_completion()
                .unwrap()
                .expect("the deferred convert recorded a completion")
                .value
        };

        let dev = edgefirst_tensor::d3d11::device().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while dev.completed_value() < recorded {
            assert!(
                std::time::Instant::now() < deadline,
                "the dropped processor never submitted its batch: fence at {}, recorded {recorded}",
                dev.completed_value()
            );
            std::thread::yield_now();
        }
    }

    /// `create_image(.., None, ..)` returns a D3D11 texture and `convert_with_fence` returns an event set when the GPU work completes.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_create_image_prefers_textures_and_convert_with_fence_returns_a_set_event() {
        use edgefirst_tensor::CpuAccess;
        use std::os::windows::io::AsRawHandle;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = ImageProcessor::new().unwrap();
        let src = synthetic_rgba(320, 240);
        let mut dst = p
            .create_image(
                320,
                240,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::Read,
            )
            .unwrap();
        if dst.memory() != TensorMemory::DmaBuf
            && !std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
        {
            crate::test_support::report_skip(&format!(
                "{} - create_image gave {:?}, not a texture (no GL backend)",
                function!(),
                dst.memory()
            ));
            return;
        }
        assert_eq!(
            dst.memory(),
            TensorMemory::DmaBuf,
            "create_image(None) must prefer a texture on Windows"
        );
        let before = dst.gpu_completion().unwrap().map(|c| c.value);
        let fence = p
            .convert_with_fence(&src, &mut dst, None)
            .unwrap()
            .expect("Windows exports a fence");
        let recorded = dst
            .gpu_completion()
            .unwrap()
            .expect("the convert recorded a completion")
            .value;
        assert!(
            before.is_none_or(|b| recorded > b),
            "the fenced convert recorded a newer value than {before:?}"
        );
        // One signal for the whole call.
        assert_eq!(
            edgefirst_tensor::d3d11::device().unwrap().last_signalled(),
            recorded,
            "a fenced convert must signal exactly once"
        );
        // SAFETY: `fence` owns a live Win32 event handle that outlives this call.
        assert_eq!(
            unsafe { WaitForSingleObject(fence.as_raw_handle() as _, 5000) },
            0,
            "fence was not set within 5 seconds"
        );
        assert!(
            edgefirst_tensor::d3d11::device().unwrap().completed_value() >= recorded,
            "the event fired, so the recorded value {recorded} is complete"
        );
        assert_eq!(
            dst.gpu_completion().unwrap().map(|c| c.value),
            Some(recorded),
            "the destination still reports the value the event waited on"
        );
        let mut expected = TensorDyn::image(
            320,
            240,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        cpu.convert(&src, &mut expected, None).unwrap();
        compare_images(&dst, &expected, 0.0, "fenced convert");
    }

    /// Whether `t` is a texture. Otherwise reports a skip, or panics when
    /// `HAL_TEST_REQUIRE_GL=1`.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    fn texture_or_skip(t: &TensorDyn, test: &str) -> bool {
        if t.memory() == TensorMemory::DmaBuf {
            return true;
        }
        if std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1") {
            panic!(
                "{test}: create_image(None) must prefer a texture on Windows, got {:?}",
                t.memory()
            );
        }
        crate::test_support::report_skip(&format!(
            "{test} - create_image gave {:?}, not a texture (no GL backend)",
            t.memory()
        ));
        false
    }

    /// A draw into a texture destination records a completion on it.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_draw_decoded_masks_into_a_texture_records_a_completion() {
        use edgefirst_tensor::{CpuAccess, Segmentation};
        use ndarray::Array3;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = ImageProcessor::new().unwrap();
        let mut dst = p
            .create_image(64, 64, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
            .unwrap();
        if !texture_or_skip(&dst, function!()) {
            return;
        }
        assert!(
            dst.gpu_completion().unwrap().is_none(),
            "nothing has written this texture yet"
        );

        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.99,
            label: 0,
        };
        let seg_arr = Array3::from_shape_fn((4, 4, 1), |_| 255u8);
        let seg = Segmentation {
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(seg_arr.view())
                .unwrap()
                .into(),
            xmin: 0.25,
            ymin: 0.25,
            xmax: 0.75,
            ymax: 0.75,
        };
        p.draw_decoded_masks(&mut dst, &[detect], &[seg], MaskOverlay::default())
            .expect("draw into a read-only texture");

        let completion = dst
            .gpu_completion()
            .unwrap()
            .expect("the draw recorded no completion on dst");
        assert!(completion.value > 0, "completion value must be non-zero");

        // The corner is cleared and the box centre carries the mask colour.
        let mut flat = vec![0u8; 64 * 64 * 4];
        dst.as_typed::<u8>()
            .unwrap()
            .copy_to_flat(&mut flat)
            .expect("read the drawn texture");
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = (y * 64 + x) * 4;
            [flat[o], flat[o + 1], flat[o + 2], flat[o + 3]]
        };
        assert_eq!(px(2, 2), [0, 0, 0, 0], "corner (2,2) was not cleared");
        assert_ne!(px(32, 32), [0, 0, 0, 0], "centre (32,32) was not coloured");
    }

    /// The full-GPU proto draw records a completion on a texture destination.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_draw_proto_masks_into_a_texture_records_a_completion() {
        use edgefirst_tensor::{CpuAccess, Tensor};

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = match with_force_backend(Some("opengl")) {
            Ok(p) => p,
            Err(e) if !std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1") => {
                crate::test_support::report_skip(&format!(
                    "{} - no OpenGL backend: {e}",
                    function!()
                ));
                return;
            }
            Err(e) => panic!("forced OpenGL backend unavailable: {e}"),
        };
        let mut dst = p
            .create_image(64, 64, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
            .unwrap();
        if !texture_or_skip(&dst, function!()) {
            return;
        }

        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.99,
            label: 0,
        };
        // One prototype channel of ones and a positive coefficient: every
        // pixel of the box is foreground.
        let proto = ProtoData {
            mask_coefficients: TensorDyn::F32(
                Tensor::<f32>::from_slice(&[1.0_f32], &[1, 1]).unwrap(),
            ),
            protos: TensorDyn::F32(
                Tensor::<f32>::from_slice(&vec![1.0_f32; 8 * 8], &[8, 8, 1]).unwrap(),
            ),
            layout: ProtoLayout::Nhwc,
        };
        p.draw_proto_masks(&mut dst, &[detect], &proto, MaskOverlay::default())
            .expect("forced-GL proto draw into a read-only texture");

        let completion = dst
            .gpu_completion()
            .unwrap()
            .expect("the proto draw recorded no completion on dst");
        assert!(completion.value > 0, "completion value must be non-zero");
    }

    /// An empty draw with a host-memory background onto a texture destination leaves the background in it, rows in order.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_draw_decoded_masks_composites_a_host_background_onto_a_texture() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = ImageProcessor::new().unwrap();
        let bg = synthetic_rgba(64, 48);
        let mut dst = p
            .create_image(64, 48, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
            .unwrap();
        if !texture_or_skip(&dst, function!()) {
            return;
        }
        p.draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::new().with_background(&bg))
            .expect("empty draw with a host background onto a texture");
        assert_pixels_identical(&dst, &bg, "host background onto a texture");
        assert!(dst.gpu_completion().unwrap().is_some());
    }

    /// An empty draw with a texture background onto a texture destination leaves the background in it.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_draw_decoded_masks_composites_a_texture_background_onto_a_texture() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut p = ImageProcessor::new().unwrap();
        let expected = synthetic_rgba(64, 48);
        let bg = p
            .create_image(
                64,
                48,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        if !texture_or_skip(&bg, function!()) {
            return;
        }
        fill_synthetic_rgba(&bg);
        let mut dst = p
            .create_image(64, 48, PixelFormat::Rgba, DType::U8, None, CpuAccess::Read)
            .unwrap();
        assert_eq!(dst.memory(), TensorMemory::DmaBuf);
        p.draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::new().with_background(&bg))
            .expect("empty draw with a texture background onto a texture");
        assert_pixels_identical(&dst, &expected, "texture background onto a texture");
        assert!(dst.gpu_completion().unwrap().is_some());
    }

    /// A draw into a `view()` of a texture fills that band and leaves its sibling alone.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_draw_decoded_masks_into_a_texture_view_fills_only_its_band() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let (w, h) = (32usize, 64usize);
        let mut p = ImageProcessor::new().unwrap();
        let parent = p
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        if !texture_or_skip(&parent, function!()) {
            return;
        }
        fill_synthetic_rgba(&parent);

        // The lower band only.
        let mut band = parent
            .view(edgefirst_tensor::Region::new(0, h / 2, w, h / 2))
            .unwrap();
        p.draw_decoded_masks(&mut band, &[], &[], MaskOverlay::default())
            .expect("empty draw into a texture view");
        drop(band);

        let mut flat = vec![0u8; w * h * 4];
        parent
            .as_typed::<u8>()
            .unwrap()
            .copy_to_flat(&mut flat)
            .expect("read the parent texture");
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let got = [flat[o], flat[o + 1], flat[o + 2], flat[o + 3]];
                let want = if y < h / 2 {
                    [x as u8, y as u8, (x + y) as u8, 255]
                } else {
                    [0, 0, 0, 0]
                };
                assert_eq!(got, want, "pixel ({x},{y})");
            }
        }

        // With GL forced, the draw into the view is refused.
        let mut forced = with_force_backend(Some("opengl")).unwrap();
        let mut band = parent
            .view(edgefirst_tensor::Region::new(0, h / 2, w, h / 2))
            .unwrap();
        let err = forced
            .draw_decoded_masks(&mut band, &[], &[], MaskOverlay::default())
            .expect_err("forced GL must decline a view destination");
        assert!(
            matches!(err, Error::NotSupported(_)),
            "expected NotSupported, got {err:?}"
        );
    }

    /// A draw GL declines renders the palette given to `set_class_colors`.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn windows_class_colors_reach_the_cpu_backend_behind_a_declined_gl_draw() {
        use edgefirst_tensor::{CpuAccess, Segmentation};
        use ndarray::Array3;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        const CUSTOM: [u8; 4] = [7, 200, 13, 255];
        assert_ne!(
            CUSTOM[..3],
            DEFAULT_COLORS_U8[0][..3],
            "the custom colour must differ from the default palette's, or \
             this test cannot tell them apart"
        );

        let (w, h) = (32usize, 64usize);
        let mut p = ImageProcessor::new().unwrap();
        let parent = p
            .create_image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        if parent.memory() != TensorMemory::DmaBuf
            && !std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
        {
            crate::test_support::report_skip(&format!(
                "{} - create_image gave {:?}, not a texture (no GL backend)",
                function!(),
                parent.memory()
            ));
            return;
        }
        p.set_class_colors(&[CUSTOM]).unwrap();

        let detect = DetectBox {
            bbox: [0.25, 0.25, 0.75, 0.75].into(),
            score: 0.99,
            label: 0,
        };
        let seg_arr = Array3::from_shape_fn((4, 4, 1), |_| 255u8);
        let seg = Segmentation {
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(seg_arr.view())
                .unwrap()
                .into(),
            xmin: 0.25,
            ymin: 0.25,
            xmax: 0.75,
            ymax: 0.75,
        };
        let mut band = parent
            .view(edgefirst_tensor::Region::new(0, h / 2, w, h / 2))
            .unwrap();
        p.draw_decoded_masks(&mut band, &[detect], &[seg], MaskOverlay::default())
            .expect("draw into a texture view");
        drop(band);

        let mut flat = vec![0u8; w * h * 4];
        parent
            .as_typed::<u8>()
            .unwrap()
            .copy_to_flat(&mut flat)
            .expect("read the parent texture");
        // The band's centre, inside the box.
        let o = ((h / 2 + h / 4) * w + w / 2) * 4;
        assert_eq!(
            &flat[o..o + 3],
            &CUSTOM[..3],
            "the CPU backend drew the default palette, so it never received \
             set_class_colors"
        );
    }

    /// Two `ImageProcessor`s converting into their own texture destinations in lockstep stay independent over 200 iterations.
    #[cfg(all(target_os = "windows", feature = "opengl"))]
    #[test]
    fn two_processors_alternate_on_texture_destinations_without_state_bleed() {
        use edgefirst_tensor::CpuAccess;

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - no zero-copy GPU buffers",
                function!()
            ));
            return;
        }
        let mut a = ImageProcessor::new().unwrap();
        let mut b = ImageProcessor::new().unwrap();
        let src_a = synthetic_rgba(512, 512);
        let src_b = synthetic_rgba(256, 256);
        let mut dst_a = a
            .create_image(
                512,
                512,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::Read,
            )
            .unwrap();
        let mut dst_b = b
            .create_image(
                256,
                256,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::Read,
            )
            .unwrap();
        let mut cpu = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        let mut ref_a = TensorDyn::image(
            512,
            512,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut ref_b = TensorDyn::image(
            256,
            256,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu.convert(&src_a, &mut ref_a, None).unwrap();
        cpu.convert(&src_b, &mut ref_b, None).unwrap();
        for _ in 0..200 {
            a.convert(&src_a, &mut dst_a, None).unwrap();
            b.convert(&src_b, &mut dst_b, None).unwrap();
            compare_images(&dst_a, &ref_a, 0.0, "processor A");
            compare_images(&dst_b, &ref_b, 0.0, "processor B");
        }
    }

    #[test]
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_all_rgba() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let dst_width = 640;
        let dst_height = 640;
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();

        let mut cpu_converter = CPUProcessor::new();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        // Every backing this host can allocate.
        let mut mem = vec![None, Some(TensorMemory::Mem)];
        if edgefirst_tensor::is_shm_available() {
            mem.push(Some(TensorMemory::Shm));
        }
        if crate::test_support::is_gpu_buffer_available() {
            mem.push(Some(TensorMemory::DmaBuf));
        } else {
            crate::test_support::report_skip(&format!(
                "{} - DmaBuf backing not exercised: no zero-copy GPU buffers",
                function!()
            ));
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
                    let cpu_dst = TensorDyn::image(
                        dst_width,
                        dst_height,
                        PixelFormat::Rgba,
                        DType::U8,
                        m,
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap();
                    let gl_dst = TensorDyn::image(
                        dst_width,
                        dst_height,
                        PixelFormat::Rgba,
                        DType::U8,
                        m,
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    )
                    .unwrap();
                    cpu_dst
                        .as_typed::<u8>()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_mut_slice()
                        .fill(114);
                    gl_dst
                        .as_typed::<u8>()
                        .unwrap()
                        .map()
                        .unwrap()
                        .as_mut_slice()
                        .fill(114);

                    let mut cpu_dst_dyn = cpu_dst;
                    cpu_converter
                        .convert(
                            &src_dyn,
                            &mut cpu_dst_dyn,
                            Some(crate::ConvertOptions::new().crop(crop)),
                        )
                        .unwrap();
                    let cpu_dst = {
                        let mut __t = cpu_dst_dyn.into_u8().unwrap();
                        __t.set_format(PixelFormat::Rgba).unwrap();
                        TensorDyn::from(__t)
                    };

                    let mut gl_dst_dyn = gl_dst;
                    gl_converter
                        .convert(
                            &src_dyn,
                            &mut gl_dst_dyn,
                            Some(crate::ConvertOptions::new().crop(crop)),
                        )
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
        // Rotates the image four times and compares it with the original.
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

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_opengl_rotate() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        let size = (1280, 720);
        // Every backing this host can allocate.
        let mut mem = vec![None, Some(TensorMemory::Mem)];
        if edgefirst_tensor::is_shm_available() {
            mem.push(Some(TensorMemory::Shm));
        }
        if crate::test_support::is_gpu_buffer_available() {
            mem.push(Some(TensorMemory::DmaBuf));
        } else {
            crate::test_support::report_skip(&format!(
                "{} - DmaBuf backing not exercised: no zero-copy GPU buffers",
                function!()
            ));
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

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
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

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
                edgefirst_tensor::CpuAccess::ReadWrite,
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_g2d_rotate - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_g2d_rotate - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
        let src = crate::load_image_test_helper(
            &file,
            Some(PixelFormat::Rgba),
            Some(TensorMemory::DmaBuf),
        )
        .unwrap();

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let dst_through_yuyv = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let dst_direct = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    #[ignore = "opengl doesn't support rendering to PixelFormat::Yuyv texture"]
    fn test_rgba_to_yuyv_resize_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - zero-copy GPU buffer (DMA-BUF / IOSurface) not available on this host",
                function!()
            ));
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
            dst.as_typed::<u8>().unwrap().map().unwrap().as_slice(),
        )
        .unwrap();
        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_rgba_to_yuyv_resize_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_rgba_to_yuyv_resize_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.rgba"),
        )
        .unwrap();

        let (dst_width, dst_height) = (1280, 720);

        let cpu_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut g2d_converter = G2DProcessor::new().unwrap();
        let crop = Crop::new();

        g2d_dst
            .as_typed::<u8>()
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
            .as_typed::<u8>()
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
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Yuyv,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgb_cpu() {
        let file = edgefirst_bench::testdata::read("camera720p.yuyv").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Yuyv,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
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

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_g2d() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_yuyv_to_rgba_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_yuyv_to_rgba_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - zero-copy GPU buffer (DMA-BUF / IOSurface) not available on this host",
                function!()
            ));
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        if let Err(crate::Error::NotSupported(msg)) = &result {
            crate::test_support::report_skip(&format!("{} — {msg}", function!()));
            return;
        }
        result.unwrap();

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    /// The GL backend renders a GREY R8 IOSurface source to RGBA with R=G=B equal to the luma.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_grey_r8_iosurface_to_rgba_opengl_macos() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };

        let (w, h) = (16usize, 16usize);
        let src = TensorDyn::image(
            w,
            h,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("GREY IOSurface (R8/L008) should allocate — proves the FourCC mapping");
        // Luma ramp: value = (x * 13 + y * 7) & 0xff.
        {
            let su8 = src.as_typed::<u8>().unwrap();
            let stride = src
                .as_typed::<u8>()
                .unwrap()
                .effective_row_stride()
                .unwrap();
            let mut m = su8.map().unwrap();
            let buf = m.as_mut_slice();
            for y in 0..h {
                for x in 0..w {
                    buf[y * stride + x] = ((x * 13 + y * 7) & 0xff) as u8;
                }
            }
        }

        let dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let (result, src_back, dst) = convert_img(
            &mut proc,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.expect("GREY(R8 IOSurface) → RGBA must convert on ANGLE (R8 binding works)");

        let src_stride = src_back
            .as_typed::<u8>()
            .unwrap()
            .effective_row_stride()
            .unwrap();
        let src_map = src_back.as_typed::<u8>().unwrap().map().unwrap();
        let sbytes = src_map.as_slice();
        let dst_stride = dst
            .as_typed::<u8>()
            .unwrap()
            .effective_row_stride()
            .unwrap();
        let dst_map = dst.as_typed::<u8>().unwrap().map().unwrap();
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

    /// A neutral-grey NV12 IOSurface converts on the GPU to `PlanarRgb` F16 planes near 0.5.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_two_pass_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("{} — init failed ({e:?})", function!()));
                return;
            }
        };
        let (w, h) = (64usize, 64usize);
        let Some(src) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                w,
                h,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "NV12 IOSurface",
        ) else {
            return;
        };
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128); // Y=U=V=128

        let Some(dst) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                w,
                h,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "F16 PlanarRgb IOSurface",
        ) else {
            return;
        };
        let mut dst = dst;
        if let Err(e) = ImageProcessorTrait::convert(&mut gpu, &src, &mut dst, None) {
            crate::test_support::report_skip(&format!(
                "{} — NV12→PlanarRgb F16 not available ({e:?})",
                function!()
            ));
            return;
        }
        let dt = dst.as_typed::<half::f16>().expect("dst is F16 PlanarRgb");
        let map = dt.map().unwrap();
        let vals = map.as_slice();
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

    /// An NV12 frame in a larger reused R8 pool converts with a letterbox crop into a `PlanarRgb` F16 destination.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_two_pass_pool_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("{} — init failed ({e:?})", function!()));
                return;
            }
        };
        // Frame 96×64 in a 256×768 R8 pool.
        let (fw, fh) = (96usize, 64usize);
        let (pool_w, pool_h) = (256usize, 768usize);
        let (model_w, model_h) = (128usize, 128usize);

        let Some(mut src) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                pool_w,
                pool_h,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "R8 pool",
        ) else {
            return;
        };
        src.configure_image(fw, fh, PixelFormat::Nv12)
            .unwrap_or_else(|e| panic!("configure_image NV12 on pool: {e}"));
        let stride = src
            .as_typed::<u8>()
            .unwrap()
            .effective_row_stride()
            .unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128); // neutral grey

        let Some(mut dst) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                model_w,
                model_h,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "F16 PlanarRgb dst",
        ) else {
            return;
        };

        let _ = model_w;
        let crop = Crop::new()
            .with_source(Some(Region::new(0, 0, fw, fh)))
            .with_fit(Fit::Letterbox {
                pad: [0, 0, 0, 255],
            });
        if let Err(e) = ImageProcessorTrait::convert(
            &mut gpu,
            &src,
            &mut dst,
            Some(ConvertOptions::new().crop(crop)),
        ) {
            crate::test_support::report_skip(&format!(
                "{} — NV12→PlanarRgb F16 unavailable ({e:?})",
                function!()
            ));
            return;
        }
        let _ = stride;
        let dt = dst.as_typed::<half::f16>().expect("dst F16");
        let map = dt.map().unwrap();
        let any_half = map.as_slice().iter().any(|&v| {
            let f = f32::from(v);
            (0.40..=0.60).contains(&f)
        });
        assert!(any_half, "expected ~0.5 grey samples in the letterbox band");
    }

    /// A processor created on one thread converts on another within a 20 s watchdog.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_cross_thread_opengl_macos() {
        use std::sync::mpsc;
        let mut proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("{} — init failed ({e:?})", function!()));
                return;
            }
        };
        let (fw, fh) = (96usize, 64usize);
        let Some(mut src) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                256,
                768,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "NV12 R8 pool",
        ) else {
            return;
        };
        src.configure_image(fw, fh, PixelFormat::Nv12).unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        let Some(mut dst) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                128,
                128,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "F16 PlanarRgb dst",
        ) else {
            return;
        };
        let crop = Crop::new().with_source(Some(Region::new(0, 0, fw, fh)));

        let (tx, rx) = mpsc::channel::<bool>();
        let worker = std::thread::spawn(move || {
            let _ = ImageProcessorTrait::convert(
                &mut proc,
                &src,
                &mut dst,
                Some(ConvertOptions::new().crop(crop)),
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

    /// One processor converting many varying-size NV frames from a reused pool keeps per-convert latency bounded.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv_to_planar_f16_varying_sizes_no_leak_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("{} — init failed ({e:?})", function!()));
                return;
            }
        };
        // A pool of source R8 tensors and a pool of `PlanarRgb` F16 destinations, both of depth 4.
        let (max_w, max_h) = (640usize, 640usize);
        let depth = 4usize;
        let mut srcs = Vec::new();
        let mut dsts = Vec::new();
        for _ in 0..depth {
            let Some(src) = crate::test_support::gpu_buffer_allocated(
                TensorDyn::image(
                    max_w,
                    max_h * 3,
                    PixelFormat::Grey,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                "R8 pool",
            ) else {
                return;
            };
            srcs.push(src);
            let Some(dst) = crate::test_support::gpu_buffer_allocated(
                TensorDyn::image(
                    640,
                    640,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                "F16 PlanarRgb dst",
            ) else {
                return;
            };
            dsts.push(dst);
        }
        // Assorted frame sizes, all ≤ max, cycled.
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
            src.as_typed::<u8>()
                .unwrap()
                .map()
                .unwrap()
                .as_mut_slice()
                .fill(128);
            let crop = Crop::new().with_source(Some(Region::new(0, 0, fw, fh)));
            let t0 = std::time::Instant::now();
            ImageProcessorTrait::convert(
                &mut gpu,
                src,
                dst,
                Some(ConvertOptions::new().crop(crop)),
            )
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

    /// NV12, NV16 and NV24 IOSurface sources convert to RGBA on the GPU matching the CPU within shader rounding.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_nv16_nv24_to_rgba_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        let mut cpu = CPUProcessor::new();

        // Fills Y and the interleaved UV plane in the semi-planar layout: the Y
        // plane is `h` rows at `stride`; the UV plane starts at `h * stride`; each
        // chroma row advances `uv_grid_rows * stride` bytes; each (Cb, Cr) pair is
        // two consecutive bytes at column `cx * 2`.
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
                (16usize, 16usize), // even dimensions
                (15, 16),           // odd-W
                (16, 15),           // odd-H
            ] {
                let mem = TensorDyn::image(
                    w,
                    h,
                    fmt,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                let mem_stride = mem
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                fill(
                    mem.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice(),
                    mem_stride,
                    fmt,
                    w,
                    h,
                );
                let cpu_dst = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Rgba,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                let (r, _s, cpu_dst) = convert_img(
                    &mut cpu,
                    mem,
                    cpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| panic!("CPU {fmt:?}->{w}x{h}->RGBA: {e}"));

                let ios = TensorDyn::image(
                    w,
                    h,
                    fmt,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h} IOSurface alloc: {e}"));
                let ios_stride = ios
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                fill(
                    ios.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice(),
                    ios_stride,
                    fmt,
                    w,
                    h,
                );
                let gpu_dst = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Rgba,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
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

                let cs = cpu_dst
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                let cmap = cpu_dst.as_typed::<u8>().unwrap().map().unwrap();
                let cb = cmap.as_slice();
                let gs = gpu_dst
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                let gmap = gpu_dst.as_typed::<u8>().unwrap().map().unwrap();
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

    /// NV frames in a larger reused R8 pool IOSurface convert on the GPU matching the CPU within 3 LSB.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv_to_rgba_larger_pool_surface_opengl_macos() {
        let mut gpu = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        let mut cpu = CPUProcessor::new();
        let (pool_w, pool_h) = (256usize, 256usize);

        // Semi-planar fill: the Y plane at the row stride, then the UV plane at
        // `h * stride`, each chroma row advancing `uv_grid_rows * stride` bytes.
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
                (40usize, 24usize), // even dimensions
                (15, 16),           // odd-W
                (16, 15),           // odd-H
            ] {
                // `ew` is the minimum even extent.
                let ew = w.next_multiple_of(2);

                // CPU reference from a tightly-packed Mem tensor at the frame size.
                let mem = TensorDyn::image(
                    w,
                    h,
                    fmt,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                let mem_stride = mem
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                fill(
                    mem.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice(),
                    mem_stride,
                    fmt,
                    w,
                    h,
                );
                let cpu_dst = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Rgba,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap();
                let (r, _s, cpu_dst) = convert_img(
                    &mut cpu,
                    mem,
                    cpu_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                );
                r.unwrap_or_else(|e| panic!("CPU {fmt:?}->{w}x{h}->RGBA: {e}"));

                // GPU source: a larger R8 pool surface reconfigured to the frame, filled at the pool's row stride.
                let Some(mut ios) = crate::test_support::gpu_buffer_allocated(
                    TensorDyn::image(
                        pool_w,
                        pool_h,
                        PixelFormat::Grey,
                        DType::U8,
                        Some(TensorMemory::DmaBuf),
                        edgefirst_tensor::CpuAccess::ReadWrite,
                    ),
                    "R8 pool IOSurface",
                ) else {
                    return;
                };
                ios.configure_image(w, h, fmt)
                    .unwrap_or_else(|e| panic!("configure_image {fmt:?} {w}x{h} on pool: {e}"));
                let ios_stride = ios
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                assert!(
                    ios_stride > ew,
                    "{fmt:?} {w}x{h}: pool stride {ios_stride} should exceed even width {ew} \
                     (test must exercise padding)"
                );
                fill(
                    ios.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice(),
                    ios_stride,
                    fmt,
                    w,
                    h,
                );

                let gpu_dst = TensorDyn::image(
                    w,
                    h,
                    PixelFormat::Rgba,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
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

                let cs = cpu_dst
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                let cmap = cpu_dst.as_typed::<u8>().unwrap().map().unwrap();
                let cb = cmap.as_slice();
                let gs = gpu_dst
                    .as_typed::<u8>()
                    .unwrap()
                    .effective_row_stride()
                    .unwrap();
                let gmap = gpu_dst.as_typed::<u8>().unwrap().map().unwrap();
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
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?}). \
                     Install ANGLE via `brew install startergo/angle/angle` \
                     and re-sign per README.md § macOS GPU Acceleration to \
                     run this test.",
                    function!()
                ));
                return;
            }
        };

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    /// The GL backend converts mid-grey YUYV to RGBA at 64×32 and at 3840×2160.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_opengl_macos_multi_resolution() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };

        for (w, h) in [(64usize, 32usize), (3840, 2160)] {
            // Synthetic YUYV: Y=128, U=V=128.
            let bytes_per_row = w * 2;
            let mut yuyv = vec![0u8; bytes_per_row * h];
            for chunk in yuyv.as_chunks_mut::<4>().0 {
                chunk[0] = 128; // Y0
                chunk[1] = 128; // U
                chunk[2] = 128; // Y1
                chunk[3] = 128; // V
            }

            let src =
                load_bytes_to_tensor(w, h, PixelFormat::Yuyv, Some(TensorMemory::DmaBuf), &yuyv)
                    .unwrap();

            let dst = TensorDyn::image(
                w,
                h,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
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
            result.expect("GL convert should succeed at this resolution");

            let dst_u8 = dst.as_typed::<u8>().unwrap();
            let dst_map = dst_u8.map().unwrap();
            let dst_bytes = dst_map.as_slice();
            assert_eq!(dst_bytes.len(), w * h * 4, "RGBA byte count");
            for px in dst_bytes.as_chunks::<4>().0 {
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

    /// YUYV and RGBA IOSurface sources convert on GL into a `420v` CoreVideo
    /// destination wrapped by `from_planar_iosurface`: both planes, read
    /// through the CoreVideo buffer, are within 2 of the CPU converting into
    /// memory, and the bytes between the planes keep their sentinel.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_into_a_planar_surface_opengl_macos() {
        use crate::test_support::corevideo_buffer;
        const SENTINEL: u8 = 0x5A;
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        gl.set_soft_declines(false).unwrap();
        let image = |w: usize, h: usize, fmt: PixelFormat, mem: TensorMemory| {
            TensorDyn::image(
                w,
                h,
                fmt,
                DType::U8,
                Some(mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let max_diff = |a: &[u8], b: &[u8]| {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
        };

        for (w, h) in [(1920usize, 1080usize), (1000, 562)] {
            let gradient = image(w, h, PixelFormat::Rgba, TensorMemory::Mem);
            {
                let t = gradient.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let mut map = t.map().unwrap();
                for y in 0..h {
                    for x in 0..w {
                        map.as_mut_slice()[y * stride + x * 4..][..4].copy_from_slice(&[
                            (40 + x * 170 / w) as u8,
                            (40 + y * 170 / h) as u8,
                            (200 - (x + y) * 150 / (w + h)) as u8,
                            255,
                        ]);
                    }
                }
            }
            for fmt in [PixelFormat::Yuyv, PixelFormat::Rgba] {
                let what = format!("{fmt:?} {w}x{h}");
                let mut src = image(w, h, fmt, TensorMemory::DmaBuf);
                CPUProcessor::new()
                    .convert(&gradient, &mut src, None)
                    .unwrap();

                let buf = corevideo_buffer(w, h, b"420v");
                buf.fill_rows(|_, _, bytes| bytes.fill(0));
                let mut gap = 0;
                buf.between_planes(|bytes| {
                    bytes.fill(SENTINEL);
                    gap = bytes.len();
                });
                let expected_gap = if (w, h) == (1000, 562) { 6144 } else { 0 };
                assert_eq!(gap, expected_gap, "{what}: bytes between the planes");
                let mut dst = unsafe { TensorDyn::from_planar_iosurface(buf.surface(), None) }
                    .expect("from_planar_iosurface");
                let mut want = image(w, h, PixelFormat::Nv12, TensorMemory::Mem);
                want.set_colorimetry(dst.colorimetry());

                gl.convert(&src, &mut dst, None)
                    .unwrap_or_else(|e| panic!("{what}: GL convert: {e}"));
                CPUProcessor::new()
                    .convert(&src, &mut want, None)
                    .unwrap_or_else(|e| panic!("{what}: CPU convert: {e}"));

                let (mut got_y, mut got_uv) = (Vec::new(), Vec::new());
                buf.fill_rows(|plane, _, bytes| match plane {
                    0 => got_y.extend_from_slice(&bytes[..w]),
                    _ => got_uv.extend_from_slice(&bytes[..w]),
                });
                let t = want.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let map = t.map().unwrap();
                let rows = |first: usize, n: usize| -> Vec<u8> {
                    (first..first + n)
                        .flat_map(|r| map.as_slice()[r * stride..][..w].to_vec())
                        .collect()
                };
                let (want_y, want_uv) = (rows(0, h), rows(h, h / 2));

                let y = max_diff(&got_y, &want_y);
                assert!(y <= 2, "{what}: luma max diff {y} > 2");
                let uv = max_diff(&got_uv, &want_uv);
                assert!(uv <= 2, "{what}: chroma max diff {uv} > 2");
                buf.between_planes(|bytes| {
                    assert!(
                        bytes.iter().all(|&b| b == SENTINEL),
                        "{what}: the bytes between the planes were written"
                    );
                });
            }
        }
    }

    /// The CPU writes a `420v` CoreVideo destination wrapped by `from_planar_iosurface`, from a
    /// `420v` source and from RGBA, byte-for-byte as it writes a memory NV12, and leaves the
    /// bytes between the planes untouched.
    #[test]
    #[cfg(target_os = "macos")]
    fn test_cpu_into_a_planar_surface_macos() {
        use crate::test_support::corevideo_buffer;
        const SENTINEL: u8 = 0x5A;
        let image = |w: usize, h: usize, fmt: PixelFormat| {
            TensorDyn::image(
                w,
                h,
                fmt,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };

        for (w, h) in [(1920usize, 1080usize), (1000, 562)] {
            let wrap = |buf: &crate::test_support::CoreVideoBuffer| {
                unsafe { TensorDyn::from_planar_iosurface(buf.surface(), None) }
                    .expect("from_planar_iosurface")
            };
            let read = |buf: &crate::test_support::CoreVideoBuffer| {
                let (mut y, mut uv) = (Vec::new(), Vec::new());
                buf.fill_rows(|plane, _, bytes| match plane {
                    0 => y.extend_from_slice(&bytes[..w]),
                    _ => uv.extend_from_slice(&bytes[..w]),
                });
                (y, uv)
            };
            let tight = |t: &TensorDyn| {
                let t = t.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let map = t.map().unwrap();
                let rows = |first: usize, n: usize| -> Vec<u8> {
                    (first..first + n)
                        .flat_map(|r| map.as_slice()[r * stride..][..w].to_vec())
                        .collect()
                };
                (rows(0, h), rows(h, h / 2))
            };
            let check_padding = |buf: &crate::test_support::CoreVideoBuffer, what: &str| {
                buf.between_planes(|bytes| {
                    assert!(
                        bytes.iter().all(|&b| b == SENTINEL),
                        "{what}: the bytes between the planes were written"
                    );
                });
            };
            let prepare = |buf: &crate::test_support::CoreVideoBuffer| {
                buf.fill_rows(|_, _, bytes| bytes.fill(0));
                buf.between_planes(|bytes| bytes.fill(SENTINEL));
            };

            // A 420v source copied into a 420v destination.
            let what = format!("420v -> 420v {w}x{h}");
            let src_buf = corevideo_buffer(w, h, b"420v");
            src_buf.fill_rows(|plane, row, bytes| {
                for (x, b) in bytes[..w].iter_mut().enumerate() {
                    *b = if plane == 0 {
                        (x * 9 + row * 5) & 0xff
                    } else {
                        (x * 11 + row * 7 + 30) & 0xff
                    } as u8;
                }
            });
            let dst_buf = corevideo_buffer(w, h, b"420v");
            prepare(&dst_buf);
            let mut src = wrap(&src_buf);
            let mut dst = wrap(&dst_buf);
            src.set_colorimetry(dst.colorimetry());
            let mut want = image(w, h, PixelFormat::Nv12);
            want.set_colorimetry(dst.colorimetry());
            CPUProcessor::new()
                .convert(&src, &mut dst, None)
                .unwrap_or_else(|e| panic!("{what}: CPU convert: {e}"));
            CPUProcessor::new()
                .convert(&src, &mut want, None)
                .unwrap_or_else(|e| panic!("{what}: CPU convert to memory: {e}"));
            assert!(
                read(&dst_buf) == tight(&want),
                "{what}: the planes differ from the CPU writing into memory"
            );
            check_padding(&dst_buf, &what);

            // An RGBA source converted into a 420v destination.
            let what = format!("RGBA -> 420v {w}x{h}");
            let rgba = image(w, h, PixelFormat::Rgba);
            {
                let t = rgba.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let mut map = t.map().unwrap();
                for y in 0..h {
                    for x in 0..w {
                        map.as_mut_slice()[y * stride + x * 4..][..4].copy_from_slice(&[
                            (40 + x * 170 / w) as u8,
                            (40 + y * 170 / h) as u8,
                            (200 - (x + y) * 150 / (w + h)) as u8,
                            255,
                        ]);
                    }
                }
            }
            let dst_buf = corevideo_buffer(w, h, b"420v");
            prepare(&dst_buf);
            let mut dst = wrap(&dst_buf);
            let mut want = image(w, h, PixelFormat::Nv12);
            want.set_colorimetry(dst.colorimetry());
            CPUProcessor::new()
                .convert(&rgba, &mut dst, None)
                .unwrap_or_else(|e| panic!("{what}: CPU convert: {e}"));
            CPUProcessor::new()
                .convert(&rgba, &mut want, None)
                .unwrap_or_else(|e| panic!("{what}: CPU convert to memory: {e}"));
            assert!(
                read(&dst_buf) == tight(&want),
                "{what}: the planes differ from the CPU writing into memory"
            );
            check_padding(&dst_buf, &what);
        }
    }

    /// A `420v` CoreVideo source wrapped by `from_planar_iosurface` converts on GL to a
    /// letterboxed `PlanarRgb` F16 model input within the float-path tolerance of the CPU.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_planar_nv12_to_planar_f16_letterbox_opengl_macos() {
        use crate::test_support::corevideo_buffer;
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        gl.set_soft_declines(false).unwrap();
        let (w, h) = (1000usize, 562usize);
        let buf = corevideo_buffer(w, h, b"420v");
        buf.fill_rows(|plane, row, bytes| {
            if plane == 0 {
                for (x, b) in bytes[..w].iter_mut().enumerate() {
                    *b = (16 + x * 200 / w + row * 20 / h) as u8;
                }
            } else {
                for (cx, pair) in bytes[..w].chunks_exact_mut(2).enumerate() {
                    pair[0] = (40 + cx * 300 / w) as u8;
                    pair[1] = (200 - row * 300 / h) as u8;
                }
            }
        });
        let src = unsafe { TensorDyn::from_planar_iosurface(buf.surface(), None) }
            .expect("from_planar_iosurface");
        let Some(mut gl_dst) = crate::test_support::gpu_buffer_allocated(
            TensorDyn::image(
                640,
                640,
                PixelFormat::PlanarRgb,
                DType::F16,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            ),
            "F16 PlanarRgb IOSurface",
        ) else {
            return;
        };
        let options = crate::ConvertOptions::new().letterbox([114, 114, 114, 255]);
        gl.convert(&src, &mut gl_dst, Some(options))
            .unwrap_or_else(|e| panic!("GL convert: {e}"));

        let mut cpu_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        CPUProcessor::new()
            .convert(&src, &mut cpu_dst, Some(options))
            .expect("CPU reference convert");

        let values = |t: &TensorDyn| -> Vec<f32> {
            let typed = t.as_typed::<half::f16>().unwrap();
            let mut out = vec![0u8; typed.shape().iter().product::<usize>() * 2];
            typed.copy_to_flat(&mut out).unwrap();
            out.chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        };
        let (g, c) = (values(&gl_dst), values(&cpu_dst));
        assert_eq!(g.len(), c.len());
        let max_diff = g
            .iter()
            .zip(&c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 4.0 / 255.0 + 1e-3,
            "planar NV12 to PlanarRgb F16 diverges from the CPU: max_diff={max_diff}"
        );
    }

    /// An RGBA IOSurface source letterboxed at an odd offset, and flipped,
    /// converts on GL into a `420v` CoreVideo destination wrapped by
    /// `from_planar_iosurface`: both planes are within 2 of the CPU, and the
    /// bytes between the planes keep their sentinel.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_nv12_letterbox_into_a_planar_surface_opengl_macos() {
        use crate::test_support::corevideo_buffer;
        const SENTINEL: u8 = 0x5A;
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        gl.set_soft_declines(false).unwrap();
        let image = |w: usize, h: usize, fmt: PixelFormat, mem: TensorMemory| {
            TensorDyn::image(
                w,
                h,
                fmt,
                DType::U8,
                Some(mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let max_diff = |a: &[u8], b: &[u8]| {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap()
        };

        // Each source letterboxes into its destination with an odd left edge.
        for ((w, h), (sw, sh)) in [
            ((1920usize, 1080usize), (600usize, 480usize)),
            ((1000, 562), (640, 480)),
        ] {
            let src = image(sw, sh, PixelFormat::Rgba, TensorMemory::DmaBuf);
            {
                let t = src.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let mut map = t.map().unwrap();
                for y in 0..sh {
                    for x in 0..sw {
                        map.as_mut_slice()[y * stride + x * 4..][..4].copy_from_slice(&[
                            (40 + x * 170 / sw) as u8,
                            (40 + y * 170 / sh) as u8,
                            (200 - (x + y) * 150 / (sw + sh)) as u8,
                            255,
                        ]);
                    }
                }
            }
            for (geometry, options) in [
                (
                    "letterbox",
                    ConvertOptions::new().letterbox([200, 30, 90, 255]),
                ),
                ("flip", ConvertOptions::new().flip(Flip::Horizontal)),
            ] {
                let what = format!("{sw}x{sh} -> {w}x{h} {geometry}");
                let buf = corevideo_buffer(w, h, b"420v");
                buf.fill_rows(|_, _, bytes| bytes.fill(0));
                buf.between_planes(|bytes| bytes.fill(SENTINEL));
                let mut dst = unsafe { TensorDyn::from_planar_iosurface(buf.surface(), None) }
                    .expect("from_planar_iosurface");
                let mut want = image(w, h, PixelFormat::Nv12, TensorMemory::Mem);
                want.set_colorimetry(dst.colorimetry());

                gl.convert(&src, &mut dst, Some(options))
                    .unwrap_or_else(|e| panic!("{what}: GL convert: {e}"));
                CPUProcessor::new()
                    .convert(&src, &mut want, Some(options))
                    .unwrap_or_else(|e| panic!("{what}: CPU convert: {e}"));

                let (mut got_y, mut got_uv) = (Vec::new(), Vec::new());
                buf.fill_rows(|plane, _, bytes| match plane {
                    0 => got_y.extend_from_slice(&bytes[..w]),
                    _ => got_uv.extend_from_slice(&bytes[..w]),
                });
                let t = want.as_u8().unwrap();
                let stride = t.effective_row_stride().unwrap();
                let map = t.map().unwrap();
                let rows = |first: usize, n: usize| -> Vec<u8> {
                    (first..first + n)
                        .flat_map(|r| map.as_slice()[r * stride..][..w].to_vec())
                        .collect()
                };
                let (want_y, want_uv) = (rows(0, h), rows(h, h / 2));

                let y = max_diff(&got_y, &want_y);
                assert!(y <= 2, "{what}: luma max diff {y} > 2");
                let uv = max_diff(&got_uv, &want_uv);
                assert!(uv <= 2, "{what}: chroma max diff {uv} > 2");
                buf.between_planes(|bytes| {
                    assert!(
                        bytes.iter().all(|&b| b == SENTINEL),
                        "{what}: the bytes between the planes were written"
                    );
                });
            }
        }
    }

    /// Two consecutive converts of one source tensor succeed and produce identical output.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_pbuffer_cache_reuses_surfaces() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };

        let mut yuyv = vec![0u8; 64 * 32 * 2];
        for chunk in yuyv.as_chunks_mut::<4>().0 {
            chunk[0] = 200;
            chunk[1] = 100;
            chunk[2] = 200;
            chunk[3] = 156;
        }
        let src =
            load_bytes_to_tensor(64, 32, PixelFormat::Yuyv, Some(TensorMemory::DmaBuf), &yuyv)
                .unwrap();
        let dst = TensorDyn::image(
            64,
            32,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        let first: Vec<u8> = dst
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();

        let (r2, _src, dst) = convert_img(
            &mut proc,
            src,
            dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        r2.unwrap();
        let second: Vec<u8> = dst
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();

        assert_eq!(first, second, "cache-hit conversion must be deterministic");
    }

    /// A convert loop over a fixed pool of IOSurface tensors creates no EGL pbuffers after the pool has been seen once.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_pbuffer_cache_steady_state() {
        let mut proc = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} — GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };

        let (w, h) = (64usize, 32usize);
        const POOL: usize = 3;
        const FRAMES: usize = 100;

        let yuyv = vec![128u8; w * h * 2];
        let pool: Vec<TensorDyn> = (0..POOL)
            .map(|_| {
                load_bytes_to_tensor(w, h, PixelFormat::Yuyv, Some(TensorMemory::DmaBuf), &yuyv)
                    .unwrap()
            })
            .collect();
        let mut dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        // Warm-up: two passes over the pool.
        for src in pool.iter().cycle().take(POOL * 2) {
            proc.convert(src, &mut dst, None).unwrap();
        }
        let warm = proc.egl_cache_stats().unwrap();

        for src in pool.iter().cycle().take(FRAMES) {
            proc.convert(src, &mut dst, None).unwrap();
        }
        let steady = proc.egl_cache_stats().unwrap();

        assert_eq!(
            warm.total_misses(),
            steady.total_misses(),
            "steady-state loop created new imports (warm {warm:?}, steady {steady:?})"
        );
        let hits = |s: &GlCacheStats| s.src.hits + s.dst.hits + s.nv_r8.hits + s.nv_uv.hits;
        assert!(
            hits(&steady) - hits(&warm) >= FRAMES as u64,
            "expected at least {FRAMES} import-cache hits over the loop, got {}",
            hits(&steady) - hits(&warm)
        );
    }

    /// The NV12 to `PlanarRgb` F16 IOSurface convert runs on the GL engine, shown by its import-cache counters.
    #[test]
    #[cfg(target_os = "macos")]
    #[cfg(feature = "opengl")]
    fn test_macos_gl_f16_planar_is_gl_backed() {
        let mut proc = ImageProcessor::new().expect("ImageProcessor");
        let Some(ref gl) = proc.opengl else {
            crate::test_support::report_skip(&format!("{} — GL backend unavailable", function!()));
            return;
        };
        if !gl.supported_render_dtypes().f16 {
            crate::test_support::report_skip(&format!(
                "{} — configuration lacks F16 color-buffer support",
                function!()
            ));
            return;
        }
        let stats_before = gl.egl_cache_stats().expect("cache stats");

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = src.as_typed::<u8>().unwrap();
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        proc.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().crop(Crop::letterbox([114, 114, 114, 255]))),
        )
        .expect("F16 capability reported but the NV12→PlanarF16 convert failed");
        let stats_after = proc
            .opengl
            .as_ref()
            .expect("GL backend present")
            .egl_cache_stats()
            .expect("cache stats");
        assert!(
            stats_after.total_misses() >= stats_before.total_misses() + 2,
            "convert succeeded but the GL engine did not import both the \
             source and the F16 destination — the work did not (fully) run \
             on the GL backend (silent CPU fallback); misses before={} after={}",
            stats_before.total_misses(),
            stats_after.total_misses()
        );
    }

    /// The fused NV12 to `PlanarRgb` F16 GL convert matches the CPU backend within the float-path tolerance.
    #[test]
    #[cfg(feature = "opengl")]
    fn test_nv12_to_planar_f16_fused_engine_vs_cpu() {
        let mut gl = match ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        }) {
            Ok(p) if p.opengl.is_some() => p,
            _ => {
                crate::test_support::report_skip(&format!(
                    "{} — GL backend unavailable",
                    function!()
                ));
                return;
            }
        };
        if !gl
            .opengl
            .as_ref()
            .map(|g| g.supported_render_dtypes().f16)
            .unwrap_or(false)
        {
            crate::test_support::report_skip(&format!("{} — no F16 render support", function!()));
            return;
        }
        let mem = if crate::test_support::is_gpu_buffer_available() {
            TensorMemory::DmaBuf
        } else {
            crate::test_support::report_skip(&format!("{} — no zero-copy buffers", function!()));
            return;
        };

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            Some(mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            // Fills the source with smooth gradients.
            let t = src.as_typed::<u8>().unwrap();
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
        let mut gl_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        // Converts on the GL backend directly.
        match gl.opengl.as_mut().expect("GL backend present").convert(
            &src,
            &mut gl_dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        ) {
            Ok(()) => {}
            Err(crate::Error::NotSupported(msg)) => {
                crate::test_support::report_skip(&format!("{} — {msg}", function!()));
                return;
            }
            Err(e) => panic!("fused NV12→PlanarF16 GL convert: {e}"),
        }

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
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu.convert(
            &src,
            &mut cpu_dst,
            Some(crate::ConvertOptions::new().crop(crop)),
        )
        .expect("CPU reference convert");

        let g = gl_dst
            .as_typed::<half::f16>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();
        let c = cpu_dst
            .as_typed::<half::f16>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();
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
        // Plane, row and column of the worst element.
        let (plane, rem) = (max_at / (640 * 640), max_at % (640 * 640));
        let (row, col) = (rem / 640, rem % 640);
        eprintln!(
            "fused-vs-cpu: max_diff={max_diff} at plane={plane} row={row} col={col} \
             gl={} cpu={}",
            g[max_at].to_f32(),
            c[max_at].to_f32()
        );
        assert!(
            max_diff <= 4.0 / 255.0 + 1e-3,
            "fused NV12→PlanarF16 diverges from CPU reference: max_diff={max_diff}"
        );
    }

    /// The GL backend, called directly, converts a zero-copy RGBA source to a heap BGRA destination.
    #[test]
    #[cfg(feature = "opengl")]
    fn test_zero_copy_src_to_mem_dst_gl_direct() {
        let mut proc = match ImageProcessor::new() {
            Ok(p) if p.opengl.is_some() => p,
            _ => {
                crate::test_support::report_skip(&format!(
                    "{} — GL backend unavailable",
                    function!()
                ));
                return;
            }
        };
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!("{} — no zero-copy buffers", function!()));
            return;
        }

        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = src.as_typed::<u8>().unwrap();
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
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        proc.opengl
            .as_mut()
            .expect("GL backend present")
            .convert(&src, &mut gl_dst, None)
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
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        cpu.convert(&src, &mut cpu_dst, None)
            .expect("CPU reference convert");

        let g = gl_dst
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();
        let c = cpu_dst
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_slice()
            .to_vec();
        assert_eq!(g.len(), c.len());
        let max_diff = g
            .iter()
            .zip(c.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(
            max_diff <= 2,
            "zero-copy src → heap dst diverges from CPU reference: max_diff={max_diff}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgb_g2d() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_yuyv_to_rgb_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_yuyv_to_rgb_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_yuyv_to_yuyv_resize_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_yuyv_to_yuyv_resize_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
        let dst_image = |memory| {
            TensorDyn::image(
                600,
                400,
                PixelFormat::Yuyv,
                DType::U8,
                memory,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut g2d_converter = G2DProcessor::new().unwrap();

        let mut untagged = dst_image(Some(TensorMemory::DmaBuf));
        let declined = g2d_converter.convert(&src, &mut untagged, None);
        assert!(
            matches!(declined, Err(Error::NotSupported(_))),
            "{declined:?}"
        );
        assert_eq!(untagged.colorimetry(), None);

        let src_cm = Some(crate::colorimetry::effective_colorimetry(&src));
        let mut g2d_dst = dst_image(Some(TensorMemory::DmaBuf));
        g2d_dst.set_colorimetry(src_cm);
        let (result, src, g2d_dst) = convert_img(
            &mut g2d_converter,
            src,
            g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        );
        result.unwrap();

        let mut cpu_dst = dst_image(None);
        cpu_dst.set_colorimetry(src_cm);
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

        // TODO: compare YUYV↔YUYV directly without a YUYV→RGB convert.
        compare_images_convert_to_rgb(&g2d_dst, &cpu_dst, 0.95, function!());
    }

    /// Whether this host has G2D and a usable DMA heap; reports a skip when it has not.
    #[cfg(target_os = "linux")]
    fn g2d_and_dma_or_skip(test: &str) -> bool {
        let why = if let Some(why) = g2d_skip_reason() {
            why
        } else if !is_dma_available() {
            "DMA not available"
        } else {
            return true;
        };
        crate::test_support::report_skip(&format!("{test} - {why}"));
        false
    }

    /// An untagged 1280x720 YUYV image of eight saturated colour bars, encoded in BT.709
    /// limited, the colorimetry its height implies.
    #[cfg(target_os = "linux")]
    fn colour_bars_yuyv_720p() -> TensorDyn {
        const BARS: [[u8; 3]; 8] = [
            [255, 255, 255],
            [255, 255, 0],
            [0, 255, 255],
            [0, 255, 0],
            [255, 0, 255],
            [255, 0, 0],
            [0, 0, 255],
            [0, 0, 0],
        ];
        let (w, h) = (1280, 720);
        let rgb = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = rgb.as_typed::<u8>().unwrap();
            let mut map = t.map().unwrap();
            for (i, px) in map.as_mut_slice().chunks_exact_mut(3).enumerate() {
                px.copy_from_slice(&BARS[(i % w) * BARS.len() / w]);
            }
        }
        let mut src = TensorDyn::image(
            w,
            h,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        CPUProcessor::new().convert(&rgb, &mut src, None).unwrap();
        src.set_colorimetry(None);
        src
    }

    /// An untagged 640x360 YUYV image (BT.601 limited by its height).
    #[cfg(target_os = "linux")]
    fn yuyv_640x360(memory: Option<TensorMemory>) -> TensorDyn {
        TensorDyn::image(
            640,
            360,
            PixelFormat::Yuyv,
            DType::U8,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap()
    }

    /// G2D declines a YUV-to-YUV conversion into an untagged destination of another
    /// colorimetry while soft declines are on. With them off it writes the source's
    /// colorimetry and tags the destination with it.
    #[test]
    #[cfg(target_os = "linux")]
    fn g2d_mismatched_yuv_untagged_destination_is_a_soft_decline() {
        if !g2d_and_dma_or_skip(function!()) {
            return;
        }
        let src = colour_bars_yuyv_720p();
        let src_cm = crate::colorimetry::effective_colorimetry(&src);
        let mut g2d = G2DProcessor::new().unwrap();
        assert!(g2d.soft_declines());

        let mut g2d_dst = yuyv_640x360(Some(TensorMemory::DmaBuf));
        let declined = g2d.convert(&src, &mut g2d_dst, None);
        assert!(
            matches!(declined, Err(Error::NotSupported(_))),
            "{declined:?}"
        );
        assert_eq!(g2d_dst.colorimetry(), None);

        g2d.set_soft_declines(false);
        g2d.convert(&src, &mut g2d_dst, None).unwrap();
        assert_eq!(g2d_dst.colorimetry(), Some(src_cm));

        let mut cpu = CPUProcessor::new();
        let mut cpu_src_coding = yuyv_640x360(None);
        cpu_src_coding.set_colorimetry(Some(src_cm));
        cpu.convert(&src, &mut cpu_src_coding, None).unwrap();
        let mut cpu_dst_coding = yuyv_640x360(None);
        cpu.convert(&src, &mut cpu_dst_coding, None).unwrap();
        assert_ne!(cpu_dst_coding.colorimetry(), Some(src_cm));

        let mean_abs_diff = |a: &TensorDyn, b: &TensorDyn| {
            let a = a.as_typed::<u8>().unwrap().map().unwrap().to_vec();
            let b = b.as_typed::<u8>().unwrap().map().unwrap().to_vec();
            let sum: u64 = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
            sum as f64 / a.len() as f64
        };
        let to_src_coding = mean_abs_diff(&g2d_dst, &cpu_src_coding);
        let to_dst_coding = mean_abs_diff(&g2d_dst, &cpu_dst_coding);
        assert!(
            to_src_coding < 1.0 && to_src_coding < to_dst_coding,
            "mean byte difference from the CPU output in the source's colorimetry \
             {to_src_coding:.3}, in the destination's {to_dst_coding:.3}"
        );
    }

    /// G2D declines a YUV-to-YUV conversion into a destination tagged with another
    /// colorimetry, with soft declines off.
    #[test]
    #[cfg(target_os = "linux")]
    fn g2d_mismatched_yuv_tagged_destination_is_a_hard_decline() {
        if !g2d_and_dma_or_skip(function!()) {
            return;
        }
        let src = colour_bars_yuyv_720p();
        let mut g2d = G2DProcessor::new().unwrap();
        g2d.set_soft_declines(false);
        let mut dst = yuyv_640x360(Some(TensorMemory::DmaBuf));
        let bt601 = crate::colorimetry::effective_colorimetry(&dst);
        dst.set_colorimetry(Some(bt601));
        let declined = g2d.convert(&src, &mut dst, None);
        assert!(
            matches!(declined, Err(Error::NotSupported(_))),
            "{declined:?}"
        );
        assert_eq!(dst.colorimetry(), Some(bt601));
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

        let dst = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let dst_target = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        compare_images(&dst, &dst_target, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_yuyv_to_rgba_crop_flip_g2d() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_yuyv_to_rgba_crop_flip_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_yuyv_to_rgba_crop_flip_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let dst_g2d = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_yuyv_to_rgba_crop_flip_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }

        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - zero-copy GPU buffer (DMA-BUF / IOSurface) not available on this host",
                function!()
            ));
            return;
        }

        let src = load_bytes_to_tensor(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::DmaBuf),
            &edgefirst_bench::testdata::read("camera720p.yuyv"),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let dst_gl = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        if let Err(crate::Error::NotSupported(msg)) = &result {
            crate::test_support::report_skip(&format!("{} — {msg}", function!()));
            return;
        }
        result.unwrap();

        let dst_cpu = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        let file = edgefirst_bench::testdata::read("camera720p.vyuy").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Vyuy,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgb_cpu() {
        let file = edgefirst_bench::testdata::read("camera720p.vyuy").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Vyuy,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
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

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "G2D does not support VYUY; re-enable when hardware support is added"]
    fn test_vyuy_to_rgba_g2d() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_vyuy_to_rgba_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_vyuy_to_rgba_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
                crate::test_support::report_skip(
                    "test_vyuy_to_rgba_g2d - G2D does not support PixelFormat::Vyuy format",
                );
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "G2D does not support VYUY; re-enable when hardware support is added"]
    fn test_vyuy_to_rgb_g2d() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("test_vyuy_to_rgb_g2d - {why}"));
            return;
        }
        if !is_dma_available() {
            crate::test_support::report_skip("test_vyuy_to_rgb_g2d - DMA memory allocation not available (permission denied or no DMA-BUF support)");
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
                crate::test_support::report_skip(
                    "test_vyuy_to_rgb_g2d - G2D does not support PixelFormat::Vyuy format",
                );
                return;
            }
            r => r.unwrap(),
        }

        let cpu_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    fn test_vyuy_to_rgba_opengl() {
        if !is_opengl_available() {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        }
        if !crate::test_support::is_gpu_buffer_available() {
            crate::test_support::report_skip(&format!(
                "{} - zero-copy GPU buffer (DMA-BUF / IOSurface) not available on this host",
                function!()
            ));
            return;
        }

        let expect = if cfg!(target_os = "linux") {
            crate::test_support::Expect::Succeeds
        } else {
            crate::test_support::Expect::MayFail("no zero-copy VYUY mapping on this platform")
        };
        let Some(src) = crate::test_support::allocated(
            load_bytes_to_tensor(
                1280,
                720,
                PixelFormat::Vyuy,
                Some(TensorMemory::DmaBuf),
                &edgefirst_bench::testdata::read("camera720p.vyuy"),
            ),
            "zero-copy VYUY source",
            expect,
        ) else {
            return;
        };

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
                crate::test_support::report_skip(&format!(
                    "{} - OpenGL does not support PixelFormat::Vyuy DMA format",
                    function!()
                ));
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        target_image
            .as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&edgefirst_bench::testdata::read("camera720p.rgba"));

        compare_images(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_nv12_to_rgba_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_odd_height_to_rgb_cpu() {
        // NV12 with odd height 5: the buffer is `[5 + ceil(5/2), 8]` = `[8, 8]`, filled with neutral grey.
        let mut src = TensorDyn::image(
            8,
            5,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(src.shape(), &[8, 8]);
        assert_eq!((src.width(), src.height()), (Some(8), Some(5)));
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let dst = TensorDyn::image(
            8,
            5,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let map = dst.as_typed::<u8>().unwrap().map().unwrap();
        for (i, &b) in map.as_slice().iter().enumerate() {
            assert!(
                (b as i16 - 128).abs() <= 2,
                "pixel byte {i} = {b}, expected ~128 for neutral-grey NV12"
            );
        }
    }

    #[test]
    fn test_nv24_to_rgb_cpu() {
        // NV24 at 8×4: the buffer is `[12, 8]`, a 4-row Y plane and an 8-row interleaved UV plane, filled with neutral grey.
        let mut src = TensorDyn::image(
            8,
            4,
            PixelFormat::Nv24,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(src.shape(), &[12, 8]);
        assert_eq!((src.width(), src.height()), (Some(8), Some(4)));
        src.as_typed::<u8>()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        src.set_colorimetry(Some(
            edgefirst_tensor::Colorimetry::default()
                .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
                .with_range(edgefirst_tensor::ColorRange::Full),
        ));

        let dst = TensorDyn::image(
            8,
            4,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let map = dst.as_typed::<u8>().unwrap().map().unwrap();
        for (i, &b) in map.as_slice().iter().enumerate() {
            assert!(
                (b as i16 - 128).abs() <= 2,
                "pixel byte {i} = {b}, expected ~128 for neutral-grey NV24"
            );
        }
    }

    #[test]
    fn cpu_nv12_to_rgb_respects_tagged_bt2020() {
        // Converts a uniform saturated NV12 frame tagged with `enc` on the CPU and returns its first RGB pixel.
        fn decode_tagged(enc: edgefirst_tensor::ColorEncoding) -> [u8; 3] {
            let mut src = TensorDyn::image(
                8,
                4,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            // NV12 8×4: 32-byte Y plane + 16-byte interleaved UV plane (4:2:0).
            assert_eq!(src.shape(), &[6, 8]);
            {
                let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
                let buf = map.as_mut_slice();
                buf[..32].fill(120); // Y
                for px in buf[32..].as_chunks_mut::<2>().0 {
                    px[0] = 180; // U / Cb
                    px[1] = 64; // V / Cr
                }
            }
            src.set_colorimetry(Some(
                edgefirst_tensor::Colorimetry::default()
                    .with_encoding(enc)
                    .with_range(edgefirst_tensor::ColorRange::Limited),
            ));
            let dst = TensorDyn::image(
                8,
                4,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
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
            let map = dst.as_typed::<u8>().unwrap().map().unwrap();
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
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_to_grey_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Grey,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        compare_images(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_nv12_to_yuyv_cpu() {
        let file = edgefirst_bench::testdata::read("zidane.nv12").to_vec();
        let src = TensorDyn::image(
            1280,
            720,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_typed::<u8>().unwrap().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)]
            .copy_from_slice(&file);

        let dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Yuyv,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

        compare_images_convert_to_rgb(&dst, &target_image, 0.95, function!());
    }

    #[test]
    fn test_cpu_resize_nv16() {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let cpu_nv16_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::Nv16,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let cpu_rgb_dst = TensorDyn::image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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

    #[test]
    fn test_cpu_resize_nv12() {
        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();
        let mk = |fmt| {
            TensorDyn::image(
                640,
                640,
                fmt,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut cpu_converter = CPUProcessor::new();
        let crop = Crop::letterbox([255, 128, 0, 255]);

        let (result, src, cpu_nv12_dst) = convert_img(
            &mut cpu_converter,
            src,
            mk(PixelFormat::Nv12),
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();
        let (result, _src, cpu_rgb_dst) = convert_img(
            &mut cpu_converter,
            src,
            mk(PixelFormat::Rgb),
            Rotation::None,
            Flip::None,
            crop,
        );
        result.unwrap();
        compare_images_convert_to_rgb(&cpu_nv12_dst, &cpu_rgb_dst, 0.99, function!());
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, Error> {
        let src = TensorDyn::image(
            width,
            height,
            format,
            DType::U8,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        src.as_typed::<u8>()
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
                PixelFormat::Rgb
                    | PixelFormat::Rgba
                    | PixelFormat::Grey
                    | PixelFormat::PlanarRgb
                    | PixelFormat::PlanarRgba
            ),
            "format must be Rgb or Rgba for comparison"
        );

        // Returns the image's bytes without row padding.
        fn flat(img: &TensorDyn) -> Vec<u8> {
            let t = img.as_typed::<u8>().expect("u8 image");
            let mut out = vec![0u8; t.shape().iter().product::<usize>()];
            t.copy_to_flat(&mut out).expect("compact the padded rows");
            out
        }

        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => image::GrayImage::from_vec(
                img1.width().unwrap() as u32,
                (img1.height().unwrap() * img1.format().unwrap().channels()) as u32,
                flat(img1),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap()
            .convert(),
            PixelFormat::Grey => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                flat(img2),
            )
            .unwrap()
            .convert(),
            PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => image::GrayImage::from_vec(
                img2.width().unwrap() as u32,
                (img2.height().unwrap() * img2.format().unwrap().channels()) as u32,
                flat(img2),
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
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut img_rgb2 = TensorDyn::image(
            img1.width().unwrap(),
            img1.height().unwrap(),
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut __cv = CPUProcessor::default();
        let r1 = __cv.convert(img1, &mut img_rgb1, None);
        let r2 = __cv.convert(img2, &mut img_rgb2, None);
        if r1.is_err() || r2.is_err() {
            // Compares the raw bytes as a greyscale strip.
            let w = img1.width().unwrap() as u32;
            let data1 = img1.as_typed::<u8>().unwrap().map().unwrap().to_vec();
            let data2 = img2.as_typed::<u8>().unwrap().map().unwrap().to_vec();
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
            img_rgb1.as_typed::<u8>().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width().unwrap() as u32,
            img_rgb2.height().unwrap() as u32,
            img_rgb2.as_typed::<u8>().unwrap().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
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
        let img = TensorDyn::image(
            width,
            height,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        assert_eq!(img.width(), Some(width));
        assert_eq!(img.height(), Some(height));
        assert_eq!(img.format().unwrap(), PixelFormat::Nv12);
        // PixelFormat::Nv12 uses shape [H*3/2, W] to store Y plane + UV plane
        assert_eq!(
            img.as_typed::<u8>().unwrap().shape(),
            &[height * 3 / 2, width]
        );
    }

    #[test]
    fn test_nv12_channels() {
        let img = TensorDyn::image(
            640,
            480,
            PixelFormat::Nv12,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
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
        let img = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
        assert_eq!(img.format(), Some(PixelFormat::Rgb));
    }

    #[test]
    fn test_tensordyn_image_planar_rgb() {
        let img = TensorDyn::image(
            640,
            480,
            PixelFormat::PlanarRgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(img.width(), Some(640));
        assert_eq!(img.height(), Some(480));
        assert_eq!(img.format(), Some(PixelFormat::PlanarRgb));
    }

    #[test]
    fn test_rgb_int8_format() {
        let img = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::I8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        let u8_img = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let i8_img = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgb,
            DType::I8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        assert_eq!(u8_img.dtype(), DType::U8);
        assert_eq!(i8_img.dtype(), DType::I8);
    }

    #[test]
    fn test_pixel_layout_packed_vs_planar() {
        assert_eq!(PixelFormat::Rgb.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Rgba.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::PlanarRgb.layout(), PixelLayout::Planar);
        assert_eq!(PixelFormat::Nv12.layout(), PixelLayout::SemiPlanar);
    }

    /// A PBO tensor refuses `pin_host` and `sync_for_cpu` with `NotImplemented` and still maps.
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[test]
    fn pbo_reports_why_it_cannot_pin() {
        use edgefirst_tensor::{Error, TensorTrait};

        let converter = ImageProcessor::new().unwrap();
        let is_pbo = converter
            .opengl
            .as_ref()
            .is_some_and(|gl| gl.transfer_backend() == opengl_headless::TransferBackend::Pbo);
        if !is_pbo {
            crate::test_support::report_skip(
                "pbo_reports_why_it_cannot_pin - transfer backend is not PBO",
            );
            return;
        }

        let t = converter
            .create_image(
                64,
                64,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(t.as_typed::<u8>().unwrap().memory(), TensorMemory::Pbo);
        let t = t.as_typed::<u8>().unwrap();

        match t.pin_host(edgefirst_tensor::CpuAccess::ReadWrite) {
            Err(Error::NotImplemented(m)) => {
                assert!(
                    m.contains("glMapBufferRange"),
                    "must name the constraint: {m}"
                );
                assert!(m.contains("map()"), "must point at the supported API: {m}");
            }
            other => panic!("expected an explained NotImplemented, got {other:?}"),
        }
        assert!(matches!(
            t.sync_for_cpu(edgefirst_tensor::CpuAccess::Read),
            Err(Error::NotImplemented(_))
        ));

        let g = t.map().expect("map must work where pin cannot");
        assert!(!g.as_slice().is_empty());
    }

    /// Converts between PBO-backed tensors from `create_image` and compares with the CPU result.
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[test]
    fn test_convert_pbo_to_pbo() {
        let mut converter = ImageProcessor::new().unwrap();

        let is_pbo = converter
            .opengl
            .as_ref()
            .is_some_and(|gl| gl.transfer_backend() == opengl_headless::TransferBackend::Pbo);
        if !is_pbo {
            crate::test_support::report_skip(
                "test_convert_pbo_to_pbo - transfer backend is not PBO",
            );
            return;
        }

        let src_w = 640;
        let src_h = 480;
        let dst_w = 320;
        let dst_h = 240;

        let pbo_src = converter
            .create_image(
                src_w,
                src_h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(
            pbo_src.as_typed::<u8>().unwrap().memory(),
            TensorMemory::Pbo,
            "create_image should produce a PBO tensor"
        );

        let file = edgefirst_bench::testdata::read("zidane.jpg").to_vec();
        let jpeg_src = crate::load_image_test_helper(&file, Some(PixelFormat::Rgba), None).unwrap();

        let mem_src = TensorDyn::image(
            src_w,
            src_h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
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

        {
            let src_data = mem_src.as_typed::<u8>().unwrap().map().unwrap();
            let mut pbo_map = pbo_src.as_typed::<u8>().unwrap().map().unwrap();
            pbo_map.copy_from_slice(&src_data);
        }

        let pbo_dst = converter
            .create_image(
                dst_w,
                dst_h,
                PixelFormat::Rgba,
                DType::U8,
                None,
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(
            pbo_dst.as_typed::<u8>().unwrap().memory(),
            TensorMemory::Pbo
        );

        let mut pbo_dst = pbo_dst;
        let result = converter.convert(&pbo_src, &mut pbo_dst, None);
        result.unwrap();

        let cpu_dst = TensorDyn::image(
            dst_w,
            dst_h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        let converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                force_backend: Some("cpu".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(converter.cpu.is_some());
        assert_eq!(converter.forced_backend, Some(ForcedBackend::Cpu));
    }

    #[test]
    fn test_force_backend_invalid() {
        let result = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                force_backend: Some("invalid".into()),
                ..Default::default()
            },
        );
        assert!(
            matches!(&result, Err(Error::ForcedBackendUnavailable(s)) if s.contains("unknown")),
            "invalid backend value should return ForcedBackendUnavailable error: {result:?}"
        );
    }

    #[test]
    fn test_force_backend_unset() {
        let converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv::default(),
        )
        .unwrap();
        assert!(converter.forced_backend.is_none());
    }

    // ========================================================================
    // Tests for hybrid mask path error handling
    // ========================================================================

    #[test]
    fn test_draw_proto_masks_no_cpu_returns_error() {
        let mut converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_gl: true,
                disable_g2d: true,
                disable_cpu: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(converter.cpu.is_none(), "CPU should be disabled");

        let dst = TensorDyn::image(
            640,
            480,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_dyn = dst;
        let det = [DetectBox {
            bbox: edgefirst_tensor::BoundingBox {
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
        let mut converter = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                force_backend: Some("cpu".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(converter.cpu.is_some());

        let dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst_dyn = dst;
        let det = [DetectBox {
            bbox: edgefirst_tensor::BoundingBox {
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
    // draw_decoded_masks / draw_proto_masks output contract, per backend:
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
    // Every test pre-fills dst with a non-zero pattern.
    // ============================================================

    /// The GL backend's soft declines are off when the processor has no CPU backend.
    #[test]
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    fn soft_declines_are_off_without_a_cpu_fallback() {
        let processor = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_cpu: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(processor.cpu.is_none());
        let Some(gl) = processor.opengl.as_ref() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        assert!(!gl.soft_declines().unwrap());
    }

    /// Soft declines apply only under `ComputeBackend::Auto` with no forced backend and a
    /// CPU backend present.
    #[test]
    fn soft_declines_apply_only_under_auto_with_a_cpu_fallback() {
        let mut processor = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        assert!(processor.soft_declines_apply(true));
        assert!(!processor.soft_declines_apply(false));
        processor.forced_backend = Some(ForcedBackend::G2d);
        assert!(!processor.soft_declines_apply(true));
        processor.forced_backend = None;
        processor.cpu = None;
        assert!(!processor.soft_declines_apply(true));
    }

    /// The processor's G2D backend soft-declines under `ComputeBackend::Auto` and not when
    /// `ComputeBackend::G2d` prefers it.
    #[test]
    #[cfg(target_os = "linux")]
    fn image_processor_g2d_soft_declines_follow_the_backend_mode() {
        if let Some(why) = g2d_skip_reason() {
            crate::test_support::report_skip(&format!("{} - {why}", function!()));
            return;
        }
        let soft = |backend, force: Option<&str>| {
            ImageProcessor::with_config_env(
                ImageProcessorConfig {
                    backend,
                    ..Default::default()
                },
                &ProcessorEnv {
                    force_backend: force.map(str::to_string),
                    ..Default::default()
                },
            )
            .unwrap()
            .g2d
            .as_ref()
            .map(G2DProcessor::soft_declines)
        };
        assert_eq!(soft(ComputeBackend::Auto, None), Some(true));
        assert_eq!(soft(ComputeBackend::G2d, None), Some(false));
        assert_eq!(soft(ComputeBackend::Auto, Some("g2d")), Some(false));
    }

    /// A GL backend whose worker takes no message when its soft declines are set is dropped
    /// and the processor converts on the CPU; under forced GL the error is returned.
    #[test]
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    fn a_gl_backend_that_takes_no_message_is_dropped_outside_forced_gl() {
        let mut processor = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv::default(),
        )
        .unwrap();
        let Some(gl) = processor.opengl.as_mut() else {
            crate::test_support::report_skip(&format!("{} - OpenGL not available", function!()));
            return;
        };
        gl.stop_worker();
        processor.set_gl_soft_declines(true).unwrap();
        assert!(processor.opengl.is_none());
        let src = TensorDyn::image(
            64,
            48,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst = TensorDyn::image(
            32,
            24,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        processor.convert(&src, &mut dst, None).unwrap();

        let mut processor = ImageProcessor::with_config_env(
            ImageProcessorConfig {
                backend: ComputeBackend::OpenGl,
                ..Default::default()
            },
            &ProcessorEnv::default(),
        )
        .unwrap();
        processor.forced_backend = Some(ForcedBackend::OpenGl);
        processor.opengl.as_mut().unwrap().stop_worker();
        assert!(processor.set_gl_soft_declines(false).is_err());
        assert!(processor.opengl.is_some());
    }

    /// `EDGEFIRST_INTERPOLATION` accepts `nearest` and `bilinear` in any case and ignores anything else.
    #[test]
    fn processor_env_reads_the_documented_names() {
        let lookup = |vars: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                vars.iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        let env = ProcessorEnv::from_lookup(lookup(&[
            ("EDGEFIRST_FORCE_BACKEND", "cpu"),
            ("EDGEFIRST_DISABLE_GL", "1"),
            ("EDGEFIRST_DISABLE_G2D", "FALSE"),
            ("EDGEFIRST_DISABLE_CPU", "0"),
            ("EDGEFIRST_INTERPOLATION", "nearest"),
        ]));
        assert_eq!(env.force_backend.as_deref(), Some("cpu"));
        assert!(env.disable_gl);
        assert!(!env.disable_g2d, "\"FALSE\" leaves G2D enabled");
        assert!(!env.disable_cpu, "\"0\" leaves the CPU enabled");
        assert_eq!(env.interpolation.as_deref(), Some("nearest"));

        let env = ProcessorEnv::from_lookup(lookup(&[
            ("EDGEFIRST_DISABLE_G2D", "yes"),
            ("EDGEFIRST_DISABLE_CPU", ""),
            ("EDGEFIRST_DISABLE_OPENGL", "1"),
        ]));
        assert!(env.disable_g2d, "any other value disables");
        assert!(env.disable_cpu, "an empty value disables");
        assert!(!env.disable_gl, "only EDGEFIRST_DISABLE_GL disables GL");
        assert_eq!(env.force_backend, None);
        assert_eq!(env.interpolation, None);
    }

    #[test]
    fn test_interpolation_env_parse() {
        for (value, want) in [
            (Some("nearest"), Some(Interpolation::Nearest)),
            (Some(" BILINEAR "), Some(Interpolation::Bilinear)),
            (Some("lanczos"), None),
            (Some(""), None),
            (None, None),
        ] {
            assert_eq!(
                interpolation_from_env(value),
                want,
                "EDGEFIRST_INTERPOLATION={value:?}"
            );
        }
    }

    /// `ImageProcessorConfig::interpolation` reaches the CPU backend and `EDGEFIRST_INTERPOLATION` overrides it.
    #[test]
    fn test_interpolation_config_and_env_reach_cpu() {
        let mut checker = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8 {
            for x in 0..8 {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255 };
                checker.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let resize = |processor: &mut dyn ImageProcessorTrait| {
            let mut src = TensorDyn::image(
                8,
                8,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            src.as_u8_mut()
                .unwrap()
                .map_mut()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&checker);
            let mut dst = TensorDyn::image(
                12,
                12,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            processor.convert(&src, &mut dst, None).unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        let nearest = resize(&mut CPUProcessor::new_nearest());
        let bilinear = resize(&mut CPUProcessor::new());
        assert_ne!(nearest, bilinear, "the checker must tell the filters apart");
        let config = |interpolation| ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            interpolation,
            ..Default::default()
        };

        let mut p = ImageProcessor::with_config_env(
            config(Interpolation::Nearest),
            &ProcessorEnv::default(),
        )
        .unwrap();
        assert_eq!(resize(&mut p), nearest, "config Nearest");
        p.set_interpolation(Interpolation::Bilinear).unwrap();
        assert_eq!(resize(&mut p), bilinear, "set_interpolation(Bilinear)");

        let mut p = ImageProcessor::with_config_env(
            config(Interpolation::Bilinear),
            &ProcessorEnv {
                interpolation: Some("nearest".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            resize(&mut p),
            nearest,
            "EDGEFIRST_INTERPOLATION pins Nearest"
        );
        p.set_interpolation(Interpolation::Bilinear).unwrap();
        assert_eq!(
            resize(&mut p),
            nearest,
            "the env pin outlives set_interpolation"
        );
    }

    /// A per-convert interpolation overrides the processor's for that convert only; `EDGEFIRST_INTERPOLATION` pins both.
    #[test]
    fn test_per_convert_interpolation_overrides_processor() {
        let mut checker = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8 {
            for x in 0..8 {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255 };
                checker.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let resize = |processor: &mut dyn ImageProcessorTrait, options: Option<ConvertOptions>| {
            let mut src = TensorDyn::image(
                8,
                8,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            src.as_u8_mut()
                .unwrap()
                .map_mut()
                .unwrap()
                .as_mut_slice()
                .copy_from_slice(&checker);
            let mut dst = TensorDyn::image(
                12,
                12,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            processor.convert(&src, &mut dst, options).unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        let nearest = resize(&mut CPUProcessor::new_nearest(), None);
        let bilinear = resize(&mut CPUProcessor::new(), None);
        assert_ne!(nearest, bilinear, "the checker must tell the filters apart");
        let cpu_only = |interpolation: Option<&str>| {
            ImageProcessor::with_config_env(
                ImageProcessorConfig {
                    backend: ComputeBackend::Cpu,
                    ..Default::default()
                },
                &ProcessorEnv {
                    interpolation: interpolation.map(str::to_string),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let model_a = ConvertOptions::new().interpolation(Interpolation::Nearest);
        let model_b = ConvertOptions::new().interpolation(Interpolation::Bicubic);

        let mut p = cpu_only(None);
        assert_eq!(p.interpolation(), Interpolation::Bilinear);
        for _ in 0..2 {
            assert_eq!(resize(&mut p, Some(model_a)), nearest, "model A: nearest");
            assert_eq!(resize(&mut p, Some(model_b)), bilinear, "model B: bicubic");
            assert_eq!(resize(&mut p, None), bilinear, "processor setting kept");
        }
        p.set_interpolation(Interpolation::Nearest).unwrap();
        assert_eq!(resize(&mut p, Some(model_b)), bilinear, "options win");
        assert_eq!(resize(&mut p, None), nearest, "processor fallback");

        let mut spec = PreprocessSpec::new();
        spec.interpolation = Interpolation::Nearest;
        let from_spec = ConvertOptions::from_spec(&spec).unwrap();
        assert_eq!(from_spec.get_interpolation(), Some(Interpolation::Nearest));
        let mut p = cpu_only(None);
        let stretch = from_spec.crop(Crop::default());
        assert_eq!(
            resize(&mut p, Some(stretch)),
            nearest,
            "from_spec carries it"
        );

        let cpu = p.cpu.as_mut().unwrap();
        cpu.set_interpolation(Interpolation::Nearest);
        p.set_interpolation(Interpolation::Bilinear).unwrap();
        assert_eq!(
            resize(p.cpu.as_mut().unwrap(), None),
            nearest,
            "an unchanged mode does not reach the backends"
        );

        let mut p = cpu_only(Some("bilinear"));
        assert_eq!(resize(&mut p, Some(model_a)), bilinear, "the env pin wins");
    }

    /// The threaded GL worker honours a per-convert interpolation and keeps the processor's otherwise.
    #[test]
    #[cfg(feature = "opengl")]
    fn test_per_convert_interpolation_reaches_gl_worker() {
        let mut p = match ImageProcessor::with_config_env(
            ImageProcessorConfig {
                backend: ComputeBackend::OpenGl,
                ..Default::default()
            },
            &ProcessorEnv::default(),
        ) {
            Ok(p) if p.opengl.is_some() => p,
            _ => {
                crate::test_support::report_skip(&format!(
                    "{} — GL backend unavailable",
                    function!()
                ));
                return;
            }
        };
        let (sw, sh, dw, dh) = (40usize, 30usize, 56usize, 42usize);
        let mut rgba = Vec::with_capacity(sw * sh * 4);
        for y in 0..sh {
            for x in 0..sw {
                let v = if (x + y) % 2 == 0 { 16u8 } else { 235 };
                rgba.extend_from_slice(&[v, 128, 255 - v, 255]);
            }
        }
        let mut src = TensorDyn::image(
            sw,
            sh,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        src.as_u8_mut()
            .unwrap()
            .map_mut()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&rgba);
        let gl_convert = |p: &mut ImageProcessor, options: Option<ConvertOptions>| {
            let mut dst = TensorDyn::image(
                dw,
                dh,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let options = p.resolve_interpolation(options.unwrap_or_default());
            p.opengl
                .as_mut()
                .unwrap()
                .convert(&src, &mut dst, Some(options))
                .unwrap();
            dst.as_u8().unwrap().map().unwrap().as_slice().to_vec()
        };
        let nearest = ConvertOptions::new().interpolation(Interpolation::Nearest);
        let per_convert = gl_convert(&mut p, Some(nearest));
        let processor_bilinear = gl_convert(&mut p, None);
        p.set_interpolation(Interpolation::Nearest).unwrap();
        let processor_nearest = gl_convert(&mut p, None);
        assert_ne!(processor_bilinear, processor_nearest);
        assert_eq!(per_convert, processor_nearest, "per-convert Nearest");
        let bilinear = ConvertOptions::new().interpolation(Interpolation::Bilinear);
        assert_eq!(gl_convert(&mut p, Some(bilinear)), processor_bilinear);
    }

    /// Each unsupported interpolation warns at most once per process.
    #[test]
    fn test_interpolation_fallback_warns_once() {
        runnable_interpolation(Interpolation::Area);
        assert!(!warn_first_fallback(Interpolation::Area));
        assert_eq!(
            runnable_interpolation(Interpolation::Area),
            Interpolation::Bilinear
        );
        assert!(!warn_first_fallback(Interpolation::Area));
    }

    /// An `ImageProcessor` built with `EDGEFIRST_FORCE_BACKEND` taken as `value`.
    fn with_force_backend(value: Option<&str>) -> Result<ImageProcessor> {
        ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                force_backend: value.map(str::to_string),
                ..ProcessorEnv::from_process()
            },
        )
    }

    /// Allocates an RGBA image tensor filled with a non-zero pattern.
    fn make_dirty_dst(w: usize, h: usize, mem: Option<TensorMemory>) -> TensorDyn {
        let dst = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            mem,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_typed::<u8>().unwrap();
            let mut map = u8t.map().unwrap();
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = 0xA0u8.wrapping_add((i as u8) & 0x3F);
            }
        }
        dst
    }

    /// Allocates an RGBA background filled with a constant colour.
    fn make_bg(w: usize, h: usize, mem: Option<TensorMemory>, rgba: [u8; 4]) -> TensorDyn {
        let bg = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            mem,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = bg.as_typed::<u8>().unwrap();
            let mut map = u8t.map().unwrap();
            for chunk in map.as_mut_slice().as_chunks_mut::<4>().0 {
                chunk.copy_from_slice(&rgba);
            }
        }
        bg
    }

    fn pixel_at(dst: &TensorDyn, x: usize, y: usize) -> [u8; 4] {
        use edgefirst_tensor::TensorMapTrait;
        let w = dst.width().unwrap();
        let off = (y * w + x) * 4;
        let u8t = dst.as_typed::<u8>().unwrap();
        let map = u8t.map().unwrap();
        let s = map.as_slice();
        [s[off], s[off + 1], s[off + 2], s[off + 3]]
    }

    fn assert_every_pixel_eq(dst: &TensorDyn, expected: [u8; 4], case: &str) {
        use edgefirst_tensor::TensorMapTrait;
        let u8t = dst.as_typed::<u8>().unwrap();
        let map = u8t.map().unwrap();
        for (i, chunk) in map.as_slice().as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                chunk, &expected,
                "{case}: pixel idx {i} = {chunk:?}, expected {expected:?}"
            );
        }
    }

    /// Empty detections and no background: dst is cleared to zero.
    fn scenario_empty_no_bg(processor: &mut ImageProcessor, case: &str) {
        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::default())
            .unwrap_or_else(|e| panic!("{case}/decoded_masks empty+no-bg failed: {e:?}"));
        assert_every_pixel_eq(&dst, [0, 0, 0, 0], &format!("{case}/decoded"));

        let mut dst = make_dirty_dst(64, 64, None);
        let proto = {
            use edgefirst_tensor::{Tensor, TensorDyn};
            // Placeholder coefficients of shape [1, 4].
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

    /// Empty detections with a background: dst equals the background.
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
            // Placeholder coefficients of shape [1, 4].
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

    /// One detection without a background: dst is zero outside the box and mask-coloured inside.
    fn scenario_detect_no_bg(processor: &mut ImageProcessor, case: &str) {
        use edgefirst_tensor::Segmentation;
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
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(seg_arr.view())
                .unwrap()
                .into(),
            xmin: 0.25,
            ymin: 0.25,
            xmax: 0.75,
            ymax: 0.75,
        };

        let mut dst = make_dirty_dst(64, 64, None);
        processor
            .draw_decoded_masks(&mut dst, &[detect], &[seg], MaskOverlay::default())
            .unwrap_or_else(|e| panic!("{case}/decoded_masks detect+no-bg failed: {e:?}"));

        let corner = pixel_at(&dst, 2, 2);
        assert_eq!(
            corner,
            [0, 0, 0, 0],
            "{case}/decoded: corner (2,2) leaked dirty pattern: {corner:?}"
        );
        let center = pixel_at(&dst, 32, 32);
        assert!(
            center != [0, 0, 0, 0],
            "{case}/decoded: center (32,32) was not coloured: {center:?}"
        );
    }

    /// One detection with a background: dst equals the background outside the box and differs inside.
    fn scenario_detect_with_bg(processor: &mut ImageProcessor, case: &str) {
        use edgefirst_tensor::Segmentation;
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
            segmentation: edgefirst_tensor::Tensor::from_arrayview3(seg_arr.view())
                .unwrap()
                .into(),
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

        let corner = pixel_at(&dst, 2, 2);
        assert_eq!(
            corner, bg_color,
            "{case}/decoded: corner (2,2) should show bg {bg_color:?} got {corner:?}"
        );
        let center = pixel_at(&dst, 32, 32);
        assert!(
            center != bg_color,
            "{case}/decoded: center (32,32) should differ from bg {bg_color:?}, got {center:?}"
        );
    }

    /// Runs all four scenarios, skipping when the processor cannot be constructed.
    fn run_all_scenarios(
        force_backend: Option<&'static str>,
        case: &'static str,
        require_dma_for_bg: bool,
    ) {
        if require_dma_for_bg && !crate::test_support::is_dma_available() {
            crate::test_support::report_skip(&format!("{case} — DMA not available on this host"));
            return;
        }
        let processor_result = with_force_backend(force_backend);
        let mut processor = match processor_result {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("{case} — backend init failed: {e:?}"));
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

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ))]
    #[cfg(feature = "opengl")]
    #[test]
    fn test_draw_masks_4_scenarios_opengl() {
        run_all_scenarios(Some("opengl"), "opengl", false);
    }

    /// Forced G2D clears or blits an empty frame and returns `NotImplemented` for detections.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_draw_masks_zero_detection_g2d_forced() {
        if !crate::test_support::is_dma_available() {
            crate::test_support::report_skip("g2d forced — DMA not available on this host");
            return;
        }
        let processor_result = with_force_backend(Some("g2d"));
        let mut processor = match processor_result {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!("g2d forced — init failed: {e:?}"));
                return;
            }
        };

        // Empty detections, no background.
        let mut dst = TensorDyn::image(
            64,
            64,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_typed_mut::<u8>().unwrap();
            let mut map = u8t.map().unwrap();
            map.as_mut_slice().fill(0xBB);
        }
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::default())
            .expect("g2d empty+no-bg");
        assert_every_pixel_eq(&dst, [0, 0, 0, 0], "g2d/case1 cleared");

        // Empty detections with a background.
        let bg_color = [7, 11, 13, 255];
        let bg = {
            let t = TensorDyn::image(
                64,
                64,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            {
                use edgefirst_tensor::TensorMapTrait;
                let u8t = t.as_typed::<u8>().unwrap();
                let mut map = u8t.map().unwrap();
                for chunk in map.as_mut_slice().as_chunks_mut::<4>().0 {
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            use edgefirst_tensor::TensorMapTrait;
            let u8t = dst.as_typed_mut::<u8>().unwrap();
            let mut map = u8t.map().unwrap();
            map.as_mut_slice().fill(0x55);
        }
        processor
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::new().with_background(&bg))
            .expect("g2d empty+bg");
        assert_every_pixel_eq(&dst, bg_color, "g2d/case2 bg blit");

        // A detection returns `NotImplemented`.
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
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
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
        let mut processor = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                force_backend: Some("cpu".into()),
                ..Default::default()
            },
        )
        .unwrap();

        let image = edgefirst_bench::testdata::read("zidane.jpg");
        let src = load_image_test_helper(&image, Some(PixelFormat::Rgba), None).unwrap();

        let mut dst =
            TensorDyn::new(&[640, 640, 3], DType::U8, Some(TensorMemory::Mem), None).unwrap();
        dst.set_format(PixelFormat::Rgb).unwrap();

        processor.convert(&src, &mut dst, None).unwrap();

        assert_eq!(dst.format(), Some(PixelFormat::Rgb));
        assert_eq!(dst.width(), Some(640));
        assert_eq!(dst.height(), Some(640));
    }

    /// Several `ImageProcessor`s on one thread each resize without deadlock or error.
    #[test]
    fn test_multiple_image_processors_same_thread() {
        let mut processors: Vec<ImageProcessor> = (0..4)
            .map(|_| ImageProcessor::new().expect("ImageProcessor::new() failed"))
            .collect();

        for proc in &mut processors {
            let src = proc
                .create_image(
                    128,
                    128,
                    PixelFormat::Rgb,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .expect("create src failed");
            let mut dst = proc
                .create_image(
                    64,
                    64,
                    PixelFormat::Rgb,
                    DType::U8,
                    None,
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .expect("create dst failed");
            proc.convert(&src, &mut dst, None).expect("convert failed");
            assert_eq!(dst.width(), Some(64));
            assert_eq!(dst.height(), Some(64));
        }
    }

    /// `ImageProcessor`s created on separate threads each resize without deadlock or error.
    #[test]
    fn test_multiple_image_processors_separate_threads() {
        use std::sync::mpsc;
        use std::time::Duration;

        if std::env::var_os("EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS").is_some() {
            crate::test_support::report_skip(
                "test_multiple_image_processors_separate_threads — known Vivante \
                 GC7000UL concurrent-EGL-teardown double-free \
                 (EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS set)",
            );
            return;
        }

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
                            .create_image(
                                128,
                                128,
                                PixelFormat::Rgb,
                                DType::U8,
                                None,
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap_or_else(|e| panic!("create src failed on thread {i}: {e}"));
                        let mut dst = proc
                            .create_image(
                                64,
                                64,
                                PixelFormat::Rgb,
                                DType::U8,
                                None,
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap_or_else(|e| panic!("create dst failed on thread {i}: {e}"));
                        proc.convert(&src, &mut dst, None)
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

    /// Four `ImageProcessor`s on separate threads convert concurrently without deadlock.
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

                        barrier.wait();

                        for round in 0..ROUNDS {
                            let src = proc
                                .create_image(
                                    128,
                                    128,
                                    PixelFormat::Rgb,
                                    DType::U8,
                                    None,
                                    edgefirst_tensor::CpuAccess::ReadWrite,
                                )
                                .unwrap_or_else(|e| {
                                    panic!("create src failed on thread {i} round {round}: {e}")
                                });
                            let mut dst = proc
                                .create_image(
                                    64,
                                    64,
                                    PixelFormat::Rgb,
                                    DType::U8,
                                    None,
                                    edgefirst_tensor::CpuAccess::ReadWrite,
                                )
                                .unwrap_or_else(|e| {
                                    panic!("create dst failed on thread {i} round {round}: {e}")
                                });
                            proc.convert(&src, &mut dst, None).unwrap_or_else(|e| {
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

    /// Four processors on four threads convert distinct inputs concurrently; each output matches its sequential oracle.
    #[test]
    fn test_parallel_processors_unique_outputs() {
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        const N: usize = 4;
        const ROUNDS: usize = 25;
        const TIMEOUT: Duration = Duration::from_secs(60);

        if std::env::var_os("EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS").is_some() {
            crate::test_support::report_skip(
                "test_parallel_processors_unique_outputs — known Vivante \
                 GC7000UL concurrent-multi-processor driver abort \
                 (EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS set)",
            );
            return;
        }

        let use_dma = crate::test_support::is_dma_available();
        if !use_dma {
            crate::test_support::report_skip(&format!(
                "{} - DmaBuf sources not exercised, running on Mem: no usable DMA heap",
                function!()
            ));
        }
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
                        let (w, h) = (640usize, 480usize);
                        let mem = if use_dma {
                            Some(TensorMemory::DmaBuf)
                        } else {
                            Some(TensorMemory::Mem)
                        };
                        let src = proc
                            .create_image(
                                w,
                                h,
                                PixelFormat::Nv12,
                                DType::U8,
                                mem,
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap();
                        {
                            let t = src.as_typed::<u8>().unwrap();
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
                                .create_image(
                                    320,
                                    320,
                                    PixelFormat::Rgba,
                                    DType::U8,
                                    mem,
                                    edgefirst_tensor::CpuAccess::ReadWrite,
                                )
                                .unwrap();
                            proc.convert(
                                &src,
                                &mut dst,
                                Some(crate::ConvertOptions::new().crop(lb)),
                            )
                            .unwrap_or_else(|e| panic!("convert failed on thread {i}: {e}"));
                            let t = dst.as_typed::<u8>().unwrap();
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

    /// Four processors on four threads each run 200 NV12 720p to RGB letterbox converts that match their oracle.
    #[test]
    #[ignore = "heavy on-demand GL-parallelism stressor; run explicitly on boards"]
    fn stress_parallel_processors_oracle() {
        use std::sync::{mpsc, Arc, Barrier};
        use std::time::Duration;

        const N: usize = 4;
        const ROUNDS: usize = 200;
        const TIMEOUT: Duration = Duration::from_secs(600);

        let use_dma = crate::test_support::is_dma_available();
        if !use_dma {
            crate::test_support::report_skip(&format!(
                "{} - DmaBuf sources not exercised, running on Mem: no usable DMA heap",
                function!()
            ));
        }
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
                        let mem = if use_dma {
                            Some(TensorMemory::DmaBuf)
                        } else {
                            Some(TensorMemory::Mem)
                        };

                        // Per-thread-distinct synthetic NV12.
                        let src = proc
                            .create_image(
                                w,
                                h,
                                PixelFormat::Nv12,
                                DType::U8,
                                mem,
                                edgefirst_tensor::CpuAccess::ReadWrite,
                            )
                            .unwrap();
                        {
                            let t = src.as_typed::<u8>().unwrap();
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
                                .create_image(
                                    640,
                                    640,
                                    PixelFormat::Rgb,
                                    DType::U8,
                                    mem,
                                    edgefirst_tensor::CpuAccess::ReadWrite,
                                )
                                .unwrap();
                            proc.convert(
                                &src,
                                &mut dst,
                                Some(crate::ConvertOptions::new().crop(lb)),
                            )
                            .unwrap_or_else(|e| panic!("convert failed on thread {i}: {e}"));
                            let t = dst.as_typed::<u8>().unwrap();
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

    /// The auto chain converts `Yuyv` to `Rgb` F32 through the CPU with finite output in `[0, 1]`.
    #[test]
    fn convert_f32_auto_never_errors_non_gl_combo() {
        const W: usize = 64;
        const H: usize = 64;

        // YUYV source, `[Y0, U, Y1, V]` per macropixel.
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for chunk in data.as_chunks_mut::<4>().0 {
                chunk[0] = 128; // Y0
                chunk[1] = 128; // U
                chunk[2] = 160; // Y1
                chunk[3] = 128; // V
            }
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(&src, &mut dst, None);
        assert!(
            result.is_ok(),
            "auto-chain Yuyv→Rgb F32 must not error: {:?}",
            result.err()
        );

        let map = dst.as_typed::<f32>().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), W * H * 3, "unexpected output element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }

        let first_non_zero = floats.iter().find(|&&v| v > 0.01);
        assert!(
            first_non_zero.is_some(),
            "all-zero output detected — CPU path likely did not write to the destination buffer"
        );
        let r0 = floats[0];
        assert!(
            (r0 - 0.502_f32).abs() < 0.05,
            "first pixel R={r0} expected ≈0.502 (Y=128 neutral grey from YUYV source)"
        );
    }

    /// A forced-CPU `Rgba` to `PlanarRgb` F16 convert matches the source within 2^-9.
    #[test]
    #[allow(clippy::needless_update)]
    fn convert_f16_forced_cpu_correct() {
        const W: usize = 16;
        const H: usize = 16;
        const TOL: f32 = 1.0 / 512.0; // 2^-9

        // pixel (y,x): R = 50+x, G = 100+y*8, B = 200
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
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
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();
        proc.convert(&src, &mut dst, None)
            .expect("forced-CPU Rgba→PlanarRgb F16 must not error");

        let src_map = src.as_typed::<u8>().unwrap().map().unwrap();
        let src_bytes = src_map.as_slice();
        let dst_map = dst.as_typed::<half::f16>().unwrap().map().unwrap();
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

                if src_bytes[i * 4] != src_bytes[i * 4 + 1] {
                    assert_ne!(r_got, g_got, "R and G planes must differ at ({x},{y})");
                }
            }
        }
    }

    /// The auto chain converts `Rgba` to `Rgb` F32 with a rotation through the CPU.
    #[test]
    fn convert_f32_with_rotation_falls_back() {
        const W: usize = 16;
        const H: usize = 16;

        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
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

        // The rotated destination is H wide and W high.
        let mut dst = TensorDyn::image(
            H, // dst W = src H
            W, // dst H = src W
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(
            &src,
            &mut dst,
            Some(crate::ConvertOptions::new().rotation(Rotation::Clockwise90)),
        );
        assert!(
            result.is_ok(),
            "auto-chain Rgba→Rgb F32 with Rot90 must not error: {:?}",
            result.err()
        );

        let map = dst.as_typed::<f32>().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), H * W * 3, "unexpected output element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
    }

    /// The OpenGL and CPU backends agree within 2^-8 on an `Rgba` to `PlanarRgb` F16 convert.
    #[test]
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows"
        ),
        feature = "opengl"
    ))]
    fn convert_f16_gl_cpu_parity_identity() {
        // Skips when GL is required and the ANGLE dlopen gate is closed.
        #[cfg(target_os = "macos")]
        if std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
            && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none()
        {
            crate::test_support::report_skip(
                "convert_f16_gl_cpu_parity_identity - ANGLE dlopen gate closed",
            );
            return;
        }
        if !is_opengl_available() {
            crate::test_support::report_skip(
                "convert_f16_gl_cpu_parity_identity - OpenGL not available",
            );
            return;
        }

        const W: usize = 32;
        const H: usize = 16;
        const TOL: f32 = 1.0 / 256.0; // 2^-8

        // pixel (y,x): R = 40+x, G = 80+y*10, B = 180
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
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

        let gl_result = {
            let mut gl_proc = match ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::OpenGl,
                ..Default::default()
            }) {
                Ok(p) => p,
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "convert_f16_gl_cpu_parity_identity - GL backend unavailable: {e}"
                    ));
                    return;
                }
            };

            if !gl_proc.supported_render_dtypes().f16 {
                crate::test_support::report_skip(
                    "convert_f16_gl_cpu_parity_identity - F16 render not supported",
                );
                return;
            }

            let Some(mut dst) = crate::test_support::gpu_buffer_allocated(
                TensorDyn::image(
                    W,
                    H,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    Some(TensorMemory::DmaBuf),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                ),
                "zero-copy F16 PlanarRgb destination",
            ) else {
                return;
            };
            if dst.memory() != TensorMemory::DmaBuf {
                crate::test_support::report_skip(&format!(
                    "convert_f16_gl_cpu_parity_identity - the zero-copy F16 \
                     destination fell back to {:?}",
                    dst.memory()
                ));
                return;
            }
            let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
            let fallbacks_before = gl_proc.convert_fallback_count();
            match gl_proc.convert(&src, &mut dst, None) {
                Ok(()) => {
                    if gl_proc.convert_fallback_count() != fallbacks_before {
                        assert!(
                            !require_gl,
                            "HAL_TEST_REQUIRE_GL=1 but \
                             convert_f16_gl_cpu_parity_identity would have skipped \
                             while reporting success: the GL F16 path declined and \
                             the CPU served the convert"
                        );
                        crate::test_support::report_skip(
                            "convert_f16_gl_cpu_parity_identity - the GL F16 path \
                             declined and the CPU served the convert; comparing that \
                             against the forced-CPU result would prove nothing",
                        );
                        return;
                    }
                    dst
                }
                Err(e) => {
                    crate::test_support::report_skip(&format!(
                        "convert_f16_gl_cpu_parity_identity - GL convert failed: {e}"
                    ));
                    return;
                }
            }
        };

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
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            cpu_proc
                .convert(&src, &mut dst, None)
                .expect("forced-CPU Rgba→PlanarRgb F16 must not error");
            dst
        };

        let flat_f16 = |t: &TensorDyn| -> Vec<half::f16> {
            let typed = t.as_typed::<half::f16>().expect("F16 image");
            let mut bytes = vec![0u8; typed.shape().iter().product::<usize>() * 2];
            typed
                .copy_to_flat(&mut bytes)
                .expect("compact any padded rows");
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| half::f16::from_le_bytes(c))
                .collect()
        };
        let gl_halfs = flat_f16(&gl_result);
        let cpu_halfs = flat_f16(&cpu_result);

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
    // supported_render_dtypes() smoke test
    // =========================================================================

    /// `supported_render_dtypes()` returns on a GL backend.
    #[test]
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "opengl"))]
    fn supported_render_dtypes_linux_smoke() {
        let proc = match ImageProcessor::new() {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "supported_render_dtypes_linux_smoke — ImageProcessor::new() failed: {e}"
                ));
                return;
            }
        };
        if proc.opengl.is_none() {
            crate::test_support::report_skip(
                "supported_render_dtypes_linux_smoke — no GL backend on this host",
            );
            return;
        }
        let support = proc.supported_render_dtypes();
        eprintln!(
            "supported_render_dtypes_linux_smoke: f16={} f32={}",
            support.f16, support.f32
        );
    }

    // =========================================================================
    // F16 PlanarRgb with a width not divisible by 4 falls back to CPU
    // =========================================================================

    /// The auto chain converts to a `PlanarRgb` F16 destination whose width is not a multiple of 4.
    #[test]
    fn convert_f16_pbo_non_4_aligned_width_falls_back() {
        const W: usize = 18; // 18 % 4 == 2
        const H: usize = 16;

        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
            let data = map.as_mut_slice();
            for chunk in data.as_chunks_mut::<4>().0 {
                chunk[0] = 128;
                chunk[1] = 64;
                chunk[2] = 200;
                chunk[3] = 255;
            }
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::new().unwrap();
        let result = proc.convert(&src, &mut dst, None);
        assert!(
            result.is_ok(),
            "auto-chain PlanarRgb F16 W%4!=0 must not error (CPU fallback): {:?}",
            result.err()
        );

        let map = dst.as_typed::<half::f16>().unwrap().map().unwrap();
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
    // NV12 → Rgb F32 and NV12 → PlanarRgb F16, forced CPU
    // =========================================================================

    /// The forced CPU backend converts NV12 to `Rgb` F32 with finite output in `[0, 1]`.
    #[test]
    #[allow(clippy::needless_update)]
    fn convert_nv12_to_rgb_f32_cpu() {
        const W: usize = 16;
        const H: usize = 16;

        // NV12 tensor of shape [H*3/2, W], luma 128, chroma 128.
        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
            map.as_mut_slice().fill(128); // Y=128, U=V=128 → neutral grey
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();

        let result = proc.convert(&src, &mut dst, None);
        assert!(
            result.is_ok(),
            "forced-CPU NV12→Rgb F32 must not error: {:?}",
            result.err()
        );

        let map = dst.as_typed::<f32>().unwrap().map().unwrap();
        let floats = map.as_slice();
        assert_eq!(floats.len(), W * H * 3, "unexpected element count");
        for (i, &v) in floats.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "output[{i}]={v} is not finite or not in [0,1]"
            );
        }
        let non_zero = floats.iter().any(|&v| v > 0.01);
        assert!(non_zero, "all-zero output from NV12→Rgb F32 CPU path");
    }

    /// The forced CPU backend converts NV12 to `PlanarRgb` F16.
    #[test]
    #[allow(clippy::needless_update)]
    fn convert_nv12_to_planar_rgb_f16_cpu() {
        const W: usize = 16;
        const H: usize = 16;

        let src = TensorDyn::image(
            W,
            H,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let mut map = src.as_typed::<u8>().unwrap().map().unwrap();
            map.as_mut_slice().fill(128);
        }

        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();

        let mut proc = ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap();

        let result = proc.convert(&src, &mut dst, None);
        assert!(
            result.is_ok(),
            "forced-CPU NV12→PlanarRgb F16 must not error: {:?}",
            result.err()
        );

        let map = dst.as_typed::<half::f16>().unwrap().map().unwrap();
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
    // create_image access and memory
    // =========================================================================

    fn access_test_src(w: usize, h: usize) -> TensorDyn {
        use edgefirst_tensor::TensorMapTrait;
        let t = TensorDyn::image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        for (i, b) in t
            .as_u8()
            .unwrap()
            .map_mut()
            .unwrap()
            .as_mut_slice()
            .iter_mut()
            .enumerate()
        {
            *b = (i % 251) as u8;
        }
        t
    }

    /// With only the CPU backend, `create_image` declares `ReadWrite` for destinations the CPU convert reads back.
    #[test]
    fn create_image_adds_readwrite_for_read_modify_write_passes() {
        use edgefirst_tensor::CpuAccess;
        let mut proc = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_gl: true,
                disable_g2d: true,
                ..Default::default()
            },
        )
        .unwrap();
        let src = access_test_src(128, 96);
        for (format, dtype) in [
            (PixelFormat::Rgb, DType::I8),
            (PixelFormat::Rgba, DType::I8),
            (PixelFormat::Bgra, DType::U8),
        ] {
            for requested in [CpuAccess::None, CpuAccess::Write] {
                let mut dst = proc
                    .create_image(64, 48, format, dtype, None, requested)
                    .unwrap();
                assert_eq!(
                    dst.cpu_access(),
                    CpuAccess::ReadWrite,
                    "{format:?}/{dtype:?} requested {requested:?}"
                );
                proc.convert(&src, &mut dst, None).unwrap_or_else(|e| {
                    panic!("CPU convert into {format:?}/{dtype:?} {requested:?}: {e}")
                });
            }
        }
    }

    /// The CPU convert refuses a `Write`-only destination it would read back.
    #[test]
    fn cpu_convert_refuses_a_write_only_destination_it_reads_back() {
        use edgefirst_tensor::{CpuAccess, Error as TensorError};
        let mut cpu = CPUProcessor::new();
        let src = access_test_src(32, 24);
        for (format, dtype) in [
            (PixelFormat::Rgb, DType::I8),
            (PixelFormat::Bgra, DType::U8),
        ] {
            let mut dst = TensorDyn::image(
                32,
                24,
                format,
                dtype,
                Some(TensorMemory::Mem),
                CpuAccess::Write,
            )
            .unwrap();
            match cpu.convert(&src, &mut dst, None) {
                Err(Error::Tensor(TensorError::CpuAccessDenied {
                    declared: CpuAccess::Write,
                    requested: CpuAccess::ReadWrite,
                })) => {}
                other => panic!("{format:?}/{dtype:?}: {other:?}"),
            }
        }
        let bgra_src = TensorDyn::image(
            32,
            24,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let mut dst = TensorDyn::image(
            32,
            24,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::Write,
        )
        .unwrap();
        cpu.convert(&bgra_src, &mut dst, None).unwrap();
    }

    /// The access `create_image` predicts is the union over the backends that may run.
    #[test]
    fn predicted_cpu_access_unions_the_backends_that_may_run() {
        use edgefirst_tensor::CpuAccess::{self, None as No, Read, ReadWrite, Write};
        let p = predicted_cpu_access;
        let takes = |a: CpuAccess| Some(Some(a));
        let declines = Some(None);
        // A forced backend.
        assert_eq!(
            p(
                Some(ForcedBackend::G2d),
                None,
                takes(ReadWrite),
                takes(Write)
            ),
            ReadWrite
        );
        assert_eq!(
            p(Some(ForcedBackend::G2d), None, declines, takes(Write)),
            No
        );
        assert_eq!(
            p(
                Some(ForcedBackend::OpenGl),
                takes(Write),
                None,
                takes(ReadWrite)
            ),
            Write
        );
        assert_eq!(
            p(Some(ForcedBackend::Cpu), takes(No), None, takes(Write)),
            Write
        );
        // A GL render with no CPU access ends the chain.
        assert_eq!(p(None, takes(No), takes(ReadWrite), takes(ReadWrite)), No);
        // A host-memory readback leaves the fallbacks predicted.
        assert_eq!(p(None, takes(Write), None, takes(ReadWrite)), ReadWrite);
        // A G2D decline leaves the CPU predicted.
        assert_eq!(p(None, declines, takes(No), takes(Write)), Write);
        assert_eq!(p(None, declines, takes(ReadWrite), None), ReadWrite);
        assert_eq!(p(None, None, None, takes(Read)), Read);
        assert_eq!(p(None, None, None, None), No);
    }

    /// A float destination GL cannot render is predicted for the CPU fallback.
    #[cfg(all(target_os = "linux", feature = "opengl"))]
    #[test]
    fn float_destinations_gl_declines_are_predicted_for_the_cpu() {
        use edgefirst_tensor::CpuAccess;
        let Ok(processor) = ImageProcessor::new() else {
            eprintln!("WARNING: SKIPPED: float prediction - no image processor");
            return;
        };
        let Some(gl) = processor.opengl.as_ref() else {
            eprintln!("WARNING: SKIPPED: float prediction - no OpenGL backend");
            return;
        };
        assert_eq!(
            gl.dst_cpu_access(TensorMemory::Mem, PixelFormat::PlanarRgb, DType::F16),
            None
        );
        assert_eq!(
            gl.dst_cpu_access(TensorMemory::DmaBuf, PixelFormat::Grey, DType::F32),
            None
        );
        let image = processor
            .create_image(
                64,
                64,
                PixelFormat::Rgb,
                DType::F32,
                Some(TensorMemory::Mem),
                CpuAccess::None,
            )
            .unwrap();
        assert!(image.cpu_access().writes());
    }

    /// A refused CPU fallback reports the GPU backend's decline when there was one.
    #[test]
    fn a_refused_cpu_fallback_reports_the_gpu_decline() {
        use edgefirst_tensor::{CpuAccess, Error as TensorError};
        let denied = || {
            Error::Tensor(TensorError::CpuAccessDenied {
                declared: CpuAccess::None,
                requested: CpuAccess::Write,
            })
        };
        let declined = || {
            vec![(
                "opengl",
                Error::NotSupported("unsupported source format".into()),
            )]
        };
        let e = with_gpu_declines(denied(), declined());
        assert!(
            matches!(
                &e,
                Error::CpuFallbackDenied {
                    denied: TensorError::CpuAccessDenied { .. },
                    ..
                }
            ),
            "{e:?}"
        );
        let msg = e.to_string();
        assert!(msg.contains("unsupported source format"), "{msg}");
        assert!(msg.contains("CPU access denied"), "{msg}");
        assert!(e.cpu_access_denied().is_some());
        assert!(matches!(
            with_gpu_declines(denied(), Vec::new()),
            Error::Tensor(TensorError::CpuAccessDenied { .. })
        ));
        assert!(matches!(
            with_gpu_declines(Error::NoConverter, declined()),
            Error::NoConverter
        ));
    }

    /// With only the CPU backend, `create_image` widens `Read` to `ReadWrite` and `None` to `Write`.
    #[test]
    fn create_image_adds_write_when_the_cpu_backend_fills_it() {
        use edgefirst_tensor::CpuAccess;
        let mut proc = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv {
                disable_gl: true,
                disable_g2d: true,
                ..Default::default()
            },
        )
        .unwrap();
        let src = access_test_src(128, 96);
        for (requested, declared) in [
            (CpuAccess::Read, CpuAccess::ReadWrite),
            (CpuAccess::None, CpuAccess::Write),
            (CpuAccess::Write, CpuAccess::Write),
            (CpuAccess::ReadWrite, CpuAccess::ReadWrite),
        ] {
            let mut dst = proc
                .create_image(64, 48, PixelFormat::Rgb, DType::U8, None, requested)
                .unwrap();
            assert_eq!(dst.cpu_access(), declared, "requested {requested:?}");
            proc.convert(&src, &mut dst, None)
                .unwrap_or_else(|e| panic!("CPU convert into a {requested:?} request: {e}"));
        }
    }

    /// A CPU convert refuses a destination declared without writes and a source declared without reads.
    #[test]
    fn cpu_convert_refuses_tensors_outside_their_declaration() {
        use edgefirst_tensor::{CpuAccess, Error as TensorError};
        let mut cpu = CPUProcessor::new();
        let src = access_test_src(32, 24);
        for declared in [CpuAccess::Read, CpuAccess::None] {
            let mut dst = TensorDyn::image(
                32,
                24,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                declared,
            )
            .unwrap();
            match cpu.convert(&src, &mut dst, None) {
                Err(Error::Tensor(TensorError::CpuAccessDenied {
                    declared: d,
                    requested: CpuAccess::Write,
                })) => assert_eq!(d, declared),
                other => panic!("convert into a {declared:?} dst: {other:?}"),
            }
        }
        let write_only = TensorDyn::image(
            32,
            24,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::Write,
        )
        .unwrap();
        let mut dst = TensorDyn::image(
            32,
            24,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let err = cpu.convert(&write_only, &mut dst, None).unwrap_err();
        assert!(
            matches!(
                err,
                Error::Tensor(TensorError::CpuAccessDenied {
                    declared: CpuAccess::Write,
                    requested: CpuAccess::Read,
                })
            ),
            "{err:?}"
        );
        assert!(err
            .to_string()
            .contains("declare CpuAccess::ReadWrite (the default)"));
    }

    /// The CPU mask draw refuses a `Write`-only destination.
    #[test]
    fn cpu_draw_requires_readwrite_on_the_destination() {
        use edgefirst_tensor::{CpuAccess, Error as TensorError};
        let mut cpu = CPUProcessor::new();
        let mut dst = TensorDyn::image(
            32,
            24,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::Write,
        )
        .unwrap();
        let err = cpu
            .draw_decoded_masks(&mut dst, &[], &[], MaskOverlay::default())
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Tensor(TensorError::CpuAccessDenied {
                    declared: CpuAccess::Write,
                    requested: CpuAccess::ReadWrite,
                })
            ),
            "{err:?}"
        );
    }

    /// `create_image_desc` matches `create_image` without a compression request and counts a `Compression::Any` fallback.
    #[test]
    fn create_image_desc_negotiates_and_counts_fallbacks() {
        use edgefirst_tensor::{CpuAccess, ImageDesc};
        let proc = ImageProcessor::new().unwrap();

        let desc =
            ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8).with_access(CpuAccess::ReadWrite);
        let plain = proc.create_image_desc(&desc).unwrap();
        let classic = proc
            .create_image(
                64,
                64,
                PixelFormat::Rgba,
                DType::U8,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(plain.memory(), classic.memory(), "same negotiation path");
        assert_eq!(plain.compression(), None);

        #[cfg(not(target_os = "android"))]
        {
            use edgefirst_tensor::Compression;
            let before = proc.compression_fallback_count();
            let desc = ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8)
                .with_access(CpuAccess::None)
                .with_compression(Compression::Any);
            let t = proc.create_image_desc(&desc).unwrap();
            assert_eq!(t.compression(), None, "no vendor tile scheme off-Android");
            assert!(
                proc.compression_fallback_count() > before,
                "Any resolving linear must count"
            );
        }
    }

    /// A contiguous request through `create_image_desc` yields a CMA DMA-BUF or an error.
    #[test]
    #[cfg(target_os = "linux")]
    fn create_image_desc_contiguous_is_cma_dma_or_an_error() {
        use edgefirst_tensor::{Contiguity, ImageDesc, TensorMemory};
        let proc = ImageProcessor::new().unwrap();
        let desc = ImageDesc::new(64, 64, PixelFormat::Rgba, DType::U8).with_contiguous(true);
        match proc.create_image_desc(&desc) {
            Ok(t) => {
                assert_eq!(t.memory(), TensorMemory::DmaBuf);
                assert_eq!(t.contiguity(), Contiguity::Contiguous);
            }
            Err(e) => {
                assert!(e.to_string().contains("linux,cma"), "{e}");
                crate::test_support::report_skip(&format!(
                    "{} - no CMA heap, the contiguous DMA-BUF arm did not run: {e}",
                    function!()
                ));
            }
        }
    }

    /// `create_image` with `TensorMemory::DmaBuf` and `DType::F32` returns `NotSupported`.
    #[test]
    #[cfg(target_os = "linux")]
    fn create_image_f32_dma_rejected() {
        let proc = ImageProcessor::new().unwrap();
        let result = proc.create_image(
            64,
            64,
            PixelFormat::Rgb,
            DType::F32,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::ReadWrite,
        );
        assert!(
            result.is_err(),
            "create_image(F32, Dma) must fail — no DRM fourcc for f32"
        );
    }

    /// `import_image` stores the supplied `Option<Colorimetry>` on the returned `TensorDyn`.
    #[test]
    #[cfg(target_os = "linux")]
    fn import_image_carries_colorimetry() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry, TensorMemory};

        let expected = Colorimetry::default()
            .with_encoding(ColorEncoding::Bt709)
            .with_range(ColorRange::Limited);

        if !is_dma_available() {
            // Without DMA, checks the colorimetry accessors on a plain `TensorDyn`.
            let mut t = TensorDyn::image(
                8,
                8,
                PixelFormat::Rgba,
                DType::U8,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .expect("alloc");
            assert_eq!(t.colorimetry(), None, "colorimetry must start as None");
            t.set_colorimetry(Some(expected));
            assert_eq!(
                t.colorimetry(),
                Some(expected),
                "set_colorimetry must round-trip"
            );
            crate::test_support::report_skip("import_image_carries_colorimetry (DMA unavailable); storage contract verified via TensorDyn");
            return;
        }

        use edgefirst_tensor::{PlaneDescriptor, Tensor};

        let rgba_bytes = 64 * 64 * 4; // 64×64 RGBA8
        let dma_tensor = Tensor::<u8>::new(
            &[rgba_bytes],
            Some(TensorMemory::DmaBuf),
            Some("import_test"),
        )
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

    fn bgra_gradient_source(w: usize, h: usize, memory: Option<TensorMemory>) -> TensorDyn {
        let src = TensorDyn::image(
            w,
            h,
            PixelFormat::Bgra,
            DType::U8,
            memory,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = src.as_u8().unwrap();
            let stride = t.effective_row_stride().unwrap();
            let mut map = t.map().unwrap();
            for y in 0..h {
                for x in 0..w {
                    map.as_mut_slice()[y * stride + x * 4..][..4].copy_from_slice(&[
                        (200 - (x + y) * 150 / (w + h)) as u8,
                        (40 + y * 170 / h) as u8,
                        (40 + x * 170 / w) as u8,
                        255,
                    ]);
                }
            }
        }
        src
    }

    fn bgra_test_bytes(t: &TensorDyn) -> Vec<u8> {
        let typed = t.as_u8().unwrap();
        let mut out = vec![0u8; typed.shape().iter().product::<usize>()];
        typed.copy_to_flat(&mut out).unwrap();
        out
    }

    #[test]
    fn test_bgra_source_through_the_image_processor() {
        let mut processor = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv::default(),
        )
        .unwrap();
        let src = bgra_gradient_source(640, 480, Some(TensorMemory::Mem));
        for format in [PixelFormat::Rgba, PixelFormat::Nv12] {
            let new_dst = || {
                TensorDyn::image(
                    320,
                    240,
                    format,
                    DType::U8,
                    Some(TensorMemory::Mem),
                    edgefirst_tensor::CpuAccess::ReadWrite,
                )
                .unwrap()
            };
            let mut got = new_dst();
            let mut want = new_dst();
            processor
                .convert(&src, &mut got, None)
                .unwrap_or_else(|e| panic!("{format:?}: ImageProcessor convert: {e}"));
            CPUProcessor::new()
                .convert(&src, &mut want, None)
                .unwrap_or_else(|e| panic!("{format:?}: CPU convert: {e}"));
            assert_eq!(
                bgra_test_bytes(&got),
                bgra_test_bytes(&want),
                "{format:?}: ImageProcessor differs from the CPU"
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_bgra_iosurface_through_the_image_processor_macos() {
        use crate::test_support::corevideo_buffer;
        let (w, h) = (1000usize, 562usize);
        let buf = corevideo_buffer(w, h, b"BGRA");
        buf.fill_rows(|_, row, bytes| {
            for (x, px) in bytes[..w * 4].chunks_exact_mut(4).enumerate() {
                px.copy_from_slice(&[
                    (200 - (x + row) * 150 / (w + h)) as u8,
                    (40 + row * 170 / h) as u8,
                    (40 + x * 170 / w) as u8,
                    255,
                ]);
            }
        });
        let src = unsafe { TensorDyn::from_planar_iosurface(buf.surface(), None) }
            .expect("from_planar_iosurface");
        assert_eq!(src.format(), Some(PixelFormat::Bgra));
        let mut processor = ImageProcessor::with_config_env(
            ImageProcessorConfig::default(),
            &ProcessorEnv::default(),
        )
        .unwrap();

        let new_dst = |w: usize, h: usize, format: PixelFormat, dtype: DType| {
            TensorDyn::image(
                w,
                h,
                format,
                dtype,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap()
        };

        let mut got = new_dst(w, h, PixelFormat::Rgba, DType::U8);
        let mut want = new_dst(w, h, PixelFormat::Rgba, DType::U8);
        processor
            .convert(&src, &mut got, None)
            .expect("ImageProcessor convert to Rgba");
        CPUProcessor::new()
            .convert(&src, &mut want, None)
            .expect("CPU convert to Rgba");
        assert_eq!(
            bgra_test_bytes(&got),
            bgra_test_bytes(&want),
            "Rgba: ImageProcessor differs from the CPU"
        );

        let options = ConvertOptions::new().letterbox([114, 114, 114, 255]);
        let mut got = new_dst(640, 640, PixelFormat::PlanarRgb, DType::F16);
        let mut want = new_dst(640, 640, PixelFormat::PlanarRgb, DType::F16);
        processor
            .convert(&src, &mut got, Some(options))
            .expect("ImageProcessor convert to PlanarRgb F16");
        CPUProcessor::new()
            .convert(&src, &mut want, Some(options))
            .expect("CPU convert to PlanarRgb F16");
        let values = |t: &TensorDyn| -> Vec<u8> {
            let typed = t.as_typed::<half::f16>().unwrap();
            let mut out = vec![0u8; typed.shape().iter().product::<usize>() * 2];
            typed.copy_to_flat(&mut out).unwrap();
            out
        };
        assert_eq!(
            values(&got),
            values(&want),
            "PlanarRgb F16: ImageProcessor differs from the CPU"
        );
    }

    #[test]
    #[cfg(feature = "opengl")]
    fn test_gl_declines_bgra_sources() {
        let mut gl = match GLProcessorThreaded::new(None) {
            Ok(p) => p,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "{} - GL engine init failed ({e:?})",
                    function!()
                ));
                return;
            }
        };
        gl.set_soft_declines(false).unwrap();
        let src = bgra_gradient_source(64, 64, Some(TensorMemory::Mem));
        let mut dst = TensorDyn::image(
            32,
            32,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let result = gl.convert(&src, &mut dst, None);
        assert!(
            matches!(result, Err(Error::NotSupported(_))),
            "GL must refuse a Bgra source, got {result:?}"
        );
    }
}
