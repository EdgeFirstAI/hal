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

The crate defines a `TensorImage` struct that represents an image as a
tensor, along with its format information. It also provides an
`ImageProcessor` struct that manages the conversion process, selecting
the appropriate conversion method based on the available hardware.

## Examples

```rust
# use edgefirst_image::{ImageProcessor, TensorImage, RGBA, RGB, Rotation, Flip, Crop, ImageProcessorTrait};
# fn main() -> Result<(), edgefirst_image::Error> {
let image = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
let img = TensorImage::load(image, Some(RGBA), None)?;
let mut converter = ImageProcessor::new()?;
let mut dst = TensorImage::new(640, 480, RGB, None)?;
converter.convert(&img, &mut dst, Rotation::None, Flip::None, Crop::default())?;
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

use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait as _};
use enum_dispatch::enum_dispatch;
use four_char_code::{four_char_code, FourCharCode};
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

/// Result of rendering a single per-instance grayscale mask.
///
/// Contains the bounding-box region in output image coordinates and the
/// raw uint8 pixel data (RED channel only, 0–255 representing sigmoid output).
#[derive(Debug, Clone)]
pub(crate) struct MaskResult {
    /// X offset of the bbox region in the output image.
    pub(crate) x: usize,
    /// Y offset of the bbox region in the output image.
    pub(crate) y: usize,
    /// Width of the bbox region.
    pub(crate) w: usize,
    /// Height of the bbox region.
    pub(crate) h: usize,
    /// Grayscale pixel data (w * h bytes, row-major).
    pub(crate) pixels: Vec<u8>,
}

/// Region metadata for a single detection within a compact mask atlas.
///
/// The atlas packs padded bounding-box strips vertically.  This struct
/// records where each detection's strip lives in the atlas and how it
/// maps back to the original output coordinate space.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct MaskRegion {
    /// Row offset of this detection's strip in the atlas.
    pub atlas_y_offset: usize,
    /// Left edge of the padded bbox in output image coordinates.
    pub padded_x: usize,
    /// Top edge of the padded bbox in output image coordinates.
    pub padded_y: usize,
    /// Width of the padded bbox.
    pub padded_w: usize,
    /// Height of the padded bbox (= number of atlas rows for this strip).
    pub padded_h: usize,
    /// Original (unpadded) bbox left edge in output image coordinates.
    pub bbox_x: usize,
    /// Original (unpadded) bbox top edge in output image coordinates.
    pub bbox_y: usize,
    /// Original (unpadded) bbox width.
    pub bbox_w: usize,
    /// Original (unpadded) bbox height.
    pub bbox_h: usize,
}

mod cpu;
mod error;
mod g2d;
#[path = "gl/mod.rs"]
mod opengl_headless;

/// 8 bit interleaved YUV422, limited range
pub const YUYV: FourCharCode = four_char_code!("YUYV");
/// 8 bit interleaved YUV422 (VYUY byte order), limited range
pub const VYUY: FourCharCode = four_char_code!("VYUY");
/// 8 bit planar YUV420, limited range
pub const NV12: FourCharCode = four_char_code!("NV12");
/// 8 bit planar YUV422, limited range
pub const NV16: FourCharCode = four_char_code!("NV16");
/// 8 bit RGBA
pub const RGBA: FourCharCode = four_char_code!("RGBA");
/// 8 bit BGRA (byte order: B, G, R, A). Used by Cairo/Wayland (ARGB32 on
/// little-endian).
pub const BGRA: FourCharCode = four_char_code!("BGRA");
/// 8 bit RGB
pub const RGB: FourCharCode = four_char_code!("RGB ");
/// 8 bit grayscale, full range
pub const GREY: FourCharCode = four_char_code!("Y800");

// TODO: planar RGB is 8BPS? https://fourcc.org/8bps/
pub const PLANAR_RGB: FourCharCode = four_char_code!("8BPS");

// TODO: What fourcc code is planar RGBA?
pub const PLANAR_RGBA: FourCharCode = four_char_code!("8BPA");

/// Packed RGB with uint8→int8 XOR 0x80 reinterpretation.
/// The underlying bytes are uint8 with MSB flipped; when cast to i8, values
/// map correctly: uint8 0 → int8 -128, uint8 128 → int8 0, uint8 255 → int8 127.
pub const RGB_INT8: FourCharCode = four_char_code!("RGBi");

/// Planar RGB (channels-first) with uint8→int8 XOR 0x80 reinterpretation.
/// The underlying bytes are uint8 with MSB flipped; when cast to i8, values
/// map correctly: uint8 0 → int8 -128, uint8 128 → int8 0, uint8 255 → int8 127.
/// Tensor shape is `[3, H, W]` (channels-first).
pub const PLANAR_RGB_INT8: FourCharCode = four_char_code!("8BPi");

/// An image represented as a tensor with associated format information.
#[derive(Debug)]
pub struct TensorImage {
    tensor: Tensor<u8>,
    fourcc: FourCharCode,
    is_planar: bool,
    /// Second plane for multiplane NV12/NV16 (separate DMA-BUF allocation).
    chroma: Option<Tensor<u8>>,
}

impl TensorImage {
    /// Creates a new `TensorImage` with the specified width, height, format,
    /// and memory type.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::TensorMemory;
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let img = TensorImage::new(640, 480, RGB, Some(TensorMemory::Mem))?;
    /// assert_eq!(img.width(), 640);
    /// assert_eq!(img.height(), 480);
    /// assert_eq!(img.fourcc(), RGB);
    /// assert!(!img.is_planar());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let channels = fourcc_channels(fourcc)?;
        let is_planar = fourcc_planar(fourcc)?;

        // NV12 is semi-planar with Y plane (W×H) + UV plane (W×H/2)
        // Total bytes = W × H × 1.5. Use shape [H*3/2, W] to encode this.
        if fourcc == NV12 {
            let shape = vec![height * 3 / 2, width];
            let tensor = Tensor::new(&shape, memory, None)?;

            return Ok(Self {
                tensor,
                fourcc,
                is_planar,
                chroma: None,
            });
        }

        // NV16 is semi-planar with Y plane (W×H) + UV plane (W×H)
        // Total bytes = W × H × 2. Use shape [H*2, W] to encode this.
        if fourcc == NV16 {
            let shape = vec![height * 2, width];
            let tensor = Tensor::new(&shape, memory, None)?;

            return Ok(Self {
                tensor,
                fourcc,
                is_planar,
                chroma: None,
            });
        }

        if is_planar {
            let shape = vec![channels, height, width];
            let tensor = Tensor::new(&shape, memory, None)?;

            return Ok(Self {
                tensor,
                fourcc,
                is_planar,
                chroma: None,
            });
        }

        let shape = vec![height, width, channels];
        let tensor = Tensor::new(&shape, memory, None)?;

        Ok(Self {
            tensor,
            fourcc,
            is_planar,
            chroma: None,
        })
    }

    /// Creates a new `TensorImage` from an existing tensor and specified
    /// format.
    ///
    /// The required tensor shape depends on the pixel format:
    ///
    /// | Format | Shape | Description |
    /// |--------|-------|-------------|
    /// | `RGB`  | `[H, W, 3]` | 3-channel interleaved |
    /// | `RGBA` | `[H, W, 4]` | 4-channel interleaved |
    /// | `GREY` | `[H, W, 1]` | Single-channel grayscale |
    /// | `YUYV` | `[H, W, 2]` | YUV 4:2:2 interleaved |
    /// | `PLANAR_RGB`  | `[3, H, W]` | Channels-first (3 planes) |
    /// | `PLANAR_RGBA` | `[4, H, W]` | Channels-first (4 planes) |
    /// | `RGB_INT8` | `[H, W, 3]` | Packed RGB, int8 via XOR 0x80 |
    /// | `PLANAR_RGB_INT8` | `[3, H, W]` | Planar RGB, int8 via XOR 0x80 |
    /// | `NV12` | `[H*3/2, W]` | Semi-planar YUV 4:2:0 (2D) |
    /// | `NV16` | `[H*2, W]`   | Semi-planar YUV 4:2:2 (2D) |
    ///
    /// Most formats use a 3D tensor where the channel dimension matches
    /// the format's channel count. The semi-planar formats NV12 and NV16
    /// are special: the Y and UV planes have different heights, so the
    /// data cannot be described as `[H, W, C]`. Instead the contiguous
    /// memory is represented as a 2D tensor whose first dimension encodes
    /// the total byte height (Y rows + UV rows).
    ///
    /// # Examples
    ///
    /// RGB (3D interleaved):
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::Tensor;
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, None)?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.width(), 1280);
    /// assert_eq!(img.height(), 720);
    /// assert_eq!(img.fourcc(), RGB);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// GREY (3D with 1 channel):
    /// ```rust
    /// use edgefirst_image::{GREY, TensorImage};
    /// use edgefirst_tensor::Tensor;
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[480, 640, 1], None, None)?;
    /// let img = TensorImage::from_tensor(tensor, GREY)?;
    /// assert_eq!(img.width(), 640);
    /// assert_eq!(img.height(), 480);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// NV12 (2D semi-planar, height*3/2 rows):
    /// ```rust
    /// use edgefirst_image::{NV12, TensorImage};
    /// use edgefirst_tensor::Tensor;
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// // 1080p NV12: 1080 Y rows + 540 UV rows = 1620 total rows
    /// let tensor = Tensor::new(&[1620, 1920], None, None)?;
    /// let img = TensorImage::from_tensor(tensor, NV12)?;
    /// assert_eq!(img.width(), 1920);
    /// assert_eq!(img.height(), 1080);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_tensor(tensor: Tensor<u8>, fourcc: FourCharCode) -> Result<Self> {
        let shape = tensor.shape();
        let is_planar = fourcc_planar(fourcc)?;

        // NV12/NV16 use 2D shape [H*3/2, W] or [H*2, W] respectively
        if fourcc == NV12 || fourcc == NV16 {
            if shape.len() != 2 {
                return Err(Error::InvalidShape(format!(
                    "Semi-planar format {} requires 2D tensor, got {}: {:?}",
                    fourcc.to_string(),
                    shape.len(),
                    shape
                )));
            }
            return Ok(Self {
                tensor,
                fourcc,
                is_planar,
                chroma: None,
            });
        }

        // All other formats use 3D shape
        if shape.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Tensor shape must have 3 dimensions, got {}: {:?}",
                shape.len(),
                shape
            )));
        }
        let channels = if is_planar { shape[0] } else { shape[2] };

        if fourcc_channels(fourcc)? != channels {
            return Err(Error::InvalidShape(format!(
                "Invalid tensor shape {:?} for format {}",
                shape,
                fourcc.to_string()
            )));
        }

        Ok(Self {
            tensor,
            fourcc,
            is_planar,
            chroma: None,
        })
    }

    /// Creates a multiplane `TensorImage` from separate Y and UV DMA-BUF tensors.
    ///
    /// This constructor supports multi-planar NV12/NV16 formats where the Y (luma)
    /// and UV (chroma) planes are in separate DMA-BUF allocations, each with its
    /// own file descriptor. This is common with V4L2 decoders and ISPs on NXP i.MX
    /// platforms that export `V4L2_PIX_FMT_NV12M`.
    ///
    /// # Arguments
    ///
    /// * `luma` - Y plane tensor with shape `[H, W]`
    /// * `chroma` - UV plane tensor with shape `[H/2, W]` for NV12, `[H, W]` for NV16
    /// * `fourcc` - Pixel format (`NV12` or `NV16`)
    ///
    /// # Returns
    ///
    /// A multiplane `TensorImage` where [`is_multiplane()`](Self::is_multiplane)
    /// returns `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `fourcc` is not `NV12` or `NV16`
    /// - Luma shape is not 2D `[H, W]`
    /// - Chroma shape does not match the expected dimensions for the format
    pub fn from_planes(luma: Tensor<u8>, chroma: Tensor<u8>, fourcc: FourCharCode) -> Result<Self> {
        if fourcc != NV12 && fourcc != NV16 {
            return Err(Error::NotSupported(format!(
                "from_planes() only supports NV12/NV16, got {}",
                fourcc.to_string()
            )));
        }

        let luma_shape = luma.shape();
        if luma_shape.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "Luma tensor must be 2D [H, W], got {}: {:?}",
                luma_shape.len(),
                luma_shape
            )));
        }
        let height = luma_shape[0];
        let width = luma_shape[1];

        let chroma_shape = chroma.shape();
        if chroma_shape.len() != 2 {
            return Err(Error::InvalidShape(format!(
                "Chroma tensor must be 2D, got {}: {:?}",
                chroma_shape.len(),
                chroma_shape
            )));
        }

        let expected_chroma_h = if fourcc == NV12 { height / 2 } else { height };
        if chroma_shape[0] != expected_chroma_h || chroma_shape[1] != width {
            return Err(Error::InvalidShape(format!(
                "Chroma shape {:?} does not match expected [{}, {}] for {} with luma [{}, {}]",
                chroma_shape,
                expected_chroma_h,
                width,
                fourcc.to_string(),
                height,
                width
            )));
        }

        Ok(Self {
            tensor: luma,
            fourcc,
            is_planar: true,
            chroma: Some(chroma),
        })
    }

    /// Returns whether this image uses separate plane allocations (true multiplane).
    ///
    /// When `true`, the luma (Y) and chroma (UV) planes are in separate tensors,
    /// typically backed by different DMA-BUF file descriptors. Use
    /// [`chroma_tensor()`](Self::chroma_tensor) to access the UV plane.
    pub fn is_multiplane(&self) -> bool {
        self.chroma.is_some()
    }

    /// Returns a reference to the chroma plane tensor for multiplane images.
    ///
    /// Returns `None` for single-buffer (contiguous) images.
    pub fn chroma_tensor(&self) -> Option<&Tensor<u8>> {
        self.chroma.as_ref()
    }

    /// Loads an image from the given byte slice, attempting to decode it as
    /// JPEG or PNG format. Exif orientation is supported. The default format is
    /// RGB.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGBA, TensorImage};
    /// use edgefirst_tensor::TensorMemory;
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let jpeg_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.png"));
    /// let img = TensorImage::load(jpeg_bytes, Some(RGBA), Some(TensorMemory::Mem))?;
    /// assert_eq!(img.width(), 1280);
    /// assert_eq!(img.height(), 720);
    /// assert_eq!(img.fourcc(), RGBA);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(
        image: &[u8],
        format: Option<FourCharCode>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        if let Ok(i) = Self::load_jpeg(image, format, memory) {
            return Ok(i);
        }
        if let Ok(i) = Self::load_png(image, format, memory) {
            return Ok(i);
        }

        Err(Error::NotSupported(
            "Could not decode as jpeg or png".to_string(),
        ))
    }

    /// Loads a JPEG image from the given byte slice. Supports EXIF orientation.
    /// The default format is RGB.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::TensorMemory;
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let jpeg_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
    /// let img = TensorImage::load_jpeg(jpeg_bytes, Some(RGB), Some(TensorMemory::Mem))?;
    /// assert_eq!(img.width(), 1280);
    /// assert_eq!(img.height(), 720);
    /// assert_eq!(img.fourcc(), RGB);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_jpeg(
        image: &[u8],
        format: Option<FourCharCode>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let colour = match format {
            Some(RGB) => ColorSpace::RGB,
            Some(RGBA) => ColorSpace::RGBA,
            Some(GREY) => ColorSpace::Luma,
            None => ColorSpace::RGB,
            Some(f) => {
                return Err(Error::NotSupported(format!(
                    "Unsupported image format {}",
                    f.display()
                )));
            }
        };
        let options = DecoderOptions::default().jpeg_set_out_colorspace(colour);
        let mut decoder = JpegDecoder::new_with_options(image, options);
        decoder.decode_headers()?;

        let image_info = decoder.info().ok_or(Error::Internal(
            "JPEG did not return decoded image info".to_string(),
        ))?;

        let converted_color_space = decoder
            .get_output_colorspace()
            .ok_or(Error::Internal("No output colorspace".to_string()))?;

        let converted_color_space = match converted_color_space {
            ColorSpace::RGB => RGB,
            ColorSpace::RGBA => RGBA,
            ColorSpace::Luma => GREY,
            _ => {
                return Err(Error::NotSupported(
                    "Unsupported JPEG decoder output".to_string(),
                ));
            }
        };

        let dest_format = format.unwrap_or(converted_color_space);

        let (rotation, flip) = decoder
            .exif()
            .map(|x| Self::read_exif_orientation(x))
            .unwrap_or((Rotation::None, Flip::None));

        if (rotation, flip) == (Rotation::None, Flip::None) {
            let mut img = Self::new(
                image_info.width as usize,
                image_info.height as usize,
                dest_format,
                memory,
            )?;

            if converted_color_space != dest_format {
                let tmp = Self::new(
                    image_info.width as usize,
                    image_info.height as usize,
                    converted_color_space,
                    Some(TensorMemory::Mem),
                )?;

                decoder.decode_into(&mut tmp.tensor.map()?)?;

                CPUProcessor::convert_format(&tmp, &mut img)?;
                return Ok(img);
            }
            decoder.decode_into(&mut img.tensor.map()?)?;
            return Ok(img);
        }

        let mut tmp = Self::new(
            image_info.width as usize,
            image_info.height as usize,
            dest_format,
            Some(TensorMemory::Mem),
        )?;

        if converted_color_space != dest_format {
            let tmp2 = Self::new(
                image_info.width as usize,
                image_info.height as usize,
                converted_color_space,
                Some(TensorMemory::Mem),
            )?;

            decoder.decode_into(&mut tmp2.tensor.map()?)?;

            CPUProcessor::convert_format(&tmp2, &mut tmp)?;
        } else {
            decoder.decode_into(&mut tmp.tensor.map()?)?;
        }

        rotate_flip_to_tensor_image(&tmp, rotation, flip, memory)
    }

    /// Loads a PNG image from the given byte slice. Supports EXIF orientation.
    /// The default format is RGB.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::TensorMemory;
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let png_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.png"));
    /// let img = TensorImage::load_png(png_bytes, Some(RGB), Some(TensorMemory::Mem))?;
    /// assert_eq!(img.width(), 1280);
    /// assert_eq!(img.height(), 720);
    /// assert_eq!(img.fourcc(), RGB);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_png(
        image: &[u8],
        format: Option<FourCharCode>,
        memory: Option<TensorMemory>,
    ) -> Result<Self> {
        let format = format.unwrap_or(RGB);
        let alpha = match format {
            RGB => false,
            RGBA => true,
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
            .map(|x| Self::read_exif_orientation(x))
            .unwrap_or((Rotation::None, Flip::None));

        if (rotation, flip) == (Rotation::None, Flip::None) {
            let img = Self::new(image_info.width, image_info.height, format, memory)?;
            decoder.decode_into(&mut img.tensor.map()?)?;
            return Ok(img);
        }

        let tmp = Self::new(
            image_info.width,
            image_info.height,
            format,
            Some(TensorMemory::Mem),
        )?;
        decoder.decode_into(&mut tmp.tensor.map()?)?;

        rotate_flip_to_tensor_image(&tmp, rotation, flip, memory)
    }

    fn read_exif_orientation(exif_: &[u8]) -> (Rotation, Flip) {
        let exifreader = exif::Reader::new();
        let Ok(exif_) = exifreader.read_raw(exif_.to_vec()) else {
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

    /// Saves the image as a JPEG file at the specified path with the given
    /// quality. Only RGB and RGBA formats are supported.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::Tensor;
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, None)?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// let save_path = "/tmp/output.jpg";
    /// img.save_jpeg(save_path, 90)?;
    /// # Ok(())
    /// # }
    pub fn save_jpeg(&self, path: &str, quality: u8) -> Result<()> {
        if self.is_planar {
            return Err(Error::NotImplemented(
                "Saving planar images is not supported".to_string(),
            ));
        }

        let colour = if self.fourcc == RGB {
            jpeg_encoder::ColorType::Rgb
        } else if self.fourcc == RGBA {
            jpeg_encoder::ColorType::Rgba
        } else {
            return Err(Error::NotImplemented(
                "Unsupported image format for saving".to_string(),
            ));
        };

        let encoder = jpeg_encoder::Encoder::new_file(path, quality)?;
        let tensor_map = self.tensor.map()?;

        encoder.encode(
            &tensor_map,
            self.width() as u16,
            self.height() as u16,
            colour,
        )?;

        Ok(())
    }

    /// Returns a reference to the underlying tensor.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// let underlying_tensor = img.tensor();
    /// assert_eq!(underlying_tensor.name(), "Tensor");
    /// # Ok(())
    /// # }
    pub fn tensor(&self) -> &Tensor<u8> {
        &self.tensor
    }

    /// Returns the FourCC code representing the image format.
    ///
    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.fourcc(), RGB);
    /// # Ok(())
    /// # }
    pub fn fourcc(&self) -> FourCharCode {
        self.fourcc
    }

    /// Override the FourCC format tag without touching the underlying tensor.
    /// Used internally for int8 ↔ uint8 format aliasing where the pixel layout
    /// is identical and only the interpretation differs.
    pub(crate) fn set_fourcc(&mut self, fourcc: FourCharCode) {
        self.fourcc = fourcc;
    }

    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert!(!img.is_planar());
    /// # Ok(())
    /// # }
    pub fn is_planar(&self) -> bool {
        self.is_planar
    }

    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.width(), 1280);
    /// # Ok(())
    /// # }
    pub fn width(&self) -> usize {
        // NV12/NV16 use 2D shape [H*k, W]
        if self.fourcc == NV12 || self.fourcc == NV16 {
            return self.tensor.shape()[1];
        }
        match self.is_planar {
            true => self.tensor.shape()[2],
            false => self.tensor.shape()[1],
        }
    }

    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.height(), 720);
    /// # Ok(())
    /// # }
    pub fn height(&self) -> usize {
        // Multiplane: luma tensor shape is [H, W] directly
        if self.chroma.is_some() {
            return self.tensor.shape()[0];
        }
        // NV12 uses shape [H*3/2, W], so height = shape[0] * 2 / 3
        if self.fourcc == NV12 {
            return self.tensor.shape()[0] * 2 / 3;
        }
        // NV16 uses shape [H*2, W], so height = shape[0] / 2
        if self.fourcc == NV16 {
            return self.tensor.shape()[0] / 2;
        }
        match self.is_planar {
            true => self.tensor.shape()[1],
            false => self.tensor.shape()[0],
        }
    }

    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.channels(), 3);
    /// # Ok(())
    /// # }
    pub fn channels(&self) -> usize {
        // NV12/NV16 use 2D shape, conceptually have 2 components (Y + interleaved UV)
        if self.fourcc == NV12 || self.fourcc == NV16 {
            return 2;
        }
        match self.is_planar {
            true => self.tensor.shape()[0],
            false => self.tensor.shape()[2],
        }
    }

    /// # Examples
    /// ```rust
    /// use edgefirst_image::{RGB, TensorImage};
    /// use edgefirst_tensor::{Tensor, TensorTrait};
    ///  # fn main() -> Result<(), edgefirst_image::Error> {
    /// let tensor = Tensor::new(&[720, 1280, 3], None, Some("Tensor"))?;
    /// let img = TensorImage::from_tensor(tensor, RGB)?;
    /// assert_eq!(img.row_stride(), 1280*3);
    /// # Ok(())
    /// # }
    pub fn row_stride(&self) -> usize {
        match self.is_planar {
            true => self.width(),
            false => self.width() * self.channels(),
        }
    }

    /// Returns the buffer identity of the underlying tensor.
    pub fn buffer_identity(&self) -> &edgefirst_tensor::BufferIdentity {
        self.tensor.buffer_identity()
    }
}

/// Trait for types that can be used as destination images for conversion.
///
/// This trait abstracts over the difference between owned (`TensorImage`) and
/// borrowed (`TensorImageRef`) image buffers, enabling the same conversion code
/// to work with both.
pub trait TensorImageDst {
    /// Returns a reference to the underlying tensor.
    fn tensor(&self) -> &Tensor<u8>;
    /// Returns a mutable reference to the underlying tensor.
    fn tensor_mut(&mut self) -> &mut Tensor<u8>;
    /// Returns the FourCC code representing the image format.
    fn fourcc(&self) -> FourCharCode;
    /// Returns whether the image is in planar format.
    fn is_planar(&self) -> bool;
    /// Returns the width of the image in pixels.
    fn width(&self) -> usize;
    /// Returns the height of the image in pixels.
    fn height(&self) -> usize;
    /// Returns the number of channels in the image.
    fn channels(&self) -> usize;
    /// Returns the row stride in bytes.
    fn row_stride(&self) -> usize;
    /// Returns the buffer identity of the underlying tensor.
    fn buffer_identity(&self) -> &edgefirst_tensor::BufferIdentity;
    /// Returns whether this image uses separate plane allocations (true multiplane).
    fn is_multiplane(&self) -> bool {
        false
    }
    /// Returns a reference to the chroma plane tensor for multiplane images.
    fn chroma_tensor(&self) -> Option<&Tensor<u8>> {
        None
    }
}

impl TensorImageDst for TensorImage {
    fn tensor(&self) -> &Tensor<u8> {
        &self.tensor
    }

    fn tensor_mut(&mut self) -> &mut Tensor<u8> {
        &mut self.tensor
    }

    fn fourcc(&self) -> FourCharCode {
        self.fourcc
    }

    fn is_planar(&self) -> bool {
        self.is_planar
    }

    fn width(&self) -> usize {
        TensorImage::width(self)
    }

    fn height(&self) -> usize {
        TensorImage::height(self)
    }

    fn channels(&self) -> usize {
        TensorImage::channels(self)
    }

    fn row_stride(&self) -> usize {
        TensorImage::row_stride(self)
    }

    fn buffer_identity(&self) -> &edgefirst_tensor::BufferIdentity {
        TensorImage::buffer_identity(self)
    }

    fn is_multiplane(&self) -> bool {
        self.chroma.is_some()
    }

    fn chroma_tensor(&self) -> Option<&Tensor<u8>> {
        self.chroma.as_ref()
    }
}

/// A borrowed view of an image tensor for zero-copy preprocessing.
///
/// `TensorImageRef` wraps a borrowed `&mut Tensor<u8>` instead of owning it,
/// enabling zero-copy operations where the HAL writes directly into an external
/// tensor (e.g., a model's pre-allocated input buffer).
///
/// # Examples
/// ```rust,ignore
/// // Create a borrowed tensor image wrapping the model's input tensor
/// let mut dst = TensorImageRef::from_borrowed_tensor(
///     model.input_tensor(0),
///     PLANAR_RGB,
/// )?;
///
/// // Preprocess directly into the model's input buffer
/// processor.convert(&src_image, &mut dst, Rotation::None, Flip::None, Crop::default())?;
///
/// // Run inference - no copy needed!
/// model.run()?;
/// ```
#[derive(Debug)]
pub struct TensorImageRef<'a> {
    pub(crate) tensor: &'a mut Tensor<u8>,
    fourcc: FourCharCode,
    is_planar: bool,
}

impl<'a> TensorImageRef<'a> {
    /// Creates a `TensorImageRef` from a borrowed tensor reference.
    ///
    /// The tensor shape must match the expected format:
    /// - For planar formats (e.g., PLANAR_RGB): shape is `[channels, height,
    ///   width]`
    /// - For interleaved formats (e.g., RGB, RGBA): shape is `[height, width,
    ///   channels]`
    ///
    /// # Arguments
    /// * `tensor` - A mutable reference to the tensor to wrap
    /// * `fourcc` - The pixel format of the image
    ///
    /// # Returns
    /// A `Result` containing the `TensorImageRef` or an error if the tensor
    /// shape doesn't match the expected format.
    pub fn from_borrowed_tensor(tensor: &'a mut Tensor<u8>, fourcc: FourCharCode) -> Result<Self> {
        let shape = tensor.shape();
        let is_planar = fourcc_planar(fourcc)?;

        // NV12/NV16 use 2D shape [H*3/2, W] or [H*2, W] respectively
        if fourcc == NV12 || fourcc == NV16 {
            if shape.len() != 2 {
                return Err(Error::InvalidShape(format!(
                    "Semi-planar format {} requires 2D tensor, got {}: {:?}",
                    fourcc.to_string(),
                    shape.len(),
                    shape
                )));
            }
            return Ok(Self {
                tensor,
                fourcc,
                is_planar,
            });
        }

        // All other formats use 3D shape
        if shape.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "Tensor shape must have 3 dimensions, got {}: {:?}",
                shape.len(),
                shape
            )));
        }
        let channels = if is_planar { shape[0] } else { shape[2] };

        if fourcc_channels(fourcc)? != channels {
            return Err(Error::InvalidShape(format!(
                "Invalid tensor shape {:?} for format {}",
                shape,
                fourcc.to_string()
            )));
        }

        Ok(Self {
            tensor,
            fourcc,
            is_planar,
        })
    }

    /// Returns a reference to the underlying tensor.
    pub fn tensor(&self) -> &Tensor<u8> {
        self.tensor
    }

    /// Returns the FourCC code representing the image format.
    pub fn fourcc(&self) -> FourCharCode {
        self.fourcc
    }

    /// Returns whether the image is in planar format.
    pub fn is_planar(&self) -> bool {
        self.is_planar
    }

    /// Returns the width of the image in pixels.
    pub fn width(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[2],
            false => self.tensor.shape()[1],
        }
    }

    /// Returns the height of the image in pixels.
    pub fn height(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[1],
            false => self.tensor.shape()[0],
        }
    }

    /// Returns the number of channels in the image.
    pub fn channels(&self) -> usize {
        match self.is_planar {
            true => self.tensor.shape()[0],
            false => self.tensor.shape()[2],
        }
    }

    /// Returns the row stride in bytes.
    pub fn row_stride(&self) -> usize {
        match self.is_planar {
            true => self.width(),
            false => self.width() * self.channels(),
        }
    }
}

impl TensorImageDst for TensorImageRef<'_> {
    fn tensor(&self) -> &Tensor<u8> {
        self.tensor
    }

    fn tensor_mut(&mut self) -> &mut Tensor<u8> {
        self.tensor
    }

    fn fourcc(&self) -> FourCharCode {
        self.fourcc
    }

    fn is_planar(&self) -> bool {
        self.is_planar
    }

    fn width(&self) -> usize {
        TensorImageRef::width(self)
    }

    fn height(&self) -> usize {
        TensorImageRef::height(self)
    }

    fn channels(&self) -> usize {
        TensorImageRef::channels(self)
    }

    fn row_stride(&self) -> usize {
        TensorImageRef::row_stride(self)
    }

    fn buffer_identity(&self) -> &edgefirst_tensor::BufferIdentity {
        self.tensor.buffer_identity()
    }
}

/// Flips the image, and the rotates it.
fn rotate_flip_to_tensor_image(
    src: &TensorImage,
    rotation: Rotation,
    flip: Flip,
    memory: Option<TensorMemory>,
) -> Result<TensorImage, Error> {
    let src_map = src.tensor.map()?;
    let dst = match rotation {
        Rotation::None | Rotation::Rotate180 => {
            TensorImage::new(src.width(), src.height(), src.fourcc(), memory)?
        }
        Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
            TensorImage::new(src.height(), src.width(), src.fourcc(), memory)?
        }
    };

    let mut dst_map = dst.tensor.map()?;

    CPUProcessor::flip_rotate_ndarray(&src_map, &mut dst_map, &dst, rotation, flip)?;

    Ok(dst)
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

    // Checks if the crop rectangles are valid for the given source and
    // destination images.
    pub fn check_crop(&self, src: &TensorImage, dst: &TensorImage) -> Result<(), Error> {
        let src = self.src_rect.is_none_or(|x| x.check_rect(src));
        let dst = self.dst_rect.is_none_or(|x| x.check_rect(dst));
        match (src, dst) {
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

    // Checks if the crop rectangles are valid for the given source and
    // destination images (using TensorImageRef for destination).
    pub fn check_crop_ref(&self, src: &TensorImage, dst: &TensorImageRef<'_>) -> Result<(), Error> {
        let src = self.src_rect.is_none_or(|x| x.check_rect(src));
        let dst = self.dst_rect.is_none_or(|x| x.check_rect_dst(dst));
        match (src, dst) {
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

    // Checks if the rectangle is valid for the given image.
    pub fn check_rect(&self, image: &TensorImage) -> bool {
        self.left + self.width <= image.width() && self.top + self.height <= image.height()
    }

    // Checks if the rectangle is valid for the given destination image.
    pub fn check_rect_dst<D: TensorImageDst>(&self, image: &D) -> bool {
        self.left + self.width <= image.width() && self.top + self.height <= image.height()
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
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()>;

    /// Converts the source image to a borrowed destination tensor for zero-copy
    /// preprocessing.
    ///
    /// This variant accepts a `TensorImageRef` as the destination, enabling
    /// direct writes into external buffers (e.g., model input tensors) without
    /// intermediate copies.
    ///
    /// # Arguments
    ///
    /// * `src` - The source image to convert from.
    /// * `dst` - A borrowed tensor image wrapping the destination buffer.
    /// * `rotation` - The rotation to apply to the destination image.
    /// * `flip` - Flips the image
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the conversion.
    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
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
    /// - G2D backend: not implemented (returns `NotImplemented`).
    ///
    /// An empty `segmentation` slice is valid — only bounding boxes are drawn.
    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
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
    /// `detect` slice is valid and returns immediately after drawing nothing.
    ///
    /// # Format requirements
    ///
    /// Same as [`draw_masks`](Self::draw_masks). G2D returns `NotImplemented`.
    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> Result<()>;

    /// Decode masks into a compact atlas buffer.
    ///
    /// Used internally by the Python/C `decode_masks` APIs. The atlas is a
    /// compact vertical strip where each detection occupies a strip sized to
    /// its padded bounding box (not the full output resolution).
    ///
    /// `output_width` and `output_height` define the coordinate space for
    /// interpreting bounding boxes — individual mask regions are bbox-sized.
    /// Mask pixels are binary: `255` = presence, `0` = background.
    ///
    /// Returns `(atlas_pixels, regions)` where `regions` describes each
    /// detection's location and bbox within the atlas.
    ///
    /// G2D backend returns `NotImplemented`.
    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<(Vec<u8>, Vec<MaskRegion>)>;

    /// Sets the colors used for rendering segmentation masks. Up to 17 colors
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
    /// processor initializes that backend plus CPU as a fallback chain.
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
    /// # use edgefirst_image::{ImageProcessor, TensorImage, RGBA, RGB, Rotation, Flip, Crop, ImageProcessorTrait};
    /// # fn main() -> Result<(), edgefirst_image::Error> {
    /// let image = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/zidane.jpg"));
    /// let img = TensorImage::load(image, Some(RGBA), None)?;
    /// let mut converter = ImageProcessor::new()?;
    /// let mut dst = TensorImage::new(640, 480, RGB, None)?;
    /// converter.convert(&img, &mut dst, Rotation::None, Flip::None, Crop::default())?;
    /// # Ok(())
    /// # }
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

    /// Create a `TensorImage` with the best available memory backend.
    ///
    /// Priority: DMA-buf → PBO → system memory.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `fourcc` - Pixel format as a FourCC code
    ///
    /// # Returns
    ///
    /// A `TensorImage` backed by the highest-performance memory type
    /// available on this system.
    ///
    /// # Errors
    ///
    /// Returns an error if all allocation strategies fail.
    pub fn create_image(
        &self,
        width: usize,
        height: usize,
        fourcc: four_char_code::FourCharCode,
    ) -> Result<TensorImage> {
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
                if let Ok(img) = TensorImage::new(
                    width,
                    height,
                    fourcc,
                    Some(edgefirst_tensor::TensorMemory::Dma),
                ) {
                    return Ok(img);
                }
            }
        }

        // Try PBO (if GL available)
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(gl) = &self.opengl {
            match gl.create_pbo_image(width, height, fourcc) {
                Ok(img) => return Ok(img),
                Err(e) => log::debug!("PBO image creation failed, falling back to Mem: {e:?}"),
            }
        }

        // Fallback to Mem
        TensorImage::new(
            width,
            height,
            fourcc,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
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
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let start = Instant::now();

        if let Some(ref forced) = self.forced_backend {
            log::trace!("ImageProcessor::convert: forced backend = {forced:?}");
        } else {
            log::trace!("ImageProcessor::convert: auto backend selection");
        }

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.convert(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => {
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        return g2d.convert(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.convert(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // ── Existing fallback chain ──────────────────────────────────
        #[cfg(target_os = "linux")]
        if let Some(g2d) = self.g2d.as_mut() {
            log::trace!("image started with g2d in {:?}", start.elapsed());
            match g2d.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!("image converted with g2d in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::debug!("G2D conversion not supported, falling back: {e:?}")
                }
            }
        }

        // if the image is just a copy without an resizing, the send it to the CPU and
        // skip OpenGL
        let src_shape = match crop.src_rect {
            Some(s) => (s.width, s.height),
            None => (src.width(), src.height()),
        };
        let dst_shape = match crop.dst_rect {
            Some(d) => (d.width, d.height),
            None => (dst.width(), dst.height()),
        };

        // TODO: Check if still use CPU when rotation or flip is enabled
        if src_shape == dst_shape && flip == Flip::None && rotation == Rotation::None {
            if let Some(cpu) = self.cpu.as_mut() {
                match cpu.convert(src, dst, rotation, flip, crop) {
                    Ok(_) => {
                        log::trace!("image converted with cpu in {:?}", start.elapsed());
                        return Ok(());
                    }
                    Err(e) => {
                        log::trace!("image didn't convert with cpu: {e:?}");
                        return Err(e);
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            log::trace!("image started with opengl in {:?}", start.elapsed());
            match opengl.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!("image converted with opengl in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::debug!("OpenGL conversion not supported, falling back to CPU: {e:?}")
                }
            }
        }
        log::trace!("image started with cpu in {:?}", start.elapsed());
        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.convert(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!("image converted with cpu in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("image didn't convert with cpu: {e:?}");
                    return Err(e);
                }
            }
        }
        Err(Error::NoConverter)
    }

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let start = Instant::now();

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.convert_ref(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => {
                    #[cfg(target_os = "linux")]
                    if let Some(g2d) = self.g2d.as_mut() {
                        return g2d.convert_ref(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("g2d".into()))
                }
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.convert_ref(src, dst, rotation, flip, crop);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // For TensorImageRef, we prefer CPU since hardware accelerators typically
        // don't support PLANAR_RGB output which is the common model input format.
        // The CPU path uses the generic conversion functions that work with any
        // TensorImageDst implementation.
        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.convert_ref(src, dst, rotation, flip, crop) {
                Ok(_) => {
                    log::trace!("image converted with cpu (ref) in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("image didn't convert with cpu (ref): {e:?}");
                    return Err(e);
                }
            }
        }

        Err(Error::NoConverter)
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> Result<()> {
        let start = Instant::now();

        if detect.is_empty() && segmentation.is_empty() {
            return Ok(());
        }

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.draw_masks(dst, detect, segmentation);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => Err(Error::NotSupported(
                    "g2d does not support draw_masks".into(),
                )),
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_masks(dst, detect, segmentation);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // skip G2D as it doesn't support rendering to image

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            log::trace!("draw_masks started with opengl in {:?}", start.elapsed());
            match opengl.draw_masks(dst, detect, segmentation) {
                Ok(_) => {
                    log::trace!("draw_masks with opengl in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_masks didn't work with opengl: {e:?}")
                }
            }
        }
        log::trace!("draw_masks started with cpu in {:?}", start.elapsed());
        if let Some(cpu) = self.cpu.as_mut() {
            match cpu.draw_masks(dst, detect, segmentation) {
                Ok(_) => {
                    log::trace!("draw_masks with cpu in {:?}", start.elapsed());
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_masks didn't work with cpu: {e:?}");
                    return Err(e);
                }
            }
        }
        Err(Error::NoConverter)
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> Result<()> {
        let start = Instant::now();

        if detect.is_empty() {
            return Ok(());
        }

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.draw_masks_proto(dst, detect, proto_data);
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => Err(Error::NotSupported(
                    "g2d does not support draw_masks_proto".into(),
                )),
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.draw_masks_proto(dst, detect, proto_data);
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        // skip G2D as it doesn't support rendering to image

        // Hybrid path: CPU materialize + GL overlay (benchmarked faster than
        // full-GPU draw_masks_proto on all tested platforms: 27× on imx8mp,
        // 4× on imx95, 2.5× on rpi5, 1.6× on x86).
        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        if let Some(opengl) = self.opengl.as_mut() {
            let Some(cpu) = self.cpu.as_ref() else {
                return Err(Error::Internal(
                    "draw_masks_proto requires CPU backend for hybrid path".into(),
                ));
            };
            log::trace!(
                "draw_masks_proto started with hybrid (cpu+opengl) in {:?}",
                start.elapsed()
            );
            let segmentation = cpu.materialize_segmentations(detect, proto_data)?;
            match opengl.draw_masks(dst, detect, &segmentation) {
                Ok(_) => {
                    log::trace!(
                        "draw_masks_proto with hybrid (cpu+opengl) in {:?}",
                        start.elapsed()
                    );
                    return Ok(());
                }
                Err(e) => {
                    log::trace!("draw_masks_proto hybrid path failed, falling back to cpu: {e:?}");
                }
            }
        }

        // CPU-only fallback (no OpenGL, or hybrid GL overlay failed)
        let Some(cpu) = self.cpu.as_mut() else {
            return Err(Error::Internal(
                "draw_masks_proto requires CPU backend for fallback path".into(),
            ));
        };
        log::trace!("draw_masks_proto started with cpu in {:?}", start.elapsed());
        cpu.draw_masks_proto(dst, detect, proto_data)
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

    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<(Vec<u8>, Vec<MaskRegion>)> {
        if detect.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        // ── Forced backend: no fallback chain ────────────────────────
        if let Some(forced) = self.forced_backend {
            return match forced {
                ForcedBackend::Cpu => {
                    if let Some(cpu) = self.cpu.as_mut() {
                        return cpu.decode_masks_atlas(
                            detect,
                            proto_data,
                            output_width,
                            output_height,
                        );
                    }
                    Err(Error::ForcedBackendUnavailable("cpu".into()))
                }
                ForcedBackend::G2d => Err(Error::NotSupported(
                    "g2d does not support decode_masks_atlas".into(),
                )),
                ForcedBackend::OpenGl => {
                    #[cfg(target_os = "linux")]
                    #[cfg(feature = "opengl")]
                    if let Some(opengl) = self.opengl.as_mut() {
                        return opengl.decode_masks_atlas(
                            detect,
                            proto_data,
                            output_width,
                            output_height,
                        );
                    }
                    Err(Error::ForcedBackendUnavailable("opengl".into()))
                }
            };
        }

        #[cfg(target_os = "linux")]
        #[cfg(feature = "opengl")]
        {
            let has_opengl = self.opengl.is_some();
            if has_opengl {
                let opengl = self.opengl.as_mut().unwrap();
                match opengl.decode_masks_atlas(detect, proto_data, output_width, output_height) {
                    Ok(r) => return Ok(r),
                    Err(e) => {
                        log::trace!("decode_masks_atlas didn't work with opengl: {e:?}");
                        return Err(e);
                    }
                }
            }
        }
        // CPU fallback: render per-detection masks and pack into compact atlas
        if let Some(cpu) = self.cpu.as_mut() {
            return cpu.decode_masks_atlas(detect, proto_data, output_width, output_height);
        }
        Err(Error::NoConverter)
    }
}

fn fourcc_channels(fourcc: FourCharCode) -> Result<usize> {
    match fourcc {
        RGBA | BGRA => Ok(4), // RGBA/BGRA: 4 channels
        RGB => Ok(3),         // RGB has 3 channels (R, G, B)
        YUYV | VYUY => Ok(2), // YUYV/VYUY has 2 channels (Y and UV)
        GREY => Ok(1),        // Y800 has 1 channel (Y)
        NV12 => Ok(2),        // NV12 has 2 channel. 2nd channel is half empty
        NV16 => Ok(2),        // NV16 has 2 channel. 2nd channel is full size
        PLANAR_RGB => Ok(3),
        PLANAR_RGBA => Ok(4),
        RGB_INT8 => Ok(3),
        PLANAR_RGB_INT8 => Ok(3),
        _ => Err(Error::NotSupported(format!(
            "Unsupported fourcc: {}",
            fourcc.to_string()
        ))),
    }
}

fn fourcc_planar(fourcc: FourCharCode) -> Result<bool> {
    match fourcc {
        RGBA | BGRA => Ok(false), // RGBA/BGRA: 4 channels, interleaved
        RGB => Ok(false),         // RGB has 3 channels (R, G, B)
        YUYV | VYUY => Ok(false), // YUYV/VYUY has 2 channels (Y and UV)
        GREY => Ok(false),        // Y800 has 1 channel (Y)
        NV12 => Ok(true),         // Planar YUV
        NV16 => Ok(true),         // Planar YUV
        PLANAR_RGB => Ok(true),   // Planar RGB
        PLANAR_RGBA => Ok(true),  // Planar RGBA
        RGB_INT8 => Ok(false),
        PLANAR_RGB_INT8 => Ok(true),
        _ => Err(Error::NotSupported(format!(
            "Unsupported fourcc: {}",
            fourcc.to_string()
        ))),
    }
}

/// Returns `true` if the format uses XOR 0x80 int8 reinterpretation.
pub(crate) fn fourcc_is_int8(fourcc: FourCharCode) -> bool {
    matches!(fourcc, RGB_INT8 | PLANAR_RGB_INT8)
}

/// Returns the uint8 equivalent of an int8 format, or the format unchanged.
#[allow(dead_code)] // Will be used by Task 5 (non-DMA int8 path)
pub(crate) fn fourcc_uint8_equivalent(fourcc: FourCharCode) -> FourCharCode {
    match fourcc {
        RGB_INT8 => RGB,
        PLANAR_RGB_INT8 => PLANAR_RGB,
        other => other,
    }
}

/// Returns `true` if the format is packed RGB (3 bytes per pixel, interleaved).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn fourcc_is_packed_rgb(fourcc: FourCharCode) -> bool {
    matches!(fourcc, RGB | RGB_INT8)
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
mod image_tests {
    use super::*;
    use crate::{CPUProcessor, Rotation};
    #[cfg(target_os = "linux")]
    use edgefirst_tensor::is_dma_available;
    use edgefirst_tensor::{TensorMapTrait, TensorMemory};
    use image::buffer::ConvertBuffer;

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
        let src = TensorImage::new(100, 100, RGB, None).unwrap();
        let dst = TensorImage::new(100, 100, RGB, None).unwrap();

        let crop = Crop::new()
            .with_src_rect(Some(Rect::new(50, 50, 60, 60)))
            .with_dst_rect(Some(Rect::new(0, 0, 150, 150)));

        let result = crop.check_crop(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Dest and Src crop invalid")
        ));

        let crop = crop.with_src_rect(Some(Rect::new(0, 0, 10, 10)));
        let result = crop.check_crop(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Dest crop invalid")
        ));

        let crop = crop
            .with_src_rect(Some(Rect::new(50, 50, 60, 60)))
            .with_dst_rect(Some(Rect::new(0, 0, 50, 50)));
        let result = crop.check_crop(&src, &dst);
        assert!(matches!(
            result,
            Err(Error::CropInvalid(e)) if e.starts_with("Src crop invalid")
        ));

        let crop = crop.with_src_rect(Some(Rect::new(50, 50, 50, 50)));

        let result = crop.check_crop(&src, &dst);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_tensor() -> Result<(), Error> {
        let tensor = Tensor::new(&[720, 1280, 4, 1], None, None)?;
        let result = TensorImage::from_tensor(tensor, RGB);
        assert!(matches!(
            result,
            Err(Error::InvalidShape(e)) if e.starts_with("Tensor shape must have 3 dimensions, got")
        ));

        let tensor = Tensor::new(&[720, 1280, 4], None, None)?;
        let result = TensorImage::from_tensor(tensor, RGB);
        assert!(matches!(
            result,
            Err(Error::InvalidShape(e)) if e.starts_with("Invalid tensor shape")
        ));

        Ok(())
    }

    #[test]
    fn test_invalid_image_file() -> Result<(), Error> {
        let result = TensorImage::load(&[123; 5000], None, None);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e == "Could not decode as jpeg or png"));

        Ok(())
    }

    #[test]
    fn test_invalid_jpeg_fourcc() -> Result<(), Error> {
        let result = TensorImage::load(&[123; 5000], Some(YUYV), None);
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
        let img = TensorImage::load_jpeg(file, Some(RGBA), None).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);

        let mut dst = TensorImage::new(640, 360, RGBA, None).unwrap();
        let mut converter = CPUProcessor::new();
        converter
            .convert(&img, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();
        assert_eq!(dst.width(), 640);
        assert_eq!(dst.height(), 360);

        dst.save_jpeg("zidane_resized.jpg", 80).unwrap();

        let file = std::fs::read("zidane_resized.jpg").unwrap();
        let img = TensorImage::load_jpeg(&file, None, None).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 360);
        assert_eq!(img.fourcc(), RGB);
    }

    #[test]
    fn test_from_tensor_planar() -> Result<(), Error> {
        let tensor = Tensor::new(&[3, 720, 1280], None, None)?;
        tensor.map()?.copy_from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.8bps"
        )));
        let planar = TensorImage::from_tensor(tensor, PLANAR_RGB)?;

        let rbga = load_bytes_to_tensor(
            1280,
            720,
            RGBA,
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
    fn test_from_tensor_invalid_fourcc() {
        let tensor = Tensor::new(&[3, 720, 1280], None, None).unwrap();
        let result = TensorImage::from_tensor(tensor, four_char_code!("TEST"));
        matches!(result, Err(Error::NotSupported(e)) if e.starts_with("Unsupported fourcc : TEST"));
    }

    #[test]
    #[should_panic(expected = "Failed to save planar RGB image")]
    fn test_save_planar() {
        let planar_img = load_bytes_to_tensor(
            1280,
            720,
            PLANAR_RGB,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.8bps"
            )),
        )
        .unwrap();

        let save_path = "/tmp/planar_rgb.jpg";
        planar_img
            .save_jpeg(save_path, 90)
            .expect("Failed to save planar RGB image");
    }

    #[test]
    #[should_panic(expected = "Failed to save YUYV image")]
    fn test_save_yuyv() {
        let planar_img = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let save_path = "/tmp/yuyv.jpg";
        planar_img
            .save_jpeg(save_path, 90)
            .expect("Failed to save YUYV image");
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

        let src = TensorImage::new(1280, 720, RGBA, None)?;
        let mut dst = TensorImage::new(640, 360, RGBA, None)?;
        let result = converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop());
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

        Ok(())
    }

    #[test]
    fn test_unsupported_conversion() {
        let src = TensorImage::new(1280, 720, NV12, None).unwrap();
        let mut dst = TensorImage::new(640, 360, NV12, None).unwrap();
        let mut converter = ImageProcessor::new().unwrap();
        let result = converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop());
        log::debug!("result: {:?}", result);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e.starts_with("Conversion from NV12 to NV12")
        ));
    }

    #[test]
    fn test_load_grey() {
        let grey_img = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey.jpg"
            )),
            Some(RGBA),
            None,
        )
        .unwrap();

        let grey_but_rgb_img = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey-rgb.jpg"
            )),
            Some(RGBA),
            None,
        )
        .unwrap();

        compare_images(&grey_img, &grey_but_rgb_img, 0.99, function!());
    }

    #[test]
    fn test_new_nv12() {
        let nv12 = TensorImage::new(1280, 720, NV12, None).unwrap();
        assert_eq!(nv12.height(), 720);
        assert_eq!(nv12.width(), 1280);
        assert_eq!(nv12.fourcc(), NV12);
        assert_eq!(nv12.channels(), 2);
        assert!(nv12.is_planar())
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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut converter_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut converter = ImageProcessor::new().unwrap();
        converter
            .convert(
                &src,
                &mut converter_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        compare_images(&converter_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_crop_skip() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut converter_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut converter = ImageProcessor::new().unwrap();
        let crop = Crop::new()
            .with_src_rect(Some(Rect::new(0, 0, 640, 640)))
            .with_dst_rect(Some(Rect::new(0, 0, 640, 640)));
        converter
            .convert(&src, &mut converter_dst, Rotation::None, Flip::None, crop)
            .unwrap();

        let mut cpu_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(&src, &mut cpu_dst, Rotation::None, Flip::None, crop)
            .unwrap();

        compare_images(&converter_dst, &cpu_dst, 0.99999, function!());
    }

    #[test]
    fn test_invalid_fourcc() {
        let result = TensorImage::new(1280, 720, four_char_code!("TEST"), None);
        assert!(matches!(
            result,
            Err(Error::NotSupported(e)) if e == "Unsupported fourcc: TEST"
        ));
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
        let loaded = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        assert_eq!(loaded.height(), 1280);
        assert_eq!(loaded.width(), 720);

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let cpu_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let (dst_width, dst_height) = (cpu_src.height(), cpu_src.width());

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &cpu_src,
                &mut cpu_dst,
                Rotation::Clockwise90,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        compare_images(&loaded, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_load_png_with_exif() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane_rotated_exif_180.png"
        ))
        .to_vec();
        let loaded = TensorImage::load_png(&file, Some(RGBA), None).unwrap();

        assert_eq!(loaded.height(), 720);
        assert_eq!(loaded.width(), 1280);

        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let cpu_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &cpu_src,
                &mut cpu_dst,
                Rotation::Rotate180,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(
                    &src,
                    &mut gl_dst,
                    Rotation::None,
                    Flip::None,
                    Crop::no_crop(),
                )
                .unwrap();

            compare_images(&gl_dst, &cpu_dst, 0.98, function!());
        }

        drop(gl_dst);
    }

    #[test]
    #[ignore] // Vivante GPU hangs with concurrent EGL contexts on i.MX8MP
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

        let img = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/grey.jpg"
            )),
            Some(GREY),
            None,
        )
        .unwrap();

        let mut gl_dst = TensorImage::new(640, 640, GREY, None).unwrap();
        let mut cpu_dst = TensorImage::new(640, 640, GREY, None).unwrap();

        let mut converter = CPUProcessor::new();

        converter
            .convert(
                &img,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.convert(
            &img,
            &mut gl_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: Some(Rect {
                        left: 0,
                        top: 0,
                        width: 640,
                        height: 360,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

        let mut g2d_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: Some(Rect {
                        left: 0,
                        top: 0,
                        width: 640,
                        height: 360,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 512, 288)),
                    dst_color: None,
                },
            )
            .unwrap();

        let mut g2d_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();
        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 512, 288)),
                    dst_color: None,
                },
            )
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        let mut g2d_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        for rot in [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Rotate180,
            Rotation::CounterClockwise90,
        ] {
            cpu_dst.tensor.map().unwrap().as_mut_slice().fill(114);
            g2d_dst.tensor.map().unwrap().as_mut_slice().fill(114);
            for flip in [Flip::None, Flip::Horizontal, Flip::Vertical] {
                cpu_converter
                    .convert(
                        &src,
                        &mut cpu_dst,
                        Rotation::None,
                        Flip::None,
                        Crop {
                            src_rect: Some(Rect::new(50, 120, 1024, 576)),
                            dst_rect: Some(Rect::new(100, 100, 512, 288)),
                            dst_color: None,
                        },
                    )
                    .unwrap();

                g2d_converter
                    .convert(
                        &src,
                        &mut g2d_dst,
                        Rotation::None,
                        Flip::None,
                        Crop {
                            src_rect: Some(Rect::new(50, 120, 1024, 576)),
                            dst_rect: Some(Rect::new(100, 100, 512, 288)),
                            dst_color: None,
                        },
                    )
                    .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: Some(Rect {
                        left: 320,
                        top: 180,
                        width: 1280 - 320,
                        height: 720 - 180,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        gl_converter
            .convert(
                &src,
                &mut gl_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: Some(Rect {
                        left: 320,
                        top: 180,
                        width: 1280 - 320,
                        height: 720 - 180,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 512, 288)),
                    dst_color: None,
                },
            )
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();
        gl_converter
            .convert(
                &src,
                &mut gl_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 512, 288)),
                    dst_color: None,
                },
            )
            .unwrap();

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
        for m in mem {
            let src = TensorImage::load_jpeg(&file, Some(RGBA), m).unwrap();

            for rot in [
                Rotation::None,
                Rotation::Clockwise90,
                Rotation::Rotate180,
                Rotation::CounterClockwise90,
            ] {
                for flip in [Flip::None, Flip::Horizontal, Flip::Vertical] {
                    let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, m).unwrap();
                    let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, m).unwrap();
                    cpu_dst.tensor.map().unwrap().as_mut_slice().fill(114);
                    gl_dst.tensor.map().unwrap().as_mut_slice().fill(114);
                    cpu_converter
                        .convert(
                            &src,
                            &mut cpu_dst,
                            Rotation::None,
                            Flip::None,
                            Crop {
                                src_rect: Some(Rect::new(50, 120, 1024, 576)),
                                dst_rect: Some(Rect::new(100, 100, 512, 288)),
                                dst_color: None,
                            },
                        )
                        .unwrap();

                    gl_converter
                        .convert(
                            &src,
                            &mut gl_dst,
                            Rotation::None,
                            Flip::None,
                            Crop {
                                src_rect: Some(Rect::new(50, 120, 1024, 576)),
                                dst_rect: Some(Rect::new(100, 100, 512, 288)),
                                dst_color: None,
                            },
                        )
                        .map_err(|e| {
                            log::error!("error mem {m:?} rot {rot:?} error: {e:?}");
                            e
                        })
                        .unwrap();

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

        let unchanged_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();
        let mut src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let (dst_width, dst_height) = match rot {
            Rotation::None | Rotation::Rotate180 => (src.width(), src.height()),
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => (src.height(), src.width()),
        };

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        // After rotating 4 times, the image should be the same as the original

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, Crop::no_crop())
            .unwrap();

        cpu_converter
            .convert(&cpu_dst, &mut src, rot, Flip::None, Crop::no_crop())
            .unwrap();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, Crop::no_crop())
            .unwrap();

        cpu_converter
            .convert(&cpu_dst, &mut src, rot, Flip::None, Crop::no_crop())
            .unwrap();

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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), tensor_memory).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, Crop::no_crop())
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, RGBA, tensor_memory).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        for _ in 0..5 {
            gl_converter
                .convert(&src, &mut gl_dst, rot, Flip::None, Crop::no_crop())
                .unwrap();
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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), Some(TensorMemory::Dma)).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut cpu_dst, rot, Flip::None, Crop::no_crop())
            .unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_converter
            .convert(&src, &mut g2d_dst, rot, Flip::None, Crop::no_crop())
            .unwrap();

        compare_images(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_rgba_to_yuyv_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            RGBA,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let mut dst = TensorImage::new(dst_width, dst_height, YUYV, None).unwrap();

        let mut dst_through_yuyv = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut dst_direct = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();

        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        cpu_converter
            .convert(
                &dst,
                &mut dst_through_yuyv,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        cpu_converter
            .convert(
                &src,
                &mut dst_direct,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        compare_images(&dst_through_yuyv, &dst_direct, 0.98, function!());
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    #[ignore = "opengl doesn't support rendering to YUYV texture"]
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
            RGBA,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 360);

        let mut dst =
            TensorImage::new(dst_width, dst_height, YUYV, Some(TensorMemory::Dma)).unwrap();

        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        gl_converter
            .convert(
                &src,
                &mut dst,
                Rotation::None,
                Flip::None,
                Crop::new()
                    .with_dst_rect(Some(Rect::new(100, 100, 100, 100)))
                    .with_dst_color(Some([255, 255, 255, 255])),
            )
            .unwrap();

        std::fs::write(
            "rgba_to_yuyv_opengl.yuyv",
            dst.tensor().map().unwrap().as_slice(),
        )
        .unwrap();
        let mut cpu_dst =
            TensorImage::new(dst_width, dst_height, YUYV, Some(TensorMemory::Dma)).unwrap();
        CPUProcessor::new()
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
            RGBA,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (1280, 720);

        let mut cpu_dst =
            TensorImage::new(dst_width, dst_height, YUYV, Some(TensorMemory::Dma)).unwrap();

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, YUYV, Some(TensorMemory::Dma)).unwrap();

        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_dst.tensor.map().unwrap().as_mut_slice().fill(128);
        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 2, 2)),
                    dst_color: None,
                },
            )
            .unwrap();

        cpu_dst.tensor.map().unwrap().as_mut_slice().fill(128);
        CPUProcessor::new()
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop {
                    src_rect: None,
                    dst_rect: Some(Rect::new(100, 100, 2, 2)),
                    dst_color: None,
                },
            )
            .unwrap();

        compare_images_convert_to_rgb(&cpu_dst, &g2d_dst, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgba_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.yuyv"
        ))
        .to_vec();
        let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
        let src = TensorImage::new(1280, 720, YUYV, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGB, None).unwrap();
        target_image
            .tensor()
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
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        gl_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut g2d_dst = TensorImage::new(1280, 720, RGB, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut cpu_dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let mut g2d_dst = TensorImage::new(600, 400, YUYV, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_converter
            .convert(
                &src,
                &mut g2d_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        let mut cpu_dst = TensorImage::new(600, 400, YUYV, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // TODO: compare YUYV and YUYV images without having to convert them to RGB
        compare_images_convert_to_rgb(&g2d_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_yuyv_to_rgba_resize_cpu() {
        let src = load_bytes_to_tensor(
            1280,
            720,
            YUYV,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (960, 540);

        let mut dst = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let mut dst_target = TensorImage::new(dst_width, dst_height, RGBA, None).unwrap();
        let src_target = load_bytes_to_tensor(
            1280,
            720,
            RGBA,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )
        .unwrap();
        cpu_converter
            .convert(
                &src_target,
                &mut dst_target,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let mut dst_g2d =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        g2d_converter
            .convert(
                &src,
                &mut dst_g2d,
                Rotation::None,
                Flip::Horizontal,
                Crop {
                    src_rect: Some(Rect {
                        left: 20,
                        top: 15,
                        width: 400,
                        height: 300,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

        let mut dst_cpu =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut dst_cpu,
                Rotation::None,
                Flip::Horizontal,
                Crop {
                    src_rect: Some(Rect {
                        left: 20,
                        top: 15,
                        width: 400,
                        height: 300,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();
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
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )
        .unwrap();

        let (dst_width, dst_height) = (640, 640);

        let mut dst_gl =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        gl_converter
            .convert(
                &src,
                &mut dst_gl,
                Rotation::None,
                Flip::Horizontal,
                Crop {
                    src_rect: Some(Rect {
                        left: 20,
                        top: 15,
                        width: 400,
                        height: 300,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();

        let mut dst_cpu =
            TensorImage::new(dst_width, dst_height, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut dst_cpu,
                Rotation::None,
                Flip::Horizontal,
                Crop {
                    src_rect: Some(Rect {
                        left: 20,
                        top: 15,
                        width: 400,
                        height: 300,
                    }),
                    dst_rect: None,
                    dst_color: None,
                },
            )
            .unwrap();
        compare_images(&dst_gl, &dst_cpu, 0.98, function!());
    }

    #[test]
    fn test_vyuy_to_rgba_cpu() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/camera720p.vyuy"
        ))
        .to_vec();
        let src = TensorImage::new(1280, 720, VYUY, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
        let src = TensorImage::new(1280, 720, VYUY, None).unwrap();
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::new(1280, 720, RGB, None).unwrap();
        target_image
            .tensor()
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
            VYUY,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        match g2d_converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop()) {
            Err(Error::G2D(_)) => {
                eprintln!("SKIPPED: test_vyuy_to_rgba_g2d - G2D does not support VYUY format");
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
            VYUY,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let mut g2d_dst = TensorImage::new(1280, 720, RGB, Some(TensorMemory::Dma)).unwrap();
        let mut g2d_converter = G2DProcessor::new().unwrap();

        match g2d_converter.convert(
            &src,
            &mut g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        ) {
            Err(Error::G2D(_)) => {
                eprintln!("SKIPPED: test_vyuy_to_rgb_g2d - G2D does not support VYUY format");
                return;
            }
            r => r.unwrap(),
        }

        let mut cpu_dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter: CPUProcessor = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

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
            VYUY,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.vyuy"
            )),
        )
        .unwrap();

        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        match gl_converter.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop()) {
            Err(Error::NotSupported(_)) => {
                eprintln!(
                    "SKIPPED: {} - OpenGL does not support VYUY DMA format",
                    function!()
                );
                return;
            }
            r => r.unwrap(),
        }

        let target_image = TensorImage::new(1280, 720, RGBA, None).unwrap();
        target_image
            .tensor()
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
        let src = TensorImage::new(1280, 720, NV12, None).unwrap();
        src.tensor().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)].copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(RGBA),
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
        let src = TensorImage::new(1280, 720, NV12, None).unwrap();
        src.tensor().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)].copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(RGB),
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
        let src = TensorImage::new(1280, 720, NV12, None).unwrap();
        src.tensor().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)].copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, GREY, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(GREY),
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
        let src = TensorImage::new(1280, 720, NV12, None).unwrap();
        src.tensor().map().unwrap().as_mut_slice()[0..(1280 * 720 * 3 / 2)].copy_from_slice(&file);

        let mut dst = TensorImage::new(1280, 720, YUYV, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        let target_image = TensorImage::load_jpeg(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.jpg"
            )),
            Some(RGB),
            None,
        )
        .unwrap();

        compare_images_convert_to_rgb(&dst, &target_image, 0.98, function!());
    }

    #[test]
    fn test_cpu_resize_planar_rgb() {
        let src = TensorImage::new(4, 4, RGBA, None).unwrap();
        #[rustfmt::skip]
        let src_image = [
                    255, 0, 0, 255,     0, 255, 0, 255,     0, 0, 255, 255,     255, 255, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
                    0, 0, 255, 0,       0, 255, 255, 255,   255, 255, 0, 0,     0, 0, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
        ];
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&src_image);

        let mut cpu_dst = TensorImage::new(5, 5, PLANAR_RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
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
            )
            .unwrap();

        #[rustfmt::skip]
        let expected_dst = [
            114, 114, 114, 114, 114,    114, 255, 0, 0, 255,    114, 255, 0, 255, 255,      114, 0, 0, 255, 0,        114, 255, 0, 255, 255,
            114, 114, 114, 114, 114,    114, 0, 255, 0, 255,    114, 0, 0, 0, 0,            114, 0, 255, 255, 0,      114, 0, 0, 0, 0,
            114, 114, 114, 114, 114,    114, 0, 0, 255, 0,      114, 0, 0, 255, 255,        114, 255, 255, 0, 0,      114, 0, 0, 255, 255,
        ];

        assert_eq!(cpu_dst.tensor().map().unwrap().as_slice(), &expected_dst);
    }

    #[test]
    fn test_cpu_resize_planar_rgba() {
        let src = TensorImage::new(4, 4, RGBA, None).unwrap();
        #[rustfmt::skip]
        let src_image = [
                    255, 0, 0, 255,     0, 255, 0, 255,     0, 0, 255, 255,     255, 255, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
                    0, 0, 255, 0,       0, 255, 255, 255,   255, 255, 0, 0,     0, 0, 0, 255,
                    255, 0, 0, 0,       0, 0, 0, 255,       255,  0, 255, 0,    255, 0, 255, 255,
        ];
        src.tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&src_image);

        let mut cpu_dst = TensorImage::new(5, 5, PLANAR_RGBA, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
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
            )
            .unwrap();

        #[rustfmt::skip]
        let expected_dst = [
            114, 114, 114, 114, 114,    114, 255, 0, 0, 255,        114, 255, 0, 255, 255,      114, 0, 0, 255, 0,        114, 255, 0, 255, 255,
            114, 114, 114, 114, 114,    114, 0, 255, 0, 255,        114, 0, 0, 0, 0,            114, 0, 255, 255, 0,      114, 0, 0, 0, 0,
            114, 114, 114, 114, 114,    114, 0, 0, 255, 0,          114, 0, 0, 255, 255,        114, 255, 255, 0, 0,      114, 0, 0, 255, 255,
            255, 255, 255, 255, 255,    255, 255, 255, 255, 255,    255, 0, 255, 0, 255,        255, 0, 255, 0, 255,      255, 0, 255, 0, 255,
        ];

        assert_eq!(cpu_dst.tensor().map().unwrap().as_slice(), &expected_dst);
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
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, PLANAR_RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();
        cpu_converter
            .convert(
                &src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::new()
                    .with_dst_rect(Some(Rect {
                        left: 102,
                        top: 102,
                        width: 440,
                        height: 440,
                    }))
                    .with_dst_color(Some([114, 114, 114, 114])),
            )
            .unwrap();

        let mut gl_dst = TensorImage::new(dst_width, dst_height, PLANAR_RGB, None).unwrap();
        let mut gl_converter = GLProcessorThreaded::new(None).unwrap();

        gl_converter
            .convert(
                &src,
                &mut gl_dst,
                Rotation::None,
                Flip::None,
                Crop::new()
                    .with_dst_rect(Some(Rect {
                        left: 102,
                        top: 102,
                        width: 440,
                        height: 440,
                    }))
                    .with_dst_color(Some([114, 114, 114, 114])),
            )
            .unwrap();
        compare_images(&gl_dst, &cpu_dst, 0.98, function!());
    }

    #[test]
    fn test_cpu_resize_nv16() {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        let mut cpu_nv16_dst = TensorImage::new(640, 640, NV16, None).unwrap();
        let mut cpu_rgb_dst = TensorImage::new(640, 640, RGB, None).unwrap();
        let mut cpu_converter = CPUProcessor::new();

        cpu_converter
            .convert(
                &src,
                &mut cpu_nv16_dst,
                Rotation::None,
                Flip::None,
                // Crop::no_crop(),
                Crop::new()
                    .with_dst_rect(Some(Rect {
                        left: 20,
                        top: 140,
                        width: 600,
                        height: 360,
                    }))
                    .with_dst_color(Some([255, 128, 0, 255])),
            )
            .unwrap();

        cpu_converter
            .convert(
                &src,
                &mut cpu_rgb_dst,
                Rotation::None,
                Flip::None,
                Crop::new()
                    .with_dst_rect(Some(Rect {
                        left: 20,
                        top: 140,
                        width: 600,
                        height: 360,
                    }))
                    .with_dst_color(Some([255, 128, 0, 255])),
            )
            .unwrap();
        compare_images_convert_to_rgb(&cpu_nv16_dst, &cpu_rgb_dst, 0.99, function!());
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorImage, Error> {
        let src = TensorImage::new(width, height, fourcc, memory)?;
        src.tensor().map()?.as_mut_slice().copy_from_slice(bytes);
        Ok(src)
    }

    fn compare_images(img1: &TensorImage, img2: &TensorImage, threshold: f64, name: &str) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(img1.fourcc(), img2.fourcc(), "FourCC differ");
        assert!(
            matches!(img1.fourcc(), RGB | RGBA | GREY | PLANAR_RGB),
            "FourCC must be RGB or RGBA for comparison"
        );

        let image1 = match img1.fourcc() {
            RGB => image::RgbImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            GREY => image::GrayImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PLANAR_RGB => image::GrayImage::from_vec(
                img1.width() as u32,
                (img1.height() * 3) as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.fourcc() {
            RGB => image::RgbImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            GREY => image::GrayImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PLANAR_RGB => image::GrayImage::from_vec(
                img2.width() as u32,
                (img2.height() * 3) as u32,
                img2.tensor().map().unwrap().to_vec(),
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
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 =
            TensorImage::new(img1.width(), img1.height(), RGB, Some(TensorMemory::Mem)).unwrap();
        let mut img_rgb2 =
            TensorImage::new(img1.width(), img1.height(), RGB, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::convert_format(img1, &mut img_rgb1).unwrap();
        CPUProcessor::convert_format(img2, &mut img_rgb2).unwrap();

        let image1 = image::RgbImage::from_vec(
            img_rgb1.width() as u32,
            img_rgb1.height() as u32,
            img_rgb1.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width() as u32,
            img_rgb2.height() as u32,
            img_rgb2.tensor().map().unwrap().to_vec(),
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
    // NV12 Format Tests
    // =========================================================================

    #[test]
    fn test_nv12_tensor_image_creation() {
        let width = 640;
        let height = 480;
        let img = TensorImage::new(width, height, NV12, None).unwrap();

        assert_eq!(img.width(), width);
        assert_eq!(img.height(), height);
        assert_eq!(img.fourcc(), NV12);
        // NV12 uses shape [H*3/2, W] to store Y plane + UV plane
        assert_eq!(img.tensor().shape(), &[height * 3 / 2, width]);
    }

    #[test]
    fn test_nv12_channels() {
        let img = TensorImage::new(640, 480, NV12, None).unwrap();
        // NV12 reports 2 channels (Y + interleaved UV)
        assert_eq!(img.channels(), 2);
    }

    // =========================================================================
    // TensorImageRef Tests
    // =========================================================================

    #[test]
    fn test_tensor_image_ref_from_planar_tensor() {
        // Create a planar RGB tensor [3, 480, 640]
        let mut tensor = Tensor::<u8>::new(&[3, 480, 640], None, None).unwrap();

        let img_ref = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        assert_eq!(img_ref.width(), 640);
        assert_eq!(img_ref.height(), 480);
        assert_eq!(img_ref.channels(), 3);
        assert_eq!(img_ref.fourcc(), PLANAR_RGB);
        assert!(img_ref.is_planar());
    }

    #[test]
    fn test_tensor_image_ref_from_interleaved_tensor() {
        // Create an interleaved RGBA tensor [480, 640, 4]
        let mut tensor = Tensor::<u8>::new(&[480, 640, 4], None, None).unwrap();

        let img_ref = TensorImageRef::from_borrowed_tensor(&mut tensor, RGBA).unwrap();

        assert_eq!(img_ref.width(), 640);
        assert_eq!(img_ref.height(), 480);
        assert_eq!(img_ref.channels(), 4);
        assert_eq!(img_ref.fourcc(), RGBA);
        assert!(!img_ref.is_planar());
    }

    #[test]
    fn test_tensor_image_ref_invalid_shape() {
        // 2D tensor should fail
        let mut tensor = Tensor::<u8>::new(&[480, 640], None, None).unwrap();
        let result = TensorImageRef::from_borrowed_tensor(&mut tensor, RGB);
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_tensor_image_ref_wrong_channels() {
        // RGBA expects 4 channels but tensor has 3
        let mut tensor = Tensor::<u8>::new(&[480, 640, 3], None, None).unwrap();
        let result = TensorImageRef::from_borrowed_tensor(&mut tensor, RGBA);
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_tensor_image_dst_trait_tensor_image() {
        let img = TensorImage::new(640, 480, RGB, None).unwrap();

        // Test TensorImageDst trait implementation
        fn check_dst<T: TensorImageDst>(dst: &T) {
            assert_eq!(dst.width(), 640);
            assert_eq!(dst.height(), 480);
            assert_eq!(dst.channels(), 3);
            assert!(!dst.is_planar());
        }

        check_dst(&img);
    }

    #[test]
    fn test_tensor_image_dst_trait_tensor_image_ref() {
        let mut tensor = Tensor::<u8>::new(&[3, 480, 640], None, None).unwrap();
        let img_ref = TensorImageRef::from_borrowed_tensor(&mut tensor, PLANAR_RGB).unwrap();

        fn check_dst<T: TensorImageDst>(dst: &T) {
            assert_eq!(dst.width(), 640);
            assert_eq!(dst.height(), 480);
            assert_eq!(dst.channels(), 3);
            assert!(dst.is_planar());
        }

        check_dst(&img_ref);
    }

    #[test]
    fn test_rgb_int8_format() {
        let img = TensorImage::new(1280, 720, RGB_INT8, Some(TensorMemory::Mem)).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);
        assert_eq!(img.channels(), 3);
        assert!(!img.is_planar());
        assert_eq!(img.fourcc(), RGB_INT8);
    }

    #[test]
    fn test_planar_rgb_int8_format() {
        let img = TensorImage::new(1280, 720, PLANAR_RGB_INT8, Some(TensorMemory::Mem)).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);
        assert_eq!(img.channels(), 3);
        assert!(img.is_planar());
        assert_eq!(img.fourcc(), PLANAR_RGB_INT8);
    }

    #[test]
    fn test_rgb_int8_from_tensor() {
        let tensor = Tensor::<u8>::new(&[720, 1280, 3], None, None).unwrap();
        let img = TensorImage::from_tensor(tensor, RGB_INT8).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);
        assert_eq!(img.channels(), 3);
        assert!(!img.is_planar());
        assert_eq!(img.fourcc(), RGB_INT8);
    }

    #[test]
    fn test_planar_rgb_int8_from_tensor() {
        let tensor = Tensor::<u8>::new(&[3, 720, 1280], None, None).unwrap();
        let img = TensorImage::from_tensor(tensor, PLANAR_RGB_INT8).unwrap();
        assert_eq!(img.width(), 1280);
        assert_eq!(img.height(), 720);
        assert_eq!(img.channels(), 3);
        assert!(img.is_planar());
        assert_eq!(img.fourcc(), PLANAR_RGB_INT8);
    }

    #[test]
    fn test_fourcc_is_int8() {
        assert!(fourcc_is_int8(RGB_INT8));
        assert!(fourcc_is_int8(PLANAR_RGB_INT8));
        assert!(!fourcc_is_int8(RGB));
        assert!(!fourcc_is_int8(PLANAR_RGB));
        assert!(!fourcc_is_int8(RGBA));
    }

    #[test]
    fn test_fourcc_uint8_equivalent() {
        assert_eq!(fourcc_uint8_equivalent(RGB_INT8), RGB);
        assert_eq!(fourcc_uint8_equivalent(PLANAR_RGB_INT8), PLANAR_RGB);
        assert_eq!(fourcc_uint8_equivalent(RGBA), RGBA);
    }

    #[test]
    fn test_fourcc_is_packed_rgb() {
        assert!(fourcc_is_packed_rgb(RGB));
        assert!(fourcc_is_packed_rgb(RGB_INT8));
        assert!(!fourcc_is_packed_rgb(PLANAR_RGB));
        assert!(!fourcc_is_packed_rgb(RGBA));
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
        let pbo_src = converter.create_image(src_w, src_h, RGBA).unwrap();
        assert_eq!(
            pbo_src.tensor().memory(),
            TensorMemory::Pbo,
            "create_image should produce a PBO tensor"
        );

        // Fill source PBO with test pattern: load JPEG then convert Mem→PBO
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let jpeg_src = TensorImage::load_jpeg(&file, Some(RGBA), None).unwrap();

        // Resize JPEG into a Mem temp of the right size, then copy into PBO
        let mut mem_src = TensorImage::new(src_w, src_h, RGBA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::new()
            .convert(
                &jpeg_src,
                &mut mem_src,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // Copy pixel data into the PBO source by mapping it
        {
            let src_data = mem_src.tensor().map().unwrap();
            let mut pbo_map = pbo_src.tensor().map().unwrap();
            pbo_map.copy_from_slice(&src_data);
        }

        // Create PBO-backed destination image
        let mut pbo_dst = converter.create_image(dst_w, dst_h, RGBA).unwrap();
        assert_eq!(pbo_dst.tensor().memory(), TensorMemory::Pbo);

        // Convert PBO→PBO (this exercises convert_pbo_to_pbo)
        converter
            .convert(
                &pbo_src,
                &mut pbo_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        // Verify: compare with CPU-only conversion of the same input
        let mut cpu_dst = TensorImage::new(dst_w, dst_h, RGBA, Some(TensorMemory::Mem)).unwrap();
        CPUProcessor::new()
            .convert(
                &mem_src,
                &mut cpu_dst,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            )
            .unwrap();

        compare_images(&pbo_dst, &cpu_dst, 0.95, function!());
        log::info!("test_convert_pbo_to_pbo: PASS — PBO-to-PBO convert matches CPU reference");
    }

    #[test]
    fn test_tensor_image_bgra() {
        let img =
            TensorImage::new(640, 480, BGRA, Some(edgefirst_tensor::TensorMemory::Mem)).unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
        assert_eq!(img.channels(), 4);
        assert_eq!(img.fourcc(), BGRA);
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
    fn test_draw_masks_proto_no_cpu_returns_error() {
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

        let mut dst = TensorImage::new(640, 480, RGBA, Some(TensorMemory::Mem)).unwrap();
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
        let proto_data = ProtoData {
            mask_coefficients: vec![vec![0.5; 4]],
            protos: edgefirst_decoder::ProtoTensor::Float(ndarray::Array3::<f32>::zeros((8, 8, 4))),
        };
        let result = converter.draw_masks_proto(&mut dst, &det, &proto_data);
        assert!(
            matches!(&result, Err(Error::Internal(s)) if s.contains("CPU backend")),
            "draw_masks_proto without CPU should return Internal error: {result:?}"
        );
    }

    #[test]
    fn test_draw_masks_proto_cpu_fallback_works() {
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

        let mut dst = TensorImage::new(64, 64, RGBA, Some(TensorMemory::Mem)).unwrap();
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
        let proto_data = ProtoData {
            mask_coefficients: vec![vec![0.5; 4]],
            protos: edgefirst_decoder::ProtoTensor::Float(ndarray::Array3::<f32>::zeros((8, 8, 4))),
        };
        let result = converter.draw_masks_proto(&mut dst, &det, &proto_data);
        assert!(result.is_ok(), "CPU fallback path should work: {result:?}");
    }
}
