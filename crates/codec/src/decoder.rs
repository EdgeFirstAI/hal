// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageDecoder`] — reusable decoder state for zero-allocation hot loops.

use crate::engine::Engine;
#[cfg(feature = "png")]
use crate::engine::{engines_to_try, unsupported, Codec, Direction};
use crate::error::CodecError;
#[cfg(not(feature = "png"))]
use crate::error::UnsupportedFeature;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{DType, Tensor, TensorDyn};
use std::io::Read;

/// IDCT selection for the software JPEG decoder.
///
/// [`Accurate`](Self::Accurate), the default, is the `islow`-class Loeffler
/// IDCT. [`Fast`](Self::Fast) is the AAN `ifast`-class kernel, with lower
/// pixel accuracy. Fast is advisory: paths without a fast kernel (non-NEON
/// tiers, hardware V4L2/nvJPEG decoders, PNG) use their accurate path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DctMethod {
    /// Accurate `islow`-class IDCT (default).
    #[default]
    Accurate,
    /// Fast AAN `ifast`-class IDCT (opt-in).
    Fast,
}

/// Chroma upsampling filter for a fused native-4:2:0 `Rgb` decode (see
/// [`ImageDecoder::set_output_format`]).
///
/// [`Box`](Self::Box), 2×2 nearest-neighbour replication, is the only
/// filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChromaUpsample {
    /// 2×2 nearest-neighbour (box) chroma replication.
    #[default]
    Box,
}

/// Reusable image decoder with internal scratch buffers.
///
/// Create one `ImageDecoder` and pass it to
/// [`ImageLoad::load_image`](crate::ImageLoad::load_image) for every image.
/// The scratch buffers grow to the high-water mark and are reused across
/// calls.
///
/// By default the decoder produces the source's native format and configures
/// the destination tensor's dimensions and pixel format accordingly (JPEG →
/// `Nv12`/`Nv16`/`Nv24`/`Grey`, PNG → `Rgb`/`Rgba`/`Grey`).
/// [`set_output_format`](Self::set_output_format) opts a JPEG decode into a
/// fused `Rgb` or `Nv12` output, where the colour conversion or chroma
/// downsample happens at the MCU write stage. The decoder never rotates or
/// resizes.
///
/// # Example
///
/// ```rust,no_run
/// use edgefirst_codec::{ImageDecoder, ImageLoad};
/// use edgefirst_tensor::{CpuAccess, Tensor, TensorTrait, TensorMemory, PixelFormat};
///
/// let mut decoder = ImageDecoder::new();
/// // Allocate a buffer large enough for the biggest expected image.
/// let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12, Some(TensorMemory::Mem),
///                                       CpuAccess::Write)
///     .expect("alloc");
///
/// for _ in 0..100 {
///     let frame = std::fs::read("frame.jpg").unwrap();
///     let _info = tensor.load_image(&mut decoder, &frame);
/// }
/// ```
pub struct ImageDecoder {
    /// Reusable JPEG decoder state (Huffman tables, MCU scratch buffers).
    pub(crate) jpeg_state: crate::jpeg::JpegDecoderState,
    /// Scratch buffer for PNG decode output.
    pub(crate) scratch: Vec<u8>,
    /// Buffer for `Read` → `&[u8]` conversion.
    pub(crate) input_buffer: Vec<u8>,
}

impl ImageDecoder {
    /// Create a new decoder with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            jpeg_state: crate::jpeg::JpegDecoderState::new(),
            scratch: Vec::new(),
            input_buffer: Vec::new(),
        }
    }

    /// Request a fused decode output format instead of the source's native
    /// format (JPEG only; PNG is unaffected).
    ///
    /// The software JPEG decoder converts at the MCU write stage; V4L2 and
    /// nvJPEG are bypassed whenever the resolved output differs from native.
    ///
    /// - `Some(Rgb)`: 4:4:4 colour JPEGs decode to interleaved RGB with no
    ///   chroma resampling; native 4:2:0 colour JPEGs decode to RGB with a 2×2
    ///   nearest-neighbour (box) chroma upsample. Other sources (4:2:2,
    ///   greyscale, non-standard subsamplings) fail with
    ///   [`CodecError::UnsupportedFormat`]; [`crate::peek_info`] reports the
    ///   source's native format.
    /// - `Some(Nv12)`: colour JPEGs decode to NV12, downsampling chroma at
    ///   the write stage (2×2 average for 4:4:4, vertical for 4:2:2).
    /// - `None` (default): native format (`Nv12`/`Nv16`/`Nv24`/`Grey`).
    pub fn set_output_format(&mut self, format: Option<edgefirst_tensor::PixelFormat>) {
        self.jpeg_state.preferred_format = format;
    }

    /// Select the software JPEG IDCT kernel class; see [`DctMethod`].
    /// Accurate by default; `EDGEFIRST_CODEC_DCT=fast` in the environment makes
    /// Fast the default of a new decoder.
    pub fn set_dct_method(&mut self, method: DctMethod) {
        self.jpeg_state.fast_dct = method == DctMethod::Fast;
    }

    /// Select the chroma upsampling filter for a fused native-4:2:0 `Rgb`
    /// decode; see [`ChromaUpsample`]. Has no effect on 4:4:4 sources or when
    /// the output format isn't `Rgb`.
    pub fn set_chroma_upsample(&mut self, mode: ChromaUpsample) {
        self.jpeg_state.chroma_upsample = mode;
    }

    /// Selects the engine that decodes.
    ///
    /// [`Engine::Auto`] (the default) tries nvJPEG (when enabled with
    /// `EDGEFIRST_ENABLE_NVJPEG`), V4L2, then the CPU for JPEG, each hardware
    /// decoder passing an image it cannot take to the next, and the CPU for
    /// PNG; [`ENGINE_ENV`](crate::ENGINE_ENV) narrows that list. Any other
    /// engine decodes alone: an image it cannot take fails with
    /// [`CodecError::Unsupported`] carrying
    /// [`UnsupportedFeature::Codec`](crate::UnsupportedFeature::Codec), and an
    /// engine this platform does not have with [`CodecError::NoDevice`].
    pub fn set_engine(&mut self, engine: Engine) {
        self.jpeg_state.engine = engine;
    }

    /// The engine selected with [`set_engine`](Self::set_engine).
    pub fn engine(&self) -> Engine {
        self.jpeg_state.engine
    }

    /// Whether the V4L2 decoder wrote the last JPEG straight into its
    /// destination, rather than copying it out of its own buffer.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    pub(crate) fn v4l2_wrote_in_place(&self) -> bool {
        self.jpeg_state.v4l2_wrote_in_place()
    }

    /// Decode image data into a typed tensor, configuring its dimensions and
    /// pixel format to the decoded output format — the source's native format
    /// unless [`set_output_format`](Self::set_output_format) selected a fused
    /// one.
    ///
    /// Detects the image format (JPEG or PNG) from magic bytes.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InsufficientCapacity`] if the image is larger than the
    ///   tensor's allocation
    /// - [`CodecError::UnsupportedDtype`] if `T` is not valid for the output
    ///   format (JPEG NV12/NV16/NV24/RGB/GREY require `u8`)
    /// - [`CodecError::InvalidData`] if the data is not a valid JPEG or PNG,
    ///   including a JPEG whose entropy-coded scan is truncated
    /// - [`CodecError::Tensor`] carrying
    ///   [`CpuAccessDenied`](edgefirst_tensor::Error::CpuAccessDenied) when
    ///   the CPU decoder must write a tensor whose declared
    ///   [`CpuAccess`](edgefirst_tensor::CpuAccess) has no write direction
    pub fn decode_into<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
    ) -> crate::Result<ImageInfo> {
        decode_into_inner(&mut self.jpeg_state, &mut self.scratch, data, dst)
    }

    /// Decode image data into a type-erased tensor.
    pub fn decode_into_dyn(
        &mut self,
        data: &[u8],
        dst: &mut TensorDyn,
    ) -> crate::Result<ImageInfo> {
        match dst.dtype() {
            DType::U8 => self.decode_into(data, dst.as_typed_mut::<u8>().expect("dtype checked")),
            DType::I8 => self.decode_into(data, dst.as_typed_mut::<i8>().expect("dtype checked")),
            DType::U16 => self.decode_into(data, dst.as_typed_mut::<u16>().expect("dtype checked")),
            DType::I16 => self.decode_into(data, dst.as_typed_mut::<i16>().expect("dtype checked")),
            DType::F32 => self.decode_into(data, dst.as_typed_mut::<f32>().expect("dtype checked")),
            other => Err(CodecError::UnsupportedDtype(other)),
        }
    }

    /// Read all bytes from a `Read` source into the internal input buffer, then
    /// decode. The input buffer is reused across calls.
    pub fn decode_from_reader<T: ImagePixel, R: Read>(
        &mut self,
        mut reader: R,
        dst: &mut Tensor<T>,
    ) -> crate::Result<ImageInfo> {
        self.input_buffer.clear();
        reader.read_to_end(&mut self.input_buffer)?;
        decode_into_inner(
            &mut self.jpeg_state,
            &mut self.scratch,
            &self.input_buffer,
            dst,
        )
    }

    /// Read all bytes from a `Read` source, then decode into a type-erased
    /// tensor.
    pub fn decode_from_reader_dyn<R: Read>(
        &mut self,
        mut reader: R,
        dst: &mut TensorDyn,
    ) -> crate::Result<ImageInfo> {
        self.input_buffer.clear();
        reader.read_to_end(&mut self.input_buffer)?;
        decode_into_inner_dyn(
            &mut self.jpeg_state,
            &mut self.scratch,
            &self.input_buffer,
            dst,
        )
    }
}

/// Decode entry point over disjoint `&mut` borrows of the [`ImageDecoder`]
/// fields.
pub(crate) fn decode_into_inner<T: ImagePixel>(
    jpeg_state: &mut crate::jpeg::JpegDecoderState,
    scratch: &mut Vec<u8>,
    data: &[u8],
    dst: &mut Tensor<T>,
) -> crate::Result<ImageInfo> {
    if is_jpeg(data) {
        crate::jpeg::decode_jpeg_into(data, dst, jpeg_state)
    } else if is_png(data) {
        {
            #[cfg(feature = "png")]
            {
                // PNG has only the CPU decoder.
                let engines = engines_to_try(jpeg_state.engine, &[Engine::Cpu])?;
                if engines.first() != Some(&Engine::Cpu) {
                    return Err(unsupported(
                        Codec::Png,
                        Direction::Decode,
                        jpeg_state.engine,
                    ));
                }
                crate::png::decode_png_into(data, dst, scratch)
            }
            #[cfg(not(feature = "png"))]
            {
                let _ = scratch;
                Err(CodecError::Unsupported(UnsupportedFeature::PngNotEnabled))
            }
        }
    } else {
        Err(CodecError::InvalidData(
            "unrecognized image format (expected JPEG or PNG magic bytes)".into(),
        ))
    }
}

pub(crate) fn decode_into_inner_dyn(
    jpeg_state: &mut crate::jpeg::JpegDecoderState,
    scratch: &mut Vec<u8>,
    data: &[u8],
    dst: &mut TensorDyn,
) -> crate::Result<ImageInfo> {
    match dst.dtype() {
        DType::U8 => decode_into_inner(
            jpeg_state,
            scratch,
            data,
            dst.as_typed_mut::<u8>().expect("dtype checked"),
        ),
        DType::I8 => decode_into_inner(
            jpeg_state,
            scratch,
            data,
            dst.as_typed_mut::<i8>().expect("dtype checked"),
        ),
        DType::U16 => decode_into_inner(
            jpeg_state,
            scratch,
            data,
            dst.as_typed_mut::<u16>().expect("dtype checked"),
        ),
        DType::I16 => decode_into_inner(
            jpeg_state,
            scratch,
            data,
            dst.as_typed_mut::<i16>().expect("dtype checked"),
        ),
        DType::F32 => decode_into_inner(
            jpeg_state,
            scratch,
            data,
            dst.as_typed_mut::<f32>().expect("dtype checked"),
        ),
        other => Err(CodecError::UnsupportedDtype(other)),
    }
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse image headers and return the native dimensions, format, and EXIF
/// orientation without decoding pixels.
///
/// The result sizes a tensor for the image:
///
/// ```rust,no_run
/// use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
/// use edgefirst_tensor::{CpuAccess, Tensor, TensorMemory};
///
/// let data = std::fs::read("image.jpg").unwrap();
/// let info = peek_info(&data).unwrap();
/// let mut tensor = Tensor::<u8>::image(info.width, info.height, info.format,
///                                       Some(TensorMemory::Mem),
///                                       CpuAccess::Write).unwrap();
/// let mut decoder = ImageDecoder::new();
/// tensor.load_image(&mut decoder, &data).unwrap();
/// ```
pub fn peek_info(data: &[u8]) -> crate::Result<ImageInfo> {
    if is_jpeg(data) {
        crate::jpeg::peek_jpeg_info(data)
    } else if is_png(data) {
        {
            #[cfg(feature = "png")]
            {
                crate::png::peek_png_info(data)
            }
            #[cfg(not(feature = "png"))]
            {
                Err(CodecError::Unsupported(UnsupportedFeature::PngNotEnabled))
            }
        }
    } else {
        Err(CodecError::InvalidData(
            "unrecognized image format (expected JPEG or PNG magic bytes)".into(),
        ))
    }
}

/// Check for JPEG magic bytes (FFD8FF).
fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

/// Check for PNG magic bytes (89504E47).
fn is_png(data: &[u8]) -> bool {
    data.len() >= 4 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_bytes_jpeg() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(&[0x89, 0x50, 0x4E, 0x47]));
        assert!(!is_jpeg(&[]));
        assert!(!is_jpeg(&[0xFF]));
    }

    #[test]
    fn magic_bytes_png() {
        assert!(is_png(&[0x89, 0x50, 0x4E, 0x47]));
        assert!(!is_png(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_png(&[]));
    }

    #[test]
    fn invalid_data() {
        let mut decoder = ImageDecoder::new();
        let mut tensor = Tensor::<u8>::image(
            100,
            100,
            edgefirst_tensor::PixelFormat::Nv12,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let result = decoder.decode_into(b"not an image", &mut tensor);
        assert!(matches!(result, Err(CodecError::InvalidData(_))));
    }
}
