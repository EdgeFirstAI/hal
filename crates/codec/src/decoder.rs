// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageDecoder`] — reusable decoder state for zero-allocation hot loops.

use crate::error::CodecError;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{Tensor, TensorDyn};
use std::io::Read;

/// Reusable image decoder with internal scratch buffers.
///
/// Create one `ImageDecoder` at program initialisation and pass it to
/// [`ImageLoad::load_image`](crate::ImageLoad::load_image) in the hot loop.
/// The scratch buffers grow to the high-water mark and are reused across calls
/// — no per-frame allocations after the first few frames.
///
/// The decoder always produces the source's native format and configures the
/// destination tensor's dimensions + pixel format accordingly (JPEG →
/// `Nv12`/`Grey`, PNG → `Rgb`/`Rgba`/`Grey`). It never colour-converts or
/// rotates; use `ImageProcessor::convert()` for that.
///
/// # Example
///
/// ```rust,no_run
/// use edgefirst_codec::{ImageDecoder, ImageLoad};
/// use edgefirst_tensor::{CpuAccess, Tensor, TensorTrait, TensorMemory, PixelFormat};
///
/// let mut decoder = ImageDecoder::new();
/// // Allocate a buffer large enough for the biggest expected image. The
/// // decoder CPU-writes the pixels, so declare `CpuAccess::Write`.
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

    /// Decode image data into a typed tensor, configuring its dimensions and
    /// pixel format to the decoded native format.
    ///
    /// Detects the image format (JPEG or PNG) from magic bytes.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InsufficientCapacity`] if the image is larger than the
    ///   tensor's allocation
    /// - [`CodecError::UnsupportedDtype`] if `T` is not valid for the native
    ///   format (JPEG NV12/GREY require `u8`)
    /// - [`CodecError::InvalidData`] if the data is not a valid JPEG or PNG
    pub fn decode_into<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
    ) -> crate::Result<ImageInfo> {
        if is_jpeg(data) {
            crate::jpeg::decode_jpeg_into(data, dst, &mut self.jpeg_state)
        } else if is_png(data) {
            crate::png::decode_png_into(data, dst, &mut self.scratch)
        } else {
            Err(CodecError::InvalidData(
                "unrecognized image format (expected JPEG or PNG magic bytes)".into(),
            ))
        }
    }

    /// Decode image data into a type-erased tensor.
    pub fn decode_into_dyn(
        &mut self,
        data: &[u8],
        dst: &mut TensorDyn,
    ) -> crate::Result<ImageInfo> {
        match dst {
            TensorDyn::U8(t) => self.decode_into(data, t),
            TensorDyn::I8(t) => self.decode_into(data, t),
            TensorDyn::U16(t) => self.decode_into(data, t),
            TensorDyn::I16(t) => self.decode_into(data, t),
            TensorDyn::F32(t) => self.decode_into(data, t),
            other => Err(CodecError::UnsupportedDtype(other.dtype())),
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

/// Internal decode entry point parameterised over disjoint `&mut` borrows so
/// callers can read from a `&[u8]` borrowed from one field of [`ImageDecoder`]
/// while mutably borrowing the others.
pub(crate) fn decode_into_inner<T: ImagePixel>(
    jpeg_state: &mut crate::jpeg::JpegDecoderState,
    scratch: &mut Vec<u8>,
    data: &[u8],
    dst: &mut Tensor<T>,
) -> crate::Result<ImageInfo> {
    if is_jpeg(data) {
        crate::jpeg::decode_jpeg_into(data, dst, jpeg_state)
    } else if is_png(data) {
        crate::png::decode_png_into(data, dst, scratch)
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
    match dst {
        TensorDyn::U8(t) => decode_into_inner(jpeg_state, scratch, data, t),
        TensorDyn::I8(t) => decode_into_inner(jpeg_state, scratch, data, t),
        TensorDyn::U16(t) => decode_into_inner(jpeg_state, scratch, data, t),
        TensorDyn::I16(t) => decode_into_inner(jpeg_state, scratch, data, t),
        TensorDyn::F32(t) => decode_into_inner(jpeg_state, scratch, data, t),
        other => Err(CodecError::UnsupportedDtype(other.dtype())),
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
/// Recommended for one-shot flows that allocate a tensor sized to the image:
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
        crate::png::peek_png_info(data)
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
