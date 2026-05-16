// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageDecoder`] — reusable decoder state for zero-allocation hot loops.

use crate::error::CodecError;
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{Tensor, TensorDyn};
use std::io::Read;

/// Reusable image decoder with internal scratch buffers.
///
/// Create one `ImageDecoder` at program initialisation and pass it to
/// [`ImageLoad::load_image`](crate::ImageLoad::load_image) in the hot loop.
/// The scratch buffers grow to the high-water mark and are reused across
/// calls — no per-frame allocations after the first few frames.
///
/// # Example
///
/// ```rust,no_run
/// use edgefirst_codec::{ImageDecoder, ImageLoad, DecodeOptions};
/// use edgefirst_tensor::{Tensor, TensorTrait, TensorMemory, PixelFormat};
///
/// let mut decoder = ImageDecoder::new();
/// let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem))
///     .expect("alloc");
///
/// for _ in 0..100 {
///     let frame = std::fs::read("frame.jpg").unwrap();
///     let _info = tensor.load_image(&mut decoder, &frame, &DecodeOptions::default());
/// }
/// ```
pub struct ImageDecoder {
    /// Scratch buffer for the contiguous decode output from zune.
    pub(crate) scratch: Vec<u8>,
    /// Buffer for `Read` → `&[u8]` conversion.
    pub(crate) input_buffer: Vec<u8>,
}

impl ImageDecoder {
    /// Create a new decoder with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            input_buffer: Vec::new(),
        }
    }

    /// Decode image data into a typed tensor.
    ///
    /// Detects the image format (JPEG or PNG) from magic bytes and decodes
    /// into `dst`. The tensor must have sufficient capacity for the decoded
    /// image dimensions.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InsufficientCapacity`] if image dimensions exceed tensor
    /// - [`CodecError::UnsupportedDtype`] if `T` is not a supported pixel type
    /// - [`CodecError::InvalidData`] if the data is not a valid JPEG or PNG
    pub fn decode_into<T: ImagePixel>(
        &mut self,
        data: &[u8],
        dst: &mut Tensor<T>,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        if is_jpeg(data) {
            crate::jpeg::decode_jpeg_into(data, dst, opts, &mut self.scratch)
        } else if is_png(data) {
            crate::png::decode_png_into(data, dst, opts, &mut self.scratch)
        } else {
            Err(CodecError::InvalidData(
                "unrecognized image format (expected JPEG or PNG magic bytes)".into(),
            ))
        }
    }

    /// Decode image data into a type-erased tensor.
    ///
    /// Dispatches to the appropriate typed decode path based on the tensor's
    /// [`DType`](edgefirst_tensor::DType).
    pub fn decode_into_dyn(
        &mut self,
        data: &[u8],
        dst: &mut TensorDyn,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        match dst {
            TensorDyn::U8(t) => self.decode_into(data, t, opts),
            TensorDyn::I8(t) => self.decode_into(data, t, opts),
            TensorDyn::U16(t) => self.decode_into(data, t, opts),
            TensorDyn::I16(t) => self.decode_into(data, t, opts),
            TensorDyn::F32(t) => self.decode_into(data, t, opts),
            other => Err(CodecError::UnsupportedDtype(other.dtype())),
        }
    }

    /// Read all bytes from a `Read` source into the internal input buffer,
    /// then decode.
    pub fn decode_from_reader<T: ImagePixel, R: Read>(
        &mut self,
        mut reader: R,
        dst: &mut Tensor<T>,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        self.input_buffer.clear();
        reader.read_to_end(&mut self.input_buffer)?;
        self.decode_into(&self.input_buffer.clone(), dst, opts)
    }

    /// Read all bytes from a `Read` source into the internal input buffer,
    /// then decode into a type-erased tensor.
    pub fn decode_from_reader_dyn<R: Read>(
        &mut self,
        mut reader: R,
        dst: &mut TensorDyn,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        self.input_buffer.clear();
        reader.read_to_end(&mut self.input_buffer)?;
        self.decode_into_dyn(&self.input_buffer.clone(), dst, opts)
    }
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self::new()
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
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let result = decoder.decode_into(b"not an image", &mut tensor, &DecodeOptions::default());
        assert!(matches!(result, Err(CodecError::InvalidData(_))));
    }
}
