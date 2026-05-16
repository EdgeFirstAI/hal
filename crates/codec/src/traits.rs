// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageLoad`] extension trait for loading images into pre-allocated tensors.

use crate::decoder::ImageDecoder;
use crate::error::CodecError;
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{Tensor, TensorDyn};
use std::io::{BufReader, Read};
use std::path::Path;

/// Extension trait for loading images into pre-allocated tensors.
///
/// Import this trait to add `load_image`, `load_image_read`, and
/// `load_image_file` methods to [`Tensor<T>`] and [`TensorDyn`].
///
/// # Performance
///
/// For best performance, allocate tensors via
/// [`ImageProcessor::create_image()`] which selects the optimal memory
/// backend (DMA → PBO → Mem) with GPU-aligned pitch. Free-standing tensors
/// work but cannot use PBO and may not have GPU-aligned pitch.
///
/// # Strided Buffers
///
/// The decoder writes directly into the tensor's strided memory layout,
/// respecting [`effective_row_stride()`]. Allocate the tensor at the maximum
/// expected image size; smaller images decode into the top-left region.
///
/// # Example
///
/// ```rust,no_run
/// use edgefirst_codec::{ImageDecoder, ImageLoad, DecodeOptions};
/// use edgefirst_tensor::{Tensor, TensorTrait, TensorMemory, PixelFormat};
///
/// let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem))
///     .expect("alloc");
/// let mut decoder = ImageDecoder::new();
///
/// // Decode from bytes
/// let jpeg = std::fs::read("image.jpg").unwrap();
/// let info = tensor.load_image(&mut decoder, &jpeg, &DecodeOptions::default()).unwrap();
/// println!("Decoded {}×{} {:?}", info.width, info.height, info.format);
///
/// // Decode from file
/// let info = tensor.load_image_file(&mut decoder, "image.jpg", &DecodeOptions::default()).unwrap();
/// ```
pub trait ImageLoad {
    /// Decode image bytes (`&[u8]`) into this tensor.
    ///
    /// The image format (JPEG or PNG) is detected from magic bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InsufficientCapacity`] if the decoded image
    /// dimensions exceed the tensor's capacity.
    fn load_image(
        &mut self,
        decoder: &mut ImageDecoder,
        data: &[u8],
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo>;

    /// Decode image data from a [`Read`] source into this tensor.
    ///
    /// The input is buffered into the decoder's internal scratch before
    /// decoding. For large inputs, prefer [`load_image`](Self::load_image)
    /// with a pre-read `&[u8]` to avoid the copy.
    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo>;

    /// Decode an image file into this tensor.
    ///
    /// Convenience wrapper that opens the file, buffers it, and decodes.
    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo>;
}

impl<T: ImagePixel> ImageLoad for Tensor<T> {
    fn load_image(
        &mut self,
        decoder: &mut ImageDecoder,
        data: &[u8],
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_into(data, self, opts)
    }

    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_from_reader(reader, self, opts)
    }

    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        let file = std::fs::File::open(path.as_ref()).map_err(CodecError::Io)?;
        self.load_image_read(decoder, BufReader::new(file), opts)
    }
}

impl ImageLoad for TensorDyn {
    fn load_image(
        &mut self,
        decoder: &mut ImageDecoder,
        data: &[u8],
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_into_dyn(data, self, opts)
    }

    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_from_reader_dyn(reader, self, opts)
    }

    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
        opts: &DecodeOptions,
    ) -> crate::Result<ImageInfo> {
        let file = std::fs::File::open(path.as_ref()).map_err(CodecError::Io)?;
        self.load_image_read(decoder, BufReader::new(file), opts)
    }
}
