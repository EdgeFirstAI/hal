// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ImageLoad`] extension trait for loading images into pre-allocated tensors.

use crate::decoder::ImageDecoder;
use crate::error::CodecError;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{Tensor, TensorDyn};
use std::io::{BufReader, Read};
use std::path::Path;

/// Extension trait for loading images into pre-allocated tensors.
///
/// Import this trait to add `load_image`, `load_image_read`, and
/// `load_image_file` methods to [`Tensor<T>`] and [`TensorDyn`].
///
/// The decoder configures the tensor's dimensions and pixel format to the
/// decoded native format (JPEG → `Nv12`/`Grey`, PNG → `Rgb`/`Rgba`/`Grey`).
/// Allocate the tensor at the maximum expected image size; smaller images
/// reconfigure it within the same allocation.
///
/// # Example
///
/// ```rust,no_run
/// use edgefirst_codec::{ImageDecoder, ImageLoad};
/// use edgefirst_tensor::{Tensor, TensorTrait, TensorMemory, PixelFormat};
///
/// let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12, Some(TensorMemory::Mem))
///     .expect("alloc");
/// let mut decoder = ImageDecoder::new();
///
/// let jpeg = std::fs::read("image.jpg").unwrap();
/// let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
/// println!("Decoded {}×{} {:?}", info.width, info.height, info.format);
///
/// let info = tensor.load_image_file(&mut decoder, "image.jpg").unwrap();
/// ```
pub trait ImageLoad {
    /// Decode image bytes (`&[u8]`) into this tensor, configuring its
    /// dimensions and format to the decoded native format.
    ///
    /// The image format (JPEG or PNG) is detected from magic bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InsufficientCapacity`] if the decoded image is
    /// larger than the tensor's allocation.
    fn load_image(&mut self, decoder: &mut ImageDecoder, data: &[u8]) -> crate::Result<ImageInfo>;

    /// Decode image data from a [`Read`] source into this tensor.
    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
    ) -> crate::Result<ImageInfo>;

    /// Decode an image file into this tensor.
    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
    ) -> crate::Result<ImageInfo>;
}

impl<T: ImagePixel> ImageLoad for Tensor<T> {
    fn load_image(&mut self, decoder: &mut ImageDecoder, data: &[u8]) -> crate::Result<ImageInfo> {
        decoder.decode_into(data, self)
    }

    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_from_reader(reader, self)
    }

    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
    ) -> crate::Result<ImageInfo> {
        let file = std::fs::File::open(path.as_ref()).map_err(CodecError::Io)?;
        self.load_image_read(decoder, BufReader::new(file))
    }
}

impl ImageLoad for TensorDyn {
    fn load_image(&mut self, decoder: &mut ImageDecoder, data: &[u8]) -> crate::Result<ImageInfo> {
        decoder.decode_into_dyn(data, self)
    }

    fn load_image_read<R: Read>(
        &mut self,
        decoder: &mut ImageDecoder,
        reader: R,
    ) -> crate::Result<ImageInfo> {
        decoder.decode_from_reader_dyn(reader, self)
    }

    fn load_image_file(
        &mut self,
        decoder: &mut ImageDecoder,
        path: impl AsRef<Path>,
    ) -> crate::Result<ImageInfo> {
        let file = std::fs::File::open(path.as_ref()).map_err(CodecError::Io)?;
        self.load_image_read(decoder, BufReader::new(file))
    }
}
