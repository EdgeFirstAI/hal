// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decode options and output metadata.

use edgefirst_tensor::PixelFormat;

/// Options controlling how an image is decoded.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Desired output pixel format. `None` uses the native format from the
    /// file (typically RGB for JPEG, RGB/RGBA for PNG).
    pub format: Option<PixelFormat>,

    /// JPEG IDCT downscale denominator (1, 2, 4, or 8). Ignored for PNG.
    /// A value of 2 decodes at half resolution, 4 at quarter, etc.
    pub scale_denom: u8,

    /// Whether to apply EXIF orientation metadata. Default: `true`.
    ///
    /// Set to `false` when the caller will handle rotation/flip externally
    /// (e.g. via [`ImageProcessor::convert()`]).
    pub apply_exif: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            format: None,
            scale_denom: 1,
            apply_exif: true,
        }
    }
}

impl DecodeOptions {
    /// Set the desired output pixel format.
    #[must_use]
    pub fn with_format(mut self, format: PixelFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the JPEG IDCT downscale denominator.
    #[must_use]
    pub fn with_scale(mut self, denom: u8) -> Self {
        self.scale_denom = denom;
        self
    }

    /// Set whether to apply EXIF orientation.
    #[must_use]
    pub fn with_exif(mut self, apply: bool) -> Self {
        self.apply_exif = apply;
        self
    }
}

/// Metadata returned after a successful image decode.
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    /// Width of the decoded image in pixels.
    pub width: usize,
    /// Height of the decoded image in pixels.
    pub height: usize,
    /// Pixel format of the decoded data written to the tensor.
    pub format: PixelFormat,
    /// Row stride in bytes used when writing into the tensor.
    pub row_stride: usize,
}
