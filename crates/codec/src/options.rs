// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Output metadata returned by the decoder.
//!
//! The codec has no decode options: it always decodes to the source's native
//! format (JPEG → NV12/GREY, PNG → RGB/RGBA/GREY), writes that format onto the
//! destination tensor, and reports EXIF orientation for the caller to apply via
//! `ImageProcessor::convert()`. It never colour-converts or rotates.

use edgefirst_tensor::PixelFormat;

/// Metadata returned after a successful image decode.
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    /// Width of the decoded image in pixels (the source's true width; the
    /// codec does not apply EXIF rotation).
    pub width: usize,
    /// Height of the decoded image in pixels (the source's true height).
    pub height: usize,
    /// Native pixel format written to the tensor (JPEG → `Nv12`/`Grey`,
    /// PNG → `Rgb`/`Rgba`/`Grey`).
    pub format: PixelFormat,
    /// Row stride in bytes used when writing into the tensor.
    pub row_stride: usize,
    /// EXIF orientation: clockwise rotation in degrees (0/90/180/270) the
    /// consumer should apply downstream (e.g. via `convert()`). The codec does
    /// **not** rotate. `0` when there is no EXIF orientation.
    pub rotation_degrees: u16,
    /// EXIF horizontal flip the consumer should apply downstream. `false` when
    /// there is no EXIF orientation.
    pub flip_horizontal: bool,
}
