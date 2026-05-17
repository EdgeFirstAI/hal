// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Error types for the codec crate.

use std::fmt;

/// A specific image-format feature that the codec does not implement.
///
/// Carried by [`CodecError::Unsupported`]. Use this to distinguish "the byte
/// string is well-formed but uses a feature we don't decode" from "the byte
/// string is corrupt" ([`CodecError::InvalidData`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedFeature {
    /// JPEG SOF2 (progressive baseline).
    ProgressiveJpeg,
    /// JPEG SOF1 (extended sequential, Huffman coded).
    ExtendedSequentialJpeg,
    /// JPEG SOF9/SOF10/SOF11/SOF13/SOF14/SOF15 (arithmetic coding, in all
    /// sequential/progressive/lossless/hierarchical flavours).
    ArithmeticCodedJpeg,
    /// JPEG SOF3 (lossless predictive).
    LosslessJpeg,
    /// JPEG SOF5/SOF6/SOF7 (hierarchical).
    HierarchicalJpeg,
    /// JPEG sample precision other than 8 bits per channel.
    JpegPrecision {
        /// The unsupported precision reported in the SOF marker.
        bits: u8,
    },
    /// JPEG with a component count we do not handle (anything other than 1
    /// for greyscale or 3 for YCbCr).
    JpegComponentCount {
        /// The unsupported component count reported in the SOF marker.
        components: u8,
    },
    /// JPEG chroma subsampling configuration where a chroma component is
    /// sampled at a higher rate than luma along some axis (rejected so the
    /// downstream upsampler does not divide by zero).
    JpegChromaSubsampling,
}

impl fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgressiveJpeg => write!(f, "progressive JPEG (SOF2)"),
            Self::ExtendedSequentialJpeg => {
                write!(f, "extended sequential JPEG (SOF1)")
            }
            Self::ArithmeticCodedJpeg => write!(
                f,
                "arithmetic-coded JPEG (SOF9/SOF10/SOF11/SOF13/SOF14/SOF15)"
            ),
            Self::LosslessJpeg => write!(f, "lossless JPEG (SOF3)"),
            Self::HierarchicalJpeg => write!(f, "hierarchical JPEG (SOF5/SOF6/SOF7)"),
            Self::JpegPrecision { bits } => {
                write!(f, "JPEG sample precision {bits} (only 8-bit is supported)")
            }
            Self::JpegComponentCount { components } => {
                write!(
                    f,
                    "JPEG component count {components} (only 1 or 3 are supported)"
                )
            }
            Self::JpegChromaSubsampling => {
                write!(f, "JPEG chroma subsampling that exceeds luma sampling rate")
            }
        }
    }
}

/// Errors that can occur during image decoding.
#[derive(Debug)]
pub enum CodecError {
    /// Image dimensions exceed the tensor's capacity.
    InsufficientCapacity {
        /// (width, height) of the decoded image.
        image: (usize, usize),
        /// (width, height) capacity of the destination tensor.
        tensor: (usize, usize),
    },
    /// The tensor's element type is not supported for image decoding.
    UnsupportedDtype(edgefirst_tensor::DType),
    /// The requested pixel format is not supported for this image type.
    UnsupportedFormat(edgefirst_tensor::PixelFormat),
    /// The image data is well-formed but uses a feature this decoder does
    /// not implement. Callers can pattern-match on the carried
    /// [`UnsupportedFeature`] to react programmatically (e.g. transcode
    /// before retry, surface a precise message, or skip gracefully).
    Unsupported(UnsupportedFeature),
    /// The image data is corrupted, truncated, or not a recognized format.
    InvalidData(String),
    /// I/O error reading image data.
    Io(std::io::Error),
    /// An error from the tensor subsystem.
    Tensor(edgefirst_tensor::Error),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientCapacity { image, tensor } => write!(
                f,
                "image dimensions {}×{} exceed tensor capacity {}×{}",
                image.0, image.1, tensor.0, tensor.1,
            ),
            Self::UnsupportedDtype(dt) => {
                write!(f, "unsupported tensor dtype {dt:?} for image decode")
            }
            Self::UnsupportedFormat(pf) => {
                write!(f, "unsupported pixel format {pf:?} for image decode")
            }
            Self::Unsupported(feat) => write!(f, "unsupported image feature: {feat}"),
            Self::InvalidData(msg) => write!(f, "invalid image data: {msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Tensor(e) => write!(f, "tensor error: {e}"),
        }
    }
}

impl std::error::Error for CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CodecError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<edgefirst_tensor::Error> for CodecError {
    fn from(e: edgefirst_tensor::Error) -> Self {
        Self::Tensor(e)
    }
}

impl From<zune_png::error::PngDecodeErrors> for CodecError {
    fn from(e: zune_png::error::PngDecodeErrors) -> Self {
        Self::InvalidData(format!("PNG: {e}"))
    }
}
