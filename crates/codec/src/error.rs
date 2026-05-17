// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Error types for the codec crate.

use std::fmt;

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
