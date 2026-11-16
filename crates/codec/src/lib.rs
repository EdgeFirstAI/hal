// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image codec for decoding JPEG/PNG into pre-allocated EdgeFirst tensors,
//! and hardware video encoding and decoding in [`video`].
//!
//! Every codec runs on an [`Engine`]; [`Engine::Auto`] picks one for the
//! platform, and [`capabilities`] lists what the engines here can do.
//!
//! This crate provides the [`ImageDecoder`] struct and [`ImageLoad`] extension
//! trait for decoding image files into existing tensor buffers, including
//! DMA-BUF tensors with GPU-aligned pitch padding. A tensor allocated at the
//! maximum expected image size is reused by [`ImageLoad::load_image`] for
//! every image.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use edgefirst_codec::{ImageDecoder, ImageLoad};
//! use edgefirst_tensor::{CpuAccess, Tensor, TensorTrait, TensorMemory, PixelFormat};
//!
//! // Allocate once, sized for the largest expected image.
//! let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12, Some(TensorMemory::Mem),
//!                                       CpuAccess::Write)
//!     .expect("allocation");
//! let mut decoder = ImageDecoder::new();
//!
//! // Decode. The decoder sets the tensor's dimensions and native format.
//! let jpeg_bytes = std::fs::read("frame.jpg").unwrap();
//! let info = tensor.load_image(&mut decoder, &jpeg_bytes).expect("decode");
//! assert!(info.width <= 1920);
//! assert!(info.height <= 1080);
//! ```
//!
//! # Performance
//!
//! Tensors allocated with
//! [`ImageProcessor::create_image()`](https://docs.rs/edgefirst-image/latest/edgefirst_image/struct.ImageProcessor.html#method.create_image)
//! use the platform's memory backend (DMA → PBO → Mem) with GPU-aligned
//! pitch. Free-standing tensors cannot use PBO and may not have aligned
//! pitch.
//!
//! # Strided Buffers
//!
//! The decoder respects `effective_row_stride()` on the destination tensor:
//! decoded pixel rows are written at the tensor's stride offsets.

mod decoder;
mod encoder;
mod engine;
mod error;
mod exif;
mod jpeg;
mod options;
mod pixel;
#[cfg(feature = "png")]
mod png;
#[cfg(test)]
mod test_support;
mod traits;
pub mod video;

pub use decoder::{peek_info, ChromaUpsample, DctMethod, ImageDecoder};
pub use encoder::{ImageEncoder, DEFAULT_JPEG_QUALITY};
pub use engine::{capabilities, Capability, Codec, Direction, Engine, Transfer, ENGINE_ENV};
pub use error::{CodecError, UnsupportedFeature};
pub use options::ImageInfo;
pub use pixel::ImagePixel;
pub use traits::ImageLoad;

/// Returns `true` when the nvJPEG GPU JPEG decoder is available at runtime:
/// Linux with CUDA and libnvjpeg loaded, and opted in via
/// `EDGEFIRST_ENABLE_NVJPEG` (off by default). When `true` the codec prefers
/// it over the V4L2/CPU backends for CUDA-backed destinations. Always `false`
/// on other platforms or when the `nvjpeg` feature is disabled.
pub fn nvjpeg_available() -> bool {
    #[cfg(all(target_os = "linux", feature = "nvjpeg"))]
    {
        jpeg::nvjpeg_available()
    }
    #[cfg(not(all(target_os = "linux", feature = "nvjpeg")))]
    {
        false
    }
}

/// Returns `true` when a V4L2 memory-to-memory JPEG decoder is present and
/// not opted out via `EDGEFIRST_DISABLE_V4L2`. Opens and drops the device
/// once. Always `false` on other platforms or when the `v4l2` feature is
/// disabled.
pub fn v4l2_available() -> bool {
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    {
        jpeg::v4l2_available()
    }
    #[cfg(not(all(target_os = "linux", feature = "v4l2")))]
    {
        false
    }
}

/// Result type for codec operations.
pub type Result<T> = std::result::Result<T, CodecError>;
