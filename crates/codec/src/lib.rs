// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image codec for decoding JPEG/PNG into pre-allocated EdgeFirst tensors.
//!
//! This crate provides the [`ImageDecoder`] struct and [`ImageLoad`] extension
//! trait for decoding image files directly into existing tensor buffers —
//! including DMA-BUF tensors with GPU-aligned pitch padding. The core design
//! goal is **allocate once, decode many**: users create a tensor at the maximum
//! expected image size during program initialisation, then call
//! [`ImageLoad::load_image`] in the hot loop with zero per-frame allocations.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use edgefirst_codec::{ImageDecoder, ImageLoad, DecodeOptions};
//! use edgefirst_tensor::{Tensor, TensorTrait, TensorMemory, PixelFormat};
//!
//! // Allocate once (at init)
//! let mut tensor = Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem))
//!     .expect("allocation");
//! let mut decoder = ImageDecoder::new();
//!
//! // Decode many (in hot loop)
//! let jpeg_bytes = std::fs::read("frame.jpg").unwrap();
//! let info = tensor.load_image(&mut decoder, &jpeg_bytes, &DecodeOptions::default())
//!     .expect("decode");
//! assert!(info.width <= 1920);
//! assert!(info.height <= 1080);
//! ```
//!
//! # Performance
//!
//! For best performance, allocate tensors via
//! [`ImageProcessor::create_image()`](https://docs.rs/edgefirst-image/latest/edgefirst_image/struct.ImageProcessor.html#method.create_image)
//! which selects the optimal memory backend (DMA → PBO → Mem) with
//! GPU-aligned pitch. Free-standing tensors work but cannot use PBO
//! and may not have aligned pitch.
//!
//! # Strided Buffers
//!
//! The decoder respects `effective_row_stride()` on the destination tensor.
//! When a DMA tensor has GPU-aligned pitch padding, decoded pixel rows are
//! written at the correct stride offsets — no intermediate contiguous buffer
//! is exposed to the caller.

mod decoder;
mod error;
mod jpeg;
mod options;
mod pixel;
mod png;
mod traits;

pub use decoder::ImageDecoder;
pub use error::CodecError;
pub use options::{DecodeOptions, ImageInfo};
pub use pixel::ImagePixel;
pub use traits::ImageLoad;

/// Result type for codec operations.
pub type Result<T> = std::result::Result<T, CodecError>;
