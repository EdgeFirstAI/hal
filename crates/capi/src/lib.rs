// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! EdgeFirst HAL C API
//!
//! This crate provides C bindings for the EdgeFirst Hardware Abstraction Layer,
//! enabling zero-copy tensor operations, hardware-accelerated image processing,
//! ML model output decoding, and multi-object tracking from C/C++ applications.

#![allow(clippy::missing_safety_doc)]
#![allow(unsafe_op_in_unsafe_fn)]

mod decoder;
mod error;
mod image;
mod tensor;
mod tracker;

pub use decoder::*;
pub use error::*;
pub use image::*;
pub use tensor::*;
pub use tracker::*;
