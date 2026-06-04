// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! EdgeFirst HAL C API
//!
//! This crate provides C bindings for the EdgeFirst Hardware Abstraction Layer,
//! enabling zero-copy tensor operations, hardware-accelerated image processing,
//! ML model output decoding, and multi-object tracking from C/C++ applications.

#![allow(clippy::missing_safety_doc)]
#![allow(unsafe_op_in_unsafe_fn)]

/// Retained constructor: installs the coverage flush-on-abort handler for this
/// crate's instrumented shared library. See `edgefirst_tensor::covguard`. Only
/// present under coverage on Linux (`.init_array` is ELF-only; flush is Linux-only).
#[cfg(all(coverage, target_os = "linux"))]
#[used]
#[link_section = ".init_array"]
static __EDGEFIRST_COV_INSTALL: extern "C" fn() = {
    extern "C" fn ctor() {
        edgefirst_tensor::covguard::install();
    }
    ctor
};

mod colorimetry;
mod decoder;
mod delegate;
mod error;
mod image;
mod log;
mod tensor;
mod trace;
mod tracker;

pub use colorimetry::*;
pub use decoder::*;
pub use delegate::*;
pub use error::*;
pub use image::*;
pub use log::*;
pub use tensor::*;
#[cfg(feature = "tracing")]
pub use trace::*;
pub use tracker::*;
