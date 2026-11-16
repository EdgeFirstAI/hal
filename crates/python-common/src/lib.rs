// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Bindings shared by the EdgeFirst Python extension modules.
//!
//! This is an rlib: every extension module statically links its own copy of
//! the PyO3 wrapper types and has its own type objects. Cross-module tensor
//! handoff goes through the capsule protocol
//! ([`edgefirst_tensor::TensorDesc`]). The codec, image and decoder modules
//! link `libedgefirst_tensor.so`; the tracker module does not.

#![cfg_attr(nightly, feature(f16))]

#[cfg(feature = "codec")]
pub mod codec;
#[cfg(feature = "tensor")]
pub mod colorimetry;
#[cfg(any(feature = "image", feature = "decoder"))]
pub mod content_region;
#[cfg(feature = "decoder")]
pub mod decoder;
#[cfg(any(feature = "image", feature = "decoder"))]
pub mod detect_boxes;
#[cfg(feature = "image")]
pub mod image;
#[cfg(feature = "decoder")]
pub mod infer;
#[cfg(feature = "tensor")]
pub mod interop;
#[cfg(feature = "kernels")]
pub mod kernels;
#[cfg(feature = "decoder")]
pub mod model_config;
#[cfg(feature = "tensor")]
pub mod tensor;
#[cfg(feature = "image")]
pub mod tiling;
#[cfg(feature = "decoder")]
pub mod tiling_merge;
#[cfg(feature = "tracker")]
pub mod tracker;

/// Install the coverage flush-on-abort handler for an instrumented cdylib.
///
/// Each extension module must call this from its `#[pymodule_init]`.
#[cfg(all(coverage, target_os = "linux"))]
pub fn install_covguard() {
    #[cfg(feature = "tensor")]
    edgefirst_tensor::covguard::install();
}

/// No-op off Linux or without coverage instrumentation.
#[cfg(not(all(coverage, target_os = "linux")))]
pub fn install_covguard() {}

/// Initialise logging for an extension module. Uses `try_init`: the first
/// module to load installs the logger and later calls leave it in place.
pub fn init_logging() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
}

/// Common module bootstrap: logging, then coverage.
pub fn init_module() {
    init_logging();
    install_covguard();
}

pub struct FunctionTimer {
    name: String,
    start: std::time::Instant,
}

impl FunctionTimer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for FunctionTimer {
    fn drop(&mut self) {
        log::trace!("{} elapsed: {:?}", self.name, self.start.elapsed())
    }
}
