// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(nightly, feature(f16))]

use pyo3::prelude::*;

pub mod decoder;
pub mod image;
pub mod tensor;

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

#[pymodule]
pub mod edgefirst_hal {
    pub use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(build_info, m)?)?;

        m.add_class::<tensor::PyTensor>()?;
        m.add_class::<image::FourCC>()?;
        m.add_class::<image::Normalization>()?;
        m.add_class::<image::PyRect>()?;
        m.add_class::<image::PyRotation>()?;
        m.add_class::<image::PyFlip>()?;
        m.add_class::<image::PyImageProcessor>()?;
        m.add_class::<image::PyTensorImage>()?;
        m.add_class::<image::PyTensorMemory>()?;
        m.add_class::<decoder::PyDecoder>()?;
        m.add_class::<decoder::PyNms>()?;

        Ok(())
    }

    #[pyfunction]
    fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Returns build configuration information including f16 implementation
    #[pyfunction]
    fn build_info() -> String {
        #[cfg(nightly)]
        let f16_impl = "native f16 (nightly, optimized)";
        #[cfg(not(nightly))]
        let f16_impl = "half::f16 (stable, compatible)";

        format!(
            "edgefirst-hal v{}\nf16 implementation: {}",
            env!("CARGO_PKG_VERSION"),
            f16_impl
        )
    }
}
