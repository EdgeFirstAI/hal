// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.codec` — JPEG/PNG decoding.

use edgefirst_python_common::{colorimetry, tensor};
use pyo3::prelude::*;

#[pymodule]
fn _codec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    // The decode target. Each extension carries its own copy of this class:
    // cross-module handoff uses the capsule protocol, not shared identity.
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tensor::PyImageInfo>()?;
    m.add_class::<tensor::PyDctMethod>()?;
    m.add_class::<tensor::PyPixelFormat>()?;
    m.add_class::<tensor::PyTensorMemory>()?;
    m.add_class::<tensor::PyRegion>()?;
    m.add_class::<colorimetry::PyColorimetry>()?;
    m.add_class::<colorimetry::PyColorSpace>()?;
    m.add_class::<colorimetry::PyColorTransfer>()?;
    m.add_class::<colorimetry::PyColorEncoding>()?;
    m.add_class::<colorimetry::PyColorRange>()?;

    m.add_function(wrap_pyfunction!(tensor::set_dct_method, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::set_output_format, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::is_v4l2_available, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::decode_file_into, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
