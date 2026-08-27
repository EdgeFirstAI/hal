// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.tracker` — ByteTrack multi-object tracking.

use edgefirst_python_common::tracker;
use pyo3::prelude::*;

#[pymodule]
fn _tracker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    m.add_class::<tracker::PyTrackInfo>()?;
    m.add_class::<tracker::PyByteTrack>()?;
    m.add_class::<tracker::PyActiveTrackInfo>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
