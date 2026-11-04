// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.kernels` — quantized and floating-point compute kernels.

use edgefirst_python_common::kernels;
use pyo3::prelude::*;

#[pymodule]
fn _kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    m.add_class::<kernels::PySoftmaxI8Plan>()?;
    m.add_class::<kernels::PySoftmaxI16Plan>()?;
    m.add_class::<kernels::PySigmoidPlan>()?;
    m.add_class::<kernels::PyDflPlan>()?;
    m.add_class::<kernels::PyDist2BboxPlan>()?;
    m.add_class::<kernels::PyDflBoxPlan>()?;
    m.add_class::<kernels::PyLutI8Plan>()?;
    m.add_class::<kernels::PyBinaryI8Plan>()?;
    m.add_class::<kernels::PyMatMulI8Plan>()?;
    m.add_class::<kernels::PySoftmaxDotI8Plan>()?;
    m.add_class::<kernels::PyAttentionI8Plan>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
