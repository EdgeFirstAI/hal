// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.image` — hardware-accelerated conversion, resize and tiling.

use edgefirst_python_common::{colorimetry, image, tensor, tiling};
use pyo3::prelude::*;

#[pymodule]
fn _image(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<image::PyPixelFormat>()?;
    m.add_class::<tensor::PyTensorMemory>()?;
    m.add_class::<image::Normalization>()?;
    m.add_class::<image::PyRegion>()?;
    m.add_class::<colorimetry::PyColorimetry>()?;
    m.add_class::<colorimetry::PyColorSpace>()?;
    m.add_class::<colorimetry::PyColorTransfer>()?;
    m.add_class::<colorimetry::PyColorEncoding>()?;
    m.add_class::<colorimetry::PyColorRange>()?;
    m.add_class::<image::PyRotation>()?;
    m.add_class::<image::PyFlip>()?;
    m.add_class::<image::PyColorMode>()?;
    m.add_class::<image::PyMaskResolution>()?;
    m.add_class::<image::PyImageProcessor>()?;
    m.add_class::<image::PyEglDisplayKind>()?;
    m.add_function(wrap_pyfunction!(image::align_width_for_gpu_pitch, m)?)?;
    m.add_function(wrap_pyfunction!(image::align_width_for_pixel_format, m)?)?;
    m.add_function(wrap_pyfunction!(
        image::gpu_dma_buf_pitch_alignment_bytes,
        m
    )?)?;
    #[cfg(target_os = "linux")]
    {
        m.add_class::<image::PyEglDisplayInfo>()?;
        m.add_function(wrap_pyfunction!(image::probe_egl_displays, m)?)?;
    }

    m.add_class::<tiling::PyFit>()?;
    m.add_class::<tiling::PyTilingConfig>()?;
    m.add_class::<tiling::PyTileSpec>()?;
    m.add_class::<tiling::PyTilePlacement>()?;
    m.add_function(wrap_pyfunction!(tiling::py_tile_grid, m)?)?;

    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
