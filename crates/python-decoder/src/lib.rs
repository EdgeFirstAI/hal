// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.decoder` — YOLO/ModelPack output decoding, NMS and tracking.

use edgefirst_python_common::{colorimetry, decoder, infer, tensor, tiling_merge};
use pyo3::prelude::*;

#[pymodule]
fn _decoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    edgefirst_python_common::init_module();

    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tensor::PyPixelFormat>()?;
    m.add_class::<tensor::PyTensorMemory>()?;
    m.add_class::<tensor::PyRegion>()?;
    m.add_class::<colorimetry::PyColorimetry>()?;
    m.add_class::<colorimetry::PyColorSpace>()?;
    m.add_class::<colorimetry::PyColorTransfer>()?;
    m.add_class::<colorimetry::PyColorEncoding>()?;
    m.add_class::<colorimetry::PyColorRange>()?;
    m.add_class::<decoder::PyDecoder>()?;
    m.add_class::<decoder::PyProtoData>()?;
    m.add_class::<decoder::PyNms>()?;
    m.add_class::<decoder::PyDecoderType>()?;
    m.add_class::<decoder::PyDecoderVersion>()?;
    m.add_class::<decoder::PyDimName>()?;
    m.add_class::<decoder::PyOutput>()?;

    m.add_class::<tiling_merge::PyMatchMetric>()?;
    m.add_class::<tiling_merge::PyMergeConfig>()?;
    m.add_class::<tiling_merge::PyTiledFrameAccumulator>()?;
    m.add_function(wrap_pyfunction!(tiling_merge::py_lift_tile_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(
        tiling_merge::py_merge_tiled_detections,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(infer::py_infer_ultralytics_schema, m)?)?;

    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
