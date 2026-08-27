// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Output-side tiled-detection merge: lift, GREEDYNMM, streaming accumulator.
//! Registered on `edgefirst.decoder`. TilePlacement is accepted by attribute
//! so an `edgefirst.image.TilePlacement` works without sharing a PyO3 type.

use crate::detect_boxes::{convert_detect_box, numpy_to_detect_boxes, PyDetOutput};
use edgefirst_decoder::tiling::{
    lift_tile_boxes, merge_tiled_detections, MatchMetric, MergeConfig, TiledFrameAccumulator,
};
use edgefirst_decoder_abi::TilePlacement;
use edgefirst_tensor::{BoundingBox, DetectBox};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Overlap metric used by the tiled-detection merge.
///
/// - ``Iou`` — Intersection-over-Union (standard NMS metric).
/// - ``Ios`` — Intersection-over-Smaller (default): merges a seam-split
///   fragment into the whole object where IoU would leave duplicates.
#[pyclass(
    name = "MatchMetric",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.decoder"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyMatchMetric {
    Iou,
    #[default]
    Ios,
}

#[pymethods]
impl PyMatchMetric {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyMatchMetric> for MatchMetric {
    fn from(val: PyMatchMetric) -> Self {
        match val {
            PyMatchMetric::Iou => MatchMetric::Iou,
            PyMatchMetric::Ios => MatchMetric::Ios,
        }
    }
}

impl From<MatchMetric> for PyMatchMetric {
    fn from(val: MatchMetric) -> Self {
        match val {
            MatchMetric::Iou => PyMatchMetric::Iou,
            MatchMetric::Ios => PyMatchMetric::Ios,
        }
    }
}

fn extract_placement(obj: &Bound<'_, PyAny>) -> PyResult<TilePlacement> {
    if let Ok(packed) = obj.call_method0("__getnewargs__") {
        if let Ok((index, count, origin, crop_size, frame_dims, letterbox)) = packed.extract() {
            return Ok(TilePlacement {
                index,
                count,
                origin,
                crop_size,
                letterbox,
                frame_dims,
            });
        }
    }
    let letterbox = match obj
        .getattr("letterbox")?
        .extract::<Option<(f32, f32, f32, f32)>>()?
    {
        Some((a, b, c, d)) => Some([a, b, c, d]),
        None => None,
    };
    Ok(TilePlacement {
        index: obj.getattr("index")?.extract()?,
        count: obj.getattr("count")?.extract()?,
        origin: obj.getattr("origin")?.extract()?,
        crop_size: obj.getattr("crop_size")?.extract()?,
        letterbox,
        frame_dims: obj.getattr("frame_dims")?.extract()?,
    })
}

fn flat_to_detect_boxes(bbox: &[f32], scores: &[f32], classes: &[usize]) -> Vec<DetectBox> {
    scores
        .iter()
        .zip(classes)
        .enumerate()
        .map(|(i, (score, label))| {
            let o = i * 4;
            DetectBox {
                bbox: BoundingBox::new(bbox[o], bbox[o + 1], bbox[o + 2], bbox[o + 3]),
                score: *score,
                label: *label,
            }
        })
        .collect()
}

/// Configuration for the tiled-detection merge (GREEDYNMM).
#[pyclass(name = "MergeConfig", from_py_object, module = "edgefirst.decoder")]
#[derive(Debug, Clone, Copy)]
pub struct PyMergeConfig(pub(crate) MergeConfig);

#[pymethods]
impl PyMergeConfig {
    #[new]
    #[pyo3(signature = (metric = PyMatchMetric::Ios, threshold = 0.5, class_agnostic = false, max_det = 300, score_threshold = 0.0))]
    pub fn new(
        metric: PyMatchMetric,
        threshold: f32,
        class_agnostic: bool,
        max_det: usize,
        score_threshold: f32,
    ) -> Self {
        PyMergeConfig(MergeConfig {
            metric: metric.into(),
            threshold,
            class_agnostic,
            max_det,
            score_threshold,
        })
    }

    #[getter]
    fn metric(&self) -> PyMatchMetric {
        self.0.metric.into()
    }

    #[getter]
    fn threshold(&self) -> f32 {
        self.0.threshold
    }

    #[getter]
    fn class_agnostic(&self) -> bool {
        self.0.class_agnostic
    }

    #[getter]
    fn max_det(&self) -> usize {
        self.0.max_det
    }

    #[getter]
    fn score_threshold(&self) -> f32 {
        self.0.score_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "MergeConfig(metric={:?}, threshold={}, class_agnostic={}, max_det={}, score_threshold={})",
            self.0.metric, self.0.threshold, self.0.class_agnostic, self.0.max_det, self.0.score_threshold,
        )
    }
}

/// Streaming collector for one frame's tiled detections. Push each tile's
/// per-tile-decoded boxes as inference completes (any order), then finalize
/// once every tile has arrived. Not thread-safe; keep one per in-flight frame.
#[pyclass(name = "TiledFrameAccumulator", module = "edgefirst.decoder")]
pub struct PyTiledFrameAccumulator(Option<TiledFrameAccumulator>);

#[pymethods]
impl PyTiledFrameAccumulator {
    #[new]
    #[pyo3(signature = (frame_dims, tiles_total, cfg, est_per_tile = 16))]
    pub fn new(
        frame_dims: (f32, f32),
        tiles_total: usize,
        cfg: &PyMergeConfig,
        est_per_tile: usize,
    ) -> Self {
        PyTiledFrameAccumulator(Some(TiledFrameAccumulator::new(
            frame_dims,
            tiles_total,
            cfg.0,
            est_per_tile,
        )))
    }

    pub fn push_tile(
        &mut self,
        py: Python<'_>,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        placement: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if bbox.shape()[1] != 4 {
            return Err(PyValueError::new_err("bbox shape must be (N, 4)"));
        }
        if bbox.shape()[0] != scores.shape()[0] || bbox.shape()[0] != classes.shape()[0] {
            return Err(PyValueError::new_err(
                "bbox, scores, classes must have the same length",
            ));
        }
        // Memcpy out of the caller's numpy buffers (contiguous fast path)
        // so DetectBox construction and lift_tile_boxes run with the GIL
        // released. `to_owned()` on a numpy view iterates under the GIL
        // and is what kept `test_push_tile_releases_the_gil` at 68%.
        let bbox = match bbox.as_array().as_slice() {
            Some(s) => s.to_vec(),
            None => bbox.as_array().iter().copied().collect(),
        };
        let scores = match scores.as_array().as_slice() {
            Some(s) => s.to_vec(),
            None => scores.as_array().iter().copied().collect(),
        };
        let classes = match classes.as_array().as_slice() {
            Some(s) => s.to_vec(),
            None => classes.as_array().iter().copied().collect(),
        };
        let placement = extract_placement(placement)?;
        py.detach(move || {
            let dets = flat_to_detect_boxes(&bbox, &scores, &classes);
            let acc = self
                .0
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("accumulator already finalized"))?;
            Ok(acc.push_tile(dets, &placement))
        })
    }

    pub fn is_complete(&self) -> PyResult<bool> {
        let acc = self
            .0
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("accumulator already finalized"))?;
        Ok(acc.is_complete())
    }

    pub fn remaining(&self) -> PyResult<usize> {
        let acc = self
            .0
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("accumulator already finalized"))?;
        Ok(acc.remaining())
    }

    pub fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<PyDetOutput<'py>> {
        let acc = self
            .0
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("accumulator already finalized"))?;
        let merged = py.detach(move || acc.finalize());
        Ok(convert_detect_box(py, &merged))
    }

    pub fn finalize_normalized<'py>(&mut self, py: Python<'py>) -> PyResult<PyDetOutput<'py>> {
        let acc = self
            .0
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("accumulator already finalized"))?;
        let merged = py.detach(move || acc.finalize_normalized());
        Ok(convert_detect_box(py, &merged))
    }
}

/// Lift tile-local **normalized** ``[0,1]`` xyxy detections (over the model
/// input) to full-frame **pixel** xyxy, inverting the letterbox if present.
#[pyfunction]
#[pyo3(name = "lift_tile_boxes")]
pub fn py_lift_tile_boxes<'py>(
    py: Python<'py>,
    bbox: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    classes: PyReadonlyArray1<usize>,
    placement: &Bound<'py, PyAny>,
) -> PyResult<PyDetOutput<'py>> {
    let dets = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
    let placement = extract_placement(placement)?;
    Ok(convert_detect_box(py, &lift_tile_boxes(dets, &placement)))
}

/// Greedy Non-Max **Merge** of lifted full-frame detections (GREEDYNMM).
#[pyfunction]
#[pyo3(name = "merge_tiled_detections")]
pub fn py_merge_tiled_detections<'py>(
    py: Python<'py>,
    bbox: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    classes: PyReadonlyArray1<usize>,
    cfg: &PyMergeConfig,
) -> PyResult<PyDetOutput<'py>> {
    let dets = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
    let cfg = cfg.0;
    let merged = py.detach(move || merge_tiled_detections(dets, &cfg));
    Ok(convert_detect_box(py, &merged))
}
