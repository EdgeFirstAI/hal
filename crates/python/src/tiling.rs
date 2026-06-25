// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for SAHI-style tiled inference: the input-side tiling grid
//! and config (`edgefirst_hal::image`), and the output-side per-tile lift /
//! cross-tile merge / streaming accumulator (`edgefirst_hal::decoder::tiling`).
//!
//! Detections cross the FFI boundary as the canonical numpy triple
//! ``(bbox (N,4) float32, scores (N,) float32, classes (N,) uintp)`` — never a
//! ``DetectBox`` class — matching the rest of the HAL Python API.

use crate::decoder::convert_detect_box;
use crate::image::{numpy_to_detect_boxes, Error, PyRegion, Result};
use edgefirst_hal::{
    decoder::tiling::{
        lift_tile_boxes, merge_tiled_detections, MatchMetric, MergeConfig, TilePlacement,
        TiledFrameAccumulator,
    },
    image::{tile_grid, Fit, TileSpec, TilingConfig},
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// How a tile crop is fit into the model input.
///
/// - ``Stretch`` — stretch the crop to fill the model input (identity for the
///   full-square tiles the grid produces; the hot path).
/// - ``Letterbox`` — preserve aspect ratio and pad with the
///   :class:`TilingConfig` ``pad`` colour.
#[pyclass(name = "Fit", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyFit {
    Stretch,
    Letterbox,
}

/// Overlap metric used by the tiled-detection merge.
///
/// - ``Iou`` — Intersection-over-Union (standard NMS metric).
/// - ``Ios`` — Intersection-over-Smaller (default): merges a seam-split
///   fragment into the whole object where IoU would leave duplicates.
#[pyclass(name = "MatchMetric", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyMatchMetric {
    Iou,
    #[default]
    Ios,
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

/// Static tiling configuration for one model. Independent of frame size.
#[pyclass(name = "TilingConfig", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyTilingConfig(pub(crate) TilingConfig);

#[pymethods]
impl PyTilingConfig {
    #[new]
    #[pyo3(signature = (tile_w, tile_h, overlap = 0.2, fit = PyFit::Stretch, pad = (114, 114, 114, 255)))]
    pub fn new(
        tile_w: usize,
        tile_h: usize,
        overlap: f32,
        fit: PyFit,
        pad: (u8, u8, u8, u8),
    ) -> Self {
        let pad = [pad.0, pad.1, pad.2, pad.3];
        let fit = match fit {
            PyFit::Stretch => Fit::Stretch,
            PyFit::Letterbox => Fit::Letterbox { pad },
        };
        let cfg = TilingConfig {
            pad,
            ..TilingConfig::new(tile_w, tile_h)
                .with_overlap(overlap)
                .with_fit(fit)
        };
        PyTilingConfig(cfg)
    }

    /// Set the minimum overlap ratio (chainable).
    pub fn with_overlap(&self, overlap: f32) -> Self {
        PyTilingConfig(self.0.with_overlap(overlap))
    }

    /// Set the fit mode (chainable). ``Fit.Letterbox`` uses the configured pad.
    pub fn with_fit(&self, fit: PyFit) -> Self {
        let fit = match fit {
            PyFit::Stretch => Fit::Stretch,
            PyFit::Letterbox => Fit::Letterbox { pad: self.0.pad },
        };
        PyTilingConfig(self.0.with_fit(fit))
    }

    #[getter]
    fn tile_w(&self) -> usize {
        self.0.tile_w
    }

    #[getter]
    fn tile_h(&self) -> usize {
        self.0.tile_h
    }

    #[getter]
    fn overlap(&self) -> f32 {
        self.0.overlap_ratio
    }

    #[getter]
    fn fit(&self) -> PyFit {
        match self.0.fit {
            Fit::Stretch => PyFit::Stretch,
            Fit::Letterbox { .. } => PyFit::Letterbox,
        }
    }

    #[getter]
    fn pad(&self) -> (u8, u8, u8, u8) {
        let [r, g, b, a] = self.0.pad;
        (r, g, b, a)
    }

    fn __repr__(&self) -> String {
        let fit = match self.0.fit {
            Fit::Stretch => "Fit.Stretch",
            Fit::Letterbox { .. } => "Fit.Letterbox",
        };
        format!(
            "TilingConfig(tile_w={}, tile_h={}, overlap={}, fit={fit})",
            self.0.tile_w, self.0.tile_h, self.0.overlap_ratio
        )
    }
}

/// One tile's native-frame crop rectangle and its grid coordinates.
#[pyclass(name = "TileSpec", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyTileSpec(pub(crate) TileSpec);

#[pymethods]
impl PyTileSpec {
    /// Native crop in full-frame pixels as a :class:`Region`.
    #[getter]
    fn source(&self) -> PyRegion {
        PyRegion {
            x: self.0.source.x,
            y: self.0.source.y,
            width: self.0.source.width,
            height: self.0.source.height,
        }
    }

    /// Row-major flat tile index, ``0..count``.
    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }

    /// Grid row.
    #[getter]
    fn row(&self) -> usize {
        self.0.row
    }

    /// Grid column.
    #[getter]
    fn col(&self) -> usize {
        self.0.col
    }

    fn __repr__(&self) -> String {
        format!(
            "TileSpec(source=Region(x={}, y={}, width={}, height={}), index={}, row={}, col={})",
            self.0.source.x,
            self.0.source.y,
            self.0.source.width,
            self.0.source.height,
            self.0.index,
            self.0.row,
            self.0.col,
        )
    }
}

/// How one tile was cut from the full frame and fed to the model — the shared
/// input↔output contract produced by the tiling grid and consumed by
/// :func:`lift_tile_boxes` and :class:`TiledFrameAccumulator`. All fields are
/// full-frame **pixels** except ``letterbox`` (normalized model-input bounds).
#[pyclass(name = "TilePlacement", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyTilePlacement(pub(crate) TilePlacement);

#[pymethods]
impl PyTilePlacement {
    #[new]
    #[pyo3(signature = (index, count, origin, crop_size, frame_dims, letterbox = None))]
    pub fn new(
        index: usize,
        count: usize,
        origin: (f32, f32),
        crop_size: (f32, f32),
        frame_dims: (f32, f32),
        letterbox: Option<[f32; 4]>,
    ) -> Self {
        PyTilePlacement(TilePlacement {
            index,
            count,
            origin,
            crop_size,
            letterbox,
            frame_dims,
        })
    }

    /// Tile index within the frame grid, ``0..count``.
    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }

    /// Total tiles for this frame (the streaming fan-in fence).
    #[getter]
    fn count(&self) -> usize {
        self.0.count
    }

    /// Native crop origin ``(ox, oy)`` in full-frame pixels.
    #[getter]
    fn origin(&self) -> (f32, f32) {
        self.0.origin
    }

    /// Native crop size ``(cw, ch)`` in full-frame pixels.
    #[getter]
    fn crop_size(&self) -> (f32, f32) {
        self.0.crop_size
    }

    /// Normalized letterbox content bounds ``(lx0, ly0, lx1, ly1)`` on the
    /// model input, or ``None`` when the crop was stretched to fill it.
    #[getter]
    fn letterbox(&self) -> Option<(f32, f32, f32, f32)> {
        self.0.letterbox.map(|[a, b, c, d]| (a, b, c, d))
    }

    /// Full-frame dimensions ``(frame_w, frame_h)`` in pixels.
    #[getter]
    fn frame_dims(&self) -> (f32, f32) {
        self.0.frame_dims
    }

    fn __repr__(&self) -> String {
        format!(
            "TilePlacement(index={}, count={}, origin={:?}, crop_size={:?}, letterbox={:?}, frame_dims={:?})",
            self.0.index, self.0.count, self.0.origin, self.0.crop_size, self.0.letterbox, self.0.frame_dims,
        )
    }
}

/// Configuration for the tiled-detection merge (GREEDYNMM).
#[pyclass(name = "MergeConfig", from_py_object)]
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
#[pyclass(name = "TiledFrameAccumulator")]
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

    /// Lift one tile's per-tile-decoded boxes to full-frame pixels and append
    /// them. Idempotent per ``placement.index``; returns ``True`` if newly
    /// accepted, ``False`` for a duplicate/out-of-range tile.
    pub fn push_tile(
        &mut self,
        bbox: PyReadonlyArray2<f32>,
        scores: PyReadonlyArray1<f32>,
        classes: PyReadonlyArray1<usize>,
        placement: &PyTilePlacement,
    ) -> Result<bool> {
        let acc = self
            .0
            .as_mut()
            .ok_or_else(|| Error::InvalidArg("accumulator already finalized".to_string()))?;
        let dets = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
        Ok(acc.push_tile(dets, &placement.0))
    }

    /// True once every tile of the frame has been pushed (by distinct index).
    pub fn is_complete(&self) -> Result<bool> {
        let acc = self
            .0
            .as_ref()
            .ok_or_else(|| Error::InvalidArg("accumulator already finalized".to_string()))?;
        Ok(acc.is_complete())
    }

    /// Tiles still outstanding.
    pub fn remaining(&self) -> Result<usize> {
        let acc = self
            .0
            .as_ref()
            .ok_or_else(|| Error::InvalidArg("accumulator already finalized".to_string()))?;
        Ok(acc.remaining())
    }

    /// Merge all accumulated detections into full-frame **pixel** boxes and
    /// return the numpy detection triple. Consumes the accumulator.
    pub fn finalize<'py>(&mut self, py: Python<'py>) -> PyResult<crate::decoder::PyDetOutput<'py>> {
        let acc = self.0.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("accumulator already finalized")
        })?;
        Ok(convert_detect_box(py, &acc.finalize()))
    }

    /// Merge then renormalize to ``[0, 1]`` by ``frame_dims`` (the non-tiled
    /// normalized-detection contract). Returns the numpy triple. Consumes the
    /// accumulator.
    pub fn finalize_normalized<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<crate::decoder::PyDetOutput<'py>> {
        let acc = self.0.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("accumulator already finalized")
        })?;
        Ok(convert_detect_box(py, &acc.finalize_normalized()))
    }
}

/// Uniform overlapping EvenDist tile grid covering a ``frame_w``×``frame_h``
/// frame. Row-major. Every tile is full-size unless the frame is smaller than
/// the tile on an axis, in which case that axis yields a single whole-frame
/// crop.
///
/// Python args are **width-first** to match :class:`Region` and
/// :meth:`ImageProcessor.create_image`.
///
/// Raises ``RuntimeError`` if ``overlap`` is not in ``[0.0, 1.0)`` or a tile
/// dimension is zero — the same validation as :class:`TilingConfig` and the C
/// API (which returns ``EINVAL``), preventing a degenerate config from
/// generating an enormous grid.
#[pyfunction]
#[pyo3(name = "tile_grid", signature = (frame_w, frame_h, tile_w, tile_h, overlap = 0.2))]
pub fn py_tile_grid(
    frame_w: usize,
    frame_h: usize,
    tile_w: usize,
    tile_h: usize,
    overlap: f32,
) -> Result<Vec<PyTileSpec>> {
    TilingConfig::new(tile_w, tile_h)
        .with_overlap(overlap)
        .validate()?;
    Ok(tile_grid(frame_h, frame_w, tile_h, tile_w, overlap)
        .into_iter()
        .map(PyTileSpec)
        .collect())
}

/// Lift tile-local **normalized** ``[0,1]`` xyxy detections (over the model
/// input) to full-frame **pixel** xyxy, inverting the letterbox if present.
/// Takes and returns the numpy detection triple.
#[pyfunction]
#[pyo3(name = "lift_tile_boxes")]
pub fn py_lift_tile_boxes<'py>(
    py: Python<'py>,
    bbox: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    classes: PyReadonlyArray1<usize>,
    placement: &PyTilePlacement,
) -> Result<crate::decoder::PyDetOutput<'py>> {
    let dets = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
    Ok(convert_detect_box(py, &lift_tile_boxes(dets, &placement.0)))
}

/// Greedy Non-Max **Merge** of lifted full-frame detections (GREEDYNMM). Takes
/// and returns the numpy detection triple.
#[pyfunction]
#[pyo3(name = "merge_tiled_detections")]
pub fn py_merge_tiled_detections<'py>(
    py: Python<'py>,
    bbox: PyReadonlyArray2<f32>,
    scores: PyReadonlyArray1<f32>,
    classes: PyReadonlyArray1<usize>,
    cfg: &PyMergeConfig,
) -> Result<crate::decoder::PyDetOutput<'py>> {
    let dets = numpy_to_detect_boxes(&bbox, &scores, &classes)?;
    Ok(convert_detect_box(
        py,
        &merge_tiled_detections(dets, &cfg.0),
    ))
}
