// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Input-side tiling: grid geometry and config (`edgefirst_image`).
//! Cross-tile merge / lift / accumulator live on `edgefirst.decoder`.

use crate::image::Result;
use crate::tensor::PyRegion;
use edgefirst_image::{tile_grid, Fit, TilePlacement, TileSpec, TilingConfig};
use pyo3::prelude::*;

/// How a tile crop is fit into the model input.
///
/// - ``Stretch`` — stretch the crop to fill the model input (identity for the
///   full-square tiles the grid produces; the hot path).
/// - ``Letterbox`` — preserve aspect ratio and pad with the
///   :class:`TilingConfig` ``pad`` colour.
#[pyclass(name = "Fit", eq, eq_int, from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyFit {
    Stretch,
    Letterbox,
}

#[pymethods]
impl PyFit {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

/// Static tiling configuration for one model. Independent of frame size.
#[pyclass(name = "TilingConfig", from_py_object, module = "edgefirst.image")]
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
#[pyclass(name = "TileSpec", from_py_object, module = "edgefirst.image")]
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

/// How one tile was cut from the full frame and fed to the model. Produced by
/// the tiling grid; consumed by :func:`edgefirst.decoder.lift_tile_boxes`.
/// All fields are full-frame **pixels** except ``letterbox``.
#[pyclass(name = "TilePlacement", from_py_object, module = "edgefirst.image")]
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

    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }

    #[getter]
    fn count(&self) -> usize {
        self.0.count
    }

    #[getter]
    fn origin(&self) -> (f32, f32) {
        self.0.origin
    }

    #[getter]
    fn crop_size(&self) -> (f32, f32) {
        self.0.crop_size
    }

    #[getter]
    fn letterbox(&self) -> Option<(f32, f32, f32, f32)> {
        self.0.letterbox.map(|[a, b, c, d]| (a, b, c, d))
    }

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

/// Uniform overlapping EvenDist tile grid covering a ``frame_w``×``frame_h``
/// frame. Row-major. Every tile is full-size unless the frame is smaller than
/// the tile on an axis, in which case that axis yields a single whole-frame
/// crop.
///
/// Python args are **width-first** to match :class:`Region` and
/// :meth:`ImageProcessor.create_image`.
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
