// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Input-side tiling: grid geometry and config (`edgefirst_image`).
//! Cross-tile merge / lift / accumulator live on `edgefirst.decoder`.

use crate::image::{PyConvertOptions, Result};
use crate::tensor::PyRegion;
use edgefirst_image::{tile_grid, TilePlacement, TileSpec, TilingConfig};
use pyo3::prelude::*;

/// Static tiling configuration for one model. Independent of frame size.
///
/// ``options`` is the :class:`ConvertOptions` every tile is converted with
/// (fit, letterbox placement and pad, normalization, interpolation), e.g.
/// from :meth:`ConvertOptions.from_spec`; each tile replaces the source crop
/// with its own. Options that rotate or flip are refused when tiles are
/// planned. The default stretches each tile with the default pixel mapping.
#[pyclass(name = "TilingConfig", from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy)]
pub struct PyTilingConfig(pub(crate) TilingConfig);

#[pymethods]
impl PyTilingConfig {
    #[new]
    #[pyo3(signature = (tile_w, tile_h, overlap = 0.2, options = None))]
    pub fn new(
        tile_w: usize,
        tile_h: usize,
        overlap: f32,
        options: Option<PyConvertOptions>,
    ) -> Self {
        let mut cfg = TilingConfig::new(tile_w, tile_h).with_overlap(overlap);
        if let Some(o) = options {
            cfg = cfg.with_options(o.0);
        }
        PyTilingConfig(cfg)
    }

    /// Set the minimum overlap ratio (chainable).
    pub fn with_overlap(&self, overlap: f32) -> Self {
        PyTilingConfig(self.0.with_overlap(overlap))
    }

    /// Set the options every tile is converted with (chainable).
    pub fn with_options(&self, options: PyConvertOptions) -> Self {
        PyTilingConfig(self.0.with_options(options.0))
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

    /// The options every tile is converted with.
    #[getter]
    fn options(&self) -> PyConvertOptions {
        PyConvertOptions(self.0.options())
    }

    fn __repr__(&self) -> String {
        format!(
            "TilingConfig(tile_w={}, tile_h={}, overlap={}, options={:?})",
            self.0.tile_w,
            self.0.tile_h,
            self.0.overlap_ratio,
            self.0.options()
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
/// All fields are full-frame **pixels** except ``content_region``, the
/// normalized ``(xmin, ymin, xmax, ymax)`` image content of a letterboxed
/// tile on the model input (``None`` when the tile was stretched).
#[pyclass(name = "TilePlacement", from_py_object, module = "edgefirst.image")]
#[derive(Debug, Clone, Copy)]
pub struct PyTilePlacement(pub(crate) TilePlacement);

#[pymethods]
impl PyTilePlacement {
    #[new]
    #[pyo3(signature = (index, count, origin, crop_size, frame_dims, content_region = None))]
    pub fn new(
        index: usize,
        count: usize,
        origin: (f32, f32),
        crop_size: (f32, f32),
        frame_dims: (f32, f32),
        content_region: Option<crate::content_region::ContentRegionArg>,
    ) -> Self {
        PyTilePlacement(TilePlacement {
            index,
            count,
            origin,
            crop_size,
            content_region: crate::content_region::normalized(content_region),
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
    fn content_region(&self) -> Option<(f32, f32, f32, f32)> {
        self.0.content_region.map(|[a, b, c, d]| (a, b, c, d))
    }

    #[getter]
    fn frame_dims(&self) -> (f32, f32) {
        self.0.frame_dims
    }

    fn __repr__(&self) -> String {
        format!(
            "TilePlacement(index={}, count={}, origin={:?}, crop_size={:?}, content_region={:?}, frame_dims={:?})",
            self.0.index, self.0.count, self.0.origin, self.0.crop_size, self.0.content_region, self.0.frame_dims,
        )
    }

    /// Constructor args as a single tuple so a decoder in another extension
    /// can extract placement without six cross-module ``getattr`` hops.
    // The tuple IS the pickle constructor arg list `__new__` expects; a type
    // alias would only rename it, not simplify it.
    #[allow(clippy::type_complexity)]
    fn __getnewargs__(
        &self,
    ) -> (
        usize,
        usize,
        (f32, f32),
        (f32, f32),
        (f32, f32),
        Option<[f32; 4]>,
    ) {
        (
            self.0.index,
            self.0.count,
            self.0.origin,
            self.0.crop_size,
            self.0.frame_dims,
            self.0.content_region,
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
