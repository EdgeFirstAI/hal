// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `edgefirst_tensor::Colorimetry` and its four axis
//! enums (`ColorSpace`, `ColorTransfer`, `ColorEncoding`, `ColorRange`).
//!
//! Each axis is optional; `None` means "undefined" and is never auto-filled.

use edgefirst_hal::tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry as RsColorimetry,
};
use pyo3::prelude::*;

// ─── Axis enums ──────────────────────────────────────────────────────────────

/// Color primaries (`color_space` in the EdgeFirst schema).
#[pyclass(name = "ColorSpace", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyColorSpace {
    Bt709,
    Bt2020,
    Srgb,
    Smpte170m,
}

impl From<PyColorSpace> for ColorSpace {
    fn from(v: PyColorSpace) -> Self {
        match v {
            PyColorSpace::Bt709 => ColorSpace::Bt709,
            PyColorSpace::Bt2020 => ColorSpace::Bt2020,
            PyColorSpace::Srgb => ColorSpace::Srgb,
            PyColorSpace::Smpte170m => ColorSpace::Smpte170m,
        }
    }
}

impl From<ColorSpace> for PyColorSpace {
    fn from(v: ColorSpace) -> Self {
        match v {
            ColorSpace::Bt709 => PyColorSpace::Bt709,
            ColorSpace::Bt2020 => PyColorSpace::Bt2020,
            ColorSpace::Srgb => PyColorSpace::Srgb,
            ColorSpace::Smpte170m => PyColorSpace::Smpte170m,
            _ => unreachable!("unmapped ColorSpace variant: {v:?}"),
        }
    }
}

/// Transfer function (`color_transfer` in the EdgeFirst schema).
#[pyclass(name = "ColorTransfer", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyColorTransfer {
    Bt709,
    Srgb,
    Pq,
    Hlg,
    Linear,
}

impl From<PyColorTransfer> for ColorTransfer {
    fn from(v: PyColorTransfer) -> Self {
        match v {
            PyColorTransfer::Bt709 => ColorTransfer::Bt709,
            PyColorTransfer::Srgb => ColorTransfer::Srgb,
            PyColorTransfer::Pq => ColorTransfer::Pq,
            PyColorTransfer::Hlg => ColorTransfer::Hlg,
            PyColorTransfer::Linear => ColorTransfer::Linear,
        }
    }
}

impl From<ColorTransfer> for PyColorTransfer {
    fn from(v: ColorTransfer) -> Self {
        match v {
            ColorTransfer::Bt709 => PyColorTransfer::Bt709,
            ColorTransfer::Srgb => PyColorTransfer::Srgb,
            ColorTransfer::Pq => PyColorTransfer::Pq,
            ColorTransfer::Hlg => PyColorTransfer::Hlg,
            ColorTransfer::Linear => PyColorTransfer::Linear,
            _ => unreachable!("unmapped ColorTransfer variant: {v:?}"),
        }
    }
}

/// YCbCr encoding matrix (`color_encoding` in the EdgeFirst schema).
#[pyclass(name = "ColorEncoding", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyColorEncoding {
    Bt601,
    Bt709,
    Bt2020,
}

impl From<PyColorEncoding> for ColorEncoding {
    fn from(v: PyColorEncoding) -> Self {
        match v {
            PyColorEncoding::Bt601 => ColorEncoding::Bt601,
            PyColorEncoding::Bt709 => ColorEncoding::Bt709,
            PyColorEncoding::Bt2020 => ColorEncoding::Bt2020,
        }
    }
}

impl From<ColorEncoding> for PyColorEncoding {
    fn from(v: ColorEncoding) -> Self {
        match v {
            ColorEncoding::Bt601 => PyColorEncoding::Bt601,
            ColorEncoding::Bt709 => PyColorEncoding::Bt709,
            ColorEncoding::Bt2020 => PyColorEncoding::Bt2020,
            _ => unreachable!("unmapped ColorEncoding variant: {v:?}"),
        }
    }
}

/// Quantization range (`color_range` in the EdgeFirst schema).
#[pyclass(name = "ColorRange", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyColorRange {
    Full,
    Limited,
}

impl From<PyColorRange> for ColorRange {
    fn from(v: PyColorRange) -> Self {
        match v {
            PyColorRange::Full => ColorRange::Full,
            PyColorRange::Limited => ColorRange::Limited,
        }
    }
}

impl From<ColorRange> for PyColorRange {
    fn from(v: ColorRange) -> Self {
        match v {
            ColorRange::Full => PyColorRange::Full,
            ColorRange::Limited => PyColorRange::Limited,
            _ => unreachable!("unmapped ColorRange variant: {v:?}"),
        }
    }
}

// ─── Colorimetry ─────────────────────────────────────────────────────────────

/// Colorimetry metadata: four optional axes (color primaries, transfer
/// function, YCbCr encoding, quantization range).
///
/// Each axis is `None` when undefined. Construct directly with keyword
/// arguments, or from raw V4L2 integers via ``from_v4l2``.
#[pyclass(name = "Colorimetry", from_py_object)]
#[derive(Debug, Clone, Default)]
pub struct PyColorimetry(pub(crate) RsColorimetry);

impl From<RsColorimetry> for PyColorimetry {
    fn from(c: RsColorimetry) -> Self {
        PyColorimetry(c)
    }
}

impl From<PyColorimetry> for RsColorimetry {
    fn from(c: PyColorimetry) -> Self {
        c.0
    }
}

#[pymethods]
impl PyColorimetry {
    #[new]
    #[pyo3(signature = (space = None, transfer = None, encoding = None, range = None))]
    fn new(
        space: Option<PyColorSpace>,
        transfer: Option<PyColorTransfer>,
        encoding: Option<PyColorEncoding>,
        range: Option<PyColorRange>,
    ) -> Self {
        PyColorimetry(RsColorimetry {
            space: space.map(Into::into),
            transfer: transfer.map(Into::into),
            encoding: encoding.map(Into::into),
            range: range.map(Into::into),
        })
    }

    /// Build from the four raw V4L2 colorimetry integers. Each ``0`` /
    /// unrecognised value maps to ``None`` on that axis.
    #[staticmethod]
    fn from_v4l2(colorspace: u32, xfer: u32, ycbcr_enc: u32, quant: u32) -> Self {
        PyColorimetry(RsColorimetry::from_v4l2(colorspace, xfer, ycbcr_enc, quant))
    }

    /// Color primaries, or ``None`` if undefined.
    #[getter]
    fn space(&self) -> Option<PyColorSpace> {
        self.0.space.map(Into::into)
    }

    /// Transfer function, or ``None`` if undefined.
    #[getter]
    fn transfer(&self) -> Option<PyColorTransfer> {
        self.0.transfer.map(Into::into)
    }

    /// YCbCr encoding matrix, or ``None`` if undefined.
    #[getter]
    fn encoding(&self) -> Option<PyColorEncoding> {
        self.0.encoding.map(Into::into)
    }

    /// Quantization range, or ``None`` if undefined.
    #[getter]
    fn range(&self) -> Option<PyColorRange> {
        self.0.range.map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "Colorimetry(space={:?}, transfer={:?}, encoding={:?}, range={:?})",
            self.0.space, self.0.transfer, self.0.encoding, self.0.range
        )
    }
}
