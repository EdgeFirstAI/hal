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

// Rs→Py is fallible: the Rust enums are `#[non_exhaustive]`, so a variant added
// in a newer core may have no Python binding. Map it to `Err` and let the
// getters surface it as `None` (unknown) rather than panicking.
impl TryFrom<ColorSpace> for PyColorSpace {
    type Error = ();
    fn try_from(v: ColorSpace) -> Result<Self, ()> {
        match v {
            ColorSpace::Bt709 => Ok(PyColorSpace::Bt709),
            ColorSpace::Bt2020 => Ok(PyColorSpace::Bt2020),
            ColorSpace::Srgb => Ok(PyColorSpace::Srgb),
            ColorSpace::Smpte170m => Ok(PyColorSpace::Smpte170m),
            _ => Err(()),
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

impl TryFrom<ColorTransfer> for PyColorTransfer {
    type Error = ();
    fn try_from(v: ColorTransfer) -> Result<Self, ()> {
        match v {
            ColorTransfer::Bt709 => Ok(PyColorTransfer::Bt709),
            ColorTransfer::Srgb => Ok(PyColorTransfer::Srgb),
            ColorTransfer::Pq => Ok(PyColorTransfer::Pq),
            ColorTransfer::Hlg => Ok(PyColorTransfer::Hlg),
            ColorTransfer::Linear => Ok(PyColorTransfer::Linear),
            _ => Err(()),
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

impl TryFrom<ColorEncoding> for PyColorEncoding {
    type Error = ();
    fn try_from(v: ColorEncoding) -> Result<Self, ()> {
        match v {
            ColorEncoding::Bt601 => Ok(PyColorEncoding::Bt601),
            ColorEncoding::Bt709 => Ok(PyColorEncoding::Bt709),
            ColorEncoding::Bt2020 => Ok(PyColorEncoding::Bt2020),
            _ => Err(()),
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

impl TryFrom<ColorRange> for PyColorRange {
    type Error = ();
    fn try_from(v: ColorRange) -> Result<Self, ()> {
        match v {
            ColorRange::Full => Ok(PyColorRange::Full),
            ColorRange::Limited => Ok(PyColorRange::Limited),
            _ => Err(()),
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

    /// Build from the four raw V4L2 colorimetry integers. A ``DEFAULT`` (0)
    /// ``ycbcr_enc``/``quant`` is resolved from the colorspace (e.g.
    /// ``V4L2_COLORSPACE_JPEG`` → BT.601 full-range) per the kernel
    /// ``V4L2_MAP_*_DEFAULT`` rules; an unrecognised value maps to ``None``.
    #[staticmethod]
    fn from_v4l2(colorspace: u32, xfer: u32, ycbcr_enc: u32, quant: u32) -> Self {
        PyColorimetry(RsColorimetry::from_v4l2(colorspace, xfer, ycbcr_enc, quant))
    }

    /// Color primaries, or ``None`` if undefined (or a variant with no Python
    /// binding — see the `TryFrom` impls).
    #[getter]
    fn space(&self) -> Option<PyColorSpace> {
        self.0.space.and_then(|v| PyColorSpace::try_from(v).ok())
    }

    /// Transfer function, or ``None`` if undefined.
    #[getter]
    fn transfer(&self) -> Option<PyColorTransfer> {
        self.0
            .transfer
            .and_then(|v| PyColorTransfer::try_from(v).ok())
    }

    /// YCbCr encoding matrix, or ``None`` if undefined.
    #[getter]
    fn encoding(&self) -> Option<PyColorEncoding> {
        self.0
            .encoding
            .and_then(|v| PyColorEncoding::try_from(v).ok())
    }

    /// Quantization range, or ``None`` if undefined.
    #[getter]
    fn range(&self) -> Option<PyColorRange> {
        self.0.range.and_then(|v| PyColorRange::try_from(v).ok())
    }

    fn __repr__(&self) -> String {
        format!(
            "Colorimetry(space={:?}, transfer={:?}, encoding={:?}, range={:?})",
            self.0.space, self.0.transfer, self.0.encoding, self.0.range
        )
    }
}
