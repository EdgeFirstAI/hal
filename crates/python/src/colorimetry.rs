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

/// Generate a PyO3 mirror of a core `edgefirst_tensor` colorimetry enum plus
/// the bidirectional conversions. `From<Py> for Rs` is total; `TryFrom<Rs> for
/// Py` is fallible because the core enums are `#[non_exhaustive]` — a variant
/// added in a newer core has no Python binding, so it maps to `Err(())` (the
/// getters surface that as `None` rather than panicking).
macro_rules! bridge_enum {
    (
        $(#[$meta:meta])*
        $py:ident <=> $rs:ident as $name:literal { $($variant:ident),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[pyclass(name = $name, eq, eq_int, from_py_object)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $py {
            $($variant),+
        }

        impl From<$py> for $rs {
            fn from(v: $py) -> Self {
                match v {
                    $($py::$variant => $rs::$variant),+
                }
            }
        }

        impl TryFrom<$rs> for $py {
            type Error = ();
            fn try_from(v: $rs) -> Result<Self, ()> {
                match v {
                    $($rs::$variant => Ok($py::$variant),)+
                    _ => Err(()),
                }
            }
        }
    };
}

bridge_enum! {
    /// Color primaries (`color_space` in the EdgeFirst schema).
    PyColorSpace <=> ColorSpace as "ColorSpace" { Bt709, Bt2020, Srgb, Smpte170m }
}

bridge_enum! {
    /// Transfer function (`color_transfer` in the EdgeFirst schema).
    PyColorTransfer <=> ColorTransfer as "ColorTransfer" { Bt709, Srgb, Pq, Hlg, Linear }
}

bridge_enum! {
    /// YCbCr encoding matrix (`color_encoding` in the EdgeFirst schema).
    PyColorEncoding <=> ColorEncoding as "ColorEncoding" { Bt601, Bt709, Bt2020 }
}

bridge_enum! {
    /// Quantization range (`color_range` in the EdgeFirst schema).
    PyColorRange <=> ColorRange as "ColorRange" { Full, Limited }
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
