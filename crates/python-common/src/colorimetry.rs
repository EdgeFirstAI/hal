// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `edgefirst_tensor::Colorimetry` and its four axis
//! enums (`ColorSpace`, `ColorTransfer`, `ColorEncoding`, `ColorRange`).
//!
//! Each axis is optional; `None` means "undefined" and is never auto-filled.

use edgefirst_tensor::{
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
        // No `eq`/`eq_int` -- see `PyTensorMemory`'s comment in tensor.rs;
        // `__eq__`/`__ne__` are hand-written below instead, for the
        // cross-package fallback.
        $(#[$meta])*
        #[pyclass(name = $name, skip_from_py_object, module = "edgefirst.tensor")]
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

        impl $py {
            /// Reconstruct from the `__int__()` discriminant of a sibling
            /// package's copy of this enum. See
            /// `crate::tensor::extract_eq_int_enum`.
            fn from_discriminant(v: i64) -> Option<Self> {
                $(if v == $py::$variant as i64 {
                    return Some($py::$variant);
                })+
                None
            }
        }

        /// A sibling `edgefirst.*` package's copy of this enum names the
        /// same variant but is a distinct PyO3 type object -- see
        /// `PyTensorMemory`'s `FromPyObject` impl in `tensor.rs` for the
        /// full story. Try the native downcast first, then fall back to
        /// the `__int__()` discriminant.
        impl<'a, 'py> FromPyObject<'a, 'py> for $py {
            type Error = PyErr;

            fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
                crate::tensor::extract_eq_int_enum(obj, $name, Self::from_discriminant)
            }
        }

        /// See `PyTensorMemory`'s `__eq__`/`__ne__` in tensor.rs for why
        /// these are hand-written: the auto-generated `eq_int` richcmp
        /// resolves `other` by native identity or a bare int only, so a
        /// sibling package's copy of this enum silently compares unequal.
        #[pymethods]
        impl $py {
            fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                crate::tensor::eq_int_richcmp(*self, other, false, $name, Self::from_discriminant)
            }

            fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                crate::tensor::eq_int_richcmp(*self, other, true, $name, Self::from_discriminant)
            }

            /// See `PyTensorMemory::__hash__` in tensor.rs -- same
            /// discriminant-is-the-hash story.
            fn __hash__(&self) -> isize {
                *self as isize
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
#[pyclass(name = "Colorimetry", skip_from_py_object, module = "edgefirst.tensor")]
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

/// `Colorimetry` is a value struct, not an `eq_int` enum, so a sibling
/// package's copy (same identity problem as `PyTensorMemory` -- see its
/// `FromPyObject` impl) is accepted by reading its four axis getters back
/// instead of a discriminant. Each axis then resolves cross-package too,
/// via that axis enum's own `FromPyObject` fallback (generated by
/// `bridge_enum!` above).
impl<'a, 'py> FromPyObject<'a, 'py> for PyColorimetry {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(guard) = obj.extract::<pyo3::PyClassGuard<'_, Self>>() {
            return Ok(guard.clone());
        }
        let space: Option<PyColorSpace> = obj.getattr("space")?.extract()?;
        let transfer: Option<PyColorTransfer> = obj.getattr("transfer")?.extract()?;
        let encoding: Option<PyColorEncoding> = obj.getattr("encoding")?.extract()?;
        let range: Option<PyColorRange> = obj.getattr("range")?.extract()?;
        Ok(PyColorimetry(RsColorimetry {
            space: space.map(Into::into),
            transfer: transfer.map(Into::into),
            encoding: encoding.map(Into::into),
            range: range.map(Into::into),
        }))
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
