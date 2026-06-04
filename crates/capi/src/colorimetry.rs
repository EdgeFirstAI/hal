// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Colorimetry C API - stable, V4L2-decoupled colorimetry representation.
//!
//! Colorimetry is exposed to C as four independent integer axes. The integer
//! values are **stable HAL constants**, deliberately decoupled from the raw
//! V4L2 enum values. `hal_colorimetry_from_v4l2()` is the only bridge from
//! raw V4L2 integers into this representation.

use edgefirst_tensor::{ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry};
use libc::c_int;

/// Colorimetry as four independent axes. 0 = unknown/unspecified on each axis.
/// Values are stable HAL constants (NOT raw V4L2):
///   space:    1=bt709 2=bt2020 3=srgb 4=smpte170m
///   transfer: 1=bt709 2=srgb 3=pq 4=hlg 5=linear
///   encoding: 1=bt601 2=bt709 3=bt2020
///   range:    1=full 2=limited
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct hal_colorimetry {
    pub space: c_int,
    pub transfer: c_int,
    pub encoding: c_int,
    pub range: c_int,
}

// --- Per-axis mappings (stable HAL ints <-> Option<Enum>). ---
//
// These live in the capi crate so the stable C-facing integer convention is
// not baked into the tensor crate. None <-> 0; unknown ints decode to None.

fn space_to_int(s: Option<ColorSpace>) -> c_int {
    match s {
        Some(ColorSpace::Bt709) => 1,
        Some(ColorSpace::Bt2020) => 2,
        Some(ColorSpace::Srgb) => 3,
        Some(ColorSpace::Smpte170m) => 4,
        _ => 0,
    }
}

fn space_from_int(v: c_int) -> Option<ColorSpace> {
    match v {
        1 => Some(ColorSpace::Bt709),
        2 => Some(ColorSpace::Bt2020),
        3 => Some(ColorSpace::Srgb),
        4 => Some(ColorSpace::Smpte170m),
        _ => None,
    }
}

fn transfer_to_int(t: Option<ColorTransfer>) -> c_int {
    match t {
        Some(ColorTransfer::Bt709) => 1,
        Some(ColorTransfer::Srgb) => 2,
        Some(ColorTransfer::Pq) => 3,
        Some(ColorTransfer::Hlg) => 4,
        Some(ColorTransfer::Linear) => 5,
        _ => 0,
    }
}

fn transfer_from_int(v: c_int) -> Option<ColorTransfer> {
    match v {
        1 => Some(ColorTransfer::Bt709),
        2 => Some(ColorTransfer::Srgb),
        3 => Some(ColorTransfer::Pq),
        4 => Some(ColorTransfer::Hlg),
        5 => Some(ColorTransfer::Linear),
        _ => None,
    }
}

fn encoding_to_int(e: Option<ColorEncoding>) -> c_int {
    match e {
        Some(ColorEncoding::Bt601) => 1,
        Some(ColorEncoding::Bt709) => 2,
        Some(ColorEncoding::Bt2020) => 3,
        _ => 0,
    }
}

fn encoding_from_int(v: c_int) -> Option<ColorEncoding> {
    match v {
        1 => Some(ColorEncoding::Bt601),
        2 => Some(ColorEncoding::Bt709),
        3 => Some(ColorEncoding::Bt2020),
        _ => None,
    }
}

fn range_to_int(r: Option<ColorRange>) -> c_int {
    match r {
        Some(ColorRange::Full) => 1,
        Some(ColorRange::Limited) => 2,
        _ => 0,
    }
}

fn range_from_int(v: c_int) -> Option<ColorRange> {
    match v {
        1 => Some(ColorRange::Full),
        2 => Some(ColorRange::Limited),
        _ => None,
    }
}

impl From<Colorimetry> for hal_colorimetry {
    fn from(c: Colorimetry) -> Self {
        hal_colorimetry {
            space: space_to_int(c.space),
            transfer: transfer_to_int(c.transfer),
            encoding: encoding_to_int(c.encoding),
            range: range_to_int(c.range),
        }
    }
}

impl From<hal_colorimetry> for Colorimetry {
    fn from(c: hal_colorimetry) -> Self {
        Colorimetry {
            space: space_from_int(c.space),
            transfer: transfer_from_int(c.transfer),
            encoding: encoding_from_int(c.encoding),
            range: range_from_int(c.range),
        }
    }
}

/// Build a hal_colorimetry from the four raw V4L2 colorimetry ints (each
/// 0/unknown -> 0).
///
/// This is the only bridge from raw V4L2 integer values into the stable HAL
/// colorimetry representation. Each axis is mapped via
/// `edgefirst_tensor::Colorimetry::from_v4l2`, then re-encoded as stable HAL
/// constants (see `hal_colorimetry`).
///
/// @param colorspace Raw V4L2 `colorspace` field
/// @param xfer       Raw V4L2 `xfer_func` field
/// @param ycbcr_enc  Raw V4L2 `ycbcr_enc` field
/// @param quant      Raw V4L2 `quantization` field
/// @param out        Output colorimetry (must not be NULL)
/// @return 0 on success, -1 if `out` is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_colorimetry_from_v4l2(
    colorspace: u32,
    xfer: u32,
    ycbcr_enc: u32,
    quant: u32,
    out: *mut hal_colorimetry,
) -> c_int {
    if out.is_null() {
        return crate::error::set_error(libc::EINVAL);
    }
    let c = Colorimetry::from_v4l2(colorspace, xfer, ycbcr_enc, quant);
    unsafe { *out = c.into() };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_each_axis() {
        let c = Colorimetry {
            space: Some(ColorSpace::Bt2020),
            transfer: Some(ColorTransfer::Pq),
            encoding: Some(ColorEncoding::Bt2020),
            range: Some(ColorRange::Limited),
        };
        let hc: hal_colorimetry = c.into();
        assert_eq!(hc.space, 2);
        assert_eq!(hc.transfer, 3);
        assert_eq!(hc.encoding, 3);
        assert_eq!(hc.range, 2);
        let back: Colorimetry = hc.into();
        assert_eq!(back, c);
    }

    #[test]
    fn none_maps_to_zero_and_back() {
        let c = Colorimetry::default();
        let hc: hal_colorimetry = c.into();
        assert_eq!(
            hc,
            hal_colorimetry {
                space: 0,
                transfer: 0,
                encoding: 0,
                range: 0
            }
        );
        let back: Colorimetry = hc.into();
        assert_eq!(back, Colorimetry::default());
    }

    #[test]
    fn unknown_ints_decode_to_none() {
        let hc = hal_colorimetry {
            space: 99,
            transfer: -1,
            encoding: 7,
            range: 42,
        };
        let c: Colorimetry = hc.into();
        assert_eq!(c, Colorimetry::default());
    }

    #[test]
    fn from_v4l2_null_out_errors() {
        let r = unsafe { hal_colorimetry_from_v4l2(3, 1, 2, 1, std::ptr::null_mut()) };
        assert_eq!(r, -1);
    }
}
