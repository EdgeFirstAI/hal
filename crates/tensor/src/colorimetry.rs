// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Colorimetry metadata for image/video tensors.
//!
//! Four orthogonal axes mirroring V4L2's `struct v4l2_format` colorimetry
//! fields and `libcamera::ColorSpace`, named to match the EdgeFirst
//! `CameraFrame.msg` schema so values round-trip through the ROS layer.
//! Each enum is `#[non_exhaustive]`; unknown/`_DEFAULT` values map to `None`.

use core::fmt;
use serde::{Deserialize, Serialize};

// V4L2 UAPI constants (stable kernel ABI) — mirrored from <linux/videodev2.h>.
const V4L2_COLORSPACE_SMPTE170M: u32 = 1;
const V4L2_COLORSPACE_REC709: u32 = 3;
const V4L2_COLORSPACE_470_SYSTEM_M: u32 = 5;
const V4L2_COLORSPACE_470_SYSTEM_BG: u32 = 6;
const V4L2_COLORSPACE_JPEG: u32 = 7;
const V4L2_COLORSPACE_SRGB: u32 = 8;
const V4L2_COLORSPACE_BT2020: u32 = 10;
const V4L2_XFER_FUNC_709: u32 = 1;
const V4L2_XFER_FUNC_SRGB: u32 = 2;
const V4L2_XFER_FUNC_NONE: u32 = 5;
const V4L2_XFER_FUNC_SMPTE2084: u32 = 7;
const V4L2_YCBCR_ENC_601: u32 = 1;
const V4L2_YCBCR_ENC_709: u32 = 2;
const V4L2_YCBCR_ENC_BT2020: u32 = 6;
const V4L2_QUANTIZATION_FULL_RANGE: u32 = 1;
const V4L2_QUANTIZATION_LIM_RANGE: u32 = 2;

/// Color primaries (`color_space` in the EdgeFirst schema).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    Bt709,
    Bt2020,
    Srgb,
    Smpte170m,
}

/// Transfer function (`color_transfer` in the EdgeFirst schema).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorTransfer {
    Bt709,
    Srgb,
    Pq,
    /// Hybrid Log-Gamma. Present for EdgeFirst-schema / libcamera parity; the
    /// V4L2 UAPI defines no `V4L2_XFER_FUNC_HLG`, so `from_v4l2` never yields
    /// this variant.
    Hlg,
    Linear,
}

/// YCbCr encoding matrix (`color_encoding` in the EdgeFirst schema).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorEncoding {
    Bt601,
    Bt709,
    Bt2020,
}

/// Quantization range (`color_range` in the EdgeFirst schema).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorRange {
    Full,
    Limited,
}

impl ColorSpace {
    /// Short string label matching the EdgeFirst schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt709 => "bt709",
            Self::Bt2020 => "bt2020",
            Self::Srgb => "srgb",
            Self::Smpte170m => "smpte170m",
        }
    }

    /// Map a raw V4L2 `colorspace` field to a [`ColorSpace`].
    ///
    /// Returns `None` for `V4L2_COLORSPACE_DEFAULT` (0) and any
    /// unrecognised value.
    pub fn from_v4l2(v: u32) -> Option<Self> {
        match v {
            V4L2_COLORSPACE_SMPTE170M => Some(Self::Smpte170m),
            V4L2_COLORSPACE_REC709 => Some(Self::Bt709),
            // legacy NTSC (470M) / PAL-SECAM (470BG): close enough to SMPTE 170M primaries for conversion
            V4L2_COLORSPACE_470_SYSTEM_M | V4L2_COLORSPACE_470_SYSTEM_BG => Some(Self::Smpte170m),
            V4L2_COLORSPACE_JPEG | V4L2_COLORSPACE_SRGB => Some(Self::Srgb),
            V4L2_COLORSPACE_BT2020 => Some(Self::Bt2020),
            _ => None,
        }
    }
}

impl ColorTransfer {
    /// Short string label matching the EdgeFirst schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt709 => "bt709",
            Self::Srgb => "srgb",
            Self::Pq => "pq",
            Self::Hlg => "hlg",
            Self::Linear => "linear",
        }
    }

    /// Map a raw V4L2 `xfer_func` field to a [`ColorTransfer`].
    ///
    /// Returns `None` for `V4L2_XFER_FUNC_DEFAULT` (0) and any
    /// unrecognised value.
    ///
    /// Note: OPRGB (3), SMPTE240M (4), and DCI_P3 (6) have no HAL equivalent
    /// and map to `None`. [`ColorTransfer::Hlg`] is never produced because the
    /// V4L2 UAPI defines no `V4L2_XFER_FUNC_HLG` value.
    pub fn from_v4l2(v: u32) -> Option<Self> {
        match v {
            V4L2_XFER_FUNC_709 => Some(Self::Bt709),
            V4L2_XFER_FUNC_SRGB => Some(Self::Srgb),
            V4L2_XFER_FUNC_NONE => Some(Self::Linear),
            V4L2_XFER_FUNC_SMPTE2084 => Some(Self::Pq),
            _ => None,
        }
    }
}

impl ColorEncoding {
    /// Short string label matching the EdgeFirst schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bt601 => "bt601",
            Self::Bt709 => "bt709",
            Self::Bt2020 => "bt2020",
        }
    }

    /// Map a raw V4L2 `ycbcr_enc` field to a [`ColorEncoding`].
    ///
    /// Returns `None` for `V4L2_YCBCR_ENC_DEFAULT` (0) and any
    /// unrecognised value.
    pub fn from_v4l2(v: u32) -> Option<Self> {
        match v {
            V4L2_YCBCR_ENC_601 => Some(Self::Bt601),
            V4L2_YCBCR_ENC_709 => Some(Self::Bt709),
            V4L2_YCBCR_ENC_BT2020 => Some(Self::Bt2020),
            _ => None,
        }
    }
}

impl ColorRange {
    /// Short string label matching the EdgeFirst schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Limited => "limited",
        }
    }

    /// Map a raw V4L2 `quantization` field to a [`ColorRange`].
    ///
    /// Returns `None` for `V4L2_QUANTIZATION_DEFAULT` (0) and any
    /// unrecognised value.
    pub fn from_v4l2(v: u32) -> Option<Self> {
        match v {
            V4L2_QUANTIZATION_FULL_RANGE => Some(Self::Full),
            V4L2_QUANTIZATION_LIM_RANGE => Some(Self::Limited),
            _ => None,
        }
    }
}

macro_rules! display_via_as_str {
    ($($t:ty),*) => {$(
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    )*};
}
display_via_as_str!(ColorSpace, ColorTransfer, ColorEncoding, ColorRange);

/// Full 4-axis colorimetry. Each axis is `Option`; `None` means "undefined"
/// and is never auto-filled — consumers (e.g. `convert()`) resolve missing
/// axes at use-time without mutating the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Colorimetry {
    pub space: Option<ColorSpace>,
    pub transfer: Option<ColorTransfer>,
    pub encoding: Option<ColorEncoding>,
    pub range: Option<ColorRange>,
}

impl Colorimetry {
    /// JPEG/JFIF colorimetry: sRGB primaries, sRGB transfer, BT.601
    /// encoding, full range.
    pub fn jfif() -> Self {
        Self {
            space: Some(ColorSpace::Srgb),
            transfer: Some(ColorTransfer::Srgb),
            encoding: Some(ColorEncoding::Bt601),
            range: Some(ColorRange::Full),
        }
    }

    /// Build from the four raw V4L2 colorimetry integers.
    ///
    /// Each `0` / unrecognised value maps to `None` on that axis.
    pub fn from_v4l2(colorspace: u32, xfer: u32, ycbcr_enc: u32, quant: u32) -> Self {
        Self {
            space: ColorSpace::from_v4l2(colorspace),
            transfer: ColorTransfer::from_v4l2(xfer),
            encoding: ColorEncoding::from_v4l2(ycbcr_enc),
            range: ColorRange::from_v4l2(quant),
        }
    }

    /// Set color primaries (consuming builder).
    pub fn with_space(mut self, s: ColorSpace) -> Self {
        self.space = Some(s);
        self
    }

    /// Set transfer function (consuming builder).
    pub fn with_transfer(mut self, t: ColorTransfer) -> Self {
        self.transfer = Some(t);
        self
    }

    /// Set YCbCr encoding matrix (consuming builder).
    pub fn with_encoding(mut self, e: ColorEncoding) -> Self {
        self.encoding = Some(e);
        self
    }

    /// Set quantization range (consuming builder).
    pub fn with_range(mut self, r: ColorRange) -> Self {
        self.range = Some(r);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_v4l2_known_and_default() {
        assert_eq!(ColorEncoding::from_v4l2(1), Some(ColorEncoding::Bt601));
        assert_eq!(ColorEncoding::from_v4l2(2), Some(ColorEncoding::Bt709));
        assert_eq!(ColorEncoding::from_v4l2(6), Some(ColorEncoding::Bt2020));
        assert_eq!(ColorEncoding::from_v4l2(0), None); // DEFAULT
        assert_eq!(ColorEncoding::from_v4l2(3), None); // XV601 (unsurfaced)
        assert_eq!(ColorRange::from_v4l2(1), Some(ColorRange::Full));
        assert_eq!(ColorRange::from_v4l2(2), Some(ColorRange::Limited));
        assert_eq!(ColorSpace::from_v4l2(7), Some(ColorSpace::Srgb)); // JPEG→sRGB
        assert_eq!(ColorTransfer::from_v4l2(5), Some(ColorTransfer::Linear)); // NONE→linear

        // ColorSpace: additional arms
        assert_eq!(ColorSpace::from_v4l2(1), Some(ColorSpace::Smpte170m)); // SMPTE170M
        assert_eq!(ColorSpace::from_v4l2(3), Some(ColorSpace::Bt709)); // REC709
        assert_eq!(ColorSpace::from_v4l2(10), Some(ColorSpace::Bt2020)); // BT2020
        assert_eq!(ColorSpace::from_v4l2(5), Some(ColorSpace::Smpte170m)); // 470_SYSTEM_M
        assert_eq!(ColorSpace::from_v4l2(6), Some(ColorSpace::Smpte170m)); // 470_SYSTEM_BG

        // ColorTransfer: additional arms and unmapped values
        assert_eq!(ColorTransfer::from_v4l2(1), Some(ColorTransfer::Bt709)); // XFER_FUNC_709
        assert_eq!(ColorTransfer::from_v4l2(2), Some(ColorTransfer::Srgb)); // XFER_FUNC_SRGB
        assert_eq!(ColorTransfer::from_v4l2(7), Some(ColorTransfer::Pq)); // XFER_FUNC_SMPTE2084
        assert_eq!(ColorTransfer::from_v4l2(3), None); // OPRGB — no HAL equivalent
        assert_eq!(ColorTransfer::from_v4l2(6), None); // DCI_P3 — no HAL equivalent

        // Hlg is never produced by from_v4l2 — no V4L2_XFER_FUNC_HLG exists in the kernel UAPI
        for v in 0u32..=10 {
            assert_ne!(ColorTransfer::from_v4l2(v), Some(ColorTransfer::Hlg));
        }
    }

    #[test]
    fn jfif_is_bt601_full_srgb() {
        let c = Colorimetry::jfif();
        assert_eq!(c.space, Some(ColorSpace::Srgb));
        assert_eq!(c.transfer, Some(ColorTransfer::Srgb));
        assert_eq!(c.encoding, Some(ColorEncoding::Bt601));
        assert_eq!(c.range, Some(ColorRange::Full));
    }

    #[test]
    fn from_v4l2_struct_maps_all_axes_and_unknown_to_none() {
        let c = Colorimetry::from_v4l2(3, 1, 2, 1); // REC709, XFER709, ENC709, FULL
        assert_eq!(c.space, Some(ColorSpace::Bt709));
        assert_eq!(c.transfer, Some(ColorTransfer::Bt709));
        assert_eq!(c.encoding, Some(ColorEncoding::Bt709));
        assert_eq!(c.range, Some(ColorRange::Full));
        let d = Colorimetry::from_v4l2(0, 0, 0, 0); // all DEFAULT
        assert_eq!(d, Colorimetry::default()); // all None
    }

    #[test]
    fn as_str_matches_schema() {
        assert_eq!(ColorEncoding::Bt601.as_str(), "bt601");
        assert_eq!(ColorRange::Full.as_str(), "full");
        assert_eq!(ColorSpace::Smpte170m.as_str(), "smpte170m");
        assert_eq!(ColorTransfer::Pq.as_str(), "pq");
        assert_eq!(ColorTransfer::Hlg.as_str(), "hlg");
    }
}
