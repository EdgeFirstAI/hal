// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;

/// Pixel format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Packed RGB [H, W, 3]
    Rgb = 1,
    /// Packed RGBA [H, W, 4]
    Rgba,
    /// Packed BGRA [H, W, 4]
    Bgra,
    /// Grayscale [H, W, 1]
    Grey,
    /// Packed YUV 4:2:2, YUYV byte order [H, W, 2]
    Yuyv,
    /// Packed YUV 4:2:2, VYUY byte order [H, W, 2]
    Vyuy,
    /// Semi-planar YUV 4:2:0 [H*3/2, W] or multiplane [H, W] + [H/2, W]
    Nv12,
    /// Semi-planar YUV 4:2:2 [H*2, W] or multiplane [H, W] + [H, W]
    Nv16,
    /// Planar RGB, channels-first [3, H, W]
    PlanarRgb,
    /// Planar RGBA, channels-first [4, H, W]
    PlanarRgba,
}

/// Memory layout category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PixelLayout {
    /// Interleaved channels: [H, W, C]
    Packed,
    /// Channels-first: [C, H, W]
    Planar,
    /// Luma plane + interleaved chroma plane
    SemiPlanar,
}

/// FourCC code constants (V4L2/DRM compatible).
const FOURCC_RGB: u32 = u32::from_le_bytes(*b"RGB ");
const FOURCC_RGBA: u32 = u32::from_le_bytes(*b"RGBA");
const FOURCC_BGRA: u32 = u32::from_le_bytes(*b"BGRA");
const FOURCC_GREY: u32 = u32::from_le_bytes(*b"Y800");
const FOURCC_YUYV: u32 = u32::from_le_bytes(*b"YUYV");
const FOURCC_VYUY: u32 = u32::from_le_bytes(*b"VYUY");
const FOURCC_NV12: u32 = u32::from_le_bytes(*b"NV12");
const FOURCC_NV16: u32 = u32::from_le_bytes(*b"NV16");

impl PixelFormat {
    /// Returns the number of channels for this pixel format.
    ///
    /// For semi-planar formats (NV12, NV16), this returns 1 (the luma channel
    /// count for the primary plane). For packed formats, this is the total
    /// number of interleaved components per pixel.
    pub const fn channels(&self) -> usize {
        match self {
            Self::Rgb | Self::PlanarRgb => 3,
            Self::Rgba | Self::Bgra | Self::PlanarRgba => 4,
            Self::Grey | Self::Nv12 | Self::Nv16 => 1,
            Self::Yuyv | Self::Vyuy => 2,
        }
    }

    /// Returns the memory layout category for this pixel format.
    pub const fn layout(&self) -> PixelLayout {
        match self {
            Self::Rgb | Self::Rgba | Self::Bgra | Self::Grey | Self::Yuyv | Self::Vyuy => {
                PixelLayout::Packed
            }
            Self::PlanarRgb | Self::PlanarRgba => PixelLayout::Planar,
            Self::Nv12 | Self::Nv16 => PixelLayout::SemiPlanar,
        }
    }

    /// Returns `true` if this format encodes YUV (luma/chroma) data.
    pub const fn is_yuv(&self) -> bool {
        matches!(self, Self::Yuyv | Self::Vyuy | Self::Nv12 | Self::Nv16)
    }

    /// Returns `true` if this format includes an alpha channel.
    pub const fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba | Self::Bgra | Self::PlanarRgba)
    }

    /// Returns the V4L2/DRM FourCC code for this format, or `0` for formats
    /// that have no standard FourCC representation (e.g., `PlanarRgb`).
    pub const fn to_fourcc(&self) -> u32 {
        match self {
            Self::Rgb => FOURCC_RGB,
            Self::Rgba => FOURCC_RGBA,
            Self::Bgra => FOURCC_BGRA,
            Self::Grey => FOURCC_GREY,
            Self::Yuyv => FOURCC_YUYV,
            Self::Vyuy => FOURCC_VYUY,
            Self::Nv12 => FOURCC_NV12,
            Self::Nv16 => FOURCC_NV16,
            Self::PlanarRgb | Self::PlanarRgba => 0,
        }
    }

    /// Converts a V4L2/DRM FourCC code to a `PixelFormat`, returning `None`
    /// for unrecognized or zero codes.
    pub const fn from_fourcc(fourcc: u32) -> Option<Self> {
        match fourcc {
            FOURCC_RGB => Some(Self::Rgb),
            FOURCC_RGBA => Some(Self::Rgba),
            FOURCC_BGRA => Some(Self::Bgra),
            FOURCC_GREY => Some(Self::Grey),
            FOURCC_YUYV => Some(Self::Yuyv),
            FOURCC_VYUY => Some(Self::Vyuy),
            FOURCC_NV12 => Some(Self::Nv12),
            FOURCC_NV16 => Some(Self::Nv16),
            _ => None,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fcc = self.to_fourcc();
        if fcc != 0 {
            let bytes = fcc.to_le_bytes();
            for &b in &bytes {
                if b == b' ' {
                    break;
                }
                write!(f, "{}", b as char)?;
            }
            Ok(())
        } else {
            write!(f, "{self:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channels() {
        assert_eq!(PixelFormat::Rgb.channels(), 3);
        assert_eq!(PixelFormat::Rgba.channels(), 4);
        assert_eq!(PixelFormat::Bgra.channels(), 4);
        assert_eq!(PixelFormat::Grey.channels(), 1);
        assert_eq!(PixelFormat::Yuyv.channels(), 2);
        assert_eq!(PixelFormat::Vyuy.channels(), 2);
        assert_eq!(PixelFormat::Nv12.channels(), 1);
        assert_eq!(PixelFormat::Nv16.channels(), 1);
        assert_eq!(PixelFormat::PlanarRgb.channels(), 3);
        assert_eq!(PixelFormat::PlanarRgba.channels(), 4);
    }

    #[test]
    fn layout() {
        assert_eq!(PixelFormat::Rgb.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Rgba.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Bgra.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Grey.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Yuyv.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Vyuy.layout(), PixelLayout::Packed);
        assert_eq!(PixelFormat::Nv12.layout(), PixelLayout::SemiPlanar);
        assert_eq!(PixelFormat::Nv16.layout(), PixelLayout::SemiPlanar);
        assert_eq!(PixelFormat::PlanarRgb.layout(), PixelLayout::Planar);
        assert_eq!(PixelFormat::PlanarRgba.layout(), PixelLayout::Planar);
    }

    #[test]
    fn is_yuv() {
        assert!(!PixelFormat::Rgb.is_yuv());
        assert!(!PixelFormat::Rgba.is_yuv());
        assert!(PixelFormat::Yuyv.is_yuv());
        assert!(PixelFormat::Vyuy.is_yuv());
        assert!(PixelFormat::Nv12.is_yuv());
        assert!(PixelFormat::Nv16.is_yuv());
        assert!(!PixelFormat::PlanarRgb.is_yuv());
    }

    #[test]
    fn has_alpha() {
        assert!(!PixelFormat::Rgb.has_alpha());
        assert!(PixelFormat::Rgba.has_alpha());
        assert!(PixelFormat::Bgra.has_alpha());
        assert!(!PixelFormat::Grey.has_alpha());
        assert!(!PixelFormat::Yuyv.has_alpha());
        assert!(!PixelFormat::PlanarRgb.has_alpha());
        assert!(PixelFormat::PlanarRgba.has_alpha());
    }

    #[test]
    fn fourcc_roundtrip() {
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
            PixelFormat::Grey,
            PixelFormat::Yuyv,
            PixelFormat::Vyuy,
            PixelFormat::Nv12,
            PixelFormat::Nv16,
        ] {
            let fcc = fmt.to_fourcc();
            assert_ne!(fcc, 0, "{fmt:?} should have a fourcc code");
            assert_eq!(
                PixelFormat::from_fourcc(fcc),
                Some(fmt),
                "roundtrip failed for {fmt:?}"
            );
        }
    }

    #[test]
    fn fourcc_planar_returns_zero() {
        assert_eq!(PixelFormat::PlanarRgb.to_fourcc(), 0);
        assert_eq!(PixelFormat::PlanarRgba.to_fourcc(), 0);
    }

    #[test]
    fn from_fourcc_unknown() {
        assert_eq!(PixelFormat::from_fourcc(0), None);
        assert_eq!(PixelFormat::from_fourcc(0xDEADBEEF), None);
    }

    #[test]
    fn display_fourcc_formats() {
        assert_eq!(format!("{}", PixelFormat::Rgba), "RGBA");
        assert_eq!(format!("{}", PixelFormat::Nv12), "NV12");
        assert_eq!(format!("{}", PixelFormat::Yuyv), "YUYV");
        // Grey uses V4L2 FourCC "Y800", not "GREY"
        assert_eq!(format!("{}", PixelFormat::Grey), "Y800");
    }

    #[test]
    fn display_planar_formats() {
        assert_eq!(format!("{}", PixelFormat::PlanarRgb), "PlanarRgb");
        assert_eq!(format!("{}", PixelFormat::PlanarRgba), "PlanarRgba");
    }

    #[test]
    fn repr_starts_at_one() {
        assert_eq!(PixelFormat::Rgb as u8, 1);
    }

    #[test]
    fn serde_roundtrip() {
        let fmt = PixelFormat::Nv12;
        let json = serde_json::to_string(&fmt).unwrap();
        let back: PixelFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(fmt, back);
    }
}
