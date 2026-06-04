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
    /// Semi-planar YUV 4:4:4, contiguous shape `[H*3, W]`. Full-resolution
    /// chroma: Y plane (H rows of W bytes) + interleaved Cb/Cr plane (H image
    /// rows of W pairs = 2W bytes/row, laid out as 2H rows of W) → 3H rows
    /// total. Multiplane NV24 is not yet supported (see `from_planes`). Added
    /// last to keep the existing `#[repr(u8)]` discriminants (and any
    /// serialized values) stable.
    Nv24,
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

/// Chroma addressing parameters for a semi-planar (NV12/NV16/NV24) format —
/// the single source of truth shared by the codec writer, CPU readers, and the
/// Linux + macOS GL shaders. See [`PixelFormat::chroma_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChromaLayout {
    /// Right-shift applied to the luma `x` to get the chroma column: 1 = half
    /// horizontal resolution (NV12/NV16), 0 = full resolution (NV24).
    pub shift_x: u32,
    /// Right-shift applied to the luma `y` to get the chroma row: 1 = half
    /// vertical resolution (NV12), 0 = full vertical resolution (NV16/NV24).
    pub shift_y: u32,
    /// Physical buffer rows the UV plane advances per chroma line: 1 for
    /// NV12/NV16 (one `(Cb,Cr)` line fits in a single stride-wide row), 2 for
    /// NV24 (a full-width `2*W`-byte chroma line spans two stride-wide rows).
    pub uv_rows_per_luma: usize,
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
const FOURCC_NV24: u32 = u32::from_le_bytes(*b"NV24");

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
            Self::Grey | Self::Nv12 | Self::Nv16 | Self::Nv24 => 1,
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
            Self::Nv12 | Self::Nv16 | Self::Nv24 => PixelLayout::SemiPlanar,
        }
    }

    /// The tensor shape for this format at `width`×`height`, or `None` if the
    /// dimensions are invalid for the format, or the format is an unsupported
    /// semi-planar variant (any `SemiPlanar` variant other than `Nv12`,
    /// `Nv16`, and `Nv24`).
    ///
    /// Odd dimensions are fully supported.  The combined-plane height for NV12
    /// is `height + ceil(height / 2)` (luma rows + chroma rows), which equals
    /// the classic `height * 3 / 2` for even heights and stays exact for odd
    /// ones — e.g. 483 → 725 rows (483 luma + 242 chroma).
    ///
    /// For semi-planar formats the shape carries the **logical** width as-is
    /// (odd widths are preserved, e.g. `[720, 789]` for a 789×384 NV12).
    /// The row stride recorded separately on the tensor is `>= even(width)` and
    /// 64-byte aligned; it may exceed the logical width.  Use
    /// `effective_row_stride()` to determine the true byte pitch for
    /// mapping and allocation.  Allocation byte size = `total_h * row_stride`,
    /// NOT the shape product.
    pub fn image_shape(&self, width: usize, height: usize) -> Option<Vec<usize>> {
        match self.layout() {
            PixelLayout::Packed => Some(vec![height, width, self.channels()]),
            PixelLayout::Planar => Some(vec![self.channels(), height, width]),
            PixelLayout::SemiPlanar => {
                // Shape carries logical width; row_stride (>= even(width), 64-aligned)
                // is stored separately on the Tensor and governs byte layout.
                Some(vec![self.combined_plane_height(height)?, width])
            }
        }
    }

    /// Combined-plane height in physical (stride-wide) rows for a semi-planar
    /// format: the Y rows plus the interleaved-UV rows.
    ///
    ///   * NV12 (4:2:0): `H + ceil(H/2)` — exact for odd heights (e.g. 483 →
    ///     725 = 483 luma + 242 chroma), equals the classic `H*3/2` for even.
    ///   * NV16 (4:2:2): `2H` (one full-height chroma row per luma row).
    ///   * NV24 (4:4:4): `3H` (a full-width `2W`-byte chroma line spans two
    ///     stride-wide buffer rows, so `2H` chroma rows).
    ///
    /// Returns `None` for non-semi-planar formats (and unsupported SemiPlanar
    /// variants). This is the single source of truth for the vertical extent of
    /// the contiguous NV* buffer — [`image_shape`](Self::image_shape), the GL
    /// DMA-BUF/IOSurface imports, the PBO allocator, and the gpu-probe all
    /// derive from it, so the combined-plane height can never drift between them.
    pub const fn combined_plane_height(&self, height: usize) -> Option<usize> {
        match self {
            PixelFormat::Nv12 => Some(height + height.div_ceil(2)),
            PixelFormat::Nv16 => Some(height * 2),
            PixelFormat::Nv24 => Some(height * 3),
            _ => None,
        }
    }

    /// Per-format semi-planar chroma addressing parameters, shared by the codec
    /// writer ([`uv_rows_per_luma`](ChromaLayout::uv_rows_per_luma)), the CPU
    /// readers, and both GL shaders so the combined-plane chroma geometry has a
    /// single source of truth. Returns `None` for non-semi-planar formats.
    pub const fn chroma_layout(&self) -> Option<ChromaLayout> {
        match self {
            // 4:2:0 — half horizontal & vertical chroma resolution.
            PixelFormat::Nv12 => Some(ChromaLayout {
                shift_x: 1,
                shift_y: 1,
                uv_rows_per_luma: 1,
            }),
            // 4:2:2 — half horizontal, full vertical.
            PixelFormat::Nv16 => Some(ChromaLayout {
                shift_x: 1,
                shift_y: 0,
                uv_rows_per_luma: 1,
            }),
            // 4:4:4 — full resolution; the 2W-byte chroma line spans two rows.
            PixelFormat::Nv24 => Some(ChromaLayout {
                shift_x: 0,
                shift_y: 0,
                uv_rows_per_luma: 2,
            }),
            _ => None,
        }
    }

    /// Physical GPU-surface dimensions `(pitch_width, total_h)` in texels for a
    /// semi-planar combined plane bound as one `bpe`-byte-per-element texture,
    /// or `None` for non-semi-planar formats.
    ///
    /// The width is rounded up to the 64-aligned row pitch (`== bytes_per_row`)
    /// rather than left at the even logical width. ANGLE (and tiled GPUs in
    /// general) will not address texels beyond a surface's declared width via
    /// `texelFetch`, so a surface narrower than its padded `bytes_per_row`
    /// leaves the padding columns unreachable. That is fatal for NV24 (4:4:4):
    /// its chroma line is `2*W` interleaved bytes, which spills past the even
    /// width into those padding columns whenever the row is padded
    /// (`bytes_per_row > even_width`). Making the surface width equal the pitch
    /// keeps every byte addressable and costs nothing — `bytes_per_row` is
    /// already this value.
    ///
    /// Single source of truth for both IOSurface allocators (the tensor crate's
    /// `IoSurfaceTensor::new_image` and the image crate's `ImageLayout`), so
    /// they cannot diverge.
    pub fn semi_planar_surface_dims(
        &self,
        width: usize,
        height: usize,
        bpe: usize,
    ) -> Option<(usize, usize)> {
        let total_h = self.combined_plane_height(height)?;
        // image_shape carries the logical width; round its byte pitch up to 64
        // (bpe == 1 for the R8 combined-plane binding, so pitch == aligned width).
        let pitch_width = (width * bpe).next_multiple_of(64) / bpe;
        Some((pitch_width, total_h))
    }

    /// Returns `true` if this format encodes YUV (luma/chroma) data.
    pub const fn is_yuv(&self) -> bool {
        matches!(
            self,
            Self::Yuyv | Self::Vyuy | Self::Nv12 | Self::Nv16 | Self::Nv24
        )
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
            Self::Nv24 => FOURCC_NV24,
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
            FOURCC_NV24 => Some(Self::Nv24),
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
        assert_eq!(PixelFormat::Nv24.channels(), 1);
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
        assert_eq!(PixelFormat::Nv24.layout(), PixelLayout::SemiPlanar);
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
        assert!(PixelFormat::Nv24.is_yuv());
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
            PixelFormat::Nv24,
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
