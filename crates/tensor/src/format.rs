// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;

// Declared through `ef_vocabulary!` (see `vocabulary.rs`): `.code()` is this
// crate's wire numbering, mirrored by `protocol::format`. `as_str()` returns
// the wire string, which is distinct from both `Display` (FourCC-derived:
// `RGB`, `Y800`, ...) and the `serde` names (PascalCase variant names). The
// RGB-family and `mono8` wire strings are ROS `sensor_msgs/Image` encodings
// (lowercase); the YUV-family wire strings are FourCC codes (uppercase).
crate::ef_vocabulary! {
    /// Pixel format identifier.
    #[derive(Serialize, Deserialize)]
    #[non_exhaustive]
    pub enum PixelFormat {
        /// Packed RGB [H, W, 3]
        Rgb = 1, "rgb8", RGB,
        /// Packed RGBA [H, W, 4]
        Rgba = 2, "rgba8", RGBA,
        /// Packed BGRA [H, W, 4]
        Bgra = 3, "bgra8", BGRA,
        /// Grayscale [H, W, 1]
        Grey = 4, "mono8", GREY,
        /// Packed YUV 4:2:2, YUYV byte order [H, W, 2]
        Yuyv = 5, "YUYV", YUYV,
        /// Packed YUV 4:2:2, VYUY byte order [H, W, 2]
        Vyuy = 6, "VYUY", VYUY,
        /// Semi-planar YUV 4:2:0 [H*3/2, W] or multiplane [H, W] + [H/2, W]
        Nv12 = 7, "NV12", NV12,
        /// Semi-planar YUV 4:2:2 [H*2, W] or multiplane [H, W] + [H, W]
        Nv16 = 8, "NV16", NV16,
        /// Planar RGB, channels-first [3, H, W]
        PlanarRgb = 9, "rgb8_planar", PLANAR_RGB,
        /// Planar RGBA, channels-first [4, H, W]
        PlanarRgba = 10, "rgba8_planar", PLANAR_RGBA,
        /// Semi-planar YUV 4:4:4, contiguous shape `[H*3, W]`. Full-resolution
        /// chroma: Y plane (H rows of W bytes) + interleaved Cb/Cr plane (H
        /// image rows of W pairs = 2W bytes/row, laid out as 2H rows of W) → 3H
        /// rows total. Multiplane NV24 is not supported (see `from_planes`).
        Nv24 = 11, "NV24", NV24,
    }
    // Emitted wire-code module, re-exported by `protocol::format`.
    #[doc(hidden)]
    pub mod pixel_format_wire;
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

/// Chroma addressing parameters for a semi-planar (NV12/NV16/NV24) format. See
/// [`PixelFormat::chroma_layout`].
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

/// Where one plane's bytes physically live inside a tensor's allocation.
///
/// The counterpart to [`PixelFormat::addressing_shape`]: the grid says how a
/// consumer indexes the image, this says where the bytes are. Field names and
/// widths mirror `TensorPlane.msg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneGeometry {
    /// Byte offset of this plane from the start of the allocation.
    pub offset: u64,
    /// Bytes per line of this plane.
    pub stride: u64,
    /// Plane extent in bytes (`stride * rows`).
    pub size: u64,
}

// V4L2 pixel format codes (`V4L2_PIX_FMT_*` from <linux/videodev2.h>).
/// `V4L2_PIX_FMT_RGB24`.
const FOURCC_RGB: u32 = u32::from_le_bytes(*b"RGB3");
/// `V4L2_PIX_FMT_RGBA32` (bytes R, G, B, A; also DRM `ABGR8888`).
const FOURCC_RGBA: u32 = u32::from_le_bytes(*b"AB24");
/// `V4L2_PIX_FMT_ABGR32` (bytes B, G, R, A; also DRM `ARGB8888`).
const FOURCC_BGRA: u32 = u32::from_le_bytes(*b"AR24");
/// `V4L2_PIX_FMT_GREY`.
const FOURCC_GREY: u32 = u32::from_le_bytes(*b"GREY");
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

    /// The allocation geometry for this format at `width`×`height`, or `None`
    /// if the dimensions are invalid for the format, or the format is an
    /// unsupported semi-planar variant (any `SemiPlanar` variant other than
    /// `Nv12`, `Nv16`, and `Nv24`).
    ///
    /// This is the shape that sizes a buffer. For a subsampled format it
    /// differs from the shape a consumer indexes with: NV12 640×480 allocates
    /// `[720, 640]` here but addresses a `[480, 640]` luma grid.
    /// [`addressing_shape`](Self::addressing_shape) returns the latter.
    ///
    /// Odd dimensions are supported. The combined-plane height for NV12 is
    /// `height + ceil(height / 2)` (luma rows + chroma rows), e.g. 483 → 725
    /// rows (483 luma + 242 chroma).
    ///
    /// For semi-planar formats the shape carries the logical width as-is (odd
    /// widths are preserved, e.g. `[720, 789]` for a 789×384 NV12). The row
    /// stride recorded separately on the tensor is `>= even(width)` and 64-byte
    /// aligned; it may exceed the logical width. Allocation byte size is
    /// `total_h * row_stride`, not the shape product.
    pub fn allocation_shape(&self, width: usize, height: usize) -> Option<Vec<usize>> {
        match self.layout() {
            PixelLayout::Packed => Some(vec![height, width, self.channels()]),
            PixelLayout::Planar => Some(vec![self.channels(), height, width]),
            PixelLayout::SemiPlanar => {
                // The shape carries the logical width; the row stride is stored
                // on the tensor.
                Some(vec![self.combined_plane_height(height)?, width])
            }
        }
    }

    /// The addressing grid: how a consumer logically indexes this image.
    ///
    /// This is the shape carried in a descriptor, a blob, or a `Tensor.msg`.
    /// For a subsampled format `product(addressing_shape) * dtype_size` is
    /// smaller than the buffer: NV12 640×480 addresses a 640×480 luma grid
    /// inside a 460 800-byte allocation.
    /// [`allocation_shape`](Self::allocation_shape) returns the buffer
    /// geometry.
    pub fn addressing_shape(&self, width: usize, height: usize) -> Option<Vec<usize>> {
        match self.layout() {
            // A channel dimension only when there is more than one channel:
            // `mono8` is `[h, w]`, not `[h, w, 1]`.
            PixelLayout::Packed if self.channels() == 1 => Some(vec![height, width]),
            PixelLayout::Packed => Some(vec![height, width, self.channels()]),
            PixelLayout::Planar => Some(vec![self.channels(), height, width]),
            // The luma/sample grid. Validated for the same dimensions as
            // `allocation_shape`.
            PixelLayout::SemiPlanar => {
                self.combined_plane_height(height)?;
                Some(vec![height, width])
            }
        }
    }

    /// Where each plane's bytes live, given the allocation's row pitch.
    ///
    /// `row_stride` is the byte pitch of a luma/packed row: the value
    /// [`Tensor::effective_row_stride`](crate::Tensor::effective_row_stride)
    /// reports, which on a DMA backing is 64-byte aligned and may exceed
    /// `width * bytes_per_pixel`. Planes are laid out contiguously in order (Y
    /// then UV; R then G then B).
    ///
    /// Returns `None` for an unsupported semi-planar variant, or if the
    /// geometry overflows `u64`.
    pub fn plane_table(
        &self,
        width: usize,
        height: usize,
        row_stride: usize,
    ) -> Option<Vec<PlaneGeometry>> {
        let _ = width;
        let stride = u64::try_from(row_stride).ok()?;
        let rows = u64::try_from(height).ok()?;
        let plane_bytes = stride.checked_mul(rows)?;

        match self.layout() {
            PixelLayout::Packed => Some(vec![PlaneGeometry {
                offset: 0,
                stride,
                size: plane_bytes,
            }]),
            PixelLayout::Planar => {
                let n = u64::try_from(self.channels()).ok()?;
                (0..n)
                    .map(|i| {
                        Some(PlaneGeometry {
                            offset: i.checked_mul(plane_bytes)?,
                            stride,
                            size: plane_bytes,
                        })
                    })
                    .collect()
            }
            PixelLayout::SemiPlanar => {
                let total_rows = u64::try_from(self.combined_plane_height(height)?).ok()?;
                let chroma_rows = total_rows.checked_sub(rows)?;
                Some(vec![
                    PlaneGeometry {
                        offset: 0,
                        stride,
                        size: plane_bytes,
                    },
                    PlaneGeometry {
                        offset: plane_bytes,
                        stride,
                        size: stride.checked_mul(chroma_rows)?,
                    },
                ])
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
    /// variants).
    pub const fn combined_plane_height(&self, height: usize) -> Option<usize> {
        match self {
            PixelFormat::Nv12 => Some(height + height.div_ceil(2)),
            PixelFormat::Nv16 => Some(height * 2),
            PixelFormat::Nv24 => Some(height * 3),
            _ => None,
        }
    }

    /// The image height whose combined plane height is `combined`, or `None`
    /// when no image height produces it (721 for NV12, say) and for every
    /// non-semi-planar format.
    ///
    /// The inverse of [`combined_plane_height`](Self::combined_plane_height),
    /// verified against it before returning.
    pub fn image_height_from_combined(&self, combined: usize) -> Option<usize> {
        let candidate = match self {
            PixelFormat::Nv12 => combined.checked_mul(2)? / 3,
            PixelFormat::Nv16 => combined / 2,
            PixelFormat::Nv24 => combined / 3,
            _ => return None,
        };
        (self.combined_plane_height(candidate) == Some(combined)).then_some(candidate)
    }

    /// Per-format semi-planar chroma addressing parameters. Returns `None` for
    /// non-semi-planar formats.
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
    /// The width is rounded up to the 64-aligned row pitch (`==
    /// bytes_per_row`).
    pub fn semi_planar_surface_dims(
        &self,
        width: usize,
        height: usize,
        bpe: usize,
    ) -> Option<(usize, usize)> {
        let total_h = self.combined_plane_height(height)?;
        // Round the byte pitch up to 64 and convert it back to texels.
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

    /// Returns `true` if this is an RGB or HSV pixel format.
    ///
    /// This is the `is_rgb_or_hsv` input of the kernel's
    /// `V4L2_MAP_QUANTIZATION_DEFAULT()` macro, which
    /// [`Colorimetry::from_v4l2`](crate::Colorimetry::from_v4l2) takes to
    /// resolve a `DEFAULT` quantization: these formats default to full range.
    ///
    /// `Rgb`, `Rgba`, `Bgra`, `PlanarRgb` and `PlanarRgba` return `true`.
    /// The YUV formats and `Grey` return `false`; the kernel lists greyscale
    /// as a YUV encoding. `PixelFormat` has no Bayer or HSV formats, which a
    /// V4L2 classifier by fourcc also counts as RGB.
    ///
    /// ```
    /// use edgefirst_tensor::PixelFormat;
    ///
    /// assert!(PixelFormat::Rgb.is_rgb_or_hsv());
    /// assert!(!PixelFormat::Yuyv.is_rgb_or_hsv());
    /// assert!(!PixelFormat::Grey.is_rgb_or_hsv());
    /// ```
    pub const fn is_rgb_or_hsv(&self) -> bool {
        match self {
            Self::Rgb | Self::Rgba | Self::Bgra | Self::PlanarRgb | Self::PlanarRgba => true,
            Self::Grey | Self::Yuyv | Self::Vyuy | Self::Nv12 | Self::Nv16 | Self::Nv24 => false,
        }
    }

    /// Returns `true` if this format includes an alpha channel.
    pub const fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba | Self::Bgra | Self::PlanarRgba)
    }

    /// Returns the V4L2 pixel format code (`V4L2_PIX_FMT_*`) for this
    /// format, or `0` for formats that have no V4L2 code (e.g., `PlanarRgb`).
    ///
    /// The codes for the four-channel and YUV formats coincide with the DRM
    /// FourCC for the same memory layout; `Rgb` (`RGB3`) and `Grey` (`GREY`)
    /// are V4L2-only spellings. This is not the [`PixelFormat::as_str`] wire
    /// name, nor the [`Display`](fmt::Display) text.
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

    /// Converts a V4L2 pixel format code (see [`Self::to_fourcc`]) to a
    /// `PixelFormat`, returning `None` for unrecognized or zero codes.
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

/// Human-readable name for messages and logs. Neither the wire name
/// ([`PixelFormat::as_str`]) nor the V4L2 code ([`PixelFormat::to_fourcc`]).
impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Rgb => "RGB",
            Self::Rgba => "RGBA",
            Self::Bgra => "BGRA",
            Self::Grey => "Y800",
            Self::Yuyv => "YUYV",
            Self::Vyuy => "VYUY",
            Self::Nv12 => "NV12",
            Self::Nv16 => "NV16",
            Self::Nv24 => "NV24",
            Self::PlanarRgb => "PlanarRgb",
            Self::PlanarRgba => "PlanarRgba",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_layout_matches_the_subsampling_of_every_format() {
        for &fmt in PixelFormat::all() {
            let expected = match fmt {
                PixelFormat::Nv12 => Some(ChromaLayout {
                    shift_x: 1,
                    shift_y: 1,
                    uv_rows_per_luma: 1,
                }),
                PixelFormat::Nv16 => Some(ChromaLayout {
                    shift_x: 1,
                    shift_y: 0,
                    uv_rows_per_luma: 1,
                }),
                PixelFormat::Nv24 => Some(ChromaLayout {
                    shift_x: 0,
                    shift_y: 0,
                    uv_rows_per_luma: 2,
                }),
                _ => None,
            };
            assert_eq!(fmt.chroma_layout(), expected, "{fmt:?}");
        }
    }

    #[test]
    fn semi_planar_surface_dims_rounds_the_byte_pitch_to_64() {
        // Two bytes per element: 100 texels = 200 bytes, padded to 256 bytes
        // = 128 texels.
        assert_eq!(
            PixelFormat::Nv12.semi_planar_surface_dims(100, 50, 2),
            Some((128, 75))
        );
        assert_eq!(
            PixelFormat::Nv12.semi_planar_surface_dims(100, 50, 1),
            Some((128, 75))
        );
        assert_eq!(
            PixelFormat::Nv16.semi_planar_surface_dims(100, 50, 1),
            Some((128, 100))
        );
        assert_eq!(
            PixelFormat::Nv24.semi_planar_surface_dims(321, 241, 1),
            Some((384, 723))
        );
        // Four bytes per element: 40 texels = 160 bytes, padded to 192 bytes
        // = 48 texels.
        assert_eq!(
            PixelFormat::Nv24.semi_planar_surface_dims(40, 8, 4),
            Some((48, 24))
        );
        assert_eq!(
            PixelFormat::Nv12.semi_planar_surface_dims(64, 64, 1),
            Some((64, 96))
        );
    }

    #[test]
    fn semi_planar_surface_dims_is_none_for_non_semi_planar_formats() {
        for &fmt in PixelFormat::all() {
            if fmt.chroma_layout().is_none() {
                assert_eq!(fmt.semi_planar_surface_dims(64, 64, 1), None, "{fmt:?}");
            }
        }
    }

    #[test]
    fn addressing_shape_is_the_grid_not_the_allocation() {
        assert_eq!(
            PixelFormat::Nv12.addressing_shape(640, 480).unwrap(),
            vec![480, 640]
        );
        assert_eq!(
            PixelFormat::Rgb.addressing_shape(640, 480).unwrap(),
            vec![480, 640, 3]
        );
        assert_eq!(
            PixelFormat::PlanarRgb.addressing_shape(640, 480).unwrap(),
            vec![3, 480, 640]
        );
        assert_eq!(
            PixelFormat::Grey.addressing_shape(4, 2).unwrap(),
            vec![2, 4]
        );
    }

    #[test]
    fn allocation_shape_still_sizes_the_buffer() {
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 480).unwrap(),
            vec![720, 640]
        );
        // Odd height stays exact: 483 luma + 242 chroma rows.
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 483).unwrap(),
            vec![725, 640]
        );
        for f in [PixelFormat::Rgb, PixelFormat::PlanarRgb, PixelFormat::Rgba] {
            assert_eq!(
                f.allocation_shape(64, 48),
                f.addressing_shape(64, 48),
                "{f:?}: allocation and grid coincide when nothing is subsampled"
            );
        }
        assert_eq!(
            PixelFormat::Grey.allocation_shape(64, 48).unwrap(),
            vec![48, 64, 1]
        );
        assert_eq!(
            PixelFormat::Grey.addressing_shape(64, 48).unwrap(),
            vec![48, 64]
        );
        let a: usize = PixelFormat::Grey
            .allocation_shape(64, 48)
            .unwrap()
            .iter()
            .product();
        let g: usize = PixelFormat::Grey
            .addressing_shape(64, 48)
            .unwrap()
            .iter()
            .product();
        assert_eq!(
            a, g,
            "dropping a 1-sized dimension cannot change the element count"
        );
    }

    #[test]
    fn the_grid_is_not_the_buffer_size_for_subsampled_formats() {
        let grid: usize = PixelFormat::Nv12
            .addressing_shape(640, 480)
            .unwrap()
            .iter()
            .product();
        let alloc: usize = PixelFormat::Nv12
            .allocation_shape(640, 480)
            .unwrap()
            .iter()
            .product();
        assert_eq!(grid, 640 * 480, "the grid addresses luma samples only");
        assert_eq!(alloc, 640 * 480 * 3 / 2, "the buffer holds luma + chroma");
        assert!(
            grid < alloc,
            "product(shape) * dtype_size is NOT the buffer size"
        );
    }

    #[test]
    fn nv12_plane_table_matches_the_schemas_golden() {
        let planes = PixelFormat::Nv12.plane_table(640, 480, 640).unwrap();
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].offset, 0);
        assert_eq!(planes[0].stride, 640);
        assert_eq!(planes[0].size, 640 * 480);
        assert_eq!(planes[1].offset, 640 * 480);
        assert_eq!(planes[1].stride, 640);
        assert_eq!(planes[1].size, 640 * 480 / 2);
    }

    #[test]
    fn plane_counts_follow_the_spec_table() {
        assert_eq!(PixelFormat::Rgb.plane_table(64, 48, 192).unwrap().len(), 1);
        assert_eq!(PixelFormat::Grey.plane_table(64, 48, 64).unwrap().len(), 1);
        assert_eq!(
            PixelFormat::PlanarRgb
                .plane_table(64, 48, 64)
                .unwrap()
                .len(),
            3
        );
        assert_eq!(PixelFormat::Nv12.plane_table(64, 48, 64).unwrap().len(), 2);
    }

    #[test]
    fn plane_table_honours_a_padded_luma_stride() {
        let planes = PixelFormat::Nv12.plane_table(640, 480, 704).unwrap();
        assert_eq!(planes[0].stride, 704);
        assert_eq!(
            planes[1].offset,
            704 * 480,
            "chroma starts after the PADDED luma plane, not after w*h"
        );
        assert_eq!(planes[1].size, 704 * 240);
    }

    #[test]
    fn plane_table_extent_agrees_with_allocation_shape() {
        for (w, h) in [(640, 480), (789, 383), (64, 1)] {
            let stride = w;
            let planes = PixelFormat::Nv12.plane_table(w, h, stride).unwrap();
            let total: u64 = planes.iter().map(|p| p.size).sum();
            let rows = PixelFormat::Nv12.combined_plane_height(h).unwrap();
            assert_eq!(
                total,
                (stride * rows) as u64,
                "{w}x{h}: plane bytes must equal combined_plane_height * stride"
            );
        }
    }

    #[test]
    fn plane_table_rejects_overflow_rather_than_wrapping() {
        assert!(PixelFormat::Nv12
            .plane_table(usize::MAX, usize::MAX, usize::MAX)
            .is_none());
    }

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
    fn is_rgb_or_hsv_is_every_format_but_yuv_and_grey() {
        for &format in PixelFormat::all() {
            assert_eq!(
                format.is_rgb_or_hsv(),
                !format.is_yuv() && format != PixelFormat::Grey,
                "{format:?}"
            );
        }
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
    fn combined_plane_height_inverts_for_every_semi_planar_height() {
        for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
            for h in 1..=64usize {
                let combined = fmt.combined_plane_height(h).unwrap();
                assert_eq!(
                    fmt.image_height_from_combined(combined),
                    Some(h),
                    "{fmt:?} height {h} -> combined {combined}"
                );
            }
        }
    }

    #[test]
    fn image_height_from_combined_rejects_unproduced_heights() {
        // NV12 maps 480 -> 720 and 481 -> 722; nothing maps to 721.
        assert_eq!(PixelFormat::Nv12.image_height_from_combined(721), None);
        // NV16 doubles, so every odd combined height is unreachable.
        assert_eq!(PixelFormat::Nv16.image_height_from_combined(7), None);
        // NV24 triples.
        assert_eq!(PixelFormat::Nv24.image_height_from_combined(8), None);
    }

    #[test]
    fn image_height_from_combined_is_none_for_non_semi_planar() {
        for fmt in [
            PixelFormat::Rgba,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Yuyv,
            PixelFormat::PlanarRgb,
        ] {
            assert_eq!(fmt.image_height_from_combined(12), None, "{fmt:?}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn to_fourcc_matches_edgefirst_v4l2() {
        use edgefirst_v4l2::uapi;
        let expect = [
            (PixelFormat::Rgb, uapi::V4L2_PIX_FMT_RGB24),
            // `V4L2_PIX_FMT_RGBA32`, built with `uapi::fourcc`.
            (PixelFormat::Rgba, uapi::fourcc(b'A', b'B', b'2', b'4')),
            (PixelFormat::Bgra, uapi::V4L2_PIX_FMT_ABGR32),
            (PixelFormat::Grey, uapi::V4L2_PIX_FMT_GREY),
            (PixelFormat::Yuyv, uapi::V4L2_PIX_FMT_YUYV),
            (PixelFormat::Vyuy, uapi::V4L2_PIX_FMT_VYUY),
            (PixelFormat::Nv12, uapi::V4L2_PIX_FMT_NV12),
            (PixelFormat::Nv16, uapi::V4L2_PIX_FMT_NV16),
            (PixelFormat::Nv24, uapi::V4L2_PIX_FMT_NV24),
        ];
        for (fmt, code) in expect {
            assert_eq!(
                fmt.to_fourcc(),
                code,
                "{fmt:?}: {} vs {}",
                uapi::fourcc_str(fmt.to_fourcc()),
                uapi::fourcc_str(code)
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
        assert_eq!(format!("{}", PixelFormat::Rgb), "RGB");
        assert_eq!(format!("{}", PixelFormat::Bgra), "BGRA");
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
