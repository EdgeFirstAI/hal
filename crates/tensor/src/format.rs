// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::fmt;

// Declared through `ef_vocabulary!` (see `vocabulary.rs`): `.code()` is this
// crate's wire numbering, mirrored by `protocol::format` and (via an
// explicit, fallible outlier mapping -- see `to_hal_pixel_format` in the
// capi crate) the C ABI's `hal_pixel_format`, which was numbered
// independently before this vocabulary was unified and keeps its own frozen
// values.
//
// Stays `#[non_exhaustive]`, deliberately: this attribute protects
// DOWNSTREAM, not us -- `profiler`, `mobile-sdk` and any third party with an
// exhaustive `match PixelFormat` keep compiling the day a variant such as
// `Nv21` is added here. Removing it would narrow a published promise, and
// narrowing is not reversible: re-adding it later breaks every downstream
// exhaustive match that has grown in the meantime. That a new variant is
// already a coordinated, single-declaration change *in this repo* (the
// macro's whole point) says nothing about matches written outside it.
// Consequence: a C mapping of `PixelFormat` cannot be a
// compile-enforced exhaustive match (rustc requires a wildcard arm on a
// `#[non_exhaustive]` enum outside its defining crate) -- it keeps a
// wildcard `Err` arm.
//
// `as_str()`'s wire strings are a THIRD spelling, deliberately distinct from
// both `Display` (FourCC-derived: `RGB`, `Y800`, ...) and the `serde` derive
// (PascalCase variant names: `Rgb`, `PlanarRgb`, ...) -- those two existed
// before this design and stay as they are; unifying them with the wire
// vocabulary is not a cleanup, it is a breaking change to two formats that
// are not this one. This declaration is the format table the schemas repo's
// `Tensor.msg` cites as canonical, derived by grepping every format-string
// literal already in use across that repo:
//
// | Variant      | `as_str()`     | Convention                                |
// |--------------|----------------|--------------------------------------------|
// | `Rgb`        | `"rgb8"`       | ROS `sensor_msgs/Image` encoding (36 uses)  |
// | `Rgba`       | `"rgba8"`      | ROS encoding                               |
// | `Bgra`       | `"bgra8"`      | ROS encoding                               |
// | `Grey`       | `"mono8"`      | ROS encoding (2 uses)                      |
// | `Yuyv`       | `"YUYV"`       | FourCC                                     |
// | `Vyuy`       | `"VYUY"`       | FourCC                                     |
// | `Nv12`       | `"NV12"`       | FourCC (11 uses, vs. 3 stray `"nv12"`)     |
// | `Nv16`       | `"NV16"`       | FourCC                                     |
// | `PlanarRgb`  | `"rgb8_planar"`  | ROS encoding, planar; no dimension-order  |
// | `PlanarRgba` | `"rgba8_planar"` | suffix -- `shape` is the addressing grid  |
// | `Nv24`       | `"NV24"`       | FourCC                                     |
//
// Case follows the format's home ecosystem, not one blanket rule: RGB-family
// names and `Grey`/`mono8` come from ROS `sensor_msgs/Image`, which is
// lowercase; YUV-family names are FourCC codes, which are uppercase. The two
// planar spellings drop the `_nchw` suffix the one pre-existing schemas-repo
// site (`rgb8_planar_nchw`) still carries -- redundant now that dimension
// order lives in `shape`, not in the format name. That site needs
// reconciling in the schemas repo; it is not changed here.
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
        /// chroma: Y plane (H rows of W bytes) + interleaved Cb/Cr plane (H image
        /// rows of W pairs = 2W bytes/row, laid out as 2H rows of W) → 3H rows
        /// total. Multiplane NV24 is not yet supported (see `from_planes`). Added
        /// last to keep the existing discriminants (and any serialized values)
        /// stable.
        Nv24 = 11, "NV24", NV24,
    }
    // `#[doc(hidden)]`: has to be `pub` for `protocol::format`'s re-export
    // below to compile (see `ef_vocabulary!`'s doc comment), but it is
    // emission plumbing, not a second documented API -- `protocol::format`
    // is the canonical, documented path.
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

/// Where one plane's bytes physically live inside a tensor's allocation.
///
/// The counterpart to [`PixelFormat::addressing_shape`]: the grid says how a
/// consumer indexes the image, this says where the bytes are. Buffer extent
/// comes from here, never from the shape — for a subsampled format
/// `product(shape) * dtype_size` is smaller than the allocation.
///
/// Field names and widths mirror `TensorPlane.msg` so the two map 1:1.
/// `modifier` is deliberately absent: it describes how a *producer* tiled or
/// compressed the plane (AFBC on Mali, UBWC on Adreno) and cannot be derived
/// from the format alone, so it is carried on the tensor rather than computed
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneGeometry {
    /// Byte offset of this plane from the start of the allocation.
    pub offset: u64,
    /// Bytes per line of this plane.
    pub stride: u64,
    /// Plane extent in bytes (`stride * rows`).
    pub size: u64,
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

    /// The **allocation geometry** for this format at `width`×`height`, or
    /// `None` if the dimensions are invalid for the format, or the format is
    /// an unsupported semi-planar variant (any `SemiPlanar` variant other than
    /// `Nv12`, `Nv16`, and `Nv24`).
    ///
    /// This is the shape that *sizes a buffer*. For a subsampled format it is
    /// deliberately not the shape a consumer indexes with — NV12 640×480
    /// allocates `[720, 640]` here but addresses a `[480, 640]` luma grid.
    /// Use [`addressing_shape`](Self::addressing_shape) for the latter, which
    /// is what descriptors, blobs and `Tensor.msg` carry.
    ///
    /// Renamed from `image_shape` when the two meanings were separated: every
    /// caller of the old name sized an allocation with it (`Tensor::image`,
    /// `import_image`, and the two PBO helpers), so a single name called "the
    /// shape" made picking the wrong one the default. There is now no such
    /// name.
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
    pub fn allocation_shape(&self, width: usize, height: usize) -> Option<Vec<usize>> {
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

    /// The **addressing grid**: how a consumer logically indexes this image.
    ///
    /// This is the shape that belongs in a descriptor, a blob, or a
    /// `Tensor.msg` — never the allocation size. For a subsampled format
    /// `product(addressing_shape) * dtype_size` is *smaller* than the buffer:
    /// NV12 640×480 addresses a 640×480 luma grid inside a 460 800-byte
    /// allocation. Buffer extent comes from the plane table, never from here.
    ///
    /// Contrast [`allocation_shape`](Self::allocation_shape), which is the
    /// buffer geometry every allocating caller must keep using.
    ///
    /// The per-format convention is pinned here so producers cannot each
    /// invent one; it matches the merged `schemas` golden fixtures.
    pub fn addressing_shape(&self, width: usize, height: usize) -> Option<Vec<usize>> {
        match self.layout() {
            // A channel dimension only when there is more than one channel to
            // address: `mono8` is `[h, w]`, not `[h, w, 1]`. That keeps it
            // consistent with the semi-planar grid below (also one sample per
            // pixel position), matches how every array library presents
            // grayscale, and matches the merged schemas `mono8` golden. The
            // spec's format table does not cover single-channel packed
            // formats; this is the rule chosen for them.
            PixelLayout::Packed if self.channels() == 1 => Some(vec![height, width]),
            PixelLayout::Packed => Some(vec![height, width, self.channels()]),
            PixelLayout::Planar => Some(vec![self.channels(), height, width]),
            // The luma/sample grid. Chroma is subsampled and is not separately
            // addressable through `shape`; its geometry lives in the plane
            // table. Validated for the same dimension support as
            // `allocation_shape` so the two agree on what they accept.
            PixelLayout::SemiPlanar => {
                self.combined_plane_height(height)?;
                Some(vec![height, width])
            }
        }
    }

    /// Where each plane's bytes live, given the allocation's row pitch.
    ///
    /// `row_stride` is the byte pitch of a **luma/packed** row — the value
    /// [`Tensor::effective_row_stride`](crate::Tensor::effective_row_stride)
    /// reports, which on a DMA backing is 64-byte aligned and may exceed
    /// `width * bytes_per_pixel`. Planes are laid out contiguously in the
    /// order a consumer expects them (Y then UV; R then G then B).
    ///
    /// Returns `None` for an unsupported semi-planar variant, or if the
    /// geometry overflows `u64` — dimensions reaching this from an imported
    /// descriptor are untrusted, and a wrapped size would look small enough
    /// to pass a capacity check it should fail.
    ///
    /// Chroma extent is derived from
    /// [`combined_plane_height`](Self::combined_plane_height) rather than
    /// recomputed, so the two cannot drift on odd heights.
    pub fn plane_table(
        &self,
        width: usize,
        height: usize,
        row_stride: usize,
    ) -> Option<Vec<PlaneGeometry>> {
        let _ = width; // extent is governed by row_stride, not logical width
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
                // Derived, never recomputed: `H + ceil(H/2)` for NV12 is exact
                // for odd heights where a hand-written `H/2` is not.
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
    /// variants). This is the single source of truth for the vertical extent of
    /// the contiguous NV* buffer — [`allocation_shape`](Self::allocation_shape), the GL
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
        // allocation_shape carries the logical width; round its byte pitch up to 64
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
    fn addressing_shape_is_the_grid_not_the_allocation() {
        // The spec's format table, and the merged schemas goldens
        // (schemas 71b56ed, fixture "Tensor": NV12 640x480 -> shape [480, 640]).
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
        // schemas fixture "Tensor_inline" uses format "mono8" with shape [2, 4].
        assert_eq!(
            PixelFormat::Grey.addressing_shape(4, 2).unwrap(),
            vec![2, 4]
        );
    }

    #[test]
    fn allocation_shape_still_sizes_the_buffer() {
        // Unchanged from `image_shape`: NV12 keeps the combined-plane height,
        // because four callers size real allocations with this.
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 480).unwrap(),
            vec![720, 640]
        );
        // Odd height stays exact: 483 luma + 242 chroma rows.
        assert_eq!(
            PixelFormat::Nv12.allocation_shape(640, 483).unwrap(),
            vec![725, 640]
        );
        // Multi-channel, non-subsampled formats: the two agree exactly.
        for f in [PixelFormat::Rgb, PixelFormat::PlanarRgb, PixelFormat::Rgba] {
            assert_eq!(
                f.allocation_shape(64, 48),
                f.addressing_shape(64, 48),
                "{f:?}: allocation and grid coincide when nothing is subsampled"
            );
        }
        // Single-channel packed is the second place they legitimately differ:
        // the allocation keeps its trailing 1-channel dimension, the grid
        // drops it. Same element count either way, so nothing is mis-sized.
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
        // The consequence the spec says must be documented, not discovered.
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
        // (golden) schemas 71b56ed, scripts/generate_cdr_testdata.py fixture
        // "Tensor": one allocation, Y at 0 and UV at w*h, both stride 640.
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
        // The case with no representation before the plane table existed: a
        // 64-byte-aligned luma pitch wider than the logical width pushes
        // chroma's offset out. Implicit `w*h` chroma placement gets this wrong.
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
        // The two must never drift: total plane bytes == the allocation the
        // combined-plane height implies. Odd height included, since that is
        // where a duplicated `h/2` would diverge from `h + ceil(h/2)`.
        for (w, h) in [(640, 480), (789, 383), (64, 1)] {
            let stride = w; // tight, so the comparison is exact
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
        // Dimensions come from untrusted descriptors elsewhere in this crate;
        // a wrapped size would look small enough to pass a capacity check.
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
