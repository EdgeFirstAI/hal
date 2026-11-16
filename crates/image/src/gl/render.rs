// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Portable GL render lowering: the pure geometry and planning math the
//! tile/batch renderer uses. No platform types appear here.
//!
//! * `region_to_viewport`: the bottom-left y-origin flip for a bottom-up
//!   render target. A tile at pixel `(x, y)` of size `(w, h)` in a
//!   `parent_h`-row target maps to viewport `y' = parent_h − (y + h)`.
//! * `source_uv`: the source sampling rectangle in normalised `[0,1]` UV,
//!   derived from a `Region`.
//! * `plan_batch`: the chunk planner. A batch of `n` destination tiles
//!   renders into one reused import when every tile fits the GPU's
//!   `GL_MAX_*` limits; otherwise it splits into chunks (one import per
//!   chunk) or one import per tile.

use crate::Region;
use edgefirst_tensor::{PixelFormat, PixelLayout, TensorMemory};

/// How a convert's destination is realized on the GPU. One lowering per
/// destination memory class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DstLowering {
    /// The destination buffer itself becomes the FBO colour attachment. The
    /// render writes the buffer directly, with no readback.
    ZeroCopy,
    /// Offscreen texture render target, read back into the mapped tensor.
    TextureMem,
    /// Offscreen texture render target, read back into the destination PBO's
    /// PACK binding.
    TexturePbo,
}

/// Classify the destination lowering from the platform's zero-copy import
/// capability and the destination's memory class. A DMA destination without
/// import support lowers to the mapped texture path.
pub(super) fn lower_dst(zero_copy_import: bool, dst_mem: TensorMemory) -> DstLowering {
    match dst_mem {
        TensorMemory::DmaBuf if zero_copy_import => DstLowering::ZeroCopy,
        TensorMemory::Pbo => DstLowering::TexturePbo,
        _ => DstLowering::TextureMem,
    }
}

/// How a convert renders. One plan per (source format, source residence,
/// destination format, destination lowering).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConvertPlan {
    /// One render pass into the bound destination target (packed via the
    /// texture shaders, planar via the planar shader), then the lowering's
    /// readback (none for zero-copy).
    SinglePass,
    /// Zero-copy packed-RGB only: pass 1 renders RGBA into an intermediate
    /// texture, pass 2 packs it into the destination reinterpreted as
    /// RGBA8 at `W*3/4 × H`.
    TwoPassPackedRgb,
    /// Planar destination: pass 1 renders the source into an intermediate
    /// RGBA texture through the ordinary packed convert, pass 2 deinterleaves
    /// RGBA into the planar destination. Selected on every lowering for NV*
    /// and packed Rgb sources and for host-memory sources, and on texture
    /// lowerings for every source.
    TwoPassNvPlanar,
    /// Grey destination from an RGB-family source: pass 1 renders RGBA into
    /// an intermediate texture through the ordinary packed convert, pass 2
    /// writes its BT.601 luma into the destination. YUV sources stay
    /// single-pass, drawing their own luma plane.
    TwoPassGrey,
}

/// Decide the render plan from the source format, source residence,
/// destination format and destination lowering. `src_host` is a source in host
/// memory (`Mem`, `Shm`).
pub(super) fn plan_convert(
    src_fmt: PixelFormat,
    src_host: bool,
    dst_fmt: PixelFormat,
    lowering: DstLowering,
) -> ConvertPlan {
    // A zero-copy packed RGB destination takes the two-pass W*3/4 reinterpretation.
    if lowering == DstLowering::ZeroCopy && dst_fmt == PixelFormat::Rgb {
        return ConvertPlan::TwoPassPackedRgb;
    }
    if dst_fmt == PixelFormat::Grey
        && matches!(
            src_fmt,
            PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Bgra
        )
    {
        return ConvertPlan::TwoPassGrey;
    }
    // A texture-lowered planar destination is two-pass for every source format.
    if dst_fmt.layout() == PixelLayout::Planar && lowering != DstLowering::ZeroCopy {
        return ConvertPlan::TwoPassNvPlanar;
    }
    // A zero-copy planar destination is two-pass for a host source.
    if dst_fmt.layout() == PixelLayout::Planar && src_host {
        return ConvertPlan::TwoPassNvPlanar;
    }
    // A zero-copy planar destination is two-pass for NV* and packed Rgb sources.
    if dst_fmt.layout() == PixelLayout::Planar
        && matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24 | PixelFormat::Rgb
        )
    {
        return ConvertPlan::TwoPassNvPlanar;
    }
    ConvertPlan::SinglePass
}

/// A GL viewport / scissor rectangle in pixels, bottom-left origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Viewport {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) w: i32,
    pub(super) h: i32,
}

/// Lower a destination tile to its GL band rectangle on a top-down target (GL
/// framebuffer row 0 == memory row 0): the band's GL `y` is `region.y`. Used
/// for both the band `glViewport` and the matching `glScissor`.
#[inline]
pub(super) fn region_to_viewport_top_down(region: Region) -> Viewport {
    Viewport {
        x: region.x as i32,
        y: region.y as i32,
        w: region.width as i32,
        h: region.height as i32,
    }
}

/// Lower a top-left `region` of an image `parent_h` rows tall to a bottom-left
/// GL viewport. The horizontal axis is unchanged; the vertical axis is flipped.
#[allow(dead_code)]
pub(super) fn region_to_viewport(region: Region, parent_h: usize) -> Viewport {
    // Bottom-left origin: the GL y of the region is the number of rows below
    // it, `parent_h - (y + h)`, saturating at 0.
    let y_flipped = parent_h.saturating_sub(region.y.saturating_add(region.height));
    Viewport {
        x: region.x as i32,
        y: y_flipped as i32,
        w: region.width as i32,
        h: region.height as i32,
    }
}

/// Source sampling rectangle as normalised `[u_min, v_min, u_extent, v_extent]`
/// over a `src_w × src_h` texture. `None` (whole source) is the identity
/// `[0, 0, 1, 1]`.
#[allow(dead_code)]
pub(super) fn source_uv(region: Option<Region>, src_w: usize, src_h: usize) -> [f32; 4] {
    match region {
        Some(r) if src_w > 0 && src_h > 0 => [
            r.x as f32 / src_w as f32,
            r.y as f32 / src_h as f32,
            r.width as f32 / src_w as f32,
            r.height as f32 / src_h as f32,
        ],
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

/// How a batch of `n` destination tiles maps onto GPU imports.
///
/// `OneImport`: the whole destination is imported once and each tile is a
/// `glViewport` into it. `Chunked(tiles_per_chunk)`: the batch is split into
/// chunks of `tiles_per_chunk`, one import per chunk. `PerTileImport`: one
/// import per tile.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchPath {
    OneImport,
    Chunked(usize),
    PerTileImport,
}

/// Decide the batch path from the per-tile render-target row count and the GPU
/// row limit (`GL_MAX_TEXTURE_SIZE` / `GL_MAX_VIEWPORT_DIMS`, whichever binds).
///
/// `rows_per_tile` is the render target height a single tile occupies (`H` for a
/// packed `HWC` destination, `3·H` for a planar `CHW` packed render target).
/// `max_rows` is the device limit. The result is:
///
/// * `n_tiles · rows_per_tile ≤ max_rows`  → `OneImport`
/// * `rows_per_tile ≤ max_rows < n·rows`   → `Chunked(max_rows / rows_per_tile)`
/// * `rows_per_tile > max_rows`            → `PerTileImport`
#[allow(dead_code)]
pub(super) fn plan_batch(n_tiles: usize, rows_per_tile: usize, max_rows: usize) -> BatchPath {
    if rows_per_tile == 0 || n_tiles == 0 {
        return BatchPath::OneImport;
    }
    if rows_per_tile > max_rows {
        return BatchPath::PerTileImport;
    }
    let tiles_per_chunk = max_rows / rows_per_tile;
    if tiles_per_chunk >= n_tiles {
        BatchPath::OneImport
    } else {
        BatchPath::Chunked(tiles_per_chunk)
    }
}

/// Where a tensor's logical image sits on the texture its import covers. A
/// platform whose import is the logical image reports [`ImportMap::WHOLE`],
/// which every helper here treats as the identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ImportMap {
    /// Texel size of the imported texture when it can exceed the logical
    /// image ([`super::platform::GlPlatform::import_extent`]); `None` when
    /// the import is exactly the logical image.
    pub(super) extent: Option<(u32, u32)>,
    /// Texels from the texture's origin to the logical image's first pixel
    /// ([`super::platform::GlPlatform::import_origin`]).
    pub(super) origin: (u32, u32),
}

impl ImportMap {
    /// The import is exactly the logical image: no scale, no shift.
    pub(super) const WHOLE: Self = Self {
        extent: None,
        origin: (0, 0),
    };
}

/// How one axis of a logical image maps onto the texture an import covers:
/// `(origin, scale, limit)` — the normalized coordinate the logical image
/// starts at, the factor that places it over its share of the texture, and the
/// normalized coordinate of its far edge. A coordinate `t` over the logical
/// image lands at `(origin + t * logical) / extent`.
///
/// A zero `extent` or `logical`, and a map whose shifted image would run past
/// the texture, map as the identity `(0, 1, 1)`.
fn axis_map(logical: usize, origin: u32, extent: u32) -> (f32, f32, f32) {
    let span = (origin as u64).saturating_add(logical as u64);
    if extent == 0 || logical == 0 || span > extent as u64 {
        return (0.0, 1.0, 1.0);
    }
    let texture = extent as f32;
    (
        origin as f32 / texture,
        logical as f32 / texture,
        span as f32 / texture,
    )
}

/// Map a source rectangle from the tensor's logical image onto the larger
/// texture its import covers.
///
/// `roi` is normalized over the logical image; `map` is where that image sits
/// on the texture. Each corner is shifted and scaled per axis through
/// [`axis_map`] and held inside the logical image's far edge. A map with no
/// extent leaves `roi` untouched.
pub(super) fn scale_roi_to_import(
    roi: super::RegionOfInterest,
    logical: (usize, usize),
    map: ImportMap,
) -> super::RegionOfInterest {
    let Some((pw, ph)) = map.extent else {
        return roi;
    };
    let (ox, sx, lx) = axis_map(logical.0, map.origin.0, pw);
    let (oy, sy, ly) = axis_map(logical.1, map.origin.1, ph);
    super::RegionOfInterest {
        left: (ox + roi.left * sx).min(lx),
        top: (oy + roi.top * sy).min(ly),
        right: (ox + roi.right * sx).min(lx),
        bottom: (oy + roi.bottom * sy).min(ly),
    }
}

/// [`scale_roi_to_import`] for the float paths' `src_rect_uv`, which is an
/// origin and a size rather than two corners: the origin shifts and scales,
/// and the size is trimmed to keep the far edge inside the same limit.
pub(super) fn scale_uv_rect_to_import(
    rect: [f32; 4],
    logical: (usize, usize),
    map: ImportMap,
) -> [f32; 4] {
    let Some((pw, ph)) = map.extent else {
        return rect;
    };
    let (ox, sx, lx) = axis_map(logical.0, map.origin.0, pw);
    let (oy, sy, ly) = axis_map(logical.1, map.origin.1, ph);
    let left = (ox + rect[0] * sx).min(lx);
    let top = (oy + rect[1] * sy).min(ly);
    [
        left,
        top,
        (rect[2] * sx).min(lx - left).max(0.0),
        (rect[3] * sy).min(ly - top).max(0.0),
    ]
}

/// The rectangle a source sample may reach, as the `src_extent` uniform of
/// the source-sampling shaders: `[u_min, v_min, u_max, v_max]`, the logical
/// image's share of the texture in normalized coordinates, inset by half a
/// physical texel on each side.
///
/// A degenerate logical size yields the whole texture, `[0, 0, 1, 1]`.
pub(super) fn sample_clamp_rect(logical: (usize, usize), map: ImportMap) -> [f32; 4] {
    sample_clamp_rect_within(logical, map, [0.0, 0.0, 1.0, 1.0])
}

/// [`sample_clamp_rect`] narrowed to a source crop: `crop` is
/// `[u_min, v_min, u_max, v_max]` normalized to the logical image, and the
/// result brackets the crop's texels.
pub(super) fn sample_clamp_rect_within(
    logical: (usize, usize),
    map: ImportMap,
    crop: [f32; 4],
) -> [f32; 4] {
    fn axis(logical: usize, origin: u32, extent: Option<u32>, lo: f32, hi: f32) -> (f32, f32) {
        if logical == 0 {
            return (0.0, 1.0);
        }
        let span = (origin as u64).saturating_add(logical as u64);
        let (texture, first) = match extent {
            Some(e) if e as u64 > logical as u64 && e as u64 >= span => (e as f32, origin as f32),
            // No extent, an extent no larger than the image, or a
            // contradictory shift: the texture is the image at its origin.
            _ => (logical as f32, 0.0),
        };
        let lo = (lo * logical as f32).clamp(0.0, logical as f32);
        let hi = (hi * logical as f32).clamp(lo, logical as f32);
        // A crop narrower than one texel collapses to its centre.
        let (lo, hi) = if hi - lo < 1.0 {
            let c = (lo + hi) * 0.5;
            (c, c)
        } else {
            (lo + 0.5, hi - 0.5)
        };
        ((first + lo) / texture, (first + hi) / texture)
    }
    let (u_min, u_max) = axis(
        logical.0,
        map.origin.0,
        map.extent.map(|e| e.0),
        crop[0],
        crop[2],
    );
    let (v_min, v_max) = axis(
        logical.1,
        map.origin.1,
        map.extent.map(|e| e.1),
        crop[1],
        crop[3],
    );
    [u_min, v_min, u_max, v_max]
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn region(x: usize, y: usize, w: usize, h: usize) -> Region {
        Region::new(x, y, w, h)
    }

    #[test]
    fn viewport_top_down_is_identity() {
        let ys: Vec<i32> = (0..3)
            .map(|n| region_to_viewport_top_down(region(0, n * 16, 64, 16)).y)
            .collect();
        assert_eq!(ys, vec![0, 16, 32]);
        assert_eq!(
            region_to_viewport_top_down(region(10, 4, 20, 8)),
            Viewport {
                x: 10,
                y: 4,
                w: 20,
                h: 8
            }
        );
    }

    #[test]
    fn viewport_whole_image_is_origin() {
        let vp = region_to_viewport(region(0, 0, 64, 48), 48);
        assert_eq!(
            vp,
            Viewport {
                x: 0,
                y: 0,
                w: 64,
                h: 48
            }
        );
    }

    #[test]
    fn viewport_top_tile_lands_at_top_in_gl_coords() {
        let vp = region_to_viewport(region(0, 0, 64, 16), 48);
        assert_eq!(
            vp,
            Viewport {
                x: 0,
                y: 32,
                w: 64,
                h: 16
            }
        );
    }

    #[test]
    fn viewport_bottom_tile_lands_at_origin() {
        let vp = region_to_viewport(region(0, 32, 64, 16), 48);
        assert_eq!(
            vp,
            Viewport {
                x: 0,
                y: 0,
                w: 64,
                h: 16
            }
        );
    }

    #[test]
    fn viewport_batch_bands_tile_without_overlap() {
        let h = 16;
        let ys: Vec<i32> = (0..3)
            .map(|n| region_to_viewport(region(0, n * h, 64, h), 3 * h).y)
            .collect();
        assert_eq!(ys, vec![32, 16, 0]);
    }

    #[test]
    fn viewport_x_offset_preserved() {
        let vp = region_to_viewport(region(10, 4, 20, 8), 32);
        assert_eq!(
            vp,
            Viewport {
                x: 10,
                y: 32 - 12,
                w: 20,
                h: 8
            }
        );
    }

    #[test]
    fn source_uv_none_is_identity() {
        assert_eq!(source_uv(None, 640, 480), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn source_uv_half_rect() {
        assert_eq!(
            source_uv(Some(region(320, 0, 320, 480)), 640, 480),
            [0.5, 0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn source_uv_zero_dims_falls_back_to_identity() {
        assert_eq!(
            source_uv(Some(region(0, 0, 1, 1)), 0, 0),
            [0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn plan_batch_fits_one_import() {
        assert_eq!(plan_batch(4, 64, 2048), BatchPath::OneImport);
    }

    #[test]
    fn plan_batch_chunks_when_over_limit() {
        assert_eq!(plan_batch(40, 64, 2048), BatchPath::Chunked(32));
    }

    #[test]
    fn plan_batch_single_oversize_tile_is_per_tile() {
        assert_eq!(plan_batch(8, 3072, 2048), BatchPath::PerTileImport);
    }

    #[test]
    fn plan_batch_exact_fit_is_one_import() {
        assert_eq!(plan_batch(32, 64, 2048), BatchPath::OneImport);
    }

    #[test]
    fn plan_batch_degenerate_inputs() {
        assert_eq!(plan_batch(0, 64, 2048), BatchPath::OneImport);
        assert_eq!(plan_batch(4, 0, 2048), BatchPath::OneImport);
    }

    #[test]
    fn plan_convert_full_table() {
        use DstLowering::*;
        use PixelFormat::*;
        // NV* and packed Rgb sources.
        let nv = [Nv12, Nv16, Nv24, Rgb];
        let non_nv = [Rgba, Bgra, Grey, Yuyv];
        // Zero-copy: packed RGB destination two-pass; NV/Rgb→planar two-pass;
        // the rest single-pass.
        for src in nv.iter().chain(&non_nv) {
            assert_eq!(
                plan_convert(*src, false, Rgb, ZeroCopy),
                ConvertPlan::TwoPassPackedRgb,
                "{src:?}->Rgb zero-copy"
            );
        }
        for src in nv {
            assert_eq!(
                plan_convert(src, false, PlanarRgb, ZeroCopy),
                ConvertPlan::TwoPassNvPlanar
            );
            assert_eq!(
                plan_convert(src, false, PlanarRgba, ZeroCopy),
                ConvertPlan::TwoPassNvPlanar
            );
            assert_eq!(
                plan_convert(src, false, Rgba, ZeroCopy),
                ConvertPlan::SinglePass
            );
        }
        for src in non_nv {
            assert_eq!(
                plan_convert(src, false, PlanarRgb, ZeroCopy),
                ConvertPlan::SinglePass
            );
            assert_eq!(
                plan_convert(src, false, Rgba, ZeroCopy),
                ConvertPlan::SinglePass
            );
        }
        // Zero-copy with a host source: planar destinations two-pass; packed
        // destinations unaffected.
        for src in nv.iter().chain(&non_nv) {
            for dst in [PlanarRgb, PlanarRgba] {
                assert_eq!(
                    plan_convert(*src, true, dst, ZeroCopy),
                    ConvertPlan::TwoPassNvPlanar,
                    "host {src:?}->{dst:?} zero-copy"
                );
            }
            assert_eq!(
                plan_convert(*src, true, Rgb, ZeroCopy),
                ConvertPlan::TwoPassPackedRgb
            );
            assert_eq!(
                plan_convert(*src, true, Rgba, ZeroCopy),
                ConvertPlan::SinglePass
            );
        }
        // Texture lowerings: packed RGB single-pass; every source→planar two-pass.
        for lowering in [TextureMem, TexturePbo] {
            for src in nv.iter().chain(&non_nv) {
                for dst in [PlanarRgb, PlanarRgba] {
                    assert_eq!(
                        plan_convert(*src, false, dst, lowering),
                        ConvertPlan::TwoPassNvPlanar,
                        "{src:?}->{dst:?} via {lowering:?}"
                    );
                }
                for dst in [Rgba, Bgra, Rgb] {
                    assert_eq!(
                        plan_convert(*src, false, dst, lowering),
                        ConvertPlan::SinglePass,
                        "{src:?}->{dst:?} via {lowering:?}"
                    );
                }
            }
        }
        // Grey destination: RGB-family sources two-pass on every lowering and
        // residence; YUV and Grey sources single-pass.
        for lowering in [ZeroCopy, TextureMem, TexturePbo] {
            for host in [false, true] {
                for src in [Rgb, Rgba, Bgra] {
                    assert_eq!(
                        plan_convert(src, host, Grey, lowering),
                        ConvertPlan::TwoPassGrey,
                        "{src:?}->Grey via {lowering:?}"
                    );
                }
                for src in [Nv12, Nv16, Nv24, Yuyv, Grey] {
                    assert_eq!(
                        plan_convert(src, host, Grey, lowering),
                        ConvertPlan::SinglePass,
                        "{src:?}->Grey via {lowering:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn lower_dst_full_table() {
        use TensorMemory::*;
        // With zero-copy import: DMA is zero-copy, PBO reads back through PACK,
        // Mem/Shm through the map.
        assert_eq!(lower_dst(true, DmaBuf), DstLowering::ZeroCopy);
        assert_eq!(lower_dst(true, Pbo), DstLowering::TexturePbo);
        assert_eq!(lower_dst(true, Mem), DstLowering::TextureMem);
        assert_eq!(lower_dst(true, Shm), DstLowering::TextureMem);
        // Without import support a DMA destination lowers to the mapped texture path.
        assert_eq!(lower_dst(false, DmaBuf), DstLowering::TextureMem);
        assert_eq!(lower_dst(false, Pbo), DstLowering::TexturePbo);
        assert_eq!(lower_dst(false, Mem), DstLowering::TextureMem);
        assert_eq!(lower_dst(false, Shm), DstLowering::TextureMem);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod import_extent_tests {
    use super::*;

    /// An `ImportMap` with an extent at origin `(0, 0)`.
    fn narrowed(extent: (u32, u32)) -> ImportMap {
        ImportMap {
            extent: Some(extent),
            origin: (0, 0),
        }
    }

    fn roi() -> super::super::RegionOfInterest {
        super::super::RegionOfInterest {
            left: 0.0,
            top: 1.0,
            right: 1.0,
            bottom: 0.0,
        }
    }

    /// 128x96 in a 128x128 texture: u is untouched and v is scaled to the image's share.
    #[test]
    fn a_narrowed_logical_image_scales_by_its_share_of_the_texture() {
        let r = scale_roi_to_import(roi(), (128, 96), narrowed((128, 128)));
        assert_eq!(r.left, 0.0);
        assert_eq!(r.right, 1.0);
        assert_eq!(r.top, 0.75);
        assert_eq!(r.bottom, 0.0);
    }

    #[test]
    fn both_axes_scale_and_an_interior_rectangle_is_unchanged_in_shape() {
        let src = super::super::RegionOfInterest {
            left: 0.25,
            top: 0.75,
            right: 0.5,
            bottom: 0.25,
        };
        let r = scale_roi_to_import(src, (64, 100), narrowed((128, 200)));
        assert_eq!((r.left, r.right), (0.125, 0.25));
        assert_eq!((r.top, r.bottom), (0.375, 0.125));
    }

    #[test]
    fn equal_extents_and_no_extent_leave_the_rectangle_alone() {
        let r = scale_roi_to_import(roi(), (128, 96), narrowed((128, 96)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
        let r = scale_roi_to_import(roi(), (128, 96), ImportMap::WHOLE);
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    /// A logical image larger than the reported extent leaves the rectangle unchanged.
    #[test]
    fn a_logical_image_larger_than_the_extent_is_not_magnified() {
        let r = scale_roi_to_import(roi(), (256, 256), narrowed((128, 128)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    #[test]
    fn a_degenerate_extent_leaves_the_rectangle_alone() {
        let r = scale_roi_to_import(roi(), (128, 96), narrowed((0, 0)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    /// A UV rect's origin scales and its size is trimmed to the far edge.
    #[test]
    fn a_uv_rect_scales_its_origin_and_trims_its_size() {
        let r = scale_uv_rect_to_import([0.0, 0.0, 1.0, 1.0], (128, 96), narrowed((128, 128)));
        assert_eq!([r[0], r[1]], [0.0, 0.0]);
        assert_eq!(r[2], 1.0);
        assert_eq!(r[3], 0.75);
    }

    #[test]
    fn a_uv_rect_keeps_an_interior_crop_and_passes_no_extent_through() {
        let r = scale_uv_rect_to_import([0.25, 0.25, 0.5, 0.5], (96, 96), narrowed((128, 128)));
        assert_eq!([r[0], r[1]], [0.1875, 0.1875]);
        assert_eq!([r[2], r[3]], [0.375, 0.375]);
        let r = scale_uv_rect_to_import([0.25, 0.25, 0.5, 0.5], (96, 96), ImportMap::WHOLE);
        assert_eq!(r, [0.25, 0.25, 0.5, 0.5]);
    }

    /// 128x96 in a 128x128 texture: u is inset half a texel, v stops half a texel short of row 96.
    #[test]
    fn a_narrowed_image_clamps_samples_half_a_texel_inside_its_edge() {
        let r = sample_clamp_rect((128, 96), narrowed((128, 128)));
        assert_eq!(r, [0.5 / 128.0, 0.5 / 128.0, 127.5 / 128.0, 95.5 / 128.0]);
    }

    /// Without an extent the clamp is the half-texel inset of the whole texture on both axes.
    #[test]
    fn no_extent_clamps_to_the_textures_own_edge_texels() {
        let r = sample_clamp_rect((128, 96), ImportMap::WHOLE);
        assert_eq!(r, [0.5 / 128.0, 0.5 / 96.0, 127.5 / 128.0, 95.5 / 96.0]);
        assert_eq!(sample_clamp_rect((128, 96), narrowed((128, 96))), r);
    }

    /// An extent smaller than the logical image is ignored.
    #[test]
    fn an_extent_smaller_than_the_image_is_ignored() {
        assert_eq!(
            sample_clamp_rect((256, 256), narrowed((128, 128))),
            sample_clamp_rect((256, 256), ImportMap::WHOLE)
        );
    }

    #[test]
    fn a_degenerate_logical_size_leaves_the_whole_texture_reachable() {
        assert_eq!(
            sample_clamp_rect((0, 0), narrowed((128, 128))),
            [0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            sample_clamp_rect((0, 96), ImportMap::WHOLE),
            [0.0, 0.5 / 96.0, 1.0, 95.5 / 96.0]
        );
    }

    /// One logical texel in a larger texture clamps min and max to that texel's centre.
    #[test]
    fn a_one_texel_image_clamps_to_its_single_texel_centre() {
        assert_eq!(
            sample_clamp_rect((1, 1), narrowed((128, 128))),
            [0.5 / 128.0; 4]
        );
        assert_eq!(sample_clamp_rect((1, 1), ImportMap::WHOLE), [0.5; 4]);
    }

    /// A 16-wide image at texel 8 of a 24-wide texture maps the ROI onto `[8/24, 24/24]`.
    #[test]
    fn scale_roi_to_import_offsets_by_the_origin() {
        let roi = super::super::RegionOfInterest {
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
        };
        let map = ImportMap {
            extent: Some((24, 16)),
            origin: (8, 0),
        };
        let got = scale_roi_to_import(roi, (16, 16), map);
        assert!((got.left - 8.0 / 24.0).abs() < 1e-6, "left {}", got.left);
        assert!((got.right - 1.0).abs() < 1e-6, "right {}", got.right);
        assert!((got.top - 0.0).abs() < 1e-6, "top {}", got.top);
        assert!((got.bottom - 1.0).abs() < 1e-6, "bottom {}", got.bottom);
    }

    /// A half-width ROI on a shifted import has its midpoint at `(8 + 8) / 24`.
    #[test]
    fn scale_roi_to_import_composes_origin_and_scale() {
        let roi = super::super::RegionOfInterest {
            left: 0.0,
            top: 0.25,
            right: 0.5,
            bottom: 0.75,
        };
        let map = ImportMap {
            extent: Some((24, 16)),
            origin: (8, 0),
        };
        let got = scale_roi_to_import(roi, (16, 16), map);
        assert!((got.left - 8.0 / 24.0).abs() < 1e-6, "left {}", got.left);
        assert!(
            (got.right - 16.0 / 24.0).abs() < 1e-6,
            "right {}",
            got.right
        );
        assert!((got.top - 0.25).abs() < 1e-6, "top {}", got.top);
        assert!((got.bottom - 0.75).abs() < 1e-6, "bottom {}", got.bottom);
    }

    /// `ImportMap::WHOLE` is the identity on all three helpers.
    #[test]
    fn whole_import_map_is_the_identity() {
        let roi = super::super::RegionOfInterest {
            left: 0.1,
            top: 0.2,
            right: 0.8,
            bottom: 0.9,
        };
        let got = scale_roi_to_import(roi, (64, 64), ImportMap::WHOLE);
        assert_eq!(
            (got.left, got.top, got.right, got.bottom),
            (0.1, 0.2, 0.8, 0.9)
        );
        assert_eq!(
            scale_uv_rect_to_import([0.1, 0.2, 0.3, 0.4], (64, 64), ImportMap::WHOLE),
            [0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(ImportMap::default(), ImportMap::WHOLE);
        let [u0, v0, u1, v1] = sample_clamp_rect((64, 64), ImportMap::WHOLE);
        assert!((u0 - 0.5 / 64.0).abs() < 1e-9 && (v0 - 0.5 / 64.0).abs() < 1e-9);
        assert!((u1 - 63.5 / 64.0).abs() < 1e-9 && (v1 - 63.5 / 64.0).abs() < 1e-9);
    }

    /// A source crop narrows the clamp to its own texels, half a texel inside its edges.
    #[test]
    fn sample_clamp_rect_within_brackets_a_crop() {
        let crop = [16.0 / 64.0, 8.0 / 32.0, 48.0 / 64.0, 24.0 / 32.0];
        let [u0, v0, u1, v1] = sample_clamp_rect_within((64, 32), ImportMap::WHOLE, crop);
        assert!((u0 - 16.5 / 64.0).abs() < 1e-6 && (u1 - 47.5 / 64.0).abs() < 1e-6);
        assert!((v0 - 8.5 / 32.0).abs() < 1e-6 && (v1 - 23.5 / 32.0).abs() < 1e-6);
        let map = ImportMap {
            extent: Some((80, 32)),
            origin: (8, 0),
        };
        let [u0, _, u1, _] = sample_clamp_rect_within((64, 32), map, crop);
        assert!((u0 - 24.5 / 80.0).abs() < 1e-6 && (u1 - 55.5 / 80.0).abs() < 1e-6);
        assert_eq!(
            sample_clamp_rect_within((64, 32), map, [0.0, 0.0, 1.0, 1.0]),
            sample_clamp_rect((64, 32), map)
        );
    }

    /// On a shifted texture the clamp rectangle brackets the logical image's texels.
    #[test]
    fn sample_clamp_rect_brackets_a_shifted_image() {
        let map = ImportMap {
            extent: Some((24, 16)),
            origin: (8, 0),
        };
        let [u0, v0, u1, v1] = sample_clamp_rect((16, 16), map);
        assert!((u0 - 8.5 / 24.0).abs() < 1e-6, "u0 {u0}");
        assert!((u1 - 23.5 / 24.0).abs() < 1e-6, "u1 {u1}");
        assert!((v0 - 0.5 / 16.0).abs() < 1e-6, "v0 {v0}");
        assert!((v1 - 15.5 / 16.0).abs() < 1e-6, "v1 {v1}");
    }

    /// A shifted image that would run past the texture maps as the identity.
    #[test]
    fn a_shifted_image_that_overruns_the_extent_maps_as_the_identity() {
        let roi = super::super::RegionOfInterest {
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
        };
        let map = ImportMap {
            extent: Some((20, 16)),
            origin: (8, 0),
        };
        let got = scale_roi_to_import(roi, (16, 16), map);
        assert_eq!((got.left, got.right), (0.0, 1.0));
        let [u0, _, u1, _] = sample_clamp_rect((16, 16), map);
        assert!((u0 - 0.5 / 16.0).abs() < 1e-6, "u0 {u0}");
        assert!((u1 - 15.5 / 16.0).abs() < 1e-6, "u1 {u1}");
    }

    /// An origin with no extent is dropped by every helper: the map is the identity.
    #[test]
    fn an_origin_without_an_extent_is_dropped_by_every_helper() {
        let map = ImportMap {
            extent: None,
            origin: (8, 0),
        };
        let r = roi();
        let got = scale_roi_to_import(r, (16, 16), map);
        assert_eq!(
            (got.left, got.top, got.right, got.bottom),
            (r.left, r.top, r.right, r.bottom)
        );
        assert_eq!(
            scale_uv_rect_to_import([0.1, 0.2, 0.3, 0.4], (16, 16), map),
            [0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(
            sample_clamp_rect((16, 16), map),
            sample_clamp_rect((16, 16), ImportMap::WHOLE)
        );
    }

    /// A UV rect on a shifted import has its origin shifted and its size trimmed.
    #[test]
    fn scale_uv_rect_to_import_offsets_by_the_origin() {
        let map = ImportMap {
            extent: Some((24, 16)),
            origin: (8, 0),
        };
        let got = scale_uv_rect_to_import([0.0, 0.0, 1.0, 1.0], (16, 16), map);
        assert!((got[0] - 8.0 / 24.0).abs() < 1e-6, "x {}", got[0]);
        assert!((got[2] - 16.0 / 24.0).abs() < 1e-6, "w {}", got[2]);
        assert!((got[1] - 0.0).abs() < 1e-6, "y {}", got[1]);
        assert!((got[3] - 1.0).abs() < 1e-6, "h {}", got[3]);
    }
}
