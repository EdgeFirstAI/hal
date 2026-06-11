// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Portable GL render lowering — the single home for the pure geometry math the
//! converged tile/batch renderer relies on. **No platform types** (no gbm,
//! IOSurface, EGL, or `gls` symbols) appear here so the same lowering serves the
//! Linux DMA-BUF and macOS IOSurface backends behind one seam.
//!
//! Three responsibilities live here, each previously open-coded at several call
//! sites in `processor/mod.rs`:
//!
//! 1. **`region_to_viewport`** — the bottom-left y-origin flip for a *bottom-up*
//!    render target. GL's window origin is bottom-left; HAL regions are
//!    top-left. A tile at pixel `(x, y)` of size `(w, h)` in a `parent_h`-row
//!    target maps to viewport `y' = parent_h − (y + h)`. NOTE: the live Linux
//!    DMA-BUF destination batch path is **top-down** (GL row 0 == memory row 0,
//!    verified on-target — the renderer's texcoords already flip the image
//!    upright), so it uses `region.y` directly and does NOT call this. Kept for a
//!    future bottom-up surface (e.g. the macOS pbuffer batch path).
//!
//! 2. **`source_uv`** — the source sampling rectangle in normalised `[0,1]` UV,
//!    derived from a `Region`. Mirrors the `src_rect_uv` half of
//!    [`crate::gl::core::float_crop_uniforms`].
//!
//! 3. **`plan_batch`** — the chunk planner. A batch of `n` destination tiles
//!    renders into one reused import when every tile fits the GPU's
//!    `GL_MAX_*` limits; otherwise it splits into chunks (one import per chunk)
//!    or, degenerately, one import per tile. The tile region is *render state*
//!    (`glViewport`), never part of the EGL cache key — that is what makes
//!    "N tiles → 1 import" hold.

use crate::Region;
use edgefirst_tensor::{PixelFormat, PixelLayout, TensorMemory};

/// How a convert's destination is realized on the GPU — the pure half of the
/// destination lowering (`bind_dst` in `processor/mod.rs` performs the GL
/// work). One lowering per *destination memory class*, never per platform:
/// platform differences surface as the `zero_copy_import` capability bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DstLowering {
    /// The destination buffer itself becomes the FBO colour attachment
    /// (EGLImage renderbuffer/texture today, IOSurface pbuffer after the
    /// platform seam). The render writes the buffer directly — no readback.
    ZeroCopy,
    /// Offscreen texture render target, read back into the mapped tensor.
    TextureMem,
    /// Offscreen texture render target, read back into the destination PBO's
    /// PACK binding (the tensor must never be mapped on the GL thread).
    TexturePbo,
}

/// Classify the destination lowering from the platform's zero-copy import
/// capability and the destination's memory class. A DMA destination without
/// import support (e.g. dma-heap present but `EGL_EXT_image_dma_buf_import`
/// missing) degrades to the mapped texture path rather than failing.
pub(super) fn lower_dst(zero_copy_import: bool, dst_mem: TensorMemory) -> DstLowering {
    match dst_mem {
        TensorMemory::Dma if zero_copy_import => DstLowering::ZeroCopy,
        TensorMemory::Pbo => DstLowering::TexturePbo,
        _ => DstLowering::TextureMem,
    }
}

/// How a convert renders — the pure plan half of the engine
/// (`convert_via_engine` in `processor/mod.rs` executes it). Exactly one
/// plan per (source format, destination format, destination lowering)
/// triple; the two-pass plans exist only for zero-copy destinations, whose
/// buffers GL cannot write in the requested layout in one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConvertPlan {
    /// One render pass into the bound destination target (packed via the
    /// texture shaders, planar via the planar shader), then the lowering's
    /// readback (none for zero-copy).
    SinglePass,
    /// Zero-copy packed-RGB: pass 1 renders RGBA into an intermediate
    /// texture, pass 2 packs it into the destination reinterpreted as
    /// RGBA8 at `W*3/4 × H` (GL has no 3-byte render format).
    TwoPassPackedRgb,
    /// Zero-copy NV*→planar: pass 1 converts NV*→RGBA through the full
    /// `select_nv_path` machinery (colorimetry-exact ShaderR8 + Vivante
    /// carve-out), pass 2 deinterleaves RGBA into the planar destination.
    /// Also the Vivante GC7000UL single-pass GPU-hang workaround
    /// (EDGEAI-1180).
    TwoPassNvPlanar,
}

/// Decide the render plan. Pure (src format, dst format, dst lowering) →
/// plan; capability differences arrive via the lowering, never as platform
/// branches here.
pub(super) fn plan_convert(
    src_fmt: PixelFormat,
    dst_fmt: PixelFormat,
    lowering: DstLowering,
) -> ConvertPlan {
    if lowering != DstLowering::ZeroCopy {
        // Texture destinations always render once and read back; packed RGB
        // and planar layouts are handled by the readback format / planar
        // shader, not by reinterpreting the destination buffer.
        return ConvertPlan::SinglePass;
    }
    if dst_fmt == PixelFormat::Rgb {
        return ConvertPlan::TwoPassPackedRgb;
    }
    if dst_fmt.layout() == PixelLayout::Planar
        && matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        )
    {
        return ConvertPlan::TwoPassNvPlanar;
    }
    ConvertPlan::SinglePass
}

/// A GL viewport / scissor rectangle in **pixels**, bottom-left origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Viewport {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) w: i32,
    pub(super) h: i32,
}

/// Lower a destination tile to its GL band rectangle on the live Linux
/// DMA-BUF path, which is **top-down** (GL framebuffer row 0 == memory row 0,
/// verified on-target): the band's GL `y` is simply `region.y`, with no
/// bottom-left flip — the renderer's texcoord flip keeps the image upright.
/// The single home for the orientation convention: BOTH the band `glViewport`
/// (`bind_dst` setup) and the matching `glScissor` (letterbox-clear
/// confinement in `convert_to`) lower through here so they can never
/// disagree. [`region_to_viewport`] is the bottom-up twin for a future
/// bottom-left-origin surface (e.g. the macOS pbuffer batch path).
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
/// GL viewport. The horizontal axis is unchanged; the vertical axis is flipped
/// so the region's *top* edge maps to the correct GL row.
///
/// Staged for the first bottom-left-origin render target (macOS pbuffer batch
/// path, PR-A); the live Linux DMA-BUF path is top-down and uses
/// [`region_to_viewport_top_down`]. Until then only the unit tests call this.
#[allow(dead_code)]
pub(super) fn region_to_viewport(region: Region, parent_h: usize) -> Viewport {
    // Bottom-left origin: the GL y of the region's top-left corner is the number
    // of rows *below* the region — `parent_h - (y + h)`. Saturating so a region
    // flush against the bottom edge (or a degenerate over-tall region) yields a
    // non-negative origin rather than wrapping.
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
/// `[0, 0, 1, 1]`. Matches the `src_rect_uv` produced by
/// [`crate::gl::core::float_crop_uniforms`] so the converged renderer and the
/// float path agree on source addressing.
///
/// Staged for the source-view sampling path (the u8 renderer still computes
/// its `RegionOfInterest` corners inline); until wired, only the unit tests
/// call this.
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
/// `OneImport` is the goal — the whole destination is imported once and each
/// tile is a `glViewport` into it. `Chunked(tiles_per_chunk)` is the fallback
/// when the full batched render target would exceed a `GL_MAX_*` limit: the
/// batch is split into chunks of `tiles_per_chunk`, one import per chunk.
/// `PerTileImport` is the degenerate floor (a single tile already too large to
/// batch) — equivalent to the legacy per-call path.
/// Staged for the `GL_MAX_*` chunking follow-up (split an over-limit batch
/// into per-chunk imports); until wired into `convert_batch`, only the unit
/// tests exercise this and [`plan_batch`].
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn region(x: usize, y: usize, w: usize, h: usize) -> Region {
        Region::new(x, y, w, h)
    }

    #[test]
    fn viewport_top_down_is_identity() {
        // The live DMA path is top-down: GL y == memory row y, no flip. Three
        // stacked 16-row tiles keep their memory order in GL coordinates.
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
        // A region covering the whole image maps to the full viewport at (0,0):
        // y_flip = H - (0 + H) = 0.
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
        // Top 16-row band of a 48-tall image: GL y = 48 - (0 + 16) = 32
        // (the top of a top-left image is the HIGH end of bottom-left GL space).
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
        // Bottom 16-row band (y=32): GL y = 48 - (32 + 16) = 0.
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
        // Three stacked 16-row tiles of a 48-tall batched target partition the
        // viewport exactly: GL y bands are [32,48), [16,32), [0,16).
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
        // Right half of a 640×480 source.
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
        // 4 tiles × 64 rows = 256 ≤ 2048 → one import.
        assert_eq!(plan_batch(4, 64, 2048), BatchPath::OneImport);
    }

    #[test]
    fn plan_batch_chunks_when_over_limit() {
        // 40 tiles × 64 rows = 2560 > 2048; floor(2048/64)=32 per chunk.
        assert_eq!(plan_batch(40, 64, 2048), BatchPath::Chunked(32));
    }

    #[test]
    fn plan_batch_single_oversize_tile_is_per_tile() {
        // One planar tile of 3·1024 = 3072 rows > 2048 → degenerate per-tile.
        assert_eq!(plan_batch(8, 3072, 2048), BatchPath::PerTileImport);
    }

    #[test]
    fn plan_batch_exact_fit_is_one_import() {
        // 32 tiles × 64 = 2048 == limit → still one import (≤, not <).
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
        let nv = [Nv12, Nv16, Nv24];
        let non_nv = [Rgba, Bgra, Grey, Yuyv];
        // Zero-copy: packed RGB always two-pass; NV→planar two-pass; the rest
        // single-pass (incl. non-NV → planar, which the planar shader handles
        // in one pass).
        for src in nv.iter().chain(&non_nv) {
            assert_eq!(
                plan_convert(*src, Rgb, ZeroCopy),
                ConvertPlan::TwoPassPackedRgb,
                "{src:?}->Rgb zero-copy"
            );
        }
        for src in nv {
            assert_eq!(
                plan_convert(src, PlanarRgb, ZeroCopy),
                ConvertPlan::TwoPassNvPlanar
            );
            assert_eq!(
                plan_convert(src, PlanarRgba, ZeroCopy),
                ConvertPlan::TwoPassNvPlanar
            );
            assert_eq!(plan_convert(src, Rgba, ZeroCopy), ConvertPlan::SinglePass);
        }
        for src in non_nv {
            assert_eq!(
                plan_convert(src, PlanarRgb, ZeroCopy),
                ConvertPlan::SinglePass
            );
            assert_eq!(plan_convert(src, Rgba, ZeroCopy), ConvertPlan::SinglePass);
        }
        // Texture lowerings: ALWAYS single-pass, for every format pair — the
        // two-pass plans only exist to write zero-copy buffers in-place.
        for lowering in [TextureMem, TexturePbo] {
            for src in nv.iter().chain(&non_nv) {
                for dst in [Rgba, Bgra, Rgb, PlanarRgb, Grey] {
                    assert_eq!(
                        plan_convert(*src, dst, lowering),
                        ConvertPlan::SinglePass,
                        "{src:?}->{dst:?} via {lowering:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn lower_dst_full_table() {
        use TensorMemory::*;
        // With zero-copy import: only a DMA destination is zero-copy; PBO
        // keeps its PACK readback; Mem/Shm read back through the map.
        assert_eq!(lower_dst(true, Dma), DstLowering::ZeroCopy);
        assert_eq!(lower_dst(true, Pbo), DstLowering::TexturePbo);
        assert_eq!(lower_dst(true, Mem), DstLowering::TextureMem);
        assert_eq!(lower_dst(true, Shm), DstLowering::TextureMem);
        // Without import support a DMA destination degrades to the mapped
        // texture path (dma-heap without EGL dma_buf_import) — never an error.
        assert_eq!(lower_dst(false, Dma), DstLowering::TextureMem);
        assert_eq!(lower_dst(false, Pbo), DstLowering::TexturePbo);
        assert_eq!(lower_dst(false, Mem), DstLowering::TextureMem);
        assert_eq!(lower_dst(false, Shm), DstLowering::TextureMem);
    }
}
