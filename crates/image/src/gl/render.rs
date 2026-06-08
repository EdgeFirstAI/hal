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

// `Viewport`/`source_uv`/`plan_batch`/`BatchPath` stage the chunking follow-up
// (split a batch that exceeds `GL_MAX_*` into per-chunk imports) and the
// source-UV/bottom-up-surface paths; the live top-down DMA batch path computes
// its band viewport inline (see the orientation note above). Until those land,
// only the unit tests exercise these — drop the allow when they are wired in.
#![allow(dead_code)]

use crate::Region;

/// A GL viewport / scissor rectangle in **pixels**, bottom-left origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Viewport {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) w: i32,
    pub(super) h: i32,
}

/// Lower a top-left `region` of an image `parent_h` rows tall to a bottom-left
/// GL viewport. The horizontal axis is unchanged; the vertical axis is flipped
/// so the region's *top* edge maps to the correct GL row.
///
/// This is the single source of the y-origin convention for the converged
/// renderer — a destination tile, a letterbox inner box, and a whole-image
/// render all lower through here.
pub(super) fn region_to_viewport(region: Region, parent_h: usize) -> Viewport {
    // Bottom-left origin: the GL y of the region's top-left corner is the number
    // of rows *below* the region — `parent_h - (y + h)`. Saturating so a region
    // flush against the bottom edge (or a degenerate over-tall region) yields a
    // non-negative origin rather than wrapping.
    let y_flipped = parent_h.saturating_sub(region.y + region.height);
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
}
