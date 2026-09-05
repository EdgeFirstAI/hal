// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Portable GL render lowering — the single home for the pure geometry math the
//! converged tile/batch renderer relies on. **No platform types** (no gbm,
//! IOSurface, EGL, or `edgefirst_gl` symbols) appear here so the same lowering serves the
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
        TensorMemory::DmaBuf if zero_copy_import => DstLowering::ZeroCopy,
        TensorMemory::Pbo => DstLowering::TexturePbo,
        _ => DstLowering::TextureMem,
    }
}

/// How a convert renders — the pure plan half of the engine
/// (`convert_via_engine` in `processor/mod.rs` executes it). Exactly one
/// plan per (source format, destination format, destination lowering)
/// triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConvertPlan {
    /// One render pass into the bound destination target (packed via the
    /// texture shaders, planar via the planar shader), then the lowering's
    /// readback (none for zero-copy).
    SinglePass,
    /// Zero-copy packed-RGB only: pass 1 renders RGBA into an intermediate
    /// texture, pass 2 packs it into the destination reinterpreted as
    /// RGBA8 at `W*3/4 × H` (GL has no 3-byte render format). Texture
    /// lowerings render genuine RGB in one pass instead — see
    /// `setup_renderbuffer_non_dma`/`setup_renderbuffer_from_pbo`.
    TwoPassPackedRgb,
    /// Planar destination, driven by an ordinary packed convert: pass 1
    /// renders the source into an intermediate RGBA texture via the full
    /// `convert_to` machinery (any source format — `select_nv_path`'s
    /// colorimetry-exact ShaderR8 + Vivante carve-out for NV*, the plain
    /// upload/import path for everything else), pass 2 deinterleaves RGBA
    /// into the planar destination (`bind_dst` classifies pass 2's target
    /// the same way `SinglePass` does, so a texture lowering reads back
    /// same as `SinglePass` would). Selected on EVERY lowering for NV*
    /// sources, and additionally on texture lowerings for every source:
    /// the single-pass planar shader (`draw_camera_texture_to_rgb_planar`)
    /// exists only where `Platform::EXTERNAL_OES` is true (false on
    /// ANGLE/Android — `texture_program_planar` is `None` there,
    /// regardless of source format), and even where it exists its source
    /// import is DMA-only, with no upload path for a heap/Mem source.
    /// Zero-copy keeps the non-NV single-pass route (proven, EXTERNAL_OES
    /// path with a DMA source). Also the Vivante GC7000UL single-pass
    /// GPU-hang workaround for NV* (EDGEAI-1180).
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
    // Packed RGB only needs the two-pass W*3/4 reinterpretation on a
    // zero-copy destination (no native 3-byte GL render format); texture
    // destinations already render genuine RGB via `setup_renderbuffer_non_dma`
    // / `setup_renderbuffer_from_pbo`, so they stay single-pass.
    if lowering == DstLowering::ZeroCopy && dst_fmt == PixelFormat::Rgb {
        return ConvertPlan::TwoPassPackedRgb;
    }
    // A texture-lowered (Mem/Pbo) planar destination has NO single-pass
    // route for any source format: `texture_program_planar` (the
    // EXTERNAL_OES planar shader `draw_camera_texture_to_rgb_planar`
    // needs) is built only where `Platform::EXTERNAL_OES` is true — never
    // on ANGLE or Android — and even on Linux (where it IS built) the
    // source import (`get_or_create_egl_image`) is unconditionally
    // DMA-only, so a heap/Mem source fails there too. The two-pass plan's
    // pass 1 is the ordinary `convert_to` packed path, which already
    // handles every source format (including a Mem source) on every
    // platform.
    if dst_fmt.layout() == PixelLayout::Planar && lowering != DstLowering::ZeroCopy {
        return ConvertPlan::TwoPassNvPlanar;
    }
    // Zero-copy NV*→planar still needs the two-pass ShaderR8 route: the
    // single-pass planar shader has no multi-plane EGLImage/IOSurface
    // binding for semi-planar NV even when EXTERNAL_OES is available.
    // Zero-copy non-NV→planar stays single-pass (proven EXTERNAL_OES +
    // DMA-source route).
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

/// How one axis of a logical image maps onto the texture an import covers:
/// `(scale, limit)` — the factor that places the logical image over its share
/// of the texture, and the normalized coordinate of its far edge. An axis that
/// is not narrowed (and a degenerate or contradictory extent) maps as the
/// identity.
///
/// The far edge is not pulled half a physical texel inside here. These
/// coordinates are the endpoints of the quad's texcoord attribute, so moving
/// one rescales the whole affine dst-to-source mapping rather than excluding
/// an edge: measured on a narrowed 128x96 source, the inset left the identity
/// convert differing from a fresh buffer in 34380 of 49152 bytes.
///
/// The inset belongs on the sample instead, and that is where it now is: every
/// shader that samples through this mapping clamps each sample coordinate to
/// [`sample_clamp_rect`], so a `LINEAR` kernel centred within half a texel of
/// the logical edge cannot blend the stale texel beyond it. Measured on the
/// RTX 3070 and on WARP, the last row and column of an upscaled narrowed
/// source then match a fresh buffer of the same geometry byte for byte
/// (492-762 bytes differed by up to 48 without the clamp).
fn axis_map(logical: usize, extent: u32) -> (f32, f32) {
    if extent == 0 || logical == 0 || logical as u64 >= extent as u64 {
        return (1.0, 1.0);
    }
    let scale = logical as f32 / extent as f32;
    (scale, scale)
}

/// Map a source rectangle from the tensor's logical image onto the larger
/// texture its import covers.
///
/// `roi` is normalized over the logical image; `extent` is
/// `GlPlatform::import_extent`. The logical image occupies the texture's own
/// texels from its origin, so mapping one onto the other is one multiply per
/// axis — the half-texel inset the crop math already applied scales with it —
/// held inside the logical edge by [`axis_map`]. `None` and an extent no
/// larger than the logical image leave `roi` untouched.
pub(super) fn scale_roi_to_import(
    roi: super::RegionOfInterest,
    logical: (usize, usize),
    extent: Option<(u32, u32)>,
) -> super::RegionOfInterest {
    let Some((pw, ph)) = extent else {
        return roi;
    };
    let (sx, lx) = axis_map(logical.0, pw);
    let (sy, ly) = axis_map(logical.1, ph);
    super::RegionOfInterest {
        left: (roi.left * sx).min(lx),
        top: (roi.top * sy).min(ly),
        right: (roi.right * sx).min(lx),
        bottom: (roi.bottom * sy).min(ly),
    }
}

/// [`scale_roi_to_import`] for the float paths' `src_rect_uv`, which is an
/// origin and a size rather than two corners: the origin scales, and the size
/// is trimmed to keep the far edge inside the same limit.
pub(super) fn scale_uv_rect_to_import(
    rect: [f32; 4],
    logical: (usize, usize),
    extent: Option<(u32, u32)>,
) -> [f32; 4] {
    let Some((pw, ph)) = extent else {
        return rect;
    };
    let (sx, lx) = axis_map(logical.0, pw);
    let (sy, ly) = axis_map(logical.1, ph);
    let left = (rect[0] * sx).min(lx);
    let top = (rect[1] * sy).min(ly);
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
/// A `LINEAR` sample at those bounds is centred on the logical image's edge
/// texel, so the kernel cannot reach the texel beyond it. On a texture that
/// is exactly the logical image (`extent` is `None`, or no larger than
/// `logical`) that texel does not exist and `CLAMP_TO_EDGE` already returns
/// the edge value, so the clamp changes nothing there. On an import that
/// covers more of the texture than the logical image the texel beyond the
/// edge holds the pool buffer's previous content, which an upscale would
/// otherwise blend into the last row or column.
///
/// A degenerate logical size yields the whole texture, `[0, 0, 1, 1]`.
pub(super) fn sample_clamp_rect(logical: (usize, usize), extent: Option<(u32, u32)>) -> [f32; 4] {
    fn axis(logical: usize, extent: Option<u32>) -> (f32, f32) {
        if logical == 0 {
            return (0.0, 1.0);
        }
        let texture = match extent {
            Some(e) if e as u64 > logical as u64 => e as f32,
            _ => logical as f32,
        };
        (0.5 / texture, (logical as f32 - 0.5) / texture)
    }
    let (u_min, u_max) = axis(logical.0, extent.map(|e| e.0));
    let (v_min, v_max) = axis(logical.1, extent.map(|e| e.1));
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
        // Texture lowerings: packed-RGB single-pass (the texture destination
        // renders genuine RGB); EVERY source→planar is two-pass regardless
        // of format — the single-pass planar shader
        // (`draw_camera_texture_to_rgb_planar`) needs
        // `Platform::EXTERNAL_OES` (false on ANGLE/Android) AND a DMA
        // source, neither of which a texture-lowered destination can rely
        // on, so non-NV sources route through the same two-pass plan as NV*
        // instead of single-pass.
        for lowering in [TextureMem, TexturePbo] {
            for src in nv.iter().chain(&non_nv) {
                for dst in [PlanarRgb, PlanarRgba] {
                    assert_eq!(
                        plan_convert(*src, dst, lowering),
                        ConvertPlan::TwoPassNvPlanar,
                        "{src:?}->{dst:?} via {lowering:?}"
                    );
                }
                for dst in [Rgba, Bgra, Rgb, Grey] {
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
        assert_eq!(lower_dst(true, DmaBuf), DstLowering::ZeroCopy);
        assert_eq!(lower_dst(true, Pbo), DstLowering::TexturePbo);
        assert_eq!(lower_dst(true, Mem), DstLowering::TextureMem);
        assert_eq!(lower_dst(true, Shm), DstLowering::TextureMem);
        // Without import support a DMA destination degrades to the mapped
        // texture path (dma-heap without EGL dma_buf_import) — never an error.
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

    fn roi() -> super::super::RegionOfInterest {
        super::super::RegionOfInterest {
            left: 0.0,
            top: 1.0,
            right: 1.0,
            bottom: 0.0,
        }
    }

    /// 128x96 written into a 128x128 pool texture: u is untouched (that axis
    /// is not narrowed), v is scaled to the logical image's share.
    #[test]
    fn a_narrowed_logical_image_scales_by_its_share_of_the_texture() {
        let r = scale_roi_to_import(roi(), (128, 96), Some((128, 128)));
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
        let r = scale_roi_to_import(src, (64, 100), Some((128, 200)));
        assert_eq!((r.left, r.right), (0.125, 0.25));
        assert_eq!((r.top, r.bottom), (0.375, 0.125));
    }

    #[test]
    fn equal_extents_and_no_extent_leave_the_rectangle_alone() {
        let r = scale_roi_to_import(roi(), (128, 96), Some((128, 96)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
        let r = scale_roi_to_import(roi(), (128, 96), None);
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    /// A logical image larger than the reported extent is a contradiction; the
    /// rectangle is left alone rather than magnified past the texture.
    #[test]
    fn a_logical_image_larger_than_the_extent_is_not_magnified() {
        let r = scale_roi_to_import(roi(), (256, 256), Some((128, 128)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    #[test]
    fn a_degenerate_extent_leaves_the_rectangle_alone() {
        let r = scale_roi_to_import(roi(), (128, 96), Some((0, 0)));
        assert_eq!((r.left, r.top, r.right, r.bottom), (0.0, 1.0, 1.0, 0.0));
    }

    /// The float paths' origin+size rectangle: the origin scales and the size
    /// is trimmed so the far edge lands where a corner would.
    #[test]
    fn a_uv_rect_scales_its_origin_and_trims_its_size() {
        let r = scale_uv_rect_to_import([0.0, 0.0, 1.0, 1.0], (128, 96), Some((128, 128)));
        assert_eq!([r[0], r[1]], [0.0, 0.0]);
        assert_eq!(r[2], 1.0);
        assert_eq!(r[3], 0.75);
    }

    #[test]
    fn a_uv_rect_keeps_an_interior_crop_and_passes_no_extent_through() {
        // Half the width starting a quarter in, on an axis narrowed to 3/4:
        // both scale, and the far edge (0.25 + 0.5) * 0.75 stays interior.
        let r = scale_uv_rect_to_import([0.25, 0.25, 0.5, 0.5], (96, 96), Some((128, 128)));
        assert_eq!([r[0], r[1]], [0.1875, 0.1875]);
        assert_eq!([r[2], r[3]], [0.375, 0.375]);
        let r = scale_uv_rect_to_import([0.25, 0.25, 0.5, 0.5], (96, 96), None);
        assert_eq!(r, [0.25, 0.25, 0.5, 0.5]);
    }

    /// 128x96 in a 128x128 texture: u spans the whole texture inset by half
    /// a texel, v stops half a texel short of row 96.
    #[test]
    fn a_narrowed_image_clamps_samples_half_a_texel_inside_its_edge() {
        let r = sample_clamp_rect((128, 96), Some((128, 128)));
        assert_eq!(r, [0.5 / 128.0, 0.5 / 128.0, 127.5 / 128.0, 95.5 / 128.0]);
    }

    /// Without an extent the texture is the logical image, and the clamp is
    /// the half-texel inset of the whole texture on both axes.
    #[test]
    fn no_extent_clamps_to_the_textures_own_edge_texels() {
        let r = sample_clamp_rect((128, 96), None);
        assert_eq!(r, [0.5 / 128.0, 0.5 / 96.0, 127.5 / 128.0, 95.5 / 96.0]);
        assert_eq!(sample_clamp_rect((128, 96), Some((128, 96))), r);
    }

    /// An extent smaller than the logical image is a contradiction; the
    /// texture is taken to be the logical image, as in `scale_roi_to_import`.
    #[test]
    fn an_extent_smaller_than_the_image_is_ignored() {
        assert_eq!(
            sample_clamp_rect((256, 256), Some((128, 128))),
            sample_clamp_rect((256, 256), None)
        );
    }

    #[test]
    fn a_degenerate_logical_size_leaves_the_whole_texture_reachable() {
        assert_eq!(
            sample_clamp_rect((0, 0), Some((128, 128))),
            [0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            sample_clamp_rect((0, 96), None),
            [0.0, 0.5 / 96.0, 1.0, 95.5 / 96.0]
        );
    }

    /// One logical texel in a larger texture: min and max meet at that
    /// texel's centre. The only input where the bound could invert.
    #[test]
    fn a_one_texel_image_clamps_to_its_single_texel_centre() {
        assert_eq!(
            sample_clamp_rect((1, 1), Some((128, 128))),
            [0.5 / 128.0; 4]
        );
        assert_eq!(sample_clamp_rect((1, 1), None), [0.5; 4]);
    }
}
