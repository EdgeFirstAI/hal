// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SAHI-style input tiling: cover a high-resolution frame with a uniform
//! overlapping grid of model-input-sized tiles and preprocess them zero-copy.
//!
//! The grid uses the **EvenDist** spacing (ported from the adis-uav-model
//! `sahi()` reference): `overlap_ratio` is treated as a *minimum*; the actual
//! overlap is redistributed evenly so every tile is full-size and the last tile
//! lands exactly on the far edge. This is the single spacing method — there is
//! no flush-to-edge/clip configurability.
//!
//! The per-tile detections produced downstream are lifted to full-frame
//! coordinates and merged by [`edgefirst_decoder::tiling`]; the shared
//! [`TilePlacement`] is produced here and consumed there.

use crate::{Crop, Fit, ImageProcessor, ImageProcessorTrait, Result};
use crate::{Error, Flip, Rotation};
use edgefirst_tensor::{DType, PixelFormat, Region, TensorDyn, TensorMemory};

pub use edgefirst_decoder::tiling::TilePlacement;

/// One tile's native-frame crop rectangle and its grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSpec {
    /// Native crop in full-frame pixels: `(x=ox, y=oy, width=cw, height=ch)`.
    pub source: Region,
    /// Row-major flat index, `0..count`.
    pub index: usize,
    /// Grid row.
    pub row: usize,
    /// Grid column.
    pub col: usize,
}

/// Static tiling configuration for one model. Independent of frame size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilingConfig {
    /// Tile width = model input width.
    pub tile_w: usize,
    /// Tile height = model input height.
    pub tile_h: usize,
    /// Minimum overlap ratio between adjacent tiles, `0.0 <= r < 1.0`
    /// (default 0.2). The realized overlap is redistributed evenly and is
    /// always `>=` this.
    pub overlap_ratio: f32,
    /// Letterbox pad colour (RGBA), used only when `fit == Fit::Letterbox`.
    pub pad: [u8; 4],
    /// How a tile crop is fit into the model input. Defaults to
    /// [`Fit::Stretch`], which is identity for the full square tiles the grid
    /// produces (`crop == model input`).
    pub fit: Fit,
}

impl TilingConfig {
    /// Tile size = model input; deploy defaults (overlap 0.2, stretch fit,
    /// `[114,114,114,255]` pad).
    pub fn new(tile_w: usize, tile_h: usize) -> Self {
        Self {
            tile_w,
            tile_h,
            overlap_ratio: 0.2,
            pad: [114, 114, 114, 255],
            fit: Fit::Stretch,
        }
    }

    /// Set the minimum overlap ratio (builder).
    pub fn with_overlap(mut self, overlap_ratio: f32) -> Self {
        self.overlap_ratio = overlap_ratio;
        self
    }

    /// Set the fit mode (builder). For `Fit::Letterbox` the configured `pad` is
    /// used.
    pub fn with_fit(mut self, fit: Fit) -> Self {
        self.fit = fit;
        self
    }

    /// Validate the config: `overlap_ratio` in `[0.0, 1.0)` and non-zero tile
    /// size. Called by every tiling method, and exposed so callers (e.g. the
    /// language bindings) can reject bad input before generating a grid.
    ///
    /// # Errors
    /// Returns [`Error::CropInvalid`] if the overlap is out of range or a tile
    /// dimension is zero.
    pub fn validate(&self) -> Result<()> {
        if !(self.overlap_ratio >= 0.0 && self.overlap_ratio < 1.0) {
            return Err(Error::CropInvalid(format!(
                "tiling overlap_ratio must be in [0.0, 1.0), got {}",
                self.overlap_ratio
            )));
        }
        if self.tile_w == 0 || self.tile_h == 0 {
            return Err(Error::CropInvalid(
                "tiling tile size must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// EvenDist 1-D tile origins along one axis. Mirrors adis-uav-model `sahi()`:
/// `overlap` is the minimum; origins are redistributed evenly so the first is
/// `0`, the last is `frame - tile`, and the realized step is `>= (1-overlap)`.
/// Returns `[0]` when the frame is no larger than the tile.
fn axis_origins(frame: usize, tile: usize, overlap: f64) -> Vec<usize> {
    if frame <= tile {
        return vec![0];
    }
    let last = frame - tile;
    // saturating_sub + max(1): never underflow or div_ceil(0), even for a
    // pathological overlap (config validation rejects >= 1.0 up front).
    let max_step = tile.saturating_sub((overlap * tile as f64) as usize).max(1);
    let total = last.div_ceil(max_step);
    (0..=total)
        .map(|i| (i as f32 * last as f32 / total as f32).round() as usize)
        .collect()
}

/// Uniform overlapping EvenDist tile grid covering a `frame_h`×`frame_w` frame.
/// Row-major (all columns of row 0, then row 1, …). Every tile is full-size
/// (`tile_w`×`tile_h`) unless the frame is smaller than the tile on an axis, in
/// which case that axis yields a single whole-frame crop.
pub fn tile_grid(
    frame_h: usize,
    frame_w: usize,
    tile_h: usize,
    tile_w: usize,
    overlap_ratio: f32,
) -> Vec<TileSpec> {
    let cw = tile_w.min(frame_w);
    let ch = tile_h.min(frame_h);
    let xs = axis_origins(frame_w, tile_w, overlap_ratio as f64);
    let ys = axis_origins(frame_h, tile_h, overlap_ratio as f64);
    let mut tiles = Vec::with_capacity(xs.len() * ys.len());
    let mut index = 0;
    for (row, &oy) in ys.iter().enumerate() {
        for (col, &ox) in xs.iter().enumerate() {
            debug_assert!(
                ox + cw <= frame_w && oy + ch <= frame_h,
                "tile out of bounds"
            );
            tiles.push(TileSpec {
                source: Region::new(ox, oy, cw, ch),
                index,
                row,
                col,
            });
            index += 1;
        }
    }
    tiles
}

/// Normalize a resolved letterbox dst-rect to `[lx0, ly0, lx1, ly1]` on the
/// model input, or `None` for a stretched (whole-destination) placement.
fn placement_letterbox(
    crop: &Crop,
    src_w: usize,
    src_h: usize,
    tile_w: usize,
    tile_h: usize,
) -> Option<[f32; 4]> {
    let resolved = crop.resolve(src_w, src_h, tile_w, tile_h).ok()?;
    resolved.dst_rect.map(|r| {
        let (dw, dh) = (tile_w as f32, tile_h as f32);
        [
            r.left as f32 / dw,
            r.top as f32 / dh,
            (r.left + r.width) as f32 / dw,
            (r.top + r.height) as f32 / dh,
        ]
    })
}

impl ImageProcessor {
    /// Allocate the tall packed batch destination `[tile_w, n*tile_h]` — a
    /// single GL-importable parent that stacks `n` tiles vertically. Uses
    /// [`ImageProcessor::create_image`] so DMA pitch alignment and
    /// `Dma>Pbo>Mem` selection apply. Caller-owned for pool reuse.
    ///
    /// # Errors
    /// Returns an error if `cfg` is invalid (overlap not in `[0,1)`, or zero
    /// tile size) or if the underlying allocation fails.
    pub fn alloc_tile_batch(
        &self,
        n: usize,
        cfg: &TilingConfig,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
    ) -> Result<TensorDyn> {
        cfg.validate()?;
        self.create_image(
            cfg.tile_w,
            n.saturating_mul(cfg.tile_h),
            format,
            dtype,
            memory,
        )
    }

    /// Compute the tile grid and per-tile [`TilePlacement`] metadata for a
    /// `src_w`×`src_h` frame **without touching the GPU**. A pipelined runtime
    /// calls this once per frame to size pools and drive its tile stream.
    ///
    /// # Errors
    /// Returns an error if `cfg` is invalid (overlap not in `[0,1)`, or zero
    /// tile size).
    pub fn plan_tiles(
        &self,
        src_w: usize,
        src_h: usize,
        cfg: &TilingConfig,
    ) -> Result<Vec<TilePlacement>> {
        cfg.validate()?;
        let grid = tile_grid(src_h, src_w, cfg.tile_h, cfg.tile_w, cfg.overlap_ratio);
        let count = grid.len();
        let crop = Crop::default().with_fit(cfg.fit);
        Ok(grid
            .iter()
            .map(|t| {
                let lb = placement_letterbox(
                    &crop.with_source(Some(t.source)),
                    t.source.width,
                    t.source.height,
                    cfg.tile_w,
                    cfg.tile_h,
                );
                TilePlacement {
                    index: t.index,
                    count,
                    origin: (t.source.x as f32, t.source.y as f32),
                    crop_size: (t.source.width as f32, t.source.height as f32),
                    letterbox: lb,
                    frame_dims: (src_w as f32, src_h as f32),
                }
            })
            .collect())
    }

    /// Render every tile of `src` into `dst_batched` (a tall packed parent from
    /// [`Self::alloc_tile_batch`]), one deferred convert per tile, then a single
    /// `flush`. Returns the per-tile [`TilePlacement`] in tile-index order.
    ///
    /// The source crop is selected via [`Crop::with_source`] (a sampled
    /// sub-rect of the whole-frame `src`) — **not** `src.view()` — so all tiles
    /// share one source import. Each destination tile is a `view` row-band of
    /// the shared parent.
    ///
    /// # Errors
    /// Returns an error if `cfg` is invalid, if `src`/`dst_batched` are not
    /// images, if `dst_batched` is shorter than `count * tile_h`, or if a tile
    /// convert fails.
    pub fn tile_into(
        &mut self,
        src: &TensorDyn,
        dst_batched: &mut TensorDyn,
        cfg: &TilingConfig,
    ) -> Result<Vec<TilePlacement>> {
        cfg.validate()?;
        let (src_w, src_h) = (
            src.width().ok_or(Error::NotAnImage)?,
            src.height().ok_or(Error::NotAnImage)?,
        );
        let placements = self.plan_tiles(src_w, src_h, cfg)?;
        let count = placements.len();

        let dst_h = dst_batched.height().ok_or(Error::NotAnImage)?;
        if dst_h < count.saturating_mul(cfg.tile_h) {
            return Err(Error::InvalidShape(format!(
                "tile_into dst height {dst_h} < count*tile_h {}",
                count * cfg.tile_h
            )));
        }

        for p in &placements {
            self.render_tile(src, dst_batched, p, cfg)?;
        }
        self.flush()?;
        Ok(placements)
    }

    /// Render exactly one tile of `src` into `dst_slot` (a single model-input
    /// sized destination, e.g. a slot of a caller-owned ring). Deferred — the
    /// caller flushes on its own cadence so tiles overlap with inference. Holds
    /// no frame-wide state; all geometry rides in `placement` (from
    /// [`Self::plan_tiles`]).
    ///
    /// A deferred tile is not CPU/CUDA-readable until [`ImageProcessorTrait::flush`]
    /// (mirrors [`ImageProcessorTrait::convert_deferred`]).
    ///
    /// # Errors
    /// Returns an error if `cfg` is invalid or the tile convert fails.
    pub fn tile_one(
        &mut self,
        src: &TensorDyn,
        dst_slot: &mut TensorDyn,
        placement: &TilePlacement,
        cfg: &TilingConfig,
    ) -> Result<()> {
        cfg.validate()?;
        self.render_tile(src, dst_slot, placement, cfg)
    }

    /// Issue one deferred convert sampling `placement.source` from `src` into
    /// `dst` (a tall-parent band or a single slot).
    fn render_tile(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        placement: &TilePlacement,
        cfg: &TilingConfig,
    ) -> Result<()> {
        let (ox, oy) = placement.origin;
        let (cw, ch) = placement.crop_size;
        let source = Region::new(ox as usize, oy as usize, cw as usize, ch as usize);
        let crop = Crop::default().with_source(Some(source)).with_fit(cfg.fit);

        // Destination is a row-band of the tall parent (packed) or the slot
        // itself. For a tall batched parent, address this tile's band.
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        if dst_h > cfg.tile_h {
            // Tall parent: this tile's vertical band.
            let mut band = dst.view(Region::new(
                0,
                placement.index * cfg.tile_h,
                cfg.tile_w,
                cfg.tile_h,
            ))?;
            self.convert_deferred(src, &mut band, Rotation::None, Flip::None, crop)?;
        } else {
            self.convert_deferred(src, dst, Rotation::None, Flip::None, crop)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EvenDist geometry (pure, no GPU) ---------------------------------

    #[test]
    fn axis_origins_matches_adis_worked_example() {
        // adis-uav-model sahi(1920, 640, 0.1) => [0, 427, 853, 1280]
        assert_eq!(axis_origins(1920, 640, 0.1), vec![0, 427, 853, 1280]);
    }

    #[test]
    fn axis_origins_4k_axes() {
        // 3840 / 640 / 0.2 => 8 full tiles, last at frame-tile.
        let xs = axis_origins(3840, 640, 0.2);
        assert_eq!(xs, vec![0, 457, 914, 1371, 1829, 2286, 2743, 3200]);
        // 2160 / 640 / 0.2 => 4 tiles.
        let ys = axis_origins(2160, 640, 0.2);
        assert_eq!(ys, vec![0, 507, 1013, 1520]);
        assert_eq!(*xs.last().unwrap(), 3840 - 640);
        assert_eq!(*ys.last().unwrap(), 2160 - 640);
        assert_eq!(xs[0], 0);
    }

    #[test]
    fn axis_origins_frame_le_tile_single() {
        assert_eq!(axis_origins(640, 640, 0.2), vec![0]); // == tile
        assert_eq!(axis_origins(400, 640, 0.2), vec![0]); // < tile
    }

    #[test]
    fn axis_origins_frame_tile_plus_one() {
        // Smallest 2-tile grid; div_ceil boundary.
        assert_eq!(axis_origins(641, 640, 0.2), vec![0, 1]);
    }

    #[test]
    fn axis_origins_overlap_zero_is_exact_tiling() {
        // overlap 0 => step == tile, no overlap.
        assert_eq!(axis_origins(1920, 640, 0.0), vec![0, 640, 1280]);
    }

    #[test]
    fn axis_origins_overlap_near_one_no_panic() {
        // overlap 0.99 must not panic (max_step clamps to >= 1).
        let xs = axis_origins(1920, 640, 0.99);
        assert_eq!(xs[0], 0);
        assert_eq!(*xs.last().unwrap(), 1280);
        // monotone non-decreasing
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
        assert!(xs.len() > 8); // heavy overlap => many tiles
    }

    #[test]
    fn axis_origins_8k_no_overflow() {
        // Large frame: f32 rounding path, no overflow.
        let xs = axis_origins(7680, 640, 0.2);
        assert_eq!(xs[0], 0);
        assert_eq!(*xs.last().unwrap(), 7680 - 640);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn tile_grid_4k_all_full_size_and_in_bounds() {
        let grid = tile_grid(2160, 3840, 640, 640, 0.2);
        assert_eq!(grid.len(), 8 * 4); // 32 tiles
        for t in &grid {
            assert_eq!(t.source.width, 640);
            assert_eq!(t.source.height, 640);
            assert!(t.source.x + 640 <= 3840);
            assert!(t.source.y + 640 <= 2160);
        }
        // row-major: index increments by column then row
        assert_eq!(grid[0].source, Region::new(0, 0, 640, 640));
        assert_eq!(grid[8].source, Region::new(0, 507, 640, 640)); // start of row 1
    }

    #[test]
    fn tile_grid_realized_overlap_at_least_requested() {
        let grid = tile_grid(2160, 3840, 640, 640, 0.2);
        // realized step between first two columns
        let step = grid[1].source.x - grid[0].source.x;
        let realized_overlap = 1.0 - (step as f64 / 640.0);
        assert!(
            realized_overlap >= 0.2 - 1e-9,
            "realized {realized_overlap} < 0.2"
        );
    }

    #[test]
    fn tile_grid_frame_smaller_than_tile_single_whole_frame() {
        let grid = tile_grid(400, 500, 640, 640, 0.2);
        assert_eq!(grid.len(), 1);
        assert_eq!(grid[0].source, Region::new(0, 0, 500, 400));
    }

    #[test]
    fn rounding_uses_f32_path() {
        // overlap 0.333 on a large frame: exercise the f32 round() path.
        let xs = axis_origins(4000, 640, 0.333);
        assert_eq!(xs[0], 0);
        assert_eq!(*xs.last().unwrap(), 4000 - 640);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn tiling_config_validate_rejects_bad_overlap() {
        assert!(TilingConfig::new(640, 640)
            .with_overlap(1.0)
            .validate()
            .is_err());
        assert!(TilingConfig::new(640, 640)
            .with_overlap(-0.1)
            .validate()
            .is_err());
        assert!(TilingConfig::new(640, 640)
            .with_overlap(0.2)
            .validate()
            .is_ok());
    }

    #[test]
    fn tiling_config_validate_rejects_zero_tile_size() {
        assert!(TilingConfig::new(0, 640).validate().is_err());
        assert!(TilingConfig::new(640, 0).validate().is_err());
    }
}
