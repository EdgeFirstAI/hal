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
//! coordinates and merged by `edgefirst_decoder::tiling`; the shared
//! [`TilePlacement`] is produced here and consumed there.

use crate::Error;
use crate::{ConvertOptions, Flip, ImageProcessor, ImageProcessorTrait, Result, Rotation};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMemory};

pub use edgefirst_decoder_abi::TilePlacement;

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
    options: ConvertOptions,
}

impl TilingConfig {
    /// Tile size = model input; overlap 0.2, tiles stretched to the model
    /// input (identity for the full-size tiles the grid produces) with the
    /// default pixel mapping.
    pub fn new(tile_w: usize, tile_h: usize) -> Self {
        Self {
            tile_w,
            tile_h,
            overlap_ratio: 0.2,
            options: ConvertOptions::new(),
        }
    }

    /// Set the minimum overlap ratio (builder).
    pub fn with_overlap(mut self, overlap_ratio: f32) -> Self {
        self.overlap_ratio = overlap_ratio;
        self
    }

    /// Set the options every tile is converted with: fit, letterbox
    /// placement and pad, normalization and interpolation, e.g. from
    /// [`ConvertOptions::from_spec`]. Each tile replaces the source crop
    /// with its own. Rotation and flip are refused by [`validate`](Self::validate),
    /// since tiles could not be lifted back to the frame.
    pub fn with_options(mut self, options: ConvertOptions) -> Self {
        self.options = options;
        self
    }

    /// The options every tile is converted with.
    pub fn options(&self) -> ConvertOptions {
        self.options
    }

    /// Validate the config: `overlap_ratio` in `[0.0, 1.0)`, non-zero tile
    /// size, and options without rotation or flip. Called by every tiling
    /// method, and exposed so callers (e.g. the language bindings) can reject
    /// bad input before generating a grid.
    ///
    /// # Errors
    /// Returns [`Error::CropInvalid`] if the overlap is out of range or a tile
    /// dimension is zero, and [`Error::InvalidOptions`] for a rotation or flip.
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
        if self.options.get_rotation() != Rotation::None || self.options.get_flip() != Flip::None {
            return Err(Error::InvalidOptions(
                "tiling options cannot rotate or flip".into(),
            ));
        }
        Ok(())
    }
}

/// EvenDist 1-D tile origins along one axis. Mirrors adis-uav-model `sahi()`:
/// `overlap` is the *minimum*; origins are redistributed evenly so the first
/// is `0`, the last is `frame - tile`, and the realized step is
/// `<= floor(tile * (1 - overlap))` (realized overlap `>=` requested).
/// Returns `[0]` when the frame is no larger than the tile.
fn axis_origins(frame: usize, tile: usize, overlap: f64) -> Vec<usize> {
    if frame <= tile {
        return vec![0];
    }
    let last = frame - tile;
    // Use floor(tile*(1-overlap)), not tile-floor(overlap*tile): the latter can
    // be 1px larger than tile*(1-overlap) for non-integer ratios (e.g. 0.333),
    // which would under-count tiles and realize less than the requested overlap.
    // Clamp to >= 1 so a pathological overlap never div_ceil(0) (config
    // validation already rejects overlap >= 1.0 up front).
    let max_step = ((1.0 - overlap) * tile as f64).floor().max(1.0) as usize;
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

impl ImageProcessor {
    /// Allocate the tall packed batch destination `[tile_w, n*tile_h]` — a
    /// single GL-importable parent that stacks `n` tiles vertically. Uses
    /// [`ImageProcessor::create_image`] so DMA pitch alignment and
    /// `Dma>Pbo>Mem` selection apply. Caller-owned for pool reuse.
    ///
    /// `access` declares planned CPU involvement (see
    /// [`edgefirst_tensor::CpuAccess`]); use `None` for NPU-bound tile
    /// batches and `ReadWrite` when the batch will be CPU-mapped. As for
    /// `create_image`, what this processor's own convert may do on the CPU
    /// is added; tiles are written through views, so any predicted CPU
    /// write is added as `ReadWrite`.
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
        access: CpuAccess,
    ) -> Result<TensorDyn> {
        cfg.validate()?;
        self.create_image_with(
            cfg.tile_w,
            n.saturating_mul(cfg.tile_h),
            format,
            dtype,
            memory,
            access,
            true,
        )
    }

    /// Compute the tile grid and per-tile [`TilePlacement`] metadata for a
    /// `src_w`×`src_h` frame **without touching the GPU**. A pipelined runtime
    /// calls this once per frame to size pools and drive its tile stream.
    ///
    /// The realized tile count can exceed the naive `ceil(frame / tile)`: the
    /// grid treats `cfg.overlap_ratio` as a *minimum* and redistributes the
    /// surplus evenly, so allocate from `len()` rather than estimating.
    ///
    /// The returned placements are consumed by
    /// `edgefirst_decoder::tiling` to lift per-tile detections back to
    /// full-frame coordinates.
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
        let span = tracing::trace_span!(
            "image.plan_tiles",
            tiles = tracing::field::Empty,
            overlap = cfg.overlap_ratio,
        );
        let _s = span.enter();
        cfg.validate()?;
        let grid = tile_grid(src_h, src_w, cfg.tile_h, cfg.tile_w, cfg.overlap_ratio);
        let count = grid.len();
        span.record("tiles", count);
        grid.iter()
            .map(|t| {
                let region = cfg
                    .options
                    .source(Some(t.source))
                    .content_region(src_w, src_h, cfg.tile_w, cfg.tile_h)?;
                Ok(TilePlacement {
                    index: t.index,
                    count,
                    origin: (t.source.x as f32, t.source.y as f32),
                    crop_size: (t.source.width as f32, t.source.height as f32),
                    content_region: region.letterboxed(),
                    frame_dims: (src_w as f32, src_h as f32),
                })
            })
            .collect()
    }

    /// Render every tile of `src` into `dst_batched` (a tall packed parent from
    /// [`Self::alloc_tile_batch`]), one deferred convert per tile, then a single
    /// `flush`. Returns the per-tile [`TilePlacement`] in tile-index order.
    ///
    /// The source crop is selected via [`ConvertOptions::source`] (a sampled
    /// sub-rect of the whole-frame `src`) — **not** `src.view()` — so all tiles
    /// share one source import. Each destination tile is a `view` row-band of
    /// the shared parent.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use edgefirst_image::{ImageProcessor, TilingConfig};
    /// # use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn};
    /// # fn run(src: &TensorDyn) -> Result<(), edgefirst_image::Error> {
    /// let mut processor = ImageProcessor::new()?;
    /// let cfg = TilingConfig::new(640, 640).with_overlap(0.2);
    ///
    /// // Pure geometry — size the batch before touching the GPU.
    /// let planned = processor.plan_tiles(3840, 2160, &cfg)?;
    /// let mut batch = processor.alloc_tile_batch(
    ///     planned.len(), &cfg, PixelFormat::Rgba, DType::U8, None, CpuAccess::None,
    /// )?;
    ///
    /// // One import, one flush, N tiles.
    /// let placements = processor.tile_into(src, &mut batch, &cfg)?;
    /// assert_eq!(placements.len(), planned.len());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Backend note
    ///
    /// The GL band lowering covers single-pass geometry only. A zero-copy
    /// (DMA) `dst_batched` in packed `Rgb` or a planar layout reinterprets the
    /// render-target shape, which the band path does not yet compute, so GL
    /// declines and each tile falls back to CPU. `Rgba`, `Bgra`, and `Grey`
    /// batches stay on the batched GPU path.
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
        let span = tracing::trace_span!("image.tile_into", tiles = tracing::field::Empty);
        let _s = span.enter();
        cfg.validate()?;
        let (src_w, src_h) = (
            src.width().ok_or(Error::NotAnImage)?,
            src.height().ok_or(Error::NotAnImage)?,
        );
        let placements = self.plan_tiles(src_w, src_h, cfg)?;
        let count = placements.len();
        span.record("tiles", count);

        let dst_h = dst_batched.height().ok_or(Error::NotAnImage)?;
        let required_h = count.saturating_mul(cfg.tile_h);
        if dst_h < required_h {
            return Err(Error::InvalidShape(format!(
                "tile_into dst height {dst_h} < count*tile_h {required_h}"
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
        let _s = tracing::trace_span!(
            "image.tile_one",
            index = placement.index,
            count = placement.count,
        )
        .entered();
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
        let options = cfg
            .options
            .source(Some(placement_to_source_region(placement)?));

        // Destination is a row-band of the tall parent (packed) or the slot
        // itself. Decide batched vs slot from capacity vs. `placement.count`,
        // not merely `dst_h > tile_h` — a padded `tile_one` slot can be taller
        // than `tile_h` without being a tall parent.
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let batched = dst_h >= placement.count.saturating_mul(cfg.tile_h);
        if batched {
            let mut band = dst.view(Region::new(
                0,
                placement.index * cfg.tile_h,
                cfg.tile_w,
                cfg.tile_h,
            ))?;
            self.convert_deferred(src, &mut band, Some(options))?;
        } else {
            self.convert_deferred(src, dst, Some(options))?;
        }
        Ok(())
    }
}

/// Validate a [`TilePlacement`]'s origin/crop and convert to a pixel [`Region`].
///
/// Rejects negative, non-finite, or non-integral (after truncation) geometry
/// that Python/C callers could otherwise smuggle in via the public placement
/// constructors.
fn placement_to_source_region(placement: &TilePlacement) -> Result<Region> {
    let (ox, oy) = placement.origin;
    let (cw, ch) = placement.crop_size;
    for (name, v) in [
        ("origin.x", ox),
        ("origin.y", oy),
        ("crop.w", cw),
        ("crop.h", ch),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(Error::CropInvalid(format!(
                "tile placement {name} must be finite and non-negative, got {v}"
            )));
        }
    }
    let (ox_i, oy_i, cw_i, ch_i) = (ox as usize, oy as usize, cw as usize, ch as usize);
    // Reject fractional values that would silently truncate (e.g. origin.x=1.5).
    for (name, f, i) in [
        ("origin.x", ox, ox_i),
        ("origin.y", oy, oy_i),
        ("crop.w", cw, cw_i),
        ("crop.h", ch, ch_i),
    ] {
        if (f - i as f32).abs() > f32::EPSILON {
            return Err(Error::CropInvalid(format!(
                "tile placement {name} must be an integral pixel value, got {f}"
            )));
        }
    }
    if cw_i == 0 || ch_i == 0 {
        return Err(Error::CropInvalid(
            "tile placement crop size must be non-zero".into(),
        ));
    }
    Ok(Region::new(ox_i, oy_i, cw_i, ch_i))
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
        // overlap 0.333 on a large frame: exercise the f32 round() path AND
        // the floor(tile*(1-overlap)) minimum-overlap contract (max_step must
        // be 426, not 427 — tile - floor(overlap*tile) would under-count).
        let xs = axis_origins(4000, 640, 0.333);
        assert_eq!(xs[0], 0);
        assert_eq!(*xs.last().unwrap(), 4000 - 640);
        assert!(xs.windows(2).all(|w| w[0] <= w[1]));
        let max_step = xs.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        let realized = 1.0 - (max_step as f64 / 640.0);
        assert!(
            realized >= 0.333 - 1e-9,
            "realized overlap {realized} < 0.333 (max_step={max_step})"
        );
    }

    #[test]
    fn tiling_options_refuse_rotation_and_flip() {
        let cfg = TilingConfig::new(640, 640);
        assert!(cfg.validate().is_ok());
        let rotated = cfg.with_options(ConvertOptions::new().rotation(Rotation::Clockwise90));
        assert!(matches!(rotated.validate(), Err(Error::InvalidOptions(_))));
        let flipped = cfg.with_options(ConvertOptions::new().flip(Flip::Horizontal));
        assert!(matches!(flipped.validate(), Err(Error::InvalidOptions(_))));
    }

    fn cpu() -> ImageProcessor {
        ImageProcessor::with_config(crate::ImageProcessorConfig {
            backend: crate::ComputeBackend::Cpu,
            ..Default::default()
        })
        .unwrap()
    }

    /// A frame shorter than the tile letterboxes every tile; tiles away from
    /// the frame origin carry the same content region as the first.
    #[test]
    fn plan_tiles_content_region_on_every_tile() {
        let options = ConvertOptions::new()
            .letterbox([114, 114, 114, 255])
            .placement(crate::LetterboxPlacement::TopLeft);
        let cfg = TilingConfig::new(640, 640).with_options(options);
        let plan = cpu().plan_tiles(1500, 400, &cfg).unwrap();
        assert!(plan.len() >= 3);
        for p in &plan {
            assert_eq!(
                p.content_region,
                Some([0.0, 0.0, 1.0, 0.625]),
                "tile {}",
                p.index
            );
        }
        let centred = cfg.with_options(options.placement(crate::LetterboxPlacement::Center));
        let plan = cpu().plan_tiles(1500, 400, &centred).unwrap();
        assert!(plan
            .iter()
            .all(|p| p.content_region == Some([0.0, 0.1875, 1.0, 0.8125])));
    }

    /// Lifting a box over a tile's whole content region gives back the
    /// tile's crop in frame pixels.
    #[test]
    fn top_left_tile_boxes_lift_exactly() {
        use edgefirst_tensor::{BoundingBox, DetectBox};
        let cfg = TilingConfig::new(640, 640).with_options(
            ConvertOptions::new()
                .letterbox([0, 0, 0, 255])
                .placement(crate::LetterboxPlacement::TopLeft),
        );
        for p in cpu().plan_tiles(1500, 400, &cfg).unwrap() {
            let [x0, y0, x1, y1] = p.content_region.unwrap();
            let b = DetectBox {
                bbox: BoundingBox {
                    xmin: x0,
                    ymin: y0,
                    xmax: x1,
                    ymax: y1,
                },
                score: 1.0,
                label: 0,
            };
            let lifted = edgefirst_decoder::tiling::lift_tile_boxes(vec![b], &p);
            let l = lifted[0].bbox;
            let want = [p.origin.0, 0.0, p.origin.0 + p.crop_size.0, 400.0];
            for (got, want) in [l.xmin, l.ymin, l.xmax, l.ymax].iter().zip(want) {
                assert!(
                    (got - want).abs() < 1e-3,
                    "tile {}: {l:?} vs {want:?}",
                    p.index
                );
            }
        }
    }

    /// A tile is converted with the base options plus its own crop.
    #[test]
    fn tile_render_applies_options() {
        use edgefirst_tensor::{TensorMapTrait, TensorTrait};
        let mut proc = cpu();
        let src = TensorDyn::image(
            1500,
            400,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = src.as_u8().unwrap();
            let mut m = t.map().unwrap();
            for (i, v) in m.as_mut_slice().iter_mut().enumerate() {
                *v = (i * 7 % 251) as u8;
            }
        }
        let options = ConvertOptions::new()
            .letterbox([0, 0, 0, 255])
            .placement(crate::LetterboxPlacement::TopLeft)
            .pad_output([0.0; 4])
            .normalization(crate::Normalization::imagenet());
        let cfg = TilingConfig::new(320, 320).with_options(options);
        let plan = proc.plan_tiles(1500, 400, &cfg).unwrap();
        let p = plan[plan.len() / 2];
        let image = || {
            TensorDyn::image(
                320,
                320,
                PixelFormat::PlanarRgb,
                DType::F32,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut tile = image();
        proc.tile_one(&src, &mut tile, &p, &cfg).unwrap();
        proc.flush().unwrap();
        let mut direct = image();
        let source = Region::new(
            p.origin.0 as usize,
            p.origin.1 as usize,
            p.crop_size.0 as usize,
            p.crop_size.1 as usize,
        );
        proc.convert(&src, &mut direct, Some(options.source(Some(source))))
            .unwrap();
        let a = tile
            .as_f32()
            .unwrap()
            .map_read()
            .unwrap()
            .as_slice()
            .to_vec();
        let b = direct
            .as_f32()
            .unwrap()
            .map_read()
            .unwrap()
            .as_slice()
            .to_vec();
        assert_eq!(a, b);
        assert!(a.iter().any(|v| *v < 0.0), "imagenet normalization applied");
    }

    /// A tile resamples with the base options' interpolation, not the
    /// processor's.
    #[test]
    fn tile_render_applies_interpolation() {
        use edgefirst_tensor::{TensorMapTrait, TensorTrait};
        let mut proc = cpu();
        let src = TensorDyn::image(
            200,
            100,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        {
            let t = src.as_u8().unwrap();
            let mut m = t.map().unwrap();
            for (i, v) in m.as_mut_slice().iter_mut().enumerate() {
                *v = (i * 7 % 251) as u8;
            }
        }
        let base = ConvertOptions::new().letterbox([0, 0, 0, 255]);
        let options = base.interpolation(crate::Interpolation::Nearest);
        let cfg = TilingConfig::new(320, 320).with_options(options);
        let plan = proc.plan_tiles(200, 100, &cfg).unwrap();
        let read = |t: &TensorDyn| t.as_u8().unwrap().map().unwrap().as_slice().to_vec();
        let image = || {
            TensorDyn::image(
                320,
                320,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let mut tile = image();
        proc.tile_one(&src, &mut tile, &plan[0], &cfg).unwrap();
        proc.flush().unwrap();
        let mut nearest = image();
        proc.convert(&src, &mut nearest, Some(options)).unwrap();
        let mut bilinear = image();
        proc.convert(&src, &mut bilinear, Some(base)).unwrap();
        assert_ne!(read(&nearest), read(&bilinear));
        assert_eq!(read(&tile), read(&nearest));
    }

    #[test]
    fn placement_to_source_region_rejects_invalid() {
        let good = TilePlacement {
            index: 0,
            count: 1,
            origin: (10.0, 20.0),
            crop_size: (640.0, 640.0),
            content_region: None,
            frame_dims: (3840.0, 2160.0),
        };
        assert!(placement_to_source_region(&good).is_ok());

        let mut bad = good;
        bad.origin.0 = -1.0;
        assert!(placement_to_source_region(&bad).is_err());
        bad = good;
        bad.origin.0 = f32::NAN;
        assert!(placement_to_source_region(&bad).is_err());
        bad = good;
        bad.origin.0 = 1.5;
        assert!(placement_to_source_region(&bad).is_err());
        bad = good;
        bad.crop_size.0 = 0.0;
        assert!(placement_to_source_region(&bad).is_err());
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

    // --- span instrumentation ----------------------------------------------

    /// Minimal `tracing::Subscriber` that records every span's name, in
    /// creation order, into a shared `Vec`. Enough to assert *which* spans
    /// fired without pulling in `tracing-subscriber` (not a dependency of
    /// this crate) or `tracing-test`.
    struct SpanNameCapture {
        names: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        next_id: std::sync::atomic::AtomicU64,
    }

    impl tracing::Subscriber for SpanNameCapture {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            self.names
                .lock()
                .unwrap()
                .push(span.metadata().name().to_string());
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            tracing::span::Id::from_u64(id)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, _event: &tracing::Event<'_>) {}
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    #[test]
    fn tiling_emits_spans() {
        let names = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = SpanNameCapture {
            names: names.clone(),
            next_id: std::sync::atomic::AtomicU64::new(0),
        };

        // A second dispatcher stops tracing-core using another thread's default to rebuild interest.
        let _other = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        tracing::subscriber::with_default(subscriber, || {
            let mut proc = ImageProcessor::new().expect("processor");
            let cfg = TilingConfig::new(320, 240);
            let placements = proc.plan_tiles(640, 480, &cfg).expect("plan_tiles");

            let src = TensorDyn::image(
                640,
                480,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .expect("src");
            let mut dst = TensorDyn::image(
                320,
                240,
                PixelFormat::Rgb,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .expect("dst");

            proc.tile_one(&src, &mut dst, &placements[0], &cfg)
                .expect("tile_one");
            let _ = proc.flush();
        });

        let names = names.lock().unwrap();
        assert!(
            names.iter().any(|n| n == "image.plan_tiles"),
            "expected image.plan_tiles in {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "image.tile_one"),
            "expected image.tile_one in {names:?}"
        );
    }
}
