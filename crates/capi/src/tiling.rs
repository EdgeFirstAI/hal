// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tiling C API - SAHI-style input tiling and tiled-detection postprocessing.
//!
//! This module exposes the input-side grid/preprocess primitives from
//! `edgefirst_image::tiling` and the output-side lift/merge/accumulate
//! primitives from `edgefirst_decoder::tiling`.
//!
//! ## POD configuration structs
//!
//! [`HalTilingConfig`] and [`HalMergeConfig`] are passed by value. Zero-init
//! (`memset(0)`) does **not** match the library defaults — e.g. overlap would
//! be `0` instead of `0.2`, and the merge metric/`max_det` would not match
//! IOS / 300 — even though `HAL_FIT_STRETCH` happens to be `0`. Always seed
//! them with [`hal_tiling_config_default`] / [`hal_merge_config_default`]
//! before overriding fields.
//!
//! ## Box list I/O
//!
//! Detection boxes cross the boundary as the existing opaque
//! `hal_detect_box_list` (see `decoder.rs`). Inputs are **borrowed** (cloned
//! internally before calling the by-value Rust APIs); outputs are freshly
//! allocated lists the caller frees with `hal_detect_box_list_free()`.

use crate::decoder::HalDetectBoxList;
use crate::error::{set_error, set_error_null};
use crate::image::{
    HalImageProcessor, HalPixelFormat, HalRegion, HAL_FIT_LETTERBOX, HAL_FIT_STRETCH,
};
use crate::tensor::{HalDtype, HalTensor, HalTensorMemory};
use crate::{check_null, check_null_ret_null};
use edgefirst_decoder::tiling::{
    lift_tile_boxes, merge_tiled_detections, MatchMetric, MergeConfig, TilePlacement,
    TiledFrameAccumulator,
};
use edgefirst_decoder::DetectBox;
use edgefirst_image::{tile_grid, Fit, TileSpec, TilingConfig};
use edgefirst_tensor::TensorMemory;
use libc::{c_int, size_t};

// ============================================================================
// Match metric enum
// ============================================================================

/// Overlap metric used by the tiled-detection merge.
///
/// The default (used by `hal_merge_config_default()`) is
/// `HAL_MATCH_METRIC_IOS`: intersection-over-smaller merges seam-split
/// fragments that IoU would leave separate.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalMatchMetric {
    /// Intersection-over-Union (standard NMS metric).
    Iou = 0,
    /// Intersection-over-Smaller (default): `inter / min(area_a, area_b)`.
    Ios = 1,
}

impl From<HalMatchMetric> for MatchMetric {
    fn from(m: HalMatchMetric) -> Self {
        match m {
            HalMatchMetric::Iou => MatchMetric::Iou,
            HalMatchMetric::Ios => MatchMetric::Ios,
        }
    }
}

impl From<MatchMetric> for HalMatchMetric {
    fn from(m: MatchMetric) -> Self {
        match m {
            MatchMetric::Iou => HalMatchMetric::Iou,
            MatchMetric::Ios => HalMatchMetric::Ios,
        }
    }
}

// ============================================================================
// POD configuration / placement / spec structs
// ============================================================================

/// Static tiling configuration (independent of frame size). Passed by value.
///
/// Seed with `hal_tiling_config_default()` then override fields; do **not**
/// `memset(0)` — that zeroes `overlap_ratio` (default is 0.2) even though
/// `fit == 0` happens to equal `HAL_FIT_STRETCH`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalTilingConfig {
    /// Tile width = model input width (pixels).
    pub tile_w: size_t,
    /// Tile height = model input height (pixels).
    pub tile_h: size_t,
    /// Minimum overlap ratio between adjacent tiles, `0.0 <= r < 1.0`.
    pub overlap_ratio: f32,
    /// Letterbox pad colour (RGBA), used only when `fit == HAL_FIT_LETTERBOX`.
    pub pad: [u8; 4],
    /// Fit mode: `HAL_FIT_STRETCH` (0) or `HAL_FIT_LETTERBOX` (1).
    pub fit: c_int,
}

impl From<HalTilingConfig> for TilingConfig {
    fn from(c: HalTilingConfig) -> Self {
        let fit = if c.fit == HAL_FIT_LETTERBOX {
            Fit::Letterbox { pad: c.pad }
        } else {
            Fit::Stretch
        };
        TilingConfig {
            tile_w: c.tile_w,
            tile_h: c.tile_h,
            overlap_ratio: c.overlap_ratio,
            pad: c.pad,
            fit,
        }
    }
}

/// One tile's native-frame crop rectangle and grid coordinates. Passed by value.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalTileSpec {
    /// Native crop in full-frame pixels.
    pub source: HalRegion,
    /// Row-major flat index, `0..count`.
    pub index: size_t,
    /// Grid row.
    pub row: size_t,
    /// Grid column.
    pub col: size_t,
}

impl From<&TileSpec> for HalTileSpec {
    fn from(t: &TileSpec) -> Self {
        HalTileSpec {
            source: HalRegion {
                x: t.source.x,
                y: t.source.y,
                width: t.source.width,
                height: t.source.height,
            },
            index: t.index,
            row: t.row,
            col: t.col,
        }
    }
}

/// How one tile was cut from the full frame and fed to the model. Passed by
/// value. Produced by `hal_image_processor_plan_tiles()` /
/// `hal_image_processor_tile_into()`, consumed by `hal_lift_tile_boxes()` and
/// the accumulator.
///
/// The Rust `Option<[f32; 4]>` letterbox is flattened to a `has_letterbox`
/// flag plus an inline `letterbox[4]` array — never a pointer in a by-value
/// struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalTilePlacement {
    /// Tile index within the frame grid, `0..count`.
    pub index: size_t,
    /// Total tiles for this frame (the streaming fan-in fence).
    pub count: size_t,
    /// Native crop origin x in full-frame pixels.
    pub origin_x: f32,
    /// Native crop origin y in full-frame pixels.
    pub origin_y: f32,
    /// Native crop width in full-frame pixels.
    pub crop_w: f32,
    /// Native crop height in full-frame pixels.
    pub crop_h: f32,
    /// Full-frame width in pixels.
    pub frame_w: f32,
    /// Full-frame height in pixels.
    pub frame_h: f32,
    /// Whether `letterbox` carries valid content bounds (else the crop was
    /// stretched to fill the model input).
    pub has_letterbox: bool,
    /// Normalized letterbox content bounds `[lx0, ly0, lx1, ly1]` on the model
    /// input. Only valid when `has_letterbox` is true.
    pub letterbox: [f32; 4],
}

impl From<&TilePlacement> for HalTilePlacement {
    fn from(p: &TilePlacement) -> Self {
        let (has_letterbox, letterbox) = match p.letterbox {
            Some(lb) => (true, lb),
            None => (false, [0.0; 4]),
        };
        HalTilePlacement {
            index: p.index,
            count: p.count,
            origin_x: p.origin.0,
            origin_y: p.origin.1,
            crop_w: p.crop_size.0,
            crop_h: p.crop_size.1,
            frame_w: p.frame_dims.0,
            frame_h: p.frame_dims.1,
            has_letterbox,
            letterbox,
        }
    }
}

impl From<&HalTilePlacement> for TilePlacement {
    fn from(p: &HalTilePlacement) -> Self {
        TilePlacement {
            index: p.index,
            count: p.count,
            origin: (p.origin_x, p.origin_y),
            crop_size: (p.crop_w, p.crop_h),
            letterbox: if p.has_letterbox {
                Some(p.letterbox)
            } else {
                None
            },
            frame_dims: (p.frame_w, p.frame_h),
        }
    }
}

/// Configuration for the tiled-detection merge. Passed by value.
///
/// Seed with `hal_merge_config_default()` then override fields; do **not**
/// `memset(0)` — that selects `HAL_MATCH_METRIC_IOU` and `max_det == 0`,
/// which is **not** the intended default (IOS / 300).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalMergeConfig {
    /// Overlap metric: `HAL_MATCH_METRIC_IOU` (0) or `HAL_MATCH_METRIC_IOS` (1).
    pub metric: c_int,
    /// Merge two boxes when `metric(a, b) >= threshold`.
    pub threshold: f32,
    /// Merge across classes when true.
    pub class_agnostic: bool,
    /// Cap on returned detections after the merge.
    pub max_det: size_t,
    /// Drop merged groups whose max score is below this (0.0 = keep all).
    pub score_threshold: f32,
}

impl From<HalMergeConfig> for MergeConfig {
    fn from(c: HalMergeConfig) -> Self {
        let metric = if c.metric == HalMatchMetric::Iou as c_int {
            MatchMetric::Iou
        } else {
            MatchMetric::Ios
        };
        MergeConfig {
            metric,
            threshold: c.threshold,
            class_agnostic: c.class_agnostic,
            max_det: c.max_det,
            score_threshold: c.score_threshold,
        }
    }
}

impl From<MergeConfig> for HalMergeConfig {
    fn from(c: MergeConfig) -> Self {
        HalMergeConfig {
            metric: HalMatchMetric::from(c.metric) as c_int,
            threshold: c.threshold,
            class_agnostic: c.class_agnostic,
            max_det: c.max_det,
            score_threshold: c.score_threshold,
        }
    }
}

// ============================================================================
// Box list <-> Vec<DetectBox> bridges
// ============================================================================

/// Borrow a `hal_detect_box_list` as a cloned `Vec<DetectBox>` for a by-value
/// Rust API. The input list is left untouched (the caller still owns and frees
/// it).
///
/// # Safety
/// `list` must be NULL or a valid `HalDetectBoxList` pointer.
unsafe fn clone_boxes(list: *const HalDetectBoxList) -> Vec<DetectBox> {
    if list.is_null() {
        Vec::new()
    } else {
        (*list).boxes.clone()
    }
}

// ============================================================================
// Default constructors (mandatory — zero-init selects the wrong defaults)
// ============================================================================

/// Build a tiling configuration with deploy defaults (overlap 0.2, stretch
/// fit, `[114,114,114,255]` pad) and the given tile size.
///
/// @param tile_w Tile width = model input width (pixels)
/// @param tile_h Tile height = model input height (pixels)
/// @return Fully-initialized tiling configuration by value
#[no_mangle]
pub extern "C" fn hal_tiling_config_default(tile_w: size_t, tile_h: size_t) -> HalTilingConfig {
    let cfg = TilingConfig::new(tile_w, tile_h);
    HalTilingConfig {
        tile_w: cfg.tile_w,
        tile_h: cfg.tile_h,
        overlap_ratio: cfg.overlap_ratio,
        pad: cfg.pad,
        fit: match cfg.fit {
            Fit::Letterbox { .. } => HAL_FIT_LETTERBOX,
            Fit::Stretch => HAL_FIT_STRETCH,
        },
    }
}

/// Build a merge configuration with library defaults (IOS metric, threshold
/// 0.5, class-aware, `max_det` 300, score threshold 0.0).
///
/// @return Fully-initialized merge configuration by value
#[no_mangle]
pub extern "C" fn hal_merge_config_default() -> HalMergeConfig {
    HalMergeConfig::from(MergeConfig::default())
}

// ============================================================================
// Opaque list handles
// ============================================================================

/// List of tile specs (grid output). Free with `hal_tile_spec_list_free()`.
pub struct HalTileSpecList {
    specs: Vec<TileSpec>,
}

/// List of tile placements (plan output). Free with
/// `hal_tile_placement_list_free()`.
pub struct HalTilePlacementList {
    placements: Vec<TilePlacement>,
}

/// Streaming collector for one frame's tiled detections.
///
/// Created by `hal_tiled_frame_accumulator_new()`. The `_finalize` /
/// `_finalize_normalized` calls **consume** the handle (it is freed exactly
/// once internally); do not call `_free` after finalizing. Use `_free` only to
/// abandon an accumulator without finalizing.
pub struct HalTiledFrameAccumulator {
    inner: TiledFrameAccumulator,
}

// ============================================================================
// Grid
// ============================================================================

/// Compute the uniform overlapping EvenDist tile grid for a frame.
///
/// Row-major (all columns of row 0, then row 1, …). Every tile is full-size
/// unless the frame is smaller than the tile on an axis, in which case that
/// axis yields a single whole-frame crop.
///
/// @param frame_h Frame height in pixels
/// @param frame_w Frame width in pixels
/// @param tile_h Tile height in pixels (must be non-zero)
/// @param tile_w Tile width in pixels (must be non-zero)
/// @param overlap_ratio Minimum overlap ratio, `0.0 <= r < 1.0`
/// @return Tile spec list (free with `hal_tile_spec_list_free()`), or NULL on
///         error
/// @par Errors (errno):
/// - EINVAL: Zero tile size or overlap_ratio outside `[0.0, 1.0)`
#[no_mangle]
pub extern "C" fn hal_tile_grid(
    frame_h: size_t,
    frame_w: size_t,
    tile_h: size_t,
    tile_w: size_t,
    overlap_ratio: f32,
) -> *mut HalTileSpecList {
    if tile_w == 0 || tile_h == 0 || !(0.0..1.0).contains(&overlap_ratio) {
        return set_error_null(libc::EINVAL);
    }
    let specs = tile_grid(frame_h, frame_w, tile_h, tile_w, overlap_ratio);
    Box::into_raw(Box::new(HalTileSpecList { specs }))
}

// ============================================================================
// ImageProcessor tiling (GPU paths)
// ============================================================================

/// Allocate a tall packed batch destination that stacks `n` tiles vertically.
///
/// @param processor Image processor handle
/// @param n Number of tiles to stack
/// @param config Tiling configuration (must be non-NULL)
/// @param format Pixel format (HAL_PIXEL_FORMAT_*)
/// @param dtype Data type of tensor elements (HAL_DTYPE_*)
/// @param memory Memory allocation type (HAL_TENSOR_DMA recommended)
/// @param access Declared CPU access (see HalCpuAccess)
/// @return New tensor handle on success, NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL argument or invalid config (zero tile size, bad overlap)
/// - ENOTSUP: Unsupported format/operation
/// - EIO: DMA/GPU buffer allocation failed
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_alloc_tile_batch(
    processor: *mut HalImageProcessor,
    n: size_t,
    config: *const HalTilingConfig,
    format: HalPixelFormat,
    dtype: HalDtype,
    memory: HalTensorMemory,
    access: crate::tensor::HalCpuAccess,
) -> *mut HalTensor {
    check_null_ret_null!(processor, config);
    let cfg: TilingConfig = (*config).into();
    let mem_opt: Option<TensorMemory> = memory.into();

    match (*processor).inner.alloc_tile_batch(
        n,
        &cfg,
        format.to_pixel_format(),
        dtype.into(),
        mem_opt,
        access.into(),
    ) {
        Ok(t) => Box::into_raw(Box::new(HalTensor { inner: t })),
        Err(e) => set_error_null(image_err_to_errno(&e)),
    }
}

/// Compute the tile grid and per-tile placement metadata for a frame without
/// touching the GPU.
///
/// @param processor Image processor handle
/// @param src_w Source frame width in pixels
/// @param src_h Source frame height in pixels
/// @param config Tiling configuration (must be non-NULL)
/// @return Tile placement list (free with `hal_tile_placement_list_free()`),
///         or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL argument or invalid config
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_plan_tiles(
    processor: *mut HalImageProcessor,
    src_w: size_t,
    src_h: size_t,
    config: *const HalTilingConfig,
) -> *mut HalTilePlacementList {
    check_null_ret_null!(processor, config);
    let cfg: TilingConfig = (*config).into();
    match (*processor).inner.plan_tiles(src_w, src_h, &cfg) {
        Ok(placements) => Box::into_raw(Box::new(HalTilePlacementList { placements })),
        Err(e) => set_error_null(image_err_to_errno(&e)),
    }
}

/// Render every tile of `src` into `dst` (a tall packed parent from
/// `hal_image_processor_alloc_tile_batch()`), then flush once.
///
/// @param processor Image processor handle
/// @param src Source image tensor (whole frame)
/// @param dst Destination batched tensor (tall parent)
/// @param config Tiling configuration (must be non-NULL)
/// @return Tile placement list (free with `hal_tile_placement_list_free()`),
///         or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL argument or invalid config / non-image src/dst
/// - ENOSPC: Destination too small to hold all tile bands
/// - EIO: Tile convert failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_tile_into(
    processor: *mut HalImageProcessor,
    src: *const HalTensor,
    dst: *mut HalTensor,
    config: *const HalTilingConfig,
) -> *mut HalTilePlacementList {
    check_null_ret_null!(processor, src, dst, config);
    let cfg: TilingConfig = (*config).into();
    match (*processor)
        .inner
        .tile_into(&(*src).inner, &mut (*dst).inner, &cfg)
    {
        Ok(placements) => Box::into_raw(Box::new(HalTilePlacementList { placements })),
        Err(e) => set_error_null(image_err_to_errno(&e)),
    }
}

/// Render exactly one tile of `src` into `dst` (a single model-input sized
/// slot). Deferred — the caller flushes on its own cadence.
///
/// @param processor Image processor handle
/// @param src Source image tensor (whole frame)
/// @param dst Destination slot tensor (model-input sized)
/// @param placement Tile placement from `hal_image_processor_plan_tiles()`
/// @param config Tiling configuration (must be non-NULL)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL argument or invalid config
/// - EIO: Tile convert failed
#[no_mangle]
pub unsafe extern "C" fn hal_image_processor_tile_one(
    processor: *mut HalImageProcessor,
    src: *const HalTensor,
    dst: *mut HalTensor,
    placement: *const HalTilePlacement,
    config: *const HalTilingConfig,
) -> c_int {
    check_null!(processor, src, dst, placement, config);
    let cfg: TilingConfig = (*config).into();
    let p: TilePlacement = (&*placement).into();
    match (*processor)
        .inner
        .tile_one(&(*src).inner, &mut (*dst).inner, &p, &cfg)
    {
        Ok(()) => 0,
        Err(e) => set_error(image_err_to_errno(&e)),
    }
}

/// Map an `edgefirst_image::Error` to a POSIX errno consistent with the rest of
/// the HAL C API: caller-fault config/shape -> EINVAL, capacity -> ENOSPC,
/// unsupported -> ENOTSUP, I/O -> EIO, otherwise ENOMEM.
fn image_err_to_errno(e: &edgefirst_image::Error) -> c_int {
    use edgefirst_image::Error as E;
    match e {
        E::CropInvalid(_) | E::InvalidShape(_) | E::NotAnImage | E::AliasedBuffers(_) => {
            libc::EINVAL
        }
        E::UnsupportedFormat(_) | E::NotSupported(_) | E::NotImplemented(_) | E::NoConverter => {
            libc::ENOTSUP
        }
        E::Io(io) => io.raw_os_error().unwrap_or(libc::EIO),
        E::Tensor(te) => match te {
            edgefirst_tensor::Error::InvalidArgument(_)
            | edgefirst_tensor::Error::InvalidShape(_)
            | edgefirst_tensor::Error::ShapeMismatch(_) => libc::EINVAL,
            edgefirst_tensor::Error::IoError(io) => io.raw_os_error().unwrap_or(libc::EIO),
            edgefirst_tensor::Error::NotImplemented(_) => libc::ENOTSUP,
            _ => libc::ENOMEM,
        },
        _ => libc::EIO,
    }
}

// ============================================================================
// Detection lift / merge
// ============================================================================

/// Lift tile-local normalized detections to full-frame pixel coordinates.
///
/// The input list is borrowed (cloned internally); the caller still owns it.
///
/// @param boxes Tile-local detection box list (NULL treated as empty)
/// @param placement Tile placement describing the tile geometry (non-NULL)
/// @return New detection box list in full-frame pixels (free with
///         `hal_detect_box_list_free()`), or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL placement
#[no_mangle]
pub unsafe extern "C" fn hal_lift_tile_boxes(
    boxes: *const HalDetectBoxList,
    placement: *const HalTilePlacement,
) -> *mut HalDetectBoxList {
    check_null_ret_null!(placement);
    let p: TilePlacement = (&*placement).into();
    let lifted = lift_tile_boxes(clone_boxes(boxes), &p);
    Box::into_raw(Box::new(HalDetectBoxList { boxes: lifted }))
}

/// Merge lifted full-frame detections across tiles (greedy non-max merge).
///
/// The input list is borrowed (cloned internally); the caller still owns it.
///
/// @param boxes Lifted full-frame detection box list (NULL treated as empty)
/// @param config Merge configuration (non-NULL; seed with
///        `hal_merge_config_default()`)
/// @return New merged detection box list (free with
///         `hal_detect_box_list_free()`), or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL config
#[no_mangle]
pub unsafe extern "C" fn hal_merge_tiled_detections(
    boxes: *const HalDetectBoxList,
    config: *const HalMergeConfig,
) -> *mut HalDetectBoxList {
    check_null_ret_null!(config);
    let cfg: MergeConfig = (*config).into();
    let merged = merge_tiled_detections(clone_boxes(boxes), &cfg);
    Box::into_raw(Box::new(HalDetectBoxList { boxes: merged }))
}

// ============================================================================
// Streaming accumulator
// ============================================================================

/// Create a streaming accumulator for one frame's tiled detections.
///
/// @param frame_w Full-frame width in pixels (for `_finalize_normalized`)
/// @param frame_h Full-frame height in pixels (for `_finalize_normalized`)
/// @param tiles_total Total tiles for this frame (the fan-in fence)
/// @param config Merge configuration (non-NULL; seed with
///        `hal_merge_config_default()`)
/// @param est_per_tile Estimated detections per tile (pre-reserves the buffer)
/// @return New accumulator handle (free with
///         `hal_tiled_frame_accumulator_free()` or consume via a finalize
///         call), or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL config
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_new(
    frame_w: f32,
    frame_h: f32,
    tiles_total: size_t,
    config: *const HalMergeConfig,
    est_per_tile: size_t,
) -> *mut HalTiledFrameAccumulator {
    check_null_ret_null!(config);
    let cfg: MergeConfig = (*config).into();
    let inner = TiledFrameAccumulator::new((frame_w, frame_h), tiles_total, cfg, est_per_tile);
    Box::into_raw(Box::new(HalTiledFrameAccumulator { inner }))
}

/// Push one tile's per-tile-decoded boxes into the accumulator.
///
/// Lifts the boxes to full-frame pixels and appends them. Idempotent per
/// `placement.index`. The input list is borrowed (cloned internally).
///
/// @param acc Accumulator handle
/// @param boxes Tile-local detection box list (NULL treated as empty)
/// @param placement Tile placement for this tile (non-NULL)
/// @return 1 if the tile was newly accepted, 0 if it was a
///         duplicate/out-of-range, -1 on error (errno set)
/// @par Errors (errno):
/// - EINVAL: NULL acc or placement
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_push_tile(
    acc: *mut HalTiledFrameAccumulator,
    boxes: *const HalDetectBoxList,
    placement: *const HalTilePlacement,
) -> c_int {
    check_null!(acc, placement);
    let p: TilePlacement = (&*placement).into();
    let accepted = (*acc).inner.push_tile(clone_boxes(boxes), &p);
    c_int::from(accepted)
}

/// Whether every tile of the frame has been pushed.
///
/// @param acc Accumulator handle
/// @return true if complete, false otherwise (or if acc is NULL)
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_is_complete(
    acc: *const HalTiledFrameAccumulator,
) -> bool {
    if acc.is_null() {
        return false;
    }
    (*acc).inner.is_complete()
}

/// Number of tiles still outstanding.
///
/// @param acc Accumulator handle
/// @return Tiles remaining, or 0 if acc is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_remaining(
    acc: *const HalTiledFrameAccumulator,
) -> size_t {
    if acc.is_null() {
        return 0;
    }
    (*acc).inner.remaining()
}

/// Finalize the accumulator into merged full-frame **pixel** detections.
///
/// **Consumes** the accumulator: the handle is freed internally and must not be
/// used or freed again after this call.
///
/// @param acc Accumulator handle (consumed)
/// @return Merged detection box list in full-frame pixels (free with
///         `hal_detect_box_list_free()`), or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL acc
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_finalize(
    acc: *mut HalTiledFrameAccumulator,
) -> *mut HalDetectBoxList {
    check_null_ret_null!(acc);
    let owned = Box::from_raw(acc);
    let merged = owned.inner.finalize();
    Box::into_raw(Box::new(HalDetectBoxList { boxes: merged }))
}

/// Finalize the accumulator into merged detections renormalized to `[0,1]` by
/// `frame_dims` (the non-tiled normalized-detection contract, e.g. for the
/// tracker).
///
/// **Consumes** the accumulator: the handle is freed internally and must not be
/// used or freed again after this call.
///
/// @param acc Accumulator handle (consumed)
/// @return Merged detection box list in normalized `[0,1]` coordinates (free
///         with `hal_detect_box_list_free()`), or NULL on error
/// @par Errors (errno):
/// - EINVAL: NULL acc
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_finalize_normalized(
    acc: *mut HalTiledFrameAccumulator,
) -> *mut HalDetectBoxList {
    check_null_ret_null!(acc);
    let owned = Box::from_raw(acc);
    let merged = owned.inner.finalize_normalized();
    Box::into_raw(Box::new(HalDetectBoxList { boxes: merged }))
}

/// Free an accumulator without finalizing (the abandon path). No-op if NULL.
///
/// Do **not** call this after a finalize call — that would double-free.
///
/// @param acc Accumulator handle to free (can be NULL)
#[no_mangle]
pub unsafe extern "C" fn hal_tiled_frame_accumulator_free(acc: *mut HalTiledFrameAccumulator) {
    if !acc.is_null() {
        drop(Box::from_raw(acc));
    }
}

// ============================================================================
// Tile spec list accessors
// ============================================================================

/// Number of tile specs in a list.
///
/// @param list Tile spec list handle
/// @return Count, or 0 if list is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tile_spec_list_len(list: *const HalTileSpecList) -> size_t {
    if list.is_null() {
        return 0;
    }
    (*list).specs.len()
}

/// Get a tile spec from a list by index.
///
/// @param list Tile spec list handle
/// @param index Index (0-based)
/// @param out_spec Output parameter for the tile spec
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL list/out_spec, or index out of bounds
#[no_mangle]
pub unsafe extern "C" fn hal_tile_spec_list_get(
    list: *const HalTileSpecList,
    index: size_t,
    out_spec: *mut HalTileSpec,
) -> c_int {
    check_null!(list, out_spec);
    if index >= (*list).specs.len() {
        return set_error(libc::EINVAL);
    }
    *out_spec = HalTileSpec::from(&(&(*list).specs)[index]);
    0
}

/// Free a tile spec list (can be NULL, no-op).
///
/// @param list Tile spec list handle to free
#[no_mangle]
pub unsafe extern "C" fn hal_tile_spec_list_free(list: *mut HalTileSpecList) {
    if !list.is_null() {
        drop(Box::from_raw(list));
    }
}

// ============================================================================
// Tile placement list accessors
// ============================================================================

/// Number of tile placements in a list.
///
/// @param list Tile placement list handle
/// @return Count, or 0 if list is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_tile_placement_list_len(list: *const HalTilePlacementList) -> size_t {
    if list.is_null() {
        return 0;
    }
    (*list).placements.len()
}

/// Get a tile placement from a list by index.
///
/// @param list Tile placement list handle
/// @param index Index (0-based)
/// @param out_placement Output parameter for the tile placement
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: NULL list/out_placement, or index out of bounds
#[no_mangle]
pub unsafe extern "C" fn hal_tile_placement_list_get(
    list: *const HalTilePlacementList,
    index: size_t,
    out_placement: *mut HalTilePlacement,
) -> c_int {
    check_null!(list, out_placement);
    if index >= (*list).placements.len() {
        return set_error(libc::EINVAL);
    }
    *out_placement = HalTilePlacement::from(&(&(*list).placements)[index]);
    0
}

/// Free a tile placement list (can be NULL, no-op).
///
/// @param list Tile placement list handle to free
#[no_mangle]
pub unsafe extern "C" fn hal_tile_placement_list_free(list: *mut HalTilePlacementList) {
    if !list.is_null() {
        drop(Box::from_raw(list));
    }
}
