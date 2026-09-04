// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tiled inference: merging detections from many tiles back into one frame.
//!
//! Here rather than in a sibling of its own because the types live in
//! `edgefirst-decoder` and the work is detection merging. The four tiling
//! entry points that need an `ImageProcessor` — planning a grid, allocating a
//! tile batch, and blitting into it — belong to `libedgefirst-image` instead,
//! since that is where the processor is.

use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder::tiling::{
    lift_tile_boxes, merge_tiled_detections, MatchMetric, MergeConfig, MergeMode,
    TiledFrameAccumulator,
};
use edgefirst_decoder_abi::{EfMergeConfig, EfTilePlacement, TilePlacement};
use edgefirst_tensor::{BoundingBox, DetectBox};

use crate::decode::{EfDetectBox, EfDetectBoxList};

fn metric_from(v: u32) -> Option<MatchMetric> {
    match v {
        0 => Some(MatchMetric::Iou),
        1 => Some(MatchMetric::Ios),
        _ => None,
    }
}

fn metric_to(m: MatchMetric) -> u32 {
    match m {
        MatchMetric::Iou => 0,
        MatchMetric::Ios => 1,
    }
}

fn mode_from(v: u32) -> Option<MergeMode> {
    match v {
        0 => Some(MergeMode::KeepBest),
        1 => Some(MergeMode::Union),
        _ => None,
    }
}

fn mode_to(m: MergeMode) -> u32 {
    match m {
        MergeMode::KeepBest => 0,
        MergeMode::Union => 1,
    }
}

fn merge_config_from(c: &EfMergeConfig) -> Option<MergeConfig> {
    Some(MergeConfig {
        metric: metric_from(c.metric)?,
        threshold: c.threshold,
        class_agnostic: c.class_agnostic != 0,
        max_det: c.max_det,
        score_threshold: c.score_threshold,
        mode: mode_from(c.mode)?,
    })
}

fn to_rust(b: &EfDetectBox) -> DetectBox {
    DetectBox {
        bbox: BoundingBox {
            xmin: b.xmin,
            ymin: b.ymin,
            xmax: b.xmax,
            ymax: b.ymax,
        },
        score: b.score,
        label: b.label as usize,
    }
}

fn to_c(d: &DetectBox) -> EfDetectBox {
    EfDetectBox {
        xmin: d.bbox.xmin,
        ymin: d.bbox.ymin,
        xmax: d.bbox.xmax,
        ymax: d.bbox.ymax,
        score: d.score,
        label: d.label as u32,
    }
}

/// Fill `out` with the library's default merge configuration.
///
/// The default `mode` is `0` (keep-best): the highest-scoring box of each
/// matched group is kept and the boxes it matched are dropped. Set `mode = 1`
/// for the enclosing-union merge, which measured about 0.05 AP50 worse on
/// every frame of the Ocean Cleanup ADIS 4K validation (TOP2-836).
///
/// # Safety
/// `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_merge_config_default(out: *mut EfMergeConfig) -> c_int {
    unsafe {
        if out.is_null() {
            return libc::EINVAL;
        }
        catch_unwind(AssertUnwindSafe(|| {
            let d = MergeConfig::default();
            *out = EfMergeConfig {
                metric: metric_to(d.metric),
                threshold: d.threshold,
                class_agnostic: i32::from(d.class_agnostic),
                max_det: d.max_det,
                score_threshold: d.score_threshold,
                mode: mode_to(d.mode),
            };
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Accumulates detections from every tile of one frame.
pub struct EfTiledFrameAccumulator {
    inner: Option<TiledFrameAccumulator>,
}

/// Create an accumulator for a frame of `tiles_total` tiles, merging as
/// `cfg` says — including its `mode`.
///
/// @return the accumulator, or `NULL` for a null `cfg`, zero tiles, or a
///         `metric`/`mode` value this library does not know.
///
/// # Safety
/// `cfg` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_new(
    frame_width: f32,
    frame_height: f32,
    tiles_total: usize,
    cfg: *const EfMergeConfig,
    est_per_tile: usize,
) -> *mut EfTiledFrameAccumulator {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if cfg.is_null() || tiles_total == 0 {
                return std::ptr::null_mut();
            }
            let Some(c) = merge_config_from(&*cfg) else {
                return std::ptr::null_mut();
            };
            Box::into_raw(Box::new(EfTiledFrameAccumulator {
                inner: Some(TiledFrameAccumulator::new(
                    (frame_width, frame_height),
                    tiles_total,
                    c,
                    est_per_tile,
                )),
            }))
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Free an accumulator. Freeing `NULL` is a no-op.
///
/// # Safety
/// `a` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_free(a: *mut EfTiledFrameAccumulator) {
    unsafe {
        if a.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(a))));
    }
}

/// Add one tile's detections.
///
/// **Idempotent per tile index.** A duplicate index, an out-of-range one, or a
/// placement from a different plan (its `count` disagreeing with this
/// accumulator's tile total) is ignored and its detections dropped. That is
/// what makes out-of-order *and* at-least-once delivery converge to the same
/// frame — a retried tile does not double-count, and a tile from another
/// frame cannot corrupt this one's fan-in.
///
/// Use [`ef_tiled_frame_accumulator_is_complete`] to test for completion; this
/// return value answers a different question.
///
/// @return 1 when the tile was newly accepted, 0 when it was ignored as a
///         duplicate or foreign placement, `-1` on a bad argument or after
///         finalize.
///
/// # Safety
/// `boxes` must point to `count` elements; `placement` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_push_tile(
    a: *mut EfTiledFrameAccumulator,
    boxes: *const EfDetectBox,
    count: usize,
    placement: *const EfTilePlacement,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if a.is_null() || placement.is_null() || (boxes.is_null() && count != 0) {
                return -1;
            }
            let Some(acc) = (*a).inner.as_mut() else {
                // Already finalized; pushing more would silently produce a result
                // the caller can never retrieve.
                return -1;
            };
            let tile: Vec<DetectBox> = if count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(boxes, count)
                    .iter()
                    .map(to_rust)
                    .collect()
            };
            let p = TilePlacement::from(&*placement);
            i32::from(acc.push_tile(tile, &p))
        }))
        .unwrap_or(-1)
    }
}

/// Whether every tile has been seen.
///
/// # Safety
/// `a` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_is_complete(
    a: *const EfTiledFrameAccumulator,
) -> c_int {
    unsafe {
        if a.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| {
            (*a).inner
                .as_ref()
                .map_or(0, |i| i32::from(i.is_complete()))
        }))
        .unwrap_or(0)
    }
}

/// How many tiles are still outstanding.
///
/// # Safety
/// `a` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_remaining(
    a: *const EfTiledFrameAccumulator,
) -> usize {
    unsafe {
        if a.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| {
            (*a).inner.as_ref().map_or(0, |i| i.remaining())
        }))
        .unwrap_or(0)
    }
}

/// Merge every pushed tile into one detection list.
///
/// Consumes the accumulator's contents: a second call returns `NULL`, because
/// merging is destructive and returning an empty list would be
/// indistinguishable from a frame that genuinely found nothing.
///
/// `normalized` non-zero returns frame-normalized coordinates.
///
/// @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
///
/// # Safety
/// `a` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_tiled_frame_accumulator_finalize(
    a: *mut EfTiledFrameAccumulator,
    normalized: c_int,
) -> *mut crate::decode::EfDetectBoxList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if a.is_null() {
                return std::ptr::null_mut();
            }
            let Some(acc) = (*a).inner.take() else {
                return std::ptr::null_mut();
            };
            let merged = if normalized != 0 {
                acc.finalize_normalized()
            } else {
                acc.finalize()
            };
            crate::decode::box_list_from(merged.iter().map(to_c).collect())
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Lift a tile's detections into frame coordinates.
///
/// @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
///
/// # Safety
/// `boxes` must point to `count` elements; `placement` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_lift_tile_boxes(
    boxes: *const EfDetectBox,
    count: usize,
    placement: *const EfTilePlacement,
) -> *mut crate::decode::EfDetectBoxList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if placement.is_null() || (boxes.is_null() && count != 0) {
                return std::ptr::null_mut();
            }
            let input: Vec<DetectBox> = if count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(boxes, count)
                    .iter()
                    .map(to_rust)
                    .collect()
            };
            let p = TilePlacement::from(&*placement);
            crate::decode::box_list_from(lift_tile_boxes(input, &p).iter().map(to_c).collect())
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Merge overlapping detections that already share one coordinate space,
/// merging as `cfg` says — including its `mode`.
///
/// @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`
///         for a null `cfg`, a null `boxes` with a non-zero `count`, or a
///         `metric`/`mode` value this library does not know.
///
/// # Safety
/// `boxes` must point to `count` elements; `cfg` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_merge_tiled_detections(
    boxes: *const EfDetectBox,
    count: usize,
    cfg: *const EfMergeConfig,
) -> *mut EfDetectBoxList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if cfg.is_null() || (boxes.is_null() && count != 0) {
                return std::ptr::null_mut();
            }
            let Some(c) = merge_config_from(&*cfg) else {
                return std::ptr::null_mut();
            };
            let input: Vec<DetectBox> = if count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(boxes, count)
                    .iter()
                    .map(to_rust)
                    .collect()
            };
            crate::decode::box_list_from(
                merge_tiled_detections(input, &c).iter().map(to_c).collect(),
            )
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{ef_detect_box_list_free, ef_detect_box_list_get, ef_detect_box_list_len};

    /// The library default, seeded from a struct whose every field is wrong
    /// (`mode: 99` is not a value this library knows) so a `default()` that
    /// skipped a field would fail the assertions below rather than pass by
    /// accident.
    fn cfg() -> EfMergeConfig {
        let mut c = EfMergeConfig {
            metric: 0,
            threshold: 0.0,
            class_agnostic: 0,
            max_det: 0,
            score_threshold: 0.0,
            mode: 99,
        };
        unsafe { ef_merge_config_default(&mut c) };
        c
    }

    fn ebox(b: [f32; 4], score: f32, label: u32) -> EfDetectBox {
        EfDetectBox {
            xmin: b[0],
            ymin: b[1],
            xmax: b[2],
            ymax: b[3],
            score,
            label,
        }
    }

    /// Copy the boxes out of a list (and free it) so a test can assert on
    /// coordinates without holding a borrowed pointer.
    unsafe fn drain(l: *mut crate::decode::EfDetectBoxList) -> Vec<EfDetectBox> {
        unsafe {
            assert!(!l.is_null(), "merge returned NULL");
            let n = ef_detect_box_list_len(l);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut b = EfDetectBox::default();
                assert_eq!(ef_detect_box_list_get(l, i, &mut b), 0);
                out.push(b);
            }
            ef_detect_box_list_free(l);
            out
        }
    }

    /// The partial-overlap pair the merge-mode tests share: `b` matches `a`
    /// at IoS exactly 0.5 and extends past it, so the two modes give
    /// visibly different boxes.
    fn overlapping_pair() -> [EfDetectBox; 2] {
        [
            ebox([0.0, 0.0, 100.0, 100.0], 0.9, 0),
            ebox([50.0, 0.0, 150.0, 100.0], 0.8, 0),
        ]
    }

    fn placement(index: usize, count: usize) -> EfTilePlacement {
        EfTilePlacement {
            index,
            count,
            origin_x: 0.0,
            origin_y: 0.0,
            crop_width: 1.0,
            crop_height: 1.0,
            has_letterbox: 0,
            letterbox: [0.0; 4],
            frame_width: 640.0,
            frame_height: 480.0,
        }
    }

    #[test]
    fn the_default_merge_metric_is_intersection_over_smaller() {
        // Not cosmetic: an object split across a tile seam has LOW IoU with
        // its own fragment, so an IoU default would keep both halves.
        let c = cfg();
        assert_eq!(c.metric, 1, "default must be IoS, not IoU");
    }

    #[test]
    fn the_default_merge_mode_is_keep_best() {
        // The Ocean Cleanup ADIS 4K study (TOP2-836) put the enclosing union
        // at about -0.05 AP50 on every frame; keep-best is the default and
        // `0`, though a zero-initialised struct still is not the library
        // default -- `metric` would be IoU rather than the IoS default.
        let c = cfg();
        assert_eq!(c.mode, 0, "default must be keep-best (0), not union (1)");
        assert_eq!(c.metric, 1, "IoS stays the default metric");
        assert_eq!(c.threshold, 0.5);
        assert_eq!(c.max_det, 300);
        assert!(unsafe { ef_merge_config_default(std::ptr::null_mut()) } == libc::EINVAL);
    }

    #[test]
    fn keep_best_keeps_the_base_box_and_union_encloses_it() {
        unsafe {
            let boxes = overlapping_pair();

            let mut keep = cfg();
            keep.mode = 0;
            let out = drain(ef_merge_tiled_detections(boxes.as_ptr(), 2, &keep));
            assert_eq!(out.len(), 1);
            assert_eq!(
                [out[0].xmin, out[0].ymin, out[0].xmax, out[0].ymax],
                [0.0, 0.0, 100.0, 100.0],
                "keep-best must leave the base box exactly as decoded"
            );
            assert_eq!(out[0].score, 0.9);

            let mut union = cfg();
            union.mode = 1;
            let out = drain(ef_merge_tiled_detections(boxes.as_ptr(), 2, &union));
            assert_eq!(out.len(), 1);
            assert_eq!(
                [out[0].xmin, out[0].ymin, out[0].xmax, out[0].ymax],
                [0.0, 0.0, 150.0, 100.0],
                "union must grow the base box to the group's enclosing union"
            );
            assert_eq!(out[0].score, 0.9);
        }
    }

    #[test]
    fn a_default_config_merges_keep_best_through_both_entry_points() {
        // The standalone merge and the accumulator must agree: a config
        // straight from `ef_merge_config_default` keeps the base box rather
        // than growing it to the union these inputs used to produce.
        unsafe {
            let boxes = overlapping_pair();
            let c = cfg();
            let out = drain(ef_merge_tiled_detections(boxes.as_ptr(), 2, &c));
            assert_eq!(out.len(), 1);
            assert_eq!(
                [out[0].xmin, out[0].ymin, out[0].xmax, out[0].ymax],
                [0.0, 0.0, 100.0, 100.0]
            );

            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 4);
            assert!(!a.is_null());
            let p = placement(0, 1);
            // Whole-frame placement with a unit crop: boxes lift unchanged.
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, boxes.as_ptr(), 2, &p),
                1
            );
            let out = drain(ef_tiled_frame_accumulator_finalize(a, 0));
            ef_tiled_frame_accumulator_free(a);
            assert_eq!(out.len(), 1);
            assert_eq!(
                [out[0].xmin, out[0].ymin, out[0].xmax, out[0].ymax],
                [0.0, 0.0, 100.0, 100.0]
            );
        }
    }

    #[test]
    fn an_accumulator_honours_an_explicit_union_mode() {
        unsafe {
            let boxes = overlapping_pair();
            let p = placement(0, 1);

            let mut union = cfg();
            union.mode = 1;
            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &union, 4);
            assert!(!a.is_null());
            assert_eq!(ef_tiled_frame_accumulator_remaining(a), 1);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, boxes.as_ptr(), 2, &p),
                1
            );
            assert_eq!(ef_tiled_frame_accumulator_is_complete(a), 1);
            let out = drain(ef_tiled_frame_accumulator_finalize(a, 0));
            ef_tiled_frame_accumulator_free(a);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].xmax, 150.0, "union mode reaches the accumulator");

            assert!(ef_tiled_frame_accumulator_new(640.0, 480.0, 0, &union, 4).is_null());
            assert!(ef_tiled_frame_accumulator_new(640.0, 480.0, 1, std::ptr::null(), 4).is_null());
        }
    }

    #[test]
    fn an_unknown_mode_is_refused() {
        unsafe {
            let mut c = cfg();
            c.mode = 99;
            assert!(ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1).is_null());
            assert!(ef_merge_tiled_detections(std::ptr::null(), 0, &c).is_null());

            let mut c = cfg();
            c.metric = 99;
            assert!(ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1).is_null());
            assert!(ef_merge_tiled_detections(std::ptr::null(), 0, &c).is_null());

            let c = cfg();
            assert!(ef_merge_tiled_detections(std::ptr::null(), 0, std::ptr::null()).is_null());
            assert!(ef_merge_tiled_detections(std::ptr::null(), 3, &c).is_null());
            let empty = ef_merge_tiled_detections(std::ptr::null(), 0, &c);
            assert!(!empty.is_null());
            assert_eq!(ef_detect_box_list_len(empty), 0);
            ef_detect_box_list_free(empty);
        }
    }

    #[test]
    fn an_accumulator_reports_progress_and_completes() {
        unsafe {
            let c = cfg();
            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 2, &c, 4);
            assert!(!a.is_null());
            assert_eq!(ef_tiled_frame_accumulator_remaining(a), 2);
            assert_eq!(ef_tiled_frame_accumulator_is_complete(a), 0);

            let b = [EfDetectBox {
                xmin: 0.1,
                ymin: 0.1,
                xmax: 0.2,
                ymax: 0.2,
                score: 0.9,
                label: 0,
            }];
            let p0 = placement(0, 2);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, b.as_ptr(), 1, &p0),
                1,
                "a first, in-range tile is newly accepted"
            );
            assert_eq!(
                ef_tiled_frame_accumulator_is_complete(a),
                0,
                "one of two tiles is not a complete frame"
            );
            assert_eq!(ef_tiled_frame_accumulator_remaining(a), 1);

            let p1 = placement(1, 2);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, b.as_ptr(), 1, &p1),
                1,
                "the second tile is also newly accepted"
            );
            assert_eq!(ef_tiled_frame_accumulator_is_complete(a), 1);

            let merged = ef_tiled_frame_accumulator_finalize(a, 0);
            assert!(!merged.is_null());
            ef_detect_box_list_free(merged);
            ef_tiled_frame_accumulator_free(a);
        }
    }

    #[test]
    fn a_retried_or_foreign_tile_is_ignored_rather_than_double_counted() {
        // The property that makes at-least-once delivery safe: a retried tile
        // must not advance completion or contribute its boxes twice, and a
        // tile from a different plan must not corrupt this frame's fan-in.
        unsafe {
            let c = cfg();
            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 2, &c, 4);
            let p0 = placement(0, 2);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &p0),
                1
            );
            assert_eq!(ef_tiled_frame_accumulator_remaining(a), 1);

            // Same index again -- a retry.
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &p0),
                0,
                "a duplicate index must be ignored"
            );
            assert_eq!(
                ef_tiled_frame_accumulator_remaining(a),
                1,
                "a retry must not advance completion"
            );

            // A placement from a plan with a different tile count.
            let foreign = placement(1, 7);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &foreign),
                0,
                "a placement from another plan must be rejected"
            );
            assert_eq!(ef_tiled_frame_accumulator_remaining(a), 1);

            // Out of range.
            let oob = placement(5, 2);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &oob),
                0
            );
            ef_tiled_frame_accumulator_free(a);
        }
    }

    #[test]
    fn finalize_is_destructive_and_says_so() {
        // Returning an empty list on the second call would be
        // indistinguishable from a frame that genuinely found nothing.
        unsafe {
            let c = cfg();
            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1);
            let p = placement(0, 1);
            ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &p);
            let first = ef_tiled_frame_accumulator_finalize(a, 0);
            assert!(!first.is_null());
            ef_detect_box_list_free(first);

            assert!(
                ef_tiled_frame_accumulator_finalize(a, 0).is_null(),
                "a second finalize must fail, not return an empty result"
            );
            let p2 = placement(0, 1);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &p2),
                -1,
                "pushing after finalize must fail"
            );
            ef_tiled_frame_accumulator_free(a);
        }
    }

    #[test]
    fn an_empty_tile_is_valid_but_a_null_array_with_a_count_is_not() {
        unsafe {
            let c = cfg();
            let a = ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1);
            let p = placement(0, 1);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a, std::ptr::null(), 0, &p),
                1,
                "a tile with no detections is a valid tile"
            );
            let a2 = ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1);
            assert_eq!(
                ef_tiled_frame_accumulator_push_tile(a2, std::ptr::null(), 5, &p),
                -1
            );
            ef_tiled_frame_accumulator_free(a);
            ef_tiled_frame_accumulator_free(a2);
        }
    }

    #[test]
    fn an_unknown_metric_is_refused() {
        unsafe {
            let mut c = cfg();
            c.metric = 99;
            assert!(ef_tiled_frame_accumulator_new(640.0, 480.0, 1, &c, 1).is_null());
            assert!(ef_merge_tiled_detections(std::ptr::null(), 0, &c).is_null());
        }
    }

    #[test]
    fn lift_and_merge_accept_an_empty_input() {
        unsafe {
            let c = cfg();
            let p = placement(0, 1);
            let lifted = ef_lift_tile_boxes(std::ptr::null(), 0, &p);
            assert!(!lifted.is_null());
            assert_eq!(ef_detect_box_list_len(lifted), 0);
            ef_detect_box_list_free(lifted);

            let merged = ef_merge_tiled_detections(std::ptr::null(), 0, &c);
            assert!(!merged.is_null());
            ef_detect_box_list_free(merged);

            assert!(ef_lift_tile_boxes(std::ptr::null(), 0, std::ptr::null()).is_null());
        }
    }

    #[test]
    fn null_accumulator_arguments_are_errors_not_crashes() {
        unsafe {
            let c = cfg();
            assert!(ef_tiled_frame_accumulator_new(1.0, 1.0, 0, &c, 1).is_null());
            assert!(ef_tiled_frame_accumulator_new(1.0, 1.0, 1, std::ptr::null(), 1).is_null());
            assert_eq!(ef_tiled_frame_accumulator_remaining(std::ptr::null()), 0);
            assert_eq!(ef_tiled_frame_accumulator_is_complete(std::ptr::null()), 0);
            assert!(ef_tiled_frame_accumulator_finalize(std::ptr::null_mut(), 0).is_null());
            ef_tiled_frame_accumulator_free(std::ptr::null_mut());
        }
    }
}
