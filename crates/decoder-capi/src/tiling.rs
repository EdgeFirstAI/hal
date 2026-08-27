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
    lift_tile_boxes, merge_tiled_detections, MatchMetric, MergeConfig, TiledFrameAccumulator,
};
use edgefirst_decoder_abi::{EfMergeConfig, EfTilePlacement, TilePlacement};
use edgefirst_tensor::{BoundingBox, DetectBox};

use crate::decode::EfDetectBox;

fn merge_config_from(c: &EfMergeConfig) -> Option<MergeConfig> {
    Some(MergeConfig {
        metric: match c.metric {
            0 => MatchMetric::Iou,
            1 => MatchMetric::Ios,
            _ => return None,
        },
        threshold: c.threshold,
        class_agnostic: c.class_agnostic != 0,
        max_det: c.max_det,
        score_threshold: c.score_threshold,
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
                metric: match d.metric {
                    MatchMetric::Iou => 0,
                    MatchMetric::Ios => 1,
                },
                threshold: d.threshold,
                class_agnostic: i32::from(d.class_agnostic),
                max_det: d.max_det,
                score_threshold: d.score_threshold,
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

/// Create an accumulator for a frame of `tiles_total` tiles.
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

/// Merge overlapping detections that already share one coordinate space.
///
/// @return a list the caller frees with `ef_detect_box_list_free`, or `NULL`.
///
/// # Safety
/// `boxes` must point to `count` elements; `cfg` must be valid.
#[no_mangle]
pub unsafe extern "C" fn ef_merge_tiled_detections(
    boxes: *const EfDetectBox,
    count: usize,
    cfg: *const EfMergeConfig,
) -> *mut crate::decode::EfDetectBoxList {
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
    use crate::decode::{ef_detect_box_list_free, ef_detect_box_list_len};

    fn cfg() -> EfMergeConfig {
        let mut c = EfMergeConfig {
            metric: 0,
            threshold: 0.0,
            class_agnostic: 0,
            max_det: 0,
            score_threshold: 0.0,
        };
        unsafe { ef_merge_config_default(&mut c) };
        c
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
        assert!(unsafe { ef_merge_config_default(std::ptr::null_mut()) } == libc::EINVAL);
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
