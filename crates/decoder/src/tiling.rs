// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! SAHI-style tiled-inference postprocessing: lift per-tile detections to
//! full-frame coordinates and merge them across tiles.
//!
//! A high-resolution frame is covered by a uniform overlapping grid of tiles
//! (geometry lives in the `edgefirst-image` crate). Each tile is run through
//! the small tile-input model and decoded independently to normalized `[0,1]`
//! detections over the model input. This module lifts those to full-frame
//! pixels ([`lift_tile_boxes`]) and merges duplicates at tile seams
//! ([`merge_tiled_detections`]) using **GREEDYNMM** with the **IOS**
//! (intersection-over-smaller) match metric. [`TiledFrameAccumulator`] is a
//! streaming collector so a pipelined runtime can push each tile's detections
//! as inference completes and finalize once the frame's last tile arrives.
//!
//! The merge reproduces ModelPack's reference runtime
//! (`metrics/tiled.py::merge_tiled_detections`) numerically. IOS matters
//! because an object split across a tile overlap appears as two partial boxes
//! whose IoU is low but whose IoS is high, so IoS merges them where IoU leaves
//! duplicates.
//!
//! # Per-tile decode guidance (affects mAP)
//!
//! Run the per-tile [`crate::Decoder`] with a **low score threshold** (e.g.
//! 0.05) and **class-aware** NMS, and a modest per-tile `max_det`. The merge's
//! own [`MergeConfig::score_threshold`] defaults to `0.0` precisely because
//! per-tile decode is the real flood control — a high per-tile threshold
//! discards true-positive fragments before the merge can join them, collapsing
//! the recall the IOS design buys. Final score gating belongs in
//! [`MergeConfig::score_threshold`].
//!
//! # Known limitations
//!
//! - **Objects larger than one tile** cannot be reconstructed: every tile sees
//!   only a fragment, and with no whole-object box to anchor the union the
//!   fragments may not mutually pass the IOS threshold. Choose a tile size that
//!   exceeds the largest expected object, or add the optional full-frame
//!   downscaled pass (push it as one extra tile into the accumulator).
//!
//! # Reference implementations
//!
//! - **Grid spacing (EvenDist):** the canonical authority is HAL's own
//!   `edgefirst_image::tile_grid` (ported from the adis-uav-model `sahi()`
//!   function). `overlap_ratio` is a *minimum*; realized overlap is never
//!   rounded below it. ModelPack's validator is slated to adopt this same grid.
//! - **Merge:** ModelPack `metrics/tiled.py::merge_tiled_detections` — mirrored
//!   here numerically. The only deliberate difference is tie-breaking on exactly
//!   equal scores (ascending original index here vs NumPy's unstable `argsort`),
//!   which makes the streaming accumulator order-independent; results are
//!   identical on non-degenerate inputs.

use crate::float::{box_area, intersection_area, ios_value, iou_value};
use crate::{BoundingBox, DetectBox};

/// Overlap metric used by the tiled-detection merge to decide whether two boxes
/// belong to the same object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchMetric {
    /// Intersection-over-Union (standard NMS metric).
    Iou,
    /// Intersection-over-Smaller (default): `inter / min(area_a, area_b)`. A
    /// seam-split object has low IoU but high IoS, so IoS merges the fragment.
    #[default]
    Ios,
}

impl MatchMetric {
    /// Metric value in `[0, 1]` for two boxes.
    #[inline]
    pub fn value(self, a: &BoundingBox, b: &BoundingBox) -> f32 {
        match self {
            MatchMetric::Iou => iou_value(a, b),
            MatchMetric::Ios => ios_value(a, b),
        }
    }
}

/// How one tile was cut from the full frame and fed to the model. Produced by
/// the input side (the `edgefirst-image` tiling API), consumed by
/// [`lift_tile_boxes`]. All fields are native full-frame **pixels** except
/// `letterbox`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilePlacement {
    /// Tile index within the frame grid, `0..count`.
    pub index: usize,
    /// Total tiles for this frame (the streaming fan-in fence).
    pub count: usize,
    /// Native crop origin `(ox, oy)` in full-frame pixels.
    pub origin: (f32, f32),
    /// Native crop size `(cw, ch)` in full-frame pixels. Equals the tile size
    /// for the full-size tiles the EvenDist grid produces.
    pub crop_size: (f32, f32),
    /// Normalized letterbox content bounds `[lx0, ly0, lx1, ly1]` on the model
    /// input, or `None` when the crop was stretched to fill it (the hot path).
    pub letterbox: Option<[f32; 4]>,
    /// Full-frame dimensions `(frame_w, frame_h)` in pixels.
    pub frame_dims: (f32, f32),
}

/// Configuration for the tiled-detection merge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeConfig {
    /// Overlap metric (default [`MatchMetric::Ios`]).
    pub metric: MatchMetric,
    /// Merge two boxes when `metric.value(a, b) >= threshold` (default 0.5).
    pub threshold: f32,
    /// Merge across classes when true (default false).
    pub class_agnostic: bool,
    /// Cap on returned detections after the merge (default 300).
    pub max_det: usize,
    /// Drop merged groups whose max score is below this (default 0.0 = keep
    /// all; per-tile decode is the real flood control).
    pub score_threshold: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            metric: MatchMetric::Ios,
            threshold: 0.5,
            class_agnostic: false,
            max_det: 300,
            score_threshold: 0.0,
        }
    }
}

/// Invert a letterbox: map a [`BoundingBox`] normalized over the model input
/// back to normalized-over-the-crop, given the content bounds
/// `[lx0, ly0, lx1, ly1]`. The box is canonicalised first, a degenerate
/// (zero-span) letterbox axis maps with unit scale (no divide-by-zero), and the
/// result is clamped to `[0, 1]`.
///
/// This is the single home for the inverse-letterbox transform;
/// `edgefirst_image::unletter_bbox` is a thin wrapper around it (the `image`
/// crate depends on `decoder`, so the shared math lives here, in the lower
/// crate).
#[must_use]
#[inline]
pub fn unletter_norm(b: BoundingBox, lb: [f32; 4]) -> BoundingBox {
    let b = b.to_canonical();
    let [lx0, ly0, lx1, ly1] = lb;
    let inv_w = if lx1 > lx0 { 1.0 / (lx1 - lx0) } else { 1.0 };
    let inv_h = if ly1 > ly0 { 1.0 / (ly1 - ly0) } else { 1.0 };
    BoundingBox {
        xmin: ((b.xmin - lx0) * inv_w).clamp(0.0, 1.0),
        ymin: ((b.ymin - ly0) * inv_h).clamp(0.0, 1.0),
        xmax: ((b.xmax - lx0) * inv_w).clamp(0.0, 1.0),
        ymax: ((b.ymax - ly0) * inv_h).clamp(0.0, 1.0),
    }
}

/// Lift tile-local **normalized** `[0,1]` xyxy detections (over the model
/// input) to full-frame **pixel** xyxy. Mirrors
/// `metrics/tiled.py::lift_tile_boxes`: optionally invert the letterbox, then
/// `full = origin + norm * crop_size`. Consumes and rewrites `boxes` in place.
///
/// # Examples
/// ```
/// use edgefirst_decoder::tiling::{lift_tile_boxes, TilePlacement};
/// use edgefirst_decoder::{BoundingBox, DetectBox};
///
/// let placement = TilePlacement {
///     index: 0, count: 1,
///     origin: (100.0, 200.0), crop_size: (640.0, 640.0),
///     letterbox: None, frame_dims: (3840.0, 2160.0),
/// };
/// let tile_local = DetectBox { bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0), score: 0.9, label: 0 };
/// let lifted = lift_tile_boxes(vec![tile_local], &placement);
/// assert_eq!(lifted[0].bbox, BoundingBox::new(100.0, 200.0, 740.0, 840.0));
/// ```
#[must_use]
pub fn lift_tile_boxes(mut boxes: Vec<DetectBox>, placement: &TilePlacement) -> Vec<DetectBox> {
    let (ox, oy) = placement.origin;
    let (cw, ch) = placement.crop_size;
    for d in &mut boxes {
        let n = match placement.letterbox {
            Some(lb) => unletter_norm(d.bbox, lb),
            None => d.bbox,
        };
        d.bbox = BoundingBox {
            xmin: ox + n.xmin * cw,
            ymin: oy + n.ymin * ch,
            xmax: ox + n.xmax * cw,
            ymax: oy + n.ymax * ch,
        };
    }
    boxes
}

/// Match metric using precomputed areas (avoids recomputing both operands'
/// areas on every pair in the O(N^2) merge). Equivalent to
/// [`MatchMetric::value`].
#[inline]
fn metric_value_with_areas(
    metric: MatchMetric,
    a: &BoundingBox,
    area_a: f32,
    b: &BoundingBox,
    area_b: f32,
) -> f32 {
    let inter = intersection_area(a, b);
    let denom = match metric {
        MatchMetric::Iou => area_a + area_b - inter,
        MatchMetric::Ios => area_a.min(area_b),
    };
    inter / denom.max(1e-9)
}

/// Greedy Non-Max **Merge** of lifted full-frame detections. Mirrors
/// `metrics/tiled.py::merge_tiled_detections`:
///
/// 1. Sort descending by score (ties broken by ascending original index so the
///    result is deterministic — this differs from NumPy's unstable `argsort`
///    only on exact ties).
/// 2. For each unused `base` in order, find later unused boxes (same class
///    unless `class_agnostic`) whose `metric.value(base, cand) >= threshold` —
///    matched against the **original** base box. Replace the group with its
///    **enclosing union** carrying the group's **max** score and the base's
///    label.
/// 3. Drop groups below `score_threshold` and truncate to `max_det`.
///
/// Operates in pixel space (the metric's `1e-9` epsilon is calibrated to pixel
/// areas).
///
/// # Examples
/// ```
/// use edgefirst_decoder::tiling::{merge_tiled_detections, MatchMetric, MergeConfig};
/// use edgefirst_decoder::{BoundingBox, DetectBox};
///
/// // A fragment (B) fully inside the full detection (A): IoS=1.0, IoU≈0.17.
/// let a = DetectBox { bbox: BoundingBox::new(100.0, 100.0, 400.0, 300.0), score: 0.9, label: 0 };
/// let b = DetectBox { bbox: BoundingBox::new(350.0, 100.0, 400.0, 300.0), score: 0.7, label: 0 };
///
/// // IOS merges the fragment into one box carrying the group's max score…
/// let ios = merge_tiled_detections(vec![a, b], &MergeConfig::default());
/// assert_eq!(ios.len(), 1);
/// assert_eq!(ios[0].score, 0.9);
///
/// // …while IOU leaves the two separate.
/// let cfg = MergeConfig { metric: MatchMetric::Iou, ..MergeConfig::default() };
/// assert_eq!(merge_tiled_detections(vec![a, b], &cfg).len(), 2);
/// ```
#[must_use]
pub fn merge_tiled_detections(dets: Vec<DetectBox>, cfg: &MergeConfig) -> Vec<DetectBox> {
    if dets.is_empty() {
        return dets;
    }

    // Descending score, ties by ascending original index (deterministic).
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&i, &j| dets[j].score.total_cmp(&dets[i].score).then(i.cmp(&j)));

    // Canonicalize once so degenerate (inverted) boxes have well-defined areas,
    // and precompute each box's area once (the O(N^2) loop below would otherwise
    // recompute both operands' areas on every pair).
    let boxes: Vec<BoundingBox> = dets.iter().map(|d| d.bbox.to_canonical()).collect();
    let areas: Vec<f32> = boxes.iter().map(box_area).collect();

    let n = dets.len();
    let mut used = vec![false; n];
    let mut out: Vec<DetectBox> = Vec::with_capacity(n);

    for oi in 0..n {
        let i = order[oi];
        if used[i] {
            continue;
        }
        used[i] = true;
        let base_box = boxes[i];
        let base_area = areas[i];
        let base_label = dets[i].label;
        let mut acc = base_box;
        let mut max_score = dets[i].score;

        for &j in &order[(oi + 1)..] {
            if used[j] {
                continue;
            }
            if !cfg.class_agnostic && dets[j].label != base_label {
                continue;
            }
            // Metric measured against the ORIGINAL base box (parity with the
            // reference), using cached areas.
            if metric_value_with_areas(cfg.metric, &base_box, base_area, &boxes[j], areas[j])
                >= cfg.threshold
            {
                used[j] = true;
                let c = boxes[j];
                acc.xmin = acc.xmin.min(c.xmin);
                acc.ymin = acc.ymin.min(c.ymin);
                acc.xmax = acc.xmax.max(c.xmax);
                acc.ymax = acc.ymax.max(c.ymax);
                max_score = max_score.max(dets[j].score);
            }
        }

        out.push(DetectBox {
            bbox: acc,
            score: max_score,
            label: base_label,
        });
    }

    if cfg.score_threshold > 0.0 {
        out.retain(|d| d.score >= cfg.score_threshold);
    }
    out.truncate(cfg.max_det);
    out
}

/// Streaming collector for one frame's tiled detections. A pipelined runtime
/// pushes each tile's per-tile-decoded boxes as inference completes (any
/// order), then finalizes once every tile has arrived — the "collect after the
/// final tile" fan-in. Not internally synchronized; keep one accumulator per
/// in-flight frame.
#[derive(Debug, Clone)]
pub struct TiledFrameAccumulator {
    frame_dims: (f32, f32),
    tiles_total: usize,
    /// Per-tile-index arrival flags — makes `push_tile` idempotent and the
    /// completion fence robust to duplicate / out-of-range / retried pushes
    /// (an at-least-once async pipeline can deliver the same tile twice).
    seen: Vec<bool>,
    received: usize,
    dets: Vec<DetectBox>,
    cfg: MergeConfig,
}

impl TiledFrameAccumulator {
    /// Create an accumulator for a frame with `tiles_total` tiles. `frame_dims`
    /// is `(frame_w, frame_h)` in pixels, used by [`Self::finalize_normalized`].
    /// `est_per_tile` pre-reserves the detection buffer.
    ///
    /// # Examples
    /// ```
    /// use edgefirst_decoder::tiling::{MergeConfig, TiledFrameAccumulator};
    /// let acc = TiledFrameAccumulator::new((1920.0, 1080.0), 12, MergeConfig::default(), 16);
    /// assert_eq!(acc.remaining(), 12);
    /// assert!(!acc.is_complete());
    /// ```
    pub fn new(
        frame_dims: (f32, f32),
        tiles_total: usize,
        cfg: MergeConfig,
        est_per_tile: usize,
    ) -> Self {
        Self {
            frame_dims,
            tiles_total,
            seen: vec![false; tiles_total],
            received: 0,
            dets: Vec::with_capacity(tiles_total.saturating_mul(est_per_tile)),
            cfg,
        }
    }

    /// Lift one tile's per-tile-decoded boxes to full-frame pixels and append
    /// them. Idempotent per `placement.index`: a duplicate, out-of-range, or
    /// foreign placement (one whose `count` disagrees with this accumulator's
    /// `tiles_total`, i.e. from a different plan/frame) is ignored and its boxes
    /// dropped, so out-of-order **and** at-least-once delivery both converge to
    /// the same result. Returns `true` if the tile was newly accepted, `false`
    /// otherwise.
    pub fn push_tile(&mut self, tile_boxes: Vec<DetectBox>, placement: &TilePlacement) -> bool {
        let idx = placement.index;
        // Runtime guard (not debug-only): reject placements from a different
        // plan so a mixed-frame mistake can't corrupt fan-in completion.
        if placement.count != self.tiles_total || idx >= self.tiles_total || self.seen[idx] {
            return false;
        }
        self.seen[idx] = true;
        self.dets.extend(lift_tile_boxes(tile_boxes, placement));
        self.received += 1;
        true
    }

    /// True once every tile of the frame has been pushed (by distinct index).
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.received >= self.tiles_total
    }

    /// Tiles still outstanding.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.tiles_total.saturating_sub(self.received)
    }

    /// Merge all accumulated detections into full-frame **pixel** boxes.
    #[must_use]
    pub fn finalize(self) -> Vec<DetectBox> {
        merge_tiled_detections(self.dets, &self.cfg)
    }

    /// Merge then renormalize to `[0,1]` by `frame_dims` (for the tracker,
    /// matching the non-tiled normalized-detection contract).
    ///
    /// Returns an empty list when `frame_dims` are non-finite or non-positive
    /// rather than emitting Inf/NaN coordinates.
    #[must_use]
    pub fn finalize_normalized(self) -> Vec<DetectBox> {
        let (fw, fh) = self.frame_dims;
        if !(fw.is_finite() && fh.is_finite() && fw > 0.0 && fh > 0.0) {
            return Vec::new();
        }
        let inv_w = 1.0 / fw;
        let inv_h = 1.0 / fh;
        let mut merged = merge_tiled_detections(self.dets, &self.cfg);
        for d in &mut merged {
            d.bbox.xmin *= inv_w;
            d.bbox.xmax *= inv_w;
            d.bbox.ymin *= inv_h;
            d.bbox.ymax *= inv_h;
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(b: [f32; 4], score: f32, label: usize) -> DetectBox {
        DetectBox {
            bbox: BoundingBox::new(b[0], b[1], b[2], b[3]),
            score,
            label,
        }
    }

    // --- lift -------------------------------------------------------------

    fn placement(origin: (f32, f32), crop: (f32, f32)) -> TilePlacement {
        TilePlacement {
            index: 0,
            count: 1,
            origin,
            crop_size: crop,
            letterbox: None,
            frame_dims: (3840.0, 2160.0),
        }
    }

    #[test]
    fn lift_no_letterbox_matches_origin_plus_scale() {
        let p = placement((100.0, 200.0), (640.0, 640.0));
        let lifted = lift_tile_boxes(
            vec![
                det([0.0, 0.0, 1.0, 1.0], 0.9, 0),
                det([0.25, 0.5, 0.75, 1.0], 0.8, 0),
            ],
            &p,
        );
        assert_eq!(lifted[0].bbox, BoundingBox::new(100.0, 200.0, 740.0, 840.0));
        assert_eq!(lifted[1].bbox, BoundingBox::new(260.0, 520.0, 580.0, 840.0));
    }

    #[test]
    fn lift_with_letterbox_inverts_then_scales() {
        // A box filling the letterbox content region should, after un-padding,
        // fill the crop and lift identically to the no-letterbox full-crop box.
        let mut p = placement((0.0, 0.0), (640.0, 640.0));
        p.letterbox = Some([0.1, 0.1, 0.9, 0.9]);
        let lifted = lift_tile_boxes(vec![det([0.1, 0.1, 0.9, 0.9], 0.9, 0)], &p);
        let b = lifted[0].bbox;
        assert!((b.xmin - 0.0).abs() < 1e-3);
        assert!((b.ymin - 0.0).abs() < 1e-3);
        assert!((b.xmax - 640.0).abs() < 1e-3);
        assert!((b.ymax - 640.0).abs() < 1e-3);
    }

    #[test]
    fn lift_empty_is_empty() {
        let p = placement((0.0, 0.0), (640.0, 640.0));
        assert!(lift_tile_boxes(vec![], &p).is_empty());
    }

    #[test]
    fn lift_roundtrip_with_letterbox() {
        // Project a known full-frame box into tile-normalized (letterboxed)
        // coords, then lift it back and confirm it returns to the original.
        let p = TilePlacement {
            index: 0,
            count: 1,
            origin: (100.0, 200.0),
            crop_size: (640.0, 640.0),
            letterbox: Some([0.1, 0.1, 0.9, 0.9]),
            frame_dims: (1920.0, 1080.0),
        };
        // Full-frame target [228,328,420,520] -> crop-normalized (subtract origin,
        // /crop) -> letterbox-normalized (scale by lb extent + offset).
        let crop_n = [
            (228.0 - 100.0) / 640.0,
            (328.0 - 200.0) / 640.0,
            (420.0 - 100.0) / 640.0,
            (520.0 - 200.0) / 640.0,
        ];
        let [lx0, ly0, lx1, ly1] = [0.1, 0.1, 0.9, 0.9];
        let model_n = det(
            [
                lx0 + crop_n[0] * (lx1 - lx0),
                ly0 + crop_n[1] * (ly1 - ly0),
                lx0 + crop_n[2] * (lx1 - lx0),
                ly0 + crop_n[3] * (ly1 - ly0),
            ],
            0.9,
            0,
        );
        let lifted = lift_tile_boxes(vec![model_n], &p);
        let b = lifted[0].bbox;
        assert!((b.xmin - 228.0).abs() < 1e-2, "xmin {}", b.xmin);
        assert!((b.ymin - 328.0).abs() < 1e-2, "ymin {}", b.ymin);
        assert!((b.xmax - 420.0).abs() < 1e-2, "xmax {}", b.xmax);
        assert!((b.ymax - 520.0).abs() < 1e-2, "ymax {}", b.ymax);
    }

    #[test]
    fn lift_letterbox_clamp_fires_and_no_div_by_zero() {
        // A box outside the letterbox content region clamps to the edge; a
        // degenerate (zero-span) letterbox axis must not divide by zero.
        let p = TilePlacement {
            index: 0,
            count: 1,
            origin: (0.0, 0.0),
            crop_size: (640.0, 640.0),
            letterbox: Some([0.1, 0.1, 0.9, 0.9]),
            frame_dims: (640.0, 640.0),
        };
        // Box at the very top-left, outside [0.1,0.1] content origin.
        let lifted = lift_tile_boxes(vec![det([0.0, 0.0, 0.05, 0.05], 0.9, 0)], &p);
        assert_eq!(lifted[0].bbox.xmin, 0.0);
        assert_eq!(lifted[0].bbox.ymin, 0.0);

        // Degenerate letterbox (lx0 == lx1): unit scale, finite result.
        let pd = TilePlacement {
            letterbox: Some([0.5, 0.1, 0.5, 0.9]),
            ..p
        };
        let out = lift_tile_boxes(vec![det([0.2, 0.2, 0.8, 0.8], 0.9, 0)], &pd);
        assert!(out[0].bbox.xmin.is_finite() && out[0].bbox.xmax.is_finite());
    }

    // --- merge: the canonical IOS-vs-IOU case ----------------------------

    #[test]
    fn ios_merges_fragment_iou_does_not() {
        // From modelpack tests/test_tiled_merge.py: B fully inside A
        // (IoS=1.0, IoU=0.167).
        let a = det([100.0, 100.0, 400.0, 300.0], 0.9, 0);
        let b = det([350.0, 100.0, 400.0, 300.0], 0.7, 0);

        let ios = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                metric: MatchMetric::Ios,
                threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(ios.len(), 1);
        assert_eq!(ios[0].bbox, BoundingBox::new(100.0, 100.0, 400.0, 300.0));
        assert_eq!(ios[0].score, 0.9);

        let iou = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                metric: MatchMetric::Iou,
                threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(iou.len(), 2);
    }

    #[test]
    fn merge_respects_class_unless_agnostic() {
        let a = det([100.0, 100.0, 400.0, 300.0], 0.9, 0);
        let b = det([350.0, 100.0, 400.0, 300.0], 0.7, 1); // different class

        let aware = merge_tiled_detections(vec![a, b], &MergeConfig::default());
        assert_eq!(aware.len(), 2);

        let agnostic = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                class_agnostic: true,
                ..Default::default()
            },
        );
        assert_eq!(agnostic.len(), 1);
        // Merged box keeps the base (highest-score) label.
        assert_eq!(agnostic[0].label, 0);
        assert_eq!(
            agnostic[0].bbox,
            BoundingBox::new(100.0, 100.0, 400.0, 300.0)
        );
    }

    #[test]
    fn merge_enclosing_union_for_partial_overlap() {
        // Two boxes overlapping >=0.5 IoS merge to their enclosing union.
        let a = det([0.0, 0.0, 100.0, 100.0], 0.9, 0);
        let b = det([50.0, 0.0, 150.0, 100.0], 0.8, 0); // IoS = 0.5
        let merged = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                metric: MatchMetric::Ios,
                threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, BoundingBox::new(0.0, 0.0, 150.0, 100.0));
        assert_eq!(merged[0].score, 0.9);
    }

    #[test]
    fn merge_disjoint_boxes_pass_through() {
        let a = det([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        let b = det([100.0, 100.0, 110.0, 110.0], 0.8, 0);
        let merged = merge_tiled_detections(vec![a, b], &MergeConfig::default());
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn merge_empty_is_empty() {
        assert!(merge_tiled_detections(vec![], &MergeConfig::default()).is_empty());
    }

    #[test]
    fn merge_threshold_boundary_is_inclusive() {
        // IoS exactly == threshold must merge (>=, not >).
        let a = det([0.0, 0.0, 100.0, 100.0], 0.9, 0);
        let b = det([50.0, 0.0, 150.0, 100.0], 0.8, 0); // IoS = 0.5 exactly
        let merged = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                metric: MatchMetric::Ios,
                threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_score_threshold_drops_low_groups() {
        let a = det([0.0, 0.0, 10.0, 10.0], 0.3, 0);
        let b = det([100.0, 100.0, 110.0, 110.0], 0.8, 0);
        let merged = merge_tiled_detections(
            vec![a, b],
            &MergeConfig {
                score_threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].score, 0.8);
    }

    #[test]
    fn merge_max_det_caps_highest_scoring() {
        let dets: Vec<DetectBox> = (0..10)
            .map(|i| {
                det(
                    [i as f32 * 50.0, 0.0, i as f32 * 50.0 + 10.0, 10.0],
                    1.0 - i as f32 * 0.05,
                    0,
                )
            })
            .collect();
        let merged = merge_tiled_detections(
            dets,
            &MergeConfig {
                max_det: 3,
                ..Default::default()
            },
        );
        assert_eq!(merged.len(), 3);
        assert!(merged[0].score >= merged[1].score);
        assert!(merged[1].score >= merged[2].score);
    }

    #[test]
    fn merge_max_det_exact_boundary() {
        // N disjoint boxes; max_det == N keeps all, max_det == N-1 drops one.
        let make = || -> Vec<DetectBox> {
            (0..5)
                .map(|i| {
                    det(
                        [i as f32 * 50.0, 0.0, i as f32 * 50.0 + 10.0, 10.0],
                        1.0 - i as f32 * 0.05,
                        0,
                    )
                })
                .collect()
        };
        assert_eq!(
            merge_tiled_detections(
                make(),
                &MergeConfig {
                    max_det: 5,
                    ..Default::default()
                }
            )
            .len(),
            5
        );
        assert_eq!(
            merge_tiled_detections(
                make(),
                &MergeConfig {
                    max_det: 4,
                    ..Default::default()
                }
            )
            .len(),
            4
        );
    }

    #[test]
    fn merge_score_threshold_boundary_is_inclusive() {
        // A group whose max score == score_threshold is kept (>=).
        let a = det([0.0, 0.0, 10.0, 10.0], 0.5, 0);
        let merged = merge_tiled_detections(
            vec![a],
            &MergeConfig {
                score_threshold: 0.5,
                ..Default::default()
            },
        );
        assert_eq!(merged.len(), 1);
    }

    // --- accumulator ------------------------------------------------------

    fn empty_placement(index: usize, count: usize) -> TilePlacement {
        TilePlacement {
            index,
            count,
            origin: (0.0, 0.0),
            crop_size: (640.0, 640.0),
            letterbox: None,
            frame_dims: (640.0, 640.0),
        }
    }

    #[test]
    fn accumulator_fan_in_fence() {
        let mut acc = TiledFrameAccumulator::new((640.0, 640.0), 3, MergeConfig::default(), 8);
        assert!(!acc.is_complete());
        assert_eq!(acc.remaining(), 3);
        assert!(acc.push_tile(vec![], &empty_placement(0, 3)));
        assert!(acc.push_tile(vec![], &empty_placement(1, 3)));
        assert!(!acc.is_complete());
        assert_eq!(acc.remaining(), 1);
        assert!(acc.push_tile(vec![], &empty_placement(2, 3)));
        assert!(acc.is_complete());
        assert_eq!(acc.remaining(), 0);
        assert!(acc.finalize().is_empty());
    }

    #[test]
    fn accumulator_dedups_and_ignores_overpush() {
        let mut acc = TiledFrameAccumulator::new((640.0, 640.0), 2, MergeConfig::default(), 8);
        assert!(acc.push_tile(vec![], &empty_placement(0, 2)));
        // Duplicate index 0 is ignored (idempotent under at-least-once delivery).
        assert!(!acc.push_tile(vec![], &empty_placement(0, 2)));
        assert_eq!(acc.remaining(), 1);
        assert!(!acc.is_complete());
        assert!(acc.push_tile(vec![], &empty_placement(1, 2)));
        assert!(acc.is_complete());
        // Out-of-range index is ignored, never over-counts.
        assert!(!acc.push_tile(vec![], &empty_placement(2, 2)));
        assert_eq!(acc.remaining(), 0);
    }

    #[test]
    fn accumulator_rejects_foreign_plan_count() {
        // A placement from a different plan (count != tiles_total) is rejected
        // at runtime — not just under debug_assert — so a mixed-frame mistake
        // can't corrupt fan-in completion.
        let mut acc = TiledFrameAccumulator::new((640.0, 640.0), 3, MergeConfig::default(), 8);
        assert!(!acc.push_tile(vec![], &empty_placement(0, 4)));
        assert_eq!(acc.remaining(), 3);
        assert!(!acc.is_complete());
        // The same index from the correct plan is still accepted afterward.
        assert!(acc.push_tile(vec![], &empty_placement(0, 3)));
        assert_eq!(acc.remaining(), 2);
    }

    #[test]
    fn accumulator_out_of_order_equals_in_order() {
        let cfg = MergeConfig::default();
        let frame = (1280.0, 640.0);
        // Two tiles side by side, a box near the seam in each.
        let p0 = TilePlacement {
            index: 0,
            count: 2,
            origin: (0.0, 0.0),
            crop_size: (640.0, 640.0),
            letterbox: None,
            frame_dims: frame,
        };
        let p1 = TilePlacement {
            index: 1,
            count: 2,
            origin: (640.0, 0.0),
            crop_size: (640.0, 640.0),
            letterbox: None,
            frame_dims: frame,
        };
        let t0 = vec![det([0.9, 0.4, 1.0, 0.6], 0.8, 0)];
        let t1 = vec![det([0.0, 0.4, 0.1, 0.6], 0.9, 0)];

        let mut a = TiledFrameAccumulator::new(frame, 2, cfg, 8);
        a.push_tile(t0.clone(), &p0);
        a.push_tile(t1.clone(), &p1);
        let in_order = a.finalize();

        let mut b = TiledFrameAccumulator::new(frame, 2, cfg, 8);
        b.push_tile(t1, &p1);
        b.push_tile(t0, &p0);
        let out_order = b.finalize();

        assert_eq!(in_order.len(), out_order.len());
        for (x, y) in in_order.iter().zip(out_order.iter()) {
            assert_eq!(x.bbox, y.bbox);
            assert_eq!(x.score, y.score);
        }
    }

    #[test]
    fn finalize_normalized_equals_finalize_over_frame_dims() {
        let cfg = MergeConfig::default();
        let frame = (1280.0, 640.0);
        let p = TilePlacement {
            index: 0,
            count: 1,
            origin: (100.0, 50.0),
            crop_size: (640.0, 640.0),
            letterbox: None,
            frame_dims: frame,
        };
        let boxes = vec![det([0.1, 0.1, 0.4, 0.4], 0.9, 0)];

        let mut a = TiledFrameAccumulator::new(frame, 1, cfg, 8);
        a.push_tile(boxes.clone(), &p);
        let px = a.finalize();

        let mut b = TiledFrameAccumulator::new(frame, 1, cfg, 8);
        b.push_tile(boxes, &p);
        let norm = b.finalize_normalized();

        assert_eq!(px.len(), norm.len());
        let (fw, fh) = frame;
        for (p, n) in px.iter().zip(norm.iter()) {
            assert!((n.bbox.xmin - p.bbox.xmin / fw).abs() < 1e-4);
            assert!((n.bbox.ymin - p.bbox.ymin / fh).abs() < 1e-4);
            assert!((n.bbox.xmax - p.bbox.xmax / fw).abs() < 1e-4);
            assert!((n.bbox.ymax - p.bbox.ymax / fh).abs() < 1e-4);
        }
    }

    #[test]
    fn finalize_normalized_rejects_invalid_frame_dims() {
        let cfg = MergeConfig::default();
        let boxes = vec![det([10.0, 10.0, 40.0, 40.0], 0.9, 0)];
        for frame in [(0.0, 640.0), (1280.0, 0.0), (f32::NAN, 640.0), (1280.0, f32::INFINITY)]
        {
            let p = TilePlacement {
                index: 0,
                count: 1,
                origin: (0.0, 0.0),
                crop_size: (640.0, 640.0),
                letterbox: None,
                frame_dims: frame,
            };
            let mut acc = TiledFrameAccumulator::new(frame, 1, cfg, 8);
            acc.push_tile(boxes.clone(), &p);
            assert!(
                acc.finalize_normalized().is_empty(),
                "expected empty for frame_dims={frame:?}"
            );
        }
    }

    // --- end-to-end: accumulator -> merge -> tracker ----------------------

    /// One tile sees the whole object, an overlapping tile sees a contained
    /// fragment of it. IOS merges them into a single full-frame detection that
    /// the tracker resolves to ONE track; IOU leaves two, yielding TWO tracks
    /// (the negative control proving IOS does its job, not that it collapses
    /// everything).
    #[cfg(feature = "tracker")]
    #[test]
    fn e2e_ios_one_track_iou_two_tracks() {
        use edgefirst_tracker::{ByteTrackBuilder, Tracker};

        // Two distinct tiles (indices 0 and 1) of a 2-tile frame, each lifted via
        // a whole-frame placement so tile 0 yields the full object and tile 1 a
        // contained fragment — simulating overlapping tiles seeing the same object.
        let p0 = TilePlacement {
            index: 0,
            count: 2,
            origin: (0.0, 0.0),
            crop_size: (640.0, 640.0),
            letterbox: None,
            frame_dims: (640.0, 640.0),
        };
        let p1 = TilePlacement { index: 1, ..p0 };
        let full = det(
            [100.0 / 640.0, 100.0 / 640.0, 400.0 / 640.0, 300.0 / 640.0],
            0.9,
            0,
        );
        let frag = det(
            [350.0 / 640.0, 100.0 / 640.0, 400.0 / 640.0, 300.0 / 640.0],
            0.7,
            0,
        );

        let run = |metric: MatchMetric| -> (usize, usize) {
            let cfg = MergeConfig {
                metric,
                threshold: 0.5,
                ..Default::default()
            };
            let mut acc = TiledFrameAccumulator::new((640.0, 640.0), 2, cfg, 4);
            acc.push_tile(vec![full], &p0);
            acc.push_tile(vec![frag], &p1);
            let merged = acc.finalize_normalized();
            let mut tracker = ByteTrackBuilder::new().build::<DetectBox>();
            let _ = tracker.update(&merged, 1_000);
            (merged.len(), tracker.get_active_tracks().len())
        };

        let (ios_merged, ios_tracks) = run(MatchMetric::Ios);
        assert_eq!(ios_merged, 1, "IOS should merge the fragment");
        assert_eq!(ios_tracks, 1, "merged detection yields one track");

        let (iou_merged, iou_tracks) = run(MatchMetric::Iou);
        assert_eq!(iou_merged, 2, "IOU should leave the fragment separate");
        assert_eq!(iou_tracks, 2, "two detections yield two tracks");
    }
}
