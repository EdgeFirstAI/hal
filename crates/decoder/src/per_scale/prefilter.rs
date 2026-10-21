// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Score-threshold prefilter for the per-scale decoder.
//!
//! Most anchors of a frame score far below any useful threshold (on a
//! VisDrone yolov8n frame about 7% of anchors reach 0.001), yet decoding a
//! box costs a DFL softmax per side. The prefilter compares each anchor's
//! raw integer class scores against the threshold mapped into the
//! quantized domain, gathers the anchors that can still pass into compact
//! scratch, and lets the unchanged level kernels decode only those.
//!
//! The quantized bound sits one step below the threshold, so the kept set
//! is a superset of the anchors that pass: the float candidate filter
//! downstream still makes the exact decision, and the detections match
//! the dense decode bit for bit.

use super::kernels::dispatch::InputView;
use super::outputs::LayoutScratch;
use super::plan::{Layout, LevelPlan};
use super::Activation;
use crate::{DecoderError, DecoderResult, Quantization};

/// Kept anchors are decoded in whole groups of this many. Every role's
/// packed slice is then a multiple of 16 elements, the widest NEON chunk
/// of the level kernels, so each element takes the vector path exactly
/// as it does in a full level instead of a scalar tail that rounds
/// differently.
pub(crate) const PACK_ANCHORS: usize = 16;

/// Per-level scratch for the compact decode, reused frame to frame.
#[derive(Debug)]
pub(crate) struct LevelScratch {
    /// Copy of the level's plan whose grid holds the kept anchors' grid
    /// cells; `h * w` is the packed count, as the level kernels expect.
    pub(crate) plan: LevelPlan,
    /// Kept anchor indices within the level, ascending, then padded by
    /// [`LevelScratch::pack`].
    pub(crate) anchors: Vec<u32>,
    /// Anchors kept this frame, before padding.
    pub(crate) kept: usize,
    pub(crate) boxes: LayoutScratch,
    pub(crate) scores: LayoutScratch,
    pub(crate) mask_coefs: LayoutScratch,
}

impl LevelScratch {
    pub(crate) fn for_level(level: &LevelPlan) -> Self {
        let n = level.h * level.w;
        Self {
            plan: LevelPlan {
                stride: level.stride,
                h: 1,
                w: 0,
                reg_max: level.reg_max,
                anchor_offset: 0,
                grid_x: vec![0.0; n].into_boxed_slice(),
                grid_y: vec![0.0; n].into_boxed_slice(),
                box_shape: level.box_shape.clone(),
                score_shape: level.score_shape.clone(),
                mc_shape: level.mc_shape.clone(),
                layout: Layout::Nhwc,
            },
            anchors: Vec::with_capacity(n),
            kept: 0,
            boxes: LayoutScratch::default(),
            scores: LayoutScratch::default(),
            mask_coefs: LayoutScratch::default(),
        }
    }

    /// Pad the kept anchors to a whole number of [`PACK_ANCHORS`] groups
    /// by repeating the last one, and point the compact plan at their grid
    /// cells in `level`. Returns the kept count before padding, also left
    /// in `kept`; the padded anchors' outputs are scratch that the next
    /// level overwrites.
    pub(crate) fn pack(&mut self, level: &LevelPlan) -> usize {
        let kept = self.anchors.len();
        self.kept = kept;
        if let Some(&last) = self.anchors.last() {
            self.anchors
                .resize(kept.next_multiple_of(PACK_ANCHORS), last);
        }
        if self.plan.grid_x.len() < self.anchors.len() {
            self.plan.grid_x = vec![0.0; self.anchors.len()].into_boxed_slice();
            self.plan.grid_y = vec![0.0; self.anchors.len()].into_boxed_slice();
        }
        self.plan.w = self.anchors.len();
        for (k, &a) in self.anchors.iter().enumerate() {
            self.plan.grid_x[k] = level.grid_x[a as usize];
            self.plan.grid_y[k] = level.grid_y[a as usize];
        }
        kept
    }
}

/// Lowest raw score that can still reach `threshold` once dequantized and
/// activated, less one quantization step of margin. `None` when the
/// prefilter cannot bound the threshold, and the caller decodes densely.
pub(crate) fn raw_score_floor(
    threshold: f32,
    q: Quantization,
    activation: Activation,
) -> Option<i32> {
    let bounded = threshold > 0.0 && threshold < 1.0 && q.scale.is_finite() && q.scale > 0.0;
    if !bounded {
        return None;
    }
    let bound = match activation {
        Activation::Sigmoid => (threshold / (1.0 - threshold)).ln(),
        Activation::None => threshold,
    };
    let raw = (f64::from(bound) / f64::from(q.scale) + f64::from(q.zero_point)).floor() - 1.0;
    Some(raw.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32)
}

/// Append to `anchors` every anchor whose largest raw class score is at or
/// above `floor`. `input` is one level's NHWC `[h * w, num_classes]` scores.
pub(crate) fn select_anchors(
    input: &InputView<'_>,
    num_classes: usize,
    floor: i32,
    anchors: &mut Vec<u32>,
) -> DecoderResult<()> {
    fn scan<T: Copy + Into<i32>>(rows: &[T], nc: usize, floor: i32, anchors: &mut Vec<u32>) {
        for (a, row) in rows.chunks_exact(nc).enumerate() {
            if row.iter().any(|&v| v.into() >= floor) {
                anchors.push(a as u32);
            }
        }
    }
    anchors.clear();
    if num_classes == 0 {
        return Ok(());
    }
    match input {
        InputView::I8(s) => scan(s, num_classes, floor, anchors),
        InputView::U8(s) => scan(s, num_classes, floor, anchors),
        InputView::I16(s) => scan(s, num_classes, floor, anchors),
        InputView::U16(s) => scan(s, num_classes, floor, anchors),
        InputView::F16(_) | InputView::F32(_) => {
            return Err(DecoderError::Internal(
                "score prefilter reached a float score tensor".into(),
            ))
        }
    }
    Ok(())
}

/// Copy the `row`-element rows of the kept `anchors` out of `input` into
/// `scratch`, returning an `InputView` over the packed rows.
pub(crate) fn gather_rows<'s>(
    input: &InputView<'_>,
    row: usize,
    anchors: &[u32],
    scratch: &'s mut LayoutScratch,
) -> InputView<'s> {
    macro_rules! gather {
        ($src:expr, $ensure:ident, $variant:ident) => {{
            let dst = scratch.$ensure(anchors.len() * row);
            for (out, &a) in dst.chunks_exact_mut(row).zip(anchors) {
                let a = a as usize * row;
                out.copy_from_slice(&$src[a..a + row]);
            }
            InputView::$variant(dst)
        }};
    }
    match input {
        InputView::I8(s) => gather!(s, ensure_i8, I8),
        InputView::U8(s) => gather!(s, ensure_u8, U8),
        InputView::I16(s) => gather!(s, ensure_i16, I16),
        InputView::U16(s) => gather!(s, ensure_u16, U16),
        InputView::F16(s) => gather!(s, ensure_f16, F16),
        InputView::F32(s) => gather!(s, ensure_f32, F32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Every raw value whose activated score reaches the threshold is at or
    /// above the floor, for both activations and a spread of quantizations.
    #[test]
    fn floor_never_drops_a_passing_score() {
        let quants = [
            (0.17, 108),
            (0.0039, -128),
            (0.52, 23),
            (0.05, 0),
            (1.3, -7),
        ];
        let thresholds = [0.001, 0.01, 0.25, 0.5, 0.9, 0.999];
        for &(scale, zp) in &quants {
            let q = Quantization {
                scale,
                zero_point: zp,
            };
            for &t in &thresholds {
                for act in [Activation::Sigmoid, Activation::None] {
                    let floor = raw_score_floor(t, q, act).unwrap();
                    for raw in -128..=127_i32 {
                        let x = (raw - zp) as f32 * scale;
                        let s = match act {
                            Activation::Sigmoid => sigmoid(x),
                            Activation::None => x,
                        };
                        if s >= t {
                            assert!(raw >= floor, "q={q:?} t={t} {act:?}: raw {raw} < {floor}");
                        }
                    }
                }
            }
        }
    }

    /// The floor sits within two quantization steps of the exact bound, so
    /// it rejects nearly everything that cannot pass.
    #[test]
    fn floor_is_tight() {
        let q = Quantization {
            scale: 0.17,
            zero_point: 108,
        };
        let floor = raw_score_floor(0.25, q, Activation::Sigmoid).unwrap();
        let exact = ((0.25_f32 / 0.75).ln() / 0.17 + 108.0).ceil() as i32;
        assert!(exact - floor <= 2, "exact {exact} floor {floor}");
    }

    #[test]
    fn floor_declines_unboundable_thresholds() {
        let q = Quantization {
            scale: 0.1,
            zero_point: 0,
        };
        for t in [0.0, -0.5, 1.0, 2.0, f32::NAN] {
            assert_eq!(raw_score_floor(t, q, Activation::Sigmoid), None, "t={t}");
        }
        let zero = Quantization {
            scale: 0.0,
            zero_point: 0,
        };
        assert_eq!(raw_score_floor(0.25, zero, Activation::Sigmoid), None);
    }

    #[test]
    fn select_keeps_rows_with_any_class_at_floor() {
        let scores: [i8; 9] = [-5, -9, -9, -9, -9, -9, -9, 3, -9];
        let mut anchors = vec![42];
        select_anchors(&InputView::I8(&scores), 3, -5, &mut anchors).unwrap();
        assert_eq!(anchors, vec![0, 2]);
    }

    #[test]
    fn select_rejects_float_scores() {
        let scores = [0.5_f32; 4];
        let mut anchors = Vec::new();
        assert!(select_anchors(&InputView::F32(&scores), 2, 0, &mut anchors).is_err());
    }

    #[test]
    fn pack_pads_to_whole_groups_with_the_last_anchor() {
        let (gx, gy) = crate::per_scale::kernels::grids::make_anchor_grid(8, 4);
        let level = LevelPlan {
            stride: 8.0,
            h: 4,
            w: 8,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 4, 8, 64].into_boxed_slice(),
            score_shape: vec![1, 4, 8, 2].into_boxed_slice(),
            mc_shape: None,
            layout: Layout::Nhwc,
        };
        let mut scratch = LevelScratch::for_level(&level);
        scratch.anchors.extend([3, 9, 30]);
        assert_eq!(scratch.pack(&level), 3);
        assert_eq!(scratch.kept, 3);
        assert_eq!(scratch.anchors.len(), PACK_ANCHORS);
        assert!(scratch.anchors[3..].iter().all(|&a| a == 30));
        assert_eq!(scratch.plan.h * scratch.plan.w, PACK_ANCHORS);
        assert_eq!(scratch.plan.grid_x[1], level.grid_x[9]);
        assert_eq!(scratch.plan.grid_y[2], level.grid_y[30]);

        scratch.anchors.clear();
        assert_eq!(scratch.pack(&level), 0);
        assert!(scratch.anchors.is_empty());
    }

    #[test]
    fn gather_packs_kept_rows_in_order() {
        let boxes: Vec<u8> = (0..12).collect();
        let mut scratch = LayoutScratch::default();
        match gather_rows(&InputView::U8(&boxes), 3, &[1, 3], &mut scratch) {
            InputView::U8(v) => assert_eq!(v, &[3, 4, 5, 9, 10, 11]),
            other => panic!("unexpected view {other:?}"),
        }
    }
}
