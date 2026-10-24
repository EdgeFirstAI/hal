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
//! The bound is found by running the level's own score kernel over every
//! raw value, so it accounts for the kernel's rounding and approximations:
//! the kept set is a superset of the anchors that pass, the float
//! candidate filter downstream still makes the exact decision, and the
//! detections match the dense decode bit for bit.
//!
//! A level whose anchor count is not a whole number of [`PACK_ANCHORS`]
//! groups ends in a partial group, whose elements the level kernels may
//! decode in a scalar tail that rounds differently from their vector path.
//! Only the whole groups (the body) are packed; the trailing partial group
//! is always decoded as one contiguous slice, so every element in it takes
//! the same path it takes in the dense decode.

use super::kernels::dispatch::{DstSliceMut, InputView, ScoreLevelDispatch};
use super::outputs::LayoutScratch;
use super::plan::{Layout, LevelPlan};
use super::{Activation, DecodeDtype};
use crate::{DecoderError, DecoderResult, Quantization};
use edgefirst_tensor::DType;
use half::f16;

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
    /// Anchors in the level's whole [`PACK_ANCHORS`] groups. The anchors
    /// from here to the end of the level form the tail.
    pub(crate) body: usize,
    /// The level's plan cut down to its tail anchors, decoded whole.
    pub(crate) tail: LevelPlan,
    /// Kept body anchor indices within the level, ascending, then padded
    /// by [`LevelScratch::pack`].
    pub(crate) anchors: Vec<u32>,
    /// Body anchors kept this frame, before padding.
    pub(crate) kept: usize,
    /// Last floor from [`raw_score_floor`], keyed by the score quantization
    /// and threshold bits it was found for.
    floor: Option<((u32, i32, u32), i32)>,
    pub(crate) boxes: LayoutScratch,
    pub(crate) scores: LayoutScratch,
    pub(crate) mask_coefs: LayoutScratch,
}

impl LevelScratch {
    pub(crate) fn for_level(level: &LevelPlan) -> Self {
        let n = level.h * level.w;
        let body = n - n % PACK_ANCHORS;
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
            body,
            tail: LevelPlan {
                stride: level.stride,
                h: 1,
                w: n - body,
                reg_max: level.reg_max,
                anchor_offset: 0,
                grid_x: level.grid_x[body..].into(),
                grid_y: level.grid_y[body..].into(),
                box_shape: level.box_shape.clone(),
                score_shape: level.score_shape.clone(),
                mc_shape: level.mc_shape.clone(),
                layout: Layout::Nhwc,
            },
            anchors: Vec::with_capacity(body),
            kept: 0,
            floor: None,
            boxes: LayoutScratch::default(),
            scores: LayoutScratch::default(),
            mask_coefs: LayoutScratch::default(),
        }
    }

    /// Pad the kept body anchors to a whole number of [`PACK_ANCHORS`]
    /// groups by repeating the last one, and point the compact plan at
    /// their grid cells in `level`. Returns the kept count before padding,
    /// also left in `kept`; the padded anchors' outputs are scratch that
    /// the tail or the next level overwrites.
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

/// Lowest raw score whose decoded score passes the candidate filter's
/// `score >= threshold` test, or `i32::MAX` when none does. The level's
/// score kernel is run over every raw value of `dtype`, a multiple of
/// [`PACK_ANCHORS`] long so each takes the vector path, and the result is
/// cached in `scratch` for the quantization and threshold. `None` when
/// the scores are not integers or the threshold keeps every anchor.
pub(crate) fn raw_score_floor(
    scratch: &mut LevelScratch,
    dispatch: &ScoreLevelDispatch,
    dtype: DType,
    out: DecodeDtype,
    q: Quantization,
    activation: Activation,
    threshold: f32,
) -> DecoderResult<Option<i32>> {
    if threshold.is_nan() || threshold <= 0.0 {
        return Ok(None);
    }
    let key = (q.scale.to_bits(), q.zero_point, threshold.to_bits());
    if let Some((k, floor)) = scratch.floor {
        if k == key {
            return Ok(Some(floor));
        }
    }
    let (lo, hi) = match dtype {
        DType::I8 => (i32::from(i8::MIN), i32::from(i8::MAX)),
        DType::U8 => (0, i32::from(u8::MAX)),
        DType::I16 => (i32::from(i16::MIN), i32::from(i16::MAX)),
        DType::U16 => (0, i32::from(u16::MAX)),
        _ => return Ok(None),
    };
    let raws = lo..=hi;
    let i8s: Vec<i8>;
    let u8s: Vec<u8>;
    let i16s: Vec<i16>;
    let u16s: Vec<u16>;
    // Every raw value of these dtypes converts losslessly.
    let input = match dtype {
        DType::I8 => {
            i8s = raws.map(|r| r as i8).collect();
            InputView::I8(&i8s)
        }
        DType::U8 => {
            u8s = raws.map(|r| r as u8).collect();
            InputView::U8(&u8s)
        }
        DType::I16 => {
            i16s = raws.map(|r| r as i16).collect();
            InputView::I16(&i16s)
        }
        _ => {
            u16s = raws.map(|r| r as u16).collect();
            InputView::U16(&u16s)
        }
    };
    let n = (hi - lo + 1) as usize;
    debug_assert!(n.is_multiple_of(PACK_ANCHORS));
    let w = std::mem::replace(&mut scratch.plan.w, n);
    let first_pass = match out {
        DecodeDtype::F32 => {
            let mut dst = vec![0.0_f32; n];
            dispatch.run(
                input,
                q,
                1,
                &scratch.plan,
                activation,
                DstSliceMut::F32(&mut dst),
            )?;
            dst.iter().position(|&s| s >= threshold)
        }
        DecodeDtype::F16 => {
            let mut dst = vec![f16::ZERO; n];
            dispatch.run(
                input,
                q,
                1,
                &scratch.plan,
                activation,
                DstSliceMut::F16(&mut dst),
            )?;
            dst.iter().position(|&s| s.to_f32() >= threshold)
        }
    };
    scratch.plan.w = w;
    let floor = first_pass.map_or(i32::MAX, |i| lo + i as i32);
    scratch.floor = Some((key, floor));
    Ok(Some(floor))
}

/// An integer raw score type.
trait RawScore: Copy + Ord + TryFrom<i32> {
    const LOWEST: Self;
}

impl RawScore for i8 {
    const LOWEST: Self = i8::MIN;
}
impl RawScore for u8 {
    const LOWEST: Self = u8::MIN;
}
impl RawScore for i16 {
    const LOWEST: Self = i16::MIN;
}
impl RawScore for u16 {
    const LOWEST: Self = u16::MIN;
}

/// Raw scores are tested for a value at the floor this many at a time; only
/// a block that holds one has its rows examined.
const SCAN_BLOCK: usize = 64;

/// Append to `anchors` every anchor whose largest raw class score is at or
/// above `floor`. `input` holds NHWC `[anchors, num_classes]` score rows.
pub(crate) fn select_anchors(
    input: &InputView<'_>,
    num_classes: usize,
    floor: i32,
    anchors: &mut Vec<u32>,
) -> DecoderResult<()> {
    fn scan<T: RawScore>(rows: &[T], nc: usize, floor: i32, anchors: &mut Vec<u32>) {
        let floor = match T::try_from(floor) {
            Ok(f) => f,
            Err(_) if floor > 0 => return,
            Err(_) => {
                anchors.extend(0..(rows.len() / nc) as u32);
                return;
            }
        };
        // The first row not yet examined: a row spanning two blocks that
        // both hold a passing value is examined once.
        let mut next = 0;
        for (b, block) in rows.chunks(SCAN_BLOCK).enumerate() {
            if block.iter().fold(T::LOWEST, |m, &v| m.max(v)) < floor {
                continue;
            }
            let start = b * SCAN_BLOCK;
            let last = (start + block.len() - 1) / nc;
            for a in (start / nc).max(next)..=last {
                if rows[a * nc..(a + 1) * nc].iter().any(|&v| v >= floor) {
                    anchors.push(a as u32);
                }
            }
            next = last + 1;
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

    use crate::per_scale::kernels::cpu_features::CpuFeatures;

    fn scratch() -> LevelScratch {
        let mut s = LevelScratch::for_level(&LevelPlan {
            stride: 8.0,
            h: 1,
            w: 16,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: vec![0.0; 16].into_boxed_slice(),
            grid_y: vec![0.0; 16].into_boxed_slice(),
            box_shape: Box::new([1, 1, 16, 64]),
            score_shape: Box::new([1, 1, 16, 1]),
            mc_shape: None,
            layout: Layout::Nhwc,
        });
        s.plan.w = 16;
        s
    }

    fn floor(
        s: &mut LevelScratch,
        dtype: DType,
        out: DecodeDtype,
        q: Quantization,
        act: Activation,
        t: f32,
    ) -> Option<i32> {
        let f = CpuFeatures::from_env_or_probe().unwrap();
        let dispatch = ScoreLevelDispatch::select(dtype, out, &f).unwrap();
        raw_score_floor(s, &dispatch, dtype, out, q, act, t).unwrap()
    }

    /// Kernel scores of every u8 raw value, as the candidate filter sees them.
    fn u8_scores(out: DecodeDtype, q: Quantization, act: Activation) -> Vec<f32> {
        let f = CpuFeatures::from_env_or_probe().unwrap();
        let dispatch = ScoreLevelDispatch::select(DType::U8, out, &f).unwrap();
        let raws: Vec<u8> = (0..=255).collect();
        let mut s = scratch();
        s.plan.w = 256;
        match out {
            DecodeDtype::F32 => {
                let mut d = vec![0.0_f32; 256];
                dispatch
                    .run(
                        InputView::U8(&raws),
                        q,
                        1,
                        &s.plan,
                        act,
                        DstSliceMut::F32(&mut d),
                    )
                    .unwrap();
                d
            }
            DecodeDtype::F16 => {
                let mut d = vec![f16::ZERO; 256];
                dispatch
                    .run(
                        InputView::U8(&raws),
                        q,
                        1,
                        &s.plan,
                        act,
                        DstSliceMut::F16(&mut d),
                    )
                    .unwrap();
                d.iter().map(|v| v.to_f32()).collect()
            }
        }
    }

    /// The floor is the lowest raw value whose kernel score passes, so no
    /// raw value below it passes.
    #[test]
    fn floor_is_the_lowest_passing_raw_value() {
        for (scale, zp) in [(0.17, 108), (0.0039, 0), (0.05, 200)] {
            let q = Quantization {
                scale,
                zero_point: zp,
            };
            for out in [DecodeDtype::F32, DecodeDtype::F16] {
                for act in [Activation::Sigmoid, Activation::None] {
                    let scores = u8_scores(out, q, act);
                    for t in [0.001, 0.25, 0.5, 0.999] {
                        let f = floor(&mut scratch(), DType::U8, out, q, act, t).unwrap();
                        let first = scores.iter().position(|&s| s >= t);
                        let label = format!("q={q:?} {out:?} {act:?} t={t}");
                        assert_eq!(first.map_or(i32::MAX, |i| i as i32), f, "{label}");
                    }
                }
            }
        }
    }

    /// F16 output rounds a dequantized score up by many raw steps, so the
    /// floor sits below the exact bound.
    #[test]
    fn floor_follows_f16_rounding() {
        let q = Quantization {
            scale: 1.0 / 65535.0,
            zero_point: 0,
        };
        let t = 0.5004;
        let exact = (f64::from(t) * 65535.0).ceil() as i32;
        let f = floor(
            &mut scratch(),
            DType::U16,
            DecodeDtype::F16,
            q,
            Activation::None,
            t,
        )
        .unwrap();
        assert!(f < exact, "floor {f} exact {exact}");
        let f = floor(
            &mut scratch(),
            DType::U16,
            DecodeDtype::F32,
            q,
            Activation::None,
            t,
        )
        .unwrap();
        assert!((f - exact).abs() <= 1, "floor {f} exact {exact}");
    }

    #[test]
    fn floor_is_found_again_for_a_new_threshold() {
        let q = Quantization {
            scale: 0.1,
            zero_point: 0,
        };
        let mut s = scratch();
        let (out, act) = (DecodeDtype::F32, Activation::None);
        assert_eq!(floor(&mut s, DType::I8, out, q, act, 2.0), Some(20));
        assert_eq!(floor(&mut s, DType::I8, out, q, act, 5.0), Some(50));
        assert_eq!(floor(&mut s, DType::I8, out, q, act, 50.0), Some(i32::MAX));
        assert_eq!(s.plan.w, 16, "the probe restores the compact plan");
    }

    #[test]
    fn floor_declines_thresholds_that_keep_everything_and_float_scores() {
        let q = Quantization {
            scale: 0.1,
            zero_point: 0,
        };
        let (out, act) = (DecodeDtype::F32, Activation::Sigmoid);
        for t in [0.0, -0.5, f32::NAN] {
            assert_eq!(
                floor(&mut scratch(), DType::I8, out, q, act, t),
                None,
                "t={t}"
            );
        }
        assert_eq!(floor(&mut scratch(), DType::F32, out, q, act, 0.25), None);
    }

    #[test]
    fn select_keeps_rows_with_any_class_at_floor() {
        let scores: [i8; 9] = [-5, -9, -9, -9, -9, -9, -9, 3, -9];
        let mut anchors = vec![42];
        select_anchors(&InputView::I8(&scores), 3, -5, &mut anchors).unwrap();
        assert_eq!(anchors, vec![0, 2]);
    }

    /// The block scan keeps exactly the rows a row-by-row scan keeps, for
    /// rows that straddle blocks, blocks holding several rows, and floors
    /// outside the raw type's range.
    #[test]
    fn select_matches_a_row_by_row_scan() {
        let mut x = 0x1234_5678_u32;
        let mut next = move || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x >> 8
        };
        for nc in [1, 3, 4, 10, 80, 100] {
            let rows = 37;
            let scores: Vec<i8> = (0..rows * nc)
                .map(|_| {
                    if next() % 61 == 0 {
                        (next() % 128) as i8
                    } else {
                        -100
                    }
                })
                .collect();
            for floor in [-200, -100, -99, 0, 50, 127, 128, 500] {
                let want: Vec<u32> = (0..rows as u32)
                    .filter(|&a| {
                        let a = a as usize;
                        scores[a * nc..(a + 1) * nc]
                            .iter()
                            .any(|&v| i32::from(v) >= floor)
                    })
                    .collect();
                let mut got = Vec::new();
                select_anchors(&InputView::I8(&scores), nc, floor, &mut got).unwrap();
                assert_eq!(got, want, "nc={nc} floor={floor}");
                let unsigned: Vec<u16> = scores
                    .iter()
                    .map(|&v| (i32::from(v) + 128) as u16)
                    .collect();
                select_anchors(&InputView::U16(&unsigned), nc, floor + 128, &mut got).unwrap();
                assert_eq!(got, want, "u16 nc={nc} floor={floor}");
            }
        }
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
