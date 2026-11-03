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
use super::kernels::CpuFeatures;
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
    /// This frame's score quantization and floor.
    pub(crate) frame_floor: (Quantization, i32),
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
            frame_floor: (Quantization::identity(), i32::MAX),
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
    /// The largest of `s` with NEON, [`RawScore::LOWEST`] when empty.
    #[cfg(target_arch = "aarch64")]
    fn max_neon(s: &[Self]) -> Self;
    /// Bit `b` set when [`SCAN_BLOCK`] block `b` of `sup` holds a value at
    /// or above `floor`.
    #[cfg(target_arch = "aarch64")]
    fn block_hits_neon(sup: &[Self], floor: Self) -> u32;
}

/// `max_neon` for one raw type: four accumulators over 64-byte steps, then
/// 16-byte steps, a lane reduction and a scalar tail.
#[cfg(target_arch = "aarch64")]
macro_rules! raw_score {
    ($t:ty, $lanes:expr, $dup:ident, $ld:ident, $max:ident, $maxv:ident) => {
        impl RawScore for $t {
            const LOWEST: Self = <$t>::MIN;
            #[inline]
            fn max_neon(s: &[Self]) -> Self {
                use std::arch::aarch64::*;
                // SAFETY: NEON is the aarch64 baseline; each load reads a whole `chunks_exact` chunk.
                unsafe {
                    let mut acc = [$dup(<$t>::MIN); 4];
                    let mut wide = s.chunks_exact(4 * $lanes);
                    for c in &mut wide {
                        let p = c.as_ptr();
                        acc[0] = $max(acc[0], $ld(p));
                        acc[1] = $max(acc[1], $ld(p.add($lanes)));
                        acc[2] = $max(acc[2], $ld(p.add(2 * $lanes)));
                        acc[3] = $max(acc[3], $ld(p.add(3 * $lanes)));
                    }
                    let mut narrow = wide.remainder().chunks_exact($lanes);
                    for c in &mut narrow {
                        acc[0] = $max(acc[0], $ld(c.as_ptr()));
                    }
                    let m = $maxv($max($max(acc[0], acc[1]), $max(acc[2], acc[3])));
                    narrow.remainder().iter().fold(m, |m, &v| m.max(v))
                }
            }
            #[inline]
            fn block_hits_neon(sup: &[Self], floor: Self) -> u32 {
                use std::arch::aarch64::*;
                if sup.len() != SCAN_SUPER_BLOCK {
                    return sup
                        .chunks(SCAN_BLOCK)
                        .enumerate()
                        .filter(|(_, b)| Self::max_neon(b) >= floor)
                        .fold(0, |bits, (i, _)| bits | 1 << i);
                }
                // Ascending loads keep the hardware prefetcher streaming; most super blocks stop here.
                if Self::max_neon(sup) < floor {
                    return 0;
                }
                // SAFETY: NEON is the aarch64 baseline; each load lies inside the whole super block `sup`.
                unsafe {
                    let p = sup.as_ptr();
                    let lowest = $dup(<$t>::MIN);
                    let (mut a0, mut a1, mut a2, mut a3) = (lowest, lowest, lowest, lowest);
                    // The block re-read hits L1; the four blocks advance in step so their chains overlap.
                    let mut k = 0;
                    while k < SCAN_BLOCK {
                        a0 = $max(a0, $ld(p.add(k)));
                        a1 = $max(a1, $ld(p.add(SCAN_BLOCK + k)));
                        a2 = $max(a2, $ld(p.add(2 * SCAN_BLOCK + k)));
                        a3 = $max(a3, $ld(p.add(3 * SCAN_BLOCK + k)));
                        k += $lanes;
                    }
                    let hit = |a| u32::from($maxv(a) >= floor);
                    hit(a0) | hit(a1) << 1 | hit(a2) << 2 | hit(a3) << 3
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
raw_score!(i8, 16, vdupq_n_s8, vld1q_s8, vmaxq_s8, vmaxvq_s8);
#[cfg(target_arch = "aarch64")]
raw_score!(u8, 16, vdupq_n_u8, vld1q_u8, vmaxq_u8, vmaxvq_u8);
#[cfg(target_arch = "aarch64")]
raw_score!(i16, 8, vdupq_n_s16, vld1q_s16, vmaxq_s16, vmaxvq_s16);
#[cfg(target_arch = "aarch64")]
raw_score!(u16, 8, vdupq_n_u16, vld1q_u16, vmaxq_u16, vmaxvq_u16);

#[cfg(not(target_arch = "aarch64"))]
macro_rules! raw_score {
    ($t:ty) => {
        impl RawScore for $t {
            const LOWEST: Self = <$t>::MIN;
        }
    };
}

#[cfg(not(target_arch = "aarch64"))]
raw_score!(i8);
#[cfg(not(target_arch = "aarch64"))]
raw_score!(u8);
#[cfg(not(target_arch = "aarch64"))]
raw_score!(i16);
#[cfg(not(target_arch = "aarch64"))]
raw_score!(u16);

/// Raw scores are tested for a value at the floor this many at a time; only
/// a block that holds one has its rows examined.
const SCAN_BLOCK: usize = 64;

/// The NEON scan first tests this many raw scores at once and skips them all
/// when none reaches the floor.
#[cfg(target_arch = "aarch64")]
const SCAN_SUPER_BLOCK: usize = 4 * SCAN_BLOCK;

/// Append every anchor from `next` through the one holding element `end - 1`
/// of `rows` whose row reaches `floor`, starting at the row of element
/// `start`; returns the anchor after the last one examined. A row spanning
/// two blocks that both hold a passing value is examined once.
fn examine<T: RawScore>(
    rows: &[T],
    nc: usize,
    floor: T,
    (start, end): (usize, usize),
    next: usize,
    anchors: &mut Vec<u32>,
) -> usize {
    let last = (end - 1) / nc;
    for a in (start / nc).max(next)..=last {
        let row = rows.get(a * nc..(a + 1) * nc);
        if row.is_some_and(|r| r.iter().any(|&v| v >= floor)) {
            anchors.push(a as u32);
        }
    }
    last + 1
}

/// Append to `anchors` every anchor whose largest raw class score is at or
/// above `floor`. `input` holds NHWC `[anchors, num_classes]` score rows.
/// The selection is the same on every tier; NEON only finds the blocks to
/// examine faster.
pub(crate) fn select_anchors(
    input: &InputView<'_>,
    num_classes: usize,
    floor: i32,
    anchors: &mut Vec<u32>,
    features: &CpuFeatures,
) -> DecoderResult<()> {
    fn scan<T: RawScore>(
        rows: &[T],
        nc: usize,
        floor: i32,
        anchors: &mut Vec<u32>,
        features: &CpuFeatures,
    ) {
        let floor = match T::try_from(floor) {
            Ok(f) => f,
            Err(_) if floor > 0 => return,
            Err(_) => {
                anchors.extend(0..(rows.len() / nc) as u32);
                return;
            }
        };
        #[cfg(target_arch = "aarch64")]
        if features.neon_baseline {
            scan_neon(rows, nc, floor, anchors);
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = features;
        let mut next = 0;
        for (b, block) in rows.chunks(SCAN_BLOCK).enumerate() {
            if block.iter().fold(T::LOWEST, |m, &v| m.max(v)) < floor {
                continue;
            }
            let start = b * SCAN_BLOCK;
            next = examine(rows, nc, floor, (start, start + block.len()), next, anchors);
        }
    }
    #[cfg(target_arch = "aarch64")]
    fn scan_neon<T: RawScore>(rows: &[T], nc: usize, floor: T, anchors: &mut Vec<u32>) {
        let mut next = 0;
        for (s, sup) in rows.chunks(SCAN_SUPER_BLOCK).enumerate() {
            let hits = T::block_hits_neon(sup, floor);
            if hits == 0 {
                continue;
            }
            for (b, block) in sup.chunks(SCAN_BLOCK).enumerate() {
                if hits & 1 << b == 0 {
                    continue;
                }
                let start = s * SCAN_SUPER_BLOCK + b * SCAN_BLOCK;
                next = examine(rows, nc, floor, (start, start + block.len()), next, anchors);
            }
        }
    }
    anchors.clear();
    if num_classes == 0 {
        return Ok(());
    }
    match input {
        InputView::I8(s) => scan(s, num_classes, floor, anchors, features),
        InputView::U8(s) => scan(s, num_classes, floor, anchors, features),
        InputView::I16(s) => scan(s, num_classes, floor, anchors, features),
        InputView::U16(s) => scan(s, num_classes, floor, anchors, features),
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
        let f = crate::per_scale::kernels::decoder_cpu_features().unwrap();
        let dispatch = ScoreLevelDispatch::select(dtype, out, &f).unwrap();
        raw_score_floor(s, &dispatch, dtype, out, q, act, t).unwrap()
    }

    /// Kernel scores of every u8 raw value, as the candidate filter sees them.
    fn u8_scores(out: DecodeDtype, q: Quantization, act: Activation) -> Vec<f32> {
        let f = crate::per_scale::kernels::decoder_cpu_features().unwrap();
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
        for f in tiers() {
            select_anchors(&InputView::I8(&scores), 3, -5, &mut anchors, &f).unwrap();
            assert_eq!(anchors, vec![0, 2]);
        }
    }

    /// Every tier this CPU can run.
    fn tiers() -> Vec<CpuFeatures> {
        ["scalar", "neon", "neon_fp16", "neon_dotprod"]
            .into_iter()
            .filter_map(|t| crate::per_scale::kernels::decoder_cpu_features_from(Some(t)).ok())
            .collect()
    }

    /// The anchors a row-by-row scan keeps.
    fn row_by_row(scores: &[i32], nc: usize, floor: i32) -> Vec<u32> {
        (0..(scores.len() / nc) as u32)
            .filter(|&a| {
                let a = a as usize;
                scores[a * nc..(a + 1) * nc].iter().any(|&v| v >= floor)
            })
            .collect()
    }

    /// The block scan keeps exactly the rows a row-by-row scan keeps on
    /// every tier and raw dtype, for rows that straddle blocks, blocks
    /// holding several rows, levels spanning many blocks, dense and sparse
    /// passing values, and floors outside the raw type's range.
    #[test]
    fn select_matches_a_row_by_row_scan() {
        let mut x = 0x1234_5678_u32;
        let mut next = move || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x >> 8
        };
        for nc in [1, 3, 4, 10, 80, 100] {
            for (rows, density) in [(37, 61), (300, 61), (300, 3), (1000, 997)] {
                let scores: Vec<i8> = (0..rows * nc)
                    .map(|_| {
                        if next() % density == 0 {
                            (next() % 128) as i8
                        } else {
                            -100 - (next() % 29) as i8
                        }
                    })
                    .collect();
                let wide: Vec<i32> = scores.iter().map(|&v| i32::from(v)).collect();
                for floor in [-200, -129, -128, -100, -99, 0, 50, 127, 128, 500] {
                    let want = row_by_row(&wide, nc, floor);
                    let label = format!("nc={nc} rows={rows} density={density} floor={floor}");
                    for f in tiers() {
                        let mut got = vec![7];
                        select_anchors(&InputView::I8(&scores), nc, floor, &mut got, &f).unwrap();
                        assert_eq!(got, want, "i8 {label} {f:?}");
                        let u8s: Vec<u8> = scores.iter().map(|&v| (v as u8) ^ 0x80).collect();
                        select_anchors(&InputView::U8(&u8s), nc, floor + 128, &mut got, &f)
                            .unwrap();
                        assert_eq!(got, want, "u8 {label} {f:?}");
                        let i16s: Vec<i16> = wide.iter().map(|&v| (v * 256) as i16).collect();
                        let i16_floor = (floor * 256).clamp(-40_000, 40_000);
                        let want16 = row_by_row(
                            &i16s.iter().map(|&v| i32::from(v)).collect::<Vec<_>>(),
                            nc,
                            i16_floor,
                        );
                        select_anchors(&InputView::I16(&i16s), nc, i16_floor, &mut got, &f)
                            .unwrap();
                        assert_eq!(got, want16, "i16 {label} {f:?}");
                        let u16s: Vec<u16> = wide.iter().map(|&v| (v + 128) as u16 * 257).collect();
                        let u16_floor = (floor + 128) * 257;
                        let want16 = row_by_row(
                            &u16s.iter().map(|&v| i32::from(v)).collect::<Vec<_>>(),
                            nc,
                            u16_floor,
                        );
                        select_anchors(&InputView::U16(&u16s), nc, u16_floor, &mut got, &f)
                            .unwrap();
                        assert_eq!(got, want16, "u16 {label} {f:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn select_rejects_float_scores() {
        let scores = [0.5_f32; 4];
        let mut anchors = Vec::new();
        let f = CpuFeatures::default();
        assert!(select_anchors(&InputView::F32(&scores), 2, 0, &mut anchors, &f).is_err());
    }

    #[test]
    fn pack_pads_to_whole_groups_with_the_last_anchor() {
        let (gx, gy) = edgefirst_kernels::prim::grids::make_anchor_grid(8, 4);
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
