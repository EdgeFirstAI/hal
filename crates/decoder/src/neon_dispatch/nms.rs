// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Greedy NMS on the NEON tiers: `prim::neon_nms` over the boxes gathered into padded
//! coordinate arrays, with the same keep/suppress decisions as the scalar loops in `float.rs`
//! and `byte.rs`.

use super::Isa;
use crate::BoundingBox;

/// Boxes below which NMS runs the scalar loop on every tier: gathering them costs more than the
/// loop (measured crossover on a Cortex-A55 with clustered boxes).
pub(crate) const NMS_MIN_BOXES: usize = 16;

/// Boxes below which the public NMS functions run the scalar loop without resolving the tier:
/// reading `EDGEFIRST_DECODER_FORCE_KERNEL` moves their crossover up from [`NMS_MIN_BOXES`].
pub(crate) const NMS_LOOKUP_MIN_BOXES: usize = 20;

/// The tier a public NMS over `n` boxes runs: [`Isa::current_or_scalar`], or the scalar code
/// below [`NMS_LOOKUP_MIN_BOXES`].
pub(crate) fn nms_isa(n: usize) -> Isa {
    if n < NMS_LOOKUP_MIN_BOXES {
        Isa::SCALAR
    } else {
        Isa::current_or_scalar()
    }
}

/// Boxes up to which NMS sorts in place and tests every pair in one f32 pass over stack arrays:
/// grouping by label and FP16 only repay themselves above it.
const SMALL_NMS_BOXES: usize = 64;

/// Boxes from which the score sort runs on the rayon pool.
const PAR_SORT_MIN_BOXES: usize = 1 << 14;

/// Sort `boxes` by descending `score` in IEEE total order, equal scores keeping their order.
pub(crate) fn sort_by_score_desc<T>(boxes: &mut Vec<T>, score: impl Fn(&T) -> f32) {
    use rayon::slice::ParallelSliceMut;
    let n = boxes.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_NMS_BOXES || u32::try_from(n).is_err() {
        boxes.sort_by(|a, b| score(b).total_cmp(&score(a)));
        return;
    }
    // Descending score in the high half, the position in the low half: unique keys, so an
    // unstable sort gives the stable order.
    let mut keys: Vec<u64> = boxes
        .iter()
        .enumerate()
        .map(|(k, b)| {
            let bits = score(b).to_bits();
            let ascending = if bits >> 31 == 0 {
                bits | 0x8000_0000
            } else {
                !bits
            };
            (u64::from(!ascending) << 32) | k as u64
        })
        .collect();
    if n >= PAR_SORT_MIN_BOXES {
        keys.par_sort_unstable();
    } else {
        keys.sort_unstable();
    }
    let mut slots: Vec<Option<T>> = boxes.drain(..).map(Some).collect();
    boxes.extend(
        keys.iter()
            .filter_map(|&key| slots.get_mut((key & 0xFFFF_FFFF) as usize)?.take()),
    );
}

/// Labels a class-aware NMS groups with a counting sort: above it (relative to the box count)
/// the label table would cost more than sorting the boxes.
#[cfg(target_arch = "aarch64")]
const COUNTING_SORT_SPARE_LABELS: usize = 1024;

/// Box indices grouped by label, each group in ascending index order, and the end of each
/// group in that list.
#[cfg(target_arch = "aarch64")]
fn group_by_label<T>(boxes: &[T], label: impl Fn(&T) -> usize) -> (Vec<u32>, Vec<u32>) {
    let n = boxes.len();
    let max = boxes.iter().map(&label).max().unwrap_or(0);
    let mut order = vec![0u32; n];
    let mut ends = Vec::new();
    if max <= 4 * n + COUNTING_SORT_SPARE_LABELS {
        let mut next = vec![0u32; max + 2];
        for b in boxes {
            next[label(b) + 1] += 1;
        }
        for l in 0..=max {
            next[l + 1] += next[l];
            if next[l + 1] != next[l] {
                ends.push(next[l + 1]);
            }
        }
        for (k, b) in boxes.iter().enumerate() {
            let slot = &mut next[label(b)];
            order[*slot as usize] = k as u32;
            *slot += 1;
        }
    } else {
        let mut keyed: Vec<(usize, u32)> = boxes
            .iter()
            .enumerate()
            .map(|(k, b)| (label(b), k as u32))
            .collect();
        keyed.sort_unstable();
        for (k, pair) in keyed.iter().enumerate() {
            order[k] = pair.1;
            if keyed.get(k + 1).is_none_or(|next| next.0 != pair.0) {
                ends.push(k as u32 + 1);
            }
        }
    }
    (order, ends)
}

/// The power of two that brings coordinates of magnitude up to `max` into `(32, 64]`, where
/// box areas and their sums are finite and normal in f16; `None` for a zero or non-finite `max`.
#[cfg(target_arch = "aarch64")]
fn f16_coordinate_scale(max: f32) -> Option<f32> {
    if !(max.is_finite() && max >= f32::MIN_POSITIVE) {
        return None;
    }
    let scale = 2f32.powi(6 - max.log2().ceil() as i32);
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

/// How a group's overlap tests run.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct Overlap {
    iou: f32,
    /// Survivors after which a pass stops.
    cap: usize,
    /// The coordinate scale of FP16 tests; `None` for f32 tests.
    f16_scale: Option<f32>,
}

/// Coordinate arrays and flags of one group of boxes, reused from group to group.
#[cfg(target_arch = "aarch64")]
#[derive(Default)]
struct Gathered {
    coords: Vec<f32>,
    flags: Vec<u32>,
    half: Vec<half::f16>,
    half_flags: Vec<u16>,
}

#[cfg(target_arch = "aarch64")]
impl Gathered {
    /// Greedy NMS over the boxes `group` indexes, in that order, updating `dead`, with the
    /// tests `overlap` describes.
    fn suppress<T>(
        &mut self,
        boxes: &[T],
        group: impl ExactSizeIterator<Item = usize> + Clone,
        bbox: &impl Fn(&T) -> &BoundingBox,
        dead: &mut [bool],
        overlap: Overlap,
    ) -> bool {
        let Overlap {
            iou,
            cap,
            f16_scale,
        } = overlap;
        use edgefirst_kernels::prim::neon_convert::narrow_f32_to_f16_neon;
        use edgefirst_kernels::prim::neon_nms::{
            nms_greedy_neon, nms_greedy_neon_fp16, LANES, LANES_FP16,
        };
        let m = group.len();
        let lanes = if f16_scale.is_some() {
            LANES_FP16
        } else {
            LANES
        };
        let np = m.next_multiple_of(lanes);
        let scale = f16_scale.unwrap_or(1.0);
        self.coords.clear();
        self.coords.resize(4 * np, 0.0);
        let (xmin, rest) = self.coords.split_at_mut(np);
        let (ymin, rest) = rest.split_at_mut(np);
        let (xmax, ymax) = rest.split_at_mut(np);
        for (k, i) in group.clone().enumerate() {
            let Some(r) = boxes.get(i).map(bbox) else {
                return false;
            };
            xmin[k] = r.xmin * scale;
            ymin[k] = r.ymin * scale;
            xmax[k] = r.xmax * scale;
            ymax[k] = r.ymax * scale;
        }
        let is_dead = |k: usize, i: usize| k >= m || dead.get(i).copied().unwrap_or(true);
        let ran = if f16_scale.is_some() {
            self.half.clear();
            self.half.resize(5 * np, half::f16::ZERO);
            let (coords, area) = self.half.split_at_mut(4 * np);
            self.half_flags.clear();
            self.half_flags.resize(np, !0);
            for (k, i) in group.clone().enumerate() {
                self.half_flags[k] = if is_dead(k, i) { !0 } else { 0 };
            }
            // SAFETY: NEON is part of the aarch64 baseline, and a scale is only given on a
            // CPU with FEAT_FP16.
            unsafe {
                narrow_f32_to_f16_neon(&self.coords, coords);
                let (x0, rest) = coords.split_at(np);
                let (y0, rest) = rest.split_at(np);
                let (x1, y1) = rest.split_at(np);
                let iou = half::f16::from_f32(iou);
                nms_greedy_neon_fp16([x0, y0, x1, y1], area, iou, cap, &mut self.half_flags)
            }
        } else {
            self.flags.clear();
            self.flags.resize(np, !0);
            for (k, i) in group.clone().enumerate() {
                self.flags[k] = if is_dead(k, i) { !0 } else { 0 };
            }
            let coords = [&*xmin, &*ymin, &*xmax, &*ymax];
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { nms_greedy_neon(coords, None, iou, cap, &mut self.flags) }
        };
        if !ran {
            return false;
        }
        for (k, i) in group.enumerate() {
            let flagged = if f16_scale.is_some() {
                self.half_flags[k] != 0
            } else {
                self.flags[k] != 0
            };
            if let Some(d) = dead.get_mut(i) {
                *d = flagged;
            }
        }
        true
    }
}

/// [`nms_dead`] in one f32 pass over every box, labels compared in the kernel, on stack arrays
/// for at most [`SMALL_NMS_BOXES`] boxes. `None` for a label above `u32::MAX`.
#[cfg(target_arch = "aarch64")]
fn single_pass_dead<T>(
    boxes: &[T],
    bbox: impl Fn(&T) -> &BoundingBox,
    class_aware: bool,
    label: impl Fn(&T) -> usize,
    dead: impl Fn(&T) -> bool,
    iou: f32,
    cap: usize,
) -> Option<Vec<bool>> {
    use edgefirst_kernels::prim::neon_nms::{nms_greedy_neon, LANES};
    let n = boxes.len();
    let np = n.next_multiple_of(LANES);
    let mut stack_coords = [0f32; 4 * SMALL_NMS_BOXES];
    let mut stack_words = [0u32; 2 * SMALL_NMS_BOXES];
    let (mut heap_coords, mut heap_words);
    let (coords, words) = if np <= SMALL_NMS_BOXES {
        (&mut stack_coords[..4 * np], &mut stack_words[..2 * np])
    } else {
        heap_coords = vec![0f32; 4 * np];
        heap_words = vec![0u32; 2 * np];
        (&mut heap_coords[..], &mut heap_words[..])
    };
    let (xmin, rest) = coords.split_at_mut(np);
    let (ymin, rest) = rest.split_at_mut(np);
    let (xmax, ymax) = rest.split_at_mut(np);
    let (flags, labels) = words.split_at_mut(np);
    flags.fill(!0);
    for (k, b) in boxes.iter().enumerate() {
        let r = bbox(b);
        (xmin[k], ymin[k], xmax[k], ymax[k]) = (r.xmin, r.ymin, r.xmax, r.ymax);
        flags[k] = if dead(b) { !0 } else { 0 };
        if class_aware {
            labels[k] = u32::try_from(label(b)).ok()?;
        }
    }
    let labels = class_aware.then_some(&*labels);
    let coords = [&*xmin, &*ymin, &*xmax, &*ymax];
    // SAFETY: NEON is part of the aarch64 baseline.
    let ran = unsafe { nms_greedy_neon(coords, labels, iou, cap, flags) };
    ran.then(|| flags.iter().take(n).map(|&f| f != 0).collect())
}

/// The suppression flags of a greedy NMS over `boxes`, sorted by descending score, on `isa`.
///
/// Box `k` is flagged when `dead(&boxes[k])` holds (the scalar loops' "already suppressed"
/// score) or an earlier unflagged box overlaps it (`jaccard(b_k, b_i, iou)`, and with its label
/// when `class_aware`), the scan stopping once `cap` boxes have survived; the caller outputs
/// the first `cap` unflagged boxes that `kept` accepts, and those equal the scalar loop's.
///
/// More than [`SMALL_NMS_BOXES`] boxes that are each dead or kept run class-aware NMS one label
/// at a time (each label's pass stopping at `cap` survivors of its own), and with
/// [`Isa::f16_nms`] and finite coordinates a pass over more than [`SMALL_NMS_BOXES`] boxes runs
/// its overlap tests in FP16 on coordinates scaled by a power of two. A live box the caller does not keep (a NaN score) takes a survivor's
/// place in the scalar scan, so those boxes run the one pass over every box instead.
///
/// `None` when the caller runs its scalar loop: the scalar tier, fewer than [`NMS_MIN_BOXES`]
/// boxes, or a label above `u32::MAX` in the one pass over every box.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nms_dead<T>(
    isa: Isa,
    boxes: &[T],
    bbox: impl Fn(&T) -> &BoundingBox,
    class_aware: bool,
    label: impl Fn(&T) -> usize,
    dead: impl Fn(&T) -> bool,
    kept: impl Fn(&T) -> bool,
    iou: f32,
    cap: usize,
) -> Option<Vec<bool>> {
    if isa.is_scalar() || boxes.len() < NMS_MIN_BOXES || u32::try_from(boxes.len()).is_err() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let n = boxes.len();
        if n <= SMALL_NMS_BOXES || !boxes.iter().all(|b| dead(b) || kept(b)) {
            return single_pass_dead(boxes, bbox, class_aware, label, dead, iou, cap);
        }
        // The coordinate scale of the FP16 tests, found only when a group is large enough to
        // take them.
        let f16_scale = || {
            let max = boxes.iter().map(&bbox).fold(0f32, |m, r| {
                m.max(r.xmin.abs())
                    .max(r.ymin.abs())
                    .max(r.xmax.abs())
                    .max(r.ymax.abs())
            });
            let finite = boxes.iter().map(&bbox).all(|r| {
                r.xmin.is_finite() && r.ymin.is_finite() && r.xmax.is_finite() && r.ymax.is_finite()
            });
            finite.then(|| f16_coordinate_scale(max)).flatten()
        };
        if !class_aware {
            let overlap = Overlap {
                iou,
                cap,
                f16_scale: isa.f16_nms().then(f16_scale).flatten(),
            };
            if overlap.f16_scale.is_none() {
                // One group of f32 tests is the one pass.
                return single_pass_dead(boxes, bbox, false, label, dead, iou, cap);
            }
            let mut flags: Vec<bool> = boxes.iter().map(&dead).collect();
            let ran = Gathered::default().suppress(boxes, 0..n, &bbox, &mut flags, overlap);
            return ran.then_some(flags);
        }
        let (order, ends) = group_by_label(boxes, label);
        let mut largest = 0u32;
        let mut start = 0u32;
        for &end in &ends {
            largest = largest.max(end - start);
            start = end;
        }
        let fp16 = isa.f16_nms() && largest as usize > SMALL_NMS_BOXES;
        let f16_scale = fp16.then(f16_scale).flatten();
        let mut flags: Vec<bool> = boxes.iter().map(&dead).collect();
        let mut gathered = Gathered::default();
        let mut start = 0usize;
        let ran = ends.iter().all(|&end| {
            let group = order.get(start..end as usize).unwrap_or(&[]);
            start = end as usize;
            // A small group gains nothing from eight boxes per vector and keeps the f32 tests.
            let overlap = Overlap {
                iou,
                cap,
                f16_scale: f16_scale.filter(|_| group.len() > SMALL_NMS_BOXES),
            };
            group.len() < 2
                || gathered.suppress(
                    boxes,
                    group.iter().map(|&i| i as usize),
                    &bbox,
                    &mut flags,
                    overlap,
                )
        });
        ran.then_some(flags)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (boxes, bbox, class_aware, label, dead, kept, iou, cap);
        None
    }
}

/// `[jaccard(a, b[k], iou); 4]` as one NEON vector on `isa`; `None` when the caller runs the
/// scalar tests (scalar tier).
pub(crate) fn jaccard4_kernel(
    isa: Isa,
    a: &BoundingBox,
    b: &[BoundingBox; 4],
    iou: f32,
) -> Option<[bool; 4]> {
    #[cfg(test)]
    jaccard4_ran::record(!isa.is_scalar());
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let b = b.map(<[f32; 4]>::from);
        // SAFETY: NEON is part of the aarch64 baseline.
        Some(unsafe { edgefirst_kernels::prim::neon_nms::jaccard4_neon((*a).into(), &b, iou) })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (a, b, iou);
        None
    }
}

/// Whether the calling thread's last [`jaccard4_kernel`] call ran on a NEON tier.
#[cfg(test)]
pub(crate) mod jaccard4_ran {
    use std::cell::Cell;

    thread_local! {
        static RAN: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub(super) fn record(neon: bool) {
        RAN.with(|r| r.set(Some(neon)));
    }

    /// The last call's tier since the previous `take`, `Some(true)` for NEON.
    pub(crate) fn take() -> Option<bool> {
        RAN.with(Cell::take)
    }
}

/// The boxes of `boxes` not flagged in `dead` that `keep` accepts, in order, at most `cap`.
pub(crate) fn keep_unflagged<T>(
    boxes: Vec<T>,
    dead: Vec<bool>,
    keep: impl Fn(&T) -> bool,
    cap: usize,
) -> Vec<T> {
    boxes
        .into_iter()
        .zip(dead)
        .filter(|(b, d)| !d && keep(b))
        .map(|(b, _)| b)
        .take(cap)
        .collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
