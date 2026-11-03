// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CPU kernels for the semantic segmentation decoder.
//!
//! Scores are read in their stored type. A positive per-tensor quantization
//! scale preserves both the argmax and bilinear interpolation (the weights
//! sum to 1), so integer inputs are never dequantized except where a real
//! value is needed: the single-channel threshold and the confidence map.
//!
//! Each kernel takes the decode's [`Isa`]: on the NEON tiers rows run the shared kernels in
//! `edgefirst_kernels::prim::neon_semantic` (through `neon_dispatch::semantic`), bit-identical to
//! the scalar code here except the sigmoid confidence (within `2^-21` relative); layouts and
//! types without a kernel, and the scalar tier, run the scalar code.

use edgefirst_kernels::prim::softmax::exp_nonpositive;
use half::f16;
use rayon::prelude::*;

use crate::neon_dispatch::semantic::{self as nd, ScoreRow, Taps};
use crate::neon_dispatch::{Isa, Ran};

/// Output rows handed to one rayon task at minimum.
const MIN_ROWS_PER_TASK: usize = 8;

/// A score element the kernels can compare and interpolate.
pub(super) trait Score: Copy + Send + Sync + PartialOrd + 'static {
    fn to_f32(self) -> f32;
}

macro_rules! impl_score {
    ($($t:ty),*) => {$(
        impl Score for $t {
            #[inline(always)]
            fn to_f32(self) -> f32 {
                self as f32
            }
        }
    )*};
}
impl_score!(u8, i8, u16, i16, f32);

impl Score for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

/// A class-map output element.
pub(super) trait ClassOut: Copy + Send + Sync + Default + 'static {
    /// Number of distinct classes the type holds.
    const CAPACITY: usize;
    fn from_class(c: u16) -> Self;
}

impl ClassOut for u8 {
    const CAPACITY: usize = 1 << 8;
    #[inline(always)]
    fn from_class(c: u16) -> Self {
        c as u8
    }
}

impl ClassOut for u16 {
    const CAPACITY: usize = 1 << 16;
    #[inline(always)]
    fn from_class(c: u16) -> Self {
        c
    }
}

/// A confidence-map output element.
pub(super) trait ConfOut: Copy + Send + Sync + 'static {
    fn from_conf(v: f32) -> Self;
}

impl ConfOut for f32 {
    #[inline(always)]
    fn from_conf(v: f32) -> Self {
        v
    }
}

impl ConfOut for u8 {
    #[inline(always)]
    fn from_conf(v: f32) -> Self {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }
}

/// Element strides of a score map: `offset = c*cs + y*ys + x*xs`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Grid {
    pub classes: usize,
    pub height: usize,
    pub width: usize,
    pub cs: usize,
    pub ys: usize,
    pub xs: usize,
}

/// Affine map from stored values to real values: `(v - zero_point) * scale`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Dequant {
    pub scale: f32,
    pub zero_point: f32,
}

impl Dequant {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        zero_point: 0.0,
    };

    #[inline(always)]
    fn real(self, v: f32) -> f32 {
        (v - self.zero_point) * self.scale
    }

    /// The stored value equal to the real value `v`.
    pub fn stored(self, v: f32) -> f32 {
        v / self.scale + self.zero_point
    }
}

/// Source sample positions for one output axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Tap {
    pub lo: u32,
    pub hi: u32,
    pub frac: f32,
}

/// Bilinear taps for `out` samples of the source span `[roi0, roi0 + roi_len)`
/// on an axis of `len` pixels, following PyTorch `interpolate` semantics.
///
/// `hi == lo` whenever the upper tap has no weight, so the corners a pixel
/// actually reads are the only ones compared by the early-out.
pub(super) fn bilinear_taps(
    out: usize,
    len: usize,
    roi0: f32,
    roi_len: f32,
    align_corners: bool,
) -> Vec<Tap> {
    let last = len.saturating_sub(1) as f32;
    (0..out)
        .map(|d| {
            let src = if align_corners {
                if out > 1 {
                    roi0 + d as f32 * ((roi_len - 1.0) / (out - 1) as f32)
                } else {
                    roi0
                }
            } else {
                roi0 + (roi_len / out as f32) * (d as f32 + 0.5) - 0.5
            };
            let src = src.clamp(0.0, last);
            let lo = src as u32;
            let frac = src - lo as f32;
            let hi = if frac > 0.0 && (lo as f32) < last {
                lo + 1
            } else {
                lo
            };
            let frac = if hi == lo { 0.0 } else { frac };
            Tap { lo, hi, frac }
        })
        .collect()
}

/// Nearest source index for `out` samples of `[roi0, roi0 + roi_len)`, using
/// pixel centres so the class map stays aligned with the bilinear path.
pub(super) fn nearest_taps(out: usize, len: usize, roi0: f32, roi_len: f32) -> Vec<u32> {
    let last = len.saturating_sub(1) as f32;
    (0..out)
        .map(|d| (roi0 + (roi_len / out as f32) * (d as f32 + 0.5)).clamp(0.0, last) as u32)
        .collect()
}

/// The scores from row `y`'s first one to the end of `data`.
fn row_scores<T>(data: &[T], g: Grid, y: usize) -> &[T] {
    data.get(y * g.ys..).unwrap_or_default()
}

/// Row `y` of a single channel when its pixels are contiguous.
fn contiguous_row<T>(data: &[T], g: Grid, y: usize) -> Option<&[T]> {
    (g.xs == 1)
        .then(|| row_scores(data, g, y).get(..g.width))
        .flatten()
}

/// Per-pixel argmax over the class axis; ties go to the lowest index.
pub(super) fn argmax_map<T: Score>(isa: Isa, data: &[T], g: Grid) -> Vec<u16> {
    let mut out = vec![0u16; g.height * g.width];
    let layout = ScoreRow::of(g.classes, g.cs, g.xs);
    out.par_chunks_mut(g.width)
        .enumerate()
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(y, row)| {
            if let Some(layout) = layout {
                let src = row_scores(data, g, y);
                if nd::argmax_row(isa, src, layout, g.classes, row) != Ran::Scalar {
                    return;
                }
            }
            if g.xs == 1 {
                argmax_row_planar(data, g, y, row)
            } else {
                for (x, slot) in row.iter_mut().enumerate() {
                    *slot = argmax_pixel(data, g, y * g.ys + x * g.xs);
                }
            }
        });
    out
}

/// Running maximum across contiguous class planes, one row at a time.
fn argmax_row_planar<T: Score>(data: &[T], g: Grid, y: usize, row: &mut [u16]) {
    let base = y * g.ys;
    let mut best: Vec<T> = data[base..base + g.width].to_vec();
    for c in 1..g.classes {
        let plane = &data[base + c * g.cs..base + c * g.cs + g.width];
        for ((b, i), &v) in best.iter_mut().zip(row.iter_mut()).zip(plane) {
            if v > *b {
                *b = v;
                *i = c as u16;
            }
        }
    }
}

#[inline]
fn argmax_pixel<T: Score>(data: &[T], g: Grid, base: usize) -> u16 {
    let mut best = data[base];
    let mut idx = 0u16;
    for c in 1..g.classes {
        let v = data[base + c * g.cs];
        if v > best {
            best = v;
            idx = c as u16;
        }
    }
    idx
}

/// Single-channel foreground map: 1 where the stored value exceeds `cut`.
pub(super) fn threshold_map<T: Score>(isa: Isa, data: &[T], g: Grid, cut: f32) -> Vec<u16> {
    let mut out = vec![0u16; g.height * g.width];
    for (y, row) in out.chunks_mut(g.width).enumerate() {
        if let Some(src) = contiguous_row(data, g, y) {
            if nd::threshold_row(isa, src, cut, row) != Ran::Scalar {
                continue;
            }
        }
        for (x, slot) in row.iter_mut().enumerate() {
            *slot = u16::from(data[y * g.ys + x * g.xs].to_f32() > cut);
        }
    }
    out
}

/// How the confidence of the winning class is computed.
#[derive(Debug, Clone, Copy)]
pub(super) enum ConfKind {
    /// Scores are probabilities already: the winner's real value.
    Probability,
    /// Independent per-class logits: the winner's sigmoid.
    Sigmoid,
    /// Logits: the maximum of the softmax across classes.
    Softmax,
    /// One logit channel: `max(p, 1 - p)` with `p = sigmoid(v)`.
    BinaryLogit,
    /// One probability channel: `max(p, 1 - p)`.
    BinaryProbability,
}

/// Confidence of the per-pixel winner at the score map's resolution.
pub(super) fn confidence_map<T: Score>(
    isa: Isa,
    data: &[T],
    g: Grid,
    winners: &[u16],
    dq: Dequant,
    kind: ConfKind,
) -> Vec<f32> {
    let mut out = vec![0f32; g.height * g.width];
    out.par_chunks_mut(g.width)
        .enumerate()
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each_init(
            || (vec![0f32; g.width], vec![0f32; g.width]),
            |scratch, (y, row)| {
                let win = &winners[y * g.width..][..g.width];
                let at =
                    |c: usize, x: usize| dq.real(data[c * g.cs + y * g.ys + x * g.xs].to_f32());
                if confidence_row_neon(isa, data, g, y, win, dq, kind, scratch, row) {
                    return;
                }
                match kind {
                    ConfKind::Probability => winner_row(row, win, at),
                    ConfKind::Sigmoid => winner_row(row, win, |c, x| sigmoid(at(c, x))),
                    ConfKind::Softmax => softmax_row(data, g, y, win, dq, scratch, row),
                    ConfKind::BinaryLogit => binary_row(row, |x| sigmoid(at(0, x))),
                    ConfKind::BinaryProbability => binary_row(row, |x| at(0, x)),
                }
            },
        );
    out
}

/// [`confidence_map`]'s row `y` on a NEON tier; `false` when the scalar code computes it.
#[allow(clippy::too_many_arguments)]
fn confidence_row_neon<T: Score>(
    isa: Isa,
    data: &[T],
    g: Grid,
    y: usize,
    win: &[u16],
    dq: Dequant,
    kind: ConfKind,
    (values, _): &mut (Vec<f32>, Vec<f32>),
    row: &mut [f32],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    let stored = |x: usize, c: u16| data[c as usize * g.cs + y * g.ys + x * g.xs].to_f32();
    let ran = match kind {
        ConfKind::Softmax => {
            let Some(layout) = ScoreRow::of(g.classes, g.cs, g.xs) else {
                return false;
            };
            for (x, (t, &c)) in values.iter_mut().zip(win).enumerate() {
                *t = stored(x, c);
            }
            let src = row_scores(data, g, y);
            nd::softmax_row(isa, src, layout, g.classes, values, dq.scale, row)
        }
        ConfKind::Sigmoid => {
            for (x, (v, &c)) in values.iter_mut().zip(win).enumerate() {
                *v = dq.real(stored(x, c));
            }
            nd::sigmoid_row(isa, values, row)
        }
        ConfKind::BinaryLogit | ConfKind::BinaryProbability => {
            let Some(src) = contiguous_row(data, g, y) else {
                return false;
            };
            let logit = matches!(kind, ConfKind::BinaryLogit);
            nd::binary_row(isa, src, dq.zero_point, dq.scale, logit, row)
        }
        ConfKind::Probability => Ran::Scalar,
    };
    ran != Ran::Scalar
}

#[inline(always)]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// A function of each pixel's winning class and column.
#[inline(always)]
fn winner_row(row: &mut [f32], win: &[u16], f: impl Fn(usize, usize) -> f32) {
    for (x, (slot, &c)) in row.iter_mut().zip(win).enumerate() {
        *slot = f(c as usize, x);
    }
}

/// `max(p, 1 - p)` of a single-channel probability.
#[inline(always)]
fn binary_row(row: &mut [f32], p: impl Fn(usize) -> f32) {
    for (x, slot) in row.iter_mut().enumerate() {
        let p = p(x);
        *slot = p.max(1.0 - p);
    }
}

/// The winner's softmax probability, `1 / sum(exp(v - top))`.
#[inline(always)]
fn softmax_row<T: Score>(
    data: &[T],
    g: Grid,
    y: usize,
    win: &[u16],
    dq: Dequant,
    (top, sum): &mut (Vec<f32>, Vec<f32>),
    row: &mut [f32],
) {
    let at = |c: usize, x: usize| data[c * g.cs + y * g.ys + x * g.xs].to_f32();
    for (x, t) in top.iter_mut().enumerate() {
        *t = at(win[x] as usize, x);
    }
    sum.fill(0.0);
    for c in 0..g.classes {
        if g.xs == 1 {
            let plane = &data[c * g.cs + y * g.ys..][..g.width];
            for ((s, &t), &v) in sum.iter_mut().zip(top.iter()).zip(plane) {
                *s += exp_nonpositive(dq.scale * (v.to_f32() - t));
            }
        } else {
            for (x, (s, &t)) in sum.iter_mut().zip(top.iter()).enumerate() {
                *s += exp_nonpositive(dq.scale * (at(c, x) - t));
            }
        }
    }
    for (slot, &s) in row.iter_mut().zip(sum.iter()) {
        *slot = 1.0 / s;
    }
}

/// Nearest-neighbour upsample of a low-resolution class map.
pub(super) fn upsample_classes<O: ClassOut>(
    isa: Isa,
    map: &[u16],
    width: usize,
    ys: &[u32],
    xs: &[u32],
    dst: &mut [O],
) {
    if !nd::nearest_classes(isa, map, width, ys, xs, dst) {
        upsample_nearest(map, width, ys, xs, dst, O::from_class);
    }
}

/// Nearest-neighbour upsample of a low-resolution confidence map.
pub(super) fn upsample_conf_nearest<C: ConfOut>(
    isa: Isa,
    map: &[f32],
    width: usize,
    ys: &[u32],
    xs: &[u32],
    dst: &mut [C],
) {
    if !nd::nearest_confidence(isa, map, width, ys, xs, dst) {
        upsample_nearest(map, width, ys, xs, dst, C::from_conf);
    }
}

/// The lower and upper positions and weights of a tap list, as separate arrays.
struct SplitTaps {
    lo: Vec<u32>,
    hi: Vec<u32>,
    frac: Vec<f32>,
}

impl SplitTaps {
    fn new(taps: &[Tap]) -> Self {
        Self {
            lo: taps.iter().map(|t| t.lo).collect(),
            hi: taps.iter().map(|t| t.hi).collect(),
            frac: taps.iter().map(|t| t.frac).collect(),
        }
    }

    fn view(&self) -> Taps<'_> {
        Taps {
            lo: &self.lo,
            hi: &self.hi,
            frac: &self.frac,
        }
    }
}

/// Nearest-neighbour upsample of a low-resolution map, converting each value.
fn upsample_nearest<S: Copy + Sync, O: Send>(
    map: &[S],
    width: usize,
    ys: &[u32],
    xs: &[u32],
    dst: &mut [O],
    convert: impl Fn(S) -> O + Sync,
) {
    dst.par_chunks_mut(xs.len())
        .zip(ys.par_iter())
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(row, &sy)| {
            let src = &map[sy as usize * width..][..width];
            for (o, &sx) in row.iter_mut().zip(xs) {
                *o = convert(src[sx as usize]);
            }
        });
}

/// Bilinear upsample of a low-resolution confidence map.
pub(super) fn upsample_conf_bilinear<C: ConfOut>(
    isa: Isa,
    map: &[f32],
    width: usize,
    ys: &[Tap],
    xs: &[Tap],
    dst: &mut [C],
) {
    if !isa.is_scalar() {
        let (ty, tx) = (SplitTaps::new(ys), SplitTaps::new(xs));
        if nd::bilinear_confidence(isa, map, width, ty.view(), tx.view(), dst) {
            return;
        }
    }
    dst.par_chunks_mut(xs.len())
        .zip(ys.par_iter())
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(row, ty)| {
            let r0 = &map[ty.lo as usize * width..][..width];
            let r1 = &map[ty.hi as usize * width..][..width];
            for (o, tx) in row.iter_mut().zip(xs) {
                let v = lerp2(
                    ty.frac,
                    tx.frac,
                    r0[tx.lo as usize],
                    r0[tx.hi as usize],
                    r1[tx.lo as usize],
                    r1[tx.hi as usize],
                );
                *o = C::from_conf(v);
            }
        });
}

/// PyTorch's bilinear blend, in its operation order.
#[inline(always)]
fn lerp2(fy: f32, fx: f32, v00: f32, v01: f32, v10: f32, v11: f32) -> f32 {
    let (wy0, wx0) = (1.0 - fy, 1.0 - fx);
    wy0 * (wx0 * v00 + fx * v01) + fy * (wx0 * v10 + fx * v11)
}

/// Bilinear upsample of a single score channel, thresholded at `cut`
/// (in stored units).
pub(super) fn upsample_threshold<T: Score, O: ClassOut>(
    isa: Isa,
    data: &[T],
    g: Grid,
    cut: f32,
    ys: &[Tap],
    xs: &[Tap],
    dst: &mut [O],
) {
    if !isa.is_scalar() {
        let (ty, tx) = (SplitTaps::new(ys), SplitTaps::new(xs));
        let fill = |y: usize, row: &mut [f32]| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = data[y * g.ys + x * g.xs].to_f32();
            }
        };
        if nd::bilinear_threshold(isa, g.width, fill, ty.view(), tx.view(), cut, dst) {
            return;
        }
    }
    dst.par_chunks_mut(xs.len())
        .zip(ys.par_iter())
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(row, ty)| {
            let at = |y: u32, x: u32| data[y as usize * g.ys + x as usize * g.xs].to_f32();
            for (o, tx) in row.iter_mut().zip(xs) {
                let v = lerp2(
                    ty.frac,
                    tx.frac,
                    at(ty.lo, tx.lo),
                    at(ty.lo, tx.hi),
                    at(ty.hi, tx.lo),
                    at(ty.hi, tx.hi),
                );
                *o = O::from_class(u16::from(v > cut));
            }
        });
}

/// Bilinear upsample of every class channel followed by argmax.
///
/// Bilinear interpolation is a convex combination, so where the low-resolution
/// argmax agrees at every corner a pixel reads, that class is the result.
/// Pixels whose corners disagree interpolate only the candidate classes of
/// their cell: a class whose largest corner score is below the best smallest
/// corner score of any class cannot win anywhere in the cell.
///
/// Output rows are processed in bands that read the same pair of source rows.
/// Each band gathers the corner scores of its boundary cells once, streaming
/// one class plane at a time, since gathering a class vector across planes of
/// a power-of-two-like stride thrashes the cache.
pub(super) fn upsample_argmax<T: Score, O: ClassOut>(
    isa: Isa,
    data: &[T],
    g: Grid,
    winners: &[u16],
    ys: &[Tap],
    xs: &[Tap],
    dst: &mut [O],
) {
    let out_w = xs.len();
    let mut bands: Vec<(usize, &mut [O])> = Vec::new();
    let mut rest = dst;
    let mut start = 0;
    while start < ys.len() {
        let lo = ys[start].lo;
        let end = start + ys[start..].iter().take_while(|t| t.lo == lo).count();
        let (band, tail) = rest.split_at_mut((end - start) * out_w);
        bands.push((start, band));
        rest = tail;
        start = end;
    }
    // Output columns grouped by the source column their left tap reads.
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (dx, tx) in xs.iter().enumerate() {
        match runs.last_mut() {
            Some((lo, _, end)) if *lo == tx.lo as usize => *end = dx + 1,
            _ => runs.push((tx.lo as usize, dx, dx + 1)),
        }
    }
    let fx: Vec<f32> = xs.iter().map(|t| t.frac).collect();
    bands.into_par_iter().for_each_init(
        || (Band::new(g, isa), vec![0u16; out_w]),
        |(cells, picks), (first, band)| {
            let y0 = ys[first].lo as usize;
            cells.prepare(data, winners, y0, (y0 + 1).min(g.height - 1));
            for (r, row) in band.chunks_mut(out_w).enumerate() {
                let ty = &ys[first + r];
                for &(x0, start, end) in &runs {
                    match cells.uniform[x0] {
                        Some(class) => row[start..end].fill(O::from_class(class)),
                        None => {
                            let (classes, corners) = cells.candidates(x0);
                            let picks = &mut picks[start..end];
                            let fx = &fx[start..end];
                            if nd::cell_argmax(isa, classes, corners, ty.frac, fx, picks)
                                != Ran::Scalar
                            {
                                for (o, &c) in row[start..end].iter_mut().zip(picks.iter()) {
                                    *o = O::from_class(c);
                                }
                                continue;
                            }
                            for (o, tx) in row[start..end].iter_mut().zip(&xs[start..end]) {
                                *o = O::from_class(cells.argmax(x0, ty.frac, tx.frac));
                            }
                        }
                    }
                }
            }
        },
    );
}

/// Appends to `classes` and `out` each class (by its position in `corners`) whose largest corner
/// reaches `floor`, the best smallest corner of any class, in class order.
fn push_candidates(
    corners: &[[f32; 4]],
    floor: f32,
    classes: &mut Vec<u16>,
    out: &mut Vec<[f32; 4]>,
) {
    for (c, v) in corners.iter().enumerate() {
        if v[0].max(v[1]).max(v[2]).max(v[3]) >= floor {
            classes.push(c as u16);
            out.push(*v);
        }
    }
}

/// Candidate classes and their corner scores for the boundary cells of one
/// band (a pair of source rows).
struct Band {
    g: Grid,
    isa: Isa,
    /// Per source column: index into `cols` once gathered.
    col_slot: Vec<u32>,
    cols: Vec<usize>,
    /// Gathered scores, `[class][slot]`, for the band's two source rows; `[slot][class]` when
    /// gathered on a NEON tier.
    rows: [Vec<f32>; 2],
    /// Per cell (left source column): the class of a cell whose corners
    /// agree, `None` for a boundary cell.
    uniform: Vec<Option<u16>>,
    /// Per cell (left source column): `(start, len)` into `classes`/`corners`.
    spans: Vec<(u32, u32)>,
    classes: Vec<u16>,
    corners: Vec<[f32; 4]>,
    /// One cell's corner scores per class, before candidate selection; sized on first use.
    scratch: Vec<[f32; 4]>,
}

impl Band {
    fn new(g: Grid, isa: Isa) -> Self {
        Self {
            g,
            isa,
            col_slot: vec![u32::MAX; g.width],
            cols: Vec::with_capacity(g.width),
            rows: [Vec::new(), Vec::new()],
            uniform: vec![None; g.width],
            spans: vec![(0, 0); g.width],
            classes: Vec::new(),
            corners: Vec::new(),
            scratch: Vec::new(),
        }
    }

    fn prepare<T: Score>(&mut self, data: &[T], winners: &[u16], y0: usize, y1: usize) {
        let g = self.g;
        let w0 = &winners[y0 * g.width..][..g.width];
        let w1 = &winners[y1 * g.width..][..g.width];
        self.col_slot.fill(u32::MAX);
        self.cols.clear();
        self.classes.clear();
        self.corners.clear();
        let mut boundary = Vec::new();
        for x in 0..g.width {
            let x1 = (x + 1).min(g.width - 1);
            let c = w0[x];
            let agree = c == w0[x1] && c == w1[x] && c == w1[x1];
            self.uniform[x] = agree.then_some(c);
            if !agree {
                boundary.push(x);
                for col in [x, x1] {
                    if self.col_slot[col] == u32::MAX {
                        self.col_slot[col] = self.cols.len() as u32;
                        self.cols.push(col);
                    }
                }
            }
        }
        if boundary.is_empty() {
            return;
        }
        let n = self.cols.len();
        if self.gather_by_slot(data, y0, y1) {
            for x in boundary {
                self.build_by_slot(x);
            }
            return;
        }
        for (row, y) in self.rows.iter_mut().zip([y0, y1]) {
            row.clear();
            row.resize(g.classes * n, 0.0);
            for c in 0..g.classes {
                let base = c * g.cs + y * g.ys;
                let out = &mut row[c * n..][..n];
                for (o, &x) in out.iter_mut().zip(&self.cols) {
                    *o = data[base + x * g.xs].to_f32();
                }
            }
        }
        for x in boundary {
            self.build(x);
        }
    }

    /// The `[slot][class]` gather of both source rows on a NEON tier; `false` when the scalar
    /// `[class][slot]` gather runs instead.
    fn gather_by_slot<T: Score>(&mut self, data: &[T], y0: usize, y1: usize) -> bool {
        if self.isa.is_scalar() {
            return false;
        }
        let g = self.g;
        let cols: Vec<u32> = self.cols.iter().map(|&x| x as u32).collect();
        let layout = (g.cs, g.xs, g.classes);
        self.rows.iter_mut().zip([y0, y1]).all(|(row, y)| {
            row.clear();
            row.resize(g.classes * cols.len(), 0.0);
            let src = row_scores(data, g, y);
            nd::gather_columns(self.isa, src, layout, &cols, row) != Ran::Scalar
        })
    }

    /// [`Self::build`] from `[slot][class]` rows.
    fn build_by_slot(&mut self, x0: usize) {
        let g = self.g;
        let a = self.col_slot[x0] as usize;
        let b = self.col_slot[(x0 + 1).min(g.width - 1)] as usize;
        let slot = |r: usize, s: usize| &self.rows[r][s * g.classes..][..g.classes];
        let corners = [slot(0, a), slot(0, b), slot(1, a), slot(1, b)];
        let start = self.classes.len() as u32;
        let (classes, out) = (&mut self.classes, &mut self.corners);
        if nd::cell_candidates(self.isa, corners, classes, out) == Ran::Scalar {
            self.scratch.resize(g.classes, [0.0; 4]);
            let mut floor = f32::NEG_INFINITY;
            for (c, v) in self.scratch.iter_mut().enumerate() {
                *v = corners.map(|k| k[c]);
                floor = floor.max(v[0].min(v[1]).min(v[2]).min(v[3]));
            }
            push_candidates(&self.scratch, floor, classes, out);
        }
        self.spans[x0] = (start, self.classes.len() as u32 - start);
    }

    fn build(&mut self, x0: usize) {
        let g = self.g;
        let n = self.cols.len();
        let a = self.col_slot[x0] as usize;
        let b = self.col_slot[(x0 + 1).min(g.width - 1)] as usize;
        let start = self.classes.len() as u32;
        self.scratch.resize(g.classes, [0.0; 4]);
        let mut floor = f32::NEG_INFINITY;
        for (c, v) in self.scratch.iter_mut().enumerate() {
            let (r0, r1) = (&self.rows[0][c * n..], &self.rows[1][c * n..]);
            *v = [r0[a], r0[b], r1[a], r1[b]];
            floor = floor.max(v[0].min(v[1]).min(v[2]).min(v[3]));
        }
        push_candidates(&self.scratch, floor, &mut self.classes, &mut self.corners);
        self.spans[x0] = (start, self.classes.len() as u32 - start);
    }

    /// Candidate classes and their corner scores of the boundary cell whose
    /// left column is `x0`.
    fn candidates(&self, x0: usize) -> (&[u16], &[[f32; 4]]) {
        let (start, len) = self.spans[x0];
        let range = start as usize..(start + len) as usize;
        (&self.classes[range.clone()], &self.corners[range])
    }

    /// Argmax of the interpolated scores at weights `(fy, fx)` in the
    /// boundary cell whose left column is `x0`; ties go to the lowest class.
    fn argmax(&self, x0: usize, fy: f32, fx: f32) -> u16 {
        let (start, len) = self.spans[x0];
        let range = start as usize..(start + len) as usize;
        let mut best = f32::NEG_INFINITY;
        let mut idx = 0;
        for (&class, v) in self.classes[range.clone()].iter().zip(&self.corners[range]) {
            let s = lerp2(fy, fx, v[0], v[1], v[2], v[3]);
            if s > best {
                best = s;
                idx = class;
            }
        }
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_taps_match_pytorch_half_pixel() {
        // 4 -> 8, align_corners=false: src = (d + 0.5) / 2 - 0.5, clamped at 0.
        let t = bilinear_taps(8, 4, 0.0, 4.0, false);
        assert_eq!(
            t[0],
            Tap {
                lo: 0,
                hi: 0,
                frac: 0.0
            }
        );
        assert_eq!(
            t[1],
            Tap {
                lo: 0,
                hi: 1,
                frac: 0.25
            }
        );
        assert_eq!(
            t[2],
            Tap {
                lo: 0,
                hi: 1,
                frac: 0.75
            }
        );
        assert_eq!(
            t[7],
            Tap {
                lo: 3,
                hi: 3,
                frac: 0.0
            }
        );
    }

    #[test]
    fn bilinear_taps_align_corners() {
        // 3 -> 5, align_corners=true: src = d * 2 / 4.
        let t = bilinear_taps(5, 3, 0.0, 3.0, true);
        assert_eq!(
            t[0],
            Tap {
                lo: 0,
                hi: 0,
                frac: 0.0
            }
        );
        assert_eq!(
            t[1],
            Tap {
                lo: 0,
                hi: 1,
                frac: 0.5
            }
        );
        assert_eq!(
            t[2],
            Tap {
                lo: 1,
                hi: 1,
                frac: 0.0
            }
        );
        assert_eq!(
            t[4],
            Tap {
                lo: 2,
                hi: 2,
                frac: 0.0
            }
        );
    }

    #[test]
    fn nearest_taps_use_pixel_centres() {
        assert_eq!(nearest_taps(8, 4, 0.0, 4.0), vec![0, 0, 1, 1, 2, 2, 3, 3]);
        // A region covering source rows [1, 3) of 4.
        assert_eq!(nearest_taps(4, 4, 1.0, 2.0), vec![1, 1, 2, 2]);
    }

    #[test]
    fn argmax_ties_take_lowest_index() {
        // NCHW, 3 classes, 1x2: pixel 0 ties classes 1 and 2, pixel 1 ties all.
        let data: [f32; 6] = [0.0, 5.0, 3.0, 5.0, 3.0, 5.0];
        let g = Grid {
            classes: 3,
            height: 1,
            width: 2,
            cs: 2,
            ys: 2,
            xs: 1,
        };
        assert_eq!(argmax_map(Isa::SCALAR, &data, g), vec![1, 0]);
        let nhwc = Grid {
            cs: 1,
            ys: 6,
            xs: 3,
            ..g
        };
        let data: [f32; 6] = [0.0, 3.0, 3.0, 5.0, 5.0, 5.0];
        assert_eq!(argmax_map(Isa::SCALAR, &data, nhwc), vec![1, 0]);
    }
}
