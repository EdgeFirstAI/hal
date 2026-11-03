// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON kernels for the semantic segmentation decoder: the per-pixel class argmax, the
//! single-channel threshold, the winner's confidence, nearest and bilinear resampling rows, the
//! argmax of boundary cells and class-index narrowing.
//!
//! Every kernel except the two sigmoid ones equals the decoder's scalar code bit for bit:
//! comparisons, conversions and copies are exact, and float arithmetic runs the scalar operations
//! in the same order with no FMA (the softmax confidence sums `prim::softmax::exp_nonpositive`
//! class by class per pixel, as the scalar loop does). The argmax keeps the first class holding
//! the largest score: a strict `>` from class 0, so a NaN score never wins and a NaN first score
//! is never replaced. Planar scores (class planes with a stride) run in strips of 256 pixels,
//! class by class: each class's row segment is compared with the strip's running best, held in an
//! L1 buffer, while the next class's segment is prefetched; a ragged tail and rows under 16 pixels
//! run a 16-pixel block with every class's best and index in registers. Interleaved scores
//! (classes contiguous per pixel) are transposed a tile at a time and run that register-blocked
//! kernel on the L1-resident tile. `f16` scores compare as f32
//! after `FCVTL` (exact), or natively in the `*_neon_fp16` kernels, which need FEAT_FP16. The
//! sigmoid kernels use `neon_sigmoid::sigmoid4`, within `2^-21` of `1 / (1 + e^-x)` relative to
//! the result. A kernel returns `false` and writes nothing when its arguments are invalid.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use half::f16;

use super::fp16_asm::fcmgt_f16x8;
use super::neon_candidates::prefetch;
use super::neon_convert::{F32x16, LoadF32};
use super::neon_sigmoid::sigmoid4;
use super::softmax::{EXP2_TAYLOR, EXP_MIN};
use super::transpose::{nchw_to_nhwc, TransposeElement};

/// Pixels per vector block.
const BLOCK: usize = 16;

/// Pixels per pass of the planar kernels over the class planes: the running best or sum of a
/// strip stays in L1 while each class's row segment streams in.
const STRIP: usize = 256;

/// Bytes of interleaved scores transposed into one planar tile.
const TILE_BYTES: usize = 16 * 1024;

/// Largest class count an index of `u16` lanes holds.
const MAX_CLASSES: usize = 1 << 16;

/// Whether class planes `cs` apart hold `n` pixels each within `len` elements.
fn planar_fits(len: usize, cs: usize, classes: usize, n: usize) -> bool {
    (1..=MAX_CLASSES).contains(&classes)
        && (classes == 1 || cs > 0)
        && (classes - 1)
            .checked_mul(cs)
            .and_then(|e| e.checked_add(n))
            .is_some_and(|e| e <= len)
}

/// Runs `block(at)` for the 16-pixel blocks covering `n >= 16` pixels, the last one overlapping
/// its predecessor so no pixel is left over.
#[inline(always)]
fn for_blocks(n: usize, mut block: impl FnMut(usize)) {
    let mut at = 0;
    while at + BLOCK < n {
        block(at);
        at += BLOCK;
    }
    block(n - BLOCK);
}

/// Runs `planar(tile, 16, classes, out)` on the `n < 16` pixels of planes `cs` apart copied into
/// a zero-padded tile of 16-pixel planes, then keeps the first `n` results.
fn padded<S: Copy + Default, D: Copy + Default>(
    src: &[S],
    cs: usize,
    classes: usize,
    dst: &mut [D],
    planar: impl FnOnce(&[S], &mut [D; BLOCK]),
) {
    let n = dst.len();
    let mut tile = vec![S::default(); classes * BLOCK];
    for (c, plane) in tile.chunks_exact_mut(BLOCK).enumerate() {
        if let Some(s) = src.get(c * cs..c * cs + n) {
            plane[..n].copy_from_slice(s);
        }
    }
    let mut out = [D::default(); BLOCK];
    planar(&tile, &mut out);
    dst.copy_from_slice(&out[..n]);
}

/// Pixels of `classes` interleaved `bytes`-byte scores per transposed tile: a multiple of 16.
fn tile_pixels(classes: usize, bytes: usize) -> usize {
    (TILE_BYTES / (classes * bytes) / BLOCK).max(1) * BLOCK
}

/// Runs `planar(tile, out, first)` on tiles of the `n` pixels of interleaved `src` (`classes`
/// scores per pixel) transposed into class planes, `out` being the tile's slice of `dst`.
fn interleaved<T: TransposeElement + Default, D>(
    src: &[T],
    classes: usize,
    dst: &mut [D],
    mut planar: impl FnMut(&[T], &mut [D], usize) -> bool,
) -> bool {
    let n = dst.len();
    if !(1..=MAX_CLASSES).contains(&classes) || n.checked_mul(classes).is_none_or(|e| e > src.len())
    {
        return false;
    }
    if n == 0 {
        return true;
    }
    let p = tile_pixels(classes, size_of::<T>()).min(n);
    let mut tile = vec![T::default(); p * classes];
    let mut first = 0;
    for (chunk, out) in src.chunks(p * classes).zip(dst.chunks_mut(p)) {
        let q = out.len();
        let (Some(chunk), Some(t)) = (chunk.get(..q * classes), tile.get_mut(..q * classes)) else {
            return false;
        };
        nchw_to_nhwc(chunk, 1, classes, q, t);
        if !planar(t, out, first) {
            return false;
        }
        first += q;
    }
    true
}

/// Score lanes of the planar argmax: 16 pixels.
trait ArgmaxLanes {
    type T: Copy + Default;
    type V: Copy;

    /// How a running best is kept in memory.
    type B: Copy + Default;

    /// # Safety
    /// `p` must be valid for reading 16 scores.
    unsafe fn load(p: *const Self::T) -> Self::V;

    /// # Safety
    /// `p` must be valid for reading 16 values.
    unsafe fn load_best(p: *const Self::B) -> Self::V;

    /// # Safety
    /// `p` must be valid for writing 16 values.
    unsafe fn store_best(p: *mut Self::B, v: Self::V);

    /// Lanes where `v > best`, as halfword masks of pixels 0–7 and 8–15, with `best` set to `v`
    /// there.
    fn update(best: &mut Self::V, v: Self::V) -> [uint16x8_t; 2];
}

/// A byte-lane mask as two halfword-lane masks.
#[inline(always)]
fn widen_mask8(m: uint8x16_t) -> [uint16x8_t; 2] {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        [
            vreinterpretq_u16_u8(vzip1q_u8(m, m)),
            vreinterpretq_u16_u8(vzip2q_u8(m, m)),
        ]
    }
}

/// Four word-lane masks as two halfword-lane masks.
#[inline(always)]
fn narrow_mask32(m: [uint32x4_t; 4]) -> [uint16x8_t; 2] {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        [
            vuzp1q_u16(vreinterpretq_u16_u32(m[0]), vreinterpretq_u16_u32(m[1])),
            vuzp1q_u16(vreinterpretq_u16_u32(m[2]), vreinterpretq_u16_u32(m[3])),
        ]
    }
}

macro_rules! byte_lanes {
    ($name:ident, $t:ty, $v:ty, $ld:ident, $st:ident, $gt:ident, $max:ident) => {
        struct $name;

        impl ArgmaxLanes for $name {
            type T = $t;
            type V = $v;
            type B = $t;

            #[inline(always)]
            unsafe fn load(p: *const $t) -> $v {
                // SAFETY: the caller guarantees 16 readable scores.
                unsafe { $ld(p) }
            }

            #[inline(always)]
            unsafe fn load_best(p: *const $t) -> $v {
                // SAFETY: the caller guarantees 16 readable values.
                unsafe { $ld(p) }
            }

            #[inline(always)]
            unsafe fn store_best(p: *mut $t, v: $v) {
                // SAFETY: the caller guarantees 16 writable values.
                unsafe { $st(p, v) }
            }

            #[inline(always)]
            fn update(best: &mut $v, v: $v) -> [uint16x8_t; 2] {
                // SAFETY: NEON is baseline; the larger integer is `v` exactly where `v > best`.
                unsafe {
                    let m = $gt(v, *best);
                    *best = $max(v, *best);
                    widen_mask8(m)
                }
            }
        }
    };
}
byte_lanes!(U8, u8, uint8x16_t, vld1q_u8, vst1q_u8, vcgtq_u8, vmaxq_u8);
byte_lanes!(I8, i8, int8x16_t, vld1q_s8, vst1q_s8, vcgtq_s8, vmaxq_s8);

macro_rules! half_lanes {
    ($name:ident, $t:ty, $v:ty, $ld:ident, $st:ident, $gt:ident, $max:ident) => {
        struct $name;

        impl ArgmaxLanes for $name {
            type T = $t;
            type V = [$v; 2];
            type B = $t;

            #[inline(always)]
            unsafe fn load(p: *const $t) -> [$v; 2] {
                // SAFETY: the caller guarantees 16 readable scores.
                unsafe { [$ld(p), $ld(p.add(8))] }
            }

            #[inline(always)]
            unsafe fn load_best(p: *const $t) -> [$v; 2] {
                // SAFETY: the caller guarantees 16 readable values.
                unsafe { [$ld(p), $ld(p.add(8))] }
            }

            #[inline(always)]
            unsafe fn store_best(p: *mut $t, v: [$v; 2]) {
                // SAFETY: the caller guarantees 16 writable values.
                unsafe {
                    $st(p, v[0]);
                    $st(p.add(8), v[1]);
                }
            }

            #[inline(always)]
            fn update(best: &mut [$v; 2], v: [$v; 2]) -> [uint16x8_t; 2] {
                // SAFETY: as in `byte_lanes!`.
                unsafe {
                    let m = [$gt(v[0], best[0]), $gt(v[1], best[1])];
                    *best = [$max(v[0], best[0]), $max(v[1], best[1])];
                    m
                }
            }
        }
    };
}
half_lanes!(U16, u16, uint16x8_t, vld1q_u16, vst1q_u16, vcgtq_u16, vmaxq_u16);
half_lanes!(I16, i16, int16x8_t, vld1q_s16, vst1q_s16, vcgtq_s16, vmaxq_s16);

/// f32 lanes compared with `FCMGT` and kept with `BSL`, so NaN never wins.
#[inline(always)]
fn update_f32(best: &mut F32x16, v: F32x16) -> [uint16x8_t; 2] {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let m = [0, 1, 2, 3].map(|k| vcgtq_f32(v[k], best[k]));
        *best = [0, 1, 2, 3].map(|k| vbslq_f32(m[k], v[k], best[k]));
        narrow_mask32(m)
    }
}

struct F32;

impl ArgmaxLanes for F32 {
    type T = f32;
    type V = F32x16;
    type B = f32;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable scores.
        unsafe { f32::load16(p) }
    }

    #[inline(always)]
    unsafe fn load_best(p: *const f32) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable values.
        unsafe { f32::load16(p) }
    }

    #[inline(always)]
    unsafe fn store_best(p: *mut f32, v: F32x16) {
        // SAFETY: the caller guarantees 16 writable values.
        unsafe { store_f32(p, v) }
    }

    #[inline(always)]
    fn update(best: &mut F32x16, v: F32x16) -> [uint16x8_t; 2] {
        update_f32(best, v)
    }
}

/// `f16` scores widened to f32 (exact) and compared there.
struct F16Widened;

impl ArgmaxLanes for F16Widened {
    type T = f16;
    type V = F32x16;
    type B = f32;

    #[inline(always)]
    unsafe fn load(p: *const f16) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable scores.
        unsafe { f16::load16(p) }
    }

    #[inline(always)]
    unsafe fn load_best(p: *const f32) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable values.
        unsafe { f32::load16(p) }
    }

    #[inline(always)]
    unsafe fn store_best(p: *mut f32, v: F32x16) {
        // SAFETY: the caller guarantees 16 writable values.
        unsafe { store_f32(p, v) }
    }

    #[inline(always)]
    fn update(best: &mut F32x16, v: F32x16) -> [uint16x8_t; 2] {
        update_f32(best, v)
    }
}

/// `f16` scores compared with FP16 instructions.
struct F16Native;

impl ArgmaxLanes for F16Native {
    type T = f16;
    type V = [uint16x8_t; 2];
    type B = u16;

    #[inline(always)]
    unsafe fn load(p: *const f16) -> [uint16x8_t; 2] {
        let p = p.cast::<u16>();
        // SAFETY: the caller guarantees 16 readable scores; `f16` is a transparent `u16`.
        unsafe { [vld1q_u16(p), vld1q_u16(p.add(8))] }
    }

    #[inline(always)]
    unsafe fn load_best(p: *const u16) -> [uint16x8_t; 2] {
        // SAFETY: the caller guarantees 16 readable values.
        unsafe { [vld1q_u16(p), vld1q_u16(p.add(8))] }
    }

    #[inline(always)]
    unsafe fn store_best(p: *mut u16, v: [uint16x8_t; 2]) {
        // SAFETY: the caller guarantees 16 writable values.
        unsafe { store_u16(p, v) }
    }

    #[inline(always)]
    fn update(best: &mut [uint16x8_t; 2], v: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
        // SAFETY: the native FP16 lanes run only inside the `*_neon_fp16` kernels.
        let m = unsafe { [fcmgt_f16x8(v[0], best[0]), fcmgt_f16x8(v[1], best[1])] };
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            *best = [
                vbslq_u16(m[0], v[0], best[0]),
                vbslq_u16(m[1], v[1], best[1]),
            ]
        };
        m
    }
}

/// The argmax of 16 pixels whose class `c` scores start at `p + c * cs`.
///
/// # Safety
/// `p + c * cs` must be valid for reading 16 scores for every `c < classes`, `classes <= 65536`.
#[inline(always)]
unsafe fn argmax_block<L: ArgmaxLanes>(
    p: *const L::T,
    cs: usize,
    classes: usize,
) -> [uint16x8_t; 2] {
    // SAFETY: the caller guarantees every read.
    unsafe {
        let mut best = L::load(p);
        let mut idx = [vdupq_n_u16(0); 2];
        for c in 1..classes {
            let m = L::update(&mut best, L::load(p.add(c * cs)));
            let class = vdupq_n_u16(c as u16);
            idx = [
                vbslq_u16(m[0], class, idx[0]),
                vbslq_u16(m[1], class, idx[1]),
            ];
        }
        idx
    }
}

/// Hints the `bytes` from `p` into L1, one cache line at a time.
#[inline(always)]
fn prefetch_span<T>(p: *const T, bytes: usize) {
    let p = p.cast::<u8>();
    for off in (0..bytes).step_by(64) {
        prefetch(p.wrapping_add(off));
    }
}

/// The argmax of `m` (a multiple of 16, at most [`STRIP`]) pixels whose class `c` scores start
/// at `p + c * cs`, written to `d`, one class row segment at a time.
///
/// # Safety
/// `p + c * cs` must be valid for reading `m` scores for every `c < classes`, `d` for writing
/// `m` classes, `classes <= 65536`.
#[inline(always)]
unsafe fn argmax_strip<L: ArgmaxLanes>(
    p: *const L::T,
    cs: usize,
    classes: usize,
    m: usize,
    d: *mut u16,
) {
    let mut best = [L::B::default(); STRIP];
    let b = best.as_mut_ptr();
    let bytes = m * size_of::<L::T>();
    // SAFETY: the caller guarantees every access; `best` holds `STRIP >= m` values.
    unsafe {
        for at in (0..m).step_by(BLOCK) {
            L::store_best(b.add(at), L::load(p.add(at)));
            store_u16(d.add(at), [vdupq_n_u16(0); 2]);
        }
        for c in 1..classes {
            let q = p.add(c * cs);
            if c + 1 < classes {
                prefetch_span(q.add(cs), bytes);
            }
            let class = vdupq_n_u16(c as u16);
            for at in (0..m).step_by(BLOCK) {
                let mut v = L::load_best(b.add(at));
                let mask = L::update(&mut v, L::load(q.add(at)));
                L::store_best(b.add(at), v);
                let i = [vld1q_u16(d.add(at)), vld1q_u16(d.add(at + 8))];
                store_u16(
                    d.add(at),
                    [
                        vbslq_u16(mask[0], class, i[0]),
                        vbslq_u16(mask[1], class, i[1]),
                    ],
                );
            }
        }
    }
}

/// [`argmax_strip`] over the whole blocks of `dst`, then [`argmax_block`] on the last 16 pixels
/// when blocks leave a tail; see [`SemanticScore::argmax_planar_neon`]. A `cached` source (a
/// transposed tile in L1) runs [`argmax_block`] throughout.
fn argmax_planar<L: ArgmaxLanes>(
    src: &[L::T],
    cs: usize,
    classes: usize,
    dst: &mut [u16],
    cached: bool,
) -> bool {
    let n = dst.len();
    if !planar_fits(src.len(), cs, classes, n) {
        return false;
    }
    if n < BLOCK {
        padded(src, cs, classes, dst, |tile, out| {
            argmax_planar::<L>(tile, BLOCK, classes, out, true);
        });
        return true;
    }
    let (p, d) = (src.as_ptr(), dst.as_mut_ptr());
    if cached {
        // SAFETY: `planar_fits` bounds every class plane's `n` pixels and blocks end by `n`.
        for_blocks(n, |at| unsafe {
            store_u16(d.add(at), argmax_block::<L>(p.add(at), cs, classes));
        });
        return true;
    }
    let whole = n - n % BLOCK;
    // SAFETY: `planar_fits` bounds each plane's `n` pixels; strips and blocks end by `n`.
    unsafe {
        for at in (0..whole).step_by(STRIP) {
            argmax_strip::<L>(p.add(at), cs, classes, STRIP.min(whole - at), d.add(at));
        }
        if whole < n {
            let idx = argmax_block::<L>(p.add(n - BLOCK), cs, classes);
            store_u16(d.add(n - BLOCK), idx);
        }
    }
    true
}

/// `1` where a lane mask is set, `0` elsewhere.
#[inline(always)]
fn mask_to_class(m: [uint16x8_t; 2]) -> [uint16x8_t; 2] {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { [vshrq_n_u16::<15>(m[0]), vshrq_n_u16::<15>(m[1])] }
}

/// Runs `step` on 16-element blocks of `a`, `b` and `dst`; the tail goes through zeroed blocks.
#[inline(always)]
fn zip_steps<A: Copy + Default, B: Copy + Default, D: Copy + Default>(
    a: &[A],
    b: &[B],
    dst: &mut [D],
    step: impl Fn(*const A, *const B, *mut D),
) -> bool {
    if a.len() != dst.len() || b.len() != dst.len() {
        return false;
    }
    let body = dst.len() - dst.len() % BLOCK;
    let (d, tail) = dst.split_at_mut(body);
    for ((a, b), d) in a
        .chunks_exact(BLOCK)
        .zip(b.chunks_exact(BLOCK))
        .zip(d.chunks_exact_mut(BLOCK))
    {
        step(a.as_ptr(), b.as_ptr(), d.as_mut_ptr());
    }
    if tail.is_empty() {
        return true;
    }
    let (mut ab, mut bb, mut db) = (
        [A::default(); BLOCK],
        [B::default(); BLOCK],
        [D::default(); BLOCK],
    );
    ab[..tail.len()].copy_from_slice(&a[body..]);
    bb[..tail.len()].copy_from_slice(&b[body..]);
    step(ab.as_ptr(), bb.as_ptr(), db.as_mut_ptr());
    tail.copy_from_slice(&db[..tail.len()]);
    true
}

/// [`zip_steps`] with one input.
#[inline(always)]
fn map_steps<S: Copy + Default, D: Copy + Default>(
    src: &[S],
    dst: &mut [D],
    step: impl Fn(*const S, *mut D),
) -> bool {
    zip_steps(src, src, dst, |s, _, d| step(s, d))
}

/// Stores 16 `u16` lanes.
///
/// # Safety
/// `p` must be valid for writing 16 elements.
#[inline(always)]
unsafe fn store_u16(p: *mut u16, v: [uint16x8_t; 2]) {
    // SAFETY: the caller guarantees 16 writable elements.
    unsafe {
        vst1q_u16(p, v[0]);
        vst1q_u16(p.add(8), v[1]);
    }
}

/// `e^x` of 4 lanes exactly as `prim::softmax::exp_nonpositive` computes it.
#[inline(always)]
fn exp_nonpositive4(x: float32x4_t) -> float32x4_t {
    let [c5, c4, c3, c2] = EXP2_TAYLOR;
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let t = vmulq_n_f32(
            vmaxnmq_f32(x, vdupq_n_f32(EXP_MIN)),
            std::f32::consts::LOG2_E,
        );
        let k = vcvtq_s32_f32(vsubq_f32(t, vdupq_n_f32(0.5)));
        let f = vsubq_f32(t, vcvtq_f32_s32(k));
        let horner = |c: f32, p: float32x4_t| vaddq_f32(vdupq_n_f32(c), vmulq_f32(f, p));
        let p = horner(c4, vdupq_n_f32(c5));
        let p = horner(c3, p);
        let p = horner(c2, p);
        let p = horner(std::f32::consts::LN_2, p);
        let p = horner(1.0, p);
        let scale = vshlq_n_s32::<23>(vaddq_s32(k, vdupq_n_s32(127)));
        vmulq_f32(vreinterpretq_f32_s32(scale), p)
    }
}

/// `1 / Σ_c e^(scale · (x_c − top))` of 16 pixels whose class `c` scores start at `p + c * cs`.
///
/// # Safety
/// `p + c * cs` must be valid for reading 16 scores for every `c < classes`, and `top` for 16.
#[inline(always)]
unsafe fn softmax_block<S: LoadF32>(
    p: *const S,
    cs: usize,
    classes: usize,
    top: *const f32,
    scale: f32,
) -> F32x16 {
    // SAFETY: the caller guarantees every read.
    unsafe {
        let top = f32::load16(top);
        let mut sum = [vdupq_n_f32(0.0); 4];
        for c in 0..classes {
            let v = S::load16(p.add(c * cs));
            for k in 0..4 {
                let x = vmulq_n_f32(vsubq_f32(v[k], top[k]), scale);
                sum[k] = vaddq_f32(sum[k], exp_nonpositive4(x));
            }
        }
        sum.map(|s| vdivq_f32(vdupq_n_f32(1.0), s))
    }
}

/// The softmax confidence of `m` (a multiple of 16) pixels whose class `c` scores start at
/// `p + c * cs`, summed in `d` one class row segment at a time.
///
/// # Safety
/// `p + c * cs` must be valid for reading `m` scores for every `c < classes`, `top` for `m` and
/// `d` for writing `m`.
#[inline(always)]
unsafe fn softmax_strip<S: LoadF32>(
    p: *const S,
    cs: usize,
    classes: usize,
    top: *const f32,
    scale: f32,
    m: usize,
    d: *mut f32,
) {
    let bytes = m * size_of::<S>();
    // SAFETY: the caller guarantees every access.
    unsafe {
        for at in (0..m).step_by(BLOCK) {
            store_f32(d.add(at), [vdupq_n_f32(0.0); 4]);
        }
        for c in 0..classes {
            let q = p.add(c * cs);
            if c + 1 < classes {
                prefetch_span(q.add(cs), bytes);
            }
            for at in (0..m).step_by(BLOCK) {
                let (v, t, mut sum) = (
                    S::load16(q.add(at)),
                    f32::load16(top.add(at)),
                    f32::load16(d.add(at)),
                );
                for k in 0..4 {
                    let x = vmulq_n_f32(vsubq_f32(v[k], t[k]), scale);
                    sum[k] = vaddq_f32(sum[k], exp_nonpositive4(x));
                }
                store_f32(d.add(at), sum);
            }
        }
        for at in (0..m).step_by(BLOCK) {
            let sum = f32::load16(d.add(at));
            store_f32(d.add(at), sum.map(|s| vdivq_f32(vdupq_n_f32(1.0), s)));
        }
    }
}

/// [`softmax_strip`] over the whole blocks of `dst`, then [`softmax_block`] on the last 16
/// pixels when blocks leave a tail; see [`SemanticScore::softmax_planar_neon`]. A `cached`
/// source (a transposed tile in L1) runs [`softmax_block`] throughout.
fn softmax_planar<S: LoadF32>(
    src: &[S],
    cs: usize,
    classes: usize,
    top: &[f32],
    scale: f32,
    dst: &mut [f32],
    cached: bool,
) -> bool {
    let n = dst.len();
    if !planar_fits(src.len(), cs, classes, n) || top.len() != n {
        return false;
    }
    if n < BLOCK {
        let mut t = [0f32; BLOCK];
        t[..n].copy_from_slice(top);
        padded(src, cs, classes, dst, |tile, out| {
            softmax_planar(tile, BLOCK, classes, &t, scale, out, true);
        });
        return true;
    }
    let (p, t, d) = (src.as_ptr(), top.as_ptr(), dst.as_mut_ptr());
    if cached {
        // SAFETY: `planar_fits` bounds every class plane's `n` pixels and blocks end by `n`.
        for_blocks(n, |at| unsafe {
            let sum = softmax_block(p.add(at), cs, classes, t.add(at), scale);
            store_f32(d.add(at), sum);
        });
        return true;
    }
    let whole = n - n % BLOCK;
    // SAFETY: `planar_fits` bounds each plane's `n` pixels; strips and blocks end by `n`.
    unsafe {
        for at in (0..whole).step_by(STRIP) {
            let m = STRIP.min(whole - at);
            softmax_strip(p.add(at), cs, classes, t.add(at), scale, m, d.add(at));
        }
        if whole < n {
            let at = n - BLOCK;
            store_f32(
                d.add(at),
                softmax_block(p.add(at), cs, classes, t.add(at), scale),
            );
        }
    }
    true
}

/// Stores 16 f32 lanes.
///
/// # Safety
/// `p` must be valid for writing 16 elements.
#[inline(always)]
unsafe fn store_f32(p: *mut f32, v: F32x16) {
    // SAFETY: the caller guarantees 16 writable elements.
    unsafe {
        for (k, x) in v.into_iter().enumerate() {
            vst1q_f32(p.add(4 * k), x);
        }
    }
}

/// `max(p, 1 − p)` of `p = (x − zero_point) · scale`, after a sigmoid when `logit`.
fn binary_confidence<S: LoadF32>(
    src: &[S],
    zero_point: f32,
    scale: f32,
    logit: bool,
    dst: &mut [f32],
) -> bool {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer; NEON is baseline.
    map_steps(src, dst, |s, d| unsafe {
        let one = vdupq_n_f32(1.0);
        let p = S::load16(s).map(|x| {
            let p = vmulq_n_f32(vsubq_f32(x, vdupq_n_f32(zero_point)), scale);
            let p = if logit { sigmoid4(p) } else { p };
            vmaxnmq_f32(p, vsubq_f32(one, p))
        });
        store_f32(d, p);
    })
}

/// `1` where `x > cut`, else `0`, of each score of `src` as f32.
fn threshold<S: LoadF32>(src: &[S], cut: f32, dst: &mut [u16]) -> bool {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer; NEON is baseline.
    map_steps(src, dst, |s, d| unsafe {
        let m = S::load16(s).map(|x| vcgtq_f32(x, vdupq_n_f32(cut)));
        store_u16(d, mask_to_class(narrow_mask32(m)));
    })
}

/// A score type of the semantic decoder: `i8`, `u8`, `i16`, `u16`, `f32` or `f16`.
pub trait SemanticScore: Copy + Default {
    /// For each pixel `x < dst.len()`, the first class `c < classes` holding the largest
    /// `src[c * cs + x]` (strict `>` from class 0: NaN never wins and a NaN class 0 stays).
    /// `false` when a plane does not fit `src`, `classes` is 0 or above 65536, or `cs` is 0
    /// with several classes.
    #[doc(hidden)]
    fn argmax_planar_neon(src: &[Self], cs: usize, classes: usize, dst: &mut [u16]) -> bool;

    /// [`Self::argmax_planar_neon`] of pixels holding `classes` contiguous scores each,
    /// `src[x * classes + c]`; `false` when they do not fit `src` or `classes` is 0 or above
    /// 65536.
    #[doc(hidden)]
    fn argmax_interleaved_neon(src: &[Self], classes: usize, dst: &mut [u16]) -> bool;

    /// `1 / Σ_c exp_nonpositive(scale · (src[c * cs + x] − top[x]))` for each pixel `x`, summed
    /// from class 0; `false` as [`Self::argmax_planar_neon`] or when `top` and `dst` differ in
    /// length.
    #[doc(hidden)]
    fn softmax_planar_neon(
        src: &[Self],
        cs: usize,
        classes: usize,
        top: &[f32],
        scale: f32,
        dst: &mut [f32],
    ) -> bool;

    /// [`Self::softmax_planar_neon`] of interleaved scores `src[x * classes + c]`.
    #[doc(hidden)]
    fn softmax_interleaved_neon(
        src: &[Self],
        classes: usize,
        top: &[f32],
        scale: f32,
        dst: &mut [f32],
    ) -> bool;

    /// `max(p, 1 − p)` of `p = (x − zero_point) · scale` for each score `x` of `src`, with
    /// `p` through [`sigmoid_neon`]'s sigmoid first when `logit`; `false` when the lengths
    /// differ.
    #[doc(hidden)]
    fn binary_confidence_neon(
        src: &[Self],
        zero_point: f32,
        scale: f32,
        logit: bool,
        dst: &mut [f32],
    ) -> bool;

    /// `1` where a score of `src` as f32 is above `cut`, else `0`; `false` when the lengths
    /// differ.
    #[doc(hidden)]
    fn threshold_neon(src: &[Self], cut: f32, dst: &mut [u16]) -> bool;

    /// The scores of columns `cols` of one row, class by class and as f32, column-major:
    /// `out[j * classes + c] = src[c * cs + cols[j] * xs]`; `false` when a score is outside
    /// `src`, `classes` is 0, or `out` is not `cols.len() * classes` long.
    #[doc(hidden)]
    fn gather_columns_neon(
        src: &[Self],
        cs: usize,
        xs: usize,
        classes: usize,
        cols: &[u32],
        out: &mut [f32],
    ) -> bool;
}

macro_rules! semantic_score {
    ($t:ty, $lanes:ty) => {
        impl SemanticScore for $t {
            #[inline]
            fn argmax_planar_neon(
                src: &[Self],
                cs: usize,
                classes: usize,
                dst: &mut [u16],
            ) -> bool {
                argmax_planar::<$lanes>(src, cs, classes, dst, false)
            }

            #[inline]
            fn argmax_interleaved_neon(src: &[Self], classes: usize, dst: &mut [u16]) -> bool {
                interleaved(src, classes, dst, |t, out, _| {
                    argmax_planar::<$lanes>(t, out.len(), classes, out, true)
                })
            }

            #[inline]
            fn softmax_planar_neon(
                src: &[Self],
                cs: usize,
                classes: usize,
                top: &[f32],
                scale: f32,
                dst: &mut [f32],
            ) -> bool {
                softmax_planar(src, cs, classes, top, scale, dst, false)
            }

            #[inline]
            fn softmax_interleaved_neon(
                src: &[Self],
                classes: usize,
                top: &[f32],
                scale: f32,
                dst: &mut [f32],
            ) -> bool {
                top.len() == dst.len()
                    && interleaved(src, classes, dst, |t, out, first| {
                        top.get(first..first + out.len()).is_some_and(|top| {
                            softmax_planar(t, out.len(), classes, top, scale, out, true)
                        })
                    })
            }

            #[inline]
            fn binary_confidence_neon(
                src: &[Self],
                zero_point: f32,
                scale: f32,
                logit: bool,
                dst: &mut [f32],
            ) -> bool {
                binary_confidence(src, zero_point, scale, logit, dst)
            }

            #[inline]
            fn threshold_neon(src: &[Self], cut: f32, dst: &mut [u16]) -> bool {
                threshold(src, cut, dst)
            }

            #[inline]
            fn gather_columns_neon(
                src: &[Self],
                cs: usize,
                xs: usize,
                classes: usize,
                cols: &[u32],
                out: &mut [f32],
            ) -> bool {
                gather_columns(src, cs, xs, classes, cols, out)
            }
        }
    };
}
semantic_score!(u8, U8);
semantic_score!(i8, I8);
semantic_score!(u16, U16);
semantic_score!(i16, I16);
semantic_score!(f32, F32);
semantic_score!(f16, F16Widened);

/// [`SemanticScore::argmax_planar_neon`] for `f16` with FP16 compares.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline]
pub unsafe fn argmax_planar_f16_neon_fp16(
    src: &[f16],
    cs: usize,
    classes: usize,
    dst: &mut [u16],
) -> bool {
    argmax_planar::<F16Native>(src, cs, classes, dst, false)
}

/// [`SemanticScore::argmax_interleaved_neon`] for `f16` with FP16 compares.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline]
pub unsafe fn argmax_interleaved_f16_neon_fp16(
    src: &[f16],
    classes: usize,
    dst: &mut [u16],
) -> bool {
    interleaved(src, classes, dst, |t, out, _| {
        argmax_planar::<F16Native>(t, out.len(), classes, out, true)
    })
}

/// `1 / (1 + e^-x)` of each element by `neon_sigmoid::sigmoid4`, within `2^-21` of the exact
/// value relative to it; `false` when the lengths differ.
pub fn sigmoid_neon(src: &[f32], dst: &mut [f32]) -> bool {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
    map_steps(src, dst, |s, d| unsafe {
        store_f32(d, f32::load16(s).map(sigmoid4))
    })
}

/// `(x.clamp(0, 1) · 255 + 0.5) as u8` of each element; `false` when the lengths differ.
pub fn confidence_to_u8_neon(src: &[f32], dst: &mut [u8]) -> bool {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
    map_steps(src, dst, |s, d| unsafe {
        vst1q_u8(d, conf_u8(f32::load16(s)))
    })
}

/// `(x.clamp(0, 1) · 255 + 0.5) as u8` of 16 lanes: `FMAX`/`FMIN` keep NaN, `FCVTZU` sends it
/// to 0 and truncates, and the narrowing saturates.
#[inline(always)]
fn conf_u8(v: F32x16) -> uint8x16_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let u = v.map(|x| {
            let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
            vcvtq_u32_f32(vaddq_f32(vmulq_n_f32(x, 255.0), vdupq_n_f32(0.5)))
        });
        let lo = vqmovn_high_u32(vqmovn_u32(u[0]), u[1]);
        let hi = vqmovn_high_u32(vqmovn_u32(u[2]), u[3]);
        vqmovn_high_u16(vqmovn_u16(lo), hi)
    }
}

/// `wa · a + wb · b` per element (each product rounded, then the sum); `false` when the lengths
/// differ.
pub fn lerp_neon(a: &[f32], b: &[f32], wa: &[f32], wb: &[f32], dst: &mut [f32]) -> bool {
    let n = dst.len();
    if a.len() != n || b.len() != n || wa.len() != n || wb.len() != n {
        return false;
    }
    let body = n - n % 4;
    for i in (0..body).step_by(4) {
        // SAFETY: `i + 4 <= n` for every slice.
        unsafe {
            let x = vmulq_f32(vld1q_f32(wa.as_ptr().add(i)), vld1q_f32(a.as_ptr().add(i)));
            let y = vmulq_f32(vld1q_f32(wb.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            vst1q_f32(dst.as_mut_ptr().add(i), vaddq_f32(x, y));
        }
    }
    let tail = a[body..]
        .iter()
        .zip(&b[body..])
        .zip(wa[body..].iter().zip(&wb[body..]));
    for (d, ((&a, &b), (&wa, &wb))) in dst[body..].iter_mut().zip(tail) {
        *d = wa * a + wb * b;
    }
    true
}

/// The vertical blend `wa · a + wb · b` of two rows, 16 lanes.
///
/// # Safety
/// `a` and `b` must be valid for reading 16 elements.
#[inline(always)]
unsafe fn blend16(a: *const f32, b: *const f32, wa: f32, wb: f32) -> F32x16 {
    // SAFETY: the caller guarantees the reads.
    unsafe {
        let (a, b) = (f32::load16(a), f32::load16(b));
        [0, 1, 2, 3].map(|k| vaddq_f32(vmulq_n_f32(a[k], wa), vmulq_n_f32(b[k], wb)))
    }
}

/// `wa · a + wb · b` per element; `false` when the lengths differ.
pub fn blend_neon(a: &[f32], b: &[f32], wa: f32, wb: f32, dst: &mut [f32]) -> bool {
    // SAFETY: `zip_steps` hands out 16 valid elements at each pointer.
    zip_steps(a, b, dst, |a, b, d| unsafe {
        store_f32(d, blend16(a, b, wa, wb))
    })
}

/// [`confidence_to_u8_neon`] of [`blend_neon`]; `false` when the lengths differ.
pub fn blend_confidence_u8_neon(a: &[f32], b: &[f32], wa: f32, wb: f32, dst: &mut [u8]) -> bool {
    // SAFETY: `zip_steps` hands out 16 valid elements at each pointer.
    zip_steps(a, b, dst, |a, b, d| unsafe {
        vst1q_u8(d, conf_u8(blend16(a, b, wa, wb)))
    })
}

/// A class-map element: `u8` or `u16`.
pub trait ClassLane: Copy + Default {
    /// Stores 16 classes, truncated to the type.
    ///
    /// # Safety
    /// `p` must be valid for writing 16 elements.
    #[doc(hidden)]
    unsafe fn store_classes(p: *mut Self, v: [uint16x8_t; 2]);
}

impl ClassLane for u8 {
    #[inline(always)]
    unsafe fn store_classes(p: *mut u8, v: [uint16x8_t; 2]) {
        // SAFETY: the caller guarantees 16 writable elements.
        unsafe { vst1q_u8(p, vmovn_high_u16(vmovn_u16(v[0]), v[1])) }
    }
}

impl ClassLane for u16 {
    #[inline(always)]
    unsafe fn store_classes(p: *mut u16, v: [uint16x8_t; 2]) {
        // SAFETY: the caller guarantees 16 writable elements.
        unsafe { store_u16(p, v) }
    }
}

/// `1` where [`blend_neon`] is above `cut`, else `0`; `false` when the lengths differ.
pub fn blend_threshold_neon<O: ClassLane>(
    a: &[f32],
    b: &[f32],
    wa: f32,
    wb: f32,
    cut: f32,
    dst: &mut [O],
) -> bool {
    // SAFETY: `zip_steps` hands out 16 valid elements at each pointer; NEON is baseline.
    zip_steps(a, b, dst, |a, b, d| unsafe {
        let m = blend16(a, b, wa, wb).map(|x| vcgtq_f32(x, vdupq_n_f32(cut)));
        O::store_classes(d, mask_to_class(narrow_mask32(m)));
    })
}

/// [`SemanticScore::gather_columns_neon`]: interleaved classes convert a column at a time; class
/// planes gather a plane at a time, hinting the next plane's span of the row into L1 first.
fn gather_columns<S: LoadF32 + Into<f32>>(
    src: &[S],
    cs: usize,
    xs: usize,
    classes: usize,
    cols: &[u32],
    out: &mut [f32],
) -> bool {
    let (Some(&lo), Some(&hi)) = (cols.iter().min(), cols.iter().max()) else {
        return out.is_empty();
    };
    let last = (classes.wrapping_sub(1))
        .checked_mul(cs)
        .zip((hi as usize).checked_mul(xs))
        .and_then(|(a, b)| a.checked_add(b));
    if classes == 0 || last.is_none_or(|l| l >= src.len()) || out.len() != cols.len() * classes {
        return false;
    }
    if cs == 1 {
        for (&x, o) in cols.iter().zip(out.chunks_exact_mut(classes)) {
            let at = x as usize * xs;
            // SAFETY: `last` bounds the column; `map_steps` hands out 16 valid elements each.
            map_steps(&src[at..at + classes], o, |s, d| unsafe {
                store_f32(d, S::load16(s))
            });
        }
        return true;
    }
    let span = (hi - lo) as usize * xs * size_of::<S>() + size_of::<S>();
    for c in 0..classes {
        let row = c * cs;
        if c + 1 < classes {
            prefetch_span(src.as_ptr().wrapping_add(row + cs + lo as usize * xs), span);
        }
        for (j, &x) in cols.iter().enumerate() {
            // `last` bounds every class's columns, and `out` holds `cols.len() * classes`.
            out[j * classes + c] = src[row + x as usize * xs].into();
        }
    }
    true
}

/// The candidate classes of a boundary cell whose corner scores per class are `v00`, `v01`,
/// `v10` and `v11`, pushed in class order with their corners: every class whose largest corner
/// is at least the largest over classes of each class's smallest corner (`f32::min`/`f32::max`,
/// which ignore a NaN); `false` when the four slices differ in length or there are more than
/// 65536 classes.
pub fn cell_candidates_neon(
    v00: &[f32],
    v01: &[f32],
    v10: &[f32],
    v11: &[f32],
    classes: &mut Vec<u16>,
    corners: &mut Vec<[f32; 4]>,
) -> bool {
    let n = v00.len();
    if v01.len() != n || v10.len() != n || v11.len() != n || n > MAX_CLASSES {
        return false;
    }
    let body = n - n % 4;
    // SAFETY: NEON is part of the aarch64 baseline; every load reads 4 values below `body`.
    unsafe {
        let load = |v: &[f32], c: usize| vld1q_f32(v.as_ptr().add(c));
        let mut floor = vdupq_n_f32(f32::NEG_INFINITY);
        for c in (0..body).step_by(4) {
            let low = vminnmq_f32(
                vminnmq_f32(vminnmq_f32(load(v00, c), load(v01, c)), load(v10, c)),
                load(v11, c),
            );
            floor = vmaxnmq_f32(floor, low);
        }
        let mut floor = vmaxnmvq_f32(floor);
        for c in body..n {
            floor = floor.max(v00[c].min(v01[c]).min(v10[c]).min(v11[c]));
        }
        let fl = vdupq_n_f32(floor);
        for c in (0..body).step_by(4) {
            let (a, b, d, e) = (load(v00, c), load(v01, c), load(v10, c), load(v11, c));
            let high = vmaxnmq_f32(vmaxnmq_f32(vmaxnmq_f32(a, b), d), e);
            if vmaxvq_u32(vcgeq_f32(high, fl)) == 0 {
                continue;
            }
            for k in c..c + 4 {
                let v = [v00[k], v01[k], v10[k], v11[k]];
                if v[0].max(v[1]).max(v[2]).max(v[3]) >= floor {
                    classes.push(k as u16);
                    corners.push(v);
                }
            }
        }
        for k in body..n {
            let v = [v00[k], v01[k], v10[k], v11[k]];
            if v[0].max(v[1]).max(v[2]).max(v[3]) >= floor {
                classes.push(k as u16);
                corners.push(v);
            }
        }
    }
    true
}

/// For each pixel `i`, the first of `classes` (by position) whose corners `[v00, v01, v10, v11]`
/// interpolate highest at weights `(fy, fx[i])` as `(1 − fy) · ((1 − fx) · v00 + fx · v01) + fy
/// · ((1 − fx) · v10 + fx · v11)`, starting from class 0 at −∞ (so a pixel no candidate beats
/// gets 0); `false` when `classes` and `corners`, or `fx` and `dst`, differ in length.
pub fn cell_argmax_neon(
    classes: &[u16],
    corners: &[[f32; 4]],
    fy: f32,
    fx: &[f32],
    dst: &mut [u16],
) -> bool {
    if classes.len() != corners.len() || fx.len() != dst.len() {
        return false;
    }
    let wy0 = 1.0 - fy;
    for (fx, out) in fx.chunks(4).zip(dst.chunks_mut(4)) {
        let mut w = [0f32; 4];
        w[..fx.len()].copy_from_slice(fx);
        // SAFETY: NEON is part of the aarch64 baseline; `w` and `lanes` hold 4 elements.
        unsafe {
            let fx = vld1q_f32(w.as_ptr());
            let wx0 = vsubq_f32(vdupq_n_f32(1.0), fx);
            let mut best = vdupq_n_f32(f32::NEG_INFINITY);
            let mut idx = vdupq_n_u32(0);
            for (&class, v) in classes.iter().zip(corners) {
                let top = vaddq_f32(vmulq_n_f32(wx0, v[0]), vmulq_n_f32(fx, v[1]));
                let bottom = vaddq_f32(vmulq_n_f32(wx0, v[2]), vmulq_n_f32(fx, v[3]));
                let s = vaddq_f32(vmulq_n_f32(top, wy0), vmulq_n_f32(bottom, fy));
                let m = vcgtq_f32(s, best);
                best = vbslq_f32(m, s, best);
                idx = vbslq_u32(m, vdupq_n_u32(u32::from(class)), idx);
            }
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), idx);
            for (o, &l) in out.iter_mut().zip(&lanes) {
                *o = l as u16;
            }
        }
    }
    true
}

/// Source positions of a row resampling, read as `TBL` lookups into 64-byte windows of the
/// source row; blocks whose positions span more than a window read element by element.
pub struct RowGather {
    src_len: usize,
    bytes: usize,
    keep: usize,
    index: Vec<u32>,
    /// Per 16-output block: the window's first element and the lookup of each kept byte.
    blocks: Vec<Option<(usize, [uint8x16_t; 4])>>,
}

impl RowGather {
    /// Positions `index` into a row of `src_len` `bytes`-byte elements, keeping the low `keep`
    /// bytes of each; `None` when a position is outside the row.
    fn new(index: &[u32], src_len: usize, bytes: usize, keep: usize) -> Option<Self> {
        if index.iter().any(|&i| i as usize >= src_len) {
            return None;
        }
        let window = 64 / bytes;
        let n = index.len();
        let mut blocks = Vec::new();
        if n >= BLOCK {
            for_blocks(n, |at| {
                let pos = &index[at..at + BLOCK];
                let lo = pos.iter().copied().min().unwrap_or(0) as usize;
                let hi = pos.iter().copied().max().unwrap_or(0) as usize;
                let base = if src_len >= window {
                    lo.min(src_len - window)
                } else {
                    0
                };
                if hi >= base + window {
                    blocks.push(None);
                    return;
                }
                let mut tbl = [[0u8; 16]; 4];
                for (k, &p) in pos.iter().enumerate() {
                    for b in 0..keep {
                        let at = k * keep + b;
                        tbl[at / 16][at % 16] = ((p as usize - base) * bytes + b) as u8;
                    }
                }
                // SAFETY: NEON is part of the aarch64 baseline; each table holds 16 bytes.
                blocks.push(Some((base, tbl.map(|t| unsafe { vld1q_u8(t.as_ptr()) }))));
            });
        }
        Some(Self {
            src_len,
            bytes,
            keep,
            index: index.to_vec(),
            blocks,
        })
    }

    /// Positions of `u16` elements kept whole.
    pub fn u16(index: &[u32], src_len: usize) -> Option<Self> {
        Self::new(index, src_len, 2, 2)
    }

    /// Positions of `u16` elements kept as their low byte (`as u8`).
    pub fn u16_low_byte(index: &[u32], src_len: usize) -> Option<Self> {
        Self::new(index, src_len, 2, 1)
    }

    /// Positions of `f32` elements.
    pub fn f32(index: &[u32], src_len: usize) -> Option<Self> {
        Self::new(index, src_len, 4, 4)
    }

    /// The gather of `src` (`src_len` elements of `bytes` bytes) into `dst` (`index.len()`
    /// elements of `keep` bytes).
    fn run(&self, src: &[u8], dst: &mut [u8]) -> bool {
        let n = self.index.len();
        if src.len() != self.src_len * self.bytes || dst.len() != n * self.keep {
            return false;
        }
        let mut padded = [0u8; 64];
        let window = if src.len() < 64 {
            padded[..src.len()].copy_from_slice(src);
            &padded[..]
        } else {
            src
        };
        let one = |i: usize, p: u32, dst: &mut [u8]| {
            let at = p as usize * self.bytes;
            if let (Some(d), Some(s)) = (
                dst.get_mut(i * self.keep..(i + 1) * self.keep),
                src.get(at..at + self.keep),
            ) {
                d.copy_from_slice(s);
            }
        };
        if n < BLOCK {
            for (i, &p) in self.index.iter().enumerate() {
                one(i, p, dst);
            }
            return true;
        }
        let mut k = 0;
        for_blocks(n, |at| {
            match self.blocks.get(k) {
                Some(Some((base, tbl))) => {
                    // SAFETY: `new` placed the window in the row or padded copy; `dst` was checked.
                    unsafe {
                        let win = vld1q_u8_x4(window.as_ptr().add(base * self.bytes));
                        for (j, t) in tbl.iter().take(self.keep).enumerate() {
                            let out = dst.as_mut_ptr().add(at * self.keep + 16 * j);
                            vst1q_u8(out, vqtbl4q_u8(win, *t));
                        }
                    }
                }
                _ => {
                    for (i, &p) in self.index.iter().enumerate().skip(at).take(BLOCK) {
                        one(i, p, dst);
                    }
                }
            }
            k += 1;
        });
        true
    }
}

/// The bytes of a slice of plain numbers.
fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    // SAFETY: callers pass padding-free integer and float slices.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), std::mem::size_of_val(s)) }
}

/// The bytes of a mutable slice of plain numbers.
fn bytes_of_mut<T: Copy>(s: &mut [T]) -> &mut [u8] {
    // SAFETY: as in `bytes_of`; every byte pattern is a valid value of these types.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast(), std::mem::size_of_val(s)) }
}

/// `dst[i] = src[index[i]]` by a [`RowGather::u16`]; `false` when the plan or lengths differ.
pub fn gather_u16_neon(plan: &RowGather, src: &[u16], dst: &mut [u16]) -> bool {
    (plan.bytes, plan.keep) == (2, 2) && plan.run(bytes_of(src), bytes_of_mut(dst))
}

/// `dst[i] = src[index[i]] as u8` by a [`RowGather::u16_low_byte`]; `false` when the plan or
/// lengths differ.
pub fn gather_u16_to_u8_neon(plan: &RowGather, src: &[u16], dst: &mut [u8]) -> bool {
    (plan.bytes, plan.keep) == (2, 1) && plan.run(bytes_of(src), dst)
}

/// `dst[i] = src[index[i]]` by a [`RowGather::f32`]; `false` when the plan or lengths differ.
pub fn gather_f32_neon(plan: &RowGather, src: &[f32], dst: &mut [f32]) -> bool {
    (plan.bytes, plan.keep) == (4, 4) && plan.run(bytes_of(src), bytes_of_mut(dst))
}

/// A class-map input element of the decoder: `u8`, `u16`, `i32`, `u32` or `i64`.
pub trait ClassIndex: Copy {
    /// Each value of `src` as `u16`; `false` when a value is negative or not below `limit`
    /// (at most 65536), or the lengths differ, and `dst` is then unspecified.
    #[doc(hidden)]
    fn class_index_neon(src: &[Self], limit: usize, dst: &mut [u16]) -> bool;
}

/// [`ClassIndex::class_index_neon`] of 8 values at a time: `load` widens or narrows 8 values to
/// two `u32` vectors and a flag of any value at or above the limit.
#[inline(always)]
fn class_index<T: Copy>(
    src: &[T],
    limit: usize,
    dst: &mut [u16],
    load: impl Fn(*const T) -> ([uint32x4_t; 2], uint32x4_t),
) -> bool {
    if src.len() != dst.len() || limit > MAX_CLASSES {
        return false;
    }
    let body = src.len() - src.len() % 8;
    // SAFETY: NEON is part of the aarch64 baseline.
    let mut bad = unsafe { vdupq_n_u32(0) };
    for i in (0..body).step_by(8) {
        // SAFETY: `i + 8 <= len` for both slices.
        unsafe {
            let (v, over) = load(src.as_ptr().add(i));
            bad = vorrq_u32(bad, over);
            vst1q_u16(
                dst.as_mut_ptr().add(i),
                vmovn_high_u32(vmovn_u32(v[0]), v[1]),
            );
        }
    }
    // SAFETY: as above.
    if unsafe { vmaxvq_u32(bad) } != 0 {
        return false;
    }
    let (rest, out) = (&src[body..], &mut dst[body..]);
    let Some(&first) = rest.first() else {
        return true;
    };
    let mut buf = [first; 8];
    buf[..rest.len()].copy_from_slice(rest);
    let mut tail = [0u16; 8];
    if !class_index(&buf, limit, &mut tail, load) {
        return false;
    }
    out.copy_from_slice(&tail[..rest.len()]);
    true
}

impl ClassIndex for u8 {
    #[inline]
    fn class_index_neon(src: &[u8], limit: usize, dst: &mut [u16]) -> bool {
        let lim = (limit as u32).min(256);
        class_index(src, limit, dst, |p| {
            // SAFETY: `class_index` passes 8 readable values.
            unsafe {
                let w = vmovl_u8(vld1_u8(p));
                let v = [vmovl_u16(vget_low_u16(w)), vmovl_high_u16(w)];
                let l = vdupq_n_u32(lim);
                (v, vorrq_u32(vcgeq_u32(v[0], l), vcgeq_u32(v[1], l)))
            }
        })
    }
}

impl ClassIndex for u16 {
    #[inline]
    fn class_index_neon(src: &[u16], limit: usize, dst: &mut [u16]) -> bool {
        let lim = limit as u32;
        class_index(src, limit, dst, |p| {
            // SAFETY: `class_index` passes 8 readable values.
            unsafe {
                let w = vld1q_u16(p);
                let v = [vmovl_u16(vget_low_u16(w)), vmovl_high_u16(w)];
                let l = vdupq_n_u32(lim);
                (v, vorrq_u32(vcgeq_u32(v[0], l), vcgeq_u32(v[1], l)))
            }
        })
    }
}

/// Eight 32-bit values; a negative `i32` reads as a `u32` of at least 2^31, above any limit.
#[inline(always)]
fn words(p: *const u32, limit: usize) -> ([uint32x4_t; 2], uint32x4_t) {
    // SAFETY: callers pass 8 readable values; NEON is part of the aarch64 baseline.
    unsafe {
        let v = [vld1q_u32(p), vld1q_u32(p.add(4))];
        let l = vdupq_n_u32(limit as u32);
        (v, vorrq_u32(vcgeq_u32(v[0], l), vcgeq_u32(v[1], l)))
    }
}

impl ClassIndex for u32 {
    #[inline]
    fn class_index_neon(src: &[u32], limit: usize, dst: &mut [u16]) -> bool {
        class_index(src, limit, dst, |p| words(p, limit))
    }
}

impl ClassIndex for i32 {
    #[inline]
    fn class_index_neon(src: &[i32], limit: usize, dst: &mut [u16]) -> bool {
        class_index(src, limit, dst, |p| words(p.cast(), limit))
    }
}

impl ClassIndex for i64 {
    #[inline]
    fn class_index_neon(src: &[i64], limit: usize, dst: &mut [u16]) -> bool {
        class_index(src, limit, dst, |p| {
            // SAFETY: `class_index` passes 8 readable values; a negative `i64` is ≥ 2^63 as `u64`.
            unsafe {
                let p = p.cast::<u64>();
                let v = [0, 2, 4, 6].map(|k| vld1q_u64(p.add(k)));
                let l = vdupq_n_u64(limit as u64);
                let over = v.map(|x| vcgeq_u64(x, l));
                let over = vorrq_u32(
                    vreinterpretq_u32_u64(vorrq_u64(over[0], over[1])),
                    vreinterpretq_u32_u64(vorrq_u64(over[2], over[3])),
                );
                let w = [
                    vmovn_high_u64(vmovn_u64(v[0]), v[1]),
                    vmovn_high_u64(vmovn_u64(v[2]), v[3]),
                ];
                (w, over)
            }
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
