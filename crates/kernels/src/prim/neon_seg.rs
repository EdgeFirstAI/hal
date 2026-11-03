// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON kernels for ModelPack segmentation maps, bit-identical to the decoder's scalar code.
//!
//! Integer scores become `u8` by their top byte, sign bit flipped for signed types (`LD2`/`LD4`
//! de-interleave, `EOR`). Float scores are normalized as `((x − min) / (max − min) · 255) as u8`
//! over the map's minimum and maximum (0 and 255 when it holds a NaN), each step rounded in the
//! score type: f32, or f16 in `half` arithmetic (f32 operations rounded to f16 after each step,
//! or native FP16 instructions, which round the same since f32 carries twice f16's precision
//! plus two bits). A zero's sign in the minimum or maximum cannot change a `u8` result, so the
//! vector reductions may pick either. The per-pixel argmax over interleaved channels returns the
//! first channel holding the largest score, as a `u8` (the index modulo 256).

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR
use std::mem::MaybeUninit;

use half::f16;

use super::fp16_asm::{fcvtzu_u16_from_f16x8, fdiv_f16x8, fmul_f16x8, fsub_f16x8};
use super::neon_convert::{map_steps_uninit, round_to_f16, F32x16, LoadF32};

/// An integer score type of a segmentation map.
pub trait SegToU8: Copy + Default {
    /// The top byte of each score of `src`, sign bit flipped for signed types (`(x + 2^(n−1)) >>
    /// (n − 8)`), into `dst`; writes nothing and returns `false` when the lengths differ.
    #[doc(hidden)]
    fn seg_to_u8_neon(src: &[Self], dst: &mut [MaybeUninit<u8>]) -> bool;
}

macro_rules! seg_to_u8 {
    ($t:ty, |$p:ident| $top:expr, $flip:expr) => {
        impl SegToU8 for $t {
            #[inline]
            fn seg_to_u8_neon(src: &[Self], dst: &mut [MaybeUninit<u8>]) -> bool {
                if src.len() != dst.len() {
                    return false;
                }
                // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
                map_steps_uninit(src, dst, |s, d| unsafe {
                    let $p = s.cast::<u8>();
                    let top: uint8x16_t = $top;
                    let top = if $flip {
                        veorq_u8(top, vdupq_n_u8(0x80))
                    } else {
                        top
                    };
                    vst1q_u8(d, top);
                });
                true
            }
        }
    };
}
seg_to_u8!(i8, |p| vld1q_u8(p), true);
seg_to_u8!(u16, |p| vld2q_u8(p).1, false);
seg_to_u8!(i16, |p| vld2q_u8(p).1, true);
seg_to_u8!(u32, |p| vld4q_u8(p).3, false);
seg_to_u8!(i32, |p| vld4q_u8(p).3, true);

/// The smallest and largest value of `src` as f32, or `None` when it is empty or holds a NaN.
#[inline(always)]
fn min_max<S: LoadF32>(src: &[S]) -> Option<(f32, f32)> {
    let &first = src.first()?;
    // SAFETY: NEON is part of the aarch64 baseline; `FMIN`/`FMAX` propagate NaN.
    unsafe {
        let mut lo = [vdupq_n_f32(f32::INFINITY); 4];
        let mut hi = [vdupq_n_f32(f32::NEG_INFINITY); 4];
        let mut fold = |v: F32x16| {
            for k in 0..4 {
                lo[k] = vminq_f32(lo[k], v[k]);
                hi[k] = vmaxq_f32(hi[k], v[k]);
            }
        };
        let mut blocks = src.chunks_exact(16);
        for b in &mut blocks {
            fold(S::load16(b.as_ptr()));
        }
        let mut pad = [first; 16];
        for (p, &v) in pad.iter_mut().zip(blocks.remainder()) {
            *p = v;
        }
        fold(S::load16(pad.as_ptr()));
        let lo = vminvq_f32(vminq_f32(vminq_f32(lo[0], lo[1]), vminq_f32(lo[2], lo[3])));
        let hi = vmaxvq_f32(vmaxq_f32(vmaxq_f32(hi[0], hi[1]), vmaxq_f32(hi[2], hi[3])));
        (!lo.is_nan() && !hi.is_nan()).then_some((lo, hi))
    }
}

/// 16 f32 lanes `as u8`: truncated, saturated, NaN to 0 (`FCVTZU`, then `UQXTN` twice).
#[inline(always)]
fn to_u8(v: F32x16) -> uint8x16_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let [a, b, c, d] = v.map(|x| vcvtq_u32_f32(x));
        let lo = vcombine_u16(vqmovn_u32(a), vqmovn_u32(b));
        let hi = vcombine_u16(vqmovn_u32(c), vqmovn_u32(d));
        vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi))
    }
}

/// A float score type of a segmentation map.
pub trait SegNormalize: Copy + Default {
    /// `((x − min) / (max − min) · 255) as u8` for each `x` of `src` in `Self` arithmetic, with
    /// `min` and `max` the smallest and largest scores, or 0 and 255 when one is NaN; writes
    /// nothing and returns `false` when the lengths differ.
    #[doc(hidden)]
    fn seg_normalize_neon(src: &[Self], dst: &mut [MaybeUninit<u8>]) -> bool;
}

impl SegNormalize for f32 {
    #[inline]
    fn seg_normalize_neon(src: &[Self], dst: &mut [MaybeUninit<u8>]) -> bool {
        if src.len() != dst.len() {
            return false;
        }
        let (min, max) = min_max(src).unwrap_or((0.0, 255.0));
        // SAFETY: NEON is part of the aarch64 baseline.
        let (lo, range, k) =
            unsafe { (vdupq_n_f32(min), vdupq_n_f32(max - min), vdupq_n_f32(255.0)) };
        // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
        map_steps_uninit(src, dst, |s, d| unsafe {
            let v = f32::load16(s).map(|x| vmulq_f32(vdivq_f32(vsubq_f32(x, lo), range), k));
            vst1q_u8(d, to_u8(v));
        });
        true
    }
}

/// The `f16` minimum and maximum of `src` (0 and 255 for a NaN or an empty map) and their
/// difference rounded to f16, all as exact f32.
#[inline(always)]
fn f16_bounds(src: &[f16]) -> (f32, f32) {
    let (min, max) = min_max(src).unwrap_or((0.0, 255.0));
    (min, f16::from_f32(max - min).to_f32())
}

impl SegNormalize for f16 {
    #[inline]
    fn seg_normalize_neon(src: &[Self], dst: &mut [MaybeUninit<u8>]) -> bool {
        if src.len() != dst.len() {
            return false;
        }
        let (min, range) = f16_bounds(src);
        // SAFETY: NEON is part of the aarch64 baseline.
        let (lo, range, k) = unsafe { (vdupq_n_f32(min), vdupq_n_f32(range), vdupq_n_f32(255.0)) };
        // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
        map_steps_uninit(src, dst, |s, d| unsafe {
            let t = round_to_f16(f16::load16(s).map(|x| vsubq_f32(x, lo)));
            let q = round_to_f16(t.map(|x| vdivq_f32(x, range)));
            let m = round_to_f16(q.map(|x| vmulq_f32(x, k)));
            vst1q_u8(d, to_u8(m));
        });
        true
    }
}

/// [`SegNormalize::seg_normalize_neon`] for `f16` with FP16 arithmetic (`FSUB`, `FDIV`, `FMUL`,
/// `FCVTZU .8h`, `UQXTN`), the same bits.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[target_feature(enable = "neon")]
pub unsafe fn seg_normalize_f16_neon_fp16(src: &[f16], dst: &mut [MaybeUninit<u8>]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    let (min, range) = f16_bounds(src);
    let (lo, range) = (f16::from_f32(min).to_bits(), f16::from_f32(range).to_bits());
    let (lo, range, k) = (
        vdupq_n_u16(lo),
        vdupq_n_u16(range),
        vdupq_n_u16(f16::from_f32(255.0).to_bits()),
    );
    let step = |x: uint16x8_t| -> uint8x8_t {
        // SAFETY: the caller guarantees FEAT_FP16.
        unsafe {
            let x = fmul_f16x8(fdiv_f16x8(fsub_f16x8(x, lo), range), k);
            vqmovn_u16(fcvtzu_u16_from_f16x8(x))
        }
    };
    // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
    map_steps_uninit(src, dst, |s, d| unsafe {
        let s = s.cast::<u16>();
        let (a, b) = (vld1q_u16(s), vld1q_u16(s.add(8)));
        vst1q_u8(d, vcombine_u8(step(a), step(b)));
    });
    true
}

/// 16 byte-lane flags as a nibble each, lane 0 lowest (`SHRN`).
#[inline(always)]
fn nibbles(m: uint8x16_t) -> u64 {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            m,
        ))))
    }
}

/// The index of the first largest of `C` interleaved planes, per lane.
#[inline(always)]
fn argmax_planes<const C: usize>(planes: [uint8x16_t; C]) -> uint8x16_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let mut best = planes[0];
        let mut idx = vdupq_n_u8(0);
        for (c, &p) in planes.iter().enumerate().skip(1) {
            idx = vbslq_u8(vcgtq_u8(p, best), vdupq_n_u8(c as u8), idx);
            best = vmaxq_u8(best, p);
        }
        idx
    }
}

/// Sixteen pixels of `C` channels at a time through `step`, the tail through a padded block.
#[inline(always)]
fn pixels_by_16<const C: usize>(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    step: impl Fn(*const u8) -> uint8x16_t,
) {
    let mut s = src.chunks_exact(16 * C);
    let mut d = dst.chunks_exact_mut(16);
    for (si, di) in (&mut s).zip(&mut d) {
        // SAFETY: `si` holds 16 pixels and `di` 16 bytes.
        unsafe { vst1q_u8(di.as_mut_ptr().cast::<u8>(), step(si.as_ptr())) };
    }
    let (s, d) = (s.remainder(), d.into_remainder());
    if d.is_empty() {
        return;
    }
    let mut sbuf = [0u8; 64];
    let mut dbuf = [0u8; 16];
    for (b, &v) in sbuf.iter_mut().zip(s) {
        *b = v;
    }
    // SAFETY: `sbuf` holds 16 pixels of at most 4 channels and `dbuf` 16 bytes.
    unsafe { vst1q_u8(dbuf.as_mut_ptr(), step(sbuf.as_ptr())) };
    for (o, &v) in d.iter_mut().zip(&dbuf) {
        o.write(v);
    }
}

/// Lanes below `n`.
#[inline(always)]
fn lanes_below(n: usize) -> uint8x16_t {
    const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    // SAFETY: NEON is part of the aarch64 baseline; `IOTA` holds 16 bytes.
    unsafe { vcltq_u8(vld1q_u8(IOTA.as_ptr()), vdupq_n_u8(n.min(16) as u8)) }
}

/// The `n ≤ 16` bytes of `src` at `at`, zero in the lanes above.
#[inline(always)]
fn load_part(src: &[u8], at: usize, n: usize, below: uint8x16_t) -> uint8x16_t {
    // SAFETY: the direct load reads 16 bytes inside `src`; NEON is the aarch64 baseline.
    unsafe {
        if at + 16 <= src.len() {
            return vandq_u8(vld1q_u8(src.as_ptr().add(at)), below);
        }
        let mut buf = [0u8; 16];
        if let Some(part) = src.get(at..at + n) {
            for (b, &v) in buf.iter_mut().zip(part) {
                *b = v;
            }
        }
        vld1q_u8(buf.as_ptr())
    }
}

/// Channels up to which [`argmax_blocks`] runs: every index then fits a byte lane.
const MAX_TREE_CHANNELS: usize = 256;

/// Lane `p` of the result is the pairwise reduction `f` of the 16 lanes of `v[p]` (a tree of
/// 8 + 4 + 2 + 1 pairwise operations).
#[inline(always)]
fn tree(v: [uint8x16_t; 16], f: impl Fn(uint8x16_t, uint8x16_t) -> uint8x16_t) -> uint8x16_t {
    let [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15] = v;
    let (a0, a1, a2, a3) = (f(v0, v1), f(v2, v3), f(v4, v5), f(v6, v7));
    let (a4, a5, a6, a7) = (f(v8, v9), f(v10, v11), f(v12, v13), f(v14, v15));
    let (b0, b1, b2, b3) = (f(a0, a1), f(a2, a3), f(a4, a5), f(a6, a7));
    f(f(b0, b1), f(b2, b3))
}

/// `[f(0, lane 0 of m), …, f(15, lane 15 of m)]`, each lane broadcast with `DUP`.
macro_rules! per_lane {
    ($m:expr, |$px:ident, $t:ident| $body:expr) => {
        per_lane!(@ $m, $px, $t, $body, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    (@ $m:expr, $px:ident, $t:ident, $body:expr, $($n:literal)*) => {
        [$({
            let $px: usize = $n;
            let $t = vdupq_laneq_u8::<$n>($m);
            $body
        }),*]
    };
}

/// Channel indices 0 to 15.
const IOTA: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// The first-max channel of the 16 pixels of `c ≤ 16` channels at `p`, one per lane: the
/// pixels' maxima by a pairwise-max tree, then the smallest channel equal to each by a
/// pairwise-min tree; lanes past `c` are zeroed by `tail`.
///
/// # Safety
/// `p` must be valid for reading `15 · c + 16` bytes.
#[inline(always)]
unsafe fn argmax_block_short(p: *const u8, c: usize, tail: uint8x16_t) -> uint8x16_t {
    // SAFETY: the caller guarantees the reads; NEON is part of the aarch64 baseline.
    unsafe {
        let mut q = p;
        let mut v = [tail; 16];
        for v in &mut v {
            *v = vandq_u8(vld1q_u8(q), tail);
            q = q.wrapping_add(c);
        }
        let max = tree(v, |a, b| vpmaxq_u8(a, b));
        let (iota, none) = (vld1q_u8(IOTA.as_ptr()), vdupq_n_u8(u8::MAX));
        let first = per_lane!(max, |px, target| vbslq_u8(
            vceqq_u8(v[px], target),
            iota,
            none
        ));
        tree(first, |a, b| vpminq_u8(a, b))
    }
}

/// [`argmax_block_short`] for `c` of 17 to 32: two 16-channel chunks per pixel, the second's
/// lanes past `c` zeroed by `tail`.
///
/// # Safety
/// `p` must be valid for reading `15 · c + 32` bytes.
#[inline(always)]
unsafe fn argmax_block_pair(p: *const u8, c: usize, tail: uint8x16_t) -> uint8x16_t {
    // SAFETY: the caller guarantees the reads; NEON is part of the aarch64 baseline.
    unsafe {
        let pair = |q: *const u8| (vld1q_u8(q), vandq_u8(vld1q_u8(q.wrapping_add(16)), tail));
        let mut q = p;
        let mut lanes = [tail; 16];
        for l in &mut lanes {
            let (lo, hi) = pair(q);
            *l = vmaxq_u8(lo, hi);
            q = q.wrapping_add(c);
        }
        let max = tree(lanes, |a, b| vpmaxq_u8(a, b));
        let (iota, none) = (vld1q_u8(IOTA.as_ptr()), vdupq_n_u8(u8::MAX));
        let iota_hi = vaddq_u8(iota, vdupq_n_u8(16));
        let first = per_lane!(max, |px, target| {
            let (lo, hi) = pair(p.wrapping_add(px * c));
            vminq_u8(
                vbslq_u8(vceqq_u8(lo, target), iota, none),
                vbslq_u8(vceqq_u8(hi, target), iota_hi, none),
            )
        });
        tree(first, |a, b| vpminq_u8(a, b))
    }
}

/// [`argmax_block_short`] for `c` of 33 to [`MAX_TREE_CHANNELS`]: each pixel's lanewise
/// maximum over its 16-channel chunks, the last chunk's lanes past `c` zeroed by `tail`.
///
/// # Safety
/// `p` must be valid for reading `15 · c + 16 · ⌈c / 16⌉` bytes.
#[inline(always)]
unsafe fn argmax_block_long(p: *const u8, c: usize, tail: uint8x16_t) -> uint8x16_t {
    let full = (c - 1) / 16;
    // SAFETY: the caller guarantees the reads; NEON is part of the aarch64 baseline.
    unsafe {
        let last = |px: usize| vandq_u8(vld1q_u8(p.add(px * c + 16 * full)), tail);
        let mut lanes = [tail; 16];
        for (px, l) in lanes.iter_mut().enumerate() {
            let mut m = last(px);
            for k in 0..full {
                m = vmaxq_u8(m, vld1q_u8(p.add(px * c + 16 * k)));
            }
            *l = m;
        }
        let max = tree(lanes, |a, b| vpmaxq_u8(a, b));
        let (iota, none) = (vld1q_u8(IOTA.as_ptr()), vdupq_n_u8(u8::MAX));
        let step = vdupq_n_u8(16);
        let first = per_lane!(max, |px, target| {
            let mut at = iota;
            let mut best = none;
            for k in 0..full {
                let eq = vceqq_u8(vld1q_u8(p.add(px * c + 16 * k)), target);
                best = vminq_u8(best, vbslq_u8(eq, at, none));
                at = vaddq_u8(at, step);
            }
            vminq_u8(best, vbslq_u8(vceqq_u8(last(px), target), at, none))
        });
        tree(first, |a, b| vpminq_u8(a, b))
    }
}

/// [`channel_argmax_u8_neon`] for 5 to [`MAX_TREE_CHANNELS`] channels, 16 pixels per `block`;
/// blocks whose reads would pass the end of `src` go through a padded stack copy.
#[inline(always)]
fn argmax_blocks(
    src: &[u8],
    c: usize,
    dst: &mut [MaybeUninit<u8>],
    block: impl Fn(*const u8, uint8x16_t) -> uint8x16_t,
) {
    let span = 15 * c + 16 * c.div_ceil(16);
    let tail = lanes_below(match c % 16 {
        0 => 16,
        r => r,
    });
    let padded = |pixels: &[u8]| {
        let mut pad = [0u8; 16 * MAX_TREE_CHANNELS + 16];
        for (o, &v) in pad.iter_mut().zip(pixels) {
            *o = v;
        }
        let mut out = [0u8; 16];
        // SAFETY: `pad` holds `16 · c + 16 ≥ span` bytes and `out` 16.
        unsafe { vst1q_u8(out.as_mut_ptr(), block(pad.as_ptr(), tail)) };
        out
    };
    let mut blocks = dst.chunks_exact_mut(16);
    let mut at = 0;
    for d in &mut blocks {
        if at + span <= src.len() {
            // SAFETY: `span` bytes from `at` lie inside `src`; `d` holds 16 bytes.
            unsafe {
                vst1q_u8(
                    d.as_mut_ptr().cast::<u8>(),
                    block(src.as_ptr().add(at), tail),
                )
            };
        } else {
            let out = padded(src.get(at..at + 16 * c).unwrap_or_default());
            for (o, &v) in d.iter_mut().zip(&out) {
                o.write(v);
            }
        }
        at += 16 * c;
    }
    let rest = blocks.into_remainder();
    if !rest.is_empty() {
        let out = padded(src.get(at..).unwrap_or_default());
        for (o, &v) in rest.iter_mut().zip(&out) {
            o.write(v);
        }
    }
}

/// For each pixel of `src`, `channels` interleaved scores, the index of the first largest score
/// as `u8` (the index modulo 256) into `dst`.
///
/// Writes every element of `dst` and returns `true`, or writes nothing and returns `false` when
/// `channels` is 0 or `src` does not hold `dst.len()` pixels.
pub fn channel_argmax_u8_neon(src: &[u8], channels: usize, dst: &mut [MaybeUninit<u8>]) -> bool {
    if channels == 0 || Some(src.len()) != dst.len().checked_mul(channels) {
        return false;
    }
    // SAFETY (each step): the pointer holds 16 pixels; NEON is part of the aarch64 baseline.
    match channels {
        1 => {
            for o in dst.iter_mut() {
                o.write(0);
            }
        }
        2 => pixels_by_16::<2>(src, dst, |p| unsafe {
            let v = vld2q_u8(p);
            argmax_planes([v.0, v.1])
        }),
        3 => pixels_by_16::<3>(src, dst, |p| unsafe {
            let v = vld3q_u8(p);
            argmax_planes([v.0, v.1, v.2])
        }),
        4 => pixels_by_16::<4>(src, dst, |p| unsafe {
            let v = vld4q_u8(p);
            argmax_planes([v.0, v.1, v.2, v.3])
        }),
        // SAFETY (each block): `argmax_blocks` hands out pointers to `span` readable bytes.
        c if c <= 16 => argmax_blocks(src, c, dst, |p, tail| unsafe {
            argmax_block_short(p, c, tail)
        }),
        c if c <= 32 => argmax_blocks(src, c, dst, |p, tail| unsafe {
            argmax_block_pair(p, c, tail)
        }),
        c if c <= MAX_TREE_CHANNELS => argmax_blocks(src, c, dst, |p, tail| unsafe {
            argmax_block_long(p, c, tail)
        }),
        c => {
            let (full, rest) = (c / 16, c % 16);
            let below = lanes_below(rest);
            for (k, o) in dst.iter_mut().enumerate() {
                let at = k * c;
                let part = |j: usize| {
                    if j < full {
                        // SAFETY: the 16 bytes lie inside this pixel.
                        unsafe { vld1q_u8(src.as_ptr().add(at + 16 * j)) }
                    } else {
                        load_part(src, at + 16 * j, rest, below)
                    }
                };
                let parts = full + usize::from(rest > 0);
                // SAFETY: NEON is part of the aarch64 baseline.
                let target = unsafe {
                    let mut m = vdupq_n_u8(0);
                    for j in 0..parts {
                        m = vmaxq_u8(m, part(j));
                    }
                    vdupq_n_u8(vmaxvq_u8(m))
                };
                let mut idx = 0;
                for j in 0..parts {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    let bits = unsafe { nibbles(vceqq_u8(part(j), target)) };
                    if bits != 0 {
                        idx = 16 * j + bits.trailing_zeros() as usize / 4;
                        break;
                    }
                }
                o.write(idx as u8);
            }
        }
    }
    true
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
