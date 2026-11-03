// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON elementwise conversions, bit-identical to their scalar counterparts on every input.
//!
//! Each kernel converts 16 elements per step and runs its tail through the same vector code on
//! a zero-padded stack block, so short slices are vectorized too; the 8-bit → f32 dequantizers
//! and the integer → f32 `scale_bias` kernels are a plain loop the compiler vectorizes. Float
//! arithmetic is done in the scalar order with no FMA, and f16 conversions use the ARMv8.0
//! `FCVTL`/`FCVTN`, so results match `half` and `prim::dequant` bit for bit, NaN payloads
//! included. The `scale_bias` kernels compute the decoder's generic dequantize
//! `x · scale + bias` into possibly uninitialized outputs. Every kernel writes nothing when its
//! slices differ in length.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR
use std::mem::MaybeUninit;

use half::f16;

use crate::Quant;

/// Elements per vector step.
const STEP: usize = 16;

/// Four f32 vectors: one step.
pub(crate) type F32x16 = [float32x4_t; 4];

/// Runs `step` over 16-element blocks of `src` and `dst`; the tail goes through padded blocks.
///
/// `step` reads 16 `S` at its first pointer and writes 16 `D` at its second.
#[inline(always)]
pub(crate) fn map_steps<S: Copy + Default, D: Copy + Default>(
    src: &[S],
    dst: &mut [D],
    step: impl Fn(*const S, *mut D),
) {
    if src.len() != dst.len() {
        return;
    }
    let mut s = src.chunks_exact(STEP);
    let mut d = dst.chunks_exact_mut(STEP);
    for (si, di) in (&mut s).zip(&mut d) {
        step(si.as_ptr(), di.as_mut_ptr());
    }
    let (s, d) = (s.remainder(), d.into_remainder());
    if s.is_empty() {
        return;
    }
    let mut sbuf = [S::default(); STEP];
    let mut dbuf = [D::default(); STEP];
    for (b, &v) in sbuf.iter_mut().zip(s) {
        *b = v;
    }
    step(sbuf.as_ptr(), dbuf.as_mut_ptr());
    for (o, &v) in d.iter_mut().zip(&dbuf) {
        *o = v;
    }
}

/// `FCVTL`/`FCVTL2`: eight f16 lanes to two f32 vectors.
#[inline(always)]
pub(crate) fn widen8(h: uint16x8_t) -> [float32x4_t; 2] {
    let (lo, hi): (float32x4_t, float32x4_t);
    // SAFETY: register-only ARMv8.0 Advanced SIMD instructions.
    unsafe {
        core::arch::asm!(
            "fcvtl {lo:v}.4s, {h:v}.4h",
            "fcvtl2 {hi:v}.4s, {h:v}.8h",
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            h = in(vreg) h,
            options(pure, nomem, nostack),
        );
    }
    [lo, hi]
}

/// `FCVTN`/`FCVTN2`: two f32 vectors to eight f16 lanes, round to nearest even.
#[inline(always)]
pub(crate) fn narrow8(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    let r: uint16x8_t;
    // SAFETY: register-only ARMv8.0 Advanced SIMD instructions.
    unsafe {
        core::arch::asm!(
            "fcvtn {r:v}.4h, {lo:v}.4s",
            "fcvtn2 {r:v}.8h, {hi:v}.4s",
            r = out(vreg) r,
            lo = in(vreg) lo,
            hi = in(vreg) hi,
            options(pure, nomem, nostack),
        );
    }
    r
}

/// Loads 16 elements as f32 lanes, exactly.
pub(crate) trait LoadF32: Copy + Default {
    /// # Safety
    /// `p` must be valid for reading 16 elements.
    unsafe fn load16(p: *const Self) -> F32x16;
}

/// `TBL` indices that move bytes `4k..4k + 4` into byte `at` of four 32-bit lanes, zeroing the rest.
const fn byte_lanes(k: usize, at: usize) -> [u8; 16] {
    let mut idx = [0xff; 16];
    let mut lane = 0;
    while lane < 4 {
        idx[lane * 4 + at] = (4 * k + lane) as u8;
        lane += 1;
    }
    idx
}

/// Spreads 16 bytes into four vectors of 32-bit lanes, byte `AT` of each lane.
///
/// # Safety
/// `p` must be valid for reading 16 bytes.
#[inline(always)]
unsafe fn spread_bytes<const AT: usize>(p: *const u8) -> [uint8x16_t; 4] {
    const fn table(at: usize) -> [[u8; 16]; 4] {
        [
            byte_lanes(0, at),
            byte_lanes(1, at),
            byte_lanes(2, at),
            byte_lanes(3, at),
        ]
    }
    let t = const { table(AT) };
    // SAFETY: the caller guarantees 16 readable bytes; the tables are 16 bytes each.
    unsafe {
        let v = vld1q_u8(p);
        [
            vqtbl1q_u8(v, vld1q_u8(t[0].as_ptr())),
            vqtbl1q_u8(v, vld1q_u8(t[1].as_ptr())),
            vqtbl1q_u8(v, vld1q_u8(t[2].as_ptr())),
            vqtbl1q_u8(v, vld1q_u8(t[3].as_ptr())),
        ]
    }
}

impl LoadF32 for i8 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        // The byte sits in the top of each lane, so a 24-bit fixed-point conversion is exact.
        // SAFETY: the caller guarantees 16 readable elements; NEON is the aarch64 baseline.
        unsafe {
            spread_bytes::<3>(p.cast()).map(|v| vcvtq_n_f32_s32::<24>(vreinterpretq_s32_u8(v)))
        }
    }
}

impl LoadF32 for u8 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable elements; NEON is the aarch64 baseline.
        unsafe { spread_bytes::<0>(p).map(|v| vcvtq_f32_u32(vreinterpretq_u32_u8(v))) }
    }
}

impl LoadF32 for i16 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let (a, b) = (vld1q_s16(p), vld1q_s16(p.add(8)));
            [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))),
                vcvtq_f32_s32(vmovl_high_s16(a)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))),
                vcvtq_f32_s32(vmovl_high_s16(b)),
            ]
        }
    }
}

impl LoadF32 for u16 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let (a, b) = (vld1q_u16(p), vld1q_u16(p.add(8)));
            [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(a))),
                vcvtq_f32_u32(vmovl_high_u16(a)),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(b))),
                vcvtq_f32_u32(vmovl_high_u16(b)),
            ]
        }
    }
}

impl LoadF32 for f16 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        let p = p.cast::<u16>();
        // SAFETY: the caller guarantees 16 readable elements; `f16` is a transparent `u16`.
        let (a, b) = unsafe { (vld1q_u16(p), vld1q_u16(p.add(8))) };
        let ([a0, a1], [b0, b1]) = (widen8(a), widen8(b));
        [a0, a1, b0, b1]
    }
}

impl LoadF32 for f32 {
    #[inline(always)]
    unsafe fn load16(p: *const Self) -> F32x16 {
        // SAFETY: the caller guarantees 16 readable elements.
        unsafe {
            [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
            ]
        }
    }
}

/// Stores 16 f32 lanes as f32 or, rounded to nearest even, as f16.
pub(crate) trait StoreFloat: Copy + Default {
    /// # Safety
    /// `p` must be valid for writing 16 elements.
    unsafe fn store16(p: *mut Self, v: F32x16);
}

impl StoreFloat for f32 {
    #[inline(always)]
    unsafe fn store16(p: *mut Self, v: F32x16) {
        // SAFETY: the caller guarantees 16 writable elements.
        unsafe {
            vst1q_f32(p, v[0]);
            vst1q_f32(p.add(4), v[1]);
            vst1q_f32(p.add(8), v[2]);
            vst1q_f32(p.add(12), v[3]);
        }
    }
}

impl StoreFloat for f16 {
    #[inline(always)]
    unsafe fn store16(p: *mut Self, v: F32x16) {
        let p = p.cast::<u16>();
        // SAFETY: the caller guarantees 16 writable elements; `f16` is a transparent `u16`.
        unsafe {
            vst1q_u16(p, narrow8(v[0], v[1]));
            vst1q_u16(p.add(8), narrow8(v[2], v[3]));
        }
    }
}

/// `(x − zero_point) · scale` per lane, as `prim::dequant` computes it.
#[inline(always)]
pub(crate) fn affine(v: F32x16, zp: float32x4_t, scale: float32x4_t) -> F32x16 {
    // SAFETY: NEON is part of the aarch64 baseline.
    v.map(|x| unsafe { vmulq_f32(vsubq_f32(x, zp), scale) })
}

/// `prim::dequant` for any input and float output: plain conversion for `Quant::IDENTITY`.
#[inline(always)]
fn dequant<S: LoadF32, D: StoreFloat>(src: &[S], q: Quant, dst: &mut [D]) {
    if q == Quant::IDENTITY {
        // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
        map_steps(src, dst, |s, d| unsafe { D::store16(d, S::load16(s)) });
        return;
    }
    // SAFETY: NEON is part of the aarch64 baseline.
    let (zp, scale) = unsafe { (vdupq_n_f32(q.zero_point as f32), vdupq_n_f32(q.scale)) };
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
    map_steps(src, dst, |s, d| unsafe {
        D::store16(d, affine(S::load16(s), zp, scale));
    });
}

/// `round_half_away(x · inv_scale) + zero_point` per lane, saturating in i32.
#[inline(always)]
fn quantize_i32(v: F32x16, inv_scale: float32x4_t, zp: int32x4_t) -> [int32x4_t; 4] {
    // FCVTAS rounds ties away from zero, saturates, and maps NaN to 0, as `quantize` does.
    // SAFETY: NEON is part of the aarch64 baseline.
    v.map(|x| unsafe { vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(x, inv_scale)), zp) })
}

macro_rules! quantize_kernel {
    ($(#[$doc:meta])* $name:ident, $t:ty, |$q:ident, $p:ident| { $($store:tt)* }) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(src: &[f32], inv_scale: f32, zero_point: i32, dst: &mut [$t]) {
            let inv = vdupq_n_f32(inv_scale);
            let zp = vdupq_n_s32(zero_point);
            map_steps(src, dst, |s, $p| {
                // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
                let $q = quantize_i32(unsafe { f32::load16(s) }, inv, zp);
                // SAFETY: as above.
                unsafe { $($store)* }
            });
        }
    };
}

quantize_kernel!(
    /// [`quantize`](crate::quantize) of every element into `i8`, bit for bit.
    quantize_f32_to_i8_neon,
    i8,
    |q, p| {
        let a = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
        let b = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
        vst1q_s8(p, vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)));
    }
);

quantize_kernel!(
    /// [`quantize`](crate::quantize) of every element into `u8`, bit for bit.
    quantize_f32_to_u8_neon,
    u8,
    |q, p| {
        let a = vcombine_u16(vqmovun_s32(q[0]), vqmovun_s32(q[1]));
        let b = vcombine_u16(vqmovun_s32(q[2]), vqmovun_s32(q[3]));
        vst1q_u8(p, vcombine_u8(vqmovn_u16(a), vqmovn_u16(b)));
    }
);

quantize_kernel!(
    /// [`quantize`](crate::quantize) of every element into `i16`, bit for bit.
    quantize_f32_to_i16_neon,
    i16,
    |q, p| {
        vst1q_s16(p, vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1])));
        vst1q_s16(p.add(8), vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3])));
    }
);

quantize_kernel!(
    /// [`quantize`](crate::quantize) of every element into `u16`, bit for bit.
    quantize_f32_to_u16_neon,
    u16,
    |q, p| {
        vst1q_u16(p, vcombine_u16(vqmovun_s32(q[0]), vqmovun_s32(q[1])));
        vst1q_u16(p.add(8), vcombine_u16(vqmovun_s32(q[2]), vqmovun_s32(q[3])));
    }
);

/// `f16::from_f32` of every element (`FCVTN`, round to nearest even), bit for bit.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn narrow_f32_to_f16_neon(src: &[f32], dst: &mut [f16]) {
    dequant(src, Quant::IDENTITY, dst);
}

/// `f16::to_f32` of every element (`FCVTL`), bit for bit.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn widen_f16_to_f32_neon(src: &[f16], dst: &mut [f32]) {
    dequant(src, Quant::IDENTITY, dst);
}

/// [`dequant`] for one type in and out: `Quant::IDENTITY` copies, as `prim::dequant` does.
#[inline(always)]
fn dequant_same<T: LoadF32 + StoreFloat>(src: &[T], q: Quant, dst: &mut [T]) {
    if q != Quant::IDENTITY {
        dequant(src, q, dst);
    } else if src.len() == dst.len() {
        dst.copy_from_slice(src);
    }
}

macro_rules! dequant_kernel {
    ($(#[$doc:meta])* $name:ident, $s:ty, $d:ty $(, $body:ident)?) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(src: &[$s], q: Quant, dst: &mut [$d]) {
            dequant_kernel!(@body $($body)?)(src, q, dst);
        }
    };
    (@body) => { dequant };
    (@body $body:ident) => { $body };
}

dequant_kernel!(
    /// `prim::dequant::dequant_f32_to_f32`, bit for bit; `Quant::IDENTITY` copies.
    dequant_f32_to_f32_neon,
    f32,
    f32,
    dequant_same
);
dequant_kernel!(
    /// `prim::dequant::dequant_f16_to_f16`, bit for bit; `Quant::IDENTITY` copies.
    dequant_f16_to_f16_neon,
    f16,
    f16,
    dequant_same
);
dequant_kernel!(
    /// `prim::dequant::dequant_f16_to_f32`, bit for bit.
    dequant_f16_to_f32_neon,
    f16,
    f32
);
dequant_kernel!(
    /// `prim::dequant::dequant_f32_to_f16`, bit for bit.
    dequant_f32_to_f16_neon,
    f32,
    f16
);

/// `(x − zero_point) · scale` as a plain loop the compiler vectorizes; it must stay FMA-free.
#[inline(always)]
fn dequant_loop<S: Copy + Into<f32>>(src: &[S], q: Quant, dst: &mut [f32]) {
    if src.len() != dst.len() {
        return;
    }
    let zp = q.zero_point as f32;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = (s.into() - zp) * q.scale;
    }
}

dequant_kernel!(
    /// `prim::dequant::dequant_i8_to_f32`, bit for bit (no FMA, unlike
    /// `neon_baseline::dequant_i8_to_f32_neon`).
    dequant_i8_to_f32_exact_neon,
    i8,
    f32,
    dequant_loop
);
dequant_kernel!(
    /// `prim::dequant::dequant_u8_to_f32`, bit for bit.
    dequant_u8_to_f32_exact_neon,
    u8,
    f32,
    dequant_loop
);
dequant_kernel!(
    /// `prim::dequant::dequant_i16_to_f32`, bit for bit.
    dequant_i16_to_f32_exact_neon,
    i16,
    f32
);
dequant_kernel!(
    /// `prim::dequant::dequant_u16_to_f32`, bit for bit.
    dequant_u16_to_f32_exact_neon,
    u16,
    f32
);
dequant_kernel!(
    /// `prim::dequant::dequant_i8_to_f16`, bit for bit (no FMA, unlike
    /// `neon_baseline::dequant_i8_to_f16_neon`).
    dequant_i8_to_f16_exact_neon,
    i8,
    f16
);
dequant_kernel!(
    /// `prim::dequant::dequant_u8_to_f16`, bit for bit.
    dequant_u8_to_f16_exact_neon,
    u8,
    f16
);
dequant_kernel!(
    /// `prim::dequant::dequant_i16_to_f16`, bit for bit.
    dequant_i16_to_f16_exact_neon,
    i16,
    f16
);
dequant_kernel!(
    /// `prim::dequant::dequant_u16_to_f16`, bit for bit.
    dequant_u16_to_f16_exact_neon,
    u16,
    f16
);

/// [`map_steps`] into a possibly uninitialized `dst`, every element of which it writes when the
/// lengths match.
#[inline(always)]
pub(crate) fn map_steps_uninit<S: Copy + Default, D: Copy + Default>(
    src: &[S],
    dst: &mut [MaybeUninit<D>],
    step: impl Fn(*const S, *mut D),
) {
    if src.len() != dst.len() {
        return;
    }
    let mut s = src.chunks_exact(STEP);
    let mut d = dst.chunks_exact_mut(STEP);
    for (si, di) in (&mut s).zip(&mut d) {
        step(si.as_ptr(), di.as_mut_ptr().cast::<D>());
    }
    let (s, d) = (s.remainder(), d.into_remainder());
    if s.is_empty() {
        return;
    }
    let mut sbuf = [S::default(); STEP];
    let mut dbuf = [D::default(); STEP];
    for (b, &v) in sbuf.iter_mut().zip(s) {
        *b = v;
    }
    step(sbuf.as_ptr(), dbuf.as_mut_ptr());
    for (o, &v) in d.iter_mut().zip(&dbuf) {
        o.write(v);
    }
}

/// `x · scale + bias` (`x · scale` without a bias) into f32 as a plain loop the compiler
/// vectorizes, rounded after the product and after the sum.
#[inline(always)]
fn scale_bias_loop<S: Copy + Into<f32>>(
    src: &[S],
    scale: f32,
    bias: Option<f32>,
    dst: &mut [MaybeUninit<f32>],
) {
    if src.len() != dst.len() {
        return;
    }
    match bias {
        Some(b) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                d.write(s.into() * scale + b);
            }
        }
        None => {
            for (d, &s) in dst.iter_mut().zip(src) {
                d.write(s.into() * scale);
            }
        }
    }
}

/// Each f32 lane rounded to f16 and back (`FCVTN` then `FCVTL`).
#[inline(always)]
pub(crate) fn round_to_f16(v: F32x16) -> F32x16 {
    let ([a, b], [c, d]) = (widen8(narrow8(v[0], v[1])), widen8(narrow8(v[2], v[3])));
    [a, b, c, d]
}

/// `x · scale + bias` into f16 with `half`'s roundings: `x` to f16, the product to f16, the sum
/// to f16. The f32 product of two f16 values is exact and an f32 sum of two rounds to the
/// correctly rounded f16 sum, so each step equals the f16 operation.
#[inline(always)]
fn scale_bias_f16<S: LoadF32>(
    src: &[S],
    scale: f16,
    bias: Option<f16>,
    dst: &mut [MaybeUninit<f16>],
) {
    let exact_in_f16 = std::mem::size_of::<S>() == 1;
    let to_f16 = |x: F32x16| if exact_in_f16 { x } else { round_to_f16(x) };
    // SAFETY: NEON is part of the aarch64 baseline.
    let sc = unsafe { vdupq_n_f32(scale.to_f32()) };
    match bias {
        Some(b) => {
            // SAFETY: as above.
            let b = unsafe { vdupq_n_f32(b.to_f32()) };
            // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
            map_steps_uninit(src, dst, |s, d| unsafe {
                let p = round_to_f16(to_f16(S::load16(s)).map(|x| vmulq_f32(x, sc)));
                f16::store16(d, p.map(|x| vaddq_f32(x, b)));
            });
        }
        None => {
            // SAFETY: `map_steps_uninit` hands out 16 valid elements at each pointer.
            map_steps_uninit(src, dst, |s, d| unsafe {
                f16::store16(d, to_f16(S::load16(s)).map(|x| vmulq_f32(x, sc)));
            });
        }
    }
}

macro_rules! scale_bias_kernel {
    ($(#[$doc:meta])* $name:ident, $s:ty, $d:ty, $body:ident) => {
        $(#[$doc])*
        ///
        /// Writes every element of `dst` when the lengths match, nothing otherwise.
        ///
        /// # Safety
        /// The CPU must support NEON (every aarch64 CPU does).
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(
            src: &[$s],
            scale: $d,
            bias: Option<$d>,
            dst: &mut [MaybeUninit<$d>],
        ) {
            $body(src, scale, bias, dst);
        }
    };
}

scale_bias_kernel!(
    /// `x · scale + bias` (`x · scale` without a bias) per element, rounded after the product
    /// and after the sum and never fused: the decoder's generic dequantize, bit for bit.
    scale_bias_i8_to_f32_neon,
    i8,
    f32,
    scale_bias_loop
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f32_neon`] for `u8`.
    scale_bias_u8_to_f32_neon,
    u8,
    f32,
    scale_bias_loop
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f32_neon`] for `i16`.
    scale_bias_i16_to_f32_neon,
    i16,
    f32,
    scale_bias_loop
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f32_neon`] for `u16`.
    scale_bias_u16_to_f32_neon,
    u16,
    f32,
    scale_bias_loop
);
scale_bias_kernel!(
    /// `x · scale + bias` (`x · scale` without a bias) in `half` arithmetic: `x` rounded to
    /// f16, then each operation rounded to f16, bit for bit.
    scale_bias_i8_to_f16_neon,
    i8,
    f16,
    scale_bias_f16
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f16_neon`] for `u8`.
    scale_bias_u8_to_f16_neon,
    u8,
    f16,
    scale_bias_f16
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f16_neon`] for `i16`.
    scale_bias_i16_to_f16_neon,
    i16,
    f16,
    scale_bias_f16
);
scale_bias_kernel!(
    /// [`scale_bias_i8_to_f16_neon`] for `u16`.
    scale_bias_u16_to_f16_neon,
    u16,
    f16,
    scale_bias_f16
);

/// An integer input of the per-channel dequantize.
pub trait PerChannelIn: Copy + Default {
    /// `(x − zero_point[c]) · scale[c]` of each element `c` of `src`, as the scalar per-channel
    /// DFL computes it, bit for bit; writes nothing unless the four slices have one length.
    #[doc(hidden)]
    fn dequant_per_channel_neon(src: &[Self], zero_point: &[f32], scale: &[f32], dst: &mut [f32]);
}

/// [`PerChannelIn::dequant_per_channel_neon`] for any input [`LoadF32`] reads exactly.
#[inline(always)]
fn dequant_per_channel<S: LoadF32 + Into<f32>>(
    src: &[S],
    zero_point: &[f32],
    scale: &[f32],
    dst: &mut [f32],
) {
    let n = src.len();
    if zero_point.len() != n || scale.len() != n || dst.len() != n {
        return;
    }
    let steps = src
        .chunks_exact(STEP)
        .zip(zero_point.chunks_exact(STEP))
        .zip(scale.chunks_exact(STEP).zip(dst.chunks_exact_mut(STEP)));
    for ((s, z), (c, d)) in steps {
        // SAFETY: every chunk holds 16 elements; NEON is part of the aarch64 baseline.
        unsafe {
            let mut v = S::load16(s.as_ptr());
            for (k, x) in v.iter_mut().enumerate() {
                let (z, c) = (
                    vld1q_f32(z.as_ptr().add(4 * k)),
                    vld1q_f32(c.as_ptr().add(4 * k)),
                );
                *x = vmulq_f32(vsubq_f32(*x, z), c);
            }
            f32::store16(d.as_mut_ptr(), v);
        }
    }
    let done = n / STEP * STEP;
    let rest = src
        .iter()
        .zip(zero_point)
        .zip(scale.iter().zip(dst.iter_mut()))
        .skip(done);
    for ((&s, &z), (&c, d)) in rest {
        *d = (s.into() - z) * c;
    }
}

macro_rules! per_channel_in {
    ($($t:ty),*) => {$(
        impl PerChannelIn for $t {
            #[inline(always)]
            fn dequant_per_channel_neon(
                src: &[Self],
                zero_point: &[f32],
                scale: &[f32],
                dst: &mut [f32],
            ) {
                dequant_per_channel(src, zero_point, scale, dst);
            }
        }
    )*};
}
per_channel_in!(i8, u8, i16, u16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::dequant as dq;
    use crate::quant::{quantize, QuantInt};
    use crate::Quant;
    use half::f16;

    const LENGTHS: [usize; 12] = [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 33, 67];

    /// `values` repeated to `n` elements, starting at `offset` so every value meets every lane.
    fn cycle<T: Copy>(values: &[T], n: usize, offset: usize) -> Vec<T> {
        (0..n)
            .map(|i| values[(i + offset) % values.len()])
            .collect()
    }

    fn quantize_values<T: QuantInt>(inv_scale: f32) -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e30,
            -1e30,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            1e-45,
            2147483520.0,
            -2147483648.0,
            2147483648.0,
        ];
        for k in 0..8 {
            v.push(k as f32 + 0.5);
            v.push(-(k as f32) - 0.5);
        }
        for e in [T::MIN, T::MAX] {
            for d in [-1, 0, 1] {
                let t = (e + d) as f32;
                v.extend([t, t + 0.5, t - 0.5, t / inv_scale, (t + 0.5) / inv_scale]);
            }
        }
        v
    }

    fn check_quantize<T: QuantInt + PartialEq + std::fmt::Debug + Default>(
        neon: unsafe fn(&[f32], f32, i32, &mut [T]),
    ) {
        let mut checked = 0usize;
        for inv_scale in [1.0f32, 3.7, 1e-6, 1e6] {
            let values = quantize_values::<T>(inv_scale);
            for zp in [T::MIN, T::MAX, 0, (T::MIN + T::MAX) / 2] {
                for n in LENGTHS.into_iter().chain([values.len()]) {
                    for offset in [0, 5] {
                        let src = cycle(&values, n, offset);
                        let mut got = vec![T::default(); n];
                        // SAFETY: NEON is part of the aarch64 baseline.
                        unsafe { neon(&src, inv_scale, zp, &mut got) };
                        for (i, (&x, g)) in src.iter().zip(&got).enumerate() {
                            let want = quantize::<T>(x, inv_scale, zp);
                            assert_eq!(*g, want, "x={x:e} inv={inv_scale} zp={zp} n={n} i={i}");
                        }
                        checked += n;
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn quantize_neon_matches_scalar() {
        check_quantize::<i8>(quantize_f32_to_i8_neon);
        check_quantize::<u8>(quantize_f32_to_u8_neon);
        check_quantize::<i16>(quantize_f32_to_i16_neon);
        check_quantize::<u16>(quantize_f32_to_u16_neon);
    }

    #[test]
    fn quantize_neon_length_mismatch_writes_nothing() {
        let src = [1.0f32; 17];
        let mut dst = [7i8; 16];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { quantize_f32_to_i8_neon(&src, 1.0, 0, &mut dst) };
        assert_eq!(dst, [7; 16]);
        let mut dst = [7u16; 18];
        // SAFETY: as above.
        unsafe { quantize_f32_to_u16_neon(&src, 1.0, 0, &mut dst) };
        assert_eq!(dst, [7; 18]);
    }

    /// f32 values on, next to and between every pair of adjacent f16 values, and the NaN classes.
    fn narrow_values() -> Vec<f32> {
        let mut v = Vec::new();
        for bits in 0..=u16::MAX {
            let h = f16::from_bits(bits);
            if h.is_nan() {
                continue;
            }
            let a = h.to_f32();
            v.extend([a, a.next_up(), a.next_down()]);
            if h.is_finite() {
                let b = f16::from_bits(bits.wrapping_add(1)).to_f32();
                if b.is_finite() && b != a {
                    let m = (a + b) * 0.5;
                    v.extend([m, m.next_up(), m.next_down()]);
                }
            }
        }
        v.extend([
            65504.0,
            65519.996,
            65520.0,
            -65520.0,
            1e10,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-45,
            -1e-45,
            2.9802322e-8,
            2.9802326e-8,
            5.9604645e-8,
        ]);
        for bits in [
            0x7fc0_0000u32,
            0xffc0_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7f80_1fff,
            0x7f80_2000,
            0x7fbf_ffff,
            0x7fff_ffff,
            0xffff_e000,
            0x7fa5_5a5a,
        ] {
            v.push(f32::from_bits(bits));
        }
        v
    }

    #[test]
    fn narrow_widen_match_half() {
        let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        for n in LENGTHS.into_iter().chain([all.len()]) {
            for offset in [0, 3] {
                let src = cycle(&all, n, offset);
                let mut got = vec![0.0f32; n];
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { widen_f16_to_f32_neon(&src, &mut got) };
                for (&h, g) in src.iter().zip(&got) {
                    assert_eq!(
                        g.to_bits(),
                        h.to_f32().to_bits(),
                        "f16 {:#06x}",
                        h.to_bits()
                    );
                }
            }
        }
        let values = narrow_values();
        for n in LENGTHS.into_iter().chain([values.len()]) {
            for offset in [0, 7] {
                let src = cycle(&values, n, offset);
                let mut got = vec![f16::ZERO; n];
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { narrow_f32_to_f16_neon(&src, &mut got) };
                for (&x, g) in src.iter().zip(&got) {
                    let want = f16::from_f32(x).to_bits();
                    assert_eq!(g.to_bits(), want, "f32 {:#010x}", x.to_bits());
                }
            }
        }
        let mut short = [f16::ONE; 3];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { narrow_f32_to_f16_neon(&[0.0; 4], &mut short) };
        assert_eq!(short, [f16::ONE; 3]);
        let mut short = [1.0f32; 5];
        // SAFETY: as above.
        unsafe { widen_f16_to_f32_neon(&[f16::ZERO; 4], &mut short) };
        assert_eq!(short, [1.0; 5]);
    }

    fn float_values() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.5,
            3.25,
            1e-3,
            65504.0,
            1e30,
            -1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
            1e-45,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffa5_5a5a),
        ];
        v.extend((0..40).map(|i| (i as f32 - 20.0) * 0.37));
        v
    }

    const QUANTS: [Quant; 5] = [
        Quant::IDENTITY,
        Quant {
            scale: 0.1,
            zero_point: 3,
        },
        Quant {
            scale: 1.0,
            zero_point: -7,
        },
        Quant {
            scale: 2.5e-3,
            zero_point: 0,
        },
        Quant {
            scale: 3.0e7,
            zero_point: i32::MIN,
        },
    ];

    fn bits_eq_f32(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len()
    }

    fn bits_eq_f16(a: &[f16], b: &[f16]) -> bool {
        a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len()
    }

    #[test]
    fn dequant_float_neon_matches_scalar() {
        let f32s = float_values();
        let f16s: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        for q in QUANTS {
            for n in LENGTHS.into_iter().chain([f32s.len()]) {
                let src = cycle(&f32s, n, 1);
                let (mut want, mut got) = (vec![0.0f32; n], vec![1.0f32; n]);
                dq::dequant_f32_to_f32(&src, q, &mut want);
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { dequant_f32_to_f32_neon(&src, q, &mut got) };
                assert!(bits_eq_f32(&got, &want), "f32→f32 {q:?} n={n}");
                let (mut want, mut got) = (vec![f16::ZERO; n], vec![f16::ONE; n]);
                dq::dequant_f32_to_f16(&src, q, &mut want);
                // SAFETY: as above.
                unsafe { dequant_f32_to_f16_neon(&src, q, &mut got) };
                assert!(bits_eq_f16(&got, &want), "f32→f16 {q:?} n={n}");
            }
            for n in LENGTHS.into_iter().chain([f16s.len()]) {
                let src = cycle(&f16s, n, 2);
                let (mut want, mut got) = (vec![0.0f32; n], vec![1.0f32; n]);
                dq::dequant_f16_to_f32(&src, q, &mut want);
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { dequant_f16_to_f32_neon(&src, q, &mut got) };
                assert!(bits_eq_f32(&got, &want), "f16→f32 {q:?} n={n}");
                let (mut want, mut got) = (vec![f16::ZERO; n], vec![f16::ONE; n]);
                dq::dequant_f16_to_f16(&src, q, &mut want);
                // SAFETY: as above.
                unsafe { dequant_f16_to_f16_neon(&src, q, &mut got) };
                assert!(bits_eq_f16(&got, &want), "f16→f16 {q:?} n={n}");
            }
        }
        let mut short = [1.0f32; 3];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { dequant_f32_to_f32_neon(&[0.0; 4], Quant::IDENTITY, &mut short) };
        assert_eq!(short, [1.0; 3]);
        let mut short = [f16::ONE; 3];
        // SAFETY: as above.
        unsafe { dequant_f16_to_f16_neon(&[f16::ZERO; 4], Quant::IDENTITY, &mut short) };
        assert_eq!(short, [f16::ONE; 3]);
    }

    /// A scalar dequantize and its NEON kernel.
    type Pair<T, F> = (fn(&[T], Quant, &mut [F]), unsafe fn(&[T], Quant, &mut [F]));

    fn check_int_dequant<T: QuantInt>(all: &[T], to_f32: Pair<T, f32>, to_f16: Pair<T, f16>) {
        for zp in [T::MIN, T::MAX, 0, 1 << 25] {
            for scale in [0.1f32, 1.0, 3.7e-5, 1e30] {
                let q = Quant::new(scale, zp);
                for n in LENGTHS.into_iter().chain([all.len()]) {
                    let src = cycle(all, n, 3);
                    let (mut want, mut got) = (vec![0.0f32; n], vec![1.0f32; n]);
                    to_f32.0(&src, q, &mut want);
                    // SAFETY: NEON is part of the aarch64 baseline.
                    unsafe { to_f32.1(&src, q, &mut got) };
                    assert!(bits_eq_f32(&got, &want), "{q:?} n={n}");
                    let (mut want, mut got) = (vec![f16::ZERO; n], vec![f16::ONE; n]);
                    to_f16.0(&src, q, &mut want);
                    // SAFETY: as above.
                    unsafe { to_f16.1(&src, q, &mut got) };
                    assert!(bits_eq_f16(&got, &want), "{q:?} n={n}");
                }
            }
        }
    }

    #[test]
    fn dequant_int_exact_neon_matches_scalar() {
        let i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        let u8s: Vec<u8> = (u8::MIN..=u8::MAX).collect();
        let i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let u16s: Vec<u16> = (u16::MIN..=u16::MAX).collect();
        check_int_dequant(
            &i8s,
            (dq::dequant_i8_to_f32, dequant_i8_to_f32_exact_neon),
            (dq::dequant_i8_to_f16, dequant_i8_to_f16_exact_neon),
        );
        check_int_dequant(
            &u8s,
            (dq::dequant_u8_to_f32, dequant_u8_to_f32_exact_neon),
            (dq::dequant_u8_to_f16, dequant_u8_to_f16_exact_neon),
        );
        check_int_dequant(
            &i16s,
            (dq::dequant_i16_to_f32, dequant_i16_to_f32_exact_neon),
            (dq::dequant_i16_to_f16, dequant_i16_to_f16_exact_neon),
        );
        check_int_dequant(
            &u16s,
            (dq::dequant_u16_to_f32, dequant_u16_to_f32_exact_neon),
            (dq::dequant_u16_to_f16, dequant_u16_to_f16_exact_neon),
        );
        let mut short = [1.0f32; 3];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { dequant_u8_to_f32_exact_neon(&[0; 4], Quant::new(1.0, 0), &mut short) };
        assert_eq!(short, [1.0; 3]);
    }

    #[test]
    fn per_channel_dequant_matches_scalar() {
        fn check<S: PerChannelIn + Copy + Into<f32> + std::fmt::Debug>(values: &[S]) {
            for n in [0usize, 1, 15, 16, 17, 64, 68, 256] {
                let src: Vec<S> = values.iter().copied().cycle().take(n).collect();
                let zp: Vec<f32> = (0..n).map(|c| (c % 7) as f32 * 3.0 - 9.0).collect();
                let scale: Vec<f32> = (0..n).map(|c| 0.013 * (1.0 + (c % 5) as f32)).collect();
                let mut got = vec![0f32; n];
                S::dequant_per_channel_neon(&src, &zp, &scale, &mut got);
                for (i, &g) in got.iter().enumerate() {
                    let want = (src[i].into() - zp[i]) * scale[i];
                    assert_eq!(g.to_bits(), want.to_bits(), "n {n} at {i}");
                }
            }
            let mut dst = [7.0f32; 4];
            S::dequant_per_channel_neon(&[values[0]; 4], &[0.0; 3], &[1.0; 4], &mut dst);
            assert_eq!(dst, [7.0; 4]);
        }
        let raw: Vec<i32> = (0..1000).map(|i| (i * 7919 + 17) % 65536).collect();
        check(&raw.iter().map(|&v| v as i8).collect::<Vec<_>>());
        check(&raw.iter().map(|&v| v as u8).collect::<Vec<_>>());
        check(&raw.iter().map(|&v| v as i16).collect::<Vec<_>>());
        check(&raw.iter().map(|&v| v as u16).collect::<Vec<_>>());
    }

    /// `s` as possibly uninitialized elements; the kernels only write initialized values.
    fn uninit<T>(s: &mut [T]) -> &mut [MaybeUninit<T>] {
        // SAFETY: `MaybeUninit<T>` has `T`'s layout, and nothing writes an uninitialized value.
        unsafe { &mut *(s as *mut [T] as *mut [MaybeUninit<T>]) }
    }

    /// Every value of each 8- and 16-bit type through `x · scale + bias` into f32 and f16; the
    /// outputs start as NaN, which no case produces, so an unwritten element fails.
    #[test]
    fn scale_bias_matches_the_generic_formula() {
        type F32Kernel<S> = unsafe fn(&[S], f32, Option<f32>, &mut [MaybeUninit<f32>]);
        type F16Kernel<S> = unsafe fn(&[S], f16, Option<f16>, &mut [MaybeUninit<f16>]);
        fn check<S: Copy + Into<f32> + std::fmt::Debug>(
            all: &[S],
            to_f32: F32Kernel<S>,
            to_f16: &[(&str, F16Kernel<S>)],
        ) {
            for (scale, zp) in [
                (0.004_081_187_3f32, -123i32),
                (1.0, 0),
                (0.1, 7),
                (3.5e-5, 31000),
                (-0.25, 3),
                (-0.5, 0),
            ] {
                let bias = (zp != 0).then(|| -(zp as f32) * scale);
                for n in [all.len(), 0, 1, 15, 17] {
                    let src = &all[all.len() - n..];
                    let mut got = vec![f32::NAN; n];
                    // SAFETY: NEON is part of the aarch64 baseline.
                    unsafe { to_f32(src, scale, bias, uninit(&mut got)) };
                    for (&x, &g) in src.iter().zip(&got) {
                        let want = match bias {
                            Some(b) => x.into() * scale + b,
                            None => x.into() * scale,
                        };
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "f32 {x:?} scale {scale} zp {zp}"
                        );
                    }
                    let (sh, zh) = (f16::from_f32(scale), f16::from_f32(zp as f32));
                    let bh = (zp != 0).then(|| -zh * sh);
                    for (name, kernel) in to_f16 {
                        let mut got = vec![f16::NAN; n];
                        // SAFETY: the caller lists only kernels this CPU supports.
                        unsafe { kernel(src, sh, bh, uninit(&mut got)) };
                        for (&x, &g) in src.iter().zip(&got) {
                            let xh = f16::from_f32(x.into());
                            let want = match bh {
                                Some(b) => xh * sh + b,
                                None => xh * sh,
                            };
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "{name} {x:?} scale {scale} zp {zp}"
                            );
                        }
                    }
                }
            }
        }
        let fp16 = std::arch::is_aarch64_feature_detected!("fp16");
        if !fp16 {
            crate::test_support::report_skip(
                "scale_bias_matches_the_generic_formula FP16 kernels need FEAT_FP16",
            );
        }
        macro_rules! kernels {
            ($emulated:path, $native:path) => {{
                let mut k: Vec<(&str, F16Kernel<_>)> = vec![("neon", $emulated)];
                if fp16 {
                    k.push(("neon_fp16", $native));
                }
                k
            }};
        }
        use crate::prim::neon_fp16 as nf;
        let i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        let u8s: Vec<u8> = (0..=u8::MAX).collect();
        let i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let u16s: Vec<u16> = (0..=u16::MAX).collect();
        check(
            &i8s,
            scale_bias_i8_to_f32_neon,
            &kernels!(
                scale_bias_i8_to_f16_neon,
                nf::scale_bias_i8_to_f16_neon_fp16
            ),
        );
        check(
            &u8s,
            scale_bias_u8_to_f32_neon,
            &kernels!(
                scale_bias_u8_to_f16_neon,
                nf::scale_bias_u8_to_f16_neon_fp16
            ),
        );
        check(
            &i16s,
            scale_bias_i16_to_f32_neon,
            &kernels!(
                scale_bias_i16_to_f16_neon,
                nf::scale_bias_i16_to_f16_neon_fp16
            ),
        );
        check(
            &u16s,
            scale_bias_u16_to_f32_neon,
            &kernels!(
                scale_bias_u16_to_f16_neon,
                nf::scale_bias_u16_to_f16_neon_fp16
            ),
        );
    }

    #[test]
    fn scale_bias_writes_nothing_on_a_length_mismatch() {
        let mut short = [1.0f32; 3];
        let mut short_h = [f16::ONE; 3];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            scale_bias_i16_to_f32_neon(&[0; 4], 1.0, Some(2.0), uninit(&mut short));
            scale_bias_u8_to_f32_neon(&[0; 4], 1.0, None, uninit(&mut short));
            scale_bias_i8_to_f16_neon(&[0; 4], f16::ONE, None, uninit(&mut short_h));
        }
        assert_eq!(short, [1.0; 3]);
        assert_eq!(short_h, [f16::ONE; 3]);
    }
}
