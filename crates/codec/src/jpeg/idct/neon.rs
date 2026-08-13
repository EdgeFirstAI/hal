// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised 8×8 IDCT on 16-bit coefficients (libjpeg-turbo model).
//!
//! The entropy decoder hands us **quantised** `i16` coefficients; this kernel
//! dequantises 8 lanes at a time (`vmulq_s16`, wrapping — same contract as
//! turbo's `jsimd_idct_islow_neon`), runs the Loeffler butterfly with
//! `vmull_s16` 16×16→32 multiplies, and keeps the inter-pass workspace in
//! `int16x8` registers (no memory round-trip). Sparse shortcuts skip the
//! arithmetic for the two dominant block shapes of quantised natural images:
//! all-zero bottom rows (4–7) in pass 1 and all-zero right columns (4–7) in
//! pass 2. A further per-half DC-only path (rows 1–7 of a 4×8 half all zero)
//! broadcasts `dequant(row0) << PASS1_BITS`, matching turbo islow special
//! case 2 — the remaining in-order gap after whole-block DC-only is taken in
//! the MCU loop.

use std::arch::aarch64::*;

/// Fixed-point precision (must match scalar).
const PASS1_BITS: i32 = 2;
const CONST_BITS: i32 = 13;

/// Loeffler constants (must match scalar). All fit in i16 for `vmull_s16`.
const FIX_0_298: i16 = 2446;
const FIX_0_390: i16 = 3196;
const FIX_0_541: i16 = 4433;
const FIX_0_765: i16 = 6270;
const FIX_0_899: i16 = 7373;
const FIX_1_175: i16 = 9633;
const FIX_1_501: i16 = 12299;
const FIX_1_847: i16 = 15137;
const FIX_1_961: i16 = 16069;
const FIX_2_053: i16 = 16819;
const FIX_2_562: i16 = 20995;
const FIX_3_072: i16 = 25172;
/// FIX_0_541 + FIX_0_765 — single multiply in the sparse even part.
const FIX_1_306: i16 = FIX_0_541 + FIX_0_765;

/// Butterfly constants packed for lane-indexed multiplies.
///
/// Written as `vmull_s16(x, vdup_n_s16(K))`, each constant is rebuilt with a
/// `mov`/`dup` pair at every use site: the butterfly is inlined four times
/// (two halves × two passes), so the kernel carried 64 `dup`s per block —
/// 17% of its instructions. The `_laneq_` forms below name the constant as a
/// lane of a register instead, so each multiply is a bare `smull`/`smlal`.
const K0: [i16; 8] = [
    FIX_0_541, FIX_0_765, FIX_1_847, FIX_1_175, FIX_0_298, FIX_2_053, FIX_3_072, FIX_1_501,
];
const K1: [i16; 8] = [
    -FIX_0_899, -FIX_2_562, FIX_1_961, FIX_0_390, FIX_1_306, 0, 0, 0,
];

/// Full Loeffler butterfly on 4 lanes of 8 inputs. Returns the 8 outputs in
/// natural order **before** descaling (32-bit, no rounding bias applied).
#[inline(always)]
#[allow(clippy::too_many_arguments)] // eight DCT coefficients is the Loeffler contract
unsafe fn butterfly8(
    s0: int16x4_t,
    s1: int16x4_t,
    s2: int16x4_t,
    s3: int16x4_t,
    s4: int16x4_t,
    s5: int16x4_t,
    s6: int16x4_t,
    s7: int16x4_t,
    k0: int16x8_t,
    k1: int16x8_t,
) -> [int32x4_t; 8] {
    // Even part.
    let tmp0 = vshll_n_s16::<{ CONST_BITS }>(s0);
    let tmp2 = vshll_n_s16::<{ CONST_BITS }>(s4);
    let tmp10 = vaddq_s32(tmp0, tmp2);
    let tmp11 = vsubq_s32(tmp0, tmp2);

    let z1 = vadd_s16(s2, s6);
    let tmp13 = vmlal_laneq_s16::<1>(vmull_laneq_s16::<0>(z1, k0), s2, k0);
    let tmp12 = vmlsl_laneq_s16::<2>(vmull_laneq_s16::<0>(z1, k0), s6, k0);

    let t0 = vaddq_s32(tmp10, tmp13);
    let t3 = vsubq_s32(tmp10, tmp13);
    let t1 = vaddq_s32(tmp11, tmp12);
    let t2 = vsubq_s32(tmp11, tmp12);

    // Odd part.
    let z1 = vadd_s16(s7, s1);
    let z2 = vadd_s16(s5, s3);
    let z3 = vadd_s16(s7, s3);
    let z4 = vadd_s16(s5, s1);
    let z5 = vmlal_laneq_s16::<3>(vmull_laneq_s16::<3>(z3, k0), z4, k0);

    let p7 = vmull_laneq_s16::<4>(s7, k0);
    let p5 = vmull_laneq_s16::<5>(s5, k0);
    let p3 = vmull_laneq_s16::<6>(s3, k0);
    let p1 = vmull_laneq_s16::<7>(s1, k0);

    let z1n = vmull_laneq_s16::<0>(z1, k1);
    let z2n = vmull_laneq_s16::<1>(z2, k1);
    let z3n = vsubq_s32(z5, vmull_laneq_s16::<2>(z3, k1));
    let z4n = vsubq_s32(z5, vmull_laneq_s16::<3>(z4, k1));

    let o0 = vaddq_s32(vaddq_s32(p7, z1n), z3n);
    let o1 = vaddq_s32(vaddq_s32(p5, z2n), z4n);
    let o2 = vaddq_s32(vaddq_s32(p3, z2n), z3n);
    let o3 = vaddq_s32(vaddq_s32(p1, z1n), z4n);

    [
        vaddq_s32(t0, o3),
        vaddq_s32(t1, o2),
        vaddq_s32(t2, o1),
        vaddq_s32(t3, o0),
        vsubq_s32(t3, o0),
        vsubq_s32(t2, o1),
        vsubq_s32(t1, o2),
        vsubq_s32(t0, o3),
    ]
}

/// Sparse Loeffler butterfly: inputs 4–7 are known zero (bottom rows in
/// pass 1, right columns in pass 2). Algebraically identical to [`butterfly8`]
/// with `s4 = s5 = s6 = s7 = 0`; ten multiplies instead of fourteen.
#[inline(always)]
unsafe fn butterfly8_sparse(
    s0: int16x4_t,
    s1: int16x4_t,
    s2: int16x4_t,
    s3: int16x4_t,
    k0: int16x8_t,
    k1: int16x8_t,
) -> [int32x4_t; 8] {
    // Even part: tmp2 = 0 so tmp10 = tmp11 = tmp0; z1 collapses to s2.
    let tmp0 = vshll_n_s16::<{ CONST_BITS }>(s0);
    let tmp13 = vmull_laneq_s16::<4>(s2, k1);
    let tmp12 = vmull_laneq_s16::<0>(s2, k0);

    let t0 = vaddq_s32(tmp0, tmp13);
    let t3 = vsubq_s32(tmp0, tmp13);
    let t1 = vaddq_s32(tmp0, tmp12);
    let t2 = vsubq_s32(tmp0, tmp12);

    // Odd part: z1 = s1, z2 = z3 = s3, z4 = s1; p7 = p5 = 0.
    let z5 = vmlal_laneq_s16::<3>(vmull_laneq_s16::<3>(s3, k0), s1, k0);
    let p3 = vmull_laneq_s16::<6>(s3, k0);
    let p1 = vmull_laneq_s16::<7>(s1, k0);

    let z1n = vmull_laneq_s16::<0>(s1, k1);
    let z2n = vmull_laneq_s16::<1>(s3, k1);
    let z3n = vsubq_s32(z5, vmull_laneq_s16::<2>(s3, k1));
    let z4n = vsubq_s32(z5, vmull_laneq_s16::<3>(s1, k1));

    let o0 = vaddq_s32(z1n, z3n);
    let o1 = vaddq_s32(z2n, z4n);
    let o2 = vaddq_s32(vaddq_s32(p3, z2n), z3n);
    let o3 = vaddq_s32(vaddq_s32(p1, z1n), z4n);

    [
        vaddq_s32(t0, o3),
        vaddq_s32(t1, o2),
        vaddq_s32(t2, o1),
        vaddq_s32(t3, o0),
        vsubq_s32(t3, o0),
        vsubq_s32(t2, o1),
        vsubq_s32(t1, o2),
        vsubq_s32(t0, o3),
    ]
}

/// Descale one pass-1 output: round-to-nearest shift by CONST_BITS-PASS1_BITS,
/// **saturating** into `i16`.
///
/// `sqrshrn` rather than plain `rshrn`, at identical cost, so that a coefficient
/// stream extreme enough to overflow the 16-bit workspace lands on the same
/// pixels here as under the x86 kernels' `packssdw` and the scalar reference —
/// see [`super::IdctFn`]. Real streams never come near the saturation point.
#[inline(always)]
unsafe fn descale_p1(x: int32x4_t) -> int16x4_t {
    vqrshrn_n_s32::<{ CONST_BITS - PASS1_BITS }>(x)
}

/// NEON 8×8 IDCT on quantised i16 coefficients (natural order) with the
/// component quant table (natural order).
#[inline(always)]
pub fn idct_8x8_neon(coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8], stride: usize) {
    unsafe { idct_8x8_neon_inner(coeffs, quant, output, stride) }
}

/// NEON 8×8 IDCT with an entropy-derived sparsity hint.
///
/// `last_k` is the highest zigzag index the block decoder wrote (an upper
/// bound on occupancy — see [`huffman::BlockInfo`]). Because the zigzag order
/// fills the top-left corner first, small `last_k` proves whole regions zero
/// without touching the coefficient memory:
///
/// - `last_k ≤ 1`: AC only at (0,1) — left 4×8 half is DC-broadcast shape,
///   right half entirely zero.
/// - `last_k ≤ 9`: the prefix stays in rows 0–3 / cols 0–2 — sparse pass-1
///   butterfly on the left half, right half zero.
/// - `last_k ≤ 13`: adds naturals 32/25/18/11 (row 4 possible, cols still
///   ≤ 3) — full-width pass-1 with rows 5–7 as zero registers.
/// - otherwise: the value-based detection kernel (which still finds shapes
///   the prefix bound cannot prove, e.g. all-zero bottom rows behind a high
///   `last_k`).
///
/// The proven-sparse tiers use only 64-bit `vld1_s16` loads for the rows that
/// participate — the Cortex-A53's load path is 64 bits wide, and the skipped
/// rows/halves never touch the cache. Bit-exact with [`idct_8x8_neon`]: every
/// dropped term is exactly zero, and the shared tail is the same code.
///
/// [`huffman::BlockInfo`]: crate::jpeg::huffman::BlockInfo
#[inline(always)]
pub fn idct_8x8_neon_k(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    last_k: u8,
    output: &mut [u8],
    stride: usize,
) {
    unsafe { idct_8x8_neon_k_inner(coeffs, quant, last_k, output, stride) }
}

#[target_feature(enable = "neon")]
unsafe fn idct_8x8_neon_k_inner(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    last_k: u8,
    output: &mut [u8],
    stride: usize,
) {
    if last_k > 13 {
        return idct_8x8_neon_inner(coeffs, quant, output, stride);
    }

    let k0 = vld1q_s16(std::hint::black_box(K0.as_ptr()));
    let k1 = vld1q_s16(std::hint::black_box(K1.as_ptr()));
    let zero4 = vdup_n_s16(0);

    // Dequantise one 4-lane left-half row (cols 0–3): 64-bit loads only.
    let dqd = |i: usize| -> int16x4_t {
        vmul_s16(
            vld1_s16(coeffs.as_ptr().add(i * 8)),
            vreinterpret_s16_u16(vld1_u16(quant.as_ptr().add(i * 8))),
        )
    };

    let r0 = dqd(0);
    let wl: [int16x4_t; 8] = if last_k <= 1 {
        // Column IDCT of [v, 0, …, 0] is saturate(v << PASS1_BITS) per row.
        let dc = vqmovn_s32(vshll_n_s16::<{ PASS1_BITS }>(r0));
        [dc; 8]
    } else {
        let x = if last_k <= 9 {
            butterfly8_sparse(r0, dqd(1), dqd(2), dqd(3), k0, k1)
        } else {
            butterfly8(
                r0,
                dqd(1),
                dqd(2),
                dqd(3),
                dqd(4),
                zero4,
                zero4,
                zero4,
                k0,
                k1,
            )
        };
        [
            descale_p1(x[0]),
            descale_p1(x[1]),
            descale_p1(x[2]),
            descale_p1(x[3]),
            descale_p1(x[4]),
            descale_p1(x[5]),
            descale_p1(x[6]),
            descale_p1(x[7]),
        ]
    };

    finish_pass2_store(wl, [zero4; 8], true, k0, k1, output, stride);
}

#[target_feature(enable = "neon")]
unsafe fn idct_8x8_neon_inner(
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    output: &mut [u8],
    stride: usize,
) {
    // Load quantised rows and detect the two dominant sparsity shapes before
    // dequantising (zero quantised ⇒ zero dequantised).
    let c0 = vld1q_s16(coeffs.as_ptr());
    let c1 = vld1q_s16(coeffs.as_ptr().add(8));
    let c2 = vld1q_s16(coeffs.as_ptr().add(16));
    let c3 = vld1q_s16(coeffs.as_ptr().add(24));
    let c4 = vld1q_s16(coeffs.as_ptr().add(32));
    let c5 = vld1q_s16(coeffs.as_ptr().add(40));
    let c6 = vld1q_s16(coeffs.as_ptr().add(48));
    let c7 = vld1q_s16(coeffs.as_ptr().add(56));

    let bottom_or = vorrq_s16(vorrq_s16(c4, c5), vorrq_s16(c6, c7));
    let top_ac = vorrq_s16(vorrq_s16(c1, c2), c3);
    let ac_or = vorrq_s16(top_ac, bottom_or);
    let all_or = vorrq_s16(c0, ac_or);
    let bottom_u64 = vreinterpretq_u64_s16(bottom_or);
    let bottom_zero = (vgetq_lane_u64::<0>(bottom_u64) | vgetq_lane_u64::<1>(bottom_u64)) == 0;
    // Columns 4–7 across all rows sit in the high 64 bits of each row vector.
    let right_zero = vgetq_lane_u64::<1>(vreinterpretq_u64_s16(all_or)) == 0;
    // Rows 1–7 of a 4×8 half all zero → that half is DC-only (turbo islow).
    // Whole-block DC-only is already taken in the MCU loop via `has_ac`; this
    // catches the common "AC only in the top-left 4×4" / "row 0 only" shapes
    // that still enter the full kernel.
    let ac_u64 = vreinterpretq_u64_s16(ac_or);
    let left_ac_zero = vgetq_lane_u64::<0>(ac_u64) == 0;
    let right_ac_zero = vgetq_lane_u64::<1>(ac_u64) == 0;

    // Dequantise (16-bit wrapping multiply — turbo's contract; valid baseline
    // streams never overflow, corrupt ones only garble their own pixels).
    let dq = |c: int16x8_t, i: usize| -> int16x8_t {
        vmulq_s16(
            c,
            vreinterpretq_s16_u16(vld1q_u16(quant.as_ptr().add(i * 8))),
        )
    };
    let r0 = dq(c0, 0);
    let zero8 = vdupq_n_s16(0);
    // Row-0-only (both 4×8 halves DC in pass 1): skip AC dequant entirely.
    let skip_ac_dequant = left_ac_zero && (right_zero || right_ac_zero);
    let (r1, r2, r3, r4, r5, r6, r7) = if skip_ac_dequant {
        (zero8, zero8, zero8, zero8, zero8, zero8, zero8)
    } else if bottom_zero {
        (dq(c1, 1), dq(c2, 2), dq(c3, 3), zero8, zero8, zero8, zero8)
    } else {
        (
            dq(c1, 1),
            dq(c2, 2),
            dq(c3, 3),
            dq(c4, 4),
            dq(c5, 5),
            dq(c6, 6),
            dq(c7, 7),
        )
    };

    // Butterfly constants, loaded once per block rather than rebuilt at every
    // multiply (see [`K0`]). The barrier is load-bearing: with the addresses
    // visible LLVM folds each lane back to its literal value and re-emits the
    // `dup`s the lane form was meant to remove, undoing the whole change. Hiding
    // the pointers forces one real load and keeps both vectors live across the
    // four butterflies. Costs two more spill slots and saves ~100 instructions
    // per block; a no-op for correctness if a future LLVM stops honouring it.
    let k0 = vld1q_s16(std::hint::black_box(K0.as_ptr()));
    let k1 = vld1q_s16(std::hint::black_box(K1.as_ptr()));

    // Pass 1 (columns): lanes are columns, inputs indexed by row. Left half
    // (cols 0–3) from the low halves, right half from the highs.
    let zero4 = vdup_n_s16(0);
    let pass1_half = |h0: int16x4_t,
                      h1: int16x4_t,
                      h2: int16x4_t,
                      h3: int16x4_t,
                      h4: int16x4_t,
                      h5: int16x4_t,
                      h6: int16x4_t,
                      h7: int16x4_t,
                      sparse: bool|
     -> [int16x4_t; 8] {
        let x = if sparse {
            butterfly8_sparse(h0, h1, h2, h3, k0, k1)
        } else {
            butterfly8(h0, h1, h2, h3, h4, h5, h6, h7, k0, k1)
        };
        [
            descale_p1(x[0]),
            descale_p1(x[1]),
            descale_p1(x[2]),
            descale_p1(x[3]),
            descale_p1(x[4]),
            descale_p1(x[5]),
            descale_p1(x[6]),
            descale_p1(x[7]),
        ]
    };

    let wl = if left_ac_zero {
        // Column IDCT of [dc, 0,0,0,0,0,0,0] is saturate(dc << PASS1_BITS)
        // in every row — same as descale_p1 of the 32-bit even part.
        let dc = vqmovn_s32(vshll_n_s16::<{ PASS1_BITS }>(vget_low_s16(r0)));
        [dc; 8]
    } else {
        pass1_half(
            vget_low_s16(r0),
            vget_low_s16(r1),
            vget_low_s16(r2),
            vget_low_s16(r3),
            vget_low_s16(r4),
            vget_low_s16(r5),
            vget_low_s16(r6),
            vget_low_s16(r7),
            bottom_zero,
        )
    };
    let wr = if right_zero {
        [zero4; 8]
    } else if right_ac_zero {
        let dc = vqmovn_s32(vshll_n_s16::<{ PASS1_BITS }>(vget_high_s16(r0)));
        [dc; 8]
    } else {
        pass1_half(
            vget_high_s16(r0),
            vget_high_s16(r1),
            vget_high_s16(r2),
            vget_high_s16(r3),
            vget_high_s16(r4),
            vget_high_s16(r5),
            vget_high_s16(r6),
            vget_high_s16(r7),
            bottom_zero,
        )
    };

    finish_pass2_store(wl, wr, right_zero, k0, k1, output, stride);
}

/// Shared tail of the 8×8 kernels: combine the pass-1 half results, transpose,
/// run pass 2 (sparse when the right workspace half is known zero), descale,
/// and store. `#[inline(always)]` so each caller keeps its registers.
#[inline(always)]
unsafe fn finish_pass2_store(
    wl: [int16x4_t; 8],
    wr: [int16x4_t; 8],
    right_zero: bool,
    k0: int16x8_t,
    k1: int16x8_t,
    output: &mut [u8],
    stride: usize,
) {
    // Workspace rows (lanes = columns) → transpose to columns (lanes = rows).
    let w0 = vcombine_s16(wl[0], wr[0]);
    let w1 = vcombine_s16(wl[1], wr[1]);
    let w2 = vcombine_s16(wl[2], wr[2]);
    let w3 = vcombine_s16(wl[3], wr[3]);
    let w4 = vcombine_s16(wl[4], wr[4]);
    let w5 = vcombine_s16(wl[5], wr[5]);
    let w6 = vcombine_s16(wl[6], wr[6]);
    let w7 = vcombine_s16(wl[7], wr[7]);

    let (t0, t1, t2, t3, t4, t5, t6, t7) = transpose_8x8_s16(w0, w1, w2, w3, w4, w5, w6, w7);

    // Pass 2 (rows): lanes are rows. If the right workspace half was zero the
    // transposed columns 4–7 are zero — sparse butterfly, no check needed.
    let pass2_half = |g0: int16x4_t,
                      g1: int16x4_t,
                      g2: int16x4_t,
                      g3: int16x4_t,
                      g4: int16x4_t,
                      g5: int16x4_t,
                      g6: int16x4_t,
                      g7: int16x4_t|
     -> [int32x4_t; 8] {
        if right_zero {
            butterfly8_sparse(g0, g1, g2, g3, k0, k1)
        } else {
            butterfly8(g0, g1, g2, g3, g4, g5, g6, g7, k0, k1)
        }
    };

    let lo = pass2_half(
        vget_low_s16(t0),
        vget_low_s16(t1),
        vget_low_s16(t2),
        vget_low_s16(t3),
        vget_low_s16(t4),
        vget_low_s16(t5),
        vget_low_s16(t6),
        vget_low_s16(t7),
    );
    let hi = pass2_half(
        vget_high_s16(t0),
        vget_high_s16(t1),
        vget_high_s16(t2),
        vget_high_s16(t3),
        vget_high_s16(t4),
        vget_high_s16(t5),
        vget_high_s16(t6),
        vget_high_s16(t7),
    );

    // Final descale: ((x + 2^15) >> 16 + 2) >> 2 + 128, saturated to u8 —
    // equal to the scalar ((x + 2^17 + 128·2^18) >> 18).clamp(0,255) within ±1.
    let off128 = vdupq_n_s16(128);
    let narrow = |a: int32x4_t, b: int32x4_t| -> uint8x8_t {
        let v = vcombine_s16(vrshrn_n_s32::<16>(a), vrshrn_n_s32::<16>(b));
        vqmovun_s16(vaddq_s16(vrshrq_n_s16::<2>(v), off128))
    };
    // u_j lanes = rows 0..7 of output column j.
    let u0 = narrow(lo[0], hi[0]);
    let u1 = narrow(lo[1], hi[1]);
    let u2 = narrow(lo[2], hi[2]);
    let u3 = narrow(lo[3], hi[3]);
    let u4 = narrow(lo[4], hi[4]);
    let u5 = narrow(lo[5], hi[5]);
    let u6 = narrow(lo[6], hi[6]);
    let u7 = narrow(lo[7], hi[7]);

    // Transpose 8×8 bytes (columns → rows) and store row-major.
    let (y0, y1, y2, y3, y4, y5, y6, y7) = transpose_8x8_u8(u0, u1, u2, u3, u4, u5, u6, u7);
    let out = output.as_mut_ptr();
    vst1_u8(out, y0);
    vst1_u8(out.add(stride), y1);
    vst1_u8(out.add(2 * stride), y2);
    vst1_u8(out.add(3 * stride), y3);
    vst1_u8(out.add(4 * stride), y4);
    vst1_u8(out.add(5 * stride), y5);
    vst1_u8(out.add(6 * stride), y6);
    vst1_u8(out.add(7 * stride), y7);
}

/// 8×8 transpose of int16x8 vectors (three trn stages: 16 → 32 → 64 bit).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn transpose_8x8_s16(
    w0: int16x8_t,
    w1: int16x8_t,
    w2: int16x8_t,
    w3: int16x8_t,
    w4: int16x8_t,
    w5: int16x8_t,
    w6: int16x8_t,
    w7: int16x8_t,
) -> (
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
) {
    let a0 = vtrn1q_s16(w0, w1);
    let a1 = vtrn2q_s16(w0, w1);
    let a2 = vtrn1q_s16(w2, w3);
    let a3 = vtrn2q_s16(w2, w3);
    let a4 = vtrn1q_s16(w4, w5);
    let a5 = vtrn2q_s16(w4, w5);
    let a6 = vtrn1q_s16(w6, w7);
    let a7 = vtrn2q_s16(w6, w7);

    let t32 = |x: int16x8_t| vreinterpretq_s32_s16(x);
    let f32 = |x: int32x4_t| vreinterpretq_s16_s32(x);
    let b0 = f32(vtrn1q_s32(t32(a0), t32(a2)));
    let b2 = f32(vtrn2q_s32(t32(a0), t32(a2)));
    let b1 = f32(vtrn1q_s32(t32(a1), t32(a3)));
    let b3 = f32(vtrn2q_s32(t32(a1), t32(a3)));
    let b4 = f32(vtrn1q_s32(t32(a4), t32(a6)));
    let b6 = f32(vtrn2q_s32(t32(a4), t32(a6)));
    let b5 = f32(vtrn1q_s32(t32(a5), t32(a7)));
    let b7 = f32(vtrn2q_s32(t32(a5), t32(a7)));

    let t64 = |x: int16x8_t| vreinterpretq_s64_s16(x);
    let f64 = |x: int64x2_t| vreinterpretq_s16_s64(x);
    (
        f64(vtrn1q_s64(t64(b0), t64(b4))),
        f64(vtrn1q_s64(t64(b1), t64(b5))),
        f64(vtrn1q_s64(t64(b2), t64(b6))),
        f64(vtrn1q_s64(t64(b3), t64(b7))),
        f64(vtrn2q_s64(t64(b0), t64(b4))),
        f64(vtrn2q_s64(t64(b1), t64(b5))),
        f64(vtrn2q_s64(t64(b2), t64(b6))),
        f64(vtrn2q_s64(t64(b3), t64(b7))),
    )
}

/// 8×8 transpose of uint8x8 vectors (three trn stages: 8 → 16 → 32 bit).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn transpose_8x8_u8(
    u0: uint8x8_t,
    u1: uint8x8_t,
    u2: uint8x8_t,
    u3: uint8x8_t,
    u4: uint8x8_t,
    u5: uint8x8_t,
    u6: uint8x8_t,
    u7: uint8x8_t,
) -> (
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
    uint8x8_t,
) {
    let a0 = vtrn1_u8(u0, u1);
    let a1 = vtrn2_u8(u0, u1);
    let a2 = vtrn1_u8(u2, u3);
    let a3 = vtrn2_u8(u2, u3);
    let a4 = vtrn1_u8(u4, u5);
    let a5 = vtrn2_u8(u4, u5);
    let a6 = vtrn1_u8(u6, u7);
    let a7 = vtrn2_u8(u6, u7);

    let t16 = |x: uint8x8_t| vreinterpret_u16_u8(x);
    let f16 = |x: uint16x4_t| vreinterpret_u8_u16(x);
    let b0 = f16(vtrn1_u16(t16(a0), t16(a2)));
    let b2 = f16(vtrn2_u16(t16(a0), t16(a2)));
    let b1 = f16(vtrn1_u16(t16(a1), t16(a3)));
    let b3 = f16(vtrn2_u16(t16(a1), t16(a3)));
    let b4 = f16(vtrn1_u16(t16(a4), t16(a6)));
    let b6 = f16(vtrn2_u16(t16(a4), t16(a6)));
    let b5 = f16(vtrn1_u16(t16(a5), t16(a7)));
    let b7 = f16(vtrn2_u16(t16(a5), t16(a7)));

    let t32 = |x: uint8x8_t| vreinterpret_u32_u8(x);
    let f32 = |x: uint32x2_t| vreinterpret_u8_u32(x);
    (
        f32(vtrn1_u32(t32(b0), t32(b4))),
        f32(vtrn1_u32(t32(b1), t32(b5))),
        f32(vtrn1_u32(t32(b2), t32(b6))),
        f32(vtrn1_u32(t32(b3), t32(b7))),
        f32(vtrn2_u32(t32(b0), t32(b4))),
        f32(vtrn2_u32(t32(b1), t32(b5))),
        f32(vtrn2_u32(t32(b2), t32(b6))),
        f32(vtrn2_u32(t32(b3), t32(b7))),
    )
}

/// NEON DC-only IDCT: fill 8×8 block with a single value.
#[inline(always)]
pub fn idct_dc_only_neon(dc_value: i32, output: &mut [u8], stride: usize) {
    let range_shift = CONST_BITS + PASS1_BITS + 3;
    let round = 1 << (range_shift - 1);
    let bias = round + (128 << range_shift);
    // Wrapping: in-contract values cannot overflow here, but the argument is an
    // i32 and a caller that hands over a wider one must get the same defined
    // result the block kernels give, not a debug-build panic.
    let scaled = dc_value.wrapping_shl((CONST_BITS + PASS1_BITS) as u32);
    let val = (scaled.wrapping_add(bias) >> range_shift).clamp(0, 255) as u8;

    // Use NEON to fill 8 bytes at once
    unsafe {
        let fill = vdup_n_u8(val);
        for row in 0..8 {
            vst1_u8(output.as_mut_ptr().add(row * stride), fill);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::scalar::{idct_8x8_scalar, idct_dc_only_scalar};
    use super::{idct_8x8_neon, idct_8x8_neon_k, idct_dc_only_neon};
    use crate::jpeg::types::ZIGZAG;

    const UNIT_QUANT: [u16; 64] = [1u16; 64];

    /// Deterministic synthetic IDCT coefficient block: DC + a few AC terms.
    /// Values chosen so that no output pixel saturates (all outputs stay in ~30..220).
    fn make_test_coeffs() -> [i16; 64] {
        let mut c = [0i16; 64];
        // DC coefficient
        c[0] = 256;
        // A handful of AC terms at positions that give non-trivial output
        c[1] = 32;
        c[8] = 16;
        c[2] = -24;
        c[16] = 20;
        c[9] = -12;
        c[3] = 8;
        c[24] = -8;
        c
    }

    fn assert_parity(coeffs: &[i16; 64], quant: &[u16; 64], label: &str) {
        let mut scalar_out = [0u8; 64];
        let mut simd_out = [0u8; 64];
        idct_8x8_scalar(coeffs, quant, &mut scalar_out, 8);
        idct_8x8_neon(coeffs, quant, &mut simd_out, 8);
        for i in 0..64 {
            let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
            assert!(
                diff <= 1,
                "{label}: parity mismatch at index {i}: scalar={}, simd={}, diff={}",
                scalar_out[i],
                simd_out[i],
                diff
            );
        }
    }

    #[test]
    fn idct_8x8_parity() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        // Default coefficients live in rows 0–3 / cols 0–3: exercises both
        // sparse shortcuts (bottom rows zero + right columns zero).
        assert_parity(&make_test_coeffs(), &UNIT_QUANT, "sparse");
    }

    /// Rows 1–7 zero: per-half DC-only shortcut (turbo islow case 2).
    #[test]
    fn idct_8x8_parity_row0_only() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 256;
        coeffs[1] = 40;
        coeffs[2] = -16;
        coeffs[3] = 8;
        assert_parity(&coeffs, &UNIT_QUANT, "row0-left");
        coeffs[4] = 22;
        coeffs[7] = -9;
        assert_parity(&coeffs, &UNIT_QUANT, "row0-both-halves");
    }

    /// Left 4×8 is DC-only; right 4×8 has AC — the two halves take different
    /// shortcuts.
    #[test]
    fn idct_8x8_parity_left_dc_right_ac() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 200;
        coeffs[5] = 18;
        coeffs[12] = -14;
        coeffs[23] = 9;
        assert_parity(&coeffs, &UNIT_QUANT, "left-dc-right-ac");
    }

    /// Coefficients in columns 4–7 defeat the right-half skip; parity must
    /// hold on the full-width path too.
    #[test]
    fn idct_8x8_parity_right_half_coeffs() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut coeffs = make_test_coeffs();
        coeffs[5] = 18; // row 0, col 5
        coeffs[12] = -14; // row 1, col 4
        coeffs[23] = 9; // row 2, col 7
        assert_parity(&coeffs, &UNIT_QUANT, "right-half");
    }

    /// Coefficients in rows 4–7 defeat the bottom-rows skip (full pass-1
    /// butterfly), while columns 4–7 stay zero (sparse pass 2).
    #[test]
    fn idct_8x8_parity_bottom_row_coeffs() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut coeffs = make_test_coeffs();
        coeffs[32] = 15; // row 4, col 0
        coeffs[41] = -11; // row 5, col 1
        coeffs[58] = 7; // row 7, col 2
        assert_parity(&coeffs, &UNIT_QUANT, "bottom-rows");
    }

    /// Dense block: every sparsity shortcut defeated.
    #[test]
    fn idct_8x8_parity_dense() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut coeffs = make_test_coeffs();
        coeffs[5] = 18; // row 0, col 5
        coeffs[36] = -9; // row 4, col 4
        coeffs[63] = 5; // row 7, col 7
        assert_parity(&coeffs, &UNIT_QUANT, "dense");
    }

    /// Real-shape quant table must be applied identically to scalar.
    #[test]
    fn idct_8x8_parity_with_quant() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut quant = [0u16; 64];
        for (i, q) in quant.iter_mut().enumerate() {
            *q = 2 + (i as u16 % 13);
        }
        let mut coeffs = [0i16; 64];
        coeffs[0] = 64;
        coeffs[1] = 12;
        coeffs[8] = -9;
        coeffs[18] = 5;
        coeffs[35] = -3; // row 4, col 3
        assert_parity(&coeffs, &quant, "quant");
    }

    #[test]
    fn idct_8x8_parity_zero_block() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        assert_parity(&[0i16; 64], &UNIT_QUANT, "zero-block");
    }

    #[test]
    fn idct_8x8_parity_strided() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        let coeffs = make_test_coeffs();
        let stride = 16usize;
        let mut scalar_out = vec![0u8; stride * 8];
        let mut simd_out = vec![0u8; stride * 8];

        idct_8x8_scalar(&coeffs, &UNIT_QUANT, &mut scalar_out, stride);
        idct_8x8_neon(&coeffs, &UNIT_QUANT, &mut simd_out, stride);

        for row in 0..8 {
            for col in 0..8 {
                let i = row * stride + col;
                let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "strided parity mismatch at row={row} col={col}: scalar={}, simd={}, diff={}",
                    scalar_out[i],
                    simd_out[i],
                    diff
                );
            }
        }
    }

    /// The entropy-hinted kernel must be bit-identical to the value-detection
    /// kernel for every tier. For each `last_k`, build a block whose occupied
    /// zigzag prefix ends exactly at `last_k` (worst case for the tier bound)
    /// and compare against the scalar reference.
    #[test]
    fn idct_8x8_k_tiers_match_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }
        let mut quant = [0u16; 64];
        for (i, q) in quant.iter_mut().enumerate() {
            *q = 1 + (i as u16 % 7);
        }
        for last_k in 1..64u8 {
            let mut coeffs = [0i16; 64];
            coeffs[0] = 300;
            // Fill the whole prefix with non-zeros: every position the bound
            // permits is occupied, so a tier that loads too little fails loudly.
            for k in 1..=last_k {
                let natural = ZIGZAG[k as usize] as usize;
                coeffs[natural] = if k % 2 == 0 { 17 } else { -13 };
            }
            let mut scalar_out = [0u8; 64];
            let mut simd_out = [0u8; 64];
            idct_8x8_scalar(&coeffs, &quant, &mut scalar_out, 8);
            idct_8x8_neon_k(&coeffs, &quant, last_k, &mut simd_out, 8);
            for i in 0..64 {
                let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "last_k={last_k}: mismatch at {i}: scalar={} simd={}",
                    scalar_out[i],
                    simd_out[i]
                );
            }
            // A sparser block under the same bound (only the last position
            // occupied) must also match — the hint is an upper bound.
            let mut coeffs2 = [0i16; 64];
            coeffs2[0] = -180;
            coeffs2[ZIGZAG[last_k as usize] as usize] = 21;
            let mut scalar2 = [0u8; 64];
            let mut simd2 = [0u8; 64];
            idct_8x8_scalar(&coeffs2, &quant, &mut scalar2, 8);
            idct_8x8_neon_k(&coeffs2, &quant, last_k, &mut simd2, 8);
            for i in 0..64 {
                let diff = (scalar2[i] as i32 - simd2[i] as i32).abs();
                assert!(diff <= 1, "sparse last_k={last_k}: mismatch at {i}");
            }
        }
    }

    #[test]
    fn idct_dc_only_parity() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("SIMD feature not available, skipping");
            return;
        }

        for dc in [0i32, 8, 64, 128, -64, 255, -255] {
            let mut scalar_out = [0u8; 64];
            let mut simd_out = [0u8; 64];

            idct_dc_only_scalar(dc, &mut scalar_out, 8);
            idct_dc_only_neon(dc, &mut simd_out, 8);

            for i in 0..64 {
                let diff = (scalar_out[i] as i32 - simd_out[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "dc_only parity mismatch dc={dc} at index {i}: scalar={}, simd={}, diff={}",
                    scalar_out[i],
                    simd_out[i],
                    diff
                );
            }
        }
    }
}
