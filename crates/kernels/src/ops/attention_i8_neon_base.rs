// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON attention for CPUs without FEAT_DotProd (Cortex-A53 class), built on widening
//! multiply-accumulates, which issue one per cycle there.
//!
//! The scores use `smlal` by element. Keys are packed as `[8-token group][dim][8 tokens]` and
//! queries as `[4-row block][dim][4 rows]`, so each sign-extended key vector (eight tokens of one
//! dim) meets one query value per row; the accumulators start at the zero-point terms.
//!
//! The outputs use `umlal` on bytes offset to unsigned, `P = p + 128` and `V = v + 128`, with
//! values packed as `[16-dim group][token][16 dims]`. The 16-bit accumulators wrap, but
//! `Σ_j P_j·V_j ≤ 255·Σ_j P_j`, so they stay exact while every row's `Σ_j P_j` is at most 257:
//! tokens run in chunks under that bound, each widened into 32-bit sums. Tokens whose
//! probability is zero in all four rows add nothing and are dropped first.
//!
//! Every accumulator ends as the same exact integer as in [`super::attention_i8_neon`], and the
//! softmax rows are the same, so the outputs match that kernel's byte for byte.

#![cfg(target_arch = "aarch64")]

use core::mem::MaybeUninit;
use core::ops::Range;
use std::arch::aarch64::*; // NOSONAR

use super::attention_i8::{
    AttentionI8Plan, AttentionScratch, Blocks, Buffers, Geometry, ZeroPoints,
    ATTENTION_MAX_HEAD_DIM, PACK_BLOCKS, ROW,
};
use super::attention_i8_neon::{task_range, PackJob, Part, Rows};
use super::matmul_i8::{OutPtr, TILE};
use super::matmul_i8_neon::{requant4, sum_i8};
use crate::pool::Pool;
use crate::prim::transpose::interleave4;
use crate::quant::Rounding;
use crate::view::View;

/// `sxtl`, opaque to the compiler, which would otherwise narrow `smlal` of two sign-extended
/// bytes into `smull` and `saddw`, slower on the A53.
#[inline(always)]
fn sxtl(x: int8x8_t) -> int16x8_t {
    let r: int16x8_t;
    // SAFETY: register-only.
    unsafe {
        core::arch::asm!(
            "sxtl {r:v}.8h, {x:v}.8b",
            r = lateout(vreg) r,
            x = in(vreg) x,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    r
}

/// `sxtl2`, opaque as [`sxtl`].
#[inline(always)]
fn sxtl2(x: int8x16_t) -> int16x8_t {
    let r: int16x8_t;
    // SAFETY: register-only.
    unsafe {
        core::arch::asm!(
            "sxtl2 {r:v}.8h, {x:v}.16b",
            r = lateout(vreg) r,
            x = in(vreg) x,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    r
}

/// The largest per-row `Σ P` of one chunk: `255 · 257` is the largest 16-bit value.
const CHUNK_SUM: u32 = 257;

/// Length of the per-block entry arrays: every token, and the zero entry past them.
const ENTRIES: usize = ROW + 1;

/// Token groups of eight.
#[inline(always)]
fn m8(b: &Blocks) -> usize {
    b.m4.div_ceil(2)
}

/// Offset of head `h`'s keys `[dim d][8 tokens]` of token group `jg`.
#[inline(always)]
fn key8(b: &Blocks, h: usize, jg: usize, d: usize) -> usize {
    ((h * m8(b) + jg) * b.dk4 * 4 + d) * 8
}

/// Offset of head `h`'s 16 values of dim group `eg` for token `j`, of `m` tokens.
#[inline(always)]
fn value16(b: &Blocks, h: usize, eg: usize, j: usize, m: usize) -> usize {
    ((h * b.dv4g + eg) * m + j) * 16
}

/// One task of [`pack`].
///
/// # Safety
/// As [`PackJob::run`]: query task `c` writes query blocks `16c..16c + 16` and their sums, key
/// task `c` key groups `8c..8c + 8` and their sums and biases, value task `c` the values of
/// tokens `64c..64c + 64`, all of head `h`.
#[target_feature(enable = "neon")]
unsafe fn widened_task(job: &PackJob<'_>, part: Part, h: usize, c: usize, rows: &mut Rows) {
    let b = job.b;
    let (dk, dv, m) = (job.dk, job.dv, job.tokens);
    let dkp = b.dk4 * 4;
    // SAFETY: every write is to task `c` of head `h`'s part (above), inside the lengths `pack`
    // checked.
    unsafe {
        match part {
            Part::Queries => {
                for ib in task_range(c, PACK_BLOCKS, b.n4) {
                    job.stage(part, rows, 4, h, 4 * ib);
                    for (r, row) in rows.iter().take(4).enumerate() {
                        if 4 * ib + r < job.queries {
                            *job.query_sums.add(h * b.n4 * 4 + 4 * ib + r) = sum_i8(&row[..dk]);
                        }
                    }
                    let at = job.queries_packed.add(b.query_block(h, ib, 0));
                    queries4(rows, dkp, core::slice::from_raw_parts_mut(at, dkp * 4));
                }
            }
            Part::Keys => {
                for jg in task_range(c, PACK_BLOCKS / 2, m8(&b)) {
                    job.stage(part, rows, 8, h, 8 * jg);
                    let at = job.keys.add(key8(&b, h, jg, 0));
                    keys8(rows, dkp, core::slice::from_raw_parts_mut(at, dkp * 8));
                    for (r, row) in rows.iter().enumerate() {
                        let sum = sum_i8(&row[..dk]);
                        let at = b.token(h, 8 * jg + r);
                        *job.key_sums.add(at) = sum;
                        *job.key_bias.add(at) = job.dkzz - job.zq * sum;
                    }
                }
            }
            Part::Values => {
                // Bytes past `dv` are staged as zero and flip to zero: padding adds nothing.
                let flips: [int8x16_t; ATTENTION_MAX_HEAD_DIM / 16] = core::array::from_fn(|eg| {
                    let lanes: [i8; 16] =
                        core::array::from_fn(|c| if 16 * eg + c < dv { i8::MIN } else { 0 });
                    vld1q_s8(lanes.as_ptr())
                });
                for j in task_range(c, 4 * PACK_BLOCKS, m) {
                    job.stage(part, rows, 1, h, j);
                    for (eg, &flip) in flips.iter().enumerate().take(b.dv4g) {
                        vst1q_s8(
                            job.values.add(value16(&b, h, eg, j, m)),
                            veorq_s8(vld1q_s8(rows[0].as_ptr().add(16 * eg)), flip),
                        );
                    }
                }
            }
        }
    }
}

/// Packs `q` (through `table`), `k` and `v` for [`block`], with the query and key sums and the
/// key biases. Every byte [`block`] reads is written, padding included.
///
/// Each head splits into tasks of [`PACK_BLOCKS`] query blocks, as many tokens of keys, or four
/// times as many value tokens, run on `pool`.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does; `q`, `k`, `v` must be the views `s`
/// was built for, `s` sized by its constructor.
#[target_feature(enable = "neon")]
pub(super) unsafe fn pack(
    s: &mut AttentionScratch,
    q: &View<'_, i8>,
    table: Option<&[u8; 256]>,
    k: &View<'_, i8>,
    v: &View<'_, i8>,
    z: ZeroPoints,
    pool: Option<&dyn Pool>,
) {
    let b = s.blocks;
    let (heads, dk, dv, m) = (s.heads, s.dk, s.dv, s.tokens);
    let dkp = b.dk4 * 4;
    debug_assert!(dk <= ATTENTION_MAX_HEAD_DIM && dv <= ATTENTION_MAX_HEAD_DIM);
    debug_assert!(
        s.queries_packed.len() >= heads * b.n4 * dkp * 4
            && s.query_sums.len() >= heads * b.n4 * 4
            && s.keys.len() >= heads * m8(&b) * dkp * 8
            && s.key_sums.len() >= heads * b.m4g * 16
            && s.key_bias.len() >= heads * b.m4g * 16
            && m8(&b) * 8 <= b.m4g * 16
            && s.values.len() >= heads * b.dv4g * m * 16
    );
    let tasks = [
        b.n4.div_ceil(PACK_BLOCKS),
        m8(&b).div_ceil(PACK_BLOCKS / 2),
        m.div_ceil(4 * PACK_BLOCKS),
    ];
    let job = PackJob::new(s, q, table, k, v, z, tasks);
    // SAFETY: `widened_task` writes only its own task's part, inside the scratch's lengths.
    unsafe { job.run(pool, widened_task) };
}

/// Four rows of `dkp` bytes (a multiple of 4) as `[dim][4 rows]` into `dst`.
#[target_feature(enable = "neon")]
fn queries4(rows: &Rows, dkp: usize, dst: &mut [i8]) {
    assert!(dkp <= ATTENTION_MAX_HEAD_DIM && dst.len() >= dkp * 4);
    let full = dkp / 16;
    for g in 0..full {
        // SAFETY: `16·g + 16 <= dkp <= ATTENTION_MAX_HEAD_DIM` bytes of each row, and `64·g + 64
        // <= 4·dkp` bytes of `dst` (both checked above).
        unsafe {
            let r: [int8x16_t; 4] =
                core::array::from_fn(|t| vld1q_s8(rows[t].as_ptr().add(16 * g)));
            let o = interleave4(r);
            vst1q_s8_x4(
                dst.as_mut_ptr().add(64 * g),
                int8x16x4_t(o[0], o[1], o[2], o[3]),
            );
        }
    }
    for d in 16 * full..dkp {
        for (r, row) in rows.iter().take(4).enumerate() {
            dst[d * 4 + r] = row[d];
        }
    }
}

/// Eight rows of `dkp` bytes (a multiple of 4) as `[dim][8 rows]` into `dst`.
#[target_feature(enable = "neon")]
fn keys8(rows: &Rows, dkp: usize, dst: &mut [i8]) {
    assert!(dkp <= ATTENTION_MAX_HEAD_DIM && dst.len() >= dkp * 8);
    let full = dkp / 16;
    for g in 0..full {
        // SAFETY: `16·g + 16 <= dkp <= ATTENTION_MAX_HEAD_DIM` bytes of each row, and `128·g +
        // 128` bytes of `dst` (both checked above).
        unsafe {
            let r: [int8x16_t; 8] =
                core::array::from_fn(|t| vld1q_s8(rows[t].as_ptr().add(16 * g)));
            let w = |x: int8x16_t| vreinterpretq_s16_s8(x);
            let d = |x: int16x8_t| vreinterpretq_s32_s16(x);
            // Pairs of rows interleaved by byte, then quads by halfword, then all eight by word.
            let p: [[int8x16_t; 2]; 4] = core::array::from_fn(|t| {
                [
                    vzip1q_s8(r[2 * t], r[2 * t + 1]),
                    vzip2q_s8(r[2 * t], r[2 * t + 1]),
                ]
            });
            let quad = |a: [int8x16_t; 2], b: [int8x16_t; 2]| -> [int16x8_t; 4] {
                [
                    vzip1q_s16(w(a[0]), w(b[0])),
                    vzip2q_s16(w(a[0]), w(b[0])),
                    vzip1q_s16(w(a[1]), w(b[1])),
                    vzip2q_s16(w(a[1]), w(b[1])),
                ]
            };
            let (lo, hi) = (quad(p[0], p[1]), quad(p[2], p[3]));
            let o = dst.as_mut_ptr().add(128 * g);
            for u in 0..4 {
                let (x, y) = (d(lo[u]), d(hi[u]));
                vst1q_s8(o.add(32 * u), vreinterpretq_s8_s32(vzip1q_s32(x, y)));
                vst1q_s8(o.add(32 * u + 16), vreinterpretq_s8_s32(vzip2q_s32(x, y)));
            }
        }
    }
    for dd in 16 * full..dkp {
        for (t, row) in rows.iter().enumerate() {
            dst[dd * 8 + t] = row[dd];
        }
    }
}

/// One block of query rows `rows` of head `h`.
///
/// # Safety
/// The CPU must support NEON. `dst` must be the output view's base, the geometry validated
/// against the views, the scratch packed by [`pack`] for this call, and no other task may write
/// these rows.
#[target_feature(enable = "neon")]
pub(super) unsafe fn block(
    p: &AttentionI8Plan,
    g: &Geometry,
    s: &AttentionScratch,
    dst: OutPtr,
    h: usize,
    rows: Range<usize>,
    bufs: &mut Buffers,
) {
    let b = s.blocks;
    let count = rows.len().min(TILE);
    let m = g.m;
    let ib = rows.start / 4;

    // SAFETY: the scratch was packed by `pack` for this call.
    unsafe { scores(p, s, h, ib, bufs) };

    for r in 0..count {
        let (Some(s8), Some(p8)) = (bufs.s8[r].get(..m), bufs.p8[r].get_mut(..m)) else {
            continue;
        };
        p.softmax.row(s8, p8);
        bufs.p_sums[r] = sum_i8(&bufs.p8[r][..m]);
    }

    // Each row's Σ P over all tokens, which the zero-point terms need.
    let total: [i32; TILE] = core::array::from_fn(|r| {
        if r < count {
            bufs.p_sums[r] + 128 * m as i32
        } else {
            0
        }
    });
    let one_chunk = total.iter().all(|&t| t as u32 <= CHUNK_SUM);
    let mut entries = [const { MaybeUninit::<u64>::uninit() }; ENTRIES];
    let mut ends = [const { MaybeUninit::<u16>::uninit() }; ENTRIES];
    let mut groups = Groups {
        count: 0,
        start: [const { MaybeUninit::uninit() }; ROW / 16],
        sums: [const { MaybeUninit::uninit() }; ROW / 16],
    };
    let live = compact(bufs, count, m, &mut entries, &mut groups, !one_chunk);
    let chunks = if one_chunk {
        ends[0].write(live as u16);
        1
    } else {
        // SAFETY: `compact` wrote the entries and the groups with their sums.
        unsafe { split(&entries, live, &groups, &mut ends) }
    };

    let o = Outputs {
        p,
        g,
        values: s.values.as_ptr().wrapping_add(value16(&b, h, 0, 0, m)),
        entries: entries.as_ptr().cast(),
        ends: &ends[..chunks],
        total,
        dst,
        h,
        rows,
    };
    let mut eg = 0;
    while eg < b.dv4g {
        // SAFETY: `pack` laid out `dv4g` groups of `m` tokens; `entries[..=live]` and
        // `ends[..chunks]` were written, the last end equal to `live`.
        unsafe {
            if eg + 1 < b.dv4g {
                o.groups::<2>(eg);
                eg += 2;
            } else {
                o.groups::<1>(eg);
                eg += 1;
            }
        }
    }
}

/// Eight `smlal` by element: `a[r][0..2] += k · q[l + r]` over the key vector's two halves.
macro_rules! smlal8 {
    ($k:literal, $q:literal, $l:literal) => {
        concat!(
            "smlal v16.4s, v",
            $k,
            ".4h, v",
            $q,
            ".h[",
            $l,
            "]\n",
            "smlal2 v17.4s, v",
            $k,
            ".8h, v",
            $q,
            ".h[",
            $l,
            "]\n",
            "smlal v18.4s, v",
            $k,
            ".4h, v",
            $q,
            ".h[",
            $l,
            "+1]\n",
            "smlal2 v19.4s, v",
            $k,
            ".8h, v",
            $q,
            ".h[",
            $l,
            "+1]\n",
            "smlal v20.4s, v",
            $k,
            ".4h, v",
            $q,
            ".h[",
            $l,
            "+2]\n",
            "smlal2 v21.4s, v",
            $k,
            ".8h, v",
            $q,
            ".h[",
            $l,
            "+2]\n",
            "smlal v22.4s, v",
            $k,
            ".4h, v",
            $q,
            ".h[",
            $l,
            "+3]\n",
            "smlal2 v23.4s, v",
            $k,
            ".8h, v",
            $q,
            ".h[",
            $l,
            "+3]\n",
        )
    };
}

/// Four dims of [`dots32`] against query registers `$q0` (dims 0, 1) and `$q1` (dims 2, 3).
/// The raw keys are in v24 (dims 0, 1) and v25 (dims 2, 3), dim 0 already widened in v26; the
/// next step's raw keys load from `$next` (bytes past `{k}`) as these are consumed.
macro_rules! dots4 {
    ($q0:literal, $q1:literal, $next:literal) => {
        concat!(
            "sxtl2 v27.8h, v24.16b\n",
            smlal8!(26, $q0, 0),
            "sxtl v28.8h, v25.8b\n",
            "ldr q24, [{k}, #",
            $next,
            "]\n",
            smlal8!(27, $q0, 4),
            "sxtl2 v29.8h, v25.16b\n",
            "ldr q25, [{k}, #",
            $next,
            "+16]\n",
            smlal8!(28, $q1, 0),
            "sxtl v26.8h, v24.8b\n",
            smlal8!(29, $q1, 4),
        )
    };
}

/// `a[r][t] += Σ_d k[d][4t..4t + 4] · q[r][d]` over 32 dims: the eight key bytes of dim `d` at
/// `keys + 8d`, and row `r`'s dim `d` in lane `4·(d % 2) + r` of `q[d / 2]`.
///
/// The queries stay in v0–v15 (`smlal` by 16-bit element only indexes those) and each step's
/// keys load while the previous step's multiply: about 40 cycles per four dims on a
/// Cortex-A53, against 44 with the queries streamed.
///
/// # Safety
/// `keys` must be readable for 256 bytes, and for 32 more past them.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dots32(
    a: [[int32x4_t; 2]; TILE],
    q: &[int16x8_t; 16],
    keys: *const i8,
) -> [[int32x4_t; 2]; TILE] {
    let [[mut a0, mut a1], [mut a2, mut a3], [mut a4, mut a5], [mut a6, mut a7]] = a;
    // SAFETY: register-only but for the key loads, which the caller guarantees.
    unsafe {
        core::arch::asm!(
            "ldr q24, [{k}]",
            "ldr q25, [{k}, #16]",
            "sxtl v26.8h, v24.8b",
            dots4!(0, 1, 32),
            dots4!(2, 3, 64),
            dots4!(4, 5, 96),
            dots4!(6, 7, 128),
            dots4!(8, 9, 160),
            dots4!(10, 11, 192),
            dots4!(12, 13, 224),
            dots4!(14, 15, 256),
            k = in(reg) keys,
            in("v0") q[0], in("v1") q[1], in("v2") q[2], in("v3") q[3],
            in("v4") q[4], in("v5") q[5], in("v6") q[6], in("v7") q[7],
            in("v8") q[8], in("v9") q[9], in("v10") q[10], in("v11") q[11],
            in("v12") q[12], in("v13") q[13], in("v14") q[14], in("v15") q[15],
            inout("v16") a0, inout("v17") a1, inout("v18") a2, inout("v19") a3,
            inout("v20") a4, inout("v21") a5, inout("v22") a6, inout("v23") a7,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _,
            options(nostack, readonly, preserves_flags),
        );
    }
    [[a0, a1], [a2, a3], [a4, a5], [a6, a7]]
}

/// The 32-dim scores of `groups` groups of eight tokens, requantized as Ruy rounds and stored to
/// four rows `ROW` bytes apart from `out`: [`dots32`] with the bias and requantization in the
/// same loop, the last step loading the next group's first keys.
///
/// `bias` holds each token's key bias, `row_term` each row's query term, and the requantization
/// is `sqdmulh` by `multiplier`, a rounding shift by `shift`, and the zero point `zs`.
///
/// # Safety
/// `keys` must be readable for `256·groups + 32` bytes, `bias` for `8·groups` i32, and each of
/// the four rows at `out` writable for `8·groups` bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn scores32_ruy(
    q: &[int16x8_t; 16],
    keys: *const i8,
    bias: *const i32,
    row_term: [i32; TILE],
    (multiplier, shift, zs): (i32, i32, i32),
    out: *mut i8,
    groups: usize,
) {
    if groups == 0 {
        return;
    }
    // SAFETY: the caller guarantees every address read and written.
    unsafe {
        core::arch::asm!(
            "ldr q24, [{k}]",
            "ldr q25, [{k}, #16]",
            "sxtl v26.8h, v24.8b",
            "dup v30.4s, {shift:w}",
            "dup v31.4s, {zs:w}",
            "2:",
            "ldp q27, q28, [{b}], #32",
            "dup v29.4s, {r0:w}",
            "add v16.4s, v27.4s, v29.4s",
            "add v17.4s, v28.4s, v29.4s",
            "dup v29.4s, {r1:w}",
            "add v18.4s, v27.4s, v29.4s",
            "add v19.4s, v28.4s, v29.4s",
            "dup v29.4s, {r2:w}",
            "add v20.4s, v27.4s, v29.4s",
            "add v21.4s, v28.4s, v29.4s",
            "dup v29.4s, {r3:w}",
            "add v22.4s, v27.4s, v29.4s",
            "add v23.4s, v28.4s, v29.4s",
            dots4!(0, 1, 32),
            dots4!(2, 3, 64),
            dots4!(4, 5, 96),
            dots4!(6, 7, 128),
            dots4!(8, 9, 160),
            dots4!(10, 11, 192),
            dots4!(12, 13, 224),
            dots4!(14, 15, 256),
            "add {k}, {k}, #256",
            "dup v29.4s, {mul:w}",
            "sqdmulh v16.4s, v16.4s, v29.4s",
            "sqdmulh v17.4s, v17.4s, v29.4s",
            "sqdmulh v18.4s, v18.4s, v29.4s",
            "sqdmulh v19.4s, v19.4s, v29.4s",
            "sqdmulh v20.4s, v20.4s, v29.4s",
            "sqdmulh v21.4s, v21.4s, v29.4s",
            "sqdmulh v22.4s, v22.4s, v29.4s",
            "sqdmulh v23.4s, v23.4s, v29.4s",
            "srshl v16.4s, v16.4s, v30.4s",
            "srshl v17.4s, v17.4s, v30.4s",
            "srshl v18.4s, v18.4s, v30.4s",
            "srshl v19.4s, v19.4s, v30.4s",
            "srshl v20.4s, v20.4s, v30.4s",
            "srshl v21.4s, v21.4s, v30.4s",
            "srshl v22.4s, v22.4s, v30.4s",
            "srshl v23.4s, v23.4s, v30.4s",
            "add v16.4s, v16.4s, v31.4s",
            "add v17.4s, v17.4s, v31.4s",
            "add v18.4s, v18.4s, v31.4s",
            "add v19.4s, v19.4s, v31.4s",
            "add v20.4s, v20.4s, v31.4s",
            "add v21.4s, v21.4s, v31.4s",
            "add v22.4s, v22.4s, v31.4s",
            "add v23.4s, v23.4s, v31.4s",
            "sqxtn v27.4h, v16.4s",
            "sqxtn v28.4h, v18.4s",
            "sqxtn v29.4h, v20.4s",
            "sqxtn v16.4h, v22.4s",
            "sqxtn2 v27.8h, v17.4s",
            "sqxtn2 v28.8h, v19.4s",
            "sqxtn2 v29.8h, v21.4s",
            "sqxtn2 v16.8h, v23.4s",
            "sqxtn v27.8b, v27.8h",
            "sqxtn v28.8b, v28.8h",
            "sqxtn v29.8b, v29.8h",
            "sqxtn v16.8b, v16.8h",
            "str d27, [{s}]",
            "str d28, [{s}, #{row1}]",
            "str d29, [{s}, #{row2}]",
            "str d16, [{s}, #{row3}]",
            "add {s}, {s}, #8",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            k = inout(reg) keys => _,
            b = inout(reg) bias => _,
            s = inout(reg) out => _,
            n = inout(reg) groups => _,
            r0 = in(reg) row_term[0],
            r1 = in(reg) row_term[1],
            r2 = in(reg) row_term[2],
            r3 = in(reg) row_term[3],
            mul = in(reg) multiplier,
            shift = in(reg) shift,
            zs = in(reg) zs,
            row1 = const ROW,
            row2 = const 2 * ROW,
            row3 = const 3 * ROW,
            in("v0") q[0], in("v1") q[1], in("v2") q[2], in("v3") q[3],
            in("v4") q[4], in("v5") q[5], in("v6") q[6], in("v7") q[7],
            in("v8") q[8], in("v9") q[9], in("v10") q[10], in("v11") q[11],
            in("v12") q[12], in("v13") q[13], in("v14") q[14], in("v15") q[15],
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
}

/// [`dots32`] for any number of four-dim steps, queries streamed from `queries`.
///
/// # Safety
/// `queries` must be readable for `16·steps` bytes and `keys` for `32·steps`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dots(
    mut a: [[int32x4_t; 2]; TILE],
    queries: *const i8,
    keys: *const i8,
    steps: usize,
) -> [[int32x4_t; 2]; TILE] {
    for dc in 0..steps {
        // SAFETY: 16 query and 32 key bytes of dims `4dc..4dc + 4` (caller).
        let (qb, k01, k23) = unsafe {
            (
                vld1q_s8(queries.add(16 * dc)),
                vld1q_s8(keys.add(32 * dc)),
                vld1q_s8(keys.add(32 * dc + 16)),
            )
        };
        let (q01, q23) = (sxtl(vget_low_s8(qb)), sxtl2(qb));
        let k = [
            sxtl(vget_low_s8(k01)),
            sxtl2(k01),
            sxtl(vget_low_s8(k23)),
            sxtl2(k23),
        ];
        macro_rules! dim {
            ($k:expr, $q:ident, $l:literal) => {
                a[0][0] = vmlal_laneq_s16::<{ $l }>(a[0][0], vget_low_s16($k), $q);
                a[0][1] = vmlal_high_laneq_s16::<{ $l }>(a[0][1], $k, $q);
                a[1][0] = vmlal_laneq_s16::<{ $l + 1 }>(a[1][0], vget_low_s16($k), $q);
                a[1][1] = vmlal_high_laneq_s16::<{ $l + 1 }>(a[1][1], $k, $q);
                a[2][0] = vmlal_laneq_s16::<{ $l + 2 }>(a[2][0], vget_low_s16($k), $q);
                a[2][1] = vmlal_high_laneq_s16::<{ $l + 2 }>(a[2][1], $k, $q);
                a[3][0] = vmlal_laneq_s16::<{ $l + 3 }>(a[3][0], vget_low_s16($k), $q);
                a[3][1] = vmlal_high_laneq_s16::<{ $l + 3 }>(a[3][1], $k, $q);
            };
        }
        dim!(k[0], q01, 0);
        dim!(k[1], q01, 4);
        dim!(k[2], q23, 0);
        dim!(k[3], q23, 4);
    }
    a
}

/// The scores of query block `ib` of head `h` into `bufs.s8`, eight tokens per tile.
///
/// # Safety
/// The scratch must have been packed by [`pack`] for this call.
#[target_feature(enable = "neon")]
unsafe fn scores(
    p: &AttentionI8Plan,
    s: &AttentionScratch,
    h: usize,
    ib: usize,
    bufs: &mut Buffers,
) {
    let b = s.blocks;
    let qsum: [i32; TILE] = core::array::from_fn(|r| {
        s.query_sums
            .get(h * b.n4 * 4 + ib * 4 + r)
            .copied()
            .unwrap_or(0)
    });
    let row_term: [int32x4_t; TILE] = core::array::from_fn(|r| vdupq_n_s32(-p.zk * qsum[r]));
    let zs = vdupq_n_s32(p.zs);
    // SAFETY: block (h, ib, 0) and the next dk4 blocks lie in the packed queries.
    let queries = unsafe { s.queries_packed.as_ptr().add(b.query_block(h, ib, 0)) };
    // Row `r`'s dim `d` in lane `4·(d % 2) + r` of register `d / 2`, for `dots32`.
    let resident = b.dk4 == 8;
    let q: [int16x8_t; 16] = core::array::from_fn(|i| {
        if resident {
            // SAFETY: the 128 query bytes of the block.
            let qb = unsafe { vld1q_s8(queries.add(16 * (i / 2))) };
            if i % 2 == 0 {
                sxtl(vget_low_s8(qb))
            } else {
                sxtl2(qb)
            }
        } else {
            vdupq_n_s16(0)
        }
    });
    let m8 = m8(&b);
    if resident && p.rounding == Rounding::Ruy {
        let r = p.scores_requant;
        // SAFETY: the head's `m8` key groups (and 32 bytes past them) and biases were packed;
        // each row of `s8` holds `m8 · 8 <= ROW` bytes.
        unsafe {
            scores32_ruy(
                &q,
                s.keys.as_ptr().add(key8(&b, h, 0, 0)),
                s.key_bias.as_ptr().add(b.token(h, 0)),
                core::array::from_fn(|r| -p.zk * qsum[r]),
                (r.multiplier, r.shift, p.zs),
                bufs.s8.as_mut_ptr().cast::<i8>(),
                m8,
            );
        }
        return;
    }
    for jg in 0..m8 {
        // SAFETY: the `dk4 · 32` key bytes of group `jg` and its eight biases were packed.
        let (keys, bias) = unsafe {
            (
                s.keys.as_ptr().add(key8(&b, h, jg, 0)),
                s.key_bias.as_ptr().add(b.token(h, 8 * jg)),
            )
        };
        // SAFETY: eight biases.
        let (b0, b1) = unsafe { (vld1q_s32(bias), vld1q_s32(bias.add(4))) };
        let init: [[int32x4_t; 2]; TILE] =
            core::array::from_fn(|r| [vaddq_s32(b0, row_term[r]), vaddq_s32(b1, row_term[r])]);
        // SAFETY: group `jg`'s keys, and the 32 bytes past them, lie in the padded keys.
        let a = unsafe {
            if resident {
                dots32(init, &q, keys)
            } else {
                dots(init, queries, keys, b.dk4)
            }
        };
        for (r, a) in a.iter().enumerate() {
            let lo = vaddq_s32(requant4(a[0], p.scores_requant, p.rounding), zs);
            let hi = vaddq_s32(requant4(a[1], p.scores_requant, p.rounding), zs);
            let n = vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
            // SAFETY: `8·jg + 8 <= m8 · 8 <= ROW`.
            unsafe { vst1_s8(bufs.s8[r].as_mut_ptr().add(8 * jg), n) };
        }
    }
}

/// Per-row `Σ P` of a run of tokens: four 16-bit lanes, row `r` in lane `r`.
///
/// No lane passes `CHUNK_SUM + 16·255`, so lanes never carry into each other, and a lane passes
/// [`CHUNK_SUM`] exactly when adding [`OVER_BIAS`] sets its top bit.
type Sums = u64;
const OVER_BIAS: u64 = (0x8000 - (CHUNK_SUM as u64 + 1)) * 0x0001_0001_0001_0001;
const OVER_TOP: u64 = 0x8000_8000_8000_8000;

#[inline(always)]
fn over(s: Sums) -> bool {
    (s + OVER_BIAS) & OVER_TOP != 0
}

/// The [`Sums`] of one entry.
#[inline(always)]
fn entry_sums(e: u64) -> Sums {
    let w = e & 0xffff_ffff;
    let x = (w | w << 16) & 0x0000_ffff_0000_ffff;
    (x | x << 8) & 0x00ff_00ff_00ff_00ff
}

/// The groups of 16 tokens that kept entries: where their entries start, and their [`Sums`]
/// when `compact` was asked for them.
struct Groups {
    count: usize,
    start: [MaybeUninit<u16>; ROW / 16],
    sums: [MaybeUninit<Sums>; ROW / 16],
}

/// The tokens where some row's probability is nonzero, as `P | j << 32`: `P` holds row `r`'s
/// `p + 128` in byte `r` (zero past `count`), `j` is the token. Returns how many there are, and
/// writes one zero entry past them, which the output loop may read ahead.
#[target_feature(enable = "neon")]
fn compact(
    bufs: &mut Buffers,
    count: usize,
    m: usize,
    entries: &mut [MaybeUninit<u64>; ENTRIES],
    groups: &mut Groups,
    want_sums: bool,
) -> usize {
    assert!(m.next_multiple_of(16) <= ROW && count <= TILE);
    let flip = vdupq_n_u8(0x80);
    let mut listed = [const { MaybeUninit::<u16>::uninit() }; ROW / 16];
    // The words of every group of 16 tokens with a nonzero probability, transposed with NEON
    // into `bufs.p`, which this kernel has no other use for; they are read back only after all
    // are stored, as the A53 forwards NEON stores to general-register loads slowly.
    let wp = bufs.p.as_mut_ptr().cast::<u8>();
    let mut ng = 0;
    for g16 in 0..m.div_ceil(16) {
        let first = 16 * g16;
        let r: [uint8x16_t; TILE] = core::array::from_fn(|r| {
            if r < count {
                // SAFETY: `first + 16 <= ROW` bytes of the row.
                let x = unsafe { vld1q_s8(bufs.p8[r].as_ptr().add(first)) };
                veorq_u8(vreinterpretq_u8_s8(x), flip)
            } else {
                vdupq_n_u8(0)
            }
        });
        if vmaxvq_u8(vorrq_u8(vorrq_u8(r[0], r[1]), vorrq_u8(r[2], r[3]))) == 0 {
            continue;
        }
        let w = |x: uint8x16_t| vreinterpretq_u16_u8(x);
        let (a, c) = (vzip1q_u8(r[0], r[1]), vzip2q_u8(r[0], r[1]));
        let (bb, d) = (vzip1q_u8(r[2], r[3]), vzip2q_u8(r[2], r[3]));
        let b = |x: uint16x8_t| vreinterpretq_u8_u16(x);
        // SAFETY: `4·(first + 16) <= 4·ROW <= bufs.p.len()` bytes.
        unsafe {
            let o = wp.add(4 * first);
            vst1q_u8(o, b(vzip1q_u16(w(a), w(bb))));
            vst1q_u8(o.add(16), b(vzip2q_u16(w(a), w(bb))));
            vst1q_u8(o.add(32), b(vzip1q_u16(w(c), w(d))));
            vst1q_u8(o.add(48), b(vzip2q_u16(w(c), w(d))));
        }
        if want_sums {
            // Tokens past `m` hold leftovers.
            let lanes: [u8; 16] = core::array::from_fn(|t| if first + t < m { 0xff } else { 0 });
            // SAFETY: 16 readable bytes.
            let keep = unsafe { vld1q_u8(lanes.as_ptr()) };
            let h: [uint16x8_t; TILE] = core::array::from_fn(|i| vpaddlq_u8(vandq_u8(r[i], keep)));
            let x = vpaddq_u16(vpaddq_u16(h[0], h[1]), vpaddq_u16(h[2], h[3]));
            let x = vpaddq_u16(x, x);
            groups.sums[ng].write(vgetq_lane_u64::<0>(vreinterpretq_u64_u16(vpaddq_u16(x, x))));
        }
        listed[ng].write(g16 as u16);
        ng += 1;
    }
    let ep = entries.as_mut_ptr().cast::<u64>();
    let mut live = 0;
    // Branch-free: every token's entry is written, and only the nonzero ones are kept.
    let keep = |j: usize, live: &mut usize| {
        // SAFETY: word `j` of a listed group was written; `live <= j < ENTRIES`.
        unsafe {
            let x = wp.add(4 * j).cast::<u32>().read_unaligned();
            ep.add(*live).write(u64::from(x) | (j as u64) << 32);
            *live += usize::from(x != 0);
        }
    };
    for (g, start) in listed[..ng].iter().zip(&mut groups.start) {
        // SAFETY: the first `ng` groups were listed.
        let first = 16 * usize::from(unsafe { g.assume_init() });
        start.write(live as u16);
        if first + 16 <= m {
            for t in 0..16 {
                keep(first + t, &mut live);
            }
        } else {
            for j in first..m {
                keep(j, &mut live);
            }
        }
    }
    groups.count = ng;
    entries[live].write(0);
    live
}

/// Splits the `live` compacted tokens into chunks whose per-row `Σ P` is at most
/// [`CHUNK_SUM`]: whole groups while they fit, token by token within a group that does not;
/// writes each chunk's end into `ends` and returns how many.
///
/// # Safety
/// `entries[..live]` and the groups, with their sums, must be written by `compact`.
unsafe fn split(
    entries: &[MaybeUninit<u64>; ENTRIES],
    live: usize,
    groups: &Groups,
    ends: &mut [MaybeUninit<u16>; ENTRIES],
) -> usize {
    let mut sums: Sums = 0;
    let mut n = 0;
    for g in 0..groups.count {
        // SAFETY: written by `compact` (caller).
        let (start, gs) = unsafe {
            (
                usize::from(groups.start[g].assume_init()),
                groups.sums[g].assume_init(),
            )
        };
        let end = if g + 1 < groups.count {
            // SAFETY: as above.
            usize::from(unsafe { groups.start[g + 1].assume_init() })
        } else {
            live
        };
        if !over(sums + gs) {
            sums += gs;
        } else if !over(gs) {
            ends[n].write(start as u16);
            n += 1;
            sums = gs;
        } else {
            for (c, e) in entries.iter().enumerate().take(end).skip(start) {
                // SAFETY: written by `compact` (caller).
                let x = entry_sums(unsafe { e.assume_init() });
                if over(sums + x) {
                    ends[n].write(c as u16);
                    n += 1;
                    sums = x;
                } else {
                    sums += x;
                }
            }
        }
    }
    ends[n].write(live as u16);
    n + 1
}

/// `umlal` of one token's four rows against its 32 value bytes, for [`tokens32`]: the values in
/// `$v0`, `$v1`, the probabilities broadcast in `$p0..$p3`. Meanwhile the next entry loads and its
/// values and broadcasts go to `$n0`, `$n1`, `$q0..$q3`.
macro_rules! token32 {
    ($v0:literal, $v1:literal, $p0:literal, $p1:literal, $p2:literal, $p3:literal,
     $n0:literal, $n1:literal, $q0:literal, $q1:literal, $q2:literal, $q3:literal) => {
        concat!(
            "umlal v16.8h, v",
            $v0,
            ".8b, v",
            $p0,
            ".8b\n",
            "ldr x10, [{e}], #8\n",
            "umlal2 v17.8h, v",
            $v0,
            ".16b, v",
            $p0,
            ".16b\n",
            "umlal v18.8h, v",
            $v1,
            ".8b, v",
            $p0,
            ".8b\n",
            "umlal2 v19.8h, v",
            $v1,
            ".16b, v",
            $p0,
            ".16b\n",
            "umlal v20.8h, v",
            $v0,
            ".8b, v",
            $p1,
            ".8b\n",
            "lsr x11, x10, #32\n",
            "umlal2 v21.8h, v",
            $v0,
            ".16b, v",
            $p1,
            ".16b\n",
            "umlal v22.8h, v",
            $v1,
            ".8b, v",
            $p1,
            ".8b\n",
            "add x12, {s0}, x11, lsl #4\n",
            "umlal2 v23.8h, v",
            $v1,
            ".16b, v",
            $p1,
            ".16b\n",
            "add x13, {s1}, x11, lsl #4\n",
            "ldr q",
            $n0,
            ", [x12]\n",
            "umlal v24.8h, v",
            $v0,
            ".8b, v",
            $p2,
            ".8b\n",
            "umlal2 v25.8h, v",
            $v0,
            ".16b, v",
            $p2,
            ".16b\n",
            "ldr q",
            $n1,
            ", [x13]\n",
            "umlal v26.8h, v",
            $v1,
            ".8b, v",
            $p2,
            ".8b\n",
            "lsr w14, w10, #8\n",
            "umlal2 v27.8h, v",
            $v1,
            ".16b, v",
            $p2,
            ".16b\n",
            "dup v",
            $q0,
            ".16b, w10\n",
            "lsr w15, w10, #16\n",
            "umlal v28.8h, v",
            $v0,
            ".8b, v",
            $p3,
            ".8b\n",
            "umlal2 v29.8h, v",
            $v0,
            ".16b, v",
            $p3,
            ".16b\n",
            "lsr w9, w10, #24\n",
            "dup v",
            $q1,
            ".16b, w14\n",
            "umlal v30.8h, v",
            $v1,
            ".8b, v",
            $p3,
            ".8b\n",
            "umlal2 v31.8h, v",
            $v1,
            ".16b, v",
            $p3,
            ".16b\n",
            "dup v",
            $q2,
            ".16b, w15\n",
            "dup v",
            $q3,
            ".16b, w9\n",
        )
    };
}

/// `acc[r][x] += Σ_c P_c[r] · V_c` over entries `entries[..n]`, with `V_c` the 32 value bytes of
/// the entry's token, 16 in each stream; `acc[r][2s]` holds stream `s`'s low eight dims, and
/// `acc[r][2s + 1]` its high eight.
///
/// Each token's values and probability broadcasts load while the previous token multiplies,
/// alternating between two register sets: 24 cycles per token on a Cortex-A53.
///
/// # Safety
/// `entries[..=n]` must be readable, with every token's 16 bytes readable in both streams.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tokens32(
    acc: &mut [[uint16x8_t; 4]; TILE],
    entries: *const u64,
    n: usize,
    s0: *const u8,
    s1: *const u8,
) {
    if n == 0 {
        return;
    }
    let [[a0, a1, a2, a3], [a4, a5, a6, a7], [a8, a9, a10, a11], [a12, a13, a14, a15]] = acc;
    // SAFETY: the caller guarantees every address the loop reads, the entry past the last
    // included.
    unsafe {
        core::arch::asm!(
            "ldr x10, [{e}], #8",
            "lsr x11, x10, #32",
            "add x12, {s0}, x11, lsl #4",
            "add x13, {s1}, x11, lsl #4",
            "ldr q0, [x12]",
            "ldr q1, [x13]",
            "lsr w14, w10, #8",
            "lsr w15, w10, #16",
            "lsr w9, w10, #24",
            "dup v4.16b, w10",
            "dup v5.16b, w14",
            "dup v6.16b, w15",
            "dup v7.16b, w9",
            "cbz {pairs}, 3f",
            "2:",
            token32!(0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 10, 11),
            token32!(2, 3, 8, 9, 10, 11, 0, 1, 4, 5, 6, 7),
            "subs {pairs}, {pairs}, #1",
            "b.ne 2b",
            "3:",
            "cbz {odd}, 4f",
            token32!(0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 10, 11),
            "4:",
            e = inout(reg) entries => _,
            pairs = inout(reg) n / 2 => _,
            odd = in(reg) n % 2,
            s0 = in(reg) s0,
            s1 = in(reg) s1,
            out("x9") _, out("x10") _, out("x11") _, out("x12") _,
            out("x13") _, out("x14") _, out("x15") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            inout("v16") *a0, inout("v17") *a1, inout("v18") *a2, inout("v19") *a3,
            inout("v20") *a4, inout("v21") *a5, inout("v22") *a6, inout("v23") *a7,
            inout("v24") *a8, inout("v25") *a9, inout("v26") *a10, inout("v27") *a11,
            inout("v28") *a12, inout("v29") *a13, inout("v30") *a14, inout("v31") *a15,
            options(nostack, readonly),
        );
    }
}

/// The output phase of one block.
struct Outputs<'a> {
    p: &'a AttentionI8Plan,
    g: &'a Geometry,
    /// Head `h`'s packed values.
    values: *const i8,
    entries: *const u64,
    ends: &'a [MaybeUninit<u16>],
    /// Each row's `Σ_j P_j`.
    total: [i32; TILE],
    dst: OutPtr,
    h: usize,
    rows: Range<usize>,
}

impl Outputs<'_> {
    /// Value dims `16·eg..16·(eg + W)` of every row.
    ///
    /// # Safety
    /// Groups `eg..eg + W` must exist; the entries and chunk ends must be written.
    #[target_feature(enable = "neon")]
    unsafe fn groups<const W: usize>(&self, eg: usize) {
        let m = self.g.m;
        // `[row][4-dim vector]` of `Σ P·V`.
        let mut sums = [[vdupq_n_u32(0); 8]; TILE];
        // SAFETY: groups `eg..eg + W` of the head's values exist (caller).
        let streams: [*const u8; W] =
            core::array::from_fn(|x| unsafe { self.values.add((eg + x) * m * 16).cast::<u8>() });
        let mut c0 = 0usize;
        for end in self.ends {
            // SAFETY: written (caller).
            let c1 = usize::from(unsafe { end.assume_init() });
            let mut acc = [[vdupq_n_u16(0); 4]; TILE];
            if W == 2 {
                // SAFETY: entries `c0..=c1` are written; their tokens lie in both streams.
                unsafe {
                    tokens32(
                        &mut acc,
                        self.entries.add(c0),
                        c1 - c0,
                        streams[0],
                        streams[W - 1],
                    )
                };
            } else {
                for c in c0..c1 {
                    // SAFETY: `c < live`; token `j < m` has a 16-byte row in the stream.
                    unsafe {
                        let e = *self.entries.add(c);
                        let w = vreinterpret_u8_u32(vdup_n_u32(e as u32));
                        let v = vld1q_u8(streams[0].add(16 * (e >> 32) as usize));
                        macro_rules! row {
                            ($r:literal) => {
                                let pr = vdupq_lane_u8::<$r>(w);
                                acc[$r][0] = vmlal_u8(acc[$r][0], vget_low_u8(v), vget_low_u8(pr));
                                acc[$r][1] = vmlal_high_u8(acc[$r][1], v, pr);
                            };
                        }
                        row!(0);
                        row!(1);
                        row!(2);
                        row!(3);
                    }
                }
            }
            for (s, a) in sums.iter_mut().zip(&acc) {
                for x in 0..2 * W {
                    s[2 * x] = vaddw_u16(s[2 * x], vget_low_u16(a[x]));
                    s[2 * x + 1] = vaddw_high_u16(s[2 * x + 1], a[x]);
                }
            }
            c0 = c1;
        }
        let p = self.p;
        let zo = vdupq_n_s32(p.zo);
        let [oh, os1, os2] = self.g.out_strides;
        for (r, i) in self.rows.clone().enumerate().take(TILE) {
            // Σ P·(v − zv) = Σ P·V − (128 + zv)·Σ P.
            let bias = vdupq_n_s32(-(128 + p.zv) * self.total[r]);
            for x in 0..W {
                let q: [int32x4_t; 4] = core::array::from_fn(|u| {
                    let acc = vaddq_s32(vreinterpretq_s32_u32(sums[r][4 * x + u]), bias);
                    vaddq_s32(requant4(acc, p.out_requant, p.rounding), zo)
                });
                let lo = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
                let hi = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
                let v = vqmovn_high_s16(vqmovn_s16(lo), hi);
                let e0 = 16 * (eg + x);
                let lanes = self.g.dv.saturating_sub(e0).min(16);
                let base = self.h * oh + i * os1 + e0 * os2;
                if os2 == 1 && lanes == 16 {
                    // SAFETY: 16 contiguous output bytes of row `i`, this task's alone.
                    unsafe { vst1q_s8(self.dst.0.add(base), v) };
                } else {
                    let mut tmp = [0i8; 16];
                    // SAFETY: 16 writable bytes.
                    unsafe { vst1q_s8(tmp.as_mut_ptr(), v) };
                    for (c, &y) in tmp.iter().enumerate().take(lanes) {
                        // SAFETY: (h, i, e0 + c) lies in the validated output view and only this
                        // task writes query row `i` of head `h`.
                        unsafe { self.dst.0.add(base + c * os2).write(y) };
                    }
                }
            }
        }
    }
}
