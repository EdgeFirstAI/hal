// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int8 softmax kernels.
//!
//! A row's outputs depend only on the table index `255 - max + x`. Long rows sum their
//! exponentials from the f32 table in four interleaved partial sums, which keeps the sum within
//! an ulp or so of TFLite's sequential one, then build a per-row byte table of quantized
//! outputs and map the row through it with `tbl`. Only the top of the table needs building:
//! where the probabilities have decayed to the zero point the lookup takes that value by
//! default, which for attention scores usually leaves a single 64-entry slice. Short rows with
//! a wide value range gather and quantize each element instead, summing in four partial sums. Rows of exactly sixteen run sixteen at a time, one row per
//! lane after an in-register transpose, with every exponential from the split tables.
//!
//! Every path is within ±1 LSB of the scalar kernel's sequential f32 sum, and unbiased.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::rows::RowGroup;
#[cfg(test)]
use super::softmax_i8::slice_max;
use super::softmax_i8::{quantize_prob, table_index, table_offset};
use crate::prim::transpose::transpose16;

/// Longest row that may take the per-element path.
const DIRECT_MAX: usize = 64;

/// NEON softmax of one row; processes `min(src.len(), dst.len())` elements.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_row_i8_neon(
    table: &[f32; 256],
    slice_max: &[f32; 4],
    out_scale: f32,
    zp: i32,
    src: &[i8],
    dst: &mut [i8],
) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    if n == 0 {
        return;
    }
    let (min, max) = min_max(src);
    let off = table_offset(max);
    let range = usize::from(max.abs_diff(min));
    // Table entries to build, rounded up to whole vectors and ending at index 255.
    let entries = (range + 16) & !15;
    if n <= DIRECT_MAX && entries >= n {
        direct(table, out_scale, zp, off, src, dst);
        return;
    }
    // SAFETY: `max` is the row's maximum (`min_max` above).
    let inv = 1.0 / (unsafe { gather_sum(table, max, src) } * out_scale);
    // 64-entry slices, counted from the top, that the row's indices reach. Below the top `s`
    // slices every index quantizes to `zero` when their largest entry does, since scaling by
    // `inv` and rounding are monotonic; those lanes take the default instead of a lookup.
    let zero = quantize_prob(0.0, zp);
    let reach = 4 - (255 - range) / 64;
    let slices = (1..reach)
        .find(|&s| {
            let below = slice_max.iter().take(4 - s).fold(0.0f32, |m, &t| m.max(t));
            quantize_prob(below * inv, zp) == zero
        })
        .unwrap_or(reach);
    let mut otab = [zero; 256];
    build_out_table(
        table,
        inv,
        zp,
        (256 - entries).max(256 - 64 * slices),
        &mut otab,
    );
    match slices {
        1 => lookup::<1>(&otab, zero, off, src, dst),
        2 => lookup::<2>(&otab, zero, off, src, dst),
        3 => lookup::<3>(&otab, zero, off, src, dst),
        _ => lookup::<4>(&otab, zero, off, src, dst),
    }
}

#[target_feature(enable = "neon")]
fn min_max(src: &[i8]) -> (i8, i8) {
    let (pairs, rest) = src.as_chunks::<32>();
    let (chunks, tail) = rest.as_chunks::<16>();
    let (mut lo, mut hi) = (i8::MAX, i8::MIN);
    if !(pairs.is_empty() && chunks.is_empty()) {
        // Two accumulator pairs, so each load has the other pair's work to hide behind.
        let mut vlo = [vdupq_n_s8(i8::MAX); 2];
        let mut vhi = [vdupq_n_s8(i8::MIN); 2];
        for c in pairs {
            // SAFETY: `c` is 32 readable bytes.
            let v = unsafe { [vld1q_s8(c.as_ptr()), vld1q_s8(c.as_ptr().add(16))] };
            vlo = [vminq_s8(vlo[0], v[0]), vminq_s8(vlo[1], v[1])];
            vhi = [vmaxq_s8(vhi[0], v[0]), vmaxq_s8(vhi[1], v[1])];
        }
        for c in chunks {
            // SAFETY: `c` is 16 readable bytes.
            let v = unsafe { vld1q_s8(c.as_ptr()) };
            vlo[0] = vminq_s8(vlo[0], v);
            vhi[0] = vmaxq_s8(vhi[0], v);
        }
        lo = vminvq_s8(vminq_s8(vlo[0], vlo[1]));
        hi = vmaxvq_s8(vmaxq_s8(vhi[0], vhi[1]));
    }
    for &x in tail {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo, hi)
}

/// The row sum of `table[255 - max + x]`, in four interleaved f32 accumulators (element `j`
/// into accumulator `j % 4`); `max` must be the row's maximum.
///
/// Whole 16-element chunks run in a software-pipelined loop for the in-order Cortex-A55: the
/// bytes go straight to general registers, each sign-extended byte is the element's offset from
/// `table + 255 - max`, and every `fadd` trails its load by four elements, the last four of a
/// chunk being added at the top of the next.
///
/// # Safety
/// `max` must be at least every element of `src`: the loop reads `table[255 − max + x]` unchecked.
#[target_feature(enable = "neon")]
unsafe fn gather_sum(table: &[f32; 256], max: i8, src: &[i8]) -> f32 {
    let off = table_offset(max);
    let (chunks, rest) = src.as_chunks::<16>();
    let mut a = [0.0f32; 4];
    if !chunks.is_empty() {
        // Up to 383 entries past the start, so possibly past the end; only `base + x` is read.
        let base = table.as_ptr().wrapping_offset(255 - isize::from(max));
        let [mut a0, mut a1, mut a2, mut a3] = a;
        // SAFETY: reads the chunks' bytes, and table entries `255 - max + x` for bytes `x` of
        // the row, which lie in `0..=255` because `-128 <= x <= max <= 127`.
        unsafe {
            core::arch::asm!(
                "movi d20, #0",
                "movi d21, #0",
                "movi d22, #0",
                "movi d23, #0",
                "2:",
                "ldp x10, x11, [{src}], #16",
                "fadd s0, s0, s20",
                "fadd s1, s1, s21",
                "fadd s2, s2, s22",
                "fadd s3, s3, s23",
                "sbfx x12, x10, #0, #8",
                "sbfx x13, x10, #8, #8",
                "ldr s16, [{base}, x12, lsl #2]",
                "sbfx x14, x10, #16, #8",
                "ldr s17, [{base}, x13, lsl #2]",
                "sbfx x15, x10, #24, #8",
                "ldr s18, [{base}, x14, lsl #2]",
                "sbfx x12, x10, #32, #8",
                "ldr s19, [{base}, x15, lsl #2]",
                "sbfx x13, x10, #40, #8",
                "ldr s20, [{base}, x12, lsl #2]",
                "fadd s0, s0, s16",
                "sbfx x14, x10, #48, #8",
                "ldr s21, [{base}, x13, lsl #2]",
                "fadd s1, s1, s17",
                "sbfx x15, x10, #56, #8",
                "ldr s22, [{base}, x14, lsl #2]",
                "fadd s2, s2, s18",
                "sbfx x12, x11, #0, #8",
                "ldr s23, [{base}, x15, lsl #2]",
                "fadd s3, s3, s19",
                "sbfx x13, x11, #8, #8",
                "ldr s16, [{base}, x12, lsl #2]",
                "fadd s0, s0, s20",
                "sbfx x14, x11, #16, #8",
                "ldr s17, [{base}, x13, lsl #2]",
                "fadd s1, s1, s21",
                "sbfx x15, x11, #24, #8",
                "ldr s18, [{base}, x14, lsl #2]",
                "fadd s2, s2, s22",
                "sbfx x12, x11, #32, #8",
                "ldr s19, [{base}, x15, lsl #2]",
                "fadd s3, s3, s23",
                "sbfx x13, x11, #40, #8",
                "ldr s20, [{base}, x12, lsl #2]",
                "fadd s0, s0, s16",
                "sbfx x14, x11, #48, #8",
                "ldr s21, [{base}, x13, lsl #2]",
                "fadd s1, s1, s17",
                "sbfx x15, x11, #56, #8",
                "ldr s22, [{base}, x14, lsl #2]",
                "fadd s2, s2, s18",
                "ldr s23, [{base}, x15, lsl #2]",
                "fadd s3, s3, s19",
                "subs {n}, {n}, #1",
                "b.ne 2b",
                "fadd s0, s0, s20",
                "fadd s1, s1, s21",
                "fadd s2, s2, s22",
                "fadd s3, s3, s23",
                src = inout(reg) chunks.as_ptr() => _,
                base = in(reg) base,
                n = inout(reg) chunks.len() => _,
                out("x10") _, out("x11") _, out("x12") _, out("x13") _, out("x14") _, out("x15") _,
                inout("v0") a0, inout("v1") a1, inout("v2") a2, inout("v3") a3,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                options(nostack, readonly),
            );
        }
        a = [a0, a1, a2, a3];
    }
    let (quads, tail) = rest.as_chunks::<4>();
    for q in quads {
        for (acc, &x) in a.iter_mut().zip(q) {
            *acc += table[table_index(x, off)];
        }
    }
    let mut s = (a[0] + a[1]) + (a[2] + a[3]);
    for &x in tail {
        s += table[table_index(x, off)];
    }
    s
}

/// `sum` in the module's summation order.
fn sum4(vals: &[f32]) -> f32 {
    let mut a = [0.0f32; 4];
    let (quads, tail) = vals.as_chunks::<4>();
    for q in quads {
        for (acc, &t) in a.iter_mut().zip(q) {
            *acc += t;
        }
    }
    let mut s = (a[0] + a[1]) + (a[2] + a[3]);
    for &t in tail {
        s += t;
    }
    s
}

/// Quantizes 16 probabilities `t * inv` to i8 with zero point `zp`, saturating like the scalar path.
#[target_feature(enable = "neon")]
fn quantize16(t: &[f32; 16], inv: float32x4_t, zp: int32x4_t) -> int8x16_t {
    // SAFETY: `t` is 16 readable floats.
    let v = unsafe { vld1q_f32_x4(t.as_ptr()) };
    let q0 = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(v.0, inv)), zp);
    let q1 = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(v.1, inv)), zp);
    let q2 = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(v.2, inv)), zp);
    let q3 = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(v.3, inv)), zp);
    let lo = vqmovn_high_s32(vqmovn_s32(q0), q1);
    let hi = vqmovn_high_s32(vqmovn_s32(q2), q3);
    vqmovn_high_s16(vqmovn_s16(lo), hi)
}

/// Short rows: gather, sum, then quantize every element.
#[target_feature(enable = "neon")]
fn direct(table: &[f32; 256], out_scale: f32, zp: i32, off: u8, src: &[i8], dst: &mut [i8]) {
    let mut buf = [0.0f32; DIRECT_MAX];
    for (b, &x) in buf.iter_mut().zip(src) {
        *b = table[table_index(x, off)];
    }
    let vals = buf.get(..src.len()).unwrap_or(&buf);
    let inv = 1.0 / (sum4(vals) * out_scale);
    let (invv, zpv) = (vdupq_n_f32(inv), vdupq_n_s32(zp));
    let (chunks, tail) = vals.as_chunks::<16>();
    let (dchunks, dtail) = dst.as_chunks_mut::<16>();
    for (c, d) in chunks.iter().zip(dchunks.iter_mut()) {
        // SAFETY: `d` is 16 writable bytes.
        unsafe { vst1q_s8(d.as_mut_ptr(), quantize16(c, invv, zpv)) };
    }
    for (d, &t) in dtail.iter_mut().zip(tail) {
        *d = quantize_prob(t * inv, zp);
    }
}

/// Fills `otab[start..]` with the quantized output of each table index; `start` is a multiple of 16.
#[target_feature(enable = "neon")]
fn build_out_table(table: &[f32; 256], inv: f32, zp: i32, start: usize, otab: &mut [i8; 256]) {
    let (invv, zpv) = (vdupq_n_f32(inv), vdupq_n_s32(zp));
    let (Some(t), Some(o)) = (table.get(start..), otab.get_mut(start..)) else {
        return;
    };
    for (t, o) in t.as_chunks::<16>().0.iter().zip(o.as_chunks_mut::<16>().0) {
        // SAFETY: `o` is 16 writable bytes.
        unsafe { vst1q_s8(o.as_mut_ptr(), quantize16(t, invv, zpv)) };
    }
}

/// Maps each `x` to `otab[255 - max + x]`, where indices below the top `S` 64-byte slices all
/// map to `zero`.
///
/// The slices are loaded XOR `zero`, so each slice's `tbl` gives 0 for lanes outside it and
/// the slices combine with independent `tbl`s and `orr`s instead of a chain of `tbx`.
#[target_feature(enable = "neon")]
fn lookup<const S: usize>(otab: &[i8; 256], zero: i8, off: u8, src: &[i8], dst: &mut [i8]) {
    let z = vdupq_n_s8(zero);
    let t: [int8x16x4_t; S] = core::array::from_fn(|j| {
        // SAFETY: slice `4 - S + j` is 64 readable bytes of the 256-byte table.
        let v = unsafe { vld1q_s8_x4(otab.as_ptr().add(64 * (4 - S + j))) };
        int8x16x4_t(
            veorq_s8(v.0, z),
            veorq_s8(v.1, z),
            veorq_s8(v.2, z),
            veorq_s8(v.3, z),
        )
    });
    let base: [uint8x16_t; S] = core::array::from_fn(|j| vdupq_n_u8((64 * (4 - S + j)) as u8));
    let offv = vdupq_n_u8(off);
    let map = |x: int8x16_t| {
        let idx = vaddq_u8(vreinterpretq_u8_s8(x), offv);
        let mut r = vdupq_n_s8(0);
        for (tj, bj) in t.iter().zip(&base) {
            r = vorrq_s8(r, vqtbl4q_s8(*tj, vsubq_u8(idx, *bj)));
        }
        veorq_s8(r, z)
    };
    // Four vectors per step, each stage across all four, so the in-order core has independent
    // work while a load or `tbl` completes.
    let (quads, rest) = src.as_chunks::<64>();
    let (dquads, drest) = dst.as_chunks_mut::<64>();
    for (c, d) in quads.iter().zip(dquads.iter_mut()) {
        // SAFETY: `c` is 64 readable bytes and `d` 64 writable ones.
        unsafe {
            let x = vld1q_s8_x4(c.as_ptr());
            let r = [x.0, x.1, x.2, x.3].map(map);
            vst1q_s8_x4(d.as_mut_ptr(), int8x16x4_t(r[0], r[1], r[2], r[3]));
        }
    }
    let (chunks, tail) = rest.as_chunks::<16>();
    let (dchunks, dtail) = drest.as_chunks_mut::<16>();
    for (c, d) in chunks.iter().zip(dchunks.iter_mut()) {
        // SAFETY: `c` is 16 readable bytes and `d` 16 writable ones.
        unsafe { vst1q_s8(d.as_mut_ptr(), map(vld1q_s8(c.as_ptr()))) };
    }
    for (d, &x) in dtail.iter_mut().zip(tail) {
        *d = otab[table_index(x, off)];
    }
}

/// A 16×16 group transposed into registers: `v[j]` holds element `j` of every row, and `max`
/// each row's largest element.
///
/// # Safety
/// The group must be 16 rows of 16.
#[inline]
#[target_feature(enable = "neon")]
pub(super) unsafe fn load16(g: &RowGroup<'_, i8>) -> ([uint8x16_t; 16], int8x16_t) {
    // SAFETY: each of the 16 rows is readable for 16 elements (caller).
    let rows: [uint8x16_t; 16] = core::array::from_fn(|k| unsafe { vld1q_u8(g.src_ptr(k).cast()) });
    let v = transpose16(rows);
    let mut max = vreinterpretq_s8_u8(v[0]);
    for &x in &v[1..] {
        max = vmaxq_s8(max, vreinterpretq_s8_u8(x));
    }
    (v, max)
}

/// The byte planes of split exponential tables (see `split_exp_tables`) in registers.
pub(super) struct SplitExp {
    planes: [uint8x16_t; 4],
}

impl SplitExp {
    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn new(t: &[[u8; 16]; 4]) -> Self {
        // SAFETY: four 16-byte tables.
        Self {
            planes: core::array::from_fn(|i| unsafe { vld1q_u8(t[i].as_ptr()) }),
        }
    }

    /// For each of four transposed elements `x[u]`, the `A` and `B` entries of `max − x[u]`
    /// as u16 halves: rows 0..8, then rows 8..16.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) fn entries(
        &self,
        max: int8x16_t,
        x: &[uint8x16_t; 4],
    ) -> ([[uint16x8_t; 2]; 4], [[uint16x8_t; 2]; 4]) {
        let [al, ah, bl, bh] = self.planes;
        let low4 = vdupq_n_u8(15);
        let d: [uint8x16_t; 4] =
            core::array::from_fn(|u| vreinterpretq_u8_s8(vsubq_s8(max, vreinterpretq_s8_u8(x[u]))));
        let pair = |l: uint8x16_t, h: uint8x16_t| {
            [
                vreinterpretq_u16_u8(vzip1q_u8(l, h)),
                vreinterpretq_u16_u8(vzip2q_u8(l, h)),
            ]
        };
        let a = core::array::from_fn(|u| {
            let hi = vshrq_n_u8::<4>(d[u]);
            pair(vqtbl1q_u8(al, hi), vqtbl1q_u8(ah, hi))
        });
        let b = core::array::from_fn(|u| {
            let lo = vandq_u8(d[u], low4);
            pair(vqtbl1q_u8(bl, lo), vqtbl1q_u8(bh, lo))
        });
        (a, b)
    }
}

/// Sixteen rows of sixteen elements at once, one row per lane: the rows are transposed in
/// registers, each lane's max taken across the sixteen element vectors, every `exp` formed
/// from the split tables of [`super::softmax_i8::exp16_tables`] as an exact `A·B` product
/// (1.0 is 65535²), and the quantized outputs transposed back. Each `exp` is within about
/// 2⁻¹⁶ of the float table's and the products are rounded, not truncated, so outputs stay
/// within ±1 LSB of the float-table row kernel without bias. Each lane computes exactly what
/// [`super::softmax_i8::softmax_row16_split`] computes for its row.
///
/// Does nothing unless the group is 16 rows of 16.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn rows16(exp16: &[[u8; 16]; 4], out_scale: f32, g: &RowGroup<'_, i8>) {
    if g.count() != 16 || g.len() != 16 || g.dst_len() != 16 {
        return;
    }
    // SAFETY: the group is 16 rows of 16 (checked above).
    let (v, max) = unsafe { load16(g) };
    let tables = SplitExp::new(exp16);
    // `e[j][q]`: element `j` of rows `4q..4q + 4`; every entry is written before it is read.
    let mut e = core::mem::MaybeUninit::<[[uint32x4_t; 4]; 16]>::uninit();
    let ep: *mut [uint32x4_t; 4] = e.as_mut_ptr().cast();
    let mut sum = [vdupq_n_u32(0); 4];
    // Four elements per step, each stage across all four, so the in-order core always has
    // independent work.
    for (j0, x) in v.as_chunks::<4>().0.iter().enumerate() {
        let (a, b) = tables.entries(max, x);
        for u in 0..4 {
            let p = [
                vmull_u16(vget_low_u16(a[u][0]), vget_low_u16(b[u][0])),
                vmull_high_u16(a[u][0], b[u][0]),
                vmull_u16(vget_low_u16(a[u][1]), vget_low_u16(b[u][1])),
                vmull_high_u16(a[u][1], b[u][1]),
            ];
            // Sixteen terms below 2³² each fit a u32 lane after a rounding shift by 4.
            for (s, &t) in sum.iter_mut().zip(&p) {
                *s = vrsraq_n_u32::<4>(*s, t);
            }
            // SAFETY: `4·j0 + u < 16` entries of `e`.
            unsafe { ep.add(4 * j0 + u).write(p) };
        }
    }
    // SAFETY: the loop above wrote all 16 entries.
    let e = unsafe { e.assume_init() };
    // `q = e / (16·sum·out_scale)`, `e` and `16·sum` both in units of 65535².
    let one = vdupq_n_f32(1.0);
    let c: [float32x4_t; 4] = core::array::from_fn(|q| {
        vdivq_f32(one, vmulq_n_f32(vcvtq_f32_u32(sum[q]), 16.0 * out_scale))
    });
    // The output zero point is -128 (the plan requires it): `q` lies in 0..=256, so it is
    // saturated to a byte and its top bit flipped, which subtracts 128.
    let flip = vdupq_n_u8(0x80);
    let mut out = [vdupq_n_u8(0); 16];
    for (o, ej) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(e.as_chunks::<4>().0)
    {
        let q: [[int32x4_t; 4]; 4] = core::array::from_fn(|u| {
            core::array::from_fn(|i| vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_u32(ej[u][i]), c[i])))
        });
        for (u, ou) in o.iter_mut().enumerate() {
            let lo = vmovn_high_s32(vmovn_s32(q[u][0]), q[u][1]);
            let hi = vmovn_high_s32(vmovn_s32(q[u][2]), q[u][3]);
            *ou = veorq_u8(vqmovun_high_s16(vqmovun_s16(lo), hi), flip);
        }
    }
    for (k, row) in transpose16(out).into_iter().enumerate() {
        // SAFETY: each of the 16 rows is writable for 16 elements and this group's alone.
        unsafe { vst1q_u8(g.dst_ptr(k).cast(), row) };
    }
}

#[cfg(test)]
mod tests {
    use super::super::softmax_i8::softmax_row_i8_portable;
    use super::*;
    use rand::{RngExt, SeedableRng};

    fn table(input_scale: f32) -> [f32; 256] {
        let mut t = [0.0f32; 256];
        for (v, e) in t.iter_mut().rev().enumerate() {
            *e = (-input_scale * v as f32).exp();
        }
        t
    }

    /// Compares the NEON row with the portable reference; returns the off-by-one count.
    fn check(table: &[f32; 256], row: &[i8], what: &str) -> usize {
        let scale = 1.0 / 256.0;
        let mut want = vec![0x55i8; row.len()];
        let mut got = vec![0x55i8; row.len()];
        softmax_row_i8_portable(table, scale, -128, row, &mut want);
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i8_neon(table, &slice_max(table), scale, -128, row, &mut got) };
        let mut off = 0;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let d = (i32::from(g) - i32::from(w)).abs();
            assert!(
                d <= 1,
                "{what}: len {} element {i}: neon {g} portable {w}",
                row.len()
            );
            off += usize::from(d == 1);
        }
        off
    }

    #[test]
    fn neon_row_matches_portable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x50F7_3A11);
        let mut total = 0usize;
        let mut off = 0usize;
        for &scale in &[0.0125f32, 0.05, 0.128, 0.5] {
            let t = table(scale);
            let mut lens: Vec<usize> = (1..=70).collect();
            lens.extend([255, 256, 257, 1008, 1009, 4097, 70_000]);
            for &n in &lens {
                // Spans of 1, 16, 63, 64, 65, 128, 192 and 256 values exercise every table-slice count.
                for &span in &[1i32, 16, 63, 64, 65, 128, 192, 256] {
                    let base = rng.random_range(-128..=(128 - span));
                    let row: Vec<i8> = (0..n)
                        .map(|_| (base + rng.random_range(0..span)) as i8)
                        .collect();
                    off += check(&t, &row, &format!("random span {span}"));
                    total += n;
                }
                for v in [-128i8, -1, 0, 127] {
                    off += check(&t, &vec![v; n], "all equal");
                    total += n;
                }
                let mut ext: Vec<i8> = (0..n)
                    .map(|i| if i % 2 == 0 { -128 } else { 127 })
                    .collect();
                off += check(&t, &ext, "alternating extremes");
                ext.reverse();
                off += check(&t, &ext, "alternating extremes reversed");
                total += 2 * n;
            }
        }
        eprintln!("neon vs portable: {off} of {total} off by one");
    }

    /// The pipelined gather adds element `j` into accumulator `j % 4`, in order, exactly as a
    /// plain loop does, for every row maximum (negative ones put the table base past its end).
    #[test]
    fn gather_sum_is_the_interleaved_sum_bit_for_bit() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x6A7E_5011);
        let t = table(0.0947);
        for n in [16usize, 17, 31, 32, 47, 64, 400, 401, 403] {
            for max in [-128i8, -100, -1, 0, 1, 37, 127] {
                let row: Vec<i8> = (0..n)
                    .map(|i| {
                        if i == n / 2 {
                            max
                        } else {
                            rng.random_range(-128..=i32::from(max)) as i8
                        }
                    })
                    .collect();
                let off = table_offset(max);
                let mut a = [0.0f32; 4];
                let quads = n / 4 * 4;
                for (j, &x) in row.iter().take(quads).enumerate() {
                    a[j % 4] += t[table_index(x, off)];
                }
                let mut want = (a[0] + a[1]) + (a[2] + a[3]);
                for &x in row.iter().skip(quads) {
                    want += t[table_index(x, off)];
                }
                // SAFETY: NEON is part of the aarch64 baseline.
                let got = unsafe { gather_sum(&t, max, &row) };
                assert_eq!(got.to_bits(), want.to_bits(), "len {n} max {max}");
            }
        }
    }

    #[test]
    fn neon_row_handles_mismatched_and_empty_slices() {
        let t = table(0.1);
        let src = [1i8, 2, 3];
        let mut dst = [0x55i8; 2];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i8_neon(&t, &slice_max(&t), 1.0 / 256.0, -128, &src, &mut dst) };
        let mut want = [0x55i8; 2];
        softmax_row_i8_portable(&t, 1.0 / 256.0, -128, &src, &mut want);
        assert_eq!(dst, want);
        let mut empty: [i8; 0] = [];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i8_neon(&t, &slice_max(&t), 1.0 / 256.0, -128, &[], &mut empty) };
    }

    /// Rows of 16 in groups of 16 (the DFL layout: `[4, anchors, 16]` read from
    /// `[anchors, 64]`) stay within ±1 LSB of the portable plan, rarely and without bias.
    #[test]
    fn row_groups_match_portable() {
        use super::super::softmax_i8::{SoftmaxI8Params, SoftmaxI8Plan};
        use crate::cpu::CpuFeatures;
        use crate::quant::Quant;
        use crate::view::{TensorMut, TensorRef, View, ViewMut};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDF1_0016);
        let (mut total, mut off, mut signed) = (0usize, 0usize, 0i64);
        for &scale in &[0.0125f32, 0.05, 0.128, 0.5] {
            let p = SoftmaxI8Params {
                input: Quant::new(scale, 0),
                output: Quant::new(1.0 / 256.0, -128),
                beta: 1.0,
            };
            let neon = SoftmaxI8Plan::new(p).unwrap();
            let portable = SoftmaxI8Plan::with_features(p, CpuFeatures::default()).unwrap();
            for anchors in [1usize, 4, 15, 16, 17, 37, 400] {
                let span = rng.random_range(1..=256i32);
                let base = rng.random_range(-128..=(128 - span));
                let src: Vec<i8> = (0..anchors * 64)
                    .map(|_| (base + rng.random_range(0..span)) as i8)
                    .collect();
                let iv = View::strided(&src, &[4, anchors, 16], &[16, 64, 1]).unwrap();
                let run = |plan: &SoftmaxI8Plan| {
                    let mut out = vec![0x55i8; anchors * 64];
                    plan.run(
                        TensorRef::I8(iv),
                        TensorMut::I8(ViewMut::contiguous(&mut out, &[4, anchors, 16]).unwrap()),
                        None,
                    )
                    .unwrap();
                    out
                };
                let (got, want) = (run(&neon), run(&portable));
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    let d = i32::from(g) - i32::from(w);
                    assert!(
                        d.abs() <= 1,
                        "scale {scale} anchors {anchors} at {i}: {g} vs {w}"
                    );
                    off += usize::from(d != 0);
                    signed += i64::from(d);
                }
                total += got.len();
            }
        }
        assert!(off * 100 <= total, "{off} of {total} off by one");
        assert!(
            signed.unsigned_abs() as usize * 4 <= off.max(8),
            "biased: net {signed} of {off}"
        );
    }

    /// The 16-row kernel gives each row exactly what the plan's single-row path gives it, so a
    /// 16-element row's result does not depend on whether it was grouped.
    #[test]
    fn row_groups_equal_single_rows() {
        use super::super::softmax_i8::{SoftmaxI8Params, SoftmaxI8Plan};
        use crate::quant::Quant;
        use crate::view::{TensorMut, TensorRef, View, ViewMut};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x16_0016);
        for &scale in &[0.0125f32, 0.05, 0.0947, 0.128, 0.5, 2.0] {
            let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
                input: Quant::new(scale, 0),
                output: Quant::new(1.0 / 256.0, -128),
                beta: 1.0,
            })
            .unwrap();
            let rows = 16 * 8;
            for span in [1i32, 16, 64, 256] {
                let base = rng.random_range(-128..=(128 - span));
                let src: Vec<i8> = (0..rows * 16)
                    .map(|_| (base + rng.random_range(0..span)) as i8)
                    .collect();
                let mut grouped = vec![0x55i8; rows * 16];
                plan.run(
                    TensorRef::I8(View::contiguous(&src, &[rows, 16]).unwrap()),
                    TensorMut::I8(ViewMut::contiguous(&mut grouped, &[rows, 16]).unwrap()),
                    None,
                )
                .unwrap();
                let mut single = vec![0x55i8; rows * 16];
                for (s, d) in src.chunks_exact(16).zip(single.chunks_exact_mut(16)) {
                    plan.row(s, d);
                }
                assert_eq!(grouped, single, "scale {scale} span {span}");
            }
        }
    }
}
