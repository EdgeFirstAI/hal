// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int8 softmax row kernel.
//!
//! A row's outputs depend only on the table index `255 - max + x`, so after one gathered f32
//! sum the kernel builds a per-row byte table of quantized outputs and maps the row through it
//! with `tbl`/`tbx`. Short rows with a wide value range quantize each element instead, which is
//! cheaper than building the table. Both paths sum in the same order: four interleaved
//! accumulators (element `j` into lane `j % 4`), combined pairwise, then the `n % 4` tail.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::softmax_i8::{quantize_prob, table_index, table_offset};

/// Longest row that may take the per-element path.
const DIRECT_MAX: usize = 64;

/// NEON softmax of one row; processes `min(src.len(), dst.len())` elements.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_row_i8_neon(
    table: &[f32; 256],
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
    let inv = 1.0 / (gather_sum(table, off, src) * out_scale);
    let mut otab = [0i8; 256];
    build_out_table(table, inv, zp, 256 - entries, &mut otab);
    match 4 - (255 - range) / 64 {
        1 => lookup::<1>(&otab, off, src, dst),
        2 => lookup::<2>(&otab, off, src, dst),
        3 => lookup::<3>(&otab, off, src, dst),
        _ => lookup::<4>(&otab, off, src, dst),
    }
}

#[target_feature(enable = "neon")]
fn min_max(src: &[i8]) -> (i8, i8) {
    let (chunks, tail) = src.as_chunks::<16>();
    let (mut lo, mut hi) = (i8::MAX, i8::MIN);
    if !chunks.is_empty() {
        let (mut vlo, mut vhi) = (vdupq_n_s8(i8::MAX), vdupq_n_s8(i8::MIN));
        for c in chunks {
            // SAFETY: `c` is 16 readable bytes.
            let v = unsafe { vld1q_s8(c.as_ptr()) };
            vlo = vminq_s8(vlo, v);
            vhi = vmaxq_s8(vhi, v);
        }
        lo = vminvq_s8(vlo);
        hi = vmaxvq_s8(vhi);
    }
    for &x in tail {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo, hi)
}

/// The row sum of `table[255 - max + x]`, in the module's summation order.
#[target_feature(enable = "neon")]
fn gather_sum(table: &[f32; 256], off: u8, src: &[i8]) -> f32 {
    let offv = vdupq_n_u8(off);
    let mut a = [0.0f32; 4];
    let (chunks, rest) = src.as_chunks::<16>();
    for c in chunks {
        // SAFETY: `c` is 16 readable bytes.
        let v = unsafe { vld1q_s8(c.as_ptr()) };
        let idx = vreinterpretq_u64_u8(vaddq_u8(vreinterpretq_u8_s8(v), offv));
        // Unpacking two u64 lanes keeps the gather at one load per element.
        for w in [vgetq_lane_u64::<0>(idx), vgetq_lane_u64::<1>(idx)] {
            a[0] += table[(w & 0xff) as usize];
            a[1] += table[((w >> 8) & 0xff) as usize];
            a[2] += table[((w >> 16) & 0xff) as usize];
            a[3] += table[((w >> 24) & 0xff) as usize];
            a[0] += table[((w >> 32) & 0xff) as usize];
            a[1] += table[((w >> 40) & 0xff) as usize];
            a[2] += table[((w >> 48) & 0xff) as usize];
            a[3] += table[((w >> 56) & 0xff) as usize];
        }
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

/// Maps each `x` to `otab[255 - max + x]`; indices all lie in the top `S` 64-byte slices.
#[target_feature(enable = "neon")]
fn lookup<const S: usize>(otab: &[i8; 256], off: u8, src: &[i8], dst: &mut [i8]) {
    let p = otab.as_ptr();
    // SAFETY: each load reads 64 bytes at offset 0, 64, 128 or 192 of the 256-byte table.
    let (t0, t1, t2, t3) = unsafe {
        (
            vld1q_s8_x4(p),
            vld1q_s8_x4(p.add(64)),
            vld1q_s8_x4(p.add(128)),
            vld1q_s8_x4(p.add(192)),
        )
    };
    let offv = vdupq_n_u8(off);
    let (k64, k128, k192) = (vdupq_n_u8(64), vdupq_n_u8(128), vdupq_n_u8(192));
    let (chunks, tail) = src.as_chunks::<16>();
    let (dchunks, dtail) = dst.as_chunks_mut::<16>();
    for (c, d) in chunks.iter().zip(dchunks.iter_mut()) {
        // SAFETY: `c` is 16 readable bytes.
        let x = unsafe { vld1q_s8(c.as_ptr()) };
        let idx = vaddq_u8(vreinterpretq_u8_s8(x), offv);
        // `tbl` zeroes and `tbx` keeps lanes whose shifted index falls outside its 64-byte slice.
        let mut r = vqtbl4q_s8(t3, vsubq_u8(idx, k192));
        if S >= 2 {
            r = vqtbx4q_s8(r, t2, vsubq_u8(idx, k128));
        }
        if S >= 3 {
            r = vqtbx4q_s8(r, t1, vsubq_u8(idx, k64));
        }
        if S >= 4 {
            r = vqtbx4q_s8(r, t0, idx);
        }
        // SAFETY: `d` is 16 writable bytes.
        unsafe { vst1q_s8(d.as_mut_ptr(), r) };
    }
    for (d, &x) in dtail.iter_mut().zip(tail) {
        *d = otab[table_index(x, off)];
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
        unsafe { softmax_row_i8_neon(table, scale, -128, row, &mut got) };
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

    #[test]
    fn neon_row_handles_mismatched_and_empty_slices() {
        let t = table(0.1);
        let src = [1i8, 2, 3];
        let mut dst = [0x55i8; 2];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i8_neon(&t, 1.0 / 256.0, -128, &src, &mut dst) };
        let mut want = [0x55i8; 2];
        softmax_row_i8_portable(&t, 1.0 / 256.0, -128, &src, &mut want);
        assert_eq!(dst, want);
        let mut empty: [i8; 0] = [];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { softmax_row_i8_neon(&t, 1.0 / 256.0, -128, &[], &mut empty) };
    }
}
