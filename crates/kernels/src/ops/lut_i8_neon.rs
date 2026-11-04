// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON 256-entry byte table lookup.
//!
//! The table sits in up to four `uint8x16x4_t` registers of 64 bytes each. `tbl` on the first
//! block zeroes lanes whose index is 64 or more; `tbx` on each later block `b`, with the index
//! lowered by `64·b`, fills only the lanes that land in that block and leaves the rest.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::lut_i8::{lut_portable, Byte, Window};
use super::rows::for_each_transposed_vector;
use crate::error::Result;
use crate::pool::Pool;
use crate::view::{View, ViewMut};

/// The first `B` (at most 4) 64-byte blocks of `table`, for [`lookup`].
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn table_blocks<const B: usize>(table: &[u8; 256]) -> [uint8x16x4_t; B] {
    // SAFETY: `B <= 4` blocks of 64 lie in the 256-byte table.
    core::array::from_fn(|b| unsafe { vld1q_u8_x4(table.as_ptr().add(64 * (b % 4))) })
}

/// Each lane of `idx` looked up in `blocks`; lanes at `64·B` or more come out 0.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn lookup<const B: usize>(blocks: &[uint8x16x4_t; B], idx: uint8x16_t) -> uint8x16_t {
    let mut r = vqtbl4q_u8(blocks[0], idx);
    for (b, block) in blocks.iter().enumerate().skip(1) {
        // Indices below the block wrap past 191 and leave their lanes unchanged.
        r = vqtbx4q_u8(r, *block, vsubq_u8(idx, vdupq_n_u8((64 * b) as u8)));
    }
    r
}

/// `f` over whole 16-byte vectors of the first `min(src.len(), dst.len())` elements, then
/// `tail` over the rest.
#[inline]
#[target_feature(enable = "neon")]
fn map_vectors<T: Byte>(
    src: &[T],
    dst: &mut [T],
    f: impl Fn(uint8x16_t) -> uint8x16_t,
    tail: impl FnOnce(&[T], &mut [T]),
) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    let body = n - n % 16;
    let s = src.as_ptr().cast::<u8>();
    let d = dst.as_mut_ptr().cast::<u8>();
    // SAFETY: `T` is one byte, any bit pattern valid; every access stays below `body <= n`.
    unsafe {
        let mut i = 0usize;
        while i + 32 <= body {
            let a = vld1q_u8(s.add(i));
            let b = vld1q_u8(s.add(i + 16));
            vst1q_u8(d.add(i), f(a));
            vst1q_u8(d.add(i + 16), f(b));
            i += 32;
        }
        if i < body {
            vst1q_u8(d.add(i), f(vld1q_u8(s.add(i))));
        }
    }
    if let (Some(s), Some(d)) = (src.get(body..), dst.get_mut(body..)) {
        tail(s, d);
    }
}

/// Looks up `min(src.len(), dst.len())` elements through `table`.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn lut_neon<T: Byte>(table: &[u8; 256], src: &[T], dst: &mut [T]) {
    let blocks = table_blocks::<4>(table);
    map_vectors(
        src,
        dst,
        |x| lookup(&blocks, x),
        |s, d| lut_portable(table, s, d),
    );
}

/// `w`'s lookup of 16 values: each clamped into the window, then mapped through `B` blocks.
#[inline]
#[target_feature(enable = "neon")]
fn window_lookup<const B: usize>(w: &Window) -> impl Fn(uint8x16_t) -> uint8x16_t + Copy {
    let blocks = table_blocks::<B>(&w.table);
    let (flip, lo, hi) = (vdupq_n_u8(w.flip), vdupq_n_u8(w.lo), vdupq_n_u8(w.hi));
    move |x| {
        lookup(
            &blocks,
            vsubq_u8(vminq_u8(vmaxq_u8(veorq_u8(x, flip), lo), hi), lo),
        )
    }
}

/// Looks up `min(src.len(), dst.len())` elements through `w`: each value clamped into the
/// window and mapped through `B` 64-byte blocks instead of four.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does; `B` must be `w.blocks()` (1 to 3).
#[target_feature(enable = "neon")]
pub(super) unsafe fn lut_window<const B: usize, T: Byte>(w: &Window, src: &[T], dst: &mut [T]) {
    map_vectors(src, dst, window_lookup::<B>(w), |s, d| {
        w.lookup_portable(s, d)
    });
}

/// [`lut_window`] over transposed views: each 16×16 tile is transposed in registers, looked up
/// and stored straight to the output (see [`for_each_transposed_vector`]); `None`, touching
/// nothing, when the views do not suit it.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does; `B` must be `w.blocks()` (1 to 4).
#[target_feature(enable = "neon")]
pub(super) unsafe fn lut_window_transposed<const B: usize, T: Byte + Send + Sync>(
    w: &Window,
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
) -> Option<Result<()>> {
    for_each_transposed_vector(input, output, pool, &window_lookup::<B>(w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn neon_lut_matches_portable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0071_7b11);
        let mut lens: Vec<usize> = (0..=70).collect();
        lens.extend([1009, 4096, 4097]);
        for _ in 0..8 {
            let table: [u8; 256] = core::array::from_fn(|_| rng.random());
            for &n in &lens {
                let src: Vec<u8> = (0..n).map(|_| rng.random()).collect();
                let mut want = vec![0x55u8; n];
                let mut got = vec![0xaau8; n];
                lut_portable(&table, &src, &mut want);
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { lut_neon(&table, &src, &mut got) };
                assert_eq!(got, want, "u8 len {n}");

                let src: Vec<i8> = src.iter().map(|&b| b as i8).collect();
                let mut want = vec![0x55i8; n];
                let mut got = vec![0x2ai8; n];
                lut_portable(&table, &src, &mut want);
                // SAFETY: as above.
                unsafe { lut_neon(&table, &src, &mut got) };
                assert_eq!(got, want, "i8 len {n}");
            }
        }
        // Every byte value through the vector loop.
        let table: [u8; 256] = core::array::from_fn(|i| (i as u8).wrapping_mul(167) ^ 0x5a);
        let all: Vec<u8> = (0..=255u8).cycle().take(256 * 17).collect();
        let mut want = vec![0u8; all.len()];
        let mut got = vec![0u8; all.len()];
        lut_portable(&table, &all, &mut want);
        // SAFETY: as above.
        unsafe { lut_neon(&table, &all, &mut got) };
        assert_eq!(got, want, "all bytes");
    }

    /// Windowed lookups equal the full table on every byte, for windows of one to four blocks.
    #[test]
    fn window_lut_matches_portable() {
        use crate::quant::DType;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x3a1d_0007);
        let windows = [
            (100, 140),
            (10, 200),
            (0, 63),
            (64, 127),
            (5, 5),
            (30, 160),
            (0, 255),
        ];
        for dtype in [DType::I8, DType::U8] {
            let flip = if dtype == DType::I8 { 0x80u8 } else { 0 };
            for &(lo, hi) in &windows {
                let mut table = [0u8; 256];
                for v in 0..256usize {
                    table[usize::from(v as u8 ^ flip)] = if v < lo {
                        7
                    } else if v > hi {
                        200
                    } else {
                        rng.random()
                    };
                }
                let w = Window::new(&table, dtype);
                let all: Vec<u8> = (0..=255u8).cycle().take(256 * 3 + 9).collect();
                let mut want = vec![0u8; all.len()];
                lut_portable(&table, &all, &mut want);
                let mut got = vec![0x55u8; all.len()];
                w.lookup_portable(&all, &mut got);
                assert_eq!(got, want, "portable window {dtype:?} {lo}..={hi}");
                let mut got = vec![0x55u8; all.len()];
                // SAFETY: NEON is part of the aarch64 baseline; `B` matches `blocks()`.
                unsafe {
                    match w.blocks() {
                        1 => lut_window::<1, u8>(&w, &all, &mut got),
                        2 => lut_window::<2, u8>(&w, &all, &mut got),
                        3 => lut_window::<3, u8>(&w, &all, &mut got),
                        _ => lut_neon(&table, &all, &mut got),
                    }
                }
                assert_eq!(
                    got,
                    want,
                    "neon window {dtype:?} {lo}..={hi} ({} blocks)",
                    w.blocks()
                );
            }
        }
    }
}
