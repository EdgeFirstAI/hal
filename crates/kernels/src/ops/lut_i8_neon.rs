// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON 256-entry byte table lookup.
//!
//! The table sits in four `uint8x16x4_t` registers of 64 bytes each. `tbl` on the first block
//! zeroes lanes whose index is 64 or more; `tbx` on each later block, with the index lowered by
//! another 64, fills only the lanes that land in that block and leaves the rest.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use super::lut_i8::{lut_portable, Byte};

/// Looks up `min(src.len(), dst.len())` elements through `table`.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn lut_neon<T: Byte>(table: &[u8; 256], src: &[T], dst: &mut [T]) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    let body = n - n % 16;
    // SAFETY: `T` is one byte, any bit pattern valid; accesses stay below `body <= n` and within `table`.
    unsafe {
        let t = table.as_ptr();
        let (t0, t1, t2, t3) = (
            vld1q_u8_x4(t),
            vld1q_u8_x4(t.add(64)),
            vld1q_u8_x4(t.add(128)),
            vld1q_u8_x4(t.add(192)),
        );
        let step = vdupq_n_u8(64);
        let s = src.as_ptr().cast::<u8>();
        let d = dst.as_mut_ptr().cast::<u8>();
        let lookup = |x: uint8x16_t| -> uint8x16_t {
            let x1 = vsubq_u8(x, step);
            let x2 = vsubq_u8(x1, step);
            let x3 = vsubq_u8(x2, step);
            let r = vqtbl4q_u8(t0, x);
            let r = vqtbx4q_u8(r, t1, x1);
            let r = vqtbx4q_u8(r, t2, x2);
            vqtbx4q_u8(r, t3, x3)
        };
        let mut i = 0usize;
        while i + 32 <= body {
            let a = vld1q_u8(s.add(i));
            let b = vld1q_u8(s.add(i + 16));
            vst1q_u8(d.add(i), lookup(a));
            vst1q_u8(d.add(i + 16), lookup(b));
            i += 32;
        }
        if i < body {
            vst1q_u8(d.add(i), lookup(vld1q_u8(s.add(i))));
        }
    }
    if let (Some(s), Some(d)) = (src.get(body..), dst.get_mut(body..)) {
        lut_portable(table, s, d);
    }
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
}
