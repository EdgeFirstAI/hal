// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NCHW → NHWC tensor transpose used by the per-scale pipeline when a
//! schema-declared NCHW child is bound at frame time.
//!
//! The per-scale level kernels (`level_box`, `level_score`, `level_mc`)
//! take flat slices and use NHWC-contiguous indexing — each anchor's
//! channel block is read as one consecutive run of `channels` elements.
//! NCHW children, by contrast, place all `h*w` values of channel 0 first,
//! then channel 1, and so on; the per-anchor strided read pattern
//! breaks both the kernel's index math and cache locality.
//!
//! Rather than maintain a parallel NCHW kernel matrix (~doubles the
//! kernel count) the pipeline pays a small per-frame transpose cost
//! into a scratch buffer and then dispatches the existing NHWC kernel.
//! Cost on Cortex-A53 scalar: ~6 ms / frame for a yolo at 640x640
//! (memory-bandwidth-bound). Phase 2 NEON tile-transpose (the same
//! pattern that powers `tiled_proto_transpose_nchw_to_nhwc` on `main`)
//! drops this to ~1.5 ms.
//!
//! NHWC schemas (TFLite, current canonical fixtures) take the
//! pass-through code path in `pipeline::run` and never invoke this
//! module — zero overhead for the canonical case.

/// Generic NCHW → NHWC element transpose.
///
/// `src` is laid out as `[N, C, H, W]` with `N == 1` (the per-scale
/// subsystem only handles single-batch tensors). `dst` is written as
/// `[N, H, W, C]`. Both must be exactly `h * w * c` elements.
///
/// Walks the destination in row-major (`hi`, `wi`, `ci`) order so the
/// destination stores are sequential — typically faster on cores with
/// write-combining stores than walking the source sequentially. Both
/// orderings have the same cache-line count; the difference is in store
/// merging.
///
/// On aarch64, byte-sized inputs (`u8` / `i8`) are routed through a
/// 16×16 NEON tile transpose for the bulk of `hw × c` and tail rows /
/// columns drop to scalar. The byte path is the only one that hits the
/// fast lane today; halfword (i16/u16) and float dtypes use the scalar
/// fallback because they're rarer in the per-scale-NCHW use case
/// (Ara-240 quantizes to int8 throughout).
pub(crate) fn nchw_to_nhwc<T: Copy>(src: &[T], h: usize, w: usize, c: usize, dst: &mut [T]) {
    debug_assert_eq!(src.len(), h * w * c, "src length mismatch");
    debug_assert_eq!(dst.len(), h * w * c, "dst length mismatch");

    // Byte-size dtype fast path on aarch64. The bit pattern of i8/u8 is
    // preserved by the transpose (it's a pure permutation), so we can
    // safely punt the call through the u8 NEON routine via raw bytes.
    #[cfg(target_arch = "aarch64")]
    if std::mem::size_of::<T>() == 1 && c >= 16 && (h * w) >= 16 {
        // SAFETY: T is byte-sized so the raw-byte view is well-formed and
        // the transpose is a bitwise permutation. Dispatching by size
        // matches both i8 and u8 (and any other 1-byte Copy type).
        unsafe {
            let src_bytes = core::slice::from_raw_parts(src.as_ptr() as *const u8, src.len());
            let dst_bytes = core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len());
            nchw_to_nhwc_u8_neon(src_bytes, h, w, c, dst_bytes);
        }
        return;
    }

    let hw = h * w;
    let mut dst_idx = 0;
    for hi in 0..h {
        for wi in 0..w {
            let src_base = hi * w + wi;
            for ci in 0..c {
                dst[dst_idx] = src[ci * hw + src_base];
                dst_idx += 1;
            }
        }
    }
}

/// NEON 16×16 byte tile transpose for NCHW → NHWC.
///
/// Source layout: `[c, hw]` with row stride `hw`. Destination layout:
/// `[hw, c]` with row stride `c`. Bulk of the matrix is processed in
/// 16×16 byte tiles using the canonical 4-stage TRN1/TRN2 pattern; the
/// right and bottom edges fall through to scalar.
///
/// The tile transpose reads each 16-byte register-wide row of source
/// then permutes lanes via four pairwise interleaving stages
/// (`.16b` → `.8h` → `.4s` → `.2d`). Each stage halves the column-vs-row
/// disagreement; after four stages the loaded "rows" hold what were
/// originally "columns", at which point we store them out as the
/// transposed 16×16 destination block.
///
/// # Safety
/// Requires aarch64 NEON (mandatory on aarch64). Caller must ensure
/// `src.len() == dst.len() == h * w * c` and that `c` and `h * w` are
/// each at least 16 — the bulk path indexes 16-wide tiles unchecked.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn nchw_to_nhwc_u8_neon(
    src: &[u8],
    h: usize,
    w: usize,
    c: usize,
    dst: &mut [u8],
) {
    use core::arch::aarch64::*;
    let hw = h * w;
    debug_assert_eq!(src.len(), hw * c);
    debug_assert_eq!(dst.len(), hw * c);
    debug_assert!(hw >= 16 && c >= 16);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // 16-wide tile counts. Tail rows / columns are scalar; the kernel
    // is most useful when both dims are exact multiples of 16 (channel
    // counts of 32 / 64 / 80 are the common per-scale cases).
    let n_tiles_hw = hw / 16;
    let n_tiles_c = c / 16;

    for tile_c in 0..n_tiles_c {
        let c_base = tile_c * 16;
        for tile_hw in 0..n_tiles_hw {
            let hw_base = tile_hw * 16;

            // Load 16 source rows of 16 bytes — each row is 16 spatial
            // positions of one channel.
            let r0 = vld1q_u8(src_ptr.add((c_base) * hw + hw_base));
            let r1 = vld1q_u8(src_ptr.add((c_base + 1) * hw + hw_base));
            let r2 = vld1q_u8(src_ptr.add((c_base + 2) * hw + hw_base));
            let r3 = vld1q_u8(src_ptr.add((c_base + 3) * hw + hw_base));
            let r4 = vld1q_u8(src_ptr.add((c_base + 4) * hw + hw_base));
            let r5 = vld1q_u8(src_ptr.add((c_base + 5) * hw + hw_base));
            let r6 = vld1q_u8(src_ptr.add((c_base + 6) * hw + hw_base));
            let r7 = vld1q_u8(src_ptr.add((c_base + 7) * hw + hw_base));
            let r8 = vld1q_u8(src_ptr.add((c_base + 8) * hw + hw_base));
            let r9 = vld1q_u8(src_ptr.add((c_base + 9) * hw + hw_base));
            let r10 = vld1q_u8(src_ptr.add((c_base + 10) * hw + hw_base));
            let r11 = vld1q_u8(src_ptr.add((c_base + 11) * hw + hw_base));
            let r12 = vld1q_u8(src_ptr.add((c_base + 12) * hw + hw_base));
            let r13 = vld1q_u8(src_ptr.add((c_base + 13) * hw + hw_base));
            let r14 = vld1q_u8(src_ptr.add((c_base + 14) * hw + hw_base));
            let r15 = vld1q_u8(src_ptr.add((c_base + 15) * hw + hw_base));

            // Stage 1: pair-wise byte interleave (TRN1/TRN2 .16b).
            let a0 = vtrn1q_u8(r0, r1);
            let a1 = vtrn2q_u8(r0, r1);
            let a2 = vtrn1q_u8(r2, r3);
            let a3 = vtrn2q_u8(r2, r3);
            let a4 = vtrn1q_u8(r4, r5);
            let a5 = vtrn2q_u8(r4, r5);
            let a6 = vtrn1q_u8(r6, r7);
            let a7 = vtrn2q_u8(r6, r7);
            let a8 = vtrn1q_u8(r8, r9);
            let a9 = vtrn2q_u8(r8, r9);
            let a10 = vtrn1q_u8(r10, r11);
            let a11 = vtrn2q_u8(r10, r11);
            let a12 = vtrn1q_u8(r12, r13);
            let a13 = vtrn2q_u8(r12, r13);
            let a14 = vtrn1q_u8(r14, r15);
            let a15 = vtrn2q_u8(r14, r15);

            // Stage 2: pair-wise halfword interleave (TRN1/TRN2 .8h).
            macro_rules! trn_h {
                ($lo:expr, $hi:expr, $kind:ident) => {
                    vreinterpretq_u8_u16($kind(
                        vreinterpretq_u16_u8($lo),
                        vreinterpretq_u16_u8($hi),
                    ))
                };
            }
            let b0 = trn_h!(a0, a2, vtrn1q_u16);
            let b1 = trn_h!(a1, a3, vtrn1q_u16);
            let b2 = trn_h!(a0, a2, vtrn2q_u16);
            let b3 = trn_h!(a1, a3, vtrn2q_u16);
            let b4 = trn_h!(a4, a6, vtrn1q_u16);
            let b5 = trn_h!(a5, a7, vtrn1q_u16);
            let b6 = trn_h!(a4, a6, vtrn2q_u16);
            let b7 = trn_h!(a5, a7, vtrn2q_u16);
            let b8 = trn_h!(a8, a10, vtrn1q_u16);
            let b9 = trn_h!(a9, a11, vtrn1q_u16);
            let b10 = trn_h!(a8, a10, vtrn2q_u16);
            let b11 = trn_h!(a9, a11, vtrn2q_u16);
            let b12 = trn_h!(a12, a14, vtrn1q_u16);
            let b13 = trn_h!(a13, a15, vtrn1q_u16);
            let b14 = trn_h!(a12, a14, vtrn2q_u16);
            let b15 = trn_h!(a13, a15, vtrn2q_u16);

            // Stage 3: pair-wise word interleave (TRN1/TRN2 .4s).
            macro_rules! trn_s {
                ($lo:expr, $hi:expr, $kind:ident) => {
                    vreinterpretq_u8_u32($kind(
                        vreinterpretq_u32_u8($lo),
                        vreinterpretq_u32_u8($hi),
                    ))
                };
            }
            let d0 = trn_s!(b0, b4, vtrn1q_u32);
            let d1 = trn_s!(b1, b5, vtrn1q_u32);
            let d2 = trn_s!(b2, b6, vtrn1q_u32);
            let d3 = trn_s!(b3, b7, vtrn1q_u32);
            let d4 = trn_s!(b0, b4, vtrn2q_u32);
            let d5 = trn_s!(b1, b5, vtrn2q_u32);
            let d6 = trn_s!(b2, b6, vtrn2q_u32);
            let d7 = trn_s!(b3, b7, vtrn2q_u32);
            let d8 = trn_s!(b8, b12, vtrn1q_u32);
            let d9 = trn_s!(b9, b13, vtrn1q_u32);
            let d10 = trn_s!(b10, b14, vtrn1q_u32);
            let d11 = trn_s!(b11, b15, vtrn1q_u32);
            let d12 = trn_s!(b8, b12, vtrn2q_u32);
            let d13 = trn_s!(b9, b13, vtrn2q_u32);
            let d14 = trn_s!(b10, b14, vtrn2q_u32);
            let d15 = trn_s!(b11, b15, vtrn2q_u32);

            // Stage 4: pair-wise doubleword interleave (TRN1/TRN2 .2d).
            macro_rules! trn_d {
                ($lo:expr, $hi:expr, $kind:ident) => {
                    vreinterpretq_u8_u64($kind(
                        vreinterpretq_u64_u8($lo),
                        vreinterpretq_u64_u8($hi),
                    ))
                };
            }
            let t0 = trn_d!(d0, d8, vtrn1q_u64);
            let t1 = trn_d!(d1, d9, vtrn1q_u64);
            let t2 = trn_d!(d2, d10, vtrn1q_u64);
            let t3 = trn_d!(d3, d11, vtrn1q_u64);
            let t4 = trn_d!(d4, d12, vtrn1q_u64);
            let t5 = trn_d!(d5, d13, vtrn1q_u64);
            let t6 = trn_d!(d6, d14, vtrn1q_u64);
            let t7 = trn_d!(d7, d15, vtrn1q_u64);
            let t8 = trn_d!(d0, d8, vtrn2q_u64);
            let t9 = trn_d!(d1, d9, vtrn2q_u64);
            let t10 = trn_d!(d2, d10, vtrn2q_u64);
            let t11 = trn_d!(d3, d11, vtrn2q_u64);
            let t12 = trn_d!(d4, d12, vtrn2q_u64);
            let t13 = trn_d!(d5, d13, vtrn2q_u64);
            let t14 = trn_d!(d6, d14, vtrn2q_u64);
            let t15 = trn_d!(d7, d15, vtrn2q_u64);

            // Store 16 destination rows of 16 bytes — each row is 16
            // channels at one spatial position.
            vst1q_u8(dst_ptr.add((hw_base) * c + c_base), t0);
            vst1q_u8(dst_ptr.add((hw_base + 1) * c + c_base), t1);
            vst1q_u8(dst_ptr.add((hw_base + 2) * c + c_base), t2);
            vst1q_u8(dst_ptr.add((hw_base + 3) * c + c_base), t3);
            vst1q_u8(dst_ptr.add((hw_base + 4) * c + c_base), t4);
            vst1q_u8(dst_ptr.add((hw_base + 5) * c + c_base), t5);
            vst1q_u8(dst_ptr.add((hw_base + 6) * c + c_base), t6);
            vst1q_u8(dst_ptr.add((hw_base + 7) * c + c_base), t7);
            vst1q_u8(dst_ptr.add((hw_base + 8) * c + c_base), t8);
            vst1q_u8(dst_ptr.add((hw_base + 9) * c + c_base), t9);
            vst1q_u8(dst_ptr.add((hw_base + 10) * c + c_base), t10);
            vst1q_u8(dst_ptr.add((hw_base + 11) * c + c_base), t11);
            vst1q_u8(dst_ptr.add((hw_base + 12) * c + c_base), t12);
            vst1q_u8(dst_ptr.add((hw_base + 13) * c + c_base), t13);
            vst1q_u8(dst_ptr.add((hw_base + 14) * c + c_base), t14);
            vst1q_u8(dst_ptr.add((hw_base + 15) * c + c_base), t15);
        }

        // Right edge: hw % 16 columns. Scalar tail walks the unfilled
        // spatial positions, each touching 16 channels at this c-tile.
        let hw_tail_start = n_tiles_hw * 16;
        for hw_idx in hw_tail_start..hw {
            for ci in 0..16 {
                let src_idx = (c_base + ci) * hw + hw_idx;
                let dst_idx = hw_idx * c + (c_base + ci);
                *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx);
            }
        }
    }

    // Bottom edge: c % 16 channel rows. Scalar tail covers all spatial
    // positions for any unaligned trailing channel block.
    let c_tail_start = n_tiles_c * 16;
    for hw_idx in 0..hw {
        for ci in c_tail_start..c {
            let src_idx = ci * hw + hw_idx;
            let dst_idx = hw_idx * c + ci;
            *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-checked small case: NCHW `[1, 2, 2, 3]` → NHWC `[1, 2, 3, 2]`.
    /// Source layout (channel-major): chan0[(0,0), (0,1), (0,2), (1,0), …]
    /// chan1[(0,0), (0,1), (0,2), (1,0), …].
    #[test]
    fn nchw_to_nhwc_small_hand_check() {
        // h=2, w=3, c=2 → 12 elements.
        // NCHW src (channel-major):
        //   chan0: 0  1  2 | 3  4  5
        //   chan1: 6  7  8 | 9 10 11
        let src: [i32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        // Expected NHWC dst (anchor-major, then channel):
        //   row 0: (c0,c1) = (0,6), (1,7), (2,8)
        //   row 1: (c0,c1) = (3,9), (4,10), (5,11)
        let expected: [i32; 12] = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11];
        let mut dst = [0i32; 12];
        nchw_to_nhwc(&src, 2, 3, 2, &mut dst);
        assert_eq!(dst, expected);
    }

    /// Roundtrip: NCHW → NHWC → NCHW recovers the original.
    /// Verifies the inverse exists (we don't ship the inverse but the
    /// algebra confirms the forward transpose is well-defined).
    #[test]
    fn nchw_to_nhwc_roundtrip_random() {
        let h = 5;
        let w = 7;
        let c = 4;
        let n = h * w * c;
        let src: Vec<i16> = (0..n as i16).collect();
        let mut nhwc = vec![0i16; n];
        nchw_to_nhwc(&src, h, w, c, &mut nhwc);
        // Manual inverse: NHWC → NCHW.
        let mut back = vec![0i16; n];
        let hw = h * w;
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let nhwc_idx = (hi * w + wi) * c + ci;
                    let nchw_idx = ci * hw + hi * w + wi;
                    back[nchw_idx] = nhwc[nhwc_idx];
                }
            }
        }
        assert_eq!(back, src);
    }

    /// Single-channel NCHW vs NHWC are bit-equal (the layout collapses).
    #[test]
    fn nchw_to_nhwc_single_channel_passthrough() {
        let h = 4;
        let w = 5;
        let c = 1;
        let src: Vec<u8> = (0..(h * w) as u8).collect();
        let mut dst = vec![0u8; h * w];
        nchw_to_nhwc(&src, h, w, c, &mut dst);
        assert_eq!(dst, src);
    }

    /// Single-pixel NCHW vs NHWC are bit-equal (h=w=1).
    #[test]
    fn nchw_to_nhwc_single_pixel_passthrough() {
        let src: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = [0f32; 8];
        nchw_to_nhwc(&src, 1, 1, 8, &mut dst);
        assert_eq!(dst, src);
    }

    /// Bit-exact parity between NEON tile transpose and the scalar
    /// reference. Generic `nchw_to_nhwc::<u8>` dispatches to the NEON
    /// kernel on aarch64 (when c >= 16 and hw >= 16), so we run both
    /// the scalar reference and the same dispatch and compare.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nchw_to_nhwc_u8_neon_matches_scalar_aligned_16_16() {
        // Exact 16×16 case: one full tile, no edges.
        let h = 4;
        let w = 4;
        let c = 16;
        let n = h * w * c;
        // Use a unique value per (channel, spatial) so any swap shows up.
        let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut dst_neon = vec![0u8; n];
        let mut dst_scalar = vec![0u8; n];
        super::nchw_to_nhwc(&src, h, w, c, &mut dst_neon);
        // Force scalar by inlining the loop here (mirrors the generic
        // fallback after the cfg block).
        let hw = h * w;
        let mut idx = 0;
        for hi in 0..h {
            for wi in 0..w {
                let base = hi * w + wi;
                for ci in 0..c {
                    dst_scalar[idx] = src[ci * hw + base];
                    idx += 1;
                }
            }
        }
        assert_eq!(dst_neon, dst_scalar, "NEON 16x16 tile mismatched scalar");
    }

    /// NEON path with a tail in BOTH directions: c=20 (one 16-tile +
    /// 4-channel tail), hw=20 (one 16-tile + 4-spatial tail). Exercises
    /// every code path in the kernel.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nchw_to_nhwc_u8_neon_matches_scalar_with_both_tails() {
        let h = 5;
        let w = 4;
        let c = 20;
        let n = h * w * c;
        let src: Vec<u8> = (0..n).map(|i| ((i * 37) % 251) as u8).collect();
        let mut dst_neon = vec![0u8; n];
        let mut dst_scalar = vec![0u8; n];
        super::nchw_to_nhwc(&src, h, w, c, &mut dst_neon);
        let hw = h * w;
        let mut idx = 0;
        for hi in 0..h {
            for wi in 0..w {
                let base = hi * w + wi;
                for ci in 0..c {
                    dst_scalar[idx] = src[ci * hw + base];
                    idx += 1;
                }
            }
        }
        assert_eq!(dst_neon, dst_scalar);
    }

    /// Realistic per-scale shape: c=80 (score), h*w=400 (level 2's
    /// 20×20 grid). Exercises 5 channel tiles × 25 spatial tiles.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nchw_to_nhwc_u8_neon_realistic_score_shape() {
        let h = 20;
        let w = 20;
        let c = 80;
        let n = h * w * c;
        let src: Vec<u8> = (0..n).map(|i| ((i ^ (i >> 7)) % 251) as u8).collect();
        let mut dst_neon = vec![0u8; n];
        let mut dst_scalar = vec![0u8; n];
        super::nchw_to_nhwc(&src, h, w, c, &mut dst_neon);
        let hw = h * w;
        let mut idx = 0;
        for hi in 0..h {
            for wi in 0..w {
                let base = hi * w + wi;
                for ci in 0..c {
                    dst_scalar[idx] = src[ci * hw + base];
                    idx += 1;
                }
            }
        }
        assert_eq!(dst_neon, dst_scalar);
    }

    /// i8 inputs go through the same NEON byte path (size_of::<i8>() == 1).
    /// Bit-pattern preservation is what we rely on; this confirms it
    /// holds across the full signed range including negatives.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nchw_to_nhwc_i8_neon_matches_scalar() {
        let h = 4;
        let w = 4;
        let c = 32;
        let n = h * w * c;
        let src: Vec<i8> = (0..n).map(|i| (i as i32 - 100) as i8).collect();
        let mut dst_neon = vec![0i8; n];
        let mut dst_scalar = vec![0i8; n];
        super::nchw_to_nhwc(&src, h, w, c, &mut dst_neon);
        let hw = h * w;
        let mut idx = 0;
        for hi in 0..h {
            for wi in 0..w {
                let base = hi * w + wi;
                for ci in 0..c {
                    dst_scalar[idx] = src[ci * hw + base];
                    idx += 1;
                }
            }
        }
        assert_eq!(dst_neon, dst_scalar);
    }
}
