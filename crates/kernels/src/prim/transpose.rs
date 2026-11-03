// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NCHW → NHWC transpose of one image, for NCHW inputs to NHWC kernels.

mod sealed {
    pub trait Sealed {}
}

/// An element [`nchw_to_nhwc`] moves: a plain 1-, 2- or 4-byte number with no padding, so its
/// bits can be moved as an integer lane. Sealed.
pub trait TransposeElement: Copy + sealed::Sealed {
    /// Size in bytes: 1, 2 or 4.
    #[doc(hidden)]
    const BYTES: usize;
}

macro_rules! transpose_element {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl TransposeElement for $t {
            const BYTES: usize = std::mem::size_of::<$t>();
        }
    )*};
}
transpose_element!(u8, i8, u16, i16, half::f16, u32, i32, f32);

/// NCHW → NHWC transpose of one image: `src` is `[c, h, w]` and `dst` is written as `[h, w, c]`.
///
/// Both slices must hold exactly `h * w * c` elements; otherwise nothing is written and `dst`
/// is left unchanged. Never panics. The transpose is a pure permutation: every element's bits
/// move unchanged, NaN payloads included.
///
/// When `c` or `h * w` is 1 the layouts coincide and `src` is copied. On aarch64 the rest runs
/// NEON: `ST2`/`ST3`/`ST4` interleaving for `c` of 2 to 4, and square register tiles (16×16
/// bytes, 8×8 halfwords, 4×4 words) otherwise, with edge tiles overlapped or staged through
/// the stack so no shape falls back to an element loop. Other targets walk the destination in
/// order.
#[inline]
pub fn nchw_to_nhwc<T: TransposeElement>(src: &[T], h: usize, w: usize, c: usize, dst: &mut [T]) {
    // The NEON paths read and write unchecked, so both lengths must match the shape.
    let n = h.checked_mul(w).and_then(|hw| hw.checked_mul(c));
    if n != Some(src.len()) || n != Some(dst.len()) || src.is_empty() {
        return;
    }
    let hw = h * w;
    let route = route(hw, c);
    if route == Route::Copy {
        dst.copy_from_slice(src);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `TransposeElement` types are padding-free `T::BYTES`-byte numbers aligned to their size, and both slices hold `hw * c` elements with `hw, c >= 2`.
    unsafe {
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        match T::BYTES {
            1 => neon::run::<u8>(s.cast(), hw, c, d.cast(), route),
            2 => neon::run::<u16>(s.cast(), hw, c, d.cast(), route),
            _ => neon::run::<u32>(s.cast(), hw, c, d.cast(), route),
        }
    };
    #[cfg(not(target_arch = "aarch64"))]
    nchw_to_nhwc_scalar(src, h, w, c, dst);
}

/// [`nchw_to_nhwc`] as the element loop on every target, for a forced scalar tier: walks the
/// destination in order; writes nothing unless both slices hold `h * w * c` elements.
#[doc(hidden)]
#[inline]
pub fn nchw_to_nhwc_scalar<T: Copy>(src: &[T], h: usize, w: usize, c: usize, dst: &mut [T]) {
    let n = h.checked_mul(w).and_then(|hw| hw.checked_mul(c));
    if n != Some(src.len()) || n != Some(dst.len()) {
        return;
    }
    let hw = h * w;
    for (p, out) in dst.chunks_exact_mut(c.max(1)).enumerate() {
        for (ci, o) in out.iter_mut().enumerate() {
            if let Some(&v) = src.get(ci * hw + p) {
                *o = v;
            }
        }
    }
}

/// How a transpose runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// `c` or `h * w` is 1: the layouts coincide.
    Copy,
    /// `ST2`/`ST3`/`ST4` interleaving of 2 to 4 channel planes (aarch64).
    Interleave,
    /// Square register tiles (aarch64).
    Tile,
}

/// The route for a non-empty shape; compiled on every target so the rule is tested everywhere.
const fn route(hw: usize, c: usize) -> Route {
    if c == 1 || hw == 1 {
        Route::Copy
    } else if c <= 4 {
        Route::Interleave
    } else {
        Route::Tile
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Route;
    use core::arch::aarch64::*; // NOSONAR
    use core::ptr::copy_nonoverlapping;

    /// An element moved as a NEON lane.
    pub(super) trait Lane: Copy {
        /// Lanes per 128-bit vector, the tile side.
        const L: usize;

        /// Whether full tiles at this shape run [`position_outer`] rather than [`channel_outer`].
        fn position_outer(hw: usize, c: usize) -> bool;

        /// Full tiles channel tile by channel tile ([`channel_outer`] or a faster walk).
        ///
        /// # Safety
        /// As [`channel_outer`].
        unsafe fn channel_tiles(src: *const Self, hw: usize, c: usize, dst: *mut Self);

        /// Transposes an `L`×`L` tile: row `r` of `src` (at `src + r * ss`) becomes column `r`
        /// of `dst`, whose rows are at `dst + j * ds` and are stored in increasing `j`.
        ///
        /// # Safety
        /// Every addressed row must be valid for `L` elements.
        unsafe fn tile(src: *const Self, ss: usize, dst: *mut Self, ds: usize);

        /// Interleaves `L` positions of `C` channel planes `hw` apart into `L * C` elements.
        ///
        /// # Safety
        /// `C` is 2, 3 or 4; the planes and `dst` must be valid for those elements.
        unsafe fn interleave<const C: usize>(src: *const Self, hw: usize, dst: *mut Self);

        /// [`Lane::interleave`] of the last `cols` (< `L`) positions, through stack blocks.
        ///
        /// # Safety
        /// As [`Lane::interleave`], for `cols` positions.
        unsafe fn interleave_tail<const C: usize>(
            src: *const Self,
            hw: usize,
            cols: usize,
            dst: *mut Self,
        );

        /// Transposes `[c, hw]` to `[hw, c]` for `c < L <= hw`: blocks of `L` positions are
        /// tiled into a stack block and copied out as `L * c` contiguous elements.
        ///
        /// # Safety
        /// As [`run`], with `2 <= c < L <= hw`.
        unsafe fn few_channels(src: *const Self, hw: usize, c: usize, dst: *mut Self);

        /// [`Lane::tile`] of the `rows`×`cols` corner of a tile, through stack tiles.
        ///
        /// # Safety
        /// `rows` and `cols` are in `1..=L`; `src` rows must be valid for `cols` elements and
        /// `dst` rows for `rows`.
        unsafe fn partial_tile(
            src: *const Self,
            ss: usize,
            rows: usize,
            cols: usize,
            dst: *mut Self,
            ds: usize,
        );
    }

    /// Transposes `[c, hw]` at `src` to `[hw, c]` at `dst` by `route`.
    ///
    /// # Safety
    /// `src` and `dst` are valid, non-overlapping and hold `hw * c` elements; `hw >= 2` and
    /// `c >= 2`, and `c <= 4` for [`Route::Interleave`].
    #[inline]
    pub(super) unsafe fn run<E: Lane>(
        src: *const E,
        hw: usize,
        c: usize,
        dst: *mut E,
        route: Route,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe {
            match (route, c) {
                (Route::Interleave, 2) => interleave::<E, 2>(src, hw, dst),
                (Route::Interleave, 3) => interleave::<E, 3>(src, hw, dst),
                (Route::Interleave, _) => interleave::<E, 4>(src, hw, dst),
                _ => tiles(src, hw, c, dst),
            }
        }
    }

    /// # Safety
    /// As [`run`] with `c == C`.
    #[inline(always)]
    unsafe fn interleave<E: Lane, const C: usize>(src: *const E, hw: usize, dst: *mut E) {
        let full = hw - hw % E::L;
        let mut p = 0;
        // SAFETY: positions `p..p + L` (or the tail) lie inside every plane and `dst`.
        unsafe {
            while p < full {
                E::interleave::<C>(src.add(p), hw, dst.add(p * C));
                p += E::L;
            }
            if p < hw {
                E::interleave_tail::<C>(src.add(p), hw, hw - p, dst.add(p * C));
            }
        }
    }

    /// Full tiles when both dimensions reach `L`, stack-staged blocks otherwise.
    ///
    /// # Safety
    /// As [`run`].
    #[inline(always)]
    unsafe fn tiles<E: Lane>(src: *const E, hw: usize, c: usize, dst: *mut E) {
        let l = E::L;
        // SAFETY: every tile addresses channels below `c` and positions below `hw` only.
        unsafe {
            if hw >= l && c >= l {
                if E::position_outer(hw, c) {
                    position_outer(src, hw, c, dst);
                } else {
                    E::channel_tiles(src, hw, c, dst);
                }
            } else if hw >= l {
                E::few_channels(src, hw, c, dst);
            } else if c >= l {
                let mut c0 = 0;
                loop {
                    let ch = c0.min(c - l);
                    E::partial_tile(src.add(ch * hw), hw, l, hw, dst.add(ch), c);
                    if ch == c - l {
                        break;
                    }
                    c0 += l;
                }
            } else {
                E::partial_tile(src, hw, c, hw, dst, c);
            }
        }
    }

    /// Full tiles, channel tile by channel tile, each walking every position block; the last
    /// tile of a dimension that is not a multiple of `L` overlaps the one before and rewrites
    /// the same values.
    ///
    /// # Safety
    /// As [`run`], with `hw >= L` and `c >= L`.
    #[inline(always)]
    unsafe fn channel_outer<E: Lane>(src: *const E, hw: usize, c: usize, dst: *mut E) {
        // SAFETY: forwarded from the caller.
        unsafe { channel_outer_n(src, hw, c, c, dst) }
    }

    /// [`channel_outer`] over the first `n` (`>= L`) of `c` channels.
    ///
    /// # Safety
    /// As [`channel_outer`], with `L <= n <= c`.
    #[inline(always)]
    unsafe fn channel_outer_n<E: Lane>(src: *const E, hw: usize, n: usize, c: usize, dst: *mut E) {
        let l = E::L;
        let (ct, pt) = (n / l, hw / l);
        // SAFETY: every tile starts at a channel `<= c - L` and a position `<= hw - L`.
        unsafe {
            for k in 0..=ct {
                let ch = if k < ct {
                    k * l
                } else if !n.is_multiple_of(l) {
                    n - l
                } else {
                    break;
                };
                for t in 0..pt {
                    E::tile(src.add(ch * hw + t * l), hw, dst.add(t * l * c + ch), c);
                }
                if !hw.is_multiple_of(l) {
                    let p = hw - l;
                    E::tile(src.add(ch * hw + p), hw, dst.add(p * c + ch), c);
                }
            }
        }
    }

    /// [`channel_outer`] with the loops swapped: each position block through every channel
    /// tile, so the destination is written in order.
    ///
    /// # Safety
    /// As [`channel_outer`].
    #[inline(always)]
    unsafe fn position_outer<E: Lane>(src: *const E, hw: usize, c: usize, dst: *mut E) {
        let l = E::L;
        let mut p0 = 0;
        // SAFETY: every tile starts at a channel `<= c - L` and a position `<= hw - L`.
        unsafe {
            loop {
                let p = p0.min(hw - l);
                let (s, d) = (src.add(p), dst.add(p * c));
                let mut c0 = 0;
                loop {
                    let ch = c0.min(c - l);
                    E::tile(s.add(ch * hw), hw, d.add(ch), c);
                    if ch == c - l {
                        break;
                    }
                    c0 += l;
                }
                if p == hw - l {
                    break;
                }
                p0 += l;
            }
        }
    }

    /// Bytes above which u8 tiles walk positions outermost.
    const U8_POSITION_OUTER_BYTES: usize = 256 * 1024;

    /// Large u8 shapes walk positions outermost, except those whose last one or two channels
    /// [`channel_outer_u8`] stores by lane.
    #[inline(always)]
    fn position_outer_u8(hw: usize, c: usize) -> bool {
        hw.saturating_mul(c) > U8_POSITION_OUTER_BYTES && !matches!(c % 16, 1 | 2)
    }

    #[inline(always)]
    fn channel_outer_only(_hw: usize, _c: usize) -> bool {
        false
    }

    /// u8 channel tiles: the 16×16 walk with its own addressing for tile-aligned shapes,
    /// [`channel_outer`] otherwise.
    ///
    /// # Safety
    /// As [`channel_outer`].
    #[inline(always)]
    unsafe fn channel_outer_u8(src: *const u8, hw: usize, c: usize, dst: *mut u8) {
        // SAFETY: every tile and lane store stays inside the `[c, hw]` and `[hw, c]` buffers.
        unsafe {
            let r = c % 16;
            if (r == 1 || r == 2) && c > 16 {
                channel_outer_n(src, hw, c - r, c, dst);
                let (s, d) = (src.add((c - r) * hw), dst.add(c - r));
                return if r == 1 {
                    last_channel_u8(s, hw, c, d)
                } else {
                    last_two_channels_u8(s, hw, c, d)
                };
            }
            if !c.is_multiple_of(16) || !hw.is_multiple_of(16) {
                return channel_outer(src, hw, c, dst);
            }
            for c_base in (0..c).step_by(16) {
                for hw_base in (0..hw).step_by(16) {
                    tile_u8_at(
                        |k| src.add((c_base + k) * hw + hw_base),
                        |j| dst.add((hw_base + j) * c + c_base),
                    );
                }
            }
        }
    }

    /// One channel plane into column `c - 1`: 16 positions per load, one byte lane per store.
    ///
    /// # Safety
    /// `src` holds `hw` bytes and `dst` rows `0..hw` at stride `c` are writable at offset 0.
    #[inline(never)]
    unsafe fn last_channel_u8(src: *const u8, hw: usize, c: usize, dst: *mut u8) {
        let full = hw - hw % 16;
        // SAFETY: positions below `hw` only.
        unsafe {
            let mut p = 0;
            while p < full {
                let v = vld1q_u8(src.add(p));
                let d = dst.add(p * c);
                macro_rules! lanes {
                    ($($k:literal)*) => {$(vst1q_lane_u8::<$k>(d.add($k * c), v);)*};
                }
                lanes!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                p += 16;
            }
            while p < hw {
                *dst.add(p * c) = *src.add(p);
                p += 1;
            }
        }
    }

    /// Two channel planes into columns `c - 2` and `c - 1`: 16 positions per load pair, one
    /// byte pair per store.
    ///
    /// # Safety
    /// `src` holds two `hw`-byte planes and `dst` rows `0..hw` at stride `c` are writable for
    /// two bytes.
    #[inline(never)]
    unsafe fn last_two_channels_u8(src: *const u8, hw: usize, c: usize, dst: *mut u8) {
        let full = hw - hw % 16;
        // SAFETY: positions below `hw` only; the pair stores are unaligned writes.
        unsafe {
            let mut p = 0;
            while p < full {
                let (a, b) = (vld1q_u8(src.add(p)), vld1q_u8(src.add(hw + p)));
                let (lo, hi) = (
                    vreinterpretq_u16_u8(vzip1q_u8(a, b)),
                    vreinterpretq_u16_u8(vzip2q_u8(a, b)),
                );
                let d = dst.add(p * c);
                macro_rules! pairs {
                    ($v:ident, $base:literal: $($k:literal)*) => {$(
                        d.add(($base + $k) * c).cast::<u16>().write_unaligned(vgetq_lane_u16::<$k>($v));
                    )*};
                }
                pairs!(lo, 0: 0 1 2 3 4 5 6 7);
                pairs!(hi, 8: 0 1 2 3 4 5 6 7);
                p += 16;
            }
            while p < hw {
                *dst.add(p * c) = *src.add(p);
                *dst.add(p * c + 1) = *src.add(hw + p);
                p += 1;
            }
        }
    }

    macro_rules! lane {
        ($t:ty, $l:literal, $pos_outer:ident, $chan:path, $tile:ident, $ld:ident, $st2:ident, $x2:ident,
         $st3:ident, $x3:ident, $st4:ident, $x4:ident) => {
            impl Lane for $t {
                const L: usize = $l;

                #[inline(always)]
                fn position_outer(hw: usize, c: usize) -> bool {
                    $pos_outer(hw, c)
                }

                #[inline(always)]
                unsafe fn channel_tiles(src: *const Self, hw: usize, c: usize, dst: *mut Self) {
                    // SAFETY: forwarded from the caller.
                    unsafe { $chan(src, hw, c, dst) }
                }

                #[inline(always)]
                unsafe fn tile(src: *const Self, ss: usize, dst: *mut Self, ds: usize) {
                    // SAFETY: forwarded from the caller.
                    unsafe { $tile(src, ss, dst, ds) }
                }

                #[inline(always)]
                unsafe fn interleave<const C: usize>(src: *const Self, hw: usize, dst: *mut Self) {
                    // SAFETY: forwarded from the caller.
                    unsafe {
                        let (r0, r1) = ($ld(src), $ld(src.add(hw)));
                        if C == 2 {
                            $st2(dst, $x2(r0, r1));
                        } else if C == 3 {
                            $st3(dst, $x3(r0, r1, $ld(src.add(2 * hw))));
                        } else {
                            let (r2, r3) = ($ld(src.add(2 * hw)), $ld(src.add(3 * hw)));
                            $st4(dst, $x4(r0, r1, r2, r3));
                        }
                    }
                }

                #[inline(never)]
                unsafe fn interleave_tail<const C: usize>(
                    src: *const Self,
                    hw: usize,
                    cols: usize,
                    dst: *mut Self,
                ) {
                    let mut a = [0 as $t; 4 * $l];
                    let mut b = [0 as $t; 4 * $l];
                    // SAFETY: `cols < L` and `C <= 4` keep every access inside `a`, `b` and the caller's.
                    unsafe {
                        for ch in 0..C {
                            copy_nonoverlapping(
                                src.add(ch * hw),
                                a.as_mut_ptr().add(ch * $l),
                                cols,
                            );
                        }
                        Self::interleave::<C>(a.as_ptr(), $l, b.as_mut_ptr());
                        copy_nonoverlapping(b.as_ptr(), dst, cols * C);
                    }
                }

                #[inline(never)]
                unsafe fn few_channels(src: *const Self, hw: usize, c: usize, dst: *mut Self) {
                    // Rows `c..L` of `a` stay zero; their columns are overwritten by the next row or fall past `L * c`.
                    let mut a = [0 as $t; $l * $l];
                    let mut b = [0 as $t; $l * $l + $l];
                    let mut p0 = 0;
                    // SAFETY: `c < L <= hw` keeps every access inside `a`, `b` and the caller's.
                    unsafe {
                        loop {
                            let p = p0.min(hw - $l);
                            for r in 0..c {
                                copy_nonoverlapping(
                                    src.add(r * hw + p),
                                    a.as_mut_ptr().add(r * $l),
                                    $l,
                                );
                            }
                            $tile(a.as_ptr(), $l, b.as_mut_ptr(), c);
                            copy_nonoverlapping(b.as_ptr(), dst.add(p * c), $l * c);
                            if p == hw - $l {
                                break;
                            }
                            p0 += $l;
                        }
                    }
                }

                #[inline(never)]
                unsafe fn partial_tile(
                    src: *const Self,
                    ss: usize,
                    rows: usize,
                    cols: usize,
                    dst: *mut Self,
                    ds: usize,
                ) {
                    let mut a = [0 as $t; $l * $l];
                    let mut b = [0 as $t; $l * $l];
                    // SAFETY: `rows, cols <= L` keep every access inside `a`, `b` and the caller's rows.
                    unsafe {
                        for r in 0..rows {
                            copy_nonoverlapping(src.add(r * ss), a.as_mut_ptr().add(r * $l), cols);
                        }
                        $tile(a.as_ptr(), $l, b.as_mut_ptr(), $l);
                        for j in 0..cols {
                            copy_nonoverlapping(b.as_ptr().add(j * $l), dst.add(j * ds), rows);
                        }
                    }
                }
            }
        };
    }

    lane!(
        u8,
        16,
        position_outer_u8,
        channel_outer_u8,
        tile_u8,
        vld1q_u8,
        vst2q_u8,
        uint8x16x2_t,
        vst3q_u8,
        uint8x16x3_t,
        vst4q_u8,
        uint8x16x4_t
    );
    lane!(
        u16,
        8,
        channel_outer_only,
        channel_outer::<u16>,
        tile_u16,
        vld1q_u16,
        vst2q_u16,
        uint16x8x2_t,
        vst3q_u16,
        uint16x8x3_t,
        vst4q_u16,
        uint16x8x4_t
    );
    lane!(
        u32,
        4,
        channel_outer_only,
        channel_outer::<u32>,
        tile_u32,
        vld1q_u32,
        vst2q_u32,
        uint32x4x2_t,
        vst3q_u32,
        uint32x4x3_t,
        vst4q_u32,
        uint32x4x4_t
    );

    /// 16×16 byte tile: four `TRN1`/`TRN2` stages (`.16b`, `.8h`, `.4s`, `.2d`).
    ///
    /// # Safety
    /// As [`Lane::tile`].
    #[inline(always)]
    unsafe fn tile_u8(src: *const u8, ss: usize, dst: *mut u8, ds: usize) {
        // SAFETY: forwarded from the caller.
        unsafe { tile_u8_at(|k| src.add(k * ss), |j| dst.add(j * ds)) }
    }

    /// [`tile_u8`] with row `k`'s source at `ld(k)` and row `j`'s destination at `st(j)`.
    ///
    /// # Safety
    /// Every `ld(k)` must be valid for reading 16 bytes and every `st(j)` for writing 16.
    #[inline(always)]
    unsafe fn tile_u8_at(ld: impl Fn(usize) -> *const u8, st: impl Fn(usize) -> *mut u8) {
        // SAFETY: forwarded from the caller.
        unsafe {
            // 16 source rows: 16 positions of one channel each.
            let r0 = vld1q_u8(ld(0));
            let r1 = vld1q_u8(ld(1));
            let r2 = vld1q_u8(ld(2));
            let r3 = vld1q_u8(ld(3));
            let r4 = vld1q_u8(ld(4));
            let r5 = vld1q_u8(ld(5));
            let r6 = vld1q_u8(ld(6));
            let r7 = vld1q_u8(ld(7));
            let r8 = vld1q_u8(ld(8));
            let r9 = vld1q_u8(ld(9));
            let r10 = vld1q_u8(ld(10));
            let r11 = vld1q_u8(ld(11));
            let r12 = vld1q_u8(ld(12));
            let r13 = vld1q_u8(ld(13));
            let r14 = vld1q_u8(ld(14));
            let r15 = vld1q_u8(ld(15));

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

            // 16 destination rows: 16 channels of one position each.
            vst1q_u8(st(0), t0);
            vst1q_u8(st(1), t1);
            vst1q_u8(st(2), t2);
            vst1q_u8(st(3), t3);
            vst1q_u8(st(4), t4);
            vst1q_u8(st(5), t5);
            vst1q_u8(st(6), t6);
            vst1q_u8(st(7), t7);
            vst1q_u8(st(8), t8);
            vst1q_u8(st(9), t9);
            vst1q_u8(st(10), t10);
            vst1q_u8(st(11), t11);
            vst1q_u8(st(12), t12);
            vst1q_u8(st(13), t13);
            vst1q_u8(st(14), t14);
            vst1q_u8(st(15), t15);
        }
    }

    /// 8×8 halfword tile: three `TRN1`/`TRN2` stages (`.8h`, `.4s`, `.2d`).
    ///
    /// # Safety
    /// As [`Lane::tile`].
    #[inline(always)]
    unsafe fn tile_u16(src: *const u16, ss: usize, dst: *mut u16, ds: usize) {
        // SAFETY: forwarded from the caller.
        unsafe {
            let r: [uint16x8_t; 8] = core::array::from_fn(|i| vld1q_u16(src.add(i * ss)));
            let a: [uint32x4_t; 8] = core::array::from_fn(|i| {
                let (lo, hi) = (r[i & !1], r[i | 1]);
                vreinterpretq_u32_u16(if i & 1 == 0 {
                    vtrn1q_u16(lo, hi)
                } else {
                    vtrn2q_u16(lo, hi)
                })
            });
            // b[0..4] hold columns {0,4}, {1,5}, {2,6}, {3,7} of rows 0–3; b[4..8] of rows 4–7.
            let b: [uint64x2_t; 8] = core::array::from_fn(|i| {
                let base = (i & 4) + (i & 1);
                let (lo, hi) = (a[base], a[base + 2]);
                vreinterpretq_u64_u32(if i & 2 == 0 {
                    vtrn1q_u32(lo, hi)
                } else {
                    vtrn2q_u32(lo, hi)
                })
            });
            for j in 0..8 {
                let (lo, hi) = (b[j & 3], b[(j & 3) + 4]);
                let t = if j < 4 {
                    vtrn1q_u64(lo, hi)
                } else {
                    vtrn2q_u64(lo, hi)
                };
                vst1q_u16(dst.add(j * ds), vreinterpretq_u16_u64(t));
            }
        }
    }

    /// 4×4 word tile: two `TRN1`/`TRN2` stages (`.4s`, `.2d`).
    ///
    /// # Safety
    /// As [`Lane::tile`].
    #[inline(always)]
    unsafe fn tile_u32(src: *const u32, ss: usize, dst: *mut u32, ds: usize) {
        // SAFETY: forwarded from the caller.
        unsafe {
            let r: [uint32x4_t; 4] = core::array::from_fn(|i| vld1q_u32(src.add(i * ss)));
            let a0 = vreinterpretq_u64_u32(vtrn1q_u32(r[0], r[1]));
            let a1 = vreinterpretq_u64_u32(vtrn2q_u32(r[0], r[1]));
            let a2 = vreinterpretq_u64_u32(vtrn1q_u32(r[2], r[3]));
            let a3 = vreinterpretq_u64_u32(vtrn2q_u32(r[2], r[3]));
            vst1q_u32(dst, vreinterpretq_u32_u64(vtrn1q_u64(a0, a2)));
            vst1q_u32(dst.add(ds), vreinterpretq_u32_u64(vtrn1q_u64(a1, a3)));
            vst1q_u32(dst.add(2 * ds), vreinterpretq_u32_u64(vtrn2q_u64(a0, a2)));
            vst1q_u32(dst.add(3 * ds), vreinterpretq_u32_u64(vtrn2q_u64(a1, a3)));
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

    /// Slices shorter than `h * w * c`, or a shape that overflows, leave `dst` untouched.
    #[test]
    fn nchw_to_nhwc_length_mismatch_writes_nothing() {
        let (h, w, c) = (4, 4, 16);
        let n = h * w * c;
        let src: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let mut dst = vec![0xAA_u8; n];
        nchw_to_nhwc(&src[..n - 1], h, w, c, &mut dst);
        assert!(dst.iter().all(|&b| b == 0xAA));
        let mut short = vec![0xAA_u8; n - 1];
        nchw_to_nhwc(&src, h, w, c, &mut short);
        assert!(short.iter().all(|&b| b == 0xAA));
        let mut empty: [u8; 0] = [];
        nchw_to_nhwc(&[], usize::MAX, 2, 16, &mut empty);
        let mut wide = vec![0xAA_u8; 16];
        nchw_to_nhwc(&src[..16], usize::MAX / 2 + 1, 2, 16, &mut wide);
        assert!(wide.iter().all(|&b| b == 0xAA));
        let src32: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst32 = vec![-1.0_f32; n];
        nchw_to_nhwc(&src32[..n - 1], h, w, c, &mut dst32);
        assert!(dst32.iter().all(|&v| v == -1.0));
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

    fn scalar_reference<T: Copy + Default>(src: &[T], h: usize, w: usize, c: usize) -> Vec<T> {
        let hw = h * w;
        let mut out = vec![T::default(); hw * c];
        for p in 0..hw {
            for ci in 0..c {
                out[p * c + ci] = src[ci * hw + p];
            }
        }
        out
    }

    #[test]
    fn route_by_shape() {
        // (hw, c) -> route
        let table = [
            ((16, 16), Route::Tile),
            ((400, 80), Route::Tile),
            ((7, 5), Route::Tile),
            ((2, 5), Route::Tile),
            ((7, 2), Route::Interleave),
            ((400, 3), Route::Interleave),
            ((17, 4), Route::Interleave),
            ((17, 1), Route::Copy),
            ((1, 80), Route::Copy),
        ];
        for ((hw, c), want) in table {
            assert_eq!(route(hw, c), want, "hw={hw} c={c}");
        }
    }

    /// u8 shapes above the position-outer threshold, aligned and with every channel remainder.
    #[test]
    fn nchw_to_nhwc_u8_large_shapes_match_scalar() {
        for (c, h, w) in [
            (80, 60, 60),
            (81, 60, 60),
            (82, 60, 60),
            (83, 60, 60),
            (96, 47, 61),
        ] {
            let src: Vec<u8> = (0..c * h * w).map(|i| (i * 131 % 251) as u8).collect();
            let mut dst = vec![0u8; src.len()];
            nchw_to_nhwc(&src, h, w, c, &mut dst);
            assert!(dst == scalar_reference(&src, h, w, c), "c={c} h={h} w={w}");
        }
    }

    /// The element loop is the reference for any `Copy` type, padded or wider ones included.
    #[test]
    fn scalar_loop_matches_reference_for_any_copy_type() {
        for (h, w, c) in [(1, 1, 3), (2, 3, 5), (4, 4, 17), (3, 1, 1)] {
            let src: Vec<[u8; 3]> = (0..h * w * c).map(|i| [(i % 256) as u8; 3]).collect();
            let mut dst = vec![[0u8; 3]; src.len()];
            nchw_to_nhwc_scalar(&src, h, w, c, &mut dst);
            assert_eq!(dst, scalar_reference(&src, h, w, c));
            let src: Vec<u64> = (0..h * w * c).map(|i| i as u64).collect();
            let mut dst = vec![0u64; src.len()];
            nchw_to_nhwc_scalar(&src, h, w, c, &mut dst);
            assert_eq!(dst, scalar_reference(&src, h, w, c));
        }
        let mut dst = [9u64; 5];
        nchw_to_nhwc_scalar(&[1u64; 6], 2, 3, 1, &mut dst);
        assert_eq!(dst, [9; 5]);
    }

    /// Every element size, channel count and spatial size against a plain loop, bit for bit.
    #[test]
    fn nchw_to_nhwc_neon_matches_scalar() {
        fn check<T: TransposeElement + Default, B: PartialEq + std::fmt::Debug>(
            make: impl Fn(usize) -> T,
            bits: impl Fn(&T) -> B,
        ) {
            for c in [1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 33, 34, 80] {
                for (h, w) in [(1, 1), (2, 1), (1, 3), (7, 1), (4, 4), (1, 17), (20, 20)] {
                    let src: Vec<T> = (0..h * w * c).map(&make).collect();
                    let mut dst = vec![T::default(); src.len()];
                    nchw_to_nhwc(&src, h, w, c, &mut dst);
                    let want = scalar_reference(&src, h, w, c);
                    let (got, want): (Vec<B>, Vec<B>) = (
                        dst.iter().map(&bits).collect(),
                        want.iter().map(&bits).collect(),
                    );
                    assert_eq!(got, want, "c={c} h={h} w={w}");
                }
            }
        }
        check(|i| (i * 37 % 251) as u8, |&v| v);
        check(|i| (i * 37 % 251) as u8 as i8, |&v| v);
        check(|i| (i * 7919 % 65521) as u16, |&v| v);
        check(|i| half::f16::from_bits((i * 7919) as u16), |v| v.to_bits());
        check(|i| (i as u32).wrapping_mul(2_654_435_761), |&v| v);
        check(|i| (i as i16).wrapping_mul(-7), |&v| v);
        check(|i| (i as i32).wrapping_mul(-1_640_531_527), |&v| v);
        check(|i| f32::from_bits(0x7f80_0001 + i as u32), |v| v.to_bits());
    }

    /// A 4-byte element at a full byte-tile shape (c=16, h*w=16) moves whole elements.
    #[test]
    fn nchw_to_nhwc_f32_at_a_full_tile_shape_moves_whole_elements() {
        let (h, w, c) = (4, 4, 16);
        let hw = h * w;
        let src: Vec<f32> = (0..hw * c).map(|i| i as f32).collect();
        let mut dst = vec![0f32; hw * c];
        nchw_to_nhwc(&src, h, w, c, &mut dst);
        for p in 0..hw {
            for ci in 0..c {
                assert_eq!(dst[p * c + ci], (ci * hw + p) as f32, "p={p} ci={ci}");
            }
        }
    }

    /// Sixteen channels but only fifteen spatial positions: not a full tile,
    /// so the result must still be a correct transpose.
    #[test]
    fn nchw_to_nhwc_u8_with_fewer_than_a_tile_of_positions() {
        let (h, w, c) = (1, 15, 16);
        let src: Vec<u8> = (0..h * w * c).map(|i| (i % 251) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        nchw_to_nhwc(&src, h, w, c, &mut dst);
        assert_eq!(dst, scalar_reference(&src, h, w, c));
    }

    /// Single-pixel NCHW vs NHWC are bit-equal (h=w=1).
    #[test]
    fn nchw_to_nhwc_single_pixel_passthrough() {
        let src: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = [0f32; 8];
        nchw_to_nhwc(&src, 1, 1, 8, &mut dst);
        assert_eq!(dst, src);
    }

    /// Bit-exact parity between the NEON 16×16 byte tile and a plain loop.
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
