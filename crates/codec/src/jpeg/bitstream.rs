// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! 64-bit big-endian bit buffer with a branch-light bulk refill and FF/00
//! byte-stuffing handling, engineered for the priority cores: in-order
//! Cortex-A53/A55 and out-of-order Cortex-A7x / Apple Silicon.
//!
//! Hot-path contract (libjpeg-turbo `BITREAD_*` in spirit):
//!
//! - [`BitStream::refill`] guarantees **≥ 32 buffered bits** whenever more
//!   entropy data exists — enough for one Huffman code (≤ 16 bits) plus its
//!   magnitude bits (≤ 11) — so the block decoder performs a single refill
//!   check per coefficient instead of one per `peek`.
//! - [`BitStream::peek`] / [`BitStream::consume`] are pure register ops:
//!   no memory access, no fill branch. This only holds while nothing hands a
//!   `&mut BitStream` to an out-of-line callee — see [`BitStream::refill_slow`]
//!   for why both cold paths are inlined instead of marked `#[cold]`.
//! - The fast refill is one unaligned 8-byte GPR load (cheap on every target
//!   core, including Cortex-A53's 64-bit load path) plus a SWAR scan for
//!   0xFF. Stuffing/markers divert to a cold byte-wise path.
//!
//! At end-of-data the buffer zero-pads (turbo inserts dummy zeroes the same
//! way), so a truncated stream decodes to garbage rather than reading out of
//! bounds. Block-level validation catches most of that garbage, but not all of
//! it: zero is a valid DC code and a valid EOB, so a scan that simply stops
//! decodes as well-formed zero blocks. [`BitStream::overran`] records that the
//! decoder was handed bits the file never contained, which is what lets the
//! scan loop tell "ran out of data" from "decoded to zeroes".

/// All-ones in every byte lane; used by the SWAR 0xFF detector.
const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

/// True when any byte of `w` equals 0xFF (stuffing or marker sentinel).
#[inline(always)]
fn any_byte_ff(w: u64) -> bool {
    // Standard zero-byte test applied to the complement: byte == 0xFF ⇔
    // complement byte == 0x00.
    let x = !w;
    (x.wrapping_sub(LO)) & !x & HI != 0
}

/// Bit-stream reader for JPEG entropy-coded data.
///
/// Bits are buffered top-aligned in a u64 accumulator. Handles JPEG's
/// byte-stuffing convention (0xFF 0x00 → single 0xFF data byte).
pub struct BitStream<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bits buffered in the HIGH bits of the u64; low bits are zero.
    bits: u64,
    /// Number of valid top bits (may briefly go ≤ 0 on truncated streams,
    /// where the zero-padded low bits stand in for missing data).
    avail: i32,
    /// Entropy prefetch distance in bytes (0 disables).
    prefetch: usize,
    /// Set once the buffer has handed out bits past the end of the entropy
    /// data. Only ever written on the exhausted branch of the cold refill.
    overran: bool,
}

impl<'a> BitStream<'a> {
    /// Create a new bit stream starting at `offset` within `data`.
    #[cfg(test)]
    pub fn new(data: &'a [u8], offset: usize) -> Self {
        Self::with_prefetch(data, offset, 192)
    }

    /// Create with an explicit entropy prefetch distance (0 disables).
    pub fn with_prefetch(data: &'a [u8], offset: usize, prefetch: usize) -> Self {
        let mut bs = Self {
            data,
            pos: offset,
            bits: 0,
            avail: 0,
            prefetch,
            overran: false,
        };
        bs.refill();
        bs
    }

    /// Current byte position in the underlying data.
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    fn prefetch_ahead(&self) {
        if self.prefetch == 0 {
            return;
        }
        let idx = self.pos + self.prefetch;
        if idx < self.data.len() {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: idx is in-bounds; prfm is a hint (dual-issues free on
            // A53/A55).
            unsafe {
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) self.data.as_ptr().add(idx),
                    options(nostack, preserves_flags, readonly)
                );
            }
            #[cfg(target_arch = "x86_64")]
            // SAFETY: idx is in-bounds; prefetch is a hint.
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    self.data.as_ptr().add(idx) as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }

    /// Top up the buffer to **≥ 32 bits** (when data remains).
    ///
    /// Fast path: one unaligned 8-byte load + SWAR 0xFF scan, then append as
    /// many whole bytes as fit. Runs at most once per decoded coefficient.
    #[inline(always)]
    pub fn refill(&mut self) {
        if self.avail >= 32 {
            return;
        }
        if self.pos + 8 <= self.data.len() {
            // SAFETY: bounds checked above; entropy data has no alignment.
            let w = u64::from_be(unsafe {
                (self.data.as_ptr().add(self.pos) as *const u64).read_unaligned()
            });
            if !any_byte_ff(w) {
                let avail = self.avail; // 0..=31 here
                let take = (64 - avail) & !7; // whole bytes only, 40..=64 bits
                                              // Keep exactly `take` top bits of `w`, placed right below the
                                              // `avail` bits already buffered.
                self.bits |= (w >> (64 - take)) << (64 - avail - take);
                self.pos += (take >> 3) as usize;
                self.avail = avail + take;
                self.prefetch_ahead();
                return;
            }
        }
        self.refill_slow();
    }

    /// Byte-wise refill handling 0xFF stuffing, markers, and end-of-data.
    ///
    /// Inlined despite being the rare path, and despite its size. Out of line
    /// it takes `&mut BitStream`, which escapes the pointer and forces the
    /// coefficient loop to keep `bits`/`avail` in memory: `perf annotate`
    /// showed a load *and* a store against the stack slot on every `consume`.
    /// Inlining it lets the whole buffer stay in registers, worth ~6% of
    /// decode time on Rocket Lake for ~3% more retired instructions.
    #[inline(always)]
    fn refill_slow(&mut self) {
        if self.avail < 0 {
            // Negative means a consumer was already handed more bits than the
            // file held: the padding zeroes stood in for them. Reachable only
            // at true end-of-data — while whole bytes remain, a refill tops up
            // to 32 bits and no single symbol consumes that many.
            self.overran = true;
            self.avail = 0;
        }
        while self.avail <= 56 {
            if self.pos >= self.data.len() {
                break;
            }
            let b = self.data[self.pos];
            if b == 0xFF {
                if self.pos + 1 < self.data.len() && self.data[self.pos + 1] == 0x00 {
                    self.pos += 2; // stuffed 0xFF data byte
                } else {
                    break; // marker (or trailing 0xFF): stop before it
                }
            } else {
                self.pos += 1;
            }
            self.bits |= (b as u64) << (56 - self.avail);
            self.avail += 8;
        }
    }

    /// Whether the stream was ever asked for bits past the end of the entropy
    /// data, i.e. the scan is truncated and part of the image is zero padding
    /// rather than decoded content.
    #[inline]
    pub fn overran(&self) -> bool {
        self.overran
    }

    /// Peek at the top `n` bits without consuming them (n ≤ 56).
    ///
    /// Does **not** refill — the caller must uphold the [`refill`] contract.
    ///
    /// [`refill`]: Self::refill
    #[inline(always)]
    pub fn peek(&self, n: u8) -> u32 {
        debug_assert!((1..=56).contains(&n));
        (self.bits >> (64 - n as u32)) as u32
    }

    /// Consume (drop) `n` bits from the buffer. Pure register ops.
    #[inline(always)]
    pub fn consume(&mut self, n: u8) {
        self.bits <<= n;
        self.avail -= n as i32;
    }

    /// Read `n` buffered bits right-aligned. No refill — see [`Self::refill`].
    #[inline(always)]
    pub fn get_bits(&mut self, n: u8) -> u32 {
        let val = self.peek(n);
        self.consume(n);
        val
    }

    /// Read `n` bits with an internal refill (test helper).
    #[cfg(test)]
    pub fn read_bits(&mut self, n: u8) -> u32 {
        self.refill();
        self.get_bits(n)
    }

    /// JPEG sign-extension (libjpeg `HUFF_EXTEND`) — branchless.
    #[inline(always)]
    pub fn extend(val: u32, nbits: u8) -> i32 {
        debug_assert!((1..=15).contains(&nbits));
        let vt = val as i32;
        let m = 1i32 << (nbits - 1);
        // if vt < m { vt - ((1<<nbits)-1) } else { vt }
        vt + (((vt - m) >> 31) & (((-1i32) << nbits) + 1))
    }

    /// Drop remaining sub-byte bits (restart boundary).
    pub fn align_to_byte(&mut self) {
        self.bits = 0;
        self.avail = 0;
    }

    /// Skip past the next restart marker (0xFF 0xDn).
    ///
    /// Buffered bits never contain marker bytes (both refill paths stop at
    /// 0xFF-non-stuffing), so scanning forward from `pos` is sound.
    pub fn skip_restart_marker(&mut self) -> bool {
        self.align_to_byte();

        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos += 2;
                    self.refill();
                    return true;
                } else if m == 0x00 {
                    self.pos += 2;
                    continue;
                } else {
                    break;
                }
            }
            self.pos += 1;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bits_basic() {
        let data = [0b1010_1100, 0b0011_0000];
        let mut bs = BitStream::new(&data, 0);
        assert_eq!(bs.read_bits(4), 0b1010);
        assert_eq!(bs.read_bits(4), 0b1100);
        assert_eq!(bs.read_bits(4), 0b0011);
    }

    #[test]
    fn byte_stuffing() {
        let data = [0xFF, 0x00, 0xAB];
        let mut bs = BitStream::new(&data, 0);
        assert_eq!(bs.read_bits(8), 0xFF);
        assert_eq!(bs.read_bits(8), 0xAB);
    }

    #[test]
    fn extend_positive() {
        assert_eq!(BitStream::extend(5, 3), 5);
        assert_eq!(BitStream::extend(7, 3), 7);
    }

    #[test]
    fn extend_negative() {
        assert_eq!(BitStream::extend(0, 3), -7);
        assert_eq!(BitStream::extend(1, 3), -6);
        assert_eq!(BitStream::extend(3, 3), -4);
    }

    #[test]
    fn extend_one_bit() {
        assert_eq!(BitStream::extend(0, 1), -1);
        assert_eq!(BitStream::extend(1, 1), 1);
    }

    #[test]
    fn bulk_refill_allows_high_bit_non_ff() {
        let data = [0x80, 0x81, 0x7E, 0x7F, 0x01, 0x02, 0x03, 0x04];
        let mut bs = BitStream::new(&data, 0);
        assert!(bs.avail >= 32, "expected bulk refill, avail={}", bs.avail);
        assert_eq!(bs.read_bits(8), 0x80);
        assert_eq!(bs.read_bits(8), 0x81);
        assert_eq!(bs.read_bits(8), 0x7E);
        assert_eq!(bs.read_bits(8), 0x7F);
    }

    #[test]
    fn bulk_refill_aborts_on_ff_stuffing() {
        let data = [0x11, 0xFF, 0x00, 0x22, 0x33, 0x44, 0x55, 0x66];
        let mut bs = BitStream::new(&data, 0);
        assert_eq!(bs.read_bits(8), 0x11);
        assert_eq!(bs.read_bits(8), 0xFF);
        assert_eq!(bs.read_bits(8), 0x22);
    }

    #[test]
    fn swar_ff_detector() {
        assert!(!any_byte_ff(0x1122_3344_5566_7788));
        assert!(!any_byte_ff(0x0000_0000_0000_0000));
        assert!(!any_byte_ff(0x80FE_7F01_ECAB_1002));
        for lane in 0..8 {
            let w = 0x1122_3344_5566_7788u64 & !(0xFFu64 << (lane * 8)) | (0xFFu64 << (lane * 8));
            assert!(any_byte_ff(w), "lane {lane}");
        }
    }

    /// Randomized parity vs a trivial bit-at-a-time reference reader, with
    /// partial-byte refills exercised (avail lands on non-multiples of 8).
    #[test]
    fn refill_parity_random_reads() {
        // Deterministic LCG; bytes 0x00..=0xFE (no 0xFF → pure fast path).
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let data: Vec<u8> = (0..4096).map(|_| (next() % 0xFF) as u8).collect();

        let mut bs = BitStream::new(&data, 0);
        let mut bitpos = 0usize; // reference cursor
        for _ in 0..2000 {
            let n = (next() % 16 + 1) as u8; // 1..=16
            if bitpos + n as usize > data.len() * 8 {
                break;
            }
            let mut expect = 0u32;
            for _ in 0..n {
                let byte = data[bitpos / 8];
                let bit = (byte >> (7 - (bitpos % 8))) & 1;
                expect = (expect << 1) | bit as u32;
                bitpos += 1;
            }
            assert_eq!(bs.read_bits(n), expect, "at bitpos {bitpos} n={n}");
        }
    }

    /// Same parity check across a stream salted with FF00 stuffing (slow path
    /// interleaves with the bulk path).
    #[test]
    fn refill_parity_with_stuffing() {
        let mut raw = Vec::new(); // logical (unstuffed) bytes
        let mut data = Vec::new(); // physical stream
        let mut state = 0x9E37_79B9u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for i in 0..2048 {
            let b = if i % 37 == 0 {
                0xFF
            } else {
                (next() % 0xFF) as u8
            };
            raw.push(b);
            data.push(b);
            if b == 0xFF {
                data.push(0x00);
            }
        }

        let mut bs = BitStream::new(&data, 0);
        let mut bitpos = 0usize;
        loop {
            let n = (next() % 12 + 1) as u8;
            if bitpos + n as usize > raw.len() * 8 - 64 {
                break;
            }
            let mut expect = 0u32;
            for _ in 0..n {
                let byte = raw[bitpos / 8];
                let bit = (byte >> (7 - (bitpos % 8))) & 1;
                expect = (expect << 1) | bit as u32;
                bitpos += 1;
            }
            assert_eq!(bs.read_bits(n), expect, "at bitpos {bitpos} n={n}");
        }
    }
}
