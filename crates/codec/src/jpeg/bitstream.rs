// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! 64-bit bit buffer with bulk refill and FF/00 byte-stuffing handling.

/// Bit-stream reader for JPEG entropy-coded data.
///
/// Uses a 64-bit accumulator filled in big-endian order. Handles JPEG's
/// byte-stuffing convention (0xFF followed by 0x00 → single 0xFF data byte).
pub struct BitStream<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bits buffered in the high bits of the u64.
    bits: u64,
    /// Number of valid bits currently in `bits` (0–64).
    available: u8,
}

impl<'a> BitStream<'a> {
    /// Create a new bit stream starting at `offset` within `data`.
    pub fn new(data: &'a [u8], offset: usize) -> Self {
        let mut bs = Self {
            data,
            pos: offset,
            bits: 0,
            available: 0,
        };
        bs.fill();
        bs
    }

    /// Current byte position in the underlying data.
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Ensure at least 32 bits are available by reading up to 8 bytes.
    ///
    /// Uses a fast path that batches non-FF bytes. When the remaining data
    /// contains no 0xFF markers (the common case), multiple bytes are loaded
    /// without per-byte branching.
    #[inline]
    pub fn fill(&mut self) {
        // Fast path: if we need ≥ 4 bytes and the next 4 have no 0xFF,
        // bulk-load them without per-byte checks.
        while self.available <= 32 && self.pos + 3 < self.data.len() {
            let b0 = self.data[self.pos];
            let b1 = self.data[self.pos + 1];
            let b2 = self.data[self.pos + 2];
            let b3 = self.data[self.pos + 3];
            if (b0 | b1 | b2 | b3) & 0x80 != 0 {
                // At least one byte has the high bit set — could be 0xFF.
                // Fall through to byte-at-a-time.
                break;
            }
            // No byte is ≥ 0x80, so none can be 0xFF. Bulk-load all four.
            let word = (b0 as u64) << 24 | (b1 as u64) << 16 | (b2 as u64) << 8 | (b3 as u64);
            self.bits |= word << (32 - self.available);
            self.available += 32;
            self.pos += 4;
        }

        // Byte-at-a-time path for remaining fills (handles 0xFF stuffing)
        while self.available <= 56 && self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;

            if b == 0xFF {
                if self.pos < self.data.len() {
                    if self.data[self.pos] == 0x00 {
                        self.pos += 1;
                        self.bits |= (0xFF_u64) << (56 - self.available);
                        self.available += 8;
                    } else {
                        self.pos -= 1;
                        break;
                    }
                } else {
                    break;
                }
            } else {
                self.bits |= (b as u64) << (56 - self.available);
                self.available += 8;
            }
        }
    }

    /// Peek at the top `n` bits without consuming them (n ≤ 56).
    #[inline]
    pub fn peek(&mut self, n: u8) -> u32 {
        debug_assert!(n <= 56);
        if self.available < n {
            self.fill();
        }
        (self.bits >> (64 - n as u32)) as u32
    }

    /// Consume (drop) `n` bits from the buffer.
    #[inline]
    pub fn consume(&mut self, n: u8) {
        debug_assert!(n <= self.available);
        self.bits <<= n;
        self.available -= n;
    }

    /// Read exactly `n` bits and return them right-aligned.
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> u32 {
        let val = self.peek(n);
        self.consume(n);
        val
    }

    /// JPEG sign-extension: if the top bit of a value of length `n` is 0,
    /// the value is negative and needs to be offset.
    ///
    /// Equivalent to libjpeg's `HUFF_EXTEND(v, n)`.
    #[inline]
    pub fn extend(val: u32, nbits: u8) -> i32 {
        // The threshold is 2^(nbits-1). If val < threshold, subtract 2^nbits - 1.
        let threshold = 1u32 << (nbits - 1);
        if val < threshold {
            val as i32 - ((1i32 << nbits) - 1)
        } else {
            val as i32
        }
    }

    /// Whether the bit stream has reached a marker or end of data.
    #[allow(dead_code)]
    pub fn is_at_marker(&self) -> bool {
        self.available == 0
            && self.pos < self.data.len().saturating_sub(1)
            && self.data[self.pos] == 0xFF
            && self.data[self.pos + 1] != 0x00
    }

    /// Skip any remaining bits to the next byte boundary and advance past
    /// any 0xFF padding bytes. Used after restart markers.
    pub fn align_to_byte(&mut self) {
        self.bits = 0;
        self.available = 0;
    }

    /// Skip past the next restart marker (0xFF 0xDn).
    pub fn skip_restart_marker(&mut self) -> bool {
        // First align
        self.align_to_byte();

        // Scan for 0xFF followed by RST marker
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF {
                let m = self.data[self.pos + 1];
                if (0xD0..=0xD7).contains(&m) {
                    self.pos += 2;
                    self.fill();
                    return true;
                } else if m == 0x00 {
                    self.pos += 2;
                    continue;
                } else {
                    // Non-restart marker
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
        // 0xFF 0x00 → single 0xFF data byte
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
        // For 3 bits, threshold=4. Values 0..3 are negative.
        assert_eq!(BitStream::extend(0, 3), -7);
        assert_eq!(BitStream::extend(1, 3), -6);
        assert_eq!(BitStream::extend(3, 3), -4);
    }

    #[test]
    fn extend_one_bit() {
        assert_eq!(BitStream::extend(0, 1), -1);
        assert_eq!(BitStream::extend(1, 1), 1);
    }
}
