// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Huffman table construction and decoding with a tiered fast lookahead LUT.
//!
//! Fast-table width is selected per CPU tier ([`cpu::entropy_huffman_fast_bits`]):
//! 10 bits everywhere except the widest tiers (aarch64 `High`, x86 `Avx2`),
//! which take 11 — see the tier docs for the memory/hit-rate trade-off.
//! Entries are packed `u16 = (symbol << 8) | len` (libjpeg-turbo style).

use crate::error::CodecError;
use crate::jpeg::bitstream::BitStream;
use crate::jpeg::cpu;

/// Maximum Huffman code length in JPEG.
const MAX_CODE_LEN: usize = 16;

/// A Huffman lookup table with a runtime-sized fast path.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Packed fast lookup: `(symbol << 8) | code_length`. `code_length == 0`
    /// means the code is longer than `fast_bits` (slow path).
    fast: Vec<u16>,
    /// Combined code+magnitude fast lookup (stb_image `fast_ac` style). For
    /// lookahead windows where the Huffman code **and** its magnitude bits both
    /// fit in `fast_bits`, the entry packs the fully decoded coefficient:
    ///
    /// `(sign-extended value as u16) << 16 | run << 8 | total_bits`
    ///
    /// where `total_bits = code_len + magnitude_bits` is the single `consume`
    /// amount. `0` means miss (EOB/ZRL, size-0 symbols, or codes that spill
    /// past the window) — fall back to the two-step path. For DC tables the
    /// symbol *is* the size, so `run == 0` on every valid hit.
    fast_ac: Vec<u32>,
    /// Lookahead width used to index `fast`.
    fast_bits: u8,
    /// Code values indexed by increasing code length. Used for slow decode.
    symbols: Vec<u8>,
    /// `max_code[i]` = maximum code value for codes of length `i+1`.
    max_code: [i32; MAX_CODE_LEN],
    /// `val_offset[i]` = index into `symbols` for the first code of length `i+1`.
    val_offset: [i32; MAX_CODE_LEN],
}

impl HuffmanTable {
    /// Build a Huffman table using the process-wide NEON tier's fast-bit width.
    pub fn build(counts: &[u8; 16], values: &[u8]) -> crate::Result<Self> {
        Self::build_with_fast_bits(counts, values, cpu::entropy_huffman_fast_bits())
    }

    /// Build with an explicit fast-lookup width (8..=12).
    pub fn build_with_fast_bits(
        counts: &[u8; 16],
        values: &[u8],
        fast_bits: u8,
    ) -> crate::Result<Self> {
        let fast_bits = fast_bits.clamp(8, 12);
        let fast_size = 1usize << fast_bits;
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if values.len() < total {
            return Err(CodecError::InvalidData(
                "DHT: fewer values than count sum".into(),
            ));
        }

        // The counts come straight from an attacker-controlled DHT segment, and
        // the fast-table fill below only stays inside its allocation for a
        // canonical prefix code: `base` is the code value shifted up to
        // `fast_bits`, so an oversubscribed level walks off the end (three
        // 1-bit codes put the third `base` exactly at the table length).
        // Kraft: after `count` codes of length L, no more than 2^L can exist.
        let mut assigned: u32 = 0;
        for (i, &count) in counts.iter().enumerate() {
            assigned += count as u32;
            if assigned > (1u32 << (i + 1)) {
                return Err(CodecError::InvalidData(
                    "DHT: oversubscribed Huffman code lengths".into(),
                ));
            }
            assigned <<= 1;
        }

        let mut max_code = [-1i32; MAX_CODE_LEN];
        let mut val_offset = [0i32; MAX_CODE_LEN];

        let mut code: u32 = 0;
        let mut si = 0usize;

        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                val_offset[i] = si as i32 - code as i32;
                si += count as usize;
                max_code[i] = (code + count as u32 - 1) as i32;
                code += count as u32;
            }
            code <<= 1;
        }

        let mut fast = vec![0u16; fast_size];
        code = 0;
        si = 0;
        for (i, &count) in counts.iter().enumerate() {
            let bit_len = (i + 1) as u8;
            for _ in 0..count {
                if bit_len <= fast_bits {
                    let symbol = values[si];
                    let fill = 1 << (fast_bits - bit_len);
                    let base = (code << (fast_bits - bit_len)) as usize;
                    let packed = ((symbol as u16) << 8) | (bit_len as u16);
                    for j in 0..fill {
                        fast[base + j] = packed;
                    }
                }
                code += 1;
                si += 1;
            }
            code <<= 1;
        }

        // Combined code+magnitude table: for every window whose code hit the
        // fast table AND whose magnitude bits also fit in the window, bake the
        // sign-extended coefficient value so the hot loop decodes run + value
        // + consume-length with a single lookup.
        let mut fast_ac = vec![0u32; fast_size];
        for (idx, &packed) in fast.iter().enumerate() {
            let len = (packed & 0xFF) as u8;
            if len == 0 {
                continue;
            }
            let symbol = (packed >> 8) as u8;
            let run = (symbol >> 4) as u32;
            let size = symbol & 0x0F;
            if size == 0 || len + size > fast_bits {
                continue; // EOB/ZRL or magnitude spills past the window
            }
            let mag = ((idx >> (fast_bits - len - size)) as u32) & ((1u32 << size) - 1);
            let val = BitStream::extend(mag, size) as i16;
            fast_ac[idx] = ((val as u16 as u32) << 16) | (run << 8) | (len + size) as u32;
        }

        Ok(Self {
            fast,
            fast_ac,
            fast_bits,
            symbols: values[..total].to_vec(),
            max_code,
            val_offset,
        })
    }

    /// Probe the combined code+magnitude table with a `fast_bits`-wide window.
    /// Non-zero hit ⇒ `(value << 16) | (run << 8) | total_bits`.
    ///
    /// Takes the stream rather than a caller-computed index: the window is
    /// `peek(self.fast_bits)`, so reading it here makes "index is in range" a
    /// property of this function instead of an invariant every call site has to
    /// re-establish for the unchecked load to stay sound.
    #[inline(always)]
    pub fn probe_fast_ac(&self, bs: &BitStream<'_>) -> u32 {
        let window = bs.peek(self.fast_bits) as usize;
        // SAFETY: peek returns fast_bits bits, so window < 1 << fast_bits,
        // which is exactly the length fast_ac was allocated with.
        unsafe { *self.fast_ac.get_unchecked(window) }
    }

    /// Decode one Huffman symbol from the bit stream.
    ///
    /// Contract: the caller refilled the buffer to ≥ 32 bits ([`BitStream::refill`]),
    /// covering the longest code (16) plus its magnitude bits without another
    /// memory touch. Pure register ops on the fast path: peek → LUT → consume.
    #[inline(always)]
    pub fn decode_symbol(&self, bs: &mut BitStream<'_>) -> crate::Result<u8> {
        let peek = bs.peek(self.fast_bits) as usize;
        // peek is always < 2^fast_bits; table was sized to match.
        let packed = unsafe { *self.fast.get_unchecked(peek) };
        let len = (packed & 0xFF) as u8;
        if len > 0 {
            bs.consume(len);
            return Ok((packed >> 8) as u8);
        }
        self.decode_slow(bs)
    }

    /// Slow Huffman decode for codes longer than `fast_bits` (rare: <1% of
    /// symbols at 10-bit lookahead). Consumes at most 16 buffered bits, within
    /// the caller's refill guarantee.
    ///
    /// Inlined for the same reason as [`BitStream::refill_slow`]: taking
    /// `&mut BitStream` out of line escapes the pointer and pushes the bit
    /// buffer out of registers for the whole coefficient loop.
    #[inline(always)]
    fn decode_slow(&self, bs: &mut BitStream<'_>) -> crate::Result<u8> {
        let fb = self.fast_bits;
        let mut code = bs.peek(fb) as i32;
        bs.consume(fb);

        for i in (fb as usize)..MAX_CODE_LEN {
            code = (code << 1) | bs.get_bits(1) as i32;

            if code <= self.max_code[i] {
                let idx = (code + self.val_offset[i]) as usize;
                if idx < self.symbols.len() {
                    return Ok(self.symbols[idx]);
                }
            }
        }

        Err(CodecError::InvalidData("JPEG: invalid Huffman code".into()))
    }
}

/// Decode one 8×8 block of **quantised** DCT coefficients from the
/// entropy-coded data. Dequantisation happens inside the IDCT kernels
/// (libjpeg-turbo `JCOEF` model): the entropy loop stays multiply-free and the
/// coefficient block is a 128-byte `i16` array instead of 256-byte `i32`.
///
/// Returns `true` when any non-zero AC coefficient was decoded.
///
/// `prev_has_ac` tracks whether the shared `coeffs` scratch still holds stale AC
/// values from the previous block. When the previous block was DC-only only
/// `coeffs[0]` needs clearing — avoids a 128-byte memset on every MCU block
/// (COCO 4:4:4 does ~3× more blocks than 4:2:0).
///
/// Refill discipline (turbo `decode_mcu_fast` shape): one [`BitStream::refill`]
/// per coefficient covers the Huffman code (≤ 16 bits) plus its magnitude bits
/// (≤ 11), so everything between refills is register arithmetic. The combined
/// `fast_ac` probe additionally resolves code + run + sign-extended value with
/// a single table load and a single `consume` when both fit the lookahead.
#[inline]
pub fn decode_block(
    bs: &mut BitStream<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    dc_pred: &mut i32,
    prev_has_ac: &mut bool,
) -> crate::Result<bool> {
    if *prev_has_ac {
        *coeffs = [0i16; 64];
    } else {
        coeffs[0] = 0;
    }

    bs.refill(); // ≥ 32 bits: DC code (≤16) + DC magnitude (≤11)
                 // Combined fast path: one lookup yields the sign-extended DC diff and the
                 // total bits to consume. `run != 0` never occurs in a valid DC hit (the
                 // symbol is the size, ≤ 11), so any non-zero-run entry falls through.
    let dc_fa = dc_table.probe_fast_ac(bs);
    if dc_fa != 0 && dc_fa & 0xFF00 == 0 {
        bs.consume((dc_fa & 0xFF) as u8);
        *dc_pred += (dc_fa >> 16) as u16 as i16 as i32;
    } else {
        let dc_size = dc_table.decode_symbol(bs)?;
        if dc_size > 0 {
            if dc_size > 11 {
                return Err(CodecError::InvalidData("JPEG: DC size > 11".into()));
            }
            let dc_val = bs.get_bits(dc_size);
            let dc_diff = BitStream::extend(dc_val, dc_size);
            *dc_pred += dc_diff;
        }
    }
    coeffs[0] = *dc_pred as i16;

    let mut has_ac = false;
    let mut k = 1;
    while k < 64 {
        bs.refill(); // ≥ 32 bits: AC code (≤16) + AC magnitude (≤10)

        // Combined fast path: code + run + sign-extended value in one probe,
        // one consume. Covers the overwhelming majority of AC coefficients
        // (small magnitudes with short codes).
        let fa = ac_table.probe_fast_ac(bs);
        if fa != 0 {
            bs.consume((fa & 0xFF) as u8);
            k += ((fa >> 8) & 0xFF) as usize;
            if k >= 64 {
                return Err(CodecError::InvalidData(
                    "JPEG: AC coefficient past end".into(),
                ));
            }
            // SAFETY: k < 64 (checked above); ZIGZAG values are all < 64.
            unsafe {
                let natural_idx = *crate::jpeg::types::ZIGZAG.get_unchecked(k) as usize;
                *coeffs.get_unchecked_mut(natural_idx) = (fa >> 16) as u16 as i16;
            }
            has_ac = true;
            k += 1;
            continue;
        }

        let symbol = ac_table.decode_symbol(bs)?;
        if symbol == 0x00 {
            break; // EOB
        }

        let run = (symbol >> 4) as usize;
        let size = symbol & 0x0F;

        if size == 0 {
            // ZRL (0xF0) — sixteen zero coefficients; any other size-0 is invalid
            // except EOB (handled above).
            if symbol != 0xF0 {
                return Err(CodecError::InvalidData("JPEG: invalid AC RS".into()));
            }
            k += 16;
            if k > 64 {
                return Err(CodecError::InvalidData("JPEG: AC run past end".into()));
            }
            continue;
        }

        k += run;
        if k >= 64 {
            return Err(CodecError::InvalidData(
                "JPEG: AC coefficient past end".into(),
            ));
        }

        if size > 10 {
            return Err(CodecError::InvalidData(format!(
                "JPEG: AC coefficient size {size} exceeds spec maximum (10)"
            )));
        }
        let val = bs.get_bits(size);
        let coeff = BitStream::extend(val, size);
        // SAFETY: k < 64 (checked above); ZIGZAG values are all < 64.
        unsafe {
            let natural_idx = *crate::jpeg::types::ZIGZAG.get_unchecked(k) as usize;
            *coeffs.get_unchecked_mut(natural_idx) = coeff as i16;
        }
        has_ac = true;
        k += 1;
    }

    *prev_has_ac = has_ac;
    Ok(has_ac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_table() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0x00, 0x01];
        let table = HuffmanTable::build(&counts, &values).unwrap();

        let data = [0b1000_0000];
        let mut bs = BitStream::new(&data, 0);
        assert_eq!(table.decode_symbol(&mut bs).unwrap(), 1);
    }

    #[test]
    fn roundtrip_dc_values() {
        let mut counts = [0u8; 16];
        counts[0] = 0;
        counts[1] = 1;
        counts[2] = 5;
        counts[3] = 1;
        counts[4] = 1;
        counts[5] = 1;
        counts[6] = 1;
        counts[7] = 1;
        counts[8] = 1;
        let values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let table = HuffmanTable::build(&counts, &values).unwrap();
        assert!(!table.fast.is_empty());
    }

    #[test]
    fn build_rejects_truncated_values() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0u8];
        let result = HuffmanTable::build(&counts, &values);
        assert!(result.is_err());
    }

    /// A DHT segment is attacker-controlled, and only a canonical prefix code
    /// keeps the fast-table fill inside its allocation: three 1-bit codes make
    /// the third `base` land exactly at the table length. Rejected up front, so
    /// a malformed table is an error rather than a panic.
    #[test]
    fn build_rejects_oversubscribed_code_lengths() {
        let mut counts = [0u8; 16];
        counts[0] = 3; // only two 1-bit codes can exist
        let values = [0u8, 1, 2];
        assert!(HuffmanTable::build(&counts, &values).is_err());

        // Oversubscription deeper in the tree, past the fast-table width.
        let mut counts = [0u8; 16];
        counts[0] = 1;
        counts[1] = 3; // one 1-bit code leaves room for only two 2-bit codes
        let values = [0u8, 1, 2, 3];
        assert!(HuffmanTable::build(&counts, &values).is_err());

        // A full canonical code must still build: 2 one-bit codes is the limit.
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0u8, 1];
        assert!(HuffmanTable::build(&counts, &values).is_ok());
    }
}
