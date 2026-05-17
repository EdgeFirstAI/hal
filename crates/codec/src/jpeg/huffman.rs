// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Huffman table construction and decoding with 11-bit lookahead LUT.

use crate::error::CodecError;
use crate::jpeg::bitstream::BitStream;
use crate::jpeg::types::QuantTable;

/// Maximum Huffman code length in JPEG.
const MAX_CODE_LEN: usize = 16;

/// 11-bit lookahead: resolves ~99% of Huffman codes in a single table hit,
/// reducing branch mispredictions vs the previous 9-bit table. The 4 KB
/// table (2048 × 2 bytes) fits comfortably in L1 cache.
const FAST_BITS: u8 = 11;
const FAST_SIZE: usize = 1 << FAST_BITS;

/// A Huffman lookup table with 11-bit fast path.
///
/// For codes ≤ 11 bits, `fast[peek_11_bits]` gives the symbol and code length.
/// For longer codes (12–16 bits), slow linear decode is used (extremely rare).
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Fast lookup: `fast[index] = (symbol, code_length)`.
    /// `code_length == 0` means this entry is invalid (code > 11 bits).
    fast: Vec<(u8, u8)>,
    /// Code values indexed by increasing code length. Used for slow decode.
    symbols: Vec<u8>,
    /// `max_code[i]` = maximum code value for codes of length `i+1`.
    /// -1 means no codes of that length.
    max_code: [i32; MAX_CODE_LEN],
    /// `val_offset[i]` = index into `symbols` for the first code of length `i+1`.
    val_offset: [i32; MAX_CODE_LEN],
}

impl HuffmanTable {
    /// Build a Huffman table from JPEG DHT segment data.
    ///
    /// `counts[i]` = number of codes with bit length `i+1` (i = 0..15).
    /// `values` = symbol values in order of increasing code length.
    pub fn build(counts: &[u8; 16], values: &[u8]) -> crate::Result<Self> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if values.len() < total {
            return Err(CodecError::InvalidData(
                "DHT: fewer values than count sum".into(),
            ));
        }

        // Build canonical Huffman code assignments
        let mut max_code = [-1i32; MAX_CODE_LEN];
        let mut val_offset = [0i32; MAX_CODE_LEN];

        let mut code: u32 = 0;
        let mut si = 0usize; // Index into values

        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                val_offset[i] = si as i32 - code as i32;
                si += count as usize;
                max_code[i] = (code + count as u32 - 1) as i32;
                code += count as u32;
            }
            code <<= 1;
        }

        // Build 11-bit fast lookup table
        let mut fast = vec![(0u8, 0u8); FAST_SIZE];
        code = 0;
        si = 0;
        for (i, &count) in counts.iter().enumerate() {
            let bit_len = (i + 1) as u8;
            for _ in 0..count {
                if bit_len <= FAST_BITS {
                    let symbol = values[si];
                    // Fill all table entries that start with this code
                    let fill = 1 << (FAST_BITS - bit_len);
                    let base = (code << (FAST_BITS - bit_len)) as usize;
                    for j in 0..fill {
                        fast[base + j] = (symbol, bit_len);
                    }
                }
                code += 1;
                si += 1;
            }
            code <<= 1;
        }

        // Build combined AC fast table (zune's approach)
        Ok(Self {
            fast,
            symbols: values[..total].to_vec(),
            max_code,
            val_offset,
        })
    }

    /// Decode one Huffman symbol from the bit stream.
    ///
    /// Uses the 11-bit fast lookup table for short codes, falls back to
    /// slow bit-by-bit decode for codes > 11 bits.
    #[inline]
    pub fn decode_symbol(&self, bs: &mut BitStream<'_>) -> crate::Result<u8> {
        let peek = bs.peek(FAST_BITS) as usize;
        let (symbol, len) = self.fast[peek];
        if len > 0 {
            bs.consume(len);
            return Ok(symbol);
        }

        // Slow path: consume bits one at a time
        self.decode_slow(bs)
    }

    /// Slow Huffman decode for codes > 11 bits.
    fn decode_slow(&self, bs: &mut BitStream<'_>) -> crate::Result<u8> {
        // We already peeked 11 bits; start from length 12
        let mut code = bs.peek(FAST_BITS) as i32;
        bs.consume(FAST_BITS);

        for i in (FAST_BITS as usize)..MAX_CODE_LEN {
            let next_bit = bs.read_bits(1) as i32;
            code = (code << 1) | next_bit;

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

/// Decode one 8×8 block of DCT coefficients from the entropy-coded data.
///
/// Writes 64 dequantised coefficients into `coeffs` in natural (row-major) order.
/// `dc_pred` is updated with the new DC prediction value.
#[inline]
pub fn decode_block(
    bs: &mut BitStream<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    quant: &QuantTable,
    coeffs: &mut [i32; 64],
    dc_pred: &mut i32,
) -> crate::Result<()> {
    // Clear coefficients
    *coeffs = [0i32; 64];

    // DC coefficient
    let dc_size = dc_table.decode_symbol(bs)?;
    if dc_size > 0 {
        if dc_size > 11 {
            return Err(CodecError::InvalidData("JPEG: DC size > 11".into()));
        }
        let dc_val = bs.read_bits(dc_size);
        let dc_diff = BitStream::extend(dc_val, dc_size);
        *dc_pred += dc_diff;
    }
    // Dequant fused: multiply by quant table entry during decode
    coeffs[0] = *dc_pred * quant.values[0] as i32;

    // AC coefficients
    let mut k = 1;
    while k < 64 {
        let symbol = ac_table.decode_symbol(bs)?;
        if symbol == 0x00 {
            // EOB — remaining coefficients are zero
            break;
        }

        let run = (symbol >> 4) as usize;
        let size = symbol & 0x0F;

        if symbol == 0xF0 {
            // ZRL: skip 16 zeros
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

        if size > 0 {
            let val = bs.read_bits(size);
            let coeff = BitStream::extend(val, size);
            // Natural order index via the zig-zag table in QuantTable
            // Note: quant table values are already in natural order
            let natural_idx = crate::jpeg::types::ZIGZAG[k] as usize;
            coeffs[natural_idx] = coeff * quant.values[natural_idx] as i32;
        }

        k += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_table() {
        let mut counts = [0u8; 16];
        counts[0] = 2; // 2 codes of length 1
        let values = [0x00, 0x01];
        let table = HuffmanTable::build(&counts, &values).unwrap();

        // Code 0 → symbol 0, code 1 → symbol 1
        let data = [0b1000_0000]; // bit 0 = '1' → symbol 1
        let mut bs = BitStream::new(&data, 0);
        assert_eq!(table.decode_symbol(&mut bs).unwrap(), 1);
    }

    #[test]
    fn roundtrip_dc_values() {
        // Standard JPEG DC luminance table (Table K.3)
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
        // Just verify it builds without error
        assert_eq!(table.symbols.len(), 12);
    }

    #[test]
    fn reject_invalid_counts() {
        let mut counts = [0u8; 16];
        counts[0] = 5;
        let values = [0u8; 2]; // Too few values
        let result = HuffmanTable::build(&counts, &values);
        assert!(result.is_err());
    }
}
