// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Huffman table construction and decoding with a tiered fast lookahead LUT.
//!
//! Fast-table width is selected per CPU tier ([`cpu::entropy_huffman_fast_bits`]):
//! 10 bits everywhere except the widest tiers (aarch64 `High`, x86 `Avx2`),
//! which take 11 — see the tier docs for the memory/hit-rate trade-off.
//! Entries are packed `u16 = (symbol << 8) | len` (libjpeg-turbo style).

use crate::error::CodecError;
use crate::jpeg::bitstream::{BitCursor, BitStream};
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
    /// amount. `0` means miss (size-0 symbols other than the baked ones below,
    /// or codes that spill past the window) — fall back to the two-step path.
    ///
    /// Three size-0 symbols are baked in rather than treated as misses:
    ///
    /// - **DC size 0** (`is_ac == false`, symbol 0x00): predictor unchanged —
    ///   a plain hit with `value = 0, run = 0`. Very common in flat regions.
    /// - **EOB** (`is_ac == true`, symbol 0x00): encoded with the sentinel
    ///   `run = 0xFF`, which forces `k += run` past 63 so the block decoder's
    ///   *existing* bounds branch resolves end-of-block — no extra hot-path
    ///   compare, and the two-step fallback disappears from the common exit.
    /// - **ZRL** (`is_ac == true`, symbol 0xF0): `value = 0, run = 15` writes
    ///   a zero into a slot the clear discipline already guarantees is zero —
    ///   sixteen positions are skipped, matching the two-step path.
    ///
    /// For DC tables the symbol *is* the size, so `run == 0` on every valid
    /// hit and the sentinels never appear.
    fast_ac: Vec<u32>,
    /// Paired-coefficient lookup (AC tables only; empty for DC). Decodes up to
    /// **two** coefficients per probe when both fit the lookahead window —
    /// on COCO val2017 ~54% of AC coefficients pair at 10 bits, and the serial
    /// peek → load → consume chain is the in-order bottleneck, so halving the
    /// probes on paired coefficients attacks it directly.
    ///
    /// Entry: `val2:i8 << 24 | val1:i8 << 16 | run2:u4 << 12 | run1:u4 << 8 |
    /// total_bits:u8`. `0` = miss (EOB/ZRL windows, codes past the window,
    /// magnitudes over 7 bits).
    ///
    /// Single coefficients are baked as **phantom pairs** (`val2 = 0,
    /// run2 = 0`) so the probe branch keeps the ~94% hit rate of the single
    /// table instead of introducing a ~50/50 pair-vs-single branch that an
    /// in-order branch predictor would keep missing. The block decoder
    /// discriminates real-vs-phantom *branchlessly*: it always stores `val2`
    /// at the second position — a phantom writes `0` into the next undecoded
    /// slot, which the clear discipline already guarantees is zero — and
    /// advances `k` by `(val2 != 0)`. Sound because a real second value is
    /// never zero: JPEG magnitude coding cannot encode 0 (`size ≥ 1 ⇒
    /// |value| ≥ 1`), and `size ≤ 7 ⇒ |value| ≤ 127` fits the i8 lane.
    pair_ac: Vec<u32>,
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
    ///
    /// `is_ac` selects the run/size interpretation of size-0 symbols when
    /// baking the combined table — see [`Self::fast_ac`].
    pub fn build(counts: &[u8; 16], values: &[u8], is_ac: bool) -> crate::Result<Self> {
        Self::build_with_fast_bits(counts, values, cpu::entropy_huffman_fast_bits(), is_ac)
    }

    /// Build with an explicit fast-lookup width (8..=12).
    pub fn build_with_fast_bits(
        counts: &[u8; 16],
        values: &[u8],
        fast_bits: u8,
        is_ac: bool,
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
            if size == 0 {
                // Size-0 symbols: bake the three meaningful ones (see the
                // `fast_ac` field docs); everything else stays a miss so the
                // two-step path can reject it.
                fast_ac[idx] = match (is_ac, symbol) {
                    (false, 0x00) => len as u32,              // DC diff 0
                    (true, 0x00) => (0xFF << 8) | len as u32, // EOB sentinel
                    (true, 0xF0) => (15 << 8) | len as u32,   // ZRL
                    _ => 0,
                };
                continue;
            }
            if len + size > fast_bits {
                continue; // magnitude spills past the window
            }
            let mag = ((idx >> (fast_bits - len - size)) as u32) & ((1u32 << size) - 1);
            let val = BitStream::extend(mag, size) as i16;
            fast_ac[idx] = ((val as u16 as u32) << 16) | (run << 8) | (len + size) as u32;
        }

        // Paired-coefficient table (see the `pair_ac` field docs). Decodes a
        // window as coefficient-1 [+ coefficient-2] entirely at build time.
        let mut pair_ac = Vec::new();
        if is_ac {
            pair_ac = vec![0u32; fast_size];
            // First code+magnitude of a window, when it is a pairable
            // coefficient: size 1..=7 (value fits i8) fully inside the window.
            let coeff_at = |idx: usize, offset: u8| -> Option<(u8, u8, i8)> {
                let sub = (idx << offset) & (fast_size - 1);
                let packed = fast[sub];
                let len = (packed & 0xFF) as u8;
                if len == 0 {
                    return None;
                }
                let symbol = (packed >> 8) as u8;
                let run = symbol >> 4;
                let size = symbol & 0x0F;
                if size == 0 || size > 7 || offset + len + size > fast_bits {
                    return None; // EOB/ZRL, wide magnitude, or spills the window
                }
                // `sub` already carries the window shifted up by `offset`, so
                // the magnitude sits `len` bits below its top.
                let shift = fast_bits - len - size;
                let mag = ((sub >> shift) as u32) & ((1u32 << size) - 1);
                let val = BitStream::extend(mag, size) as i8;
                Some((len + size, run, val))
            };
            for (idx, entry) in pair_ac.iter_mut().enumerate() {
                let Some((used1, run1, val1)) = coeff_at(idx, 0) else {
                    continue;
                };
                // Second coefficient from the remaining window bits, else a
                // phantom (val2 = 0, run2 = 0) single.
                let (total, run2, val2) = match coeff_at(idx, used1) {
                    Some((used2, run2, val2)) => (used1 + used2, run2, val2),
                    None => (used1, 0, 0),
                };
                *entry = ((val2 as u8 as u32) << 24)
                    | ((val1 as u8 as u32) << 16)
                    | ((run2 as u32) << 12)
                    | ((run1 as u32) << 8)
                    | total as u32;
            }
        }

        Ok(Self {
            fast,
            fast_ac,
            pair_ac,
            fast_bits,
            symbols: values[..total].to_vec(),
            max_code,
            val_offset,
        })
    }

    /// Probe the paired-coefficient table (AC tables only — the table is empty
    /// for DC tables and this must not be called on them). Non-zero hit ⇒ one
    /// or two fully decoded coefficients; see [`Self::pair_ac`] for the layout.
    #[inline(always)]
    pub fn probe_pair_ac(&self, cur: &BitCursor<'_>) -> u32 {
        debug_assert!(!self.pair_ac.is_empty(), "pair probe on a DC table");
        let window = cur.peek(self.fast_bits) as usize;
        // SAFETY: peek returns fast_bits bits, so window < 1 << fast_bits,
        // the length pair_ac was allocated with for every AC table.
        unsafe { *self.pair_ac.get_unchecked(window) }
    }

    /// Probe the combined code+magnitude table with a `fast_bits`-wide window.
    /// Non-zero hit ⇒ `(value << 16) | (run << 8) | total_bits`.
    ///
    /// Takes the cursor rather than a caller-computed index: the window is
    /// `peek(self.fast_bits)`, so reading it here makes "index is in range" a
    /// property of this function instead of an invariant every call site has to
    /// re-establish for the unchecked load to stay sound.
    #[inline(always)]
    pub fn probe_fast_ac(&self, cur: &BitCursor<'_>) -> u32 {
        let window = cur.peek(self.fast_bits) as usize;
        // SAFETY: peek returns fast_bits bits, so window < 1 << fast_bits,
        // which is exactly the length fast_ac was allocated with.
        unsafe { *self.fast_ac.get_unchecked(window) }
    }

    /// Decode one Huffman symbol from the bit cursor.
    ///
    /// Contract: the caller refilled the buffer to ≥ 32 bits ([`BitCursor::refill`]),
    /// covering the longest code (16) plus its magnitude bits without another
    /// memory touch. Pure register ops on the fast path: peek → LUT → consume.
    #[inline(always)]
    pub fn decode_symbol(&self, cur: &mut BitCursor<'_>) -> crate::Result<u8> {
        let peek = cur.peek(self.fast_bits) as usize;
        // peek is always < 2^fast_bits; table was sized to match.
        let packed = unsafe { *self.fast.get_unchecked(peek) };
        let len = (packed & 0xFF) as u8;
        if len > 0 {
            cur.consume(len);
            return Ok((packed >> 8) as u8);
        }
        self.decode_slow(cur)
    }

    /// Slow Huffman decode for codes longer than `fast_bits` (rare: <1% of
    /// symbols at 10-bit lookahead). Consumes at most 16 buffered bits, within
    /// the caller's refill guarantee.
    ///
    /// Inlined for the same reason as [`BitCursor::refill_slow`]: taking
    /// the cursor out of line escapes the pointer and pushes the bit
    /// buffer out of registers for the whole coefficient loop.
    #[inline(always)]
    fn decode_slow(&self, cur: &mut BitCursor<'_>) -> crate::Result<u8> {
        let fb = self.fast_bits;
        let mut code = cur.peek(fb) as i32;
        cur.consume(fb);

        for i in (fb as usize)..MAX_CODE_LEN {
            code = (code << 1) | cur.get_bits(1) as i32;

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

/// Per-slot DHT fingerprint plus the built LUT, reused across frames.
///
/// Camera streams keep the same DHT for every frame; rebuilding the lookahead
/// tables is then pure overhead. The LUTs are **moved** into the parse result
/// for the decode (no `clone` of the `Vec` tables) and [`Self::restore`]d
/// afterwards so the hot loop stays allocation-free on a cache hit.
#[derive(Debug, Default)]
pub(crate) struct HuffmanCache {
    dc: [CachedDht; 4],
    ac: [CachedDht; 4],
}

#[derive(Debug, Default)]
struct CachedDht {
    counts: [u8; 16],
    values: Vec<u8>,
    table: Option<HuffmanTable>,
}

impl HuffmanCache {
    /// Return a built table for this DHT payload, taking a cached LUT when the
    /// counts and symbols match the previous frame.
    pub(crate) fn intern(
        &mut self,
        class: u8,
        id: usize,
        counts: &[u8; 16],
        values: &[u8],
    ) -> crate::Result<HuffmanTable> {
        let slot = if class == 0 {
            &mut self.dc[id]
        } else {
            &mut self.ac[id]
        };
        if slot.counts == *counts && slot.values.as_slice() == values {
            if let Some(table) = slot.table.take() {
                return Ok(table);
            }
        }
        // Rebuilding for a different payload: drop any table still parked in
        // the slot. It was built for the OLD payload, and leaving it there
        // lets a later matching intern hand it out under the new fingerprint
        // (reachable via a duplicate DHT for the same slot within one frame,
        // or an error exit that skipped `restore`) — decoding with the wrong
        // codes.
        slot.table = None;
        let table = HuffmanTable::build(counts, values, class != 0)?;
        slot.counts = *counts;
        slot.values.clear();
        slot.values.extend_from_slice(values);
        Ok(table)
    }

    /// Move tables from a finished parse/decode back into the cache.
    pub(crate) fn restore(
        &mut self,
        dc_tables: &mut [Option<HuffmanTable>; 4],
        ac_tables: &mut [Option<HuffmanTable>; 4],
    ) {
        for i in 0..4 {
            if let Some(t) = dc_tables[i].take() {
                self.dc[i].table = Some(t);
            }
            if let Some(t) = ac_tables[i].take() {
                self.ac[i].table = Some(t);
            }
        }
    }
}

/// Entropy-decode result for one block: whether any AC coefficient was
/// written, and the highest zigzag index that was written (`last_k`, 0 for a
/// DC-only block). `last_k` is a free by-product of the scan order — zigzag
/// indices only grow — and bounds the block's occupied zigzag prefix, which
/// the NEON IDCT uses to pick a sparse tier without re-deriving sparsity from
/// the coefficient vectors.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    /// Any non-zero AC coefficient decoded (DC-only shortcut gate).
    pub has_ac: bool,
    /// Highest zigzag index written (upper bound; a baked ZRL writes a zero).
    pub last_k: u8,
}

/// Decode one 8×8 block of **quantised** DCT coefficients from the
/// entropy-coded data. Dequantisation happens inside the IDCT kernels
/// (libjpeg-turbo `JCOEF` model): the entropy loop stays multiply-free and the
/// coefficient block is a 128-byte `i16` array instead of 256-byte `i32`.
///
/// `prev_has_ac` tracks whether the shared `coeffs` scratch still holds stale AC
/// values from the previous block. When the previous block was DC-only only
/// `coeffs[0]` needs clearing — avoids a 128-byte memset on every MCU block
/// (COCO 4:4:4 does ~3× more blocks than 4:2:0).
///
/// Refill discipline (turbo `decode_mcu_fast` shape): one [`BitCursor::refill`]
/// per coefficient covers the Huffman code (≤ 16 bits) plus its magnitude bits
/// (≤ 11), so everything between refills is register arithmetic. The combined
/// `fast_ac` probe additionally resolves code + run + sign-extended value with
/// a single table load and a single `consume` when both fit the lookahead;
/// EOB arrives through the same probe as a `run = 0xFF` sentinel that the
/// existing `k >= 64` bounds branch resolves (see [`HuffmanTable::fast_ac`]).
///
/// The whole block runs on a [`BitCursor`] so the bit buffer lives in
/// registers; state is committed back to the stream on every exit, error
/// exits included.
///
/// `#[inline(never)]` is load-bearing. With the cursor pattern LLVM finds
/// inlining this into the MCU loop attractive (it can then eliminate the
/// cursor copies against the caller's `BitStream`), but `decode_image`'s
/// frame already spills, and merging the two register-allocation problems
/// costs ~20% of decode time on Cortex-A53 (measured: 7.68 ms → 9.23 ms
/// p50 on imx8mp COCO when this inlines). Outlined, the cursor is this
/// function's own non-escaping local and lives entirely in registers.
#[inline(never)]
pub fn decode_block<const PAIR: bool>(
    bs: &mut BitStream<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    dc_pred: &mut i32,
    prev_has_ac: &mut bool,
) -> crate::Result<BlockInfo> {
    let mut cur = bs.cursor();
    let res = decode_block_cur::<PAIR>(&mut cur, dc_table, ac_table, coeffs, dc_pred, prev_has_ac);
    bs.commit(cur);
    res
}

/// Cursor-side body of [`decode_block`]. `#[inline(always)]` so the cursor
/// stays a provably non-escaping local of the caller and its fields live in
/// registers for the whole coefficient loop.
#[inline(always)]
fn decode_block_cur<const PAIR: bool>(
    cur: &mut BitCursor<'_>,
    dc_table: &HuffmanTable,
    ac_table: &HuffmanTable,
    coeffs: &mut [i16; 64],
    dc_pred: &mut i32,
    prev_has_ac: &mut bool,
) -> crate::Result<BlockInfo> {
    if *prev_has_ac {
        *coeffs = [0i16; 64];
    } else {
        coeffs[0] = 0;
    }

    cur.refill(); // ≥ 32 bits: DC code (≤16) + DC magnitude (≤11)
                  // Combined fast path: one lookup yields the sign-extended DC diff and the
                  // total bits to consume. `run != 0` never occurs in a valid DC hit (the
                  // symbol is the size, ≤ 11, and size 0 is baked with run 0), so any
                  // non-zero-run entry falls through.
    let dc_fa = dc_table.probe_fast_ac(cur);
    if dc_fa != 0 && dc_fa & 0xFF00 == 0 {
        cur.consume((dc_fa & 0xFF) as u8);
        *dc_pred += (dc_fa >> 16) as u16 as i16 as i32;
    } else {
        let dc_size = dc_table.decode_symbol(cur)?;
        if dc_size > 0 {
            if dc_size > 11 {
                return Err(CodecError::InvalidData("JPEG: DC size > 11".into()));
            }
            let dc_val = cur.get_bits(dc_size);
            let dc_diff = BitStream::extend(dc_val, dc_size);
            *dc_pred += dc_diff;
        }
    }
    coeffs[0] = *dc_pred as i16;

    let mut has_ac = false;
    let mut last_k = 0usize;
    let mut k = 1;
    while k < 64 {
        cur.refill(); // ≥ 32 bits: covers every probe below (pair total ≤ 12)

        // Paired fast path: up to two coefficients in one probe and one
        // consume (~54% of COCO AC coefficients pair at 10 bits; singles hit
        // as phantom pairs, so this branch keeps the single table's ~94% hit
        // rate). The second store is branchless: a phantom writes value 0
        // into the next undecoded slot — already zero by the clear
        // discipline — and `k` advances by `(val2 != 0)`.
        let pa = if PAIR { ac_table.probe_pair_ac(cur) } else { 0 };
        if pa != 0 {
            let run1 = ((pa >> 8) & 0xF) as usize;
            let k1 = k + run1;
            // SAFETY: k1 ≤ 63 + 15 < ZIGZAG_EXT length (320). Issued before
            // the bounds branch to hide the L1 hit from the store below.
            let zz1 = unsafe { *crate::jpeg::types::ZIGZAG_EXT.get_unchecked(k1) } as usize;
            if k1 >= 64 {
                return Err(CodecError::InvalidData(
                    "JPEG: AC coefficient past end".into(),
                ));
            }
            let val2 = ((pa >> 24) as u8 as i8) as i16;
            let pos2 = k1 + 1 + ((pa >> 12) & 0xF) as usize;
            if pos2 < 64 {
                cur.consume((pa & 0xFF) as u8);
                // SAFETY: k1 < 64 and pos2 < 64 (checked above) — both
                // indices come from the real zigzag prefix. A phantom's
                // second store writes 0 at the next undecoded position
                // (pos2 == k1 + 1), which is already zero.
                unsafe {
                    *coeffs.get_unchecked_mut(zz1) = ((pa >> 16) as u8 as i8) as i16;
                    *coeffs.get_unchecked_mut(
                        *crate::jpeg::types::ZIGZAG_EXT.get_unchecked(pos2) as usize,
                    ) = val2;
                }
                k = pos2 + (val2 != 0) as usize;
                has_ac = true;
                last_k = k - 1;
                continue;
            }
            if val2 == 0 || k1 == 63 {
                // The block ends exactly at coefficient 1 — the entry's
                // "second symbol" was decoded from the next block's bits at
                // build time and must not be consumed. The single-coefficient
                // probe is a guaranteed hit here (the pair conditions are a
                // strict subset of its own) and carries coefficient 1's true
                // bit count.
                let fa = ac_table.probe_fast_ac(cur);
                debug_assert!(fa != 0);
                debug_assert_eq!((fa >> 8) & 0xFF, run1 as u32);
                cur.consume((fa & 0xFF) as u8);
                // SAFETY: k1 < 64 (checked above).
                unsafe {
                    *coeffs.get_unchecked_mut(zz1) = ((pa >> 16) as u8 as i8) as i16;
                }
                k = k1 + 1;
                has_ac = true;
                last_k = k1;
                continue;
            }
            return Err(CodecError::InvalidData(
                "JPEG: AC coefficient past end".into(),
            ));
        }

        // Combined fast path: code + run + sign-extended value in one probe,
        // one consume. Still first choice for what the pair table skips:
        // EOB (run-0xFF sentinel), ZRL (run 15, value 0), and single
        // coefficients with magnitudes wider than 7 bits.
        let fa = ac_table.probe_fast_ac(cur);
        if fa != 0 {
            cur.consume((fa & 0xFF) as u8);
            let run = ((fa >> 8) & 0xFF) as usize;
            k += run;
            // Issued before the bounds branch so the L1 hit is underway by the
            // time the coefficient store needs the address.
            // SAFETY: k <= 63 + 255 < ZIGZAG_EXT length (320).
            let natural_idx = unsafe { *crate::jpeg::types::ZIGZAG_EXT.get_unchecked(k) } as usize;
            if k >= 64 {
                if run == 0xFF {
                    break; // EOB, baked into the probe table
                }
                return Err(CodecError::InvalidData(
                    "JPEG: AC coefficient past end".into(),
                ));
            }
            // SAFETY: k < 64 (checked above), so natural_idx came from the
            // real zigzag prefix and is < 64.
            unsafe {
                *coeffs.get_unchecked_mut(natural_idx) = (fa >> 16) as u16 as i16;
            }
            has_ac = true;
            last_k = k;
            k += 1;
            continue;
        }

        let symbol = ac_table.decode_symbol(cur)?;
        if symbol == 0x00 {
            break; // EOB (code longer than the fast window)
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
        let val = cur.get_bits(size);
        let coeff = BitStream::extend(val, size);
        // SAFETY: k < 64 (checked above); ZIGZAG values are all < 64.
        unsafe {
            let natural_idx = *crate::jpeg::types::ZIGZAG.get_unchecked(k) as usize;
            *coeffs.get_unchecked_mut(natural_idx) = coeff as i16;
        }
        has_ac = true;
        last_k = k;
        k += 1;
    }

    *prev_has_ac = has_ac;
    Ok(BlockInfo {
        has_ac,
        last_k: last_k as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_table() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0x00, 0x01];
        let table = HuffmanTable::build(&counts, &values, true).unwrap();

        let data = [0b1000_0000];
        let bs = BitStream::new(&data, 0);
        let mut cur = bs.cursor();
        assert_eq!(table.decode_symbol(&mut cur).unwrap(), 1);
    }

    /// DC size-0 (predictor unchanged) must be a `fast_ac` hit with value 0;
    /// the same symbol in an AC table is EOB and must carry the 0xFF run
    /// sentinel; ZRL must decode as run 15 / value 0.
    #[test]
    fn fast_ac_bakes_size_zero_symbols() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0x00, 0xF0];

        let dc = HuffmanTable::build(&counts, &values, false).unwrap();
        // Window of all-zero bits selects the first (size-0) code.
        assert_eq!(dc.fast_ac[0], 1, "DC size-0: value 0, run 0, 1 bit");

        let ac = HuffmanTable::build(&counts, &values, true).unwrap();
        assert_eq!(ac.fast_ac[0], (0xFF << 8) | 1, "EOB: run sentinel 0xFF");
        // Second 1-bit code (window with the top bit set) is ZRL.
        let zrl = ac.fast_ac[1 << (ac.fast_bits - 1)];
        assert_eq!(zrl, (15 << 8) | 1, "ZRL: value 0, run 15, 1 bit");
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
        let table = HuffmanTable::build(&counts, &values, false).unwrap();
        assert!(!table.fast.is_empty());
    }

    #[test]
    fn build_rejects_truncated_values() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0u8];
        let result = HuffmanTable::build(&counts, &values, true);
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
        assert!(HuffmanTable::build(&counts, &values, true).is_err());

        // Oversubscription deeper in the tree, past the fast-table width.
        let mut counts = [0u8; 16];
        counts[0] = 1;
        counts[1] = 3; // one 1-bit code leaves room for only two 2-bit codes
        let values = [0u8, 1, 2, 3];
        assert!(HuffmanTable::build(&counts, &values, true).is_err());

        // A full canonical code must still build: 2 one-bit codes is the limit.
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0u8, 1];
        assert!(HuffmanTable::build(&counts, &values, true).is_ok());
    }

    /// Minimal JPEG entropy encoder for round-trip tests: canonical codes for
    /// a caller-chosen DHT, MSB-first bit writer with FF stuffing.
    struct TestEncoder {
        /// (code, len) per DC symbol value.
        dc_codes: std::collections::HashMap<u8, (u32, u8)>,
        /// (code, len) per AC symbol value.
        ac_codes: std::collections::HashMap<u8, (u32, u8)>,
        bits: Vec<u8>,
        acc: u64,
        nbits: u32,
    }

    impl TestEncoder {
        fn canonical(counts: &[u8; 16], values: &[u8]) -> std::collections::HashMap<u8, (u32, u8)> {
            let mut codes = std::collections::HashMap::new();
            let mut code = 0u32;
            let mut si = 0usize;
            for (i, &count) in counts.iter().enumerate() {
                for _ in 0..count {
                    codes.insert(values[si], (code, (i + 1) as u8));
                    code += 1;
                    si += 1;
                }
                code <<= 1;
            }
            codes
        }

        fn new(
            dc_counts: &[u8; 16],
            dc_values: &[u8],
            ac_counts: &[u8; 16],
            ac_values: &[u8],
        ) -> Self {
            Self {
                dc_codes: Self::canonical(dc_counts, dc_values),
                ac_codes: Self::canonical(ac_counts, ac_values),
                bits: Vec::new(),
                acc: 0,
                nbits: 0,
            }
        }

        fn put_bits(&mut self, value: u32, n: u8) {
            for i in (0..n).rev() {
                self.acc = (self.acc << 1) | ((value >> i) & 1) as u64;
                self.nbits += 1;
                if self.nbits == 8 {
                    let b = self.acc as u8;
                    self.bits.push(b);
                    if b == 0xFF {
                        self.bits.push(0x00);
                    }
                    self.acc = 0;
                    self.nbits = 0;
                }
            }
        }

        fn put_dc_symbol(&mut self, symbol: u8) {
            let (code, len) = self.dc_codes[&symbol];
            self.put_bits(code, len);
        }

        fn put_ac_symbol(&mut self, symbol: u8) {
            let (code, len) = self.ac_codes[&symbol];
            self.put_bits(code, len);
        }

        /// Encode one AC coefficient as (run, value != 0).
        fn put_coeff(&mut self, run: u8, val: i32) {
            let size = (32 - (val.unsigned_abs()).leading_zeros()) as u8;
            self.put_ac_symbol((run << 4) | size);
            let mag = if val < 0 {
                (val + (1 << size) - 1) as u32
            } else {
                val as u32
            };
            self.put_bits(mag, size);
        }

        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                let pad = 8 - self.nbits;
                let b = ((self.acc << pad) | ((1u64 << pad) - 1)) as u8;
                self.bits.push(b);
                if b == 0xFF {
                    self.bits.push(0x00);
                }
            }
            self.bits
        }
    }

    /// DHT with 4 three-bit DC size codes and enough four-bit AC codes to
    /// cover the shapes the pair table bakes (small runs/sizes, ZRL, EOB) plus
    /// a size-8 symbol that must fall back to the single-coefficient probe.
    fn test_tables() -> (
        HuffmanTable,
        HuffmanTable,
        [u8; 16],
        Vec<u8>,
        [u8; 16],
        Vec<u8>,
    ) {
        let mut dc_counts = [0u8; 16];
        dc_counts[2] = 4; // 3-bit codes for DC sizes 0..=3
        let dc_values = vec![0u8, 1, 2, 3];
        let mut ac_counts = [0u8; 16];
        ac_counts[3] = 10; // 4-bit codes
        let ac_values = vec![0x00u8, 0x01, 0x02, 0x12, 0x21, 0xF0, 0xF1, 0xE1, 0xD1, 0x08];
        let dc = HuffmanTable::build_with_fast_bits(&dc_counts, &dc_values, 10, false).unwrap();
        let ac = HuffmanTable::build_with_fast_bits(&ac_counts, &ac_values, 10, true).unwrap();
        (dc, ac, dc_counts, dc_values, ac_counts, ac_values)
    }

    /// Encode blocks, decode them with `decode_block`, and compare the full
    /// coefficient array against a directly constructed expectation.
    /// Exercises the pair fast path, phantom singles, the block-boundary
    /// rewind (coefficient landing exactly on zigzag 63), ZRL, EOB, and the
    /// size-8 fallback — across consecutive blocks sharing one bit stream so
    /// any over-consume corrupts the following block and fails loudly.
    #[test]
    fn decode_block_roundtrip_pair_paths() {
        use crate::jpeg::types::ZIGZAG;
        let (dc, ac, dc_counts, dc_values, ac_counts, ac_values) = test_tables();

        // Each block: (dc_diff, &[(zero_run, value)], explicit_eob)
        // Zigzag positions fill as k = 1 + Σ(run_i + 1).
        type Block = (i32, Vec<(u8, i32)>, bool);
        let blocks: Vec<Block> = vec![
            // Adjacent small pairs (pair hits), then EOB.
            (2, vec![(0, 3), (0, -1), (0, 2), (1, -2)], true),
            // Single small coefficient (phantom), EOB.
            (-1, vec![(2, 1)], true),
            // DC-only block (EOB immediately).
            (0, vec![], true),
            // Coefficient landing exactly at zigzag 63: no EOB after —
            // the boundary rewind must not eat the next block's bits.
            (1, vec![(15, 1), (15, -1), (15, 1), (14, 1)], false),
            // Next block right after the boundary case: pairs again.
            (3, vec![(0, -3), (0, 3)], true),
            // ZRL then a coefficient, then a size-8 value (single-probe path).
            (0, vec![(15, 1), (0, 1), (0, 200)], true),
            // Dense run of 63 alternating values (max pair pressure).
            (
                -2,
                (0..63)
                    .map(|i| (0u8, if i % 2 == 0 { 1 } else { -1 }))
                    .collect(),
                false,
            ),
        ];

        let mut enc = TestEncoder::new(&dc_counts, &dc_values, &ac_counts, &ac_values);
        for (dc_diff, coeffs, eob) in &blocks {
            let size = (32 - dc_diff.unsigned_abs().leading_zeros()) as u8;
            enc.put_dc_symbol(size);
            if size > 0 {
                let mag = if *dc_diff < 0 {
                    (dc_diff + (1 << size) - 1) as u32
                } else {
                    *dc_diff as u32
                };
                enc.put_bits(mag, size);
            }
            let mut k = 1;
            for &(run, val) in coeffs {
                let mut r = run;
                while r >= 16 {
                    enc.put_ac_symbol(0xF0);
                    r -= 16;
                    k += 16;
                }
                enc.put_coeff(r, val);
                k += r as usize + 1;
            }
            if *eob {
                assert!(k < 64, "test case bug: EOB after full block");
                enc.put_ac_symbol(0x00);
            } else {
                assert_eq!(k, 64, "test case bug: unfinished block without EOB");
            }
        }
        let data = enc.finish();

        let mut bs = BitStream::new(&data, 0);
        let mut dc_pred = 0i32;
        let mut prev_has_ac = true;
        let mut coeffs = [0i16; 64];
        let mut expect_pred = 0i32;
        for (bi, (dc_diff, entries, _)) in blocks.iter().enumerate() {
            let info = decode_block::<true>(
                &mut bs,
                &dc,
                &ac,
                &mut coeffs,
                &mut dc_pred,
                &mut prev_has_ac,
            )
            .unwrap_or_else(|e| panic!("block {bi}: {e:?}"));
            expect_pred += dc_diff;
            let mut expected = [0i16; 64];
            expected[0] = expect_pred as i16;
            let mut k = 1usize;
            let mut expect_last = 0usize;
            for &(run, val) in entries {
                k += run as usize;
                expected[ZIGZAG[k] as usize] = val as i16;
                expect_last = k;
                k += 1;
            }
            assert_eq!(coeffs, expected, "block {bi} coefficients");
            assert_eq!(info.has_ac, !entries.is_empty(), "block {bi} has_ac");
            if info.has_ac {
                assert!(
                    (info.last_k as usize) >= expect_last,
                    "block {bi}: last_k {} < highest nonzero {}",
                    info.last_k,
                    expect_last
                );
            }
        }
        assert!(!bs.overran(), "stream over-consumed");
    }

    /// A fingerprint change must evict any table still parked in the slot:
    /// intern payload A, restore it, intern payload B (miss) with NO restore
    /// (as after a decode error), then intern B again — the hit must not hand
    /// back A's table under B's fingerprint.
    #[test]
    fn cache_rebuild_evicts_stale_table() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values_a = [0u8, 1];
        let values_b = [2u8, 3];
        let mut cache = HuffmanCache::default();

        let ta = cache.intern(1, 0, &counts, &values_a).unwrap();
        cache.restore(
            &mut [None, None, None, None],
            &mut [Some(ta), None, None, None],
        );

        // Payload change; the returned table is dropped without a restore.
        let _tb = cache.intern(1, 0, &counts, &values_b).unwrap();

        // Same payload again: must be a freshly built B table, not stale A.
        let tb2 = cache.intern(1, 0, &counts, &values_b).unwrap();
        let data = [0b0000_0000];
        let bs = BitStream::new(&data, 0);
        let mut cur = bs.cursor();
        assert_eq!(
            tb2.decode_symbol(&mut cur).unwrap(),
            2,
            "stale table for the previous DHT payload was returned"
        );
    }

    #[test]
    fn cache_hit_takes_and_restore_returns() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        let values = [0u8, 1];
        let mut cache = HuffmanCache::default();
        let t1 = cache.intern(0, 0, &counts, &values).unwrap();
        assert!(cache.dc[0].table.is_none());
        cache.restore(
            &mut [Some(t1), None, None, None],
            &mut [None, None, None, None],
        );
        assert!(cache.dc[0].table.is_some());

        let t2 = cache.intern(0, 0, &counts, &values).unwrap();
        assert!(cache.dc[0].table.is_none());
        // Same payload: intern must not rebuild (fingerprint still matches).
        cache.restore(
            &mut [Some(t2), None, None, None],
            &mut [None, None, None, None],
        );

        // Different symbols: miss, rebuild, then a subsequent matching intern hits.
        let values_miss = [0u8, 2];
        let t3 = cache.intern(0, 0, &counts, &values_miss).unwrap();
        cache.restore(
            &mut [Some(t3), None, None, None],
            &mut [None, None, None, None],
        );
        assert!(cache.dc[0].table.is_some());
        let _t4 = cache.intern(0, 0, &counts, &values_miss).unwrap();
        assert!(cache.dc[0].table.is_none());
    }
}
