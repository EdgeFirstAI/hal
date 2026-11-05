// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON attention for CPUs with FEAT_DotProd: indexed dot products over the packed blocks of
//! [`super::attention_i8`]; other CPUs use [`super::attention_i8_neon_base`].
//!
//! `sdot v.4s, block.16b, rows.4b[r]` adds, to each of four lanes, the dot product of four
//! bytes of the block (one token's four dims, or one value dim's four tokens) with row `r`'s
//! four bytes. Four blocks are interleaved per 64-byte step, so a 4×4 tile of accumulators
//! (16 tokens or value dims by four query rows) reads one stream. The step loop is
//! hand-scheduled for in-order cores such as the Cortex-A55. The accumulators start at the
//! zero-point terms, so they end as the exact `Σ (a − za)(b − zb)`; only the int8 softmax rows
//! differ from the portable kernel (±1 LSB, as [`super::softmax_i8::SoftmaxI8Plan`] on NEON).

#![cfg(target_arch = "aarch64")]

use core::ops::Range;
use std::arch::aarch64::*; // NOSONAR

use super::attention_i8::{
    gather_row, AttentionI8Plan, AttentionScratch, Blocks, Buffers, Geometry, ZeroPoints,
    ATTENTION_MAX_HEAD_DIM, PACK_BLOCKS,
};
use super::lut_i8_neon::{lookup, table_blocks};
use super::matmul_i8::{OutPtr, TILE};
use super::matmul_i8_neon::{narrow16, requant4, sum_i8};
use crate::pool::{for_each_block, Pool};
use crate::prim::transpose::interleave4;
use crate::view::View;

/// A 4×4 tile of accumulators: `acc[t][r]`, block `t`, query row `r`.
type Tile = [[int32x4_t; TILE]; 4];

/// One step of [`dot_tile`]'s loop: the 16 `sdot`s of the row vector in `$r` against the four
/// blocks, with the next step's row vector loaded into `$nx` and its first block into `v1`.
macro_rules! dot_step {
    ($r:literal, $nx:literal) => {
        concat!(
            "sdot v16.4s, v1.16b, ",
            $r,
            ".4b[0]\n",
            "ldr d2, [{pb}, #16]\n",
            "sdot v17.4s, v1.16b, ",
            $r,
            ".4b[1]\n",
            "ldr x9, [{pb}, #24]\n",
            "sdot v18.4s, v1.16b, ",
            $r,
            ".4b[2]\n",
            "ins v2.d[1], x9\n",
            "sdot v19.4s, v1.16b, ",
            $r,
            ".4b[3]\n",
            "ldr d3, [{pb}, #32]\n",
            "sdot v20.4s, v2.16b, ",
            $r,
            ".4b[0]\n",
            "ldr x9, [{pb}, #40]\n",
            "sdot v21.4s, v2.16b, ",
            $r,
            ".4b[1]\n",
            "ins v3.d[1], x9\n",
            "sdot v22.4s, v2.16b, ",
            $r,
            ".4b[2]\n",
            "ldr d4, [{pb}, #48]\n",
            "sdot v23.4s, v2.16b, ",
            $r,
            ".4b[3]\n",
            "ldr x9, [{pb}, #56]\n",
            "sdot v24.4s, v3.16b, ",
            $r,
            ".4b[0]\n",
            "ins v4.d[1], x9\n",
            "sdot v25.4s, v3.16b, ",
            $r,
            ".4b[1]\n",
            "ldr d",
            $nx,
            ", [{pr}, #16]!\n",
            "sdot v26.4s, v3.16b, ",
            $r,
            ".4b[2]\n",
            "ldr x9, [{pr}, #8]\n",
            "sdot v27.4s, v3.16b, ",
            $r,
            ".4b[3]\n",
            "ins v",
            $nx,
            ".d[1], x9\n",
            "sdot v28.4s, v4.16b, ",
            $r,
            ".4b[0]\n",
            "ldr d1, [{pb}, #64]!\n",
            "sdot v29.4s, v4.16b, ",
            $r,
            ".4b[1]\n",
            "ldr x9, [{pb}, #8]\n",
            "sdot v30.4s, v4.16b, ",
            $r,
            ".4b[2]\n",
            "ins v1.d[1], x9\n",
            "sdot v31.4s, v4.16b, ",
            $r,
            ".4b[3]\n",
        )
    };
}

/// `acc[t][r] += Σ_s block(t, s) · rows(s)[lane r]` over `steps` steps, block `(t, s)` at
/// `blocks + 64·s + 16·t` and the row vector of step `s` at `rows + 16·s`.
///
/// Each 128-bit operand is loaded as a 64-bit `ldr d` and a general-register
/// `ldr x` merged with `ins`, which issue alongside the `sdot`s, each just before its first
/// use; the next step's row vector and first block load after their last use in this one. Steps
/// go in pairs with the row vector alternating between two registers. That runs at about
/// 1.0 cycle per `sdot` on a Cortex-A55, against about 1.7 with `ldr q`.
///
/// # Safety
/// FEAT_DotProd; every block and row of steps `0..steps` readable, and one more step past the
/// last (16 bytes of rows, 64 of blocks).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot_tile(mut acc: Tile, rows: *const i8, blocks: *const i8, steps: usize) -> Tile {
    if steps == 0 {
        return acc;
    }
    let [[mut a0, mut a1, mut a2, mut a3], [mut a4, mut a5, mut a6, mut a7], [mut a8, mut a9, mut a10, mut a11], [mut a12, mut a13, mut a14, mut a15]] =
        acc;
    // SAFETY: the caller guarantees FEAT_DotProd and every address the loop reads.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod",
            "ldr d0, [{pr}]",
            "ldr x9, [{pr}, #8]",
            "ins v0.d[1], x9",
            "ldr d1, [{pb}]",
            "ldr x9, [{pb}, #8]",
            "ins v1.d[1], x9",
            "cbz {pairs}, 3f",
            "2:",
            dot_step!("v0", "5"),
            dot_step!("v5", "0"),
            "subs {pairs}, {pairs}, #1",
            "b.ne 2b",
            "3:",
            "cbz {odd}, 4f",
            dot_step!("v0", "5"),
            "4:",
            pr = inout(reg) rows => _,
            pb = inout(reg) blocks => _,
            pairs = inout(reg) steps / 2 => _,
            odd = in(reg) steps % 2,
            out("x9") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            inout("v16") a0, inout("v17") a1, inout("v18") a2, inout("v19") a3,
            inout("v20") a4, inout("v21") a5, inout("v22") a6, inout("v23") a7,
            inout("v24") a8, inout("v25") a9, inout("v26") a10, inout("v27") a11,
            inout("v28") a12, inout("v29") a13, inout("v30") a14, inout("v31") a15,
            options(nostack, readonly),
        );
    }
    acc = [
        [a0, a1, a2, a3],
        [a4, a5, a6, a7],
        [a8, a9, a10, a11],
        [a12, a13, a14, a15],
    ];
    acc
}

///
/// # Safety
/// `bias` must be readable for 16 i32.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn tile_init(bias: *const i32, row: [int32x4_t; TILE]) -> Tile {
    // SAFETY: the caller guarantees 16 readable i32.
    let b: [int32x4_t; 4] = core::array::from_fn(|t| unsafe { vld1q_s32(bias.add(4 * t)) });
    core::array::from_fn(|t| core::array::from_fn(|r| vaddq_s32(b[t], row[r])))
}

/// One block of query rows `rows` of head `h`.
///
/// # Safety
/// The CPU must support FEAT_DotProd. `dst` must be the output view's
/// base, the geometry validated against the views, the scratch packed for this call, and no
/// other task may write these rows.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines)]
pub(super) unsafe fn block(
    p: &AttentionI8Plan,
    g: &Geometry,
    s: &AttentionScratch,
    dst: OutPtr,
    h: usize,
    rows: Range<usize>,
    bufs: &mut Buffers,
) {
    let b = s.blocks;
    let count = rows.len();
    let m = g.m;
    let ib = rows.start / 4;
    // SAFETY: block (h, ib, 0) and the next dk4 blocks lie in the packed queries.
    let queries = unsafe { s.queries_packed.as_ptr().add(b.query_block(h, ib, 0)) };
    let qsum: [i32; TILE] = core::array::from_fn(|r| {
        s.query_sums
            .get(h * b.n4 * 4 + ib * 4 + r)
            .copied()
            .unwrap_or(0)
    });

    // Scores, 16 tokens per tile; the accumulators start at the zero-point terms.
    let row_term: [int32x4_t; TILE] = core::array::from_fn(|r| vdupq_n_s32(-p.zk * qsum[r]));
    let zs = vdupq_n_s32(p.zs);
    for jg in 0..b.m4g {
        let j0 = jg * 16;
        // SAFETY: group `jg` of head `h` (and, on dotprod CPUs, one step past it) lies in the
        // padded keys; 16 biases of the group lie in `key_bias`.
        let acc = unsafe {
            let init = tile_init(s.key_bias.as_ptr().add(b.token(h, j0)), row_term);
            dot_tile(
                init,
                queries,
                s.keys.as_ptr().add(b.key_block(h, jg * 4, 0)),
                b.dk4,
            )
        };
        for (r, row) in bufs.s8.iter_mut().enumerate() {
            let q: [int32x4_t; 4] = core::array::from_fn(|t| {
                vaddq_s32(requant4(acc[t][r], p.scores_requant, p.rounding), zs)
            });
            // SAFETY: `j0 + 16 <= m4g · 16 <= ROW`.
            unsafe { vst1q_s8(row.as_mut_ptr().add(j0), narrow16(q)) };
        }
    }

    // Probabilities, row by row.
    for r in 0..count {
        let (Some(s8), Some(p8)) = (bufs.s8[r].get(..m), bufs.p8[r].get_mut(..m)) else {
            continue;
        };
        p.softmax.row(s8, p8);
        let sum = sum_i8(bufs.p8[r].get(..m).unwrap_or(&[]));
        bufs.p_sums[r] = sum;
    }
    // Token block jc of row r is the 32-bit word jc of row r; transposing 4×4 words gives
    // four interleaved blocks. Rows past `count` and tokens past `m` hold leftovers, which
    // meet zero values or go unstored.
    for g4 in 0..b.m4g {
        // SAFETY: `16 · (g4 + 1) <= ROW` bytes in each row and `64 · (g4 + 1)` within `p`.
        unsafe {
            let w: [int32x4_t; TILE] = core::array::from_fn(|r| {
                vreinterpretq_s32_s8(vld1q_s8(bufs.p8[r].as_ptr().add(g4 * 16)))
            });
            let t0 = vreinterpretq_s64_s32(vtrn1q_s32(w[0], w[1]));
            let t1 = vreinterpretq_s64_s32(vtrn2q_s32(w[0], w[1]));
            let t2 = vreinterpretq_s64_s32(vtrn1q_s32(w[2], w[3]));
            let t3 = vreinterpretq_s64_s32(vtrn2q_s32(w[2], w[3]));
            let out = bufs.p.as_mut_ptr().add(g4 * 64);
            vst1q_s8(out, vreinterpretq_s8_s64(vtrn1q_s64(t0, t2)));
            vst1q_s8(out.add(16), vreinterpretq_s8_s64(vtrn1q_s64(t1, t3)));
            vst1q_s8(out.add(32), vreinterpretq_s8_s64(vtrn2q_s64(t0, t2)));
            vst1q_s8(out.add(48), vreinterpretq_s8_s64(vtrn2q_s64(t1, t3)));
        }
    }

    // Outputs, 16 value dims per tile; the accumulators start at the zero-point terms.
    let [oh, os1, os2] = g.out_strides;
    let probs = bufs.p.as_ptr();
    let row_term: [int32x4_t; TILE] = core::array::from_fn(|r| vdupq_n_s32(-p.zv * bufs.p_sums[r]));
    let zo = vdupq_n_s32(p.zo);
    for eg in 0..b.dv4g {
        let e0 = eg * 16;
        // SAFETY: group `eg` of head `h` (and one step past it) lies in the padded values;
        // `bufs.p` holds `m4` steps and one more; 16 biases lie in `value_bias`.
        let acc = unsafe {
            let init = tile_init(s.value_bias.as_ptr().add(b.value_dim(h, e0)), row_term);
            dot_tile(
                init,
                probs,
                s.values.as_ptr().add(b.value_block(h, eg * 4, 0)),
                b.m4,
            )
        };
        let lanes = g.dv.saturating_sub(e0).min(16);
        for (r, i) in rows.clone().enumerate() {
            let q: [int32x4_t; 4] = core::array::from_fn(|t| {
                vaddq_s32(requant4(acc[t][r], p.out_requant, p.rounding), zo)
            });
            let v = narrow16(q);
            let base = h * oh + i * os1 + e0 * os2;
            if os2 == 1 && lanes == 16 {
                // SAFETY: 16 contiguous output bytes of row `i`, this task's alone.
                unsafe { vst1q_s8(dst.0.add(base), v) };
            } else {
                let mut tmp = [0i8; 16];
                // SAFETY: 16 writable bytes.
                unsafe { vst1q_s8(tmp.as_mut_ptr(), v) };
                for (c, &x) in tmp.iter().enumerate().take(lanes) {
                    // SAFETY: (h, i, e0 + c) lies in the validated output view and only this
                    // task writes query row `i` of head `h`.
                    unsafe { dst.0.add(base + c * os2).write(x) };
                }
            }
        }
    }
}

/// Four staged rows of `dk` bytes as dim blocks: block `dc`, at `dst + dc·stride`, holds bytes
/// `4dc..4dc + 4` of each row in turn. Returns each row's sum.
///
/// # Safety
/// `dk` must be a multiple of 16 and at most [`ATTENTION_MAX_HEAD_DIM`]; `dst + dc·stride` must be
/// writable for 16 bytes for every `dc < dk / 4`.
#[target_feature(enable = "neon")]
unsafe fn words4(rows: &Rows, dk: usize, dst: *mut i8, stride: usize) -> [i32; 4] {
    let mut acc = vdupq_n_s32(0);
    for g in 0..dk / 16 {
        // SAFETY: `16·g + 16 <= dk` bytes of each row (caller).
        let r: [int32x4_t; 4] = core::array::from_fn(|t| unsafe {
            vreinterpretq_s32_s8(vld1q_s8(rows[t].as_ptr().add(16 * g)))
        });
        let t0 = vreinterpretq_s64_s32(vtrn1q_s32(r[0], r[1]));
        let t1 = vreinterpretq_s64_s32(vtrn2q_s32(r[0], r[1]));
        let t2 = vreinterpretq_s64_s32(vtrn1q_s32(r[2], r[3]));
        let t3 = vreinterpretq_s64_s32(vtrn2q_s32(r[2], r[3]));
        let blocks = [
            vtrn1q_s64(t0, t2),
            vtrn1q_s64(t1, t3),
            vtrn2q_s64(t0, t2),
            vtrn2q_s64(t1, t3),
        ];
        for (u, blk) in blocks.into_iter().enumerate() {
            let blk = vreinterpretq_s8_s64(blk);
            // SAFETY: block `4g + u` is writable (caller).
            unsafe { vst1q_s8(dst.add((4 * g + u) * stride), blk) };
            // Lane `t` gathers row `t`'s four bytes of the block.
            acc = vpadalq_s16(acc, vpaddlq_s8(blk));
        }
    }
    [
        vgetq_lane_s32::<0>(acc),
        vgetq_lane_s32::<1>(acc),
        vgetq_lane_s32::<2>(acc),
        vgetq_lane_s32::<3>(acc),
    ]
}

/// Up to eight rows of a view staged for packing.
pub(super) type Rows = [[i8; ATTENTION_MAX_HEAD_DIM]; 8];

/// Which operand a pack task packs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Part {
    Queries,
    Keys,
    Values,
}

/// What a pack task runs: its part, head and index within the part, with rows to stage into.
pub(super) type PackTask = unsafe fn(&PackJob<'_>, Part, usize, usize, &mut Rows);

/// One call's packing, split per head into query, key and value tasks; each task writes only its
/// own part of the scratch buffers, which outlive the job.
pub(super) struct PackJob<'a> {
    pub(super) b: Blocks,
    q: &'a View<'a, i8>,
    k: &'a View<'a, i8>,
    v: &'a View<'a, i8>,
    table_blocks: Option<[uint8x16x4_t; 4]>,
    heads: usize,
    pub(super) queries: usize,
    pub(super) tokens: usize,
    pub(super) dk: usize,
    pub(super) dv: usize,
    pub(super) dkzz: i32,
    pub(super) zq: i32,
    /// Query, key and value tasks per head.
    pub(super) tasks: [usize; 3],
    pub(super) queries_packed: *mut i8,
    pub(super) query_sums: *mut i32,
    pub(super) keys: *mut i8,
    pub(super) key_sums: *mut i32,
    pub(super) key_bias: *mut i32,
    pub(super) values: *mut i8,
    pub(super) value_partials: *mut i32,
}

// SAFETY: tasks write disjoint parts of the buffers (each kernel's task function documents its
// part), which outlive them; the views and table are only read.
unsafe impl Sync for PackJob<'_> {}

impl<'a> PackJob<'a> {
    #[target_feature(enable = "neon")]
    pub(super) fn new(
        s: &mut AttentionScratch,
        q: &'a View<'a, i8>,
        table: Option<&'a [u8; 256]>,
        k: &'a View<'a, i8>,
        v: &'a View<'a, i8>,
        z: ZeroPoints,
        tasks: [usize; 3],
    ) -> Self {
        Self {
            b: s.blocks,
            q,
            k,
            v,
            table_blocks: table.map(|t| table_blocks::<4>(t)),
            heads: s.heads,
            queries: s.queries,
            tokens: s.tokens,
            dk: s.dk,
            dv: s.dv,
            dkzz: (s.dk as i32) * z.q * z.k,
            zq: z.q,
            tasks,
            queries_packed: s.queries_packed.as_mut_ptr(),
            query_sums: s.query_sums.as_mut_ptr(),
            keys: s.keys.as_mut_ptr(),
            key_sums: s.key_sums.as_mut_ptr(),
            key_bias: s.key_bias.as_mut_ptr(),
            values: s.values.as_mut_ptr(),
            value_partials: s.value_partials.as_mut_ptr(),
        }
    }

    /// Runs `task` for every task of every head, on `pool`.
    ///
    /// # Safety
    /// `task` must write only its own task's part of the buffers, within the scratch's lengths.
    pub(super) unsafe fn run(&self, pool: Option<&dyn Pool>, task: PackTask) {
        let [tq, tk, tv] = self.tasks;
        let per_head = tq + tk + tv;
        for_each_block(self.heads * per_head, pool, &|range| {
            let mut rows = [[0i8; ATTENTION_MAX_HEAD_DIM]; 8];
            for t in range {
                let (h, t) = (t / per_head, t % per_head);
                let (part, c) = if t < tq {
                    (Part::Queries, t)
                } else if t < tq + tk {
                    (Part::Keys, t - tq)
                } else {
                    (Part::Values, t - tq - tk)
                };
                // SAFETY: `for_each_block` hands each task to one call (caller's contract).
                unsafe { task(self, part, h, c, &mut rows) };
            }
        });
    }

    /// Stages rows `first..first + n` (`n <= 8`) of head `h` of `part`'s view into `rows`:
    /// each row's bytes up to the next multiple of 16 past its length, zero past the view's
    /// rows and past its length; queries go through the table.
    #[target_feature(enable = "neon")]
    pub(super) fn stage(&self, part: Part, rows: &mut Rows, n: usize, h: usize, first: usize) {
        let (view, count, len) = match part {
            Part::Queries => (self.q, self.queries, self.dk),
            Part::Keys => (self.k, self.tokens, self.dk),
            Part::Values => (self.v, self.tokens, self.dv),
        };
        let padded = len.next_multiple_of(16).min(ATTENTION_MAX_HEAD_DIM);
        let table = match part {
            Part::Queries => self.table_blocks.as_ref(),
            Part::Keys | Part::Values => None,
        };
        for (t, row) in rows.iter_mut().take(n).enumerate() {
            let row = &mut row[..padded];
            if first + t >= count {
                row.fill(0);
                continue;
            }
            gather_row(view, h, first + t, &mut row[..len]);
            if let Some(blocks) = table {
                // The whole padded row goes through the table; the padding is zeroed after.
                for c in (0..padded).step_by(16) {
                    // SAFETY: bytes `c..c + 16 <= padded` of `row`.
                    unsafe {
                        let p = row.as_mut_ptr().add(c).cast::<u8>();
                        vst1q_u8(p, lookup(blocks, vld1q_u8(p)));
                    }
                }
            }
            row[len..].fill(0);
        }
    }
}

/// Items `t·per..(t + 1)·per` of `n`.
pub(super) fn task_range(t: usize, per: usize, n: usize) -> Range<usize> {
    (t * per).min(n)..((t + 1) * per).min(n)
}

/// One task of [`pack`].
///
/// # Safety
/// As [`PackJob::run`]: query task `c` writes query blocks `16c..16c + 16` and their sums, key
/// task `c` key blocks `16c..16c + 16` and their sums and biases, value task `c` token blocks
/// `16c..16c + 16` and partial-sum row `c`, all of head `h`.
#[target_feature(enable = "neon")]
unsafe fn block_task(job: &PackJob<'_>, part: Part, h: usize, c: usize, rows: &mut Rows) {
    let b = job.b;
    // SAFETY: every write is to task `c` of head `h`'s part (above), inside the lengths `pack`
    // checked.
    unsafe {
        match part {
            Part::Queries => {
                for ib in task_range(c, PACK_BLOCKS, b.n4) {
                    job.stage(part, rows, 4, h, 4 * ib);
                    let at = job.queries_packed.add(b.query_block(h, ib, 0));
                    for (r, &sum) in words4(rows, job.dk, at, 16).iter().enumerate() {
                        if 4 * ib + r < job.queries {
                            *job.query_sums.add(h * b.n4 * 4 + 4 * ib + r) = sum;
                        }
                    }
                }
            }
            Part::Keys => {
                for jb in task_range(c, PACK_BLOCKS, b.m4g * 4) {
                    job.stage(part, rows, 4, h, 4 * jb);
                    let at = job.keys.add(b.key_block(h, jb, 0));
                    for (r, &sum) in words4(rows, job.dk, at, 64).iter().enumerate() {
                        let at = b.token(h, 4 * jb + r);
                        *job.key_sums.add(at) = sum;
                        *job.key_bias.add(at) = job.dkzz - job.zq * sum;
                    }
                }
            }
            Part::Values => {
                let dv = job.dv;
                let mut acc = [vdupq_n_s32(0); ATTENTION_MAX_HEAD_DIM / 4];
                for jc in task_range(c, PACK_BLOCKS, b.m4) {
                    job.stage(part, rows, 4, h, 4 * jc);
                    for g in 0..dv / 16 {
                        let r: [int8x16_t; 4] =
                            core::array::from_fn(|t| vld1q_s8(rows[t].as_ptr().add(16 * g)));
                        // Dim-major blocks of four tokens each.
                        let blocks = interleave4(r);
                        vst1q_s8_x4(
                            job.values.add(b.value_block(h, 4 * g, jc)),
                            int8x16x4_t(blocks[0], blocks[1], blocks[2], blocks[3]),
                        );
                        for (u, blk) in blocks.into_iter().enumerate() {
                            if let Some(a) = acc.get_mut(4 * g + u) {
                                *a = vpadalq_s16(*a, vpaddlq_s8(blk));
                            }
                        }
                    }
                }
                let partial = job.value_partials.add((h * job.tasks[2] + c) * dv);
                for (e4, a) in acc.iter().enumerate().take(dv / 4) {
                    vst1q_s32(partial.add(4 * e4), *a);
                }
            }
        }
    }
}

/// [`AttentionScratch::pack`] for `dk` and `dv` multiples of 16: four rows at a time, staged and
/// transposed into blocks in registers, the sums from pairwise adds of the blocks. Writes the
/// same bytes and sums as the scalar pack, padding included.
///
/// Each head splits into tasks of [`PACK_BLOCKS`] query, key or value-token blocks, run on
/// `pool`; value tasks leave per-task dim sums that are added up afterwards.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does; `q`, `k`, `v` must be the views `s`
/// was built for, `s` sized by its constructor for the block layout with `dk` and `dv`
/// multiples of 16.
#[target_feature(enable = "neon")]
pub(super) unsafe fn pack(
    s: &mut AttentionScratch,
    q: &View<'_, i8>,
    table: Option<&[u8; 256]>,
    k: &View<'_, i8>,
    v: &View<'_, i8>,
    z: ZeroPoints,
    pool: Option<&dyn Pool>,
) {
    let b = s.blocks;
    let (heads, dk, dv) = (s.heads, s.dk, s.dv);
    let tasks = [
        b.n4.div_ceil(PACK_BLOCKS),
        (b.m4g * 4).div_ceil(PACK_BLOCKS),
        b.m4.div_ceil(PACK_BLOCKS),
    ];
    debug_assert!(
        dk % 16 == 0
            && dv % 16 == 0
            && dk <= ATTENTION_MAX_HEAD_DIM
            && dv <= ATTENTION_MAX_HEAD_DIM
    );
    debug_assert!(
        s.queries_packed.len() >= heads * b.n4 * dk * 4
            && s.query_sums.len() >= heads * b.n4 * 4
            && s.keys.len() >= heads * b.m4g * dk * 16
            && s.key_sums.len() >= heads * b.m4g * 16
            && s.key_bias.len() >= heads * b.m4g * 16
            && s.values.len() >= heads * b.dv4g * b.m4 * 64
            && s.value_partials.len() >= heads * tasks[2] * dv
            && s.value_sums.len() >= heads * b.dv4g * 16
    );
    let job = PackJob::new(s, q, table, k, v, z, tasks);
    // SAFETY: `block_task` writes only its own task's part, inside the scratch's lengths.
    unsafe { job.run(pool, block_task) };
    let tv = tasks[2];
    let mzz = (s.tokens as i32) * (-128) * z.v;
    for h in 0..heads {
        for e in 0..dv {
            let sum = (0..tv)
                .map(|c| {
                    s.value_partials
                        .get((h * tv + c) * dv + e)
                        .copied()
                        .unwrap_or(0)
                })
                .sum::<i32>();
            let at = b.value_dim(h, e);
            if let (Some(o), Some(bias)) = (s.value_sums.get_mut(at), s.value_bias.get_mut(at)) {
                *o = sum;
                *bias = 128 * sum + mzz;
            }
        }
    }
}
