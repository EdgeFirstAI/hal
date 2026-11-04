// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Fused int8 scaled dot-product attention: `softmax(Q·Kᵀ)·V` per head, over strided views.
//!
//! Keys and transposed values are packed once per call into the caller's
//! [`AttentionScratch`], in blocks of 4×4 bytes: keys as `[token/4][dim/4][4 tokens][4 dims]`,
//! values as `[dim/4][token/4][4 dims][4 tokens]`. A task takes four query rows of one head,
//! interleaved the same way, so each 16-byte block meets a query (or probability) block in one
//! indexed dot product whose four lanes are four tokens (or four value dims), with no
//! horizontal reductions. Zero points enter through sums, as in [`super::matmul_i8`]. CPUs
//! without FEAT_DotProd use the layout of `attention_i8_neon_base` instead.

use std::borrow::Cow;

use super::matmul_i8::{MatMulI8Params, MatMulI8Plan, OutPtr, Requant, TILE};
use super::softmax_i8::{SoftmaxI8Params, SoftmaxI8Plan};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::{for_each_block, Pool};
use crate::quant::{Quant, Rounding, TfliteKernels};
use crate::view::{TensorMut, TensorRef, View};

/// Most key/value tokens per head: each task keeps four rows of scores on the stack.
pub(crate) const ATTENTION_MAX_TOKENS: usize = 2048;

/// Longest query/key and value vectors per head.
pub(crate) const ATTENTION_MAX_HEAD_DIM: usize = 256;

/// The scalar `MUL` a TFLite graph applies to Q before the scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionQMul {
    /// The `MUL` output for every Q value, indexed by the raw byte; see
    /// [`crate::BinaryI8Plan::table_for_rhs`].
    pub table: [u8; 256],
    /// The quantization of the `MUL` output.
    pub output: Quant,
}

/// Parameters of an int8 attention block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionI8Params {
    pub q: Quant,
    /// An elementwise scale applied to Q first, if the graph has one.
    pub q_mul: Option<AttentionQMul>,
    pub k: Quant,
    pub v: Quant,
    /// The quantization of the int8 scores.
    pub scores: Quant,
    /// The softmax `beta`.
    pub beta: f32,
    /// The quantization of the int8 probabilities: scale 1/256, zero point −128.
    pub probs: Quant,
    pub output: Quant,
    /// The TFLite kernels whose `BATCH_MATMUL` rounding the two products reproduce.
    pub kernels: TfliteKernels,
}

/// Blocks of four rows (queries, keys or value tokens) per task of the NEON pack.
pub(super) const PACK_BLOCKS: usize = 16;

/// Block counts of one call's packed layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Blocks {
    /// Token blocks of four.
    pub(super) m4: usize,
    /// Key dim blocks of four.
    pub(super) dk4: usize,
    /// Value dim blocks of four.
    pub(super) dv4: usize,
    /// Query blocks of four.
    pub(super) n4: usize,
    /// Groups of four token blocks, and of four value dim blocks.
    pub(super) m4g: usize,
    pub(super) dv4g: usize,
}

impl Blocks {
    fn new(queries: usize, tokens: usize, dk: usize, dv: usize) -> Self {
        let (m4, dv4) = (tokens.div_ceil(4), dv.div_ceil(4));
        Self {
            m4,
            dk4: dk.div_ceil(4),
            dv4,
            n4: queries.div_ceil(4),
            m4g: m4.div_ceil(4),
            dv4g: dv4.div_ceil(4),
        }
    }

    /// Offset of head `h`'s 16-byte block `(token block jb, dim block dc)` of keys.
    ///
    /// Four token blocks are interleaved per dim block, so a group of 16 tokens is one
    /// stream of 64-byte steps.
    #[inline(always)]
    pub(super) fn key_block(&self, h: usize, jb: usize, dc: usize) -> usize {
        (((h * self.m4g + jb / 4) * self.dk4 + dc) * 4 + jb % 4) * 16
    }

    /// Offset of head `h`'s 16-byte block `(query block ib, dim block dc)` of queries.
    #[inline(always)]
    pub(super) fn query_block(&self, h: usize, ib: usize, dc: usize) -> usize {
        ((h * self.n4 + ib) * self.dk4 + dc) * 16
    }

    /// Offset of head `h`'s 16-byte block `(dim block eb, token block jc)` of values.
    ///
    /// Four dim blocks are interleaved per token block, so a group of 16 value dims is one
    /// stream of 64-byte steps.
    #[inline(always)]
    pub(super) fn value_block(&self, h: usize, eb: usize, jc: usize) -> usize {
        (((h * self.dv4g + eb / 4) * self.m4 + jc) * 4 + eb % 4) * 16
    }

    /// Index of token `j`'s key sum and bias in head `h`.
    #[inline(always)]
    pub(super) fn token(&self, h: usize, j: usize) -> usize {
        h * self.m4g * 16 + j
    }

    /// Index of value dim `e`'s sum and bias in head `h`.
    #[inline(always)]
    pub(super) fn value_dim(&self, h: usize, e: usize) -> usize {
        h * self.dv4g * 16 + e
    }
}

/// The zero points the packed biases fold in.
#[derive(Clone, Copy, Debug)]
pub(super) struct ZeroPoints {
    pub(super) q: i32,
    pub(super) k: i32,
    pub(super) v: i32,
}

/// Copies row `(h, i)` of a rank-3 view into `dst` (`dst.len()` elements).
pub(super) fn gather_row(view: &View<'_, i8>, h: usize, i: usize, dst: &mut [i8]) {
    let (data, s) = (view.data(), view.strides());
    let (s0, s1, s2) = (
        s[0].unsigned_abs(),
        s[1].unsigned_abs(),
        s[2].unsigned_abs(),
    );
    let base = h * s0 + i * s1;
    if s2 == 1 {
        if let Some(src) = data.get(base..base + dst.len()) {
            dst.copy_from_slice(src);
        }
    } else {
        for (d, o) in dst.iter_mut().enumerate() {
            *o = data.get(base + d * s2).copied().unwrap_or(0);
        }
    }
}

/// Which packed layout an [`AttentionScratch`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// The 4×4 blocks of the portable and dotprod kernels.
    Blocks,
    /// The layout of `attention_i8_neon_base`, for CPUs without FEAT_DotProd.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Widened,
}

/// Packed queries, keys and values for one call, owned by the caller so `run` does not
/// allocate. Build it with [`AttentionI8Plan::scratch`].
///
/// A scratch serves the plan that built it, and any plan with the same kernel, for any number
/// of calls, one at a time.
#[derive(Clone, Debug)]
pub struct AttentionScratch {
    pub(super) layout: Layout,
    pub(super) heads: usize,
    pub(super) queries: usize,
    pub(super) tokens: usize,
    pub(super) dk: usize,
    pub(super) dv: usize,
    pub(super) blocks: Blocks,
    pub(super) queries_packed: Vec<i8>,
    /// `[heads][n4 · 4]` `Σ_d q'`.
    pub(super) query_sums: Vec<i32>,
    pub(super) keys: Vec<i8>,
    /// `[heads][m4g · 16]` `Σ_d k`, zero past the last token.
    pub(super) key_sums: Vec<i32>,
    /// `[heads][m4g · 16]` `dk·zq·zk − zq·Σ_d k`: the scores' constant terms per token.
    pub(super) key_bias: Vec<i32>,
    pub(super) values: Vec<i8>,
    /// `[heads][dv4g · 16]` `Σ_j v`, zero past the last dim; empty in the widened layout.
    pub(super) value_sums: Vec<i32>,
    /// `[heads][dv4g · 16]` `128·Σ_j v − 128·tokens·zv`: the outputs' constant terms per dim;
    /// empty in the widened layout.
    pub(super) value_bias: Vec<i32>,
    /// `[heads][value tasks][dv]` per-task `Σ_j v` of the NEON block pack; empty otherwise.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) value_partials: Vec<i32>,
}

/// A zeroed vector of the product of `dims`, or an error when it overflows or cannot be
/// allocated.
fn zeroed<T: Copy + Default>(dims: &[usize], extra: usize) -> Result<Vec<T>> {
    let n = dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .and_then(|n| n.checked_add(extra))
        .ok_or(KernelError::InvalidArgument(
            "attention_i8 scratch size overflows",
        ))?;
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| {
        KernelError::Unsupported(Cow::Borrowed("attention_i8 scratch allocation failed"))
    })?;
    v.resize(n, T::default());
    Ok(v)
}

impl AttentionScratch {
    fn new(
        layout: Layout,
        heads: usize,
        queries: usize,
        tokens: usize,
        dk: usize,
        dv: usize,
    ) -> Result<Self> {
        AttentionI8Plan::check_shape(tokens, dk, dv)?;
        let b = Blocks::new(queries, tokens, dk, dv);
        let blocks = layout == Layout::Blocks;
        let neon_pack = blocks
            && cfg!(target_arch = "aarch64")
            && dk.is_multiple_of(16)
            && dv.is_multiple_of(16);
        Ok(Self {
            layout,
            heads,
            queries,
            tokens,
            dk,
            dv,
            blocks: b,
            // The dotprod kernel reads one step (16 bytes of rows, 64 of blocks) past the end.
            queries_packed: zeroed(&[heads, b.n4, b.dk4, 16], 16)?,
            query_sums: zeroed(&[heads, b.n4, 4], 0)?,
            keys: zeroed(&[heads, b.m4g, b.dk4, 64], 64)?,
            key_sums: zeroed(&[heads, b.m4g, 16], 0)?,
            key_bias: zeroed(&[heads, b.m4g, 16], 0)?,
            values: zeroed(&[heads, b.dv4g, b.m4, 64], 64)?,
            value_sums: zeroed(&[heads, b.dv4g, usize::from(blocks) * 16], 0)?,
            value_bias: zeroed(&[heads, b.dv4g, usize::from(blocks) * 16], 0)?,
            value_partials: zeroed(
                &[
                    heads,
                    b.m4.div_ceil(PACK_BLOCKS),
                    usize::from(neon_pack) * dv,
                ],
                0,
            )?,
        })
    }

    /// Packs `q` (through `table`), `k` and `v` with their sums and biases; padding stays
    /// zero from construction.
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
    fn pack(
        &mut self,
        q: &View<'_, i8>,
        table: Option<&[u8; 256]>,
        k: &View<'_, i8>,
        v: &View<'_, i8>,
        z: ZeroPoints,
        pool: Option<&dyn Pool>,
    ) {
        #[cfg(target_arch = "aarch64")]
        if self.dk.is_multiple_of(16) && self.dv.is_multiple_of(16) {
            // SAFETY: NEON is part of the aarch64 baseline; both dims are multiples of 16.
            unsafe { super::attention_i8_neon::pack(self, q, table, k, v, z, pool) };
            return;
        }
        let b = self.blocks;
        let mut row = [0i8; ATTENTION_MAX_HEAD_DIM];
        let dkzz = (self.dk as i32) * z.q * z.k;
        for h in 0..self.heads {
            for i in 0..self.queries {
                let Some(r) = row.get_mut(..self.dk) else {
                    return;
                };
                gather_row(q, h, i, r);
                if let Some(t) = table {
                    for x in r.iter_mut() {
                        *x = t[usize::from(*x as u8)] as i8;
                    }
                }
                let sum = r.iter().map(|&x| i32::from(x)).sum();
                if let Some(o) = self.query_sums.get_mut(h * b.n4 * 4 + i) {
                    *o = sum;
                }
                for (dc, c) in row.chunks_exact(4).take(b.dk4).enumerate() {
                    let at = b.query_block(h, i / 4, dc) + (i % 4) * 4;
                    if let Some(o) = self.queries_packed.get_mut(at..at + 4) {
                        o.copy_from_slice(c);
                    }
                }
            }
            for j in 0..b.m4g * 16 {
                let sum = if j < self.tokens {
                    let Some(r) = row.get_mut(..self.dk) else {
                        return;
                    };
                    gather_row(k, h, j, r);
                    for (dc, c) in row.chunks_exact(4).take(b.dk4).enumerate() {
                        let at = b.key_block(h, j / 4, dc) + (j % 4) * 4;
                        if let Some(o) = self.keys.get_mut(at..at + 4) {
                            o.copy_from_slice(c);
                        }
                    }
                    row.iter().take(self.dk).map(|&x| i32::from(x)).sum()
                } else {
                    0
                };
                let at = b.token(h, j);
                if let (Some(s), Some(bias)) =
                    (self.key_sums.get_mut(at), self.key_bias.get_mut(at))
                {
                    *s = sum;
                    *bias = dkzz - z.q * sum;
                }
            }
            let mut sums = [0i32; ATTENTION_MAX_HEAD_DIM + 16];
            let mut rows = [[0i8; ATTENTION_MAX_HEAD_DIM]; 4];
            for jc in 0..b.m4 {
                for (t, r) in rows.iter_mut().enumerate() {
                    let Some(r) = r.get_mut(..self.dv) else {
                        return;
                    };
                    let j = jc * 4 + t;
                    if j < self.tokens {
                        gather_row(v, h, j, r);
                    } else {
                        r.fill(0);
                    }
                }
                for eb in 0..b.dv4 {
                    let at = b.value_block(h, eb, jc);
                    let Some(block) = self.values.get_mut(at..at + 16) else {
                        return;
                    };
                    for c in 0..4 {
                        let e = eb * 4 + c;
                        for t in 0..4 {
                            let x = rows[t].get(e).copied().unwrap_or(0);
                            block[c * 4 + t] = x;
                            sums[e] += i32::from(x);
                        }
                    }
                }
            }
            let mzz = (self.tokens as i32) * (-128) * z.v;
            for (e, &sum) in sums.iter().enumerate().take(b.dv4g * 16) {
                let at = b.value_dim(h, e);
                if let (Some(s), Some(bias)) =
                    (self.value_sums.get_mut(at), self.value_bias.get_mut(at))
                {
                    *s = sum;
                    *bias = 128 * sum + mzz;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    NeonDotprod,
}

/// Fused int8 attention: for each head `h` and query `i`,
/// `out[h, i, :] = softmax(q'[h, i, :] · k[h, :, :]ᵀ) · v[h, :, :]`, where `q'` is Q through
/// the optional [`AttentionQMul`].
///
/// It replaces TFLite's `MUL` (on Q) → `BATCH_MATMUL` → `SOFTMAX` → `BATCH_MATMUL` chain and
/// the `TRANSPOSE`, `RESHAPE` and `SLICE` operators around it: Q, K and V are strided views
/// (for example head slices of one token-major tensor), the output is a strided view, and the
/// attention map is never materialized beyond four rows at a time.
///
/// It reproduces the chain's int8 roundings: the `MUL` table, the scores requantized as
/// TFLite's `BATCH_MATMUL`, the float-table `SOFTMAX` (as [`SoftmaxI8Plan`]), and the output
/// requantized as `BATCH_MATMUL`.
///
/// Each task keeps about 25 KB of buffers on the stack, and about 22 KB more on CPUs without
/// FEAT_DotProd.
#[derive(Clone, Debug)]
pub struct AttentionI8Plan {
    table: Option<[u8; 256]>,
    /// Zero point of the Q values the scores use (after the `MUL` table).
    pub(super) zq: i32,
    pub(super) zk: i32,
    pub(super) zv: i32,
    pub(super) zs: i32,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) zo: i32,
    pub(super) scores_requant: Requant,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) out_requant: Requant,
    pub(super) rounding: Rounding,
    pub(super) softmax: SoftmaxI8Plan,
    /// The second product, whose requantization finishes each output.
    out: MatMulI8Plan,
    kernel: Kernel,
}

fn check_quant(q: Quant) -> Result<()> {
    if !(q.scale.is_finite() && q.scale > 0.0) {
        return Err(KernelError::InvalidArgument(
            "attention_i8 scales must be finite and positive",
        ));
    }
    if !(-128..=127).contains(&q.zero_point) {
        return Err(KernelError::InvalidArgument(
            "attention_i8 zero points must lie in the int8 range",
        ));
    }
    Ok(())
}

/// One call's geometry, shared by its tasks.
pub(super) struct Geometry {
    pub(super) dk: usize,
    pub(super) dv: usize,
    pub(super) m: usize,
    pub(super) out_strides: [usize; 3],
}

/// Row length of the per-task buffers: scores are computed in whole groups of 16 tokens.
pub(super) const ROW: usize = ATTENTION_MAX_TOKENS + 16;

/// Per-task buffers. Every block writes the same prefixes, so the zero padding the dot
/// products read past them stays zero.
pub(super) struct Buffers {
    /// The int8 scores, then the softmax rows.
    pub(super) s8: [[i8; ROW]; TILE],
    pub(super) p8: [[i8; ROW]; TILE],
    /// Probabilities, `[m4][4 rows][4 tokens]`, zero point −128.
    /// 16 bytes longer than the largest use, for the dotprod kernel's read-ahead.
    pub(super) p: [i8; ROW * TILE + 64],
    /// `Σ_j p` of the raw values.
    pub(super) p_sums: [i32; TILE],
}

impl Buffers {
    fn new() -> Self {
        Self {
            s8: [[0; ROW]; TILE],
            p8: [[0; ROW]; TILE],
            p: [0; ROW * TILE + 64],
            p_sums: [0; TILE],
        }
    }
}

impl AttentionI8Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: AttentionI8Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows; reads no environment.
    pub fn with_features(p: AttentionI8Params, f: CpuFeatures) -> Result<Self> {
        for q in [p.q, p.k, p.v, p.scores, p.probs, p.output] {
            check_quant(q)?;
        }
        if let Some(m) = p.q_mul {
            check_quant(m.output)?;
        }
        let q_scaled = p.q_mul.map_or(p.q, |m| m.output);
        // Validates the probabilities' quantization (1/256, −128) and beta.
        let softmax = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: p.scores,
                output: p.probs,
                beta: p.beta,
            },
            f,
        )?;
        // The two products, validated as the `BATCH_MATMUL`s they replace.
        let scores = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_batch_matmul(q_scaled, p.k, p.scores, p.kernels)?,
            CpuFeatures::default(),
        )?;
        let out = MatMulI8Plan::with_features(
            MatMulI8Params::tflite_batch_matmul(p.probs, p.v, p.output, p.kernels)?,
            CpuFeatures::default(),
        )?;
        Ok(Self {
            table: p.q_mul.map(|m| m.table),
            zq: scores.zl,
            zk: scores.zr,
            zv: out.zr,
            zs: scores.zo,
            zo: out.zo,
            scores_requant: scores.column(0).0,
            out_requant: out.column(0).0,
            rounding: scores.rounding,
            softmax,
            out,
            kernel: select_kernel(f),
        })
    }

    /// `Ok` when this crate's attention kernels take `tokens` key/value tokens per head and
    /// `dk`-long queries and keys and `dv`-long values; `Unsupported` otherwise (the kernels
    /// keep rows of scores and vectors in fixed stack buffers).
    pub fn check_shape(tokens: usize, dk: usize, dv: usize) -> Result<()> {
        if tokens > ATTENTION_MAX_TOKENS
            || dk > ATTENTION_MAX_HEAD_DIM
            || dv > ATTENTION_MAX_HEAD_DIM
        {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "attention_i8 takes at most 2048 tokens and head dims of 256",
            )));
        }
        Ok(())
    }

    /// Scratch for this plan's kernel: `heads` heads of `queries` queries attending over
    /// `tokens` key/value tokens, with `dk`-long queries and keys and `dv`-long values.
    ///
    /// Fails as [`AttentionI8Plan::check_shape`] does, and when the sizes overflow or cannot
    /// be allocated.
    pub fn scratch(
        &self,
        heads: usize,
        queries: usize,
        tokens: usize,
        dk: usize,
        dv: usize,
    ) -> Result<AttentionScratch> {
        AttentionScratch::new(self.layout(), heads, queries, tokens, dk, dv)
    }

    fn layout(&self) -> Layout {
        match self.kernel {
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Layout::Widened,
            _ => Layout::Blocks,
        }
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        match self.kernel {
            Kernel::Portable => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Tier::Neon,
            #[cfg(target_arch = "aarch64")]
            Kernel::NeonDotprod => Tier::NeonDotprod,
        }
    }

    /// Attention of `q` `[heads, queries, dk]` over `k` `[heads, tokens, dk]` and `v`
    /// `[heads, tokens, dv]` into `output` `[heads, queries, dv]`, all i8 views with any
    /// strides; `scratch` must have been built for these heads, tokens, `dk` and `dv`.
    ///
    /// Work is split across `pool` when one is given. The output must not overlap itself.
    /// Arguments are validated before anything is written.
    pub fn run(
        &self,
        q: TensorRef<'_>,
        k: TensorRef<'_>,
        v: TensorRef<'_>,
        output: TensorMut<'_>,
        scratch: &mut AttentionScratch,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span =
            tracing::trace_span!("kernels.attention_i8", tier = self.tier().as_str()).entered();
        let (TensorRef::I8(q), TensorRef::I8(k), TensorRef::I8(v), TensorMut::I8(mut out)) =
            (q, k, v, output)
        else {
            return Err(KernelError::InvalidArgument("attention_i8 needs i8 views"));
        };
        let (Ok(&[h, n, dk]), Ok(&[hk, m, dk2]), Ok(&[hv, m2, dv]), Ok(&[ho, n2, dv2])) = (
            <&[usize; 3]>::try_from(q.shape()),
            <&[usize; 3]>::try_from(k.shape()),
            <&[usize; 3]>::try_from(v.shape()),
            <&[usize; 3]>::try_from(out.shape()),
        ) else {
            return Err(KernelError::InvalidArgument(
                "attention_i8 views need rank 3: [heads, tokens, dim]",
            ));
        };
        if hk != h || hv != h || ho != h || dk2 != dk || m2 != m || n2 != n || dv2 != dv {
            return Err(KernelError::InvalidArgument(
                "attention_i8 shapes do not match",
            ));
        }
        if (
            scratch.heads,
            scratch.queries,
            scratch.tokens,
            scratch.dk,
            scratch.dv,
        ) != (h, n, m, dk, dv)
        {
            return Err(KernelError::InvalidArgument(
                "attention_i8 scratch was built for other shapes",
            ));
        }
        if scratch.layout != self.layout() {
            return Err(KernelError::InvalidArgument(
                "attention_i8 scratch was built for a plan with another kernel",
            ));
        }
        super::rows::check_no_overlap(out.shape(), out.strides())?;
        if h == 0 || n == 0 || dv == 0 || m == 0 {
            return Ok(());
        }
        let z = ZeroPoints {
            q: self.zq,
            k: self.zk,
            v: self.zv,
        };
        match self.kernel {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is baseline; the views are the ones validated above.
            Kernel::Neon => unsafe {
                super::attention_i8_neon_base::pack(
                    scratch,
                    &q,
                    self.table.as_ref(),
                    &k,
                    &v,
                    z,
                    pool,
                );
            },
            _ => scratch.pack(&q, self.table.as_ref(), &k, &v, z, pool),
        }
        let mut out_strides = [0usize; 3];
        for (d, &s) in out_strides.iter_mut().zip(out.strides()) {
            *d = s.unsigned_abs();
        }
        let g = Geometry {
            dk,
            dv,
            m,
            out_strides,
        };
        let dst = OutPtr(out.data_mut().as_mut_ptr());
        let scratch: &AttentionScratch = scratch;
        let row_blocks = n.div_ceil(TILE);
        for_each_block(h * row_blocks, pool, &|range| {
            let mut bufs = Buffers::new();
            for t in range {
                let (head, first) = (t / row_blocks, (t % row_blocks) * TILE);
                let rows = first..(first + TILE).min(n);
                match self.kernel {
                    Kernel::Portable => {
                        self.block_portable(&g, scratch, dst, head, rows, &mut bufs)
                    }
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is baseline; the rows are this task's alone.
                    Kernel::Neon => unsafe {
                        super::attention_i8_neon_base::block(
                            self, &g, scratch, dst, head, rows, &mut bufs,
                        );
                    },
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: as above; selected only with FEAT_DotProd.
                    Kernel::NeonDotprod => unsafe {
                        super::attention_i8_neon::block(
                            self, &g, scratch, dst, head, rows, &mut bufs,
                        );
                    },
                }
            }
        });
        Ok(())
    }

    /// The scalar pipeline for one block of query rows.
    fn block_portable(
        &self,
        g: &Geometry,
        s: &AttentionScratch,
        dst: OutPtr,
        h: usize,
        rows: core::ops::Range<usize>,
        bufs: &mut Buffers,
    ) {
        let b = s.blocks;
        let count = rows.len();
        let ib = rows.start / 4;
        let qv = |r: usize, d: usize| {
            i32::from(s.queries_packed[b.query_block(h, ib, d / 4) + r * 4 + d % 4])
        };
        let qsum = |r: usize| s.query_sums[h * b.n4 * 4 + rows.start + r];
        let dkzz = (g.dk as i32) * self.zq * self.zk;
        for r in 0..count {
            for j in 0..g.m {
                let mut dot = 0i32;
                for d in 0..g.dk {
                    let kv = s.keys[b.key_block(h, j / 4, d / 4) + (j % 4) * 4 + d % 4];
                    dot += qv(r, d) * i32::from(kv);
                }
                let ksum = s.key_sums[b.token(h, j)];
                let acc = dot - self.zk * qsum(r) - self.zq * ksum + dkzz;
                bufs.s8[r][j] = (self.zs + self.scores_requant.apply(acc, self.rounding))
                    .clamp(-128, 127) as i8;
            }
            let (s8, p8) = (&bufs.s8[r][..g.m], &mut bufs.p8[r][..g.m]);
            self.softmax.row(s8, p8);
            let mut sum = 0i32;
            for j in 0..g.m {
                let p = bufs.p8[r][j];
                bufs.p[(j / 4) * 16 + r * 4 + j % 4] = p;
                sum += i32::from(p);
            }
            bufs.p_sums[r] = sum;
        }
        let mzz = (g.m as i32) * (-128) * self.zv;
        let [oh, os1, os2] = g.out_strides;
        for (r, i) in rows.enumerate() {
            for e in 0..g.dv {
                let mut dot = 0i32;
                for j in 0..g.m {
                    let vv = s.values[b.value_block(h, e / 4, j / 4) + (e % 4) * 4 + j % 4];
                    dot += i32::from(bufs.p[(j / 4) * 16 + r * 4 + j % 4]) * i32::from(vv);
                }
                let vsum = s.value_sums[b.value_dim(h, e)];
                let acc = dot - self.zv * bufs.p_sums[r] + 128 * vsum + mzz;
                let q = self.output_value(acc);
                // SAFETY: (h, i, e) lies in the validated output view and only this task
                // writes query row `i` of head `h`.
                unsafe { dst.0.add(h * oh + i * os1 + e * os2).write(q) };
            }
        }
    }

    /// The output byte of an exact accumulator `Σ_j (p + 128)(v − zv)`.
    #[inline(always)]
    pub(super) fn output_value(&self, acc: i32) -> i8 {
        self.out.finish(acc, 0)
    }
}

#[cfg(target_arch = "aarch64")]
fn select_kernel(f: CpuFeatures) -> Kernel {
    if f.dotprod_usable() {
        Kernel::NeonDotprod
    } else if f.neon_baseline {
        Kernel::Neon
    } else {
        Kernel::Portable
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn select_kernel(_: CpuFeatures) -> Kernel {
    Kernel::Portable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::binary_i8::{BinaryI8Params, BinaryI8Plan, BinaryOp};
    use crate::ops::matmul_i8::{MatMulI8Params, MatMulI8Plan};
    use crate::quant::{Activation, TfliteKernels};
    use crate::view::ViewMut;

    /// Heads, tokens (queries attend over as many keys), and the query/key and value dims.
    #[derive(Clone, Copy, Debug)]
    struct Shape {
        h: usize,
        n: usize,
        dk: usize,
        dv: usize,
    }

    impl Shape {
        /// Channels per head in the token-major input: q, k, v.
        fn c(self) -> usize {
            2 * self.dk + self.dv
        }
    }

    /// yolo26n's first attention block.
    const YOLO26N: Shape = Shape {
        h: 2,
        n: 400,
        dk: 32,
        dv: 64,
    };

    fn tiers() -> Vec<CpuFeatures> {
        let mut v = vec![CpuFeatures::default()];
        for t in ["neon", "neon_dotprod"] {
            if let Ok(f) = CpuFeatures::from_forced_tier(Some(t)) {
                v.push(f);
            }
        }
        v
    }

    fn input(sh: Shape, seed: u32) -> Vec<i8> {
        let mut s = seed;
        (0..sh.n * sh.h * sh.c())
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // Values spread like a layer output: mostly small, some large.
                let v = (s % 97) as i32 - 48 + ((s >> 8) % 3) as i32 * ((s >> 12) % 40) as i32;
                v.clamp(-128, 127) as i8
            })
            .collect()
    }

    /// yolo26n's first attention block.
    fn params() -> (AttentionI8Params, BinaryI8Params) {
        let x = Quant::new(0.1251, -15);
        let mul = BinaryI8Params {
            op: BinaryOp::Mul,
            lhs: x,
            rhs: Quant::new(0.000_693_2, -128),
            output: Quant::new(0.010_18, 6),
            activation: Activation::None,
            kernels: TfliteKernels::NeonRuy,
        };
        let table = BinaryI8Plan::new(mul)
            .unwrap()
            .table_for_rhs(127, 16)
            .unwrap();
        let p = AttentionI8Params {
            q: x,
            q_mul: Some(AttentionQMul {
                table,
                output: mul.output,
            }),
            k: x,
            v: x,
            scores: Quant::new(0.1279, -9),
            beta: 1.0,
            probs: Quant::new(1.0 / 256.0, -128),
            output: Quant::new(0.063_33, 2),
            kernels: TfliteKernels::NeonRuy,
        };
        (p, mul)
    }

    /// q, k, v views of head slices of the token-major `[n, h·c]` input.
    fn views(sh: Shape, x: &[i8]) -> (View<'_, i8>, View<'_, i8>, View<'_, i8>) {
        let st = [sh.c() as isize, (sh.h * sh.c()) as isize, 1];
        (
            View::strided(x, &[sh.h, sh.n, sh.dk], &st).unwrap(),
            View::strided(&x[sh.dk..], &[sh.h, sh.n, sh.dk], &st).unwrap(),
            View::strided(&x[2 * sh.dk..], &[sh.h, sh.n, sh.dv], &st).unwrap(),
        )
    }

    /// The token-major `[n, h·dv]` output as `[h, n, dv]`.
    fn out_view(sh: Shape, out: &mut [i8]) -> ViewMut<'_, i8> {
        let st = [sh.dv as isize, (sh.h * sh.dv) as isize, 1];
        ViewMut::strided(out, &[sh.h, sh.n, sh.dv], &st).unwrap()
    }

    fn run_attention(sh: Shape, p: AttentionI8Params, f: CpuFeatures, x: &[i8]) -> Vec<i8> {
        let plan = AttentionI8Plan::with_features(p, f).unwrap();
        let (q, k, v) = views(sh, x);
        let mut out = vec![0i8; sh.n * sh.h * sh.dv];
        let mut scratch = plan.scratch(sh.h, sh.n, sh.n, sh.dk, sh.dv).unwrap();
        plan.run(
            TensorRef::I8(q),
            TensorRef::I8(k),
            TensorRef::I8(v),
            TensorMut::I8(out_view(sh, &mut out)),
            &mut scratch,
            Some(&crate::ScopedThreads::new(
                core::num::NonZeroUsize::new(3).unwrap(),
            )),
        )
        .unwrap();
        out
    }

    /// The TFLite chain, op by op, with this crate's single-op plans.
    fn run_chain(sh: Shape, p: AttentionI8Params, f: CpuFeatures, x: &[i8]) -> Vec<i8> {
        let Shape { h, n, dk, dv } = sh;
        let c = sh.c();
        let qm = p.q_mul.unwrap();
        let (_, k, v) = views(sh, x);
        let mut q2 = vec![0i8; h * n * dk];
        for hh in 0..h {
            for i in 0..n {
                for d in 0..dk {
                    let raw = x[i * h * c + hh * c + d];
                    q2[(hh * n + i) * dk + d] = qm.table[usize::from(raw as u8)] as i8;
                }
            }
        }
        let bmm1 =
            MatMulI8Params::tflite_batch_matmul(qm.output, p.k, p.scores, p.kernels).unwrap();
        let bmm1 = MatMulI8Plan::with_features(bmm1, f).unwrap();
        let mut s = vec![0i8; h * n * n];
        let kt = View::strided(k.data(), &[h, dk, n], &[c as isize, 1, (h * c) as isize]).unwrap();
        bmm1.run(
            TensorRef::I8(View::contiguous(&q2, &[h, n, dk]).unwrap()),
            TensorRef::I8(kt),
            TensorMut::I8(ViewMut::contiguous(&mut s, &[h, n, n]).unwrap()),
            None,
        )
        .unwrap();
        let sm = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: p.scores,
                output: p.probs,
                beta: p.beta,
            },
            f,
        )
        .unwrap();
        let mut pr = vec![0i8; h * n * n];
        sm.run(
            TensorRef::I8(View::contiguous(&s, &[h, n, n]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut pr, &[h, n, n]).unwrap()),
            None,
        )
        .unwrap();
        let bmm2 = MatMulI8Params::tflite_batch_matmul(p.probs, p.v, p.output, p.kernels).unwrap();
        let bmm2 = MatMulI8Plan::with_features(bmm2, f).unwrap();
        let mut out = vec![0i8; n * h * dv];
        bmm2.run(
            TensorRef::I8(View::contiguous(&pr, &[h, n, n]).unwrap()),
            TensorRef::I8(v),
            TensorMut::I8(out_view(sh, &mut out)),
            None,
        )
        .unwrap();
        out
    }

    #[test]
    fn equals_the_tflite_chain_on_every_tier() {
        let x = input(YOLO26N, 7);
        let (p, _) = params();
        for f in tiers() {
            assert_eq!(
                run_attention(YOLO26N, p, f, &x),
                run_chain(YOLO26N, p, f, &x),
                "{:?}",
                f.tier()
            );
        }
    }

    /// Odd step counts (11 token steps, 5 dim steps), partial value tiles and query blocks.
    #[test]
    fn odd_shapes_equal_the_tflite_chain() {
        let (p, _) = params();
        for sh in [
            Shape {
                h: 1,
                n: 41,
                dk: 20,
                dv: 24,
            },
            Shape {
                h: 3,
                n: 37,
                dk: 12,
                dv: 9,
            },
            Shape {
                h: 2,
                n: 6,
                dk: 4,
                dv: 17,
            },
            Shape {
                h: 3,
                n: 77,
                dk: 16,
                dv: 48,
            },
        ] {
            let x = input(sh, 11);
            for f in tiers() {
                assert_eq!(
                    run_attention(sh, p, f, &x),
                    run_chain(sh, p, f, &x),
                    "{sh:?} {:?}",
                    f.tier()
                );
            }
        }
    }

    /// Every query aligned with the `hot` tokens' keys and opposed to the others', so the
    /// scores saturate and the probability splits evenly over the hot tokens; every other value
    /// dim at 127, so a sum of probabilities past 257 overflows 16 bits.
    fn hot_keys(sh: Shape, hot: impl Fn(usize) -> bool) -> Vec<i8> {
        let mut x = input(sh, 5);
        let c = sh.c();
        for i in 0..sh.n {
            for hh in 0..sh.h {
                let at = i * sh.h * c + hh * c;
                x[at..at + sh.dk].fill(100);
                x[at + sh.dk..at + 2 * sh.dk].fill(if hot(i) { 100 } else { -100 });
                for e in (0..sh.dv).step_by(2) {
                    x[at + 2 * sh.dk + e] = 127;
                }
            }
        }
        x
    }

    /// Rows whose probabilities sum past 257, which the A53 kernel splits: evenly over every
    /// token (one each), over six adjacent tokens (43 each, within one group of 16), across a
    /// group boundary, and spread; with a partial last block of rows and value dims in a pair
    /// of 16-dim groups and a single one.
    #[test]
    fn rounded_up_probabilities_equal_the_tflite_chain() {
        let (p, _) = params();
        type Hot = fn(usize) -> bool;
        let hot: [(&str, Hot); 4] = [
            ("all", |_| true),
            ("six in a group", |i| (40..46).contains(&i)),
            ("six across groups", |i| (13..19).contains(&i)),
            ("six spread", |i| i % 50 == 7),
        ];
        for sh in [
            Shape {
                h: 2,
                n: 301,
                dk: 32,
                dv: 48,
            },
            Shape {
                h: 1,
                n: 77,
                dk: 20,
                dv: 9,
            },
        ] {
            for (name, hot) in hot {
                let x = hot_keys(sh, hot);
                for f in tiers() {
                    assert_eq!(
                        run_attention(sh, p, f, &x),
                        run_chain(sh, p, f, &x),
                        "{sh:?} {name} {:?}",
                        f.tier()
                    );
                }
            }
        }
    }

    /// The reference rounding, on the 32-dim path and a streamed one, and a long row.
    #[test]
    fn other_roundings_and_lengths_equal_the_tflite_chain() {
        let (p, _) = params();
        let reference = AttentionI8Params {
            kernels: TfliteKernels::Reference,
            ..p
        };
        for (sh, p) in [
            (YOLO26N, reference),
            (
                Shape {
                    h: 1,
                    n: 50,
                    dk: 64,
                    dv: 32,
                },
                reference,
            ),
            (
                Shape {
                    h: 1,
                    n: 1100,
                    dk: 32,
                    dv: 64,
                },
                p,
            ),
        ] {
            let x = input(sh, 13);
            for f in tiers() {
                assert_eq!(
                    run_attention(sh, p, f, &x),
                    run_chain(sh, p, f, &x),
                    "{sh:?} {:?} {:?}",
                    p.kernels,
                    f.tier()
                );
            }
        }
    }

    /// A scratch serves every plan whose kernel packs its layout, in any order, and is
    /// rejected, before anything is written, by a plan whose kernel packs another.
    #[test]
    fn a_scratch_serves_plans_with_its_layout() {
        for sh in [
            Shape {
                h: 2,
                n: 37,
                dk: 12,
                dv: 24,
            },
            YOLO26N,
        ] {
            let x = input(sh, 17);
            let (p, _) = params();
            let (q, k, v) = views(sh, &x);
            let tiers = tiers();
            for a in &tiers {
                let owner = AttentionI8Plan::with_features(p, *a).unwrap();
                let mut scratch = owner.scratch(sh.h, sh.n, sh.n, sh.dk, sh.dv).unwrap();
                for f in tiers.iter().chain(tiers.iter().rev()) {
                    let plan = AttentionI8Plan::with_features(p, *f).unwrap();
                    let mut out = vec![5i8; sh.n * sh.h * sh.dv];
                    let r = plan.run(
                        TensorRef::I8(q),
                        TensorRef::I8(k),
                        TensorRef::I8(v),
                        TensorMut::I8(out_view(sh, &mut out)),
                        &mut scratch,
                        None,
                    );
                    if plan.layout() == owner.layout() {
                        r.unwrap();
                        assert_eq!(out, run_attention(sh, p, *f, &x), "{:?}", f.tier());
                    } else {
                        assert!(r.is_err(), "{:?} on {:?}", f.tier(), a.tier());
                        assert!(out.iter().all(|&o| o == 5));
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_mismatched_scratch_and_shapes() {
        let Shape { h, n, dk, dv } = YOLO26N;
        let x = input(YOLO26N, 3);
        let (p, _) = params();
        let plan = AttentionI8Plan::new(p).unwrap();
        let (q, k, v) = views(YOLO26N, &x);
        let mut out = vec![5i8; n * h * dv];
        let mut scratch = plan.scratch(h, n, n - 1, dk, dv).unwrap();
        let r = plan.run(
            TensorRef::I8(q),
            TensorRef::I8(k),
            TensorRef::I8(v),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[h, n, dv]).unwrap()),
            &mut scratch,
            None,
        );
        assert!(r.is_err());
        assert!(out.iter().all(|&o| o == 5));
        for (tokens, dk, dv) in [
            (ATTENTION_MAX_TOKENS + 1, 8, 8),
            (8, ATTENTION_MAX_HEAD_DIM + 1, 8),
            (8, 8, ATTENTION_MAX_HEAD_DIM + 1),
        ] {
            assert!(AttentionI8Plan::check_shape(tokens, dk, dv).is_err());
            assert!(plan.scratch(1, 4, tokens, dk, dv).is_err());
        }
        assert!(AttentionI8Plan::check_shape(ATTENTION_MAX_TOKENS, 256, 256).is_ok());
        assert!(plan.scratch(usize::MAX, usize::MAX, 8, 8, 8).is_err());
        assert!(plan.scratch(usize::MAX / 2, 4, 8, 8, 8).is_err());
    }
}
