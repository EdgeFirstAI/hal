// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Row and chunk walking shared by the plans, split across a pool.

use core::marker::PhantomData;
use core::ops::Range;

use crate::error::{KernelError, Result};
use crate::pool::{for_each_block, Pool};
use crate::view::{View, ViewMut, MAX_RANK};

/// Longest strided row that is staged through a stack buffer; longer ones go element-wise.
pub(crate) const STAGE: usize = 4096;

/// A strided input row, read element by element.
#[derive(Clone, Copy)]
pub(crate) struct StridedRow<'a, T> {
    data: &'a [T],
    offset: usize,
    stride: usize,
    len: usize,
}

impl<T: Copy + Default> StridedRow<'_, T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The row's elements in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = T> + Clone + '_ {
        (0..self.len).map(|k| {
            self.data
                .get(self.offset + k * self.stride)
                .copied()
                .unwrap_or_default()
        })
    }
}

/// A strided output row, written element by element; writes past `len` are ignored.
pub(crate) struct StridedRowMut<'a, T> {
    base: RowsPtr<T>,
    offset: usize,
    stride: usize,
    len: usize,
    _row: PhantomData<&'a mut [T]>,
}

impl<T> StridedRowMut<'_, T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, k: usize, v: T) {
        if k < self.len {
            // SAFETY: element `k < len` of this row is inside the output view and only this row owns it.
            unsafe { self.base.get().add(self.offset + k * self.stride).write(v) };
        }
    }
}

/// The output base pointer, shared by the workers of one call.
struct RowsPtr<T>(*mut T);

impl<T> Clone for RowsPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for RowsPtr<T> {}

// SAFETY: rows never overlap (checked) and each block runs once; a safe `Pool` cannot keep the task past `run`, so the caller sees every write.
unsafe impl<T: Send> Sync for RowsPtr<T> {}

impl<T> RowsPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Rejects a view whose axes, sorted by stride, do not each step past the extent of the smaller ones.
///
/// This guarantees no element is addressed twice; it also rejects interleaved layouts that do not alias.
pub(crate) fn check_no_overlap(shape: &[usize], strides: &[isize]) -> Result<()> {
    if shape.contains(&0) {
        return Ok(());
    }
    let mut axes = [(usize::MAX, 1usize); MAX_RANK];
    let mut slots = axes.iter_mut();
    for (&d, &s) in shape.iter().zip(strides) {
        if d > 1 {
            if let Some(slot) = slots.next() {
                *slot = (s.unsigned_abs(), d);
            }
        }
    }
    axes.sort_unstable();
    let mut extent = 1usize;
    for (s, d) in axes {
        if s < extent {
            return Err(KernelError::InvalidArgument("output view strides overlap"));
        }
        extent = s.saturating_mul(d - 1).saturating_add(extent);
    }
    Ok(())
}

/// Element offset of row `r` given the outer shape and the per-axis strides.
fn row_offset(mut r: usize, outer: &[usize], strides: &[usize]) -> usize {
    let mut off = 0usize;
    for (&d, &s) in outer.iter().zip(strides).rev() {
        let d = d.max(1);
        off += (r % d) * s;
        r /= d;
    }
    off
}

fn unsigned_strides(strides: &[isize]) -> [usize; MAX_RANK] {
    let mut out = [0usize; MAX_RANK];
    for (d, &s) in out.iter_mut().zip(strides) {
        *d = s.unsigned_abs();
    }
    out
}

/// Row geometry of a non-contiguous call.
struct Geometry<'a> {
    outer: &'a [usize],
    in_strides: &'a [usize],
    out_strides: &'a [usize],
    n_in: usize,
    n_out: usize,
}

/// Calls `row(input_row, output_row)` for every last-axis row; rows may run concurrently on `pool`.
///
/// Input and output must have the same rank (≥ 1) and outer shape; their last axes give the
/// per-row lengths, which may differ. The output must pass [`check_no_overlap`]. Everything is
/// validated before the first write. Rows of length 0 are skipped. Strided rows are staged
/// through stack buffers when they fit in [`STAGE`] elements; longer strided rows call
/// `strided` instead, with element-wise accessors.
pub(crate) fn for_each_row<I, O>(
    input: &View<'_, I>,
    output: &mut ViewMut<'_, O>,
    pool: Option<&dyn Pool>,
    row: &(dyn Fn(&[I], &mut [O]) + Sync),
    strided: &(dyn Fn(StridedRow<'_, I>, &mut StridedRowMut<'_, O>) + Sync),
) -> Result<()>
where
    I: Copy + Default + Sync,
    O: Copy + Default + Send,
{
    let (Some((&n_in, outer)), Some((&n_out, out_outer))) =
        (input.shape().split_last(), output.shape().split_last())
    else {
        return Err(KernelError::InvalidArgument("views need rank >= 1"));
    };
    if outer != out_outer {
        return Err(KernelError::InvalidArgument(
            "input and output outer shapes differ",
        ));
    }
    check_no_overlap(output.shape(), output.strides())?;
    if n_in == 0 || n_out == 0 || outer.contains(&0) {
        return Ok(());
    }
    let rows = outer
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or(KernelError::InvalidArgument("row count overflows"))?;

    if let (Some(src), Some(dst)) = (input.as_contiguous(), output.as_contiguous_mut()) {
        if Some(src.len()) != rows.checked_mul(n_in) || Some(dst.len()) != rows.checked_mul(n_out) {
            return Err(KernelError::InvalidArgument("view length mismatch"));
        }
        let out = RowsPtr(dst.as_mut_ptr());
        for_each_block(rows, pool, &|r: Range<usize>| {
            if r.end > rows {
                return;
            }
            let Some(src) = src.get(r.start * n_in..r.end * n_in) else {
                return;
            };
            let (lo, len) = (r.start * n_out, (r.end - r.start) * n_out);
            // SAFETY: `lo..lo + len` lies in `dst` and is this block's alone.
            let dst = unsafe { core::slice::from_raw_parts_mut(out.get().add(lo), len) };
            for (s, d) in src.chunks_exact(n_in).zip(dst.chunks_exact_mut(n_out)) {
                row(s, d);
            }
        });
        return Ok(());
    }

    let rank = input.rank();
    let (in_strides, out_strides) = (
        unsigned_strides(input.strides()),
        unsigned_strides(output.strides()),
    );
    let geom = Geometry {
        outer,
        in_strides: in_strides.get(..rank).unwrap_or(&[]),
        out_strides: out_strides.get(..rank).unwrap_or(&[]),
        n_in,
        n_out,
    };
    let src = input.data();
    let out = RowsPtr(output.data_mut().as_mut_ptr());
    for_each_block(rows, pool, &|r: Range<usize>| {
        if r.end <= rows {
            strided_block(r, &geom, src, out, row, strided);
        }
    });
    Ok(())
}

fn strided_block<I: Copy + Default, O: Copy + Default>(
    rows: Range<usize>,
    g: &Geometry<'_>,
    src: &[I],
    out: RowsPtr<O>,
    row: &(dyn Fn(&[I], &mut [O]) + Sync),
    strided: &(dyn Fn(StridedRow<'_, I>, &mut StridedRowMut<'_, O>) + Sync),
) {
    let (Some(&s_in), Some(&s_out)) = (g.in_strides.last(), g.out_strides.last()) else {
        return;
    };
    let (n_in, n_out) = (g.n_in, g.n_out);
    let in_unit = s_in == 1 || n_in == 1;
    let out_unit = s_out == 1 || n_out == 1;
    if (!in_unit && n_in > STAGE) || (!out_unit && n_out > STAGE) {
        for r in rows {
            let input = StridedRow {
                data: src,
                offset: row_offset(r, g.outer, g.in_strides),
                stride: s_in,
                len: n_in,
            };
            let mut output = StridedRowMut {
                base: out,
                offset: row_offset(r, g.outer, g.out_strides),
                stride: s_out,
                len: n_out,
                _row: PhantomData,
            };
            strided(input, &mut output);
        }
        return;
    }
    // Stage buffers exist, and are zeroed, only on the side that needs them.
    let mut in_store;
    let in_buf: &mut [I] = if in_unit {
        &mut []
    } else {
        in_store = [I::default(); STAGE];
        in_store.get_mut(..n_in).unwrap_or(&mut [])
    };
    let mut out_store;
    let out_buf: &mut [O] = if out_unit {
        &mut []
    } else {
        out_store = [O::default(); STAGE];
        out_store.get_mut(..n_out).unwrap_or(&mut [])
    };
    for r in rows {
        let ioff = row_offset(r, g.outer, g.in_strides);
        let ooff = row_offset(r, g.outer, g.out_strides);
        let row_in: &[I] = if in_unit {
            src.get(ioff..ioff + n_in).unwrap_or(&[])
        } else {
            for (k, b) in in_buf.iter_mut().enumerate() {
                *b = src.get(ioff + k * s_in).copied().unwrap_or_default();
            }
            in_buf
        };
        if row_in.len() != n_in {
            continue;
        }
        if out_unit {
            // SAFETY: the row's `n_out` elements are inside the output view and only this row owns them.
            let dst = unsafe { core::slice::from_raw_parts_mut(out.get().add(ooff), n_out) };
            row(row_in, dst);
        } else {
            row(row_in, out_buf);
            for (k, &v) in out_buf.iter().enumerate() {
                // SAFETY: element `k < n_out` of this row is inside the output view and only this row owns it.
                unsafe { out.get().add(ooff + k * s_out).write(v) };
            }
        }
    }
}

/// The segment, first row and row count of block `b` when each segment splits into blocks of
/// `max_rows` from its own start.
fn locate_block(
    segments: &[usize],
    max_rows: usize,
    mut b: usize,
) -> Option<(usize, usize, usize)> {
    let mut first = 0usize;
    for (seg, &n) in segments.iter().enumerate() {
        let blocks = n.div_ceil(max_rows);
        if b < blocks {
            let off = b * max_rows;
            return Some((seg, first + off, (n - off).min(max_rows)));
        }
        b -= blocks;
        first += n;
    }
    None
}

/// Copies rows `first..first + len` of a rank-2 view with strides `(s0, s1)` into `buf`,
/// row-major, walking whichever axis is closer together innermost.
fn gather_rows<I: Copy + Default>(
    src: &[I],
    first: usize,
    len: usize,
    n_in: usize,
    s0: usize,
    s1: usize,
    buf: &mut [I],
) {
    if s1 <= s0 {
        for r in 0..len {
            let base = (first + r) * s0;
            for k in 0..n_in {
                if let Some(b) = buf.get_mut(r * n_in + k) {
                    *b = src.get(base + k * s1).copied().unwrap_or_default();
                }
            }
        }
    } else {
        for k in 0..n_in {
            let base = first * s0 + k * s1;
            for r in 0..len {
                if let Some(b) = buf.get_mut(r * n_in + k) {
                    *b = src.get(base + r * s0).copied().unwrap_or_default();
                }
            }
        }
    }
}

/// A [`for_each_row_block`] callback: `(segment, first row, input rows, output rows)`.
pub(crate) type RowBlockFn<'a, I, O> = dyn Fn(usize, usize, &[I], &mut [O]) + Sync + 'a;

/// Calls `block(segment, first_row, input_rows, output_rows)` on runs of at most `max_rows`
/// consecutive rows of two rank-2 views; blocks may run concurrently on `pool`.
///
/// `segments` are row counts, in order, that must add up to the row count. Each block lies
/// inside one segment and starts at a multiple of `max_rows` from the segment's start, whatever
/// the pool. A block's input arrives as one row-major `rows × n_in` slice and its output as
/// `rows × n_out`; non-contiguous sides are staged through stack buffers, so `max_rows · n_in`
/// and `max_rows · n_out` must fit [`STAGE`]. The output must pass [`check_no_overlap`].
/// Everything is validated before the first write. `block` must write its whole output slice.
pub(crate) fn for_each_row_block<I, O>(
    input: &View<'_, I>,
    output: &mut ViewMut<'_, O>,
    pool: Option<&dyn Pool>,
    segments: &[usize],
    max_rows: usize,
    block: &RowBlockFn<'_, I, O>,
) -> Result<()>
where
    I: Copy + Default + Sync,
    O: Copy + Default + Send,
{
    let (&[rows, n_in], &[out_rows, n_out]) = (input.shape(), output.shape()) else {
        return Err(KernelError::InvalidArgument("row blocks need rank-2 views"));
    };
    if rows != out_rows {
        return Err(KernelError::InvalidArgument(
            "input and output row counts differ",
        ));
    }
    if segments.iter().try_fold(0usize, |a, &n| a.checked_add(n)) != Some(rows) {
        return Err(KernelError::InvalidArgument(
            "segments do not add up to the row count",
        ));
    }
    let fits = |n: usize| max_rows.checked_mul(n).is_some_and(|v| v <= STAGE);
    if max_rows == 0 || !fits(n_in) || !fits(n_out) {
        return Err(KernelError::InvalidArgument(
            "row block does not fit the stage buffer",
        ));
    }
    check_no_overlap(output.shape(), output.strides())?;
    if rows == 0 || n_out == 0 {
        return Ok(());
    }
    let n_blocks: usize = segments.iter().map(|&n| n.div_ceil(max_rows)).sum();
    let [s0, s1, ..] = unsigned_strides(input.strides());
    let [o0, o1, ..] = unsigned_strides(output.strides());
    let src_dense = input.as_contiguous();
    let src = input.data();
    let out_dense = output.as_contiguous_mut().is_some();
    let data = output.data_mut();
    let cap = data.len();
    let out = RowsPtr(data.as_mut_ptr());

    for_each_block(n_blocks, pool, &|range: Range<usize>| {
        // Stage buffers exist, and are zeroed, only on the side that needs them.
        let mut in_store;
        let in_buf: &mut [I] = if src_dense.is_some() {
            &mut []
        } else {
            in_store = [I::default(); STAGE];
            &mut in_store
        };
        let mut out_store;
        let out_buf: &mut [O] = if out_dense {
            &mut []
        } else {
            out_store = [O::default(); STAGE];
            &mut out_store
        };
        for b in range {
            let Some((seg, first, len)) = locate_block(segments, max_rows, b) else {
                continue;
            };
            let (n_src, n_dst) = (len * n_in, len * n_out);
            let rows_in: &[I] = match src_dense {
                Some(s) => s.get(first * n_in..first * n_in + n_src).unwrap_or(&[]),
                None => {
                    gather_rows(src, first, len, n_in, s0, s1, in_buf);
                    in_buf.get(..n_src).unwrap_or(&[])
                }
            };
            if rows_in.len() != n_src {
                continue;
            }
            if out_dense {
                let lo = first * n_out;
                if lo + n_dst > cap {
                    continue;
                }
                // SAFETY: rows `first..first + len` of the dense output are `lo..lo + n_dst`, and
                // only this block writes them.
                let dst = unsafe { core::slice::from_raw_parts_mut(out.get().add(lo), n_dst) };
                block(seg, first, rows_in, dst);
            } else {
                let Some(buf) = out_buf.get_mut(..n_dst) else {
                    continue;
                };
                block(seg, first, rows_in, buf);
                for (r, row) in buf.chunks_exact(n_out).enumerate() {
                    let base = (first + r) * o0;
                    for (k, &v) in row.iter().enumerate() {
                        let off = base + k * o1;
                        if off < cap {
                            // SAFETY: `off` addresses an element of a row of this block, which
                            // only this block writes.
                            unsafe { out.get().add(off).write(v) };
                        }
                    }
                }
            }
        }
    });
    Ok(())
}

/// Elements per chunk of [`for_each_chunk`]; a multiple of every kernel's vector width.
pub(crate) const CHUNK: usize = 1024;

/// Element offsets of a view in row-major order, starting from a flat index.
struct Offsets<'a> {
    shape: &'a [usize],
    strides: [usize; MAX_RANK],
    idx: [usize; MAX_RANK],
    off: usize,
}

impl<'a> Offsets<'a> {
    fn new(shape: &'a [usize], strides: &[isize], start: usize) -> Self {
        let strides = unsigned_strides(strides);
        let mut idx = [0usize; MAX_RANK];
        let (mut r, mut off) = (start, 0usize);
        for ((i, &d), &s) in idx.iter_mut().zip(shape).zip(&strides).rev() {
            let d = d.max(1);
            *i = r % d;
            r /= d;
            off += *i * s;
        }
        Self {
            shape,
            strides,
            idx,
            off,
        }
    }

    /// The current offset; then steps to the next element.
    #[inline(always)]
    fn step(&mut self) -> usize {
        let cur = self.off;
        for ((i, &d), &s) in self.idx.iter_mut().zip(self.shape).zip(&self.strides).rev() {
            *i += 1;
            self.off += s;
            if *i < d {
                return cur;
            }
            self.off -= s * *i;
            *i = 0;
        }
        cur
    }
}

/// Calls `chunk(input, output)` on consecutive runs of at most [`CHUNK`] elements in row-major
/// order; chunks may run concurrently on `pool`.
///
/// Input and output must have the same shape (any rank, including 0). Every chunk starts at a
/// multiple of [`CHUNK`], so a kernel whose result depends on an element's position modulo its
/// vector width gives the same result as one call over the whole flattened tensor, whatever
/// the strides and the pool. Non-contiguous sides are staged through stack buffers. The output
/// must pass [`check_no_overlap`]. Everything is validated before the first write. `chunk` must
/// write every element of its output slice.
pub(crate) fn for_each_chunk<I, O>(
    input: &View<'_, I>,
    output: &mut ViewMut<'_, O>,
    pool: Option<&dyn Pool>,
    chunk: &(dyn Fn(&[I], &mut [O]) + Sync),
) -> Result<()>
where
    I: Copy + Default + Sync,
    O: Copy + Default + Send,
{
    if input.shape() != output.shape() {
        return Err(KernelError::InvalidArgument(
            "input and output shapes differ",
        ));
    }
    check_no_overlap(output.shape(), output.strides())?;
    let n = input.len();
    if n == 0 {
        return Ok(());
    }
    let shape = input.shape();
    let in_strides = input.strides();
    let mut out_strides = [0isize; MAX_RANK];
    for (d, &s) in out_strides.iter_mut().zip(output.strides()) {
        *d = s;
    }
    let out_strides = out_strides.get(..shape.len()).unwrap_or(&[]);
    let src_dense = input.as_contiguous();
    let out_dense = output.as_contiguous_mut().is_some();
    let src = input.data();
    let data = output.data_mut();
    let cap = data.len();
    let out = RowsPtr(data.as_mut_ptr());

    for_each_block(n.div_ceil(CHUNK), pool, &|r: Range<usize>| {
        for c in r {
            let lo = c.saturating_mul(CHUNK);
            if lo >= n {
                return;
            }
            let len = CHUNK.min(n - lo);
            let mut in_store;
            let chunk_in: &[I] = match src_dense {
                Some(s) => s.get(lo..lo + len).unwrap_or(&[]),
                None => {
                    in_store = [I::default(); CHUNK];
                    let mut at = Offsets::new(shape, in_strides, lo);
                    for b in in_store.iter_mut().take(len) {
                        *b = src.get(at.step()).copied().unwrap_or_default();
                    }
                    in_store.get(..len).unwrap_or(&[])
                }
            };
            if chunk_in.len() != len {
                return;
            }
            if out_dense {
                if lo + len > cap {
                    return;
                }
                // SAFETY: `lo..lo + len` lies in the dense output and is this chunk's alone.
                let dst = unsafe { core::slice::from_raw_parts_mut(out.get().add(lo), len) };
                chunk(chunk_in, dst);
            } else {
                let mut out_store = [O::default(); CHUNK];
                let Some(buf) = out_store.get_mut(..len) else {
                    return;
                };
                chunk(chunk_in, buf);
                let mut at = Offsets::new(shape, out_strides, lo);
                for &v in buf.iter() {
                    let off = at.step();
                    if off < cap {
                        // SAFETY: `off` addresses an element of the view that only this chunk owns.
                        unsafe { out.get().add(off).write(v) };
                    }
                }
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_check() {
        assert!(check_no_overlap(&[4, 8], &[8, 1]).is_ok());
        assert!(check_no_overlap(&[4, 8], &[1, 4]).is_ok());
        assert!(check_no_overlap(&[8, 4], &[1, 8]).is_ok());
        assert!(check_no_overlap(&[1, 8], &[0, 1]).is_ok());
        assert!(check_no_overlap(&[0, 8], &[0, 0]).is_ok());
        assert!(check_no_overlap(&[2, 3, 4], &[12, 4, 1]).is_ok());
        assert!(check_no_overlap(&[2, 3, 4], &[1, 8, 2]).is_ok());
        assert!(check_no_overlap(&[4, 8], &[0, 1]).is_err());
        assert!(check_no_overlap(&[4, 8], &[7, 1]).is_err());
        assert!(check_no_overlap(&[4, 8], &[8, 0]).is_err());
        assert!(check_no_overlap(&[2, 3, 4], &[1, 4, 2]).is_err());
        // Injective but interleaved: rejected by design.
        assert!(check_no_overlap(&[3, 2], &[2, 3]).is_err());
    }

    #[test]
    fn row_offset_walks_outer_axes() {
        assert_eq!(row_offset(0, &[2, 3], &[100, 10, 1]), 0);
        assert_eq!(row_offset(4, &[2, 3], &[100, 10, 1]), 110);
        assert_eq!(row_offset(5, &[2, 3], &[100, 10, 1]), 120);
        assert_eq!(row_offset(3, &[], &[1]), 0);
    }

    /// Row op with differing types and lengths: each output row is `[sum, len]` of its input row.
    fn sum_row(s: &[i16], d: &mut [f32]) {
        if let [a, b] = d {
            *a = s.iter().map(|&v| f32::from(v)).sum();
            *b = s.len() as f32;
        }
    }

    fn sum_strided(s: StridedRow<'_, i16>, d: &mut StridedRowMut<'_, f32>) {
        d.set(0, s.iter().map(f32::from).sum());
        d.set(1, s.len() as f32);
        d.set(d.len(), -1.0);
    }

    fn run_sum(rows: usize, n: usize, in_t: bool, out_t: bool) -> Vec<f32> {
        let data: Vec<i16> = (0..rows * n).map(|i| (i % 97) as i16 - 40).collect();
        let input = if in_t {
            let mut t = vec![0i16; data.len()];
            for i in 0..rows {
                for j in 0..n {
                    t[j * rows + i] = data[i * n + j];
                }
            }
            t
        } else {
            data
        };
        let in_strides = if in_t {
            [1, rows as isize]
        } else {
            [n as isize, 1]
        };
        let view = View::strided(&input, &[rows, n], &in_strides).unwrap();
        let mut out = vec![0.0f32; rows * 2];
        let out_strides = if out_t { [1, rows as isize] } else { [2, 1] };
        let mut dst = ViewMut::strided(&mut out, &[rows, 2], &out_strides).unwrap();
        for_each_row(&view, &mut dst, None, &sum_row, &sum_strided).unwrap();
        if out_t {
            (0..rows).flat_map(|i| [out[i], out[rows + i]]).collect()
        } else {
            out
        }
    }

    #[test]
    fn rows_of_different_types_and_lengths() {
        for n in [3usize, STAGE + 5] {
            let want = run_sum(4, n, false, false);
            assert_eq!(want[1], n as f32);
            for (in_t, out_t) in [(true, false), (false, true), (true, true)] {
                assert_eq!(
                    run_sum(4, n, in_t, out_t),
                    want,
                    "n {n} in_t {in_t} out_t {out_t}"
                );
            }
        }
    }

    #[test]
    fn row_blocks_align_to_segments() {
        let segments = [5usize, 0, 9, 3];
        let blocks: Vec<_> = (0..8).map(|b| locate_block(&segments, 4, b)).collect();
        assert_eq!(
            blocks,
            [
                Some((0, 0, 4)),
                Some((0, 4, 1)),
                Some((2, 5, 4)),
                Some((2, 9, 4)),
                Some((2, 13, 1)),
                Some((3, 14, 3)),
                None,
                None,
            ]
        );
    }

    /// Each output row is `[segment, first row of block, row index, sum of the input row]`.
    fn run_blocks(
        rows: usize,
        n: usize,
        in_t: bool,
        out_t: bool,
        pool: Option<&dyn Pool>,
    ) -> Vec<f32> {
        let data: Vec<i16> = (0..rows * n).map(|i| (i % 89) as i16 - 30).collect();
        let input = if in_t {
            let mut t = vec![0i16; data.len()];
            for i in 0..rows {
                for j in 0..n {
                    t[j * rows + i] = data[i * n + j];
                }
            }
            t
        } else {
            data
        };
        let in_strides = if in_t {
            [1, rows as isize]
        } else {
            [n as isize, 1]
        };
        let view = View::strided(&input, &[rows, n], &in_strides).unwrap();
        let mut out = vec![f32::NAN; rows * 4];
        let out_strides = if out_t { [1, rows as isize] } else { [4, 1] };
        let mut dst = ViewMut::strided(&mut out, &[rows, 4], &out_strides).unwrap();
        let segments = [rows / 3, 0, rows - rows / 3];
        for_each_row_block(
            &view,
            &mut dst,
            pool,
            &segments,
            7,
            &|seg, first, src, d| {
                for (r, (s, o)) in src.chunks_exact(n).zip(d.chunks_exact_mut(4)).enumerate() {
                    o[0] = seg as f32;
                    o[1] = first as f32;
                    o[2] = (first + r) as f32;
                    o[3] = s.iter().map(|&v| f32::from(v)).sum();
                }
            },
        )
        .unwrap();
        if out_t {
            (0..rows)
                .flat_map(|i| [out[i], out[rows + i], out[2 * rows + i], out[3 * rows + i]])
                .collect()
        } else {
            out
        }
    }

    #[test]
    fn row_blocks_stage_strided_sides() {
        let scoped = crate::pool::ScopedThreads::new(std::num::NonZeroUsize::new(3).unwrap());
        let (rows, n) = (40usize, 9usize);
        let want = run_blocks(rows, n, false, false, None);
        for r in 0..rows {
            let seg = if r < rows / 3 { 0.0 } else { 2.0 };
            let start = if r < rows / 3 { 0 } else { rows / 3 };
            assert_eq!(want[r * 4], seg);
            assert_eq!(want[r * 4 + 1], (start + (r - start) / 7 * 7) as f32);
            assert_eq!(want[r * 4 + 2], r as f32);
        }
        for (in_t, out_t) in [(true, false), (false, true), (true, true)] {
            for pool in [None, Some(&scoped as &dyn Pool)] {
                assert_eq!(
                    run_blocks(rows, n, in_t, out_t, pool),
                    want,
                    "in_t {in_t} out_t {out_t}"
                );
            }
        }
    }

    #[test]
    fn row_blocks_validate_before_writing() {
        let input = [1i16; 40];
        let mut out = [7.0f32; 40];
        let v = View::contiguous(&input, &[10, 4]).unwrap();
        let fill = |_: usize, _: usize, _: &[i16], d: &mut [f32]| d.fill(0.0);
        let mut d = ViewMut::contiguous(&mut out, &[10, 4]).unwrap();
        assert!(for_each_row_block(&v, &mut d, None, &[9], 4, &fill).is_err());
        assert!(for_each_row_block(&v, &mut d, None, &[10, usize::MAX], 4, &fill).is_err());
        assert!(for_each_row_block(&v, &mut d, None, &[10], 0, &fill).is_err());
        assert!(for_each_row_block(&v, &mut d, None, &[10], STAGE, &fill).is_err());
        let mut d = ViewMut::strided(&mut out, &[10, 4], &[1, 1]).unwrap();
        assert!(for_each_row_block(&v, &mut d, None, &[10], 4, &fill).is_err());
        let mut d = ViewMut::contiguous(&mut out[..36], &[9, 4]).unwrap();
        assert!(for_each_row_block(&v, &mut d, None, &[10], 4, &fill).is_err());
        let v3 = View::contiguous(&input, &[10, 2, 2]).unwrap();
        let mut d = ViewMut::contiguous(&mut out, &[10, 4]).unwrap();
        assert!(for_each_row_block(&v3, &mut d, None, &[10], 4, &fill).is_err());
        assert!(out.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn rejects_mismatched_outer_shapes_and_rank_zero() {
        let input = [0i16; 8];
        let mut out = [0.0f32; 8];
        let v = View::contiguous(&input, &[2, 4]).unwrap();
        let mut d = ViewMut::contiguous(&mut out, &[4, 2]).unwrap();
        assert!(for_each_row(&v, &mut d, None, &sum_row, &sum_strided).is_err());
        let v0 = View::contiguous(&input[..1], &[]).unwrap();
        let mut d0 = ViewMut::contiguous(&mut out[..1], &[]).unwrap();
        assert!(for_each_row(&v0, &mut d0, None, &sum_row, &sum_strided).is_err());
    }
}
