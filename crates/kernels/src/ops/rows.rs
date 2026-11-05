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
    let mut out_strides = [0isize; MAX_RANK];
    for (d, &s) in out_strides.iter_mut().zip(output.strides()) {
        *d = s;
    }
    let out_strides = out_strides.get(..shape.len()).unwrap_or(&[]);
    let src_dense = input.as_contiguous();
    let out_dense = output.as_contiguous_mut().is_some();
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
            let mut in_store = [I::default(); CHUNK];
            let chunk_in = gather_chunk(src_dense, input, &mut in_store, lo, len);
            if chunk_in.len() != len {
                return;
            }
            write_chunk(out, cap, out_dense, shape, out_strides, lo, len, |dst| {
                chunk(chunk_in, dst);
            });
        }
    });
    Ok(())
}

/// For kernels whose result for an element does not depend on its position: when some axis
/// has unit stride in both views, calls `chunk(input run, output run)` on every run along it
/// (the longest such axis); runs may go concurrently on `pool`.
///
/// Returns `None`, touching nothing, when no axis is contiguous in both views.
pub(crate) fn for_each_unit_run<T>(
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
    chunk: &(dyn Fn(&[T], &mut [T]) + Sync),
) -> Option<Result<()>>
where
    T: Copy + Send + Sync,
{
    if input.shape() != output.shape() {
        return None;
    }
    let shape = input.shape();
    let (is, os) = (input.strides(), output.strides());
    let rank = shape.len();
    let axis = (0..rank)
        .filter(|&a| is[a] == 1 && os[a] == 1)
        .max_by_key(|&a| shape[a])?;
    if let Err(e) = check_no_overlap(shape, os) {
        return Some(Err(e));
    }
    let len = shape[axis];
    let mut others = [(1usize, 0usize, 0usize); MAX_RANK];
    let mut n_others = 0;
    for a in (0..rank).filter(|&a| a != axis) {
        if let Some(o) = others.get_mut(n_others) {
            *o = (shape[a], is[a].unsigned_abs(), os[a].unsigned_abs());
        }
        n_others += 1;
    }
    let others = others.get(..n_others).unwrap_or(&[]);
    let runs: usize = others.iter().map(|o| o.0).product();
    if len == 0 || runs == 0 {
        return Some(Ok(()));
    }
    let src = input.data();
    let data = output.data_mut();
    let cap = data.len();
    let out = RowsPtr(data.as_mut_ptr());
    let axes = axes_of(others);
    let axes = axes.get(..others.len()).unwrap_or(&[]);
    for_each_block(runs, pool, &|r: Range<usize>| {
        for (bi, bo, _) in outer_items(r, axes, 1) {
            let Some(s) = src.get(bi..bi + len) else {
                return;
            };
            if bo + len > cap {
                return;
            }
            // SAFETY: the run lies in the output view (checked above), runs of different
            // indices are disjoint (the output does not overlap itself), and only this task
            // writes run `t`.
            let d = unsafe { core::slice::from_raw_parts_mut(out.get().add(bo), len) };
            chunk(s, d);
        }
    });
    Some(Ok(()))
}

/// Rows (input-contiguous axis) per tile of [`for_each_transposed_tile`]; one NEON transpose.
const TILE_EDGE: usize = 16;

/// Columns (output-contiguous axis) per tile of [`for_each_transposed_tile`].
const TILE_COLS: usize = 64;

/// For kernels whose result for an element does not depend on its position: when the input
/// is contiguous along one axis and the output along another (a transpose), calls
/// `chunk(input tile, output tile)` on 16×64 tiles of those two axes, transposed from the
/// input into a stack buffer in output order and written back as output row segments.
///
/// A task owns one 64-wide column block and walks every input block of it, so each source
/// row is read once while cached and the output is written in whole 64-element runs; tasks
/// may run concurrently on `pool`.
///
/// Returns `None`, touching nothing, when the views are not such a pair; the caller then takes
/// [`for_each_chunk`].
pub(crate) fn for_each_transposed_tile<T>(
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
    chunk: &(dyn Fn(&[T], &mut [T]) + Sync),
) -> Option<Result<()>>
where
    T: Copy + Default + Send + Sync,
{
    if input.shape() != output.shape() {
        return None;
    }
    let shape = input.shape();
    let (is, os) = (input.strides(), output.strides());
    let rank = shape.len();
    let ai = (0..rank).find(|&a| shape[a] > 1 && is[a] == 1)?;
    let ao = (0..rank).find(|&a| a != ai && shape[a] > 1 && os[a] == 1)?;
    if let Err(e) = check_no_overlap(shape, os) {
        return Some(Err(e));
    }
    // The other axes: (extent, input stride, output stride).
    let mut others = [(1usize, 0usize, 0usize); MAX_RANK];
    let mut n_others = 0;
    for a in (0..rank).filter(|&a| a != ai && a != ao) {
        if let Some(o) = others.get_mut(n_others) {
            *o = (shape[a], is[a].unsigned_abs(), os[a].unsigned_abs());
        }
        n_others += 1;
    }
    let others = others.get(..n_others).unwrap_or(&[]);
    let outer: usize = others.iter().map(|o| o.0).product();
    let (ni, no) = (shape[ai], shape[ao]);
    let (in_step, out_step) = (is[ao].unsigned_abs(), os[ai].unsigned_abs());
    let col_blocks = no.div_ceil(TILE_COLS);
    let axes = axes_of(others);
    let axes = axes.get(..others.len()).unwrap_or(&[]);
    let src = input.data();
    let data = output.data_mut();
    let cap = data.len();
    let out = RowsPtr(data.as_mut_ptr());
    for_each_block(outer * col_blocks, pool, &|r: Range<usize>| {
        let mut tile_in = [T::default(); TILE_EDGE * TILE_COLS];
        let mut tile_out = [T::default(); TILE_EDGE * TILE_COLS];
        for (bi, bo, ob) in outer_items(r, axes, col_blocks) {
            let o0 = ob * TILE_COLS;
            let to = (no - o0).min(TILE_COLS);
            for i0 in (0..ni).step_by(TILE_EDGE) {
                let ti = (ni - i0).min(TILE_EDGE);
                let first = bi + i0 + o0 * in_step;
                if first + (to - 1) * in_step + ti > src.len() {
                    return;
                }
                let n = ti * to;
                let (Some(tin), Some(tout)) = (tile_in.get_mut(..n), tile_out.get_mut(..n)) else {
                    return;
                };
                for o in 0..to {
                    let base = first + o * in_step;
                    let Some(row) = src.get(base..base + ti) else {
                        return;
                    };
                    for (i, &v) in row.iter().enumerate() {
                        if let Some(b) = tin.get_mut(i * to + o) {
                            *b = v;
                        }
                    }
                }
                chunk(tin, tout);
                for i in 0..ti {
                    let base = bo + (i0 + i) * out_step + o0;
                    if base + to > cap {
                        return;
                    }
                    // SAFETY: row `i0 + i`, columns `o0..o0 + to`, lies in the output view (the
                    // check above) and only this task writes columns `o0..o0 + to` of this
                    // outer index; the tile buffer does not alias the output.
                    unsafe { copy_row(tout.as_ptr().add(i * to), out.get().add(base), to) };
                }
            }
        }
    });
    Some(Ok(()))
}

/// For byte kernels that map 16-byte vectors independently of position: the transposed views
/// of [`for_each_transposed_tile`] when both tile axes are multiples of 16, as 16×16 tiles
/// transposed in registers with each of the 16 vectors passed through `f`; and when the input
/// axis is two or four long, sixteen of its runs at a time deinterleaved with `ld2`/`ld4`.
/// Axes contiguous on both sides are merged first, so a channel slice of an NHWC tensor
/// written channel-major is one long transpose. A 16×64 strip is
/// staged on the stack so every output row is written as one 64-byte run: with output rows far
/// apart, short scattered stores cost more than the arithmetic. Tasks own 64-wide column
/// blocks as there.
///
/// Returns `None`, touching nothing, when the views are not such a pair, the elements are not
/// bytes, or an axis does not suit either path.
#[cfg(target_arch = "aarch64")]
pub(crate) fn for_each_transposed_vector<T, F>(
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
    f: &F,
) -> Option<Result<()>>
where
    T: Copy + Send + Sync,
    F: Fn(core::arch::aarch64::uint8x16_t) -> core::arch::aarch64::uint8x16_t + Sync,
{
    use core::arch::aarch64::{uint8x16_t, vdupq_n_u8, vld1q_u8, vld2q_u8, vld4q_u8, vst1q_u8};
    if core::mem::size_of::<T>() != 1 || input.shape() != output.shape() {
        return None;
    }
    // Axes of length one dropped, and neighbours that are contiguous on both sides merged.
    let (mut shape, mut is, mut os) = ([0usize; MAX_RANK], [0isize; MAX_RANK], [0isize; MAX_RANK]);
    let mut rank = 0;
    for ((&d, &si), &so) in input
        .shape()
        .iter()
        .zip(input.strides())
        .zip(output.strides())
    {
        if d == 1 {
            continue;
        }
        let w = d as isize;
        if rank > 0 && is[rank - 1] == si * w && os[rank - 1] == so * w {
            shape[rank - 1] *= d;
            is[rank - 1] = si;
            os[rank - 1] = so;
        } else if rank < MAX_RANK {
            (shape[rank], is[rank], os[rank]) = (d, si, so);
            rank += 1;
        }
    }
    let (shape, is, os) = (&shape[..rank], &is[..rank], &os[..rank]);
    let ai = (0..rank).find(|&a| shape[a] > 1 && is[a] == 1)?;
    let ao = (0..rank).find(|&a| a != ai && shape[a] > 1 && os[a] == 1)?;
    let (ni, no) = (shape[ai], shape[ao]);
    // Input runs of two or four bytes: sixteen of them deinterleaved by `ld2`/`ld4`.
    let narrow = matches!(ni, 2 | 4);
    if (!narrow && ni % TILE_EDGE != 0) || no % TILE_EDGE != 0 {
        return None;
    }
    if let Err(e) = check_no_overlap(output.shape(), output.strides()) {
        return Some(Err(e));
    }
    let mut others = [(1usize, 0usize, 0usize); MAX_RANK];
    let mut n_others = 0;
    for a in (0..rank).filter(|&a| a != ai && a != ao) {
        if let Some(o) = others.get_mut(n_others) {
            *o = (shape[a], is[a].unsigned_abs(), os[a].unsigned_abs());
        }
        n_others += 1;
    }
    let others = others.get(..n_others).unwrap_or(&[]);
    let outer: usize = others.iter().map(|o| o.0).product();
    let (in_step, out_step) = (is[ao].unsigned_abs(), os[ai].unsigned_abs());
    let col_blocks = no.div_ceil(TILE_COLS);
    let axes = axes_of(others);
    let axes = axes.get(..others.len()).unwrap_or(&[]);
    let src = input.data();
    let data = output.data_mut();
    let cap = data.len();
    let out = RowsPtr(data.as_mut_ptr());
    for_each_block(outer * col_blocks, pool, &|r: Range<usize>| {
        for (bi, bo, ob) in outer_items(r, axes, col_blocks) {
            let o0 = ob * TILE_COLS;
            let to = (no - o0).min(TILE_COLS);
            // One 16-row strip of the block, staged so each output row is written as one run.
            let mut strip = [0u8; TILE_EDGE * TILE_COLS];
            if narrow {
                let first = bi + o0 * in_step;
                let last_out = bo + (ni - 1) * out_step + o0 + to;
                if first + (to - 1) * in_step + ni > src.len() || last_out > cap {
                    return;
                }
                let s = src.as_ptr().cast::<u8>();
                let d = out.get().cast::<u8>();
                let sp = strip.as_mut_ptr();
                // SAFETY: the `to` source runs hold `ni` readable bytes each and the `ni` output
                // rows `to` writable bytes each (both checked above); `strip` holds `ni <= 16`
                // rows of `to <= 64`; only this task writes columns `o0..o0 + to` of this outer
                // index; `T` is one byte.
                unsafe {
                    for o in (0..to).step_by(TILE_EDGE) {
                        let mut runs = [0u8; 4 * TILE_EDGE];
                        let rp = runs.as_mut_ptr();
                        for k in 0..TILE_EDGE {
                            let at = s.add(first + (o + k) * in_step);
                            if ni == 4 {
                                rp.add(4 * k)
                                    .cast::<u32>()
                                    .write_unaligned(at.cast::<u32>().read_unaligned());
                            } else {
                                rp.add(2 * k)
                                    .cast::<u16>()
                                    .write_unaligned(at.cast::<u16>().read_unaligned());
                            }
                        }
                        let zero = vdupq_n_u8(0);
                        let lanes: [uint8x16_t; 4] = if ni == 4 {
                            let v = vld4q_u8(rp);
                            [v.0, v.1, v.2, v.3]
                        } else {
                            let v = vld2q_u8(rp);
                            [v.0, v.1, zero, zero]
                        };
                        for (c, x) in lanes.into_iter().take(ni).enumerate() {
                            vst1q_u8(sp.add(c * TILE_COLS + o), f(x));
                        }
                    }
                    for c in 0..ni {
                        copy_row(sp.add(c * TILE_COLS), d.add(bo + c * out_step + o0), to);
                    }
                }
                continue;
            }
            for i0 in (0..ni).step_by(TILE_EDGE) {
                let first = bi + i0 + o0 * in_step;
                let last_out = bo + (i0 + TILE_EDGE - 1) * out_step + o0 + to;
                if first + (to - 1) * in_step + TILE_EDGE > src.len() || last_out > cap {
                    return;
                }
                let s = src.as_ptr().cast::<u8>();
                let d = out.get().cast::<u8>();
                let sp = strip.as_mut_ptr();
                // SAFETY: the `to` source rows hold 16 readable bytes each and the 16 output rows
                // `to` writable bytes each (both checked above); `strip` holds 16 rows of `to ≤
                // 64`; only this task writes columns `o0..o0 + to` of this outer index; `T` is
                // one byte.
                unsafe {
                    for o in (0..to).step_by(TILE_EDGE) {
                        let rows =
                            core::array::from_fn(|k| vld1q_u8(s.add(first + (o + k) * in_step)));
                        let v = crate::prim::transpose::transpose16(rows);
                        for (i, x) in v.into_iter().enumerate() {
                            vst1q_u8(sp.add(i * TILE_COLS + o), f(x));
                        }
                    }
                    for i in 0..TILE_EDGE {
                        copy_row(
                            sp.add(i * TILE_COLS),
                            d.add(bo + (i0 + i) * out_step + o0),
                            to,
                        );
                    }
                }
            }
        }
    });
    Some(Ok(()))
}

/// Copies `n` elements; a full row of one-byte elements moves as four 16-byte registers rather
/// than through a `memcpy` call.
///
/// # Safety
/// As [`core::ptr::copy_nonoverlapping`].
#[inline(always)]
unsafe fn copy_row<T>(src: *const T, dst: *mut T, n: usize) {
    #[cfg(target_arch = "aarch64")]
    if core::mem::size_of::<T>() == 1 && n == TILE_COLS {
        use core::arch::aarch64::{vld1q_u8_x4, vst1q_u8_x4};
        // SAFETY: 64 bytes on each side, forwarded from the caller.
        unsafe { vst1q_u8_x4(dst.cast(), vld1q_u8_x4(src.cast())) };
        return;
    }
    // SAFETY: forwarded from the caller.
    unsafe { core::ptr::copy_nonoverlapping(src, dst, n) };
}

/// Rows per group of [`for_each_row_group`].
pub(crate) const ROW_GROUP: usize = 16;

/// Up to [`ROW_GROUP`] rows, unit-stride on both sides, each at its own offset: what
/// [`for_each_row_group`] hands its kernel. Input rows share one length and output rows
/// another. Every row was bounds-checked by the walk and the output rows belong to this group
/// alone.
pub(crate) struct RowGroup<'a, T> {
    src: &'a [T],
    dst: RowsPtr<T>,
    len: usize,
    dst_len: usize,
    count: usize,
    dst_contiguous: bool,
    src_at: [usize; ROW_GROUP],
    dst_at: [usize; ROW_GROUP],
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
impl<'a, T> RowGroup<'a, T> {
    /// Rows in the group, 1 to [`ROW_GROUP`].
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    /// Elements per input row.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Elements per output row.
    pub(crate) fn dst_len(&self) -> usize {
        self.dst_len
    }

    /// Whether the output rows follow one another in memory, row `k` at `dst_ptr(0) + k·dst_len`.
    pub(crate) fn dst_contiguous(&self) -> bool {
        self.dst_contiguous
    }

    /// Start of input row `k`, readable for [`Self::len`] elements when `k < count()`.
    pub(crate) fn src_ptr(&self, k: usize) -> *const T {
        let at = self.src_at.get(k).copied().unwrap_or(0);
        self.src.as_ptr().wrapping_add(at)
    }

    /// Start of output row `k`, writable for [`Self::dst_len`] elements when `k < count()` and
    /// written by nothing else while the kernel runs.
    pub(crate) fn dst_ptr(&self, k: usize) -> *mut T {
        let at = self.dst_at.get(k).copied().unwrap_or(0);
        self.dst.get().wrapping_add(at)
    }

    /// Input row `k` (empty past the group).
    pub(crate) fn input(&self, k: usize) -> &'a [T] {
        if k >= self.count {
            return &[];
        }
        let at = self.src_at.get(k).copied().unwrap_or(0);
        self.src.get(at..at + self.len).unwrap_or(&[])
    }

    /// Output row `k` (empty past the group).
    pub(crate) fn output(&mut self, k: usize) -> &mut [T] {
        if k >= self.count {
            return &mut [];
        }
        // SAFETY: row `k` lies in the output (checked by the walk), output rows do not overlap
        // and this group alone writes them; `&mut self` keeps the slice unique.
        unsafe { core::slice::from_raw_parts_mut(self.dst_ptr(k), self.dst_len) }
    }
}

/// For row kernels that run several rows at once: when every row is unit-stride on both
/// sides, calls `group` on consecutive groups of [`ROW_GROUP`] rows (fewer for the last);
/// groups may run concurrently on `pool`.
///
/// Rows are walked in output memory order (outer axes sorted by output stride, largest first),
/// so consecutive groups write consecutive memory.
///
/// The views agree on every axis but the last, which holds the rows; input and output rows may
/// differ in length. Returns `None`, touching nothing, when the outer axes differ or a row is
/// not unit-stride.
pub(crate) fn for_each_row_group<T>(
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
    group: &(dyn Fn(&mut RowGroup<'_, T>) + Sync),
) -> Option<Result<()>>
where
    T: Copy + Send + Sync,
{
    let (&n, outer) = input.shape().split_last()?;
    let (&m, out_outer) = output.shape().split_last()?;
    if outer != out_outer {
        return None;
    }
    let rank = input.rank();
    let (is, os) = (
        unsigned_strides(input.strides()),
        unsigned_strides(output.strides()),
    );
    if (n > 1 && is.get(rank - 1) != Some(&1)) || (m > 1 && os.get(rank - 1) != Some(&1)) {
        return None;
    }
    if let Err(e) = check_no_overlap(output.shape(), output.strides()) {
        return Some(Err(e));
    }
    let rows: usize = outer.iter().product();
    if rows == 0 || n == 0 || m == 0 {
        return Some(Ok(()));
    }
    // Unit axes never move; dropping them keeps the innermost walked axis a real one.
    let mut axes = [Axis::default(); MAX_RANK];
    let mut n_axes = 0;
    for (a, &extent) in outer.iter().enumerate().filter(|&(_, &d)| d > 1) {
        if let Some(x) = axes.get_mut(n_axes) {
            *x = Axis {
                extent,
                src: is.get(a).copied().unwrap_or(0),
                dst: os.get(a).copied().unwrap_or(0),
            };
        }
        n_axes += 1;
    }
    let axes = axes.get_mut(..n_axes)?;
    axes.sort_by_key(|x| core::cmp::Reverse(x.dst));
    let axes: &[Axis] = axes;
    let src = input.data();
    let data = output.data_mut();
    let cap = data.len();
    let dst = RowsPtr(data.as_mut_ptr());
    for_each_block(rows.div_ceil(ROW_GROUP), pool, &|blocks: Range<usize>| {
        let mut walk = Odometer::new(blocks.start * ROW_GROUP, axes);
        for blk in blocks {
            let count = (rows - blk * ROW_GROUP).min(ROW_GROUP);
            let mut g = RowGroup {
                src,
                dst,
                len: n,
                dst_len: m,
                count,
                dst_contiguous: false,
                src_at: [0; ROW_GROUP],
                dst_at: [0; ROW_GROUP],
            };
            let (src_end, dst_end) = if let Some((s0, d0)) = walk.take_run(axes, count) {
                // One run along the innermost axis: offsets are affine in the row.
                let (ss, ds) = (walk.inner.src, walk.inner.dst);
                for (k, (s, d)) in g.src_at.iter_mut().zip(g.dst_at.iter_mut()).enumerate() {
                    (*s, *d) = (s0 + k * ss, d0 + k * ds);
                }
                g.dst_contiguous = ds == m;
                (s0 + (count - 1) * ss, d0 + (count - 1) * ds)
            } else {
                let (mut src_end, mut dst_end) = (0usize, 0usize);
                for (s, d) in g.src_at.iter_mut().zip(g.dst_at.iter_mut()).take(count) {
                    (*s, *d) = walk.next(axes);
                    src_end = src_end.max(*s);
                    dst_end = dst_end.max(*d);
                }
                let first = g.dst_at[0];
                g.dst_contiguous = g.dst_at[..count]
                    .iter()
                    .enumerate()
                    .all(|(k, &at)| at == first + k * m);
                (src_end, dst_end)
            };
            if src_end + n > src.len() || dst_end + m > cap {
                return;
            }
            group(&mut g);
        }
    });
    Some(Ok(()))
}

/// An outer axis of a row walk: extent and the two element strides.
#[derive(Clone, Copy, Default)]
struct Axis {
    extent: usize,
    src: usize,
    dst: usize,
}

/// Items `range` of `outer × inner`, inner fastest, the outer index walking `axes`: each item's
/// input and output offsets and inner index, stepped without division.
fn outer_items(
    range: Range<usize>,
    axes: &[Axis],
    inner: usize,
) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    let inner = inner.max(1);
    let mut odo = Odometer::new(range.start / inner, axes);
    let mut cur = odo.next(axes);
    let mut j = range.start % inner;
    range.map(move |_| {
        let item = (cur.0, cur.1, j);
        j += 1;
        if j == inner {
            j = 0;
            cur = odo.next(axes);
        }
        item
    })
}

/// The axes of `others` (extent, input stride, output stride) as [`Axis`] values.
fn axes_of(others: &[(usize, usize, usize)]) -> [Axis; MAX_RANK] {
    core::array::from_fn(|i| {
        others.get(i).map_or(
            Axis {
                extent: 1,
                src: 0,
                dst: 0,
            },
            |&(extent, src, dst)| Axis {
                extent: extent.max(1),
                src,
                dst,
            },
        )
    })
}

/// Row offsets in both views, stepped row by row without division; the innermost axis steps
/// without touching the others.
struct Odometer {
    idx: [usize; MAX_RANK],
    at: (usize, usize),
    inner: Axis,
}

impl Odometer {
    /// Starts at row `r` of `axes` (outermost first; none means a single row).
    fn new(mut r: usize, axes: &[Axis]) -> Self {
        let mut idx = [0usize; MAX_RANK];
        let mut at = (0usize, 0usize);
        for (i, x) in idx.iter_mut().zip(axes).rev() {
            *i = r % x.extent;
            r /= x.extent;
            at.0 += *i * x.src;
            at.1 += *i * x.dst;
        }
        let inner = axes.last().copied().unwrap_or(Axis {
            extent: 1,
            src: 0,
            dst: 0,
        });
        Self { idx, at, inner }
    }

    /// The current row's (input, output) offsets; then advances to the next row.
    #[inline(always)]
    fn next(&mut self, axes: &[Axis]) -> (usize, usize) {
        let cur = self.at;
        let last = axes.len().saturating_sub(1);
        if let Some(i) = self.idx.get_mut(last) {
            if *i + 1 < self.inner.extent {
                *i += 1;
                self.at.0 += self.inner.src;
                self.at.1 += self.inner.dst;
                return cur;
            }
        }
        self.carry(axes);
        cur
    }

    /// The step that wraps the innermost axis.
    #[cold]
    fn carry(&mut self, axes: &[Axis]) {
        for (i, x) in self.idx.iter_mut().zip(axes).rev() {
            *i += 1;
            self.at.0 += x.src;
            self.at.1 += x.dst;
            if *i < x.extent {
                break;
            }
            self.at.0 -= x.extent * x.src;
            self.at.1 -= x.extent * x.dst;
            *i = 0;
        }
    }

    /// When the next `count ≥ 1` rows all lie along the innermost axis, the first one's offsets,
    /// advancing past them; otherwise `None`, touching nothing.
    #[inline(always)]
    fn take_run(&mut self, axes: &[Axis], count: usize) -> Option<(usize, usize)> {
        let last = axes.len().checked_sub(1)?;
        let i = self.idx.get_mut(last)?;
        let end = i.checked_add(count)?;
        if count == 0 || end > self.inner.extent {
            return None;
        }
        let cur = self.at;
        let steps = count - 1;
        *i += steps;
        self.at.0 += steps * self.inner.src;
        self.at.1 += steps * self.inner.dst;
        self.next(axes);
        Some(cur)
    }
}

/// Runs `f` on output elements `lo..lo + len` in row-major order: in place when the output is
/// dense, else on a staged copy that is then scattered.
#[allow(clippy::too_many_arguments)]
fn write_chunk<O: Copy + Default>(
    out: RowsPtr<O>,
    cap: usize,
    dense: bool,
    shape: &[usize],
    strides: &[isize],
    lo: usize,
    len: usize,
    f: impl FnOnce(&mut [O]),
) {
    if dense {
        if lo + len > cap {
            return;
        }
        // SAFETY: `lo..lo + len` lies in the dense output and is this chunk's alone.
        f(unsafe { core::slice::from_raw_parts_mut(out.get().add(lo), len) });
        return;
    }
    let mut store = [O::default(); CHUNK];
    let Some(buf) = store.get_mut(..len) else {
        return;
    };
    f(buf);
    let mut at = Offsets::new(shape, strides, lo);
    for &v in buf.iter() {
        let off = at.step();
        if off < cap {
            // SAFETY: `off` addresses an element of the view that only this chunk owns.
            unsafe { out.get().add(off).write(v) };
        }
    }
}

/// A chunk of `src` in row-major order: the elements themselves when the view is dense, else a
/// copy in `store`.
fn gather_chunk<'s, T: Copy + Default>(
    dense: Option<&'s [T]>,
    view: &View<'_, T>,
    store: &'s mut [T; CHUNK],
    lo: usize,
    len: usize,
) -> &'s [T] {
    if let Some(s) = dense {
        return s.get(lo..lo + len).unwrap_or(&[]);
    }
    let src = view.data();
    let mut at = Offsets::new(view.shape(), view.strides(), lo);
    for b in store.iter_mut().take(len) {
        *b = src.get(at.step()).copied().unwrap_or_default();
    }
    store.get(..len).unwrap_or(&[])
}

/// What [`for_each_chunk2`] calls on each run: its flat start, the two input runs and the output
/// run.
type Chunk2<'a, A, B, O> = dyn Fn(usize, &[A], &[B], &mut [O]) + Sync + 'a;

/// [`for_each_chunk`] over two inputs: calls `chunk(start, lhs, rhs, output)` on consecutive
/// runs of at most [`CHUNK`] elements in row-major order, `start` being the run's flat index.
///
/// All three views must have the same shape; a broadcast input is a view with zero strides.
/// Runs never cross a multiple of [`CHUNK`], and when every view's last axis has unit stride
/// they are row segments read and written in place, so they may start anywhere; a kernel
/// whose result depends on the flat index must go by `start`. The output must pass
/// [`check_no_overlap`]. Everything is validated before the first write. `chunk` must write
/// every element of its output slice.
pub(crate) fn for_each_chunk2<A, B, O>(
    lhs: &View<'_, A>,
    rhs: &View<'_, B>,
    output: &mut ViewMut<'_, O>,
    pool: Option<&dyn Pool>,
    chunk: &Chunk2<'_, A, B, O>,
) -> Result<()>
where
    A: Copy + Default + Sync,
    B: Copy + Default + Sync,
    O: Copy + Default + Send,
{
    if lhs.shape() != output.shape() || rhs.shape() != output.shape() {
        return Err(KernelError::InvalidArgument(
            "input and output shapes differ",
        ));
    }
    check_no_overlap(output.shape(), output.strides())?;
    let n = output.len();
    if n == 0 {
        return Ok(());
    }
    let mut shape = [0usize; MAX_RANK];
    shape[..output.shape().len()].copy_from_slice(output.shape());
    let shape = shape.get(..output.shape().len()).unwrap_or(&[]);
    let mut out_strides = [0isize; MAX_RANK];
    out_strides[..shape.len()].copy_from_slice(output.strides());
    let out_strides = out_strides.get(..shape.len()).unwrap_or(&[]);
    let (lhs_dense, rhs_dense) = (lhs.as_contiguous(), rhs.as_contiguous());
    let out_dense = output.as_contiguous_mut().is_some();
    // Row segments in place, unless the dense path below already passes whole chunks.
    let row_runs = !(lhs_dense.is_some() && rhs_dense.is_some() && out_dense)
        && [lhs.strides(), rhs.strides(), out_strides]
            .iter()
            .all(|s| s.last() == Some(&1));
    let row_len = shape.last().copied().unwrap_or(1).max(1);
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
            if row_runs {
                let (end, mut at) = (lo + len, lo);
                while at < end {
                    let seg = (row_len - at % row_len).min(end - at);
                    let ao = Offsets::new(shape, lhs.strides(), at).off;
                    let bo = Offsets::new(shape, rhs.strides(), at).off;
                    let oo = Offsets::new(shape, out_strides, at).off;
                    let (Some(a), Some(b)) =
                        (lhs.data().get(ao..ao + seg), rhs.data().get(bo..bo + seg))
                    else {
                        return;
                    };
                    if oo + seg > cap {
                        return;
                    }
                    // SAFETY: the segment's `seg` output elements are consecutive (unit last
                    // stride), lie in the buffer (checked) and belong to this chunk alone.
                    let dst = unsafe { core::slice::from_raw_parts_mut(out.get().add(oo), seg) };
                    chunk(at, a, b, dst);
                    at += seg;
                }
                continue;
            }
            let (mut a_store, mut b_store) = ([A::default(); CHUNK], [B::default(); CHUNK]);
            let a = gather_chunk(lhs_dense, lhs, &mut a_store, lo, len);
            let b = gather_chunk(rhs_dense, rhs, &mut b_store, lo, len);
            if a.len() != len || b.len() != len {
                return;
            }
            write_chunk(out, cap, out_dense, shape, out_strides, lo, len, |dst| {
                chunk(lo, a, b, dst);
            });
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
