// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DFL (Distribution Focal Loss) primitives for Hailo-style YOLOv8/v11
//! split-output decoding. Pure-function module, no tensor-framework
//! dependencies — operates on `&[f32]` / `Vec<f32>` so it's trivially
//! unit-testable and can be SIMD-ified later without API churn.
//!
//! Reference: `HAILORT_DECODER.md` §"Reference Implementation Tour"
//! and `edgefirst-validator @ feature/DE-823-hailort`:
//! `edgefirst/validator/runners/processing/hailo_decode.py`.

/// Build a row-major anchor grid with centres offset by `+0.5`
/// (Ultralytics convention). Returns `(grid_x, grid_y)` of length
/// `h * w` each, indexed `[y * w + x]`. Coordinates are in **grid
/// units**, not pixels — the caller multiplies by stride inside
/// `dist2bbox`.
pub(crate) fn make_anchor_grid(h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = Vec::with_capacity(h * w);
    let mut gy = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            gx.push(x as f32 + 0.5);
            gy.push(y as f32 + 0.5);
        }
    }
    (gx, gy)
}

/// DFL weighted-sum coefficients: `[0.0, 1.0, ..., (reg_max-1) as f32]`.
pub(crate) fn dfl_bins(reg_max: usize) -> Vec<f32> {
    (0..reg_max).map(|i| i as f32).collect()
}

/// Numerically stable softmax in-place: subtracts `max(buf)` before
/// `exp` to prevent overflow on large logits.
pub(crate) fn softmax_inplace(buf: &mut [f32]) {
    if buf.is_empty() {
        return;
    }
    let mut m = buf[0];
    for &v in buf.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in buf.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in buf.iter_mut() {
        *v *= inv;
    }
}

/// Decode one FPN-level DFL bbox tensor to xcycwh pixel coordinates.
///
/// * `bbox` — flattened `(H, W, 4 * reg_max)` f32 buffer, row-major
///   (NHWC inner layout, as HailoRT returns post-dequant).
/// * `grid_x`, `grid_y` — precomputed per-anchor grid centres in grid
///   units (see [`make_anchor_grid`]).
/// * `stride` — FPN stride in pixels (e.g. 8, 16, 32).
///
/// Returns a flattened `(H * W, 4)` buffer `[xc, yc, w, h, xc, yc,
/// w, h, ...]` in pixel coordinates of the model input.
pub(crate) fn decode_dfl_level(
    bbox: &[f32],
    h: usize,
    w: usize,
    reg_max: usize,
    grid_x: &[f32],
    grid_y: &[f32],
    stride: f32,
) -> Vec<f32> {
    debug_assert_eq!(bbox.len(), h * w * 4 * reg_max);
    debug_assert_eq!(grid_x.len(), h * w);
    debug_assert_eq!(grid_y.len(), h * w);

    let bins = dfl_bins(reg_max);
    let mut out = Vec::with_capacity(h * w * 4);
    let mut scratch = vec![0.0f32; reg_max];
    for anchor in 0..(h * w) {
        let base = anchor * 4 * reg_max;
        let mut d = [0.0f32; 4]; // [left, top, right, bottom]
        for (side, d_side) in d.iter_mut().enumerate() {
            let slot = &bbox[base + side * reg_max..base + (side + 1) * reg_max];
            scratch.copy_from_slice(slot);
            softmax_inplace(&mut scratch);
            let mut s = 0.0f32;
            for (i, p) in scratch.iter().enumerate() {
                s += p * bins[i];
            }
            *d_side = s;
        }
        let gx = grid_x[anchor];
        let gy = grid_y[anchor];
        // Order of operations matters — `(grid + (r - l) / 2) * stride`
        // matches Ultralytics bit-for-bit; a `grid * stride + ...`
        // refactoring would differ by float ulps.
        let xc = (gx + (d[2] - d[0]) * 0.5) * stride;
        let yc = (gy + (d[3] - d[1]) * 0.5) * stride;
        let bw = (d[0] + d[2]) * stride;
        let bh = (d[1] + d[3]) * stride;
        out.push(xc);
        out.push(yc);
        out.push(bw);
        out.push(bh);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_anchor_grid_centres_match_ultralytics_convention() {
        let (gx, gy) = make_anchor_grid(2, 2);
        // Row-major (y-outer, x-inner). Centres: (0.5, 0.5), (1.5, 0.5),
        // (0.5, 1.5), (1.5, 1.5).
        assert_eq!(gx, vec![0.5, 1.5, 0.5, 1.5]);
        assert_eq!(gy, vec![0.5, 0.5, 1.5, 1.5]);
    }

    #[test]
    fn dfl_bins_is_arange() {
        assert_eq!(dfl_bins(4), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(dfl_bins(16).len(), 16);
        assert_eq!(dfl_bins(16)[15], 15.0);
    }

    #[test]
    fn softmax_stable_against_large_logits() {
        // Naive `exp(1002)` overflows; stable subtract-max keeps it finite.
        let mut buf = [1000.0f32, 1001.0, 1002.0];
        softmax_inplace(&mut buf);
        let sum: f32 = buf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(buf.iter().all(|v| v.is_finite()));
        assert!(buf[2] > buf[1] && buf[1] > buf[0]);
    }

    #[test]
    fn softmax_uniform_is_uniform() {
        let mut buf = [1.0f32; 8];
        softmax_inplace(&mut buf);
        for v in buf {
            assert!((v - 0.125).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_dfl_level_uniform_distribution() {
        // All-equal DFL logits → uniform softmax → distance = (reg_max-1)/2
        // for each side. reg_max=16, stride=8, one cell at grid (0.5, 0.5):
        //   d_left = d_right = d_top = d_bottom = 7.5
        //   xc = (0.5 + 0) * 8 = 4.0
        //   w  = (7.5 + 7.5) * 8 = 120.0
        let reg_max = 16usize;
        let bbox = vec![1.0f32; 4 * reg_max];
        let (gx, gy) = make_anchor_grid(1, 1);
        let out = decode_dfl_level(&bbox, 1, 1, reg_max, &gx, &gy, 8.0);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 4.0).abs() < 1e-4, "xc={}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-4, "yc={}", out[1]);
        assert!((out[2] - 120.0).abs() < 1e-3, "w={}", out[2]);
        assert!((out[3] - 120.0).abs() < 1e-3, "h={}", out[3]);
    }

    #[test]
    fn decode_dfl_level_concentrated_distribution() {
        // One-hot DFL distribution: logit at bin k huge, others 0
        // → distance = k. reg_max=16, stride=16, k=5:
        //   xc = (0.5 + 0) * 16 = 8.0
        //   w  = (5 + 5) * 16 = 160.0
        let reg_max = 16usize;
        let mut bbox = vec![0.0f32; 4 * reg_max];
        for side in 0..4 {
            bbox[side * reg_max + 5] = 1000.0;
        }
        let (gx, gy) = make_anchor_grid(1, 1);
        let out = decode_dfl_level(&bbox, 1, 1, reg_max, &gx, &gy, 16.0);
        assert!((out[0] - 8.0).abs() < 1e-4, "xc={}", out[0]);
        assert!((out[2] - 160.0).abs() < 1e-3, "w={}", out[2]);
    }

    #[test]
    fn decode_dfl_level_multi_cell_orders_row_major() {
        // 2x2 grid at stride 8, uniform logits → every cell has the same
        // width (120) and its xc/yc follows (gx+0)*8 / (gy+0)*8.
        let reg_max = 16usize;
        let bbox = vec![1.0f32; 2 * 2 * 4 * reg_max];
        let (gx, gy) = make_anchor_grid(2, 2);
        let out = decode_dfl_level(&bbox, 2, 2, reg_max, &gx, &gy, 8.0);
        assert_eq!(out.len(), 4 * 4);
        // Anchor 0: grid (0.5, 0.5) → xc=4, yc=4
        assert!((out[0] - 4.0).abs() < 1e-4);
        assert!((out[1] - 4.0).abs() < 1e-4);
        // Anchor 1: grid (1.5, 0.5) → xc=12, yc=4
        assert!((out[4] - 12.0).abs() < 1e-4);
        assert!((out[5] - 4.0).abs() < 1e-4);
        // Anchor 2: grid (0.5, 1.5) → xc=4, yc=12
        assert!((out[8] - 4.0).abs() < 1e-4);
        assert!((out[9] - 12.0).abs() < 1e-4);
        // Anchor 3: grid (1.5, 1.5) → xc=12, yc=12
        assert!((out[12] - 12.0).abs() < 1e-4);
        assert!((out[13] - 12.0).abs() < 1e-4);
    }
}
