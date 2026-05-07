// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Pre-computed per-FPN-level anchor centres.

/// Build row-major `(grid_x, grid_y)` arrays of length `h*w` with centres
/// at `+0.5` (Ultralytics convention).  Indexing: `[y * w + x]`. Output
/// is in grid units; caller multiplies by stride during dist2bbox.
#[allow(dead_code)]
pub(crate) fn make_anchor_grid(h: usize, w: usize) -> (Box<[f32]>, Box<[f32]>) {
    let n = h * w;
    let mut gx = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            gx.push(x as f32 + 0.5);
            gy.push(y as f32 + 0.5);
        }
    }
    (gx.into_boxed_slice(), gy.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_anchor_grid_2x2_centres_at_half_offset() {
        let (gx, gy) = make_anchor_grid(2, 2);
        // Row-major (y-outer, x-inner): centres (0.5,0.5), (1.5,0.5), (0.5,1.5), (1.5,1.5)
        assert_eq!(&*gx, &[0.5_f32, 1.5, 0.5, 1.5]);
        assert_eq!(&*gy, &[0.5_f32, 0.5, 1.5, 1.5]);
    }

    #[test]
    fn make_anchor_grid_lengths_match_h_times_w() {
        for (h, w) in [(80, 80), (40, 40), (20, 20), (1, 1), (3, 5)] {
            let (gx, gy) = make_anchor_grid(h, w);
            assert_eq!(gx.len(), h * w);
            assert_eq!(gy.len(), h * w);
        }
    }

    #[test]
    fn make_anchor_grid_zero_size_is_empty() {
        let (gx, gy) = make_anchor_grid(0, 5);
        assert!(gx.is_empty());
        assert!(gy.is_empty());
    }
}
