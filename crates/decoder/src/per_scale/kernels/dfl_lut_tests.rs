// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The exp-table DFL decode checked against the decoder's softmax level kernels.

use super::level_box::{
    decode_box_level_dfl_f32_to_f32, decode_box_level_dfl_i8_to_f32, decode_box_level_dfl_u8_to_f32,
};
use crate::per_scale::plan::{Layout, LevelPlan};
use crate::Quantization;
use edgefirst_kernels::prim::dfl_lut::{decode_dfl_lut, DflExpTable};
use edgefirst_kernels::prim::grids::make_anchor_grid;

fn level(w: usize, h: usize, reg_max: usize) -> LevelPlan {
    let (grid_x, grid_y) = make_anchor_grid(h, w);
    LevelPlan {
        stride: 16.0,
        h,
        w,
        reg_max,
        anchor_offset: 0,
        grid_x,
        grid_y,
        box_shape: vec![1, h, w, 4 * reg_max].into_boxed_slice(),
        score_shape: vec![1, h, w, 1].into_boxed_slice(),
        mc_shape: None,
        layout: Layout::Nhwc,
    }
}

/// DFL logits shaped like a trained head: each side peaks at a bin and
/// falls off, with noise, and saturated or flat sides mixed in.
fn bins(n: usize, reg_max: usize, seed: u64) -> Vec<i8> {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 32) as i32
    };
    let mut v = Vec::with_capacity(n * 4 * reg_max);
    for _ in 0..n * 4 {
        let kind = next().rem_euclid(16);
        let peak = next().rem_euclid(reg_max as i32);
        let top = next().rem_euclid(200) - 100;
        let fall = next().rem_euclid(12);
        for i in 0..reg_max as i32 {
            let b = match kind {
                0 => 127,
                1 => -128,
                2 => {
                    if i == peak {
                        127
                    } else {
                        -128
                    }
                }
                _ => top - fall * (i - peak).abs() + next().rem_euclid(9) - 4,
            };
            v.push(b.clamp(-128, 127) as i8);
        }
    }
    v
}

/// The f64 softmax expectation of each side, through dist2bbox in f64.
fn reference(input: &[i8], q: Quantization, level: &LevelPlan) -> Vec<f64> {
    let rm = level.reg_max;
    let mut out = Vec::new();
    for (a, anchor) in input.chunks_exact(4 * rm).enumerate() {
        let mut d = [0.0_f64; 4];
        for (s, side) in anchor.chunks_exact(rm).enumerate() {
            let x: Vec<f64> = side
                .iter()
                .map(|&b| (f64::from(b) - f64::from(q.zero_point)) * f64::from(q.scale))
                .collect();
            let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            d[s] = e.iter().enumerate().map(|(i, e)| e * i as f64).sum::<f64>() / sum;
        }
        let (gx, gy, st) = (
            f64::from(level.grid_x[a]),
            f64::from(level.grid_y[a]),
            f64::from(level.stride),
        );
        out.extend([
            (gx + (d[2] - d[0]) * 0.5) * st,
            (gy + (d[3] - d[1]) * 0.5) * st,
            (d[0] + d[2]) * st,
            (d[1] + d[3]) * st,
        ]);
    }
    out
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
}

/// The table kernel is at least as close to an f64 reference as the f32
/// dequant-then-softmax kernel, and within a few f32 rounding steps.
#[test]
fn matches_f64_reference_closer_than_f32_softmax() {
    for (scale, zp) in [(0.147, 8), (0.1, -40), (0.02, 0), (0.6, 100)] {
        let q = Quantization::new(scale, zp);
        let table = DflExpTable::new(scale);
        for reg_max in [16, 8, 7] {
            let lvl = level(9, 5, reg_max);
            let input = bins(45, reg_max, 7 + reg_max as u64);
            let want = reference(&input, q, &lvl);
            let mut lut = vec![0.0_f32; 4 * 45];
            decode_dfl_lut(&input, &table, &lvl.dfl_geom(), &mut lut);
            let mut f32_softmax = vec![0.0_f32; 4 * 45];
            decode_box_level_dfl_i8_to_f32(&input, q, &lvl, &mut f32_softmax);
            let (e_lut, e_f32) = (max_err(&lut, &want), max_err(&f32_softmax, &want));
            assert!(e_lut < 2e-4, "scale={scale} rm={reg_max}: lut err {e_lut}");
            assert!(
                e_lut <= e_f32 * 1.01 + 1e-5,
                "scale={scale} rm={reg_max}: lut {e_lut} f32 softmax {e_f32}"
            );
        }
    }
}

/// Unsigned bins decode as their signed equivalents shifted by 128.
#[test]
fn unsigned_bins_match_shifted_signed_bins() {
    let lvl = level(7, 3, 16);
    let input = bins(21, 16, 99);
    let unsigned: Vec<u8> = input.iter().map(|&b| (i16::from(b) + 128) as u8).collect();
    let table = DflExpTable::new(0.12);
    let mut s = vec![0.0_f32; 84];
    let mut u = vec![0.0_f32; 84];
    decode_dfl_lut(&input, &table, &lvl.dfl_geom(), &mut s);
    decode_dfl_lut(&unsigned, &table, &lvl.dfl_geom(), &mut u);
    assert_eq!(s, u);
    let mut legacy = vec![0.0_f32; 84];
    decode_box_level_dfl_u8_to_f32(&unsigned, Quantization::new(0.12, 128), &lvl, &mut legacy);
    assert!(
        max_err(
            &u,
            &legacy.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        ) < 1e-3
    );
}

#[test]
fn one_hot_bins_decode_to_their_index() {
    let lvl = level(1, 1, 16);
    let mut input = [-128_i8; 64];
    for (side, bin) in [3, 5, 7, 9].into_iter().enumerate() {
        input[side * 16 + bin] = 127;
    }
    let mut out = [0.0_f32; 4];
    decode_dfl_lut(&input, &DflExpTable::new(0.5), &lvl.dfl_geom(), &mut out);
    let mut want = [0.0_f32; 4];
    let mut logits = [0.0_f32; 64];
    for (side, bin) in [3, 5, 7, 9].into_iter().enumerate() {
        logits[side * 16 + bin] = 127.5;
    }
    decode_box_level_dfl_f32_to_f32(&logits, Quantization::identity(), &lvl, &mut want);
    assert_eq!(out, want);
}
