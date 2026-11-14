// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Deterministic xorshift values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The decoder's scalar `float::jaccard`, verbatim.
fn jaccard(a: [f32; 4], b: [f32; 4], iou: f32) -> bool {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = a[2].min(b[2]);
    let bottom = a[3].min(b[3]);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - intersection;
    intersection > iou * union
}

/// The decoder's scalar greedy loop over `dead` flags.
fn greedy(boxes: &[[f32; 4]], labels: Option<&[u32]>, iou: f32, cap: usize, dead: &mut [bool]) {
    let mut survivors = 0;
    for i in 0..boxes.len() {
        if dead[i] {
            continue;
        }
        for j in (i + 1)..boxes.len() {
            if dead[j] {
                continue;
            }
            let same = labels.is_none_or(|l| l[j] == l[i]);
            if same && jaccard(boxes[j], boxes[i], iou) {
                dead[j] = true;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
}

/// Coordinates with specials: NaN (quiet, signalling, negative), infinities, signed zeros.
fn coord(rng: &mut Rng) -> f32 {
    match rng.next() % 97 {
        0 => f32::NAN,
        1 => f32::from_bits(0x7f80_0001),
        2 => -f32::NAN,
        3 => f32::INFINITY,
        4 => f32::NEG_INFINITY,
        5 => -0.0,
        6 => 0.0,
        _ => rng.unit(),
    }
}

/// Boxes in a few heavily overlapping clusters, with identical, zero-area, inverted and
/// special-valued boxes mixed in.
fn boxes(rng: &mut Rng, n: usize, specials: bool) -> Vec<[f32; 4]> {
    let centres: Vec<[f32; 2]> = (0..5).map(|_| [rng.unit(), rng.unit()]).collect();
    let mut out: Vec<[f32; 4]> = Vec::with_capacity(n);
    for k in 0..n {
        let b = match rng.next() % 16 {
            0 if k > 0 => out[(rng.next() as usize) % k],
            1 => {
                let x = rng.unit();
                [x, x, x, rng.unit()]
            }
            2 => [rng.unit(), rng.unit(), rng.unit() - 1.0, rng.unit() - 1.0],
            3 if specials => [coord(rng), coord(rng), coord(rng), coord(rng)],
            _ => {
                let [cx, cy] = centres[(rng.next() % 5) as usize];
                let (w, h) = (0.05 + 0.2 * rng.unit(), 0.05 + 0.2 * rng.unit());
                let (dx, dy) = (0.05 * rng.unit(), 0.05 * rng.unit());
                [cx + dx - w, cy + dy - h, cx + dx + w, cy + dy + h]
            }
        };
        out.push(b);
    }
    out
}

/// Runs the kernel on `boxes` padded to a multiple of four, `pre_dead` boxes flagged.
fn kernel(
    boxes: &[[f32; 4]],
    labels: Option<&[u32]>,
    iou: f32,
    cap: usize,
    pre_dead: &[bool],
) -> Vec<bool> {
    let n = boxes.len();
    let np = n.next_multiple_of(LANES);
    let mut coords = vec![vec![0f32; np]; 4];
    for (k, b) in boxes.iter().enumerate() {
        for c in 0..4 {
            coords[c][k] = b[c];
        }
    }
    let labels = labels.map(|l| {
        let mut v = l.to_vec();
        v.resize(np, 0);
        v
    });
    let mut dead: Vec<u32> = (0..np)
        .map(|k| if k >= n || pre_dead[k] { !0 } else { 0 })
        .collect();
    let c = [
        &coords[0][..],
        &coords[1][..],
        &coords[2][..],
        &coords[3][..],
    ];
    // SAFETY: NEON is part of the aarch64 baseline.
    assert!(unsafe { nms_greedy_neon(c, labels.as_deref(), iou, cap, &mut dead) });
    dead[..n].iter().map(|&d| d != 0).collect()
}

/// The kernel's suppression flags equal the scalar loop's for 23 box counts up to 333 (every
/// tail), clusters, identical, zero-area, inverted and NaN/infinite boxes, class-agnostic and
/// class-aware, eight thresholds, four caps and pre-flagged boxes.
#[test]
fn greedy_matches_the_scalar_loop() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let counts = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 64, 100, 257, 300, 301, 333,
    ];
    let ious = [0.0, 0.3, 0.45, 0.5, 0.7, 0.999, -0.5, f32::NAN];
    let mut cases = 0;
    for &n in &counts {
        for specials in [false, true] {
            let b = boxes(&mut rng, n, specials);
            let labels: Vec<u32> = (0..n).map(|_| (rng.next() % 3) as u32).collect();
            let pre_dead: Vec<bool> = (0..n).map(|_| rng.next().is_multiple_of(9)).collect();
            for &iou in &ious {
                for cap in [usize::MAX, 0, 1, 10] {
                    for class_aware in [false, true] {
                        let l = class_aware.then_some(labels.as_slice());
                        let mut want = pre_dead.clone();
                        greedy(&b, l, iou, cap, &mut want);
                        let got = kernel(&b, l, iou, cap, &pre_dead);
                        assert_eq!(
                            got, want,
                            "n {n} specials {specials} iou {iou} cap {cap} class_aware {class_aware}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, counts.len() * 2 * ious.len() * 4 * 2);
}

/// Identical boxes: the first suppresses every other one of its class.
#[test]
fn identical_boxes_leave_one_per_class() {
    let b = vec![[0.1f32, 0.1, 0.5, 0.5]; 11];
    let labels: Vec<u32> = (0..11).map(|k| (k % 2) as u32).collect();
    let got = kernel(&b, Some(&labels), 0.5, usize::MAX, &[false; 11]);
    let want: Vec<bool> = (0..11).map(|k| k > 1).collect();
    assert_eq!(got, want);
    let got = kernel(&b, None, 0.5, usize::MAX, &[false; 11]);
    assert_eq!(got, (0..11).map(|k| k > 0).collect::<Vec<_>>());
}

/// Mismatched lengths, a length not a multiple of four, or a short label slice: `false`, and
/// `dead` untouched.
#[test]
fn invalid_arguments_write_nothing() {
    let x = [0.0f32; 8];
    let x7 = [0.0f32; 7];
    let one = [1.0f32; 8];
    let labels = [0u32; 4];
    for (coords, labels, len, valid) in [
        ([&x[..], &x[..], &one[..], &one[..]], None, 8, true),
        ([&x7[..], &x7[..], &x7[..], &x7[..]], None, 7, false),
        ([&x[..], &x7[..], &one[..], &one[..]], None, 8, false),
        (
            [&x[..], &x[..], &one[..], &one[..]],
            Some(&labels[..]),
            8,
            false,
        ),
    ] {
        let mut dead = vec![7u32; len];
        // SAFETY: NEON is part of the aarch64 baseline.
        let ok = unsafe { nms_greedy_neon(coords, labels, 0.5, usize::MAX, &mut dead) };
        assert_eq!(ok, valid, "len {len}");
        assert!(dead.iter().all(|&d| d == 7));
    }
    let mut dead = [0u32; 4];
    // SAFETY: NEON is part of the aarch64 baseline.
    let ok = unsafe { nms_greedy_neon([&x[..4]; 4], None, 0.5, 1, &mut dead[..3]) };
    assert!(!ok);
}

/// `jaccard4_neon` equals the scalar test for both argument orders over random and special
/// boxes.
#[test]
fn jaccard4_matches_the_scalar_test() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..20_000 {
        let all = boxes(&mut rng, 5, true);
        let a = all[0];
        let b = [all[1], all[2], all[3], all[4]];
        for iou in [0.0, 0.5, 0.9, f32::NAN] {
            // SAFETY: NEON is part of the aarch64 baseline.
            let got = unsafe { jaccard4_neon(a, &b, iou) };
            for k in 0..4 {
                assert_eq!(got[k], jaccard(a, b[k], iou), "{a:?} {:?} {iou}", b[k]);
                assert_eq!(got[k], jaccard(b[k], a, iou), "{a:?} {:?} {iou}", b[k]);
            }
        }
    }
}

/// One FP16 operation: every `+`, `−` and `×` of two f16 values is exact in f32 arithmetic up to
/// one rounding, so rounding the f32 result to f16 rounds the exact result once.
fn h(x: f32) -> f16 {
    f16::from_f32(x)
}

/// The FP16 kernel's test, one rounded operation at a time.
fn jaccard_f16(a: [f16; 4], b: [f16; 4], iou: f16) -> bool {
    let [a, b] = [a, b].map(|v| v.map(f16::to_f32));
    let area = |v: [f32; 4]| h(h(v[2] - v[0]).to_f32() * h(v[3] - v[1]).to_f32()).to_f32();
    let w = h(a[2].min(b[2]) - a[0].max(b[0])).to_f32().max(0.0);
    let ht = h(a[3].min(b[3]) - a[1].max(b[1])).to_f32().max(0.0);
    let inter = h(w * ht).to_f32();
    let union = h(h(area(b) + area(a)).to_f32() - inter).to_f32();
    inter > h(iou.to_f32() * union).to_f32()
}

/// Runs the FP16 kernel on `boxes` padded to a multiple of eight, `pre_dead` boxes flagged.
fn kernel_fp16(boxes: &[[f16; 4]], iou: f16, cap: usize, pre_dead: &[bool]) -> Vec<bool> {
    let n = boxes.len();
    let np = n.next_multiple_of(LANES_FP16);
    let mut coords = vec![vec![f16::ZERO; np]; 4];
    for (k, b) in boxes.iter().enumerate() {
        for c in 0..4 {
            coords[c][k] = b[c];
        }
    }
    let mut area = vec![f16::ZERO; np];
    let mut dead: Vec<u16> = (0..np)
        .map(|k| if k >= n || pre_dead[k] { !0 } else { 0 })
        .collect();
    let c = [
        &coords[0][..],
        &coords[1][..],
        &coords[2][..],
        &coords[3][..],
    ];
    // SAFETY: the callers checked FEAT_FP16.
    assert!(unsafe { nms_greedy_neon_fp16(c, &mut area, iou, cap, &mut dead) });
    dead[..n].iter().map(|&d| d != 0).collect()
}

/// The FP16 kernel's flags equal a scalar greedy loop over the same rounded operations, for
/// every tail up to 333 boxes, clustered, identical, zero-area, inverted, infinite and quiet-NaN
/// boxes at scales from 1 to 64, six thresholds, four caps and pre-flagged boxes.
#[test]
fn fp16_greedy_matches_the_rounded_scalar_loop() {
    if !std::arch::is_aarch64_feature_detected!("fp16") {
        crate::test_support::report_skip(
            "fp16_greedy_matches_the_rounded_scalar_loop needs FEAT_FP16",
        );
        return;
    }
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let counts = [
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 41, 64, 100, 257, 300, 333,
    ];
    let ious = [0.0, 0.3, 0.45, 0.5, 0.7, 0.999].map(h);
    let mut cases = 0;
    for &n in &counts {
        for (specials, scale) in [(false, 1.0), (false, 64.0), (true, 40.0)] {
            let b: Vec<[f16; 4]> = boxes(&mut rng, n, specials)
                .into_iter()
                .map(|b| {
                    b.map(|x| {
                        // Signalling NaNs are quieted: the model's f32 detour would quiet them.
                        if x.is_nan() {
                            f16::NAN
                        } else {
                            h(x * scale)
                        }
                    })
                })
                .collect();
            let pre_dead: Vec<bool> = (0..n).map(|_| rng.next().is_multiple_of(9)).collect();
            for &iou in &ious {
                for cap in [usize::MAX, 0, 1, 10] {
                    let mut want = pre_dead.clone();
                    let mut survivors = 0;
                    for i in 0..n {
                        if want[i] {
                            continue;
                        }
                        for j in (i + 1)..n {
                            want[j] |= jaccard_f16(b[j], b[i], iou);
                        }
                        survivors += 1;
                        if survivors >= cap {
                            break;
                        }
                    }
                    let got = kernel_fp16(&b, iou, cap, &pre_dead);
                    assert_eq!(got, want, "n {n} specials {specials} iou {iou} cap {cap}");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, counts.len() * 3 * ious.len() * 4);
}

/// On boxes whose overlaps are far from the threshold, the FP16 kernel flags the boxes the f32
/// kernel flags.
#[test]
fn fp16_agrees_with_f32_away_from_the_threshold() {
    if !std::arch::is_aarch64_feature_detected!("fp16") {
        crate::test_support::report_skip(
            "fp16_agrees_with_f32_away_from_the_threshold needs FEAT_FP16",
        );
        return;
    }
    // A 6 x 6 grid of disjoint cells, each holding a box and two near copies of it.
    let mut b = Vec::new();
    for cell in 0..36 {
        let (x, y) = ((cell % 6) as f32 * 10.0, (cell / 6) as f32 * 10.0);
        for shift in [0.0, 0.25, 0.5] {
            b.push([x + shift, y + shift, x + 8.0 + shift, y + 8.0 + shift]);
        }
    }
    let f32_dead = kernel(&b, None, 0.5, usize::MAX, &vec![false; b.len()]);
    let b16: Vec<[f16; 4]> = b.iter().map(|b| b.map(h)).collect();
    let fp16_dead = kernel_fp16(&b16, h(0.5), usize::MAX, &vec![false; b.len()]);
    assert_eq!(fp16_dead, f32_dead);
    assert_eq!(fp16_dead.iter().filter(|&&d| !d).count(), 36);
}

/// Mismatched lengths or a length not a multiple of eight: `false`, and `dead` untouched.
#[test]
fn fp16_invalid_arguments_write_nothing() {
    if !std::arch::is_aarch64_feature_detected!("fp16") {
        crate::test_support::report_skip("fp16_invalid_arguments_write_nothing needs FEAT_FP16");
        return;
    }
    let x = [f16::ZERO; 16];
    for (lens, area_len, dead_len, valid) in [
        ([16, 16, 16, 16], 16, 16, true),
        ([12, 12, 12, 12], 12, 12, false),
        ([16, 8, 16, 16], 16, 16, false),
        ([16, 16, 16, 16], 8, 16, false),
        ([8, 8, 8, 8], 8, 16, false),
    ] {
        let mut area = vec![f16::ONE; area_len];
        let mut dead = vec![7u16; dead_len];
        let c = lens.map(|l| &x[..l]);
        // SAFETY: FEAT_FP16 was checked above.
        let ok = unsafe { nms_greedy_neon_fp16(c, &mut area, h(0.5), usize::MAX, &mut dead) };
        assert_eq!(ok, valid, "{lens:?} {area_len} {dead_len}");
        if !valid {
            assert!(dead.iter().all(|&d| d == 7));
            assert!(area.iter().all(|&a| a == f16::ONE));
        }
    }
}
