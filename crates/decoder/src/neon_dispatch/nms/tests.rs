// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Every public NMS function on every tier this CPU has against the scalar tier, decision for
//! decision and in order, and the kernel each tier runs.

use super::super::tiers;
use super::*;
use crate::byte::{nms_class_aware_int, nms_extra_class_aware_int, nms_extra_int, nms_int};
use crate::float::{
    jaccard, jaccard_batch4, nms_class_aware_float, nms_extra_class_aware_float, nms_extra_float,
    nms_float,
};
use crate::per_scale::kernels::forced_for_test;
use crate::{DetectBox, DetectBoxQuantized};

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

/// A box in one of a few overlapping clusters, or identical to an earlier one, zero-area,
/// inverted, or (with `specials`) holding NaN or infinite coordinates.
fn bbox(rng: &mut Rng, earlier: &[BoundingBox], specials: bool) -> BoundingBox {
    let special = |rng: &mut Rng| match rng.next() % 6 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        3 => -0.0,
        _ => rng.unit(),
    };
    match rng.next() % 16 {
        0 if !earlier.is_empty() => earlier[(rng.next() as usize) % earlier.len()],
        1 => {
            let x = rng.unit();
            BoundingBox::new(x, x, x, rng.unit())
        }
        2 => BoundingBox::new(rng.unit(), rng.unit(), rng.unit() - 1.0, rng.unit() - 1.0),
        3 if specials => BoundingBox::new(special(rng), special(rng), special(rng), special(rng)),
        _ => {
            let c = (rng.next() % 4) as f32 * 0.25 + 0.1;
            let (w, h) = (0.05 + 0.15 * rng.unit(), 0.05 + 0.15 * rng.unit());
            let (dx, dy) = (0.05 * rng.unit(), 0.05 * rng.unit());
            BoundingBox::new(c + dx - w, c + dy - h, c + dx + w, c + dy + h)
        }
    }
}

/// `n` float boxes with tied, negative and (with `specials`) NaN scores, labels from 3 classes.
fn float_boxes(rng: &mut Rng, n: usize, specials: bool) -> Vec<DetectBox> {
    let mut seen = Vec::new();
    (0..n)
        .map(|_| {
            let b = bbox(rng, &seen, specials);
            seen.push(b);
            let score = match rng.next() % 23 {
                0 => -0.5,
                1 if specials => f32::NAN,
                2 if specials => -f32::NAN,
                3 => 0.5,
                _ => rng.unit(),
            };
            DetectBox {
                bbox: b,
                score,
                label: (rng.next() % 3) as usize,
            }
        })
        .collect()
}

/// `n` quantized boxes with tied and minimum scores.
fn int_boxes(rng: &mut Rng, n: usize, specials: bool) -> Vec<DetectBoxQuantized<i8>> {
    float_boxes(rng, n, specials)
        .into_iter()
        .map(|b| DetectBoxQuantized {
            bbox: b.bbox,
            score: match rng.next() % 17 {
                0 => i8::MIN,
                _ => (rng.next() % 9) as i8 - 4,
            },
            label: b.label,
        })
        .collect()
}

/// A box as comparable bits.
fn key(b: &BoundingBox, score: u64, label: usize) -> [u64; 6] {
    [
        b.xmin.to_bits() as u64,
        b.ymin.to_bits() as u64,
        b.xmax.to_bits() as u64,
        b.ymax.to_bits() as u64,
        score,
        label as u64,
    ]
}

/// The result of `run` under each tier this CPU has equals the scalar tier's.
fn same_on_every_tier(what: &str, run: impl Fn() -> Vec<[u64; 7]>) {
    let want = forced_for_test::with("scalar", &run);
    for (name, _) in tiers() {
        let got = forced_for_test::with(name, &run);
        assert_eq!(got, want, "{what} on {name}");
    }
}

/// The eight public NMS functions equal the scalar tier on every tier, for 12 box counts up to
/// 333 (every tail of four), clustered, identical, zero-area, inverted and NaN/infinite boxes,
/// negative/NaN/minimum and tied scores, mixed classes, six thresholds and four caps.
#[test]
fn nms_matches_the_scalar_tier_on_every_tier() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for n in [0, 1, 3, 4, 5, 7, 8, 17, 64, 100, 300, 333] {
        for specials in [false, true] {
            let fb = float_boxes(&mut rng, n, specials);
            let ib = int_boxes(&mut rng, n, specials);
            for iou in [0.0, 0.3, 0.5, 0.7, f32::NAN, 1.0] {
                for max_det in [None, Some(0), Some(1), Some(10)] {
                    let what = format!("n {n} specials {specials} iou {iou} max_det {max_det:?}");
                    let f = |b: &DetectBox, e: usize| {
                        let [a, b2, c, d, s, l] = key(&b.bbox, b.score.to_bits() as u64, b.label);
                        [a, b2, c, d, s, l, e as u64]
                    };
                    let i = |b: &DetectBoxQuantized<i8>, e: usize| {
                        let [a, b2, c, d, s, l] = key(&b.bbox, b.score as u8 as u64, b.label);
                        [a, b2, c, d, s, l, e as u64]
                    };
                    let indexed = |v: &[DetectBox]| v.iter().copied().zip(0..).collect::<Vec<_>>();
                    let indexed_i = |v: &[DetectBoxQuantized<i8>]| {
                        v.iter().copied().zip(0..).collect::<Vec<_>>()
                    };
                    same_on_every_tier(&format!("nms_float {what}"), || {
                        let r = nms_float(iou, max_det, fb.clone());
                        r.iter().map(|b| f(b, 0)).collect()
                    });
                    same_on_every_tier(&format!("nms_class_aware_float {what}"), || {
                        let r = nms_class_aware_float(iou, max_det, fb.clone());
                        r.iter().map(|b| f(b, 0)).collect()
                    });
                    same_on_every_tier(&format!("nms_extra_float {what}"), || {
                        let r = nms_extra_float(iou, max_det, indexed(&fb));
                        r.iter().map(|(b, e)| f(b, *e)).collect()
                    });
                    same_on_every_tier(&format!("nms_extra_class_aware_float {what}"), || {
                        let r = nms_extra_class_aware_float(iou, max_det, indexed(&fb));
                        r.iter().map(|(b, e)| f(b, *e)).collect()
                    });
                    same_on_every_tier(&format!("nms_int {what}"), || {
                        let r = nms_int(iou, max_det, ib.clone());
                        r.iter().map(|b| i(b, 0)).collect()
                    });
                    same_on_every_tier(&format!("nms_class_aware_int {what}"), || {
                        let r = nms_class_aware_int(iou, max_det, ib.clone());
                        r.iter().map(|b| i(b, 0)).collect()
                    });
                    same_on_every_tier(&format!("nms_extra_int {what}"), || {
                        let r = nms_extra_int(iou, max_det, indexed_i(&ib));
                        r.iter().map(|(b, e)| i(b, *e)).collect()
                    });
                    same_on_every_tier(&format!("nms_extra_class_aware_int {what}"), || {
                        let r = nms_extra_class_aware_int(iou, max_det, indexed_i(&ib));
                        r.iter().map(|(b, e)| i(b, *e)).collect()
                    });
                }
            }
        }
    }
}

/// Every NEON tier runs the NMS kernel for class-agnostic and class-aware boxes and the scalar
/// tier leaves it to the scalar loop. Among at most `SMALL_NMS_BOXES` a label above `u32::MAX`
/// runs scalar; above it, labels too sparse for the counting sort flag the boxes dense labels do.
#[test]
fn every_neon_tier_runs_the_nms_kernel() {
    let mut rng = Rng(42);
    let small = float_boxes(&mut rng, 37, false);
    let large = float_boxes(&mut rng, 2 * SMALL_NMS_BOXES + 9, false);
    let sparse = |b: &DetectBox| b.label * (u32::MAX as usize) + 7;
    for (name, isa) in tiers() {
        let run = |boxes: &[DetectBox], class_aware: bool, label: &dyn Fn(&DetectBox) -> usize| {
            nms_dead(
                isa,
                boxes,
                |b| &b.bbox,
                class_aware,
                label,
                |b| b.score < 0.0,
                |b| b.score >= 0.0,
                0.5,
                usize::MAX,
            )
        };
        for boxes in [&small, &large] {
            for class_aware in [false, true] {
                let r = run(boxes, class_aware, &|b| b.label);
                assert_eq!(
                    r.is_some(),
                    !isa.is_scalar(),
                    "{name} n {} class_aware {class_aware}",
                    boxes.len()
                );
            }
        }
        assert!(
            run(&small, true, &sparse).is_none(),
            "{name}: small, sparse"
        );
        assert_eq!(
            run(&large, true, &sparse),
            run(&large, true, &|b| b.label),
            "{name}: large, sparse"
        );
    }
}

/// `sort_by_score_desc` is the stable descending sort in IEEE total order: NaN of either sign,
/// infinities, signed zeros and ties included, for sizes on both sides of the parallel cut.
#[test]
fn score_sort_is_the_stable_total_order_sort() {
    let mut rng = Rng(0xfeed_beef);
    for n in [
        0,
        1,
        2,
        3,
        17,
        300,
        PAR_SORT_MIN_BOXES,
        PAR_SORT_MIN_BOXES + 13,
    ] {
        let mut boxes: Vec<(f32, usize)> = (0..n)
            .map(|k| {
                let score = match rng.next() % 12 {
                    0 => f32::NAN,
                    1 => -f32::NAN,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    4 => 0.0,
                    5 => -0.0,
                    6 => 0.5,
                    7 => -0.5,
                    _ => rng.unit() - 0.5,
                };
                (score, k)
            })
            .collect();
        let mut want = boxes.clone();
        want.sort_by(|a, b| b.0.total_cmp(&a.0));
        sort_by_score_desc(&mut boxes, |b| b.0);
        let bits = |v: &[(f32, usize)]| v.iter().map(|b| (b.0.to_bits(), b.1)).collect::<Vec<_>>();
        assert_eq!(bits(&boxes), bits(&want), "n {n}");
    }
}

/// FP16 overlap tests flag the boxes the f32 tests flag when no overlap is near the threshold,
/// at pixel and normalized scales, and non-finite coordinates take the f32 tests.
#[test]
fn f16_nms_agrees_with_f32_away_from_the_threshold() {
    for (name, isa) in tiers() {
        if !isa.fp16() {
            continue;
        }
        assert!(isa.with_f16_nms().f16_nms() && !isa.f16_nms(), "{name}");
        for scale in [1.0 / 640.0, 1.0, 3.0] {
            // Disjoint cells, each holding a box and two near copies of it; two labels, so a
            // class-aware pass is long enough for the FP16 tests.
            let mut boxes = Vec::new();
            for cell in 0..144 {
                let (x, y) = ((cell % 12) as f32 * 100.0, (cell / 12) as f32 * 100.0);
                for (k, shift) in [0.0, 2.0, 4.0].into_iter().enumerate() {
                    let c = |v: f32| (v + shift) * scale;
                    boxes.push(DetectBox {
                        bbox: BoundingBox::new(c(x), c(y), c(x + 64.0), c(y + 64.0)),
                        score: 1.0 - (3 * cell + k) as f32 / 2000.0,
                        label: cell % 2,
                    });
                }
            }
            sort_by_score_desc(&mut boxes, |b| b.score);
            for class_aware in [false, true] {
                let run = |isa: Isa, boxes: &[DetectBox]| {
                    nms_dead(
                        isa,
                        boxes,
                        |b| &b.bbox,
                        class_aware,
                        |b| b.label,
                        |_| false,
                        |_| true,
                        0.5,
                        usize::MAX,
                    )
                };
                let f32_dead = run(isa, &boxes);
                assert_eq!(run(isa.with_f16_nms(), &boxes), f32_dead, "{name} {scale}");
                let kept = f32_dead.map(|d| d.iter().filter(|&&d| !d).count());
                assert_eq!(kept, Some(144), "{name} {scale}");

                let mut special = boxes.clone();
                special[5].bbox.xmax = f32::NAN;
                special[40].bbox.ymin = f32::INFINITY;
                assert_eq!(
                    run(isa.with_f16_nms(), &special),
                    run(isa, &special),
                    "{name} {scale}: non-finite coordinates"
                );
            }
        }
    }
}

/// FP16 overlap tests keep the extent of small boxes far from the origin: 65 copies of a box a
/// quarter pixel wide at 640 leave one, as in f32, and boxes too thin for f16 areas take the
/// f32 tests.
#[test]
fn f16_nms_keeps_small_boxes_far_from_the_origin() {
    for (name, isa) in tiers() {
        if !isa.fp16() {
            continue;
        }
        for (w, h) in [(0.25, 8.0), (8.0, 0.25), (0.001, 8.0), (0.0, 8.0)] {
            let boxes: Vec<DetectBox> = (0..65)
                .map(|k| DetectBox {
                    bbox: BoundingBox::new(
                        640.0 - w / 2.0,
                        640.0 - h / 2.0,
                        640.0 + w / 2.0,
                        640.0 + h / 2.0,
                    ),
                    score: 1.0 - k as f32 / 100.0,
                    label: 0,
                })
                .collect();
            for class_aware in [false, true] {
                let run = |isa: Isa| {
                    nms_dead(
                        isa,
                        &boxes,
                        |b| &b.bbox,
                        class_aware,
                        |b| b.label,
                        |_| false,
                        |_| true,
                        0.7,
                        usize::MAX,
                    )
                };
                let f32_dead = run(isa);
                assert_eq!(run(isa.with_f16_nms()), f32_dead, "{name} {w}x{h}");
                let kept = f32_dead.map(|d| d.iter().filter(|&&d| !d).count());
                let want = if w > 0.0 { 1 } else { 65 };
                assert_eq!(kept, Some(want), "{name} {w}x{h}");
            }
        }
    }
}

/// A public NMS over fewer than `NMS_LOOKUP_MIN_BOXES` boxes runs scalar without resolving the
/// tier, and from it on the tier; the kernel declines fewer than `NMS_MIN_BOXES` boxes.
#[test]
fn small_nms_runs_the_scalar_loop() {
    let mut rng = Rng(9);
    let boxes = float_boxes(&mut rng, NMS_MIN_BOXES, false);
    for (name, isa) in tiers() {
        let small = forced_for_test::with(name, || nms_isa(NMS_LOOKUP_MIN_BOXES - 1));
        assert_eq!(small, Isa::SCALAR, "{name}");
        let large = forced_for_test::with(name, || nms_isa(NMS_LOOKUP_MIN_BOXES));
        assert_eq!(large, isa, "{name}");
        let run = |b: &[DetectBox]| {
            nms_dead(
                isa,
                b,
                |b| &b.bbox,
                false,
                |b| b.label,
                |_| false,
                |_| true,
                0.5,
                usize::MAX,
            )
        };
        assert!(run(&boxes[1..]).is_none(), "{name}");
        assert_eq!(run(&boxes).is_some(), !isa.is_scalar(), "{name}");
    }
}

/// Many identical and overlapping same-class boxes keep exactly one per class on every tier.
#[test]
fn nms_keeps_one_of_each_identical_class_cluster() {
    let mut boxes = Vec::new();
    for k in 0..310 {
        boxes.push(DetectBox {
            bbox: BoundingBox::new(0.1, 0.1, 0.4, 0.4),
            score: 1.0 - k as f32 / 1000.0,
            label: k % 2,
        });
    }
    for (name, _) in tiers() {
        let kept = forced_for_test::with(name, || nms_class_aware_float(0.5, None, boxes.clone()));
        let labels: Vec<usize> = kept.iter().map(|b| b.label).collect();
        assert_eq!(labels, [0, 1], "{name}");
        let kept = forced_for_test::with(name, || nms_float(0.5, None, boxes.clone()));
        assert_eq!(kept.len(), 1, "{name}");
    }
}

/// `jaccard_batch4` runs the NEON test on NEON tiers and the scalar one on the scalar tier.
#[test]
fn jaccard_batch4_follows_the_forced_tier() {
    let a = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
    let b = [a; 4];
    for (name, isa) in tiers() {
        let _ = super::jaccard4_ran::take();
        let got = forced_for_test::with(name, || jaccard_batch4(&a, &b, 0.5));
        assert_eq!(got, [true; 4], "{name}");
        assert_eq!(
            super::jaccard4_ran::take(),
            Some(!isa.is_scalar()),
            "{name}"
        );
    }
}

/// `jaccard_batch4` equals `jaccard` on random, special and NaN boxes.
#[test]
fn jaccard_batch4_equals_jaccard_on_every_input() {
    let mut rng = Rng(7);
    for _ in 0..5000 {
        let mut seen = Vec::new();
        for _ in 0..5 {
            let b = bbox(&mut rng, &seen, true);
            seen.push(b);
        }
        let b = [seen[1], seen[2], seen[3], seen[4]];
        for iou in [0.0, 0.5, f32::NAN] {
            let got = jaccard_batch4(&seen[0], &b, iou);
            for k in 0..4 {
                assert_eq!(
                    got[k],
                    jaccard(&seen[0], &b[k], iou),
                    "{:?} {:?}",
                    seen[0],
                    b[k]
                );
            }
        }
    }
}
