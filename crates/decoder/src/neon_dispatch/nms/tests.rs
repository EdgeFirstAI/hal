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

/// Every NEON tier runs the NMS kernel for class-agnostic and class-aware boxes; the scalar
/// tier and labels above `u32::MAX` leave it to the scalar loop.
#[test]
fn every_neon_tier_runs_the_nms_kernel() {
    let mut rng = Rng(42);
    let boxes = float_boxes(&mut rng, 37, false);
    for (name, isa) in tiers() {
        for class_aware in [false, true] {
            let r = nms_dead(
                isa,
                &boxes,
                |b| &b.bbox,
                class_aware,
                |b| b.label,
                |b| b.score < 0.0,
                0.5,
                usize::MAX,
            );
            assert_eq!(
                r.is_some(),
                !isa.is_scalar(),
                "{name} class_aware {class_aware}"
            );
        }
        let r = nms_dead(
            isa,
            &boxes,
            |b| &b.bbox,
            true,
            |b| b.label + u32::MAX as usize,
            |b| b.score < 0.0,
            0.5,
            usize::MAX,
        );
        assert!(r.is_none(), "{name}: a label above u32::MAX runs scalar");
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
