// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Greedy NMS on the NEON tiers: `prim::neon_nms` over the boxes gathered into padded
//! coordinate arrays, with the same keep/suppress decisions as the scalar loops in `float.rs`
//! and `byte.rs`.

use super::Isa;
use crate::BoundingBox;

/// Boxes below which NMS runs the scalar loop on every tier: gathering them costs more than the
/// loop (measured crossover on a Cortex-A55 with clustered boxes).
pub(crate) const NMS_MIN_BOXES: usize = 16;

/// Boxes below which the public NMS functions run the scalar loop without resolving the tier:
/// reading `EDGEFIRST_DECODER_FORCE_KERNEL` moves their crossover up from [`NMS_MIN_BOXES`].
pub(crate) const NMS_LOOKUP_MIN_BOXES: usize = 20;

/// The tier a public NMS over `n` boxes runs: [`Isa::current_or_scalar`], or the scalar code
/// below [`NMS_LOOKUP_MIN_BOXES`].
pub(crate) fn nms_isa(n: usize) -> Isa {
    if n < NMS_LOOKUP_MIN_BOXES {
        Isa::SCALAR
    } else {
        Isa::current_or_scalar()
    }
}

/// Boxes whose coordinate arrays live on the stack rather than the heap.
#[cfg(target_arch = "aarch64")]
const STACK_BOXES: usize = 64;

/// The suppression flags of a greedy NMS over `boxes`, sorted by descending score, on `isa`.
///
/// Box `k` is flagged when `dead(&boxes[k])` holds (the scalar loops' "already suppressed"
/// score) or an earlier unflagged box overlaps it (`jaccard(b_k, b_i, iou)`, and with its label
/// when `class_aware`), the scan stopping once `cap` boxes have survived. `None` when the caller
/// runs its scalar loop: the scalar tier, fewer than [`NMS_MIN_BOXES`] boxes, or a label above
/// `u32::MAX`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nms_dead<T>(
    isa: Isa,
    boxes: &[T],
    bbox: impl Fn(&T) -> &BoundingBox,
    class_aware: bool,
    label: impl Fn(&T) -> usize,
    dead: impl Fn(&T) -> bool,
    iou: f32,
    cap: usize,
) -> Option<Vec<bool>> {
    if isa.is_scalar() || boxes.len() < NMS_MIN_BOXES {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_nms::{nms_greedy_neon, LANES};
        let n = boxes.len();
        let np = n.next_multiple_of(LANES);
        let mut stack_coords = [0f32; 4 * STACK_BOXES];
        let mut stack_words = [0u32; 2 * STACK_BOXES];
        let (mut heap_coords, mut heap_words);
        let (coords, words) = if np <= STACK_BOXES {
            (&mut stack_coords[..4 * np], &mut stack_words[..2 * np])
        } else {
            heap_coords = vec![0f32; 4 * np];
            heap_words = vec![0u32; 2 * np];
            (&mut heap_coords[..], &mut heap_words[..])
        };
        let (xmin, rest) = coords.split_at_mut(np);
        let (ymin, rest) = rest.split_at_mut(np);
        let (xmax, ymax) = rest.split_at_mut(np);
        let (flags, labels) = words.split_at_mut(np);
        flags.fill(!0);
        for (k, b) in boxes.iter().enumerate() {
            let r = bbox(b);
            (xmin[k], ymin[k], xmax[k], ymax[k]) = (r.xmin, r.ymin, r.xmax, r.ymax);
            flags[k] = if dead(b) { !0 } else { 0 };
            if class_aware {
                labels[k] = u32::try_from(label(b)).ok()?;
            }
        }
        let labels = class_aware.then_some(&*labels);
        let coords = [&*xmin, &*ymin, &*xmax, &*ymax];
        // SAFETY: NEON is part of the aarch64 baseline.
        let ran = unsafe { nms_greedy_neon(coords, labels, iou, cap, flags) };
        ran.then(|| flags.iter().take(n).map(|&f| f != 0).collect())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (boxes, bbox, class_aware, label, dead, iou, cap);
        None
    }
}

/// `[jaccard(a, b[k], iou); 4]` as one NEON vector on `isa`; `None` when the caller runs the
/// scalar tests (scalar tier).
pub(crate) fn jaccard4_kernel(
    isa: Isa,
    a: &BoundingBox,
    b: &[BoundingBox; 4],
    iou: f32,
) -> Option<[bool; 4]> {
    #[cfg(test)]
    jaccard4_ran::record(!isa.is_scalar());
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let b = b.map(<[f32; 4]>::from);
        // SAFETY: NEON is part of the aarch64 baseline.
        Some(unsafe { edgefirst_kernels::prim::neon_nms::jaccard4_neon((*a).into(), &b, iou) })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (a, b, iou);
        None
    }
}

/// Whether the calling thread's last [`jaccard4_kernel`] call ran on a NEON tier.
#[cfg(test)]
pub(crate) mod jaccard4_ran {
    use std::cell::Cell;

    thread_local! {
        static RAN: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub(super) fn record(neon: bool) {
        RAN.with(|r| r.set(Some(neon)));
    }

    /// The last call's tier since the previous `take`, `Some(true)` for NEON.
    pub(crate) fn take() -> Option<bool> {
        RAN.with(Cell::take)
    }
}

/// The boxes of `boxes` not flagged in `dead` that `keep` accepts, in order, at most `cap`.
pub(crate) fn keep_unflagged<T>(
    boxes: Vec<T>,
    dead: Vec<bool>,
    keep: impl Fn(&T) -> bool,
    cap: usize,
) -> Vec<T> {
    boxes
        .into_iter()
        .zip(dead)
        .filter(|(b, d)| !d && keep(b))
        .map(|(b, _)| b)
        .take(cap)
        .collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
