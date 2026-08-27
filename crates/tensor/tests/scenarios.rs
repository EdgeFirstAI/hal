// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Aliasing scenarios for `TensorDyn`'s raw-handle borrow shapes, run under
//! Miri with both aliasing models (`scripts/miri.sh` runs this file with
//! Stacked Borrows, then again with `MIRIFLAGS=-Zmiri-tree-borrows`).
//!
//! This crate is built with its plain default features (`static`,
//! `ndarray`), so `TensorDyn::into_raw`/`from_raw`/`with_raw` are the
//! `static` backend's own implementations (`tensor_dyn/static_backend.rs`)
//! -- `Box::into_raw`/`Box::from_raw`/`ManuallyDrop` over a boxed,
//! in-process `TensorDyn`, not the `dynamic` backend's `NonNull<EfTensor>`
//! over an opaque handle from `libedgefirst_tensor.so`. That distinction is
//! not a downgrade of what this file tests: both backends expose the
//! identical `into_raw`/`from_raw`/`with_raw` names and reborrow contract
//! precisely so a `-capi` leaf's calling convention does not change with
//! the backend (see `Raw`'s doc comment in both `static_backend.rs` and
//! `dynamic_backend.rs`), and the hazard this file guards against -- a
//! `with_raw` reborrow's guard dropped before the handle's window of use,
//! aliasing two `&mut` over the same allocation -- lives entirely in that
//! shared shape, not in how the bytes on the other side of the pointer
//! happen to be produced.
//!
//! Platform storage (`dma.rs` and friends) drops its `dma-heap` dependency
//! under Miri specifically (`Cargo.toml`'s `cfg(all(target_os = "linux",
//! not(miri)))` target table -- Miri cannot execute the real ioctls that
//! crate needs), but the `Tensor::new(..., None, ...)` auto-select path
//! falls back to a plain heap (`Mem`) tensor when `Dma` construction is
//! unavailable, so these tests run unmodified regardless: the raw-handle
//! borrow shapes below are pure Rust and never touch platform storage at
//! all.
use edgefirst_tensor::{Tensor, TensorDyn, TensorTrait};

#[test]
fn with_raw_reborrow_is_exclusive_for_its_whole_window() {
    let t: Tensor<u8> = Tensor::new(&[8], None, None).expect("alloc");
    let d: TensorDyn = t.into();
    let raw = TensorDyn::into_raw(d);
    // SAFETY: `raw` is live and unaliased for this call.
    unsafe {
        TensorDyn::with_raw(raw, |a| {
            let lens = a.as_typed_mut::<u8>().expect("u8");
            lens.map_write().expect("map")[0] = 1;
        });
        drop(TensorDyn::from_raw(raw));
    }
}

/// The unsound shape, kept FAILING on purpose under Stacked Borrows.
/// `#[ignore]`d, so `cargo test`/`miri.sh`'s two aliasing-model runs never
/// execute it on their own -- **`scripts/miri.sh` runs it explicitly, as a
/// third, separate check**, with `--ignored` naming it directly, under
/// Stacked Borrows only. It is not enough for that run to merely fail: a
/// Miri version bump, an OOM, a build error, or an unrelated bit of UB
/// would all satisfy a bare "did it fail", so `miri.sh` greps its output
/// for the specific retag-invalidation signature this test is documented
/// to produce (the phrase `trying to retag from`, plus provenance
/// references to all three of lines 79/80/81 below) before counting it as
/// confirmation the hazard still holds. If a future Miri ever accepts this
/// shape under Stacked Borrows, that check reports it as its own distinct
/// result -- not a code regression, but news that the model changed and
/// this diagnostic needs re-deriving -- rather than staying silent. See
/// `scripts/miri.sh` for that logic.
///
/// It must fail because Miri REJECTS THE BORROW SHAPE, not because the body
/// panics. A diagnostic with an `unimplemented!()` body asserts nothing and
/// would keep "passing its failure" long after the hazard was gone.
///
/// **The two models disagree on this one, and that disagreement is itself
/// established, not assumed:** under the default Stacked Borrows model this
/// test fails with a genuine retag-invalidation error (a `Unique` retag for
/// `b` invalidates `a`'s tag before `a.dtype()` reads through it). Under
/// `MIRIFLAGS=-Zmiri-tree-borrows` it currently PASSES — Tree Borrows is
/// more permissive about two raw-pointer-derived `&mut` that are each only
/// ever read through (neither `a` nor `b` is written before `.dtype()`
/// reads it), and does not flag this particular interleaving. That is a
/// real, informative gap between the two models for this exact shape, not
/// a bug in this test — see task-11's report for the aliasing-model
/// disagreement this surfaced. This test is a bonus diagnostic (Task 11's
/// brief), not part of what gates G7's two live aliasing-model runs -- but
/// `miri.sh`'s third check above does still gate on it holding.
#[test]
#[ignore = "diagnostic: fails under Stacked Borrows, passes under Tree Borrows -- a real model disagreement; run by scripts/miri.sh's third check, see task-11-report.md"]
fn unwrap_then_use_aliases_the_same_tensor() {
    let t: Tensor<u8> = Tensor::new(&[8], None, None).expect("alloc");
    let d: TensorDyn = t.into();
    let raw = TensorDyn::into_raw(d);
    // SAFETY: deliberately UNSOUND — this models the superseded
    // `unwrap_tensor` shape. Two `&mut` reach the same allocation because the
    // first borrow's guard is dropped before the handle is used.
    unsafe {
        let a = &mut *(raw as *mut TensorDyn); // borrow taken...
        let b = &mut *(raw as *mut TensorDyn); // ...and a second, aliasing it
        let _ = a.dtype();
        let _ = b.dtype();
        drop(TensorDyn::from_raw(raw));
    }
}
