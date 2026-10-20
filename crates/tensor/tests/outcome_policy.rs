// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The allocation policy every crate's tests share through `support/outcome.rs`.

#[path = "support/outcome.rs"]
mod outcome;

use outcome::{allocated, expect_dma, Expect};

#[test]
fn a_successful_allocation_is_returned_under_either_expectation() {
    assert_eq!(allocated(Ok::<_, ()>(7), "x", Expect::Succeeds), Some(7));
    assert_eq!(
        allocated(Ok::<_, ()>(7), "x", Expect::MayFail("why")),
        Some(7)
    );
}

#[test]
fn an_allowed_failure_skips() {
    assert_eq!(
        allocated(Err::<u8, _>("ENOMEM"), "x", Expect::MayFail("why")),
        None
    );
}

#[test]
#[should_panic(expected = "failed on a host expected to serve it")]
fn an_unexpected_failure_fails_the_test() {
    let _ = allocated(Err::<u8, _>("ENOMEM"), "x", Expect::Succeeds);
}

#[test]
fn dma_failures_are_expected_only_without_a_usable_heap() {
    assert_eq!(expect_dma(true), Expect::Succeeds);
    assert!(matches!(expect_dma(false), Expect::MayFail(_)));
}
