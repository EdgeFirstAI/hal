// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Coverage of `decide`, the decision core in
//! `crates/tensor/tests/support/cuda_require.rs` that gates the D3D11 CUDA
//! interop tests under `HAL_TEST_REQUIRE_CUDA=1`.

/// The Windows guard's own module, included whole.
#[path = "support/cuda_require.rs"]
#[allow(dead_code)]
mod cuda_require;

use cuda_require::decide;

/// Satisfied and required: runs, no assertion, no output.
#[test]
fn satisfied_and_required_runs() {
    assert!(decide(true, true, "what", "why"));
}

/// Satisfied and not required: runs regardless of the opt-in.
#[test]
fn satisfied_and_not_required_runs() {
    assert!(decide(true, false, "what", "why"));
}

/// Unsatisfied and not required: an ordinary skip.
#[test]
fn unsatisfied_and_not_required_is_an_ordinary_skip() {
    assert!(!decide(false, false, "what", "why"));
}

/// Unsatisfied and required: a panic naming the opt-in, `what` and `why`.
#[test]
fn unsatisfied_and_required_panics_naming_the_opt_in_what_and_why() {
    // Silences the default panic hook while the panic is provoked, then restores it.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(|| decide(false, true, "probe", "some reason"));
    std::panic::set_hook(prev_hook);

    let payload = caught.expect_err("a required-but-absent precondition must panic, not skip");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("panic payload should be a string");
    assert!(
        message.contains("HAL_TEST_REQUIRE_CUDA=1"),
        "panic message must name the opt-in: {message}"
    );
    assert!(
        message.contains("probe"),
        "panic message must name the caller: {message}"
    );
    assert!(
        message.contains("some reason"),
        "panic message must name the reason: {message}"
    );
}
