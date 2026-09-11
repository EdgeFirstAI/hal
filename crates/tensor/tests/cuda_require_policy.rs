// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform-generic coverage of `decide`, the pure decision core in
//! `crates/tensor/tests/support/cuda_require.rs` that gates the three D3D11
//! CUDA interop tests in `crates/tensor/tests/d3d11_tensor.rs` (Windows-only)
//! under `HAL_TEST_REQUIRE_CUDA=1`.
//!
//! The support module is `#[path]`-included here rather than copied, so these
//! tests pin the function the Windows guard actually calls: the policy cannot
//! regress in `d3d11_tensor.rs` while this file stays green. `decide` itself
//! has no OS dependency -- it is a plain function of two bools and two
//! strings -- so this file is not Windows-gated, and every lane (this Linux
//! box included, where `d3d11_tensor.rs` compiles to nothing) exercises it.

/// The Windows guard's own module, included whole so the tests below run
/// against the shipped `decide` and not a copy of it. `cuda_available_or_skip`
/// has no caller here -- it reads the environment and probes for a CUDA
/// runtime, which is what makes it unsuitable as a unit under test -- hence
/// the `dead_code` allowance, which is scoped to this include and leaves
/// `d3d11_tensor.rs`'s copy of the module linted as usual.
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

/// Unsatisfied and not required: an ordinary skip -- lanes without CUDA
/// (this box included) must stay green.
#[test]
fn unsatisfied_and_not_required_is_an_ordinary_skip() {
    assert!(!decide(false, false, "what", "why"));
}

/// Unsatisfied and required: a failure, not a skip. The panic message must
/// name the opt-in plus both the caller (`what`) and the reason (`why`), so
/// a CI failure is self-explanatory without extra digging. This never
/// touches `HAL_TEST_REQUIRE_CUDA` itself -- `decide` takes `require` as a
/// plain bool -- so the test means the same thing on every host and needs
/// no env save/restore.
#[test]
fn unsatisfied_and_required_panics_naming_the_opt_in_what_and_why() {
    // The panic itself is the point of this test. Silence the default
    // hook's "thread '...' panicked at ..." stderr line while we provoke
    // it, so this test's own output -- and the lane's output as a whole --
    // stays pristine; restore the previous hook immediately after so it
    // never leaks into any other test.
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
