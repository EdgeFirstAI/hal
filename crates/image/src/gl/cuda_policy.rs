// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CUDA test policy: a skip must not be able to masquerade as coverage.
//!
//! The CUDA tests in `gl/tests.rs` used to return early whenever a
//! precondition -- a CUDA runtime, a GL context, F32 render support, a
//! PBO-backed destination, CUDA-GL interop -- was unmet, printing `SKIP` and
//! reporting `ok`. On a host whose runtime soname the probe did not
//! recognise (or that silently failed one of those other preconditions)
//! that is indistinguishable from real coverage -- which is exactly how
//! CUDA 13 hosts ran green while exercising nothing. `HAL_TEST_REQUIRE_CUDA=1`
//! turns every one of those skips into a failure, and `make test-cuda` sets
//! it whenever it located a runtime, so the lane cannot silently stop
//! testing again.
//!
//! [`decide`] is the pure core: given whether a precondition held and
//! whether coverage is required, it returns, prints, or panics -- no
//! environment access, so its own tests never touch
//! `HAL_TEST_REQUIRE_CUDA` and mean the same thing on every host.
//! [`cuda_available_or_skip`] and [`require_or_skip`] are the two thin,
//! env-reading callers `gl/tests.rs` uses.
//!
//! Same philosophy as `tests/gpu_policy.py` for GPU-backed tests: on a
//! platform where the feature must work, a vacuous skip is a product bug.

/// `true` when `HAL_TEST_REQUIRE_CUDA=1` is set -- the opt-in that turns a
/// quiet skip into a visible failure.
fn required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_CUDA").as_deref() == Ok("1")
}

/// The pure decision: `true` when `satisfied`; otherwise a failure or a skip
/// depending on `require`. No environment access -- callers read
/// `HAL_TEST_REQUIRE_CUDA` and pass the result in as `require`, which is what
/// keeps this function (and its tests) meaningful on every host.
///
/// # Panics
///
/// When `require` is `true` and `satisfied` is `false` -- the opt-in that
/// turns "quietly skipped" into a visible failure. The panic message names
/// `HAL_TEST_REQUIRE_CUDA=1`, `what`, and `why`.
fn decide(satisfied: bool, require: bool, what: &str, why: &str) -> bool {
    if satisfied {
        return true;
    }
    assert!(
        !require,
        "HAL_TEST_REQUIRE_CUDA=1 but {what} would have skipped while \
         reporting success: {why}"
    );
    eprintln!("SKIP {what}: {why}");
    false
}

/// `true` when the caller should run; `false` when it should return early.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and CUDA is unavailable -- see [`decide`].
pub(crate) fn cuda_available_or_skip(what: &str) -> bool {
    decide(
        edgefirst_tensor::is_cuda_available(),
        required(),
        what,
        "no libcudart (a supported soname missing from cuda.rs's probe \
         list, or LD_LIBRARY_PATH does not reach the runtime)",
    )
}

/// `true` when the caller should run; `false` when it should return early.
/// For preconditions inside a CUDA test other than the runtime itself --
/// a GL context, F32 render support, a PBO-backed destination, CUDA-GL
/// interop -- so that under `HAL_TEST_REQUIRE_CUDA=1` a skip anywhere in a
/// required CUDA test is a failure, not just an absent runtime.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and `satisfied` is `false` -- see
/// [`decide`].
pub(crate) fn require_or_skip(satisfied: bool, what: &str, why: &str) -> bool {
    decide(satisfied, required(), what, why)
}

#[cfg(test)]
mod tests {
    use super::decide;

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

    /// Unsatisfied and not required: an ordinary skip -- CI lanes without
    /// the precondition must stay green.
    #[test]
    fn unsatisfied_and_not_required_is_an_ordinary_skip() {
        assert!(!decide(false, false, "what", "why"));
    }

    /// Unsatisfied and required: a failure, not a skip. The panic message
    /// must name the opt-in plus both the caller (`what`) and the reason
    /// (`why`), so a CI failure is self-explanatory without extra digging.
    /// This never touches `HAL_TEST_REQUIRE_CUDA` itself -- `decide` takes
    /// `require` as a plain bool -- so the test means the same thing on
    /// every host and needs no env save/restore.
    #[test]
    fn unsatisfied_and_required_panics_naming_the_opt_in_what_and_why() {
        // The panic itself is the point of this test. Silence the default
        // hook's "thread '...' panicked at ..." stderr line while we
        // provoke it, so this test's own output -- and `cargo test`'s
        // output for the lane as a whole -- stays pristine; restore the
        // previous hook immediately after so it never leaks into any other
        // test.
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
}
