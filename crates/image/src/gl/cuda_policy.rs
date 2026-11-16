// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CUDA test policy: whether an unmet precondition skips a test or fails it.
//!
//! The CUDA tests in `gl/tests.rs` skip when a precondition -- a CUDA
//! runtime, a GL context, F32 render support, a PBO-backed destination,
//! CUDA-GL interop -- is unmet. `HAL_TEST_REQUIRE_CUDA=1` turns every one of
//! those skips into a failure.
//!
//! [`decide`] takes whether a precondition held and whether coverage is
//! required, and returns or panics. [`cuda_available_or_skip`] and
//! [`require_or_skip`] read the environment, call it, and print
//! `SKIP {what}: {why}` when it returns `false`.

/// `true` when `HAL_TEST_REQUIRE_CUDA=1` is set.
fn required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_CUDA").as_deref() == Ok("1")
}

/// `true` when `satisfied`; otherwise a failure or a skip depending on
/// `require`. Reads no environment and prints nothing.
///
/// # Panics
///
/// When `require` is `true` and `satisfied` is `false`. The panic message
/// names `HAL_TEST_REQUIRE_CUDA=1`, `what`, and `why`.
fn decide(satisfied: bool, require: bool, what: &str, why: &str) -> bool {
    if satisfied {
        return true;
    }
    assert!(
        !require,
        "HAL_TEST_REQUIRE_CUDA=1 but {what} would have skipped while \
         reporting success: {why}"
    );
    false
}

/// `true` when the caller should run; `false` when it should return early.
/// Prints `SKIP {what}: {why}` to stderr on a `false` return.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and CUDA is unavailable -- see [`decide`].
pub(crate) fn cuda_available_or_skip(what: &str) -> bool {
    let why = "no libcudart (a supported soname missing from cuda.rs's probe \
         list, or LD_LIBRARY_PATH does not reach the runtime)";
    let run = decide(edgefirst_tensor::is_cuda_available(), required(), what, why);
    if !run {
        crate::test_support::report_skip(&format!("{what}: {why}"));
    }
    run
}

/// `true` when the caller should run; `false` when it should return early.
/// For preconditions inside a CUDA test other than the runtime itself: a GL
/// context, F32 render support, a PBO-backed destination, CUDA-GL interop.
/// Prints `SKIP {what}: {why}` to stderr on a `false` return.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and `satisfied` is `false` -- see
/// [`decide`].
pub(crate) fn require_or_skip(satisfied: bool, what: &str, why: &str) -> bool {
    let run = decide(satisfied, required(), what, why);
    if !run {
        crate::test_support::report_skip(&format!("{what}: {why}"));
    }
    run
}

#[cfg(test)]
mod tests {
    use super::decide;

    /// Satisfied and required: runs.
    #[test]
    fn satisfied_and_required_runs() {
        assert!(decide(true, true, "what", "why"));
    }

    /// Satisfied and not required: runs.
    #[test]
    fn satisfied_and_not_required_runs() {
        assert!(decide(true, false, "what", "why"));
    }

    /// Unsatisfied and not required: an ordinary skip.
    #[test]
    fn unsatisfied_and_not_required_is_an_ordinary_skip() {
        assert!(!decide(false, false, "what", "why"));
    }

    /// Unsatisfied and required: panics, naming the opt-in, `what` and `why`.
    #[test]
    fn unsatisfied_and_required_panics_naming_the_opt_in_what_and_why() {
        // Silence the default panic hook while provoking the panic, then
        // restore it.
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
