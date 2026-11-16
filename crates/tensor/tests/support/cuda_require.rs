// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CUDA test policy for the D3D11 interop tests.
//!
//! When `edgefirst_tensor::is_cuda_available()` is false the CUDA tests in
//! `d3d11_tensor.rs` skip. `HAL_TEST_REQUIRE_CUDA=1` turns that skip into a
//! failure naming the test and the reason.
//!
//! [`decide`] is the pure core; [`cuda_available_or_skip`] reads the
//! environment and reports the skip.

/// `true` when `HAL_TEST_REQUIRE_CUDA=1` is set.
fn required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_CUDA").as_deref() == Ok("1")
}

/// The pure decision: `true` when `satisfied`; otherwise a failure or a skip
/// depending on `require`. No environment access and no output: the caller
/// reports the skip on a `false` return.
///
/// # Panics
///
/// When `require` is `true` and `satisfied` is `false`. The panic message
/// names `HAL_TEST_REQUIRE_CUDA=1`, `what`, and `why`.
pub fn decide(satisfied: bool, require: bool, what: &str, why: &str) -> bool {
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
/// Writes `SKIPPED: {what} - {why}` straight to stderr on a `false` return.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and CUDA is unavailable; see [`decide`].
pub fn cuda_available_or_skip(what: &str) -> bool {
    use std::io::Write;
    let why = "no CUDA runtime (a supported soname missing from cuda.rs's \
         probe list -- cudart64_13.dll, cudart64_12.dll, cudart64_110.dll -- \
         or it is not on PATH or CUDA_PATH\\bin)";
    let run = decide(edgefirst_tensor::is_cuda_available(), required(), what, why);
    if !run {
        let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {what} - {why}");
    }
    run
}
