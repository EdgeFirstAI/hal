// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CUDA test policy for the D3D11 interop tests: a skip must not be able to
//! masquerade as coverage.
//!
//! `d3d11_tensor.rs`'s CUDA tests used to return early whenever
//! `edgefirst_tensor::is_cuda_available()` was false, printing
//! `SKIP: no CUDA runtime` and reporting `ok`. On an NVIDIA Windows box
//! whose runtime soname `cuda.rs`'s probe does not recognise (or that is
//! simply not on `PATH`/`CUDA_PATH\bin`), that is indistinguishable from
//! real coverage -- the same vacuity PR #168 closed on Linux for
//! `crates/image/src/gl/cuda_policy.rs`. `HAL_TEST_REQUIRE_CUDA=1` turns
//! that skip into a failure naming the test and the reason, and
//! `scripts/test-windows.ps1 -RequireCuda` sets it whenever an NVIDIA
//! adapter is present, so the lane cannot silently stop testing again.
//!
//! The WARP-adapter skip (`cudaD3D11GetDevice` has no ordinal for WARP) is
//! correct by design and does not go through this module. Its callers test
//! for it *before* calling in here, so an armed gate cannot turn that skip
//! into a failure -- see `d3d11_tensor.rs`'s `warp_has_no_cuda_device`.
//!
//! [`decide`] is the pure core, and it is `pub` so this module can be
//! `#[path]`-included by `crates/tensor/tests/cuda_require_policy.rs` as
//! well as by `d3d11_tensor.rs`. Integration tests are separate compilation
//! units, so that include is the sharing mechanism: the policy tests cover
//! the very function the Windows guard calls, on every lane -- the Linux
//! ones included, where `d3d11_tensor.rs` compiles to nothing -- rather
//! than a copy that could drift from it.
//! [`cuda_available_or_skip`] is the one env-reading, printing caller
//! `d3d11_tensor.rs` uses.

/// `true` when `HAL_TEST_REQUIRE_CUDA=1` is set -- the opt-in that turns a
/// quiet skip into a visible failure.
fn required() -> bool {
    std::env::var("HAL_TEST_REQUIRE_CUDA").as_deref() == Ok("1")
}

/// The pure decision: `true` when `satisfied`; otherwise a failure or a skip
/// depending on `require`. No environment access and no output -- the
/// caller reads `HAL_TEST_REQUIRE_CUDA` and passes the result in as
/// `require`, and prints the `SKIP` line itself on a `false` return. That
/// purity is what lets `crates/tensor/tests/cuda_require_policy.rs` test
/// this exact function on hosts with no CUDA and no D3D11.
///
/// # Panics
///
/// When `require` is `true` and `satisfied` is `false` -- the opt-in that
/// turns "quietly skipped" into a visible failure. The panic message names
/// `HAL_TEST_REQUIRE_CUDA=1`, `what`, and `why`.
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
/// Prints `SKIP {what}: {why}` to stderr on a `false` return.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_CUDA=1` and CUDA is unavailable -- see [`decide`].
pub fn cuda_available_or_skip(what: &str) -> bool {
    let why = "no CUDA runtime (a supported soname missing from cuda.rs's \
         probe list -- cudart64_13.dll, cudart64_12.dll, cudart64_110.dll -- \
         or it is not on PATH or CUDA_PATH\\bin)";
    let run = decide(edgefirst_tensor::is_cuda_available(), required(), what, why);
    if !run {
        eprintln!("SKIP {what}: {why}");
    }
    run
}
