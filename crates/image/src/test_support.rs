// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Test-only helpers: skip reporting and the allocation policy live in the
//! shared `outcome` file; this module adds the DMA and G2D probes.

#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[allow(unused_imports)]
pub(crate) use outcome::{allocated, current_test, report_skip, Expect};

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;

/// [`edgefirst_tensor::is_dma_available`] under the `HAL_TEST_REQUIRE_DMA=1`
/// policy: on a lane that requires a DMA heap, an unavailable one fails the
/// calling test instead of letting it skip while reporting success.
pub(crate) fn is_dma_available() -> bool {
    dma_require::decide(
        edgefirst_tensor::is_dma_available(),
        dma_require::required(),
        &current_test(),
        "no usable DMA heap (no /dev/dma_heap node, or it is not openable by this user)",
    )
}

/// [`edgefirst_tensor::is_gpu_buffer_available`], under the `HAL_TEST_REQUIRE_DMA=1` policy on Linux
/// where the buffers are DMA-BUFs.
#[cfg_attr(not(feature = "opengl"), allow(dead_code))]
pub(crate) fn is_gpu_buffer_available() -> bool {
    #[cfg(target_os = "linux")]
    return is_dma_available();
    #[cfg(not(target_os = "linux"))]
    edgefirst_tensor::is_gpu_buffer_available()
}

/// Unwraps a DMA-BUF allocation: it must succeed when the heap probe (under
/// `HAL_TEST_REQUIRE_DMA`) says the heap is usable, and may fail with a warning otherwise.
pub(crate) fn dma_allocated<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => allocated(Err(e), what, outcome::expect_dma(is_dma_available())),
    }
}

/// Unwraps a zero-copy GPU buffer allocation (DMA-BUF, IOSurface or D3D11): it must succeed when
/// [`is_gpu_buffer_available`] reports the backing, and may fail with a warning otherwise.
#[cfg_attr(not(feature = "opengl"), allow(dead_code))]
pub(crate) fn gpu_buffer_allocated<T, E: std::fmt::Debug>(
    result: Result<T, E>,
    what: &str,
) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            let expect = if is_gpu_buffer_available() {
                Expect::Succeeds
            } else {
                Expect::MayFail("no zero-copy GPU buffer backing on this host")
            };
            allocated(Err(e), what, expect)
        }
    }
}

/// Applies the `HAL_TEST_REQUIRE_G2D=1` policy to a G2D availability probe.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_G2D=1` and `available` is `false`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn g2d_required_or(available: bool) -> bool {
    let required = std::env::var("HAL_TEST_REQUIRE_G2D").as_deref() == Ok("1");
    g2d_decide(available, required, &current_test())
}

fn g2d_decide(available: bool, required: bool, what: &str) -> bool {
    assert!(
        available || !required,
        "HAL_TEST_REQUIRE_G2D=1 but {what} would have skipped while reporting success: \
         libg2d.so.2 did not load or G2D failed to initialize"
    );
    available
}

#[test]
fn g2d_policy_skips_unless_required() {
    assert!(g2d_decide(true, true, "t"));
    assert!(g2d_decide(true, false, "t"));
    assert!(!g2d_decide(false, false, "t"));
    let required_but_missing = std::panic::catch_unwind(|| g2d_decide(false, true, "t"));
    assert!(required_but_missing.is_err());
}
