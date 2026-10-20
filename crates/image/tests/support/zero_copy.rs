// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The platform zero-copy buffer (`TensorMemory::DmaBuf`: a DMA-BUF on Linux,
//! an IOSurface on Apple, a D3D11 texture on Windows) under the allocation
//! policy: it must allocate where the host serves it, and may fail with a
//! warning elsewhere.
//!
//! `#[path]`-include it next to `outcome` and `dma_require`, declared under
//! those names at the test crate root.

use super::{dma_require, outcome};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

/// Whether a missing zero-copy buffer fails the test even without a probe:
/// under `HAL_TEST_REQUIRE_GL=1` on macOS and Windows, where the buffer needs
/// no device node, and on Linux under `HAL_TEST_REQUIRE_GL=1` on a host that
/// has a DMA heap node.
fn required_with_gl() -> bool {
    let set = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
    if cfg!(target_os = "linux") {
        set("HAL_TEST_REQUIRE_GL") && std::path::Path::new("/dev/dma_heap").exists()
    } else {
        set("HAL_TEST_REQUIRE_GL")
    }
}

/// Whether a zero-copy allocation may fail on this host.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_DMA=1` and the host serves no zero-copy buffer.
pub fn expect() -> outcome::Expect {
    let served = dma_require::decide(
        edgefirst_tensor::is_gpu_buffer_available(),
        dma_require::required(),
        &outcome::current_test(),
        "no platform zero-copy buffer (no usable DMA heap, IOSurface or D3D11 device)",
    );
    if served || required_with_gl() {
        outcome::Expect::Succeeds
    } else {
        outcome::Expect::MayFail("no platform zero-copy buffer on this host")
    }
}

/// `true` when this host must serve the zero-copy buffer; otherwise reports a
/// skip and returns `false`.
pub fn served_or_skip() -> bool {
    match expect() {
        outcome::Expect::Succeeds => true,
        outcome::Expect::MayFail(why) => {
            outcome::report_skip(&format!("{}: {why}", outcome::current_test()));
            false
        }
    }
}

/// A failed allocation's reason, printed without the quotes `String`'s `Debug` adds.
pub struct Failure(String);

impl std::fmt::Debug for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Treats a zero-copy request that landed on other memory as a failed allocation.
pub fn zero_copy_only(
    result: Result<TensorDyn, edgefirst_tensor::Error>,
) -> Result<TensorDyn, Failure> {
    match result {
        Ok(t) if t.memory() == TensorMemory::DmaBuf => Ok(t),
        Ok(t) => Err(Failure(format!(
            "zero-copy request fell back to {:?}",
            t.memory()
        ))),
        Err(e) => Err(Failure(format!("{e:?}"))),
    }
}

/// A `w`x`h` image on the platform zero-copy buffer, under [`expect`]: `None`,
/// after a warning, only where the host may not serve one.
pub fn image(w: usize, h: usize, fmt: PixelFormat, dtype: DType) -> Option<TensorDyn> {
    outcome::allocated(
        zero_copy_only(TensorDyn::image(
            w,
            h,
            fmt,
            dtype,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )),
        &format!("{w}x{h} {fmt:?}/{dtype:?} zero-copy image"),
        expect(),
    )
}
