// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Devices render into and read from uncached DMA-BUF images.
//!
//! A `CpuAccess::None` or `CpuAccess::Write` DMA-BUF image comes from the
//! uncached CMA heap when it can serve the allocation. Each test drives a
//! device through such an image and checks the pixels through a second device
//! pass into a CPU-readable image, against the same work done in cached
//! memory.
//!
//! The tests run only where an allocation is actually served uncached, which a
//! probe allocation's exporter name establishes; elsewhere they skip loudly,
//! and the cached fallback is the tensor crate's to test. Missing backends
//! skip as well, except under `HAL_TEST_REQUIRE_DMA=1`, `HAL_TEST_REQUIRE_GL=1`
//! and `HAL_TEST_REQUIRE_G2D=1`, which turn the matching skip into a failure.

#![cfg(target_os = "linux")]

use edgefirst_image::{
    ComputeBackend, G2DProcessor, GLProcessorThreaded, ImageProcessor, ImageProcessorConfig,
    ImageProcessorTrait,
};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
use outcome::report_skip as skip;

const W: usize = 320;
const H: usize = 240;

fn required(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| v == "1")
}

/// The heap a DMA-BUF image came from, as `/proc/self/fdinfo` names it.
fn exporter(t: &TensorDyn) -> Option<String> {
    use std::os::fd::AsRawFd;
    let fd = t.clone_fd().ok()?;
    let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.as_raw_fd())).ok()?;
    info.lines()
        .find_map(|l| l.strip_prefix("exp_name:"))
        .map(|n| n.trim().to_owned())
}

fn is_uncached(t: &TensorDyn) -> bool {
    exporter(t).is_some_and(|n| n.ends_with("cma-uncached"))
}

fn dma_image(access: CpuAccess) -> TensorDyn {
    TensorDyn::image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        access,
    )
    .expect("DMA-BUF image on a host with a usable DMA heap")
}

/// Whether the test named `what` runs: a usable DMA heap (under
/// `HAL_TEST_REQUIRE_DMA`) whose uncached CMA heap actually serves a
/// write-only image.
fn uncached_or_skip(what: &str) -> bool {
    if !dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), what) {
        return false;
    }
    let probe = dma_image(CpuAccess::Write);
    if is_uncached(&probe) {
        return true;
    }
    skip(&format!(
        "{what} - write-only images are not served uncached here (exporter {:?})",
        exporter(&probe)
    ));
    false
}

fn gl_image(gl: &ImageProcessor, access: CpuAccess) -> TensorDyn {
    gl.create_image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        access,
    )
    .expect("DMA-BUF image")
}

fn fill(t: &TensorDyn) {
    let mut m = t.as_u8().unwrap().map_for_write().unwrap();
    for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
        *b = (i * 13 % 251) as u8;
    }
}

fn read(t: &TensorDyn) -> Vec<u8> {
    t.as_u8().unwrap().map_read().unwrap().as_slice().to_vec()
}

/// GL renders into a hardware-only image, as a compositor-bound canvas, and
/// samples it and a CPU-written image back into readable memory unchanged.
#[test]
fn gl_renders_into_and_samples_uncached_images() {
    const NAME: &str = "gl_renders_into_and_samples_uncached_images";
    if !uncached_or_skip(NAME) {
        return;
    }
    if let Err(e) = GLProcessorThreaded::new(None) {
        assert!(
            !required("HAL_TEST_REQUIRE_GL"),
            "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
        );
        return skip(&format!("{NAME} - no GL backend: {e}"));
    }
    let mut gl = ImageProcessor::with_config(ImageProcessorConfig {
        backend: ComputeBackend::OpenGl,
        ..Default::default()
    })
    .expect("GL processor");
    let mut canvas = gl_image(&gl, CpuAccess::None);
    // A GL backend that cannot render a hardware-only image widens the
    // declaration so its CPU fallback can; that is a capability, not a bug.
    if canvas.cpu_access() != CpuAccess::None {
        return skip(&format!(
            "{NAME} - this GL backend cannot render a hardware-only image"
        ));
    }
    assert!(
        is_uncached(&canvas),
        "canvas exporter {:?}",
        exporter(&canvas)
    );

    let src = gl_image(&gl, CpuAccess::ReadWrite);
    fill(&src);
    let mut want = gl_image(&gl, CpuAccess::ReadWrite);
    gl.convert(&src, &mut want, None).unwrap();

    gl.convert(&src, &mut canvas, None).unwrap();
    let mut got = gl_image(&gl, CpuAccess::ReadWrite);
    gl.convert(&canvas, &mut got, None).unwrap();
    assert!(
        read(&got) == read(&want),
        "GL round trip through a hardware-only image"
    );

    let overlay = gl_image(&gl, CpuAccess::Write);
    assert!(
        is_uncached(&overlay),
        "overlay exporter {:?}",
        exporter(&overlay)
    );
    fill(&overlay);
    let mut got = gl_image(&gl, CpuAccess::ReadWrite);
    gl.convert(&overlay, &mut got, None).unwrap();
    assert!(
        read(&got) == read(&src),
        "GL sampling a CPU-written uncached image"
    );
}

/// G2D blits from a CPU-written uncached image into a hardware-only one and
/// back out unchanged.
#[test]
fn g2d_blits_between_uncached_images() {
    const NAME: &str = "g2d_blits_between_uncached_images";
    if !uncached_or_skip(NAME) {
        return;
    }
    let mut g2d = match G2DProcessor::new() {
        Ok(g2d) => g2d,
        Err(e) => {
            assert!(
                !required("HAL_TEST_REQUIRE_G2D"),
                "HAL_TEST_REQUIRE_G2D=1 but G2D failed to initialize: {e}"
            );
            return skip(&format!("{NAME} - no G2D: {e}"));
        }
    };

    let src = dma_image(CpuAccess::Write);
    assert!(is_uncached(&src), "source exporter {:?}", exporter(&src));
    fill(&src);
    let mut canvas = dma_image(CpuAccess::None);
    assert!(
        is_uncached(&canvas),
        "canvas exporter {:?}",
        exporter(&canvas)
    );
    g2d.convert(&src, &mut canvas, None)
        .expect("G2D blit into an uncached hardware-only image");
    let mut got = dma_image(CpuAccess::ReadWrite);
    g2d.convert(&canvas, &mut got, None)
        .expect("G2D blit from an uncached hardware-only image");
    let mut want = dma_image(CpuAccess::ReadWrite);
    g2d.convert(&src, &mut want, None)
        .expect("G2D blit from an uncached image");
    assert!(
        read(&got) == read(&want),
        "G2D round trip through uncached images"
    );
}
