// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Devices render into and read from uncached DMA-BUF images.
//!
//! A `CpuAccess::None` or `CpuAccess::Write` DMA-BUF image comes from the
//! uncached CMA heap where the system has one. Each test drives a device
//! through such an image and checks the pixels through a second device pass
//! into a CPU-readable image, against the same work done in cached memory.
//! Skips loudly where the backend or the DMA heap is missing.

#![cfg(target_os = "linux")]

use edgefirst_image::{ComputeBackend, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const W: usize = 320;
const H: usize = 240;

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "WARNING: SKIPPED: {why}");
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

fn uncached_heap_present() -> bool {
    std::path::Path::new("/dev/dma_heap/linux,cma-uncached").exists()
}

fn processor(backend: ComputeBackend) -> Option<ImageProcessor> {
    ImageProcessor::with_config(ImageProcessorConfig {
        backend,
        ..Default::default()
    })
    .ok()
}

fn image(
    proc: &ImageProcessor,
    memory: Option<TensorMemory>,
    access: CpuAccess,
) -> Option<TensorDyn> {
    proc.create_image(W, H, PixelFormat::Rgba, DType::U8, memory, access)
        .ok()
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

/// An image declared `access` is a DMA-BUF from the uncached heap.
fn assert_uncached(t: &TensorDyn, access: CpuAccess) {
    assert_eq!(t.cpu_access(), access);
    assert_eq!(t.memory(), TensorMemory::DmaBuf, "{access:?}");
    if let Some(name) = exporter(t) {
        assert!(
            name.ends_with("cma-uncached"),
            "{access:?}: exporter {name}"
        );
    }
}

/// GL renders into a hardware-only image, as a compositor-bound canvas, and
/// samples it and a CPU-written image back into readable memory unchanged.
#[test]
fn gl_renders_into_and_samples_uncached_images() {
    if !uncached_heap_present() {
        return skip("no uncached CMA heap");
    }
    let Some(mut gl) = processor(ComputeBackend::OpenGl) else {
        return skip("no OpenGL backend");
    };
    let Some(canvas) = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::None) else {
        return skip("no DMA-BUF images");
    };
    if canvas.cpu_access() != CpuAccess::None {
        return skip("the GL backend cannot render a hardware-only image here");
    }
    assert_uncached(&canvas, CpuAccess::None);
    let mut canvas = canvas;

    let src = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::ReadWrite).unwrap();
    fill(&src);
    let mut want = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::ReadWrite).unwrap();
    gl.convert(&src, &mut want, None).unwrap();

    gl.convert(&src, &mut canvas, None).unwrap();
    let mut got = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::ReadWrite).unwrap();
    gl.convert(&canvas, &mut got, None).unwrap();
    assert!(
        read(&got) == read(&want),
        "GL round trip through a hardware-only image"
    );

    let overlay = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::Write).unwrap();
    assert_uncached(&overlay, CpuAccess::Write);
    fill(&overlay);
    let mut got = image(&gl, Some(TensorMemory::DmaBuf), CpuAccess::ReadWrite).unwrap();
    gl.convert(&overlay, &mut got, None).unwrap();
    assert!(
        read(&got) == read(&src),
        "GL sampling a CPU-written uncached image"
    );
}

/// G2D blits from a CPU-written uncached image into a hardware-only one.
#[test]
fn g2d_blits_between_uncached_images() {
    if !uncached_heap_present() {
        return skip("no uncached CMA heap");
    }
    let Some(mut g2d) = processor(ComputeBackend::G2d) else {
        return skip("no G2D backend");
    };
    let alloc = |access| {
        TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            access,
        )
    };
    let Ok(src) = alloc(CpuAccess::Write) else {
        return skip("no DMA-BUF images");
    };
    assert_uncached(&src, CpuAccess::Write);
    fill(&src);
    let mut canvas = alloc(CpuAccess::None).unwrap();
    assert_uncached(&canvas, CpuAccess::None);
    if let Err(e) = g2d.convert(&src, &mut canvas, None) {
        return skip(&format!("G2D declined the blit: {e}"));
    }
    let mut got = alloc(CpuAccess::ReadWrite).unwrap();
    g2d.convert(&canvas, &mut got, None).unwrap();
    let mut want = alloc(CpuAccess::ReadWrite).unwrap();
    g2d.convert(&src, &mut want, None).unwrap();
    assert!(
        read(&got) == read(&want),
        "G2D round trip through uncached images"
    );
}
