// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A host pin stays valid across `convert()` into the pinned buffer.
//!
//! Asserted per convert:
//!
//! 1. The pin's address does not move.
//! 2. The pin and a fresh map guard agree byte-for-byte.
//! 3. The pixels are the ones the convert was asked for.
//!
//! The destination is a `TensorMemory::DmaBuf` image (DMA-BUF on Linux,
//! IOSurface on macOS); the file skips where none can be allocated. The
//! CPU-fallback count is reported in a note.

use edgefirst_image::{Flip, ImageProcessor, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, HostPin, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const W: usize = 64;
const H: usize = 48;

/// TFLite's `kDefaultTensorAlignment`.
const TFLITE_ALIGNMENT: usize = 64;

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[path = "support/zero_copy.rs"]
#[allow(dead_code)]
mod zero_copy;

/// Write a note to stderr, bypassing libtest's output capture.
fn note(what: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "NOTE: {what}");
}

/// The analytic source ramp: a distinct value per pixel per channel.
fn want(x: usize, y: usize) -> [u8; 3] {
    [x as u8, y as u8, ((x + y) / 2) as u8]
}

fn image(fmt: PixelFormat) -> TensorDyn {
    TensorDyn::image(
        W,
        H,
        fmt,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("allocate a zero-copy image")
}

/// An RGBA source carrying [`want`] with an opaque alpha.
fn ramp_src() -> TensorDyn {
    let mut src = image(PixelFormat::Rgba);
    let stride = src.effective_row_stride().expect("src row stride");
    {
        let u8t = src.as_u8_mut().expect("u8 source");
        let mut map = u8t.map_mut().expect("map source");
        let buf = map.as_mut_slice();
        for y in 0..H {
            for x in 0..W {
                let p = y * stride + x * 4;
                let w = want(x, y);
                buf[p..p + 3].copy_from_slice(&w);
                buf[p + 3] = 0xFF;
            }
        }
    }
    src
}

/// Read the pinned window, bracketed with `sync_for_cpu` and `sync_for_device`.
fn read_through_pin(t: &TensorDyn, pin: &HostPin<'static>) -> Vec<u8> {
    t.sync_for_cpu(CpuAccess::Read).expect("sync_for_cpu");
    // SAFETY: the convert has completed and the read is bracketed above.
    let bytes = unsafe { pin.as_slice() }.to_vec();
    t.sync_for_device(CpuAccess::Read).expect("sync_for_device");
    bytes
}

/// Read the same tensor through a map guard, which exposes the padded extent.
fn read_through_map(t: &TensorDyn) -> Vec<u8> {
    let u8t = t.as_u8().expect("u8 tensor");
    let map = u8t.map_read().expect("map for read");
    map.as_slice().to_vec()
}

/// Assert every RGB pixel reachable within the pinned window equals
/// `expect(x, y)`. Returns the number of rows checked; rows past the end of
/// the window are skipped.
fn assert_pixels(
    label: &str,
    bytes: &[u8],
    stride: usize,
    expect: impl Fn(usize, usize) -> [u8; 3],
) -> usize {
    let row_bytes = W * 3;
    let mut rows = 0;
    for y in 0..H {
        let start = y * stride;
        if start + row_bytes > bytes.len() {
            break;
        }
        for x in 0..W {
            let p = start + x * 3;
            let got = [bytes[p], bytes[p + 1], bytes[p + 2]];
            let w = expect(x, y);
            assert_eq!(
                got, w,
                "{label}: pixel ({x},{y}) is {got:?}, expected {w:?} -- \
                 the pin is not looking at what convert wrote"
            );
        }
        rows += 1;
    }
    rows
}

#[test]
fn pin_survives_convert_into_the_pinned_destination() {
    if !zero_copy::served_or_skip() {
        return;
    }

    let src = ramp_src();
    let mut dst = image(PixelFormat::Rgb);
    let stride = dst.effective_row_stride().expect("dst row stride");

    // Pinned before the first convert and held across both.
    let pin = dst.pin_host(CpuAccess::ReadWrite).expect("pin destination");
    let addr = pin.as_ptr();
    assert!(!addr.is_null(), "pin handed back a null address");
    assert_eq!(
        pin.len(),
        W * H * 3,
        "pin must cover the logical extent, not the padded capacity"
    );
    assert!(
        pin.alignment() >= TFLITE_ALIGNMENT,
        "convert destination is {}-byte aligned; TFLite's custom allocation \
         wants >= {TFLITE_ALIGNMENT} and can crash in Invoke() below it",
        pin.alignment()
    );

    let mut processor = ImageProcessor::new().expect("create ImageProcessor");

    // ── Frame 1 ──────────────────────────────────────────────────────────
    processor
        .convert(&src, &mut dst, None)
        .expect("convert frame 1");

    assert_eq!(
        pin.as_ptr(),
        addr,
        "convert moved the destination's host address -- a runtime holding the \
         pin would now be reading freed or unrelated memory"
    );

    let from_pin = read_through_pin(&dst, &pin);
    let from_map = read_through_map(&dst);
    assert_eq!(
        from_pin,
        &from_map[..from_pin.len()],
        "the pin and a fresh map guard disagree -- the pin is not aliasing the \
         buffer convert rendered into"
    );
    let rows = assert_pixels("frame 1", &from_pin, stride, want);
    assert!(rows > 0, "no row fitted inside the pinned window");

    // ── Frame 2: same destination, same pin, different content ───────────
    processor
        .convert(
            &src,
            &mut dst,
            Some(edgefirst_image::ConvertOptions::new().flip(Flip::Horizontal)),
        )
        .expect("convert frame 2");

    assert_eq!(
        pin.as_ptr(),
        addr,
        "the address survived one convert but not a second -- the steady-state \
         loop is where a runtime actually lives"
    );

    let flipped = read_through_pin(&dst, &pin);
    assert_ne!(
        flipped, from_pin,
        "frame 2 is byte-identical to frame 1 -- the second convert did not \
         reach the pinned buffer"
    );
    assert_pixels("frame 2", &flipped, stride, |x, y| want(W - 1 - x, y));

    note(&format!(
        "pin_survives_convert: {rows}/{H} rows checked, stride={stride}, \
         alignment={}, gl_fallbacks={} (a non-zero count means convert fell \
         back to CPU and the EGL import was never exercised)",
        pin.alignment(),
        processor.convert_fallback_count(),
    ));
}

#[test]
fn pin_taken_after_convert_sees_the_converted_frame() {
    if !zero_copy::served_or_skip() {
        return;
    }

    let src = ramp_src();
    let mut dst = image(PixelFormat::Rgb);
    let stride = dst.effective_row_stride().expect("dst row stride");
    let mut processor = ImageProcessor::new().expect("create ImageProcessor");

    processor
        .convert(&src, &mut dst, None)
        .expect("convert before pinning");

    let pin = dst
        .pin_host(CpuAccess::ReadWrite)
        .expect("pin after convert");
    let bytes = read_through_pin(&dst, &pin);
    let rows = assert_pixels("late pin", &bytes, stride, want);
    assert!(rows > 0, "no row fitted inside the pinned window");
}
