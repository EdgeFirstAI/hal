// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Issue #134, criterion 1 — the `convert()` half.
//!
//! `crates/tensor/tests/pin.rs` proves a host pin survives every *map guard*
//! drop and that a subview's pin is offset-adjusted. Neither touches the case
//! the issue was actually filed for: a pin handed to an inference runtime while
//! the HAL keeps converting camera frames into the same buffer. That path goes
//! through a **different** owner — `ImageProcessor::convert` imports the
//! destination as an `EGLImage` (Linux DMA-BUF via
//! `EGL_EXT_image_dma_buf_import`, macOS IOSurface via
//! `EGL_ANGLE_iosurface_client_buffer`), renders into it, and releases the
//! import. If any of that re-created the mapping, re-allocated the buffer, or
//! wrote somewhere other than where the pin points, a runtime holding the
//! pointer would read a stale or dead frame forever and never fault.
//!
//! Three things are asserted per convert, and the third is the one that makes
//! the first two mean anything:
//!
//! 1. **The address does not move.** Captured before the first convert and
//!    re-checked after each.
//! 2. **The pin and a fresh map guard agree byte-for-byte**, so the pin is
//!    looking at the buffer the converter actually wrote — not a copy that
//!    happens to hold the right bytes.
//! 3. **The pixels are the ones the convert was asked for.** Without this a
//!    convert that silently did nothing passes (1) and (2) trivially. A second
//!    convert with a horizontal flip follows, because "the address survived one
//!    convert" and "the address survives a steady-state loop" are different
//!    claims — and the flip's output differs from the first frame's everywhere
//!    except the centre column, so a convert that no-ops the second time is
//!    caught rather than confirmed.
//!
//! **`TensorMemory::DmaBuf` is the portable spelling of "platform zero-copy
//! buffer"** — DMA-BUF on Linux, IOSurface on macOS — so this file covers both
//! without a cfg. It skips where no such buffer can be allocated.
//!
//! A board with no GL backend still runs every assertion; `convert` falls back
//! to CPU, which cannot exercise the EGL import at all. That is a weaker run,
//! not a failing one, so the fallback count is reported rather than asserted —
//! a vacuous pass is visible in the log instead of silent.

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, HostPin, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const W: usize = 64;
const H: usize = 48;

/// TFLite's `kDefaultTensorAlignment`. Checked here because the convert
/// destination *is* the buffer criterion 3 hands to
/// `set_custom_allocation_for_input`, and upstream warns that a misaligned
/// custom allocation can crash inside `Invoke()` rather than being rejected.
const TFLITE_ALIGNMENT: usize = 64;

/// Report a skip so it survives to the log.
///
/// libtest captures `println!`/`eprintln!` and discards it for **passing**
/// tests, so a test that skips and returns is indistinguishable from one that
/// did the work. Writing to `std::io::stderr()` directly bypasses that capture.
fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

/// Emit a note that must survive a *passing* test, for the same reason.
fn note(what: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "NOTE: {what}");
}

/// The analytic source ramp — a distinct value per pixel per channel, so a
/// convert that lands the wrong region, the wrong row, or nothing at all cannot
/// coincidentally match the right answer.
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

/// Read the pinned window, bracketed for CPU read as the coherency contract
/// requires. On a coherent backend both syncs are documented no-ops; on a
/// cached DMA heap they are the invalidate that makes the GPU's writes visible.
fn read_through_pin(t: &TensorDyn, pin: &HostPin<'static>) -> Vec<u8> {
    t.sync_for_cpu(CpuAccess::Read).expect("sync_for_cpu");
    // SAFETY: the convert has completed (`convert` is blocking; the fenced
    // variant is a separate entry point), and the read is bracketed above.
    let bytes = unsafe { pin.as_slice() }.to_vec();
    t.sync_for_device(CpuAccess::Read).expect("sync_for_device");
    bytes
}

/// Read the same tensor through a conventional map guard, which brackets
/// itself and exposes the padded extent.
fn read_through_map(t: &TensorDyn) -> Vec<u8> {
    let u8t = t.as_u8().expect("u8 tensor");
    let map = u8t.map_read().expect("map for read");
    map.as_slice().to_vec()
}

/// Assert every RGB pixel reachable within the pinned window equals
/// `expect(x, y)`. Returns the number of rows actually checked.
///
/// The pin covers the tensor's **logical** extent (`W * H * 3`), which for a
/// pitch-padded destination is shorter than `row_stride * H` — so the tail rows
/// fall outside it. Skipping those is correct rather than a gap: the pinned
/// window is exactly what a consumer is allowed to read, and a padded row that
/// the pin cannot reach is not part of the contract under test.
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
    if !edgefirst_tensor::is_gpu_buffer_available() {
        skip("no zero-copy GPU buffer (no DMA heap / no IOSurface) on this host");
        return;
    }

    let src = ramp_src();
    let mut dst = image(PixelFormat::Rgb);
    let stride = dst.effective_row_stride().expect("dst row stride");

    // Pin BEFORE the first convert and hold it across everything below: that is
    // the lifetime an inference runtime's custom allocation has.
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
        .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
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
    // A horizontal flip differs from frame 1 at every column but the centre, so
    // a second convert that silently no-ops fails here instead of passing.
    processor
        .convert(
            &src,
            &mut dst,
            Rotation::None,
            Flip::Horizontal,
            Crop::default(),
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
    // The mirror case: a runtime that pins *late*, after the pipeline has
    // already been running. The pin must expose the current frame rather than
    // a fresh mapping of stale or zeroed pages.
    if !edgefirst_tensor::is_gpu_buffer_available() {
        skip("no zero-copy GPU buffer (no DMA heap / no IOSurface) on this host");
        return;
    }

    let src = ramp_src();
    let mut dst = image(PixelFormat::Rgb);
    let stride = dst.effective_row_stride().expect("dst row stride");
    let mut processor = ImageProcessor::new().expect("create ImageProcessor");

    processor
        .convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())
        .expect("convert before pinning");

    let pin = dst
        .pin_host(CpuAccess::ReadWrite)
        .expect("pin after convert");
    let bytes = read_through_pin(&dst, &pin);
    let rows = assert_pixels("late pin", &bytes, stride, want);
    assert!(rows > 0, "no row fitted inside the pinned window");
}
