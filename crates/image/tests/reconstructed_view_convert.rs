// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Issue #161 at the `convert()` level — the path the bug was actually
//! reported through.
//!
//! `crates/tensor/src/iosurface.rs`'s three offset tests prove the storage
//! contract: `set_plane_offset` moves the CPU-map window, `set_format`
//! clears it, and nested subviews do not compound. None of them proves the
//! thing a user sees, which is
//! `ImageProcessor.convert(reconstructed_view, dst)` returning the parent
//! image's top-left tile instead of the region that was asked for. A fix
//! that satisfied `map()` and left the converter reading the parent origin
//! would pass all three and still ship the reported bug.
//!
//! So this covers the converter, and covers it from **both** directions:
//!
//! * **A fresh `view()`**, which was never broken — each backend's own
//!   `view()` sets its own offset. It is here as the control: if A fails,
//!   the harness is wrong, not the fix.
//! * **A reconstructed view**, rebuilt the way `import_storage`'s
//!   `kind::IOSURFACE` arm rebuilds one (`lookup_by_id` + `from_iosurface`,
//!   spelled `TensorDyn::from_iosurface_id`) and handed its offset back the
//!   way `interop::apply_plane_offset` hands it back. This is the case
//!   `TensorDesc` drops the offset for.
//! * **A whole-image tensor carrying a foreign offset**, the shape a
//!   producer hands over when pixel data does not start at byte 0. Distinct
//!   from a view: its logical extent is not a sub-rectangle, so it exercises
//!   the offset without `view()` having been involved at all.
//!
//! **`TensorMemory::DmaBuf` is the portable spelling of "platform zero-copy
//! buffer"** — DMA-BUF on Linux, IOSurface on macOS — so this runs on both
//! without a cfg, and skips where no such buffer can be allocated.
//!
//! **RGBA, not RGB, and that is load-bearing on macOS.** An RGB IOSurface
//! has no zero-copy GL mapping, so an RGB source silently converts on the
//! CPU: under `EDGEFIRST_FORCE_BACKEND=opengl` it fails outright with
//! `NotSupported("Opengl doesn't support RGB source texture")`. An RGB
//! version of this test would therefore pass without the GL path ever
//! running — vacuous on exactly the platform it exists for.
//!
//! Verified non-vacuous under `EDGEFIRST_FORCE_BACKEND=opengl` (which
//! errors rather than falling back, so the GL path is proven, not assumed):
//! all three cases pass with the fix, and B reads the parent's origin when
//! the `set_plane_offset` call is removed.

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};

const W: usize = 64;
const H: usize = 64;
const X0: usize = 8;
const Y0: usize = 8;
const SIDE: usize = 16;
const BPP: usize = 4;

/// libtest discards `println!` for *passing* tests, so a skip would be
/// indistinguishable from a run. Write to stderr directly.
fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

/// Distinct per-row and per-column values, so a convert that lands on the
/// wrong region cannot coincidentally match the expected tile.
fn want(x: usize, y: usize) -> [u8; BPP] {
    [((y * 4) % 256) as u8, ((x * 4) % 256) as u8, 255, 255]
}

fn tile(x0: usize, y0: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(SIDE * SIDE * BPP);
    for y in y0..y0 + SIDE {
        for x in x0..x0 + SIDE {
            v.extend_from_slice(&want(x, y));
        }
    }
    v
}

fn read_all(t: &TensorDyn) -> Vec<u8> {
    let m = t.map_bytes(CpuAccess::Read).expect("map destination");
    m.as_slice().to_vec()
}

fn dst() -> TensorDyn {
    TensorDyn::new(&[SIDE, SIDE, BPP], DType::U8, Some(TensorMemory::Mem), None)
        .expect("destination alloc")
        .with_format(PixelFormat::Rgba)
        .expect("destination format")
}

/// Assert the convert landed on the requested tile, and say *how* it went
/// wrong when it did — reading the parent's origin is the specific
/// signature of a lost plane offset, and worth naming in the failure.
fn assert_tile(label: &str, out: &[u8]) {
    let expected = tile(X0, Y0);
    if out == expected.as_slice() {
        return;
    }
    let origin = tile(0, 0);
    if out == origin.as_slice() {
        panic!(
            "{label}: convert() read the parent buffer's ORIGIN, i.e. the \
             plane offset was lost (issue #161)"
        );
    }
    panic!(
        "{label}: convert() read neither the requested tile nor the parent's \
         origin; out[0..4]={:?} expected[0..4]={:?}",
        &out[..BPP],
        &expected[..BPP]
    );
}

#[test]
fn reconstructed_view_converts_its_own_sub_region_not_the_parents_origin() {
    let src = match TensorDyn::new(&[H, W, BPP], DType::U8, Some(TensorMemory::DmaBuf), None) {
        Ok(t) if t.memory() == TensorMemory::DmaBuf => t,
        Ok(t) => {
            skip(&format!("zero-copy request fell back to {:?}", t.memory()));
            return;
        }
        Err(e) => {
            skip(&format!("no zero-copy buffer here: {e}"));
            return;
        }
    };
    let src = src.with_format(PixelFormat::Rgba).expect("source format");
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map source");
        let s = m.as_mut_slice();
        for y in 0..H {
            for x in 0..W {
                s[(y * W + x) * BPP..][..BPP].copy_from_slice(&want(x, y));
            }
        }
    }

    let mut proc = ImageProcessor::new().expect("create ImageProcessor");

    // A — the control: a fresh view was correct on every platform already.
    let fresh = src
        .view(Region::new(X0, Y0, SIDE, SIDE))
        .expect("fresh view");
    let mut dst_a = dst();
    proc.convert(
        &fresh,
        &mut dst_a,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .expect("convert fresh view");
    assert_tile("fresh view", &read_all(&dst_a));

    // B — the regression: a view rebuilt from a descriptor. IOSurface-only,
    // because `from_iosurface_id` is the Apple spelling of the import; the
    // Linux DMA-BUF equivalent is already covered end-to-end by
    // `test_view_converts_its_own_sub_region_not_the_parents_origin` in
    // `tests/interop/test_cross_package.py`, which cannot run on macOS.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let id = src.iosurface_id().expect("source is IOSurface-backed");
        let mut rebuilt = TensorDyn::from_iosurface_id(id, &[SIDE, SIDE, BPP], DType::U8, None)
            .expect("reconstruct at the view's shape");
        rebuilt
            .set_format(PixelFormat::Rgba)
            .expect("rebuilt format");
        // The parent's pitch, as a descriptor round trip restores it: the
        // sub-region's rows are strided across the parent, not contiguous.
        rebuilt.set_row_stride(W * BPP).expect("rebuilt row stride");
        // What `interop::apply_plane_offset` puts back, and what this PR
        // makes take effect on IOSurface.
        rebuilt.set_plane_offset((Y0 * W + X0) * BPP);

        let mut dst_b = dst();
        proc.convert(
            &rebuilt,
            &mut dst_b,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .expect("convert reconstructed view");
        assert_tile("reconstructed view", &read_all(&dst_b));

        // C — a whole-image tensor whose pixels start one row in. Not a
        // view: no sub-rectangle is involved, so this pins the offset
        // itself rather than `view()`'s bookkeeping.
        let mut whole = TensorDyn::from_iosurface_id(id, &[H - 1, W, BPP], DType::U8, None)
            .expect("reconstruct whole image");
        whole.set_format(PixelFormat::Rgba).expect("whole format");
        whole.set_plane_offset(W * BPP);

        let mut dst_c = TensorDyn::new(&[H - 1, W, BPP], DType::U8, Some(TensorMemory::Mem), None)
            .expect("dst_c alloc")
            .with_format(PixelFormat::Rgba)
            .expect("dst_c format");
        proc.convert(
            &whole,
            &mut dst_c,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .expect("convert whole image at a foreign offset");
        let out = read_all(&dst_c);
        assert_eq!(
            &out[..BPP],
            &want(0, 1),
            "a whole image at a one-row offset must start at parent row 1, \
             not row 0 (issue #161)"
        );
    }
}
