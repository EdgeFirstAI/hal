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
//! buffer"** — DMA-BUF on Linux, IOSurface on macOS, a D3D11 texture on
//! Windows — so this runs on all three without a cfg on the control, and
//! skips where no such buffer can be allocated.
//!
//! On Windows the reconstructed cases go through the descriptor itself
//! (`import_descriptor`, which now accepts a window of the texture), and the
//! control is not a formality: the ANGLE D3D11 import binds the whole
//! texture from its origin, so before the GL leaf refused to attach a source
//! carrying a plane offset, a *fresh* `view()` converted the parent's origin
//! on the GL path too.
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
    // A D3D11 texture is only ever allocated through the image constructor
    // (`TensorDyn::new` with `DmaBuf` refuses by name on Windows), which
    // sets the format itself; the byte-bag spelling stays for the others so
    // the surface they get is the one the fix was verified on.
    #[cfg(target_os = "windows")]
    let alloc = TensorDyn::image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    );
    #[cfg(not(target_os = "windows"))]
    let alloc = TensorDyn::new(&[H, W, BPP], DType::U8, Some(TensorMemory::DmaBuf), None)
        .and_then(|t| t.with_format(PixelFormat::Rgba));
    let src = match alloc {
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
    // Rows at the allocation's own pitch: a texture's driver may pad them.
    let pitch = src.effective_row_stride().unwrap_or(W * BPP);
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map source");
        let s = m.as_mut_slice();
        for y in 0..H {
            for x in 0..W {
                s[y * pitch + x * BPP..][..BPP].copy_from_slice(&want(x, y));
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

        // D -- the destination side, the mirror of B. A destination rebuilt
        // from a descriptor carries the offset but not the `view_origin` a
        // fresh `view()` would have given it, so the engine has no viewport
        // to place it by. The ANGLE IOSurface import binds the whole surface
        // from its origin and has no plane-offset attribute, so if such a
        // destination were zero-copy attached the tile would land at the
        // canvas's top-left. Windows refuses that import explicitly
        // (`dst_import_places`); Apple has no override, so this pins the
        // behaviour rather than assuming it.
        const BLANK: u8 = 0x55;
        let canvas = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .expect("canvas alloc");
        {
            let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
            m.as_mut_slice().fill(BLANK);
        }
        let canvas_pitch = canvas.effective_row_stride().unwrap_or(W * BPP);
        let canvas_id = canvas.iosurface_id().expect("canvas is IOSurface-backed");
        let mut rebuilt_dst =
            TensorDyn::from_iosurface_id(canvas_id, &[SIDE, SIDE, BPP], DType::U8, None)
                .expect("reconstruct the destination at the window's shape");
        rebuilt_dst
            .set_format(PixelFormat::Rgba)
            .expect("rebuilt destination format");
        rebuilt_dst
            .set_row_stride(canvas_pitch)
            .expect("rebuilt destination row stride");
        rebuilt_dst.set_plane_offset(Y0 * canvas_pitch + X0 * BPP);

        proc.convert(
            &fresh,
            &mut rebuilt_dst,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .expect("convert into a reconstructed destination view");
        let out = read_all(&canvas);
        let px = |x: usize, y: usize| &out[y * canvas_pitch + x * BPP..][..BPP];
        assert_eq!(
            px(0, 0),
            &[BLANK; BPP],
            "a reconstructed destination view wrote the canvas's ORIGIN, i.e. \
             the plane offset was lost (issue #161)"
        );
        assert_eq!(
            px(X0 - 1, Y0),
            &[BLANK; BPP],
            "left of the window untouched"
        );
        assert_eq!(px(X0, Y0 - 1), &[BLANK; BPP], "above the window untouched");
        for y in Y0..Y0 + SIDE {
            for x in X0..X0 + SIDE {
                assert_eq!(px(x, y), &want(x, y), "tile pixel ({x}, {y})");
            }
        }
    }

    // B and C on Windows, spelled through the descriptor itself: a D3D11
    // texture is imported from its NT handle, which names the whole texture,
    // and `import_descriptor` narrows it to the descriptor's window. The
    // offset is put back the way `interop::apply_plane_offset` puts it back.
    #[cfg(target_os = "windows")]
    {
        let mut rebuilt = TensorDyn::import_descriptor(&fresh.descriptor_pinned(None))
            .expect("reconstruct the view from its descriptor");
        assert_eq!(
            rebuilt.shape(),
            &[SIDE, SIDE, BPP],
            "imported at the window's shape"
        );
        assert_eq!(
            rebuilt.effective_row_stride(),
            Some(pitch),
            "a narrowed import keeps the texture's pitch"
        );
        rebuilt.set_plane_offset(fresh.plane_offset().expect("a view carries its offset"));

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

        // C -- the whole texture narrowed by one row and offset by one row.
        let mut whole = TensorDyn::import_descriptor(&src.descriptor_pinned(None))
            .expect("reconstruct whole image");
        whole
            .set_logical_shape(&[H - 1, W, BPP])
            .expect("narrow by one row");
        whole.set_plane_offset(pitch);

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

        // D -- a one-row view. Its descriptor carries the tight stride
        // `view()` records for a single row, while its offset is measured in
        // the texture's pitch; the round trip must not confuse the two.
        let row = src
            .view(Region::new(X0, Y0, SIDE, 1))
            .expect("one-row view");
        let mut rebuilt_row = TensorDyn::import_descriptor(&row.descriptor_pinned(None))
            .expect("reconstruct the one-row view from its descriptor");
        rebuilt_row.set_plane_offset(row.plane_offset().expect("a view carries its offset"));
        let mut dst_d = TensorDyn::new(&[1, SIDE, BPP], DType::U8, Some(TensorMemory::Mem), None)
            .expect("dst_d alloc")
            .with_format(PixelFormat::Rgba)
            .expect("dst_d format");
        proc.convert(
            &rebuilt_row,
            &mut dst_d,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .expect("convert one-row reconstructed view");
        let expected_row: Vec<u8> = (X0..X0 + SIDE).flat_map(|x| want(x, Y0)).collect();
        assert_eq!(
            read_all(&dst_d),
            expected_row,
            "a one-row reconstructed view must read parent row {Y0} from \
             column {X0} (issue #161)"
        );

        // E -- the destination side. A destination window rebuilt from a
        // descriptor has the offset but not the `view_origin` a fresh view()
        // has, so the engine has no viewport to place it by; the ANGLE
        // import binds the whole texture from its origin, and before the
        // engine lowered such a destination to the mapped texture path the
        // tile landed at the canvas's top-left.
        const BLANK: u8 = 0x55;
        let canvas = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .expect("canvas alloc");
        {
            let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
            m.as_mut_slice().fill(BLANK);
        }
        let fresh_dst = canvas
            .view(Region::new(X0, Y0, SIDE, SIDE))
            .expect("fresh destination view");
        let mut rebuilt_dst = TensorDyn::import_descriptor(&fresh_dst.descriptor_pinned(None))
            .expect("reconstruct the destination view from its descriptor");
        rebuilt_dst.set_plane_offset(fresh_dst.plane_offset().expect("a view carries its offset"));
        proc.convert(
            &fresh,
            &mut rebuilt_dst,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )
        .expect("convert into a reconstructed destination view");
        let out = read_all(&canvas);
        let px = |x: usize, y: usize| &out[y * pitch + x * BPP..][..BPP];
        assert_eq!(
            px(0, 0),
            &[BLANK; BPP],
            "a reconstructed destination view wrote the canvas's ORIGIN, i.e. \
             the plane offset was lost (issue #161)"
        );
        assert_eq!(
            px(X0 - 1, Y0),
            &[BLANK; BPP],
            "left of the window untouched"
        );
        assert_eq!(px(X0, Y0 - 1), &[BLANK; BPP], "above the window untouched");
        for y in Y0..Y0 + SIDE {
            for x in X0..X0 + SIDE {
                assert_eq!(px(x, y), &want(x, y), "tile pixel ({x}, {y})");
            }
        }
    }
}
