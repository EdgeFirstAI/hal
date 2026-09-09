// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Issue #161 at the `convert()` level — the path the bug was actually
//! reported through.
//!
//! `crates/tensor/src/iosurface.rs`'s offset tests prove the storage
//! contract: `set_plane_offset` moves the CPU-map window, `set_format`
//! clears it, and nested subviews do not compound. None of them proves the
//! thing a user sees, which is
//! `ImageProcessor.convert(reconstructed_view, dst)` returning the parent
//! image's top-left tile instead of the region that was asked for. A fix
//! that satisfied `map()` and left the converter reading the parent origin
//! would pass all of them and still ship the reported bug.
//!
//! So this covers the converter, from every direction the offset can arrive:
//!
//! * **A — a fresh `view()`.** Each backend's own `view()` sets its own
//!   offset, so the *storage* was always right. The GL leaf is another
//!   matter: ANGLE binds a whole buffer from its origin (over a D3D11
//!   texture and over an IOSurface alike) and has no offset attribute, so
//!   until the leaf refused to attach an offset source, a fresh view
//!   converted the parent's origin on the zero-copy path. A is a control for
//!   the harness *and* the pin for that refusal.
//! * **B — a view rebuilt from its descriptor**, the way a capsule consumer
//!   rebuilds one (`import_descriptor`), and handed its offset back the way
//!   `interop::apply_plane_offset` hands it back. This is the case
//!   `TensorDesc` drops the offset for.
//! * **C — a whole image at a foreign offset.** Not a view: its logical
//!   extent is not a sub-rectangle, so it exercises the offset without
//!   `view()`'s bookkeeping.
//! * **D — a one-row view.** Its descriptor carries the tight stride
//!   `view()` records for a single row while its offset is measured in the
//!   parent's pitch; the round trip must not confuse the two.
//! * **E — the destination side.** A destination rebuilt from a descriptor
//!   has the offset but not the `view_origin` a fresh view has, so the
//!   engine has no viewport to place it by and must take the mapped-texture
//!   path (`GlPlatform::dst_import_places`).
//!
//! **The source is an image-formatted surface, not a byte-bag.** On Apple,
//! `TensorDyn::new(.., DmaBuf)` allocates a one-row `L008` byte-bag that
//! ANGLE cannot bind an RGBA pbuffer over, so a test built on it takes the
//! upload path (which honours the offset) without the zero-copy import ever
//! running — vacuous on exactly the path this file exists for.
//! `TensorDyn::image(..)` allocates a real `BGRA` surface on Apple and a real
//! texture on Windows.
//!
//! **`W` is 50, not a multiple of 16.** IOSurface 64-aligns its pitch, so a
//! 50-texel RGBA row (200 B) is stored at 256 B, and "the import kept the
//! pitch" is only a meaningful assertion when the pitch differs from the
//! natural row.
//!
//! **Known red on i.MX 95 (Mali).** There the GL engine converts a source
//! `view()` of an RGBA DMA-BUF to zeros -- neither the tile nor the origin
//! -- at any width and with either allocation spelling, and the file as
//! merged in #163 fails there the same way; V3D and the CPU path are
//! correct. That is a pre-existing Linux defect in the zero-copy import of
//! an offset source view on Mali, not this file's subject, and needs its
//! own pin and fix.
//!
//! **RGBA, not RGB, and that is load-bearing on macOS.** An RGB IOSurface
//! has no zero-copy GL mapping, so an RGB source silently converts on the
//! CPU: under `EDGEFIRST_FORCE_BACKEND=opengl` it fails outright with
//! `NotSupported("Opengl doesn't support RGB source texture")`.
//!
//! `TensorMemory::DmaBuf` is the portable spelling of "platform zero-copy
//! buffer" — DMA-BUF on Linux, IOSurface on macOS, a D3D11 texture on
//! Windows — so A runs on all three without a cfg, and the file skips where
//! no such buffer can be allocated. B–E go through `import_descriptor`,
//! which reconstructs a window on Apple and Windows; Linux's DMA-BUF
//! equivalent is covered end-to-end by
//! `test_view_converts_its_own_sub_region_not_the_parents_origin` in
//! `tests/interop/test_cross_package.py`.

use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};

const W: usize = 50;
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

/// A `W`x`H` RGBA image on the platform's zero-copy buffer, or `None` (with
/// the reason on stderr) where none can be allocated.
///
/// Under `HAL_TEST_REQUIRE_GL=1` -- the macOS and Windows lanes, whose
/// zero-copy buffer needs no device node -- a skip is a failure, as it is
/// for a GL backend that fails to come up: everything this file pins is
/// meaningless without the buffer, and a silent skip would leave the lane
/// green with nothing run.
fn zero_copy_image() -> Option<TensorDyn> {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    match TensorDyn::image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) if t.memory() == TensorMemory::DmaBuf => Some(t),
        Ok(t) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the zero-copy request fell back to {:?}",
                t.memory()
            );
            skip(&format!("zero-copy request fell back to {:?}", t.memory()));
            None
        }
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but no zero-copy buffer could be allocated: {e}"
            );
            skip(&format!("no zero-copy buffer here: {e}"));
            None
        }
    }
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
    let Some(src) = zero_copy_image() else {
        return;
    };
    // Rows at the allocation's own pitch: the platform may pad them.
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

    // A — a fresh view. The storage was always right here; the GL leaf was
    // not, until it refused to zero-copy attach a source at an offset.
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

    // B–E: rebuilt through the descriptor itself, on the platforms whose
    // import can reconstruct a window of a whole buffer.
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios"))]
    {
        // IOSurface 64-aligns its pitch, so at W = 50 the pitch (256) must
        // differ from the natural row (200) or the stride assertions below
        // cannot tell a restored pitch from a dropped one. Windows' pitch is
        // whatever the driver chose; no precondition there.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_ne!(
            pitch,
            W * BPP,
            "precondition: the surface pitch must be padded past the natural row"
        );

        // B — the view rebuilt from its descriptor. The handle names the
        // whole parent, `import_descriptor` narrows to the window, and the
        // offset is put back the way `interop::apply_plane_offset` puts it.
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
            "a narrowed import keeps the parent's pitch (restore_imported_row_stride)"
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

        // C — the whole image narrowed by one row and offset by one row.
        let mut whole = TensorDyn::import_descriptor(&src.descriptor_pinned(None))
            .expect("reconstruct whole image");
        whole
            .set_logical_shape(&[H - 1, W, BPP])
            .expect("narrow by one row");
        assert_eq!(
            whole.effective_row_stride(),
            Some(pitch),
            "narrowing keeps the pitch (Tensor::set_logical_shape)"
        );
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

        // D — a one-row view: tight descriptor stride, pitched offset.
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

        // E — the destination side. Offset but no `view_origin`, so no
        // viewport to place it by; the engine must lower it to the mapped
        // texture path (`dst_import_places`) or the tile lands at the
        // canvas's top-left.
        const BLANK: u8 = 0x55;
        let canvas = zero_copy_image().expect("a second zero-copy image after the first succeeded");
        let canvas_pitch = canvas.effective_row_stride().unwrap_or(W * BPP);
        {
            let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
            m.as_mut_slice().fill(BLANK);
        }
        let fresh_dst = canvas
            .view(Region::new(X0, Y0, SIDE, SIDE))
            .expect("fresh destination view");
        let mut rebuilt_dst = TensorDyn::import_descriptor(&fresh_dst.descriptor_pinned(None))
            .expect("reconstruct the destination view from its descriptor");
        assert_eq!(
            rebuilt_dst.effective_row_stride(),
            Some(canvas_pitch),
            "a rebuilt destination keeps the canvas's pitch"
        );
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
}
