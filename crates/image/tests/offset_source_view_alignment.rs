// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A source at a byte offset inside a DMA-BUF must convert to its own window
//! through the GL backend at every offset, not only at the ones that happen to
//! be 64-byte aligned. Issue #165.
//!
//! On Mali (i.MX 95) the DMA-BUF import silently samples zeros when
//! `EGL_DMA_BUF_PLANE0_OFFSET_EXT` is not a multiple of 64 -- no EGL error,
//! the import succeeds -- so a 16x16 view at (8, 8) of a 256-byte-pitched RGBA
//! surface (offset 2080) came back all `[0, 0, 0, 255]` while (0, 8) (offset
//! 2048) and (16, 0) (offset 64) were fine; V3D converts all of them. The
//! engine imports such a source from the 64-byte-aligned base below its
//! offset, widened by the remainder in pixels, and folds that shift into the
//! sampling rectangle -- so the view stays zero-copy on Mali (issue #170).
//! The NV12 case below still declines and uploads: its combined-plane R8
//! import has no room to widen, its pitch being its width.
//!
//! The NV12 case also pins #166: a declined NV source must reach the R8
//! *upload*, not the CPU converter -- `GLProcessorThreaded` is driven
//! directly, so a CPU fallback cannot pass this file.
//!
//! **The NV12 case is a plane offset, not a `view()`.** `Tensor::view` is
//! packed-only (`view() supports packed formats only (got Nv12)`), so a
//! semi-planar sub-rectangle does not exist as an API. The offset a real NV
//! source arrives at is a whole frame inside a bigger buffer -- a pool slot,
//! or the window `interop::apply_plane_offset` hands back -- which is what
//! `set_plane_offset` builds here, on a real DMA-BUF NV12 surface so the
//! zero-copy R8 import is the path actually under test.
//!
//! Every offset here is asserted, including the aligned ones, so a regression
//! on the drivers that handle unaligned offsets natively (V3D, Vivante) is
//! caught by the same file.
//!
//! The ANGLE leaves (`platform/angle.rs`, `platform/windows.rs`) refuse an
//! offset source through the same `refuse_offset_source` import-failure arms
//! this file pins, but this file is Linux-only, so an IOSurface or D3D11
//! offset NV source is not built here -- that upload is covered separately
//! by `offset_nv_source_upload.rs`, which drives `Mem`.

#![cfg(all(target_os = "linux", feature = "opengl"))]

use edgefirst_image::{
    CPUProcessor, Crop, Flip, GLProcessorThreaded, ImageProcessorTrait, Rotation,
};
use edgefirst_tensor::{
    ColorEncoding, ColorRange, Colorimetry, CpuAccess, DType, PixelFormat, Region, TensorDyn,
    TensorMapTrait, TensorMemory,
};

const W: usize = 64;
const H: usize = 64;
const SIDE: usize = 16;
const BPP: usize = 4;

/// The alignment Mali's DMA-BUF import needs; mirrored from the engine so the
/// NV12 case can build one offset on either side of it.
const MALI_ALIGN: usize = 64;

/// The origins the imx95 probe classified; the two unaligned ones (offsets 32
/// and 2080 at a 256-byte pitch) are the regression, the rest the control.
const ORIGINS: [(usize, usize); 6] = [(0, 0), (8, 8), (0, 8), (8, 0), (16, 0), (0, 1)];

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

fn want(x: usize, y: usize) -> [u8; BPP] {
    [((y * 4) % 256) as u8, ((x * 4) % 256) as u8, 255, 255]
}

fn gl_or_skip() -> Option<GLProcessorThreaded> {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    match GLProcessorThreaded::new(None) {
        Ok(gl) => Some(gl),
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
            );
            skip(&format!("no GL backend: {e}"));
            None
        }
    }
}

/// A DMA-BUF image, or a skip. Under `HAL_TEST_REQUIRE_GL=1` a missing or
/// downgraded DMA-BUF is a FAILURE, not a skip: the boards this file exists
/// for all have a heap, so a heap that needs privileges (or an allocation that
/// quietly lands on the heap allocator) would otherwise let the whole file
/// pass without importing anything -- vacuously green on exactly the path
/// under test. The GL bring-up above carries the same rule.
fn dmabuf_image_or_skip(w: usize, h: usize, fmt: PixelFormat) -> Option<TensorDyn> {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    match TensorDyn::image(
        w,
        h,
        fmt,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) if t.memory() == TensorMemory::DmaBuf => Some(t),
        Ok(t) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the {fmt:?} zero-copy request fell back to {:?}",
                t.memory()
            );
            skip(&format!(
                "{fmt:?}: zero-copy request fell back to {:?}",
                t.memory()
            ));
            None
        }
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but no DMA-BUF could be allocated for {fmt:?}: {e}"
            );
            skip(&format!("{fmt:?}: no DMA-BUF here: {e}"));
            None
        }
    }
}

fn rgba_dst(w: usize, h: usize) -> TensorDyn {
    TensorDyn::new(&[h, w, BPP], DType::U8, Some(TensorMemory::Mem), None)
        .expect("dst")
        .with_format(PixelFormat::Rgba)
        .expect("dst format")
}

fn bytes(t: &TensorDyn) -> Vec<u8> {
    t.map_bytes(CpuAccess::Read)
        .expect("map")
        .as_slice()
        .to_vec()
}

#[test]
fn rgba_source_views_convert_their_own_tile_at_every_origin() {
    let Some(mut gl) = gl_or_skip() else { return };
    let Some(src) = dmabuf_image_or_skip(W, H, PixelFormat::Rgba) else {
        return;
    };
    let pitch = src.effective_row_stride().unwrap_or(W * BPP);
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map src");
        let s = m.as_mut_slice();
        for y in 0..H {
            for x in 0..W {
                s[y * pitch + x * BPP..][..BPP].copy_from_slice(&want(x, y));
            }
        }
    }
    let before = gl.convert_stats().expect("convert stats");
    let mut failures = Vec::new();
    for (x0, y0) in ORIGINS {
        let view = src.view(Region::new(x0, y0, SIDE, SIDE)).expect("view");
        let mut dst = rgba_dst(SIDE, SIDE);
        if let Err(e) = gl.convert(&view, &mut dst, Rotation::None, Flip::None, Crop::default()) {
            failures.push(format!(
                "({x0},{y0}) offset {}: convert failed: {e}",
                y0 * pitch + x0 * BPP
            ));
            continue;
        }
        let got = bytes(&dst);
        let mut expected = Vec::with_capacity(SIDE * SIDE * BPP);
        for y in 0..SIDE {
            for x in 0..SIDE {
                expected.extend_from_slice(&want(x0 + x, y0 + y));
            }
        }
        if got != expected {
            let kind = if got.chunks(BPP).all(|p| p == [0, 0, 0, 255]) {
                "ZEROS (issue #165: the import sampled nothing)"
            } else {
                "wrong pixels"
            };
            failures.push(format!(
                "({x0},{y0}) offset {}: {kind}; first={:?} expected={:?}",
                y0 * pitch + x0 * BPP,
                &got[..BPP],
                &expected[..BPP]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pitch {pitch}:\n{}",
        failures.join("\n")
    );

    let after = gl.convert_stats().expect("convert stats");
    let imports = after.src_imports - before.src_imports;
    let uploads = after.src_uploads - before.src_uploads;
    let pbo = after.src_pbo_uploads - before.src_pbo_uploads;
    // A driver with no DMA-BUF import at all feeds every one of them by
    // copy and there is no route to assert -- the desktop NVIDIA GPU is
    // exactly that: it has a DMA heap, so the allocation above is a real
    // DMA-BUF and the file does not skip, but its EGL has no working
    // DMA-BUF import. The copy is a `map()` upload on some drivers and a
    // PBO on others, so both count here; leaving PBO out let a PBO-fed
    // desktop fall past this guard into the assertions below. The pixels
    // above are the whole assertion on such a driver.
    if imports == 0 && (uploads + pbo) as usize == ORIGINS.len() {
        skip("this GL driver cannot import DMA-BUF at all; the pixels above are the assertion");
        return;
    }
    // Issue #170: an unaligned source is no longer declined into an upload.
    // It imports from a 64-byte-aligned base widened by the remainder in
    // pixels, and the engine folds the shift into the sampling rectangle --
    // so every one of these origins is a zero-copy import on every driver
    // with a DMA-BUF import, Mali included, where offsets 32 and 2080 used
    // to take the upload path.
    //
    // Asserted alongside the pixels, not instead of them: an engine that
    // silently uploaded would still produce the right tile and pass the
    // loop above, which is exactly how #170 could regress unnoticed.
    assert_eq!(
        uploads + pbo,
        0,
        "{uploads} of {} source views were uploaded and {pbo} went by PBO, \
         not imported (issue #170: the aligned-base rebase must keep them \
         zero-copy); imports={imports}",
        ORIGINS.len()
    );
    assert_eq!(
        imports as usize,
        ORIGINS.len(),
        "every origin must be fed by import; uploads={uploads} pbo={pbo}"
    );
    assert_eq!(
        after.zero_copy_declines, before.zero_copy_declines,
        "no source import may be declined at any of these origins"
    );
}

/// A whole RGBA frame at a byte offset inside a bigger DMA-BUF -- the shape a
/// pool slot or an `interop::apply_plane_offset` window arrives in -- on a
/// surface whose pitch has no room to widen.
///
/// This is the case a Mali decline keyed on the FORMAT gets wrong. 64 RGBA
/// pixels is a 256-byte row, already 64-byte aligned, so nothing is padded on
/// and the pitch is tight; folding a 32-byte offset would need the import
/// widened to 72 pixels (288 bytes), which runs past that pitch, so
/// `from_tensor` keeps the frame's own unaligned offset and Mali must go on
/// declining it into the upload. Exempt it because "RGBA is a format the fold
/// covers" and it imports unaligned and comes back black (issue #170).
///
/// The pixels are the assertion that separates the drivers, and they have to
/// be: nothing here can name the renderer. On Mali a wrongly-exempted import
/// samples zeros and this fails; on V3D and Vivante the same import is read
/// correctly and it passes, which is the behaviour those two must keep. The
/// counters pin the other half -- exactly one source feed per convert, so a
/// silent CPU fallback cannot pass either.
#[test]
fn rgba_offset_frames_convert_at_a_pitch_with_no_room_to_rebase() {
    let Some(mut gl) = gl_or_skip() else { return };
    // Three frames' worth of rows so a window can start past the first frame
    // and still be fully backed, then re-tagged to one frame's geometry at an
    // offset, which keeps the surface pitch -- as in the NV12 case below.
    let Some(mut src) = dmabuf_image_or_skip(W, H * 3, PixelFormat::Rgba) else {
        return;
    };
    let pitch = src.effective_row_stride().unwrap_or(W * BPP);
    assert_eq!(
        pitch,
        W * BPP,
        "precondition: {W} RGBA pixels is {} bytes, already 64-aligned, so the \
         surface pitch is tight and the rebase has no room to widen into",
        W * BPP
    );
    // A pattern whose period (31) is coprime with both the pitch and 64, so a
    // window read at the wrong offset cannot coincide with the right one.
    // Written BEFORE the re-tag, while the map still covers every row.
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map src");
        let s = m.as_mut_slice();
        for (i, b) in s.iter_mut().enumerate() {
            *b = 112 + (i % 31) as u8;
        }
    }
    src.set_logical_shape(&[H, W, BPP])
        .expect("one frame's geometry");
    assert_eq!(src.height(), Some(H), "precondition: one frame is {H} rows");
    assert_eq!(
        src.effective_row_stride(),
        Some(pitch),
        "precondition: the re-tag kept the surface pitch"
    );

    let frame = pitch * H;
    let aligned = frame.next_multiple_of(MALI_ALIGN);
    let unaligned = aligned + MALI_ALIGN / 2;
    assert_ne!(
        unaligned % MALI_ALIGN,
        0,
        "precondition: {unaligned} is not aligned"
    );

    let mut failures = Vec::new();
    for offset in [aligned, unaligned] {
        src.set_plane_offset(offset);
        assert_eq!(
            src.plane_offset(),
            Some(offset),
            "precondition: the offset stuck"
        );
        let mut reference = rgba_dst(W, H);
        CPUProcessor::new()
            .convert(
                &src,
                &mut reference,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )
            .expect("CPU reference");
        let want = bytes(&reference);

        let before = gl.convert_stats().expect("convert stats");
        let mut dst = rgba_dst(W, H);
        let outcome = gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default());
        let after = gl.convert_stats().expect("convert stats");
        let imports = after.src_imports - before.src_imports;
        let uploads = after.src_uploads - before.src_uploads;
        let pbo = after.src_pbo_uploads - before.src_pbo_uploads;

        match outcome {
            Err(e) => failures.push(format!("offset {offset}: GL refused the source: {e}")),
            Ok(()) => {
                let got = bytes(&dst);
                if got != want {
                    let kind = if got.chunks(BPP).all(|p| p == [0, 0, 0, 255]) {
                        "ZEROS (issue #170: an unrebased offset was imported anyway)"
                    } else {
                        "wrong pixels"
                    };
                    failures.push(format!(
                        "offset {offset}: {kind}; first={:?} expected={:?} \
                         (imports={imports} uploads={uploads} pbo={pbo})",
                        &got[..BPP],
                        &want[..BPP]
                    ));
                }
            }
        }
        if imports + uploads + pbo != 1 {
            failures.push(format!(
                "offset {offset}: the source must be fed exactly once, not \
                 imports={imports} uploads={uploads} pbo={pbo} -- a silent CPU \
                 fallback would pass the pixels above"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pitch {pitch}, aligned {aligned}, unaligned {unaligned}:\n{}",
        failures.join("\n")
    );
}

/// An NV12 DMA-BUF frame at a 64-aligned and at an unaligned plane offset,
/// each against a CPU reference over the same tensor. On Mali the unaligned
/// one is declined by the zero-copy R8 import and must come back through the
/// R8 *upload* (issue #166's routing), which `GLProcessorThreaded` proves by
/// returning `Ok` at all: the old failure arm called `draw_src_texture`, which
/// has no NV arm and returned `NotSupported`.
#[test]
fn nv12_offset_sources_convert_through_gl_at_aligned_and_unaligned_offsets() {
    let Some(mut gl) = gl_or_skip() else { return };
    // Three frames' worth of rows so a window can start past the first frame
    // and still be fully backed. Allocated as a real NV12 surface, then
    // re-tagged to one frame's geometry at an offset, which keeps the driver's
    // pitch -- the shape a pool slot arrives in.
    let Some(mut src) = dmabuf_image_or_skip(W, H * 3, PixelFormat::Nv12) else {
        return;
    };
    let pitch = src.effective_row_stride().unwrap_or(W);
    // A pattern whose period (31) is coprime with both the pitch and 64, so a
    // window read at the wrong offset cannot coincide with the right one.
    // Kept inside 112..=142 so neither the luma nor the chroma it also feeds
    // drives any RGB channel into clamping, where two different reads could
    // agree by saturation. Written BEFORE the re-tag: afterwards the map is one
    // frame long and every window this test reads would stay at zero.
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map src");
        let s = m.as_mut_slice();
        for (i, b) in s.iter_mut().enumerate() {
            *b = 112 + (i % 31) as u8;
        }
    }
    src.set_logical_shape(&[H * 3 / 2, W])
        .expect("one frame's geometry");
    assert_eq!(src.height(), Some(H), "precondition: one frame is {H} rows");
    assert_eq!(
        src.effective_row_stride(),
        Some(pitch),
        "precondition: the re-tag kept the surface pitch"
    );
    // Full-range BT.601 so both converters resolve the same matrix instead of
    // the untagged SD heuristic (which is limited-range and would still agree,
    // but says so by accident rather than by contract).
    src.set_colorimetry(Some(
        Colorimetry::default()
            .with_encoding(ColorEncoding::Bt601)
            .with_range(ColorRange::Full),
    ));

    let frame = pitch * (H * 3 / 2);
    let aligned = frame.next_multiple_of(MALI_ALIGN);
    let unaligned = aligned + MALI_ALIGN / 2;
    assert_eq!(
        aligned % MALI_ALIGN,
        0,
        "precondition: {aligned} is aligned"
    );
    assert_ne!(
        unaligned % MALI_ALIGN,
        0,
        "precondition: {unaligned} is not aligned"
    );

    let mut failures = Vec::new();
    let mut references = Vec::new();
    for offset in [aligned, unaligned] {
        src.set_plane_offset(offset);
        assert_eq!(
            src.plane_offset(),
            Some(offset),
            "precondition: the offset stuck"
        );
        let mut reference = rgba_dst(W, H);
        CPUProcessor::new()
            .convert(
                &src,
                &mut reference,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )
            .expect("CPU reference");
        let want = bytes(&reference);
        let mut dst = rgba_dst(W, H);
        match gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default()) {
            Err(e) => failures.push(format!(
                "offset {offset}: GL refused the NV12 source: {e} \
                 (issue #166: the declined import must reach the R8 upload)"
            )),
            Ok(()) => {
                let got = bytes(&dst);
                let bad = got
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| g.abs_diff(**w) > 8)
                    .count();
                if bad > 0 {
                    // A flat output means the import sampled one constant.
                    // Issue #165's zero-sampling reads Y=U=V=0, which
                    // full-range BT.601 turns into [0, 136, 0, 255] -- not the
                    // [0, 0, 0, 255] the RGBA case above looks for.
                    let flat = got.chunks(BPP).all(|p| p == &got[..BPP]);
                    failures.push(format!(
                        "offset {offset}: {bad} bytes differ from the CPU reference{}; \
                         first got={:?} want={:?}",
                        if flat {
                            " (FLAT: issue #165, the import sampled a constant)"
                        } else {
                            ""
                        },
                        &got[..BPP],
                        &want[..BPP]
                    ));
                }
            }
        }
        references.push(want);
    }
    // The two windows must actually differ, or "GL matches CPU" would hold for
    // an engine that ignored the offset entirely.
    assert_ne!(
        references[0], references[1],
        "precondition: offsets {aligned} and {unaligned} name different windows"
    );
    assert!(
        failures.is_empty(),
        "pitch {pitch}:\n{}",
        failures.join("\n")
    );
}

/// A DESTINATION at a plane offset Mali cannot *sample* from. Rendering into an
/// unaligned base is a different operation from sampling one, and the
/// destination exemption in `get_or_create_egl_image` claims it is safe, so it
/// is measured here rather than assumed.
///
/// Measured: Mali (i.MX 95) and V3D render into offset 2080 correctly, so Mali
/// destinations keep the zero-copy import. Vivante does NOT — `eglCreateImage`
/// returns `EGL_BAD_ACCESS` at 2080 while 2048 renders — which this test found
/// and `vivante_rejects_dst_import_offset` now lowers to the mapped-texture
/// path. Both offsets are asserted on every driver, so either half regressing
/// is caught here.
///
/// The destination is rebuilt from a `view()`'s descriptor, which is the only
/// shape that reaches the import at a nonzero offset: a fresh `view()` collapses
/// onto its parent's key at offset 0 (`cache.rs`'s `for_dst` arm), while a
/// rebuilt one carries the offset and no `view_origin`, so on Linux
/// `dst_import_places` keeps it zero-copy and the import base IS the offset.
/// Same pattern as `reconstructed_view_convert.rs` case E, at both an aligned
/// and an unaligned offset.
#[test]
fn rgba_offset_destinations_place_their_tile_at_aligned_and_unaligned_offsets() {
    const BLANK: u8 = 0x55;
    let Some(mut gl) = gl_or_skip() else { return };
    let Some(canvas) = dmabuf_image_or_skip(W, H, PixelFormat::Rgba) else {
        return;
    };
    let pitch = canvas.effective_row_stride().unwrap_or(W * BPP);

    let mut failures = Vec::new();
    // (0, 8) is 8 * pitch, aligned whenever the pitch is; (8, 8) adds
    // 8 * BPP = 32, which is never 64-aligned. Same construction as the source
    // test, so the two halves are measured at the same two offsets.
    for (x0, y0) in [(0usize, 8usize), (8, 8)] {
        let offset = y0 * pitch + x0 * BPP;
        {
            let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
            m.as_mut_slice().fill(BLANK);
        }
        let fresh = canvas
            .view(Region::new(x0, y0, SIDE, SIDE))
            .expect("fresh destination view");
        assert_eq!(
            fresh.plane_offset(),
            Some(offset),
            "precondition: the view's offset is the pitch arithmetic above"
        );
        let mut dst = TensorDyn::import_descriptor(&fresh.descriptor_pinned(None))
            .expect("rebuild the destination view from its descriptor");
        assert_eq!(
            dst.effective_row_stride(),
            Some(pitch),
            "precondition: the rebuilt destination kept the canvas pitch"
        );
        dst.set_plane_offset(offset);
        assert_eq!(
            dst.view_origin(),
            None,
            "precondition: rebuilt, so no viewport to place it by"
        );

        // The source carries the tile's ABSOLUTE canvas colours, so a tile that
        // lands at the wrong origin is wrong in value, not just in place.
        let src = rgba_dst(SIDE, SIDE);
        {
            let mut m = src.map_bytes(CpuAccess::Write).expect("map src");
            let s = m.as_mut_slice();
            for y in 0..SIDE {
                for x in 0..SIDE {
                    s[(y * SIDE + x) * BPP..][..BPP].copy_from_slice(&want(x0 + x, y0 + y));
                }
            }
        }
        if let Err(e) = gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default()) {
            failures.push(format!("({x0},{y0}) offset {offset}: convert failed: {e}"));
            continue;
        }
        let out = bytes(&canvas);
        let px = |x: usize, y: usize| &out[y * pitch + x * BPP..][..BPP];
        if px(0, 0) != [BLANK; BPP] {
            failures.push(format!(
                "({x0},{y0}) offset {offset}: the tile landed at the canvas ORIGIN, \
                 i.e. the import ignored the offset; (0,0)={:?}",
                px(0, 0)
            ));
            continue;
        }
        if y0 > 0 && px(x0, y0 - 1) != [BLANK; BPP] {
            failures.push(format!(
                "({x0},{y0}) offset {offset}: the row above the window was written: {:?}",
                px(x0, y0 - 1)
            ));
        }
        if x0 > 0 && px(x0 - 1, y0) != [BLANK; BPP] {
            failures.push(format!(
                "({x0},{y0}) offset {offset}: the column left of the window was written: {:?}",
                px(x0 - 1, y0)
            ));
        }
        let mut wrong = 0usize;
        let mut first = None;
        for y in y0..y0 + SIDE {
            for x in x0..x0 + SIDE {
                if px(x, y) != want(x, y) {
                    wrong += 1;
                    first.get_or_insert((x, y, px(x, y).to_vec(), want(x, y)));
                }
            }
        }
        if let Some((x, y, got, exp)) = first {
            let flat = out.chunks(BPP).take(SIDE).all(|p| p == [0, 0, 0, 255]);
            failures.push(format!(
                "({x0},{y0}) offset {offset}: {wrong} tile pixels wrong{}; \
                 ({x},{y}) got={got:?} expected={exp:?}",
                if flat {
                    " (ZEROS: the render target sampled/wrote nothing)"
                } else {
                    ""
                }
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "pitch {pitch}:\n{}",
        failures.join("\n")
    );
}
