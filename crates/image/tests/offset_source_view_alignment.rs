// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A source at a byte offset inside a DMA-BUF converts to its own window
//! through the GL backend at every offset, 64-byte aligned or not.
//!
//! `GLProcessorThreaded` is driven directly. The NV12 case sets a plane offset
//! on a DMA-BUF NV12 surface with `set_plane_offset`.

#![cfg(all(target_os = "linux", feature = "opengl"))]

use edgefirst_image::{CPUProcessor, GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{
    ColorEncoding, ColorRange, Colorimetry, CpuAccess, DType, PixelFormat, Region, TensorDyn,
    TensorMapTrait, TensorMemory,
};

const W: usize = 64;
const H: usize = 64;
const SIDE: usize = 16;
const BPP: usize = 4;

/// The alignment Mali's DMA-BUF import needs.
const MALI_ALIGN: usize = 64;

/// The view origins under test. Two give unaligned offsets (32 and 2080 at a
/// 256-byte pitch); the rest are aligned.
const ORIGINS: [(usize, usize); 6] = [(0, 0), (8, 8), (0, 8), (8, 0), (16, 0), (0, 1)];

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[path = "support/zero_copy.rs"]
#[allow(dead_code)]
mod zero_copy;
use outcome::report_skip as skip;

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

/// A DMA-BUF image, or `None` after a skip warning where the host has no usable heap.
fn dmabuf_image_or_skip(w: usize, h: usize, fmt: PixelFormat) -> Option<TensorDyn> {
    zero_copy::image(w, h, fmt, DType::U8)
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
        if let Err(e) = gl.convert(&view, &mut dst, None) {
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
                "ZEROS (the import sampled nothing)"
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
    // Skips the route assertions on a driver that feeds every source by copy.
    if imports == 0 && (uploads + pbo) as usize == ORIGINS.len() {
        skip("this GL driver cannot import DMA-BUF at all; the pixels above are the assertion");
        return;
    }
    assert_eq!(
        uploads + pbo,
        0,
        "{uploads} of {} source views were uploaded and {pbo} went by PBO, \
         not imported (the aligned-base rebase must keep them \
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

/// An RGBA frame at an aligned and an unaligned plane offset matches the CPU reference.
#[test]
fn rgba_offset_frames_convert_at_a_pitch_with_no_room_to_rebase() {
    let Some(mut gl) = gl_or_skip() else { return };
    // Three frames' worth of rows, later re-tagged to one frame's geometry at an offset.
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
    // A pattern with period 31, written before the re-tag while the map covers every row.
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
            .convert(&src, &mut reference, None)
            .expect("CPU reference");
        let want = bytes(&reference);

        let before = gl.convert_stats().expect("convert stats");
        let mut dst = rgba_dst(W, H);
        let outcome = gl.convert(&src, &mut dst, None);
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
                        "ZEROS (an unrebased offset was imported anyway)"
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

/// An NV12 DMA-BUF frame at an aligned and an unaligned plane offset matches the CPU reference.
#[test]
fn nv12_offset_sources_convert_through_gl_at_aligned_and_unaligned_offsets() {
    let Some(mut gl) = gl_or_skip() else { return };
    // Three frames' worth of rows, later re-tagged to one frame's geometry at an offset.
    let Some(mut src) = dmabuf_image_or_skip(W, H * 3, PixelFormat::Nv12) else {
        return;
    };
    let pitch = src.effective_row_stride().unwrap_or(W);
    // A pattern with period 31 inside 112..=142, written before the re-tag
    // while the map covers every row.
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
    // Tags the source full-range BT.601.
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
            .convert(&src, &mut reference, None)
            .expect("CPU reference");
        let want = bytes(&reference);
        let mut dst = rgba_dst(W, H);
        match gl.convert(&src, &mut dst, None) {
            Err(e) => failures.push(format!(
                "offset {offset}: GL refused the NV12 source: {e} \
                 (the declined import must reach the R8 upload)"
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
                    let flat = got.chunks(BPP).all(|p| p == &got[..BPP]);
                    failures.push(format!(
                        "offset {offset}: {bad} bytes differ from the CPU reference{}; \
                         first got={:?} want={:?}",
                        if flat {
                            " (FLAT: the import sampled a constant)"
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

/// A convert into an RGBA destination at a plane offset, aligned or not, lands its tile there.
#[test]
fn rgba_offset_destinations_place_their_tile_at_aligned_and_unaligned_offsets() {
    const BLANK: u8 = 0x55;
    let Some(mut gl) = gl_or_skip() else { return };
    let Some(canvas) = dmabuf_image_or_skip(W, H, PixelFormat::Rgba) else {
        return;
    };
    let pitch = canvas.effective_row_stride().unwrap_or(W * BPP);

    let mut failures = Vec::new();
    let mut routes: Vec<(usize, usize, usize, u64)> = Vec::new();
    // (0, 8) is 8 * pitch; (8, 8) adds 8 * BPP = 32, which is not 64-aligned.
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

        // The source carries the tile's absolute canvas colours.
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
        let before = gl
            .convert_stats()
            .expect("convert stats")
            .dst_import_fallbacks;
        if let Err(e) = gl.convert(&src, &mut dst, None) {
            failures.push(format!("({x0},{y0}) offset {offset}: convert failed: {e}"));
            continue;
        }
        // Destination-import fallbacks this convert took: 0 when the driver
        // imported the destination, 1 when the mapped-texture readback served it.
        let took = gl
            .convert_stats()
            .expect("convert stats")
            .dst_import_fallbacks
            - before;
        if took > 1 {
            failures.push(format!(
                "({x0},{y0}) offset {offset}: {took} destination-import fallbacks for \
                 one convert; at most one is expected"
            ));
        }
        if offset.is_multiple_of(MALI_ALIGN) && took != 0 {
            failures.push(format!(
                "({x0},{y0}) offset {offset}: the destination import was refused at an \
                 ALIGNED offset, so this driver lost the zero-copy destination path \
                 for a buffer it should accept"
            ));
        }
        routes.push((x0, y0, offset, took));
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
    eprintln!(
        "destination routes (x0, y0, offset, import fallbacks): {routes:?} \
         -- 1 means the driver refused the import and the mapped-texture \
         readback served the convert"
    );
    assert!(
        failures.is_empty(),
        "pitch {pitch}:\n{}",
        failures.join("\n")
    );
}
