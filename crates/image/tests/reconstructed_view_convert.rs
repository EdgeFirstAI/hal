// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `ImageProcessor::convert` of a tensor that carries a plane offset reads,
//! or writes, the region the tensor names and not its buffer's origin.
//!
//! The cases:
//!
//! * **A**: a fresh `view()`.
//! * **B**: a view rebuilt from its descriptor (`import_descriptor`) with its
//!   offset set back.
//! * **C**: a whole image at a foreign offset.
//! * **D**: a one-row view rebuilt from its descriptor.
//! * **E**: a destination rebuilt from a descriptor, which has the offset but
//!   no `view_origin`.
//!
//! The source is an image-formatted `TensorMemory::DmaBuf` surface: DMA-BUF on
//! Linux, IOSurface on macOS, a D3D11 texture on Windows. A runs on all three;
//! B–E run on Apple and Windows.

use edgefirst_image::{ImageProcessor, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};

const W: usize = 50;
const H: usize = 64;
const X0: usize = 8;
const Y0: usize = 8;
const SIDE: usize = 16;
const BPP: usize = 4;

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[path = "support/zero_copy.rs"]
#[allow(dead_code)]
mod zero_copy;

/// Distinct per-row and per-column values.
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

/// A `W`x`H` RGBA image on the platform's zero-copy buffer, or `None` after a skip warning.
fn zero_copy_image() -> Option<TensorDyn> {
    zero_copy::image(W, H, PixelFormat::Rgba, DType::U8)
}

/// Assert the convert landed on the requested tile; the failure says whether
/// it read the parent's origin instead.
fn assert_tile(label: &str, out: &[u8]) {
    let expected = tile(X0, Y0);
    if out == expected.as_slice() {
        return;
    }
    let origin = tile(0, 0);
    if out == origin.as_slice() {
        panic!(
            "{label}: convert() read the parent buffer's ORIGIN, i.e. the \
             plane offset was lost"
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

    // A: a fresh view.
    let fresh = src
        .view(Region::new(X0, Y0, SIDE, SIDE))
        .expect("fresh view");
    let mut dst_a = dst();
    proc.convert(&fresh, &mut dst_a, None)
        .expect("convert fresh view");
    assert_tile("fresh view", &read_all(&dst_a));

    // B–E: rebuilt through the descriptor.
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios"))]
    {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_ne!(
            pitch,
            W * BPP,
            "precondition: the surface pitch must be padded past the natural row"
        );

        // B: the view rebuilt from its descriptor, with its offset set back.
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
        proc.convert(&rebuilt, &mut dst_b, None)
            .expect("convert reconstructed view");
        assert_tile("reconstructed view", &read_all(&dst_b));

        // C: the whole image narrowed by one row and offset by one row.
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
        proc.convert(&whole, &mut dst_c, None)
            .expect("convert whole image at a foreign offset");
        let out = read_all(&dst_c);
        assert_eq!(
            &out[..BPP],
            &want(0, 1),
            "a whole image at a one-row offset must start at parent row 1, \
             not row 0"
        );

        // D: a one-row view rebuilt from its descriptor.
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
        proc.convert(&rebuilt_row, &mut dst_d, None)
            .expect("convert one-row reconstructed view");
        let expected_row: Vec<u8> = (X0..X0 + SIDE).flat_map(|x| want(x, Y0)).collect();
        assert_eq!(
            read_all(&dst_d),
            expected_row,
            "a one-row reconstructed view must read parent row {Y0} from \
             column {X0}"
        );

        // E: a destination rebuilt from its descriptor, with the offset and no `view_origin`.
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
        proc.convert(&fresh, &mut rebuilt_dst, None)
            .expect("convert into a reconstructed destination view");
        let out = read_all(&canvas);
        let px = |x: usize, y: usize| &out[y * canvas_pitch + x * BPP..][..BPP];
        assert_eq!(
            px(0, 0),
            &[BLANK; BPP],
            "a reconstructed destination view wrote the canvas's ORIGIN, i.e. \
             the plane offset was lost"
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
