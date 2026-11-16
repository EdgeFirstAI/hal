// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A `Mem` NV12 source at a plane offset (a frame inside a larger buffer)
//! reaches the GL engine's NV R8 upload.
//!
//! A `GLProcessorThreaded` is driven directly.

#![cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]

use edgefirst_image::{CPUProcessor, GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory};

const W: usize = 32;
const H: usize = 16;
const N: usize = 3;
/// Flat luma per frame.
const LUMA: [u8; N] = [40, 200, 120];
const CHROMA: u8 = 128;

#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
use outcome::report_skip as skip;

fn rgba_dst() -> TensorDyn {
    TensorDyn::image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("destination alloc")
}

fn bytes(t: &TensorDyn) -> Vec<u8> {
    t.map_bytes(CpuAccess::Read)
        .expect("map destination")
        .as_slice()
        .to_vec()
}

#[test]
fn gl_uploads_an_offset_nv12_frame_from_its_own_offset_not_twice_it() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    #[cfg(target_os = "macos")]
    if require_gl && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
        skip("ANGLE dlopen gate closed");
        return;
    }
    let mut gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
            );
            skip(&format!("no GL backend: {e}"));
            return;
        }
    };

    // `N` NV12 frames back to back in one buffer, each a flat-luma frame
    // with neutral chroma. The tensor is given one frame's geometry and a
    // plane offset onto frame 1.
    let combined_h = H * 3 / 2;
    let elem = combined_h * W;
    let mut src = TensorDyn::new(&[N * elem], DType::U8, Some(TensorMemory::Mem), None)
        .expect("alloc N frames");
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map frames");
        let s = m.as_mut_slice();
        for (n, luma) in LUMA.iter().enumerate() {
            s[n * elem..n * elem + H * W].fill(*luma);
            s[n * elem + H * W..(n + 1) * elem].fill(CHROMA);
        }
    }
    src.set_logical_shape(&[combined_h, W])
        .expect("one frame's geometry");
    src.set_format(PixelFormat::Nv12).expect("nv12 format");
    src.set_plane_offset(elem);
    assert_eq!(src.plane_offset(), Some(elem), "precondition: frame 1");
    // Tags the source BT.601 full-range.
    src.set_colorimetry(Some(
        edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full),
    ));

    let mut reference = rgba_dst();
    CPUProcessor::new()
        .convert(&src, &mut reference, None)
        .expect("CPU reference convert");
    let want = bytes(&reference);
    assert!(
        want[0].abs_diff(LUMA[1]) <= 8,
        "the CPU reference read frame {} (luma {}) instead of frame 1 (luma {})",
        if want[0].abs_diff(LUMA[0]) <= 8 { 0 } else { 2 },
        want[0],
        LUMA[1]
    );

    let mut dst = rgba_dst();
    gl.convert(&src, &mut dst, None)
        .expect("GL convert of an offset NV12 frame");
    let got = bytes(&dst);

    const TOLERANCE: u8 = 8;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            g.abs_diff(*w) <= TOLERANCE,
            "byte {i}: GL wrote {g}, CPU reference {w}; the upload read the \
             wrong frame (plane offset applied twice on top of map())"
        );
    }
}
