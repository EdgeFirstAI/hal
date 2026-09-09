// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The GL draw places its frame at the destination surface's origin --
//! viewport 0,0 on an imported surface, buffer offset 0 in a readback -- so
//! a destination it cannot place that way is refused up front
//! (`check_draw_dst_placement`), and `ImageProcessor`'s CPU fallback handles
//! it instead. This pins the rebuilt-destination spelling of that refusal:
//! a destination reconstructed from a descriptor carries a plane offset but
//! no `view_origin`, so `check_draw_dst_placement` must refuse it rather
//! than let the zero-copy path draw at the canvas's top-left.
//!
//! Driven through `GLProcessorThreaded` directly so no CPU fallback can mask
//! a missing refusal, on the platforms whose zero-copy buffer needs no
//! device node.

#![cfg(all(
    any(target_os = "macos", target_os = "ios", target_os = "windows"),
    feature = "opengl"
))]

use edgefirst_image::{GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
};

const W: usize = 50;
const H: usize = 64;
const X0: usize = 8;
const Y0: usize = 8;
const SIDE: usize = 16;
const BLANK: u8 = 0x55;

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

#[test]
fn overlay_into_a_reconstructed_destination_view_is_refused_not_drawn_at_the_origin() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    // The macOS coverage lane runs pass 1 with the ANGLE dlopen gate closed;
    // the same guard `gl_backend_available_canary` carries.
    #[cfg(target_os = "macos")]
    if require_gl && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
        skip("ANGLE dlopen gate closed (coverage pass 1)");
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

    let canvas = match TensorDyn::image(
        W,
        H,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) if t.memory() == TensorMemory::DmaBuf => t,
        Ok(t) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the zero-copy request fell back to {:?}",
                t.memory()
            );
            skip(&format!("zero-copy request fell back to {:?}", t.memory()));
            return;
        }
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but no zero-copy buffer could be allocated: {e}"
            );
            skip(&format!("no zero-copy buffer here: {e}"));
            return;
        }
    };
    {
        let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
        m.as_mut_slice().fill(BLANK);
    }

    // A destination window rebuilt the way a capsule consumer rebuilds one:
    // the descriptor carries the window's shape and pitch, the offset is put
    // back afterwards, and there is no `view_origin`.
    let fresh = canvas
        .view(Region::new(X0, Y0, SIDE, SIDE))
        .expect("fresh view");
    let mut rebuilt = TensorDyn::import_descriptor(&fresh.descriptor_pinned(None))
        .expect("reconstruct the destination view from its descriptor");
    rebuilt.set_plane_offset(fresh.plane_offset().expect("a view carries its offset"));
    assert!(
        rebuilt.view_origin().is_none(),
        "precondition: a rebuilt view has no view_origin"
    );

    let err = gl
        .draw_decoded_masks(&mut rebuilt, &[], &[], Default::default())
        .expect_err(
            "a GL overlay into a destination it cannot place must be refused, not drawn at the origin",
        );
    assert!(
        matches!(err, edgefirst_image::Error::NotSupported(_)),
        "expected Error::NotSupported, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("sub-region destination"),
        "expected check_draw_dst_placement's refusal message, got: {msg}"
    );

    // Refused before any GL work touched the canvas: every byte is still
    // the poison fill, not just the window's.
    let out = canvas.map_bytes(CpuAccess::Read).expect("map canvas");
    assert!(
        out.as_slice().iter().all(|&b| b == BLANK),
        "the canvas was written to despite the draw being refused"
    );
}
