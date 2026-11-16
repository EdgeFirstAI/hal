// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A GL draw into a destination reconstructed from a descriptor (a plane
//! offset, no `view_origin`) is refused by `check_draw_dst_placement`.
//!
//! `GLProcessorThreaded` is driven directly.

#![cfg(all(
    any(target_os = "macos", target_os = "ios", target_os = "windows"),
    feature = "opengl"
))]

use edgefirst_image::{GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMapTrait};

const W: usize = 50;
const H: usize = 64;
const X0: usize = 8;
const Y0: usize = 8;
const SIDE: usize = 16;
const BLANK: u8 = 0x55;

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

#[test]
fn overlay_into_a_reconstructed_destination_view_is_refused_not_drawn_at_the_origin() {
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

    let Some(canvas) = zero_copy::image(W, H, PixelFormat::Rgba, DType::U8) else {
        return;
    };
    {
        let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
        m.as_mut_slice().fill(BLANK);
    }

    // A destination rebuilt from a view's descriptor, with the offset set and no `view_origin`.
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

    let out = canvas.map_bytes(CpuAccess::Read).expect("map canvas");
    assert!(
        out.as_slice().iter().all(|&b| b == BLANK),
        "the canvas was written to despite the draw being refused"
    );
}
