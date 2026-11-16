// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A `PlanarRgb` F16 destination reconstructed from a descriptor (a plane
//! offset, no `view_origin`) is refused by the GL backend, and
//! `ImageProcessor`'s convert lands the bytes at that offset through the CPU
//! fallback.
//!
//! macOS/iOS only.

#![cfg(all(any(target_os = "macos", target_os = "ios"), feature = "opengl"))]

use edgefirst_image::{GLProcessorThreaded, ImageProcessor, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory};

const W: usize = 64;
const H: usize = 16;
const BLANK: u8 = 0x55;
const PIXEL: [u8; 4] = [200, 100, 50, 255];

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
fn reconstructed_planar_f16_destination_receives_its_convert_at_its_offset() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    #[cfg(target_os = "macos")]
    if require_gl && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
        skip("ANGLE dlopen gate closed");
        return;
    }

    // A processor used only to query `supported_render_dtypes`, dropped before the converts below.
    let support = {
        let proc = ImageProcessor::new().expect("ImageProcessor::new");
        proc.supported_render_dtypes()
    };
    if !support.f16 {
        assert!(
            !require_gl,
            "HAL_TEST_REQUIRE_GL=1 but f16 render is not supported: {support:?}"
        );
        skip(&format!("f16 render not supported: {support:?}"));
        return;
    }

    let Some(canvas) = zero_copy::image(W, H, PixelFormat::PlanarRgb, DType::F16) else {
        return;
    };

    // The RGBA16F packing stores the planar `[3, H, W]` stream at `(W/4, 3H)`
    // texels of 4 `f16`s each, so one planar row is one surface row.
    let pitch = canvas
        .effective_row_stride()
        .expect("planar f16 surface pitch");
    assert!(
        pitch >= 128 && pitch % 64 == 0,
        "pitch {pitch} is not a plausible planar f16 row stride for width {W}"
    );
    {
        let mut m = canvas.map_bytes(CpuAccess::Write).expect("map canvas");
        m.as_mut_slice().fill(BLANK);
    }

    // A destination rebuilt from the whole surface's descriptor, narrowed to
    // the 15 rows past the first, with the offset set.
    let mut rebuilt = TensorDyn::import_descriptor(&canvas.descriptor_pinned(None))
        .expect("reconstruct the destination from its descriptor");
    rebuilt
        .set_logical_shape(&[3, H - 1, W])
        .expect("narrow to the 15 rows past the first");
    rebuilt.set_plane_offset(pitch);
    assert!(
        rebuilt.view_origin().is_none(),
        "precondition: a rebuilt tensor has no view_origin"
    );
    assert_eq!(
        rebuilt.effective_row_stride(),
        Some(pitch),
        "the narrowed tensor keeps the parent surface's pitch"
    );

    let src = TensorDyn::image(
        W,
        H - 1,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc source");
    {
        let mut m = src.map_bytes(CpuAccess::Write).expect("map source");
        let s = m.as_mut_slice();
        for i in (0..s.len()).step_by(4) {
            s[i..i + 4].copy_from_slice(&PIXEL);
        }
    }

    // First half: the GL backend refuses this destination. The GL context is
    // dropped at the end of the scope.
    {
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
        let err = gl
            .convert(&src, &mut rebuilt, None)
            .expect_err("the GL float path must refuse a destination it cannot place");
        assert!(
            matches!(err, edgefirst_image::Error::NotSupported(_)),
            "expected Error::NotSupported, got {err:?}"
        );
        let msg = err.to_string();
        // Only the message's prefix is asserted.
        assert!(
            msg.contains("GL float destination at plane offset"),
            "expected the float dispatch gate's refusal message, got: {msg}"
        );
    }

    // Second half: `ImageProcessor` converts through the CPU fallback.
    let mut proc = ImageProcessor::new().expect("ImageProcessor::new");
    proc.convert(&src, &mut rebuilt, None)
        .expect("convert into the reconstructed planar destination");

    let out = canvas.map_bytes(CpuAccess::Read).expect("map canvas");
    let bytes = out.as_slice();
    assert!(
        bytes[0..pitch].iter().all(|&b| b == BLANK),
        "row 0 of the canvas -- the byte before the reconstructed tensor's \
         offset -- must stay untouched: a render at the buffer's origin \
         would write here instead"
    );
    assert!(
        !bytes[pitch..2 * pitch].iter().all(|&b| b == BLANK),
        "row 1 of the canvas -- the reconstructed tensor's own row 0, at \
         its plane offset -- must have received the convert"
    );

    // Row 1's first f16 is the R plane's first packed texel, channel 0.
    let actual = half::f16::from_le_bytes([bytes[pitch], bytes[pitch + 1]]);
    let expected = half::f16::from_f32(PIXEL[0] as f32 / 255.0);
    assert!(
        actual.to_bits().abs_diff(expected.to_bits()) <= 32,
        "row 1's first f16 is {actual:?} ({:#06x}), expected near {expected:?} ({:#06x})",
        actual.to_bits(),
        expected.to_bits()
    );
}
