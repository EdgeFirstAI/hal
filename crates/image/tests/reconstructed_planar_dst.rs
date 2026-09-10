// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The zero-copy float render targets (`convert_float_to_zero_copy`,
//! `convert_nv_to_float_two_pass`) import their destination through
//! `get_or_create_egl_image_rgb` -> `import_buffer_packed`, which on ANGLE
//! over IOSurface binds the whole buffer from its origin -- the same
//! problem `bind_dst` and `convert_via_engine` already gate on
//! `dst_import_places` for their own destination imports.
//!
//! This pins the float dispatch's own gate: a `PlanarRgb` F16 destination
//! reconstructed from a descriptor (a plane offset, no `view_origin`) must
//! be refused by the GL backend directly, and `ImageProcessor`'s convert
//! (CPU fallback) must still land the bytes at that offset, not at the
//! buffer's origin.
//!
//! macOS/iOS only: a D3D11 descriptor import accepts only packed windows, so
//! a narrowed planar destination cannot exist on Windows, and Linux's
//! DMA-BUF import expresses the offset itself, so `dst_import_places`
//! defaults to `true` there and this reconstruction hits the ordinary
//! zero-copy path.
//!
//! Two halves, so the gate cannot pass vacuously: first `GLProcessorThreaded`
//! is driven directly and asserted to *refuse* the destination (the gate
//! actually firing, not just "some path happened to place the bytes
//! correctly"); the GL context is dropped before `ImageProcessor` opens its
//! own, since nothing here establishes that two live ANGLE contexts coexist
//! safely, and only one is ever needed at a time. Then `ImageProcessor`
//! drives the same convert, which falls back to CPU on the same refusal, and
//! its placement is checked.
//!
//! The decoded `f16` value check is intentionally coarse (32 ULPs): native
//! FP16 widening on Apple silicon and an `f32`-divide-then-round golden can
//! disagree by more than a couple of bits, and the untouched/written
//! byte-range checks above it are the assertions this file actually exists
//! to make.

#![cfg(all(any(target_os = "macos", target_os = "ios"), feature = "opengl"))]

use edgefirst_image::{
    Crop, Flip, GLProcessorThreaded, ImageProcessor, ImageProcessorTrait, Rotation,
};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory};

const W: usize = 64;
const H: usize = 16;
const BLANK: u8 = 0x55;
const PIXEL: [u8; 4] = [200, 100, 50, 255];

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

#[test]
fn reconstructed_planar_f16_destination_receives_its_convert_at_its_offset() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    // The macOS coverage lane runs pass 1 with the ANGLE dlopen gate closed;
    // the same guard `gl_backend_available_canary` carries.
    #[cfg(target_os = "macos")]
    if require_gl && std::env::var_os("HAL_TEST_ALLOW_DLOPEN_ANGLE").is_none() {
        skip("ANGLE dlopen gate closed (coverage pass 1)");
        return;
    }

    // `supported_render_dtypes` is only public on `ImageProcessor`; this
    // instance exists only to ask the question and is dropped immediately
    // after, before either of the two convert halves below opens its own
    // GL context.
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

    let canvas = match TensorDyn::image(
        W,
        H,
        PixelFormat::PlanarRgb,
        DType::F16,
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

    // The RGBA16F packing stores the planar `[3, H, W]` stream at `(W/4, 3H)`
    // texels of 4 `f16`s each, so one planar row is one surface row. The
    // exact byte count is a platform packing/alignment detail, not something
    // this test should pin: only that it is at least 64 `f16`s wide (`W`
    // samples, 2 bytes each) and a multiple of the packing's 4-wide texel
    // (`4 * 2 * 8` bytes) in that unit.
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

    // A destination window rebuilt the way a capsule consumer rebuilds one:
    // the descriptor carries the whole surface, then it is narrowed to the
    // 15 rows past the first (skipping row 0) and the offset put back --
    // `PlanarRgb`/`PlanarRgba` reject `Tensor::view()`, so a real capsule
    // consumer narrows this way rather than through a view.
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

    // First half: the GL backend itself must refuse this destination, not
    // merely happen to place the bytes correctly. Scoped so the GL context
    // is dropped before `ImageProcessor` opens its own below.
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
            .convert(
                &src,
                &mut rebuilt,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )
            .expect_err("the GL float path must refuse a destination it cannot place");
        assert!(
            matches!(err, edgefirst_image::Error::NotSupported(_)),
            "expected Error::NotSupported, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("has no view origin to place it by"),
            "expected the float dispatch gate's refusal message, got: {msg}"
        );
    }

    // Second half: `ImageProcessor`'s CPU fallback on that same refusal must
    // still land the convert at the reconstructed tensor's own offset.
    let mut proc = ImageProcessor::new().expect("ImageProcessor::new");
    proc.convert(
        &src,
        &mut rebuilt,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
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

    // Row 1's first f16 is the R plane's first packed texel, channel 0: the
    // source is one constant RGBA pixel, so it must be near R/255 regardless
    // of which of the 4 packed positions it is. Coarse on purpose -- see the
    // module doc comment.
    let actual = half::f16::from_le_bytes([bytes[pitch], bytes[pitch + 1]]);
    let expected = half::f16::from_f32(PIXEL[0] as f32 / 255.0);
    assert!(
        actual.to_bits().abs_diff(expected.to_bits()) <= 32,
        "row 1's first f16 is {actual:?} ({:#06x}), expected near {expected:?} ({:#06x})",
        actual.to_bits(),
        expected.to_bits()
    );
}
