// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Test-only helpers: skip reporting and the allocation policy live in the
//! shared `outcome` file; this module adds the DMA and G2D probes.

#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;
#[allow(unused_imports)]
pub(crate) use outcome::{allocated, current_test, report_skip, Expect};

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;

/// [`edgefirst_tensor::is_dma_available`] under the `HAL_TEST_REQUIRE_DMA=1`
/// policy: on a lane that requires a DMA heap, an unavailable one fails the
/// calling test instead of letting it skip while reporting success.
pub(crate) fn is_dma_available() -> bool {
    dma_require::decide(
        edgefirst_tensor::is_dma_available(),
        dma_require::required(),
        &current_test(),
        "no usable DMA heap (no /dev/dma_heap node, or it is not openable by this user)",
    )
}

/// [`edgefirst_tensor::is_gpu_buffer_available`], under the `HAL_TEST_REQUIRE_DMA=1` policy on Linux
/// where the buffers are DMA-BUFs.
#[cfg_attr(not(feature = "opengl"), allow(dead_code))]
pub(crate) fn is_gpu_buffer_available() -> bool {
    #[cfg(target_os = "linux")]
    return is_dma_available();
    #[cfg(not(target_os = "linux"))]
    edgefirst_tensor::is_gpu_buffer_available()
}

/// Unwraps a DMA-BUF allocation: it must succeed when the heap probe (under
/// `HAL_TEST_REQUIRE_DMA`) says the heap is usable, and may fail with a warning otherwise.
pub(crate) fn dma_allocated<T, E: std::fmt::Debug>(result: Result<T, E>, what: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => allocated(Err(e), what, outcome::expect_dma(is_dma_available())),
    }
}

/// Unwraps a zero-copy GPU buffer allocation (DMA-BUF, IOSurface or D3D11): it must succeed when
/// [`is_gpu_buffer_available`] reports the backing, and may fail with a warning otherwise.
#[cfg_attr(not(feature = "opengl"), allow(dead_code))]
pub(crate) fn gpu_buffer_allocated<T, E: std::fmt::Debug>(
    result: Result<T, E>,
    what: &str,
) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            let expect = if is_gpu_buffer_available() {
                Expect::Succeeds
            } else {
                Expect::MayFail("no zero-copy GPU buffer backing on this host")
            };
            allocated(Err(e), what, expect)
        }
    }
}

/// Applies the `HAL_TEST_REQUIRE_G2D=1` policy to a G2D availability probe.
///
/// # Panics
///
/// When `HAL_TEST_REQUIRE_G2D=1` and `available` is `false`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn g2d_required_or(available: bool) -> bool {
    let required = std::env::var("HAL_TEST_REQUIRE_G2D").as_deref() == Ok("1");
    g2d_decide(available, required, &current_test())
}

fn g2d_decide(available: bool, required: bool, what: &str) -> bool {
    assert!(
        available || !required,
        "HAL_TEST_REQUIRE_G2D=1 but {what} would have skipped while reporting success: \
         libg2d.so.2 did not load or G2D failed to initialize"
    );
    available
}

#[test]
fn g2d_policy_skips_unless_required() {
    assert!(g2d_decide(true, true, "t"));
    assert!(g2d_decide(true, false, "t"));
    assert!(!g2d_decide(false, false, "t"));
    let required_but_missing = std::panic::catch_unwind(|| g2d_decide(false, true, "t"));
    assert!(required_but_missing.is_err());
}

/// Background of the box-outline fixture; its alpha is not 255 so a test
/// can tell whether a draw wrote the destination's alpha channel.
pub(crate) const BOX_BG: [u8; 4] = [10, 20, 30, 200];
/// Side of the square box-outline fixture.
pub(crate) const BOX_DIM: usize = 32;

/// Draws one box over [`BOX_BG`] with `processor` using `color`, `width`
/// (`None` keeps the processor's) and `opacity`, and returns the RGBA
/// pixels. The box is the middle half of
/// a [`BOX_DIM`]-square image, so its outline sits well inside the border.
pub(crate) fn draw_box_fixture(
    processor: &mut dyn crate::ImageProcessorTrait,
    color: [u8; 4],
    width: Option<u32>,
    opacity: f32,
) -> Vec<u8> {
    use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};

    let image = || {
        TensorDyn::image(
            BOX_DIM,
            BOX_DIM,
            PixelFormat::Rgba,
            DType::U8,
            None,
            CpuAccess::ReadWrite,
        )
        .unwrap()
    };
    let bg = image();
    {
        let bg_u8 = bg.as_typed::<u8>().unwrap();
        let mut map = bg_u8.map_mut().unwrap();
        for px in map.as_mut_slice().chunks_exact_mut(4) {
            px.copy_from_slice(&BOX_BG);
        }
    }
    let mut dst = image();
    processor.set_class_colors(&[color]).unwrap();
    if let Some(width) = width {
        processor.set_box_line_width(width).unwrap();
    }
    let detect = edgefirst_tensor::DetectBox {
        bbox: [0.25, 0.25, 0.75, 0.75].into(),
        score: 0.9,
        label: 0,
    };
    processor
        .draw_decoded_masks(
            &mut dst,
            &[detect],
            &[],
            crate::MaskOverlay::new()
                .with_background(&bg)
                .with_opacity(opacity),
        )
        .unwrap();
    let dst_u8 = dst.as_typed::<u8>().unwrap();
    let mut flat = vec![0u8; BOX_DIM * BOX_DIM * 4];
    dst_u8.copy_to_flat(&mut flat).unwrap();
    flat
}

/// The fixture's pixel at `(x, y)`.
pub(crate) fn box_px(flat: &[u8], x: usize, y: usize) -> [u8; 4] {
    let o = (y * BOX_DIM + x) * 4;
    flat[o..o + 4].try_into().unwrap()
}

/// The x coordinates on row `y` whose pixel differs from [`BOX_BG`].
pub(crate) fn box_drawn_columns(flat: &[u8], y: usize) -> Vec<usize> {
    (0..BOX_DIM)
        .filter(|&x| box_px(flat, x, y) != BOX_BG)
        .collect()
}

/// The y coordinates in column `x` whose pixel differs from [`BOX_BG`].
pub(crate) fn box_drawn_rows(flat: &[u8], x: usize) -> Vec<usize> {
    (0..BOX_DIM)
        .filter(|&y| box_px(flat, x, y) != BOX_BG)
        .collect()
}

/// `color` blended over [`BOX_BG`] with coverage `alpha / 255`, keeping the
/// background's alpha.
pub(crate) fn box_blend(color: [u8; 4], alpha: u16) -> [u8; 4] {
    let mix = |c: u8, d: u8| ((c as u16 * alpha + d as u16 * (255 - alpha)) / 255) as u8;
    [
        mix(color[0], BOX_BG[0]),
        mix(color[1], BOX_BG[1]),
        mix(color[2], BOX_BG[2]),
        BOX_BG[3],
    ]
}

/// The columns (on a middle row) or rows (in a middle column) a `width`-pixel
/// outline of the fixture's box covers: `width` pixels outside each edge,
/// clipped at the image border.
pub(crate) fn box_expected_span(width: usize) -> Vec<usize> {
    let (lo, hi) = (BOX_DIM / 4, BOX_DIM * 3 / 4);
    (lo.saturating_sub(width)..lo)
        .chain(hi..(hi + width).min(BOX_DIM))
        .collect()
}
