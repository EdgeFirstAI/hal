// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
#![allow(unused_mut)]

//! End-to-end odd-dimension CPU conversion tests.
//!
//! Covers the representation assertion helper and reuse cell, CPU convert
//! cells for odd-W/H JPEG fixtures and analytically-constructed odd-both
//! sources, and the resize strided-source path.

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::{CPUProcessor, Crop, ImageProcessorTrait};
use edgefirst_tensor::{
    DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;
#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;

/// A padded-stride image, or `None` after a warning where it may fail to
/// allocate.
fn strided_image(w: usize, h: usize, fmt: PixelFormat, stride: usize) -> Option<Tensor<u8>> {
    let expect = if cfg!(target_os = "linux") {
        outcome::expect_dma(dma_require::decide(
            edgefirst_tensor::is_dma_available(),
            dma_require::required(),
            &outcome::current_test(),
            "no usable DMA heap for a padded-stride image",
        ))
    } else {
        outcome::Expect::MayFail("image_with_stride needs DMA-BUF, which only Linux has")
    };
    outcome::allocated(
        Tensor::<u8>::image_with_stride(
            w,
            h,
            fmt,
            stride,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        ),
        &format!("{w}x{h} {fmt:?} image at a {stride} B stride"),
        expect,
    )
}

// ---------------------------------------------------------------------------
// Test-data helper
// ---------------------------------------------------------------------------

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata")
        });
    let path = root.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Representation assertion helper
// ---------------------------------------------------------------------------

/// Assert the representation invariants for a semi-planar / grey tensor.
///
/// * logical `width()` and `height()` match `logical_w` / `logical_h`.
/// * `format()` matches `fmt`.
/// * `effective_row_stride()` is 64-byte aligned and >= even(logical_w).
/// * The mapped buffer is large enough to hold all planes at that stride.
fn assert_allocation_geometry(
    tensor: &Tensor<u8>,
    logical_w: usize,
    logical_h: usize,
    fmt: PixelFormat,
) {
    assert_eq!(
        tensor.width(),
        Some(logical_w),
        "logical width mismatch for {fmt:?}"
    );
    assert_eq!(
        tensor.height(),
        Some(logical_h),
        "logical height mismatch for {fmt:?}"
    );
    assert_eq!(tensor.format(), Some(fmt), "format mismatch");

    let s = tensor
        .effective_row_stride()
        .expect("semi-planar/grey tensor must have a stride");
    assert_eq!(s % 64, 0, "{fmt:?}: stride {s} is not 64-byte aligned");
    assert!(
        s >= logical_w.next_multiple_of(2),
        "{fmt:?}: stride {s} < even(width)={}",
        logical_w.next_multiple_of(2)
    );

    // `as_slice().len()` is the physical buffer capacity.
    let map = tensor.map().unwrap();
    let buf_len = map.as_slice().len();
    let total_rows = match fmt {
        PixelFormat::Nv12 => logical_h + logical_h.div_ceil(2),
        PixelFormat::Nv16 => logical_h * 2,
        PixelFormat::Nv24 => logical_h * 3,
        PixelFormat::Grey => logical_h,
        _ => logical_h,
    };
    assert!(
        buf_len >= s * total_rows,
        "{fmt:?}: buf_len={buf_len} < stride*total_rows={s}*{total_rows}={}",
        s * total_rows
    );
}

// ---------------------------------------------------------------------------
// Representation assertion cells for each fixture
// ---------------------------------------------------------------------------

#[test]
fn nv12_odd_w_allocation_geometry() {
    let jpeg = testdata("jaguar.jpg");
    let mut t = Tensor::<u8>::image(
        789,
        384,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_allocation_geometry(&t, 789, 384, PixelFormat::Nv12);

    let map = t.map().unwrap();
    let s = t.effective_row_stride().unwrap();
    let nonzero = map.as_slice()[..s * 384]
        .iter()
        .filter(|&&v| v != 0)
        .count();
    assert!(
        nonzero > 1000,
        "Y plane should be non-trivial, got {nonzero} nonzero bytes"
    );
}

#[test]
fn nv16_odd_w_allocation_geometry() {
    let jpeg = testdata("jaguar_422.jpg");
    let mut t = Tensor::<u8>::image(
        789,
        384,
        PixelFormat::Nv16,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_allocation_geometry(&t, 789, 384, PixelFormat::Nv16);
}

#[test]
fn nv24_odd_w_allocation_geometry() {
    let jpeg = testdata("jaguar_444.jpg");
    let mut t = Tensor::<u8>::image(
        789,
        384,
        PixelFormat::Nv24,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_allocation_geometry(&t, 789, 384, PixelFormat::Nv24);
}

#[test]
fn grey_odd_h_allocation_geometry() {
    let jpeg = testdata("grey.jpg");
    let mut t = Tensor::<u8>::image(
        1024,
        681,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_allocation_geometry(&t, 1024, 681, PixelFormat::Grey);
}

#[test]
fn nv12_odd_h_allocation_geometry() {
    let jpeg = testdata("grey-rgb.jpg");
    let mut t = Tensor::<u8>::image(
        1024,
        681,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_allocation_geometry(&t, 1024, 681, PixelFormat::Nv12);
}

// ---------------------------------------------------------------------------
// Pre-alloc-max + decode-small reuse cell
// ---------------------------------------------------------------------------

#[test]
fn max_alloc_tensor_is_reused_for_a_small_decode() {
    let mut pool = Tensor::<u8>::image(
        1920,
        1088,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let large_stride = pool.effective_row_stride().unwrap();
    assert_eq!(large_stride % 64, 0);
    assert!(large_stride >= 1920);

    let jpeg = testdata("jaguar.jpg");
    let mut dec = ImageDecoder::new();
    pool.load_image(&mut dec, &jpeg).unwrap();

    assert_eq!(pool.width(), Some(789), "logical width must update to 789");
    assert_eq!(
        pool.height(),
        Some(384),
        "logical height must update to 384"
    );

    let post_stride = pool.effective_row_stride().unwrap();
    assert_eq!(
        post_stride, large_stride,
        "stride must stay at pre-alloc pitch {large_stride}, got {post_stride}"
    );

    let map = pool.map().unwrap();
    let luma_nonzero = (0..384usize)
        .flat_map(|r| {
            map.as_slice()[r * post_stride..r * post_stride + 789]
                .iter()
                .filter(|&&v| v != 0)
        })
        .count();
    assert!(
        luma_nonzero > 1000,
        "decoded luma should be non-trivial, got {luma_nonzero} non-zero bytes"
    );
}

// ---------------------------------------------------------------------------
// Helpers for the JPEG-based convert cells
// ---------------------------------------------------------------------------

/// MAE and max_diff between our CPU-converted RGB and the `image` crate's RGB.
/// Both are indexed at their respective effective_row_stride.
fn mae_vs_image_crate_rgb(
    our_rgb: &Tensor<u8>,
    ref_rgb: &image::RgbImage,
    w: usize,
    h: usize,
) -> (f64, u32) {
    let dst_stride = our_rgb.effective_row_stride().unwrap_or(w * 3);
    let map = our_rgb.map().unwrap();
    let our = map.as_slice();
    let mut total: u64 = 0;
    let mut max_diff: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let oi = y * dst_stride + x * 3;
            let ref_p = ref_rgb.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                let d = (our[oi + c] as i32 - ref_p[c] as i32).unsigned_abs();
                total += d as u64;
                max_diff = max_diff.max(d);
            }
        }
    }
    let mae = total as f64 / (w * h * 3) as f64;
    (mae, max_diff)
}

/// Decode JPEG→NV tensor, convert to RGB, return MAE vs image-crate reference.
fn jpeg_nv_to_rgb_mae(fixture: &str, fmt: PixelFormat, w: usize, h: usize) -> (f64, u32) {
    let jpeg = testdata(fixture);
    let ref_rgb = image::load_from_memory(&jpeg).unwrap().to_rgb8();
    assert_eq!(ref_rgb.dimensions(), (w as u32, h as u32));

    let mut src = Tensor::<u8>::image(
        w,
        h,
        fmt,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();
    assert_eq!(src.width(), Some(w));
    assert_eq!(src.height(), Some(h));

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_dyn, &mut dst_dyn, None).unwrap();
    let dst_u8 = dst_dyn.as_u8().unwrap();
    mae_vs_image_crate_rgb(dst_u8, &ref_rgb, w, h)
}

// ---------------------------------------------------------------------------
// NV12 odd-W
// ---------------------------------------------------------------------------

#[test]
fn nv12_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar.jpg", PixelFormat::Nv12, 789, 384);
    eprintln!("NV12 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(mae < 3.0, "NV12 odd-W MAE {mae:.3} exceeds tolerance 3.0");
}

// ---------------------------------------------------------------------------
// NV16 odd-W
// ---------------------------------------------------------------------------

#[test]
fn nv16_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar_422.jpg", PixelFormat::Nv16, 789, 384);
    eprintln!("NV16 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(mae < 4.0, "NV16 odd-W MAE {mae:.3} exceeds tolerance 4.0");
}

// ---------------------------------------------------------------------------
// NV24 odd-W
// ---------------------------------------------------------------------------

#[test]
fn nv24_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar_444.jpg", PixelFormat::Nv24, 789, 384);
    eprintln!("NV24 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(mae < 4.0, "NV24 odd-W MAE {mae:.3} exceeds tolerance 4.0");
}

// ---------------------------------------------------------------------------
// GREY odd-H
// ---------------------------------------------------------------------------

#[test]
fn grey_odd_h_to_rgb() {
    let jpeg = testdata("grey.jpg");
    let w = 1024usize;
    let h = 681usize;
    let ref_luma = image::load_from_memory(&jpeg).unwrap().to_luma8();
    assert_eq!(ref_luma.dimensions(), (w as u32, h as u32));

    let mut src = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_dyn, &mut dst_dyn, None).unwrap();

    let dst_u8 = dst_dyn.as_u8().unwrap();
    let dst_stride = dst_u8.effective_row_stride().unwrap_or(w * 3);
    let map = dst_u8.map().unwrap();
    let our = map.as_slice();

    // Grey→RGB gives R=G=B=Y; compares against the luma reference.
    let mut total: u64 = 0;
    let mut max_diff: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let oi = y * dst_stride + x * 3;
            let ref_y = ref_luma.get_pixel(x as u32, y as u32)[0] as i32;
            for c in 0..3usize {
                let d = (our[oi + c] as i32 - ref_y).unsigned_abs();
                total += d as u64;
                max_diff = max_diff.max(d);
            }
        }
    }
    let mae = total as f64 / (w * h * 3) as f64;
    eprintln!("GREY odd-H: MAE={mae:.3}, max_diff={max_diff}");
    assert!(mae < 3.0, "GREY odd-H MAE {mae:.3} exceeds tolerance 3.0");
}

// ---------------------------------------------------------------------------
// NV12 odd-H
// ---------------------------------------------------------------------------

#[test]
fn nv12_odd_h_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("grey-rgb.jpg", PixelFormat::Nv12, 1024, 681);
    eprintln!("NV12 odd-H: MAE={mae:.3}, max_diff={max_diff}");
    assert!(mae < 3.0, "NV12 odd-H MAE {mae:.3} exceeds tolerance 3.0");
}

// ---------------------------------------------------------------------------
// NV16 odd-H (analytic, even-W 64×63)
// ---------------------------------------------------------------------------

#[test]
fn nv16_odd_h_to_rgb() {
    run_odd_both(64, 63, PixelFormat::Nv16, "NV16 64x63 odd-H");
}

// ---------------------------------------------------------------------------
// NV24 odd-H (analytic, even-W 64×63)
// ---------------------------------------------------------------------------

#[test]
fn nv24_odd_h_to_rgb() {
    run_odd_both(64, 63, PixelFormat::Nv24, "NV24 64x63 odd-H");
}

// ---------------------------------------------------------------------------
// Degenerate mini-cases (1×1, 1×3, 3×1) for NV12/NV16/NV24
// ---------------------------------------------------------------------------

#[test]
fn degenerate_nv12_1x1() {
    run_odd_both(1, 1, PixelFormat::Nv12, "degenerate NV12 1x1");
}

#[test]
fn degenerate_nv12_1x3() {
    run_odd_both(1, 3, PixelFormat::Nv12, "degenerate NV12 1x3");
}

#[test]
fn degenerate_nv12_3x1() {
    run_odd_both(3, 1, PixelFormat::Nv12, "degenerate NV12 3x1");
}

#[test]
fn degenerate_nv16_1x1() {
    run_odd_both(1, 1, PixelFormat::Nv16, "degenerate NV16 1x1");
}

#[test]
fn degenerate_nv16_1x3() {
    run_odd_both(1, 3, PixelFormat::Nv16, "degenerate NV16 1x3");
}

#[test]
fn degenerate_nv16_3x1() {
    run_odd_both(3, 1, PixelFormat::Nv16, "degenerate NV16 3x1");
}

#[test]
fn degenerate_nv24_1x1() {
    run_odd_both(1, 1, PixelFormat::Nv24, "degenerate NV24 1x1");
}

#[test]
fn degenerate_nv24_1x3() {
    run_odd_both(1, 3, PixelFormat::Nv24, "degenerate NV24 1x3");
}

#[test]
fn degenerate_nv24_3x1() {
    run_odd_both(3, 1, PixelFormat::Nv24, "degenerate NV24 3x1");
}

// ---------------------------------------------------------------------------
// Analytically-constructed odd-both sources
// ---------------------------------------------------------------------------

/// BT.601 full-range YCbCr→RGB.
fn bt601_ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = y as f64;
    let cb = cb as f64 - 128.0;
    let cr = cr as f64 - 128.0;
    let r = (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
    let g = (y - 0.344136 * cb - 0.714136 * cr)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8;
    [r, g, b]
}

/// Build a strided NV12/NV16/NV24 Tensor<u8> with a synthetic pattern that
/// varies in both X and Y, and return the analytically-expected RGB output.
///
/// Pattern: `Y(r,c) = (r*3 + c*5) % 256`
/// Chroma:  `Cb(cc,cr) = (cc*7 + cr*11 + 40) % 256`
///          `Cr(cc,cr) = (cc*13 + cr*3 + 80) % 256`
/// where `cc`/`cr` are chroma-cell coordinates (divided by subsampling factor).
fn make_odd_both_source(w: usize, h: usize, fmt: PixelFormat) -> (Tensor<u8>, Vec<[u8; 3]>) {
    let mut src = Tensor::<u8>::image(
        w,
        h,
        fmt,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let stride = src.effective_row_stride().unwrap();

    let (cw_shift, ch_shift) = match fmt {
        PixelFormat::Nv12 => (1usize, 1usize),
        PixelFormat::Nv16 => (1, 0),
        PixelFormat::Nv24 => (0, 0),
        _ => panic!("unsupported format {fmt:?}"),
    };
    // Chroma rows and columns: ceil(dim / divisor).
    let chroma_h = h.div_ceil(1 << ch_shift);
    let chroma_w = w.div_ceil(1 << cw_shift);

    {
        let mut map = src.map().unwrap();
        let buf = map.as_mut_slice();
        let uv_start = stride * h;

        // Fill luma
        for r in 0..h {
            for c in 0..w {
                buf[r * stride + c] = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        // Fill chroma (interleaved [Cb, Cr] per chroma column). The NV12/NV16
        // UV row pitch is `stride`; the NV24 UV row pitch is `stride * 2`.
        let uv_row_stride = match fmt {
            PixelFormat::Nv24 => stride * 2,
            _ => stride,
        };
        for cr_row in 0..chroma_h {
            for cc in 0..chroma_w {
                let cb_val = ((cc * 7 + cr_row * 11 + 40) % 256) as u8;
                let cr_val = ((cc * 13 + cr_row * 3 + 80) % 256) as u8;
                let uv_byte = uv_start + cr_row * uv_row_stride + cc * 2;
                buf[uv_byte] = cb_val;
                buf[uv_byte + 1] = cr_val;
            }
        }
    }

    // Tags the source BT.601 full-range.
    src.set_colorimetry(Some(
        edgefirst_tensor::Colorimetry::default()
            .with_encoding(edgefirst_tensor::ColorEncoding::Bt601)
            .with_range(edgefirst_tensor::ColorRange::Full),
    ));

    // Analytic RGB reference from the same pattern, BT.601 full-range.
    let mut expected = vec![[0u8; 3]; w * h];
    for r in 0..h {
        for c in 0..w {
            let y = ((r * 3 + c * 5) % 256) as u8;
            let cc = c >> cw_shift;
            let cr_row = r >> ch_shift;
            let cb = ((cc * 7 + cr_row * 11 + 40) % 256) as u8;
            let cr = ((cc * 13 + cr_row * 3 + 80) % 256) as u8;
            expected[r * w + c] = bt601_ycbcr_to_rgb(y, cb, cr);
        }
    }

    (src, expected)
}

/// Compare the converted RGB tensor against an analytic reference.
fn check_analytic_rgb(
    dst: &Tensor<u8>,
    expected: &[[u8; 3]],
    w: usize,
    h: usize,
    label: &str,
    tolerance: u8,
) {
    let dst_stride = dst.effective_row_stride().unwrap_or(w * 3);
    let map = dst.map().unwrap();
    let out = map.as_slice();
    type PixMismatch = (usize, usize, [u8; 3], [u8; 3], i32);
    let mut first_fail: Option<PixMismatch> = None;
    let mut max_diff = 0i32;
    for y in 0..h {
        for x in 0..w {
            let oi = y * dst_stride + x * 3;
            let got = [out[oi], out[oi + 1], out[oi + 2]];
            let exp = expected[y * w + x];
            for c in 0..3usize {
                let d = (got[c] as i32 - exp[c] as i32).abs();
                if d > max_diff {
                    max_diff = d;
                }
                if d > tolerance as i32 && first_fail.is_none() {
                    first_fail = Some((x, y, got, exp, d));
                }
            }
        }
    }
    assert!(
        first_fail.is_none(),
        "{label}: pixel mismatch max_diff={max_diff} (tol={tolerance}): first bad at {:?}",
        first_fail
    );
}

fn run_odd_both(w: usize, h: usize, fmt: PixelFormat, label: &str) {
    let (src, expected) = make_odd_both_source(w, h, fmt);
    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_dyn, &mut dst_dyn, None).unwrap();
    let dst_u8 = dst_dyn.as_u8().unwrap();
    check_analytic_rgb(dst_u8, &expected, w, h, label, 4);
}

#[test]
fn nv12_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv12, "NV12 65x63");
}

#[test]
fn nv12_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv12, "NV12 789x383");
}

#[test]
fn nv16_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv16, "NV16 65x63");
}

#[test]
fn nv16_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv16, "NV16 789x383");
}

#[test]
fn nv24_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv24, "NV24 65x63");
}

#[test]
fn nv24_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv24, "NV24 789x383");
}

// ---------------------------------------------------------------------------
// NV12→RGB i8 dtype
// ---------------------------------------------------------------------------

/// NV12→RGB into i8 equals the u8 output XOR 0x80.
#[test]
fn nv12_to_rgb_i8() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    let mut src_i8 = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_i8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_i8_dyn = TensorDyn::from(src_i8);
    let mut dst_i8_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::I8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_i8_dyn, &mut dst_i8_dyn, None).unwrap();

    let mut src_u8 = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_u8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_u8_dyn = TensorDyn::from(src_u8);
    let mut dst_u8_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&src_u8_dyn, &mut dst_u8_dyn, None).unwrap();

    let i8_t = dst_i8_dyn.as_i8().unwrap();
    let u8_t = dst_u8_dyn.as_u8().unwrap();
    let i8_map = i8_t.map().unwrap();
    let u8_map = u8_t.map().unwrap();
    let i8_stride = i8_t.effective_row_stride().unwrap_or(w * 3);
    let u8_stride = u8_t.effective_row_stride().unwrap_or(w * 3);

    let mut max_diff = 0i32;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                // i8 output = u8 XOR 0x80 bias; reinterpret as u8 for comparison.
                let i8_as_u8 = i8_map.as_slice()[y * i8_stride + x * 3 + c] as u8;
                let expected = u8_map.as_slice()[y * u8_stride + x * 3 + c] ^ 0x80u8;
                let d = (i8_as_u8 as i32 - expected as i32).abs();
                max_diff = max_diff.max(d);
            }
        }
    }
    assert!(
        max_diff <= 1,
        "i8 output vs XOR-biased u8, max_diff={max_diff} > 1"
    );
}

// ---------------------------------------------------------------------------
// NV12→RGB f16 dtype
// ---------------------------------------------------------------------------

#[test]
fn nv12_to_rgb_f16() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    let mut src_f16 = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_f16.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_f16_dyn = TensorDyn::from(src_f16);
    let mut dst_f16_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::F16,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_f16_dyn, &mut dst_f16_dyn, None).unwrap();

    let mut src_u8 = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_u8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_u8_dyn = TensorDyn::from(src_u8);
    let mut dst_u8_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&src_u8_dyn, &mut dst_u8_dyn, None).unwrap();

    let f16_t = dst_f16_dyn.as_f16().unwrap();
    let u8_ref = dst_u8_dyn.as_u8().unwrap();
    let f16_map = f16_t.map().unwrap();
    let u8_map = u8_ref.map().unwrap();
    // effective_row_stride() returns bytes per row; f16 elements are 2 bytes.
    let f16_stride_bytes = f16_t.effective_row_stride().unwrap_or(w * 3 * 2);
    let f16_stride = f16_stride_bytes / std::mem::size_of::<half::f16>();
    let u8_stride = u8_ref.effective_row_stride().unwrap_or(w * 3);

    let tol = 0.004f32;
    let mut max_err = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                let f = f16_map.as_slice()[y * f16_stride + x * 3 + c].to_f32();
                let expected = u8_map.as_slice()[y * u8_stride + x * 3 + c] as f32 / 255.0;
                max_err = max_err.max((f - expected).abs());
            }
        }
    }
    assert!(
        max_err <= tol,
        "f16 output max error {max_err:.5} > tolerance {tol}"
    );
}

// ---------------------------------------------------------------------------
// Letterbox-resize sanity
// ---------------------------------------------------------------------------

#[test]
fn letterbox_resize_sanity_nv12_to_rgb_640() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    let mut src = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();

    // Letterbox 789×384 into 640×640.
    let _ = h;
    let crop = Crop::letterbox([114, 114, 114, 255]);

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn = TensorDyn::image(
        640,
        640,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_dyn,
        &mut dst_dyn,
        Some(edgefirst_image::ConvertOptions::new().crop(crop)),
    )
    .unwrap();

    let dst_u8 = dst_dyn.as_u8().unwrap();
    let dst_stride = dst_u8.effective_row_stride().unwrap_or(640 * 3);
    let map = dst_u8.map().unwrap();
    let out = map.as_slice();

    let mid_row = 640 / 2;
    let content_nonzero = out[mid_row * dst_stride..mid_row * dst_stride + 640 * 3]
        .iter()
        .filter(|&&v| v != 114)
        .count();
    assert!(
        content_nonzero > 10,
        "letterbox mid row should have image content, got {content_nonzero} non-padding pixels"
    );

    let pad_row = 10usize;
    assert_eq!(
        out[pad_row * dst_stride..pad_row * dst_stride + 3],
        [114, 114, 114],
        "padding row pixel must be [114,114,114]"
    );
}

// ---------------------------------------------------------------------------
// RGB→NV16 encoder stride correctness
// ---------------------------------------------------------------------------

/// RGB→NV16→RGB through a stride-aligned NV16 matches the same round trip through a second NV16.
#[test]
fn rgb_to_nv16_stride_round_trip() {
    let w = 789usize; // odd width
    let h = 64usize;

    let build_ref_rgb = || -> Tensor<u8> {
        let mut s = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let src_stride = s.effective_row_stride().unwrap_or(w * 3);
        let mut map = s.map().unwrap();
        let buf = map.as_mut_slice();
        for r in 0..h {
            for c in 0..w {
                let base = r * src_stride + c * 3;
                buf[base] = ((r * 7 + c * 3) % 256) as u8;
                buf[base + 1] = ((r * 5 + c * 11) % 256) as u8;
                buf[base + 2] = ((r * 13 + c * 7) % 256) as u8;
            }
        }
        s
    };

    // Strided NV16 path.
    let mut nv16 = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv16,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let nv16_stride = nv16.effective_row_stride().unwrap();
    assert_eq!(nv16_stride % 64, 0, "NV16 stride must be 64-byte aligned");
    assert!(
        nv16_stride > w,
        "NV16 stride {nv16_stride} should exceed odd width {w}"
    );

    let src_dyn = TensorDyn::from(build_ref_rgb());
    let mut nv16_dyn = TensorDyn::from(nv16);
    let mut proc = CPUProcessor::default();
    proc.convert(&src_dyn, &mut nv16_dyn, None).unwrap();

    let mut out_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&nv16_dyn, &mut out_dyn, None).unwrap();

    // Reference: direct RGB→RGB with the same pattern.
    let src2_dyn = TensorDyn::from(build_ref_rgb());
    let mut ref_dyn = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    // The same round trip through a second NV16 tensor.
    let mut nv16_ref = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv16,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut nv16_ref_dyn = TensorDyn::from(nv16_ref);
    proc.convert(&src2_dyn, &mut nv16_ref_dyn, None).unwrap();
    proc.convert(&nv16_ref_dyn, &mut ref_dyn, None).unwrap();

    let out_u8 = out_dyn.as_u8().unwrap();
    let ref_u8 = ref_dyn.as_u8().unwrap();
    let out_stride = out_u8.effective_row_stride().unwrap_or(w * 3);
    let ref_stride = ref_u8.effective_row_stride().unwrap_or(w * 3);
    let out_map = out_u8.map().unwrap();
    let ref_map = ref_u8.map().unwrap();

    let mut max_diff = 0i32;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                let ov = out_map.as_slice()[y * out_stride + x * 3 + c] as i32;
                let rv = ref_map.as_slice()[y * ref_stride + x * 3 + c] as i32;
                max_diff = max_diff.max((ov - rv).abs());
            }
        }
    }
    assert_eq!(
        max_diff, 0,
        "d3: strided NV16 round-trip differs from reference: max_diff={max_diff}"
    );
}

// ---------------------------------------------------------------------------
// CPU resize strided-source path
// ---------------------------------------------------------------------------

/// Resizing a strided RGB tensor matches resizing the same tight source.
#[test]
fn resize_strided_rgb_source() {
    let w = 64usize;
    let h = 48usize;
    let dst_w = 32usize;
    let dst_h = 24usize;

    let fill_pattern = |buf: &mut [u8], row_stride: usize| {
        for r in 0..h {
            for c in 0..w {
                let base = r * row_stride + c * 3;
                buf[base] = ((r * 11 + c * 7) % 256) as u8;
                buf[base + 1] = ((r * 3 + c * 17) % 256) as u8;
                buf[base + 2] = ((r * 19 + c * 5) % 256) as u8;
            }
        }
    };

    let tight_src = {
        let mut s = Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Rgb,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let stride = s.effective_row_stride().unwrap_or(w * 3);
        assert_eq!(stride, w * 3);
        {
            let mut map = s.map().unwrap();
            fill_pattern(map.as_mut_slice(), stride);
        }
        s
    };

    // A source with a row stride larger than w*3, or the tight allocation where unsupported.
    let padded_stride = 256usize; // 64-aligned, > w*3=192
    let (strided_src, strided_stride) = match strided_image(w, h, PixelFormat::Rgb, padded_stride) {
        Some(mut s) => {
            let stride = s.effective_row_stride().unwrap_or(padded_stride);
            {
                let mut map = s.map().unwrap();
                fill_pattern(map.as_mut_slice(), stride);
            }
            (s, stride)
        }
        None => {
            // No padded-stride image: use tight.
            let mut s = Tensor::<u8>::image(
                w,
                h,
                PixelFormat::Rgb,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
            let stride = s.effective_row_stride().unwrap_or(w * 3);
            {
                let mut map = s.map().unwrap();
                fill_pattern(map.as_mut_slice(), stride);
            }
            (s, stride)
        }
    };

    let mut proc = CPUProcessor::default();

    let tight_dyn = TensorDyn::from(tight_src);
    let mut tight_dst = TensorDyn::image(
        dst_w,
        dst_h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&tight_dyn, &mut tight_dst, None).unwrap();

    let strided_dyn = TensorDyn::from(strided_src);
    let mut strided_dst = TensorDyn::image(
        dst_w,
        dst_h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&strided_dyn, &mut strided_dst, None).unwrap();

    if strided_stride == w * 3 {
        // The tight fallback was used for both sources.
        return;
    }

    let t_u8 = tight_dst.as_u8().unwrap();
    let s_u8 = strided_dst.as_u8().unwrap();
    let t_map = t_u8.map().unwrap();
    let s_map = s_u8.map().unwrap();
    let t_stride = t_u8.effective_row_stride().unwrap_or(dst_w * 3);
    let s_stride = s_u8.effective_row_stride().unwrap_or(dst_w * 3);
    let mut max_diff = 0i32;
    for y in 0..dst_h {
        for x in 0..dst_w {
            for c in 0..3usize {
                let tv = t_map.as_slice()[y * t_stride + x * 3 + c] as i32;
                let sv = s_map.as_slice()[y * s_stride + x * 3 + c] as i32;
                max_diff = max_diff.max((tv - sv).abs());
            }
        }
    }
    assert_eq!(
        max_diff, 0,
        "d3: strided-src resize differs from tight-src resize: max_diff={max_diff}"
    );
}

// ---------------------------------------------------------------------------
// Packed-YUV to GREY and planar round trips
// ---------------------------------------------------------------------------

/// Convert a packed-YUV tensor to GREY through the public CPU pipeline and
/// return the tightly-packed `w*h` luma bytes.
fn cpu_to_grey_packed(src: Tensor<u8>, w: usize, h: usize) -> Vec<u8> {
    let mut proc = CPUProcessor::default();
    let src_dyn = TensorDyn::from(src);
    let mut dst = TensorDyn::image(
        w,
        h,
        PixelFormat::Grey,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&src_dyn, &mut dst, None).unwrap();
    let dst_u8 = dst.as_u8().unwrap();
    let stride = dst_u8.effective_row_stride().unwrap_or(w);
    let map = dst_u8.map().unwrap();
    let s = map.as_slice();
    let mut out = vec![0u8; w * h];
    for r in 0..h {
        out[r * w..(r + 1) * w].copy_from_slice(&s[r * stride..r * stride + w]);
    }
    out
}

/// Packed YUYV/VYUY → GREY gives the same output for a tight source and a
/// padded-stride source. `luma_off` is the byte offset of the luma sample
/// within each 2-byte unit (0 for YUYV's `[Y,C]`, 1 for VYUY's `[C,Y]`).
fn packed_yuv_to_grey_strided_matches_tight(fmt: PixelFormat, luma_off: usize) {
    let w = 64usize; // even width: one YUYV/VYUY macropixel spans 2 px
    let h = 16usize;

    let fill = |buf: &mut [u8], stride: usize| {
        for r in 0..h {
            for x in 0..w {
                let base = r * stride + x * 2;
                buf[base + luma_off] = ((r * 7 + x * 3) % 220 + 16) as u8; // luma
                buf[base + (1 - luma_off)] = ((r * 5 + x * 11) % 256) as u8; // chroma
            }
        }
    };

    let tight = {
        let mut s = Tensor::<u8>::image(
            w,
            h,
            fmt,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let stride = s.effective_row_stride().unwrap_or(w * 2);
        {
            let mut m = s.map().unwrap();
            fill(m.as_mut_slice(), stride);
        }
        s
    };
    let tight_grey = cpu_to_grey_packed(tight, w, h);

    // Strided source, where `image_with_stride` is supported.
    let padded = 192usize; // 64-aligned, > w*2 = 128
    if let Some(mut s) = strided_image(w, h, fmt, padded) {
        let stride = s.effective_row_stride().unwrap_or(padded);
        assert!(stride > w * 2, "expected a padded stride, got {stride}");
        {
            let mut m = s.map().unwrap();
            fill(m.as_mut_slice(), stride);
        }
        let strided_grey = cpu_to_grey_packed(s, w, h);
        assert_eq!(
            tight_grey, strided_grey,
            "{fmt:?}→Grey: strided source diverges from tight source (stride bug)"
        );
    }
}

#[test]
fn yuyv_to_grey_strided_matches_tight() {
    packed_yuv_to_grey_strided_matches_tight(PixelFormat::Yuyv, 0);
}

#[test]
fn vyuy_to_grey_strided_matches_tight() {
    packed_yuv_to_grey_strided_matches_tight(PixelFormat::Vyuy, 1);
}

/// Planar → packed read-back (`convert_8bps_*` / `convert_prgba_*`) round-trips losslessly.
#[test]
fn planar_packed_roundtrip() {
    let w = 40usize;
    let h = 24usize;

    let convert = |src: &TensorDyn, dst_fmt: PixelFormat, dst_ch: usize| -> TensorDyn {
        let mut proc = CPUProcessor::default();
        let mut dst = TensorDyn::image(
            w,
            h,
            dst_fmt,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let _ = dst_ch;
        proc.convert(src, &mut dst, None).unwrap();
        dst
    };

    let read_packed = |t: &TensorDyn, ch: usize| -> Vec<u8> {
        let u8t = t.as_u8().unwrap();
        let stride = u8t.effective_row_stride().unwrap_or(w * ch);
        let map = u8t.map().unwrap();
        let s = map.as_slice();
        let mut out = vec![0u8; w * h * ch];
        for r in 0..h {
            out[r * w * ch..(r + 1) * w * ch].copy_from_slice(&s[r * stride..r * stride + w * ch]);
        }
        out
    };

    // RGB → PlanarRgb → RGB is the identity.
    let mut rgb = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let stride = rgb.effective_row_stride().unwrap_or(w * 3);
        let mut m = rgb.map().unwrap();
        let s = m.as_mut_slice();
        for r in 0..h {
            for x in 0..w {
                let b = r * stride + x * 3;
                s[b] = ((r * 9 + x * 5) % 256) as u8;
                s[b + 1] = ((r * 3 + x * 13) % 256) as u8;
                s[b + 2] = ((r * 17 + x * 2) % 256) as u8;
            }
        }
    }
    let rgb_dyn = TensorDyn::from(rgb);
    let rgb_orig = read_packed(&rgb_dyn, 3);
    let planar = convert(&rgb_dyn, PixelFormat::PlanarRgb, 3);
    let back = convert(&planar, PixelFormat::Rgb, 3);
    assert_eq!(
        rgb_orig,
        read_packed(&back, 3),
        "RGB→PlanarRgb→RGB not identity"
    );

    // RGBA → PlanarRgba → RGBA is the identity.
    let mut rgba = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Rgba,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let stride = rgba.effective_row_stride().unwrap_or(w * 4);
        let mut m = rgba.map().unwrap();
        let s = m.as_mut_slice();
        for r in 0..h {
            for x in 0..w {
                let b = r * stride + x * 4;
                s[b] = ((r + x) % 256) as u8;
                s[b + 1] = ((r * 2 + x * 3) % 256) as u8;
                s[b + 2] = ((r * 4 + x) % 256) as u8;
                s[b + 3] = ((r * 6 + x * 7) % 256) as u8;
            }
        }
    }
    let rgba_dyn = TensorDyn::from(rgba);
    let rgba_orig = read_packed(&rgba_dyn, 4);
    let planar_a = convert(&rgba_dyn, PixelFormat::PlanarRgba, 4);
    let back_a = convert(&planar_a, PixelFormat::Rgba, 4);
    assert_eq!(
        rgba_orig,
        read_packed(&back_a, 4),
        "RGBA→PlanarRgba→RGBA not identity"
    );
}

// ---------------------------------------------------------------------------
// Fused NV→PlanarRgb / PlanarRgba (strip decode) parity
// ---------------------------------------------------------------------------

fn fused_nv_planar_parity(w: usize, h: usize, fmt: PixelFormat, with_alpha: bool, label: &str) {
    let mut src = Tensor::<u8>::image(
        w,
        h,
        fmt,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = src.map().unwrap();
        for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37).wrapping_add(11)) & 0xff) as u8;
        }
    }
    let src_dyn = TensorDyn::from(src);
    let mut proc = CPUProcessor::default();

    let pfmt = if with_alpha {
        PixelFormat::PlanarRgba
    } else {
        PixelFormat::PlanarRgb
    };
    let mut planar = TensorDyn::image(
        w,
        h,
        pfmt,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&src_dyn, &mut planar, None).unwrap();

    // Reference: the NV→RGB packed conversion.
    let mut rgb = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    proc.convert(&src_dyn, &mut rgb, None).unwrap();

    let planar_t = planar.as_u8().unwrap();
    let rgb_t = rgb.as_u8().unwrap();
    let p_stride = planar_t.effective_row_stride().unwrap_or(w);
    let rgb_stride = rgb_t.effective_row_stride().unwrap_or(w * 3);
    let p_plane = p_stride * h;
    let pm = planar_t.map().unwrap();
    let rm = rgb_t.map().unwrap();
    let p = pm.as_slice();
    let r = rm.as_slice();

    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                assert_eq!(
                    p[c * p_plane + y * p_stride + x],
                    r[y * rgb_stride + x * 3 + c],
                    "{label}: plane {c} at ({x},{y}) differs from packed reference"
                );
            }
            if with_alpha {
                assert_eq!(
                    p[3 * p_plane + y * p_stride + x],
                    255,
                    "{label}: alpha at ({x},{y}) must be 255"
                );
            }
        }
    }
}

#[test]
fn fused_nv_planar_parity_matrix() {
    // (w, h): clean strip multiple, strip-tail, odd-w, odd-h, odd-both, tiny.
    let dims = [
        (64, 64),
        (128, 100),
        (101, 96),
        (96, 97),
        (97, 95),
        (3, 3),
        (1, 1),
    ];
    for fmt in [PixelFormat::Nv12, PixelFormat::Nv16, PixelFormat::Nv24] {
        for &(w, h) in &dims {
            for with_alpha in [false, true] {
                let label = format!(
                    "{fmt:?} {w}x{h} {}",
                    if with_alpha {
                        "PlanarRgba"
                    } else {
                        "PlanarRgb"
                    }
                );
                fused_nv_planar_parity(w, h, fmt, with_alpha, &label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Odd-width YUYV destinations
// ---------------------------------------------------------------------------

/// Convert `src_fmt` → YUYV at `w`×`h` and assert every destination byte was
/// written.
fn yuyv_dst_fully_written(w: usize, h: usize, src_fmt: PixelFormat, label: &str) {
    const POISON: u8 = 0xAB;

    let mut src = TensorDyn::image(
        w,
        h,
        src_fmt,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut map = src.as_u8_mut().unwrap().map_mut().unwrap();
        for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
            *b = (i.wrapping_mul(37).wrapping_add(11) % 251) as u8;
        }
    }

    let mut dst = TensorDyn::image(
        w,
        h,
        PixelFormat::Yuyv,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    dst.as_u8_mut()
        .unwrap()
        .map_mut()
        .unwrap()
        .as_mut_slice()
        .fill(POISON);

    CPUProcessor::default()
        .convert(&src, &mut dst, None)
        .unwrap_or_else(|e| panic!("{label}: convert failed: {e}"));

    let dst_u8 = dst.as_u8().unwrap();
    let stride = dst_u8.effective_row_stride().unwrap_or(w * 2);
    let map = dst_u8.map().unwrap();
    let out = map.as_slice();
    let row_bytes = w * 2;

    for y in 0..h {
        let row = &out[y * stride..y * stride + row_bytes];
        assert!(
            !row.iter().all(|&b| b == POISON),
            "{label}: row {y} was not written at all"
        );
        // The trailing pixel of an odd-width row.
        let tail = &row[row_bytes.saturating_sub(2)..];
        assert!(
            tail.iter().any(|&b| b != POISON),
            "{label}: row {y} trailing pixel left unwritten (bytes {tail:?})"
        );
    }
}

#[test]
fn odd_width_rgb_to_yuyv_writes_every_byte() {
    for (w, h) in [(5, 3), (7, 1), (1, 4), (65, 33)] {
        yuyv_dst_fully_written(w, h, PixelFormat::Rgb, &format!("rgb→yuyv {w}x{h}"));
    }
}

#[test]
fn odd_width_rgba_to_yuyv_writes_every_byte() {
    for (w, h) in [(5, 3), (7, 1), (1, 4), (65, 33)] {
        yuyv_dst_fully_written(w, h, PixelFormat::Rgba, &format!("rgba→yuyv {w}x{h}"));
    }
}

#[test]
fn odd_width_grey_to_yuyv_writes_every_byte() {
    for (w, h) in [(5, 3), (7, 1), (1, 4), (65, 33)] {
        yuyv_dst_fully_written(w, h, PixelFormat::Grey, &format!("grey→yuyv {w}x{h}"));
    }
}

/// Even-width RGB → YUYV writes every destination byte.
#[test]
fn even_width_rgb_to_yuyv_writes_every_byte() {
    for (w, h) in [(4, 2), (64, 32)] {
        yuyv_dst_fully_written(w, h, PixelFormat::Rgb, &format!("rgb→yuyv {w}x{h}"));
    }
}
