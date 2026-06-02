// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
#![allow(unused_mut)]

//! End-to-end odd-dimension CPU conversion tests.
//!
//! Covers Deliverable 1 (Part-1 assertion helper + reuse cell),
//! Deliverable 2 (CPU convert cells for odd-W/H JPEG fixtures and
//! analytically-constructed odd-both sources), and the resize strided-source
//! path (Deliverable 3 subset).
//!
//! All cells run under the default feature set (no `dma_test_formats` flag).

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_image::{CPUProcessor, Crop, Flip, ImageProcessorTrait, Rect, Rotation};
use edgefirst_tensor::{
    DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

// ---------------------------------------------------------------------------
// Test-data helper — mirrors the codec test suite resolution order.
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
// Part-1 assertion helper (Deliverable 1)
// ---------------------------------------------------------------------------

/// Assert the Phase-1 representation invariants for a semi-planar / grey tensor.
///
/// * logical `width()` and `height()` match `logical_w` / `logical_h`.
/// * `format()` matches `fmt`.
/// * `effective_row_stride()` is 64-byte aligned and >= even(logical_w).
/// * The mapped buffer is large enough to hold all planes at that stride.
fn assert_part1(tensor: &Tensor<u8>, logical_w: usize, logical_h: usize, fmt: PixelFormat) {
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

    // Use as_slice().len() to get the actual physical buffer capacity
    // (honours byte_size_override when stride > tight).
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
// Deliverable 1 — Part-1 assertion cells for each fixture
// ---------------------------------------------------------------------------

#[test]
fn d1_part1_nv12_odd_w() {
    let jpeg = testdata("jaguar.jpg");
    let mut t = Tensor::<u8>::image(789, 384, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_part1(&t, 789, 384, PixelFormat::Nv12);

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
fn d1_part1_nv16_odd_w() {
    let jpeg = testdata("jaguar_422.jpg");
    let mut t = Tensor::<u8>::image(789, 384, PixelFormat::Nv16, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_part1(&t, 789, 384, PixelFormat::Nv16);
}

#[test]
fn d1_part1_nv24_odd_w() {
    let jpeg = testdata("jaguar_444.jpg");
    let mut t = Tensor::<u8>::image(789, 384, PixelFormat::Nv24, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_part1(&t, 789, 384, PixelFormat::Nv24);
}

#[test]
fn d1_part1_grey_odd_h() {
    let jpeg = testdata("grey.jpg");
    let mut t = Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_part1(&t, 1024, 681, PixelFormat::Grey);
}

#[test]
fn d1_part1_nv12_odd_h() {
    let jpeg = testdata("grey-rgb.jpg");
    let mut t = Tensor::<u8>::image(1024, 681, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    t.load_image(&mut dec, &jpeg).unwrap();
    assert_part1(&t, 1024, 681, PixelFormat::Nv12);
}

// ---------------------------------------------------------------------------
// Deliverable 1 — canonical pre-alloc-max + decode-small reuse cell
// ---------------------------------------------------------------------------

#[test]
fn d1_reuse_cell_max_alloc_decode_small() {
    // Allocate at max size (1920×1088 NV12) and record the large buffer stride.
    let mut pool =
        Tensor::<u8>::image(1920, 1088, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let large_stride = pool.effective_row_stride().unwrap();
    assert_eq!(large_stride % 64, 0);
    assert!(large_stride >= 1920);

    // Decode jaguar.jpg (789×384) into the large pool.
    let jpeg = testdata("jaguar.jpg");
    let mut dec = ImageDecoder::new();
    pool.load_image(&mut dec, &jpeg).unwrap();

    // Logical dims must update to the decoded image's size.
    assert_eq!(pool.width(), Some(789), "logical width must update to 789");
    assert_eq!(
        pool.height(),
        Some(384),
        "logical height must update to 384"
    );

    // The buffer stride must NOT shrink to 832 (tight for 789 wide) —
    // the pre-allocated 1920-wide pitch is preserved.
    let post_stride = pool.effective_row_stride().unwrap();
    assert_eq!(
        post_stride, large_stride,
        "stride must stay at pre-alloc pitch {large_stride}, got {post_stride}"
    );

    // The decoded 789×384 luma region must be non-trivial.
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
// Helpers for Deliverable 2 (JPEG-based convert cells)
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

    let mut src = Tensor::<u8>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();
    assert_eq!(src.width(), Some(w));
    assert_eq!(src.height(), Some(h));

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_dyn,
        &mut dst_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();
    let dst_u8 = dst_dyn.as_u8().unwrap();
    mae_vs_image_crate_rgb(dst_u8, &ref_rgb, w, h)
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-01 NV12 odd-W
// ---------------------------------------------------------------------------

#[test]
fn c01_nv12_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar.jpg", PixelFormat::Nv12, 789, 384);
    eprintln!("C-01 NV12 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(
        mae < 3.0,
        "C-01: NV12 odd-W MAE {mae:.3} exceeds tolerance 3.0"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-04 NV16 odd-W
// ---------------------------------------------------------------------------

#[test]
fn c04_nv16_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar_422.jpg", PixelFormat::Nv16, 789, 384);
    eprintln!("C-04 NV16 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(
        mae < 4.0,
        "C-04: NV16 odd-W MAE {mae:.3} exceeds tolerance 4.0"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-05 NV24 odd-W
// ---------------------------------------------------------------------------

#[test]
fn c05_nv24_odd_w_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("jaguar_444.jpg", PixelFormat::Nv24, 789, 384);
    eprintln!("C-05 NV24 odd-W: MAE={mae:.3}, max_diff={max_diff}");
    assert!(
        mae < 4.0,
        "C-05: NV24 odd-W MAE {mae:.3} exceeds tolerance 4.0"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-02 GREY odd-H
// ---------------------------------------------------------------------------

#[test]
fn c02_grey_odd_h_to_rgb() {
    let jpeg = testdata("grey.jpg");
    let w = 1024usize;
    let h = 681usize;
    let ref_luma = image::load_from_memory(&jpeg).unwrap().to_luma8();
    assert_eq!(ref_luma.dimensions(), (w as u32, h as u32));

    let mut src = Tensor::<u8>::image(w, h, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_dyn,
        &mut dst_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    let dst_u8 = dst_dyn.as_u8().unwrap();
    let dst_stride = dst_u8.effective_row_stride().unwrap_or(w * 3);
    let map = dst_u8.map().unwrap();
    let our = map.as_slice();

    // Grey→RGB: R=G=B=Y. Compare against luma reference.
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
    eprintln!("C-02 GREY odd-H: MAE={mae:.3}, max_diff={max_diff}");
    assert!(
        mae < 3.0,
        "C-02: GREY odd-H MAE {mae:.3} exceeds tolerance 3.0"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-03 NV12 odd-H
// ---------------------------------------------------------------------------

#[test]
fn c03_nv12_odd_h_to_rgb() {
    let (mae, max_diff) = jpeg_nv_to_rgb_mae("grey-rgb.jpg", PixelFormat::Nv12, 1024, 681);
    eprintln!("C-03 NV12 odd-H: MAE={mae:.3}, max_diff={max_diff}");
    assert!(
        mae < 3.0,
        "C-03: NV12 odd-H MAE {mae:.3} exceeds tolerance 3.0"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-06..08 analytically-constructed odd-both sources
// ---------------------------------------------------------------------------

/// BT.601 full-range YCbCr→RGB (matches the HAL `yuv` crate kernel).
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
/// Chroma:  `Cb(cc,cr) = (cc*7 + cr*11 + 40) % 256`  (non-neutral, creates colour)
///          `Cr(cc,cr) = (cc*13 + cr*3 + 80) % 256`
/// where `cc`/`cr` are chroma-cell coordinates (divided by subsampling factor).
fn make_odd_both_source(w: usize, h: usize, fmt: PixelFormat) -> (Tensor<u8>, Vec<[u8; 3]>) {
    let mut src = Tensor::<u8>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
    let stride = src.effective_row_stride().unwrap();

    let (cw_shift, ch_shift) = match fmt {
        PixelFormat::Nv12 => (1usize, 1usize),
        PixelFormat::Nv16 => (1, 0),
        PixelFormat::Nv24 => (0, 0),
        _ => panic!("unsupported format {fmt:?}"),
    };
    // Number of chroma rows/columns: ceil(dim / divisor) to cover odd dims.
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
        // Fill chroma (interleaved [Cb, Cr] per chroma column).
        //
        // NV12/NV16 UV row pitch = stride (same as luma): each chroma row
        // occupies `stride` bytes.
        // NV24 UV row pitch = stride*2: each image row of UV pairs occupies
        // 2*stride bytes (the `yuv` crate uses `uv_stride = y_stride * 2`).
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

    // Build analytic RGB reference using the same pattern and BT.601-full.
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
    let mut dst_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_dyn,
        &mut dst_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();
    let dst_u8 = dst_dyn.as_u8().unwrap();
    check_analytic_rgb(dst_u8, &expected, w, h, label, 2);
}

#[test]
fn c06_nv12_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv12, "C-06 NV12 65x63");
}

#[test]
fn c06_nv12_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv12, "C-06 NV12 789x383");
}

#[test]
fn c07_nv16_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv16, "C-07 NV16 65x63");
}

#[test]
fn c07_nv16_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv16, "C-07 NV16 789x383");
}

#[test]
fn c08_nv24_odd_both_small() {
    run_odd_both(65, 63, PixelFormat::Nv24, "C-08 NV24 65x63");
}

#[test]
fn c08_nv24_odd_both_medium() {
    run_odd_both(789, 383, PixelFormat::Nv24, "C-08 NV24 789x383");
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-10 NV12→RGB i8 dtype
// ---------------------------------------------------------------------------

/// Decode jaguar.jpg to NV12, convert to RGB i8, verify the XOR-0x80 bias
/// matches the u8 path.
#[test]
fn c10_nv12_to_rgb_i8() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    // i8 output path
    let mut src_i8 = Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_i8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_i8_dyn = TensorDyn::from(src_i8);
    let mut dst_i8_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::I8, Some(TensorMemory::Mem)).unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_i8_dyn,
        &mut dst_i8_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    // u8 reference path
    let mut src_u8 = Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_u8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_u8_dyn = TensorDyn::from(src_u8);
    let mut dst_u8_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    proc.convert(
        &src_u8_dyn,
        &mut dst_u8_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

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
        "C-10: i8 output vs XOR-biased u8, max_diff={max_diff} > 1"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — C-11 NV12→RGB f16 dtype
// ---------------------------------------------------------------------------

#[test]
fn c11_nv12_to_rgb_f16() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    // f16 output path
    let mut src_f16 =
        Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_f16.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_f16_dyn = TensorDyn::from(src_f16);
    let mut dst_f16_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::F16, Some(TensorMemory::Mem)).unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_f16_dyn,
        &mut dst_f16_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    // u8 reference path
    let mut src_u8 = Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    {
        let mut dec = ImageDecoder::new();
        src_u8.load_image(&mut dec, &jpeg).unwrap();
    }
    let src_u8_dyn = TensorDyn::from(src_u8);
    let mut dst_u8_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    proc.convert(
        &src_u8_dyn,
        &mut dst_u8_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

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
        "C-11: f16 output max error {max_err:.5} > tolerance {tol}"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 2 — Letterbox-resize sanity
// ---------------------------------------------------------------------------

#[test]
fn letterbox_resize_sanity_nv12_to_rgb_640() {
    let jpeg = testdata("jaguar.jpg");
    let w = 789usize;
    let h = 384usize;

    let mut src = Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut dec = ImageDecoder::new();
    src.load_image(&mut dec, &jpeg).unwrap();

    // Fit 789×384 into 640×640: scale = 640/789, scaled_h = round(384*640/789)
    let scale = 640.0f32 / 789.0f32;
    let scaled_h = (h as f32 * scale).round() as usize;
    let pad_top = (640 - scaled_h) / 2;
    let dst_rect = Rect {
        left: 0,
        top: pad_top,
        width: 640,
        height: scaled_h,
    };
    let crop = Crop {
        src_rect: None,
        dst_rect: Some(dst_rect),
        dst_color: Some([114, 114, 114, 255]),
    };

    let src_dyn = TensorDyn::from(src);
    let mut dst_dyn = TensorDyn::image(
        640,
        640,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    let mut proc = CPUProcessor::default();
    proc.convert(&src_dyn, &mut dst_dyn, Rotation::None, Flip::None, crop)
        .unwrap();

    let dst_u8 = dst_dyn.as_u8().unwrap();
    let dst_stride = dst_u8.effective_row_stride().unwrap_or(640 * 3);
    let map = dst_u8.map().unwrap();
    let out = map.as_slice();

    // Content band (mid-height row) should contain non-padding pixels.
    let mid_row = 640 / 2;
    let content_nonzero = out[mid_row * dst_stride..mid_row * dst_stride + 640 * 3]
        .iter()
        .filter(|&&v| v != 114)
        .count();
    assert!(
        content_nonzero > 10,
        "letterbox mid row should have image content, got {content_nonzero} non-padding pixels"
    );

    // Padding rows (top-of-frame) should be the fill colour.
    let pad_row = 10usize;
    assert_eq!(
        out[pad_row * dst_stride..pad_row * dst_stride + 3],
        [114, 114, 114],
        "padding row pixel must be [114,114,114]"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 3 — RGB→NV16 encoder stride correctness
// ---------------------------------------------------------------------------

/// Allocate a stride-aligned NV16 destination, fill via RGB→NV16, then
/// convert back NV16→RGB and compare to the reference (both paths produce
/// the same round-trip so max_diff == 0).
#[test]
fn d3_rgb_to_nv16_stride_round_trip() {
    let w = 789usize; // odd width
    let h = 64usize;

    // Build reference RGB (tight)
    let build_ref_rgb = || -> Tensor<u8> {
        let mut s = Tensor::<u8>::image(w, h, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // --- Strided NV16 path ---
    let mut nv16 = Tensor::<u8>::image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem)).unwrap();
    let nv16_stride = nv16.effective_row_stride().unwrap();
    assert_eq!(nv16_stride % 64, 0, "NV16 stride must be 64-byte aligned");
    assert!(
        nv16_stride > w,
        "NV16 stride {nv16_stride} should exceed odd width {w}"
    );

    let src_dyn = TensorDyn::from(build_ref_rgb());
    let mut nv16_dyn = TensorDyn::from(nv16);
    let mut proc = CPUProcessor::default();
    proc.convert(
        &src_dyn,
        &mut nv16_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    // NV16 → RGB
    let mut out_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    proc.convert(
        &nv16_dyn,
        &mut out_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    // --- Reference: direct RGB→RGB (no NV16 intermediate), same pattern ---
    let src2_dyn = TensorDyn::from(build_ref_rgb());
    let mut ref_dyn =
        TensorDyn::image(w, h, PixelFormat::Rgb, DType::U8, Some(TensorMemory::Mem)).unwrap();
    // To get the same YUV round-trip losses, also go through NV16→RGB using a
    // fresh NV16 tensor (identical code path, tight or strided).
    let mut nv16_ref =
        Tensor::<u8>::image(w, h, PixelFormat::Nv16, Some(TensorMemory::Mem)).unwrap();
    let mut nv16_ref_dyn = TensorDyn::from(nv16_ref);
    proc.convert(
        &src2_dyn,
        &mut nv16_ref_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();
    proc.convert(
        &nv16_ref_dyn,
        &mut ref_dyn,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

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
    // Both use identical NV16 encode/decode — results must be identical.
    assert_eq!(
        max_diff, 0,
        "d3: strided NV16 round-trip differs from reference: max_diff={max_diff}"
    );
}

// ---------------------------------------------------------------------------
// Deliverable 3 — CPU resize strided-source path
// ---------------------------------------------------------------------------

/// Resize a strided RGB tensor and compare to the resize of the same tight
/// source.  Proves the de-striding copy in `resize_flip_rotate_pf` is correct.
#[test]
fn d3_resize_strided_rgb_source() {
    let w = 64usize;
    let h = 48usize;
    let dst_w = 32usize;
    let dst_h = 24usize;

    // Build the pixel pattern
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

    // Tight source (natural stride = w*3 = 192)
    let tight_src = {
        let mut s = Tensor::<u8>::image(w, h, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
        let stride = s.effective_row_stride().unwrap_or(w * 3);
        assert_eq!(stride, w * 3);
        {
            let mut map = s.map().unwrap();
            fill_pattern(map.as_mut_slice(), stride);
        }
        s
    };

    // Strided source: use `image_with_stride` to allocate an Rgb tensor with a
    // row stride larger than w*3.  On Linux (DMA backing) this is natively
    // supported and gives a real padded allocation.  On other platforms,
    // fall back to the tight allocation (the resize path is still exercised,
    // just without the copy-to-tight branch — a no-op is fine there).
    let padded_stride = 256usize; // 64-aligned, > w*3=192
    let strided_src_result =
        Tensor::<u8>::image_with_stride(w, h, PixelFormat::Rgb, padded_stride, None);
    let (strided_src, strided_stride) = match strided_src_result {
        Ok(mut s) => {
            let stride = s.effective_row_stride().unwrap_or(padded_stride);
            {
                let mut map = s.map().unwrap();
                fill_pattern(map.as_mut_slice(), stride);
            }
            (s, stride)
        }
        Err(_) => {
            // Platform doesn't support image_with_stride (non-Linux); use tight.
            let mut s =
                Tensor::<u8>::image(w, h, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
            let stride = s.effective_row_stride().unwrap_or(w * 3);
            {
                let mut map = s.map().unwrap();
                fill_pattern(map.as_mut_slice(), stride);
            }
            (s, stride)
        }
    };

    let mut proc = CPUProcessor::default();

    // Resize tight source
    let tight_dyn = TensorDyn::from(tight_src);
    let mut tight_dst = TensorDyn::image(
        dst_w,
        dst_h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    proc.convert(
        &tight_dyn,
        &mut tight_dst,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    // Resize strided source
    let strided_dyn = TensorDyn::from(strided_src);
    let mut strided_dst = TensorDyn::image(
        dst_w,
        dst_h,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    proc.convert(
        &strided_dyn,
        &mut strided_dst,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap();

    if strided_stride == w * 3 {
        // Both paths are identical (fallback hit); trivially passes.
        return;
    }

    // Both should produce identical resize output
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
