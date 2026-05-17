// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! End-to-end EXIF orientation coverage for both JPEG and PNG.
//!
//! The 16 fixtures `testdata/zidane_exif_<N>.{jpg,png}` for N in 1..=8 are
//! produced by `scripts/generate_exif_fixtures.py` and carry the same pixel
//! data with only the EXIF orientation tag varying. The tests therefore
//! exercise the codec's rotation/flip logic for every spec-defined
//! orientation against a single source-of-truth pixel buffer.
//!
//! EXIF orientation reference (per EXIF/TIFF spec):
//!
//!   1: Identity
//!   2: Mirror horizontal (flip across the vertical axis)
//!   3: Rotate 180°
//!   4: Mirror vertical (flip across the horizontal axis = 180° + flip-H)
//!   5: Rotate 90° CW + mirror horizontal (transpose along main diagonal)
//!   6: Rotate 90° CW
//!   7: Rotate 90° CCW + mirror horizontal (transpose along anti-diagonal)
//!   8: Rotate 90° CCW

use edgefirst_codec::{peek_info, DecodeOptions, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};
use std::path::PathBuf;

const W: usize = 1280;
const H: usize = 720;
const CH: usize = 3;

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata")
        });
    let path = root.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read testdata {}: {e}", path.display()))
}

/// Decode the file at `name` with apply_exif=`apply`, return (width, height, pixels).
fn decode_to_rgb(name: &str, apply: bool, max_w: usize, max_h: usize) -> (usize, usize, Vec<u8>) {
    let data = testdata(name);
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(apply);
    let mut tensor =
        Tensor::<u8>::image(max_w, max_h, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(&mut decoder, &data, &opts)
        .unwrap_or_else(|e| panic!("decode {name}: {e}"));
    let stride = tensor.effective_row_stride().unwrap_or(max_w * CH);
    let map = tensor.map().unwrap();
    let bytes: &[u8] = &map;
    // Compact rows: each row is `info.width * CH` valid bytes followed by stride padding.
    let mut packed = vec![0u8; info.width * info.height * CH];
    let row_bytes = info.width * CH;
    for y in 0..info.height {
        let src = y * stride;
        let dst = y * row_bytes;
        packed[dst..dst + row_bytes].copy_from_slice(&bytes[src..src + row_bytes]);
    }
    (info.width, info.height, packed)
}

/// Apply EXIF orientation `o` to a contiguous (w × h × 3) u8 RGB buffer.
/// Returns the transformed (w', h', pixels) where w'/h' may swap for 5/6/7/8.
/// This is the test oracle — implemented in the most explicit possible form
/// so we trust the comparison rather than the codec's apply_exif_u8.
fn apply_exif_oracle(w: usize, h: usize, src: &[u8], o: u32) -> (usize, usize, Vec<u8>) {
    assert_eq!(src.len(), w * h * CH);
    let (new_w, new_h) = match o {
        5..=8 => (h, w),
        _ => (w, h),
    };
    let mut dst = vec![0u8; new_w * new_h * CH];
    for y in 0..h {
        for x in 0..w {
            let src_off = (y * w + x) * CH;
            let (nx, ny) = match o {
                1 => (x, y),
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (h - 1 - y, x),
                7 => (h - 1 - y, w - 1 - x),
                8 => (y, w - 1 - x),
                _ => unreachable!("orientation {o} out of range"),
            };
            let dst_off = (ny * new_w + nx) * CH;
            dst[dst_off..dst_off + CH].copy_from_slice(&src[src_off..src_off + CH]);
        }
    }
    (new_w, new_h, dst)
}

/// Sum of absolute differences across two equal-length buffers.
fn sad(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y) as u64)
        .sum()
}

// ---------------------------------------------------------------------------
// JPEG
// ---------------------------------------------------------------------------

#[test]
fn jpeg_exif_apply_false_all_identical() {
    // Every variant has identical pixel content (piexif only rewrote the
    // EXIF APP1 segment). With apply_exif=false the decoder must produce
    // bit-identical output across all 8 fixtures.
    let (_, _, reference) = decode_to_rgb("zidane_exif_1.jpg", false, W, H);
    for o in 2..=8 {
        let (w, h, pixels) = decode_to_rgb(&format!("zidane_exif_{o}.jpg"), false, W, H);
        assert_eq!((w, h), (W, H), "orientation {o} dims differ from baseline");
        assert_eq!(
            pixels, reference,
            "orientation {o} pixels differ from orientation=1 with apply_exif=false"
        );
    }
}

#[test]
fn jpeg_exif_peek_swaps_dims_for_5_6_7_8() {
    for o in 1..=8u32 {
        let data = testdata(&format!("zidane_exif_{o}.jpg"));
        let opts = DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(true);
        let info = peek_info(&data, &opts).unwrap();
        match o {
            5..=8 => assert_eq!(
                (info.width, info.height),
                (H, W),
                "orientation {o} should peek post-rotation (H, W)"
            ),
            _ => assert_eq!(
                (info.width, info.height),
                (W, H),
                "orientation {o} should peek pre-rotation (W, H)"
            ),
        }
    }
}

#[test]
fn jpeg_exif_apply_true_matches_oracle_orientation_1() {
    // Trivial identity check.
    let (w, h, decoded) = decode_to_rgb("zidane_exif_1.jpg", true, W, H);
    let (_, _, reference) = decode_to_rgb("zidane_exif_1.jpg", false, W, H);
    assert_eq!((w, h), (W, H));
    assert_eq!(decoded, reference);
}

/// Parameterised test: each orientation's apply_exif=true output matches the
/// oracle transform of the apply_exif=false orientation=1 reference.
///
/// Tolerance is conservative — the JPEG decode itself is deterministic, so
/// the only difference vs the oracle is whether the codec's rotation matches
/// the EXIF spec direction. A perfect match (SAD=0) per row demonstrates
/// the codec's apply_exif_u8 is bit-exact with the oracle.
fn check_orientation(o: u32) {
    let (ref_w, ref_h, reference) = decode_to_rgb("zidane_exif_1.jpg", false, W, H);
    assert_eq!((ref_w, ref_h), (W, H));
    let (exp_w, exp_h, expected) = apply_exif_oracle(ref_w, ref_h, &reference, o);

    let (got_w, got_h, decoded) =
        decode_to_rgb(&format!("zidane_exif_{o}.jpg"), true, W.max(H), W.max(H));
    assert_eq!(
        (got_w, got_h),
        (exp_w, exp_h),
        "orientation {o}: decoded dims differ from oracle"
    );
    let pixel_count = exp_w * exp_h * CH;
    let s = sad(&decoded[..pixel_count], &expected[..pixel_count]);
    // Bit-exact match expected — apply_exif_u8 is a pure byte rearrangement.
    assert_eq!(
        s, 0,
        "orientation {o}: SAD={s} (expected 0 — apply_exif_u8 is a pure byte rearrangement)"
    );
}

#[test]
fn jpeg_exif_orientation_1() {
    check_orientation(1);
}

#[test]
fn jpeg_exif_orientation_2() {
    check_orientation(2);
}

#[test]
fn jpeg_exif_orientation_3() {
    check_orientation(3);
}

#[test]
fn jpeg_exif_orientation_4() {
    check_orientation(4);
}

#[test]
fn jpeg_exif_orientation_5() {
    check_orientation(5);
}

#[test]
fn jpeg_exif_orientation_6() {
    check_orientation(6);
}

#[test]
fn jpeg_exif_orientation_7() {
    check_orientation(7);
}

#[test]
fn jpeg_exif_orientation_8() {
    check_orientation(8);
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

#[test]
fn png_exif_apply_false_all_identical() {
    // PNG variants share identical IDAT (deterministic deflate, same source
    // pixels) and differ only in the eXIf chunk.
    let (_, _, reference) = decode_to_rgb("zidane_exif_1.png", false, W, H);
    for o in 2..=8 {
        let (w, h, pixels) = decode_to_rgb(&format!("zidane_exif_{o}.png"), false, W, H);
        assert_eq!((w, h), (W, H));
        assert_eq!(
            pixels, reference,
            "PNG orientation {o} pixels differ with apply_exif=false"
        );
    }
}

#[test]
fn png_exif_peek_swaps_dims_for_5_6_7_8() {
    for o in 1..=8u32 {
        let data = testdata(&format!("zidane_exif_{o}.png"));
        let opts = DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(true);
        let info = peek_info(&data, &opts).unwrap();
        match o {
            5..=8 => assert_eq!(
                (info.width, info.height),
                (H, W),
                "PNG orientation {o} should peek post-rotation"
            ),
            _ => assert_eq!(
                (info.width, info.height),
                (W, H),
                "PNG orientation {o} should peek pre-rotation"
            ),
        }
    }
}

fn check_png_orientation(o: u32) {
    let (ref_w, ref_h, reference) = decode_to_rgb("zidane_exif_1.png", false, W, H);
    let (exp_w, exp_h, expected) = apply_exif_oracle(ref_w, ref_h, &reference, o);
    let (got_w, got_h, decoded) =
        decode_to_rgb(&format!("zidane_exif_{o}.png"), true, W.max(H), W.max(H));
    assert_eq!(
        (got_w, got_h),
        (exp_w, exp_h),
        "PNG orientation {o}: decoded dims differ from oracle"
    );
    let pixel_count = exp_w * exp_h * CH;
    let s = sad(&decoded[..pixel_count], &expected[..pixel_count]);
    assert_eq!(s, 0, "PNG orientation {o}: SAD={s} (expected 0)");
}

#[test]
fn png_exif_orientation_1() {
    check_png_orientation(1);
}

#[test]
fn png_exif_orientation_2() {
    check_png_orientation(2);
}

#[test]
fn png_exif_orientation_3() {
    check_png_orientation(3);
}

#[test]
fn png_exif_orientation_4() {
    check_png_orientation(4);
}

#[test]
fn png_exif_orientation_5() {
    check_png_orientation(5);
}

#[test]
fn png_exif_orientation_6() {
    check_png_orientation(6);
}

#[test]
fn png_exif_orientation_7() {
    check_png_orientation(7);
}

#[test]
fn png_exif_orientation_8() {
    check_png_orientation(8);
}
