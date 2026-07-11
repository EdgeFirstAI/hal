// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! End-to-end EXIF orientation reporting for both JPEG and PNG.
//!
//! The 16 fixtures `testdata/zidane_exif_<N>.{jpg,png}` for N in 1..=8 are
//! produced by `scripts/generate_exif_fixtures.py` and carry the same pixel
//! data with only the EXIF orientation tag varying.
//!
//! The codec **reports** the EXIF orientation in [`ImageInfo`] but never rotates
//! pixels: `info.width`/`info.height` are always the source's true (unrotated)
//! dimensions, and `info.rotation_degrees`/`info.flip_horizontal` carry the
//! geometry the consumer should apply downstream.
//!
//! EXIF orientation → (rotation_degrees, flip_horizontal) mapping:
//!
//!   1 → (0,   false)   Identity
//!   2 → (0,   true)    Mirror horizontal
//!   3 → (180, false)   Rotate 180°
//!   4 → (180, true)    Mirror vertical
//!   5 → (270, true)    Transpose (90° CW + mirror)
//!   6 → (90,  false)   Rotate 90° CW
//!   7 → (90,  true)    Transverse (90° CCW + mirror)
//!   8 → (270, false)   Rotate 90° CCW

use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory};
use std::path::PathBuf;

const W: usize = 1280;
const H: usize = 720;

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

/// Expected `(rotation_degrees, flip_horizontal)` for EXIF orientation tag `o`.
fn expected_orientation(o: u32) -> (u16, bool) {
    match o {
        1 => (0, false),
        2 => (0, true),
        3 => (180, false),
        4 => (180, true),
        5 => (270, true),
        6 => (90, false),
        7 => (90, true),
        8 => (270, false),
        _ => unreachable!("orientation {o} out of range"),
    }
}

// ---------------------------------------------------------------------------
// JPEG — colour JPEGs decode to NV12; dims are the unrotated source dims.
// ---------------------------------------------------------------------------

fn check_jpeg_orientation(o: u32) {
    let data = testdata(&format!("zidane_exif_{o}.jpg"));

    // peek reports the source's true (unrotated) dims and the EXIF orientation.
    let peeked = peek_info(&data).unwrap();
    assert_eq!(
        (peeked.width, peeked.height),
        (W, H),
        "orientation {o}: peek dims should be unrotated source dims"
    );
    assert_eq!(peeked.format, PixelFormat::Nv12, "colour JPEG → NV12");
    let (exp_rot, exp_flip) = expected_orientation(o);
    assert_eq!(
        (peeked.rotation_degrees, peeked.flip_horizontal),
        (exp_rot, exp_flip),
        "orientation {o}: peek orientation mismatch"
    );

    // A full decode reports the same dims and orientation; the codec never
    // rotates, so the destination tensor is sized for the source dims.
    let mut tensor = Tensor::<u8>::image(
        W,
        H,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(&mut decoder, &data)
        .unwrap_or_else(|e| panic!("decode orientation {o}: {e}"));
    assert_eq!(
        (info.width, info.height),
        (W, H),
        "orientation {o}: decode dims should be unrotated source dims"
    );
    assert_eq!(info.format, PixelFormat::Nv12);
    assert_eq!(
        (info.rotation_degrees, info.flip_horizontal),
        (exp_rot, exp_flip),
        "orientation {o}: decode orientation mismatch"
    );
}

#[test]
fn jpeg_exif_orientation_1() {
    check_jpeg_orientation(1);
}

#[test]
fn jpeg_exif_orientation_2() {
    check_jpeg_orientation(2);
}

#[test]
fn jpeg_exif_orientation_3() {
    check_jpeg_orientation(3);
}

#[test]
fn jpeg_exif_orientation_4() {
    check_jpeg_orientation(4);
}

#[test]
fn jpeg_exif_orientation_5() {
    check_jpeg_orientation(5);
}

#[test]
fn jpeg_exif_orientation_6() {
    check_jpeg_orientation(6);
}

#[test]
fn jpeg_exif_orientation_7() {
    check_jpeg_orientation(7);
}

#[test]
fn jpeg_exif_orientation_8() {
    check_jpeg_orientation(8);
}

// ---------------------------------------------------------------------------
// PNG — native RGB; dims are the unrotated source dims.
// ---------------------------------------------------------------------------

fn check_png_orientation(o: u32) {
    let data = testdata(&format!("zidane_exif_{o}.png"));

    let peeked = peek_info(&data).unwrap();
    assert_eq!(
        (peeked.width, peeked.height),
        (W, H),
        "PNG orientation {o}: peek dims should be unrotated source dims"
    );
    assert_eq!(peeked.format, PixelFormat::Rgb, "PNG → RGB");
    let (exp_rot, exp_flip) = expected_orientation(o);
    assert_eq!(
        (peeked.rotation_degrees, peeked.flip_horizontal),
        (exp_rot, exp_flip),
        "PNG orientation {o}: peek orientation mismatch"
    );

    let mut tensor = Tensor::<u8>::image(
        W,
        H,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(&mut decoder, &data)
        .unwrap_or_else(|e| panic!("decode PNG orientation {o}: {e}"));
    assert_eq!(
        (info.width, info.height),
        (W, H),
        "PNG orientation {o}: decode dims should be unrotated source dims"
    );
    assert_eq!(info.format, PixelFormat::Rgb);
    assert_eq!(
        (info.rotation_degrees, info.flip_horizontal),
        (exp_rot, exp_flip),
        "PNG orientation {o}: decode orientation mismatch"
    );
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
