// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: PNG decode into Mem tensors.

use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};

fn testdata(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("testdata")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

#[test]
fn decode_zidane_png_rgb() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &png,
            &DecodeOptions::default().with_format(PixelFormat::Rgb),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgb);

    let map = tensor.map().unwrap();
    let pixels: &[u8] = &map;
    let nonzero = pixels[..info.width * info.height * 3]
        .iter()
        .filter(|&&v| v != 0)
        .count();
    assert!(
        nonzero > 1000,
        "expected many non-zero pixels, got {nonzero}"
    );
}

#[test]
fn decode_zidane_png_rgba() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &png,
            &DecodeOptions::default().with_format(PixelFormat::Rgba),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgba);
}

#[test]
fn decode_png_f32() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<f32>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &png,
            &DecodeOptions::default().with_format(PixelFormat::Rgb),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);

    let map = tensor.map().unwrap();
    let pixels: &[f32] = &map;
    for &v in &pixels[..info.width * 3] {
        assert!((0.0..=1.0).contains(&v), "pixel value {v} out of range");
    }
}

#[test]
fn decode_png_into_larger_tensor() {
    let png = testdata("zidane.png"); // 1280×720
    let mut tensor =
        Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &png,
            &DecodeOptions::default().with_format(PixelFormat::Rgb),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert!(info.row_stride >= 1920 * 3);
}

#[test]
fn decode_png_capacity_error() {
    let png = testdata("zidane.png"); // 1280×720
    let mut tensor =
        Tensor::<u8>::image(640, 480, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &png,
        &DecodeOptions::default().with_format(PixelFormat::Rgb),
    );
    assert!(result.is_err());
}
