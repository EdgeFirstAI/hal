// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: JPEG decode into Mem tensors with various configurations.

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
fn decode_zidane_rgb() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgb);

    // Verify pixels are non-zero (not a blank decode)
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
fn decode_zidane_rgba() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgba, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgba)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgba);
}

#[test]
fn decode_grey_jpeg() {
    let jpeg = testdata("grey.jpg");
    // grey.jpg is 1024×681
    let mut tensor =
        Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Grey)
                .with_exif(false),
        )
        .unwrap();
    assert!(info.width > 0 && info.height > 0);
    assert_eq!(info.format, PixelFormat::Grey);
}

#[test]
fn decode_person_rgb() {
    let jpeg = testdata("person.jpg");
    // person.jpg is 4256×2532 — allocate larger
    let mut tensor =
        Tensor::<u8>::image(4256, 2532, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, 4256);
    assert_eq!(info.height, 2532);
}

#[test]
fn decode_f32_jpeg() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<f32>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);

    // Verify f32 values are in [0.0, 1.0]
    let map = tensor.map().unwrap();
    let pixels: &[f32] = &map;
    let sample_count = info.width * 3; // first row
    for &v in &pixels[..sample_count] {
        assert!((0.0..=1.0).contains(&v), "pixel value {v} out of range");
    }
}

#[test]
fn decode_capacity_error() {
    let jpeg = testdata("zidane.jpg"); // 1280×720
                                       // Allocate too small
    let mut tensor =
        Tensor::<u8>::image(100, 100, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &jpeg,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        edgefirst_codec::CodecError::InsufficientCapacity { image, tensor } => {
            assert_eq!(image, (1280, 720));
            assert_eq!(tensor, (100, 100));
        }
        other => panic!("expected InsufficientCapacity, got {other}"),
    }
}

#[test]
fn decode_reuse_pattern() {
    // The hot-loop reuse pattern: decode multiple different images into the same tensor
    // Allocate at max size (person.jpg is 4256×2532)
    let mut tensor =
        Tensor::<u8>::image(4256, 2532, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let images = ["zidane.jpg", "giraffe.jpg", "jaguar.jpg"];
    for name in &images {
        let jpeg = testdata(name);
        let info = tensor
            .load_image(
                &mut decoder,
                &jpeg,
                &DecodeOptions::default()
                    .with_format(PixelFormat::Rgb)
                    .with_exif(false),
            )
            .unwrap();
        assert!(info.width > 0 && info.height > 0, "failed to decode {name}");
    }
}
