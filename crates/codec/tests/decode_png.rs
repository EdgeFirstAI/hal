// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: PNG decode into Mem tensors.

use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};

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

#[test]
fn decode_png_u16() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<u16>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // zidane.png is 8-bit RGB, so u16 values should be multiples of 257
    let map = tensor.map().unwrap();
    let pixels: &[u16] = &map;
    let sample_count = info.width * 3;
    let nonzero = pixels[..sample_count].iter().filter(|&&v| v != 0).count();
    assert!(
        nonzero > 100,
        "expected many non-zero u16 pixels, got {nonzero}"
    );
}

#[test]
fn decode_png_i8() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<i8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // i8 via XOR: should span negative and positive values
    let map = tensor.map().unwrap();
    let pixels: &[i8] = &map;
    let sample_count = info.width * 3;
    let min = pixels[..sample_count].iter().copied().min().unwrap();
    let max = pixels[..sample_count].iter().copied().max().unwrap();
    assert!(min < 0, "expected negative i8 pixels, min={min}");
    assert!(max > 0, "expected positive i8 pixels, max={max}");
}

#[test]
fn decode_png_i16() {
    let png = testdata("zidane.png");
    let mut tensor =
        Tensor::<i16>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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
    let pixels: &[i16] = &map;
    let sample_count = info.width * 3;
    let min = pixels[..sample_count].iter().copied().min().unwrap();
    let max = pixels[..sample_count].iter().copied().max().unwrap();
    assert!(min < 0, "expected negative i16 pixels, min={min}");
    assert!(max > 0, "expected positive i16 pixels, max={max}");
}

#[test]
fn decode_png_i8_xor_consistency() {
    // Verify PNG i8 decode matches manual u8→i8 XOR conversion
    let png = testdata("zidane.png");

    let mut u8_tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut i8_tensor =
        Tensor::<i8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default().with_format(PixelFormat::Rgb);

    u8_tensor.load_image(&mut decoder, &png, &opts).unwrap();
    i8_tensor.load_image(&mut decoder, &png, &opts).unwrap();

    let u8_map = u8_tensor.map().unwrap();
    let i8_map = i8_tensor.map().unwrap();
    let u8_pixels: &[u8] = &u8_map;
    let i8_pixels: &[i8] = &i8_map;

    for i in 0..1000 {
        let expected = (u8_pixels[i] ^ 0x80) as i8;
        assert_eq!(
            i8_pixels[i], expected,
            "pixel {i}: u8={}, i8={}, expected={}",
            u8_pixels[i], i8_pixels[i], expected
        );
    }
}
