// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration test: TensorDyn decode path.

use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMemory};

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
fn decode_tensordyn_u8() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
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
}

#[test]
fn decode_tensordyn_f32() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::F32,
        Some(TensorMemory::Mem),
    )
    .unwrap();
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
}

#[test]
fn decode_tensordyn_unsupported_dtype() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::I8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &jpeg,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(result.is_err());
}

#[test]
fn decode_file_path() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("testdata")
        .join("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image_file(
            &mut decoder,
            &path,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
}
