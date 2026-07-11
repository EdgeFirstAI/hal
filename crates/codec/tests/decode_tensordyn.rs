// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration test: TensorDyn decode path.
//!
//! Colour JPEGs decode to NV12 (u8 only). Non-u8 dtypes are rejected for JPEG
//! with [`CodecError::UnsupportedDtype`]. PNG still supports the wide types, so
//! the wide-dtype decode coverage uses a PNG fixture.

use edgefirst_codec::{CodecError, ImageDecoder, ImageLoad};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMemory};

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
fn decode_tensordyn_u8_nv12() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Nv12);
}

#[test]
fn decode_tensordyn_jpeg_f32_unsupported() {
    // NV12 is a u8 format; an f32 destination for a JPEG decode is rejected.
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Nv12,
        DType::F32,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    assert!(
        matches!(result, Err(CodecError::UnsupportedDtype(_))),
        "expected UnsupportedDtype for f32 JPEG decode, got {result:?}"
    );
}

#[test]
fn decode_tensordyn_unsupported_dtype() {
    // I32 is not a valid image decode dtype for either codec path.
    let jpeg = testdata("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Nv12,
        DType::I32,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    assert!(result.is_err());
}

#[test]
fn decode_tensordyn_png_f32() {
    // PNG still supports wide dtypes; exercise the TensorDyn F32 PNG decode.
    let png = testdata("zidane.png");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::F32,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &png).unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgb);
}

#[test]
fn decode_tensordyn_png_i16() {
    let png = testdata("zidane.png");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Rgb,
        DType::I16,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &png).unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Rgb);
}

#[test]
fn decode_file_path() {
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
    let path = root.join("zidane.jpg");
    let mut tensor = TensorDyn::image(
        1280,
        720,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image_file(&mut decoder, &path).unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Nv12);
}
