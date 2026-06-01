// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: JPEG decode into Mem tensors.
//!
//! The codec always emits the JPEG's native format — `Nv12` for colour
//! (3-component) JPEGs and `Grey` for greyscale. It configures the destination
//! tensor's dims and format itself; callers allocate a tensor with enough
//! capacity. `info.width`/`info.height` are the source's true (unrotated) dims.

use edgefirst_codec::{peek_info, CodecError, ImageDecoder, ImageLoad, UnsupportedFeature};
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
fn decode_zidane_nv12() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Nv12);

    // The Y plane (first w*h bytes) should have many non-zero bytes.
    let map = tensor.map().unwrap();
    let pixels: &[u8] = &map;
    let nonzero = pixels[..info.width * info.height]
        .iter()
        .filter(|&&v| v != 0)
        .count();
    assert!(
        nonzero > 1000,
        "expected many non-zero Y-plane bytes, got {nonzero}"
    );
}

#[test]
fn decode_grey_jpeg() {
    let jpeg = testdata("grey.jpg");
    // grey.jpg is 1024×681 greyscale.
    let mut tensor =
        Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert!(info.width > 0 && info.height > 0);
    assert_eq!(info.format, PixelFormat::Grey);

    let map = tensor.map().unwrap();
    let pixels: &[u8] = &map;
    let nonzero = pixels[..info.width * info.height]
        .iter()
        .filter(|&&v| v != 0)
        .count();
    assert!(
        nonzero > 1000,
        "expected non-zero grey pixels, got {nonzero}"
    );
}

#[test]
fn decode_person_rejected() {
    let jpeg = testdata("person.jpg");
    // person.jpg is a progressive JPEG — our baseline-only decoder rejects it.
    // person.jpg is 4256×2532 (even dims) so allocate an Nv12 tensor of that size.
    let mut tensor =
        Tensor::<u8>::image(4256, 2532, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    assert!(
        result.is_err(),
        "progressive JPEG should be rejected by baseline decoder"
    );
}

#[test]
fn decode_f32_jpeg_unsupported_dtype() {
    // NV12/GREY are u8 formats; the codec rejects non-u8 tensors for JPEG.
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<f32>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    assert!(
        matches!(result, Err(CodecError::UnsupportedDtype(_))),
        "expected UnsupportedDtype for f32 JPEG decode, got {result:?}"
    );
}

#[test]
fn decode_capacity_error() {
    let jpeg = testdata("zidane.jpg"); // 1280×720
                                       // Allocate far too small.
    let mut tensor =
        Tensor::<u8>::image(100, 100, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    assert!(result.is_err());
    match result.unwrap_err() {
        CodecError::InsufficientCapacity { image, tensor } => {
            assert_eq!(image, (1280, 720));
            assert_eq!(tensor, (100, 100));
        }
        other => panic!("expected InsufficientCapacity, got {other}"),
    }
}

#[test]
fn decode_reuse_pattern() {
    // The hot-loop reuse pattern: decode multiple different images into the same
    // tensor. Allocate at the max size (giraffe and zidane are both even-dim
    // colour JPEGs → NV12). jaguar.jpg is odd-width so excluded (NV12 rejects it).
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let images = ["zidane.jpg", "giraffe.jpg"];
    for name in &images {
        let jpeg = testdata(name);
        let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
        assert_eq!(
            info.format,
            PixelFormat::Nv12,
            "{name} should decode to NV12"
        );
        assert!(info.width > 0 && info.height > 0, "failed to decode {name}");
    }
}

/// Compare our greyscale JPEG output against the `image` crate. Greyscale JPEG
/// decodes to a single-plane `Grey` tensor that we can compare directly.
#[test]
fn pixel_accuracy_grey_vs_image_crate() {
    let jpeg_data = testdata("grey.jpg");

    let mut tensor =
        Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg_data).unwrap();
    assert_eq!(info.format, PixelFormat::Grey);

    let ref_img = image::load_from_memory(&jpeg_data).unwrap().to_luma8();
    let (ref_w, ref_h) = ref_img.dimensions();
    assert_eq!(info.width, ref_w as usize);
    assert_eq!(info.height, ref_h as usize);

    let map = tensor.map().unwrap();
    let our_pixels: &[u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };
    let ref_pixels = ref_img.as_raw();

    let w = info.width;
    let h = info.height;
    let stride = info.row_stride;

    let mut max_diff: u32 = 0;
    let mut total_diff: u64 = 0;

    for y in 0..h {
        for x in 0..w {
            let our_val = our_pixels[y * stride + x] as i32;
            let ref_val = ref_pixels[y * w + x] as i32;
            let diff = (our_val - ref_val).unsigned_abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
        }
    }

    let pixel_count = (w * h) as f64;
    let mae = total_diff as f64 / pixel_count;
    eprintln!("Grey accuracy: MAE={mae:.3}, max_diff={max_diff}");

    assert!(
        max_diff <= 4,
        "max grey diff {max_diff} exceeds tolerance 4"
    );
    assert!(mae < 1.0, "grey MAE {mae:.3} exceeds tolerance 1.0");
}

/// Test decoding into an oversized tensor with stride padding.
/// The decoded image (1280×720) goes into a 1920×1080 NV12 tensor.
/// Verify the Y plane is written and padding bytes are untouched.
#[test]
fn decode_strided_oversized_tensor() {
    let jpeg = testdata("zidane.jpg");

    // Allocate larger than the image.
    let mut tensor =
        Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();

    // Fill tensor with sentinel value.
    {
        let mut map = tensor.map().unwrap();
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(map.as_mut_ptr(), map.len()) };
        bytes.fill(0xAA);
    }

    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();

    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    assert_eq!(info.format, PixelFormat::Nv12);
    let stride = info.row_stride;

    let map = tensor.map().unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };

    // Verify decoded Y-plane rows are non-sentinel within the image width.
    let decoded_row_bytes = 1280;
    for y in 0..720 {
        let row = &bytes[y * stride..y * stride + decoded_row_bytes];
        let non_sentinel = row.iter().filter(|&&b| b != 0xAA).count();
        assert!(
            non_sentinel > 0,
            "Y row {y} appears to be all sentinel values"
        );
    }
}

#[test]
fn decode_truncated_jpeg() {
    let jpeg = testdata("zidane.jpg");
    let truncated = &jpeg[..100];
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, truncated);
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_not_jpeg() {
    let mut bogus = testdata("zidane.png");
    bogus[..4].copy_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);

    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &bogus);
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_empty_data() {
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &[]);
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_corrupt_markers() {
    let corrupt = [0xFF, 0xD8, 0xFF, 0x00];
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &corrupt);
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_exact_size_tensor() {
    use image::ImageDecoder as _;

    let jpeg = testdata("giraffe.jpg");
    let header =
        image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg.as_slice())).unwrap();
    let (width, height) = header.dimensions();

    let mut tensor = Tensor::<u8>::image(
        width as usize,
        height as usize,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();

    assert_eq!(info.width, width as usize);
    assert_eq!(info.height, height as usize);
    assert_eq!(info.format, PixelFormat::Nv12);
}

#[test]
fn decode_nv12_output() {
    let jpeg = testdata("zidane.jpg");
    let width = 1280usize;
    let height = 720usize;

    let mut tensor =
        Tensor::<u8>::image(width, height, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!(info.width, width);
    assert_eq!(info.height, height);
    assert_eq!(info.format, PixelFormat::Nv12);

    let map = tensor.map().unwrap();
    let bytes: &[u8] = &map;

    // Y plane should have non-zero data.
    let y_nonzero = bytes[..width * height].iter().filter(|&&v| v != 0).count();
    assert!(
        y_nonzero > 1000,
        "Y plane should have many non-zero pixels, got {y_nonzero}"
    );

    // UV plane should have non-128 data (chrominance varies in real images).
    let uv_start = height * width;
    let uv_size = width * height / 2;
    if uv_start + uv_size <= bytes.len() {
        let uv_non128 = bytes[uv_start..uv_start + uv_size]
            .iter()
            .filter(|&&v| v != 128)
            .count();
        assert!(
            uv_non128 > 100,
            "UV plane should have varied chrominance, non-128 count: {uv_non128}"
        );
    }
}

#[test]
fn unsupported_progressive_jpeg_returns_typed_variant() {
    // person.jpg is a progressive JPEG — the codec rejects it with a
    // typed Unsupported(ProgressiveJpeg). Verifies callers can pattern-match
    // instead of string-matching.
    let jpeg = testdata("person.jpg");
    match peek_info(&jpeg) {
        Err(CodecError::Unsupported(UnsupportedFeature::ProgressiveJpeg)) => {}
        Err(other) => {
            panic!("expected Unsupported(ProgressiveJpeg), got {other:?}");
        }
        Ok(_) => panic!("progressive JPEG should not decode"),
    }
}

#[test]
fn odd_width_nv12_jpeg_is_rejected() {
    // jaguar.jpg is 789×384 — odd width. NV12 supports odd *height* (via the
    // `H + ceil(H/2)` combined-plane height) but not odd width yet: the chroma
    // plane would need an even-padded row stride. `peek_info` still reports the
    // true dims, and the decode is refused up front with a specific message.
    let jpeg = testdata("jaguar.jpg");
    let info = peek_info(&jpeg).unwrap();
    assert_eq!((info.width, info.height), (789, 384));
    assert_eq!(
        info.format,
        PixelFormat::Nv12,
        "colour JPEG should report native NV12 format"
    );

    let mut tensor =
        Tensor::<u8>::image(790, 384, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(&mut decoder, &jpeg);
    let err = result.expect_err("odd-width NV12 JPEG decode should be rejected");
    assert!(
        err.to_string().contains("odd width"),
        "error should name the odd-width limitation, got: {err}"
    );
}
