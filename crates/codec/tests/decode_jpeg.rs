// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: JPEG decode into Mem tensors with various configurations.

use edgefirst_codec::{CodecError, DecodeOptions, ImageDecoder, ImageLoad};
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
    // person.jpg is a progressive JPEG — our baseline-only decoder rejects it
    let mut tensor =
        Tensor::<u8>::image(4256, 2532, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &jpeg,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(
        result.is_err(),
        "progressive JPEG should be rejected by baseline decoder"
    );
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

#[test]
fn decode_jpeg_u16() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<u16>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // u16 pixels should be scaled: u8 0→0, 255→65535
    let map = tensor.map().unwrap();
    let pixels: &[u16] = &map;
    let sample_count = info.width * 3;
    for &v in &pixels[..sample_count] {
        // All values should be multiples of 257 (u8 * 257 scaling)
        assert_eq!(v % 257, 0, "u16 pixel {v} is not a multiple of 257");
    }
    // Verify non-trivial content
    let nonzero = pixels[..sample_count].iter().filter(|&&v| v != 0).count();
    assert!(
        nonzero > 100,
        "expected many non-zero u16 pixels, got {nonzero}"
    );
}

#[test]
fn decode_jpeg_i8() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<i8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // i8 uses XOR trick: u8 0→-128, 128→0, 255→127
    let map = tensor.map().unwrap();
    let pixels: &[i8] = &map;
    let sample_count = info.width * 3;
    // Should have a range spanning negative and positive
    let min = pixels[..sample_count].iter().copied().min().unwrap();
    let max = pixels[..sample_count].iter().copied().max().unwrap();
    assert!(min < 0, "expected negative i8 pixels, min={min}");
    assert!(max > 0, "expected positive i8 pixels, max={max}");
}

#[test]
fn decode_jpeg_i16() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<i16>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
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

    // i16 uses XOR trick: u8 scale to u16 then XOR 0x8000
    let map = tensor.map().unwrap();
    let pixels: &[i16] = &map;
    let sample_count = info.width * 3;
    let min = pixels[..sample_count].iter().copied().min().unwrap();
    let max = pixels[..sample_count].iter().copied().max().unwrap();
    assert!(min < 0, "expected negative i16 pixels, min={min}");
    assert!(max > 0, "expected positive i16 pixels, max={max}");
}

#[test]
fn decode_jpeg_i8_xor_consistency() {
    // Verify that JPEG i8 decode matches manual u8→i8 XOR conversion
    let jpeg = testdata("zidane.jpg");

    let mut u8_tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut i8_tensor =
        Tensor::<i8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);

    u8_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();
    i8_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();

    let u8_map = u8_tensor.map().unwrap();
    let i8_map = i8_tensor.map().unwrap();
    let u8_pixels: &[u8] = &u8_map;
    let i8_pixels: &[i8] = &i8_map;

    // Check first 1000 pixels match the XOR trick
    for i in 0..1000 {
        let expected = (u8_pixels[i] ^ 0x80) as i8;
        assert_eq!(
            i8_pixels[i], expected,
            "pixel {i}: u8={}, i8={}, expected={}",
            u8_pixels[i], i8_pixels[i], expected
        );
    }
}

/// Compare our custom JPEG decoder output against the `image` crate pixel-by-pixel.
/// JPEG decoders may differ by ±1-2 LSBs due to IDCT rounding, so we allow
/// a per-pixel tolerance and compute mean absolute error (MAE).
#[test]
fn pixel_accuracy_vs_image_crate() {
    let jpeg_data = testdata("zidane.jpg");

    // Decode with our custom decoder
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg_data,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();

    // Decode with the `image` crate
    let ref_img = image::load_from_memory(&jpeg_data).unwrap().to_rgb8();
    let (ref_w, ref_h) = ref_img.dimensions();
    assert_eq!(info.width, ref_w as usize);
    assert_eq!(info.height, ref_h as usize);

    let map = tensor.map().unwrap();
    let our_pixels: &[u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };
    let ref_pixels = ref_img.as_raw();

    let w = info.width;
    let h = info.height;
    let stride = info.row_stride;
    let channels = 3;

    let mut max_diff: u32 = 0;
    let mut total_diff: u64 = 0;
    let mut pixel_count: u64 = 0;

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let our_val = our_pixels[y * stride + x * channels + c] as i32;
                let ref_val = ref_pixels[(y * w + x) * channels + c] as i32;
                let diff = (our_val - ref_val).unsigned_abs();
                max_diff = max_diff.max(diff);
                total_diff += diff as u64;
                pixel_count += 1;
            }
        }
    }

    let mae = total_diff as f64 / pixel_count as f64;
    eprintln!("Pixel accuracy: MAE={mae:.3}, max_diff={max_diff}, pixels={pixel_count}");

    // JPEG decoders commonly differ by ±2 IDCT LSBs, amplified by color
    // conversion to ±8-12. The IEEE 1180 IDCT conformance spec allows ±1
    // peak error per coefficient; after color conversion this compounds.
    assert!(
        max_diff <= 12,
        "max pixel difference {max_diff} exceeds tolerance 12"
    );
    assert!(
        mae < 0.5,
        "mean absolute error {mae:.3} exceeds tolerance 0.5"
    );
}

/// Compare our greyscale JPEG output against the `image` crate.
#[test]
fn pixel_accuracy_grey_vs_image_crate() {
    let jpeg_data = testdata("grey.jpg");

    let mut tensor =
        Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg_data,
            &DecodeOptions::default()
                .with_format(PixelFormat::Grey)
                .with_exif(false),
        )
        .unwrap();

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
/// The decoded image (1280×720) goes into a 1920×1080 tensor.
/// Verify padding bytes are untouched and decoded region is correct.
#[test]
fn decode_strided_oversized_tensor() {
    let jpeg = testdata("zidane.jpg");

    // Allocate larger than the image
    let mut tensor =
        Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();

    // Fill tensor with sentinel value
    {
        let mut map = tensor.map().unwrap();
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(map.as_mut_ptr(), map.len()) };
        bytes.fill(0xAA);
    }

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
    let stride = info.row_stride;

    let map = tensor.map().unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };

    // Verify decoded pixels are non-sentinel
    let decoded_row_bytes = 1280 * 3;
    for y in 0..720 {
        let row = &bytes[y * stride..y * stride + decoded_row_bytes];
        // At least some pixels should not be 0xAA
        let non_sentinel = row.iter().filter(|&&b| b != 0xAA).count();
        assert!(
            non_sentinel > 0,
            "row {y} appears to be all sentinel values"
        );
    }

    // Verify rows beyond the decoded height are untouched (sentinel)
    for y in 720..1080 {
        let row_start = y * stride;
        if row_start + decoded_row_bytes <= bytes.len() {
            let row = &bytes[row_start..row_start + decoded_row_bytes];
            assert!(
                row.iter().all(|&b| b == 0xAA),
                "row {y} should be untouched sentinel"
            );
        }
    }
}

#[test]
fn decode_truncated_jpeg() {
    let jpeg = testdata("zidane.jpg");
    let truncated = &jpeg[..100];
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        truncated,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_not_jpeg() {
    let mut bogus = testdata("zidane.png");
    bogus[..4].copy_from_slice(&[0xFF, 0xD8, 0xFF, 0xE0]);

    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &bogus,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_empty_data() {
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &[],
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_corrupt_markers() {
    let corrupt = [0xFF, 0xD8, 0xFF, 0x00];
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let result = tensor.load_image(
        &mut decoder,
        &corrupt,
        &DecodeOptions::default()
            .with_format(PixelFormat::Rgb)
            .with_exif(false),
    );
    assert!(matches!(result, Err(CodecError::InvalidData(_))));
}

#[test]
fn decode_bgra_format() {
    let jpeg = testdata("zidane.jpg");
    let mut rgb =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut bgra =
        Tensor::<u8>::image(1280, 720, PixelFormat::Bgra, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let rgb_info = rgb
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    let bgra_info = bgra
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Bgra)
                .with_exif(false),
        )
        .unwrap();

    assert_eq!(bgra_info.width, rgb_info.width);
    assert_eq!(bgra_info.height, rgb_info.height);
    assert_eq!(bgra_info.format, PixelFormat::Bgra);
    assert_eq!(bgra_info.row_stride, bgra_info.width * 4);

    let rgb_map = rgb.map().unwrap();
    let bgra_map = bgra.map().unwrap();
    let rgb_pixels: &[u8] = &rgb_map;
    let bgra_pixels: &[u8] = &bgra_map;
    let pixel_count = bgra_info.width * bgra_info.height;

    for i in 0..pixel_count {
        assert_eq!(bgra_pixels[i * 4], rgb_pixels[i * 3 + 2]);
        assert_eq!(bgra_pixels[i * 4 + 1], rgb_pixels[i * 3 + 1]);
        assert_eq!(bgra_pixels[i * 4 + 2], rgb_pixels[i * 3]);
        assert_eq!(bgra_pixels[i * 4 + 3], 255);
    }
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
        PixelFormat::Rgb,
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

    assert_eq!(info.width, width as usize);
    assert_eq!(info.height, height as usize);
    assert_eq!(info.format, PixelFormat::Rgb);
    assert_eq!(info.row_stride, info.width * PixelFormat::Rgb.channels());
}

#[test]
fn decode_grey_to_rgb() {
    let jpeg = testdata("grey.jpg");
    let mut grey =
        Tensor::<u8>::image(1024, 681, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut rgb =
        Tensor::<u8>::image(1024, 681, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let grey_info = grey
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Grey)
                .with_exif(false),
        )
        .unwrap();
    let rgb_info = rgb
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();

    assert_eq!(rgb_info.width, grey_info.width);
    assert_eq!(rgb_info.height, grey_info.height);
    assert_eq!(rgb_info.format, PixelFormat::Rgb);

    let grey_map = grey.map().unwrap();
    let rgb_map = rgb.map().unwrap();
    let grey_pixels: &[u8] = &grey_map;
    let rgb_pixels: &[u8] = &rgb_map;

    for y in 0..rgb_info.height {
        for x in 0..rgb_info.width {
            let grey_val = grey_pixels[y * grey_info.row_stride + x];
            let base = y * rgb_info.row_stride + x * 3;
            assert_eq!(rgb_pixels[base], grey_val);
            assert_eq!(rgb_pixels[base + 1], grey_val);
            assert_eq!(rgb_pixels[base + 2], grey_val);
        }
    }
}

#[test]
fn decode_with_exif_rotation() {
    use image::ImageDecoder as _;

    let jpeg = testdata("zidane_rotated_exif.jpg");
    let header =
        image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(jpeg.as_slice())).unwrap();
    let (stored_w, stored_h) = header.dimensions();
    let max_dim = stored_w.max(stored_h) as usize;

    let mut plain =
        Tensor::<u8>::image(max_dim, max_dim, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut oriented =
        Tensor::<u8>::image(max_dim, max_dim, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    let plain_info = plain
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(false),
        )
        .unwrap();
    let oriented_info = oriented
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Rgb)
                .with_exif(true),
        )
        .unwrap();

    assert_eq!(plain_info.width, stored_w as usize);
    assert_eq!(plain_info.height, stored_h as usize);
    assert!(
        (oriented_info.width, oriented_info.height) == (plain_info.width, plain_info.height)
            || (oriented_info.width, oriented_info.height) == (plain_info.height, plain_info.width),
        "EXIF-applied dimensions should match stored dims or their rotation"
    );

    let map = oriented.map().unwrap();
    let pixels: &[u8] = &map;
    let decoded_bytes = oriented_info.width * oriented_info.height * PixelFormat::Rgb.channels();
    let nonzero = pixels[..decoded_bytes].iter().filter(|&&v| v != 0).count();
    assert!(nonzero > 1000, "expected EXIF decode to produce image data");
}

#[test]
fn decode_u16_scaling_consistency() {
    let jpeg = testdata("zidane.jpg");
    let mut u8_tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut u16_tensor =
        Tensor::<u16>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);

    let u8_info = u8_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();
    let u16_info = u16_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();

    assert_eq!(u8_info.width, u16_info.width);
    assert_eq!(u8_info.height, u16_info.height);

    let u8_map = u8_tensor.map().unwrap();
    let u16_map = u16_tensor.map().unwrap();
    let u8_pixels: &[u8] = &u8_map;
    let u16_pixels: &[u16] = &u16_map;
    let u16_stride = u16_info.row_stride / std::mem::size_of::<u16>();

    for y in 0..u8_info.height {
        for x in 0..u8_info.width * 3 {
            let u8_val = u8_pixels[y * u8_info.row_stride + x];
            let u16_val = u16_pixels[y * u16_stride + x];
            assert_eq!(u16_val, u16::from(u8_val) * 257);
        }
    }
}

#[test]
fn decode_f32_scaling_consistency() {
    let jpeg = testdata("zidane.jpg");
    let mut u8_tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut f32_tensor =
        Tensor::<f32>::image(1280, 720, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let opts = DecodeOptions::default()
        .with_format(PixelFormat::Rgb)
        .with_exif(false);

    let u8_info = u8_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();
    let f32_info = f32_tensor.load_image(&mut decoder, &jpeg, &opts).unwrap();

    assert_eq!(u8_info.width, f32_info.width);
    assert_eq!(u8_info.height, f32_info.height);

    let u8_map = u8_tensor.map().unwrap();
    let f32_map = f32_tensor.map().unwrap();
    let u8_pixels: &[u8] = &u8_map;
    let f32_pixels: &[f32] = &f32_map;
    let f32_stride = f32_info.row_stride / std::mem::size_of::<f32>();

    for y in 0..u8_info.height {
        for x in 0..u8_info.width * 3 {
            let u8_val = u8_pixels[y * u8_info.row_stride + x];
            let expected = f32::from(u8_val) / 255.0;
            let actual = f32_pixels[y * f32_stride + x];
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn decode_stride_padding_untouched() {
    let jpeg = testdata("zidane.jpg");
    let mut tensor =
        Tensor::<u8>::image(1920, 1080, PixelFormat::Rgb, Some(TensorMemory::Mem)).unwrap();

    {
        let mut map = tensor.map().unwrap();
        let bytes: &mut [u8] = &mut map;
        bytes.fill(0x5A);
    }

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

    let decoded_row_bytes = info.width * PixelFormat::Rgb.channels();
    let map = tensor.map().unwrap();
    let bytes: &[u8] = &map;

    for y in 0..info.height {
        let row = &bytes[y * info.row_stride..(y + 1) * info.row_stride];
        assert!(
            row[..decoded_row_bytes].iter().any(|&b| b != 0x5A),
            "decoded portion of row {y} should contain image data"
        );
        assert!(
            row[decoded_row_bytes..].iter().all(|&b| b == 0x5A),
            "padding after decoded width in row {y} should remain untouched"
        );
    }

    for y in info.height..1080 {
        let row = &bytes[y * info.row_stride..(y + 1) * info.row_stride];
        assert!(
            row.iter().all(|&b| b == 0x5A),
            "rows past decoded height should remain untouched"
        );
    }
}

#[test]
fn decode_nv12_output() {
    let jpeg = testdata("zidane.jpg");
    // NV12 needs H * W + H/2 * W = 1.5 * H * W bytes.
    // For 1280×720: Y plane = 921600, UV plane = 460800, total = 1382400
    let width = 1280usize;
    let height = 720usize;
    let nv12_size = width * height * 3 / 2;

    // Create a tensor large enough for NV12 (1 channel, height*1.5 rows)
    let mut tensor = Tensor::<u8>::image(
        width,
        height * 3 / 2,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();

    let info = tensor
        .load_image(
            &mut decoder,
            &jpeg,
            &DecodeOptions::default()
                .with_format(PixelFormat::Nv12)
                .with_exif(false),
        )
        .unwrap();
    assert_eq!(info.width, width);
    assert_eq!(info.height, height);
    assert_eq!(info.format, PixelFormat::Nv12);

    let map = tensor.map().unwrap();
    let bytes: &[u8] = &map;

    // Y plane should have non-zero data
    let y_nonzero = bytes[..width * height].iter().filter(|&&v| v != 0).count();
    assert!(
        y_nonzero > 1000,
        "Y plane should have many non-zero pixels, got {y_nonzero}"
    );

    // UV plane should have non-128 data (chrominance varies in real images)
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

    let _ = nv12_size;
}
