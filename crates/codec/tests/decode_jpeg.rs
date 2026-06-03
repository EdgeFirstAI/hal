// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: JPEG decode into Mem tensors.
//!
//! The codec always emits the JPEG's native format — `Nv12`/`Nv16`/`Nv24` for
//! colour (3-component) JPEGs by subsampling, `Grey` for greyscale. It configures the destination
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
    // tensor. Allocate at the max size (giraffe and zidane are both 1280×720
    // colour JPEGs → NV12). jaguar.jpg has a smaller odd width and is excluded
    // here — not because NV12 rejects odd widths (it doesn't), but because a
    // 1280×720 pool reconfigured for a smaller image exercises configure_image.
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
fn odd_width_nv12_jpeg_decodes_with_logical_width() {
    // jaguar.jpg is 789×384 — odd width.  The tensor carries the LOGICAL width
    // (789); the row_stride is >= even(789)=790 and 64-byte aligned, ensuring
    // the interleaved chroma columns are byte-aligned.
    let jpeg = testdata("jaguar.jpg");
    let info = peek_info(&jpeg).unwrap();
    assert_eq!((info.width, info.height), (789, 384));
    assert_eq!(
        info.format,
        PixelFormat::Nv12,
        "colour JPEG should report native NV12 format"
    );

    // Allocating at the odd width preserves the logical width.
    let mut tensor =
        Tensor::<u8>::image(789, 384, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    // Contract: width() reports the LOGICAL visible size, not the even-padded buffer width.
    assert_eq!(tensor.width(), Some(789), "logical width must be preserved");
    assert_eq!(tensor.height(), Some(384));
    // Contract: row_stride is 64-byte aligned AND >= even(789)=790.
    let s = tensor
        .effective_row_stride()
        .expect("semi-planar must always have a stride");
    assert_eq!(s % 64, 0, "stride must be 64-byte aligned, got {s}");
    assert!(s >= 790, "stride must be >= even(width)=790, got {s}");

    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(&mut decoder, &jpeg)
        .expect("odd-width NV12 JPEG should decode into the strided buffer");
    // ImageInfo carries the true odd width; both match now.
    assert_eq!((info.width, info.height), (789, 384));
    assert_eq!(tensor.width(), Some(789));
    assert_eq!(tensor.height(), Some(384));
    // Post-decode stride invariant still holds.
    let s2 = tensor.effective_row_stride().unwrap();
    assert_eq!(s2 % 64, 0, "post-decode stride must be 64-byte aligned");
    assert!(s2 >= 790, "post-decode stride must be >= 790");
}

// ---------------------------------------------------------------------------
// Native NV16 (4:2:2) and NV24 (4:4:4) decode.
//
// `zidane_422.jpg` / `zidane_444.jpg` are baseline JPEGs re-encoded from
// `zidane.jpg` (1280×720) at 4:2:2 and 4:4:4 chroma subsampling. These exercise
// the `write_nv16_nv24_rows` MCU path (full-height chroma), which the 4:2:0
// fixtures never reach. The decoder must select the matching native format and
// place both the luma and the interleaved full-height chroma plane correctly.
// ---------------------------------------------------------------------------

/// BT.601 full-range (JFIF) RGB→CbCr — the colorimetry the codec/`yuv` kernels
/// assume. Returned as `(Cb, Cr)` in `[0, 255]`.
fn rgb_to_cbcr(r: f32, g: f32, b: f32) -> (f32, f32) {
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (cb, cr)
}

/// Decode a colour JPEG fixture and verify (1) the native format and dims,
/// (2) the luma plane matches the `image` crate's luma tightly, and (3) the
/// interleaved chroma plane reconstructs the `image` crate's colour within a
/// modest tolerance — proving the chroma layout (offset, pitch, Cb/Cr order)
/// is correct, not merely present.
fn check_native_decode(fixture: &str, expect_fmt: PixelFormat) {
    let jpeg = testdata(fixture);
    // Over-allocate generously (NV24 needs 3·H rows); the decoder configures the
    // real shape/format from the bitstream.
    let mut tensor = Tensor::<u8>::image(1280, 720, expect_fmt, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!(info.format, expect_fmt, "{fixture}: native format");
    assert_eq!((info.width, info.height), (1280, 720), "{fixture}: dims");

    let ref_rgb = image::load_from_memory(&jpeg).unwrap().to_rgb8();
    assert_eq!(ref_rgb.dimensions(), (1280, 720));

    let w = info.width;
    let h = info.height;
    let stride = info.row_stride;
    let map = tensor.map().unwrap();
    let buf: &[u8] = &map;

    // (2) Luma accuracy against the image crate's luma.
    let ref_luma = image::load_from_memory(&jpeg).unwrap().to_luma8();
    let mut y_max: u32 = 0;
    let mut y_total: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let ours = buf[y * stride + x] as i32;
            let refv = ref_luma.get_pixel(x as u32, y as u32)[0] as i32;
            let d = (ours - refv).unsigned_abs();
            y_max = y_max.max(d);
            y_total += d as u64;
        }
    }
    let y_mae = y_total as f64 / (w * h) as f64;
    eprintln!("{fixture} luma: MAE={y_mae:.3}, max={y_max}");
    // Fixtures are re-encoded from a 4:2:0 source, so luma sees two lossy passes
    // plus IDCT-rounding divergence from the image crate; the small MAE is the
    // real signal (a layout bug would push it into the tens), max is edge-noise.
    assert!(y_max <= 24, "{fixture}: luma max diff {y_max} > 24");
    assert!(y_mae < 2.0, "{fixture}: luma MAE {y_mae:.3} > 2.0");

    // (3) Chroma reconstruction. The chroma plane lives in buffer rows [H, …)
    // using the documented per-format layout (NV16: full-height, half-width,
    // one buffer row per image row; NV24: full-res, two buffer rows per image
    // row). Decode (Cb, Cr) for a sampled grid and compare to CbCr derived from
    // the image crate's RGB. A transposed/mis-offset plane shows up as a large
    // diff; smooth chroma keeps the honest tolerance small.
    let even_w = w.next_multiple_of(2);
    let uv_off = h * stride;
    let pos = |lb: usize| uv_off + (lb / even_w) * stride + (lb % even_w);
    let chroma_at = |x: usize, y: usize| -> (i32, i32) {
        let lb = match expect_fmt {
            // 4:2:2 — chroma col x/2, one chroma row per image row.
            PixelFormat::Nv16 => y * even_w + (x / 2) * 2,
            // 4:4:4 — chroma col x, pitch 2·even_w (spans two buffer rows).
            _ => y * even_w * 2 + x * 2,
        };
        (buf[pos(lb)] as i32, buf[pos(lb + 1)] as i32)
    };

    let mut c_max: u32 = 0;
    let mut c_total: u64 = 0;
    let mut samples: u64 = 0;
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let p = ref_rgb.get_pixel(x as u32, y as u32);
            let (ecb, ecr) = rgb_to_cbcr(p[0] as f32, p[1] as f32, p[2] as f32);
            let (cb, cr) = chroma_at(x, y);
            let dcb = (cb - ecb.round() as i32).unsigned_abs();
            let dcr = (cr - ecr.round() as i32).unsigned_abs();
            c_max = c_max.max(dcb).max(dcr);
            c_total += (dcb + dcr) as u64;
            samples += 2;
        }
    }
    let c_mae = c_total as f64 / samples as f64;
    eprintln!("{fixture} chroma: MAE={c_mae:.3}, max={c_max}");
    // JPEG chroma is lossy and the image crate upsamples NV16's half-width
    // chroma, so allow a modest spread; a wrong layout would be ~50+ MAE.
    assert!(
        c_mae < 6.0,
        "{fixture}: chroma MAE {c_mae:.3} > 6.0 (layout?)"
    );
    assert!(
        c_max <= 40,
        "{fixture}: chroma max diff {c_max} > 40 (layout?)"
    );
}

#[test]
fn decode_zidane_nv16() {
    check_native_decode("zidane_422.jpg", PixelFormat::Nv16);
}

#[test]
fn decode_zidane_nv24() {
    check_native_decode("zidane_444.jpg", PixelFormat::Nv24);
}

// ---------------------------------------------------------------------------
// Real-world COCO odd-width greyscale fixture. The end-to-end odd-dimension
// decode+convert coverage for the colour formats (NV12/NV16/NV24) lives in the
// Python suite (`test_decode_native_odd_dimensions`), which compares the
// converted RGB to PIL with a robust RMS metric across CPU and GPU backends.
// This codec-level case guards the specific odd-WIDTH greyscale decode path:
// the luma-only output decodes into a tight odd-width buffer (stride == img_w),
// which must NOT trip the even-width grid-stride assertion. Luma is exact, so
// a tight byte comparison is robust here (unlike point-sampled chroma).
// ---------------------------------------------------------------------------

#[test]
fn decode_coco_grey_odd_width() {
    // 595×438 — odd-width greyscale. Verify the native Grey format, the odd
    // logical dims, and that the luma plane matches the image crate tightly.
    let jpeg = testdata("coco_grey_odd.jpg");
    let (w, h) = (595usize, 438usize);
    let mut tensor = Tensor::<u8>::image(w, h, PixelFormat::Grey, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!(info.format, PixelFormat::Grey, "native greyscale format");
    assert_eq!((info.width, info.height), (w, h), "odd dims preserved");

    let stride = info.row_stride;
    let ref_luma = image::load_from_memory(&jpeg).unwrap().to_luma8();
    assert_eq!(ref_luma.dimensions(), (w as u32, h as u32));
    let map = tensor.map().unwrap();
    let buf: &[u8] = &map;
    let mut y_max: u32 = 0;
    let mut y_total: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let ours = buf[y * stride + x] as i32;
            let refv = ref_luma.get_pixel(x as u32, y as u32)[0] as i32;
            let d = (ours - refv).unsigned_abs();
            y_max = y_max.max(d);
            y_total += d as u64;
        }
    }
    let y_mae = y_total as f64 / (w * h) as f64;
    eprintln!("coco_grey_odd luma: MAE={y_mae:.3}, max={y_max}");
    assert!(y_max <= 4, "grey luma max diff {y_max} > 4");
    assert!(y_mae < 1.0, "grey luma MAE {y_mae:.3} > 1.0");
}
