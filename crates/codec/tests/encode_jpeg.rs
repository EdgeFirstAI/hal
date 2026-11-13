// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! JPEG encoding with [`ImageEncoder`], checked by decoding each file with
//! [`ImageDecoder`] and comparing the pixels.

use edgefirst_codec::{Codec, CodecError, Engine, ImageDecoder, ImageEncoder};
use edgefirst_tensor::{
    ColorRange, Colorimetry, CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory,
};

fn image(w: usize, h: usize, format: PixelFormat) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        format,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap()
}

/// A smooth luma ramp with gently varying chroma.
fn luma(x: usize, y: usize) -> u8 {
    (32 + (x + 2 * y) % 192) as u8
}
fn cb(x: usize, _y: usize) -> u8 {
    (96 + x / 16 % 64) as u8
}
fn cr(_x: usize, y: usize) -> u8 {
    (96 + y / 16 % 64) as u8
}

/// Writes the test pattern into `t` in its own layout.
fn fill(t: &TensorDyn) {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let format = t.format().unwrap();
    let mut m = t.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        for x in 0..w {
            let row = y * stride;
            match format {
                PixelFormat::Grey => m[row + x] = luma(x, y),
                PixelFormat::Nv12 | PixelFormat::Nv16 => {
                    m[row + x] = luma(x, y);
                    let cy = if format == PixelFormat::Nv12 {
                        y / 2
                    } else {
                        y
                    };
                    let c = (h + cy) * stride + (x & !1);
                    m[c] = cb(x & !1, y & !1);
                    m[c + 1] = cr(x & !1, y & !1);
                }
                PixelFormat::Yuyv => {
                    m[row + 2 * x] = luma(x, y);
                    m[row + 4 * (x / 2) + 1] = cb(x & !1, y);
                    m[row + 4 * (x / 2) + 3] = cr(x & !1, y);
                }
                PixelFormat::Rgb | PixelFormat::Rgba => {
                    let n = format.channels();
                    let v = luma(x, y);
                    m[row + n * x] = v;
                    m[row + n * x + 1] = v / 2;
                    m[row + n * x + 2] = 255 - v;
                    if n == 4 {
                        m[row + n * x + 3] = 255;
                    }
                }
                other => panic!("{other}"),
            }
        }
    }
}

/// The decoded luma (or, for RGB, the first channel) of `jpeg`, row by row.
fn decoded_first_channel(jpeg: &[u8], w: usize, h: usize, rgb: bool) -> Vec<u8> {
    let mut decoder = ImageDecoder::new();
    decoder.set_engine(Engine::Cpu);
    if rgb {
        decoder.set_output_format(Some(PixelFormat::Rgb));
    }
    let format = if rgb {
        PixelFormat::Rgb
    } else {
        PixelFormat::Nv24
    };
    let mut out = image(w, h, format);
    let info = decoder.decode_into_dyn(jpeg, &mut out).unwrap();
    assert_eq!((info.width, info.height), (w, h));
    let stride = out.effective_row_stride().unwrap();
    let step = if rgb { 3 } else { 1 };
    let m = out.map_bytes(CpuAccess::Read).unwrap();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| m[y * stride + step * x])
        .collect()
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a.abs_diff(b)))
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn each_source_format_round_trips_through_a_jpeg() {
    let (w, h) = (320usize, 240usize);
    let mut encoder = ImageEncoder::new(Codec::Jpeg).unwrap();
    encoder.set_quality(95).unwrap();
    for format in [
        PixelFormat::Nv12,
        PixelFormat::Nv16,
        PixelFormat::Yuyv,
        PixelFormat::Grey,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
    ] {
        let src = image(w, h, format);
        fill(&src);
        let jpeg = encoder.encode(&src).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "{format}: SOI");
        let rgb = matches!(format, PixelFormat::Rgb | PixelFormat::Rgba);
        let got = decoded_first_channel(&jpeg, w, h, rgb);
        let want: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| luma(x, y)))
            .collect();
        let diff = mean_abs_diff(&got, &want);
        assert!(diff < 2.0, "{format}: mean difference {diff}");
    }
}

#[test]
fn a_padded_row_stride_is_followed() {
    let (w, h) = (100usize, 60usize);
    let src = TensorDyn::image_with_stride(
        w,
        h,
        PixelFormat::Nv12,
        DType::U8,
        128,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    );
    let Ok(src) = src else {
        return eprintln!("WARNING: SKIPPED: a_padded_row_stride_is_followed - no strided images");
    };
    fill(&src);
    let jpeg = ImageEncoder::new(Codec::Jpeg)
        .unwrap()
        .encode(&src)
        .unwrap();
    let got = decoded_first_channel(&jpeg, w, h, false);
    let want: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).map(move |x| luma(x, y)))
        .collect();
    assert!(mean_abs_diff(&got, &want) < 3.0);
}

#[test]
fn limited_range_sources_are_written_at_full_range() {
    let (w, h) = (64usize, 64usize);
    let mut src = image(w, h, PixelFormat::Grey);
    {
        let stride = src.effective_row_stride().unwrap();
        let mut m = src.map_bytes(CpuAccess::Write).unwrap();
        for y in 0..h {
            m[y * stride..y * stride + w].fill(if y < h / 2 { 16 } else { 235 });
        }
    }
    src.set_colorimetry(Some(Colorimetry {
        range: Some(ColorRange::Limited),
        ..Colorimetry::default()
    }));
    let jpeg = ImageEncoder::new(Codec::Jpeg)
        .unwrap()
        .encode(&src)
        .unwrap();
    let got = decoded_first_channel(&jpeg, w, h, false);
    assert!(got[w * 4] < 4, "black: {}", got[w * 4]);
    assert!(got[w * (h - 4)] > 251, "white: {}", got[w * (h - 4)]);
}

#[test]
fn unsupported_sources_and_engines_are_refused() {
    let mut encoder = ImageEncoder::new(Codec::Jpeg).unwrap();
    let planar = image(64, 64, PixelFormat::PlanarRgb);
    assert!(matches!(
        encoder.encode(&planar),
        Err(CodecError::UnsupportedFormat(PixelFormat::PlanarRgb))
    ));
    encoder.set_engine(Engine::VideoToolbox);
    assert!(matches!(
        encoder.encode(&image(64, 64, PixelFormat::Nv12)),
        Err(CodecError::NoDevice(_))
    ));
}
