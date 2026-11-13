// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MJPEG encoding with [`VideoEncoder`], and still JPEG on a hardware
//! [`ImageEncoder`] engine. Each picture is decoded on the CPU and compared
//! with its source. Tests skip (print `SKIPPED` and pass) where the platform
//! has no hardware JPEG encoder (only V4L2 devices such as the i.MX 95's have
//! one today).

use edgefirst_codec::video::{EncoderConfig, FrameOptions, VideoEncoder};
use edgefirst_codec::{Codec, CodecError, Engine, ImageDecoder, ImageEncoder};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

fn skip(test: &str, why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {test} - {why}");
}

fn source(w: usize, h: usize, frame: usize) -> Option<TensorDyn> {
    let t = TensorDyn::image(
        w,
        h,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .ok()?;
    let stride = t.effective_row_stride().unwrap();
    {
        let mut m = t.map_bytes(CpuAccess::Write).unwrap();
        for y in 0..h {
            for x in 0..w {
                m[y * stride + x] = (32 + (x + 2 * y + 4 * frame) % 192) as u8;
            }
        }
        for y in 0..h / 2 {
            for x in 0..w {
                m[(h + y) * stride + x] = 128;
            }
        }
    }
    Some(t)
}

/// The mean luma difference between `jpeg`, decoded on the CPU, and `src`.
fn luma_difference(jpeg: &[u8], src: &TensorDyn) -> f64 {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
    let (w, h) = (src.width().unwrap(), src.height().unwrap());
    let mut out = TensorDyn::image(
        w,
        h,
        PixelFormat::Nv24,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    decoder.set_engine(Engine::Cpu);
    let info = decoder.decode_into_dyn(jpeg, &mut out).unwrap();
    assert_eq!((info.width, info.height), (w, h));
    let (so, ss) = (
        out.effective_row_stride().unwrap(),
        src.effective_row_stride().unwrap(),
    );
    let (a, b) = (
        out.map_bytes(CpuAccess::Read).unwrap(),
        src.map_bytes(CpuAccess::Read).unwrap(),
    );
    let mut sum = 0.0;
    for y in 0..h {
        for x in 0..w {
            sum += f64::from(a[y * so + x].abs_diff(b[y * ss + x]));
        }
    }
    sum / (w * h) as f64
}

#[test]
fn mjpeg_pictures_decode_to_their_sources() {
    const TEST: &str = "mjpeg_pictures_decode_to_their_sources";
    let (w, h) = (640usize, 480usize);
    let mut config = EncoderConfig::new(Codec::Mjpeg, w as u32, h as u32, PixelFormat::Nv12, 30.0);
    config.quality = Some(90);
    let mut encoder = match VideoEncoder::new(config) {
        Ok(e) => e,
        Err(CodecError::NoDevice(why)) => return skip(TEST, &why),
        Err(CodecError::Unsupported(why)) => return skip(TEST, &why.to_string()),
        Err(e) => panic!("{e}"),
    };
    let sources: Vec<TensorDyn> = (0..3).filter_map(|f| source(w, h, f)).collect();
    if sources.len() < 3 {
        return skip(TEST, "no DMA-BUF images");
    }
    let mut frames = Vec::new();
    for (pts, src) in sources.iter().enumerate() {
        let opts = FrameOptions {
            pts: pts as u64,
            ..FrameOptions::default()
        };
        frames.extend(encoder.encode(src, &opts).unwrap());
    }
    frames.extend(encoder.flush().unwrap());
    assert_eq!(frames.len(), sources.len());
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.pts, i as u64);
        assert!(frame.keyframe);
        let diff = luma_difference(&frame.data, &sources[i]);
        assert!(diff < 2.0, "picture {i}: mean luma difference {diff}");
    }
}

#[test]
fn a_hardware_image_encoder_writes_the_source() {
    const TEST: &str = "a_hardware_image_encoder_writes_the_source";
    let Some(src) = source(640, 480, 0) else {
        return skip(TEST, "no DMA-BUF images");
    };
    let mut encoder = ImageEncoder::new(Codec::Jpeg).unwrap();
    encoder.set_engine(Engine::V4l2);
    let jpeg = match encoder.encode(&src) {
        Ok(jpeg) => jpeg,
        Err(CodecError::NoDevice(why)) => return skip(TEST, &why),
        Err(CodecError::Unsupported(why)) => return skip(TEST, &why.to_string()),
        Err(e) => panic!("{e}"),
    };
    assert!(luma_difference(&jpeg, &src) < 2.0);
}
