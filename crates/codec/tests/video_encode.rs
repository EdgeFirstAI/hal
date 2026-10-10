// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware H.264 encoding. Each test skips (prints `SKIPPED` and passes)
//! where there is no encoder or no DMA-BUF heap.
//!
//! Setting `EDGEFIRST_VIDEO_OUT` to a directory writes each test's stream
//! there as `<test>.h264`, with the source pictures of the first frame as
//! `<test>.nv12`, for checking with an external decoder.

use std::time::{Duration, Instant};

use edgefirst_codec::video::{EncodedFrame, EncoderConfig, FrameOptions, VideoEncoder};
use edgefirst_codec::CodecError;
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMemory};

fn skip(why: &str) {
    eprintln!("SKIPPED: {why}");
}

/// Opens an encoder, or `None` (after printing why) when the platform has
/// none.
fn open(config: EncoderConfig) -> Option<VideoEncoder> {
    if !edgefirst_tensor::is_dma_available() {
        skip("no DMA-BUF heap");
        return None;
    }
    match VideoEncoder::new(config) {
        Ok(e) => Some(e),
        Err(CodecError::NoDevice(why)) => {
            skip(&why);
            None
        }
        Err(e) => panic!("encoder open failed: {e}"),
    }
}

/// A DMA-BUF NV12 image holding a gradient that moves with `frame`.
fn nv12(width: usize, height: usize) -> TensorDyn {
    TensorDyn::image(
        width,
        height,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("DMA-BUF NV12 image")
}

fn paint(img: &TensorDyn, frame: usize) {
    let w = img.width().unwrap();
    let h = img.height().unwrap();
    let stride = img.effective_row_stride().unwrap();
    let mut px = img.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        for x in 0..w {
            px[y * stride + x] = ((x / 4 + y / 4 + 2 * frame) & 0xff) as u8;
        }
    }
    let chroma = stride * h;
    for y in 0..h / 2 {
        for x in (0..w).step_by(2) {
            px[chroma + y * stride + x] = (64 + (x / 16 + frame) % 128) as u8;
            px[chroma + y * stride + x + 1] = (64 + (y / 8) % 128) as u8;
        }
    }
}

/// The dense NV12 bytes of `region` of `img`.
fn nv12_region(img: &TensorDyn, r: Region) -> Vec<u8> {
    let stride = img.effective_row_stride().unwrap();
    let h = img.height().unwrap();
    let px = img.map_bytes(CpuAccess::Read).unwrap();
    let mut out = Vec::with_capacity(r.width * r.height * 3 / 2);
    for y in r.y..r.y + r.height {
        out.extend_from_slice(&px[y * stride + r.x..y * stride + r.x + r.width]);
    }
    for y in r.y / 2..(r.y + r.height) / 2 {
        let row = stride * h + y * stride;
        out.extend_from_slice(&px[row + r.x..row + r.x + r.width]);
    }
    out
}

fn save(name: &str, frames: &[EncodedFrame], first: &[u8]) {
    let Some(dir) = std::env::var_os("EDGEFIRST_VIDEO_OUT") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let stream: Vec<u8> = frames.iter().flat_map(|f| f.data.iter().copied()).collect();
    std::fs::write(dir.join(format!("{name}.h264")), stream).unwrap();
    std::fs::write(dir.join(format!("{name}.nv12")), first).unwrap();
}

fn parameter_sets_and_idr(au: &[u8]) -> (bool, bool) {
    let mut types = Vec::new();
    for i in 0..au.len().saturating_sub(3) {
        if au[i..i + 3] == [0, 0, 1] {
            types.push(au[i + 3] & 0x1f);
        }
    }
    (types.contains(&7) && types.contains(&8), types.contains(&5))
}

#[test]
fn encodes_1080p_nv12_at_30_fps_with_forced_keyframe_and_drain() {
    const FRAMES: usize = 60;
    const FORCED: usize = 37;
    let Some(mut enc) = open(EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0)) else {
        return;
    };
    eprintln!("encoder {}", enc.device());
    let src = nv12(1920, 1080);
    let mut out = Vec::new();
    let mut first = Vec::new();
    let mut busy = Duration::ZERO;
    for i in 0..FRAMES {
        paint(&src, i);
        if i == 0 {
            first = nv12_region(&src, Region::new(0, 0, 1920, 1080));
        }
        let opts = FrameOptions {
            pts: 1_000_000_000 + i as u64 * 33_333_333,
            keyframe: i == FORCED,
            ..FrameOptions::default()
        };
        let t = Instant::now();
        let au = enc.encode(&src, &opts).unwrap();
        busy += t.elapsed();
        out.extend(au);
    }
    let returned_before_flush = out.len();
    out.extend(enc.flush().unwrap());
    save("nv12_1080p", &out, &first);

    let per_frame = busy / FRAMES as u32;
    eprintln!(
        "{FRAMES} frames, {returned_before_flush} before flush, {per_frame:?} per frame, {} bytes",
        out.iter().map(|f| f.data.len()).sum::<usize>()
    );
    assert_eq!(out.len(), FRAMES, "every frame comes back once");
    for (i, f) in out.iter().enumerate() {
        assert_eq!(
            f.pts,
            1_000_000_000 + i as u64 * 33_333_333,
            "frame {i} pts"
        );
    }
    let (ps, idr) = parameter_sets_and_idr(&out[0].data);
    assert!(
        ps && idr && out[0].keyframe,
        "first frame is an IDR with SPS/PPS"
    );
    let (ps, idr) = parameter_sets_and_idr(&out[FORCED].data);
    assert!(
        ps && idr && out[FORCED].keyframe,
        "forced frame {FORCED} is an IDR with SPS/PPS"
    );
    assert!(
        per_frame < Duration::from_micros(33_333),
        "{per_frame:?} per frame is slower than 30 fps"
    );
}

#[test]
fn encodes_a_crop_of_a_larger_source() {
    let crop = Region::new(320, 180, 1280, 720);
    let Some(mut enc) = open(EncoderConfig::h264(1280, 720, PixelFormat::Nv12, 30.0)) else {
        return;
    };
    let src = nv12(1920, 1080);
    let mut out = Vec::new();
    let mut first = Vec::new();
    for i in 0..10 {
        paint(&src, i);
        if i == 0 {
            first = nv12_region(&src, crop);
        }
        let opts = FrameOptions {
            pts: i as u64,
            crop: Some(crop),
            ..FrameOptions::default()
        };
        out.extend(enc.encode(&src, &opts).unwrap());
    }
    out.extend(enc.flush().unwrap());
    save("nv12_crop_720p", &out, &first);
    assert_eq!(out.len(), 10);
    assert!(out[0].keyframe);
}

#[test]
fn encoder_is_reusable_after_flush() {
    let Some(mut enc) = open(EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0)) else {
        return;
    };
    let src = nv12(640, 480);
    for round in 0..2u64 {
        let mut n = 0;
        for i in 0..5 {
            paint(&src, i);
            let opts = FrameOptions {
                pts: round * 100 + i as u64,
                ..FrameOptions::default()
            };
            n += usize::from(enc.encode(&src, &opts).unwrap().is_some());
        }
        let rest = enc.flush().unwrap();
        assert_eq!(n + rest.len(), 5, "round {round}");
        if let Some(last) = rest.last() {
            assert_eq!(last.pts, round * 100 + 4);
        }
    }
}

#[test]
fn rejects_a_source_of_the_wrong_size() {
    let Some(mut enc) = open(EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0)) else {
        return;
    };
    let src = nv12(320, 240);
    let err = enc.encode(&src, &FrameOptions::default()).unwrap_err();
    assert!(matches!(err, CodecError::InvalidInput(_)), "{err}");
}
