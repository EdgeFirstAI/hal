// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware H.264 encoding. Each test skips (prints `SKIPPED` and passes)
//! where there is no encoder or no GPU buffer allocator (a DMA-BUF heap on
//! Linux, D3D11 on Windows); under `HAL_TEST_REQUIRE_DMA=1` a missing one
//! fails instead.
//!
//! Setting `EDGEFIRST_VIDEO_OUT` to a directory writes each test's stream
//! there as `<test>.h264`, with the source pictures of the first frame as
//! `<test>.nv12`, for checking with an external decoder.

use std::time::{Duration, Instant};

use edgefirst_codec::video::{
    DecodeEvent, DecoderConfig, EncodedFrame, EncoderConfig, FrameOptions, H264Level, H264Profile,
    VideoDecoder, VideoEncoder,
};
use edgefirst_codec::{CodecError, Engine, Transfer};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMemory};

#[path = "common/pictures.rs"]
mod pictures;
#[path = "common/sessions.rs"]
mod sessions;

use pictures::{nv12, nv12_region, paint, psnr};

fn skip(test: &str, why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {test} - {why}");
}

/// The DMA require gate: a skip here fails under `HAL_TEST_REQUIRE_DMA=1`.
#[path = "../../tensor/tests/support/dma_require.rs"]
mod dma_require;

/// Opens an encoder, or `None` (after printing why) when the platform has
/// none.
fn open(test: &str, config: EncoderConfig) -> Option<VideoEncoder> {
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), test) {
        return None;
    }
    match VideoEncoder::new(config) {
        Ok(e) => {
            log_hardware(test, &e);
            Some(e)
        }
        Err(CodecError::NoDevice(why)) => {
            skip(test, &why);
            None
        }
        Err(e) => panic!("encoder open failed: {e}"),
    }
}

/// Encodes `src`. On VideoToolbox, submits it again while the encoder
/// refuses it with [`CodecError::Timeout`].
fn encode(
    enc: &mut VideoEncoder,
    src: &TensorDyn,
    opts: &FrameOptions,
) -> Result<Option<EncodedFrame>, CodecError> {
    let vt = enc.info().engine == Engine::VideoToolbox;
    encode_retry::retry(
        || enc.encode(src, opts),
        |e| vt && matches!(e, CodecError::Timeout(_)),
    )
}

#[path = "support/encode_retry.rs"]
mod encode_retry;

/// Prints which encoder runs `test`, and whether it is hardware.
fn log_hardware(test: &str, e: &VideoEncoder) {
    let info = e.info();
    eprintln!("{test}: {} hardware {}", info.device, info.hardware);
}

/// Decodes `stream` and returns the dense NV12 bytes of its first picture,
/// or `None` (after printing why) when there is no decoder.
fn decode_first(test: &str, stream: &[EncodedFrame]) -> Option<Vec<u8>> {
    let mut decoder = match VideoDecoder::new(DecoderConfig::h264()) {
        Ok(d) => d,
        Err(CodecError::NoDevice(why)) => {
            skip(test, &why);
            return None;
        }
        Err(e) => panic!("decoder open failed: {e}"),
    };
    let mut events = Vec::new();
    for f in stream {
        events.extend(decoder.decode(&f.data, f.pts).expect("decode"));
    }
    events.extend(decoder.flush().expect("flush"));
    events.into_iter().find_map(|e| match e {
        DecodeEvent::Frame(f) => {
            let t = f.tensor();
            let r = Region::new(0, 0, t.width().unwrap(), t.height().unwrap());
            Some(nv12_region(t, r))
        }
        DecodeEvent::Reconfigured(_) => None,
    })
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

/// `profile_idc` and `level_idc` of the first SPS in `au`.
fn sps_profile_and_level(au: &[u8]) -> Option<(u8, u8)> {
    (0..au.len().saturating_sub(6))
        .find(|&i| au[i..i + 3] == [0, 0, 1] && au[i + 3] & 0x1f == 7)
        .map(|i| (au[i + 4], au[i + 6]))
}

#[test]
fn the_stream_carries_the_requested_profile_and_level() {
    let _sessions = sessions::take(1);
    const TEST: &str = "the_stream_carries_the_requested_profile_and_level";
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), TEST) {
        return;
    }
    let src = nv12(1920, 1080);
    for (profile, profile_idc) in [(H264Profile::High, 100), (H264Profile::Main, 77)] {
        let mut config = EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0);
        config.profile = Some(profile);
        config.level = Some(H264Level::L4_0);
        let mut enc = match VideoEncoder::new(config) {
            Ok(e) => {
                log_hardware(TEST, &e);
                e
            }
            Err(CodecError::NoDevice(why)) => return skip(TEST, &why),
            Err(CodecError::InvalidConfig(why)) => {
                skip(TEST, &format!("{profile:?}: {why}"));
                continue;
            }
            Err(e) => panic!("encoder open failed: {e}"),
        };
        let mut out = Vec::new();
        for i in 0..3 {
            paint(&src, i);
            let opts = FrameOptions {
                pts: i as u64,
                ..FrameOptions::default()
            };
            out.extend(enc.encode(&src, &opts).unwrap());
        }
        out.extend(enc.flush().unwrap());
        let sps = sps_profile_and_level(&out[0].data);
        assert_eq!(
            sps,
            Some((profile_idc, 40)),
            "{profile:?} level 4.0 from {}",
            enc.device()
        );
    }
}

#[test]
fn encodes_1080p_nv12_at_30_fps_with_forced_keyframe_and_drain() {
    let _sessions = sessions::take(1);
    const FRAMES: usize = 60;
    const FORCED: usize = 37;
    let Some(mut enc) = open(
        "encodes_1080p_nv12_at_30_fps_with_forced_keyframe_and_drain",
        EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0),
    ) else {
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
        let au = encode(&mut enc, &src, &opts).unwrap();
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
    let _sessions = sessions::take(1);
    let crop = Region::new(320, 180, 1280, 720);
    let Some(mut enc) = open(
        "encodes_a_crop_of_a_larger_source",
        EncoderConfig::h264(1280, 720, PixelFormat::Nv12, 30.0),
    ) else {
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
        out.extend(encode(&mut enc, &src, &opts).unwrap());
    }
    out.extend(enc.flush().unwrap());
    save("nv12_crop_720p", &out, &first);
    assert_eq!(out.len(), 10);
    assert!(out[0].keyframe);
}

#[test]
fn a_copied_crop_decodes_to_the_source_region() {
    let _sessions = sessions::take(2);
    const TEST: &str = "a_copied_crop_decodes_to_the_source_region";
    // A host-memory source, which V4L2 copies into a buffer it reads.
    let crop = Region::new(64, 32, 640, 480);
    let Some(mut enc) = open(TEST, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0)) else {
        return;
    };
    let src = TensorDyn::image(
        1000,
        600,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    paint(&src, 0);
    let want = nv12_region(&src, crop);
    let mut out = Vec::new();
    for i in 0..5 {
        let opts = FrameOptions {
            pts: i,
            crop: Some(crop),
            ..FrameOptions::default()
        };
        out.extend(encode(&mut enc, &src, &opts).unwrap());
    }
    out.extend(enc.flush().unwrap());
    assert_eq!(out.len(), 5);
    if enc.info().engine == Engine::V4l2 {
        assert_eq!(enc.info().transfer, Transfer::CpuCopy);
    }
    let Some(got) = decode_first(TEST, &out) else {
        return;
    };
    let db = psnr(&want, &got);
    assert!(db > 30.0, "the decoded crop is {db:.1} dB from the source");
}

#[test]
fn a_source_from_create_source_is_read_in_place() {
    let _sessions = sessions::take(2);
    const TEST: &str = "a_source_from_create_source_is_read_in_place";
    let Some(mut enc) = open(
        TEST,
        EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0),
    ) else {
        return;
    };
    let src = enc.create_source().unwrap();
    assert_eq!((src.width(), src.height()), (Some(1920), Some(1080)));
    assert_eq!(src.format(), Some(PixelFormat::Nv12));
    paint(&src, 0);
    let want = nv12_region(&src, Region::new(0, 0, 1920, 1080));
    let mut out = Vec::new();
    for i in 0..5 {
        paint(&src, i as usize);
        let opts = FrameOptions {
            pts: i,
            ..FrameOptions::default()
        };
        out.extend(encode(&mut enc, &src, &opts).unwrap());
        if enc.info().engine == Engine::V4l2 {
            assert_eq!(enc.info().transfer, Transfer::ZeroCopy, "frame {i}");
        }
    }
    out.extend(enc.flush().unwrap());
    assert_eq!(out.len(), 5);
    let Some(got) = decode_first(TEST, &out) else {
        return;
    };
    let db = psnr(&want, &got);
    assert!(
        db > 30.0,
        "the decoded picture is {db:.1} dB from the source"
    );
}

/// A host-memory source is copied into a buffer the device reads.
#[test]
fn encodes_a_host_memory_source_where_the_encoder_copies() {
    let _sessions = sessions::take(1);
    let Some(mut enc) = open(
        "encodes_a_host_memory_source_where_the_encoder_copies",
        EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0),
    ) else {
        return;
    };
    let src = TensorDyn::image(
        640,
        480,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut out = Vec::new();
    for i in 0..10 {
        paint(&src, i);
        let opts = FrameOptions {
            pts: i as u64,
            ..FrameOptions::default()
        };
        out.extend(encode(&mut enc, &src, &opts).unwrap());
    }
    out.extend(enc.flush().unwrap());
    assert_eq!(out.len(), 10);
    assert!(out[0].keyframe);
    let pts: Vec<u64> = out.iter().map(|f| f.pts).collect();
    assert_eq!(pts, (0..10).collect::<Vec<_>>());
}

/// More encoders than a device has sessions open one after another, each
/// dropped after a frame, half of them before their stream is drained.
#[test]
fn a_dropped_encoder_frees_its_session() {
    let _sessions = sessions::take(1);
    const ROUNDS: usize = 40;
    let src = match TensorDyn::image(
        320,
        240,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(src) => src,
        Err(e) => {
            return skip(
                "a_dropped_encoder_frees_its_session",
                &format!("no DMA-BUF image: {e}"),
            )
        }
    };
    for round in 0..ROUNDS {
        let Some(mut enc) = open(
            "a_dropped_encoder_frees_its_session",
            EncoderConfig::h264(320, 240, PixelFormat::Nv12, 30.0),
        ) else {
            assert_eq!(round, 0, "the encoder went away after {round} sessions");
            return;
        };
        paint(&src, round);
        enc.encode(&src, &FrameOptions::default())
            .unwrap_or_else(|e| panic!("session {round}: {e}"));
        if round % 2 == 0 {
            enc.flush()
                .unwrap_or_else(|e| panic!("session {round}: {e}"));
        }
    }
}

#[test]
fn encoder_is_reusable_after_flush() {
    let _sessions = sessions::take(1);
    let Some(mut enc) = open(
        "encoder_is_reusable_after_flush",
        EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0),
    ) else {
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
    let _sessions = sessions::take(1);
    let Some(mut enc) = open(
        "rejects_a_source_of_the_wrong_size",
        EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0),
    ) else {
        return;
    };
    let src = nv12(320, 240);
    let err = enc.encode(&src, &FrameOptions::default()).unwrap_err();
    assert!(matches!(err, CodecError::InvalidInput(_)), "{err}");
}

#[test]
fn rejects_a_16_bit_source() {
    let _sessions = sessions::take(1);
    let Some(mut enc) = open(
        "rejects_a_16_bit_source",
        EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0),
    ) else {
        return;
    };
    // The element type is checked first, for any memory.
    let src = TensorDyn::image(
        640,
        480,
        PixelFormat::Nv12,
        DType::U16,
        None,
        CpuAccess::ReadWrite,
    )
    .expect("16-bit NV12 image");
    let err = enc.encode(&src, &FrameOptions::default()).unwrap_err();
    assert!(matches!(err, CodecError::InvalidInput(_)), "{err}");
}
