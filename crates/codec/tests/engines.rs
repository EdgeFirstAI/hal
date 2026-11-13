// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Engine selection, capabilities and transfer reporting across the codec
//! families. Tests that need a video engine skip (print `SKIPPED` and pass)
//! where the platform has none.

use edgefirst_codec::video::{DecoderConfig, EncoderConfig, FrameOptions, VideoEncoder};
use edgefirst_codec::{
    capabilities, Codec, CodecError, Direction, Engine, ImageDecoder, Transfer, UnsupportedFeature,
};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

fn skip(test: &str, why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {test} - {why}");
}

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

/// The engine this platform runs video on, if this build has one.
fn video_engine() -> Option<Engine> {
    if cfg!(all(target_os = "linux", feature = "v4l2")) {
        Some(Engine::V4l2)
    } else if cfg!(all(windows, feature = "media-foundation")) {
        Some(Engine::MediaFoundation)
    } else {
        None
    }
}

fn is_codec_refusal(e: &CodecError, codec: Codec, engine: Engine) -> bool {
    matches!(
        e,
        CodecError::Unsupported(UnsupportedFeature::Codec { codec: c, engine: en, .. })
            if *c == codec && *en == engine
    )
}

#[test]
fn video_capabilities_are_the_platform_engine_and_h264() {
    let caps = capabilities();
    let video: Vec<_> = caps.iter().filter(|c| c.codec.is_video()).collect();
    for c in &video {
        assert_eq!(Some(c.engine), video_engine(), "{c:?}");
        assert_eq!(c.codec, Codec::H264, "{c:?}");
        assert!(!c.formats.is_empty(), "{c:?}");
    }
    if video.is_empty() {
        return skip(
            "video_capabilities_are_the_platform_engine_and_h264",
            "no video engine here",
        );
    }
    for c in video.iter().filter(|c| c.direction == Direction::Decode) {
        assert_eq!(c.formats, vec![PixelFormat::Nv12], "{c:?}");
    }
}

#[test]
fn still_image_capabilities_end_with_the_cpu_decoder() {
    let caps = capabilities();
    let jpeg: Vec<_> = caps
        .iter()
        .filter(|c| c.codec == Codec::Jpeg && c.direction == Direction::Decode)
        .collect();
    let last = jpeg.last().expect("a JPEG decoder");
    assert_eq!((last.engine, last.hardware), (Engine::Cpu, false));
    assert!(last.formats.contains(&PixelFormat::Nv12));
}

#[test]
fn engines_that_cannot_run_a_video_codec_are_refused() {
    let encoder = |engine, codec| {
        VideoEncoder::new(
            EncoderConfig::new(codec, 640, 480, PixelFormat::Nv12, 30.0).with_engine(engine),
        )
        .map(|_| ())
        .unwrap_err()
    };
    let e = encoder(Engine::Cpu, Codec::H264);
    assert!(is_codec_refusal(&e, Codec::H264, Engine::Cpu), "{e}");
    let e = encoder(Engine::VideoToolbox, Codec::H264);
    assert!(matches!(e, CodecError::NoDevice(_)), "{e}");
    let e = encoder(Engine::Auto, Codec::Jpeg);
    match video_engine() {
        Some(_) => assert!(is_codec_refusal(&e, Codec::Jpeg, Engine::Auto), "{e}"),
        None => assert!(matches!(e, CodecError::NoDevice(_)), "{e}"),
    }
    let decoder = |engine, codec| {
        edgefirst_codec::video::VideoDecoder::new(DecoderConfig::new(codec).with_engine(engine))
            .map(|_| ())
            .unwrap_err()
    };
    if let Some(engine) = video_engine() {
        for codec in [Codec::Hevc, Codec::Vp8, Codec::Vp9, Codec::Av1] {
            let e = decoder(engine, codec);
            assert!(is_codec_refusal(&e, codec, engine), "{e}");
            let e = decoder(Engine::Auto, codec);
            assert!(is_codec_refusal(&e, codec, Engine::Auto), "{e}");
        }
    }
    let e = decoder(Engine::NvJpeg, Codec::H264);
    if cfg!(all(target_os = "linux", feature = "nvjpeg")) {
        assert!(is_codec_refusal(&e, Codec::H264, Engine::NvJpeg), "{e}");
    } else {
        assert!(matches!(e, CodecError::NoDevice(_)), "{e}");
    }
}

#[test]
fn an_encoder_reports_its_engine_and_how_sources_reach_it() {
    const TEST: &str = "an_encoder_reports_its_engine_and_how_sources_reach_it";
    if !edgefirst_tensor::is_gpu_buffer_available() {
        return skip(TEST, "no GPU buffers");
    }
    let config = EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0);
    let mut encoder = match VideoEncoder::new(config) {
        Ok(e) => e,
        Err(CodecError::NoDevice(why)) => return skip(TEST, &why),
        Err(e) => panic!("{e}"),
    };
    let info = encoder.info();
    assert_eq!(Some(info.engine), video_engine());
    assert!(!info.device.is_empty());
    let src = TensorDyn::image(
        640,
        480,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    for pts in 0..3 {
        encoder
            .encode(
                &src,
                &FrameOptions {
                    pts,
                    ..FrameOptions::default()
                },
            )
            .unwrap();
    }
    let after = encoder.info();
    match after.engine {
        Engine::V4l2 => assert_eq!(after.transfer, Transfer::ZeroCopy),
        // A hardware encoder on D3D11 reads native NV12 sources in place or
        // copies them on the GPU; a software one takes system memory.
        Engine::MediaFoundation if after.hardware => {
            assert_ne!(after.transfer, Transfer::CpuCopy, "{after:?}")
        }
        Engine::MediaFoundation => assert_eq!(after.transfer, Transfer::CpuCopy),
        other => panic!("unexpected engine {other}"),
    }
}

#[test]
fn an_image_decoder_on_an_explicit_engine_decodes_or_refuses() {
    let jpeg = testdata("zidane.jpg");
    let mut dst = TensorDyn::image(
        1280,
        720,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    assert_eq!(decoder.engine(), Engine::Auto);
    decoder.set_engine(Engine::Cpu);
    let info = decoder.decode_into_dyn(&jpeg, &mut dst).unwrap();
    assert_eq!((info.width, info.height), (1280, 720));

    decoder.set_engine(Engine::MediaFoundation);
    let e = decoder.decode_into_dyn(&jpeg, &mut dst).unwrap_err();
    if cfg!(all(windows, feature = "media-foundation")) {
        assert!(
            is_codec_refusal(&e, Codec::Jpeg, Engine::MediaFoundation),
            "{e}"
        );
    } else {
        assert!(matches!(e, CodecError::NoDevice(_)), "{e}");
    }
    decoder.set_engine(Engine::VideoToolbox);
    let e = decoder.decode_into_dyn(&jpeg, &mut dst).unwrap_err();
    assert!(matches!(e, CodecError::NoDevice(_)), "{e}");
}
