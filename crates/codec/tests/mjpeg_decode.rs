// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MJPEG decoding with [`VideoDecoder`]: each picture is compared with the
//! CPU JPEG decoder's. Tests skip (print `SKIPPED` and pass) where the
//! platform has no MJPEG decoder.

use edgefirst_codec::video::{DecodeEvent, DecodedFrame, DecoderConfig, VideoDecoder};
use edgefirst_codec::{peek_info, Codec, CodecError, Engine, ImageDecoder};
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

fn open(test: &str) -> Option<VideoDecoder> {
    match VideoDecoder::new(DecoderConfig::new(Codec::Mjpeg)) {
        Ok(d) => Some(d),
        Err(CodecError::NoDevice(why)) => {
            skip(test, &why);
            None
        }
        Err(CodecError::Unsupported(why)) => {
            skip(test, &why.to_string());
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// The luma rows of an NV12 or Grey image, without padding.
fn luma(t: &TensorDyn) -> Vec<u8> {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let m = t.map_bytes(CpuAccess::Read).unwrap();
    (0..h)
        .flat_map(|y| m[y * stride..y * stride + w].to_vec())
        .collect()
}

/// The CPU decoder's luma of `jpeg`.
fn reference_luma(jpeg: &[u8]) -> Vec<u8> {
    let info = peek_info(jpeg).unwrap();
    let mut t = TensorDyn::image(
        info.width,
        info.height,
        PixelFormat::Nv24,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut decoder = ImageDecoder::new();
    decoder.set_engine(Engine::Cpu);
    decoder.decode_into_dyn(jpeg, &mut t).unwrap();
    luma(&t)
}

fn frames(events: Vec<DecodeEvent>) -> Vec<DecodedFrame> {
    events
        .into_iter()
        .filter_map(|e| match e {
            DecodeEvent::Frame(f) => Some(f),
            DecodeEvent::Reconfigured(_) => None,
        })
        .collect()
}

#[test]
fn each_mjpeg_picture_matches_the_cpu_decoder() {
    const TEST: &str = "each_mjpeg_picture_matches_the_cpu_decoder";
    let Some(mut decoder) = open(TEST) else {
        return;
    };
    let pictures = ["zidane.jpg", "zidane.jpg", "giraffe.jpg", "zidane.jpg"];
    let mut got = Vec::new();
    for (pts, name) in pictures.iter().enumerate() {
        let jpeg = testdata(name);
        for frame in frames(decoder.decode(&jpeg, pts as u64).unwrap()) {
            got.push((frame.pts(), luma(frame.tensor()), frame.tensor().format()));
        }
    }
    for frame in frames(decoder.flush().unwrap()) {
        got.push((frame.pts(), luma(frame.tensor()), frame.tensor().format()));
    }
    eprintln!(
        "{TEST}: {} ({})",
        decoder.device(),
        decoder
            .stream_info()
            .map(|i| i.transfer.to_string())
            .unwrap_or_default()
    );
    let info = decoder.stream_info().expect("stream info").clone();
    assert_eq!(info.engine, decoder.engine());
    assert_eq!(got.len(), pictures.len(), "every picture comes back");
    for (i, (pts, y, format)) in got.iter().enumerate() {
        assert_eq!(*pts, i as u64);
        assert_eq!(*format, Some(PixelFormat::Nv12));
        let want = reference_luma(&testdata(pictures[i]));
        assert_eq!(y.len(), want.len(), "picture {i} size");
        let mean = y
            .iter()
            .zip(&want)
            .map(|(&a, &b)| f64::from(a.abs_diff(b)))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mean < 1.5, "picture {i}: mean luma difference {mean}");
    }
}

/// Cameras send 4:2:2 MJPEG; the decoders return NV12 for it where they
/// decode it at all.
#[test]
fn a_4_2_2_mjpeg_picture_decodes_to_nv12_or_is_refused() {
    const TEST: &str = "a_4_2_2_mjpeg_picture_decodes_to_nv12_or_is_refused";
    let Some(mut decoder) = open(TEST) else {
        return;
    };
    let jpeg = testdata("zidane_422.jpg");
    let events = match decoder.decode(&jpeg, 7) {
        Ok(events) => events,
        Err(CodecError::Unsupported(why)) => return skip(TEST, &why.to_string()),
        Err(e) => panic!("{e}"),
    };
    let mut got = frames(events);
    got.extend(frames(decoder.flush().unwrap()));
    assert_eq!(got.len(), 1);
    let t = got[0].tensor();
    assert_eq!(got[0].pts(), 7);
    assert_eq!(t.format(), Some(PixelFormat::Nv12));
    let (y, want) = (luma(t), reference_luma(&jpeg));
    assert_eq!(y.len(), want.len());
    let mean = y
        .iter()
        .zip(&want)
        .map(|(&a, &b)| f64::from(a.abs_diff(b)))
        .sum::<f64>()
        / y.len() as f64;
    assert!(mean < 1.5, "mean luma difference {mean}");
}
