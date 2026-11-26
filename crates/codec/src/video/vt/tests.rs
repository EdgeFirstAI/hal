// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use edgefirst_tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry, CpuAccess, DType,
    PixelFormat, Region, TensorDyn, TensorMemory,
};

use super::ffi::*;
use super::*;
use crate::test_support::{dma_require, report_skip, sessions};
use crate::video::h264;
use crate::video::h264_syntax::{ParameterSets, SliceHeader, Sps};
use crate::video::{
    access_units, DecodeEvent, DecodedFrame, DecoderBackend, DecoderConfig, EncodedFrame,
    EncoderBackend, EncoderConfig, FrameOptions, H264Profile, OutputMemory, OutputPool, StreamInfo,
    VideoDecoder,
};

#[path = "../../../tests/support/encode_retry.rs"]
mod encode_retry;
mod zero_copy;

/// 30 frames of 1920x1080 Main profile with two B-frames between
/// references, GOP 15 (x264).
const MAIN_BFRAMES_FILE: &str = "h264_1080p_main_bframes.h264";
/// FNV-1a hashes of the visible NV12 bytes of each picture of
/// [`MAIN_BFRAMES_FILE`] as FFmpeg decodes it, in presentation order.
const MAIN_BFRAMES: [u64; 30] = [
    0xe698161fa65c8ba0,
    0xebee7174dbd0f4ea,
    0x449a6be07244aa1e,
    0xfb84ac037ed6298a,
    0x752ac1e45bf5f0f3,
    0xfc35c4efc08f4c18,
    0x1b4b83d9550bbf65,
    0xe1de97bb9ce7db1a,
    0xf560120fa345fcf6,
    0x6833f0bc3775b3c0,
    0x13b65ff0196c8f15,
    0x4cb9d98534a27bc7,
    0x215f2d77b2e3e0a5,
    0x37441abb43470f1d,
    0x5b496a4034f56bb9,
    0x558e34e748e5c5f4,
    0x5c4565ddc5d790d5,
    0x74b87442b09bfc40,
    0xcadde80e7d9c6ffa,
    0x49d0f4f1eaee590f,
    0x2d0d686dc82985a5,
    0x4c6218456542c034,
    0xabdf2c86ce62ab06,
    0x16d1a44710bf7825,
    0x36d5ba765e4ce5e5,
    0x11d38007e9f18d97,
    0x4b71086dd31eeeb7,
    0x109076f44515bc97,
    0xbf245f700c3506d2,
    0x9af53f3a1e11ce92,
];

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").into());
    let path = std::path::Path::new(&root).join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// A VideoToolbox decoder with `output`, or `None` after reporting why the
/// test skips.
fn open(test: &str, output: OutputPool) -> Option<VideoDecoder> {
    let _ = env_logger::builder().is_test(true).try_init();
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), test) {
        return None;
    }
    let config = DecoderConfig::h264()
        .with_engine(Engine::VideoToolbox)
        .with_output(output);
    match VideoDecoder::new(config) {
        Ok(d) => Some(d),
        Err(CodecError::NoDevice(why)) => {
            report_skip(&format!("{test} - {why}"));
            None
        }
        Err(e) => panic!("decoder open failed: {e}"),
    }
}

/// FNV-1a over the visible NV12 bytes of a frame, multiplane or in one
/// buffer: luma rows, then chroma rows, without row padding.
fn picture_hash(t: &TensorDyn) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let (w, rows) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let luma = t.map_bytes(CpuAccess::Read).unwrap();
    for y in 0..rows {
        eat(&luma[y * stride..y * stride + w]);
    }
    if let Some(chroma) = t.chroma_dyn() {
        let cstride = chroma.effective_row_stride().unwrap();
        let cmap = chroma.map_bytes(CpuAccess::Read).unwrap();
        for y in 0..rows.div_ceil(2) {
            eat(&cmap[y * cstride..y * cstride + w]);
        }
    } else {
        for y in 0..rows.div_ceil(2) {
            let start = (rows + y) * stride;
            eat(&luma[start..start + w]);
        }
    }
    h
}

/// The ID of the IOSurface a frame wraps.
fn surface_id(f: &DecodedFrame) -> u32 {
    let surface = f.tensor().iosurface_ref().expect("an IOSurface frame");
    // SAFETY: the frame's tensor keeps the surface live.
    unsafe { IOSurfaceGetID(surface) }
}

/// What a decode run produced.
#[derive(Default)]
struct Run {
    hashes: Vec<u64>,
    pts: Vec<u64>,
    infos: Vec<StreamInfo>,
}

impl Run {
    /// Records `events`, returning the frames for the caller to hold or
    /// drop.
    fn events(&mut self, events: Vec<DecodeEvent>) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        for e in events {
            match e {
                DecodeEvent::Frame(f) => {
                    self.hashes.push(picture_hash(f.tensor()));
                    self.pts.push(f.pts());
                    frames.push(f);
                }
                DecodeEvent::Reconfigured(info) => self.infos.push(info),
            }
        }
        frames
    }
}

#[test]
fn allocate_lends_multiplane_iosurface_frames_with_colorimetry() {
    let _sessions = sessions::take(1);
    let test = "allocate_lends_multiplane_iosurface_frames_with_colorimetry";
    let Some(mut decoder) = open(test, OutputPool::Allocate { count: 4 }) else {
        return;
    };
    let mut infos: Vec<StreamInfo> = Vec::new();
    let mut hashes = Vec::new();
    let mut check = |events: Vec<DecodeEvent>| {
        for e in events {
            let f = match e {
                DecodeEvent::Reconfigured(info) => {
                    infos.push(info);
                    continue;
                }
                DecodeEvent::Frame(f) => f,
            };
            let t = f.tensor();
            assert!(t.is_multiplane(), "frame {} is multiplane", f.pts());
            assert_eq!(t.memory(), TensorMemory::DmaBuf);
            assert_eq!(t.format(), Some(PixelFormat::Nv12));
            assert!(
                t.iosurface_ref().is_some(),
                "frame {} wraps an IOSurface",
                f.pts()
            );
            assert_eq!((t.width(), t.height()), (Some(1920), Some(1080)));
            let info = infos.last().expect("a format before the first frame");
            assert_eq!(t.colorimetry(), info.colorimetry);
            hashes.push(picture_hash(t));
        }
    };
    for (i, au) in access_units(&testdata(MAIN_BFRAMES_FILE)).enumerate() {
        check(decoder.decode(au, i as u64).expect("decode"));
    }
    check(decoder.flush().expect("flush"));
    assert_eq!(hashes, MAIN_BFRAMES);
    assert_eq!(infos.len(), 1, "{infos:?}");
    let info = &infos[0];
    assert_eq!(info.engine, Engine::VideoToolbox);
    assert_eq!(info.transfer, Transfer::ZeroCopy);
    assert_eq!(info.memory, OutputMemory::Exported);
    assert_eq!(info.buffers, 5);
    assert_eq!(info.format, PixelFormat::Nv12);
    assert_eq!((info.width, info.height), (1920, 1080));
    assert_eq!((info.coded_width, info.coded_height), (1920, 1088));
    assert_eq!(info.crop, Region::new(0, 0, 1920, 1080));
    let colorimetry = info.colorimetry.expect("the stream's colorimetry");
    assert!(colorimetry.range.is_some(), "{colorimetry:?}");
}

#[test]
fn holding_count_plus_one_frames_exhausts_the_pool_until_one_drops() {
    let _sessions = sessions::take(1);
    let test = "holding_count_plus_one_frames_exhausts_the_pool_until_one_drops";
    let Some(mut decoder) = open(test, OutputPool::Allocate { count: 3 }) else {
        return;
    };
    let mut run = Run::default();
    let mut held: Vec<DecodedFrame> = Vec::new();
    let mut first = None;
    let mut exhausted = 0;
    let stream = testdata(MAIN_BFRAMES_FILE);
    let units: Vec<&[u8]> = access_units(&stream).collect();
    let mut first_cleared = None;
    for (i, au) in units.iter().enumerate() {
        let mut refused = 0;
        let events = loop {
            match decoder.decode(au, i as u64) {
                Ok(events) => break events,
                Err(CodecError::PoolExhausted { held: h, pool }) => {
                    refused += 1;
                    exhausted += 1;
                    first.get_or_insert((h, pool));
                    assert_eq!(h, held.len(), "the caller holds every lent frame");
                    held.remove(0);
                }
                Err(e) => panic!("decode: {e}"),
            }
        };
        if refused > 0 {
            first_cleared.get_or_insert(refused);
        }
        held.extend(run.events(events));
    }
    loop {
        match decoder.flush() {
            Ok(events) => {
                held.extend(run.events(events));
                break;
            }
            Err(CodecError::PoolExhausted { .. }) => {
                held.remove(0);
            }
            Err(e) => panic!("flush: {e}"),
        }
    }
    assert_eq!(first, Some((4, 4)), "the first exhaustion");
    assert_eq!(
        first_cleared,
        Some(1),
        "the first refused access unit decodes once one frame drops"
    );
    assert!(exhausted > 0);
    assert_eq!(run.hashes, MAIN_BFRAMES, "no picture lost or repeated");
    let pts: BTreeSet<u64> = run.pts.iter().copied().collect();
    assert_eq!(pts, (0..MAIN_BFRAMES.len() as u64).collect());
}

#[test]
fn a_lent_frame_keeps_its_pixel_buffer_from_reuse() {
    let _sessions = sessions::take(1);
    let test = "a_lent_frame_keeps_its_pixel_buffer_from_reuse";
    let Some(mut decoder) = open(test, OutputPool::Allocate { count: 2 }) else {
        return;
    };
    let mut kept: Option<(DecodedFrame, u64, u32)> = None;
    let mut later = Vec::new();
    let mut take = |events: Vec<DecodeEvent>| {
        for e in events {
            let DecodeEvent::Frame(f) = e else { continue };
            if kept.is_none() {
                let (hash, id) = (picture_hash(f.tensor()), surface_id(&f));
                kept = Some((f, hash, id));
            } else {
                later.push(surface_id(&f));
            }
        }
    };
    for (i, au) in access_units(&testdata(MAIN_BFRAMES_FILE)).enumerate() {
        take(decoder.decode(au, i as u64).expect("decode"));
    }
    take(decoder.flush().expect("flush"));
    let (frame, hash, id) = kept.expect("a first frame");
    assert_eq!(later.len(), MAIN_BFRAMES.len() - 1);
    assert!(
        later.iter().all(|&l| l != id),
        "surface {id} was lent again while held: {later:?}"
    );
    assert_eq!(
        picture_hash(frame.tensor()),
        hash,
        "the held picture changed"
    );
    assert_eq!(hash, MAIN_BFRAMES[0]);
}

#[test]
fn repeated_parameter_sets_do_not_reconfigure() {
    let _sessions = sessions::take(1);
    let test = "repeated_parameter_sets_do_not_reconfigure";
    let Some(mut decoder) = open(test, OutputPool::default()) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let mut run = Run::default();
    let mut pts = 0u64;
    for _ in 0..2 {
        for au in access_units(&stream) {
            run.events(decoder.decode(au, pts).expect("decode"));
            pts += 1;
        }
    }
    run.events(decoder.flush().expect("flush"));
    assert_eq!(run.infos.len(), 1, "{:?}", run.infos);
    assert_eq!(run.hashes, [MAIN_BFRAMES, MAIN_BFRAMES].concat());
    let got: BTreeSet<u64> = run.pts.iter().copied().collect();
    assert_eq!(got.len(), run.pts.len(), "a PTS came back twice");
    assert_eq!(got, (0..pts).collect());
}

#[test]
fn access_units_before_the_parameter_sets_are_refused() {
    let _sessions = sessions::take(1);
    let test = "access_units_before_the_parameter_sets_are_refused";
    let Some(mut decoder) = open(test, OutputPool::default()) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let units: Vec<&[u8]> = access_units(&stream).collect();
    for (i, au) in units.iter().enumerate().take(5).skip(1) {
        let result = decoder.decode(au, i as u64);
        assert!(
            matches!(result, Err(CodecError::InvalidData(_))),
            "access unit {i}: {result:?}"
        );
    }
    let mut run = Run::default();
    for (i, au) in units.iter().enumerate() {
        run.events(decoder.decode(au, i as u64).expect("decode"));
    }
    run.events(decoder.flush().expect("flush"));
    assert_eq!(run.hashes, MAIN_BFRAMES);
}

/// An IOSurface-backed `fourcc` pixel buffer wrapped as a multiplane NV12
/// tensor.
fn corevideo_surface(w: usize, h: usize, fourcc: u32) -> TensorDyn {
    // SAFETY: the dictionaries hold live CF objects, `buffer` is a valid
    // out-pointer, and `CVPixelBufferCreate` returns +1.
    let buffer = unsafe {
        let surface = cf_dictionary(&[]).unwrap();
        let attributes =
            cf_dictionary(&[(kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr())]).unwrap();
        let mut buffer = std::ptr::null_mut();
        let status = CVPixelBufferCreate(
            std::ptr::null(),
            w,
            h,
            fourcc,
            attributes.as_ptr(),
            &mut buffer,
        );
        assert_eq!(status, kCVReturnSuccess);
        CfOwned::from_create(buffer).unwrap()
    };
    // SAFETY: the pixel buffer is live, and the tensor retains the surface.
    unsafe { TensorDyn::from_planar_iosurface(CVPixelBufferGetIOSurface(buffer.as_ptr()), None) }
        .unwrap()
}

/// `n` 1920x1080 `420v` CoreVideo surfaces.
fn multiplane_pool(n: usize) -> Vec<TensorDyn> {
    (0..n)
        .map(|_| corevideo_surface(1920, 1080, FOURCC_420V))
        .collect()
}

/// Decodes the whole stream into `pool`, holding no frame past its event.
fn decode_into(test: &str, pool: Vec<TensorDyn>) -> Option<(Run, Vec<Option<Colorimetry>>)> {
    let mut decoder = open(test, OutputPool::Provided(pool))?;
    let mut run = Run::default();
    let mut colours = Vec::new();
    let mut take = |run: &mut Run, events: Vec<DecodeEvent>| {
        for f in run.events(events) {
            colours.push(f.tensor().colorimetry());
        }
    };
    for (i, au) in access_units(&testdata(MAIN_BFRAMES_FILE)).enumerate() {
        take(&mut run, decoder.decode(au, i as u64).expect("decode"));
    }
    take(&mut run, decoder.flush().expect("flush"));
    Some((run, colours))
}

fn check_provided_run(run: &Run, colours: &[Option<Colorimetry>]) {
    assert_eq!(run.hashes, MAIN_BFRAMES);
    assert_eq!(run.infos.len(), 1, "{:?}", run.infos);
    let info = &run.infos[0];
    assert_eq!(info.transfer, Transfer::CpuCopy);
    assert_eq!(info.memory, OutputMemory::Imported);
    assert_eq!(info.buffers, 6);
    assert_eq!(info.engine, Engine::VideoToolbox);
    assert_eq!((info.width, info.height), (1920, 1080));
    assert!(info.colorimetry.is_some());
    assert!(
        colours.iter().all(|c| *c == info.colorimetry),
        "{colours:?}"
    );
}

#[test]
fn a_provided_multiplane_pool_receives_the_pictures() {
    let _sessions = sessions::take(1);
    let test = "a_provided_multiplane_pool_receives_the_pictures";
    let Some((run, colours)) = decode_into(test, multiplane_pool(6)) else {
        return;
    };
    check_provided_run(&run, &colours);
}

#[test]
fn a_provided_single_buffer_pool_receives_the_pictures() {
    let _sessions = sessions::take(1);
    let test = "a_provided_single_buffer_pool_receives_the_pictures";
    let pool = (0..6)
        .map(|_| {
            TensorDyn::image(
                1920,
                1080,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::DmaBuf),
                CpuAccess::ReadWrite,
            )
            .unwrap()
        })
        .collect();
    let Some((run, colours)) = decode_into(test, pool) else {
        return;
    };
    check_provided_run(&run, &colours);
}

/// The pictures one flush releases after the whole of
/// [`MAIN_BFRAMES_FILE`] is decoded with every frame dropped.
fn drain_size(test: &str) -> Option<usize> {
    let mut decoder = open(test, OutputPool::Allocate { count: 4 })?;
    for (i, au) in access_units(&testdata(MAIN_BFRAMES_FILE)).enumerate() {
        drop(decoder.decode(au, i as u64).expect("decode"));
    }
    let frames = decoder
        .flush()
        .expect("flush")
        .into_iter()
        .filter(|e| matches!(e, DecodeEvent::Frame(_)))
        .count();
    Some(frames)
}

/// Decodes the whole of [`MAIN_BFRAMES_FILE`] into `decoder`, keeping the
/// last `keep` frames and dropping the rest. A decode refused with
/// [`CodecError::PoolExhausted`] drops the oldest kept frame and decodes
/// the same access unit again.
fn decode_keeping(
    decoder: &mut VideoDecoder,
    run: &mut Run,
    keep: usize,
) -> std::collections::VecDeque<DecodedFrame> {
    let mut kept = std::collections::VecDeque::new();
    for (i, au) in access_units(&testdata(MAIN_BFRAMES_FILE)).enumerate() {
        loop {
            match decoder.decode(au, i as u64) {
                Ok(events) => {
                    kept.extend(run.events(events));
                    break;
                }
                Err(CodecError::PoolExhausted { .. }) => {
                    kept.pop_front().expect("a kept frame to drop");
                }
                Err(e) => panic!("decode {i}: {e}"),
            }
        }
        while kept.len() > keep {
            kept.pop_front();
        }
    }
    kept
}

#[test]
fn a_flush_that_needs_more_tensors_than_the_provided_pool_holds_is_refused() {
    let _sessions = sessions::take(1);
    let test = "a_flush_that_needs_more_tensors_than_the_provided_pool_holds_is_refused";
    let Some(drain) = drain_size(test) else {
        return;
    };
    eprintln!("{test}: a flush releases {drain} pictures");
    if drain < 2 {
        report_skip(&format!("{test} - a flush releases only {drain} pictures"));
        return;
    }
    let Some(mut decoder) = open(test, OutputPool::Provided(multiplane_pool(drain - 1))) else {
        return;
    };
    let mut run = Run::default();
    drop(decode_keeping(&mut decoder, &mut run, 0));
    let before = run.hashes.clone();
    for attempt in 0..2 {
        let e = decoder.flush().unwrap_err();
        assert!(
            matches!(&e, CodecError::InvalidConfig(why) if why.contains(&drain.to_string())),
            "flush {attempt}: {e}"
        );
    }
    assert_eq!(before, MAIN_BFRAMES[..MAIN_BFRAMES.len() - drain]);
}

#[test]
fn a_flush_while_the_caller_holds_provided_tensors_lends_nothing_until_they_drop() {
    let _sessions = sessions::take(1);
    let test = "a_flush_while_the_caller_holds_provided_tensors_lends_nothing_until_they_drop";
    let Some(drain) = drain_size(test) else {
        return;
    };
    let pool = drain + 1;
    let Some(mut decoder) = open(test, OutputPool::Provided(multiplane_pool(pool))) else {
        return;
    };
    let mut run = Run::default();
    let mut kept = decode_keeping(&mut decoder, &mut run, 2);
    assert_eq!(kept.len(), 2);
    let e = decoder.flush().unwrap_err();
    eprintln!("{test}: a flush releases {drain} pictures into {pool} tensors, 2 held: {e}");
    assert!(
        matches!(e, CodecError::PoolExhausted { held: 2, pool: p } if p == pool),
        "{e}"
    );
    assert_eq!(run.hashes, MAIN_BFRAMES[..MAIN_BFRAMES.len() - drain]);
    kept.pop_front();
    let frames = run.events(decoder.flush().expect("flush once a frame has dropped"));
    assert_eq!(frames.len(), drain);
    assert_eq!(run.hashes, MAIN_BFRAMES);
    let mut pts = run.pts.clone();
    pts.dedup();
    assert_eq!(pts.len(), MAIN_BFRAMES.len());
}

/// A VideoToolbox decoder backend with `output`, or `None` after reporting
/// why the test skips.
fn open_backend(test: &str, output: OutputPool) -> Option<decoder::VtDecoder> {
    let _ = env_logger::builder().is_test(true).try_init();
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), test) {
        return None;
    }
    let config = DecoderConfig::h264()
        .with_engine(Engine::VideoToolbox)
        .with_output(output);
    Some(decoder::VtDecoder::open(config).expect("decoder open"))
}

#[test]
fn a_flush_after_a_session_failure_returns_the_decoded_pictures() {
    let _sessions = sessions::take(1);
    let test = "a_flush_after_a_session_failure_returns_the_decoded_pictures";
    let Some(mut decoder) = open_backend(test, OutputPool::Allocate { count: 4 }) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let units: Vec<&[u8]> = access_units(&stream).collect();
    let mut run = Run::default();
    for (i, au) in units.iter().take(10).enumerate() {
        run.events(decoder.decode(au, i as u64).expect("decode"));
    }
    decoder.invalidate_session_for_test();
    run.events(decoder.flush().expect("flush after the session failed"));
    assert_eq!(run.hashes, MAIN_BFRAMES[..10]);
    assert!(decoder.flush().expect("a second flush").is_empty());
    let mut again = Run::default();
    for (i, au) in units.iter().enumerate() {
        again.events(
            decoder
                .decode(au, i as u64)
                .expect("decode on a new session"),
        );
    }
    again.events(decoder.flush().expect("flush"));
    assert_eq!(again.hashes, MAIN_BFRAMES);
}

#[test]
fn a_picture_decoded_before_a_decode_error_is_returned_corrupted() {
    let _sessions = sessions::take(1);
    let test = "a_picture_decoded_before_a_decode_error_is_returned_corrupted";
    let bad_data: OSStatus = -12909;
    let Some(mut decoder) = open_backend(test, OutputPool::Allocate { count: 4 }) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let first = access_units(&stream).next().expect("an access unit");
    decoder.fail_next_decode_for_test(bad_data);
    let e = decoder.decode(first, 0).unwrap_err();
    assert!(
        matches!(&e, CodecError::Device { op, .. } if op == "VTDecompressionSessionDecodeFrame"),
        "{e}"
    );
    let frames: Vec<DecodedFrame> = decoder
        .flush()
        .expect("flush")
        .into_iter()
        .filter_map(|e| match e {
            DecodeEvent::Frame(f) => Some(f),
            DecodeEvent::Reconfigured(_) => None,
        })
        .collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].pts(), 0);
    assert!(frames[0].corrupted());
}

#[test]
fn a_chroma_plane_must_start_after_the_luma_plane() {
    let (pitches, rows) = ([2048, 2048], [1080, 540]);
    let luma = 0x1000_0000;
    let luma_len = 2048 * 1080;
    assert_eq!(
        chroma_offset(luma, luma + luma_len, pitches, rows),
        Some((luma_len, luma_len + 2048 * 540))
    );
    assert_eq!(chroma_offset(luma, luma - 4096, pitches, rows), None);
    assert_eq!(chroma_offset(luma, luma - 1, pitches, rows), None);
    assert_eq!(
        chroma_offset(luma, luma + luma_len - 1, pitches, rows),
        None
    );
    assert_eq!(chroma_offset(luma, luma, pitches, rows), None);
    assert_eq!(chroma_offset(0, luma_len, pitches, rows), None);
    assert_eq!(chroma_offset(luma, 0, pitches, rows), None);
    assert_eq!(
        chroma_offset(luma, luma + 16, [usize::MAX, 1], [2, 1]),
        None
    );
    assert_eq!(
        chroma_offset(luma, luma + luma_len, [2048, usize::MAX / 2], [1080, 2]),
        None
    );
}

#[test]
fn capabilities_list_the_h264_decoder() {
    let mut caps = Vec::new();
    capabilities(&mut caps);
    let decoders: Vec<_> = caps
        .iter()
        .filter(|c| c.direction == Direction::Decode)
        .collect();
    assert_eq!(decoders.len(), 1, "{caps:?}");
    let d = decoders[0];
    assert_eq!((d.engine, d.codec), (Engine::VideoToolbox, Codec::H264));
    assert_eq!(d.formats, vec![PixelFormat::Nv12]);
    assert_eq!(d.transfer, Transfer::ZeroCopy);
    assert!(!d.device.is_empty());
}

#[test]
fn cf_owned_releases_what_it_holds() {
    let held = cf_string("cf_owned_releases_what_it_holds").unwrap();
    let ptr = held.as_ptr();
    // SAFETY: `held` keeps the string live for every count.
    let count = || unsafe { CFGetRetainCount(ptr) };
    let base = count();
    let copy = held.clone();
    assert_eq!(count(), base + 1);
    // SAFETY: `ptr` is live, held by `held`.
    let again = unsafe { CfOwned::retained(ptr) }.unwrap();
    assert_eq!(count(), base + 2);
    drop(copy);
    drop(again);
    assert_eq!(count(), base);
    // SAFETY: null is allowed.
    assert!(unsafe { CfOwned::retained(std::ptr::null()) }.is_none());
}

#[test]
fn a_failed_status_names_the_operation() {
    let e = status("VTSessionSetProperty", -12902);
    assert!(
        matches!(&e, CodecError::Device { op, source }
            if op == "VTSessionSetProperty" && source.to_string() == "OSStatus -12902"),
        "{e}"
    );
}

#[test]
fn core_media_structs_have_the_sdk_layout() {
    use std::mem::{align_of, size_of};
    assert_eq!((size_of::<CMTime>(), align_of::<CMTime>()), (24, 4));
    assert_eq!(
        (
            size_of::<CMSampleTimingInfo>(),
            align_of::<CMSampleTimingInfo>()
        ),
        (72, 4)
    );
    assert_eq!(
        (
            size_of::<VTDecompressionOutputCallbackRecord>(),
            align_of::<VTDecompressionOutputCallbackRecord>()
        ),
        (16, 4)
    );
    assert_eq!(
        (
            std::mem::offset_of!(CMTime, timescale),
            std::mem::offset_of!(CMTime, flags),
            std::mem::offset_of!(CMTime, epoch)
        ),
        (8, 12, 16)
    );
    assert_eq!(
        std::mem::offset_of!(CMSampleTimingInfo, decode_time_stamp),
        48
    );
    assert_eq!(
        std::mem::offset_of!(VTDecompressionOutputCallbackRecord, ref_con),
        8
    );
}

/// A VideoToolbox encoder for `config`, or `None` after reporting why the
/// test skips. Prints whether the encoder is hardware.
fn open_encoder(test: &str, config: EncoderConfig) -> Option<encoder::VtEncoder> {
    let _ = env_logger::builder().is_test(true).try_init();
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), test) {
        return None;
    }
    match encoder::VtEncoder::open(config.with_engine(Engine::VideoToolbox)) {
        Ok(e) => {
            let info = e.info();
            eprintln!("{test}: {} hardware {}", info.device, info.hardware);
            Some(e)
        }
        Err(CodecError::NoDevice(why)) => {
            report_skip(&format!("{test} - {why}"));
            None
        }
        Err(e) => panic!("encoder open failed: {e}"),
    }
}

/// Encodes `src`, submitting it again while the encoder refuses it with
/// [`CodecError::Timeout`].
fn encode_retrying(
    enc: &mut encoder::VtEncoder,
    src: &TensorDyn,
    opts: &FrameOptions,
) -> Option<EncodedFrame> {
    encode_retry::retry(
        || enc.encode(src, opts),
        |e| matches!(e, CodecError::Timeout(_)),
    )
    .unwrap_or_else(|e| panic!("encode {}: {e}", opts.pts))
}

fn image(w: usize, h: usize, format: PixelFormat) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        format,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap()
}

/// Paints an NV12 image with a gradient that moves with `frame`.
fn paint_nv12(t: &TensorDyn, frame: usize) {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let mut px = t.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        for x in 0..w {
            px[y * stride + x] = ((x / 4 + y / 4 + 3 * frame) & 0xff) as u8;
        }
    }
    for y in 0..h / 2 {
        let row = (h + y) * stride;
        for x in 0..w {
            px[row + x] = (96 + (x / 8 + y / 8 + frame) % 64) as u8;
        }
    }
}

/// Encodes `frames` NV12 pictures of `src` with pts `0..frames`, then
/// flushes, returning every access unit.
fn encode_nv12(enc: &mut encoder::VtEncoder, src: &TensorDyn, frames: usize) -> Vec<EncodedFrame> {
    let mut out = Vec::new();
    for i in 0..frames {
        paint_nv12(src, i);
        let opts = FrameOptions {
            pts: i as u64,
            ..FrameOptions::default()
        };
        out.extend(encode_retrying(enc, src, &opts));
    }
    out.extend(enc.flush().expect("flush"));
    out
}

/// Whether `au` holds an SPS and a PPS, and whether it holds an IDR slice.
fn parameter_sets_and_idr(au: &[u8]) -> (bool, bool) {
    let types: Vec<u8> = h264::nal_units(au).map(|n| n[0] & 0x1f).collect();
    (
        types.contains(&h264::NAL_SPS) && types.contains(&h264::NAL_PPS),
        types.contains(&h264::NAL_IDR),
    )
}

/// The first SPS in `au`.
fn sps_of(au: &[u8]) -> Sps {
    let nal = h264::nal_units(au)
        .find(|n| n[0] & 0x1f == h264::NAL_SPS)
        .expect("an SPS");
    Sps::parse(nal).expect("a valid SPS")
}

/// The VUI colour description (primaries, transfer, matrix) and full-range
/// flag of an SPS.
type VuiColour = (Option<(u8, u8, u8)>, Option<bool>);

/// The VUI colour of the first SPS in `au`.
fn vui_colour(au: &[u8]) -> VuiColour {
    let vui = sps_of(au).vui.expect("a VUI");
    (
        vui.colour.map(|c| (c.primaries, c.transfer, c.matrix)),
        vui.full_range,
    )
}

/// The VUI colour of the first access unit of a 3-frame stream from `src`.
fn stream_colour(test: &str, src: &TensorDyn) -> Option<VuiColour> {
    let (w, h) = (src.width().unwrap(), src.height().unwrap());
    let mut enc = open_encoder(
        test,
        EncoderConfig::h264(w as u32, h as u32, PixelFormat::Nv12, 30.0),
    )?;
    let out = encode_nv12(&mut enc, src, 3);
    assert_eq!(out.len(), 3);
    Some(vui_colour(&out[0].data))
}

fn full_range_smpte170m() -> Colorimetry {
    Colorimetry {
        space: Some(ColorSpace::Smpte170m),
        transfer: Some(ColorTransfer::Bt709),
        encoding: Some(ColorEncoding::Bt601),
        range: Some(ColorRange::Full),
    }
}

/// Decodes `units` on VideoToolbox and returns each frame.
fn decode_all(test: &str, units: &[EncodedFrame]) -> Option<(Vec<StreamInfo>, Vec<DecodedFrame>)> {
    let mut decoder = open(test, OutputPool::Allocate { count: units.len() })?;
    let mut infos = Vec::new();
    let mut frames = Vec::new();
    let mut take = |events: Vec<DecodeEvent>| {
        for e in events {
            match e {
                DecodeEvent::Frame(f) => frames.push(f),
                DecodeEvent::Reconfigured(info) => infos.push(info),
            }
        }
    };
    for au in units {
        take(decoder.decode(&au.data, au.pts).expect("decode"));
    }
    take(decoder.flush().expect("flush"));
    Some((infos, frames))
}

/// The mean of the luma samples of a decoded frame.
fn mean_luma(t: &TensorDyn) -> f64 {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let px = t.map_bytes(CpuAccess::Read).unwrap();
    let sum: u64 = (0..h)
        .flat_map(|y| &px[y * stride..y * stride + w])
        .map(|&b| u64::from(b))
        .sum();
    sum as f64 / (w * h) as f64
}

#[test]
fn without_repeated_headers_only_the_first_idr_carries_parameter_sets() {
    let _sessions = sessions::take(1);
    let test = "without_repeated_headers_only_the_first_idr_carries_parameter_sets";
    let mut config = EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0);
    config.gop = Some(10);
    config.repeat_headers = false;
    let Some(mut enc) = open_encoder(test, config) else {
        return;
    };
    let out = encode_nv12(&mut enc, &image(640, 480, PixelFormat::Nv12), 30);
    assert_eq!(out.len(), 30);
    assert_eq!(parameter_sets_and_idr(&out[0].data), (true, true));
    assert!(out[0].keyframe);
    for i in [10, 20] {
        assert_eq!(
            parameter_sets_and_idr(&out[i].data),
            (false, true),
            "access unit {i}"
        );
        assert!(out[i].keyframe, "access unit {i}");
    }
    for (i, au) in out.iter().enumerate().skip(1) {
        assert!(!parameter_sets_and_idr(&au.data).0, "access unit {i}");
    }
}

#[test]
fn the_encoder_emits_no_b_frames() {
    let _sessions = sessions::take(1);
    let test = "the_encoder_emits_no_b_frames";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let out = encode_nv12(&mut enc, &image(640, 480, PixelFormat::Nv12), 30);
    assert_eq!(out.len(), 30);
    let mut sets = ParameterSets::default();
    for (i, au) in out.iter().enumerate() {
        assert_eq!(au.pts, i as u64, "access unit {i}");
        for nal in h264::nal_units(&au.data) {
            match nal[0] & 0x1f {
                h264::NAL_SPS | h264::NAL_PPS => {
                    sets.update(nal).unwrap();
                }
                1 | h264::NAL_IDR => {
                    let slice = SliceHeader::parse(nal, &sets).unwrap();
                    assert!(
                        matches!(slice.slice_type % 5, 0 | 2),
                        "access unit {i} has slice type {}",
                        slice.slice_type
                    );
                }
                _ => {}
            }
        }
    }
}

#[test]
fn the_frame_after_a_session_error_is_an_idr() {
    let _sessions = sessions::take(1);
    let test = "the_frame_after_a_session_error_is_an_idr";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let src = image(640, 480, PixelFormat::Nv12);
    let mut out = Vec::new();
    let opts = |pts| FrameOptions {
        pts,
        ..FrameOptions::default()
    };
    for i in 0..5 {
        paint_nv12(&src, i);
        out.extend(enc.encode(&src, &opts(i as u64)).expect("encode"));
    }
    enc.invalidate_session_for_test();
    paint_nv12(&src, 5);
    let e = enc
        .encode(&src, &opts(5))
        .and_then(|_| enc.flush().map(drop))
        .unwrap_err();
    assert!(matches!(e, CodecError::Device { .. }), "{e}");
    paint_nv12(&src, 6);
    out.extend(enc.encode(&src, &opts(6)).expect("encode after the error"));
    out.extend(enc.flush().expect("flush"));
    let after = out
        .iter()
        .find(|f| f.pts == 6)
        .expect("the frame after the error");
    assert_eq!(parameter_sets_and_idr(&after.data), (true, true));
    assert!(after.keyframe);
    assert!(out.iter().all(|f| f.pts != 5));
}

#[test]
fn untagged_sources_follow_the_height_rule() {
    let test = "untagged_sources_follow_the_height_rule";
    for ((w, h), colour) in [((640, 480), (6, 1, 6)), ((1280, 720), (1, 1, 1))] {
        let Some(got) = stream_colour(test, &image(w, h, PixelFormat::Nv12)) else {
            return;
        };
        assert_eq!(got, (Some(colour), Some(false)), "{w}x{h}");
    }
}

#[test]
fn a_full_range_source_is_described_as_full_range() {
    let _sessions = sessions::take(1);
    let test = "a_full_range_source_is_described_as_full_range";
    let src = image(640, 480, PixelFormat::Nv12).with_colorimetry(full_range_smpte170m());
    let Some(got) = stream_colour(test, &src) else {
        return;
    };
    assert_eq!(got, (Some((6, 1, 6)), Some(true)));
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Yuyv, 30.0))
    else {
        return;
    };
    let src = image(640, 480, PixelFormat::Yuyv).with_colorimetry(full_range_smpte170m());
    let mut out = Vec::new();
    for pts in 0..3 {
        let opts = FrameOptions {
            pts,
            ..FrameOptions::default()
        };
        out.extend(enc.encode(&src, &opts).expect("encode"));
    }
    out.extend(enc.flush().expect("flush"));
    assert_eq!(
        vui_colour(&out[0].data),
        (Some((6, 1, 6)), Some(true)),
        "YUYV"
    );
}

#[test]
fn a_source_with_other_colorimetry_than_the_stream_is_refused() {
    let _sessions = sessions::take(1);
    let test = "a_source_with_other_colorimetry_than_the_stream_is_refused";
    let Some(mut enc) = open_encoder(
        test,
        EncoderConfig::h264(1280, 720, PixelFormat::Nv12, 30.0),
    ) else {
        return;
    };
    let src = image(1280, 720, PixelFormat::Nv12);
    paint_nv12(&src, 0);
    enc.encode(&src, &FrameOptions::default()).expect("encode");
    let full = src.with_colorimetry(Colorimetry {
        range: Some(ColorRange::Full),
        ..Colorimetry::default()
    });
    let e = enc
        .encode(
            &full,
            &FrameOptions {
                pts: 1,
                ..FrameOptions::default()
            },
        )
        .unwrap_err();
    assert!(matches!(e, CodecError::InvalidInput(_)), "{e}");
}

#[test]
fn decoded_frames_carry_the_stream_colorimetry() {
    let _sessions = sessions::take(2);
    let test = "decoded_frames_carry_the_stream_colorimetry";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let src = image(640, 480, PixelFormat::Nv12).with_colorimetry(full_range_smpte170m());
    let units = encode_nv12(&mut enc, &src, 10);
    let Some((infos, frames)) = decode_all(test, &units) else {
        return;
    };
    assert_eq!(infos.len(), 1, "{infos:?}");
    assert_eq!(infos[0].colorimetry, Some(full_range_smpte170m()));
    assert_eq!(frames.len(), 10);
    for f in &frames {
        assert_eq!(
            f.tensor().colorimetry(),
            Some(full_range_smpte170m()),
            "frame {}",
            f.pts()
        );
    }
}

#[test]
fn yuyv_from_a_hal_tensor_is_copied() {
    let _sessions = sessions::take(2);
    let test = "yuyv_from_a_hal_tensor_is_copied";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Yuyv, 30.0))
    else {
        return;
    };
    let src = image(640, 480, PixelFormat::Yuyv);
    let level = |i: usize| (40 + 15 * i) as u8;
    let mut units = Vec::new();
    for i in 0..10 {
        {
            let stride = src.effective_row_stride().unwrap();
            let mut px = src.map_bytes(CpuAccess::Write).unwrap();
            for y in 0..480 {
                for pair in px[y * stride..y * stride + 1280].chunks_exact_mut(4) {
                    pair.copy_from_slice(&[level(i), 128, level(i), 128]);
                }
            }
        }
        let opts = FrameOptions {
            pts: i as u64,
            ..FrameOptions::default()
        };
        units.extend(encode_retrying(&mut enc, &src, &opts));
    }
    assert_eq!(enc.info().transfer, Transfer::CpuCopy);
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
    assert!(units[0].keyframe);
    let Some((_, frames)) = decode_all(test, &units) else {
        return;
    };
    assert_eq!(frames.len(), 10);
    for f in &frames {
        let want = f64::from(level(f.pts() as usize));
        let got = mean_luma(f.tensor());
        assert!(
            (got - want).abs() <= 3.0,
            "frame {}: mean luma {got}, painted {want}",
            f.pts()
        );
    }
}

#[test]
fn capabilities_list_the_h264_encoders() {
    let mut caps = Vec::new();
    capabilities(&mut caps);
    let encoders: Vec<_> = caps
        .iter()
        .filter(|c| c.direction == Direction::Encode)
        .collect();
    assert!(!encoders.is_empty(), "{caps:?}");
    for e in encoders {
        eprintln!("{}: hardware {} {:?}", e.device, e.hardware, e.transfer);
        assert_eq!((e.engine, e.codec), (Engine::VideoToolbox, Codec::H264));
        assert_eq!(e.formats, vec![PixelFormat::Nv12, PixelFormat::Yuyv]);
        let best = if e.hardware {
            Transfer::ZeroCopy
        } else {
            Transfer::CpuCopy
        };
        assert_eq!(e.transfer, best);
        assert!(!e.device.is_empty());
    }
}

#[test]
fn ten_seconds_at_1080p30_keep_near_the_target_bit_rate() {
    let _sessions = sessions::take(1);
    let test = "ten_seconds_at_1080p30_keep_near_the_target_bit_rate";
    let config = EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0);
    let target = config.target_bitrate();
    let Some(mut enc) = open_encoder(test, config) else {
        return;
    };
    let (w, h) = (1920, 1080);
    let src = image(w, h, PixelFormat::Nv12);
    let stride = src.effective_row_stride().unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let pictures: Vec<Vec<u8>> = (0..30)
        .map(|i| {
            paint_nv12(&src, i);
            let px = src.map_bytes(CpuAccess::Read).unwrap();
            (0..h * 3 / 2)
                .flat_map(|y| &px[y * stride..y * stride + w])
                .map(|&b| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b.saturating_add((state >> 60) as u8).saturating_sub(8)
                })
                .collect()
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..300 {
        {
            let picture = &pictures[i % pictures.len()];
            let mut px = src.map_bytes(CpuAccess::Write).unwrap();
            for y in 0..h * 3 / 2 {
                px[y * stride..y * stride + w].copy_from_slice(&picture[y * w..(y + 1) * w]);
            }
        }
        let opts = FrameOptions {
            pts: i as u64,
            ..FrameOptions::default()
        };
        out.extend(encode_retrying(&mut enc, &src, &opts));
    }
    out.extend(enc.flush().expect("flush"));
    assert_eq!(out.len(), 300);
    let bits: usize = out.iter().map(|f| f.data.len() * 8).sum();
    let bps = bits as f64 / 10.0;
    eprintln!(
        "{test}: {bps:.0} bps over 10 s, target {target} bps ({:.2} of the target)",
        bps / f64::from(target)
    );
    assert!(
        bps <= f64::from(target) * 1.25,
        "{bps:.0} bps is over the {target} bps target"
    );
}

/// Paints a multiplane NV12 image with a gradient that moves with `frame`.
fn paint_planes(t: &TensorDyn, frame: usize) {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    {
        let mut px = t.map_bytes(CpuAccess::Write).unwrap();
        for y in 0..h {
            for x in 0..w {
                px[y * stride + x] = ((x / 4 + y / 4 + 3 * frame) & 0xff) as u8;
            }
        }
    }
    let chroma = t.chroma_dyn().unwrap();
    let cstride = chroma.effective_row_stride().unwrap();
    let mut px = chroma.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h / 2 {
        px[y * cstride..y * cstride + w].fill(128);
    }
}

/// Encodes `frames` pictures of the multiplane `src` with pts `0..frames`,
/// then flushes.
fn encode_planes(
    enc: &mut encoder::VtEncoder,
    src: &TensorDyn,
    frames: usize,
) -> crate::Result<Vec<EncodedFrame>> {
    let mut out = Vec::new();
    for i in 0..frames {
        paint_planes(src, i);
        let opts = FrameOptions {
            pts: i as u64,
            ..FrameOptions::default()
        };
        out.extend(enc.encode(src, &opts)?);
    }
    out.extend(enc.flush()?);
    Ok(out)
}

#[test]
fn constrained_baseline_streams_say_so_in_the_sps() {
    let _sessions = sessions::take(2);
    let test = "constrained_baseline_streams_say_so_in_the_sps";
    for profile in [H264Profile::ConstrainedBaseline, H264Profile::Baseline] {
        let mut config = EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0);
        config.profile = Some(profile);
        let Some(mut enc) = open_encoder(test, config) else {
            return;
        };
        let units = encode_nv12(&mut enc, &image(640, 480, PixelFormat::Nv12), 10);
        assert_eq!(units.len(), 10);
        for au in units.iter().filter(|au| au.keyframe) {
            let sps = h264::nal_units(&au.data)
                .find(|n| n[0] & 0x1f == h264::NAL_SPS)
                .expect("an SPS");
            eprintln!("{test}: {profile:?} constraint flags {:#04x}", sps[2]);
            assert_eq!(sps[1], 66, "{profile:?}");
            if profile == H264Profile::ConstrainedBaseline {
                assert_eq!(sps[2] & 0xc0, 0xc0, "{profile:?}: {:#04x}", sps[2]);
            }
        }
        let Some((_, frames)) = decode_all(test, &units) else {
            return;
        };
        assert_eq!(frames.len(), 10, "{profile:?}");
        assert!(frames.iter().all(|f| !f.corrupted()), "{profile:?}");
    }
}

#[test]
fn an_untagged_420f_surface_is_encoded_as_full_range() {
    let _sessions = sessions::take(2);
    let test = "an_untagged_420f_surface_is_encoded_as_full_range";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let src = corevideo_surface(640, 480, FOURCC_420F);
    assert_eq!(src.colorimetry(), None);
    let units = encode_planes(&mut enc, &src, 5).expect("encode");
    assert_eq!(units.len(), 5);
    assert_eq!(vui_colour(&units[0].data), (Some((6, 1, 6)), Some(true)));
    let Some((infos, frames)) = decode_all(test, &units) else {
        return;
    };
    assert_eq!(frames.len(), 5);
    let range = infos[0].colorimetry.and_then(|c| c.range);
    assert_eq!(range, Some(ColorRange::Full));
}

#[test]
fn a_420f_surface_tagged_limited_is_refused() {
    let _sessions = sessions::take(1);
    let test = "a_420f_surface_tagged_limited_is_refused";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let src = corevideo_surface(640, 480, FOURCC_420F).with_colorimetry(Colorimetry {
        range: Some(ColorRange::Limited),
        ..Colorimetry::default()
    });
    let e = enc.encode(&src, &FrameOptions::default()).unwrap_err();
    assert!(matches!(e, CodecError::InvalidInput(_)), "{e}");
}

#[test]
fn a_stream_after_a_flush_may_change_colorimetry() {
    let _sessions = sessions::take(1);
    let test = "a_stream_after_a_flush_may_change_colorimetry";
    let Some(mut enc) = open_encoder(test, EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0))
    else {
        return;
    };
    let untagged = image(640, 480, PixelFormat::Nv12);
    let first = encode_nv12(&mut enc, &untagged, 3);
    assert_eq!(first.len(), 3);
    assert_eq!(vui_colour(&first[0].data).1, Some(false));
    let full = image(640, 480, PixelFormat::Nv12).with_colorimetry(Colorimetry {
        range: Some(ColorRange::Full),
        ..Colorimetry::default()
    });
    let second = encode_nv12(&mut enc, &full, 3);
    assert_eq!(second.len(), 3);
    assert_eq!(parameter_sets_and_idr(&second[0].data), (true, true));
    assert_eq!(vui_colour(&second[0].data).1, Some(true));
}
