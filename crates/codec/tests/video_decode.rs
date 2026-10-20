// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware H.264 decoding. Each test skips (prints `SKIPPED` and passes)
//! where there is no decoder or no DMA-BUF heap; under
//! `HAL_TEST_REQUIRE_DMA=1` a missing heap fails instead.
//!
//! The expected pictures are FNV-1a hashes of the visible NV12 bytes of
//! each frame as FFmpeg decodes the fixture, in presentation order: H.264
//! decoding is exact, so a hardware decoder must produce the same bytes.

use std::collections::{BTreeSet, VecDeque};

use edgefirst_codec::video::{
    access_units, DecodeEvent, DecodedFrame, DecoderConfig, EncoderConfig, FrameOptions,
    OutputMemory, OutputPool, StreamInfo, VideoDecoder, VideoEncoder,
};
use edgefirst_codec::CodecError;
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

/// 30 frames of 1920x1080 Main profile with two B-frames between
/// references, GOP 15 (x264).
const MAIN_BFRAMES_FILE: &str = "h264_1080p_main_bframes.h264";
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

/// 10 frames of 640x480 then 10 of 1280x720, Baseline profile, each part
/// starting with its own SPS, PPS and IDR (x264).
const RESOLUTION_CHANGE_FILE: &str = "h264_resolution_change.h264";
const RESOLUTION_CHANGE: [u64; 20] = [
    0xa1bf4f84cdf2b986,
    0xfca591ffd85329b5,
    0x2004b49ed594567d,
    0xee49c411d81e999f,
    0x5071c5bca7fda7cf,
    0x5b7436a0a696718d,
    0x92f26f3fa2166e96,
    0x33839395f3d42a9e,
    0xe14b0deafbfa2387,
    0x9ff5cdb1638a095c,
    0x338dab745595f627,
    0xf8572ab0727cc038,
    0xd944b446041ec3dd,
    0xa7e631c3f6624e4d,
    0x780dec02b9ca0304,
    0xd893a8335c816c48,
    0x83d0f094f35fc4eb,
    0x865a9002bce7cc55,
    0x50131b5fbe969272,
    0xf07d7d444f1f9dd6,
];

fn skip(test: &str, why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {test} - {why}");
}

/// The DMA require gate: a skip here fails under `HAL_TEST_REQUIRE_DMA=1`.
#[path = "../../tensor/tests/support/dma_require.rs"]
mod dma_require;

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

/// Opens a decoder, or `None` (after printing why) when the platform has
/// none.
fn open(test: &str, config: DecoderConfig) -> Option<VideoDecoder> {
    let _ = env_logger::builder().is_test(true).try_init();
    if !dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), test) {
        return None;
    }
    match VideoDecoder::new(config) {
        Ok(d) => Some(d),
        Err(CodecError::NoDevice(why)) => {
            skip(test, &why);
            None
        }
        Err(e) => panic!("decoder open failed: {e}"),
    }
}

/// FNV-1a over the visible NV12 bytes of a frame: luma rows, then chroma
/// rows, without row padding.
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
    let chroma = t.chroma_dyn().expect("NV12 frames have a chroma plane");
    let cstride = chroma.effective_row_stride().unwrap();
    let cmap = chroma.map_bytes(CpuAccess::Read).unwrap();
    for y in 0..rows.div_ceil(2) {
        eat(&cmap[y * cstride..y * cstride + w]);
    }
    h
}

/// Everything a decode run produced.
#[derive(Default)]
struct Run {
    hashes: Vec<u64>,
    pts: Vec<u64>,
    sizes: Vec<(usize, usize)>,
    infos: Vec<StreamInfo>,
    corrupted: usize,
}

impl Run {
    fn frame(&mut self, f: &DecodedFrame) {
        let t = f.tensor();
        assert_eq!(t.format(), Some(PixelFormat::Nv12));
        assert_eq!(
            t.memory(),
            TensorMemory::DmaBuf,
            "frames are DMA-BUF tensors"
        );
        self.hashes.push(picture_hash(t));
        self.pts.push(f.pts());
        self.sizes.push((t.width().unwrap(), t.height().unwrap()));
        self.corrupted += usize::from(f.corrupted());
    }

    /// Records `events`, returning the frames for the caller to hold or
    /// drop.
    fn events(&mut self, events: Vec<DecodeEvent>) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        for e in events {
            match e {
                DecodeEvent::Frame(f) => {
                    self.frame(&f);
                    frames.push(f);
                }
                DecodeEvent::Reconfigured(info) => self.infos.push(info),
            }
        }
        frames
    }
}

/// Decodes `stream` access unit by access unit, tagging each with its
/// index, then flushes.
fn decode_all(decoder: &mut VideoDecoder, stream: &[u8], run: &mut Run, first_pts: u64) -> usize {
    let mut n = 0;
    for au in access_units(stream) {
        run.events(decoder.decode(au, first_pts + n as u64).expect("decode"));
        n += 1;
    }
    run.events(decoder.flush().expect("flush"));
    n
}

/// Every PTS in `first..first + n` came back exactly once.
fn assert_each_pts_once(pts: &[u64], first: u64, n: usize) {
    let got: BTreeSet<u64> = pts.iter().copied().collect();
    assert_eq!(got.len(), pts.len(), "a PTS came back twice: {pts:?}");
    assert_eq!(got, (first..first + n as u64).collect(), "PTS: {pts:?}");
}

#[test]
fn decodes_main_profile_with_b_frames_exactly() {
    let Some(mut decoder) = open(
        "decodes_main_profile_with_b_frames_exactly",
        DecoderConfig::h264(),
    ) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let mut run = Run::default();
    let units = decode_all(&mut decoder, &stream, &mut run, 0);

    assert_eq!(units, MAIN_BFRAMES.len());
    assert_eq!(run.hashes.len(), units, "one picture per access unit");
    assert_eq!(
        run.hashes, MAIN_BFRAMES,
        "pictures differ from the reference decode"
    );
    assert_each_pts_once(&run.pts, 0, units);
    assert_eq!(run.corrupted, 0);

    assert_eq!(run.infos.len(), 1, "one format: {:?}", run.infos);
    let info = &run.infos[0];
    assert_eq!((info.width, info.height), (1920, 1080));
    assert!(
        info.coded_height >= 1080 && info.coded_height % 16 == 0,
        "{info:?}"
    );
    assert_eq!(info.format, PixelFormat::Nv12);
    assert_eq!(decoder.stream_info(), Some(info));
}

#[test]
fn follows_a_mid_stream_resolution_change() {
    let Some(mut decoder) = open(
        "follows_a_mid_stream_resolution_change",
        DecoderConfig::h264(),
    ) else {
        return;
    };
    let stream = testdata(RESOLUTION_CHANGE_FILE);
    let mut run = Run::default();
    let units = decode_all(&mut decoder, &stream, &mut run, 0);

    assert_eq!(run.hashes.len(), units);
    assert_eq!(run.hashes, RESOLUTION_CHANGE);
    let sizes: Vec<(usize, usize)> = run.infos.iter().map(|i| (i.width, i.height)).collect();
    assert_eq!(sizes, vec![(640, 480), (1280, 720)]);
    assert!(run.sizes[..10].iter().all(|&s| s == (640, 480)));
    assert!(run.sizes[10..].iter().all(|&s| s == (1280, 720)));
    assert_each_pts_once(&run.pts, 0, units);
}

#[test]
fn decodes_into_a_provided_pool() {
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "decodes_into_a_provided_pool",
    ) {
        return;
    }
    // Larger than any device's NV12 1080p buffer (1.5 bytes per pixel of a
    // padded picture).
    let size = 2048 * 1088 * 2;
    let pool: Vec<TensorDyn> = (0..12)
        .map(|_| TensorDyn::new(&[size], DType::U8, Some(TensorMemory::DmaBuf), None).unwrap())
        .collect();
    let config = DecoderConfig::h264().with_output(OutputPool::Provided(pool));
    let Some(mut decoder) = open("decodes_into_a_provided_pool", config) else {
        return;
    };
    let mut run = Run::default();
    let units = decode_all(&mut decoder, &testdata(MAIN_BFRAMES_FILE), &mut run, 0);
    assert_eq!(run.hashes.len(), units);
    assert_eq!(run.hashes, MAIN_BFRAMES);
    assert_eq!(run.infos[0].buffers, 12);
    assert_eq!(run.infos[0].memory, OutputMemory::Imported);
}

#[test]
fn frames_held_across_a_resolution_change_keep_their_provided_buffers() {
    let test = "frames_held_across_a_resolution_change_keep_their_provided_buffers";
    if !dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), test) {
        return;
    }
    let size = 2048 * 1088 * 2;
    let pool: Vec<TensorDyn> = (0..14)
        .map(|_| TensorDyn::new(&[size], DType::U8, Some(TensorMemory::DmaBuf), None).unwrap())
        .collect();
    let config = DecoderConfig::h264().with_output(OutputPool::Provided(pool));
    let Some(mut decoder) = open(test, config) else {
        return;
    };
    // The new format's pool is over the same tensors: the last pictures of
    // the old one, still held, must not be decoded over.
    const HOLD: usize = 3;
    let mut held: VecDeque<(DecodedFrame, u64)> = VecDeque::new();
    let mut changed = false;
    let mut frames = 0;
    let mut take = |events: Vec<DecodeEvent>| {
        for e in events {
            let DecodeEvent::Frame(f) = e else { continue };
            frames += 1;
            changed |= f.tensor().width() != Some(640);
            if !changed {
                let hash = picture_hash(f.tensor());
                held.push_back((f, hash));
                if held.len() > HOLD {
                    held.pop_front();
                }
            }
        }
    };
    let stream = testdata(RESOLUTION_CHANGE_FILE);
    for (i, au) in access_units(&stream).enumerate() {
        take(decoder.decode(au, i as u64).expect("decode"));
    }
    take(decoder.flush().expect("flush"));
    assert_eq!(frames, RESOLUTION_CHANGE.len());
    assert_eq!(held.len(), HOLD);
    for (f, hash) in &held {
        assert_eq!(f.tensor().width(), Some(640));
        assert_eq!(
            picture_hash(f.tensor()),
            *hash,
            "picture {} was overwritten",
            f.pts()
        );
    }
}

#[test]
fn rejects_a_provided_pool_too_small_for_the_stream() {
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "rejects_a_provided_pool_too_small_for_the_stream",
    ) {
        return;
    }
    let pool: Vec<TensorDyn> = (0..8)
        .map(|_| TensorDyn::new(&[4096], DType::U8, Some(TensorMemory::DmaBuf), None).unwrap())
        .collect();
    let config = DecoderConfig::h264().with_output(OutputPool::Provided(pool));
    let Some(mut decoder) = open("rejects_a_provided_pool_too_small_for_the_stream", config) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let err = access_units(&stream)
        .enumerate()
        .find_map(|(i, au)| decoder.decode(au, i as u64).err())
        .expect("a 4 KiB pool cannot hold 1080p pictures");
    assert!(matches!(err, CodecError::InvalidConfig(_)), "{err}");
}

#[test]
fn a_small_pool_recycles_dropped_frames() {
    let Some(mut decoder) = open(
        "a_small_pool_recycles_dropped_frames",
        DecoderConfig::h264().with_output(OutputPool::Allocate { count: 1 }),
    ) else {
        return;
    };
    // Holding one frame while decoding the next is what a count of 1 allows.
    let mut held = None;
    let mut run = Run::default();
    let stream = testdata(MAIN_BFRAMES_FILE);
    for (i, au) in access_units(&stream).enumerate() {
        let frames = run.events(decoder.decode(au, i as u64).expect("decode"));
        if let Some(last) = frames.into_iter().last() {
            held = Some(last);
        }
    }
    drop(held);
    run.events(decoder.flush().unwrap());
    assert_eq!(run.hashes, MAIN_BFRAMES);
    assert!(
        run.infos[0].buffers < MAIN_BFRAMES.len(),
        "the pool ({}) must be smaller than the stream to show reuse",
        run.infos[0].buffers
    );
}

#[test]
fn holding_every_frame_exhausts_the_pool_until_frames_are_dropped() {
    let Some(mut decoder) = open(
        "holding_every_frame_exhausts_the_pool_until_frames_are_dropped",
        DecoderConfig::h264().with_output(OutputPool::Allocate { count: 2 }),
    ) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let mut run = Run::default();
    let mut held: Vec<DecodedFrame> = Vec::new();
    let mut exhausted = 0;
    for (i, au) in access_units(&stream).enumerate() {
        let events = match decoder.decode(au, i as u64) {
            Ok(events) => events,
            Err(CodecError::PoolExhausted { held: h, pool }) => {
                exhausted += 1;
                assert!(h > 0 && h <= pool, "{h} of {pool}");
                assert!(held.len() >= h, "the caller holds {} frames", held.len());
                // Dropping the frames frees the pool; the same access unit
                // then decodes.
                held.clear();
                decoder
                    .decode(au, i as u64)
                    .expect("decodes once frames are dropped")
            }
            Err(e) => panic!("decode: {e}"),
        };
        held.extend(run.events(events));
    }
    held.clear();
    run.events(decoder.flush().unwrap());
    assert!(
        exhausted > 0,
        "a pool of {} never ran out",
        run.infos[0].buffers
    );
    assert_eq!(run.hashes, MAIN_BFRAMES, "no picture lost or repeated");
}

#[test]
fn decodes_again_after_a_flush() {
    let Some(mut decoder) = open("decodes_again_after_a_flush", DecoderConfig::h264()) else {
        return;
    };
    let stream = testdata(MAIN_BFRAMES_FILE);
    let mut first = Run::default();
    let n = decode_all(&mut decoder, &stream, &mut first, 0);
    let mut second = Run::default();
    decode_all(&mut decoder, &stream, &mut second, 1000);
    assert_eq!(first.hashes, MAIN_BFRAMES);
    assert_eq!(
        second.hashes, MAIN_BFRAMES,
        "the stream decodes the same after a flush"
    );
    assert_each_pts_once(&second.pts, 1000, n);
}

#[test]
fn decodes_what_the_hardware_encoder_produces() {
    let test = "decodes_what_the_hardware_encoder_produces";
    if !dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), test) {
        return;
    }
    let (w, h) = (640, 480);
    let mut encoder = match VideoEncoder::new(EncoderConfig::h264(
        w as u32,
        h as u32,
        PixelFormat::Nv12,
        30.0,
    )) {
        Ok(e) => e,
        Err(CodecError::NoDevice(why)) => return skip(test, &why),
        Err(e) => panic!("encoder open failed: {e}"),
    };
    let Some(mut decoder) = open(test, DecoderConfig::h264()) else {
        return;
    };
    let src = TensorDyn::image(
        w,
        h,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut units = Vec::new();
    for n in 0..30u64 {
        {
            let stride = src.effective_row_stride().unwrap();
            let mut px = src.map_bytes(CpuAccess::Write).unwrap();
            for (i, b) in px.iter_mut().enumerate() {
                *b = ((i % stride) as u64 / 4 + (i / stride) as u64 / 4 + 3 * n) as u8;
            }
        }
        let opts = FrameOptions {
            pts: n,
            ..FrameOptions::default()
        };
        units.extend(encoder.encode(&src, &opts).unwrap());
    }
    units.extend(encoder.flush().unwrap());
    assert_eq!(units.len(), 30);

    let mut run = Run::default();
    for au in &units {
        run.events(decoder.decode(&au.data, au.pts).unwrap());
    }
    run.events(decoder.flush().unwrap());
    assert_eq!(run.hashes.len(), units.len(), "one picture per access unit");
    assert_eq!(
        run.pts,
        (0..30).collect::<Vec<_>>(),
        "no reordering without B-frames"
    );
    assert!(run.sizes.iter().all(|&s| s == (w, h)));
    assert_eq!(run.corrupted, 0);
}

#[test]
fn rejects_an_empty_access_unit() {
    let Some(mut decoder) = open("rejects_an_empty_access_unit", DecoderConfig::h264()) else {
        return;
    };
    assert!(matches!(
        decoder.decode(&[], 0),
        Err(CodecError::InvalidInput(_))
    ));
}

#[test]
fn rejects_an_empty_or_host_memory_pool() {
    let err =
        VideoDecoder::new(DecoderConfig::h264().with_output(OutputPool::Provided(Vec::new())))
            .unwrap_err();
    assert!(matches!(err, CodecError::InvalidConfig(_)), "{err}");

    let host = TensorDyn::new(&[4096], DType::U8, Some(TensorMemory::Mem), None).unwrap();
    let err =
        VideoDecoder::new(DecoderConfig::h264().with_output(OutputPool::Provided(vec![host])))
            .unwrap_err();
    assert!(matches!(err, CodecError::InvalidConfig(_)), "{err}");
}
