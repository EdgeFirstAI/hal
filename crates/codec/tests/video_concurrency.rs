// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hardware video sessions used at the same time: two encoders on two
//! threads, an encoder and a decoder on two threads, and two processes that
//! encode at once. Each test skips (prints `SKIPPED` and passes) where there
//! is no encoder or decoder or no GPU buffer allocator; under
//! `HAL_TEST_REQUIRE_DMA=1` a missing allocator fails instead.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use edgefirst_codec::video::{
    access_units, DecodeEvent, DecoderConfig, EncodedFrame, EncoderConfig, FrameOptions,
    VideoDecoder, VideoEncoder,
};
use edgefirst_codec::{capabilities, Codec, CodecError, Direction, Engine};
use edgefirst_tensor::Region;

#[path = "support/encode_retry.rs"]
mod encode_retry;
#[path = "common/pictures.rs"]
mod pictures;
#[path = "common/sessions.rs"]
mod sessions;

use pictures::{nv12, nv12_region, paint, psnr};

/// The DMA require gate: a skip here fails under `HAL_TEST_REQUIRE_DMA=1`.
#[path = "../../tensor/tests/support/dma_require.rs"]
mod dma_require;

/// Names the process a parent test started, in the child's environment.
const CHILD: &str = "EDGEFIRST_CODEC_TEST_CHILD";
/// The directory a parent test and its children meet in.
const MEETING: &str = "EDGEFIRST_CODEC_TEST_MEETING";
/// How long sessions that run together wait for each other.
const MEET_TIMEOUT: Duration = Duration::from_secs(60);

const W: usize = 640;
const H: usize = 480;
const FRAMES: usize = 10;

fn skip(test: &str, why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {test} - {why}");
}

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata")
        });
    let path = root.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Whether this machine has a hardware H.264 codec for `direction`.
fn has_h264(direction: Direction) -> bool {
    capabilities()
        .iter()
        .any(|c| c.codec == Codec::H264 && c.direction == direction && c.hardware)
}

/// Opens an H.264 encoder for `W`x`H` NV12, or `None` when there is none.
fn open_encoder() -> Option<VideoEncoder> {
    match VideoEncoder::new(EncoderConfig::h264(
        W as u32,
        H as u32,
        edgefirst_tensor::PixelFormat::Nv12,
        30.0,
    )) {
        Ok(e) => Some(e),
        Err(CodecError::NoDevice(_)) => None,
        Err(e) => panic!("encoder open failed: {e}"),
    }
}

/// Encodes `FRAMES` frames of the gradient at `frame`, and returns the
/// stream and the dense NV12 bytes of the source. On VideoToolbox, a frame
/// the encoder refuses with [`CodecError::Timeout`] is submitted again.
fn encode(encoder: &mut VideoEncoder, frame: usize) -> (Vec<EncodedFrame>, Vec<u8>) {
    let src = nv12(W, H);
    paint(&src, frame);
    let want = nv12_region(&src, Region::new(0, 0, W, H));
    let mut out = Vec::new();
    for i in 0..FRAMES {
        let opts = FrameOptions {
            pts: i as u64,
            keyframe: i == 0,
            ..FrameOptions::default()
        };
        let vt = encoder.info().engine == Engine::VideoToolbox;
        let au = encode_retry::retry(
            || encoder.encode(&src, &opts),
            |e| vt && matches!(e, CodecError::Timeout(_)),
        );
        out.extend(au.expect("encode"));
    }
    out.extend(encoder.flush().expect("flush"));
    (out, want)
}

/// The dense NV12 bytes of the first picture of `stream`.
fn first_picture(stream: &[EncodedFrame]) -> Vec<u8> {
    let mut decoder = VideoDecoder::new(DecoderConfig::h264()).expect("decoder open");
    let mut first = None;
    let mut keep = |events: Vec<DecodeEvent>| {
        for e in events {
            if let DecodeEvent::Frame(f) = e {
                if first.is_none() {
                    let t = f.tensor();
                    let whole = Region::new(0, 0, t.width().unwrap(), t.height().unwrap());
                    first = Some(nv12_region(t, whole));
                }
            }
        }
    };
    for f in stream {
        keep(decoder.decode(&f.data, f.pts).expect("decode"));
    }
    keep(decoder.flush().expect("flush"));
    first.expect("the stream has a picture")
}

#[test]
fn two_encoders_on_two_threads_each_decode_to_their_own_source() {
    const TEST: &str = "two_encoders_on_two_threads_each_decode_to_their_own_source";
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), TEST) {
        return;
    }
    if !has_h264(Direction::Encode) || !has_h264(Direction::Decode) {
        return skip(TEST, "no hardware H.264 encoder and decoder");
    }
    let _sessions = sessions::take(2);
    let both_open = Arc::new(Barrier::new(2));
    let workers: Vec<_> = [0usize, 60]
        .into_iter()
        .map(|frame| {
            let both_open = Arc::clone(&both_open);
            std::thread::spawn(move || {
                let encoder = open_encoder();
                both_open.wait();
                let mut encoder = encoder.expect("an encoder listed in the capabilities");
                encode(&mut encoder, frame)
            })
        })
        .collect();
    let results: Vec<_> = workers
        .into_iter()
        .map(|w| w.join().expect("encoder thread"))
        .collect();
    let pictures: Vec<Vec<u8>> = results
        .iter()
        .map(|(stream, _)| {
            assert_eq!(stream.len(), FRAMES, "every frame comes back once");
            first_picture(stream)
        })
        .collect();
    for (i, (_, own)) in results.iter().enumerate() {
        let other = &results[1 - i].1;
        let mine = psnr(own, &pictures[i]);
        let theirs = psnr(other, &pictures[i]);
        assert!(mine > 30.0, "stream {i} is {mine:.1} dB from its source");
        assert!(
            theirs + 10.0 < mine,
            "stream {i} is {theirs:.1} dB from the other source and {mine:.1} dB from its own"
        );
    }
}

#[test]
fn an_encoder_and_a_decoder_run_on_two_threads_at_once() {
    const TEST: &str = "an_encoder_and_a_decoder_run_on_two_threads_at_once";
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), TEST) {
        return;
    }
    if !has_h264(Direction::Encode) || !has_h264(Direction::Decode) {
        return skip(TEST, "no hardware H.264 encoder and decoder");
    }
    let stream = testdata("h264_1080p_main_bframes.h264");
    let _sessions = sessions::take(2);
    let both_open = Arc::new(Barrier::new(2));

    let encoding = {
        let both_open = Arc::clone(&both_open);
        std::thread::spawn(move || {
            let encoder = open_encoder();
            both_open.wait();
            let mut encoder = encoder.expect("an encoder listed in the capabilities");
            encode(&mut encoder, 7).0.len()
        })
    };
    let decoding = std::thread::spawn(move || {
        let decoder = VideoDecoder::new(DecoderConfig::h264());
        both_open.wait();
        let mut decoder = decoder.expect("a decoder listed in the capabilities");
        let mut pts = Vec::new();
        let mut corrupted = 0;
        let mut take = |events: Vec<DecodeEvent>| {
            for e in events {
                if let DecodeEvent::Frame(f) = e {
                    pts.push(f.pts());
                    corrupted += usize::from(f.corrupted());
                }
            }
        };
        let mut units = 0;
        for (i, au) in access_units(&stream).enumerate() {
            take(decoder.decode(au, i as u64).expect("decode"));
            units += 1;
        }
        take(decoder.flush().expect("flush"));
        pts.sort_unstable();
        (units, pts, corrupted)
    });

    assert_eq!(
        encoding.join().expect("encoder thread"),
        FRAMES,
        "every encoded frame comes back once"
    );
    let (units, pts, corrupted) = decoding.join().expect("decoder thread");
    assert_eq!(
        pts,
        (0..units as u64).collect::<Vec<_>>(),
        "one picture per access unit"
    );
    assert_eq!(corrupted, 0);
}

/// Waits until `path` exists.
fn wait_for(path: &Path) {
    let deadline = Instant::now() + MEET_TIMEOUT;
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "{} did not appear within {MEET_TIMEOUT:?}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The child's part of the two-process test: opens an encoder, meets the
/// other child, and encodes.
fn encode_as_child(id: &str, meeting: &Path) {
    let mut encoder = open_encoder().expect("an encoder listed in the capabilities");
    std::fs::write(meeting.join(format!("open-{id}")), b"").expect("writing the open marker");
    let peer = if id == "0" { "1" } else { "0" };
    wait_for(&meeting.join(format!("open-{peer}")));
    let (stream, _) = encode(&mut encoder, id.parse::<usize>().unwrap() * 60);
    assert_eq!(stream.len(), FRAMES, "every frame comes back once");
    assert!(stream[0].keyframe);
}

#[test]
fn two_processes_encode_at_the_same_time() {
    const TEST: &str = "two_processes_encode_at_the_same_time";
    if let (Ok(id), Ok(meeting)) = (std::env::var(CHILD), std::env::var(MEETING)) {
        return encode_as_child(&id, Path::new(&meeting));
    }
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), TEST) {
        return;
    }
    if !has_h264(Direction::Encode) {
        return skip(TEST, "no hardware H.264 encoder");
    }
    let meeting = std::env::temp_dir().join(format!(
        "edgefirst-codec-meeting-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&meeting);
    std::fs::create_dir_all(&meeting).unwrap();
    // The two children's encoders.
    let _sessions = sessions::take(2);
    let exe = std::env::current_exe().unwrap();
    let children: Vec<_> = ["0", "1"]
        .into_iter()
        .map(|id| {
            std::process::Command::new(&exe)
                .args(["--exact", TEST, "--nocapture"])
                .env(CHILD, id)
                .env(MEETING, &meeting)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .expect("starting a child test process")
        })
        .collect();
    for (id, child) in children.into_iter().enumerate() {
        let out = child.wait_with_output().expect("waiting for a child");
        assert!(
            out.status.success(),
            "child {id} failed ({}):\n{}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("1 passed"),
            "child {id} ran no test:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
    std::fs::remove_dir_all(&meeting).ok();
}
