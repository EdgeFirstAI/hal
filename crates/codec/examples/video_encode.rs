// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Encodes NV12 frames to an H.264 Annex B file with the hardware encoder.
//!
//! Usage: video-encode <out.h264> [options]
//!
//!   --size WxH           picture size (default 1920x1080)
//!   --frames N           frames to encode (default 300)
//!   --fps F              frame rate (default 30)
//!   --bitrate BPS        target bitrate (default: the device's)
//!   --gop N              frames between key frames (default: the device's)
//!   --keyframe-every N   also force a key frame every N frames
//!   --input FILE         read raw NV12 frames from FILE instead of drawing
//!                        a moving test pattern
//!
//! `EDGEFIRST_CODEC_V4L2_ENCODER=/dev/videoN` picks the encoder node.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use edgefirst_codec::video::{Bitrate, EncodedFrame, EncoderConfig, FrameOptions, VideoEncoder};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

struct Args {
    out: String,
    width: usize,
    height: usize,
    frames: usize,
    fps: f64,
    bitrate: Option<u32>,
    gop: Option<u32>,
    keyframe_every: Option<usize>,
    input: Option<String>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        out: String::new(),
        width: 1920,
        height: 1080,
        frames: 300,
        fps: 30.0,
        bitrate: None,
        gop: None,
        keyframe_every: None,
        input: None,
    };
    let value = |it: &mut dyn Iterator<Item = String>, flag: &str| {
        it.next().ok_or_else(|| format!("{flag} needs a value"))
    };
    while let Some(a) = it.next() {
        let num = |s: String| s.parse::<u64>().map_err(|e| format!("{a}: {e}"));
        match a.as_str() {
            "--size" => {
                let v = value(&mut it, &a)?;
                let (w, h) = v.split_once('x').ok_or(format!("--size {v}: want WxH"))?;
                args.width = w.parse().map_err(|e| format!("--size: {e}"))?;
                args.height = h.parse().map_err(|e| format!("--size: {e}"))?;
            }
            "--frames" => args.frames = num(value(&mut it, &a)?)? as usize,
            "--fps" => {
                args.fps = value(&mut it, &a)?
                    .parse()
                    .map_err(|e| format!("--fps: {e}"))?
            }
            "--bitrate" => args.bitrate = Some(num(value(&mut it, &a)?)? as u32),
            "--gop" => args.gop = Some(num(value(&mut it, &a)?)? as u32),
            "--keyframe-every" => args.keyframe_every = Some(num(value(&mut it, &a)?)? as usize),
            "--input" => args.input = Some(value(&mut it, &a)?),
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            s if args.out.is_empty() => args.out = s.to_owned(),
            s => return Err(format!("unexpected argument {s}")),
        }
    }
    if args.out.is_empty() {
        return Err("missing output file".into());
    }
    Ok(args)
}

/// The output file and what has been written to it.
struct Sink {
    out: std::io::BufWriter<std::fs::File>,
    encoded: usize,
    keyframes: usize,
    bytes: usize,
}

impl Sink {
    fn put(&mut self, f: &EncodedFrame) -> std::io::Result<()> {
        self.encoded += 1;
        self.keyframes += usize::from(f.keyframe);
        self.bytes += f.data.len();
        self.out.write_all(&f.data)
    }
}

/// Writes frame `n` of a moving gradient into `img`.
fn draw(img: &TensorDyn, n: usize) {
    let (w, h) = (img.width().unwrap(), img.height().unwrap());
    let stride = img.effective_row_stride().unwrap();
    let mut px = img.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        for x in 0..w {
            px[y * stride + x] = ((x / 4 + y / 4 + 2 * n) & 0xff) as u8;
        }
    }
    for y in 0..h / 2 {
        let row = stride * h + y * stride;
        for x in (0..w).step_by(2) {
            px[row + x] = (64 + (x / 16 + n) % 128) as u8;
            px[row + x + 1] = (64 + (y / 8) % 128) as u8;
        }
    }
}

/// Copies one dense NV12 frame from `input` into `img`. `false` at the end
/// of the input.
fn load(img: &TensorDyn, input: &mut impl Read, frame: &mut [u8]) -> std::io::Result<bool> {
    match input.read_exact(frame) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(e) => return Err(e),
    }
    let (w, h) = (img.width().unwrap(), img.height().unwrap());
    let stride = img.effective_row_stride().unwrap();
    let mut px = img.map_bytes(CpuAccess::Write).unwrap();
    for (y, row) in frame.chunks_exact(w).enumerate().take(h + h / 2) {
        px[y * stride..y * stride + w].copy_from_slice(row);
    }
    Ok(true)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse().map_err(|e| format!("{e}\nusage: video-encode <out.h264> [--size WxH] [--frames N] [--fps F] [--bitrate BPS] [--gop N] [--keyframe-every N] [--input FILE]"))?;
    let mut config = EncoderConfig::h264(
        args.width as u32,
        args.height as u32,
        PixelFormat::Nv12,
        args.fps,
    );
    if let Some(bps) = args.bitrate {
        config.bitrate = Bitrate::Bps(bps);
    }
    config.gop = args.gop;
    let mut encoder = VideoEncoder::new(config)?;
    eprintln!("encoder {}", encoder.device());

    let img = TensorDyn::image(
        args.width,
        args.height,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )?;
    let mut input = args.input.as_deref().map(std::fs::File::open).transpose()?;
    let mut frame = vec![0u8; args.width * args.height * 3 / 2];
    let mut sink = Sink {
        out: std::io::BufWriter::new(std::fs::File::create(&args.out)?),
        encoded: 0,
        keyframes: 0,
        bytes: 0,
    };

    let frame_ns = (1e9 / args.fps) as u64;
    let mut busy = Duration::ZERO;
    for n in 0..args.frames {
        match input.as_mut() {
            Some(file) => {
                if !load(&img, file, &mut frame)? {
                    break;
                }
            }
            None => draw(&img, n),
        }
        let opts = FrameOptions {
            pts: n as u64 * frame_ns,
            keyframe: args
                .keyframe_every
                .is_some_and(|k| k > 0 && n > 0 && n % k == 0),
            ..FrameOptions::default()
        };
        let t = Instant::now();
        let au = encoder.encode(&img, &opts)?;
        busy += t.elapsed();
        if let Some(f) = au {
            sink.put(&f)?;
        }
    }
    for f in encoder.flush()? {
        sink.put(&f)?;
    }
    sink.out.flush()?;
    let per_frame = busy
        .checked_div(sink.encoded.max(1) as u32)
        .unwrap_or_default();
    eprintln!(
        "{} frames ({} key frames), {} bytes, {per_frame:?} per frame in encode() -> {}",
        sink.encoded, sink.keyframes, sink.bytes, args.out
    );
    Ok(())
}
