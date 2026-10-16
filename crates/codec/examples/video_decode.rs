// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decodes an H.264 Annex B file with the hardware decoder.
//!
//! Usage: video-decode <in.h264> [options]
//!
//!   --out FILE        write the visible NV12 pictures to FILE, one after
//!                     another (a stream that changes size writes each
//!                     picture at its own size)
//!   --hold N          keep the last N frames alive, as a consumer that
//!                     works on several pictures at once would (default 0)
//!   --pool N          frames the caller may hold (default 4)
//!
//! `EDGEFIRST_CODEC_V4L2_DECODER=/dev/videoN` picks the decoder node.

use std::collections::VecDeque;
use std::io::Write;
use std::time::{Duration, Instant};

use edgefirst_codec::video::{
    access_units, DecodeEvent, DecodedFrame, DecoderConfig, OutputPool, VideoDecoder,
};
use edgefirst_tensor::{CpuAccess, TensorDyn};

struct Args {
    input: String,
    out: Option<String>,
    hold: usize,
    pool: usize,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        input: String::new(),
        out: None,
        hold: 0,
        pool: 4,
    };
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--out" => args.out = Some(value()?),
            "--hold" => args.hold = value()?.parse().map_err(|e| format!("--hold: {e}"))?,
            "--pool" => args.pool = value()?.parse().map_err(|e| format!("--pool: {e}"))?,
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            s if args.input.is_empty() => args.input = s.to_owned(),
            s => return Err(format!("unexpected argument {s}")),
        }
    }
    if args.input.is_empty() {
        return Err("missing input file".into());
    }
    Ok(args)
}

/// Writes the visible picture of an NV12 tensor, row by row.
fn write_nv12(out: &mut impl Write, t: &TensorDyn) -> std::io::Result<()> {
    let (w, h) = (t.width().unwrap_or(0), t.height().unwrap_or(0));
    let luma = t
        .map_bytes(CpuAccess::Read)
        .map_err(std::io::Error::other)?;
    let stride = t.effective_row_stride().unwrap_or(w);
    for y in 0..h {
        out.write_all(&luma[y * stride..y * stride + w])?;
    }
    let chroma = t
        .chroma_dyn()
        .ok_or_else(|| std::io::Error::other("no chroma plane"))?;
    let cmap = chroma
        .map_bytes(CpuAccess::Read)
        .map_err(std::io::Error::other)?;
    let cstride = chroma.effective_row_stride().unwrap_or(w);
    for y in 0..h.div_ceil(2) {
        out.write_all(&cmap[y * cstride..y * cstride + w])?;
    }
    Ok(())
}

struct Sink {
    out: Option<std::io::BufWriter<std::fs::File>>,
    held: VecDeque<DecodedFrame>,
    hold: usize,
    frames: usize,
    corrupted: usize,
    last_pts: Option<u64>,
    reordered: usize,
}

impl Sink {
    fn events(&mut self, events: Vec<DecodeEvent>) -> std::io::Result<()> {
        for e in events {
            match e {
                DecodeEvent::Frame(f) => self.put(f)?,
                DecodeEvent::Reconfigured(info) => eprintln!(
                    "stream {}x{} (coded {}x{}, crop {:?}, {} {:?} buffers, {:?})",
                    info.width,
                    info.height,
                    info.coded_width,
                    info.coded_height,
                    info.crop,
                    info.buffers,
                    info.memory,
                    info.colorimetry
                ),
            }
        }
        Ok(())
    }

    fn put(&mut self, f: DecodedFrame) -> std::io::Result<()> {
        self.frames += 1;
        self.corrupted += usize::from(f.corrupted());
        if self.last_pts.is_some_and(|p| f.pts() <= p) {
            self.reordered += 1;
        }
        self.last_pts = Some(f.pts());
        if let Some(out) = self.out.as_mut() {
            write_nv12(out, f.tensor())?;
        }
        self.held.push_back(f);
        while self.held.len() > self.hold {
            self.held.pop_front();
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse().map_err(|e| {
        format!("{e}\nusage: video-decode <in.h264> [--out FILE] [--hold N] [--pool N]")
    })?;
    env_logger::init();
    let stream = std::fs::read(&args.input)?;
    let config = DecoderConfig::h264().with_output(OutputPool::Allocate { count: args.pool });
    let mut decoder = VideoDecoder::new(config)?;
    eprintln!("decoder {}", decoder.device());

    let mut sink = Sink {
        out: args
            .out
            .as_deref()
            .map(std::fs::File::create)
            .transpose()?
            .map(std::io::BufWriter::new),
        held: VecDeque::new(),
        hold: args.hold,
        frames: 0,
        corrupted: 0,
        last_pts: None,
        reordered: 0,
    };
    let mut units = 0usize;
    let mut busy = Duration::ZERO;
    let started = Instant::now();
    for au in access_units(&stream) {
        let t = Instant::now();
        let events = decoder.decode(au, units as u64)?;
        busy += t.elapsed();
        units += 1;
        sink.events(events)?;
    }
    let t = Instant::now();
    let rest = decoder.flush()?;
    busy += t.elapsed();
    sink.events(rest)?;
    if let Some(out) = sink.out.as_mut() {
        out.flush()?;
    }
    let per_unit = busy.checked_div(units.max(1) as u32).unwrap_or_default();
    eprintln!(
        "{units} access units -> {} frames ({} corrupted, {} reordered), {per_unit:?} per access unit, {:?} total",
        sink.frames,
        sink.corrupted,
        sink.reordered,
        started.elapsed()
    );
    if sink.frames != units {
        return Err(format!("decoded {} frames from {units} access units", sink.frames).into());
    }
    Ok(())
}
