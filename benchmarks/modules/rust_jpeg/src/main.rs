// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Rust-ecosystem JPEG decode reference arms for the decoder A/B.
//!
//! Mirrors the `turbojpeg` bench protocol: sorted COCO subset, warmup on the
//! same set, per-image wall time, p50/p95/p99. Two engines:
//!
//! - `--engine zune`: `zune-jpeg` directly, headers + `decode_into` a reused
//!   buffer. `--format yuv` decodes to interleaved YCbCr (its closest
//!   native-output analogue); `--format rgb` to RGB.
//! - `--engine image`: the `image` crate exactly as the codec's historical
//!   comparison uses it (`load_from_memory_with_format` + `to_rgb8`,
//!   allocating per call; RGB only). Uses zune-jpeg internally.

use anyhow::{Context, Result};
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long, env = "EDGEFIRST_BENCH_COCO")]
    coco: std::path::PathBuf,
    #[arg(long, default_value_t = 200)]
    limit: usize,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value = "zune")]
    engine: String,
    #[arg(long, default_value = "yuv")]
    format: String,
    #[arg(long, default_value = "")]
    board: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut files: Vec<_> = std::fs::read_dir(&args.coco)
        .with_context(|| format!("reading {}", args.coco.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    files.sort();
    files.truncate(args.limit);
    anyhow::ensure!(!files.is_empty(), "no JPEGs in {}", args.coco.display());

    let data: Vec<Vec<u8>> = files
        .iter()
        .map(|p| std::fs::read(p).map_err(Into::into))
        .collect::<Result<_>>()?;

    let colorspace = match args.format.as_str() {
        "yuv" => zune_core::colorspace::ColorSpace::YCbCr,
        "rgb" => zune_core::colorspace::ColorSpace::RGB,
        other => anyhow::bail!("unknown --format {other}"),
    };
    anyhow::ensure!(
        args.engine == "zune" || args.format == "rgb",
        "--engine image supports --format rgb only"
    );

    let mut buf = vec![0u8; 64 << 20];
    let mut mp_total = 0f64;
    let mut decode_one = |jpeg: &[u8]| -> Result<(usize, usize)> {
        match args.engine.as_str() {
            "zune" => {
                let opts = zune_core::options::DecoderOptions::default()
                    .jpeg_set_out_colorspace(colorspace);
                let mut dec =
                    zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(jpeg), opts);
                dec.decode_headers().context("headers")?;
                let need = dec.output_buffer_size().context("size")?;
                if buf.len() < need {
                    buf.resize(need, 0);
                }
                dec.decode_into(&mut buf).context("decode")?;
                let (w, h) = dec.dimensions().context("dims")?;
                Ok((w, h))
            }
            "image" => {
                let img =
                    image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)?
                        .to_rgb8();
                Ok((img.width() as usize, img.height() as usize))
            }
            other => anyhow::bail!("unknown --engine {other}"),
        }
    };

    for jpeg in data.iter().take(args.warmup) {
        let _ = decode_one(jpeg);
    }

    // Per-image failures are skipped and counted, not fatal: e.g. zune-jpeg
    // rejects forced-YCbCr output for COCO's greyscale images.
    let mut skipped = 0usize;
    let mut times = Vec::with_capacity(data.len());
    for jpeg in &data {
        let t0 = Instant::now();
        match decode_one(jpeg) {
            Ok((w, h)) => {
                times.push(t0.elapsed().as_secs_f64() * 1e3);
                mp_total += (w * h) as f64 / 1e6;
            }
            Err(_) => skipped += 1,
        }
    }
    anyhow::ensure!(!times.is_empty(), "every image failed to decode");
    if skipped > 0 {
        eprintln!("  note: {skipped} images skipped (decode error)");
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| times[(((times.len() - 1) as f64) * p) as usize];
    let total_ms: f64 = times.iter().sum();
    println!(
        "  p50={:.3} ms  p95={:.3} ms  p99={:.3} ms  {:.1} MP/s  n={}",
        pct(0.50),
        pct(0.95),
        pct(0.99),
        mp_total / (total_ms / 1e3),
        times.len()
    );
    let _ = &args.board;
    Ok(())
}
