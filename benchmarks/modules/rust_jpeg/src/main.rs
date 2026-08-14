// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Rust-ecosystem JPEG decode reference arms for the decoder A/B.
//!
//! Mirrors the `turbojpeg` bench protocol: sorted, **evenly spaced** COCO
//! subset (the same `i * total / n` stride as `benchmarks/common` and
//! `bench.c`, so every arm decodes the same images for a given `--limit`),
//! warmup repeated on the first image, per-image wall time, p50 (with ~95%
//! CI) / mean / p95 / p99. Two engines:
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
    #[arg(long, default_value = "unknown")]
    board: String,
    /// Write summary CSV (same schema as benchmarks/common) to this path.
    #[arg(long)]
    csv: Option<std::path::PathBuf>,
}

/// Process CPU time, as the C harnesses measure it (CLOCK_PROCESS_CPUTIME_ID;
/// the decode loop is single-threaded so this doubles as busiest-core).
fn process_cpu_seconds() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) } != 0 {
        return 0.0;
    }
    ts.tv_sec as f64 + ts.tv_nsec as f64 / 1e9
}

/// Peak RSS in MB, matching peak_rss_mb() in benchmarks/common and the C
/// harnesses. Linux ru_maxrss is kilobytes; Darwin reports bytes.
fn peak_rss_mb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0.0;
    }
    if cfg!(target_os = "macos") {
        ru.ru_maxrss as f64 / (1024.0 * 1024.0)
    } else {
        ru.ru_maxrss as f64 / 1024.0
    }
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
    // Evenly spaced subset, exactly as benchmarks/common list_jpegs() and
    // bench.c preload() select: COCO val2017's lexicographic prefix is biased
    // (4:4:4-dominant), and every arm must see the same images.
    if args.limit > 0 && args.limit < files.len() {
        let len = files.len();
        files = (0..args.limit)
            .map(|i| files[i * len / args.limit].clone())
            .collect();
    }
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

    // High-water scratch, grown on demand by decode_one (the turbo/wuffs
    // pattern): starting empty keeps the reported peak RSS an honest measure
    // of the decoder's working set instead of a pre-allocated 64 MB floor.
    let mut buf = Vec::new();
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
                    image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)?.to_rgb8();
                Ok((img.width() as usize, img.height() as usize))
            }
            other => anyhow::bail!("unknown --engine {other}"),
        }
    };

    // Warmup on the first *decodable* image, repeated as the hal_cpu and
    // turbojpeg harnesses do; if the corpus leads with an image this engine
    // rejects (e.g. greyscale under forced YCbCr), fall through to the next
    // so warmup is never a silent no-op.
    if let Some(first) = data.iter().find(|jpeg| decode_one(jpeg).is_ok()) {
        for _ in 1..args.warmup {
            let _ = decode_one(first);
        }
    } else {
        eprintln!("  warn: no image in the corpus warms up under this engine/format");
    }

    // Per-image failures are skipped and counted, not fatal: e.g. zune-jpeg
    // rejects forced-YCbCr output for COCO's greyscale images.
    let mut skipped = 0usize;
    let mut times = Vec::with_capacity(data.len());
    let cpu0 = process_cpu_seconds();
    let wall0 = Instant::now();
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
    let n = times.len();
    // `round(p * (n - 1))`, the same index as benchmarks/common and bench.c.
    let pct = |p: f64| times[((p * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    let total_ms: f64 = times.iter().sum();
    // ~95% CI for the median: 1-based ranks n/2 ∓ 1.96·√n/2, as in
    // benchmarks/common median_ci_indices().
    let half_width = 0.98 * (n as f64).sqrt();
    let ci_lo = times[((n as f64) / 2.0 - half_width).floor().max(1.0) as usize - 1];
    let ci_hi = times[((n as f64) / 2.0 + 1.0 + half_width).ceil().min(n as f64) as usize - 1];
    let wall_s = wall0.elapsed().as_secs_f64();
    let cpu_s = process_cpu_seconds() - cpu0;
    let cpu_pct = if wall_s > 0.0 {
        100.0 * cpu_s / wall_s
    } else {
        0.0
    };
    let mpix_per_s = mp_total / (total_ms / 1e3);
    println!(
        "  p50={:.3} ms  ci95=[{:.3},{:.3}]  mean={:.3}  p95={:.3} ms  p99={:.3} ms  {:.1} MP/s  n={}",
        pct(0.50),
        ci_lo,
        ci_hi,
        total_ms / n as f64,
        pct(0.95),
        pct(0.99),
        mpix_per_s,
        n
    );
    if let Some(csv) = &args.csv {
        // Same schema as benchmarks/common write_summary_csv / bench.c.
        if let Some(parent) = csv.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let notes = format!(
            "backend={engine}-{format};scope=decode-only;harness=rust;skipped={skipped}",
            engine = args.engine,
            format = args.format
        );
        std::fs::write(
            csv,
            format!(
                "board,class,module,ms_p50,ms_p95,ms_p99,ms_mean,ms_p50_ci_lo,ms_p50_ci_hi,\
                 mpix_per_s,peak_rss_mb,cpu_pct_process,cpu_pct_system,cpu_pct_peak_core,\
                 n_images,notes\n\
                 {board},decode,{module},{p50:.3},{p95:.3},{p99:.3},{mean:.3},{ci_lo:.3},\
                 {ci_hi:.3},{mpix:.3},{rss:.1},{cpu:.1},0.0,{cpu:.1},{n},{notes}\n",
                board = args.board,
                module = args.engine,
                p50 = pct(0.50),
                p95 = pct(0.95),
                p99 = pct(0.99),
                mean = total_ms / n as f64,
                ci_lo = ci_lo,
                ci_hi = ci_hi,
                mpix = mpix_per_s,
                rss = peak_rss_mb(),
                cpu = cpu_pct,
                n = n,
                notes = notes
            ),
        )?;
    }
    Ok(())
}
