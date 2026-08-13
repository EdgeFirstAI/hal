// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Fast-vs-accurate DCT pixel similarity over a JPEG corpus.
//!
//! Decodes every image twice — `DctMethod::Accurate` (the default `islow`
//! kernel) and `DctMethod::Fast` (opt-in AAN) — and reports per-corpus
//! cosine similarity, PSNR, and worst pixel delta of the raw decoded planes.
//! This is the interim accuracy evidence for the fast mode until the mAP
//! comparison lands (measure the model, not just the pixels).
//!
//! ```bash
//! dct_compare --dir /data/coco/val2017 --limit 500
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use edgefirst_codec::{DctMethod, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};

#[derive(Parser)]
struct Args {
    /// JPEG corpus directory (searched non-recursively for *.jpg/*.jpeg).
    #[arg(long, env = "EDGEFIRST_BENCH_COCO")]
    dir: std::path::PathBuf,
    /// Maximum number of images (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Print each image's stats, not just the aggregate.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut files: Vec<_> = std::fs::read_dir(&args.dir)
        .with_context(|| format!("reading {}", args.dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    files.sort();
    if args.limit > 0 {
        files.truncate(args.limit);
    }
    anyhow::ensure!(!files.is_empty(), "no JPEGs in {}", args.dir.display());

    let mut t_acc = Tensor::<u8>::image(
        8192,
        8192,
        PixelFormat::Nv24,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )?;
    let mut t_fast = Tensor::<u8>::image(
        8192,
        8192,
        PixelFormat::Nv24,
        Some(TensorMemory::Mem),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )?;
    let mut dec_acc = ImageDecoder::new();
    dec_acc.set_dct_method(DctMethod::Accurate);
    let mut dec_fast = ImageDecoder::new();
    dec_fast.set_dct_method(DctMethod::Fast);

    let mut n_images = 0usize;
    let mut worst_cosine = 1.0f64;
    let mut worst_psnr = f64::INFINITY;
    let mut global_max_diff = 0i32;
    let mut sum_cosine = 0.0f64;
    let mut sum_psnr = 0.0f64;
    let mut exceeds_8 = 0usize;

    for path in &files {
        let data = std::fs::read(path)?;
        let ia = match t_acc.load_image(&mut dec_acc, &data) {
            Ok(i) => i,
            Err(_) => continue, // corrupt / unsupported: skip both arms
        };
        let _ = t_fast.load_image(&mut dec_fast, &data)?;

        // Compare the decoded planes over the valid image region only: rows
        // of `width` bytes at the tensor stride, over the combined plane
        // height (Y + interleaved UV for Nv24; Y-only for Grey).
        let stride = t_acc.effective_row_stride().unwrap_or(ia.width);
        let plane_rows = match ia.format {
            PixelFormat::Grey => ia.height,
            _ => ia.height * 3, // Nv24: Y rows + UV rows (2 bytes/px over w/2… full-res UV)
        };
        let a = t_acc.map()?;
        let f = t_fast.map()?;
        let (mut dot, mut na, mut nf, mut se) = (0f64, 0f64, 0f64, 0f64);
        let mut max_diff = 0i32;
        let mut count = 0usize;
        for r in 0..plane_rows {
            let off = r * stride;
            let w = ia.width.min(stride);
            for i in off..off + w {
                let (x, y) = (a[i] as f64, f[i] as f64);
                dot += x * y;
                na += x * x;
                nf += y * y;
                let d = (a[i] as i32 - f[i] as i32).abs();
                se += (d * d) as f64;
                max_diff = max_diff.max(d);
                count += 1;
            }
        }
        drop(a);
        drop(f);
        let cosine = dot / (na.sqrt() * nf.sqrt()).max(1e-12);
        let mse = se / count as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };
        n_images += 1;
        sum_cosine += cosine;
        sum_psnr += psnr.min(99.0);
        worst_cosine = worst_cosine.min(cosine);
        worst_psnr = worst_psnr.min(psnr);
        global_max_diff = global_max_diff.max(max_diff);
        if max_diff > 8 {
            exceeds_8 += 1;
        }
        if args.verbose {
            println!(
                "{}: cosine={cosine:.7} psnr={psnr:.2} dB max|d|={max_diff}",
                path.file_name().unwrap().to_string_lossy()
            );
        }
    }

    println!("== fast-vs-accurate DCT similarity over {n_images} images");
    println!(
        "  cosine:  mean={:.7}  worst={:.7}",
        sum_cosine / n_images as f64,
        worst_cosine
    );
    println!(
        "  psnr:    mean={:.2} dB  worst={:.2} dB",
        sum_psnr / n_images as f64,
        worst_psnr
    );
    println!(
        "  max|d|:  {global_max_diff}  (images with max|d| > 8: {exceeds_8}/{n_images})"
    );
    Ok(())
}
