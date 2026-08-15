// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Pixel-parity check for EdgeFirst's fused native-4:2:0 RGB decode
//! (`write_rgb_rows_420`/`expand_row_2x`, 2x2 nearest-neighbour "box" chroma
//! upsample) against libjpeg-turbo's matched-accuracy-class RGB path
//! (`TJFLAG_ACCURATEDCT | TJFLAG_FASTUPSAMPLE` — same IDCT, same box/nearest
//! chroma upsample, no fancy/triangle filtering).
//!
//! This is `docs/BENCHMARKS_JPEG_REVIEW_v3.13.md` item #1 (blocking): the two
//! implementations can both be correctly described as "box" and still differ
//! by a half-pixel if their replication phase disagrees — which source chroma
//! sample each of the 2x2 output pixels inherits. A phase bug would corrupt
//! every 4:2:0 RGB decode silently, since throughput benchmarks never look at
//! pixel values.
//!
//! Reports max abs delta / PSNR / cosine similarity the same way
//! `dct_compare` does, plus a phase-shift diagnostic: mean abs diff between
//! the two decodes at 8 neighbouring 1px (dx, dy) offsets. If any shifted
//! alignment scores lower mean abs diff than the unshifted (0, 0) comparison,
//! that is direct evidence of a systematic replication-phase disagreement.
//!
//! ```bash
//! rgb_parity --dir /data/coco/val2017-yuv420 --limit 200
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory, TensorTrait};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uchar, c_ulong, c_void, CStr};

const TJPF_RGB: c_int = 0;
const TJFLAG_FASTUPSAMPLE: c_int = 256;
const TJFLAG_ACCURATEDCT: c_int = 4096;

#[derive(Parser)]
struct Args {
    /// JPEG corpus directory (native 4:2:0 sources — non-matching images are
    /// skipped and counted, since the fused path only engages for them).
    #[arg(long, env = "EDGEFIRST_BENCH_COCO")]
    dir: std::path::PathBuf,
    /// Maximum number of images (0 = all).
    #[arg(long, default_value_t = 200)]
    limit: usize,
    /// Print each image's stats, not just the aggregate.
    #[arg(long)]
    verbose: bool,
    /// Override the libturbojpeg search (same env var turbojpeg_bench uses).
    #[arg(long, env = "EDGEFIRST_TURBOJPEG_LIB")]
    lib: Option<String>,
}

type TjInit = unsafe extern "C" fn() -> *mut c_void;
type TjDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type TjHeader = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type TjDecompress = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type TjErrorStr = unsafe extern "C" fn() -> *const c_char;

/// Same candidate list as `benchmarks/modules/turbojpeg/bench.c`'s `tj_load`.
const CANDIDATES: &[&str] = &[
    #[cfg(target_os = "macos")]
    "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",
    #[cfg(target_os = "macos")]
    "/opt/homebrew/lib/libturbojpeg.dylib",
    #[cfg(target_os = "macos")]
    "/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib",
    #[cfg(target_os = "macos")]
    "libturbojpeg.dylib",
    #[cfg(target_os = "macos")]
    "libturbojpeg.0.dylib",
    "libturbojpeg.so.0",
    "libturbojpeg.so",
    "libturbojpeg.so.0.2.0",
    "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
    "/usr/lib/aarch64-linux-gnu/libturbojpeg.so",
    "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
    "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",
    "/opt/libjpeg-turbo/lib64/libturbojpeg.so",
];

struct TurboJpeg {
    // Leaked for a 'static lifetime, matching the codec crate's own nvJPEG
    // dlopen loader (jpeg/nvjpeg/loader.rs) — this is a short-lived CLI tool,
    // so leaking once at startup is simpler than threading a lifetime through
    // the resolved function-pointer table.
    _lib: &'static Library,
    handle: *mut c_void,
    destroy: TjDestroy,
    header: TjHeader,
    decompress: TjDecompress,
    error: TjErrorStr,
    path: String,
}

impl TurboJpeg {
    fn load(override_path: Option<&str>) -> Result<Self> {
        let (lib, path) = if let Some(p) = override_path.filter(|p| !p.is_empty()) {
            let lib = unsafe { Library::new(p) }
                .with_context(|| format!("EDGEFIRST_TURBOJPEG_LIB={p}: dlopen failed"))?;
            (lib, p.to_string())
        } else {
            let mut found = None;
            for cand in CANDIDATES {
                if let Ok(lib) = unsafe { Library::new(*cand) } {
                    found = Some((lib, cand.to_string()));
                    break;
                }
            }
            found.context("libturbojpeg not found")?
        };
        let lib: &'static Library = Box::leak(Box::new(lib));

        // SAFETY: symbol types match the documented turbojpeg.h ABI (same
        // signatures benchmarks/modules/turbojpeg/bench.c dlsyms).
        let init: TjInit = *unsafe { lib.get(b"tjInitDecompress\0") }?;
        let destroy: TjDestroy = *unsafe { lib.get(b"tjDestroy\0") }?;
        let header: TjHeader = *unsafe { lib.get(b"tjDecompressHeader3\0") }?;
        let decompress: TjDecompress = *unsafe { lib.get(b"tjDecompress2\0") }?;
        let error: TjErrorStr = *unsafe { lib.get(b"tjGetErrorStr\0") }?;
        let handle = unsafe { init() };
        if handle.is_null() {
            bail!("tjInitDecompress failed");
        }
        Ok(Self {
            _lib: lib,
            handle,
            destroy,
            header,
            decompress,
            error,
            path,
        })
    }

    fn err(&self) -> String {
        unsafe {
            let p = (self.error)();
            if p.is_null() {
                "(no error string)".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    /// Decode to interleaved RGB with accurate IDCT + box/nearest chroma
    /// upsample (`TJFLAG_FASTUPSAMPLE`) — the matched-accuracy-class
    /// comparison point for EdgeFirst's fused box-upsample RGB path.
    fn decode_rgb_box(&self, jpeg: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
        let mut w = 0i32;
        let mut h = 0i32;
        let mut subsamp = 0i32;
        let mut colorspace = 0i32;
        unsafe {
            if (self.header)(
                self.handle,
                jpeg.as_ptr(),
                jpeg.len() as c_ulong,
                &mut w,
                &mut h,
                &mut subsamp,
                &mut colorspace,
            ) != 0
            {
                bail!("tjDecompressHeader3: {}", self.err());
            }
            let mut buf = vec![0u8; (w as usize) * (h as usize) * 3];
            if (self.decompress)(
                self.handle,
                jpeg.as_ptr(),
                jpeg.len() as c_ulong,
                buf.as_mut_ptr(),
                w,
                0,
                h,
                TJPF_RGB,
                TJFLAG_ACCURATEDCT | TJFLAG_FASTUPSAMPLE,
            ) != 0
            {
                bail!("tjDecompress2: {}", self.err());
            }
            Ok((w as usize, h as usize, buf))
        }
    }
}

impl Drop for TurboJpeg {
    fn drop(&mut self) {
        // Best-effort; process exits shortly after in every use of this tool.
        unsafe { (self.destroy)(self.handle) };
    }
}

/// mean|d|, max|d| (RGB channel bytes) between `a` and `b` when `b` is
/// shifted by `(dx, dy)` pixels relative to `a`, over the interior region
/// common to both after the shift (a 1px border is more than enough margin
/// for a 1px probe). `stride`/`w3` are byte strides (`width * 3`).
fn shifted_diff(a: &[u8], b: &[u8], w: usize, h: usize, dx: i32, dy: i32) -> f64 {
    let w3 = w * 3;
    let mut sum = 0f64;
    let mut count = 0usize;
    let y0 = dy.max(0) as usize;
    let y1 = (h as i32 + dy.min(0)) as usize;
    let x0 = dx.max(0) as usize;
    let x1 = (w as i32 + dx.min(0)) as usize;
    for y in y0..y1 {
        let ay = y;
        let by = (y as i32 - dy) as usize;
        for x in x0..x1 {
            let ax = x;
            let bx = (x as i32 - dx) as usize;
            for c in 0..3 {
                let av = a[ay * w3 + ax * 3 + c] as f64;
                let bv = b[by * w3 + bx * 3 + c] as f64;
                sum += (av - bv).abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tj = TurboJpeg::load(args.lib.as_deref())?;
    eprintln!("libturbojpeg: {}", tj.path);

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

    let mut tensor = Tensor::<u8>::image(
        8192,
        8192,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )?;
    let mut decoder = ImageDecoder::new();
    decoder.set_output_format(Some(PixelFormat::Rgb));

    let mut n_images = 0usize;
    let mut n_skipped_not_420 = 0usize;
    let mut n_dim_mismatch = 0usize;
    let mut worst_cosine = 1.0f64;
    let mut worst_psnr = f64::INFINITY;
    let mut global_max_diff = 0i32;
    let mut sum_cosine = 0.0f64;
    let mut sum_psnr = 0.0f64;
    // Best (lowest) mean|d| shift seen across the whole corpus, and how often
    // a nonzero shift beat (0, 0) on a given image — the phase-bug signal.
    let mut zero_shift_never_best = 0usize;
    let offsets: Vec<(i32, i32)> = (-1..=1)
        .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
        .collect();

    for path in &files {
        let data = std::fs::read(path)?;
        let hal = match tensor.load_image(&mut decoder, &data) {
            Ok(info) => info,
            // Not 4:2:0/4:4:4-equivalent (`UnsupportedFormat`) or otherwise
            // corrupt/unsupported: skip both arms. `UnsupportedFormat` is
            // the expected, common case here — most corpora mix in a few
            // greyscale/4:2:2 images fused RGB doesn't engage for.
            Err(edgefirst_codec::CodecError::UnsupportedFormat(_)) => {
                n_skipped_not_420 += 1;
                continue;
            }
            Err(_) => continue,
        };
        let (tw, th, turbo_rgb) = match tj.decode_rgb_box(&data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if tw != hal.width || th != hal.height {
            n_dim_mismatch += 1;
            continue;
        }

        let map = tensor.map()?;
        let hal_data: &[u8] = &map;
        let stride = hal.row_stride;
        let w3 = hal.width * 3;
        // Tight-pack HAL's output for the shift-diagnostic helper (which
        // assumes `width*3` stride); turbo's buffer is already tight.
        let mut hal_tight = vec![0u8; w3 * hal.height];
        for y in 0..hal.height {
            hal_tight[y * w3..y * w3 + w3].copy_from_slice(&hal_data[y * stride..y * stride + w3]);
        }
        drop(map);

        let (mut dot, mut na, mut nf, mut se) = (0f64, 0f64, 0f64, 0f64);
        let mut max_diff = 0i32;
        for i in 0..w3 * hal.height {
            let (x, y) = (hal_tight[i] as f64, turbo_rgb[i] as f64);
            dot += x * y;
            na += x * x;
            nf += y * y;
            let d = (hal_tight[i] as i32 - turbo_rgb[i] as i32).abs();
            se += (d * d) as f64;
            max_diff = max_diff.max(d);
        }
        let cosine = dot / (na.sqrt() * nf.sqrt()).max(1e-12);
        let mse = se / (w3 * hal.height) as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };

        let zero_shift_diff = shifted_diff(&hal_tight, &turbo_rgb, hal.width, hal.height, 0, 0);
        let mut best_shift = (0i32, 0i32);
        let mut best_diff = zero_shift_diff;
        for &(dx, dy) in &offsets {
            if dx == 0 && dy == 0 {
                continue;
            }
            let d = shifted_diff(&hal_tight, &turbo_rgb, hal.width, hal.height, dx, dy);
            if d < best_diff {
                best_diff = d;
                best_shift = (dx, dy);
            }
        }
        if best_shift != (0, 0) {
            zero_shift_never_best += 1;
        }

        n_images += 1;
        sum_cosine += cosine;
        sum_psnr += psnr.min(99.0);
        worst_cosine = worst_cosine.min(cosine);
        worst_psnr = worst_psnr.min(psnr);
        global_max_diff = global_max_diff.max(max_diff);
        if args.verbose {
            println!(
                "{}: cosine={cosine:.7} psnr={psnr:.2} dB max|d|={max_diff} \
                 zero_shift_mean|d|={zero_shift_diff:.4} best_shift={best_shift:?} \
                 best_mean|d|={best_diff:.4}",
                path.file_name().unwrap().to_string_lossy()
            );
        }
    }

    anyhow::ensure!(n_images > 0, "no native-4:2:0 images decoded on both arms");

    println!(
        "== EdgeFirst box-upsample RGB vs turbo TJFLAG_FASTUPSAMPLE RGB over {n_images} images"
    );
    println!("   (skipped: {n_skipped_not_420} not native-4:2:0, {n_dim_mismatch} dim mismatch)");
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
    println!("  max|d|:  {global_max_diff}");
    println!(
        "  phase:   {zero_shift_never_best}/{n_images} images where a ±1px shift beat zero-shift \
         alignment (0 expected if the replication phase matches)"
    );
    Ok(())
}
