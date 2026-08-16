// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Pixel-parity check against libjpeg-turbo `islow` — the single in-process
//! accuracy reference every other decode arm is measured against (see
//! BENCHMARKS.md § JPEG Decode). `islow` is never measured against itself;
//! every other arm gets a cosine / mean\|Δ\| / max\|Δ\| / PSNR row, plus a
//! phase-shift diagnostic on 4:2:0 sources.
//!
//! Two reference configurations select Table A (conformance, matched filter
//! class) vs Table B (accuracy cost, always plain accurate `islow`):
//!
//! - `--reference matched` (default): 4:2:0 sources compare against
//!   `islow + TJFLAG_FASTUPSAMPLE` (turbo's own box/nearest chroma upsample —
//!   the like-for-like comparison for EdgeFirst's box-upsample RGB path);
//!   4:4:4 sources are unaffected by the flag (no chroma resampling either
//!   side).
//! - `--reference accurate`: always plain `islow`, no upsample flag — what a
//!   speed option (EdgeFirst `fast`, turbo `ifast`) trades away against the
//!   accurate baseline, independent of chroma-filter choice.
//!
//! Four engines: `edgefirst-accurate`/`edgefirst-fast` (this crate, via
//! `edgefirst-codec`), `zune` (zune-jpeg direct), `image` (the `image` crate,
//! RGB only — wraps zune, see the arm-table note on why both are kept).
//!
//! `--yplane` compares EdgeFirst's native NV12/16/24 Y-plane against
//! `tjDecompressToYUV2` instead of the RGB path (`edgefirst-*` engines only —
//! the RGB-only comparator engines have no raw-YUV output to compare).
//!
//! ```bash
//! rgb_parity --dir /data/coco/val2017-yuv420 --engine edgefirst-accurate --reference matched
//! rgb_parity --dir /data/coco/val2017 --engine zune --reference accurate
//! rgb_parity --dir /data/coco/val2017-yuv420 --engine edgefirst-accurate --yplane
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use edgefirst_codec::{CodecError, DctMethod, ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory, TensorTrait};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uchar, c_ulong, c_void, CStr};

const TJPF_RGB: c_int = 0;
const TJFLAG_FASTUPSAMPLE: c_int = 256;
const TJFLAG_FASTDCT: c_int = 2048;
const TJFLAG_ACCURATEDCT: c_int = 4096;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Engine {
    EdgefirstAccurate,
    EdgefirstFast,
    Zune,
    Image,
    /// Table B only: turbo `ifast` vs plain accurate `islow` — both sides
    /// decoded in-process via the same dlopen'd library, just two
    /// `tjDecompress2` calls with different flags. `--reference` is ignored
    /// (always plain `islow`; `ifast` has no box-upsample filter class of
    /// its own to be "matched" against).
    TurboIfast,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Reference {
    /// Table A (conformance): 4:2:0 sources vs `islow + TJFLAG_FASTUPSAMPLE`.
    Matched,
    /// Table B (accuracy cost): always plain accurate `islow`.
    Accurate,
}

#[derive(Parser)]
struct Args {
    /// JPEG corpus directory.
    #[arg(long, env = "EDGEFIRST_BENCH_COCO")]
    dir: std::path::PathBuf,
    /// Maximum number of images (0 = all).
    #[arg(long, default_value_t = 200)]
    limit: usize,
    #[arg(long, value_enum, default_value_t = Engine::EdgefirstAccurate)]
    engine: Engine,
    #[arg(long, value_enum, default_value_t = Reference::Matched)]
    reference: Reference,
    /// Compare EdgeFirst's native NV12/16/24 Y-plane against
    /// `tjDecompressToYUV2` instead of the RGB path. `edgefirst-*` engines
    /// only.
    #[arg(long)]
    yplane: bool,
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
type TjBufSizeYuv2 = unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> c_ulong;
type TjDecompressToYuv2 = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type TjPlaneWidth = unsafe extern "C" fn(c_int, c_int, c_int) -> c_int;
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
    buf_size_yuv2: TjBufSizeYuv2,
    decompress_to_yuv2: TjDecompressToYuv2,
    plane_width: TjPlaneWidth,
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
        let buf_size_yuv2: TjBufSizeYuv2 = *unsafe { lib.get(b"tjBufSizeYUV2\0") }?;
        let decompress_to_yuv2: TjDecompressToYuv2 = *unsafe { lib.get(b"tjDecompressToYUV2\0") }?;
        let plane_width: TjPlaneWidth = *unsafe { lib.get(b"tjPlaneWidth\0") }?;
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
            buf_size_yuv2,
            decompress_to_yuv2,
            plane_width,
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

    /// Decode to interleaved RGB with accurate IDCT, optionally with
    /// `TJFLAG_FASTUPSAMPLE` (box/nearest chroma upsample — Table A's matched
    /// config for 4:2:0 sources; omitted for Table B's accurate-vs-accurate
    /// comparison).
    fn decode_rgb(&self, jpeg: &[u8], flags: c_int) -> Result<(usize, usize, Vec<u8>)> {
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
                flags,
            ) != 0
            {
                bail!("tjDecompress2: {}", self.err());
            }
            Ok((w as usize, h as usize, buf))
        }
    }

    /// Decode to a planar Y/U/V buffer (row alignment 1) with accurate IDCT
    /// and return a *tight* `width * height` Y plane — the comparison
    /// surface for EdgeFirst's semi-planar NV* arms, whose Y plane is
    /// likewise full-resolution and independent of chroma subsampling.
    ///
    /// `align=1` does **not** mean the Y plane's row stride equals `width`:
    /// `tjPlaneWidth(0, width, subsamp)` rounds up to the chroma-subsampling
    /// grid (e.g. 427 → 428 for 4:2:0, matching EdgeFirst's own
    /// `even_width` convention) even at align 1, since the luma plane must
    /// stay aligned with the chroma blocks it feeds. Assuming stride==width
    /// on an odd-width source silently drifts the row-extraction offset by
    /// one byte per row, compounding into a large apparent mismatch by the
    /// bottom of the image — this bit a first draft of this function
    /// (confirmed via cross-check against `decode_rgb`'s RGB→Y before
    /// tracking down `tjPlaneWidth`; the codec itself was never wrong).
    fn decode_y_plane(&self, jpeg: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
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
            let need = (self.buf_size_yuv2)(w, 1, h, subsamp);
            let mut buf = vec![0u8; need as usize];
            if (self.decompress_to_yuv2)(
                self.handle,
                jpeg.as_ptr(),
                jpeg.len() as c_ulong,
                buf.as_mut_ptr(),
                w,
                1,
                h,
                TJFLAG_ACCURATEDCT,
            ) != 0
            {
                bail!("tjDecompressToYUV2: {}", self.err());
            }
            let y_stride = (self.plane_width)(0, w, subsamp) as usize;
            let (wu, hu) = (w as usize, h as usize);
            let mut y_plane = vec![0u8; wu * hu];
            for row in 0..hu {
                y_plane[row * wu..row * wu + wu]
                    .copy_from_slice(&buf[row * y_stride..row * y_stride + wu]);
            }
            Ok((wu, hu, y_plane))
        }
    }
}

impl Drop for TurboJpeg {
    fn drop(&mut self) {
        // Best-effort; process exits shortly after in every use of this tool.
        unsafe { (self.destroy)(self.handle) };
    }
}

/// mean|d|, max|d| between two same-length single-channel or interleaved
/// byte buffers, and cosine/PSNR over the same data.
struct Stats {
    cosine: f64,
    mean_abs: f64,
    max_abs: i32,
    psnr: f64,
}

fn compare(a: &[u8], b: &[u8]) -> Stats {
    let (mut dot, mut na, mut nf, mut se, mut sum_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut max_diff = 0i32;
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nf += y * y;
        let d = (a[i] as i32 - b[i] as i32).abs();
        se += (d * d) as f64;
        sum_abs += d as f64;
        max_diff = max_diff.max(d);
    }
    let cosine = dot / (na.sqrt() * nf.sqrt()).max(1e-12);
    let mse = se / a.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    Stats {
        cosine,
        mean_abs: sum_abs / a.len() as f64,
        max_abs: max_diff,
        psnr,
    }
}

/// mean|d| (RGB channel bytes) between `a` and `b` when `b` is shifted by
/// `(dx, dy)` pixels relative to `a`, over the interior region common to both
/// after the shift. `w`/`h` are pixel dimensions; `channels` is 3 for RGB, 1
/// for a Y plane.
fn shifted_diff(a: &[u8], b: &[u8], w: usize, h: usize, channels: usize, dx: i32, dy: i32) -> f64 {
    let stride = w * channels;
    let mut sum = 0f64;
    let mut count = 0usize;
    let y0 = dy.max(0) as usize;
    let y1 = (h as i32 + dy.min(0)) as usize;
    let x0 = dx.max(0) as usize;
    let x1 = (w as i32 + dx.min(0)) as usize;
    for y in y0..y1 {
        let by = (y as i32 - dy) as usize;
        for x in x0..x1 {
            let bx = (x as i32 - dx) as usize;
            for c in 0..channels {
                let av = a[y * stride + x * channels + c] as f64;
                let bv = b[by * stride + bx * channels + c] as f64;
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

/// One image's worth of EdgeFirst decode, in RGB mode: tight-packed
/// (`width*3` stride) so it's directly comparable to turbo's buffer. `Ok(None)`
/// means this arm doesn't support the source (e.g. `CodecError::UnsupportedFormat`
/// on a non-4:4:4/non-4:2:0 subsampling, or a zune/image decode error on an
/// image that engine can't handle — same "skip, not a fault" convention the
/// original box-upsample-only version of this tool used).
fn decode_engine_rgb(
    engine: Engine,
    ef_tensor: &mut Tensor<u8>,
    ef_decoder: &mut ImageDecoder,
    zune_buf: &mut Vec<u8>,
    data: &[u8],
) -> Result<Option<(usize, usize, Vec<u8>)>> {
    match engine {
        Engine::EdgefirstAccurate | Engine::EdgefirstFast => {
            match ef_tensor.load_image(ef_decoder, data) {
                Ok(info) => {
                    let map = ef_tensor.map()?;
                    let hal_data: &[u8] = &map;
                    let stride = info.row_stride;
                    let w3 = info.width * 3;
                    let mut tight = vec![0u8; w3 * info.height];
                    for y in 0..info.height {
                        tight[y * w3..y * w3 + w3]
                            .copy_from_slice(&hal_data[y * stride..y * stride + w3]);
                    }
                    Ok(Some((info.width, info.height, tight)))
                }
                Err(CodecError::UnsupportedFormat(_)) => Ok(None),
                Err(_) => Ok(None),
            }
        }
        Engine::Zune => {
            let opts = zune_core::options::DecoderOptions::default()
                .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
            let mut dec =
                zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(data), opts);
            if dec.decode_headers().is_err() {
                return Ok(None);
            }
            let Some(need) = dec.output_buffer_size() else {
                return Ok(None);
            };
            if zune_buf.len() < need {
                zune_buf.resize(need, 0);
            }
            if dec.decode_into(zune_buf).is_err() {
                return Ok(None);
            }
            let Some((w, h)) = dec.dimensions() else {
                return Ok(None);
            };
            Ok(Some((w, h, zune_buf[..need].to_vec())))
        }
        Engine::Image => {
            match image::load_from_memory_with_format(data, image::ImageFormat::Jpeg) {
                Ok(img) => {
                    let rgb = img.to_rgb8();
                    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                    Ok(Some((w, h, rgb.into_raw())))
                }
                Err(_) => Ok(None),
            }
        }
        Engine::TurboIfast => unreachable!("run_rgb branches TurboIfast to tj.decode_rgb directly"),
    }
}

fn list_files(dir: &std::path::Path, limit: usize) -> Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    files.sort();
    if limit > 0 && limit < files.len() {
        let len = files.len();
        files = (0..limit).map(|i| files[i * len / limit].clone()).collect();
    }
    anyhow::ensure!(!files.is_empty(), "no JPEGs in {}", dir.display());
    Ok(files)
}

fn engine_name(e: Engine) -> &'static str {
    match e {
        Engine::EdgefirstAccurate => "edgefirst-accurate",
        Engine::EdgefirstFast => "edgefirst-fast",
        Engine::Zune => "zune",
        Engine::Image => "image",
        Engine::TurboIfast => "turbo-ifast",
    }
}

fn run_rgb(args: &Args, tj: &TurboJpeg, files: &[std::path::PathBuf]) -> Result<()> {
    // TurboIfast always compares against plain accurate islow — Table B
    // only, no matched-config filter class to join Table A under.
    let turbo_flags = if args.engine == Engine::TurboIfast {
        TJFLAG_ACCURATEDCT
    } else {
        TJFLAG_ACCURATEDCT
            | if args.reference == Reference::Matched {
                TJFLAG_FASTUPSAMPLE
            } else {
                0
            }
    };

    let mut tensor = Tensor::<u8>::image(
        8192,
        8192,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )?;
    let mut decoder = ImageDecoder::new();
    decoder.set_output_format(Some(PixelFormat::Rgb));
    if args.engine == Engine::EdgefirstFast {
        decoder.set_dct_method(DctMethod::Fast);
    }
    let mut zune_buf = Vec::new();

    let mut n_images = 0usize;
    let mut n_skipped = 0usize;
    let mut n_dim_mismatch = 0usize;
    let mut worst_cosine = 1.0f64;
    let mut worst_psnr = f64::INFINITY;
    let mut global_max_diff = 0i32;
    let mut sum_cosine = 0.0f64;
    let mut sum_mean_abs = 0.0f64;
    let mut sum_psnr = 0.0f64;
    let mut zero_shift_never_best = 0usize;
    let offsets: Vec<(i32, i32)> = (-1..=1)
        .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
        .collect();

    for path in files {
        let data = std::fs::read(path)?;
        let Some((ew, eh, engine_rgb)) = (if args.engine == Engine::TurboIfast {
            tj.decode_rgb(&data, TJFLAG_FASTDCT).ok()
        } else {
            decode_engine_rgb(args.engine, &mut tensor, &mut decoder, &mut zune_buf, &data)?
        }) else {
            n_skipped += 1;
            continue;
        };
        let (tw, th, turbo_rgb) = match tj.decode_rgb(&data, turbo_flags) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if tw != ew || th != eh {
            n_dim_mismatch += 1;
            continue;
        }

        let stats = compare(&engine_rgb, &turbo_rgb);

        let zero_shift_diff = shifted_diff(&engine_rgb, &turbo_rgb, ew, eh, 3, 0, 0);
        let mut best_shift = (0i32, 0i32);
        let mut best_diff = zero_shift_diff;
        for &(dx, dy) in &offsets {
            if dx == 0 && dy == 0 {
                continue;
            }
            let d = shifted_diff(&engine_rgb, &turbo_rgb, ew, eh, 3, dx, dy);
            if d < best_diff {
                best_diff = d;
                best_shift = (dx, dy);
            }
        }
        if best_shift != (0, 0) {
            zero_shift_never_best += 1;
        }

        n_images += 1;
        sum_cosine += stats.cosine;
        sum_mean_abs += stats.mean_abs;
        sum_psnr += stats.psnr.min(99.0);
        worst_cosine = worst_cosine.min(stats.cosine);
        worst_psnr = worst_psnr.min(stats.psnr);
        global_max_diff = global_max_diff.max(stats.max_abs);
        if args.verbose {
            println!(
                "{}: cosine={:.7} mean|d|={:.4} max|d|={} psnr={:.2} \
                 zero_shift_mean|d|={zero_shift_diff:.4} best_shift={best_shift:?}",
                path.file_name().unwrap().to_string_lossy(),
                stats.cosine,
                stats.mean_abs,
                stats.max_abs,
                stats.psnr,
            );
        }
    }

    anyhow::ensure!(n_images > 0, "no images decoded on both arms");

    let ref_label = if args.engine == Engine::TurboIfast {
        "islow accurate (Table B, --reference ignored)"
    } else {
        match args.reference {
            Reference::Matched => "islow + TJFLAG_FASTUPSAMPLE (matched, Table A)",
            Reference::Accurate => "islow accurate (Table B)",
        }
    };
    println!(
        "== {} RGB vs turbo {ref_label} over {n_images} images",
        engine_name(args.engine)
    );
    println!("   (skipped: {n_skipped} unsupported, {n_dim_mismatch} dim mismatch)");
    println!(
        "  cosine:   mean={:.7}  worst={:.7}",
        sum_cosine / n_images as f64,
        worst_cosine
    );
    println!("  mean|d|:  {:.4}", sum_mean_abs / n_images as f64);
    println!("  max|d|:   {global_max_diff}");
    println!(
        "  psnr:     mean={:.2} dB  worst={:.2} dB",
        sum_psnr / n_images as f64,
        worst_psnr
    );
    println!(
        "  phase:    {zero_shift_never_best}/{n_images} images where a ±1px shift beat zero-shift \
         alignment (0 expected if the replication phase matches)"
    );
    Ok(())
}

fn run_yplane(args: &Args, tj: &TurboJpeg, files: &[std::path::PathBuf]) -> Result<()> {
    let mut tensor = Tensor::<u8>::image(
        8192,
        8192,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )?;
    let mut decoder = ImageDecoder::new();
    // No set_output_format: leave the decoder on its native NV12/16/24
    // output — the Y plane is full-resolution and identically laid out
    // (row_stride pitch, first `height * row_stride` bytes) regardless of
    // which NV* variant the source's chroma subsampling selects.
    if args.engine == Engine::EdgefirstFast {
        decoder.set_dct_method(DctMethod::Fast);
    }

    let mut n_images = 0usize;
    let mut n_skipped = 0usize;
    let mut n_dim_mismatch = 0usize;
    let mut worst_cosine = 1.0f64;
    let mut worst_psnr = f64::INFINITY;
    let mut global_max_diff = 0i32;
    let mut sum_cosine = 0.0f64;
    let mut sum_mean_abs = 0.0f64;
    let mut sum_psnr = 0.0f64;

    for path in files {
        let data = std::fs::read(path)?;
        let info = match tensor.load_image(&mut decoder, &data) {
            Ok(info) => info,
            Err(_) => {
                n_skipped += 1;
                continue;
            }
        };
        let (tw, th, turbo_y) = match tj.decode_y_plane(&data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if tw != info.width || th != info.height {
            n_dim_mismatch += 1;
            continue;
        }

        let map = tensor.map()?;
        let hal_data: &[u8] = &map;
        let stride = info.row_stride;
        let mut ef_y = vec![0u8; info.width * info.height];
        for y in 0..info.height {
            ef_y[y * info.width..y * info.width + info.width]
                .copy_from_slice(&hal_data[y * stride..y * stride + info.width]);
        }
        drop(map);

        let stats = compare(&ef_y, &turbo_y);
        n_images += 1;
        sum_cosine += stats.cosine;
        sum_mean_abs += stats.mean_abs;
        sum_psnr += stats.psnr.min(99.0);
        worst_cosine = worst_cosine.min(stats.cosine);
        worst_psnr = worst_psnr.min(stats.psnr);
        global_max_diff = global_max_diff.max(stats.max_abs);
        if args.verbose {
            println!(
                "{}: cosine={:.7} mean|d|={:.4} max|d|={} psnr={:.2}",
                path.file_name().unwrap().to_string_lossy(),
                stats.cosine,
                stats.mean_abs,
                stats.max_abs,
                stats.psnr,
            );
        }
    }

    anyhow::ensure!(n_images > 0, "no images decoded on both arms");

    println!(
        "== {} Y-plane (native NV12/16/24) vs turbo tjDecompressToYUV2 islow over {n_images} images",
        engine_name(args.engine)
    );
    println!("   (skipped: {n_skipped} unsupported, {n_dim_mismatch} dim mismatch)");
    println!(
        "  cosine:   mean={:.7}  worst={:.7}",
        sum_cosine / n_images as f64,
        worst_cosine
    );
    println!("  mean|d|:  {:.4}", sum_mean_abs / n_images as f64);
    println!("  max|d|:   {global_max_diff}");
    println!(
        "  psnr:     mean={:.2} dB  worst={:.2} dB",
        sum_psnr / n_images as f64,
        worst_psnr
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.yplane {
        anyhow::ensure!(
            matches!(
                args.engine,
                Engine::EdgefirstAccurate | Engine::EdgefirstFast
            ),
            "--yplane only supports --engine edgefirst-accurate|edgefirst-fast: the Y-plane \
             comparison targets EdgeFirst's native NV12/16/24 output, which zune/image have no \
             equivalent of (RGB-only comparator engines)"
        );
    }

    let tj = TurboJpeg::load(args.lib.as_deref())?;
    eprintln!("libturbojpeg: {}", tj.path);

    let files = list_files(&args.dir, args.limit)?;

    if args.yplane {
        run_yplane(&args, &tj, &files)
    } else {
        run_rgb(&args, &tj, &files)
    }
}
