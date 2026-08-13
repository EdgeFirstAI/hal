// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for Article-1 JPEG → letterbox benchmark modules.

mod cpu;

pub use cpu::{CpuPeakTracker, CpuSnapshot, CpuStats};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_image::{
    ComputeBackend, Crop, Flip, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait,
    Region, Rotation,
};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const MODEL_W: usize = 640;
pub const MODEL_H: usize = 640;
pub const LETTERBOX_PAD: [u8; 4] = [114, 114, 114, 255];
pub const DEFAULT_WARMUP: usize = 10;
pub const DEFAULT_LIMIT: usize = 50;

/// CLI shared by every Rust HAL module.
#[derive(Debug, Parser)]
#[command(about = "JPEG decode → 640×640 letterbox latency sweep")]
pub struct BenchArgs {
    /// Directory of JPEG images (COCO val2017).
    #[arg(
        long,
        env = "EDGEFIRST_BENCH_COCO",
        default_value = "~/Datasets/COCO/val2017"
    )]
    pub coco: String,

    /// Max images to process (smoke). Omit or 0 for the full directory.
    #[arg(long, default_value_t = DEFAULT_LIMIT)]
    pub limit: usize,

    /// Warmup iterations before measured passes (applied once on the first image).
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    pub warmup: usize,

    /// Board / host label written into the CSV.
    #[arg(long, default_value = "unknown")]
    pub board: String,

    /// Module / class label (e.g. hal_cpu, hal_gl).
    #[arg(long)]
    pub module: Option<String>,

    /// Write summary CSV to this path.
    #[arg(long)]
    pub csv: Option<PathBuf>,

    /// Also print per-image timings to stderr.
    #[arg(long)]
    pub verbose: bool,

    /// Force decode/output tensor memory backend.
    ///
    /// `auto` follows HAL (DMA→PBO→Mem). Prior `codec_benchmark` numbers used
    /// Mem; decoding into DMA-BUF pays sync/coherency cost on the CPU write path.
    #[arg(long, value_enum, default_value_t = TensorMem::Auto)]
    pub tensor_mem: TensorMem,

    /// Write a Chrome/Perfetto JSON trace of HAL `tracing` spans (one shot).
    /// Also accepts `EDGEFIRST_TRACE=/path/to/trace.json`.
    #[arg(long, env = "EDGEFIRST_TRACE")]
    pub trace: Option<PathBuf>,

    /// Decoder output format: `native` (NV24/NV16/NV12 per source), `rgb`
    /// (fused YCbCr→RGB for 4:4:4 sources; convert becomes a pure resize),
    /// or `nv12` (fused chroma downsample).
    #[arg(long, value_enum, env = "EDGEFIRST_BENCH_DECODE_FMT", default_value_t = DecodeFmt::Native)]
    pub decode_fmt: DecodeFmt,

    /// Time JPEG→raster only (skip `ImageProcessor::convert` / letterbox).
    ///
    /// Use with `--decode-fmt native` (YUV/NV*) or `--decode-fmt rgb` for a
    /// fair decoder A/B against TurboJPEG `--decode-only --format yuv|rgb`.
    /// Model-input preprocessing is out of scope for this mode.
    #[arg(long, env = "EDGEFIRST_BENCH_DECODE_ONLY", default_value_t = false)]
    pub decode_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum DecodeFmt {
    #[default]
    Native,
    Rgb,
    Nv12,
}

impl DecodeFmt {
    pub fn to_preference(self) -> Option<PixelFormat> {
        match self {
            DecodeFmt::Native => None,
            DecodeFmt::Rgb => Some(PixelFormat::Rgb),
            DecodeFmt::Nv12 => Some(PixelFormat::Nv12),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum TensorMem {
    #[default]
    Auto,
    Dma,
    Mem,
}

impl TensorMem {
    pub fn to_option(self) -> Option<TensorMemory> {
        match self {
            TensorMem::Auto => None,
            #[cfg(target_os = "linux")]
            TensorMem::Dma => Some(TensorMemory::Dma),
            #[cfg(not(target_os = "linux"))]
            TensorMem::Dma => None,
            TensorMem::Mem => Some(TensorMemory::Mem),
        }
    }
}

impl BenchArgs {
    pub fn coco_dir(&self) -> PathBuf {
        expand_tilde(&self.coco)
    }

    pub fn effective_limit(&self) -> Option<usize> {
        if self.limit == 0 {
            None
        } else {
            Some(self.limit)
        }
    }
}

pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    if path == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

/// Sorted list of `.jpg` / `.jpeg` paths under `dir`.
///
/// When `limit` is set and smaller than the full set, returns **evenly spaced**
/// samples across the sorted directory. COCO val2017's lexicographic prefix is
/// dominated by 4:4:4 JPEGs; stride sampling avoids a biased smoke set.
pub fn list_jpegs(dir: &Path, limit: Option<usize>) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        bail!("COCO directory not found: {}", dir.display());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| {
                    let e = e.to_ascii_lowercase();
                    e == "jpg" || e == "jpeg"
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if let Some(n) = limit {
        if n > 0 && n < paths.len() {
            let len = paths.len();
            paths = (0..n)
                .map(|i| paths[i * len / n].clone())
                .collect();
        } else if n > 0 {
            paths.truncate(n);
        }
    }
    if paths.is_empty() {
        bail!("no JPEG files in {}", dir.display());
    }
    Ok(paths)
}

#[derive(Debug, Clone)]
pub struct TimingStats {
    pub n: usize,
    pub ms_p50: f64,
    pub ms_p95: f64,
    pub ms_p99: f64,
    pub mpix_per_s: f64,
    pub peak_rss_mb: f64,
}

impl TimingStats {
    pub fn from_samples(samples_ms: &mut [f64], total_mpix: f64) -> Self {
        samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples_ms.len();
        let pct = |p: f64| -> f64 {
            if n == 0 {
                return 0.0;
            }
            let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
            samples_ms[idx]
        };
        let sum_ms: f64 = samples_ms.iter().sum();
        let mpix_per_s = if sum_ms > 0.0 {
            total_mpix / (sum_ms / 1000.0)
        } else {
            0.0
        };
        Self {
            n,
            ms_p50: pct(0.50),
            ms_p95: pct(0.95),
            ms_p99: pct(0.99),
            mpix_per_s,
            peak_rss_mb: peak_rss_mb(),
        }
    }
}

pub fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    let kb: f64 = rest
                        .split_whitespace()
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    return kb / 1024.0;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // Darwin reports ru_maxrss in bytes (Linux uses kilobytes).
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            return ru.ru_maxrss as f64 / (1024.0 * 1024.0);
        }
    }
    0.0
}

pub fn write_summary_csv(
    path: &Path,
    board: &str,
    class: &str,
    module: &str,
    stats: &TimingStats,
    cpu: &CpuStats,
    notes: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(
        f,
        "board,class,module,ms_p50,ms_p95,ms_p99,mpix_per_s,peak_rss_mb,\
         cpu_pct_process,cpu_pct_system,cpu_pct_peak_core,n_images,notes"
    )?;
    writeln!(
        f,
        "{board},{class},{module},{:.3},{:.3},{:.3},{:.3},{:.1},{:.1},{:.1},{:.1},{},{}",
        stats.ms_p50,
        stats.ms_p95,
        stats.ms_p99,
        stats.mpix_per_s,
        stats.peak_rss_mb,
        cpu.process_pct,
        cpu.system_pct,
        cpu.peak_core_pct_max.max(cpu.peak_core_pct),
        stats.n,
        csv_escape(notes)
    )?;
    Ok(())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// How HAL modules configure decode / convert for a composed arm.
#[derive(Debug, Clone, Copy)]
pub struct HalModuleConfig {
    /// Plan class label for CSV (`cpu`, `hybrid_gl`, `hybrid_2d`, `hw_gl`).
    pub class: &'static str,
    /// Module directory name.
    pub module: &'static str,
    /// Value for `EDGEFIRST_FORCE_BACKEND` (`cpu` / `opengl` / `g2d`).
    pub force_backend: &'static str,
    /// When true, set `EDGEFIRST_DISABLE_V4L2=1` (software JPEG only).
    pub disable_v4l2: bool,
    /// When true and `--tensor-mem=auto`, force Mem decode source (matches
    /// historical `codec_benchmark` Mem numbers; avoids DMA write sync cost).
    pub prefer_heap_src: bool,
}

impl HalModuleConfig {
    pub fn apply_env(&self) {
        // Must be set before ImageProcessor::new / first decode.
        std::env::set_var("EDGEFIRST_FORCE_BACKEND", self.force_backend);
        if self.disable_v4l2 {
            std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
        } else {
            std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
        }
    }
}

/// Allocate the reusable decode source pool (Studio profiler pattern).
///
/// Sized as a single-plane Grey/R8 surface with height `3·max_h` so NV24
/// (and every smaller native JPEG layout) fits without reallocating. Allocating
/// as `Nv24` FourCC instead of Grey forces GL off the zero-copy R8 bind path.
fn create_decode_source_pool(
    processor: &ImageProcessor,
    max_w: usize,
    max_h: usize,
    memory: Option<TensorMemory>,
) -> Result<TensorDyn> {
    let pool_w = max_w.next_multiple_of(2);
    let pool_h = max_h
        .checked_mul(3)
        .context("decode source pool height overflow (3·max_h)")?;
    processor
        .create_image(
            pool_w,
            pool_h,
            PixelFormat::Grey,
            DType::U8,
            memory,
            CpuAccess::ReadWrite,
        )
        .context("create decode source pool (Grey R8, NV24 capacity)")
}

fn resolve_src_memory(cfg: &HalModuleConfig, args: &BenchArgs) -> Option<TensorMemory> {
    match args.tensor_mem {
        TensorMem::Auto if cfg.prefer_heap_src => Some(TensorMemory::Mem),
        other => other.to_option(),
    }
}

/// Letterbox crop; when the decoded image is smaller than the pooled tensor,
/// pin `source` so the GPU samples only the live sub-rect (profiler pattern).
fn letterbox_crop(info_w: usize, info_h: usize, tensor: &TensorDyn) -> Crop {
    let mut crop = Crop::letterbox(LETTERBOX_PAD);
    let shape = tensor.shape();
    let tensor_h = shape.first().copied().unwrap_or(0);
    let tensor_w = shape.get(1).copied().unwrap_or(0);
    if info_w < tensor_w || info_h < tensor_h {
        crop.source = Some(Region::new(0, 0, info_w, info_h));
    }
    crop
}

/// HAL JPEG latency sweep: decode → optional letterbox convert.
///
/// Hot loop reuses a single decode-source DMA pool (and, unless
/// `--decode-only`, a single 640×640 output tensor) — `load_image` / `convert`
/// only reconfigure logical dims, never allocate. JPEG bytes are preloaded so
/// disk I/O is outside the timed section.
///
/// With `--decode-only`, only `load_image` is timed (JPEG bytes → YUV/NV* or
/// fused RGB in the source pool). That is the fair decoder A/B against
/// TurboJPEG `--decode-only`; convert / letterbox is preprocessing and is not
/// included.
pub fn run_hal_module(cfg: HalModuleConfig, args: &BenchArgs) -> Result<TimingStats> {
    cfg.apply_env();

    let mut tracing_active = false;
    if let Some(path) = args.trace.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        match edgefirst_hal::trace::start_tracing(path.to_str().unwrap_or("/tmp/hal-bench-trace.json"))
        {
            Ok(()) => {
                tracing_active = true;
                eprintln!("  chrome trace → {}", path.display());
            }
            Err(e) => eprintln!("  WARN: EDGEFIRST_TRACE not started: {e}"),
        }
    }

    let coco = args.coco_dir();
    let paths = list_jpegs(&coco, args.effective_limit())?;
    let scope = if args.decode_only {
        "decode-only"
    } else {
        "decode+letterbox"
    };
    eprintln!(
        "=== {} ({}, {scope}) — {} images from {} ===",
        cfg.module,
        cfg.class,
        paths.len(),
        coco.display()
    );

    // Preload JPEG bytes + size the pool for the largest frame (init, not timed).
    let mut max_w = MODEL_W;
    let mut max_h = MODEL_H;
    let mut images: Vec<(String, Vec<u8>)> = Vec::with_capacity(paths.len());
    for p in &paths {
        let bytes = fs::read(p).with_context(|| format!("read {}", p.display()))?;
        let info = peek_info(&bytes).with_context(|| format!("peek {}", p.display()))?;
        max_w = max_w.max(info.width);
        max_h = max_h.max(info.height);
        images.push((
            p.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes,
        ));
    }
    eprintln!("  max source: {max_w}×{max_h}  (preloaded {} JPEGs)", images.len());

    let src_memory = resolve_src_memory(&cfg, args);
    // Keep dst on the same policy for CPU modules; GPU modules keep auto (DMA).
    let dst_memory = if cfg.prefer_heap_src && args.tensor_mem == TensorMem::Auto {
        Some(TensorMemory::Mem)
    } else {
        args.tensor_mem.to_option()
    };

    let mut proc = ImageProcessor::new().context("ImageProcessor::new")?;
    let mut input = create_decode_source_pool(&proc, max_w, max_h, src_memory)?;
    let mut output = if args.decode_only {
        None
    } else {
        Some(
            proc.create_image(
                MODEL_W,
                MODEL_H,
                PixelFormat::Rgb,
                DType::U8,
                dst_memory,
                // GPU path declares None; CPU fallback still writes the destination.
                CpuAccess::Write,
            )
            .context("create output")?,
        )
    };

    let src_mem = format!("{:?}", input.memory());
    let dst_mem = output
        .as_ref()
        .map(|o| format!("{:?}", o.memory()))
        .unwrap_or_else(|| "n/a".into());
    let pool_w = max_w.next_multiple_of(2);
    let pool_h = max_h * 3;
    let src_cap = pool_w * pool_h; // Grey R8 bytes (= element count)
    let src_id0 = input.buffer_identity().id();
    let dst_id0 = output.as_ref().map(|o| o.buffer_identity().id()).unwrap_or(0);
    eprintln!(
        "  pool: Grey R8  {pool_w}×{pool_h}  capacity≈{src_cap} B  src_mem={src_mem}  dst_mem={dst_mem}  src_id={src_id0}  dst_id={dst_id0}  tensor_mem={:?}",
        args.tensor_mem
    );

    // Host GL/PBO cannot upload Nv24; keep a CPU processor for those frames.
    let mut cpu_fallback: Option<ImageProcessor> = if cfg.force_backend == "cpu" || args.decode_only
    {
        None
    } else {
        Some(
            ImageProcessor::with_config(ImageProcessorConfig {
                backend: ComputeBackend::Cpu,
                ..Default::default()
            })
            .context("CPU fallback ImageProcessor")?,
        )
    };
    let mut cpu_fallback_frames = 0usize;
    let mut identity_churn = 0usize;
    let mut decoder = ImageDecoder::new();
    decoder.set_output_format(args.decode_fmt.to_preference());
    if args.decode_fmt != DecodeFmt::Native {
        eprintln!("  decode-fmt: {:?} (fused decode output)", args.decode_fmt);
    }
    if args.decode_only {
        eprintln!("  decode-only: skip ImageProcessor::convert (decoder A/B)");
    }

    let convert_one = |proc: &mut ImageProcessor,
                       cpu: &mut Option<ImageProcessor>,
                       input: &TensorDyn,
                       output: &mut TensorDyn,
                       crop: Crop,
                       fallbacks: &mut usize|
     -> Result<()> {
        match proc.convert(input, output, Rotation::None, Flip::None, crop) {
            Ok(()) => Ok(()),
            Err(e) => {
                if let Some(cpu_proc) = cpu.as_mut() {
                    *fallbacks += 1;
                    cpu_proc
                        .convert(input, output, Rotation::None, Flip::None, crop)
                        .with_context(|| format!("CPU fallback after: {e}"))?;
                    Ok(())
                } else {
                    Err(e.into())
                }
            }
        }
    };

    // Warm-fill + warmup (page faults / EGL import out of the timed loop).
    let first = &images[0].1;
    for _ in 0..args.warmup {
        let info = input
            .load_image(&mut decoder, first)
            .context("warmup decode")?;
        if let Some(output) = output.as_mut() {
            let crop = letterbox_crop(info.width, info.height, &input);
            convert_one(
                &mut proc,
                &mut cpu_fallback,
                &input,
                output,
                crop,
                &mut cpu_fallback_frames,
            )
            .context("warmup convert")?;
        }
    }
    cpu_fallback_frames = 0;

    let rss_before = peak_rss_mb();
    let cpu_start = CpuSnapshot::now();
    let mut cpu_peak = CpuPeakTracker::start();
    let mut samples_ms = Vec::with_capacity(images.len());
    let mut decode_ms = Vec::with_capacity(images.len());
    let mut convert_ms = Vec::with_capacity(images.len());
    let mut total_mpix = 0.0f64;

    for (i, (name, bytes)) in images.iter().enumerate() {
        let t0 = Instant::now();
        let info = input
            .load_image(&mut decoder, bytes)
            .with_context(|| format!("decode {name}"))?;
        let t1 = Instant::now();
        let t2 = if let Some(output) = output.as_mut() {
            let crop = letterbox_crop(info.width, info.height, &input);
            convert_one(
                &mut proc,
                &mut cpu_fallback,
                &input,
                output,
                crop,
                &mut cpu_fallback_frames,
            )
            .with_context(|| format!("convert {name}"))?;
            Instant::now()
        } else {
            t1
        };

        // BufferIdentity must stay stable — a new id means a silent realloc.
        let dst_changed = output
            .as_ref()
            .is_some_and(|o| o.buffer_identity().id() != dst_id0);
        if input.buffer_identity().id() != src_id0 || dst_changed {
            identity_churn += 1;
        }

        let ms = t2.duration_since(t0).as_secs_f64() * 1000.0;
        samples_ms.push(ms);
        decode_ms.push(t1.duration_since(t0).as_secs_f64() * 1000.0);
        convert_ms.push(t2.duration_since(t1).as_secs_f64() * 1000.0);
        total_mpix += (info.width as f64) * (info.height as f64) / 1_000_000.0;

        // Cheap /proc/stat sample every frame for peak-core tracking.
        if i % 4 == 0 {
            cpu_peak.sample();
        }

        if args.verbose {
            eprintln!(
                "  [{:>4}/{}] {:.3} ms (dec {:.3} + cvt {:.3})  {}×{} {:?}  src_id={}",
                i + 1,
                images.len(),
                ms,
                decode_ms[i],
                convert_ms[i],
                info.width,
                info.height,
                info.format,
                input.buffer_identity().id()
            );
        }
    }

    let cpu_end = CpuSnapshot::now();
    let mut cpu = cpu_end.delta_since(&cpu_start);
    cpu.peak_core_pct_max = cpu
        .peak_core_pct_max
        .max(cpu.peak_core_pct)
        .max(cpu_peak.peak_core_pct_max());

    let rss_after = peak_rss_mb();
    let stats = TimingStats::from_samples(&mut samples_ms, total_mpix);
    decode_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    convert_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dec_p50 = decode_ms
        .get(decode_ms.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let cvt_p50 = convert_ms
        .get(convert_ms.len() / 2)
        .copied()
        .unwrap_or(0.0);

    eprintln!(
        "  p50={:.3} ms  (decode_p50={:.3} convert_p50={:.3})  p95={:.3}  p99={:.3}  {:.1} MP/s",
        stats.ms_p50, dec_p50, cvt_p50, stats.ms_p95, stats.ms_p99, stats.mpix_per_s
    );
    eprintln!(
        "  cpu: process={:.1}%  system={:.1}%  peak_core={:.1}%  fallback_frames={cpu_fallback_frames}",
        cpu.process_pct, cpu.system_pct, cpu.peak_core_pct_max
    );
    eprintln!(
        "  reuse: src_id={src_id0} dst_id={dst_id0} identity_churn={identity_churn}  RSS {rss_before:.1}→{rss_after:.1} MB  n={}",
        stats.n
    );
    if identity_churn > 0 {
        eprintln!(
            "  WARNING: BufferIdentity changed during the hot loop — pool reuse is broken"
        );
    }

    let module_name = args.module.as_deref().unwrap_or(cfg.module);
    let csv_class = if args.decode_only { "decode" } else { cfg.class };
    if let Some(csv) = &args.csv {
        write_summary_csv(
            csv,
            &args.board,
            csv_class,
            module_name,
            &stats,
            &cpu,
            &format!(
                "backend={};decode_fmt={:?};scope={scope};src_mem={src_mem};dst_mem={dst_mem};src_id={src_id0};identity_churn={identity_churn};cpu_fallback_frames={cpu_fallback_frames};decode_p50={dec_p50:.3};convert_p50={cvt_p50:.3}",
                cfg.force_backend, args.decode_fmt
            ),
        )?;
        eprintln!("  wrote {}", csv.display());
    }

    if tracing_active {
        edgefirst_hal::trace::stop_tracing();
    }

    Ok(stats)
}
