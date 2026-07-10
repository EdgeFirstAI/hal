// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Android link-validation shim and on-device validation entry points.
//!
//! Two jobs, one crate (the Android counterpart of `ios-validation`, plus
//! the runtime role iOS defers to a future app shell):
//!
//! 1. **Link validation** (CI, no device): builds the full HAL dependency
//!    closure into a single `staticlib` and forces the AHardwareBuffer /
//!    GL symbol references into the archive so
//!    `scripts/validate-android-link.sh` can link it against the NDK
//!    system libraries and prove the native symbol closure is complete.
//!
//! 2. **On-device validation** (Device Farm): C-ABI entry points that
//!    exercise the REAL `ImageProcessor` (not probe code) — a GL-vs-CPU
//!    correctness check and a convert benchmark with an import-cache
//!    steady-state gate. The hal-mobile Android app links this staticlib
//!    and calls these through JNI, running them on the Galaxy S26 Ultra
//!    pool (see `hal-mobile/ASSESSMENT-ANDROID.md` for the Phase-1
//!    numbers these gate against).
//!
//! This crate is `publish = false` and intentionally NOT in the
//! workspace's `default-members` — it is built only on demand.

#![allow(dead_code)]

use edgefirst_hal as hal;

/// Force the Android native-library closure into the staticlib so the link
/// step in `scripts/validate-android-link.sh` genuinely validates it.
///
/// The body *calls* real constructors (not `TypeId::of`, which emits no
/// code): the concrete monomorphizations are codegen'd into this crate's
/// object file, so the archive carries actual undefined references to
/// `AHardwareBuffer_allocate`, `AHardwareBuffer_lock`, etc. That is what
/// makes the `-lnativewindow` flag at the link step load-bearing — drop it
/// and the link fails with undefined symbols. (EGL/GLES entry points are
/// resolved at RUNTIME via `libloading`/`eglGetProcAddress`, so they carry
/// no link-time reference; the script validates them separately with
/// `llvm-nm` against the NDK's libEGL stub.)
///
/// Never called at runtime by the validation; the script's `main.c` calls
/// it only so the linker pulls this archive member in (an archive member
/// is linked only to resolve an undefined symbol — without the call the
/// whole `.a` is dropped and the check is a false positive).
#[no_mangle]
pub extern "C" fn edgefirst_android_validation_force_closure() {
    // (1) tensor AHardwareBuffer path — `TensorMemory::Dma` routes to
    // `AHardwareBufferTensor::<u8>::new` → `AHardwareBuffer_allocate` (see
    // crates/tensor/src/ahardwarebuffer.rs), pulling that crate's
    // `#[link(name = "nativewindow")]` extern block into the archive.
    if let Ok(t) = hal::tensor::Tensor::<u8>::new(&[64], Some(hal::tensor::TensorMemory::Dma), None)
    {
        std::hint::black_box(&t);
    }

    // (2) image GL closure — constructing the `ImageProcessor` (default
    // features include `opengl`) pulls `GLProcessorThreaded` and the whole
    // `gl/` module — including the AHardwareBuffer EGLImage import path —
    // into the link, proving the full Android GL Rust closure is
    // self-consistent and links.
    if let Ok(p) = hal::image::ImageProcessor::new() {
        std::hint::black_box(&p);
    }
}

/// Keep `edgefirst_android_validation_force_closure` rooted through LTO
/// (same mechanism as the iOS shim).
#[used]
#[no_mangle]
static EDGEFIRST_ANDROID_VALIDATION_ROOT: extern "C" fn() =
    edgefirst_android_validation_force_closure;

// ---------------------------------------------------------------------------
// On-device validation entry points (Android only).
// ---------------------------------------------------------------------------

#[cfg(target_os = "android")]
mod device {
    use super::hal;
    use hal::image::{
        ComputeBackend, Crop, Flip, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait,
        Rotation,
    };
    use hal::tensor::{
        DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait as _, TensorMemory, TensorTrait,
    };

    /// Result codes for the C-ABI entry points. 0 = pass; negative =
    /// failure (logged via `log` — the embedding app installs the Android
    /// logger).
    pub const OK: i32 = 0;
    pub const ERR_GL_UNAVAILABLE: i32 = -1;
    pub const ERR_ALLOC: i32 = -2;
    pub const ERR_CONVERT: i32 = -3;
    pub const ERR_MISMATCH: i32 = -4;
    pub const ERR_MAP: i32 = -5;
    pub const ERR_INVALID_ARG: i32 = -6;

    /// Build a processor with the GL backend, or report why not.
    fn gl_processor() -> Result<ImageProcessor, i32> {
        let cfg = ImageProcessorConfig {
            backend: ComputeBackend::OpenGl,
            ..Default::default()
        };
        let p = ImageProcessor::with_config(cfg).map_err(|e| {
            log::error!("android-validation: ImageProcessor(OpenGl): {e:?}");
            ERR_GL_UNAVAILABLE
        })?;
        if p.opengl.is_none() {
            log::error!("android-validation: GL backend did not initialize");
            return Err(ERR_GL_UNAVAILABLE);
        }
        Ok(p)
    }

    fn cpu_processor() -> Result<ImageProcessor, i32> {
        let cfg = ImageProcessorConfig {
            backend: ComputeBackend::Cpu,
            ..Default::default()
        };
        ImageProcessor::with_config(cfg).map_err(|e| {
            log::error!("android-validation: ImageProcessor(Cpu): {e:?}");
            ERR_ALLOC
        })
    }

    /// Deterministic gradient source so the comparison exercises resampling
    /// (a solid fill would hide wrong-pitch sampling).
    fn make_gradient_src(w: usize, h: usize) -> Result<TensorDyn, i32> {
        let src = Tensor::<u8>::image(w, h, PixelFormat::Rgba, Some(TensorMemory::Dma))
            .or_else(|_| Tensor::<u8>::image(w, h, PixelFormat::Rgba, None))
            .map_err(|e| {
                log::error!("android-validation: src alloc {w}x{h}: {e:?}");
                ERR_ALLOC
            })?;
        {
            let mut m = src.map().map_err(|e| {
                log::error!("android-validation: src map: {e:?}");
                ERR_MAP
            })?;
            let stride = src.effective_row_stride().unwrap_or(w * 4);
            let buf = m.as_mut_slice();
            for y in 0..h {
                for x in 0..w {
                    let o = y * stride + x * 4;
                    if o + 3 < buf.len() {
                        buf[o] = (x * 255 / w.max(1)) as u8;
                        buf[o + 1] = (y * 255 / h.max(1)) as u8;
                        buf[o + 2] = ((x + y) * 255 / (w + h).max(1)) as u8;
                        buf[o + 3] = 255;
                    }
                }
            }
        }
        Ok(src.into())
    }

    fn convert_with(
        p: &mut ImageProcessor,
        src: &TensorDyn,
        dst: &mut TensorDyn,
    ) -> Result<(), i32> {
        p.convert(src, dst, Rotation::None, Flip::None, Crop::default())
            .map_err(|e| {
                log::error!("android-validation: convert: {e:?}");
                ERR_CONVERT
            })
    }

    /// GL-vs-CPU oracle on the F16 planar-RGB model-input path: the same
    /// gradient source converted by both backends must agree within a
    /// small tolerance (the established pattern of the HAL's own GL test
    /// suite — no letterbox-geometry assumptions to drift).
    pub fn verify_f16_planar(src_w: usize, src_h: usize, dst_size: usize) -> i32 {
        let mut gl = match gl_processor() {
            Ok(p) => p,
            Err(c) => return c,
        };
        let mut cpu = match cpu_processor() {
            Ok(p) => p,
            Err(c) => return c,
        };

        let src = match make_gradient_src(src_w, src_h) {
            Ok(s) => s,
            Err(c) => return c,
        };

        let mut dst_gl =
            match gl.create_image(dst_size, dst_size, PixelFormat::PlanarRgb, DType::F16, None) {
                Ok(d) => d,
                Err(e) => {
                    log::error!("android-validation: GL dst alloc: {e:?}");
                    return ERR_ALLOC;
                }
            };
        let mut dst_cpu = match TensorDyn::image(
            dst_size,
            dst_size,
            PixelFormat::PlanarRgb,
            DType::F16,
            Some(TensorMemory::Mem),
        ) {
            Ok(d) => d,
            Err(e) => {
                log::error!("android-validation: CPU dst alloc: {e:?}");
                return ERR_ALLOC;
            }
        };

        if let Err(c) = convert_with(&mut gl, &src, &mut dst_gl) {
            return c;
        }
        if let Err(c) = convert_with(&mut cpu, &src, &mut dst_cpu) {
            return c;
        }

        let (Some(a), Some(b)) = (dst_gl.as_f16(), dst_cpu.as_f16()) else {
            log::error!("android-validation: dst tensors are not F16");
            return ERR_MISMATCH;
        };
        let (Ok(ma), Ok(mb)) = (a.map(), b.map()) else {
            log::error!("android-validation: dst map failed");
            return ERR_MAP;
        };
        let (sa, sb) = (ma.as_slice(), mb.as_slice());
        if sa.len() != sb.len() {
            log::error!(
                "android-validation: dst length mismatch ({} vs {})",
                sa.len(),
                sb.len()
            );
            return ERR_MISMATCH;
        }
        // Normalized [0,1] outputs; GPU linear filtering + f16 rounding vs
        // the CPU path justifies a small tolerance (the suite uses 4/255 on
        // u8 paths — mirror it in float space).
        let tolerance = 4.0 / 255.0;
        let mut max_diff = 0f32;
        for (va, vb) in sa.iter().zip(sb.iter()) {
            let d = (va.to_f32() - vb.to_f32()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        if max_diff > tolerance {
            log::error!(
                "android-validation: GL vs CPU F16 max diff {max_diff} > tolerance {tolerance}"
            );
            return ERR_MISMATCH;
        }
        log::info!(
            "android-validation: verify_f16_planar {src_w}x{src_h}→{dst_size}: \
             max diff {max_diff} (tolerance {tolerance}) — PASS"
        );
        OK
    }

    /// One benchmark cell: repeated full `convert()` (letterbox, normalize,
    /// and F16 NCHW pack, including the sync) with a warmup phase,
    /// reporting medians/p95 and the import-cache miss delta across the
    /// steady-state phase — which MUST be zero (a nonzero delta means the
    /// zero-copy import is re-created per frame).
    pub struct BenchOut {
        pub warmup_median_us: u64,
        pub median_us: u64,
        pub p95_us: u64,
        pub cache_miss_delta: u64,
    }

    pub fn bench_f16_planar(
        src_w: usize,
        src_h: usize,
        dst_size: usize,
        warmup: usize,
        iters: usize,
    ) -> Result<BenchOut, i32> {
        if iters == 0 {
            return Err(ERR_INVALID_ARG);
        }
        let mut gl = gl_processor()?;
        let src = make_gradient_src(src_w, src_h)?;
        let mut dst = gl
            .create_image(dst_size, dst_size, PixelFormat::PlanarRgb, DType::F16, None)
            .map_err(|e| {
                log::error!("android-validation: bench dst alloc: {e:?}");
                ERR_ALLOC
            })?;

        let time_one = |gl: &mut ImageProcessor, dst: &mut TensorDyn| -> Result<u64, i32> {
            let t0 = std::time::Instant::now();
            convert_with(gl, &src, dst)?;
            Ok(t0.elapsed().as_micros() as u64)
        };

        let mut warm: Vec<u64> = Vec::with_capacity(warmup);
        for _ in 0..warmup {
            warm.push(time_one(&mut gl, &mut dst)?);
        }

        // Steady state begins: the import cache must not miss again.
        let misses_before = gl
            .opengl
            .as_ref()
            .and_then(|g| g.egl_cache_stats().ok())
            .map(|s| s.total_misses())
            .unwrap_or(0);

        let mut times: Vec<u64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            times.push(time_one(&mut gl, &mut dst)?);
        }

        let misses_after = gl
            .opengl
            .as_ref()
            .and_then(|g| g.egl_cache_stats().ok())
            .map(|s| s.total_misses())
            .unwrap_or(0);

        times.sort_unstable();
        warm.sort_unstable();
        let median = times[times.len() / 2];
        let p95 = times[((times.len() as f64 * 0.95).ceil() as usize).clamp(1, times.len()) - 1];
        let warm_median = warm.get(warm.len() / 2).copied().unwrap_or(0);

        Ok(BenchOut {
            warmup_median_us: warm_median,
            median_us: median,
            p95_us: p95,
            cache_miss_delta: misses_after.saturating_sub(misses_before),
        })
    }
}

/// C-ABI benchmark result (see [`device::bench_f16_planar`]).
#[repr(C)]
pub struct EdgefirstAndroidBench {
    pub warmup_median_us: u64,
    pub median_us: u64,
    pub p95_us: u64,
    /// Import-cache misses observed during the steady-state phase — must
    /// be 0 (nonzero means the zero-copy import is re-created per frame).
    pub cache_miss_delta: u64,
}

/// Run the GL-vs-CPU F16 planar-RGB correctness oracle on the real
/// `ImageProcessor`. Returns 0 on pass, a negative code on failure (see
/// the `device` module constants).
#[no_mangle]
#[cfg(target_os = "android")]
pub extern "C" fn edgefirst_android_validation_verify(
    src_w: usize,
    src_h: usize,
    dst_size: usize,
) -> i32 {
    device::verify_f16_planar(src_w, src_h, dst_size)
}

/// Run one benchmark cell (letterbox+normalize → F16 NCHW via the real
/// `ImageProcessor`) and fill `out`. Returns 0 on success, a negative code
/// on failure. `out` must be a valid pointer.
///
/// # Safety
///
/// `out` must point to a writable `EdgefirstAndroidBench`.
#[no_mangle]
#[cfg(target_os = "android")]
pub unsafe extern "C" fn edgefirst_android_validation_bench(
    src_w: usize,
    src_h: usize,
    dst_size: usize,
    warmup: usize,
    iters: usize,
    out: *mut EdgefirstAndroidBench,
) -> i32 {
    if out.is_null() {
        return device::ERR_INVALID_ARG;
    }
    match device::bench_f16_planar(src_w, src_h, dst_size, warmup, iters) {
        Ok(b) => {
            unsafe {
                (*out).warmup_median_us = b.warmup_median_us;
                (*out).median_us = b.median_us;
                (*out).p95_us = b.p95_us;
                (*out).cache_miss_delta = b.cache_miss_delta;
            }
            device::OK
        }
        Err(code) => code,
    }
}
