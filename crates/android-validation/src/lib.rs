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
//!    exercise the REAL `ImageProcessor` (not probe code) — GL-vs-CPU
//!    correctness oracles and a convert benchmark with an import-cache
//!    steady-state gate. The hal-mobile Android app links this staticlib
//!    and calls these through JNI, running them on the Galaxy S26 Ultra
//!    pool (see `hal-mobile/ASSESSMENT-ANDROID.md` for the Phase-1
//!    numbers these gate against).
//!
//! The validation cells fail LOUDLY when the zero-copy path is not
//! actually in play (source or destination not AHardwareBuffer-backed,
//! cache statistics unreadable): a harness whose job is validating
//! zero-copy must never report green because it silently fell back to a
//! copying path.
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
// Android logcat logger (Android only).
// ---------------------------------------------------------------------------

/// Minimal `log::Log` → Android logcat bridge (`__android_log_write` from
/// the NDK's liblog). Without an installed logger every `log::error!` in
/// the HAL and in these entry points is a silent no-op, which would make
/// Device Farm failures undebuggable. The validation entry points install
/// it lazily; embedding apps may also call
/// [`edgefirst_android_validation_init_logging`] explicitly (idempotent).
#[cfg(target_os = "android")]
mod alog {
    use std::ffi::CString;
    use std::sync::Once;

    #[link(name = "log")]
    extern "C" {
        fn __android_log_write(
            prio: i32,
            tag: *const std::os::raw::c_char,
            text: *const std::os::raw::c_char,
        ) -> i32;
    }

    struct LogcatLogger;

    impl log::Log for LogcatLogger {
        fn enabled(&self, _metadata: &log::Metadata) -> bool {
            true
        }

        fn log(&self, record: &log::Record) {
            // android_LogPriority: VERBOSE=2, DEBUG=3, INFO=4, WARN=5, ERROR=6.
            let prio = match record.level() {
                log::Level::Error => 6,
                log::Level::Warn => 5,
                log::Level::Info => 4,
                log::Level::Debug => 3,
                log::Level::Trace => 2,
            };
            let tag = c"edgefirst-hal";
            // NUL bytes inside the message would truncate CString::new —
            // replace rather than drop the record.
            let text = format!("{}", record.args()).replace('\0', "\\0");
            if let Ok(text) = CString::new(text) {
                // SAFETY: both pointers are valid NUL-terminated C strings
                // for the duration of the call.
                unsafe { __android_log_write(prio, tag.as_ptr(), text.as_ptr()) };
            }
        }

        fn flush(&self) {}
    }

    static INIT: Once = Once::new();

    pub(crate) fn init() {
        INIT.call_once(|| {
            // Ignore the error: the embedding app may have installed its
            // own logger already, which is exactly what we want.
            if log::set_logger(&LogcatLogger).is_ok() {
                log::set_max_level(log::LevelFilter::Debug);
            }
        });
    }
}

/// Install the crate's logcat logger (idempotent; a no-op if the embedding
/// app already installed one). The validation entry points call this
/// themselves — exposed for apps that want HAL logs before the first call.
#[no_mangle]
#[cfg(target_os = "android")]
pub extern "C" fn edgefirst_android_validation_init_logging() {
    alog::init();
}

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
    /// failure (details on logcat — the entry points install a logger).
    pub const OK: i32 = 0;
    pub const ERR_GL_UNAVAILABLE: i32 = -1;
    pub const ERR_ALLOC: i32 = -2;
    pub const ERR_CONVERT: i32 = -3;
    pub const ERR_MISMATCH: i32 = -4;
    pub const ERR_MAP: i32 = -5;
    pub const ERR_INVALID_ARG: i32 = -6;
    /// A tensor that MUST be AHardwareBuffer-backed for the cell to mean
    /// anything is not — the zero-copy path silently fell back. Reported
    /// loudly instead of letting the cell pass while measuring a copy path.
    pub const ERR_NOT_ZERO_COPY: i32 = -7;
    /// The import-cache statistics could not be read; the steady-state
    /// miss gate would be vacuous, so the bench fails instead.
    pub const ERR_STATS: i32 = -8;

    /// Upper bound on warmup/iteration counts — far above any sane bench
    /// configuration; guards the `Vec` pre-allocations against a garbage
    /// argument aborting across the FFI boundary.
    const MAX_BENCH_COUNT: usize = 100_000;

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

    /// The tensor must be AHardwareBuffer-backed or the cell is not
    /// validating the zero-copy path at all.
    fn require_ahb(t: &TensorDyn, role: &str) -> Result<(), i32> {
        if t.hardware_buffer_ptr().is_none() {
            log::error!(
                "android-validation: {role} tensor is not AHardwareBuffer-backed — \
                 the zero-copy path is not in play (allocation fell back?)"
            );
            return Err(ERR_NOT_ZERO_COPY);
        }
        Ok(())
    }

    /// Deterministic gradient source (distinct R/G/B functions defeat both
    /// solid-fill and R↔B-swap false negatives) on an AHardwareBuffer.
    /// Fails loudly if the buffer is not AHB-backed — no silent Mem
    /// fallback (see module docs).
    fn make_gradient_src(w: usize, h: usize) -> Result<TensorDyn, i32> {
        let src =
            Tensor::<u8>::image(w, h, PixelFormat::Rgba, Some(TensorMemory::Dma)).map_err(|e| {
                log::error!("android-validation: AHardwareBuffer src alloc {w}x{h}: {e:?}");
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
        let src: TensorDyn = src.into();
        require_ahb(&src, "source")?;
        Ok(src)
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

    fn gl_dst(
        gl: &ImageProcessor,
        size: usize,
        format: PixelFormat,
        dtype: DType,
    ) -> Result<TensorDyn, i32> {
        let dst = gl
            .create_image(size, size, format, dtype, None)
            .map_err(|e| {
                log::error!("android-validation: GL dst alloc ({format:?}/{dtype:?}): {e:?}");
                ERR_ALLOC
            })?;
        // The render destination must be zero-copy too — a PBO/Mem dst
        // would silently bench/verify the readback path instead.
        require_ahb(&dst, "destination")?;
        Ok(dst)
    }

    fn cpu_dst(size: usize, format: PixelFormat, dtype: DType) -> Result<TensorDyn, i32> {
        TensorDyn::image(size, size, format, dtype, Some(TensorMemory::Mem)).map_err(|e| {
            log::error!("android-validation: CPU dst alloc ({format:?}/{dtype:?}): {e:?}");
            ERR_ALLOC
        })
    }

    /// Row-walking comparison shared by the typed compares. gralloc may
    /// pad the GL destination's row pitch (recorded on the tensor as
    /// `effective_row_stride`), in which case the mapped slice is the full
    /// padded buffer — comparing flat against the tight CPU reference
    /// would misalign after the first row. Walks `rows × row_elems`
    /// logical elements using each side's own stride.
    // Internal helper with two callers; the argument list mirrors the two
    // (slice, stride) sides plus geometry — a struct would only rename it.
    #[allow(clippy::too_many_arguments)]
    fn compare_rows<T: Copy>(
        sa: &[T],
        sb: &[T],
        stride_a_elems: usize,
        stride_b_elems: usize,
        rows: usize,
        row_elems: usize,
        cell: &str,
        mut diff: impl FnMut(T, T) -> f32,
        tolerance: f32,
    ) -> i32 {
        let need_a = (rows - 1) * stride_a_elems + row_elems;
        let need_b = (rows - 1) * stride_b_elems + row_elems;
        if rows == 0 || sa.len() < need_a || sb.len() < need_b {
            log::error!(
                "android-validation: {cell}: mapped sizes too small \
                 (a={} need {need_a}, b={} need {need_b})",
                sa.len(),
                sb.len()
            );
            return ERR_MISMATCH;
        }
        let mut max_diff = 0f32;
        for r in 0..rows {
            let ra = &sa[r * stride_a_elems..r * stride_a_elems + row_elems];
            let rb = &sb[r * stride_b_elems..r * stride_b_elems + row_elems];
            for (va, vb) in ra.iter().zip(rb.iter()) {
                let d = diff(*va, *vb);
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        if max_diff > tolerance {
            log::error!("android-validation: {cell}: GL vs CPU max diff {max_diff} > {tolerance}");
            return ERR_MISMATCH;
        }
        log::info!("android-validation: {cell}: max diff {max_diff} (tol {tolerance}) — PASS");
        OK
    }

    /// Compare two PlanarRgb F16 destinations of `dst_size`² — logical
    /// geometry: `3 × dst_size` rows of `dst_size` f16 elements.
    fn compare_f16(a: &TensorDyn, b: &TensorDyn, dst_size: usize, cell: &str) -> i32 {
        let (Some(ta), Some(tb)) = (a.as_f16(), b.as_f16()) else {
            log::error!("android-validation: {cell}: dst tensors are not F16");
            return ERR_MISMATCH;
        };
        let elem = 2; // sizeof(f16)
        let row_elems = dst_size;
        let stride_a = ta.effective_row_stride().map_or(row_elems, |s| s / elem);
        let stride_b = tb.effective_row_stride().map_or(row_elems, |s| s / elem);
        let (Ok(ma), Ok(mb)) = (ta.map(), tb.map()) else {
            log::error!("android-validation: {cell}: dst map failed");
            return ERR_MAP;
        };
        // Normalized [0,1] outputs; GPU linear filtering + f16 rounding vs
        // the CPU path justifies a small tolerance (the suite uses 4/255 on
        // u8 paths — mirror it in float space).
        compare_rows(
            ma.as_slice(),
            mb.as_slice(),
            stride_a,
            stride_b,
            3 * dst_size,
            row_elems,
            cell,
            |va: hal::tensor::f16, vb: hal::tensor::f16| (va.to_f32() - vb.to_f32()).abs(),
            4.0 / 255.0,
        )
    }

    /// Compare two packed-u8 destinations of `dst_size`² with `channels`
    /// bytes per pixel — logical geometry: `dst_size` rows of
    /// `dst_size × channels` bytes.
    fn compare_u8(
        a: &TensorDyn,
        b: &TensorDyn,
        dst_size: usize,
        channels: usize,
        cell: &str,
    ) -> i32 {
        let (Some(ta), Some(tb)) = (a.as_u8(), b.as_u8()) else {
            log::error!("android-validation: {cell}: dst tensors are not U8");
            return ERR_MISMATCH;
        };
        let row_elems = dst_size * channels;
        let stride_a = ta.effective_row_stride().unwrap_or(row_elems);
        let stride_b = tb.effective_row_stride().unwrap_or(row_elems);
        let (Ok(ma), Ok(mb)) = (ta.map(), tb.map()) else {
            log::error!("android-validation: {cell}: dst map failed");
            return ERR_MAP;
        };
        compare_rows(
            ma.as_slice(),
            mb.as_slice(),
            stride_a,
            stride_b,
            dst_size,
            row_elems,
            cell,
            |va: u8, vb: u8| (va as i16 - vb as i16).unsigned_abs() as f32,
            4.0,
        )
    }

    /// Cell 1 — GL-vs-CPU oracle on the F16 planar-RGB model-input path
    /// (RGBA8 AHB source → RGBA16F-packed AHB destination): the same
    /// gradient source converted by both backends must agree within a
    /// small tolerance (the established pattern of the HAL's own GL test
    /// suite — no letterbox-geometry assumptions to drift).
    fn verify_f16_planar(
        gl: &mut ImageProcessor,
        cpu: &mut ImageProcessor,
        src: &TensorDyn,
        dst_size: usize,
    ) -> i32 {
        let mut dst_gl = match gl_dst(gl, dst_size, PixelFormat::PlanarRgb, DType::F16) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let mut dst_cpu = match cpu_dst(dst_size, PixelFormat::PlanarRgb, DType::F16) {
            Ok(d) => d,
            Err(c) => return c,
        };
        if let Err(c) = convert_with(gl, src, &mut dst_gl) {
            return c;
        }
        if let Err(c) = convert_with(cpu, src, &mut dst_cpu) {
            return c;
        }
        compare_f16(&dst_gl, &dst_cpu, dst_size, "verify_f16_planar")
    }

    /// Cell 2 — GL-vs-CPU oracle on the RGBA8 → RGBA8 resize path (the
    /// other probe-validated zero-copy import: TEXTURE_2D sampling of an
    /// RGBA8 AHardwareBuffer).
    fn verify_rgba8(
        gl: &mut ImageProcessor,
        cpu: &mut ImageProcessor,
        src: &TensorDyn,
        dst_size: usize,
    ) -> i32 {
        let mut dst_gl = match gl_dst(gl, dst_size, PixelFormat::Rgba, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let mut dst_cpu = match cpu_dst(dst_size, PixelFormat::Rgba, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        if let Err(c) = convert_with(gl, src, &mut dst_gl) {
            return c;
        }
        if let Err(c) = convert_with(cpu, src, &mut dst_cpu) {
            return c;
        }
        compare_u8(&dst_gl, &dst_cpu, dst_size, 4, "verify_rgba8")
    }

    /// Cell 4 — GL-vs-CPU oracle on the packed RGB u8 INT8-NPU output path
    /// (RGBA8 AHB source → RGB-in-RGBA8888 AHB destination via the
    /// two-pass packed-RGB shader). The destination must be zero-copy — a
    /// Mem fallback would silently verify the CPU repack instead of the
    /// `packed_rgb888_layout` surface this cell exists to prove.
    fn verify_rgb8_packed(
        gl: &mut ImageProcessor,
        cpu: &mut ImageProcessor,
        src: &TensorDyn,
        dst_size: usize,
    ) -> i32 {
        let mut dst_gl = match gl_dst(gl, dst_size, PixelFormat::Rgb, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let mut dst_cpu = match cpu_dst(dst_size, PixelFormat::Rgb, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        if let Err(c) = convert_with(gl, src, &mut dst_gl) {
            return c;
        }
        if let Err(c) = convert_with(cpu, src, &mut dst_cpu) {
            return c;
        }
        compare_u8(&dst_gl, &dst_cpu, dst_size, 3, "verify_rgb8_packed")
    }

    /// Cell 3 — the external-import route: wrap the source buffer's raw
    /// AHardwareBuffer pointer via `TensorDyn::from_hardware_buffer` (the
    /// CameraX/JNI entry point) and confirm the wrapped tensor converts
    /// identically to the CPU reference. Exercises acquire/describe,
    /// footprint validation, and the imported-buffer GL route.
    fn verify_wrapped_import(
        gl: &mut ImageProcessor,
        cpu: &mut ImageProcessor,
        src: &TensorDyn,
        src_w: usize,
        src_h: usize,
        dst_size: usize,
    ) -> i32 {
        let Some(ptr) = src.hardware_buffer_ptr() else {
            return ERR_NOT_ZERO_COPY; // unreachable: make_gradient_src checked
        };
        // SAFETY: `ptr` is borrowed from `src`, which outlives `wrapped`
        // in this scope; from_hardware_buffer acquires its own reference.
        let mut wrapped = match unsafe {
            TensorDyn::from_hardware_buffer(ptr, &[src_h, src_w, 4], DType::U8, Some("wrapped"))
        } {
            Ok(t) => t,
            Err(e) => {
                log::error!("android-validation: from_hardware_buffer: {e:?}");
                return ERR_ALLOC;
            }
        };
        // A wrapped buffer carries a shape but no image metadata — exactly
        // what a CameraX/JNI consumer must also do: declare the pixel
        // format (and, when padded, the row stride) before convert().
        if let Err(e) = wrapped.configure_image(src_w, src_h, PixelFormat::Rgba) {
            log::error!("android-validation: configure_image(wrapped): {e:?}");
            return ERR_ALLOC;
        }
        if let Some(stride) = src.as_u8().and_then(|t| t.row_stride()) {
            if let Err(e) = wrapped.set_row_stride(stride) {
                log::error!("android-validation: set_row_stride(wrapped): {e:?}");
                return ERR_ALLOC;
            }
        }
        let mut dst_gl = match gl_dst(gl, dst_size, PixelFormat::Rgba, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let mut dst_cpu = match cpu_dst(dst_size, PixelFormat::Rgba, DType::U8) {
            Ok(d) => d,
            Err(c) => return c,
        };
        if let Err(c) = convert_with(gl, &wrapped, &mut dst_gl) {
            return c;
        }
        if let Err(c) = convert_with(cpu, src, &mut dst_cpu) {
            return c;
        }
        compare_u8(&dst_gl, &dst_cpu, dst_size, 4, "verify_wrapped_import")
    }

    /// Cell 5 — the GL→NPU fence handoff: `convert_with_fence` returns a
    /// sync-fence fd instead of blocking (every S26-class device exposes
    /// `EGL_ANDROID_native_fence_sync`; a blocking `None` fallback is
    /// logged loudly but not failed — emulators may lack the extension).
    /// After the fence signals, the output must match the CPU reference
    /// exactly like the blocking cell — proving the fence really guards
    /// the render's completion.
    fn verify_fence_handoff(
        gl: &mut ImageProcessor,
        cpu: &mut ImageProcessor,
        src: &TensorDyn,
        dst_size: usize,
    ) -> i32 {
        use std::os::fd::AsRawFd as _;
        let mut dst_gl = match gl_dst(gl, dst_size, PixelFormat::PlanarRgb, DType::F16) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let mut dst_cpu = match cpu_dst(dst_size, PixelFormat::PlanarRgb, DType::F16) {
            Ok(d) => d,
            Err(c) => return c,
        };
        let submit_start = std::time::Instant::now();
        let fence = match gl.convert_with_fence(
            src,
            &mut dst_gl,
            Rotation::None,
            Flip::None,
            Crop::default(),
        ) {
            Ok(f) => f,
            Err(e) => {
                log::error!("android-validation: convert_with_fence: {e:?}");
                return ERR_CONVERT;
            }
        };
        let submit_us = submit_start.elapsed().as_micros();
        match fence {
            Some(fd) => {
                let wait_start = std::time::Instant::now();
                let mut pfd = libc::pollfd {
                    fd: fd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                // SAFETY: pfd is a valid pollfd over an owned live fd.
                let rc = unsafe { libc::poll(&mut pfd, 1, 1000) };
                if rc != 1 {
                    log::error!(
                        "android-validation: verify_fence_handoff: poll(fence) \
                         returned {rc} (timeout/error) — fence never signaled"
                    );
                    return ERR_CONVERT;
                }
                log::info!(
                    "android-validation: verify_fence_handoff: submit {submit_us}µs, \
                     fence wait {}µs",
                    wait_start.elapsed().as_micros()
                );
            }
            None => {
                // Loud but not fatal: the blocking contract is still
                // correct, and emulator lanes may lack the extension.
                log::warn!(
                    "android-validation: verify_fence_handoff: no native fence on \
                     this device (blocking fallback, {submit_us}µs) — \
                     EGL_ANDROID_native_fence_sync missing?"
                );
            }
        }
        if let Err(c) = convert_with(cpu, src, &mut dst_cpu) {
            return c;
        }
        compare_f16(&dst_gl, &dst_cpu, dst_size, "verify_fence_handoff")
    }

    /// Run all verification cells; returns the first failure.
    pub fn verify(src_w: usize, src_h: usize, dst_size: usize) -> i32 {
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

        let rc = verify_f16_planar(&mut gl, &mut cpu, &src, dst_size);
        if rc != OK {
            return rc;
        }
        let rc = verify_rgba8(&mut gl, &mut cpu, &src, dst_size);
        if rc != OK {
            return rc;
        }
        let rc = verify_rgb8_packed(&mut gl, &mut cpu, &src, dst_size);
        if rc != OK {
            return rc;
        }
        let rc = verify_fence_handoff(&mut gl, &mut cpu, &src, dst_size);
        if rc != OK {
            return rc;
        }
        verify_wrapped_import(&mut gl, &mut cpu, &src, src_w, src_h, dst_size)
    }

    /// One benchmark cell: repeated full `convert()` (letterbox, normalize,
    /// and F16 NCHW pack, including the sync) with a warmup phase,
    /// reporting the cold first convert, medians/p95, and the import-cache
    /// miss delta across the steady-state phase.
    ///
    /// With `fresh_dst = false` (the pooled/steady-state cell) the miss
    /// delta MUST be zero — a nonzero delta means the zero-copy import is
    /// re-created per frame. With `fresh_dst = true` a NEW destination
    /// tensor is allocated per iteration (allocation excluded from the
    /// timed window), making the re-import churn cost observable: the
    /// expected delta is `iters`, and the median gap vs the pooled cell is
    /// the per-frame price of not pooling buffers.
    pub struct BenchOut {
        pub first_convert_us: u64,
        pub warmup_median_us: u64,
        pub median_us: u64,
        pub p95_us: u64,
        pub cache_miss_delta: u64,
    }

    /// Read the GL source-feed telemetry, or fail the bench (an unreadable
    /// counter would make the zero-upload gate vacuous).
    fn convert_feed_stats(gl: &ImageProcessor) -> Result<hal::image::ConvertStats, i32> {
        gl.opengl.as_ref().ok_or(ERR_GL_UNAVAILABLE).and_then(|g| {
            g.convert_stats().map_err(|e| {
                log::error!("android-validation: convert_stats: {e:?}");
                ERR_STATS
            })
        })
    }

    /// Read the GL import-cache miss counter, or fail the bench: an
    /// unreadable counter would make the steady-state gate vacuous.
    fn cache_misses(gl: &ImageProcessor) -> Result<u64, i32> {
        gl.opengl
            .as_ref()
            .ok_or(ERR_GL_UNAVAILABLE)
            .and_then(|g| {
                g.egl_cache_stats().map_err(|e| {
                    log::error!("android-validation: egl_cache_stats: {e:?}");
                    ERR_STATS
                })
            })
            .map(|s| s.total_misses())
    }

    pub fn bench_f16_planar(
        src_w: usize,
        src_h: usize,
        dst_size: usize,
        warmup: usize,
        iters: usize,
        fresh_dst: bool,
    ) -> Result<BenchOut, i32> {
        // warmup >= 1 is required: the cold import must land in the warmup
        // phase or the steady-state miss gate is false-negative by design.
        if iters == 0 || warmup == 0 || iters > MAX_BENCH_COUNT || warmup > MAX_BENCH_COUNT {
            log::error!(
                "android-validation: bench args out of range \
                 (warmup={warmup}, iters={iters}, max={MAX_BENCH_COUNT})"
            );
            return Err(ERR_INVALID_ARG);
        }
        let mut gl = gl_processor()?;
        let src = make_gradient_src(src_w, src_h)?;
        let mut dst = gl_dst(&gl, dst_size, PixelFormat::PlanarRgb, DType::F16)?;

        let mut warm: Vec<u64> = Vec::with_capacity(warmup);
        for _ in 0..warmup {
            let t0 = std::time::Instant::now();
            convert_with(&mut gl, &src, &mut dst)?;
            warm.push(t0.elapsed().as_micros() as u64);
        }
        let first_convert_us = warm[0];

        // Steady state begins. Pooled cell: the import cache must not miss
        // again. Fresh-dst cell: every iteration re-imports by design.
        let misses_before = cache_misses(&gl)?;
        let feed_before = convert_feed_stats(&gl)?;

        let mut times: Vec<u64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            if fresh_dst {
                // Allocation is deliberately OUTSIDE the timed window: the
                // cell isolates the re-import cost, not gralloc.
                dst = gl_dst(&gl, dst_size, PixelFormat::PlanarRgb, DType::F16)?;
            }
            let t0 = std::time::Instant::now();
            convert_with(&mut gl, &src, &mut dst)?;
            times.push(t0.elapsed().as_micros() as u64);
        }

        let misses_after = cache_misses(&gl)?;
        let feed_after = convert_feed_stats(&gl)?;
        // True-zero-copy gate: every steady-state frame must feed the source
        // by import — a single CPU map+upload means the float source import
        // silently degraded (the exact regression this suite exists to
        // catch; see ConvertStats).
        let upload_delta = feed_after.src_uploads - feed_before.src_uploads;
        if upload_delta != 0 {
            log::error!(
                "android-validation: bench: {upload_delta} steady-state frames fed the \
                 source via CPU upload (before={feed_before:?} after={feed_after:?}) — \
                 the zero-copy source import is not in play"
            );
            return Err(ERR_NOT_ZERO_COPY);
        }

        times.sort_unstable();
        warm.sort_unstable();
        let median = times[times.len() / 2];
        let p95 = times[((times.len() as f64 * 0.95).ceil() as usize).clamp(1, times.len()) - 1];
        let warm_median = warm[warm.len() / 2];

        Ok(BenchOut {
            first_convert_us,
            warmup_median_us: warm_median,
            median_us: median,
            p95_us: p95,
            cache_miss_delta: misses_after.saturating_sub(misses_before),
        })
    }
}

/// C-ABI benchmark result (see `device::bench_f16_planar`).
#[repr(C)]
pub struct EdgefirstAndroidBench {
    /// The very first (cold) convert, including the one-time EGLImage
    /// import — the first-frame-latency signal.
    pub first_convert_us: u64,
    pub warmup_median_us: u64,
    pub median_us: u64,
    pub p95_us: u64,
    /// Import-cache misses observed during the steady-state phase. Pooled
    /// cell (`fresh_dst = false`): must be 0 (nonzero means the zero-copy
    /// import is re-created per frame). Fresh-dst cell: expected == iters.
    pub cache_miss_delta: u64,
}

/// Run the on-device correctness cells against the real `ImageProcessor`:
/// GL-vs-CPU on the F16 planar model-input path, on the RGBA8 resize path,
/// and through the `from_hardware_buffer` external-import route. Returns 0
/// on pass, a negative code on failure (see the `device` module constants;
/// details are logged to logcat, tag `edgefirst-hal`).
#[no_mangle]
#[cfg(target_os = "android")]
pub extern "C" fn edgefirst_android_validation_verify(
    src_w: usize,
    src_h: usize,
    dst_size: usize,
) -> i32 {
    alog::init();
    device::verify(src_w, src_h, dst_size)
}

/// Run one benchmark cell (letterbox+normalize → F16 NCHW via the real
/// `ImageProcessor`) and fill `out`. `fresh_dst = false` is the pooled
/// steady-state cell (miss gate == 0); `fresh_dst = true` allocates a new
/// destination per iteration to expose the re-import churn cost. Returns 0
/// on success, a negative code on failure. Requires `warmup >= 1` and
/// `iters >= 1`.
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
    fresh_dst: bool,
    out: *mut EdgefirstAndroidBench,
) -> i32 {
    alog::init();
    if out.is_null() {
        return device::ERR_INVALID_ARG;
    }
    match device::bench_f16_planar(src_w, src_h, dst_size, warmup, iters, fresh_dst) {
        Ok(b) => {
            unsafe {
                (*out).first_convert_us = b.first_convert_us;
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

#[cfg(test)]
mod tests {
    /// Host-side smoke test: the closure function must be callable on any
    /// platform (Dma allocation and GL bring-up are allowed to fail — the
    /// point is that the full HAL closure links and runs).
    #[test]
    fn force_closure_runs_on_host() {
        super::edgefirst_android_validation_force_closure();
    }
}
