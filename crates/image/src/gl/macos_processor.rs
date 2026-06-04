// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS GL image processor backed by ANGLE + IOSurface.
//!
//! Mirrors the role of `GLProcessorThreaded` on Linux — the parallel
//! Linux implementation lives in `gl/threaded.rs` and is structurally
//! more elaborate because of vendor-driver thread-safety constraints
//! (Vivante galcore in particular). ANGLE's Metal backend is
//! thread-safe enough that we run GL inline under a process-wide mutex
//! instead of through a dedicated thread + command channel.
//!
//! Format coverage in this initial implementation:
//!   * YUYV → RGBA — full shader-based BT.601 full-range conversion (interim
//!     colorimetry stop-gap; see crates/image/ARCHITECTURE.md "Colorimetry")
//!
//! Other format pairs and the mask-rendering / decoder paths return
//! `NotImplemented` and fall back to the CPU backend, matching the
//! contract the Linux backend uses for unsupported combinations on a
//! given GPU driver.
//!
//! ## Resource model
//!
//! The ANGLE EGL display + context + dummy pbuffer are *process-global*,
//! shared via `SHARED_DISPLAY` on first construction. The Linux backend
//! makes the same choice for the same reason — `eglTerminate` is
//! ref-counted but never safely terminable mid-process, and ANGLE's
//! Metal device is a singleton. Per-instance state is limited to the
//! cached shader program, VBO/VAO/FBO, transient texture handles, and
//! the IOSurface→pbuffer cache.
//!
//! GL/EGL calls are serialised behind a single static `GL_MUTEX` so
//! concurrent `MacosGlProcessor` instances do not race on the shared
//! context's current-thread state.
//!
//! See `crates/image/src/gl/platform/macos.rs` for the platform helpers
//! this processor builds on, and `crates/image/src/gl/iosurface_import.rs`
//! for the IOSurface allocation + EGL pbuffer attribute setup.

#![cfg(target_os = "macos")]

use super::iosurface_import;
use super::platform::macos::MacosPlatform;
// `MacosPlatform::{load_egl_lib, create_display}` are the two macOS-specific
// helpers; everything else (pbuffer creation, texture binding, FBO setup,
// shader compilation) is inline here. See platform/mod.rs for the seam
// rationale.
use super::Egl;
use crate::{Crop, Error, Flip, ImageProcessorTrait, MaskOverlay, Result, Rotation};
use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMemory};
use khronos_egl as egl;
use log::debug;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------
// EGL constants reused across the macOS path. The "production" constants in
// `super::iosurface_import` cover the IOSurface-pbuffer attribute set; these
// are the additional constants needed at MacosGlProcessor::new time.
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_BACK_BUFFER: i32 = 0x3084;

// ---------------------------------------------------------------------------
// Shaders. YUYV-as-GL_RG sampling: each source texel is (Y, C) where C
// alternates U/V every other column. We sample the current and partner
// texel to recover both chroma values for each output pixel, then apply
// the BT.601 full-range matrix (interim colorimetry stop-gap).
//
// The shader matches the spike at `spikes/angle_iosurface/`. Bit-near-
// exact (≤1 LSB) match to the CPU scalar reference was validated there.
// ---------------------------------------------------------------------------

const VERTEX_SHADER: &str = r#"#version 300 es
precision mediump float;
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv_in;
out vec2 v_uv;
void main() {
    v_uv = uv_in;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"#;

const YUYV_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D src;
uniform vec2 src_size;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 texel = vec2(1.0) / src_size;
    vec2 col = floor(v_uv * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;

    vec4 self_rg = texture(src, self_uv);
    vec4 pair_rg = texture(src, pair_uv);
    float y = self_rg.r;
    float u, v;
    if (even) { u = self_rg.g; v = pair_rg.g; }
    else      { v = self_rg.g; u = pair_rg.g; }

    // INTERIM COLORIMETRY STOP-GAP (see crates/image/ARCHITECTURE.md
    // "Colorimetry"): BT.601 full-range (JFIF) to match the codec and the Linux
    // backends until per-source colorimetry tagging lands. Full range → luma is
    // used directly (no 16/235 expansion); BT.601 coefficients.
    float yp = y;
    float up = u - 128.0/255.0;
    float vp = v - 128.0/255.0;
    float r = clamp(yp + 1.402 * vp, 0.0, 1.0);
    float g = clamp(yp - 0.344 * up - 0.714 * vp, 0.0, 1.0);
    float b = clamp(yp + 1.772 * up, 0.0, 1.0);
    frag = vec4(r, g, b, 1.0);
}
"#;

// GREY (single-channel R8 / `L008` IOSurface) → RGBA. This is also the probe
// that proves ANGLE's Metal IOSurface-client-buffer path accepts an
// `L008`→`GL_RED` binding; the semi-planar YUV shaders below build on the same
// R8 source binding. Portable GLES 3.0 (`sampler2D` over a GL_RED texture).
const GREY_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D src;
in vec2 v_uv;
out vec4 frag;
void main() {
    float y = texture(src, v_uv).r;
    frag = vec4(y, y, y, 1.0);
}
"#;

// Semi-planar YUV (NV12/NV16/NV24) → RGBA, sampling the contiguous combined-
// plane buffer as ONE R8 (`L008`/GL_RED) texture of `[total_h, even_width]`.
// The shader computes the Y and interleaved-UV texel positions itself via
// `texelFetch`, parameterised by uniforms so one program serves all three
// subsamplings (and is portable to Linux DMA-BUF / embedded GLES — no
// platform-specific multi-plane sampler):
//   * `img_size`      logical (W, H); Y plane occupies rows [0, H).
//   * `tex_width`     R8 texture width == the buffer's physical row pitch
//                     (`bytes_per_row`). The semi-planar surface is allocated
//                     so its IOSurface width equals this pitch, so every byte
//                     of every row is addressable by `texelFetch`.
//   * `chroma_shift`  (cx, cy) right-shifts on (x, y): NV12 (1,1), NV16 (1,0),
//                     NV24 (0,0).
//   * `uv_row_bytes`  bytes the UV plane advances per chroma row: `stride` for
//                     NV12/NV16 (W/2 pairs fit in one row), `2*stride` for NV24
//                     (W pairs == two grid rows).
// A linear UV byte offset is converted to a 2D texel so NV24's 2-row-per-
// chroma-line layout is handled with no special case. The byte offsets match
// the codec's `row * grid_row_stride + col` writer exactly. The YUV→RGB matrix
// + range come from the per-tensor colorimetry uniforms (y_offset/y_scale/
// c_vr/c_ug/c_vg/c_ub), matching the Linux Path B NV shader and CPU kernels.
const NV_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D src;
uniform ivec2 img_size;
uniform int tex_width;
uniform ivec2 chroma_shift;
uniform int uv_row_bytes;
// Per-tensor colorimetry (YUV→RGB matrix + range), set per-convert from the
// source tensor's resolved colorimetry — mirrors the Linux Path B NV shader.
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec2 v_uv;
out vec4 frag;

float fetch_r(int b) {
    return texelFetch(src, ivec2(b % tex_width, b / tex_width), 0).r;
}

void main() {
    int w = img_size.x;
    int h = img_size.y;
    int x = clamp(int(v_uv.x * float(w)), 0, w - 1);
    int y = clamp(int(v_uv.y * float(h)), 0, h - 1);

    float yv = fetch_r(y * tex_width + x); // Y plane, rows [0, H)

    int ccol = x >> chroma_shift.x;
    int crow = y >> chroma_shift.y;
    int uv_base = h * tex_width;           // UV plane byte offset
    int cb = uv_base + crow * uv_row_bytes + ccol * 2;
    float u = fetch_r(cb);
    float v = fetch_r(cb + 1);

    float yp = (yv - y_offset) * y_scale;
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    float r = clamp(yp + c_vr * vp, 0.0, 1.0);
    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);
    float b = clamp(yp + c_ub * up, 0.0, 1.0);
    frag = vec4(r, g, b, 1.0);
}
"#;

/// RGBA8 source → packed RGBA16F PlanarRgb F16 destination, one draw
/// call. F32 destinations are intentionally not supported — see the
/// crate-level docs and `image_iosurface_layout` for the ANGLE
/// constraint.
///
/// The destination IOSurface is sized `(W/4, 3*H)` RGBA16F pixels.
/// Each output pixel holds 4 contiguous half-floats of the planar
/// `[3, H, W]` byte stream — same byte layout as a (hypothetical)
/// R16F `(W, 3*H)` surface would have, but bound through ANGLE's
/// single supported float `(GL_HALF_FLOAT, GL_RGBA)` combination.
///
/// For output pixel `(ox, oy)`:
///   * `plane = oy / H` (0=R, 1=G, 2=B), `in_plane_y = oy % H`
///   * The 4 packed elements correspond to logical
///     `in_plane_x = ox*4 + 0..3` — all in the same plane and row.
///   * Each element samples the RGBA8 source at the proper plane
///     channel; pad with `pad_color[plane]` outside `dst_rect_px`.
///
/// The texture sampler returns RGBA8 values normalized to `[0, 1]`
/// already (GL_TEXTURE_2D with internal format RGBA8), so there's no
/// `/ 255.0` divide in the shader — the GL fixed-function texture
/// fetch performs the normalize for free. Resize is implicit in the
/// `src_rect_uv` mapping; `GL_LINEAR` filtering handles interpolation.
///
/// Width must be a multiple of 4 (enforced by HAL at IOSurface
/// allocation time).
///
/// Single source of truth lives in [`super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT`],
/// shared with the Linux PBO/DMA-BUF path (`shaders.rs`).
const RGBA8_TO_PLANAR_F16_PACKED_FRAGMENT: &str =
    super::shaders_common::PLANAR_RGB_F16_PACKED_FRAGMENT;

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table.
//
// `gls::load_with` populates global function pointers — exists once per
// process. We load via EGL's `eglGetProcAddress` so the symbols come
// from ANGLE's libGLESv2.dylib.
// ---------------------------------------------------------------------------

static GL_LOADED: OnceLock<()> = OnceLock::new();

fn load_gl_once(egl: &Egl) {
    GL_LOADED.get_or_init(|| {
        gls::load_with(|name| match egl.get_proc_address(name) {
            Some(ptr) => ptr as *const c_void,
            None => std::ptr::null(),
        });
    });
}

// ---------------------------------------------------------------------------
// Process-global ANGLE EGL display + context + dummy pbuffer.
//
// `eglTerminate` is ref-counted but never safely terminable mid-process
// (ANGLE's Metal device is a per-process singleton, and any in-flight GL
// command from any thread aborts when the display goes away). The Linux
// backend uses a `SharedEglDisplay` in `context.rs` for the same reason.
// Sharing here also avoids hammering `eglInitialize` from every
// `MacosGlProcessor::new()` call.
//
// `GL_MUTEX` serialises every `eglMakeCurrent` + GL call across all
// `MacosGlProcessor` instances — ANGLE's Metal backend is internally
// thread-safe enough that a single global mutex is the right granularity.
// Per-instance mutexes would race on the current-thread context state
// because the EGL context is shared.
// ---------------------------------------------------------------------------

/// All process-global EGL state. Use [`shared_display`] to access.
struct SharedAngleDisplay {
    /// Static-lifetime EGL handle. The actual ANGLE libEGL.dylib is
    /// leaked at first dlopen and never closed.
    egl: Egl,
    display: egl::Display,
    config: egl::Config,
    context: egl::Context,
    /// Tiny scratch surface kept alive so the context can be made
    /// current outside of a `convert` call (e.g. for shader compile,
    /// resource allocation, or `Drop`-time cleanup).
    dummy_pbuffer: egl::Surface,
    /// `GL_EXT_color_buffer_float` is exposed by this ANGLE/Metal
    /// configuration. Gates F32 destination tensors on the IOSurface
    /// render path.
    supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` is exposed by this
    /// ANGLE/Metal configuration. Gates F16 destination tensors on
    /// the IOSurface render path.
    supports_f16_color: bool,
}

// SAFETY: every member is either a leak'd static, an EGL handle (which
// the ANGLE driver synchronises internally), or a pointer to driver-
// owned state. Access is gated by GL_MUTEX.
unsafe impl Send for SharedAngleDisplay {}
unsafe impl Sync for SharedAngleDisplay {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedAngleDisplay, String>> = OnceLock::new();
static GL_MUTEX: Mutex<()> = Mutex::new(());

/// Acquire a reference to the process-global ANGLE display, initialising
/// it on first call. Subsequent calls return the same handle. The error
/// case is cached too — once ANGLE fails to load we don't keep retrying.
fn shared_display() -> Result<&'static SharedAngleDisplay> {
    SHARED_DISPLAY
        .get_or_init(|| init_shared_display().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|s| Error::Io(std::io::Error::other(s.clone())))
}

fn init_shared_display() -> Result<SharedAngleDisplay> {
    let _span =
        tracing::info_span!("image.gl_init", platform = "macos", backend = "iosurface",).entered();

    // 1. Load ANGLE libEGL and bring up an EGL instance.
    let egl_lib = MacosPlatform::load_egl_lib()
        .map_err(|e| Error::Io(std::io::Error::other(format!("ANGLE libEGL: {e}"))))?;
    let egl: Egl = unsafe {
        khronos_egl::Instance::<
            khronos_egl::Dynamic<&'static libloading::Library, khronos_egl::EGL1_4>,
        >::load_required_from(egl_lib)
    }
    .map_err(|e| Error::Io(std::io::Error::other(format!("EGL load: {e:?}"))))?;

    // 2. Metal-backed display from MacosPlatform.
    let display = MacosPlatform::create_display(&egl)?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    debug!("MacosGlProcessor: EGL {maj}.{min} initialised via ANGLE (process-global)");

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. Choose an EGL config that supports GLES 3 + PBUFFER +
    //    EGL_BIND_TO_TEXTURE_TARGET_ANGLE = EGL_TEXTURE_2D.
    //
    // 8-bit RGBA color sizes are explicit but the config doesn't
    // restrict half-float IOSurface texture binding — ANGLE's
    // `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` overrides at pbuffer
    // creation time. Verified by reading ANGLE's
    // `EGLIOSurfaceClientBufferTest::RenderToRGBA16FIOSurface` which
    // also binds u8 surfaces in the same context.
    let cfg_attribs = [
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        iosurface_import::EGL_BIND_TO_TEXTURE_TARGET_ANGLE,
        0x305F, // EGL_TEXTURE_2D
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &cfg_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglChooseConfig: {e:?}"))))?
        .ok_or_else(|| {
            Error::NotSupported("no EGL config with GLES3+PBUFFER+TEXTURE_2D bind".into())
        })?;

    // 4. GLES3 context.
    let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = egl
        .create_context(display, config, None, &ctx_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglCreateContext: {e:?}"))))?;

    // 5. Dummy pbuffer for context-current bring-up.
    let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
    let dummy_pbuffer = egl
        .create_pbuffer_surface(display, config, &dummy_attribs)
        .map_err(|e| {
            // Clean up the context we just created before bailing.
            let _ = egl.destroy_context(display, context);
            Error::Io(std::io::Error::other(format!(
                "eglCreatePbufferSurface(dummy): {e:?}"
            )))
        })?;

    // 6. Load GL function pointers via the now-initialised display.
    //    Make-current is required for some drivers to expose GLES symbols.
    if let Err(e) = egl.make_current(
        display,
        Some(dummy_pbuffer),
        Some(dummy_pbuffer),
        Some(context),
    ) {
        let _ = egl.destroy_surface(display, dummy_pbuffer);
        let _ = egl.destroy_context(display, context);
        return Err(Error::Io(std::io::Error::other(format!(
            "eglMakeCurrent(dummy): {e:?}"
        ))));
    }
    load_gl_once(&egl);

    // Probe the float-color-buffer extensions while the context is
    // still current. ANGLE's Metal backend exposes both extensions on
    // Apple Silicon + recent ANGLE bundles; on older configurations
    // either or both may be missing — consumers fall back per dtype.
    let extensions = unsafe {
        let ptr = gls::gl::GetString(gls::gl::EXTENSIONS);
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        }
    };
    let supports_f32_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_float");
    let supports_f16_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_half_float");
    debug!(
        "MacosGlProcessor: GL_EXT_color_buffer_float={supports_f32_color}, \
         GL_EXT_color_buffer_half_float={supports_f16_color}"
    );

    let _ = egl.make_current(display, None, None, None);

    Ok(SharedAngleDisplay {
        egl,
        display,
        config,
        context,
        dummy_pbuffer,
        supports_f32_color,
        supports_f16_color,
    })
}

// ---------------------------------------------------------------------------
// Mutex helpers.
//
// `GL_MUTEX.lock()` can return a `PoisonError` if a previous panic left
// the mutex poisoned. Recover by extracting the inner guard — the GL
// state behind it is just a `()` and there's no invariant to honour.
// Using `unwrap()` here would turn any panic in `convert_*` into a
// permanent failure of every subsequent call.
// ---------------------------------------------------------------------------

fn lock_gl() -> MutexGuard<'static, ()> {
    GL_MUTEX.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// The processor itself.
//
// Holds: EGL display + config + context, the compiled YUYV→RGBA program,
// a fullscreen-quad VAO/VBO, an FBO for off-screen rendering, and a
// pair of GL textures used for transient binding of the source/dest
// IOSurface pbuffers.
//
// GL state is shared across calls to amortize shader compilation and
// VAO/FBO setup. A mutex serializes calls to `convert` so EGL state
// changes (eglMakeCurrent, eglBindTexImage) are not racing.
// ---------------------------------------------------------------------------

pub struct MacosGlProcessor {
    /// Per-instance GL resources. EGL display/context/dummy_pbuffer
    /// live in `SHARED_DISPLAY` (process-global).
    program_yuyv_to_rgba: u32,
    uniform_src: i32,
    uniform_src_size: i32,
    /// Program: GREY (R8 `L008` IOSurface) → RGBA. Shares the `src` sampler
    /// uniform; needs no `src_size`. Also the building block that proves the
    /// R8 IOSurface binding works on ANGLE.
    program_grey_to_rgba: u32,
    uniform_grey_src: i32,
    /// Program: semi-planar YUV (NV12/NV16/NV24, R8 `L008` IOSurface) → RGBA.
    /// One program for all three subsamplings (uniforms select the layout).
    program_nv_to_rgba: u32,
    uniform_nv_src: i32,
    uniform_nv_img_size: i32,
    uniform_nv_tex_width: i32,
    uniform_nv_chroma_shift: i32,
    uniform_nv_uv_row_bytes: i32,
    /// Per-tensor colorimetry uniforms for the NV→RGBA program (YUV→RGB
    /// matrix + range), mirroring the Linux Path B NV shader.
    uniform_nv_y_offset: i32,
    uniform_nv_y_scale: i32,
    uniform_nv_c_vr: i32,
    uniform_nv_c_ug: i32,
    uniform_nv_c_vg: i32,
    uniform_nv_c_ub: i32,
    /// Program: RGBA8 source IOSurface → PlanarRgb F16 destination
    /// IOSurface — zero-copy preprocessing for ONNX+CoreML. The shader
    /// writes RGBA16F-packed half-floats; GL handles the implicit
    /// f32→f16 narrow at fragment-output time.
    program_rgba8_to_planar_f16: u32,
    uniform_rgba8_to_planar_f16_src: i32,
    uniform_rgba8_to_planar_f16_dst_size: i32,
    uniform_rgba8_to_planar_f16_src_rect_uv: i32,
    uniform_rgba8_to_planar_f16_dst_rect_px: i32,
    uniform_rgba8_to_planar_f16_pad_color: i32,
    vao: u32,
    vbo: u32,
    fbo: u32,
    src_tex: u32,
    dst_tex: u32,

    /// (IOSurfaceID, format-as-u32) → cached EGL pbuffer surface.
    /// Same-tensor convert() calls reuse the pbuffer instead of paying
    /// `eglCreatePbufferFromClientBuffer` every frame.
    ///
    /// The cache is guarded by `GL_MUTEX` (so accessed only while the
    /// caller holds the GL lock), but lives on the processor rather
    /// than globally so each processor's resource lifetime is
    /// independent and easy to reason about. ANGLE's pbuffers are
    /// per-display, not per-context, so this is sound.
    ///
    /// LIVE-SURFACE ASSUMPTION: an `IOSurfaceID` is unique only among *live*
    /// surfaces — the kernel may reuse an ID after its surface is freed, and
    /// this cache never evicts on tensor drop. Reuse is safe today because a
    /// HAL `Tensor` keeps its IOSurface alive (`Arc<IoSurfaceHandle>`) for as
    /// long as it can be converted, so a cached pbuffer's surface is always the
    /// one its ID currently names. A caller that frees a tensor and then
    /// converts a *different* tensor whose surface was assigned the recycled ID
    /// would get a stale pbuffer; if that pattern is ever introduced, key the
    /// cache on the tensor's `BufferIdentity` (a process-unique token) instead.
    pbuf_cache: Mutex<HashMap<PbufferCacheKey, egl::Surface>>,
    /// Reusable full-resolution RGBA8 IOSurface for the two-pass YUV→PlanarRgb
    /// path (`(W, H, tensor)`). Pass 1 renders the YUV source into it; pass 2
    /// (`convert_rgba8_to_planar_float`) reads it with letterbox/resize. Cached
    /// by dimensions so the steady-state hot loop never reallocates.
    intermediate_rgba: Mutex<Option<(usize, usize, TensorDyn)>>,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct PbufferCacheKey {
    iosurface_id: u32,
    /// Discriminant of the [`PixelFormat`] — ANGLE validates
    /// FourCC/GL-format agreement at pbuffer-creation time, so two
    /// different formats over the same IOSurface need distinct pbuffers
    /// even though that pairing is unusual in practice.
    format_disc: u8,
    /// Discriminant of the [`edgefirst_tensor::DType`] — a single
    /// IOSurface can only match one dtype (the FourCC encodes the
    /// bytes-per-element), but a tensor handed to the GL processor as
    /// `Tensor<u8>` vs `Tensor<f32>` over the same surface ID would
    /// otherwise collide under the previous (id, format) key.
    dtype_disc: u8,
}

impl std::fmt::Debug for MacosGlProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacosGlProcessor")
            .field("backend", &"ANGLE+IOSurface")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RAII guards.
// ---------------------------------------------------------------------------

/// Makes the shared EGL context current on the calling thread for its
/// lifetime, then releases it on drop. Drop runs even on panic, so the
/// next `MacosGlProcessor::convert*` call on a different processor (or
/// the same one after a panic recovery) sees a clean make-current state.
struct MakeCurrentGuard<'d> {
    egl: &'d Egl,
    display: egl::Display,
}

impl<'d> MakeCurrentGuard<'d> {
    fn new(d: &'d SharedAngleDisplay) -> Result<Self> {
        d.egl
            .make_current(
                d.display,
                Some(d.dummy_pbuffer),
                Some(d.dummy_pbuffer),
                Some(d.context),
            )
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglMakeCurrent: {e:?}"))))?;
        Ok(Self {
            egl: &d.egl,
            display: d.display,
        })
    }
}

impl Drop for MakeCurrentGuard<'_> {
    fn drop(&mut self) {
        // Release the context on this thread. Failure here is logged
        // but ignored — Drop must not panic, and the next make-current
        // will overwrite the state anyway.
        let _ = self.egl.make_current(self.display, None, None, None);
    }
}

/// Owns an `eglBindTexImage` binding. On drop calls `eglReleaseTexImage`
/// — required by the EGL spec before the pbuffer can be destroyed and
/// strictly necessary for ANGLE on some Metal device states.
struct BoundTexImage<'d> {
    egl: &'d Egl,
    display: egl::Display,
    pbuf: egl::Surface,
    bound: bool,
}

impl<'d> BoundTexImage<'d> {
    fn bind(d: &'d SharedAngleDisplay, pbuf: egl::Surface) -> Result<Self> {
        d.egl
            .bind_tex_image(d.display, pbuf, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))?;
        Ok(Self {
            egl: &d.egl,
            display: d.display,
            pbuf,
            bound: true,
        })
    }
}

impl Drop for BoundTexImage<'_> {
    fn drop(&mut self) {
        if self.bound {
            let _ = self
                .egl
                .release_tex_image(self.display, self.pbuf, EGL_BACK_BUFFER);
        }
    }
}

impl MacosGlProcessor {
    pub fn new() -> Result<Self> {
        // SHARED_DISPLAY caches both successes and failures, so this is
        // cheap on the hot path. It also surfaces "ANGLE not installed"
        // exactly once per process.
        let d = shared_display()?;

        // Per-instance setup runs under the GL mutex so we don't race
        // with another processor's convert() on context-current state.
        let _guard = lock_gl();
        let _current = MakeCurrentGuard::new(d)?;

        // SAFETY: serialized via `_guard`; context is current via
        // `_current`. Each helper handles its own internal cleanup on
        // error; if a step later in this sequence fails, the
        // `InstanceCleanup` scope guard below tears down the partially
        // built state.
        unsafe {
            // Build the per-instance resources behind a scope guard so
            // any error path below cleans up GL allocations rather than
            // leaking them.
            let program = compile_program(VERTEX_SHADER, YUYV_TO_RGBA_FRAGMENT)?;
            // From here on, every fallible step must reach Drop-cleanup
            // for `program` if it fails. The simplest pattern: stash
            // resources in `Option<u32>` and let `InstanceCleanup` Drop
            // delete whichever are still `Some`.

            struct InstanceCleanup {
                program: Option<u32>,
                vbo: Option<u32>,
                vao: Option<u32>,
                fbo: Option<u32>,
                src_tex: Option<u32>,
                dst_tex: Option<u32>,
            }
            impl Drop for InstanceCleanup {
                fn drop(&mut self) {
                    // SAFETY: only one current context per thread; we
                    // hold the GL mutex transitively via the caller.
                    unsafe {
                        if let Some(p) = self.program {
                            gls::gl::DeleteProgram(p);
                        }
                        if let Some(b) = self.vbo {
                            gls::gl::DeleteBuffers(1, &b);
                        }
                        if let Some(a) = self.vao {
                            gls::gl::DeleteVertexArrays(1, &a);
                        }
                        if let Some(f) = self.fbo {
                            gls::gl::DeleteFramebuffers(1, &f);
                        }
                        if let Some(t) = self.src_tex {
                            gls::gl::DeleteTextures(1, &t);
                        }
                        if let Some(t) = self.dst_tex {
                            gls::gl::DeleteTextures(1, &t);
                        }
                    }
                }
            }
            let mut cleanup = InstanceCleanup {
                program: Some(program),
                vbo: None,
                vao: None,
                fbo: None,
                src_tex: None,
                dst_tex: None,
            };

            let (uniform_src, uniform_src_size) = {
                let loc_src = gls::gl::GetUniformLocation(program, c"src".as_ptr() as *const _);
                let loc_size =
                    gls::gl::GetUniformLocation(program, c"src_size".as_ptr() as *const _);
                (loc_src, loc_size)
            };

            // Fullscreen-quad VBO + VAO.
            #[rustfmt::skip]
            let quad: [f32; 16] = [
                -1.0,-1.0,  0.0, 0.0,
                 1.0,-1.0,  1.0, 0.0,
                -1.0, 1.0,  0.0, 1.0,
                 1.0, 1.0,  1.0, 1.0,
            ];
            let mut vbo = 0u32;
            let mut vao = 0u32;
            gls::gl::GenBuffers(1, &mut vbo);
            cleanup.vbo = Some(vbo);
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, vbo);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                std::mem::size_of_val(&quad) as isize,
                quad.as_ptr() as *const _,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::GenVertexArrays(1, &mut vao);
            cleanup.vao = Some(vao);
            gls::gl::BindVertexArray(vao);
            gls::gl::VertexAttribPointer(0, 2, gls::gl::FLOAT, 0, 16, std::ptr::null());
            gls::gl::EnableVertexAttribArray(0);
            gls::gl::VertexAttribPointer(1, 2, gls::gl::FLOAT, 0, 16, 8 as *const _);
            gls::gl::EnableVertexAttribArray(1);

            // FBO + two transient texture handles.
            let mut fbo = 0u32;
            let mut src_tex = 0u32;
            let mut dst_tex = 0u32;
            gls::gl::GenFramebuffers(1, &mut fbo);
            cleanup.fbo = Some(fbo);
            gls::gl::GenTextures(1, &mut src_tex);
            cleanup.src_tex = Some(src_tex);
            gls::gl::GenTextures(1, &mut dst_tex);
            cleanup.dst_tex = Some(dst_tex);
            for tex in [src_tex, dst_tex] {
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::NEAREST as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::NEAREST as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_WRAP_S,
                    gls::gl::CLAMP_TO_EDGE as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_WRAP_T,
                    gls::gl::CLAMP_TO_EDGE as i32,
                );
            }

            // Construction succeeded — disarm `cleanup` so its Drop
            // doesn't tear down the resources we're about to hand out.
            let program = cleanup.program.take().unwrap();
            let vbo = cleanup.vbo.take().unwrap();
            let vao = cleanup.vao.take().unwrap();
            let fbo = cleanup.fbo.take().unwrap();
            let src_tex = cleanup.src_tex.take().unwrap();
            let dst_tex = cleanup.dst_tex.take().unwrap();
            std::mem::forget(cleanup);

            // Compile the RGBA8 → PlanarRgb F16 (RGBA16F-packed) program
            // eagerly too — shader compilation is cheap and avoids a
            // per-first-call latency spike on the hot path. The program
            // is only dispatched when the caller actually requests an
            // F16 PlanarRgb destination, so configurations missing
            // GL_EXT_color_buffer_half_float still work for the existing
            // u8 paths (the framebuffer-completeness check at first use
            // is what surfaces the extension gap).
            let program_rgba8_planar_f16 =
                compile_program(VERTEX_SHADER, RGBA8_TO_PLANAR_F16_PACKED_FRAGMENT)?;
            let uniform_rgba8_to_planar_f16_src =
                gls::gl::GetUniformLocation(program_rgba8_planar_f16, c"src".as_ptr());
            let uniform_rgba8_to_planar_f16_dst_size =
                gls::gl::GetUniformLocation(program_rgba8_planar_f16, c"dst_image_size".as_ptr());
            let uniform_rgba8_to_planar_f16_src_rect_uv =
                gls::gl::GetUniformLocation(program_rgba8_planar_f16, c"src_rect_uv".as_ptr());
            let uniform_rgba8_to_planar_f16_dst_rect_px =
                gls::gl::GetUniformLocation(program_rgba8_planar_f16, c"dst_rect_px".as_ptr());
            let uniform_rgba8_to_planar_f16_pad_color =
                gls::gl::GetUniformLocation(program_rgba8_planar_f16, c"pad_color".as_ptr());

            // GREY (R8) → RGBA program — also the R8-binding probe.
            let program_grey = compile_program(VERTEX_SHADER, GREY_TO_RGBA_FRAGMENT)?;
            let uniform_grey_src =
                gls::gl::GetUniformLocation(program_grey, c"src".as_ptr() as *const _);

            // Semi-planar YUV (NV12/NV16/NV24, R8) → RGBA program.
            let program_nv = compile_program(VERTEX_SHADER, NV_TO_RGBA_FRAGMENT)?;
            let uniform_nv_src = gls::gl::GetUniformLocation(program_nv, c"src".as_ptr());
            let uniform_nv_img_size = gls::gl::GetUniformLocation(program_nv, c"img_size".as_ptr());
            let uniform_nv_tex_width =
                gls::gl::GetUniformLocation(program_nv, c"tex_width".as_ptr());
            let uniform_nv_chroma_shift =
                gls::gl::GetUniformLocation(program_nv, c"chroma_shift".as_ptr());
            let uniform_nv_uv_row_bytes =
                gls::gl::GetUniformLocation(program_nv, c"uv_row_bytes".as_ptr());
            let uniform_nv_y_offset = gls::gl::GetUniformLocation(program_nv, c"y_offset".as_ptr());
            let uniform_nv_y_scale = gls::gl::GetUniformLocation(program_nv, c"y_scale".as_ptr());
            let uniform_nv_c_vr = gls::gl::GetUniformLocation(program_nv, c"c_vr".as_ptr());
            let uniform_nv_c_ug = gls::gl::GetUniformLocation(program_nv, c"c_ug".as_ptr());
            let uniform_nv_c_vg = gls::gl::GetUniformLocation(program_nv, c"c_vg".as_ptr());
            let uniform_nv_c_ub = gls::gl::GetUniformLocation(program_nv, c"c_ub".as_ptr());

            Ok(Self {
                program_yuyv_to_rgba: program,
                uniform_src,
                uniform_src_size,
                program_grey_to_rgba: program_grey,
                uniform_grey_src,
                program_nv_to_rgba: program_nv,
                uniform_nv_src,
                uniform_nv_img_size,
                uniform_nv_tex_width,
                uniform_nv_chroma_shift,
                uniform_nv_uv_row_bytes,
                uniform_nv_y_offset,
                uniform_nv_y_scale,
                uniform_nv_c_vr,
                uniform_nv_c_ug,
                uniform_nv_c_vg,
                uniform_nv_c_ub,
                program_rgba8_to_planar_f16: program_rgba8_planar_f16,
                uniform_rgba8_to_planar_f16_src,
                uniform_rgba8_to_planar_f16_dst_size,
                uniform_rgba8_to_planar_f16_src_rect_uv,
                uniform_rgba8_to_planar_f16_dst_rect_px,
                uniform_rgba8_to_planar_f16_pad_color,
                vao,
                vbo,
                fbo,
                src_tex,
                dst_tex,
                pbuf_cache: Mutex::new(HashMap::new()),
                intermediate_rgba: Mutex::new(None),
            })
        }
    }

    /// Two-pass GPU path for the profiler's hot loop: a semi-planar YUV source
    /// (`NV12`/`NV16`/`NV24`, R8 IOSurface) → `PlanarRgb` F16, fully on the GPU.
    ///
    /// Pass 1 samples the YUV source and renders full-resolution RGBA8 into a
    /// cached intermediate IOSurface (reusing the verified `convert_yuyv_to_rgba`
    /// render with the format-selected shader). Pass 2 runs the already-verified
    /// `convert_rgba8_to_planar_float`, which applies the letterbox crop/resize
    /// and packs into the RGBA16F-packed PlanarRgb F16 destination. Both passes
    /// are zero-copy IOSurface↔IOSurface.
    fn convert_nv_to_planar_float(
        &self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        src_fmt: PixelFormat,
        crop: crate::Crop,
    ) -> Result<()> {
        let w = src
            .width()
            .ok_or_else(|| Error::InvalidShape("src width".into()))?;
        let h = src
            .height()
            .ok_or_else(|| Error::InvalidShape("src height".into()))?;

        // Span for the two-pass NV*→PlanarRgb F16 chain. The `_inner` pass
        // helpers below deliberately bypass the public-entry spans (to hold one
        // GL session across both passes), so this is the only span covering the
        // macOS NV→planar-float hot path — catalogued in ARCHITECTURE.md
        // alongside the Linux `image.convert.gl.nv_to_planar.*` spans.
        let _span =
            tracing::trace_span!("image.convert.gl.macos.nv_to_planar", w, h, ?src_fmt).entered();

        // Reuse (or allocate) the full-res RGBA8 intermediate IOSurface.
        //
        // PERF (revisit in profiler measurement): this reallocates when the
        // frame size changes. A reused dataset with many distinct sizes thrashes
        // it. The decoupling now in place (physical grid vs logical ROI) makes
        // the clean fix a single max-size intermediate reconfigured per frame —
        // mirroring the R8 source pool — but that needs
        // `convert_rgba8_to_planar_float` to sample a physical-dims-bound RGBA8
        // source over a logical sub-rect, so it is deferred until the end-to-end
        // numbers show this realloc matters.
        let mut slot = self
            .intermediate_rgba
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !matches!(*slot, Some((iw, ih, _)) if iw == w && ih == h) {
            let interm =
                TensorDyn::image(w, h, PixelFormat::Rgba, DType::U8, Some(TensorMemory::Dma))
                    .map_err(|e| {
                        Error::NotSupported(format!(
                            "convert_nv_to_planar_float: RGBA8 intermediate IOSurface alloc \
                         {w}x{h} failed: {e}"
                        ))
                    })?;
            *slot = Some((w, h, interm));
        }
        let interm = &mut slot.as_mut().unwrap().2;

        // Both passes run under ONE GL session: a single `lock_gl` +
        // `MakeCurrent` held across them so the process-global ANGLE context is
        // never released between pass 1 and pass 2. Releasing/re-acquiring the
        // context mid-operation (two separate sessions) deadlocked under the
        // profiler's pipelined concurrency at depth > 1. Each pass still issues
        // its own `glFinish` before releasing its tex-image bindings (so the GPU
        // is done reading/writing the shared intermediate before it is rebound).
        let d = shared_display()?;
        let _gl_guard = lock_gl();
        let _current = MakeCurrentGuard::new(d)?;
        // Pass 1: YUV(R8) → RGBA8, full-resolution, no crop.
        self.convert_yuyv_to_rgba_inner(d, src, interm, src_fmt, PixelFormat::Rgba)?;
        // Pass 2: RGBA8 → PlanarRgb F16 with letterbox/resize.
        self.convert_rgba8_to_planar_float_inner(d, interm, dst, crop)
    }

    /// RGBA8 IOSurface source → PlanarRgb F16 IOSurface destination.
    ///
    /// Single-pass GPU resize + normalize + HWC→CHW transpose. The
    /// destination IOSurface must be allocated as
    /// `Tensor::<f16>::image(W, H, PixelFormat::PlanarRgb,
    /// Some(TensorMemory::Dma))` — HAL allocates an RGBA16F-packed
    /// surface sized `(W/4, 3*H)` and this method writes 4 contiguous
    /// f16 planar elements per fragment. The IOSurface byte layout
    /// matches the tensor's `[3, H, W]` f16 shape exactly so ORT (or
    /// any NCHW consumer) reads the locked base address without
    /// transpose.
    ///
    /// `dst_w` / `dst_h` are the logical image dimensions (the model's
    /// input shape), NOT the IOSurface's `(W/4, 3H)` footprint. The
    /// caller (typically the EdgeFirst profiler's ONNX path) passes
    /// the model's input width/height.
    ///
    /// Source is any RGBA8 image-shaped tensor with IOSurface storage.
    /// Both tensors must already be allocated; this method does not
    /// allocate.
    ///
    /// # Errors
    ///
    /// - `NotSupported` if either tensor is not IOSurface-backed, the
    ///   destination is not F16 PlanarRgb, or the source is not RGBA.
    /// - `NotSupported` if `GL_EXT_color_buffer_half_float` is missing
    ///   on this configuration. Callers should pre-probe via
    ///   [`Self::supported_render_dtypes`].
    ///
    /// F32 destinations are intentionally not supported: ANGLE's
    /// `iosurface_client_buffer` extension rejects every
    /// `(GL_FLOAT, *)` IOSurface binding with `EGL_BAD_ATTRIBUTE`, and
    /// HAL's `image_iosurface_layout` returns `None` for any F32
    /// combination — so the F32 surface allocation would already have
    /// failed before reaching this method.
    pub fn convert_rgba8_to_planar_float(
        &self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        crop: crate::Crop,
    ) -> Result<()> {
        let d = shared_display()?;
        let _gl_guard = lock_gl();
        let _current = MakeCurrentGuard::new(d)?;
        self.convert_rgba8_to_planar_float_inner(d, src, dst, crop)
    }

    /// RGBA8 → PlanarRgb F16 render, assuming the GL session (`lock_gl` held +
    /// a context current for `d`) is ALREADY established by the caller.
    /// Standalone via [`Self::convert_rgba8_to_planar_float`]; pass 2 of the
    /// single-session two-pass [`Self::convert_nv_to_planar_float`], which holds
    /// ONE session across both passes (no context release between them).
    fn convert_rgba8_to_planar_float_inner(
        &self,
        d: &SharedAngleDisplay,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        crop: crate::Crop,
    ) -> Result<()> {
        let src_u8 = src
            .as_u8()
            .ok_or_else(|| Error::NotSupported("source tensor must be u8 RGBA".into()))?;
        let src_fmt = src_u8.format().ok_or_else(|| {
            Error::InvalidShape("source tensor missing PixelFormat metadata".into())
        })?;
        if src_fmt != PixelFormat::Rgba {
            return Err(Error::NotSupported(format!(
                "convert_rgba8_to_planar_float: source format must be Rgba, got {src_fmt:?}"
            )));
        }
        let src_w = src_u8
            .width()
            .ok_or_else(|| Error::InvalidShape("src width".into()))?;
        let src_h = src_u8
            .height()
            .ok_or_else(|| Error::InvalidShape("src height".into()))?;

        // Resolve the destination IOSurface up-front so the borrow on
        // `dst` is single-arm and doesn't overlap the convert call.
        // F16 is the only supported dtype on this path; F32 IOSurface
        // bindings are rejected by ANGLE and never reach this method.
        let (dst_fmt, dst_w, dst_h, dst_iosurface, dst_id) = match dst {
            TensorDyn::F16(t) => {
                let fmt = t.format().ok_or_else(|| {
                    Error::InvalidShape("destination tensor missing PixelFormat metadata".into())
                })?;
                let w = t
                    .width()
                    .ok_or_else(|| Error::InvalidShape("dst width".into()))?;
                let h = t
                    .height()
                    .ok_or_else(|| Error::InvalidShape("dst height".into()))?;
                let surface = t.iosurface_ref().ok_or_else(|| {
                    Error::NotSupported(
                        "convert_rgba8_to_planar_float: dst is not IOSurface-backed".into(),
                    )
                })?;
                let id = t.iosurface_id().unwrap_or(0);
                (fmt, w, h, surface, id)
            }
            other => {
                return Err(Error::NotSupported(format!(
                    "convert_rgba8_to_planar_float: dst dtype must be F16, got {:?} \
                     (F32 IOSurface bindings are not accepted by ANGLE)",
                    other.dtype()
                )));
            }
        };
        if dst_fmt != PixelFormat::PlanarRgb {
            return Err(Error::NotSupported(format!(
                "convert_rgba8_to_planar_float: dst format must be PlanarRgb, got {dst_fmt:?} \
                 (PlanarRgba would mis-render because the RGBA16F-packed shader is sized for \
                 3 channel planes; a 4-plane variant is the right follow-up)"
            )));
        }

        let src_iosurface = src_u8.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("convert_rgba8_to_planar_float: src is not IOSurface-backed".into())
        })?;
        let src_id = src_u8.iosurface_id().unwrap_or(0);

        if !d.supports_f16_color {
            return Err(Error::NotSupported(
                "GL_EXT_color_buffer_half_float not exposed by this ANGLE/Metal configuration — \
                 F16 PlanarRgb IOSurface render path unavailable"
                    .into(),
            ));
        }
        let dst_dtype = DType::F16;

        // Validate and resolve crop rectangles into shader uniforms.
        // `src_rect_uv` is normalized to the source's full
        // dimensions; `dst_rect_px` is in single-plane pixel coords.
        // `pad_color` is normalized [0,1] from the optional u8
        // `dst_color`. When either rect is None we paint/sample the
        // whole image — equivalent to a no-op identity transform.
        crop.check_crop_dims(src_w, src_h, dst_w, dst_h)?;
        let src_rect_uv = match crop.src_rect {
            Some(r) => [
                r.left as f32 / src_w as f32,
                r.top as f32 / src_h as f32,
                r.width as f32 / src_w as f32,
                r.height as f32 / src_h as f32,
            ],
            None => [0.0, 0.0, 1.0, 1.0],
        };
        let dst_rect_px = match crop.dst_rect {
            Some(r) => [r.left as f32, r.top as f32, r.width as f32, r.height as f32],
            None => [0.0, 0.0, dst_w as f32, dst_h as f32],
        };
        let pad_color = match crop.dst_color {
            Some([r, g, b, _]) => [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0],
            None => [0.0, 0.0, 0.0],
        };

        let src_pbuf = self.get_or_create_pbuffer(
            d,
            src_id,
            src_iosurface,
            PixelFormat::Rgba,
            DType::U8,
            src_w,
            src_h,
        )?;
        let dst_pbuf =
            self.get_or_create_pbuffer(d, dst_id, dst_iosurface, dst_fmt, dst_dtype, dst_w, dst_h)?;

        // SAFETY: GL mutex held; context current via `_current`. RAII
        // guards release tex-image bindings on Drop even on panic.
        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            let _src_bound = BoundTexImage::bind(d, src_pbuf)?;

            // Source filter — bilinear gives smooth resize. The
            // PlanarRgb destination is RGBA16F-packed (4 contiguous
            // f16 planar elements per fragment); linear filtering on
            // the source RGBA8 texture is unconditionally supported.
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.dst_tex);
            let _dst_bound = BoundTexImage::bind(d, dst_pbuf)?;
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.dst_tex,
                0,
            );
            let fbo_status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if fbo_status != gls::gl::FRAMEBUFFER_COMPLETE {
                return Err(Error::Io(std::io::Error::other(format!(
                    "{dst_dtype:?} PlanarRgb FBO incomplete: 0x{fbo_status:x} — \
                     likely missing float color-buffer extension at runtime"
                ))));
            }

            // Packed RGBA16F surface is sized (W/4, 3*H) — each
            // pixel holds 4 contiguous f16 elements of the planar
            // tensor's [3, H, W] byte stream. Viewport covers the
            // full surface so the fragment shader visits every
            // packed pixel. `dst_w` must be a multiple of 4 for the
            // packing to align — the IOSurface allocator
            // (`packed_rgba16f_layout`) rejects non-multiple-of-4 widths, so a
            // correctly-allocated dst can't reach here misaligned; guard anyway
            // against a future caller constructing the dst differently.
            debug_assert!(
                dst_w % 4 == 0,
                "RGBA16F-packed planar dst width {dst_w} must be a multiple of 4"
            );
            let surface_w = (dst_w / 4) as i32;
            let surface_h = (dst_h * 3) as i32;
            gls::gl::Viewport(0, 0, surface_w, surface_h);
            gls::gl::UseProgram(self.program_rgba8_to_planar_f16);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            gls::gl::Uniform1i(self.uniform_rgba8_to_planar_f16_src, 0);
            gls::gl::Uniform2f(
                self.uniform_rgba8_to_planar_f16_dst_size,
                dst_w as f32,
                dst_h as f32,
            );
            gls::gl::Uniform4f(
                self.uniform_rgba8_to_planar_f16_src_rect_uv,
                src_rect_uv[0],
                src_rect_uv[1],
                src_rect_uv[2],
                src_rect_uv[3],
            );
            gls::gl::Uniform4f(
                self.uniform_rgba8_to_planar_f16_dst_rect_px,
                dst_rect_px[0],
                dst_rect_px[1],
                dst_rect_px[2],
                dst_rect_px[3],
            );
            gls::gl::Uniform3f(
                self.uniform_rgba8_to_planar_f16_pad_color,
                pad_color[0],
                pad_color[1],
                pad_color[2],
            );
            gls::gl::BindVertexArray(self.vao);
            gls::gl::DrawArrays(gls::gl::TRIANGLE_STRIP, 0, 4);
            gls::gl::Finish();
        }
        Ok(())
    }

    /// Report which float dtypes the underlying ANGLE/Metal display
    /// can render to as IOSurface color attachments. Probes are run
    /// once at process-global display init (cheap to call repeatedly).
    pub fn supported_render_dtypes(&self) -> crate::RenderDtypeSupport {
        match shared_display() {
            Ok(d) => crate::RenderDtypeSupport {
                f32: d.supports_f32_color,
                f16: d.supports_f16_color,
            },
            // The display failed to initialise; the GL backend itself
            // would be unusable in that case, so report no float
            // support to keep callers on the CPU fallback path.
            Err(_) => crate::RenderDtypeSupport::default(),
        }
    }

    /// Whether the requested conversion is supported by the GL backend.
    /// Used by `ImageProcessor::convert` to decide whether to dispatch
    /// here or fall back to CPU.
    ///
    /// Supported conversions:
    /// * `YUYV → RGBA` — original GL fast path.
    /// * `RGBA → PlanarRgb` (F16 only via dispatch) — the IOSurface
    ///   render path for ONNX/CoreML zero-copy preprocessing.
    ///
    /// Not in the set:
    /// * `YUYV → BGRA` — the current shader writes `vec4(r, g, b, 1.0)`,
    ///   which lands as RGBA bytes in CPU readback regardless of
    ///   IOSurface FourCC. A dedicated BGRA shader (writing
    ///   `vec4(b, g, r, 1.0)`) needs to land before we widen.
    /// * `RGBA → PlanarRgba` — the RGBA16F-packed shader and viewport
    ///   are sized for 3 channel planes (`dst_h * 3`). PlanarRgba is
    ///   correctly allocated as a 4-plane IOSurface by
    ///   `image_iosurface_layout`, but the convert would leave the
    ///   alpha plane uninitialised. Adding a 4-plane shader variant
    ///   is the right follow-up; until then PlanarRgba dispatches to
    ///   CPU rather than misrendering.
    pub fn supports(src_fmt: PixelFormat, dst_fmt: PixelFormat) -> bool {
        matches!(
            (src_fmt, dst_fmt),
            (PixelFormat::Yuyv, PixelFormat::Rgba)
                | (PixelFormat::Grey, PixelFormat::Rgba)
                | (PixelFormat::Nv12, PixelFormat::Rgba)
                | (PixelFormat::Nv16, PixelFormat::Rgba)
                | (PixelFormat::Nv24, PixelFormat::Rgba)
                | (PixelFormat::Rgba, PixelFormat::PlanarRgb)
                // Two-pass YUV → PlanarRgb F16 (the profiler's preprocess).
                | (PixelFormat::Nv12, PixelFormat::PlanarRgb)
                | (PixelFormat::Nv16, PixelFormat::PlanarRgb)
                | (PixelFormat::Nv24, PixelFormat::PlanarRgb)
        )
    }

    /// The actual conversion path. Caller guarantees `supports(src_fmt, dst_fmt)`.
    fn convert_yuyv_to_rgba(
        &self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
    ) -> Result<()> {
        let _span = tracing::trace_span!(
            "image.convert",
            backend = "gl",
            platform = "macos",
            src_fmt = ?src_fmt,
            dst_fmt = ?dst_fmt,
        )
        .entered();

        let d = shared_display()?;
        let _gl_guard = lock_gl();
        let _current = MakeCurrentGuard::new(d)?;
        self.convert_yuyv_to_rgba_inner(d, src, dst, src_fmt, dst_fmt)
    }

    /// Render NV*/YUYV/GREY → RGBA into `dst`, assuming the GL session
    /// (`lock_gl` held + a context current for `d`) is ALREADY established by
    /// the caller. Standalone callers use [`Self::convert_yuyv_to_rgba`]; the
    /// two-pass [`Self::convert_nv_to_planar_float`] calls this as pass 1 under a
    /// single shared session so the process-global ANGLE context is never
    /// released between the passes — that mid-operation release/re-acquire
    /// deadlocked under the profiler's pipelined concurrency (depth > 1).
    fn convert_yuyv_to_rgba_inner(
        &self,
        d: &SharedAngleDisplay,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
    ) -> Result<()> {
        let src_w = src
            .width()
            .ok_or_else(|| Error::InvalidShape("src width".into()))?;
        let src_h = src
            .height()
            .ok_or_else(|| Error::InvalidShape("src height".into()))?;
        let dst_w = dst
            .width()
            .ok_or_else(|| Error::InvalidShape("dst width".into()))?;
        let dst_h = dst
            .height()
            .ok_or_else(|| Error::InvalidShape("dst height".into()))?;

        // Validation: same-size only in this first cut. Resize support
        // is straightforward (just change the viewport + texture sample
        // ratio) but not in scope for the initial integration.
        if src_w != dst_w || src_h != dst_h {
            return Err(Error::NotImplemented(format!(
                "MacosGlProcessor: resize not yet supported (src {src_w}×{src_h} → dst {dst_w}×{dst_h}); CPU fallback handles this"
            )));
        }

        let src_u8 = src
            .as_u8()
            .ok_or_else(|| Error::NotSupported("GL backend requires u8 source tensor".into()))?;
        let dst_u8 = dst.as_u8_mut().ok_or_else(|| {
            Error::NotSupported("GL backend requires u8 destination tensor".into())
        })?;

        // Both tensors MUST be IOSurface-backed for the zero-copy path.
        let src_iosurface = src_u8.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("GL convert: source tensor is not IOSurface-backed".into())
        })?;
        let dst_iosurface = dst_u8.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("GL convert: destination tensor is not IOSurface-backed".into())
        })?;
        let src_id = src_u8.iosurface_id().unwrap_or(0);
        let dst_id = dst_u8.iosurface_id().unwrap_or(0);

        // Look up (or create) the source/dest pbuffers in the cache.
        // Cache miss path calls `eglCreatePbufferFromClientBuffer` and
        // inserts; cache hit returns the existing surface.
        // The src/dst as_u8/as_u8_mut checks above mean both tensors
        // are u8 today. Float-dtype rendering routes through the
        // separate `convert_rgba8_to_planar_float` call site.
        let src_pbuf = match src_fmt {
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24 => {
                // The semi-planar source is a single contiguous R8 plane that may
                // be a reused pool surface larger than this frame. Bind the WHOLE
                // physical IOSurface as one GL_RED texture (not the frame-sized
                // sub-region) so a single cached pbuffer serves every frame: the
                // shader addresses each Y/UV texel within it via `texelFetch`,
                // which Metal resolves to `row * bytesPerRow + col` against the
                // surface's real pitch. Binding at frame dims would instead need
                // a fresh pbuffer per distinct frame size (the regression's tax)
                // and a stale cache entry across sizes.
                let (pw, ph) = src.iosurface_physical_dims().ok_or_else(|| {
                    Error::NotSupported(
                        "GL convert: semi-planar source is not IOSurface-backed".into(),
                    )
                })?;
                self.get_or_create_pbuffer(
                    d,
                    src_id,
                    src_iosurface,
                    PixelFormat::Grey,
                    DType::U8,
                    pw,
                    ph,
                )?
            }
            _ => self.get_or_create_pbuffer(
                d,
                src_id,
                src_iosurface,
                src_fmt,
                DType::U8,
                src_w,
                src_h,
            )?,
        };
        let dst_pbuf =
            self.get_or_create_pbuffer(d, dst_id, dst_iosurface, dst_fmt, DType::U8, dst_w, dst_h)?;

        // SAFETY: GL mutex held; context current via `_current`. Each
        // pbuffer's tex-image binding is owned by a `BoundTexImage` RAII
        // guard so eglReleaseTexImage runs even on panic.
        unsafe {
            // Source texture binding.
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            let _src_bound = BoundTexImage::bind(d, src_pbuf)?;

            // Destination texture binding + attach to FBO.
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.dst_tex);
            let _dst_bound = BoundTexImage::bind(d, dst_pbuf)?;
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.dst_tex,
                0,
            );
            let fbo_status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if fbo_status != gls::gl::FRAMEBUFFER_COMPLETE {
                return Err(Error::Io(std::io::Error::other(format!(
                    "FBO incomplete: 0x{fbo_status:x}"
                ))));
            }

            // Render. Select the source-sampling program by source format:
            //   YUYV  → GL_RG packed sampler;
            //   GREY  → GL_RED, identity luma;
            //   NV12/16/24 → GL_RED single-plane sampler with in-shader
            //     semi-planar addressing (uniforms select the layout).
            gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            match src_fmt {
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24 => {
                    // The combined plane's physical row pitch (== bytes_per_row,
                    // == the bound texture's texel width — the surface is now
                    // allocated so width == pitch, see `iosurface::new_image`).
                    // Every Y/UV byte is addressed in this physical-stride space:
                    // `texelFetch(b % stride, b / stride)` lands exactly on byte
                    // `b`, matching the codec's `row * grid_row_stride + col`
                    // writer. This is what makes NV24's `2*W`-byte chroma line —
                    // which exceeds the even width once the row is padded —
                    // addressable; binding at the even width would leave its
                    // tail columns outside the texture.
                    let stride = src_u8.effective_row_stride().unwrap_or(src_w);
                    // Chroma geometry from the shared single source of truth
                    // (`PixelFormat::chroma_layout`): `shift_x`/`shift_y` select
                    // the sub-sampling, and the UV plane advances
                    // `uv_rows_per_luma * stride` bytes per chroma row (NV24's
                    // full-resolution 2*W-byte line == two grid rows).
                    let layout = src_fmt
                        .chroma_layout()
                        .expect("NV12/NV16/NV24 always have a chroma layout");
                    let uv_row_bytes = (layout.uv_rows_per_luma * stride) as i32;
                    gls::gl::UseProgram(self.program_nv_to_rgba);
                    gls::gl::Uniform1i(self.uniform_nv_src, 0);
                    gls::gl::Uniform2i(self.uniform_nv_img_size, src_w as i32, src_h as i32);
                    gls::gl::Uniform1i(self.uniform_nv_tex_width, stride as i32);
                    gls::gl::Uniform2i(
                        self.uniform_nv_chroma_shift,
                        layout.shift_x as i32,
                        layout.shift_y as i32,
                    );
                    gls::gl::Uniform1i(self.uniform_nv_uv_row_bytes, uv_row_bytes);
                    // YUV→RGB matrix + range from the source colorimetry
                    // (mirrors the Linux Path B NV shader; missing axes filled
                    // by the SD/HD height heuristic).
                    let cm = crate::colorimetry::resolve_colorimetry(
                        src_u8.colorimetry(),
                        src_u8.height(),
                    );
                    let coeffs = crate::colorimetry::yuv_to_rgb_coeffs(
                        cm.encoding
                            .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709),
                        cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited),
                    );
                    gls::gl::Uniform1f(self.uniform_nv_y_offset, coeffs.y_offset);
                    gls::gl::Uniform1f(self.uniform_nv_y_scale, coeffs.y_scale);
                    gls::gl::Uniform1f(self.uniform_nv_c_vr, coeffs.c_vr);
                    gls::gl::Uniform1f(self.uniform_nv_c_ug, coeffs.c_ug);
                    gls::gl::Uniform1f(self.uniform_nv_c_vg, coeffs.c_vg);
                    gls::gl::Uniform1f(self.uniform_nv_c_ub, coeffs.c_ub);
                }
                PixelFormat::Grey => {
                    gls::gl::UseProgram(self.program_grey_to_rgba);
                    gls::gl::Uniform1i(self.uniform_grey_src, 0);
                }
                _ => {
                    gls::gl::UseProgram(self.program_yuyv_to_rgba);
                    gls::gl::Uniform1i(self.uniform_src, 0);
                    gls::gl::Uniform2f(self.uniform_src_size, src_w as f32, src_h as f32);
                }
            }
            gls::gl::BindVertexArray(self.vao);
            gls::gl::DrawArrays(gls::gl::TRIANGLE_STRIP, 0, 4);
            gls::gl::Finish();

            // `_src_bound` and `_dst_bound` Drop release the tex-image
            // bindings here. The pbuffers themselves stay in the cache.
        }
        Ok(())
    }

    /// Look up or create the EGL pbuffer wrapping a given IOSurface.
    ///
    /// Cache key is `(iosurface_id, format_discriminant)`. The cache is
    /// keyed by IOSurfaceID rather than `BufferIdentity` so externally
    /// imported surfaces (via `Tensor::from_iosurface`) share a cache
    /// slot with internally allocated ones — same underlying surface,
    /// same pbuffer.
    ///
    /// IOSurfaceID `0` is treated as un-cacheable: it's the sentinel
    /// returned by `iosurface_id()` when the tensor's IOSurface backing
    /// is somehow malformed (shouldn't happen but the path stays sound).
    //
    // Eight args is one over clippy's default; every one is needed
    // (display state, identity, raw ref, image shape, and dtype for
    // the cache key). Bundling them into a struct would just move the
    // verbosity to the call sites without gaining clarity.
    #[allow(clippy::too_many_arguments)]
    fn get_or_create_pbuffer(
        &self,
        d: &SharedAngleDisplay,
        iosurface_id: u32,
        surface_ref: *mut c_void,
        format: PixelFormat,
        dtype: DType,
        width: usize,
        height: usize,
    ) -> Result<egl::Surface> {
        let key = PbufferCacheKey {
            iosurface_id,
            format_disc: pixel_format_discriminant(format),
            dtype_disc: dtype as u8,
        };
        if iosurface_id != 0 {
            let cache = self.pbuf_cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(&pbuf) = cache.get(&key) {
                return Ok(pbuf);
            }
        }
        // SAFETY: surface_ref borrowed from a live tensor; config has
        // EGL_BIND_TO_TEXTURE_TARGET_ANGLE set.
        let pbuf = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &d.egl,
                d.display,
                d.config,
                surface_ref,
                format,
                dtype,
                width,
                height,
            )?
        };
        if iosurface_id != 0 {
            let mut cache = self.pbuf_cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.insert(key, pbuf);
        }
        Ok(pbuf)
    }
}

fn pixel_format_discriminant(fmt: PixelFormat) -> u8 {
    // PixelFormat is #[repr(u8)] so the cast is a guaranteed
    // collision-free discriminant.
    fmt as u8
}

impl Drop for MacosGlProcessor {
    fn drop(&mut self) {
        // Best-effort cleanup; Drop must not panic.
        let Ok(d) = shared_display() else {
            return; // ANGLE never initialised — nothing to clean up.
        };
        let _gl_guard = lock_gl();
        let _current = match MakeCurrentGuard::new(d) {
            Ok(g) => g,
            Err(_) => return,
        };
        unsafe {
            // Destroy cached pbuffers.
            let mut cache = self
                .pbuf_cache
                .get_mut()
                .map(std::mem::take)
                .unwrap_or_default();
            for (_, pbuf) in cache.drain() {
                let _ = d.egl.destroy_surface(d.display, pbuf);
            }
            // Per-instance GL resources.
            gls::gl::DeleteFramebuffers(1, &self.fbo);
            gls::gl::DeleteTextures(1, &self.src_tex);
            gls::gl::DeleteTextures(1, &self.dst_tex);
            gls::gl::DeleteBuffers(1, &self.vbo);
            gls::gl::DeleteVertexArrays(1, &self.vao);
            gls::gl::DeleteProgram(self.program_yuyv_to_rgba);
            gls::gl::DeleteProgram(self.program_rgba8_to_planar_f16);
            // Shared EGL display/context/dummy_pbuffer outlive every
            // processor instance and are never destroyed — see the
            // module docstring for why.
        }
    }
}

impl ImageProcessorTrait for MacosGlProcessor {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        if !matches!(rotation, Rotation::None) || !matches!(flip, Flip::None) {
            return Err(Error::NotImplemented(
                "MacosGlProcessor: rotation/flip not yet supported; CPU fallback handles this"
                    .into(),
            ));
        }
        let (src_fmt, dst_fmt) = match (src.format(), dst.format()) {
            (Some(s), Some(d)) => (s, d),
            _ => {
                return Err(Error::NotSupported(
                    "MacosGlProcessor: untyped tensors (None format) not supported".into(),
                ));
            }
        };
        if !Self::supports(src_fmt, dst_fmt) {
            return Err(Error::NotSupported(format!(
                "MacosGlProcessor: {src_fmt:?} → {dst_fmt:?} not in the initial GL coverage set"
            )));
        }
        // The float render-path decision is named once for every platform by
        // `gl::float_dispatch::FloatRenderPath`; the F16 arm below corresponds
        // to `FloatRenderPath::IoSurfaceF16Nchw`. It is NOT routed through the
        // shared `classify_float_render` because (a) a macOS IOSurface tensor
        // reports `TensorMemory::Dma` (the IOSurface backing shares the `Dma`
        // slot), so the classifier cannot distinguish it from a Linux DMA-BUF
        // destination, and (b) this match also dispatches the non-float YUYV
        // path and a format-specific F16-required error that are not part of
        // the float decision. Converging it onto the classifier would be a
        // non-mechanical rewrite of working, ANGLE-tested code we cannot
        // compile or exercise on the Linux host; the inline match is retained
        // deliberately. See `gl::float_dispatch` for the shared definition.
        match (src_fmt, dst_fmt, dst.dtype()) {
            // Zero-copy preprocessing path: RGBA8 → PlanarRgb F16 via
            // RGBA16F-packed IOSurface. F32 is intentionally NOT in the
            // set — ANGLE's `iosurface_client_buffer` extension rejects
            // every `(GL_FLOAT, *)` pair with `EGL_BAD_ATTRIBUTE`, so
            // F32 IOSurface allocation also fails upstream in
            // `image_iosurface_layout`. PlanarRgba is excluded for the
            // reasons documented on `supports()`.
            (PixelFormat::Rgba, PixelFormat::PlanarRgb, DType::F16) => {
                self.convert_rgba8_to_planar_float(src, dst, crop)
            }
            // Semi-planar YUV → PlanarRgb F16: two GPU passes (YUV→RGBA8 then
            // the verified RGBA8→PlanarRgb F16 with letterbox/resize).
            (
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24,
                PixelFormat::PlanarRgb,
                DType::F16,
            ) => self.convert_nv_to_planar_float(src, dst, src_fmt, crop),
            (PixelFormat::Rgba, PixelFormat::PlanarRgb, other) => {
                Err(Error::NotSupported(format!(
                    "MacosGlProcessor: Rgba → PlanarRgb requires F16 destination on the \
                     IOSurface fast path (got {other:?}). F32 IOSurface bindings are not \
                     accepted by ANGLE; export the model with F16 inputs or fall back to \
                     a CPU stage."
                )))
            }
            _ => {
                if crop.src_rect.is_some() || crop.dst_rect.is_some() {
                    return Err(Error::NotImplemented(
                        "MacosGlProcessor: crop not yet supported for YUYV→RGBA; \
                         CPU fallback handles this"
                            .into(),
                    ));
                }
                self.convert_yuyv_to_rgba(src, dst, src_fmt, dst_fmt)
            }
        }
    }

    fn draw_decoded_masks(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[DetectBox],
        _segmentation: &[Segmentation],
        _overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "MacosGlProcessor: draw_decoded_masks not yet ported (use CPU backend)".into(),
        ))
    }

    fn draw_proto_masks(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[DetectBox],
        _proto_data: &ProtoData,
        _overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "MacosGlProcessor: draw_proto_masks not yet ported (use CPU backend)".into(),
        ))
    }

    fn set_class_colors(&mut self, _colors: &[[u8; 4]]) -> Result<()> {
        // Class-color lookup table is only used by mask rendering, which
        // currently falls back to CPU on macOS. Accepting the call as a
        // no-op keeps the API surface symmetric with Linux.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shader helpers
// ---------------------------------------------------------------------------

unsafe fn compile_program(vertex_src: &str, fragment_src: &str) -> Result<u32> {
    let vs = compile_shader(gls::gl::VERTEX_SHADER, vertex_src)?;
    // From this point on, `vs` and (later) `fs` and `program` are owned
    // by the helper and must be cleaned up on any error path. Track them
    // in an Option and let `ProgramBuild`'s Drop clean up whatever is
    // still present when we leave the function abnormally.
    struct ProgramBuild {
        vs: Option<u32>,
        fs: Option<u32>,
        program: Option<u32>,
    }
    impl Drop for ProgramBuild {
        fn drop(&mut self) {
            unsafe {
                if let Some(p) = self.program {
                    gls::gl::DeleteProgram(p);
                }
                if let Some(s) = self.fs {
                    gls::gl::DeleteShader(s);
                }
                if let Some(s) = self.vs {
                    gls::gl::DeleteShader(s);
                }
            }
        }
    }
    let mut state = ProgramBuild {
        vs: Some(vs),
        fs: None,
        program: None,
    };

    let fs = compile_shader(gls::gl::FRAGMENT_SHADER, fragment_src)?;
    state.fs = Some(fs);

    let program = gls::gl::CreateProgram();
    state.program = Some(program);
    gls::gl::AttachShader(program, vs);
    gls::gl::AttachShader(program, fs);
    gls::gl::LinkProgram(program);
    let mut ok = 0i32;
    gls::gl::GetProgramiv(program, gls::gl::LINK_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut len = 0i32;
        gls::gl::GetProgramInfoLog(
            program,
            log.len() as i32,
            &mut len,
            log.as_mut_ptr() as *mut _,
        );
        // `state` Drop deletes program + fs + vs as we return.
        return Err(Error::Internal(format!(
            "program link failed: {}",
            String::from_utf8_lossy(&log[..len.max(0) as usize])
        )));
    }

    // Success: detach shaders + delete them (GL drops them when
    // unreferenced by any program). Disarm state so it doesn't delete
    // the program we're returning.
    gls::gl::DeleteShader(state.vs.take().unwrap());
    gls::gl::DeleteShader(state.fs.take().unwrap());
    let program = state.program.take().unwrap();
    std::mem::forget(state);
    Ok(program)
}

unsafe fn compile_shader(kind: u32, src: &str) -> Result<u32> {
    let shader = gls::gl::CreateShader(kind);
    let c = CString::new(src).map_err(|e| Error::Internal(format!("shader CString: {e}")))?;
    let ptr = c.as_ptr();
    let len = src.len() as i32;
    gls::gl::ShaderSource(shader, 1, &ptr, &len);
    gls::gl::CompileShader(shader);
    let mut ok = 0i32;
    gls::gl::GetShaderiv(shader, gls::gl::COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut len = 0i32;
        gls::gl::GetShaderInfoLog(
            shader,
            log.len() as i32,
            &mut len,
            log.as_mut_ptr() as *mut _,
        );
        return Err(Error::Internal(format!(
            "shader compile failed (kind=0x{kind:x}): {}",
            String::from_utf8_lossy(&log[..len.max(0) as usize])
        )));
    }
    Ok(shader)
}
