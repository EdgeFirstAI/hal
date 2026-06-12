// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS implementation of [`GlPlatform`]: ANGLE (GLES→Metal) display
//! bring-up and IOSurface buffer import.
//!
//! Owns the **process-global** shared ANGLE display ([`SharedAngleDisplay`]
//! / [`shared_display`]) — `eglTerminate` is ref-counted but never safely
//! terminable mid-process (ANGLE's Metal device is a per-process
//! singleton), exactly the reason Linux keeps a `SharedEglDisplay` in
//! `context.rs`. Both the legacy `MacosGlProcessor` (shared-context model,
//! deleted at the end of the convergence) and the per-processor
//! [`AngleDisplay`] below initialize through this one entry point.
//!
//! [`AngleDisplay`] is the convergence-model display: each processor's
//! dedicated worker thread creates ITS OWN EGL context (plus a small dummy
//! pbuffer so the context can be current without a convert in flight),
//! makes it current ONCE, and holds it for the thread's life — the same
//! shape as the Linux `GlContext`. Contexts on one shared ANGLE display
//! scale across threads (A0 spike: render S(4)≈5.1 at 720p, per-convert
//! `eglBindTexImage`/`eglReleaseTexImage` S(4)≈3.5 — ANGLE's EGL
//! entry-point lock is not a practical serializer; cold imports serialize
//! at ~0.7 µs each on the cache-miss path only).
//!
//! Buffer import wraps the tensor's IOSurface as an EGL pbuffer via
//! `EGL_ANGLE_iosurface_client_buffer` (see `iosurface_import.rs` for the
//! attribute assembly). The owned [`IoSurfacePbuffer`] destroys the
//! pbuffer on drop; the [`super::super::cache::ImportCache`] holds these
//! exactly like Linux `EglImage`s.

use super::super::iosurface_import;
use super::super::Egl;
use super::macos::MacosPlatform;
use super::GlPlatform;
use crate::{Error, Result};
use edgefirst_tensor::{DType, PixelFormat, Tensor};
use khronos_egl as egl;
use log::debug;
use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// EGL constants shared by the display bring-up and per-processor context
// creation (from EGL/eglext headers; khronos_egl does not export these).
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
pub(in crate::opengl_headless) const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_BACK_BUFFER: i32 = 0x3084;

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table.
//
// `gls::load_with` populates global function pointers — exists once per
// process. We load via EGL's `eglGetProcAddress` so the symbols come
// from ANGLE's libGLESv2.dylib. The pointers are display-global, so a
// single load serves every context created on the shared display.
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
// Process-global ANGLE EGL display.
// ---------------------------------------------------------------------------

/// All process-global EGL state. Use [`shared_display`] to access.
pub(in crate::opengl_headless) struct SharedAngleDisplay {
    /// Static-lifetime EGL handle. The actual ANGLE libEGL.dylib is
    /// leaked at first dlopen and never closed.
    pub(in crate::opengl_headless) egl: Egl,
    pub(in crate::opengl_headless) display: egl::Display,
    pub(in crate::opengl_headless) config: egl::Config,
    /// The legacy shared context used by `MacosGlProcessor` (every
    /// instance serializes on it behind that module's `GL_MUTEX`).
    /// The convergence model does not use it — each [`AngleDisplay`]
    /// creates a private context — and it is deleted together with
    /// `MacosGlProcessor` at the end of the convergence.
    pub(in crate::opengl_headless) context: egl::Context,
    /// Tiny scratch surface kept alive so the shared context can be made
    /// current outside of a `convert` call (e.g. for shader compile,
    /// resource allocation, or `Drop`-time cleanup).
    pub(in crate::opengl_headless) dummy_pbuffer: egl::Surface,
    /// `GL_EXT_color_buffer_float` is exposed by this ANGLE/Metal
    /// configuration. Gates F32 destination tensors on the IOSurface
    /// render path.
    pub(in crate::opengl_headless) supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` is exposed by this
    /// ANGLE/Metal configuration. Gates F16 destination tensors on
    /// the IOSurface render path.
    pub(in crate::opengl_headless) supports_f16_color: bool,
}

// SAFETY: every member is either a leak'd static, an EGL handle (which
// the ANGLE driver synchronises internally), or a pointer to driver-
// owned state. Display-level EGL entry points are internally
// synchronized by ANGLE; the legacy shared `context` is only made
// current under `macos_processor`'s GL_MUTEX.
unsafe impl Send for SharedAngleDisplay {}
unsafe impl Sync for SharedAngleDisplay {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedAngleDisplay, String>> = OnceLock::new();

/// Acquire a reference to the process-global ANGLE display, initialising
/// it on first call. Subsequent calls return the same handle. The error
/// case is cached too — once ANGLE fails to load we don't keep retrying.
pub(in crate::opengl_headless) fn shared_display() -> Result<&'static SharedAngleDisplay> {
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
    debug!("ANGLE EGL {maj}.{min} initialised (process-global shared display)");

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

    // 4. GLES3 context (the legacy shared one — see the field docs).
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
        "ANGLE: GL_EXT_color_buffer_float={supports_f32_color}, \
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
// Per-processor display: a private context on the shared ANGLE display.
// ---------------------------------------------------------------------------

/// One processor's GL bring-up state: a PRIVATE EGL context (plus dummy
/// pbuffer) on the process-global shared ANGLE display.
///
/// Created on the processor's dedicated worker thread, made current there
/// ONCE, and held current for the thread's life — the Linux `GlContext`
/// model. NOT `Send`: the context is current on its creating thread and
/// must be dropped there too (the dispatch wrapper guarantees both).
pub(in crate::opengl_headless) struct AngleDisplay {
    pub(in crate::opengl_headless) shared: &'static SharedAngleDisplay,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Duck-typed counterparts of the two `GlContext` members the
    /// portable engine reads. IOSurface is the only transfer backend on
    /// macOS; ANGLE's Metal backend exposes GLES 3.0 (no 3.1 compute).
    pub(in crate::opengl_headless) transfer_backend: super::super::TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
    /// Texture attachments made via `eglBindTexImage` since the last
    /// `GlPlatform::end_gpu_pass` — released there, after the engine's
    /// sync point (rendering must complete before release per the EGL
    /// pbuffer contract). Single-threaded access (the owning worker).
    active_binds: std::cell::RefCell<Vec<egl::Surface>>,
}

impl AngleDisplay {
    /// Assemble the capability surface for this display (see
    /// `PlatformCaps`): IOSurface transfer, float render support from the
    /// shared display's extension probes, and no process-wide GL
    /// serialization — ANGLE/Metal contexts run in parallel (A0 spike).
    pub(in crate::opengl_headless) fn platform_caps(&self) -> super::PlatformCaps {
        super::PlatformCaps {
            transfer_backend: super::super::TransferBackend::IOSurface,
            render_dtypes: crate::RenderDtypeSupport {
                f32: self.shared.supports_f32_color,
                f16: self.shared.supports_f16_color,
            },
            serialize_gl: false,
            external_oes: false,
        }
    }
}

impl Drop for AngleDisplay {
    fn drop(&mut self) {
        // Runs on the owning worker thread (the only place the wrapper
        // drops it): release the context from this thread, then destroy.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        let _ = d.egl.destroy_context(d.display, self.context);
    }
}

/// An owned IOSurface→EGL-pbuffer import. Dropping destroys the pbuffer
/// (display-level op; ANGLE synchronizes internally). Cached in
/// `ImportCache<IoSurfacePbuffer>` keyed by `BufferImportKey`, exactly
/// like Linux `EglImage`s.
pub(in crate::opengl_headless) struct IoSurfacePbuffer {
    shared: &'static SharedAngleDisplay,
    pub(in crate::opengl_headless) surface: egl::Surface,
}

impl Drop for IoSurfacePbuffer {
    fn drop(&mut self) {
        let _ = self
            .shared
            .egl
            .destroy_surface(self.shared.display, self.surface);
    }
}

/// Marker type: macOS ANGLE + IOSurface platform.
pub(crate) struct AngleClientBuffer;

impl GlPlatform for AngleClientBuffer {
    type Display = AngleDisplay;
    type Import = IoSurfacePbuffer;
    type ImportHandle = egl::Surface;

    // eglBindTexImage attachments are released at end_gpu_pass — the
    // engine's binding-skip cache must stay cold on macOS.
    const PERSISTENT_TEX_BINDINGS: bool = false;
    const EXTERNAL_OES: bool = false;

    fn load_gl_once(_display: &AngleDisplay) {
        // Loaded once at shared-display init (`load_gl_once` above runs
        // inside `init_shared_display`, before any context exists).
    }

    fn import_handle(import: &IoSurfacePbuffer) -> egl::Surface {
        import.surface
    }

    unsafe fn attach_tex_image_2d(display: &AngleDisplay, handle: egl::Surface) -> Result<()> {
        display
            .shared
            .egl
            .bind_tex_image(display.shared.display, handle, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))?;
        display.active_binds.borrow_mut().push(handle);
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &AngleDisplay,
        _handle: egl::Surface,
    ) -> Result<()> {
        // Unreachable in practice: PlatformCaps::external_oes is false on
        // macOS, so path selection never picks the external-sampler path.
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES is not available on ANGLE/Metal".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &AngleDisplay,
        _handle: egl::Surface,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "renderbuffer import targets are not available on ANGLE/Metal              (EDGEFIRST_OPENGL_RENDERSURFACE has no effect on macOS)"
                .into(),
        ))
    }

    fn end_gpu_pass(display: &AngleDisplay) {
        // Called after the engine's sync point: every recorded pbuffer
        // attachment can now be released (EGL requires release before the
        // pbuffer may be rebound elsewhere or destroyed).
        for surface in display.active_binds.borrow_mut().drain(..) {
            let _ = display.shared.egl.release_tex_image(
                display.shared.display,
                surface,
                EGL_BACK_BUFFER,
            );
        }
    }

    fn import_buffer_packed<T>(
        display: &AngleDisplay,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> Result<IoSurfacePbuffer>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        let surface_ref = img.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("packed import: tensor is not IOSurface-backed".into())
        })?;
        // The caller passes PACKED surface dims. ImageLayout speaks
        // logical dims, so map each packed format onto the layout arm
        // that produces exactly this surface: RGBA8 maps 1:1; the
        // RGBA16F planar packing is (W/4, 3·H) — reconstruct the
        // logical (W, H) the (PlanarRgb, F16) arm derives it from.
        let (pf, dt, w, h) = match fmt {
            super::PackedImportFormat::Rgba8888 => (PixelFormat::Rgba, DType::U8, width, height),
            super::PackedImportFormat::Rgba16161616F => {
                if !height.is_multiple_of(3) {
                    return Err(Error::NotSupported(format!(
                        "packed RGBA16F surface height {height} is not a 3-plane stack"
                    )));
                }
                (PixelFormat::PlanarRgb, DType::F16, width * 4, height / 3)
            }
        };
        let shared = display.shared;
        // SAFETY: as in `import_buffer`.
        let surface = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &shared.egl,
                shared.display,
                shared.config,
                surface_ref,
                pf,
                dt,
                w,
                h,
            )?
        };
        Ok(IoSurfacePbuffer { shared, surface })
    }

    fn init_display(kind: Option<super::super::EglDisplayKind>) -> Result<AngleDisplay> {
        if let Some(kind) = kind {
            debug!("EglDisplayKind::{kind} ignored on macOS — ANGLE/Metal is the only display");
        }
        let shared = shared_display()?;
        let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
        // No share-list: each processor owns its programs/VAOs/FBOs, the
        // same isolation Linux contexts have.
        let context = shared
            .egl
            .create_context(shared.display, shared.config, None, &ctx_attribs)
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "eglCreateContext (per-processor): {e:?}"
                )))
            })?;
        let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
        let dummy_pbuffer = shared
            .egl
            .create_pbuffer_surface(shared.display, shared.config, &dummy_attribs)
            .map_err(|e| {
                let _ = shared.egl.destroy_context(shared.display, context);
                Error::Io(std::io::Error::other(format!(
                    "eglCreatePbufferSurface (per-processor dummy): {e:?}"
                )))
            })?;
        // Made current ONCE on the calling (worker) thread and held for
        // the thread's life — per-message make-current was the legacy
        // shared-context model's cost, not this one's.
        if let Err(e) = shared.egl.make_current(
            shared.display,
            Some(dummy_pbuffer),
            Some(dummy_pbuffer),
            Some(context),
        ) {
            let _ = shared.egl.destroy_surface(shared.display, dummy_pbuffer);
            let _ = shared.egl.destroy_context(shared.display, context);
            return Err(Error::Io(std::io::Error::other(format!(
                "eglMakeCurrent (per-processor): {e:?}"
            ))));
        }
        Ok(AngleDisplay {
            shared,
            context,
            dummy_pbuffer,
            transfer_backend: super::super::TransferBackend::IOSurface,
            has_compute: false,
            active_binds: std::cell::RefCell::new(Vec::new()),
        })
    }

    fn import_buffer(
        display: &AngleDisplay,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> Result<IoSurfacePbuffer> {
        let surface_ref = img.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("GL convert: tensor is not IOSurface-backed".into())
        })?;
        // Semi-planar YUV has no multi-plane pbuffer binding on ANGLE —
        // the engine selects the ShaderR8 strategy (import_buffer_nv_r8)
        // for NV sources on this platform.
        if matches!(
            fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) {
            return Err(Error::NotSupported(
                "ANGLE IOSurface import has no multi-plane NV binding — use the R8 path".into(),
            ));
        }
        // A destination view()/batch() tile imports its PARENT surface
        // (the per-tile offset is viewport state, mirroring the Linux
        // dst-view collapse in DmaImportAttrs/BufferImportKey).
        let (w, h) = match (for_dst, img.view_origin()) {
            (true, Some(vo)) => (vo.parent_width, vo.parent_height),
            _ => (
                img.width()
                    .ok_or_else(|| Error::InvalidShape("import width".into()))?,
                img.height()
                    .ok_or_else(|| Error::InvalidShape("import height".into()))?,
            ),
        };
        let shared = display.shared;
        // SAFETY: surface_ref borrowed from a live tensor; the import
        // cache entry's `guard` ties the pbuffer's lifetime to it.
        let surface = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &shared.egl,
                shared.display,
                shared.config,
                surface_ref,
                fmt,
                DType::U8,
                w,
                h,
            )?
        };
        Ok(IoSurfacePbuffer { shared, surface })
    }

    fn import_buffer_nv_r8(
        display: &AngleDisplay,
        img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<IoSurfacePbuffer> {
        let surface_ref = img.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("GL convert: NV source is not IOSurface-backed".into())
        })?;
        // Bind the WHOLE physical IOSurface as one R8 (`L008`) texture so
        // a single cached pbuffer serves every frame of a reused pool
        // surface; the shader addresses Y/UV texels within it (same
        // rationale as the legacy NV binding in `macos_processor`).
        let (pw, ph) = img.iosurface_physical_dims().ok_or_else(|| {
            Error::NotSupported("GL convert: NV source has no IOSurface physical dims".into())
        })?;
        let shared = display.shared;
        // SAFETY: as in `import_buffer`.
        let surface = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &shared.egl,
                shared.display,
                shared.config,
                surface_ref,
                PixelFormat::Grey,
                DType::U8,
                pw,
                ph,
            )?
        };
        Ok(IoSurfacePbuffer { shared, surface })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Per-processor context bring-up latency (PR-A A4 gate: each
    /// processor now creates its own context — document the cost,
    /// budget <50 ms each). Ignored: needs ANGLE + a GPU; run on demand:
    /// `cargo test -p edgefirst-image --release --lib angle::tests -- --ignored --nocapture`
    #[test]
    #[ignore = "needs ANGLE + Apple GPU; run on demand"]
    fn per_processor_context_bring_up_latency() {
        // First call pays shared-display init (dlopen + eglInitialize);
        // measure it separately from the steady per-processor cost.
        let t0 = Instant::now();
        let first = AngleClientBuffer::init_display(None).expect("first display");
        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(first);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let t0 = Instant::now();
                    let d = AngleClientBuffer::init_display(None).expect("display");
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    drop(d);
                    (i, ms)
                })
            })
            .collect();
        for h in handles {
            let (i, ms) = h.join().expect("thread");
            println!("per-processor context {i}: {ms:.1} ms");
            assert!(
                ms < 50.0,
                "context bring-up {ms:.1} ms exceeds 50 ms budget"
            );
        }
        println!("first (incl. shared display init): {first_ms:.1} ms");
    }
}
