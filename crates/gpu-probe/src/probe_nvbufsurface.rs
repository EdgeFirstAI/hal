// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
//! NvBufSurface dma-buf interop probe — Jetson only (O1–O3).
//!
//! Validates the hypothesis that on Jetson we can keep the existing
//! `TensorMemory::Dma` contract by swapping the allocator from `/dev/dma_heap`
//! (absent on this L4T R36.4 image) to **NvBufSurface** (dlopen'd at runtime).
//!
//! The probe is fully **gated**: if `libnvbufsurface.so` cannot be dlopen'd
//! (i.e. we are not on a Jetson) the whole module reports SKIP and is a no-op,
//! so cross-platform / non-Jetson runs are unaffected.
//!
//! Objectives:
//! * **O1 — alloc + fd:** `NvBufSurfaceCreate` a pitch-linear SURFACE_ARRAY
//!   surface sized for our packed RGBA16F `(W/4, 3H)` layout; read the
//!   `surfaceList[0].bufferDesc` dma-buf fd and `fstat` it.
//! * **O2 — GL import + render:** import that fd as an `EGLImage`
//!   (`DRM_FORMAT_ABGR16161616F`), bind as an EGLImage-backed renderbuffer,
//!   render a known color, read it back and verify — the decisive test that
//!   orin-nano OpenGL accepts an NvBufSurface-created dma-buf.
//! * **O3 — fd round-trip:** `NvBufSurfaceFromFd(fd, &out)` to prove the
//!   `clone_fd`-style sharing interops with NvBufSurface.
//!
//! ## ABI note
//! The structs below mirror `/usr/src/jetson_multimedia_api/include/nvbufsurface.h`
//! exactly (`#[repr(C)]`). We only mirror enough of `NvBufSurfaceParams` to
//! reach `bufferDesc` (offset 24); the trailing fields are padded out with a
//! reserved tail so the parent `NvBufSurface.surfaceList[0]` indexing and the
//! library's own writes land at the right offsets. A runtime offset assertion
//! guards the critical `bufferDesc` position.

use crate::egl_context::GpuContext;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::os::raw::c_int;

/// `DRM_FORMAT_ABGR16161616F` — packed little-endian R,G,B,A half-floats; the
/// only float DRM fourcc usable as a renderable RGBA16F dma-buf. Matches
/// `probe_float_render`.
const DRM_FORMAT_ABGR16161616F: u32 = 0x4834_4241;

/// `DRM_FORMAT_ABGR8888` (`'AB24'`) — matches the 8-bit RGBA NvBufSurface the
/// fallback allocator produces; used to prove the EGLImage import path works at
/// all when the fourcc agrees with the driver-allocated surface format.
const DRM_FORMAT_ABGR8888: u32 = 0x3432_4241;

// ---------------------------------------------------------------------------
// NvBufSurface enum values (from nvbufsurface.h)
// ---------------------------------------------------------------------------

const NVBUF_MEM_SURFACE_ARRAY: u32 = 4;
const NVBUF_LAYOUT_PITCH: u32 = 0;
/// `NVBUF_COLOR_FORMAT_RGBA` — index 19 in the color-format enum (counted from
/// INVALID=0). `NvBufSurfaceCreate` validates colorFormat even when `size` is
/// set (a 0 colorFormat returns rc=-1), so a valid format is always supplied.
const NVBUF_COLOR_FORMAT_RGBA: u32 = 19;

// ---------------------------------------------------------------------------
// #[repr(C)] ABI mirror of nvbufsurface.h
// ---------------------------------------------------------------------------

/// Mirror of `NvBufSurfaceCreateParams` (flat params — simpler than the nested
/// `NvBufSurfaceAllocateParams`). All enums are C `int`/`u32` (4 bytes).
#[repr(C)]
struct NvBufSurfaceCreateParams {
    gpu_id: u32,
    width: u32,
    height: u32,
    size: u32,
    is_contiguous: bool,
    color_format: u32, // NvBufSurfaceColorFormat
    layout: u32,       // NvBufSurfaceLayout
    mem_type: u32,     // NvBufSurfaceMemType
}

/// Partial mirror of `NvBufSurfaceParams`. We mirror the leading fields up to
/// and including `bufferDesc` (the dma-buf fd) exactly, then pad the remainder
/// out to the real struct size so `surfaceList[0]` indexing is correct and the
/// library writes into valid memory. The real struct's tail is
/// `dataSize(u32) + pad + dataPtr(ptr) + planeParams + mappedAddr + paramex(ptr)
/// + _reserved[3]`; we over-reserve generously and never read it.
#[repr(C)]
struct NvBufSurfaceParams {
    width: u32,
    height: u32,
    pitch: u32,
    color_format: u32, // NvBufSurfaceColorFormat
    layout: u32,       // NvBufSurfaceLayout
    // 5 * u32 = 20 bytes, then 4 bytes pad → bufferDesc at offset 24 (8-aligned).
    buffer_desc: u64,
    // Opaque tail. The real NvBufSurfaceParams continues with dataSize, dataPtr,
    // NvBufSurfacePlaneParams, NvBufSurfaceMappedAddr, paramex, _reserved[3].
    // NvBufSurfacePlaneParams: u32 num_planes + 4*NVBUF_MAX_PLANES of
    // {u32 width,height,pitch,offset,psize,bytesPerPix} + 4*u64 offset
    // + 4*u64 pitch ... — large. Reserve a comfortable 512 bytes so the
    // allocator never scribbles past our backing store.
    _tail: [u8; 512],
}

/// Mirror of `NvBufSurface`. `surfaceList` is an 8-byte aligned pointer; the
/// 3 preceding u32 + bool(1)+3pad + u32 memType = 20 bytes → 4 bytes pad →
/// surfaceList at offset 24, matching the C layout.
#[repr(C)]
struct NvBufSurface {
    gpu_id: u32,
    batch_size: u32,
    num_filled: u32,
    is_contiguous: bool,
    mem_type: u32, // NvBufSurfaceMemType
    surface_list: *mut NvBufSurfaceParams,
    _reserved: [*mut c_void; 4], // STRUCTURE_PADDING == 4
}

// Function signatures (dlsym'd).
type FnCreate = unsafe extern "C" fn(
    surf: *mut *mut NvBufSurface,
    batch_size: u32,
    params: *const NvBufSurfaceCreateParams,
) -> c_int;
type FnDestroy = unsafe extern "C" fn(surf: *mut NvBufSurface) -> c_int;
type FnFromFd = unsafe extern "C" fn(dmabuf_fd: c_int, buffer: *mut *mut c_void) -> c_int;

/// dlopen `libnvbufsurface.so` from the Jetson nvidia path, then the bare name.
fn open_lib() -> Option<Library> {
    let candidates = [
        "/usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so",
        "libnvbufsurface.so",
    ];
    for path in candidates {
        if let Ok(lib) = unsafe { Library::new(path) } {
            log::debug!("dlopen'd {path}");
            return Some(lib);
        }
    }
    None
}

/// fstat a raw fd and return its size in bytes, or an error string.
///
/// Borrows the fd without taking ownership so we never close it out from under
/// the live NvBufSurface that owns it.
fn fstat_size(fd: c_int) -> Result<u64, String> {
    use std::os::fd::BorrowedFd;
    if fd < 0 {
        return Err(format!("invalid fd {fd}"));
    }
    // SAFETY: fd is a valid dma-buf fd owned by the live NvBufSurface.
    let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
    match nix::sys::stat::fstat(borrowed) {
        Ok(st) => Ok(st.st_size as u64),
        Err(e) => Err(format!("fstat({fd}): {e}")),
    }
}

/// Result of a single allocation attempt.
struct AllocResult {
    surf: *mut NvBufSurface,
    fd: c_int,
    /// The surface's reported width/height/colorFormat — the driver may pick
    /// its own geometry regardless of what we requested, so import must use
    /// these, not the requested dimensions.
    width: u32,
    height: u32,
    color_format: u32,
    pitch: u32,
    size: u64,
    strategy: &'static str,
}

/// Try `NvBufSurfaceCreate` with the given params; on success extract the fd
/// from `surfaceList[0].bufferDesc`.
unsafe fn try_alloc(
    create: &Symbol<FnCreate>,
    params: &NvBufSurfaceCreateParams,
    strategy: &'static str,
) -> Result<AllocResult, String> {
    let mut surf: *mut NvBufSurface = std::ptr::null_mut();
    let rc = create(&mut surf, 1, params);
    if rc != 0 {
        return Err(format!("NvBufSurfaceCreate rc={rc}"));
    }
    if surf.is_null() {
        return Err("NvBufSurfaceCreate returned null surface".into());
    }
    let s = &*surf;
    if s.surface_list.is_null() {
        return Err("surfaceList is null".into());
    }
    let p = &*s.surface_list;
    let fd = p.buffer_desc as c_int;
    let size = fstat_size(fd)?;
    Ok(AllocResult {
        surf,
        fd,
        width: p.width,
        height: p.height,
        color_format: p.color_format,
        pitch: p.pitch,
        size,
        strategy,
    })
}

/// A DRM-fourcc import candidate to try against the allocated NvBufSurface.
struct ImportCase {
    /// Human label, e.g. `"ABGR16161616F"`.
    label: &'static str,
    /// DRM fourcc passed to `eglCreateImage(EGL_LINUX_DRM_FOURCC)`.
    fourcc: u32,
    /// EGLImage geometry width (in fourcc texels).
    w: u32,
    /// EGLImage geometry height.
    h: u32,
    /// `glReadPixels` client format.
    rd_format: u32,
    /// `glReadPixels` client type.
    rd_type: u32,
    /// Bytes per readback texel.
    bpp: u32,
}

/// Render a known color into the FBO and read back, verifying non-zero bytes.
/// Assumes the FBO is currently bound and complete.
unsafe fn render_and_verify(case: &ImportCase) -> bool {
    gls::gl::Viewport(0, 0, case.w as i32, case.h as i32);
    gls::gl::ClearColor(0.5, 0.25, 0.75, 1.0);
    gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
    gls::gl::Finish();

    let mut buf = vec![0u8; (case.w * case.h * case.bpp) as usize];
    gls::gl::ReadPixels(
        0,
        0,
        case.w as i32,
        case.h as i32,
        case.rd_format,
        case.rd_type,
        buf.as_mut_ptr() as *mut c_void,
    );
    buf.iter().any(|&b| b != 0)
}

/// O2 inner: import `fd` as an EGLImage under `case`'s fourcc/geometry, then try
/// to make it an FBO color attachment two ways — first as a **renderbuffer**
/// (`EGLImageTargetRenderbufferStorageOES`), then, if that fails, as a
/// **TEXTURE_2D** (`EGLImageTargetTexture2DOES`). On Tegra the renderbuffer
/// route is frequently rejected for imported EGLImages while the texture route
/// works, so reporting both disambiguates the HAL import strategy.
unsafe fn gl_try_import(ctx: &GpuContext, fd: c_int, pitch: u32, case: &ImportCase) -> String {
    let egl_image =
        match ctx.create_egl_image_dma(fd, case.w as i32, case.h as i32, case.fourcc, pitch as i32)
        {
            Ok(img) => img,
            Err(e) => return format!("import-rejected ({e})"),
        };

    // --- Route 1: renderbuffer attachment ---
    let mut rbo: u32 = 0;
    gls::gl::GenRenderbuffers(1, &mut rbo);
    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
    let _ = gls::gl::GetError();
    gls::gl::EGLImageTargetRenderbufferStorageOES(gls::gl::RENDERBUFFER, egl_image.as_ptr());
    let rbo_err = gls::gl::GetError();

    let mut fbo: u32 = 0;
    gls::gl::GenFramebuffers(1, &mut fbo);
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
    gls::gl::FramebufferRenderbuffer(
        gls::gl::FRAMEBUFFER,
        gls::gl::COLOR_ATTACHMENT0,
        gls::gl::RENDERBUFFER,
        rbo,
    );
    let rbo_status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    let rbo_complete = rbo_status == gls::gl::FRAMEBUFFER_COMPLETE && rbo_err == 0;
    let mut rbo_verified = false;
    if rbo_complete {
        rbo_verified = render_and_verify(case);
    }
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::DeleteFramebuffers(1, &fbo);
    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
    gls::gl::DeleteRenderbuffers(1, &rbo);

    if rbo_complete && rbo_verified {
        let _ = ctx.destroy_egl_image(egl_image);
        return "PASS (renderbuffer rendered + read back)".into();
    }

    // --- Route 2: TEXTURE_2D attachment ---
    let mut tex: u32 = 0;
    gls::gl::GenTextures(1, &mut tex);
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
    let _ = gls::gl::GetError();
    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, egl_image.as_ptr());
    let tex_err = gls::gl::GetError();

    let mut tfbo: u32 = 0;
    gls::gl::GenFramebuffers(1, &mut tfbo);
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, tfbo);
    gls::gl::FramebufferTexture2D(
        gls::gl::FRAMEBUFFER,
        gls::gl::COLOR_ATTACHMENT0,
        gls::gl::TEXTURE_2D,
        tex,
        0,
    );
    let tex_status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    let tex_complete = tex_status == gls::gl::FRAMEBUFFER_COMPLETE && tex_err == 0;
    let mut tex_verified = false;
    if tex_complete {
        tex_verified = render_and_verify(case);
    }
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::DeleteFramebuffers(1, &tfbo);
    gls::gl::BindTexture(gls::gl::TEXTURE_2D, 0);
    gls::gl::DeleteTextures(1, &tex);
    let _ = ctx.destroy_egl_image(egl_image);

    if tex_complete && tex_verified {
        "PASS (texture rendered + read back)".into()
    } else if tex_complete {
        "PARTIAL (texture FBO complete, readback empty)".into()
    } else {
        format!(
            "FAIL (rbo: {} {:#x} err={}; tex: {} {:#x} err={})",
            fbo_status_name(rbo_status),
            rbo_status,
            gl_err_name(rbo_err),
            fbo_status_name(tex_status),
            tex_status,
            gl_err_name(tex_err),
        )
    }
}

fn gl_err_name(e: u32) -> &'static str {
    match e {
        0 => "GL_NO_ERROR",
        0x0500 => "GL_INVALID_ENUM",
        0x0501 => "GL_INVALID_VALUE",
        0x0502 => "GL_INVALID_OPERATION",
        0x0505 => "GL_OUT_OF_MEMORY",
        0x0506 => "GL_INVALID_FRAMEBUFFER_OPERATION",
        _ => "GL_UNKNOWN",
    }
}

fn fbo_status_name(status: u32) -> &'static str {
    match status {
        gls::gl::FRAMEBUFFER_COMPLETE => "COMPLETE",
        gls::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => "INCOMPLETE_ATTACHMENT",
        gls::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => "INCOMPLETE_MISSING_ATTACHMENT",
        gls::gl::FRAMEBUFFER_UNSUPPORTED => "UNSUPPORTED",
        _ => "UNKNOWN",
    }
}

/// O2: try several DRM-fourcc tags against the same NvBufSurface fd and report
/// each, returning a one-line summary plus the best outcome.
///
/// Geometry is derived from the surface's *actual* reported width/height/pitch
/// (the driver picks its own, ignoring our requested size), so the EGLImage
/// import stays self-consistent:
///   * ABGR8888  at (surface_w, surface_h)        — matches the allocated RGBA8.
///   * ABGR16161616F at (pitch/8, surface_h)      — our packed HAL layout
///     reinterpreting the same bytes as half-float RGBA (only when pitch%8==0).
unsafe fn gl_import_render(ctx: &GpuContext, a: &AllocResult) -> String {
    if !ctx.has_rgb8_renderbuffer() {
        return "N/A (no glEGLImageTargetRenderbufferStorageOES)".into();
    }
    let pitch = a.pitch;
    let mut cases = vec![ImportCase {
        label: "ABGR8888",
        fourcc: DRM_FORMAT_ABGR8888,
        w: a.width,
        h: a.height,
        rd_format: gls::gl::RGBA,
        rd_type: gls::gl::UNSIGNED_BYTE,
        bpp: 4,
    }];
    if pitch.is_multiple_of(8) {
        cases.push(ImportCase {
            label: "ABGR16161616F",
            fourcc: DRM_FORMAT_ABGR16161616F,
            w: pitch / 8,
            h: a.height,
            rd_format: gls::gl::RGBA,
            rd_type: gls::gl::HALF_FLOAT,
            bpp: 8,
        });
    }

    let mut best = String::from("FAIL (no fourcc accepted)");
    let mut any_pass = false;
    for case in &cases {
        let _ = gls::gl::GetError(); // clear stale
        let r = gl_try_import(ctx, a.fd, pitch, case);
        println!(
            "     [{}@{}x{} pitch={}] {}",
            case.label, case.w, case.h, pitch, r
        );
        if r.starts_with("PASS") && !any_pass {
            best = format!("PASS via {}", case.label);
            any_pass = true;
        } else if r.starts_with("PARTIAL") && !any_pass {
            best = format!("PARTIAL via {}", case.label);
        }
    }
    best
}

/// Run the NvBufSurface probe (O1–O3). Gated: SKIP if not on a Jetson.
pub fn run(ctx: &GpuContext) {
    println!("=== NvBufSurface dma-buf Probe (Jetson O1-O3) ===");

    // Compile-time-ish ABI sanity: bufferDesc must sit at offset 24.
    let dummy = std::mem::MaybeUninit::<NvBufSurfaceParams>::uninit();
    let base = dummy.as_ptr() as usize;
    let bd_off = unsafe { std::ptr::addr_of!((*dummy.as_ptr()).buffer_desc) as usize } - base;
    let sl_off = std::mem::offset_of!(NvBufSurface, surface_list);
    println!("  ABI: NvBufSurfaceParams.bufferDesc offset = {bd_off} (expect 24)");
    println!("  ABI: NvBufSurface.surfaceList offset      = {sl_off} (expect 24)");
    if bd_off != 24 || sl_off != 24 {
        println!("  VERDICT_NVBUFSURFACE status=ABI_MISMATCH bufferDesc_off={bd_off} surfaceList_off={sl_off}");
        println!();
        return;
    }

    let lib = match open_lib() {
        Some(l) => l,
        None => {
            println!("  SKIP: libnvbufsurface.so not dlopen-able (not a Jetson)");
            println!("  VERDICT_NVBUFSURFACE status=SKIP reason=no_libnvbufsurface");
            println!();
            return;
        }
    };

    let create: Symbol<FnCreate> = match unsafe { lib.get(b"NvBufSurfaceCreate\0") } {
        Ok(s) => s,
        Err(e) => {
            println!("  SKIP: NvBufSurfaceCreate not found: {e}");
            println!("  VERDICT_NVBUFSURFACE status=SKIP reason=no_symbol");
            println!();
            return;
        }
    };
    let destroy: Option<Symbol<FnDestroy>> = unsafe { lib.get(b"NvBufSurfaceDestroy\0") }.ok();
    let from_fd: Option<Symbol<FnFromFd>> = unsafe { lib.get(b"NvBufSurfaceFromFd\0") }.ok();

    // -------------------------------------------------------------------
    // O1 — allocate + extract fd
    // -------------------------------------------------------------------
    // Packed RGBA16F for a 256x256-equivalent: (W/4, 3H) = (64, 768),
    // pitch = 64 * 8 = 512, byte_count = 512 * 768 = 393216.
    let (pw, ph) = (64u32, 768u32);
    let pitch_expect = pw * 8;
    let byte_count = pitch_expect * ph;

    // NOTE: the header claims a non-zero `size` makes all other params ignored,
    // but empirically NvBufSurfaceCreate still validates colorFormat (rc=-1 +
    // "invalid colorFormat 0" when set to 0). So set a valid colorFormat even
    // on the size-based path.
    let size_params = NvBufSurfaceCreateParams {
        gpu_id: 0,
        width: pw,
        height: ph,
        size: byte_count, // size set → drives the byte count
        is_contiguous: false,
        color_format: NVBUF_COLOR_FORMAT_RGBA,
        layout: NVBUF_LAYOUT_PITCH,
        mem_type: NVBUF_MEM_SURFACE_ARRAY,
    };

    println!("  -- O1: NvBufSurfaceCreate (size-based, {byte_count} bytes) --");
    let alloc = match unsafe { try_alloc(&create, &size_params, "size-raw") } {
        Ok(a) => Some(a),
        Err(e) => {
            println!("     size-based alloc failed: {e}; trying RGBA color format");
            // Fallback: real RGBA color format at width=128,height=768 → pitch 512.
            let rgba_params = NvBufSurfaceCreateParams {
                gpu_id: 0,
                width: 128,
                height: 768,
                size: 0,
                is_contiguous: false,
                color_format: NVBUF_COLOR_FORMAT_RGBA,
                layout: NVBUF_LAYOUT_PITCH,
                mem_type: NVBUF_MEM_SURFACE_ARRAY,
            };
            match unsafe { try_alloc(&create, &rgba_params, "rgba8") } {
                Ok(a) => Some(a),
                Err(e2) => {
                    println!("     RGBA alloc also failed: {e2}");
                    None
                }
            }
        }
    };

    let (o1_pass, o2_report, o3_report) = match alloc {
        None => {
            println!("  O1 FAIL: no allocation strategy succeeded");
            (
                false,
                "SKIP (no fd)".to_string(),
                "SKIP (no fd)".to_string(),
            )
        }
        Some(a) => {
            println!(
                "  O1 PASS: strategy={} fd={} surface={}x{} colorFmt={} pitch={} size={} (requested {} bytes)",
                a.strategy, a.fd, a.width, a.height, a.color_format, a.pitch, a.size, byte_count
            );

            // ---------------------------------------------------------------
            // O2 — GL import + render + readback (the core test)
            // ---------------------------------------------------------------
            println!("  -- O2: GL import of NvBufSurface fd + render --");
            let o2 = unsafe { gl_import_render(ctx, &a) };
            println!("  O2: {o2}");

            // ---------------------------------------------------------------
            // O3 — fd round-trip via NvBufSurfaceFromFd
            // ---------------------------------------------------------------
            println!("  -- O3: NvBufSurfaceFromFd round-trip --");
            let o3 = match &from_fd {
                None => "FAIL (NvBufSurfaceFromFd symbol absent)".to_string(),
                Some(f) => {
                    let mut out: *mut c_void = std::ptr::null_mut();
                    let rc = unsafe { f(a.fd, &mut out) };
                    if rc == 0 && !out.is_null() {
                        format!("PASS (rc=0, buffer={out:p})")
                    } else {
                        format!("FAIL (rc={rc}, buffer={out:p})")
                    }
                }
            };
            println!("  O3: {o3}");

            // Cleanup the surface.
            if let Some(d) = &destroy {
                let rc = unsafe { d(a.surf) };
                log::debug!("NvBufSurfaceDestroy rc={rc}");
            }

            (true, o2, o3)
        }
    };

    let o2_pass = o2_report.starts_with("PASS");
    let o3_pass = o3_report.starts_with("PASS");
    println!(
        "  VERDICT_NVBUFSURFACE status=RAN o1={} o2={} o3={} o2_detail=\"{}\" o3_detail=\"{}\"",
        yn(o1_pass),
        yn(o2_pass),
        yn(o3_pass),
        o2_report,
        o3_report,
    );
    println!();
}

fn yn(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
