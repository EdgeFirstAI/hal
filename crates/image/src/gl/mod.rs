// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(target_os = "linux", target_os = "macos"))]
#![cfg(feature = "opengl")]
// Several types defined at the `gl` module root (EglDisplayKind,
// TransferBackend, RegionOfInterest, etc.) are consumed only by the
// Linux-only inner modules (`context`, `processor`, ...). The macOS
// path uses its own `MacosGlProcessor` + `iosurface_import` modules and
// does not touch every shared type, so some appear unused on macOS.
// Rather than fragmenting the type definitions per platform, suppress
// the dead-code lint on non-Linux targets.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

// Module layout:
//   - `platform/` — cross-platform display/EGL-loader seam (both OSes)
//   - Linux-only:  `context`, `processor`, `threaded`, `dma_import`,
//                  `cache`, `resources`, `shaders`, `tests`
//   - macOS-only:  `iosurface_import`, `macos_processor`
// The macOS processor is parallel to (not a refactor of) the Linux
// threaded processor — see `crates/image/ARCHITECTURE.md` for the
// rationale and the planned convergence story.

#[cfg(target_os = "linux")]
macro_rules! function {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);

        // Find and cut the rest of the path
        match &name[..name.len() - 3].rfind(':') {
            Some(pos) => &name[pos + 1..name.len() - 3],
            None => &name[..name.len() - 3],
        }
    }};
}

#[cfg(target_os = "linux")]
mod cache;
#[cfg(target_os = "linux")]
mod context;
#[cfg(target_os = "linux")]
mod dma_import;
#[cfg(target_os = "macos")]
mod iosurface_import;
#[cfg(target_os = "macos")]
mod macos_processor;
mod platform;
#[cfg(target_os = "linux")]
mod processor;
#[cfg(target_os = "linux")]
mod resources;
#[cfg(target_os = "linux")]
mod shaders;
#[cfg(target_os = "linux")]
mod tests;
#[cfg(target_os = "linux")]
mod threaded;

#[cfg(target_os = "linux")]
pub use context::probe_egl_displays;
// These are accessed by sibling sub-modules via `super::context::` directly.
// No re-export needed at the mod.rs level.
#[cfg(target_os = "macos")]
pub use macos_processor::MacosGlProcessor;
#[cfg(target_os = "linux")]
pub use threaded::GLProcessorThreaded;

/// Dynamically-loaded EGL 1.4 instance. The lifetime parameter is
/// `'static` because the underlying `libloading::Library` is intentionally
/// leaked at first load (see `EGL_LIB` in `context.rs` and the equivalent
/// on macOS — drivers may retain internal state past explicit cleanup, so
/// dlclose can SIGBUS on process exit).
///
/// Defined here at the `gl` module root so the `platform/` trait and both
/// platform implementations can name it without dragging in a cross-cfg
/// re-export. The Linux `context.rs` and the macOS `platform/macos.rs`
/// both use this same alias.
pub(super) type Egl =
    khronos_egl::Instance<khronos_egl::Dynamic<&'static libloading::Library, khronos_egl::EGL1_4>>;

/// Identifies the type of EGL display used for headless OpenGL ES rendering.
///
/// The HAL creates a surfaceless GLES 3.0 context
/// (`EGL_KHR_surfaceless_context` + `EGL_KHR_no_config_context`) and
/// renders exclusively through FBOs backed by EGLImages imported from
/// DMA-buf file descriptors. No window or PBuffer surface is created.
///
/// Displays are probed in priority order: PlatformDevice first (zero
/// external dependencies), then GBM, then Default. Use
/// [`probe_egl_displays`] to discover which are available and
/// [`ImageProcessorConfig::egl_display`](crate::ImageProcessorConfig::egl_display)
/// to override the auto-detection.
///
/// # Display Types
///
/// - **`PlatformDevice`** — Uses `EGL_EXT_device_enumeration` to query
///   available EGL devices via `eglQueryDevicesEXT`, then selects the first
///   device with `eglGetPlatformDisplay(EGL_EXT_platform_device, ...)`.
///   Headless and compositor-free with zero external library dependencies.
///   Works on NVIDIA GPUs and newer Vivante drivers.
///
/// - **`Gbm`** — Opens a DRM render node (e.g. `/dev/dri/renderD128`) and
///   creates a GBM (Generic Buffer Manager) device, then calls
///   `eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm_device)`. Requires
///   `libgbm` and a DRM render node. Needed on ARM Mali (i.MX95) and older
///   Vivante drivers that do not expose `EGL_EXT_platform_device`.
///
/// - **`Default`** — Calls `eglGetDisplay(EGL_DEFAULT_DISPLAY)`, letting the
///   EGL implementation choose the display. On Wayland systems this connects
///   to the compositor; on X11 it connects to the X server. May block on
///   headless systems where a compositor is expected but not running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EglDisplayKind {
    Gbm,
    PlatformDevice,
    Default,
}

impl std::fmt::Display for EglDisplayKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EglDisplayKind::Gbm => write!(f, "GBM"),
            EglDisplayKind::PlatformDevice => write!(f, "PlatformDevice"),
            EglDisplayKind::Default => write!(f, "Default"),
        }
    }
}

/// A validated, available EGL display discovered by [`probe_egl_displays`].
#[derive(Debug, Clone)]
pub struct EglDisplayInfo {
    /// The type of EGL display.
    pub kind: EglDisplayKind,
    /// Human-readable description for logging/diagnostics
    /// (e.g. "GBM via /dev/dri/renderD128").
    pub description: String,
}

/// Tracks which data-transfer method is active for moving pixels
/// between CPU memory and GPU textures/framebuffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferBackend {
    /// Zero-copy via EGLImage imported from DMA-buf file descriptors.
    /// Available on i.MX8 (Vivante), i.MX95 (Mali), Jetson, and any
    /// platform where `EGL_EXT_image_dma_buf_import` is present AND
    /// the GPU can actually render through DMA-buf-backed textures.
    DmaBuf,

    /// Zero-copy via `EGL_ANGLE_iosurface_client_buffer` (macOS).
    /// Available when ANGLE's Metal backend is loaded and the EGL
    /// extension is advertised. The IOSurface is wrapped as an EGL
    /// pbuffer and bound to a 2D texture via `eglBindTexImage`.
    IOSurface,

    /// GPU buffer via Pixel Buffer Object. Used when DMA-buf is unavailable
    /// but OpenGL is present. Data stays in GPU-accessible memory.
    Pbo,

    /// Synchronous `glTexSubImage2D` upload + `glReadnPixels` readback.
    /// Used when DMA-buf is unavailable or when the DMA-buf verification
    /// probe fails (e.g. NVIDIA discrete GPUs where EGLImage creation
    /// succeeds but rendered data is all zeros).
    Sync,
}

impl TransferBackend {
    /// Returns `true` if DMA-buf zero-copy is available.
    pub(crate) fn is_dma(self) -> bool {
        self == TransferBackend::DmaBuf
    }
}

/// Interpolation mode for int8 proto textures (GL_R8I cannot use GL_LINEAR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8InterpolationMode {
    /// texelFetch at nearest texel — simplest, fastest GPU execution.
    Nearest,
    /// texelFetch × 4 neighbors with shader-computed bilinear weights (default).
    Bilinear,
    /// Two-pass: dequant int8→f16 FBO, then existing f16 shader with GL_LINEAR.
    TwoPass,
}

/// A rectangular region of interest expressed as normalised [0, 1] coordinates.
#[derive(Debug, Clone, Copy)]
pub(super) struct RegionOfInterest {
    pub(super) left: f32,
    pub(super) top: f32,
    pub(super) right: f32,
    pub(super) bottom: f32,
}

impl RegionOfInterest {
    /// Build a source ROI from a pixel-space crop rectangle with a half-texel
    /// inset. The inset ensures that `GL_LINEAR` filtering never samples
    /// outside the crop boundary — at the extreme texture coordinates the
    /// bilinear kernel is centred on the boundary texel and cannot reach
    /// adjacent padding pixels.
    ///
    /// The result is clamped to [0, 1] so an out-of-bounds crop rectangle
    /// cannot produce invalid texture coordinates.
    ///
    /// `crop`: pixel-space rectangle (left, top, width, height).
    /// `tex_w`, `tex_h`: full texture dimensions in pixels.
    pub(super) fn from_crop_clamped(crop: &crate::Rect, tex_w: usize, tex_h: usize) -> Self {
        let half_x = 0.5 / tex_w as f32;
        let half_y = 0.5 / tex_h as f32;
        RegionOfInterest {
            left: (crop.left as f32 / tex_w as f32 + half_x).clamp(0.0, 1.0),
            top: ((crop.top + crop.height) as f32 / tex_h as f32 - half_y).clamp(0.0, 1.0),
            right: ((crop.left + crop.width) as f32 / tex_w as f32 - half_x).clamp(0.0, 1.0),
            bottom: (crop.top as f32 / tex_h as f32 + half_y).clamp(0.0, 1.0),
        }
    }
}
