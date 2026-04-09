// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]
#![cfg(feature = "opengl")]

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

mod cache;
mod context;
mod dma_import;
mod processor;
mod resources;
mod shaders;
mod tests;
mod threaded;

pub use context::probe_egl_displays;
// These are accessed by sibling sub-modules via `super::context::` directly.
// No re-export needed at the mod.rs level.
pub use threaded::GLProcessorThreaded;

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
