// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "android",
    target_os = "windows"
))]
#![cfg(feature = "opengl")]
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

// Module layout:
//   - `platform/` — cross-platform display/EGL-loader seam (all OSes)
//   - Linux-only:   `context`, `dma_import`, `fourcc`
//   - macOS/iOS:    `iosurface_import`
//   - Android-only: `ahardwarebuffer_import`
//   - Linux and Android: `native_fence`
//   - Windows:      `d3d11_import`, `platform/windows.rs` (ANGLE/D3D11)
// The engine (`processor`, `threaded`, `cache`, `resources`, `shaders`)
// reaches platform buffers only through the `GlPlatform` trait.

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
// CPU wake-latency request held around GL work.
#[cfg(target_os = "linux")]
mod context;
mod cpu_latency;
#[cfg(target_os = "linux")]
mod dma_import;
// Portable renderer helpers and the float render-path classifier.
mod core;
mod float_dispatch;
// Decision table for the proto-segmentation render path.
mod proto_dispatch;
// Portable GL render lowering (y-flip viewport, source UV, batch chunk planner).
mod render;
// AHardwareBuffer EGLImage import.
#[cfg(target_os = "android")]
mod ahardwarebuffer_import;
// CUDA test policy.
#[cfg(test)]
mod cuda_policy;
// Adreno `msm_gbm.so` mapping bookkeeping.
#[cfg(target_os = "linux")]
mod adreno_unmap;
// `EGL_ANGLE_image_d3d11_texture` attribute assembly.
#[cfg(target_os = "windows")]
pub(super) mod d3d11_import;
#[cfg(target_os = "linux")]
mod fourcc;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod iosurface_import;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod native_fence;
mod platform;
mod processor;
mod resources;
mod shaders;
mod shaders_common;
// Engine GL tests; per-item cfg gates inside the modules select what runs
// on each platform.
mod norm_tests;
mod tests;
mod threaded;

#[cfg(target_os = "linux")]
pub use context::probe_egl_displays;
// Cache statistics and the threaded processor.
pub use cache::{CacheStats, ConvertStats, GlCacheStats};
pub use threaded::GLProcessorThreaded;

/// Dynamically-loaded EGL 1.4 instance over the leaked, `'static`
/// `libloading::Library`.
pub(super) type Egl = edgefirst_egl::Instance<
    edgefirst_egl::Dynamic<&'static libloading::Library, edgefirst_egl::EGL1_4>,
>;

/// Handle returned by `GlPlatform::export_completion_fence` (and surfaced
/// by `convert_with_fence`): a sync_file fd on Linux and Android
/// (`EGL_ANDROID_native_fence_sync`), an event NT handle on Windows, set
/// when the D3D11 device fence reaches the value the convert queued.
pub use crate::CompletionFence;

/// Identifies the type of EGL display used for headless OpenGL ES rendering.
///
/// The HAL creates a surfaceless GLES 3.0 context
/// (`EGL_KHR_surfaceless_context` + `EGL_KHR_no_config_context`) and
/// renders exclusively through FBOs backed by EGLImages imported from
/// DMA-buf file descriptors. No window or PBuffer surface is created.
///
/// Displays are probed in priority order: PlatformDevice first, then GBM,
/// then Default. Use [`probe_egl_displays`] to discover which are available
/// and
/// [`ImageProcessorConfig::egl_display`](crate::ImageProcessorConfig::egl_display)
/// to override the auto-detection.
///
/// # Display Types
///
/// - **`PlatformDevice`** — Uses `EGL_EXT_device_enumeration` to query
///   available EGL devices via `eglQueryDevicesEXT`, then selects the first
///   device with `eglGetPlatformDisplay(EGL_EXT_platform_device, ...)`.
///   Headless and compositor-free with no external library dependencies.
///   Available on NVIDIA GPUs and newer Vivante drivers. Skipped on Qualcomm
///   Adreno, where GBM is used.
///
/// - **`Gbm`** — Opens a DRM render node (e.g. `/dev/dri/renderD128`) and
///   creates a GBM (Generic Buffer Manager) device, then calls
///   `eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm_device)`. Requires
///   `libgbm` and a DRM render node. Used on ARM Mali (i.MX95) and older
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
    /// Zero-copy via EGLImage imported from DMA-buf file descriptors
    /// (`EGL_EXT_image_dma_buf_import`).
    DmaBuf,

    /// Zero-copy via `EGL_ANGLE_iosurface_client_buffer` (macOS/iOS). The
    /// IOSurface is wrapped as an EGL pbuffer and bound to a 2D texture via
    /// `eglBindTexImage`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    IOSurface,

    /// Zero-copy via `EGL_ANDROID_image_native_buffer` (Android). The
    /// AHardwareBuffer is wrapped as an EGLImage via
    /// `eglGetNativeClientBufferANDROID` → `eglCreateImageKHR` and bound
    /// to a 2D texture via `glEGLImageTargetTexture2DOES`.
    #[cfg(target_os = "android")]
    AHardwareBuffer,

    /// Zero-copy via `EGL_ANGLE_image_d3d11_texture` (Windows): the tensor's
    /// `ID3D11Texture2D` wrapped as an EGLImage and bound with
    /// `glEGLImageTargetTexture2DOES` on every use.
    #[cfg(target_os = "windows")]
    D3d11Texture,

    /// GPU buffer via Pixel Buffer Object. Used when DMA-buf is unavailable
    /// but OpenGL is present. Data stays in GPU-accessible memory.
    Pbo,

    /// Synchronous `glTexSubImage2D` upload + `glReadPixels` readback.
    /// Used when DMA-buf is unavailable or when the DMA-buf verification
    /// probe fails.
    Sync,
}

impl TransferBackend {
    /// Returns `true` if DMA-buf zero-copy is available.
    pub(crate) fn is_dma(self) -> bool {
        self == TransferBackend::DmaBuf
    }

    /// Returns `true` if the platform can import `TensorMemory::DmaBuf`
    /// tensors zero-copy: DMA-BUF EGLImages on Linux, IOSurface pbuffers on
    /// macOS/iOS, AHardwareBuffer EGLImages on Android, D3D11 textures on
    /// Windows.
    pub(crate) fn is_zero_copy(self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self == TransferBackend::IOSurface {
            return true;
        }
        #[cfg(target_os = "android")]
        if self == TransferBackend::AHardwareBuffer {
            return true;
        }
        #[cfg(target_os = "windows")]
        if self == TransferBackend::D3d11Texture {
            return true;
        }
        self == TransferBackend::DmaBuf
    }
}

/// Interpolation mode for int8 proto textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8InterpolationMode {
    /// texelFetch at nearest texel.
    Nearest,
    /// texelFetch × 4 neighbors with shader-computed bilinear weights (default).
    Bilinear,
    /// Two-pass: dequant int8→f16 FBO, then the f16 shader with GL_LINEAR.
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
    /// Build a source ROI spanning a pixel-space crop rectangle's true edges,
    /// clamped to [0, 1].
    ///
    /// `crop`: pixel-space rectangle (left, top, width, height).
    /// `tex_w`, `tex_h`: full texture dimensions in pixels.
    pub(super) fn from_crop(crop: &crate::Rect, tex_w: usize, tex_h: usize) -> Self {
        RegionOfInterest {
            left: (crop.left as f32 / tex_w as f32).clamp(0.0, 1.0),
            top: ((crop.top + crop.height) as f32 / tex_h as f32).clamp(0.0, 1.0),
            right: ((crop.left + crop.width) as f32 / tex_w as f32).clamp(0.0, 1.0),
            bottom: (crop.top as f32 / tex_h as f32).clamp(0.0, 1.0),
        }
    }

    /// `[u_min, v_min, u_max, v_max]` of this ROI, whichever way it is flipped.
    pub(super) fn uv_bounds(&self) -> [f32; 4] {
        [
            self.left.min(self.right),
            self.top.min(self.bottom),
            self.left.max(self.right),
            self.top.max(self.bottom),
        ]
    }
}

#[cfg(all(test, target_os = "windows"))]
mod d3d11_backend_tests {
    use super::TransferBackend;

    #[test]
    fn d3d11_texture_is_zero_copy_and_not_dma() {
        assert!(TransferBackend::D3d11Texture.is_zero_copy());
        assert!(!TransferBackend::D3d11Texture.is_dma());
    }
}
