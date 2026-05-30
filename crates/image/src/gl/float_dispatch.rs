// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Single source of truth for the float render-path decision.
//!
//! The "`(PixelFormat, DType, TensorMemory)` → float render path" classification
//! is a pure, GL-free decision that both platform backends must agree on. It
//! lives here — compiled on **both** Linux and macOS — so there is exactly one
//! definition of [`FloatRenderPath`] and [`classify_float_render`] across the
//! whole crate.
//!
//! * The Linux processor (`gl::processor::float`) re-exports these and dispatches
//!   on [`FloatRenderPath::PboF16Nchw`] / [`PboF32Nhwc`] / [`DmaF16Nchw`].
//! * The macOS processor (`gl::macos_processor`) targets
//!   [`FloatRenderPath::IoSurfaceF16Nchw`] (the zero-copy ANGLE/IOSurface F16
//!   path). Because a macOS IOSurface tensor reports `TensorMemory::Dma` (the
//!   IOSurface backing shares the `Dma` memory slot), the classifier cannot
//!   distinguish it from a Linux DMA-BUF destination, so macOS keeps its own
//!   platform-gated dispatch and this enum simply names that decision in the
//!   one shared place. See the module note in `macos_processor` for details.
//!
//! No GL/EGL/gbm types appear in this module — it matches purely on
//! [`edgefirst_tensor`] pixel/dtype/memory enums plus the reported
//! [`crate::RenderDtypeSupport`].
//!
//! [`PboF32Nhwc`]: FloatRenderPath::PboF32Nhwc
//! [`DmaF16Nchw`]: FloatRenderPath::DmaF16Nchw

/// Which GL float render path should be used for a given conversion.
///
/// `None` means no float GL path applies — fall through to the existing u8
/// route (which for F16/F32 destinations hits the u8 rejection and therefore
/// CPU fallback, exactly as before this seam was added).
///
/// The enum spans all platforms' float paths so the decision has one
/// definition: the PBO/DMA variants are the Linux render targets; the IOSurface
/// variant is the macOS (ANGLE) zero-copy target.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum FloatRenderPath {
    /// No GL float render path applies; fall through to existing logic.
    None,
    /// RGBA → PlanarRgb F16, PBO destination (RGBA16F-packed shader). Linux.
    PboF16Nchw,
    /// RGBA → Rgb F32, PBO destination (R32F-wide shader). Linux.
    PboF32Nhwc,
    /// RGBA → PlanarRgb F16, DMA-buf destination (zero-copy F16 render via
    /// `convert_float_to_dma`). Linux.
    DmaF16Nchw,
    /// RGBA → PlanarRgb F16, IOSurface destination (zero-copy F16 render via
    /// ANGLE's `iosurface_client_buffer`). macOS.
    ///
    /// Only constructed/consumed on macOS; on Linux the variant exists purely
    /// so the enum names every platform's float path in one place, hence the
    /// targeted dead-code allow on non-macOS targets.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    IoSurfaceF16Nchw,
}

/// Classify whether a conversion should use a GL float render target.
///
/// Gated on source/destination pixel format, destination dtype, destination
/// memory kind, and the float-render capability reported by the current GPU.
/// Returns [`FloatRenderPath::None`] when the combination is not supported so
/// callers can fall through to the existing u8 path.
///
/// This is the single definition of the float-path decision for the
/// host-memory-discriminated paths: the Linux PBO/DMA targets (`Pbo*` /
/// `Dma*`) returned for [`TensorMemory::Pbo`] / [`TensorMemory::Dma`]
/// destinations.
///
/// [`FloatRenderPath::IoSurfaceF16Nchw`] is intentionally **not** returned
/// here: a macOS IOSurface tensor reports `TensorMemory::Dma` (IOSurface and
/// Linux DMA-BUF share the same `TensorMemory::Dma` slot — see
/// [`edgefirst_tensor::TensorMemory::Dma`]), so it is indistinguishable from a
/// Linux DMA-BUF destination by the classifier inputs alone. The macOS backend
/// therefore keeps its own platform-gated dispatch and uses
/// `IoSurfaceF16Nchw` as the conceptual name for that decision; this shared
/// enum is the one place that decision is named for every platform.
///
/// [`TensorMemory::Pbo`]: edgefirst_tensor::TensorMemory::Pbo
/// [`TensorMemory::Dma`]: edgefirst_tensor::TensorMemory::Dma
pub(super) fn classify_float_render(
    src: edgefirst_tensor::PixelFormat,
    dst: edgefirst_tensor::PixelFormat,
    dtype: edgefirst_tensor::DType,
    dst_mem: edgefirst_tensor::TensorMemory,
    support: crate::RenderDtypeSupport,
) -> FloatRenderPath {
    use edgefirst_tensor::{DType, PixelFormat::*, TensorMemory};
    match (src, dst, dtype, dst_mem) {
        (Rgba, PlanarRgb, DType::F16, TensorMemory::Pbo) if support.f16 => {
            FloatRenderPath::PboF16Nchw
        }
        (Rgba, PlanarRgb, DType::F16, TensorMemory::Dma) if support.f16 => {
            FloatRenderPath::DmaF16Nchw
        }
        (Rgba, Rgb, DType::F32, TensorMemory::Pbo) if support.f32 => FloatRenderPath::PboF32Nhwc,
        _ => FloatRenderPath::None,
    }
}
