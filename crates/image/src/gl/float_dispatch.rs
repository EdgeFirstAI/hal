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
//! The Linux processor (`gl::processor::float`) re-exports these and dispatches
//! on [`FloatRenderPath::PboF16Nchw`] / [`PboF32Nhwc`] / [`ZeroCopyF16Nchw`].
//!
//! [`ZeroCopyF16Nchw`] deliberately covers BOTH platforms' zero-copy F16
//! render targets: a macOS IOSurface tensor reports `TensorMemory::Dma`
//! (IOSurface shares the `Dma` memory slot), so the same
//! `(Rgba, PlanarRgb, F16, Dma)` tuple that selects the Linux DMA-BUF render
//! selects the IOSurface render once the engine runs on macOS — which buffer
//! object backs the render is the platform seam's business
//! (`GlPlatform::import_buffer`), not the classifier's. Until the legacy
//! `macos_processor` is deleted, its inline `(src, dst, dtype)` match remains
//! the macOS dispatch; this classifier is the single definition both
//! converge on.
//!
//! No GL/EGL/gbm types appear in this module — it matches purely on
//! [`edgefirst_tensor`] pixel/dtype/memory enums plus the reported
//! [`crate::RenderDtypeSupport`].
//!
//! [`PboF32Nhwc`]: FloatRenderPath::PboF32Nhwc
//! [`ZeroCopyF16Nchw`]: FloatRenderPath::ZeroCopyF16Nchw

/// Which GL float render path should be used for a given conversion.
///
/// `None` means no float GL path applies — fall through to the existing u8
/// route (which for F16/F32 destinations hits the u8 rejection and therefore
/// CPU fallback, exactly as before this seam was added).
///
/// These are the host-memory-discriminated render targets the classifier can
/// decide from its inputs alone. `ZeroCopyF16Nchw` covers both platforms'
/// zero-copy F16 targets (DMA-BUF and IOSurface) — see the module docs.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum FloatRenderPath {
    /// No GL float render path applies; fall through to existing logic.
    None,
    /// RGBA → PlanarRgb F16, PBO destination (RGBA16F-packed shader). Linux.
    PboF16Nchw,
    /// RGBA → Rgb F32, PBO destination (R32F-wide shader). Linux.
    PboF32Nhwc,
    /// RGBA → PlanarRgb F16 into a zero-copy GPU buffer destination via
    /// `convert_float_to_zero_copy` — a DMA-BUF on Linux, an IOSurface on
    /// macOS (both report `TensorMemory::Dma`).
    ZeroCopyF16Nchw,
}

/// Classify whether a conversion should use a GL float render target.
///
/// Gated on source/destination pixel format, destination dtype, destination
/// memory kind, and the float-render capability reported by the current GPU.
/// Returns [`FloatRenderPath::None`] when the combination is not supported so
/// callers can fall through to the existing u8 path.
///
/// This is the single definition of the float-path decision. A
/// [`TensorMemory::Dma`] destination means "the platform's zero-copy GPU
/// buffer" — a DMA-BUF on Linux, an IOSurface on macOS (they share the
/// `Dma` slot — see [`edgefirst_tensor::TensorMemory::Dma`]); the platform
/// seam, not this classifier, resolves which import backs the render.
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
            FloatRenderPath::ZeroCopyF16Nchw
        }
        (Rgba, Rgb, DType::F32, TensorMemory::Pbo) if support.f32 => FloatRenderPath::PboF32Nhwc,
        _ => FloatRenderPath::None,
    }
}

// Shared (NOT cfg(target_os)) so the classifier is exercised on the macOS
// coverage lane too — the `gl::tests` `dispatch_*` tests are Linux-only, so
// without this `classify_float_render` is compiled-but-untested on macOS.
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{classify_float_render, FloatRenderPath};
    use crate::RenderDtypeSupport;
    use edgefirst_tensor::{DType, PixelFormat, TensorMemory};

    const YES: RenderDtypeSupport = RenderDtypeSupport {
        f32: true,
        f16: true,
    };
    const NO: RenderDtypeSupport = RenderDtypeSupport {
        f32: false,
        f16: false,
    };

    #[test]
    fn pbo_f16_when_supported() {
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo,
                YES
            ),
            FloatRenderPath::PboF16Nchw
        );
    }

    #[test]
    fn dma_f16_when_supported() {
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Dma,
                YES
            ),
            FloatRenderPath::ZeroCopyF16Nchw
        );
    }

    #[test]
    fn pbo_f32_when_supported() {
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Pbo,
                YES
            ),
            FloatRenderPath::PboF32Nhwc
        );
    }

    #[test]
    fn no_path_when_capability_absent() {
        // The format/dtype/memory tuple matches, but the GPU does not report
        // the corresponding float capability → the guard fails → None.
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo,
                NO
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Pbo,
                NO
            ),
            FloatRenderPath::None
        );
    }

    #[test]
    fn no_path_for_unhandled_tuples() {
        // Non-Rgba source, integer dtype, and host-memory destination all fall
        // through to the catch-all None arm.
        assert_eq!(
            classify_float_render(
                PixelFormat::Bgra,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo,
                YES
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::U8,
                TensorMemory::Pbo,
                YES
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Mem,
                YES
            ),
            FloatRenderPath::None
        );
    }
}
