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
//! render targets: a macOS IOSurface tensor reports `TensorMemory::DmaBuf`
//! (IOSurface shares the `Dma` memory slot), so the same
//! `(Rgba, PlanarRgb, F16, Dma)` tuple that selects the Linux DMA-BUF render
//! selects the IOSurface render on macOS — which buffer object backs the
//! render is the platform seam's business (`GlPlatform::import_buffer`),
//! not the classifier's. The unified engine runs on both platforms, and
//! this classifier is its single float-path dispatch.
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
    /// macOS (both report `TensorMemory::DmaBuf`).
    ZeroCopyF16Nchw,
    /// RGBA → PlanarRgb / PlanarRgba F32 into a zero-copy texture: the same
    /// packed program as F16 rendering into an RGBA32F attachment (Windows).
    ZeroCopyF32Nchw,
    /// RGBA → Rgb F16 / F32 interleaved into a zero-copy texture of
    /// (W*3/4, H) RGBA16F / RGBA32F texels (Windows).
    ZeroCopyFloatNhwc,
    /// RGBA → Rgba F16 / F32 into a W x H float texture (Windows).
    ZeroCopyFloatRgba,
}

/// Which zero-copy float render paths a platform leaf is known to serve.
///
/// The three variants past `ZeroCopyF16Nchw`, and the `PlanarRgba` widening
/// of `ZeroCopyF16Nchw` itself, are shaders this branch wrote and ran on
/// Windows only. Every one of those tuples used to classify as
/// [`FloatRenderPath::None`] on Linux, macOS and Android and fall through to
/// the CPU converter that produced those platforms' reference output, and
/// they still do: the leaf says what it has run, and the classifier reads it.
///
/// This is a capability, not a `cfg`: the classifier stays pure and testable
/// against both sets, and widening a leaf becomes a one-line change backed by
/// an on-target run.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum ZeroCopyFloatSet {
    /// `Rgba -> PlanarRgb` F16 only, the path Linux, macOS and Android have
    /// always taken.
    PlanarF16,
    /// Every zero-copy float destination the packed shaders cover: planar
    /// F16 and F32 (`PlanarRgb` and `PlanarRgba`), interleaved `Rgb`, and
    /// `Rgba`. Validated on Windows.
    All,
}

/// Classify whether a conversion should use a GL float render target.
///
/// Gated on source/destination pixel format, destination dtype, destination
/// memory kind, and the float-render capability reported by the current GPU.
/// Returns [`FloatRenderPath::None`] when the combination is not supported so
/// callers can fall through to the existing u8 path.
///
/// This is the single definition of the float-path decision. A
/// [`TensorMemory::DmaBuf`] destination means "the platform's zero-copy GPU
/// buffer" — a DMA-BUF on Linux, an IOSurface on macOS (they share the
/// `Dma` slot — see [`edgefirst_tensor::TensorMemory::DmaBuf`]); the platform
/// seam, not this classifier, resolves which import backs the render.
///
/// [`TensorMemory::Pbo`]: edgefirst_tensor::TensorMemory::Pbo
/// [`TensorMemory::DmaBuf`]: edgefirst_tensor::TensorMemory::DmaBuf
pub(super) fn classify_float_render(
    src: edgefirst_tensor::PixelFormat,
    dst: edgefirst_tensor::PixelFormat,
    dtype: edgefirst_tensor::DType,
    dst_mem: edgefirst_tensor::TensorMemory,
    support: crate::RenderDtypeSupport,
    zero_copy: ZeroCopyFloatSet,
) -> FloatRenderPath {
    use edgefirst_tensor::{DType, PixelFormat::*, TensorMemory};
    let all = zero_copy == ZeroCopyFloatSet::All;
    match (src, dst, dtype, dst_mem) {
        (Rgba, PlanarRgb, DType::F16, TensorMemory::Pbo) if support.f16 => {
            FloatRenderPath::PboF16Nchw
        }
        (Rgba, PlanarRgb, DType::F16, TensorMemory::DmaBuf) if support.f16 => {
            FloatRenderPath::ZeroCopyF16Nchw
        }
        (Rgba, PlanarRgba, DType::F16, TensorMemory::DmaBuf) if all && support.f16 => {
            FloatRenderPath::ZeroCopyF16Nchw
        }
        (Rgba, PlanarRgb | PlanarRgba, DType::F32, TensorMemory::DmaBuf) if all && support.f32 => {
            FloatRenderPath::ZeroCopyF32Nchw
        }
        (Rgba, Rgb, DType::F16, TensorMemory::DmaBuf) if all && support.f16 => {
            FloatRenderPath::ZeroCopyFloatNhwc
        }
        (Rgba, Rgb, DType::F32, TensorMemory::DmaBuf) if all && support.f32 => {
            FloatRenderPath::ZeroCopyFloatNhwc
        }
        (Rgba, Rgba, DType::F16, TensorMemory::DmaBuf) if all && support.f16 => {
            FloatRenderPath::ZeroCopyFloatRgba
        }
        (Rgba, Rgba, DType::F32, TensorMemory::DmaBuf) if all && support.f32 => {
            FloatRenderPath::ZeroCopyFloatRgba
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
    use super::{classify_float_render, FloatRenderPath, ZeroCopyFloatSet};
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
    /// What Linux, macOS and Android report.
    const PLANAR: ZeroCopyFloatSet = ZeroCopyFloatSet::PlanarF16;
    /// What the Windows leaf reports.
    const ALL: ZeroCopyFloatSet = ZeroCopyFloatSet::All;

    #[test]
    fn pbo_paths_do_not_depend_on_the_zero_copy_set() {
        for set in [PLANAR, ALL] {
            assert_eq!(
                classify_float_render(
                    PixelFormat::Rgba,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    TensorMemory::Pbo,
                    YES,
                    set,
                ),
                FloatRenderPath::PboF16Nchw
            );
            assert_eq!(
                classify_float_render(
                    PixelFormat::Rgba,
                    PixelFormat::Rgb,
                    DType::F32,
                    TensorMemory::Pbo,
                    YES,
                    set,
                ),
                FloatRenderPath::PboF32Nhwc
            );
        }
    }

    #[test]
    fn the_planar_f16_render_is_live_in_both_capability_sets() {
        for set in [PLANAR, ALL] {
            assert_eq!(
                classify_float_render(
                    PixelFormat::Rgba,
                    PixelFormat::PlanarRgb,
                    DType::F16,
                    TensorMemory::DmaBuf,
                    YES,
                    set,
                ),
                FloatRenderPath::ZeroCopyF16Nchw
            );
        }
    }

    #[test]
    fn no_path_when_capability_absent() {
        // The format/dtype/memory tuple matches, but the GPU does not report
        // the corresponding float capability -> the guard fails -> None.
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo,
                NO,
                ALL,
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Pbo,
                NO,
                ALL,
            ),
            FloatRenderPath::None
        );
    }

    #[test]
    fn no_path_for_unhandled_tuples() {
        // Non-Rgba source, integer dtype, and host-memory destination all fall
        // through to the catch-all None arm.
        let c = |src, dst, dt, mem| classify_float_render(src, dst, dt, mem, YES, ALL);
        assert_eq!(
            c(
                PixelFormat::Bgra,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Pbo
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            c(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::U8,
                TensorMemory::Pbo
            ),
            FloatRenderPath::None
        );
        assert_eq!(
            c(
                PixelFormat::Rgba,
                PixelFormat::PlanarRgb,
                DType::F16,
                TensorMemory::Mem
            ),
            FloatRenderPath::None
        );
    }

    /// The Windows leaf's set: every zero-copy float destination the packed
    /// shaders cover.
    #[test]
    fn the_full_set_covers_every_float_layout() {
        use FloatRenderPath::*;
        let c = |dst, dt| {
            classify_float_render(PixelFormat::Rgba, dst, dt, TensorMemory::DmaBuf, YES, ALL)
        };
        assert_eq!(c(PixelFormat::PlanarRgb, DType::F16), ZeroCopyF16Nchw);
        assert_eq!(c(PixelFormat::PlanarRgba, DType::F16), ZeroCopyF16Nchw);
        assert_eq!(c(PixelFormat::PlanarRgb, DType::F32), ZeroCopyF32Nchw);
        assert_eq!(c(PixelFormat::PlanarRgba, DType::F32), ZeroCopyF32Nchw);
        assert_eq!(c(PixelFormat::Rgb, DType::F16), ZeroCopyFloatNhwc);
        assert_eq!(c(PixelFormat::Rgb, DType::F32), ZeroCopyFloatNhwc);
        assert_eq!(c(PixelFormat::Rgba, DType::F16), ZeroCopyFloatRgba);
        assert_eq!(c(PixelFormat::Rgba, DType::F32), ZeroCopyFloatRgba);
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::DmaBuf,
                NO,
                ALL,
            ),
            None
        );
        assert_eq!(
            classify_float_render(
                PixelFormat::Rgba,
                PixelFormat::Rgb,
                DType::F32,
                TensorMemory::Mem,
                YES,
                ALL,
            ),
            None
        );
    }

    /// The set Linux, macOS and Android report: the planar F16 render and
    /// nothing else. Every other tuple falls through to the CPU converter
    /// those platforms' reference output came from, which is what it did
    /// before the packed float shaders were written.
    #[test]
    fn the_planar_set_leaves_the_windows_only_shaders_unreachable() {
        let c = |dst, dt| {
            classify_float_render(
                PixelFormat::Rgba,
                dst,
                dt,
                TensorMemory::DmaBuf,
                YES,
                PLANAR,
            )
        };
        for (dst, dt) in [
            (PixelFormat::PlanarRgba, DType::F16),
            (PixelFormat::PlanarRgb, DType::F32),
            (PixelFormat::PlanarRgba, DType::F32),
            (PixelFormat::Rgb, DType::F16),
            (PixelFormat::Rgb, DType::F32),
            (PixelFormat::Rgba, DType::F16),
            (PixelFormat::Rgba, DType::F32),
        ] {
            assert_eq!(
                c(dst, dt),
                FloatRenderPath::None,
                "{dst:?}/{dt:?} must stay on the CPU converter here"
            );
        }
    }
}
