// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The float render-path decision.
//!
//! [`classify_float_render`] maps `(PixelFormat, DType, TensorMemory)` and
//! the reported [`crate::RenderDtypeSupport`] to a [`FloatRenderPath`]. It is
//! GL-free and compiled on every platform.
//!
//! [`ZeroCopyF16Nchw`] covers every platform's zero-copy F16 render target:
//! each platform's zero-copy buffer reports `TensorMemory::DmaBuf`.
//!
//! [`ZeroCopyF16Nchw`]: FloatRenderPath::ZeroCopyF16Nchw

/// Which GL float render path is used for a given conversion.
///
/// `None` means no float GL path applies and the caller falls through to the
/// u8 route.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum FloatRenderPath {
    /// No GL float render path applies.
    None,
    /// RGBA → PlanarRgb F16, PBO destination (RGBA16F-packed shader). Linux.
    PboF16Nchw,
    /// RGBA → Rgb F32, PBO destination (R32F-wide shader). Linux.
    PboF32Nhwc,
    /// RGBA → PlanarRgb F32, PBO destination (the planar packer into an
    /// RGBA32F surface).
    PboF32Nchw,
    /// RGBA → Rgb F16, PBO destination (the interleaved packer into an
    /// RGBA16F surface).
    PboF16Nhwc,
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

/// Which zero-copy float render paths a platform leaf serves.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) enum ZeroCopyFloatSet {
    /// `Rgba -> PlanarRgb` F16 only.
    PlanarF16,
    /// Every zero-copy float destination the packed shaders cover: planar
    /// F16 and F32 (`PlanarRgb` and `PlanarRgba`), interleaved `Rgb`, and
    /// `Rgba`.
    All,
}

/// Classify whether a conversion uses a GL float render target.
///
/// Decides from the source/destination pixel format, destination dtype,
/// destination memory kind, the float-render capability reported by the GPU
/// and the platform's zero-copy float set. Returns [`FloatRenderPath::None`]
/// when the combination is not supported.
///
/// A [`TensorMemory::DmaBuf`] destination is the platform's zero-copy GPU
/// buffer.
///
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
        (Rgba, PlanarRgb, DType::F32, TensorMemory::Pbo) if support.f32 => {
            FloatRenderPath::PboF32Nchw
        }
        (Rgba, Rgb, DType::F16, TensorMemory::Pbo) if support.f16 => FloatRenderPath::PboF16Nhwc,
        _ => FloatRenderPath::None,
    }
}

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
        // The tuple matches but the GPU does not report the float capability.
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
        // Non-Rgba source, integer dtype, and host-memory destination classify
        // as None.
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

    /// The full set covers every zero-copy float destination the packed shaders cover.
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

    /// The planar set classifies only the planar F16 render; every other tuple is `None`.
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
