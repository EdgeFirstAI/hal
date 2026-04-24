// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::CPUProcessor;
use crate::Result;
use edgefirst_decoder::{DetectBox, Segmentation};
use ndarray::{linalg::general_mat_mul, Array3, Axis};
use rayon::prelude::*;

/// Reusable scratch buffers for the batched `materialize_masks` path.
///
/// Held on [`CPUProcessor`] so repeated calls (e.g. COCO validation) reuse
/// the dequantised-protos and logits allocations. All buffers grow with
/// `resize` (which reuses capacity) rather than being reallocated.
#[derive(Debug, Clone, Default)]
pub(crate) struct MaskScratch {
    /// Dequantised protos flattened to `(H*W, K)` row-major.
    dequant: Vec<f32>,
    /// Detection mask coefficients packed to `(N, K)` row-major for GEMM.
    coeffs: Vec<f32>,
    /// GEMM output `(N, H*W)` row-major — one logit plane per detection.
    logits: Vec<f32>,
}

impl MaskScratch {
    /// Compute per-detection logit planes `(N, H*W)` at proto resolution via
    /// a single GEMM. Returns a view into the internal buffer so per-detection
    /// post-processing can read it without copying.
    ///
    /// Shape contract:
    /// - `coeffs`: slice of `N` coefficient vectors, each length `K`.
    /// - `protos`: `(H, W, K)` float or `(H, W, K)` i8 + quantization.
    /// - Returns `(N, H*W)` f32 logits (pre-sigmoid).
    fn compute_logits(
        &mut self,
        coeffs: &[Vec<f32>],
        protos: &edgefirst_decoder::ProtoTensor,
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
    ) -> Result<()> {
        use edgefirst_decoder::ProtoTensor;

        let n = coeffs.len();
        let hw = proto_h * proto_w;

        // Pack coefficients into contiguous (N, K). Reject early if any
        // detection has the wrong coefficient count — cheaper to check here
        // than per-pixel later.
        self.coeffs.clear();
        self.coeffs.reserve(n * num_protos);
        for coeff in coeffs {
            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }
            self.coeffs.extend_from_slice(coeff);
        }

        // Dequantise protos into (H*W, K) row-major. For Float protos we
        // still copy to guarantee a contiguous (H*W, K) layout regardless of
        // the source tensor's stride (ndarray protos may arrive with a
        // non-standard layout from the decoder's tensor_bridge).
        //
        // Only grow the buffer when needed — the dequant and GEMM loops
        // below overwrite every element in `0..required`, so we can reuse
        // whatever stale data is past `0..required` from previous calls
        // and skip the per-frame memset that otherwise costs ~0.5% on
        // i.MX 95 validation runs.
        let dequant_required = hw * num_protos;
        if self.dequant.len() < dequant_required {
            self.dequant.resize(dequant_required, 0.0);
        }
        let dst = &mut self.dequant[..dequant_required];
        match protos {
            ProtoTensor::Float(arr) => {
                // Fast path: standard-layout (H, W, K) with K innermost is
                // already contiguous row-major and matches our dst layout
                // byte-for-byte — avoid the per-element indexing overhead.
                if let Some(src_slice) = arr.as_slice() {
                    dst.copy_from_slice(src_slice);
                } else {
                    let src = arr.view();
                    for y in 0..proto_h {
                        for x in 0..proto_w {
                            let row = (y * proto_w + x) * num_protos;
                            for k in 0..num_protos {
                                dst[row + k] = src[[y, x, k]];
                            }
                        }
                    }
                }
            }
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                let scale = quantization.scale;
                let zp = quantization.zero_point as f32;
                if let Some(src_slice) = protos.as_slice() {
                    dequant_i8_to_f32(src_slice, dst, zp, scale);
                } else {
                    let src = protos.view();
                    for y in 0..proto_h {
                        for x in 0..proto_w {
                            let row = (y * proto_w + x) * num_protos;
                            for k in 0..num_protos {
                                dst[row + k] = (src[[y, x, k]] as f32 - zp) * scale;
                            }
                        }
                    }
                }
            }
        }

        // GEMM: logits (N, HW) = coeffs (N, K) · dequant.T (K, HW)
        // We build views over the scratch buffers — no extra allocation.
        // Same grow-only discipline: GEMM with beta=0.0 writes every
        // element in `0..logits_required`; anything past that is stale
        // but ignored by `logits()`.
        let logits_required = n * hw;
        if self.logits.len() < logits_required {
            self.logits.resize(logits_required, 0.0);
        }
        let a = ndarray::ArrayView2::from_shape((n, num_protos), &self.coeffs)
            .expect("coeffs buffer matches (N, K)");
        let b =
            ndarray::ArrayView2::from_shape((hw, num_protos), &self.dequant[..dequant_required])
                .expect("dequant buffer matches (HW, K)");
        let mut c =
            ndarray::ArrayViewMut2::from_shape((n, hw), &mut self.logits[..logits_required])
                .expect("logits buffer matches (N, HW)");
        // `b.t()` is a view — no copy — giving us (K, HW).
        general_mat_mul(1.0, &a, &b.t(), 0.0, &mut c);

        Ok(())
    }

    /// Read-only view of the last computed logits as `(N, H*W)`.
    ///
    /// The buffer may be larger than `n * hw` because we grow the Vec
    /// only when required and leave stale trailing data to avoid a
    /// per-frame memset — callers only read the first `n * hw` entries.
    fn logits(&self, n: usize, hw: usize) -> &[f32] {
        debug_assert!(self.logits.len() >= n * hw);
        &self.logits[..n * hw]
    }
}

/// For `MaskResolution::Proto`, the per-detection ROI kernel only touches
/// the bbox's worth of proto pixels. The batched path dequantises and
/// GEMMs over the *entire* proto plane regardless of bbox size, so it
/// only wins once the per-detection work exceeds the up-front full-plane
/// cost. Measured crossover: `N ~= 16` on Cortex-A55, `N ~= 12` on x86.
/// Picked conservatively at 16 to avoid regressions on embedded targets
/// at the cost of a small miss at the crossover on desktop.
const BATCHED_GEMM_MIN_N_PROTO: usize = 16;

/// For `MaskResolution::Scaled`, the per-detection work is an
/// `out_w × out_h × K` bilinear+dot+sigmoid loop (multi-ms per detection
/// at 640×640), so the GEMM amortises cleanly across detections. At
/// `N=1` on Cortex-A55 the batched full-plane dequant+GEMM slightly
/// overshoots the single-detection scalar work, so the threshold is 2.
const BATCHED_GEMM_MIN_N_SCALED: usize = 2;

/// Development/benchmarking toggle: force `materialize_masks` to always use
/// the legacy per-detection fused kernels, bypassing the batched-GEMM path.
/// Set `EDGEFIRST_LEGACY_MATERIALIZE=1` to enable. Intended for A/B
/// profiling only — production callers should never set it.
fn legacy_materialize_forced() -> bool {
    std::env::var_os("EDGEFIRST_LEGACY_MATERIALIZE").is_some()
}

/// Dequantise a contiguous i8 slice into a contiguous f32 slice of the same
/// length: `dst[i] = (src[i] as f32 - zp) * scale`.
///
/// On aarch64 this dispatches to the NEON path below (16 elements per
/// iteration via a cascade of `sxtl`, `scvtf`, `fsub`, `fmul`); on other
/// architectures the compiler typically auto-vectorises the scalar
/// fallback well enough that this function is not the bottleneck.
#[inline]
fn dequant_i8_to_f32(src: &[i8], dst: &mut [f32], zp: f32, scale: f32) {
    debug_assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "aarch64")]
    {
        // Safe: target_arch=aarch64 guarantees NEON is available (unlike
        // armv7 where NEON is optional). The SIMD path itself only uses
        // intrinsics that are in the base aarch64 NEON ISA.
        unsafe { dequant_i8_to_f32_neon(src, dst, zp, scale) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = (*s as f32 - zp) * scale;
        }
    }
}

/// NEON cascade: 16 i8 → 16 f32 per iteration.
///
///   sxtl   i8 → i16   (16→8 halves × 2)
///   sxtl   i16 → i32  (8→4 quarters × 4)
///   scvtf  i32 → f32  (× 4)
///   fsub   -= zp
///   fmul   *= scale
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_i8_to_f32_neon(src: &[i8], dst: &mut [f32], zp: f32, scale: f32) {
    use core::arch::aarch64::*;

    let zp_v = vdupq_n_f32(zp);
    let scale_v = vdupq_n_f32(scale);

    let n = src.len();
    let chunks = n / 16;
    let mut i = 0;
    for _ in 0..chunks {
        let v_i8 = vld1q_s8(src.as_ptr().add(i));
        let v_i16_lo = vmovl_s8(vget_low_s8(v_i8));
        let v_i16_hi = vmovl_s8(vget_high_s8(v_i8));
        let v_i32_0 = vmovl_s16(vget_low_s16(v_i16_lo));
        let v_i32_1 = vmovl_s16(vget_high_s16(v_i16_lo));
        let v_i32_2 = vmovl_s16(vget_low_s16(v_i16_hi));
        let v_i32_3 = vmovl_s16(vget_high_s16(v_i16_hi));
        let v_f0 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(v_i32_0), zp_v), scale_v);
        let v_f1 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(v_i32_1), zp_v), scale_v);
        let v_f2 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(v_i32_2), zp_v), scale_v);
        let v_f3 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(v_i32_3), zp_v), scale_v);
        vst1q_f32(dst.as_mut_ptr().add(i), v_f0);
        vst1q_f32(dst.as_mut_ptr().add(i + 4), v_f1);
        vst1q_f32(dst.as_mut_ptr().add(i + 8), v_f2);
        vst1q_f32(dst.as_mut_ptr().add(i + 12), v_f3);
        i += 16;
    }
    // Scalar tail for n % 16 remainder.
    while i < n {
        *dst.get_unchecked_mut(i) = (*src.get_unchecked(i) as f32 - zp) * scale;
        i += 1;
    }
}

impl CPUProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_modelpack_segmentation(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        segmentation: &Segmentation,
        opacity: f32,
    ) -> Result<()> {
        use ndarray_stats::QuantileExt;

        let seg = &segmentation.segmentation;
        let [seg_height, seg_width, seg_classes] = *seg.shape() else {
            unreachable!("Array3 did not have [usize; 3] as shape");
        };
        let start_y = (dst_h as f32 * segmentation.ymin).round();
        let end_y = (dst_h as f32 * segmentation.ymax).round();
        let start_x = (dst_w as f32 * segmentation.xmin).round();
        let end_x = (dst_w as f32 * segmentation.xmax).round();

        let scale_x = (seg_width as f32 - 1.0) / ((end_x - start_x) - 1.0);
        let scale_y = (seg_height as f32 - 1.0) / ((end_y - start_y) - 1.0);

        let start_x_u = (start_x as usize).min(dst_w);
        let start_y_u = (start_y as usize).min(dst_h);
        let end_x_u = (end_x as usize).min(dst_w);
        let end_y_u = (end_y as usize).min(dst_h);

        let argmax = seg.map_axis(Axis(2), |r| r.argmax().unwrap());
        let get_value_at_nearest = |x: f32, y: f32| -> usize {
            let x = x.round() as usize;
            let y = y.round() as usize;
            argmax
                .get([y.min(seg_height - 1), x.min(seg_width - 1)])
                .copied()
                .unwrap_or(0)
        };

        for y in start_y_u..end_y_u {
            for x in start_x_u..end_x_u {
                let seg_x = (x as f32 - start_x) * scale_x;
                let seg_y = (y as f32 - start_y) * scale_y;
                let label = get_value_at_nearest(seg_x, seg_y);

                if label == seg_classes - 1 {
                    continue;
                }

                let color = self.colors[label % self.colors.len()];

                let alpha = if opacity == 1.0 {
                    color[3] as u16
                } else {
                    (color[3] as f32 * opacity).round() as u16
                };

                let dst_index = (y * dst_rs) + (x * dst_c);
                for c in 0..3 {
                    dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                        + dst_slice[dst_index + c] as u16 * (255 - alpha))
                        / 255) as u8;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_yolo_segmentation(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        segmentation: &Segmentation,
        class: usize,
        opacity: f32,
    ) -> Result<()> {
        let seg = &segmentation.segmentation;
        let [seg_height, seg_width, classes] = *seg.shape() else {
            unreachable!("Array3 did not have [usize;3] as shape");
        };
        debug_assert_eq!(classes, 1);

        let start_y = (dst_h as f32 * segmentation.ymin).round();
        let end_y = (dst_h as f32 * segmentation.ymax).round();
        let start_x = (dst_w as f32 * segmentation.xmin).round();
        let end_x = (dst_w as f32 * segmentation.xmax).round();

        let scale_x = (seg_width as f32 - 1.0) / ((end_x - start_x) - 1.0);
        let scale_y = (seg_height as f32 - 1.0) / ((end_y - start_y) - 1.0);

        let start_x_u = (start_x as usize).min(dst_w);
        let start_y_u = (start_y as usize).min(dst_h);
        let end_x_u = (end_x as usize).min(dst_w);
        let end_y_u = (end_y as usize).min(dst_h);

        for y in start_y_u..end_y_u {
            for x in start_x_u..end_x_u {
                let seg_x = ((x as f32 - start_x) * scale_x) as usize;
                let seg_y = ((y as f32 - start_y) * scale_y) as usize;
                let val = *seg.get([seg_y, seg_x, 0]).unwrap_or(&0);

                if val < 127 {
                    continue;
                }

                let color = self.colors[class % self.colors.len()];

                let alpha = if opacity == 1.0 {
                    color[3] as u16
                } else {
                    (color[3] as f32 * opacity).round() as u16
                };

                let dst_index = (y * dst_rs) + (x * dst_c);
                for c in 0..3 {
                    dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                        + dst_slice[dst_index + c] as u16 * (255 - alpha))
                        / 255) as u8;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_box(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        detect: &[DetectBox],
        color_mode: crate::ColorMode,
    ) -> Result<()> {
        const LINE_THICKNESS: usize = 3;

        for (idx, d) in detect.iter().enumerate() {
            use edgefirst_decoder::BoundingBox;

            let color_index = color_mode.index(idx, d.label);
            let [r, g, b, _] = self.colors[color_index % self.colors.len()];
            let bbox = d.bbox.to_canonical();
            let bbox = BoundingBox {
                xmin: bbox.xmin.clamp(0.0, 1.0),
                ymin: bbox.ymin.clamp(0.0, 1.0),
                xmax: bbox.xmax.clamp(0.0, 1.0),
                ymax: bbox.ymax.clamp(0.0, 1.0),
            };
            let inner = [
                ((dst_w - 1) as f32 * bbox.xmin - 0.5).round() as usize,
                ((dst_h - 1) as f32 * bbox.ymin - 0.5).round() as usize,
                ((dst_w - 1) as f32 * bbox.xmax + 0.5).round() as usize,
                ((dst_h - 1) as f32 * bbox.ymax + 0.5).round() as usize,
            ];

            let outer = [
                inner[0].saturating_sub(LINE_THICKNESS),
                inner[1].saturating_sub(LINE_THICKNESS),
                (inner[2] + LINE_THICKNESS).min(dst_w),
                (inner[3] + LINE_THICKNESS).min(dst_h),
            ];

            // top line
            for y in outer[1] + 1..=inner[1] {
                for x in outer[0] + 1..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }

            // left and right lines
            for y in inner[1]..inner[3] {
                for x in outer[0] + 1..=inner[0] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }

                for x in inner[2]..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }

            // bottom line
            for y in inner[3]..outer[3] {
                for x in outer[0] + 1..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }
        }
        Ok(())
    }

    /// Materialize segmentation masks from proto data into `Vec<Segmentation>`.
    ///
    /// This is the CPU-side decode step of the hybrid mask rendering path:
    /// call this to get pre-decoded masks, then pass them to
    /// [`draw_decoded_masks`](crate::ImageProcessorTrait::draw_decoded_masks) for GPU overlay.
    /// Benchmarks show this hybrid path (CPU decode + GL overlay) is faster
    /// than the fused GPU `draw_proto_masks` on all tested platforms.
    ///
    /// Optimized path:
    /// - `N >= BATCHED_GEMM_MIN_N`: one `(N, K) × (K, H*W)` GEMM via
    ///   `ndarray`'s `matrixmultiply` backend (SIMD-vectorised) shared across
    ///   all detections; per-detection tile assembly is rayon-parallel using
    ///   `MaskScratch` buffers pooled on `CPUProcessor`.
    /// - `N < BATCHED_GEMM_MIN_N`: per-detection fused dequant+dot+sigmoid
    ///   (the original path) — avoids the 3.1MB dequant allocation when the
    ///   GEMM savings can't amortise it.
    pub fn materialize_segmentations(
        &mut self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
        letterbox: Option<[f32; 4]>,
    ) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }
        let min_n = if legacy_materialize_forced() {
            usize::MAX
        } else {
            BATCHED_GEMM_MIN_N_PROTO
        };
        if detect.len() < min_n {
            materialize_segmentations_fused(detect, proto_data, letterbox)
        } else {
            materialize_segmentations_batched(&mut self.mask_scratch, detect, proto_data, letterbox)
        }
    }

    /// Produce per-detection masks at `(width, height)` pixel resolution by
    /// upsampling the full proto plane once then cropping per bbox. Each
    /// `det.bbox` is assumed to be in model-input normalized coordinates
    /// (the convention used by the decoder output); when `letterbox` is
    /// `Some`, `(width, height)` are original-content pixel dims and the
    /// inverse letterbox transform is applied to both the bbox (for the
    /// crop region and returned `Segmentation` metadata) and each output
    /// pixel (for proto-plane sampling). Mask values are binary
    /// `uint8 {0, 255}` after thresholding sigmoid > 0.5.
    ///
    /// Used by [`ImageProcessor::materialize_masks`] when the caller selects
    /// [`MaskResolution::Scaled`](crate::MaskResolution::Scaled).
    pub fn materialize_scaled_segmentations(
        &mut self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
        letterbox: Option<[f32; 4]>,
        width: u32,
        height: u32,
    ) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
        use edgefirst_decoder::ProtoTensor;

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidShape(
                "Scaled mask width/height must be positive".into(),
            ));
        }

        if !legacy_materialize_forced() && detect.len() >= BATCHED_GEMM_MIN_N_SCALED {
            return scaled_segmentations_batched(
                &mut self.mask_scratch,
                detect,
                proto_data,
                letterbox,
                width,
                height,
            );
        }

        match &proto_data.protos {
            ProtoTensor::Float(protos) => scaled_segmentations_float(
                detect,
                &proto_data.mask_coefficients,
                protos,
                letterbox,
                width,
                height,
            ),
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => scaled_segmentations_quant_i8(
                detect,
                &proto_data.mask_coefficients,
                protos,
                *quantization,
                letterbox,
                width,
                height,
            ),
        }
    }
}

// =========================================================================
// Batched-GEMM paths (shared dequant + single matmul + rayon per-detection)
// =========================================================================

fn proto_shape(protos: &edgefirst_decoder::ProtoTensor) -> (usize, usize, usize) {
    use edgefirst_decoder::ProtoTensor;
    match protos {
        ProtoTensor::Quantized { protos, .. } => {
            (protos.shape()[0], protos.shape()[1], protos.shape()[2])
        }
        ProtoTensor::Float(arr) => (arr.shape()[0], arr.shape()[1], arr.shape()[2]),
    }
}

/// Small-N fused path: exactly the pre-batched implementation, retained for
/// the `N < BATCHED_GEMM_MIN_N` case where the up-front dequant cost would
/// dominate the per-detection work.
fn materialize_segmentations_fused(
    detect: &[crate::DetectBox],
    proto_data: &crate::ProtoData,
    letterbox: Option<[f32; 4]>,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    use edgefirst_decoder::ProtoTensor;

    let (proto_h, proto_w, num_protos) = proto_shape(&proto_data.protos);
    let (lx0, inv_lw, ly0, inv_lh) = inv_letterbox(letterbox);

    detect
        .iter()
        .zip(proto_data.mask_coefficients.iter())
        .map(|(det, coeff)| {
            let (x0, y0, x1, y1) = bbox_to_proto_roi(det, proto_w, proto_h);
            let roi_w = x1.saturating_sub(x0).max(1);
            let roi_h = y1.saturating_sub(y0).max(1);

            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }

            let mask = match &proto_data.protos {
                ProtoTensor::Quantized {
                    protos,
                    quantization,
                } => fused_dequant_dot_sigmoid_i8(
                    protos,
                    coeff,
                    quantization.scale,
                    quantization.zero_point as f32,
                    y0,
                    x0,
                    roi_h,
                    roi_w,
                    num_protos,
                ),
                ProtoTensor::Float(protos) => {
                    fused_dot_sigmoid_f32(protos, coeff, y0, x0, roi_h, roi_w, num_protos)
                }
            };

            Ok(proto_roi_to_segmentation(
                mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
            ))
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Batched path for `MaskResolution::Proto`: one GEMM, then rayon-parallel
/// ROI crop + sigmoid+quantize per detection.
fn materialize_segmentations_batched(
    scratch: &mut MaskScratch,
    detect: &[crate::DetectBox],
    proto_data: &crate::ProtoData,
    letterbox: Option<[f32; 4]>,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = proto_shape(&proto_data.protos);
    let hw = proto_h * proto_w;
    let n = detect.len();

    // One GEMM across all detections.
    scratch.compute_logits(
        &proto_data.mask_coefficients,
        &proto_data.protos,
        proto_h,
        proto_w,
        num_protos,
    )?;

    // Precompute inverse letterbox transform (shared across workers).
    let (lx0, inv_lw, ly0, inv_lh) = inv_letterbox(letterbox);

    // Shared read-only view of the logits buffer across rayon workers.
    let logits: &[f32] = scratch.logits(n, hw);

    detect
        .par_iter()
        .enumerate()
        .map(|(i, det)| {
            let (x0, y0, x1, y1) = bbox_to_proto_roi(det, proto_w, proto_h);
            let roi_w = x1.saturating_sub(x0).max(1);
            let roi_h = y1.saturating_sub(y0).max(1);
            let row = &logits[i * hw..(i + 1) * hw];

            let mut mask = Array3::<u8>::zeros((roi_h, roi_w, 1));
            for yi in 0..roi_h {
                let src_row_start = (y0 + yi) * proto_w + x0;
                let src_row = &row[src_row_start..src_row_start + roi_w];
                for (xi, &logit) in src_row.iter().enumerate() {
                    let s = fast_sigmoid(logit);
                    mask[[yi, xi, 0]] = (s * 255.0 + 0.5) as u8;
                }
            }

            Ok(proto_roi_to_segmentation(
                mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
            ))
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Batched path for `MaskResolution::Scaled`: GEMM at proto resolution, then
/// per-detection bilinear-upsample the (pre-sigmoid) logits into the bbox
/// tile at output resolution, threshold, and emit binary {0, 255} mask.
fn scaled_segmentations_batched(
    scratch: &mut MaskScratch,
    detect: &[crate::DetectBox],
    proto_data: &crate::ProtoData,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = proto_shape(&proto_data.protos);
    let hw = proto_h * proto_w;
    let n = detect.len();

    scratch.compute_logits(
        &proto_data.mask_coefficients,
        &proto_data.protos,
        proto_h,
        proto_w,
        num_protos,
    )?;

    // letterbox semantics match `scaled_segmentations_float` / `_quant_i8`.
    let (lx0, lw, ly0, lh) = match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = (lx1 - lx0).max(f32::EPSILON);
            let lh = (ly1 - ly0).max(f32::EPSILON);
            (lx0, lw, ly0, lh)
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    };
    let out_w = width as usize;
    let out_h = height as usize;

    let logits: &[f32] = scratch.logits(n, hw);

    detect
        .par_iter()
        .enumerate()
        .map(|(i, det)| {
            let bbox = det.bbox.to_canonical();
            let xmin = ((bbox.xmin - lx0) / lw).clamp(0.0, 1.0);
            let ymin = ((bbox.ymin - ly0) / lh).clamp(0.0, 1.0);
            let xmax = ((bbox.xmax - lx0) / lw).clamp(0.0, 1.0);
            let ymax = ((bbox.ymax - ly0) / lh).clamp(0.0, 1.0);

            let px0 = (xmin * out_w as f32).round() as usize;
            let py0 = (ymin * out_h as f32).round() as usize;
            let px1 = ((xmax * out_w as f32).round() as usize).min(out_w);
            let py1 = ((ymax * out_h as f32).round() as usize).min(out_h);
            let bbox_w = px1.saturating_sub(px0).max(1);
            let bbox_h = py1.saturating_sub(py0).max(1);

            let mut tile = Array3::<u8>::zeros((bbox_h, bbox_w, 1));
            let row = &logits[i * hw..(i + 1) * hw];

            // Bilinear sample the logit plane at output resolution, then
            // threshold at `acc > 0`. Dispatches to a NEON-vectorised
            // inner kernel on aarch64; scalar fallback elsewhere.
            fill_scaled_tile(
                row,
                proto_w,
                proto_h,
                px0,
                py0,
                bbox_w,
                bbox_h,
                out_w,
                out_h,
                lx0,
                lw,
                ly0,
                lh,
                tile.as_slice_mut()
                    .expect("Array3<u8>::zeros is contiguous"),
            );

            Ok(edgefirst_decoder::Segmentation {
                xmin,
                ymin,
                xmax,
                ymax,
                segmentation: tile,
            })
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Fill a `bbox_h × bbox_w` u8 tile by bilinear-sampling a flat logit plane
/// and thresholding at `acc > 0` (scaled-mask convention).
///
/// `tile` is a contiguous `bbox_h * bbox_w` slice in row-major order.
#[allow(clippy::too_many_arguments)]
fn fill_scaled_tile(
    plane: &[f32],
    proto_w: usize,
    proto_h: usize,
    px0: usize,
    py0: usize,
    bbox_w: usize,
    bbox_h: usize,
    out_w: usize,
    out_h: usize,
    lx0: f32,
    lw: f32,
    ly0: f32,
    lh: f32,
    tile: &mut [u8],
) {
    debug_assert_eq!(tile.len(), bbox_h * bbox_w);
    debug_assert_eq!(plane.len(), proto_h * proto_w);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            fill_scaled_tile_neon(
                plane, proto_w, proto_h, px0, py0, bbox_w, bbox_h, out_w, out_h, lx0, lw, ly0, lh,
                tile,
            )
        };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let proto_wf = proto_w as f32;
        let proto_hf = proto_h as f32;
        let out_wf = out_w as f32;
        let out_hf = out_h as f32;
        for yi in 0..bbox_h {
            let py = (py0 + yi) as f32;
            let model_y_norm = ly0 + (py + 0.5) / out_hf * lh;
            let sample_y = model_y_norm * proto_hf - 0.5;
            for xi in 0..bbox_w {
                let px = (px0 + xi) as f32;
                let model_x_norm = lx0 + (px + 0.5) / out_wf * lw;
                let sample_x = model_x_norm * proto_wf - 0.5;
                let acc = bilinear_sample_plane(plane, sample_x, sample_y, proto_w, proto_h);
                tile[yi * bbox_w + xi] = if acc > 0.0 { 255 } else { 0 };
            }
        }
    }
}

/// NEON-vectorised `fill_scaled_tile`.
///
/// Strategy: process 4 output pixels per iteration along the `xi` axis.
/// The `y`-related state (sample_y, y0, y1, fy) is scalar — shared across
/// all 4 xi in a row. The 4-wide inner loop vectorises the px→sample_x
/// chain, bilinear-weight computation, corner fetch (via 4× scalar loads
/// assembled into a NEON register), bilinear accumulation, and the
/// `acc > 0 → {0, 255}` compare-and-pack.
///
/// Gather is the single non-vectorisable op: pre-SVE NEON has no native
/// gather, so each of the 4 corners (v00, v10, v01, v11) costs 4 scalar
/// loads that the CPU issues into the load pipe. Still a net win because
/// the arithmetic (roughly 2/3 of the original scalar work) goes 4-wide.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_scaled_tile_neon(
    plane: &[f32],
    proto_w: usize,
    proto_h: usize,
    px0: usize,
    py0: usize,
    bbox_w: usize,
    bbox_h: usize,
    out_w: usize,
    out_h: usize,
    lx0: f32,
    lw: f32,
    ly0: f32,
    lh: f32,
    tile: &mut [u8],
) {
    use core::arch::aarch64::*;

    // Precompute scalar constants reused across the whole tile.
    let proto_wf = proto_w as f32;
    let proto_hf = proto_h as f32;
    let inv_out_w = 1.0_f32 / out_w as f32;
    let inv_out_h = 1.0_f32 / out_h as f32;
    let proto_w_max_i32 = (proto_w as i32) - 1;
    let proto_h_max_i32 = (proto_h as i32) - 1;

    // x-offset lane pattern for the 4-wide SIMD sweep: xi + [0, 1, 2, 3]
    let lane_offsets = [0.0_f32, 1.0, 2.0, 3.0];
    let v_lane_offsets = vld1q_f32(lane_offsets.as_ptr());

    // Broadcast constants.
    let v_lx0 = vdupq_n_f32(lx0);
    let v_lw_over_outw = vdupq_n_f32(lw * inv_out_w);
    let v_proto_wf = vdupq_n_f32(proto_wf);
    let v_half = vdupq_n_f32(0.5);
    let v_neg_half = vdupq_n_f32(-0.5);
    let v_one = vdupq_n_f32(1.0);
    let v_zero = vdupq_n_f32(0.0);
    let v_proto_w_max = vdupq_n_s32(proto_w_max_i32);
    let v_zero_i32 = vdupq_n_s32(0);
    let v_mask_byte = vdupq_n_u32(0xFF);

    let plane_ptr = plane.as_ptr();

    for yi in 0..bbox_h {
        // Scalar y state shared across all xi in this row.
        let py = (py0 + yi) as f32;
        let model_y_norm = ly0 + (py + 0.5) * inv_out_h * lh;
        let sample_y = model_y_norm * proto_hf - 0.5;
        let y0_i32 = sample_y.floor() as i32;
        let y0 = y0_i32.clamp(0, proto_h_max_i32) as usize;
        let y1 = (y0 + 1).min(proto_h.saturating_sub(1));
        let fy = sample_y - sample_y.floor();
        let one_minus_fy = 1.0 - fy;
        let v_fy = vdupq_n_f32(fy);
        let v_one_minus_fy = vdupq_n_f32(one_minus_fy);
        let y0_row = y0 * proto_w;
        let y1_row = y1 * proto_w;
        let tile_row_off = yi * bbox_w;

        let mut xi = 0;
        // Main SIMD loop: 4 output pixels per iteration.
        while xi + 4 <= bbox_w {
            // px = px0 + xi + [0, 1, 2, 3]
            let v_xi_base = vdupq_n_f32((px0 + xi) as f32);
            let v_px = vaddq_f32(v_xi_base, v_lane_offsets);
            // model_x = lx0 + (px + 0.5) * (lw / out_w)
            let v_model_x = vfmaq_f32(v_lx0, vaddq_f32(v_px, v_half), v_lw_over_outw);
            // sample_x = model_x * proto_w - 0.5
            let v_sample_x = vfmaq_f32(v_neg_half, v_model_x, v_proto_wf);
            // floor(sample_x) → int32, clamp to [0, proto_w-1]; also keep float floor for fx
            let v_floor = vrndmq_f32(v_sample_x); // round toward -inf
            let v_fx = vsubq_f32(v_sample_x, v_floor);
            let v_x0_raw = vcvtq_s32_f32(v_floor);
            let v_x0 = vminq_s32(vmaxq_s32(v_x0_raw, v_zero_i32), v_proto_w_max);
            let v_x1 = vminq_s32(vaddq_s32(v_x0, vdupq_n_s32(1)), v_proto_w_max);
            // Weights: w00 = (1-fx)(1-fy), w10 = fx(1-fy), w01 = (1-fx)fy, w11 = fx*fy
            let v_one_minus_fx = vsubq_f32(v_one, v_fx);
            let v_w00 = vmulq_f32(v_one_minus_fx, v_one_minus_fy);
            let v_w10 = vmulq_f32(v_fx, v_one_minus_fy);
            let v_w01 = vmulq_f32(v_one_minus_fx, v_fy);
            let v_w11 = vmulq_f32(v_fx, v_fy);

            // Extract the 4 x0 / x1 indices as usize for the gather.
            // Using vgetq_lane_s32 keeps the values in GP regs for the
            // address math below.
            let x0_0 = vgetq_lane_s32(v_x0, 0) as usize;
            let x0_1 = vgetq_lane_s32(v_x0, 1) as usize;
            let x0_2 = vgetq_lane_s32(v_x0, 2) as usize;
            let x0_3 = vgetq_lane_s32(v_x0, 3) as usize;
            let x1_0 = vgetq_lane_s32(v_x1, 0) as usize;
            let x1_1 = vgetq_lane_s32(v_x1, 1) as usize;
            let x1_2 = vgetq_lane_s32(v_x1, 2) as usize;
            let x1_3 = vgetq_lane_s32(v_x1, 3) as usize;

            // Gather the 4 corners — no native NEON gather; scalar loads
            // land in regular f32 regs then get packed into a NEON vector.
            let v00_arr = [
                *plane_ptr.add(y0_row + x0_0),
                *plane_ptr.add(y0_row + x0_1),
                *plane_ptr.add(y0_row + x0_2),
                *plane_ptr.add(y0_row + x0_3),
            ];
            let v10_arr = [
                *plane_ptr.add(y0_row + x1_0),
                *plane_ptr.add(y0_row + x1_1),
                *plane_ptr.add(y0_row + x1_2),
                *plane_ptr.add(y0_row + x1_3),
            ];
            let v01_arr = [
                *plane_ptr.add(y1_row + x0_0),
                *plane_ptr.add(y1_row + x0_1),
                *plane_ptr.add(y1_row + x0_2),
                *plane_ptr.add(y1_row + x0_3),
            ];
            let v11_arr = [
                *plane_ptr.add(y1_row + x1_0),
                *plane_ptr.add(y1_row + x1_1),
                *plane_ptr.add(y1_row + x1_2),
                *plane_ptr.add(y1_row + x1_3),
            ];
            let v_v00 = vld1q_f32(v00_arr.as_ptr());
            let v_v10 = vld1q_f32(v10_arr.as_ptr());
            let v_v01 = vld1q_f32(v01_arr.as_ptr());
            let v_v11 = vld1q_f32(v11_arr.as_ptr());

            // acc = w00*v00 + w10*v10 + w01*v01 + w11*v11 (FMA chain)
            let mut v_acc = vmulq_f32(v_w00, v_v00);
            v_acc = vfmaq_f32(v_acc, v_w10, v_v10);
            v_acc = vfmaq_f32(v_acc, v_w01, v_v01);
            v_acc = vfmaq_f32(v_acc, v_w11, v_v11);

            // Threshold: (acc > 0) → 0xFFFFFFFF, else 0; mask with 0xFF, pack.
            let v_gt = vcgtq_f32(v_acc, v_zero); // u32x4 of 0 or -1
            let v_bytes_u32 = vandq_u32(v_gt, v_mask_byte);
            // Narrow u32x4 → u16x4, zero-combine → u16x8, narrow → u8x8, keep low 4 bytes.
            let v_u16 = vqmovn_u32(v_bytes_u32);
            let v_u8 = vqmovn_u16(vcombine_u16(v_u16, vdup_n_u16(0)));
            // Store the low 4 lanes as one 32-bit write.
            let out = vget_lane_u32::<0>(vreinterpret_u32_u8(v_u8));
            core::ptr::write_unaligned(tile.as_mut_ptr().add(tile_row_off + xi) as *mut u32, out);
            xi += 4;
        }

        // Scalar tail for bbox_w % 4.
        while xi < bbox_w {
            let px = (px0 + xi) as f32;
            let model_x_norm = lx0 + (px + 0.5) * inv_out_w * lw;
            let sample_x = model_x_norm * proto_wf - 0.5;
            let acc = bilinear_sample_plane(plane, sample_x, sample_y, proto_w, proto_h);
            *tile.get_unchecked_mut(tile_row_off + xi) = if acc > 0.0 { 255 } else { 0 };
            xi += 1;
        }
    }
}

/// Inverse letterbox factors for mapping from model-input normalized
/// coordinates to output-content normalized coordinates. `None` → identity.
fn inv_letterbox(letterbox: Option<[f32; 4]>) -> (f32, f32, f32, f32) {
    match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = lx1 - lx0;
            let lh = ly1 - ly0;
            (
                lx0,
                if lw > 0.0 { 1.0 / lw } else { 1.0 },
                ly0,
                if lh > 0.0 { 1.0 / lh } else { 1.0 },
            )
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    }
}

/// Canonicalise `det.bbox`, clamp to `[0, 1]`, and map into proto-pixel
/// coordinates. Returns `(x0, y0, x1, y1)` with `x1/y1` inclusive of the
/// ceil'd max so `x1 - x0` gives the ROI width.
fn bbox_to_proto_roi(
    det: &crate::DetectBox,
    proto_w: usize,
    proto_h: usize,
) -> (usize, usize, usize, usize) {
    let bbox = det.bbox.to_canonical();
    let xmin = bbox.xmin.clamp(0.0, 1.0);
    let ymin = bbox.ymin.clamp(0.0, 1.0);
    let xmax = bbox.xmax.clamp(0.0, 1.0);
    let ymax = bbox.ymax.clamp(0.0, 1.0);
    let x0 = ((xmin * proto_w as f32) as usize).min(proto_w.saturating_sub(1));
    let y0 = ((ymin * proto_h as f32) as usize).min(proto_h.saturating_sub(1));
    let x1 = ((xmax * proto_w as f32).ceil() as usize).min(proto_w);
    let y1 = ((ymax * proto_h as f32).ceil() as usize).min(proto_h);
    (x0, y0, x1, y1)
}

#[allow(clippy::too_many_arguments)]
fn proto_roi_to_segmentation(
    mask: ndarray::Array3<u8>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    proto_w: usize,
    proto_h: usize,
    lx0: f32,
    inv_lw: f32,
    ly0: f32,
    inv_lh: f32,
) -> edgefirst_decoder::Segmentation {
    let seg_xmin = ((x0 as f32 / proto_w as f32) - lx0) * inv_lw;
    let seg_ymin = ((y0 as f32 / proto_h as f32) - ly0) * inv_lh;
    let seg_xmax = ((x1 as f32 / proto_w as f32) - lx0) * inv_lw;
    let seg_ymax = ((y1 as f32 / proto_h as f32) - ly0) * inv_lh;
    edgefirst_decoder::Segmentation {
        xmin: seg_xmin.clamp(0.0, 1.0),
        ymin: seg_ymin.clamp(0.0, 1.0),
        xmax: seg_xmax.clamp(0.0, 1.0),
        ymax: seg_ymax.clamp(0.0, 1.0),
        segmentation: mask,
    }
}

/// Bilinear sample a flat `(H * W)` plane at subpixel `(px, py)`. Out-of-bounds
/// samples clamp to the plane's edge — matches `bilinear_dot_quant_i8`'s
/// implicit boundary handling.
#[inline]
fn bilinear_sample_plane(plane: &[f32], px: f32, py: f32, w: usize, h: usize) -> f32 {
    let x0 = (px.floor() as isize).clamp(0, w as isize - 1) as usize;
    let y0 = (py.floor() as isize).clamp(0, h as isize - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = px - px.floor();
    let fy = py - py.floor();
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    let v00 = plane[y0 * w + x0];
    let v10 = plane[y0 * w + x1];
    let v01 = plane[y1 * w + x0];
    let v11 = plane[y1 * w + x1];
    w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
}

fn scaled_segmentations_float(
    detect: &[crate::DetectBox],
    mask_coefficients: &[Vec<f32>],
    protos: &ndarray::Array3<f32>,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = (protos.shape()[0], protos.shape()[1], protos.shape()[2]);

    // letterbox = [lx0, ly0, lx1, ly1] in model-input normalized coords.
    // Two uses for the same (lx0, lw) pair:
    //   * Forward (sampling): each output pixel maps to model-input normalized
    //     via model_norm = lx0 + out_norm * lw, then to proto pixels.
    //   * Inverse (bbox frame): det.bbox arrives in model-input normalized
    //     (decoder output); the crop region and returned Segmentation are in
    //     output-content normalized, obtained via (bbox - lx0) / lw.
    // When letterbox is None, (lx0=0, lw=1) makes both directions the identity.
    let (lx0, lw, ly0, lh) = match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = (lx1 - lx0).max(f32::EPSILON);
            let lh = (ly1 - ly0).max(f32::EPSILON);
            (lx0, lw, ly0, lh)
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    };

    let out_w = width as usize;
    let out_h = height as usize;

    detect
        .iter()
        .zip(mask_coefficients.iter())
        .map(|(det, coeff)| {
            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }

            // Canonicalise, then inverse-letterbox into output-content
            // normalized space for bbox crop + Segmentation metadata.
            // Matches the end-of-pipeline transform in the Proto path
            // (see `materialize_segmentations`).
            let bbox = det.bbox.to_canonical();
            let xmin = ((bbox.xmin - lx0) / lw).clamp(0.0, 1.0);
            let ymin = ((bbox.ymin - ly0) / lh).clamp(0.0, 1.0);
            let xmax = ((bbox.xmax - lx0) / lw).clamp(0.0, 1.0);
            let ymax = ((bbox.ymax - ly0) / lh).clamp(0.0, 1.0);

            let px0 = (xmin * out_w as f32).round() as usize;
            let py0 = (ymin * out_h as f32).round() as usize;
            let px1 = ((xmax * out_w as f32).round() as usize).min(out_w);
            let py1 = ((ymax * out_h as f32).round() as usize).min(out_h);
            let bbox_w = px1.saturating_sub(px0).max(1);
            let bbox_h = py1.saturating_sub(py0).max(1);

            let mut tile = ndarray::Array3::<u8>::zeros((bbox_h, bbox_w, 1));

            // Map each output pixel within bbox back to proto-plane
            // coordinates. Center-of-pixel offset (-0.5) matches torch
            // align_corners=False, the Ultralytics retina convention.
            for yi in 0..bbox_h {
                let py = (py0 + yi) as f32;
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                let sample_y = model_y_norm * proto_h as f32 - 0.5;
                for xi in 0..bbox_w {
                    let px = (px0 + xi) as f32;
                    let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                    let sample_x = model_x_norm * proto_w as f32 - 0.5;
                    let acc = bilinear_dot(
                        protos, coeff, num_protos, sample_x, sample_y, proto_w, proto_h,
                    );
                    // sigmoid(x) > 0.5  ⟺  x > 0 (sigmoid is monotonic).
                    tile[[yi, xi, 0]] = if acc > 0.0 { 255 } else { 0 };
                }
            }

            Ok(edgefirst_decoder::Segmentation {
                xmin,
                ymin,
                xmax,
                ymax,
                segmentation: tile,
            })
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Quantized i8 proto variant of [`scaled_segmentations_float`]. Dequantizes
/// inline during bilinear sample + dot product:
///   dequant = (i8 - zero_point) * scale
/// Factoring scale out of the dot product:
///   sigmoid(scale * Σ coef_k * (i8 - zp))
/// so we can run a single scale multiply on the accumulator.
#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_quant_i8(
    detect: &[crate::DetectBox],
    mask_coefficients: &[Vec<f32>],
    protos: &ndarray::Array3<i8>,
    quant: edgefirst_decoder::Quantization,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = (protos.shape()[0], protos.shape()[1], protos.shape()[2]);

    // See scaled_segmentations_float for the forward / inverse letterbox
    // convention — det.bbox arrives in model-input normalized; crop region
    // and Segmentation metadata are output-content normalized via
    // (bbox - lx0) / lw; the sampling transform goes the other direction.
    let (lx0, lw, ly0, lh) = match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = (lx1 - lx0).max(f32::EPSILON);
            let lh = (ly1 - ly0).max(f32::EPSILON);
            (lx0, lw, ly0, lh)
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    };

    let out_w = width as usize;
    let out_h = height as usize;
    // `scale` is no longer needed: the scaled-mask threshold collapses to
    // `acc > 0` regardless of the positive scale factor (see below).
    let _ = quant.scale;
    let zp = quant.zero_point as f32;

    detect
        .iter()
        .zip(mask_coefficients.iter())
        .map(|(det, coeff)| {
            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }

            let bbox = det.bbox.to_canonical();
            let xmin = ((bbox.xmin - lx0) / lw).clamp(0.0, 1.0);
            let ymin = ((bbox.ymin - ly0) / lh).clamp(0.0, 1.0);
            let xmax = ((bbox.xmax - lx0) / lw).clamp(0.0, 1.0);
            let ymax = ((bbox.ymax - ly0) / lh).clamp(0.0, 1.0);

            let px0 = (xmin * out_w as f32).round() as usize;
            let py0 = (ymin * out_h as f32).round() as usize;
            let px1 = ((xmax * out_w as f32).round() as usize).min(out_w);
            let py1 = ((ymax * out_h as f32).round() as usize).min(out_h);
            let bbox_w = px1.saturating_sub(px0).max(1);
            let bbox_h = py1.saturating_sub(py0).max(1);

            let mut tile = ndarray::Array3::<u8>::zeros((bbox_h, bbox_w, 1));

            for yi in 0..bbox_h {
                let py = (py0 + yi) as f32;
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                let sample_y = model_y_norm * proto_h as f32 - 0.5;
                for xi in 0..bbox_w {
                    let px = (px0 + xi) as f32;
                    let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                    let sample_x = model_x_norm * proto_w as f32 - 0.5;
                    let acc = bilinear_dot_quant_i8(
                        protos, coeff, num_protos, sample_x, sample_y, proto_w, proto_h, zp,
                    );
                    // sigmoid(scale * acc) > 0.5  ⟺  scale * acc > 0  ⟺  acc > 0
                    // (quantization scale is always positive).
                    tile[[yi, xi, 0]] = if acc > 0.0 { 255 } else { 0 };
                }
            }

            Ok(edgefirst_decoder::Segmentation {
                xmin,
                ymin,
                xmax,
                ymax,
                segmentation: tile,
            })
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Bilinear sample + zero-point-subtracting dot product over an i8 proto
/// tensor. Returns the scaled-but-not-yet-sigmoid accumulator; caller applies
/// the quantization scale and sigmoid.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bilinear_dot_quant_i8(
    protos: &ndarray::Array3<i8>,
    coeff: &[f32],
    num_protos: usize,
    px: f32,
    py: f32,
    proto_w: usize,
    proto_h: usize,
    zp: f32,
) -> f32 {
    let x0 = (px.floor() as isize).clamp(0, proto_w as isize - 1) as usize;
    let y0 = (py.floor() as isize).clamp(0, proto_h as isize - 1) as usize;
    let x1 = (x0 + 1).min(proto_w - 1);
    let y1 = (y0 + 1).min(proto_h - 1);

    let fx = px - px.floor();
    let fy = py - py.floor();
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let mut acc = 0.0f32;
    for p in 0..num_protos {
        let v00 = protos[[y0, x0, p]] as f32 - zp;
        let v10 = protos[[y0, x1, p]] as f32 - zp;
        let v01 = protos[[y1, x0, p]] as f32 - zp;
        let v11 = protos[[y1, x1, p]] as f32 - zp;
        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
        acc += coeff[p] * val;
    }
    acc
}

/// Bilinear interpolation of proto values at `(px, py)` combined with dot
/// product against `coeff`. Returns the scalar accumulator before sigmoid.
///
/// Samples the four nearest proto texels, weights by bilinear coefficients,
/// and simultaneously computes the dot product with the mask coefficients.
#[inline]
pub(super) fn bilinear_dot(
    protos: &ndarray::Array3<f32>,
    coeff: &[f32],
    num_protos: usize,
    px: f32,
    py: f32,
    proto_w: usize,
    proto_h: usize,
) -> f32 {
    let x0 = (px.floor() as isize).clamp(0, proto_w as isize - 1) as usize;
    let y0 = (py.floor() as isize).clamp(0, proto_h as isize - 1) as usize;
    let x1 = (x0 + 1).min(proto_w - 1);
    let y1 = (y0 + 1).min(proto_h - 1);

    let fx = px - px.floor();
    let fy = py - py.floor();

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let mut acc = 0.0f32;
    for p in 0..num_protos {
        let val = w00 * protos[[y0, x0, p]]
            + w10 * protos[[y0, x1, p]]
            + w01 * protos[[y1, x0, p]]
            + w11 * protos[[y1, x1, p]];
        acc += coeff[p] * val;
    }
    acc
}

/// Fast sigmoid approximation: `1 / (1 + exp(-x))`.
///
/// Uses bit-manipulation for exp() (same approach as `fast_math::exp_raw`).
/// Max relative error < 1.1% for normal results, which is well within the
/// precision needed for mask thresholding and u8 quantization.
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    if x >= 16.0 {
        return 1.0;
    }
    if x <= -16.0 {
        return 0.0;
    }
    // Fast exp(-x) via bit manipulation (Schraudolph's algorithm).
    // f32 bits: 2^23 * log2(e) * x + (127 << 23) approximates exp(x).
    const A: f32 = (1u32 << 23) as f32; // 8388608.0
    const B: f32 = A * std::f32::consts::LOG2_E; // A / ln(2)
    const C: u32 = 127 << 23; // exponent bias
    let neg_x = -x;
    let bits = (B * neg_x) as i32 + C as i32;
    let exp_neg_x = f32::from_bits(bits as u32);
    1.0 / (1.0 + exp_neg_x)
}

/// Fused dequantization + dot product + sigmoid for quantized (i8) protos.
///
/// For each pixel in the ROI, computes:
///   acc = sum_k(coeff[k] * (proto[y, x, k] as f32 - zp) * scale)
///   mask[y, x] = fast_sigmoid(acc) * 255
///
/// This avoids allocating a full f32 proto tensor (3.1MB for 160x160x32)
/// and the hidden `to_shape` copy on non-contiguous ROI slices.
#[allow(clippy::too_many_arguments)]
fn fused_dequant_dot_sigmoid_i8(
    protos: &ndarray::Array3<i8>,
    coeff: &[f32],
    scale: f32,
    zp: f32,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    debug_assert!(
        protos.strides().iter().all(|&s| s >= 0),
        "negative strides unsupported"
    );
    // Pre-scale coefficients: coeff[k] * scale, so the inner loop is
    // just fma: acc += scaled_coeff[k] * (proto_i8 - zp)
    let scaled_coeff: Vec<f32> = coeff.iter().map(|&c| c * scale).collect();
    // Pre-compute coeff_sum * (-zp * scale) offset:
    // sum_k(coeff[k] * (proto - zp) * scale) = sum_k(scaled_coeff[k] * proto) - zp * sum_k(scaled_coeff[k])
    // But since zp is a constant per-pixel term, factor it out:
    // acc = sum_k(scaled_coeff[k] * proto_i8_as_f32) - zp * sum_k(scaled_coeff[k])
    let zp_offset: f32 = zp * scaled_coeff.iter().sum::<f32>();

    let proto_stride_y = protos.strides()[0] as usize;
    let proto_stride_x = protos.strides()[1] as usize;
    let proto_stride_k = protos.strides()[2] as usize;
    let proto_ptr = protos.as_ptr();

    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));

    for y in 0..roi_h {
        for x in 0..roi_w {
            // Base pointer for protos[y0+y, x0+x, 0]
            let base = (y0 + y) * proto_stride_y + (x0 + x) * proto_stride_x;

            let mut acc = 0.0f32;
            let mut k = 0;

            // Process 4 protos at a time for better ILP
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k+3 < num_protos <= protos.shape()[2].
                unsafe {
                    let p0 = *proto_ptr.add(base + k * proto_stride_k) as f32;
                    let p1 = *proto_ptr.add(base + (k + 1) * proto_stride_k) as f32;
                    let p2 = *proto_ptr.add(base + (k + 2) * proto_stride_k) as f32;
                    let p3 = *proto_ptr.add(base + (k + 3) * proto_stride_k) as f32;
                    acc += scaled_coeff[k] * p0
                        + scaled_coeff[k + 1] * p1
                        + scaled_coeff[k + 2] * p2
                        + scaled_coeff[k + 3] * p3;
                }
                k += 4;
            }
            // Remainder
            while k < num_protos {
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k < num_protos <= protos.shape()[2].
                unsafe {
                    let p = *proto_ptr.add(base + k * proto_stride_k) as f32;
                    acc += scaled_coeff[k] * p;
                }
                k += 1;
            }

            acc -= zp_offset;
            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

/// Fused dot product + sigmoid for f32 protos (no dequantization needed).
fn fused_dot_sigmoid_f32(
    protos: &ndarray::Array3<f32>,
    coeff: &[f32],
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    debug_assert!(
        protos.strides().iter().all(|&s| s >= 0),
        "negative strides unsupported"
    );
    let proto_stride_y = protos.strides()[0] as usize;
    let proto_stride_x = protos.strides()[1] as usize;
    let proto_stride_k = protos.strides()[2] as usize;
    let proto_ptr = protos.as_ptr();

    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));

    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * proto_stride_y + (x0 + x) * proto_stride_x;

            let mut acc = 0.0f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k+3 < num_protos <= protos.shape()[2].
                unsafe {
                    let p0 = *proto_ptr.add(base + k * proto_stride_k);
                    let p1 = *proto_ptr.add(base + (k + 1) * proto_stride_k);
                    let p2 = *proto_ptr.add(base + (k + 2) * proto_stride_k);
                    let p3 = *proto_ptr.add(base + (k + 3) * proto_stride_k);
                    acc +=
                        coeff[k] * p0 + coeff[k + 1] * p1 + coeff[k + 2] * p2 + coeff[k + 3] * p3;
                }
                k += 4;
            }
            while k < num_protos {
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k < num_protos <= protos.shape()[2].
                unsafe {
                    let p = *proto_ptr.add(base + k * proto_stride_k);
                    acc += coeff[k] * p;
                }
                k += 1;
            }

            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

#[cfg(test)]
mod scaled_tests {
    use super::*;
    use edgefirst_decoder::{BoundingBox, DetectBox, ProtoData, ProtoTensor};
    use ndarray::Array3;

    fn make_cpu() -> CPUProcessor {
        CPUProcessor::new()
    }

    /// A synthetic proto plane where channel 0 is a centred gaussian peaking
    /// at 10.0 and tailing off to ~0.0 at the edges; other channels zero.
    /// Mask coefficients (1.0, 0, 0, ...) select only channel 0.
    fn synthetic_proto_data(proto_h: usize, proto_w: usize, num_protos: usize) -> ProtoData {
        let mut protos = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        let cy = (proto_h as f32 - 1.0) / 2.0;
        let cx = (proto_w as f32 - 1.0) / 2.0;
        for y in 0..proto_h {
            for x in 0..proto_w {
                let dy = (y as f32 - cy) / cy;
                let dx = (x as f32 - cx) / cx;
                protos[[y, x, 0]] = 10.0 * (-(dx * dx + dy * dy)).exp();
            }
        }
        let mut coeffs = vec![0.0_f32; num_protos];
        coeffs[0] = 1.0;
        ProtoData {
            mask_coefficients: vec![coeffs],
            protos: ProtoTensor::Float(protos),
        }
    }

    #[test]
    fn scaled_central_bbox_produces_foreground_blob() {
        let mut cpu = make_cpu();
        let proto_data = synthetic_proto_data(16, 16, 4);

        // bbox covers the centre 50% of the plane.
        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.25, 0.25, 0.75, 0.75),
            score: 0.9,
            label: 0,
        }];

        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, None, 64, 64)
            .expect("scaled mask rendering must succeed");

        assert_eq!(out.len(), 1);
        let seg = &out[0];
        assert_eq!(seg.segmentation.shape(), &[32, 32, 1]);
        let values: Vec<u8> = seg.segmentation.iter().copied().collect();
        assert!(
            values.iter().all(|&v| v == 0 || v == 255),
            "scaled mask must be binary {{0, 255}}, found other values"
        );
        let fg_count = values.iter().filter(|&&v| v == 255).count();
        let fg_frac = fg_count as f32 / values.len() as f32;
        assert!(
            fg_frac > 0.80,
            "expected >80% foreground for centred gaussian, got {fg_frac:.2}"
        );
    }

    /// Realistic 160×160 → 640×640 parity vs hand-rolled bilinear + sigmoid
    /// reference. Tolerate <0.5% pixel mismatch due to fast_sigmoid's ~1.1%
    /// approximation error near the decision boundary.
    #[test]
    fn scaled_realistic_160x160_to_640x640_matches_reference() {
        let mut rng_seed: u32 = 0xD00D_F00D;
        let next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*s as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        let proto_h = 160;
        let proto_w = 160;
        let num_protos = 32;
        let mut protos = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    protos[[y, x, k]] = next(&mut rng_seed) * 3.0;
                }
            }
        }
        let mut coeffs = vec![0.0_f32; num_protos];
        for c in &mut coeffs {
            *c = next(&mut rng_seed);
        }
        let proto_data = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Float(protos.clone()),
        };

        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.1, 0.2, 0.6, 0.9),
            score: 0.9,
            label: 0,
        }];

        let mut cpu = make_cpu();
        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, None, 640, 640)
            .unwrap();
        assert_eq!(out.len(), 1);
        let hal_tile = &out[0].segmentation;

        let px0 = (0.1_f32 * 640.0_f32).round() as usize;
        let py0 = (0.2_f32 * 640.0_f32).round() as usize;
        let px1 = (0.6_f32 * 640.0_f32).round() as usize;
        let py1 = (0.9_f32 * 640.0_f32).round() as usize;
        let bbox_h = py1 - py0;
        let bbox_w = px1 - px0;
        assert_eq!(hal_tile.shape(), &[bbox_h, bbox_w, 1]);

        let sx = proto_w as f32 / 640.0_f32;
        let sy = proto_h as f32 / 640.0_f32;
        let mut mismatches = 0_usize;
        for yi in 0..bbox_h {
            let py = (py0 + yi) as f32;
            let sy_coord = (py + 0.5) * sy - 0.5;
            for xi in 0..bbox_w {
                let px = (px0 + xi) as f32;
                let sx_coord = (px + 0.5) * sx - 0.5;
                let x0 = sx_coord.floor().clamp(0.0, proto_w as f32 - 1.0) as usize;
                let y0 = sy_coord.floor().clamp(0.0, proto_h as f32 - 1.0) as usize;
                let x1 = (x0 + 1).min(proto_w - 1);
                let y1 = (y0 + 1).min(proto_h - 1);
                let fx = sx_coord - sx_coord.floor();
                let fy = sy_coord - sy_coord.floor();
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                let mut acc = 0.0_f32;
                for p in 0..num_protos {
                    let val = w00 * protos[[y0, x0, p]]
                        + w10 * protos[[y0, x1, p]]
                        + w01 * protos[[y1, x0, p]]
                        + w11 * protos[[y1, x1, p]];
                    acc += coeffs[p] * val;
                }
                let sigmoid = 1.0_f32 / (1.0 + (-acc).exp());
                let expected: u8 = if sigmoid > 0.5 { 255 } else { 0 };
                if hal_tile[[yi, xi, 0]] != expected {
                    mismatches += 1;
                }
            }
        }
        let total = bbox_h * bbox_w;
        let mismatch_rate = mismatches as f32 / total as f32;
        assert!(
            mismatch_rate < 0.005,
            "mismatch rate {mismatch_rate:.4} > 0.5% tolerance \
             ({mismatches}/{total} pixels)"
        );
    }

    /// Letterbox path: the inverse letterbox transform must shift the
    /// bbox sample into the proto plane correctly when (width, height)
    /// are original-content pixel dims. A centred gaussian on the proto
    /// plane, aligned with the content region via letterbox, should
    /// still produce a centred foreground blob in the original-content
    /// coordinate frame.
    #[test]
    fn scaled_letterbox_original_content_coords() {
        let mut cpu = make_cpu();
        let proto_data = synthetic_proto_data(16, 16, 4);

        // Bbox in *model-input* normalized coords — the frame used by the
        // decoder output and consumed by `materialize_scaled_segmentations`.
        // Under the letterbox below, this maps to output-content normalized
        // (0.25, 0.3, 0.75, 0.7) and centres on the proto-plane centre.
        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.25, 0.3875, 0.75, 0.6125),
            score: 0.9,
            label: 0,
        }];

        // Letterbox: content fills full width, centred 56.25% vertically.
        // Caller requests Scaled(640, 360) in original-content space.
        let letterbox = Some([0.0_f32, 0.21875, 1.0, 0.78125]);
        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, letterbox, 640, 360)
            .expect("letterbox scaled rendering must succeed");

        assert_eq!(out.len(), 1);
        let seg = &out[0];
        let bbox_w = (0.5_f32 * 640.0_f32).round() as usize; // 320
        let bbox_h = (0.4_f32 * 360.0_f32).round() as usize; // 144
        assert_eq!(seg.segmentation.shape(), &[bbox_h, bbox_w, 1]);
        let uniq: std::collections::BTreeSet<u8> = seg.segmentation.iter().copied().collect();
        assert!(
            uniq.iter().all(|&v| v == 0 || v == 255),
            "letterbox scaled mask must be binary {{0, 255}}, got {uniq:?}"
        );

        // Middle-25%-square of the bbox should be solidly foreground
        // (the proto gaussian is centred; the bbox is centred on the
        // original-content frame; letterbox preserves centre alignment).
        let cy = bbox_h / 2;
        let cx = bbox_w / 2;
        let patch_h = bbox_h / 4;
        let patch_w = bbox_w / 4;
        let mut fg = 0_usize;
        let mut total = 0_usize;
        for y in cy.saturating_sub(patch_h / 2)..=(cy + patch_h / 2).min(bbox_h - 1) {
            for x in cx.saturating_sub(patch_w / 2)..=(cx + patch_w / 2).min(bbox_w - 1) {
                if seg.segmentation[[y, x, 0]] == 255 {
                    fg += 1;
                }
                total += 1;
            }
        }
        let fg_frac = fg as f32 / total as f32;
        assert!(
            fg_frac > 0.95,
            "centre of letterboxed bbox should be >95% foreground, got {fg_frac:.2}"
        );
    }

    /// Quantized i8 protos must produce the same binary mask as a float
    /// equivalent, modulo quantization rounding error.
    #[test]
    fn scaled_quantized_proto_produces_same_result_as_float() {
        use edgefirst_decoder::Quantization;

        let proto_h = 32;
        let proto_w = 32;
        let num_protos = 8;
        let mut protos_f32 = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    let v = ((y + x + k * 3) as f32 * 0.05).sin() * 3.0;
                    protos_f32[[y, x, k]] = v;
                }
            }
        }
        let scale = 0.1_f32;
        let zp = 0_i32;
        let protos_i8 = protos_f32.mapv(|v| (v / scale).round().clamp(-127.0, 127.0) as i8);
        let coeffs: Vec<f32> = (0..num_protos).map(|k| (k as f32 - 3.5) * 0.3).collect();

        let pd_float = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Float(protos_f32.clone()),
        };
        let pd_quant = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Quantized {
                protos: protos_i8,
                quantization: Quantization {
                    scale,
                    zero_point: zp,
                },
            },
        };

        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.1, 0.1, 0.9, 0.9),
            score: 0.9,
            label: 0,
        }];

        let mut cpu = make_cpu();
        let out_f = cpu
            .materialize_scaled_segmentations(&detect, &pd_float, None, 320, 320)
            .unwrap();
        let out_q = cpu
            .materialize_scaled_segmentations(&detect, &pd_quant, None, 320, 320)
            .unwrap();
        let mismatches = out_f[0]
            .segmentation
            .iter()
            .zip(out_q[0].segmentation.iter())
            .filter(|(a, b)| a != b)
            .count();
        let total = out_f[0].segmentation.len();
        assert!(
            mismatches < total / 200,
            "quantized and float diverged at {mismatches}/{total} pixels (>0.5%)"
        );
    }

    // ─── Batched-GEMM path regression tests ───────────────────────────────

    /// Build a `ProtoData` + detection list large enough to trigger the
    /// batched GEMM path (`N >= BATCHED_GEMM_MIN_N`) with realistic 160×160
    /// proto dims and 32 channels. Deterministic via seeded LCG so output
    /// is reproducible across runs.
    fn realistic_proto_data(n: usize) -> (Vec<DetectBox>, ProtoData) {
        let mut rng_seed: u32 = 0xABCD_0001;
        let mut next = || -> f32 {
            rng_seed = rng_seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng_seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let proto_h = 160;
        let proto_w = 160;
        let num_protos = 32;
        let mut protos = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    protos[[y, x, k]] = next() * 3.0;
                }
            }
        }
        let mask_coefficients: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..num_protos).map(|_| next()).collect())
            .collect();
        let detect: Vec<DetectBox> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                DetectBox {
                    bbox: BoundingBox::new(0.1 + 0.6 * t, 0.15, 0.3 + 0.6 * t, 0.85),
                    score: 0.9,
                    label: 0,
                }
            })
            .collect();
        let proto_data = ProtoData {
            mask_coefficients,
            protos: ProtoTensor::Float(protos),
        };
        (detect, proto_data)
    }

    /// `MaskResolution::Proto` batched path must match the fused path to
    /// within the tolerance of `fast_sigmoid`'s ~1.1% approximation error.
    /// Both paths share the same kernel math, so any larger divergence
    /// means the batched GEMM or ROI-crop mapping is wrong.
    #[test]
    fn batched_proto_matches_fused_reference_at_n_equals_5() {
        // Use N=16+ so the batched-GEMM path (threshold 16) actually fires.
        let (detect, proto_data) = realistic_proto_data(16);
        debug_assert!(detect.len() >= BATCHED_GEMM_MIN_N_PROTO);

        // Batched path via the public entry point.
        let mut cpu = make_cpu();
        let out_batched = cpu
            .materialize_segmentations(&detect, &proto_data, None)
            .unwrap();

        // Fused reference path via the small-N helper (bypasses the
        // BATCHED_GEMM_MIN_N dispatch in the public method).
        let out_fused = materialize_segmentations_fused(&detect, &proto_data, None).unwrap();

        assert_eq!(out_batched.len(), out_fused.len());
        for (b, f) in out_batched.iter().zip(out_fused.iter()) {
            assert_eq!(b.segmentation.shape(), f.segmentation.shape());
            // Count pixels differing by more than the u8-quant tolerance
            // (fast_sigmoid error ~1.1% ⇒ ~3 u8 units at the decision band).
            let mut mismatches = 0usize;
            for (a, c) in b.segmentation.iter().zip(f.segmentation.iter()) {
                if (*a as i32 - *c as i32).abs() > 3 {
                    mismatches += 1;
                }
            }
            let total = b.segmentation.len();
            assert!(
                mismatches < total / 100,
                "batched vs fused diverged at {mismatches}/{total} pixels (>1%)"
            );
        }
    }

    /// `MaskResolution::Scaled` batched path must match the per-detection
    /// non-batched path within the binary-threshold tolerance.
    #[test]
    fn batched_scaled_matches_unbatched_reference_at_n_equals_5() {
        // Use N=16+ so the batched-GEMM path (threshold 16) actually fires.
        let (detect, proto_data) = realistic_proto_data(16);
        debug_assert!(detect.len() >= BATCHED_GEMM_MIN_N_PROTO);

        let mut cpu = make_cpu();
        let out_batched = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, None, 320, 320)
            .unwrap();

        // Reference: call the per-detection helper directly. ProtoData is
        // Float so `scaled_segmentations_float` is the right reference.
        let protos = match &proto_data.protos {
            ProtoTensor::Float(p) => p,
            _ => unreachable!("test fixture uses Float protos"),
        };
        let out_ref = scaled_segmentations_float(
            &detect,
            &proto_data.mask_coefficients,
            protos,
            None,
            320,
            320,
        )
        .unwrap();

        assert_eq!(out_batched.len(), out_ref.len());
        for (b, r) in out_batched.iter().zip(out_ref.iter()) {
            assert_eq!(b.segmentation.shape(), r.segmentation.shape());
            let mut mismatches = 0usize;
            for (a, c) in b.segmentation.iter().zip(r.segmentation.iter()) {
                if *a != *c {
                    mismatches += 1;
                }
            }
            let total = b.segmentation.len();
            // Binary masks — tolerate up to 2% boundary pixel differences
            // due to sigmoid approximation at the decision band.
            assert!(
                mismatches < total / 50,
                "batched vs ref scaled diverged at {mismatches}/{total} pixels (>2%)"
            );
        }
    }

    /// MaskScratch buffers must be reused (capacity ≥ previous call) rather
    /// than reallocated each call. Regression guard for the pooling
    /// contract that validation loops depend on.
    #[test]
    fn mask_scratch_reuses_buffers_across_calls() {
        // N must be >= BATCHED_GEMM_MIN_N_PROTO so the batched path fires
        // and populates the scratch buffers.
        let (detect, proto_data) = realistic_proto_data(BATCHED_GEMM_MIN_N_PROTO);
        let mut cpu = make_cpu();
        cpu.materialize_segmentations(&detect, &proto_data, None)
            .unwrap();
        let cap_dequant = cpu.mask_scratch.dequant.capacity();
        let cap_coeffs = cpu.mask_scratch.coeffs.capacity();
        let cap_logits = cpu.mask_scratch.logits.capacity();
        assert!(cap_dequant > 0 && cap_coeffs > 0 && cap_logits > 0);

        // Second call with identical shape must reuse capacity.
        cpu.materialize_segmentations(&detect, &proto_data, None)
            .unwrap();
        assert_eq!(cpu.mask_scratch.dequant.capacity(), cap_dequant);
        assert_eq!(cpu.mask_scratch.coeffs.capacity(), cap_coeffs);
        assert_eq!(cpu.mask_scratch.logits.capacity(), cap_logits);
    }
}
