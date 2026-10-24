// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale quantized decoder — see
//! `.claude/plans/2026-04-28-per-scale-decoder-optimized-design.md`.

pub mod helper;
pub(crate) mod kernels;
pub(crate) mod outputs;
pub(crate) mod pipeline;
pub(crate) mod plan;
pub(crate) mod prefilter;

pub use helper::apply_schema_quant;

/// Output element type chosen by the user at `DecoderBuilder::with_decode_dtype()`.
///
/// The whole post-merge pipeline (boxes, scores, mask coefs, protos) is
/// emitted in this dtype. `F16` saves ~2× memory bandwidth at the cost of
/// 10-bit mantissa precision — empirically safe for YOLO-family models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeDtype {
    /// 32-bit float output (default).
    #[default]
    F32,
    /// 16-bit float output. Halves the memory bandwidth of the decode
    /// pipeline; on aarch64 it also keeps the NEON FP16 kernels in their
    /// native width instead of widening.
    F16,
}

/// Activation function applied after dequantization on a logical output.
///
/// Sourced from the schema's `activation_required` field. Currently only
/// `Sigmoid` is wired through the per-scale pipeline; future activations
/// (e.g. `Softmax` on objectness) extend this enum without ripple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // consumed by per-scale pipeline tasks
pub(crate) enum Activation {
    #[default]
    None,
    Sigmoid,
}

impl Activation {
    /// Translate a schema activation to a per_scale Activation.
    /// Returns Activation::None when the schema declares no activation.
    #[allow(dead_code)] // consumed by per-scale pipeline tasks
    pub(crate) fn from_schema(s: Option<crate::schema::Activation>) -> Self {
        match s {
            Some(crate::schema::Activation::Sigmoid) => Self::Sigmoid,
            _ => Self::None,
        }
    }
}

pub(crate) use outputs::{DecodedOutputBuffers, DecodedOutputsRef};
pub(crate) use plan::PerScalePlan;

/// Per-scale decoder for schema-v2 per-scale models. Built once at
/// `DecoderBuilder::build()` time; consumed per-frame via `run()`.
#[derive(Debug)]
#[allow(dead_code)] // Wired by Task 24's Decoder integration.
pub(crate) struct PerScaleDecoder {
    pub(crate) plan: PerScalePlan,
    pub(crate) buffers: DecodedOutputBuffers,
}

impl PerScaleDecoder {
    /// Build a decoder from a plan, allocating output buffers.
    #[allow(dead_code)] // Wired by Task 23's builder.
    pub(crate) fn new(plan: PerScalePlan) -> Self {
        let buffers = DecodedOutputBuffers::new(
            plan.out_dtype,
            plan.total_anchors,
            plan.num_classes,
            plan.num_mask_coefs,
            plan.proto_nhwc_shape.as_deref(),
        );
        Self { plan, buffers }
    }

    /// Decode one frame's worth of inputs.
    ///
    /// With a `score_threshold`, only anchors that can reach it are
    /// decoded and the outputs hold just those, in anchor order; `None`
    /// decodes every anchor.
    #[allow(dead_code)] // Wired by Task 24.
    pub(crate) fn run<'a>(
        &'a mut self,
        inputs: &[&edgefirst_tensor::TensorDyn],
        score_threshold: Option<f32>,
    ) -> crate::DecoderResult<DecodedOutputsRef<'a>> {
        pipeline::run(&self.plan, &mut self.buffers, inputs, score_threshold)
    }
}

/// Owned f32 snapshot of pre-NMS per-scale outputs.
///
/// Returned by [`crate::Decoder::_testing_run_per_scale_pre_nms`] and
/// used by integration tests to compare against fixture intermediates
/// without the noise of NMS ordering.
#[doc(hidden)]
pub struct PreNmsCapture {
    pub boxes_xywh: ndarray::Array2<f32>,
    pub scores: ndarray::Array2<f32>,
    pub mask_coefs: Option<ndarray::Array2<f32>>,
    pub protos: Option<ndarray::Array4<f32>>,
}

/// Kernel-level hooks for the decoder benchmark. Not a stable API.
#[doc(hidden)]
pub mod bench {
    use super::kernels::dfl_lut::{decode_dfl_lut, DflExpTable};
    use super::kernels::grids::make_anchor_grid;
    use super::kernels::level_box;
    use super::plan::{Layout, LevelPlan};
    use crate::Quantization;

    /// A DFL box kernel for 8-bit bins.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DflKernel {
        /// Dequantize, then f32 libm softmax per side (portable).
        Softmax,
        /// Exp table, portable.
        Lut,
        /// Dequantize, then fused NEON f32 softmax and weighted sum.
        NeonFused,
        /// Dequantize, then fused NEON softmax with an FP16 exp.
        NeonFusedFp16,
        /// Exp table with the four sides in NEON lanes.
        LutNeon,
    }

    impl DflKernel {
        /// Every kernel this CPU can run.
        pub fn available() -> Vec<Self> {
            let v = vec![Self::Softmax, Self::Lut];
            #[cfg(target_arch = "aarch64")]
            let v = {
                let mut v = v;
                v.extend([Self::NeonFused, Self::LutNeon]);
                if std::arch::is_aarch64_feature_detected!("fp16") {
                    v.push(Self::NeonFusedFp16);
                }
                v
            };
            v
        }
    }

    /// One DFL level (`reg_max` 16) and its quantization.
    pub struct DflLevel {
        plan: LevelPlan,
        q: Quantization,
        table: DflExpTable,
    }

    impl DflLevel {
        pub fn new(w: usize, h: usize, stride: f32, scale: f32, zero_point: i32) -> Self {
            let (grid_x, grid_y) = make_anchor_grid(h, w);
            Self {
                plan: LevelPlan {
                    stride,
                    h,
                    w,
                    reg_max: 16,
                    anchor_offset: 0,
                    grid_x,
                    grid_y,
                    box_shape: vec![1, h, w, 64].into_boxed_slice(),
                    score_shape: vec![1, h, w, 1].into_boxed_slice(),
                    mc_shape: None,
                    layout: Layout::Nhwc,
                },
                q: Quantization::new(scale, zero_point),
                table: DflExpTable::new(scale),
            }
        }

        pub fn anchors(&self) -> usize {
            self.plan.h * self.plan.w
        }

        /// Decode `input` (`anchors() * 64` bins) into `dst` (`anchors() * 4`).
        pub fn decode(&self, kernel: DflKernel, input: &[i8], dst: &mut [f32]) {
            let (p, q) = (&self.plan, self.q);
            match kernel {
                DflKernel::Softmax => level_box::decode_box_level_dfl_i8_to_f32(input, q, p, dst),
                DflKernel::Lut => decode_dfl_lut(input, &self.table, p, dst),
                #[cfg(target_arch = "aarch64")]
                DflKernel::NeonFused => {
                    level_box::decode_box_level_dfl_i8_to_f32_neon_fused(input, q, p, dst)
                }
                #[cfg(target_arch = "aarch64")]
                DflKernel::NeonFusedFp16 => {
                    assert!(std::arch::is_aarch64_feature_detected!("fp16"));
                    level_box::decode_box_level_dfl_i8_to_f32_neon_fused_fp16(input, q, p, dst)
                }
                #[cfg(target_arch = "aarch64")]
                DflKernel::LutNeon => {
                    super::kernels::dfl_lut::decode_dfl_lut_neon(input, &self.table, p, dst)
                }
                #[cfg(not(target_arch = "aarch64"))]
                _ => panic!("{kernel:?} is not available on this CPU"),
            }
        }
    }
}
