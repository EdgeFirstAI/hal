// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale quantized decoder — see
//! `.claude/plans/2026-04-28-per-scale-decoder-optimized-design.md`.

pub mod helper;
pub(crate) mod kernels;
pub(crate) mod outputs;
pub(crate) mod pipeline;
pub(crate) mod plan;

pub use helper::apply_schema_quant;

/// Output element type chosen by the user at `DecoderBuilder::with_decode_dtype()`.
///
/// The whole post-merge pipeline (boxes, scores, mask coefs, protos) is
/// emitted in this dtype. `F16` saves ~2× memory bandwidth at the cost of
/// 10-bit mantissa precision — empirically safe for YOLO-family models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeDtype {
    #[default]
    F32,
    F16,
}

/// Activation function applied after dequantization on a logical output.
///
/// Sourced from the schema's `activation_required` field. Currently only
/// `Sigmoid` is wired through the per-scale pipeline; future activations
/// (e.g. `Softmax` on objectness) extend this enum without ripple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // consumed by later per-scale phase 1 tasks
pub(crate) enum Activation {
    #[default]
    None,
    Sigmoid,
}

impl Activation {
    /// Translate a schema activation to a per_scale Activation.
    /// Returns Activation::None when the schema declares no activation.
    #[allow(dead_code)] // consumed by later per-scale phase 1 tasks
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
            plan.proto_shape.as_deref(),
        );
        Self { plan, buffers }
    }

    /// Decode one frame's worth of inputs.
    #[allow(dead_code)] // Wired by Task 24.
    pub(crate) fn run<'a>(
        &'a mut self,
        inputs: &[&edgefirst_tensor::TensorDyn],
    ) -> crate::DecoderResult<DecodedOutputsRef<'a>> {
        pipeline::run(&self.plan, &mut self.buffers, inputs)
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
