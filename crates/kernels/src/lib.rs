// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Quantized and floating-point compute kernels for edge AI inference.

pub mod convert;
mod cpu;
mod error;
mod ops;
mod pool;
#[doc(hidden)]
pub mod prim;
mod quant;
#[cfg(test)]
mod test_support;
mod tflite;
mod view;

pub use cpu::{CpuFeatures, Tier};
pub use error::{KernelError, Result};
pub use ops::attention_i8::{AttentionI8Params, AttentionI8Plan, AttentionQMul, AttentionScratch};
pub use ops::binary_i8::{BinaryI8Params, BinaryI8Plan, BinaryOp};
pub use ops::dfl::{DflParams, DflPlan};
pub use ops::dfl_box::{DflBoxParams, DflBoxPlan};
pub use ops::dist2bbox::{Dist2BboxParams, Dist2BboxPlan};
pub use ops::levels::{BoxFormat, Level};
pub use ops::lut_i8::LutI8Plan;
pub use ops::matmul_i8::{MatMulI8Params, MatMulI8Plan, Requant};
pub use ops::sigmoid::{SigmoidParams, SigmoidPlan};
pub use ops::softmax_dot_i8::{SoftmaxDotI8Params, SoftmaxDotI8Plan};
pub use ops::softmax_i16::{SoftmaxI16Params, SoftmaxI16Plan};
pub use ops::softmax_i8::{SoftmaxI8Params, SoftmaxI8Plan};
pub use pool::{Pool, ScopedThreads};
pub use quant::{quantize, Activation, DType, Quant, QuantInt, TfliteKernels};
pub use view::{ByteSpan, TensorMut, TensorRef, View, ViewMut, MAX_RANK};
