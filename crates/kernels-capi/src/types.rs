// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The `#[repr(C)]` value types that cross the ABI.

use std::ffi::c_void;

use edgefirst_kernels as k;

/// Affine quantization: `real = scale * (q - zero_point)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfKernelQuant {
    pub scale: f32,
    pub zero_point: i32,
}

impl From<EfKernelQuant> for k::Quant {
    fn from(q: EfKernelQuant) -> Self {
        k::Quant::new(q.scale, q.zero_point)
    }
}

/// Element type of a tensor. Params and views carry it as a `uint32_t` holding one of these values.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfKernelDtype {
    I8 = 0,
    U8 = 1,
    I16 = 2,
    U16 = 3,
    /// 16-bit IEEE half, stored as `uint16_t`.
    F16 = 4,
    F32 = 5,
}

impl EfKernelDtype {
    /// The kernels dtype for a raw ABI value, or `None` if the value is unknown.
    pub fn from_raw(v: u32) -> Option<k::DType> {
        Some(match v {
            0 => k::DType::I8,
            1 => k::DType::U8,
            2 => k::DType::I16,
            3 => k::DType::U16,
            4 => k::DType::F16,
            5 => k::DType::F32,
            _ => return None,
        })
    }
}

/// Layout of the four decoded box values.
///
/// cbindgen:prefix-with-name=false
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum EfKernelBoxFormat {
    /// `[xc, yc, w, h]`.
    EfKernelBoxXywh = 0,
    /// `[x1, y1, x2, y2]`.
    EfKernelBoxXyxy = 1,
}

impl EfKernelBoxFormat {
    /// The kernels box format for a raw ABI value, or `None` if the value is unknown.
    pub fn from_raw(v: u32) -> Option<k::BoxFormat> {
        match v {
            0 => Some(k::BoxFormat::Xywh),
            1 => Some(k::BoxFormat::Xyxy),
            _ => None,
        }
    }
}

/// A caller-owned tensor: data pointer, dtype, shape and optional element strides.
///
/// A plan run rejects with EINVAL a view whose `data` is not aligned to its element type
/// (2 bytes for 16-bit types, 4 for float32), and an input and output whose memory spans
/// overlap, interleaved views of one buffer included. A span runs from `data` to one past the
/// last element the shape and strides address.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfKernelView {
    /// First element, aligned to the element type; may be NULL only when the view has no elements.
    pub data: *mut c_void,
    /// An `ef_kernel_dtype` value.
    pub dtype: u32,
    /// Number of dimensions, at most 6.
    pub ndim: usize,
    /// `ndim` extents.
    pub shape: *const usize,
    /// `ndim` strides in elements (not bytes), or NULL for a contiguous row-major layout.
    pub strides: *const isize,
}

/// One FPN level of a concatenated anchor axis.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfKernelLevel {
    pub anchors: usize,
    pub grid_w: usize,
    pub stride: f32,
}

impl From<EfKernelLevel> for k::Level {
    fn from(l: EfKernelLevel) -> Self {
        k::Level {
            anchors: l.anchors,
            grid_w: l.grid_w,
            stride: l.stride,
        }
    }
}

/// A caller-supplied thread pool.
///
/// `run` must call `task(arg, i)` for each `i` in `0..n_tasks` (on any threads, in any order)
/// and return only after all calls have finished. `ctx` is passed back unchanged. `run` is
/// invoked from whichever thread calls `plan_run`, so it must tolerate concurrent calls.
/// Tasks keep staging buffers of up to 32 KiB on the stack, so worker threads need about
/// 64 KiB of stack headroom beyond the pool's own use.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfKernelPool {
    pub ctx: *mut c_void,
    pub run: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            n_tasks: usize,
            task: unsafe extern "C" fn(arg: *mut c_void, index: usize),
            arg: *mut c_void,
        ),
    >,
}

/// Parameters of an int8 softmax; the output must be quantized (1/256, -128).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfSoftmaxI8Params {
    pub input: EfKernelQuant,
    pub output: EfKernelQuant,
    pub beta: f32,
}

/// Parameters of an int16 softmax.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfSoftmaxI16Params {
    pub input: EfKernelQuant,
    pub output: EfKernelQuant,
    pub beta: f32,
}

/// Parameters of a sigmoid.
///
/// A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
/// float tensor may pass `{0, 0}`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfSigmoidParams {
    pub input_dtype: u32,
    pub input: EfKernelQuant,
    pub output_dtype: u32,
    pub output: EfKernelQuant,
}

/// Parameters of a distribution-focal-loss decode.
///
/// A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
/// float tensor may pass `{0, 0}`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfDflParams {
    pub input_dtype: u32,
    pub input: EfKernelQuant,
    pub output_dtype: u32,
    pub output: EfKernelQuant,
    pub reg_max: usize,
}

/// Parameters of a distance-to-box decode.
///
/// A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
/// float tensor may pass `{0, 0}`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfDist2bboxParams {
    pub input_dtype: u32,
    pub input: EfKernelQuant,
    pub output_dtype: u32,
    pub output: EfKernelQuant,
    /// `n_levels` entries.
    pub levels: *const EfKernelLevel,
    pub n_levels: usize,
    /// An `ef_kernel_box_format` value.
    pub box_format: u32,
}

/// Parameters of the fused DFL plus distance-to-box decode.
///
/// A float `input_dtype` ignores `input` and a float `output_dtype` ignores `output`, so a
/// float tensor may pass `{0, 0}`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfDflBoxParams {
    pub input_dtype: u32,
    pub input: EfKernelQuant,
    pub output_dtype: u32,
    pub output: EfKernelQuant,
    pub reg_max: usize,
    /// `n_levels` entries.
    pub levels: *const EfKernelLevel,
    pub n_levels: usize,
    /// An `ef_kernel_box_format` value.
    pub box_format: u32,
}

/// Which of TFLite's int8 kernels a plan reproduces, where they round differently. Params carry
/// it as a `uint32_t` holding one of these values.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfTfliteKernels {
    /// int8 `ADD` and `MUL` in 16-element NEON blocks, matrix products through ruy: every aarch64
    /// build.
    NeonRuy = 0,
    /// NEON blocks, matrix products through gemmlowp (rounded as the reference kernels): x86
    /// builds with SSE4.1, such as the LiteRT Python wheels.
    NeonGemmlowp = 1,
    /// Scalar `ADD` and `MUL`, matrix products through ruy: builds without NEON.
    Ruy = 2,
    /// The reference kernels (`BuiltinRefOpResolver`).
    Reference = 3,
}

impl EfTfliteKernels {
    /// The kernels variant for a raw ABI value, or `None` if the value is unknown.
    pub fn from_raw(v: u32) -> Option<k::TfliteKernels> {
        match v {
            0 => Some(k::TfliteKernels::NeonRuy),
            1 => Some(k::TfliteKernels::NeonGemmlowp),
            2 => Some(k::TfliteKernels::Ruy),
            3 => Some(k::TfliteKernels::Reference),
            _ => None,
        }
    }
}

/// A fused activation, applied as the output clamp TFLite derives from the output quantization.
/// The values are TFLite's `TfLiteFusedActivation`; params carry it as a `uint32_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum EfKernelActivation {
    None = 0,
    Relu = 1,
    /// Clamp to [-1, 1].
    ReluN1_To_1 = 2,
    Relu6 = 3,
}

impl EfKernelActivation {
    /// The activation for a raw ABI value, or `None` if the value is unknown.
    pub fn from_raw(v: u32) -> Option<k::Activation> {
        match v {
            0 => Some(k::Activation::None),
            1 => Some(k::Activation::Relu),
            2 => Some(k::Activation::ReluN1To1),
            3 => Some(k::Activation::Relu6),
            _ => None,
        }
    }
}

/// The elementwise operation of an int8 binary plan; params carry it as a `uint32_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfBinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
}

impl EfBinaryOp {
    /// The operation for a raw ABI value, or `None` if the value is unknown.
    pub fn from_raw(v: u32) -> Option<k::BinaryOp> {
        match v {
            0 => Some(k::BinaryOp::Add),
            1 => Some(k::BinaryOp::Sub),
            2 => Some(k::BinaryOp::Mul),
            _ => None,
        }
    }
}

/// A TFLite fixed-point requantization: `acc * multiplier * 2^(shift - 31)`, rounded as the
/// plan's kernels variant does.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EfKernelRequant {
    pub multiplier: i32,
    pub shift: i32,
}

impl From<EfKernelRequant> for k::Requant {
    fn from(r: EfKernelRequant) -> Self {
        k::Requant {
            multiplier: r.multiplier,
            shift: r.shift,
        }
    }
}

impl From<k::Requant> for EfKernelRequant {
    fn from(r: k::Requant) -> Self {
        Self {
            multiplier: r.multiplier,
            shift: r.shift,
        }
    }
}

/// Parameters of an int8 `ADD`, `SUB` or `MUL`: `output = lhs op rhs`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfBinaryI8Params {
    /// An `ef_binary_op` value.
    pub op: u32,
    pub lhs: EfKernelQuant,
    pub rhs: EfKernelQuant,
    pub output: EfKernelQuant,
    /// An `ef_kernel_activation` value.
    pub activation: u32,
    /// An `ef_tflite_kernels` value.
    pub kernels: u32,
}

/// Parameters of an int8 matrix multiplication
/// `out[.., m, n] = clamp(zo + R[n](sum_k (lhs[.., m, k] - zl) * (rhs[.., k, n] - zr) + bias[n]))`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfMatmulI8Params {
    pub lhs_zero_point: i32,
    pub rhs_zero_point: i32,
    /// The output zero point and scale; the scale only places the activation clamp.
    pub output: EfKernelQuant,
    /// `n_requant` entries: one for every column, or one per output column.
    pub requant: *const EfKernelRequant,
    pub n_requant: usize,
    /// `n_bias` entries added to each column's accumulator; `n_bias` 0 for none.
    pub bias: *const i32,
    pub n_bias: usize,
    /// An `ef_kernel_activation` value.
    pub activation: u32,
    /// An `ef_tflite_kernels` value.
    pub kernels: u32,
}

/// Parameters of an int8 softmax over the last axis followed by a product with one weight
/// column (a YOLO DFL head); `projection` must have one requantization, at most one bias and
/// the softmax output's zero point as its left zero point.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfSoftmaxDotI8Params {
    pub softmax: EfSoftmaxI8Params,
    /// `n_weights` entries, one per element of a row.
    pub weights: *const i8,
    pub n_weights: usize,
    pub projection: EfMatmulI8Params,
}

/// The scalar `MUL` a TFLite graph applies to Q before the attention scores.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfAttentionQMul {
    /// The `MUL` output for every Q value, indexed by the raw byte (see
    /// `ef_binary_i8_plan_table_for_rhs`).
    pub table: [u8; 256],
    /// The quantization of the `MUL` output.
    pub output: EfKernelQuant,
}

/// Parameters of an int8 attention block: `[MUL] -> BATCH_MATMUL -> SOFTMAX -> BATCH_MATMUL`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfAttentionI8Params {
    pub q: EfKernelQuant,
    /// The scale applied to Q first, or NULL when the graph has none.
    pub q_mul: *const EfAttentionQMul,
    pub k: EfKernelQuant,
    pub v: EfKernelQuant,
    /// The quantization of the int8 scores.
    pub scores: EfKernelQuant,
    /// The softmax `beta`.
    pub beta: f32,
    /// The quantization of the int8 probabilities: scale 1/256, zero point -128.
    pub probs: EfKernelQuant,
    pub output: EfKernelQuant,
    /// An `ef_tflite_kernels` value.
    pub kernels: u32,
}

macro_rules! opaque_plan {
    ($($name:ident($inner:ty)),* $(,)?) => {$(
        /// An opaque plan; its definition is private.
        pub struct $name(pub(crate) $inner);
    )*};
}

opaque_plan! {
    EfSoftmaxI8Plan(k::SoftmaxI8Plan),
    EfSoftmaxI16Plan(k::SoftmaxI16Plan),
    EfSigmoidPlan(k::SigmoidPlan),
    EfDflPlan(k::DflPlan),
    EfDist2bboxPlan(k::Dist2BboxPlan),
    EfDflBoxPlan(k::DflBoxPlan),
    EfLutI8Plan(k::LutI8Plan),
    EfBinaryI8Plan(k::BinaryI8Plan),
    EfMatmulI8Plan(k::MatMulI8Plan),
    EfSoftmaxDotI8Plan(k::SoftmaxDotI8Plan),
    EfAttentionI8Plan(k::AttentionI8Plan),
}

/// Opaque per-call packing scratch of an attention plan; its definition is private.
pub struct EfAttentionScratch(pub(crate) k::AttentionScratch);
