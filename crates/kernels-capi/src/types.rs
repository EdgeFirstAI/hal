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
}
