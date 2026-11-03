// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The NEON mask plans: one per call, one mask per detection, a row chunk of logits at a time.

use edgefirst_kernels::prim::neon_mask::{
    mask_dot8_nchw_neon, mask_dot8_nhwc_dotprod, mask_dot8_nhwc_neon, mask_dot_f32_nchw_neon,
    mask_dot_f32_nhwc_neon, mask_dot_wide_nchw_neon, mask_dot_wide_nhwc_neon,
    mask_sigmoid_u8_f32_neon, mask_sigmoid_u8_i32_neon, MaskFloatIn, MaskInt, MaskInt8, MAX_K,
};
use half::f16;
use ndarray::{Array3, ArrayView1, ArrayView3};
use num_traits::{AsPrimitive, PrimInt};

use super::{cast, proto_memory, Crop, Isa, Layout};
#[cfg(test)]
use super::{FloatKernel, QuantKernel};
use crate::{DecoderError, Quantization};

/// Logits per kernel call: one stack buffer.
const CHUNK: usize = 64;
/// Coefficients gathered on the stack; more channels take a heap copy.
const STACK_K: usize = 256;

/// A slice of one of the integer types the NEON kernels take.
#[derive(Clone, Copy)]
enum Ints<'a> {
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U16(&'a [u16]),
}

impl<'a> Ints<'a> {
    fn of<T: 'static>(s: &'a [T]) -> Option<Self> {
        cast(s)
            .map(Self::I8)
            .or_else(|| cast(s).map(Self::U8))
            .or_else(|| cast(s).map(Self::I16))
            .or_else(|| cast(s).map(Self::U16))
    }

    fn zp_ok(&self, zp: i32) -> bool {
        fn ok<T: MaskInt>(zp: i32) -> bool {
            (T::MIN..=T::MAX).contains(&zp)
        }
        match self {
            Self::I8(_) => ok::<i8>(zp),
            Self::U8(_) => ok::<u8>(zp),
            Self::I16(_) => ok::<i16>(zp),
            Self::U16(_) => ok::<u16>(zp),
        }
    }

    #[cfg(test)]
    fn is_8bit(&self) -> bool {
        matches!(self, Self::I8(_) | Self::U8(_))
    }
}

/// A slice of one of the float types the NEON kernels take.
#[derive(Clone, Copy)]
enum Floats<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

/// One detection's values as `U`, contiguous: on the stack up to `STACK_K` channels.
fn gather<T: Copy, U: Copy, R>(
    row: ArrayView1<T>,
    zero: U,
    conv: impl Fn(T) -> U,
    f: impl FnOnce(&[U]) -> R,
) -> R {
    if row.len() <= STACK_K {
        let mut buf = [zero; STACK_K];
        for (d, &s) in buf.iter_mut().zip(row.iter()) {
            *d = conv(s);
        }
        f(&buf[..row.len()])
    } else {
        f(&row.iter().map(|&s| conv(s)).collect::<Vec<U>>())
    }
}

fn kernel_failed() -> DecoderError {
    DecoderError::Internal("mask kernel rejected its arguments".into())
}

/// Fills the `u8` mask of `crop` a row chunk at a time: `dot(px, logits)` writes the logits of
/// the pixels from grid index `px` on, `sig` turns them into mask values.
fn fill_rows<L: Copy + Default>(
    crop: Crop,
    w: usize,
    out: &mut [u8],
    dot: impl Fn(usize, &mut [L]) -> bool,
    sig: impl Fn(&[L], &mut [u8]) -> bool,
) -> Result<(), DecoderError> {
    let cw = crop.x1 - crop.x0;
    if cw == 0 {
        return Ok(());
    }
    let mut buf = [L::default(); CHUNK];
    for (y, row) in (crop.y0..crop.y1).zip(out.chunks_exact_mut(cw)) {
        let mut px = y * w + crop.x0;
        for chunk in row.chunks_mut(CHUNK) {
            let logits = &mut buf[..chunk.len()];
            if !(dot(px, logits) && sig(logits, chunk)) {
                return Err(kernel_failed());
            }
            px += chunk.len();
        }
    }
    Ok(())
}

/// An empty `(H, W, 1)` mask for `crop`, and the call that fills it.
fn crop_mask(
    crop: Crop,
    fill: impl FnOnce(&mut [u8]) -> Result<(), DecoderError>,
) -> Result<Array3<u8>, DecoderError> {
    let (ch, cw) = (crop.y1 - crop.y0, crop.x1 - crop.x0);
    let mut out = vec![0u8; ch * cw];
    fill(&mut out)?;
    Array3::from_shape_vec((ch, cw, 1), out)
        .map_err(|e| DecoderError::Internal(format!("mask shape: {e}")))
}

/// A quantized mask call on a NEON tier, set up once per call.
pub(crate) struct QuantPlan<'a> {
    dotprod: bool,
    protos: Ints<'a>,
    layout: Layout,
    h: usize,
    w: usize,
    k: usize,
    zm: i32,
    zp: i32,
    scale: f32,
}

impl<'a> QuantPlan<'a> {
    /// `None` when the call needs the scalar code.
    pub(crate) fn new<MASK: 'static, PROTO: 'static>(
        isa: Isa,
        protos: &'a ArrayView3<'_, PROTO>,
        quant_masks: Quantization,
        quant_protos: Quantization,
    ) -> Option<Self> {
        if isa.is_scalar() {
            return None;
        }
        let mask_kind = Ints::of::<MASK>(&[])?;
        let (mem, layout) = proto_memory(protos)?;
        let ints = Ints::of(mem)?;
        let (h, w, k) = protos.dim();
        let ok = k <= MAX_K
            && mask_kind.zp_ok(quant_masks.zero_point)
            && ints.zp_ok(quant_protos.zero_point);
        ok.then_some(Self {
            dotprod: isa.dotprod(),
            protos: ints,
            layout,
            h,
            w,
            k,
            zm: quant_masks.zero_point,
            zp: quant_protos.zero_point,
            scale: quant_masks.scale * quant_protos.scale,
        })
    }

    /// The kernel this call runs for `MASK` coefficients.
    #[cfg(test)]
    pub(crate) fn kernel<MASK: 'static>(&self) -> QuantKernel {
        let eight = self.protos.is_8bit() && Ints::of::<MASK>(&[]).is_some_and(|m| m.is_8bit());
        match (self.layout, eight, self.dotprod) {
            (Layout::Nhwc, false, _) => QuantKernel::WideNhwc,
            (Layout::Nchw, false, _) => QuantKernel::WideNchw,
            (Layout::Nhwc, true, true) => QuantKernel::Dot8NhwcDotprod,
            (Layout::Nhwc, true, false) => QuantKernel::Dot8Nhwc,
            (Layout::Nchw, true, _) => QuantKernel::Dot8Nchw,
        }
    }

    /// The `(H, W, 1)` mask of one detection over `crop`.
    pub(crate) fn mask<MASK: PrimInt + 'static>(
        &self,
        coeff: ArrayView1<MASK>,
        crop: Crop,
    ) -> Result<Array3<u8>, DecoderError> {
        if coeff.len() != self.k {
            return Err(kernel_failed());
        }
        crop_mask(crop, |out| {
            gather(
                coeff,
                MASK::zero(),
                |v| v,
                |c| match Ints::of(c) {
                    Some(Ints::I8(c)) => self.eight(c, crop, out),
                    Some(Ints::U8(c)) => self.eight(c, crop, out),
                    Some(Ints::I16(c)) => self.wide(c, crop, out),
                    Some(Ints::U16(c)) => self.wide(c, crop, out),
                    None => Err(kernel_failed()),
                },
            )
        })
    }

    fn eight<M: MaskInt8>(&self, c: &[M], crop: Crop, out: &mut [u8]) -> Result<(), DecoderError> {
        match self.protos {
            Ints::I8(p) => self.rows8(c, p, crop, out),
            Ints::U8(p) => self.rows8(c, p, crop, out),
            Ints::I16(p) => self.rows_wide(c, p, crop, out),
            Ints::U16(p) => self.rows_wide(c, p, crop, out),
        }
    }

    fn wide<M: MaskInt>(&self, c: &[M], crop: Crop, out: &mut [u8]) -> Result<(), DecoderError> {
        match self.protos {
            Ints::I8(p) => self.rows_wide(c, p, crop, out),
            Ints::U8(p) => self.rows_wide(c, p, crop, out),
            Ints::I16(p) => self.rows_wide(c, p, crop, out),
            Ints::U16(p) => self.rows_wide(c, p, crop, out),
        }
    }

    fn rows8<M: MaskInt8, P: MaskInt8>(
        &self,
        c: &[M],
        p: &[P],
        crop: Crop,
        out: &mut [u8],
    ) -> Result<(), DecoderError> {
        let (k, zm, zp, scale) = (self.k, self.zm, self.zp, self.scale);
        // SAFETY: NEON is part of the aarch64 baseline.
        let sig = |l: &[i32], o: &mut [u8]| unsafe { mask_sigmoid_u8_i32_neon(l, scale, o) };
        match self.layout {
            Layout::Nhwc => {
                let f = if self.dotprod {
                    mask_dot8_nhwc_dotprod::<M, P>
                } else {
                    mask_dot8_nhwc_neon::<M, P>
                };
                let dot = |px: usize, d: &mut [i32]| {
                    // SAFETY: NEON is baseline; `dotprod` implies detected FEAT_DotProd (`Isa::of`).
                    p.get(px * k..(px + d.len()) * k)
                        .is_some_and(|s| unsafe { f(c, zm, s, zp, d) })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
            Layout::Nchw => {
                let stride = self.h * self.w;
                let dot = |px: usize, d: &mut [i32]| {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    p.get(px..)
                        .is_some_and(|s| unsafe { mask_dot8_nchw_neon(c, zm, s, stride, zp, d) })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
        }
    }

    fn rows_wide<M: MaskInt, P: MaskInt>(
        &self,
        c: &[M],
        p: &[P],
        crop: Crop,
        out: &mut [u8],
    ) -> Result<(), DecoderError> {
        let (k, zm, zp, scale) = (self.k, self.zm, self.zp, self.scale);
        // SAFETY: NEON is part of the aarch64 baseline.
        let sig = |l: &[f32], o: &mut [u8]| unsafe { mask_sigmoid_u8_f32_neon(l, scale, o) };
        match self.layout {
            Layout::Nhwc => {
                let dot = |px: usize, d: &mut [f32]| {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    p.get(px * k..(px + d.len()) * k)
                        .is_some_and(|s| unsafe { mask_dot_wide_nhwc_neon(c, zm, s, zp, d) })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
            Layout::Nchw => {
                let stride = self.h * self.w;
                let dot = |px: usize, d: &mut [f32]| {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    p.get(px..).is_some_and(|s| unsafe {
                        mask_dot_wide_nchw_neon(c, zm, s, stride, zp, d)
                    })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
        }
    }
}

/// A float mask call on a NEON tier, set up once per call.
pub(crate) struct FloatPlan<'a> {
    protos: Floats<'a>,
    layout: Layout,
    h: usize,
    w: usize,
    k: usize,
}

impl<'a> FloatPlan<'a> {
    /// `None` when the call needs the scalar code.
    pub(crate) fn new<PROTO: 'static>(isa: Isa, protos: &'a ArrayView3<'_, PROTO>) -> Option<Self> {
        if isa.is_scalar() {
            return None;
        }
        let (mem, layout) = proto_memory(protos)?;
        let floats = cast(mem)
            .map(Floats::F32)
            .or_else(|| cast(mem).map(Floats::F16))?;
        let (h, w, k) = protos.dim();
        Some(Self {
            protos: floats,
            layout,
            h,
            w,
            k,
        })
    }

    /// The kernel this call runs.
    #[cfg(test)]
    pub(crate) fn kernel(&self) -> FloatKernel {
        match self.layout {
            Layout::Nhwc => FloatKernel::F32Nhwc,
            Layout::Nchw => FloatKernel::F32Nchw,
        }
    }

    /// The `(H, W, 1)` mask of one detection over `crop`.
    pub(crate) fn mask<MASK: Copy + AsPrimitive<f32>>(
        &self,
        coeff: ArrayView1<MASK>,
        crop: Crop,
    ) -> Result<Array3<u8>, DecoderError> {
        if coeff.len() != self.k {
            return Err(kernel_failed());
        }
        crop_mask(crop, |out| {
            gather(
                coeff,
                0.0,
                |v| v.as_(),
                |c| match self.protos {
                    Floats::F32(p) => self.rows(c, p, crop, out),
                    Floats::F16(p) => self.rows(c, p, crop, out),
                },
            )
        })
    }

    fn rows<P: MaskFloatIn>(
        &self,
        c: &[f32],
        p: &[P],
        crop: Crop,
        out: &mut [u8],
    ) -> Result<(), DecoderError> {
        let k = self.k;
        // SAFETY: NEON is part of the aarch64 baseline.
        let sig = |l: &[f32], o: &mut [u8]| unsafe { mask_sigmoid_u8_f32_neon(l, 1.0, o) };
        match self.layout {
            Layout::Nhwc => {
                let dot = |px: usize, d: &mut [f32]| {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    p.get(px * k..(px + d.len()) * k)
                        .is_some_and(|s| unsafe { mask_dot_f32_nhwc_neon(c, s, d) })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
            Layout::Nchw => {
                let stride = self.h * self.w;
                let dot = |px: usize, d: &mut [f32]| {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    p.get(px..)
                        .is_some_and(|s| unsafe { mask_dot_f32_nchw_neon(c, s, stride, d) })
                };
                fill_rows(crop, self.w, out, dot, sig)
            }
        }
    }
}
