// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decoded scores of 8-bit score tensors by table lookup.
//!
//! An 8-bit score has 256 raw values, so a level's score kernel is run over
//! all of them once per quantization and every score of a frame is then
//! read from the table. The probe is a whole number of the kernel's
//! vector chunks, so each entry is the value the kernel's vector path
//! gives that raw score.

use super::kernels::dispatch::{DstSliceMut, InputView};
use super::plan::{Layout, LevelPlan, PerScalePlan};
use super::DecodeDtype;
use crate::{DecoderResult, Quantization};
use edgefirst_tensor::DType;
use half::f16;

/// Raw values of an 8-bit score.
const RAW_VALUES: usize = 256;

#[derive(Debug)]
enum Values {
    F32(Box<[f32; RAW_VALUES]>),
    F16(Box<[f16; RAW_VALUES]>),
}

/// One level's decoded score per raw value, indexed by the raw byte.
#[derive(Debug, Default)]
pub(crate) struct ScoreTable {
    /// Score dtype, scale bits and zero point `values` was built for.
    key: Option<(DType, u32, i32)>,
    values: Option<Values>,
    /// A 1 x 256 level for the probe.
    probe: Option<LevelPlan>,
}

/// `dst[k] = table[src[k]]`, the raw byte as the index. Eight indices are read before their
/// eight lookups, so the lookups are independent loads an in-order core can overlap.
fn lookup<S: Copy, T: Copy>(table: &[T; RAW_VALUES], src: &[S], dst: &mut [T], index: fn(S) -> u8) {
    const STEP: usize = 8;
    let mut d = dst.chunks_exact_mut(STEP);
    let mut s = src.chunks_exact(STEP);
    for (d, s) in (&mut d).zip(&mut s) {
        let at: [usize; STEP] =
            std::array::from_fn(|k| s.get(k).map_or(0, |&s| usize::from(index(s))));
        let values = at.map(|i| table[i]);
        d.copy_from_slice(&values);
    }
    for (d, &s) in d.into_remainder().iter_mut().zip(s.remainder()) {
        *d = table[usize::from(index(s))];
    }
}

impl ScoreTable {
    /// Point the table at `dtype` scores quantized by `q` on `level`: built
    /// when the dtype or quantization changes, and empty for scores that
    /// are not 8-bit.
    pub(crate) fn update(
        &mut self,
        plan: &PerScalePlan,
        level: &LevelPlan,
        dtype: DType,
        q: Quantization,
    ) -> DecoderResult<()> {
        if !matches!(dtype, DType::I8 | DType::U8) {
            self.key = None;
            self.values = None;
            return Ok(());
        }
        let key = (dtype, q.scale.to_bits(), q.zero_point);
        if self.key == Some(key) && self.values.is_some() {
            return Ok(());
        }
        self.key = None;
        let probe = self.probe.get_or_insert_with(|| LevelPlan {
            stride: level.stride,
            h: 1,
            w: RAW_VALUES,
            reg_max: level.reg_max,
            anchor_offset: 0,
            grid_x: Box::default(),
            grid_y: Box::default(),
            box_shape: level.box_shape.clone(),
            score_shape: level.score_shape.clone(),
            mc_shape: level.mc_shape.clone(),
            layout: Layout::Nhwc,
        });
        let bytes: [u8; RAW_VALUES] = std::array::from_fn(|i| i as u8);
        let signed = bytes.map(|b| b as i8);
        let input = match dtype {
            DType::I8 => InputView::I8(&signed),
            _ => InputView::U8(&bytes),
        };
        let run = |dst: DstSliceMut<'_>| {
            plan.score_dispatch
                .run(input, q, 1, probe, plan.score_activation, dst)
        };
        let values = match plan.out_dtype {
            DecodeDtype::F32 => {
                let mut v = Box::new([0f32; RAW_VALUES]);
                run(DstSliceMut::F32(&mut v[..]))?;
                Values::F32(v)
            }
            DecodeDtype::F16 => {
                let mut v = Box::new([f16::ZERO; RAW_VALUES]);
                run(DstSliceMut::F16(&mut v[..]))?;
                Values::F16(v)
            }
        };
        self.values = Some(values);
        self.key = Some(key);
        Ok(())
    }

    /// Decode `input` into `dst` from the table. `Some(dst)`, untouched,
    /// when the table does not cover the call: no table, another dtype
    /// than it was built for, or mismatched lengths.
    pub(crate) fn decode<'d>(
        &self,
        input: &InputView<'_>,
        dst: DstSliceMut<'d>,
    ) -> Option<DstSliceMut<'d>> {
        let (Some((dtype, ..)), Some(values)) = (self.key, self.values.as_ref()) else {
            return Some(dst);
        };
        if input.len() != dst.len() {
            return Some(dst);
        }
        match (input, values, dst) {
            (InputView::I8(s), Values::F32(t), DstSliceMut::F32(d)) if dtype == DType::I8 => {
                lookup(t, s, d, |s| s as u8);
            }
            (InputView::U8(s), Values::F32(t), DstSliceMut::F32(d)) if dtype == DType::U8 => {
                lookup(t, s, d, |s| s);
            }
            (InputView::I8(s), Values::F16(t), DstSliceMut::F16(d)) if dtype == DType::I8 => {
                lookup(t, s, d, |s| s as u8);
            }
            (InputView::U8(s), Values::F16(t), DstSliceMut::F16(d)) if dtype == DType::U8 => {
                lookup(t, s, d, |s| s);
            }
            (_, _, dst) => return Some(dst),
        }
        None
    }
}
