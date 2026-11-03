// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Targets without NEON: no plan is ever built, so every mask call runs the scalar code.
#![allow(clippy::extra_unused_type_parameters)]

use std::marker::PhantomData;

use ndarray::{Array3, ArrayView1, ArrayView3};

use super::{Crop, Isa};
#[cfg(test)]
use super::{FloatKernel, QuantKernel};
use crate::{DecoderError, Quantization};

fn unavailable() -> DecoderError {
    DecoderError::Internal("no vector mask kernels on this target".into())
}

/// Never built off aarch64.
pub(crate) struct QuantPlan<'a>(PhantomData<&'a ()>);

impl<'a> QuantPlan<'a> {
    pub(crate) fn new<MASK: 'static, PROTO: 'static>(
        _: Isa,
        _: &'a ArrayView3<'_, PROTO>,
        _: Quantization,
        _: Quantization,
    ) -> Option<Self> {
        None
    }

    #[cfg(test)]
    pub(crate) fn kernel<MASK: 'static>(&self) -> QuantKernel {
        QuantKernel::Scalar
    }

    pub(crate) fn mask<MASK>(
        &self,
        _: ArrayView1<MASK>,
        _: Crop,
    ) -> Result<Array3<u8>, DecoderError> {
        Err(unavailable())
    }
}

/// Never built off aarch64.
pub(crate) struct FloatPlan<'a>(PhantomData<&'a ()>);

impl<'a> FloatPlan<'a> {
    pub(crate) fn new<PROTO: 'static>(_: Isa, _: &'a ArrayView3<'_, PROTO>) -> Option<Self> {
        None
    }

    #[cfg(test)]
    pub(crate) fn kernel(&self) -> FloatKernel {
        FloatKernel::Scalar
    }

    pub(crate) fn mask<MASK>(
        &self,
        _: ArrayView1<MASK>,
        _: Crop,
    ) -> Result<Array3<u8>, DecoderError> {
        Err(unavailable())
    }
}
