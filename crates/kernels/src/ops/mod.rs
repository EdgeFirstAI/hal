// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Operator plans: validated once at creation, then run on views.

pub(crate) mod dfl;
pub(crate) mod dfl_box;
pub(crate) mod dist2bbox;
pub(crate) mod levels;
pub(crate) mod lut_i8;
mod lut_i8_neon;
mod rows;
pub(crate) mod sigmoid;
mod sigmoid_i16_neon;
pub(crate) mod softmax_i16;
mod softmax_i16_neon;
pub(crate) mod softmax_i8;
mod softmax_i8_neon;
