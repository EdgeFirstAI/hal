// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON-optimised chroma upsampling.
//!
//! Placeholder: delegates to scalar. Full NEON implementation will be added
//! after scalar validation.

/// NEON horizontal 2× upsample (currently delegates to scalar).
pub fn upsample_h2_neon(input: &[u8], output: &mut [u8], width: usize) {
    super::scalar::upsample_h2(input, output, width);
}
