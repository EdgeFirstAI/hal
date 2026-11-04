// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! TFLite v2.19.0 integer helpers (and the gemmlowp primitives they use), ported bit for bit.
//!
//! Where the C++ has undefined behaviour (signed overflow, out-of-range shifts) these wrap or
//! clamp instead of panicking; callers keep their inputs out of those ranges.

/// Number of interpolation steps in a 16-bit table; the table has one more entry.
const LUT_STEPS: usize = 512;

/// `std::frexp`: `x = m * 2^e` with `0.5 <= |m| < 1`; zero and non-finite `x` give `(x, 0)`.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    // Subnormals are scaled into the normal range first.
    let (x, adjust) = if x.abs() < f64::MIN_POSITIVE {
        (x * 2f64.powi(64), -64)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (m, biased - 1022 + adjust)
}

/// `QuantizeMultiplier`: `real ≈ multiplier * 2^(shift - 31)` with `|multiplier|` in `[2^30, 2^31)`.
///
/// Zero, and magnitudes below 2^-32, give `(0, 0)`. Non-finite input (a TFLite `CHECK` failure)
/// also gives `(0, 0)`.
pub(crate) fn quantize_multiplier(real: f64) -> (i32, i32) {
    if real == 0.0 || !real.is_finite() {
        return (0, 0);
    }
    let (q, mut shift) = frexp(real);
    let mut q_fixed = (q * (1i64 << 31) as f64).round() as i64;
    if q_fixed == 1i64 << 31 {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        return (0, 0);
    }
    (i32::try_from(q_fixed).unwrap_or(i32::MAX), shift)
}

/// gemmlowp `SaturatingRoundingDoublingHighMul`: `round(a * b / 2^31)`, ties toward +inf; the
/// same computation as `vqrdmulh`.
#[inline(always)]
pub(crate) fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == b && a == i32::MIN {
        return i32::MAX;
    }
    let ab = i64::from(a) * i64::from(b);
    let nudge = if ab >= 0 {
        1i64 << 30
    } else {
        1 - (1i64 << 30)
    };
    // Truncating division, as in C++; the quotient always fits in an i32.
    ((ab + nudge) / (1i64 << 31)) as i32
}

/// gemmlowp `RoundingDivideByPOT`: `x / 2^exponent` rounded half away from zero; `exponent <= 31`.
#[inline(always)]
pub(crate) fn rounding_divide_by_pot(x: i32, exponent: u32) -> i32 {
    let e = exponent.min(31);
    let mask = ((1i64 << e) - 1) as i32;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> e) + i32::from(remainder > threshold)
}

/// `MultiplyByQuantizedMultiplier(int32_t, int32_t, int)`, the double-rounding variant.
///
/// The left shift of `x` wraps on overflow (undefined in C++), and shifts beyond ±31 clamp.
#[inline(always)]
pub(crate) fn multiply_by_quantized_multiplier(x: i32, multiplier: i32, shift: i32) -> i32 {
    let left = shift.clamp(0, 31) as u32;
    let right = shift.saturating_neg().clamp(0, 31) as u32;
    rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(x.wrapping_shl(left), multiplier),
        right,
    )
}

/// `std::max(v, lo)` then `std::min(·, hi)`, with the C++ comparison order.
fn cpp_clamp(v: f64, lo: f64, hi: f64) -> f64 {
    let v = if v < lo { lo } else { v };
    if hi < v {
        hi
    } else {
        v
    }
}

/// `LUTPopulateInt16<double>`: 512 interpolation steps of `f` over the input range, plus the end point.
///
/// Each entry is the rounded sample less half the rounded error of the chord at the step's
/// midpoint, so linear interpolation between entries tracks `f` more closely.
pub(crate) fn lut_populate_i16(
    in_scale: f64,
    in_zp: i32,
    out_scale: f64,
    out_zp: i32,
    f: impl Fn(f64) -> f64,
) -> [i16; 513] {
    let (q_min, q_max) = (i64::from(i16::MIN), i64::from(i16::MAX));
    let input_min = in_scale * (q_min - i64::from(in_zp)) as f64;
    let input_max = in_scale * (q_max - i64::from(in_zp)) as f64;
    let output_min = out_scale * (q_min - i64::from(out_zp)) as f64;
    let output_max = out_scale * (q_max - i64::from(out_zp)) as f64;

    let step = (input_max - input_min) / LUT_STEPS as f64;
    let half_step = step / 2.0;
    let output_scaling_inv = (q_max - q_min + 1) as f64 / (output_max - output_min);
    let (table_min, table_max) = (q_min as f64, q_max as f64);

    let mut lut = [0i16; LUT_STEPS + 1];
    for (i, e) in lut.iter_mut().take(LUT_STEPS).enumerate() {
        let val = f(input_min + i as f64 * step);
        let val_midpoint = f(input_min + i as f64 * step + half_step);
        let val_next = f(input_min + (i + 1) as f64 * step);

        let sample_val = (val * output_scaling_inv).round();
        let midpoint_interp_val =
            ((val_next * output_scaling_inv + (val * output_scaling_inv).round()) / 2.0).round();
        let midpoint_val = (val_midpoint * output_scaling_inv).round();
        let midpoint_err = midpoint_interp_val - midpoint_val;
        let bias = (midpoint_err / 2.0).round();

        *e = cpp_clamp(sample_val - bias, table_min, table_max) as i16;
    }
    if let Some(last) = lut.last_mut() {
        *last = cpp_clamp(
            (f(input_max) * output_scaling_inv).round(),
            table_min,
            table_max,
        ) as i16;
    }
    lut
}

/// `LUTLookup(int16_t, const int16_t*)`: the table entry for the top 9 bits of `v`, linearly
/// interpolated by the low 7 bits.
#[inline(always)]
pub(crate) fn lut_lookup_i16(v: i16, lut: &[i16; 513]) -> i16 {
    // `256 + (v >> 7)` lies in 0..=511, so both reads are in bounds.
    let index = (256 + (i32::from(v) >> 7)) as usize;
    let offset = i32::from(v & 0x7f);
    let base = lut[index];
    let slope = lut[index + 1].wrapping_sub(base);
    let delta = (i32::from(slope) * offset + 64) >> 7;
    (i32::from(base) + delta) as i16
}

/// `CheckedLog2(float)`: the rounded `log2(x)`, and whether `x` is within 1e-3 (in log2) of that power of two.
pub(crate) fn checked_log2(x: f32) -> (bool, i32) {
    let x_log2 = x.ln() * (1.0f32 / core::f32::consts::LN_2);
    let rounded = x_log2.round();
    ((x_log2 - rounded).abs() < 1e-3, rounded as i32)
}

/// The int16 branch of `SigmoidPrepare`: `(input_multiplier, input_left_shift)`.
///
/// A multiplier of 0 means the power-of-two case, which `SigmoidEval` runs through
/// [`logistic_i16_pot`]; otherwise it runs [`logistic_i16_reference`] with these values.
pub(crate) fn logistic_i16_prepare(input_scale: f32) -> (i32, i32) {
    const INPUT_INTEGER_BITS: i32 = 3;
    let (pot, log2) = checked_log2(input_scale);
    let left_shift = (15 - INPUT_INTEGER_BITS) + log2;
    if pot && left_shift == 0 {
        return (0, 0);
    }
    // Rescales the input to 1/(3·4096), the scale the reference table expects.
    let mut multiplier = f64::from(input_scale) * 4096.0 * 3.0;
    let mut left_shift = 0;
    while multiplier <= 32767.0 / 2.0 && left_shift <= 30 {
        left_shift += 1;
        multiplier *= 2.0;
    }
    (multiplier as i32, left_shift)
}

/// `sigmoid_table_uint16`: `sigmoid(x)` in Q0.16 for `x` in `[0, 255/24]` in steps of 1/24.
pub(crate) const SIGMOID_TABLE_U16: [u16; 256] = [
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498, 40149, 40794,
    41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369, 47911, 48443,
    48964, 49475, 49975, 50464, 50942, 51409, 51865, 52311, 52745, 53169, 53581, 53983, 54374,
    54755, 55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544,
    58800, 59048, 59288, 59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279,
    61441, 61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886, 62990,
    63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835, 63903, 63968, 64030,
    64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652,
    64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900, 64926, 64950, 64974, 64997, 65019,
    65039, 65060, 65079, 65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234,
    65246, 65258, 65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425, 65429, 65433,
    65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474, 65476,
    65479, 65481, 65483, 65485, 65488, 65489, 65491, 65493, 65495, 65497, 65498, 65500, 65501,
    65503, 65504, 65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516,
    65517, 65517, 65518, 65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
    65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529, 65529,
    65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532, 65532, 65532,
    65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533, 65533, 65533, 65533, 65534, 65534,
    65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535,
];

/// `reference_integer_ops::Logistic(int32_t, int32_t, ..., const int16_t*, int16_t*)` for one element.
///
/// The product of the input and the multiplier wraps, as the C++ does in practice; the plan only
/// uses multipliers up to 65535, for which it cannot overflow.
#[inline(always)]
pub(crate) fn logistic_i16_reference(input_multiplier: i32, input_left_shift: i32, x: i16) -> i16 {
    let (multiplier, shift) = if input_multiplier == 0 {
        (3i32.wrapping_shl(input_left_shift.clamp(0, 31) as u32), 0)
    } else {
        (input_multiplier, input_left_shift.clamp(0, 31))
    };
    let round = if shift > 0 { 1i32 << (shift - 1) } else { 0 };
    let input = i32::from(x).wrapping_mul(multiplier).wrapping_add(round) >> shift;
    let abs = input.unsigned_abs();
    let uh = (abs >> 9) as usize;
    let result = match (SIGMOID_TABLE_U16.get(uh), SIGMOID_TABLE_U16.get(uh + 1)) {
        (Some(&ua), Some(&ub)) if uh < 255 => {
            let (ua, ub) = (u32::from(ua), u32::from(ub));
            (ua << 9).wrapping_add((abs & 0x1ff).wrapping_mul(ub.wrapping_sub(ua)))
        }
        _ => 0x7fff << 10,
    };
    let result = if input >= 0 {
        result.wrapping_add(1 << 9)
    } else {
        (1u32 << 25).wrapping_sub(result).wrapping_add((1 << 9) - 1)
    };
    (result >> 10) as i16
}

/// gemmlowp `SaturatingRoundingDoublingHighMul<int16_t>`; the same computation as 16-bit `vqrdmulh`.
#[inline(always)]
fn srdhm_i16(a: i16, b: i16) -> i16 {
    if a == b && a == i16::MIN {
        return i16::MAX;
    }
    let ab = i32::from(a) * i32::from(b);
    let nudge = if ab >= 0 { 1 << 14 } else { 1 - (1 << 14) };
    ((ab + nudge) / (1 << 15)) as i16
}

/// gemmlowp `RoundingDivideByPOT<int16_t>`, which equals the int32 one on the widened value.
#[inline(always)]
fn rdbp_i16(x: i16, exponent: u32) -> i16 {
    rounding_divide_by_pot(i32::from(x), exponent) as i16
}

/// gemmlowp `SaturatingRoundingMultiplyByPOT<E>` on int16 for `E` in `1..=14`.
#[inline(always)]
fn saturating_mul_by_pot_i16(x: i16, exponent: u32) -> i16 {
    let e = exponent.min(14);
    let threshold = (1i32 << (15 - e)) - 1;
    let x = i32::from(x);
    if x > threshold {
        i16::MAX
    } else if x < -threshold {
        i16::MIN
    } else {
        (x << e) as i16
    }
}

/// gemmlowp `RoundingHalfSum<int16_t>`.
#[inline(always)]
fn rounding_half_sum_i16(a: i16, b: i16) -> i16 {
    let sum = i32::from(a) + i32::from(b);
    let sign = if sum >= 0 { 1 } else { -1 };
    ((sum + sign) / 2) as i16
}

/// int16 gemmlowp constants: `RescaleConstantInitializer` of the int32 raw values.
pub(crate) mod q16 {
    pub(crate) const EXP_MINUS_1_8: i16 = 28918;
    pub(crate) const ONE_THIRD: i16 = 10923;
    /// `exp(-2^k)` for k = -2..=2, applied for remainder bits 10..=14 of a Q3.12 input.
    pub(crate) const EXP_BARREL: [(u32, i16); 5] =
        [(10, 25520), (11, 19875), (12, 12055), (13, 4435), (14, 600)];
    pub(crate) const C48_OVER_17: i16 = 23130;
    pub(crate) const NEG_32_OVER_17: i16 = -15420;
    pub(crate) const ONE_HALF: i16 = 16384;
}

/// gemmlowp `exp_on_interval_between_negative_one_quarter_and_0_excl` on Q0.15.
#[inline(always)]
fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i16) -> i16 {
    let x = a.wrapping_add(1 << 12);
    let x2 = srdhm_i16(x, x);
    let x3 = srdhm_i16(x2, x);
    let x4 = srdhm_i16(x2, x2);
    let x4_over_4 = rdbp_i16(x4, 2);
    let poly = rdbp_i16(
        srdhm_i16(x4_over_4.wrapping_add(x3), q16::ONE_THIRD).wrapping_add(x2),
        1,
    );
    q16::EXP_MINUS_1_8.saturating_add(srdhm_i16(q16::EXP_MINUS_1_8, x.wrapping_add(poly)))
}

/// gemmlowp `exp_on_negative_values` from Q3.12 to Q0.15.
#[inline(always)]
fn exp_on_negative_values_q3(a: i16) -> i16 {
    const ONE_QUARTER: i16 = 1 << 10;
    let a_mod_quarter_minus_one_quarter = (a & (ONE_QUARTER - 1)).wrapping_sub(ONE_QUARTER);
    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(
        saturating_mul_by_pot_i16(a_mod_quarter_minus_one_quarter, 3),
    );
    let remainder = a_mod_quarter_minus_one_quarter.wrapping_sub(a);
    for (bit, multiplier) in q16::EXP_BARREL {
        if remainder & (1 << bit) != 0 {
            result = srdhm_i16(result, multiplier);
        }
    }
    if a == 0 {
        i16::MAX
    } else {
        result
    }
}

/// gemmlowp `one_over_one_plus_x_for_x_in_0_1` on Q0.15, by Newton-Raphson in Q2.13.
#[inline(always)]
fn one_over_one_plus_x_for_x_in_0_1(a: i16) -> i16 {
    const Q2_ONE: i16 = 1 << 13;
    let half_denominator = rounding_half_sum_i16(a, i16::MAX);
    let mut x = q16::C48_OVER_17.wrapping_add(srdhm_i16(half_denominator, q16::NEG_32_OVER_17));
    for _ in 0..3 {
        let one_minus_half_denominator_times_x =
            Q2_ONE.wrapping_sub(srdhm_i16(half_denominator, x));
        x = x.wrapping_add(saturating_mul_by_pot_i16(
            srdhm_i16(x, one_minus_half_denominator_times_x),
            2,
        ));
    }
    saturating_mul_by_pot_i16(x, 1)
}

/// `optimized_ops::Logistic(const LogisticParams&, ..., const int16_t*, int16_t*)` for one
/// element: gemmlowp `logistic` from Q3.12 to Q0.15. Negations wrap, as the int16 lanes do.
#[inline(always)]
pub(crate) fn logistic_i16_pot(x: i16) -> i16 {
    let positive = x > 0;
    let abs = if positive { x } else { x.wrapping_neg() };
    let result_if_positive =
        one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values_q3(abs.wrapping_neg()));
    if x == 0 {
        q16::ONE_HALF
    } else if positive {
        result_if_positive
    } else {
        i16::MAX.wrapping_sub(result_if_positive)
    }
}

/// gemmlowp's NEON `RoundingDivideByPOT(int32x4_t, int)`: `vqaddq` of a −1 fix-up for negative
/// `x`, then the rounding shift `vrshlq`. It equals [`rounding_divide_by_pot`] except at
/// `x = i32::MIN`, where the fix-up saturates.
#[inline(always)]
pub(crate) fn rounding_divide_by_pot_neon(x: i32, exponent: u32) -> i32 {
    let e = exponent.min(31);
    if e == 0 {
        return x;
    }
    let fixed = if x < 0 { x.saturating_sub(1) } else { x };
    ((i64::from(fixed) + (1i64 << (e - 1))) >> e) as i32
}

/// `QuantizeMultiplierSmallerThanOneExp`: `(multiplier, shift)` with `shift <= 0`, or `None`
/// for a `real` outside (0, 1), where TFLite's `CHECK` fails.
pub(crate) fn quantize_multiplier_smaller_than_one(real: f64) -> Option<(i32, i32)> {
    if !(real > 0.0 && real < 1.0) {
        return None;
    }
    let (m, shift) = quantize_multiplier(real);
    (shift <= 0).then_some((m, shift))
}

/// `Quantize(scale, zero_point, f)` of `kernel_util.cc`: `zero_point + round(f / scale)`, the
/// division and rounding in f32.
fn quantize_activation_bound(scale: f32, zero_point: i32, f: f32) -> i32 {
    zero_point.saturating_add((f / scale).round() as i32)
}

/// `CalculateActivationRangeQuantized` for an int8 output: the clamp bounds of a fused
/// activation, given the output's scale and zero point.
pub(crate) fn activation_range_i8(
    act: crate::Activation,
    scale: f32,
    zero_point: i32,
) -> (i32, i32) {
    let (qmin, qmax) = (i32::from(i8::MIN), i32::from(i8::MAX));
    let q = |f: f32| quantize_activation_bound(scale, zero_point, f);
    match act {
        crate::Activation::None => (qmin, qmax),
        crate::Activation::Relu => (qmin.max(q(0.0)), qmax),
        crate::Activation::Relu6 => (qmin.max(q(0.0)), qmax.min(q(6.0))),
        crate::Activation::ReluN1To1 => (qmin.max(q(-1.0)), qmax.min(q(1.0))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neon_rounding_divide_matches_scalar_except_at_min() {
        let xs = [
            i32::MIN + 1,
            -1_000_001,
            -96,
            -65,
            -64,
            -63,
            -1,
            0,
            1,
            63,
            64,
            65,
            96,
            1_000_001,
            i32::MAX,
        ];
        for e in 0..=31u32 {
            for x in xs {
                assert_eq!(
                    rounding_divide_by_pot_neon(x, e),
                    rounding_divide_by_pot(x, e),
                    "x={x} e={e}"
                );
            }
        }
    }

    #[test]
    fn smaller_than_one_rejects_out_of_range() {
        assert_eq!(quantize_multiplier_smaller_than_one(0.0), None);
        assert_eq!(quantize_multiplier_smaller_than_one(1.0), None);
        assert_eq!(quantize_multiplier_smaller_than_one(f64::NAN), None);
        assert_eq!(
            quantize_multiplier_smaller_than_one(0.25),
            Some((1 << 30, -1))
        );
    }

    #[test]
    fn activation_ranges_follow_tflite() {
        use crate::Activation;
        assert_eq!(activation_range_i8(Activation::None, 0.1, 3), (-128, 127));
        assert_eq!(activation_range_i8(Activation::Relu, 0.1, -10), (-10, 127));
        assert_eq!(
            activation_range_i8(Activation::Relu6, 0.05, -128),
            (-128, -8)
        );
        assert_eq!(
            activation_range_i8(Activation::ReluN1To1, 0.01, 0),
            (-100, 100)
        );
        assert_eq!(
            activation_range_i8(Activation::Relu6, 0.01, -128),
            (-128, 127)
        );
    }

    #[test]
    fn int16_constants_are_rescaled_int32_constants() {
        let c = |v: i32| rounding_divide_by_pot(v, 16) as i16;
        assert_eq!(q16::EXP_MINUS_1_8, c(1_895_147_668));
        assert_eq!(q16::ONE_THIRD, c(715_827_883));
        let raw = [
            1_672_461_947,
            1_302_514_674,
            790_015_084,
            290_630_308,
            39_332_535,
        ];
        for ((bit, m), (k, r)) in q16::EXP_BARREL.iter().zip(raw.iter().enumerate()) {
            assert_eq!(*bit, 10 + k as u32);
            assert_eq!(*m, c(*r));
        }
        // exp(-8) and exp(-16) only apply with more than 3 integer bits.
        assert_eq!(q16::C48_OVER_17, c(1_515_870_810));
        assert_eq!(q16::NEG_32_OVER_17, c(-1_010_580_540));
        assert_eq!(q16::ONE_HALF, c(1 << 30));
        assert_eq!(SIGMOID_TABLE_U16[0], 32768);
        assert_eq!(SIGMOID_TABLE_U16[255], 65535);
        assert!(SIGMOID_TABLE_U16.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn srdhm_i16_matches_vqrdmulh_model() {
        let vqrdmulh = |a: i16, b: i16| -> i16 {
            ((2 * i64::from(a) * i64::from(b) + (1 << 15)) >> 16).clamp(-32768, 32767) as i16
        };
        for a in (i16::MIN..=i16::MAX)
            .step_by(97)
            .chain([i16::MIN, -1, 0, 1, i16::MAX])
        {
            for b in (i16::MIN..=i16::MAX)
                .step_by(89)
                .chain([i16::MIN, -1, 0, 1, i16::MAX])
            {
                assert_eq!(srdhm_i16(a, b), vqrdmulh(a, b), "{a} * {b}");
            }
        }
    }

    #[test]
    fn checked_log2_and_prepare() {
        assert_eq!(checked_log2(2f32.powi(-12)), (true, -12));
        assert_eq!(checked_log2(0.25), (true, -2));
        assert!(!checked_log2(0.0005).0);
        // Within 1e-3 in log2 of 2^-12 still counts as a power of two.
        assert_eq!(checked_log2(2f32.powi(-12) * 1.0005), (true, -12));
        assert_eq!(logistic_i16_prepare(2f32.powi(-12)), (0, 0));
        // A power of two other than 2^-12 takes the multiplier path: 2^-11·12288 = 6, then 6·2^12.
        assert_eq!(logistic_i16_prepare(2f32.powi(-11)), (24576, 12));
        // 0.0005·12288 = 6.144, doubled until above 16383.5: 6.144·2^12 = 25165.824.
        assert_eq!(logistic_i16_prepare(0.0005), (25165, 12));
        // Tiny scales run out of shift with a multiplier below one, which TFLite treats as POT.
        assert_eq!(logistic_i16_prepare(1e-14).0, 0);
        assert_eq!(logistic_i16_prepare(1e-12), (26, 31));
    }

    #[test]
    fn logistic_i16_known_values() {
        // Q3.12: 0 -> 0.5, sigmoid(8) = 0.99966 -> 32757, sigmoid(1) = 0.731 -> 23955.
        assert_eq!(logistic_i16_pot(0), 16384);
        assert!((i32::from(logistic_i16_pot(i16::MAX)) - 32757).abs() <= 2);
        let one = logistic_i16_pot(4096);
        assert!((i32::from(one) - 23955).abs() <= 4, "{one}");
        for x in (i16::MIN + 1..=i16::MAX).step_by(13) {
            let s = (f64::from(x) / 4096.0).exp().recip();
            let want = 32768.0 / (1.0 + s);
            let got = f64::from(logistic_i16_pot(x));
            assert!((got - want).abs() <= 8.0, "pot {x}: {got} vs {want}");
            let reference = f64::from(logistic_i16_reference(3, 0, x));
            assert!(
                (reference - want).abs() <= 8.0,
                "ref {x}: {reference} vs {want}"
            );
        }
        assert_eq!(logistic_i16_reference(3, 0, 0), 16384);
        // Saturated: the table index reaches 255.
        assert_eq!(logistic_i16_reference(30000, 0, i16::MAX), 32767);
        assert_eq!(logistic_i16_reference(30000, 0, i16::MIN), 1);
        // A zero multiplier is the reference kernel's own POT case: 3 << shift.
        assert_eq!(
            logistic_i16_reference(0, 0, 1234),
            logistic_i16_reference(3, 0, 1234)
        );
    }

    #[test]
    fn tflite_helpers_match_known_values() {
        assert_eq!(quantize_multiplier(0.5), (1 << 30, 0));
        assert_eq!(quantize_multiplier(1.0), (1 << 30, 1));
        assert_eq!(quantize_multiplier(0.75), (3 << 29, 0));
        assert_eq!(quantize_multiplier(3.2), (1_717_986_918, 2));
        assert_eq!(quantize_multiplier(0.0), (0, 0));
        // Rounds up to 2^31, which renormalises to 2^30 with the exponent bumped.
        assert_eq!(quantize_multiplier(1.0 - 1e-12), (1 << 30, 1));
        // Below 2^-32 the multiplier flushes to zero.
        assert_eq!(quantize_multiplier(2f64.powi(-33)), (0, 0));
        assert_eq!(quantize_multiplier(2f64.powi(-32)), (1 << 30, -31));
        // Subnormal input.
        assert_eq!(quantize_multiplier(f64::MIN_POSITIVE / 4.0), (0, 0));

        assert_eq!(multiply_by_quantized_multiplier(100, 1 << 30, 0), 50);
        // -1.5 rounds toward +inf in the doubling high multiply.
        assert_eq!(multiply_by_quantized_multiplier(-3, 1 << 30, 0), -1);
        assert_eq!(multiply_by_quantized_multiplier(3, 1 << 30, 0), 2);
        // Right shift by 1 of -1 (exact -0.5 after the multiply) rounds half away from zero.
        assert_eq!(multiply_by_quantized_multiplier(-2, 1 << 30, -1), -1);
        assert_eq!(multiply_by_quantized_multiplier(-6, 1 << 30, -2), -1);
        assert_eq!(multiply_by_quantized_multiplier(-5, 1 << 30, -1), -1);
        // 2.5 rounds to 3 in the multiply, then 1.5 rounds to 2: the double rounding.
        assert_eq!(multiply_by_quantized_multiplier(5, 1 << 30, -1), 2);
        assert_eq!(multiply_by_quantized_multiplier(-7, 1 << 30, -1), -2);
        assert_eq!(multiply_by_quantized_multiplier(7, 1 << 30, -1), 2);
        assert_eq!(
            multiply_by_quantized_multiplier(-1000, 1_717_986_918, 2),
            -3200
        );
        assert_eq!(
            multiply_by_quantized_multiplier(-1001, 1_717_986_918, 2),
            -3203
        );
        assert_eq!(
            multiply_by_quantized_multiplier(i32::MIN, i32::MIN, 0),
            i32::MAX
        );
        assert_eq!(multiply_by_quantized_multiplier(i32::MIN, 1 << 30, -31), -1);

        let mut lut = [0i16; 513];
        for (i, e) in lut.iter_mut().enumerate() {
            *e = (i * 10) as i16;
        }
        // Index 256 + (v >> 7), offset v & 127, `(slope * offset + 64) >> 7`.
        assert_eq!(lut_lookup_i16(0, &lut), 2560);
        assert_eq!(lut_lookup_i16(127, &lut), 2570);
        assert_eq!(lut_lookup_i16(128, &lut), 2570);
        assert_eq!(lut_lookup_i16(6, &lut), 2560);
        assert_eq!(lut_lookup_i16(7, &lut), 2561);
        assert_eq!(lut_lookup_i16(-1, &lut), 2560);
        assert_eq!(lut_lookup_i16(-128, &lut), 2550);
        assert_eq!(lut_lookup_i16(i16::MIN, &lut), 0);
        assert_eq!(lut_lookup_i16(i16::MAX, &lut), 5120);
        lut[300] = -5;
        // Negative slope: (-2995 * 63 + 64) >> 7 = -1474, the arithmetic shift floors.
        assert_eq!(lut_lookup_i16((299 - 256) * 128 + 63, &lut), 2990 - 1474);
    }

    #[test]
    fn lut_populate_matches_tflite_softmax_tables() {
        let exp = lut_populate_i16(10.0 / 65535.0, 32767, 2.0 / 65535.0, 0, f64::exp);
        let recip = lut_populate_i16(1.0 / 65535.0, -32768, 2.0 / 65535.0, 0, |x| 1.0 / (1.0 + x));
        // exp(0) = 1 maps to 32768, clamped to the table's maximum.
        assert_eq!(exp[512], 32767);
        assert!(exp.windows(2).all(|w| w[0] <= w[1]), "exp table increases");
        assert_eq!(recip[0], 32767);
        assert_eq!(recip[512], 16384);
        assert!(
            recip.windows(2).all(|w| w[0] >= w[1]),
            "recip table decreases"
        );
        // Spot values and sums from an independent float64 port of `LUTPopulateInt16<double>`.
        let at = |t: &[i16; 513], idx: [usize; 10]| idx.map(|i| t[i]);
        let idx = [0, 1, 2, 100, 255, 256, 400, 510, 511, 512];
        assert_eq!(
            at(&exp, idx),
            [2, 2, 2, 11, 217, 221, 3677, 31512, 32133, 32767]
        );
        assert_eq!(
            at(&recip, idx),
            [32767, 32704, 32640, 27413, 21874, 21845, 18396, 16416, 16400, 16384]
        );
        let sum = |t: &[i16; 513]| t.iter().map(|&v| i64::from(v)).sum::<i64>();
        assert_eq!(sum(&exp), 1_694_030);
        assert_eq!(sum(&recip), 11_653_643);
        // The midpoint correction moves entry 0 off the rounded sample exp(-10) * 32768 = 1.49.
        assert_eq!(((-10f64).exp() * 32768.0).round(), 1.0);
    }
}
