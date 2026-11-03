// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Single FEAT_FP16 instructions on `.8h` vectors carried as `uint16x8_t` / `int16x8_t`.
//!
//! Stable Rust has no f16 intrinsics, so each helper is one inline-asm instruction under
//! `.arch_extension fp16`. Every helper is `unsafe`: CPUs without FEAT_FP16 fault on them, so
//! callers run them only after `CpuFeatures::fp16_usable` (or `is_aarch64_feature_detected!`).

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

/// `r = a <op> b` on eight f16 lanes.
macro_rules! binary {
    ($(#[$doc:meta])* $name:ident, $insn:literal) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support FEAT_FP16.
        #[inline(always)]
        pub(crate) unsafe fn $name(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            let r: uint16x8_t;
            // SAFETY: register-only FP16 instruction; the caller guarantees FEAT_FP16.
            unsafe {
                core::arch::asm!(
                    ".arch_extension fp16",
                    concat!($insn, " {r:v}.8h, {a:v}.8h, {b:v}.8h"),
                    r = out(vreg) r,
                    a = in(vreg) a,
                    b = in(vreg) b,
                    options(pure, nomem, nostack),
                );
            }
            r
        }
    };
}

/// `acc <op>= a · b` on eight f16 lanes, fused.
macro_rules! fused {
    ($(#[$doc:meta])* $name:ident, $insn:literal) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support FEAT_FP16.
        #[inline(always)]
        pub(crate) unsafe fn $name(acc: uint16x8_t, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            let r: uint16x8_t;
            // SAFETY: register-only FP16 instruction; the caller guarantees FEAT_FP16.
            unsafe {
                core::arch::asm!(
                    ".arch_extension fp16",
                    concat!($insn, " {acc:v}.8h, {a:v}.8h, {b:v}.8h"),
                    acc = inout(vreg) acc => r,
                    a = in(vreg) a,
                    b = in(vreg) b,
                    options(pure, nomem, nostack),
                );
            }
            r
        }
    };
}

/// One-operand conversion of eight lanes from `$from` to `$to`.
macro_rules! convert {
    ($(#[$doc:meta])* $name:ident, $insn:literal, $from:ty, $to:ty) => {
        $(#[$doc])*
        ///
        /// # Safety
        /// The CPU must support FEAT_FP16.
        #[inline(always)]
        pub(crate) unsafe fn $name(x: $from) -> $to {
            let r: $to;
            // SAFETY: register-only FP16 instruction; the caller guarantees FEAT_FP16.
            unsafe {
                core::arch::asm!(
                    ".arch_extension fp16",
                    concat!($insn, " {r:v}.8h, {x:v}.8h"),
                    r = out(vreg) r,
                    x = in(vreg) x,
                    options(pure, nomem, nostack),
                );
            }
            r
        }
    };
}

binary!(
    /// `FADD`: `a + b`.
    fadd_f16x8,
    "fadd"
);
binary!(
    /// `FSUB`: `a − b`.
    fsub_f16x8,
    "fsub"
);
binary!(
    /// `FMUL`: `a · b`.
    fmul_f16x8,
    "fmul"
);
binary!(
    /// `FDIV`: `a / b`.
    fdiv_f16x8,
    "fdiv"
);
binary!(
    /// `FMAX`: the lane maximum; a NaN lane gives NaN.
    fmax_f16x8,
    "fmax"
);
binary!(
    /// `FCMGT`: all ones where `a > b`.
    fcmgt_f16x8,
    "fcmgt"
);
binary!(
    /// `FCMGE`: all ones where `a ≥ b`.
    fcmge_f16x8,
    "fcmge"
);
binary!(
    /// `FCMEQ`: all ones where `a = b`.
    fcmeq_f16x8,
    "fcmeq"
);
fused!(
    /// `FMLA`: `acc + a · b`.
    fmla_f16x8,
    "fmla"
);
fused!(
    /// `FMLS`: `acc − a · b`.
    fmls_f16x8,
    "fmls"
);
convert!(
    /// `FCVTNS`: f16 to s16, round to nearest with ties to even.
    fcvtns_s16_from_f16x8,
    "fcvtns",
    uint16x8_t,
    int16x8_t
);
convert!(
    /// `FCVTZU`: f16 to u16, round toward zero, saturating.
    fcvtzu_u16_from_f16x8,
    "fcvtzu",
    uint16x8_t,
    uint16x8_t
);
convert!(
    /// `SCVTF`: s16 to f16, round to nearest even.
    scvtf_f16_from_s16x8,
    "scvtf",
    int16x8_t,
    uint16x8_t
);
convert!(
    /// `UCVTF`: u16 to f16, round to nearest even.
    ucvtf_f16_from_u16x8,
    "ucvtf",
    uint16x8_t,
    uint16x8_t
);

/// `FMAXV` over eight f16 lanes, as bits; any NaN lane gives NaN.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline(always)]
pub(crate) unsafe fn fmaxv_f16x8(v: uint16x8_t) -> u16 {
    let r: uint16x8_t;
    // SAFETY: register-only FP16 instruction; the caller guarantees FEAT_FP16.
    unsafe {
        core::arch::asm!(
            ".arch_extension fp16",
            "fmaxv {r:h}, {v:v}.8h",
            r = out(vreg) r,
            v = in(vreg) v,
            options(pure, nomem, nostack),
        );
        vgetq_lane_u16::<0>(r)
    }
}
