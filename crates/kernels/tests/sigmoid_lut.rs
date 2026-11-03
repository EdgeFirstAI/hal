// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

mod common;

use common::Fixture;
use edgefirst_kernels::prim::dequant::dequant_i8_to_f32;
use edgefirst_kernels::prim::sigmoid::sigmoid_slice_f32;
use edgefirst_kernels::{
    CpuFeatures, DType, KernelError, LutI8Plan, Pool, Quant, SigmoidParams, SigmoidPlan, TensorMut,
    TensorRef, Tier, View, ViewMut,
};
use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};

fn as_i8(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|&b| b as i8).collect()
}

fn as_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn params(f: &Fixture, dtype: DType) -> SigmoidParams {
    SigmoidParams {
        input_dtype: dtype,
        input: f.input_quant,
        output_dtype: dtype,
        output: f.output_quant,
    }
}

/// Runs `plan` on contiguous views of `shape` and returns the output.
fn run_i8(plan: &SigmoidPlan, input: &[i8], shape: &[usize], pool: Option<&dyn Pool>) -> Vec<i8> {
    let mut out = vec![0x55u8 as i8; input.len()];
    plan.run(
        TensorRef::I8(View::contiguous(input, shape).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, shape).unwrap()),
        pool,
    )
    .unwrap();
    out
}

fn run_lut<T: Copy + Default>(
    plan: &LutI8Plan,
    input: &[T],
    wrap: impl Fn(&[T]) -> TensorRef<'_>,
    wrap_mut: impl Fn(&mut [T]) -> TensorMut<'_>,
) -> Vec<T> {
    let mut out = vec![T::default(); input.len()];
    plan.run(wrap(input), wrap_mut(&mut out), None).unwrap();
    out
}

fn mismatches<T: PartialEq>(got: &[T], want: &[T]) -> usize {
    got.iter().zip(want).filter(|(g, w)| g != w).count()
}

#[test]
fn plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LutI8Plan>();
    assert_send_sync::<SigmoidPlan>();
    assert_send_sync::<SigmoidParams>();
}

#[test]
fn lut_sigmoid_i8_u8_bit_exact_vs_tflite() {
    let fi8 = common::load("logistic_i8_scores");
    let fu8 = common::load("logistic_u8_scores");
    let (in_i8, want_i8) = (as_i8(&fi8.input), as_i8(&fi8.output));
    for feats in common::tiers() {
        let neon = feats.neon_baseline;
        let expect_tier = if neon { Tier::Neon } else { Tier::Scalar };

        let plan = SigmoidPlan::with_features(params(&fi8, DType::I8), feats).unwrap();
        assert_eq!(plan.tier(), expect_tier);
        let got = run_i8(&plan, &in_i8, &fi8.shape, None);
        let bad = mismatches(&got, &want_i8);
        assert_eq!(
            bad,
            0,
            "i8 tier {:?}: {bad} of {} differ",
            plan.tier(),
            got.len()
        );

        let lut =
            LutI8Plan::sigmoid_with_features(DType::I8, fi8.input_quant, fi8.output_quant, feats)
                .unwrap();
        assert_eq!(lut.tier(), expect_tier);
        let got = run_lut(
            &lut,
            &in_i8,
            |s| TensorRef::I8(View::contiguous(s, &fi8.shape).unwrap()),
            |s| TensorMut::I8(ViewMut::contiguous(s, &fi8.shape).unwrap()),
        );
        assert_eq!(
            mismatches(&got, &want_i8),
            0,
            "LutI8Plan i8 tier {:?}",
            lut.tier()
        );

        let plan = SigmoidPlan::with_features(params(&fu8, DType::U8), feats).unwrap();
        assert_eq!(plan.tier(), expect_tier);
        let mut got = vec![0x55u8; fu8.input.len()];
        plan.run(
            TensorRef::U8(View::contiguous(&fu8.input, &fu8.shape).unwrap()),
            TensorMut::U8(ViewMut::contiguous(&mut got, &fu8.shape).unwrap()),
            None,
        )
        .unwrap();
        let bad = mismatches(&got, &fu8.output);
        assert_eq!(
            bad,
            0,
            "u8 tier {:?}: {bad} of {} differ",
            plan.tier(),
            got.len()
        );

        let lut =
            LutI8Plan::sigmoid_with_features(DType::U8, fu8.input_quant, fu8.output_quant, feats)
                .unwrap();
        let got = run_lut(
            &lut,
            &fu8.input,
            |s| TensorRef::U8(View::contiguous(s, &fu8.shape).unwrap()),
            |s| TensorMut::U8(ViewMut::contiguous(s, &fu8.shape).unwrap()),
        );
        assert_eq!(mismatches(&got, &fu8.output), 0, "LutI8Plan u8");
    }
}

#[test]
fn sigmoid_i16_bit_exact_vs_tflite() {
    for name in ["logistic_i16_nonpot", "logistic_i16_pot"] {
        let f = common::load(name);
        let (input, want) = (as_i16(&f.input), as_i16(&f.output));
        for feats in common::tiers() {
            let plan = SigmoidPlan::with_features(params(&f, DType::I16), feats).unwrap();
            let expect_tier = if feats.neon_baseline {
                Tier::Neon
            } else {
                Tier::Scalar
            };
            assert_eq!(plan.tier(), expect_tier, "{name}");
            let mut got = vec![0x5555i16; input.len()];
            plan.run(
                TensorRef::I16(View::contiguous(&input, &f.shape).unwrap()),
                TensorMut::I16(ViewMut::contiguous(&mut got, &f.shape).unwrap()),
                None,
            )
            .unwrap();
            let bad = mismatches(&got, &want);
            assert_eq!(
                bad,
                0,
                "{name} tier {:?}: {bad} of {} differ",
                plan.tier(),
                want.len()
            );
        }
    }
}

/// Float64 sigmoid of the dequantized input, quantized with round-half-away and clamped.
fn f64_reference(x: f64, in_q: Quant, out_q: Quant, lo: i64, hi: i64) -> i64 {
    let v = f64::from(in_q.scale) * (x - f64::from(in_q.zero_point));
    let y = 1.0 / (1.0 + (-v).exp());
    let q = (y / f64::from(out_q.scale)).round() as i64 + i64::from(out_q.zero_point);
    q.clamp(lo, hi)
}

#[test]
fn sigmoid_other_quant_within_one_lsb_of_f64() {
    let all_i16: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    let all_u16: Vec<u16> = (0..=u16::MAX).collect();
    let all_i8: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    for feats in common::tiers() {
        let check = |got: &[i64], input: &[f64], in_q: Quant, out_q: Quant, lo, hi, what: &str| {
            let mut off = 0usize;
            for (i, (&g, &x)) in got.iter().zip(input).enumerate() {
                let w = f64_reference(x, in_q, out_q, lo, hi);
                assert!((g - w).abs() <= 1, "{what}: element {i}: got {g} want {w}");
                off += usize::from(g != w);
            }
            eprintln!("sigmoid {what}: {off} of {} off by one", got.len());
        };

        // i16 -> i16 with a non-zero output zero point leaves the TFLite path.
        let (in_q, out_q) = (Quant::new(0.0005, 0), Quant::new(1.0 / 32768.0, 5));
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I16,
                input: in_q,
                output_dtype: DType::I16,
                output: out_q,
            },
            feats,
        )
        .unwrap();
        let mut out = vec![0i16; all_i16.len()];
        plan.run(
            TensorRef::I16(View::contiguous(&all_i16, &[all_i16.len()]).unwrap()),
            TensorMut::I16(ViewMut::contiguous(&mut out, &[all_i16.len()]).unwrap()),
            None,
        )
        .unwrap();
        let got: Vec<i64> = out.iter().map(|&v| i64::from(v)).collect();
        let xs: Vec<f64> = all_i16.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, in_q, out_q, -32768, 32767, "i16->i16 zp 5");

        let (in_q, out_q) = (Quant::new(0.0003, 32768), Quant::new(1.0 / 65535.0, 0));
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::U16,
                input: in_q,
                output_dtype: DType::U16,
                output: out_q,
            },
            feats,
        )
        .unwrap();
        let mut out = vec![0u16; all_u16.len()];
        plan.run(
            TensorRef::U16(View::contiguous(&all_u16, &[256, 256]).unwrap()),
            TensorMut::U16(ViewMut::contiguous(&mut out, &[256, 256]).unwrap()),
            None,
        )
        .unwrap();
        let got: Vec<i64> = out.iter().map(|&v| i64::from(v)).collect();
        let xs: Vec<f64> = all_u16.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, in_q, out_q, 0, 65535, "u16->u16");

        let (in_q, out_q) = (Quant::new(0.1, -30), Quant::new(1.0 / 32768.0, 0));
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: in_q,
                output_dtype: DType::I16,
                output: out_q,
            },
            feats,
        )
        .unwrap();
        let mut out = vec![0i16; all_i8.len()];
        plan.run(
            TensorRef::I8(View::contiguous(&all_i8, &[all_i8.len()]).unwrap()),
            TensorMut::I16(ViewMut::contiguous(&mut out, &[all_i8.len()]).unwrap()),
            None,
        )
        .unwrap();
        let got: Vec<i64> = out.iter().map(|&v| i64::from(v)).collect();
        let xs: Vec<f64> = all_i8.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, in_q, out_q, -32768, 32767, "i8->i16");

        // Float input to a quantized output, and an 8-bit pair of differing types.
        let floats: Vec<f32> = (0..4001).map(|i| (i as f32 - 2000.0) * 0.005).collect();
        let (in_q, out_q) = (Quant::IDENTITY, Quant::new(1.0 / 255.0, 0));
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::F32,
                input: in_q,
                output_dtype: DType::U8,
                output: out_q,
            },
            feats,
        )
        .unwrap();
        let mut out = vec![0u8; floats.len()];
        plan.run(
            TensorRef::F32(View::contiguous(&floats, &[floats.len()]).unwrap()),
            TensorMut::U8(ViewMut::contiguous(&mut out, &[floats.len()]).unwrap()),
            None,
        )
        .unwrap();
        let got: Vec<i64> = out.iter().map(|&v| i64::from(v)).collect();
        let xs: Vec<f64> = floats.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, in_q, out_q, 0, 255, "f32->u8");

        let (in_q, out_q) = (Quant::new(0.1, -30), Quant::new(1.0 / 256.0, 0));
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: in_q,
                output_dtype: DType::U8,
                output: out_q,
            },
            feats,
        )
        .unwrap();
        let mut out = vec![0u8; all_i8.len()];
        plan.run(
            TensorRef::I8(View::contiguous(&all_i8, &[all_i8.len()]).unwrap()),
            TensorMut::U8(ViewMut::contiguous(&mut out, &[all_i8.len()]).unwrap()),
            None,
        )
        .unwrap();
        let got: Vec<i64> = out.iter().map(|&v| i64::from(v)).collect();
        let xs: Vec<f64> = all_i8.iter().map(|&v| f64::from(v)).collect();
        check(&got, &xs, in_q, out_q, 0, 255, "i8->u8");
    }
}

/// What the decoder's score path computes for i8 -> f32 on `feats`, over the whole buffer.
fn decoder_i8_to_f32(input: &[i8], q: Quant, feats: CpuFeatures) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    #[cfg(target_arch = "aarch64")]
    if feats.neon_baseline {
        // SAFETY: NEON is part of the aarch64 baseline; lengths are equal.
        unsafe {
            edgefirst_kernels::prim::neon_baseline::dequant_sigmoid_i8_to_f32_neon(
                input, q, &mut out,
            )
        };
        return out;
    }
    let _ = feats;
    dequant_i8_to_f32(input, q, &mut out);
    sigmoid_slice_f32(&mut out);
    out
}

#[test]
fn sigmoid_float_outputs_match_existing_kernels() {
    // Not a multiple of the 16-lane vector width, and longer than one chunk.
    let n = 3 * 1024 + 13;
    let in_i8: Vec<i8> = (0..n)
        .map(|i| ((i * 37 + i / 7) % 256) as u8 as i8)
        .collect();
    let q8 = Quant::new(0.07, -12);
    for feats in common::tiers() {
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: q8,
                output_dtype: DType::F32,
                output: Quant::IDENTITY,
            },
            feats,
        )
        .unwrap();
        let want = decoder_i8_to_f32(&in_i8, q8, feats);
        let expect_tier = if cfg!(target_arch = "aarch64") && feats.neon_baseline {
            Tier::Neon
        } else {
            Tier::Scalar
        };
        assert_eq!(plan.tier(), expect_tier);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let three = edgefirst_kernels::ScopedThreads::new(std::num::NonZeroUsize::new(3).unwrap());
        let pools: [Option<&dyn Pool>; 3] = [None, Some(&three), Some(&common::SequentialPool)];
        for pool in pools {
            let mut got = vec![0.0f32; n];
            plan.run(
                TensorRef::I8(View::contiguous(&in_i8, &[n]).unwrap()),
                TensorMut::F32(ViewMut::contiguous(&mut got, &[n]).unwrap()),
                pool,
            )
            .unwrap();
            assert_eq!(bits(&got), bits(&want), "i8->f32 tier {:?}", plan.tier());
        }
        // A strided view of the same elements gives the same bits.
        let mut transposed = vec![0i8; n];
        let (r, c) = (617, 5);
        assert_eq!(r * c, n);
        for i in 0..r {
            for j in 0..c {
                transposed[j * r + i] = in_i8[i * c + j];
            }
        }
        let mut got = vec![0.0f32; n];
        plan.run(
            TensorRef::I8(View::strided(&transposed, &[r, c], &[1, r as isize]).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut got, &[r, c]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(bits(&got), bits(&want), "strided i8->f32");
    }
}

/// One input tensor of every dtype, with the quantization the tests use.
struct Inputs {
    i8: Vec<i8>,
    u8: Vec<u8>,
    i16: Vec<i16>,
    u16: Vec<u16>,
    f16: Vec<f16>,
    f32: Vec<f32>,
}

impl Inputs {
    /// Every 8- and 16-bit value (8-bit ones repeated past one chunk), every f16 bit pattern,
    /// and an f32 sweep with specials.
    fn exhaustive() -> Self {
        let mut f32s: Vec<f32> = (0..80_001).map(|i| (i as f32 - 40_000.0) * 5e-4).collect();
        f32s.extend([
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            1e30,
            -1e30,
            1e-30,
        ]);
        Self {
            i8: (0..3 * 1024 + 13).map(|i| (i % 256) as u8 as i8).collect(),
            u8: (0..3 * 1024 + 13).map(|i| (i % 256) as u8).collect(),
            i16: (i16::MIN..=i16::MAX).collect(),
            u16: (0..=u16::MAX).collect(),
            f16: (0..=u16::MAX).map(f16::from_bits).collect(),
            f32: f32s,
        }
    }

    fn quant(d: DType) -> Quant {
        match d {
            DType::I8 => Quant::new(0.07, -12),
            DType::U8 => Quant::new(0.15, 30),
            DType::I16 => Quant::new(0.0005, 7),
            DType::U16 => Quant::new(0.0004, 30000),
            _ => Quant::IDENTITY,
        }
    }

    fn input(&self, d: DType) -> TensorRef<'_> {
        self.input_without(d, 0)
    }

    /// The input of `d` without its last `cut` elements.
    fn input_without(&self, d: DType, cut: usize) -> TensorRef<'_> {
        fn view<T>(v: &[T], cut: usize) -> View<'_, T> {
            let v = &v[..v.len() - cut];
            View::contiguous(v, &[v.len()]).unwrap()
        }
        match d {
            DType::I8 => TensorRef::I8(view(&self.i8, cut)),
            DType::U8 => TensorRef::U8(view(&self.u8, cut)),
            DType::I16 => TensorRef::I16(view(&self.i16, cut)),
            DType::U16 => TensorRef::U16(view(&self.u16, cut)),
            DType::F16 => TensorRef::F16(view(&self.f16, cut)),
            DType::F32 => TensorRef::F32(view(&self.f32, cut)),
        }
    }

    /// The dequantized inputs in float64, exactly.
    fn exact(&self, d: DType) -> Vec<f64> {
        let q = Self::quant(d);
        let deq = |v: f64| (v - f64::from(q.zero_point)) * f64::from(q.scale);
        match d {
            DType::I8 => self.i8.iter().map(|&v| deq(f64::from(v))).collect(),
            DType::U8 => self.u8.iter().map(|&v| deq(f64::from(v))).collect(),
            DType::I16 => self.i16.iter().map(|&v| deq(f64::from(v))).collect(),
            DType::U16 => self.u16.iter().map(|&v| deq(f64::from(v))).collect(),
            DType::F16 => self.f16.iter().map(|v| v.to_f64()).collect(),
            DType::F32 => self.f32.iter().map(|&v| f64::from(v)).collect(),
        }
    }
}

const ALL_DTYPES: [DType; 6] = [
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::F16,
    DType::F32,
];

fn float_plan(input: DType, output: DType, feats: CpuFeatures) -> SigmoidPlan {
    SigmoidPlan::with_features(
        SigmoidParams {
            input_dtype: input,
            input: Inputs::quant(input),
            output_dtype: output,
            output: Quant::IDENTITY,
        },
        feats,
    )
    .unwrap()
}

fn run_to_f16(plan: &SigmoidPlan, input: TensorRef<'_>) -> Vec<f16> {
    let n = input.shape().iter().product();
    let mut out = vec![f16::from_bits(0x5555); n];
    plan.run(
        input,
        TensorMut::F16(ViewMut::contiguous(&mut out, &[n]).unwrap()),
        None,
    )
    .unwrap();
    out
}

fn run_to_f32(plan: &SigmoidPlan, input: TensorRef<'_>) -> Vec<f32> {
    let n = input.shape().iter().product();
    let mut out = vec![0.0f32; n];
    plan.run(
        input,
        TensorMut::F32(ViewMut::contiguous(&mut out, &[n]).unwrap()),
        None,
    )
    .unwrap();
    out
}

#[test]
fn sigmoid_f16_output_is_narrowed_f32_path() {
    let inputs = Inputs::exhaustive();
    let mut checked = 0;
    for feats in common::tiers() {
        for d in ALL_DTYPES {
            let to_f16 = float_plan(d, DType::F16, feats);
            if to_f16.tier() == Tier::NeonFp16 {
                continue;
            }
            let to_f32 = float_plan(d, DType::F32, feats);
            assert_eq!(to_f16.tier(), to_f32.tier(), "{d:?}");
            // Every tail length a 16-element step can leave.
            for cut in 0..16 {
                let want: Vec<u16> = run_to_f32(&to_f32, inputs.input_without(d, cut))
                    .iter()
                    .map(|&v| f16::from_f32(v).to_bits())
                    .collect();
                let got: Vec<u16> = run_to_f16(&to_f16, inputs.input_without(d, cut))
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(got, want, "{d:?} -> f16 tier {:?} cut {cut}", to_f16.tier());
            }
            checked += 1;
        }
    }
    assert!(checked >= ALL_DTYPES.len());
}

/// f16 ulps the FP16 tier's f16 outputs may be from the float64 sigmoid rounded to f16.
const FP16_ULP_BOUND: i32 = 2;

/// On the FP16 tier, every input type's f16 output is within 2 f16 ulps of the float64
/// sigmoid of the exactly dequantized input, rounded to f16, at the fixed quantizations and a
/// seeded random sweep; NaN gives NaN.
#[test]
fn sigmoid_f16_output_fp16_within_bound() {
    let Some(feats) = common::tiers().into_iter().find(|f| f.neon_fp16) else {
        eprintln!("WARNING: SKIPPED: sigmoid_f16_output_fp16_within_bound: no neon_fp16 tier");
        return;
    };
    let inputs = Inputs::exhaustive();
    for d in ALL_DTYPES {
        let plan = float_plan(d, DType::F16, feats);
        assert_eq!(plan.tier(), Tier::NeonFp16, "{d:?}");
        let got = run_to_f16(&plan, inputs.input(d));
        assert_within_fp16_bound(&got, &inputs.exact(d), &format!("{d:?}"));
    }
    // A seeded sweep of random scales and zero points over every value of each integer type.
    let mut rng = StdRng::seed_from_u64(0x5167_f016);
    let all_i8: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    let all_u8: Vec<u8> = (0..=u8::MAX).collect();
    for (d, sweeps, log2_scale, zp_range) in [
        (DType::I8, 400, -14.0..1.0, -128..=127),
        (DType::U8, 400, -14.0..1.0, 0..=255),
        (DType::I16, 200, -18.0..-5.0, -32768..=32767),
        (DType::U16, 200, -18.0..-5.0, 0..=65535),
    ] {
        for _ in 0..sweeps {
            let q = Quant::new(
                2f32.powf(rng.random_range(log2_scale.clone())),
                rng.random_range(zp_range.clone()),
            );
            let plan = SigmoidPlan::with_features(
                SigmoidParams {
                    input_dtype: d,
                    input: q,
                    output_dtype: DType::F16,
                    output: Quant::IDENTITY,
                },
                feats,
            )
            .unwrap();
            let shape = |n: usize| [n];
            let (input, values): (TensorRef<'_>, Vec<f64>) = match d {
                DType::I8 => (
                    TensorRef::I8(View::contiguous(&all_i8, &shape(256)).unwrap()),
                    all_i8.iter().map(|&v| f64::from(v)).collect(),
                ),
                DType::U8 => (
                    TensorRef::U8(View::contiguous(&all_u8, &shape(256)).unwrap()),
                    all_u8.iter().map(|&v| f64::from(v)).collect(),
                ),
                DType::I16 => (
                    inputs.input(DType::I16),
                    inputs.i16.iter().map(|&v| f64::from(v)).collect(),
                ),
                _ => (
                    inputs.input(DType::U16),
                    inputs.u16.iter().map(|&v| f64::from(v)).collect(),
                ),
            };
            let xs: Vec<f64> = values
                .iter()
                .map(|&v| (v - f64::from(q.zero_point)) * f64::from(q.scale))
                .collect();
            assert_within_fp16_bound(&run_to_f16(&plan, input), &xs, &format!("{d:?} {q:?}"));
        }
    }
}

/// Asserts every `got` is within `FP16_ULP_BOUND` of the f64 sigmoid of `xs`; NaN gives NaN.
fn assert_within_fp16_bound(got: &[f16], xs: &[f64], what: &str) {
    for (&g, &x) in got.iter().zip(xs) {
        if x.is_nan() {
            assert!(g.is_nan(), "{what}: NaN gave {g}");
            continue;
        }
        let want = f16::from_f64(1.0 / (1.0 + (-x).exp()));
        let ulps = (i32::from(g.to_bits()) - i32::from(want.to_bits())).abs();
        assert!(
            ulps <= FP16_ULP_BOUND,
            "{what}: x {x}: got {g}, want {want} ({ulps} ulp)"
        );
    }
}

/// f32 outputs keep the f32 exp on the FP16 tier: the same kernels, and bits, as `neon`.
#[test]
fn sigmoid_f32_output_on_fp16_tier_is_the_neon_kernel() {
    let Some(fp16) = common::tiers().into_iter().find(|f| f.neon_fp16) else {
        eprintln!("WARNING: SKIPPED: sigmoid_f32_output_on_fp16_tier_is_the_neon_kernel: no neon_fp16 tier");
        return;
    };
    let neon = CpuFeatures::from_forced_tier(Some("neon")).unwrap();
    let inputs = Inputs::exhaustive();
    for d in ALL_DTYPES {
        let plan = float_plan(d, DType::F32, fp16);
        assert_eq!(plan.tier(), Tier::Neon, "{d:?}");
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let got = bits(run_to_f32(&plan, inputs.input(d)));
        let want = bits(run_to_f32(
            &float_plan(d, DType::F32, neon),
            inputs.input(d),
        ));
        assert_eq!(got, want, "{d:?}");
    }
}

#[test]
fn from_table_rejects_non_byte_dtypes() {
    let table: [u8; 256] = core::array::from_fn(|i| 255 - i as u8);
    for d in [DType::I16, DType::U16, DType::F16, DType::F32] {
        assert!(
            matches!(
                LutI8Plan::from_table(d, table),
                Err(KernelError::InvalidArgument(_))
            ),
            "{d:?}"
        );
    }
    let plan = LutI8Plan::from_table(DType::I8, table).unwrap();
    let input: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    let mut out = vec![0i8; 256];
    plan.run(
        TensorRef::I8(View::contiguous(&input, &[256]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[256]).unwrap()),
        None,
    )
    .unwrap();
    for (&x, &y) in input.iter().zip(&out) {
        assert_eq!(y as u8, 255 - x as u8, "raw byte {x}");
    }
}

#[test]
fn from_table_rejects_i16() {
    assert!(matches!(
        LutI8Plan::from_table(DType::I16, [0; 256]),
        Err(KernelError::InvalidArgument(_))
    ));
}

#[test]
fn new_rejects_bad_params() {
    let good = SigmoidParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, -30),
        output_dtype: DType::I8,
        output: Quant::new(1.0 / 256.0, -128),
    };
    assert!(SigmoidPlan::new(good).is_ok());
    let invalid =
        |p: SigmoidParams| matches!(SigmoidPlan::new(p), Err(KernelError::InvalidArgument(_)));
    for s in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            invalid(SigmoidParams {
                input: Quant::new(s, 0),
                ..good
            }),
            "input scale {s}"
        );
        assert!(
            invalid(SigmoidParams {
                output: Quant::new(s, 0),
                ..good
            }),
            "output scale {s}"
        );
        for d in [DType::I8, DType::U8] {
            assert!(matches!(
                LutI8Plan::sigmoid(d, Quant::new(s, 0), good.output),
                Err(KernelError::InvalidArgument(_))
            ));
            assert!(matches!(
                LutI8Plan::sigmoid(d, good.input, Quant::new(s, 0)),
                Err(KernelError::InvalidArgument(_))
            ));
        }
    }
    // Zero points outside the integer type's range.
    assert!(invalid(SigmoidParams {
        input: Quant::new(0.1, 128),
        ..good
    }));
    assert!(invalid(SigmoidParams {
        output: Quant::new(0.1, -129),
        ..good
    }));
    assert!(invalid(SigmoidParams {
        output_dtype: DType::U16,
        output: Quant::new(0.1, 65536),
        ..good
    }));
    assert!(matches!(
        LutI8Plan::sigmoid(DType::U8, Quant::new(0.1, -1), good.output),
        Err(KernelError::InvalidArgument(_))
    ));
    assert!(matches!(
        LutI8Plan::sigmoid(DType::I16, good.input, good.output),
        Err(KernelError::InvalidArgument(_))
    ));
    // Float types carry no zero point range.
    assert!(SigmoidPlan::new(SigmoidParams {
        input_dtype: DType::F32,
        input: Quant::new(1.0, 1000),
        output_dtype: DType::F16,
        output: Quant::IDENTITY,
    })
    .is_ok());
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let lut_plan = SigmoidPlan::new(SigmoidParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::I8,
        output: Quant::new(1.0 / 256.0, -128),
    })
    .unwrap();
    let float_plan = SigmoidPlan::new(SigmoidParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
    })
    .unwrap();
    let input = vec![1i8; 800];
    let wide = vec![1i16; 800];
    let bytes = vec![1u8; 800];

    for (plan, float_out) in [(&lut_plan, false), (&float_plan, true)] {
        let mut out = vec![0x55i8; 800];
        let mut f_out = vec![7.0f32; 800];
        // Wrong input dtype.
        let r = plan.run(
            TensorRef::I16(View::contiguous(&wide, &[2, 400]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 400]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        let r = plan.run(
            TensorRef::U8(View::contiguous(&bytes, &[2, 400]).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut f_out, &[2, 400]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        // Shape mismatch, in either output dtype.
        let r = plan.run(
            TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out[..798], &[2, 399]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        let r = plan.run(
            TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut f_out, &[400, 2]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        // Overlapping output.
        let r = plan.run(
            TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut out, &[2, 400], &[0, 1]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        let r = plan.run(
            TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
            TensorMut::F32(ViewMut::strided(&mut f_out, &[2, 400], &[400, 0]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        // The other output dtype from the plan's.
        let wrong_out = if !float_out {
            plan.run(
                TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
                TensorMut::F32(ViewMut::contiguous(&mut f_out, &[2, 400]).unwrap()),
                None,
            )
        } else {
            plan.run(
                TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 400]).unwrap()),
                None,
            )
        };
        assert!(
            matches!(wrong_out, Err(KernelError::InvalidArgument(_))),
            "{wrong_out:?}"
        );
        assert!(out.iter().all(|&b| b == 0x55));
        assert!(f_out.iter().all(|&v| v == 7.0));
    }

    let lut = LutI8Plan::from_table(DType::U8, [9; 256]).unwrap();
    let mut out = vec![0x55i8; 800];
    let r = lut.run(
        TensorRef::I8(View::contiguous(&input, &[800]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[800]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    let mut out_u8 = vec![0x55u8; 800];
    let r = lut.run(
        TensorRef::U8(View::contiguous(&bytes, &[800]).unwrap()),
        TensorMut::U8(ViewMut::contiguous(&mut out_u8[..400], &[400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x55));
    assert!(out_u8.iter().all(|&b| b == 0x55));
}

#[test]
fn zero_sized_and_rank_zero_views() {
    let plan = SigmoidPlan::new(SigmoidParams {
        input_dtype: DType::I16,
        input: Quant::new(2f32.powi(-12), 0),
        output_dtype: DType::I16,
        output: Quant::new(1.0 / 32768.0, 0),
    })
    .unwrap();
    let empty: [i16; 0] = [];
    let mut empty_out: [i16; 0] = [];
    plan.run(
        TensorRef::I16(View::contiguous(&empty, &[3, 0, 4]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut empty_out, &[3, 0, 4]).unwrap()),
        None,
    )
    .unwrap();
    let scalar = [0i16];
    let mut scalar_out = [0x5555i16];
    plan.run(
        TensorRef::I16(View::contiguous(&scalar, &[]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut scalar_out, &[]).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(scalar_out, [16384]);
}

/// The fixture input as `[rows, cols]` transposed in memory, with its strides.
fn transpose<T: Copy + Default>(input: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::default(); input.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = input[i * cols + j];
        }
    }
    t
}

#[test]
fn strided_views_match_contiguous() {
    let f = common::load("logistic_i8_scores");
    let input = as_i8(&f.input);
    let (rows, cols) = (80, 8400);
    assert_eq!(f.shape, [1, rows, cols]);
    let t = transpose(&input, rows, cols);
    let ts = [1isize, rows as isize];
    for feats in common::tiers() {
        let lut = LutI8Plan::sigmoid_with_features(DType::I8, f.input_quant, f.output_quant, feats)
            .unwrap();
        let mut want = vec![0i8; input.len()];
        lut.run(
            TensorRef::I8(View::contiguous(&input, &[rows, cols]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut want, &[rows, cols]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(want, as_i8(&f.output));

        // Strided input, contiguous output.
        let mut got = vec![0i8; input.len()];
        lut.run(
            TensorRef::I8(View::strided(&t, &[rows, cols], &ts).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut got, &[rows, cols]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "strided input, tier {:?}", lut.tier());

        // Strided input and output: the output is the transposed result.
        let mut got_t = vec![0x55i8; input.len()];
        lut.run(
            TensorRef::I8(View::strided(&t, &[rows, cols], &ts).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut got_t, &[rows, cols], &ts).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got_t, transpose(&want, rows, cols), "transposed output");

        // Padded rows on a small slice: padding is left untouched.
        let (r, c, pitch) = (5usize, 17usize, 20usize);
        let mut padded = vec![0x55i8; r * pitch];
        lut.run(
            TensorRef::I8(View::contiguous(&input[..r * c], &[r, c]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut padded, &[r, c], &[pitch as isize, 1]).unwrap()),
            None,
        )
        .unwrap();
        for i in 0..r {
            assert_eq!(&padded[i * pitch..i * pitch + c], &want[i * c..(i + 1) * c]);
            assert!(padded[i * pitch + c..(i + 1) * pitch]
                .iter()
                .all(|&b| b == 0x55));
        }
    }

    // The int16 path through strided views.
    let f = common::load("logistic_i16_nonpot");
    let input = as_i16(&f.input);
    let t = transpose(&input, rows, cols);
    let plan = SigmoidPlan::new(params(&f, DType::I16)).unwrap();
    let mut got_t = vec![0i16; input.len()];
    plan.run(
        TensorRef::I16(View::strided(&t, &[rows, cols], &ts).unwrap()),
        TensorMut::I16(ViewMut::strided(&mut got_t, &[rows, cols], &ts).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(got_t, transpose(&as_i16(&f.output), rows, cols));
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let fi8 = common::load("logistic_i8_scores");
    let in_i8 = as_i8(&fi8.input);
    let f16_ = common::load("logistic_i16_pot");
    let in_i16 = as_i16(&f16_.input);
    for feats in common::tiers() {
        let plan = SigmoidPlan::with_features(params(&fi8, DType::I8), feats).unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            run_i8(&plan, &in_i8, &fi8.shape, pool)
                .into_iter()
                .map(|v| v as u8)
                .collect()
        });

        let plan = SigmoidPlan::with_features(params(&f16_, DType::I16), feats).unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0i16; in_i16.len()];
            plan.run(
                TensorRef::I16(View::contiguous(&in_i16, &f16_.shape).unwrap()),
                TensorMut::I16(ViewMut::contiguous(&mut out, &f16_.shape).unwrap()),
                pool,
            )
            .unwrap();
            out.iter().flat_map(|v| v.to_le_bytes()).collect()
        });

        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: fi8.input_quant,
                output_dtype: DType::F32,
                output: Quant::IDENTITY,
            },
            feats,
        )
        .unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0.0f32; in_i8.len()];
            plan.run(
                TensorRef::I8(View::contiguous(&in_i8, &fi8.shape).unwrap()),
                TensorMut::F32(ViewMut::contiguous(&mut out, &fi8.shape).unwrap()),
                pool,
            )
            .unwrap();
            out.iter().flat_map(|v| v.to_le_bytes()).collect()
        });

        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: fi8.input_quant,
                output_dtype: DType::U16,
                output: Quant::new(1.0 / 65535.0, 0),
            },
            feats,
        )
        .unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0u16; in_i8.len()];
            plan.run(
                TensorRef::I8(View::contiguous(&in_i8, &fi8.shape).unwrap()),
                TensorMut::U16(ViewMut::contiguous(&mut out, &fi8.shape).unwrap()),
                pool,
            )
            .unwrap();
            out.iter().flat_map(|v| v.to_le_bytes()).collect()
        });
    }
}

#[test]
fn float_tensors_ignore_their_quant() {
    let zero = Quant::new(0.0, 0);
    let x: Vec<f32> = (-40..40).map(|i| i as f32 * 0.25).collect();
    let shape = [x.len()];
    let float_plan = |input: Quant, output: Quant, dtype: DType| {
        SigmoidPlan::new(SigmoidParams {
            input_dtype: dtype,
            input,
            output_dtype: dtype,
            output,
        })
        .unwrap()
    };

    let run_f32 = |plan: &SigmoidPlan| {
        let mut out = vec![0f32; x.len()];
        plan.run(
            TensorRef::F32(View::contiguous(&x, &shape).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut out, &shape).unwrap()),
            None,
        )
        .unwrap();
        out
    };
    let want = run_f32(&float_plan(Quant::IDENTITY, Quant::IDENTITY, DType::F32));
    assert_eq!(run_f32(&float_plan(zero, zero, DType::F32)), want);
    assert_eq!(
        run_f32(&float_plan(Quant::new(0.1, 7), zero, DType::F32)),
        want
    );

    let xh: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
    let run_f16 = |plan: &SigmoidPlan| {
        let mut out = vec![f16::ZERO; xh.len()];
        plan.run(
            TensorRef::F16(View::contiguous(&xh, &shape).unwrap()),
            TensorMut::F16(ViewMut::contiguous(&mut out, &shape).unwrap()),
            None,
        )
        .unwrap();
        out
    };
    let want_h = run_f16(&float_plan(Quant::IDENTITY, Quant::IDENTITY, DType::F16));
    assert_eq!(run_f16(&float_plan(zero, zero, DType::F16)), want_h);

    // A float input feeding a quantized output ignores the input quant only.
    let out_q = Quant::new(1.0 / 256.0, 0);
    let to_u8 = |input: Quant| {
        let plan = SigmoidPlan::new(SigmoidParams {
            input_dtype: DType::F32,
            input,
            output_dtype: DType::U8,
            output: out_q,
        })
        .unwrap();
        let mut out = vec![0u8; x.len()];
        plan.run(
            TensorRef::F32(View::contiguous(&x, &shape).unwrap()),
            TensorMut::U8(ViewMut::contiguous(&mut out, &shape).unwrap()),
            None,
        )
        .unwrap();
        out
    };
    assert_eq!(to_u8(zero), to_u8(Quant::IDENTITY));
    assert!(matches!(
        SigmoidPlan::new(SigmoidParams {
            input_dtype: DType::F32,
            input: zero,
            output_dtype: DType::U8,
            output: zero,
        }),
        Err(KernelError::InvalidArgument(_))
    ));
}
