// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic segmentation decoder: PyTorch goldens, a naive reference across
//! layouts and dtypes, thresholding, class maps, letterbox and errors.

use edgefirst_decoder::schema::{Activation, DecodeMode, Upsample, UpsampleMethod};
use edgefirst_decoder::semantic::{decode_semantic, SemanticConfig, SemanticOptions};
use edgefirst_decoder::{DecoderBuilder, DecoderError};
use edgefirst_tensor::{
    DType, Element, Quantization, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use half::f16;

fn named<T: Element + Copy>(shape: &[usize], values: &[T], name: Option<&str>) -> TensorDyn
where
    TensorDyn: From<Tensor<T>>,
{
    let t = Tensor::<T>::new(shape, Some(TensorMemory::Mem), name).unwrap();
    t.map().unwrap().as_mut_slice().copy_from_slice(values);
    t.into()
}

fn tensor<T: Element + Copy>(shape: &[usize], values: &[T]) -> TensorDyn
where
    TensorDyn: From<Tensor<T>>,
{
    named(shape, values, None)
}

fn zeros(shape: &[usize], dtype: DType) -> TensorDyn {
    TensorDyn::new(shape, dtype, Some(TensorMemory::Mem), None).unwrap()
}

fn read<T: Element + Copy>(t: &TensorDyn) -> Vec<T> {
    t.as_typed::<T>()
        .unwrap()
        .map()
        .unwrap()
        .as_slice()
        .to_vec()
}

fn classes(t: &TensorDyn) -> Vec<u16> {
    match t.dtype() {
        DType::U8 => read::<u8>(t).into_iter().map(u16::from).collect(),
        DType::U16 => read::<u16>(t),
        other => panic!("class map dtype {other:?}"),
    }
}

/// Deterministic pseudo-random values in `[-4, 4)` on a 1/16 grid.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 128) as f32 / 16.0 - 4.0
        })
        .collect()
}

fn nchw_to_nhwc(v: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0; v.len()];
    for k in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + k] = v[(k * h + y) * w + x];
            }
        }
    }
    out
}

/// Every-pixel, every-channel reference on NCHW `f32` scores.
#[allow(clippy::too_many_arguments)]
fn reference(
    v: &[f32],
    (c, h, w): (usize, usize, usize),
    (oh, ow): (usize, usize),
    letterbox: [f32; 4],
    mode: DecodeMode,
    align: bool,
) -> Vec<u16> {
    let roi = [
        letterbox[0] * w as f32,
        letterbox[1] * h as f32,
        (letterbox[2] - letterbox[0]) * w as f32,
        (letterbox[3] - letterbox[1]) * h as f32,
    ];
    let at = |k: usize, y: usize, x: usize| v[(k * h + y) * w + x];
    let argmax = |f: &dyn Fn(usize) -> f32| {
        let mut best = (0usize, f(0));
        for k in 1..c {
            let s = f(k);
            if s > best.1 {
                best = (k, s);
            }
        }
        best.0 as u16
    };
    let bil = |out: usize, len: usize, r0: f32, rl: f32, d: usize| {
        let src = if align {
            if out > 1 {
                r0 + d as f32 * ((rl - 1.0) / (out - 1) as f32)
            } else {
                r0
            }
        } else {
            r0 + (rl / out as f32) * (d as f32 + 0.5) - 0.5
        };
        let src = src.clamp(0.0, (len - 1) as f32);
        let lo = src as usize;
        (lo, (lo + 1).min(len - 1), src - lo as f32)
    };
    let near = |out: usize, len: usize, r0: f32, rl: f32, d: usize| {
        ((r0 + (rl / out as f32) * (d as f32 + 0.5)).clamp(0.0, (len - 1) as f32)) as usize
    };
    let mut out = Vec::with_capacity(oh * ow);
    for dy in 0..oh {
        for dx in 0..ow {
            out.push(match mode {
                DecodeMode::Accurate => {
                    let (y0, y1, fy) = bil(oh, h, roi[1], roi[3], dy);
                    let (x0, x1, fx) = bil(ow, w, roi[0], roi[2], dx);
                    argmax(&|k| {
                        (1.0 - fy) * ((1.0 - fx) * at(k, y0, x0) + fx * at(k, y0, x1))
                            + fy * ((1.0 - fx) * at(k, y1, x0) + fx * at(k, y1, x1))
                    })
                }
                _ => {
                    let sy = near(oh, h, roi[1], roi[3], dy);
                    let sx = near(ow, w, roi[0], roi[2], dx);
                    argmax(&|k| at(k, sy, sx))
                }
            });
        }
    }
    out
}

fn options(mode: DecodeMode, letterbox: [f32; 4]) -> SemanticOptions {
    SemanticOptions::new()
        .with_mode(mode)
        .with_letterbox(letterbox)
}

fn upsample(align_corners: bool) -> Upsample {
    serde_json::from_value(serde_json::json!({
        "method": "bilinear",
        "align_corners": align_corners
    }))
    .unwrap()
}

#[test]
fn matches_pytorch_golden() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/decoder/semantic_torch_golden.json"
    );
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let shape: Vec<usize> = serde_json::from_value(golden["shape"].clone()).unwrap();
    let [oh, ow]: [usize; 2] = serde_json::from_value(golden["out"].clone()).unwrap();
    let logits: Vec<f32> = serde_json::from_value(golden["logits"].clone()).unwrap();
    let input = tensor(&shape, &logits);
    for case in golden["cases"].as_array().unwrap() {
        let mode: DecodeMode = serde_json::from_value(case["mode"].clone()).unwrap();
        let align = case["align_corners"].as_bool().unwrap();
        let letterbox: [f32; 4] = serde_json::from_value(case["letterbox"].clone()).unwrap();
        let expected: Vec<u16> = serde_json::from_value(case["classes"].clone()).unwrap();
        let config = SemanticConfig::scores(&shape, 1, 2, 3)
            .unwrap()
            .with_upsample(upsample(align))
            .unwrap();
        let mut dst = zeros(&[oh, ow], DType::U8);
        decode_semantic(&input, &config, &mut dst, None, &options(mode, letterbox)).unwrap();
        assert_eq!(classes(&dst), expected, "{case}");
    }
}

#[test]
fn layouts_and_dtypes_match_reference() {
    let (c, h, w) = (6, 9, 11);
    let (oh, ow) = (23, 31);
    let v = values(c * h * w, 7);
    let letterboxes = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.8, 0.6],
        [0.1, 0.2, 0.9, 0.8],
    ];
    for nhwc in [false, true] {
        let (shape, axes) = if nhwc {
            (vec![1, h, w, c], (3, 1, 2))
        } else {
            (vec![1, c, h, w], (1, 2, 3))
        };
        let stored = if nhwc {
            nchw_to_nhwc(&v, c, h, w)
        } else {
            v.clone()
        };
        // Values sit on a 1/16 grid in [-4, 4): exact in f16, and in i8/u8
        // as 16*v (+128 for u8) with a matching scale and zero point.
        let inputs = [
            tensor(&shape, &stored),
            tensor(
                &shape,
                &stored.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>(),
            ),
            tensor(
                &shape,
                &stored.iter().map(|&x| (x * 16.0) as i8).collect::<Vec<_>>(),
            ),
            tensor(
                &shape,
                &stored
                    .iter()
                    .map(|&x| (x * 16.0 + 128.0) as u8)
                    .collect::<Vec<_>>(),
            ),
        ];
        let quants = [
            None,
            None,
            Some(Quantization::per_tensor_symmetric(1.0 / 16.0)),
            Some(Quantization::per_tensor(1.0 / 16.0, 128)),
        ];
        // u8 carries a zero point, so its interpolated values round
        // differently from the real ones; it is checked on stored values,
        // which differ from real-value argmax only at exact ties.
        let u8_nchw: Vec<f32> = v.iter().map(|&x| x * 16.0 + 128.0).collect();
        for (input, quant) in inputs.into_iter().zip(quants) {
            let mut input = input;
            let nchw = if input.dtype() == DType::U8 {
                &u8_nchw
            } else {
                &v
            };
            if let Some(q) = quant {
                input.set_quantization(q).unwrap();
            }
            for align in [false, true] {
                let config = SemanticConfig::scores(&shape, axes.0, axes.1, axes.2)
                    .unwrap()
                    .with_upsample(upsample(align))
                    .unwrap();
                for mode in [DecodeMode::Accurate, DecodeMode::Fast] {
                    for lb in letterboxes {
                        let mut dst = zeros(&[oh, ow], DType::U8);
                        decode_semantic(&input, &config, &mut dst, None, &options(mode, lb))
                            .unwrap();
                        let expected = reference(&nchw[..], (c, h, w), (oh, ow), lb, mode, align);
                        assert_eq!(
                            classes(&dst),
                            expected,
                            "nhwc={nhwc} dtype={:?} align={align} mode={mode:?} lb={lb:?}",
                            input.dtype()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn per_channel_quantization_dequantizes_first() {
    let (c, h, w) = (3, 4, 5);
    let v = values(c * h * w, 11);
    // Channel k stores v / scale_k; the argmax must be taken on real values.
    let scales = [0.5f32, 0.25, 0.125];
    let stored: Vec<i8> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (x / scales[i / (h * w)]).round() as i8)
        .collect();
    let mut input = tensor(&[1, c, h, w], &stored);
    input
        .set_quantization(Quantization::per_channel_symmetric(scales.to_vec(), 1).unwrap())
        .unwrap();
    let real: Vec<f32> = stored
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * scales[i / (h * w)])
        .collect();
    let config = SemanticConfig::scores(&[1, c, h, w], 1, 2, 3).unwrap();
    for mode in [DecodeMode::Accurate, DecodeMode::Fast] {
        let mut dst = zeros(&[9, 11], DType::U8);
        let opts = SemanticOptions::new().with_mode(mode);
        decode_semantic(&input, &config, &mut dst, None, &opts).unwrap();
        let expected = reference(&real, (c, h, w), (9, 11), [0.0, 0.0, 1.0, 1.0], mode, false);
        assert_eq!(classes(&dst), expected, "{mode:?}");
    }
}

#[test]
fn per_channel_quantization_needs_one_scale_per_class() {
    // The schema's quantization is not checked against the tensor shape.
    let input = tensor(&[1, 3, 2, 2], &[0i8; 12]);
    let config = SemanticConfig::scores(&[1, 3, 2, 2], 1, 2, 3)
        .unwrap()
        .with_quantization(Quantization::per_channel_symmetric(vec![0.5, 0.25], 1).unwrap());
    let mut dst = zeros(&[2, 2], DType::U8);
    let err =
        decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap_err();
    assert!(matches!(err, DecoderError::InvalidConfig(_)), "{err:?}");
}

#[test]
fn per_channel_axis_is_matched_across_unit_axes() {
    // The tensor drops the batch axis, so its class axis is 0, not 1.
    let (c, h, w) = (3, 4, 5);
    let v = values(c * h * w, 13);
    let scales = [0.5f32, 0.25, 0.125];
    let stored: Vec<i8> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (x / scales[i / (h * w)]).round() as i8)
        .collect();
    let mut input = tensor(&[c, h, w], &stored);
    input
        .set_quantization(Quantization::per_channel_symmetric(scales.to_vec(), 0).unwrap())
        .unwrap();
    let real: Vec<f32> = stored
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * scales[i / (h * w)])
        .collect();
    let config = SemanticConfig::scores(&[1, c, h, w], 1, 2, 3).unwrap();
    let mut dst = zeros(&[h, w], DType::U8);
    decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap();
    let expected = reference(
        &real,
        (c, h, w),
        (h, w),
        [0.0, 0.0, 1.0, 1.0],
        DecodeMode::Accurate,
        false,
    );
    assert_eq!(classes(&dst), expected);
}

#[test]
fn integer_scores_need_quantization_for_real_values() {
    let config = SemanticConfig::scores(&[1, 1, 2, 2], 1, 2, 3).unwrap();
    let input = tensor(&[1, 1, 2, 2], &[0u8, 100, 200, 255]);
    let mut dst = zeros(&[2, 2], DType::U8);
    let err =
        decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap_err();
    assert!(matches!(err, DecoderError::QuantMissing { .. }), "{err:?}");

    // A multi-class argmax needs no real values; confidence does.
    let config = SemanticConfig::scores(&[1, 2, 2, 2], 1, 2, 3).unwrap();
    let input = tensor(&[1, 2, 2, 2], &[0u8, 9, 0, 9, 9, 0, 9, 0]);
    decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap();
    assert_eq!(classes(&dst), vec![1, 0, 1, 0]);
    let mut conf = zeros(&[2, 2], DType::F32);
    let err = decode_semantic(
        &input,
        &config,
        &mut dst,
        Some(&mut conf),
        &SemanticOptions::new(),
    )
    .unwrap_err();
    assert!(matches!(err, DecoderError::QuantMissing { .. }), "{err:?}");
}

#[test]
fn an_out_of_range_position_is_an_error() {
    let config = SemanticConfig::scores(&[1, 2, 2, 2], 1, 2, 3)
        .unwrap()
        .with_position(3, 2);
    let a = tensor(&[1, 2, 2, 2], &[0.0f32; 8]);
    let b = tensor(&[1, 2, 2, 2], &[0.0f32; 8]);
    let err = edgefirst_decoder::semantic::select_output(&[&a, &b], &config).unwrap_err();
    assert!(matches!(err, DecoderError::InvalidConfig(_)), "{err:?}");
}

#[test]
fn class_map_follows_its_dshape_axes() {
    // Stored width-major: shape [1, W=3, H=2].
    let map: Vec<u8> = vec![0, 3, 1, 4, 2, 5];
    let config = SemanticConfig::class_map(&[1, 3, 2], 2, 1).unwrap();
    let mut dst = zeros(&[2, 3], DType::U8);
    let input = tensor(&[1, 3, 2], &map);
    decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap();
    assert_eq!(classes(&dst), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn ties_take_the_lowest_class() {
    // Every class scores the same everywhere: torch.argmax picks class 0.
    let input = tensor(&[1, 4, 3, 3], &[1.0f32; 36]);
    let config = SemanticConfig::scores(&[1, 4, 3, 3], 1, 2, 3).unwrap();
    for mode in [DecodeMode::Accurate, DecodeMode::Fast] {
        let mut dst = zeros(&[7, 7], DType::U8);
        let opts = SemanticOptions::new().with_mode(mode);
        decode_semantic(&input, &config, &mut dst, None, &opts).unwrap();
        assert!(classes(&dst).iter().all(|&k| k == 0), "{mode:?}");
    }
    // Classes 1 and 3 tie above the rest: class 1 wins.
    let mut v = vec![0.0f32; 36];
    v[9..18].fill(2.0);
    v[27..36].fill(2.0);
    let input = tensor(&[1, 4, 3, 3], &v);
    let mut dst = zeros(&[7, 7], DType::U8);
    decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap();
    assert!(classes(&dst).iter().all(|&k| k == 1));
}

#[test]
fn single_channel_thresholds() {
    // A horizontal ramp of logits from -2 to 2.
    let v: Vec<f32> = (0..5)
        .flat_map(|_| (0..5).map(|x| x as f32 - 2.0))
        .collect();
    let logits = SemanticConfig::scores(&[1, 1, 5, 5], 1, 2, 3).unwrap();
    for mode in [DecodeMode::Accurate, DecodeMode::Fast] {
        let mut dst = zeros(&[5, 5], DType::U8);
        decode_semantic(
            &tensor(&[1, 1, 5, 5], &v),
            &logits,
            &mut dst,
            None,
            &SemanticOptions::new().with_mode(mode),
        )
        .unwrap();
        // Logit threshold 0 is exclusive: column 2 (logit 0) is background.
        assert_eq!(&classes(&dst)[..5], &[0, 0, 0, 1, 1], "{mode:?}");
    }

    // Probabilities in u8 with scale 1/255: the default threshold 0.5 is
    // compared as a real value, not as the raw integer.
    let probs: Vec<u8> = (0..5).flat_map(|_| [0u8, 64, 127, 128, 255]).collect();
    let mut input = tensor(&[1, 5, 5, 1], &probs);
    input
        .set_quantization(Quantization::per_tensor(1.0 / 255.0, 0))
        .unwrap();
    let config = SemanticConfig::scores(&[1, 5, 5, 1], 3, 1, 2)
        .unwrap()
        .with_activation(Activation::Sigmoid);
    let mut dst = zeros(&[5, 5], DType::U8);
    let mut conf = zeros(&[5, 5], DType::F32);
    decode_semantic(
        &input,
        &config,
        &mut dst,
        Some(&mut conf),
        &SemanticOptions::new(),
    )
    .unwrap();
    assert_eq!(&classes(&dst)[..5], &[0, 0, 0, 1, 1]);
    let conf = read::<f32>(&conf);
    assert!((conf[0] - 1.0).abs() < 1e-6);
    assert!((conf[1] - (1.0 - 64.0 / 255.0)).abs() < 1e-6);

    let config = config.with_threshold(0.2);
    decode_semantic(&input, &config, &mut dst, None, &SemanticOptions::new()).unwrap();
    assert_eq!(&classes(&dst)[..5], &[0, 1, 1, 1, 1]);
}

#[test]
fn confidence_is_the_softmax_maximum() {
    // Two classes, logits 0 and ln 3 everywhere: p = 3/4.
    let v = [vec![0.0f32; 4], vec![3f32.ln(); 4]].concat();
    let input = tensor(&[1, 2, 2, 2], &v);
    let config = SemanticConfig::scores(&[1, 2, 2, 2], 1, 2, 3).unwrap();
    for mode in [DecodeMode::Accurate, DecodeMode::Fast] {
        let mut dst = zeros(&[4, 4], DType::U8);
        let mut conf = zeros(&[4, 4], DType::U8);
        let opts = SemanticOptions::new().with_mode(mode);
        decode_semantic(&input, &config, &mut dst, Some(&mut conf), &opts).unwrap();
        assert!(classes(&dst).iter().all(|&k| k == 1));
        assert!(read::<u8>(&conf).iter().all(|&p| p == 191), "{mode:?}");
    }
}

#[test]
fn class_map_input_resizes_with_nearest() {
    let map: Vec<i64> = vec![0, 1, 2, 3, 4, 5];
    let config = SemanticConfig::class_map(&[1, 2, 3], 1, 2).unwrap();
    let mut dst = zeros(&[4, 6], DType::U8);
    decode_semantic(
        &tensor(&[1, 2, 3], &map),
        &config,
        &mut dst,
        None,
        &SemanticOptions::new(),
    )
    .unwrap();
    assert_eq!(
        classes(&dst),
        vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5]
    );
    // A letterbox keeping the left two columns.
    let mut dst = zeros(&[2, 4], DType::U16);
    let opts = SemanticOptions::new().with_letterbox([0.0, 0.0, 2.0 / 3.0, 1.0]);
    decode_semantic(&tensor(&[1, 2, 3], &map), &config, &mut dst, None, &opts).unwrap();
    assert_eq!(classes(&dst), vec![0, 0, 1, 1, 3, 3, 4, 4]);

    for (dtype, values) in [
        (DType::I32, vec![0i32, 1, 2, 3, 300, 5]),
        (DType::I32, vec![0, -1, 2, 3, 4, 5]),
    ] {
        let mut dst = zeros(&[2, 3], DType::U8);
        let err = decode_semantic(
            &tensor(&[1, 2, 3], &values),
            &config,
            &mut dst,
            None,
            &SemanticOptions::new(),
        )
        .unwrap_err();
        assert!(
            matches!(err, DecoderError::InvalidConfig(_)),
            "{dtype:?} {err:?}"
        );
    }
}

#[test]
fn centred_letterbox_crops_both_sides() {
    // Columns 0 and 3 are padding; the content is columns 1-2 of class 1 and 2.
    let mut v = vec![0.0f32; 2 * 2 * 4];
    for y in 0..2 {
        v[y * 4 + 1] = 1.0; // class 0 low, class 1 at column 1
        v[8 + y * 4 + 2] = 1.0; // class 1 at column 2
    }
    // Class 0 scores 1 on the padding columns.
    for y in 0..2 {
        v[y * 4] = 1.0;
        v[y * 4 + 3] = 1.0;
        v[y * 4 + 1] = 0.0;
    }
    let config = SemanticConfig::scores(&[1, 2, 2, 4], 1, 2, 3).unwrap();
    let opts = SemanticOptions::new()
        .with_mode(DecodeMode::Fast)
        .with_letterbox([0.25, 0.0, 0.75, 1.0]);
    let mut dst = zeros(&[2, 4], DType::U8);
    decode_semantic(&tensor(&[1, 2, 2, 4], &v), &config, &mut dst, None, &opts).unwrap();
    assert_eq!(classes(&dst), vec![0, 0, 1, 1, 0, 0, 1, 1]);
}

#[test]
fn rejects_invalid_requests() {
    let input = tensor(&[1, 300, 2, 2], &vec![0.0f32; 1200]);
    let config = SemanticConfig::scores(&[1, 300, 2, 2], 1, 2, 3).unwrap();
    let opts = SemanticOptions::new();
    let err = |dst: &mut TensorDyn, opts: &SemanticOptions| {
        decode_semantic(&input, &config, dst, None, opts).unwrap_err()
    };
    // 300 classes do not fit u8, but do fit u16.
    assert!(matches!(
        err(&mut zeros(&[4, 4], DType::U8), &opts),
        DecoderError::NotSupported(_)
    ));
    decode_semantic(
        &input,
        &config,
        &mut zeros(&[4, 4], DType::U16),
        None,
        &opts,
    )
    .unwrap();
    assert!(matches!(
        err(&mut zeros(&[4, 4], DType::I32), &opts),
        DecoderError::NotSupported(_)
    ));
    assert!(matches!(
        err(&mut zeros(&[1, 4, 4], DType::U16), &opts),
        DecoderError::InvalidShape(_)
    ));
    for lb in [
        [0.5, 0.0, 0.5, 1.0],
        [0.0, 0.0, 1.5, 1.0],
        [f32::NAN, 0.0, 1.0, 1.0],
    ] {
        let bad = SemanticOptions::new().with_letterbox(lb);
        assert!(matches!(
            err(&mut zeros(&[4, 4], DType::U16), &bad),
            DecoderError::InvalidConfig(_)
        ));
    }
    let other = tensor(&[1, 3, 2, 2], &[0.0f32; 12]);
    let e = decode_semantic(
        &other,
        &config,
        &mut zeros(&[4, 4], DType::U16),
        None,
        &opts,
    )
    .unwrap_err();
    assert!(matches!(e, DecoderError::InvalidShape(_)));
    let bicubic: Upsample =
        serde_json::from_value(serde_json::json!({"method": "bicubic"})).unwrap();
    assert!(matches!(
        config.clone().with_upsample(bicubic),
        Err(DecoderError::NotSupported(_))
    ));
    let nearest = Upsample {
        method: UpsampleMethod::Nearest,
        ..bicubic
    };
    config.with_upsample(nearest).unwrap();
}

const SEGFORMER: &str = r#"{
  "schema_version": 2,
  "input": {"shape": [1, 3, 32, 48], "dshape": [{"batch": 1}, {"channels": 3}, {"height": 32}, {"width": 48}]},
  "outputs": [
    {"name": "logits", "type": "segmentation", "decoder": "semantic",
     "shape": [1, 4, 8, 12],
     "dshape": [{"batch": 1}, {"num_classes": 4}, {"height": 8}, {"width": 12}],
     "dtype": "float32", "stride": 4,
     "upsample": {"method": "bilinear", "align_corners": false},
     "decode_mode": "fast"},
    {"name": "aux", "shape": [1, 4, 8, 12],
     "dshape": [{"batch": 1}, {"num_classes": 4}, {"height": 8}, {"width": 12}],
     "dtype": "float32"}
  ]
}"#;

#[test]
fn decoder_from_schema_decodes_semantic_output() {
    let decoder = DecoderBuilder::new()
        .with_config_json_str(SEGFORMER.to_string())
        .build()
        .unwrap();
    let semantic = decoder.semantic().unwrap();
    assert_eq!(semantic.classes(), 4);
    assert_eq!(semantic.mode(), DecodeMode::Fast);
    assert_eq!(semantic.name(), Some("logits"));

    let v = values(4 * 8 * 12, 3);
    let logits = named(&[1, 4, 8, 12], &v, Some("logits"));
    let aux = named(&[1, 4, 8, 12], &vec![0.0f32; 384], Some("aux"));
    let lb = [0.0, 0.0, 1.0, 0.75];
    for (mode, opts) in [
        (DecodeMode::Fast, SemanticOptions::new().with_letterbox(lb)),
        (DecodeMode::Accurate, options(DecodeMode::Accurate, lb)),
    ] {
        let mut dst = zeros(&[30, 60], DType::U8);
        decoder
            .decode_semantic(&[&aux, &logits], &mut dst, None, &opts)
            .unwrap();
        assert_eq!(
            classes(&dst),
            reference(&v, (4, 8, 12), (30, 60), lb, mode, false),
            "{mode:?}"
        );
    }

    // Unnamed tensors with the same shape bind by schema position when all
    // outputs are passed, and are ambiguous otherwise.
    let a = tensor(&[1, 4, 8, 12], &v);
    let b = tensor(&[1, 4, 8, 12], &vec![0.0f32; 384]);
    let mut dst = zeros(&[8, 12], DType::U8);
    decoder
        .decode_semantic(&[&a, &b], &mut dst, None, &SemanticOptions::new())
        .unwrap();
    let mut by_position = zeros(&[8, 12], DType::U8);
    decoder
        .decode_semantic(&[&b, &a], &mut by_position, None, &SemanticOptions::new())
        .unwrap();
    assert!(classes(&by_position).iter().all(|&k| k == 0));
    assert!(classes(&dst).iter().any(|&k| k != 0));
    let err = decoder
        .decode_semantic(&[&a, &b, &a], &mut dst, None, &SemanticOptions::new())
        .unwrap_err();
    assert!(matches!(err, DecoderError::InvalidConfig(_)));

    let mut boxes = Vec::new();
    let mut masks = Vec::new();
    let err = decoder
        .decode(&[&logits], &mut boxes, &mut masks)
        .unwrap_err();
    assert!(matches!(err, DecoderError::NotSupported(_)));
}

#[test]
fn semantic_output_cannot_be_combined_with_detection() {
    let json = SEGFORMER.replace(
        r#"{"name": "aux", "shape""#,
        r#"{"name": "aux", "type": "scores", "decoder": "ultralytics", "shape""#,
    );
    let err = DecoderBuilder::new()
        .with_config_json_str(json)
        .build()
        .unwrap_err();
    assert!(matches!(err, DecoderError::NotSupported(_)), "{err:?}");
}

#[test]
fn modelpack_segmentation_decodes_through_semantic() {
    let yaml = r#"
outputs:
  - decoder: modelpack
    type: segmentation
    shape: [1, 6, 8, 3]
    quantization: [0.0625, 128]
"#;
    let decoder = DecoderBuilder::new()
        .with_config_yaml_str(yaml.to_string())
        .build()
        .unwrap();
    let semantic = decoder.semantic().unwrap();
    assert_eq!(semantic.mode(), DecodeMode::Fast);
    assert_eq!(semantic.classes(), 3);

    let v = values(3 * 6 * 8, 5);
    let nhwc: Vec<u8> = nchw_to_nhwc(&v, 3, 6, 8)
        .iter()
        .map(|&x| (x * 16.0 + 128.0) as u8)
        .collect();
    let output = tensor(&[1, 6, 8, 3], &nhwc);
    let mut dst = zeros(&[12, 16], DType::U8);
    decoder
        .decode_semantic(&[&output], &mut dst, None, &SemanticOptions::new())
        .unwrap();
    assert_eq!(
        classes(&dst),
        reference(
            &v,
            (3, 6, 8),
            (12, 16),
            [0.0, 0.0, 1.0, 1.0],
            DecodeMode::Fast,
            false
        )
    );
}
