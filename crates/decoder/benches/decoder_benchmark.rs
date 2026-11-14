// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::excessive_precision)]

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_decoder::{
    byte::{nms_int, postprocess_boxes_multilabel_index_quant, postprocess_boxes_quant},
    configs, dequant_detect_box, dequantize_cpu, dequantize_cpu_chunked, dequantize_ndarray,
    float::{nms_float, postprocess_boxes_float, postprocess_boxes_multilabel_index_float},
    per_scale::DecodeDtype,
    schema::SchemaV2,
    ConfigOutput, DecoderBuilder, Nms, Quantization, XYWH,
};
use edgefirst_tensor::{
    Quantization as TQ, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use ndarray::s;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

fn bench_yolo_quant(suite: &mut BenchSuite) {
    let raw = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
    let tensor = TensorDyn::I8(Tensor::<i8>::from_slice(raw, &[1, 84, 8400]).unwrap());

    let detection = configs::Detection {
        decoder: configs::DecoderType::Ultralytics,
        quantization: Some(configs::QuantTuple(0.0040811873, -123)),
        shape: vec![1, 84, 8400],
        dshape: vec![],
        anchors: None,
        normalized: Some(true),
    };
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.25)
        .with_iou_threshold(0.70)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(detection))
        .build()
        .expect("yolo det quant decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&tensor];
    let result = run_bench("decoder/yolo/quant", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut masks = Vec::new();
        decoder
            .decode(&inputs, &mut output_boxes, &mut masks)
            .unwrap();
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_quant_decode_boxes(suite: &mut BenchSuite) {
    let score_threshold = 0.25;
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();

    let result = run_bench("decoder/quant/decode_boxes", WARMUP, ITERATIONS, || {
        let _ = postprocess_boxes_quant::<XYWH, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant,
        );
    });
    result.print_summary();
    suite.record(&result);
}

/// Multi-label candidate selection on the transposed score view a DMA-BUF
/// output produces, at a deployment and a validation score threshold.
fn bench_quant_decode_boxes_multilabel(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
    for (name, score_threshold) in [
        ("decoder/quant/decode_boxes_multilabel", 0.25f32),
        ("decoder/quant/decode_boxes_multilabel_val", 0.001f32),
    ] {
        let threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let _ = postprocess_boxes_multilabel_index_quant::<XYWH, _, _>(
                threshold,
                boxes_tensor,
                scores_tensor,
                quant,
            );
        });
        result.print_summary();
        suite.record(&result);
    }
}

/// Argmax candidate selection on transposed `i16` scores and boxes, the i8 tensor widened.
fn bench_quant_decode_boxes_i16(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out: Vec<i16> = out.iter().map(|&x| (x as i8 as i16) << 8).collect();
    let out = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
    let quant = Quantization {
        scale: 0.0040811873 / 256.0,
        zero_point: -123 * 256,
    };
    let threshold = (0.25 / quant.scale + quant.zero_point as f32) as i16;
    let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
    let result = run_bench("decoder/quant/decode_boxes_i16", WARMUP, ITERATIONS, || {
        let boxes =
            postprocess_boxes_quant::<XYWH, _, _>(threshold, boxes_tensor, scores_tensor, quant);
        std::hint::black_box(boxes);
    });
    result.print_summary();
    suite.record(&result);
}

/// Argmax candidate selection on contiguous `[N, C]` score rows (the row-major layout).
fn bench_decode_boxes_rows(suite: &mut BenchSuite) {
    let raw = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let rows = ndarray::Array2::from_shape_vec((84, 8400), raw.to_vec())
        .unwrap()
        .reversed_axes()
        .as_standard_layout()
        .into_owned();
    let threshold = (0.25 / quant.scale + quant.zero_point as f32) as i8;
    let boxes = rows.slice(s![.., ..4]);
    let scores = rows.slice(s![.., 4..]);
    let result = run_bench(
        "decoder/quant/decode_boxes_rows",
        WARMUP,
        ITERATIONS,
        || {
            let kept = postprocess_boxes_quant::<XYWH, _, _>(threshold, boxes, scores, quant);
            std::hint::black_box(kept);
        },
    );
    result.print_summary();
    suite.record(&result);

    let f32s = rows.mapv(|v| (v as f32 - quant.zero_point as f32) * quant.scale);
    let boxes = f32s.slice(s![.., ..4]);
    let scores = f32s.slice(s![.., 4..]);
    let result = run_bench("decoder/f32/decode_boxes_rows", WARMUP, ITERATIONS, || {
        let kept = postprocess_boxes_float::<XYWH, _, _>(0.25f32, boxes, scores);
        std::hint::black_box(kept);
    });
    result.print_summary();
    suite.record(&result);

    let f16s = f32s.mapv(half::f16::from_f32);
    let boxes = f16s.slice(s![.., ..4]);
    let scores = f16s.slice(s![.., 4..]);
    let threshold = half::f16::from_f32(0.25);
    let result = run_bench("decoder/f16/decode_boxes_rows", WARMUP, ITERATIONS, || {
        let kept = postprocess_boxes_float::<XYWH, _, _>(threshold, boxes, scores);
        std::hint::black_box(kept);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_quant_nms(suite: &mut BenchSuite) {
    let score_threshold = 0.01;
    let iou_threshold = 0.70;
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.clone()).unwrap();
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
    let boxes =
        postprocess_boxes_quant::<XYWH, _, _>(score_threshold, boxes_tensor, scores_tensor, quant);

    let result = run_bench("decoder/quant/nms", WARMUP, ITERATIONS, || {
        let boxes = boxes.clone();
        let cap = output_boxes.capacity();
        let boxes = nms_int(iou_threshold, (cap > 0).then_some(cap), boxes);
        let len = cap.min(boxes.len());
        output_boxes.clear();
        for b in boxes.iter().take(len) {
            output_boxes.push(dequant_detect_box(b, quant));
        }
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_yolo_f32(suite: &mut BenchSuite) {
    // Pre-dequantize once outside the timed loop. The bench measures
    // the float decode path; mixing dequant cost into each iteration
    // (as the legacy version did) inflated wall-time by the dequant.
    let raw = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![0.0f32; 84 * 8400];
    dequantize_cpu_chunked(raw, quant, &mut buf);
    let tensor = TensorDyn::F32(Tensor::<f32>::from_slice(&buf, &[1, 84, 8400]).unwrap());

    let detection = configs::Detection {
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, 84, 8400],
        dshape: vec![],
        anchors: None,
        normalized: Some(true),
    };
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.25)
        .with_iou_threshold(0.70)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(detection))
        .build()
        .expect("yolo det f32 decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&tensor];
    let result = run_bench("decoder/yolo/f32", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut masks = Vec::new();
        decoder
            .decode(&inputs, &mut output_boxes, &mut masks)
            .unwrap();
        std::hint::black_box(output_boxes);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_dequantize_i8(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];

    let result = run_bench("decoder/dequantize/i8", WARMUP, ITERATIONS, || {
        let mut buf = buf.clone();
        dequantize_cpu(&out, quant, &mut buf);
        let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
        std::hint::black_box(out);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_dequantize_i8_chunked(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];

    let result = run_bench("decoder/dequantize/i8_chunked", WARMUP, ITERATIONS, || {
        let mut buf = buf.clone();
        dequantize_cpu_chunked(&out, quant, &mut buf);
        let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
        std::hint::black_box(out);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_dequantize_i16(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out: Vec<_> = out.iter().map(|x| *x as i16).collect();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];

    let result = run_bench("decoder/dequantize/i16", WARMUP, ITERATIONS, || {
        let mut buf = buf.clone();
        dequantize_cpu(&out, quant, &mut buf);
        let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
        std::hint::black_box(out);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_dequantize_i16_chunked(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out: Vec<_> = out.iter().map(|x| *x as i16).collect();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];

    let result = run_bench("decoder/dequantize/i16_chunked", WARMUP, ITERATIONS, || {
        let mut buf = buf.clone();
        dequantize_cpu_chunked(&out, quant, &mut buf);
        let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
        std::hint::black_box(out);
    });
    result.print_summary();
    suite.record(&result);
}

/// The generic dequantize into f16 and into a preallocated f32 buffer, and `dequantize_ndarray`.
fn bench_dequantize_f16_and_ndarray(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![half::f16::ZERO; 84 * 8400];
    let result = run_bench("decoder/dequantize/i8_f16", WARMUP, ITERATIONS, || {
        dequantize_cpu(&out, quant, &mut buf);
        std::hint::black_box(&buf);
    });
    result.print_summary();
    suite.record(&result);

    let mut buf = vec![0.0f32; 84 * 8400];
    let result = run_bench("decoder/dequantize/i8_into", WARMUP, ITERATIONS, || {
        dequantize_cpu(&out, quant, &mut buf);
        std::hint::black_box(&buf);
    });
    result.print_summary();
    suite.record(&result);

    let wide: Vec<i16> = out.iter().map(|&x| i16::from(x) << 8).collect();
    let result = run_bench("decoder/dequantize/i16_into", WARMUP, ITERATIONS, || {
        dequantize_cpu(&wide, quant, &mut buf);
        std::hint::black_box(&buf);
    });
    result.print_summary();
    suite.record(&result);

    let arr = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
    let result = run_bench("decoder/dequantize/ndarray_i8", WARMUP, ITERATIONS, || {
        let deq: ndarray::Array2<f32> = dequantize_ndarray(arr.view(), quant);
        std::hint::black_box(deq);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_f32_decode_boxes(suite: &mut BenchSuite) {
    let score_threshold = 0.25;
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![0.0; 84 * 8400];
    dequantize_cpu_chunked(&out, quant, &mut buf);

    let result = run_bench("decoder/f32/decode_boxes", WARMUP, ITERATIONS, || {
        let out = buf.clone();
        let out = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
        let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
        let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
        let boxes =
            postprocess_boxes_float::<XYWH, _, _>(score_threshold, boxes_tensor, scores_tensor);
        std::hint::black_box(boxes);
    });
    result.print_summary();
    suite.record(&result);
}

/// Argmax candidate selection on the transposed f16 and f32 views (no per-iteration copy) and
/// multi-label on the f32 one.
fn bench_float_decode_boxes_views(suite: &mut BenchSuite) {
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![0.0f32; 84 * 8400];
    dequantize_cpu_chunked(out, quant, &mut buf);
    let f32s = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
    let f16s = f32s.mapv(half::f16::from_f32);

    let boxes_tensor = f16s.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = f16s.slice(s![4..(80 + 4), ..,]).reversed_axes();
    let threshold = half::f16::from_f32(0.25);
    let result = run_bench("decoder/f16/decode_boxes", WARMUP, ITERATIONS, || {
        let boxes = postprocess_boxes_float::<XYWH, _, _>(threshold, boxes_tensor, scores_tensor);
        std::hint::black_box(boxes);
    });
    result.print_summary();
    suite.record(&result);

    let boxes_tensor = f32s.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = f32s.slice(s![4..(80 + 4), ..,]).reversed_axes();
    let result = run_bench("decoder/f32/decode_boxes_view", WARMUP, ITERATIONS, || {
        let boxes = postprocess_boxes_float::<XYWH, _, _>(0.25f32, boxes_tensor, scores_tensor);
        std::hint::black_box(boxes);
    });
    result.print_summary();
    suite.record(&result);

    let result = run_bench(
        "decoder/f32/decode_boxes_multilabel",
        WARMUP,
        ITERATIONS,
        || {
            let boxes = postprocess_boxes_multilabel_index_float::<XYWH, _, _>(
                0.25f32,
                boxes_tensor,
                scores_tensor,
            );
            std::hint::black_box(boxes);
        },
    );
    result.print_summary();
    suite.record(&result);
}

fn bench_f32_nms(suite: &mut BenchSuite) {
    let score_threshold = 0.01;
    let iou_threshold = 0.70;
    let out = edgefirst_bench::testdata::read("yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![0.0; 84 * 8400];
    dequantize_cpu_chunked(&out, quant, &mut buf);
    let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
    let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
    let boxes = postprocess_boxes_float::<XYWH, _, _>(score_threshold, boxes_tensor, scores_tensor);

    let result = run_bench("decoder/f32/nms", WARMUP, ITERATIONS, || {
        let boxes = boxes.clone();
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let cap = output_boxes.capacity();
        let boxes = nms_float(iou_threshold, (cap > 0).then_some(cap), boxes);
        let len = cap.min(boxes.len());
        output_boxes.clear();
        for b in boxes.into_iter().take(len) {
            output_boxes.push(b);
        }
        std::hint::black_box(output_boxes);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_modelpack_u8(suite: &mut BenchSuite) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let boxes = edgefirst_bench::testdata::read("modelpack_boxes_1935x1x4.bin");
    let boxes_tensor = TensorDyn::U8(Tensor::<u8>::from_slice(&boxes, &[1, 1935, 1, 4]).unwrap());

    let scores = edgefirst_bench::testdata::read("modelpack_scores_1935x1.bin");
    let scores_tensor = TensorDyn::U8(Tensor::<u8>::from_slice(&scores, &[1, 1935, 1]).unwrap());

    let quant_boxes: configs::QuantTuple = (0.004656755365431309, 21).into();
    let quant_scores: configs::QuantTuple = (0.0019603664986789227, 0).into();

    let decoder = DecoderBuilder::default()
        .with_config_modelpack_det(
            configs::Boxes {
                decoder: configs::DecoderType::ModelPack,
                quantization: Some(quant_boxes),
                shape: vec![1, 1935, 1, 4],
                dshape: vec![
                    (configs::DimName::Batch, 1),
                    (configs::DimName::NumBoxes, 1935),
                    (configs::DimName::Padding, 1),
                    (configs::DimName::BoxCoords, 4),
                ],
                normalized: Some(true),
            },
            configs::Scores {
                decoder: configs::DecoderType::ModelPack,
                quantization: Some(quant_scores),
                shape: vec![1, 1935, 1],
                dshape: vec![
                    (configs::DimName::Batch, 1),
                    (configs::DimName::NumBoxes, 1935),
                    (configs::DimName::NumClasses, 1),
                ],
            },
        )
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .expect("modelpack u8 decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&boxes_tensor, &scores_tensor];
    let result = run_bench("decoder/modelpack/u8", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        decoder
            .decode(&inputs, &mut output_boxes, &mut output_masks)
            .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_modelpack_split_u8(suite: &mut BenchSuite) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;

    let detect0 = edgefirst_bench::testdata::read("modelpack_split_9x15x18.bin");
    let detect0_tensor =
        TensorDyn::U8(Tensor::<u8>::from_slice(&detect0, &[1, 9, 15, 18]).unwrap());

    let detect1 = edgefirst_bench::testdata::read("modelpack_split_17x30x18.bin");
    let detect1_tensor =
        TensorDyn::U8(Tensor::<u8>::from_slice(&detect1, &[1, 17, 30, 18]).unwrap());

    let detect_config0 = configs::Detection {
        decoder: configs::DecoderType::ModelPack,
        shape: vec![1, 9, 15, 18],
        anchors: Some(vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ]),
        quantization: Some((0.08547406643629074, 174).into()),
        dshape: vec![
            (configs::DimName::Batch, 1),
            (configs::DimName::Height, 9),
            (configs::DimName::Width, 15),
            (configs::DimName::NumAnchorsXFeatures, 18),
        ],
        normalized: Some(true),
    };

    let detect_config1 = configs::Detection {
        decoder: configs::DecoderType::ModelPack,
        shape: vec![1, 17, 30, 18],
        anchors: Some(vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ]),
        quantization: Some((0.09929127991199493, 183).into()),
        dshape: vec![
            (configs::DimName::Batch, 1),
            (configs::DimName::Height, 17),
            (configs::DimName::Width, 30),
            (configs::DimName::NumAnchorsXFeatures, 18),
        ],
        normalized: Some(true),
    };

    let decoder = DecoderBuilder::default()
        .with_config_modelpack_det_split(vec![detect_config1, detect_config0])
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .expect("modelpack split u8 decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&detect0_tensor, &detect1_tensor];
    let result = run_bench("decoder/modelpack/split_u8", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        decoder
            .decode(&inputs, &mut output_boxes, &mut output_masks)
            .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
    result.print_summary();
    suite.record(&result);
}

/// `n` detection candidates in clusters of about eight overlapping boxes (realistic NMS input),
/// or all disjoint (every box survives: the full O(N²) scan), labels from 4 classes.
fn nms_candidates(n: usize, disjoint: bool) -> Vec<edgefirst_decoder::DetectBox> {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    let unit = |rng: &mut XorShift| rng.next() as f32 / u32::MAX as f32;
    (0..n)
        .map(|k| {
            let (cx, cy, half) = if disjoint {
                let side = (n as f32).sqrt().ceil();
                let (gx, gy) = ((k as f32 % side), (k as f32 / side).floor());
                ((gx + 0.5) / side, (gy + 0.5) / side, 0.3 / side)
            } else {
                let c = (k / 8) as u32;
                let mut crng = XorShift(0x1234_5678 + c as u64);
                let (cx, cy) = (unit(&mut crng), unit(&mut crng));
                let half = 0.02 + 0.06 * unit(&mut crng);
                let j = 0.15 * half;
                (cx + j * unit(&mut rng), cy + j * unit(&mut rng), half)
            };
            edgefirst_decoder::DetectBox {
                bbox: edgefirst_decoder::BoundingBox::new(
                    cx - half,
                    cy - half,
                    cx + half,
                    cy + half,
                ),
                score: 0.25 + 0.75 * unit(&mut rng),
                label: (rng.below(4)) as usize,
            }
        })
        .collect()
}

/// Greedy NMS over 300 candidates (the default `pre_nms_top_k`), capped at 300 survivors:
/// clustered class-agnostic, class-aware and quantized, and disjoint (all survive); and over 6.
fn bench_nms_300(suite: &mut BenchSuite) {
    use edgefirst_decoder::float::nms_class_aware_float;
    let clustered = nms_candidates(300, false);
    let disjoint = nms_candidates(300, true);
    let few = nms_candidates(6, false);
    let quantized: Vec<edgefirst_decoder::DetectBoxQuantized<i8>> = clustered
        .iter()
        .map(|b| edgefirst_decoder::DetectBoxQuantized {
            bbox: b.bbox,
            score: (b.score * 254.0 - 127.0) as i8,
            label: b.label,
        })
        .collect();
    let cases: [(&str, &dyn Fn() -> usize); 5] = [
        ("decoder/nms/f32_6", &|| {
            nms_float(0.45, Some(300), few.clone()).len()
        }),
        ("decoder/nms/f32_300", &|| {
            nms_float(0.45, Some(300), clustered.clone()).len()
        }),
        ("decoder/nms/f32_300_class_aware", &|| {
            nms_class_aware_float(0.45, Some(300), clustered.clone()).len()
        }),
        ("decoder/nms/f32_300_disjoint", &|| {
            nms_float(0.45, Some(300), disjoint.clone()).len()
        }),
        ("decoder/nms/i8_300", &|| {
            nms_int(0.45, Some(300), quantized.clone()).len()
        }),
    ];
    for (name, run) in cases {
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            std::hint::black_box(run());
        });
        result.print_summary();
        suite.record(&result);
    }
}

/// A ModelPack split detection config for one `(1, h, w, 3 · (nc + 5))` output.
fn modelpack_split_config(
    h: usize,
    w: usize,
    nc: usize,
    quantization: Option<configs::QuantTuple>,
) -> configs::Detection {
    let c = 3 * (nc + 5);
    configs::Detection {
        decoder: configs::DecoderType::ModelPack,
        shape: vec![1, h, w, c],
        anchors: Some(vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ]),
        quantization,
        dshape: vec![
            (configs::DimName::Batch, 1),
            (configs::DimName::Height, h),
            (configs::DimName::Width, w),
            (configs::DimName::NumAnchorsXFeatures, c),
        ],
        normalized: Some(true),
    }
}

/// ModelPack split decode of one 40×40 output with 3 anchors and 80 classes (40×40×3×85), `u8`
/// and `f32`.
fn bench_modelpack_split_40x40(suite: &mut BenchSuite) {
    let (h, w, nc) = (40, 40, 80);
    let n = h * w * 3 * (nc + 5);
    let mut rng = XorShift(0x0dd0_0dd0_0dd0_0dd0);
    let raw: Vec<u8> = (0..n).map(|_| (rng.next() >> 24) as u8).collect();
    let quant: configs::QuantTuple = (0.0855, 174).into();
    let floats: Vec<f32> = raw.iter().map(|&x| (x as f32 - 174.0) * 0.0855).collect();
    let shape = [1, h, w, 3 * (nc + 5)];
    let u8_tensor = TensorDyn::U8(Tensor::<u8>::from_slice(&raw, &shape).unwrap());
    let f32_tensor = TensorDyn::F32(Tensor::<f32>::from_slice(&floats, &shape).unwrap());
    for (name, quantization, tensor) in [
        (
            "decoder/modelpack/split_u8_40x40x255",
            Some(quant),
            &u8_tensor,
        ),
        ("decoder/modelpack/split_f32_40x40x255", None, &f32_tensor),
    ] {
        let decoder = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![modelpack_split_config(h, w, nc, quantization)])
            .with_score_threshold(0.25)
            .with_iou_threshold(0.45)
            .build()
            .expect("modelpack split decoder must build");
        let inputs: Vec<&TensorDyn> = vec![tensor];
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let mut output_boxes = Vec::with_capacity(50);
            let mut output_masks = Vec::with_capacity(50);
            decoder
                .decode(&inputs, &mut output_boxes, &mut output_masks)
                .unwrap();
            std::hint::black_box(output_boxes);
        });
        result.print_summary();
        suite.record(&result);
    }
}

/// ModelPack segmentation: decode of a `(1, 8, 240, 320)` NCHW output as `i8` (to `u8`) and
/// `f32` and `f16` (normalized to `u8`), and `segmentation_to_mask` (the per-pixel argmax) of
/// 240×320×8, 240×320×21 and 160×160×2 maps.
fn bench_modelpack_seg(suite: &mut BenchSuite) {
    let (c, h, w) = (8, 240, 320);
    let mut rng = XorShift(0x5e9_5e9);
    let raw: Vec<i8> = (0..c * h * w).map(|_| (rng.next() >> 24) as i8).collect();
    let floats: Vec<f32> = raw.iter().map(|&x| x as f32 * 0.05).collect();
    let shape = [1, c, h, w];
    let i8_tensor = TensorDyn::I8(Tensor::<i8>::from_slice(&raw, &shape).unwrap());
    let f32_tensor = TensorDyn::F32(Tensor::<f32>::from_slice(&floats, &shape).unwrap());
    let halves: Vec<half::f16> = floats.iter().map(|&x| half::f16::from_f32(x)).collect();
    let f16_tensor = TensorDyn::F16(Tensor::<half::f16>::from_slice(&halves, &shape).unwrap());
    for (name, quantization, tensor) in [
        (
            "decoder/modelpack/seg_i8_8x240x320",
            Some((0.05, 0).into()),
            &i8_tensor,
        ),
        ("decoder/modelpack/seg_f32_8x240x320", None, &f32_tensor),
        ("decoder/modelpack/seg_f16_8x240x320", None, &f16_tensor),
    ] {
        let decoder = DecoderBuilder::default()
            .with_config_modelpack_seg(configs::Segmentation {
                decoder: configs::DecoderType::ModelPack,
                quantization,
                shape: shape.to_vec(),
                dshape: vec![
                    (configs::DimName::Batch, 1),
                    (configs::DimName::NumClasses, c),
                    (configs::DimName::Height, h),
                    (configs::DimName::Width, w),
                ],
            })
            .build()
            .expect("modelpack seg decoder must build");
        let inputs: Vec<&TensorDyn> = vec![tensor];
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            let mut output_boxes = Vec::new();
            let mut output_masks = Vec::with_capacity(1);
            decoder
                .decode(&inputs, &mut output_boxes, &mut output_masks)
                .unwrap();
            std::hint::black_box(output_masks);
        });
        result.print_summary();
        suite.record(&result);
    }
    for (name, (mh, mw, mc)) in [
        ("decoder/modelpack/seg_to_mask_240x320x8", (240, 320, 8)),
        ("decoder/modelpack/seg_to_mask_240x320x21", (240, 320, 21)),
        ("decoder/modelpack/seg_to_mask_160x160x2", (160, 160, 2)),
    ] {
        let map = ndarray::Array3::from_shape_fn((mh, mw, mc), |_| (rng.next() >> 24) as u8);
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            std::hint::black_box(edgefirst_decoder::segmentation_to_mask(map.view()).unwrap());
        });
        result.print_summary();
        suite.record(&result);
    }
}

fn bench_masks_f32(suite: &mut BenchSuite) {
    // Pre-dequantize boxes + protos once outside the timed loop. The bench
    // measures the float seg-det decode + mask materialization; the legacy
    // version dequantized each iteration which masked the actual decode time.
    let boxes_raw = edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
    let boxes_raw =
        unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
    let boxes_arr = ndarray::Array2::from_shape_vec((116, 8400), boxes_raw.to_vec()).unwrap();
    let quant_boxes = Quantization {
        scale: 0.01948494464159012,
        zero_point: 20,
    };
    let boxes_f32 = dequantize_ndarray::<_, _, f32>(boxes_arr.view(), quant_boxes);
    let boxes_tensor = TensorDyn::F32(
        Tensor::<f32>::from_slice(boxes_f32.as_slice().unwrap(), &[1, 116, 8400]).unwrap(),
    );

    let protos_raw = edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
    let protos_raw =
        unsafe { std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len()) };
    let protos_arr = ndarray::Array3::from_shape_vec((160, 160, 32), protos_raw.to_vec()).unwrap();
    let quant_protos = Quantization {
        scale: 0.020889872685074806,
        zero_point: -115,
    };
    let protos_f32 = dequantize_ndarray::<_, _, f32>(protos_arr.view(), quant_protos);
    let protos_tensor = TensorDyn::F32(
        Tensor::<f32>::from_slice(protos_f32.as_slice().unwrap(), &[1, 160, 160, 32]).unwrap(),
    );

    let detection = configs::Detection {
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, 116, 8400],
        dshape: vec![],
        anchors: None,
        normalized: Some(true),
    };
    let protos_cfg = configs::Protos {
        decoder: configs::DecoderType::Ultralytics,
        quantization: None,
        shape: vec![1, 160, 160, 32],
        dshape: vec![],
    };
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.45)
        .with_iou_threshold(0.45)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(detection))
        .add_output(ConfigOutput::Protos(protos_cfg))
        .build()
        .expect("yolo segdet f32 decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&boxes_tensor, &protos_tensor];
    let result = run_bench("decoder/masks/f32", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        decoder
            .decode(&inputs, &mut output_boxes, &mut output_masks)
            .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_masks_i8(suite: &mut BenchSuite) {
    let boxes_raw = edgefirst_bench::testdata::read("yolov8_boxes_116x8400.bin");
    let boxes_raw =
        unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
    let boxes_tensor = TensorDyn::I8(Tensor::<i8>::from_slice(boxes_raw, &[1, 116, 8400]).unwrap());

    let protos_raw = edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
    let protos_raw =
        unsafe { std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len()) };
    let protos_tensor =
        TensorDyn::I8(Tensor::<i8>::from_slice(protos_raw, &[1, 160, 160, 32]).unwrap());

    let detection = configs::Detection {
        decoder: configs::DecoderType::Ultralytics,
        quantization: Some(configs::QuantTuple(0.01948494464159012, 20)),
        shape: vec![1, 116, 8400],
        dshape: vec![],
        anchors: None,
        normalized: Some(true),
    };
    let protos_cfg = configs::Protos {
        decoder: configs::DecoderType::Ultralytics,
        quantization: Some(configs::QuantTuple(0.020889872685074806, -115)),
        shape: vec![1, 160, 160, 32],
        dshape: vec![],
    };
    let decoder = DecoderBuilder::default()
        .with_score_threshold(0.45)
        .with_iou_threshold(0.45)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(detection))
        .add_output(ConfigOutput::Protos(protos_cfg))
        .build()
        .expect("yolo segdet i8 decoder must build");

    let inputs: Vec<&TensorDyn> = vec![&boxes_tensor, &protos_tensor];
    let result = run_bench("decoder/masks/i8", WARMUP, ITERATIONS, || {
        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        decoder
            .decode(&inputs, &mut output_boxes, &mut output_masks)
            .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
    result.print_summary();
    suite.record(&result);
}

/// Box quantization of the yolov8 seg fixtures.
const SEG_BOX_Q: (f32, i32) = (0.01948494464159012, 20);
/// Proto quantization of the yolov8 seg fixtures.
const SEG_PROTO_Q: (f32, i32) = (0.020889872685074806, -115);

/// A `[116, 8400]` int8 yolov8-seg output with exactly `n` detections that survive NMS: 0.2 x 0.2
/// boxes on a 10 x 10 grid (neighbour IoU 1/3), score 0.9, seeded random mask coefficients.
fn synthetic_seg_boxes(n: usize) -> Vec<i8> {
    let (scale, zp) = SEG_BOX_Q;
    let q = |v: f32| ((v / scale).round() as i32 + zp).clamp(-128, 127) as i8;
    let mut out = vec![-128i8; 116 * 8400];
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for j in 0..n.min(100) {
        let a = j * 83;
        let (xc, yc) = (0.05 + 0.1 * (j % 10) as f32, 0.05 + 0.1 * (j / 10) as f32);
        for (row, v) in [xc, yc, 0.2, 0.2].into_iter().enumerate() {
            out[row * 8400 + a] = q(v);
        }
        out[(4 + j % 80) * 8400 + a] = q(0.9);
        for c in 0..32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            out[(84 + c) * 8400 + a] = ((seed >> 32) % 121) as i8 - 60;
        }
    }
    out
}

/// A yolov8-seg decoder over `[1, 116, 8400]` boxes and `[1, 160, 160, 32]` protos.
fn seg_decoder(quantized: bool) -> edgefirst_decoder::Decoder {
    let quant = |q: (f32, i32)| quantized.then_some(configs::QuantTuple(q.0, q.1));
    DecoderBuilder::default()
        .with_score_threshold(0.45)
        .with_iou_threshold(0.45)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(configs::Detection {
            decoder: configs::DecoderType::Ultralytics,
            quantization: quant(SEG_BOX_Q),
            shape: vec![1, 116, 8400],
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        }))
        .add_output(ConfigOutput::Protos(configs::Protos {
            decoder: configs::DecoderType::Ultralytics,
            quantization: quant(SEG_PROTO_Q),
            shape: vec![1, 160, 160, 32],
            dshape: vec![],
        }))
        .build()
        .expect("yolo segdet decoder must build")
}

/// The yolov8 proto fixture, `[160, 160, 32]` int8.
fn seg_protos_i8() -> Vec<i8> {
    let raw = edgefirst_bench::testdata::read("yolov8_protos_160x160x32.bin");
    raw.iter().map(|&b| b as i8).collect()
}

/// `decode` masks for 10, 50 and 100 detections at 160x160 protos and K = 32, as i8, u8 and
/// f32 tensors, and `decode_proto` with i8 protos (kept raw) and u8 protos (dequantized).
fn bench_masks_counts(suite: &mut BenchSuite) {
    let protos = seg_protos_i8();
    let dq = |v: &[i8], (scale, zp): (f32, i32)| -> Vec<f32> {
        v.iter()
            .map(|&x| (f32::from(x) - zp as f32) * scale)
            .collect()
    };
    let to_u8 = |v: &[i8]| -> Vec<u8> { v.iter().map(|&x| (i16::from(x) + 128) as u8).collect() };
    let protos_i8 = TensorDyn::I8(Tensor::from_slice(&protos, &[1, 160, 160, 32]).unwrap());
    let protos_u8 = TensorDyn::U8(Tensor::from_slice(&to_u8(&protos), &[1, 160, 160, 32]).unwrap());
    let protos_f32 =
        TensorDyn::F32(Tensor::from_slice(&dq(&protos, SEG_PROTO_Q), &[1, 160, 160, 32]).unwrap());
    let shift = |(s, z): (f32, i32)| (s, z + 128);
    let dec_i8 = seg_decoder(true);
    let dec_f32 = seg_decoder(false);
    let dec_u8 = DecoderBuilder::default()
        .with_score_threshold(0.45)
        .with_iou_threshold(0.45)
        .with_nms(Some(Nms::ClassAgnostic))
        .add_output(ConfigOutput::Detection(configs::Detection {
            decoder: configs::DecoderType::Ultralytics,
            quantization: Some(configs::QuantTuple(SEG_BOX_Q.0, shift(SEG_BOX_Q).1)),
            shape: vec![1, 116, 8400],
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        }))
        .add_output(ConfigOutput::Protos(configs::Protos {
            decoder: configs::DecoderType::Ultralytics,
            quantization: Some(configs::QuantTuple(SEG_PROTO_Q.0, shift(SEG_PROTO_Q).1)),
            shape: vec![1, 160, 160, 32],
            dshape: vec![],
        }))
        .build()
        .expect("u8 yolo segdet decoder must build");
    for n in [10usize, 50, 100] {
        let raw = synthetic_seg_boxes(n);
        let boxes_i8 = TensorDyn::I8(Tensor::from_slice(&raw, &[1, 116, 8400]).unwrap());
        let boxes_u8 = TensorDyn::U8(Tensor::from_slice(&to_u8(&raw), &[1, 116, 8400]).unwrap());
        let boxes_f32 =
            TensorDyn::F32(Tensor::from_slice(&dq(&raw, SEG_BOX_Q), &[1, 116, 8400]).unwrap());
        let cases: [(&str, &edgefirst_decoder::Decoder, [&TensorDyn; 2]); 3] = [
            ("i8", &dec_i8, [&boxes_i8, &protos_i8]),
            ("u8", &dec_u8, [&boxes_u8, &protos_u8]),
            ("f32", &dec_f32, [&boxes_f32, &protos_f32]),
        ];
        for (label, decoder, inputs) in cases {
            let mut boxes = Vec::new();
            let mut masks = Vec::new();
            decoder.decode(&inputs, &mut boxes, &mut masks).unwrap();
            assert_eq!(
                boxes.len(),
                n,
                "{label}: synthetic detections must all survive"
            );
            let name = format!("decoder/masks/{label}/n{n}");
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                let mut output_boxes = Vec::with_capacity(n);
                let mut output_masks = Vec::with_capacity(n);
                decoder
                    .decode(&inputs, &mut output_boxes, &mut output_masks)
                    .unwrap();
                std::hint::black_box((output_boxes, output_masks));
            });
            result.print_summary();
            suite.record(&result);
        }
        if n != 50 {
            continue;
        }
        for (label, decoder, inputs) in [
            ("i8", &dec_i8, [&boxes_i8, &protos_i8]),
            ("u8", &dec_u8, [&boxes_u8, &protos_u8]),
        ] {
            let name = format!("decoder/masks/decode_proto/{label}/n{n}");
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                let mut output_boxes = Vec::with_capacity(n);
                let proto = decoder.decode_proto(&inputs, &mut output_boxes).unwrap();
                std::hint::black_box((output_boxes, proto));
            });
            result.print_summary();
            suite.record(&result);
        }
    }
}

/// Build a per-scale-capable decoder for benchmarking, plus the 6
/// zero-int8 input tensors (3 box levels + 3 score levels) with
/// attached quantization.
///
/// Schema: yolov8n-style 3-level DFL detection-only (no mc, no
/// protos). Detection-only avoids the legacy validator's mc/protos
/// dshape requirements while still exercising the full DFL kernel
/// path on a realistic 8400-anchor workload.
fn build_dfl_per_scale_bench_inputs(
    out_dtype: DecodeDtype,
    nchw: bool,
) -> (edgefirst_decoder::Decoder, Vec<TensorDyn>) {
    let schema_json = r#"{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, 640, 640, 3],
            "dshape": [{"batch": 1}, {"height": 640}, {"width": 640}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 8400],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}],
                "encoding": "dfl", "decoder": "ultralytics", "normalized": true,
                "outputs": [
                    {"name": "boxes_0", "type": "boxes", "stride": 8, "scale_index": 0,
                     "shape": [1, 80, 80, 64], "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_features": 64}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "boxes_1", "type": "boxes", "stride": 16, "scale_index": 1,
                     "shape": [1, 40, 40, 64], "dshape": [{"batch": 1}, {"height": 40}, {"width": 40}, {"num_features": 64}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "boxes_2", "type": "boxes", "stride": 32, "scale_index": 2,
                     "shape": [1, 20, 20, 64], "dshape": [{"batch": 1}, {"height": 20}, {"width": 20}, {"num_features": 64}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}}
                ]
            },
            {
                "name": "scores", "type": "scores",
                "shape": [1, 80, 8400],
                "dshape": [{"batch": 1}, {"num_classes": 80}, {"num_boxes": 8400}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [
                    {"name": "scores_0", "type": "scores", "stride": 8, "scale_index": 0,
                     "shape": [1, 80, 80, 80], "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "scores_1", "type": "scores", "stride": 16, "scale_index": 1,
                     "shape": [1, 40, 40, 80], "dshape": [{"batch": 1}, {"height": 40}, {"width": 40}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "scores_2", "type": "scores", "stride": 32, "scale_index": 2,
                     "shape": [1, 20, 20, 80], "dshape": [{"batch": 1}, {"height": 20}, {"width": 20}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}}
                ]
            }
        ]
    }"#;
    let mut schema: serde_json::Value = serde_json::from_str(schema_json).expect("schema parse");
    let mut shapes: Vec<Vec<usize>> = vec![
        vec![1, 80, 80, 64],
        vec![1, 80, 80, 80],
        vec![1, 40, 40, 64],
        vec![1, 40, 40, 80],
        vec![1, 20, 20, 64],
        vec![1, 20, 20, 80],
    ];
    if nchw {
        // Every child `[1, H, W, C]` becomes `[1, C, H, W]`, its dshape reordered to match.
        for logical in schema["outputs"].as_array_mut().unwrap() {
            for child in logical["outputs"].as_array_mut().unwrap() {
                for key in ["shape", "dshape"] {
                    let v = child[key].as_array_mut().unwrap();
                    v[1..].rotate_right(1);
                }
            }
        }
        for s in &mut shapes {
            s[1..].rotate_right(1);
        }
    }
    let schema: SchemaV2 = serde_json::from_value(schema).expect("schema parse");

    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(out_dtype)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.25)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("decoder build");

    let tensors: Vec<TensorDyn> = shapes
        .iter()
        .map(|s| {
            let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            TensorDyn::I8(t)
        })
        .collect();

    (decoder, tensors)
}

/// Build LTRB variant of the same bench setup. Detection-only, 3 levels,
/// 4-channel boxes (yolo26 style).
fn build_ltrb_per_scale_bench_inputs(
    out_dtype: DecodeDtype,
) -> (edgefirst_decoder::Decoder, Vec<TensorDyn>) {
    let schema_json = r#"{
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, 640, 640, 3],
            "dshape": [{"batch": 1}, {"height": 640}, {"width": 640}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes",
                "shape": [1, 4, 8400],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}],
                "encoding": "ltrb", "decoder": "ultralytics", "normalized": true,
                "outputs": [
                    {"name": "boxes_0", "type": "boxes", "stride": 8, "scale_index": 0,
                     "shape": [1, 80, 80, 4], "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"box_coords": 4}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "boxes_1", "type": "boxes", "stride": 16, "scale_index": 1,
                     "shape": [1, 40, 40, 4], "dshape": [{"batch": 1}, {"height": 40}, {"width": 40}, {"box_coords": 4}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "boxes_2", "type": "boxes", "stride": 32, "scale_index": 2,
                     "shape": [1, 20, 20, 4], "dshape": [{"batch": 1}, {"height": 20}, {"width": 20}, {"box_coords": 4}],
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}}
                ]
            },
            {
                "name": "scores", "type": "scores",
                "shape": [1, 80, 8400],
                "dshape": [{"batch": 1}, {"num_classes": 80}, {"num_boxes": 8400}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": [
                    {"name": "scores_0", "type": "scores", "stride": 8, "scale_index": 0,
                     "shape": [1, 80, 80, 80], "dshape": [{"batch": 1}, {"height": 80}, {"width": 80}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "scores_1", "type": "scores", "stride": 16, "scale_index": 1,
                     "shape": [1, 40, 40, 80], "dshape": [{"batch": 1}, {"height": 40}, {"width": 40}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}},
                    {"name": "scores_2", "type": "scores", "stride": 32, "scale_index": 2,
                     "shape": [1, 20, 20, 80], "dshape": [{"batch": 1}, {"height": 20}, {"width": 20}, {"num_classes": 80}],
                     "activation_required": "sigmoid",
                     "dtype": "int8", "quantization": {"scale": 0.1, "zero_point": 0, "dtype": "int8"}}
                ]
            }
        ]
    }"#;
    let schema: SchemaV2 = serde_json::from_str(schema_json).expect("schema parse");

    let decoder = DecoderBuilder::default()
        .with_schema(schema)
        .with_decode_dtype(out_dtype)
        .with_iou_threshold(0.7)
        .with_score_threshold(0.25)
        .with_nms(Some(Nms::ClassAgnostic))
        .build()
        .expect("decoder build");

    let shapes: [&[usize]; 6] = [
        &[1, 80, 80, 4],
        &[1, 80, 80, 80],
        &[1, 40, 40, 4],
        &[1, 40, 40, 80],
        &[1, 20, 20, 4],
        &[1, 20, 20, 80],
    ];
    let tensors: Vec<TensorDyn> = shapes
        .iter()
        .map(|s| {
            let mut t = Tensor::<i8>::new(s, Some(TensorMemory::Mem), None).unwrap();
            t.set_quantization(TQ::per_tensor(0.1, 0)).unwrap();
            TensorDyn::I8(t)
        })
        .collect();

    (decoder, tensors)
}

fn bench_per_scale_dfl_i8_to_f32(suite: &mut BenchSuite) {
    let (decoder, tensors) = build_dfl_per_scale_bench_inputs(DecodeDtype::F32, false);
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();

    let result = run_bench(
        "decoder/per_scale/dfl_i8_to_f32",
        WARMUP,
        ITERATIONS,
        || {
            let mut output_boxes = Vec::with_capacity(50);
            let mut masks = Vec::new();
            decoder
                .decode(&inputs, &mut output_boxes, &mut masks)
                .unwrap();
        },
    );
    result.print_summary();
    suite.record(&result);
}

/// [`bench_per_scale_dfl_i8_to_f32`] with every input NCHW, so each level is transposed first.
fn bench_per_scale_dfl_i8_to_f32_nchw(suite: &mut BenchSuite) {
    let (decoder, tensors) = build_dfl_per_scale_bench_inputs(DecodeDtype::F32, true);
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();

    let result = run_bench(
        "decoder/per_scale/dfl_i8_to_f32_nchw",
        WARMUP,
        ITERATIONS,
        || {
            let mut output_boxes = Vec::with_capacity(50);
            let mut masks = Vec::new();
            decoder
                .decode(&inputs, &mut output_boxes, &mut masks)
                .unwrap();
        },
    );
    result.print_summary();
    suite.record(&result);
}

fn bench_per_scale_dfl_i8_to_f16(suite: &mut BenchSuite) {
    let (decoder, tensors) = build_dfl_per_scale_bench_inputs(DecodeDtype::F16, false);
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();

    let result = run_bench(
        "decoder/per_scale/dfl_i8_to_f16",
        WARMUP,
        ITERATIONS,
        || {
            let mut output_boxes = Vec::with_capacity(50);
            let mut masks = Vec::new();
            decoder
                .decode(&inputs, &mut output_boxes, &mut masks)
                .unwrap();
        },
    );
    result.print_summary();
    suite.record(&result);
}

fn bench_per_scale_ltrb_i8_to_f32(suite: &mut BenchSuite) {
    let (decoder, tensors) = build_ltrb_per_scale_bench_inputs(DecodeDtype::F32);
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();

    let result = run_bench(
        "decoder/per_scale/ltrb_i8_to_f32",
        WARMUP,
        ITERATIONS,
        || {
            let mut output_boxes = Vec::with_capacity(50);
            let mut masks = Vec::new();
            decoder
                .decode(&inputs, &mut output_boxes, &mut masks)
                .unwrap();
        },
    );
    result.print_summary();
    suite.record(&result);
}

/// Process CPU time (all threads) consumed so far, where the platform
/// reports it.
#[cfg(unix)]
fn process_cpu_time() -> Option<std::time::Duration> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer; the clock id is a constant
    // every supported Unix defines.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    (rc == 0).then(|| std::time::Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
}

#[cfg(not(unix))]
fn process_cpu_time() -> Option<std::time::Duration> {
    None
}

/// `run_bench`, also printing the process CPU time per iteration, which
/// counts rayon workers (including their idle spinning) alongside the
/// calling thread.
fn run_bench_cpu<F: FnMut()>(
    suite: &mut BenchSuite,
    name: &str,
    warmup: usize,
    iterations: usize,
    f: F,
) {
    let cpu0 = process_cpu_time();
    let result = run_bench(name, warmup, iterations, f);
    let cpu1 = process_cpu_time();
    result.print_summary();
    if let (Some(cpu0), Some(cpu1)) = (cpu0, cpu1) {
        let cpu = (cpu1 - cpu0) / (warmup + iterations) as u32;
        println!("  {:50} cpu/iter={cpu:>10.1?}", "");
    }
    suite.record(&result);
}

/// Deterministic xorshift generator for synthetic model outputs.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// Per-scale int8 DFL detection model with a `width x height` input, three
/// levels at strides 8/16/32 and `nc` classes, as the EdgeFirst converter
/// declares it. Quantization matches captures of yolov8n smart exports on
/// the i.MX 95 (boxes about 0.12-0.15 per step, scores about 0.2).
fn realistic_per_scale_schema(width: usize, height: usize, nc: usize) -> SchemaV2 {
    serde_json::from_value(realistic_per_scale_json(width, height, nc, false))
        .expect("schema parse")
}

/// [`realistic_per_scale_schema`] as JSON; `seg` adds yolov8-seg mask
/// coefficients (32 per anchor) and 160x160x32 protos.
fn realistic_per_scale_json(
    width: usize,
    height: usize,
    nc: usize,
    seg: bool,
) -> serde_json::Value {
    let strides = [8usize, 16, 32];
    let total: usize = strides.iter().map(|s| (width / s) * (height / s)).sum();
    let child = |role: &str, i: usize, s: usize, c: usize, q: (f32, i32)| {
        let (h, w) = (height / s, width / s);
        let last = if role == "scores" {
            serde_json::json!({"num_classes": c})
        } else {
            serde_json::json!({"num_features": c})
        };
        let mut v = serde_json::json!({
            "name": format!("{role}_{i}"), "type": role, "stride": s, "scale_index": i,
            "shape": [1, h, w, c],
            "dshape": [{"batch": 1}, {"height": h}, {"width": w}, last],
            "dtype": "int8",
            "quantization": {"scale": q.0, "zero_point": q.1, "dtype": "int8"}
        });
        if role == "scores" {
            v["activation_required"] = "sigmoid".into();
        }
        v
    };
    let boxes: Vec<_> = strides
        .iter()
        .enumerate()
        .map(|(i, &s)| child("boxes", i, s, 64, (0.147, 8)))
        .collect();
    let scores: Vec<_> = strides
        .iter()
        .enumerate()
        .map(|(i, &s)| child("scores", i, s, nc, (0.2, 110)))
        .collect();
    let mut schema = serde_json::json!({
        "schema_version": 2,
        "nms": "class_agnostic",
        "input": {
            "shape": [1, height, width, 3],
            "dshape": [{"batch": 1}, {"height": height}, {"width": width}, {"num_features": 3}],
            "cameraadaptor": "rgb"
        },
        "outputs": [
            {
                "name": "boxes", "type": "boxes", "shape": [1, 4, total],
                "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": total}],
                "encoding": "dfl", "decoder": "ultralytics", "normalized": true,
                "outputs": boxes
            },
            {
                "name": "scores", "type": "scores", "shape": [1, nc, total],
                "dshape": [{"batch": 1}, {"num_classes": nc}, {"num_boxes": total}],
                "score_format": "per_class", "decoder": "ultralytics",
                "outputs": scores
            }
        ]
    });
    if seg {
        let mc: Vec<_> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| child("mask_coefs", i, s, 32, (0.0312, 4)))
            .collect();
        let (ph, pw) = (height / 4, width / 4);
        let outputs = schema["outputs"].as_array_mut().expect("outputs");
        outputs.push(serde_json::json!({
            "name": "mask_coefs", "type": "mask_coefs", "shape": [1, 32, total],
            "dshape": [{"batch": 1}, {"num_features": 32}, {"num_boxes": total}],
            "decoder": "ultralytics",
            "outputs": mc
        }));
        outputs.push(serde_json::json!({
            "name": "protos", "type": "protos", "shape": [1, ph, pw, 32],
            "dshape": [{"batch": 1}, {"height": ph}, {"width": pw}, {"num_protos": 32}],
            "stride": 4, "dtype": "int8",
            "quantization": {"scale": 0.0203, "zero_point": -83, "dtype": "int8"}
        }));
    }
    schema
}

/// Synthetic int8 outputs for [`realistic_per_scale_schema`], shaped like
/// captured yolov8n frames: each DFL side peaks at one bin and falls off
/// by about 0.4-1.2 logits per bin, about 0.3% of anchors carry a class
/// above 0.25, another 2% sit between about 0.05 and 0.25, and the rest
/// score below 0.02.
fn realistic_per_scale_inputs(schema: &SchemaV2) -> Vec<TensorDyn> {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut tensors = Vec::new();
    for logical in &schema.outputs {
        use edgefirst_decoder::schema::LogicalType;
        let role = match logical.type_ {
            Some(LogicalType::Boxes) => "boxes",
            Some(LogicalType::Scores) => "scores",
            Some(LogicalType::MaskCoefs) => "mask_coefs",
            Some(LogicalType::Protos) => {
                let mut t =
                    Tensor::<i8>::new(&logical.shape, Some(TensorMemory::Mem), None).unwrap();
                for b in t.map().unwrap().as_mut_slice() {
                    *b = rng.next() as i8;
                }
                t.set_quantization(TQ::per_tensor(0.0203, -83)).unwrap();
                tensors.push(TensorDyn::I8(t));
                continue;
            }
            _ => continue,
        };
        for child in &logical.outputs {
            let shape = child.shape.clone();
            let c = shape[3];
            let mut t = Tensor::<i8>::new(&shape, Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                let v = m.as_mut_slice();
                for row in v.chunks_exact_mut(c) {
                    if role == "boxes" {
                        for side in row.chunks_exact_mut(16) {
                            let peak = rng.below(16) as i32;
                            let top = 20 + rng.below(50) as i32;
                            let fall = 3 + rng.below(6) as i32;
                            for (i, b) in side.iter_mut().enumerate() {
                                let noise = rng.below(9) as i32 - 4;
                                let x = top - fall * (i as i32 - peak).abs() + noise;
                                *b = x.clamp(-128, 127) as i8;
                            }
                        }
                    } else if role == "mask_coefs" {
                        for b in row.iter_mut() {
                            *b = (rng.below(97) as i32 - 48) as i8;
                        }
                    } else {
                        for b in row.iter_mut() {
                            *b = (40 + rng.below(50)) as i8;
                        }
                        let r = rng.below(1000);
                        let class = rng.below(c as u32) as usize;
                        if r < 3 {
                            row[class] = (106 + rng.below(22)) as i8;
                        } else if r < 23 {
                            row[class] = (96 + rng.below(9)) as i8;
                        }
                    }
                }
            }
            let q = child.quantization.as_ref().expect("child quantization");
            t.set_quantization(TQ::per_tensor(
                q.scale[0],
                q.zero_point.as_ref().map_or(0, |z| z[0]),
            ))
            .unwrap();
            tensors.push(TensorDyn::I8(t));
        }
    }
    tensors
}

/// The per-scale DFL pipeline on realistic int8 outputs, for the 960x544
/// four-class model (levels of 8160/2040/510 anchors, none a multiple of
/// the prefilter's pack) and the 640x640 COCO model. `dense` decodes every
/// anchor; `prefilter` passes the 0.25 score threshold the way `decode`
/// does; `decode` adds the candidate filter and NMS.
fn bench_per_scale_realistic(suite: &mut BenchSuite) {
    for (label, width, height, nc) in [("960x544", 960, 544, 4), ("640x640", 640, 640, 80)] {
        let schema = realistic_per_scale_schema(width, height, nc);
        let tensors = realistic_per_scale_inputs(&schema);
        let inputs: Vec<&TensorDyn> = tensors.iter().collect();
        let decoder = DecoderBuilder::default()
            .with_schema(schema)
            .with_decode_dtype(DecodeDtype::F32)
            .with_iou_threshold(0.7)
            .with_score_threshold(0.25)
            .with_nms(Some(Nms::ClassAgnostic))
            .build()
            .expect("decoder build");

        let kept = decoder._bench_run_per_scale(&inputs, Some(0.25)).unwrap();
        let mut boxes = Vec::new();
        decoder
            .decode(&inputs, &mut boxes, &mut Vec::new())
            .unwrap();
        println!(
            "  per_scale {label}: prefilter keeps {kept} anchors, {} detections",
            boxes.len()
        );

        for (mode, threshold) in [("dense", None), ("prefilter", Some(0.25))] {
            run_bench_cpu(
                suite,
                &format!("decoder/per_scale/{label}/{mode}"),
                WARMUP,
                ITERATIONS * 3,
                || {
                    decoder._bench_run_per_scale(&inputs, threshold).unwrap();
                },
            );
        }
        run_bench_cpu(
            suite,
            &format!("decoder/per_scale/{label}/decode"),
            WARMUP,
            ITERATIONS * 3,
            || {
                let mut output_boxes = Vec::with_capacity(50);
                decoder
                    .decode(&inputs, &mut output_boxes, &mut Vec::new())
                    .unwrap();
            },
        );
    }
}

/// The 640x640 COCO yolov8-seg model through `decode_proto` with f32 and
/// f16 outputs: every per-scale role, the prefilter, and the f16 widen.
fn bench_per_scale_seg(suite: &mut BenchSuite) {
    let schema: SchemaV2 =
        serde_json::from_value(realistic_per_scale_json(640, 640, 80, true)).expect("schema");
    let tensors = realistic_per_scale_inputs(&schema);
    let inputs: Vec<&TensorDyn> = tensors.iter().collect();
    for (label, dtype) in [("f32", DecodeDtype::F32), ("f16", DecodeDtype::F16)] {
        let decoder = DecoderBuilder::default()
            .with_schema(schema.clone())
            .with_decode_dtype(dtype)
            .with_iou_threshold(0.7)
            .with_score_threshold(0.25)
            .with_nms(Some(Nms::ClassAgnostic))
            .build()
            .expect("decoder build");
        let mut boxes = Vec::new();
        decoder.decode_proto(&inputs, &mut boxes).unwrap();
        println!(
            "  per_scale 640x640_seg {label}: {} detections",
            boxes.len()
        );
        run_bench_cpu(
            suite,
            &format!("decoder/per_scale/640x640_seg/{label}/decode_proto"),
            WARMUP,
            ITERATIONS,
            || {
                let mut output_boxes = Vec::with_capacity(50);
                decoder.decode_proto(&inputs, &mut output_boxes).unwrap();
            },
        );
    }
}

/// Tiers a cell bench runs on, as `EDGEFIRST_DECODER_FORCE_KERNEL` names.
const CELL_TIERS: [&str; 3] = ["scalar", "neon", "neon_fp16"];

/// Synthetic raw values of one dtype: logits in about `lo..hi` under `q`.
fn cell_input(
    dtype: edgefirst_tensor::DType,
    n: usize,
    lo: f32,
    hi: f32,
    q: Quantization,
) -> CellData {
    use edgefirst_tensor::DType;
    let mut rng = XorShift(0x5DEE_CE66_D1CE_4E5B);
    let mut x = || lo + (hi - lo) * (rng.next() % 65536) as f32 / 65536.0;
    let raw = |v: f32| (v / q.scale + q.zero_point as f32).round();
    match dtype {
        DType::I8 => CellData::I8(
            (0..n)
                .map(|_| raw(x()).clamp(-128.0, 127.0) as i8)
                .collect(),
        ),
        DType::U8 => CellData::U8((0..n).map(|_| raw(x()).clamp(0.0, 255.0) as u8).collect()),
        DType::I16 => CellData::I16(
            (0..n)
                .map(|_| raw(x()).clamp(-32768.0, 32767.0) as i16)
                .collect(),
        ),
        DType::U16 => CellData::U16(
            (0..n)
                .map(|_| raw(x()).clamp(0.0, 65535.0) as u16)
                .collect(),
        ),
        DType::F16 => CellData::F16((0..n).map(|_| half::f16::from_f32(x())).collect()),
        _ => CellData::F32((0..n).map(|_| x()).collect()),
    }
}

enum CellData {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F16(Vec<half::f16>),
    F32(Vec<f32>),
}

impl CellData {
    fn src(&self) -> edgefirst_decoder::per_scale::bench::Src<'_> {
        use edgefirst_decoder::per_scale::bench::Src;
        match self {
            CellData::I8(v) => Src::I8(v),
            CellData::U8(v) => Src::U8(v),
            CellData::I16(v) => Src::I16(v),
            CellData::U16(v) => Src::U16(v),
            CellData::F16(v) => Src::F16(v),
            CellData::F32(v) => Src::F32(v),
        }
    }
}

/// Every score, mask-coefficient and proto cell the audit flagged, per tier:
/// 8400 anchors x 80 classes of scores, x 32 mask coefficients, and the
/// 160x160x32 protos.
fn bench_per_scale_cells(suite: &mut BenchSuite) {
    use edgefirst_decoder::per_scale::bench::{Cell, CellKind, Dst};
    use edgefirst_tensor::DType;
    let int_q = |dt: DType| match dt {
        DType::I16 | DType::U16 => {
            Quantization::new(0.0007, if dt == DType::U16 { 32768 } else { 0 })
        }
        DType::U8 => Quantization::new(0.1, 160),
        DType::I8 => Quantization::new(0.1, 32),
        _ => Quantization::identity(),
    };
    let (f, h) = (DecodeDtype::F32, DecodeDtype::F16);
    // (label, kind, anchors, channels, logit range, (input, output) pairs)
    type Case<'a> = (
        &'a str,
        CellKind,
        usize,
        usize,
        f32,
        f32,
        &'a [(DType, DecodeDtype)],
    );
    let cases: &[Case] = &[
        (
            "score_sigmoid",
            CellKind::ScoreSigmoid,
            8400,
            80,
            -16.0,
            8.0,
            &[
                (DType::I8, f),
                (DType::I8, h),
                (DType::U8, h),
                (DType::I16, f),
                (DType::I16, h),
                (DType::U16, h),
                (DType::F16, f),
                (DType::F16, h),
                (DType::F32, f),
                (DType::F32, h),
            ],
        ),
        (
            "score",
            CellKind::Score,
            8400,
            80,
            -16.0,
            8.0,
            &[(DType::I8, h), (DType::F16, f)],
        ),
        (
            "mask_coefs",
            CellKind::MaskCoefs,
            8400,
            32,
            -2.0,
            2.0,
            &[
                (DType::I8, f),
                (DType::I8, h),
                (DType::I16, h),
                (DType::F16, f),
                (DType::F16, h),
                (DType::F32, h),
            ],
        ),
        (
            "protos",
            CellKind::Protos,
            25600,
            32,
            -1.0,
            3.0,
            &[
                (DType::I8, f),
                (DType::I8, h),
                (DType::F16, f),
                (DType::F16, h),
                (DType::F32, h),
            ],
        ),
    ];
    for &(label, kind, anchors, channels, lo, hi, pairs) in cases {
        let n = anchors * channels;
        for &(input, output) in pairs {
            let q = int_q(input);
            let data = cell_input(input, n, lo, hi, q);
            let mut d32 = vec![0f32; n];
            let mut d16 = vec![half::f16::ZERO; n];
            let pair = format!(
                "{}_{}",
                format!("{input:?}").to_lowercase(),
                format!("{output:?}").to_lowercase()
            );
            for tier in CELL_TIERS {
                let Ok(cell) = Cell::new(kind, input, output, tier, anchors, channels) else {
                    continue;
                };
                let name = format!("decoder/per_scale/cell/{label}/{pair}/{tier}");
                let result = run_bench(&name, WARMUP, ITERATIONS, || {
                    let dst = match output {
                        DecodeDtype::F32 => Dst::F32(&mut d32),
                        DecodeDtype::F16 => Dst::F16(&mut d16),
                    };
                    cell.run(data.src(), q, dst).unwrap();
                });
                result.print_summary();
                println!("  {:50} kernel={}", "", cell.kernel());
                suite.record(&result);
            }
        }
    }
}

/// Every per-scale box cell per tier over one 105x80 level (8400 anchors): DFL `reg_max` 16
/// per-tensor and per-channel, DFL `reg_max` 8, and LTRB.
fn bench_per_scale_box_cells(suite: &mut BenchSuite) {
    use edgefirst_decoder::per_scale::bench::{BoxCell, BoxKind, Dst};
    use edgefirst_tensor::DType;
    let (h, w) = (105usize, 80usize);
    let anchors = h * w;
    let int_q = |dt: DType| match dt {
        DType::I16 => Quantization::new(0.0007, 0),
        DType::U16 => Quantization::new(0.0007, 32768),
        DType::U8 => Quantization::new(0.1, 160),
        DType::I8 => Quantization::new(0.1, 32),
        _ => Quantization::identity(),
    };
    let (f, hf) = (DecodeDtype::F32, DecodeDtype::F16);
    let dfl: &[(DType, DecodeDtype)] = &[
        (DType::I8, f),
        (DType::I8, hf),
        (DType::I16, f),
        (DType::I16, hf),
        (DType::U16, hf),
        (DType::F16, f),
        (DType::F16, hf),
        (DType::F32, f),
        (DType::F32, hf),
    ];
    let per_channel: &[(DType, DecodeDtype)] = &[
        (DType::I8, f),
        (DType::I8, hf),
        (DType::U8, f),
        (DType::I16, f),
        (DType::I16, hf),
    ];
    let rm8: &[(DType, DecodeDtype)] = &[(DType::I8, f), (DType::I8, hf)];
    let ltrb: &[(DType, DecodeDtype)] = &[
        (DType::I8, f),
        (DType::I8, hf),
        (DType::U8, f),
        (DType::I16, f),
        (DType::I16, hf),
        (DType::F16, f),
        (DType::F16, hf),
        (DType::F32, f),
        (DType::F32, hf),
    ];
    // (label, kind, reg_max, logit range, pairs)
    type Case<'a> = (
        &'a str,
        BoxKind,
        usize,
        f32,
        f32,
        &'a [(DType, DecodeDtype)],
    );
    let cases: &[Case] = &[
        ("dfl", BoxKind::Dfl, 16, -8.0, 8.0, dfl),
        (
            "dfl_per_channel",
            BoxKind::DflPerChannel,
            16,
            -8.0,
            8.0,
            per_channel,
        ),
        ("dfl_rm8", BoxKind::Dfl, 8, -8.0, 8.0, rm8),
        ("ltrb", BoxKind::Ltrb, 1, 0.0, 16.0, ltrb),
    ];
    for &(label, kind, reg_max, lo, hi, pairs) in cases {
        let channels = if kind == BoxKind::Ltrb {
            4
        } else {
            4 * reg_max
        };
        let n = anchors * channels;
        let mut d32 = vec![0f32; anchors * 4];
        let mut d16 = vec![half::f16::ZERO; anchors * 4];
        for &(input, output) in pairs {
            let q = int_q(input);
            let data = cell_input(input, n, lo, hi, q);
            let scales: Vec<f32> = (0..channels)
                .map(|c| q.scale * (0.75 + (c % 7) as f32 * 0.1))
                .collect();
            let zps: Vec<i32> = (0..channels)
                .map(|c| q.zero_point + (c % 5) as i32 - 2)
                .collect();
            let pair = format!(
                "{}_{}",
                format!("{input:?}").to_lowercase(),
                format!("{output:?}").to_lowercase()
            );
            for tier in CELL_TIERS {
                let Ok(cell) = BoxCell::new(kind, input, output, tier, h, w, reg_max, q.scale)
                else {
                    continue;
                };
                let name = format!("decoder/per_scale/box/{label}/{pair}/{tier}");
                let result = run_bench(&name, WARMUP, ITERATIONS, || {
                    let dst = match output {
                        DecodeDtype::F32 => Dst::F32(&mut d32),
                        DecodeDtype::F16 => Dst::F16(&mut d16),
                    };
                    cell.run(data.src(), q, (&scales, &zps), dst).unwrap();
                });
                result.print_summary();
                println!("  {:50} kernel={}", "", cell.kernel());
                suite.record(&result);
            }
        }
    }
}

/// The prefilter's anchor selection on realistic raw scores: the 640x640
/// COCO level-0 layout (6400 anchors x 80 classes) and the 960x544 one
/// (8160 x 4), at the floor of a 0.25 threshold.
fn bench_per_scale_select(suite: &mut BenchSuite) {
    use edgefirst_decoder::per_scale::bench::{select_anchors, Src, Tier};
    for (label, anchors, nc) in [("6400x80", 6400usize, 80usize), ("8160x4", 8160, 4)] {
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        let mut i8s = vec![0i8; anchors * nc];
        for row in i8s.chunks_exact_mut(nc) {
            for b in row.iter_mut() {
                *b = (40 + rng.below(50)) as i8;
            }
            let r = rng.below(1000);
            let class = rng.below(nc as u32) as usize;
            if r < 3 {
                row[class] = (106 + rng.below(22)) as i8;
            } else if r < 23 {
                row[class] = (96 + rng.below(9)) as i8;
            }
        }
        let i16s: Vec<i16> = i8s.iter().map(|&v| i16::from(v) * 256).collect();
        let mut kept = Vec::with_capacity(anchors);
        for (dt, src, floor) in [
            ("i8", Src::I8(&i8s), 105),
            ("i16", Src::I16(&i16s), 105 * 256),
        ] {
            for tier_name in ["scalar", "neon"] {
                let Ok(tier) = Tier::new(tier_name) else {
                    continue;
                };
                let name = format!("decoder/per_scale/select_anchors/{label}/{dt}/{tier_name}");
                let result = run_bench(&name, WARMUP, ITERATIONS * 3, || {
                    select_anchors(&tier, src, nc, floor, &mut kept).unwrap();
                });
                result.print_summary();
                println!("  {:50} kept={}", "", kept.len());
                suite.record(&result);
            }
        }
    }
}

/// One 120x68 level (the largest of a 960x544 input) of realistic DFL bins
/// through each 8-bit DFL box kernel, reported per anchor.
fn bench_per_scale_dfl_kernels(suite: &mut BenchSuite) {
    use edgefirst_decoder::per_scale::bench::{DflKernel, DflLevel};
    let level = DflLevel::new(120, 68, 8.0, 0.147, 8);
    let n = level.anchors();
    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
    let mut input = vec![0i8; n * 64];
    for side in input.chunks_exact_mut(16) {
        let peak = rng.below(16) as i32;
        let top = 20 + rng.below(50) as i32;
        let fall = 3 + rng.below(6) as i32;
        for (i, b) in side.iter_mut().enumerate() {
            let noise = rng.below(9) as i32 - 4;
            *b = (top - fall * (i as i32 - peak).abs() + noise).clamp(-128, 127) as i8;
        }
    }
    let mut reference = vec![0f32; n * 4];
    level
        .decode(DflKernel::Softmax, &input, &mut reference)
        .unwrap();
    let mut dst = vec![0f32; n * 4];
    for kernel in DflKernel::available() {
        let name = format!("decoder/per_scale/dfl_kernel/{kernel:?}");
        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            level.decode(kernel, &input, &mut dst).unwrap();
        });
        let err = dst
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        result.print_summary();
        println!(
            "  {:50} ns/anchor={:>7.1}  max |Δ| vs Softmax={err:.2e} px",
            "",
            result.median().as_nanos() as f64 / n as f64
        );
        suite.record(&result);
    }
}

/// SegFormer-B1-shaped logits (150 classes, 128x168, NCHW) over a Voronoi
/// class field. Calibrated to SegFormer-B1 on ADE20K val: about 7% of
/// low-resolution cells straddle a class boundary, where a few classes score
/// close to the winner (near -2) and the rest sit far below.
fn semantic_logits() -> Vec<f32> {
    const C: usize = 150;
    const H: usize = 128;
    const W: usize = 168;
    let mut seed = 0x2164u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let sites: Vec<(f32, f32, usize)> = (0..24)
        .map(|_| {
            let y = (next() % H as u32) as f32;
            let x = (next() % W as u32) as f32;
            (y, x, next() as usize % C)
        })
        .collect();
    let mut v = vec![0f32; C * H * W];
    let mut dist = [0f32; 24];
    for y in 0..H {
        for x in 0..W {
            for (d, s) in dist.iter_mut().zip(&sites) {
                *d = ((s.0 - y as f32).powi(2) + (s.1 - x as f32).powi(2)).sqrt();
            }
            let nearest = dist.iter().copied().fold(f32::INFINITY, f32::min);
            for c in 0..C {
                let u = (next() % 1024) as f32 / 1024.0;
                v[(c * H + y) * W + x] = u * 12.0 - 30.0;
            }
            // A site's class fades with its distance beyond the nearest site.
            for (d, s) in dist.iter().zip(&sites) {
                let i = (s.2 * H + y) * W + x;
                v[i] = v[i].max(-2.0 - 1.5 * (d - nearest));
            }
        }
    }
    v
}

fn bench_semantic(suite: &mut BenchSuite) {
    use edgefirst_decoder::schema::DecodeMode;
    use edgefirst_decoder::semantic::{decode_semantic, SemanticConfig, SemanticOptions};
    use edgefirst_tensor::DType;

    let shape = [1, 150, 128, 168];
    let logits = semantic_logits();
    let f32_nchw = TensorDyn::F32(Tensor::<f32>::from_slice(&logits, &shape).unwrap());
    // Logits span [-30, -1]: scale 0.125 with zero point 112 fits i8.
    let q: Vec<i8> = logits
        .iter()
        .map(|&x| (x * 8.0 + 112.0).round().clamp(-128.0, 127.0) as i8)
        .collect();
    let mut i8_nchw = TensorDyn::I8(Tensor::<i8>::from_slice(&q, &shape).unwrap());
    i8_nchw
        .set_quantization(TQ::per_tensor(0.125, 112))
        .unwrap();
    let mut nhwc = vec![0u8; q.len()];
    for c in 0..150 {
        for p in 0..128 * 168 {
            nhwc[p * 150 + c] = (q[c * 128 * 168 + p] as i16 + 128) as u8;
        }
    }
    let mut u8_nhwc = TensorDyn::U8(Tensor::<u8>::from_slice(&nhwc, &[1, 128, 168, 150]).unwrap());
    u8_nhwc
        .set_quantization(TQ::per_tensor(0.125, 240))
        .unwrap();

    let nchw_config = SemanticConfig::scores(&shape, 1, 2, 3).unwrap();
    let nhwc_config = SemanticConfig::scores(&[1, 128, 168, 150], 3, 1, 2).unwrap();
    // A 683x512 ADE20K image letterboxed top-left into 512x672.
    let letterbox = [0.0, 0.0, 1.0, 504.0 / 512.0];
    let mut dst = TensorDyn::new(&[512, 683], DType::U8, Some(TensorMemory::Mem), None).unwrap();
    let mut conf = TensorDyn::new(&[512, 683], DType::U8, Some(TensorMemory::Mem), None).unwrap();

    let cases = [
        ("f32_nchw", &f32_nchw, &nchw_config),
        ("i8_nchw", &i8_nchw, &nchw_config),
        ("u8_nhwc", &u8_nhwc, &nhwc_config),
    ];
    for (label, input, config) in cases {
        for (mode_label, mode) in [
            ("accurate", DecodeMode::Accurate),
            ("fast", DecodeMode::Fast),
        ] {
            let options = SemanticOptions::new()
                .with_mode(mode)
                .with_content_region(letterbox);
            let name = format!("decoder/semantic/{label}/{mode_label}");
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                decode_semantic(input, config, &mut dst, None, &options).unwrap();
            });
            result.print_summary();
            suite.record(&result);
        }
    }
    let options = SemanticOptions::new().with_content_region(letterbox);
    let result = run_bench(
        "decoder/semantic/f32_nchw/accurate_confidence",
        WARMUP,
        ITERATIONS,
        || {
            decode_semantic(&f32_nchw, &nchw_config, &mut dst, Some(&mut conf), &options).unwrap();
        },
    );
    result.print_summary();
    suite.record(&result);

    let halves: Vec<half::f16> = logits.iter().map(|&x| half::f16::from_f32(x)).collect();
    let f16_nchw = TensorDyn::F16(Tensor::<half::f16>::from_slice(&halves, &shape).unwrap());
    let scales: Vec<f32> = (0..150).map(|c| 0.125 + c as f32 * 1e-4).collect();
    let mut i8_per_channel = TensorDyn::I8(Tensor::<i8>::from_slice(&q, &shape).unwrap());
    i8_per_channel
        .set_quantization(TQ::per_channel(scales, vec![112; 150], 1).unwrap())
        .unwrap();
    let sigmoid_config = nchw_config
        .clone()
        .with_activation_required(edgefirst_decoder::schema::Activation::Sigmoid)
        .unwrap();
    // One logit channel at the class map's source resolution, crossing zero in blobs.
    let binary: Vec<f32> = (0..128 * 168)
        .map(|p| {
            let (y, x) = ((p / 168) as f32, (p % 168) as f32);
            ((x * 0.07).sin() + (y * 0.05).cos()) * 4.0
        })
        .collect();
    let binary_shape = [1, 1, 128, 168];
    let binary_logits = TensorDyn::F32(Tensor::<f32>::from_slice(&binary, &binary_shape).unwrap());
    let binary_config = SemanticConfig::scores(&binary_shape, 1, 2, 3).unwrap();
    let accurate = SemanticOptions::new()
        .with_mode(DecodeMode::Accurate)
        .with_content_region(letterbox);
    let fast = SemanticOptions::new()
        .with_mode(DecodeMode::Fast)
        .with_content_region(letterbox);
    let confidence_cases = [
        (
            "f16_nchw/accurate",
            &f16_nchw,
            &nchw_config,
            &accurate,
            false,
        ),
        ("f16_nchw/fast", &f16_nchw, &nchw_config, &fast, false),
        (
            "i8_nchw_per_channel/fast",
            &i8_per_channel,
            &nchw_config,
            &fast,
            false,
        ),
        (
            "f32_nchw/fast_confidence_sigmoid",
            &f32_nchw,
            &sigmoid_config,
            &fast,
            true,
        ),
        (
            "u8_nhwc/fast_confidence",
            &u8_nhwc,
            &nhwc_config,
            &fast,
            true,
        ),
        (
            "f32_binary/accurate_confidence",
            &binary_logits,
            &binary_config,
            &accurate,
            true,
        ),
    ];
    for (label, input, config, options, with_conf) in confidence_cases {
        let name = format!("decoder/semantic/{label}");
        let result = run_bench(&name, WARMUP, ITERATIONS, || {
            let conf = with_conf.then_some(&mut conf);
            decode_semantic(input, config, &mut dst, conf, options).unwrap();
        });
        result.print_summary();
        suite.record(&result);
    }
}

#[allow(dead_code)]
#[path = "../tests/common/per_scale_fixture.rs"]
mod per_scale_fixture;

/// Accumulates the time spent inside each tracing span, by span name.
mod stage_timer {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::Layer;

    #[derive(Default)]
    pub struct Stages {
        /// Total time and entry count per span name.
        pub time: BTreeMap<&'static str, (Duration, u64)>,
        /// Last value of each counting field, as `span.field`.
        pub counts: BTreeMap<String, u64>,
    }

    pub struct StageLayer(pub Arc<Mutex<Stages>>);

    struct Entered(Instant);

    struct Counts<'a> {
        span: &'static str,
        stages: &'a mut Stages,
    }

    impl Visit for Counts<'_> {
        fn record_u64(&mut self, field: &Field, value: u64) {
            if matches!(field.name(), "n" | "n_candidates" | "n_detections") {
                self.stages
                    .counts
                    .insert(format!("{}.{}", self.span, field.name()), value);
            }
        }

        fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}
    }

    impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for StageLayer {
        fn on_new_span(&self, attrs: &Attributes<'_>, _: &Id, _: Context<'_, S>) {
            if let Ok(mut stages) = self.0.lock() {
                attrs.record(&mut Counts {
                    span: attrs.metadata().name(),
                    stages: &mut stages,
                });
            }
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
            let (Some(span), Ok(mut stages)) = (ctx.span(id), self.0.lock()) else {
                return;
            };
            values.record(&mut Counts {
                span: span.name(),
                stages: &mut stages,
            });
        }

        fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().replace(Entered(Instant::now()));
            }
        }

        fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
            let now = Instant::now();
            let Some(span) = ctx.span(id) else { return };
            let Some(Entered(start)) = span.extensions_mut().remove::<Entered>() else {
                return;
            };
            if let Ok(mut stages) = self.0.lock() {
                let slot = stages.time.entry(span.name()).or_default();
                slot.0 += now - start;
                slot.1 += 1;
            }
        }
    }
}

/// A decode regime of the dtype comparison: the deployment threshold with
/// argmax candidates, or COCO validation (multi-label at 0.001).
struct Regime {
    label: &'static str,
    score_threshold: f32,
    multi_label: bool,
    pre_nms_top_k: Option<usize>,
}

const REGIMES: [Regime; 2] = [
    Regime {
        label: "deploy",
        score_threshold: 0.25,
        multi_label: false,
        pre_nms_top_k: None,
    },
    Regime {
        label: "val",
        score_threshold: 0.001,
        multi_label: true,
        pre_nms_top_k: Some(30_000),
    },
];

/// The detection half of a captured per-scale seg fixture: its schema
/// without mask coefficients, protos or a pinned NMS mode, and the tensors
/// that schema binds.
fn detection_fixture(model: &str) -> Option<(SchemaV2, Vec<TensorDyn>)> {
    let path = edgefirst_bench::testdata::path(format!("decoder/{model}.safetensors"));
    let fixture = match per_scale_fixture::PerScaleFixture::load(&path) {
        Ok(f) => f,
        Err(e) => {
            println!("  WARNING: SKIPPED: {model}: {e}");
            return None;
        }
    };
    let mut schema: serde_json::Value =
        serde_json::from_str(fixture.schema_json()).expect("fixture schema");
    schema.as_object_mut().expect("schema object").remove("nms");
    let outputs = schema["outputs"].as_array_mut().expect("outputs");
    outputs.retain(|o| matches!(o["type"].as_str(), Some("boxes" | "scores")));
    let shapes: Vec<Vec<usize>> = outputs
        .iter()
        .flat_map(|o| o["outputs"].as_array().cloned().unwrap_or_default())
        .map(|c| serde_json::from_value(c["shape"].clone()).expect("child shape"))
        .collect();
    let schema: SchemaV2 = serde_json::from_value(schema).expect("detection schema");
    let mut tensors = fixture.build_tensors_with_quant().expect("fixture tensors");
    tensors.retain(|t| shapes.iter().any(|s| s.as_slice() == t.shape()));
    Some((schema, tensors))
}

/// FP32 against FP16 decode on captured int8 frames of the yolov8n, yolo11n
/// (DFL) and yolo26n (LTRB) smart exports, in both [`REGIMES`]: `prefilter`
/// is the per-scale decode alone, `decode` adds the candidate filter and
/// NMS. A second pass prints the mean time inside each tracing span.
fn bench_per_scale_dtype(suite: &mut BenchSuite) {
    use tracing_subscriber::layer::SubscriberExt;

    // `--filter <text>` keeps the cases whose name contains it.
    let filter = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].clone());
    let mut cases = Vec::new();
    for model in ["yolov8n-seg", "yolo11n-seg", "yolo26n-seg"] {
        let Some((schema, tensors)) = detection_fixture(model) else {
            continue;
        };
        let tensors = std::rc::Rc::new(tensors);
        for regime in &REGIMES {
            for (dtype_label, dtype) in [("f32", DecodeDtype::F32), ("f16", DecodeDtype::F16)] {
                let mut builder = DecoderBuilder::default()
                    .with_schema(schema.clone())
                    .with_decode_dtype(dtype)
                    .with_iou_threshold(0.7)
                    .with_score_threshold(regime.score_threshold)
                    .with_nms(Some(Nms::ClassAware))
                    .with_multi_label(regime.multi_label)
                    .with_max_det(300);
                if let Some(k) = regime.pre_nms_top_k {
                    builder = builder.with_pre_nms_top_k(k);
                }
                let decoder = builder.build().expect("decoder build");
                let name = format!(
                    "decoder/per_scale/dtype/{model}/{}/{dtype_label}",
                    regime.label
                );
                if filter.as_ref().is_none_or(|f| name.contains(f.as_str())) {
                    cases.push((name, regime, decoder, tensors.clone()));
                }
            }
        }
    }

    for (name, regime, decoder, tensors) in &cases {
        let inputs: Vec<&TensorDyn> = tensors.iter().collect();
        let threshold = Some(regime.score_threshold);
        let kept = decoder._bench_run_per_scale(&inputs, threshold).unwrap();
        let mut boxes = Vec::with_capacity(300);
        decoder
            .decode(&inputs, &mut boxes, &mut Vec::new())
            .unwrap();
        println!(
            "  {name}: {kept} anchors decoded, {} detections",
            boxes.len()
        );
        run_bench_cpu(
            suite,
            &format!("{name}/prefilter"),
            WARMUP,
            ITERATIONS * 3,
            || {
                decoder._bench_run_per_scale(&inputs, threshold).unwrap();
            },
        );
        run_bench_cpu(
            suite,
            &format!("{name}/decode"),
            WARMUP,
            ITERATIONS * 3,
            || {
                decoder
                    .decode(&inputs, &mut boxes, &mut Vec::new())
                    .unwrap();
            },
        );
    }

    const STAGE_ITERATIONS: u32 = 200;
    for (name, _, decoder, tensors) in &cases {
        let inputs: Vec<&TensorDyn> = tensors.iter().collect();
        let stages = std::sync::Arc::new(std::sync::Mutex::new(stage_timer::Stages::default()));
        let subscriber =
            tracing_subscriber::registry().with(stage_timer::StageLayer(stages.clone()));
        let mut boxes = Vec::with_capacity(300);
        tracing::subscriber::with_default(subscriber, || {
            for i in 0..WARMUP as u32 + STAGE_ITERATIONS {
                if i == WARMUP as u32 {
                    *stages.lock().unwrap() = stage_timer::Stages::default();
                }
                decoder
                    .decode(&inputs, &mut boxes, &mut Vec::new())
                    .unwrap();
            }
        });
        let stages = stages.lock().unwrap();
        println!("  stages {name} (mean per decode, spans traced):");
        for (span, (total, entries)) in &stages.time {
            println!(
                "    {span:58} {:>10.1}µs  x{:.1}",
                total.as_secs_f64() * 1e6 / STAGE_ITERATIONS as f64,
                *entries as f64 / STAGE_ITERATIONS as f64
            );
        }
        for (field, value) in &stages.counts {
            println!("    {field:58} {value:>10}");
        }
    }
}

/// One benchmark function.
type Bench = fn(&mut BenchSuite);

fn main() {
    env_logger::init();

    println!("Decoder Benchmark — edgefirst-bench in-process harness (no fork)");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    let mut suite = BenchSuite::from_args();
    // `--section <name>` runs one section, matched case-insensitively.
    let only = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--section")
        .map(|w| w[1].to_ascii_lowercase());
    let sections: [(&str, &[Bench]); 14] = [
        ("YOLO", &[bench_yolo_quant, bench_yolo_f32]),
        (
            "Quant",
            &[
                bench_quant_decode_boxes,
                bench_quant_decode_boxes_multilabel,
                bench_quant_decode_boxes_i16,
                bench_decode_boxes_rows,
                bench_quant_nms,
                bench_nms_300,
            ],
        ),
        (
            "Dequantize",
            &[
                bench_dequantize_i8,
                bench_dequantize_i8_chunked,
                bench_dequantize_i16,
                bench_dequantize_i16_chunked,
                bench_dequantize_f16_and_ndarray,
            ],
        ),
        (
            "Float",
            &[
                bench_f32_decode_boxes,
                bench_float_decode_boxes_views,
                bench_f32_nms,
            ],
        ),
        (
            "ModelPack",
            &[
                bench_modelpack_u8,
                bench_modelpack_split_u8,
                bench_modelpack_split_40x40,
                bench_modelpack_seg,
            ],
        ),
        (
            "Masks",
            &[bench_masks_f32, bench_masks_i8, bench_masks_counts],
        ),
        ("Semantic", &[bench_semantic]),
        (
            "Per-Scale",
            &[
                bench_per_scale_dfl_i8_to_f32,
                bench_per_scale_dfl_i8_to_f32_nchw,
                bench_per_scale_dfl_i8_to_f16,
                bench_per_scale_ltrb_i8_to_f32,
            ],
        ),
        ("Per-Scale-Realistic", &[bench_per_scale_realistic]),
        ("DFL-Kernels", &[bench_per_scale_dfl_kernels]),
        ("Per-Scale-Seg", &[bench_per_scale_seg]),
        (
            "Per-Scale-Cells",
            &[bench_per_scale_cells, bench_per_scale_select],
        ),
        ("Per-Scale-Boxes", &[bench_per_scale_box_cells]),
        ("Per-Scale-Dtype", &[bench_per_scale_dtype]),
    ];
    for (name, benches) in sections {
        if only
            .as_ref()
            .is_some_and(|o| *o != name.to_ascii_lowercase())
        {
            continue;
        }
        println!("\n== {name} ==\n");
        for bench in benches {
            bench(&mut suite);
        }
    }

    suite.finish();
    println!("\nDone.");
}
