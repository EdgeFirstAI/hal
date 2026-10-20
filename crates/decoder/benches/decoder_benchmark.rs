// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::excessive_precision)]

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_decoder::{
    byte::{nms_int, postprocess_boxes_multilabel_index_quant, postprocess_boxes_quant},
    configs, dequant_detect_box, dequantize_cpu, dequantize_cpu_chunked, dequantize_ndarray,
    float::{nms_float, postprocess_boxes_float},
    per_scale::DecodeDtype,
    schema::SchemaV2,
    ConfigOutput, DecoderBuilder, Nms, Quantization, XYWH,
};
use edgefirst_tensor::{Quantization as TQ, Tensor, TensorDyn, TensorMemory};
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
        &[1, 80, 80, 64],
        &[1, 80, 80, 80],
        &[1, 40, 40, 64],
        &[1, 40, 40, 80],
        &[1, 20, 20, 64],
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
    let (decoder, tensors) = build_dfl_per_scale_bench_inputs(DecodeDtype::F32);
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

fn bench_per_scale_dfl_i8_to_f16(suite: &mut BenchSuite) {
    let (decoder, tensors) = build_dfl_per_scale_bench_inputs(DecodeDtype::F16);
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
                .with_letterbox(letterbox);
            let name = format!("decoder/semantic/{label}/{mode_label}");
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                decode_semantic(input, config, &mut dst, None, &options).unwrap();
            });
            result.print_summary();
            suite.record(&result);
        }
    }
    let options = SemanticOptions::new().with_letterbox(letterbox);
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
}

fn main() {
    env_logger::init();

    println!("Decoder Benchmark — edgefirst-bench in-process harness (no fork)");
    println!("  warmup={WARMUP}  iterations={ITERATIONS}");

    let mut suite = BenchSuite::from_args();

    println!("\n== YOLO ==\n");
    bench_yolo_quant(&mut suite);
    bench_yolo_f32(&mut suite);

    println!("\n== Quant ==\n");
    bench_quant_decode_boxes(&mut suite);
    bench_quant_decode_boxes_multilabel(&mut suite);
    bench_quant_nms(&mut suite);

    println!("\n== Dequantize ==\n");
    bench_dequantize_i8(&mut suite);
    bench_dequantize_i8_chunked(&mut suite);
    bench_dequantize_i16(&mut suite);
    bench_dequantize_i16_chunked(&mut suite);

    println!("\n== Float ==\n");
    bench_f32_decode_boxes(&mut suite);
    bench_f32_nms(&mut suite);

    println!("\n== ModelPack ==\n");
    bench_modelpack_u8(&mut suite);
    bench_modelpack_split_u8(&mut suite);

    println!("\n== Masks ==\n");
    bench_masks_f32(&mut suite);
    bench_masks_i8(&mut suite);

    println!("\n== Semantic ==\n");
    bench_semantic(&mut suite);

    println!("\n== Per-Scale ==\n");
    bench_per_scale_dfl_i8_to_f32(&mut suite);
    bench_per_scale_dfl_i8_to_f16(&mut suite);
    bench_per_scale_ltrb_i8_to_f32(&mut suite);

    suite.finish();
    println!("\nDone.");
}
