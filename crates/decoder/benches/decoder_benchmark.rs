// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::excessive_precision)]

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_decoder::{
    byte::{nms_int, postprocess_boxes_quant},
    dequant_detect_box, dequantize_cpu, dequantize_cpu_chunked, dequantize_ndarray,
    float::{nms_float, postprocess_boxes_float},
    modelpack::{decode_modelpack_det, decode_modelpack_split_quant, ModelPackDetectionConfig},
    yolo::{
        decode_yolo_det, decode_yolo_det_float, decode_yolo_segdet_float, decode_yolo_segdet_quant,
    },
    Nms, Quantization, XYWH,
};
use ndarray::s;

const WARMUP: usize = 10;
const ITERATIONS: usize = 200;

fn bench_yolo_quant(suite: &mut BenchSuite) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };

    let result = run_bench("decoder/yolo/quant", WARMUP, ITERATIONS, || {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_det(
            (out.view(), quant),
            score_threshold,
            iou_threshold,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
        );
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_quant_decode_boxes(suite: &mut BenchSuite) {
    let score_threshold = 0.25;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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

fn bench_quant_nms(suite: &mut BenchSuite) {
    let score_threshold = 0.01;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
        let boxes = nms_int(iou_threshold, boxes);
        let len = output_boxes.capacity().min(boxes.len());
        output_boxes.clear();
        for b in boxes.iter().take(len) {
            output_boxes.push(dequant_detect_box(b, quant));
        }
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_yolo_f32(suite: &mut BenchSuite) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };

    let result = run_bench("decoder/yolo/f32", WARMUP, ITERATIONS, || {
        let out = out.clone();
        let mut buf = vec![0.0; 84 * 8400];
        dequantize_cpu_chunked(&out, quant, &mut buf);
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
        decode_yolo_det_float(
            out.view(),
            score_threshold,
            iou_threshold,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
        );
        std::hint::black_box(output_boxes);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_dequantize_i8(suite: &mut BenchSuite) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
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
        let boxes = nms_float(iou_threshold, boxes);
        let len = output_boxes.capacity().min(boxes.len());
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
    let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
    let boxes = ndarray::Array2::from_shape_vec((1935, 4), boxes.to_vec()).unwrap();

    let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
    let scores = ndarray::Array2::from_shape_vec((1935, 1), scores.to_vec()).unwrap();

    let quant_boxes = Quantization {
        scale: 0.004656755365431309,
        zero_point: 21,
    };

    let quant_scores = Quantization {
        scale: 0.0019603664986789227,
        zero_point: 0,
    };

    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let result = run_bench("decoder/modelpack/u8", WARMUP, ITERATIONS, || {
        decode_modelpack_det(
            (boxes.view(), quant_boxes),
            (scores.view(), quant_scores),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        )
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_modelpack_split_u8(suite: &mut BenchSuite) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
    let detect0 = ndarray::Array3::from_shape_vec((9, 15, 18), detect0.to_vec()).unwrap();
    let config0 = ModelPackDetectionConfig {
        anchors: vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ],
        quantization: Some(Quantization {
            scale: 0.08547406643629074,
            zero_point: 174,
        }),
    };

    let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
    let detect1 = ndarray::Array3::from_shape_vec((17, 30, 18), detect1.to_vec()).unwrap();
    let config1 = ModelPackDetectionConfig {
        anchors: vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ],
        quantization: Some(Quantization {
            scale: 0.09929127991199493,
            zero_point: 183,
        }),
    };
    let outputs = [detect0.view(), detect1.view()];
    let configs = [config0, config1];
    let mut output_boxes: Vec<_> = Vec::with_capacity(2);

    let result = run_bench("decoder/modelpack/split_u8", WARMUP, ITERATIONS, || {
        decode_modelpack_split_quant(
            &outputs,
            &configs,
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        )
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_masks_f32(suite: &mut BenchSuite) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
    let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
    let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes.to_vec()).unwrap();
    let quant_boxes = Quantization {
        scale: 0.01948494464159012,
        zero_point: 20,
    };

    let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
    let protos = unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
    let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos.to_vec()).unwrap();
    let quant_protos = Quantization {
        scale: 0.020889872685074806,
        zero_point: -115,
    };

    let result = run_bench("decoder/masks/f32", WARMUP, ITERATIONS, || {
        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        decode_yolo_segdet_float(
            seg.view(),
            protos.view(),
            score_threshold,
            iou_threshold,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
            &mut output_masks,
        )
        .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_masks_i8(suite: &mut BenchSuite) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
    let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
    let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes.to_vec()).unwrap();
    let quant_boxes = Quantization {
        scale: 0.01948494464159012,
        zero_point: 20,
    };

    let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
    let protos = unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
    let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos.to_vec()).unwrap();
    let quant_protos = Quantization {
        scale: 0.020889872685074806,
        zero_point: -115,
    };

    let result = run_bench("decoder/masks/i8", WARMUP, ITERATIONS, || {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        decode_yolo_segdet_quant(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            score_threshold,
            iou_threshold,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
            &mut output_masks,
        )
        .unwrap();
        std::hint::black_box(output_boxes);
        std::hint::black_box(output_masks);
    });
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

    suite.finish();
    println!("\nDone.");
}
