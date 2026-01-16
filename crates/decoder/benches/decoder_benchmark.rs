// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::excessive_precision)]

use divan::black_box_drop;
use edgefirst_decoder::{
    DecoderBuilder, Quantization, XYWH,
    byte::{nms_int, postprocess_boxes_quant},
    configs::{DecoderType, Detection, DimName, Protos},
    dequant_detect_box, dequantize_cpu, dequantize_cpu_chunked, dequantize_ndarray,
    float::{nms_float, postprocess_boxes_float},
    modelpack::{ModelPackDetectionConfig, decode_modelpack_det, decode_modelpack_split_quant},
    yolo::{
        decode_yolo_det, decode_yolo_det_float, decode_yolo_segdet_float, decode_yolo_segdet_quant,
    },
};
use edgefirst_tracker::ByteTrackBuilder;
use ndarray::s;

#[divan::bench()]
fn decoder_yolo_i8(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };

    bencher.bench_local(|| {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_det(
            (out.view(), quant),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
    });
}

#[divan::bench()]
fn decoder_yolo_i8_decoder(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
    let quant = (0.0040811873, -123);

    let decoder = DecoderBuilder::new()
        .with_config_yolo_det(Detection {
            decoder: DecoderType::Ultralytics,
            quantization: Some(quant.into()),
            shape: vec![1, 84, 8400],
            anchors: None,
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumFeatures, 84),
                (DimName::NumBoxes, 8400),
            ],
        })
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .unwrap();
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let mut output_masks: Vec<_> = Vec::with_capacity(50);
    bencher.bench_local(|| {
        decoder
            .decode_quantized(&[out.view().into()], &mut output_boxes, &mut output_masks)
            .unwrap()
    });
}

#[divan::bench()]
fn decoder_quant_decode_boxes(bencher: divan::Bencher) {
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
    bencher.bench_local(|| {
        let _ = postprocess_boxes_quant::<XYWH, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant,
        );
    });
}

#[divan::bench()]
fn decoder_quant_nms(bencher: divan::Bencher) {
    let score_threshold = 0.01;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
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
    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let boxes = nms_int(iou_threshold, boxes);
            let len = output_boxes.capacity().min(boxes.len());
            output_boxes.clear();
            for b in boxes.iter().take(len) {
                output_boxes.push(dequant_detect_box(b, quant));
            }
        });
}

#[divan::bench()]
fn decoder_yolo_f32(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };

    let out = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
    let out: ndarray::Array2<f32> = dequantize_ndarray(out.view(), quant);
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    bencher.bench_local(|| {
        decode_yolo_det_float(
            out.view(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
    });
}

#[divan::bench()]
fn decoder_yolo_f32_decoder(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
    let quant = (0.0040811873, -123);

    let decoder = DecoderBuilder::new()
        .with_config_yolo_det(Detection {
            decoder: DecoderType::Ultralytics,
            quantization: Some(quant.into()),
            shape: vec![1, 84, 8400],
            anchors: None,
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumFeatures, 84),
                (DimName::NumBoxes, 8400),
            ],
        })
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .unwrap();
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let mut output_masks: Vec<_> = Vec::with_capacity(50);

    let out: ndarray::Array3<f32> = dequantize_ndarray(out.view(), quant.into());
    bencher.bench_local(|| {
        decoder
            .decode_float(
                &[out.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap()
    });
}

#[divan::bench()]
fn decoder_i8_dequantize(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };

    let buf = vec![0.0; 84 * 8400];
    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|mut buf| {
            dequantize_cpu(&out, quant, &mut buf);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            black_box_drop(out);
        });
}

#[divan::bench()]
fn decoder_i8_dequantize_chunked(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];
    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|mut buf| {
            dequantize_cpu_chunked(&out, quant, &mut buf);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            black_box_drop(out);
        });
}

#[divan::bench()]
fn decoder_i16_dequantize(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out: Vec<_> = out.iter().map(|x| *x as i16 * *x as i16).collect();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];
    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|mut buf| {
            dequantize_cpu(&out, quant, &mut buf);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            black_box_drop(out);
        });
}

#[divan::bench()]
fn decoder_i16_dequantize_chunked(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out: Vec<_> = out.iter().map(|x| *x as i16).collect();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let buf = vec![0.0; 84 * 8400];
    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|mut buf| {
            dequantize_cpu_chunked(&out, quant, &mut buf);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            black_box_drop(out);
        });
}

#[divan::bench()]
fn decoder_f32_decode_boxes(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization {
        scale: 0.0040811873,
        zero_point: -123,
    };
    let mut buf = vec![0.0; 84 * 8400];
    dequantize_cpu_chunked(&out, quant, &mut buf);

    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|out| {
            let out = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
            let boxes_tensor = out.slice(s![..4, ..,]).reversed_axes();
            let scores_tensor = out.slice(s![4..(80 + 4), ..,]).reversed_axes();
            let boxes =
                postprocess_boxes_float::<XYWH, _, _>(score_threshold, boxes_tensor, scores_tensor);
            black_box_drop(boxes);
        });
}

#[divan::bench()]
fn decoder_f32_nms(bencher: divan::Bencher) {
    let score_threshold = 0.01;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
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
    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let mut output_boxes: Vec<_> = Vec::with_capacity(50);
            let boxes = nms_float(iou_threshold, boxes);
            let len = output_boxes.capacity().min(boxes.len());
            output_boxes.clear();
            for b in boxes.into_iter().take(len) {
                output_boxes.push(b);
            }
            black_box_drop(output_boxes);
        });
}

#[divan::bench()]
fn decoder_modelpack_u8(bencher: divan::Bencher) {
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
    bencher.bench_local(|| {
        decode_modelpack_det(
            (boxes.view(), quant_boxes),
            (scores.view(), quant_scores),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        )
    });
}

#[divan::bench()]
fn decoder_modelpack_split_u8(bencher: divan::Bencher) {
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
    bencher.bench_local(|| {
        decode_modelpack_split_quant(
            &outputs,
            &configs,
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        )
    });
}

#[divan::bench()]
fn decoder_modelpack_split_u8_decoder(bencher: divan::Bencher) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
    let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();
    let config0 = Detection {
        anchors: Some(vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ]),
        decoder: DecoderType::ModelPack,
        quantization: Some((0.08547406643629074, 174).into()),
        shape: vec![1, 9, 15, 18],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 9),
            (DimName::Width, 15),
            (DimName::NumAnchorsXFeatures, 18),
        ],
    };

    let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
    let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();
    let config1 = Detection {
        anchors: Some(vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ]),
        decoder: DecoderType::ModelPack,
        quantization: Some((0.09929127991199493, 183).into()),
        shape: vec![1, 17, 30, 18],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 17),
            (DimName::Width, 30),
            (DimName::NumAnchorsXFeatures, 18),
        ],
    };
    let decoder = DecoderBuilder::new()
        .with_config_modelpack_det_split(vec![config0, config1])
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .unwrap();
    let outputs = [detect0.view().into(), detect1.view().into()];
    let mut output_boxes: Vec<_> = Vec::with_capacity(2);
    let mut output_masks: Vec<_> = Vec::with_capacity(2);
    bencher.bench_local(|| {
        decoder
            .decode_quantized(&outputs, &mut output_boxes, &mut output_masks)
            .unwrap()
    });
}

#[divan::bench()]
fn decoder_modelpack_split_u8_decoder_tracked(bencher: divan::Bencher) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
    let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();
    let config0 = Detection {
        anchors: Some(vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ]),
        decoder: DecoderType::ModelPack,
        quantization: Some((0.08547406643629074, 174).into()),
        shape: vec![1, 9, 15, 18],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 9),
            (DimName::Width, 15),
            (DimName::NumAnchorsXFeatures, 18),
        ],
    };

    let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
    let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();
    let config1 = Detection {
        anchors: Some(vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ]),
        decoder: DecoderType::ModelPack,
        quantization: Some((0.09929127991199493, 183).into()),
        shape: vec![1, 17, 30, 18],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 17),
            (DimName::Width, 30),
            (DimName::NumAnchorsXFeatures, 18),
        ],
    };
    let mut decoder = DecoderBuilder::new()
        .with_config_modelpack_det_split(vec![config0, config1])
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .with_tracker(ByteTrackBuilder::new().build())
        .build()
        .unwrap();
    let outputs = [detect0.view().into(), detect1.view().into()];
    let mut output_boxes: Vec<_> = Vec::with_capacity(2);
    let mut output_masks: Vec<_> = Vec::with_capacity(2);
    let mut output_tracks: Vec<_> = Vec::with_capacity(2);
    let mut time = 0;
    bencher.bench_local(|| {
        decoder
            .decode_quantized_tracked(
                &outputs,
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                time,
            )
            .unwrap();
        time += 100_000;
    });
}

#[divan::bench()]
fn decoder_masks_f32(bencher: divan::Bencher) {
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
    let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
    let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
    bencher.bench_local(|| {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        decode_yolo_segdet_float(
            seg.view(),
            protos.view(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
            &mut output_masks,
        );
        black_box_drop(output_boxes);
        black_box_drop(output_masks);
    });
}

#[divan::bench()]
fn decoder_masks_f32_decoder(bencher: divan::Bencher) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
    let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
    let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
    let quant_boxes = (0.01948494464159012, 20);

    let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
    let protos = unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
    let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
    let quant_protos = (0.020889872685074806, -115);

    let seg_config = Detection {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_boxes.into()),
        shape: vec![1, 116, 8400],
        anchors: None,
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumFeatures, 116),
            (DimName::NumBoxes, 8400),
        ],
    };
    let protos_config = Protos {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_protos.into()),
        shape: vec![1, 160, 160, 32],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 160),
            (DimName::Width, 160),
            (DimName::NumProtos, 32),
        ],
    };

    let decoder = DecoderBuilder::new()
        .with_config_yolo_segdet(seg_config, protos_config)
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .unwrap();

    let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos.into());
    let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());

    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let mut output_masks: Vec<_> = Vec::with_capacity(50);
    bencher.bench_local(|| {
        decoder
            .decode_float(
                &[seg.view().into_dyn(), protos.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();
    });
}

#[divan::bench()]
fn decoder_masks_i8(bencher: divan::Bencher) {
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

    bencher.bench_local(|| {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        decode_yolo_segdet_quant(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
            &mut output_masks,
        );
        black_box_drop(output_boxes);
        black_box_drop(output_masks);
    });
}

#[divan::bench()]
fn decoder_masks_i8_decoder(bencher: divan::Bencher) {
    let score_threshold = 0.45;
    let iou_threshold = 0.45;
    let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
    let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
    let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
    let quant_boxes = (0.01948494464159012, 20);

    let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
    let protos = unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
    let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
    let quant_protos = (0.020889872685074806, -115);

    let seg_config = Detection {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_boxes.into()),
        shape: vec![1, 116, 8400],
        anchors: None,
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumFeatures, 116),
            (DimName::NumBoxes, 8400),
        ],
    };
    let protos_config = Protos {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_protos.into()),
        shape: vec![1, 160, 160, 32],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 160),
            (DimName::Width, 160),
            (DimName::NumProtos, 32),
        ],
    };

    let decoder = DecoderBuilder::new()
        .with_config_yolo_segdet(seg_config, protos_config)
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .build()
        .unwrap();
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let mut output_masks: Vec<_> = Vec::with_capacity(50);
    bencher.bench_local(|| {
        decoder
            .decode_quantized(
                &[boxes.view().into(), protos.view().into()],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();
    });
}

#[divan::bench()]
fn decoder_masks_i8_decoder_tracked(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.45;
    let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
    let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
    let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
    let quant_boxes = (0.01948494464159012, 20);

    let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
    let protos = unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
    let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
    let quant_protos = (0.020889872685074806, -115);

    let seg_config = Detection {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_boxes.into()),
        shape: vec![1, 116, 8400],
        anchors: None,
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::NumFeatures, 116),
            (DimName::NumBoxes, 8400),
        ],
    };
    let protos_config = Protos {
        decoder: DecoderType::Ultralytics,
        quantization: Some(quant_protos.into()),
        shape: vec![1, 160, 160, 32],
        dshape: vec![
            (DimName::Batch, 1),
            (DimName::Height, 160),
            (DimName::Width, 160),
            (DimName::NumProtos, 32),
        ],
    };

    let mut decoder = DecoderBuilder::new()
        .with_config_yolo_segdet(seg_config, protos_config)
        .with_score_threshold(score_threshold)
        .with_iou_threshold(iou_threshold)
        .with_tracker(ByteTrackBuilder::new().build())
        .build()
        .unwrap();
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let mut output_masks: Vec<_> = Vec::with_capacity(50);
    let mut output_tracks: Vec<_> = Vec::with_capacity(50);
    let mut time = 0;
    bencher.bench_local(|| {
        decoder
            .decode_quantized_tracked(
                &[boxes.view().into(), protos.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                time,
            )
            .unwrap();
        time += 100_000;
    });
}

fn main() {
    env_logger::init();
    divan::main();
}
