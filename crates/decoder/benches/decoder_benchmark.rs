use divan::black_box_drop;
use edgefirst_decoder::{
    Quantization, XYWH,
    bits8::{decode_boxes_i8, decode_i8, nms_i16},
    dequant_detect_box, dequantize_cpu, dequantize_cpu_chunked,
    float::{decode_boxes_f32, decode_f32, nms_f32},
};
use ndarray::s;

#[divan::bench()]
fn decoder_quant(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };

    bencher.bench_local(|| {
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_i8::<XYWH>(
            out.view(),
            80,
            &quant,
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
    });
}

#[divan::bench()]
fn decoder_quant_decode_boxes(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = out.slice(s![..4, ..,]);
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]);
    bencher.bench_local(|| {
        let _ = decode_boxes_i8::<XYWH>(score_threshold, scores_tensor, boxes_tensor, 80, &quant);
    });
}

#[divan::bench()]
fn decoder_quant_nms(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };
    let mut output_boxes: Vec<_> = Vec::with_capacity(50);
    let out = ndarray::Array2::from_shape_vec((84, 8400), out.clone()).unwrap();
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = out.slice(s![..4, ..,]);
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]);
    let boxes = decode_boxes_i8::<XYWH>(score_threshold, scores_tensor, boxes_tensor, 80, &quant);
    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let boxes = nms_i16(iou_threshold, boxes);
            let len = output_boxes.capacity().min(boxes.len());
            output_boxes.clear();
            for b in boxes.iter().take(len) {
                output_boxes.push(dequant_detect_box(b, &quant));
            }
        });
}

#[divan::bench()]
fn decoder_f32(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };

    bencher
        .with_inputs(|| out.clone())
        .bench_local_values(|out| {
            let mut buf = vec![0.0; 84 * 8400];
            dequantize_cpu_chunked(&quant, &out, &mut buf);
            let mut output_boxes: Vec<_> = Vec::with_capacity(50);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            decode_f32::<XYWH>(
                out.view(),
                80,
                score_threshold,
                iou_threshold,
                &mut output_boxes,
            );
            black_box_drop(output_boxes);
        });
}

#[divan::bench()]
fn decoder_f32_dequantize(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };

    bencher
        .with_inputs(|| out.clone())
        .bench_local_values(|out| {
            let mut buf = vec![0.0; 84 * 8400];
            dequantize_cpu(&quant, &out, &mut buf);
            let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
            black_box_drop(out);
        });
}

#[divan::bench()]
fn decoder_f32_dequantize_chunked(bencher: divan::Bencher) {
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };

    bencher
        .with_inputs(|| out.clone())
        .bench_local_values(|out| {
            let mut buf = vec![0.0; 84 * 8400];
            dequantize_cpu_chunked(&quant, &out, &mut buf);
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
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };
    let mut buf = vec![0.0; 84 * 8400];
    dequantize_cpu_chunked(&quant, &out, &mut buf);

    bencher
        .with_inputs(|| buf.clone())
        .bench_local_values(|out| {
            let out = ndarray::Array2::from_shape_vec((84, 8400), out).unwrap();
            let boxes_tensor = out.slice(s![..4, ..,]);
            let scores_tensor = out.slice(s![4..(80 + 4), ..,]);
            let boxes = decode_boxes_f32::<XYWH>(score_threshold, scores_tensor, boxes_tensor, 80);
            black_box_drop(boxes);
        });
}

#[divan::bench()]
fn decoder_f32_nms(bencher: divan::Bencher) {
    let score_threshold = 0.25;
    let iou_threshold = 0.70;
    let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
    let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
    let out = out.to_vec();
    // let output = ndarray::Array2::from_shape_vec((84, 8400),
    // output.clone()).unwrap();
    let quant = Quantization::<i8> {
        scale: 0.0040811873,
        zero_point: -123i8,
    };
    let mut buf = vec![0.0; 84 * 8400];
    dequantize_cpu_chunked(&quant, &out, &mut buf);
    let out = ndarray::Array2::from_shape_vec((84, 8400), buf).unwrap();
    let boxes_tensor = out.slice(s![..4, ..,]);
    let scores_tensor = out.slice(s![4..(80 + 4), ..,]);
    let boxes = decode_boxes_f32::<XYWH>(score_threshold, scores_tensor, boxes_tensor, 80);
    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let mut output_boxes: Vec<_> = Vec::with_capacity(50);
            let boxes = nms_f32(iou_threshold, boxes);
            let len = output_boxes.capacity().min(boxes.len());
            output_boxes.clear();
            for b in boxes.into_iter().take(len) {
                output_boxes.push(b);
            }
            black_box_drop(output_boxes);
        });
}

fn main() {
    env_logger::init();
    divan::main();
}
