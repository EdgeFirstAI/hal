// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Every schema `infer_ultralytics_schema` produces from a captured
//! fixture is a schema [`DecoderBuilder`] accepts and can build a
//! [`Decoder`] from.
//!
//! [`Decoder`]: edgefirst_decoder::Decoder

use edgefirst_decoder::infer::*;
use edgefirst_decoder::schema::{DType, SchemaV2};
use edgefirst_decoder::DecoderBuilder;
use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

/// Root of the `testdata/infer` fixture directory: under
/// `EDGEFIRST_TESTDATA_DIR` when set, else the manifest-relative path baked in
/// at compile time.
fn infer_fixture_dir() -> std::path::PathBuf {
    std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata"))
        .join("infer")
}

/// Loads a captured fixture into `ModelSignals`: tensor names/shapes/dtypes
/// plus the raw metadata map.
fn signals_from_fixture(name: &str) -> ModelSignals {
    let path = infer_fixture_dir().join(format!("{name}.signals.json"));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&content).expect("fixture is valid JSON");

    let source = match json["source"]
        .as_str()
        .expect("fixture `source` is a string")
    {
        "onnx" => ModelSource::Onnx,
        "tflite" => ModelSource::TfLite,
        "coreml" => ModelSource::CoreMl,
        other => panic!("fixture {name}: unknown source `{other}`"),
    };

    fn parse_dtype(s: &str) -> DType {
        match s {
            "float32" => DType::Float32,
            "float16" => DType::Float16,
            "int8" => DType::Int8,
            "uint8" => DType::Uint8,
            "int16" => DType::Int16,
            "uint16" => DType::Uint16,
            "int32" => DType::Int32,
            "uint32" => DType::Uint32,
            other => panic!("unknown dtype `{other}`"),
        }
    }

    fn parse_tensor(v: &serde_json::Value) -> TensorInfo {
        TensorInfo {
            name: v["name"]
                .as_str()
                .expect("tensor `name` is a string")
                .to_string(),
            shape: v["shape"]
                .as_array()
                .expect("tensor `shape` is an array")
                .iter()
                .map(|d| d.as_u64().expect("shape dim is a number") as usize)
                .collect(),
            dtype: parse_dtype(v["dtype"].as_str().expect("tensor `dtype` is a string")),
            quantization: None,
        }
    }

    let inputs = json["inputs"]
        .as_array()
        .expect("fixture `inputs` is an array")
        .iter()
        .map(parse_tensor)
        .collect();
    let outputs = json["outputs"]
        .as_array()
        .expect("fixture `outputs` is an array")
        .iter()
        .map(parse_tensor)
        .collect();
    let metadata = json["metadata"]
        .as_object()
        .expect("fixture `metadata` is an object")
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_str().expect("metadata value is a string").to_string(),
            )
        })
        .collect();

    ModelSignals {
        source,
        inputs,
        outputs,
        metadata,
    }
}

#[test]
fn inferred_schemas_build_decoders() {
    for name in [
        "yolov8n",
        "yolo11n",
        "yolo26n",
        "yolov8n-seg",
        "yolo11n-seg",
        "yolo26n-seg",
        "yolov8n_float32",
        "yolov8n_int8",
        "yolov8n-seg_float32",
        "yolov8n-seg_int8",
        "yolo26n_float32",
    ] {
        let r = infer_ultralytics_schema(&signals_from_fixture(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        DecoderBuilder::new()
            .with_schema(r.schema)
            .with_input_dims(640, 640)
            .with_score_threshold(0.001)
            .build()
            .unwrap_or_else(|e| panic!("{name}: builder rejected schema: {e}"));
    }
}

/// The CoreML fixture builds a `Decoder` through `DecoderBuilder`.
#[test]
fn coreml_yolo26n_fixture_builds_a_decoder() {
    let signals = signals_from_fixture("yolo26n_coreml");
    let inferred = infer_ultralytics_schema(&signals).expect("CoreML fixture must infer a schema");
    assert_eq!(
        inferred.schema.outputs[0].normalized,
        Some(false),
        "CoreML boxes are pixel-space, matching the ONNX export"
    );
    DecoderBuilder::new()
        .with_schema(inferred.schema)
        .with_input_dims(640, 640)
        .with_score_threshold(0.001)
        .build()
        .expect("inferred CoreML schema must build a Decoder");
}

/// An inferred vanilla ONNX schema round-trips through JSON, builds a decoder and decodes a known pixel-space box.
#[test]
fn inferred_yolov8n_flat_detection_normalized_flag_and_json_roundtrip() {
    let signals = signals_from_fixture("yolov8n");
    let inferred = infer_ultralytics_schema(&signals).expect("yolov8n: schema inference failed");

    let det = inferred
        .schema
        .outputs
        .iter()
        .find(|o| o.type_ == Some(edgefirst_decoder::schema::LogicalType::Detection))
        .expect("yolov8n: no `detection` logical output in inferred schema");
    assert_eq!(
        det.normalized,
        Some(false),
        "ONNX pre-NMS detection output must be declared pixel-space"
    );
    let inferred_shape = det.shape.clone();
    let inferred_dshape = det.dshape.clone();

    // Serialize, then re-parse through `SchemaV2::parse_json`.
    let json = serde_json::to_string(&inferred.schema).expect("schema serializes");
    let roundtripped = SchemaV2::parse_json(&json).expect("schema re-parses");
    let rt_det = roundtripped
        .outputs
        .iter()
        .find(|o| o.type_ == Some(edgefirst_decoder::schema::LogicalType::Detection))
        .expect("roundtripped schema: no `detection` logical output");

    assert_eq!(
        rt_det.normalized,
        Some(false),
        "`normalized: false` must survive the JSON round-trip"
    );
    assert_eq!(
        rt_det.shape, inferred_shape,
        "`shape` must survive the JSON round-trip"
    );
    assert_eq!(
        rt_det.dshape, inferred_dshape,
        "`dshape` must survive the JSON round-trip"
    );

    let decoder = DecoderBuilder::new()
        .with_schema(roundtripped)
        .with_input_dims(640, 640)
        .with_score_threshold(0.5)
        .build()
        .unwrap_or_else(|e| panic!("yolov8n: builder rejected round-tripped schema: {e}"));

    assert_eq!(
        decoder.normalized_boxes(),
        Some(false),
        "ModelType::YoloDet (detection-only, no protos) is documented as a \
         path that surfaces the raw schema flag verbatim rather than \
         normalizing internally (see `Decoder::normalized_boxes` docs)"
    );

    // Handcraft a [1, 84, 8400] f32 tensor: all scores ~0 except anchor
    // `K` where class 0 scores 0.9 and the box channels (0..4) carry a
    // pixel-space cxcywh box: cx=320, cy=320, w=128, h=64 (out of 640).
    const FEAT: usize = 84;
    const N: usize = 8400;
    const K: usize = 1234;
    let mut data = vec![0.0f32; FEAT * N];
    let set = |data: &mut [f32], channel: usize, anchor: usize, v: f32| {
        data[channel * N + anchor] = v;
    };
    set(&mut data, 0, K, 320.0);
    set(&mut data, 1, K, 320.0);
    set(&mut data, 2, K, 128.0);
    set(&mut data, 3, K, 64.0);
    set(&mut data, 4, K, 0.9); // class 0 score

    let tensor: TensorDyn = {
        let t = Tensor::<f32>::new(&[1, FEAT, N], Some(TensorMemory::Mem), None).unwrap();
        {
            let mut m = t.map().unwrap();
            m.as_mut_slice().copy_from_slice(&data);
        }
        t.into()
    };

    let mut output_boxes = Vec::new();
    let mut output_masks = Vec::new();
    decoder
        .decode(&[&tensor], &mut output_boxes, &mut output_masks)
        .expect("decode should succeed");

    assert_eq!(
        output_boxes.len(),
        1,
        "expected exactly one detection above the 0.5 score threshold, got {output_boxes:?}"
    );
    let b = &output_boxes[0];
    assert_eq!(b.label, 0);
    assert!((b.score - 0.9).abs() < 1e-3, "score: {}", b.score);

    let cx = (b.bbox.xmin + b.bbox.xmax) * 0.5;
    let cy = (b.bbox.ymin + b.bbox.ymax) * 0.5;
    let w = b.bbox.xmax - b.bbox.xmin;
    let h = b.bbox.ymax - b.bbox.ymin;
    assert!(
        !(cx > 0.99 && cy > 0.99 && w < 0.01 && h < 0.01),
        "reproduced the corruption signature: cx={cx} cy={cy} w={w} h={h} \
         (box={:?})",
        b.bbox
    );

    // The box comes back in pixel space: cx=320, cy=320, w=128, h=64.
    assert!(
        (cx - 320.0).abs() < 1.0 && (cy - 320.0).abs() < 1.0,
        "expected pixel-space center (320, 320) per the documented \
         YoloDet contract, got cx={cx} cy={cy} (box={:?})",
        b.bbox
    );
    assert!(
        (w - 128.0).abs() < 1.0 && (h - 64.0).abs() < 1.0,
        "expected pixel-space size (128, 64) per the documented YoloDet \
         contract, got w={w} h={h} (box={:?})",
        b.bbox
    );
}
