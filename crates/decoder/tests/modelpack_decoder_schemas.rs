// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Build a [`Decoder`] from the four ModelPack schema-v2 fixtures shipped
//! by ModelPack 4.0.
//!
//! ModelPack always exports raw per-FPN-scale anchor-grid outputs
//! (``type: detection``, ``encoding: anchor``, ``decoder: modelpack``),
//! optionally accompanied by a single ``type: segmentation`` output for
//! multi-task models. Each FPN scale is its own top-level output — both
//! before and after conversion — so the "smart" layout differs from the
//! Ultralytics case: converters do NOT introduce per-scale physical
//! children, they just fill in `quantization` + `dtype` on each scale in
//! place. The fixtures cover both the float (pre-conversion, no
//! quantization) and the quantized INT8 (post-conversion) variants for
//! detection, segmentation, and multi-task.
//!
//! [`Decoder`]: edgefirst_decoder::Decoder

use std::path::PathBuf;

use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture_path(stem: &str) -> PathBuf {
    workspace_root()
        .join("testdata/decoder")
        .join(format!("{stem}.json"))
}

fn build_from_fixture(stem: &str) -> (SchemaV2, edgefirst_decoder::Decoder) {
    let path = fixture_path(stem);
    let schema =
        SchemaV2::parse_file(&path).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let decoder = DecoderBuilder::new()
        .with_schema(schema.clone())
        .build()
        .unwrap_or_else(|e| panic!("build {}: {e}", path.display()));
    (schema, decoder)
}

/// Shared invariants: ModelPack v2 must NOT set `decoder_version`
/// (that field is Ultralytics-only per the metadata spec), and the
/// expected 320x320 input geometry comes through unchanged.
fn assert_modelpack_invariants(
    stem: &str,
    schema: &SchemaV2,
    decoder: &edgefirst_decoder::Decoder,
) {
    assert_eq!(
        schema.decoder_version, None,
        "{stem}: ModelPack must not set decoder_version (Ultralytics-only field)",
    );
    assert_eq!(
        decoder.input_dims(),
        Some((320, 320)),
        "{stem}: expected 320x320 input",
    );
}

#[test]
fn build_decoder_from_modelpack_det_logical() {
    let stem = "modelpack_det_logical";
    let (schema, decoder) = build_from_fixture(stem);
    assert_modelpack_invariants(stem, &schema, &decoder);

    // Detection-only ModelPack: one logical output per FPN scale, no
    // per-scale children (this is the float / logical layout).
    assert_eq!(
        schema.outputs.len(),
        2,
        "{stem}: detection-only should expose 2 per-scale logical outputs",
    );
    for out in &schema.outputs {
        assert!(
            out.outputs.is_empty(),
            "{stem}: logical layout must not carry per-scale children, found on {:?}",
            out.name,
        );
    }
}

#[test]
fn build_decoder_from_modelpack_seg_logical() {
    let stem = "modelpack_seg_logical";
    let (schema, decoder) = build_from_fixture(stem);
    assert_modelpack_invariants(stem, &schema, &decoder);

    // Segmentation-only ModelPack: a single direct logical output.
    assert_eq!(
        schema.outputs.len(),
        1,
        "{stem}: segmentation-only should expose 1 logical output",
    );
    assert!(
        schema.outputs[0].outputs.is_empty(),
        "{stem}: segmentation logical layout must not carry per-scale children",
    );
}

#[test]
fn build_decoder_from_modelpack_multitask_logical() {
    let stem = "modelpack_multitask_logical";
    let (schema, decoder) = build_from_fixture(stem);
    assert_modelpack_invariants(stem, &schema, &decoder);

    // Multi-task: 3 FPN-scale detection outputs + 1 segmentation output.
    assert_eq!(
        schema.outputs.len(),
        4,
        "{stem}: multitask should expose 3 detection + 1 segmentation = 4 logical outputs",
    );
    for out in &schema.outputs {
        assert!(
            out.outputs.is_empty(),
            "{stem}: logical layout must not carry per-scale children, found on {:?}",
            out.name,
        );
    }
}

#[test]
fn build_decoder_from_modelpack_det_smart() {
    let stem = "modelpack_det_smart";
    let (schema, decoder) = build_from_fixture(stem);
    assert_modelpack_invariants(stem, &schema, &decoder);

    // Post-conversion (INT8 quantized) ModelPack layout: each FPN scale
    // is still a top-level `type: detection` output (no nested physical
    // children — ModelPack's anchor-grid outputs are already per-scale,
    // so converters fill in `quantization` + `dtype` in place rather
    // than introducing per-scale children). Each output carries a
    // non-null `quantization` block and `dtype: int8`.
    assert_eq!(
        schema.outputs.len(),
        2,
        "{stem}: post-conversion layout should expose 2 per-scale outputs",
    );
    for out in &schema.outputs {
        assert!(
            out.outputs.is_empty(),
            "{stem}: ModelPack post-conversion layout must not carry per-scale children, found on {:?}",
            out.name,
        );
        assert!(
            out.quantization.is_some(),
            "{stem}: post-conversion ModelPack output {:?} must declare quantization",
            out.name,
        );
    }
}
