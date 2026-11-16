// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Build a [`Decoder`] from each of the three `edgefirst.json` schemas
//! extracted from the YOLOv8n-seg t-2681 quant-u8-i8 model variants.
//!
//! These fixtures cover three top-level output topologies:
//!
//! * `combined` — single fused detection tensor + protos (`YoloSegDet`).
//! * `logical`  — boxes / scores / mask_coefs split into one tensor each
//!   plus protos (`YoloSplitSegDet`, no per-scale children).
//! * `smart`    — same logical heads but each fanned out into per-scale
//!   physical child tensors (`YoloSplitSegDet` driven by the per-scale
//!   pipeline).
//!
//! [`Decoder`]: edgefirst_decoder::Decoder

mod common;

use std::path::PathBuf;

use common::testdata_root;

use edgefirst_decoder::{
    schema::{DecoderVersion, SchemaV2},
    DecoderBuilder,
};

fn fixture_path(stem: &str) -> PathBuf {
    testdata_root().join("decoder").join(format!("{stem}.json"))
}

/// Parse the JSON schema and build a [`Decoder`], returning both.
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

/// Asserts shared by every fixture: decoder version and input geometry.
fn assert_common_invariants(stem: &str, schema: &SchemaV2, decoder: &edgefirst_decoder::Decoder) {
    assert_eq!(
        schema.decoder_version,
        Some(DecoderVersion::Yolov8),
        "{stem}: expected decoder_version=yolov8",
    );
    assert_eq!(
        decoder.input_dims(),
        Some((640, 640)),
        "{stem}: expected 640x640 input",
    );
}

#[test]
fn build_decoder_from_combined_schema() {
    let stem = "yolov8n-seg-t-2681_quant-u8-i8_combined";
    let (schema, decoder) = build_from_fixture(stem);
    assert_common_invariants(stem, &schema, &decoder);

    // `combined`: two top-level outputs (detection + protos) with no
    // per-scale children.
    assert_eq!(
        schema.outputs.len(),
        2,
        "{stem}: combined topology should have 2 top-level outputs",
    );
    for out in &schema.outputs {
        assert!(
            out.outputs.is_empty(),
            "{stem}: combined topology must not carry per-scale children, found on {:?}",
            out.name,
        );
    }
}

#[test]
fn build_decoder_from_logical_schema() {
    let stem = "yolov8n-seg-t-2681_quant-u8-i8_logical";
    let (schema, decoder) = build_from_fixture(stem);
    assert_common_invariants(stem, &schema, &decoder);

    // `logical`: boxes / scores / mask_coefs and protos as 4 top-level
    // outputs with no per-scale children.
    assert_eq!(
        schema.outputs.len(),
        4,
        "{stem}: logical topology should have 4 top-level outputs",
    );
    for out in &schema.outputs {
        assert!(
            out.outputs.is_empty(),
            "{stem}: logical topology must not carry per-scale children, found on {:?}",
            out.name,
        );
    }
}

#[test]
fn build_decoder_from_smart_schema() {
    let stem = "yolov8n-seg-t-2681_quant-u8-i8_smart";
    let (schema, decoder) = build_from_fixture(stem);
    assert_common_invariants(stem, &schema, &decoder);

    // `smart`: boxes, scores and mask_coefs each carry 3 per-scale children
    // (strides 8/16/32); protos is a single tensor.
    assert_eq!(
        schema.outputs.len(),
        4,
        "{stem}: smart topology should have 4 top-level outputs",
    );
    let per_scale_children: usize = schema
        .outputs
        .iter()
        .filter(|o| !o.outputs.is_empty())
        .map(|o| o.outputs.len())
        .sum();
    assert_eq!(
        per_scale_children, 9,
        "{stem}: smart topology should expose 3 strides × 3 logical heads = 9 child tensors",
    );
}
