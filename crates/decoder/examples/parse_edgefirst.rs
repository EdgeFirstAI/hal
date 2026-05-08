// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Validate an EdgeFirst model schema and build a [`Decoder`].
//!
//! Accepts either a plain `edgefirst.json` file **or** a `.tflite`
//! model with an embedded ZIP trailer containing `edgefirst.json`
//! (the format produced by the EdgeFirst tflite-converter). The input
//! type is detected from the file's content, not its extension, so the
//! example also works against renamed or pipeline-staged files.
//!
//! The example exercises four public API entry points:
//!
//! 1. [`SchemaV2::parse_json`] — deserialize and version-gate the
//!    metadata.
//! 2. [`DecoderBuilder::with_schema`] — configure the decoder from
//!    the parsed schema (preserves per-scale FPN children that the
//!    legacy v1 [`with_config_json_str`] path strips).
//! 3. [`DecoderBuilder::build`] — finalize the [`Decoder`].
//! 4. [`Decoder::model_type`] / [`Decoder::input_dims`] /
//!    [`Decoder::normalized_boxes`] — introspect the built decoder so
//!    callers can see what the schema resolved to without running
//!    inference.
//!
//! # Usage
//!
//! ```sh
//! # From a standalone JSON file
//! cargo run --example parse_edgefirst -p edgefirst-decoder -- edgefirst.json
//!
//! # From a TFLite model with embedded metadata
//! cargo run --example parse_edgefirst -p edgefirst-decoder -- model.tflite
//! ```
//!
//! [`Decoder`]: edgefirst_decoder::Decoder
//! [`Decoder::model_type`]: edgefirst_decoder::Decoder::model_type
//! [`Decoder::input_dims`]: edgefirst_decoder::Decoder::input_dims
//! [`Decoder::normalized_boxes`]: edgefirst_decoder::Decoder::normalized_boxes
//! [`DecoderBuilder::with_schema`]: edgefirst_decoder::DecoderBuilder::with_schema
//! [`DecoderBuilder::build`]: edgefirst_decoder::DecoderBuilder::build
//! [`SchemaV2::parse_json`]: edgefirst_decoder::schema::SchemaV2::parse_json
//! [`with_config_json_str`]: edgefirst_decoder::DecoderBuilder::with_config_json_str

use edgefirst_decoder::schema::{LogicalType, SchemaV2};
use edgefirst_decoder::DecoderBuilder;
use std::env;
use std::fs;
use std::io::{Cursor, Read};

/// Try to extract `edgefirst.json` from a ZIP archive appended to a
/// TFLite flatbuffer.
///
/// The EdgeFirst tflite-converter appends a standard ZIP archive
/// (containing `edgefirst.json`, `labels.txt`, and `metadata.json`) to
/// the end of the FlatBuffer payload. The TFLite runtime ignores the
/// trailing bytes, so the model still loads with any standards-
/// compliant interpreter, while ZIP readers locate the central
/// directory by scanning backwards from EOF for `PK\x05\x06`.
///
/// Returns `Ok(None)` when the bytes are not a ZIP archive
/// (i.e. probably a plain `edgefirst.json` file). Returns `Err` when
/// the archive opens but lacks `edgefirst.json` or fails to decode.
fn try_extract_edgefirst_json(data: &[u8]) -> Result<Option<String>, String> {
    let reader = Cursor::new(data);
    let mut archive = match zip::ZipArchive::new(reader) {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };
    let mut file = archive
        .by_name("edgefirst.json")
        .map_err(|e| format!("archive present but missing edgefirst.json: {e}"))?;
    let mut json = String::new();
    file.read_to_string(&mut json)
        .map_err(|e| format!("decode edgefirst.json as utf-8: {e}"))?;
    Ok(Some(json))
}

/// Print a one-line summary of each logical output's name + role.
fn print_output_summary(schema: &SchemaV2) {
    for (i, out) in schema.outputs.iter().enumerate() {
        let name = out.name.as_deref().unwrap_or("<unnamed>");
        let role = match out.type_ {
            Some(LogicalType::Boxes) => "boxes",
            Some(LogicalType::Scores) => "scores",
            Some(LogicalType::MaskCoefs) => "mask_coefs",
            Some(LogicalType::Detection) => "detection (fused)",
            Some(LogicalType::Protos) => "protos",
            Some(_) => "other",
            None => "additional",
        };
        let physical = if out.outputs.is_empty() {
            "1 physical".to_string()
        } else {
            format!("{} per-scale physical", out.outputs.len())
        };
        let shape = format!("{:?}", out.shape);
        println!("  output[{i}] {name:<24} role={role:<18} shape={shape:<22} ({physical})");
    }
}

fn main() {
    let path = env::args().nth(1).expect("usage: parse_edgefirst <path>");

    // Load the schema JSON. Probe the file's bytes to decide whether
    // it's a TFLite-with-archive (ZIP central directory at EOF) or a
    // plain `edgefirst.json` text file. This is more robust than
    // extension sniffing — the file may have been renamed or piped.
    let bytes = fs::read(&path).expect("read");
    let json_str = match try_extract_edgefirst_json(&bytes) {
        Ok(Some(json)) => {
            println!("INPUT      {path} (TFLite with ZIP-trailer metadata)");
            json
        }
        Ok(None) => {
            // Not a ZIP archive: treat the whole file as JSON text.
            let text = String::from_utf8(bytes).unwrap_or_else(|e| {
                eprintln!("INPUT-ERR  {path}\n     not a TFLite archive and not valid UTF-8: {e}");
                std::process::exit(1);
            });
            println!("INPUT      {path} (plain edgefirst.json)");
            text
        }
        Err(e) => {
            eprintln!("INPUT-ERR  {path}\n     {e}");
            std::process::exit(1);
        }
    };

    // Step 1: Parse the JSON into a SchemaV2.
    let schema = match SchemaV2::parse_json(&json_str) {
        Ok(s) => s,
        Err(e) => {
            println!("PARSE-ERR  {path}\n     {e}");
            std::process::exit(1);
        }
    };
    println!(
        "PARSE-OK   schema_version={} decoder_version={:?} nms={:?} outputs={}",
        schema.schema_version,
        schema.decoder_version,
        schema.nms,
        schema.outputs.len()
    );
    if let Some(input) = &schema.input {
        println!(
            "  input    shape={:?} cameraadaptor={:?}",
            input.shape, input.cameraadaptor
        );
    }
    print_output_summary(&schema);

    // Step 2–3: Configure and build the Decoder from the schema.
    //
    // Note: prefer `with_schema` over `with_config_json_str`. The
    // latter downconverts to legacy v1 ConfigOutputs, which discards
    // per-scale FPN children — schemas using `quantization_split` to
    // expose per-stride children fail at decode time without the v2
    // path.
    let decoder = match DecoderBuilder::new().with_schema(schema).build() {
        Ok(d) => d,
        Err(e) => {
            println!("BUILD-ERR  {path}\n     {e}");
            std::process::exit(2);
        }
    };
    println!(
        "BUILD-OK   model_type={:?} input_dims={:?} normalized_boxes={:?}",
        decoder.model_type(),
        decoder.input_dims(),
        decoder.normalized_boxes()
    );
}
