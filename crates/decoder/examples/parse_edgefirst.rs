// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Validate an EdgeFirst model schema and build a [`Decoder`].
//!
//! Accepts either a plain `edgefirst.json` file or a `.tflite` model with
//! an embedded ZIP trailer containing `edgefirst.json`. The input type is
//! detected from the file's content, not its extension.
//!
//! The example exercises four public API entry points:
//!
//! 1. [`SchemaV2::parse_json`] — deserialize and version-gate the
//!    metadata.
//! 2. [`DecoderBuilder::with_schema`] — configure the decoder from
//!    the parsed schema.
//! 3. [`DecoderBuilder::build`] — finalize the [`Decoder`].
//! 4. [`Decoder::model_type`] / [`Decoder::input_dims`] /
//!    [`Decoder::normalized_boxes`] — introspect the built decoder.
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

use edgefirst_decoder::schema::{LogicalType, SchemaV2};
use edgefirst_decoder::DecoderBuilder;
use std::env;
use std::fs;
use std::io::{Cursor, Read};

/// Try to extract `edgefirst.json` from a ZIP archive appended to a
/// TFLite flatbuffer.
///
/// Returns `Ok(None)` when the bytes are not a ZIP archive. Returns `Err`
/// when the archive opens but lacks `edgefirst.json` or fails to decode.
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

    // Load the schema JSON from a TFLite-with-archive or a plain
    // `edgefirst.json` text file, told apart by the file's bytes.
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
