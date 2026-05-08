// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Validate an EdgeFirst model schema and build a [`Decoder`].
//!
//! Accepts either a plain `edgefirst.json` file **or** a `.tflite`
//! model with an embedded ZIP trailer containing `edgefirst.json`
//! (the format produced by the EdgeFirst tflite-converter).
//!
//! The example exercises three public API entry points:
//!
//! 1. [`SchemaV2::parse_json`] — deserialize and version-gate the
//!    metadata.
//! 2. [`DecoderBuilder::with_schema`] — configure the decoder from
//!    the parsed schema (including output layout validation).
//! 3. [`DecoderBuilder::build`] — finalize the [`Decoder`].
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
//! [`DecoderBuilder::with_schema`]: edgefirst_decoder::DecoderBuilder::with_schema
//! [`DecoderBuilder::build`]: edgefirst_decoder::DecoderBuilder::build
//! [`SchemaV2::parse_json`]: edgefirst_decoder::schema::SchemaV2::parse_json

use edgefirst_decoder::schema::SchemaV2;
use edgefirst_decoder::DecoderBuilder;
use std::env;
use std::fs;
use std::io::{Cursor, Read};

/// Extract `edgefirst.json` from a ZIP archive appended to a TFLite
/// flatbuffer.
///
/// The EdgeFirst tflite-converter appends a standard ZIP archive
/// (containing `edgefirst.json`, `labels.txt`, and `metadata.json`)
/// to the end of the FlatBuffer payload. The TFLite runtime ignores
/// the trailing bytes, so the model loads normally.
///
/// Returns `None` when the file is not a valid ZIP or does not contain
/// the expected `edgefirst.json` entry.
fn extract_edgefirst_json_from_tflite(data: &[u8]) -> Option<String> {
    let reader = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).ok()?;
    let mut file = archive.by_name("edgefirst.json").ok()?;
    let mut json = String::new();
    file.read_to_string(&mut json).ok()?;
    Some(json)
}

fn main() {
    let path = env::args().nth(1).expect("usage: parse_edgefirst <path>");

    // Load the schema JSON — either directly from file or extracted
    // from the TFLite ZIP trailer.
    let json_str = if path.ends_with(".tflite") {
        let data = fs::read(&path).expect("read");
        extract_edgefirst_json_from_tflite(&data)
            .unwrap_or_else(|| panic!("{path}: no embedded edgefirst.json in tflite"))
    } else {
        fs::read_to_string(&path).expect("read")
    };

    // Step 1: Parse the JSON into a SchemaV2.
    let schema = match SchemaV2::parse_json(&json_str) {
        Ok(s) => s,
        Err(e) => {
            println!("PARSE-ERR  {}\n     {}", path, e);
            std::process::exit(1);
        }
    };
    println!(
        "PARSE-OK   {} (decoder={:?}, outputs={})",
        path,
        schema.decoder_version,
        schema.outputs.len()
    );

    // Step 2–3: Configure and build the Decoder from the schema.
    match DecoderBuilder::new().with_schema(schema).build() {
        Ok(_) => println!("BUILD-OK   {}", path),
        Err(e) => {
            println!("BUILD-ERR  {}\n     {}", path, e);
            std::process::exit(2);
        }
    }
}
