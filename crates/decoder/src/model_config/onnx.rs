// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! ONNX `ModelProto.metadata_props`, read by walking the top-level protobuf
//! fields and seeking past every other one, so the graph is never decoded or
//! even read.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use super::ModelConfigError;

/// `ModelProto.metadata_props` (repeated `StringStringEntryProto`).
const METADATA_PROPS: u64 = 14;
/// Largest metadata entry read into memory.
const MAX_ENTRY: u64 = 16 << 20;

fn err(msg: impl Into<String>) -> ModelConfigError {
    ModelConfigError::Container(format!("ONNX: {}", msg.into()))
}

fn read_varint(r: &mut impl Read, pos: &mut u64, end: u64) -> Result<u64, ModelConfigError> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        if *pos >= end {
            return Err(err("truncated varint"));
        }
        let mut b = [0u8; 1];
        r.read_exact(&mut b).map_err(|e| err(e.to_string()))?;
        *pos += 1;
        // The tenth byte holds bit 63 only.
        if shift == 63 && b[0] > 1 {
            return Err(err("varint overflows 64 bits"));
        }
        value |= u64::from(b[0] & 0x7f) << shift;
        if b[0] & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(err("varint longer than 10 bytes"))
}

/// The `metadata_props` of the `ModelProto` in `r`, which holds `len` bytes.
pub(crate) fn metadata_props<R: Read + Seek>(
    r: &mut R,
    len: u64,
) -> Result<BTreeMap<String, String>, ModelConfigError> {
    let mut props = BTreeMap::new();
    let mut pos = 0u64;
    r.seek(SeekFrom::Start(0)).map_err(|e| err(e.to_string()))?;
    while pos < len {
        let key = read_varint(r, &mut pos, len)?;
        let (field, wire) = (key >> 3, key & 7);
        if field == 0 {
            return Err(err("field number 0"));
        }
        let skip = match wire {
            0 => {
                read_varint(r, &mut pos, len)?;
                0
            }
            1 => 8,
            5 => 4,
            2 => {
                let n = read_varint(r, &mut pos, len)?;
                if n > len - pos {
                    return Err(err(format!(
                        "field {field} claims {n} bytes past the end of the file"
                    )));
                }
                if field != METADATA_PROPS {
                    n
                } else if let Some(size) = usize::try_from(n).ok().filter(|_| n <= MAX_ENTRY) {
                    let mut buf = vec![0u8; size];
                    r.read_exact(&mut buf).map_err(|e| err(e.to_string()))?;
                    pos += n;
                    match string_entry(&buf)? {
                        Some((k, v)) => {
                            props.insert(k, v);
                        }
                        None => log::debug!("ONNX metadata entry that is not UTF-8 skipped"),
                    }
                    0
                } else {
                    log::debug!("ONNX metadata entry of {n} bytes skipped");
                    n
                }
            }
            w => return Err(err(format!("unsupported wire type {w} on field {field}"))),
        };
        if skip > 0 {
            if skip > len - pos {
                return Err(err(format!("field {field} runs past the end of the file")));
            }
            let offset = i64::try_from(skip).map_err(|_| err("field too large to skip"))?;
            r.seek(SeekFrom::Current(offset))
                .map_err(|e| err(e.to_string()))?;
            pos += skip;
        }
    }
    Ok(props)
}

/// A `StringStringEntryProto { key = 1; value = 2; }`; `None` when either is
/// not UTF-8.
fn string_entry(buf: &[u8]) -> Result<Option<(String, String)>, ModelConfigError> {
    let mut r = std::io::Cursor::new(buf);
    let end = buf.len() as u64;
    let mut pos = 0u64;
    let (mut key, mut value) = (None, None);
    while pos < end {
        let tag = read_varint(&mut r, &mut pos, end)?;
        if tag & 7 != 2 {
            return Err(err("metadata entry field is not length-delimited"));
        }
        let n = read_varint(&mut r, &mut pos, end)?;
        if n > end - pos {
            return Err(err("metadata entry runs past its end"));
        }
        let bytes = &buf[pos as usize..(pos + n) as usize];
        pos += n;
        r.set_position(pos);
        let text = std::str::from_utf8(bytes).ok().map(str::to_owned);
        match tag >> 3 {
            1 => key = text,
            2 => value = text,
            _ => {}
        }
    }
    Ok(key.zip(value))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn field_bytes(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
        varint(field << 3 | 2, out);
        varint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
    }

    /// A `ModelProto` with `ir_version`, a graph of `graph_len` bytes and the
    /// given metadata props.
    pub(crate) fn model_proto(props: &[(&str, &str)], graph_len: usize) -> Vec<u8> {
        let mut out = vec![0x08, 0x08]; // ir_version = 8
        field_bytes(2, b"pytorch", &mut out);
        field_bytes(7, &vec![0xffu8; graph_len], &mut out);
        for (k, v) in props {
            let mut entry = Vec::new();
            field_bytes(1, k.as_bytes(), &mut entry);
            field_bytes(2, v.as_bytes(), &mut entry);
            field_bytes(METADATA_PROPS, &entry, &mut out);
        }
        out
    }

    fn scan(bytes: &[u8]) -> Result<BTreeMap<String, String>, ModelConfigError> {
        metadata_props(&mut std::io::Cursor::new(bytes), bytes.len() as u64)
    }

    #[test]
    fn onnx_metadata_props_are_read() {
        let bytes = model_proto(&[("edgefirst", "{}"), ("labels", "a\nb")], 64);
        let props = scan(&bytes).unwrap();
        assert_eq!(props["edgefirst"], "{}");
        assert_eq!(props["labels"], "a\nb");
    }

    /// The graph bytes are 0xff, which are not a valid protobuf; reading them
    /// as fields would fail, so a successful scan proves they were skipped.
    #[test]
    fn onnx_scan_skips_graph_without_reading_it() {
        let bytes = model_proto(&[("hf_config", "{\"a\":1}")], 1 << 20);
        assert_eq!(scan(&bytes).unwrap()["hf_config"], "{\"a\":1}");
    }

    #[test]
    fn onnx_scan_rejects_length_past_eof() {
        let mut bytes = model_proto(&[], 0);
        varint(7 << 3 | 2, &mut bytes);
        varint(u64::MAX >> 1, &mut bytes);
        assert!(matches!(scan(&bytes), Err(ModelConfigError::Container(_))));
        let mut bytes = model_proto(&[], 0);
        bytes.extend_from_slice(&[0x80; 11]);
        assert!(scan(&bytes).is_err(), "an overlong varint is refused");
        let mut r =
            std::io::Cursor::new([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02]);
        assert!(read_varint(&mut r, &mut 0, 10).is_err(), "2^64 is refused");
        let mut r =
            std::io::Cursor::new([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]);
        assert_eq!(read_varint(&mut r, &mut 0, 10).unwrap(), u64::MAX);
        assert!(scan(&[0x0b]).is_err(), "group wire type is refused");
    }
}
