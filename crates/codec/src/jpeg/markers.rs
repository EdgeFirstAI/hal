// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! JPEG marker segment parsing (SOF, SOS, DQT, DHT, DRI, APP/EXIF).

use crate::error::CodecError;
use crate::jpeg::huffman::HuffmanTable;
use crate::jpeg::types::{marker, Component, ImageHeader, QuantTable, SamplingFactor, ZIGZAG};

/// Read a big-endian u16 from two bytes.
#[inline]
fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([data[offset], data[offset + 1]])
}

/// All state parsed from JPEG marker segments before the entropy-coded data.
#[derive(Debug)]
pub struct JpegHeaders {
    pub header: ImageHeader,
    pub quant_tables: [QuantTable; 4],
    pub dc_tables: [Option<HuffmanTable>; 4],
    pub ac_tables: [Option<HuffmanTable>; 4],
    pub restart_interval: u16,
    /// Raw EXIF data (APP1 payload) if present.
    pub exif_data: Option<Vec<u8>>,
    /// Byte offset of the entropy-coded data (after SOS).
    pub scan_data_offset: usize,
}

/// Parse all JPEG marker segments from `data` and return headers + scan offset.
///
/// # Errors
///
/// Returns `CodecError::InvalidData` for malformed markers, missing SOF/SOS,
/// progressive JPEGs, or truncated data.
pub fn parse_markers(data: &[u8]) -> crate::Result<JpegHeaders> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != marker::SOI {
        return Err(CodecError::InvalidData("not a JPEG (missing SOI)".into()));
    }

    let mut pos = 2;
    let mut header: Option<ImageHeader> = None;
    let mut quant_tables = [
        QuantTable::default(),
        QuantTable::default(),
        QuantTable::default(),
        QuantTable::default(),
    ];
    let mut dc_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
    let mut restart_interval: u16 = 0;
    let mut exif_data: Option<Vec<u8>> = None;
    let mut scan_found = false;
    let mut scan_data_offset = 0usize;

    loop {
        // Find next 0xFF marker byte
        while pos < data.len() && data[pos] != 0xFF {
            pos += 1;
        }
        // Skip fill bytes
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            return Err(CodecError::InvalidData("truncated JPEG markers".into()));
        }

        let marker_byte = data[pos];
        pos += 1;

        match marker_byte {
            0x00 => continue, // Stuffed byte, not a marker
            marker::EOI => break,
            marker::SOI => continue,

            // Restart markers — shouldn't appear in marker segments
            m if (marker::RST0..=marker::RST7).contains(&m) => continue,

            marker::SOF0 | marker::SOF2 => {
                let is_progressive = marker_byte == marker::SOF2;
                if pos + 1 >= data.len() {
                    return Err(CodecError::InvalidData("truncated SOF".into()));
                }
                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() || seg_len < 8 {
                    return Err(CodecError::InvalidData("invalid SOF length".into()));
                }
                let precision = data[pos + 2];
                let height = read_u16(data, pos + 3);
                let width = read_u16(data, pos + 5);
                let num_components = data[pos + 7] as usize;

                if precision != 8 {
                    return Err(CodecError::InvalidData(format!(
                        "unsupported precision {precision} (only 8-bit baseline)"
                    )));
                }
                if num_components == 0 || num_components > 4 {
                    return Err(CodecError::InvalidData(format!(
                        "invalid component count {num_components}"
                    )));
                }
                if seg_len < 8 + num_components * 3 {
                    return Err(CodecError::InvalidData(
                        "SOF too short for components".into(),
                    ));
                }

                let mut components = Vec::with_capacity(num_components);
                let mut max_h = 1u8;
                let mut max_v = 1u8;
                for i in 0..num_components {
                    let base = pos + 8 + i * 3;
                    let id = data[base];
                    let hv = data[base + 1];
                    let h = hv >> 4;
                    let v = hv & 0x0F;
                    let qt = data[base + 2];
                    if h == 0 || h > 4 || v == 0 || v > 4 {
                        return Err(CodecError::InvalidData(format!(
                            "invalid sampling factor {h}×{v} for component {id}"
                        )));
                    }
                    if qt > 3 {
                        return Err(CodecError::InvalidData(format!(
                            "invalid quant table id {qt}"
                        )));
                    }
                    max_h = max_h.max(h);
                    max_v = max_v.max(v);
                    components.push(Component {
                        id,
                        sampling: SamplingFactor { h, v },
                        quant_table_id: qt,
                        dc_table_id: 0,
                        ac_table_id: 0,
                    });
                }

                header = Some(ImageHeader {
                    width,
                    height,
                    components,
                    max_h_samp: max_h,
                    max_v_samp: max_v,
                    is_progressive,
                });
                pos += seg_len;
            }

            marker::DQT => {
                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() {
                    return Err(CodecError::InvalidData("truncated DQT".into()));
                }
                let mut off = pos + 2;
                let seg_end = pos + seg_len;
                while off < seg_end {
                    let pq_tq = data[off];
                    let precision_16 = (pq_tq >> 4) != 0;
                    let table_id = (pq_tq & 0x0F) as usize;
                    off += 1;
                    if table_id > 3 {
                        return Err(CodecError::InvalidData(format!(
                            "invalid DQT table id {table_id}"
                        )));
                    }
                    let entry_size = if precision_16 { 2 } else { 1 };
                    if off + 64 * entry_size > data.len() {
                        return Err(CodecError::InvalidData("truncated DQT data".into()));
                    }
                    let mut qt = QuantTable::default();
                    for i in 0..64 {
                        let val = if precision_16 {
                            read_u16(data, off + i * 2)
                        } else {
                            data[off + i] as u16
                        };
                        // Store in zig-zag order → natural order for IDCT
                        qt.values[ZIGZAG[i] as usize] = val;
                    }
                    quant_tables[table_id] = qt;
                    off += 64 * entry_size;
                }
                pos += seg_len;
            }

            marker::DHT => {
                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() {
                    return Err(CodecError::InvalidData("truncated DHT".into()));
                }
                let mut off = pos + 2;
                let seg_end = pos + seg_len;
                while off < seg_end {
                    let tc_th = data[off];
                    let table_class = tc_th >> 4; // 0 = DC, 1 = AC
                    let table_id = (tc_th & 0x0F) as usize;
                    off += 1;
                    if table_class > 1 || table_id > 3 {
                        return Err(CodecError::InvalidData(format!(
                            "invalid DHT class={table_class} id={table_id}"
                        )));
                    }
                    if off + 16 > data.len() {
                        return Err(CodecError::InvalidData("truncated DHT counts".into()));
                    }
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&data[off..off + 16]);
                    off += 16;
                    let total: usize = counts.iter().map(|&c| c as usize).sum();
                    if off + total > data.len() {
                        return Err(CodecError::InvalidData("truncated DHT values".into()));
                    }
                    let values = data[off..off + total].to_vec();
                    off += total;

                    let table = HuffmanTable::build(&counts, &values)?;
                    if table_class == 0 {
                        dc_tables[table_id] = Some(table);
                    } else {
                        ac_tables[table_id] = Some(table);
                    }
                }
                pos += seg_len;
            }

            marker::DRI => {
                let seg_len = read_u16(data, pos) as usize;
                if seg_len < 4 || pos + seg_len > data.len() {
                    return Err(CodecError::InvalidData("invalid DRI".into()));
                }
                restart_interval = read_u16(data, pos + 2);
                pos += seg_len;
            }

            marker::SOS => {
                let hdr = header
                    .as_mut()
                    .ok_or_else(|| CodecError::InvalidData("SOS before SOF".into()))?;

                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() || seg_len < 6 {
                    return Err(CodecError::InvalidData("truncated SOS".into()));
                }
                let num_scan_components = data[pos + 2] as usize;
                if seg_len < 6 + num_scan_components * 2 {
                    return Err(CodecError::InvalidData("SOS too short".into()));
                }

                let mut component_indices = Vec::with_capacity(num_scan_components);
                for i in 0..num_scan_components {
                    let base = pos + 3 + i * 2;
                    let cs = data[base]; // Component selector
                    let td_ta = data[base + 1];
                    let dc_id = td_ta >> 4;
                    let ac_id = td_ta & 0x0F;

                    // Find matching component
                    let idx = hdr
                        .components
                        .iter()
                        .position(|c| c.id == cs)
                        .ok_or_else(|| {
                            CodecError::InvalidData(format!(
                                "SOS references unknown component {cs}"
                            ))
                        })?;
                    hdr.components[idx].dc_table_id = dc_id;
                    hdr.components[idx].ac_table_id = ac_id;
                    component_indices.push(idx);
                }

                let _ = component_indices; // Consumed by component table updates above

                scan_found = true;
                scan_data_offset = pos + seg_len;
                break; // Entropy-coded data follows immediately
            }

            marker::APP1 => {
                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() {
                    return Err(CodecError::InvalidData("truncated APP1".into()));
                }
                // Check for EXIF header "Exif\0\0"
                if seg_len > 8 && &data[pos + 2..pos + 8] == b"Exif\0\0" {
                    exif_data = Some(data[pos + 2..pos + seg_len].to_vec());
                }
                pos += seg_len;
            }

            // Skip all other APP and unknown markers
            _ => {
                if pos + 1 >= data.len() {
                    return Err(CodecError::InvalidData("truncated marker segment".into()));
                }
                let seg_len = read_u16(data, pos) as usize;
                if pos + seg_len > data.len() {
                    return Err(CodecError::InvalidData("truncated marker payload".into()));
                }
                pos += seg_len;
            }
        }
    }

    let header = header.ok_or_else(|| CodecError::InvalidData("no SOF marker found".into()))?;

    if header.is_progressive {
        return Err(CodecError::InvalidData(
            "progressive JPEG not supported (baseline only)".into(),
        ));
    }

    if !scan_found {
        return Err(CodecError::InvalidData("no SOS marker found".into()));
    }

    Ok(JpegHeaders {
        header,
        quant_tables,
        dc_tables,
        ac_tables,
        restart_interval,
        exif_data,
        scan_data_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid JPEG: SOI + SOF0 + DHT(DC0) + DHT(AC0) + DQT + SOS + EOI.
    fn minimal_jpeg() -> Vec<u8> {
        let mut buf = Vec::new();
        // SOI
        buf.extend_from_slice(&[0xFF, marker::SOI]);
        // DQT: table 0, 8-bit precision, all values = 1
        buf.extend_from_slice(&[0xFF, marker::DQT]);
        let dqt_len: u16 = 2 + 1 + 64;
        buf.extend_from_slice(&dqt_len.to_be_bytes());
        buf.push(0x00); // Pq=0 (8-bit), Tq=0
        buf.extend_from_slice(&[1u8; 64]);
        // SOF0: 8×8, 1 component (greyscale)
        buf.extend_from_slice(&[0xFF, marker::SOF0]);
        let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3;
        buf.extend_from_slice(&sof_len.to_be_bytes());
        buf.push(8); // precision
        buf.extend_from_slice(&8u16.to_be_bytes()); // height
        buf.extend_from_slice(&8u16.to_be_bytes()); // width
        buf.push(1); // 1 component
        buf.extend_from_slice(&[1, 0x11, 0]); // id=1, 1×1 sampling, qt=0
                                              // DHT: DC table 0 (minimal — 1 code of length 1 for value 0)
        buf.extend_from_slice(&[0xFF, marker::DHT]);
        let dht_dc_len: u16 = 2 + 1 + 16 + 1;
        buf.extend_from_slice(&dht_dc_len.to_be_bytes());
        buf.push(0x00); // class=DC, id=0
        let mut counts = [0u8; 16];
        counts[0] = 1; // 1 code of length 1
        buf.extend_from_slice(&counts);
        buf.push(0x00); // value = 0 (DC diff = 0)
                        // DHT: AC table 0 (minimal — 1 code of length 1 for value 0x00 = EOB)
        buf.extend_from_slice(&[0xFF, marker::DHT]);
        let dht_ac_len: u16 = 2 + 1 + 16 + 1;
        buf.extend_from_slice(&dht_ac_len.to_be_bytes());
        buf.push(0x10); // class=AC, id=0
        let mut ac_counts = [0u8; 16];
        ac_counts[0] = 1;
        buf.extend_from_slice(&ac_counts);
        buf.push(0x00); // value = 0x00 (EOB)
                        // SOS: 1 component
        buf.extend_from_slice(&[0xFF, marker::SOS]);
        let sos_len: u16 = 2 + 1 + 2 + 3;
        buf.extend_from_slice(&sos_len.to_be_bytes());
        buf.push(1); // 1 component
        buf.extend_from_slice(&[1, 0x00]); // comp 1, DC=0 AC=0
        buf.extend_from_slice(&[0, 63, 0]); // Ss=0, Se=63, Ah=0|Al=0
                                            // Entropy data: we just put a few bytes then EOI
        buf.extend_from_slice(&[0x00, 0xFF, marker::EOI]);
        buf
    }

    #[test]
    fn parse_minimal_jpeg() {
        let data = minimal_jpeg();
        let headers = parse_markers(&data).unwrap();
        assert_eq!(headers.header.width, 8);
        assert_eq!(headers.header.height, 8);
        assert_eq!(headers.header.components.len(), 1);
        assert!(!headers.header.is_progressive);
        assert_eq!(headers.restart_interval, 0);
        assert!(headers.dc_tables[0].is_some());
        assert!(headers.ac_tables[0].is_some());
    }

    #[test]
    fn reject_not_jpeg() {
        let result = parse_markers(b"not a jpeg");
        assert!(result.is_err());
    }

    #[test]
    fn reject_truncated() {
        let result = parse_markers(&[0xFF, marker::SOI]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_real_jpeg() {
        let testdata = std::env::var("EDGEFIRST_TESTDATA_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").into());
        let path = std::path::Path::new(&testdata).join("zidane.jpg");
        if !path.exists() {
            return;
        }
        let data = std::fs::read(&path).unwrap();
        let headers = parse_markers(&data).unwrap();
        assert_eq!(headers.header.width, 1280);
        assert_eq!(headers.header.height, 720);
        assert_eq!(headers.header.components.len(), 3);
        assert!(!headers.header.is_progressive);
    }
}
