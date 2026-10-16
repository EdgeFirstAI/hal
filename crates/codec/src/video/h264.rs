// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 Annex B byte-stream framing (ITU-T H.264 §7.3.1 and Annex B).
//!
//! Encoders hand back Annex B access units: NAL units each preceded by a
//! `00 00 01` or `00 00 00 01` start code. This module finds the NAL units
//! and classifies the ones the encoder cares about, without parsing slice
//! data.

/// `nal_unit_type` of a coded slice of an IDR picture.
pub(crate) const NAL_IDR: u8 = 5;
/// `nal_unit_type` of a sequence parameter set.
pub(crate) const NAL_SPS: u8 = 7;
/// `nal_unit_type` of a picture parameter set.
pub(crate) const NAL_PPS: u8 = 8;
/// `nal_unit_type` of an end of sequence.
const NAL_END_OF_SEQUENCE: u8 = 10;
/// `nal_unit_type` of an end of stream.
const NAL_END_OF_STREAM: u8 = 11;
/// `nal_unit_type` of filler data.
const NAL_FILLER: u8 = 12;

/// The `nal_unit_type` values of every NAL unit in an Annex B buffer, in
/// order. Bytes before the first start code are ignored.
pub(crate) fn nal_types(data: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut i = 0;
    while let Some(start) = next_start_code(data, i) {
        if let Some(&header) = data.get(start) {
            types.push(header & 0x1f);
        }
        i = start;
    }
    types
}

/// The SPS and PPS NAL units of an Annex B buffer, each with a 4-byte
/// start code, in stream order.
pub(crate) fn parameter_sets(data: &[u8]) -> Vec<u8> {
    let mut starts = Vec::new();
    let mut i = 0;
    while let Some(start) = next_start_code(data, i) {
        starts.push(start);
        i = start;
    }
    let mut out = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let Some(&header) = data.get(start) else {
            continue;
        };
        if !matches!(header & 0x1f, NAL_SPS | NAL_PPS) {
            continue;
        }
        // The NAL unit ends where the next start code (and any zero byte of a
        // 4-byte code) begins.
        let mut end = starts.get(k + 1).map_or(data.len(), |&next| next - 3);
        while end > start && data[end - 1] == 0 && k + 1 < starts.len() {
            end -= 1;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[start..end]);
    }
    out
}

/// Index of the first byte after the next `00 00 01` start code at or
/// after `from` (a 4-byte `00 00 00 01` code ends with the same 3 bytes).
fn next_start_code(data: &[u8], from: usize) -> Option<usize> {
    data.get(from..)?
        .windows(3)
        .position(|w| w == [0, 0, 1])
        .map(|p| from + p + 3)
}

/// What an encoded buffer contains, by its NAL units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AccessUnit {
    /// Contains a slice of an IDR picture.
    pub idr: bool,
    /// Contains an SPS and a PPS.
    pub parameter_sets: bool,
    /// Contains at least one coded slice (`nal_unit_type` 1 to 5).
    pub has_picture: bool,
    /// Has NAL units, and every one may only follow a picture (end of
    /// sequence, end of stream, filler data; §7.4.1.2.3).
    pub trailing_only: bool,
}

impl AccessUnit {
    /// Classifies an Annex B buffer.
    pub(crate) fn parse(data: &[u8]) -> Self {
        let types = nal_types(data);
        Self {
            idr: types.contains(&NAL_IDR),
            parameter_sets: types.contains(&NAL_SPS) && types.contains(&NAL_PPS),
            has_picture: types.iter().any(|t| (1..=5).contains(t)),
            trailing_only: !types.is_empty()
                && types
                    .iter()
                    .all(|t| matches!(*t, NAL_END_OF_SEQUENCE | NAL_END_OF_STREAM | NAL_FILLER)),
        }
    }

    /// Belongs to the access unit before it: no picture, only NAL units that
    /// end one (an encoder's end of sequence after a drain).
    pub(crate) fn ends_previous(&self) -> bool {
        !self.has_picture && self.trailing_only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPS: &[u8] = &[0, 0, 0, 1, 0x67, 0x64, 0x00, 0x28];
    const PPS: &[u8] = &[0, 0, 0, 1, 0x68, 0xee, 0x3c, 0x80];
    const IDR: &[u8] = &[0, 0, 1, 0x65, 0x88, 0x84, 0x00];
    const P_SLICE: &[u8] = &[0, 0, 0, 1, 0x41, 0x9a, 0x02];
    const SEI: &[u8] = &[0, 0, 1, 0x06, 0x05, 0xff];

    fn cat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    #[test]
    fn finds_nal_types_after_3_and_4_byte_start_codes() {
        let au = cat(&[SPS, PPS, SEI, IDR]);
        assert_eq!(nal_types(&au), vec![NAL_SPS, NAL_PPS, 6, NAL_IDR]);
    }

    #[test]
    fn ignores_leading_garbage_and_empty_input() {
        assert!(nal_types(&[]).is_empty());
        assert!(nal_types(&[0, 0]).is_empty());
        assert_eq!(nal_types(&cat(&[&[0xaa, 0xbb], P_SLICE])), vec![1]);
    }

    #[test]
    fn start_code_at_the_very_end_has_no_nal() {
        assert_eq!(nal_types(&cat(&[P_SLICE, &[0, 0, 1]])), vec![1]);
    }

    #[test]
    fn keyframe_with_headers() {
        let au = AccessUnit::parse(&cat(&[SPS, PPS, IDR]));
        assert!(au.idr && au.parameter_sets && au.has_picture);
        assert!(!au.ends_previous());
    }

    #[test]
    fn idr_without_headers_is_still_idr() {
        let au = AccessUnit::parse(IDR);
        assert!(au.idr && !au.parameter_sets);
    }

    #[test]
    fn predicted_frame() {
        let au = AccessUnit::parse(P_SLICE);
        assert!(!au.idr && !au.parameter_sets && au.has_picture);
    }

    #[test]
    fn end_of_sequence_belongs_to_the_previous_access_unit() {
        let eos: &[u8] = &[0, 0, 0, 1, 0x0a];
        let au = AccessUnit::parse(eos);
        assert!(au.ends_previous() && !au.parameter_sets);
        assert!(AccessUnit::parse(&cat(&[eos, &[0, 0, 1, 0x0b]])).ends_previous());
        assert!(!AccessUnit::parse(&cat(&[P_SLICE, eos])).ends_previous());
        assert!(
            !AccessUnit::parse(SEI).ends_previous(),
            "SEI precedes a picture"
        );
        assert!(!AccessUnit::parse(&[]).ends_previous());
    }

    #[test]
    fn extracts_parameter_sets_with_4_byte_start_codes() {
        let au = cat(&[SPS, PPS, SEI, IDR]);
        assert_eq!(parameter_sets(&au), cat(&[SPS, PPS]));
        // 3-byte start codes come back as 4-byte ones.
        let short = cat(&[&[0, 0, 1, 0x67, 0x42], &[0, 0, 1, 0x68, 0xce], IDR]);
        assert_eq!(
            parameter_sets(&short),
            vec![0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce]
        );
        assert!(parameter_sets(&cat(&[P_SLICE, SEI])).is_empty());
        assert_eq!(parameter_sets(&cat(&[IDR, PPS])), PPS.to_vec(), "last NAL");
    }

    #[test]
    fn header_only_buffer() {
        let au = AccessUnit::parse(&cat(&[SPS, PPS]));
        assert!(au.parameter_sets && !au.has_picture && !au.ends_previous());
        assert!(!AccessUnit::parse(SPS).parameter_sets);
        assert!(!AccessUnit::parse(SEI).has_picture);
    }
}
