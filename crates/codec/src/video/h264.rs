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
/// `nal_unit_type` of an access unit delimiter.
pub(crate) const NAL_AUD: u8 = 9;
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
#[cfg_attr(
    all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"),
    allow(dead_code)
)]
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

/// The end of the first NAL unit at or after `from` in an Annex B buffer:
/// where the next start code (with the zero byte of a 4-byte code) begins,
/// or the end of the buffer.
fn nal_end(data: &[u8], from: usize) -> usize {
    let Some(first) = next_start_code(data, from) else {
        return data.len();
    };
    match next_start_code(data, first) {
        Some(next) => start_code_begin(data, next),
        None => data.len(),
    }
}

/// `au` with `parameter_sets` inserted before its first NAL unit, or after
/// it when that is an access unit delimiter, which must come first
/// (§7.4.1.2.3).
#[cfg_attr(
    not(any(
        all(windows, feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) fn with_parameter_sets(au: &[u8], parameter_sets: &[u8]) -> Vec<u8> {
    let at = if nal_types(au).first() == Some(&NAL_AUD) {
        nal_end(au, 0)
    } else {
        0
    };
    let mut out = Vec::with_capacity(au.len() + parameter_sets.len());
    out.extend_from_slice(&au[..at]);
    out.extend_from_slice(parameter_sets);
    out.extend_from_slice(&au[at..]);
    out
}

/// Converts an Annex B buffer to AVCC: every NAL unit whose `nal_unit_type`
/// is not in `drop`, each preceded by its length as a 4-byte big-endian
/// integer. Bytes before the first start code are ignored.
#[cfg_attr(
    not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")),
    allow(dead_code)
)]
pub(crate) fn annex_b_to_avcc(au: &[u8], drop: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(au.len() + 16);
    for nal in nal_units(au) {
        if let Some(&header) = nal.first() {
            if !drop.contains(&(header & 0x1f)) {
                out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                out.extend_from_slice(nal);
            }
        }
    }
    out
}

/// The NAL units of an Annex B buffer, each from its header byte to the
/// next start code, in order. Bytes before the first start code are ignored.
pub(crate) fn nal_units(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut next = next_start_code(data, 0);
    std::iter::from_fn(move || {
        let first = next?;
        next = next_start_code(data, first);
        let end = next.map_or(data.len(), |n| start_code_begin(data, n));
        Some(&data[first..end])
    })
}

/// Converts an AVCC sample to Annex B, each NAL unit preceded by a
/// `00 00 00 01` start code. `length_size` is the width in bytes of each
/// length prefix: 1, 2 or 4.
///
/// # Errors
///
/// `length_size` is not 1, 2 or 4, a length prefix is cut short, or a length
/// runs past the end of the sample. The message names the byte offset.
#[cfg_attr(
    not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")),
    allow(dead_code)
)]
pub(crate) fn avcc_to_annex_b(sample: &[u8], length_size: usize) -> Result<Vec<u8>, String> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(format!("unsupported NAL length size {length_size}"));
    }
    let mut out = Vec::with_capacity(sample.len() + 16);
    let mut at = 0;
    while at < sample.len() {
        let Some(prefix) = sample.get(at..at + length_size) else {
            return Err(format!("truncated NAL length at byte {at}"));
        };
        let len = prefix.iter().fold(0usize, |n, &b| (n << 8) | b as usize);
        let start = at + length_size;
        let Some(nal) = start
            .checked_add(len)
            .and_then(|end| sample.get(start..end))
        else {
            return Err(format!("NAL length {len} at byte {at} runs past the end"));
        };
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        at = start + len;
    }
    Ok(out)
}

/// Index of the first byte after the next `00 00 01` start code at or
/// after `from` (a 4-byte `00 00 00 01` code ends with the same 3 bytes).
fn next_start_code(data: &[u8], from: usize) -> Option<usize> {
    data.get(from..)?
        .windows(3)
        .position(|w| w == [0, 0, 1])
        .map(|p| from + p + 3)
}

/// Splits an H.264 Annex B byte stream into access units, the input
/// [`VideoDecoder::decode`](super::VideoDecoder::decode) takes.
///
/// An access unit starts at the first NAL unit after a picture that may only
/// precede one (access unit delimiter, SEI, SPS, PPS; §7.4.1.2.3), or at a
/// slice that starts a new picture (`first_mb_in_slice` is 0). Streams that
/// send a picture's slices out of order (arbitrary slice order) are split at
/// each slice that starts at macroblock 0, which such streams do not
/// guarantee. Bytes before the first start code are skipped.
pub fn access_units(stream: &[u8]) -> AccessUnits<'_> {
    AccessUnits {
        data: stream,
        next: next_start_code(stream, 0),
    }
}

/// Iterator over the access units of a byte stream, from [`access_units`].
#[derive(Debug, Clone)]
pub struct AccessUnits<'a> {
    data: &'a [u8],
    /// First byte of the next access unit's first NAL unit header.
    next: Option<usize>,
}

impl<'a> Iterator for AccessUnits<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let data = self.data;
        let first = self.next?;
        let start = start_code_begin(data, first);
        let mut picture = false;
        let mut nal = first;
        loop {
            let Some(&header) = data.get(nal) else {
                self.next = None;
                return Some(&data[start..]);
            };
            let kind = header & 0x1f;
            let slice = matches!(kind, 1 | 5);
            // ue(v) first_mb_in_slice is 0 exactly when its first bit is 1.
            let new_picture = slice && data.get(nal + 1).is_some_and(|b| b & 0x80 != 0);
            let leads = matches!(kind, 6..=9 | 14..=18);
            if picture && (leads || new_picture) {
                self.next = Some(nal);
                return Some(&data[start..start_code_begin(data, nal)]);
            }
            picture |= slice;
            match next_start_code(data, nal) {
                Some(n) => nal = n,
                None => {
                    self.next = None;
                    return Some(&data[start..]);
                }
            }
        }
    }
}

/// Where the start code before the NAL unit header at `nal` begins,
/// including the zero byte of a 4-byte code.
fn start_code_begin(data: &[u8], nal: usize) -> usize {
    let begin = nal - 3;
    if begin > 0 && data[begin - 1] == 0 {
        begin - 1
    } else {
        begin
    }
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
    #[cfg_attr(
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"),
        allow(dead_code)
    )]
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
    fn nal_units_are_split_at_3_and_4_byte_start_codes() {
        let au = cat(&[&[0xff], SPS, PPS, SEI, P_SLICE]);
        let nals: Vec<&[u8]> = nal_units(&au).collect();
        assert_eq!(nals, vec![&SPS[4..], &PPS[4..], &SEI[3..], &P_SLICE[4..]]);
        assert_eq!(nal_units(&[]).count(), 0);
        assert_eq!(nal_units(&[0, 0, 2, 0x65]).count(), 0);
    }

    #[test]
    fn annex_b_with_3_and_4_byte_start_codes_becomes_length_prefixed() {
        let au = cat(&[
            &[0, 0, 0, 1, 0x67, 1, 2],
            &[0, 0, 1, 0x68, 3],
            &[0, 0, 0, 1, 0x65, 4, 5, 6],
        ]);
        assert_eq!(
            annex_b_to_avcc(&au, &[]),
            cat(&[
                &[0, 0, 0, 3, 0x67, 1, 2],
                &[0, 0, 0, 2, 0x68, 3],
                &[0, 0, 0, 4, 0x65, 4, 5, 6]
            ])
        );
    }

    #[test]
    fn annex_b_to_avcc_drops_the_named_nal_types() {
        let au = cat(&[
            &[0, 0, 0, 1, 0x09, 0xf0],
            &[0, 0, 0, 1, 0x67, 1],
            &[0, 0, 1, 0x68, 2],
            &[0, 0, 1, 0x65, 3],
        ]);
        assert_eq!(annex_b_to_avcc(&au, &[9, 7, 8]), vec![0, 0, 0, 2, 0x65, 3]);
    }

    #[test]
    fn empty_input_converts_to_empty_output() {
        assert!(annex_b_to_avcc(&[], &[]).is_empty());
        assert_eq!(avcc_to_annex_b(&[], 4), Ok(Vec::new()));
    }

    #[test]
    fn emulation_prevention_bytes_are_kept() {
        let au = [0, 0, 1, 0x65, 0, 0, 3, 1];
        assert_eq!(
            annex_b_to_avcc(&au, &[]),
            vec![0, 0, 0, 5, 0x65, 0, 0, 3, 1]
        );
    }

    #[test]
    fn avcc_converts_to_annex_b_at_each_length_size() {
        assert_eq!(
            avcc_to_annex_b(&[0, 0, 0, 2, 0x65, 9], 4).unwrap(),
            vec![0, 0, 0, 1, 0x65, 9]
        );
        assert_eq!(
            avcc_to_annex_b(&[0, 2, 0x41, 9, 0, 1, 0x06], 2).unwrap(),
            vec![0, 0, 0, 1, 0x41, 9, 0, 0, 0, 1, 0x06]
        );
        assert_eq!(
            avcc_to_annex_b(&[1, 0x09], 1).unwrap(),
            vec![0, 0, 0, 1, 0x09]
        );
    }

    #[test]
    fn a_truncated_or_overlong_avcc_sample_is_an_error() {
        assert!(avcc_to_annex_b(&[0, 0, 0], 4).is_err());
        assert!(avcc_to_annex_b(&[0, 0, 0, 9, 0x65], 4).is_err());
        assert!(avcc_to_annex_b(&[0, 1, 0x65], 3).is_err());
    }

    #[test]
    fn access_unit_inputs_round_trip_through_avcc_with_4_byte_start_codes() {
        let p_slice2: &[u8] = &[0, 0, 1, 0x41, 0x40, 0x11];
        let aud: &[u8] = &[0, 0, 0, 1, 0x09, 0xf0];
        let eos: &[u8] = &[0, 0, 0, 1, 0x0a];
        let streams: [Vec<&[u8]>; 2] = [
            vec![SPS, PPS, IDR, P_SLICE, p_slice2, aud, SEI, P_SLICE],
            vec![IDR, eos, SPS, PPS, IDR],
        ];
        for pieces in streams {
            let expected: Vec<u8> = pieces
                .iter()
                .flat_map(|nal| {
                    let body = nal
                        .strip_prefix(&[0, 0, 0, 1])
                        .or_else(|| nal.strip_prefix(&[0, 0, 1]))
                        .unwrap();
                    [&[0, 0, 0, 1][..], body].concat()
                })
                .collect();
            let stream = pieces.concat();
            let mut back = Vec::new();
            for au in access_units(&stream) {
                let converted = avcc_to_annex_b(&annex_b_to_avcc(au, &[]), 4).unwrap();
                assert_eq!(nal_types(&converted), nal_types(au));
                back.extend_from_slice(&converted);
            }
            assert_eq!(back, expected);
        }
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
    fn splits_access_units_at_headers_and_new_pictures() {
        // A second slice of the same picture: first_mb_in_slice = 1 (ue "010").
        let p_slice2: &[u8] = &[0, 0, 1, 0x41, 0x40, 0x11];
        let aud: &[u8] = &[0, 0, 0, 1, 0x09, 0xf0];
        let stream = cat(&[SPS, PPS, IDR, P_SLICE, p_slice2, aud, SEI, P_SLICE]);
        let aus: Vec<&[u8]> = access_units(&stream).collect();
        assert_eq!(aus.len(), 3);
        assert_eq!(aus[0], cat(&[SPS, PPS, IDR]).as_slice());
        assert_eq!(aus[1], cat(&[P_SLICE, p_slice2]).as_slice());
        assert_eq!(aus[2], cat(&[aud, SEI, P_SLICE]).as_slice());
        assert_eq!(aus.concat(), stream, "nothing is lost or duplicated");
    }

    #[test]
    fn end_of_sequence_stays_with_its_picture() {
        let eos: &[u8] = &[0, 0, 0, 1, 0x0a];
        let stream = cat(&[IDR, eos, SPS, PPS, IDR]);
        let aus: Vec<&[u8]> = access_units(&stream).collect();
        assert_eq!(aus, vec![cat(&[IDR, eos]), cat(&[SPS, PPS, IDR])]);
    }

    #[test]
    fn access_units_of_degenerate_input() {
        assert_eq!(access_units(&[]).count(), 0);
        assert_eq!(access_units(&[1, 2, 3]).count(), 0);
        assert_eq!(
            access_units(&cat(&[&[0xaa], SPS])).collect::<Vec<_>>(),
            vec![SPS]
        );
        // A start code with no NAL header after it ends the last unit.
        let tail = cat(&[IDR, &[0, 0, 1]]);
        assert_eq!(
            access_units(&tail).collect::<Vec<_>>(),
            vec![tail.as_slice()]
        );
    }

    #[test]
    fn a_nal_unit_ends_where_the_next_start_code_begins() {
        let stream = cat(&[SPS, IDR, P_SLICE]);
        assert_eq!(nal_end(&stream, 0), SPS.len());
        assert_eq!(nal_end(&stream, SPS.len()), SPS.len() + IDR.len());
        assert_eq!(nal_end(&stream, SPS.len() + IDR.len()), stream.len());
        assert_eq!(nal_end(&[1, 2, 3], 0), 3);
    }

    #[test]
    fn parameter_sets_follow_an_access_unit_delimiter() {
        let sps_pps = cat(&[SPS, PPS]);
        let aud: &[u8] = &[0, 0, 0, 1, 0x09, 0xf0];
        let au = cat(&[aud, IDR]);
        assert_eq!(
            with_parameter_sets(&au, &sps_pps),
            cat(&[aud, SPS, PPS, IDR])
        );
        assert_eq!(with_parameter_sets(IDR, &sps_pps), cat(&[SPS, PPS, IDR]));
        let short_aud: &[u8] = &[0, 0, 1, 0x09, 0xf0];
        let au = cat(&[short_aud, IDR]);
        assert_eq!(
            with_parameter_sets(&au, &sps_pps),
            cat(&[short_aud, SPS, PPS, IDR])
        );
    }

    #[test]
    fn header_only_buffer() {
        let au = AccessUnit::parse(&cat(&[SPS, PPS]));
        assert!(au.parameter_sets && !au.has_picture && !au.ends_previous());
        assert!(!AccessUnit::parse(SPS).parameter_sets);
        assert!(!AccessUnit::parse(SEI).has_picture);
    }
}
