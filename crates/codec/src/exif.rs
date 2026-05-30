// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! EXIF orientation parsing.
//!
//! The codec reports the source's EXIF orientation in [`ImageInfo`] but never
//! rotates pixels — geometry is applied downstream by `ImageProcessor::convert()`.
//! The orientation tag follows the standard EXIF mapping (1 = identity,
//! 3 = 180°, 6 = 90° CW, 8 = 270° CW, 2/4/5/7 = same with horizontal flip).

/// Map a raw EXIF orientation tag (1..=8) to `(rotation_degrees, flip_horizontal)`.
/// Unknown values map to the identity `(0, false)`.
pub(crate) fn orientation_from_tag(tag: u16) -> (u16, bool) {
    match tag {
        1 => (0, false),
        2 => (0, true),
        3 => (180, false),
        4 => (180, true),
        5 => (270, true),
        6 => (90, false),
        7 => (90, true),
        8 => (270, false),
        _ => (0, false),
    }
}

/// Read the EXIF orientation tag and return `(rotation_degrees, flip_horizontal)`.
///
/// Accepts both the JPEG APP1 segment payload (which starts with the
/// `"Exif\0\0"` identifier before the TIFF header) and a raw TIFF block (as
/// carried in a PNG `eXIf` chunk). When the identifier is present it is
/// stripped before parsing so kamadak-exif sees the byte-order marker
/// (`MM` or `II`) first.
pub(crate) fn read_exif_orientation(exif_data: &[u8]) -> (u16, bool) {
    let tiff_data = if exif_data.starts_with(b"Exif\0\0") {
        &exif_data[6..]
    } else {
        exif_data
    };
    let reader = exif::Reader::new();
    let Ok(parsed) = reader.read_raw(tiff_data.to_vec()) else {
        return (0, false);
    };
    let Some(orient) = parsed.get_field(exif::Tag::Orientation, exif::In::PRIMARY) else {
        return (0, false);
    };
    orientation_from_tag(orient.value.get_uint(0).unwrap_or(1) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_orientation_default() {
        assert_eq!(read_exif_orientation(&[]), (0, false));
    }

    #[test]
    fn orientation_tag_mapping() {
        assert_eq!(orientation_from_tag(1), (0, false));
        assert_eq!(orientation_from_tag(2), (0, true));
        assert_eq!(orientation_from_tag(3), (180, false));
        assert_eq!(orientation_from_tag(4), (180, true));
        assert_eq!(orientation_from_tag(5), (270, true));
        assert_eq!(orientation_from_tag(6), (90, false));
        assert_eq!(orientation_from_tag(7), (90, true));
        assert_eq!(orientation_from_tag(8), (270, false));
        assert_eq!(orientation_from_tag(99), (0, false));
    }
}
