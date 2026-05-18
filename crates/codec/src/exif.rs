// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! EXIF orientation parsing and in-place pixel rotation/flip.
//!
//! Used by both JPEG and PNG decode paths. The orientation tag follows the
//! standard EXIF mapping (1 = identity, 3 = 180°, 6 = 90° CW, 8 = 270° CW,
//! 2/4/5/7 = same with horizontal flip).

/// Read EXIF orientation tag and return `(rotation_degrees, flip_horizontal)`.
///
/// Accepts both the JPEG APP1 segment payload (which starts with the
/// `"Exif\0\0"` identifier before the TIFF header) and a raw TIFF block
/// (as carried in a PNG `eXIf` chunk). When the identifier is present it
/// is stripped before parsing so kamadak-exif sees the byte-order marker
/// (`MM` or `II`) as the first bytes.
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
    match orient.value.get_uint(0).unwrap_or(1) {
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

/// Apply EXIF rotation/flip to a contiguous pixel buffer.
///
/// The buffer is treated as `*h` rows of `*w` pixels each, with `bytes_per_pixel`
/// bytes per pixel (so it works for u8 RGB at `bytes_per_pixel=3`, u8 RGBA at
/// `bytes_per_pixel=4`, u16 RGB at `bytes_per_pixel=6`, etc.). After a 90°/270°
/// rotation, `*w` and `*h` are swapped in place.
///
/// `stride` is the byte stride between consecutive rows in `data` on input
/// (may exceed `*w * bytes_per_pixel` for pitch-padded buffers). After a
/// 90°/270° rotation the buffer is rewritten at the tight stride
/// `new_w * bytes_per_pixel`; 180° rotation and flip happen in-place and
/// preserve the input stride. `scratch` is grown to
/// `new_w * new_h * bytes_per_pixel` for 90°/270° rotations and is reused
/// across calls.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_exif_u8(
    data: &mut [u8],
    stride: usize,
    w: &mut usize,
    h: &mut usize,
    bytes_per_pixel: usize,
    rotation_deg: u16,
    flip_h: bool,
    scratch: &mut Vec<u8>,
) {
    let img_w = *w;
    let img_h = *h;

    if flip_h {
        for y in 0..img_h {
            let row_start = y * stride;
            for x in 0..img_w / 2 {
                let left = row_start + x * bytes_per_pixel;
                let right = row_start + (img_w - 1 - x) * bytes_per_pixel;
                for c in 0..bytes_per_pixel {
                    data.swap(left + c, right + c);
                }
            }
        }
    }

    match rotation_deg {
        90 => {
            let new_w = img_h;
            let new_h = img_w;
            scratch.resize(new_w * new_h * bytes_per_pixel, 0);
            for y in 0..img_h {
                for x in 0..img_w {
                    let src_off = y * stride + x * bytes_per_pixel;
                    let dst_x = img_h - 1 - y;
                    let dst_y = x;
                    let dst_off = dst_y * new_w * bytes_per_pixel + dst_x * bytes_per_pixel;
                    scratch[dst_off..dst_off + bytes_per_pixel]
                        .copy_from_slice(&data[src_off..src_off + bytes_per_pixel]);
                }
            }
            data[..scratch.len()].copy_from_slice(scratch);
            *w = new_w;
            *h = new_h;
        }
        180 => {
            // In-place 180° rotation: swap pixel (x, y) with (W-1-x, H-1-y).
            // Iterate over the first half of the buffer in raster order so
            // each pair is visited exactly once; for odd img_w * img_h the
            // central pixel maps to itself and is skipped by total / 2.
            let total = img_w * img_h;
            for i in 0..total / 2 {
                let j = total - 1 - i;
                let a_y = i / img_w;
                let a_x = i % img_w;
                let b_y = j / img_w;
                let b_x = j % img_w;
                let a = a_y * stride + a_x * bytes_per_pixel;
                let b = b_y * stride + b_x * bytes_per_pixel;
                for c in 0..bytes_per_pixel {
                    data.swap(a + c, b + c);
                }
            }
        }
        270 => {
            let new_w = img_h;
            let new_h = img_w;
            scratch.resize(new_w * new_h * bytes_per_pixel, 0);
            for y in 0..img_h {
                for x in 0..img_w {
                    let src_off = y * stride + x * bytes_per_pixel;
                    let dst_x = y;
                    let dst_y = img_w - 1 - x;
                    let dst_off = dst_y * new_w * bytes_per_pixel + dst_x * bytes_per_pixel;
                    scratch[dst_off..dst_off + bytes_per_pixel]
                        .copy_from_slice(&data[src_off..src_off + bytes_per_pixel]);
                }
            }
            data[..scratch.len()].copy_from_slice(scratch);
            *w = new_w;
            *h = new_h;
        }
        _ => {}
    }
}

/// Return the post-rotation `(width, height)` given source dims and rotation.
pub(crate) fn rotated_dims(width: usize, height: usize, rotation_deg: u16) -> (usize, usize) {
    match rotation_deg {
        90 | 270 => (height, width),
        _ => (width, height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_orientation_default() {
        assert_eq!(read_exif_orientation(&[]), (0, false));
    }

    #[test]
    fn rotated_dims_swap() {
        assert_eq!(rotated_dims(10, 20, 0), (10, 20));
        assert_eq!(rotated_dims(10, 20, 90), (20, 10));
        assert_eq!(rotated_dims(10, 20, 180), (10, 20));
        assert_eq!(rotated_dims(10, 20, 270), (20, 10));
    }

    /// Build a 3-wide × 2-tall RGB image at the given stride (bytes per row,
    /// at least 9). Pixel (x, y) has channel value `(y * 3 + x) * 10 + c`;
    /// padding bytes are `0xFF` so test failures show up obviously.
    fn make_strided_rgb(stride: usize) -> Vec<u8> {
        assert!(stride >= 9);
        let mut buf = vec![0xFF; stride * 2];
        for y in 0..2usize {
            for x in 0..3usize {
                let off = y * stride + x * 3;
                let base = ((y * 3 + x) * 10) as u8;
                buf[off] = base;
                buf[off + 1] = base + 1;
                buf[off + 2] = base + 2;
            }
        }
        buf
    }

    /// Tight RGB pixel value at logical (x, y) in the 3×2 source image.
    fn rgb_at(buf: &[u8], stride: usize, x: usize, y: usize) -> [u8; 3] {
        let off = y * stride + x * 3;
        [buf[off], buf[off + 1], buf[off + 2]]
    }

    #[test]
    fn rotate_180_respects_strided_input() {
        // Padded stride: 12 bytes/row instead of the tight 9.
        let stride = 12;
        let mut buf = make_strided_rgb(stride);
        let mut scratch = Vec::new();
        let mut w = 3usize;
        let mut h = 2usize;
        apply_exif_u8(
            &mut buf,
            stride,
            &mut w,
            &mut h,
            3,
            180,
            false,
            &mut scratch,
        );
        // 180° rotation in place: (x,y) <-> (W-1-x, H-1-y). Stride preserved.
        assert_eq!((w, h), (3, 2));
        for y in 0..2 {
            for x in 0..3 {
                let src_base = (((1 - y) * 3 + (2 - x)) * 10) as u8;
                assert_eq!(
                    rgb_at(&buf, stride, x, y),
                    [src_base, src_base + 1, src_base + 2],
                    "180° mismatch at ({x},{y}) with strided input"
                );
            }
        }
    }

    #[test]
    fn rotate_90_respects_strided_input() {
        // Padded stride: 12 bytes/row.
        let stride = 12;
        let mut buf = make_strided_rgb(stride);
        let mut scratch = Vec::new();
        let mut w = 3usize;
        let mut h = 2usize;
        apply_exif_u8(&mut buf, stride, &mut w, &mut h, 3, 90, false, &mut scratch);
        // 90° CW: dims swap to 2×3. Output is tight at `new_w * 3 = 6` bytes/row.
        // (x', y') = (H - 1 - y, x) — so source (x, y) lands at destination
        // (1 - y, x).
        assert_eq!((w, h), (2, 3));
        let new_stride = 6;
        for src_y in 0..2 {
            for src_x in 0..3 {
                let dst_x = 1 - src_y;
                let dst_y = src_x;
                let base = ((src_y * 3 + src_x) * 10) as u8;
                assert_eq!(
                    rgb_at(&buf, new_stride, dst_x, dst_y),
                    [base, base + 1, base + 2],
                    "90° mismatch: src ({src_x},{src_y}) -> dst ({dst_x},{dst_y})"
                );
            }
        }
    }
}
