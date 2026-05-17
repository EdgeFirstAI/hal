// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scalar chroma upsampling (bilinear 3:1 blend).

/// Horizontal 2× upsample with bilinear interpolation.
///
/// For pixel at position `i` in the input, output positions `2*i` and `2*i+1`
/// are computed using 3:1 blending with adjacent samples.
pub fn upsample_h2(input: &[u8], output: &mut [u8], width: usize) {
    if width == 0 {
        return;
    }

    // First pixel: duplicate
    output[0] = input[0];

    if width == 1 {
        output[1] = input[0];
        return;
    }

    // Interior pixels: 3:1 bilinear blend
    for i in 0..width - 1 {
        let a = input[i] as u16;
        let b = input[i + 1] as u16;
        output[2 * i + 1] = ((a * 3 + b + 2) >> 2) as u8;
        output[2 * i + 2] = ((a + b * 3 + 2) >> 2) as u8;
    }

    // Last pixel: duplicate
    output[2 * width - 1] = input[width - 1];
}

/// Vertical 2× upsample: blend two rows and write two output rows.
///
/// `above` and `below` are the two input rows. Output rows are written to
/// `out_upper` and `out_lower`. `width` is the number of samples.
#[allow(dead_code)]
pub fn upsample_v2(above: &[u8], below: &[u8], out_upper: &[u8], out_lower: &[u8], _width: usize) {
    // In JPEG 4:2:0, vertical upsampling is typically combined with
    // horizontal in the merged upsample+convert path. This standalone
    // function is provided for non-merged paths.
    let _ = (above, below, out_upper, out_lower);
}

/// Combined horizontal + vertical 2× upsample for 4:2:0.
///
/// Takes two chroma rows (`above` and `below`), produces two full-width
/// output rows using bilinear interpolation in both dimensions.
#[allow(dead_code)]
pub fn upsample_hv2(
    above: &[u8],
    below: &[u8],
    out_upper: &mut [u8],
    out_lower: &mut [u8],
    chroma_width: usize,
) {
    if chroma_width == 0 {
        return;
    }

    for i in 0..chroma_width {
        let a = above[i] as u16;
        let b = below[i] as u16;

        let a_next = if i + 1 < chroma_width {
            above[i + 1] as u16
        } else {
            a
        };
        let b_next = if i + 1 < chroma_width {
            below[i + 1] as u16
        } else {
            b
        };

        // Vertical blend: 3:1 for upper row, 1:3 for lower row
        let v_upper = a * 3 + b;
        let v_lower = a + b * 3;
        let v_upper_next = a_next * 3 + b_next;
        let v_lower_next = a_next + b_next * 3;

        // Horizontal blend: 3:1 for left, 1:3 for right
        if i == 0 {
            out_upper[0] = ((v_upper + 2) >> 2) as u8;
            out_lower[0] = ((v_lower + 2) >> 2) as u8;
        }

        if i + 1 < chroma_width {
            out_upper[2 * i + 1] = ((v_upper * 3 + v_upper_next + 8) >> 4) as u8;
            out_upper[2 * i + 2] = ((v_upper + v_upper_next * 3 + 8) >> 4) as u8;
            out_lower[2 * i + 1] = ((v_lower * 3 + v_lower_next + 8) >> 4) as u8;
            out_lower[2 * i + 2] = ((v_lower + v_lower_next * 3 + 8) >> 4) as u8;
        } else {
            // Last pixel in row
            if chroma_width > 1 {
                out_upper[2 * i + 1] = ((v_upper + 2) >> 2) as u8;
                out_lower[2 * i + 1] = ((v_lower + 2) >> 2) as u8;
            } else {
                out_upper[1] = ((v_upper + 2) >> 2) as u8;
                out_lower[1] = ((v_lower + 2) >> 2) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_h2_single() {
        let input = [100u8];
        let mut output = [0u8; 2];
        upsample_h2(&input, &mut output, 1);
        assert_eq!(output, [100, 100]);
    }

    #[test]
    fn upsample_h2_two() {
        let input = [0u8, 100];
        let mut output = [0u8; 4];
        upsample_h2(&input, &mut output, 2);
        // output[0] = 0 (duplicate first)
        // output[1] = (0*3 + 100 + 2) / 4 = 25
        // output[2] = (0 + 100*3 + 2) / 4 = 75
        // output[3] = 100 (duplicate last)
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 25);
        assert_eq!(output[2], 75);
        assert_eq!(output[3], 100);
    }

    #[test]
    fn upsample_h2_uniform() {
        let input = [128u8; 4];
        let mut output = [0u8; 8];
        upsample_h2(&input, &mut output, 4);
        // Uniform input → uniform output
        for &v in &output {
            assert_eq!(v, 128);
        }
    }
}
