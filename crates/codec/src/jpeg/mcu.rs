// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MCU (Minimum Coded Unit) decode loop.
//!
//! Orchestrates: Huffman decode → IDCT → native output (NV12 with 4:2:0 chroma
//! or GREY) written at strided offsets into the destination buffer. The codec
//! never converts to RGB — colour conversion is `ImageProcessor::convert()`.

use crate::error::CodecError;
use crate::jpeg::bitstream::BitStream;
use crate::jpeg::huffman::{self, HuffmanTable};
use crate::jpeg::idct::{self, IdctDcOnlyFn, IdctFn};
use crate::jpeg::markers::JpegHeaders;
use edgefirst_tensor::PixelFormat;

/// Scratch buffers reused across MCU decode iterations.
pub struct McuScratch {
    /// Per-component IDCT output buffers for one MCU row band. Indexed by
    /// component, each `mcus_x * sampling.h * 8` wide × `sampling.v * 8` tall.
    component_bufs: Vec<Vec<u8>>,
}

impl McuScratch {
    /// Allocate scratch buffers for the given image header.
    pub fn new(headers: &JpegHeaders) -> Self {
        let hdr = &headers.header;
        let mut component_bufs = Vec::with_capacity(hdr.components.len());
        for comp in &hdr.components {
            let mcu_w = comp.sampling.h as usize * 8;
            let mcu_h = comp.sampling.v as usize * 8;
            let row_pixels = hdr.mcus_x() * mcu_w;
            component_bufs.push(vec![0u8; row_pixels * mcu_h]);
        }
        Self { component_bufs }
    }

    /// Grow buffers if needed (for a larger image than previously seen).
    pub fn ensure_capacity(&mut self, headers: &JpegHeaders) {
        let hdr = &headers.header;
        for (i, comp) in hdr.components.iter().enumerate() {
            let row_pixels = hdr.mcus_x() * comp.sampling.h as usize * 8;
            let buf_size = row_pixels * comp.sampling.v as usize * 8;
            if i >= self.component_bufs.len() {
                self.component_bufs.push(vec![0u8; buf_size]);
            } else if self.component_bufs[i].len() < buf_size {
                self.component_bufs[i].resize(buf_size, 0);
            }
        }
    }
}

/// Decode all MCUs and write output pixels into `dst` on a fixed **physical
/// grid** whose row pitch is `grid_row_stride` bytes.
///
/// `grid_row_stride` is the destination's physical row stride — the IOSurface
/// `bytesPerRow` / allocator pitch for a reused pool, which is `>=` the format's
/// natural row width (`even_width = ceil(img_w/2)*2` for semi-planar). The
/// logical image wraps on its natural width while every row is *placed* at
/// `grid_row_stride`, so the same byte layout is produced for a tightly-packed
/// buffer and a padded pool surface. This is the identical split the GPU R8
/// sampling shader applies (logical wrap vs. physical placement), keeping the
/// codec and shader provably in agreement. `output_format` must be `Grey`,
/// `Nv12`, `Nv16`, or `Nv24`.
pub fn decode_image(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
) -> crate::Result<()> {
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;
    let num_components = hdr.components.len();

    // The physical grid must be wide enough to place a full natural row. A
    // luma-only output (greyscale JPEG, or an explicit `Grey` target) writes
    // exactly `img_w` bytes per row, so a tightly-packed odd-width buffer
    // (stride == odd `img_w`) is valid. Semi-planar colour formats interleave
    // full-width UV pairs, so they need `even(img_w)` to keep chroma columns
    // byte-aligned. The bound is computed inline (not bound to a `let`) so it
    // vanishes cleanly in release builds where `debug_assert!` is compiled out.
    debug_assert!(
        grid_row_stride
            >= if num_components == 1 || output_format == PixelFormat::Grey {
                img_w
            } else {
                img_w.next_multiple_of(2)
            },
        "grid_row_stride {grid_row_stride} too small for {output_format:?} at {img_w}x{img_h} \
         (need img_w for luma-only, even(img_w) for semi-planar chroma alignment)",
    );

    let idct_fn: IdctFn = idct::select_idct();
    let idct_dc_fn: IdctDcOnlyFn = idct::select_idct_dc_only();

    let is_greyscale = num_components == 1;

    let dc_tables: Vec<&HuffmanTable> = hdr
        .components
        .iter()
        .map(|c| {
            headers.dc_tables[c.dc_table_id as usize]
                .as_ref()
                .ok_or_else(|| {
                    CodecError::InvalidData(format!("missing DC Huffman table {}", c.dc_table_id))
                })
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let ac_tables: Vec<&HuffmanTable> = hdr
        .components
        .iter()
        .map(|c| {
            headers.ac_tables[c.ac_table_id as usize]
                .as_ref()
                .ok_or_else(|| {
                    CodecError::InvalidData(format!("missing AC Huffman table {}", c.ac_table_id))
                })
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let mut dc_pred = vec![0i32; num_components];
    let mut bs = BitStream::new(data, headers.scan_data_offset);

    let mcus_x = hdr.mcus_x();
    let mcus_y = hdr.mcus_y();
    let max_v = hdr.max_v_samp as usize;
    let restart_interval = headers.restart_interval as usize;
    let mut mcu_count = 0usize;

    let mut coeffs = [0i32; 64];

    for mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if restart_interval > 0 && mcu_count > 0 && mcu_count.is_multiple_of(restart_interval) {
                bs.skip_restart_marker();
                dc_pred.fill(0);
            }

            for (ci, comp) in hdr.components.iter().enumerate() {
                let blocks_h = comp.sampling.h as usize;
                let blocks_v = comp.sampling.v as usize;
                let comp_stride = mcus_x * blocks_h * 8;
                let mcu_col = _mcu_col;

                for bv in 0..blocks_v {
                    for bh in 0..blocks_h {
                        huffman::decode_block(
                            &mut bs,
                            dc_tables[ci],
                            ac_tables[ci],
                            &headers.quant_tables[comp.quant_table_id as usize],
                            &mut coeffs,
                            &mut dc_pred[ci],
                        )?;

                        let x_offset = mcu_col * blocks_h * 8 + bh * 8;
                        let y_offset = bv * 8;
                        let buf_offset = y_offset * comp_stride + x_offset;
                        let buf = &mut scratch.component_bufs[ci];

                        let is_dc_only = coeffs[1..].iter().all(|&c| c == 0);
                        if is_dc_only {
                            idct_dc_fn(coeffs[0], &mut buf[buf_offset..], comp_stride);
                        } else {
                            idct_fn(&coeffs, &mut buf[buf_offset..], comp_stride);
                        }
                    }
                }
            }

            mcu_count += 1;
        }

        let mcu_pixel_h = max_v * 8;
        let y_start = mcu_row * mcu_pixel_h;
        let num_rows = mcu_pixel_h.min(img_h - y_start);

        if is_greyscale || output_format == PixelFormat::Grey {
            // Y plane / luma copy. The Y component is stored at native pixel
            // resolution, so the same copy covers a greyscale JPEG and the
            // luma channel of a colour JPEG written as GREY.
            let y_stride = mcus_x * hdr.components[0].sampling.h as usize * 8;
            write_grey_rows(
                &scratch.component_bufs[0],
                y_stride,
                dst,
                grid_row_stride,
                y_start,
                num_rows,
                img_w,
            );
        } else if output_format == PixelFormat::Nv12 {
            write_nv12_rows(
                hdr,
                &scratch.component_bufs,
                mcus_x,
                dst,
                grid_row_stride,
                y_start,
                num_rows,
                img_w,
                img_h,
            );
        } else if output_format == PixelFormat::Nv16 || output_format == PixelFormat::Nv24 {
            // Native (matching-subsampling) chroma: a direct interleave copy of
            // the IDCT chroma buffers — no `avg_block` resampling.
            write_nv16_nv24_rows(
                hdr,
                &scratch.component_bufs,
                mcus_x,
                dst,
                grid_row_stride,
                y_start,
                num_rows,
                img_w,
                img_h,
                output_format,
            );
        } else {
            return Err(CodecError::UnsupportedFormat(output_format));
        }
    }

    Ok(())
}

/// Copy luma rows into the GREY destination.
#[allow(clippy::too_many_arguments)]
fn write_grey_rows(
    y_buf: &[u8],
    y_stride: usize,
    dst: &mut [u8],
    grid_row_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
) {
    for row in 0..num_rows {
        let s = row * y_stride;
        let d = (y_start + row) * grid_row_stride;
        dst[d..d + img_w].copy_from_slice(&y_buf[s..s + img_w]);
    }
}

/// Average an `xs`×`ys` block of a chroma component plane (row stride
/// `stride`) whose top-left source sample is `(x0, y0)`. Out-of-range samples
/// are skipped so edge blocks remain correct.
fn avg_block(plane: &[u8], stride: usize, x0: usize, y0: usize, xs: usize, ys: usize) -> u8 {
    let mut sum = 0u32;
    let mut n = 0u32;
    for dy in 0..ys {
        for dx in 0..xs {
            let idx = (y0 + dy) * stride + (x0 + dx);
            if idx < plane.len() {
                sum += plane[idx] as u32;
                n += 1;
            }
        }
    }
    (sum / n.max(1)) as u8
}

/// Write NV12 output: full-resolution Y plane + interleaved Cb/Cr downsampled
/// to 4:2:0. Correct for any source subsampling (4:2:0 → passthrough, 4:2:2 →
/// vertical average, 4:4:4 → 2×2 average).
///
/// NV12 layout: `img_h` rows of `img_w` luma bytes at offset 0, then
/// `ceil(img_h/2)` rows of interleaved Cb/Cr bytes at offset
/// `img_h * grid_row_stride`.
#[allow(clippy::too_many_arguments)]
fn write_nv12_rows(
    hdr: &crate::jpeg::types::ImageHeader,
    comp_bufs: &[Vec<u8>],
    mcus_x: usize,
    dst: &mut [u8],
    grid_row_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    img_h: usize,
) {
    let max_h = hdr.max_h_samp as usize;
    let max_v = hdr.max_v_samp as usize;
    let y_comp = &hdr.components[0];
    let cb = &hdr.components[1];

    let y_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let c_stride = mcus_x * cb.sampling.h as usize * 8;

    // Source chroma samples per output (4:2:0) chroma sample.
    let x_samples = ((2 * cb.sampling.h as usize) / max_h).max(1);
    let y_samples = ((2 * cb.sampling.v as usize) / max_v).max(1);

    // Y plane.
    for row in 0..num_rows {
        let s = row * y_stride;
        let d = (y_start + row) * grid_row_stride;
        dst[d..d + img_w].copy_from_slice(&comp_bufs[0][s..s + img_w]);
    }

    // UV plane (4:2:0). The component buffer's local row 0 corresponds to the
    // global source-chroma row `band_src0`.
    let uv_plane_offset = img_h * grid_row_stride;
    // `ceil(img_w/2)` UV pairs per chroma row keeps the rightmost column for odd
    // widths. The destination buffer width is rounded up to even, so the extra
    // pair (`img_w + 1` bytes for odd `img_w`) stays within `grid_row_stride`.
    let chroma_w = img_w.div_ceil(2);
    let band_src0 = (y_start * cb.sampling.v as usize) / max_v;
    let out_cy_start = y_start / 2;
    // Round up so an odd `img_h` keeps its final chroma row (e.g. a 483-tall
    // image needs ceil(483/2) = 242 chroma rows, not 241). Only the last band
    // reaches an odd boundary; intermediate bands end on even MCU heights where
    // `div_ceil(2)` equals `/2`, so this never overlaps the next band.
    let out_cy_end = (y_start + num_rows).div_ceil(2);

    for ocy in out_cy_start..out_cy_end {
        let uv_off = uv_plane_offset + ocy * grid_row_stride;
        let src_y0 = ocy * y_samples - band_src0;
        for ocx in 0..chroma_w {
            let src_x0 = ocx * x_samples;
            let cbv = avg_block(
                &comp_bufs[1],
                c_stride,
                src_x0,
                src_y0,
                x_samples,
                y_samples,
            );
            let crv = avg_block(
                &comp_bufs[2],
                c_stride,
                src_x0,
                src_y0,
                x_samples,
                y_samples,
            );
            dst[uv_off + ocx * 2] = cbv;
            dst[uv_off + ocx * 2 + 1] = crv;
        }
    }
}

/// Write NV16 (4:2:2) or NV24 (4:4:4) output: full-resolution Y plane plus an
/// interleaved Cb/Cr plane at the source's NATIVE chroma resolution — a direct
/// copy, no averaging. The codec only selects these formats when the JPEG's
/// chroma matches (NV16 ⇔ 4:2:2, NV24 ⇔ 4:4:4), so the source chroma buffers
/// are already at the output resolution.
///
/// The destination uses the `[total_h, even_width]` grid (even_width =
/// `ceil(img_w/2)*2`), where each grid row is `grid_row_stride` bytes (≥ even_width;
/// padded for IOSurface). The interleaved UV plane starts after the `img_h`
/// luma rows. A chroma row is `even_width` bytes for NV16 (one grid row) and
/// `2*even_width` bytes for NV24 (two grid rows). Each UV byte is placed by
/// mapping its linear offset to a `(grid_row, col)` cell, so NV24's two-row
/// wrap is correct for any `grid_row_stride` (matches the GPU R8 sampling shader).
#[allow(clippy::too_many_arguments)]
fn write_nv16_nv24_rows(
    hdr: &crate::jpeg::types::ImageHeader,
    comp_bufs: &[Vec<u8>],
    mcus_x: usize,
    dst: &mut [u8],
    grid_row_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    img_h: usize,
    output_format: PixelFormat,
) {
    let y_comp = &hdr.components[0];
    let cb = &hdr.components[1];
    let y_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let c_stride = mcus_x * cb.sampling.h as usize * 8;

    // Y plane — full-resolution copy.
    for row in 0..num_rows {
        let s = row * y_stride;
        let d = (y_start + row) * grid_row_stride;
        dst[d..d + img_w].copy_from_slice(&comp_bufs[0][s..s + img_w]);
    }

    // Both keep full-height chroma (one chroma row per luma row); they differ in
    // horizontal resolution.
    // NV16 (4:2:2): one chroma row per luma row, each row is grid_row_stride
    //   bytes, holding img_w/2 (Cb,Cr) pairs.
    // NV24 (4:4:4): two chroma rows per luma row (2W bytes of UV per luma row),
    //   each physical row is grid_row_stride bytes, holding img_w (Cb,Cr) pairs.
    let chroma_cols = match output_format {
        PixelFormat::Nv16 => img_w.div_ceil(2), // 4:2:2
        _ => img_w,                             // 4:4:4
    };
    // How many grid rows the UV plane advances per luma row:
    //   NV16: 1 grid row (UV bytes_per_luma_row == grid_row_stride)
    //   NV24: 2 grid rows (UV bytes_per_luma_row == 2*grid_row_stride)
    let uv_grid_rows_per_luma = match output_format {
        PixelFormat::Nv16 => 1usize,
        _ => 2usize,
    };
    let uv_plane_offset = img_h * grid_row_stride;

    let cb_buf = &comp_bufs[1];
    let cr_buf = &comp_bufs[2];
    for row in 0..num_rows {
        let oy = y_start + row; // chroma row == luma row (full-height chroma)
        let src = row * c_stride;
        // Each (Cb, Cr) pair at column ocx occupies two consecutive bytes on the
        // same grid row.  For NV24 the UV for luma row `oy` spans two grid rows
        // starting at grid row `oy * 2`.
        //
        // Grid row for the (ocx * 2)th UV byte of luma row oy:
        //   grid_row = oy * uv_grid_rows_per_luma + (ocx * 2) / grid_row_stride
        //   col      = (ocx * 2) % grid_row_stride
        // Because ocx * 2 < img_w * 2 <= grid_row_stride * uv_grid_rows_per_luma
        // (the assertion at the function top guarantees grid_row_stride >= even_width)
        // the grid_row never overflows for either format.
        for ocx in 0..chroma_cols {
            let linear_uv = oy * uv_grid_rows_per_luma * grid_row_stride + ocx * 2;
            let grid_row = linear_uv / grid_row_stride;
            let col = linear_uv % grid_row_stride;
            let off = uv_plane_offset + grid_row * grid_row_stride + col;
            dst[off] = cb_buf[src + ocx];
            dst[off + 1] = cr_buf[src + ocx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avg_block_2x2() {
        // 4×4 plane.
        let p = [
            10u8, 20, 30, 40, //
            50, 60, 70, 80, //
            12, 12, 12, 12, //
            12, 12, 12, 12,
        ];
        assert_eq!(
            avg_block(&p, 4, 0, 0, 2, 2),
            ((10 + 20 + 50 + 60) / 4) as u8
        ); // 35
        assert_eq!(
            avg_block(&p, 4, 2, 0, 2, 2),
            ((30 + 40 + 70 + 80) / 4) as u8
        ); // 55
        assert_eq!(avg_block(&p, 4, 0, 2, 2, 2), 12);
    }

    #[test]
    fn avg_block_1x1_passthrough() {
        let p = [5u8, 6, 7, 8];
        assert_eq!(avg_block(&p, 2, 1, 1, 1, 1), 8);
        assert_eq!(avg_block(&p, 2, 0, 0, 1, 1), 5);
    }

    fn test_jpeg(name: &str) -> Vec<u8> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("testdata")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    /// The fixed-grid contract: decoding into a buffer whose physical row pitch
    /// is padded beyond the natural width must place the same pixel bytes,
    /// row-for-row, as a tightly-packed decode. This is what lets the profiler
    /// reuse one oversized pool surface (padded `bytesPerRow`) for every frame.
    #[test]
    fn decode_padded_grid_matches_tight() {
        let jpeg = test_jpeg("zidane.jpg"); // 1280×720, NV12 (4:2:0), even dims
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let fmt = super::super::native_format(&headers).unwrap();
        assert_eq!(fmt, PixelFormat::Nv12);

        let even_w = img_w.next_multiple_of(2);
        // NV12 combined plane height = luma + ceil(h/2) chroma rows; over-size to
        // 2·h rows so both the tight and padded buffers hold the full grid.
        let rows = img_h * 2;
        let tight = even_w;
        let padded = even_w + 64; // physical pitch > natural width

        let mut scratch = McuScratch::new(&headers);
        scratch.ensure_capacity(&headers);
        let mut buf_tight = vec![0u8; rows * tight];
        let mut buf_padded = vec![0u8; rows * padded];

        decode_image(&jpeg, &headers, &mut scratch, &mut buf_tight, tight, fmt).unwrap();
        decode_image(&jpeg, &headers, &mut scratch, &mut buf_padded, padded, fmt).unwrap();

        // Luma (img_h rows) + chroma (ceil(img_h/2) rows) live on the same grid,
        // each placed at its stride. The natural-width content of every used row
        // must be byte-identical.
        let used_rows = img_h + img_h.div_ceil(2);
        for gr in 0..used_rows {
            let t = &buf_tight[gr * tight..gr * tight + even_w];
            let p = &buf_padded[gr * padded..gr * padded + even_w];
            assert_eq!(
                t, p,
                "grid row {gr} differs between tight and padded layout"
            );
        }
    }
}
