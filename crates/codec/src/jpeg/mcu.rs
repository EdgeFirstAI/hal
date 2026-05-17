// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MCU (Minimum Coded Unit) decode loop.
//!
//! Orchestrates: Huffman decode → IDCT → chroma upsample → color convert →
//! strided output into the destination buffer.

use crate::error::CodecError;
use crate::jpeg::bitstream::BitStream;
use crate::jpeg::color::{self, ColorConvertFn};
use crate::jpeg::huffman::{self, HuffmanTable};
use crate::jpeg::idct::{self, IdctDcOnlyFn, IdctFn};
use crate::jpeg::markers::JpegHeaders;
use crate::jpeg::upsample;
use edgefirst_tensor::PixelFormat;

/// Scratch buffers reused across MCU decode iterations.
pub struct McuScratch {
    /// Per-component IDCT output buffers for one MCU row.
    /// Indexed by component index, each is `mcu_blocks_h * 8` wide ×
    /// `mcu_blocks_v * 8` tall.
    component_bufs: Vec<Vec<u8>>,
    /// Upsampled chroma row buffers (full image width).
    cb_row: Vec<u8>,
    cr_row: Vec<u8>,
    /// Output row buffer for color conversion before writing to tensor.
    output_row: Vec<u8>,
}

impl McuScratch {
    /// Allocate scratch buffers for the given image header.
    pub fn new(headers: &JpegHeaders) -> Self {
        let hdr = &headers.header;
        let _max_h = hdr.max_h_samp as usize;
        let _max_v = hdr.max_v_samp as usize;
        let w = hdr.width as usize;

        let mut component_bufs = Vec::with_capacity(hdr.components.len());
        for comp in &hdr.components {
            let blocks_h = comp.sampling.h as usize;
            let blocks_v = comp.sampling.v as usize;
            // Width of one MCU column of blocks for this component
            let mcu_w = blocks_h * 8;
            let mcu_h = blocks_v * 8;
            // Full MCU row: mcus_x MCU columns × mcu_h rows
            let row_pixels = hdr.mcus_x() * mcu_w;
            let buf_size = row_pixels * mcu_h;
            component_bufs.push(vec![0u8; buf_size]);
        }

        let output_channels = 4; // Max (RGBA)
        Self {
            component_bufs,
            cb_row: vec![0u8; w + 16], // Padding for SIMD
            cr_row: vec![0u8; w + 16],
            output_row: vec![0u8; (w + 16) * output_channels],
        }
    }

    /// Grow buffers if needed (for a larger image than previously seen).
    pub fn ensure_capacity(&mut self, headers: &JpegHeaders) {
        let hdr = &headers.header;
        let w = hdr.width as usize;

        for (i, comp) in hdr.components.iter().enumerate() {
            let blocks_h = comp.sampling.h as usize;
            let blocks_v = comp.sampling.v as usize;
            let row_pixels = hdr.mcus_x() * blocks_h * 8;
            let buf_size = row_pixels * blocks_v * 8;
            if i >= self.component_bufs.len() {
                self.component_bufs.push(vec![0u8; buf_size]);
            } else if self.component_bufs[i].len() < buf_size {
                self.component_bufs[i].resize(buf_size, 0);
            }
        }

        let needed = w + 16;
        if self.cb_row.len() < needed {
            self.cb_row.resize(needed, 0);
        }
        if self.cr_row.len() < needed {
            self.cr_row.resize(needed, 0);
        }
        let output_needed = needed * 4;
        if self.output_row.len() < output_needed {
            self.output_row.resize(output_needed, 0);
        }
    }
}

/// Decode all MCUs and write output pixels into `dst` at `dst_stride` byte
/// offsets.
///
/// `dst` is the mapped tensor buffer (u8 pixels in the target format).
pub fn decode_image(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    dst_stride: usize,
    output_format: PixelFormat,
) -> crate::Result<()> {
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;
    let num_components = hdr.components.len();

    // Select kernel functions
    let idct_fn: IdctFn = idct::select_idct();
    let idct_dc_fn: IdctDcOnlyFn = idct::select_idct_dc_only();

    let is_greyscale = num_components == 1;

    // Validate Huffman tables
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

    // DC prediction values (one per component)
    let mut dc_pred = vec![0i32; num_components];

    // Create bit stream starting at the entropy data
    let mut bs = BitStream::new(data, headers.scan_data_offset);

    let mcus_x = hdr.mcus_x();
    let mcus_y = hdr.mcus_y();
    let max_v = hdr.max_v_samp as usize;
    let restart_interval = headers.restart_interval as usize;
    let mut mcu_count = 0usize;

    // Coefficient buffer for one 8×8 block
    let mut coeffs = [0i32; 64];

    // Process MCU rows
    for mcu_row in 0..mcus_y {
        // Decode all MCUs in this row
        for mcu_col in 0..mcus_x {
            // Check for restart marker
            if restart_interval > 0 && mcu_count > 0 && mcu_count.is_multiple_of(restart_interval) {
                bs.skip_restart_marker();
                dc_pred.fill(0);
            }

            // Decode all blocks in this MCU
            for (ci, comp) in hdr.components.iter().enumerate() {
                let blocks_h = comp.sampling.h as usize;
                let blocks_v = comp.sampling.v as usize;
                let comp_stride = mcus_x * blocks_h * 8;

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

                        // IDCT into component buffer
                        let x_offset = mcu_col * blocks_h * 8 + bh * 8;
                        let y_offset = bv * 8;
                        let buf_offset = y_offset * comp_stride + x_offset;
                        let buf = &mut scratch.component_bufs[ci];

                        // Check if DC-only (all AC coefficients are zero)
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

        // After decoding all MCUs in this row, perform upsampling + color
        // conversion and write to the output buffer.
        let mcu_pixel_h = max_v * 8;
        let y_start = mcu_row * mcu_pixel_h;

        if is_greyscale || output_format == PixelFormat::Grey {
            // The Y plane (component_bufs[0]) is stored at native pixel
            // resolution for both 1-component (greyscale) JPEGs and the
            // luma channel of multi-component JPEGs, so the same write
            // path covers both cases — chroma planes are simply skipped.
            let grey_fn = color::select_grey_copy();
            write_greyscale_rows(
                &scratch.component_bufs[0],
                mcus_x * hdr.components[0].sampling.h as usize * 8,
                dst,
                dst_stride,
                y_start,
                mcu_pixel_h.min(img_h - y_start),
                img_w,
                output_format,
                grey_fn,
                &mut scratch.output_row,
            );
        } else if output_format == PixelFormat::Nv12 {
            write_nv12_rows(
                hdr,
                &scratch.component_bufs,
                mcus_x,
                dst,
                dst_stride,
                y_start,
                mcu_pixel_h.min(img_h - y_start),
                img_w,
                img_h,
            );
        } else {
            let color_fn = color::select_color_convert(output_format)
                .ok_or(CodecError::UnsupportedFormat(output_format))?;
            let upsample_h_fn = upsample::select_upsample_h();

            write_color_rows(
                hdr,
                &scratch.component_bufs,
                mcus_x,
                dst,
                dst_stride,
                y_start,
                mcu_pixel_h.min(img_h - y_start),
                img_w,
                output_format,
                color_fn,
                upsample_h_fn,
                &mut scratch.cb_row,
                &mut scratch.cr_row,
                &mut scratch.output_row,
            );
        }
    }

    Ok(())
}

/// Write greyscale rows from the Y component buffer to the output.
#[allow(clippy::too_many_arguments)]
fn write_greyscale_rows(
    y_buf: &[u8],
    y_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    format: PixelFormat,
    grey_fn: color::GreyCopyFn,
    output_row: &mut [u8],
) {
    let channels = format.channels();
    for row in 0..num_rows {
        let y_row = &y_buf[row * y_stride..row * y_stride + img_w];
        let dst_offset = (y_start + row) * dst_stride;

        if format == PixelFormat::Grey {
            grey_fn(y_row, &mut dst[dst_offset..], img_w);
        } else {
            // Expand grey to RGB/RGBA
            for i in 0..img_w {
                let v = y_row[i];
                match channels {
                    3 => {
                        output_row[i * 3] = v;
                        output_row[i * 3 + 1] = v;
                        output_row[i * 3 + 2] = v;
                    }
                    4 => {
                        output_row[i * 4] = v;
                        output_row[i * 4 + 1] = v;
                        output_row[i * 4 + 2] = v;
                        output_row[i * 4 + 3] = 255;
                    }
                    _ => {
                        output_row[i] = v;
                    }
                }
            }
            let row_bytes = img_w * channels;
            dst[dst_offset..dst_offset + row_bytes].copy_from_slice(&output_row[..row_bytes]);
        }
    }
}

/// Write color rows: upsample chroma + convert YCbCr → target format.
#[allow(clippy::too_many_arguments)]
fn write_color_rows(
    hdr: &crate::jpeg::types::ImageHeader,
    comp_bufs: &[Vec<u8>],
    mcus_x: usize,
    dst: &mut [u8],
    dst_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    format: PixelFormat,
    color_fn: ColorConvertFn,
    upsample_h_fn: upsample::UpsampleHFn,
    cb_row_buf: &mut [u8],
    cr_row_buf: &mut [u8],
    output_row: &mut [u8],
) {
    let channels = format.channels();
    let y_comp = &hdr.components[0];
    let cb_comp = &hdr.components[1];
    let cr_comp = &hdr.components[2];

    let y_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let cb_stride = mcus_x * cb_comp.sampling.h as usize * 8;
    let cr_stride = mcus_x * cr_comp.sampling.h as usize * 8;

    let h_ratio = y_comp.sampling.h / cb_comp.sampling.h;
    let v_ratio = y_comp.sampling.v / cb_comp.sampling.v;

    let chroma_w = img_w.div_ceil(h_ratio as usize);

    for row in 0..num_rows {
        // Y row from component buffer
        let y_row = &comp_bufs[0][row * y_stride..];

        // Chroma rows (may be subsampled vertically)
        let chroma_row = row / v_ratio as usize;
        let cb_src = &comp_bufs[1][chroma_row * cb_stride..];
        let cr_src = &comp_bufs[2][chroma_row * cr_stride..];

        // Upsample chroma to full width if needed
        if h_ratio > 1 {
            upsample_h_fn(cb_src, cb_row_buf, chroma_w);
            upsample_h_fn(cr_src, cr_row_buf, chroma_w);
        } else {
            cb_row_buf[..chroma_w].copy_from_slice(&cb_src[..chroma_w]);
            cr_row_buf[..chroma_w].copy_from_slice(&cr_src[..chroma_w]);
        }

        // Color convert
        color_fn(y_row, cb_row_buf, cr_row_buf, output_row, img_w);

        // Write to destination at stride offset
        let dst_offset = (y_start + row) * dst_stride;
        let row_bytes = img_w * channels;
        dst[dst_offset..dst_offset + row_bytes].copy_from_slice(&output_row[..row_bytes]);
    }
}

/// Write NV12 output: Y plane + interleaved UV plane.
///
/// NV12 layout in the destination buffer:
/// - Y plane: `img_h` rows of `img_w` bytes at offset 0
/// - UV plane: `img_h/2` rows of `img_w` bytes (Cb/Cr interleaved) at offset
///   `img_h * dst_stride`
///
/// For 4:2:0 JPEGs, the Cb/Cr components are already at half resolution,
/// so we copy them directly (no upsampling needed).
/// For 4:4:4 JPEGs, we subsample by averaging 2×2 blocks.
#[allow(clippy::too_many_arguments)]
fn write_nv12_rows(
    hdr: &crate::jpeg::types::ImageHeader,
    comp_bufs: &[Vec<u8>],
    mcus_x: usize,
    dst: &mut [u8],
    dst_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    img_h: usize,
) {
    let y_comp = &hdr.components[0];
    let cb_comp = &hdr.components[1];

    let y_comp_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let cb_comp_stride = mcus_x * cb_comp.sampling.h as usize * 8;

    let v_ratio = y_comp.sampling.v / cb_comp.sampling.v;
    let uv_plane_offset = img_h * dst_stride;

    // Copy Y plane rows directly
    for row in 0..num_rows {
        let src_offset = row * y_comp_stride;
        let dst_offset = (y_start + row) * dst_stride;
        let copy_len = img_w.min(y_comp_stride - (src_offset % y_comp_stride));
        dst[dst_offset..dst_offset + img_w.min(copy_len)]
            .copy_from_slice(&comp_bufs[0][src_offset..src_offset + img_w.min(copy_len)]);
    }

    // Write UV plane (interleaved Cb/Cr at half height, half width)
    // Only write UV rows for even Y rows (or when v_ratio == 1, subsample)
    let chroma_h = num_rows / v_ratio as usize;
    let chroma_w = img_w / 2;

    for crow in 0..chroma_h {
        let chroma_src_row = crow;
        let cb_src = &comp_bufs[1][chroma_src_row * cb_comp_stride..];
        let cr_src = &comp_bufs[2][chroma_src_row * cb_comp_stride..];

        let uv_row_idx = y_start / 2 + crow;
        let uv_offset = uv_plane_offset + uv_row_idx * dst_stride;

        // Interleave Cb/Cr pairs
        for x in 0..chroma_w {
            dst[uv_offset + x * 2] = cb_src[x];
            dst[uv_offset + x * 2 + 1] = cr_src[x];
        }
    }
}
