// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! MCU (Minimum Coded Unit) decode loop.
//!
//! Orchestrates: Huffman decode → IDCT → output written at strided offsets
//! into the destination buffer. The default target is the source's native
//! semi-planar format (or GREY); opt-in fused targets convert at the write
//! stage: `Rgb` (4:4:4 and 4:2:0 sources, YCbCr→RGB per row) and `Nv12`
//! (chroma box-average from any subsampling).

use crate::error::CodecError;
use crate::jpeg::bitstream::BitStream;
use crate::jpeg::huffman::{self, HuffmanTable};
use crate::jpeg::idct;
use crate::jpeg::markers::JpegHeaders;
use edgefirst_tensor::PixelFormat;

/// Scratch buffers reused across MCU decode iterations.
pub struct McuScratch {
    /// Per-component IDCT output buffers for one MCU row band. Indexed by
    /// component, each `mcus_x * sampling.h * 8` wide × `sampling.v * 8` tall.
    component_bufs: Vec<Vec<u8>>,
    /// Full-width horizontally-upsampled Cb/Cr rows for the fused 4:2:0→RGB
    /// write. Allocated to `img_w` by `new`; `ensure_capacity` grows them for a
    /// wider image.
    upsample_cb: Vec<u8>,
    upsample_cr: Vec<u8>,
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
        let img_w = hdr.width as usize;
        Self {
            component_bufs,
            upsample_cb: vec![0u8; img_w],
            upsample_cr: vec![0u8; img_w],
        }
    }

    /// Grow the buffers for a larger image.
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
        let img_w = hdr.width as usize;
        if self.upsample_cb.len() < img_w {
            self.upsample_cb.resize(img_w, 0);
            self.upsample_cr.resize(img_w, 0);
        }
    }
}

/// True for the canonical 4:2:0 shape: 3 components, luma sampled 2×2, both
/// chroma components sampled 1×1.
pub(crate) fn is_native_420(hdr: &crate::jpeg::types::ImageHeader) -> bool {
    hdr.components.len() == 3
        && hdr.max_h_samp == 2
        && hdr.max_v_samp == 2
        && hdr.components[0].sampling.h == 2
        && hdr.components[0].sampling.v == 2
        && hdr.components[1].sampling == crate::jpeg::types::SamplingFactor { h: 1, v: 1 }
        && hdr.components[2].sampling == crate::jpeg::types::SamplingFactor { h: 1, v: 1 }
}

/// Decode all MCUs and write output pixels into `dst` on a fixed physical
/// grid whose row pitch is `grid_row_stride` bytes.
///
/// `grid_row_stride` is the destination's physical row stride, `>=` the
/// format's natural row width (`even_width = ceil(img_w/2)*2` for
/// semi-planar). The logical image wraps on its natural width while every
/// row is placed at `grid_row_stride`. `output_format` must be `Grey`,
/// `Nv12`, `Nv16`, `Nv24` or `Rgb`.
#[allow(clippy::too_many_arguments)]
pub fn decode_image(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
    fast_dct: bool,
) -> crate::Result<()> {
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;
    let num_components = hdr.components.len();

    // Fused RGB output is offered for 4:4:4-equivalent sources (every
    // component at the maximum sampling) and native 4:2:0. The grid must hold
    // a full natural row (`img_w` bytes for luma-only output, `even(img_w)`
    // for semi-planar formats) and `dst` must hold every row.
    let is_rgb = output_format == PixelFormat::Rgb;
    if is_rgb {
        let all_match_max = num_components == 3
            && hdr
                .components
                .iter()
                .all(|c| c.sampling.h == hdr.max_h_samp && c.sampling.v == hdr.max_v_samp);
        if !all_match_max && !is_native_420(hdr) {
            return Err(CodecError::UnsupportedFormat(output_format));
        }
    }

    let writes_only_luma = num_components == 1 || output_format == PixelFormat::Grey;
    let min_stride = if is_rgb {
        img_w * 3
    } else if writes_only_luma {
        img_w
    } else {
        img_w.next_multiple_of(2)
    };
    if grid_row_stride < min_stride {
        return Err(CodecError::InvalidData(format!(
            "grid_row_stride {grid_row_stride} < required {min_stride} for {output_format:?} \
             at {img_w}x{img_h}"
        )));
    }
    let total_rows = if writes_only_luma || is_rgb {
        img_h
    } else {
        output_format
            .combined_plane_height(img_h)
            .ok_or(CodecError::UnsupportedFormat(output_format))?
    };
    let needed = grid_row_stride.checked_mul(total_rows).ok_or_else(|| {
        CodecError::InvalidData(format!("decode size overflow at {img_w}x{img_h}"))
    })?;
    if dst.len() < needed {
        return Err(CodecError::InvalidData(format!(
            "destination buffer {} bytes < required {needed} for {output_format:?} \
             {img_w}x{img_h} @ stride {grid_row_stride}",
            dst.len()
        )));
    }

    // Select the IDCT kernel and its dequant tables: the fast (AAN) kernel
    // exists on the NEON path only and takes AAN-prescaled tables.
    #[cfg(target_arch = "aarch64")]
    {
        use crate::jpeg::cpu::{neon_tier, NeonTier};
        if !matches!(neon_tier(), NeonTier::Scalar)
            && std::arch::is_aarch64_feature_detected!("neon")
        {
            if fast_dct {
                let quants: [[u16; 64]; 4] = core::array::from_fn(|i| {
                    idct::fast::derive_fast_quant(&headers.quant_tables[i].values)
                });
                // SAFETY: NEON was probed above.
                return unsafe {
                    decode_image_neon_fast(
                        data,
                        headers,
                        scratch,
                        dst,
                        grid_row_stride,
                        output_format,
                        &quants,
                    )
                };
            }
            let quants: [[u16; 64]; 4] = core::array::from_fn(|i| headers.quant_tables[i].values);
            // SAFETY: NEON was probed above.
            return unsafe {
                decode_image_neon(
                    data,
                    headers,
                    scratch,
                    dst,
                    grid_row_stride,
                    output_format,
                    &quants,
                )
            };
        }
    }
    let _ = fast_dct; // no fast kernels outside the NEON path
    let quants: [[u16; 64]; 4] = core::array::from_fn(|i| headers.quant_tables[i].values);
    // Non-NEON kernels ignore the entropy-derived `last_k` sparsity hint.
    let idct_fn = idct::select_idct();
    decode_image_idct(
        data,
        headers,
        scratch,
        dst,
        grid_row_stride,
        output_format,
        &quants,
        move |coeffs: &[i16; 64], quant: &[u16; 64], _last_k: u8, out: &mut [u8], stride: usize| {
            idct_fn(coeffs, quant, out, stride)
        },
        idct::select_idct_dc_only(),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_image_neon(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
    quants: &[[u16; 64]; 4],
) -> crate::Result<()> {
    decode_image_idct(
        data,
        headers,
        scratch,
        dst,
        grid_row_stride,
        output_format,
        quants,
        crate::jpeg::idct::neon::idct_8x8_neon_k,
        crate::jpeg::idct::neon::idct_dc_only_neon,
    )
}

/// Fast (AAN) kernel binding — `quants` must be AAN-prescaled
/// ([`idct::fast::derive_fast_quant`]).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_image_neon_fast(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
    quants: &[[u16; 64]; 4],
) -> crate::Result<()> {
    decode_image_idct(
        data,
        headers,
        scratch,
        dst,
        grid_row_stride,
        output_format,
        quants,
        crate::jpeg::idct::fast::idct_8x8_neon_fast_k,
        crate::jpeg::idct::fast::idct_dc_only_fast_neon,
    )
}

/// Decode loop for 3 components at 1×1 sampling (4:4:4), writing NV24 or
/// fused RGB per MCU: one Y, one Cb and one Cr block per MCU.
#[allow(clippy::too_many_arguments)]
fn decode_image_444_direct<Idct, IdctDc>(
    data: &[u8],
    headers: &JpegHeaders,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
    quants: &[[u16; 64]; 4],
    idct_fn: Idct,
    idct_dc_fn: IdctDc,
) -> crate::Result<()>
where
    Idct: Copy + Fn(&[i16; 64], &[u16; 64], u8, &mut [u8], usize),
    IdctDc: Copy + Fn(i32, &mut [u8], usize),
{
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;
    let is_rgb = output_format == PixelFormat::Rgb;

    // Resolve the per-component tables.
    let comp_tables = |i: usize| -> crate::Result<(&HuffmanTable, &HuffmanTable, &[u16; 64])> {
        let c = &hdr.components[i];
        let dc = headers.dc_tables[c.dc_table_id as usize]
            .as_ref()
            .ok_or_else(|| {
                CodecError::InvalidData(format!("missing DC Huffman table {}", c.dc_table_id))
            })?;
        let ac = headers.ac_tables[c.ac_table_id as usize]
            .as_ref()
            .ok_or_else(|| {
                CodecError::InvalidData(format!("missing AC Huffman table {}", c.ac_table_id))
            })?;
        Ok((dc, ac, &quants[c.quant_table_id as usize]))
    };
    let (dc_y, ac_y, q_y) = comp_tables(0)?;
    let (dc_cb, ac_cb, q_cb) = comp_tables(1)?;
    let (dc_cr, ac_cr, q_cr) = comp_tables(2)?;

    // Select the block decoder for the CPU tier.
    type DecodeBlockFn = for<'d> fn(
        &mut BitStream<'d>,
        &HuffmanTable,
        &HuffmanTable,
        &mut [i16; 64],
        &mut i32,
        &mut bool,
    ) -> crate::Result<huffman::BlockInfo>;
    let decode_block_fn: DecodeBlockFn = if crate::jpeg::cpu::entropy_pair_decode() {
        huffman::decode_block::<true>
    } else {
        huffman::decode_block::<false>
    };

    let mut dc_pred = [0i32; 3];
    let mut bs = BitStream::with_prefetch(
        data,
        headers.scan_data_offset,
        crate::jpeg::cpu::entropy_prefetch_distance(),
    );

    let mcus_x = hdr.mcus_x();
    let mcus_y = hdr.mcus_y();
    let restart_interval = headers.restart_interval as usize;
    let mut mcu_count = 0usize;

    let mut coeffs = [0i16; 64];
    let mut prev_has_ac = true; // force full clear on first block
    let mut y_blk = [0u8; 64];
    let mut cb_blk = [0u8; 64];
    let mut cr_blk = [0u8; 64];

    for mcu_row in 0..mcus_y {
        let y_start = mcu_row * 8;
        let num_rows = 8usize.min(img_h - y_start);
        let _entropy =
            tracing::trace_span!("codec.decode_jpeg.mcu_row.huffman_idct", row = mcu_row).entered();

        let mut pend_y = [0u8; 64];
        let mut pend_cb = [0u8; 64];
        let mut pend_cr = [0u8; 64];
        let mut pend_x = 0usize;
        let mut has_pend = false;
        for mcu_col in 0..mcus_x {
            if restart_interval > 0 && mcu_count > 0 && mcu_count.is_multiple_of(restart_interval) {
                bs.skip_restart_marker();
                dc_pred = [0i32; 3];
                prev_has_ac = true;
            }

            let x_offset = mcu_col * 8;

            // Y block: straight into dst for interior NV24 MCUs, else via
            // y_blk (fused RGB, or edge clipping).
            let blk = decode_block_fn(
                &mut bs,
                dc_y,
                ac_y,
                &mut coeffs,
                &mut dc_pred[0],
                &mut prev_has_ac,
            )?;
            if !is_rgb {
                let dst_off = y_start * grid_row_stride + x_offset;
                let cols_fit = grid_row_stride.saturating_sub(x_offset).min(8);
                if num_rows == 8 && cols_fit == 8 {
                    idct_into(
                        blk.has_ac,
                        blk.last_k,
                        &coeffs,
                        q_y,
                        &mut dst[dst_off..],
                        grid_row_stride,
                        idct_fn,
                        idct_dc_fn,
                    );
                } else {
                    idct_into(
                        blk.has_ac, blk.last_k, &coeffs, q_y, &mut y_blk, 8, idct_fn, idct_dc_fn,
                    );
                    for r in 0..num_rows {
                        let d = dst_off + r * grid_row_stride;
                        dst[d..d + cols_fit].copy_from_slice(&y_blk[r * 8..r * 8 + cols_fit]);
                    }
                }
            } else {
                idct_into(
                    blk.has_ac, blk.last_k, &coeffs, q_y, &mut y_blk, 8, idct_fn, idct_dc_fn,
                );
            }

            let blk = decode_block_fn(
                &mut bs,
                dc_cb,
                ac_cb,
                &mut coeffs,
                &mut dc_pred[1],
                &mut prev_has_ac,
            )?;
            idct_into(
                blk.has_ac,
                blk.last_k,
                &coeffs,
                q_cb,
                &mut cb_blk,
                8,
                idct_fn,
                idct_dc_fn,
            );

            let blk = decode_block_fn(
                &mut bs,
                dc_cr,
                ac_cr,
                &mut coeffs,
                &mut dc_pred[2],
                &mut prev_has_ac,
            )?;
            idct_into(
                blk.has_ac,
                blk.last_k,
                &coeffs,
                q_cr,
                &mut cr_blk,
                8,
                idct_fn,
                idct_dc_fn,
            );

            // Write stage.
            let x = x_offset;
            let cols = img_w.saturating_sub(x).min(8);
            let rows = num_rows;
            if is_rgb {
                if has_pend {
                    crate::jpeg::color::ycbcr_to_rgb_blocks(
                        &pend_y,
                        &pend_cb,
                        &pend_cr,
                        Some((&y_blk, &cb_blk, &cr_blk)),
                        dst,
                        grid_row_stride,
                        pend_x,
                        y_start,
                        8,
                        cols,
                        rows,
                    );
                    has_pend = false;
                } else if mcu_col + 1 < mcus_x
                    && cols == 8
                    && img_w.saturating_sub(x + 8).min(8) == 8
                {
                    pend_y = y_blk;
                    pend_cb = cb_blk;
                    pend_cr = cr_blk;
                    pend_x = x;
                    has_pend = true;
                } else if cols > 0 && rows > 0 {
                    crate::jpeg::color::ycbcr_to_rgb_blocks(
                        &y_blk,
                        &cb_blk,
                        &cr_blk,
                        None,
                        dst,
                        grid_row_stride,
                        x,
                        y_start,
                        cols,
                        0,
                        rows,
                    );
                }
            } else if cols > 0 && rows > 0 {
                write_nv24_uv_block(
                    &cb_blk,
                    &cr_blk,
                    dst,
                    grid_row_stride,
                    x,
                    y_start,
                    cols,
                    rows,
                    img_h,
                );
            }

            mcu_count += 1;
        }
        debug_assert!(!has_pend, "pending RGB MCU not flushed at end of row");
    }

    if bs.overran() {
        return Err(CodecError::InvalidData(
            "truncated entropy-coded scan".into(),
        ));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_image_idct<Idct, IdctDc>(
    data: &[u8],
    headers: &JpegHeaders,
    scratch: &mut McuScratch,
    dst: &mut [u8],
    grid_row_stride: usize,
    output_format: PixelFormat,
    quants: &[[u16; 64]; 4],
    idct_fn: Idct,
    idct_dc_fn: IdctDc,
) -> crate::Result<()>
where
    Idct: Copy + Fn(&[i16; 64], &[u16; 64], u8, &mut [u8], usize),
    IdctDc: Copy + Fn(i32, &mut [u8], usize),
{
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;
    let num_components = hdr.components.len();
    let is_rgb = output_format == PixelFormat::Rgb;

    let is_greyscale = num_components == 1;

    // 4:4:4 (1×1 sampling on every component) to NV24 or fused RGB: dedicated loop.
    if num_components == 3
        && hdr.max_h_samp == 1
        && hdr.max_v_samp == 1
        && matches!(output_format, PixelFormat::Nv24 | PixelFormat::Rgb)
    {
        return decode_image_444_direct(
            data,
            headers,
            dst,
            grid_row_stride,
            output_format,
            quants,
            idct_fn,
            idct_dc_fn,
        );
    }

    // Fixed-size per-component state, for at most 4 components.
    if num_components > 4 {
        return Err(CodecError::InvalidData(format!(
            "unsupported component count {num_components}"
        )));
    }
    let mut dc_tables: [Option<&HuffmanTable>; 4] = [None; 4];
    let mut ac_tables: [Option<&HuffmanTable>; 4] = [None; 4];
    for (i, c) in hdr.components.iter().enumerate() {
        dc_tables[i] = Some(
            headers.dc_tables[c.dc_table_id as usize]
                .as_ref()
                .ok_or_else(|| {
                    CodecError::InvalidData(format!("missing DC Huffman table {}", c.dc_table_id))
                })?,
        );
        ac_tables[i] = Some(
            headers.ac_tables[c.ac_table_id as usize]
                .as_ref()
                .ok_or_else(|| {
                    CodecError::InvalidData(format!("missing AC Huffman table {}", c.ac_table_id))
                })?,
        );
    }

    // Select the block decoder for the CPU tier (see cpu::entropy_pair_decode).
    type DecodeBlockFn = for<'d> fn(
        &mut BitStream<'d>,
        &HuffmanTable,
        &HuffmanTable,
        &mut [i16; 64],
        &mut i32,
        &mut bool,
    ) -> crate::Result<huffman::BlockInfo>;
    let decode_block_fn: DecodeBlockFn = if crate::jpeg::cpu::entropy_pair_decode() {
        huffman::decode_block::<true>
    } else {
        huffman::decode_block::<false>
    };
    let mut dc_pred = [0i32; 4];
    let mut bs = BitStream::with_prefetch(
        data,
        headers.scan_data_offset,
        crate::jpeg::cpu::entropy_prefetch_distance(),
    );

    let mcus_x = hdr.mcus_x();
    let mcus_y = hdr.mcus_y();
    let max_v = hdr.max_v_samp as usize;
    let restart_interval = headers.restart_interval as usize;
    let mut mcu_count = 0usize;

    let mut coeffs = [0i16; 64];
    let mut prev_has_ac = true; // force full clear on first block

    for mcu_row in 0..mcus_y {
        let mcu_pixel_h = max_v * 8;
        let y_start = mcu_row * mcu_pixel_h;
        let num_rows = mcu_pixel_h.min(img_h - y_start);
        // Full MCU-row bands write Y straight into dst; partial bottom bands
        // go through scratch and a clipped copy.
        let y_direct = !is_greyscale
            && output_format != PixelFormat::Grey
            && !is_rgb
            && num_rows == mcu_pixel_h;

        {
            let _entropy =
                tracing::trace_span!("codec.decode_jpeg.mcu_row.huffman_idct", row = mcu_row)
                    .entered();
            let mut mcu_col = 0usize;
            while mcu_col < mcus_x {
                if restart_interval > 0
                    && mcu_count > 0
                    && mcu_count.is_multiple_of(restart_interval)
                {
                    bs.skip_restart_marker();
                    dc_pred = [0i32; 4];
                    prev_has_ac = true;
                }

                let mut ci = 0usize;
                while ci < num_components {
                    let comp = &hdr.components[ci];
                    let blocks_h = comp.sampling.h as usize;
                    let blocks_v = comp.sampling.v as usize;
                    let comp_stride = mcus_x * blocks_h * 8;

                    let quant = &quants[comp.quant_table_id as usize];

                    let mut bv = 0usize;
                    while bv < blocks_v {
                        let mut bh = 0usize;
                        while bh < blocks_h {
                            let blk = decode_block_fn(
                                &mut bs,
                                dc_tables[ci].expect("resolved above for every component"),
                                ac_tables[ci].expect("resolved above for every component"),
                                &mut coeffs,
                                &mut dc_pred[ci],
                                &mut prev_has_ac,
                            )?;
                            let has_ac = blk.has_ac;
                            let last_k = blk.last_k;

                            let x_offset = mcu_col * blocks_h * 8 + bh * 8;
                            let y_offset = bv * 8;

                            if y_direct && ci == 0 {
                                let dst_off = (y_start + y_offset) * grid_row_stride + x_offset;
                                if x_offset + 8 <= grid_row_stride {
                                    idct_into(
                                        has_ac,
                                        last_k,
                                        &coeffs,
                                        quant,
                                        &mut dst[dst_off..],
                                        grid_row_stride,
                                        idct_fn,
                                        idct_dc_fn,
                                    );
                                } else {
                                    // Right-edge MCU whose 8-wide block extends past the
                                    // physical row: IDCT to a local block, then copy the
                                    // columns that fit. A block wholly past the edge is MCU
                                    // padding and is not written.
                                    let cols = grid_row_stride.saturating_sub(x_offset); // 0..8
                                    if cols > 0 {
                                        let mut tmp = [0u8; 64];
                                        idct_into(
                                            has_ac, last_k, &coeffs, quant, &mut tmp, 8, idct_fn,
                                            idct_dc_fn,
                                        );
                                        for r in 0..8 {
                                            let d = dst_off + r * grid_row_stride;
                                            let end = d + cols;
                                            if end > dst.len() {
                                                // `dst` holds fewer rows than the image
                                                // needs at this stride.
                                                return Err(CodecError::Tensor(
                                                    edgefirst_tensor::Error::InsufficientCapacity {
                                                        needed: end,
                                                        capacity: dst.len(),
                                                    },
                                                ));
                                            }
                                            dst[d..end].copy_from_slice(&tmp[r * 8..r * 8 + cols]);
                                        }
                                    }
                                }
                            } else {
                                let buf_offset = y_offset * comp_stride + x_offset;
                                let buf = &mut scratch.component_bufs[ci];
                                idct_into(
                                    has_ac,
                                    last_k,
                                    &coeffs,
                                    quant,
                                    &mut buf[buf_offset..],
                                    comp_stride,
                                    idct_fn,
                                    idct_dc_fn,
                                );
                            }
                            bh += 1;
                        }
                        bv += 1;
                    }
                    ci += 1;
                }

                mcu_count += 1;
                mcu_col += 1;
            }
        }

        if is_rgb && is_native_420(hdr) {
            let _w =
                tracing::trace_span!("codec.decode_jpeg.write_rgb_420", row = mcu_row).entered();
            let y_stride = mcus_x * hdr.max_h_samp as usize * 8;
            let c_stride = mcus_x * hdr.components[1].sampling.h as usize * 8;
            // Chroma-row offset of the component buffer's local row 0.
            let band_src0 = y_start / 2;
            write_rgb_rows_420(
                &scratch.component_bufs,
                y_stride,
                c_stride,
                &mut scratch.upsample_cb,
                &mut scratch.upsample_cr,
                dst,
                grid_row_stride,
                y_start,
                num_rows,
                img_w,
                band_src0,
            );
        } else if is_rgb {
            let _w = tracing::trace_span!("codec.decode_jpeg.write_rgb", row = mcu_row).entered();
            // All components sample at the maximum rate: one shared
            // full-resolution buffer stride.
            let stride = mcus_x * hdr.max_h_samp as usize * 8;
            write_rgb_rows(
                &scratch.component_bufs,
                stride,
                dst,
                grid_row_stride,
                y_start,
                num_rows,
                img_w,
            );
        } else if is_greyscale || output_format == PixelFormat::Grey {
            let _w = tracing::trace_span!("codec.decode_jpeg.write_grey", row = mcu_row).entered();
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
            let _w = tracing::trace_span!("codec.decode_jpeg.write_nv12", row = mcu_row).entered();
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
                y_direct,
            );
        } else if output_format == PixelFormat::Nv24 {
            let _w = tracing::trace_span!("codec.decode_jpeg.write_nv24", row = mcu_row).entered();
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
                y_direct,
            );
        } else if output_format == PixelFormat::Nv16 {
            let _w = tracing::trace_span!("codec.decode_jpeg.write_nv16", row = mcu_row).entered();
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
                y_direct,
            );
        } else {
            return Err(CodecError::UnsupportedFormat(output_format));
        }
    }

    // Reject a scan that read past the end of its data.
    if bs.overran() {
        return Err(CodecError::InvalidData(
            "truncated entropy-coded scan".into(),
        ));
    }

    Ok(())
}

/// Dequantise the DC coefficient for the DC-only IDCT shortcut, in wrapping
/// `i16` arithmetic as the full kernels do.
#[inline(always)]
fn dc_only_coefficient(coeff: i16, quant: u16) -> i32 {
    coeff.wrapping_mul(quant as i16) as i32
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct_into<Idct, IdctDc>(
    has_ac: bool,
    last_k: u8,
    coeffs: &[i16; 64],
    quant: &[u16; 64],
    dst: &mut [u8],
    stride: usize,
    idct_fn: Idct,
    idct_dc_fn: IdctDc,
) where
    Idct: Copy + Fn(&[i16; 64], &[u16; 64], u8, &mut [u8], usize),
    IdctDc: Copy + Fn(i32, &mut [u8], usize),
{
    if has_ac {
        idct_fn(coeffs, quant, last_k, dst, stride);
    } else {
        let dc = dc_only_coefficient(coeffs[0], quant[0]);
        idct_dc_fn(dc, dst, stride);
    }
}

/// Interleave one 8×8 Cb/Cr block into the NV24 UV plane (4:4:4 1×1).
///
/// UV for luma row `y` starts at `img_h * stride + y * 2 * stride` and is
/// `2*img_w` contiguous bytes.
#[allow(clippy::too_many_arguments)]
#[inline]
fn write_nv24_uv_block(
    cb: &[u8; 64],
    cr: &[u8; 64],
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    cols: usize,
    rows: usize,
    img_h: usize,
) {
    let uv_off = img_h * stride;
    if cols == 8 {
        #[cfg(target_arch = "aarch64")]
        {
            use crate::jpeg::cpu::{neon_tier, NeonTier};
            if !matches!(neon_tier(), NeonTier::Scalar)
                && std::arch::is_aarch64_feature_detected!("neon")
            {
                // SAFETY: 8 Cb + 8 Cr bytes → 16 interleaved destination bytes.
                unsafe {
                    use core::arch::aarch64::{uint8x8x2_t, vld1_u8, vst2_u8};
                    for r in 0..rows {
                        let base = uv_off + (y + r) * 2 * stride + x * 2;
                        let src = r * 8;
                        let b = vld1_u8(cb.as_ptr().add(src));
                        let rv = vld1_u8(cr.as_ptr().add(src));
                        vst2_u8(dst.as_mut_ptr().add(base), uint8x8x2_t(b, rv));
                    }
                }
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if crate::jpeg::cpu::intel_tier().has_sse2() && is_x86_feature_detected!("sse2") {
                // SAFETY: 8+8 source bytes unpack to 16 destination bytes.
                unsafe {
                    use core::arch::x86_64::{
                        _mm_loadl_epi64, _mm_storeu_si128, _mm_unpacklo_epi8,
                    };
                    for r in 0..rows {
                        let base = uv_off + (y + r) * 2 * stride + x * 2;
                        let src = r * 8;
                        let b = _mm_loadl_epi64(cb.as_ptr().add(src) as *const _);
                        let rv = _mm_loadl_epi64(cr.as_ptr().add(src) as *const _);
                        _mm_storeu_si128(
                            dst.as_mut_ptr().add(base) as *mut _,
                            _mm_unpacklo_epi8(b, rv),
                        );
                    }
                }
                return;
            }
        }
    }
    for r in 0..rows {
        let base = uv_off + (y + r) * 2 * stride + x * 2;
        let src = r * 8;
        for c in 0..cols {
            dst[base + c * 2] = cb[src + c];
            dst[base + c * 2 + 1] = cr[src + c];
        }
    }
}

/// Write interleaved RGB rows from the 4:4:4 planar band scratch. All three
/// component buffers share `stride`.
#[allow(clippy::too_many_arguments)]
fn write_rgb_rows(
    comp_bufs: &[Vec<u8>],
    stride: usize,
    dst: &mut [u8],
    grid_row_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
) {
    let (y_buf, cb_buf, cr_buf) = (&comp_bufs[0], &comp_bufs[1], &comp_bufs[2]);
    for row in 0..num_rows {
        let s = row * stride;
        let d = (y_start + row) * grid_row_stride;
        crate::jpeg::color::ycbcr_to_rgb_row(
            &y_buf[s..s + img_w],
            &cb_buf[s..s + img_w],
            &cr_buf[s..s + img_w],
            &mut dst[d..d + img_w * 3],
            img_w,
        );
    }
}

/// Nearest-neighbour horizontal 2× upsample: `src[i]` → `dst[2i]`,
/// `dst[2i+1]`. `dst.len()` may be odd (an odd image width), in which case
/// the last source sample contributes only the final `dst` entry.
#[inline]
fn expand_row_2x(src: &[u8], dst: &mut [u8]) {
    let pairs = dst.len() / 2;
    for i in 0..pairs {
        let v = src[i];
        dst[i * 2] = v;
        dst[i * 2 + 1] = v;
    }
    if dst.len() % 2 == 1 {
        dst[pairs * 2] = src[pairs];
    }
}

/// Write interleaved RGB rows for a native 4:2:0 source (see
/// [`is_native_420`]): a 2×2 nearest-neighbour chroma upsample followed by
/// [`crate::jpeg::color::ycbcr_to_rgb_row`]. Luma is full resolution
/// (`y_stride`-wide band); Cb/Cr are half resolution on both axes
/// (`c_stride`-wide), and one expanded chroma row serves two luma rows.
///
/// `upsample_cb`/`upsample_cr` are scratch rows, each at least `img_w`
/// bytes. `band_src0` is the chroma-row offset of the component buffers'
/// local row 0 in the full-image chroma grid (`y_start / 2`).
#[allow(clippy::too_many_arguments)]
fn write_rgb_rows_420(
    comp_bufs: &[Vec<u8>],
    y_stride: usize,
    c_stride: usize,
    upsample_cb: &mut [u8],
    upsample_cr: &mut [u8],
    dst: &mut [u8],
    grid_row_stride: usize,
    y_start: usize,
    num_rows: usize,
    img_w: usize,
    band_src0: usize,
) {
    let (y_buf, cb_buf, cr_buf) = (&comp_bufs[0], &comp_bufs[1], &comp_bufs[2]);
    let chroma_w = img_w.div_ceil(2);

    let mut row = 0usize;
    while row < num_rows {
        let global_luma_row = y_start + row;
        let c_local_row = global_luma_row / 2 - band_src0;
        let c_off = c_local_row * c_stride;
        expand_row_2x(&cb_buf[c_off..c_off + chroma_w], &mut upsample_cb[..img_w]);
        expand_row_2x(&cr_buf[c_off..c_off + chroma_w], &mut upsample_cr[..img_w]);

        let s0 = row * y_stride;
        let d0 = (y_start + row) * grid_row_stride;
        crate::jpeg::color::ycbcr_to_rgb_row(
            &y_buf[s0..s0 + img_w],
            &upsample_cb[..img_w],
            &upsample_cr[..img_w],
            &mut dst[d0..d0 + img_w * 3],
            img_w,
        );

        if row + 1 < num_rows {
            let s1 = s0 + y_stride;
            let d1 = d0 + grid_row_stride;
            crate::jpeg::color::ycbcr_to_rgb_row(
                &y_buf[s1..s1 + img_w],
                &upsample_cb[..img_w],
                &upsample_cr[..img_w],
                &mut dst[d1..d1 + img_w * 3],
                img_w,
            );
        }
        row += 2;
    }
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
/// are skipped.
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
/// to 4:2:0 from any source subsampling (4:2:0 → passthrough, 4:2:2 →
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
    y_already_written: bool,
) {
    let max_h = hdr.max_h_samp as usize;
    let max_v = hdr.max_v_samp as usize;
    let y_comp = &hdr.components[0];
    let cb = &hdr.components[1];

    let y_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let c_stride = mcus_x * cb.sampling.h as usize * 8;

    let x_samples = ((2 * cb.sampling.h as usize) / max_h).max(1);
    let y_samples = ((2 * cb.sampling.v as usize) / max_v).max(1);

    if !y_already_written {
        for row in 0..num_rows {
            let s = row * y_stride;
            let d = (y_start + row) * grid_row_stride;
            dst[d..d + img_w].copy_from_slice(&comp_bufs[0][s..s + img_w]);
        }
    }

    // UV plane (4:2:0). The component buffer's local row 0 corresponds to the
    // global source-chroma row `band_src0`.
    let uv_plane_offset = img_h * grid_row_stride;
    // `ceil(img_w/2)` UV pairs per chroma row.
    let chroma_w = img_w.div_ceil(2);
    let band_src0 = (y_start * cb.sampling.v as usize) / max_v;
    let out_cy_start = y_start / 2;
    // Rounded up: an odd `img_h` keeps its final chroma row.
    let out_cy_end = (y_start + num_rows).div_ceil(2);

    // Native 4:2:0 (1×1): interleave without averaging.
    if x_samples == 1 && y_samples == 1 {
        for ocy in out_cy_start..out_cy_end {
            let uv_off = uv_plane_offset + ocy * grid_row_stride;
            let src_y0 = ocy - band_src0;
            let src = src_y0 * c_stride;
            interleave_uv_row(
                &comp_bufs[1][src..src + chroma_w],
                &comp_bufs[2][src..src + chroma_w],
                &mut dst[uv_off..uv_off + chroma_w * 2],
                chroma_w,
            );
        }
        return;
    }

    // Structured downsamples: 4:4:4 → 2×2 box average; 4:2:2 → 1×2 vertical
    // average.
    if (x_samples == 2 || x_samples == 1) && y_samples == 2 {
        for ocy in out_cy_start..out_cy_end {
            let uv_off = uv_plane_offset + ocy * grid_row_stride;
            let src_y0 = ocy * 2 - band_src0;
            let r0 = src_y0 * c_stride;
            let r1 = r0 + c_stride;
            let in_w = chroma_w * x_samples;
            let out = &mut dst[uv_off..uv_off + chroma_w * 2];
            if x_samples == 2 {
                downsample_uv_row_2x2(
                    &comp_bufs[1][r0..r0 + in_w],
                    &comp_bufs[1][r1..r1 + in_w],
                    &comp_bufs[2][r0..r0 + in_w],
                    &comp_bufs[2][r1..r1 + in_w],
                    out,
                    chroma_w,
                );
            } else {
                downsample_uv_row_v2(
                    &comp_bufs[1][r0..r0 + in_w],
                    &comp_bufs[1][r1..r1 + in_w],
                    &comp_bufs[2][r0..r0 + in_w],
                    &comp_bufs[2][r1..r1 + in_w],
                    out,
                    chroma_w,
                );
            }
        }
        return;
    }

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

/// 2×2 truncating box-average of two full-resolution chroma rows into one
/// interleaved UV row (`n` Cb/Cr pairs from `2·n` input columns per row).
#[inline]
fn downsample_uv_row_2x2(cb0: &[u8], cb1: &[u8], cr0: &[u8], cr1: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(
        cb0.len() >= n * 2 && cb1.len() >= n * 2 && cr0.len() >= n * 2 && cr1.len() >= n * 2
    );
    debug_assert!(dst.len() >= n * 2);
    #[cfg_attr(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        allow(unused_mut)
    )]
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    if !matches!(
        crate::jpeg::cpu::neon_tier(),
        crate::jpeg::cpu::NeonTier::Scalar
    ) && std::arch::is_aarch64_feature_detected!("neon")
    {
        // SAFETY: bounds asserted above; the loop reads 16 input bytes and
        // writes 16 output bytes per 8 pairs.
        unsafe {
            use core::arch::aarch64::*; // NOSONAR
            while i + 8 <= n {
                let s = i * 2;
                // Pairwise-add within each row (u8→u16), add rows, then a truncating >>2.
                let b = vaddq_u16(
                    vpaddlq_u8(vld1q_u8(cb0.as_ptr().add(s))),
                    vpaddlq_u8(vld1q_u8(cb1.as_ptr().add(s))),
                );
                let r = vaddq_u16(
                    vpaddlq_u8(vld1q_u8(cr0.as_ptr().add(s))),
                    vpaddlq_u8(vld1q_u8(cr1.as_ptr().add(s))),
                );
                let b8 = vmovn_u16(vshrq_n_u16::<2>(b));
                let r8 = vmovn_u16(vshrq_n_u16::<2>(r));
                vst2_u8(dst.as_mut_ptr().add(i * 2), uint8x8x2_t(b8, r8));
                i += 8;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    if crate::jpeg::cpu::intel_tier().has_sse2() && is_x86_feature_detected!("sse2") {
        // SAFETY: bounds asserted above.
        unsafe {
            downsample_uv_row_2x2_sse2(cb0, cb1, cr0, cr1, dst, n, &mut i);
        }
    }
    for ocx in i..n {
        let s = ocx * 2;
        let b = (cb0[s] as u32 + cb0[s + 1] as u32 + cb1[s] as u32 + cb1[s + 1] as u32) / 4;
        let r = (cr0[s] as u32 + cr0[s + 1] as u32 + cr1[s] as u32 + cr1[s + 1] as u32) / 4;
        dst[ocx * 2] = b as u8;
        dst[ocx * 2 + 1] = r as u8;
    }
}

/// 1×2 truncating vertical average of two half-width chroma rows into one
/// interleaved UV row (`n` Cb/Cr pairs, one input column per pair per row).
#[inline]
fn downsample_uv_row_v2(cb0: &[u8], cb1: &[u8], cr0: &[u8], cr1: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(cb0.len() >= n && cb1.len() >= n && cr0.len() >= n && cr1.len() >= n);
    debug_assert!(dst.len() >= n * 2);
    #[cfg_attr(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        allow(unused_mut)
    )]
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    if !matches!(
        crate::jpeg::cpu::neon_tier(),
        crate::jpeg::cpu::NeonTier::Scalar
    ) && std::arch::is_aarch64_feature_detected!("neon")
    {
        // SAFETY: bounds asserted above.
        unsafe {
            use core::arch::aarch64::*; // NOSONAR
            while i + 16 <= n {
                // vhaddq: truncating halving add, (a+b)/2.
                let b = vhaddq_u8(vld1q_u8(cb0.as_ptr().add(i)), vld1q_u8(cb1.as_ptr().add(i)));
                let r = vhaddq_u8(vld1q_u8(cr0.as_ptr().add(i)), vld1q_u8(cr1.as_ptr().add(i)));
                vst2q_u8(dst.as_mut_ptr().add(i * 2), uint8x16x2_t(b, r));
                i += 16;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    if crate::jpeg::cpu::intel_tier().has_sse2() && is_x86_feature_detected!("sse2") {
        unsafe {
            downsample_uv_row_v2_sse2(cb0, cb1, cr0, cr1, dst, n, &mut i);
        }
    }
    for ocx in i..n {
        dst[ocx * 2] = ((cb0[ocx] as u32 + cb1[ocx] as u32) / 2) as u8;
        dst[ocx * 2 + 1] = ((cr0[ocx] as u32 + cr1[ocx] as u32) / 2) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn downsample_uv_row_2x2_sse2(
    cb0: &[u8],
    cb1: &[u8],
    cr0: &[u8],
    cr1: &[u8],
    dst: &mut [u8],
    n: usize,
    i: &mut usize,
) {
    use core::arch::x86_64::*;
    let ones = _mm_set1_epi16(1);
    let zero = _mm_setzero_si128();
    while *i + 8 <= n {
        let s = *i * 2;
        let avg = |row0: &[u8], row1: &[u8]| -> [u8; 8] {
            let a = _mm_loadu_si128(row0.as_ptr().add(s) as *const __m128i);
            let b = _mm_loadu_si128(row1.as_ptr().add(s) as *const __m128i);
            let a_lo = _mm_unpacklo_epi8(a, zero);
            let a_hi = _mm_unpackhi_epi8(a, zero);
            let b_lo = _mm_unpacklo_epi8(b, zero);
            let b_hi = _mm_unpackhi_epi8(b, zero);
            let s0 = _mm_srai_epi32::<2>(_mm_add_epi32(
                _mm_madd_epi16(a_lo, ones),
                _mm_madd_epi16(b_lo, ones),
            ));
            let s1 = _mm_srai_epi32::<2>(_mm_add_epi32(
                _mm_madd_epi16(a_hi, ones),
                _mm_madd_epi16(b_hi, ones),
            ));
            // Narrow i32→i16 then i16→u8 (values are 0..255).
            let packed16 = _mm_packs_epi32(s0, s1);
            let packed8 = _mm_packus_epi16(packed16, zero);
            let mut out = [0u8; 16];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, packed8);
            let mut eight = [0u8; 8];
            eight.copy_from_slice(&out[..8]);
            eight
        };
        let bv = avg(cb0, cb1);
        let rv = avg(cr0, cr1);
        for k in 0..8 {
            dst[(*i + k) * 2] = bv[k];
            dst[(*i + k) * 2 + 1] = rv[k];
        }
        *i += 8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn downsample_uv_row_v2_sse2(
    cb0: &[u8],
    cb1: &[u8],
    cr0: &[u8],
    cr1: &[u8],
    dst: &mut [u8],
    n: usize,
    i: &mut usize,
) {
    use core::arch::x86_64::*;
    while *i + 16 <= n {
        let b0 = _mm_loadu_si128(cb0.as_ptr().add(*i) as *const __m128i);
        let b1 = _mm_loadu_si128(cb1.as_ptr().add(*i) as *const __m128i);
        let r0 = _mm_loadu_si128(cr0.as_ptr().add(*i) as *const __m128i);
        let r1 = _mm_loadu_si128(cr1.as_ptr().add(*i) as *const __m128i);
        // Truncating (a+b)/2 by widen, add and shift.
        let zero = _mm_setzero_si128();
        let avg16 = |a: __m128i, b: __m128i| {
            let alo = _mm_unpacklo_epi8(a, zero);
            let ahi = _mm_unpackhi_epi8(a, zero);
            let blo = _mm_unpacklo_epi8(b, zero);
            let bhi = _mm_unpackhi_epi8(b, zero);
            let slo = _mm_srli_epi16::<1>(_mm_add_epi16(alo, blo));
            let shi = _mm_srli_epi16::<1>(_mm_add_epi16(ahi, bhi));
            _mm_packus_epi16(slo, shi)
        };
        let b = avg16(b0, b1);
        let r = avg16(r0, r1);
        let lo = _mm_unpacklo_epi8(b, r);
        let hi = _mm_unpackhi_epi8(b, r);
        _mm_storeu_si128(dst.as_mut_ptr().add(*i * 2) as *mut __m128i, lo);
        _mm_storeu_si128(dst.as_mut_ptr().add(*i * 2 + 16) as *mut __m128i, hi);
        *i += 16;
    }
}

/// Write NV16 (4:2:2) or NV24 (4:4:4) output: full-resolution Y plane plus an
/// interleaved Cb/Cr plane copied at the source's native chroma resolution.
///
/// The destination uses the `[total_h, even_width]` grid (even_width =
/// `ceil(img_w/2)*2`), where each grid row is `grid_row_stride` bytes (≥
/// even_width). The interleaved UV plane starts after the `img_h` luma rows.
/// A chroma row is `even_width` bytes for NV16 (one grid row) and
/// `2*even_width` bytes for NV24 (two grid rows). Each UV byte is placed by
/// mapping its linear offset to a `(grid_row, col)` cell.
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
    y_already_written: bool,
) {
    let y_comp = &hdr.components[0];
    let cb = &hdr.components[1];
    let y_stride = mcus_x * y_comp.sampling.h as usize * 8;
    let c_stride = mcus_x * cb.sampling.h as usize * 8;

    if !y_already_written {
        for row in 0..num_rows {
            let s = row * y_stride;
            let d = (y_start + row) * grid_row_stride;
            dst[d..d + img_w].copy_from_slice(&comp_bufs[0][s..s + img_w]);
        }
    }

    // Both keep full-height chroma (one chroma row per luma row):
    //   NV16 (4:2:2): shift_x=1 → ceil(W/2) pairs, uv_rows_per_luma=1.
    //   NV24 (4:4:4): shift_x=0 → W pairs (2W bytes), uv_rows_per_luma=2.
    let layout = output_format
        .chroma_layout()
        .expect("write_nv16_nv24_rows is only called for semi-planar NV16/NV24");
    let chroma_cols = img_w.div_ceil(1 << layout.shift_x);
    let uv_plane_offset = img_h * grid_row_stride;

    let cb_buf = &comp_bufs[1];
    let cr_buf = &comp_bufs[2];
    for row in 0..num_rows {
        let oy = y_start + row; // chroma row == luma row (full-height chroma)
        let src = row * c_stride;
        // The UV row of luma row `oy` starts `oy * uv_rows_per_luma` grid rows
        // into the plane.
        let base = uv_plane_offset + oy * layout.uv_rows_per_luma * grid_row_stride;
        interleave_uv_row(
            &cb_buf[src..src + chroma_cols],
            &cr_buf[src..src + chroma_cols],
            &mut dst[base..base + chroma_cols * 2],
            chroma_cols,
        );
    }
}

/// Interleave planar Cb/Cr sample rows into packed UV (`Cb,Cr,Cb,Cr,…`).
/// Uses NEON `vst2` / SSE2 unpack on supported targets.
#[inline]
fn interleave_uv_row(cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    debug_assert!(cb.len() >= n && cr.len() >= n && dst.len() >= n * 2);
    #[cfg(target_arch = "aarch64")]
    {
        use crate::jpeg::cpu::{neon_tier, NeonTier};
        if !matches!(neon_tier(), NeonTier::Scalar)
            && std::arch::is_aarch64_feature_detected!("neon")
        {
            // SAFETY: lengths checked above; neon path only reads/writes `n`.
            unsafe { interleave_uv_row_neon(cb, cr, dst, n) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::jpeg::cpu::intel_tier().has_sse2() && is_x86_feature_detected!("sse2") {
            // SAFETY: lengths checked above.
            unsafe { interleave_uv_row_sse2(cb, cr, dst, n) };
            return;
        }
    }
    interleave_uv_row_scalar(cb, cr, dst, 0, n);
}

#[inline]
fn interleave_uv_row_scalar(cb: &[u8], cr: &[u8], dst: &mut [u8], start: usize, n: usize) {
    for i in start..n {
        dst[i * 2] = cb[i];
        dst[i * 2 + 1] = cr[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interleave_uv_row_neon(cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    use core::arch::aarch64::{uint8x16x2_t, vld1q_u8, vst2q_u8};
    let mut i = 0usize;
    while i + 16 <= n {
        let b = vld1q_u8(cb.as_ptr().add(i));
        let r = vld1q_u8(cr.as_ptr().add(i));
        vst2q_u8(dst.as_mut_ptr().add(i * 2), uint8x16x2_t(b, r));
        i += 16;
    }
    interleave_uv_row_scalar(cb, cr, dst, i, n);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn interleave_uv_row_sse2(cb: &[u8], cr: &[u8], dst: &mut [u8], n: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*; // NOSONAR
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*; // NOSONAR
    let mut i = 0usize;
    while i + 16 <= n {
        let b = _mm_loadu_si128(cb.as_ptr().add(i) as *const __m128i);
        let r = _mm_loadu_si128(cr.as_ptr().add(i) as *const __m128i);
        let lo = _mm_unpacklo_epi8(b, r);
        let hi = _mm_unpackhi_epi8(b, r);
        _mm_storeu_si128(dst.as_mut_ptr().add(i * 2) as *mut __m128i, lo);
        _mm_storeu_si128(dst.as_mut_ptr().add(i * 2 + 16) as *mut __m128i, hi);
        i += 16;
    }
    interleave_uv_row_scalar(cb, cr, dst, i, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_uv_row_matches_scalar() {
        let n = 37;
        let cb: Vec<u8> = (0..n as u8).collect();
        let cr: Vec<u8> = (100..100 + n as u8).collect();
        let mut got = vec![0u8; n * 2];
        let mut exp = vec![0u8; n * 2];
        interleave_uv_row(&cb, &cr, &mut got, n);
        interleave_uv_row_scalar(&cb, &cr, &mut exp, 0, n);
        assert_eq!(got, exp);
    }

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
        // Read from EDGEFIRST_TESTDATA_DIR when set, else from the source tree.
        let path = std::env::var_os("EDGEFIRST_TESTDATA_DIR")
            .map(|d| std::path::PathBuf::from(d).join(name))
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|p| p.parent())
                    .unwrap()
                    .join("testdata")
                    .join(name)
            });
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    /// A `grid_row_stride` that covers the even-rounded logical width but not
    /// the MCU-aligned width decodes and matches a padded reference decode.
    #[test]
    fn undersized_row_stride_clips_padding_instead_of_panicking() {
        let jpeg = test_jpeg("coco_420_odd.jpg");
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let fmt = super::super::native_format(&headers).unwrap();
        assert_eq!(fmt, PixelFormat::Nv12, "precondition: 4:2:0 source");
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        assert_eq!(
            (img_w, img_h),
            (500, 375),
            "precondition: the reported dims"
        );
        assert!(
            !img_w.is_multiple_of(16),
            "precondition: width must not already be MCU-aligned, or this \
             test cannot reproduce the right-edge spill"
        );

        let total_rows = fmt.combined_plane_height(img_h).unwrap();

        // Reference: a stride past the MCU-aligned width.
        let ref_stride = img_w.next_multiple_of(64);
        let mut ref_scratch = McuScratch::new(&headers);
        let mut reference = vec![0u8; ref_stride * total_rows];
        decode_image(
            &jpeg,
            &headers,
            &mut ref_scratch,
            &mut reference,
            ref_stride,
            fmt,
            false,
        )
        .unwrap();

        // Under test: the even-rounded logical stride.
        let stride = img_w.next_multiple_of(2);
        let mut scratch = McuScratch::new(&headers);
        let mut dst = vec![0u8; stride * total_rows]; // exactly `needed`, no slack
        decode_image(&jpeg, &headers, &mut scratch, &mut dst, stride, fmt, false)
            .expect("an MCU-unaligned row stride must clip the padding, not panic or error");

        for row in 0..img_h {
            let got = &dst[row * stride..row * stride + img_w];
            let want = &reference[row * ref_stride..row * ref_stride + img_w];
            assert_eq!(
                got, want,
                "luma row {row} diverges from the reference decode"
            );
        }
    }

    /// The structured NV12 downsample rows match `avg_block` exactly.
    #[test]
    fn downsample_rows_match_avg_block() {
        let stride = 48;
        let rows = 2;
        let mut state = 0xABCD_1234u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let cb: Vec<u8> = (0..stride * rows).map(|_| next()).collect();
        let cr: Vec<u8> = (0..stride * rows).map(|_| next()).collect();

        // 2×2 (4:4:4 → NV12).
        let n = 23;
        let mut got = vec![0u8; n * 2];
        downsample_uv_row_2x2(
            &cb[..n * 2],
            &cb[stride..stride + n * 2],
            &cr[..n * 2],
            &cr[stride..stride + n * 2],
            &mut got,
            n,
        );
        for ocx in 0..n {
            assert_eq!(
                got[ocx * 2],
                avg_block(&cb, stride, ocx * 2, 0, 2, 2),
                "cb {ocx}"
            );
            assert_eq!(
                got[ocx * 2 + 1],
                avg_block(&cr, stride, ocx * 2, 0, 2, 2),
                "cr {ocx}"
            );
        }

        // 1×2 vertical (4:2:2 → NV12).
        let n = 37;
        let mut got = vec![0u8; n * 2];
        downsample_uv_row_v2(
            &cb[..n],
            &cb[stride..stride + n],
            &cr[..n],
            &cr[stride..stride + n],
            &mut got,
            n,
        );
        for ocx in 0..n {
            assert_eq!(
                got[ocx * 2],
                avg_block(&cb, stride, ocx, 0, 1, 2),
                "cb {ocx}"
            );
            assert_eq!(
                got[ocx * 2 + 1],
                avg_block(&cr, stride, ocx, 0, 1, 2),
                "cr {ocx}"
            );
        }
    }

    /// A scan that stops early is an error.
    #[test]
    fn truncated_scan_is_rejected() {
        let jpeg = test_jpeg("zidane.jpg");
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let fmt = super::super::native_format(&headers).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let stride = img_w.next_multiple_of(2);
        let mut scratch = McuScratch::new(&headers);
        let mut out = vec![0u8; stride * img_h * 3];

        // Whole file decodes; the same file cut to its first MCUs does not.
        decode_image(&jpeg, &headers, &mut scratch, &mut out, stride, fmt, false).unwrap();

        for keep in [0usize, 64, 4096] {
            let cut = (headers.scan_data_offset + keep).min(jpeg.len());
            let err = decode_image(
                &jpeg[..cut],
                &headers,
                &mut scratch,
                &mut out,
                stride,
                fmt,
                false,
            )
            .expect_err("truncated scan decoded as if complete");
            assert!(
                matches!(err, CodecError::InvalidData(_)),
                "keep={keep}: unexpected error {err:?}"
            );
        }
    }

    #[test]
    fn fused_rgb_matches_nv24_plus_row_conversion() {
        for name in ["zidane_444.jpg", "coco_444_odd.jpg"] {
            let jpeg = test_jpeg(name);
            let headers = super::super::markers::parse_markers(&jpeg).unwrap();
            let img_w = headers.header.width as usize;
            let img_h = headers.header.height as usize;
            assert_eq!(
                super::super::native_format(&headers).unwrap(),
                PixelFormat::Nv24,
                "{name} must be 4:4:4"
            );
            let even_w = img_w.next_multiple_of(2);

            let mut scratch = McuScratch::new(&headers);
            let mut nv24 = vec![0u8; even_w * img_h * 3];
            decode_image(
                &jpeg,
                &headers,
                &mut scratch,
                &mut nv24,
                even_w,
                PixelFormat::Nv24,
                false,
            )
            .unwrap();

            let rgb_stride = img_w * 3;
            let mut rgb = vec![0u8; rgb_stride * img_h];
            decode_image(
                &jpeg,
                &headers,
                &mut scratch,
                &mut rgb,
                rgb_stride,
                PixelFormat::Rgb,
                false,
            )
            .unwrap();

            let uv_plane = img_h * even_w;
            let mut cb_row = vec![0u8; img_w];
            let mut cr_row = vec![0u8; img_w];
            let mut expect = vec![0u8; img_w * 3];
            for y in 0..img_h {
                let y_row = &nv24[y * even_w..y * even_w + img_w];
                let base = uv_plane + y * 2 * even_w; // NV24: 2 grid rows per luma row
                for x in 0..img_w {
                    cb_row[x] = nv24[base + x * 2];
                    cr_row[x] = nv24[base + x * 2 + 1];
                }
                crate::jpeg::color::ycbcr_to_rgb_row(y_row, &cb_row, &cr_row, &mut expect, img_w);
                assert_eq!(
                    &rgb[y * rgb_stride..y * rgb_stride + img_w * 3],
                    &expect[..],
                    "{name} row {y}"
                );
            }
        }
    }

    /// Fused NV12 from a 4:4:4 source: luma identical to the NV24 decode;
    /// interior chroma equals the truncating 2×2 average of the NV24 chroma.
    #[test]
    fn fused_nv12_from_444_averages_chroma() {
        let jpeg = test_jpeg("coco_444_odd.jpg");
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let even_w = img_w.next_multiple_of(2);

        let mut scratch = McuScratch::new(&headers);
        let mut nv24 = vec![0u8; even_w * img_h * 3];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut nv24,
            even_w,
            PixelFormat::Nv24,
            false,
        )
        .unwrap();

        let mut nv12 = vec![0u8; even_w * (img_h + img_h.div_ceil(2))];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut nv12,
            even_w,
            PixelFormat::Nv12,
            false,
        )
        .unwrap();

        for y in 0..img_h {
            assert_eq!(
                &nv12[y * even_w..y * even_w + img_w],
                &nv24[y * even_w..y * even_w + img_w],
                "luma row {y}"
            );
        }

        let uv24 = img_h * even_w;
        let uv12 = img_h * even_w;
        let cb24 = |x: usize, y: usize| nv24[uv24 + y * 2 * even_w + x * 2] as u32;
        let cr24 = |x: usize, y: usize| nv24[uv24 + y * 2 * even_w + x * 2 + 1] as u32;
        for ocy in 0..img_h / 2 {
            for ocx in 0..img_w / 2 {
                let (x, y) = (ocx * 2, ocy * 2);
                let eb = (cb24(x, y) + cb24(x + 1, y) + cb24(x, y + 1) + cb24(x + 1, y + 1)) / 4;
                let er = (cr24(x, y) + cr24(x + 1, y) + cr24(x, y + 1) + cr24(x + 1, y + 1)) / 4;
                let got_b = nv12[uv12 + ocy * even_w + ocx * 2] as u32;
                let got_r = nv12[uv12 + ocy * even_w + ocx * 2 + 1] as u32;
                assert_eq!(got_b, eb, "cb at ({ocx},{ocy})");
                assert_eq!(got_r, er, "cr at ({ocx},{ocy})");
            }
        }
    }

    /// When the MCU-aligned luma width exceeds the grid width, the direct luma
    /// path clips the right-edge block; the Grey decode is the reference.
    #[test]
    fn y_direct_edge_mcu_does_not_spill() {
        let jpeg = test_jpeg("coco_444_odd.jpg");
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let even_w = img_w.next_multiple_of(2);
        let mut scratch = McuScratch::new(&headers);
        let mut nv24 = vec![0u8; even_w * img_h * 3];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut nv24,
            even_w,
            PixelFormat::Nv24,
            false,
        )
        .unwrap();
        let mut grey = vec![0u8; img_w * img_h];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut grey,
            img_w,
            PixelFormat::Grey,
            false,
        )
        .unwrap();
        let mut diffs = 0;
        for y in 0..img_h {
            for x in 0..img_w {
                if nv24[y * even_w + x] != grey[y * img_w + x] {
                    diffs += 1;
                }
            }
        }
        assert_eq!(diffs, 0, "y_direct luma differs from scratch-copied luma");
    }

    /// Fused RGB is refused for sources that are neither 4:4:4-equivalent nor
    /// native 4:2:0.
    #[test]
    fn fused_rgb_rejects_non_420_subsampled_sources() {
        let jpeg = test_jpeg("jaguar_422.jpg"); // luma 2x2, chroma 1x2 (4:2:2)
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let mut scratch = McuScratch::new(&headers);
        let mut rgb = vec![0u8; img_w * 3 * img_h];
        let err = decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut rgb,
            img_w * 3,
            PixelFormat::Rgb,
            false,
        );
        assert!(err.is_err());
    }

    #[test]
    fn expand_row_2x_nearest_neighbour() {
        let src = [10u8, 20, 30];
        let mut dst = [0u8; 6];
        expand_row_2x(&src, &mut dst);
        assert_eq!(dst, [10, 10, 20, 20, 30, 30]);

        // Odd destination width: the last source sample contributes once.
        let mut dst_odd = [0u8; 5];
        expand_row_2x(&src, &mut dst_odd);
        assert_eq!(dst_odd, [10, 10, 20, 20, 30]);
    }

    /// Reference for [`write_rgb_rows_420`]: expand chroma via
    /// [`expand_row_2x`] on both axes and convert with
    /// [`crate::jpeg::color::ycbcr_to_rgb_row`], row by row.
    fn reference_rgb_420(y: &[u8], cb: &[u8], cr: &[u8], img_w: usize, img_h: usize) -> Vec<u8> {
        let chroma_w = img_w.div_ceil(2);
        let mut out = vec![0u8; img_w * 3 * img_h];
        let mut cb_row = vec![0u8; img_w];
        let mut cr_row = vec![0u8; img_w];
        for row in 0..img_h {
            let c_row = row / 2;
            expand_row_2x(
                &cb[c_row * chroma_w..c_row * chroma_w + chroma_w],
                &mut cb_row,
            );
            expand_row_2x(
                &cr[c_row * chroma_w..c_row * chroma_w + chroma_w],
                &mut cr_row,
            );
            crate::jpeg::color::ycbcr_to_rgb_row(
                &y[row * img_w..row * img_w + img_w],
                &cb_row,
                &cr_row,
                &mut out[row * img_w * 3..(row + 1) * img_w * 3],
                img_w,
            );
        }
        out
    }

    /// Synthetic Y/Cb/Cr planes with distinct per-pixel values for
    /// `img_w x img_h` (`y_stride == img_w`, chroma planes
    /// `ceil(w/2) x ceil(h/2)`).
    fn synthetic_420_planes(img_w: usize, img_h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let chroma_w = img_w.div_ceil(2);
        let chroma_h = img_h.div_ceil(2);
        let y: Vec<u8> = (0..img_w * img_h).map(|i| (i * 7 + 11) as u8).collect();
        let cb: Vec<u8> = (0..chroma_w * chroma_h)
            .map(|i| (i * 5 + 40) as u8)
            .collect();
        let cr: Vec<u8> = (0..chroma_w * chroma_h)
            .map(|i| (i * 3 + 90) as u8)
            .collect();
        (y, cb, cr)
    }

    /// `write_rgb_rows_420` matches the reference at `1x1`, odd-width,
    /// odd-height and odd x odd sizes.
    #[test]
    fn write_rgb_rows_420_odd_dimensions() {
        for (img_w, img_h) in [(1, 1), (3, 2), (2, 3), (3, 3), (5, 4), (4, 5), (7, 9)] {
            let (y, cb, cr) = synthetic_420_planes(img_w, img_h);
            let expect = reference_rgb_420(&y, &cb, &cr, img_w, img_h);

            let comp_bufs = vec![y, cb, cr];
            let mut upsample_cb = vec![0u8; img_w];
            let mut upsample_cr = vec![0u8; img_w];
            let mut dst = vec![0u8; img_w * 3 * img_h];
            write_rgb_rows_420(
                &comp_bufs,
                img_w,
                img_w.div_ceil(2),
                &mut upsample_cb,
                &mut upsample_cr,
                &mut dst,
                img_w * 3,
                0,
                img_h,
                img_w,
                0,
            );
            assert_eq!(dst, expect, "{img_w}x{img_h}");
        }
    }

    /// Fused RGB from a native 4:2:0 source equals the NV12 decode's chroma,
    /// box-upsampled 2×2, run through `ycbcr_to_rgb_row`.
    #[test]
    fn fused_rgb_420_matches_nv12_plus_box_upsample() {
        let jpeg = test_jpeg("zidane.jpg"); // 1280x720, native 4:2:0
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        assert!(is_native_420(&headers.header));
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let even_w = img_w.next_multiple_of(2);

        let mut scratch = McuScratch::new(&headers);
        let mut nv12 = vec![0u8; even_w * (img_h + img_h.div_ceil(2))];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut nv12,
            even_w,
            PixelFormat::Nv12,
            false,
        )
        .unwrap();

        let rgb_stride = img_w * 3;
        let mut rgb = vec![0u8; rgb_stride * img_h];
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut rgb,
            rgb_stride,
            PixelFormat::Rgb,
            false,
        )
        .unwrap();

        let uv_plane = img_h * even_w;
        let chroma_w = img_w.div_ceil(2);
        let mut cb_row = vec![0u8; img_w];
        let mut cr_row = vec![0u8; img_w];
        let mut cb_half = vec![0u8; chroma_w];
        let mut cr_half = vec![0u8; chroma_w];
        let mut expect = vec![0u8; img_w * 3];
        for y in 0..img_h {
            let y_row = &nv12[y * even_w..y * even_w + img_w];
            let uv_row = uv_plane + (y / 2) * even_w;
            for x in 0..chroma_w {
                cb_half[x] = nv12[uv_row + x * 2];
                cr_half[x] = nv12[uv_row + x * 2 + 1];
            }
            expand_row_2x(&cb_half, &mut cb_row);
            expand_row_2x(&cr_half, &mut cr_row);
            crate::jpeg::color::ycbcr_to_rgb_row(y_row, &cb_row, &cr_row, &mut expect, img_w);
            assert_eq!(
                &rgb[y * rgb_stride..y * rgb_stride + img_w * 3],
                &expect[..],
                "row {y}"
            );
        }
    }

    /// Decoding into a buffer whose row pitch is padded beyond the natural width
    /// places the same pixel bytes, row for row, as a tightly-packed decode.
    #[test]
    fn decode_padded_grid_matches_tight() {
        let jpeg = test_jpeg("zidane.jpg"); // 1280×720, NV12 (4:2:0), even dims
        let headers = super::super::markers::parse_markers(&jpeg).unwrap();
        let img_w = headers.header.width as usize;
        let img_h = headers.header.height as usize;
        let fmt = super::super::native_format(&headers).unwrap();
        assert_eq!(fmt, PixelFormat::Nv12);

        let even_w = img_w.next_multiple_of(2);
        // 2·h rows hold the NV12 luma and chroma rows in both buffers.
        let rows = img_h * 2;
        let tight = even_w;
        let padded = even_w + 64; // physical pitch > natural width

        let mut scratch = McuScratch::new(&headers);
        scratch.ensure_capacity(&headers);
        let mut buf_tight = vec![0u8; rows * tight];
        let mut buf_padded = vec![0u8; rows * padded];

        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut buf_tight,
            tight,
            fmt,
            false,
        )
        .unwrap();
        decode_image(
            &jpeg,
            &headers,
            &mut scratch,
            &mut buf_padded,
            padded,
            fmt,
            false,
        )
        .unwrap();

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
