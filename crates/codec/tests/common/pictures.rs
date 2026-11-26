// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NV12 test pictures: allocation, a moving gradient, region extraction
//! and comparison.

#![allow(dead_code)]

use edgefirst_tensor::{CpuAccess, DType, PixelFormat, Region, TensorDyn, TensorMemory};

/// A DMA-BUF NV12 image.
pub fn nv12(width: usize, height: usize) -> TensorDyn {
    TensorDyn::image(
        width,
        height,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("DMA-BUF NV12 image")
}

/// The row stride of the luma plane, and the start and row stride of the
/// chroma plane: in its own tensor, or in the luma buffer after the rows.
pub fn nv12_planes(img: &TensorDyn) -> (usize, Option<TensorDyn>, usize, usize) {
    let stride = img.effective_row_stride().unwrap();
    match img.chroma_dyn() {
        Some(c) => {
            let cstride = c.effective_row_stride().unwrap();
            (stride, Some(c), 0, cstride)
        }
        None => (stride, None, stride * img.height().unwrap(), stride),
    }
}

/// Paints `img` with a gradient that moves with `frame`.
pub fn paint(img: &TensorDyn, frame: usize) {
    let w = img.width().unwrap();
    let h = img.height().unwrap();
    let (stride, chroma_tensor, chroma, cstride) = nv12_planes(img);
    let mut luma = img.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        for x in 0..w {
            luma[y * stride + x] = ((x / 4 + y / 4 + 2 * frame) & 0xff) as u8;
        }
    }
    let mut chroma_map = chroma_tensor.map(|c| c.map_bytes(CpuAccess::Write).unwrap());
    let cpx: &mut [u8] = match chroma_map.as_mut() {
        Some(m) => m,
        None => &mut luma,
    };
    for y in 0..h / 2 {
        for x in (0..w).step_by(2) {
            cpx[chroma + y * cstride + x] = (64 + (x / 16 + frame) % 128) as u8;
            cpx[chroma + y * cstride + x + 1] = (64 + (y / 8) % 128) as u8;
        }
    }
}

/// The dense NV12 bytes of `region` of `img`.
pub fn nv12_region(img: &TensorDyn, r: Region) -> Vec<u8> {
    let (stride, chroma_tensor, chroma, cstride) = nv12_planes(img);
    let luma = img.map_bytes(CpuAccess::Read).unwrap();
    let mut out = Vec::with_capacity(r.width * r.height * 3 / 2);
    for y in r.y..r.y + r.height {
        out.extend_from_slice(&luma[y * stride + r.x..y * stride + r.x + r.width]);
    }
    let chroma_map = chroma_tensor.map(|c| c.map_bytes(CpuAccess::Read).unwrap());
    let cpx: &[u8] = chroma_map.as_deref().unwrap_or(&luma);
    for y in r.y / 2..(r.y + r.height) / 2 {
        let row = chroma + y * cstride;
        out.extend_from_slice(&cpx[row + r.x..row + r.x + r.width]);
    }
    out
}

/// Peak signal-to-noise ratio of `got` against `want`, in dB.
pub fn psnr(want: &[u8], got: &[u8]) -> f64 {
    assert_eq!(want.len(), got.len());
    let se: f64 = want
        .iter()
        .zip(got)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let mse = se / want.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}
