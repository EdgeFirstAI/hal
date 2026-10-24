// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The CPU backend never reads a destination declared without CPU reads.
//!
//! `CpuAccess::None` and `CpuAccess::Write` DMA-BUF images come from the
//! uncached CMA heap where the system has one, so a CPU pass that read such a
//! destination would be slow. The declaration is enforced on every map, but
//! a pass that maps `Write` and then reads what it mapped would get past
//! that check. Each convert here runs twice into a `Write` destination whose
//! bytes a `ReadWrite` alias of the same dma-buf pre-fills differently: a
//! convert that succeeds must write identical bytes both times, so it read
//! nothing back, and one that would read back must be refused.
//!
//! Needs a DMA heap; skips loudly without one.

#![cfg(target_os = "linux")]

use edgefirst_image::{
    CPUProcessor, ConvertOptions, Crop, Error, Flip, ImageProcessorTrait, MaskOverlay, Rotation,
};
use edgefirst_tensor::{
    CpuAccess, DType, DetectBox, Error as TensorError, PixelFormat, PixelLayout, Region, Tensor,
    TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

const W: usize = 64;
const H: usize = 48;

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "WARNING: SKIPPED: {why}");
}

fn source(format: PixelFormat) -> TensorDyn {
    let t = TensorDyn::image(
        W,
        H,
        format,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    for (i, b) in t
        .as_u8()
        .unwrap()
        .map_mut()
        .unwrap()
        .as_mut_slice()
        .iter_mut()
        .enumerate()
    {
        *b = (i * 7 % 253) as u8;
    }
    t
}

/// A `Write`-declared DMA-BUF destination and a `ReadWrite` byte alias of
/// its whole buffer, or `None` without a DMA heap.
fn destination(
    format: PixelFormat,
    dtype: DType,
    access: CpuAccess,
) -> Option<(TensorDyn, Tensor<u8>)> {
    let dst = TensorDyn::image(W, H, format, dtype, Some(TensorMemory::DmaBuf), access).ok()?;
    let fd = dst.clone_fd().unwrap();
    let len = std::fs::File::from(fd.try_clone().unwrap())
        .metadata()
        .unwrap()
        .len() as usize;
    let alias = Tensor::<u8>::from_fd(fd, &[len], None).unwrap();
    Some((dst, alias))
}

fn fill(alias: &Tensor<u8>, byte: u8) {
    alias.map().unwrap().as_mut_slice().fill(byte);
}

fn bytes(alias: &Tensor<u8>) -> Vec<u8> {
    alias.map().unwrap().as_slice().to_vec()
}

/// The pixel bytes of `dst`, read through `alias`: each row without the
/// pitch padding and the page rounding past the last row, which no convert
/// writes.
fn pixels(dst: &TensorDyn, alias: &Tensor<u8>) -> Vec<u8> {
    let format = dst.format().unwrap();
    let elem = dst.dtype().size();
    let (row, rows) = match format.layout() {
        PixelLayout::Packed => (W * format.channels() * elem, H),
        PixelLayout::Planar => (W * elem, format.channels() * H),
        _ => (W * elem, dst.shape()[0]),
    };
    let stride = dst.effective_row_stride().unwrap_or(row);
    let all = bytes(alias);
    (0..rows)
        .flat_map(|r| all[r * stride..r * stride + row].iter().copied())
        .collect()
}

fn is_denied(e: &Error) -> bool {
    e.cpu_access_denied().is_some()
}

#[test]
fn cpu_converts_never_read_a_write_only_destination() {
    if destination(PixelFormat::Rgba, DType::U8, CpuAccess::Write).is_none() {
        return skip("no DMA heap");
    }
    let half = Region::new(8, 4, W / 2, H / 2);
    let options: [(&str, Option<ConvertOptions>); 5] = [
        ("plain", None),
        (
            "letterbox",
            Some(
                ConvertOptions::new()
                    .source(Some(half))
                    .letterbox([114, 114, 114, 255]),
            ),
        ),
        (
            "crop",
            Some(ConvertOptions::new().crop(Crop::new()).source(Some(half))),
        ),
        ("flip", Some(ConvertOptions::new().flip(Flip::Horizontal))),
        (
            "rotate",
            Some(ConvertOptions::new().rotation(Rotation::Rotate180)),
        ),
    ];
    let mut cpu = CPUProcessor::new();
    let mut ran = 0;
    for src_format in [
        PixelFormat::Rgba,
        PixelFormat::Rgb,
        PixelFormat::Yuyv,
        PixelFormat::Nv12,
    ] {
        let src = source(src_format);
        for (dst_format, dtype) in [
            (PixelFormat::Rgba, DType::U8),
            (PixelFormat::Rgb, DType::U8),
            (PixelFormat::Bgra, DType::U8),
            (PixelFormat::Grey, DType::U8),
            (PixelFormat::PlanarRgb, DType::U8),
            (PixelFormat::Rgb, DType::I8),
            (PixelFormat::PlanarRgb, DType::I8),
        ] {
            for (name, opts) in &options {
                let label = format!("{src_format:?} -> {dst_format:?}/{dtype:?} {name}");
                let (mut dst, alias) = destination(dst_format, dtype, CpuAccess::Write).unwrap();
                let mut outputs = Vec::new();
                for prefill in [0x00, 0xA5] {
                    fill(&alias, prefill);
                    match cpu.convert(&src, &mut dst, *opts) {
                        Ok(()) => outputs.push(pixels(&dst, &alias)),
                        Err(e) if is_denied(&e) => break,
                        Err(Error::NotSupported(_) | Error::NotImplemented(_)) => break,
                        Err(e) => panic!("{label}: {e}"),
                    }
                }
                if let [a, b] = &outputs[..] {
                    ran += 1;
                    assert!(
                        a == b,
                        "{label}: the output depends on the destination's earlier bytes"
                    );
                }
            }
        }
    }
    assert!(ran > 0, "no convert ran into a write-only destination");
}

/// Drawing blends over the destination, so the CPU draw refuses `Write` and
/// `None` destinations before touching them.
#[test]
fn cpu_draw_refuses_destinations_it_cannot_read() {
    let mut cpu = CPUProcessor::new();
    let boxes = [DetectBox {
        bbox: [0.1, 0.1, 0.6, 0.6].into(),
        score: 0.9,
        label: 0,
    }];
    for access in [CpuAccess::Write, CpuAccess::None] {
        let Some((mut dst, alias)) = destination(PixelFormat::Rgba, DType::U8, access) else {
            return skip("no DMA heap");
        };
        fill(&alias, 0x5A);
        match cpu.draw_decoded_masks(&mut dst, &boxes, &[], MaskOverlay::default()) {
            Err(Error::Tensor(TensorError::CpuAccessDenied { declared, .. })) => {
                assert_eq!(declared, access)
            }
            other => panic!("{access:?}: {other:?}"),
        }
        assert!(
            bytes(&alias).iter().all(|&b| b == 0x5A),
            "{access:?}: a refused draw wrote the destination"
        );
    }
}

/// A hardware-only source is never read by the CPU convert.
#[test]
fn cpu_convert_refuses_a_hardware_only_source() {
    let mut cpu = CPUProcessor::new();
    for access in [CpuAccess::None, CpuAccess::Write] {
        let Some((src, _alias)) = destination(PixelFormat::Rgba, DType::U8, access) else {
            return skip("no DMA heap");
        };
        let mut dst = TensorDyn::image(
            W,
            H,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        match cpu.convert(&src, &mut dst, None) {
            Err(e) if is_denied(&e) => {}
            other => panic!("{access:?}: {other:?}"),
        }
    }
}
