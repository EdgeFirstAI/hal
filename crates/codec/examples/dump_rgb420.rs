// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Scratch tool: dump the fused native-4:2:0-\>RGB decode of a JPEG as a raw
//! PPM (P6), for accuracy comparison against a reference decoder (e.g. PIL).
//! Not part of the public API surface; not wired into CI.

use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory, TensorTrait};

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args.next().expect("usage: dump_rgb420 <in.jpg> <out.ppm>");
    let output = args.next().expect("usage: dump_rgb420 <in.jpg> <out.ppm>");
    let jpeg = std::fs::read(&input).expect("read input");

    let mut decoder = ImageDecoder::new();
    decoder.set_output_format(Some(PixelFormat::Rgb));

    // Peek dimensions first so the tensor is sized exactly (Rgb stride is
    // img_w*3, no even-width padding needed).
    let info = peek_info(&jpeg).expect("peek");
    let mut tensor = Tensor::<u8>::image(
        info.width,
        info.height,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let decoded = tensor.load_image(&mut decoder, &jpeg).expect("decode");
    assert_eq!(decoded.format, PixelFormat::Rgb);

    let map = tensor.map().expect("map");
    let data: &[u8] = &map;
    let mut out = std::fs::File::create(&output).expect("create output");
    use std::io::Write;
    write!(out, "P6\n{} {}\n255\n", decoded.width, decoded.height).unwrap();
    // Strip stride padding if row_stride > width*3.
    let stride = decoded.row_stride;
    let w3 = decoded.width * 3;
    for y in 0..decoded.height {
        let row = &data[y * stride..y * stride + w3];
        out.write_all(row).unwrap();
    }
    eprintln!(
        "wrote {output}: {}x{} RGB (box 2x2 chroma upsample, native 4:2:0 fused path)",
        decoded.width, decoded.height
    );
}
