// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A frame from VideoToolbox's H.264 decoder, lent as a multiplane NV12
//! tensor over the decoder's own IOSurface, converts to RGBA on ANGLE as it
//! does on the CPU.
//!
//! Skips (prints `SKIPPED`) without GPU buffers, a decoder or GL.

#![cfg(all(target_os = "macos", feature = "opengl", feature = "codec"))]

use edgefirst_codec::video::{
    access_units, DecodeEvent, DecodedFrame, DecoderConfig, VideoDecoder,
};
use edgefirst_codec::{CodecError, Engine};
use edgefirst_image::{CPUProcessor, GLProcessorThreaded, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

#[path = "../../codec/tests/common/sessions.rs"]
mod sessions;

#[path = "../../tensor/tests/support/dma_require.rs"]
#[allow(dead_code)]
mod dma_require;

const TEST: &str = "a_videotoolbox_frame_converts_to_rgba_on_angle";

fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {TEST} - {why}");
}

fn testdata(name: &str) -> Vec<u8> {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("testdata")
        });
    let path = root.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn rgba(w: usize, h: usize, memory: TensorMemory) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        PixelFormat::Rgba,
        DType::U8,
        Some(memory),
        CpuAccess::ReadWrite,
    )
    .unwrap()
}

/// The RGBA pixels of `t` without row padding.
fn tight_rgba(t: &TensorDyn) -> Vec<u8> {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    let map = t.map_bytes(CpuAccess::Read).unwrap();
    (0..h)
        .flat_map(|y| map[y * stride..y * stride + w * 4].iter().copied())
        .collect()
}

/// The first picture of `stream` in display order.
fn first_frame(decoder: &mut VideoDecoder, stream: &[u8]) -> DecodedFrame {
    let first = |events: Vec<DecodeEvent>| {
        events.into_iter().find_map(|e| match e {
            DecodeEvent::Frame(f) => Some(f),
            DecodeEvent::Reconfigured(_) => None,
        })
    };
    for (i, au) in access_units(stream).enumerate() {
        if let Some(f) = first(decoder.decode(au, i as u64).expect("decode")) {
            return f;
        }
    }
    first(decoder.flush().expect("flush")).expect("the stream has a picture")
}

#[test]
fn a_videotoolbox_frame_converts_to_rgba_on_angle() {
    let _sessions = sessions::take(1);
    if !dma_require::available_or_skip(edgefirst_tensor::is_gpu_buffer_available(), TEST) {
        return;
    }
    let config = DecoderConfig::h264().with_engine(Engine::VideoToolbox);
    let mut decoder = match VideoDecoder::new(config) {
        Ok(d) => d,
        Err(CodecError::NoDevice(why)) => return skip(&why),
        Err(e) => panic!("decoder open failed: {e}"),
    };
    let mut gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => return skip(&format!("GL not available: {e}")),
    };
    let frame = first_frame(&mut decoder, &testdata("h264_1080p_main_bframes.h264"));
    let src = frame.tensor();
    assert!(src.is_multiplane(), "the frame is multiplane NV12");
    let (w, h) = (src.width().unwrap(), src.height().unwrap());
    assert_eq!((w, h), (1920, 1080));

    let mut gpu = rgba(w, h, TensorMemory::DmaBuf);
    gl.convert(src, &mut gpu, None)
        .expect("GL converts a VideoToolbox frame");
    let mut cpu = rgba(w, h, TensorMemory::Mem);
    CPUProcessor::new()
        .convert(src, &mut cpu, None)
        .expect("the CPU converts a VideoToolbox frame");

    let (gpu, cpu) = (tight_rgba(&gpu), tight_rgba(&cpu));
    let worst = gpu
        .chunks_exact(4)
        .zip(cpu.chunks_exact(4))
        .flat_map(|(g, c)| (0..3).map(move |i| g[i].abs_diff(c[i])))
        .max()
        .unwrap();
    assert!(worst <= 3, "GL and CPU RGBA differ by up to {worst}");
}
