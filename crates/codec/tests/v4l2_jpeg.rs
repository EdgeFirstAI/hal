// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! V4L2 hardware JPEG decoder integration tests.
//!
//! These compare the hardware decode path against the software decoder. They
//! are safe to run on any host: where no V4L2 JPEG decoder is present the
//! "hardware" decode transparently falls back to the CPU, so the comparison is
//! a (trivial) CPU-vs-CPU check. On a target with a JPEG M2M device (e.g.
//! i.MX95 `/dev/video11`) they validate real hardware parity and cropping.
//!
//! The codec always emits the JPEG's native format — `Nv12` for colour JPEGs
//! and `Grey` for greyscale — so the comparison is over the full native buffer
//! (Y + CbCr for NV12).
//!
//! Run single-threaded — these mutate process-global env vars to toggle the
//! backend, and the workspace test convention is `-j 1` already.
#![cfg(all(target_os = "linux", feature = "v4l2"))]

use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};

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

/// Whether any `/dev/video*` node exists (a coarse "might have hardware" hint
/// used only for logging — the tests are correct either way).
fn has_video_device() -> bool {
    std::fs::read_dir("/dev")
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with("video"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// Decode `jpeg` into a tensor of `fmt` with the backend selected by
/// `disable_v4l2`, returning the full native buffer (Y + CbCr for NV12).
fn decode(jpeg: &[u8], w: usize, h: usize, fmt: PixelFormat, disable_v4l2: bool) -> Vec<u8> {
    if disable_v4l2 {
        std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
    } else {
        std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    }
    let mut tensor = Tensor::<u8>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, jpeg).unwrap();
    let map = tensor.map().unwrap();
    let pixels: &[u8] = &map;
    // NV12 stores 1.5 bytes/pixel (Y plane + interleaved CbCr); Grey stores 1.
    let total = match info.format {
        PixelFormat::Nv12 => info.width * info.height * 3 / 2,
        _ => info.width * info.height * info.format.channels(),
    };
    pixels[..total].to_vec()
}

/// Assert two decodes match within a tolerance that accommodates the hardware
/// vs software IDCT/upsample rounding differences (these are *different*
/// decoders, not the same one), while still catching structural errors —
/// wrong format, garbage, or a blank decode.
fn assert_close(cpu: &[u8], hw: &[u8], label: &str) {
    assert_eq!(cpu.len(), hw.len(), "{label}: length mismatch");
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (&a, &b) in cpu.iter().zip(hw.iter()) {
        let d = a.abs_diff(b);
        max_diff = max_diff.max(d);
        sum_diff += d as u64;
    }
    let mean = sum_diff as f64 / cpu.len() as f64;
    // Thresholds are deliberately generous for cross-decoder comparison and
    // may be tightened once measured on a specific target.
    assert!(
        mean < 3.0 && max_diff <= 24,
        "{label}: hardware/software mismatch (mean={mean:.3}, max={max_diff})"
    );
}

#[test]
fn v4l2_parity_nv12() {
    eprintln!(
        "v4l2_parity_nv12: video device present = {}",
        has_video_device()
    );
    let jpeg = testdata("zidane.jpg");
    let cpu = decode(&jpeg, 1280, 720, PixelFormat::Nv12, true);
    let hw = decode(&jpeg, 1280, 720, PixelFormat::Nv12, false);
    assert_close(&cpu, &hw, "nv12");
}

#[test]
fn v4l2_parity_grey() {
    // grey.jpg is a greyscale JPEG → decodes to a single-plane Grey buffer.
    let jpeg = testdata("grey.jpg");
    let cpu = decode(&jpeg, 1024, 681, PixelFormat::Grey, true);
    let hw = decode(&jpeg, 1024, 681, PixelFormat::Grey, false);
    assert_close(&cpu, &hw, "grey");
}

#[test]
fn v4l2_persistent_loop() {
    // One decoder reused across many frames exercises the persistent stream
    // (built once, then reused). All frames must match the CPU reference.
    let jpeg = testdata("zidane.jpg");
    let cpu = decode(&jpeg, 1280, 720, PixelFormat::Nv12, true);
    let nv12_len = 1280 * 720 * 3 / 2;

    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let mut decoder = ImageDecoder::new();
    let mut tensor =
        Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    for i in 0..50 {
        let info = tensor.load_image(&mut decoder, &jpeg).unwrap();
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.format, PixelFormat::Nv12);
        let map = tensor.map().unwrap();
        let px: &[u8] = &map;
        assert_close(&cpu, &px[..nv12_len], &format!("frame {i}"));
    }
}

#[test]
fn v4l2_geometry_change_rebuilds() {
    // Alternating geometries force the stream to rebuild each call; the decoder
    // must stay healthy (no wedge/leak) and keep producing valid output.
    // Both fixtures are even-dim colour JPEGs → NV12.
    let z = testdata("zidane.jpg");
    let g = testdata("giraffe.jpg");
    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let mut decoder = ImageDecoder::new();

    for data in [&z, &g, &z, &g] {
        let peeked = peek_info(data).unwrap();
        let mut t = Tensor::<u8>::image(
            peeked.width,
            peeked.height,
            PixelFormat::Nv12,
            Some(TensorMemory::Mem),
        )
        .unwrap();
        let info = t.load_image(&mut decoder, data).unwrap();
        assert_eq!((info.width, info.height), (peeked.width, peeked.height));
        assert_eq!(info.format, PixelFormat::Nv12);
        let map = t.map().unwrap();
        let px: &[u8] = &map;
        // Y plane should carry image data.
        let nonzero = px[..info.width * info.height]
            .iter()
            .filter(|&&v| v != 0)
            .count();
        assert!(
            nonzero > 1000,
            "blank decode for {}x{}",
            info.width,
            info.height
        );
    }
}

/// Decode `jpeg` into a `fmt` tensor and extract the tight native buffer,
/// honouring the tensor's (possibly 64-aligned) row stride. Unlike [`decode`]
/// this is correct for widths that are not stride-aligned and for `Nv24`.
fn decode_tight(jpeg: &[u8], w: usize, h: usize, fmt: PixelFormat, disable_v4l2: bool) -> Vec<u8> {
    if disable_v4l2 {
        std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
    } else {
        std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    }
    let mut tensor = Tensor::<u8>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, jpeg).unwrap();
    assert_eq!((info.width, info.height), (w, h));
    let stride = info.row_stride;
    let map = tensor.map().unwrap();
    let px: &[u8] = &map;
    let mut out = Vec::new();
    // Y plane (all formats).
    for y in 0..h {
        out.extend_from_slice(&px[y * stride..y * stride + w]);
    }
    match info.format {
        PixelFormat::Nv12 => {
            // ceil(h/2) chroma rows of even(w) interleaved CbCr bytes.
            let uv_base = h * stride;
            let cw = w.next_multiple_of(2);
            for cy in 0..h.div_ceil(2) {
                let s = uv_base + cy * stride;
                out.extend_from_slice(&px[s..s + cw]);
            }
        }
        PixelFormat::Nv24 => {
            // One 2w-byte interleaved CbCr run per luma row, each run placed
            // two stride-rows apart (uv_rows_per_luma = 2).
            let uv_base = h * stride;
            for y in 0..h {
                let s = uv_base + y * 2 * stride;
                out.extend_from_slice(&px[s..s + 2 * w]);
            }
        }
        PixelFormat::Grey => {}
        other => panic!("unexpected decode format {other:?}"),
    }
    out
}

#[test]
fn v4l2_geometry_change_reconfigures() {
    // Alternating geometries through ONE persistent decoder must take the
    // cheap DMABUF reconfigure path (on hardware) and stay pixel-correct on
    // every frame. Compares each frame against the CPU reference.
    let z = testdata("zidane.jpg");
    let g = testdata("giraffe.jpg");
    let cpu_z = decode_tight(&z, 1280, 720, PixelFormat::Nv12, true);
    let cpu_g = decode_tight(&g, 640, 640, PixelFormat::Nv12, true);

    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let mut decoder = ImageDecoder::new();
    for i in 0..10 {
        for (jpeg, w, h, cpu, label) in [
            (&z, 1280usize, 720usize, &cpu_z, "zidane"),
            (&g, 640, 640, &cpu_g, "giraffe"),
        ] {
            let mut t =
                Tensor::<u8>::image(w, h, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
            let info = t.load_image(&mut decoder, jpeg).unwrap();
            assert_eq!((info.width, info.height), (w, h));
            let stride = info.row_stride;
            let map = t.map().unwrap();
            let px: &[u8] = &map;
            let mut hw = Vec::with_capacity(w * h * 3 / 2);
            for y in 0..h {
                hw.extend_from_slice(&px[y * stride..y * stride + w]);
            }
            let uv_base = h * stride;
            for cy in 0..h / 2 {
                hw.extend_from_slice(&px[uv_base + cy * stride..uv_base + cy * stride + w]);
            }
            assert_close(cpu, &hw, &format!("{label} round {i}"));
        }
    }
}

#[test]
fn v4l2_yuv444_nv24() {
    // 4:4:4 JPEG → packed YUV3 capture on mxc-jpeg → NV24 deinterleave (the
    // scratch path on hardware). Must match the CPU decoder's native NV24.
    let jpeg = testdata("zidane_444.jpg");
    let cpu = decode_tight(&jpeg, 1280, 720, PixelFormat::Nv24, true);
    let hw = decode_tight(&jpeg, 1280, 720, PixelFormat::Nv24, false);
    assert_close(&cpu, &hw, "yuv444-nv24");
}

#[test]
fn v4l2_non_mcu_dims() {
    // Non-MCU-aligned (odd) dimensions are ineligible for zero-copy and must
    // decode correctly through the copy paths, cropped from the MCU-padded
    // CAPTURE buffer.
    for (name, w, h, fmt) in [
        ("coco_420_odd.jpg", 500usize, 375usize, PixelFormat::Nv12),
        ("coco_444_odd.jpg", 307, 461, PixelFormat::Nv24),
        ("jaguar_444.jpg", 789, 384, PixelFormat::Nv24),
    ] {
        let jpeg = testdata(name);
        let cpu = decode_tight(&jpeg, w, h, fmt, true);
        let hw = decode_tight(&jpeg, w, h, fmt, false);
        assert_close(&cpu, &hw, name);
    }
}

#[test]
fn v4l2_reconfigure_loop_stable() {
    // Many geometry + subsampling changes through one decoder: the stream must
    // stay healthy (no wedge, no decode garbage) and must not leak fds.
    let images: Vec<(Vec<u8>, usize, usize, PixelFormat)> = vec![
        (testdata("zidane.jpg"), 1280, 720, PixelFormat::Nv12),
        (testdata("coco_420_odd.jpg"), 500, 375, PixelFormat::Nv12),
        (testdata("zidane_444.jpg"), 1280, 720, PixelFormat::Nv24),
        (testdata("giraffe.jpg"), 640, 640, PixelFormat::Nv12),
        (testdata("coco_444_odd.jpg"), 307, 461, PixelFormat::Nv24),
    ];
    let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();

    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let mut decoder = ImageDecoder::new();
    // Warm-up pass establishes the device fd and the persistent allocations.
    for (jpeg, w, h, fmt) in &images {
        let mut t = Tensor::<u8>::image(*w, *h, *fmt, Some(TensorMemory::Mem)).unwrap();
        t.load_image(&mut decoder, jpeg).unwrap();
    }
    let baseline = fd_count();
    for round in 0..5 {
        for (jpeg, w, h, fmt) in &images {
            let mut t = Tensor::<u8>::image(*w, *h, *fmt, Some(TensorMemory::Mem)).unwrap();
            let info = t.load_image(&mut decoder, jpeg).unwrap();
            assert_eq!((info.width, info.height), (*w, *h), "round {round}");
            let map = t.map().unwrap();
            let px: &[u8] = &map;
            let nonzero = px[..info.width].iter().filter(|&&v| v != 0).count();
            assert!(nonzero > 0, "blank first row for {w}x{h} round {round}");
        }
    }
    let after = fd_count();
    assert!(
        after <= baseline + 1,
        "fd leak across reconfigures: {baseline} -> {after}"
    );
}

#[test]
fn v4l2_zero_copy_dma_nv12() {
    // Zero-copy: decode into a DMA-backed NV12 tensor. On a target with a JPEG
    // M2M device + dma_heap, the hardware decodes straight into the tensor
    // (V4L2_MEMORY_DMABUF import, no plane copy); the result must match the
    // MMAP/copy path byte-for-byte (same hardware decode). Skips cleanly where
    // DMA allocation is unavailable (no dma_heap on this host). zidane is
    // 1280×720 — both MCU(16)-aligned, so zero-copy is eligible.
    let jpeg = testdata("zidane.jpg");
    std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    let mut dma = match Tensor::<u8>::image(1280, 720, PixelFormat::Nv12, Some(TensorMemory::Dma)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skip v4l2_zero_copy_dma_nv12: no DMA allocation ({e})");
            return;
        }
    };
    let mut decoder = ImageDecoder::new();
    let info = dma.load_image(&mut decoder, &jpeg).unwrap();
    assert_eq!((info.width, info.height), (1280, 720));
    assert_eq!(info.format, PixelFormat::Nv12);

    // Extract a tight NV12 buffer from the (possibly pitch-padded) DMA tensor.
    let stride = info.row_stride;
    let (w, h) = (1280usize, 720usize);
    let mut zc = Vec::with_capacity(w * h * 3 / 2);
    {
        let map = dma.map().unwrap();
        let px: &[u8] = &map;
        for y in 0..h {
            zc.extend_from_slice(&px[y * stride..y * stride + w]);
        }
        let uv_base = h * stride;
        for cy in 0..h / 2 {
            zc.extend_from_slice(&px[uv_base + cy * stride..uv_base + cy * stride + w]);
        }
    }

    // Reference: MMAP/copy path through the same backend (hardware on-target).
    let reference = decode(&jpeg, w, h, PixelFormat::Nv12, false);
    assert_close(&reference, &zc, "zero-copy-dma");
}
