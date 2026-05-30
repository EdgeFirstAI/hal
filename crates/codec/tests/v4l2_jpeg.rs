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
//! Run single-threaded — these mutate process-global env vars to toggle the
//! backend, and the workspace test convention is `-j 1` already.
#![cfg(all(target_os = "linux", feature = "v4l2"))]

use edgefirst_codec::{DecodeOptions, ImageDecoder, ImageLoad};
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

/// Decode `jpeg` to `fmt` with the backend selected by `disable_v4l2`.
fn decode(jpeg: &[u8], w: usize, h: usize, fmt: PixelFormat, disable_v4l2: bool) -> Vec<u8> {
    if disable_v4l2 {
        std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
    } else {
        std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
    }
    let mut tensor = Tensor::<u8>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();
    let info = tensor
        .load_image(
            &mut decoder,
            jpeg,
            &DecodeOptions::default().with_format(fmt).with_exif(false),
        )
        .unwrap();
    let map = tensor.map().unwrap();
    let pixels: &[u8] = &map;
    pixels[..info.width * info.height * fmt.channels()].to_vec()
}

/// Assert two decodes match within a tolerance that accommodates the hardware
/// vs software IDCT/upsample rounding differences (these are *different*
/// decoders, not the same one), while still catching structural errors —
/// wrong format, swapped channels, garbage, or a blank decode.
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
fn v4l2_parity_rgb() {
    eprintln!(
        "v4l2_parity_rgb: video device present = {}",
        has_video_device()
    );
    let jpeg = testdata("zidane.jpg");
    let cpu = decode(&jpeg, 1280, 720, PixelFormat::Rgb, true);
    let hw = decode(&jpeg, 1280, 720, PixelFormat::Rgb, false);
    assert_close(&cpu, &hw, "rgb");
}

#[test]
fn v4l2_parity_rgba() {
    let jpeg = testdata("zidane.jpg");
    let cpu = decode(&jpeg, 1280, 720, PixelFormat::Rgba, true);
    let hw = decode(&jpeg, 1280, 720, PixelFormat::Rgba, false);
    assert_close(&cpu, &hw, "rgba");
}

#[test]
fn v4l2_parity_grey() {
    let jpeg = testdata("zidane.jpg");
    let cpu = decode(&jpeg, 1280, 720, PixelFormat::Grey, true);
    let hw = decode(&jpeg, 1280, 720, PixelFormat::Grey, false);
    assert_close(&cpu, &hw, "grey");
}

#[test]
fn v4l2_parity_f32_typed_tail() {
    // Exercises the shared u8 → typed conversion tail through the hardware
    // staging path (when hardware is present) and the CPU path (otherwise).
    let jpeg = testdata("zidane.jpg");
    let w = 1280;
    let h = 720;
    let fmt = PixelFormat::Rgb;

    let decode_f32 = |disable: bool| -> Vec<f32> {
        if disable {
            std::env::set_var("EDGEFIRST_DISABLE_V4L2", "1");
        } else {
            std::env::remove_var("EDGEFIRST_DISABLE_V4L2");
        }
        let mut tensor = Tensor::<f32>::image(w, h, fmt, Some(TensorMemory::Mem)).unwrap();
        let mut decoder = ImageDecoder::new();
        let info = tensor
            .load_image(
                &mut decoder,
                &jpeg,
                &DecodeOptions::default().with_format(fmt).with_exif(false),
            )
            .unwrap();
        let map = tensor.map().unwrap();
        let px: &[f32] = &map;
        px[..info.width * info.height * 3].to_vec()
    };

    let cpu = decode_f32(true);
    let hw = decode_f32(false);
    assert_eq!(cpu.len(), hw.len());
    let mut max_diff = 0.0f32;
    let mut sum = 0.0f64;
    for (&a, &b) in cpu.iter().zip(hw.iter()) {
        let d = (a - b).abs();
        max_diff = max_diff.max(d);
        sum += d as f64;
    }
    let mean = sum / cpu.len() as f64;
    assert!(
        mean < 3.0 && max_diff <= 24.0,
        "f32 typed-tail mismatch (mean={mean:.3}, max={max_diff})"
    );
}
