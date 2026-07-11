// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! On-target smoke test for the nvJPEG GPU decode backend.
//!
//! Decodes a JPEG through the public codec API into a PBO destination created by
//! `ImageProcessor::create_image` (a CUDA-registered PBO on Jetson), reports
//! which backend fired (RGB ⇒ nvJPEG, NV12/Grey ⇒ V4L2/CPU), verifies the
//! decoded pixels are non-trivial, and times the steady-state decode.
//!
//! nvJPEG is opt-in (off by default, to avoid contending with CUDA inference),
//! so `EDGEFIRST_ENABLE_NVJPEG=1` is required for the GPU path to engage:
//!
//! ```bash
//! EDGEFIRST_ENABLE_NVJPEG=1 \
//!   LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib \
//!   cargo run -p edgefirst-codec --example nvjpeg_decode -- image.jpg
//! ```

use edgefirst_codec::{nvjpeg_available, peek_info, ImageDecoder, ImageLoad};
use edgefirst_image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, Rotation};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMemory, TensorTrait};
use std::time::Instant;

/// min / max / mean over every 7th byte — a cheap "is this a real image?" check.
/// A coherency failure (stale/zero buffer) shows up as min==max or all-zero.
fn stats(bytes: &[u8]) -> (u8, u8, f64) {
    let mut min = u8::MAX;
    let mut max = 0u8;
    let mut sum = 0u64;
    let mut n = 0u64;
    for &b in bytes.iter().step_by(7) {
        min = min.min(b);
        max = max.max(b);
        sum += b as u64;
        n += 1;
    }
    (min, max, if n > 0 { sum as f64 / n as f64 } else { 0.0 })
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    let path = std::env::args()
        .nth(1)
        .expect("usage: nvjpeg_decode <file.jpg>");
    let data = std::fs::read(&path).expect("read jpeg");

    println!("nvjpeg_available() = {}", nvjpeg_available());
    let probe = peek_info(&data).expect("peek_info");
    println!(
        "image: {}x{} native {:?} ({} bytes)",
        probe.width,
        probe.height,
        probe.format,
        data.len()
    );

    let mut processor = ImageProcessor::new().expect("ImageProcessor::new");
    let mut tensor = processor
        .create_image(
            probe.width,
            probe.height,
            PixelFormat::Rgb,
            DType::U8,
            None,
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("create_image");
    println!("destination tensor memory = {:?}", tensor.memory());
    println!(
        "destination has CUDA handle = {}",
        tensor.cuda_map().is_some()
    );

    let mut decoder = ImageDecoder::new();
    let info = tensor.load_image(&mut decoder, &data).expect("decode");
    let backend = match info.format {
        PixelFormat::Rgb => "nvJPEG (GPU, RGB)",
        PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24 | PixelFormat::Grey => {
            "V4L2/CPU (native)"
        }
        _ => "other",
    };
    println!(
        "decoded format = {:?}  →  backend = {}",
        info.format, backend
    );

    // Correctness 1: read back what the decoder wrote into the device buffer
    // (CUDA-write correctness). cuda_map → cudaMemcpy DeviceToHost.
    if let Some(map) = tensor.cuda_map() {
        let bytes = info.width * info.height * 3;
        let n = bytes.min(map.len());
        let mut host = vec![0u8; n];
        // SAFETY: host holds n bytes; device_ptr covers the mapped PBO (len()).
        let ok = unsafe {
            edgefirst_tensor::memcpy_device_to_host(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                map.device_ptr(),
                n,
            )
        };
        let (mn, mx, mean) = stats(&host);
        println!(
            "decoded PBO readback: memcpy_ok={ok} min={mn} max={mx} mean={mean:.1} \
             ({})",
            if mx > mn {
                "real image ✓"
            } else {
                "UNIFORM — coherency suspect ✗"
            }
        );
    }

    // Correctness 2: full zero-copy chain — convert() consumes the RGB PBO
    // source (GL-read of the CUDA-written buffer) and letterboxes to a CPU
    // tensor we can verify.
    let mut dst = processor
        .create_image(
            640,
            640,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .expect("create dst");
    match processor.convert(
        &tensor,
        &mut dst,
        Rotation::None,
        Flip::None,
        Crop::letterbox([0, 0, 0, 255]),
    ) {
        Ok(()) => {
            if let TensorDyn::U8(t) = &dst {
                let m = t.map().expect("map dst");
                let (mn, mx, mean) = stats(&m);
                println!(
                    "convert() → 640x640 RGB: min={mn} max={mx} mean={mean:.1} ({})",
                    if mx > mn {
                        "real output ✓ (decode→convert chain verified)"
                    } else {
                        "UNIFORM — chain broken ✗"
                    }
                );
            }
        }
        Err(e) => println!("convert() FAILED: {e}"),
    }

    // Steady-state timing.
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        tensor.load_image(&mut decoder, &data).expect("decode");
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("avg decode = {ms:.2} ms over {iters} iters");
}
