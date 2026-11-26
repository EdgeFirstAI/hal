// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A V4L2 JPEG decoder reconfigured for each image's geometry and
//! subsampling keeps its file descriptor count steady. Where no V4L2 JPEG
//! decoder is present the decodes run on the CPU.
#![cfg(all(target_os = "linux", feature = "v4l2"))]

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};

#[path = "common/sessions.rs"]
mod sessions;

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

#[test]
fn v4l2_reconfigure_loop_stable() {
    let _sessions = sessions::take(1);
    // Geometry and subsampling changes through one decoder, without leaking fds.
    let images: Vec<(Vec<u8>, usize, usize, PixelFormat)> = vec![
        (testdata("zidane.jpg"), 1280, 720, PixelFormat::Nv12),
        (testdata("coco_420_odd.jpg"), 500, 375, PixelFormat::Nv12),
        (testdata("zidane_444.jpg"), 1280, 720, PixelFormat::Nv24),
        (testdata("giraffe.jpg"), 640, 640, PixelFormat::Nv12),
        (testdata("coco_444_odd.jpg"), 307, 461, PixelFormat::Nv24),
    ];
    let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();

    let mut decoder = ImageDecoder::new();
    // Warm-up pass establishes the device fd and the persistent allocations.
    for (jpeg, w, h, fmt) in &images {
        let mut t = Tensor::<u8>::image(
            *w,
            *h,
            *fmt,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        t.load_image(&mut decoder, jpeg).unwrap();
    }
    let baseline = fd_count();
    for round in 0..5 {
        for (jpeg, w, h, fmt) in &images {
            let mut t = Tensor::<u8>::image(
                *w,
                *h,
                *fmt,
                Some(TensorMemory::Mem),
                edgefirst_tensor::CpuAccess::ReadWrite,
            )
            .unwrap();
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
