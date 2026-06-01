// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Zero-allocation validation for the codec hot path.
//!
//! Run under `strace` to verify no `brk`/`mmap` syscalls occur after warmup:
//!
//! ```bash
//! cargo build --release -p edgefirst-codec --example zero_alloc_check
//! strace -e brk,mmap -f ./target/release/examples/zero_alloc_check 2>&1 \
//!     | grep -A9999 'HOT LOOP START'
//! ```
//!
//! Expected: zero `brk` or `mmap` lines between "HOT LOOP START" and
//! "HOT LOOP END".

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory};

fn main() {
    let testdata = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("testdata"));
    let jpeg = std::fs::read(testdata.join("zidane.jpg")).expect("testdata/zidane.jpg");

    // Colour JPEGs decode to NV12; allocate at the max expected image size.
    let mut tensor =
        Tensor::<u8>::image(1920, 1080, PixelFormat::Nv12, Some(TensorMemory::Mem)).unwrap();
    let mut decoder = ImageDecoder::new();

    // Warmup: scratch buffer grows to required size
    tensor.load_image(&mut decoder, &jpeg).unwrap();
    eprintln!("=== HOT LOOP START ===");

    for _ in 0..100 {
        tensor.load_image(&mut decoder, &jpeg).unwrap();
    }

    eprintln!("=== HOT LOOP END ===");
    eprintln!("100 decode iterations completed");
}
