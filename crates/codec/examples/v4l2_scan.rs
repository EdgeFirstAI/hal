// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic: decode every JPEG in a directory through the codec (hardware
//! path when available) and report per-image decode time, flagging slow
//! frames. Used to isolate images that stall a V4L2 decoder — this is how the
//! mxc-jpeg APP13 embedded-thumbnail wedge (COCO 000000122046.jpg) was found.
//!
//! Usage: v4l2_scan <dir> [slow_ms]

use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: v4l2_scan <dir> [slow_ms]");
    let slow_ms: u128 = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    let mut decoder = ImageDecoder::new();
    let mut slow = 0usize;
    let mut failed = 0usize;
    let total = files.len();
    for (i, path) in files.iter().enumerate() {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("READ-FAIL {} {e}", path.display());
                failed += 1;
                continue;
            }
        };
        let Ok(info) = peek_info(&data) else {
            eprintln!("PEEK-FAIL {}", path.display());
            failed += 1;
            continue;
        };
        // NV24 capacity covers every native output format the codec emits.
        let mut t = Tensor::<u8>::image(
            info.width,
            info.height,
            PixelFormat::Nv24,
            Some(TensorMemory::Mem),
        )
        .expect("tensor alloc");
        let t0 = Instant::now();
        match t.load_image(&mut decoder, &data) {
            Ok(_) => {
                let ms = t0.elapsed().as_millis();
                if ms >= slow_ms {
                    slow += 1;
                    eprintln!(
                        "SLOW {ms:5}ms {}x{} {}",
                        info.width,
                        info.height,
                        path.display()
                    );
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("DECODE-FAIL {} {e}", path.display());
            }
        }
        if (i + 1) % 1000 == 0 {
            eprintln!("... {}/{total}", i + 1);
        }
    }
    eprintln!("done: {total} images, {slow} slow (>= {slow_ms}ms), {failed} failed");
}
