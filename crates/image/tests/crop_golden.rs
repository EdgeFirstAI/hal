// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Golden fixtures for the CPU backend's cropped-convert output, frozen
//! **before** any crop-contract implementation work starts.
//!
//! `generate_golden` (marked `#[ignore]`) regenerates
//! `tests/data/crop_golden.<arch>.json` from whatever the CPU backend
//! currently produces. `cropped_convert_matches_golden` replays the same
//! matrix against the fixture and is the guard later crop-contract tasks must
//! keep green — any behavioural change to cropped-convert byte output shows
//! up here first.
//!
//! The fixtures are **per-architecture**: the `yuv` crate's NEON (aarch64)
//! and AVX2/SSE (x86_64) kernels round YUV→RGB differently, so every
//! NV-source case's bytes differ between the two (48 of 80 cases; the 32
//! RGB-source resize-only cases are identical — `fast_image_resize` is
//! arch-consistent). Both fixture files were generated from the PRE-change
//! implementation at the Task 1 freeze commit: aarch64 on an Apple M2 Max
//! (validated against Linux aarch64 NEON by CI), x86_64 in an AVX2 x86
//! container from the same tree. Within one architecture the frozen bytes
//! are the byte-exactness arbiter; neither file is ever regenerated.

use edgefirst_image::{CPUProcessor, Crop, Fit, Flip, ImageProcessorTrait, Region, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use serde::{Deserialize, Serialize};

const SRC_W: usize = 640;
const SRC_H: usize = 480;

/// One golden-fixture entry: the case parameters plus the CRC32 of the CPU
/// backend's converted output bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Case {
    src_fmt: String,
    dst_fmt: String,
    src_w: usize,
    src_h: usize,
    crop: [usize; 4],
    dst_w: usize,
    dst_h: usize,
    crc32: u32,
}

fn checksum(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

/// Source pixel formats under test: two sub-sampling ratios (Nv12, Nv16),
/// full-resolution chroma (Nv24), and the two packed RGB families.
fn src_formats() -> [PixelFormat; 5] {
    [
        PixelFormat::Nv12,
        PixelFormat::Nv16,
        PixelFormat::Nv24,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
    ]
}

/// Destination pixel formats: packed (fast direct path) and planar (the
/// path Task 2-4's crop-contract work will change).
fn dst_formats() -> [PixelFormat; 2] {
    [PixelFormat::Rgb, PixelFormat::PlanarRgb]
}

/// Source-crop rectangles `(left, top, width, height)` inside the 640x480
/// source: interior, top-left corner, bottom-right (flush against the source
/// edge), and an even-origin variant. The interior crop's origin (101, 53) is
/// already odd on both axes, so a separate "odd-origin" case would be
/// identical — deduped to these four per the plan's ambiguity resolution.
fn crops() -> [(usize, usize, usize, usize); 4] {
    [
        (101, 53, 320, 240),                  // interior / odd-origin
        (0, 0, 320, 240),                     // top-left
        (SRC_W - 320, SRC_H - 240, 320, 240), // bottom-right (flush)
        (100, 52, 320, 240),                  // even-origin
    ]
}

/// Destination sizes: scale-identity (matches the crop size) and a 2x
/// downscale, so the matrix covers both the direct and resize paths.
fn dst_sizes() -> [(usize, usize); 2] {
    [(320, 240), (160, 120)]
}

/// Build a 640x480 source tensor in `fmt`, filled with a deterministic
/// gradient across every mapped byte: `p = ((i*7 + i/640*13) & 0xFF) as u8`.
fn make_src(fmt: PixelFormat) -> TensorDyn {
    let src = TensorDyn::image(
        SRC_W,
        SRC_H,
        fmt,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let u8t = src.as_u8().unwrap();
        let mut map = u8t.map_mut().unwrap();
        for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
            *b = ((i * 7 + i / 640 * 13) & 0xFF) as u8;
        }
    }
    src
}

/// Run one cropped-convert case on the CPU backend and return the CRC32 of
/// the destination bytes, or `None` if this (src_fmt, dst_fmt, crop) triple
/// is rejected by the current implementation.
fn run_case(
    proc_: &mut CPUProcessor,
    src: &TensorDyn,
    dst_fmt: PixelFormat,
    crop_rect: (usize, usize, usize, usize),
    dst_w: usize,
    dst_h: usize,
) -> Option<u32> {
    let mut dst = TensorDyn::image(
        dst_w,
        dst_h,
        dst_fmt,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .ok()?;

    let (left, top, width, height) = crop_rect;
    let crop = Crop::new()
        .with_source(Some(Region::new(left, top, width, height)))
        .with_fit(Fit::Stretch);

    proc_
        .convert(src, &mut dst, Rotation::None, Flip::None, crop)
        .ok()?;

    let u8t = dst.as_u8().unwrap();
    let map = u8t.map().unwrap();
    Some(checksum(map.as_slice()))
}

/// Enumerate the full matrix (5 src formats x 2 dst formats x 4 crops x 2 dst
/// sizes = 80 cells), skipping any cell the current implementation rejects.
fn generate_cases() -> Vec<Case> {
    let mut proc_ = CPUProcessor::default();
    let mut cases = Vec::new();

    for &src_fmt in &src_formats() {
        let src = make_src(src_fmt);
        for &dst_fmt in &dst_formats() {
            for &crop_rect in &crops() {
                for &(dst_w, dst_h) in &dst_sizes() {
                    let Some(crc32) = run_case(&mut proc_, &src, dst_fmt, crop_rect, dst_w, dst_h)
                    else {
                        eprintln!(
                            "crop_golden: skipping rejected case {src_fmt:?} -> {dst_fmt:?} \
                             crop={crop_rect:?} dst={dst_w}x{dst_h}"
                        );
                        continue;
                    };
                    cases.push(Case {
                        src_fmt: format!("{src_fmt:?}"),
                        dst_fmt: format!("{dst_fmt:?}"),
                        src_w: SRC_W,
                        src_h: SRC_H,
                        crop: [crop_rect.0, crop_rect.1, crop_rect.2, crop_rect.3],
                        dst_w,
                        dst_h,
                        crc32,
                    });
                }
            }
        }
    }
    cases
}

/// The architecture-specific fixture, as a compiled-in string. Architectures
/// without a frozen fixture fail to compile here deliberately: freeze one
/// from the pre-change implementation before enabling the guard on a new
/// architecture, never from a tree with crop-contract work already applied.
#[cfg(target_arch = "aarch64")]
const GOLDEN_JSON: &str = include_str!("data/crop_golden.aarch64.json");
#[cfg(target_arch = "x86_64")]
const GOLDEN_JSON: &str = include_str!("data/crop_golden.x86_64.json");

#[test]
#[ignore = "generator: writes tests/data/crop_golden.<arch>.json from the CURRENT implementation"]
fn generate_golden() {
    let cases = generate_cases();
    assert!(!cases.is_empty(), "no golden cases were generated");

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/data/crop_golden.{}.json",
        std::env::consts::ARCH
    ));
    let file = std::fs::File::create(&path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    serde_json::to_writer_pretty(file, &cases).unwrap();
    eprintln!(
        "crop_golden: wrote {} cases to {}",
        cases.len(),
        path.display()
    );
}

#[test]
fn cropped_convert_matches_golden() {
    let fixtures: Vec<Case> = serde_json::from_str(GOLDEN_JSON).unwrap();
    assert!(!fixtures.is_empty(), "golden fixture file is empty");

    // Sanity: fixtures must have distinct checksums (cases actually differ),
    // otherwise the matrix isn't exercising anything meaningful.
    let mut distinct: Vec<u32> = fixtures.iter().map(|c| c.crc32).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        fixtures.len(),
        "golden fixtures contain duplicate checksums across distinct cases"
    );

    let mut proc_ = CPUProcessor::default();
    for case in &fixtures {
        let src_fmt = parse_pixel_format(&case.src_fmt);
        let dst_fmt = parse_pixel_format(&case.dst_fmt);
        let src = make_src(src_fmt);
        let crop_rect = (case.crop[0], case.crop[1], case.crop[2], case.crop[3]);

        let crc32 = run_case(&mut proc_, &src, dst_fmt, crop_rect, case.dst_w, case.dst_h)
            .unwrap_or_else(|| panic!("case now rejected by the CPU backend: {case:?}"));

        assert_eq!(crc32, case.crc32, "checksum mismatch for {case:?}");
    }
}

fn parse_pixel_format(name: &str) -> PixelFormat {
    match name {
        "Nv12" => PixelFormat::Nv12,
        "Nv16" => PixelFormat::Nv16,
        "Nv24" => PixelFormat::Nv24,
        "Rgb" => PixelFormat::Rgb,
        "Rgba" => PixelFormat::Rgba,
        "PlanarRgb" => PixelFormat::PlanarRgb,
        other => panic!("unexpected pixel format in fixture: {other}"),
    }
}
