// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Frozen fixtures for CPU mask rendering, generated **before** the
//! `Segmentation` container change and the semantic-argmax rewrite.
//!
//! # Why this exists
//!
//! Nothing checked a single rendered mask pixel. `test_segmentation` in
//! `src/cpu/tests.rs` — the only test exercising the semantic (`C > 1`) argmax
//! path — renders, calls `save_jpeg`, and asserts nothing, so it can only fail
//! by panicking; the argmax could return any class and the suite stayed green.
//! `test_segmentation_yolo` asserts one colour-table entry, no pixels. Both
//! mask families are frozen here before the rework touches them.
//!
//! # What is deliberately NOT in the fixture
//!
//! The background is a synthetic analytic ramp, not a decoded JPEG. A decoded
//! source would fold the JPEG decoder's arithmetic into the checksum, so an
//! unrelated codec change would break this gate and a real mask regression
//! could hide behind a "known" fixture churn. The ramp keeps the CRC a
//! function of the mask math alone. The masks themselves are real model
//! outputs — a synthetic mask would not exercise the argmax meaningfully.
//!
//! # Per-architecture
//!
//! The fixture is stored per-arch like `crop_golden`'s, but measured: the
//! aarch64 and x86_64 CRCs came out **identical** (3505191700 / 2921905178),
//! so mask rendering — unlike the YUV convert kernels — has no per-target
//! divergence to accommodate. The split files are kept anyway, because a
//! future kernel could introduce one and a shared file would then have no
//! place to record it. An arch with no frozen file SKIPS loudly rather than
//! passing.
//!
//! **Never regenerate** to make a failure go away — the frozen bytes are the
//! arbiter, and `generate_golden` exists only for the freeze commit.

use edgefirst_image::{CPUProcessor, ImageProcessorTrait};
use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};
use serde::{Deserialize, Serialize};

const DST_W: usize = 320;
const DST_H: usize = 240;

/// One frozen case: the parameters plus the CRC32 of the rendered RGBA bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Case {
    name: String,
    seg_shape: [usize; 3],
    crc32: u32,
}

/// A deterministic RGBA background — every byte a function of its offset, so a
/// mask that lands in the wrong place cannot coincidentally match.
fn background() -> TensorDyn {
    let mut t = TensorDyn::image(
        DST_W,
        DST_H,
        PixelFormat::Rgba,
        DType::U8,
        None,
        CpuAccess::ReadWrite,
    )
    .expect("allocate background");
    {
        let u8t = t.as_u8_mut().expect("u8 background");
        let mut map = u8t.map_mut().expect("map background");
        for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
            *b = (i.wrapping_mul(31).wrapping_add(7) % 251) as u8;
        }
    }
    t
}

/// The two mask families the renderer dispatches between:
/// semantic (`C = num_classes`, argmax per pixel) and instance (`C = 1`,
/// threshold at 128). `chw` marks a fixture stored channel-first that must be
/// transposed to `[H, W, C]` before rendering.
struct Family {
    name: &'static str,
    fixture: &'static str,
    shape: (usize, usize, usize),
    chw: bool,
    /// Instance masks are drawn per detection, so the `C = 1` case needs a
    /// box; the semantic case renders the whole plane and takes none.
    detect: Option<([f32; 4], usize)>,
}

fn families() -> [Family; 2] {
    [
        Family {
            name: "semantic_c2",
            fixture: "modelpack_seg_2x160x160.bin",
            shape: (2, 160, 160),
            chw: true,
            detect: None,
        },
        Family {
            name: "instance_c1",
            fixture: "yolov8_seg_crop_76x55.bin",
            shape: (76, 55, 1),
            chw: false,
            detect: Some(([0.59375, 0.25, 0.9375, 0.725], 1)),
        },
    ]
}

// ── The two functions Task 4 rewrites ───────────────────────────────────────
//
// Everything above is container-agnostic. These two are the only places that
// know `Segmentation` currently carries an `ndarray::Array3<u8>`; when it
// becomes a `TensorDyn` these change and nothing else does. The frozen CRCs
// must not move across that swap — it is a container change, not a value one.

fn build_segmentation(hwc: &[u8], dims: [usize; 3]) -> edgefirst_tensor::Segmentation {
    let segmentation = ndarray::Array3::from_shape_vec((dims[0], dims[1], dims[2]), hwc.to_vec())
        .expect("segmentation shape");
    edgefirst_tensor::Segmentation {
        segmentation: edgefirst_tensor::Tensor::from_arrayview3(segmentation.view())
            .unwrap()
            .into(),
        xmin: 0.0,
        ymin: 0.0,
        xmax: 1.0,
        ymax: 1.0,
    }
}

fn draw(
    proc: &mut CPUProcessor,
    dst: &mut TensorDyn,
    seg: edgefirst_tensor::Segmentation,
    detect: Option<([f32; 4], usize)>,
    name: &str,
) {
    let boxes: Vec<edgefirst_tensor::DetectBox> = detect
        .into_iter()
        .map(|(bbox, label)| edgefirst_tensor::DetectBox {
            bbox: bbox.into(),
            score: 0.99,
            label,
        })
        .collect();
    proc.draw_decoded_masks(dst, &boxes, &[seg], Default::default())
        .unwrap_or_else(|e| panic!("{name}: draw_decoded_masks: {e}"));
}

/// Render one family onto the analytic background and return
/// `(rendered_bytes, [h, w, c])`.
fn render(f: &Family) -> (Vec<u8>, [usize; 3]) {
    let raw = edgefirst_bench::testdata::read(f.fixture);
    let (d0, d1, d2) = f.shape;

    // Transpose CHW -> HWC where needed, by hand rather than via ndarray, so
    // this harness does not depend on a crate the code under test is dropping.
    let (hwc, dims) = if f.chw {
        let (c, h, w) = (d0, d1, d2);
        let mut out = vec![0u8; c * h * w];
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out[(y * w + x) * c + ci] = raw[ci * h * w + y * w + x];
                }
            }
        }
        (out, [h, w, c])
    } else {
        (raw.to_vec(), [d0, d1, d2])
    };
    assert_eq!(hwc.len(), dims[0] * dims[1] * dims[2], "{}: size", f.name);

    let seg = build_segmentation(&hwc, dims);
    let mut dst = background();

    let mut proc = CPUProcessor::new();
    draw(&mut proc, &mut dst, seg, f.detect, f.name);

    let u8t = dst.as_u8().expect("u8 dst");
    let map = u8t.map_read().expect("map dst");
    (map.as_slice().to_vec(), dims)
}

fn fixture_path() -> String {
    format!("tests/data/mask_golden.{}.json", std::env::consts::ARCH)
}

#[test]
#[ignore = "regenerates the frozen fixture; run only at the freeze commit"]
fn generate_golden() {
    let cases: Vec<Case> = families()
        .iter()
        .map(|f| {
            let (bytes, seg_shape) = render(f);
            Case {
                name: f.name.to_string(),
                seg_shape,
                crc32: crc32fast::hash(&bytes),
            }
        })
        .collect();
    std::fs::write(
        fixture_path(),
        serde_json::to_string_pretty(&cases).expect("serialize"),
    )
    .expect("write fixture");
}

/// The fixture is compiled in rather than read from `tests/data/` at runtime.
/// A relative path resolves against the *current working directory*, which on
/// an on-target run is a deployed binary directory with no source tree — so
/// the replay test skipped on every board while looking perfectly green
/// locally. `include_str!` resolves at compile time and travels with the
/// binary, which is what `crop_golden` does for the same reason.
#[cfg(target_arch = "aarch64")]
const GOLDEN_JSON: Option<&str> = Some(include_str!("data/mask_golden.aarch64.json"));
#[cfg(target_arch = "x86_64")]
const GOLDEN_JSON: Option<&str> = Some(include_str!("data/mask_golden.x86_64.json"));
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const GOLDEN_JSON: Option<&str> = None;

#[test]
fn masks_match_golden() {
    let Some(raw) = GOLDEN_JSON else {
        // Not a silent pass: an arch with no frozen fixture must say so, and
        // libtest discards println! for passing tests.
        use std::io::Write;
        let _ = writeln!(
            &mut std::io::stderr(),
            "SKIPPED: no frozen mask fixture for {}",
            std::env::consts::ARCH
        );
        return;
    };
    let want: Vec<Case> = serde_json::from_str(raw).expect("parse fixture");
    assert_eq!(want.len(), families().len(), "fixture case count drifted");

    for (f, w) in families().iter().zip(want) {
        assert_eq!(f.name, w.name, "fixture case order drifted");
        let (bytes, seg_shape) = render(f);
        assert_eq!(
            seg_shape, w.seg_shape,
            "{}: segmentation shape drifted",
            f.name
        );
        assert_eq!(
            crc32fast::hash(&bytes),
            w.crc32,
            "{}: rendered mask bytes changed against the frozen fixture. \
             Fix the code, never the fixture.",
            f.name
        );
    }
}
