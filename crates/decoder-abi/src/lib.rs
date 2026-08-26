// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Layout-only C ABI contract for the EdgeFirst detection vocabulary.
//!
//! Decoding model outputs into user-consumable structures — detection boxes,
//! segmentation masks, tile placements — is what the decoder library *is*, so
//! these types belong to it. They lived in `edgefirst-tensor` because of the
//! r2 "plain-values-only boundary rule", whose premise (sibling libraries
//! never link each other) spec Revision 3 overturned. A plain value that
//! crosses needs a shared DECLARATION, not a shared LINK.
//!
//! Emitted into `edgefirst/decoder.h` by cbindgen's `[parse] include`,
//! exactly as `edgefirst-tensor-abi`'s types are emitted into `tensor.h`.

use std::ffi::c_int;

/// A detection: a normalized box, a score, and a label index.
///
/// Coordinates are normalized to `[0, 1]` against the *model input*, not the
/// source image — un-letterboxing is the consumer's job and needs the letterbox
/// parameters, which this type deliberately does not carry.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EfDetectBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    /// Model-specific confidence; higher is more confident.
    pub score: f32,
    /// Label index into the model's class list.
    pub label: u32,
}

/// One segmentation result, as plain values.
///
/// Crosses library boundaries by the same rule as [`EfDetectBox`]: plain
/// values, never a handle. `mask` **borrows** the producing list's buffer and
/// is valid only until that list is freed — stated here because C gives a
/// caller no way to discover it.
///
/// The bounds describe the **mask region**, which is snapped to the proto grid
/// and therefore encloses rather than equals the companion detection's box.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfSegmentation {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    /// Row-major mask bytes, `height * width`, borrowed from the list.
    pub mask: *const u8,
    pub width: u32,
    pub height: u32,
}

/// How overlapping detections from neighbouring tiles are merged.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfMergeConfig {
    /// 0 = IoU, 1 = Intersection-over-Smaller.
    ///
    /// IoS is the default for a reason: an object split across a tile seam has
    /// *low* IoU with its own fragment but high IoS, so IoU would keep both
    /// halves as separate detections.
    pub metric: u32,
    pub threshold: f32,
    /// Non-zero to merge across classes.
    pub class_agnostic: c_int,
    pub max_det: usize,
    pub score_threshold: f32,
}

/// One tile's placement within a frame grid.
///
/// Produced by `edgefirst_image::ImageProcessor::plan_tiles` and consumed by
/// `edgefirst_decoder::tiling` to lift per-tile detections back to full-frame
/// coordinates. It lives here because it is the contract *between* those two
/// crates: it was previously defined in `decoder` and re-exported by `image`,
/// so the producer imported its own output type from its consumer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilePlacement {
    /// Tile index within the frame grid, `0..count`.
    pub index: usize,
    /// Total tiles for this frame (the streaming fan-in fence).
    pub count: usize,
    /// Native crop origin `(ox, oy)` in full-frame pixels.
    pub origin: (f32, f32),
    /// Native crop size `(cw, ch)` in full-frame pixels. Equals the tile size
    /// for the full-size tiles the EvenDist grid produces.
    pub crop_size: (f32, f32),
    /// Normalized letterbox content bounds `[lx0, ly0, lx1, ly1]` on the model
    /// input, or `None` when the crop was stretched to fill it (the hot path).
    pub letterbox: Option<[f32; 4]>,
    /// Full-frame dimensions `(frame_w, frame_h)` in pixels.
    pub frame_dims: (f32, f32),
}

/// C layout of `ef_tile_placement` in `detect.h`.
///
/// One declaration for image-capi (producer) and decoder-capi (consumer).
/// cbindgen excludes this type so the header-only `detect.h` definition wins.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTilePlacement {
    pub index: usize,
    pub count: usize,
    pub origin_x: f32,
    pub origin_y: f32,
    pub crop_width: f32,
    pub crop_height: f32,
    pub has_letterbox: c_int,
    pub letterbox: [f32; 4],
    pub frame_width: f32,
    pub frame_height: f32,
}

impl From<&TilePlacement> for EfTilePlacement {
    fn from(p: &TilePlacement) -> Self {
        let (has_letterbox, letterbox) = match p.letterbox {
            Some(lb) => (1, lb),
            None => (0, [0.0; 4]),
        };
        Self {
            index: p.index,
            count: p.count,
            origin_x: p.origin.0,
            origin_y: p.origin.1,
            crop_width: p.crop_size.0,
            crop_height: p.crop_size.1,
            has_letterbox,
            letterbox,
            frame_width: p.frame_dims.0,
            frame_height: p.frame_dims.1,
        }
    }
}

impl From<&EfTilePlacement> for TilePlacement {
    fn from(p: &EfTilePlacement) -> Self {
        Self {
            index: p.index,
            count: p.count,
            origin: (p.origin_x, p.origin_y),
            crop_size: (p.crop_width, p.crop_height),
            letterbox: if p.has_letterbox != 0 {
                Some(p.letterbox)
            } else {
                None
            },
            frame_dims: (p.frame_width, p.frame_height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load-bearing: `edgefirst-image` and `edgefirst-tracker` depend on this
    /// crate to NAME decoder's types without linking decoder's implementation.
    /// A single dependency here would drag code into both and defeat that.
    #[test]
    fn this_crate_has_no_dependencies_and_never_may() {
        let manifest = include_str!("../Cargo.toml");
        assert!(
            !manifest.contains("[dependencies."),
            "dependency sub-tables are dependencies too"
        );
        let deps = manifest
            .split("[dependencies]")
            .nth(1)
            .expect("manifest has a [dependencies] table");
        let body = deps.split("\n[").next().unwrap_or(deps);
        let lines: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(
            lines.is_empty(),
            "edgefirst-decoder-abi grew dependencies: {lines:?}"
        );
    }

    #[test]
    fn detect_box_is_six_packed_32_bit_fields() {
        assert_eq!(std::mem::size_of::<EfDetectBox>(), 24);
        assert_eq!(std::mem::align_of::<EfDetectBox>(), 4);
    }
}
