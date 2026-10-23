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
    /// An `EF_MATCH_METRIC_*` code: IoU, or Intersection-over-Smaller.
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
    /// An `EF_MERGE_MODE_*` code. `EF_MERGE_MODE_KEEP_BEST` (default): the
    /// highest-scoring box of each matched group is kept and the boxes it
    /// matched are dropped. `EF_MERGE_MODE_UNION`: the group becomes its
    /// enclosing union carrying the max score.
    ///
    /// Keep-best is `0`, but note that a zero-initialised struct is still
    /// not the library default: `metric` defaults to `EF_MATCH_METRIC_IOS`,
    /// so an all-zero struct selects IoU. Fill with `ef_merge_config_default`.
    /// The union measured about 0.05 AP50 worse on every frame of the Ocean
    /// Cleanup ADIS 4K validation (TOP2-836), which is why it is opt-in.
    ///
    /// Occupies what was this struct's 4-byte tail pad, so adding it left
    /// every other field's offset and the 32-byte size unchanged.
    pub mode: u32,
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
    /// The image content on the model input as normalized
    /// `[xmin, ymin, xmax, ymax]` (a letterbox's content region), or `None`
    /// when the crop was stretched to fill it (the hot path).
    pub content_region: Option<[f32; 4]>,
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
    pub has_content_region: c_int,
    pub content_region: [f32; 4],
    pub frame_width: f32,
    pub frame_height: f32,
}

impl From<&TilePlacement> for EfTilePlacement {
    fn from(p: &TilePlacement) -> Self {
        let (has_content_region, content_region) = match p.content_region {
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
            has_content_region,
            content_region,
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
            content_region: if p.has_content_region != 0 {
                Some(p.content_region)
            } else {
                None
            },
            frame_dims: (p.frame_width, p.frame_height),
        }
    }
}

/// How a source is fit into the model input. C `ef_fit` in `detect.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfFit {
    /// Stretch the source to fill the model input.
    Stretch = 0,
    /// Keep the aspect ratio and pad the rest (a letterbox).
    Letterbox = 1,
}

/// Where a letterboxed image sits. C `ef_letterbox_placement` in `detect.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfLetterboxPlacement {
    /// Centred, padding split between both sides.
    Center = 0,
    /// Anchored at the top-left corner, padding on the right and bottom.
    TopLeft = 1,
}

/// The space a letterbox pad is expressed in. C `ef_pad_space` in `detect.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfPadSpace {
    /// A 0-255 pixel value, normalized like the image.
    Pixel = 0,
    /// A value written as-is into the normalized destination.
    Normalized = 1,
}

/// A resampling filter. C `ef_interpolation` in `detect.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfInterpolation {
    /// Bilinear (the default).
    Bilinear = 0,
    /// Nearest neighbour.
    Nearest = 1,
    /// Bicubic; the image library runs bilinear in its place.
    Bicubic = 2,
    /// Pixel-area averaging; the image library runs bilinear in its place.
    Area = 3,
}

/// A model's input preprocessing as plain values. C `ef_preprocess_spec` in
/// `detect.h`, produced by the decoder library and consumed by the image
/// library.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EfPreprocessSpec {
    /// Model input width in pixels, 0 when unknown.
    pub width: u32,
    /// Model input height in pixels, 0 when unknown.
    pub height: u32,
    /// An `EF_FIT_*` code.
    pub fit: u32,
    /// An `EF_LETTERBOX_PLACEMENT_*` code; ignored by a stretch.
    pub placement: u32,
    /// An `EF_PAD_SPACE_*` code for `pad`.
    pub pad_space: u32,
    /// Letterbox pad per RGB channel.
    pub pad: [f32; 3],
    /// Normalization mean per RGB channel, on 0-255 values.
    pub mean: [f32; 3],
    /// Normalization standard deviation per RGB channel, on 0-255 values.
    pub std: [f32; 3],
    /// An `EF_INTERPOLATION_*` code.
    pub interpolation: u32,
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

    #[test]
    fn segmentation_layout_is_pinned() {
        assert_eq!(std::mem::size_of::<EfSegmentation>(), 32);
        assert_eq!(std::mem::align_of::<EfSegmentation>(), 8);
    }

    #[test]
    fn merge_config_layout_is_pinned() {
        // `max_det: usize` forces 8-byte alignment and 4 bytes of pad after
        // `class_agnostic`; `mode` fills what was the tail pad at 28, so the
        // struct is 32 bytes with or without it. Matches `detect.h`.
        assert_eq!(std::mem::size_of::<EfMergeConfig>(), 32);
        assert_eq!(std::mem::align_of::<EfMergeConfig>(), 8);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, metric), 0);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, threshold), 4);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, class_agnostic), 8);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, max_det), 16);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, score_threshold), 24);
        assert_eq!(std::mem::offset_of!(EfMergeConfig, mode), 28);
    }

    #[test]
    fn tile_placement_layout_is_pinned() {
        // 60 bytes of fields, align 8 -> 64. Matches `detect.h`.
        assert_eq!(std::mem::size_of::<EfTilePlacement>(), 64);
        assert_eq!(std::mem::align_of::<EfTilePlacement>(), 8);
    }

    #[test]
    fn preprocess_spec_layout_is_pinned() {
        // Five u32, three [f32; 3], one u32: 60 bytes, align 4. Matches
        // `detect.h`.
        type S = EfPreprocessSpec;
        assert_eq!(std::mem::size_of::<S>(), 60);
        assert_eq!(std::mem::align_of::<S>(), 4);
        assert_eq!(std::mem::offset_of!(S, width), 0);
        assert_eq!(std::mem::offset_of!(S, height), 4);
        assert_eq!(std::mem::offset_of!(S, fit), 8);
        assert_eq!(std::mem::offset_of!(S, placement), 12);
        assert_eq!(std::mem::offset_of!(S, pad_space), 16);
        assert_eq!(std::mem::offset_of!(S, pad), 20);
        assert_eq!(std::mem::offset_of!(S, mean), 32);
        assert_eq!(std::mem::offset_of!(S, std), 44);
        assert_eq!(std::mem::offset_of!(S, interpolation), 56);
    }

    #[test]
    fn tile_placement_offsets_are_pinned() {
        type P = EfTilePlacement;
        assert_eq!(std::mem::offset_of!(P, has_content_region), 32);
        assert_eq!(std::mem::offset_of!(P, content_region), 36);
        assert_eq!(std::mem::offset_of!(P, frame_width), 52);
    }

    #[test]
    fn detect_h_declares_the_preprocess_vocabulary() {
        let h = include_str!("../include/edgefirst/detect.h");
        let codes = [
            ("EF_FIT_STRETCH", EfFit::Stretch as u32),
            ("EF_FIT_LETTERBOX", EfFit::Letterbox as u32),
            (
                "EF_LETTERBOX_PLACEMENT_CENTER",
                EfLetterboxPlacement::Center as u32,
            ),
            (
                "EF_LETTERBOX_PLACEMENT_TOP_LEFT",
                EfLetterboxPlacement::TopLeft as u32,
            ),
            ("EF_PAD_SPACE_PIXEL", EfPadSpace::Pixel as u32),
            ("EF_PAD_SPACE_NORMALIZED", EfPadSpace::Normalized as u32),
            (
                "EF_INTERPOLATION_BILINEAR",
                EfInterpolation::Bilinear as u32,
            ),
            ("EF_INTERPOLATION_NEAREST", EfInterpolation::Nearest as u32),
            ("EF_INTERPOLATION_BICUBIC", EfInterpolation::Bicubic as u32),
            ("EF_INTERPOLATION_AREA", EfInterpolation::Area as u32),
        ];
        for (name, code) in codes {
            let line = format!("{name} = {code},");
            assert!(h.contains(&line), "detect.h does not declare `{line}`");
        }
        assert!(h.contains("typedef struct ef_preprocess_spec {"));
    }
}
