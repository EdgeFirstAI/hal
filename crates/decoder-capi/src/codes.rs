// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The decoder option vocabularies, as named C enumerators.
//!
//! Each enumerator's value is asserted against the `code()` of the
//! `edgefirst_decoder` vocabulary it names, at compile time, so the header
//! cannot drift from the Rust declaration. The functions and struct fields
//! that take one keep a plain `uint32_t`: an out-of-range value is rejected
//! rather than transmuted into a Rust enum, which would be undefined
//! behaviour for a value no variant names.

use edgefirst_decoder::configs::{
    DecoderType, DecoderVersion, DimName, Nms, OutputType, NMS_OFF_CODE,
};
use edgefirst_decoder::tiling::{MatchMetric, MergeMode};
use edgefirst_decoder::ModelSource;

/// NMS mode, for `ef_decoder_params_set_nms`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfNms {
    /// No NMS (end-to-end models with NMS in the graph).
    Off = 0,
    /// The model config's mode, else class-aware.
    Auto = 1,
    /// Suppress only boxes that share a class label.
    ClassAware = 2,
    /// Suppress overlapping boxes regardless of class label.
    ClassAgnostic = 3,
}

/// Post-processing family, for `ef_decoder_params_add_output`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfDecoderType {
    Ultralytics = 0,
    ModelPack = 1,
}

/// Ultralytics architecture, for `ef_decoder_params_set_decoder_version`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfDecoderVersion {
    Yolov5 = 0,
    Yolov8 = 1,
    Yolo11 = 2,
    /// End-to-end, with NMS in the model.
    Yolo26 = 3,
}

/// Kind of a model output, for `ef_decoder_params_add_output`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfOutputType {
    Detection = 0,
    Boxes = 1,
    Scores = 2,
    Protos = 3,
    Segmentation = 4,
    MaskCoefficients = 5,
    Mask = 6,
    Classes = 7,
}

/// A named tensor axis, for the `dims` of `ef_decoder_params_add_output`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfDimName {
    Batch = 0,
    Height = 1,
    Width = 2,
    NumClasses = 3,
    NumFeatures = 4,
    NumBoxes = 5,
    NumProtos = 6,
    NumAnchorsXFeatures = 7,
    Padding = 8,
    BoxCoords = 9,
    /// An axis the decoder does not interpret. It never satisfies a
    /// required dimension.
    Unknown = 10,
}

/// Overlap metric, for `ef_merge_config.metric`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfMatchMetric {
    /// Intersection-over-Union.
    Iou = 0,
    /// Intersection-over-Smaller (default).
    Ios = 1,
}

/// What a tiled merge emits per matched group, for `ef_merge_config.mode`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfMergeMode {
    /// Keep the highest-scoring box (default).
    KeepBest = 0,
    /// Replace the group with its enclosing union.
    Union = 1,
}

/// Container format the model was read from, for `ef_infer_signals_new`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfModelSource {
    Onnx = 0,
    TfLite = 1,
    /// Accepted, but inference refuses it: the box convention is unknown.
    Other = 2,
    CoreMl = 3,
}

const _: () = {
    assert!(EfNms::Off as u32 == NMS_OFF_CODE);
    assert!(EfNms::Auto as u32 == Nms::Auto.code());
    assert!(EfNms::ClassAware as u32 == Nms::ClassAware.code());
    assert!(EfNms::ClassAgnostic as u32 == Nms::ClassAgnostic.code());

    assert!(EfDecoderType::Ultralytics as u32 == DecoderType::Ultralytics.code());
    assert!(EfDecoderType::ModelPack as u32 == DecoderType::ModelPack.code());

    assert!(EfDecoderVersion::Yolov5 as u32 == DecoderVersion::Yolov5.code());
    assert!(EfDecoderVersion::Yolov8 as u32 == DecoderVersion::Yolov8.code());
    assert!(EfDecoderVersion::Yolo11 as u32 == DecoderVersion::Yolo11.code());
    assert!(EfDecoderVersion::Yolo26 as u32 == DecoderVersion::Yolo26.code());

    assert!(EfOutputType::Detection as u32 == OutputType::Detection.code());
    assert!(EfOutputType::Boxes as u32 == OutputType::Boxes.code());
    assert!(EfOutputType::Scores as u32 == OutputType::Scores.code());
    assert!(EfOutputType::Protos as u32 == OutputType::Protos.code());
    assert!(EfOutputType::Segmentation as u32 == OutputType::Segmentation.code());
    assert!(EfOutputType::MaskCoefficients as u32 == OutputType::MaskCoefficients.code());
    assert!(EfOutputType::Mask as u32 == OutputType::Mask.code());
    assert!(EfOutputType::Classes as u32 == OutputType::Classes.code());

    assert!(EfDimName::Batch as u32 == DimName::Batch.code());
    assert!(EfDimName::Height as u32 == DimName::Height.code());
    assert!(EfDimName::Width as u32 == DimName::Width.code());
    assert!(EfDimName::NumClasses as u32 == DimName::NumClasses.code());
    assert!(EfDimName::NumFeatures as u32 == DimName::NumFeatures.code());
    assert!(EfDimName::NumBoxes as u32 == DimName::NumBoxes.code());
    assert!(EfDimName::NumProtos as u32 == DimName::NumProtos.code());
    assert!(EfDimName::NumAnchorsXFeatures as u32 == DimName::NumAnchorsXFeatures.code());
    assert!(EfDimName::Padding as u32 == DimName::Padding.code());
    assert!(EfDimName::BoxCoords as u32 == DimName::BoxCoords.code());
    assert!(EfDimName::Unknown as u32 == DimName::Unknown.code());

    assert!(EfMatchMetric::Iou as u32 == MatchMetric::Iou.code());
    assert!(EfMatchMetric::Ios as u32 == MatchMetric::Ios.code());

    assert!(EfMergeMode::KeepBest as u32 == MergeMode::KeepBest.code());
    assert!(EfMergeMode::Union as u32 == MergeMode::Union.code());

    assert!(EfModelSource::Onnx as u32 == ModelSource::Onnx.code());
    assert!(EfModelSource::TfLite as u32 == ModelSource::TfLite.code());
    assert!(EfModelSource::Other as u32 == ModelSource::Other.code());
    assert!(EfModelSource::CoreMl as u32 == ModelSource::CoreMl.code());
};

#[cfg(test)]
mod tests {
    use super::*;

    fn header_text() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/decoder.h"
        ))
        .expect("cbindgen wrote include/edgefirst/decoder.h")
    }

    /// The C enumerator for a variant, spelled the way cbindgen's
    /// `ScreamingSnakeCase` spells the matching `Ef*` variant (which shares
    /// the Rust variant's name): `NumAnchorsXFeatures` ->
    /// `NUM_ANCHORS_X_FEATURES`.
    fn enumerator(prefix: &str, v: impl std::fmt::Debug) -> String {
        let mut out = format!("{prefix}_");
        for (i, c) in format!("{v:?}").chars().enumerate() {
            if c.is_ascii_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    }

    #[test]
    fn every_vocabulary_variant_has_a_c_enumerator() {
        // The const block proves the listed values agree; this proves the
        // lists are complete.
        assert_eq!(Nms::all().len(), 3, "add the EfNms enumerator");
        assert_eq!(
            DecoderType::all().len(),
            2,
            "add the EfDecoderType enumerator"
        );
        assert_eq!(
            DecoderVersion::all().len(),
            4,
            "add the EfDecoderVersion enumerator"
        );
        assert_eq!(
            OutputType::all().len(),
            8,
            "add the EfOutputType enumerator"
        );
        assert_eq!(DimName::all().len(), 11, "add the EfDimName enumerator");
        assert_eq!(
            MatchMetric::all().len(),
            2,
            "add the EfMatchMetric enumerator"
        );
        assert_eq!(MergeMode::all().len(), 2, "add the EfMergeMode enumerator");
        assert_eq!(
            ModelSource::all().len(),
            4,
            "add the EfModelSource enumerator"
        );
    }

    #[test]
    fn the_header_declares_every_option_code() {
        let header = header_text();
        let mut expected: Vec<(String, u32)> = vec![("EF_NMS_OFF".into(), NMS_OFF_CODE)];
        expected.extend(
            Nms::all()
                .iter()
                .map(|&v| (enumerator("EF_NMS", v), v.code())),
        );
        expected.extend(
            DecoderType::all()
                .iter()
                .map(|&v| (enumerator("EF_DECODER_TYPE", v), v.code())),
        );
        expected.extend(
            DecoderVersion::all()
                .iter()
                .map(|&v| (enumerator("EF_DECODER_VERSION", v), v.code())),
        );
        expected.extend(
            OutputType::all()
                .iter()
                .map(|&v| (enumerator("EF_OUTPUT_TYPE", v), v.code())),
        );
        expected.extend(
            DimName::all()
                .iter()
                .map(|&v| (enumerator("EF_DIM_NAME", v), v.code())),
        );
        expected.extend(
            MatchMetric::all()
                .iter()
                .map(|&v| (enumerator("EF_MATCH_METRIC", v), v.code())),
        );
        expected.extend(
            MergeMode::all()
                .iter()
                .map(|&v| (enumerator("EF_MERGE_MODE", v), v.code())),
        );
        expected.extend(
            ModelSource::all()
                .iter()
                .map(|&v| (enumerator("EF_MODEL_SOURCE", v), v.code())),
        );
        for (name, code) in expected {
            let line = format!("{name} = {code},");
            assert!(
                header.contains(&line),
                "decoder.h does not declare `{line}`"
            );
        }
    }

    #[test]
    fn unknown_option_codes_decode_to_nothing() {
        assert_eq!(Nms::from_option_code(4), None);
        assert_eq!(DecoderType::from_code(2), None);
        assert_eq!(DimName::from_code(11), None);
        assert_eq!(DecoderVersion::from_code(4), None);
        assert_eq!(OutputType::from_code(8), None);
        assert_eq!(MatchMetric::from_code(2), None);
        assert_eq!(MergeMode::from_code(2), None);
        assert_eq!(ModelSource::from_code(4), None);
    }
}
