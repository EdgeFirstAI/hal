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
use edgefirst_decoder::model_config::{ConfigField, ConfigSource, ModelFamily, ModelTask};
use edgefirst_decoder::schema::DecodeMode;
use edgefirst_decoder::tiling::{MatchMetric, MergeMode};
use edgefirst_decoder::ModelSource;
use edgefirst_decoder_abi::{EfFit, EfInterpolation, EfLetterboxPlacement, EfPadSpace};
use edgefirst_tensor::{FitMode, Interpolation, LetterboxPlacement, PadSpace};

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
    /// Generic semantic segmentation, decoded by `ef_decoder_decode_semantic`.
    Semantic = 2,
}

/// Semantic segmentation decode mode, for `ef_semantic_options_set_mode`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfDecodeMode {
    /// Upsample the scores of every class, then take the argmax.
    Accurate = 0,
    /// Argmax at the model's output resolution, then nearest upsampling.
    Fast = 1,
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
    /// Query axis of mask-classification outputs.
    NumQueries = 11,
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

/// The source of a resolved model configuration value, from
/// `ef_model_config_source`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfConfigSource {
    /// `edgefirst.json` embedded in the model file.
    EmbeddedEdgefirst = 0,
    /// `edgefirst.json` (and `labels.txt`) beside the model file.
    SidecarEdgefirst = 1,
    /// Hugging Face configs embedded in the model file.
    EmbeddedHf = 2,
    /// Hugging Face configs beside the model file.
    SidecarHf = 3,
    /// Ultralytics export metadata.
    UltralyticsMetadata = 4,
    /// Guessed from the model's input and output names and shapes.
    Heuristic = 5,
    /// The documented default.
    Default = 6,
    /// A caller override (`ef_model_overrides_set`).
    Override = 7,
}

/// The family a model configuration identifies, from
/// `ef_model_config_family`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfModelFamily {
    /// Described by an `edgefirst.json`.
    EdgeFirst = 0,
    /// An Ultralytics export.
    Ultralytics = 1,
    /// A Hugging Face export.
    HuggingFace = 2,
    /// Configured only by overrides.
    Unknown = 3,
}

/// What a model's outputs decode to, from `ef_model_config_task`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfModelTask {
    /// Bounding boxes.
    Detect = 0,
    /// Bounding boxes with instance masks.
    Segment = 1,
    /// A per-pixel class map.
    Semantic = 2,
}

/// A resolved value whose source is recorded, for `ef_model_config_source`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfConfigField {
    /// Input normalization.
    Normalization = 0,
    /// Resize mode.
    Resize = 1,
    /// Letterbox placement.
    Placement = 2,
    /// Letterbox pad value and its space.
    Pad = 3,
    /// Resampling filter.
    Interpolation = 4,
    /// Model task.
    Task = 5,
    /// Semantic decode mode.
    DecodeMode = 6,
    /// Class labels.
    Labels = 7,
    /// Model input size.
    Size = 8,
}

const _: () = {
    assert!(EfNms::Off as u32 == NMS_OFF_CODE);
    assert!(EfNms::Auto as u32 == Nms::Auto.code());
    assert!(EfNms::ClassAware as u32 == Nms::ClassAware.code());
    assert!(EfNms::ClassAgnostic as u32 == Nms::ClassAgnostic.code());

    assert!(EfDecoderType::Ultralytics as u32 == DecoderType::Ultralytics.code());
    assert!(EfDecoderType::ModelPack as u32 == DecoderType::ModelPack.code());
    assert!(EfDecoderType::Semantic as u32 == DecoderType::Semantic.code());

    assert!(EfDecodeMode::Accurate as u32 == DecodeMode::Accurate.code());
    assert!(EfDecodeMode::Fast as u32 == DecodeMode::Fast.code());

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
    assert!(EfDimName::NumQueries as u32 == DimName::NumQueries.code());

    assert!(EfMatchMetric::Iou as u32 == MatchMetric::Iou.code());
    assert!(EfMatchMetric::Ios as u32 == MatchMetric::Ios.code());

    assert!(EfMergeMode::KeepBest as u32 == MergeMode::KeepBest.code());
    assert!(EfMergeMode::Union as u32 == MergeMode::Union.code());

    assert!(EfModelSource::Onnx as u32 == ModelSource::Onnx.code());
    assert!(EfModelSource::TfLite as u32 == ModelSource::TfLite.code());
    assert!(EfModelSource::Other as u32 == ModelSource::Other.code());
    assert!(EfModelSource::CoreMl as u32 == ModelSource::CoreMl.code());

    assert!(EfConfigSource::EmbeddedEdgefirst as u32 == ConfigSource::EmbeddedEdgefirst.code());
    assert!(EfConfigSource::SidecarEdgefirst as u32 == ConfigSource::SidecarEdgefirst.code());
    assert!(EfConfigSource::EmbeddedHf as u32 == ConfigSource::EmbeddedHf.code());
    assert!(EfConfigSource::SidecarHf as u32 == ConfigSource::SidecarHf.code());
    assert!(EfConfigSource::UltralyticsMetadata as u32 == ConfigSource::UltralyticsMetadata.code());
    assert!(EfConfigSource::Heuristic as u32 == ConfigSource::Heuristic.code());
    assert!(EfConfigSource::Default as u32 == ConfigSource::Default.code());
    assert!(EfConfigSource::Override as u32 == ConfigSource::Override.code());

    assert!(EfModelFamily::EdgeFirst as u32 == ModelFamily::EdgeFirst.code());
    assert!(EfModelFamily::Ultralytics as u32 == ModelFamily::Ultralytics.code());
    assert!(EfModelFamily::HuggingFace as u32 == ModelFamily::HuggingFace.code());
    assert!(EfModelFamily::Unknown as u32 == ModelFamily::Unknown.code());

    assert!(EfModelTask::Detect as u32 == ModelTask::Detect.code());
    assert!(EfModelTask::Segment as u32 == ModelTask::Segment.code());
    assert!(EfModelTask::Semantic as u32 == ModelTask::Semantic.code());

    assert!(EfConfigField::Normalization as u32 == ConfigField::Normalization.code());
    assert!(EfConfigField::Resize as u32 == ConfigField::Resize.code());
    assert!(EfConfigField::Placement as u32 == ConfigField::Placement.code());
    assert!(EfConfigField::Pad as u32 == ConfigField::Pad.code());
    assert!(EfConfigField::Interpolation as u32 == ConfigField::Interpolation.code());
    assert!(EfConfigField::Task as u32 == ConfigField::Task.code());
    assert!(EfConfigField::DecodeMode as u32 == ConfigField::DecodeMode.code());
    assert!(EfConfigField::Labels as u32 == ConfigField::Labels.code());
    assert!(EfConfigField::Size as u32 == ConfigField::Size.code());

    assert!(EfFit::Stretch as u32 == FitMode::Stretch.code());
    assert!(EfFit::Letterbox as u32 == FitMode::Letterbox.code());
    assert!(EfLetterboxPlacement::Center as u32 == LetterboxPlacement::Center.code());
    assert!(EfLetterboxPlacement::TopLeft as u32 == LetterboxPlacement::TopLeft.code());
    assert!(EfPadSpace::Pixel as u32 == PadSpace::Pixel.code());
    assert!(EfPadSpace::Normalized as u32 == PadSpace::Normalized.code());
    assert!(EfInterpolation::Bilinear as u32 == Interpolation::Bilinear.code());
    assert!(EfInterpolation::Nearest as u32 == Interpolation::Nearest.code());
    assert!(EfInterpolation::Bicubic as u32 == Interpolation::Bicubic.code());
    assert!(EfInterpolation::Area as u32 == Interpolation::Area.code());
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
            3,
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
        assert_eq!(DimName::all().len(), 12, "add the EfDimName enumerator");
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
        assert_eq!(
            ConfigSource::all().len(),
            8,
            "add the EfConfigSource enumerator"
        );
        assert_eq!(
            ModelFamily::all().len(),
            4,
            "add the EfModelFamily enumerator"
        );
        assert_eq!(ModelTask::all().len(), 3, "add the EfModelTask enumerator");
        assert_eq!(
            ConfigField::all().len(),
            9,
            "add the EfConfigField enumerator"
        );
        assert_eq!(FitMode::all().len(), 2, "add the EfFit enumerator");
        assert_eq!(
            LetterboxPlacement::all().len(),
            2,
            "add the EfLetterboxPlacement enumerator"
        );
        assert_eq!(PadSpace::all().len(), 2, "add the EfPadSpace enumerator");
        assert_eq!(
            Interpolation::all().len(),
            4,
            "add the EfInterpolation enumerator"
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
        expected.extend(
            [DecodeMode::Accurate, DecodeMode::Fast]
                .iter()
                .map(|&v| (enumerator("EF_DECODE_MODE", v), v.code())),
        );
        expected.extend(
            ConfigSource::all()
                .iter()
                .map(|&v| (enumerator("EF_CONFIG_SOURCE", v), v.code())),
        );
        expected.extend(
            ModelFamily::all()
                .iter()
                .map(|&v| (enumerator("EF_MODEL_FAMILY", v), v.code())),
        );
        expected.extend(
            ModelTask::all()
                .iter()
                .map(|&v| (enumerator("EF_MODEL_TASK", v), v.code())),
        );
        expected.extend(
            ConfigField::all()
                .iter()
                .map(|&v| (enumerator("EF_CONFIG_FIELD", v), v.code())),
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
        assert_eq!(DecoderType::from_code(3), None);
        assert_eq!(DecodeMode::from_code(2), None);
        assert_eq!(DimName::from_code(12), None);
        assert_eq!(DecoderVersion::from_code(4), None);
        assert_eq!(OutputType::from_code(8), None);
        assert_eq!(MatchMetric::from_code(2), None);
        assert_eq!(MergeMode::from_code(2), None);
        assert_eq!(ModelSource::from_code(4), None);
        assert_eq!(ConfigSource::from_code(8), None);
        assert_eq!(ModelFamily::from_code(4), None);
        assert_eq!(ModelTask::from_code(3), None);
        assert_eq!(ConfigField::from_code(9), None);
    }
}
