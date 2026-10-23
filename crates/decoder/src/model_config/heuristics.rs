// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic segmentation schemas built from a model's input and output
//! tensors, and the Hugging Face export recognised by their names.

use crate::configs::DimName;
use crate::infer::{ModelSignals, TensorInfo};
use crate::schema::{
    DecodeMode, DecoderKind, InputSpec, LogicalType, SchemaV2, Stride, Upsample, UpsampleMethod,
};

use super::ModelConfigError;

/// Score-map strides the Hugging Face name heuristic accepts.
const HEURISTIC_STRIDES: [usize; 4] = [1, 2, 4, 8];

/// A semantic schema and what was learned building it.
#[derive(Debug, Clone)]
pub(crate) struct SemanticExport {
    pub schema: SchemaV2,
    pub num_classes: usize,
    pub stride: usize,
    /// Model input `(width, height)`.
    pub size: (usize, usize),
}

/// `[1, 3, H, W]` → `(W, H)`.
fn nchw_image(t: &TensorInfo) -> Option<(usize, usize)> {
    match t.shape.as_slice() {
        [1, 3, h, w] => Some((*w, *h)),
        _ => None,
    }
}

/// The class count, layout and stride of a score map for an input of
/// `(w, h)`: NCHW `[1, C, h/s, w/s]` first, then NHWC `[1, h/s, w/s, C]`.
fn score_layout(shape: &[usize], (w, h): (usize, usize)) -> Option<(usize, bool, usize)> {
    let [1, a, b, c] = *shape else {
        return None;
    };
    let stride = |sh: usize, sw: usize| {
        (sh > 0 && sw > 0 && h % sh == 0 && w % sw == 0 && h / sh == w / sw).then(|| h / sh)
    };
    if let Some(s) = stride(b, c) {
        return Some((a, true, s));
    }
    stride(a, b).map(|s| (c, false, s))
}

fn single<'a>(tensors: &'a [TensorInfo], name: &str) -> Option<&'a TensorInfo> {
    tensors.iter().find(|t| t.name == name).or(match tensors {
        [t] => Some(t),
        _ => None,
    })
}

/// A semantic schema for a model whose input is an image and whose output
/// is a per-pixel score map. The input is the one named `pixel_values`, or
/// the only one; the output the one named `logits`, or the only rank-4 one.
pub(crate) fn semantic_schema(signals: &ModelSignals) -> Result<SemanticExport, ModelConfigError> {
    let unresolved = |what: &str| ModelConfigError::Unresolved {
        missing: format!("a semantic segmentation layout: {what}"),
        overrides: "an edgefirst.json describing the outputs".into(),
    };
    let input = single(&signals.inputs, "pixel_values")
        .ok_or_else(|| unresolved("no single image input"))?;
    let (w, h, input_dshape) = match input.shape.as_slice() {
        [1, 3, h, w] => (*w, *h, nchw_dims()),
        [1, h, w, 3] => (*w, *h, nhwc_dims()),
        s => {
            return Err(unresolved(&format!(
                "input shape {s:?} is not a 3-channel image"
            )))
        }
    };
    let rank4: Vec<&TensorInfo> = signals
        .outputs
        .iter()
        .filter(|t| t.shape.len() == 4)
        .collect();
    let output = signals
        .outputs
        .iter()
        .find(|t| t.name == "logits")
        .or(match rank4.as_slice() {
            [t] => Some(*t),
            _ => None,
        })
        .ok_or_else(|| unresolved("no single rank-4 score output"))?;
    let (classes, nchw, stride) = score_layout(&output.shape, (w, h)).ok_or_else(|| {
        unresolved(&format!(
            "output shape {:?} is not a score map of the {w}x{h} input",
            output.shape
        ))
    })?;
    let dshape = if nchw {
        vec![
            (DimName::Batch, 1),
            (DimName::NumClasses, classes),
            (DimName::Height, h / stride),
            (DimName::Width, w / stride),
        ]
    } else {
        vec![
            (DimName::Batch, 1),
            (DimName::Height, h / stride),
            (DimName::Width, w / stride),
            (DimName::NumClasses, classes),
        ]
    };
    let mut schema = SchemaV2::new();
    schema.input = Some(InputSpec {
        shape: input.shape.clone(),
        dshape: input_dshape
            .into_iter()
            .zip(input.shape.iter().copied())
            .collect(),
        cameraadaptor: Some("rgb".into()),
        preprocessing: None,
    });
    let mut out = super::resolve::empty_logical();
    out.name = Some(output.name.clone());
    out.type_ = Some(LogicalType::Segmentation);
    out.decoder = Some(DecoderKind::Semantic);
    out.shape = output.shape.clone();
    out.dshape = dshape;
    out.stride = Some(Stride::Square(stride as u32));
    out.dtype = Some(output.dtype);
    out.quantization = output.quantization.clone();
    out.semantic.upsample = Some(Upsample {
        method: UpsampleMethod::Bilinear,
        align_corners: false,
    });
    out.semantic.decode_mode = Some(DecodeMode::Accurate);
    schema.outputs.push(out);
    schema.validate()?;
    Ok(SemanticExport {
        schema,
        num_classes: classes,
        stride,
        size: (w, h),
    })
}

fn nchw_dims() -> [DimName; 4] {
    [
        DimName::Batch,
        DimName::NumFeatures,
        DimName::Height,
        DimName::Width,
    ]
}

fn nhwc_dims() -> [DimName; 4] {
    [
        DimName::Batch,
        DimName::Height,
        DimName::Width,
        DimName::NumFeatures,
    ]
}

/// The family a Hugging Face semantic export with this score-map stride
/// belongs to, for the defaults its image processor would use.
pub(crate) fn hf_family_name(stride: usize) -> &'static str {
    if stride == 4 {
        "SegFormer default"
    } else {
        "Hugging Face segmentation default"
    }
}

/// A Hugging Face semantic segmentation export recognised by its names alone:
/// input `pixel_values` `[1, 3, H, W]`, output `logits` `[1, C, H/s, W/s]`
/// with `s` in {1, 2, 4, 8}.
pub(crate) fn hf_semantic_by_names(signals: &ModelSignals) -> Option<SemanticExport> {
    let input = signals.inputs.iter().find(|t| t.name == "pixel_values")?;
    let output = signals.outputs.iter().find(|t| t.name == "logits")?;
    let size = nchw_image(input)?;
    let (_, nchw, stride) = score_layout(&output.shape, size)?;
    if !nchw || !HEURISTIC_STRIDES.contains(&stride) {
        return None;
    }
    semantic_schema(signals).ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::infer::ModelSource;
    use crate::schema::DType;

    pub(crate) fn tensor(name: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            shape: shape.to_vec(),
            dtype: DType::Float32,
            quantization: None,
        }
    }

    pub(crate) fn segformer_signals(classes: usize) -> ModelSignals {
        ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![tensor("pixel_values", &[1, 3, 512, 512])],
            outputs: vec![tensor("logits", &[1, classes, 128, 128])],
            metadata: Default::default(),
        }
    }

    #[test]
    fn segformer_names_are_recognised() {
        let e = hf_semantic_by_names(&segformer_signals(150)).unwrap();
        assert_eq!((e.num_classes, e.stride, e.size), (150, 4, (512, 512)));
        let out = &e.schema.outputs[0];
        assert_eq!(out.decoder, Some(DecoderKind::Semantic));
        crate::semantic::SemanticConfig::from_logical(out).unwrap();
        assert_eq!(hf_family_name(e.stride), "SegFormer default");
    }

    #[test]
    fn other_names_or_strides_are_not_guessed() {
        let mut s = segformer_signals(19);
        s.outputs[0].name = "output".into();
        assert!(hf_semantic_by_names(&s).is_none());
        let mut s = segformer_signals(19);
        s.outputs[0].shape = vec![1, 19, 32, 32];
        assert!(
            hf_semantic_by_names(&s).is_none(),
            "stride 16 is not a guess"
        );
        assert_eq!(semantic_schema(&s).unwrap().stride, 16);
    }

    #[test]
    fn nhwc_score_maps_are_laid_out() {
        let s = ModelSignals {
            source: ModelSource::TfLite,
            inputs: vec![tensor("input", &[1, 256, 320, 3])],
            outputs: vec![tensor("out", &[1, 64, 80, 21])],
            metadata: Default::default(),
        };
        let e = semantic_schema(&s).unwrap();
        assert_eq!((e.num_classes, e.stride, e.size), (21, 4, (320, 256)));
        assert_eq!(e.schema.outputs[0].dshape[3], (DimName::NumClasses, 21));
    }
}
