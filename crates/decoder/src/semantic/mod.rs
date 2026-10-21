// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic segmentation decoding.
//!
//! Turns a per-pixel score map (`[C, h, w]` in any axis order) or a class map
//! the model already computed into an `[H, W]` class map at any resolution,
//! written into a caller-supplied tensor. Each output pixel is mapped through
//! the letterbox geometry of the model input, so the padding is cropped and
//! the class map covers the original image.
//!
//! Two decode modes are available ([`DecodeMode`]):
//!
//! - **Accurate** upsamples the scores of every class, then takes the argmax.
//!   Pixels whose low-resolution neighbours agree on the winner skip the
//!   per-class interpolation, which is exact because bilinear interpolation
//!   is a convex combination.
//! - **Fast** takes the argmax at the score map's resolution, then
//!   upsamples the class map with nearest-neighbour sampling.
//!
//! Single-channel outputs are thresholded instead of argmaxed: foreground is
//! class 1 and background class 0.
//!
//! # Examples
//!
//! ```rust,no_run
//! use edgefirst_decoder::semantic::{decode_semantic, SemanticConfig, SemanticLayout, SemanticOptions};
//! use edgefirst_decoder::schema::DecodeMode;
//! use edgefirst_tensor::{DType, TensorDyn, TensorMemory};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = SemanticConfig::scores(&[1, 150, 128, 168], 1, 2, 3)?;
//! # let logits = TensorDyn::new(&[1, 150, 128, 168], DType::F32, Some(TensorMemory::Mem), None)?;
//! let mut class_map = TensorDyn::new(&[480, 640], DType::U8, Some(TensorMemory::Mem), None)?;
//! let options = SemanticOptions::new()
//!     .with_mode(DecodeMode::Fast)
//!     .with_letterbox([0.0, 0.0, 1.0, 0.75]);
//! decode_semantic(&logits, &config, &mut class_map, None, &options)?;
//! # Ok(())
//! # }
//! ```

mod kernels;

use edgefirst_tensor::{CpuAccess, DType, QuantMode, TensorDyn, TensorMapTrait, TensorTrait};
use half::f16;

use crate::configs::DimName;
use crate::schema::{
    Activation, DecodeMode, DecoderKind, LogicalOutput, LogicalType, Upsample, UpsampleMethod,
};
use crate::{DecoderError, DecoderResult};
use kernels::{ClassOut, ConfKind, ConfOut, Dequant, Grid, Score};

/// Largest class count a class map can index (`u16`).
const MAX_CLASSES: usize = 1 << 16;

/// What a semantic output holds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SemanticInput {
    /// Per-pixel class scores: logits or probabilities.
    Scores,
    /// A class index per pixel, with the argmax already applied in the graph.
    ClassMap,
}

/// Axis positions of a semantic output in its tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticLayout {
    /// Class axis; `None` for a single channel or a class map.
    pub classes: Option<usize>,
    /// Height axis.
    pub height: usize,
    /// Width axis.
    pub width: usize,
}

/// Static description of a semantic output, usually built from the model's
/// `edgefirst.json` by [`SemanticConfig::from_logical`].
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticConfig {
    input: SemanticInput,
    name: Option<String>,
    position: Option<(usize, usize)>,
    shape: Vec<usize>,
    layout: SemanticLayout,
    upsample: Upsample,
    mode: DecodeMode,
    threshold: Option<f32>,
    activation: Option<Activation>,
    activation_required: Option<Activation>,
    quantization: Option<edgefirst_tensor::Quantization>,
}

impl SemanticConfig {
    /// Score map of `shape`, with the class, height and width axes at the
    /// given positions. Every other axis must have size 1.
    ///
    /// Defaults: bilinear upsampling with `align_corners = false`,
    /// [`DecodeMode::Accurate`], no activation applied, not quantized.
    pub fn scores(
        shape: &[usize],
        class_axis: usize,
        height_axis: usize,
        width_axis: usize,
    ) -> DecoderResult<Self> {
        let layout = SemanticLayout {
            classes: Some(class_axis),
            height: height_axis,
            width: width_axis,
        };
        Self::new(SemanticInput::Scores, shape, layout)
    }

    /// Class map of `shape` with the height and width axes at the given
    /// positions. Every other axis must have size 1.
    pub fn class_map(
        shape: &[usize],
        height_axis: usize,
        width_axis: usize,
    ) -> DecoderResult<Self> {
        let layout = SemanticLayout {
            classes: None,
            height: height_axis,
            width: width_axis,
        };
        Self::new(SemanticInput::ClassMap, shape, layout)
    }

    fn new(input: SemanticInput, shape: &[usize], layout: SemanticLayout) -> DecoderResult<Self> {
        let axes = [layout.classes, Some(layout.height), Some(layout.width)];
        let named: Vec<usize> = axes.iter().flatten().copied().collect();
        if named.iter().any(|&a| a >= shape.len()) {
            return Err(DecoderError::InvalidShape(format!(
                "semantic axes {named:?} out of range for shape {shape:?}"
            )));
        }
        if named.len() != named.iter().collect::<std::collections::HashSet<_>>().len() {
            return Err(DecoderError::InvalidShape(format!(
                "semantic axes {named:?} must be distinct"
            )));
        }
        if let Some((axis, &n)) = shape
            .iter()
            .enumerate()
            .find(|&(i, &n)| n != 1 && !named.contains(&i))
        {
            return Err(DecoderError::InvalidShape(format!(
                "semantic output shape {shape:?} has unnamed axis {axis} of size {n}"
            )));
        }
        if shape.contains(&0) {
            return Err(DecoderError::InvalidShape(format!(
                "semantic output shape {shape:?} is empty"
            )));
        }
        let config = Self {
            input,
            name: None,
            position: None,
            shape: shape.to_vec(),
            layout,
            upsample: Upsample {
                method: UpsampleMethod::Bilinear,
                align_corners: false,
            },
            mode: DecodeMode::Accurate,
            threshold: None,
            activation: None,
            activation_required: None,
            quantization: None,
        };
        if config.classes() > MAX_CLASSES {
            return Err(DecoderError::NotSupported(format!(
                "{} classes exceed the decoder's limit of {MAX_CLASSES}",
                config.classes()
            )));
        }
        Ok(config)
    }

    /// Builds the configuration of a `segmentation` or `masks` logical output
    /// with `decoder: semantic`, or of a ModelPack `segmentation` output, which
    /// is NHWC and decoded in [`DecodeMode::Fast`] unless it says otherwise.
    pub fn from_logical(logical: &LogicalOutput) -> DecoderResult<Self> {
        let label = logical.name.as_deref().unwrap_or("<anonymous>");
        if logical.is_split() {
            return Err(DecoderError::NotSupported(format!(
                "semantic output `{label}` is split into physical tensors"
            )));
        }
        let (input, role) = match logical.type_ {
            Some(LogicalType::Segmentation) => (SemanticInput::Scores, "segmentation"),
            Some(LogicalType::Masks) => (SemanticInput::ClassMap, "masks"),
            other => {
                return Err(DecoderError::InvalidConfig(format!(
                    "semantic output `{label}` has type {other:?}; expected segmentation or masks"
                )))
            }
        };
        crate::configs::validate_dshape(role, &logical.shape, &logical.dshape)?;
        let modelpack = logical.decoder == Some(DecoderKind::ModelPack);
        let mut config = if modelpack && input == SemanticInput::Scores {
            let (c, h, w) = modelpack_axes(&logical.shape, &logical.dshape)?;
            Self::scores(&logical.shape, c, h, w)?.with_mode(DecodeMode::Fast)
        } else {
            Self::new(input, &logical.shape, layout_from_dshape(logical, input)?)?
        };
        config.name = logical.name.clone();
        if let Some(upsample) = logical.semantic.upsample {
            config = config.with_upsample(upsample)?;
        }
        if let Some(mode) = logical.semantic.decode_mode {
            config.mode = mode;
        }
        config.threshold = logical.semantic.threshold;
        config.activation = logical.activation_applied;
        config.activation_required = logical.activation_required;
        config.check_activation()?;
        if let Some(q) = &logical.quantization {
            config.quantization = Some(q.try_into().map_err(|e| {
                DecoderError::InvalidConfig(format!("semantic output `{label}` quantization: {e}"))
            })?);
        }
        Ok(config)
    }

    /// The segmentation output of a ModelPack model configured without a
    /// schema: NHWC unless `dshape` names the axes, decoded in
    /// [`DecodeMode::Fast`] by default.
    pub(crate) fn from_modelpack(seg: &crate::configs::Segmentation) -> DecoderResult<Self> {
        let (classes, height, width) = modelpack_axes(&seg.shape, &seg.dshape)?;
        let mut config =
            Self::scores(&seg.shape, classes, height, width)?.with_mode(DecodeMode::Fast);
        if let Some(q) = &seg.quantization {
            config.quantization = Some(edgefirst_tensor::Quantization::per_tensor(q.0, q.1));
        }
        Ok(config)
    }

    /// Sets the score-map upsampling. Bicubic is not supported.
    pub fn with_upsample(mut self, upsample: Upsample) -> DecoderResult<Self> {
        if !matches!(
            upsample.method,
            UpsampleMethod::Bilinear | UpsampleMethod::Nearest
        ) {
            return Err(DecoderError::NotSupported(format!(
                "semantic upsampling {:?} is not supported",
                upsample.method
            )));
        }
        self.upsample = upsample;
        Ok(self)
    }

    /// Sets the default decode mode.
    pub fn with_mode(mut self, mode: DecodeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the single-channel foreground threshold, in the space of the
    /// scores after `activation`.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Declares an activation the model already applied to the scores.
    ///
    /// # Errors
    ///
    /// See [`Self::with_activation_required`].
    pub fn with_activation(mut self, activation: Activation) -> DecoderResult<Self> {
        self.activation = Some(activation);
        self.check_activation()?;
        Ok(self)
    }

    /// Declares the activation the scores still need. It is applied after
    /// resampling, as reference pipelines do, so it leaves the class map
    /// unchanged and sets how [`decode_semantic`] computes confidence.
    ///
    /// # Errors
    ///
    /// [`DecoderError::InvalidConfig`] when an activation is both applied
    /// and required, or a single channel declares softmax;
    /// [`DecoderError::NotSupported`] for `tanh`.
    pub fn with_activation_required(mut self, activation: Activation) -> DecoderResult<Self> {
        self.activation_required = Some(activation);
        self.check_activation()?;
        Ok(self)
    }

    fn check_activation(&self) -> DecoderResult<()> {
        let single = self.input == SemanticInput::Scores && self.grid().classes == 1;
        match (self.activation, self.activation_required) {
            (Some(_), Some(_)) => Err(DecoderError::InvalidConfig(
                "a semantic output cannot both apply and require an activation".to_string(),
            )),
            (Some(Activation::Tanh), _) | (_, Some(Activation::Tanh)) => Err(
                DecoderError::NotSupported("tanh semantic scores are not supported".to_string()),
            ),
            (Some(Activation::Softmax), _) | (_, Some(Activation::Softmax)) if single => {
                Err(DecoderError::InvalidConfig(
                    "softmax over a single channel is constant; declare sigmoid".to_string(),
                ))
            }
            _ => Ok(()),
        }
    }

    /// Quantization of integer scores, used when the tensor carries none.
    pub fn with_quantization(mut self, quantization: edgefirst_tensor::Quantization) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Names the output so [`select_output`] prefers the tensor of that name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Records the output's index among the model's `count` outputs, which
    /// [`select_output`] uses when several outputs share its shape.
    pub fn with_position(mut self, index: usize, count: usize) -> Self {
        self.position = Some((index, count));
        self
    }

    /// What the output holds.
    pub fn input(&self) -> SemanticInput {
        self.input
    }

    /// Output name, when known.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Expected tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Axis positions in [`Self::shape`].
    pub fn layout(&self) -> SemanticLayout {
        self.layout
    }

    /// Number of classes the decoded class map can contain: the channel
    /// count, 2 for a single thresholded channel, 0 when unknown (a class
    /// map).
    pub fn classes(&self) -> usize {
        match (self.input, self.layout.classes) {
            (SemanticInput::ClassMap, _) => 0,
            (_, Some(axis)) if self.shape[axis] > 1 => self.shape[axis],
            _ => 2,
        }
    }

    /// Score-map height.
    pub fn height(&self) -> usize {
        self.shape[self.layout.height]
    }

    /// Score-map width.
    pub fn width(&self) -> usize {
        self.shape[self.layout.width]
    }

    /// Score-map upsampling.
    pub fn upsample(&self) -> Upsample {
        self.upsample
    }

    /// Default decode mode.
    pub fn mode(&self) -> DecodeMode {
        self.mode
    }

    /// Single-channel foreground threshold in the space of the scores:
    /// the configured value, else 0.5 for probabilities and 0.0 for logits.
    pub fn threshold(&self) -> f32 {
        self.threshold.unwrap_or(match self.activation {
            Some(Activation::Sigmoid) | Some(Activation::Softmax) => 0.5,
            _ => 0.0,
        })
    }

    /// Whether `tensor` has this output's shape, ignoring axes of size 1.
    pub fn matches(&self, tensor: &TensorDyn) -> bool {
        squeeze(tensor.shape()) == squeeze(&self.shape)
    }

    fn grid(&self) -> Grid {
        let mut strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        let (classes, cs) = match self.layout.classes {
            Some(axis) => (self.shape[axis], strides[axis]),
            None => (1, 0),
        };
        Grid {
            classes,
            height: self.height(),
            width: self.width(),
            cs,
            ys: strides[self.layout.height],
            xs: strides[self.layout.width],
        }
    }

    /// Element strides of `input`, following its row pitch when its rows
    /// are padded.
    fn grid_for(&self, input: &TensorDyn) -> DecoderResult<Grid> {
        let mut grid = self.grid();
        let Some(strides) = crate::decoder::tensor_bridge::padded_strides(input)? else {
            return Ok(grid);
        };
        // Shapes match up to unit axes, whose strides are never used.
        let stride = |axis: usize, packed: usize| {
            squeezed_axis(&self.shape, axis)
                .and_then(|k| {
                    input
                        .shape()
                        .iter()
                        .enumerate()
                        .filter(|&(_, &n)| n > 1)
                        .nth(k)
                })
                .map_or(packed, |(i, _)| strides[i])
        };
        if let Some(axis) = self.layout.classes {
            grid.cs = stride(axis, grid.cs);
        }
        grid.ys = stride(self.layout.height, grid.ys);
        grid.xs = stride(self.layout.width, grid.xs);
        Ok(grid)
    }
}

/// Fails unless every element `g` addresses lies within `len` elements.
fn check_extent(len: usize, g: Grid, what: &str) -> DecoderResult<()> {
    let extent = (g.classes - 1) * g.cs + (g.height - 1) * g.ys + (g.width - 1) * g.xs + 1;
    if extent > len {
        return Err(DecoderError::InvalidShape(format!(
            "{what} holds {len} elements but its layout addresses {extent}"
        )));
    }
    Ok(())
}

fn squeeze(shape: &[usize]) -> Vec<usize> {
    shape.iter().copied().filter(|&n| n != 1).collect()
}

/// Class axis of a rank-4 ModelPack segmentation output: the axis named
/// `num_classes`, else the NHWC channel axis.
pub(crate) fn modelpack_class_axis(dshape: &[(DimName, usize)]) -> usize {
    dshape
        .iter()
        .position(|(n, _)| *n == DimName::NumClasses)
        .unwrap_or(3)
}

/// `(classes, height, width)` axes of a ModelPack segmentation output: NHWC,
/// with any axis `dshape` names taking its named position.
fn modelpack_axes(
    shape: &[usize],
    dshape: &[(DimName, usize)],
) -> DecoderResult<(usize, usize, usize)> {
    if shape.len() != 4 {
        return Err(DecoderError::InvalidShape(format!(
            "ModelPack segmentation shape {shape:?}; expected rank 4"
        )));
    }
    let find =
        |name: DimName, nhwc: usize| dshape.iter().position(|(n, _)| *n == name).unwrap_or(nhwc);
    Ok((
        modelpack_class_axis(dshape),
        find(DimName::Height, 1),
        find(DimName::Width, 2),
    ))
}

fn layout_from_dshape(
    logical: &LogicalOutput,
    input: SemanticInput,
) -> DecoderResult<SemanticLayout> {
    let label = logical.name.as_deref().unwrap_or("<anonymous>");
    let find = |name: DimName| logical.dshape.iter().position(|(n, _)| *n == name);
    if !logical.dshape.is_empty() {
        let (Some(height), Some(width)) = (find(DimName::Height), find(DimName::Width)) else {
            return Err(DecoderError::InvalidConfig(format!(
                "semantic output `{label}` dshape must name height and width"
            )));
        };
        let classes = match input {
            SemanticInput::Scores => find(DimName::NumClasses),
            SemanticInput::ClassMap => None,
        };
        return Ok(SemanticLayout {
            classes,
            height,
            width,
        });
    }
    // Without dshape only an unambiguous single-plane layout is accepted.
    let spatial: Vec<usize> = (0..logical.shape.len())
        .filter(|&i| logical.shape[i] != 1)
        .collect();
    match spatial.as_slice() {
        [h, w] => Ok(SemanticLayout {
            classes: None,
            height: *h,
            width: *w,
        }),
        _ => Err(DecoderError::InvalidConfig(format!(
            "semantic output `{label}` needs a dshape naming num_classes, height and width"
        ))),
    }
}

/// Runtime options of one semantic decode.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct SemanticOptions {
    /// Decode mode; `None` uses the model's default.
    pub mode: Option<DecodeMode>,
    /// Image content within the model input as normalized
    /// `[xmin, ymin, xmax, ymax]`, as produced by a letterbox resize. The
    /// region is cropped and stretched over the whole class map. `None`
    /// means the image fills the input.
    pub letterbox: Option<[f32; 4]>,
}

impl SemanticOptions {
    /// Default options: the model's decode mode, no letterbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the model's decode mode.
    pub fn with_mode(mut self, mode: DecodeMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the letterbox content region.
    pub fn with_letterbox(mut self, letterbox: [f32; 4]) -> Self {
        self.letterbox = Some(letterbox);
        self
    }
}

/// Picks the tensor of a semantic output from a model's outputs: the one
/// named like the output, else the only one with its shape, else the one at
/// the output's position when every model output was passed.
pub fn select_output<'a>(
    outputs: &[&'a TensorDyn],
    config: &SemanticConfig,
) -> DecoderResult<&'a TensorDyn> {
    let mut found = outputs.iter().copied().filter(|t| config.matches(t));
    let (first, second) = (found.next(), found.next());
    if let Some(name) = config.name() {
        if let Some(t) = outputs
            .iter()
            .copied()
            .find(|t| t.name() == name && config.matches(t))
        {
            return Ok(t);
        }
    }
    match (first, second, config.position) {
        (Some(t), None, _) => Ok(t),
        (Some(_), Some(_), Some((index, count))) if outputs.len() == count => outputs
            .get(index)
            .copied()
            .filter(|t| config.matches(t))
            .ok_or_else(|| {
                DecoderError::InvalidConfig(format!(
                    "the output at position {index} does not have the semantic output's shape {:?}",
                    config.shape()
                ))
            }),
        (Some(_), Some(_), _) => Err(DecoderError::InvalidConfig(format!(
            "several outputs have the semantic output's shape {:?}; pass the outputs in model order or name the tensors",
            config.shape()
        ))),
        (None, _, _) => Err(DecoderError::InvalidShape(format!(
            "no output has the semantic output's shape {:?}; got {:?}",
            config.shape(),
            outputs.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>()
        ))),
    }
}

/// Decodes a semantic output into `dst`, an `[H, W]` class map of dtype
/// `u8` (up to 256 classes) or `u16` (up to 65 536).
///
/// `confidence`, when given, receives the winning class's probability as an
/// `[H, W]` map of `f32` in `[0, 1]` or `u8` scaled to 255. It is computed at
/// the score map's resolution and upsampled, so it is approximate and meant
/// for overlays. Class-map inputs have no confidence.
///
/// # Errors
///
/// - [`DecoderError::InvalidShape`] if `input` does not have the configured
///   shape, or `dst` and `confidence` are not matching `[H, W]` tensors.
/// - [`DecoderError::NotSupported`] for unsupported dtypes, or when the
///   class count exceeds what the dtype of `dst` holds.
/// - [`DecoderError::InvalidConfig`] for an invalid letterbox or
///   quantization, or a class-map value `dst` cannot hold.
/// - [`DecoderError::Tensor`] carrying
///   [`CpuAccessDenied`](edgefirst_tensor::Error::CpuAccessDenied) when
///   `input` is not declared CPU-readable, or `dst` or `confidence` not
///   CPU-writable.
pub fn decode_semantic(
    input: &TensorDyn,
    config: &SemanticConfig,
    dst: &mut TensorDyn,
    confidence: Option<&mut TensorDyn>,
    options: &SemanticOptions,
) -> DecoderResult<()> {
    if !config.matches(input) {
        return Err(DecoderError::InvalidShape(format!(
            "semantic input shape {:?} does not match the configured {:?}",
            input.shape(),
            config.shape()
        )));
    }
    let (out_h, out_w) = class_map_dims(dst, "class map")?;
    if let Some(conf) = confidence.as_deref() {
        if class_map_dims(conf, "confidence")? != (out_h, out_w) {
            return Err(DecoderError::InvalidShape(format!(
                "confidence shape {:?} differs from the class map's {:?}",
                conf.shape(),
                dst.shape()
            )));
        }
        if config.input == SemanticInput::ClassMap {
            return Err(DecoderError::NotSupported(
                "a class-map output has no confidence".to_string(),
            ));
        }
    }
    let capacity = match dst.dtype() {
        DType::U8 => u8::CAPACITY,
        DType::U16 => u16::CAPACITY,
        other => {
            return Err(DecoderError::NotSupported(format!(
                "class map dtype {other:?}; use u8 or u16"
            )))
        }
    };
    if config.classes() > capacity {
        return Err(DecoderError::NotSupported(format!(
            "{} classes do not fit a {:?} class map",
            config.classes(),
            dst.dtype()
        )));
    }

    input.cpu_access().require(CpuAccess::Read)?;
    dst.cpu_access().require(CpuAccess::Write)?;
    if let Some(conf) = confidence.as_deref() {
        conf.cpu_access().require(CpuAccess::Write)?;
    }

    let geometry = Geometry::new(config, options, out_h, out_w)?;
    match config.input {
        SemanticInput::ClassMap => decode_class_map(input, config, &geometry, dst),
        SemanticInput::Scores => decode_scores(input, config, &geometry, dst, confidence),
    }
}

/// Validates an `[H, W]` output tensor and returns its dimensions.
fn class_map_dims(t: &TensorDyn, what: &str) -> DecoderResult<(usize, usize)> {
    match *t.shape() {
        [h, w] if h > 0 && w > 0 => {
            if t.row_stride().is_some_and(|s| s != w * t.dtype().size()) {
                return Err(DecoderError::NotSupported(format!(
                    "{what} tensor has a padded row stride"
                )));
            }
            Ok((h, w))
        }
        _ => Err(DecoderError::InvalidShape(format!(
            "{what} tensor shape {:?}; expected [H, W]",
            t.shape()
        ))),
    }
}

/// Output-to-source sampling resolved from the letterbox and mode.
struct Geometry {
    mode: DecodeMode,
    method: UpsampleMethod,
    align_corners: bool,
    roi: [f32; 4],
    out_h: usize,
    out_w: usize,
    src_h: usize,
    src_w: usize,
}

impl Geometry {
    fn new(
        config: &SemanticConfig,
        options: &SemanticOptions,
        out_h: usize,
        out_w: usize,
    ) -> DecoderResult<Self> {
        let [x0, y0, x1, y1] = options.letterbox.unwrap_or([0.0, 0.0, 1.0, 1.0]);
        let valid =
            |a: f32, b: f32| a.is_finite() && b.is_finite() && 0.0 <= a && a < b && b <= 1.0;
        if !valid(x0, x1) || !valid(y0, y1) {
            return Err(DecoderError::InvalidConfig(format!(
                "letterbox {:?} must satisfy 0 <= min < max <= 1",
                [x0, y0, x1, y1]
            )));
        }
        let (src_h, src_w) = (config.height(), config.width());
        Ok(Self {
            mode: options.mode.unwrap_or(config.mode),
            method: config.upsample.method,
            align_corners: config.upsample.align_corners,
            roi: [
                x0 * src_w as f32,
                y0 * src_h as f32,
                (x1 - x0) * src_w as f32,
                (y1 - y0) * src_h as f32,
            ],
            out_h,
            out_w,
            src_h,
            src_w,
        })
    }

    /// Whether scores are resampled before the argmax or threshold.
    fn bilinear_scores(&self) -> bool {
        self.mode == DecodeMode::Accurate && self.method == UpsampleMethod::Bilinear
    }

    fn nearest(&self) -> (Vec<u32>, Vec<u32>) {
        let [x0, y0, w, h] = self.roi;
        (
            kernels::nearest_taps(self.out_h, self.src_h, y0, h),
            kernels::nearest_taps(self.out_w, self.src_w, x0, w),
        )
    }

    fn bilinear(&self) -> (Vec<kernels::Tap>, Vec<kernels::Tap>) {
        let [x0, y0, w, h] = self.roi;
        (
            kernels::bilinear_taps(self.out_h, self.src_h, y0, h, self.align_corners),
            kernels::bilinear_taps(self.out_w, self.src_w, x0, w, self.align_corners),
        )
    }
}

/// Real-value map of a score tensor: per-tensor affine, or per-channel
/// scales that require dequantizing first.
enum Quant {
    PerTensor(Dequant),
    PerChannel {
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    },
}

/// Position of `axis` among the axes of `shape` longer than 1, so axes of
/// two shapes that differ only in unit axes compare equal.
fn squeezed_axis(shape: &[usize], axis: usize) -> Option<usize> {
    (shape.get(axis)? > &1).then(|| shape[..axis].iter().filter(|&&n| n > 1).count())
}

/// The real-value map of `input`. `needs_real` marks decodes that compare
/// against a real threshold or compute a confidence; integer scores without
/// quantization are then an error rather than taken at face value.
fn resolve_quant(
    input: &TensorDyn,
    config: &SemanticConfig,
    needs_real: bool,
) -> DecoderResult<Quant> {
    if matches!(input.dtype(), DType::F16 | DType::F32) {
        return Ok(Quant::PerTensor(Dequant::IDENTITY));
    }
    let (q, shape) = match (input.quantization(), config.quantization.as_ref()) {
        (Some(q), _) => (q, input.shape()),
        (None, Some(q)) => (q, config.shape()),
        (None, None) if needs_real => {
            return Err(DecoderError::QuantMissing {
                dtype: input.dtype(),
                role: "semantic scores",
                level: 0,
            })
        }
        (None, None) => return Ok(Quant::PerTensor(Dequant::IDENTITY)),
    };
    let check = |s: f32| {
        if s.is_finite() && s > 0.0 {
            Ok(s)
        } else {
            Err(DecoderError::InvalidConfig(format!(
                "semantic quantization scale {s} must be positive"
            )))
        }
    };
    let per_channel = |scales: &[f32], zps: Vec<i32>, axis: usize| {
        let class_axis = config
            .layout
            .classes
            .and_then(|a| squeezed_axis(config.shape(), a));
        if class_axis.is_none() || squeezed_axis(shape, axis) != class_axis {
            return Err(DecoderError::NotSupported(format!(
                "per-channel quantization on axis {axis}, which is not the class axis"
            )));
        }
        let classes = config.grid().classes;
        if scales.len() != classes || zps.len() != classes {
            return Err(DecoderError::InvalidConfig(format!(
                "per-channel quantization has {} scales and {} zero points for {classes} classes",
                scales.len(),
                zps.len()
            )));
        }
        scales.iter().try_for_each(|&s| check(s).map(|_| ()))?;
        Ok(Quant::PerChannel {
            scales: scales.to_vec(),
            zero_points: zps,
        })
    };
    match q.mode() {
        QuantMode::PerTensorSymmetric { scale } => Ok(Quant::PerTensor(Dequant {
            scale: check(scale)?,
            zero_point: 0.0,
        })),
        QuantMode::PerTensor { scale, zero_point } => Ok(Quant::PerTensor(Dequant {
            scale: check(scale)?,
            zero_point: zero_point as f32,
        })),
        QuantMode::PerChannelSymmetric { scales, axis } => {
            per_channel(scales, vec![0; scales.len()], axis)
        }
        QuantMode::PerChannel {
            scales,
            zero_points,
            axis,
        } => per_channel(scales, zero_points.to_vec(), axis),
    }
}

fn map_err(what: &str) -> impl FnOnce(edgefirst_tensor::Error) -> DecoderError + '_ {
    move |e| {
        DecoderError::from_map(e, |e| {
            DecoderError::InvalidConfig(format!("{what} map failed: {e}"))
        })
    }
}

fn decode_scores(
    input: &TensorDyn,
    config: &SemanticConfig,
    geometry: &Geometry,
    dst: &mut TensorDyn,
    confidence: Option<&mut TensorDyn>,
) -> DecoderResult<()> {
    let g = config.grid_for(input)?;
    let needs_real = g.classes == 1 || confidence.is_some();
    macro_rules! typed {
        ($t:ty) => {{
            let map = input
                .as_typed::<$t>()
                .ok_or_else(|| DecoderError::Internal("semantic input dtype".to_string()))?
                .map_read()
                .map_err(map_err("semantic input"))?;
            check_extent(map.as_slice().len(), g, "semantic input")?;
            match resolve_quant(input, config, needs_real)? {
                Quant::PerTensor(dq) => {
                    scores_into(map.as_slice(), g, config, dq, geometry, dst, confidence)
                }
                Quant::PerChannel {
                    scales,
                    zero_points,
                } => {
                    let real = dequantize_per_channel(map.as_slice(), g, &scales, &zero_points);
                    scores_into(
                        &real,
                        g,
                        config,
                        Dequant::IDENTITY,
                        geometry,
                        dst,
                        confidence,
                    )
                }
            }
        }};
    }
    match input.dtype() {
        DType::F32 => typed!(f32),
        DType::F16 => typed!(f16),
        DType::U8 => typed!(u8),
        DType::I8 => typed!(i8),
        DType::U16 => typed!(u16),
        DType::I16 => typed!(i16),
        other => Err(DecoderError::NotSupported(format!(
            "semantic score dtype {other:?}"
        ))),
    }
}

fn dequantize_per_channel<T: Score>(
    data: &[T],
    g: Grid,
    scales: &[f32],
    zero_points: &[i32],
) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for (c, (&s, &z)) in scales.iter().zip(zero_points).enumerate() {
        let z = z as f32;
        for y in 0..g.height {
            for x in 0..g.width {
                let i = c * g.cs + y * g.ys + x * g.xs;
                out[i] = (data[i].to_f32() - z) * s;
            }
        }
    }
    out
}

fn scores_into<T: Score>(
    data: &[T],
    g: Grid,
    config: &SemanticConfig,
    dq: Dequant,
    geometry: &Geometry,
    dst: &mut TensorDyn,
    confidence: Option<&mut TensorDyn>,
) -> DecoderResult<()> {
    let single = g.classes == 1;
    let cut = dq.stored(config.threshold());
    // The low-resolution winners feed the fast path, the accurate early-out
    // and the confidence map; single-channel accurate decodes need them only
    // for confidence.
    let needs_winners = !single || !geometry.bilinear_scores() || confidence.is_some();
    let winners = needs_winners.then(|| {
        if single {
            kernels::threshold_map(data, g, cut)
        } else {
            kernels::argmax_map(data, g)
        }
    });
    match dst.dtype() {
        DType::U8 => write_classes::<T, u8>(data, g, cut, winners.as_deref(), geometry, dst)?,
        _ => write_classes::<T, u16>(data, g, cut, winners.as_deref(), geometry, dst)?,
    }
    let (Some(conf), Some(winners)) = (confidence, winners) else {
        return Ok(());
    };
    let kind = match (single, config.activation, config.activation_required) {
        (true, Some(_), _) => ConfKind::BinaryProbability,
        (true, None, _) => ConfKind::BinaryLogit,
        (false, Some(_), _) => ConfKind::Probability,
        (false, None, Some(Activation::Sigmoid)) => ConfKind::Sigmoid,
        (false, None, _) => ConfKind::Softmax,
    };
    let low = kernels::confidence_map(data, g, &winners, dq, kind);
    match conf.dtype() {
        DType::F32 => write_confidence::<f32>(&low, g.width, geometry, conf),
        DType::U8 => write_confidence::<u8>(&low, g.width, geometry, conf),
        other => Err(DecoderError::NotSupported(format!(
            "confidence dtype {other:?}; use f32 or u8"
        ))),
    }
}

fn write_classes<T: Score, O: ClassOut + edgefirst_tensor::Element>(
    data: &[T],
    g: Grid,
    cut: f32,
    winners: Option<&[u16]>,
    geometry: &Geometry,
    dst: &mut TensorDyn,
) -> DecoderResult<()> {
    let mut map = dst
        .as_typed_mut::<O>()
        .ok_or_else(|| DecoderError::Internal("class map dtype".to_string()))?
        .map_for_write()
        .map_err(map_err("class map"))?;
    let out = map.as_mut_slice();
    match (geometry.bilinear_scores(), winners) {
        (true, Some(winners)) if g.classes > 1 => {
            let (ys, xs) = geometry.bilinear();
            kernels::upsample_argmax(data, g, winners, &ys, &xs, out);
        }
        (true, _) => {
            let (ys, xs) = geometry.bilinear();
            kernels::upsample_threshold(data, g, cut, &ys, &xs, out);
        }
        (false, Some(winners)) => {
            let (ys, xs) = geometry.nearest();
            kernels::upsample_nearest(winners, g.width, &ys, &xs, out, O::from_class);
        }
        (false, None) => {
            return Err(DecoderError::Internal(
                "nearest decode without a class map".to_string(),
            ))
        }
    }
    Ok(())
}

fn write_confidence<C: ConfOut + edgefirst_tensor::Element>(
    low: &[f32],
    width: usize,
    geometry: &Geometry,
    conf: &mut TensorDyn,
) -> DecoderResult<()> {
    let mut map = conf
        .as_typed_mut::<C>()
        .ok_or_else(|| DecoderError::Internal("confidence dtype".to_string()))?
        .map_for_write()
        .map_err(map_err("confidence"))?;
    let out = map.as_mut_slice();
    if geometry.bilinear_scores() {
        let (ys, xs) = geometry.bilinear();
        kernels::upsample_conf_bilinear(low, width, &ys, &xs, out);
    } else {
        let (ys, xs) = geometry.nearest();
        kernels::upsample_nearest(low, width, &ys, &xs, out, C::from_conf);
    }
    Ok(())
}

/// Resizes a class map the model already computed; always nearest.
fn decode_class_map(
    input: &TensorDyn,
    config: &SemanticConfig,
    geometry: &Geometry,
    dst: &mut TensorDyn,
) -> DecoderResult<()> {
    let g = config.grid_for(input)?;
    macro_rules! typed {
        ($t:ty) => {{
            let map = input
                .as_typed::<$t>()
                .ok_or_else(|| DecoderError::Internal("class map input dtype".to_string()))?
                .map_read()
                .map_err(map_err("class map input"))?;
            check_extent(map.as_slice().len(), g, "class map input")?;
            class_map_into(map.as_slice(), g, geometry, dst)
        }};
    }
    match input.dtype() {
        DType::U8 => typed!(u8),
        DType::U16 => typed!(u16),
        DType::I32 => typed!(i32),
        DType::U32 => typed!(u32),
        DType::I64 => typed!(i64),
        other => Err(DecoderError::NotSupported(format!(
            "class map input dtype {other:?}"
        ))),
    }
}

fn class_map_into<T>(
    data: &[T],
    g: Grid,
    geometry: &Geometry,
    dst: &mut TensorDyn,
) -> DecoderResult<()>
where
    T: Copy + TryInto<u16> + std::fmt::Display,
{
    let limit = match dst.dtype() {
        DType::U8 => u8::CAPACITY,
        _ => u16::CAPACITY,
    };
    let classes = (0..g.height)
        .flat_map(|y| (0..g.width).map(move |x| data[y * g.ys + x * g.xs]))
        .map(|v| {
            v.try_into()
                .ok()
                .filter(|&c: &u16| (c as usize) < limit)
                .ok_or_else(|| {
                    DecoderError::InvalidConfig(format!(
                        "class index {v} does not fit a {:?} class map",
                        dst.dtype()
                    ))
                })
        })
        .collect::<DecoderResult<Vec<u16>>>()?;
    let (ys, xs) = geometry.nearest();
    macro_rules! write {
        ($o:ty) => {{
            let mut map = dst
                .as_typed_mut::<$o>()
                .ok_or_else(|| DecoderError::Internal("class map dtype".to_string()))?
                .map_for_write()
                .map_err(map_err("class map"))?;
            kernels::upsample_nearest(
                &classes,
                geometry.src_w,
                &ys,
                &xs,
                map.as_mut_slice(),
                <$o>::from_class,
            );
        }};
    }
    match dst.dtype() {
        DType::U8 => write!(u8),
        _ => write!(u16),
    }
    Ok(())
}
