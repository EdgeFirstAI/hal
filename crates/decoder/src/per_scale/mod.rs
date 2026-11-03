// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-scale quantized decoder — see
//! `.claude/plans/2026-04-28-per-scale-decoder-optimized-design.md`.

pub mod helper;
pub(crate) mod kernels;
pub(crate) mod outputs;
pub(crate) mod pipeline;
pub(crate) mod plan;
pub(crate) mod prefilter;

pub use helper::apply_schema_quant;

/// Output element type chosen by the user at `DecoderBuilder::with_decode_dtype()`.
///
/// The whole post-merge pipeline (boxes, scores, mask coefs, protos) is
/// emitted in this dtype. `F16` saves ~2× memory bandwidth at the cost of
/// 10-bit mantissa precision — empirically safe for YOLO-family models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeDtype {
    /// 32-bit float output (default).
    #[default]
    F32,
    /// 16-bit float output. Halves the memory bandwidth of the decode
    /// pipeline; on aarch64 it also keeps the NEON FP16 kernels in their
    /// native width instead of widening.
    F16,
}

/// Activation function applied after dequantization on a logical output.
///
/// Sourced from the schema's `activation_required` field. Currently only
/// `Sigmoid` is wired through the per-scale pipeline; future activations
/// (e.g. `Softmax` on objectness) extend this enum without ripple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // consumed by per-scale pipeline tasks
pub(crate) enum Activation {
    #[default]
    None,
    Sigmoid,
}

impl Activation {
    /// Translate a schema activation to a per_scale Activation.
    /// Returns Activation::None when the schema declares no activation.
    #[allow(dead_code)] // consumed by per-scale pipeline tasks
    pub(crate) fn from_schema(s: Option<crate::schema::Activation>) -> Self {
        match s {
            Some(crate::schema::Activation::Sigmoid) => Self::Sigmoid,
            _ => Self::None,
        }
    }
}

pub(crate) use outputs::{DecodedOutputBuffers, DecodedOutputsRef};
pub(crate) use plan::PerScalePlan;

/// Per-scale decoder for schema-v2 per-scale models. Built once at
/// `DecoderBuilder::build()` time; consumed per-frame via `run()`.
#[derive(Debug)]
#[allow(dead_code)] // Wired by Task 24's Decoder integration.
pub(crate) struct PerScaleDecoder {
    pub(crate) plan: PerScalePlan,
    pub(crate) buffers: DecodedOutputBuffers,
    /// f32 storage the bridge widens f16 outputs into, reused frame to frame.
    pub(crate) widen: outputs::WidenScratch,
}

impl PerScaleDecoder {
    /// Build a decoder from a plan, allocating output buffers.
    #[allow(dead_code)] // Wired by Task 23's builder.
    pub(crate) fn new(plan: PerScalePlan) -> Self {
        let buffers = DecodedOutputBuffers::new(
            plan.out_dtype,
            plan.total_anchors,
            plan.num_classes,
            plan.num_mask_coefs,
            plan.proto_nhwc_shape.as_deref(),
        );
        let widen = outputs::WidenScratch {
            features: plan.cpu_features,
            ..Default::default()
        };
        Self {
            plan,
            buffers,
            widen,
        }
    }

    /// Decode one frame's worth of inputs.
    ///
    /// With a `score_threshold`, only anchors that can reach it are
    /// decoded and the outputs hold just those, in anchor order; `None`
    /// decodes every anchor.
    #[allow(dead_code)] // Wired by Task 24.
    pub(crate) fn run<'a>(
        &'a mut self,
        inputs: &[&edgefirst_tensor::TensorDyn],
        score_threshold: Option<f32>,
    ) -> crate::DecoderResult<DecodedOutputsRef<'a>> {
        pipeline::run(&self.plan, &mut self.buffers, inputs, score_threshold)
    }

    /// [`Self::run`], also lending the scratch the bridge widens f16 outputs into.
    pub(crate) fn run_with_widen<'a>(
        &'a mut self,
        inputs: &[&edgefirst_tensor::TensorDyn],
        score_threshold: Option<f32>,
    ) -> crate::DecoderResult<(DecodedOutputsRef<'a>, &'a mut outputs::WidenScratch)> {
        let decoded = pipeline::run(&self.plan, &mut self.buffers, inputs, score_threshold)?;
        Ok((decoded, &mut self.widen))
    }
}

/// Owned f32 snapshot of pre-NMS per-scale outputs.
///
/// Returned by [`crate::Decoder::_testing_run_per_scale_pre_nms`] and
/// used by integration tests to compare against fixture intermediates
/// without the noise of NMS ordering.
#[doc(hidden)]
pub struct PreNmsCapture {
    pub boxes_xywh: ndarray::Array2<f32>,
    pub scores: ndarray::Array2<f32>,
    pub mask_coefs: Option<ndarray::Array2<f32>>,
    pub protos: Option<ndarray::Array4<f32>>,
}

/// Kernel-level hooks for the decoder benchmark. Not a stable API.
#[doc(hidden)]
pub mod bench {
    use super::kernels::level_box;
    use super::plan::{Layout, LevelPlan};
    use crate::Quantization;
    use edgefirst_kernels::prim::dfl_lut::{decode_dfl_lut, DflExpTable};
    use edgefirst_kernels::prim::grids::make_anchor_grid;

    /// A DFL box kernel for 8-bit bins.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DflKernel {
        /// Dequantize, then f32 libm softmax per side (portable).
        Softmax,
        /// Exp table, portable.
        Lut,
        /// Dequantize, then fused NEON softmax with an FP16 exp.
        NeonFusedFp16,
        /// Exp table with the four sides in NEON lanes.
        LutNeon,
    }

    impl DflKernel {
        /// Every kernel this CPU can run.
        pub fn available() -> Vec<Self> {
            let v = vec![Self::Softmax, Self::Lut];
            #[cfg(target_arch = "aarch64")]
            let v = {
                let mut v = v;
                v.push(Self::LutNeon);
                if std::arch::is_aarch64_feature_detected!("fp16") {
                    v.push(Self::NeonFusedFp16);
                }
                v
            };
            v
        }
    }

    /// One DFL level (`reg_max` 16) and its quantization.
    pub struct DflLevel {
        plan: LevelPlan,
        q: Quantization,
        table: DflExpTable,
    }

    impl DflLevel {
        pub fn new(w: usize, h: usize, stride: f32, scale: f32, zero_point: i32) -> Self {
            let (grid_x, grid_y) = make_anchor_grid(h, w);
            Self {
                plan: LevelPlan {
                    stride,
                    h,
                    w,
                    reg_max: 16,
                    anchor_offset: 0,
                    grid_x,
                    grid_y,
                    box_shape: vec![1, h, w, 64].into_boxed_slice(),
                    score_shape: vec![1, h, w, 1].into_boxed_slice(),
                    mc_shape: None,
                    layout: Layout::Nhwc,
                },
                q: Quantization::new(scale, zero_point),
                table: DflExpTable::new(scale),
            }
        }

        pub fn anchors(&self) -> usize {
            self.plan.h * self.plan.w
        }

        /// Decode `input` (`anchors() * 64` bins) into `dst` (`anchors() * 4`); `InvalidShape`,
        /// before writing anything, for other lengths.
        pub fn decode(
            &self,
            kernel: DflKernel,
            input: &[i8],
            dst: &mut [f32],
        ) -> DecoderResult<()> {
            let (p, q) = (&self.plan, self.q);
            let n = self.anchors();
            if Some(input.len()) != n.checked_mul(64) || Some(dst.len()) != n.checked_mul(4) {
                return Err(DecoderError::InvalidShape(format!(
                    "DFL level of {n} anchors: {} input and {} output values do not match",
                    input.len(),
                    dst.len()
                )));
            }
            match kernel {
                DflKernel::Softmax => level_box::decode_box_level_dfl_i8_to_f32(input, q, p, dst),
                DflKernel::Lut => decode_dfl_lut(input, &self.table, &p.dfl_geom(), dst),
                #[cfg(target_arch = "aarch64")]
                DflKernel::NeonFusedFp16 => {
                    if !std::arch::is_aarch64_feature_detected!("fp16") {
                        return Err(unavailable(kernel));
                    }
                    // SAFETY: FEAT_FP16 was just detected; the lengths match the level.
                    unsafe { level_box::dfl_neon(input, q, p, level_box::Exp::Fp16, dst) }
                }
                #[cfg(target_arch = "aarch64")]
                DflKernel::LutNeon => edgefirst_kernels::prim::dfl_lut::decode_dfl_lut_neon(
                    input,
                    &self.table,
                    &p.dfl_geom(),
                    dst,
                ),
                #[cfg(not(target_arch = "aarch64"))]
                _ => return Err(unavailable(kernel)),
            }
            Ok(())
        }
    }

    /// The error for a [`DflKernel`] this CPU cannot run.
    fn unavailable(kernel: DflKernel) -> DecoderError {
        DecoderError::NotSupported(format!("{kernel:?} is not available on this CPU"))
    }

    use super::kernels::dispatch::{
        DstSliceMut, InputView, MaskCoefDispatch, ProtoDispatch, ScoreLevelDispatch,
    };
    use super::kernels::{decoder_cpu_features_from, CpuFeatures};
    use super::{Activation, DecodeDtype};
    use crate::{DecoderError, DecoderResult};
    use edgefirst_tensor::DType;
    use half::f16;

    /// Raw input of a [`Cell`].
    #[derive(Debug, Clone, Copy)]
    pub enum Src<'a> {
        I8(&'a [i8]),
        U8(&'a [u8]),
        I16(&'a [i16]),
        U16(&'a [u16]),
        F16(&'a [f16]),
        F32(&'a [f32]),
    }

    impl<'a> Src<'a> {
        fn view(self) -> InputView<'a> {
            match self {
                Src::I8(s) => InputView::I8(s),
                Src::U8(s) => InputView::U8(s),
                Src::I16(s) => InputView::I16(s),
                Src::U16(s) => InputView::U16(s),
                Src::F16(s) => InputView::F16(s),
                Src::F32(s) => InputView::F32(s),
            }
        }
    }

    /// Output of a [`Cell`].
    #[derive(Debug)]
    pub enum Dst<'a> {
        F32(&'a mut [f32]),
        F16(&'a mut [f16]),
    }

    impl<'a> Dst<'a> {
        fn view(self) -> DstSliceMut<'a> {
            match self {
                Dst::F32(d) => DstSliceMut::F32(d),
                Dst::F16(d) => DstSliceMut::F16(d),
            }
        }
    }

    /// The per-scale role a [`Cell`] decodes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CellKind {
        /// Score dequantize only.
        Score,
        /// Score dequantize and sigmoid.
        ScoreSigmoid,
        MaskCoefs,
        Protos,
    }

    #[derive(Debug, Clone, Copy)]
    enum CellDispatch {
        Score(ScoreLevelDispatch),
        MaskCoefs(MaskCoefDispatch),
        Protos(ProtoDispatch),
    }

    /// One score, mask-coefficient or proto cell as the plan selects it for a
    /// forced tier (an `EDGEFIRST_DECODER_FORCE_KERNEL` name), over a level
    /// of `anchors` rows of `channels`.
    #[derive(Debug)]
    pub struct Cell {
        kind: CellKind,
        dispatch: CellDispatch,
        plan: LevelPlan,
        anchors: usize,
        channels: usize,
    }

    impl Cell {
        pub fn new(
            kind: CellKind,
            input: DType,
            output: DecodeDtype,
            tier: &str,
            anchors: usize,
            channels: usize,
        ) -> DecoderResult<Self> {
            let f = decoder_cpu_features_from(Some(tier))?;
            let dispatch = match kind {
                CellKind::Score | CellKind::ScoreSigmoid => {
                    CellDispatch::Score(ScoreLevelDispatch::select(input, output, &f)?)
                }
                CellKind::MaskCoefs => {
                    CellDispatch::MaskCoefs(MaskCoefDispatch::select(input, output, &f)?)
                }
                CellKind::Protos => CellDispatch::Protos(ProtoDispatch::select(input, output, &f)?),
            };
            let (grid_x, grid_y) = make_anchor_grid(1, anchors);
            Ok(Self {
                kind,
                dispatch,
                plan: LevelPlan {
                    stride: 8.0,
                    h: 1,
                    w: anchors,
                    reg_max: 16,
                    anchor_offset: 0,
                    grid_x,
                    grid_y,
                    box_shape: vec![1, 1, anchors, 64].into_boxed_slice(),
                    score_shape: vec![1, 1, anchors, channels].into_boxed_slice(),
                    mc_shape: Some(vec![1, 1, anchors, channels].into_boxed_slice()),
                    layout: Layout::Nhwc,
                },
                anchors,
                channels,
            })
        }

        /// The selected kernel cell.
        pub fn kernel(&self) -> String {
            match self.dispatch {
                CellDispatch::Score(d) => format!("{d:?}"),
                CellDispatch::MaskCoefs(d) => format!("{:?}", d.0),
                CellDispatch::Protos(d) => format!("{:?}", d.0),
            }
        }

        /// Decode `src` (`anchors * channels` elements) into `dst` (as many); `InvalidShape`,
        /// before writing anything, for other lengths.
        pub fn run(&self, src: Src<'_>, q: Quantization, dst: Dst<'_>) -> DecoderResult<()> {
            let (input, dst) = (src.view(), dst.view());
            let n = self.anchors.checked_mul(self.channels);
            if n != Some(input.len()) || n != Some(dst.len()) {
                return Err(DecoderError::InvalidShape(format!(
                    "cell of {} anchors x {} channels: {} input and {} output values do not match",
                    self.anchors,
                    self.channels,
                    input.len(),
                    dst.len()
                )));
            }
            match self.dispatch {
                CellDispatch::Score(d) => {
                    let act = if self.kind == CellKind::ScoreSigmoid {
                        Activation::Sigmoid
                    } else {
                        Activation::None
                    };
                    d.run(input, q, self.channels, &self.plan, act, dst)
                }
                CellDispatch::MaskCoefs(d) => d.run(input, q, self.channels, &self.plan, dst),
                CellDispatch::Protos(d) => d.run(input, q, dst),
            }
        }
    }

    /// CPU features for a forced tier (an `EDGEFIRST_DECODER_FORCE_KERNEL` name), resolved once.
    #[derive(Debug, Clone, Copy)]
    pub struct Tier(CpuFeatures);

    impl Tier {
        pub fn new(name: &str) -> DecoderResult<Self> {
            decoder_cpu_features_from(Some(name)).map(Self)
        }
    }

    /// The bridge's f16 → f32 widen of one buffer on `tier`.
    pub fn widen_f16(tier: &Tier, src: &[f16], dst: &mut [f32]) {
        super::outputs::widen_f16_into(&tier.0, src, dst);
    }

    /// The prefilter's anchor selection over `[anchors, nc]` raw scores on `tier`.
    pub fn select_anchors(
        tier: &Tier,
        src: Src<'_>,
        nc: usize,
        floor: i32,
        anchors: &mut Vec<u32>,
    ) -> DecoderResult<()> {
        super::prefilter::select_anchors(&src.view(), nc, floor, anchors, &tier.0)
    }

    /// The box encoding of a [`BoxCell`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BoxKind {
        /// DFL with per-tensor quantization.
        Dfl,
        /// DFL with per-channel quantization.
        DflPerChannel,
        /// LTRB distances (YOLO26).
        Ltrb,
    }

    /// One level's box cell as the plan selects it for a forced tier, over `h * w` anchors.
    pub struct BoxCell {
        kind: BoxKind,
        dispatch: super::kernels::dispatch::BoxLevelDispatch,
        per_channel: super::kernels::dispatch::DflPerChannelDispatch,
        plan: LevelPlan,
        table: DflExpTable,
    }

    impl BoxCell {
        /// A cell for `kind` boxes with `reg_max` bins per side (ignored for LTRB), whose exp
        /// table (8-bit DFL) is built for `scale`.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            kind: BoxKind,
            input: DType,
            output: DecodeDtype,
            tier: &str,
            h: usize,
            w: usize,
            reg_max: usize,
            scale: f32,
        ) -> DecoderResult<Self> {
            use super::kernels::dispatch::{BoxLevelDispatch, DflPerChannelDispatch};
            use crate::schema::BoxEncoding;
            let f = decoder_cpu_features_from(Some(tier))?;
            let (encoding, reg_max) = match kind {
                BoxKind::Ltrb => (BoxEncoding::Direct, 1),
                _ => (BoxEncoding::Dfl, reg_max),
            };
            let dispatch = BoxLevelDispatch::select(encoding, input, output, &f)?;
            let per_channel = DflPerChannelDispatch::select(output, &f);
            let (grid_x, grid_y) = make_anchor_grid(h, w);
            Ok(Self {
                kind,
                dispatch,
                per_channel,
                plan: LevelPlan {
                    stride: 8.0,
                    h,
                    w,
                    reg_max,
                    anchor_offset: 0,
                    grid_x,
                    grid_y,
                    box_shape: vec![1, h, w, 4 * reg_max].into_boxed_slice(),
                    score_shape: vec![1, h, w, 1].into_boxed_slice(),
                    mc_shape: None,
                    layout: Layout::Nhwc,
                },
                table: DflExpTable::new(scale),
            })
        }

        /// The selected kernel cell.
        pub fn kernel(&self) -> String {
            match self.kind {
                BoxKind::DflPerChannel => format!("{:?}", self.per_channel),
                _ => format!("{:?}", self.dispatch),
            }
        }

        /// Decode `src` (`h * w * 4 * reg_max` values) into `dst` (`h * w * 4`) with `q`, or
        /// with one scale and zero point per channel for [`BoxKind::DflPerChannel`];
        /// `InvalidShape`, before writing anything, for other lengths.
        pub fn run(
            &self,
            src: Src<'_>,
            q: Quantization,
            per_channel: (&[f32], &[i32]),
            dst: Dst<'_>,
        ) -> DecoderResult<()> {
            let (input, dst) = (src.view(), dst.view());
            match self.kind {
                BoxKind::DflPerChannel => {
                    let (scales, zps) = per_channel;
                    self.per_channel.run(input, scales, zps, &self.plan, dst)
                }
                _ => self
                    .dispatch
                    .run(input, q, &self.plan, Some(&self.table), dst),
            }
        }
    }
}

#[cfg(test)]
mod bench_tests {
    use super::bench::*;
    use super::DecodeDtype;
    use crate::{DecoderError, Quantization};
    use edgefirst_tensor::DType;
    use half::f16;

    const TIERS: [&str; 4] = ["scalar", "neon", "neon_fp16", "neon_dotprod"];

    fn invalid_shape<T>(r: crate::DecoderResult<T>) -> bool {
        matches!(r, Err(DecoderError::InvalidShape(_)))
    }

    /// A cell given a short or long `src` or `dst` fails with `InvalidShape` and writes nothing.
    #[test]
    fn cell_rejects_mismatched_lengths_before_writing() {
        let (anchors, channels) = (40, 8);
        let n = anchors * channels;
        let q = Quantization::new(0.1, 3);
        let kinds = [
            CellKind::Score,
            CellKind::ScoreSigmoid,
            CellKind::MaskCoefs,
            CellKind::Protos,
        ];
        for tier in TIERS {
            for kind in kinds {
                for input in [DType::I8, DType::U8, DType::I16, DType::F32] {
                    for output in [DecodeDtype::F32, DecodeDtype::F16] {
                        let Ok(cell) = Cell::new(kind, input, output, tier, anchors, channels)
                        else {
                            continue;
                        };
                        let (i8s, u8s) = (vec![1i8; n + 16], vec![1u8; n + 16]);
                        let (i16s, f32s) = (vec![1i16; n + 16], vec![1f32; n + 16]);
                        let src = |len: usize| match input {
                            DType::I8 => Src::I8(&i8s[..len]),
                            DType::U8 => Src::U8(&u8s[..len]),
                            DType::I16 => Src::I16(&i16s[..len]),
                            _ => Src::F32(&f32s[..len]),
                        };
                        let label = format!("{tier} {kind:?} {input:?}->{output:?}");
                        for (s, d) in [
                            (n, n - 1),
                            (n, 1),
                            (n, 0),
                            (n, n + 1),
                            (n - 1, n),
                            (n + 16, n),
                        ] {
                            let (mut d32, mut d16) = (vec![-7f32; d], vec![f16::from_f32(-7.0); d]);
                            let dst = match output {
                                DecodeDtype::F32 => Dst::F32(&mut d32),
                                DecodeDtype::F16 => Dst::F16(&mut d16),
                            };
                            let r = cell.run(src(s), q, dst);
                            assert!(invalid_shape(r), "{label} src {s} dst {d}");
                            assert!(d32.iter().all(|&v| v == -7.0), "{label} wrote f32");
                            assert!(d16.iter().all(|&v| v.to_f32() == -7.0), "{label} wrote f16");
                        }
                        let (mut d32, mut d16) = (vec![0f32; n], vec![f16::ZERO; n]);
                        let dst = match output {
                            DecodeDtype::F32 => Dst::F32(&mut d32),
                            DecodeDtype::F16 => Dst::F16(&mut d16),
                        };
                        assert!(cell.run(src(n), q, dst).is_ok(), "{label}");
                    }
                }
            }
        }
    }

    /// A box cell given a short `src` or `dst` fails with `InvalidShape` and writes nothing.
    #[test]
    fn box_cell_rejects_mismatched_lengths_before_writing() {
        let (h, w) = (3, 20);
        let q = Quantization::new(0.1, 3);
        for tier in TIERS {
            for kind in [BoxKind::Dfl, BoxKind::DflPerChannel, BoxKind::Ltrb] {
                for output in [DecodeDtype::F32, DecodeDtype::F16] {
                    let Ok(cell) = BoxCell::new(kind, DType::I8, output, tier, h, w, 16, 0.1)
                    else {
                        continue;
                    };
                    let per = if kind == BoxKind::Ltrb { 4 } else { 64 };
                    let (n, m) = (h * w * per, h * w * 4);
                    let src = vec![1i8; n];
                    let (scales, zps) = (vec![0.1f32; 64], vec![0i32; 64]);
                    for (s, d) in [(n, m - 1), (n, 0), (n - 1, m)] {
                        let (mut d32, mut d16) = (vec![-7f32; d], vec![f16::from_f32(-7.0); d]);
                        let dst = match output {
                            DecodeDtype::F32 => Dst::F32(&mut d32),
                            DecodeDtype::F16 => Dst::F16(&mut d16),
                        };
                        let r = cell.run(Src::I8(&src[..s]), q, (&scales, &zps), dst);
                        assert!(invalid_shape(r), "{tier} {kind:?} {output:?} {s} {d}");
                        assert!(d32.iter().all(|&v| v == -7.0));
                        assert!(d16.iter().all(|&v| v.to_f32() == -7.0));
                    }
                }
            }
        }
    }

    /// Every DFL kernel hook given a short `input` or `dst` fails with `InvalidShape` and writes
    /// nothing.
    #[test]
    fn dfl_level_rejects_mismatched_lengths_before_writing() {
        let level = DflLevel::new(5, 4, 8.0, 0.1, 2);
        let n = level.anchors();
        let input = vec![3i8; n * 64];
        for kernel in DflKernel::available() {
            for (s, d) in [(n * 64, n * 4 - 1), (n * 64, 0), (n * 64 - 1, n * 4)] {
                let mut dst = vec![-7f32; d];
                let r = level.decode(kernel, &input[..s], &mut dst);
                assert!(invalid_shape(r), "{kernel:?} {s} {d}");
                assert!(dst.iter().all(|&v| v == -7.0), "{kernel:?}");
            }
            let mut dst = vec![0f32; n * 4];
            assert!(level.decode(kernel, &input, &mut dst).is_ok(), "{kernel:?}");
        }
    }
}
