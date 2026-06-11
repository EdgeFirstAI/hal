//! Pure decision table for the GL proto-segmentation render path.
//!
//! `plan_proto` maps `(proto dtype, layout, capabilities, interpolation
//! mode)` to a [`ProtoPlan`] — which upload strategy feeds the proto
//! texture, which shader program samples it, and which count uniform that
//! program consumes. The table is GL-free and host-tested exhaustively,
//! mirroring `float_dispatch::classify_float_render` and
//! `render::plan_convert`: capability differences arrive as inputs, never
//! as platform branches inside the renderer.
//!
//! `dequant_coeffs` is the shared coefficient-widening helper for the
//! integer mask-coefficient dtypes (the GL shaders consume f32 uniforms
//! regardless of source dtype).

use edgefirst_decoder::ProtoLayout;
use edgefirst_tensor::{DType, QuantMode};

use super::Int8InterpolationMode;

/// How proto layers reach the GPU texture (all paths land in the shared
/// immutable `TexStorage3D` allocation via `ensure_proto_texture`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProtoUpload {
    /// HWC int8 bytes to an SSBO; a GLES 3.1 compute shader transposes
    /// into the R32I texture via `imageStore`.
    I8Compute,
    /// CPU HWC→CHW repack, uploaded as R8I.
    I8CpuRepack,
    /// CPU HWC→CHW repack, uploaded as R32F (one proto per layer).
    F32R32f,
    /// f32→f16 repack packing 4 protos per RGBA16F layer — the capability
    /// fallback for GPUs without `GL_OES_texture_float_linear`.
    F32ToRgba16f,
    /// Native-f16 shuffle packing 4 protos per RGBA16F layer (no widening
    /// — uploading f16 protos as F32 would double the upload bytes).
    F16Rgba16f,
}

/// Which shader program samples the proto texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProtoProgram {
    Int8Nearest,
    Int8Bilinear,
    /// Two-pass: dequant int8→RGBA16F render, then the f16 program with
    /// hardware GL_LINEAR.
    Int8TwoPass,
    F32,
    /// The RGBA16F-packed program — shared by the f32 fallback and the
    /// native f16 path.
    F16,
}

/// Which layer-count uniform the selected program consumes: `num_protos`
/// (one proto per layer) or `num_layers` (4 protos packed per RGBA layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CountUniform {
    NumProtos,
    NumLayers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ProtoPlan {
    pub(super) upload: ProtoUpload,
    pub(super) program: ProtoProgram,
    pub(super) count_uniform: CountUniform,
}

/// Decide the proto render plan. Pure: capabilities are inputs.
///
/// * `compute_active` — the processor compiled the proto-repack compute
///   program (GLES 3.1 compute present AND `EDGEFIRST_PROTO_COMPUTE=1`).
/// * `has_float_linear` — `GL_OES_texture_float_linear` (possibly forced
///   off via `EDGEFIRST_GL_NO_FLOAT_LINEAR`).
///
/// Errors preserve the dispatcher's user-facing messages: NCHW layouts and
/// unsupported proto dtypes are `NotSupported`/`InvalidShape` exactly as
/// before the table existed.
pub(super) fn plan_proto(
    proto_dtype: DType,
    layout: ProtoLayout,
    compute_active: bool,
    has_float_linear: bool,
    int8_mode: Int8InterpolationMode,
) -> crate::Result<ProtoPlan> {
    // The GL upload path assumes NHWC for ArrayView3 creation; NCHW would
    // need a transpose first — the caller falls back to CPU.
    if layout == ProtoLayout::Nchw {
        return Err(crate::Error::NotSupported(
            "GL segmentation path does not yet support NCHW proto layout; \
             caller should use CPU fallback"
                .into(),
        ));
    }

    let plan = match proto_dtype {
        DType::I8 => {
            let upload = if compute_active {
                ProtoUpload::I8Compute
            } else {
                ProtoUpload::I8CpuRepack
            };
            let program = match int8_mode {
                Int8InterpolationMode::Nearest => ProtoProgram::Int8Nearest,
                Int8InterpolationMode::Bilinear => ProtoProgram::Int8Bilinear,
                Int8InterpolationMode::TwoPass => ProtoProgram::Int8TwoPass,
            };
            // The two-pass dequant target packs 4 protos per RGBA16F layer
            // and samples through the f16 program, but its layer math is
            // internal to the pass — the int8 entry points consume
            // `num_protos`.
            ProtoPlan {
                upload,
                program,
                count_uniform: CountUniform::NumProtos,
            }
        }
        DType::F32 => {
            if has_float_linear {
                ProtoPlan {
                    upload: ProtoUpload::F32R32f,
                    program: ProtoProgram::F32,
                    count_uniform: CountUniform::NumProtos,
                }
            } else {
                ProtoPlan {
                    upload: ProtoUpload::F32ToRgba16f,
                    program: ProtoProgram::F16,
                    count_uniform: CountUniform::NumLayers,
                }
            }
        }
        DType::F16 => ProtoPlan {
            upload: ProtoUpload::F16Rgba16f,
            program: ProtoProgram::F16,
            count_uniform: CountUniform::NumLayers,
        },
        other => {
            return Err(crate::Error::InvalidShape(format!(
                "GL seg path: proto dtype {other:?} not supported"
            )));
        }
    };
    Ok(plan)
}

/// HWC → layer-first repack: one proto per texture layer
/// (`out[k][y][x] = in[y][x][k]`). Monomorphized per element type — the
/// int8 (R8I) and f32 (R32F) upload paths share this exact transpose.
pub(super) fn repack_layers<T: Copy + Default>(protos: ndarray::ArrayView3<'_, T>) -> Vec<T> {
    let (height, width, num_protos) = protos.dim();
    let mut out = vec![T::default(); height * width * num_protos];
    for k in 0..num_protos {
        for y in 0..height {
            for x in 0..width {
                out[k * height * width + y * width + x] = protos[[y, x, k]];
            }
        }
    }
    out
}

/// HWC → RGBA16F-packed layers: 4 protos per layer, zero-padded tail
/// channels when `num_protos % 4 != 0`. Returns the raw f16 bits as bytes
/// plus the layer count. `conv` is monomorphized per source dtype — the
/// f32 fallback narrows (`f16::from_f32`), the native-f16 path passes
/// through; protos are never widened to f32 (doubles upload bytes).
pub(super) fn repack_rgba_f16_layers<S: Copy>(
    protos: ndarray::ArrayView3<'_, S>,
    conv: impl Fn(S) -> half::f16,
) -> (Vec<u8>, usize) {
    let (height, width, num_protos) = protos.dim();
    let num_layers = num_protos.div_ceil(4);
    // Each layer is H*W*4 half-floats, each half-float is 2 bytes.
    let layer_stride = height * width * 4;
    let mut buf = vec![0u16; layer_stride * num_layers];

    for y in 0..height {
        for x in 0..width {
            for k in 0..num_layers * 4 {
                let val = if k < num_protos {
                    conv(protos[[y, x, k]])
                } else {
                    half::f16::ZERO
                };
                let layer = k / 4;
                let channel = k % 4;
                buf[layer * layer_stride + y * width * 4 + x * 4 + channel] = val.to_bits();
            }
        }
    }

    // Reinterpret the u16 buffer as bytes for the GL upload.
    let byte_buf =
        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 2).to_vec() };
    (byte_buf, num_layers)
}

/// Widen integer mask coefficients to the f32 the GL shaders consume,
/// applying per-tensor dequantization. `None` quantization is a plain
/// cast. Per-channel modes are rejected with the dispatcher's original
/// message (a single scale/zp uniform cannot express them).
pub(super) fn dequant_coeffs<T: Copy + Into<f32>>(
    vals: &[T],
    mode: Option<QuantMode<'_>>,
    dtype_label: &str,
) -> crate::Result<Vec<f32>> {
    let (scale, zp) = match mode {
        None => (1.0_f32, 0.0_f32),
        Some(QuantMode::PerTensor { scale, zero_point }) => (scale, zero_point as f32),
        Some(QuantMode::PerTensorSymmetric { scale }) => (scale, 0.0),
        Some(other) => {
            return Err(crate::Error::NotSupported(format!(
                "{dtype_label} mask_coefficients quantization mode {other:?} \
                 not supported on GL seg path"
            )));
        }
    };
    Ok(vals.iter().map(|&v| (v.into() - zp) * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [Int8InterpolationMode; 3] = [
        Int8InterpolationMode::Nearest,
        Int8InterpolationMode::Bilinear,
        Int8InterpolationMode::TwoPass,
    ];

    /// Every (dtype, capability, mode) row of the table, exhaustively.
    #[test]
    fn plan_proto_full_table() {
        for compute in [false, true] {
            for linear in [false, true] {
                // I8: upload follows compute_active, program follows mode,
                // float-linear is irrelevant.
                for mode in MODES {
                    let plan =
                        plan_proto(DType::I8, ProtoLayout::Nhwc, compute, linear, mode).unwrap();
                    assert_eq!(
                        plan.upload,
                        if compute {
                            ProtoUpload::I8Compute
                        } else {
                            ProtoUpload::I8CpuRepack
                        }
                    );
                    assert_eq!(
                        plan.program,
                        match mode {
                            Int8InterpolationMode::Nearest => ProtoProgram::Int8Nearest,
                            Int8InterpolationMode::Bilinear => ProtoProgram::Int8Bilinear,
                            Int8InterpolationMode::TwoPass => ProtoProgram::Int8TwoPass,
                        }
                    );
                    assert_eq!(plan.count_uniform, CountUniform::NumProtos);
                }

                // F32: float-linear picks native R32F vs the RGBA16F
                // fallback; compute and int8 mode are irrelevant.
                for mode in MODES {
                    let plan =
                        plan_proto(DType::F32, ProtoLayout::Nhwc, compute, linear, mode).unwrap();
                    if linear {
                        assert_eq!(plan.upload, ProtoUpload::F32R32f);
                        assert_eq!(plan.program, ProtoProgram::F32);
                        assert_eq!(plan.count_uniform, CountUniform::NumProtos);
                    } else {
                        assert_eq!(plan.upload, ProtoUpload::F32ToRgba16f);
                        assert_eq!(plan.program, ProtoProgram::F16);
                        assert_eq!(plan.count_uniform, CountUniform::NumLayers);
                    }
                }

                // F16 native: one row regardless of capabilities.
                for mode in MODES {
                    let plan =
                        plan_proto(DType::F16, ProtoLayout::Nhwc, compute, linear, mode).unwrap();
                    assert_eq!(plan.upload, ProtoUpload::F16Rgba16f);
                    assert_eq!(plan.program, ProtoProgram::F16);
                    assert_eq!(plan.count_uniform, CountUniform::NumLayers);
                }
            }
        }
    }

    #[test]
    fn plan_proto_rejects_nchw_for_every_dtype() {
        for dtype in [DType::I8, DType::F32, DType::F16] {
            let err = plan_proto(
                dtype,
                ProtoLayout::Nchw,
                false,
                true,
                Int8InterpolationMode::Bilinear,
            )
            .unwrap_err();
            assert!(
                matches!(err, crate::Error::NotSupported(ref m) if m.contains("NCHW")),
                "{dtype:?}: {err:?}"
            );
        }
    }

    #[test]
    fn plan_proto_rejects_unsupported_dtypes() {
        for dtype in [DType::U8, DType::I16, DType::I32, DType::F64] {
            let err = plan_proto(
                dtype,
                ProtoLayout::Nhwc,
                false,
                true,
                Int8InterpolationMode::Bilinear,
            )
            .unwrap_err();
            assert!(
                matches!(err, crate::Error::InvalidShape(ref m) if m.contains("not supported")),
                "{dtype:?}: {err:?}"
            );
        }
    }

    #[test]
    fn repack_layers_transposes_hwc_to_layer_first() {
        // 2x2x3 HWC: value encodes (y, x, k) as y*100 + x*10 + k.
        let data: Vec<i32> = (0..2 * 2 * 3)
            .map(|i| {
                let (y, x, k) = (i / 6, (i / 3) % 2, i % 3);
                y * 100 + x * 10 + k
            })
            .collect();
        let view = ndarray::ArrayView3::from_shape((2, 2, 3), &data).unwrap();
        let out = repack_layers(view);
        // out[k][y][x]
        assert_eq!(out.len(), 12);
        for k in 0..3 {
            for y in 0..2 {
                for x in 0..2 {
                    assert_eq!(out[k * 4 + y * 2 + x] as usize, y * 100 + x * 10 + k);
                }
            }
        }
    }

    #[test]
    fn repack_rgba_f16_layers_packs_quads_and_zero_pads_tail() {
        // 1x2x6 HWC → 2 RGBA layers; 6 protos fill layer 0 fully and half
        // of layer 1 (channels 2,3 of every pixel zero-padded).
        let data: Vec<f32> = (0..2 * 6).map(|i| i as f32).collect();
        let view = ndarray::ArrayView3::from_shape((1, 2, 6), &data).unwrap();
        let (bytes, num_layers) = repack_rgba_f16_layers(view, half::f16::from_f32);
        assert_eq!(num_layers, 2);
        let halves: Vec<half::f16> = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        let layer_stride = 2 * 4; // H*W*4 per layer
        for x in 0..2 {
            for k in 0..8 {
                let expected = if k < 6 { (x * 6 + k) as f32 } else { 0.0 };
                let got = halves[(k / 4) * layer_stride + x * 4 + (k % 4)].to_f32();
                assert_eq!(got, expected, "x={x} k={k}");
            }
        }

        // Native-f16 identity conversion produces the identical packing.
        let data_f16: Vec<half::f16> = data.iter().map(|v| half::f16::from_f32(*v)).collect();
        let view_f16 = ndarray::ArrayView3::from_shape((1, 2, 6), &data_f16).unwrap();
        let (bytes_native, layers_native) = repack_rgba_f16_layers(view_f16, |v| v);
        assert_eq!(layers_native, 2);
        assert_eq!(bytes_native, bytes);
    }

    #[test]
    fn dequant_coeffs_per_tensor() {
        let vals: [i8; 4] = [-115, 0, 20, 127];
        let out = dequant_coeffs(
            &vals,
            Some(QuantMode::PerTensor {
                scale: 0.5,
                zero_point: 20,
            }),
            "I8",
        )
        .unwrap();
        assert_eq!(out, vec![-67.5, -10.0, 0.0, 53.5]);
    }

    #[test]
    fn dequant_coeffs_symmetric_and_passthrough() {
        let vals: [i16; 3] = [-2, 0, 2];
        let sym = dequant_coeffs(
            &vals,
            Some(QuantMode::PerTensorSymmetric { scale: 0.25 }),
            "I16",
        )
        .unwrap();
        assert_eq!(sym, vec![-0.5, 0.0, 0.5]);

        // No quantization metadata → plain cast.
        let plain = dequant_coeffs(&vals, None, "I16").unwrap();
        assert_eq!(plain, vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn dequant_coeffs_rejects_per_channel() {
        let vals: [i8; 2] = [1, 2];
        let scales = [0.5_f32];
        let err = dequant_coeffs(
            &vals,
            Some(QuantMode::PerChannelSymmetric {
                scales: &scales,
                axis: 0,
            }),
            "I8",
        )
        .unwrap_err();
        assert!(
            matches!(err, crate::Error::NotSupported(ref m) if m.contains("quantization mode")),
            "{err:?}"
        );
    }
}
