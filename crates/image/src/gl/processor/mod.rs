// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{
    DType, PixelFormat, PixelLayout, Tensor, TensorMapTrait, TensorMemory, TensorTrait,
};
use edgefirst_tensor::{DetectBox, ProtoData, ProtoLayout, Segmentation};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr};
use std::time::Instant;

use super::EglDisplayKind;

use super::cache::{BufferImportKey, CacheKind, GlCacheStats, ImportCache};
use super::platform::{GlPlatform, NvDstPlanes, Platform};

/// The platform display/context owned by this processor — `GlContext` on
/// Linux, `AngleDisplay` (per-processor ANGLE context) on macOS.
type PlatformDisplay = <Platform as GlPlatform>::Display;
/// The owned platform import the caches hold (`EglImage` / `IoSurfacePbuffer`).
type PlatformImport = <Platform as GlPlatform>::Import;
/// The `Copy` import handle (`egl::Image` / `egl::Surface`).
type PlatformHandle = <Platform as GlPlatform>::ImportHandle;
use super::resources::{Buffer, FrameBuffer, GlProgram, Texture, UploadSource};
use super::shaders::{
    check_gl_error, generate_color_shader, generate_float_nhwc_packed_shader,
    generate_float_rgba_shader, generate_instanced_segmentation_shader,
    generate_nv_to_rgba_bilinear_int8_shader_2d, generate_nv_to_rgba_bilinear_shader_2d,
    generate_nv_to_rgba_hw_bilinear_int8_shader_2d, generate_nv_to_rgba_hw_bilinear_shader_2d,
    generate_nv_to_rgba_int8_shader_2d, generate_nv_to_rgba_shader_2d,
    generate_nv_to_rgba_y2y_int8_shader, generate_nv_to_rgba_y2y_shader,
    generate_packed_f32_nhwc_shader, generate_packed_rgba8_int8_shader_2d,
    generate_packed_rgba8_shader_2d, generate_planar_rgb_f16_packed_shader,
    generate_planar_rgb_int8_shader, generate_planar_rgb_int8_shader_2d,
    generate_planar_rgb_shader, generate_planar_rgb_shader_2d, generate_proto_dequant_shader_int8,
    generate_proto_repack_compute_shader, generate_proto_segmentation_shader,
    generate_proto_segmentation_shader_f32, generate_proto_segmentation_shader_int8_bilinear,
    generate_proto_segmentation_shader_int8_nearest,
    generate_rgb_r8_to_rgba_bilinear_int8_shader_2d, generate_rgb_r8_to_rgba_bilinear_shader_2d,
    generate_rgb_r8_to_rgba_hw_bilinear_int8_shader_2d,
    generate_rgb_r8_to_rgba_hw_bilinear_shader_2d, generate_rgb_r8_to_rgba_int8_shader_2d,
    generate_rgb_r8_to_rgba_shader_2d, generate_segmentation_shader,
    generate_texture_fragment_shader, generate_texture_fragment_shader_yuv,
    generate_texture_int8_shader, generate_texture_int8_shader_yuv, generate_vertex_shader,
    quant_mapped,
};
use super::{Int8InterpolationMode, RegionOfInterest, TransferBackend};
use crate::{Error, Flip, ImageProcessorTrait, ResolvedCrop, Rotation, DEFAULT_COLORS};
use edgefirst_tensor::TensorDyn;

/// GL float (F16/F32) preprocessing paths.
mod float;
// Re-exports the float classifier/support items at the `processor` module path.
pub(super) use float::{classify_float_render, float_render_support, FloatRenderPath};
// Re-exported for the `gl::tests` unit tests.
#[allow(unused_imports)]
pub(super) use float::packed_planar_layout;

/// Which GPU/CPU path was taken for the most recent NV* convert call.
///
/// Not reset by non-NV* converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvConvertPath {
    /// Driver-decoded YUV via `samplerExternalOES` EGLImage. The GPU driver
    /// performs YUV→RGB (matrix + chroma upsampling).
    ExternalSampler,
    /// R8 `texelFetch` shader applying the per-tensor YUV→RGB matrix in the
    /// shader (ES 3.0, no extension).
    ShaderR8,
    /// CPU fallback (EGLImage creation failed or format not DMA-backed).
    Cpu,
    /// Not yet set (initial state, or non-NV* convert).
    None,
}

/// Client preference for the NV* GPU conversion path, set via
/// `EDGEFIRST_NV_CONVERT_PATH` (`sampler` | `shader` | `auto`).
///
/// `Auto` leaves the choice to `select_nv_path`. A forced path that is
/// unavailable logs a warning and the available one is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NvPathPref {
    Auto,
    ForceSampler,
    ForceShader,
}

/// Bound destination render target for one convert, produced by
/// [`GLProcessorST::bind_dst`]. The GL counterpart of
/// [`super::render::DstLowering`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DstTarget {
    /// The destination buffer itself is the FBO colour attachment (EGLImage
    /// renderbuffer or texture): the render writes it directly.
    ZeroCopyImage,
    /// Offscreen texture render target; `readback` copies the result out.
    Texture { readback: DstReadback },
}

/// How a [`DstTarget::Texture`] render reaches the destination tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DstReadback {
    /// `glReadPixels` straight into the mapped Mem tensor.
    Mem,
    /// `glReadPixels` into this destination PBO's PACK binding.
    Pbo(u32),
}

impl DstTarget {
    /// The `dst_feed` value for this target: `zero_copy` when the render
    /// writes the destination buffer itself, `mapped_texture` when an
    /// offscreen texture is read back through the tensor's `map()`, `pbo`
    /// when it is read back into the destination PBO.
    fn feed(self) -> &'static str {
        match self {
            DstTarget::ZeroCopyImage => "zero_copy",
            DstTarget::Texture {
                readback: DstReadback::Mem,
            } => "mapped_texture",
            DstTarget::Texture {
                readback: DstReadback::Pbo(_),
            } => "pbo",
        }
    }
}

impl DstReadback {
    /// The PBO id in the `Option` shape `readback_rendered` consumes.
    fn pbo_id(self) -> Option<u32> {
        match self {
            DstReadback::Mem => None,
            DstReadback::Pbo(id) => Some(id),
        }
    }
}

/// Borrowed proto data for the layered-float render plans, matched by
/// `render_proto_layers` against `ProtoPlan::upload`.
enum ProtoLayersData<'a> {
    F32(ndarray::ArrayView3<'a, f32>),
    F16(ndarray::ArrayView3<'a, half::f16>),
}

/// Uniform locations for one `nv_r8` program variant, resolved once at link
/// time.
#[derive(Clone, Copy)]
struct NvUniformLocs {
    img_size: i32,
    tex_width: i32,
    chroma_shift: i32,
    chroma_lines: i32,
    src_rect: i32,
    luma_scale: i32,
    luma_clamp: i32,
    chroma_scale: i32,
    chroma_clamp: i32,
    src_extent: i32,
    y_offset: i32,
    y_scale: i32,
    c_vr: i32,
    c_ug: i32,
    c_vg: i32,
    c_ub: i32,
}

/// Uniform locations for one float-path program, resolved once per program
/// and read by `draw_float_quad`. A location of `-1` means the uniform is
/// absent in that program variant.
#[derive(Clone, Copy)]
pub(super) struct FloatQuadLocs {
    pub(super) sampler: i32,
    pub(super) src_rect_uv: i32,
    pub(super) src_extent: i32,
    pub(super) dst_rect_px: i32,
    pub(super) pad_color: i32,
    pub(super) dst_image_size: i32,
    pub(super) norm_scale: i32,
    pub(super) norm_offset: i32,
}

/// Per-channel output mapping for the float render paths, in sampler units
/// (`output = sample * scale + offset`), plus the letterbox pad as written.
/// Identity outside a convert that asks for normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FloatNorm {
    pub(super) scale: [f32; 4],
    pub(super) offset: [f32; 4],
    pub(super) pad: Option<[f32; 4]>,
}

impl FloatNorm {
    pub(super) const IDENTITY: Self = Self {
        scale: [1.0; 4],
        offset: [0.0; 4],
        pad: None,
    };
}

/// The `quantize()` uniforms of the int8 programs, in sampler units for the
/// RGB channels: `clamp(floor(v * scale + offset + 0.5) + zero, lo, hi)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct IntQuant {
    pub(super) scale: [f32; 3],
    pub(super) offset: [f32; 3],
    pub(super) zero: [f32; 3],
    pub(super) lo: [f32; 3],
    pub(super) hi: [f32; 3],
}

impl IntQuant {
    /// From an integer output map in RGB order.
    pub(super) fn from_map(map: &crate::options::OutputMap) -> Self {
        let b = map.bounds.unwrap_or([(0.0, 255.0); 4]);
        Self {
            scale: std::array::from_fn(|c| map.scale[c] * 255.0),
            offset: std::array::from_fn(|c| map.offset[c]),
            zero: std::array::from_fn(|c| map.zero_point[c]),
            lo: std::array::from_fn(|c| b[c].0),
            hi: std::array::from_fn(|c| b[c].1),
        }
    }
}

/// One build of every int8 program, with the link-time state its draws read.
/// The processor's int8 fields hold the active build and `int8_alt` the
/// other; [`GLProcessorST::use_int8_set`] swaps them.
struct Int8Set {
    texture: GlProgram,
    texture_yuv: Option<GlProgram>,
    planar: Option<GlProgram>,
    planar_2d: GlProgram,
    yuyv_2d: GlProgram,
    yuyv_2d_locs: [i32; 8],
    packed_rgba8_2d: GlProgram,
    /// In [`NvSampling::program`] order, int8 entries only.
    nv_r8: [GlProgram; 3],
    nv_r8_uniforms: [NvUniformState; 3],
    nv_y2y: Option<(GlProgram, NvUniformState)>,
    /// In [`NvSampling::program`] order, int8 entries only.
    rgb_r8: [GlProgram; 3],
    rgb_r8_locs: [PackedR8Locs; 3],
    /// In [`NvSampling::program`] order, int8 entries only.
    yuyv_luma_r8: [GlProgram; 3],
    yuyv_luma_r8_locs: [PackedR8Locs; 3],
    luma_2d: GlProgram,
    texture_src_extent_loc: i32,
    yuv_src_extent_loc: i32,
    planar_src_extent_loc: i32,
}

impl Int8Set {
    /// Link the int8 programs: `quantize()` as uniforms when `mapped`, else
    /// the plain bias in literals. The Y2Y program is built only with `y2y`,
    /// and is `None` if the driver refuses it.
    fn new(mapped: bool, y2y: bool) -> Result<Self, Error> {
        let build = |src: &str| {
            if mapped {
                GlProgram::new(generate_vertex_shader(), &quant_mapped(src))
            } else {
                GlProgram::new(generate_vertex_shader(), src)
            }
        };
        let external = |src: &str| Platform::EXTERNAL_OES.then(|| build(src)).transpose();
        let texture = build(generate_texture_int8_shader())?;
        let texture_yuv = external(generate_texture_int8_shader_yuv())?;
        let planar = external(generate_planar_rgb_int8_shader())?;
        let planar_2d = build(generate_planar_rgb_int8_shader_2d())?;
        planar_2d.load_uniform_1i(c"tex", 0)?;
        let yuyv_2d = build(super::shaders_common::YUYV_RGBA_2D_INT8_FRAGMENT)?;
        let yuyv_2d_locs = yuyv_2d_locs(&yuyv_2d);
        let packed_rgba8_2d = build(generate_packed_rgba8_int8_shader_2d())?;
        packed_rgba8_2d.load_uniform_1i(c"tex", 1)?;
        let nv_r8 = [
            build(generate_nv_to_rgba_int8_shader_2d())?,
            build(generate_nv_to_rgba_bilinear_int8_shader_2d())?,
            build(generate_nv_to_rgba_hw_bilinear_int8_shader_2d())?,
        ];
        let nv_r8_uniforms = nv_r8.each_ref().map(NvUniformState::resolve);
        let rgb_r8 = [
            build(generate_rgb_r8_to_rgba_int8_shader_2d())?,
            build(generate_rgb_r8_to_rgba_bilinear_int8_shader_2d())?,
            build(generate_rgb_r8_to_rgba_hw_bilinear_int8_shader_2d())?,
        ];
        let rgb_r8_locs = rgb_r8.each_ref().map(PackedR8Locs::resolve);
        let yuyv_luma_r8 = [
            build(super::shaders_common::YUYV_LUMA_R8_INT8_FRAGMENT)?,
            build(super::shaders_common::YUYV_LUMA_R8_BILINEAR_INT8_FRAGMENT)?,
            build(super::shaders_common::YUYV_LUMA_R8_HW_BILINEAR_INT8_FRAGMENT)?,
        ];
        let yuyv_luma_r8_locs = yuyv_luma_r8.each_ref().map(PackedR8Locs::resolve);
        let luma_2d = build(super::shaders_common::LUMA_2D_INT8_FRAGMENT)?;
        luma_2d.load_uniform_1i(c"tex", 0)?;
        let nv_y2y = if y2y {
            match build(generate_nv_to_rgba_y2y_int8_shader()) {
                Ok(p) => {
                    let u = NvUniformState::resolve(&p);
                    Some((p, u))
                }
                Err(e) => {
                    log::info!("the int8 Y2Y NV program did not build: {e:?}");
                    None
                }
            }
        } else {
            None
        };
        let src_extent = |p: Option<&GlProgram>| {
            p.map_or(-1, |p| {
                // SAFETY: `p` was linked above on this thread's current
                // context and the name is a `c"..."` literal.
                unsafe { edgefirst_gl::gl::GetUniformLocation(p.id, c"src_extent".as_ptr()) }
            })
        };
        Ok(Self {
            texture_src_extent_loc: src_extent(Some(&texture)),
            yuv_src_extent_loc: src_extent(texture_yuv.as_ref()),
            planar_src_extent_loc: src_extent(planar.as_ref()),
            texture,
            texture_yuv,
            planar,
            planar_2d,
            yuyv_2d,
            yuyv_2d_locs,
            packed_rgba8_2d,
            nv_r8,
            nv_r8_uniforms,
            nv_y2y,
            rgb_r8,
            rgb_r8_locs,
            yuyv_luma_r8,
            yuyv_luma_r8_locs,
            luma_2d,
        })
    }
}

/// Bind `tex` to unit 0 and resolve the YUYV program's uniforms, in the
/// order `(src_size, y_offset, y_scale, c_vr, c_ug, c_vg, c_ub, src_extent)`.
fn yuyv_2d_locs(p: &GlProgram) -> [i32; 8] {
    // SAFETY: `p` is linked on this thread's current context; each name is a
    // `c"..."` literal.
    unsafe {
        edgefirst_gl::gl::UseProgram(p.id);
        let tex = edgefirst_gl::gl::GetUniformLocation(p.id, c"tex".as_ptr());
        edgefirst_gl::gl::Uniform1i(tex, 0);
        [
            c"src_size",
            c"y_offset",
            c"y_scale",
            c"c_vr",
            c"c_ug",
            c"c_vg",
            c"c_ub",
            c"src_extent",
        ]
        .map(|name| edgefirst_gl::gl::GetUniformLocation(p.id, name.as_ptr()))
    }
}

/// Per-program uniform state for one `nv_r8` variant: the uniform
/// locations, the (encoding, range) last uploaded (`last_colorimetry`),
/// and whether it was uploaded with the chroma terms zeroed
/// (`last_luma_only`).
struct NvUniformState {
    locs: NvUniformLocs,
    last_colorimetry: Option<(
        edgefirst_tensor::ColorEncoding,
        edgefirst_tensor::ColorRange,
    )>,
    last_luma_only: bool,
}

impl NvUniformState {
    fn resolve(program: &GlProgram) -> Self {
        unsafe {
            edgefirst_gl::gl::UseProgram(program.id);
            let loc = |name: &std::ffi::CStr| {
                edgefirst_gl::gl::GetUniformLocation(program.id, name.as_ptr())
            };
            // The NV shaders sample `src` from unit 0, and the hardware-filtered
            // program its chroma view from unit 1.
            edgefirst_gl::gl::Uniform1i(loc(c"src"), 0);
            edgefirst_gl::gl::Uniform1i(loc(c"uv_tex"), 1);
            NvUniformState {
                locs: NvUniformLocs {
                    img_size: loc(c"img_size"),
                    tex_width: loc(c"tex_width"),
                    chroma_shift: loc(c"chroma_shift"),
                    chroma_lines: loc(c"chroma_lines"),
                    src_rect: loc(c"src_rect"),
                    luma_scale: loc(c"luma_scale"),
                    luma_clamp: loc(c"luma_clamp"),
                    chroma_scale: loc(c"chroma_scale"),
                    chroma_clamp: loc(c"chroma_clamp"),
                    src_extent: loc(c"src_extent"),
                    y_offset: loc(c"y_offset"),
                    y_scale: loc(c"y_scale"),
                    c_vr: loc(c"c_vr"),
                    c_ug: loc(c"c_ug"),
                    c_vg: loc(c"c_vg"),
                    c_ub: loc(c"c_ub"),
                },
                last_colorimetry: None,
                last_luma_only: false,
            }
        }
    }
}

/// The source pixels an `nv_r8` draw samples: the crop rectangle in source
/// pixels (`right`/`bottom` exclusive) and whether the draw rescales it.
#[derive(Clone, Copy, Debug)]
struct NvSampleRect {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    resample: bool,
    /// Pixels the draw covers on the destination.
    dst_pixels: usize,
}

impl NvSampleRect {
    /// `src_crop` and `dst_crop` default to the whole source and destination.
    /// A quarter-turn rotation swaps the crop's axes on the destination.
    fn new(
        src_crop: Option<crate::Rect>,
        dst_crop: Option<crate::Rect>,
        (src_w, src_h): (usize, usize),
        (dst_w, dst_h): (usize, usize),
        rotation: crate::Rotation,
    ) -> Self {
        let src = src_crop.unwrap_or(crate::Rect::new(0, 0, src_w, src_h));
        let dst = dst_crop.unwrap_or(crate::Rect::new(0, 0, dst_w, dst_h));
        let (w, h) = match rotation {
            crate::Rotation::Clockwise90 | crate::Rotation::CounterClockwise90 => {
                (src.height, src.width)
            }
            _ => (src.width, src.height),
        };
        NvSampleRect {
            left: src.left,
            top: src.top,
            right: src.left + src.width,
            bottom: src.top + src.height,
            resample: (w, h) != (dst.width, dst.height),
            dst_pixels: dst.width * dst.height,
        }
    }

    /// The rectangle as the [`RegionOfInterest`] of a `src_w`x`src_h` source,
    /// for the draws that sample through a filtered sampler.
    fn roi(&self, src_w: usize, src_h: usize) -> RegionOfInterest {
        let crop = crate::Rect::new(
            self.left,
            self.top,
            self.right - self.left,
            self.bottom - self.top,
        );
        RegionOfInterest::from_crop(&crop, src_w, src_h)
    }
}

/// The (encoding, range) an NV draw's in-shader matrix applies: the source's
/// own colorimetry, with missing axes filled by the SD/HD height heuristic.
fn nv_colorimetry(
    colorimetry: Option<edgefirst_tensor::Colorimetry>,
    height: Option<usize>,
) -> (
    edgefirst_tensor::ColorEncoding,
    edgefirst_tensor::ColorRange,
) {
    let cm = crate::colorimetry::resolve_colorimetry(colorimetry, height);
    (
        cm.encoding
            .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709),
        cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited),
    )
}

/// Largest source-to-output pixel ratio at which a bilinear NV resize takes
/// [`NvSampling::HardwareYuv`].
const Y2Y_MAX_SOURCE_PER_OUTPUT_PIXEL: usize = 10;

/// Whether a bilinear resize of a `src_pixels` source into `dst_pixels`
/// output pixels takes [`NvSampling::HardwareYuv`]: at most
/// [`Y2Y_MAX_SOURCE_PER_OUTPUT_PIXEL`] source pixels per output pixel.
fn y2y_pays_off(src_pixels: usize, dst_pixels: usize) -> bool {
    src_pixels <= Y2Y_MAX_SOURCE_PER_OUTPUT_PIXEL * dst_pixels
}

/// How an `nv_r8` draw samples the source, which selects its program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NvSampling {
    /// One texel per pixel: a draw that does not rescale, or
    /// [`crate::Interpolation::Nearest`].
    Single = 0,
    /// Bilinear blended in the shader from four single-texel conversions.
    Shader = 1,
    /// Bilinear through the texture units' LINEAR filter, on the combined
    /// plane for luma and a (U, V) view of the chroma plane.
    Hardware = 2,
    /// Bilinear through the texture unit from one NV12 import sampled as raw
    /// YUV (`GL_EXT_YUV_target`). Drawn by the Y2Y programs, not
    /// `nv_r8_programs`.
    HardwareYuv = 3,
}

/// What an `nv_r8` draw can bind, which decides its [`NvSampling`].
#[derive(Clone, Copy, Debug)]
struct NvSamplingInputs {
    /// The draw rescales and the interpolation is bilinear.
    bilinear: bool,
    /// The texture units may filter (`nv_hw_filter`).
    hw_filter: bool,
    /// The source is a DMA-BUF import rather than an upload.
    imported: bool,
    /// The import's chroma plane is bound as its own (U, V) texture.
    chroma_view: bool,
    /// The Y2Y programs built, the source is NV12, and [`y2y_pays_off`].
    y2y: bool,
}

impl NvSampling {
    /// The sampling for a draw that can bind `inputs`: `Single` without
    /// bilinear, `Hardware` for an upload or an import with a chroma view,
    /// `HardwareYuv` where Y2Y is available, `Shader` otherwise.
    fn choose(inputs: NvSamplingInputs) -> Self {
        let NvSamplingInputs {
            bilinear,
            hw_filter,
            imported,
            chroma_view,
            y2y,
        } = inputs;
        if !bilinear {
            NvSampling::Single
        } else if hw_filter && (!imported || chroma_view) {
            NvSampling::Hardware
        } else if hw_filter && y2y {
            NvSampling::HardwareYuv
        } else {
            NvSampling::Shader
        }
    }

    /// Index into `nv_r8_programs`: two programs (u8, int8) per sampling.
    fn program(self, is_int8: bool) -> usize {
        self as usize * 2 + usize::from(is_int8)
    }
}

/// Uniform locations for one packed R8-view program (see
/// [`NvUniformLocs`]). `y_offset`/`y_scale` exist only in the YUYV luma
/// programs and are -1 in the RGB ones.
#[derive(Clone, Copy)]
struct PackedR8Locs {
    img_size: i32,
    inv_tex: i32,
    src_rect: i32,
    y_offset: i32,
    y_scale: i32,
}

impl PackedR8Locs {
    fn resolve(program: &GlProgram) -> Self {
        // SAFETY: `program` is linked on this thread's current context and
        // each name is a `c"..."` literal.
        unsafe {
            edgefirst_gl::gl::UseProgram(program.id);
            let loc = |name: &std::ffi::CStr| {
                edgefirst_gl::gl::GetUniformLocation(program.id, name.as_ptr())
            };
            // The packed programs sample `src` from unit 0.
            edgefirst_gl::gl::Uniform1i(loc(c"src"), 0);
            PackedR8Locs {
                img_size: loc(c"img_size"),
                inv_tex: loc(c"inv_tex"),
                src_rect: loc(c"src_rect"),
                y_offset: loc(c"y_offset"),
                y_scale: loc(c"y_scale"),
            }
        }
    }
}

/// The route a refused zero-copy destination import takes instead.
const TEXTURE_READBACK: &str =
    "rendering to a texture and reading back through map() (slower, still correct)";

/// The RGBA -> NV12 programs and their link-time uniform locations.
struct Nv12Programs {
    /// `NV12_LUMA_FRAGMENT`.
    luma: GlProgram,
    /// `NV12_CHROMA_R8_FRAGMENT`.
    chroma: GlProgram,
    /// `NV12_CHROMA_FILL_FRAGMENT`.
    fill: GlProgram,
    /// `y_coeffs`, `y_offset` of `luma`.
    luma_locs: [i32; 2],
    /// `u_coeffs`, `v_coeffs`, `c_offset`, `chroma_row` of `chroma`.
    chroma_locs: [i32; 4],
    /// `uv` of `fill`.
    fill_uv: i32,
}

impl Nv12Programs {
    fn new() -> Result<Self, crate::Error> {
        let build = |fragment| GlProgram::new(generate_vertex_shader(), fragment);
        let luma = build(super::shaders_common::NV12_LUMA_FRAGMENT)?;
        let chroma = build(super::shaders_common::NV12_CHROMA_R8_FRAGMENT)?;
        let fill = build(super::shaders_common::NV12_CHROMA_FILL_FRAGMENT)?;
        luma.load_uniform_1i(c"tex", 0)?;
        chroma.load_uniform_1i(c"tex", 0)?;
        let loc = |program: &GlProgram, name: &CStr| {
            // SAFETY: `program` is linked on this thread's current context.
            unsafe { edgefirst_gl::gl::GetUniformLocation(program.id, name.as_ptr()) }
        };
        Ok(Self {
            luma_locs: [loc(&luma, c"y_coeffs"), loc(&luma, c"y_offset")],
            chroma_locs: [
                loc(&chroma, c"u_coeffs"),
                loc(&chroma, c"v_coeffs"),
                loc(&chroma, c"c_offset"),
                loc(&chroma, c"chroma_row"),
            ],
            fill_uv: loc(&fill, c"uv"),
            luma,
            chroma,
            fill,
        })
    }
}

/// The NV12 programs of a processor, built on its first NV12 destination.
enum Nv12ProgramState {
    /// Not built yet.
    Unbuilt,
    Built(Nv12Programs),
    /// The build failed with this message; NV12 destinations are declined.
    Failed(String),
}

/// How the most recent packed-`Rgb` DMA-BUF source reached the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RgbSourcePath {
    /// Imported as `BG24`, one RGB texel per pixel, filtered by the texture
    /// unit.
    Native,
    /// Imported as an R8 view and assembled by the RGB program named by the
    /// sampling ([`NvSampling::Single`], `Shader` or `Hardware`).
    R8Import(NvSampling),
}

/// OpenGL single-threaded image converter.
pub struct GLProcessorST {
    camera_eglimage_texture: Texture,
    camera_normal_texture: Texture,
    render_texture: Texture,
    segmentation_texture: Texture,
    segmentation_program: GlProgram,
    instanced_segmentation_program: GlProgram,
    proto_texture: Texture,
    proto_segmentation_program: GlProgram,
    proto_segmentation_int8_nearest_program: GlProgram,
    proto_segmentation_int8_bilinear_program: GlProgram,
    proto_dequant_int8_program: GlProgram,
    proto_segmentation_f32_program: GlProgram,
    color_program: GlProgram,
    /// Bounding-box outline thickness in destination pixels; never 0.
    box_line_width: f32,
    /// Last opacity value set on shader uniforms.
    cached_opacity: f32,
    /// Allocated proto texture dimensions: (w, h, layers, internal_fmt).
    proto_tex_dims: (usize, usize, usize, u32),
    /// Whether GL_OES_texture_float_linear is available (allows GL_LINEAR on R32F textures).
    has_float_linear: bool,
    /// Whether GL_EXT_texture_format_BGRA8888 is available (allows BGRA destinations).
    pub(super) has_bgra: bool,
    /// Whether `GL_EXT_color_buffer_float` is advertised in the extension
    /// string: the GPU can render to an F32 color attachment.
    pub(super) supports_f32_color: bool,
    /// Whether `GL_EXT_color_buffer_half_float` is advertised: the GPU can
    /// render to an F16 color attachment.
    pub(super) supports_f16_color: bool,
    /// Interpolation mode for int8 proto textures.
    int8_interpolation_mode: Int8InterpolationMode,
    /// Intermediate FBO texture for two-pass int8 dequant path.
    proto_dequant_texture: Texture,
    /// Allocated dequant texture dims: (w, h, layers, internal_fmt).
    proto_dequant_tex_dims: (usize, usize, usize, u32),
    /// Persistent FBO for the two-pass int8 dequant render.
    proto_dequant_fbo: FrameBuffer,
    /// Per-detection quad uniform locations, `(program_id, mask_coeff,
    /// class_index)`, re-resolved when the bound proto program changes;
    /// id 0 = unresolved.
    proto_quad_locs: std::cell::Cell<(u32, i32, i32)>,
    /// Compute shader program for HWC→CHW proto repack (GLES 3.1 only).
    proto_repack_compute_program: Option<u32>,
    /// `(width, height, num_protos)` uniform locations for the compute
    /// program.
    proto_compute_locs: (i32, i32, i32),
    /// SSBO for proto data upload (compute shader path).
    proto_ssbo: u32,
    /// Current allocated size of proto SSBO in bytes (0 = not allocated).
    proto_ssbo_size: usize,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    /// Persistent FBO for the convert() render path.
    convert_fbo: FrameBuffer,
    /// Persistent FBO for the draw_decoded_masks / draw_proto_masks render path.
    draw_fbo: FrameBuffer,
    /// Texture used as FBO color attachment for draw operations (DMA path).
    draw_render_texture: Texture,
    /// EGLImage cache for source DMA buffers.
    src_egl_cache: ImportCache<PlatformImport>,
    /// EGLImage cache for destination DMA buffers.
    dst_egl_cache: ImportCache<PlatformImport>,
    /// Whether the BGRA byte-swap warning has been logged.
    bgra_warned: bool,
    /// Reusable RGBA staging buffer for `read_pixels_into`'s indirect read.
    /// Grown on demand.
    readback_scratch: Vec<u8>,
    /// Test hook: make every zero-copy destination import fail
    /// (`get_or_create_egl_image` for `CacheKind::Dst` and
    /// `get_or_create_egl_image_rgb`).
    #[cfg(test)]
    pub(super) fail_dst_import: bool,
    /// Test hook: make the NV12 program build fail.
    #[cfg(test)]
    pub(super) fail_nv12_programs: bool,
    /// While set, a failed zero-copy destination import propagates instead of
    /// falling back to the mapped-texture readback. Set by
    /// `verify_dma_buf_roundtrip`.
    probe_dst_import_only: bool,
    /// Buffers whose destination import has already been reported as declined
    /// into the mapped-texture readback: the first occurrence warns, repeats
    /// log at debug.
    dst_import_warned: std::collections::HashSet<u64>,
    /// Whether the GPU is a Verisilicon/Vivante core (detected via GL_RENDERER).
    pub(super) is_vivante: bool,
    /// Whether the GPU is an Arm Mali core (GL_RENDERER).
    is_mali: bool,
    /// Whether the GPU is a virtualized/paravirtual device (GL_RENDERER).
    is_virtual_gpu: bool,
    /// Whether the GL implementation is ANGLE (GLES over Metal/D3D).
    is_angle: bool,
    /// Whether the GPU is a Qualcomm Adreno on the proprietary driver
    /// (GL_RENDERER); see [`RendererTraits::adreno`].
    pub(super) is_adreno: bool,
    /// The NV12 destinations this GPU renders.
    nv12_policy: Nv12DstPolicy,
    /// Whether [`DeclineClass::Soft`] declines apply.
    pub(super) soft_declines: bool,
    /// `GL_RENDERER`, or `"unknown"` when the driver reports none.
    renderer_name: String,
    /// Whether to use renderbuffer-backed EGLImages for DMA destinations.
    /// Enabled by `EDGEFIRST_OPENGL_RENDERSURFACE=1`; defaults to `false`
    /// (texture path).
    use_renderbuffer: bool,
    /// Intermediate RGBA texture for two-pass packed RGB conversion.
    /// Pass 1 renders YUYV/NV12→RGBA here; Pass 2 packs RGBA→RGB to DMA dest.
    packed_rgb_intermediate_tex: Texture,
    /// FBO for pass 1 of packed RGB conversion (renders to intermediate texture).
    packed_rgb_fbo: FrameBuffer,
    /// Current allocated size of the intermediate texture (0,0 = unallocated).
    packed_rgb_intermediate_size: (usize, usize),
    texture_program: GlProgram,
    /// External-OES sampler programs (`None` where the platform lacks
    /// `GL_OES_EGL_image_external_essl3`).
    texture_program_yuv: Option<GlProgram>,
    /// Int8 variant of texture_program — applies XOR 0x80 bias in fragment shader.
    texture_int8_program: GlProgram,
    /// Int8 variant of texture_program_yuv — applies XOR 0x80 bias in fragment shader.
    texture_int8_program_yuv: Option<GlProgram>,
    texture_program_planar: Option<GlProgram>,
    /// Shader: planar RGB with int8 bias (XOR 0x80) applied to output.
    texture_program_planar_int8: Option<GlProgram>,
    /// YUYV (RG-sampled) → RGBA, portable `sampler2D`: an IOSurface attach
    /// on macOS, and an upload from host memory or a PBO everywhere.
    yuyv_program_2d: GlProgram,
    /// `yuyv_program_2d` plus the int8 XOR-0x80 bias.
    yuyv_int8_program_2d: GlProgram,
    /// Link-time uniform locations for `yuyv_program_2d` and
    /// `yuyv_int8_program_2d`, in that order
    /// (src_size, y_offset, y_scale, c_vr, c_ug, c_vg, c_ub, src_extent).
    yuyv_2d_locs: [[i32; 8]; 2],
    /// Link-time `src_extent` locations for `texture_program` and
    /// `texture_int8_program`, the `sampler2D` source programs
    /// `draw_src_texture` selects between;
    /// see [`Self::texture_src_extent_loc`].
    texture_src_extent_locs: [i32; 2],
    /// Link-time `src_extent` locations for the four external-OES programs,
    /// in the order `[yuv, yuv_int8, planar, planar_int8]`. `-1` on a
    /// platform without external OES.
    external_src_extent_locs: [i32; 4],
    /// Shader: packed RGB -> RGBA8 packing (2D texture source, pass 2).
    packed_rgba8_program_2d: GlProgram,
    /// Shader: packed RGB int8 -> RGBA8 packing with XOR 0x80 (2D texture source, pass 2).
    packed_rgba8_int8_program_2d: GlProgram,
    /// Shader: planar RGB from 2D texture (pass 2 of the two-pass planar convert).
    texture_program_planar_2d: GlProgram,
    /// Shader: planar RGB int8 from 2D texture (pass 2 of the two-pass planar
    /// convert).
    texture_program_planar_int8_2d: GlProgram,
    /// Shader: RGBA8 → R32F-wide F32 NHWC `[H,W,3]` packed render (PBO float path).
    float_f32_nhwc_program: GlProgram,
    /// Shader: RGBA8 → four-elements-per-texel packed NCHW `[C,H,W]` render.
    /// Serves the F16 PBO path and the F16/F32 zero-copy planar paths.
    float_f16_nchw_program: GlProgram,
    /// Shader: RGBA8 → four-elements-per-texel packed NHWC `[H,W,3]` render
    /// (F16/F32 zero-copy interleaved path).
    float_nhwc_pack_program: GlProgram,
    /// Shader: RGBA8 → float `Rgba` render, one texel per pixel (F16/F32
    /// zero-copy RGBA path).
    float_rgba_program: GlProgram,
    /// Float render target texture (R32F or RGBA16F), reattached to `convert_fbo`.
    float_render_texture: Texture,
    /// Cached storage spec of `float_render_texture`: `(packed_w, packed_h,
    /// internal_format)`. `(0, 0, 0)` means unallocated.
    float_render_tex_dims: (usize, usize, u32),
    /// NV12/NV16/NV24 R8 → RGBA8 shaders, indexed by
    /// [`NvSampling::program`]: one per sampling, u8 or int8 (XOR 0x80 bias)
    /// output.
    nv_r8_programs: [GlProgram; 6],
    /// Link-time uniform locations and last uploaded colorimetry, per program.
    nv_r8_uniforms: [NvUniformState; 6],
    /// The chroma plane of an NV source as (U, V) texels, for
    /// [`NvSampling::Hardware`]: an upload, or a view of the source's
    /// DMA-BUF.
    nv_uv_texture: Texture,
    /// Chroma-plane imports for `nv_uv_texture`, keyed like `nv_r8_egl_cache`.
    nv_uv_egl_cache: ImportCache<PlatformImport>,
    /// Y2Y NV programs (u8, int8) and their uniforms, where
    /// `GL_EXT_YUV_target` is available ([`NvSampling::HardwareYuv`]).
    nv_y2y_programs: Option<([GlProgram; 2], [NvUniformState; 2])>,
    /// Set once the driver refuses a chroma-plane import.
    nv_uv_import_refused: bool,
    /// Packed-RGB shaders: an R8 view of a packed `Rgb` source → RGBA8,
    /// indexed by [`NvSampling::program`] like `nv_r8_programs` (no
    /// `HardwareYuv`).
    rgb_r8_programs: [GlProgram; 6],
    rgb_r8_locs: [PackedR8Locs; 6],
    /// YUYV luma for a Grey destination from the R8 view, indexed like
    /// `nv_r8_programs`.
    yuyv_luma_r8_programs: [GlProgram; 6],
    yuyv_luma_r8_locs: [PackedR8Locs; 6],
    /// RGBA -> Grey (BT.601 luma), pass 2 of `ConvertPlan::TwoPassGrey`.
    luma_2d_program: GlProgram,
    /// `luma_2d_program` plus the int8 bias.
    luma_2d_int8_program: GlProgram,
    /// Passes 2 and 3 of `ConvertPlan::ThreePassNv12`.
    nv12_programs: Nv12ProgramState,
    /// Render target for the chroma plane of a split NV12 destination.
    nv12_chroma_texture: Texture,
    /// Whether a packed `Rgb` DMA-BUF source imports natively (`BG24`);
    /// otherwise it takes the R8 view.
    rgb_native_import: bool,
    /// Whether a packed `Rgb` or `Yuyv` DMA-BUF source may be imported as an
    /// R8 view ([`GlPlatform::PACKED_R8_IMPORT`]).
    packed_r8_import: bool,
    /// How the most recent packed-`Rgb` DMA-BUF source reached the GPU.
    pub(super) last_rgb_source_path: Option<RgbSourcePath>,
    /// Texture for the R8 EGLImage source (TEXTURE_2D, not EXTERNAL_OES): an
    /// NV combined plane or a packed `Rgb` buffer.
    nv_r8_texture: Texture,
    /// EGLImage cache for R8 source imports (keyed like src_egl_cache).
    nv_r8_egl_cache: ImportCache<PlatformImport>,
    /// Which path ran for the most recent NV* convert.
    pub(super) last_nv_convert_path: NvConvertPath,
    /// How the last `nv_r8` draw sampled its source.
    pub(super) last_nv_sampling: Option<NvSampling>,
    /// Zero-copy feed telemetry (see `ConvertStats`).
    pub(super) convert_stats: super::cache::ConvertStats,
    /// NV import failures already warned about (keyed by buffer identity):
    /// the first occurrence warns, repeats log at debug.
    nv_import_warned: std::collections::HashSet<u64>,
    /// The PBOs this processor's context allocated; see [`OwnedPbos`].
    /// Shared with the `GLProcessorThreaded` front end, which records each
    /// PBO as it hands it out.
    pub(super) owned_pbos: std::sync::Arc<OwnedPbos>,
    /// Uniform locations per float-path program (see [`FloatQuadLocs`]).
    pub(super) float_quad_locs: std::collections::HashMap<u32, FloatQuadLocs>,
    /// Output mapping applied by `draw_float_quad` for the current convert.
    pub(super) float_norm: FloatNorm,
    /// The mapped int8 programs, with their `q_*` uniform locations, and the
    /// uniforms they hold. Empty until the mapped build links.
    int8_programs: Vec<(u32, [i32; 6])>,
    int_quant: Option<IntQuant>,
    /// Whether the int8 fields hold the mapped build.
    int8_mapped: bool,
    /// The int8 build not in use, once both exist.
    int8_alt: Option<Int8Set>,
    /// Letterbox clear bytes for the current integer convert, when mapped.
    int_pad: Option<[u8; 4]>,
    /// Whether that pad is an output-domain value, which a two-pass route
    /// (pixel pad in pass 1, mapped in pass 2) cannot write.
    int_pad_output: bool,
    /// Static full-screen quad VBOs `(pos, uv)` for the float paths,
    /// uploaded once (`STATIC_DRAW`). `0` = not yet created. Deleted in `Drop`.
    pub(super) float_quad_pos_vbo: u32,
    pub(super) float_quad_uv_vbo: u32,
    /// Client preference for NV* path selection (`EDGEFIRST_NV_CONVERT_PATH`).
    nv_path_pref: NvPathPref,
    /// Colorimetry/performance trade-off (see [`crate::ColorimetryMode`]).
    colorimetry_mode: crate::ColorimetryMode,
    /// `true` when `EDGEFIRST_COLORIMETRY` pinned the mode for this
    /// processor's lifetime — `set_colorimetry_mode` then logs and keeps it.
    colorimetry_env_pinned: bool,
    /// How resizing converts resample (see [`crate::Interpolation`]): the
    /// source textures' filter and the NV program family.
    interpolation: crate::Interpolation,
    /// Whether a bilinear NV resize may filter through the texture units
    /// ([`NvSampling::Hardware`]). `EDGEFIRST_GL_NO_NV_HW_FILTER=1` clears it.
    nv_hw_filter: bool,
    /// When `true`, a convert's terminal `glFinish` is skipped. Set for the
    /// duration of a single [`convert_deferred`](Self::convert_deferred) and
    /// reset on every exit; `false` between calls.
    pub(super) defer_finish: bool,
    /// `true` when one or more `convert_deferred` calls have rendered without a
    /// `glFinish` and a [`flush`](Self::flush) is owed. Cleared by the flush;
    /// see [`flush_pending`](Self::flush_pending).
    pub(super) pending_flush: bool,
    /// `true` when a `record_completion` inside a batch queued a device fence
    /// signal without submitting the command buffer. Cleared by
    /// [`flush_pending`](Self::flush_pending).
    pub(super) pending_submit: bool,
    pub(super) gl_context: PlatformDisplay,
}

impl Drop for GLProcessorST {
    fn drop(&mut self) {
        // Submits the device signals a batch left queued.
        if self.pending_submit {
            Platform::submit_recorded(&self.gl_context);
            self.pending_submit = false;
        }
        unsafe {
            {
                if self.proto_ssbo != 0 {
                    edgefirst_gl::gl::DeleteBuffers(1, &self.proto_ssbo);
                }
                if self.float_quad_pos_vbo != 0 {
                    let ids = [self.float_quad_pos_vbo, self.float_quad_uv_vbo];
                    edgefirst_gl::gl::DeleteBuffers(2, ids.as_ptr());
                }
                if let Some(program) = self.proto_repack_compute_program {
                    edgefirst_gl::gl::DeleteProgram(program);
                }
            }
        }
    }
}

/// Emit a warning the first time a draw operation falls back from the
/// DMA-BUF path to the CPU readback; later fallbacks log at debug. The
/// message includes the call site, the failing setup function and the
/// underlying error.
fn warn_slow_path_once(call_site: &str, failing_setup: &str, err: &crate::Error) {
    use std::sync::Once;
    static SLOW_PATH_WARNED: Once = Once::new();
    let mut emitted = false;
    SLOW_PATH_WARNED.call_once(|| {
        log::warn!(
            "{call_site}: GL DMA-BUF fast path unavailable, falling back to CPU \
             readback (10–20× slower). Cause: {failing_setup} returned {err:?}. \
             On Mali Valhall (i.MX 95) this is usually because the destination \
             tensor's row pitch is not 64-byte aligned — see \
             `ImageProcessor::create_image` for automatic alignment. \
             Subsequent fallbacks will be logged at debug level."
        );
        emitted = true;
    });
    if !emitted {
        log::debug!("{call_site}: GL DMA-BUF fast path unavailable ({failing_setup}: {err:?})");
    }
}

/// Reinterpret a `&mut Tensor<i8>` as `&mut Tensor<u8>`.
///
/// # Safety
/// `i8` and `u8` have identical size, alignment, and validity for all bit
/// patterns, and the `T` parameter does not affect the layout of
/// `Tensor<T>`. The returned reference must not be used to access
/// `chroma()`.
unsafe fn tensor_i8_as_u8_mut(t: &mut Tensor<i8>) -> &mut Tensor<u8> {
    unsafe { &mut *(t as *mut Tensor<i8> as *mut Tensor<u8>) }
}

/// Reinterpret a `&Tensor<i8>` as `&Tensor<u8>`.
///
/// # Safety
/// As for [`tensor_i8_as_u8_mut`]. The returned reference must not be used
/// to access `chroma()`.
unsafe fn tensor_i8_as_u8(t: &Tensor<i8>) -> &Tensor<u8> {
    unsafe { &*(t as *const Tensor<i8> as *const Tensor<u8>) }
}

/// Extract `&Tensor<u8>` and `PixelFormat` from a `&TensorDyn` source.
/// For I8 sources, reinterprets the bytes as u8.
fn dyn_to_u8_src(src: &TensorDyn) -> crate::Result<(&Tensor<u8>, PixelFormat)> {
    match src.dtype() {
        DType::U8 => {
            let t = src.as_typed::<u8>().expect("dtype checked");
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            Ok((t, fmt))
        }
        DType::I8 => {
            let t = src.as_typed::<i8>().expect("dtype checked");
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            // SAFETY: i8/u8 are layout-identical
            Ok((unsafe { tensor_i8_as_u8(t) }, fmt))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "GL backend requires u8 or i8 source, got {:?}",
            src.dtype()
        ))),
    }
}

/// Extract `&mut Tensor<u8>`, `PixelFormat`, and `is_int8` from a `&mut TensorDyn` destination.
/// For I8 destinations, reinterprets the bytes as u8 and sets `is_int8 = true`.
fn dyn_to_u8_dst(dst: &mut TensorDyn) -> crate::Result<(&mut Tensor<u8>, PixelFormat, bool)> {
    match dst.dtype() {
        DType::U8 => {
            let t = dst.as_typed_mut::<u8>().expect("dtype checked");
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            Ok((t, fmt, false))
        }
        DType::I8 => {
            let t = dst.as_typed_mut::<i8>().expect("dtype checked");
            let fmt = t.format().ok_or(Error::NotAnImage)?;
            // SAFETY: i8/u8 are layout-identical
            Ok((unsafe { tensor_i8_as_u8_mut(t) }, fmt, true))
        }
        _ => Err(Error::UnsupportedFormat(format!(
            "GL backend requires u8 or i8 destination, got {:?}",
            dst.dtype()
        ))),
    }
}

/// Refuse a `view()`/`batch()` draw destination, or one carrying a
/// `plane_offset`.
fn check_draw_dst_placement(dst: &TensorDyn) -> crate::Result<()> {
    let is_view = dst.view_origin().is_some();
    let plane_offset = dst.plane_offset().unwrap_or(0);
    if is_view || plane_offset != 0 {
        return Err(Error::NotSupported(format!(
            "GL mask draw places its frame at the destination surface's \
             origin -- viewport 0,0 on an imported surface, buffer offset 0 \
             in a readback -- so it cannot fill a sub-region destination \
             (view={is_view}, plane_offset={plane_offset}); CPU fallback \
             handles it"
        )));
    }
    Ok(())
}

impl ImageProcessorTrait for GLProcessorST {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<crate::ConvertOptions>,
    ) -> crate::Result<()> {
        self.convert_inner(src, dst, &options.unwrap_or_default())?;
        crate::colorimetry::tag_yuv_destination(dst);
        // Records completion after every convert; inside a batch the device
        // submission is left to `flush_pending`.
        let submit = !self.defer_finish;
        if Platform::record_completion(&self.gl_context, dst, submit).is_some() && !submit {
            self.pending_submit = true;
        }
        Ok(())
    }

    fn convert_deferred(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: Option<crate::ConvertOptions>,
    ) -> Result<(), crate::Error> {
        // Renders with the finish deferred and resets the flag on every exit;
        // a successful render sets `pending_flush`.
        self.defer_finish = true;
        let result = self.convert(src, dst, options);
        self.defer_finish = false;
        if result.is_ok() {
            self.pending_flush = true;
        } else if self.pending_submit {
            // Submits the device signals earlier tiles of this batch left queued.
            Platform::submit_recorded(&self.gl_context);
            self.pending_submit = false;
        }
        result
    }

    fn flush(&mut self) -> Result<(), crate::Error> {
        let _span = tracing::trace_span!("image.flush.gl", pending = self.pending_flush).entered();
        self.flush_pending();
        Ok(())
    }

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<(), crate::Error> {
        check_draw_dst_placement(dst)?;
        let bg = overlay.background.map(|bg| dyn_to_u8_src(bg)).transpose()?;
        {
            let (dst_u8, dst_fmt, _is_int8) = dyn_to_u8_dst(dst)?;
            self.draw_decoded_masks_impl(
                dst_u8,
                dst_fmt,
                detect,
                segmentation,
                overlay.opacity,
                bg,
                overlay.color_mode,
            )?;
        }
        Platform::record_completion(&self.gl_context, dst, true);
        Ok(())
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: crate::MaskOverlay<'_>,
    ) -> crate::Result<()> {
        check_draw_dst_placement(dst)?;
        let bg = overlay.background.map(|bg| dyn_to_u8_src(bg)).transpose()?;
        {
            let (dst_u8, dst_fmt, _is_int8) = dyn_to_u8_dst(dst)?;
            self.draw_proto_masks_impl(
                dst_u8,
                dst_fmt,
                detect,
                proto_data,
                overlay.opacity,
                bg,
                overlay.color_mode,
            )?;
        }
        Platform::record_completion(&self.gl_context, dst, true);
        Ok(())
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> crate::Result<()> {
        if colors.is_empty() {
            return Ok(());
        }
        let colors_f32 = colors
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            })
            .take(20)
            .collect::<Vec<[f32; 4]>>();

        self.segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.instanced_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_nearest_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_bilinear_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_f32_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.color_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        Ok(())
    }

    fn set_box_line_width(&mut self, width: u32) -> crate::Result<()> {
        crate::check_box_line_width(width)?;
        self.box_line_width = width as f32;
        Ok(())
    }
}

/// Whether `EDGEFIRST_ALLOW_SOFTWARE_GL=1` is set, opting in to running the GL
/// backend on a software renderer (Mesa llvmpipe/softpipe/swrast).
fn software_gl_override_enabled() -> bool {
    std::env::var_os("EDGEFIRST_ALLOW_SOFTWARE_GL").is_some_and(|v| v == "1")
}

/// Per-cache import-cache capacity when neither the caller nor the
/// environment asks for one. See
/// [`ImageProcessorConfig::egl_cache_capacity`](crate::ImageProcessorConfig::egl_cache_capacity)
/// for what the number bounds.
const DEFAULT_EGL_CACHE_CAPACITY: usize = 16;

/// Resolve the import-cache capacity: config > environment > default.
///
/// `env` is the raw `EDGEFIRST_EGL_CACHE_CAPACITY` value. Zero and
/// unparseable values are rejected at both levels and the next source is
/// used.
fn resolve_egl_cache_capacity(config: Option<usize>, env: Option<&str>) -> usize {
    config
        .filter(|&c| c > 0)
        .or_else(|| env.and_then(|v| v.parse::<usize>().ok()).filter(|&c| c > 0))
        .unwrap_or(DEFAULT_EGL_CACHE_CAPACITY)
}

/// Whether to reject an initialized GL context for being a software
/// renderer: when the renderer is software and the override is not enabled.
fn should_reject_software_gl(is_software_renderer: bool, override_enabled: bool) -> bool {
    is_software_renderer && !override_enabled
}

fn parse_nv_path_pref() -> NvPathPref {
    match std::env::var("EDGEFIRST_NV_CONVERT_PATH") {
        Ok(v) => match v.to_ascii_lowercase().as_str() {
            "sampler" | "external" | "a" => NvPathPref::ForceSampler,
            "shader" | "r8" | "b" => NvPathPref::ForceShader,
            "auto" | "" => NvPathPref::Auto,
            other => {
                log::warn!(
                    "EDGEFIRST_NV_CONVERT_PATH={other:?} not recognised \
                     (expected sampler|shader|auto), using auto"
                );
                NvPathPref::Auto
            }
        },
        Err(_) => NvPathPref::Auto,
    }
}

fn parse_colorimetry_env() -> Option<crate::ColorimetryMode> {
    match std::env::var("EDGEFIRST_COLORIMETRY") {
        Ok(v) => match v.to_ascii_lowercase().as_str() {
            "exact" => Some(crate::ColorimetryMode::Exact),
            "fast" => Some(crate::ColorimetryMode::Fast),
            "" => None,
            other => {
                log::warn!(
                    "EDGEFIRST_COLORIMETRY={other:?} not recognised \
                     (expected fast|exact), ignoring"
                );
                None
            }
        },
        Err(_) => None,
    }
}

fn apply_forced_transfer(backend: &mut TransferBackend) {
    let Ok(val) = std::env::var("EDGEFIRST_FORCE_TRANSFER") else {
        return;
    };
    let forced = match val.to_ascii_lowercase().as_str() {
        "dmabuf" | "dma" => Some(TransferBackend::DmaBuf),
        "pbo" => Some(TransferBackend::Pbo),
        "sync" => Some(TransferBackend::Sync),
        other => {
            log::warn!(
                "EDGEFIRST_FORCE_TRANSFER={other:?} not recognised \
                 (expected dmabuf|pbo|sync), ignoring"
            );
            None
        }
    };
    if let Some(forced) = forced {
        log::info!(
            "EDGEFIRST_FORCE_TRANSFER override: {:?} → {forced:?}",
            backend
        );
        *backend = forced;
    }
}

/// GL_RENDERER-derived driver traits that feed per-driver policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RendererTraits {
    /// Verisilicon/Vivante core.
    vivante: bool,
    /// Arm Mali.
    mali: bool,
    /// Software rasterizer (llvmpipe/softpipe/swrast).
    software: bool,
    /// Virtualized/paravirtual GPU.
    virtual_gpu: bool,
    /// ANGLE, the GLES-over-Metal/D3D translation layer.
    angle: bool,
    /// Qualcomm Adreno on the proprietary KGSL stack.
    adreno: bool,
}

/// Whether this driver needs per-message global serialization
/// (`SerializationPolicy::Full`) rather than the parallel `LifecycleOnly`
/// policy: Vivante, virtualized GPUs, ANGLE and Adreno.
fn requires_full_serialization(t: RendererTraits) -> bool {
    t.vivante || t.virtual_gpu || t.angle || t.adreno
}

/// Capability probe results from a freshly-current GL context
/// (`gl_check_support`): extension-derived feature flags plus the
/// GL_RENDERER-derived driver traits.
struct GlSupport {
    has_float_linear: bool,
    /// `GL_EXT_YUV_target` on a platform with external-OES imports: the Y2Y
    /// NV programs can sample raw, filtered YUV from an NV12 import.
    has_yuv_target: bool,
    has_bgra: bool,
    renderer: RendererTraits,
    /// `GL_RENDERER`, or `"unknown"` when the driver reports none.
    renderer_name: String,
    supports_f32_color: bool,
    supports_f16_color: bool,
}

/// Whether `glReadPixels(format, UNSIGNED_BYTE)` is accepted on the bound
/// read framebuffer: `RGBA` always is; anything else only when it is the
/// implementation-defined pair (ES 3.0 § 4.3.2).
///
/// # Safety
/// Must run on the GL thread with a complete read framebuffer bound.
unsafe fn direct_read_supported(format: u32) -> bool {
    if format == edgefirst_gl::gl::RGBA {
        return true;
    }
    let mut impl_fmt = 0i32;
    let mut impl_type = 0i32;
    // SAFETY: two GetIntegerv queries on a current context.
    unsafe {
        edgefirst_gl::gl::GetIntegerv(
            edgefirst_gl::gl::IMPLEMENTATION_COLOR_READ_FORMAT,
            &mut impl_fmt,
        );
        edgefirst_gl::gl::GetIntegerv(
            edgefirst_gl::gl::IMPLEMENTATION_COLOR_READ_TYPE,
            &mut impl_type,
        );
    }
    impl_fmt as u32 == format && impl_type as u32 == edgefirst_gl::gl::UNSIGNED_BYTE
}

/// Bytes `glReadPixels` writes per pixel for a pack format.
fn pack_channels(format: u32) -> usize {
    match format {
        edgefirst_gl::gl::RGB => 3,
        edgefirst_gl::gl::RED => 1,
        _ => 4,
    }
}

/// Re-space the `rows` tightly packed rows of `tight_row` bytes at the head of
/// `buf` so that row `y` starts at `y * stride`.
///
/// Rows move back to front. `buf` must hold
/// `(rows - 1) * stride + tight_row` bytes; see [`padded_readback_bytes`].
fn spread_rows(buf: &mut [u8], rows: usize, tight_row: usize, stride: usize) {
    if stride <= tight_row || rows < 2 || tight_row == 0 {
        return;
    }
    for y in (1..rows).rev() {
        buf.copy_within(y * tight_row..y * tight_row + tight_row, y * stride);
    }
}

/// Bytes a destination must map to hold `rows` rows of `tight_row` bytes at
/// `stride`, or `None` when the destination is tight.
fn padded_readback_bytes(rows: usize, tight_row: usize, stride: usize) -> Option<usize> {
    if stride <= tight_row || rows == 0 {
        return None;
    }
    Some((rows - 1) * stride + tight_row)
}

/// `GL_PACK_ROW_LENGTH` that lays rows out at a `stride`-byte pitch, or `None`
/// when the pitch is tight or is not a whole number of `pixel_bytes`-byte
/// pixels.
///
/// `pixel_bytes` is what one pixel of the pack format occupies: the channel
/// count for the u8 readback, and the texel size for the float one (4 for
/// `RED`/`FLOAT`, 8 for `RGBA`/`HALF_FLOAT`). Assumes `PACK_ALIGNMENT` is 1.
fn pack_row_length(tight_row: usize, stride: usize, pixel_bytes: usize) -> Option<i32> {
    if stride <= tight_row || pixel_bytes == 0 || !stride.is_multiple_of(pixel_bytes) {
        return None;
    }
    i32::try_from(stride / pixel_bytes).ok()
}

/// `GL_UNPACK_ROW_LENGTH` for a source whose rows sit `stride` bytes apart, or
/// `0` for a tight source. Returns an error for a pitch that is not a whole
/// number of pixels.
///
/// `pixel_bytes` is what one pixel of the unpack format occupies (the
/// channel count for the u8 paths, 4 for the RGBA float PBO arm).
/// `what` names the format in the refusal. Assumes `UNPACK_ALIGNMENT` is 1.
fn unpack_row_length(
    stride: Option<usize>,
    tight_px: usize,
    pixel_bytes: usize,
    what: &str,
) -> Result<i32, Error> {
    let Some(stride) = stride else {
        return Ok(0);
    };
    if pixel_bytes == 0 || !stride.is_multiple_of(pixel_bytes) {
        return Err(Error::NotSupported(format!(
            "source row pitch {stride} B is not a whole number of \
             {what} pixels ({pixel_bytes} B/px); upload would \
             shear rows"
        )));
    }
    let px = stride / pixel_bytes;
    if px == tight_px {
        return Ok(0);
    }
    i32::try_from(px).map_err(|_| {
        Error::NotSupported(format!(
            "source row pitch {stride} B is {px} {what} pixels, past what \
             GL_UNPACK_ROW_LENGTH can express"
        ))
    })
}

/// The PBOs one GL context allocated, recorded by the address of their
/// `PboOpsVtable`. Each record also holds a `Weak` of the PBO's own ops;
/// an address matches only while that allocation is alive.
#[derive(Default)]
pub(crate) struct OwnedPbos(std::sync::Mutex<Vec<OwnedPbo>>);

struct OwnedPbo {
    vtable: usize,
    ops: std::sync::Weak<dyn edgefirst_tensor::PboOps>,
}

impl std::fmt::Debug for OwnedPbos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let owned = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f.debug_tuple("OwnedPbos").field(&owned.len()).finish()
    }
}

impl OwnedPbos {
    /// Record a PBO this context allocated. `ops` must be the operations
    /// created for that one allocation.
    pub(crate) fn insert(
        &self,
        vtable: Option<*const std::ffi::c_void>,
        ops: &std::sync::Arc<dyn edgefirst_tensor::PboOps>,
    ) {
        let Some(vtable) = vtable else { return };
        let mut owned = self.0.lock().unwrap_or_else(|e| e.into_inner());
        owned.retain(|p| p.ops.strong_count() > 0);
        owned.push(OwnedPbo {
            vtable: vtable as usize,
            ops: std::sync::Arc::downgrade(ops),
        });
    }

    fn contains(&self, vtable: Option<*const std::ffi::c_void>) -> bool {
        let Some(vtable) = vtable else { return false };
        let owned = self.0.lock().unwrap_or_else(|e| e.into_inner());
        owned
            .iter()
            .any(|p| p.vtable == vtable as usize && p.ops.strong_count() > 0)
    }
}

/// A source's pixels opened for a texture upload: its PBO when this context
/// owns it, otherwise a CPU map held until the upload has read it.
///
/// An owned PBO is read through `PIXEL_UNPACK_BUFFER` at the source's
/// `plane_offset`. Any other PBO is mapped through its own operations
/// (see [`OwnedPbos`]).
enum SrcPixels<'a> {
    Pbo { buffer_id: u32, offset: usize },
    Mapped(edgefirst_tensor::HostView<'a, u8>),
}

impl SrcPixels<'_> {
    fn open(src: &Tensor<u8>, owned: &OwnedPbos) -> Result<Self, Error> {
        match src.pbo_id() {
            Some(buffer_id) if owned.contains(src.pbo_vtable_ptr()) => {
                if src.pbo_is_mapped() == Some(true) {
                    return Err(Error::OpenGl(
                        "Cannot convert from a mapped PBO tensor".to_string(),
                    ));
                }
                Ok(Self::Pbo {
                    buffer_id,
                    offset: src.plane_offset().unwrap_or(0),
                })
            }
            _ => Ok(Self::Mapped(src.map_read()?)),
        }
    }

    fn source(&self) -> UploadSource<'_> {
        match self {
            Self::Pbo { buffer_id, offset } => UploadSource::Pbo {
                buffer_id: *buffer_id,
                offset: *offset,
            },
            Self::Mapped(map) => UploadSource::Bytes(map.as_slice()),
        }
    }

    /// Count this source feed and name it on the convert's span.
    fn record(&self, stats: &mut super::cache::ConvertStats, span: &tracing::Span) {
        match self {
            Self::Pbo { .. } => {
                stats.src_pbo_uploads += 1;
                span.record("src_feed", "pbo");
            }
            Self::Mapped(_) => {
                stats.src_uploads += 1;
                span.record("src_feed", "upload");
            }
        }
    }
}

/// What a PBO readback knows about its destination before it issues the
/// `glReadPixels`: the pitch to place the rows at, the bytes they occupy,
/// and the `GL_PACK_ROW_LENGTH` that expresses the pitch. Produced by
/// [`plan_pbo_readback`].
struct PboReadbackLayout {
    /// The destination's row pitch in bytes.
    stride: usize,
    /// Bytes of one row that are pixels.
    tight_row: usize,
    /// Bytes the `rows` rows occupy at `stride`, which the PBO must hold.
    needed: usize,
    /// `GL_PACK_ROW_LENGTH` in pixels, or `None` for read-tight-and-spread.
    row_length: Option<i32>,
}

/// Reconcile a PBO destination's declared pitch with the tight rows
/// `glReadPixels` produces, or refuse the destination.
///
/// Refused:
///
/// * A pitch shorter than one row.
/// * A `view()`/`batch()` destination, or one carrying a `plane_offset`.
/// * A PBO too small for the rows at that pitch.
fn plan_pbo_readback(
    stride: usize,
    is_view: bool,
    plane_offset: usize,
    capacity: usize,
    width: usize,
    rows: usize,
    pixel_bytes: usize,
) -> crate::Result<PboReadbackLayout> {
    let tight_row = width * pixel_bytes;
    if stride < tight_row {
        return Err(crate::Error::OpenGl(format!(
            "destination row pitch is {stride} B but one {width}-pixel row \
             of this readback format is {tight_row} B"
        )));
    }
    if is_view || plane_offset != 0 {
        return Err(crate::Error::NotSupported(format!(
            "GL PBO readback writes at buffer offset 0, so it cannot fill a \
             sub-region destination (view={is_view}, plane_offset={plane_offset}); \
             CPU fallback handles it"
        )));
    }
    let needed = padded_readback_bytes(rows, tight_row, stride).unwrap_or(rows * tight_row);
    if capacity < needed {
        return Err(crate::Error::OpenGl(format!(
            "PBO destination is {capacity} B but the readback needs {needed} B \
             at a {stride} B pitch"
        )));
    }
    Ok(PboReadbackLayout {
        stride,
        tight_row,
        needed,
        row_length: pack_row_length(tight_row, stride, pixel_bytes),
    })
}

/// Map the first `len` bytes of PBO `buffer_id` with `access` bits for the
/// duration of `f`, then unmap and unbind.
///
/// # Safety
///
/// Must run on the GL thread while nothing else holds `buffer_id` mapped, and
/// `len` must not exceed the buffer's allocation.
unsafe fn with_mapped_pbo(
    buffer_id: u32,
    len: usize,
    access: u32,
    what: &str,
    f: impl FnOnce(&mut [u8]),
) -> crate::Result<()> {
    // SAFETY: the caller's contract -- GL thread, `buffer_id` not already
    // mapped, `len` within the allocation -- so `MapBufferRange` returns a
    // pointer to `len` writable bytes, valid until the `UnmapBuffer` below.
    let ok = unsafe {
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, buffer_id);
        let ptr = edgefirst_gl::gl::MapBufferRange(
            edgefirst_gl::gl::PIXEL_PACK_BUFFER,
            0,
            len as isize,
            access,
        );
        if ptr.is_null() {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
            return Err(crate::Error::OpenGl(format!(
                "glMapBufferRange returned null for the {what}"
            )));
        }
        f(std::slice::from_raw_parts_mut(ptr as *mut u8, len));
        let ok = edgefirst_gl::gl::UnmapBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER);
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
        ok
    };
    // GL_FALSE from glUnmapBuffer means the contents were lost while mapped.
    if ok == edgefirst_gl::gl::FALSE {
        return Err(crate::Error::OpenGl(format!(
            "PBO data was corrupted during mapping ({what})"
        )));
    }
    Ok(())
}

/// `glReadPixels` a `w`x`h` RGBA frame into `scratch`, growing it to
/// `w * h * 4` bytes first. `scratch` is not shrunk.
///
/// # Safety
/// Must run on the GL thread with a complete read framebuffer bound,
/// `ReadBuffer` selected and no `PIXEL_PACK_BUFFER` bound.
unsafe fn read_pixels_rgba_scratch(w: usize, h: usize, scratch: &mut Vec<u8>) {
    if scratch.len() < w * h * 4 {
        scratch.resize(w * h * 4, 0);
    }
    // SAFETY: the caller's contract, and `scratch` holds `w * h * 4` bytes,
    // which is what an RGBA/UNSIGNED_BYTE read of this frame writes at
    // `PACK_ALIGNMENT` 1.
    unsafe {
        edgefirst_gl::gl::ReadPixels(
            0,
            0,
            w as i32,
            h as i32,
            edgefirst_gl::gl::RGBA,
            edgefirst_gl::gl::UNSIGNED_BYTE,
            scratch.as_mut_ptr() as *mut c_void,
        );
    }
}

/// `glReadPixels` the bound read-framebuffer into `out`. When the direct
/// read of `format` is unsupported, reads RGBA into `scratch` and repacks
/// the leading channels.
///
/// `scratch` is the caller's reusable staging buffer
/// (`GLProcessorST::readback_scratch`), grown on demand.
///
/// `stride` is how far apart `out`'s rows sit; only the first
/// `w * channels` bytes of each row are written. Where the pitch is a
/// whole number of pixels GL lays the rows out through
/// `GL_PACK_ROW_LENGTH`; where it is not, the frame is read tight into
/// `scratch` and the rows are copied out one at a time.
///
/// # Safety
/// Must run on the GL thread with a complete read framebuffer bound and
/// `ReadBuffer` selected, and `out` must hold `(h - 1) * stride + w *
/// channels` bytes for the pack format.
unsafe fn read_pixels_into(
    w: usize,
    h: usize,
    format: u32,
    scratch: &mut Vec<u8>,
    out: &mut [u8],
    stride: usize,
) {
    let channels = pack_channels(format);
    let tight_row = w * channels;
    unsafe {
        if direct_read_supported(format) {
            if stride == tight_row {
                edgefirst_gl::gl::ReadPixels(
                    0,
                    0,
                    w as i32,
                    h as i32,
                    format,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    out.as_mut_ptr() as *mut c_void,
                );
                return;
            }
            if let Some(px) = pack_row_length(tight_row, stride, channels) {
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, px);
                edgefirst_gl::gl::ReadPixels(
                    0,
                    0,
                    w as i32,
                    h as i32,
                    format,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    out.as_mut_ptr() as *mut c_void,
                );
                // Resets `PACK_ROW_LENGTH` to its default.
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, 0);
                return;
            }
            // A pitch `GL_PACK_ROW_LENGTH` cannot express: the tight frame is read
            // into the scratch and each row is copied to its place.
            if scratch.len() < h * tight_row {
                scratch.resize(h * tight_row, 0);
            }
            edgefirst_gl::gl::ReadPixels(
                0,
                0,
                w as i32,
                h as i32,
                format,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                scratch.as_mut_ptr() as *mut c_void,
            );
            for y in 0..h {
                out[y * stride..y * stride + tight_row]
                    .copy_from_slice(&scratch[y * tight_row..y * tight_row + tight_row]);
            }
            return;
        }
        read_pixels_rgba_scratch(w, h, scratch);
        // Row by row, at the destination's pitch.
        let rgba = scratch.as_chunks::<4>().0;
        for y in 0..h {
            for (px, dst_px) in rgba[y * w..]
                .iter()
                .take(w)
                .zip(out[y * stride..y * stride + tight_row].chunks_exact_mut(channels))
            {
                dst_px.copy_from_slice(&px[..channels]);
            }
        }
    }
}

/// Set the green and blue swizzles of the `TEXTURE_EXTERNAL_OES` texture
/// bound to the active unit.
///
/// # Safety
/// A GL context must be current with the external texture bound.
unsafe fn set_external_gb_swizzle(g: u32, b: u32) {
    for (swizzle, comp) in [
        (edgefirst_gl::gl::TEXTURE_SWIZZLE_G, g),
        (edgefirst_gl::gl::TEXTURE_SWIZZLE_B, b),
    ] {
        unsafe {
            edgefirst_gl::gl::TexParameteri(
                edgefirst_gl::gl::TEXTURE_EXTERNAL_OES,
                swizzle,
                comp as i32,
            )
        };
    }
}

/// Classify a GL_RENDERER string into driver-policy traits.
fn classify_renderer(renderer: &str) -> RendererTraits {
    let lower = renderer.to_ascii_lowercase();
    RendererTraits {
        vivante: lower.contains("vivante") || lower.contains("gc7000") || lower.contains("galcore"),
        mali: lower.contains("mali"),
        software: lower.contains("llvmpipe")
            || lower.contains("softpipe")
            || lower.contains("swrast")
            || lower.contains("software rasterizer")
            // D3D11 WARP.
            || lower.contains("basic render driver"),
        virtual_gpu: lower.contains("paravirtual") || lower.contains("virtio"),
        angle: lower.contains("angle"),
        adreno: lower.contains("adreno"),
    }
}

/// How a GL decline applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclineClass {
    /// Declines whether or not soft declines are on.
    Hard,
    /// Declines only while soft declines are on.
    Soft,
}

/// Same-size NV12 conversions with fewer destination pixels are soft declines on Adreno.
pub(super) const NV12_SAME_SIZE_MIN_PIXELS_ADRENO: usize = 1920 * 1080;

/// Same-size NV12 conversions with fewer destination pixels are soft declines on ANGLE.
pub(super) const NV12_SAME_SIZE_MIN_PIXELS_ANGLE: usize = 1600 * 900;

/// Which NV12 destinations a GPU renders; see [`nv12_dst_decline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Nv12DstPolicy {
    /// Whether the GPU renders NV12 destinations at all.
    pub gpu_renders: bool,
    /// Whether a YUYV or VYUY source is a hard decline.
    pub packed_yuv_src_broken: bool,
    /// A same-size conversion into fewer destination pixels is a soft decline.
    pub same_size_min_pixels: usize,
    /// Whether an exact same-size YUV copy is a soft decline.
    pub exact_copy_on_cpu: bool,
    /// Whether the GPU has a policy of its own; every conversion is a soft decline otherwise.
    pub measured: bool,
}

impl Nv12DstPolicy {
    /// The policy for a GPU with `traits`.
    fn for_renderer(traits: RendererTraits) -> Self {
        let other = Self {
            gpu_renders: true,
            packed_yuv_src_broken: false,
            same_size_min_pixels: usize::MAX,
            exact_copy_on_cpu: true,
            measured: false,
        };
        let measured = Self {
            measured: true,
            ..other
        };
        if traits.software {
            Self {
                gpu_renders: false,
                ..measured
            }
        } else if traits.vivante {
            Self {
                packed_yuv_src_broken: true,
                ..measured
            }
        } else if traits.mali {
            Self {
                same_size_min_pixels: 0,
                exact_copy_on_cpu: false,
                ..measured
            }
        } else if traits.adreno {
            Self {
                same_size_min_pixels: NV12_SAME_SIZE_MIN_PIXELS_ADRENO,
                ..measured
            }
        } else if traits.angle {
            Self {
                same_size_min_pixels: NV12_SAME_SIZE_MIN_PIXELS_ANGLE,
                ..measured
            }
        } else {
            other
        }
    }
}

/// How GL declines a `src_fmt` source into a `w`x`h` NV12 destination under `policy`, and why,
/// or `None` when it renders it. `same_size` is a conversion with no resize, crop, placement,
/// flip or rotation; `exact_copy` is a same-size one from a YUV source with the destination's
/// matrix and range. `renderer` is the GPU's `GL_RENDERER`.
pub(super) fn nv12_dst_decline(
    policy: Nv12DstPolicy,
    renderer: &str,
    src_fmt: PixelFormat,
    same_size: bool,
    exact_copy: bool,
    (w, h): (usize, usize),
) -> Option<(DeclineClass, String)> {
    let gpu =
        || format!("the GL backend does not render NV12 destinations on this GPU ({renderer})");
    if !Platform::NV12_DST || !policy.gpu_renders {
        return Some((DeclineClass::Hard, gpu()));
    }
    if policy.packed_yuv_src_broken && matches!(src_fmt, PixelFormat::Yuyv | PixelFormat::Vyuy) {
        return Some((
            DeclineClass::Hard,
            format!(
                "the GL backend does not render NV12 destinations from {src_fmt} on this GPU \
                 ({renderer})"
            ),
        ));
    }
    if !policy.measured {
        return Some((
            DeclineClass::Soft,
            format!(
                "the GL backend leaves NV12 destinations on this unmeasured GPU ({renderer}) to \
                 the CPU"
            ),
        ));
    }
    if exact_copy && policy.exact_copy_on_cpu {
        return Some((
            DeclineClass::Soft,
            format!("the GL backend leaves an exact same-size {src_fmt} copy to NV12 to the CPU"),
        ));
    }
    if same_size && w * h < policy.same_size_min_pixels {
        return Some((
            DeclineClass::Soft,
            format!(
                "the GL backend leaves a same-size {src_fmt} to NV12 conversion of {w}x{h} to \
                 the CPU"
            ),
        ));
    }
    None
}

/// The plane-offset alignment, in bytes, checked for a DMA-BUF import on
/// drivers that misplace unaligned offsets.
pub(super) const DMA_IMPORT_OFFSET_ALIGN: usize = 64;

/// Whether `plane_offset` is not [`DMA_IMPORT_OFFSET_ALIGN`]-byte aligned.
fn unaligned_import_offset(plane_offset: usize) -> bool {
    !plane_offset.is_multiple_of(DMA_IMPORT_OFFSET_ALIGN)
}

/// Whether a source import at `plane_offset` would be misread on a driver
/// that misplaces unaligned offsets. `misplaces` is
/// [`GLProcessorST::misplaces_unaligned_offsets`].
///
/// Checks one plane's offset; the caller checks every plane the import
/// passes to EGL (see [`nv12_plane1_offset`]). For a source, plane 0's
/// offset is [`source_import_plane0_offset`].
fn rejects_unaligned_import_offset(misplaces: bool, plane_offset: usize) -> bool {
    misplaces && unaligned_import_offset(plane_offset)
}

/// The plane-0 offset a source import of `img` presents to EGL: the
/// offset the DMA-BUF import resolves on Linux, the tensor's own offset
/// elsewhere.
fn source_import_plane0_offset(img: &Tensor<u8>, img_fmt: PixelFormat) -> usize {
    #[cfg(target_os = "linux")]
    {
        super::dma_import::resolved_source_plane0_offset(img, img_fmt)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = img_fmt;
        img.plane_offset().unwrap_or(0)
    }
}

/// The byte offset a destination import bases at on the route that
/// collapses a view to its parent (`bind_dst`'s
/// `get_or_create_egl_image(Dst)`): 0 for a view, the plane offset
/// otherwise.
fn view_collapsed_dst_base(
    view_origin: Option<edgefirst_tensor::ViewOrigin>,
    plane_offset: Option<usize>,
) -> usize {
    if view_origin.is_some() {
        0
    } else {
        plane_offset.unwrap_or(0)
    }
}

/// Whether a semi-planar source's chroma plane lives in a different DMA-BUF
/// from its luma, by buffer identity.
fn chroma_in_separate_buffer(img: &Tensor<u8>) -> bool {
    img.chroma()
        .is_some_and(|c| c.buffer_identity().id() != img.buffer_identity().id())
}

/// The plane-1 (interleaved CbCr) byte offset a contiguous NV12 import hands
/// EGL: `pitch * height` past the luma offset, saturating.
fn nv12_plane1_offset(plane0_offset: usize, pitch: usize, height: usize) -> usize {
    plane0_offset.saturating_add(pitch.saturating_mul(height))
}

impl GLProcessorST {
    /// Issue the single batched GPU sync if a `convert_deferred` is still owed,
    /// then clear the pending flag. No-op when nothing is pending.
    ///
    /// # Safety contract
    /// Must run on the GL worker thread that owns the context.
    pub(super) fn flush_pending(&mut self) {
        if self.pending_flush {
            // SAFETY: called on the context-owning GL worker thread.
            unsafe { float::finish_via_fence() };
            self.pending_flush = false;
        }
        if self.pending_submit {
            // One device submission for the whole batch.
            Platform::submit_recorded(&self.gl_context);
            self.pending_submit = false;
        }
    }

    /// Build a processor.
    ///
    /// `capacity` is the per-cache import-cache bound
    /// ([`ImageProcessorConfig::egl_cache_capacity`]); `None` (or `Some(0)`)
    /// defers to `EDGEFIRST_EGL_CACHE_CAPACITY` and then to the default —
    /// see [`resolve_egl_cache_capacity`].
    ///
    /// [`ImageProcessorConfig::egl_cache_capacity`]: crate::ImageProcessorConfig::egl_cache_capacity
    pub fn new(
        kind: Option<EglDisplayKind>,
        capacity: Option<usize>,
    ) -> Result<GLProcessorST, crate::Error> {
        let gl_context = Platform::init_display(kind)?;
        // Loads the GL function pointers once per process.
        Platform::load_gl_once(&gl_context);

        let GlSupport {
            has_float_linear,
            has_yuv_target,
            has_bgra,
            renderer,
            renderer_name,
            supports_f32_color,
            supports_f16_color,
        } = Self::gl_check_support()?;
        let RendererTraits {
            vivante: is_vivante,
            mali: is_mali,
            software: is_software_renderer,
            virtual_gpu: is_virtual_gpu,
            angle: is_angle,
            adreno: is_adreno,
        } = renderer;

        // Rejects software renderers (llvmpipe, softpipe, swrast) unless
        // `EDGEFIRST_ALLOW_SOFTWARE_GL=1`.
        if should_reject_software_gl(is_software_renderer, software_gl_override_enabled()) {
            return Err(crate::Error::NotSupported(
                "software OpenGL renderer detected (llvmpipe/softpipe/swrast); \
                 GL backend disabled — check EGL ICD configuration if a \
                 hardware GPU is expected (set EDGEFIRST_ALLOW_SOFTWARE_GL=1 \
                 to override, e.g. for coverage on Mesa llvmpipe)"
                    .into(),
            ));
        }
        if is_software_renderer {
            log::warn!(
                "software OpenGL renderer in use (EDGEFIRST_ALLOW_SOFTWARE_GL=1); \
                 slower than the CPU backend — intended for CI coverage only"
            );
        }

        // Uploads and downloads are all packed with no alignment requirements
        unsafe {
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ALIGNMENT, 1);
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ALIGNMENT, 1);
        }

        // External-OES sampler programs are built only where the platform binds
        // imports through them.
        let texture_program_planar = if Platform::EXTERNAL_OES {
            Some(GlProgram::new(
                generate_vertex_shader(),
                generate_planar_rgb_shader(),
            )?)
        } else {
            None
        };

        let texture_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_fragment_shader())?;

        let texture_program_yuv = if Platform::EXTERNAL_OES {
            Some(GlProgram::new(
                generate_vertex_shader(),
                generate_texture_fragment_shader_yuv(),
            )?)
        } else {
            None
        };

        // The default build; the mapped one links on the first mapped convert.
        let int8 = Int8Set::new(false, has_yuv_target)?;
        // SAFETY: the programs were linked above on this thread's current
        // context; the name is a `c"..."` literal.
        let texture_src_extent_locs = [
            unsafe {
                edgefirst_gl::gl::GetUniformLocation(texture_program.id, c"src_extent".as_ptr())
            },
            int8.texture_src_extent_loc,
        ];

        let yuyv_program_2d = GlProgram::new(
            generate_vertex_shader(),
            super::shaders_common::YUYV_RGBA_2D_FRAGMENT,
        )?;
        let yuyv_2d_locs = [yuyv_2d_locs(&yuyv_program_2d), int8.yuyv_2d_locs];

        let segmentation_program =
            GlProgram::new(generate_vertex_shader(), generate_segmentation_shader())?;
        segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;
        let instanced_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_instanced_segmentation_shader(),
        )?;
        instanced_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // f16 proto shader (RGBA16F, 4 protos per layer)
        let proto_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader(),
        )?;
        proto_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Int8 proto shaders (R8I, 1 proto per layer, 32 layers)
        let proto_segmentation_int8_nearest_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_nearest(),
        )?;
        proto_segmentation_int8_nearest_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_segmentation_int8_bilinear_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_bilinear(),
        )?;
        proto_segmentation_int8_bilinear_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_dequant_int8_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_dequant_shader_int8(),
        )?;

        // F32 proto shader (R32F, 1 proto per layer, 32 layers)
        let proto_segmentation_f32_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_f32(),
        )?;
        proto_segmentation_f32_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let color_program = GlProgram::new(generate_vertex_shader(), generate_color_shader())?;
        color_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // SAFETY: each program, when present, was linked above on this
        // thread's current context; the name is a `c"..."` literal.
        let [yuv_loc, planar_loc] = [
            texture_program_yuv.as_ref(),
            texture_program_planar.as_ref(),
        ]
        .map(|p| {
            p.map_or(-1, |p| unsafe {
                edgefirst_gl::gl::GetUniformLocation(p.id, c"src_extent".as_ptr())
            })
        });
        let external_src_extent_locs = [
            yuv_loc,
            int8.yuv_src_extent_loc,
            planar_loc,
            int8.planar_src_extent_loc,
        ];

        // Planar RGB shaders with sampler2D, for pass 2 of the two-pass planar
        // convert.
        let texture_program_planar_2d =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader_2d())?;

        // RGB packing shaders (2D only), for pass 2 of the two-pass pipeline.
        let packed_rgba8_program_2d =
            GlProgram::new(generate_vertex_shader(), generate_packed_rgba8_shader_2d())?;

        // Float render programs: full-viewport fragment shaders driven by
        // `gl_FragCoord`. Crop and letterbox are uniform-driven (src_rect_uv/
        // dst_rect_px/pad_color).
        let float_f32_nhwc_program =
            GlProgram::new(generate_vertex_shader(), generate_packed_f32_nhwc_shader())?;
        let float_f16_nchw_program = GlProgram::new(
            generate_vertex_shader(),
            generate_planar_rgb_f16_packed_shader(),
        )?;
        let float_nhwc_pack_program = GlProgram::new(
            generate_vertex_shader(),
            generate_float_nhwc_packed_shader(),
        )?;
        let float_rgba_program =
            GlProgram::new(generate_vertex_shader(), generate_float_rgba_shader())?;

        // NV12/NV16/NV24 → RGBA via the R8 texelFetch shader.
        let Int8Set {
            texture: texture_int8_program,
            texture_yuv: texture_int8_program_yuv,
            planar: texture_program_planar_int8,
            planar_2d: texture_program_planar_int8_2d,
            yuyv_2d: yuyv_int8_program_2d,
            packed_rgba8_2d: packed_rgba8_int8_program_2d,
            nv_r8: [nv_int8, nv_bilinear_int8, nv_hw_bilinear_int8],
            nv_r8_uniforms: [nv_int8_u, nv_bilinear_int8_u, nv_hw_bilinear_int8_u],
            nv_y2y: y2y_int8,
            rgb_r8: [rgb_int8, rgb_bilinear_int8, rgb_hw_bilinear_int8],
            rgb_r8_locs: [rgb_int8_l, rgb_bilinear_int8_l, rgb_hw_bilinear_int8_l],
            yuyv_luma_r8: [yuyv_int8, yuyv_bilinear_int8, yuyv_hw_bilinear_int8],
            yuyv_luma_r8_locs: [yuyv_int8_l, yuyv_bilinear_int8_l, yuyv_hw_bilinear_int8_l],
            luma_2d: luma_2d_int8_program,
            ..
        } = int8;
        let [nv, nv_bilinear, nv_hw_bilinear] = [
            generate_nv_to_rgba_shader_2d(),
            generate_nv_to_rgba_bilinear_shader_2d(),
            generate_nv_to_rgba_hw_bilinear_shader_2d(),
        ]
        .map(|src| GlProgram::new(generate_vertex_shader(), src));
        let [nv, nv_bilinear, nv_hw_bilinear] = [nv?, nv_bilinear?, nv_hw_bilinear?];
        // Resolves uniform locations and uploads the constant sampler bindings:
        // pass-2 packing samples the intermediate on unit 1, planar-2d on unit 0.
        let [nv_u, nv_bilinear_u, nv_hw_bilinear_u] =
            [&nv, &nv_bilinear, &nv_hw_bilinear].map(NvUniformState::resolve);
        let nv_r8_programs = [
            nv,
            nv_int8,
            nv_bilinear,
            nv_bilinear_int8,
            nv_hw_bilinear,
            nv_hw_bilinear_int8,
        ];
        let nv_r8_uniforms = [
            nv_u,
            nv_int8_u,
            nv_bilinear_u,
            nv_bilinear_int8_u,
            nv_hw_bilinear_u,
            nv_hw_bilinear_int8_u,
        ];
        // Packed RGB from an R8 view, indexed like `nv_r8_programs`.
        let [rgb, rgb_bilinear, rgb_hw_bilinear] = [
            generate_rgb_r8_to_rgba_shader_2d(),
            generate_rgb_r8_to_rgba_bilinear_shader_2d(),
            generate_rgb_r8_to_rgba_hw_bilinear_shader_2d(),
        ]
        .map(|src| GlProgram::new(generate_vertex_shader(), src));
        let [rgb, rgb_bilinear, rgb_hw_bilinear] = [rgb?, rgb_bilinear?, rgb_hw_bilinear?];
        let [rgb_l, rgb_bilinear_l, rgb_hw_bilinear_l] =
            [&rgb, &rgb_bilinear, &rgb_hw_bilinear].map(PackedR8Locs::resolve);
        let rgb_r8_programs = [
            rgb,
            rgb_int8,
            rgb_bilinear,
            rgb_bilinear_int8,
            rgb_hw_bilinear,
            rgb_hw_bilinear_int8,
        ];
        let rgb_r8_locs = [
            rgb_l,
            rgb_int8_l,
            rgb_bilinear_l,
            rgb_bilinear_int8_l,
            rgb_hw_bilinear_l,
            rgb_hw_bilinear_int8_l,
        ];
        // YUYV luma for Grey destinations, from the same R8 view, indexed
        // like `nv_r8_programs`.
        let [yuyv, yuyv_bilinear, yuyv_hw_bilinear] = [
            super::shaders_common::YUYV_LUMA_R8_FRAGMENT,
            super::shaders_common::YUYV_LUMA_R8_BILINEAR_FRAGMENT,
            super::shaders_common::YUYV_LUMA_R8_HW_BILINEAR_FRAGMENT,
        ]
        .map(|src| GlProgram::new(generate_vertex_shader(), src));
        let [yuyv, yuyv_bilinear, yuyv_hw_bilinear] = [yuyv?, yuyv_bilinear?, yuyv_hw_bilinear?];
        let [yuyv_l, yuyv_bilinear_l, yuyv_hw_bilinear_l] =
            [&yuyv, &yuyv_bilinear, &yuyv_hw_bilinear].map(PackedR8Locs::resolve);
        let yuyv_luma_r8_programs = [
            yuyv,
            yuyv_int8,
            yuyv_bilinear,
            yuyv_bilinear_int8,
            yuyv_hw_bilinear,
            yuyv_hw_bilinear_int8,
        ];
        let yuyv_luma_r8_locs = [
            yuyv_l,
            yuyv_int8_l,
            yuyv_bilinear_l,
            yuyv_bilinear_int8_l,
            yuyv_hw_bilinear_l,
            yuyv_hw_bilinear_int8_l,
        ];
        // Pass 2 of a colour source into a Grey destination.
        let luma_2d_program = GlProgram::new(
            generate_vertex_shader(),
            super::shaders_common::LUMA_2D_FRAGMENT,
        )?;
        luma_2d_program.load_uniform_1i(c"tex", 0)?;
        let nv_y2y_programs = if has_yuv_target {
            match (
                GlProgram::new(generate_vertex_shader(), generate_nv_to_rgba_y2y_shader()),
                y2y_int8,
            ) {
                (Ok(u8_program), Some((int8_program, int8_uniforms))) => {
                    let u8_uniforms = NvUniformState::resolve(&u8_program);
                    Some(([u8_program, int8_program], [u8_uniforms, int8_uniforms]))
                }
                (a, _) => {
                    log::info!(
                        "GL_EXT_YUV_target present but the Y2Y NV programs did not build \
                         ({:?}); bilinear NV resizes of imports filter in the shader",
                        a.err()
                    );
                    None
                }
            }
        } else {
            None
        };
        packed_rgba8_program_2d.load_uniform_1i(c"tex", 1)?;
        texture_program_planar_2d.load_uniform_1i(c"tex", 0)?;

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let draw_render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let proto_texture = Texture::new();
        let proto_dequant_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

        // Native packed-RGB import where the display lists `BG24`, the GPU is
        // not Adreno and `samplerExternalOES` is available.
        let rgb_native_import =
            Platform::EXTERNAL_OES && !is_adreno && Platform::lists_packed_rgb_import(&gl_context);
        log::debug!("packed Rgb DMA-BUF sources import natively: {rgb_native_import}");

        // EGLImage cache capacity (per cache: src / dst / nv_r8 / nv_uv).
        let egl_cache_capacity = resolve_egl_cache_capacity(
            capacity,
            std::env::var("EDGEFIRST_EGL_CACHE_CAPACITY")
                .ok()
                .as_deref(),
        );

        // NV* conversion-path preference: `auto` (default), or `sampler`/`shader`
        // to force a path.
        let nv_path_pref = parse_nv_path_pref();
        if nv_path_pref != NvPathPref::Auto {
            log::info!("EDGEFIRST_NV_CONVERT_PATH override: {nv_path_pref:?}");
        }

        // Colorimetry mode: the env var pins it for the processor's lifetime;
        // otherwise it starts as Fast.
        let colorimetry_env = parse_colorimetry_env();
        if let Some(mode) = colorimetry_env {
            log::info!("EDGEFIRST_COLORIMETRY override: {mode:?}");
        }

        let mut converter = GLProcessorST {
            gl_context,
            texture_program,
            texture_program_yuv,
            texture_int8_program,
            texture_int8_program_yuv,
            texture_program_planar,
            texture_program_planar_int8,
            yuyv_program_2d,
            yuyv_int8_program_2d,
            yuyv_2d_locs,
            texture_src_extent_locs,
            external_src_extent_locs,
            packed_rgba8_program_2d,
            packed_rgba8_int8_program_2d,
            texture_program_planar_2d,
            texture_program_planar_int8_2d,
            float_f32_nhwc_program,
            float_f16_nchw_program,
            float_nhwc_pack_program,
            float_rgba_program,
            float_render_texture: Texture::new(),
            float_render_tex_dims: (0, 0, 0),
            camera_eglimage_texture,
            camera_normal_texture,
            segmentation_texture,
            proto_texture,
            proto_segmentation_int8_nearest_program,
            proto_segmentation_int8_bilinear_program,
            proto_dequant_int8_program,
            proto_segmentation_f32_program,
            has_float_linear,
            has_bgra,
            supports_f32_color,
            supports_f16_color,
            int8_interpolation_mode: Int8InterpolationMode::Bilinear,
            proto_dequant_texture,
            proto_dequant_tex_dims: (0, 0, 0, 0),
            proto_dequant_fbo: FrameBuffer::new(),
            proto_quad_locs: std::cell::Cell::new((0, -1, -1)),
            vertex_buffer,
            texture_buffer,
            convert_fbo: FrameBuffer::new(),
            draw_fbo: FrameBuffer::new(),
            draw_render_texture,
            src_egl_cache: ImportCache::new(egl_cache_capacity),
            dst_egl_cache: ImportCache::new(egl_cache_capacity),
            bgra_warned: false,
            readback_scratch: Vec::new(),
            #[cfg(test)]
            fail_dst_import: false,
            #[cfg(test)]
            fail_nv12_programs: false,
            probe_dst_import_only: false,
            dst_import_warned: std::collections::HashSet::new(),
            is_vivante,
            is_mali,
            is_virtual_gpu,
            is_angle,
            is_adreno,
            nv12_policy: Nv12DstPolicy::for_renderer(renderer),
            soft_declines: true,
            renderer_name,
            use_renderbuffer: std::env::var("EDGEFIRST_OPENGL_RENDERSURFACE")
                .map(|v| v == "1")
                .unwrap_or(false),
            packed_rgb_intermediate_tex: Texture::new(),
            packed_rgb_fbo: FrameBuffer::new(),
            packed_rgb_intermediate_size: (0, 0),
            render_texture,
            segmentation_program,
            instanced_segmentation_program,
            proto_segmentation_program,
            color_program,
            box_line_width: crate::DEFAULT_BOX_LINE_WIDTH as f32,
            cached_opacity: f32::NAN, // sentinel: forces first set_opacity_uniform to initialize all shaders
            proto_tex_dims: (0, 0, 0, 0),
            proto_repack_compute_program: None,
            proto_compute_locs: (-1, -1, -1),
            proto_ssbo: 0,
            proto_ssbo_size: 0,
            nv_r8_programs,
            nv_r8_uniforms,
            rgb_r8_programs,
            rgb_r8_locs,
            yuyv_luma_r8_programs,
            yuyv_luma_r8_locs,
            luma_2d_program,
            luma_2d_int8_program,
            nv12_programs: Nv12ProgramState::Unbuilt,
            nv12_chroma_texture: Texture::new(),
            rgb_native_import,
            packed_r8_import: Platform::PACKED_R8_IMPORT,
            last_rgb_source_path: None,
            nv_r8_texture: Texture::new(),
            nv_r8_egl_cache: ImportCache::new(egl_cache_capacity),
            nv_uv_texture: Texture::new(),
            nv_uv_egl_cache: ImportCache::new(egl_cache_capacity),
            nv_uv_import_refused: false,
            nv_y2y_programs,
            last_nv_convert_path: NvConvertPath::None,
            last_nv_sampling: None,
            convert_stats: Default::default(),
            nv_import_warned: std::collections::HashSet::new(),
            owned_pbos: Default::default(),
            float_quad_locs: std::collections::HashMap::new(),
            float_norm: FloatNorm::IDENTITY,
            int8_programs: Vec::new(),
            int8_mapped: false,
            int8_alt: None,
            int_quant: None,
            int_pad: None,
            int_pad_output: false,
            float_quad_pos_vbo: 0,
            float_quad_uv_vbo: 0,
            nv_path_pref,
            colorimetry_mode: colorimetry_env.unwrap_or_default(),
            colorimetry_env_pinned: colorimetry_env.is_some(),
            interpolation: crate::Interpolation::default(),
            nv_hw_filter: std::env::var("EDGEFIRST_GL_NO_NV_HW_FILTER").map_or(true, |v| v != "1"),
            defer_finish: false,
            pending_flush: false,
            pending_submit: false,
        };
        check_gl_error(function!(), line!())?;

        // Compile the proto repack compute shader when GLES 3.1 is available
        // and EDGEFIRST_PROTO_COMPUTE=1.
        let compute_enabled = converter.gl_context.has_compute
            && std::env::var("EDGEFIRST_PROTO_COMPUTE")
                .map(|v| v == "1")
                .unwrap_or(false);
        if compute_enabled {
            match Self::compile_compute_program(generate_proto_repack_compute_shader()) {
                Ok(program) => {
                    log::info!("Proto repack compute shader compiled successfully");
                    converter.proto_repack_compute_program = Some(program);
                    unsafe {
                        converter.proto_compute_locs = (
                            edgefirst_gl::gl::GetUniformLocation(program, c"width".as_ptr()),
                            edgefirst_gl::gl::GetUniformLocation(program, c"height".as_ptr()),
                            edgefirst_gl::gl::GetUniformLocation(program, c"num_protos".as_ptr()),
                        );
                    }
                    let mut ssbo = 0u32;
                    unsafe { edgefirst_gl::gl::GenBuffers(1, &mut ssbo) };
                    converter.proto_ssbo = ssbo;
                }
                Err(e) => {
                    log::warn!("Proto repack compute shader failed: {e}; using CPU fallback");
                }
            }
        }

        log::debug!(
            "GLProcessorST: DMA destination attachment mode: {}",
            if converter.use_renderbuffer {
                "renderbuffer (EDGEFIRST_OPENGL_RENDERSURFACE=1)"
            } else {
                "texture (default)"
            }
        );

        // Verify that a DMA-buf render round-trips.
        if converter.gl_context.transfer_backend.is_dma() && !converter.verify_dma_buf_roundtrip() {
            log::info!("DMA-buf verification failed — falling back to PBO transfers");
            tracing::event!(
                tracing::Level::INFO,
                from = "dmabuf",
                to = "pbo",
                "image.convert.gl.transfer_downgrade"
            );
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
        }

        // Without DMA-buf, use PBO transfers.
        if converter.gl_context.transfer_backend == TransferBackend::Sync {
            log::info!("Upgrading transfer backend from Sync to Pbo (GL context available)");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
        }

        // Applies the transfer-backend env override: "dmabuf", "pbo" or
        // "sync" (case-insensitive).
        apply_forced_transfer(&mut converter.gl_context.transfer_backend);

        log::debug!(
            "GLConverter created (transfer={:?})",
            converter.gl_context.transfer_backend,
        );
        Ok(converter)
    }

    /// Verify that DMA-buf EGLImage round-trip works on this GPU.
    ///
    /// Renders a solid red quad to a 64x64 DMA-buf-backed RGBA texture via
    /// EGLImage, then reads it back and checks that the center pixel is red.
    /// Returns `true` if the data round-trips correctly.
    fn verify_dma_buf_roundtrip(&mut self) -> bool {
        let _probe = super::shaders::ProbeScope::enter();
        // Allocate a 64x64 RGBA DMA source tensor and fill it with solid red
        let src = match Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::Write,
        ) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA source: {e}");
                return false;
            }
        };

        {
            let mut map = match src.map_write() {
                Ok(m) => m,
                Err(e) => {
                    log::info!("verify_dma_buf_roundtrip: failed to map DMA source: {e}");
                    return false;
                }
            };
            for pixel in map.as_chunks_mut::<4>().0 {
                pixel[0] = 255; // R
                pixel[1] = 0; // G
                pixel[2] = 0; // B
                pixel[3] = 255; // A
            }
        }

        // Allocate a 64x64 RGBA DMA destination tensor
        let mut dst = match Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            edgefirst_tensor::CpuAccess::Read,
        ) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA destination: {e}");
                return false;
            }
        };

        // Run the DMA-buf EGLImage render pipeline (RGBA→RGBA DMA) with the
        // mapped-texture fallback disabled.
        self.probe_dst_import_only = true;
        let outcome = self.convert_via_engine(
            &mut dst,
            PixelFormat::Rgba,
            &src,
            PixelFormat::Rgba,
            false,
            Rotation::None,
            Flip::None,
            ResolvedCrop::no_crop(),
        );
        self.probe_dst_import_only = false;
        if let Err(e) = outcome {
            log::info!("verify_dma_buf_roundtrip: convert failed: {e}");
            return false;
        }

        // Read back the center pixel at (32, 32) from the destination
        let map = match dst.map_read() {
            Ok(m) => m,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to map DMA destination: {e}");
                return false;
            }
        };

        let offset = (32 * 64 + 32) * 4;
        if map.len() < offset + 4 {
            log::info!("verify_dma_buf_roundtrip: destination buffer too small");
            return false;
        }

        let r = map[offset];
        let g = map[offset + 1];
        let b = map[offset + 2];
        let a = map[offset + 3];

        let pass = r > 250 && g < 5 && b < 5 && a > 250;

        if pass {
            log::info!("verify_dma_buf_roundtrip: PASSED (center pixel RGBA={r},{g},{b},{a})");
        } else {
            log::info!(
                "verify_dma_buf_roundtrip: FAILED (center pixel RGBA={r},{g},{b},{a}, \
                 expected ~255,0,0,255)"
            );
        }

        pass
    }

    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) {
        self.int8_interpolation_mode = mode;
        log::debug!("Int8 interpolation mode set to {:?}", mode);
    }

    /// Sets the colorimetry/performance trade-off (see
    /// [`crate::ColorimetryMode`]). When `EDGEFIRST_COLORIMETRY` pinned the
    /// mode at construction, the env value wins: this logs and keeps it.
    pub fn set_colorimetry_mode(&mut self, mode: crate::ColorimetryMode) {
        if self.colorimetry_env_pinned {
            if mode != self.colorimetry_mode {
                log::info!(
                    "ColorimetryMode::{mode:?} requested but EDGEFIRST_COLORIMETRY pins \
                     {:?} — keeping the env override",
                    self.colorimetry_mode
                );
            }
            return;
        }
        self.colorimetry_mode = mode;
        log::debug!("Colorimetry mode set to {:?}", mode);
    }

    /// Sets how resizing converts resample (see [`crate::Interpolation`]).
    pub fn set_interpolation(&mut self, mode: crate::Interpolation) {
        self.interpolation = mode;
        log::debug!("Interpolation set to {mode:?}");
    }

    /// Whether the Y2Y NV programs ([`NvSampling::HardwareYuv`]) built.
    #[cfg(test)]
    pub(super) fn has_y2y(&self) -> bool {
        self.nv_y2y_programs.is_some()
    }

    /// Whether a packed `Rgb` DMA-BUF source imports natively here.
    #[cfg(test)]
    pub(super) fn rgb_native_import(&self) -> bool {
        self.rgb_native_import
    }

    /// Withhold routes from packed `Rgb` DMA-BUF sources. Only narrows: a
    /// route this GPU lacks is not granted.
    #[cfg(test)]
    pub(super) fn restrict_rgb_source_routes(&mut self, native: bool, r8_import: bool) {
        self.rgb_native_import &= native;
        self.packed_r8_import &= r8_import;
    }

    /// Allow or forbid [`NvSampling::Hardware`] (see `nv_hw_filter`).
    #[cfg(test)]
    pub(super) fn set_nv_hw_filter(&mut self, enabled: bool) {
        self.nv_hw_filter = enabled;
    }

    /// `MIN`/`MAG` filter for a source texture that a draw resamples.
    pub(super) fn source_filter(&self) -> u32 {
        match self.interpolation.supported() {
            crate::Interpolation::Nearest => edgefirst_gl::gl::NEAREST,
            _ => edgefirst_gl::gl::LINEAR,
        }
    }

    /// Snapshot the EGLImage cache counters (src, dst, NV R8).
    pub(crate) fn egl_cache_stats(&self) -> GlCacheStats {
        GlCacheStats {
            src: self.src_egl_cache.stats(),
            dst: self.dst_egl_cache.stats(),
            nv_r8: self.nv_r8_egl_cache.stats(),
            nv_uv: self.nv_uv_egl_cache.stats(),
        }
    }

    /// A tensor's memory as this context can use it: a PBO allocated by
    /// another context counts as host memory (see [`OwnedPbos`]).
    fn usable_memory(
        &self,
        memory: TensorMemory,
        pbo_vtable: Option<*const std::ffi::c_void>,
    ) -> TensorMemory {
        match memory {
            TensorMemory::Pbo if !self.owned_pbos.contains(pbo_vtable) => TensorMemory::Mem,
            other => other,
        }
    }

    pub(crate) fn convert_stats(&self) -> super::cache::ConvertStats {
        self.convert_stats
    }

    /// The body of [`ImageProcessorTrait::convert`]. A per-convert
    /// interpolation in `options` replaces the processor's for this call only.
    fn convert_inner(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: &crate::ConvertOptions,
    ) -> crate::Result<()> {
        let processor_mode = self.interpolation;
        if let Some(mode) = options.get_interpolation() {
            self.interpolation = mode;
        }
        let result = self.convert_resampling(src, dst, options);
        self.interpolation = processor_mode;
        result
    }

    /// [`Self::convert_inner`] with `self.interpolation` already chosen.
    fn convert_resampling(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: &crate::ConvertOptions,
    ) -> crate::Result<()> {
        let (rotation, flip, crop) = (
            options.get_rotation(),
            options.get_flip(),
            options.get_crop(),
        );
        // Declines a view()/batch() destination that is not a u8/i8 DMA buffer
        // on a zero-copy backend.
        if dst.view_origin().is_some()
            && !(self.gl_context.transfer_backend.is_zero_copy()
                && dst.memory() == TensorMemory::DmaBuf
                && matches!(
                    dst.dtype(),
                    edgefirst_tensor::DType::U8 | edgefirst_tensor::DType::I8
                ))
        {
            return Err(Error::NotSupported(
                "GL view()/batch() destination is supported only for u8/i8 DMA buffers; \
                 CPU fallback handles other cases"
                    .into(),
            ));
        }

        let crop = crop.resolve(
            src.width().unwrap_or(0),
            src.height().unwrap_or(0),
            dst.width().unwrap_or(0),
            dst.height().unwrap_or(0),
            options.get_placement(),
        )?;
        let mapping = crate::options::output_map_for(options, dst, crop.dst_rect.is_some())?;
        let pad_output = mapping.as_ref().and_then(|(_, p)| *p);
        let mut crop = crop;
        // A Grey destination is padded with the colour's luma in every channel.
        if dst.format() == Some(PixelFormat::Grey) {
            crop.dst_color = crop.dst_color.map(|c| {
                let [l] = crate::cpu::CPUProcessor::rgba_to_grey(c);
                [l, l, l, c[3]]
            });
        }
        let mut pad_output_exact = false;
        self.float_norm = FloatNorm::IDENTITY;
        self.int_pad = None;
        self.int_pad_output = false;
        let mut int_mapped = false;
        if let Some((map, pad_output)) = &mapping {
            let pad = crop.dst_color.map(|p| map.pad_values(p, *pad_output));
            if map.bounds.is_some() {
                if dst.format() == Some(PixelFormat::Bgra) {
                    return Err(Error::NotSupported(
                        "GL integer output mapping of Bgra; CPU fallback".into(),
                    ));
                }
                self.use_int8_set(true)?;
                self.load_int_quant(IntQuant::from_map(map))?;
                self.int_pad = pad.map(|v| v.map(|x| x as i32 as u8));
                int_mapped = true;
                // An output pad that some sample maps to exactly becomes that pixel pad.
                if let (Some(p), Some([_, _, _, a])) = (pad, crop.dst_color) {
                    // A Grey destination fills all three channels with its grey sample.
                    let samples = if dst.format() == Some(PixelFormat::Grey) {
                        [map.sample_for(0, p[0]); 3]
                    } else {
                        [
                            map.sample_for(0, p[0]),
                            map.sample_for(1, p[1]),
                            map.sample_for(2, p[2]),
                        ]
                    };
                    if let [Some(sr), Some(sg), Some(sb)] = samples {
                        crop.dst_color = Some([sr, sg, sb, a]);
                        pad_output_exact = true;
                    }
                }
            } else {
                // The float shaders sample in [0, 1]; the map is per 8-bit value.
                self.float_norm = FloatNorm {
                    scale: map.scale.map(|s| s * 255.0),
                    offset: map.offset,
                    pad,
                };
            }
        }
        if !int_mapped && dst.dtype() == edgefirst_tensor::DType::I8 {
            self.use_int8_set(false)?;
        }

        // F16/F32 destination: dispatch to a GL float render path when one
        // exists.
        let dst_dtype = dst.dtype();
        if matches!(
            dst_dtype,
            edgefirst_tensor::DType::F16 | edgefirst_tensor::DType::F32
        ) {
            let src_fmt = src.format().ok_or(Error::NotAnImage)?;
            let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
            let support = float_render_support(
                self.is_vivante,
                self.float_import_refused(),
                self.supports_f32_color,
                self.supports_f16_color,
            );
            // NV*→float takes two engine passes: NV→RGBA into an intermediate, then
            // the packed float render, classified on the RGBA intermediate. A
            // packed Rgb DMA-BUF source takes the same route.
            let nv_src = matches!(
                src_fmt,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) || (src_fmt == PixelFormat::Rgb
                && Platform::PACKED_R8_IMPORT
                && src.memory() == TensorMemory::DmaBuf
                && self.gl_context.transfer_backend.is_zero_copy());
            let path = classify_float_render(
                if nv_src { PixelFormat::Rgba } else { src_fmt },
                dst_fmt,
                dst_dtype,
                self.usable_memory(dst.memory(), dst.pbo_vtable_ptr()),
                support,
                <Platform as GlPlatform>::ZERO_COPY_FLOAT,
            );
            // Declines a zero-copy float destination the platform cannot place.
            if matches!(
                path,
                FloatRenderPath::ZeroCopyF16Nchw
                    | FloatRenderPath::ZeroCopyF32Nchw
                    | FloatRenderPath::ZeroCopyFloatNhwc
                    | FloatRenderPath::ZeroCopyFloatRgba
            ) {
                let places = match dst_dtype {
                    edgefirst_tensor::DType::F16 => Platform::dst_import_places(
                        dst.as_typed::<half::f16>().expect("dtype checked"),
                    ),
                    edgefirst_tensor::DType::F32 => {
                        Platform::dst_import_places(dst.as_typed::<f32>().expect("dtype checked"))
                    }
                    _ => unreachable!("dst_dtype is F16 or F32 in this branch"),
                };
                if !places {
                    return Err(Error::NotSupported(format!(
                        "GL float destination at plane offset {} cannot be placed by \
                         this platform/driver -- either no view origin and an import \
                         that binds the whole buffer from its origin, or a driver whose \
                         import rejects the offset -- and there is no mapped-texture \
                         readback for a float DMA destination to lower to; CPU fallback \
                         handles it",
                        dst.plane_offset().unwrap_or(0)
                    )));
                }
            }
            match path {
                FloatRenderPath::ZeroCopyF16Nchw
                | FloatRenderPath::ZeroCopyF32Nchw
                | FloatRenderPath::ZeroCopyFloatNhwc
                | FloatRenderPath::ZeroCopyFloatRgba => {
                    let _span = Self::enter_image_convert_gl_span(
                        src_fmt,
                        dst_fmt,
                        false,
                        src.memory(),
                        dst.memory(),
                    );
                    let convert_span = tracing::Span::current();
                    return if nv_src {
                        self.convert_nv_to_float_two_pass(
                            src,
                            dst,
                            path,
                            rotation,
                            flip,
                            crop,
                            &convert_span,
                        )
                    } else {
                        self.convert_float_to_zero_copy(
                            src,
                            dst,
                            path,
                            rotation,
                            flip,
                            crop,
                            &convert_span,
                        )
                    };
                }
                FloatRenderPath::PboF16Nchw
                | FloatRenderPath::PboF32Nhwc
                | FloatRenderPath::PboF32Nchw
                | FloatRenderPath::PboF16Nhwc => {
                    let _span = Self::enter_image_convert_gl_span(
                        src_fmt,
                        dst_fmt,
                        false,
                        src.memory(),
                        dst.memory(),
                    );
                    let convert_span = tracing::Span::current();
                    return self.convert_float_to_pbo(
                        src,
                        dst,
                        path,
                        rotation,
                        flip,
                        crop,
                        &convert_span,
                    );
                }
                FloatRenderPath::None => {}
            }
            // `FloatRenderPath::None` continues to the u8 path, which rejects
            // F16/F32.
        }

        // Odd destination dims, for the error wrap at the end.
        let dst_odd = match (dst.width(), dst.height()) {
            (Some(w), Some(h)) if w % 2 != 0 || h % 2 != 0 => Some((w, h)),
            _ => None,
        };

        let (src_u8, src_fmt) = dyn_to_u8_src(src)?;
        let (dst_u8, dst_fmt, is_int8) = dyn_to_u8_dst(dst)?;
        // A mapped u8 destination renders through the int8 programs too.
        let is_int8 = is_int8 || int_mapped;
        self.int_pad_output = int_mapped && pad_output.is_some() && !pad_output_exact;

        let result = self.convert_impl(
            src_u8, src_fmt, dst_u8, dst_fmt, is_int8, rotation, flip, crop,
        );

        // An EGL/GL failure on an odd-sized destination is returned as
        // `NotSupported` carrying the underlying error.
        match result {
            Err(e) if dst_odd.is_some() && matches!(e, Error::EGL(_) | Error::OpenGl(_)) => {
                let (w, h) = dst_odd.unwrap();
                Err(Error::NotSupported(format!(
                    "Conversion failed with {e:?} and target tensor has odd \
                     dimensions {w}x{h}, which is not supported by all platforms"
                )))
            }
            other => other,
        }
    }

    /// Convert without the terminal CPU sync, exporting a native fence fd
    /// that signals when the GPU work completes.
    ///
    /// Paths that read back to the CPU (PBO/upload destinations) sync
    /// internally, so the fence is already signaled there. When the platform
    /// has no native fence (`Ok(None)` / export error) the convert blocks
    /// with an explicit `glFinish`.
    pub(crate) fn convert_fenced(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        options: crate::ConvertOptions,
    ) -> Result<Option<super::CompletionFence>, crate::Error> {
        self.defer_finish = true;
        let result = self.convert_inner(src, dst, &options);
        self.defer_finish = false;
        result?;
        // Records completion on the destination and exports the fence for the
        // same sync point.
        let recorded = Platform::record_completion(&self.gl_context, dst, true);
        match Platform::export_completion_fence(&self.gl_context, recorded) {
            Ok(Some(fd)) => Ok(Some(fd)),
            Ok(None) => {
                unsafe { edgefirst_gl::gl::Finish() };
                check_gl_error(function!(), line!())?;
                Ok(None)
            }
            Err(e) => {
                // A failed export falls back to `glFinish`.
                log::debug!("native fence export failed ({e:?}); falling back to glFinish");
                unsafe { edgefirst_gl::gl::Finish() };
                check_gl_error(function!(), line!())?;
                Ok(None)
            }
        }
    }

    // Internal methods operating on Tensor<u8> + PixelFormat directly.

    /// The convert span both the u8 engine and the float route record feed
    /// fields on. Recording sites must use this handle, not `Span::current()`.
    pub(super) fn enter_image_convert_gl_span(
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
        is_int8: bool,
        src_memory: TensorMemory,
        dst_memory: TensorMemory,
    ) -> tracing::span::EnteredSpan {
        tracing::trace_span!(
            "image.convert.gl",
            ?src_fmt,
            ?dst_fmt,
            is_int8,
            src_memory = ?src_memory,
            dst_memory = ?dst_memory,
            // Recorded at the source-feed site: import | pbo | upload.
            src_feed = tracing::field::Empty,
            // Recorded at the destination-feed site: zero_copy | mapped_texture |
            // pbo (see `DstTarget::feed`).
            dst_feed = tracing::field::Empty,
        )
        .entered()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert_impl(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let _span = Self::enter_image_convert_gl_span(
            src_fmt,
            dst_fmt,
            is_int8,
            src.memory(),
            dst.memory(),
        );
        if !Self::check_src_format_supported(self.gl_context.transfer_backend, src, src_fmt) {
            if src_fmt == PixelFormat::Vyuy
                && self.gl_context.transfer_backend.is_dma()
                && src.memory() == TensorMemory::DmaBuf
            {
                log::warn!(
                    "VYUY format not supported via EGL DMA-BUF import; \
                     falling back to CPU/G2D path"
                );
            }
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {src_fmt} source texture",
            )));
        }

        if !Self::dst_format_renderable(
            self.gl_context.transfer_backend,
            dst.memory(),
            dst_fmt,
            self.has_bgra,
            Platform::NV12_DST,
        ) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {dst_fmt} destination texture",
            )));
        }

        log::debug!(
            "dst tensor: {:?} src tensor :{:?}",
            dst.memory(),
            src.memory()
        );
        check_gl_error(function!(), line!())?;
        log::trace!(
            "GL convert_impl: {src_fmt}→{dst_fmt} int8={is_int8} \
             src_mem={:?} dst_mem={:?} transfer={:?}",
            src.memory(),
            dst.memory(),
            self.gl_context.transfer_backend,
        );

        self.convert_via_engine(dst, dst_fmt, src, src_fmt, is_int8, rotation, flip, crop)
    }

    /// The u8/i8 convert engine: classify the destination
    /// ([`super::render::lower_dst`]), pick the render plan
    /// ([`super::render::plan_convert`]), then `bind_dst` → render →
    /// readback. The two-pass plans delegate to their render strategies.
    #[allow(clippy::too_many_arguments)]
    fn convert_via_engine(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        // Declines a view()/batch() destination in a packed-RGB, NV12 or
        // planar format.
        if dst.view_origin().is_some()
            && (dst_fmt == PixelFormat::Rgb
                || dst_fmt == PixelFormat::Nv12
                || dst_fmt.layout() == PixelLayout::Planar)
        {
            return Err(crate::Error::NotSupported(
                "GL view()/batch() destination not yet supported for packed-RGB / \
                 NV12 / planar formats (CPU fallback handles it)"
                    .into(),
            ));
        }
        if dst_fmt == PixelFormat::Nv12 && is_int8 {
            return Err(crate::Error::NotSupported(
                "GL renders NV12 destinations as U8 only".into(),
            ));
        }
        let dst_offset = view_collapsed_dst_base(dst.view_origin(), dst.plane_offset());
        // A destination import at an unaligned base is not placed on Adreno.
        let places = Platform::dst_import_places(dst)
            && !(self.is_adreno && unaligned_import_offset(dst_offset));
        if !places {
            log::debug!(
                "convert_via_engine: zero-copy destination importing at base offset \
                 {dst_offset} cannot be placed by this platform/driver; rendering \
                 to a texture and reading back through map()"
            );
        }
        // The `image.convert.gl` span, captured before the engine span is
        // entered.
        let convert_span = tracing::Span::current();
        let mut lowering = super::render::lower_dst(
            self.gl_context.transfer_backend.is_zero_copy() && places,
            self.usable_memory(dst.memory(), dst.pbo_vtable_ptr()),
        );
        // A PBO counts as a host source.
        let src_host = matches!(
            src.memory(),
            TensorMemory::Mem | TensorMemory::Shm | TensorMemory::Pbo
        );
        let mut plan = super::render::plan_convert(src_fmt, src_host, dst_fmt, lowering);
        // Requests the packed-RGB plan's destination import before any
        // rendering; a refusal re-plans onto the mapped-texture path.
        if plan == super::render::ConvertPlan::TwoPassPackedRgb {
            match self.probe_packed_rgb_dst_import(dst, dst_fmt) {
                // Records the destination feed for the packed-RGB plan.
                Ok(()) => {
                    convert_span.record("dst_feed", DstTarget::ZeroCopyImage.feed());
                }
                // A probe returns the refusal without falling back.
                Err(e) if self.probe_dst_import_only => return Err(e),
                Err(e) => {
                    self.note_dst_import_fallback(dst, dst_fmt, &e, TEXTURE_READBACK);
                    lowering = super::render::lower_dst(false, dst.memory());
                    plan = super::render::plan_convert(src_fmt, src_host, dst_fmt, lowering);
                }
            }
        }
        if dst_fmt == PixelFormat::Nv12 && plan != super::render::ConvertPlan::ThreePassNv12 {
            return Err(crate::Error::NotSupported(format!(
                "GL renders an NV12 destination only into a zero-copy import, \
                 not via {lowering:?}"
            )));
        }
        let _span = tracing::trace_span!(
            "image.convert.gl.engine",
            ?plan,
            ?lowering,
            src_pbo = src.memory() == TensorMemory::Pbo,
        )
        .entered();

        match plan {
            super::render::ConvertPlan::TwoPassPackedRgb => {
                return self.convert_to_packed_rgb(
                    src,
                    src_fmt,
                    dst,
                    dst_fmt,
                    is_int8,
                    rotation,
                    flip,
                    crop,
                    &convert_span,
                )
            }
            super::render::ConvertPlan::TwoPassNvPlanar => {
                return self.convert_nv_to_planar_two_pass(
                    src,
                    src_fmt,
                    dst,
                    dst_fmt,
                    is_int8,
                    rotation,
                    flip,
                    crop,
                    lowering,
                    &convert_span,
                )
            }
            super::render::ConvertPlan::TwoPassGrey => {
                return self.convert_to_grey_two_pass(
                    src,
                    src_fmt,
                    dst,
                    is_int8,
                    rotation,
                    flip,
                    crop,
                    lowering,
                    &convert_span,
                )
            }
            super::render::ConvertPlan::ThreePassNv12 => {
                return self.convert_to_nv12(src, dst, src_fmt, rotation, flip, crop, &convert_span)
            }
            super::render::ConvertPlan::SinglePass => {}
        }

        let (target, bound) = self.bind_dst(dst, dst_fmt, crop, lowering)?;
        // Re-plans against the lowering that was bound; a plan other than
        // `SinglePass` binds the destination itself.
        let replanned = super::render::plan_convert(src_fmt, src_host, dst_fmt, bound);
        if replanned != super::render::ConvertPlan::SinglePass {
            debug_assert_eq!(
                replanned,
                super::render::ConvertPlan::TwoPassNvPlanar,
                "a texture lowering can only re-plan onto the planar two-pass route"
            );
            return self.convert_nv_to_planar_two_pass(
                src,
                src_fmt,
                dst,
                dst_fmt,
                is_int8,
                rotation,
                flip,
                crop,
                bound,
                &convert_span,
            );
        }
        Self::record_dst_feed(&convert_span, target);
        // Only the zero-copy target carries the view's band; the texture
        // lowerings render the tile at origin (0, 0).
        let band = match target {
            DstTarget::ZeroCopyImage => dst.view_origin(),
            DstTarget::Texture { .. } => None,
        };

        // Bias the letterbox clear colour for int8.
        let crop = self.int8_bias_clear(is_int8, crop);

        let start = Instant::now();
        self.render_packed_or_planar(
            src,
            src_fmt,
            dst,
            dst_fmt,
            band,
            is_int8,
            rotation,
            flip,
            crop,
            &convert_span,
        )?;
        log::debug!("engine render ({plan:?}) takes {:?}", start.elapsed());

        if let DstTarget::Texture { readback } = target {
            let start = Instant::now();
            self.readback_rendered(dst, dst_fmt, readback.pbo_id())?;
            log::debug!("engine readback takes {:?}", start.elapsed());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_decoded_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        opacity: f32,
        background: Option<(&Tensor<u8>, PixelFormat)>,
        color_mode: crate::ColorMode,
    ) -> Result<(), crate::Error> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_decoded_masks");
        if !matches!(
            dst_fmt,
            PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb
        ) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let memory = self.usable_memory(dst.memory(), dst.pbo_vtable_ptr());
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "draw_decoded_masks: dst.memory()={memory:?} {dst_w}x{dst_h} fmt={dst_fmt:?}"
            );
        }
        let pbo_buffer_id = if memory == TensorMemory::Pbo {
            match (dst.pbo_id(), dst.pbo_is_mapped()) {
                (Some(id), Some(false)) => Some(id),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            TensorMemory::DmaBuf => match self.setup_draw_renderbuffer_dma(dst, dst_fmt) {
                Ok(()) => {
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "draw_decoded_masks: DMA fast path (setup_draw_renderbuffer_dma OK)"
                        );
                    }
                    true
                }
                Err(e) => {
                    warn_slow_path_once("draw_decoded_masks", "setup_draw_renderbuffer_dma", &e);
                    if let Some(buffer_id) = pbo_buffer_id {
                        self.setup_renderbuffer_from_pbo(dst, dst_fmt, buffer_id)?;
                    } else {
                        self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                    }
                    false
                }
            },
            _ if pbo_buffer_id.is_some() => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_decoded_masks: PBO path");
                }
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_decoded_masks: non-DMA fallback (memory={memory:?})");
                }
                self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                false
            }
        };

        // Write the base layer of the framebuffer before running mask passes:
        //
        // - background + zero-copy destination: draw bg into the destination
        //   framebuffer (see `draw_background_into_bound_fbo`).
        // - background + non-DMA destination: upload bg pixels into the
        //   render_texture that backs the FBO.
        // - no background: glClear(0x00000000) on the framebuffer.
        if let Some((bg, bg_fmt)) = background {
            if bg.width() != Some(dst_w) || bg.height() != Some(dst_h) {
                return Err(crate::Error::InvalidShape(
                    "background dimensions do not match dst".into(),
                ));
            }
            if is_dma {
                self.draw_background_into_bound_fbo(bg, bg_fmt)?;
            } else {
                // Non-DMA destination: upload bg pixels into the render_texture
                // attached to the FBO.
                self.upload_pixels_to_render_texture(bg, bg_fmt, dst_w, dst_h)?;
            }
        } else {
            // No background: clear the framebuffer.
            unsafe {
                edgefirst_gl::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            }
        }

        edgefirst_gl::enable(edgefirst_gl::gl::BLEND);
        edgefirst_gl::blend_func_separate(
            edgefirst_gl::gl::SRC_ALPHA,
            edgefirst_gl::gl::ONE_MINUS_SRC_ALPHA,
            edgefirst_gl::gl::ZERO,
            edgefirst_gl::gl::ONE,
        );

        self.set_opacity_uniform(opacity)?;
        self.render_box(dst_w, dst_h, detect, color_mode)?;
        self.render_segmentation(detect, segmentation, color_mode)?;

        edgefirst_gl::finish();
        if !is_dma {
            let format = match dst_fmt {
                PixelFormat::Rgb => edgefirst_gl::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA,
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                self.read_pixels_into_pbo(
                    dst,
                    buffer_id,
                    dst_w,
                    dst_h,
                    format,
                    dst_fmt == PixelFormat::Bgra,
                )?;
            } else {
                self.read_pixels_into_tensor(
                    dst,
                    dst_w,
                    dst_h,
                    format,
                    dst_fmt == PixelFormat::Bgra,
                )?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_proto_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        opacity: f32,
        background: Option<(&Tensor<u8>, PixelFormat)>,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_proto_masks");
        if !matches!(
            dst_fmt,
            PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Rgb
        ) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let memory = self.usable_memory(dst.memory(), dst.pbo_vtable_ptr());
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "draw_proto_masks: dst.memory()={memory:?} {dst_w}x{dst_h} fmt={dst_fmt:?}"
            );
        }
        let pbo_buffer_id = if memory == TensorMemory::Pbo {
            match (dst.pbo_id(), dst.pbo_is_mapped()) {
                (Some(id), Some(false)) => Some(id),
                _ => None,
            }
        } else {
            None
        };

        let is_dma = match memory {
            TensorMemory::DmaBuf => match self.setup_draw_renderbuffer_dma(dst, dst_fmt) {
                Ok(()) => {
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "draw_proto_masks: DMA fast path (setup_draw_renderbuffer_dma OK)"
                        );
                    }
                    true
                }
                Err(e) => {
                    warn_slow_path_once("draw_proto_masks", "setup_draw_renderbuffer_dma", &e);
                    if let Some(buffer_id) = pbo_buffer_id {
                        self.setup_renderbuffer_from_pbo(dst, dst_fmt, buffer_id)?;
                    } else {
                        self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                    }
                    false
                }
            },
            _ if pbo_buffer_id.is_some() => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_proto_masks: PBO path");
                }
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, pbo_buffer_id.unwrap())?;
                false
            }
            _ => {
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!("draw_proto_masks: non-DMA fallback (memory={memory:?})");
                }
                self.setup_renderbuffer_non_dma(dst, dst_fmt, ResolvedCrop::no_crop())?;
                false
            }
        };

        // Write the base layer of the framebuffer before running mask passes.
        if let Some((bg, bg_fmt)) = background {
            if bg.width() != Some(dst_w) || bg.height() != Some(dst_h) {
                return Err(crate::Error::InvalidShape(
                    "background dimensions do not match dst".into(),
                ));
            }
            if is_dma {
                self.draw_background_into_bound_fbo(bg, bg_fmt)?;
            } else {
                self.upload_pixels_to_render_texture(bg, bg_fmt, dst_w, dst_h)?;
            }
        } else {
            unsafe {
                edgefirst_gl::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            }
        }

        edgefirst_gl::enable(edgefirst_gl::gl::BLEND);
        edgefirst_gl::blend_func_separate(
            edgefirst_gl::gl::SRC_ALPHA,
            edgefirst_gl::gl::ONE_MINUS_SRC_ALPHA,
            edgefirst_gl::gl::ZERO,
            edgefirst_gl::gl::ONE,
        );

        self.set_opacity_uniform(opacity)?;
        self.render_box(dst_w, dst_h, detect, color_mode)?;
        self.render_proto_segmentation(detect, proto_data, color_mode)?;
        edgefirst_gl::disable(edgefirst_gl::gl::BLEND);

        edgefirst_gl::finish();
        if !is_dma {
            let format = match dst_fmt {
                PixelFormat::Rgb => edgefirst_gl::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA,
                _ => unreachable!(),
            };
            if let Some(buffer_id) = pbo_buffer_id {
                self.read_pixels_into_pbo(
                    dst,
                    buffer_id,
                    dst_w,
                    dst_h,
                    format,
                    dst_fmt == PixelFormat::Bgra,
                )?;
            } else {
                self.read_pixels_into_tensor(
                    dst,
                    dst_w,
                    dst_h,
                    format,
                    dst_fmt == PixelFormat::Bgra,
                )?;
            }
        }

        Ok(())
    }

    pub(super) fn check_src_format_supported(
        backend: TransferBackend,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> bool {
        if backend.is_zero_copy() && img.memory() == TensorMemory::DmaBuf {
            // Zero-copy import: RGBA, GREY, YUYV and NV12 through
            // samplerExternalOES, NV16 and NV24 through the R8 texelFetch shader,
            // and packed Rgb where the platform imports it as an R8 view.
            (fmt == PixelFormat::Rgb && Platform::PACKED_R8_IMPORT)
                || matches!(
                    fmt,
                    PixelFormat::Rgba
                        | PixelFormat::Grey
                        | PixelFormat::Yuyv
                        | PixelFormat::Nv12
                        | PixelFormat::Nv16
                        | PixelFormat::Nv24
                )
        } else {
            // Non-DMA: packed RGB(A)/Grey upload via draw_src_texture, plus
            // single-plane NV12/NV16/NV24 via the R8-upload ShaderR8 path. YUYV, and
            // any NV source in a PBO, only where the platform has no zero-copy
            // import.
            let uploads_yuv = !backend.is_zero_copy();
            let nv_single_plane = matches!(
                fmt,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) && !img.is_multiplane();
            matches!(
                fmt,
                PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey
            ) || (fmt == PixelFormat::Yuyv && uploads_yuv)
                || (nv_single_plane && (img.memory() != TensorMemory::Pbo || uploads_yuv))
        }
    }

    /// Whether the GL backend renders a `fmt` destination held in `memory`.
    /// `nv12_dst` is the platform's [`GlPlatform::NV12_DST`].
    pub(crate) fn dst_format_renderable(
        backend: TransferBackend,
        memory: TensorMemory,
        fmt: PixelFormat,
        has_bgra: bool,
        nv12_dst: bool,
    ) -> bool {
        if fmt == PixelFormat::Bgra && !has_bgra {
            return false;
        }
        if backend.is_zero_copy() && memory == TensorMemory::DmaBuf {
            matches!(
                fmt,
                PixelFormat::Rgba
                    | PixelFormat::Bgra
                    | PixelFormat::Grey
                    | PixelFormat::PlanarRgb
                    | PixelFormat::PlanarRgba
                    | PixelFormat::Rgb
            ) || (fmt == PixelFormat::Nv12 && nv12_dst)
        } else {
            // Planar RGB(A) into heap (`Mem`) or `Pbo` memory renders through the
            // two-pass plan (`TwoPassNvPlanar`).
            matches!(
                fmt,
                PixelFormat::Rgb
                    | PixelFormat::Rgba
                    | PixelFormat::Bgra
                    | PixelFormat::Grey
                    | PixelFormat::PlanarRgb
                    | PixelFormat::PlanarRgba
            )
        }
    }

    /// Query GL capabilities and detect GPU vendor/renderer type; see
    /// [`GlSupport`].
    fn gl_check_support() -> Result<GlSupport, crate::Error> {
        if let Ok(version) = edgefirst_gl::get_string(edgefirst_gl::gl::SHADING_LANGUAGE_VERSION) {
            log::debug!("GL Shading Language Version: {version:?}");
        } else {
            log::warn!("Could not get GL Shading Language Version");
        }

        // Detect GPU vendor / software / virtualized renderers via GL_RENDERER.
        let renderer_name = edgefirst_gl::get_string(edgefirst_gl::gl::RENDERER).ok();
        let mut traits = renderer_name
            .as_deref()
            .map(|r| {
                log::info!("GL_RENDERER: {r}");
                classify_renderer(r)
            })
            .unwrap_or_default();
        let renderer_name = renderer_name.unwrap_or_else(|| "unknown".to_string());
        // The Adreno trait applies on Linux only.
        traits.adreno &= cfg!(target_os = "linux");
        let RendererTraits {
            vivante: is_vivante,
            mali: is_mali,
            software: is_software_renderer,
            virtual_gpu: is_virtual_gpu,
            angle: is_angle,
            adreno: is_adreno,
        } = traits;
        if is_vivante {
            log::warn!(
                "Vivante GPU detected — NV12 → planar RGB conversions will use \
                 two-pass workaround to avoid GPU hang"
            );
        }
        if is_mali {
            log::info!(
                "Mali GPU detected — a source DMA-BUF whose plane offset is not \
                 {DMA_IMPORT_OFFSET_ALIGN}-byte aligned will be uploaded \
                 rather than imported: Mali's import samples zeros from such an \
                 offset without reporting an EGL error"
            );
        }
        if is_software_renderer {
            log::warn!(
                "Software OpenGL renderer detected — GPU backend will be disabled. \
                 Image processing will use the CPU backend instead. \
                 Check EGL ICD configuration if a hardware GPU is expected."
            );
        }
        if is_virtual_gpu {
            log::warn!(
                "Virtualized GPU detected — concurrent GL across contexts \
                 mis-renders on paravirtual Metal (observed on macOS CI \
                 runners); processors will serialize per-message (Full \
                 policy, same as Vivante). Override with \
                 EDGEFIRST_GL_SERIALIZE=lifecycle."
            );
        } else if is_angle {
            log::info!(
                "ANGLE detected — a deferred draw does not reliably execute \
                 with the per-draw state its own processor programmed while \
                 another context is active, so processors will serialize \
                 per-message (Full policy). Costs throughput only when one \
                 process runs several ImageProcessors concurrently. Override \
                 with EDGEFIRST_GL_SERIALIZE=lifecycle."
            );
        }
        if is_adreno {
            log::info!(
                "Adreno GPU detected — processors will serialize per-message \
                 (Full policy; override with EDGEFIRST_GL_SERIALIZE=lifecycle), \
                 DMA-BUF imports at a plane offset that is not \
                 {DMA_IMPORT_OFFSET_ALIGN}-byte aligned and two-buffer NV12 \
                 sources take the upload / readback routes"
            );
        }

        let extensions = unsafe {
            let str = edgefirst_gl::gl::GetString(edgefirst_gl::gl::EXTENSIONS);
            if str.is_null() {
                return Err(crate::Error::GLVersion(
                    "GL returned no supported extensions".to_string(),
                ));
            }
            CStr::from_ptr(str as *const c_char)
                .to_string_lossy()
                .to_string()
        };
        log::debug!("GL Extensions: {extensions}");
        // The external-OES extension is required only where the platform binds
        // imports through it.
        let required_ext: &[&str] = if Platform::EXTERNAL_OES {
            &["GL_OES_EGL_image_external_essl3"]
        } else {
            &[]
        };
        let extensions = extensions.split_ascii_whitespace().collect::<BTreeSet<_>>();
        for required in required_ext {
            if !extensions.contains(required) {
                return Err(crate::Error::GLVersion(format!(
                    "GL does not support {required} extension",
                )));
            }
        }

        // EDGEFIRST_GL_NO_FLOAT_LINEAR=1 forces the float-linear capability off.
        let force_no_float_linear = std::env::var("EDGEFIRST_GL_NO_FLOAT_LINEAR")
            .map(|v| v == "1")
            .unwrap_or(false);
        let has_float_linear =
            extensions.contains("GL_OES_texture_float_linear") && !force_no_float_linear;
        if force_no_float_linear {
            log::info!("GL_OES_texture_float_linear forced OFF via EDGEFIRST_GL_NO_FLOAT_LINEAR");
        }
        log::debug!("GL_OES_texture_float_linear: {has_float_linear}");

        let has_bgra = extensions.contains("GL_EXT_texture_format_BGRA8888");
        log::debug!("GL_EXT_texture_format_BGRA8888: {has_bgra}");

        // The float-color-buffer extensions, probed separately, gate F32 and
        // F16 destinations.
        let supports_f32_color = extensions.contains("GL_EXT_color_buffer_float");
        let supports_f16_color = extensions.contains("GL_EXT_color_buffer_half_float");
        log::debug!("GL_EXT_color_buffer_float: {supports_f32_color}");
        log::debug!("GL_EXT_color_buffer_half_float: {supports_f16_color}");

        let has_yuv_target = Platform::EXTERNAL_OES && extensions.contains("GL_EXT_YUV_target");
        log::debug!("GL_EXT_YUV_target: {has_yuv_target}");

        Ok(GlSupport {
            has_float_linear,
            has_yuv_target,
            has_bgra,
            renderer: traits,
            renderer_name,
            supports_f32_color,
            supports_f16_color,
        })
    }

    /// Invalidate EGL binding state on all destination textures.
    fn invalidate_dst_textures(&mut self) {
        self.render_texture.invalidate_egl_binding();
        self.draw_render_texture.invalidate_egl_binding();
    }

    /// Invalidate EGL binding state on all source textures.
    fn invalidate_src_textures(&mut self) {
        self.camera_eglimage_texture.invalidate_egl_binding();
        self.nv_r8_texture.invalidate_egl_binding();
        self.nv_uv_texture.invalidate_egl_binding();
        self.camera_normal_texture.invalidate_egl_binding();
    }

    /// Request the packed-RGB plan's destination import before any rendering:
    /// the same surface geometry and packed format as
    /// `convert_to_packed_rgb`'s pass 2.
    ///
    /// Returns `Ok(())` without importing when the geometry does not resolve.
    fn probe_packed_rgb_dst_import(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
    ) -> crate::Result<()> {
        let (Some(dst_w), Some(dst_h)) = (dst.width(), dst.height()) else {
            return Ok(());
        };
        let Some(layout) = edgefirst_tensor::packed_rgb888_layout(dst_w, dst_h) else {
            return Ok(());
        };
        self.get_or_create_egl_image_rgb(
            dst,
            dst_fmt,
            layout.surface_w,
            layout.surface_h,
            super::platform::PackedImportFormat::Rgba8888,
        )
        .map(|_| ())
    }

    /// Record a zero-copy destination import that the driver refused: count
    /// it in [`ConvertStats::dst_import_fallbacks`], warn once per buffer
    /// naming `then`, the route the convert takes instead, log repeats at
    /// debug, and drain the GL error queue.
    ///
    /// [`ConvertStats::dst_import_fallbacks`]: super::cache::ConvertStats::dst_import_fallbacks
    fn note_dst_import_fallback(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        e: &Error,
        then: &str,
    ) {
        self.convert_stats.dst_import_fallbacks += 1;
        let w = dst.width().unwrap_or(0);
        let h = dst.height().unwrap_or(0);
        let offset = dst.plane_offset().unwrap_or(0);
        if self.dst_import_warned.insert(dst.buffer_identity().id()) {
            log::warn!(
                "zero-copy destination import refused for {dst_fmt} ({w}x{h}) at plane \
                 offset {offset}; {then}: {e}"
            );
        } else {
            log::debug!(
                "zero-copy destination import refused for {dst_fmt} ({w}x{h}) at plane \
                 offset {offset} (repeat): {e}"
            );
        }
        // Reads at most 8 error flags, stopping at the first NO_ERROR.
        unsafe {
            for _ in 0..8 {
                if edgefirst_gl::gl::GetError() == edgefirst_gl::gl::NO_ERROR {
                    break;
                }
            }
        }
    }

    /// Classify and bind the destination render target. The classification
    /// lives in [`super::render::lower_dst`]; this performs the GL work for
    /// `lowering` and returns the target with the effective lowering.
    fn bind_dst(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        crop: ResolvedCrop,
        lowering: super::render::DstLowering,
    ) -> crate::Result<(DstTarget, super::render::DstLowering)> {
        let effective = match lowering {
            super::render::DstLowering::ZeroCopy => {
                // A failed zero-copy setup falls back to the mapped-texture readback.
                match self.setup_renderbuffer_dma(dst, dst_fmt) {
                    Ok(()) => (DstTarget::ZeroCopyImage, lowering),
                    Err(e) if self.probe_dst_import_only => return Err(e),
                    Err(e) => {
                        self.note_dst_import_fallback(dst, dst_fmt, &e, TEXTURE_READBACK);
                        self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;
                        (
                            DstTarget::Texture {
                                readback: DstReadback::Mem,
                            },
                            super::render::DstLowering::TextureMem,
                        )
                    }
                }
            }
            super::render::DstLowering::TexturePbo => {
                let id = dst.pbo_id().ok_or_else(|| {
                    crate::Error::OpenGl(
                        "bind_dst: PBO-lowered destination is not a PBO tensor".to_string(),
                    )
                })?;
                if dst.pbo_is_mapped() == Some(true) {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert to a mapped PBO tensor".to_string(),
                    ));
                }
                self.setup_renderbuffer_from_pbo(dst, dst_fmt, id)?;
                (
                    DstTarget::Texture {
                        readback: DstReadback::Pbo(id),
                    },
                    lowering,
                )
            }
            super::render::DstLowering::TextureMem => {
                self.setup_renderbuffer_non_dma(dst, dst_fmt, crop)?;
                (
                    DstTarget::Texture {
                        readback: DstReadback::Mem,
                    },
                    lowering,
                )
            }
        };
        Ok(effective)
    }

    /// Record on the span how the convert's output reaches the destination.
    fn record_dst_feed(convert_span: &tracing::Span, target: DstTarget) {
        convert_span.record("dst_feed", target.feed());
    }

    fn setup_renderbuffer_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
    ) -> crate::Result<()> {
        self.convert_fbo.bind();
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let (width, height) = if dst_fmt.layout() == PixelLayout::Planar {
            (dst_w as i32, (dst_h * dst_fmt.channels()) as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let dst_key = BufferImportKey::from_tensor(dst, dst_fmt, true);
        let luma_id = dst_key.luma_id;

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst, dst_fmt) {
            Some(rbo) => unsafe {
                edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                edgefirst_gl::gl::FramebufferRenderbuffer(
                    edgefirst_gl::gl::FRAMEBUFFER,
                    edgefirst_gl::gl::COLOR_ATTACHMENT0,
                    edgefirst_gl::gl::RENDERBUFFER,
                    rbo,
                );
                check_gl_error(function!(), line!())?;
            },
            None => unsafe {
                if let Some(p) = &self.texture_program_yuv {
                    edgefirst_gl::gl::UseProgram(p.id);
                }
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.render_texture.id);
                super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::LINEAR);
                if self
                    .render_texture
                    .bind_egl_image(&self.gl_context, dst_key, dest_egl)?
                {
                    edgefirst_gl::gl::FramebufferTexture2D(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::TEXTURE_2D,
                        self.render_texture.id,
                        0,
                    );
                    log::trace!("setup_renderbuffer_dma: bound new dst EGLImage id={luma_id:#x}");
                } else {
                    log::trace!(
                        "setup_renderbuffer_dma: reusing bound dst EGLImage id={luma_id:#x}"
                    );
                }
                check_gl_error(function!(), line!())?;
            },
        }

        // A view()/batch() destination renders into its band of the parent
        // import via `glViewport`; a whole tensor fills its full surface.
        let vp = match dst.view_origin() {
            Some(vo) => super::render::region_to_viewport_top_down(crate::Region::new(
                vo.x, vo.y, dst_w, dst_h,
            )),
            None => super::render::Viewport {
                x: 0,
                y: 0,
                w: width,
                h: height,
            },
        };
        unsafe {
            edgefirst_gl::gl::Viewport(vp.x, vp.y, vp.w, vp.h);
        }
        Ok(())
    }

    /// Variant of [`setup_renderbuffer_dma`] used by the draw operations
    /// (`draw_decoded_masks`, `draw_proto_masks`), with a dedicated FBO
    /// (`draw_fbo`) and texture (`draw_render_texture`).
    fn setup_draw_renderbuffer_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
    ) -> crate::Result<()> {
        self.draw_fbo.bind();
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let (width, height) = if dst_fmt.layout() == PixelLayout::Planar {
            (dst_w as i32, (dst_h * dst_fmt.channels()) as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let dst_key = BufferImportKey::from_tensor(dst, dst_fmt, true);
        let luma_id = dst_key.luma_id;

        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst, dst_fmt)?;
        match self.cached_dst_renderbuffer(dst, dst_fmt) {
            Some(rbo) => unsafe {
                edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                edgefirst_gl::gl::FramebufferRenderbuffer(
                    edgefirst_gl::gl::FRAMEBUFFER,
                    edgefirst_gl::gl::COLOR_ATTACHMENT0,
                    edgefirst_gl::gl::RENDERBUFFER,
                    rbo,
                );
                check_gl_error(function!(), line!())?;
            },
            None => unsafe {
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                edgefirst_gl::gl::BindTexture(
                    edgefirst_gl::gl::TEXTURE_2D,
                    self.draw_render_texture.id,
                );
                super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::LINEAR);
                if self
                    .draw_render_texture
                    .bind_egl_image(&self.gl_context, dst_key, dest_egl)?
                {
                    edgefirst_gl::gl::FramebufferTexture2D(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::TEXTURE_2D,
                        self.draw_render_texture.id,
                        0,
                    );
                    log::trace!(
                        "setup_draw_renderbuffer_dma: bound new dst EGLImage id={luma_id:#x}"
                    );
                } else {
                    log::trace!(
                        "setup_draw_renderbuffer_dma: reusing bound dst EGLImage id={luma_id:#x}"
                    );
                }
                check_gl_error(function!(), line!())?;
            },
        }

        unsafe {
            edgefirst_gl::gl::Viewport(0, 0, width, height);
        }
        Ok(())
    }

    /// The int8 programs, whose `quantize()` uniforms `load_int_quant` sets,
    /// with their `quantize()` uniform locations in [`Self::upload_quant`] order.
    fn collect_int8_programs(&self) -> Vec<(u32, [i32; 6])> {
        let mut ids = vec![
            self.texture_int8_program.id,
            self.yuyv_int8_program_2d.id,
            self.packed_rgba8_int8_program_2d.id,
            self.texture_program_planar_int8_2d.id,
            self.luma_2d_int8_program.id,
        ];
        ids.extend(self.texture_int8_program_yuv.iter().map(|p| p.id));
        ids.extend(self.texture_program_planar_int8.iter().map(|p| p.id));
        // `NvSampling::program`: u8 at even indices, int8 at odd.
        ids.extend(self.nv_r8_programs.iter().skip(1).step_by(2).map(|p| p.id));
        if let Some((programs, _)) = &self.nv_y2y_programs {
            ids.push(programs[1].id);
        }
        ids.extend(self.rgb_r8_programs.iter().skip(1).step_by(2).map(|p| p.id));
        ids.extend(
            self.yuyv_luma_r8_programs
                .iter()
                .skip(1)
                .step_by(2)
                .map(|p| p.id),
        );
        ids.into_iter()
            .map(|id| {
                // SAFETY: on the GL thread with this processor's context
                // current; `id` is a linked program and the names C literals.
                let locs = [
                    c"q_scale",
                    c"q_offset",
                    c"q_lo",
                    c"q_hi",
                    c"q_zero",
                    c"q_wrap",
                ]
                .map(|name| unsafe { edgefirst_gl::gl::GetUniformLocation(id, name.as_ptr()) });
                (id, locs)
            })
            .collect()
    }

    /// Upload `q` to one program's `quantize()` uniforms, with the rounding
    /// and zero point folded in.
    ///
    /// # Safety
    /// On the GL thread with this processor's context current and `id` bound.
    unsafe fn upload_quant(locs: &[i32; 6], q: &IntQuant) {
        let offset = q.offset.map(|o| o + 0.5);
        let lo: [f32; 3] = std::array::from_fn(|c| q.lo[c] - q.zero[c]);
        let hi: [f32; 3] = std::array::from_fn(|c| q.hi[c] - q.zero[c]);
        let zero = q.zero.map(|z| z / 255.0);
        let wrap = q.zero.map(|z| -z - 0.5);
        for (loc, v) in locs.iter().zip([&q.scale, &offset, &lo, &hi, &zero, &wrap]) {
            // SAFETY: the caller's contract.
            unsafe { edgefirst_gl::gl::Uniform3fv(*loc, 1, v.as_ptr()) };
        }
    }

    /// Put the mapped int8 build in the int8 fields when `mapped`, else the
    /// default one, linking the mapped build on first use.
    fn use_int8_set(&mut self, mapped: bool) -> crate::Result<()> {
        if self.int8_mapped == mapped {
            return Ok(());
        }
        let mut alt = match self.int8_alt.take() {
            Some(alt) => alt,
            None => {
                let set = Int8Set::new(true, self.nv_y2y_programs.is_some())?;
                if set.nv_y2y.is_none() && self.nv_y2y_programs.take().is_some() {
                    log::info!(
                        "the mapped int8 Y2Y program did not build; NV filters in the shader"
                    );
                }
                set
            }
        };
        self.swap_int8_set(&mut alt);
        self.int8_alt = Some(alt);
        self.int8_mapped = mapped;
        if mapped && self.int8_programs.is_empty() {
            self.int8_programs = self.collect_int8_programs();
        }
        Ok(())
    }

    /// Exchange the int8 programs and their link-time state with `set`.
    fn swap_int8_set(&mut self, set: &mut Int8Set) {
        use std::mem::swap;
        swap(&mut self.texture_int8_program, &mut set.texture);
        swap(&mut self.texture_int8_program_yuv, &mut set.texture_yuv);
        swap(&mut self.texture_program_planar_int8, &mut set.planar);
        swap(&mut self.texture_program_planar_int8_2d, &mut set.planar_2d);
        swap(&mut self.yuyv_int8_program_2d, &mut set.yuyv_2d);
        swap(&mut self.yuyv_2d_locs[1], &mut set.yuyv_2d_locs);
        swap(
            &mut self.packed_rgba8_int8_program_2d,
            &mut set.packed_rgba8_2d,
        );
        for (i, (program, uniforms)) in set
            .nv_r8
            .iter_mut()
            .zip(set.nv_r8_uniforms.iter_mut())
            .enumerate()
        {
            swap(&mut self.nv_r8_programs[2 * i + 1], program);
            swap(&mut self.nv_r8_uniforms[2 * i + 1], uniforms);
        }
        for (i, (program, locs)) in set
            .rgb_r8
            .iter_mut()
            .zip(set.rgb_r8_locs.iter_mut())
            .enumerate()
        {
            swap(&mut self.rgb_r8_programs[2 * i + 1], program);
            swap(&mut self.rgb_r8_locs[2 * i + 1], locs);
        }
        for (i, (program, locs)) in set
            .yuyv_luma_r8
            .iter_mut()
            .zip(set.yuyv_luma_r8_locs.iter_mut())
            .enumerate()
        {
            swap(&mut self.yuyv_luma_r8_programs[2 * i + 1], program);
            swap(&mut self.yuyv_luma_r8_locs[2 * i + 1], locs);
        }
        swap(&mut self.luma_2d_int8_program, &mut set.luma_2d);
        if let (Some((programs, uniforms)), Some((program, u))) =
            (self.nv_y2y_programs.as_mut(), set.nv_y2y.as_mut())
        {
            swap(&mut programs[1], program);
            swap(&mut uniforms[1], u);
        }
        swap(
            &mut self.texture_src_extent_locs[1],
            &mut set.texture_src_extent_loc,
        );
        swap(
            &mut self.external_src_extent_locs[1],
            &mut set.yuv_src_extent_loc,
        );
        swap(
            &mut self.external_src_extent_locs[3],
            &mut set.planar_src_extent_loc,
        );
    }

    /// Upload `q` to every mapped int8 program, unless they already hold it.
    fn load_int_quant(&mut self, q: IntQuant) -> crate::Result<()> {
        if self.int_quant == Some(q) {
            return Ok(());
        }
        // SAFETY: on the GL thread with this processor's context current; the
        // ids are programs this processor linked and still owns.
        unsafe {
            let mut current = 0i32;
            edgefirst_gl::gl::GetIntegerv(edgefirst_gl::gl::CURRENT_PROGRAM, &mut current);
            for (id, locs) in &self.int8_programs {
                edgefirst_gl::gl::UseProgram(*id);
                Self::upload_quant(locs, &q);
            }
            edgefirst_gl::gl::UseProgram(current as u32);
            check_gl_error(function!(), line!())?;
        }
        self.int_quant = Some(q);
        Ok(())
    }

    /// Point a mapped int8 `program`'s `quantize()` at one plane's colour
    /// channel, for the planar draws that swizzle each plane's channel into
    /// `.r`; `None` restores the loaded per-channel uniforms. A no-op for
    /// other programs and for coefficients equal across channels.
    ///
    /// # Safety
    /// On the GL thread with this processor's context current and `program`
    /// bound.
    unsafe fn set_plane_quant(&self, program: u32, plane: Option<usize>) {
        let Some(q) = self.int_quant else {
            return;
        };
        let same = |v: &[f32; 3]| v.iter().all(|&x| x == v[0]);
        if [&q.scale, &q.offset, &q.zero, &q.lo, &q.hi]
            .into_iter()
            .all(same)
        {
            return;
        }
        let Some((_, locs)) = self.int8_programs.iter().find(|(p, _)| *p == program) else {
            return;
        };
        let q = match plane {
            Some(p) if p < 3 => IntQuant {
                scale: [q.scale[p]; 3],
                offset: [q.offset[p]; 3],
                zero: [q.zero[p]; 3],
                lo: [q.lo[p]; 3],
                hi: [q.hi[p]; 3],
            },
            _ => q,
        };
        // SAFETY: the caller's contract; `program` is bound.
        unsafe { Self::upload_quant(locs, &q) };
    }

    /// The letterbox clear colour for integer destinations: the mapped pad of
    /// the current convert, else the plain XOR 0x80 for int8. No-op when there
    /// is no fill colour.
    fn int8_bias_clear(&self, is_int8: bool, mut crop: ResolvedCrop) -> ResolvedCrop {
        if let (Some(color), Some(pad)) = (crop.dst_color.as_mut(), self.int_pad) {
            *color = pad;
        } else if is_int8 {
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
        }
        crop
    }

    /// Render `src` into the already-bound destination FBO as `dst_fmt`,
    /// dispatching packed vs planar; the draws select their int8 program
    /// variants from `is_int8` at draw time.
    #[allow(clippy::too_many_arguments)]
    fn render_packed_or_planar(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        band: Option<edgefirst_tensor::ViewOrigin>,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if dst_fmt.layout() == PixelLayout::Planar {
            self.convert_to_planar(
                src,
                src_fmt,
                dst,
                dst_fmt,
                is_int8,
                rotation,
                flip,
                crop,
                convert_span,
            )
        } else {
            self.convert_to(
                src,
                src_fmt,
                dst,
                dst_fmt,
                band,
                is_int8,
                rotation,
                flip,
                crop,
                convert_span,
            )
        }
    }

    /// `glReadPixels` the bound read-framebuffer into `dst`'s own mapping,
    /// placing each row at the destination's row pitch and undoing the BGRA
    /// channel order when `swap_rb`. Nothing is written between the rows; see
    /// [`read_pixels_into`].
    fn read_pixels_into_tensor(
        &mut self,
        dst: &mut Tensor<u8>,
        width: usize,
        rows: usize,
        format: u32,
        swap_rb: bool,
    ) -> crate::Result<()> {
        let tight_row = width * pack_channels(format);
        let stride = dst.effective_row_stride().unwrap_or(tight_row);
        // Refuses a pitch shorter than one row.
        if stride < tight_row {
            return Err(crate::Error::OpenGl(format!(
                "destination row pitch is {stride} B but one {width}-pixel row \
                 of this readback format is {tight_row} B"
            )));
        }
        let mut dst_map = dst.map_for_write()?;
        let out = dst_map.as_mut_slice();
        // Checks that the last row's end lies within `out`.
        let needed = padded_readback_bytes(rows, tight_row, stride).unwrap_or(rows * tight_row);
        if out.len() < needed {
            return Err(crate::Error::OpenGl(format!(
                "destination maps {} B, needs {needed} B at a {stride} B pitch",
                out.len()
            )));
        }
        // SAFETY: called on the GL thread with a complete read framebuffer
        // bound; the rows fit `out` at this pitch by the check above.
        unsafe {
            edgefirst_gl::gl::ReadBuffer(edgefirst_gl::gl::COLOR_ATTACHMENT0);
            read_pixels_into(width, rows, format, &mut self.readback_scratch, out, stride);
        }
        check_gl_error(function!(), line!())?;
        if swap_rb {
            // Swaps row by row, skipping the padding between rows.
            for y in 0..rows {
                for chunk in out[y * stride..y * stride + tight_row]
                    .as_chunks_mut::<4>()
                    .0
                {
                    chunk.swap(0, 2);
                }
            }
        }
        Ok(())
    }

    /// `glReadPixels` the bound read-framebuffer into the PBO behind `dst`,
    /// placing each row at the destination's row pitch and undoing the BGRA
    /// channel order when `swap_rb`.
    ///
    /// Where the pitch is a whole number of pixels GL lays the rows out
    /// through `GL_PACK_ROW_LENGTH`; a pitch no pixel size divides is read
    /// tight and spread in a mapping; a format without a direct read is read
    /// as RGBA into the scratch and repacked row by row at the pitch.
    fn read_pixels_into_pbo(
        &mut self,
        dst: &Tensor<u8>,
        buffer_id: u32,
        width: usize,
        rows: usize,
        format: u32,
        swap_rb: bool,
    ) -> crate::Result<()> {
        let channels = pack_channels(format);
        let PboReadbackLayout {
            stride,
            tight_row,
            needed,
            row_length,
        } = plan_pbo_readback(
            dst.effective_row_stride().unwrap_or(width * channels),
            dst.view_origin().is_some(),
            dst.plane_offset().unwrap_or(0),
            dst.capacity_bytes(),
            width,
            rows,
            channels,
        )?;
        // Set when the rows were read tight into a padded destination.
        let mut spread = false;
        // SAFETY: called on the GL thread with a complete read framebuffer
        // bound; every span written lies within `needed`, checked above.
        unsafe {
            edgefirst_gl::gl::ReadBuffer(edgefirst_gl::gl::COLOR_ATTACHMENT0);
            if direct_read_supported(format) {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, buffer_id);
                if let Some(pixels) = row_length {
                    edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, pixels);
                }
                edgefirst_gl::gl::ReadPixels(
                    0,
                    0,
                    width as i32,
                    rows as i32,
                    format,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    std::ptr::null_mut(),
                );
                if row_length.is_some() {
                    edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::PACK_ROW_LENGTH, 0);
                }
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_PACK_BUFFER, 0);
                edgefirst_gl::gl::Finish();
                spread = stride > tight_row && row_length.is_none();
            } else {
                // Reads RGBA into the scratch and repacks each row's leading channels
                // into the mapped PBO at the pitch.
                read_pixels_rgba_scratch(width, rows, &mut self.readback_scratch);
                let scratch = &self.readback_scratch;
                // A padded destination is mapped read-write; a tight one with the
                // invalidating map.
                let access = if stride > tight_row {
                    edgefirst_gl::gl::MAP_READ_BIT | edgefirst_gl::gl::MAP_WRITE_BIT
                } else {
                    edgefirst_gl::gl::MAP_WRITE_BIT | edgefirst_gl::gl::MAP_INVALIDATE_RANGE_BIT
                };
                with_mapped_pbo(buffer_id, needed, access, "PBO readback repack", |out| {
                    for (y, row) in out.chunks_mut(stride).take(rows).enumerate() {
                        let pixels = scratch[y * width * 4..(y + 1) * width * 4]
                            .as_chunks::<4>()
                            .0;
                        for (px, dst_px) in pixels
                            .iter()
                            .zip(row[..tight_row].chunks_exact_mut(channels))
                        {
                            dst_px.copy_from_slice(&px[..channels]);
                        }
                    }
                })?;
            }
        }
        check_gl_error(function!(), line!())?;
        if spread || swap_rb {
            // SAFETY: GL thread, the read above has completed, and `needed`
            // lies within the allocation.
            unsafe {
                with_mapped_pbo(
                    buffer_id,
                    needed,
                    edgefirst_gl::gl::MAP_READ_BIT | edgefirst_gl::gl::MAP_WRITE_BIT,
                    "PBO readback row placement",
                    |buf| {
                        if spread {
                            spread_rows(buf, rows, tight_row, stride);
                        }
                        if swap_rb {
                            // Swaps row by row, skipping the padding between rows.
                            for y in 0..rows {
                                for chunk in buf[y * stride..y * stride + tight_row]
                                    .as_chunks_mut::<4>()
                                    .0
                                {
                                    chunk.swap(0, 2);
                                }
                            }
                        }
                    },
                )?;
            }
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Read the rendered FBO colour attachment (`COLOR_ATTACHMENT0`) into the
    /// destination, applying the BGRA byte-swap when `dst_fmt` is BGRA. `pbo_id`
    /// selects the target: `None` reads into the mapped Mem tensor; `Some(id)`
    /// reads into the PBO.
    fn readback_rendered(
        &mut self,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        pbo_id: Option<u32>,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let dest_format = match dst_fmt {
            PixelFormat::Rgb => edgefirst_gl::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA,
            PixelFormat::Grey => edgefirst_gl::gl::RED,
            // A single-channel R8 target: full dst_w, dst_h stacked `channels`
            // times, which is the destination's CHW byte layout.
            PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => edgefirst_gl::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "GL readback not supported for {dst_fmt}"
                )))
            }
        };
        let read_h = if dst_fmt.layout() == PixelLayout::Planar {
            dst_h * dst_fmt.channels()
        } else {
            dst_h
        };
        match pbo_id {
            None => self.read_pixels_into_tensor(
                dst,
                dst_w,
                read_h,
                dest_format,
                dst_fmt == PixelFormat::Bgra,
            ),
            Some(buffer_id) => self.read_pixels_into_pbo(
                dst,
                buffer_id,
                dst_w,
                read_h,
                dest_format,
                dst_fmt == PixelFormat::Bgra,
            ),
        }
    }

    fn setup_renderbuffer_non_dma(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        crop: ResolvedCrop,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let is_planar = dst_fmt.layout() == PixelLayout::Planar;

        let (width, height) = if is_planar {
            // Full dst_w, with dst_h stacked `channels` times in a single-channel
            // R8 target.
            let height = match dst_fmt.channels() {
                4 => dst_h * 4,
                3 => dst_h * 3,
                1 => dst_h,
                _ => unreachable!(),
            };
            (dst_w as i32, height as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        // A BGRA destination renders into an RGBA texture: its pixel data is
        // byte-swapped (R↔B) before upload, and the readback swaps back.
        let is_bgra = !is_planar && dst_fmt == PixelFormat::Bgra;
        if is_bgra && !self.bgra_warned {
            log::warn!(
                "BGRA destination: using RGBA internal format with CPU R↔B byte-swap workaround"
            );
            self.bgra_warned = true;
        }
        let format = if is_planar {
            edgefirst_gl::gl::RED
        } else {
            match dst_fmt {
                PixelFormat::Rgb => edgefirst_gl::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA, // BGRA uses RGBA internally
                PixelFormat::Grey => edgefirst_gl::gl::RED,
                _ => unreachable!(),
            }
        };

        let start = Instant::now();
        self.convert_fbo.bind();

        let map;
        let mut swapped_buf;

        // Seeds the render target with the destination's current pixels when a
        // partial destination rectangle has no fill colour.
        let seeded = crop.dst_color.is_none()
            && crop.dst_rect.is_some_and(|c| {
                c.top != 0 || c.left != 0 || c.height != dst_h || c.width != dst_w
            });
        let pixels = if !seeded {
            std::ptr::null()
        } else {
            map = dst.map_read()?;
            if is_bgra {
                // Swap R↔B to convert BGRA→RGBA for the RGBA texture.
                swapped_buf = map.as_slice().to_vec();
                for chunk in swapped_buf.as_chunks_mut::<4>().0 {
                    chunk.swap(0, 2);
                }
                swapped_buf.as_ptr() as *const c_void
            } else {
                map.as_ptr() as *const c_void
            }
        };
        unsafe {
            edgefirst_gl::gl::UseProgram(self.texture_program.id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.render_texture.id);
            super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::LINEAR);

            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                self.render_internal_format(format),
                width,
                height,
                0,
                format,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                pixels,
            );
            // Drops the EGLImage binding the upload replaced.
            self.render_texture.invalidate_egl_binding();
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::Viewport(0, 0, width, height);
        }
        log::debug!("Set up framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    /// Draw `bg` as the base layer of the bound zero-copy destination
    /// framebuffer, with blending off. Where external-OES sampling exists a
    /// zero-copy background is blitted through its EGLImage; otherwise it
    /// goes through the 2D source texture (`draw_src_texture`).
    fn draw_background_into_bound_fbo(
        &mut self,
        bg: &Tensor<u8>,
        bg_fmt: PixelFormat,
    ) -> crate::Result<()> {
        edgefirst_gl::disable(edgefirst_gl::gl::BLEND);
        let full_src = RegionOfInterest {
            left: 0.0,
            top: 1.0,
            right: 1.0,
            bottom: 0.0,
        };
        let full_dst = RegionOfInterest {
            left: -1.0,
            top: 1.0,
            right: 1.0,
            bottom: -1.0,
        };
        if Platform::EXTERNAL_OES && bg.memory() == TensorMemory::DmaBuf {
            let bg_egl = self.get_or_create_egl_image(CacheKind::Src, bg, bg_fmt)?;
            self.draw_camera_texture_eglimage(
                bg,
                bg_fmt,
                bg_egl,
                full_src,
                full_dst,
                0,
                crate::Flip::None,
                false,
                None,
            )
        } else {
            self.draw_src_texture(
                bg,
                bg_fmt,
                full_src,
                full_dst,
                0,
                crate::Flip::None,
                false,
                false,
                &tracing::Span::none(),
            )
        }
    }

    /// Upload CPU-side pixel data into `self.render_texture` and attach it as
    /// the FBO color attachment. A BGRA source is byte-swapped (R↔B) into a
    /// scratch buffer first.
    fn upload_pixels_to_render_texture(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst_w: usize,
        dst_h: usize,
    ) -> crate::Result<()> {
        use edgefirst_tensor::TensorMapTrait;
        let map = src.map_read()?;
        let src_slice = map.as_slice();
        let format = match src_fmt {
            PixelFormat::Rgb => edgefirst_gl::gl::RGB,
            PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "non-DMA bg upload not supported for {src_fmt}",
                )))
            }
        };
        let swapped: Option<Vec<u8>> = if src_fmt == PixelFormat::Bgra {
            let mut v = src_slice.to_vec();
            for chunk in v.as_chunks_mut::<4>().0 {
                chunk.swap(0, 2);
            }
            Some(v)
        } else {
            None
        };
        let pixels = swapped.as_deref().unwrap_or(src_slice).as_ptr() as *const c_void;

        unsafe {
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.render_texture.id);
            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                format as i32,
                dst_w as i32,
                dst_h as i32,
                0,
                format,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                pixels,
            );
            // Drops the EGLImage binding the upload replaced.
            self.render_texture.invalidate_egl_binding();
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Low-level PBO render-target setup with explicit GL format and type.
    ///
    /// Binds `buffer_id` as `GL_PIXEL_UNPACK_BUFFER`, calls `TexImage2D` with
    /// the provided `internal_format`, `client_format`, and `gl_type`, then
    /// attaches the texture to the FBO color attachment and sets the viewport.
    ///
    /// The `(width, height)` are the pixel dimensions of the render target.
    /// `row_length` is the destination's pitch in pixels, when it has one.
    ///
    /// # Safety (caller responsibility)
    /// `internal_format`, `client_format`, and `gl_type` must form a valid
    /// `TexImage2D` combination for the bound GLES context.
    #[allow(clippy::too_many_arguments)]
    fn setup_renderbuffer_from_pbo_inner(
        &mut self,
        width: i32,
        height: i32,
        buffer_id: u32,
        internal_format: u32,
        client_format: u32,
        gl_type: u32,
        row_length: Option<i32>,
    ) -> crate::Result<()> {
        self.convert_fbo.bind();
        unsafe {
            edgefirst_gl::gl::UseProgram(self.texture_program.id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.render_texture.id);
            super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::LINEAR);

            // Uploads the PBO's content to the render texture: with the PBO bound
            // as the UNPACK buffer, TexImage2D reads from it at offset 0.
            // `row_length` is the destination's pitch in pixels.
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, buffer_id);
            if let Some(px) = row_length {
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, px);
            }
            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                internal_format as i32,
                width,
                height,
                0,
                client_format,
                gl_type,
                std::ptr::null(),
            );
            if row_length.is_some() {
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, 0);
            }
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, 0);
            // Drops the EGLImage binding the upload replaced.
            self.render_texture.invalidate_egl_binding();

            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::Viewport(0, 0, width, height);
        }
        Ok(())
    }

    /// Set up a u8 (UNSIGNED_BYTE) PBO render target for the given format.
    ///
    /// Computes the GL `format` from `dst_fmt` (including the planar layout)
    /// and delegates to [`setup_renderbuffer_from_pbo_inner`] with
    /// `UNSIGNED_BYTE` as the GL type.
    fn setup_renderbuffer_from_pbo(
        &mut self,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        buffer_id: u32,
    ) -> crate::Result<()> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let is_planar = dst_fmt.layout() == PixelLayout::Planar;

        let (width, height) = if is_planar {
            // Full dst_w, with dst_h stacked `channels` times.
            let height = match dst_fmt.channels() {
                4 => dst_h * 4,
                3 => dst_h * 3,
                1 => dst_h,
                _ => unreachable!(),
            };
            (dst_w as i32, height as i32)
        } else {
            (dst_w as i32, dst_h as i32)
        };

        let format = if is_planar {
            edgefirst_gl::gl::RED
        } else {
            match dst_fmt {
                PixelFormat::Rgb => edgefirst_gl::gl::RGB,
                PixelFormat::Rgba | PixelFormat::Bgra => edgefirst_gl::gl::RGBA,
                PixelFormat::Grey => edgefirst_gl::gl::RED,
                _ => {
                    return Err(crate::Error::NotSupported(format!(
                        "PBO renderbuffer not supported for {dst_fmt}",
                    )))
                }
            }
        };

        // The row length of the destination's current bytes for the seed upload:
        // `None` for a tight pitch and for one that is not a whole number of
        // pixels.
        let px_bytes = match format {
            edgefirst_gl::gl::RGBA => 4,
            edgefirst_gl::gl::RGB => 3,
            _ => 1, // RED: Grey, and every plane of a planar destination
        };
        let tight_row = width as usize * px_bytes;
        let row_length = dst
            .effective_row_stride()
            .and_then(|stride| pack_row_length(tight_row, stride, px_bytes));

        self.setup_renderbuffer_from_pbo_inner(
            width,
            height,
            buffer_id,
            self.render_internal_format(format) as u32,
            format,
            edgefirst_gl::gl::UNSIGNED_BYTE,
            row_length,
        )
    }

    /// Pick the NV* GPU conversion path for `src`/`src_fmt`, honoring the
    /// `EDGEFIRST_NV_CONVERT_PATH` preference. Returns the path to attempt.
    /// Forcing an unavailable path logs a warning and returns the viable one.
    ///
    /// Capability:
    /// - [`NvConvertPath::ShaderR8`] needs a single combined-plane buffer: it
    ///   covers single-plane NV12/NV16/NV24 but not true-multiplane NV12.
    /// - [`NvConvertPath::ExternalSampler`] (samplerExternalOES) covers NV12
    ///   only (incl. multiplane).
    ///
    /// `render_fmt` is the format of the render target this draw writes. A
    /// Grey render target always takes `ShaderR8`.
    fn select_nv_path(
        &self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        render_fmt: PixelFormat,
    ) -> NvConvertPath {
        // `ShaderR8` applies to single-plane NV12/NV16/NV24; every other source
        // takes `ExternalSampler`.
        let shader_capable = matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) && !src.is_multiplane();
        if !shader_capable {
            return NvConvertPath::ExternalSampler;
        }
        // A Grey target takes `ShaderR8`; a forced sampler is refused.
        if render_fmt == PixelFormat::Grey {
            if self.nv_path_pref == NvPathPref::ForceSampler {
                log::warn!(
                    "EDGEFIRST_NV_CONVERT_PATH=sampler refused for {src_fmt:?}→Grey \
                     (the sampler returns RGB, not the source's luma); using shader"
                );
            }
            return NvConvertPath::ShaderR8;
        }
        // Single-plane NV12/NV16/NV24 from here; only NV12 also has an
        // ExternalSampler path.
        match self.nv_path_pref {
            NvPathPref::ForceShader => NvConvertPath::ShaderR8,
            NvPathPref::ForceSampler => {
                if src_fmt == PixelFormat::Nv12 {
                    NvConvertPath::ExternalSampler
                } else {
                    log::warn!(
                        "EDGEFIRST_NV_CONVERT_PATH=sampler unavailable for {src_fmt:?} \
                         (no external-sampler path); using shader"
                    );
                    NvConvertPath::ShaderR8
                }
            }
            // Auto: `ShaderR8`, except on Vivante for single-plane 4-aligned NV12,
            // where the pick follows the colorimetry mode:
            //
            // * `ColorimetryMode::Fast`: the sampler for every colorimetry.
            // * `ColorimetryMode::Exact`: the sampler only when the source resolves
            //   to BT.601 limited range.
            NvPathPref::Auto => {
                let vivante_sampler_usable = src_fmt == PixelFormat::Nv12
                    && self.is_vivante
                    && src.width().is_some_and(|w| w.is_multiple_of(4));
                let take_sampler = vivante_sampler_usable
                    && match self.colorimetry_mode {
                        crate::ColorimetryMode::Fast => true,
                        crate::ColorimetryMode::Exact => {
                            let cm = crate::colorimetry::resolve_colorimetry(
                                src.colorimetry(),
                                src.height(),
                            );
                            cm.encoding == Some(edgefirst_tensor::ColorEncoding::Bt601)
                                && cm.range == Some(edgefirst_tensor::ColorRange::Limited)
                        }
                    };
                if take_sampler {
                    NvConvertPath::ExternalSampler
                } else {
                    NvConvertPath::ShaderR8
                }
            }
        }
    }

    /// Render `src` to the currently-bound FBO at `dst`'s dimensions; see
    /// `convert_to_dims`.
    ///
    /// `band` is the destination tile's origin inside the bound render
    /// target. It is `Some` only when that target is the shared parent
    /// import a `view()`/`batch()` destination is placed in.
    #[allow(clippy::too_many_arguments)]
    fn convert_to(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &Tensor<u8>,
        _dst_fmt: PixelFormat,
        band: Option<edgefirst_tensor::ViewOrigin>,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> Result<(), crate::Error> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        self.convert_to_dims(
            src,
            src_fmt,
            dst_w,
            dst_h,
            band,
            _dst_fmt,
            is_int8,
            rotation,
            flip,
            crop,
            convert_span,
        )
    }

    /// The body of [`convert_to`], taking the destination geometry directly:
    /// renders `src` to the currently-bound FBO. `dst_fmt` is the logical
    /// destination format, which picks the NV path and, for a Grey
    /// destination, the luma-only draw of a YUV source; the render target
    /// format is whatever the bound FBO holds.
    #[allow(clippy::too_many_arguments)]
    fn convert_to_dims(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst_w: usize,
        dst_h: usize,
        dst_view_origin: Option<edgefirst_tensor::ViewOrigin>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> Result<(), crate::Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        check_gl_error(function!(), line!())?;

        // Confines the letterbox `glClear` and the draw of a view()/batch()
        // destination to its band with `glScissor`. The guard disables
        // `SCISSOR_TEST` on every exit.
        struct ScissorGuard(bool);
        impl Drop for ScissorGuard {
            fn drop(&mut self) {
                if self.0 {
                    unsafe { edgefirst_gl::gl::Disable(edgefirst_gl::gl::SCISSOR_TEST) };
                }
            }
        }
        // The band rect via `region_to_viewport_top_down`.
        let _scissor = match dst_view_origin {
            Some(vo) => {
                let vp = super::render::region_to_viewport_top_down(crate::Region::new(
                    vo.x, vo.y, dst_w, dst_h,
                ));
                unsafe {
                    edgefirst_gl::gl::Scissor(vp.x, vp.y, vp.w, vp.h);
                    edgefirst_gl::gl::Enable(edgefirst_gl::gl::SCISSOR_TEST);
                }
                ScissorGuard(true)
            }
            None => ScissorGuard(false),
        };

        let has_crop = crop
            .dst_rect
            .is_some_and(|x| x.left != 0 || x.top != 0 || x.width != dst_w || x.height != dst_h);
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                unsafe {
                    edgefirst_gl::gl::ClearColor(
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    );
                    edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
                };
            }
        }

        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest::from_crop(&crop, src_w, src_h)
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst_w as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };
        let nv_rect = NvSampleRect::new(
            crop.src_rect,
            crop.dst_rect,
            (src_w, src_h),
            (dst_w, dst_h),
            rotation,
        );
        let drawn = self.draw_source(
            src,
            src_fmt,
            dst_fmt,
            src_roi,
            nv_rect,
            dst_roi,
            rotation_offset,
            flip,
            is_int8,
            convert_span,
        );

        // Finishes unless the batch defers it; a failed draw always finishes.
        if !self.defer_finish || drawn.is_err() {
            let start = Instant::now();
            unsafe { edgefirst_gl::gl::Finish() };
            log::debug!("gl_Finish takes {:?}", start.elapsed());
        }
        drawn?;
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Draw `src` into the bound FBO through its source route: the packed
    /// `Rgb` import, an NV path ([`Self::select_nv_path`]), the
    /// external-sampler import, or an upload. Queues GL work only; the caller
    /// drains it.
    #[allow(clippy::too_many_arguments)]
    fn draw_source(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
        src_roi: RegionOfInterest,
        nv_rect: NvSampleRect,
        dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        convert_span: &tracing::Span,
    ) -> Result<(), crate::Error> {
        // A YUV source into a Grey target draws its range-expanded luma.
        let luma_only = dst_fmt == PixelFormat::Grey && src_fmt.is_yuv();
        let zero_copy_dma =
            self.gl_context.transfer_backend.is_zero_copy() && src.memory() == TensorMemory::DmaBuf;
        if luma_only && src.is_multiplane() {
            return Err(Error::NotSupported(
                "a multiplane NV source has no luma-only GL route to a Grey destination; \
                 converting on the CPU"
                    .into(),
            ));
        }
        if luma_only && src_fmt == PixelFormat::Yuyv && zero_copy_dma && Platform::EXTERNAL_OES {
            return self.draw_yuyv_luma_dma_source(
                src,
                nv_rect,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
                convert_span,
            );
        }
        if zero_copy_dma && src_fmt == PixelFormat::Rgb {
            self.draw_rgb_dma_source(
                src,
                src_roi,
                nv_rect,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
                convert_span,
            )?;
        } else if zero_copy_dma {
            // Choose the NV* path (ShaderR8 vs ExternalSampler); see
            // `select_nv_path`.
            let chosen = self.select_nv_path(src, src_fmt, dst_fmt);

            if !Platform::EXTERNAL_OES && chosen == NvConvertPath::ExternalSampler {
                // No samplerExternalOES on this platform: the source attaches as
                // TEXTURE_2D inside draw_src_texture.
                self.draw_src_texture(
                    src,
                    src_fmt,
                    src_roi,
                    dst_roi,
                    rotation_offset,
                    flip,
                    is_int8,
                    luma_only,
                    convert_span,
                )?;
            } else if chosen == NvConvertPath::ShaderR8 {
                match self.get_or_create_nv_r8_egl_image(src, src_fmt) {
                    Ok(r8_egl) => {
                        tracing::trace!(
                            path = "ShaderR8",
                            src_fmt = ?src_fmt,
                            "image.convert.gl.nv_path"
                        );
                        self.convert_stats.src_imports += 1;
                        convert_span.record("src_feed", "import");
                        // Records the path from the draw's result.
                        match self.draw_nv_texture_2d(
                            src,
                            src_fmt,
                            Some(r8_egl),
                            nv_rect,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                            luma_only,
                            convert_span,
                        ) {
                            Ok(()) => self.last_nv_convert_path = NvConvertPath::ShaderR8,
                            Err(e) => {
                                self.last_nv_convert_path = NvConvertPath::Cpu;
                                return Err(e);
                            }
                        }
                    }
                    Err(e) => {
                        let src_w = src.width().unwrap_or(0);
                        let src_h = src.height().unwrap_or(0);
                        self.convert_stats.zero_copy_declines += 1;
                        // Warns once per buffer; repeats log at debug.
                        if self.nv_import_warned.insert(src.buffer_identity().id()) {
                            log::warn!(
                                "R8 EGLImage creation failed for {src_fmt} \
                                 ({src_w}x{src_h}); uploading the combined plane instead: {e}"
                            );
                        } else {
                            log::debug!(
                                "R8 EGLImage creation failed for {src_fmt} \
                                 ({src_w}x{src_h}) (repeat): {e}"
                            );
                        }
                        // The same shader on an R8 upload of the combined plane.
                        let start = Instant::now();
                        // Records ShaderR8 when the upload draws and Cpu when it fails.
                        match self.draw_nv_texture_2d(
                            src,
                            src_fmt,
                            None,
                            nv_rect,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                            luma_only,
                            convert_span,
                        ) {
                            Ok(()) => self.last_nv_convert_path = NvConvertPath::ShaderR8,
                            Err(e) => {
                                self.last_nv_convert_path = NvConvertPath::Cpu;
                                return Err(e);
                            }
                        }
                        log::debug!("NV R8 upload takes {:?}", start.elapsed());
                    }
                }
            } else {
                // ExternalSampler path (samplerExternalOES): NV12 (incl.
                // multiplane) and all non-shader DMA formats (RGBA/Grey/YUYV).
                match self.get_or_create_egl_image(CacheKind::Src, src, src_fmt) {
                    Ok(src_egl) => {
                        if src_fmt == PixelFormat::Nv12 {
                            tracing::trace!(path = "ExternalSampler", src_fmt = ?src_fmt, "image.convert.gl.nv_path");
                        }
                        self.convert_stats.src_imports += 1;
                        convert_span.record("src_feed", "import");
                        // Records the path from the draw's result.
                        match self.draw_camera_texture_eglimage(
                            src,
                            src_fmt,
                            src_egl,
                            src_roi,
                            dst_roi,
                            rotation_offset,
                            flip,
                            is_int8,
                            None,
                        ) {
                            Ok(()) => {
                                if src_fmt == PixelFormat::Nv12 {
                                    self.last_nv_convert_path = NvConvertPath::ExternalSampler;
                                }
                            }
                            Err(e) => {
                                if src_fmt == PixelFormat::Nv12 {
                                    self.last_nv_convert_path = NvConvertPath::Cpu;
                                }
                                return Err(e);
                            }
                        }
                    }
                    Err(e) => {
                        let src_w = src.width().unwrap_or(0);
                        let src_h = src.height().unwrap_or(0);
                        // A single-plane NV source falls back to the R8 upload of the
                        // combined plane when the sampler import is refused.
                        let nv_upload_capable = matches!(
                            src_fmt,
                            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
                        ) && !src.is_multiplane();
                        if src_fmt == PixelFormat::Nv12 && !nv_upload_capable {
                            self.last_nv_convert_path = NvConvertPath::Cpu;
                        }
                        self.convert_stats.zero_copy_declines += 1;
                        // Warns once per buffer; repeats log at debug.
                        if self.nv_import_warned.insert(src.buffer_identity().id()) {
                            log::warn!(
                                "EGL image creation failed for {src_fmt} ({src_w}x{src_h}), \
                                 falling back to texture upload (slower): {e}"
                            );
                        } else {
                            log::debug!(
                                "EGL image creation failed for {src_fmt} ({src_w}x{src_h}) \
                                 (repeat): {e}"
                            );
                        }
                        let start = Instant::now();
                        if nv_upload_capable {
                            // Records ShaderR8 when the upload draws and Cpu when it fails.
                            match self.draw_nv_texture_2d(
                                src,
                                src_fmt,
                                None,
                                nv_rect,
                                dst_roi,
                                rotation_offset,
                                flip,
                                is_int8,
                                luma_only,
                                convert_span,
                            ) {
                                Ok(()) => {
                                    if src_fmt == PixelFormat::Nv12 {
                                        self.last_nv_convert_path = NvConvertPath::ShaderR8;
                                    }
                                }
                                Err(e) => {
                                    if src_fmt == PixelFormat::Nv12 {
                                        self.last_nv_convert_path = NvConvertPath::Cpu;
                                    }
                                    return Err(e);
                                }
                            }
                            log::debug!("NV R8 upload takes {:?}", start.elapsed());
                        } else if matches!(
                            src_fmt,
                            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
                        ) {
                            // A multiplane NV source has no upload route.
                            return Err(e);
                        } else {
                            self.draw_src_texture(
                                src,
                                src_fmt,
                                src_roi,
                                dst_roi,
                                rotation_offset,
                                flip,
                                is_int8,
                                luma_only,
                                convert_span,
                            )?;
                            log::debug!("draw_src_texture takes {:?}", start.elapsed());
                        }
                    }
                }
            }
        } else if matches!(
            src_fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) && !src.is_multiplane()
        {
            // Non-DMA single-plane NV*: the ShaderR8 path with the combined buffer
            // uploaded as R8.
            tracing::trace!(path = "ShaderR8-upload", src_fmt = ?src_fmt, "image.convert.gl.nv_path");
            // Records the path from the draw's result.
            match self.draw_nv_texture_2d(
                src,
                src_fmt,
                None,
                nv_rect,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
                luma_only,
                convert_span,
            ) {
                Ok(()) => self.last_nv_convert_path = NvConvertPath::ShaderR8,
                Err(e) => {
                    self.last_nv_convert_path = NvConvertPath::Cpu;
                    return Err(e);
                }
            }
        } else {
            // Non-DMA source, non-NV (or multiplane NV): texture-upload path.
            if matches!(
                src_fmt,
                PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
            ) {
                self.last_nv_convert_path = NvConvertPath::Cpu;
            }
            let start = Instant::now();
            self.draw_src_texture(
                src,
                src_fmt,
                src_roi,
                dst_roi,
                rotation_offset,
                flip,
                is_int8,
                luma_only,
                convert_span,
            )?;
            log::debug!("draw_src_texture takes {:?}", start.elapsed());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_to_planar(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> Result<(), crate::Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let alpha = match dst_fmt {
            PixelFormat::PlanarRgb => false,
            PixelFormat::PlanarRgba => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PlanarRgb or PlanarRgba".to_string(),
                ));
            }
        };
        log::trace!(
            "convert_to_planar: int8={is_int8}, interpolation={:?}",
            self.int8_interpolation_mode
        );

        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest::from_crop(&crop, src_w, src_h)
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst_w as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst_h as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst_w as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst_h as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };

        // Imports the source before the letterbox clear.
        let src_key = BufferImportKey::from_tensor(src, src_fmt, false);
        let src_egl = self.get_or_create_egl_image(CacheKind::Src, src, src_fmt)?;

        let has_crop = crop
            .dst_rect
            .is_some_and(|x| x.left != 0 || x.top != 0 || x.width != dst_w || x.height != dst_h);
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                self.clear_rect_planar(
                    dst_w,
                    dst_h,
                    [
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    ],
                    alpha,
                )?;
            }
        }

        self.convert_stats.src_imports += 1;
        convert_span.record("src_feed", "import");
        // The import map feeds both the source rectangle and the sampling clamp.
        let src_map = self.cached_src_import_map(src, src_fmt);
        let src_extent =
            super::render::sample_clamp_rect_within((src_w, src_h), src_map, src_roi.uv_bounds());
        let src_roi = super::render::scale_roi_to_import(src_roi, (src_w, src_h), src_map);

        self.draw_camera_texture_to_rgb_planar(
            src_key,
            src_egl,
            src_roi,
            src_extent,
            dst_roi,
            rotation_offset,
            flip,
            alpha,
            is_int8,
            src_fmt == PixelFormat::Grey,
        )?;
        unsafe { edgefirst_gl::gl::Finish() };
        check_gl_error(function!(), line!())?;

        Ok(())
    }

    /// Render packed RGB (or int8 RGB) to a DMA destination buffer in two
    /// passes:
    ///
    /// **Pass 1:** Render source → intermediate RGBA texture via `convert_to()`
    /// (crop, letterbox, rotation, flip).
    ///
    /// **Pass 2:** Pack intermediate RGBA → RGB DMA destination using a
    /// packing shader with a 2D sampler. The destination DMA buffer is
    /// reinterpreted as RGBA8 at (W*3/4) x H dimensions.
    #[allow(clippy::too_many_arguments)]
    fn convert_to_packed_rgb(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if self.int_pad_output {
            return Err(Error::NotSupported(
                "GL two-pass integer convert cannot write an output-domain pad; CPU fallback"
                    .into(),
            ));
        }
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        // RGBA8888-packed geometry of the destination.
        let layout = edgefirst_tensor::packed_rgb888_layout(dst_w, dst_h).ok_or_else(|| {
            crate::Error::NotSupported(format!(
                "Packed RGB requires width%4==0 (W*3 bytes must divide into \
                 whole RGBA8888 texels), got width={dst_w}"
            ))
        })?;
        let render_w = layout.surface_w;
        let render_h = layout.surface_h;

        log::debug!(
            "convert_to_packed_rgb: {dst_w}x{dst_h} -> {render_w}x{render_h} two-pass int8={is_int8}",
        );

        // --- Pass 1: Render source → intermediate RGBA texture ---
        let _pass1 =
            tracing::trace_span!("image.convert.gl.pack_rgb.pass1_rgba", dst_w, dst_h).entered();
        self.render_to_intermediate(
            src,
            src_fmt,
            (dst_w, dst_h),
            dst_fmt,
            rotation,
            flip,
            crop,
            convert_span,
        )?;
        drop(_pass1);

        // --- Pass 2: Pack intermediate RGBA → RGB DMA destination ---
        let _pass2 =
            tracing::trace_span!("image.convert.gl.pack_rgb.pass2_pack", render_w, render_h)
                .entered();
        self.convert_fbo.bind();
        let dest_egl = self.get_or_create_egl_image_rgb(
            dst,
            dst_fmt,
            render_w,
            render_h,
            super::platform::PackedImportFormat::Rgba8888,
        )?;
        unsafe {
            match self.cached_dst_renderbuffer(dst, dst_fmt) {
                Some(rbo) => {
                    edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                    edgefirst_gl::gl::FramebufferRenderbuffer(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::RENDERBUFFER,
                        rbo,
                    );
                }
                None => {
                    Self::attach_import_target(
                        &self.gl_context,
                        &mut self.render_texture,
                        dest_egl,
                    )?;
                }
            }
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::Viewport(0, 0, render_w as i32, render_h as i32);
        }

        // Bind intermediate RGBA texture as source for the packing shader
        let program = if is_int8 {
            &self.packed_rgba8_int8_program_2d
        } else {
            &self.packed_rgba8_program_2d
        };
        unsafe {
            edgefirst_gl::gl::UseProgram(program.id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE1);
            edgefirst_gl::gl::BindTexture(
                edgefirst_gl::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
            );
            super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::NEAREST);
        }

        // Draw full-viewport quad to pack RGBA→RGB
        self.draw_fullscreen_quad()?;

        // Unbinds the intermediate from TEXTURE1 and restores unit 0 as the
        // active unit.
        unsafe {
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, 0);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
        };

        unsafe { edgefirst_gl::gl::Finish() };
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Two-pass `ConvertPlan::TwoPassNvPlanar` render: any source format
    /// into a planar destination, via a packed intermediate.
    ///
    /// **Pass 1:** src→RGBA into `packed_rgb_intermediate_tex` via
    /// `convert_to()` (resize/crop/rotation/flip, no int8 bias).
    ///
    /// **Pass 2:** RGBA→PlanarRgb from the intermediate into whatever
    /// `bind_dst` classified the destination as (zero-copy: written
    /// directly; texture: read back afterwards) via
    /// [`draw_intermediate_to_rgb_planar`] (channel deinterleave + optional
    /// int8 bias).
    #[allow(clippy::too_many_arguments)]
    fn convert_nv_to_planar_two_pass(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        dst_fmt: PixelFormat,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        lowering: super::render::DstLowering,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if self.int_pad_output {
            return Err(Error::NotSupported(
                "GL two-pass integer convert cannot write an output-domain pad; CPU fallback"
                    .into(),
            ));
        }
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let alpha = match dst_fmt {
            PixelFormat::PlanarRgb => false,
            PixelFormat::PlanarRgba => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PlanarRgb or PlanarRgba".to_string(),
                ));
            }
        };

        log::debug!(
            "convert_nv_to_planar_two_pass: {src_fmt}→{dst_fmt} {dst_w}x{dst_h} int8={is_int8}",
        );

        // --- Pass 1: src→RGBA into intermediate texture ---
        let _pass1 = tracing::trace_span!("image.convert.gl.nv_to_planar.pass1_rgba", dst_w, dst_h)
            .entered();
        self.render_to_intermediate(
            src,
            src_fmt,
            (dst_w, dst_h),
            dst_fmt,
            rotation,
            flip,
            crop,
            convert_span,
        )?;
        drop(_pass1);

        // --- Pass 2: RGBA→planar into the classified destination ---
        // bind_dst rebinds convert_fbo with the destination target and sets the
        // viewport to (dst_w, dst_h * planes).
        let _pass2 = tracing::trace_span!(
            "image.convert.gl.nv_to_planar.pass2_deinterleave",
            dst_w,
            dst_h
        )
        .entered();
        let (target, _bound) = self.bind_dst(dst, dst_fmt, crop, lowering)?;
        Self::record_dst_feed(convert_span, target);

        // Pass 2 is a fullscreen blit from the intermediate to the planar
        // destination.
        let dst_roi = RegionOfInterest {
            left: -1.,
            top: 1.,
            right: 1.,
            bottom: -1.,
        };

        self.draw_intermediate_to_rgb_planar(dst_roi, alpha, is_int8)?;

        unsafe { edgefirst_gl::gl::Finish() };
        check_gl_error(function!(), line!())?;

        // Reads back a texture-lowered destination (Mem/Pbo).
        if let DstTarget::Texture { readback } = target {
            self.readback_rendered(dst, dst_fmt, readback.pbo_id())?;
        }
        Ok(())
    }

    /// Two-pass `ConvertPlan::TwoPassGrey` render: an RGB-family source into a
    /// Grey destination.
    ///
    /// **Pass 1:** src→RGBA into `packed_rgb_intermediate_tex` via
    /// `convert_to()` (resize, crop, letterbox, rotation, flip; no int8 bias).
    ///
    /// **Pass 2:** the intermediate's BT.601 luma, 1:1, into whatever
    /// `bind_dst` classified the destination as, with the int8 mapping of
    /// channel 0.
    #[allow(clippy::too_many_arguments)]
    fn convert_to_grey_two_pass(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        dst: &mut Tensor<u8>,
        is_int8: bool,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        lowering: super::render::DstLowering,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if self.int_pad_output {
            return Err(Error::NotSupported(
                "GL two-pass integer convert cannot write an output-domain pad; CPU fallback"
                    .into(),
            ));
        }
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;

        let _pass1 =
            tracing::trace_span!("image.convert.gl.grey.pass1_rgba", dst_w, dst_h).entered();
        self.render_to_intermediate(
            src,
            src_fmt,
            (dst_w, dst_h),
            PixelFormat::Rgba,
            rotation,
            flip,
            crop,
            convert_span,
        )?;
        drop(_pass1);

        let _pass2 =
            tracing::trace_span!("image.convert.gl.grey.pass2_luma", dst_w, dst_h).entered();
        let (target, _bound) = self.bind_dst(dst, PixelFormat::Grey, crop, lowering)?;
        Self::record_dst_feed(convert_span, target);
        let program = if is_int8 {
            self.luma_2d_int8_program.id
        } else {
            self.luma_2d_program.id
        };
        unsafe {
            edgefirst_gl::gl::UseProgram(program);
        }
        self.bind_intermediate();
        unsafe {
            if is_int8 {
                self.set_plane_quant(program, Some(0));
            }
        }
        let drawn = self.draw_fullscreen_quad();
        if is_int8 {
            // SAFETY: on the GL thread with `program` bound.
            unsafe { self.set_plane_quant(program, None) };
        }
        drawn?;
        unsafe { edgefirst_gl::gl::Finish() };
        check_gl_error(function!(), line!())?;

        if let DstTarget::Texture { readback } = target {
            self.readback_rendered(dst, PixelFormat::Grey, readback.pbo_id())?;
        }
        Ok(())
    }

    /// `ConvertPlan::ThreePassNv12` into a zero-copy NV12 destination;
    /// `NotSupported` when GL leaves it to the CPU.
    #[allow(clippy::too_many_arguments)]
    fn convert_to_nv12(
        &mut self,
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        src_fmt: PixelFormat,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        let band = self.nv12_admit(src, dst, src_fmt, rotation, flip, crop)?;
        let (coeffs, fill) = Self::nv12_encoding(dst, band, crop.dst_color);
        let planes = self.import_nv12_planes(dst)?;
        Self::record_dst_feed(convert_span, DstTarget::ZeroCopyImage);
        let (dst_w, dst_h) = (dst.width().unwrap_or(0), dst.height().unwrap_or(0));

        let _pass1 =
            tracing::trace_span!("image.convert.gl.nv12.pass1_rgba", dst_w, dst_h).entered();
        self.render_to_intermediate(
            src,
            src_fmt,
            (dst_w, dst_h),
            PixelFormat::Rgba,
            rotation,
            flip,
            crop,
            convert_span,
        )?;
        drop(_pass1);

        let drawn = self.draw_nv12_planes(planes, (dst_w, dst_h), band, fill, &coeffs);
        // SAFETY: on the GL thread.
        unsafe { edgefirst_gl::gl::Disable(edgefirst_gl::gl::SCISSOR_TEST) };
        drawn?;
        unsafe { edgefirst_gl::gl::Finish() };
        check_gl_error(function!(), line!())
    }

    /// Whether GL renders `src` into the NV12 `dst`: the placement band, a sub-rectangle of
    /// `dst`, or `None` for the whole destination.
    ///
    /// # Errors
    ///
    /// `NotSupported` for an odd destination size, a decline from [`nv12_dst_decline`], a
    /// placement without a fill colour whose edges are odd, a destination the GPU misplaces,
    /// and NV12 programs that failed to build.
    fn nv12_admit(
        &mut self,
        src: &Tensor<u8>,
        dst: &Tensor<u8>,
        src_fmt: PixelFormat,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> crate::Result<Option<crate::Rect>> {
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        if !dst_w.is_multiple_of(2) || !dst_h.is_multiple_of(2) {
            return Err(Error::NotSupported(format!(
                "NV12 destination must have even width and height, got {dst_w}x{dst_h}"
            )));
        }
        let whole = |r: Option<crate::Rect>, w: usize, h: usize| {
            r.is_none_or(|r| (r.left, r.top, r.width, r.height) == (0, 0, w, h))
        };
        let (src_w, src_h) = (
            src.width().ok_or(Error::NotAnImage)?,
            src.height().ok_or(Error::NotAnImage)?,
        );
        let same_size = (src_w, src_h) == (dst_w, dst_h)
            && rotation == crate::Rotation::None
            && flip == Flip::None
            && whole(crop.src_rect, src_w, src_h)
            && whole(crop.dst_rect, dst_w, dst_h);
        let exact_copy = same_size
            && matches!(
                src_fmt,
                PixelFormat::Nv12 | PixelFormat::Yuyv | PixelFormat::Vyuy
            )
            && crate::colorimetry::same_yuv_coding(
                &crate::colorimetry::resolve_colorimetry(src.colorimetry(), src.height()),
                &crate::colorimetry::resolve_colorimetry(dst.colorimetry(), dst.height()),
            );
        if let Some(why) = self.nv12_policy_decline(src_fmt, same_size, exact_copy, (dst_w, dst_h))
        {
            return Err(Error::NotSupported(why));
        }
        let band = crop
            .dst_rect
            .filter(|_| !whole(crop.dst_rect, dst_w, dst_h));
        if let Some(r) = band {
            if crop.dst_color.is_none()
                && [r.left, r.top, r.width, r.height]
                    .iter()
                    .any(|v| !v.is_multiple_of(2))
            {
                return Err(Error::NotSupported(format!(
                    "GL NV12 placement {r:?} without a fill colour needs even edges"
                )));
            }
        }
        if self.is_adreno
            && dst
                .chroma()
                .and_then(|c| c.plane_offset())
                .is_some_and(unaligned_import_offset)
        {
            return Err(Error::NotSupported(
                "this GPU misplaces a destination imported at an unaligned plane offset".into(),
            ));
        }
        self.ensure_nv12_programs()?;
        Ok(band)
    }

    /// The decline message from [`nv12_dst_decline`] under this GPU's policy, or `None` when
    /// GL renders the conversion. A soft decline applies only while `soft_declines` is set.
    pub(super) fn nv12_policy_decline(
        &self,
        src_fmt: PixelFormat,
        same_size: bool,
        exact_copy: bool,
        dst_size: (usize, usize),
    ) -> Option<String> {
        nv12_dst_decline(
            self.nv12_policy,
            &self.renderer_name,
            src_fmt,
            same_size,
            exact_copy,
            dst_size,
        )
        .filter(|(class, _)| *class == DeclineClass::Hard || self.soft_declines)
        .map(|(_, why)| why)
    }

    /// The RGB to YUV coefficients for `dst`'s colorimetry, and the encoded `[Y, U, V]` of
    /// `dst_color` when `band` is set.
    fn nv12_encoding(
        dst: &Tensor<u8>,
        band: Option<crate::Rect>,
        dst_color: Option<[u8; 4]>,
    ) -> (crate::colorimetry::RgbToYuvCoeffs, Option<[u8; 3]>) {
        let cm = crate::colorimetry::resolve_colorimetry(dst.colorimetry(), dst.height());
        let encoding = cm
            .encoding
            .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709);
        let range = cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited);
        let coeffs = crate::colorimetry::rgb_to_yuv_coeffs(encoding, range);
        let fill = band.and(dst_color).map(|rgba| {
            let cp = crate::cpu::ColorParams::uniform(encoding, range);
            let yuyv = crate::cpu::CPUProcessor::rgba_to_yuyv(rgba, cp);
            [yuyv[0], yuyv[1], yuyv[3]]
        });
        (coeffs, fill)
    }

    /// `dst`'s NV12 render targets.
    ///
    /// # Errors
    ///
    /// `NotSupported` when the import is refused, counted in
    /// `ConvertStats::dst_import_fallbacks`.
    fn import_nv12_planes(
        &mut self,
        dst: &Tensor<u8>,
    ) -> crate::Result<NvDstPlanes<PlatformHandle>> {
        self.get_or_create_nv12_dst(dst).map_err(|e| {
            self.note_dst_import_fallback(dst, PixelFormat::Nv12, &e, "converting on the CPU");
            Error::NotSupported(format!(
                "NV12 destination ({}x{}) import refused: {e}",
                dst.width().unwrap_or(0),
                dst.height().unwrap_or(0)
            ))
        })
    }

    /// Build the NV12 programs on the first NV12 destination. A failed build
    /// is logged once, and this and every later NV12 destination then
    /// returns `NotSupported`.
    fn ensure_nv12_programs(&mut self) -> crate::Result<()> {
        if matches!(self.nv12_programs, Nv12ProgramState::Unbuilt) {
            #[cfg(test)]
            let built = if self.fail_nv12_programs {
                Err(Error::OpenGl(
                    "NV12 program build failed by the test hook".into(),
                ))
            } else {
                Nv12Programs::new()
            };
            #[cfg(not(test))]
            let built = Nv12Programs::new();
            self.nv12_programs = match built {
                Ok(programs) => Nv12ProgramState::Built(programs),
                Err(e) => {
                    log::warn!(
                        "the NV12 destination programs failed to build; NV12 destinations \
                         convert on the CPU: {e}"
                    );
                    Nv12ProgramState::Failed(e.to_string())
                }
            };
        }
        match &self.nv12_programs {
            Nv12ProgramState::Failed(e) => Err(Error::NotSupported(format!(
                "the NV12 destination programs failed to build: {e}"
            ))),
            _ => Ok(()),
        }
    }

    /// The NV12 programs [`Self::ensure_nv12_programs`] built.
    fn nv12_programs(&self) -> crate::Result<&Nv12Programs> {
        match &self.nv12_programs {
            Nv12ProgramState::Built(programs) => Ok(programs),
            _ => Err(Error::Internal("the NV12 programs are not built".into())),
        }
    }

    /// Passes 2 and 3 of [`Self::convert_to_nv12`]: fill, then luma and
    /// chroma from `packed_rgb_intermediate_tex` into `planes`. `band` is the
    /// placement and `fill` the encoded `[Y, U, V]` of its fill colour. Leaves
    /// `SCISSOR_TEST` enabled when `band` is set.
    fn draw_nv12_planes(
        &mut self,
        planes: NvDstPlanes<PlatformHandle>,
        (w, h): (usize, usize),
        band: Option<crate::Rect>,
        fill: Option<[u8; 3]>,
        coeffs: &crate::colorimetry::RgbToYuvCoeffs,
    ) -> crate::Result<()> {
        let (w, h) = (w as i32, h as i32);
        let (luma, chroma, chroma_row, chroma_h) = match planes {
            NvDstPlanes::Combined { plane, .. } => (plane, None, h, h / 2),
            NvDstPlanes::Split {
                luma,
                chroma,
                chroma_rows,
            } => (luma, Some(chroma), 0, chroma_rows as i32),
        };
        let scissor = |x: i32, y: i32, sw: i32, sh: i32| unsafe {
            edgefirst_gl::gl::Scissor(x, y, sw, sh);
            edgefirst_gl::gl::Enable(edgefirst_gl::gl::SCISSOR_TEST);
        };

        let _pass2 = tracing::trace_span!("image.convert.gl.nv12.pass2_luma", w, h).entered();
        self.convert_fbo.bind();
        // SAFETY: `luma` is owned by `dst_egl_cache` for this convert.
        unsafe { Self::attach_import_target(&self.gl_context, &mut self.render_texture, luma)? };
        if let Some([y, _, _]) = fill {
            scissor(0, 0, w, h);
            unsafe {
                edgefirst_gl::gl::ClearColor(y as f32 / 255.0, 0.0, 0.0, 1.0);
                edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            }
        }
        match band {
            Some(r) => scissor(r.left as i32, r.top as i32, r.width as i32, r.height as i32),
            None => unsafe { edgefirst_gl::gl::Disable(edgefirst_gl::gl::SCISSOR_TEST) },
        }
        let p = self.nv12_programs()?;
        unsafe {
            edgefirst_gl::gl::Viewport(0, 0, w, h);
            edgefirst_gl::gl::UseProgram(p.luma.id);
            let [y_coeffs, y_offset] = p.luma_locs;
            edgefirst_gl::gl::Uniform3f(
                y_coeffs,
                coeffs.y[0] as f32,
                coeffs.y[1] as f32,
                coeffs.y[2] as f32,
            );
            edgefirst_gl::gl::Uniform1f(y_offset, coeffs.y_offset as f32);
        }
        self.bind_intermediate();
        self.draw_fullscreen_quad()?;
        drop(_pass2);

        let _pass3 = tracing::trace_span!("image.convert.gl.nv12.pass3_chroma", w, h).entered();
        if let Some(chroma) = chroma {
            // SAFETY: `chroma` is owned by `dst_egl_cache` for this convert.
            unsafe {
                Self::attach_import_target(&self.gl_context, &mut self.nv12_chroma_texture, chroma)?
            };
        }
        unsafe { edgefirst_gl::gl::Viewport(0, chroma_row, w, chroma_h) };
        let p = self.nv12_programs()?;
        if let Some([_, u, v]) = fill {
            unsafe {
                edgefirst_gl::gl::Disable(edgefirst_gl::gl::SCISSOR_TEST);
                edgefirst_gl::gl::UseProgram(p.fill.id);
                edgefirst_gl::gl::Uniform2f(p.fill_uv, u as f32 / 255.0, v as f32 / 255.0);
            }
            self.draw_fullscreen_quad()?;
        }
        if let Some(r) = band {
            let (x0, x1) = (r.left / 2 * 2, (r.left + r.width).div_ceil(2) * 2);
            let (y0, y1) = (r.top / 2, (r.top + r.height).div_ceil(2));
            scissor(
                x0 as i32,
                chroma_row + y0 as i32,
                (x1 - x0) as i32,
                (y1 - y0) as i32,
            );
        }
        let p = self.nv12_programs()?;
        unsafe {
            edgefirst_gl::gl::UseProgram(p.chroma.id);
            let [u_coeffs, v_coeffs, c_offset, row] = p.chroma_locs;
            edgefirst_gl::gl::Uniform3f(
                u_coeffs,
                coeffs.u[0] as f32,
                coeffs.u[1] as f32,
                coeffs.u[2] as f32,
            );
            edgefirst_gl::gl::Uniform3f(
                v_coeffs,
                coeffs.v[0] as f32,
                coeffs.v[1] as f32,
                coeffs.v[2] as f32,
            );
            edgefirst_gl::gl::Uniform1f(c_offset, coeffs.c_offset as f32);
            edgefirst_gl::gl::Uniform1i(row, chroma_row);
        }
        self.bind_intermediate();
        self.draw_fullscreen_quad()
    }

    /// Attach `handle` to `texture` and `texture` to `convert_fbo`'s colour
    /// attachment, dropping the texture's cached EGLImage binding.
    ///
    /// # Errors
    ///
    /// `NotSupported` when the framebuffer is incomplete with the import
    /// attached.
    ///
    /// # Safety
    /// On the GL thread with `convert_fbo` bound and `handle`'s import alive.
    unsafe fn attach_import_target(
        gl_context: &PlatformDisplay,
        texture: &mut Texture,
        handle: PlatformHandle,
    ) -> crate::Result<()> {
        unsafe {
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, texture.id);
            super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::NEAREST);
            let attached = Platform::attach_tex_image_2d(gl_context, handle);
            texture.invalidate_egl_binding();
            attached?;
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                texture.id,
                0,
            );
            if let Err(status) = super::core::check_framebuffer_complete() {
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                return Err(Error::NotSupported(format!(
                    "GL render target import leaves the framebuffer incomplete (0x{status:x})"
                )));
            }
        }
        check_gl_error(function!(), line!())
    }

    /// Bind `packed_rgb_intermediate_tex` to texture unit 0 with `NEAREST`
    /// filtering and edge clamping.
    fn bind_intermediate(&self) {
        // SAFETY: on the GL thread.
        unsafe {
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(
                edgefirst_gl::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
            );
            super::core::set_tex_filter_clamp(
                edgefirst_gl::gl::TEXTURE_2D,
                edgefirst_gl::gl::NEAREST,
            );
        }
    }

    /// Render `src` into `packed_rgb_intermediate_tex` at `dst_w`×`dst_h`
    /// through [`Self::convert_to_dims`] with `dst_fmt` as the logical
    /// destination format, un-biased and with the finish deferred. The
    /// `defer_finish` flag is restored before an error propagates.
    #[allow(clippy::too_many_arguments)]
    fn render_to_intermediate(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        (dst_w, dst_h): (usize, usize),
        dst_fmt: PixelFormat,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        self.ensure_packed_rgb_intermediate(dst_w, dst_h)?;
        self.packed_rgb_fbo.bind();
        unsafe {
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            edgefirst_gl::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        }
        let saved_defer = self.defer_finish;
        self.defer_finish = true;
        let rendered = self.convert_to_dims(
            src,
            src_fmt,
            dst_w,
            dst_h,
            None,
            dst_fmt,
            false,
            rotation,
            flip,
            crop,
            convert_span,
        );
        self.defer_finish = saved_defer;
        rendered
    }

    /// Allocates or resizes the intermediate RGBA texture for two-pass packed RGB.
    fn ensure_packed_rgb_intermediate(&mut self, width: usize, height: usize) -> crate::Result<()> {
        if self.packed_rgb_intermediate_size == (width, height) {
            return Ok(());
        }
        unsafe {
            edgefirst_gl::gl::BindTexture(
                edgefirst_gl::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
            );
            super::core::set_tex_filter(edgefirst_gl::gl::TEXTURE_2D, edgefirst_gl::gl::NEAREST);
            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                edgefirst_gl::gl::RGBA as i32,
                width as i32,
                height as i32,
                0,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            check_gl_error(function!(), line!())?;
        }
        self.packed_rgb_intermediate_size = (width, height);
        Ok(())
    }

    /// Draw a fullscreen quad for the currently-bound shader program.
    fn draw_fullscreen_quad(&self) -> Result<(), crate::Error> {
        unsafe {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let vertices: [f32; 12] = [
                -1.0, 1.0, 0.0, // top-left
                1.0, 1.0, 0.0, // top-right
                1.0, -1.0, 0.0, // bottom-right
                -1.0, -1.0, 0.0, // bottom-left
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * vertices.len()) as isize,
                vertices.as_ptr() as *const c_void,
            );

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            // Texture coordinates for the vertex attribute layout.
            let tex_coords: [f32; 8] = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * tex_coords.len()) as isize,
                tex_coords.as_ptr() as *const c_void,
            );

            let indices: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                indices.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                indices.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Fill each plane of the `(width, split * height)` planar target with
    /// its pad value. Plane `i` is the framebuffer band of rows
    /// `[i * height, (i + 1) * height)` (the draws start from the bottom).
    fn clear_rect_planar(
        &self,
        width: usize,
        height: usize,
        color: [f32; 4],
        alpha: bool,
    ) -> Result<(), Error> {
        let split = if alpha { 4 } else { 3 };
        // SAFETY: on the GL thread with the planar target bound.
        unsafe {
            if color[..split].iter().all(|&c| c == color[0]) {
                edgefirst_gl::gl::ClearColor(color[0], color[0], color[0], 1.0);
                edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
                return Ok(());
            }
            edgefirst_gl::gl::Enable(edgefirst_gl::gl::SCISSOR_TEST);
            for (i, c) in color.iter().enumerate().take(split) {
                edgefirst_gl::gl::Scissor(0, (i * height) as i32, width as i32, height as i32);
                edgefirst_gl::gl::ClearColor(*c, *c, *c, 1.0);
                edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            }
            edgefirst_gl::gl::Disable(edgefirst_gl::gl::SCISSOR_TEST);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_camera_texture_to_rgb_planar(
        &mut self,
        src_key: BufferImportKey,
        egl_img: PlatformHandle,
        src_roi: RegionOfInterest,
        // The rectangle a sample may reach on the imported texture
        // (`render::sample_clamp_rect`).
        src_extent: [f32; 4],
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        alpha: bool,
        int8: bool,
        // An R8 (Grey) import: every colour plane draws its red channel.
        luma: bool,
    ) -> Result<(), Error> {
        let texture_target = edgefirst_gl::gl::TEXTURE_EXTERNAL_OES;
        match flip {
            Flip::None => {}
            Flip::Vertical => {
                std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
            }
            Flip::Horizontal => {
                std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
            }
        }
        let program_id = if int8 {
            &self.texture_program_planar_int8
        } else {
            &self.texture_program_planar
        }
        .as_ref()
        .ok_or_else(|| {
            Error::NotSupported("external-OES planar program unavailable on this platform".into())
        })?
        .id;
        let raw_alpha_id = self.texture_program_planar.as_ref().map_or(0, |p| p.id);
        unsafe {
            edgefirst_gl::gl::UseProgram(program_id);
            let [e0, e1, e2, e3] = src_extent;
            edgefirst_gl::gl::Uniform4f(self.external_src_extent_loc(int8, true), e0, e1, e2, e3);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            super::core::set_tex_filter(texture_target, self.source_filter());
            edgefirst_gl::gl::TexParameteri(
                texture_target,
                edgefirst_gl::gl::TEXTURE_WRAP_S,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );

            edgefirst_gl::gl::TexParameteri(
                texture_target,
                edgefirst_gl::gl::TEXTURE_WRAP_T,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );

            if self.camera_eglimage_texture.bind_egl_image_external(
                &self.gl_context,
                src_key,
                egl_img,
            )? {
                check_gl_error(function!(), line!())?;
                log::trace!(
                    "draw_camera_planar: bound new src EGLImage id={:#x}",
                    src_key.luma_id
                );
            } else {
                log::trace!(
                    "draw_camera_planar: reusing bound src EGLImage id={:#x}",
                    src_key.luma_id
                );
            }
            let y_centers = if alpha {
                vec![-3.0 / 4.0, -1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0]
            } else {
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            };
            let planes = y_centers.len() as f32;
            let swizzles = if luma {
                [
                    edgefirst_gl::gl::RED,
                    edgefirst_gl::gl::RED,
                    edgefirst_gl::gl::RED,
                    edgefirst_gl::gl::ALPHA,
                ]
            } else {
                [
                    edgefirst_gl::gl::RED,
                    edgefirst_gl::gl::GREEN,
                    edgefirst_gl::gl::BLUE,
                    edgefirst_gl::gl::ALPHA,
                ]
            };
            // starts from bottom
            for (i, y_center) in y_centers.iter().enumerate() {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top / planes + y_center,
                    0., // left top
                    dst_roi.right,
                    dst_roi.top / planes + y_center,
                    0., // right top
                    dst_roi.right,
                    dst_roi.bottom / planes + y_center,
                    0., // right bottom
                    dst_roi.left,
                    dst_roi.bottom / planes + y_center,
                    0., // left bottom
                ];
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    edgefirst_gl::gl::DYNAMIC_DRAW,
                );

                edgefirst_gl::gl::BindBuffer(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    self.texture_buffer.id,
                );
                edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let texture_vertices: [f32; 16] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];

                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * 8) as isize,
                    (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                    edgefirst_gl::gl::DYNAMIC_DRAW,
                );
                let vertices_index: [u32; 4] = [0, 1, 2, 3];

                edgefirst_gl::gl::TexParameteri(
                    texture_target,
                    edgefirst_gl::gl::TEXTURE_SWIZZLE_R,
                    swizzles[i] as i32,
                );
                if int8 && i == 3 {
                    // Integer alpha is drawn by the u8 program.
                    self.set_plane_quant(program_id, None);
                    edgefirst_gl::gl::UseProgram(raw_alpha_id);
                    let [e0, e1, e2, e3] = src_extent;
                    edgefirst_gl::gl::Uniform4f(
                        self.external_src_extent_loc(false, true),
                        e0,
                        e1,
                        e2,
                        e3,
                    );
                } else {
                    self.set_plane_quant(program_id, Some(i));
                }

                edgefirst_gl::gl::DrawElements(
                    edgefirst_gl::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    edgefirst_gl::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
            if !(int8 && alpha) {
                self.set_plane_quant(program_id, None);
            }
            // Reset the texture swizzle to identity.
            edgefirst_gl::gl::TexParameteri(
                texture_target,
                edgefirst_gl::gl::TEXTURE_SWIZZLE_R,
                edgefirst_gl::gl::RED as i32,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    /// Draw the intermediate RGBA texture to planar RGB output.
    ///
    /// Sources from `packed_rgb_intermediate_tex` (a `TEXTURE_2D`), without
    /// rotation or flip.
    fn draw_intermediate_to_rgb_planar(
        &self,
        dst_roi: RegionOfInterest,
        alpha: bool,
        int8: bool,
    ) -> Result<(), Error> {
        let texture_target = edgefirst_gl::gl::TEXTURE_2D;
        unsafe {
            let program = if int8 {
                &self.texture_program_planar_int8_2d
            } else {
                &self.texture_program_planar_2d
            };
            edgefirst_gl::gl::UseProgram(program.id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(texture_target, self.packed_rgb_intermediate_tex.id);
            // NEAREST: a 1:1 copy of pass 1's output.
            super::core::set_tex_filter_clamp(texture_target, edgefirst_gl::gl::NEAREST);

            check_gl_error(function!(), line!())?;

            let y_centers = if alpha {
                vec![-3.0 / 4.0, -1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0]
            } else {
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            };
            let planes = y_centers.len() as f32;
            let swizzles = [
                edgefirst_gl::gl::RED,
                edgefirst_gl::gl::GREEN,
                edgefirst_gl::gl::BLUE,
                edgefirst_gl::gl::ALPHA,
            ];

            // The source ROI is the whole intermediate.
            let src_roi = RegionOfInterest {
                left: 0.0,
                top: 1.0,
                right: 1.0,
                bottom: 0.0,
            };

            for (i, y_center) in y_centers.iter().enumerate() {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top / planes + y_center,
                    0., // left top
                    dst_roi.right,
                    dst_roi.top / planes + y_center,
                    0., // right top
                    dst_roi.right,
                    dst_roi.bottom / planes + y_center,
                    0., // right bottom
                    dst_roi.left,
                    dst_roi.bottom / planes + y_center,
                    0., // left bottom
                ];
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    edgefirst_gl::gl::DYNAMIC_DRAW,
                );

                edgefirst_gl::gl::BindBuffer(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    self.texture_buffer.id,
                );
                edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                // No rotation: the base texture coords are used directly.
                let texture_vertices: [f32; 8] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * texture_vertices.len()) as isize,
                    texture_vertices.as_ptr() as *const c_void,
                    edgefirst_gl::gl::DYNAMIC_DRAW,
                );

                let vertices_index: [u32; 4] = [0, 1, 2, 3];

                edgefirst_gl::gl::TexParameteri(
                    texture_target,
                    edgefirst_gl::gl::TEXTURE_SWIZZLE_R,
                    swizzles[i] as i32,
                );
                if int8 && i == 3 {
                    // Integer alpha is drawn by the u8 program.
                    self.set_plane_quant(program.id, None);
                    edgefirst_gl::gl::UseProgram(self.texture_program_planar_2d.id);
                } else {
                    self.set_plane_quant(program.id, Some(i));
                }

                edgefirst_gl::gl::DrawElements(
                    edgefirst_gl::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    edgefirst_gl::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
            if !(int8 && alpha) {
                self.set_plane_quant(program.id, None);
            }
            // Reset the texture swizzle to identity.
            edgefirst_gl::gl::TexParameteri(
                texture_target,
                edgefirst_gl::gl::TEXTURE_SWIZZLE_R,
                edgefirst_gl::gl::RED as i32,
            );
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_src_texture(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        mut src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        luma_only: bool,
        convert_span: &tracing::Span,
    ) -> Result<(), Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let texture_target = edgefirst_gl::gl::TEXTURE_2D;
        // Zero-copy source attach: on platforms whose imports bind as
        // TEXTURE_2D, a Dma-memory source is imported and attached to the
        // source texture. An import error leaves the source to the upload
        // below; YUYV is accepted only when its RG attach succeeds.
        let zero_copy_attach = if !self.gl_context.transfer_backend.is_dma()
            && self.gl_context.transfer_backend.is_zero_copy()
            && src.memory() == TensorMemory::DmaBuf
            && matches!(
                src_fmt,
                PixelFormat::Rgba | PixelFormat::Grey | PixelFormat::Yuyv
            ) {
            let key = BufferImportKey::from_tensor(src, src_fmt, false);
            match self.get_or_create_egl_image(CacheKind::Src, src, src_fmt) {
                Ok(handle) => Some((key, handle)),
                Err(e) => {
                    self.convert_stats.zero_copy_declines += 1;
                    log::debug!("zero-copy source import failed ({e:?}); uploading instead");
                    None
                }
            }
        } else {
            None
        };
        // The part of the texture the logical image occupies: `WHOLE` when the
        // source is uploaded.
        let import_map = if zero_copy_attach.is_some() {
            self.cached_src_import_map(src, src_fmt)
        } else {
            super::render::ImportMap::WHOLE
        };
        // The clamp rectangle: the crop's (or else the logical image's)
        // half-texel-inset rectangle on the texture.
        let [u0, v0, u1, v1] = super::render::sample_clamp_rect_within(
            (src_w, src_h),
            import_map,
            src_roi.uv_bounds(),
        );
        src_roi = super::render::scale_roi_to_import(src_roi, (src_w, src_h), import_map);
        let texture_format = match src_fmt {
            PixelFormat::Rgb => edgefirst_gl::gl::RGB,
            PixelFormat::Rgba => edgefirst_gl::gl::RGBA,
            PixelFormat::Grey => edgefirst_gl::gl::RED,
            // YUYV samples as RG, one texel per pixel: R is the pixel's Y, G
            // alternates U (even column) and V (odd column).
            PixelFormat::Yuyv => edgefirst_gl::gl::RG,
            _ => {
                return Err(Error::NotSupported(format!(
                    "draw_src_texture does not support {src_fmt:?} (use DMA-BUF path for YUV)",
                )));
            }
        };
        // Selects the program; the int8 program adds the XOR-0x80 bias.
        let program_id = if src_fmt == PixelFormat::Yuyv {
            if is_int8 {
                self.yuyv_int8_program_2d.id
            } else {
                self.yuyv_program_2d.id
            }
        } else if is_int8 {
            self.texture_int8_program.id
        } else {
            self.texture_program.id
        };
        unsafe {
            edgefirst_gl::gl::UseProgram(program_id);
            if src_fmt == PixelFormat::Yuyv {
                // YUYV program inputs: the source texel grid and the YUV→RGB
                // matrix/range resolved from the tensor colorimetry.
                let cm = crate::colorimetry::resolve_colorimetry(src.colorimetry(), src.height());
                let coeffs = crate::colorimetry::yuv_to_rgb_coeffs(
                    cm.encoding
                        .unwrap_or(edgefirst_tensor::ColorEncoding::Bt709),
                    cm.range.unwrap_or(edgefirst_tensor::ColorRange::Limited),
                );
                // A Grey destination uses the luma terms only.
                let coeffs = if luma_only {
                    coeffs.luma_only()
                } else {
                    coeffs
                };
                let [src_size, y_offset, y_scale, c_vr, c_ug, c_vg, c_ub, src_extent] =
                    self.yuyv_2d_locs[usize::from(is_int8)];
                // The shader rebuilds texel coordinates as `floor(tc * src_size)` on
                // the texture's own texel grid.
                debug_assert_eq!(
                    import_map.origin,
                    (0, 0),
                    "YUYV must not be sampled through a shifted import"
                );
                let (grid_w, grid_h) = import_map
                    .extent
                    .map_or((src_w as f32, src_h as f32), |(pw, ph)| {
                        (pw as f32, ph as f32)
                    });
                edgefirst_gl::gl::Uniform2f(src_size, grid_w, grid_h);
                edgefirst_gl::gl::Uniform4f(src_extent, u0, v0, u1, v1);
                edgefirst_gl::gl::Uniform1f(y_offset, coeffs.y_offset);
                edgefirst_gl::gl::Uniform1f(y_scale, coeffs.y_scale);
                edgefirst_gl::gl::Uniform1f(c_vr, coeffs.c_vr);
                edgefirst_gl::gl::Uniform1f(c_ug, coeffs.c_ug);
                edgefirst_gl::gl::Uniform1f(c_vg, coeffs.c_vg);
                edgefirst_gl::gl::Uniform1f(c_ub, coeffs.c_ub);
            } else {
                edgefirst_gl::gl::Uniform4f(self.texture_src_extent_loc(is_int8), u0, v0, u1, v1);
            }
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            super::core::set_tex_filter_clamp(texture_target, self.source_filter());
            if src_fmt == PixelFormat::Grey {
                for swizzle in [
                    edgefirst_gl::gl::TEXTURE_SWIZZLE_R,
                    edgefirst_gl::gl::TEXTURE_SWIZZLE_G,
                    edgefirst_gl::gl::TEXTURE_SWIZZLE_B,
                ] {
                    edgefirst_gl::gl::TexParameteri(
                        edgefirst_gl::gl::TEXTURE_2D,
                        swizzle,
                        edgefirst_gl::gl::RED as i32,
                    );
                }
            } else {
                for (swizzle, src_comp) in [
                    (edgefirst_gl::gl::TEXTURE_SWIZZLE_R, edgefirst_gl::gl::RED),
                    (edgefirst_gl::gl::TEXTURE_SWIZZLE_G, edgefirst_gl::gl::GREEN),
                    (edgefirst_gl::gl::TEXTURE_SWIZZLE_B, edgefirst_gl::gl::BLUE),
                ] {
                    edgefirst_gl::gl::TexParameteri(
                        edgefirst_gl::gl::TEXTURE_2D,
                        swizzle,
                        src_comp as i32,
                    );
                }
            }
            // Attaches the import, or uploads the source with `UNPACK_ROW_LENGTH`
            // set to the row length in pixels.
            if let Some((key, handle)) = zero_copy_attach {
                self.camera_normal_texture
                    .bind_egl_image(&self.gl_context, key, handle)?;
                // Forgets the upload-tracking dims so a later upload allocates fresh
                // storage.
                self.camera_normal_texture.forget_storage();
                self.convert_stats.src_imports += 1;
                convert_span.record("src_feed", "import");
            } else {
                let src_bpp = src_fmt.channels().max(1);
                // Declines a pitch that is not a whole number of pixels.
                let row_len_px = unpack_row_length(
                    src.effective_row_stride(),
                    src_w,
                    src_bpp,
                    &format!("{src_fmt:?}"),
                )?;
                // Opens the pixels before setting pixel-store state.
                let pixels = SrcPixels::open(src, &self.owned_pbos)?;
                pixels.record(&mut self.convert_stats, convert_span);
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, row_len_px);
                // The bytes GL reads: every row but the last at the row stride, plus
                // the last row's own pixels.
                let row_stride_b = if row_len_px == 0 {
                    src_w * src_bpp
                } else {
                    row_len_px as usize * src_bpp
                };
                let required = if src_h == 0 {
                    0
                } else {
                    (src_h - 1) * row_stride_b + src_w * src_bpp
                };
                let uploaded = self.camera_normal_texture.upload(
                    texture_target,
                    src_w,
                    src_h,
                    texture_format,
                    required,
                    pixels.source(),
                );
                // Resets the row length on both outcomes.
                edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, 0);
                uploaded?;
            }

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            edgefirst_gl::gl::BufferData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
                edgefirst_gl::gl::DYNAMIC_DRAW,
            );
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];

            edgefirst_gl::gl::BufferData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                edgefirst_gl::gl::DYNAMIC_DRAW,
            );
            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            check_gl_error(function!(), line!())?;

            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Draw a source bound as an external-OES texture from its EGLImage into
    /// the bound FBO. `y2y` selects the Y2Y NV programs with the source's
    /// resolved (encoding, range) for their in-shader matrix
    /// ([`NvSampling::HardwareYuv`]).
    #[allow(clippy::too_many_arguments)]
    fn draw_camera_texture_eglimage(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        egl_img: PlatformHandle,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        y2y: Option<(
            edgefirst_tensor::ColorEncoding,
            edgefirst_tensor::ColorRange,
        )>,
    ) -> Result<(), Error> {
        let src_key = BufferImportKey::from_tensor(src, src_fmt, false);
        let luma_id = src_key.luma_id;
        let crop_uv = src_roi.uv_bounds();
        // Maps the source ROI onto the texture the import covers.
        let src_roi = self.scale_src_roi(src_roi, src, src_fmt);
        // The clamp rectangle through the same mapping.
        let (src_w, src_h) = (
            src.width().ok_or(Error::NotAnImage)?,
            src.height().ok_or(Error::NotAnImage)?,
        );
        let [e0, e1, e2, e3] = super::render::sample_clamp_rect_within(
            (src_w, src_h),
            self.cached_src_import_map(src, src_fmt),
            crop_uv,
        );
        // Selects the program.
        let (program_id, extent_loc, y2y_upload) = match y2y {
            Some(colorimetry) => {
                let (programs, uniforms) = self.nv_y2y_programs.as_ref().ok_or_else(|| {
                    Error::NotSupported("Y2Y NV programs unavailable on this GPU".into())
                })?;
                let state = &uniforms[usize::from(is_int8)];
                let upload = (state.last_colorimetry != Some(colorimetry))
                    .then_some((state.locs, colorimetry));
                (
                    programs[usize::from(is_int8)].id,
                    state.locs.src_extent,
                    upload,
                )
            }
            None => {
                let program = if is_int8 {
                    &self.texture_int8_program_yuv
                } else {
                    &self.texture_program_yuv
                }
                .as_ref()
                .ok_or_else(|| {
                    Error::NotSupported(
                        "external-OES sampler program unavailable on this platform".into(),
                    )
                })?;
                (
                    program.id,
                    self.external_src_extent_loc(is_int8, false),
                    None,
                )
            }
        };
        let texture_target = edgefirst_gl::gl::TEXTURE_EXTERNAL_OES;
        unsafe {
            edgefirst_gl::gl::UseProgram(program_id);
            edgefirst_gl::gl::Uniform4f(extent_loc, e0, e1, e2, e3);
            if let Some((locs, (encoding, range))) = y2y_upload {
                let k = crate::colorimetry::yuv_to_rgb_coeffs(encoding, range);
                edgefirst_gl::gl::Uniform1f(locs.y_offset, k.y_offset);
                edgefirst_gl::gl::Uniform1f(locs.y_scale, k.y_scale);
                edgefirst_gl::gl::Uniform1f(locs.c_vr, k.c_vr);
                edgefirst_gl::gl::Uniform1f(locs.c_ug, k.c_ug);
                edgefirst_gl::gl::Uniform1f(locs.c_vg, k.c_vg);
                edgefirst_gl::gl::Uniform1f(locs.c_ub, k.c_ub);
            }
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            super::core::set_tex_filter_clamp(texture_target, self.source_filter());

            // Binds the source as an external texture. A Grey source is an R8
            // import, which samples as (l, 0, 0, 1): G and B are pointed at its
            // red channel.
            if self.camera_eglimage_texture.bind_egl_image_external(
                &self.gl_context,
                src_key,
                egl_img,
            )? {
                check_gl_error(function!(), line!())?;
                log::trace!("draw_camera: bound new src EGLImage id={luma_id:#x}");
            } else {
                log::trace!("draw_camera: reusing bound src EGLImage id={luma_id:#x}");
            }
            let luma = src_fmt == PixelFormat::Grey;
            if luma {
                set_external_gb_swizzle(edgefirst_gl::gl::RED, edgefirst_gl::gl::RED);
            }
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            // Restores the identity swizzle on the shared external texture.
            if luma {
                set_external_gb_swizzle(edgefirst_gl::gl::GREEN, edgefirst_gl::gl::BLUE);
            }
        }
        check_gl_error(function!(), line!())?;
        if let (Some(colorimetry), Some((_, uniforms))) = (y2y, self.nv_y2y_programs.as_mut()) {
            uniforms[usize::from(is_int8)].last_colorimetry = Some(colorimetry);
        }
        Ok(())
    }

    /// `ShaderR8`: render NV12/NV16/NV24 → RGBA8 via the R8 texelFetch shader.
    ///
    /// The combined semi-planar buffer is presented as a single-plane R8
    /// `TEXTURE_2D` and the shader addresses Y and UV bytes directly,
    /// parameterised by four uniforms derived from the source format.
    ///
    /// `r8_src` selects how the R8 texture is established:
    /// - `Some(egl_img)`: zero-copy EGLImage import of a DMA-BUF source.
    /// - `None`: upload of the combined buffer via `glTexImage2D`.
    ///
    /// Renders into the currently-bound FBO.
    #[allow(clippy::too_many_arguments)]
    fn draw_nv_texture_2d(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        r8_src: Option<PlatformHandle>,
        src_rect: NvSampleRect,
        dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        luma_only: bool,
        convert_span: &tracing::Span,
    ) -> Result<(), Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let tex_width = src.effective_row_stride().unwrap_or(src_w) as i32;

        // Format-specific shader uniforms from `chroma_layout`. `chroma_lines`
        // (== `uv_rows_per_luma`) is the number of R8 buffer rows each image
        // chroma row occupies (NV24's CbCr row is 2W bytes = 2 rows; NV12/NV16
        // are W bytes = 1 row).
        let layout = src_fmt.chroma_layout().ok_or_else(|| {
            Error::NotSupported(format!(
                "draw_nv_texture_2d: unsupported format {src_fmt:?}"
            ))
        })?;
        let chroma_shift_x = layout.shift_x as i32;
        let chroma_shift_y = layout.shift_y as i32;
        let chroma_lines = layout.uv_rows_per_luma as i32;

        let src_key = BufferImportKey::from_tensor(src, src_fmt, false);
        let luma_id = src_key.luma_id;

        // Selects the program: the int8 programs add the XOR-0x80 bias, and a
        // bilinear rescale filters through the texture units when the chroma
        // plane can be bound on its own.
        let bilinear =
            src_rect.resample && self.interpolation.supported() == crate::Interpolation::Bilinear;
        let colorimetry = nv_colorimetry(src.colorimetry(), src.height());
        // A limited-range luma draw filters in the shader.
        let hw_filter = self.nv_hw_filter
            && !(luma_only && colorimetry.1 == edgefirst_tensor::ColorRange::Limited);
        // A two-plane import draws with the hardware-filtered program; without
        // its chroma plane the source is uploaded as the combined plane.
        let mut r8_src = r8_src;
        let mut native_extent = r8_src.and_then(|_| Platform::nv_native_planes(src));
        // A bilinear draw that filters in the shader uploads the combined plane
        // from the CPU map when the source can be read.
        if native_extent.is_some() && bilinear && !hw_filter && src.cpu_access().reads() {
            native_extent = None;
            r8_src = None;
        }
        let mut native_planes = native_extent.is_some();
        let uv_import = if native_planes {
            match self.get_or_create_nv_uv_egl_image(src, src_fmt) {
                Ok(import) => Some(import),
                Err(e) => {
                    log::debug!("NV12 chroma plane import failed ({e}); uploading the planes");
                    native_planes = false;
                    r8_src = None;
                    None
                }
            }
        } else if bilinear && hw_filter && r8_src.is_some() {
            self.get_or_create_nv_uv_egl_image(src, src_fmt)
                .inspect_err(|e| log::debug!("NV chroma view unavailable ({e}); shader-filtered"))
                .ok()
        } else {
            None
        };
        // Y2Y is offered for NV12 colour draws only.
        let mut sampling = if native_planes {
            NvSampling::Hardware
        } else {
            NvSampling::choose(NvSamplingInputs {
                bilinear,
                hw_filter,
                imported: r8_src.is_some(),
                chroma_view: uv_import.is_some(),
                y2y: self.nv_y2y_programs.is_some()
                    && !luma_only
                    && src_fmt == PixelFormat::Nv12
                    && y2y_pays_off(src_w * src_h, src_rect.dst_pixels),
            })
        };
        if sampling == NvSampling::HardwareYuv {
            match self.get_or_create_egl_image(CacheKind::Src, src, src_fmt) {
                Ok(img) => {
                    self.draw_camera_texture_eglimage(
                        src,
                        src_fmt,
                        img,
                        src_rect.roi(src_w, src_h),
                        dst_roi,
                        rotation_offset,
                        flip,
                        is_int8,
                        Some(colorimetry),
                    )?;
                    self.last_nv_sampling = Some(sampling);
                    return Ok(());
                }
                Err(e) => {
                    log::debug!("NV12 import for Y2Y unavailable ({e}); shader-filtered");
                    sampling = NvSampling::Shader;
                }
            }
        }
        let program = sampling.program(is_int8);
        let r8_filter = if sampling == NvSampling::Hardware && (bilinear || !native_planes) {
            edgefirst_gl::gl::LINEAR
        } else {
            edgefirst_gl::gl::NEAREST
        };
        let chroma_filter = if native_planes {
            r8_filter
        } else {
            edgefirst_gl::gl::LINEAR
        };
        let (shift_x, shift_y) = (layout.shift_x, layout.shift_y);
        let chroma_w = src_w.div_ceil(1 << shift_x);
        let chroma_h = src_h.div_ceil(1 << shift_y);
        let combined_h = src_fmt.combined_plane_height(src_h).ok_or_else(|| {
            Error::NotSupported(format!(
                "draw_nv_texture_2d: {src_fmt:?} is not semi-planar"
            ))
        })?;
        let prog_id = self.nv_r8_programs[program].id;

        let state = &mut self.nv_r8_uniforms[program];
        let locs = state.locs;
        // Uploads the matrix only when this program's (encoding, range) or
        // luma-only state changed.
        let upload_colorimetry =
            state.last_colorimetry != Some(colorimetry) || state.last_luma_only != luma_only;

        unsafe {
            edgefirst_gl::gl::UseProgram(prog_id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.nv_r8_texture.id);
            // NEAREST for the texelFetch programs; LINEAR where the hardware
            // filters luma.
            super::core::set_tex_filter_clamp(edgefirst_gl::gl::TEXTURE_2D, r8_filter);

            match r8_src {
                Some(egl_img) => {
                    if self
                        .nv_r8_texture
                        .bind_egl_image(&self.gl_context, src_key, egl_img)?
                    {
                        check_gl_error(function!(), line!())?;
                        log::trace!("draw_nv_texture_2d: bound new R8 EGLImage id={luma_id:#x}");
                    } else {
                        log::trace!(
                            "draw_nv_texture_2d: reusing bound R8 EGLImage id={luma_id:#x}"
                        );
                    }
                }
                None => {
                    // Upload path: the combined semi-planar buffer as an R8 texture
                    // (tex_width × combined_plane_height), from host memory or from a
                    // PBO.
                    let combined_h = src_fmt.combined_plane_height(src_h).ok_or_else(|| {
                        Error::NotSupported(format!(
                            "draw_nv_texture_2d upload: {src_fmt:?} is not semi-planar"
                        ))
                    })?;
                    let needed = tex_width as usize * combined_h;
                    let pixels = SrcPixels::open(src, &self.owned_pbos)?;
                    pixels.record(&mut self.convert_stats, convert_span);
                    // Uploads into a fresh texture name on Adreno.
                    if self.is_adreno {
                        self.nv_r8_texture = Texture::new();
                        edgefirst_gl::gl::BindTexture(
                            edgefirst_gl::gl::TEXTURE_2D,
                            self.nv_r8_texture.id,
                        );
                        super::core::set_tex_filter_clamp(edgefirst_gl::gl::TEXTURE_2D, r8_filter);
                    }
                    // Allocates fresh storage every frame.
                    self.nv_r8_texture.forget_storage();
                    self.nv_r8_texture.upload(
                        edgefirst_gl::gl::TEXTURE_2D,
                        tex_width as usize,
                        combined_h,
                        edgefirst_gl::gl::RED,
                        needed,
                        pixels.source(),
                    )?;
                    check_gl_error(function!(), line!())?;
                    log::trace!("draw_nv_texture_2d: uploaded R8 ({tex_width}x{combined_h})");
                }
            }

            if sampling == NvSampling::Hardware {
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE1);
                let bound =
                    self.bind_nv_chroma(src, src_key, uv_import, chroma_w, chroma_h, chroma_filter);
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                bound?;
            }

            // Per-source uniforms.
            edgefirst_gl::gl::Uniform2i(locs.img_size, src_w as i32, src_h as i32);
            edgefirst_gl::gl::Uniform1i(locs.tex_width, tex_width);
            edgefirst_gl::gl::Uniform2i(locs.chroma_shift, chroma_shift_x, chroma_shift_y);
            edgefirst_gl::gl::Uniform1i(locs.chroma_lines, chroma_lines);
            if sampling == NvSampling::Hardware {
                // The crop's half-texel-inset rectangle on each texture.
                let native_extent = native_extent.filter(|_| native_planes);
                let (tw, th) = match native_extent {
                    Some((w, h)) => (w as f32, h as f32),
                    None => (tex_width as f32, combined_h as f32),
                };
                edgefirst_gl::gl::Uniform2f(locs.luma_scale, src_w as f32 / tw, src_h as f32 / th);
                edgefirst_gl::gl::Uniform4f(
                    locs.luma_clamp,
                    (src_rect.left as f32 + 0.5) / tw,
                    (src_rect.top as f32 + 0.5) / th,
                    (src_rect.right as f32 - 0.5) / tw,
                    (src_rect.bottom as f32 - 0.5) / th,
                );
                // The chroma texture is the native extent shifted by the subsampling
                // when the import has one.
                let (chroma_tex_w, chroma_tex_h) = match native_extent {
                    Some((w, h)) => (w >> shift_x, h >> shift_y),
                    None => (chroma_w, chroma_h),
                };
                let (cw, ch) = (chroma_tex_w as f32, chroma_tex_h as f32);
                edgefirst_gl::gl::Uniform2f(
                    locs.chroma_scale,
                    src_w as f32 / (chroma_tex_w << shift_x) as f32,
                    src_h as f32 / (chroma_tex_h << shift_y) as f32,
                );
                edgefirst_gl::gl::Uniform4f(
                    locs.chroma_clamp,
                    ((src_rect.left >> shift_x) as f32 + 0.5) / cw,
                    ((src_rect.top >> shift_y) as f32 + 0.5) / ch,
                    (src_rect.right.div_ceil(1 << shift_x) as f32 - 0.5) / cw,
                    (src_rect.bottom.div_ceil(1 << shift_y) as f32 - 0.5) / ch,
                );
            }
            edgefirst_gl::gl::Uniform4i(
                locs.src_rect,
                src_rect.left as i32,
                src_rect.top as i32,
                src_rect.right as i32 - 1,
                src_rect.bottom as i32 - 1,
            );

            if upload_colorimetry {
                let coeffs = crate::colorimetry::yuv_to_rgb_coeffs(colorimetry.0, colorimetry.1);
                // A Grey destination uses the luma terms only.
                let coeffs = if luma_only {
                    coeffs.luma_only()
                } else {
                    coeffs
                };
                edgefirst_gl::gl::Uniform1f(locs.y_offset, coeffs.y_offset);
                edgefirst_gl::gl::Uniform1f(locs.y_scale, coeffs.y_scale);
                edgefirst_gl::gl::Uniform1f(locs.c_vr, coeffs.c_vr);
                edgefirst_gl::gl::Uniform1f(locs.c_ug, coeffs.c_ug);
                edgefirst_gl::gl::Uniform1f(locs.c_vg, coeffs.c_vg);
                edgefirst_gl::gl::Uniform1f(locs.c_ub, coeffs.c_ub);
            }

            self.draw_sample_rect_quad(src_rect, (src_w, src_h), dst_roi, rotation_offset, flip);
            if sampling == NvSampling::Hardware {
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE1);
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, 0);
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            }
        }
        self.last_nv_sampling = Some(sampling);
        check_gl_error(function!(), line!())?;
        // Records the colorimetry uploaded to the program.
        if upload_colorimetry {
            self.nv_r8_uniforms[program].last_colorimetry = Some(colorimetry);
            self.nv_r8_uniforms[program].last_luma_only = luma_only;
        }
        Ok(())
    }

    /// Render a packed `Rgb` DMA-BUF source into the bound FBO through a
    /// zero-copy import: native `BG24` where [`Self::rgb_native_import`]
    /// holds, else an R8 view of the buffer. Returns `NotSupported` where
    /// neither import is possible.
    #[allow(clippy::too_many_arguments)]
    fn draw_rgb_dma_source(
        &mut self,
        src: &Tensor<u8>,
        src_roi: RegionOfInterest,
        src_rect: NvSampleRect,
        dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        convert_span: &tracing::Span,
    ) -> Result<(), Error> {
        self.last_rgb_source_path = None;
        if self.rgb_native_import {
            match self.get_or_create_egl_image(CacheKind::Src, src, PixelFormat::Rgb) {
                Ok(img) => {
                    self.convert_stats.src_imports += 1;
                    convert_span.record("src_feed", "import");
                    self.draw_camera_texture_eglimage(
                        src,
                        PixelFormat::Rgb,
                        img,
                        src_roi,
                        dst_roi,
                        rotation_offset,
                        flip,
                        is_int8,
                        None,
                    )?;
                    self.last_rgb_source_path = Some(RgbSourcePath::Native);
                    return Ok(());
                }
                Err(e) => log::debug!("native Rgb import declined ({e}); using the R8 view"),
            }
        }
        let import = if self.packed_r8_import {
            self.get_or_create_nv_r8_egl_image(src, PixelFormat::Rgb)
        } else {
            Err(Error::NotSupported(
                "no packed-RGB R8 import on this platform".into(),
            ))
        };
        let r8_src = match import {
            Ok(img) => img,
            Err(e) => {
                self.convert_stats.zero_copy_declines += 1;
                let (w, h) = (src.width().unwrap_or(0), src.height().unwrap_or(0));
                log::debug!("R8 import of an Rgb source ({w}x{h}) declined: {e}");
                return Err(Error::NotSupported(format!(
                    "packed Rgb source ({w}x{h}) cannot be imported zero-copy ({e}); \
                     converting on the CPU"
                )));
            }
        };
        self.convert_stats.src_imports += 1;
        convert_span.record("src_feed", "import");
        let sampling = self.draw_packed_r8_texture_2d(
            src,
            PixelFormat::Rgb,
            r8_src,
            src_rect,
            dst_roi,
            rotation_offset,
            flip,
            is_int8,
        )?;
        self.last_rgb_source_path = Some(RgbSourcePath::R8Import(sampling));
        Ok(())
    }

    /// Render the luma of a `Yuyv` DMA-BUF source into the bound Grey target
    /// through an R8 view of the buffer (the Y byte of each pixel, range
    /// expanded). A refused import returns `NotSupported`.
    #[allow(clippy::too_many_arguments)]
    fn draw_yuyv_luma_dma_source(
        &mut self,
        src: &Tensor<u8>,
        src_rect: NvSampleRect,
        dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
        convert_span: &tracing::Span,
    ) -> Result<(), Error> {
        let import = if self.packed_r8_import {
            self.get_or_create_nv_r8_egl_image(src, PixelFormat::Yuyv)
        } else {
            Err(Error::NotSupported(
                "no packed R8 import on this platform".into(),
            ))
        };
        let r8_src = import.map_err(|e| {
            self.convert_stats.zero_copy_declines += 1;
            log::debug!("R8 import of a Yuyv source for a Grey destination declined: {e}");
            Error::NotSupported(format!(
                "Yuyv source cannot be imported as an R8 view ({e}); converting on the CPU"
            ))
        })?;
        self.convert_stats.src_imports += 1;
        convert_span.record("src_feed", "import");
        self.draw_packed_r8_texture_2d(
            src,
            PixelFormat::Yuyv,
            r8_src,
            src_rect,
            dst_roi,
            rotation_offset,
            flip,
            is_int8,
        )?;
        Ok(())
    }

    /// Render a packed source through the R8 programs from its R8 import
    /// `r8_src`: one R8 texture `row_stride` texels wide and `height` rows
    /// tall. An `Rgb` source renders RGBA (each pixel its three bytes), a
    /// `Yuyv` one its range-expanded luma, for a Grey destination. Renders
    /// into the currently-bound FBO and returns the sampling drawn.
    #[allow(clippy::too_many_arguments)]
    fn draw_packed_r8_texture_2d(
        &mut self,
        src: &Tensor<u8>,
        src_fmt: PixelFormat,
        r8_src: PlatformHandle,
        src_rect: NvSampleRect,
        dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        is_int8: bool,
    ) -> Result<NvSampling, Error> {
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let tex_width = src
            .effective_row_stride()
            .unwrap_or(src_w * src_fmt.channels());
        let src_key = BufferImportKey::from_tensor(src, src_fmt, false);
        let (programs, locs) = match src_fmt {
            PixelFormat::Rgb => (&self.rgb_r8_programs, &self.rgb_r8_locs),
            PixelFormat::Yuyv => (&self.yuyv_luma_r8_programs, &self.yuyv_luma_r8_locs),
            _ => {
                return Err(Error::NotSupported(format!(
                    "no packed R8 program for {src_fmt:?}"
                )))
            }
        };

        let colorimetry =
            (src_fmt == PixelFormat::Yuyv).then(|| nv_colorimetry(src.colorimetry(), src.height()));
        // Selects the sampling: `Single` without bilinear resampling, `Hardware`
        // when the hardware filter is enabled and the luma is not limited-range,
        // `Shader` otherwise.
        let bilinear =
            src_rect.resample && self.interpolation.supported() == crate::Interpolation::Bilinear;
        let hw_filter = self.nv_hw_filter
            && !colorimetry
                .is_some_and(|(_, range)| range == edgefirst_tensor::ColorRange::Limited);
        let sampling = if !bilinear {
            NvSampling::Single
        } else if hw_filter {
            NvSampling::Hardware
        } else {
            NvSampling::Shader
        };
        let filter = if sampling == NvSampling::Hardware {
            edgefirst_gl::gl::LINEAR
        } else {
            edgefirst_gl::gl::NEAREST
        };
        let program = sampling.program(is_int8);
        let prog_id = programs[program].id;
        let locs = locs[program];
        let luma = colorimetry
            .map(|(encoding, range)| crate::colorimetry::yuv_to_rgb_coeffs(encoding, range));

        unsafe {
            edgefirst_gl::gl::UseProgram(prog_id);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.nv_r8_texture.id);
            super::core::set_tex_filter_clamp(edgefirst_gl::gl::TEXTURE_2D, filter);
            if self
                .nv_r8_texture
                .bind_egl_image(&self.gl_context, src_key, r8_src)?
            {
                check_gl_error(function!(), line!())?;
            }
            edgefirst_gl::gl::Uniform2i(locs.img_size, src_w as i32, src_h as i32);
            edgefirst_gl::gl::Uniform2f(locs.inv_tex, 1.0 / tex_width as f32, 1.0 / src_h as f32);
            edgefirst_gl::gl::Uniform4i(
                locs.src_rect,
                src_rect.left as i32,
                src_rect.top as i32,
                src_rect.right as i32 - 1,
                src_rect.bottom as i32 - 1,
            );
            if let Some(k) = luma {
                edgefirst_gl::gl::Uniform1f(locs.y_offset, k.y_offset);
                edgefirst_gl::gl::Uniform1f(locs.y_scale, k.y_scale);
            }
            self.draw_sample_rect_quad(src_rect, (src_w, src_h), dst_roi, rotation_offset, flip);
        }
        check_gl_error(function!(), line!())?;
        Ok(sampling)
    }

    /// Draw the quad of an R8-view draw (NV or packed RGB) with its program
    /// and source texture already bound: `dst_roi` on the bound target,
    /// flipped as asked, sampling `src_rect` of a `src_w`x`src_h` source
    /// turned by `rotation_offset` quarter turns.
    ///
    /// # Safety
    /// On the GL thread with this processor's context current.
    unsafe fn draw_sample_rect_quad(
        &self,
        src_rect: NvSampleRect,
        (src_w, src_h): (usize, usize),
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0.,
                dst_roi.right,
                dst_roi.top,
                0.,
                dst_roi.right,
                dst_roi.bottom,
                0.,
                dst_roi.left,
                dst_roi.bottom,
                0.,
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let src_roi = RegionOfInterest {
                left: src_rect.left as f32 / src_w as f32,
                top: src_rect.bottom as f32 / src_h as f32,
                right: src_rect.right as f32 / src_w as f32,
                bottom: src_rect.top as f32 / src_h as f32,
            };
            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }
    }

    /// Look up or create the R8 EGLImage for an NV* or packed `Rgb` source
    /// tensor, cached in `nv_r8_egl_cache`.
    fn get_or_create_nv_r8_egl_image(
        &mut self,
        img: &Tensor<u8>,
        img_fmt: PixelFormat,
    ) -> Result<PlatformHandle, crate::Error> {
        // Validates the import identity before the cache lookup.
        Platform::validate_import_identity(img, "NV source")?;
        // Refuses an unaligned plane-0 offset on GPUs that misplace it.
        let offset = img.plane_offset().unwrap_or(0);
        if rejects_unaligned_import_offset(self.misplaces_unaligned_offsets(), offset) {
            return Err(crate::Error::NotSupported(format!(
                "this GPU misreads a DMA-BUF imported at a plane offset that \
                 is not {DMA_IMPORT_OFFSET_ALIGN}-byte aligned ({offset}); \
                 uploading the window instead"
            )));
        }
        let id = BufferImportKey::from_tensor(img, img_fmt, false);
        let luma_id = id.luma_id;

        if let Some(cached) = self.nv_r8_egl_cache.lookup(&id) {
            log::trace!("nv_r8_egl_cache hit: id={luma_id:#x}");
            return Ok(Platform::import_handle(cached));
        }
        log::trace!("nv_r8_egl_cache miss: id={luma_id:#x}");

        let egl_image_obj = Platform::import_buffer_nv_r8(&self.gl_context, img, img_fmt)?;
        self.nv_r8_texture.invalidate_egl_binding();

        let handle = Platform::import_handle(&egl_image_obj);
        self.nv_r8_egl_cache.insert(id, egl_image_obj, None);
        Ok(handle)
    }

    /// Bind the chroma plane of `src` as (U, V) texels to `nv_uv_texture` on
    /// the active texture unit, with LINEAR filtering: the `import` when there
    /// is one, else an upload of the plane from the source's pixels.
    fn bind_nv_chroma(
        &mut self,
        src: &Tensor<u8>,
        src_key: BufferImportKey,
        import: Option<PlatformHandle>,
        chroma_w: usize,
        chroma_h: usize,
        filter: u32,
    ) -> Result<(), Error> {
        if self.is_adreno && import.is_none() {
            // Replaces the texture before an upload on Adreno.
            self.nv_uv_texture = Texture::new();
        }
        unsafe {
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, self.nv_uv_texture.id);
            super::core::set_tex_filter_clamp(edgefirst_gl::gl::TEXTURE_2D, filter);
        }
        if let Some(handle) = import {
            // SAFETY: `handle` comes from `nv_uv_egl_cache`, which owns the import
            // for the duration of this draw; the texture is bound to the active
            // unit.
            unsafe {
                self.nv_uv_texture
                    .bind_egl_image(&self.gl_context, src_key, handle)?;
            }
            return check_gl_error(function!(), line!());
        }
        let src_w = src.width().ok_or(Error::NotAnImage)?;
        let src_h = src.height().ok_or(Error::NotAnImage)?;
        let layout = src
            .format()
            .and_then(|f| f.chroma_layout())
            .ok_or_else(|| Error::NotSupported("NV chroma upload: not semi-planar".into()))?;
        let stride = src.effective_row_stride().unwrap_or(src_w);
        let pitch = stride * layout.uv_rows_per_luma;
        let pixels = SrcPixels::open(src, &self.owned_pbos)?;
        self.nv_uv_texture.forget_storage();
        unsafe {
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, (pitch / 2) as i32);
        }
        let uploaded = self.nv_uv_texture.upload(
            edgefirst_gl::gl::TEXTURE_2D,
            chroma_w,
            chroma_h,
            edgefirst_gl::gl::RG,
            pitch * (chroma_h - 1) + chroma_w * 2,
            pixels.source().offset_by(stride * src_h),
        );
        unsafe {
            edgefirst_gl::gl::PixelStorei(edgefirst_gl::gl::UNPACK_ROW_LENGTH, 0);
        }
        uploaded?;
        check_gl_error(function!(), line!())
    }

    /// Look up or create the chroma-plane (U, V) EGLImage of a single-plane
    /// NV* source for [`NvSampling::Hardware`], cached in `nv_uv_egl_cache`.
    /// Declines an unaligned chroma plane offset on GPUs that misplace it.
    fn get_or_create_nv_uv_egl_image(
        &mut self,
        img: &Tensor<u8>,
        img_fmt: PixelFormat,
    ) -> Result<PlatformHandle, crate::Error> {
        // A two-plane source's chroma plane is imported regardless of a recorded
        // refusal.
        let native = Platform::nv_native_planes(img).is_some();
        if !native && (!Platform::NV_CHROMA_IMPORT || self.nv_uv_import_refused) {
            return Err(crate::Error::NotSupported(
                "no chroma-plane import on this platform or driver".into(),
            ));
        }
        Platform::validate_import_identity(img, "NV chroma source")?;
        let height = img.height().ok_or(Error::NotAnImage)?;
        let stride = img
            .effective_row_stride()
            .unwrap_or(img.width().ok_or(Error::NotAnImage)?);
        let offset = img.plane_offset().unwrap_or(0) + stride * height;
        if rejects_unaligned_import_offset(self.misplaces_unaligned_offsets(), offset) {
            return Err(crate::Error::NotSupported(format!(
                "chroma plane offset {offset} is not {DMA_IMPORT_OFFSET_ALIGN}-byte aligned"
            )));
        }
        let id = BufferImportKey::from_tensor(img, img_fmt, false);
        if let Some(cached) = self.nv_uv_egl_cache.lookup(&id) {
            return Ok(Platform::import_handle(cached));
        }
        // Records a refused import so it is not attempted again.
        let egl_image_obj = Platform::import_buffer_nv_chroma(&self.gl_context, img, img_fmt)
            .inspect_err(|e| {
                if !native {
                    log::info!(
                        "NV chroma-plane import refused ({e}); resizing with the shader filter"
                    );
                    self.nv_uv_import_refused = true;
                }
            })?;
        self.nv_uv_texture.invalidate_egl_binding();
        let handle = Platform::import_handle(&egl_image_obj);
        self.nv_uv_egl_cache.insert(id, egl_image_obj, None);
        Ok(handle)
    }

    /// Look up or create an EGLImage for a DMA tensor, returning the EGL image handle.
    ///
    /// The cache retains ownership of the `EglImage` value; the handle remains
    /// valid as long as the entry lives in the cache.
    fn get_or_create_egl_image(
        &mut self,
        cache: CacheKind,
        img: &Tensor<u8>,
        img_fmt: PixelFormat,
    ) -> Result<PlatformHandle, crate::Error> {
        // Test hook: refuses a destination import.
        #[cfg(test)]
        if cache == CacheKind::Dst && self.fail_dst_import {
            return Err(crate::Error::OpenGl(
                "EDGEFIRST test hook: zero-copy destination import refused".to_string(),
            ));
        }
        // Validates the import identity before the cache lookup.
        Platform::validate_import_identity(
            img,
            match cache {
                CacheKind::Src => "source",
                CacheKind::Dst => "destination",
            },
        )?;
        // Declines a source import whose plane offsets are unaligned on GPUs
        // that misplace them.
        if cache == CacheKind::Src {
            // The plane-0 offset the import passes to EGL.
            let offset = source_import_plane0_offset(img, img_fmt);
            // The plane-1 offset of an NV12 import.
            let chroma_offset = (img_fmt == PixelFormat::Nv12).then(|| {
                if img.is_multiplane() {
                    // A separate chroma DMA-BUF imports at its own offset.
                    img.chroma().and_then(|c| c.plane_offset()).unwrap_or(0)
                } else {
                    nv12_plane1_offset(
                        offset,
                        img.effective_row_stride()
                            .unwrap_or_else(|| img.width().unwrap_or(0)),
                        img.height().unwrap_or(0),
                    )
                }
            });
            let planes = [
                Some(("plane 0", offset)),
                chroma_offset.map(|o| ("chroma plane", o)),
            ];
            if let Some((which, bad)) = planes.into_iter().flatten().find(|&(_, o)| {
                rejects_unaligned_import_offset(self.misplaces_unaligned_offsets(), o)
            }) {
                return Err(crate::Error::NotSupported(format!(
                    "this GPU misreads a DMA-BUF imported at a plane offset that \
                     is not {DMA_IMPORT_OFFSET_ALIGN}-byte aligned ({which} at \
                     {bad}); uploading the window instead"
                )));
            }
            // Declines a source with chroma in a separate DMA-BUF on Adreno.
            if self.is_adreno && chroma_in_separate_buffer(img) {
                return Err(crate::Error::NotSupported(
                    "Adreno misreads a semi-planar source whose chroma is a \
                     separate DMA-BUF (only the first buffer is imported)"
                        .into(),
                ));
            }
        }
        // The key covers identity, offset and geometry; only a destination view
        // collapses onto its parent key.
        let id = BufferImportKey::from_tensor(img, img_fmt, cache == CacheKind::Dst);
        let luma_id = id.luma_id;

        let egl_cache = match cache {
            CacheKind::Src => &mut self.src_egl_cache,
            CacheKind::Dst => &mut self.dst_egl_cache,
        };
        if let Some(cached) = egl_cache.lookup(&id) {
            log::trace!("ImportCache {:?} hit: id={luma_id:#x}", cache);
            return Ok(Platform::import_handle(cached));
        }
        log::trace!("ImportCache {:?} miss: id={luma_id:#x}", cache);

        // Creates the EGL image before evicting. A destination view imports its
        // parent; a source view imports its own region.
        let for_dst = cache == CacheKind::Dst;
        let egl_image_obj = Platform::import_buffer(&self.gl_context, img, img_fmt, for_dst)?;

        // Creates a GL renderbuffer backed by this EGLImage for use as an FBO
        // color attachment when EDGEFIRST_OPENGL_RENDERSURFACE=1.
        let rbo = if cache == CacheKind::Dst && self.use_renderbuffer {
            let mut rbo: u32 = 0;
            unsafe {
                edgefirst_gl::gl::GenRenderbuffers(1, &mut rbo);
                edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                Platform::attach_renderbuffer_storage(
                    &self.gl_context,
                    Platform::import_handle(&egl_image_obj),
                )?;
                if let Err(e) = check_gl_error(function!(), line!()) {
                    edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo);
                    return Err(e);
                }
            }
            Some(rbo)
        } else {
            None
        };

        // Invalidates the texture binding state of the cache being inserted into.
        match cache {
            CacheKind::Src => self.invalidate_src_textures(),
            CacheKind::Dst => self.invalidate_dst_textures(),
        }

        let handle = Platform::import_handle(&egl_image_obj);
        let egl_cache = match cache {
            CacheKind::Src => &mut self.src_egl_cache,
            CacheKind::Dst => &mut self.dst_egl_cache,
        };
        egl_cache.insert(id, egl_image_obj, rbo);
        Ok(handle)
    }

    /// Where the cached source import puts `img`'s logical image on the
    /// texture it covers ([`GlPlatform::import_extent`] +
    /// [`GlPlatform::import_origin`]).
    /// [`ImportMap::WHOLE`](super::render::ImportMap::WHOLE) when the import
    /// is the logical image or the source was fed by upload.
    fn cached_src_import_map(
        &self,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> super::render::ImportMap {
        let id = BufferImportKey::from_tensor(img, fmt, false);
        self.src_egl_cache
            .entries
            .get(&id)
            .map(|entry| {
                let map = super::render::ImportMap {
                    extent: Platform::import_extent(&entry.import),
                    origin: Platform::import_origin(&entry.import),
                };
                debug_assert!(
                    map.origin == (0, 0) || map.extent.is_some(),
                    "an import origin {:?} without an extent is dropped by the fold",
                    map.origin,
                );
                map
            })
            .unwrap_or(super::render::ImportMap::WHOLE)
    }

    /// [`Self::cached_src_import_map`] applied to a source rectangle: turns
    /// logical-image coordinates into texture coordinates for the uv-sampling
    /// paths.
    fn scale_src_roi(
        &self,
        roi: RegionOfInterest,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> RegionOfInterest {
        let (Some(w), Some(h)) = (img.width(), img.height()) else {
            return roi;
        };
        super::render::scale_roi_to_import(roi, (w, h), self.cached_src_import_map(img, fmt))
    }

    /// The `src_extent` location of the `sampler2D` source program
    /// `draw_src_texture` selects for `is_int8`.
    fn texture_src_extent_loc(&self, is_int8: bool) -> i32 {
        let [plain, int8] = self.texture_src_extent_locs;
        if is_int8 {
            int8
        } else {
            plain
        }
    }

    /// The `src_extent` location of the external-OES program the two camera
    /// draws select for `is_int8` and `planar`.
    fn external_src_extent_loc(&self, is_int8: bool, planar: bool) -> i32 {
        let [yuv, yuv_int8, pl, pl_int8] = self.external_src_extent_locs;
        match (planar, is_int8) {
            (false, false) => yuv,
            (false, true) => yuv_int8,
            (true, false) => pl,
            (true, true) => pl_int8,
        }
    }

    /// The float paths' `(src_rect_uv, src_extent)` uniforms. When
    /// [`FloatSrcFeed::Import`] fed this convert the uv is mapped onto the
    /// texture the import covers and the extent is the logical image's
    /// half-texel-inset rectangle on it. For the upload and PBO feeds the uv
    /// passes through and the extent is the texture's own inset.
    pub(super) fn float_src_mapping_for_feed(
        &self,
        feed: float::FloatSrcFeed,
        src_rect_uv: [f32; 4],
        src_u8: &Tensor<u8>,
        src_w: usize,
        src_h: usize,
    ) -> ([f32; 4], [f32; 4]) {
        let map = if feed == float::FloatSrcFeed::Import {
            self.cached_src_import_map(src_u8, PixelFormat::Rgba)
        } else {
            super::render::ImportMap::WHOLE
        };
        (
            super::render::scale_uv_rect_to_import(src_rect_uv, (src_w, src_h), map),
            super::render::sample_clamp_rect((src_w, src_h), map),
        )
    }

    /// Look up the renderbuffer ID for a cached destination EGLImage.
    fn cached_dst_renderbuffer<T>(&self, img: &Tensor<T>, fmt: PixelFormat) -> Option<u32>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        let id = BufferImportKey::from_tensor(img, fmt, true);
        self.dst_egl_cache
            .entries
            .get(&id)
            .and_then(|entry| entry.renderbuffer)
    }

    /// Get or create an EGLImage for a packed DMA destination with
    /// reinterpreted dimensions, cached in the dst cache by buffer identity.
    ///
    /// Generic over the tensor element type: `u8` for the packed-RGB
    /// destination (`PackedImportFormat::Rgba8888`) and `f16` for the
    /// RGBA16F-packed planar destination (`Rgba16161616F`).
    fn get_or_create_egl_image_rgb<T>(
        &mut self,
        img: &Tensor<T>,
        img_fmt: PixelFormat,
        width: usize,
        height: usize,
        packed: super::platform::PackedImportFormat,
    ) -> Result<PlatformHandle, crate::Error>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        // Test hook: refuses a destination import.
        #[cfg(test)]
        if self.fail_dst_import {
            return Err(crate::Error::OpenGl(
                "EDGEFIRST test hook: zero-copy destination import refused".to_string(),
            ));
        }
        Platform::validate_import_identity(img, "packed destination")?;
        let id = BufferImportKey::from_tensor(img, img_fmt, true);

        if let Some(cached) = self.dst_egl_cache.lookup(&id) {
            log::trace!("ImportCache dst (RGB) hit: id={:#x}", id.luma_id);
            return Ok(Platform::import_handle(cached));
        }
        log::trace!("ImportCache dst (RGB) miss: id={:#x}", id.luma_id);
        // Invalidates the destination texture binding state on a cache miss.
        self.invalidate_dst_textures();

        let egl_image_obj =
            Platform::import_buffer_packed(&self.gl_context, img, width, height, packed)?;
        let handle = Platform::import_handle(&egl_image_obj);

        let rbo = if self.use_renderbuffer {
            let mut rbo: u32 = 0;
            unsafe {
                edgefirst_gl::gl::GenRenderbuffers(1, &mut rbo);
                edgefirst_gl::gl::BindRenderbuffer(edgefirst_gl::gl::RENDERBUFFER, rbo);
                Platform::attach_renderbuffer_storage(
                    &self.gl_context,
                    Platform::import_handle(&egl_image_obj),
                )?;
                if let Err(e) = check_gl_error(function!(), line!()) {
                    edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo);
                    return Err(e);
                }
            }
            Some(rbo)
        } else {
            None
        };

        self.dst_egl_cache.insert(id, egl_image_obj, rbo);
        Ok(handle)
    }

    /// Look up or create the render targets of an NV12 destination, cached in
    /// `dst_egl_cache`: one entry for a single-plane destination, one per
    /// plane for a multiplane one.
    fn get_or_create_nv12_dst(
        &mut self,
        dst: &Tensor<u8>,
    ) -> Result<NvDstPlanes<PlatformHandle>, crate::Error> {
        // Test hook: refuses a destination import.
        #[cfg(test)]
        if self.fail_dst_import {
            return Err(crate::Error::OpenGl(
                "EDGEFIRST test hook: zero-copy destination import refused".to_string(),
            ));
        }
        Platform::validate_import_identity(dst, "NV12 destination")?;
        let (dst_w, dst_h) = (
            dst.width().ok_or(Error::NotAnImage)?,
            dst.height().ok_or(Error::NotAnImage)?,
        );
        let luma_key = BufferImportKey::from_tensor(dst, PixelFormat::Nv12, true);
        let chroma_key = dst
            .chroma()
            .map(|c| BufferImportKey::from_tensor(c, PixelFormat::Nv12, true));
        let luma_cached = self.dst_egl_cache.lookup(&luma_key).is_some();
        let chroma_cached = chroma_key
            .as_ref()
            .map(|key| self.dst_egl_cache.lookup(key).is_some());

        if !(luma_cached && chroma_cached.unwrap_or(true)) {
            let planes = Platform::import_nv12_dst(&self.gl_context, dst)?;
            self.invalidate_dst_textures();
            match planes {
                NvDstPlanes::Combined {
                    plane,
                    width,
                    height,
                } => {
                    if (height as usize) != dst_h + dst_h / 2 || (width as usize) < dst_w {
                        return Err(Error::NotSupported(format!(
                            "NV12 destination import is {width}x{height}, which does not hold a \
                             {dst_w}x{dst_h} image"
                        )));
                    }
                    self.dst_egl_cache.insert(luma_key, plane, None);
                }
                NvDstPlanes::Split { luma, chroma, .. } => {
                    if !luma_cached {
                        self.dst_egl_cache.insert(luma_key, luma, None);
                    }
                    if let (Some(false), Some(key)) = (chroma_cached, chroma_key) {
                        self.dst_egl_cache.insert(key, chroma, None);
                    }
                }
            }
        }

        let handle = |cache: &ImportCache<PlatformImport>, key: &BufferImportKey| {
            cache
                .entries
                .get(key)
                .map(|cached| Platform::import_handle(&cached.import))
                .ok_or_else(|| {
                    Error::NotSupported(
                        "the destination import cache is too small to hold an NV12 destination"
                            .into(),
                    )
                })
        };
        let luma = handle(&self.dst_egl_cache, &luma_key)?;
        Ok(match chroma_key {
            Some(key) => NvDstPlanes::Split {
                luma,
                chroma: handle(&self.dst_egl_cache, &key)?,
                chroma_rows: (dst_h / 2) as u32,
            },
            None => NvDstPlanes::Combined {
                plane: luma,
                width: dst_w as u32,
                height: (dst_h + dst_h / 2) as u32,
            },
        })
    }

    // Reshapes the segmentation to be compatible with RGBA texture array rendering.
    fn reshape_segmentation_to_rgba(&self, segmentation: &[u8], shape: [usize; 3]) -> Vec<u8> {
        let [height, width, classes] = shape;

        let n_layer_stride = height * width * 4;
        let n_row_stride = width * 4;
        let n_col_stride = 4;
        let row_stride = width * classes;
        let col_stride = classes;

        let mut new_segmentation = vec![0u8; n_layer_stride * classes.div_ceil(4)];

        for i in 0..height {
            for j in 0..width {
                for k in 0..classes.div_ceil(4) * 4 {
                    if k >= classes {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = 0;
                    } else {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = segmentation[i * row_stride + j * col_stride + k];
                    }
                }
            }
        }

        new_segmentation
    }

    fn render_modelpack_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 3],
    ) -> Result<(), crate::Error> {
        log::debug!("start render_segmentation_to_image");

        // TODO: Specialize 2 and 4 classes without rearranging the data.
        let new_segmentation = self.reshape_segmentation_to_rgba(segmentation, shape);

        let [height, width, classes] = shape;

        let format = edgefirst_gl::gl::RGBA;
        let texture_target = edgefirst_gl::gl::TEXTURE_2D_ARRAY;
        self.segmentation_program
            .load_uniform_1i(c"background_index", shape[2] as i32 - 1)?;

        edgefirst_gl::use_program(self.segmentation_program.id);

        edgefirst_gl::bind_texture(texture_target, self.segmentation_texture.id);
        edgefirst_gl::active_texture(edgefirst_gl::gl::TEXTURE0);
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_MIN_FILTER,
            edgefirst_gl::gl::LINEAR as i32,
        );
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_MAG_FILTER,
            edgefirst_gl::gl::LINEAR as i32,
        );
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_WRAP_S,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );

        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_WRAP_T,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );

        edgefirst_gl::tex_image3d(
            texture_target,
            0,
            format as i32,
            width as i32,
            height as i32,
            classes.div_ceil(4) as i32,
            0,
            format,
            edgefirst_gl::gl::UNSIGNED_BYTE,
            Some(&new_segmentation),
        );

        let src_roi = RegionOfInterest {
            left: 0.,
            top: 1.,
            right: 1.,
            bottom: 0.,
        };

        unsafe {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[0..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }

        Ok(())
    }

    /// Bind the instanced segmentation program and configure the persistent
    /// mask texture for a batch of `render_yolo_segmentation` calls.
    fn setup_yolo_segmentation_pass(&self) {
        let texture_target = edgefirst_gl::gl::TEXTURE_2D;
        edgefirst_gl::use_program(self.instanced_segmentation_program.id);
        edgefirst_gl::active_texture(edgefirst_gl::gl::TEXTURE0);
        edgefirst_gl::bind_texture(texture_target, self.segmentation_texture.id);
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_MIN_FILTER,
            edgefirst_gl::gl::LINEAR as i32,
        );
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_MAG_FILTER,
            edgefirst_gl::gl::LINEAR as i32,
        );
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_WRAP_S,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );
        edgefirst_gl::tex_parameteri(
            texture_target,
            edgefirst_gl::gl::TEXTURE_WRAP_T,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );
    }

    /// Render a single pre-decoded YOLO segmentation mask as a coloured
    /// overlay quad. The caller must invoke [`setup_yolo_segmentation_pass`]
    /// once before the first call in a batch.
    ///
    /// Each call uploads the mask via `glTexImage2D` without a zero-pad
    /// border and samples it at texel-centre UVs.
    fn render_yolo_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 2],
        class: usize,
    ) -> Result<(), crate::Error> {
        let [height, width] = shape;
        let texture_target = edgefirst_gl::gl::TEXTURE_2D;

        // Allocates and uploads the mask for this instance.
        edgefirst_gl::tex_image2d(
            texture_target,
            0,
            edgefirst_gl::gl::R8 as i32,
            width as i32,
            height as i32,
            0,
            edgefirst_gl::gl::RED,
            edgefirst_gl::gl::UNSIGNED_BYTE,
            Some(segmentation),
        );

        self.instanced_segmentation_program
            .load_uniform_1i(c"class_index", class as i32)?;

        // Texel-centre UVs.
        let u_lo = 0.5 / width as f32;
        let v_lo = 0.5 / height as f32;
        let u_hi = (width as f32 - 0.5) / width as f32;
        let v_hi = (height as f32 - 0.5) / height as f32;

        // tc.y = 0 is mask row 0 (image top); the NDC top of the quad renders
        // image bottom, so the first two tex vertices get `v_hi`.
        let src_roi = RegionOfInterest {
            left: u_lo,
            top: v_hi,
            right: u_hi,
            bottom: v_lo,
        };

        unsafe {
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.texture_buffer.id);
            edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            edgefirst_gl::gl::BufferSubData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                edgefirst_gl::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );

            // Finishes after each instance on Vivante.
            if self.is_vivante {
                edgefirst_gl::gl::Finish();
            }
        }

        Ok(())
    }

    /// Render YOLO proto segmentation masks using the fused GPU pipeline.
    ///
    /// Dispatches on the proto tensor's runtime dtype via
    /// [`edgefirst_tensor::TensorDyn::dtype`]:
    /// - `I8`: uploads raw int8 as `GL_R8I`, dequantizes in shader (requires
    ///   per-tensor [`edgefirst_tensor::Quantization`] on the proto tensor).
    /// - `F32`: uploads as `GL_R32F` with hardware bilinear (if available),
    ///   or repacks into `RGBA16F`.
    /// - `F16`: repacks into `RGBA16F`.
    fn render_proto_segmentation(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use edgefirst_tensor::{DType, QuantMode, TensorMapTrait, TensorTrait};

        if detect.is_empty() {
            return Ok(());
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(crate::Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        // Interpret shape based on physical layout.
        let (height, width, num_protos) = match proto_data.layout {
            ProtoLayout::Nhwc => (proto_shape[0], proto_shape[1], proto_shape[2]),
            ProtoLayout::Nchw => (proto_shape[1], proto_shape[2], proto_shape[0]),
        };
        let coeff_shape = proto_data.mask_coefficients.shape();
        if coeff_shape.len() != 2 || coeff_shape[1] != num_protos {
            return Err(crate::Error::InvalidShape(format!(
                "mask_coefficients shape {coeff_shape:?} incompatible with protos \
                 {proto_shape:?} (expected [N, {num_protos}])"
            )));
        }
        // No detections: nothing to render.
        if coeff_shape[0] == 0 {
            return Ok(());
        }
        let texture_target = edgefirst_gl::gl::TEXTURE_2D_ARRAY;

        // Plans the upload strategy, program and count uniform; rejects NCHW
        // layouts and unsupported proto dtypes.
        let plan = super::proto_dispatch::plan_proto(
            proto_data.protos.dtype(),
            proto_data.layout,
            self.proto_repack_compute_program.is_some(),
            self.has_float_linear,
            self.int8_interpolation_mode,
        )?;

        let _span = tracing::trace_span!(
            "image.draw.gl.proto",
            dtype = ?proto_data.protos.dtype(),
            upload = ?plan.upload,
            program = ?plan.program,
            num_protos,
            detections = detect.len(),
        )
        .entered();

        // Coefficients as an f32 slice: borrowed for F32, widened into one flat
        // `Vec<f32>` for F16.
        let mc_map_f32;
        let mc_map_f16;
        let mc_map_i8;
        let coeff_widen_f16: Vec<f32>;
        let coeff_dequant: Vec<f32>;
        let coeff_slice: &[f32] = match proto_data.mask_coefficients.dtype() {
            DType::F32 => {
                let t = proto_data.mask_coefficients.as_f32().expect("F32");
                mc_map_f32 = t.map_read()?;
                mc_map_f32.as_slice()
            }
            DType::F16 => {
                let t = proto_data.mask_coefficients.as_f16().expect("F16");
                mc_map_f16 = t.map_read()?;
                coeff_widen_f16 = mc_map_f16.as_slice().iter().map(|v| v.to_f32()).collect();
                &coeff_widen_f16[..]
            }
            DType::I8 => {
                let t = proto_data.mask_coefficients.as_i8().expect("I8");
                mc_map_i8 = t.map_read()?;
                let quant = t.quantization();
                coeff_dequant = super::proto_dispatch::dequant_coeffs(
                    mc_map_i8.as_slice(),
                    quant.as_ref().map(|q| q.mode()),
                    "I8",
                )?;
                &coeff_dequant[..]
            }
            DType::I16 => {
                let t = proto_data.mask_coefficients.as_i16().expect("I16");
                let mc_map_i16 = t.map_read()?;
                let quant = t.quantization();
                coeff_dequant = super::proto_dispatch::dequant_coeffs(
                    mc_map_i16.as_slice(),
                    quant.as_ref().map(|q| q.mode()),
                    "I16",
                )?;
                &coeff_dequant[..]
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported on GL seg path"
                )));
            }
        };

        // Each arm maps the proto tensor and passes a borrowed `ArrayView3` to
        // the helper.
        match proto_data.protos.dtype() {
            DType::I8 => {
                let t = proto_data.protos.as_i8().expect("I8");
                let m = t.map_read()?;
                let quant = t.quantization().ok_or_else(|| {
                    crate::Error::InvalidShape("I8 protos require quantization metadata".into())
                })?;
                // Per-tensor quantization only; per-channel returns NotSupported.
                let (scale, zp) = match quant.mode() {
                    QuantMode::PerTensor { scale, zero_point } => (scale, zero_point),
                    QuantMode::PerTensorSymmetric { scale } => (scale, 0_i32),
                    QuantMode::PerChannel { axis, .. }
                    | QuantMode::PerChannelSymmetric { axis, .. } => {
                        return Err(crate::Error::NotSupported(format!(
                            "GL seg path: per-channel quantization (axis={axis}) \
                             not yet supported — falls back to CPU in caller"
                        )));
                    }
                };
                let protos_view = ndarray::ArrayView3::<i8>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                let quantization = (scale, zp);
                self.render_proto_segmentation_int8(
                    plan,
                    detect,
                    coeff_slice,
                    protos_view,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            DType::F32 => {
                let t = proto_data.protos.as_f32().expect("F32");
                let m = t.map_read()?;
                let protos_view = ndarray::ArrayView3::<f32>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                self.render_proto_layers(
                    plan,
                    detect,
                    coeff_slice,
                    ProtoLayersData::F32(protos_view),
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            DType::F16 => {
                let t = proto_data.protos.as_f16().expect("F16");
                let m = t.map_read()?;
                let protos_view = ndarray::ArrayView3::<half::f16>::from_shape(
                    (height, width, num_protos),
                    m.as_slice(),
                )
                .map_err(|e| crate::Error::InvalidShape(format!("{e}")))?;
                self.render_proto_layers(
                    plan,
                    detect,
                    coeff_slice,
                    ProtoLayersData::F16(protos_view),
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "GL seg path: proto dtype {other:?} not supported"
                )));
            }
        }

        unsafe { edgefirst_gl::gl::Finish() };
        Ok(())
    }

    /// Render detection quads using the active program.
    fn render_proto_detection_quads(
        &self,
        program: &GlProgram,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        num_protos: usize,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;

        // Resolves the per-detection uniform locations once per program.
        let (cached_id, mut loc_coeff, mut loc_class) = self.proto_quad_locs.get();
        if cached_id != program.id {
            unsafe {
                loc_coeff =
                    edgefirst_gl::gl::GetUniformLocation(program.id, c"mask_coeff".as_ptr());
                loc_class =
                    edgefirst_gl::gl::GetUniformLocation(program.id, c"class_index".as_ptr());
            }
            self.proto_quad_locs.set((program.id, loc_coeff, loc_class));
        }

        // One `num_protos`-long coefficient slice per detection.
        for (idx, (det, coeff)) in detect
            .iter()
            .zip(mask_coefficients.chunks_exact(num_protos))
            .enumerate()
        {
            let color_index = color_mode.index(idx, det.label);
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (i, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[i / 4][i % 4] = *val;
            }

            unsafe {
                edgefirst_gl::gl::Uniform4fv(loc_coeff, 8, packed_coeff.as_ptr() as *const f32);
                edgefirst_gl::gl::Uniform1i(loc_class, color_index as i32);
            }

            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(det.bbox.xmin),
                top: cvt_screen_coord(det.bbox.ymax),
                right: cvt_screen_coord(det.bbox.xmax),
                bottom: cvt_screen_coord(det.bbox.ymin),
            };

            // Proto texture coords: tc.y = 0 is image top and tc.y = 1 is image
            // bottom. The NDC top of the quad takes tc.y = ymax and the NDC bottom
            // takes tc.y = ymin.
            let src_roi = RegionOfInterest {
                left: det.bbox.xmin,
                top: det.bbox.ymax,
                right: det.bbox.xmax,
                bottom: det.bbox.ymin,
            };

            unsafe {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.bottom,
                    0.,
                    dst_roi.left,
                    dst_roi.bottom,
                    0.,
                ];
                edgefirst_gl::gl::BufferSubData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                );

                edgefirst_gl::gl::BindBuffer(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    self.texture_buffer.id,
                );
                edgefirst_gl::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

                let texture_vertices: [f32; 8] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];
                edgefirst_gl::gl::BufferSubData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    texture_vertices.as_ptr() as *const c_void,
                );

                let vertices_index: [u32; 4] = [0, 1, 2, 3];
                edgefirst_gl::gl::DrawElements(
                    edgefirst_gl::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    edgefirst_gl::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        Ok(())
    }

    /// Int8 proto path: upload raw i8 protos per `plan.upload`, render per
    /// `plan.program`.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_int8(
        &mut self,
        plan: super::proto_dispatch::ProtoPlan,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        protos: ndarray::ArrayView3<'_, i8>,
        quantization: (f32, i32),
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        // Protos are (H, W, num_protos) in row-major HWC. The GL texture is
        // layer-first CHW (one proto per layer).
        if plan.upload == super::proto_dispatch::ProtoUpload::I8Compute {
            let compute_program = self.proto_repack_compute_program.ok_or_else(|| {
                crate::Error::InvalidShape(
                    "plan selected I8Compute without a compiled compute program".into(),
                )
            })?;
            // GLES 3.1 compute shader path: uploads the HWC data to an SSBO and
            // transposes it on the GPU. A non-contiguous proto array is copied
            // first.
            let mut data = match protos.as_slice() {
                Some(s) => std::borrow::Cow::Borrowed(s),
                None => std::borrow::Cow::Owned(protos.iter().copied().collect()),
            };
            // Pads a copy to a 4-byte boundary for SSBO int[] alignment.
            if data.len() % 4 != 0 {
                let mut owned = data.into_owned();
                owned.resize(owned.len().next_multiple_of(4), 0);
                data = std::borrow::Cow::Owned(owned);
            }
            let data_bytes = data.len();

            // Allocates the proto texture as R32I.
            if self.ensure_proto_texture(
                texture_target,
                edgefirst_gl::gl::R32I,
                width,
                height,
                num_protos,
            ) {
                Self::set_proto_tex_params(texture_target, edgefirst_gl::gl::NEAREST);
            }

            unsafe {
                // Upload HWC data to SSBO
                edgefirst_gl::gl::BindBuffer(
                    edgefirst_gl::gl::SHADER_STORAGE_BUFFER,
                    self.proto_ssbo,
                );
                if data_bytes > self.proto_ssbo_size {
                    edgefirst_gl::gl::BufferData(
                        edgefirst_gl::gl::SHADER_STORAGE_BUFFER,
                        data_bytes as isize,
                        data.as_ptr() as *const std::ffi::c_void,
                        edgefirst_gl::gl::STREAM_DRAW,
                    );
                    self.proto_ssbo_size = data_bytes;
                } else {
                    edgefirst_gl::gl::BufferSubData(
                        edgefirst_gl::gl::SHADER_STORAGE_BUFFER,
                        0,
                        data_bytes as isize,
                        data.as_ptr() as *const std::ffi::c_void,
                    );
                }
                edgefirst_gl::gl::BindBufferBase(
                    edgefirst_gl::gl::SHADER_STORAGE_BUFFER,
                    0,
                    self.proto_ssbo,
                );

                // Bind texture as image for compute write (R32I)
                edgefirst_gl::gl::BindImageTexture(
                    0,
                    self.proto_texture.id,
                    0,
                    edgefirst_gl::gl::TRUE,
                    0,
                    edgefirst_gl::gl::WRITE_ONLY,
                    edgefirst_gl::gl::R32I,
                );

                // Dispatch compute
                edgefirst_gl::gl::UseProgram(compute_program);
                let (loc_w, loc_h, loc_np) = self.proto_compute_locs;
                edgefirst_gl::gl::Uniform1i(loc_w, width as i32);
                edgefirst_gl::gl::Uniform1i(loc_h, height as i32);
                edgefirst_gl::gl::Uniform1i(loc_np, num_protos as i32);

                let groups_x = width.div_ceil(16) as u32;
                let groups_y = height.div_ceil(16) as u32;
                edgefirst_gl::gl::DispatchCompute(groups_x, groups_y, 1);
                edgefirst_gl::gl::MemoryBarrier(
                    edgefirst_gl::gl::TEXTURE_FETCH_BARRIER_BIT
                        | edgefirst_gl::gl::SHADER_IMAGE_ACCESS_BARRIER_BIT,
                );

                // Unbind SSBO and log any GL errors from compute dispatch
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::SHADER_STORAGE_BUFFER, 0);
                loop {
                    let err = edgefirst_gl::gl::GetError();
                    if err == edgefirst_gl::gl::NO_ERROR {
                        break;
                    }
                    log::debug!("GL error after compute dispatch: 0x{err:x}");
                }
            }
        } else {
            // GLES 3.0 path: CPU repack
            let tex_data = super::proto_dispatch::repack_layers(protos);

            self.upload_proto_texture(
                texture_target,
                edgefirst_gl::gl::R8I,
                width,
                height,
                num_protos,
                edgefirst_gl::gl::RED_INTEGER,
                edgefirst_gl::gl::BYTE,
                edgefirst_gl::gl::NEAREST,
                &tex_data,
            );
        }

        let (proto_scale, proto_zp) = quantization;
        let proto_scaled_zp = -(proto_zp as f32) * proto_scale;

        match plan.program {
            super::proto_dispatch::ProtoProgram::Int8Nearest
            | super::proto_dispatch::ProtoProgram::Int8Bilinear => {
                let program = if plan.program == super::proto_dispatch::ProtoProgram::Int8Nearest {
                    &self.proto_segmentation_int8_nearest_program
                } else {
                    &self.proto_segmentation_int8_bilinear_program
                };
                edgefirst_gl::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(
                    program,
                    detect,
                    mask_coefficients,
                    num_protos,
                    color_mode,
                )?;
            }
            super::proto_dispatch::ProtoProgram::Int8TwoPass => {
                self.render_proto_int8_two_pass(
                    detect,
                    mask_coefficients,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                    color_mode,
                )?;
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto program {other:?} is not an int8 program"
                )));
            }
        }

        Ok(())
    }

    /// Two-pass int8 path: dequant int8→RGBA16F FBO, then render with the
    /// f16 shader using GL_LINEAR.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_int8_two_pass(
        &mut self,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        quantization: (f32, i32),
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        let num_layers = num_protos.div_ceil(4);

        // Saves the caller's FBO, viewport and blend state.
        let (saved_fbo, saved_viewport, blend_was_enabled) = unsafe {
            let mut fbo: i32 = 0;
            edgefirst_gl::gl::GetIntegerv(edgefirst_gl::gl::FRAMEBUFFER_BINDING, &mut fbo);
            let mut vp = [0i32; 4];
            edgefirst_gl::gl::GetIntegerv(edgefirst_gl::gl::VIEWPORT, vp.as_mut_ptr());
            let blend =
                edgefirst_gl::gl::IsEnabled(edgefirst_gl::gl::BLEND) == edgefirst_gl::gl::TRUE;
            (fbo as u32, vp, blend)
        };

        // Pass 1: dequantize int8 → RGBA16F texture via the persistent dequant
        // FBO.
        if Self::ensure_immutable_tex_array(
            &mut self.proto_dequant_texture,
            &mut self.proto_dequant_tex_dims,
            texture_target,
            edgefirst_gl::gl::RGBA16F,
            width,
            height,
            num_layers,
        ) {
            Self::set_proto_tex_params(texture_target, edgefirst_gl::gl::LINEAR);
        }

        // Disables blending for the dequant render, which writes four
        // dequantized protos per RGBA16F texel.
        edgefirst_gl::disable(edgefirst_gl::gl::BLEND);

        let (proto_scale, proto_zp) = quantization;
        let proto_scaled_zp = -(proto_zp as f32) * proto_scale;

        let dequant_program = &self.proto_dequant_int8_program;
        edgefirst_gl::use_program(dequant_program.id);
        dequant_program.load_uniform_1f(c"proto_scale", proto_scale)?;
        dequant_program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;

        // Bind the int8 proto texture to TEXTURE0 for the dequant shader
        edgefirst_gl::active_texture(edgefirst_gl::gl::TEXTURE0);
        edgefirst_gl::bind_texture(texture_target, self.proto_texture.id);

        // Render each RGBA16F layer (4 protos per layer)
        for layer in 0..num_layers {
            self.proto_dequant_fbo.bind();
            unsafe {
                edgefirst_gl::gl::FramebufferTextureLayer(
                    edgefirst_gl::gl::FRAMEBUFFER,
                    edgefirst_gl::gl::COLOR_ATTACHMENT0,
                    self.proto_dequant_texture.id,
                    0,
                    layer as i32,
                );
                edgefirst_gl::gl::Viewport(0, 0, width as i32, height as i32);
            }
            dequant_program.load_uniform_1i(c"base_layer", (layer * 4) as i32)?;
            self.draw_fullscreen_quad()?;
        }

        // Restore the caller's FBO and viewport.
        unsafe {
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, saved_fbo);
            edgefirst_gl::gl::Viewport(
                saved_viewport[0],
                saved_viewport[1],
                saved_viewport[2],
                saved_viewport[3],
            );
        }
        // Restores the caller's blend state for pass 2.
        if blend_was_enabled {
            edgefirst_gl::enable(edgefirst_gl::gl::BLEND);
        }

        // Pass 2: render with the f16 shader reading from the dequant texture
        let program = &self.proto_segmentation_program;
        edgefirst_gl::use_program(program.id);
        edgefirst_gl::active_texture(edgefirst_gl::gl::TEXTURE0);
        edgefirst_gl::bind_texture(texture_target, self.proto_dequant_texture.id);
        program.load_uniform_1i(c"num_layers", num_layers as i32)?;
        self.render_proto_detection_quads(
            program,
            detect,
            mask_coefficients,
            num_protos,
            color_mode,
        )?;

        Ok(())
    }

    /// Render every layered-float proto plan (F32→R32F, F32→RGBA16F and
    /// F16→RGBA16F) as directed by the
    /// [`ProtoPlan`](super::proto_dispatch::ProtoPlan): the upload strategy
    /// picks the repack and texture format, `plan.program` the sampler
    /// program, `plan.count_uniform` the layer-count uniform.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_layers(
        &mut self,
        plan: super::proto_dispatch::ProtoPlan,
        detect: &[DetectBox],
        mask_coefficients: &[f32],
        data: ProtoLayersData<'_>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        use super::proto_dispatch::{CountUniform, ProtoProgram, ProtoUpload};

        let num_layers = num_protos.div_ceil(4);
        match (plan.upload, &data) {
            (ProtoUpload::F32R32f, ProtoLayersData::F32(v)) => {
                // Repack protos to layer-first layout: (num_protos, H, W)
                let tex_data = super::proto_dispatch::repack_layers(*v);
                self.upload_proto_texture(
                    texture_target,
                    edgefirst_gl::gl::R32F,
                    width,
                    height,
                    num_protos,
                    edgefirst_gl::gl::RED,
                    edgefirst_gl::gl::FLOAT,
                    edgefirst_gl::gl::LINEAR,
                    &tex_data,
                );
            }
            (ProtoUpload::F32ToRgba16f, ProtoLayersData::F32(v)) => {
                let (tex_data, _) =
                    super::proto_dispatch::repack_rgba_f16_layers(*v, half::f16::from_f32);
                self.upload_proto_texture(
                    texture_target,
                    edgefirst_gl::gl::RGBA16F,
                    width,
                    height,
                    num_layers,
                    edgefirst_gl::gl::RGBA,
                    edgefirst_gl::gl::HALF_FLOAT,
                    edgefirst_gl::gl::LINEAR,
                    &tex_data,
                );
            }
            (ProtoUpload::F16Rgba16f, ProtoLayersData::F16(v)) => {
                let (tex_data, _) = super::proto_dispatch::repack_rgba_f16_layers(*v, |x| x);
                self.upload_proto_texture(
                    texture_target,
                    edgefirst_gl::gl::RGBA16F,
                    width,
                    height,
                    num_layers,
                    edgefirst_gl::gl::RGBA,
                    edgefirst_gl::gl::HALF_FLOAT,
                    edgefirst_gl::gl::LINEAR,
                    &tex_data,
                );
            }
            (upload, _) => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto upload {upload:?} incompatible with the provided proto data"
                )));
            }
        }

        let program = match plan.program {
            ProtoProgram::F32 => &self.proto_segmentation_f32_program,
            ProtoProgram::F16 => &self.proto_segmentation_program,
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "proto program {other:?} is not a layered-float program"
                )));
            }
        };
        edgefirst_gl::use_program(program.id);
        match plan.count_uniform {
            CountUniform::NumProtos => program.load_uniform_1i(c"num_protos", num_protos as i32)?,
            CountUniform::NumLayers => program.load_uniform_1i(c"num_layers", num_layers as i32)?,
        }
        self.render_proto_detection_quads(
            program,
            detect,
            mask_coefficients,
            num_protos,
            color_mode,
        )
    }

    fn render_segmentation(
        &mut self,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        color_mode: crate::ColorMode,
    ) -> crate::Result<()> {
        if segmentation.is_empty() {
            return Ok(());
        }

        // Rejects a mask whose rank is not 3.
        if let Some(bad) = segmentation
            .iter()
            .find(|s| s.segmentation.shape().len() != 3)
        {
            return Err(crate::Error::InvalidShape(format!(
                "segmentation must be [H, W, C], got {:?}",
                bad.segmentation.shape()
            )));
        }

        let is_modelpack = segmentation[0].segmentation.shape()[2] > 1;
        // Maps a normalised [0, 1] coordinate to clip space [-1, 1].
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        if is_modelpack {
            let seg = &segmentation[0];
            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(seg.xmin),
                top: cvt_screen_coord(seg.ymax),
                right: cvt_screen_coord(seg.xmax),
                bottom: cvt_screen_coord(seg.ymin),
            };
            let segment = seg.segmentation.as_u8().ok_or(Error::Internal(
                "segmentation must be a U8 tensor".to_owned(),
            ))?;
            let segment_map = segment.map_read()?;
            let slice = segment_map.as_slice();

            self.render_modelpack_segmentation(
                dst_roi,
                slice,
                [
                    seg.segmentation.shape()[0],
                    seg.segmentation.shape()[1],
                    seg.segmentation.shape()[2],
                ],
            )?;
        } else {
            // Binds the program and texture and sets the texture parameters once
            // for all masks.
            self.setup_yolo_segmentation_pass();

            for (idx, (seg, det)) in segmentation.iter().zip(detect).enumerate() {
                let color_index = color_mode.index(idx, det.label);
                let dst_roi = RegionOfInterest {
                    left: cvt_screen_coord(seg.xmin),
                    top: cvt_screen_coord(seg.ymax),
                    right: cvt_screen_coord(seg.xmax),
                    bottom: cvt_screen_coord(seg.ymin),
                };

                let segment = seg.segmentation.as_u8().ok_or(Error::Internal(
                    "segmentation must be a U8 tensor".to_owned(),
                ))?;
                let segment_map = segment.map_read()?;
                let slice = segment_map.as_slice();

                self.render_yolo_segmentation(
                    dst_roi,
                    slice,
                    [seg.segmentation.shape()[0], seg.segmentation.shape()[1]],
                    color_index,
                )?;
            }
        }

        edgefirst_gl::disable(edgefirst_gl::gl::BLEND);
        Ok(())
    }

    /// Compile a GLES 3.1 compute shader program from source.
    fn compile_compute_program(source: &str) -> Result<u32, Error> {
        unsafe {
            let cs = edgefirst_gl::gl::CreateShader(edgefirst_gl::gl::COMPUTE_SHADER);
            if super::shaders::compile_shader_from_str(cs, source, "proto_repack_compute").is_err()
            {
                edgefirst_gl::gl::DeleteShader(cs);
                return Err(Error::OpenGl("compute shader compile failed".into()));
            }

            let program = edgefirst_gl::gl::CreateProgram();
            edgefirst_gl::gl::AttachShader(program, cs);
            edgefirst_gl::gl::LinkProgram(program);

            let mut linked: i32 = 0;
            edgefirst_gl::gl::GetProgramiv(program, edgefirst_gl::gl::LINK_STATUS, &mut linked);
            edgefirst_gl::gl::DeleteShader(cs);

            if linked == 0 {
                let mut log_len = 0;
                edgefirst_gl::gl::GetProgramiv(
                    program,
                    edgefirst_gl::gl::INFO_LOG_LENGTH,
                    &mut log_len,
                );
                let mut log_buf: Vec<u8> = vec![0; log_len as usize];
                edgefirst_gl::gl::GetProgramInfoLog(
                    program,
                    log_len,
                    std::ptr::null_mut(),
                    log_buf.as_mut_ptr() as *mut std::ffi::c_char,
                );
                let msg = String::from_utf8_lossy(&log_buf);
                edgefirst_gl::gl::DeleteProgram(program);
                return Err(Error::OpenGl(format!("compute program link failed: {msg}")));
            }
            Ok(program)
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Ensure `proto_texture` is an immutable-storage (`TexStorage3D`) array
    /// of exactly `(w, h, layers, internal_fmt)`, recreating the texture
    /// object on any change, and bind it on `TEXTURE0`. Returns `true` when
    /// the object was recreated, which resets its sampler parameters.
    fn ensure_proto_texture(
        &mut self,
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
    ) -> bool {
        Self::ensure_immutable_tex_array(
            &mut self.proto_texture,
            &mut self.proto_tex_dims,
            target,
            internal_fmt,
            w,
            h,
            layers,
        )
    }

    /// Bind `texture` on `TEXTURE0`, first recreating it with immutable
    /// storage of `(w, h, layers, internal_fmt)` when `tex_dims` differs.
    /// Returns `true` when the object was recreated.
    fn ensure_immutable_tex_array(
        texture: &mut Texture,
        tex_dims: &mut (usize, usize, usize, u32),
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
    ) -> bool {
        let dims = (w, h, layers, internal_fmt);
        edgefirst_gl::active_texture(edgefirst_gl::gl::TEXTURE0);
        if dims == *tex_dims {
            edgefirst_gl::bind_texture(target, texture.id);
            return false;
        }
        // Replaces the texture object; the previous one is deleted on drop.
        *texture = Texture::new();
        edgefirst_gl::bind_texture(target, texture.id);
        unsafe {
            edgefirst_gl::gl::TexStorage3D(
                target,
                1,
                internal_fmt,
                w as i32,
                h as i32,
                layers as i32,
            );
        }
        *tex_dims = dims;
        true
    }

    /// Apply the proto texture sampler params (`filter` + CLAMP_TO_EDGE).
    fn set_proto_tex_params(target: u32, filter: u32) {
        unsafe { super::core::set_tex_filter(target, filter) };
        edgefirst_gl::tex_parameteri(
            target,
            edgefirst_gl::gl::TEXTURE_WRAP_S,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );
        edgefirst_gl::tex_parameteri(
            target,
            edgefirst_gl::gl::TEXTURE_WRAP_T,
            edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_proto_texture<T: Copy>(
        &mut self,
        target: u32,
        internal_fmt: u32,
        w: usize,
        h: usize,
        layers: usize,
        format: u32,
        dtype: u32,
        filter: u32,
        data: &[T],
    ) {
        if self.ensure_proto_texture(target, internal_fmt, w, h, layers) {
            Self::set_proto_tex_params(target, filter);
        }
        unsafe {
            edgefirst_gl::gl::TexSubImage3D(
                target,
                0,
                0,
                0,
                0,
                w as i32,
                h as i32,
                layers as i32,
                format,
                dtype,
                data.as_ptr() as *const std::ffi::c_void,
            );
        }
    }

    /// Set the `opacity` uniform on all segmentation and color shader programs.
    /// Does nothing when `opacity` equals the cached value.
    fn set_opacity_uniform(&mut self, opacity: f32) -> Result<(), Error> {
        if (opacity - self.cached_opacity).abs() < f32::EPSILON {
            return Ok(());
        }
        for program in [
            &self.color_program,
            &self.segmentation_program,
            &self.instanced_segmentation_program,
            &self.proto_segmentation_program,
            &self.proto_segmentation_int8_nearest_program,
            &self.proto_segmentation_int8_bilinear_program,
            &self.proto_segmentation_f32_program,
        ] {
            program.load_uniform_1f(c"opacity", opacity)?;
        }
        self.cached_opacity = opacity;
        Ok(())
    }

    fn render_box(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        detect: &[DetectBox],
        color_mode: crate::ColorMode,
    ) -> Result<(), Error> {
        unsafe {
            edgefirst_gl::gl::UseProgram(self.color_program.id);
            let rescale = |x: f32| x * 2.0 - 1.0;
            let thickness = self.box_line_width;
            for (idx, d) in detect.iter().enumerate() {
                let color_index = color_mode.index(idx, d.label);
                self.color_program
                    .load_uniform_1i(c"class_index", color_index as i32)?;
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                edgefirst_gl::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let bbox: [f32; 4] = d.bbox.into();
                let outer_box = [
                    bbox[0] - thickness / dst_w as f32,
                    bbox[1] - thickness / dst_h as f32,
                    bbox[2] + thickness / dst_w as f32,
                    bbox[3] + thickness / dst_h as f32,
                ];
                let camera_vertices: [f32; 24] = [
                    rescale(bbox[0]),
                    rescale(bbox[3]),
                    0., // bottom left
                    rescale(bbox[2]),
                    rescale(bbox[3]),
                    0., // bottom right
                    rescale(bbox[2]),
                    rescale(bbox[1]),
                    0., // top right
                    rescale(bbox[0]),
                    rescale(bbox[1]),
                    0., // top left
                    rescale(outer_box[0]),
                    rescale(outer_box[3]),
                    0., // bottom left
                    rescale(outer_box[2]),
                    rescale(outer_box[3]),
                    0., // bottom right
                    rescale(outer_box[2]),
                    rescale(outer_box[1]),
                    0., // top right
                    rescale(outer_box[0]),
                    rescale(outer_box[1]),
                    0., // top left
                ];
                edgefirst_gl::gl::BufferData(
                    edgefirst_gl::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    edgefirst_gl::gl::DYNAMIC_DRAW,
                );

                let vertices_index: [u32; 10] = [0, 1, 5, 2, 6, 3, 7, 0, 4, 5];
                edgefirst_gl::gl::DrawElements(
                    edgefirst_gl::gl::TRIANGLE_STRIP,
                    vertices_index.len() as i32,
                    edgefirst_gl::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Whether the GPU is a Verisilicon/Vivante core (GL_RENDERER).
    pub(super) fn is_vivante(&self) -> bool {
        self.is_vivante
    }

    /// Whether this GPU renders a resized RGBA source into an NV12 destination.
    #[cfg(test)]
    pub(super) fn renders_nv12_dst(&self) -> bool {
        self.nv12_policy_decline(PixelFormat::Rgba, false, false, (640, 360))
            .is_none()
    }

    /// The NV12 destination policy and whether soft declines apply.
    #[cfg(test)]
    pub(super) fn nv12_decline_state(&self) -> (Nv12DstPolicy, bool) {
        (self.nv12_policy, self.soft_declines)
    }

    /// The float render support this processor advertises.
    pub(super) fn supported_render_dtypes(&self) -> crate::RenderDtypeSupport {
        float_render_support(
            self.is_vivante,
            self.float_import_refused(),
            self.supports_f32_color,
            self.supports_f16_color,
        )
    }

    /// Convert an NV12/NV16/NV24 or packed Rgb DMA-BUF source to a zero-copy
    /// float destination in two GPU passes. Pass 1 renders the source with
    /// the caller's geometry (resize, crop, letterbox) into the shared
    /// intermediate RGBA8 texture at destination resolution; pass 2 packs
    /// that texture 1:1 into the destination `path` names.
    #[allow(clippy::too_many_arguments)]
    fn convert_nv_to_float_two_pass(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        path: FloatRenderPath,
        rotation: crate::Rotation,
        flip: Flip,
        crop: ResolvedCrop,
        convert_span: &tracing::Span,
    ) -> crate::Result<()> {
        if rotation != crate::Rotation::None || flip != Flip::None {
            return Err(crate::Error::NotSupported(
                "GL fused NV→float: rotation/flip not supported; using CPU fallback".to_string(),
            ));
        }
        let src_u8 = src.as_u8().ok_or(Error::NotAnImage)?;
        let src_fmt = src.format().ok_or(Error::NotAnImage)?;
        let dst_w = dst.width().ok_or(Error::NotAnImage)?;
        let dst_h = dst.height().ok_or(Error::NotAnImage)?;
        let _span =
            tracing::trace_span!("image.convert.gl.nv_to_planar_float", dst_w, dst_h).entered();

        // Pass 1: NV → RGBA with the caller's geometry, into the shared
        // intermediate texture.
        self.render_to_intermediate(
            src_u8,
            src_fmt,
            (dst_w, dst_h),
            PixelFormat::Rgba,
            rotation,
            flip,
            crop,
            convert_span,
        )?;

        // Pass 2: 1:1 pack of the intermediate into the zero-copy float
        // destination. Samples the letterbox rectangle from the same rectangle
        // of the intermediate and writes the pad outside it.
        let (mut src_rect_uv, mut dst_rect_px, mut pad_color) =
            super::core::float_crop_uniforms(&ResolvedCrop::no_crop(), dst_w, dst_h, dst_w, dst_h)?;
        // The pixel pad in sampler units.
        if let Some(c) = crop.dst_color {
            pad_color = c.map(|v| v as f32 / 255.0);
        }
        if let Some(r) = crop.dst_rect {
            dst_rect_px = [r.left as f32, r.top as f32, r.width as f32, r.height as f32];
            src_rect_uv = [
                r.left as f32 / dst_w as f32,
                r.top as f32 / dst_h as f32,
                r.width as f32 / dst_w as f32,
                r.height as f32 / dst_h as f32,
            ];
        }
        // The sampling extent is the whole intermediate texture.
        let src_extent =
            super::render::sample_clamp_rect((dst_w, dst_h), super::render::ImportMap::WHOLE);
        self.render_float_to_zero_copy_tail(
            self.packed_rgb_intermediate_tex.id,
            src_rect_uv,
            src_extent,
            dst_rect_px,
            pad_color,
            dst,
            path,
            convert_span,
        )
    }

    /// Release per-pass platform texture attachments unless a deferred batch
    /// has a flush pending.
    pub(super) fn end_gpu_pass_if_synced(&self) {
        if !self.pending_flush {
            Platform::end_gpu_pass(&self.gl_context);
        }
    }

    /// Platform hook run by the dispatch wrapper before every message; see
    /// [`super::platform::GlPlatform::begin_gpu_pass`].
    pub(super) fn begin_gpu_pass(&self) {
        Platform::begin_gpu_pass(&self.gl_context);
    }

    /// Whether this driver's DMA-BUF import reads from the wrong place for a
    /// plane offset that is not [`DMA_IMPORT_OFFSET_ALIGN`]-byte aligned: Mali
    /// and Adreno.
    pub(super) fn misplaces_unaligned_offsets(&self) -> bool {
        self.is_mali || self.is_adreno
    }

    /// Whether this driver refuses every float DMA-BUF destination import:
    /// Adreno on a zero-copy transfer backend.
    fn float_import_refused(&self) -> bool {
        self.is_adreno && self.gl_context.transfer_backend.is_zero_copy()
    }

    /// The `glTexImage2D` internal format for a GL-owned render target whose
    /// client format is `format`: `GL_R8` for `GL_RED` on Adreno, `format`
    /// otherwise.
    fn render_internal_format(&self, format: u32) -> i32 {
        if self.is_adreno && format == edgefirst_gl::gl::RED {
            edgefirst_gl::gl::R8 as i32
        } else {
            format as i32
        }
    }

    /// Assemble the capability surface for this processor.
    pub(super) fn platform_caps(&self) -> super::platform::PlatformCaps {
        super::platform::PlatformCaps {
            transfer_backend: self.gl_context.transfer_backend,
            render_dtypes: self.supported_render_dtypes(),
            serialize_gl: requires_full_serialization(RendererTraits {
                vivante: self.is_vivante(),
                mali: self.is_mali,
                software: false,
                virtual_gpu: self.is_virtual_gpu,
                angle: self.is_angle,
                adreno: self.is_adreno,
            }),
            external_oes: Platform::EXTERNAL_OES,
            native_fence_sync: Platform::native_fence_sync(&self.gl_context),
            has_bgra: self.has_bgra,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    #[test]
    fn nv12_dst_policy_per_renderer() {
        use super::{
            Nv12DstPolicy, RendererTraits, NV12_SAME_SIZE_MIN_PIXELS_ADRENO,
            NV12_SAME_SIZE_MIN_PIXELS_ANGLE,
        };
        let policy = |gpu_renders,
                      packed_yuv_src_broken,
                      same_size_min_pixels,
                      exact_copy_on_cpu,
                      measured| Nv12DstPolicy {
            gpu_renders,
            packed_yuv_src_broken,
            same_size_min_pixels,
            exact_copy_on_cpu,
            measured,
        };
        let traits = RendererTraits::default();
        let rows = [
            (
                "adreno",
                RendererTraits {
                    adreno: true,
                    ..traits
                },
                policy(true, false, NV12_SAME_SIZE_MIN_PIXELS_ADRENO, true, true),
            ),
            (
                "mali",
                RendererTraits {
                    mali: true,
                    ..traits
                },
                policy(true, false, 0, false, true),
            ),
            (
                "angle",
                RendererTraits {
                    angle: true,
                    ..traits
                },
                policy(true, false, NV12_SAME_SIZE_MIN_PIXELS_ANGLE, true, true),
            ),
            (
                "vivante",
                RendererTraits {
                    vivante: true,
                    ..traits
                },
                policy(true, true, usize::MAX, true, true),
            ),
            (
                "software",
                RendererTraits {
                    software: true,
                    ..traits
                },
                policy(false, false, usize::MAX, true, true),
            ),
            (
                "angle over a software rasterizer",
                RendererTraits {
                    angle: true,
                    software: true,
                    ..traits
                },
                policy(false, false, usize::MAX, true, true),
            ),
            (
                "other",
                traits,
                policy(true, false, usize::MAX, true, false),
            ),
        ];
        for (name, t, want) in rows {
            assert_eq!(Nv12DstPolicy::for_renderer(t), want, "{name}");
        }
        assert_eq!(NV12_SAME_SIZE_MIN_PIXELS_ADRENO, 1920 * 1080);
        assert_eq!(NV12_SAME_SIZE_MIN_PIXELS_ANGLE, 1600 * 900);
    }

    #[test]
    fn nv12_dst_decline_classes() {
        use super::{nv12_dst_decline, DeclineClass, GlPlatform, Nv12DstPolicy, Platform};
        use crate::PixelFormat::{Nv12, Rgba, Vyuy, Yuyv};
        use DeclineClass::{Hard, Soft};
        let renders = Nv12DstPolicy {
            gpu_renders: true,
            packed_yuv_src_broken: false,
            same_size_min_pixels: 0,
            exact_copy_on_cpu: false,
            measured: true,
        };
        let class = |policy, fmt, same_size, exact_copy, size| {
            nv12_dst_decline(policy, "GPU-X", fmt, same_size, exact_copy, size).map(|(c, _)| c)
        };
        let vga = (640, 480);

        let no_gpu = Nv12DstPolicy {
            gpu_renders: false,
            ..renders
        };
        assert_eq!(class(no_gpu, Rgba, false, false, vga), Some(Hard));
        let (_, msg) = nv12_dst_decline(no_gpu, "GPU-X", Rgba, false, false, vga).unwrap();
        assert!(msg.contains("GPU-X"), "{msg}");
        if !Platform::NV12_DST {
            assert_eq!(class(renders, Rgba, false, false, vga), Some(Hard));
            return;
        }
        assert_eq!(class(renders, Rgba, false, false, vga), None);

        let broken = Nv12DstPolicy {
            packed_yuv_src_broken: true,
            measured: false,
            ..renders
        };
        assert_eq!(class(broken, Yuyv, false, false, vga), Some(Hard));
        assert_eq!(class(broken, Vyuy, false, false, vga), Some(Hard));
        assert_eq!(class(broken, Nv12, false, false, vga), Some(Soft));
        let broken = Nv12DstPolicy {
            measured: true,
            ..broken
        };
        assert_eq!(class(broken, Nv12, false, false, vga), None);
        assert_eq!(class(broken, Rgba, false, false, vga), None);

        let unmeasured = Nv12DstPolicy {
            measured: false,
            ..renders
        };
        assert_eq!(class(unmeasured, Rgba, false, false, vga), Some(Soft));
        assert_eq!(
            class(unmeasured, Yuyv, false, false, (1920, 1080)),
            Some(Soft)
        );

        let copy_on_cpu = Nv12DstPolicy {
            exact_copy_on_cpu: true,
            ..renders
        };
        assert_eq!(
            class(copy_on_cpu, Nv12, true, true, (3840, 2160)),
            Some(Soft)
        );
        assert_eq!(class(copy_on_cpu, Nv12, true, false, (3840, 2160)), None);
        assert_eq!(class(renders, Nv12, true, true, (3840, 2160)), None);
        let (_, msg) = nv12_dst_decline(copy_on_cpu, "GPU-X", Yuyv, true, true, vga).unwrap();
        assert!(msg.contains("exact same-size"), "{msg}");

        let threshold = Nv12DstPolicy {
            same_size_min_pixels: 1920 * 1080,
            ..renders
        };
        assert_eq!(class(threshold, Rgba, true, false, (1600, 900)), Some(Soft));
        assert_eq!(class(threshold, Rgba, true, false, (1920, 1080)), None);
        assert_eq!(class(threshold, Rgba, false, false, (320, 240)), None);
        let (_, msg) =
            nv12_dst_decline(threshold, "GPU-X", Rgba, true, false, (1600, 900)).unwrap();
        assert!(msg.contains("1600x900"), "{msg}");
        let every_same_size = Nv12DstPolicy {
            same_size_min_pixels: usize::MAX,
            ..renders
        };
        assert_eq!(
            class(every_same_size, Rgba, true, false, (3840, 2160)),
            Some(Soft)
        );

        for (_, msg) in [
            nv12_dst_decline(no_gpu, "GPU-X", Rgba, false, false, vga),
            nv12_dst_decline(broken, "GPU-X", Yuyv, false, false, vga),
            nv12_dst_decline(unmeasured, "GPU-X", Rgba, false, false, vga),
            nv12_dst_decline(copy_on_cpu, "GPU-X", Nv12, true, true, vga),
            nv12_dst_decline(threshold, "GPU-X", Rgba, true, false, vga),
        ]
        .into_iter()
        .flatten()
        {
            assert!(!msg.contains("runs on the CPU"), "{msg}");
        }
    }

    use super::{
        pack_row_length, padded_readback_bytes, plan_pbo_readback, resolve_egl_cache_capacity,
        should_reject_software_gl, spread_rows, unpack_row_length, Error,
        DEFAULT_EGL_CACHE_CAPACITY,
    };

    /// The RGBA->NV12 luma, chroma and chroma fill programs compile.
    #[test]
    fn nv12_encode_programs_compile() {
        use super::super::resources::GlProgram;
        use super::super::shaders::generate_vertex_shader;
        use super::super::shaders_common::{
            NV12_CHROMA_FILL_FRAGMENT, NV12_CHROMA_R8_FRAGMENT, NV12_LUMA_FRAGMENT,
        };
        let _gl = match super::GLProcessorST::new(None, None) {
            Ok(g) => g,
            Err(e) => {
                crate::test_support::report_skip(&format!(
                    "nv12_encode_programs_compile - GL not available: {e}"
                ));
                return;
            }
        };
        for fragment in [
            NV12_LUMA_FRAGMENT,
            NV12_CHROMA_R8_FRAGMENT,
            NV12_CHROMA_FILL_FRAGMENT,
        ] {
            GlProgram::new(generate_vertex_shader(), fragment).unwrap();
        }
    }

    /// An NV12 destination is renderable only as a zero-copy DMA buffer on a
    /// platform that imports NV12 destinations.
    #[test]
    fn nv12_destination_renders_only_zero_copy_where_the_platform_imports_it() {
        use super::super::TransferBackend;
        use edgefirst_tensor::{PixelFormat, TensorMemory};
        let renderable = super::GLProcessorST::dst_format_renderable;
        assert!(renderable(
            TransferBackend::DmaBuf,
            TensorMemory::DmaBuf,
            PixelFormat::Nv12,
            false,
            true
        ));
        assert!(!renderable(
            TransferBackend::DmaBuf,
            TensorMemory::DmaBuf,
            PixelFormat::Nv12,
            false,
            false
        ));
        for memory in [TensorMemory::Mem, TensorMemory::Shm, TensorMemory::Pbo] {
            assert!(
                !renderable(
                    TransferBackend::DmaBuf,
                    memory,
                    PixelFormat::Nv12,
                    false,
                    true
                ),
                "{memory:?}"
            );
        }
        for backend in [TransferBackend::Pbo, TransferBackend::Sync] {
            assert!(
                !renderable(
                    backend,
                    TensorMemory::DmaBuf,
                    PixelFormat::Nv12,
                    false,
                    true
                ),
                "{backend:?}"
            );
        }
        assert!(renderable(
            TransferBackend::DmaBuf,
            TensorMemory::DmaBuf,
            PixelFormat::Rgba,
            false,
            false
        ));
    }

    /// Checks that `y2y_pays_off` holds below ten source pixels per output pixel.
    #[test]
    fn y2y_pays_off_below_ten_source_pixels_per_output_pixel() {
        let out = 640 * 640;
        assert!(super::y2y_pays_off(1280 * 720, out));
        assert!(super::y2y_pays_off(1920 * 1080, out));
        assert!(!super::y2y_pays_off(3840 * 2160, out));
        assert!(super::y2y_pays_off(3840 * 2160, 1280 * 1280));
    }

    /// Checks `NvSampling::choose` over every combination of its inputs.
    #[test]
    fn nv_sampling_choice_covers_every_binding() {
        use super::{NvSampling, NvSamplingInputs};
        for bits in 0u8..32 {
            let inputs = NvSamplingInputs {
                bilinear: bits & 1 != 0,
                hw_filter: bits & 2 != 0,
                imported: bits & 4 != 0,
                chroma_view: bits & 8 != 0,
                y2y: bits & 16 != 0,
            };
            let expected = if !inputs.bilinear {
                NvSampling::Single
            } else if !inputs.hw_filter {
                NvSampling::Shader
            } else if !inputs.imported || inputs.chroma_view {
                NvSampling::Hardware
            } else if inputs.y2y {
                NvSampling::HardwareYuv
            } else {
                NvSampling::Shader
            };
            assert_eq!(NvSampling::choose(inputs), expected, "{inputs:?}");
        }
    }

    /// Checks that only an import without a chroma view selects `HardwareYuv`.
    #[test]
    fn nv_sampling_takes_y2y_only_for_an_import_without_a_chroma_view() {
        use super::{NvSampling, NvSamplingInputs};
        let mali = NvSamplingInputs {
            bilinear: true,
            hw_filter: true,
            imported: true,
            chroma_view: false,
            y2y: true,
        };
        assert_eq!(NvSampling::choose(mali), NvSampling::HardwareYuv);
        let chroma_view = NvSamplingInputs {
            chroma_view: true,
            ..mali
        };
        assert_eq!(NvSampling::choose(chroma_view), NvSampling::Hardware);
        let opted_out = NvSamplingInputs {
            hw_filter: false,
            ..mali
        };
        assert_eq!(NvSampling::choose(opted_out), NvSampling::Shader);
        let nearest = NvSamplingInputs {
            bilinear: false,
            ..mali
        };
        assert_eq!(NvSampling::choose(nearest), NvSampling::Single);
    }

    /// Checks that the ROI is the crop's edges in texture coordinates, flipped vertically.
    #[test]
    fn nv_sample_rect_roi_maps_the_crop_edges() {
        let rect = super::NvSampleRect::new(
            Some(crate::Rect::new(160, 90, 960, 540)),
            None,
            (1280, 720),
            (640, 640),
            crate::Rotation::None,
        );
        let roi = rect.roi(1280, 720);
        assert_eq!(roi.left, 0.125);
        assert_eq!(roi.right, 0.875);
        assert_eq!(roi.top, 0.875);
        assert_eq!(roi.bottom, 0.125);

        let whole =
            super::NvSampleRect::new(None, None, (1280, 720), (640, 640), crate::Rotation::None)
                .roi(1280, 720);
        assert_eq!(whole.uv_bounds(), [0.0, 0.0, 1.0, 1.0]);
    }

    /// Checks that `nv_colorimetry` fills a missing colorimetry axis from the height.
    #[test]
    fn nv_colorimetry_resolves_missing_axes_by_height() {
        use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry};
        assert_eq!(
            super::nv_colorimetry(None, Some(720)),
            (ColorEncoding::Bt709, ColorRange::Limited)
        );
        assert_eq!(
            super::nv_colorimetry(None, Some(480)),
            (ColorEncoding::Bt601, ColorRange::Limited)
        );
        let full_601 = Colorimetry {
            encoding: Some(ColorEncoding::Bt601),
            range: Some(ColorRange::Full),
            ..Default::default()
        };
        assert_eq!(
            super::nv_colorimetry(Some(full_601), Some(1080)),
            (ColorEncoding::Bt601, ColorRange::Full)
        );
    }

    /// Checks that `spread_rows` leaves a tight destination unchanged.
    #[test]
    fn spread_rows_leaves_a_tight_destination_alone() {
        let mut buf: Vec<u8> = (0..12).collect();
        let before = buf.clone();
        spread_rows(&mut buf, 4, 3, 3);
        assert_eq!(buf, before, "a tight pitch is a no-op");
        spread_rows(&mut buf, 4, 3, 2);
        assert_eq!(buf, before, "a pitch below the row is a no-op");
    }

    /// Checks that `spread_rows` moves nothing for zero rows or one row.
    #[test]
    fn spread_rows_handles_zero_and_one_row() {
        let mut empty: Vec<u8> = Vec::new();
        spread_rows(&mut empty, 0, 3, 8);
        assert!(empty.is_empty());

        let mut one = vec![1u8, 2, 3, 0, 0, 0, 0, 0];
        let before = one.clone();
        spread_rows(&mut one, 1, 3, 8);
        assert_eq!(one, before, "row 0 already starts at offset 0");
    }

    /// Checks that `spread_rows` places each row at its pitch offset.
    #[test]
    fn spread_rows_places_each_row_at_the_pitch() {
        let (rows, tight, stride) = (4usize, 3usize, 5usize);
        // Fills each tightly packed row with one distinct byte.
        let mut buf = vec![0u8; (rows - 1) * stride + tight];
        for y in 0..rows {
            buf[y * tight..y * tight + tight].fill(b'a' + y as u8);
        }
        spread_rows(&mut buf, rows, tight, stride);
        for y in 0..rows {
            assert_eq!(
                &buf[y * stride..y * stride + tight],
                &[b'a' + y as u8; 3],
                "row {y} did not land at offset {}",
                y * stride
            );
        }
    }

    #[test]
    fn padded_readback_bytes_reports_only_a_padded_span() {
        // Tight and shorter-than-tight pitches.
        assert_eq!(padded_readback_bytes(4, 3, 3), None);
        assert_eq!(padded_readback_bytes(4, 3, 2), None);
        // No rows.
        assert_eq!(padded_readback_bytes(0, 3, 8), None);
        // Padded: a whole pitch per row but the last, plus the last row's bytes.
        assert_eq!(padded_readback_bytes(4, 3, 5), Some(3 * 5 + 3));
        assert_eq!(padded_readback_bytes(1, 3, 5), Some(3));
    }

    /// Checks that `pack_row_length` is `Some` only for a padded pitch of whole pixels.
    #[test]
    fn pack_row_length_covers_only_a_padded_whole_pixel_pitch() {
        // Tight and shorter-than-tight pitches.
        assert_eq!(pack_row_length(64, 64, 4), None);
        assert_eq!(pack_row_length(64, 32, 4), None);
        // RGBA at a 128-byte pitch: 32 pixels per row.
        assert_eq!(pack_row_length(64, 128, 4), Some(32));
        // RGB at the same pitch: 128 is not a multiple of 3.
        assert_eq!(pack_row_length(48, 128, 3), None);
        // A single-byte pixel divides every pitch.
        assert_eq!(pack_row_length(16, 17, 1), Some(17));
        assert_eq!(pack_row_length(16, 17, 0), None);
    }

    /// Checks that `unpack_row_length` refuses a pitch that is not a whole number of pixels.
    #[test]
    fn unpack_row_length_refuses_a_pitch_that_is_not_whole_pixels() {
        // No recorded pitch.
        assert!(matches!(unpack_row_length(None, 64, 4, "Rgba"), Ok(0)));
        // A tight pitch.
        assert!(matches!(unpack_row_length(Some(256), 64, 4, "Rgba"), Ok(0)));
        // RGBA at a 320-byte pitch: 80 pixels per row.
        assert!(matches!(
            unpack_row_length(Some(320), 64, 4, "Rgba"),
            Ok(80)
        ));
        // A single-byte pixel divides every pitch.
        assert!(matches!(unpack_row_length(Some(17), 16, 1, "Grey"), Ok(17)));
        // RGB at a 1024-byte pitch: 3 does not divide 1024.
        let Err(e) = unpack_row_length(Some(1024), 341, 3, "Rgb") else {
            panic!("a pitch that is not a whole number of pixels must be refused");
        };
        assert!(
            matches!(e, Error::NotSupported(ref m) if m.contains("1024") && m.contains("Rgb")),
            "unexpected error: {e:?}"
        );
        // An RGBA pitch that 4 does not divide.
        assert!(matches!(
            unpack_row_length(Some(1026), 256, 4, "RGBA"),
            Err(Error::NotSupported(_))
        ));
        // A zero pixel size.
        assert!(matches!(
            unpack_row_length(Some(16), 16, 0, "Rgba"),
            Err(Error::NotSupported(_))
        ));
    }

    /// Checks the readbacks `plan_pbo_readback` accepts and the ones it refuses.
    #[test]
    fn plan_pbo_readback_refuses_what_it_cannot_place() {
        // RGBA8 64 wide: 256 B tight.
        let tight = plan_pbo_readback(256, false, 0, 256 * 8, 64, 8, 4).expect("tight");
        assert_eq!((tight.stride, tight.tight_row), (256, 256));
        assert_eq!(tight.needed, 256 * 8, "a tight read fills every row");
        assert_eq!(tight.row_length, None, "GL already lands tight rows");

        // A whole-texel pitch is expressed in the pack state.
        let padded = plan_pbo_readback(320, false, 0, 320 * 8, 64, 8, 4).expect("padded");
        assert_eq!(padded.row_length, Some(80));
        assert_eq!(padded.needed, 320 * 7 + 256);

        // A pitch the texel size does not divide: read tight, spread after.
        let odd = plan_pbo_readback(257, false, 0, 257 * 8, 64, 8, 4).expect("odd pitch");
        assert_eq!(odd.row_length, None);
        assert_eq!(odd.needed, 257 * 7 + 256);

        // F16 NCHW: 8-byte texels with a 4-byte pad.
        assert_eq!(
            plan_pbo_readback(132, false, 0, 1 << 16, 16, 12, 8)
                .expect("f16 plane row")
                .row_length,
            None
        );
        assert_eq!(
            plan_pbo_readback(136, false, 0, 1 << 16, 16, 12, 8)
                .expect("f16 plane row")
                .row_length,
            Some(17)
        );

        // A pitch under one row.
        assert!(plan_pbo_readback(255, false, 0, 1 << 16, 64, 8, 4).is_err());
        // A view, and a plane offset.
        assert!(plan_pbo_readback(256, true, 0, 1 << 16, 64, 8, 4).is_err());
        assert!(plan_pbo_readback(256, false, 64, 1 << 16, 64, 8, 4).is_err());
        // A PBO one byte short of the span the rows occupy.
        assert!(plan_pbo_readback(320, false, 0, 320 * 7 + 255, 64, 8, 4).is_err());
    }

    #[test]
    fn cache_capacity_config_beats_environment() {
        assert_eq!(resolve_egl_cache_capacity(Some(4), Some("64")), 4);
    }

    #[test]
    fn cache_capacity_falls_back_through_env_to_default() {
        assert_eq!(resolve_egl_cache_capacity(None, Some("64")), 64);
        assert_eq!(
            resolve_egl_cache_capacity(None, None),
            DEFAULT_EGL_CACHE_CAPACITY
        );
    }

    #[test]
    fn cache_capacity_rejects_zero_and_garbage_at_both_levels() {
        assert_eq!(resolve_egl_cache_capacity(Some(0), Some("32")), 32);
        assert_eq!(
            resolve_egl_cache_capacity(Some(0), Some("0")),
            DEFAULT_EGL_CACHE_CAPACITY
        );
        assert_eq!(
            resolve_egl_cache_capacity(None, Some("plenty")),
            DEFAULT_EGL_CACHE_CAPACITY
        );
        assert_eq!(
            resolve_egl_cache_capacity(None, Some("-1")),
            DEFAULT_EGL_CACHE_CAPACITY
        );
    }

    #[test]
    fn software_gl_rejected_by_default() {
        assert!(should_reject_software_gl(true, false));
    }

    #[test]
    fn software_gl_allowed_with_override() {
        assert!(!should_reject_software_gl(true, true));
    }

    #[test]
    fn hardware_gl_never_rejected() {
        assert!(!should_reject_software_gl(false, false));
        assert!(!should_reject_software_gl(false, true));
    }

    // Checks the traits `classify_renderer` reports for known GL_RENDERER strings.
    #[test]
    fn classify_renderer_fleet_strings() {
        use super::{classify_renderer, RendererTraits};
        let real_gpu = RendererTraits::default();

        let t = classify_renderer("Vivante GC7000UL");
        assert!(t.vivante && !t.software && !t.virtual_gpu && !t.angle && !t.mali);
        let t = classify_renderer(
            "ANGLE (Apple, ANGLE Metal Renderer: Apple M2 Max, \
             Version 26.4.1 (Build 25E253))",
        );
        assert!(t.angle && !t.virtual_gpu && !t.software && !t.vivante);
        assert_eq!(
            classify_renderer("Mali-G310"),
            RendererTraits {
                mali: true,
                ..RendererTraits::default()
            }
        );
        assert_eq!(
            classify_renderer("Mali-G52"),
            RendererTraits {
                mali: true,
                ..RendererTraits::default()
            }
        );
        assert_eq!(classify_renderer("V3D 7.1"), real_gpu);
        assert_eq!(
            classify_renderer("Adreno (TM) 663"),
            RendererTraits {
                adreno: true,
                ..RendererTraits::default()
            }
        );
        assert_eq!(
            classify_renderer("NVIDIA Tegra Orin (nvgpu)/integrated"),
            real_gpu
        );
        let t = classify_renderer("llvmpipe (LLVM 15.0.7, 256 bits)");
        assert!(t.software && !t.vivante && !t.virtual_gpu && !t.angle);
        // ANGLE over Direct3D 11 on a hardware GPU.
        let t = classify_renderer(
            "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, \
             D3D11-32.0.16.1656)",
        );
        assert!(t.angle && !t.software && !t.virtual_gpu && !t.vivante);
        // ANGLE over D3D11 WARP.
        let t = classify_renderer(
            "ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11 vs_5_0 ps_5_0, \
             D3D11-10.0.26100.1)",
        );
        assert!(t.angle && t.software && !t.virtual_gpu && !t.vivante);
        // Virtualized Metal.
        let t = classify_renderer(
            "ANGLE (Apple, ANGLE Metal Renderer: Apple Paravirtual device, \
             Version 15.7.7 (Build 24G720))",
        );
        assert!(t.virtual_gpu && t.angle && !t.software && !t.vivante);
    }

    // Checks that Mali declines a source plane offset only when it is unaligned.
    #[test]
    fn mali_declines_only_unaligned_source_offsets() {
        use super::rejects_unaligned_import_offset as rejects;

        // Mali, unaligned.
        assert!(rejects(true, 32));
        assert!(rejects(true, 2080));
        // Mali, aligned.
        assert!(!rejects(true, 0));
        assert!(!rejects(true, 64));
        assert!(!rejects(true, 2048));
        // Other drivers.
        assert!(!rejects(false, 32));
        assert!(!rejects(false, 2080));
        assert!(!rejects(true, 256));
    }

    // Checks which source plane offsets resolve to an aligned import base.
    #[cfg(target_os = "linux")]
    #[test]
    fn mali_declines_the_imports_the_fold_could_not_move() {
        use super::super::dma_import::resolve_source_plane0;
        use super::{PixelFormat, DMA_IMPORT_OFFSET_ALIGN};

        // Whether the resolved plane-0 offset is aligned.
        let aligned = |fmt, w, pitch, off| {
            resolve_source_plane0(fmt, w, pitch, off)
                .0
                .is_multiple_of(DMA_IMPORT_OFFSET_ALIGN)
        };

        // A packed view rebased onto an aligned base.
        assert!(aligned(PixelFormat::Rgba, 16, 256, 2080));
        assert!(aligned(PixelFormat::Bgra, 16, 256, 32));
        // An aligned offset.
        assert!(aligned(PixelFormat::Rgba, 64, 256, 2048));
        assert!(aligned(PixelFormat::Nv12, 64, 64, 0));
        // Packed sources that are not rebased.
        assert!(!aligned(PixelFormat::Rgba, 64, 256, 32));
        assert!(!aligned(PixelFormat::Rgb, 16, 320, 323));
        // Formats that are not rebased.
        for fmt in [
            PixelFormat::Nv12,
            PixelFormat::Nv16,
            PixelFormat::Nv24,
            PixelFormat::Yuyv,
            PixelFormat::Vyuy,
            PixelFormat::PlanarRgb,
        ] {
            assert!(!aligned(fmt, 64, 256, 32), "{fmt:?} must not be exempt");
        }
    }

    // Checks `rejects_unaligned_import_offset` for gated and ungated drivers.
    #[test]
    fn the_offset_rule_is_a_source_rule_and_only_malis() {
        use super::rejects_unaligned_import_offset;

        assert!(
            rejects_unaligned_import_offset(true, 2080),
            "the Mali source rule covers 2080"
        );
        assert!(rejects_unaligned_import_offset(true, 32));
        assert!(!rejects_unaligned_import_offset(true, 2048));
        assert!(!rejects_unaligned_import_offset(true, 0));
        assert!(rejects_unaligned_import_offset(true, 16416));
        assert!(rejects_unaligned_import_offset(true, 6176));
        assert!(!rejects_unaligned_import_offset(true, 16384));
        // An ungated driver.
        assert!(!rejects_unaligned_import_offset(false, 2080));
        assert!(!rejects_unaligned_import_offset(false, 32));
    }

    // Checks the base `view_collapsed_dst_base` gives with and without a view origin.
    #[test]
    fn a_view_destination_imports_at_base_zero() {
        use super::view_collapsed_dst_base as base;
        use edgefirst_tensor::ViewOrigin;

        // A view: the import bases at 0.
        let vo = Some(ViewOrigin {
            parent_width: 64,
            parent_height: 64,
            parent_row_stride: 256,
            x: 8,
            y: 8,
        });
        assert_eq!(base(vo, Some(2080)), 0);
        assert_eq!(base(vo, Some(2048)), 0);
        assert_eq!(base(vo, None), 0);

        // No view origin: the import bases at the plane offset.
        assert_eq!(base(None, Some(2080)), 2080);
        assert_eq!(base(None, Some(2048)), 2048);
        assert_eq!(base(None, None), 0);

        // The packed import route passes the raw plane offset.
        let raw = |plane_offset: Option<usize>| plane_offset.unwrap_or(0);
        assert_eq!(raw(Some(2080)), 2080);
        assert_ne!(base(vo, Some(2080)), raw(Some(2080)));
    }

    // Checks `nv12_plane1_offset` and the unaligned-offset rule applied to it.
    #[test]
    fn nv12_chroma_plane_offset_is_checked_independently_of_plane_zero() {
        use super::{nv12_plane1_offset as plane1, rejects_unaligned_import_offset as rejects};

        // A 64-aligned pitch at offset 0: both planes aligned.
        assert_eq!(plane1(0, 256, 64), 16384);
        assert_eq!(plane1(0, 256, 65), 16640);
        assert!(!rejects(true, plane1(0, 256, 64)));
        assert!(!rejects(true, plane1(0, 256, 65)));

        // An unpadded pitch: plane 0 aligned, plane 1 unaligned.
        assert!(!rejects(true, 0), "plane 0 looks fine");
        assert_eq!(plane1(0, 100, 3), 300);
        assert!(rejects(true, plane1(0, 100, 3)), "plane 1 must be caught");
        assert_eq!(plane1(0, 48, 5), 240);
        assert!(rejects(true, plane1(0, 48, 5)));
        assert_eq!(plane1(0, 96, 1), 96);
        assert!(rejects(true, plane1(0, 96, 1)));

        // An unpadded pitch whose height keeps plane 1 aligned.
        assert_eq!(plane1(0, 100, 64), 6400);
        assert!(!rejects(true, plane1(0, 100, 64)));
        assert_eq!(plane1(0, 96, 2), 192);
        assert!(!rejects(true, plane1(0, 96, 2)));

        // The luma offset carries into plane 1.
        assert_eq!(plane1(2048, 256, 64), 18432);
        assert!(!rejects(true, plane1(2048, 256, 64)));
        assert!(rejects(true, plane1(32, 256, 64)));

        // The offset saturates.
        assert_eq!(plane1(usize::MAX, 256, 64), usize::MAX);
        assert_eq!(plane1(0, usize::MAX, 2), usize::MAX);

        // A driver the rule does not gate.
        assert!(!rejects(false, plane1(0, 100, 3)));
        assert!(!rejects(false, plane1(0, 96, 1)));
    }

    // Checks the serialization policy each known renderer selects.
    #[test]
    fn serialization_policy_per_renderer() {
        use super::{classify_renderer, requires_full_serialization};
        let policy = |r: &str| requires_full_serialization(classify_renderer(r));

        // Parallel.
        assert!(!policy("Mali-G310"));
        assert!(!policy("V3D 7.1"));
        assert!(!policy("NVIDIA Tegra Orin (nvgpu)/integrated"));

        // Serialized.
        assert!(policy("Vivante GC7000UL"));
        assert!(policy("Adreno (TM) 663"));
        assert!(policy(
            "ANGLE (Apple, ANGLE Metal Renderer: Apple M2 Max, \
             Version 27.0 (Build 26A5388g))"
        ));
        assert!(policy(
            "ANGLE (Apple, ANGLE Metal Renderer: Apple Paravirtual device, \
             Version 15.7.7 (Build 24G720))"
        ));
        assert!(policy(
            "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, \
             D3D11-32.0.16.1656)"
        ));
    }
}
