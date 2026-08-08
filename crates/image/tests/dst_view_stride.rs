// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `convert()` into a `Tensor::view()` destination must render into that
//! sub-rectangle of the parent — every output row at the **parent's** row
//! pitch, and nothing written outside the view region.
//!
//! `Tensor::view`'s own contract says "`convert(src, &mut dst.view(rect), …)`
//! renders into that sub-rectangle of `dst`". A view of a parent that is
//! *taller* than the view has the same pitch as the view, so a tightly-packed
//! writer happens to land correctly; a parent that is **wider** does not, and a
//! tight writer packs the whole output into the head of the parent buffer as a
//! short full-width band.
//!
//! Every case here pre-fills the parent with a poison byte and asserts three
//! things: the view region matches an identical convert into an exact-sized
//! standalone tensor, every parent byte outside the view region is still
//! poison, and (implied by the first two) each output row landed at the parent
//! pitch rather than the view's tight width.
//!
//! **Planar destinations are out of scope by construction**: `Tensor::view`
//! rejects non-packed layouts (`PlanarRgb`/`PlanarRgba`/the NV family), so a
//! planar dst view cannot be built at all. The planar writers take their pitch
//! from `tensor_row_stride(dst)` already; `padded_planar_dst_honours_stride`
//! guards that seam with an explicitly strided (not viewed) planar tensor.

use edgefirst_image::{CPUProcessor, Crop, Flip, ImageProcessorTrait, Region, Rotation};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

/// Byte the parent buffer is pre-filled with; any of these left inside the view
/// region means the convert under-wrote, any byte outside that is *not* poison
/// means it over-wrote.
const POISON: u8 = 0xAB;

const SRC_W: usize = 16;
const SRC_H: usize = 8;
const PARENT_W: usize = 64;
const PARENT_H: usize = 48;

fn mem_image(w: usize, h: usize, fmt: PixelFormat) -> TensorDyn {
    TensorDyn::image(
        w,
        h,
        fmt,
        DType::U8,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("allocate image")
}

/// A deterministic, non-uniform source image: every byte is a function of its
/// offset, so a mis-strided write cannot coincidentally match.
fn make_src(w: usize, h: usize, fmt: PixelFormat) -> TensorDyn {
    let mut t = mem_image(w, h, fmt);
    {
        let u8t = t.as_u8_mut().unwrap();
        let mut map = u8t.map_mut().unwrap();
        for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
            *b = (i.wrapping_mul(37).wrapping_add(11) % 251) as u8;
        }
    }
    t
}

fn fill(t: &mut TensorDyn, byte: u8) {
    let u8t = t.as_u8_mut().unwrap();
    let mut map = u8t.map_mut().unwrap();
    map.as_mut_slice().fill(byte);
}

fn bytes(t: &TensorDyn) -> Vec<u8> {
    let u8t = t.as_u8().unwrap();
    let map = t.as_u8().map(|_| u8t.map_read().unwrap()).unwrap();
    map.as_slice().to_vec()
}

/// Convert `src` into a `region` view of a `PARENT_W × PARENT_H` parent of
/// `dst_fmt`, and into an exact-sized standalone tensor, then assert the view
/// destination landed at the parent pitch with nothing written outside it.
fn assert_view_dst_matches_standalone(
    label: &str,
    src: &TensorDyn,
    dst_fmt: PixelFormat,
    region: Region,
    crop: Crop,
) {
    let mut cpu = CPUProcessor::new();

    // Reference: the same convert into a tensor sized exactly to the view.
    let mut reference = mem_image(region.width, region.height, dst_fmt);
    fill(&mut reference, POISON);
    cpu.convert(src, &mut reference, Rotation::None, Flip::None, crop)
        .unwrap_or_else(|e| panic!("{label}: standalone convert failed: {e}"));
    let ref_bytes = bytes(&reference);
    let ref_stride = reference
        .effective_row_stride()
        .expect("reference has a stride");

    // Subject: the same convert into a view of a WIDER parent.
    let mut parent = mem_image(PARENT_W, PARENT_H, dst_fmt);
    fill(&mut parent, POISON);
    let parent_stride = parent.effective_row_stride().expect("parent has a stride");
    {
        let mut view = parent.view(region).unwrap_or_else(|e| {
            panic!("{label}: view({region:?}) of a {dst_fmt:?} parent failed: {e}")
        });
        assert_eq!(
            view.effective_row_stride(),
            Some(parent_stride),
            "{label}: a multi-row view must report the PARENT row pitch"
        );
        cpu.convert(src, &mut view, Rotation::None, Flip::None, crop)
            .unwrap_or_else(|e| panic!("{label}: view convert failed: {e}"));
    }
    let got = bytes(&parent);

    let bpp = dst_fmt.channels();
    let row_bytes = region.width * bpp;
    let x_off = region.x * bpp;

    // (a) + (c): each output row sits at the parent pitch and matches the
    // standalone convert's corresponding row.
    for row in 0..region.height {
        let start = (region.y + row) * parent_stride + x_off;
        let got_row = &got[start..start + row_bytes];
        let want_row = &ref_bytes[row * ref_stride..row * ref_stride + row_bytes];
        assert_eq!(
            got_row, want_row,
            "{label}: row {row} of the view region does not match the standalone convert \
             (parent_stride={parent_stride}, offset={start})"
        );
    }

    // (b): nothing outside the view region was touched.
    for y in 0..PARENT_H {
        for x in 0..PARENT_W * bpp {
            let inside = y >= region.y
                && y < region.y + region.height
                && x >= x_off
                && x < x_off + row_bytes;
            if inside {
                continue;
            }
            let off = y * parent_stride + x;
            assert_eq!(
                got[off], POISON,
                "{label}: byte outside the view region was overwritten at \
                 (row {y}, byte {x}), offset {off}"
            );
        }
    }
}

fn full_view() -> Region {
    Region::new(0, 0, SRC_W, SRC_H)
}

/// The downstream-reported case: a packed RGB source, an RGBA view of a wider
/// parent, identity scale.
#[test]
fn rgb_to_rgba_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→rgba",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::default(),
    );
}

/// Semi-planar source into a packed view — the JPEG-decode → tile shape.
#[test]
fn nv12_to_rgba_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Nv12);
    assert_view_dst_matches_standalone(
        "nv12→rgba",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::default(),
    );
}

/// 3-byte packed destination: a different bpp, and a different writer.
#[test]
fn nv12_to_rgb_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Nv12);
    assert_view_dst_matches_standalone(
        "nv12→rgb",
        &src,
        PixelFormat::Rgb,
        full_view(),
        Crop::default(),
    );
}

/// `Rgba → Rgba` and `Rgb → Rgb` take the whole-buffer `copy_image` path, which
/// compares total mapped lengths — a view maps `parent_stride × rows`, so this
/// is the latent `InvalidShape` case rather than a mis-write.
#[test]
fn rgba_to_rgba_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgba);
    assert_view_dst_matches_standalone(
        "rgba→rgba",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::default(),
    );
}

#[test]
fn rgb_to_rgb_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→rgb",
        &src,
        PixelFormat::Rgb,
        full_view(),
        Crop::default(),
    );
}

/// A packed source narrowing to 3 channels.
#[test]
fn rgba_to_rgb_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgba);
    assert_view_dst_matches_standalone(
        "rgba→rgb",
        &src,
        PixelFormat::Rgb,
        full_view(),
        Crop::default(),
    );
}

/// BGRA destination: converts to RGBA and then swizzles R/B across the mapped
/// buffer — the swizzle must not reach the parent pixels beside the view.
#[test]
fn rgb_to_bgra_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→bgra",
        &src,
        PixelFormat::Bgra,
        full_view(),
        Crop::default(),
    );
}

/// Single-channel packed destination.
#[test]
fn rgb_to_grey_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→grey",
        &src,
        PixelFormat::Grey,
        full_view(),
        Crop::default(),
    );
}

/// YUYV destination: a 2-byte-per-pixel packed macropixel writer.
#[test]
fn rgb_to_yuyv_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→yuyv",
        &src,
        PixelFormat::Yuyv,
        full_view(),
        Crop::default(),
    );
}

/// YUYV source into a packed view.
#[test]
fn yuyv_to_rgb_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Yuyv);
    assert_view_dst_matches_standalone(
        "yuyv→rgb",
        &src,
        PixelFormat::Rgb,
        full_view(),
        Crop::default(),
    );
}

/// A view that is not at the parent origin — the tile shape a real SAHI grid
/// produces. Exercises the `plane_offset` path as well as the pitch.
#[test]
fn rgb_to_rgba_offset_view_of_wider_parent() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→rgba@(16,8)",
        &src,
        PixelFormat::Rgba,
        Region::new(16, 8, SRC_W, SRC_H),
        Crop::default(),
    );
}

/// A genuine resize into a view destination: the resize writes the scaled
/// rect and (for a format change) a final convert copies it into `dst`.
#[test]
fn scaled_rgb_to_rgba_view_of_wider_parent() {
    let src = make_src(SRC_W / 2, SRC_H / 2, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb(8x4)→rgba(16x8) scaled",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::default(),
    );
}

/// Scaled NV12 source — the resize path with a semi-planar input.
#[test]
fn scaled_nv12_to_rgba_view_of_wider_parent() {
    let src = make_src(SRC_W / 2, SRC_H / 2, PixelFormat::Nv12);
    assert_view_dst_matches_standalone(
        "nv12(8x4)→rgba(16x8) scaled",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::default(),
    );
}

/// A letterbox fit clears the destination to the pad colour and renders the
/// aspect-preserved content into its inner region — for a view destination the
/// clear must stay inside the view, not paint the parent.
#[test]
fn letterbox_rgb_to_rgba_view_of_wider_parent() {
    // 16x4 source into a 16x8 view: a horizontal pad band above and below.
    let src = make_src(SRC_W, SRC_H / 2, PixelFormat::Rgb);
    assert_view_dst_matches_standalone(
        "rgb→rgba letterbox",
        &src,
        PixelFormat::Rgba,
        full_view(),
        Crop::letterbox([114, 114, 114, 255]),
    );
}

/// Fill a typed tensor with an out-of-band value, so any byte the convert
/// leaves alone is distinguishable from a value it could legitimately write.
fn poison(t: &mut TensorDyn) {
    match t {
        TensorDyn::I8(t) => t.map_mut().unwrap().as_mut_slice().fill(-91),
        TensorDyn::F32(t) => t.map_mut().unwrap().as_mut_slice().fill(-7.5),
        TensorDyn::F16(t) => t
            .map_mut()
            .unwrap()
            .as_mut_slice()
            .fill(half::f16::from_f32(-7.5)),
        other => panic!("poison: unhandled dtype {:?}", other.dtype()),
    }
}

/// The mapped element bytes of a typed tensor, little-endian, so rows of any
/// dtype can be compared at a byte pitch.
fn raw_bytes(t: &TensorDyn) -> Vec<u8> {
    match t {
        TensorDyn::I8(t) => t
            .map_read()
            .unwrap()
            .as_slice()
            .iter()
            .map(|v| *v as u8)
            .collect(),
        TensorDyn::F32(t) => t
            .map_read()
            .unwrap()
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorDyn::F16(t) => t
            .map_read()
            .unwrap()
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        other => panic!("raw_bytes: unhandled dtype {:?}", other.dtype()),
    }
}

/// Non-`u8` destinations run the same convert into a scratch and then widen /
/// bias into `dst`; that final write is the same seam and must also be
/// row-confined. Compared against an exact-sized standalone convert.
fn assert_typed_view_dst_matches_standalone(label: &str, dtype: DType, elem: usize) {
    let mut cpu = CPUProcessor::new();
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgb);
    let region = full_view();
    let dst_fmt = PixelFormat::Rgba;

    let alloc = |w: usize, h: usize| {
        TensorDyn::image(
            w,
            h,
            dst_fmt,
            dtype,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .expect("allocate typed image")
    };

    let mut reference = alloc(region.width, region.height);
    cpu.convert(
        &src,
        &mut reference,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .unwrap_or_else(|e| panic!("{label}: standalone convert failed: {e}"));

    let mut parent = alloc(PARENT_W, PARENT_H);
    poison(&mut parent);
    let parent_stride = parent.effective_row_stride().expect("parent has a stride");
    let before = raw_bytes(&parent);
    {
        let mut view = parent
            .view(region)
            .unwrap_or_else(|e| panic!("{label}: view failed: {e}"));
        cpu.convert(&src, &mut view, Rotation::None, Flip::None, Crop::default())
            .unwrap_or_else(|e| panic!("{label}: view convert failed: {e}"));
    }

    // Compare the raw element bytes of each row.
    let want = raw_bytes(&reference);
    let got = raw_bytes(&parent);
    let row_bytes = region.width * dst_fmt.channels() * elem;
    let ref_stride = row_bytes;
    for row in 0..region.height {
        assert_eq!(
            &got[row * parent_stride..row * parent_stride + row_bytes],
            &want[row * ref_stride..row * ref_stride + row_bytes],
            "{label}: row {row} does not match the standalone convert \
             (parent_stride={parent_stride})"
        );
    }

    // Nothing outside the view region moved.
    for (off, (g, b)) in got.iter().zip(before.iter()).enumerate() {
        let (y, x) = (off / parent_stride, off % parent_stride);
        if y < region.height && x < row_bytes {
            continue;
        }
        assert_eq!(
            g, b,
            "{label}: element byte outside the view region changed at \
             (row {y}, byte {x}), offset {off}"
        );
    }
}

#[test]
fn rgb_to_int8_rgba_view_of_wider_parent() {
    assert_typed_view_dst_matches_standalone("rgb→i8 rgba", DType::I8, 1);
}

#[test]
fn rgb_to_f32_rgba_view_of_wider_parent() {
    assert_typed_view_dst_matches_standalone("rgb→f32 rgba", DType::F32, 4);
}

#[test]
fn rgb_to_f16_rgba_view_of_wider_parent() {
    assert_typed_view_dst_matches_standalone("rgb→f16 rgba", DType::F16, 2);
}

/// `Tensor::view` cannot express a planar sub-rectangle, so the planar writers
/// are reached with a padded (strided) standalone tensor instead. They take
/// their pitch from the tensor's row stride already; this pins that.
#[test]
fn padded_planar_dst_honours_stride() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Nv12);
    let mut cpu = CPUProcessor::new();

    assert!(
        mem_image(PARENT_W, PARENT_H, PixelFormat::PlanarRgb)
            .view(full_view())
            .is_err(),
        "Tensor::view must keep rejecting planar layouts — if this starts \
         succeeding, planar dst views need their own coverage here"
    );

    // `image_with_stride` is DMA/Linux-only, so build the padded planar tensor
    // by allocating at the padded width and re-declaring the logical geometry.
    let stride = PARENT_W * 4;
    let mut padded = mem_image(stride, SRC_H, PixelFormat::PlanarRgb);
    fill(&mut padded, POISON);
    padded
        .configure_image(SRC_W, SRC_H, PixelFormat::PlanarRgb)
        .expect("shrink logical geometry");
    padded.set_row_stride(stride).expect("declare padded pitch");
    cpu.convert(
        &src,
        &mut padded,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .expect("planar convert");

    let mut reference = mem_image(SRC_W, SRC_H, PixelFormat::PlanarRgb);
    cpu.convert(
        &src,
        &mut reference,
        Rotation::None,
        Flip::None,
        Crop::default(),
    )
    .expect("planar reference convert");

    let got = bytes(&padded);
    let want = bytes(&reference);
    for plane in 0..3 {
        for row in 0..SRC_H {
            let g = &got[(plane * SRC_H + row) * stride..][..SRC_W];
            let w = &want[(plane * SRC_H + row) * SRC_W..][..SRC_W];
            assert_eq!(g, w, "planar plane {plane} row {row} mismatch");
        }
    }
}
