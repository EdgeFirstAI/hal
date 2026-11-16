// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `convert()` into a `Tensor::view()` destination renders into that
//! sub-rectangle of the parent: every output row at the parent's row pitch,
//! and nothing written outside the view region.
//!
//! Every case pre-fills the parent with a poison byte and asserts that the
//! view region matches an identical convert into an exact-sized standalone
//! tensor and that every parent byte outside the view region is still poison.
//!
//! Planar destinations are reached with a padded-stride tensor, and the GL and
//! PBO readbacks with a declared row stride.

#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
use edgefirst_image::GLProcessorThreaded;
use edgefirst_image::{CPUProcessor, Crop, ImageProcessor, ImageProcessorTrait, Region};
use edgefirst_tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

#[path = "../../tensor/tests/support/outcome.rs"]
#[allow(dead_code)]
mod outcome;

/// Byte the parent buffer is pre-filled with.
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

/// A deterministic, non-uniform source image: every byte is a function of its offset.
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
    cpu.convert(
        src,
        &mut reference,
        Some(edgefirst_image::ConvertOptions::new().crop(crop)),
    )
    .unwrap_or_else(|e| panic!("{label}: standalone convert failed: {e}"));
    let ref_bytes = bytes(&reference);
    let ref_stride = reference
        .effective_row_stride()
        .expect("reference has a stride");

    // Subject: the same convert into a view of a wider parent.
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
        cpu.convert(
            src,
            &mut view,
            Some(edgefirst_image::ConvertOptions::new().crop(crop)),
        )
        .unwrap_or_else(|e| panic!("{label}: view convert failed: {e}"));
    }
    let got = bytes(&parent);

    let bpp = dst_fmt.channels();
    let row_bytes = region.width * bpp;
    let x_off = region.x * bpp;

    // Each output row sits at the parent pitch and matches the standalone convert's row.
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

    // Nothing outside the view region was touched.
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

/// A packed RGB source into an RGBA view of a wider parent, at identity scale.
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

/// Semi-planar source into a packed view.
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

/// 3-byte packed destination.
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

/// `Rgba → Rgba` and `Rgb → Rgb`, which take the whole-buffer `copy_image` path.
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

/// BGRA destination: the R/B swizzle stays inside the view.
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

/// A view that is not at the parent origin.
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

/// A resize into a view destination.
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

/// Scaled NV12 source into a view destination.
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

/// A letterbox fit into a view: the pad clear stays inside the view.
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

/// Fill a typed tensor with an out-of-band value.
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

/// The mapped element bytes of a typed tensor, little-endian.
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

/// Convert into a non-`u8` view destination and compare against an exact-sized
/// standalone convert.
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
    cpu.convert(&src, &mut reference, None)
        .unwrap_or_else(|e| panic!("{label}: standalone convert failed: {e}"));

    let mut parent = alloc(PARENT_W, PARENT_H);
    poison(&mut parent);
    let parent_stride = parent.effective_row_stride().expect("parent has a stride");
    let before = raw_bytes(&parent);
    {
        let mut view = parent
            .view(region)
            .unwrap_or_else(|e| panic!("{label}: view failed: {e}"));
        cpu.convert(&src, &mut view, None)
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

/// A planar convert into a padded-stride tensor lands each row at the declared stride.
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

    // Builds the padded planar tensor by allocating at the padded width and
    // re-declaring the logical geometry.
    let stride = PARENT_W * 4;
    let mut padded = mem_image(stride, SRC_H, PixelFormat::PlanarRgb);
    fill(&mut padded, POISON);
    padded
        .configure_image(SRC_W, SRC_H, PixelFormat::PlanarRgb)
        .expect("shrink logical geometry");
    padded.set_row_stride(stride).expect("declare padded pitch");
    cpu.convert(&src, &mut padded, None)
        .expect("planar convert");

    let mut reference = mem_image(SRC_W, SRC_H, PixelFormat::PlanarRgb);
    cpu.convert(&src, &mut reference, None)
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

/// The GL readback lands each output row at the declared stride, for `Rgba`, `Rgb` and `Grey`.
#[test]
fn gl_padded_dst_rows_land_at_the_declared_stride() {
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgba);
    let mut gl = ImageProcessor::new().expect("create ImageProcessor");

    for dst_fmt in [PixelFormat::Rgba, PixelFormat::Rgb, PixelFormat::Grey] {
        let bpp = dst_fmt.channels();
        // Allocates at the padded width, then re-declares the logical geometry and the pitch.
        let stride = PARENT_W * bpp;
        let mut padded = mem_image(PARENT_W, SRC_H, dst_fmt);
        fill(&mut padded, POISON);
        padded
            .configure_image(SRC_W, SRC_H, dst_fmt)
            .expect("shrink logical geometry");
        padded.set_row_stride(stride).expect("declare padded pitch");
        gl.convert(&src, &mut padded, None)
            .unwrap_or_else(|e| panic!("{dst_fmt:?}: padded convert failed: {e}"));

        let mut reference = mem_image(SRC_W, SRC_H, dst_fmt);
        fill(&mut reference, POISON);
        gl.convert(&src, &mut reference, None)
            .unwrap_or_else(|e| panic!("{dst_fmt:?}: reference convert failed: {e}"));
        let ref_stride = reference.effective_row_stride().expect("reference stride");
        assert_eq!(
            ref_stride,
            SRC_W * bpp,
            "{dst_fmt:?}: the reference destination must be tight"
        );

        let got = bytes(&padded);
        let want = bytes(&reference);
        let row_bytes = SRC_W * bpp;
        for row in 0..SRC_H {
            assert_eq!(
                &got[row * stride..row * stride + row_bytes],
                &want[row * ref_stride..row * ref_stride + row_bytes],
                "{dst_fmt:?}: row {row} did not land at the {stride} B pitch"
            );
        }

        if dst_fmt == PixelFormat::Rgba
            && std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1")
        {
            assert_eq!(
                gl.convert_fallback_count(),
                0,
                "the Rgba convert fell back to the CPU backend, so the GL \
                 readback under test never ran"
            );
        }
    }

    use std::io::Write;
    let _ = writeln!(
        &mut std::io::stderr(),
        "NOTE: gl_padded_dst_rows_land_at_the_declared_stride: gl_fallbacks={}",
        gl.convert_fallback_count()
    );
}

#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
use outcome::report_skip as skip;

/// The PBO readback lands each output row at the declared stride, whole-pixel or not.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
#[test]
fn gl_padded_pbo_dst_rows_land_at_the_declared_stride() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    let src = make_src(SRC_W, SRC_H, PixelFormat::Rgba);
    let mut gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
            );
            skip(&format!(
                "gl_padded_pbo_dst_rows_land_at_the_declared_stride: no GL backend: {e}"
            ));
            return;
        }
    };

    // (destination format, bytes of pitch beyond the wide allocation's row)
    let cases = [
        (PixelFormat::Rgba, 0usize),
        (PixelFormat::Rgba, 1),
        (PixelFormat::Bgra, 0),
        (PixelFormat::Bgra, 1),
        (PixelFormat::Rgb, 0),
        (PixelFormat::Grey, 0),
    ];
    for (dst_fmt, extra) in cases {
        let bpp = dst_fmt.channels();
        let stride = PARENT_W * bpp + extra;
        let label = format!("{dst_fmt:?} at a {stride} B pitch");

        // Allocates wide with one spare row, then re-declares the logical geometry and the pitch.
        let mut padded = gl
            .create_pbo_image(PARENT_W, SRC_H + 1, dst_fmt)
            .unwrap_or_else(|e| panic!("{label}: a PBO on a live GL context: {e}"));
        padded
            .configure_image(SRC_W, SRC_H, dst_fmt)
            .expect("shrink logical geometry");
        padded.set_row_stride(stride).expect("declare padded pitch");
        let mut padded = TensorDyn::from(padded);
        fill(&mut padded, POISON);
        gl.convert(&src, &mut padded, None)
            .unwrap_or_else(|e| panic!("{label}: padded convert failed: {e}"));

        // A tight PBO of the same render is the row reference; it is also
        // compared with the Mem readback.
        let mut reference = TensorDyn::from(
            gl.create_pbo_image(SRC_W, SRC_H, dst_fmt)
                .unwrap_or_else(|e| panic!("{label}: tight PBO: {e}")),
        );
        fill(&mut reference, POISON);
        gl.convert(&src, &mut reference, None)
            .unwrap_or_else(|e| panic!("{label}: reference convert failed: {e}"));
        let row_bytes = SRC_W * bpp;
        assert_eq!(
            reference.effective_row_stride(),
            Some(row_bytes),
            "{label}: the reference destination must be tight"
        );
        let mut mem_reference = mem_image(SRC_W, SRC_H, dst_fmt);
        fill(&mut mem_reference, POISON);
        gl.convert(&src, &mut mem_reference, None)
            .unwrap_or_else(|e| panic!("{label}: Mem reference convert failed: {e}"));
        let want = bytes(&reference);
        assert_eq!(
            want,
            bytes(&mem_reference),
            "{label}: the tight PBO readback diverged from the Mem readback"
        );

        let got = bytes(&padded);
        for row in 0..SRC_H {
            assert_eq!(
                &got[row * stride..row * stride + row_bytes],
                &want[row * row_bytes..row * row_bytes + row_bytes],
                "{label}: row {row} did not land at the {stride} B pitch"
            );
            // The padding bytes between rows are not asserted.
        }
    }
}

/// An overlay draw into a PBO `view()` destination is refused and leaves the parent untouched.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_os = "windows"
    ),
    feature = "opengl"
))]
#[test]
fn gl_pbo_view_dst_is_refused_rather_than_written_at_offset_zero() {
    let require_gl = std::env::var("HAL_TEST_REQUIRE_GL").is_ok_and(|v| v == "1");
    let mut gl = match GLProcessorThreaded::new(None) {
        Ok(gl) => gl,
        Err(e) => {
            assert!(
                !require_gl,
                "HAL_TEST_REQUIRE_GL=1 but the GL backend failed to come up: {e}"
            );
            skip(&format!(
                "gl_pbo_view_dst_is_refused_rather_than_written_at_offset_zero: no GL backend: {e}"
            ));
            return;
        }
    };
    let parent = gl
        .create_pbo_image(PARENT_W, PARENT_H, PixelFormat::Rgba)
        .expect("a PBO on a live GL context");
    let mut parent = TensorDyn::from(parent);
    fill(&mut parent, POISON);
    let before = bytes(&parent);

    let view = parent
        .as_u8()
        .expect("u8 parent")
        .view(Region::new(0, 0, SRC_W, SRC_H))
        .expect("view of a PBO parent");
    assert!(
        view.view_origin().is_some(),
        "the destination must report itself as a sub-region"
    );
    let mut view = TensorDyn::from(view);

    let err = gl
        .draw_decoded_masks(&mut view, &[], &[], Default::default())
        .expect_err("an overlay into a PBO view must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("offset 0"),
        "the refusal must name the offset-0 write it is declining, got: {msg}"
    );
    assert_eq!(
        bytes(&parent),
        before,
        "the refused overlay wrote into the parent buffer anyway"
    );
}
