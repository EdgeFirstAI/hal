// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Image geometry of a formatted tensor whose shape carries a leading batch
//! dimension.
//!
//! `set_logical_shape` keeps the pixel format, so an `[N*H, W, C]` image
//! re-viewed as `[N, H, W, C]` is a formatted rank-4 tensor reachable through
//! the public API alone. Its width, height and row stride must describe one
//! element `[H, W, C]`, and `batch(n)` must snapshot the tall `(W, N*H)`
//! parent at the element's real row pitch -- not at `H * C` bytes, which is
//! what reading `shape[1]` as the width produces.

#![cfg(feature = "static")]

use edgefirst_tensor::{
    CpuAccess, Error, PixelFormat, Tensor, TensorMapTrait, TensorMemory, TensorTrait, ViewOrigin,
};

/// A `[4, 3, 5, 4]` RGBA batch: four 5-wide, 3-tall tiles stacked in one
/// `5 x 12` image allocation. Width and height differ so the two cannot be
/// confused.
fn rgba_batch<T>() -> Tensor<T>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + 'static,
{
    let mut t = Tensor::<T>::image(
        5,
        12,
        PixelFormat::Rgba,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem RGBA 5x12");
    t.set_logical_shape(&[4, 3, 5, 4])
        .expect("[4, 3, 5, 4] is the same bytes as [12, 5, 4]");
    assert_eq!(
        t.format(),
        Some(PixelFormat::Rgba),
        "set_logical_shape keeps the format"
    );
    t
}

#[test]
fn batched_packed_image_reports_the_element_geometry() {
    let t = rgba_batch::<u8>();
    assert_eq!(t.width(), Some(5));
    assert_eq!(t.height(), Some(3));
    assert_eq!(t.effective_row_stride(), Some(5 * 4));

    let t = rgba_batch::<u16>();
    assert_eq!(t.effective_row_stride(), Some(5 * 4 * 2));
}

#[test]
fn batch_of_a_packed_image_snapshots_the_tall_parent() {
    let t = rgba_batch::<u8>();
    let e = t.batch(2).expect("batch(2) of four");
    assert_eq!(e.shape(), &[3, 5, 4]);
    assert_eq!(e.plane_offset(), Some(2 * 3 * 5 * 4));
    assert_eq!(
        e.view_origin(),
        Some(ViewOrigin {
            parent_width: 5,
            parent_height: 12,
            parent_row_stride: 20,
            x: 0,
            y: 6,
        })
    );
}

#[test]
fn batch_of_a_padded_packed_image_snapshots_the_padded_pitch() {
    let mut t = rgba_batch::<u16>();
    // 5 px * 4 ch * 2 B = 40 B tight; the recorded stride wins over it.
    t.set_row_stride(64).expect("64 >= 40");
    let e = t.batch(1).expect("batch(1)");
    assert_eq!(e.view_origin().map(|v| v.parent_row_stride), Some(64));
    assert_eq!(e.view_origin().map(|v| v.y), Some(3));
}

#[test]
fn batch_writes_land_in_their_own_tile() {
    let t = rgba_batch::<u8>();
    for n in 0..4u8 {
        let e = t.batch(n as usize).unwrap();
        e.map().unwrap().as_mut_slice().fill(n + 1);
    }
    let m = t.map().unwrap();
    for (n, tile) in m.as_slice().chunks(3 * 5 * 4).enumerate() {
        assert!(tile.iter().all(|&b| b == n as u8 + 1), "tile {n}");
    }
}

#[test]
fn batch_of_a_semi_planar_image_needs_a_leading_n() {
    let mut t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem NV12 640x480");
    // Unbatched `[720, 640]`: refused, naming the missing N.
    match t.batch(0) {
        Err(Error::InvalidShape(msg)) => assert!(msg.contains("not batched"), "{msg}"),
        other => panic!("expected InvalidShape, got {other:?}"),
    }
    // `[2, 360, 640]` is two 640x240 NV12 frames.
    t.set_logical_shape(&[2, 360, 640]).unwrap();
    let e = t.batch(1).expect("batch(1) of two NV12 frames");
    assert_eq!(e.shape(), &[360, 640]);
    assert_eq!(e.width(), Some(640));
    assert_eq!(e.height(), Some(240));
    assert_eq!(
        e.view_origin(),
        None,
        "semi-planar batching keeps the per-slot path"
    );
}

#[test]
fn batch_of_an_unbatched_packed_image_is_refused() {
    let t = Tensor::<u8>::image(
        5,
        3,
        PixelFormat::Rgba,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    match t.batch(0) {
        Err(Error::InvalidShape(msg)) => {
            assert!(
                msg.contains("3-D element") && msg.contains("[3, 5, 4]"),
                "{msg}"
            )
        }
        other => panic!("expected InvalidShape, got {other:?}"),
    }
}

#[test]
fn flattened_image_has_no_image_geometry() {
    // A caller may flatten an image and leave the format behind; the
    // geometry accessors must then say "unknown" rather than index a
    // dimension that is not there.
    let mut t = Tensor::<u8>::image(
        5,
        3,
        PixelFormat::Rgba,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    t.set_logical_shape(&[60]).unwrap();
    assert_eq!(t.format(), Some(PixelFormat::Rgba));
    assert_eq!(t.width(), None);
    assert_eq!(t.height(), None);
    assert_eq!(t.effective_row_stride(), None);
}
