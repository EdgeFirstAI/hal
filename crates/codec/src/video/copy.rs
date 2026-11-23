// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Copies of NV12 and packed pictures between buffers whose planes start at
//! different offsets and whose rows have different pitches.

use edgefirst_tensor::{CpuAccess, PixelFormat, PixelLayout, TensorDyn};

use crate::CodecError;

/// Where one plane of a picture starts in a buffer, and the distance from
/// one of its rows to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plane {
    pub offset: usize,
    pub pitch: usize,
}

/// The bytes per row and the rows of each plane of a `width` x `height`
/// picture in `format`, or `None` for a format the codecs do not carry.
pub(crate) fn plane_shapes(
    format: PixelFormat,
    width: usize,
    height: usize,
) -> Option<Vec<(usize, usize)>> {
    match format {
        PixelFormat::Nv12 => Some(vec![
            (width, height),
            (width.div_ceil(2) * 2, height.div_ceil(2)),
        ]),
        PixelFormat::Yuyv | PixelFormat::Vyuy => Some(vec![(width.div_ceil(2) * 4, height)]),
        f if f.layout() == PixelLayout::Packed => Some(vec![(width * f.channels(), height)]),
        _ => None,
    }
}

/// The planes of a picture whose planes follow one another in one buffer,
/// each `pitch` bytes per row, the chroma after `rows` luma rows.
#[cfg_attr(
    not(any(
        windows,
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) fn contiguous(format: PixelFormat, pitch: usize, rows: usize) -> Vec<Plane> {
    let luma = Plane { offset: 0, pitch };
    match format {
        PixelFormat::Nv12 => vec![
            luma,
            Plane {
                offset: pitch * rows,
                pitch,
            },
        ],
        _ => vec![luma],
    }
}

/// Moves each plane of `src` by `(x, y)` pixels: the planes of the region
/// of a `format` picture that starts there. `x` and `y` must be on a chroma
/// sample.
pub(crate) fn offset(format: PixelFormat, src: &[Plane], (x, y): (usize, usize)) -> Vec<Plane> {
    let bytes_per_pixel = match format.layout() {
        PixelLayout::Packed => format.channels(),
        _ => 1,
    };
    src.iter()
        .enumerate()
        .map(|(i, p)| {
            // NV12 chroma has half the rows, and a byte per pixel per row.
            let rows = if i == 0 { y } else { y / 2 };
            Plane {
                offset: p.offset + rows * p.pitch + x * bytes_per_pixel,
                pitch: p.pitch,
            }
        })
        .collect()
}

/// Copies a `width` x `height` `format` picture from the planes `from` of
/// `src` to the planes `to` of `dst`.
///
/// # Errors
///
/// A message when a row falls outside either buffer or is longer than a
/// pitch.
pub(crate) fn copy(
    format: PixelFormat,
    (width, height): (usize, usize),
    src: &[u8],
    from: &[Plane],
    dst: &mut [u8],
    to: &[Plane],
) -> Result<(), String> {
    let shapes = plane_shapes(format, width, height)
        .ok_or_else(|| format!("{format} pictures are not copied"))?;
    if from.len() < shapes.len() || to.len() < shapes.len() {
        return Err(format!("{format} has {} planes", shapes.len()));
    }
    for ((&shape, s), d) in shapes.iter().zip(from).zip(to) {
        copy_plane(src, *s, dst, *d, shape)?;
    }
    Ok(())
}

/// Copies `rows` rows of `len` bytes from plane `s` of `src` to plane `d`
/// of `dst`.
///
/// # Errors
///
/// A message when a row falls outside either buffer or is longer than a
/// pitch.
pub(crate) fn copy_plane(
    src: &[u8],
    s: Plane,
    dst: &mut [u8],
    d: Plane,
    (len, rows): (usize, usize),
) -> Result<(), String> {
    if len > s.pitch || len > d.pitch {
        return Err(format!(
            "a {len}-byte row is longer than a pitch ({} from, {} to)",
            s.pitch, d.pitch
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    let src_end = s.offset + (rows - 1) * s.pitch + len;
    let dst_end = d.offset + (rows - 1) * d.pitch + len;
    if src_end > src.len() {
        return Err(format!(
            "the source plane ends at byte {src_end} of {}",
            src.len()
        ));
    }
    if dst_end > dst.len() {
        return Err(format!(
            "the destination plane ends at byte {dst_end} of {}",
            dst.len()
        ));
    }
    if s.pitch == len && d.pitch == len {
        dst[d.offset..dst_end].copy_from_slice(&src[s.offset..src_end]);
        return Ok(());
    }
    for r in 0..rows {
        let (a, b) = (s.offset + r * s.pitch, d.offset + r * d.pitch);
        dst[b..b + len].copy_from_slice(&src[a..a + len]);
    }
    Ok(())
}

/// Repeats the last of the `rows` rows of `len` bytes of plane `d` of `dst`
/// down to row `total`.
///
/// # Errors
///
/// A message when a row falls outside `dst` or is longer than a pitch.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn repeat_last_row(
    dst: &mut [u8],
    d: Plane,
    (len, rows): (usize, usize),
    total: usize,
) -> Result<(), String> {
    if rows == 0 || total <= rows {
        return Ok(());
    }
    if len > d.pitch {
        return Err(format!(
            "a {len}-byte row is longer than the pitch {}",
            d.pitch
        ));
    }
    let end = d.offset + (total - 1) * d.pitch + len;
    if end > dst.len() {
        return Err(format!(
            "the destination plane ends at byte {end} of {}",
            dst.len()
        ));
    }
    let last = d.offset + (rows - 1) * d.pitch;
    for r in rows..total {
        dst.copy_within(last..last + len, d.offset + r * d.pitch);
    }
    Ok(())
}

/// Copies the `size` region of `src` at `origin`, whose rows are `stride`
/// bytes apart, into the planes `to` of `dst`.
pub(crate) fn copy_source(
    format: PixelFormat,
    size: (usize, usize),
    src: &TensorDyn,
    origin: (usize, usize),
    stride: usize,
    dst: &mut [u8],
    to: &[Plane],
) -> crate::Result<()> {
    let read_error = |e: edgefirst_tensor::Error| {
        if e.cpu_access_denied().is_some() {
            CodecError::Tensor(e)
        } else {
            CodecError::InvalidInput(format!("the source cannot be read: {e}"))
        }
    };
    let shapes =
        plane_shapes(format, size.0, size.1).ok_or(CodecError::UnsupportedFormat(format))?;
    let luma = src.map_bytes(CpuAccess::Read).map_err(read_error)?;
    let copied = if src.is_multiplane() {
        let chroma = src.chroma_dyn().ok_or_else(|| {
            CodecError::InvalidInput("multi-plane source has no chroma plane".into())
        })?;
        let cstride = chroma.effective_row_stride().unwrap_or(stride);
        let cmap = chroma.map_bytes(CpuAccess::Read).map_err(read_error)?;
        let planes = offset(
            format,
            &[
                Plane {
                    offset: 0,
                    pitch: stride,
                },
                Plane {
                    offset: 0,
                    pitch: cstride,
                },
            ],
            origin,
        );
        copy_plane(&luma, planes[0], dst, to[0], shapes[0])
            .and_then(|()| copy_plane(&cmap, planes[1], dst, to[1], shapes[1]))
    } else {
        let table = format
            .plane_table(src.width().unwrap_or(0), src.height().unwrap_or(0), stride)
            .ok_or(CodecError::UnsupportedFormat(format))?;
        let planes: Vec<Plane> = table
            .iter()
            .map(|p| Plane {
                offset: p.offset as usize,
                pitch: p.stride as usize,
            })
            .collect();
        copy(
            format,
            size,
            &luma,
            &offset(format, &planes, origin),
            dst,
            to,
        )
    };
    copied.map_err(CodecError::InvalidInput)
}

/// Where the planes of a `size` NV12 image are in a CPU map of `t`.
#[cfg_attr(
    not(any(
        all(windows, feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) fn mapped_planes(t: &TensorDyn, size: (usize, usize)) -> crate::Result<Vec<Plane>> {
    let stride = t
        .effective_row_stride()
        .ok_or_else(|| CodecError::InvalidConfig("an NV12 tensor has no row stride".into()))?;
    Ok(PixelFormat::Nv12
        .plane_table(size.0, size.1, stride)
        .ok_or(CodecError::UnsupportedFormat(PixelFormat::Nv12))?
        .iter()
        .map(|p| Plane {
            offset: p.offset as usize,
            pitch: p.stride as usize,
        })
        .collect())
}

/// A CPU map of `t` for writing a whole picture.
#[cfg_attr(
    not(any(
        all(windows, feature = "media-foundation"),
        all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")
    )),
    allow(dead_code)
)]
pub(crate) fn write_map(t: &TensorDyn) -> crate::Result<edgefirst_tensor::HostView<'static, u8>> {
    Ok(match t.map_bytes(CpuAccess::Write) {
        Err(edgefirst_tensor::Error::PartialWriteRequiresReadWrite(_)) => {
            t.map_bytes(CpuAccess::ReadWrite)
        }
        other => other,
    }?)
}

/// Writes a `size` picture from the planes `from` of `src` into the NV12 or
/// YUYV image `dst`, whether it is one buffer or separate luma and chroma
/// planes.
#[cfg_attr(
    not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")),
    allow(dead_code)
)]
pub(crate) fn copy_into_image(
    format: PixelFormat,
    size: (usize, usize),
    src: &[u8],
    from: &[Plane],
    dst: &TensorDyn,
) -> crate::Result<()> {
    let write_error =
        |e: String| CodecError::InvalidInput(format!("the image cannot be written: {e}"));
    if dst.format() != Some(format) {
        return Err(CodecError::InvalidInput(format!(
            "image format {:?} is not {format}",
            dst.format()
        )));
    }
    let shapes =
        plane_shapes(format, size.0, size.1).ok_or(CodecError::UnsupportedFormat(format))?;
    let stride = dst
        .effective_row_stride()
        .ok_or_else(|| CodecError::InvalidInput("the image has no row stride".into()))?;
    if dst.is_multiplane() {
        let chroma = dst.chroma_dyn().ok_or_else(|| {
            CodecError::InvalidInput("multi-plane image has no chroma plane".into())
        })?;
        let cstride = chroma.effective_row_stride().unwrap_or(stride);
        if from.len() < shapes.len() {
            return Err(write_error(format!("{format} has {} planes", shapes.len())));
        }
        let mut luma = write_map(dst)?;
        let mut cmap = write_map(&chroma)?;
        return copy_plane(
            src,
            from[0],
            &mut luma,
            Plane {
                offset: 0,
                pitch: stride,
            },
            shapes[0],
        )
        .and_then(|()| {
            copy_plane(
                src,
                from[1],
                &mut cmap,
                Plane {
                    offset: 0,
                    pitch: cstride,
                },
                shapes[1],
            )
        })
        .map_err(write_error);
    }
    let to = if format == PixelFormat::Nv12 {
        mapped_planes(dst, size)?
    } else {
        contiguous(format, stride, size.1)
    };
    let mut map = write_map(dst)?;
    copy(format, size, src, from, &mut map, &to).map_err(write_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{check_source, EncoderConfig, FrameOptions};
    use edgefirst_tensor::{DType, Region, Tensor, TensorMemory};

    #[test]
    fn nv12_planes_move_to_the_destination_pitch() {
        // 4x2 NV12 at a pitch of 6: two luma rows, then one chroma row.
        let src: Vec<u8> = (0..18).collect();
        let mut dst = vec![0xff; 24];
        copy(
            PixelFormat::Nv12,
            (4, 2),
            &src,
            &contiguous(PixelFormat::Nv12, 6, 2),
            &mut dst,
            &contiguous(PixelFormat::Nv12, 8, 2),
        )
        .unwrap();
        assert_eq!(&dst[0..4], &[0, 1, 2, 3]);
        assert_eq!(&dst[8..12], &[6, 7, 8, 9]);
        assert_eq!(&dst[16..20], &[12, 13, 14, 15]);
        assert_eq!(dst[4], 0xff, "padding is left alone");
    }

    #[test]
    fn a_region_starts_at_its_origin_in_every_plane() {
        // An 8x4 NV12 picture at pitch 8; take the 4x2 region at (2, 2).
        let src: Vec<u8> = (0..48).collect();
        let from = offset(
            PixelFormat::Nv12,
            &contiguous(PixelFormat::Nv12, 8, 4),
            (2, 2),
        );
        assert_eq!(
            from[0],
            Plane {
                offset: 18,
                pitch: 8
            }
        );
        assert_eq!(
            from[1],
            Plane {
                offset: 32 + 8 + 2,
                pitch: 8
            }
        );
        let mut dst = vec![0; 12];
        copy(
            PixelFormat::Nv12,
            (4, 2),
            &src,
            &from,
            &mut dst,
            &contiguous(PixelFormat::Nv12, 4, 2),
        )
        .unwrap();
        assert_eq!(dst, [18, 19, 20, 21, 26, 27, 28, 29, 42, 43, 44, 45]);
    }

    #[test]
    fn yuyv_has_two_bytes_per_pixel() {
        let from = offset(
            PixelFormat::Yuyv,
            &contiguous(PixelFormat::Yuyv, 16, 2),
            (2, 1),
        );
        assert_eq!(
            from,
            vec![Plane {
                offset: 20,
                pitch: 16
            }]
        );
        assert_eq!(plane_shapes(PixelFormat::Yuyv, 6, 2), Some(vec![(12, 2)]));
    }

    #[test]
    fn tight_planes_copy_in_one_piece() {
        let src: Vec<u8> = (0..6).collect();
        let mut dst = vec![0; 6];
        let planes = contiguous(PixelFormat::Nv12, 2, 2);
        copy(PixelFormat::Nv12, (2, 2), &src, &planes, &mut dst, &planes).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn rows_outside_a_buffer_are_errors() {
        let src = vec![0; 10];
        let mut dst = vec![0; 24];
        let to = contiguous(PixelFormat::Nv12, 8, 2);
        let err = copy(
            PixelFormat::Nv12,
            (4, 2),
            &src,
            &contiguous(PixelFormat::Nv12, 6, 2),
            &mut dst,
            &to,
        )
        .unwrap_err();
        assert!(err.contains("source"), "{err}");
        let err = copy(
            PixelFormat::Nv12,
            (8, 2),
            &[0; 24],
            &to,
            &mut dst,
            &contiguous(PixelFormat::Nv12, 4, 2),
        )
        .unwrap_err();
        assert!(err.contains("longer than a pitch"), "{err}");
        assert!(copy(PixelFormat::Nv16, (1, 1), &[0; 4], &to, &mut dst, &to).is_err());
    }

    #[test]
    fn packed_rows_carry_every_channel() {
        assert_eq!(plane_shapes(PixelFormat::Rgba, 5, 3), Some(vec![(20, 3)]));
        assert_eq!(plane_shapes(PixelFormat::Grey, 5, 3), Some(vec![(5, 3)]));
        let from = offset(
            PixelFormat::Rgb,
            &contiguous(PixelFormat::Rgb, 30, 2),
            (2, 1),
        );
        assert_eq!(from[0].offset, 30 + 6);
    }

    #[test]
    fn the_last_row_fills_the_rows_below() {
        // Two 3-byte rows at pitch 4, padded to four rows.
        let mut dst = vec![0; 16];
        dst[..3].copy_from_slice(&[1, 2, 3]);
        dst[4..7].copy_from_slice(&[4, 5, 6]);
        let d = Plane {
            offset: 0,
            pitch: 4,
        };
        repeat_last_row(&mut dst, d, (3, 2), 4).unwrap();
        assert_eq!(&dst[8..11], &[4, 5, 6]);
        assert_eq!(&dst[12..15], &[4, 5, 6]);
        assert_eq!(dst[11], 0, "padding is left alone");
        assert!(repeat_last_row(&mut dst, d, (3, 2), 5).is_err());
        repeat_last_row(&mut dst, d, (3, 2), 2).unwrap();
    }

    #[test]
    fn a_source_declared_without_reads_is_refused_as_denied_access() {
        for declared in [CpuAccess::None, CpuAccess::Write] {
            let src = TensorDyn::image(
                8,
                4,
                PixelFormat::Nv12,
                DType::U8,
                Some(TensorMemory::Mem),
                declared,
            )
            .unwrap();
            let mut dst = vec![0; 48];
            let to = contiguous(PixelFormat::Nv12, 8, 4);
            let err =
                copy_source(PixelFormat::Nv12, (8, 4), &src, (0, 0), 8, &mut dst, &to).unwrap_err();
            assert!(err.cpu_access_denied().is_some(), "{declared:?}: {err}");
        }
    }

    #[test]
    fn odd_sizes_round_chroma_up() {
        assert_eq!(
            plane_shapes(PixelFormat::Nv12, 5, 3),
            Some(vec![(5, 3), (6, 2)])
        );
    }

    fn multiplane_nv12(
        luma: (usize, usize),
        chroma: (usize, usize),
        strides: (usize, usize),
    ) -> TensorDyn {
        let l = Tensor::<u8>::new(&[luma.1, luma.0], Some(TensorMemory::Mem), None).unwrap();
        let mut c =
            Tensor::<u8>::new(&[chroma.1, chroma.0], Some(TensorMemory::Mem), None).unwrap();
        c.set_row_stride_unchecked(strides.1);
        let mut t = Tensor::from_planes(l, c, PixelFormat::Nv12).unwrap();
        t.set_row_stride(strides.0).unwrap();
        TensorDyn::from(t)
    }

    fn fill(t: &TensorDyn, pitch: usize, rows: usize, base: u8) {
        let mut map = t.map_bytes(CpuAccess::ReadWrite).unwrap();
        for r in 0..rows {
            for x in 0..4 {
                map[r * pitch + x] = base + (r * 4 + x) as u8;
            }
        }
    }

    #[test]
    fn copy_source_reads_a_multiplane_source_at_each_planes_stride() {
        // 4x4 NV12: luma rows 8 bytes apart, chroma rows 6 bytes apart.
        let src = multiplane_nv12((8, 4), (8, 2), (8, 6));
        fill(&src, 8, 4, 0);
        fill(&src.chroma_dyn().unwrap(), 6, 2, 100);
        let mut dst = vec![0xff; 24];
        let to = contiguous(PixelFormat::Nv12, 4, 4);
        copy_source(PixelFormat::Nv12, (4, 4), &src, (0, 0), 8, &mut dst, &to).unwrap();
        let luma: Vec<u8> = (0..16).collect();
        assert_eq!(&dst[..16], luma.as_slice());
        assert_eq!(&dst[16..], &[100, 101, 102, 103, 104, 105, 106, 107]);
    }

    #[test]
    fn copy_into_image_writes_single_buffer_and_multiplane_nv12() {
        let tight: Vec<u8> = (0..24).collect();
        let from = contiguous(PixelFormat::Nv12, 4, 4);

        let single = TensorDyn::image(
            4,
            4,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        copy_into_image(PixelFormat::Nv12, (4, 4), &tight, &from, &single).unwrap();
        let planes = mapped_planes(&single, (4, 4)).unwrap();
        let map = single.map_bytes(CpuAccess::Read).unwrap();
        let mut back = vec![0; 24];
        copy(PixelFormat::Nv12, (4, 4), &map, &planes, &mut back, &from).unwrap();
        assert_eq!(back, tight);

        let multi = multiplane_nv12((8, 4), (8, 2), (8, 6));
        copy_into_image(PixelFormat::Nv12, (4, 4), &tight, &from, &multi).unwrap();
        let (luma, chroma) = (
            multi.map_bytes(CpuAccess::Read).unwrap(),
            multi
                .chroma_dyn()
                .unwrap()
                .map_bytes(CpuAccess::Read)
                .unwrap(),
        );
        let mut back = vec![0; 24];
        copy_plane(
            &luma,
            Plane {
                offset: 0,
                pitch: 8,
            },
            &mut back,
            from[0],
            (4, 4),
        )
        .unwrap();
        copy_plane(
            &chroma,
            Plane {
                offset: 0,
                pitch: 6,
            },
            &mut back,
            from[1],
            (4, 2),
        )
        .unwrap();
        assert_eq!(back, tight);
    }

    #[test]
    fn copy_into_image_writes_yuyv() {
        let tight: Vec<u8> = (0..16).collect();
        let dst = TensorDyn::image(
            4,
            2,
            PixelFormat::Yuyv,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        copy_into_image(
            PixelFormat::Yuyv,
            (4, 2),
            &tight,
            &contiguous(PixelFormat::Yuyv, 8, 2),
            &dst,
        )
        .unwrap();
        assert_eq!(
            &dst.map_bytes(CpuAccess::Read).unwrap()[..16],
            tight.as_slice()
        );
    }

    #[test]
    fn check_source_refuses_wrong_format_dtype_size_and_crop() {
        let config = EncoderConfig::h264(640, 480, PixelFormat::Nv12, 30.0);
        let image = |w, h, format, dtype| {
            TensorDyn::image(
                w,
                h,
                format,
                dtype,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap()
        };
        let crop = |region| FrameOptions {
            crop: Some(region),
            ..FrameOptions::default()
        };
        let none = FrameOptions::default();
        let big = image(1920, 1080, PixelFormat::Nv12, DType::U8);
        let refused = [
            check_source(
                &config,
                &image(320, 240, PixelFormat::Nv12, DType::U8),
                &none,
            ),
            check_source(
                &config,
                &image(640, 480, PixelFormat::Nv12, DType::U16),
                &none,
            ),
            check_source(
                &config,
                &image(640, 480, PixelFormat::Yuyv, DType::U8),
                &none,
            ),
            check_source(&config, &big, &crop(Region::new(0, 0, 1280, 720))),
            check_source(&config, &big, &crop(Region::new(1, 1, 640, 480))),
        ];
        for result in refused {
            assert!(
                matches!(result, Err(CodecError::InvalidInput(_))),
                "{result:?}"
            );
        }
        let ok = check_source(&config, &big, &crop(Region::new(320, 180, 640, 480)));
        assert_eq!(ok.unwrap(), (320, 180));
        let whole = image(640, 480, PixelFormat::Nv12, DType::U8);
        assert_eq!(check_source(&config, &whole, &none).unwrap(), (0, 0));
    }
}
