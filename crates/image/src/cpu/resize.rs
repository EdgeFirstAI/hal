// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Flip, FunctionTimer, Rect, ResolvedCrop, Result, Rotation};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMapTrait, TensorTrait};
use ndarray::{ArrayView3, ArrayViewMut3, Axis};
use rayon::iter::IndexedParallelIterator;

use super::{tensor_row_stride, CPUProcessor};

impl CPUProcessor {
    /// Core flip/rotate using ndarray, parameterized by dimensions.
    pub(crate) fn flip_rotate_ndarray_pf(
        src_map: &[u8],
        dst_map: &mut [u8],
        dst_w: usize,
        dst_h: usize,
        dst_c: usize,
        rotation: Rotation,
        flip: Flip,
    ) -> Result<(), crate::Error> {
        let mut dst_view = ArrayViewMut3::from_shape((dst_h, dst_w, dst_c), dst_map)?;
        let mut src_view = match rotation {
            Rotation::None | Rotation::Rotate180 => {
                ArrayView3::from_shape((dst_h, dst_w, dst_c), src_map)?
            }
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
                ArrayView3::from_shape((dst_w, dst_h, dst_c), src_map)?
            }
        };

        match flip {
            Flip::None => {}
            Flip::Vertical => {
                src_view.invert_axis(Axis(0));
            }
            Flip::Horizontal => {
                src_view.invert_axis(Axis(1));
            }
        }

        match rotation {
            Rotation::None => {}
            Rotation::Clockwise90 => {
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(1));
            }
            Rotation::Rotate180 => {
                src_view.invert_axis(Axis(0));
                src_view.invert_axis(Axis(1));
            }
            Rotation::CounterClockwise90 => {
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(0));
            }
        }

        dst_view.assign(&src_view);

        Ok(())
    }

    /// How many extra source pixels the **configured** resize filter can read
    /// beyond each edge of the crop rect along one axis, when `src_extent`
    /// cropped source pixels are resampled to `dst_extent` destination pixels.
    ///
    /// `fast_image_resize` clamps its filter window to the *image* bounds, not
    /// to the crop rect (the FIXME in [`Self::resize_flip_rotate_pf`]), so a
    /// cropped convert legitimately samples a margin of real neighbouring
    /// pixels around the crop. Reproducing that margin — the *halo* — is what
    /// lets the pre-resize intermediate shrink to the crop without changing a
    /// single output byte.
    ///
    /// The arithmetic mirrors `fast_image_resize`'s `precompute_coefficients`:
    /// with `scale = src_extent / dst_extent` the kernel radius is
    /// `support * max(scale, 1)` for `Convolution` (adaptive kernel size) and
    /// `support` for `Interpolation` (fixed kernel), and the window for the
    /// first/last output pixel reaches `ceil(radius - scale / 2)` pixels
    /// outside an integer-aligned crop edge. The filter supports are the
    /// crate's own (`Box` 0.5, `Bilinear`/`Hamming` 1, `CatmullRom`/`Mitchell`
    /// 2, `Gaussian`/`Lanczos3` 3) — read from `self.options.algorithm`, never
    /// assumed by the caller.
    ///
    /// The result is never zero — see `HALO_SLACK` below, which is load-bearing
    /// and not just slack.
    ///
    /// Returns `None` when the reach is not modelled (`SuperSampling`, or a
    /// filter/algorithm added by a future upstream release), so the caller
    /// keeps the full-frame intermediate and its unchanged output.
    pub(super) fn filter_halo(&self, src_extent: usize, dst_extent: usize) -> Option<usize> {
        use fast_image_resize::{FilterType, ResizeAlg};

        /// One spare pixel added to *every* modelled reach, so the halo is
        /// never zero. It earns its place twice over.
        ///
        /// 1. Slack. Growing the extracted rect *beyond* the filter's true
        ///    reach cannot change any output pixel — the extra columns/rows are
        ///    simply never sampled — so a spare pixel costs a few bytes and
        ///    buys immunity to f64 rounding and to small kernel changes in an
        ///    upstream release.
        /// 2. **It keeps `needs_resize` true.** With a zero halo an already
        ///    chroma-aligned crop grows by nothing, so the rebased source rect
        ///    becomes the *whole* intermediate and `needs_resize` in
        ///    [`Self::resize_flip_rotate_pf`] flips to false — handing control
        ///    to `flip_rotate_ndarray_pf`, which assumes tightly-packed
        ///    destination rows and so bypasses the padded-destination destride.
        ///    A crop that is not the whole frame must stay on the resize path
        ///    it took before this optimisation existed. A nonzero halo
        ///    guarantees that: the grown rect can only equal the crop when the
        ///    crop is clamped on all four sides, i.e. when it *is* the whole
        ///    frame — which the caller has already rejected.
        ///
        /// Only `Nearest` has a genuinely zero filter reach, so before this was
        /// applied uniformly it was the only algorithm that could reach 0.
        const HALO_SLACK: usize = 1;

        fn support(filter: FilterType) -> Option<f64> {
            Some(match filter {
                FilterType::Box => 0.5,
                FilterType::Bilinear | FilterType::Hamming => 1.0,
                FilterType::CatmullRom | FilterType::Mitchell => 2.0,
                FilterType::Gaussian | FilterType::Lanczos3 => 3.0,
                FilterType::Custom(f) => f.support(),
                _ => return None,
            })
        }

        let (support, adaptive_kernel) = match self.options.algorithm {
            // Nearest samples exactly one source pixel per output pixel, always
            // inside the crop: zero reach, plus HALO_SLACK below.
            ResizeAlg::Nearest => (0.0, false),
            ResizeAlg::Convolution(f) => (support(f)?, true),
            ResizeAlg::Interpolation(f) => (support(f)?, false),
            _ => return None,
        };
        if src_extent == 0 || dst_extent == 0 {
            return Some(HALO_SLACK);
        }

        let scale = src_extent as f64 / dst_extent as f64;
        let radius = support * if adaptive_kernel { scale.max(1.0) } else { 1.0 };
        let reach = (radius - scale / 2.0).ceil();
        if !reach.is_finite() {
            return None;
        }
        Some(reach.max(0.0) as usize + HALO_SLACK)
    }

    /// Resize/flip/rotate with explicit PixelFormat (used by convert_u8).
    pub(super) fn resize_flip_rotate_pf(
        &mut self,
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        fmt: PixelFormat,
        rotation: Rotation,
        flip: Flip,
        crop: ResolvedCrop,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let channels = fmt.channels();
        let _timer = FunctionTimer::new(format!(
            "ImageProcessor::resize_flip_rotate {}x{} to {}x{} {}",
            src_w, src_h, dst_w, dst_h, fmt,
        ));

        let src_type = match channels {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported source image format".to_string(),
                ));
            }
        };

        let actual_src_stride = tensor_row_stride(src);
        let tight_stride = src_w * channels;
        // `fast_image_resize` requires a tight (no row padding) input buffer,
        // and `flip_rotate_ndarray_pf` uses ndarray shapes that assume tightly
        // packed rows. When the source has a larger stride (e.g. from codec
        // decode into a pre-allocated oversized tensor), copy the visible
        // pixels row-by-row into a tight scratch before proceeding.
        // The copy is gated on `actual_src_stride != tight_stride` so it is
        // a no-op for all already-tight sources.
        let mut src_map = src.map()?;
        // When the source is padded (stride != tight), de-stride it into the
        // processor-owned scratch (reused across calls — no per-call alloc).
        // `destrided` records whether we populated the scratch this call.
        let destrided = actual_src_stride != tight_stride;
        if destrided {
            let need = src_h * tight_stride;
            self.resize_destride_scratch.clear();
            self.resize_destride_scratch.resize(need, 0u8);
            let src_slice = src_map.as_slice();
            for row in 0..src_h {
                let src_row =
                    &src_slice[row * actual_src_stride..row * actual_src_stride + tight_stride];
                self.resize_destride_scratch[row * tight_stride..(row + 1) * tight_stride]
                    .copy_from_slice(src_row);
            }
        }
        let src_for_proc: &mut [u8] = if destrided {
            &mut self.resize_destride_scratch[..src_h * tight_stride]
        } else {
            src_map.as_mut_slice()
        };
        let mut dst_map = dst.map()?;

        // FIXME: fast_image_resize does not clamp its filter kernel at crop
        // boundaries — bilinear/bicubic taps can sample 1-2 pixels beyond the
        // specified crop rect, causing colour bleed from adjacent regions.
        // A proper fix would inset the crop by the filter radius or use a
        // library that supports boundary clamping.
        let options = if let Some(crop) = crop.src_rect {
            self.options.crop(
                crop.left as f64,
                crop.top as f64,
                crop.width as f64,
                crop.height as f64,
            )
        } else {
            self.options
        };

        let mut dst_rect = crop.dst_rect.unwrap_or(Rect {
            left: 0,
            top: 0,
            width: dst_w,
            height: dst_h,
        });

        // adjust crop box for rotation/flip
        Self::adjust_dest_rect_for_rotate_flip_dims(&mut dst_rect, dst_w, dst_h, rotation, flip);

        let dst_rs = tensor_row_stride(dst);

        let needs_resize = src_w != dst_w
            || src_h != dst_h
            || crop.src_rect.is_some_and(|c| {
                c != Rect {
                    left: 0,
                    top: 0,
                    width: src_w,
                    height: src_h,
                }
            })
            || crop.dst_rect.is_some_and(|c| {
                c != Rect {
                    left: 0,
                    top: 0,
                    width: dst_w,
                    height: dst_h,
                }
            });

        if needs_resize {
            let src_view = fast_image_resize::images::Image::from_slice_u8(
                src_w as u32,
                src_h as u32,
                src_for_proc,
                src_type,
            )?;

            match (rotation, flip) {
                (Rotation::None, Flip::None) => {
                    let tight_dst_stride = dst_w * channels;
                    // `fast_image_resize` requires a tightly-packed output
                    // buffer — it addresses rows by `row * width * bpp`, not
                    // by the tensor's real row stride. Passing a padded
                    // `dst_map` straight through (as the zero-copy path
                    // below does) silently smears every row after the first
                    // at the wrong offset. When the destination is padded
                    // (a DMA pitch-aligned tensor, or a `view()` narrower
                    // than its parent's stride), resize into a tight
                    // scratch — seeded with the existing dst content so
                    // pixels outside `dst_rect` (e.g. letterbox borders)
                    // survive unchanged — then copy back row-by-row at the
                    // real stride. Tight destinations keep the existing
                    // zero-copy path unchanged.
                    if dst_rs != tight_dst_stride {
                        let need = dst_h * tight_dst_stride;
                        self.resize_dst_destride_scratch.clear();
                        self.resize_dst_destride_scratch.resize(need, 0u8);
                        for row in 0..dst_h {
                            let dst_row = &dst_map[row * dst_rs..row * dst_rs + tight_dst_stride];
                            self.resize_dst_destride_scratch
                                [row * tight_dst_stride..(row + 1) * tight_dst_stride]
                                .copy_from_slice(dst_row);
                        }

                        let mut dst_view = fast_image_resize::images::Image::from_slice_u8(
                            dst_w as u32,
                            dst_h as u32,
                            &mut self.resize_dst_destride_scratch,
                            src_type,
                        )?;

                        let mut dst_view = fast_image_resize::images::CroppedImageMut::new(
                            &mut dst_view,
                            dst_rect.left as u32,
                            dst_rect.top as u32,
                            dst_rect.width as u32,
                            dst_rect.height as u32,
                        )?;

                        self.resizer.resize(&src_view, &mut dst_view, &options)?;

                        for row in 0..dst_h {
                            let scratch_row = &self.resize_dst_destride_scratch
                                [row * tight_dst_stride..(row + 1) * tight_dst_stride];
                            dst_map[row * dst_rs..row * dst_rs + tight_dst_stride]
                                .copy_from_slice(scratch_row);
                        }
                    } else {
                        let mut dst_view = fast_image_resize::images::Image::from_slice_u8(
                            dst_w as u32,
                            dst_h as u32,
                            &mut dst_map,
                            src_type,
                        )?;

                        let mut dst_view = fast_image_resize::images::CroppedImageMut::new(
                            &mut dst_view,
                            dst_rect.left as u32,
                            dst_rect.top as u32,
                            dst_rect.width as u32,
                            dst_rect.height as u32,
                        )?;

                        self.resizer.resize(&src_view, &mut dst_view, &options)?;
                    }
                }
                (Rotation::Clockwise90, _) | (Rotation::CounterClockwise90, _) => {
                    let mut tmp = vec![0; dst_rs * dst_h];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst_h as u32,
                        dst_w as u32,
                        &mut tmp,
                        src_type,
                    )?;

                    let mut tmp_view = fast_image_resize::images::CroppedImageMut::new(
                        &mut tmp_view,
                        dst_rect.left as u32,
                        dst_rect.top as u32,
                        dst_rect.width as u32,
                        dst_rect.height as u32,
                    )?;

                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    Self::flip_rotate_ndarray_pf(
                        &tmp,
                        &mut dst_map,
                        dst_w,
                        dst_h,
                        channels,
                        rotation,
                        flip,
                    )?;
                }
                (Rotation::None, _) | (Rotation::Rotate180, _) => {
                    let mut tmp = vec![0; dst_rs * dst_h];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst_w as u32,
                        dst_h as u32,
                        &mut tmp,
                        src_type,
                    )?;

                    let mut tmp_view = fast_image_resize::images::CroppedImageMut::new(
                        &mut tmp_view,
                        dst_rect.left as u32,
                        dst_rect.top as u32,
                        dst_rect.width as u32,
                        dst_rect.height as u32,
                    )?;

                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    Self::flip_rotate_ndarray_pf(
                        &tmp,
                        &mut dst_map,
                        dst_w,
                        dst_h,
                        channels,
                        rotation,
                        flip,
                    )?;
                }
            }
        } else {
            Self::flip_rotate_ndarray_pf(
                src_for_proc,
                &mut dst_map,
                dst_w,
                dst_h,
                channels,
                rotation,
                flip,
            )?;
        }
        Ok(())
    }

    fn adjust_dest_rect_for_rotate_flip_dims(
        crop: &mut Rect,
        dst_w: usize,
        dst_h: usize,
        rot: Rotation,
        flip: Flip,
    ) {
        match rot {
            Rotation::None => {}
            Rotation::Clockwise90 => {
                *crop = Rect {
                    left: crop.top,
                    top: dst_w - crop.left - crop.width,
                    width: crop.height,
                    height: crop.width,
                }
            }
            Rotation::Rotate180 => {
                *crop = Rect {
                    left: dst_w - crop.left - crop.width,
                    top: dst_h - crop.top - crop.height,
                    width: crop.width,
                    height: crop.height,
                }
            }
            Rotation::CounterClockwise90 => {
                *crop = Rect {
                    left: dst_h - crop.top - crop.height,
                    top: crop.left,
                    width: crop.height,
                    height: crop.width,
                }
            }
        }

        match flip {
            Flip::None => {}
            Flip::Vertical => crop.top = dst_h - crop.top - crop.height,
            Flip::Horizontal => crop.left = dst_w - crop.left - crop.width,
        }
    }

    /// Fill the letterbox border of one image surface with `pix`.
    ///
    /// Rows advance by `dst_stride`, and each row is clipped to its own
    /// `dst_width * N` pixel bytes: for an allocation-padded destination the
    /// remainder is dead padding, and for a `Tensor::view()` destination it is
    /// the parent image's neighbouring columns — the border must not spill into
    /// either. On a tightly-packed destination (stride == row bytes) this fills
    /// exactly the pixels the previous flat-index form did.
    pub(super) fn fill_image_outside_crop_<const N: usize>(
        (dst, dst_width, dst_height, dst_stride): (&mut [u8], usize, usize, usize),
        pix: [u8; N],
        crop: Rect,
    ) -> Result<()> {
        use rayon::{iter::ParallelIterator, prelude::ParallelSliceMut};

        let row_bytes = dst_width * N;
        super::guard_plane(
            dst.len(),
            dst_stride,
            dst_height,
            row_bytes,
            "letterbox fill",
        )?;

        let left = crop.left.min(dst_width);
        let right = (crop.left + crop.width).min(dst_width);
        let bottom = crop.top + crop.height;

        dst.par_chunks_mut(dst_stride)
            .take(dst_height)
            .enumerate()
            .for_each(|(y, row)| {
                let px = row[..row_bytes].as_chunks_mut::<N>().0;
                let border = if y < crop.top || y >= bottom {
                    // Whole row is above/below the content.
                    &mut px[..]
                } else {
                    for p in &mut px[..left] {
                        *p = pix;
                    }
                    &mut px[right..]
                };
                for p in border {
                    *p = pix;
                }
            });

        Ok(())
    }

    /// Planar variant: each of the `N` planes is `dst_height` rows of
    /// `dst_stride` bytes, and gets the border of its own channel value.
    pub(super) fn fill_image_outside_crop_planar<const N: usize>(
        (dst, dst_width, dst_height, dst_stride): (&mut [u8], usize, usize, usize),
        pix: [u8; N],
        crop: Rect,
    ) -> Result<()> {
        use rayon::{iter::ParallelIterator, prelude::ParallelSliceMut};

        let plane = dst_stride.checked_mul(dst_height).ok_or_else(|| {
            Error::InvalidShape(format!(
                "planar fill plane size overflow (stride={dst_stride}, h={dst_height})"
            ))
        })?;
        // `par_chunks_exact_mut` silently drops a short tail, which zipped with
        // `pix` would skip planes rather than fail, so require all `N` up front.
        let need = plane.saturating_mul(N);
        if dst.len() < need {
            return Err(Error::InvalidShape(format!(
                "planar fill dst {} bytes < {N} planes of {plane} (stride={dst_stride}, \
                 h={dst_height})",
                dst.len()
            )));
        }

        dst.par_chunks_exact_mut(plane)
            .zip(pix)
            .try_for_each(|(s, p)| {
                Self::fill_image_outside_crop_::<1>(
                    (s, dst_width, dst_height, dst_stride),
                    [p],
                    crop,
                )
            })
    }

    pub(super) fn fill_image_outside_crop_yuv_semiplanar(
        (dst, dst_width, dst_height, dst_stride): (&mut [u8], usize, usize, usize),
        y: u8,
        uv: [u8; 2],
        mut crop: Rect,
    ) -> Result<()> {
        // Validate the buffer holds the luma plane before splitting so a
        // caller-supplied (untrusted) dst cannot panic the `split_at_mut`.
        let luma = dst_stride.checked_mul(dst_height).ok_or_else(|| {
            Error::InvalidShape(format!(
                "semiplanar fill luma size overflow (stride={dst_stride}, h={dst_height})"
            ))
        })?;
        if dst.len() < luma {
            return Err(Error::InvalidShape(format!(
                "semiplanar fill dst {} bytes < luma plane {luma} (stride={dst_stride}, \
                 h={dst_height})",
                dst.len()
            )));
        }
        let (y_plane, uv_plane) = dst.split_at_mut(luma);
        Self::fill_image_outside_crop_::<1>(
            (y_plane, dst_width, dst_height, dst_stride),
            [y],
            crop,
        )?;
        crop.left /= 2;
        crop.width /= 2;
        Self::fill_image_outside_crop_::<2>(
            (uv_plane, dst_width / 2, dst_height, dst_stride),
            uv,
            crop,
        )?;
        Ok(())
    }
}
