// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Strided tensor views over borrowed slices.

use crate::error::{KernelError, Result};
use crate::quant::DType;
use std::borrow::Cow;

/// Highest tensor rank a view can describe.
pub const MAX_RANK: usize = 6;

#[derive(Clone, Copy, Debug)]
struct Layout {
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    rank: usize,
    count: usize,
}

fn row_major_strides(shape: &[usize], out: &mut [isize; MAX_RANK]) -> Result<()> {
    let mut acc: usize = 1;
    for (o, &d) in out.iter_mut().zip(shape).rev() {
        *o = isize::try_from(acc).map_err(|_| KernelError::InvalidArgument("shape too large"))?;
        acc = acc
            .checked_mul(d)
            .ok_or(KernelError::InvalidArgument("shape too large"))?;
    }
    Ok(())
}

impl Layout {
    /// Builds a layout and returns it with the number of elements the backing storage must hold.
    fn new(shape: &[usize], strides: Option<&[isize]>) -> Result<(Self, usize)> {
        if shape.len() > MAX_RANK {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "tensor rank above MAX_RANK",
            )));
        }
        let mut l = Layout {
            shape: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            rank: shape.len(),
            count: 0,
        };
        l.shape[..shape.len()].copy_from_slice(shape);
        match strides {
            None => row_major_strides(shape, &mut l.strides)?,
            Some(s) => {
                if s.len() != shape.len() {
                    return Err(KernelError::InvalidArgument("strides rank mismatch"));
                }
                if s.iter().any(|&v| v < 0) {
                    return Err(KernelError::InvalidArgument("negative stride"));
                }
                l.strides[..s.len()].copy_from_slice(s);
            }
        }
        if shape.contains(&0) {
            return Ok((l, 0));
        }
        l.count = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or(KernelError::InvalidArgument("element count overflows"))?;
        let mut last: usize = 0;
        for (&d, &s) in shape.iter().zip(&l.strides) {
            let term = (d - 1)
                .checked_mul(s.unsigned_abs())
                .ok_or(KernelError::InvalidArgument("view extent overflows"))?;
            last = last
                .checked_add(term)
                .ok_or(KernelError::InvalidArgument("view extent overflows"))?;
        }
        let needed = last
            .checked_add(1)
            .ok_or(KernelError::InvalidArgument("view extent overflows"))?;
        Ok((l, needed))
    }

    fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    fn strides(&self) -> &[isize] {
        &self.strides[..self.rank]
    }

    fn len(&self) -> usize {
        self.count
    }

    fn is_dense(&self) -> bool {
        let mut expected: isize = 1;
        for (&d, &s) in self.shape().iter().zip(self.strides()).rev() {
            if d != 1 {
                if s != expected {
                    return false;
                }
                expected = expected.saturating_mul(d as isize);
            }
        }
        true
    }

    fn reshaped(&self, shape: &[usize]) -> Result<Self> {
        if !self.is_dense() {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "reshape needs a contiguous view",
            )));
        }
        let (l, _) = Layout::new(shape, None)?;
        if l.len() != self.len() {
            return Err(KernelError::InvalidArgument(
                "reshape changes the element count",
            ));
        }
        Ok(l)
    }
}

/// Rejects what `slice::from_raw_parts` forbids for `needed` elements of `T` at `ptr`.
fn check_raw<T>(ptr: *const T, needed: usize) -> Result<()> {
    if ptr.is_null() {
        return Err(KernelError::InvalidArgument("null data pointer"));
    }
    if !ptr.is_aligned() {
        return Err(KernelError::InvalidArgument(
            "data pointer is not aligned to the element type",
        ));
    }
    match needed.checked_mul(size_of::<T>()) {
        Some(bytes) if bytes <= isize::MAX as usize => Ok(()),
        _ => Err(KernelError::InvalidArgument(
            "view extent exceeds isize::MAX bytes",
        )),
    }
}

/// The bytes a view reaches, from its first element to one past the last it addresses.
///
/// Bindings that build an input and an output view from raw pointers compare their spans
/// first: the views' slices cover whole spans, so overlapping spans, including interleaved
/// views of one buffer, must be rejected before either view exists.
///
/// ```
/// use edgefirst_kernels::ByteSpan;
///
/// let buf = [0u16; 8];
/// let even = ByteSpan::new(buf.as_ptr(), &[4], Some(&[2]))?;
/// let odd = ByteSpan::new(buf[1..].as_ptr(), &[4], Some(&[2]))?;
/// let tail = ByteSpan::new(buf[7..].as_ptr(), &[1], None)?;
/// assert!(even.overlaps(&odd));
/// assert!(!even.overlaps(&tail));
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteSpan {
    start: usize,
    end: usize,
}

impl ByteSpan {
    /// The span of a view of `T` at `ptr` with `shape` and `strides` (`None` for contiguous
    /// row-major), validated as the view constructors do; empty when the view has no elements.
    pub fn new<T>(ptr: *const T, shape: &[usize], strides: Option<&[isize]>) -> Result<Self> {
        let (_, needed) = Layout::new(shape, strides)?;
        let start = ptr as usize;
        let end = needed
            .checked_mul(size_of::<T>())
            .and_then(|bytes| start.checked_add(bytes))
            .ok_or(KernelError::InvalidArgument("view extent overflows"))?;
        Ok(Self { start, end })
    }

    /// The span's length in bytes.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Whether the two spans share a byte; an empty span overlaps nothing.
    pub fn overlaps(&self, other: &ByteSpan) -> bool {
        !self.is_empty() && !other.is_empty() && self.start < other.end && other.start < self.end
    }
}

/// Immutable strided view (element strides, all non-negative).
#[derive(Clone, Copy)]
pub struct View<'a, T> {
    data: &'a [T],
    layout: Layout,
}

impl<'a, T> View<'a, T> {
    pub fn contiguous(data: &'a [T], shape: &[usize]) -> Result<Self> {
        Self::build(data, shape, None)
    }

    pub fn strided(data: &'a [T], shape: &[usize], strides: &[isize]) -> Result<Self> {
        Self::build(data, shape, Some(strides))
    }

    fn build(data: &'a [T], shape: &[usize], strides: Option<&[isize]>) -> Result<Self> {
        let (layout, needed) = Layout::new(shape, strides)?;
        if needed > data.len() {
            return Err(KernelError::InvalidArgument(
                "data shorter than view extent",
            ));
        }
        Ok(Self { data, layout })
    }

    /// Builds a view over `ptr` without a slice; `None` strides mean contiguous.
    ///
    /// A NULL or misaligned `ptr`, or an extent above `isize::MAX` bytes, is `InvalidArgument`
    /// unless the view has no elements.
    ///
    /// # Safety
    /// `ptr` must be valid for reads of every element the shape and strides address, for the
    /// lifetime `'a`, with no concurrent mutation.
    pub unsafe fn from_raw_parts(
        ptr: *const T,
        shape: &[usize],
        strides: Option<&[isize]>,
    ) -> Result<Self> {
        let (layout, needed) = Layout::new(shape, strides)?;
        let data: &'a [T] = if needed == 0 {
            &[]
        } else {
            check_raw(ptr, needed)?;
            // SAFETY: `check_raw` validated `ptr`; the caller guarantees `needed` readable elements.
            unsafe { core::slice::from_raw_parts(ptr, needed) }
        };
        Ok(Self { data, layout })
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn rank(&self) -> usize {
        self.layout.rank
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as one slice, only when the view is dense row-major.
    pub fn as_contiguous(&self) -> Option<&'a [T]> {
        if self.layout.is_dense() {
            self.data.get(..self.len())
        } else {
            None
        }
    }

    /// The whole backing slice; every index the shape and strides address lies inside it.
    pub(crate) fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<View<'a, T>> {
        Ok(View {
            data: self.data,
            layout: self.layout.reshaped(shape)?,
        })
    }
}

/// Mutable strided view (element strides, all non-negative).
pub struct ViewMut<'a, T> {
    data: &'a mut [T],
    layout: Layout,
}

impl<'a, T> ViewMut<'a, T> {
    pub fn contiguous(data: &'a mut [T], shape: &[usize]) -> Result<Self> {
        Self::build(data, shape, None)
    }

    pub fn strided(data: &'a mut [T], shape: &[usize], strides: &[isize]) -> Result<Self> {
        Self::build(data, shape, Some(strides))
    }

    fn build(data: &'a mut [T], shape: &[usize], strides: Option<&[isize]>) -> Result<Self> {
        let (layout, needed) = Layout::new(shape, strides)?;
        if needed > data.len() {
            return Err(KernelError::InvalidArgument(
                "data shorter than view extent",
            ));
        }
        Ok(Self { data, layout })
    }

    /// Builds a mutable view over `ptr` without a slice; `None` strides mean contiguous.
    ///
    /// A NULL or misaligned `ptr`, or an extent above `isize::MAX` bytes, is `InvalidArgument`
    /// unless the view has no elements.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and writes of every element the shape and strides
    /// address, for the lifetime `'a`, with no other access to them.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        shape: &[usize],
        strides: Option<&[isize]>,
    ) -> Result<Self> {
        let (layout, needed) = Layout::new(shape, strides)?;
        let data: &'a mut [T] = if needed == 0 {
            &mut []
        } else {
            check_raw(ptr, needed)?;
            // SAFETY: `check_raw` validated `ptr`; the caller guarantees `needed` writable elements.
            unsafe { core::slice::from_raw_parts_mut(ptr, needed) }
        };
        Ok(Self { data, layout })
    }

    /// A view of the same elements with a shorter lifetime, leaving `self` usable afterwards.
    pub fn reborrow(&mut self) -> ViewMut<'_, T> {
        ViewMut {
            data: &mut *self.data,
            layout: self.layout,
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn rank(&self) -> usize {
        self.layout.rank
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as one mutable slice, only when the view is dense row-major.
    pub fn as_contiguous_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_dense() {
            let n = self.len();
            self.data.get_mut(..n)
        } else {
            None
        }
    }

    /// The whole backing slice; every index the shape and strides address lies inside it.
    pub(crate) fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    pub fn reshape(&mut self, shape: &[usize]) -> Result<ViewMut<'_, T>> {
        let layout = self.layout.reshaped(shape)?;
        Ok(ViewMut {
            data: &mut *self.data,
            layout,
        })
    }
}

impl<T> core::fmt::Debug for View<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("View")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("len", &self.len())
            .finish()
    }
}

impl<T> core::fmt::Debug for ViewMut<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ViewMut")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("len", &self.len())
            .finish()
    }
}

/// A type-erased immutable tensor.
#[derive(Clone, Copy, Debug)]
pub enum TensorRef<'a> {
    I8(View<'a, i8>),
    U8(View<'a, u8>),
    I16(View<'a, i16>),
    U16(View<'a, u16>),
    F16(View<'a, half::f16>),
    F32(View<'a, f32>),
}

/// A type-erased mutable tensor.
#[derive(Debug)]
pub enum TensorMut<'a> {
    I8(ViewMut<'a, i8>),
    U8(ViewMut<'a, u8>),
    I16(ViewMut<'a, i16>),
    U16(ViewMut<'a, u16>),
    F16(ViewMut<'a, half::f16>),
    F32(ViewMut<'a, f32>),
}

macro_rules! tensor_accessors {
    ($t:ident) => {
        impl $t<'_> {
            pub fn dtype(&self) -> DType {
                match self {
                    Self::I8(_) => DType::I8,
                    Self::U8(_) => DType::U8,
                    Self::I16(_) => DType::I16,
                    Self::U16(_) => DType::U16,
                    Self::F16(_) => DType::F16,
                    Self::F32(_) => DType::F32,
                }
            }

            pub fn shape(&self) -> &[usize] {
                match self {
                    Self::I8(v) => v.shape(),
                    Self::U8(v) => v.shape(),
                    Self::I16(v) => v.shape(),
                    Self::U16(v) => v.shape(),
                    Self::F16(v) => v.shape(),
                    Self::F32(v) => v.shape(),
                }
            }
        }
    };
}
tensor_accessors!(TensorRef);
tensor_accessors!(TensorMut);

impl TensorMut<'_> {
    /// A tensor over the same elements with a shorter lifetime, leaving `self` usable afterwards.
    pub fn reborrow(&mut self) -> TensorMut<'_> {
        match self {
            Self::I8(v) => TensorMut::I8(v.reborrow()),
            Self::U8(v) => TensorMut::U8(v.reborrow()),
            Self::I16(v) => TensorMut::I16(v.reborrow()),
            Self::U16(v) => TensorMut::U16(v.reborrow()),
            Self::F16(v) => TensorMut::F16(v.reborrow()),
            Self::F32(v) => TensorMut::F32(v.reborrow()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelError;

    #[test]
    fn tensor_mut_reborrow_writes_through() {
        let mut data = [0i8; 6];
        let mut t = TensorMut::I8(ViewMut::contiguous(&mut data, &[2, 3]).unwrap());
        for round in 1..=2i8 {
            let TensorMut::I8(mut v) = t.reborrow() else {
                panic!("reborrow changed the dtype");
            };
            assert_eq!(v.shape(), &[2, 3]);
            if let Some(d) = v.as_contiguous_mut() {
                d[0] = round;
            }
        }
        assert_eq!(t.dtype(), DType::I8);
        assert_eq!(data[0], 2);
    }

    #[test]
    fn contiguous_rejects_short_data() {
        let d = [0u8; 6];
        assert!(matches!(
            View::contiguous(&d, &[2, 4]),
            Err(KernelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn strided_rejects_out_of_bounds_extent() {
        let d = [0u8; 10];
        // last index = 1*4 + 3*1 = 7 < 10 ok; 1*8 + 3*1 = 11 >= 10 fails
        assert!(View::strided(&d, &[2, 4], &[4, 1]).is_ok());
        assert!(matches!(
            View::strided(&d, &[2, 4], &[8, 1]),
            Err(KernelError::InvalidArgument(_))
        ));
        assert!(View::strided(&d, &[2, 4], &[6, 1]).is_ok());
    }

    #[test]
    fn strided_rejects_negative_stride() {
        let d = [0u8; 16];
        assert!(matches!(
            View::strided(&d, &[2, 4], &[-4, 1]),
            Err(KernelError::InvalidArgument(_))
        ));
        assert!(matches!(
            View::strided(&d, &[2, 4], &[4]),
            Err(KernelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn strided_rejects_element_count_overflow() {
        let d = [0u8; 4];
        assert!(matches!(
            View::strided(&d, &[usize::MAX, 2], &[0, 0]),
            Err(KernelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rank_above_max_is_unsupported() {
        let d = [0u8; 2];
        assert!(matches!(
            View::contiguous(&d, &[1; 7]),
            Err(KernelError::Unsupported(_))
        ));
        assert!(View::contiguous(&d, &[1; MAX_RANK]).is_ok());
    }

    #[test]
    fn zero_sized_dim_is_empty_and_valid() {
        let d: [u8; 0] = [];
        let v = View::contiguous(&d, &[3, 0, 2]).unwrap();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        let s = View::strided(&d, &[3, 0], &[100, 7]).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn as_contiguous_only_for_dense_row_major() {
        let d: Vec<u8> = (0..12).collect();
        let v = View::contiguous(&d, &[3, 4]).unwrap();
        assert_eq!(v.as_contiguous(), Some(&d[..]));
        let dense = View::strided(&d, &[3, 4], &[4, 1]).unwrap();
        assert_eq!(dense.as_contiguous(), Some(&d[..]));
        let t = View::strided(&d, &[4, 3], &[1, 4]).unwrap();
        assert!(t.as_contiguous().is_none());
        let padded = View::strided(&d, &[2, 4], &[6, 1]).unwrap();
        assert!(padded.as_contiguous().is_none());
        assert_eq!(v.rank(), 2);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(v.strides(), &[4, 1]);
    }

    #[test]
    fn reshape_requires_contiguous() {
        let d: Vec<u8> = (0..12).collect();
        let v = View::contiguous(&d, &[3, 4]).unwrap();
        let r = v.reshape(&[2, 6]).unwrap();
        assert_eq!(r.shape(), &[2, 6]);
        assert!(matches!(
            v.reshape(&[5, 2]),
            Err(KernelError::InvalidArgument(_))
        ));
        let t = View::strided(&d, &[4, 3], &[1, 4]).unwrap();
        assert!(matches!(t.reshape(&[12]), Err(KernelError::Unsupported(_))));
    }

    #[test]
    fn view_mut_mirrors_view() {
        let mut d = [0u8; 6];
        assert!(ViewMut::contiguous(&mut d, &[2, 4]).is_err());
        let mut v = ViewMut::contiguous(&mut d, &[2, 3]).unwrap();
        v.as_contiguous_mut().unwrap()[0] = 9;
        assert_eq!(v.reshape(&[6]).unwrap().shape(), &[6]);
        assert_eq!(d[0], 9);
    }

    #[test]
    fn from_raw_parts_matches_slice_views() {
        let d: Vec<u8> = (0..12).collect();
        // SAFETY: d holds 12 elements, covering shape [3, 4].
        let v = unsafe { View::from_raw_parts(d.as_ptr(), &[3, 4], None) }.unwrap();
        assert_eq!(v.as_contiguous(), Some(&d[..]));
        // SAFETY: last index 3*1 + 2*4 = 11 < 12.
        let s = unsafe { View::from_raw_parts(d.as_ptr(), &[3, 4], Some(&[4, 1])) }.unwrap();
        assert_eq!(s.strides(), &[4, 1]);
    }

    #[test]
    fn from_raw_parts_rejects_misaligned_pointers() {
        let mut d = [0u32; 4];
        // A pointer one byte into an aligned buffer is misaligned for every multi-byte type.
        let p = d.as_mut_ptr().cast::<u8>().wrapping_add(1);
        // SAFETY: the misaligned pointer is rejected before any slice is built.
        unsafe {
            assert_eq!(
                View::from_raw_parts(p.cast::<f32>().cast_const(), &[2], None).unwrap_err(),
                KernelError::InvalidArgument("data pointer is not aligned to the element type")
            );
            assert!(matches!(
                View::from_raw_parts(p.cast::<i16>().cast_const(), &[2], None),
                Err(KernelError::InvalidArgument(_))
            ));
            assert!(matches!(
                ViewMut::from_raw_parts(p.cast::<f32>(), &[2], None),
                Err(KernelError::InvalidArgument(_))
            ));
            assert!(matches!(
                ViewMut::from_raw_parts(p.cast::<u16>(), &[2], None),
                Err(KernelError::InvalidArgument(_))
            ));
            // Bytes have no alignment, and an empty view never reads its pointer.
            assert!(View::from_raw_parts(p.cast_const(), &[2], None).is_ok());
            assert!(View::from_raw_parts(p.cast::<f32>().cast_const(), &[0], None).is_ok());
        }
    }

    #[test]
    fn from_raw_parts_rejects_extents_above_isize_max_bytes() {
        let mut d = [0f32; 1];
        let huge = isize::MAX as usize / 4 + 1;
        // SAFETY: the oversized extent is rejected before any slice is built.
        unsafe {
            assert_eq!(
                View::from_raw_parts(d.as_ptr(), &[huge], None).unwrap_err(),
                KernelError::InvalidArgument("view extent exceeds isize::MAX bytes")
            );
            assert!(matches!(
                ViewMut::from_raw_parts(d.as_mut_ptr(), &[2, 2], Some(&[huge as isize, 1])),
                Err(KernelError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn byte_spans_detect_overlap_including_interleaving() {
        let buf = [0f32; 16];
        let p = buf.as_ptr();
        let span = |off: usize, shape: &[usize], strides: Option<&[isize]>| {
            ByteSpan::new(p.wrapping_add(off), shape, strides).unwrap()
        };
        let whole = span(0, &[4, 4], None);
        assert_eq!(whole.len(), 64);
        let even = span(0, &[4, 2], Some(&[4, 2]));
        let odd = span(1, &[4, 2], Some(&[4, 2]));
        assert!(even.overlaps(&odd) && odd.overlaps(&even));
        let first = span(0, &[8], None);
        let second = span(8, &[8], None);
        assert!(!first.overlaps(&second) && !second.overlaps(&first));
        assert!(whole.overlaps(&second));
        let empty = span(4, &[0, 3], None);
        assert!(empty.is_empty() && !empty.overlaps(&whole) && !whole.overlaps(&empty));
        assert!(matches!(
            ByteSpan::new(p, &[2], Some(&[-1])),
            Err(KernelError::InvalidArgument(_))
        ));
        assert!(matches!(
            ByteSpan::new(p, &[usize::MAX / 2], None),
            Err(KernelError::InvalidArgument(_))
        ));
    }

    #[test]
    fn debug_prints_the_layout_not_the_data() {
        let d = [7u8; 4096];
        let v = View::contiguous(&d, &[64, 64]).unwrap();
        let s = format!("{v:?}");
        assert_eq!(s, "View { shape: [64, 64], strides: [64, 1], len: 4096 }");
        let mut m = [0u8; 6];
        let vm = ViewMut::strided(&mut m, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(
            format!("{vm:?}"),
            "ViewMut { shape: [2, 3], strides: [3, 1], len: 6 }"
        );
    }

    #[test]
    fn tensor_enums_report_dtype_and_shape() {
        let d = [0i8; 4];
        let t = TensorRef::I8(View::contiguous(&d, &[2, 2]).unwrap());
        assert_eq!(t.dtype(), crate::DType::I8);
        assert_eq!(t.shape(), &[2, 2]);
    }
}
