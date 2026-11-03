// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.kernels`: quantized and floating-point compute kernel plans over numpy arrays.

use std::num::NonZeroUsize;

use edgefirst_kernels as k;
use half::f16;
use numpy::{
    dtype as np_dtype, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    PyResult,
};

fn map_err(e: k::KernelError) -> PyErr {
    match e {
        k::KernelError::Unsupported(_) | k::KernelError::ForcedTierUnavailable { .. } => {
            PyNotImplementedError::new_err(e.to_string())
        }
        k::KernelError::InvalidArgument(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn dtype_name(dt: k::DType) -> &'static str {
    match dt {
        k::DType::I8 => "int8",
        k::DType::U8 => "uint8",
        k::DType::I16 => "int16",
        k::DType::U16 => "uint16",
        k::DType::F16 => "float16",
        k::DType::F32 => "float32",
    }
}

fn parse_dtype(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<k::DType> {
    let descr = PyArrayDescr::new(py, obj)
        .map_err(|e| PyValueError::new_err(format!("invalid dtype: {e}")))?;
    let candidates = [
        (np_dtype::<i8>(py), k::DType::I8),
        (np_dtype::<u8>(py), k::DType::U8),
        (np_dtype::<i16>(py), k::DType::I16),
        (np_dtype::<u16>(py), k::DType::U16),
        (np_dtype::<f16>(py), k::DType::F16),
        (np_dtype::<f32>(py), k::DType::F32),
    ];
    candidates
        .into_iter()
        .find(|(d, _)| d.is_equiv_to(&descr))
        .map(|(_, dt)| dt)
        .ok_or_else(|| {
            PyValueError::new_err(
                "dtype must be one of int8, uint8, int16, uint16, float16, float32",
            )
        })
}

fn quant(q: Option<(f32, i32)>) -> k::Quant {
    q.map_or(k::Quant::IDENTITY, |(s, z)| k::Quant::new(s, z))
}

fn box_format(s: &str) -> PyResult<k::BoxFormat> {
    match s {
        "xywh" => Ok(k::BoxFormat::Xywh),
        "xyxy" => Ok(k::BoxFormat::Xyxy),
        _ => Err(PyValueError::new_err("box_format must be 'xywh' or 'xyxy'")),
    }
}

fn levels(l: &[(usize, usize, f32)]) -> Vec<k::Level> {
    l.iter()
        .map(|&(anchors, grid_w, stride)| k::Level {
            anchors,
            grid_w,
            stride,
        })
        .collect()
}

/// Element strides of a numpy array, rejecting layouts a view cannot express.
fn element_strides(
    shape: &[usize],
    byte_strides: &[isize],
    itemsize: usize,
    ptr: usize,
    align: usize,
) -> PyResult<Vec<isize>> {
    if shape.contains(&0) {
        return Ok(vec![0; shape.len()]);
    }
    if !ptr.is_multiple_of(align) {
        return Err(PyValueError::new_err("array data is not aligned"));
    }
    let item = itemsize as isize;
    shape
        .iter()
        .zip(byte_strides)
        .map(|(&d, &s)| {
            if d == 1 {
                Ok(0)
            } else if s < 0 {
                Err(PyValueError::new_err(
                    "negative strides are not supported; pass a copy (numpy.ascontiguousarray)",
                ))
            } else if s % item != 0 {
                Err(PyValueError::new_err(
                    "strides must be a multiple of the element size",
                ))
            } else {
                Ok(s / item)
            }
        })
        .collect()
}

fn dtype_mismatch(role: &str, want: k::DType) -> PyErr {
    PyValueError::new_err(format!(
        "{role} must be a numpy array of dtype {}",
        dtype_name(want)
    ))
}

fn borrow_err(role: &str, e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!(
        "{role} array cannot be borrowed (overlaps the other array or is read-only): {e}"
    ))
}

enum InArray<'py> {
    I8(PyReadonlyArrayDyn<'py, i8>),
    U8(PyReadonlyArrayDyn<'py, u8>),
    I16(PyReadonlyArrayDyn<'py, i16>),
    U16(PyReadonlyArrayDyn<'py, u16>),
    F16(PyReadonlyArrayDyn<'py, f16>),
    F32(PyReadonlyArrayDyn<'py, f32>),
}

enum OutArray<'py> {
    I8(PyReadwriteArrayDyn<'py, i8>),
    U8(PyReadwriteArrayDyn<'py, u8>),
    I16(PyReadwriteArrayDyn<'py, i16>),
    U16(PyReadwriteArrayDyn<'py, u16>),
    F16(PyReadwriteArrayDyn<'py, f16>),
    F32(PyReadwriteArrayDyn<'py, f32>),
}

macro_rules! typed {
    ($arr:expr, $dt:expr, $role:expr, $ty:ty, $variant:ident, $access:ident, $wrap:ident) => {{
        let a = $arr
            .cast::<PyArrayDyn<$ty>>()
            .map_err(|_| dtype_mismatch($role, $dt))?;
        $wrap::$variant(a.$access().map_err(|e| borrow_err($role, e))?)
    }};
}

/// The bytes an array's view reaches, from the same strides the view is built with.
macro_rules! byte_span {
    ($a:expr, $ty:ty) => {{
        let st = element_strides(
            $a.shape(),
            $a.strides(),
            size_of::<$ty>(),
            $a.data() as usize,
            align_of::<$ty>(),
        )?;
        k::ByteSpan::new($a.data() as *const $ty, $a.shape(), Some(&st)).map_err(map_err)?
    }};
}

impl<'py> InArray<'py> {
    fn new(arr: &Bound<'py, PyAny>, dt: k::DType) -> PyResult<Self> {
        const R: &str = "input";
        Ok(match dt {
            k::DType::I8 => typed!(arr, dt, R, i8, I8, try_readonly, InArray),
            k::DType::U8 => typed!(arr, dt, R, u8, U8, try_readonly, InArray),
            k::DType::I16 => typed!(arr, dt, R, i16, I16, try_readonly, InArray),
            k::DType::U16 => typed!(arr, dt, R, u16, U16, try_readonly, InArray),
            k::DType::F16 => typed!(arr, dt, R, f16, F16, try_readonly, InArray),
            k::DType::F32 => typed!(arr, dt, R, f32, F32, try_readonly, InArray),
        })
    }

    fn span(&self) -> PyResult<k::ByteSpan> {
        Ok(match self {
            Self::I8(a) => byte_span!(a, i8),
            Self::U8(a) => byte_span!(a, u8),
            Self::I16(a) => byte_span!(a, i16),
            Self::U16(a) => byte_span!(a, u16),
            Self::F16(a) => byte_span!(a, f16),
            Self::F32(a) => byte_span!(a, f32),
        })
    }

    fn tensor(&self) -> PyResult<k::TensorRef<'_>> {
        macro_rules! view {
            ($a:expr, $ty:ty, $variant:ident) => {{
                let st = element_strides(
                    $a.shape(),
                    $a.strides(),
                    size_of::<$ty>(),
                    $a.data() as usize,
                    align_of::<$ty>(),
                )?;
                // SAFETY: the guard holds a numpy borrow for the view's lifetime, and the
                // extent follows from the array's own shape and strides.
                let v = unsafe {
                    k::View::from_raw_parts($a.data() as *const $ty, $a.shape(), Some(&st))
                }
                .map_err(map_err)?;
                k::TensorRef::$variant(v)
            }};
        }
        Ok(match self {
            Self::I8(a) => view!(a, i8, I8),
            Self::U8(a) => view!(a, u8, U8),
            Self::I16(a) => view!(a, i16, I16),
            Self::U16(a) => view!(a, u16, U16),
            Self::F16(a) => view!(a, f16, F16),
            Self::F32(a) => view!(a, f32, F32),
        })
    }
}

impl<'py> OutArray<'py> {
    fn new(arr: &Bound<'py, PyAny>, dt: k::DType) -> PyResult<Self> {
        const R: &str = "output";
        Ok(match dt {
            k::DType::I8 => typed!(arr, dt, R, i8, I8, try_readwrite, OutArray),
            k::DType::U8 => typed!(arr, dt, R, u8, U8, try_readwrite, OutArray),
            k::DType::I16 => typed!(arr, dt, R, i16, I16, try_readwrite, OutArray),
            k::DType::U16 => typed!(arr, dt, R, u16, U16, try_readwrite, OutArray),
            k::DType::F16 => typed!(arr, dt, R, f16, F16, try_readwrite, OutArray),
            k::DType::F32 => typed!(arr, dt, R, f32, F32, try_readwrite, OutArray),
        })
    }

    fn span(&self) -> PyResult<k::ByteSpan> {
        Ok(match self {
            Self::I8(a) => byte_span!(a, i8),
            Self::U8(a) => byte_span!(a, u8),
            Self::I16(a) => byte_span!(a, i16),
            Self::U16(a) => byte_span!(a, u16),
            Self::F16(a) => byte_span!(a, f16),
            Self::F32(a) => byte_span!(a, f32),
        })
    }

    fn tensor(&mut self) -> PyResult<k::TensorMut<'_>> {
        macro_rules! view {
            ($a:expr, $ty:ty, $variant:ident) => {{
                let st = element_strides(
                    $a.shape(),
                    $a.strides(),
                    size_of::<$ty>(),
                    $a.data() as usize,
                    align_of::<$ty>(),
                )?;
                // SAFETY: the guard holds an exclusive numpy borrow for the view's lifetime,
                // and the extent follows from the array's own shape and strides.
                let v = unsafe { k::ViewMut::from_raw_parts($a.data(), $a.shape(), Some(&st)) }
                    .map_err(map_err)?;
                k::TensorMut::$variant(v)
            }};
        }
        Ok(match self {
            Self::I8(a) => view!(a, i8, I8),
            Self::U8(a) => view!(a, u8, U8),
            Self::I16(a) => view!(a, i16, I16),
            Self::U16(a) => view!(a, u16, U16),
            Self::F16(a) => view!(a, f16, F16),
            Self::F32(a) => view!(a, f32, F32),
        })
    }
}

/// Runs `op` on the arrays with the GIL released.
fn run_op<F>(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    dtypes: (k::DType, k::DType),
    threads: Option<isize>,
    op: F,
) -> PyResult<()>
where
    F: FnOnce(k::TensorRef<'_>, k::TensorMut<'_>, Option<&dyn k::Pool>) -> k::Result<()> + Send,
{
    let pool = match threads {
        None | Some(1) => None,
        Some(n) => {
            let n = usize::try_from(n)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
            Some(k::ScopedThreads::new(n))
        }
    };
    let in_guard = InArray::new(input, dtypes.0)?;
    let mut out_guard = OutArray::new(output, dtypes.1)?;
    // Each view's slice covers its whole span, so overlapping spans must not both become views.
    if in_guard.span()?.overlaps(&out_guard.span()?) {
        return Err(PyValueError::new_err(
            "input and output overlap in memory (interleaved views of one buffer included); \
             pass separate arrays",
        ));
    }
    let in_tensor = in_guard.tensor()?;
    let out_tensor = out_guard.tensor()?;
    py.detach(move || {
        op(
            in_tensor,
            out_tensor,
            pool.as_ref().map(|p| p as &dyn k::Pool),
        )
    })
    .map_err(map_err)
}

macro_rules! softmax_plan {
    ($py:ident, $name:literal, $params:ident, $plan:ident, $dt:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(name = $name, frozen, module = "edgefirst.kernels")]
        pub struct $py {
            plan: k::$plan,
        }

        #[pymethods]
        impl $py {
            #[new]
            #[pyo3(signature = (input_scale, input_zero_point, output_scale, output_zero_point, beta=1.0))]
            fn new(
                input_scale: f32,
                input_zero_point: i32,
                output_scale: f32,
                output_zero_point: i32,
                beta: f32,
            ) -> PyResult<Self> {
                let plan = k::$plan::new(k::$params {
                    input: k::Quant::new(input_scale, input_zero_point),
                    output: k::Quant::new(output_scale, output_zero_point),
                    beta,
                })
                .map_err(map_err)?;
                Ok(Self { plan })
            }

            /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
            #[getter]
            fn tier(&self) -> &'static str {
                self.plan.tier().as_str()
            }

            /// Softmax over the last axis of ``input`` into ``output`` (same shape and dtype).
            #[pyo3(signature = (input, output, *, threads=None))]
            fn run(
                &self,
                py: Python<'_>,
                input: &Bound<'_, PyAny>,
                output: &Bound<'_, PyAny>,
                threads: Option<isize>,
            ) -> PyResult<()> {
                run_op(py, input, output, ($dt, $dt), threads, |i, o, p| {
                    self.plan.run(i, o, p)
                })
            }
        }
    };
}

softmax_plan!(
    PySoftmaxI8Plan,
    "SoftmaxI8Plan",
    SoftmaxI8Params,
    SoftmaxI8Plan,
    k::DType::I8,
    "int8 softmax over the last axis, matching TFLite's int8 ``SOFTMAX``."
);
softmax_plan!(
    PySoftmaxI16Plan,
    "SoftmaxI16Plan",
    SoftmaxI16Params,
    SoftmaxI16Plan,
    k::DType::I16,
    "int16 softmax over the last axis, matching TFLite's int16 ``SOFTMAX``."
);

/// Shared dtype bookkeeping of the plans that take explicit dtypes.
struct Dtypes {
    input: k::DType,
    output: k::DType,
}

impl Dtypes {
    fn parse(
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            input: parse_dtype(py, input)?,
            output: parse_dtype(py, output)?,
        })
    }

    fn pair(&self) -> (k::DType, k::DType) {
        (self.input, self.output)
    }
}

/// Elementwise sigmoid for any input and output dtype.
#[pyclass(name = "SigmoidPlan", frozen, module = "edgefirst.kernels")]
pub struct PySigmoidPlan {
    plan: k::SigmoidPlan,
    dtypes: Dtypes,
}

#[pymethods]
impl PySigmoidPlan {
    #[new]
    fn new(
        py: Python<'_>,
        input_dtype: &Bound<'_, PyAny>,
        input_quant: Option<(f32, i32)>,
        output_dtype: &Bound<'_, PyAny>,
        output_quant: Option<(f32, i32)>,
    ) -> PyResult<Self> {
        let dtypes = Dtypes::parse(py, input_dtype, output_dtype)?;
        let plan = k::SigmoidPlan::new(k::SigmoidParams {
            input_dtype: dtypes.input,
            input: quant(input_quant),
            output_dtype: dtypes.output,
            output: quant(output_quant),
        })
        .map_err(map_err)?;
        Ok(Self { plan, dtypes })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Apply sigmoid elementwise; ``output`` has ``input``'s shape.
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        run_op(py, input, output, self.dtypes.pair(), threads, |i, o, p| {
            self.plan.run(i, o, p)
        })
    }
}

/// Distribution focal loss: ``[anchors, 4 * reg_max]`` logits to ``[anchors, 4]`` distances.
#[pyclass(name = "DflPlan", frozen, module = "edgefirst.kernels")]
pub struct PyDflPlan {
    plan: k::DflPlan,
    dtypes: Dtypes,
}

#[pymethods]
impl PyDflPlan {
    #[new]
    fn new(
        py: Python<'_>,
        input_dtype: &Bound<'_, PyAny>,
        input_quant: Option<(f32, i32)>,
        output_dtype: &Bound<'_, PyAny>,
        output_quant: Option<(f32, i32)>,
        reg_max: usize,
    ) -> PyResult<Self> {
        let dtypes = Dtypes::parse(py, input_dtype, output_dtype)?;
        let plan = k::DflPlan::new(k::DflParams {
            input_dtype: dtypes.input,
            input: quant(input_quant),
            output_dtype: dtypes.output,
            output: quant(output_quant),
            reg_max,
        })
        .map_err(map_err)?;
        Ok(Self { plan, dtypes })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Decode ``[anchors, 4 * reg_max]`` logits into ``[anchors, 4]`` LTRB distances in grid units.
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        run_op(py, input, output, self.dtypes.pair(), threads, |i, o, p| {
            self.plan.run(i, o, p)
        })
    }
}

/// LTRB grid-unit distances to boxes in pixels over a multi-level anchor axis.
#[pyclass(name = "Dist2BboxPlan", frozen, module = "edgefirst.kernels")]
pub struct PyDist2BboxPlan {
    plan: k::Dist2BboxPlan,
    dtypes: Dtypes,
}

#[pymethods]
impl PyDist2BboxPlan {
    #[new]
    #[pyo3(signature = (input_dtype, input_quant, output_dtype, output_quant, levels, box_format="xywh"))]
    fn new(
        py: Python<'_>,
        input_dtype: &Bound<'_, PyAny>,
        input_quant: Option<(f32, i32)>,
        output_dtype: &Bound<'_, PyAny>,
        output_quant: Option<(f32, i32)>,
        levels: Vec<(usize, usize, f32)>,
        box_format: &str,
    ) -> PyResult<Self> {
        let dtypes = Dtypes::parse(py, input_dtype, output_dtype)?;
        let plan = k::Dist2BboxPlan::new(k::Dist2BboxParams {
            input_dtype: dtypes.input,
            input: quant(input_quant),
            output_dtype: dtypes.output,
            output: quant(output_quant),
            levels: self::levels(&levels),
            format: self::box_format(box_format)?,
        })
        .map_err(map_err)?;
        Ok(Self { plan, dtypes })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Decode ``[anchors, 4]`` distances into ``[anchors, 4]`` boxes in pixels.
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        run_op(py, input, output, self.dtypes.pair(), threads, |i, o, p| {
            self.plan.run(i, o, p)
        })
    }
}

/// Fused DFL and dist2bbox: ``[anchors, 4 * reg_max]`` logits to ``[anchors, 4]`` boxes in pixels.
#[pyclass(name = "DflBoxPlan", frozen, module = "edgefirst.kernels")]
pub struct PyDflBoxPlan {
    plan: k::DflBoxPlan,
    dtypes: Dtypes,
}

#[pymethods]
impl PyDflBoxPlan {
    #[new]
    #[pyo3(signature = (input_dtype, input_quant, output_dtype, output_quant, reg_max, levels, box_format="xywh"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        input_dtype: &Bound<'_, PyAny>,
        input_quant: Option<(f32, i32)>,
        output_dtype: &Bound<'_, PyAny>,
        output_quant: Option<(f32, i32)>,
        reg_max: usize,
        levels: Vec<(usize, usize, f32)>,
        box_format: &str,
    ) -> PyResult<Self> {
        let dtypes = Dtypes::parse(py, input_dtype, output_dtype)?;
        let plan = k::DflBoxPlan::new(k::DflBoxParams {
            input_dtype: dtypes.input,
            input: quant(input_quant),
            output_dtype: dtypes.output,
            output: quant(output_quant),
            reg_max,
            levels: self::levels(&levels),
            format: self::box_format(box_format)?,
        })
        .map_err(map_err)?;
        Ok(Self { plan, dtypes })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Decode ``[anchors, 4 * reg_max]`` logits into ``[anchors, 4]`` boxes in pixels.
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        run_op(py, input, output, self.dtypes.pair(), threads, |i, o, p| {
            self.plan.run(i, o, p)
        })
    }
}

/// 256-entry lookup table over int8 or uint8 data.
#[pyclass(name = "LutI8Plan", frozen, module = "edgefirst.kernels")]
pub struct PyLutI8Plan {
    plan: k::LutI8Plan,
    dtype: k::DType,
}

#[pymethods]
impl PyLutI8Plan {
    /// A plan applying ``table`` (256 uint8 entries, indexed by the input's raw byte).
    #[staticmethod]
    fn from_table(
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        table: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Self> {
        let dtype = parse_dtype(py, dtype)?;
        let entries: Vec<u8> = table.as_array().iter().copied().collect();
        let table: [u8; 256] = entries
            .try_into()
            .map_err(|_| PyValueError::new_err("table must have exactly 256 entries"))?;
        let plan = k::LutI8Plan::from_table(dtype, table).map_err(map_err)?;
        Ok(Self { plan, dtype })
    }

    /// The TFLite int8/uint8 ``LOGISTIC`` table for the given quantization.
    #[staticmethod]
    fn sigmoid(
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        input_quant: (f32, i32),
        output_quant: (f32, i32),
    ) -> PyResult<Self> {
        let dtype = parse_dtype(py, dtype)?;
        let plan = k::LutI8Plan::sigmoid(
            dtype,
            k::Quant::new(input_quant.0, input_quant.1),
            k::Quant::new(output_quant.0, output_quant.1),
        )
        .map_err(map_err)?;
        Ok(Self { plan, dtype })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Map ``input`` through the table into ``output`` (same shape and dtype).
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        run_op(
            py,
            input,
            output,
            (self.dtype, self.dtype),
            threads,
            |i, o, p| self.plan.run(i, o, p),
        )
    }
}
