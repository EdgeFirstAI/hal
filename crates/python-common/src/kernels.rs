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
    fn new(arr: &Bound<'py, PyAny>, dt: k::DType, role: &str) -> PyResult<Self> {
        Ok(match dt {
            k::DType::I8 => typed!(arr, dt, role, i8, I8, try_readonly, InArray),
            k::DType::U8 => typed!(arr, dt, role, u8, U8, try_readonly, InArray),
            k::DType::I16 => typed!(arr, dt, role, i16, I16, try_readonly, InArray),
            k::DType::U16 => typed!(arr, dt, role, u16, U16, try_readonly, InArray),
            k::DType::F16 => typed!(arr, dt, role, f16, F16, try_readonly, InArray),
            k::DType::F32 => typed!(arr, dt, role, f32, F32, try_readonly, InArray),
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

/// The pool for a `threads` argument: none for `None` or 1.
fn pool_for(threads: Option<isize>) -> PyResult<Option<k::ScopedThreads>> {
    match threads {
        None | Some(1) => Ok(None),
        Some(n) => {
            let n = usize::try_from(n)
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or_else(|| PyValueError::new_err("threads must be at least 1"))?;
            Ok(Some(k::ScopedThreads::new(n)))
        }
    }
}

/// Runs `op` on the arrays with the GIL released: `inputs` are `(array, dtype, role)`.
///
/// The output must not overlap any input; inputs may overlap each other.
fn run_many<F>(
    py: Python<'_>,
    inputs: &[(&Bound<'_, PyAny>, k::DType, &str)],
    output: &Bound<'_, PyAny>,
    out_dtype: k::DType,
    threads: Option<isize>,
    op: F,
) -> PyResult<()>
where
    F: FnOnce(Vec<k::TensorRef<'_>>, k::TensorMut<'_>, Option<&dyn k::Pool>) -> k::Result<()>
        + Send,
{
    let pool = pool_for(threads)?;
    let in_guards = inputs
        .iter()
        .map(|&(a, dt, role)| InArray::new(a, dt, role))
        .collect::<PyResult<Vec<_>>>()?;
    let mut out_guard = OutArray::new(output, out_dtype)?;
    // Each view's slice covers its whole span, so overlapping spans must not both become views.
    let out_span = out_guard.span()?;
    for (g, &(_, _, role)) in in_guards.iter().zip(inputs) {
        if g.span()?.overlaps(&out_span) {
            return Err(PyValueError::new_err(format!(
                "{role} and output overlap in memory (interleaved views of one buffer \
                 included); pass separate arrays"
            )));
        }
    }
    let in_tensors = in_guards
        .iter()
        .map(InArray::tensor)
        .collect::<PyResult<Vec<_>>>()?;
    let out_tensor = out_guard.tensor()?;
    py.detach(move || {
        op(
            in_tensors,
            out_tensor,
            pool.as_ref().map(|p| p as &dyn k::Pool),
        )
    })
    .map_err(map_err)
}

/// Runs a one-input `op` on the arrays with the GIL released.
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
    run_many(
        py,
        &[(input, dtypes.0, "input")],
        output,
        dtypes.1,
        threads,
        |mut i, o, p| match i.pop() {
            Some(i) => op(i, o, p),
            None => Err(k::KernelError::InvalidArgument("missing input")),
        },
    )
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

fn tflite_kernels(s: &str) -> PyResult<k::TfliteKernels> {
    match s {
        "neon_ruy" => Ok(k::TfliteKernels::NeonRuy),
        "neon_gemmlowp" => Ok(k::TfliteKernels::NeonGemmlowp),
        "ruy" => Ok(k::TfliteKernels::Ruy),
        "reference" => Ok(k::TfliteKernels::Reference),
        _ => Err(PyValueError::new_err(
            "kernels must be 'neon_ruy', 'neon_gemmlowp', 'ruy' or 'reference'",
        )),
    }
}

fn activation(s: &str) -> PyResult<k::Activation> {
    match s {
        "none" => Ok(k::Activation::None),
        "relu" => Ok(k::Activation::Relu),
        "relu_n1_to_1" => Ok(k::Activation::ReluN1To1),
        "relu6" => Ok(k::Activation::Relu6),
        _ => Err(PyValueError::new_err(
            "activation must be 'none', 'relu', 'relu_n1_to_1' or 'relu6'",
        )),
    }
}

fn binary_op(s: &str) -> PyResult<k::BinaryOp> {
    match s {
        "add" => Ok(k::BinaryOp::Add),
        "sub" => Ok(k::BinaryOp::Sub),
        "mul" => Ok(k::BinaryOp::Mul),
        _ => Err(PyValueError::new_err("op must be 'add', 'sub' or 'mul'")),
    }
}

fn q(t: (f32, i32)) -> k::Quant {
    k::Quant::new(t.0, t.1)
}

/// A 256-entry table from a numpy uint8 array or a sequence of ints.
fn table_256(obj: &Bound<'_, PyAny>) -> PyResult<[u8; 256]> {
    let entries: Vec<u8> = match obj.cast::<numpy::PyArray1<u8>>() {
        Ok(a) => a.readonly().as_array().iter().copied().collect(),
        Err(_) => obj.extract()?,
    };
    entries
        .try_into()
        .map_err(|_| PyValueError::new_err("table must have exactly 256 entries"))
}

/// int8 ``ADD``, ``SUB`` or ``MUL``, matching TFLite's int8 kernels bit for bit.
#[pyclass(name = "BinaryI8Plan", frozen, module = "edgefirst.kernels")]
pub struct PyBinaryI8Plan {
    plan: k::BinaryI8Plan,
}

#[pymethods]
impl PyBinaryI8Plan {
    #[new]
    #[pyo3(signature = (op, lhs_quant, rhs_quant, output_quant, activation="none", kernels="neon_ruy"))]
    fn new(
        op: &str,
        lhs_quant: (f32, i32),
        rhs_quant: (f32, i32),
        output_quant: (f32, i32),
        activation: &str,
        kernels: &str,
    ) -> PyResult<Self> {
        let plan = k::BinaryI8Plan::new(k::BinaryI8Params {
            op: binary_op(op)?,
            lhs: q(lhs_quant),
            rhs: q(rhs_quant),
            output: q(output_quant),
            activation: self::activation(activation)?,
            kernels: tflite_kernels(kernels)?,
        })
        .map_err(map_err)?;
        Ok(Self { plan })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// The output for every ``lhs`` byte with ``rhs`` fixed, for a call over ``length`` elements.
    fn table_for_rhs<'py>(
        &self,
        py: Python<'py>,
        rhs: i8,
        length: usize,
    ) -> Option<Bound<'py, numpy::PyArray1<u8>>> {
        self.plan
            .table_for_rhs(rhs, length)
            .map(|t| numpy::PyArray1::from_slice(py, &t))
    }

    /// ``output = lhs op rhs`` elementwise; all int8 of one shape (a broadcast is a zero stride).
    #[pyo3(signature = (lhs, rhs, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        lhs: &Bound<'_, PyAny>,
        rhs: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        const I8: k::DType = k::DType::I8;
        run_many(
            py,
            &[(lhs, I8, "lhs"), (rhs, I8, "rhs")],
            output,
            I8,
            threads,
            |mut i, o, p| match (i.pop(), i.pop()) {
                (Some(r), Some(l)) => self.plan.run(l, r, o, p),
                _ => Err(k::KernelError::InvalidArgument("missing input")),
            },
        )
    }
}

/// int8 matrix multiplication over the last two axes with TFLite's requantization.
#[pyclass(name = "MatMulI8Plan", frozen, module = "edgefirst.kernels")]
pub struct PyMatMulI8Plan {
    plan: k::MatMulI8Plan,
    params: k::MatMulI8Params,
}

impl PyMatMulI8Plan {
    fn build(params: k::MatMulI8Params) -> PyResult<Self> {
        let plan = k::MatMulI8Plan::new(params.clone()).map_err(map_err)?;
        Ok(Self { plan, params })
    }
}

#[pymethods]
impl PyMatMulI8Plan {
    #[new]
    #[pyo3(signature = (lhs_zero_point, rhs_zero_point, output_quant, requant, bias=None, activation="none", kernels="neon_ruy"))]
    fn new(
        lhs_zero_point: i32,
        rhs_zero_point: i32,
        output_quant: (f32, i32),
        requant: Vec<(i32, i32)>,
        bias: Option<Vec<i32>>,
        activation: &str,
        kernels: &str,
    ) -> PyResult<Self> {
        Self::build(k::MatMulI8Params {
            lhs_zero_point,
            rhs_zero_point,
            output: q(output_quant),
            requant: requant
                .into_iter()
                .map(|(multiplier, shift)| k::Requant { multiplier, shift })
                .collect(),
            bias: bias.unwrap_or_default(),
            activation: self::activation(activation)?,
            kernels: tflite_kernels(kernels)?,
        })
    }

    /// TFLite's int8 ``BATCH_MATMUL``; ``adj_x`` / ``adj_y`` are transposed views.
    #[staticmethod]
    #[pyo3(signature = (lhs_quant, rhs_quant, output_quant, kernels="neon_ruy"))]
    fn tflite_batch_matmul(
        lhs_quant: (f32, i32),
        rhs_quant: (f32, i32),
        output_quant: (f32, i32),
        kernels: &str,
    ) -> PyResult<Self> {
        Self::build(
            k::MatMulI8Params::tflite_batch_matmul(
                q(lhs_quant),
                q(rhs_quant),
                q(output_quant),
                tflite_kernels(kernels)?,
            )
            .map_err(map_err)?,
        )
    }

    /// A TFLite int8 1x1 ``CONV_2D``: ``lhs`` ``[pixels, in]``, ``rhs`` ``[in, out]``.
    #[staticmethod]
    #[pyo3(signature = (input_quant, filter_scales, output_quant, bias=None, activation="none", kernels="neon_ruy"))]
    fn tflite_conv(
        input_quant: (f32, i32),
        filter_scales: Vec<f32>,
        output_quant: (f32, i32),
        bias: Option<Vec<i32>>,
        activation: &str,
        kernels: &str,
    ) -> PyResult<Self> {
        Self::build(
            k::MatMulI8Params::tflite_conv(
                q(input_quant),
                &filter_scales,
                q(output_quant),
                bias.unwrap_or_default(),
                self::activation(activation)?,
                tflite_kernels(kernels)?,
            )
            .map_err(map_err)?,
        )
    }

    /// Raises ``NotImplementedError`` when a reduction axis of ``k`` elements is too long.
    #[staticmethod]
    fn check_depth(k: usize) -> PyResult<()> {
        k::MatMulI8Plan::check_depth(k).map_err(map_err)
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// ``output = lhs @ rhs`` over the last two axes; all int8 of one rank.
    #[pyo3(signature = (lhs, rhs, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        lhs: &Bound<'_, PyAny>,
        rhs: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        const I8: k::DType = k::DType::I8;
        run_many(
            py,
            &[(lhs, I8, "lhs"), (rhs, I8, "rhs")],
            output,
            I8,
            threads,
            |mut i, o, p| match (i.pop(), i.pop()) {
                (Some(r), Some(l)) => self.plan.run(l, r, o, p),
                _ => Err(k::KernelError::InvalidArgument("missing input")),
            },
        )
    }
}

/// int8 softmax over the last axis fused with a product by one weight column (a DFL head).
#[pyclass(name = "SoftmaxDotI8Plan", frozen, module = "edgefirst.kernels")]
pub struct PySoftmaxDotI8Plan {
    plan: k::SoftmaxDotI8Plan,
}

#[pymethods]
impl PySoftmaxDotI8Plan {
    #[new]
    #[pyo3(signature = (input_quant, weights, projection, beta=1.0, probs_quant=(1.0 / 256.0, -128)))]
    fn new(
        input_quant: (f32, i32),
        weights: &Bound<'_, PyAny>,
        projection: PyRef<'_, PyMatMulI8Plan>,
        beta: f32,
        probs_quant: (f32, i32),
    ) -> PyResult<Self> {
        let weights: Vec<i8> = match weights.cast::<numpy::PyArray1<i8>>() {
            Ok(a) => a.readonly().as_array().iter().copied().collect(),
            Err(_) => weights.extract()?,
        };
        let plan = k::SoftmaxDotI8Plan::new(k::SoftmaxDotI8Params {
            softmax: k::SoftmaxI8Params {
                input: q(input_quant),
                output: q(probs_quant),
                beta,
            },
            weights,
            projection: projection.params.clone(),
        })
        .map_err(map_err)?;
        Ok(Self { plan })
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// ``[..., n]`` int8 logits to ``[..., 1]`` int8 outputs.
    #[pyo3(signature = (input, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        let dt = k::DType::I8;
        run_op(py, input, output, (dt, dt), threads, |i, o, p| {
            self.plan.run(i, o, p)
        })
    }
}

/// `(heads, queries, tokens, dk, dv)` of one attention call.
type AttentionShape = (usize, usize, usize, usize, usize);

/// Fused int8 attention ``softmax(q' @ k.T) @ v`` per head, matching TFLite's int8 chain.
///
/// The packed scratch the kernels need is cached per plan and rebuilt when the shapes change; a
/// call that finds it in use by another thread packs into a scratch of its own.
#[pyclass(name = "AttentionI8Plan", frozen, module = "edgefirst.kernels")]
pub struct PyAttentionI8Plan {
    plan: k::AttentionI8Plan,
    scratch: std::sync::Mutex<Option<(AttentionShape, k::AttentionScratch)>>,
}

#[pymethods]
impl PyAttentionI8Plan {
    #[new]
    #[pyo3(signature = (q_quant, k_quant, v_quant, scores_quant, probs_quant, output_quant, beta=1.0, q_mul=None, kernels="neon_ruy"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        q_quant: (f32, i32),
        k_quant: (f32, i32),
        v_quant: (f32, i32),
        scores_quant: (f32, i32),
        probs_quant: (f32, i32),
        output_quant: (f32, i32),
        beta: f32,
        q_mul: Option<(Bound<'_, PyAny>, (f32, i32))>,
        kernels: &str,
    ) -> PyResult<Self> {
        let q_mul = match q_mul {
            Some((table, output)) => Some(k::AttentionQMul {
                table: table_256(&table)?,
                output: q(output),
            }),
            None => None,
        };
        let plan = k::AttentionI8Plan::new(k::AttentionI8Params {
            q: q(q_quant),
            q_mul,
            k: q(k_quant),
            v: q(v_quant),
            scores: q(scores_quant),
            beta,
            probs: q(probs_quant),
            output: q(output_quant),
            kernels: tflite_kernels(kernels)?,
        })
        .map_err(map_err)?;
        Ok(Self {
            plan,
            scratch: std::sync::Mutex::new(None),
        })
    }

    /// Raises ``NotImplementedError`` when the kernels cannot take these tokens and head dims.
    #[staticmethod]
    fn check_shape(tokens: usize, dk: usize, dv: usize) -> PyResult<()> {
        k::AttentionI8Plan::check_shape(tokens, dk, dv).map_err(map_err)
    }

    /// Kernel tier this plan runs: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``.
    #[getter]
    fn tier(&self) -> &'static str {
        self.plan.tier().as_str()
    }

    /// Attention of ``q`` ``[heads, queries, dk]`` over ``k`` ``[heads, tokens, dk]`` and
    /// ``v`` ``[heads, tokens, dv]`` into ``output`` ``[heads, queries, dv]``, all int8.
    #[pyo3(signature = (q, k, v, output, *, threads=None))]
    fn run(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        k: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
        threads: Option<isize>,
    ) -> PyResult<()> {
        const I8: k::DType = k::DType::I8;
        run_many(
            py,
            &[(q, I8, "q"), (k, I8, "k"), (v, I8, "v")],
            output,
            I8,
            threads,
            |i, o, p| {
                let [q, k, v]: [k::TensorRef<'_>; 3] = i
                    .try_into()
                    .map_err(|_| k::KernelError::InvalidArgument("missing input"))?;
                let (Ok(&[h, n, dk]), Ok(&[_, m, _]), Ok(&[_, _, dv])) = (
                    <&[usize; 3]>::try_from(q.shape()),
                    <&[usize; 3]>::try_from(k.shape()),
                    <&[usize; 3]>::try_from(v.shape()),
                ) else {
                    return Err(k::KernelError::InvalidArgument(
                        "attention_i8 views need rank 3: [heads, tokens, dim]",
                    ));
                };
                let shape = (h, n, m, dk, dv);
                let run = |s: &mut k::AttentionScratch| self.plan.run(q, k, v, o, s, p);
                match self.scratch.try_lock() {
                    Ok(mut cached) => {
                        if cached.as_ref().is_none_or(|(s, _)| *s != shape) {
                            *cached = None;
                            *cached = Some((shape, self.plan.scratch(h, n, m, dk, dv)?));
                        }
                        match cached.as_mut() {
                            Some((_, s)) => run(s),
                            None => Err(k::KernelError::InvalidArgument("no scratch")),
                        }
                    }
                    Err(_) => run(&mut self.plan.scratch(h, n, m, dk, dv)?),
                }
            },
        )
    }
}
