// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Model configuration discovery (`edgefirst_decoder::model_config`).
//! Registered on `edgefirst.decoder`.

use std::path::PathBuf;

use edgefirst_decoder::model_config::{
    self as mc, ConfigField, ConfigSource, ModelConfigError, ModelFamily, ModelOverrides,
    ModelTask, Provenance, ResolvedModelConfig,
};
use edgefirst_tensor::PreprocessSpec;
use pyo3::exceptions::{PyIsADirectoryError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::decoder::PyDecodeMode;

/// `msg` as an instance of the `edgefirst.decoder` exception class `name`.
fn decoder_error(py: Python<'_>, name: &str, msg: String) -> PyErr {
    let class = py.import("edgefirst.decoder").and_then(|m| m.getattr(name));
    match class.and_then(|c| c.call1((msg.clone(),))) {
        Ok(e) => PyErr::from_value(e),
        Err(_) => PyValueError::new_err(msg),
    }
}

/// A `LabelCountMismatchError` carrying the model's class count and the
/// override's.
fn label_count_error(py: Python<'_>, msg: String, expected: usize, found: usize) -> PyErr {
    let class = py
        .import("edgefirst.decoder")
        .and_then(|m| m.getattr("LabelCountMismatchError"));
    match class.and_then(|c| c.call1((msg.clone(), expected, found))) {
        Ok(e) => PyErr::from_value(e),
        Err(_) => PyValueError::new_err(msg),
    }
}

/// Maps a discovery error to the Python exception for its kind: `OSError`
/// (with errno and filename) for I/O, else a `ModelConfigError` subclass
/// (all `ValueError`s).
fn config_err(e: ModelConfigError) -> PyErr {
    let class = match &e {
        ModelConfigError::Io { path, source } => {
            let filename = path.display().to_string();
            return match source.raw_os_error() {
                Some(errno) => {
                    let msg = source.to_string();
                    let suffix = format!(" (os error {errno})");
                    let strerror = msg.strip_suffix(&suffix).unwrap_or(&msg).to_owned();
                    PyOSError::new_err((errno, strerror, filename))
                }
                None if source.kind() == std::io::ErrorKind::IsADirectory => {
                    PyIsADirectoryError::new_err(format!("{filename} is a directory"))
                }
                None => PyOSError::new_err(format!("cannot read {filename}: {source}")),
            };
        }
        ModelConfigError::Unresolved { .. } => "UnresolvedConfigError",
        ModelConfigError::NotSupported { .. } | ModelConfigError::UnsupportedArchitecture(_) => {
            "UnsupportedModelError"
        }
        ModelConfigError::Override(_) => "OverrideError",
        &ModelConfigError::LabelCountMismatch { expected, found } => {
            return Python::attach(|py| label_count_error(py, e.to_string(), expected, found));
        }
        _ => "ModelConfigError",
    };
    Python::attach(|py| decoder_error(py, class, e.to_string()))
}

/// A `#[pyclass]` mirror of a model-config vocabulary, its discriminants
/// pinned to the Rust codes.
macro_rules! mirror_enum {
    (
        $(#[$meta:meta])*
        $py:ident <=> $rs:ident as $name:literal {
            $( $(#[$vmeta:meta])* $variant:ident = $code:literal ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[pyclass(name = $name, eq, eq_int, from_py_object, module = "edgefirst.decoder")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $py {
            $( $(#[$vmeta])* $variant = $code ),+
        }

        const _: () = {
            $( assert!($py::$variant as u32 == $rs::$variant.code()); )+
        };

        #[pymethods]
        impl $py {
            fn __hash__(&self) -> isize {
                *self as isize
            }
        }

        impl From<$py> for $rs {
            fn from(v: $py) -> Self {
                match v {
                    $( $py::$variant => $rs::$variant ),+
                }
            }
        }

        impl TryFrom<$rs> for $py {
            type Error = PyErr;

            fn try_from(v: $rs) -> PyResult<Self> {
                match v {
                    $( $rs::$variant => Ok($py::$variant), )+
                    other => Err(PyRuntimeError::new_err(format!(
                        "{} `{}` has no Python value",
                        $name,
                        other.as_str()
                    ))),
                }
            }
        }
    };
}

mirror_enum! {
    /// Source that supplied a resolved value.
    PyConfigSource <=> ConfigSource as "ConfigSource" {
        /// ``edgefirst.json`` embedded in the model file.
        EmbeddedEdgefirst = 0,
        /// ``edgefirst.json`` (and ``labels.txt``) beside the model file.
        SidecarEdgefirst = 1,
        /// Hugging Face configs embedded in the model file.
        EmbeddedHf = 2,
        /// Hugging Face configs beside the model file.
        SidecarHf = 3,
        /// Ultralytics export metadata.
        UltralyticsMetadata = 4,
        /// Guessed from the model's input and output names and shapes.
        Heuristic = 5,
        /// The documented default.
        Default = 6,
        /// A caller override (:class:`ModelOverrides`).
        Override = 7,
    }
}

mirror_enum! {
    /// Family a model's configuration identifies; it selects the convention
    /// a bare ``letterbox`` resize override follows.
    PyModelFamily <=> ModelFamily as "ModelFamily" {
        /// Described by an ``edgefirst.json``.
        EdgeFirst = 0,
        /// An Ultralytics export.
        Ultralytics = 1,
        /// A Hugging Face export.
        HuggingFace = 2,
        /// Configured only by overrides.
        Unknown = 3,
    }
}

mirror_enum! {
    /// What a model's outputs decode to.
    PyModelTask <=> ModelTask as "ModelTask" {
        /// Bounding boxes.
        Detect = 0,
        /// Bounding boxes with instance masks.
        Segment = 1,
        /// A per-pixel class map.
        Semantic = 2,
    }
}

mirror_enum! {
    /// A resolved value whose source is recorded.
    PyConfigField <=> ConfigField as "ConfigField" {
        /// Input normalization.
        Normalization = 0,
        /// Resize mode.
        Resize = 1,
        /// Letterbox placement.
        Placement = 2,
        /// Letterbox pad value and its space.
        Pad = 3,
        /// Resampling filter.
        Interpolation = 4,
        /// Model task.
        Task = 5,
        /// Semantic decode mode.
        DecodeMode = 6,
        /// Class labels.
        Labels = 7,
        /// Model input size.
        Size = 8,
    }
}

/// A `task` override: its grammar string or a :class:`ModelTask`.
#[derive(FromPyObject)]
pub enum TaskArg {
    Task(PyModelTask),
    Name(String),
}

/// A `decode` override: its grammar string or a :class:`DecodeMode`.
#[derive(FromPyObject)]
pub enum DecodeArg {
    Mode(PyDecodeMode),
    Name(String),
}

/// A `labels` override: a list of names or one comma-separated string.
#[derive(FromPyObject)]
pub enum LabelsArg {
    List(Vec<String>),
    Joined(String),
}

/// Values that replace what discovery found, applied after every source and
/// recorded as :attr:`ConfigSource.Override`.
///
/// Each argument takes the same grammar as the C API and command lines:
///
/// - ``task``: ``"detect" | "segment" | "semantic"`` (or a :class:`ModelTask`)
/// - ``normalize``: ``"unsigned" | "signed" | "raw" | "imagenet" |
///   "R,G,B/R,G,B"`` (mean/std in the 0-1 domain)
/// - ``resize``: ``"stretch" | "letterbox" | "letterbox-center" |
///   "letterbox-topleft"``; bare ``"letterbox"`` follows the model family
/// - ``decode``: ``"accurate" | "fast"`` (or a :class:`DecodeMode`)
/// - ``labels``: a list of non-empty class names, or one comma-separated
///   string. Resolution raises :class:`LabelCountMismatchError` when their
///   number differs from a class count the model's outputs state;
///   end-to-end detections and class maps state none, so any number is
///   accepted there and the caller checks it.
///
/// :meth:`ResolvedModelConfig.describe` prints these keys in the same
/// grammar.
///
/// Raises:
///     OverrideError: A value outside its grammar; the message lists the
///         accepted forms.
#[pyclass(
    name = "ModelOverrides",
    eq,
    from_py_object,
    module = "edgefirst.decoder"
)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PyModelOverrides(pub(crate) ModelOverrides);

impl PyModelOverrides {
    fn set_str(&mut self, key: &str, value: &str) -> PyResult<()> {
        self.0.set(key, value).map_err(config_err)
    }
}

#[pymethods]
impl PyModelOverrides {
    #[new]
    #[pyo3(signature = (task = None, normalize = None, resize = None, decode = None, labels = None))]
    fn new(
        task: Option<TaskArg>,
        normalize: Option<&str>,
        resize: Option<&str>,
        decode: Option<DecodeArg>,
        labels: Option<LabelsArg>,
    ) -> PyResult<Self> {
        let mut o = Self::default();
        match task {
            Some(TaskArg::Task(t)) => o.0.task = Some(t.into()),
            Some(TaskArg::Name(s)) => o.set_str("task", &s)?,
            None => {}
        }
        if let Some(n) = normalize {
            o.set_str("normalize", n)?;
        }
        if let Some(r) = resize {
            o.set_str("resize", r)?;
        }
        match decode {
            Some(DecodeArg::Mode(m)) => o.0.decode = Some(m.into()),
            Some(DecodeArg::Name(s)) => o.set_str("decode", &s)?,
            None => {}
        }
        match labels {
            Some(LabelsArg::List(l)) => {
                if let Some(i) = l.iter().position(|n| n.trim().is_empty()) {
                    let e = ModelConfigError::Override(format!("labels has an empty name at {i}"));
                    return Err(config_err(e));
                }
                o.0.labels = Some(l);
            }
            Some(LabelsArg::Joined(s)) => o.set_str("labels", &s)?,
            None => {}
        }
        Ok(o)
    }

    /// Sets the override named ``key`` (``task``, ``normalize``, ``resize``,
    /// ``decode`` or ``labels``) from its string form; ``labels`` is
    /// comma-separated.
    ///
    /// Raises:
    ///     OverrideError: An unknown key or a value outside its grammar.
    fn set(&mut self, key: &str, value: &str) -> PyResult<()> {
        self.set_str(key, value)
    }

    /// The ``task`` override, or ``None``.
    #[getter]
    fn task(&self) -> Option<&'static str> {
        self.0.task.map(|t| t.as_str())
    }

    /// The ``normalize`` override in its grammar form, or ``None``.
    #[getter]
    fn normalize(&self) -> Option<String> {
        self.0.normalize.map(|n| n.to_string())
    }

    /// The ``resize`` override in its grammar form, or ``None``.
    #[getter]
    fn resize(&self) -> Option<String> {
        self.0.resize.map(|r| r.to_string())
    }

    /// The ``decode`` override, ``"accurate"`` or ``"fast"``, or ``None``.
    #[getter]
    fn decode(&self) -> Option<&'static str> {
        self.0.decode.map(|d| d.as_str())
    }

    /// The ``labels`` override, or ``None``.
    #[getter]
    fn labels(&self) -> Option<Vec<String>> {
        self.0.labels.clone()
    }

    /// Whether no override is set.
    #[getter]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The set overrides as ``key=value`` pairs, the form a command line
    /// takes them in.
    fn __str__(&self) -> String {
        self.pairs()
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn __repr__(&self) -> String {
        let args: Vec<String> = self
            .pairs()
            .into_iter()
            .map(|(k, v)| match (k, &self.0.labels) {
                ("labels", Some(l)) => {
                    let l: Vec<String> = l.iter().map(|s| py_quote(s)).collect();
                    format!("labels=[{}]", l.join(", "))
                }
                _ => format!("{k}={}", py_quote(&v)),
            })
            .collect();
        format!("ModelOverrides({})", args.join(", "))
    }
}

/// `s` as a single-quoted Python string literal.
fn py_quote(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

impl PyModelOverrides {
    fn pairs(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if let Some(t) = self.task() {
            out.push(("task", t.to_owned()));
        }
        if let Some(n) = self.normalize() {
            out.push(("normalize", n));
        }
        if let Some(r) = self.resize() {
            out.push(("resize", r));
        }
        if let Some(d) = self.decode() {
            out.push(("decode", d.to_owned()));
        }
        if let Some(l) = &self.0.labels {
            out.push(("labels", l.join(",")));
        }
        out
    }
}

/// `x` as the `f64` its shortest decimal form names, so `123.675_f32`
/// reads back as `123.675` rather than `123.67500305175781`.
fn shortest(x: f32) -> f64 {
    x.to_string().parse().unwrap_or(f64::from(x))
}

/// `spec` as an `edgefirst.decoder.PreprocessSpec` of plain Python values.
fn preprocess_spec_object(py: Python<'_>, spec: &PreprocessSpec) -> PyResult<Py<PyAny>> {
    let n = &spec.normalization;
    let tuple3 = |v: [f32; 3]| (shortest(v[0]), shortest(v[1]), shortest(v[2]));
    let normalization = (tuple3(n.mean()), tuple3(n.std()));
    let obj = py
        .import("edgefirst.decoder")?
        .getattr("PreprocessSpec")?
        .call1((
            normalization,
            spec.fit.as_str(),
            spec.placement.as_str(),
            tuple3(spec.pad),
            spec.pad_space.as_str(),
            spec.interpolation.as_str(),
            spec.size,
        ))?;
    Ok(obj.unbind())
}

fn provenance_tuple(p: &Provenance) -> PyResult<(PyConfigSource, Option<String>)> {
    Ok((p.source.try_into()?, p.detail.clone()))
}

/// A model's configuration as resolved from every source, with where each
/// value came from. Returned by :func:`resolve_model_config`.
#[pyclass(name = "ResolvedModelConfig", frozen, module = "edgefirst.decoder")]
pub struct PyResolvedModelConfig(pub(crate) ResolvedModelConfig);

#[pymethods]
impl PyResolvedModelConfig {
    /// The output schema as an ``edgefirst.json`` schema v2 dict, ready for
    /// ``Decoder(schema)``.
    #[getter]
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize::pythonize(py, &self.0.schema)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to convert schema: {e}")))
    }

    /// Class names in channel order, with outer whitespace trimmed; empty
    /// when no source names them.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.0.labels.clone()
    }

    /// The family the configuration identifies.
    #[getter]
    fn family(&self) -> PyResult<PyModelFamily> {
        self.0.family.try_into()
    }

    /// What the outputs decode to.
    #[getter]
    fn task(&self) -> PyResult<PyModelTask> {
        self.0.task.try_into()
    }

    /// The semantic decode mode, ``None`` for other tasks.
    #[getter]
    fn decode_mode(&self) -> Option<PyDecodeMode> {
        self.0.decode_mode.map(Into::into)
    }

    /// A one-line summary of the model.
    #[getter]
    fn description(&self) -> String {
        self.0.description.clone()
    }

    /// Deviations and skipped files worth logging.
    #[getter]
    fn notes(&self) -> Vec<String> {
        self.0.notes.clone()
    }

    /// Hugging Face ``do_reduce_labels``, exposed but not applied.
    #[getter]
    fn reduce_labels(&self) -> Option<bool> {
        self.0.reduce_labels
    }

    /// Where each value came from: ``{ConfigField: (ConfigSource, detail)}``,
    /// ``detail`` naming the file or the reason for a guess, or ``None``.
    #[getter]
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (field, p) in &self.0.provenance {
            d.set_item(PyConfigField::try_from(*field)?, provenance_tuple(p)?)?;
        }
        Ok(d)
    }

    /// Where ``field`` came from, as ``(ConfigSource, detail)``, or ``None``.
    fn source(&self, field: PyConfigField) -> PyResult<Option<(PyConfigSource, Option<String>)>> {
        self.0
            .source(field.into())
            .map(provenance_tuple)
            .transpose()
    }

    /// The model's input preprocessing, as a :class:`PreprocessSpec` for
    /// ``edgefirst.image.ConvertOptions.from_spec``. Its interpolation is the
    /// reference pipeline's; ``edgefirst.image`` runs bicubic and area as
    /// bilinear, and :attr:`notes` says so.
    #[getter]
    fn preprocessing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        preprocess_spec_object(py, &self.0.preprocessing)
    }

    /// One ``key=value [source]`` line per resolved value, for logging, e.g.
    /// ``"resize=letterbox-topleft [preprocessor_config.json]"``. Override
    /// keys print in the :class:`ModelOverrides` grammar.
    fn describe(&self) -> Vec<String> {
        self.0.describe()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResolvedModelConfig(family={:?}, task={:?}, description={:?})",
            self.0.family.as_str(),
            self.0.task.as_str(),
            self.0.description
        )
    }
}

/// Resolves the configuration of the model file at ``path``: what it
/// embeds (``edgefirst.json``, Hugging Face configs, labels), the files in
/// its directory, and ``signals`` when given. ``overrides`` apply last.
///
/// Args:
///     path: The model file (``str`` or ``os.PathLike``).
///     signals: The model's inputs, outputs and metadata as a
///         :class:`ModelSignals`; needed to build a schema from Hugging Face
///         configs, Ultralytics metadata or I/O names.
///     overrides: A :class:`ModelOverrides`.
///
/// Returns:
///     A :class:`ResolvedModelConfig`.
///
/// Raises:
///     OSError: The model or a file beside it cannot be read, or the path
///         is a directory.
///     UnresolvedConfigError: Nothing supplies a required piece; the
///         message names it and the override that supplies it.
///     UnsupportedModelError: The configuration asks for something the
///         library cannot run (also a ``NotImplementedError``).
///     OverrideError: An override cannot apply.
///     ModelConfigError: A configuration file is malformed. Every one of
///         these is a ``ValueError``.
#[pyfunction]
#[pyo3(name = "resolve_model_config", signature = (path, signals = None, overrides = None))]
pub fn py_resolve_model_config(
    py: Python<'_>,
    path: PathBuf,
    signals: Option<&Bound<'_, PyAny>>,
    overrides: Option<PyModelOverrides>,
) -> PyResult<PyResolvedModelConfig> {
    let signals = signals
        .map(crate::infer::model_signals_from_object)
        .transpose()?;
    let overrides = overrides.map(|o| o.0).unwrap_or_default();
    let resolved = py
        .detach(|| mc::resolve_model_config(&path, signals, &overrides))
        .map_err(config_err)?;
    Ok(PyResolvedModelConfig(resolved))
}
