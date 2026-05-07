//! Loader for `.safetensors` decoder reference fixtures.
//!
//! Each fixture bundles, for one TFLite model + image pair:
//! - the raw quantized per-FPN-level tensors,
//! - per-stage NumPy reference intermediates (dequant, ltrb, xywh, …),
//! - the post-NMS decoded outputs (boxes_xyxy / scores / classes / masks),
//! - the schema, quantization, NMS config, and a Markdown narrative,
//!   all in the safetensors `__metadata__` table.
//!
//! See `.claude/plans/2026-04-29-per-scale-fixture-framework-design.md`
//! and `testdata/decoder/README.md` for the format.
//!
//! Fixture keys map 1:1 to HAL kernels under
//! `crates/decoder/src/per_scale/`:
//!
//! | fixture key                               | HAL kernel               |
//! |-------------------------------------------|--------------------------|
//! | `raw.boxes_<lvl>`                         | `kernels/level_box.rs`   |
//! | `intermediate.boxes_<lvl>.dequant`        | `level_box.rs` dequant   |
//! | `intermediate.boxes_<lvl>.ltrb`           | `level_box.rs` DFL stage |
//! | `intermediate.boxes_<lvl>.xywh`           | `level_box.rs` dist2bbox |
//! | `raw.scores_<lvl>`                        | `kernels/level_score.rs` |
//! | `intermediate.scores_<lvl>.dequant`       | `level_score.rs` dequant |
//! | `intermediate.scores_<lvl>.activated`     | `level_score.rs` sigmoid |
//! | `raw.mc_<lvl>`                            | `kernels/level_mc.rs`    |
//! | `intermediate.mc_<lvl>.dequant`           | `level_mc.rs` dequant    |
//! | `raw.protos`                              | `pipeline.rs` proto path |
//! | `intermediate.protos.dequant`             | `pipeline.rs`            |

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::SafeTensors;

#[derive(Debug)]
pub enum FixtureError {
    NotPresent(PathBuf),
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixtureError::NotPresent(p) => {
                write!(
                    f,
                    "fixture {} not present (run `git lfs pull`)",
                    p.display()
                )
            }
            FixtureError::Io(e) => write!(f, "I/O error: {e}"),
            FixtureError::Parse(s) => write!(f, "fixture parse error: {s}"),
        }
    }
}
impl std::error::Error for FixtureError {}

#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub dtype: String,
}

#[derive(Debug, Clone, Copy)]
pub struct NmsConfig {
    pub iou_threshold: f32,
    pub score_threshold: f32,
    pub max_detections: u32,
}

/// Loaded reference fixture for one model + image pair.
///
/// Owns the file bytes. Tensor accessors lazily re-deserialize the
/// safetensors view (cheap — header-only parse plus offset bookkeeping)
/// because safetensors v0.7's `SafeTensors` borrows from the byte slice
/// and would otherwise tangle this struct in lifetime parameters.
pub struct PerScaleFixture {
    pub path: PathBuf,
    pub format_version: String,
    pub decoder_family: String,
    pub model_basename: String,
    pub expected_count_min: u32,
    schema_json: String,
    quantization: HashMap<String, QuantParams>,
    nms_config: NmsConfig,
    pub(crate) raw_bytes: Vec<u8>,
}

impl PerScaleFixture {
    pub fn schema_json(&self) -> &str {
        &self.schema_json
    }

    pub fn quantization_for(&self, key: &str) -> Option<&QuantParams> {
        self.quantization.get(key)
    }

    pub fn nms_config(&self) -> NmsConfig {
        self.nms_config
    }

    pub fn load(path: &Path) -> Result<Self, FixtureError> {
        if !path.exists() {
            return Err(FixtureError::NotPresent(path.to_path_buf()));
        }
        let raw_bytes = fs::read(path).map_err(FixtureError::Io)?;

        // safetensors v0.7: read_metadata is a static method that returns
        // (header_size, Metadata). The user's __metadata__ dict lives at
        // Metadata::metadata() -> &Option<HashMap<String, String>>.
        let (_n, parsed) = SafeTensors::read_metadata(&raw_bytes)
            .map_err(|e| FixtureError::Parse(format!("read_metadata: {e}")))?;
        let user_meta: &HashMap<String, String> = parsed
            .metadata()
            .as_ref()
            .ok_or_else(|| FixtureError::Parse("missing __metadata__".into()))?;

        let req = |k: &str| -> Result<&str, FixtureError> {
            user_meta
                .get(k)
                .map(String::as_str)
                .ok_or_else(|| FixtureError::Parse(format!("missing metadata key {k:?}")))
        };

        let format_version = req("format_version")?.to_string();
        let decoder_family = req("decoder_family")?.to_string();
        let model_basename = req("model_basename")?.to_string();
        let expected_count_min: u32 = req("expected_count_min")?
            .parse()
            .map_err(|e| FixtureError::Parse(format!("expected_count_min: {e}")))?;

        let schema_json = req("schema_json")?.to_string();

        let quant_raw = req("quantization_json")?;
        let quant_value: serde_json::Value = serde_json::from_str(quant_raw)
            .map_err(|e| FixtureError::Parse(format!("quantization_json: {e}")))?;
        let mut quantization: HashMap<String, QuantParams> = HashMap::new();
        if let serde_json::Value::Object(map) = quant_value {
            for (k, v) in map {
                quantization.insert(
                    k,
                    QuantParams {
                        scale: v.get("scale").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
                        zero_point: v.get("zero_point").and_then(|x| x.as_i64()).unwrap_or(0)
                            as i32,
                        dtype: v
                            .get("dtype")
                            .and_then(|x| x.as_str())
                            .unwrap_or("int8")
                            .to_string(),
                    },
                );
            }
        }

        let nms_raw = req("nms_config_json")?;
        let nms_value: serde_json::Value = serde_json::from_str(nms_raw)
            .map_err(|e| FixtureError::Parse(format!("nms_config_json: {e}")))?;
        let nms_config = NmsConfig {
            iou_threshold: nms_value
                .get("iou_threshold")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.7) as f32,
            score_threshold: nms_value
                .get("score_threshold")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.001) as f32,
            max_detections: nms_value
                .get("max_detections")
                .and_then(|x| x.as_u64())
                .unwrap_or(300) as u32,
        };

        Ok(Self {
            path: path.to_path_buf(),
            format_version,
            decoder_family,
            model_basename,
            expected_count_min,
            schema_json,
            quantization,
            nms_config,
            raw_bytes,
        })
    }
}

use ndarray::{Array1, Array2, Array3, Array4, ArrayD};
use safetensors::tensor::Dtype as StDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawDtype {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F16,
    F32,
}

impl RawDtype {
    pub fn from_safetensors(d: StDtype) -> Result<Self, FixtureError> {
        Ok(match d {
            StDtype::I8 => RawDtype::I8,
            StDtype::U8 => RawDtype::U8,
            StDtype::I16 => RawDtype::I16,
            StDtype::U16 => RawDtype::U16,
            StDtype::I32 => RawDtype::I32,
            StDtype::U32 => RawDtype::U32,
            StDtype::F16 => RawDtype::F16,
            StDtype::F32 => RawDtype::F32,
            other => {
                return Err(FixtureError::Parse(format!(
                    "unsupported raw dtype {other:?}"
                )))
            }
        })
    }
}

/// Owned tensor view from a fixture.
///
/// Carries a copy of the safetensors-stored bytes plus dtype and shape.
/// Owned because `safetensors::SafeTensors` borrows from the fixture's
/// `raw_bytes` and goes out of scope between accessor calls; copying a
/// few MB per test run is cheaper than threading lifetimes through
/// every test.
pub struct RawTensor {
    pub dtype: RawDtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl PerScaleFixture {
    pub fn raw_tensor(&self, key: &str) -> Result<RawTensor, FixtureError> {
        let st = SafeTensors::deserialize(&self.raw_bytes)
            .map_err(|e| FixtureError::Parse(format!("deserialize: {e}")))?;
        let v = st
            .tensor(key)
            .map_err(|e| FixtureError::Parse(format!("{key}: {e}")))?;
        Ok(RawTensor {
            dtype: RawDtype::from_safetensors(v.dtype())?,
            shape: v.shape().to_vec(),
            bytes: v.data().to_vec(),
        })
    }

    pub fn input_image_uint8(&self) -> Result<Array4<u8>, FixtureError> {
        let raw = self.raw_tensor("input.image")?;
        if raw.dtype != RawDtype::U8 {
            return Err(FixtureError::Parse(format!(
                "input.image must be u8, got {:?}",
                raw.dtype
            )));
        }
        if raw.shape.len() != 4 {
            return Err(FixtureError::Parse(format!(
                "input.image must be 4-D, got shape {:?}",
                raw.shape
            )));
        }
        let shape = (raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3]);
        Array4::from_shape_vec(shape, raw.bytes)
            .map_err(|e| FixtureError::Parse(format!("input.image reshape: {e}")))
    }
}

// ────────────────────────────────────────────────────────────────────
// decoded() / intermediates()
// ────────────────────────────────────────────────────────────────────

/// Post-NMS decoded outputs extracted from the fixture.
pub struct DecodedRef {
    pub boxes_xyxy: Array2<f32>,
    pub scores: Array1<f32>,
    pub classes: Array1<u32>,
    pub masks: Option<Array3<u8>>,
}

/// Borrowed accessor for per-stage f32 intermediate arrays.
///
/// Borrows `&PerScaleFixture` so the safetensors blob is only
/// re-deserialized once per `raw_tensor` call, not at construction.
pub struct Intermediates<'a> {
    fix: &'a PerScaleFixture,
}

impl<'a> Intermediates<'a> {
    fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        let raw = self.fix.raw_tensor(key).ok()?;
        if raw.dtype != RawDtype::F32 {
            return None;
        }
        let mut out = Vec::with_capacity(raw.bytes.len() / 4);
        for chunk in raw.bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Some(out)
    }

    fn get_shape(&self, key: &str, expected_rank: usize) -> Option<Vec<usize>> {
        let raw = self.fix.raw_tensor(key).ok()?;
        if raw.shape.len() != expected_rank {
            return None;
        }
        Some(raw.shape)
    }

    pub fn boxes_dequant(&self, level: usize) -> Option<ArrayD<f32>> {
        let key = format!("intermediate.boxes_{level}.dequant");
        let shape = self.get_shape(&key, 4)?;
        let data = self.get_f32_array(&key)?;
        ArrayD::from_shape_vec(shape, data).ok()
    }

    pub fn boxes_ltrb(&self, level: usize) -> Option<Array2<f32>> {
        let key = format!("intermediate.boxes_{level}.ltrb");
        let raw = self.fix.raw_tensor(&key).ok()?;
        if raw.shape.len() != 2 {
            return None;
        }
        let data = self.get_f32_array(&key)?;
        Array2::from_shape_vec((raw.shape[0], raw.shape[1]), data).ok()
    }

    pub fn boxes_xywh(&self, level: usize) -> Option<Array2<f32>> {
        let key = format!("intermediate.boxes_{level}.xywh");
        let raw = self.fix.raw_tensor(&key).ok()?;
        if raw.shape.len() != 2 {
            return None;
        }
        let data = self.get_f32_array(&key)?;
        Array2::from_shape_vec((raw.shape[0], raw.shape[1]), data).ok()
    }

    pub fn scores_dequant(&self, level: usize) -> Option<ArrayD<f32>> {
        let key = format!("intermediate.scores_{level}.dequant");
        let shape = self.get_shape(&key, 4)?;
        let data = self.get_f32_array(&key)?;
        ArrayD::from_shape_vec(shape, data).ok()
    }

    pub fn scores_activated(&self, level: usize) -> Option<ArrayD<f32>> {
        let key = format!("intermediate.scores_{level}.activated");
        let shape = self.get_shape(&key, 4)?;
        let data = self.get_f32_array(&key)?;
        ArrayD::from_shape_vec(shape, data).ok()
    }

    pub fn mc_dequant(&self, level: usize) -> Option<ArrayD<f32>> {
        let key = format!("intermediate.mc_{level}.dequant");
        let shape = self.get_shape(&key, 4)?;
        let data = self.get_f32_array(&key)?;
        ArrayD::from_shape_vec(shape, data).ok()
    }

    pub fn protos_dequant(&self) -> Option<ArrayD<f32>> {
        let key = "intermediate.protos.dequant";
        let shape = self.get_shape(key, 4)?;
        let data = self.get_f32_array(key)?;
        ArrayD::from_shape_vec(shape, data).ok()
    }
}

impl PerScaleFixture {
    pub fn decoded(&self) -> Result<DecodedRef, FixtureError> {
        let boxes_raw = self.raw_tensor("decoded.boxes_xyxy")?;
        let scores_raw = self.raw_tensor("decoded.scores")?;
        let classes_raw = self.raw_tensor("decoded.classes")?;
        let n = boxes_raw.shape.first().copied().unwrap_or(0);

        let mut boxes_data = Vec::with_capacity(n * 4);
        for chunk in boxes_raw.bytes.chunks_exact(4) {
            boxes_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let mut scores_data = Vec::with_capacity(n);
        for chunk in scores_raw.bytes.chunks_exact(4) {
            scores_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let mut classes_data = Vec::with_capacity(n);
        for chunk in classes_raw.bytes.chunks_exact(4) {
            classes_data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let masks = self.raw_tensor("decoded.masks").ok().map(|raw| {
            Array3::from_shape_vec((raw.shape[0], raw.shape[1], raw.shape[2]), raw.bytes)
                .expect("decoded.masks reshape")
        });

        Ok(DecodedRef {
            boxes_xyxy: Array2::from_shape_vec((n, 4), boxes_data)
                .map_err(|e| FixtureError::Parse(format!("decoded.boxes_xyxy: {e}")))?,
            scores: Array1::from_vec(scores_data),
            classes: Array1::from_vec(classes_data),
            masks,
        })
    }

    pub fn intermediates(&self) -> Option<Intermediates<'_>> {
        let st = SafeTensors::deserialize(&self.raw_bytes).ok()?;
        let any = st.names().iter().any(|k| k.starts_with("intermediate."));
        if any {
            Some(Intermediates { fix: self })
        } else {
            None
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// build_tensors_with_quant
// ────────────────────────────────────────────────────────────────────

use edgefirst_tensor::{Quantization, Tensor, TensorDyn};

impl PerScaleFixture {
    /// Build the `Vec<TensorDyn>` to feed into `Decoder::decode_proto`.
    ///
    /// Walks every `raw.*` key (including `raw.protos`), allocates a
    /// `TensorDyn` of the matching dtype + shape, copies the bytes, and
    /// attaches the per-tensor `Quantization` parsed from
    /// `quantization_json`. The HAL per-scale pipeline reads
    /// quantization off the tensor object via `quant_from_tensor`
    /// (`crates/decoder/src/per_scale/pipeline.rs`), so attaching here
    /// is mandatory.
    pub fn build_tensors_with_quant(&self) -> Result<Vec<TensorDyn>, FixtureError> {
        let st = SafeTensors::deserialize(&self.raw_bytes)
            .map_err(|e| FixtureError::Parse(format!("deserialize: {e}")))?;

        // Sort names for deterministic ordering across runs (HashMap iter
        // order is unspecified; HAL doesn't care about input ordering
        // because resolve_bindings uses shape-match, but determinism
        // makes test failures easier to reason about).
        let mut names: Vec<String> = st
            .names()
            .iter()
            .filter(|k| k.starts_with("raw."))
            .map(|s| s.to_string())
            .collect();
        names.sort();

        let mut tensors = Vec::with_capacity(names.len());
        for key in &names {
            let raw = self.raw_tensor(key)?;
            let lookup_key = key.trim_start_matches("raw.").to_string();
            let qp = self.quantization_for(&lookup_key);
            let t = build_one_tensor(&raw, qp)?;
            tensors.push(t);
        }
        Ok(tensors)
    }
}

fn build_one_tensor(raw: &RawTensor, qp: Option<&QuantParams>) -> Result<TensorDyn, FixtureError> {
    let shape: Vec<usize> = raw.shape.clone();
    let mut t: TensorDyn = match raw.dtype {
        RawDtype::I8 => {
            let v: Vec<i8> = raw.bytes.iter().map(|&b| b as i8).collect();
            Tensor::<i8>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("i8 tensor: {e}")))?
                .into()
        }
        RawDtype::U8 => Tensor::<u8>::from_slice(&raw.bytes, &shape)
            .map_err(|e| FixtureError::Parse(format!("u8 tensor: {e}")))?
            .into(),
        RawDtype::I16 => {
            let mut v: Vec<i16> = Vec::with_capacity(raw.bytes.len() / 2);
            for c in raw.bytes.chunks_exact(2) {
                v.push(i16::from_le_bytes([c[0], c[1]]));
            }
            Tensor::<i16>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("i16 tensor: {e}")))?
                .into()
        }
        RawDtype::U16 => {
            let mut v: Vec<u16> = Vec::with_capacity(raw.bytes.len() / 2);
            for c in raw.bytes.chunks_exact(2) {
                v.push(u16::from_le_bytes([c[0], c[1]]));
            }
            Tensor::<u16>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("u16 tensor: {e}")))?
                .into()
        }
        RawDtype::I32 => {
            let mut v: Vec<i32> = Vec::with_capacity(raw.bytes.len() / 4);
            for c in raw.bytes.chunks_exact(4) {
                v.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Tensor::<i32>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("i32 tensor: {e}")))?
                .into()
        }
        RawDtype::U32 => {
            let mut v: Vec<u32> = Vec::with_capacity(raw.bytes.len() / 4);
            for c in raw.bytes.chunks_exact(4) {
                v.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Tensor::<u32>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("u32 tensor: {e}")))?
                .into()
        }
        RawDtype::F16 => {
            let mut v: Vec<half::f16> = Vec::with_capacity(raw.bytes.len() / 2);
            for c in raw.bytes.chunks_exact(2) {
                v.push(half::f16::from_le_bytes([c[0], c[1]]));
            }
            Tensor::<half::f16>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("f16 tensor: {e}")))?
                .into()
        }
        RawDtype::F32 => {
            let mut v: Vec<f32> = Vec::with_capacity(raw.bytes.len() / 4);
            for c in raw.bytes.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Tensor::<f32>::from_slice(&v, &shape)
                .map_err(|e| FixtureError::Parse(format!("f32 tensor: {e}")))?
                .into()
        }
    };
    if let Some(qp) = qp {
        // set_quantization returns Result; floats reject quantization,
        // but a quant entry on a float tensor is a fixture-build bug we
        // surface explicitly rather than silently ignore.
        t.set_quantization(Quantization::per_tensor(qp.scale, qp.zero_point))
            .map_err(|e| FixtureError::Parse(format!("set_quantization: {e}")))?;
    }
    Ok(t)
}
