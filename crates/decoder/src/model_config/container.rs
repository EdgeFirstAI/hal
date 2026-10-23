// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The configuration files a model carries, embedded in its container or
//! beside it.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

use super::ModelConfigError;

/// `edgefirst.json`, as a file name and the legacy ONNX metadata key.
pub const EDGEFIRST_JSON: &str = "edgefirst.json";
/// Class names, one per line.
pub const LABELS_TXT: &str = "labels.txt";
/// Hugging Face model config.
pub const HF_CONFIG: &str = "config.json";
/// Hugging Face image processor config.
pub const HF_PREPROCESSOR: &str = "preprocessor_config.json";

/// Largest sidecar file read.
const MAX_SIDECAR: u64 = 16 << 20;

/// ONNX `metadata_props` key for `edgefirst.json`.
pub const ONNX_EDGEFIRST: &str = "edgefirst";
/// ONNX `metadata_props` key for the class names.
pub const ONNX_LABELS: &str = "labels";
/// ONNX `metadata_props` key for the Hugging Face `config.json`.
pub const ONNX_HF_CONFIG: &str = "hf_config";
/// ONNX `metadata_props` key for the Hugging Face `preprocessor_config.json`.
pub const ONNX_HF_PREPROCESSOR: &str = "hf_preprocessor_config";

/// One configuration file and where it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConfigFile {
    /// The file name, or the ONNX metadata key it was stored under.
    pub name: String,
    /// The file contents.
    pub text: String,
}

impl ConfigFile {
    /// A file named `name` holding `text`.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: text.into(),
        }
    }
}

/// The configuration files of one location: a model container or its
/// directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelFiles {
    /// `edgefirst.json`.
    pub edgefirst_json: Option<ConfigFile>,
    /// `labels.txt`.
    pub labels_txt: Option<ConfigFile>,
    /// Hugging Face `config.json`.
    pub hf_config: Option<ConfigFile>,
    /// Hugging Face `preprocessor_config.json`.
    pub hf_preprocessor: Option<ConfigFile>,
    /// Every ONNX metadata prop and ZIP text entry, by key or entry name.
    /// Ultralytics metadata reaches schema inference through it.
    pub metadata: BTreeMap<String, String>,
    /// Model bytes before an appended ZIP trailer, which runtimes that check
    /// the blob length need stripped.
    pub payload_len: Option<u64>,
}

impl ModelFiles {
    /// Whether nothing was found.
    pub fn is_empty(&self) -> bool {
        self.edgefirst_json.is_none()
            && self.labels_txt.is_none()
            && self.hf_config.is_none()
            && self.hf_preprocessor.is_none()
            && self.metadata.is_empty()
    }

    fn set(slot: &mut Option<ConfigFile>, name: &str, text: &str) {
        if slot.is_none() {
            *slot = Some(ConfigFile {
                name: name.to_owned(),
                text: text.to_owned(),
            });
        }
    }

    fn add_onnx_props(&mut self, props: BTreeMap<String, String>) {
        for (key, value) in &props {
            match key.as_str() {
                ONNX_EDGEFIRST | EDGEFIRST_JSON => Self::set(&mut self.edgefirst_json, key, value),
                ONNX_LABELS => Self::set(&mut self.labels_txt, key, value),
                ONNX_HF_CONFIG => Self::set(&mut self.hf_config, key, value),
                ONNX_HF_PREPROCESSOR => Self::set(&mut self.hf_preprocessor, key, value),
                _ => {}
            }
        }
        self.metadata.extend(props);
    }

    #[cfg(feature = "model-files")]
    fn add_entries(&mut self, entries: Vec<(String, String)>) {
        for (name, text) in entries {
            let base = name.rsplit('/').next().unwrap_or(&name);
            match base {
                EDGEFIRST_JSON => Self::set(&mut self.edgefirst_json, base, &text),
                LABELS_TXT => Self::set(&mut self.labels_txt, base, &text),
                HF_CONFIG => Self::set(&mut self.hf_config, base, &text),
                HF_PREPROCESSOR => Self::set(&mut self.hf_preprocessor, base, &text),
                _ => {}
            }
            self.metadata.insert(name, text);
        }
    }
}

/// The configuration files embedded in the model file at `path`: ONNX
/// `metadata_props` and, with the `model-files` feature, a ZIP trailer.
/// Neither the graph nor the model payload is read.
///
/// # Errors
///
/// [`ModelConfigError::Io`] when the file cannot be read or is a
/// directory, and [`ModelConfigError::Container`] when a `.onnx` file is not
/// a valid `ModelProto`.
pub fn read_embedded(path: impl AsRef<Path>) -> Result<ModelFiles, ModelConfigError> {
    let path = path.as_ref();
    let io = |source| ModelConfigError::Io {
        path: path.to_owned(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let meta = file.metadata().map_err(io)?;
    if meta.is_dir() {
        return Err(io(std::io::ErrorKind::IsADirectory.into()));
    }
    let len = meta.len();
    let onnx = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    read_from(&mut file, len, onnx)
}

/// [`read_embedded`] on a model already in memory. `onnx` says the bytes are
/// an ONNX model, which makes a malformed `ModelProto` an error instead of
/// "not ONNX".
///
/// # Errors
///
/// [`ModelConfigError::Container`] when `onnx` is set and the bytes are not
/// a valid `ModelProto`.
pub fn read_embedded_bytes(bytes: &[u8], onnx: bool) -> Result<ModelFiles, ModelConfigError> {
    read_from(&mut Cursor::new(bytes), bytes.len() as u64, onnx)
}

fn read_from<R: Read + Seek>(
    r: &mut R,
    len: u64,
    onnx: bool,
) -> Result<ModelFiles, ModelConfigError> {
    let mut files = ModelFiles::default();
    #[cfg(feature = "model-files")]
    if let Some(trailer) = super::ziptrailer::read_trailer(r, len)? {
        files.payload_len = Some(trailer.payload_len);
        files.add_entries(trailer.entries);
    }
    // An appended trailer is not part of the protobuf.
    let model_len = files.payload_len.unwrap_or(len);
    let mut first = [0u8; 1];
    r.seek(std::io::SeekFrom::Start(0))
        .map_err(|e| ModelConfigError::Container(e.to_string()))?;
    let looks_onnx = model_len > 0 && r.read_exact(&mut first).is_ok() && first[0] == 0x08;
    if onnx || looks_onnx {
        match super::onnx::metadata_props(r, model_len) {
            Ok(props) => files.add_onnx_props(props),
            Err(e) if onnx => return Err(e),
            Err(e) => log::debug!("not an ONNX model: {e}"),
        }
    }
    Ok(files)
}

/// The configuration files beside a model, in `dir`: `edgefirst.json`,
/// `labels.txt`, `config.json` and `preprocessor_config.json`.
///
/// # Errors
///
/// [`ModelConfigError::Io`] when a file exists but cannot be read, is not
/// UTF-8 or is larger than 16 MiB.
pub fn read_sidecars(dir: impl AsRef<Path>) -> Result<ModelFiles, ModelConfigError> {
    let dir = dir.as_ref();
    let mut files = ModelFiles::default();
    for (name, slot) in [
        (EDGEFIRST_JSON, &mut files.edgefirst_json),
        (LABELS_TXT, &mut files.labels_txt),
        (HF_CONFIG, &mut files.hf_config),
        (HF_PREPROCESSOR, &mut files.hf_preprocessor),
    ] {
        let path = dir.join(name);
        match read_capped(&path) {
            Ok(text) => {
                *slot = Some(ConfigFile {
                    name: name.to_owned(),
                    text,
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(ModelConfigError::Io { path, source }),
        }
    }
    Ok(files)
}

fn read_capped(path: &Path) -> std::io::Result<String> {
    let mut text = String::new();
    File::open(path)?
        .take(MAX_SIDECAR + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > MAX_SIDECAR {
        return Err(std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            "larger than 16 MiB",
        ));
    }
    Ok(text)
}

/// Non-empty trimmed lines of a `labels.txt`.
pub(crate) fn parse_labels(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onnx_both_key_sets_map_to_fields() {
        let bytes = super::super::onnx::tests::model_proto(
            &[
                ("edgefirst.json", "{\"legacy\":1}"),
                ("labels", "cat\ndog"),
                ("hf_config", "{}"),
                ("hf_preprocessor_config", "{\"size\":1}"),
                ("names", "{0: 'cat'}"),
            ],
            32,
        );
        let f = read_embedded_bytes(&bytes, true).unwrap();
        assert_eq!(f.edgefirst_json.unwrap().name, "edgefirst.json");
        assert_eq!(parse_labels(&f.labels_txt.unwrap().text), ["cat", "dog"]);
        assert_eq!(f.hf_config.unwrap().name, "hf_config");
        assert_eq!(f.hf_preprocessor.unwrap().text, "{\"size\":1}");
        assert_eq!(f.metadata["names"], "{0: 'cat'}");

        let bytes = super::super::onnx::tests::model_proto(&[("edgefirst", "{}")], 0);
        let f = read_embedded_bytes(&bytes, true).unwrap();
        assert_eq!(f.edgefirst_json.unwrap().name, "edgefirst");
    }

    #[test]
    fn malformed_onnx_is_an_error_only_when_named_onnx() {
        let bytes = [0x08, 0x08, 0x3a, 0x7f];
        assert!(read_embedded_bytes(&bytes, true).is_err());
        assert!(read_embedded_bytes(&bytes, false).unwrap().is_empty());
    }

    #[cfg(feature = "model-files")]
    #[test]
    fn tflite_associated_files_read() {
        let mut payload = vec![0u8; 4];
        payload.extend_from_slice(b"TFL3");
        payload.resize(512, 0x5a);
        let bytes = super::super::ziptrailer::tests::with_trailer(
            &payload,
            &[
                ("config.json", "{\"model_type\":\"segformer\"}"),
                ("preprocessor_config.json", "{}"),
                ("metadata.json", "{\"names\":{}}"),
            ],
        );
        let f = read_embedded_bytes(&bytes, false).unwrap();
        assert_eq!(f.payload_len, Some(512));
        assert_eq!(f.hf_config.unwrap().name, "config.json");
        assert!(f.hf_preprocessor.is_some());
        assert!(f.metadata.contains_key("metadata.json"));
    }

    #[cfg(feature = "model-files")]
    #[test]
    fn onnx_with_appended_zip_reads_both() {
        let model = super::super::onnx::tests::model_proto(&[("labels", "a\nb")], 64);
        let bytes =
            super::super::ziptrailer::tests::with_trailer(&model, &[("edgefirst.json", "{}")]);
        let f = read_embedded_bytes(&bytes, true).unwrap();
        assert_eq!(f.payload_len, Some(model.len() as u64));
        assert!(f.edgefirst_json.is_some());
        assert_eq!(f.labels_txt.unwrap().name, "labels");
    }

    #[test]
    fn a_directory_is_an_io_error() {
        let e = read_embedded(std::env::temp_dir()).unwrap_err();
        assert!(
            matches!(e, ModelConfigError::Io { ref source, .. } if source.kind() == std::io::ErrorKind::IsADirectory),
            "{e}"
        );
    }

    #[test]
    fn sidecars_read_from_directory() {
        let dir = std::env::temp_dir().join(format!("ef-sidecar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("edgefirst.json"), "{}").unwrap();
        std::fs::write(dir.join("labels.txt"), " a \n\nb\n").unwrap();
        let f = read_sidecars(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(parse_labels(&f.labels_txt.unwrap().text), ["a", "b"]);
        assert!(f.edgefirst_json.is_some());
        assert!(f.hf_config.is_none());
    }
}
