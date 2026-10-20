// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Parity tests: HAL semantic decoder vs the mmsegmentation reference decode
//! on real SegFormer-B1 (ADE20K) outputs.
//!
//! Each fixture holds the ONNX export's float32 logits for one `testdata`
//! image, an int8 per-tensor quantization of them, and the reference class
//! maps: the two-step bilinear decode (accurate) and argmax followed by
//! PyTorch `nearest` (fast). Accurate must agree on almost every pixel and
//! differ only where the reference's top two classes are near-tied; fast may
//! deviate within a bound.
//!
//! Fixtures are regenerated with `scripts/decoder_generate_semantic_fixture.py`.

mod common;

use std::collections::HashMap;

use edgefirst_decoder::schema::DecodeMode;
use edgefirst_decoder::semantic::SemanticOptions;
use edgefirst_decoder::DecoderBuilder;
use edgefirst_tensor::{
    DType, Element, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use half::f16;
use safetensors::SafeTensors;

use common::testdata_root;

/// Minimum share of pixels where accurate matches the accurate reference.
const ACCURATE_AGREEMENT: f64 = 0.998;
/// Largest reference top-two logit gap at which accurate may differ.
const ACCURATE_TIE_MARGIN: f32 = 0.3;
/// Minimum share of pixels where fast matches the fast reference.
const FAST_AGREEMENT: f64 = 0.97;

const FIXTURES: [&str; 3] = [
    "segformer-b1-ade-512x672_zidane.safetensors",
    "segformer-b1-ade-512x512_jaguar.safetensors",
    "segformer-b1-ade-512x512_coco_420_odd.safetensors",
];

fn load(name: &str) -> Option<Vec<u8>> {
    common::read_fixture(&testdata_root().join("decoder").join(name))
}

struct Fixture<'a> {
    name: &'a str,
    tensors: SafeTensors<'a>,
    metadata: HashMap<String, String>,
}

impl<'a> Fixture<'a> {
    fn new(name: &'a str, bytes: &'a [u8]) -> Self {
        let (_, header) = SafeTensors::read_metadata(bytes).unwrap();
        Self {
            name,
            tensors: SafeTensors::deserialize(bytes).unwrap(),
            metadata: header.metadata().clone().unwrap_or_default(),
        }
    }

    fn shape(&self, key: &str) -> Vec<usize> {
        self.tensors.tensor(key).unwrap().shape().to_vec()
    }

    fn values<T>(&self, key: &str, from: fn(&[u8]) -> T) -> Vec<T> {
        let view = self.tensors.tensor(key).unwrap();
        let size = view.data().len() / view.shape().iter().product::<usize>();
        view.data().chunks_exact(size).map(from).collect()
    }

    fn input<T: Element + Copy>(&self, key: &str, from: fn(&[u8]) -> T) -> TensorDyn
    where
        TensorDyn: From<Tensor<T>>,
    {
        let t = Tensor::<T>::new(&self.shape(key), Some(TensorMemory::Mem), None).unwrap();
        t.map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&self.values(key, from));
        t.into()
    }

    fn decode(&self, schema: &str, logits: &TensorDyn, mode: DecodeMode) -> Vec<u8> {
        let decoder = DecoderBuilder::new()
            .with_config_json_str(self.metadata[schema].clone())
            .build()
            .unwrap();
        let letterbox: [f32; 4] = serde_json::from_str(&self.metadata["letterbox"]).unwrap();
        let options = SemanticOptions::new()
            .with_mode(mode)
            .with_letterbox(letterbox);
        let shape = self.shape("ref.accurate");
        let mut dst = TensorDyn::new(&shape, DType::U8, Some(TensorMemory::Mem), None).unwrap();
        decoder
            .decode_semantic(&[logits], &mut dst, None, &options)
            .unwrap();
        let map = dst.as_typed::<u8>().unwrap().map().unwrap();
        map.as_slice().to_vec()
    }

    /// Checks accurate and fast decodes of `logits` against the references
    /// under `suffix`, and returns the accurate disagreement mask.
    fn check(&self, schema: &str, logits: &TensorDyn, suffix: &str) -> Vec<bool> {
        let name = self.name;
        let accurate = self.decode(schema, logits, DecodeMode::Accurate);
        let reference = self.values(&format!("ref.accurate{suffix}"), |b| b[0]);
        let share = agreement(&accurate, &reference);
        assert!(
            share >= ACCURATE_AGREEMENT,
            "{name}{suffix}: accurate agrees with the reference on {:.4}% of pixels",
            share * 100.0
        );

        let fast = self.decode(schema, logits, DecodeMode::Fast);
        let reference_fast = self.values(&format!("ref.fast{suffix}"), |b| b[0]);
        let share = agreement(&fast, &reference_fast);
        assert!(
            share >= FAST_AGREEMENT,
            "{name}{suffix}: fast agrees with the reference fast path on {:.4}% of pixels",
            share * 100.0
        );
        accurate
            .iter()
            .zip(&reference)
            .map(|(a, b)| a != b)
            .collect()
    }
}

fn agreement(out: &[u8], reference: &[u8]) -> f64 {
    let same = out.iter().zip(reference).filter(|(a, b)| a == b).count();
    same as f64 / reference.len() as f64
}

fn f32_le(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
fn segformer_float_outputs_match_the_reference_decode() {
    for name in FIXTURES {
        let Some(bytes) = load(name) else {
            continue;
        };
        let fixture = Fixture::new(name, &bytes);
        let logits = fixture.input("logits", f32_le);
        let differs = fixture.check("edgefirst_json", &logits, "");
        let margin = fixture.values("ref.margin", |b| f16::from_le_bytes([b[0], b[1]]).to_f32());
        let worst = differs
            .iter()
            .zip(&margin)
            .filter(|(d, _)| **d)
            .map(|(_, m)| *m)
            .fold(0.0f32, f32::max);
        assert!(
            worst < ACCURATE_TIE_MARGIN,
            "{name}: accurate differs from the reference where its top two logits are {worst} apart"
        );
    }
}

#[test]
fn segformer_int8_outputs_match_the_reference_decode() {
    for name in FIXTURES {
        let Some(bytes) = load(name) else {
            continue;
        };
        let fixture = Fixture::new(name, &bytes);
        let logits = fixture.input("logits_i8", |b| b[0] as i8);
        fixture.check("edgefirst_json_i8", &logits, "_i8");
    }
}
