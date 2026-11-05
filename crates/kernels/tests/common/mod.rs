// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the kernel integration tests: fixture loading, CPU tiers and pool checks.

#![allow(dead_code)] // Each test binary uses a subset.

use std::num::NonZeroUsize;

use edgefirst_kernels::{CpuFeatures, Pool, Quant, ScopedThreads};

/// A single-op TFLite reference case: raw little-endian input and expected output bytes.
pub struct Fixture {
    pub op: String,
    pub input: Vec<u8>,
    pub output: Vec<u8>,
    pub shape: Vec<usize>,
    pub input_quant: Quant,
    pub output_quant: Quant,
    pub beta: f32,
}

/// Loads `testdata/kernels/<name>.safetensors`.
pub fn load(name: &str) -> Fixture {
    let path = edgefirst_bench::testdata::path(format!("kernels/{name}.safetensors"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .unwrap_or_else(|e| panic!("metadata {name}: {e}"));
    let meta = header
        .metadata()
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: no metadata"));
    let get = |key: &str| -> &str {
        meta.get(key)
            .unwrap_or_else(|| panic!("{name}: missing metadata `{key}`"))
    };
    let f32_of = |key: &str| -> f32 {
        get(key)
            .parse()
            .unwrap_or_else(|_| panic!("{name}: bad `{key}`"))
    };
    let i32_of = |key: &str| -> i32 {
        get(key)
            .parse()
            .unwrap_or_else(|_| panic!("{name}: bad `{key}`"))
    };

    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("deserialize {name}: {e}"));
    let tensor = |key: &str| -> Vec<u8> {
        tensors
            .tensor(key)
            .unwrap_or_else(|e| panic!("{name}: tensor `{key}`: {e}"))
            .data()
            .to_vec()
    };

    let shape: Vec<usize> = get("shape")
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|d| {
            d.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name}: bad shape"))
        })
        .collect();

    Fixture {
        op: get("op").to_string(),
        input: tensor("input"),
        output: tensor("output"),
        shape,
        input_quant: Quant::new(f32_of("input_scale"), i32_of("input_zero_point")),
        output_quant: Quant::new(f32_of("output_scale"), i32_of("output_zero_point")),
        beta: meta.get("beta").map_or(1.0, |_| f32_of("beta")),
    }
}

/// A multi-tensor TFLite reference case: named tensors with their shapes, and string metadata.
pub struct Graph {
    pub name: String,
    meta: std::collections::HashMap<String, String>,
    tensors: std::collections::HashMap<String, (Vec<usize>, Vec<u8>)>,
}

impl Graph {
    /// A metadata entry.
    pub fn meta(&self, key: &str) -> &str {
        self.meta
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing metadata `{key}`", self.name))
    }

    /// The `<role>_scale` and `<role>_zero_point` metadata.
    pub fn quant(&self, role: &str) -> Quant {
        let scale = self.meta(&format!("{role}_scale"));
        let zp = self.meta(&format!("{role}_zero_point"));
        Quant::new(
            scale
                .parse()
                .unwrap_or_else(|_| panic!("{}: bad {role} scale", self.name)),
            zp.parse()
                .unwrap_or_else(|_| panic!("{}: bad {role} zero point", self.name)),
        )
    }

    /// A metadata entry holding a JSON list of numbers.
    pub fn f32_list(&self, key: &str) -> Vec<f32> {
        self.meta(key)
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|v| {
                v.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("{}: bad `{key}`", self.name))
            })
            .collect()
    }

    pub fn shape(&self, key: &str) -> &[usize] {
        &self.raw(key).0
    }

    pub fn i8(&self, key: &str) -> Vec<i8> {
        self.raw(key).1.iter().map(|&b| b as i8).collect()
    }

    pub fn i32(&self, key: &str) -> Vec<i32> {
        self.raw(key)
            .1
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn raw(&self, key: &str) -> &(Vec<usize>, Vec<u8>) {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing tensor `{key}`", self.name))
    }
}

/// Loads `testdata/kernels/<name>.safetensors` with every tensor and metadata entry.
pub fn load_graph(name: &str) -> Graph {
    let path = edgefirst_bench::testdata::path(format!("kernels/{name}.safetensors"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .unwrap_or_else(|e| panic!("metadata {name}: {e}"));
    let meta = header
        .metadata()
        .clone()
        .unwrap_or_else(|| panic!("{name}: no metadata"));
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("deserialize {name}: {e}"));
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(k, v)| (k, (v.shape().to_vec(), v.data().to_vec())))
        .collect();
    Graph {
        name: name.to_string(),
        meta: meta.into_iter().collect(),
        tensors,
    }
}

/// Elements of `a` and `b` that differ, and the largest difference.
pub fn diff_i8(a: &[i8], b: &[i8]) -> (usize, i32) {
    assert_eq!(a.len(), b.len(), "lengths differ");
    a.iter().zip(b).fold((0, 0), |(n, worst), (&x, &y)| {
        let d = (i32::from(x) - i32::from(y)).abs();
        (n + usize::from(d != 0), worst.max(d))
    })
}

/// Scalar plus every NEON tier the probed CPU supports.
pub fn tiers() -> Vec<CpuFeatures> {
    ["scalar", "neon", "neon_fp16", "neon_dotprod"]
        .into_iter()
        .filter_map(|name| CpuFeatures::from_forced_tier(Some(name)).ok())
        .collect()
}

/// A pool that runs tasks in index order on the calling thread.
pub struct SequentialPool;

impl Pool for SequentialPool {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        (0..n_tasks).for_each(task);
    }
}

/// A pool that calls every index twice, from two threads at once.
pub struct TwicePool;

impl Pool for TwicePool {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        std::thread::scope(|s| {
            s.spawn(|| (0..n_tasks).for_each(task));
            (0..n_tasks).rev().for_each(task);
        });
    }
}

/// A pool that calls out-of-range indices and only every other valid one.
pub struct OutOfRangePool;

impl Pool for OutOfRangePool {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        for i in [usize::MAX, n_tasks, n_tasks + 1, usize::MAX / 2] {
            task(i);
        }
        (0..n_tasks).step_by(2).for_each(task);
    }
}

/// A pool that runs nothing.
pub struct IdlePool;

impl Pool for IdlePool {
    fn run(&self, _: usize, _: &(dyn Fn(usize) + Sync)) {}
}

/// Checks that `run` yields identical bytes with no pool, several pools (misbehaving ones too) and concurrent callers.
pub fn assert_pool_and_concurrency_match<F: Fn(Option<&dyn Pool>) -> Vec<u8> + Sync>(run: F) {
    let threads = |n: usize| ScopedThreads::new(NonZeroUsize::new(n).expect("non-zero"));
    let reference = run(None);
    let (three, many) = (threads(3), threads(64));
    assert_eq!(run(Some(&three)), reference, "ScopedThreads(3)");
    assert_eq!(run(Some(&SequentialPool)), reference, "sequential pool");
    assert_eq!(run(Some(&many)), reference, "ScopedThreads(64)");
    assert_eq!(
        run(Some(&TwicePool)),
        reference,
        "pool calling each task twice"
    );
    assert_eq!(
        run(Some(&OutOfRangePool)),
        reference,
        "pool calling out-of-range tasks"
    );
    assert_eq!(run(Some(&IdlePool)), reference, "pool running nothing");
    std::thread::scope(|s| {
        let a = s.spawn(|| run(None));
        let b = s.spawn(|| run(Some(&three)));
        assert_eq!(a.join().expect("thread a"), reference, "concurrent a");
        assert_eq!(b.join().expect("thread b"), reference, "concurrent b");
    });
}
