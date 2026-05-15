# edgefirst-decoder Testing

## Test Layout

```
crates/decoder/
├── src/
│   ├── lib.rs                          # Doc-tests on the public API surface
│   ├── decoder/
│   │   ├── tests.rs                    # Builder, segmentation, parity tests
│   │   ├── builder.rs                  # Doc-tests on DecoderBuilder methods
│   │   └── config.rs                   # Doc-tests on ConfigOutputs
│   ├── per_scale/
│   │   ├── pipeline.rs                 # Per-scale pipeline correctness tests
│   │   ├── plan.rs                     # Schema → Plan validation tests
│   │   └── helper.rs                   # NEON kernel unit tests
│   └── schema.rs                       # SchemaV2 round-trip + fixture tests
├── tests/                              # Cross-cutting integration suites
│   ├── decoder_capacity.rs
│   ├── decoder_decode_vs_proto_parity.rs
│   ├── decoder_from_edgefirst_json.rs
│   ├── decoder_normalized_flag.rs
│   ├── per_scale_parity.rs
│   └── common/                         # Shared fixture loaders
└── benches/
    └── decoder_benchmark.rs
```

## Running Tests

All `cargo test` invocations below pass `-- --test-threads=1` per the
workspace single-threaded rule
([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#why-single-threaded-execution)).
Decoder tests are CPU-only, but the rule applies workspace-wide so commands
stay consistent and copy-pasteable across crates.

```bash
# Full suite (unit + integration + doc-tests)
cargo test -p edgefirst-decoder -- --test-threads=1

# Just the integration tests
cargo test -p edgefirst-decoder --tests -- --test-threads=1

# Single integration test
cargo test -p edgefirst-decoder --test per_scale_parity -- --test-threads=1

# Doc-tests only
cargo test -p edgefirst-decoder --doc
```

## Special Requirements

- **LFS testdata** — model fixtures live under `testdata/` and are tracked
  via Git LFS. Tests load them through `edgefirst_bench::testdata::read*`,
  which honors the `EDGEFIRST_TESTDATA_DIR` environment variable. CI sets
  this to `${{ github.workspace }}/testdata`. Locally, the helper falls back
  to `<workspace>/testdata` so no env var is needed.
- **No hardware gates.** All decoder tests run on any host; there is no GPU
  or NPU dependency.
- **`tracker` feature** — gates the `decode_tracked` test paths. Run
  `cargo test -p edgefirst-decoder --features tracker` to exercise them.
- **Per-scale fixtures** — synthetic schemas under
  [`testdata/per_scale/`](https://github.com/EdgeFirstAI/hal/tree/main/testdata/per_scale)
  validate the per-scale planner against multiple split-tensor topologies
  (yolov8n, yolo26n, flat). The fixture generator is in
  [`crates/decoder/tests/common/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/decoder/tests/common).
- **Decoder fixture framework** — the
  [`crates/decoder/tests/decoder_decode_vs_proto_parity.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/tests/decoder_decode_vs_proto_parity.rs)
  suite validates that `decode_quantized` and `decode_quantized_proto`
  produce identical detections for the same inputs (the key invariant for
  the GPU fused mask path).

## Benchmarks

```bash
# Native-host bench
cargo bench -p edgefirst-decoder

# Cross-compile for a target (see BENCHMARKS.md for the deploy workflow)
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release \
  -p edgefirst-decoder --bench decoder_benchmark
```

| Benchmark | Source | What it measures |
|-----------|--------|------------------|
| `decoder_benchmark` | [`benches/decoder_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/benches/decoder_benchmark.rs) | End-to-end decode latency: detection-only, segmentation, proto-data, per-scale split-tensor; both quantized and float paths |

The benchmark uses the project's standard `--bench` flag for host runs.
The aarch64 cross-compile flow plus `scp` + on-target run procedure is
documented in
[`BENCHMARKS.md`](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md).

## Coverage Notes

- The crate participates in workspace `cargo llvm-cov`; lcov output appears
  under `crates/decoder/`.
- The split-tensor framework's NEON kernels are exercised by
  `per_scale_parity` against a scalar reference implementation in the same
  test file — this is the primary correctness gate for the optimized path.
- Doc-tests on the public API surface (e.g. `DecoderBuilder` methods) are
  marked `no_run` because they require model fixtures to load. The
  underlying logic is exercised by the unit tests in `decoder/tests.rs`.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Image-side mask rendering tests: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Tracker tests: [../tracker/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md)
