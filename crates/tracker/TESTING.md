# edgefirst-tracker Testing

## Test Layout

```
crates/tracker/
├── src/
│   ├── lib.rs          # Doc-tests for Tracker / DetectionBox traits
│   ├── bytetrack.rs    # Unit tests for ByteTrackBuilder, association
│   └── kalman.rs       # Unit tests for the Kalman filter math
└── benches/
    └── tracker_benchmark.rs   # End-to-end ByteTrack throughput
```

The crate has no `tests/` directory — all coverage comes from `#[test]`
modules colocated with the source they exercise, plus doc-tests on the public
API.

## Running Tests

```bash
# All tracker tests
cargo test -p edgefirst-tracker

# Doc-tests only
cargo test -p edgefirst-tracker --doc

# Single test
cargo test -p edgefirst-tracker bytetrack::tests::two_pass_recovery
```

## Special Requirements

- **No hardware gates.** The tracker is pure CPU; tests run identically on
  any host.
- **No LFS testdata.** Detections are synthesized in-test (random walks
  through a frame to exercise association and lifecycle paths).
- **No feature flags.** All tests run under default features.

## Benchmarks

Run benchmarks from the workspace root:

```bash
# Native-host benchmark
cargo bench -p edgefirst-tracker

# Cross-compile for a target (see BENCHMARKS.md for the deploy workflow)
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release \
  -p edgefirst-tracker --bench tracker_benchmark
```

| Benchmark | Source | What it measures |
|-----------|--------|------------------|
| `tracker_benchmark` | [`benches/tracker_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/benches/tracker_benchmark.rs) | `ByteTrack::update` throughput vs. number of simultaneous tracks |

The benchmark synthesizes its detection corpus internally (see
`generate_detections` in the bench source); it does not load LFS
testdata.

## Coverage Notes

- The crate participates in workspace `cargo llvm-cov`; lcov output is
  produced under `crates/tracker/`.
- The Kalman filter math is exercised both by direct unit tests (numeric
  identities) and indirectly by `tracker_benchmark` and the bytetrack
  association tests.
- No `#[cfg(test)]`-only branches affect production builds.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Decoder testing (consumer of `Tracker`): [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
