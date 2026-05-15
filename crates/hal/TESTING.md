# edgefirst-hal Testing

## Test Layout

The umbrella crate has no unit tests of its own; all test coverage comes from
the sub-crates it re-exports. The two narrow concerns the umbrella tests
itself are:

- **Doc-tests** for the `trace::start_tracing` / `stop_tracing` API in
  [`crates/hal/src/trace.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/src/trace.rs).
- **Compile-time feature-flag conditionals** are exercised by building the
  crate with each combination of `default`, `tracker`, and `tracing` enabled.

## Running Tests

```bash
# Default features (ndarray, opengl, tracing)
cargo test -p edgefirst-hal

# All features
cargo test -p edgefirst-hal --all-features

# No default features (verifies the lib compiles minimally)
cargo test -p edgefirst-hal --no-default-features
```

The doc-tests for `trace::*` are marked `no_run` because installing a global
tracing subscriber has process-lifetime side effects; they verify only that
the example code compiles against the public API.

## Special Requirements

- The `tracing` feature pulls in `tracing-subscriber` and `tracing-chrome`.
  CI builds with `--features tracing` so doc-tests for the trace module
  participate in coverage.
- The `tracker` feature pulls in `edgefirst-tracker` and enables the
  `edgefirst_hal::tracker` re-export. CI exercises this via the workspace
  feature matrix in
  [`.github/workflows/test.yml`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml).

## Benchmarks

The umbrella crate ships no benchmarks of its own — every measurable
hot path is owned by a sub-crate. Use the sub-crate benchmarks
(`tensor_benchmark`, `image_benchmark`, `decoder_benchmark`,
`tracker_benchmark`) for per-component numbers, and combine them with
the `trace` module to capture end-to-end Chrome JSON traces. See the
[Benchmarking](https://github.com/EdgeFirstAI/hal/blob/main/README.md#benchmarking)
and
[Performance Tracing](https://github.com/EdgeFirstAI/hal/blob/main/README.md#performance-tracing)
sections of the project README.

## Coverage Notes

Coverage for the umbrella is collected through the workspace `cargo
llvm-cov` run; the small surface of `lib.rs` re-exports and the `trace`
module's doc-tests show up under `crates/hal/` in the lcov report. The
substantive coverage numbers come from the sub-crates.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Sub-crate testing guides:
  [tensor](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md),
  [image](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md),
  [decoder](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md),
  [tracker](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md)
