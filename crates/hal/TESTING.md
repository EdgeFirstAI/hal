# edgefirst-hal Testing

## Test Layout

The umbrella crate has no unit tests of its own; all test coverage comes from
the sub-crates it re-exports. The two narrow concerns the umbrella tests
itself are:

- **Doc-tests** for the `trace::start_tracing` / `stop_tracing` API in
  [`crates/hal/src/trace.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/src/trace.rs).
- **Compile-time feature-flag conditionals** are exercised in CI by:
  - the workspace doc-test job, which runs with `--all-features`
    (`cargo test --doc --workspace --all-features`); and
  - the per-platform `cargo nextest` jobs, which run with
    `--features default,opencv` on the `--workspace`.

  CI does **not** build the umbrella with `--no-default-features` and
  does not iterate every combination of `default` / `tracker` /
  `tracing`. The `tracker` feature on the umbrella is therefore only
  exercised through the all-features doc-test job; contributors who
  need a specific combination should run it locally (see the commands
  below).

## Running Tests

All `cargo test` invocations below pass `-- --test-threads=1` per the
workspace single-threaded rule
([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)).

```bash
# Default features (ndarray, opengl, tracing)
cargo test -p edgefirst-hal -- --test-threads=1

# All features
cargo test -p edgefirst-hal --all-features -- --test-threads=1

# No default features (verifies the lib compiles minimally)
cargo test -p edgefirst-hal --no-default-features -- --test-threads=1
```

The doc-tests for `trace::*` are marked `no_run` because installing a global
tracing subscriber has process-lifetime side effects; they verify only that
the example code compiles against the public API.

## Special Requirements

- The `tracing` feature pulls in `tracing-subscriber` and `tracing-chrome`.
  CI builds with `--features tracing` so doc-tests for the trace module
  participate in coverage.
- The `tracker` feature pulls in `edgefirst-tracker` and enables the
  `edgefirst_hal::tracker` re-export. CI exercises this only through
  the workspace doc-test job
  (`cargo test --doc --workspace --all-features`) — the per-platform
  `cargo nextest` jobs run with `--features default,opencv`, which
  does **not** include `tracker` (see
  [`.github/workflows/test.yml`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)).
  Run `cargo test -p edgefirst-hal --features tracker -- --test-threads=1`
  locally if you need to exercise the re-export in nextest-style tests.

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

Coverage for the umbrella is collected through the workspace
`cargo llvm-cov nextest` run; the small surface of `lib.rs` re-exports
shows up under `crates/hal/` in the lcov report. Doc-tests are not
included in the llvm-cov report — the workspace `cargo test --doc`
job runs separately and is not coverage-instrumented. The `trace`
module's executable Rust doc-tests are additionally marked `no_run`
(installing a global tracing subscriber has process-lifetime side
effects), so they only **compile** against the public API; they are
not actually executed. The substantive coverage numbers come from
the sub-crates.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Sub-crate testing guides:
  [tensor](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md),
  [image](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md),
  [decoder](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md),
  [tracker](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md)
