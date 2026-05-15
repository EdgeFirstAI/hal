# edgefirst-hal Architecture

## Overview

`edgefirst-hal` is the umbrella crate for the EdgeFirst Hardware Abstraction
Layer. Its job is to give downstream applications a single dependency that
re-exports the four core HAL crates, plus an optional process-wide tracing
subscriber for end-to-end performance analysis. The umbrella owns no domain
types of its own — every type a user interacts with is defined in one of the
sub-crates.

## Module Map

| Module | Source | Re-exports / owns |
|--------|--------|-------------------|
| `edgefirst_hal::tensor` | `edgefirst-tensor` | Zero-copy tensor memory (DMA, SHM, PBO, system memory) |
| `edgefirst_hal::image` | `edgefirst-image` | Hardware-accelerated image processing (OpenGL, G2D, CPU) |
| `edgefirst_hal::decoder` | `edgefirst-decoder` | ML model output decoding (YOLOv5/v8/v11/v26, ModelPack) |
| `edgefirst_hal::tracker` | `edgefirst-tracker` (feature `tracker`) | ByteTrack multi-object tracking |
| `edgefirst_hal::trace` | local (feature `tracing`) | Process-wide span subscriber + Chrome/Perfetto trace capture |

## Key Types

The umbrella owns only the tracing API:

- [`trace::start_tracing(path)`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/src/trace.rs) — install the global subscriber and begin writing spans to `path`.
- [`trace::stop_tracing()`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/src/trace.rs) — flush and close the trace file.

All other public types live in the sub-crates. See each sub-crate's
ARCHITECTURE.md for the type-level detail.

## Internal Architecture

### Re-export layer

`crates/hal/src/lib.rs` is a thin file (~12 lines) that does only `pub use`
re-exports. The `tracker` re-export is gated on the `tracker` Cargo feature
because tracking is an optional capability that pulls in additional
dependencies. The `trace` module is gated on the `tracing` feature.

This shape means the umbrella has zero runtime cost over depending directly on
each sub-crate; it exists only for ergonomics (one dependency, one version
number, one feature surface).

### Tracing subscriber

The optional `trace` module installs a process-wide
[`tracing-subscriber`](https://docs.rs/tracing-subscriber) once, on the first
call to `start_tracing`. The subscriber consists of a
[`tracing-chrome`](https://docs.rs/tracing-chrome) layer that writes spans to a
JSON file viewable at <https://ui.perfetto.dev/>.

Sub-crate hot paths emit `tracing::trace_span!` spans with near-zero overhead
when no subscriber is active (a single relaxed atomic load per span site). The
umbrella's tracing layer is what activates them. See
[Performance Tracing](https://github.com/EdgeFirstAI/hal/blob/main/README.md#performance-tracing)
in the project README for the user-facing guide and span name reference.

Only one trace capture session is supported per process lifetime — this is a
limitation of Rust's global subscriber model and is acceptable for profiling
workflows where a single trace per run is the norm.

## Inter-Crate Interfaces

The umbrella has no inter-crate interfaces beyond the re-exports listed in the
Module Map. Sub-crate inter-dependencies are documented in the
[Internal Dependency Graph](https://github.com/EdgeFirstAI/hal/blob/main/README.md#internal-dependency-graph)
section of the project README.

## Cross-References

- Project architecture: [../../ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md)
- Tracing usage: [README.md#performance-tracing](https://github.com/EdgeFirstAI/hal/blob/main/README.md#performance-tracing)
- Sub-crate architectures:
  [tensor](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md),
  [image](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md),
  [decoder](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md),
  [tracker](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md)
