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
│   │   ├── helper.rs                   # Per-scale helper math unit tests
│   │   └── kernels/                    # Level orchestrators over edgefirst-kernels
│   │       ├── dispatch.rs             # Kernel selection per encoding/dtype/tier
│   │       ├── dfl_lut_tests.rs        # Exp-table DFL decode vs the softmax
│   │       │                           #   level kernels and an f64 reference
│   │       ├── tier_tests.rs           # Score/mask-coef/proto cells per tier
│   │       ├── box_tier_tests.rs       # DFL (per-tensor, per-channel) and
│   │       │                           #   LTRB box cells per tier
│   │       └── level_box.rs / level_score.rs / level_mc.rs
│   ├── instance_mask/tests.rs          # Materialised masks, proto dequant,
│   │                                   #   conversions and threshold per tier
│   ├── neon_dispatch/                  # Candidate filters, dequantize, NMS,
│   │   │                               #   ModelPack and semantic per tier
│   │   ├── nms/tests.rs                # Every public NMS function vs scalar
│   │   ├── modelpack/tests.rs          # Split decode, seg maps, argmax
│   │   └── semantic/tests.rs           # Semantic decodes vs the scalar tier
│   ├── tiling.rs                       # SAHI lift/merge unit + doc-tests
│   ├── schema.rs                       # SchemaV2 round-trip + fixture tests
│   ├── infer.rs                        # Ultralytics schema inference:
│                                       #   metadata parsing, layout rules,
│                                       #   and per-fixture expectations over
│                                       #   testdata/infer/*.signals.json
│   └── model_config/                   # Config discovery: ONNX/ZIP readers
│                                       #   (hand-built and adversarial
│                                       #   containers), HF configs from
│                                       #   testdata/hf/, override
│                                       #   grammar round-trips, precedence,
│                                       #   heuristics and describe() lines
├── tests/                              # Cross-cutting integration suites
│   ├── decoder_capacity.rs
│   ├── decoder_decode_vs_proto_parity.rs
│   ├── decoder_from_edgefirst_json.rs
│   ├── decoder_modelpack_multitask.rs
│   ├── decoder_normalized_flag.rs
│   ├── infer_builder.rs                # Inferred schemas round-trip through
│                                       #   DecoderBuilder
│   ├── modelpack_coffeecup_parity.rs
│   ├── modelpack_decoder_schemas.rs
│   ├── multi_label_paths.rs            # Multi-label on every NMS decode path;
│                                       #   tracked decode stays argmax;
│                                       #   NMS mode and bypass max_det
│   ├── per_scale_parity.rs
│   ├── pre_nms_top_k.rs                # pre-NMS cap: 300 default, 0 unbounded,
│                                       #   30 000 under multi-label
│   └── common/                         # Shared fixture loaders
└── benches/
    └── decoder_benchmark.rs
```

## Running Tests

All `cargo test` invocations below pass `-- --test-threads=1` per the
workspace single-threaded rule
([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)).
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
cargo test -p edgefirst-decoder --doc -- --test-threads=1
```

## Special Requirements

- **LFS testdata** — model fixtures live under `testdata/` and are tracked
  via Git LFS. Tests load them through `edgefirst_bench::testdata::read*`,
  which honours the `EDGEFIRST_TESTDATA_DIR` environment variable. CI sets
  this to `${{ github.workspace }}/testdata`. Locally, the helper falls back
  to `<workspace>/testdata` so no env var is needed.
- **No hardware gates.** All decoder tests run on any host; there is no GPU
  or NPU dependency.
- **`tracker` feature** — gates the `decode_tracked` test paths. Run
  `cargo test -p edgefirst-decoder --features tracker -- --test-threads=1` to exercise them.
- **Per-scale fixtures** — synthetic schemas under
  [`testdata/per_scale/`](https://github.com/EdgeFirstAI/hal/tree/main/testdata/per_scale)
  validate the per-scale planner against multiple split-tensor topologies
  (yolov8n, yolo26n, flat). The fixture generator is in
  [`crates/decoder/tests/common/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/decoder/tests/common).
- **Decoder fixture framework** — the
  [`crates/decoder/tests/decoder_decode_vs_proto_parity.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/tests/decoder_decode_vs_proto_parity.rs)
  suite validates that `Decoder::decode` and `Decoder::decode_proto`
  produce bit-identical detections for the same inputs (the key invariant
  for the GPU fused mask path).
- **Multi-label coverage** — `multi_label_paths.rs` decodes one synthetic scene (one anchor above threshold for two classes) through every NMS model type in float and quantized form. It checks that argmax output is unchanged and that multi-label adds exactly the second class. With `--features tracker`, it also checks that `decode_tracked` returns the argmax result whether multi-label came from the builder or from `nms_multi_label` metadata. It also checks, per model type, that an explicit class-agnostic mode suppresses across the classes of one anchor under multi-label while class-aware keeps them, and that with NMS bypassed `max_det` keeps the highest-scoring boxes.
- **Pre-NMS cap** — `pre_nms_top_k.rs` decodes 400 disjoint anchors (800 multi-label candidates) to show the default cap of 300 on a detection-only model, `0` as unbounded, and the 30 000 default when multi-label comes from the builder or metadata.

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

`--section <name>` runs one section (for example `--section per-scale-realistic`). The `Per-Scale-Realistic` section decodes synthetic int8 outputs shaped like captured yolov8n frames for a 960x544 four-class model and a 640x640 COCO model, densely, through the score prefilter, and end to end, and prints process CPU time per iteration (rayon workers included) beside the wall time. `DFL-Kernels` (`decoder/per_scale/dfl_kernel/*`) times each 8-bit DFL box kernel on one 120x68 level through the hidden `per_scale::bench` hooks and prints nanoseconds per anchor. `Per-Scale-Seg` runs the 640x640 yolov8-seg model through `decode_proto` with f32 and f16 outputs, the f16 run including the widen to f32. `Per-Scale-Cells` times every score, mask-coefficient and proto cell (`decoder/per_scale/cell/<role>/<in>_<out>/<tier>`, 8400x80 scores, 8400x32 mask coefficients, 160x160x32 protos), the f16 widen (`widen_f16/*`) and the prefilter's anchor selection (`select_anchors/*`) on each tier the CPU has, and prints the kernel cell each tier selects. `Per-Scale-Boxes` does the same for every box cell (`decoder/per_scale/box/<encoding>/<in>_<out>/<tier>` over one 105x80 level: DFL `reg_max` 16 per-tensor and per-channel, DFL `reg_max` 8, and LTRB). `Masks` times `decode` with materialised masks on the yolov8-seg fixture (`decoder/masks/{i8,f32}`) and on synthetic outputs with exactly 10, 50 and 100 surviving 0.2 x 0.2 detections over the fixture's 160x160x32 protos (`decoder/masks/{i8,u8,f32}/n<count>`), plus `decode_proto` at 50 detections with i8 protos (kept raw) and u8 protos (dequantized to f32) (`decoder/masks/decode_proto/{i8,u8}/n50`); set `EDGEFIRST_DECODER_FORCE_KERNEL` to time one tier. `Quant` also times NMS over 6 and 300 synthetic candidates (`decoder/nms/*`: clustered class-agnostic, class-aware and `i8`, and disjoint boxes that all survive), and `ModelPack` the split decode of one 40×40×3×85 output (`split_{u8,f32}_40x40x255`), segmentation decodes of a 8×240×320 NCHW output (`seg_{i8,f32,f16}_8x240x320`) and `segmentation_to_mask` (`seg_to_mask_*`).

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
- `per_scale::kernels::tier_tests` and `box_tier_tests` run every score,
  mask-coefficient, proto and box cell on every tier the CPU has (forced
  through `EDGEFIRST_DECODER_FORCE_KERNEL` names), check that NEON tiers
  select no scalar cell, and compare each cell with the scalar tier, the
  decoder's earlier NEON composition or the float64 reference, with the
  bounds stated at the top of each file; box cells must also reject
  quantization and slice lengths that do not match the level before
  writing anything.
- `instance_mask::tests` decodes materialised masks on every tier for
  every pair of `i8`/`u8`/`i16`/`u16` coefficients and protos (plus `i32`,
  which runs scalar), `f32`/`f16`/`f64`, NHWC and NCHW protos, K of 32, 7
  and 40, and boxes covering the frame, its edges, single cells, empty,
  inverted and out-of-frame crops. It checks the kernel each tier selects,
  that NEON masks are within 1 of the scalar tier, and that the scalar
  tier's quantized masks equal an ndarray `mapv` + `dot` reference bit for
  bit; zero points outside the type's range and other proto strides must
  run scalar. Proto dequantize (NHWC and NCHW), `dequant_3d!`/`dequant_4d!`
  (values and strides, contiguous, permuted, sliced and reversed views) and
  the mask threshold must equal the scalar tier bit for bit.
- `neon_dispatch::nms::tests` runs the eight public NMS functions on every tier against the scalar tier, box for box and in order (12 box counts up to 333, clustered, identical, zero-area, inverted and NaN/infinite boxes, negative, NaN and minimum scores, mixed classes, six thresholds, four caps) checks that NEON tiers run the kernel from 16 boxes on (20 in a direct call) and that `jaccard_batch4` follows the forced tier. `neon_dispatch::modelpack::tests` compares the ModelPack split decode (`f32`/`f16`/`f64` arrays, the fused `u8`/`i8`/`u16`/`i16` dequantize, and a full 40×40 + 20×20 × 3 × 85 decode with NMS), the segmentation conversions and normalization (every type, planar, Fortran and reversed layouts) and `segmentation_to_mask` (2–21, 32, 33, 255 and 300 channels, against `argminmax`) with the scalar tier bit for bit, checks the kernel each tier selects, and that ModelPack decodes fail on an unavailable forced tier while `segmentation_to_mask` runs scalar.
- `neon_dispatch::semantic::tests` decodes on every tier against the scalar tier: every score type (`u8`, `i8`, `u16`, `i16`, `f32`, `f16`, with NaN, infinite and signed-zero floats and tie-heavy maps) in NCHW, NHWC, NHCW and NWHC order, 1–300 classes, random map and output sizes (odd widths, up- and downsampling), per-tensor and per-channel quantization, both modes and upsampling methods, `align_corners`, a letterbox region, `u8`/`u16` class maps and `f32`/`u8` confidence of every kind. Class maps and confidence must be equal, except the sigmoid confidence: within 2⁻²⁰ (`f32`), or 1 (`u8`) only where the scalar value rounds within that bound of a boundary. It also compares class-map inputs (`u8`, `u16`, `i32`, `u32`, `i64`, values out of range included, same error), checks that a decode gives the same bytes on 1, 2 and 5 rayon threads, the kernel each tier selects, and that a semantic decode fails with `ForcedKernelUnavailable` on an unavailable or unknown forced tier.
- The compute primitives themselves (dequant, sigmoid, softmax, DFL, box
  decode, transpose, the NEON tiers) live in `edgefirst-kernels`, whose
  scalar-vs-NEON parity tests are described in
  [`../kernels/TESTING.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/TESTING.md).
- Doc-tests on the public API surface (e.g. `DecoderBuilder` methods) are
  marked `no_run` because they require model fixtures to load. The
  underlying logic is exercised by the unit tests in `decoder/tests.rs`.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Image-side mask rendering tests: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Tracker tests: [../tracker/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/TESTING.md)
