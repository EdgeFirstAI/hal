# EdgeFirst HAL Python Bindings — Testing

Covers all four published wheels (`edgefirst-tensor`, `edgefirst-codec`,
`edgefirst-image`, `edgefirst-decoder`) and the `python-common` rlib they
share. Since 0.29 there is no single `edgefirst_hal` module — the bindings
are four independent extension modules under the `edgefirst.` PEP 420
namespace, so testing has to cover both the per-package behaviour *and* the
packaging invariants that keep them installable independently.

## Test Layout

```
tests/                          # Workspace-root Python tests
├── test_tensor.py              # Tensor binding coverage (single file)
├── image/                      # ImageProcessor tests
├── decoder/                    # Decoder tests
├── python/                     # PyO3 binding-specific edge cases
├── packaging/                  # NEW in 0.29 — the split's invariants
│   ├── test_namespace.py       # PEP 420: no edgefirst/__init__.py anywhere
│   ├── test_wheel_layout.py    # self-contained wheels, py.typed, abi3 tags
│   ├── test_size_baseline.py   # codec-only install stays small
│   └── check_coverage_split.sh # per-package coverage attribution
├── bench_decode_render.py      # Decoder + draw_decoded_masks benchmark
└── profile_decode_render.py    # Hot-loop profiling target for `perf record`
```

Test code lives at the workspace root rather than beside the crates so it
drives the *installed* packages exactly as an end user would — which is the
only way the namespace-package behaviour is actually exercised.

## Running Tests

```bash
# 1. Activate the project venv (per global rule — never install into
#    the system Python)
source venv/bin/activate

# 2. Build and install all four bindings in development mode
make build-python
#    or one at a time:
maturin develop -m crates/python-tensor/Cargo.toml

# 3. Run the full Python suite
python -m pytest tests/

# 4. Packaging gates only (fast; no hardware needed)
python -m pytest tests/packaging/
python3 scripts/check_wheel_layout.py target/wheels

# 5. With slipcover for coverage (preferred over coverage.py for Rust+PyO3)
python -m slipcover --xml --out target/python-coverage.xml -m pytest tests/
```

`make test-python` wraps steps 2, 3 and 5. It does **not** wrap step 4 —
`make wheels` runs the layout gate after building.

## What the packaging gates protect

These are cheap and catch the failure modes that the split introduced. They
are not optional extras; each one corresponds to a bug already hit once.

| Gate | Failure it catches |
|------|--------------------|
| `test_namespace.py` | An `edgefirst/__init__.py` in any wheel — one regular package shadows the namespace and makes the other three unimportable |
| `test_wheel_layout.py` | A wheel that is not self-contained, or is missing `py.typed`/stubs |
| abi3 tag-set check | Mixed `cp38-abi3` and `cp311-abi3` across packages, which forks the supported interpreter range silently. The comparison must include the **python** tag — `cp38-abi3` and `cp311-abi3` share abi and platform tags, so comparing only those two passes a genuinely mixed set |
| `test_size_baseline.py` | A codec-only install regaining a dependency on `edgefirst-image` |

## Special Requirements

- **Use a Python virtual environment.** Never `pip install` into the system
  Python; activate `venv/` at the workspace root first.
- **`maturin develop` rebuilds the Rust shared object** when the Rust source
  changes. Re-run after any change under `crates/`.
- **Per-module type identity is not shared.** Each extension module caches
  its own `#[pyclass]` type objects, so `edgefirst.tensor.Tensor is
  edgefirst.image.Tensor` is **False** by design. Never write an
  `isinstance`/`is` assertion across packages — cross-package handoff goes
  through the `__edgefirst_tensor__` capsule protocol and is duck-typed on
  purpose. A test that asserts type identity across two wheels is asserting
  the thing the architecture deliberately gave up.
- **Single-threaded execution** — the Python suite inherits the
  [project-wide single-threaded rule](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)
  whenever GL or G2D paths are exercised. `make test-python` invokes plain
  `pytest`, so it is already serial; don't add `-n`.
- **LFS testdata** — fixtures live under `testdata/`. Tests resolve paths
  relative to the workspace root; on-target runs export
  `EDGEFIRST_TESTDATA_DIR`.
- **Hardware gates** — GL/G2D tests skip on hosts without the required
  device nodes, mirroring the Rust `OnceLock` probes. See
  [the root TESTING.md on-target section](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#on-target-tests);
  run them for real with `./scripts/on-target-test.sh`.
- **abi3 wheels** — CI builds with `--features abi3-py311`; the release
  pipeline additionally builds an `abi3-py38` variant. All five packages
  must agree.

## Benchmarks

```bash
python tests/bench_decode_render.py --iterations 100      # 2-step decode + draw
python tests/bench_decode_render.py --fused --iterations 100
python tests/bench_decode_render.py --json results.json

perf record -F 997 --call-graph dwarf -- \
  python tests/profile_decode_render.py fused
```

`profile_decode_render.py` is kept separate from `bench_decode_render.py` so
setup (model load, EGL init) does not appear in the sampled profile.

## Coverage Notes

- Python coverage uses [`slipcover`](https://github.com/plasma-umass/slipcover)
  rather than `coverage.py` — it handles native extensions without losing
  line-level attribution.
- `tests/packaging/check_coverage_split.sh` verifies coverage is attributed
  to the right package now that one `.so` no longer covers everything.
- XML output merges with the Rust lcov and uploads to SonarCloud.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Image-side GL gating: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Decoder testing: [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
