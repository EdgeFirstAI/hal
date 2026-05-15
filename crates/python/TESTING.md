# edgefirst-hal (Python) Testing

## Test Layout

```
tests/                          # Project-level Python tests
├── test_tensor.py              # Tensor binding coverage (single file)
├── image/                      # ImageProcessor tests
├── decoder/                    # Decoder tests
├── python/                     # PyO3 binding-specific edge cases
├── bench_decode_render.py      # Decoder + draw_decoded_masks benchmark
├── profile_decode_render.py    # Hot-loop profiling target for `perf record`
└── example_seg_pipeline.py     # End-to-end pipeline example
```

Tensor tests live as a single file (`tests/test_tensor.py`) rather than
under `tests/tensor/`; the binding surface for `Tensor` is narrow enough
that a flat file is easier to navigate than a directory.

The Python crate's source under
[`crates/python/src/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/python/src)
contains the PyO3 binding layer; the actual test code lives at the
workspace root in `tests/` so it can drive the installed `edgefirst_hal`
package the same way an end user would.

## Running Tests

```bash
# 1. Activate the project venv (per global rule — never install into
#    the system Python)
source venv/bin/activate

# 2. Build and install the bindings into the venv in development mode
maturin develop -m crates/python/Cargo.toml

# 3. Run the full Python suite
python -m pytest tests/

# 4. Single test module
python -m pytest tests/image/test_image.py
python -m pytest tests/decoder/test_decoder.py

# 5. With slipcover for coverage (preferred over coverage.py for Rust+PyO3)
python -m slipcover --xml --out target/python-coverage.xml -m pytest tests/
```

The Makefile target `make test-python` wraps steps 2–4. CI uses the same
target; coverage from slipcover is uploaded to SonarCloud alongside the
Rust lcov coverage.

## Special Requirements

- **Use a Python virtual environment.** Never `pip install` into the
  system Python; activate the `venv/` directory at the workspace root
  before running the Python tooling. The Makefile targets assume `venv/`
  is the active environment.
- **`maturin develop` rebuilds the Rust shared object** when the Rust
  source changes. Run it after any change under `crates/` or after pulling
  new commits before running tests.
- **Single-threaded execution** — the Python suite inherits the
  [project-wide single-threaded rule](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)
  whenever GL or G2D code paths are exercised. `pytest -p no:randomly` is
  recommended; the Makefile already disables parallelism.
- **LFS testdata** — fixtures live under `testdata/`. The Python tests
  resolve paths relative to the workspace root; no env var is needed
  locally. CI sets `EDGEFIRST_TESTDATA_DIR` for parity with the Rust
  bench harness.
- **Hardware gates** — GL/G2D tests skip themselves on hosts without the
  required device nodes (mirroring the Rust-side `OnceLock` probes). On
  the i.MX 8M Plus and i.MX 95 hardware runners the full hardware path is
  exercised.
- **abi3 wheels** — `.github/workflows/test.yml` builds the Python
  binding with `--features abi3-py311` for the CI test legs (including
  the `Hardware Test (imx8mp)` job that downloads the wheel into a
  fresh venv). The release pipeline additionally builds an
  `abi3-py38` variant for broader compatibility. When testing against
  a release wheel locally, install it into a clean venv and point
  `pytest` at that environment.

## Benchmarks

```bash
# 2-step path: decode() then draw_decoded_masks()
python tests/bench_decode_render.py --iterations 100

# Single-call fused decode+draw
python tests/bench_decode_render.py --fused --iterations 100

# JSON output for tracking in CI
python tests/bench_decode_render.py --json results.json
```

For sampling profiles:

```bash
# Records the fused hot loop with full call stacks
perf record -F 997 --call-graph dwarf -- \
  python tests/profile_decode_render.py fused
```

`profile_decode_render.py` is intentionally separated from
`bench_decode_render.py` so that setup (model load, EGL init) does not
appear in the sampled profile.

## Coverage Notes

- Python coverage uses [`slipcover`](https://github.com/plasma-umass/slipcover)
  rather than `coverage.py` — slipcover handles native extensions
  (Rust/PyO3) without losing line-level attribution.
- The XML output (`target/python-coverage.xml`) is merged with the Rust
  lcov by the
  [`Process Hardware Coverage`](https://github.com/EdgeFirstAI/hal/blob/main/.github/workflows/test.yml)
  job and uploaded to SonarCloud.
- Test discovery uses pytest's default conventions — no `conftest.py`
  magic. Test classes that need shared fixtures use module-level
  `@pytest.fixture(scope="module")` to amortize processor / decoder
  construction across cases.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Image-side GL gating: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- Decoder testing: [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
