# edgefirst-tensor Testing

## Test Layout

```
crates/tensor/
├── src/
│   ├── lib.rs           # Doc-tests on Tensor<T> public API
│   ├── dma.rs           # DMA backend unit tests (Linux-only)
│   ├── shm.rs           # POSIX shared-memory backend unit tests
│   ├── mem.rs           # Heap backend unit tests
│   ├── pbo.rs           # PboTensor + WeakSender behavior
│   ├── iosurface.rs     # IOSurface backend (macOS/iOS) — runs on macOS hosts
│   ├── ahardwarebuffer.rs        # AHardwareBuffer backend (Android) — FFI, on-device only
│   ├── ahardwarebuffer_layout.rs # Pure Android layout/policy logic — host-tested everywhere
│   ├── tensor_dyn.rs    # TensorDyn metadata, multi-plane, format/stride checks
│   └── format.rs        # PixelFormat + DType compatibility
└── benches/
    └── tensor_benchmark.rs   # Allocation + map/unmap throughput
```

There is no `tests/` directory; per-backend coverage lives in the source
files alongside the implementation.

## Running Tests

All `cargo test` invocations below pass `-- --test-threads=1` per the
workspace single-threaded rule
([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)).

```bash
# All tensor tests on the host
cargo test -p edgefirst-tensor -- --test-threads=1

# Heap-only (skips DMA tests when DMA-heap permissions are missing)
EDGEFIRST_TENSOR_FORCE_MEM=1 cargo test -p edgefirst-tensor -- --test-threads=1

# Doc-tests only
cargo test -p edgefirst-tensor --doc -- --test-threads=1

# With ndarray feature exercised explicitly
cargo test -p edgefirst-tensor --features ndarray -- --test-threads=1
```

## Special Requirements

- **DMA tests** require Linux **and** read+write access to a DMA-heap device
  (typically `/dev/dma_heap/linux,cma` or `/dev/dma_heap/system`). On a workstation
  this usually means being in the `video` group. If permissions are missing,
  DMA tests are skipped at runtime and the suite falls back to heap-only
  coverage; nothing fails outright. Set `EDGEFIRST_TENSOR_FORCE_MEM=1` to
  skip even the probe.
- **PBO tests** live in
  [`crates/tensor/src/pbo.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/src/pbo.rs)
  and exercise `PboTensor` against a mock `PboOps` implementation. They
  are not gated on any feature — `cargo test -p edgefirst-tensor` runs
  them on every host. End-to-end PBO behavior (real GL thread, EGL
  context) is additionally exercised by the image crate's tests.
- **IOSurface tests** (`iosurface.rs`) are macOS-gated and run on the
  macOS CI leg: allocation, wrap-by-`IOSurfaceID`, stride/geometry, and
  the read-only lock round-trip (`map_read` locks with
  `kIOSurfaceLockReadOnly`; write through a read map asserts).
- **Android logic is host-tested; Android FFI is device-tested.** Every
  pure decision the AHardwareBuffer backend makes lives in
  `ahardwarebuffer_layout.rs` — the CpuAccess→lock-usage mapping,
  descriptor geometry and gralloc stride handling, packed RGB u8/i8
  RGBA8888 mapping, the `ro.hardware.egl` compression-scheme classifier
  and format-eligibility table, and the `BufferIdentity` intern policy —
  and runs as ordinary host tests on every CI lane, so drift is caught
  without a device. The FFI shell (`ahardwarebuffer.rs`) executes only
  on-device via the internal hal-mobile Device Farm harness (root
  TESTING.md § Android On-Device Validation).
- **CpuAccess map-contract tests** are host-portable: the declared-vs-
  requested map-mode table, the `writable` assert on read-only maps, and
  the `unplanned_cpu_access_count()` warn-once telemetry.
- **No LFS testdata.** All shapes and inputs are synthesized in-test.
  `tensor_benchmark` does not use `edgefirst_bench::testdata`; it
  generates its inputs in memory.
- **No feature flags required by default**; `ndarray` is on by default
  and is required for `view()` / `view_mut()` doc-tests.

## Benchmarks

```bash
# Native-host benchmark
cargo bench -p edgefirst-tensor

# Cross-compile for a target (see the project Makefile / BENCHMARKS.md
# for the full deploy workflow)
cargo-zigbuild zigbuild --target aarch64-unknown-linux-gnu --release \
  -p edgefirst-tensor --bench tensor_benchmark
```

| Benchmark | Source | What it measures |
|-----------|--------|------------------|
| `tensor_benchmark` | [`benches/tensor_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/benches/tensor_benchmark.rs) | Allocation cost across DMA/SHM/Mem backends; map/unmap throughput; clone_fd cost |

## Coverage Notes

- The crate participates in workspace `cargo llvm-cov`; lcov output
  appears under `crates/tensor/`.
- `cfg(target_os = "linux")`-gated code (`dma.rs`, `dmabuf.rs`) is only
  exercised on the Linux host runners and on the i.MX 8M Plus hardware
  runner. The macOS CI leg covers the heap and IOSurface paths.
  `cfg(target_os = "android")` code (`ahardwarebuffer.rs`) is excluded
  from Sonar coverage (no CI lane can execute it) and is covered
  on-device by the hal-mobile harness; its pure logic is host-covered
  via `ahardwarebuffer_layout.rs`. There is no Windows
  test job today; SHM coverage comes from the Linux runners (where
  `shm.rs` is `#[cfg(unix)]`-gated and active).
- The `procfs` dev-dependency on Linux is used to read open-fd counts
  for the current process via `procfs::process::Process::myself()` —
  the leak-regression tests assert that fd count returns to its
  baseline after DMA / SHM tensors are dropped. The check is fd-count
  based, not a `/proc/self/maps` parse.

## CUDA Surface

The CUDA zero-copy surface (`cuda_map()`, `is_cuda_available()`, `CudaMap`,
`CudaHandle`) is fully testable without a GPU or `libcudart`.

### GPU-independent tests (run anywhere)

| Test | Location | What it verifies |
|------|----------|------------------|
| Mock `CudaGlOps` — map/unmap/unregister call sequence | `crates/tensor/src/cuda.rs` `#[cfg(test)]` | `CudaMap` construction calls `map`; `Drop` calls `unmap`; handle drop calls `unregister` in the correct order (before PBO storage drops) |
| `Debug` impl on `CudaMap` / `CudaHandle` | Same | No panics, no format-string UB |
| Primitive degradation — absent `libcudart` | Same | `is_cuda_available()` returns `false`; `cuda_map()` returns `None`; no crash |
| ABI layout asserts | `crates/tensor/src/cuda.rs` (`mod ext_mem_layout`) | `sizeof` / `alignof` of HAL's CUDA type wrappers match CUDA 12.6 `driver_types.h` — verified as static compile-time assertions |

Run all CUDA tests on the host (no hardware required):

```bash
cargo test -p edgefirst-tensor -- --test-threads=1
```

To exercise just the CUDA tests by name:

```bash
cargo test -p edgefirst-tensor cuda -- --test-threads=1
```

### On-target validation

On-target tests require a CUDA-capable device with `libcudart.so` present.
Validated platforms: Jetson Orin-nano (O5/O6/O8) on L4T R36.4 / CUDA 12.6 /
TRT 10.3; desktop RTX 3090 on CUDA 12.6.

Cross-compile and deploy:

```bash
cargo-zigbuild test --target aarch64-unknown-linux-gnu --release --no-run \
  -p edgefirst-tensor

scp target/aarch64-unknown-linux-gnu/release/deps/edgefirst_tensor-* \
  jetson-orin-nano:/tmp/hal-tests/
ssh jetson-orin-nano '/tmp/hal-tests/edgefirst_tensor-<hash> --test-threads=1'
```

The on-target suite includes an end-to-end test that:
1. Creates a float PBO tensor via `ImageProcessor::create_image()`.
2. Runs `convert()` on a synthetic frame.
3. Calls `cuda_map()` and asserts `device_ptr()` is non-null.
4. Validates numeric max-error against a host-side reference: observed
   max\_err 0.00024 on Orin-nano at F16 precision.
5. Drops the `CudaMap` guard and asserts the PBO is usable by a subsequent
   `convert()` call (aliasing rule enforcement).

### covguard (imx8mp coverage flush)

On the imx8mp lane the Vivante EGL driver may call `abort()` during shutdown.
The `edgefirst_tensor::covguard` module — compiled only under
`-Cinstrument-coverage` — installs a `SIGABRT` handler that flushes the LLVM
profile to disk before re-raising the signal. This is transparent to normal
builds; contributors do not need to do anything special to benefit from it.

See [`TESTING.md § CUDA tensor mapping`](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#cuda-tensor-mapping)
for the cross-workspace view, including the fallback-contract test and the
CI matrix notes.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Image-side PBO testing: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- C API DMA tests: [../capi/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/TESTING.md)
