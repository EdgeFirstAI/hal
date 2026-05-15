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
│   ├── tensor_dyn.rs    # TensorDyn metadata, multi-plane, format/stride checks
│   └── format.rs        # PixelFormat + DType compatibility
└── benches/
    └── tensor_benchmark.rs   # Allocation + map/unmap throughput
```

There is no `tests/` directory; per-backend coverage lives in the source
files alongside the implementation.

## Running Tests

```bash
# All tensor tests on the host
cargo test -p edgefirst-tensor

# Heap-only (skips DMA tests when DMA-heap permissions are missing)
EDGEFIRST_TENSOR_FORCE_MEM=1 cargo test -p edgefirst-tensor

# Doc-tests only
cargo test -p edgefirst-tensor --doc

# With ndarray feature exercised explicitly
cargo test -p edgefirst-tensor --features ndarray
```

## Special Requirements

- **DMA tests** require Linux **and** read+write access to a DMA-heap device
  (typically `/dev/dma_heap/system` or `/dev/dma_heap/cma`). On a workstation
  this usually means being in the `video` group. If permissions are missing,
  DMA tests are skipped at runtime and the suite falls back to heap-only
  coverage; nothing fails outright. Set `EDGEFIRST_TENSOR_FORCE_MEM=1` to
  skip even the probe.
- **PBO tests** are gated by the `opengl` feature (which is *not* a feature
  of this crate but of `edgefirst-image`). PBO behavior is exercised
  end-to-end by the image crate's tests, not here.
- **No LFS testdata.** All shapes and inputs are synthesized in-test. The
  `tensor_benchmark` is the only consumer of `edgefirst_bench::testdata` and
  uses small synthetic blobs.
- **No feature flags required by default**; `ndarray` is on by default and
  is required for `view()` / `view_mut()` doc-tests.

## Benchmarks

```bash
# Native-host benchmark
cargo bench -p edgefirst-tensor

# Cross-compile + deploy to a target
make bench-arm
```

| Benchmark | Source | What it measures |
|-----------|--------|------------------|
| `tensor_benchmark` | [`benches/tensor_benchmark.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/benches/tensor_benchmark.rs) | Allocation cost across DMA/SHM/Mem backends; map/unmap throughput; clone_fd cost |

## Coverage Notes

- The crate participates in workspace `cargo llvm-cov`; lcov output
  appears under `crates/tensor/`.
- `cfg(target_os = "linux")`-gated code (`dma.rs`, `dmabuf.rs`) is only
  exercised on the Linux host runners and on the i.MX 8M Plus hardware
  runner. macOS and Windows CI legs cover the heap and SHM paths.
- The `procfs` dev-dependency on Linux is used to assert that mapped DMA
  buffers actually appear in `/proc/self/maps` — guards against silent
  fallback-to-heap regressions.

## Cross-References

- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Validating optimizations: [TESTING.md#validating-optimizations](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#validating-optimizations)
- Image-side PBO testing: [../image/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/TESTING.md)
- C API DMA tests: [../capi/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/TESTING.md)
