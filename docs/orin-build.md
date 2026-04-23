# Building for the NVIDIA Orin Nano

The Orin Nano uses a Cortex-A78AE core that supports the ARMv8.2-FP16
extension (FEAT_FP16). The HAL's native f16 mask kernels
(`fused_dot_sigmoid_f16_slice`) compile down to scalar `fcvt` widenings
on this target, avoiding the soft-float `__extendhfsf2` helper.

## Release build (recommended)

```bash
RUSTFLAGS="-C target-cpu=cortex-a78ae" cargo build --release \
    --target aarch64-unknown-linux-gnu --workspace --all-features
```

`target-cpu=cortex-a78ae` is a single flag that implies `+fp16`,
`+dotprod`, `+lse`, `+crypto`, and `+rcpc`. It also pins the LLVM code
model to the core's scheduler, so loop vectorizers can reason about the
pipeline correctly.

## Verifying native f16 codegen

After a release build, run the codegen audit script:

```bash
RUSTFLAGS="-C target-cpu=cortex-a78ae" \
    scripts/audit_f16_codegen.sh aarch64-unknown-linux-gnu
```

A successful run shows `fcvt` / `fcvtl` (single- and multi-lane f16→f32
widenings) in the `fused_dot_sigmoid_f16_slice` disassembly and does NOT
show `__extendhfsf2` (the soft-float helper).

## imx8mp (Cortex-A53) — no native f16

The imx8mp uses Cortex-A53 which predates FEAT_FP16. Building with
`+fp16` will emit native intrinsics that SIGILL at runtime.

On imx8mp, DO NOT set `target-cpu=cortex-a78ae` or
`target-feature=+fp16`. The workspace's `.cargo/config.toml` deliberately
omits a blanket aarch64 `+fp16` entry to keep imx8mp safe. The
non-FP16 fallback path widens f16 protos per-load via the soft-float
helper — correctness-preserving at the cost of scalar throughput.

## On-device smoke test

1. `scp` the release binary plus a YOLO-seg model (e.g.
   `yolo26x-seg-t-2592-fp16.engine`) and a test image to the device.
2. Run the HAL validator; capture mask PNG + per-frame latency.
3. Compare output against the host-recorded reference to catch numeric
   drift (bulk dequant paths should match host CPU within IoU ≥ 0.999).
