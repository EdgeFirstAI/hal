# HAL Decoder Reference Fixtures

This directory holds `.safetensors` reference fixtures for the
`edgefirst-decoder` crate's parity tests. Each fixture bundles, for
one TFLite model + image pair:

- the model's raw quantized outputs (one tensor per FPN-level child plus protos),
- per-stage NumPy reference intermediates (`dequant`, `ltrb`, `xywh`, `activated`),
- the post-NMS reference detections (`boxes_xyxy`, `scores`, `classes`, optional `masks`),
- the patched `edgefirst.json` schema (with `dshape` synthesized on logical parents),
- per-tensor quantization parameters,
- the NMS hyperparameters used at fixture-generation time,
- a Markdown narrative under `__metadata__["documentation_md"]` mapping every key to a HAL kernel.

## Why "physical output decomposition"?

The EdgeFirst tflite-converter's `quantization_split` mode strips the
quantization-sensitive tail layers (DFL, sigmoid, dist2bbox, per-scale
concat) from a YOLO model and exposes each FPN level's per-channel
outputs as separate physical tensors with independent quantization
parameters. The runtime re-implements those tail layers in higher
precision on the CPU. In HAL that re-implementation is
`crates/decoder/src/per_scale/`. These fixtures are the oracle for
that re-implementation.

The HAL `edgefirst-decoder` crate has **no TFLite/ONNX/PyTorch runtime
dependency** — it consumes already-evaluated raw tensors plus the
`edgefirst.json` schema and produces final detections. These fixtures
let HAL unit tests exercise that path without pulling in any inference
runtime.

## Currently committed fixtures

| Fixture | Source TFLite | Encoding | Source image | Reference detections |
|---------|---------------|----------|--------------|----------------------|
| `yolov8n-seg.safetensors` | `/tmp/e2e-t2681/yolov8n-seg-t-2681_per-scale.tflite` | DFL | `testdata/zidane.jpg` | 249 |
| `yolo11n-seg.safetensors` | `/tmp/e2e-t2685/yolo11n-seg-t-2685_per-scale.tflite` | DFL | `testdata/zidane.jpg` | 101 |
| `yolo26n-seg.safetensors` | `/tmp/e2e-t2686/yolo26n-seg-t-2686_per-scale.tflite` | LTRB | `testdata/zidane.jpg` |  48 |

## Pulling fixtures

The fixtures are tracked under git-lfs. After cloning, run:

```bash
git lfs pull
```

Without LFS pull the parity tests skip cleanly with an "fixture not present" message.

## Reading a fixture's documentation without code

```bash
source venv/bin/activate
python -c "
import safetensors
with safetensors.safe_open('testdata/decoder/yolov8n-seg.safetensors',
                           framework='numpy') as f:
    print(f.metadata()['documentation_md'])
"
```

This prints the Markdown narrative the generator embedded at
fixture-creation time — the fixture-key → HAL-kernel table, the NMS
configuration, and the consumer test names.

## Regenerating

```bash
source venv/bin/activate
pip install 'numpy>=1.26,<2' 'safetensors>=0.4' 'pillow>=10' 'tflite_runtime>=2.14'

python scripts/decoder_generate_fixture.py \
    <model.tflite> <image.jpg> \
    --output testdata/decoder/<name>.safetensors \
    [--no-stages]               # save ~70% by omitting per-stage intermediates
    [--score-threshold 0.001]
    [--iou-threshold 0.7]
    [--expected-count-min 10]
```

NumPy is pinned to `<2` because the `tflite_runtime 2.14` wheel was
built against the NumPy 1.x ABI. Once a NumPy-2-compatible
`tflite_runtime` is published the constraint can drop.

Regeneration is bit-stable on the same dev host with the same
numpy/safetensors versions (modulo the timestamp metadata key).

## Format

See `.claude/plans/2026-04-29-per-scale-fixture-framework-design.md`
section 4 for the canonical format spec, or read any fixture's
embedded `documentation_md`.

## Note on `testdata/per_scale/`

The synthetic schema fragments at
`testdata/per_scale/synthetic_*_schema.json` are unrelated to this
framework. They drive plan-time HAL tests that construct
`PerScalePlan` objects from synthetic schemas with no model inference
involved. They stay where they are.

## Tests that consume these fixtures

| Test | Fixture | What it checks |
|------|---------|----------------|
| `yolov8n_seg_per_scale_smoke_detection_count` | yolov8n-seg | HAL produces ≥ `expected_count_min` detections |
| `yolov8n_seg_per_scale_end_to_end_parity` | yolov8n-seg | HAL post-NMS detections match the reference |
| `yolov8n_seg_per_scale_pre_nms_parity` | yolov8n-seg | HAL pre-NMS xywh per anchor matches the reference |
| `yolo11n_seg_per_scale_*` | yolo11n-seg | (same triplet) |
| `yolo26n_seg_per_scale_*` | yolo26n-seg | (same triplet, LTRB encoding) |

The fixture-backed parity tests now run as part of the standard
suite — the post-decode NMS bridge bug that originally forced them
to be `#[ignore]`'d was fixed in the per-scale subsystem rollout.
Two `#[ignore]`-gated tests remain: external Python-only smoke
fixtures that don't exercise the HAL decode path. Everything else
runs by default with:

```bash
cargo test -p edgefirst-decoder --test per_scale_parity
```

To include the Python-only smoke tests as well:

```bash
cargo test -p edgefirst-decoder --test per_scale_parity --include-ignored
```
