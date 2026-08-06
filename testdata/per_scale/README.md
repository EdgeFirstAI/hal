# Per-scale decoder fixtures

Schema fragments for plan-time tests. These carry no tensor data — they exist so
tests can build a `PerScalePlan` from a synthetic schema without running any
model:

- `synthetic_yolov8n_schema.json` — 3 FPN levels (8/16/32), DFL reg_max=16, NC=80, NM=32. Box children: `[1, h, w, 64]` (= 4 × reg_max).
- `synthetic_yolo26n_schema.json` — 3 FPN levels (8/16/32), LTRB (4-channel boxes), NC=80, NM=32. Box children: `[1, h, w, 4]`.
- `synthetic_flat_schema.json` — non-per-scale schema (logical outputs have no children).

For the real-model fixtures that *do* carry tensor data, see
[`testdata/decoder/`](../decoder/README.md) — those are committed via git-lfs and
are what the parity tests actually decode.

## Real-model TFLite models (not committed)

Two TFLite model files are referenced by parity tests but are **not committed**
(large binaries):

- `yolov8n_seg_per_scale_int8.tflite` — yolov8n segmentation, per-scale DFL
  encoding, int8 quantized. Trained on COCO128.
- `yolo26n_seg_per_scale_int8.tflite` — yolo26n segmentation, per-scale LTRB
  encoding, int8 quantized. Trained on COCO128.

Both come from the EdgeFirst tflite-converter with `quantization_split` enabled.
Contact the validator team for the current artifacts.

When present, `crates/decoder/tests/per_scale_parity.rs` runs them through the
Python reference and the HAL. Two tests consume them
(`parity_yolov8n_seg_per_scale_int8`, `parity_yolo26n_seg_per_scale_int8`); both
carry `#[ignore]` because they are Python-only smoke checks that do not exercise
the HAL decode path, so they need `--include-ignored` even when the models are in
place. Everything else in that file skips cleanly when the models are absent.
