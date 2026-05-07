# Per-scale decoder fixtures

Schema fragments for plan-time tests:
- `synthetic_yolov8n_schema.json` — 3 FPN levels (8/16/32), DFL reg_max=16, NC=80, NM=32. Box children: `[1, h, w, 64]` (= 4 × reg_max).
- `synthetic_yolo26n_schema.json` — 3 FPN levels (8/16/32), LTRB (4-channel boxes), NC=80, NM=32. Box children: `[1, h, w, 4]`.
- `synthetic_flat_schema.json` — non-per-scale schema (logical outputs have no children).

Real-model TFLite fixtures (`yolov8n_seg_per_scale_int8.tflite`,
`yolo26n_seg_per_scale_int8.tflite`) are added separately by the
validator team and are not committed to this repository (large binary).
The parity test skips when they're not present.

## Real-model fixtures (not committed)

The following TFLite model files are referenced by parity tests but are
**not committed to this repo** (large binaries):

- `yolov8n_seg_per_scale_int8.tflite` — yolov8n segmentation, per-scale
  DFL encoding, int8 quantized. Trained on COCO128.
- `yolo26n_seg_per_scale_int8.tflite` — yolo26n segmentation, per-scale
  LTRB encoding, int8 quantized. Trained on COCO128.

These are produced by the EdgeFirst tflite-converter with
`quantization_split` enabled. Contact the validator team for the
current artifacts.

When present, the parity tests in `crates/decoder/tests/per_scale_parity.rs`
will run them through the Python reference and (Phase 2+) the HAL.
Tests gracefully skip when files are absent.
