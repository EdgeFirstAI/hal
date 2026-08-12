# hal_v4l2_gl

HAL V4L2 JPEG (e.g. mxc-jpeg) when available + HAL OpenGL letterbox.
Primarily for i.MX 95; on boards without a JPEG M2M node the codec falls
back to software decode automatically.

```bash
cargo run --release -p hal_v4l2_gl -- --limit 50 --board imx95-frdm \
  --csv ../../results/imx95-frdm/hal_v4l2_gl.csv
```
