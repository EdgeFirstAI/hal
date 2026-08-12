# hal_gl

HAL software JPEG → native chroma + HAL OpenGL letterbox.

```bash
cargo run --release -p hal_gl -- --limit 50 --board imx8mp-frdm \
  --csv ../../results/imx8mp-frdm/hal_gl.csv
```

On hosts without DMA-BUF YUV import (PBO-only), Nv24 frames fall back to CPU convert.
