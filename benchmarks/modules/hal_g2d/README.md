# hal_g2d

HAL software JPEG → native chroma + HAL G2D letterbox (i.MX / galcore).

```bash
cargo run --release -p hal_g2d -- --limit 50 --board imx8mp-frdm \
  --csv ../../results/imx8mp-frdm/hal_g2d.csv
```

Skips with `ForcedBackendUnavailable` when `libg2d` / `/dev/galcore` is absent.
