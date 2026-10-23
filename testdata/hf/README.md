# Hugging Face config fixtures

Test fixtures in the Hugging Face `config.json` and `preprocessor_config.json` formats, used by the `edgefirst-decoder` model configuration tests (`crates/decoder/src/model_config/`), the Python tests and the C link test. They are written by `generate.py` from the `transformers` library's own config and image processor defaults, with 150 synthetic labels (`class_0` … `class_149`). They describe no published checkpoint and hold no model weights. Like the rest of this repository, they are licensed under Apache-2.0.

Regenerate them from the repository root with `transformers` 4.x (which needs only numpy and Pillow) in the venv:

```sh
source venv/bin/activate
pip install "transformers<5" pillow
python testdata/hf/generate.py
```

| Directory | Shape it covers |
|---|---|
| `segformer-512` | SegFormer: 512×512, `do_reduce_labels`, ImageNet mean/std, bilinear `resample` |
| `upernet-512` | UperNet with a ConvNeXt backbone, read by `SegformerImageProcessor`, no label reduction |
| `dpt-480` | DPT: 480×480, 0.5 mean/std, bicubic `resample` |

Older export forms (an integer `size`, `reduce_labels`, `feature_extractor_type`, a config with no `transformers_version`) are covered by inline JSON in `crates/decoder/src/model_config/hf.rs`.
