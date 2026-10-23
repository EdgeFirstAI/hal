# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Hugging Face config fixtures from the transformers defaults.

Run from the repository root with transformers installed in the venv:

    python testdata/hf/generate.py
"""

import json
import shutil
from pathlib import Path

from transformers import (
    ConvNextConfig,
    DPTConfig,
    DPTImageProcessor,
    SegformerConfig,
    SegformerImageProcessor,
    UperNetConfig,
)

OUT = Path(__file__).resolve().parent
NUM_LABELS = 150
ID2LABEL = {i: f"class_{i}" for i in range(NUM_LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def labelled(config_cls, architecture, **kwargs):
    return config_cls(
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        architectures=[architecture],
        **kwargs,
    )


def save(name, config, processor):
    out = OUT / name
    shutil.rmtree(out, ignore_errors=True)
    config.save_pretrained(out)
    processor.save_pretrained(out)
    for path in out.glob("*.json"):
        data = json.loads(path.read_text())
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


save(
    "segformer-512",
    labelled(SegformerConfig, architecture="SegformerForSemanticSegmentation"),
    SegformerImageProcessor(size={"height": 512, "width": 512}, do_reduce_labels=True),
)
save(
    "upernet-512",
    labelled(
        UperNetConfig,
        architecture="UperNetForSemanticSegmentation",
        backbone_config=ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"]),
    ),
    SegformerImageProcessor(size={"height": 512, "width": 512}, rescale_factor=1 / 255),
)
save(
    "dpt-480",
    labelled(DPTConfig, architecture="DPTForSemanticSegmentation"),
    DPTImageProcessor(size={"height": 480, "width": 480}, do_reduce_labels=True),
)
