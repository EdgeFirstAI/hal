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

# ADE20K checkpoints carry untrimmed names such as "bed ", and older exports
# use the feature-extractor keys, which current transformers no longer writes.
legacy = OUT / "segformer-legacy-512"
untrimmed = {**ID2LABEL, 7: "bed ", 33: "chest of drawers"}
save(
    "segformer-legacy-512",
    SegformerConfig(
        id2label=untrimmed,
        label2id={v: k for k, v in untrimmed.items()},
        architectures=["SegformerForSemanticSegmentation"],
    ),
    SegformerImageProcessor(),
)
(legacy / "preprocessor_config.json").write_text(
    json.dumps(
        {
            "do_normalize": True,
            "do_resize": True,
            "feature_extractor_type": "SegformerFeatureExtractor",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "reduce_labels": True,
            "resample": 2,
            "size": 512,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
