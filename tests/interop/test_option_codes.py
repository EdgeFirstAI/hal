# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""The Python option enums use the C header's numbering.

Each image and decoder option vocabulary is declared once in Rust. The C
enumerators and the Python enum values are both pinned to that declaration
at compile time; this test closes the loop by comparing the two shipped
surfaces directly, for every member.
"""

from __future__ import annotations

import re
from pathlib import Path

import edgefirst.decoder as dec
import edgefirst.image as img
import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGE_H = ROOT / "crates/image-capi/include/edgefirst/image.h"
DECODER_H = ROOT / "crates/decoder-capi/include/edgefirst/decoder.h"
DETECT_H = ROOT / "crates/decoder-abi/include/edgefirst/detect.h"
HEADERS = (IMAGE_H, DECODER_H, DETECT_H)

# The codes of vocabularies new in this release, as their `ef_vocabulary!`
# declarations document them (suffixes are the declared wire constants). A
# checkout whose headers do not declare one yet is checked against this
# table; once a header does, the two must agree
# (`test_documented_codes_match_the_c_headers`).
DOCUMENTED = {
    "EF_INTERPOLATION": {"BILINEAR": 0, "NEAREST": 1, "BICUBIC": 2, "AREA": 3},
    "EF_PAD_SPACE": {"PIXEL": 0, "NORMALIZED": 1},
    "EF_CONFIG_SOURCE": {
        "EMBEDDED_EDGEFIRST": 0,
        "SIDECAR_EDGEFIRST": 1,
        "EMBEDDED_HF": 2,
        "SIDECAR_HF": 3,
        "ULTRALYTICS_METADATA": 4,
        "HEURISTIC": 5,
        "DEFAULT": 6,
        "OVERRIDE": 7,
    },
    "EF_MODEL_FAMILY": {
        "EDGEFIRST": 0,
        "ULTRALYTICS": 1,
        "HUGGINGFACE": 2,
        "UNKNOWN": 3,
    },
    "EF_MODEL_TASK": {"DETECT": 0, "SEGMENT": 1, "SEMANTIC": 2},
    "EF_CONFIG_FIELD": {
        "NORMALIZATION": 0,
        "RESIZE": 1,
        "PLACEMENT": 2,
        "PAD": 3,
        "INTERPOLATION": 4,
        "TASK": 5,
        "DECODE_MODE": 6,
        "LABELS": 7,
        "SIZE": 8,
    },
}


def _declared(prefix: str) -> dict[str, int]:
    """`prefix_*` enumerators from whichever shipped header declares them."""
    pattern = re.compile(rf"^\s*{prefix}_([A-Z0-9_]+) = (\d+),", re.MULTILINE)
    for header in HEADERS:
        if header.exists():
            found = dict(pattern.findall(header.read_text()))
            if found:
                return {name: int(code) for name, code in found.items()}
    return {}


def _enumerators(prefix: str) -> dict[str, int]:
    found = _declared(prefix)
    if found:
        return found
    if prefix in DOCUMENTED:
        return DOCUMENTED[prefix]
    if not any(h.exists() for h in HEADERS):
        pytest.skip("the C headers are not in this checkout")
    pytest.fail(f"no header declares {prefix}_* enumerators")


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def _squash(name: str) -> str:
    return name.replace("_", "")


# (Python enum, C prefix, Python member -> C suffix overrides,
#  C suffixes Python deliberately does not expose)
CASES = [
    (img.Rotation, "EF_ROTATION", {"Rotate0": "NONE"}, set()),
    (img.Flip, "EF_FLIP", {"NoFlip": "NONE"}, set()),
    (img.ColorMode, "EF_COLOR_MODE", {}, set()),
    (img.Fit, "EF_FIT", {}, set()),
    (img.LetterboxPlacement, "EF_LETTERBOX_PLACEMENT", {}, set()),
    (img.NormalizationPreset, "EF_NORMALIZATION", {}, set()),
    (img.Interpolation, "EF_INTERPOLATION", {}, set()),
    (img.PadSpace, "EF_PAD_SPACE", {}, set()),
    (dec.Nms, "EF_NMS", {}, {"OFF"}),
    (dec.DecoderType, "EF_DECODER_TYPE", {}, set()),
    (dec.DecoderVersion, "EF_DECODER_VERSION", {}, set()),
    (dec.DimName, "EF_DIM_NAME", {}, {"UNKNOWN"}),
    (dec.MatchMetric, "EF_MATCH_METRIC", {}, set()),
    (dec.MergeMode, "EF_MERGE_MODE", {}, set()),
    (dec.DecodeMode, "EF_DECODE_MODE", {}, set()),
    (dec.ConfigSource, "EF_CONFIG_SOURCE", {}, set()),
    (dec.ModelFamily, "EF_MODEL_FAMILY", {}, set()),
    (dec.ModelTask, "EF_MODEL_TASK", {}, set()),
    (dec.ConfigField, "EF_CONFIG_FIELD", {}, set()),
]


def _members(enum) -> dict[str, object]:
    return {
        name: getattr(enum, name)
        for name in dir(enum)
        if not name.startswith("_") and isinstance(getattr(enum, name), enum)
    }


@pytest.mark.parametrize(
    "enum,prefix,overrides,c_only",
    CASES,
    ids=[case[0].__name__ for case in CASES],
)
def test_python_values_match_the_c_header(enum, prefix, overrides, c_only):
    c_codes = _enumerators(prefix)
    # Word breaks in a C suffix are a spelling choice (EDGEFIRST vs
    # EDGE_FIRST); the code each name carries is the contract.
    by_squashed = {_squash(k): k for k in c_codes}
    members = _members(enum)
    assert members, f"{enum.__name__} has no members"
    seen = set()
    for name, member in members.items():
        suffix = overrides.get(name, _snake(name))
        suffix = suffix if suffix in c_codes else by_squashed.get(_squash(suffix))
        assert suffix in c_codes, f"{prefix}_{_snake(name)} is not declared"
        assert int(member) == c_codes[suffix], (
            f"{enum.__name__}.{name} is {int(member)}, "
            f"{prefix}_{suffix} is {c_codes[suffix]}"
        )
        seen.add(suffix)
    assert set(c_codes) - seen == c_only, (
        f"{enum.__name__} does not cover {sorted(set(c_codes) - seen - c_only)}"
    )


def test_nms_off_is_none_not_a_member():
    assert _enumerators("EF_NMS")["OFF"] == 0
    assert 0 not in {int(m) for m in _members(dec.Nms).values()}


@pytest.mark.parametrize("prefix", sorted(DOCUMENTED))
def test_documented_codes_match_the_c_headers(prefix):
    declared = _declared(prefix)
    if not declared:
        pytest.skip(
            f"no header declares {prefix}_* yet; Python is checked against the table"
        )
    squash = lambda codes: {_squash(k): v for k, v in codes.items()}
    assert squash(declared) == squash(DOCUMENTED[prefix])
