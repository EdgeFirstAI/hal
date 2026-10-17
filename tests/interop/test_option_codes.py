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


def _enumerators(header: Path, prefix: str) -> dict[str, int]:
    if not header.exists():
        pytest.skip(f"{header} is not in this checkout")
    pattern = re.compile(rf"^\s*{prefix}_([A-Z0-9_]+) = (\d+),", re.MULTILINE)
    found = {name: int(code) for name, code in pattern.findall(header.read_text())}
    assert found, f"{header.name} declares no {prefix}_* enumerators"
    return found


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


# (Python enum, header, C prefix, Python member -> C suffix overrides,
#  C suffixes Python deliberately does not expose)
CASES = [
    (img.Rotation, IMAGE_H, "EF_ROTATION", {"Rotate0": "NONE"}, set()),
    (img.Flip, IMAGE_H, "EF_FLIP", {"NoFlip": "NONE"}, set()),
    (img.ColorMode, IMAGE_H, "EF_COLOR_MODE", {}, set()),
    (img.Fit, IMAGE_H, "EF_FIT", {}, set()),
    (dec.Nms, DECODER_H, "EF_NMS", {}, {"OFF"}),
    (dec.DecoderType, DECODER_H, "EF_DECODER_TYPE", {}, set()),
    (dec.DecoderVersion, DECODER_H, "EF_DECODER_VERSION", {}, set()),
    (dec.DimName, DECODER_H, "EF_DIM_NAME", {}, {"UNKNOWN"}),
    (dec.MatchMetric, DECODER_H, "EF_MATCH_METRIC", {}, set()),
    (dec.MergeMode, DECODER_H, "EF_MERGE_MODE", {}, set()),
]


def _members(enum) -> dict[str, object]:
    return {
        name: getattr(enum, name)
        for name in dir(enum)
        if not name.startswith("_") and isinstance(getattr(enum, name), enum)
    }


@pytest.mark.parametrize(
    "enum,header,prefix,overrides,c_only",
    CASES,
    ids=[case[0].__name__ for case in CASES],
)
def test_python_values_match_the_c_header(enum, header, prefix, overrides, c_only):
    c_codes = _enumerators(header, prefix)
    members = _members(enum)
    assert members, f"{enum.__name__} has no members"
    seen = set()
    for name, member in members.items():
        suffix = overrides.get(name, _snake(name))
        assert suffix in c_codes, f"{prefix}_{suffix} is not in {header.name}"
        assert int(member) == c_codes[suffix], (
            f"{enum.__name__}.{name} is {int(member)}, "
            f"{prefix}_{suffix} is {c_codes[suffix]}"
        )
        seen.add(suffix)
    assert set(c_codes) - seen == c_only, (
        f"{enum.__name__} does not cover {sorted(set(c_codes) - seen - c_only)}"
    )


def test_nms_off_is_none_not_a_member():
    assert _enumerators(DECODER_H, "EF_NMS")["OFF"] == 0
    assert 0 not in {int(m) for m in _members(dec.Nms).values()}
