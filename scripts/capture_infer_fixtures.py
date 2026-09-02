#!/usr/bin/env python3
# scripts/capture_infer_fixtures.py — thin CLI over the shared module
"""Capture model I/O signals from Ultralytics exports into JSON fixtures.

Usage: scripts/capture_infer_fixtures.py MODEL [MODEL ...]

Each MODEL is an exported `.onnx` or `.tflite` file, produced with your own
Ultralytics install (see TESTING.md). One fixture per model is written to
`crates/decoder/testdata/infer/<stem>.signals.json`, which is what the
`edgefirst-decoder` inference tests read.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "crates" / "decoder" / "testdata" / "infer"

# Ultralytics stamps the absolute path of the dataset YAML from whichever
# machine trained the released weights into the `description` metadata field
# (e.g. "/usr/src/ultralytics/.../coco.yaml" from their Docker image,
# "/home/<user>/codes/.../coco.yaml" from a developer checkout). Nothing in
# the inference path reads `description`, so those paths are pure noise that
# would otherwise be committed verbatim; reduce each to its basename so the
# fixtures stay portable and diff cleanly across recaptures.
_ABS_YAML_PATH = re.compile(r"(?:/[\w.\-]+)+/([\w\-]+\.yaml)")


def scrub(value):
    """Recursively strips absolute dataset paths out of captured metadata."""
    if isinstance(value, str):
        return _ABS_YAML_PATH.sub(r"\1", value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def drop_upstream_license(metadata):
    """Blanks the exporter's own `license` string out of captured metadata.

    Nothing in the inference path reads it, and this repository ships no
    upstream model code -- the decoder is an independent Rust
    implementation. Carrying another project's licence declaration in our
    test fixtures would only misstate what this code is.

    `author` and `docs` are deliberately kept: `infer_ultralytics_schema`
    reads them to refuse a model whose metadata names a different vendor.
    """
    out = {}
    for key, value in metadata.items():
        if key == "license":
            out[key] = ""
            continue
        # TFLite carries the whole metadata document as one JSON string.
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                doc = json.loads(value)
            except json.JSONDecodeError:
                out[key] = value
                continue
            if isinstance(doc, dict) and "license" in doc:
                doc["license"] = ""
                out[key] = json.dumps(doc)
                continue
        out[key] = value
    return out


def main(argv):
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    # The signal extractor lives with the tests that consume it, not on the
    # import path; add it here rather than at module scope so the import
    # stays at the top of the file (ruff E402) and the script works from any
    # working directory.
    sys.path.insert(0, str(REPO_ROOT / "tests" / "decoder"))
    from ultralytics_signals import signals_for

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in map(Path, argv):
        out = FIXTURE_DIR / (path.stem + ".signals.json")
        signals = scrub(signals_for(path))
        signals["metadata"] = drop_upstream_license(signals["metadata"])
        out.write_text(json.dumps(signals, indent=1))
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
